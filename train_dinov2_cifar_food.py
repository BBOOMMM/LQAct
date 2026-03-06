import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import huggingface_hub
print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["MKL_SERVICE_FORCE_INTEL"]="1"
# os.environ["OMP_NUM_THREADS"] = "8"  # 控制MKL使用的线程数
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["BLAS"] = "MKL"
# os.environ["USE_CUSOLVER"] = "1"

import numpy as np
import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoFeatureExtractor, SwinForImageClassification,
    ViTImageProcessor, ViTForImageClassification,
    Trainer, TrainingArguments, EvalPrediction,
    TrainerCallback,
    AutoImageProcessor, Dinov2ForImageClassification,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import time

from meft import MeftConfig, MeftTrainer
import meft

from get_rank.dinov2 import get_dinov2_rank, get_dinov2_rank_ratio, get_dinov2_rank_binary_search_energy_ratio, get_dinov2_rank_ratio_gentle

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    parser.add_argument('--model_name', type=str, default='dinov2-base', help='预训练模型名称或路径')
    
    # 训练模式
    parser.add_argument('--vanilla_train', action='store_true', help='使用不压缩的普通lora微调')
    parser.add_argument('--rank_ratio', type=float, default=0.125, help='compress rank比例')
    parser.add_argument('--dynamic_rank', action='store_true', help='是否使用动态rank')
    parser.add_argument('--energy_ratio', type=float, default=0.5, help='动态rank时的能量保留比例')
    parser.add_argument('--energy_search', action='store_true', help='是否使用二分搜索方式确定energy_ratio')
    parser.add_argument('--gentle', action='store_true', help='是否使用gentle动态rank调整方式')
    
    # 数据集
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='数据集名称')
    parser.add_argument('--val_set_size', type=int, default=10000, help='验证集大小')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮数')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1024, help='每次批次大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0, help='权重衰减')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='模型与LoRA参数保存路径')
    parser.add_argument('--wandb_project_name', type=str, default='wandb', help='wandb项目名称')
    parser.add_argument('--wandb_run_name', type=str, default='wandb', help='wandb运行名称')
    
    # lora参数
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout概率')

    args = parser.parse_args()
    return args
args = parse_args()
print("训练参数：", args)


import random
from transformers import set_seed as hf_set_seed
seed = args.seed
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
hf_set_seed(seed)  # transformers 内部用到的随机也统一
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import wandb
# 配置 wandb
# if args.vanilla_train:
#     run_name = f"{args.model_name}-lora"
# elif not args.dynamic_rank:
#     run_name = f"{args.model_name}-loract{args.rank_ratio}"
# else:
#     run_name = f"{args.model_name}-loract{args.rank_ratio}-dksearch"

wandb.init(
    project=args.wandb_project_name,
    name=args.wandb_run_name,
    config=vars(args),
)


class ThroughputCallback(TrainerCallback):
    """记录每个 epoch 的耗时和吞吐量（samples/s），同时打到终端和 wandb。"""
    def __init__(self, train_dataset_len):
        self.train_dataset_len = train_dataset_len
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is None:
            return
        epoch_time = time.time() - self.epoch_start_time
        # 简单按一个 epoch 走完整个 train_dataset 估算吞吐量
        samples_per_sec = self.train_dataset_len / epoch_time
        epoch_idx = int(state.epoch) if state.epoch is not None else -1
        print(f"[Epoch {epoch_idx}] time: {epoch_time:.2f}s, throughput: {samples_per_sec:.2f} samples/s")
        # 记录到 wandb
        wandb.log(
            {
                "epoch": epoch_idx,
                "epoch_time_s": epoch_time,
                "epoch_samples_per_second": samples_per_sec,
            },
            step=state.global_step,
        )


if args.dataset_name == 'cifar100':
    num_labels = 100
elif args.dataset_name == 'food101':
    num_labels = 101
else:
    raise ValueError(f"不支持的数据集名称: {args.dataset_name}")


# # 加载模型
# if args.model_name == 'vit-base':
#     model_name = "google/vit-base-patch16-224-in21k"
#     processor = ViTImageProcessor.from_pretrained(model_name)
#     model = ViTForImageClassification.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         num_labels=num_labels,
#         ignore_mismatched_sizes=True,
#         # device_map="auto",
#         device_map="cuda:0",
#     )
# elif args.model_name == 'vit-large':
#     model_name = "google/vit-large-patch16-224-in21k"
#     processor = ViTImageProcessor.from_pretrained(model_name)
#     model = ViTForImageClassification.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         num_labels=num_labels,
#         ignore_mismatched_sizes=True,
#         # device_map="auto",
#         device_map="cuda:0",
#     )
# elif args.model_name == 'vit-huge':
#     model_name = "google/vit-huge-patch14-224-in21k"
#     processor = ViTImageProcessor.from_pretrained(model_name)
#     model = ViTForImageClassification.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         num_labels=num_labels,
#         ignore_mismatched_sizes=True,
#         # device_map="auto",
#         device_map="cuda:0",
#     )


if 'dinov2' in args.model_name:
    if args.model_name == 'dinov2-base':
        model_id = "facebook/dinov2-base"
    elif args.model_name == 'dinov2-large':
        model_id = "facebook/dinov2-large"
    elif args.model_name == 'dinov2-giant':
        model_id = "facebook/dinov2-giant"
    
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Dinov2ForImageClassification.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        device_map="cuda:0",
    )
    

# 配置LoRA
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["query", "value"],
    bias="none",
    modules_to_save=["classifier"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 加载数据集
if args.dataset_name == 'cifar100':
    dataset = load_dataset("cifar100")
    train_split, eval_split = "train", "test"
elif args.dataset_name == 'food101':
    dataset = load_dataset("food101")
    train_split, eval_split = "train", "validation"
else:
    raise ValueError(f"不支持的数据集名称: {args.dataset_name}")

def transform(examples):
    if args.dataset_name == 'cifar100':
        images = [x.convert("RGB") for x in examples["img"]]
        labels = examples["fine_label"]
    elif args.dataset_name == 'food101':
        images = [x.convert("RGB") for x in examples["image"]]
        labels = examples["label"]

    inputs = processor(images=images, return_tensors="pt")
    return {
        "pixel_values": inputs.pixel_values,
        "labels": labels,
    }

dataset = dataset.with_transform(transform)


# 分割训练集和验证集
if args.dataset_name == "food101":
    test_data = dataset["validation"]
else:
    test_data = dataset['test']
train_val = dataset["train"].train_test_split(
    test_size=args.val_set_size, shuffle=True, seed=42
)
train_data = train_val['train']
val_data = train_val['test']


if args.dynamic_rank:
    if args.energy_search:
        activations, rank_dict = get_dinov2_rank_binary_search_energy_ratio(model, val_data, batch_size=args.per_device_train_batch_size, patch_locations=2, rank_ratio=args.rank_ratio)
        del activations
        print(rank_dict)

        num_layers = len(rank_dict)
        hidden_size = model.dinov2.config.hidden_size
        total_rank = 0
        for layer_name, io_dict in rank_dict.items():
            layer_rank = 0
            for io_type, rank in io_dict.items():
                layer_rank += rank
            total_rank += layer_rank
    else:
        # _, rank_dict = get_vit_rank(model, val_dataset, batch_size=args.per_device_train_batch_size, patch_locations=2)
        activations, rank_dict = get_dinov2_rank_ratio(model, val_data, batch_size=args.per_device_train_batch_size, patch_locations=2, base_ratio=args.rank_ratio, energy_ratio=args.energy_ratio)
        del activations
        # print(rank_dict)

        num_layers = len(rank_dict)
        hidden_size = model.dinov2.config.hidden_size
        total_rank = 0
        for layer_name, io_dict in rank_dict.items():
            layer_rank = 0
            for io_type, rank in io_dict.items():
                layer_rank += rank
            total_rank += layer_rank

    print(f"总rank: {total_rank}")
    print(f'1/16 总Rank :{hidden_size * num_layers / 16}')
    print(f'1/8 总Rank :{hidden_size * num_layers / 8}')



def compute_metrics(eval_pred: EvalPrediction):
    evaluation = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluation.compute(predictions=predictions, references=labels)

if args.vanilla_train:
        trainer = Trainer(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            per_device_eval_batch_size = 32,
            num_train_epochs = args.epochs,
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_torch",
            bf16 = True,
            bf16_full_eval = True,
            use_liger_kernel = True,
            logging_steps = 10,
            report_to = ["wandb"],
            run_name=args.wandb_run_name,
            remove_unused_columns = False,
            label_names = ["labels"],
            # eval_strategy = "steps",
            # eval_steps = len(dataset[train_split]) // (args.per_device_train_batch_size * args.gradient_accumulation_steps),
            eval_strategy = "epoch",
            output_dir = args.output_dir,
        ),
        data_collator=None,
        train_dataset=dataset[train_split],
        eval_dataset=dataset[eval_split],
        compute_metrics=compute_metrics,
        # callbacks=[ThroughputCallback(len(train_dataset))],
    )
else:
    trainer = MeftTrainer[Trainer](
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            per_device_eval_batch_size = args.per_device_train_batch_size,
            num_train_epochs = args.epochs,
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_torch",
            bf16 = True,
            bf16_full_eval = True,
            use_liger_kernel = True,
            logging_steps = 10,
            report_to = ["wandb"],
            run_name=args.wandb_run_name,
            remove_unused_columns = False,
            label_names = ["labels"],
            # eval_strategy = "steps",
            # eval_steps = len(dataset[train_split]) // (args.per_device_train_batch_size * args.gradient_accumulation_steps),
            eval_strategy = "epoch",
            output_dir = args.output_dir,
        ),
        data_collator=None,
        train_dataset=dataset[train_split],
        eval_dataset=dataset[eval_split],
        compute_metrics=compute_metrics,
        # callbacks=[ThroughputCallback(len(train_dataset))],
        meft_config=MeftConfig(
            patch_locations=(
                "norm",
                # "attn_in",
                # "attn_out",
                # "mlp_in",
                # "mlp_out",
                # "act_fn",
                "ckpt_attn",
                "ckpt_mlp",
                # "ckpt_layer",
            ),
            compress_kwargs={
                # "rank": 0.0625,
                # "rank": rank_dict,
                "rank": rank_dict if args.dynamic_rank else args.rank_ratio,
                # "niter": 1,
            },
            # compress_workers=2,
        ),
    )


# trainer.train()
# 记录总训练时间 & 峰值显存
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

train_start = time.time()
train_output = trainer.train()
if torch.cuda.is_available():
    torch.cuda.synchronize()
train_total_time = time.time() - train_start

print(f"[Train] total time: {train_total_time:.2f}s")

if torch.cuda.is_available():
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    print(f"[Train] peak GPU memory: {peak_mem_gb:.2f} GB")
    wandb.log(
        {
            "train_total_time_s": train_total_time,
            "train_peak_gpu_mem_gb": peak_mem_gb,
        },
        step=train_output.global_step if hasattr(train_output, "global_step") else None,
    )


# trainer.evaluate()
test_results = trainer.evaluate(eval_dataset=dataset[eval_split])
print(f"测试集top-1准确率: {test_results['eval_accuracy']:.4f}")

wandb.summary["test_accuracy"] = test_results["eval_accuracy"]
