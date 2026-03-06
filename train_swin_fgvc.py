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
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import time

from meft import MeftConfig, MeftTrainer
import meft

from get_rank.swin import get_swin_rank, get_swin_rank_ratio, get_swin_rank_binary_search_energy_ratio

from fgvc_datasets_setup.loader import _DATASET_NUM_LABELS

from fgvc_datasets_setup import loader as data_loader

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    parser.add_argument('--model_name', type=str, default='swin-base', help='预训练模型名称或路径')
    
    # 训练模式
    parser.add_argument('--vanilla_train', action='store_true', help='使用不压缩的普通lora微调')
    parser.add_argument('--rank_ratio', type=float, default=0.125, help='compress rank比例')
    parser.add_argument('--dynamic_rank', action='store_true', help='是否使用动态rank')
    parser.add_argument('--energy_ratio', type=float, default=0.5, help='动态rank时的能量保留比例')
    parser.add_argument('--energy_search', action='store_true', help='是否使用二分搜索方式确定energy_ratio')
    
    # 数据集
    parser.add_argument('--dataset_name', type=str, default='CUB', help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./datasets/fgvc', help='数据集存放根目录')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮数')
    parser.add_argument('--per_device_train_batch_size', type=int, default=512, help='每次批次大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减')
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


num_labels = _DATASET_NUM_LABELS[args.dataset_name]

# 加载模型
if args.model_name == 'swin-tiny':
    pretrained_name = "microsoft/swin-tiny-patch4-window7-224"
elif args.model_name == 'swin-small':
    pretrained_name = "microsoft/swin-small-patch4-window7-224"
elif args.model_name == 'swin-base':
    pretrained_name = "microsoft/swin-base-patch4-window7-224"
elif args.model_name == 'swin-large':
    pretrained_name = "microsoft/swin-large-patch4-window7-224"
else:
    raise ValueError(f"未知的模型名称: {args.model_name}")

processor = AutoFeatureExtractor.from_pretrained(pretrained_name)

model = SwinForImageClassification.from_pretrained(
    pretrained_name,
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
def get_datasets(args, processor):
    print("Loading training data...")
    train_loader = data_loader.construct_train_dataset(args, processor)
    print("Loading validation data...")
    val_loader = data_loader.construct_val_dataset(args, processor)
    print("Loading test data...")
    test_loader = data_loader.construct_test_dataset(args, processor)
    return train_loader,  val_loader, test_loader


train_dataset, val_dataset, test_dataset = get_datasets(args, processor)


if args.dynamic_rank:
    if args.energy_search:
        activations, rank_dict = get_swin_rank_binary_search_energy_ratio(model, val_dataset, batch_size=args.per_device_train_batch_size, patch_locations=2, rank_ratio=args.rank_ratio)
        del activations
    else:
        # _, rank_dict = get_vit_rank(model, val_dataset, batch_size=args.per_device_train_batch_size, patch_locations=2)
        activations, rank_dict = get_swin_rank_ratio(model, val_dataset, batch_size=args.per_device_train_batch_size, patch_locations=2, base_ratio=args.rank_ratio, energy_ratio=args.energy_ratio)
        # activations, rank_dict = get_vit_rank_ratio_gentle(model, val_dataset, batch_size=args.per_device_train_batch_size, patch_locations=2, base_ratio=args.rank_ratio, energy_ratio=args.energy_ratio)
        del activations
        print(rank_dict)

        num_layers = len(rank_dict)
        hidden_size = model.vit.config.hidden_size
        total_rank = 0
        for layer_name, io_dict in rank_dict.items():
            layer_rank = 0
            for io_type, rank in io_dict.items():
                layer_rank += rank
            total_rank += layer_rank


breakpoint()


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
            logging_steps = 1,
            report_to = ["wandb"],
            run_name=args.wandb_run_name,
            remove_unused_columns = False,
            label_names = ["labels"],
            # eval_strategy = "steps",
            # eval_steps = len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps),
            eval_strategy = "epoch",
            output_dir = args.output_dir,
        ),
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
            logging_steps = 1,
            report_to = ["wandb"],
            run_name=args.wandb_run_name,
            remove_unused_columns = False,
            label_names = ["labels"],
            # eval_strategy = "steps",
            # eval_steps = len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps),
            eval_strategy = "epoch",
            output_dir = args.output_dir,
        ),
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"测试集top-1准确率: {test_results['eval_accuracy']:.4f}")

wandb.summary["test_accuracy"] = test_results["eval_accuracy"]
