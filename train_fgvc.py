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
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from meft import MeftConfig, MeftTrainer
import meft

from get_rank.vit import get_vit_rank, get_vit_rank_ratio

from fgvc_datasets_setup.loader import _DATASET_NUM_LABELS

from fgvc_datasets_setup import loader as data_loader

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=2026, help='随机种子')
    
    parser.add_argument('--model_name', type=str, default='vit-base', help='预训练模型名称或路径')
    
    parser.add_argument('--rank_ratio', type=float, default=0.125, help='compress rank比例')
    parser.add_argument('--dynamic_rank', action='store_true', help='是否使用动态rank')
    
    parser.add_argument('--dataset_name', type=str, default='CUB', help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./datasets/fgvc', help='数据集存放根目录')
    
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')

    

    args = parser.parse_args()
    return args
args = parse_args()
print("训练参数：", args)


import random
import numpy as np
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


num_labels = _DATASET_NUM_LABELS[args.dataset_name]

# 加载模型
if args.model_name == 'vit-base':
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        # device_map="auto",
        device_map="cuda:0",
    )
elif args.model_name == 'vit-large':
    model_name = "google/vit-large-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        # device_map="auto",
        device_map="cuda:0",
    )
elif args.model_name == 'vit-huge':
    model_name = "google/vit-huge-patch14-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        # device_map="auto",
        device_map="cuda:0",
    )
    

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
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
    # _, rank_dict = get_vit_rank(model, val_dataset, batch_size=args.batch_size, patch_locations=2)
    _, rank_dict = get_vit_rank_ratio(model, val_dataset, batch_size=args.batch_size, patch_locations=2, base_ratio=args.rank_ratio)


    num_layers = len(rank_dict)
    hidden_size = model.vit.config.hidden_size
    total_rank = 0
    for layer_name, io_dict in rank_dict.items():
        layer_rank = 0
        for io_type, rank in io_dict.items():
            layer_rank += rank
        total_rank += layer_rank

    print(f"总rank: {total_rank}")
    print(f'1/16 总Rank :{hidden_size * num_layers / 16}')
    print(f'1/8 总Rank :{hidden_size * num_layers / 8}')


breakpoint()


def compute_metrics(eval_pred: EvalPrediction):
    evaluation = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluation.compute(predictions=predictions, references=labels)


trainer = MeftTrainer[Trainer](
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=512,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=1e-3,
        weight_decay=1e-2,
        # warmup_ratio=0.1,    # 新加
        lr_scheduler_type="cosine",
        bf16=True,
        bf16_full_eval=True,
        # deepspeed={
        #     "train_batch_size": "auto",
        #     "gradient_accumulation_steps": "auto",
        #     "zero_optimization": {
        #         "stage": 1,
        #     },
        # },
        use_liger_kernel=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    ),
    data_collator=None,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
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


trainer.train()


# trainer.evaluate()
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"测试集top-1准确率: {test_results['eval_accuracy']:.4f}")
