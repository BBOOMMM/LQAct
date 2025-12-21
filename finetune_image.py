import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from typing import List, Tuple
import numpy as np

import torch
import transformers
from datasets import load_dataset
import evaluate
import argparse

from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    Trainer, TrainingArguments, EvalPrediction,
)

from meft import MeftConfig, MeftTrainer

import huggingface_hub

print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--lora_weights_output_dir", type=str, default="./output/cifar100/meft_norm_attn_mlp_0.125_compress_lora_weights")
    parser.add_argument("--dataset_path", type=str, default="uoft-cs/cifar100", help="uoft-cs/cifar100 or ufldl-stanford/svhn  cropped-digits  or ethz/food101")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--val_set_size", type=int, default=10000)
    parser.add_argument("--num_labels", type=int, default=100)
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="whether to load in 8bit")
    
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--micro_batch_size", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    
    parser.add_argument("--using_meft", action="store_true", default=False, help="whether to use mefttrainer")
    parser.add_argument("--using_compress", action="store_true", default=False, help="whether to use compress")
    parser.add_argument("--compress_rank", type=float, default=0.125, help="the ratio of compressing")
    parser.add_argument("--patch_locations", type=int, default=1, help="the location of patching,1 means ('ckpt_layer',),and 2 means (norm, ckpt_attn, ckpt_mlp,)")
    return parser.parse_args()

def main(args):
    if args.patch_locations == 1:
        meft_patch_locations = ("ckpt_layer",)
    elif args.patch_locations == 2:
        meft_patch_locations = ("norm", "ckpt_attn", "ckpt_mlp",)
    else:
        raise ValueError("patch_locations must be 1 or 2")
    
    print(
        f"Training ViT_B_16 model:\n"
        f"base_model: {args.base_model}\n"
        f"load_in_8bit: {args.load_in_8bit}\n"
        f"dataset_path: {args.dataset_path}\n"
        f"subset_name: {args.subset}"
        f"num_labels: {args.num_labels}\n"
        f"val_set_size: {args.val_set_size}\n"
        f"lora_weights_output_dir: {args.lora_weights_output_dir}\n"
        
        f"using_meft: {args.using_meft}\n"
        f"using_compress: {args.using_compress}\n"
        f"compress_rank: {args.compress_rank}\n"
        f"patch_locations: {meft_patch_locations}\n"

        f"batch_size: {args.micro_batch_size * args.gradient_accumulation_steps}\n"
        f"micro_batch_size: {args.micro_batch_size}\n"
        f"gradient_accumulation_steps: {args.gradient_accumulation_steps}\n"
        f"num_epochs: {args.num_train_epochs}\n"
        f"learning_rate: {args.learning_rate}\n"
        f"weight_decay: {args.weight_decay}\n"
        
        f"val_set_size: {args.val_set_size}\n"
        
        f"lora_r: {args.lora_r}\n"
        f"lora_alpha: {args.lora_alpha}\n"
        f"lora_dropout: {args.lora_dropout}\n"
    )
    
    # 加载数据集
    if args.dataset_path =="uoft-cs/cifar100":
        dataset = load_dataset(args.dataset_path)
        num_labels=100
    elif args.dataset_path == "ethz/food101":
        dataset = load_dataset(args.dataset_path)
        num_labels=101
    elif args.dataset_path == "ufldl-stanford/svhn":
        dataset = load_dataset(args.dataset_path, args.subset)
        num_labels=10
    else:
        raise ValueError("dataset_path must be specified")
    
    model_name = args.base_model
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        device_map=args.device_map,
    )
    

    # 数据预处理函数 - 适配不同数据集的图像字段
    def transform(examples):
        # 根据数据集类型选择正确的图像字段
        image_key = "image" if "image" in examples else "img"
        inputs = processor(
            images=[x.convert("RGB") for x in examples[image_key]],
            return_tensors="pt"
        )
        # 对于不同数据集，标签字段可能不同
        label_key = "fine_label" if "fine_label" in examples else "label"
        return {
            "pixel_values": inputs.pixel_values,
            "labels": examples[label_key]  # 统一使用"labels"作为键名，符合Trainer要求
        }

    dataset = dataset.with_transform(transform)
    
    # 分割训练集和验证集
    if args.dataset_path == "ethz/food101":
        test_data = dataset["validation"]
    else:
        test_data = dataset['test']
    train_val = dataset["train"].train_test_split(
        test_size=args.val_set_size, shuffle=True, seed=42
    )
    train_data = train_val['train']
    val_data = train_val['test']

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
    
    # 计算评估指标
    def compute_metrics(eval_pred: EvalPrediction):
        evaluation = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return evaluation.compute(predictions=predictions, references=labels)
    
    # 训练参数配置
    training_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=32,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        bf16_full_eval=True,
        use_liger_kernel=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        output_dir=args.lora_weights_output_dir,
    )
    
    # 根据参数决定使用MeftTrainer还是普通Trainer
    if args.using_meft:
        print("使用MeftTrainer进行训练")
        trainer = MeftTrainer(
            model=model,
            args=training_args,
            data_collator=None,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics,
            meft_config=MeftConfig(
                patch_locations=meft_patch_locations,
                compress_kwargs={"rank": args.compress_rank} if args.using_compress else None,
            ),
        )
    else:
        print("使用普通Trainer进行训练")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=None,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics,
        )
    
    # 开始训练和评估
    trainer.train()
    trainer.evaluate()
    
    # 在测试集上评估模型性能
    print("\n正在测试集上评估模型...")
    test_results = trainer.evaluate(eval_dataset=test_data)
    print(f"测试集top-1准确率: {test_results['eval_accuracy']:.4f}")
    
    print("保存LoRA适配器到: ", args.lora_weights_output_dir)
    model.save_pretrained(args.lora_weights_output_dir)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
