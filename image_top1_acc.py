import os

# 环境变量配置
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import numpy as np
import traceback

import torch
from datasets import load_dataset
import evaluate
import argparse

from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    Trainer, TrainingArguments, EvalPrediction,
)
from peft import PeftModel

import huggingface_hub

# Hugging Face登录（替换为你的实际Token）
print("登录到huggingface_hub...")
huggingface_hub.login(token="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")
print("登录成功!")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/vit-base-patch16-224-in21k",
                      help="预训练基础模型名称")
    parser.add_argument("--lora_model", type=str, default="./output/cifar100/meft_layer_0.25_compress_lora_weights",
                      help="LoRA权重文件路径")
    parser.add_argument("--device_map", type=str, default="auto",
                      help="设备映射（auto/cpu/cuda:0等）")
    parser.add_argument("--data_path", type=str, default="uoft-cs/cifar100",
                      help="数据集名称（支持：uoft-cs/cifar100, ufldl-stanford/svhn, ethz/food101）")
    parser.add_argument("--subset", type=str, default="cropped_digits",
                      help="数据集子集（仅SVHN需要，默认cropped_digits）")
    parser.add_argument("--output_dir", type=str, default="output/image",
                      help="评估结果输出目录")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                      help="评估时每个设备的批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="数据加载的工作进程数")
    parser.add_argument("--cifar_label_type", type=str, default="fine_label",
                      help="CIFAR100的标签类型 (fine_label 或 coarse_label)")
    return parser.parse_args()

def get_label_field(dataset, data_path, cifar_label_type=None):
    """获取数据集的标签字段，特殊处理CIFAR100"""
    if data_path == "uoft-cs/cifar100":
        if cifar_label_type not in ["fine_label", "coarse_label"]:
            raise ValueError(f"CIFAR100标签类型必须是 'fine_label' 或 'coarse_label'，当前为: {cifar_label_type}")
        return cifar_label_type
    
    # 其他数据集的通用标签字段识别
    possible_fields = ["label", "labels", "target", "category"]
    sample = dataset[0]
    for field in possible_fields:
        if field in sample:
            return field
    raise ValueError(f"未识别标签字段！数据集包含字段: {list(sample.keys())}")

def get_image_field(dataset):
    """获取数据集的图像字段"""
    possible_fields = ["image", "img", "pictures"]
    sample = dataset[0]
    for field in possible_fields:
        if field in sample:
            return field
    raise ValueError(f"未识别图像字段！数据集包含字段: {list(sample.keys())}")

def main(args):
    # 数据集配置映射
    dataset_config = {
        "uoft-cs/cifar100": {
            "num_labels": 100 if args.cifar_label_type == "fine_label" else 20,
            "test_split": "test"
        },
        "ufldl-stanford/svhn": {"num_labels": 10, "test_split": "test"},
        "ethz/food101": {"num_labels": 101, "test_split": "validation"},
    }
    
    # 检查数据集是否支持
    if args.data_path not in dataset_config:
        raise ValueError(f"不支持的数据集: {args.data_path}，支持的数据集有: {list(dataset_config.keys())}")
    
    # 获取数据集配置
    config = dataset_config[args.data_path]
    num_labels = config["num_labels"]
    test_split = config["test_split"]
    
    # 加载基础模型
    print(f"加载基础模型: {args.base_model} (类别数: {num_labels})")
    try:
        model = ViTForImageClassification.from_pretrained(
            args.base_model,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"基础模型加载失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # 加载并合并LoRA模型
    print(f"加载LoRA模型: {args.lora_model}")
    try:
        lora_model = PeftModel.from_pretrained(
            model,
            args.lora_model,
            device_map=args.device_map,
            torch_dtype=torch.float16,
        )
        merged_model = lora_model.merge_and_unload()
        merged_model.eval()  # 设置为评估模式
        print(f"模型加载完成，设备映射: {args.device_map}")
    except Exception as e:
        print(f"LoRA模型加载失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # 加载图像处理器
    try:
        processor = ViTImageProcessor.from_pretrained(args.base_model)
        target_size = (processor.size["height"], processor.size["width"])
        print(f"图像处理器配置: 目标尺寸={target_size}")
    except Exception as e:
        print(f"图像处理器加载失败: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # 加载数据集
    print(f"加载数据集: {args.data_path} (子集: {args.subset})")
    try:
        if args.data_path == "ufldl-stanford/svhn":
            dataset = load_dataset(args.data_path, args.subset)
        else:
            dataset = load_dataset(args.data_path)
        
        test_dataset = dataset[test_split]
        print(f"测试集加载完成，样本数量: {len(test_dataset)}")
    except Exception as e:
        print(f"数据集加载错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # 确定图像和标签字段
    try:
        image_field = get_image_field(test_dataset)
        label_field = get_label_field(test_dataset, args.data_path, args.cifar_label_type)
        print(f"字段映射 - 图像字段: {image_field}, 标签字段: {label_field}")
    except ValueError as e:
        print(f"字段识别错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # 预处理函数 - 确保标签格式正确
    def preprocess_function(examples):
        """预处理图像和标签，确保输出格式正确"""
        # 处理图像
        images = examples[image_field]
        processed = processor(
            images=images,
            resize_size=target_size,  # 调整图像尺寸
            return_tensors="pt"       # 返回PyTorch张量
        )
        
        # 处理标签 - 直接使用原始标签（整数列表）
        processed["labels"] = examples[label_field]
        
        return processed
    
    # 预处理测试集
    print("开始预处理测试集...")
    try:
        processed_test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=test_dataset.column_names,
            desc="预处理测试集"
        )
    except Exception as e:
        print(f"预处理错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # 定义评估指标
    def compute_metrics(p: EvalPrediction):
        predictions = np.argmax(p.predictions, axis=1)
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=predictions, references=p.label_ids)
    
    # 定义数据整理函数 - 修复标签访问错误
    def collate_fn(batch):
        """将批量数据整理为模型可接受的格式"""
        # 处理pixel_values
        pixel_values = []
        for sample in batch:
            pv = sample["pixel_values"]
            # 确保是张量
            if isinstance(pv, list):
                pv = torch.tensor(pv, dtype=torch.float16)
            # 确保维度正确
            if pv.dim() == 3:
                pv = pv.unsqueeze(0)  # 添加批次维度
            pixel_values.append(pv)
        
        # 拼接成批次张量
        pixel_values = torch.cat(pixel_values, dim=0)
        
        # 处理标签 - 直接获取整数标签，不使用下标访问
        labels = torch.tensor(
            [sample["labels"] for sample in batch],  # 关键修复：移除[0]
            dtype=torch.long
        )
        
        return {"pixel_values": pixel_values, "labels": labels}
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_eval=True,
        no_cuda=False,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        dataloader_num_workers=args.num_workers,
        report_to="none"
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=merged_model,
        args=training_args,
        eval_dataset=processed_test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    # 执行评估
    print("开始评估模型...")
    try:
        metrics = trainer.evaluate()
        print("\n" + "="*60)
        print(f"评估结果 - 数据集: {args.data_path} (标签类型: {args.cifar_label_type})")
        print(f"Top-1 准确率: {metrics['eval_accuracy']:.4f} ({metrics['eval_accuracy']*100:.2f}%)")
        print(f"评估结果已保存至: {args.output_dir}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"评估过程出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    