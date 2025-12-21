import os

# -------------------------- 1. 环境配置与依赖导入（保持不变）--------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    LlamaTokenizer, LlamaForCausalLM, DataCollatorForSeq2Seq
)
from datasets import load_dataset
from numpy.linalg import svd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=6, help="Target layer to extract activations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for sampling")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length for sampling")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model loading")
    return parser.parse_args()

def main(args):
    # -------------------------- 2. 模型与Tokenizer加载（保持不变）--------------------------
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0  # 兜底pad_token
    tokenizer.padding_side = "left"

    # 加载模型（自动分配设备，开启隐藏状态输出）
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map
    )
    model.config.output_hidden_states = True  # 关键：让模型返回隐藏层状态
    model.eval()  # 推理模式，禁用Dropout

    # -------------------------- 3. 数据加载与预处理（保持不变，仅调整后续采样逻辑）--------------------------
    cutoff_len = args.seq_len
    from utils.prompter import Prompter  # 确保utils.prompter能正常导入
    prompter = Prompter("alpaca")
    train_on_inputs = True
    add_eos_token = True
    data_path = "yahma/alpaca-cleaned"
    val_set_size = 2000  # 数据划分用，不影响后续采样


    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        # 补全eos_token（如果未截断且未添加）
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result


    def generate_and_tokenize_prompt(data_point):
        # 生成Alpaca格式prompt
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        # 若不训练输入部分，将输入对应的labels设为-100（不参与损失计算）
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                user_prompt_len -= 1  # 减去eos_token的长度
            # 输入部分labels设为-100
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt


    # 加载数据集并预处理
    if data_path.endswith((".json", ".jsonl")):
        ds = load_dataset("json", data_files=data_path)
    else:
        ds = load_dataset(data_path)

    orig_cols = ds["train"].column_names
    # 划分训练集/验证集（仅用训练集做后续采样）
    if val_set_size > 0:
        train_val = ds["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].map(  # 仅用划分后的训练集
            generate_and_tokenize_prompt,
            remove_columns=orig_cols,
            num_proc=4  # 多进程加速预处理（可选）
        )
    else:
        train_data = ds["train"].map(
            generate_and_tokenize_prompt,
            remove_columns=orig_cols,
            num_proc=4
        )

    # -------------------------- 4. 核心：100次随机采样batch并计算目标层结果 --------------------------
    # 4.1 配置参数
    num_sampling = 100  # 采样100次
    batch_size = args.batch_size    # 每次采样32个样本
    target_layer = args.layer    # 目标层（Llama-2-7B共32层，索引0-31）
    save_dir = Path(f"bsz_seq_len_act/bsz_{batch_size}_seq_len_{cutoff_len}_layer{target_layer}")  # 结果保存目录
    save_dir.mkdir(exist_ok=True, parents=True)  # 确保目录存在

    # 4.2 数据整理器（处理padding，与原代码一致）
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    # 4.3 推断模型输入设备（适配accelerate自动分配）
    def infer_forward_device(m):
        try:
            return m.get_input_embeddings().weight.device
        except Exception:
            for p in m.parameters():
                return p.device
        return torch.device("cpu")

    fwd_device = infer_forward_device(model)
    print(f"模型输入设备：{fwd_device}")

    # 4.4 循环100次采样并计算
    for sample_idx in range(num_sampling):
        print(f"正在处理第 {sample_idx + 1}/{num_sampling} 个batch...")
        
        # -------------------------- 关键：随机抽取32个不重复样本 --------------------------
        # 从train_data中随机选择32个不同的索引（无放回采样，避免重复样本）
        random_indices = np.random.choice(
            len(train_data),  # 训练集总样本数
            size=batch_size,  # 每次取32个
            replace=False     # 无放回（确保同一次batch内样本不重复）
        )
        # 根据索引抽取batch数据
        sampled_batch_data = train_data.select(random_indices)
        
        # -------------------------- 处理batch并前向推理 --------------------------
        # 用DataLoader整理batch（仅1个batch，shuffle=False）
        batch_loader = DataLoader(
            sampled_batch_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=False
        )
        # 获取整理后的batch（仅1个，直接取第1个）
        batch = next(iter(batch_loader))
        
        # 仅保留模型需要的输入（input_ids/attention_mask），并搬至设备
        inputs = {k: v.to(fwd_device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        
        # 前向推理（禁用梯度，节省内存）
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # -------------------------- 提取目标层隐藏状态并计算SVD --------------------------
        hidden_states = outputs.hidden_states  # 所有层隐藏状态，shape: [num_layers+1, B, T, H]
        # 验证目标层合法性（Llama-2-7B隐藏状态包含"输入嵌入层+32个Transformer层"，共33个，索引0-32）
        assert target_layer < len(hidden_states), f"模型仅含 {len(hidden_states)} 层（0-{len(hidden_states)-1}），目标层 {target_layer} 超出范围"
        
        # 提取目标层隐藏状态：[B, T, H]（B=32, T=序列长度, H=模型维度，Llama-2-7B为4096）
        hs = hidden_states[target_layer]
        B, T, H = hs.shape
        
        # 展平为2D矩阵（用于SVD）：[B*T, H]（将所有样本的所有token的隐藏状态拼接）
        hs2d = hs.reshape(B * T, H)
        
        # 转换为numpy数组（CPU上计算SVD）
        hs_np = hs2d.to(torch.float32).cpu().numpy()
        U, s, Vt = svd(hs_np, full_matrices=False)  # full_matrices=False：仅计算必要的奇异值
        
        # -------------------------- 保存结果（奇异值+采样信息） --------------------------
        # 保存内容：采样索引（可回溯样本）、奇异值、目标层信息
        result = {
            "sample_idx": sample_idx + 1,  # 采样序号（1-100）
            "target_layer": target_layer,  # 目标层
            "sampled_indices": random_indices.tolist(),  # 本次采样的样本索引（可回溯）
            "singular_values": s.tolist(),  # 奇异值（转换为Python列表，支持JSON序列化）
            "hs_shape": [B, T, H],  # 目标层隐藏状态形状
            "svd_shapes": {
                "U": U.shape,
                "s": s.shape,
                "Vt": Vt.shape
            }  # SVD结果形状（验证用）
        }
        
        # 保存到JSON文件（按采样序号命名，避免覆盖）
        save_path = save_dir / f"batch_{sample_idx + 1}_layer_{target_layer}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # -------------------------- 可选：为每个batch生成奇异值图 --------------------------
        ranks = np.arange(1, len(s) + 1)  # 奇异值索引（1-4096）
        
        # 1. 线性坐标图
        plt.figure(figsize=(10, 4))
        plt.plot(ranks, s, marker='o', markersize=2, linewidth=1, color="#1f77b4")
        plt.xlabel("Index of Singular Values")
        plt.ylabel("Singular Value Size")
        plt.title(f"Batch {sample_idx + 1} - Singular Values (Layer {target_layer})")
        plt.tight_layout()
        plt.savefig(save_dir / f"batch_{sample_idx + 1}_layer_{target_layer}_linear.png", dpi=150)
        plt.close()  # 关闭图，避免内存泄漏
        
        # 2. 对数坐标图（更易观察长尾）
        plt.figure(figsize=(10, 4))
        plt.plot(ranks, s, marker='o', markersize=2, linewidth=1, color="#ff7f0e")
        plt.yscale("log")
        plt.xlabel("Index of Singular Values")
        plt.ylabel("Singular Value Size (Log Scale)")
        plt.title(f"Batch {sample_idx + 1} - Singular Values (Layer {target_layer}, Log)")
        plt.tight_layout()
        plt.savefig(save_dir / f"batch_{sample_idx + 1}_layer_{target_layer}_log.png", dpi=150)
        plt.close()

    print(f"\n所有采样完成！结果已保存至：{save_dir.absolute()}")
    
if __name__ == "__main__":
    args=parse_args()
    main(args)
