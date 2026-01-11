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
from torch.utils.data import DataLoader

from meft import MeftConfig, MeftTrainer

from collections import defaultdict

import huggingface_hub

print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")

# 加载数据集
dataset = load_dataset("uoft-cs/cifar100")
num_labels=100


model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
    device_map="cuda:0",
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


val_set_size = 1000
test_data = dataset['test']
train_val = dataset["train"].train_test_split(
    test_size=val_set_size, shuffle=True, seed=42
)
train_data = train_val['train']
val_data = train_val['test']


eval_dataloader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False,
)  # [bs, c, h, w]   [16, 3, 224, 224]


device = next(model.parameters()).device
model.eval()

# (Pdb) model.vit
# ViTModel(
#   (embeddings): ViTEmbeddings(
#     (patch_embeddings): ViTPatchEmbeddings(
#       (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
#     )
#     (dropout): Dropout(p=0.0, inplace=False)
#   )
#   (encoder): ViTEncoder(
#     (layer): ModuleList(
#       (0-11): 12 x ViTLayer(
#         (attention): ViTAttention(
#           (attention): ViTSelfAttention(
#             (query): Linear(in_features=768, out_features=768, bias=True)
#             (key): Linear(in_features=768, out_features=768, bias=True)
#             (value): Linear(in_features=768, out_features=768, bias=True)
#           )
#           (output): ViTSelfOutput(
#             (dense): Linear(in_features=768, out_features=768, bias=True)
#             (dropout): Dropout(p=0.0, inplace=False)
#           )
#         )
#         (intermediate): ViTIntermediate(
#           (dense): Linear(in_features=768, out_features=3072, bias=True)
#           (intermediate_act_fn): GELUActivation()
#         )
#         (output): ViTOutput(
#           (dense): Linear(in_features=3072, out_features=768, bias=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#         )
#         (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#         (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#       )
#     )
#   )
#   (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
# )



activations = defaultdict(lambda: defaultdict(list))
handles = []

def register_hooks():
    def get_hook(name):
        def hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            y = output[0] if isinstance(output, tuple) else output
            
            # activations[name].append(input.detach().to("cpu"), output.detach().to("cpu"))
            activations[name]['input'].append(x.detach().to("cpu"))
            # activations[name]['output'].append(y.detach().to("cpu"))
        return hook

    for i, layer in enumerate(model.vit.encoder.layer):
        # attn
        handles.append(layer.attention.attention.query.register_forward_hook(get_hook(f"layer_{i}.self_attn_q_proj")))
        handles.append(layer.attention.attention.key.register_forward_hook(get_hook(f"layer_{i}.self_attn_k_proj")))
        handles.append(layer.attention.attention.value.register_forward_hook(get_hook(f"layer_{i}.self_attn_v_proj")))
        handles.append(layer.attention.output.dense.register_forward_hook(get_hook(f"layer_{i}.self_attn_o_proj")))
        
        # mlp
        handles.append(layer.intermediate.dense.register_forward_hook(get_hook(f"layer_{i}.mlp_gate_proj")))
        handles.append(layer.intermediate.intermediate_act_fn.register_forward_hook(get_hook(f"layer_{i}.mlp_up_proj")))
        handles.append(layer.output.dense.register_forward_hook(get_hook(f"layer_{i}.mlp_down_proj")))
        
        # input_layernorm
        handles.append(layer.layernorm_before.register_forward_hook(get_hook(f"layer_{i}.input_layernorm")))
        
        # post_attention_layernorm
        handles.append(layer.layernorm_after.register_forward_hook(get_hook(f"layer_{i}.post_attention_layernorm")))

register_hooks()

with torch.no_grad():
    for step, batch in enumerate(eval_dataloader):
        if step >= 1:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # outputs = model(
        #     **batch,
        #     output_hidden_states=True,  # 额外拿各层 hidden_states
        #     return_dict=True,
        # )
        # outputs.hidden_states: tuple(len = num_layers + 1)
        #   hidden_states[0] 是 embedding 之后的
        #   hidden_states[L+1] 是第 L 层 block 的输出


for h in handles:
    h.remove()


# Save activations to disk
# defaultdict -> 普通 dict，去掉 default_factory
activations_to_save = {
    layer_name: {
        io_type: tensors   # 这里还是 list[Tensor]，后面读取时再自己处理
        for io_type, tensors in io_dict.items()
    }
    for layer_name, io_dict in activations.items()
}


import pickle
import os
os.makedirs("activations_save", exist_ok=True)
with open(f"activations_save/activations_vit_cifar100.pkl", "wb") as f:
    pickle.dump(activations_to_save, f)