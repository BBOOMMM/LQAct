import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from typing import List,Tuple

import fire
import torch
import transformers
from datasets import load_dataset,load_from_disk


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PeftModel
)
from transformers import LlamaForCausalLM, LlamaTokenizer,EarlyStoppingCallback

from utils.prompter import Prompter
from meft import MeftConfig, MeftTrainer

import huggingface_hub
from transformers import BitsAndBytesConfig
from trl import SFTTrainer,SFTConfig
from datasets import load_dataset

import argparse
from tqdm import tqdm
import random

from torch.utils.data import DataLoader
from collections import defaultdict
from transformers import DataCollatorForSeq2Seq

print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")

device_map = "cuda" if torch.cuda.is_available() else "cpu"

base_model = "meta-llama/Llama-2-7b-hf"
    
model = LlamaForCausalLM.from_pretrained(
    base_model,
    # quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference


# # wikitext2
# seed = 42
# max_length = 1024
# dataset_path = "Salesforce/wikitext"
# subset_name = "wikitext-2-raw-v1"

# ds=load_dataset(dataset_path, subset_name)
# def tokenize_function(example):
#     return tokenizer(example["text"], truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')

# tokenized_datasets = ds.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
# def group_texts(examples,block_size=128):
#     # Concatenate all texts.
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#     total_length = (total_length // block_size) * block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# lm_datasets = tokenized_datasets.map(
#     group_texts,
#     batched=True,
#     batch_size=24,
#     num_proc=4,
# )

# train_data=lm_datasets["train"]
# val_data=lm_datasets["validation"]

# eval_dataloader = DataLoader(
#     val_data,
#     batch_size=8,      # 一个 batch 的样本数，按显存调
#     shuffle=False,
# )



# alpaca
cutoff_len = 256
prompt_template_name = "alpaca"
train_on_inputs = True  # if False, masks out inputs in loss
add_eos_token = False
prompter = Prompter(prompt_template_name)
data_path = "yahma/alpaca-cleaned"
val_set_size = 1000
data_name = data_path
def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
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
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    data = load_dataset("json", data_files=data_path)
else:
    data = load_dataset(data_path)
    
if val_set_size > 0:
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    # train_data = (
        # train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    # )
    # val_data = (
    #     train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    # )
    val_data = train_val["test"].shuffle().map(
        generate_and_tokenize_prompt,
        remove_columns=train_val["test"].column_names,  # 关键：去掉 instruction/input/output 等原始列
    )
else:
    # train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

data_collator=transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

eval_dataloader = DataLoader(
    val_data,
    batch_size=128,
    shuffle=False,
    collate_fn=data_collator,   # 关键：用 collator 做 pad
)


device = next(model.parameters()).device
model.eval()

# (Pdb) model.model
# LlamaModel(
#   (embed_tokens): Embedding(32000, 4096)
#   (layers): ModuleList(
#     (0-31): 32 x LlamaDecoderLayer(
#       (self_attn): LlamaAttention(
#         (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#         (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
#         (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
#         (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#       )
#       (mlp): LlamaMLP(
#         (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
#         (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
#         (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
#         (act_fn): SiLUActivation()
#       )
#       (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
#       (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
#     )
#   )
#   (norm): LlamaRMSNorm((4096,), eps=1e-05)
#   (rotary_emb): LlamaRotaryEmbedding()
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
            activations[name]['output'].append(y.detach().to("cpu"))
        return hook

    for i, layer in enumerate(model.model.layers):
        # attn
        handles.append(layer.self_attn.q_proj.register_forward_hook(get_hook(f"layer_{i}.self_attn_q_proj")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(get_hook(f"layer_{i}.self_attn_k_proj")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(get_hook(f"layer_{i}.self_attn_v_proj")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(get_hook(f"layer_{i}.self_attn_o_proj")))
        
        # mlp
        handles.append(layer.mlp.gate_proj.register_forward_hook(get_hook(f"layer_{i}.mlp_gate_proj")))
        handles.append(layer.mlp.up_proj.register_forward_hook(get_hook(f"layer_{i}.mlp_up_proj")))
        handles.append(layer.mlp.down_proj.register_forward_hook(get_hook(f"layer_{i}.mlp_down_proj")))
        handles.append(layer.mlp.act_fn.register_forward_hook(get_hook(f"layer_{i}.mlp_act_fn")))
        
        # input_layernorm
        handles.append(layer.input_layernorm.register_forward_hook(get_hook(f"layer_{i}.input_layernorm")))
        
        # post_attention_layernorm
        handles.append(layer.post_attention_layernorm.register_forward_hook(get_hook(f"layer_{i}.post_attention_layernorm")))
        
register_hooks()

with torch.no_grad():
    for step, batch in enumerate(eval_dataloader):
        if step >= 3:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            **batch,
            output_hidden_states=True,  # 额外拿各层 hidden_states
            return_dict=True,
        )
        # outputs.hidden_states: tuple(len = num_layers + 1)
        #   hidden_states[0] 是 embedding 之后的
        #   hidden_states[L+1] 是第 L 层 block 的输出


for h in handles:
    h.remove()


# Save activations to disk
import pickle
with open(f"activations_{data_name}.pkl", "wb") as f:
    pickle.dump(activations, f)

breakpoint()