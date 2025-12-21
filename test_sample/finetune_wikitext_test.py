import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from typing import List,Tuple

import fire
import torch
import transformers
from datasets import load_dataset,load_from_disk

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer,EarlyStoppingCallback

from utils.prompter import Prompter
from meft import MeftConfig, MeftTrainer

import huggingface_hub
from transformers import BitsAndBytesConfig
from trl import SFTTrainer,SFTConfig

import argparse

print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")


from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")

# print(ds)

base_model = "meta-llama/Llama-2-7b-hf"
device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
load_in_8bit = True

tokenizer = LlamaTokenizer.from_pretrained(base_model)

tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
tokenizer.padding_side = "left"  # Allow batched inference

if load_in_8bit:    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_type=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    

if load_in_8bit:
    model = prepare_model_for_kbit_training(model)
else:
    pass

config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
model = get_peft_model(model, config)

def tokenize_function(example):
    # return tokenizer(example["text"], truncation=True, max_length=512, padding='max_length', return_tensors='pt')
    return tokenizer(example["text"])

tokenized_datasets = ds.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# block_size = tokenizer.model_max_length
block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=24,
    num_proc=4,
)

training_args = transformers.TrainingArguments(
    f"{base_model}-finetuned-wikitext2",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    no_cuda = False,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

trainer.train()

lora_weights_output_dir = "./lora-wikitext2"
print("saving lora adapter to ", lora_weights_output_dir)
model.save_pretrained(lora_weights_output_dir)