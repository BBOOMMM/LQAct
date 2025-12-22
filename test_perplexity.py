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
    PeftModel
)
from transformers import LlamaForCausalLM, LlamaTokenizer,EarlyStoppingCallback

from utils.prompter import Prompter
from meft import MeftConfig, MeftTrainer

import huggingface_hub
from transformers import BitsAndBytesConfig
from trl import SFTTrainer,SFTConfig

import argparse
from tqdm import tqdm
import random

print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")

device_map = "cuda" if torch.cuda.is_available() else "cpu"

base_model = "meta-llama/Llama-2-7b-hf"

load_in_8bit = False
if load_in_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf4",    
        bnb_8bit_compute_type=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
else:
    quantization_config = None
    
model = LlamaForCausalLM.from_pretrained(
    base_model,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=True,
)


tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

# lora_model = PeftModel.from_pretrained(
#     model,
#     "./lora-wikitext2", 
#     device_map=device_map,
#     torch_dtype=torch.float16,
# )

# merged_model = lora_model.merge_and_unload()

# merged_model.train(False)

# # 比较合并前后模型的部分权重
# param_names = list(model.state_dict().keys())[-5:]  # 取前5个参数名

# print("基础模型与合并模型的权重差异：")
# for name in param_names:
#     original = model.state_dict()[name]
#     merged = merged_model.state_dict()[name]
#     # 计算权重差异的平均值
#     diff = torch.mean(torch.abs(original - merged)).item()
#     print(f"{name}: 平均差异 = {diff}")
    
def eval_ppl(model, tokenizer):
    model.eval()
    max_length = model.config.max_position_embeddings
    stride = max_length
    print(f"max_length: {max_length}, stride: {stride}")
    test = load_dataset("Salesforce/wikitext", "wikitext-2-v1",split="test")["text"]
    # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    nlls = []
    encodings = tokenizer("\n\n".join(test), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

def eval_ppl_v2(model,tokenizer):
    test_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1",split="test")
    encodings = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    stride = 512
    print(f"max_length: {max_length}, stride: {stride}")
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device_map)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl

def batched_perplexity(model, dataset, tokenizer, batch_size, stride):
    max_len = model.config.max_position_embeddings
    device = model.device
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    text_len = encodings.input_ids.size(1)
    lls = []

    for i in tqdm(range(0, text_len, batch_size * stride)):
        begin_locs, end_locs, trg_lens = [], [], []
        for j in range(batch_size):
            j = i + j * stride
            if j >= text_len:
                break
            begin_loc = max(j + stride - max_len, 0)
            end_loc = min(j + stride, text_len)
            trg_len = end_loc - j  # may be different from stride on last loop

            begin_locs.append(begin_loc)
            end_locs.append(end_loc)
            trg_lens.append(trg_len)

        input_ids = [encodings.input_ids[:, b:e] for b, e in zip(begin_locs, end_locs)]
        target_end_locs = [sen.size(-1) for sen in input_ids]
        input_ids = [
            F.pad(sen, (0, max_len - sen.size(-1)), "constant", 0) for sen in input_ids
        ] # we dont need attention mask as long as these padded token is not involved in loss calculation
        input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

        target_ids = torch.ones_like(input_ids) * -100 # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
        for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
            labels = input_ids[i, -b:e].clone()
            target_ids[i, -b:e] = labels

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs["loss"] * sum(trg_lens)

        lls.append(log_likelihood)

    ppl = torch.exp(sum(torch.stack(lls) / end_locs[-1]))
    return ppl


def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader

def get_loaders(
    name, tokenizer, train_size=128, val_size=64,seed=0, seqlen=2048, test_only=False
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer,train_size,val_size,seed,seqlen,test_only)
    # elif 'c4' in name:
    #     return get_c4(tokenizer,train_size,val_size,seed,seqlen,test_only)
    # elif 'redpajama' in name:
    #     return get_redpajama(tokenizer,train_size,val_size,seed,seqlen)
    else:
        raise NotImplementedError
    
    
def test_ppl(model, tokenizer, datasets=['wikitext2'],ppl_seqlen=256):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        if hasattr(model,'lm_head') and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model,'lm_head'):
            # for gptqmodels
            classifier = None
        elif hasattr(model,'output'):
            # for internlm
            classifier = model.output
        else:
            raise NotImplementedError
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)


        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print(f'{dataset}:{ppl}')
        results[dataset] = ppl.item()
    model.config.use_cache = use_cache
    return results

# finetuned_ppl=eval_ppl(merged_model, tokenizer)
# oringinal_ppl=eval_ppl(model, tokenizer)
# print(f"finetuned ppl: {finetuned_ppl}, original ppl: {oringinal_ppl}")

# finetuned_ppl=eval_ppl_v2(merged_model, tokenizer)
# oringinal_ppl=eval_ppl_v2(model, tokenizer)
# print(f"finetuned ppl: {finetuned_ppl}, original ppl: {oringinal_ppl}")

# 1. 加载测试集（用于batched_perplexity，需传入Dataset对象）
test_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")

# 2. 调用四种PPL函数
# print("=== 1. eval_ppl（基础版） ===")
# # finetuned_ppl1 = eval_ppl(merged_model, tokenizer)
# original_ppl1 = eval_ppl(model, tokenizer)
# print(f"原始模型PPL: {original_ppl1:.2f}\n")

print("=== 2. eval_ppl_v2（精度优化版） ===")
original_ppl2 = eval_ppl_v2(model, tokenizer)
print(f"原始模型PPL: {original_ppl2:.2f}\n")

print("=== 3. batched_perplexity（批量高效版） ===")
# finetuned_ppl3 = batched_perplexity(merged_model, test_dataset, tokenizer, batch_size=4, stride=512)
original_ppl3 = batched_perplexity(model, test_dataset, tokenizer, batch_size=4, stride=512)
print(f"原始模型PPL: {original_ppl3:.2f}\n")

print("=== 4. test_ppl（多模型适配版） ===")
# finetuned_ppl4 = test_ppl(merged_model, tokenizer, datasets=['wikitext2'], ppl_seqlen=256)['wikitext2']
original_ppl4 = test_ppl(model, tokenizer, datasets=['wikitext2'], ppl_seqlen=256)['wikitext2']
print(f"原始模型PPL: {original_ppl4:.2f}")
