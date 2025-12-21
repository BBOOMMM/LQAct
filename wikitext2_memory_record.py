import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from meft import MeftConfig, MeftTrainer
from transformers import LlamaForCausalLM, LlamaTokenizer,ViTImageProcessor, ViTForImageClassification,EvalPrediction,TrainingArguments,set_seed
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from trl import SFTTrainer,SFTConfig

import logging
import socket
from datetime import datetime, timedelta

from torch.autograd.profiler import record_function
from utils.prompter import Prompter
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.profiler._memory_profiler import _CATEGORY_TO_COLORS, _CATEGORY_TO_INDEX
import argparse
import evaluate
from pathlib import Path
import huggingface_hub

print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")


from datasets import load_dataset

def set_all_seeds(seed: int):
    from transformers import set_seed
    set_seed(seed)
    import random
    random.seed(seed)
    np = __import__("numpy")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_path", type=str, default="Salesforce/wikitext")
    parser.add_argument("--subset_name", type=str, default="wikitext-2-raw-v1")
    
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--load_in_8bit", action="store_true",default=False,help="whether to load in 8bit")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_weights_output_dir", type=str, default="./output/wikitext2/meft_norm_attn_mlp_0.125_compress_lora_weights")
    
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    
    parser.add_argument("--using_meft", action="store_true",default=False,help="whether to use mefttrainer")
    parser.add_argument("--using_compress", action="store_true",default=False,help="whether to use compress")
    parser.add_argument("--compress_rank", type=float, default=0.125,help="the ratio of compressing")
    parser.add_argument("--patch_locations",type=int,default=1,help="the location of patching,1 means ('ckpt_layer',),and 2 means (norm, ckpt_attn, ckpt_mlp,)")
    return parser.parse_args()

def main(args):
    set_all_seeds(42)
    print("device_map: ",args.device_map)
    if args.patch_locations == 1:
        meft_patch_locations = ("ckpt_layer",)
        meft_prefix="layer"
    elif args.patch_locations == 2:
        meft_patch_locations = ("norm", "ckpt_attn", "ckpt_mlp",)
        meft_prefix="norm_attn_mlp"
    else:
        raise ValueError("patch_locations must be 1 or 2")
    save_name=f"wikitext2/{meft_prefix}_{args.compress_rank}"
    if not args.using_meft:
        save_name=f"wikitext2/lora"
    
    print(
            f"Training LLaMA-2 model with MeftTrainer:\n"
            f"base_model: {args.base_model}\n"
            f"load_in_8bit: {args.load_in_8bit}\n"
            f"data_path: {args.dataset_path}\n"
            f"subset_name: {args.subset_name}\n"
            f"lora_weights_output_dir: {args.lora_weights_output_dir}\n"
            f"using_meft: {args.using_meft}\n"
            f"compress_rank: {args.compress_rank}\n"
            f"patch_locations: {meft_patch_locations}\n"
            f"batch_size: {args.micro_batch_size * args.gradient_accumulation_steps}\n"
            f"micro_batch_size: {args.micro_batch_size}\n"
            f"gradient_accumulation_steps: {args.gradient_accumulation_steps}\n"
            f"num_epochs: {args.num_epoch}\n"
            f"learning_rate: {args.learning_rate}\n"
            f"max_length: {args.max_length}\n"
            f"lora_r: {args.lora_r}\n"
            f"lora_alpha: {args.lora_alpha}\n"
            f"lora_dropout: {args.lora_dropout}\n"
    )
    ds=load_dataset(args.dataset_path,args.subset_name)
    
    base_model = args.base_model
    device_map = args.device_map
    
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    tokenizer.padding_side = "left"  # Allow batched inference

    if args.load_in_8bit:    
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
        

    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    else:
        pass

    config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model = get_peft_model(model, config)
        
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=args.max_length, padding='max_length', return_tensors='pt')

    tokenized_datasets = ds.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    def group_texts(examples,block_size=128):
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
    
    train_data=lm_datasets["train"]
    val_data=lm_datasets["validation"]
    print("slice train data to 1000 samples")
    train_data = train_data.select(range(1000))
    print(len(train_data[0]["input_ids"]))
    
    if args.using_meft:
        print("Using MeftTrainer")
        
        trainer = MeftTrainer[SFTTrainer](
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=SFTConfig(
                per_device_train_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=args.num_epoch,
                learning_rate=args.learning_rate,
                weight_decay=1e-2,
                lr_scheduler_type="cosine",
                bf16=True,
                bf16_full_eval=True,
                use_liger_kernel=True,
                logging_steps=10,
                optim="adamw_torch",
                eval_strategy="steps",
                save_strategy="steps",
                eval_steps=100,
                save_steps=200,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                output_dir=args.lora_weights_output_dir,
                save_total_limit=3,
                load_best_model_at_end=True,
                # ddp_find_unused_parameters=False if ddp else None,
                # group_by_length=group_by_length,
                report_to=None,
                # run_name=wandb_run_name if use_wandb else None,
            ),
            # data_collator=transformers.DataCollatorForSeq2Seq(
            #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            # ),
            # compute_metrics=compute_metrics,
            meft_config=MeftConfig(
                patch_locations=meft_patch_locations, # ("norm", "ckpt_attn", "ckpt_mlp",), # patch_locations=("ckpt_layer",),   
                compress_kwargs={"rank": args.compress_rank} if args.using_compress else None,
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping else None,
        )
    else:
        print("Using original Trainer")
        trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epoch,
            learning_rate=args.learning_rate,
            bf16=True,
            bf16_full_eval=True,
            use_liger_kernel=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=args.lora_weights_output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            # ddp_find_unused_parameters=False if ddp else None,
            # group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping else None,
    )
        
        
    model.config.use_cache = False

    class ProfCallback(TrainerCallback):
        def __init__(self, prof):
            self.prof = prof

        def on_step_end(self, args, state, control, **kwargs):
            self.prof.step()
    
    # Path("./profiler").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = save_name

    def trace_handler(prof: torch.profiler.profile):

        # Construct the trace file.
        # prof.export_chrome_trace(f"{file_prefix}.json.gz")

        # Construct the memory timeline file.
        prof.export_memory_timeline(f"./profiler/{file_prefix}.html", device="cuda:0")
        prof.export_memory_timeline(f"./profiler/{file_prefix}.json", device="cuda:0")


    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=0, wait=0, warmup=1, active=3, repeat=1),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
        on_trace_ready=trace_handler,
    ) as prof:
        trainer.add_callback(ProfCallback(prof=prof))
        trainer.train()
        
    with open(f"./profiler/{file_prefix}.json", "r") as f:
    # with open(f"./profiler/shijx-3090_Jul_13_04_23_54.json", "r") as f:
        mt = json.load(f)
    # with open(f"./profiler/autodl-container-476340bb9c-85c06c63_Sep_18_14_39_12.json", "r") as f:
    #     mt = json.load(f)
    times, sizes = np.array(mt[0]), np.array(mt[1])
    t_min = min(times)
    times -= t_min
    
    stacked = np.cumsum(sizes, axis=1) / 1024**3

    plt.figure(figsize=(16, 4))
    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        plt.fill_between(
            times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
        )
    plt.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS])

    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        print(category, (stacked[:, i+1] - stacked[:, i]).max())

    print(stacked[:, -1].max())
if __name__ == "__main__":
    args = parse_args()
    main(args)
