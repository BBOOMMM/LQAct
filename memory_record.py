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
from transformers import LlamaForCausalLM, LlamaTokenizer,ViTImageProcessor, ViTForImageClassification,EvalPrediction,TrainingArguments
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

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf",help="google/vit-base-patch16-224-in21k")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--device_map", type=str, default="auto")
    # parser.add_argument("--data_path", type=str, default="yahma/alpaca-cleaned",help="uoft-cs/cifar100")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca")
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--train_on_inputs", action="store_true", default=False)
    parser.add_argument("--using_meft", action="store_true", default=False)
    parser.add_argument("--vision", action="store_true", default=False)
    parser.add_argument("--compress_rank", type=float, default=0.25)
    parser.add_argument("--patch_locations", type=int, default=1,help="1:(ckpt_layer,), 2:(norm, ckpt_attn, ckpt_mlp,)")

    parser.add_argument("--subset",type=str,default=None)
    parser.add_argument("--val_set_size",type=int,default=20)
    return parser.parse_args()
    

def set_all_seeds(seed: int):
    from transformers import set_seed
    set_seed(seed)
    import random
    random.seed(seed)
    np = __import__("numpy")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   

def main():
    set_all_seeds(42)
    args = parse_args()
    
    if args.patch_locations == 1:
        meft_patch_locations = ("ckpt_layer",)
        loc_prefix="layer"
    elif args.patch_locations == 2:
        meft_patch_locations=("norm", "ckpt_attn", "ckpt_mlp",)
        loc_prefix="norm_attn_mlp"
    else:
        raise ValueError("patch_locations must be 1 or 2")
    if args.vision:
        base_model = "google/vit-base-patch16-224-in21k"
        data_path = "uoft-cs/cifar100"
        if args.compress_rank == 1:
            save_name = f"vision/{loc_prefix}_ckpt_{args.micro_batch_size}_{args.cutoff_len}"
        else:
            save_name=f"vision/{loc_prefix}_{args.compress_rank}_{args.micro_batch_size}_{args.cutoff_len}"
        
        if not args.using_meft:
            save_name=f"vision/lora_{args.micro_batch_size}_{args.cutoff_len}"
        else:
            pass
    else:
        base_model = "meta-llama/Llama-2-7b-hf"
        data_path = "yahma/alpaca-cleaned"
        if args.compress_rank == 1:
            save_name = f"language/{loc_prefix}_ckpt_{args.micro_batch_size}_{args.cutoff_len}"
        else:
            save_name=f"language/{loc_prefix}_{args.compress_rank}_{args.micro_batch_size}_{args.cutoff_len}"
        
        if not args.using_meft:
            save_name=f"language/lora_{args.micro_batch_size}_{args.cutoff_len}"
        else:
            pass
    print(
        f"base_model: {base_model}\n"
        f"device_map: {args.device_map}\n"
        f"data_path: {data_path}\n"
        f"vision: {args.vision}\n"
        f"language: {not args.vision}\n"
        f"using_meft: {args.using_meft}\n"
        f"compress_rank: {args.compress_rank}\n"
        f"patch_locations: {meft_patch_locations}\n"
        f"micro_batch_size: {args.micro_batch_size}\n"
        f"gradient_accumulation_steps: {args.gradient_accumulation_steps}\n"
        f"cutoff_len: {args.cutoff_len}\n"
        f"save_name: {save_name}\n"

    )
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point,train_on_inputs=True,add_eos_token=True):
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
    
    if not args.vision:
        print("Language task,preparing model and dataset")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
        )

        tokenizer = LlamaTokenizer.from_pretrained(base_model)

        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference
        print("finish loading tokenizer")
        
        config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        print("finish loading model")
        model.print_trainable_parameters()
        
        data=load_dataset(data_path)
        print("finish loading dataset")
        prompter = Prompter("alpaca")

        if args.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=args.val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            val_data = None
        
        print("slicing train_data to 1000 for testing")
        train_data = train_data.select(range(1000))
        print("after slicing, train_data:", train_data)
        
    else:
        print("Vision task,preparing model and dataset")
        if data_path =="uoft-cs/cifar100":
            dataset = load_dataset(data_path)
            num_labels=100
        elif data_path == "ethz/food101":
            dataset = load_dataset(data_path)
            num_labels=101
        elif data_path == "ufldl-stanford/svhn":
            dataset = load_dataset(data_path, args.subset)
            num_labels=10
        else:
            raise ValueError("dataset_path must be specified")
        
        model_name = base_model
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            device_map=args.device_map,
        )
        print("finish loading model")

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
        if data_path == "ethz/food101":
            test_data = dataset["validation"]
        else:
            test_data = dataset['test']
        train_val = dataset["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = train_val['train']
        val_data = train_val['test']
        print("finish loading dataset")

        # 配置LoRA
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["query", "value"],
            bias="none",
            modules_to_save=["classifier"],
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("slicing train_data to 1000 for testing")
        train_data = train_data.select(range(1000))
        print("after slicing, train_data:", train_data)
    
    if args.vision:
        print("vision task")
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
            num_train_epochs=1,
            learning_rate=1e-1,
            weight_decay=0.0,
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
            output_dir=args.output_dir,
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
                    compress_kwargs={"rank": args.compress_rank} if args.compress_rank < 1.0 else None,
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
    else:
        print("Language task")
        if args.using_meft:
            print("using MeftTrainer")
            trainer = MeftTrainer[SFTTrainer](
                model=model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=SFTConfig(
                    per_device_train_batch_size=args.micro_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    warmup_steps=100,
                    num_train_epochs=1,
                    learning_rate=3e-5,
                    weight_decay=1e-2,
                    lr_scheduler_type="cosine",
                    bf16=True,
                    bf16_full_eval=True,
                    use_liger_kernel=True,
                    logging_steps=10,
                    optim="adamw_torch",
                    eval_strategy="steps" if args.val_set_size > 0 else "no",
                    save_strategy="steps",
                    eval_steps=100 if args.val_set_size > 0 else None,
                    save_steps=200,
                    # metric_for_best_model="eval_loss",
                    # greater_is_better=False,
                    output_dir=args.output_dir,
                    # save_total_limit=3,
                    # load_best_model_at_end=True if val_set_size > 0 else False,
                ),
                # data_collator=transformers.DataCollatorForSeq2Seq(
                #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                # ),
                # compute_metrics=compute_metrics,
                meft_config=MeftConfig(
                    patch_locations=meft_patch_locations, # ("norm", "ckpt_attn", "ckpt_mlp",), # patch_locations=("ckpt_layer",),   
                    compress_kwargs={"rank": args.compress_rank} if args.compress_rank < 1.0 else None,
                ),
            )
        else:
            print("using original trainer")
            trainer = transformers.Trainer(
                model=model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=transformers.TrainingArguments(
                    per_device_train_batch_size=args.micro_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    warmup_steps=100,
                    num_train_epochs=1,
                    learning_rate=3e-5,
                    bf16=True,
                    bf16_full_eval=True,
                    use_liger_kernel=True,
                    logging_steps=10,
                    optim="adamw_torch",
                    eval_strategy="steps" if args.val_set_size > 0 else "no",
                    save_strategy="steps",
                    eval_steps=100 if args.val_set_size > 0 else None,
                    save_steps=200,
                    # metric_for_best_model="eval_loss",
                    # greater_is_better=False,
                    # output_dir=lora_weights_output_dir,
                    # save_total_limit=3,
                    # load_best_model_at_end=True if val_set_size > 0 else False,
                    # ddp_find_unused_parameters=False if ddp else None,
                    # group_by_length=group_by_length,
                    # report_to="wandb" if use_wandb else None,
                    # run_name=wandb_run_name if use_wandb else None,
                ),
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )
    class ProfCallback(TrainerCallback):
        def __init__(self, prof):
            self.prof = prof

        def on_step_end(self, args, state, control, **kwargs):
            self.prof.step()
    
    Path("./profiler").mkdir(parents=True, exist_ok=True)
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
        schedule=torch.profiler.schedule(skip_first=0, wait=0, warmup=0, active=3, repeat=1),
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
    main()