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
from datasets import load_dataset

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


import random
import numpy as np
from transformers import set_seed as hf_set_seed
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
hf_set_seed(seed)  # transformers 内部用到的随机也统一
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-hf",  # the only required argument
    load_in_8bit: bool = False,
    data_path: str = "yahma/alpaca-cleaned",
    lora_weights_output_dir: str = "./output/lora_weights/meft_norm_attn_mlp_0.0625_compress_results_vgpu48",
    hf_ckpt_output_dir: str = "./output/hf_ckpt/meft_norm_attn_mlp_0.0625_compress_hf_ckpt_vgpu48",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 3e-5,
    cutoff_len: int = 256,
    val_set_size: int = 4000,
    warm_up_steps: int = 100,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # Meft params
    using_meft: bool = True,
    using_compress: bool = True,
    lowrank_plus_quantization: bool = False,
    compress_rank: float = 0.0625,
    patch_locations: int = 1,  # ("norm","ckpt_attn","ckpt_mlp")  
    early_stopping: bool = False,
    early_stopping_patience: int = 3,
    device_map: str = "auto",
):
    if patch_locations == 1:
        meft_patch_locations = ("ckpt_layer",)
    elif patch_locations == 2:
        meft_patch_locations = ("norm", "ckpt_attn", "ckpt_mlp",)
    else:
        raise ValueError("patch_locations must be 1 or 2")
    print("device_map: ",device_map)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        if using_meft:
            if using_compress:
                print(
                    f"Training LLaMA-2 model with MeftTrainer:\n"
                    f"base_model: {base_model}\n"
                    f"load_in_8bit: {load_in_8bit}\n"
                    f"data_path: {data_path}\n"
                    f"lora_weights_output_dir: {lora_weights_output_dir}\n"
                    f"hf_ckpt_output_dir: {hf_ckpt_output_dir}\n"
                    f"using_meft: {using_meft}\n"
                    f"lowrank_plus_quantization: {lowrank_plus_quantization}\n"
                    f"compress_rank: {compress_rank}\n"
                    f"patch_locations: {meft_patch_locations}\n"
                    f"early_stopping: {early_stopping}\n"
                    f"early_stopping_patience: {early_stopping_patience}\n"
                    f"batch_size: {batch_size}\n"
                    f"micro_batch_size: {micro_batch_size}\n"
                    f"gradient_accumulation_steps: {batch_size // micro_batch_size}\n"
                    f"num_epochs: {num_epochs}\n"
                    f"learning_rate: {learning_rate}\n"
                    f"cutoff_len: {cutoff_len}\n"
                    f"val_set_size: {val_set_size}\n"
                    f"warm_up_steps: {warm_up_steps}\n"
                    f"lora_r: {lora_r}\n"
                    f"lora_alpha: {lora_alpha}\n"
                    f"lora_dropout: {lora_dropout}\n"
                    f"lora_target_modules: {lora_target_modules}\n"
                    f"train_on_inputs: {train_on_inputs}\n"
                    f"add_eos_token: {add_eos_token}\n"
                    f"group_by_length: {group_by_length}\n"
                    f"wandb_project: {wandb_project}\n"
                    f"wandb_run_name: {wandb_run_name}\n"
                    f"wandb_watch: {wandb_watch}\n"
                    f"wandb_log_model: {wandb_log_model}\n"
                    f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
                    f"prompt template: {prompt_template_name}\n"
                )
            else:
                print(
                    f"Training LLaMA-2 model with MeftTrainer:\n"
                    f"base_model: {base_model}\n"
                    f"load_in_8bit: {load_in_8bit}\n"
                    f"data_path: {data_path}\n"
                    f"lora_weights_output_dir: {lora_weights_output_dir}\n"
                    f"hf_ckpt_output_dir: {hf_ckpt_output_dir}\n"
                    f"using_meft: {using_meft}\n"
                    f"compress_rank: no compress\n"
                    f"patch_locations: {meft_patch_locations}\n"
                    f"early_stopping: {early_stopping}\n"
                    f"early_stopping_patience: {early_stopping_patience}\n"
                    f"batch_size: {batch_size}\n"
                    f"micro_batch_size: {micro_batch_size}\n"
                    f"gradient_accumulation_steps: {batch_size // micro_batch_size}\n"
                    f"num_epochs: {num_epochs}\n"
                    f"learning_rate: {learning_rate}\n"
                    f"cutoff_len: {cutoff_len}\n"
                    f"val_set_size: {val_set_size}\n"
                    f"warm_up_steps: {warm_up_steps}\n"
                    f"lora_r: {lora_r}\n"
                    f"lora_alpha: {lora_alpha}\n"
                    f"lora_dropout: {lora_dropout}\n"
                    f"lora_target_modules: {lora_target_modules}\n"
                    f"train_on_inputs: {train_on_inputs}\n"
                    f"add_eos_token: {add_eos_token}\n"
                    f"group_by_length: {group_by_length}\n"
                    f"wandb_project: {wandb_project}\n"
                    f"wandb_run_name: {wandb_run_name}\n"
                    f"wandb_watch: {wandb_watch}\n"
                    f"wandb_log_model: {wandb_log_model}\n"
                    f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
                    f"prompt template: {prompt_template_name}\n"
                )
        else:
            print(
                f"Training LLaMA-2 model with original Trainer:\n"
                f"base_model: {base_model}\n"
                f"load_in_8bit: {load_in_8bit}\n"
                f"data_path: {data_path}\n"
                f"lora_weights_output_dir: {lora_weights_output_dir}\n"
                f"hf_ckpt_output_dir: {hf_ckpt_output_dir}\n"
                f"using_meft: {using_meft}\n"
                f"early_stopping: {early_stopping}\n"
                f"early_stopping_patience: {early_stopping_patience}\n"
                f"batch_size: {batch_size}\n"
                f"micro_batch_size: {micro_batch_size}\n"
                f"gradient_accumulation_steps: {batch_size // micro_batch_size}\n"
                f"num_epochs: {num_epochs}\n"
                f"learning_rate: {learning_rate}\n"
                f"cutoff_len: {cutoff_len}\n"
                f"val_set_size: {val_set_size}\n"
                f"warm_up_steps: {warm_up_steps}\n"
                f"lora_r: {lora_r}\n"
                f"lora_alpha: {lora_alpha}\n"
                f"lora_dropout: {lora_dropout}\n"
                f"lora_target_modules: {lora_target_modules}\n"
                f"train_on_inputs: {train_on_inputs}\n"
                f"add_eos_token: {add_eos_token}\n"
                f"group_by_length: {group_by_length}\n"
                f"wandb_project: {wandb_project}\n"
                f"wandb_run_name: {wandb_run_name}\n"
                f"wandb_watch: {wandb_watch}\n"
                f"wandb_log_model: {wandb_log_model}\n"
                f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
                f"prompt template: {prompt_template_name}\n"
            )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    prompter = Prompter(prompt_template_name)

    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    
    ddp=False
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_in_8bit:
        print("Loading model in 8-bit mode")
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
        print("Loading model without quantization_config")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            # llm_int8_skip_modules=None,
        )
    
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

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

    def compute_metrics(eval_pred):
        """计算验证集准确率：排除 labels=-100 的位置（这些位置不参与损失计算）"""
        import numpy as np
        from sklearn.metrics import accuracy_score
        
        # 解析模型预测结果：(logits, labels)
        logits, labels = eval_pred
        
        # 从 logits 中获取预测的 token（取概率最大的索引）
        predictions = np.argmax(logits, axis=-1)
        
        # 过滤掉 labels=-100 的位置（这些是输入部分，不参与损失计算）
        mask = labels != -100
        valid_predictions = predictions[mask]
        valid_labels = labels[mask]
        
        # 计算准确率
        accuracy = accuracy_score(valid_labels, valid_predictions)
        return {"accuracy": accuracy}

    if load_in_8bit:
        print("Loading peft model in 8-bit mode")
        model = prepare_model_for_kbit_training(model)
    else:
        pass

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if using_meft:
        print("Using MeftTrainer")
        
        trainer = MeftTrainer[SFTTrainer](
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=SFTConfig(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warm_up_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=1e-2,
                lr_scheduler_type="cosine",
                bf16=True,
                bf16_full_eval=True,
                use_liger_kernel=True,
                logging_steps=10,
                optim="adamw_torch",
                eval_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=100 if val_set_size > 0 else None,
                save_steps=200,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                output_dir=lora_weights_output_dir,
                save_total_limit=3,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            # data_collator=transformers.DataCollatorForSeq2Seq(
            #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            # ),
            # compute_metrics=compute_metrics,
            meft_config=MeftConfig(
                patch_locations=meft_patch_locations, # ("norm", "ckpt_attn", "ckpt_mlp",), # patch_locations=("ckpt_layer",),   
                compress_kwargs={"rank": compress_rank,
                                 "method": "rqb",
                                 "lowrank_plus_quantization": lowrank_plus_quantization,
                                 } if using_compress else None,
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping else None,
        )
    else:
        print("Using original Trainer")

        # model.gradient_checkpointing_enable()   # 开启梯度检查点
        
        trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warm_up_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            bf16_full_eval=True,
            use_liger_kernel=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=200,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=lora_weights_output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            # gradient_checkpointing=True,   # ← 新增这一行
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping else None,
    )
        
        
    model.config.use_cache = False
    
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    # # 这里不要再对“普通 Trainer 分支”的模型做 compile 了
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     if using_meft:
    #         # 只在 MeftTrainer 分支上使用 compile，避免和 checkpoint 冲突
    #         model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

   # save the lora adapter
    print("saving lora adapter to ", lora_weights_output_dir)
    model.save_pretrained(lora_weights_output_dir)

    # # merge weights - new merging method from peft
    # print("merging weights")
    # merged_model = model.merge_and_unload()
    # print("saving merged model to ",hf_ckpt_output_dir)
    #  # save the merged model
    # merged_model.save_pretrained(hf_ckpt_output_dir)
    # print("saving tokenizer to ",hf_ckpt_output_dir)
    # tokenizer.save_pretrained(hf_ckpt_output_dir)
    

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model with MEFT")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model")
    parser.add_argument("--base_model", "-m",type=str, default="meta-llama/Llama-2-7b-hf", help="Base model name or path")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Whether to load the model in 8-bit precision")
    parser.add_argument("--data_path", "-d",type=str, default="yahma/alpaca-cleaned", help="Path to training data")
    parser.add_argument("--lora_weights_output_dir",type=str, required=True, help="Directory to save LoRA weights")
    parser.add_argument("--hf_ckpt_output_dir",type=str, help="Directory to save Hugging Face checkpoint")
    
    parser.add_argument("--warm_up_steps", "-wus",type=int, default=100, help="Number of warm-up steps")
    parser.add_argument("--early_stopping", default=False, help="Whether to use early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--batch_size", "-bs",type=int, default=128, help="Batch size for training")
    parser.add_argument("--micro_batch_size", "-mbs",type=int, default=16, help="Micro batch size for gradient accumulation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", "-lr",type=float, default=3e-5, help="Learning rate for optimizer")
    parser.add_argument("--cutoff_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--val_set_size", type=int, default=2000, help="Size of validation set")
    
    parser.add_argument("--using_meft", action="store_true", default=False, help="Whether to use MEFT")
    parser.add_argument("--using_compress",action="store_true",default=False, help="Whether to use compression in MEFT")
    parser.add_argument("--lowrank_plus_quantization", action="store_true", default=False, help="Whether to use low-rank plus quantization in MEFT")
    parser.add_argument("--compress_rank", type=float, default=0.0625, help="Compression rank for MEFT")
    parser.add_argument("--patch_locations", type=int, default=1, help="Locations to apply MEFT patches, e.g. 2 means ('norm', 'ckpt_attn', 'ckpt_mlp') 1 means ('ckpt_layer',)")

    
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=["q_proj", "v_proj"], help="Target modules for LoRA")
    
    parser.add_argument("--wandb_project", default="", help="Wandb project name for logging")
    parser.add_argument("--wandb_run_name", type=str, default="", help="Name of wandb run")
    parser.add_argument("--wandb_watch", type=str, default="", help="Wandb watch setting (false | gradients | all)")
    parser.add_argument("--wandb_log_model", type=str, default="", help="Wandb log model setting (false | true)")
    
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume training from checkpoint")
    parser.add_argument("--group_by_length", default=False, help="Whether to group sequences by length")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca", help="Prompt template name")
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    fire.Fire(train(
        base_model=args.base_model,
        load_in_8bit=args.load_in_8bit,
        data_path=args.data_path,
        lora_weights_output_dir=args.lora_weights_output_dir,
        hf_ckpt_output_dir=args.hf_ckpt_output_dir,
        
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        warm_up_steps=args.warm_up_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        patch_locations=args.patch_locations,
        using_meft=args.using_meft,
        using_compress=args.using_compress,
        lowrank_plus_quantization=args.lowrank_plus_quantization,
        compress_rank=args.compress_rank,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_watch=args.wandb_watch,
        wandb_log_model=args.wandb_log_model,
        resume_from_checkpoint=args.resume_from_checkpoint,
        group_by_length=args.group_by_length,
        prompt_template_name=args.prompt_template_name,
        device_map=args.device_map
        
    ))
    
# python finetune_meft.py --lora_weights_output_dir ./output/lora_weights/meft_layer_no_compress_results_vgpu48 --hf_ckpt_output_dir ./output/hf_ckpt/meft_layer_no_compress_hf_ckpt_vgpu48 
# --using_meft True 
# --compress_rank 1 
# --patch_locations ("ckpt_layer",)
# --early_stopping True 
# --early_stopping_patience 3 
# --batch_size 128 
# --micro_batch_size 16 
# --num_epochs 3 
# --learning_rate 3e-5 
# --cutoff_len 256 
# --val_set_size 4000 
# --warm_up_steps 100 
# --lora_r 8 
# --lora_alpha 16 
# --lora_dropout 0.05 
# --lora_target_modules q_proj v_proj 
# --wandb_project "" 
# --wandb_run_name "" 
# --wandb_watch "" 
# --wandb_log_model "" 
