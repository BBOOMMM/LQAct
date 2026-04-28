import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import huggingface_hub
print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")

import json
import numpy as np
import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoFeatureExtractor, SwinForImageClassification,
    ViTImageProcessor, ViTForImageClassification,
    Trainer, TrainingArguments, EvalPrediction,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import time

from meft import MeftConfig, MeftTrainer
import meft

from get_rank.vit import get_vit_rank, get_vit_rank_ratio, get_vit_rank_binary_search_energy_ratio, get_vit_rank_ratio_gentle

from fgvc_datasets_setup.loader import _DATASET_NUM_LABELS

from fgvc_datasets_setup import loader as data_loader

import argparse


def append_local_result(output_dir: str, args, test_results: dict, train_total_time: float, peak_mem_gb: float | None):
    os.makedirs(output_dir, exist_ok=True)
    record = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "compress_method": args.compress_method,
        "quant_method": args.quant_method,
        "rank_ratio": args.rank_ratio,
        "dynamic_rank": args.dynamic_rank,
        "energy_ratio": args.energy_ratio,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "wandb_run_name": args.wandb_run_name,
        "output_dir": args.output_dir,
        "train_total_time_s": train_total_time,
        "train_peak_gpu_mem_gb": peak_mem_gb,
        "eval_accuracy": test_results.get("eval_accuracy"),
        "raw_test_results": test_results,
    }
    result_file = os.path.join(output_dir, "results_local.jsonl")
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--model_name', type=str, default='vit-base', help='pretrained model name or path')

    # training mode
    parser.add_argument('--vanilla_train', action='store_true', help='use uncompressed vanilla lora fine-tuning')
    parser.add_argument('--rank_ratio', type=float, default=0.125, help='compress rank ratio')
    parser.add_argument('--dynamic_rank', action='store_true', help='use dynamic rank')
    parser.add_argument('--energy_ratio', type=float, default=0.5, help='energy retention ratio during dynamic rank')
    parser.add_argument('--energy_search', action='store_true', help='use binary search to determine energy_ratio')
    parser.add_argument('--patch_locations', type=int, default=2, help='patch locations used by the modeleft, 1 for only ckpt_layer, 2 for norm+ckpt_attn+ckpt_mlp, 3 for norm+attn_in+attn_out+mlp_in+mlp_out')
    parser.add_argument(
        '--compress_method',
        type=str,
        default='dynamic_fixed_rank_dynamic_quantization',
        choices=['dynamic_fixed_rank_dynamic_quantization', 'rqb', 'energy_rqb', 'probing_rqb', 'tsvd', 'rsvd', 'nyssvd'],
        help='activation compression method passed into MeftConfig'
    )
    parser.add_argument(
        '--quant_method',
        type=str,
        default='1bit_pergroupchannel',
        choices=['1bit_pertensor', '1bit_pergroupchannel', 'ternary', 'two_bit_group'],
        help='residual quantization method used when compress_method enables lowrank plus quantization'
    )

    # dataset
    parser.add_argument('--dataset_name', type=str, default='CUB', help='dataset name')
    parser.add_argument('--data_dir', type=str, default='./datasets/fgvc', help='dataset root directory')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=512, help='per-device train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='model and LoRA parameters save path')
    parser.add_argument('--wandb_project_name', type=str, default='wandb', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='wandb', help='wandb run name')

    # lora parameters
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout probability')

    args = parser.parse_args()
    return args
args = parse_args()
print("Training parameters:", args)


import random
from transformers import set_seed as hf_set_seed
seed = args.seed
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
hf_set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import wandb
wandb.init(
    project=args.wandb_project_name,
    name=args.wandb_run_name,
    config=vars(args),
)


num_labels = _DATASET_NUM_LABELS[args.dataset_name]


if args.model_name == 'vit-base':
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        device_map="cuda:0",
    )
elif args.model_name == 'vit-large':
    model_name = "google/vit-large-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        device_map="cuda:0",
    )
elif args.model_name == 'vit-huge':
    model_name = "google/vit-huge-patch14-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        device_map="cuda:0",
    )
    

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


def get_datasets(args, processor):
    print("Loading training data...")
    train_loader = data_loader.construct_train_dataset(args, processor)
    print("Loading validation data...")
    val_loader = data_loader.construct_val_dataset(args, processor)
    print("Loading test data...")
    test_loader = data_loader.construct_test_dataset(args, processor)
    return train_loader,  val_loader, test_loader


train_dataset, val_dataset, test_dataset = get_datasets(args, processor)


if args.patch_locations == 1:
    meft_patch_locations = ("ckpt_layer",)
elif args.patch_locations == 2:
    meft_patch_locations = ("norm", "ckpt_attn", "ckpt_mlp",)
elif args.patch_locations == 3:
    meft_patch_locations = ("norm", "attn_in", "attn_out", "mlp_in", "mlp_out",)
else:
    raise ValueError("Unsupported patch_locations number.")


if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

train_start = time.time()


if args.dynamic_rank:
    if args.energy_search:
        activations, rank_dict = get_vit_rank_binary_search_energy_ratio(model, val_dataset, batch_size=args.per_device_train_batch_size, patch_locations=args.patch_locations, rank_ratio=args.rank_ratio)
        del activations
        print(rank_dict)

        num_layers = len(rank_dict)
        hidden_size = model.vit.config.hidden_size
        total_rank = 0
        for layer_name, io_dict in rank_dict.items():
            layer_rank = 0
            for io_type, rank in io_dict.items():
                layer_rank += rank
            total_rank += layer_rank
    else:
        activations, rank_dict = get_vit_rank_ratio(model, val_dataset, batch_size=args.per_device_train_batch_size, patch_locations=args.patch_locations, base_ratio=args.rank_ratio, energy_ratio=args.energy_ratio)
        del activations
        print(rank_dict)

        num_layers = len(rank_dict)
        hidden_size = model.vit.config.hidden_size
        total_rank = 0
        for layer_name, io_dict in rank_dict.items():
            layer_rank = 0
            for io_type, rank in io_dict.items():
                layer_rank += rank
            total_rank += layer_rank



def compute_metrics(eval_pred: EvalPrediction):
    evaluation = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluation.compute(predictions=predictions, references=labels)

if args.vanilla_train:
        trainer = Trainer(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            per_device_eval_batch_size = 32,
            num_train_epochs = args.epochs,
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_torch",
            bf16 = True,
            bf16_full_eval = True,
            use_liger_kernel = True,
            logging_steps = 1,
            report_to = ["wandb"],
            run_name=args.wandb_run_name,
            remove_unused_columns = False,
            label_names = ["labels"],
            eval_strategy = "epoch",
            save_strategy = "epoch",
            save_total_limit = 1,
            output_dir = args.output_dir,
        ),
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
else:
    trainer = MeftTrainer[Trainer](
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            per_device_eval_batch_size = args.per_device_train_batch_size,
            num_train_epochs = args.epochs,
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_torch",
            bf16 = True,
            bf16_full_eval = True,
            use_liger_kernel = True,
            logging_steps = 1,
            report_to = ["wandb"],
            run_name=args.wandb_run_name,
            remove_unused_columns = False,
            label_names = ["labels"],
            eval_strategy = "epoch",
            save_strategy = "epoch",
            save_total_limit = 1,
            output_dir = args.output_dir,
        ),
        data_collator=None,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        meft_config=MeftConfig(
            patch_locations=meft_patch_locations,
            compress_method=args.compress_method,
            compress_kwargs={
                "rank": rank_dict if args.dynamic_rank else args.rank_ratio,
            },
            quant_method=args.quant_method,
        ),
    )


train_output = trainer.train()

if torch.cuda.is_available():
    torch.cuda.synchronize()
train_total_time = time.time() - train_start

print(f"[Train] total time: {train_total_time:.2f}s")

if torch.cuda.is_available():
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    print(f"[Train] peak GPU memory: {peak_mem_gb:.2f} GB")
    wandb.log(
        {
            "train_total_time_s": train_total_time,
            "train_peak_gpu_mem_gb": peak_mem_gb,
        },
        step=train_output.global_step if hasattr(train_output, "global_step") else None,
    )
else:
    peak_mem_gb = None


test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"test data top-1 accuracy: {test_results['eval_accuracy']:.4f}")
append_local_result(args.output_dir, args, test_results, train_total_time, peak_mem_gb)

wandb.summary["test_accuracy"] = test_results["eval_accuracy"]
