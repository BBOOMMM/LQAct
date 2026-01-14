import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import huggingface_hub
print("login to huggingface_hub")
huggingface_hub.login(token="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")  # Replace with your actual token
print("login success")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["MKL_SERVICE_FORCE_INTEL"]="1"
# os.environ["OMP_NUM_THREADS"] = "8"  # 控制MKL使用的线程数
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["BLAS"] = "MKL"
# os.environ["USE_CUSOLVER"] = "1"

import numpy as np
import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoFeatureExtractor, SwinForImageClassification,
    ViTImageProcessor, ViTForImageClassification,
    Trainer, TrainingArguments, EvalPrediction,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from meft import MeftConfig, MeftTrainer
import meft

from get_rank.vit import get_vit_rank, get_vit_rank_ratio

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


model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    num_labels=100,
    ignore_mismatched_sizes=True,
    # device_map="auto",
    device_map="cuda:0",
)

dataset = load_dataset("cifar100")

def transform(examples):
    inputs = processor(
        images=[x.convert("RGB") for x in examples["img"]],
        return_tensors="pt"
    )
    return {
        "pixel_values": inputs.pixel_values,
        "label": examples["fine_label"]
    }

dataset = dataset.with_transform(transform)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
    modules_to_save=["classifier"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


_, rank_dict = get_vit_rank(model, dataset["train"], batch_size=512, patch_locations=2)
# _, rank_dict = get_vit_rank_ratio(model, dataset["train"], batch_size=512, patch_locations=2, base_ratio=1.0/16.0)


num_layers = len(rank_dict)
hidden_size = model.vit.config.hidden_size
total_rank = 0
for layer_name, io_dict in rank_dict.items():
    layer_rank = 0
    for io_type, rank in io_dict.items():
        layer_rank += rank
    total_rank += layer_rank

print(f"总rank: {total_rank}")
print(f'1/16 总Rank :{hidden_size * num_layers / 16}')
print(f'1/8 总Rank :{hidden_size * num_layers / 8}')

breakpoint()


def compute_metrics(eval_pred: EvalPrediction):
    evaluation = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluation.compute(predictions=predictions, references=labels)


trainer = MeftTrainer[Trainer](
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=512,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=1e-3,
        weight_decay=1e-2,
        # warmup_ratio=0.1,    # 新加
        lr_scheduler_type="cosine",
        bf16=True,
        bf16_full_eval=True,
        # deepspeed={
        #     "train_batch_size": "auto",
        #     "gradient_accumulation_steps": "auto",
        #     "zero_optimization": {
        #         "stage": 1,
        #     },
        # },
        use_liger_kernel=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    ),
    data_collator=None,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    meft_config=MeftConfig(
        patch_locations=(
            "norm",
            # "attn_in",
            # "attn_out",
            # "mlp_in",
            # "mlp_out",
            # "act_fn",
            "ckpt_attn",
            "ckpt_mlp",
            # "ckpt_layer",
        ),
        compress_kwargs={
            # "rank": 0.0625,
            "rank": rank_dict,
            # "niter": 1,
        },
        # compress_workers=2,
    ),
)


trainer.train()


# trainer.evaluate()
test_results = trainer.evaluate()
print(f"测试集top-1准确率: {test_results['eval_accuracy']:.4f}")
