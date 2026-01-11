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
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from meft import MeftConfig, MeftTrainer
import meft

from get_rank.vit import get_vit_rank, get_vit_rank_ratio, get_vit_project_matrix

import json
import pickle

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


class RankProbeCallback(TrainerCallback):
    def __init__(self, dataset, batch_size: int = 256, patch_locations: int = 2, act_path: str = "vit_activations_epochs.pkl"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_locations = patch_locations
        self.act_path = act_path

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        epoch = int(state.epoch) if state.epoch is not None else -1

        # 这里会在当前 epoch 结束后用当前模型参数计算一次 rank
        activations, rank_dict = get_vit_rank(model, self.dataset, self.batch_size, self.patch_locations)
        print(f"\n[RankProbe] epoch {epoch} rank: {rank_dict}\n")

        # 如有需要，也可以把 rank_dict 存文件，便于后续分析
        # with open(f"vit_rank_epoch_{epoch}.json", "w") as f:
        #     json.dump(rank_dict, f)
        # record = {"epoch": epoch, "rank": rank_dict}
        # # 以追加模式写入，一行一个 JSON
        # with open(self.rank_path, "a", encoding="utf-8") as f:
        #     f.write(json.dumps(record) + "\n")
            
        # activations 追加到一个大 pkl 文件
        act_record = {"epoch": epoch, "activations": activations}
        with open(self.act_path, "ab") as f:  # 注意是 "ab"
            pickle.dump(act_record, f)

        return control


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

# _, rank_1 = get_vit_rank(model, dataset["train"], batch_size=512, patch_locations=2)


def compute_metrics(eval_pred: EvalPrediction):
    evaluation = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluation.compute(predictions=predictions, references=labels)


act_path="dynamic_k_save/vit_lora_activations_epochs.pkl"
os.makedirs("dynamic_k_save", exist_ok=True)

rank_callback = RankProbeCallback(dataset["train"], batch_size=16, patch_locations=2, act_path=act_path)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=128,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        learning_rate=5e-3,
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
    callbacks=[rank_callback],
)


# ====== 训练前，手动算一次 epoch=0 ======
activations, rank_dict = get_vit_rank(
    trainer.model, dataset["train"], batch_size=16, patch_locations=2
)
# print(f"\n[RankProbe] epoch 0 (before training) rank: {rank_dict}\n")

act_record = {"epoch": 0, "activations": activations}
with open(rank_callback.act_path, "ab") as f:
    pickle.dump(act_record, f)
# ======================================


trainer.train()

# _, rank_2 = get_vit_rank(model, dataset["train"], batch_size=512, patch_locations=2)

# print(f"训练前rank: {rank_1}")
# print(f"训练后rank: {rank_2}")

# trainer.evaluate()
test_results = trainer.evaluate()
print(f"测试集top-1准确率: {test_results['eval_accuracy']:.4f}")

breakpoint()
