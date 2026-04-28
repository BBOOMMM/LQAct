#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=(
    "vit-base"
    "vit-large"
    "vit-huge"
)

DATASETS=(
    "tiny-imagenet"
    "cifar100"
    "food101"
)

RANKS=(
  0.03125
  0.0625
  0.125
)

EPOCHS=50
LR=5e-4
WD=1e-3
PER_DEVICE_BATCH_SIZE=512
GRADIENT_ACCUMULATION_STEPS=1
LORA_R=64
ENERGY_RATIO=0.1


for MODEL in "${MODEL_NAME[@]}"; do
    for DATASET in "${DATASETS[@]}"; do

        # lora
        python LQAct-general.py \
        --model_name "${MODEL}" --dataset_name "${DATASET}" \
        --output_dir "./outputs_GEN/${DATASET}_${MODEL}_LoRA" \
        --wandb_project_name "GEN-${DATASET}" --wandb_run_name "${MODEL}-LoRA" \
        --vanilla_train \
        --per_device_train_batch_size 128 --gradient_accumulation_steps 4 \
        --learning_rate ${LR} --weight_decay ${WD} --epochs ${EPOCHS}

        for RANK in "${RANKS[@]}"; do

            # loract
            python LQAct-general.py \
            --model_name "${MODEL}" --dataset_name "${DATASET}" \
            --output_dir "./outputs_GEN/${DATASET}_${MODEL}_loract${RANK}" \
            --wandb_project_name "GEN-${DATASET}" --wandb_run_name "${MODEL}_loract${RANK}" \
            --rank_ratio ${RANK} \
            --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --learning_rate ${LR} --weight_decay ${WD} --epochs ${EPOCHS}

            # eras-sa
            python LQAct-general.py \
            --model_name "${MODEL}" --dataset_name "${DATASET}" \
            --output_dir "./outputs_GEN/${DATASET}_${MODEL}_dk${RANK}_er${ENERGY_RATIO}" \
            --wandb_project_name "GEN-${DATASET}" --wandb_run_name "${MODEL}_dk${RANK}_er${ENERGY_RATIO}" \
            --rank_ratio ${RANK} --dynamic_rank --energy_ratio ${ENERGY_RATIO} \
            --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --learning_rate ${LR} --weight_decay ${WD} --epochs ${EPOCHS}


            # eras-eb
            python LQAct-general.py \
            --model_name "${MODEL}" --dataset_name "${DATASET}" \
            --output_dir "./outputs_GEN/${DATASET}_${MODEL}_dk${RANK}_es" \
            --wandb_project_name "GEN-${DATASET}" --wandb_run_name "${MODEL}_dk${RANK}_es" \
            --rank_ratio ${RANK} --dynamic_rank --energy_search \
            --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --learning_rate ${LR} --weight_decay ${WD} --epochs ${EPOCHS}

        done
    done
done