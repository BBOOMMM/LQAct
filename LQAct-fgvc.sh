#!/bin/bash

set -euo pipefail

export http_proxy=http://180.209.6.222:3128
export https_proxy=http://180.209.6.222:3128
export HTTP_PROXY=http://180.209.6.222:3128
export HTTPS_PROXY=http://180.209.6.222:3128
export all_proxy=http://180.209.6.222:3128
export ALL_PROXY=http://180.209.6.222:3128

GPUS=(1 2)
GPU_COUNT=${#GPUS[@]}

MODEL_NAME=(
    "vit-base"
    # "vit-large"
)

DATASETS=(
#   "CUB"
#   "OxfordFlowers"
#   "StanfordCars"
  "StanfordDogs"
#   "nabirds"
)

RANKS=(
    0.0625
)

DATA_DIR="/data0/chenxy/code/LoRAct/datasets/fgvc"
PATCH_LOCATIONS=2
EPOCHS=30
LR=5e-4
WD=1e-3
PER_DEVICE_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
LORA_R=64
COMPRESS_METHOD="dynamic_fixed_rank_dynamic_quantization"
QUANT_METHOD="1bit_pergroupchannel"
# QUANT_METHOD="ternary"

TASKS=()
task_idx=0
for MODEL in "${MODEL_NAME[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for RANK in "${RANKS[@]}"; do
            gpu_idx=$((task_idx % GPU_COUNT))
            TASKS[gpu_idx]+="${MODEL}|${DATASET}|${RANK}"$'\n'
            task_idx=$((task_idx + 1))
        done
    done
done

for gpu_idx in "${!GPUS[@]}"; do
    CURRENT_GPU=${GPUS[$gpu_idx]}
    (
        while IFS='|' read -r MODEL DATASET RANK; do
            if [[ -z "${MODEL:-}" ]]; then
                continue
            fi

            echo "正在显卡 ${CURRENT_GPU} 上启动任务: ${MODEL}, ${DATASET}, ${RANK}"

            CUDA_VISIBLE_DEVICES=$CURRENT_GPU python LQAct-fgvc.py \
                --model_name "${MODEL}" \
                --dataset_name "${DATASET}" \
                --data_dir "${DATA_DIR}" \
                --patch_locations ${PATCH_LOCATIONS} \
                --output_dir "./outputs_FGVC_test/${DATASET}_${MODEL}_loract${RANK}" \
                --wandb_project_name "FGVC-${DATASET}-${COMPRESS_METHOD}" \
                --wandb_run_name "${MODEL}_LQAct${RANK}-${QUANT_METHOD}" \
                --rank_ratio ${RANK} \
                --compress_method "${COMPRESS_METHOD}" \
                --quant_method "${QUANT_METHOD}" \
                --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --learning_rate ${LR} \
                --weight_decay ${WD} \
                --epochs ${EPOCHS}
        done <<< "${TASKS[$gpu_idx]-}"
    ) &
done

wait
echo "所有任务已完成。"
