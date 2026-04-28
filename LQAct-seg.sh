#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

GPU=${GPU:-1}

MODEL_NAME=(
    "dinov2-base-seg"
)

DATASETS=(
    "voc2012-seg"
)

RANKS=(
    0.03125
    0.0625
    0.125
)

DATA_DIR="/data1/chenxy/datasets"
PATCH_LOCATIONS=2
EPOCHS=${EPOCHS:-30}
LR=5e-4
WD=1e-3
PER_DEVICE_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
ENERGY_RATIO=0.1
COMPRESS_METHOD="rqd"
QUANT_METHOD="1bit_pergroupchannel"

result_epoch() {
    local output_dir="$1"
    "${PYTHON_BIN}" - "$output_dir" <<'PY'
import json
import os
import sys

path = os.path.join(sys.argv[1], "results_local.jsonl")
if not os.path.exists(path):
    print("missing")
    raise SystemExit(0)

with open(path, "r", encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh if line.strip()]
if not rows:
    print("missing")
    raise SystemExit(0)

raw = rows[-1].get("raw_test_results") or {}
epoch = raw.get("epoch", rows[-1].get("epochs", 0))
print(epoch)
PY
}

run_variant() {
    local variant_id="$1"
    local output_dir="$2"
    shift 2

    local existing_epoch
    existing_epoch=$(result_epoch "${output_dir}")
    if [[ "${existing_epoch}" != "missing" ]] && "${PYTHON_BIN}" - "${existing_epoch}" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) >= 30.0 else 1)
PY
    then
        echo "Skipping completed variant ${variant_id}"
        return 0
    fi

    if [[ "${existing_epoch}" != "missing" ]]; then
        echo "Removing stale result for ${variant_id} with epoch=${existing_epoch}"
        rm -rf "${output_dir}"
    fi

    if CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" LQAct-vision.py "$@"; then
        return 0
    fi

    echo "Crashed segmentation variant ${variant_id}" >&2
    return 1
}

for MODEL in "${MODEL_NAME[@]}"; do
    for DATASET in "${DATASETS[@]}"; do

        run_variant "seg:${DATASET}:${MODEL}:LoRA" "./outputs_seg/${DATASET}_${MODEL}_LoRA" \
            --task_type semantic_segmentation \
            --model_name "${MODEL}" \
            --dataset_name "${DATASET}" \
            --data_dir "${DATA_DIR}" \
            --patch_locations ${PATCH_LOCATIONS} \
            --output_dir "./outputs_seg/${DATASET}_${MODEL}_LoRA" \
            --wandb_project_name "SEG-${DATASET}-${COMPRESS_METHOD}" \
            --wandb_run_name "${MODEL}-LoRA" \
            --vanilla_train \
            --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --learning_rate ${LR} \
            --weight_decay ${WD} \
            --epochs ${EPOCHS}

        for RANK in "${RANKS[@]}"; do
            run_variant "seg:${DATASET}:${MODEL}:loract:${RANK}" "./outputs_seg/${DATASET}_${MODEL}_loract${RANK}" \
                --task_type semantic_segmentation \
                --model_name "${MODEL}" \
                --dataset_name "${DATASET}" \
                --data_dir "${DATA_DIR}" \
                --patch_locations ${PATCH_LOCATIONS} \
                --output_dir "./outputs_seg/${DATASET}_${MODEL}_loract${RANK}" \
                --wandb_project_name "SEG-${DATASET}-${COMPRESS_METHOD}" \
                --wandb_run_name "${MODEL}_loract${RANK}" \
                --rank_ratio ${RANK} \
                --compress_method "${COMPRESS_METHOD}" \
                --quant_method "${QUANT_METHOD}" \
                --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --learning_rate ${LR} \
                --weight_decay ${WD} \
                --epochs ${EPOCHS}

            run_variant "seg:${DATASET}:${MODEL}:eras-sa:${RANK}" "./outputs_seg/${DATASET}_${MODEL}_dk${RANK}_er${ENERGY_RATIO}" \
                --task_type semantic_segmentation \
                --model_name "${MODEL}" \
                --dataset_name "${DATASET}" \
                --data_dir "${DATA_DIR}" \
                --patch_locations ${PATCH_LOCATIONS} \
                --output_dir "./outputs_seg/${DATASET}_${MODEL}_dk${RANK}_er${ENERGY_RATIO}" \
                --wandb_project_name "SEG-${DATASET}-${COMPRESS_METHOD}" \
                --wandb_run_name "${MODEL}_ERAs-SA_${RANK}" \
                --rank_ratio ${RANK} \
                --dynamic_rank \
                --energy_ratio ${ENERGY_RATIO} \
                --compress_method "${COMPRESS_METHOD}" \
                --quant_method "${QUANT_METHOD}" \
                --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --learning_rate ${LR} \
                --weight_decay ${WD} \
                --epochs ${EPOCHS}

            run_variant "seg:${DATASET}:${MODEL}:eras-eb:${RANK}" "./outputs_seg/${DATASET}_${MODEL}_dk${RANK}_es" \
                --task_type semantic_segmentation \
                --model_name "${MODEL}" \
                --dataset_name "${DATASET}" \
                --data_dir "${DATA_DIR}" \
                --patch_locations ${PATCH_LOCATIONS} \
                --output_dir "./outputs_seg/${DATASET}_${MODEL}_dk${RANK}_es" \
                --wandb_project_name "SEG-${DATASET}-${COMPRESS_METHOD}" \
                --wandb_run_name "${MODEL}_ERAs-EB_${RANK}" \
                --rank_ratio ${RANK} \
                --dynamic_rank \
                --energy_search \
                --compress_method "${COMPRESS_METHOD}" \
                --quant_method "${QUANT_METHOD}" \
                --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --learning_rate ${LR} \
                --weight_decay ${WD} \
                --epochs ${EPOCHS}
        done
    done
done
