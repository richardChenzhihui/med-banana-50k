#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to your Qwen3.5-9B checkpoint directory}"
DATASET_JSON="${DATASET_JSON:-${FINETUNE_ROOT}/data/vlm_prompt_generation_dataset.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${FINETUNE_ROOT}/output/vlm_prompt_generation}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ ! -f "${DATASET_JSON}" ]]; then
  echo "missing dataset json: ${DATASET_JSON}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

MASTER_PORT="${MASTER_PORT:-29505}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
WANDB_PROJECT="${WANDB_PROJECT:-med-banana}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${BASE_MODEL}" \
  --dataset "${DATASET_JSON}" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --attn_impl flash_attn \
  --learning_rate 1e-4 \
  --target_modules all-linear \
  --modules_to_save embed_tokens lm_head \
  --freeze_vit false \
  --freeze_aligner false \
  --packing true \
  --padding_free true \
  --gradient_checkpointing \
  --vit_gradient_checkpointing true \
  --gradient_accumulation_steps 8 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --max_length 2048 \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.05 \
  --deepspeed zero2 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --report_to "${REPORT_TO:-none}"

echo "Training complete: ${OUTPUT_DIR}"
