#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to your Qwen3.5-9B checkpoint directory}"
ADAPTER_PATH="${ADAPTER_PATH:?set ADAPTER_PATH to verifier LoRA checkpoint}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:?set MERGED_MODEL_PATH to output merged model directory}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "${MERGED_MODEL_PATH}" ]]; then
  echo "Merging LoRA adapter from ${ADAPTER_PATH} to ${MERGED_MODEL_PATH} ..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" swift export \
    --model "${BASE_MODEL}" \
    --adapters "${ADAPTER_PATH}" \
    --merge_lora true \
    --output_dir "${MERGED_MODEL_PATH}" \
    --torch_dtype bfloat16
else
  echo "Merged model found at ${MERGED_MODEL_PATH}, skipping export."
fi

echo "Merged model: ${MERGED_MODEL_PATH}"
