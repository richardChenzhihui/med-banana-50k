#!/usr/bin/env bash
set -euo pipefail
# FireRed: shares data/qwen-edit-medbanana-10 with the Qwen run; only the base model and the
# checkpoint subdirectory differ.
_DEFAULT_CUDA_DEVICES="0"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${_DEFAULT_CUDA_DEVICES}}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export DATASET_REPEAT="${DATASET_REPEAT:-1}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_medbanana_edit.sh" firered 10
