#!/usr/bin/env bash
set -euo pipefail
# 10-sample smoke training: a tiny dataset just to prove the pipeline runs end to end.
# Requires data/qwen-edit-medbanana-10, produced by
# flymayai-script/prepare_medbanana_qwen_edit_dataset.py --sizes 10
#
# GPUs: same defaults as the larger runs; override with CUDA_VISIBLE_DEVICES or
# MEDBANANA_CUDA_DEVICES.
_DEFAULT_CUDA_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${_DEFAULT_CUDA_DEVICES}}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export DATASET_REPEAT="${DATASET_REPEAT:-1}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_medbanana_edit.sh" qwen 10
