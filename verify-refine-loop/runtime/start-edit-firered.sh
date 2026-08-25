#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNTIME_DIR}/../.." && pwd)"

GPU="${GPU:-0}"
PORT="${PORT:-9101}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bf16}"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-${REPO_ROOT}/models}"
LORA_PATH="${LORA_PATH:-}"

cd "${RUNTIME_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ARGS=(
  --backend firered11
  --local-models-root "${LOCAL_MODELS_ROOT}"
  --device cuda
  --dtype "${DTYPE}"
  --host "${HOST}"
  --port "${PORT}"
)
if [[ -n "${LORA_PATH}" ]]; then
  ARGS+=(--lora-path "${LORA_PATH}")
fi

exec python edit_server.py "${ARGS[@]}"
