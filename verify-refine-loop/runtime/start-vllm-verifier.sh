#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNTIME_DIR}/../.." && pwd)"

GPU="${GPU:-0}"
PORT="${PORT:-8002}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MODEL="${MODEL:-${REPO_ROOT}/models/Qwen3.5-9B-prompt-verifier}"

export CUDA_VISIBLE_DEVICES="${GPU}"

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --disable-custom-all-reduce
