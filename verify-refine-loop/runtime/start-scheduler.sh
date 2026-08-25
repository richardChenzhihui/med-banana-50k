#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNTIME_DIR}/../.." && pwd)"

CONFIG="${CONFIG:-${RUNTIME_DIR}/config.example.json}"
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/inference/data/addition}"
BACKENDS="${BACKENDS:-qwen2511}"
MAX_ROUNDS="${MAX_ROUNDS:-}"
REFINE_BUDGET="${REFINE_BUDGET:-}"
DRY_RUN="${DRY_RUN:-0}"
RETRY_FAILED="${RETRY_FAILED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${RUNTIME_DIR}"

ARGS=(
  --config "${CONFIG}"
  --input-dir "${INPUT_DIR}"
  --backends "${BACKENDS}"
)
if [[ -n "${MAX_ROUNDS}" ]]; then
  ARGS+=(--max-rounds "${MAX_ROUNDS}")
fi
if [[ -n "${REFINE_BUDGET}" ]]; then
  ARGS+=(--refine-budget "${REFINE_BUDGET}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "${RETRY_FAILED}" == "1" ]]; then
  ARGS+=(--retry-failed)
fi

exec "${PYTHON_BIN}" scheduler.py "${ARGS[@]}"
