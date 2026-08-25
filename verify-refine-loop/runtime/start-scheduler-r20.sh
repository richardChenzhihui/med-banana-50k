#!/usr/bin/env bash
set -euo pipefail

# Same logic as start-scheduler.sh, but defaults to CONFIG=config.r20.json:
#   - edit servers on 9011/9111 (shared with r10 via start-edit-*-r10.sh; do not run this and
#     the r10 scheduler over the same cases at the same time)
#   - writes to output-r20/; max_rounds=25, refine_budget=20 (rest as config.r10.json)
#   - shares the vLLM refiner/verifier on 8001/8002 with the main run
#
# Usage:
#   bash start-scheduler-r20.sh
# Optional: INPUT_SET=one_case BACKENDS=qwen2511 DRY_RUN=1
#   RETRY_FAILED=1 INPUT_SET=removal  - re-run only the cases marked failed in
#                                       output-r20/**/progress.json

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-$RUNTIME_DIR/config.r20.json}"

exec bash "$RUNTIME_DIR/start-scheduler.sh"
