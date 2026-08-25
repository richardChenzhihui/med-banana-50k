#!/usr/bin/env bash
set -euo pipefail

# Same logic as start-scheduler.sh, but defaults to CONFIG=config.r10.json:
#   - edit servers on 9011/9111 (launch them with start-edit-*-r10.sh)
#   - writes to output-r10/; max_rounds=15, refine_budget=10 (rest as config.example.json)
#   - shares the vLLM refiner/verifier on 8001/8002 with the main run
#
# Usage:
#   bash start-scheduler-r10.sh
# Optional: INPUT_SET=one_case BACKENDS=qwen2511 DRY_RUN=1

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-$RUNTIME_DIR/config.r10.json}"

exec bash "$RUNTIME_DIR/start-scheduler.sh"
