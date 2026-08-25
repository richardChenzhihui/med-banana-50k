#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-9111}"
GPU="${GPU:-0}"
export PORT GPU
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start-edit-firered.sh"
