#!/usr/bin/env bash
set -euo pipefail
# MedBanana edit triplets -> DiffSynth train.py (upstream in DiffSynth-Studio).
# Data: MEDBANANA_TRAIN_ROOT/data/<subdir> with images/, control_images/, stem.txt layout.
# Default subdir: qwen-edit-medbanana-{10|1000|5000|10000} (override with DIFFSYNTH_MED_EDIT_DATA_SUBDIR).
# Checkpoints: DIFFSYNTH_STUDIO_ROOT/models/train/medbanana/{qwen|firered}_edit_medbanana_*
#
# Key envs:
#   MEDBANANA_TRAIN_ROOT, DIFFSYNTH_STUDIO_ROOT, DIFFSYNTH_MED_EDIT_DATA_SUBDIR
#   LOCAL_MODELS_ROOT (default <repo>/models), DIFFSYNTH_SKIP_DOWNLOAD (default true)
#   LR, NUM_EPOCHS, START_EPOCH, WEIGHT_DECAY, DATASET_REPEAT, MAX_PIXELS, LORA_RANK, DATASET_NUM_WORKERS,
#   GRADIENT_ACCUMULATION_STEPS, SAVE_STEPS
#   (EXTRA_ARGS hardcodes --use_gradient_checkpointing and --find_unused_parameters;
#    delete those two lines if you do not want them)
#   HEIGHT, WIDTH, TASK, TRAINABLE_MODELS
#   TOKENIZER_PATH, PROCESSOR_PATH
#   MODEL_PATHS_JSON (advanced), QWEN_MODEL_ID_WITH_ORIGIN_PATHS, FIRERED_MODEL_ID_WITH_ORIGIN_PATHS
#   OUTPUT_BASE, OUTPUT_PATH_OVERRIDE
#   CUDA_VISIBLE_DEVICES / MEDBANANA_CUDA_DEVICES
#
# Batch note:
#   DiffSynth runner uses DataLoader with implicit batch_size=1 and collate_fn=lambda x: x[0].
#   There is no --train_batch_size in this path. Increase effective batch via
#   GRADIENT_ACCUMULATION_STEPS and multi-process accelerate config.

TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

if [[ -n "${MEDBANANA_CUDA_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${MEDBANANA_CUDA_DEVICES}"
fi
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

MODEL_SUBCMD="${1:?usage: MODEL_SUBCMD (qwen|firered)}"
DATASET_CARD="${2:?usage: DATASET_CARD (10|1000|5000|10000)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDBANANA_TRAIN_ROOT="${MEDBANANA_TRAIN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DIFFSYNTH_ROOT="${DIFFSYNTH_STUDIO_ROOT:-$(cd "${MEDBANANA_TRAIN_ROOT}/../DiffSynth-Studio" && pwd)}"

if [[ ! -d "${DIFFSYNTH_ROOT}" ]]; then
  echo "DiffSynth repo not found: ${DIFFSYNTH_ROOT} (set DIFFSYNTH_STUDIO_ROOT)" >&2
  exit 1
fi

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-$(cd "${MEDBANANA_TRAIN_ROOT}/../.." && pwd)/models}"
if [[ ! -d "${LOCAL_MODELS_ROOT}" ]]; then
  echo "Local model root not found: ${LOCAL_MODELS_ROOT}" >&2
  exit 1
fi
if [[ "${REQUIRE_CUDA}" == "1" ]]; then
  python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print(
        "CUDA is unavailable in current torch runtime. "
        f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
        f"device_count={torch.cuda.device_count()}.",
        file=sys.stderr,
    )
    print(
        "Likely CUDA driver/runtime mismatch. "
        "Fix environment or set REQUIRE_CUDA=0 if you explicitly want CPU training.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
fi

case "${MODEL_SUBCMD}" in
  qwen) ;;
  firered) ;;
  *) echo "MODEL_SUBCMD must be qwen or firered (got ${MODEL_SUBCMD})" >&2 ; exit 1 ;;
esac

case "${DATASET_CARD}" in
  10|1000|5000|10000) ;;
  *) echo "DATASET_CARD must be 10|1000|5000|10000 (got ${DATASET_CARD})" >&2 ; exit 1 ;;
esac

_DATA_SUBDIR_DEFAULT="qwen-edit-medbanana-${DATASET_CARD}"
DATA_SUBDIR="${DIFFSYNTH_MED_EDIT_DATA_SUBDIR:-${_DATA_SUBDIR_DEFAULT}}"
DATASET_ROOT="${MEDBANANA_TRAIN_ROOT}/data/${DATA_SUBDIR}"
if [[ ! -d "${DATASET_ROOT}/images" ]] || [[ ! -d "${DATASET_ROOT}/control_images" ]]; then
  echo "Dataset root must contain images/ and control_images/: ${DATASET_ROOT}" >&2
  exit 1
fi
METADATA_PATH="${DATASET_ROOT}/diffsynth_metadata.jsonl"
EXPORT=(
  python3 "${MEDBANANA_TRAIN_ROOT}/scripts/diffsynth/export_medbanana_diffsynth_metadata.py"
  --dataset-root "${DATASET_ROOT}"
  --output "${METADATA_PATH}"
)
if [[ -n "${METADATA_MAX_SAMPLES:-}" ]]; then
  EXPORT+=(--max-samples "${METADATA_MAX_SAMPLES}")
fi
"${EXPORT[@]}"

OUTPUT_BASE="${OUTPUT_BASE:-${DIFFSYNTH_ROOT}/models/train/medbanana}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
# Defaults match DiffSynth official example: examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh
LR="${LR:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
START_EPOCH="${START_EPOCH:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
REMOVE_PREFIX_IN_CKPT="${REMOVE_PREFIX_IN_CKPT:-pipe.dit.}"
LORA_BASE_MODEL="${LORA_BASE_MODEL:-dit}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1}"
LORA_RANK="${LORA_RANK:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
TASK="${TASK:-sft}"

QWEN_MODEL_ID_WITH_ORIGIN_PATHS="${QWEN_MODEL_ID_WITH_ORIGIN_PATHS:-${LOCAL_MODELS_ROOT}/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,${LOCAL_MODELS_ROOT}/Qwen-Image:text_encoder/model*.safetensors,${LOCAL_MODELS_ROOT}/Qwen-Image:vae/diffusion_pytorch_model.safetensors}"
FIRERED_MODEL_ID_WITH_ORIGIN_PATHS="${FIRERED_MODEL_ID_WITH_ORIGIN_PATHS:-${LOCAL_MODELS_ROOT}/FireRed-Image-Edit-1.1:transformer/diffusion_pytorch_model*.safetensors,${LOCAL_MODELS_ROOT}/Qwen-Image:text_encoder/model*.safetensors,${LOCAL_MODELS_ROOT}/Qwen-Image:vae/diffusion_pytorch_model.safetensors}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${LOCAL_MODELS_ROOT}/Qwen-Image/tokenizer}"
if [[ "${MODEL_SUBCMD}" == "qwen" ]]; then
  _DEFAULT_PROCESSOR_PATH="${LOCAL_MODELS_ROOT}/Qwen-Image-Edit-2511/processor"
else
  _DEFAULT_PROCESSOR_PATH="${LOCAL_MODELS_ROOT}/FireRed-Image-Edit-1.1/processor"
fi
PROCESSOR_PATH="${PROCESSOR_PATH:-${_DEFAULT_PROCESSOR_PATH}}"

if [[ "${MODEL_SUBCMD}" == "qwen" ]] && [[ ! -d "${LOCAL_MODELS_ROOT}/Qwen-Image-Edit-2511" ]]; then
  echo "Missing local model dir: ${LOCAL_MODELS_ROOT}/Qwen-Image-Edit-2511" >&2
  exit 1
fi
if [[ "${MODEL_SUBCMD}" == "firered" ]] && [[ ! -d "${LOCAL_MODELS_ROOT}/FireRed-Image-Edit-1.1" ]]; then
  echo "Missing local model dir: ${LOCAL_MODELS_ROOT}/FireRed-Image-Edit-1.1" >&2
  exit 1
fi
if [[ ! -d "${LOCAL_MODELS_ROOT}/Qwen-Image" ]]; then
  echo "Missing local model dir: ${LOCAL_MODELS_ROOT}/Qwen-Image" >&2
  exit 1
fi
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer path not found: ${TOKENIZER_PATH}" >&2
  exit 1
fi
if [[ ! -d "${PROCESSOR_PATH}" ]]; then
  echo "Processor path not found: ${PROCESSOR_PATH}" >&2
  exit 1
fi

EXTRA_ARGS=(
  accelerate launch "${DIFFSYNTH_ROOT}/examples/qwen_image/model_training/train.py"
  --dataset_base_path "${DATASET_ROOT}"
  --dataset_metadata_path "${METADATA_PATH}"
  --data_file_keys "image,edit_image"
  --extra_inputs "edit_image"
  --max_pixels "${MAX_PIXELS}"
  --dataset_repeat "${DATASET_REPEAT}"
  --learning_rate "${LR}"
  --num_epochs "${NUM_EPOCHS}"
  --start_epoch "${START_EPOCH}"
  --weight_decay "${WEIGHT_DECAY}"
  --remove_prefix_in_ckpt "${REMOVE_PREFIX_IN_CKPT}"
  --task "${TASK}"
  --lora_base_model "${LORA_BASE_MODEL}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --lora_rank "${LORA_RANK}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --use_gradient_checkpointing
  --find_unused_parameters
)

if [[ -n "${MODEL_PATHS_JSON:-}" ]]; then
  EXTRA_ARGS+=(--model_paths "${MODEL_PATHS_JSON}")
fi
if [[ -n "${SAVE_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--save_steps "${SAVE_STEPS}")
fi
if [[ -n "${HEIGHT:-}" ]]; then
  EXTRA_ARGS+=(--height "${HEIGHT}")
fi
if [[ -n "${WIDTH:-}" ]]; then
  EXTRA_ARGS+=(--width "${WIDTH}")
fi
if [[ -n "${TRAINABLE_MODELS:-}" ]]; then
  EXTRA_ARGS+=(--trainable_models "${TRAINABLE_MODELS}")
fi
EXTRA_ARGS+=(--tokenizer_path "${TOKENIZER_PATH}")
EXTRA_ARGS+=(--processor_path "${PROCESSOR_PATH}")
if [[ -n "${LORA_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--lora_checkpoint "${LORA_CHECKPOINT}")
fi
if [[ -n "${PRESET_LORA_PATH:-}" ]]; then
  EXTRA_ARGS+=(--preset_lora_path "${PRESET_LORA_PATH}")
fi
if [[ -n "${PRESET_LORA_MODEL:-}" ]]; then
  EXTRA_ARGS+=(--preset_lora_model "${PRESET_LORA_MODEL}")
fi
if [[ -n "${FP8_MODELS:-}" ]]; then
  EXTRA_ARGS+=(--fp8_models "${FP8_MODELS}")
fi
if [[ -n "${OFFLOAD_MODELS:-}" ]]; then
  EXTRA_ARGS+=(--offload_models "${OFFLOAD_MODELS}")
fi
if [[ -n "${TEMPLATE_MODEL_ID_OR_PATH:-}" ]]; then
  EXTRA_ARGS+=(--template_model_id_or_path "${TEMPLATE_MODEL_ID_OR_PATH}")
fi
if [[ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use_gradient_checkpointing_offload)
fi
if [[ "${ENABLE_LORA_HOT_LOADING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_lora_hot_loading)
fi
if [[ "${INITIALIZE_MODEL_ON_CPU:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--initialize_model_on_cpu)
fi

if [[ "${MODEL_SUBCMD}" == "qwen" ]]; then
  OUTPUT_PATH_DEFAULT="${OUTPUT_BASE}/qwen_edit_medbanana_${DATASET_CARD}"
  OUTPUT_PATH="${OUTPUT_PATH_OVERRIDE:-${OUTPUT_PATH_DEFAULT}}"
  EXTRA_ARGS+=(
    --model_id_with_origin_paths "${QWEN_MODEL_ID_WITH_ORIGIN_PATHS}"
    --output_path "${OUTPUT_PATH}"
    --zero_cond_t
  )
else
  OUTPUT_PATH_DEFAULT="${OUTPUT_BASE}/firered_edit_medbanana_${DATASET_CARD}"
  OUTPUT_PATH="${OUTPUT_PATH_OVERRIDE:-${OUTPUT_PATH_DEFAULT}}"
  EXTRA_ARGS+=(
    --model_id_with_origin_paths "${FIRERED_MODEL_ID_WITH_ORIGIN_PATHS}"
    --output_path "${OUTPUT_PATH}"
  )
fi

cd "${DIFFSYNTH_ROOT}"
export TOKENIZERS_PARALLELISM
export DIFFSYNTH_SKIP_DOWNLOAD
echo "[medbanana-diffsynth] model=${MODEL_SUBCMD} card=${DATASET_CARD}" >&2
echo "[medbanana-diffsynth] dataset_root=${DATASET_ROOT}" >&2
echo "[medbanana-diffsynth] metadata=${METADATA_PATH}" >&2
echo "[medbanana-diffsynth] output_path=${OUTPUT_PATH}" >&2
echo "[medbanana-diffsynth] local_models_root=${LOCAL_MODELS_ROOT}" >&2
echo "[medbanana-diffsynth] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
"${EXTRA_ARGS[@]}"
