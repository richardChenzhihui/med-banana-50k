"""Load a DiffSynth-training LoRA on dit (CPU fuse test).

With ``ModelConfig(model_id=...)`` and no ``DIFFSYNTH_SKIP_DOWNLOAD``, DiffSynth pulls the base
model from ModelScope. This script instead loads from the same local directory layout as
``train_medbanana_edit.sh``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _shards(edit_root: Path, glob_pat: str) -> list[str]:
    paths = sorted(edit_root.glob(glob_pat))
    if not paths:
        _die(f"no weights under {edit_root}: {glob_pat}")
    return [str(p) for p in paths]


def _build_model_configs(backend: str, local_root: Path):
    from diffsynth.pipelines.qwen_image import ModelConfig

    te = local_root / "Qwen-Image"
    dit_name = (
        "Qwen-Image-Edit-2511"
        if backend == "qwen2511"
        else "FireRed-Image-Edit-1.1"
    )
    edit = local_root / dit_name
    vae_f = te / "vae" / "diffusion_pytorch_model.safetensors"
    if not vae_f.is_file():
        _die(f"missing VAE weights: {vae_f}")
    return [
        ModelConfig(path=_shards(edit, "transformer/diffusion_pytorch_model*.safetensors")),
        ModelConfig(path=_shards(te, "text_encoder/model*.safetensors")),
        ModelConfig(path=[str(vae_f)]),
    ]


def _processor_dir(backend: str, local_root: Path) -> Path:
    dit_name = (
        "Qwen-Image-Edit-2511"
        if backend == "qwen2511"
        else "FireRed-Image-Edit-1.1"
    )
    return local_root / dit_name / "processor"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("qwen2511", "firered11"), required=True)
    ap.add_argument("--lora-path", required=True)
    ap.add_argument(
        "--local-models-root",
        default=os.environ.get("LOCAL_MODELS_ROOT") or "",
        help="Directory holding Qwen-Image / Qwen-Image-Edit-2511 or FireRed-Image-Edit-1.1. Defaults to $LOCAL_MODELS_ROOT.",
    )
    args = ap.parse_args()

    lora_p = Path(args.lora_path)
    if not lora_p.is_file():
        sys.exit(f"lora checkpoint not found: {lora_p}")

    if not str(args.local_models_root).strip():
        _die("set --local-models-root or export LOCAL_MODELS_ROOT (same as train_medbanana_edit.sh)")

    local_root = Path(args.local_models_root).expanduser().resolve()
    if not local_root.is_dir():
        sys.exit(f"local-models-root not a directory: {local_root}")

    tok_dir = local_root / "Qwen-Image" / "tokenizer"
    if not tok_dir.is_dir():
        _die(f"missing tokenizer dir: {tok_dir}")
    proc_dir = _processor_dir(args.backend, local_root)
    if not proc_dir.is_dir():
        _die(f"missing processor dir: {proc_dir}")

    med_train = Path(__file__).resolve().parents[2]
    difroot = Path(
        os.environ.get("DIFFSYNTH_STUDIO_ROOT", str(med_train.parent / "DiffSynth-Studio"))
    ).resolve()
    if not difroot.is_dir():
        sys.exit(f"DIFFSYNTH_STUDIO_ROOT not a directory: {difroot}")

    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

    sys.path.insert(0, str(difroot))

    import torch
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    configs = _build_model_configs(args.backend, local_root)
    tokenizer_config = ModelConfig(path=str(tok_dir))
    processor_config = ModelConfig(path=str(proc_dir))

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=configs,
        tokenizer_config=tokenizer_config,
        processor_config=processor_config,
    )
    pipe.load_lora(pipe.dit, str(lora_p.resolve()))
    print(f"Loaded LoRA tensors into dit: {lora_p}")
    pipe.clear_lora()


if __name__ == "__main__":
    main()
