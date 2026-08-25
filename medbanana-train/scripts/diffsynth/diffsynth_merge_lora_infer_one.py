#!/usr/bin/env python3
"""Run DiffSynth base + fused LoRA inference on a single Med-Banana triplet.

Diffusers-style LoRA files and DiffSynth train.py outputs are NOT interchangeable;
this script uses QwenImagePipeline.load_lora, which fuses internally. See DiffSynth:
  examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2511.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _infer_dtype(dtype: str):
    import torch

    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    _die(f"unsupported dtype={dtype}")


@dataclass
class Sample:
    stem: str
    prompt_path: str
    control_path: str
    target_path: str | None


def _find_image_by_stem(folder: str, stem: str) -> str | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join(folder, f"{stem}{ext}")
        if os.path.isfile(path):
            return path
    return None


def _collect_valid_samples(images_dir: str, control_dir: str) -> list[Sample]:
    txt_files = sorted(
        os.path.join(images_dir, name)
        for name in os.listdir(images_dir)
        if name.endswith(".txt")
    )
    out: list[Sample] = []
    for prompt_path in txt_files:
        stem = os.path.splitext(os.path.basename(prompt_path))[0]
        control_path = _find_image_by_stem(control_dir, stem)
        if control_path is None:
            continue
        target_path = _find_image_by_stem(images_dir, stem)
        out.append(
            Sample(
                stem=stem,
                prompt_path=prompt_path,
                control_path=control_path,
                target_path=target_path,
            )
        )
    return out


def _pick_sample(
    data_root: str,
    sample_stem: str | None,
    sample_index_one_based: int | None,
) -> tuple[Sample, int]:
    images_dir = os.path.join(data_root, "images")
    control_dir = os.path.join(data_root, "control_images")
    if not os.path.isdir(images_dir):
        _die(f"missing images dir: {images_dir}")
    if not os.path.isdir(control_dir):
        _die(f"missing control_images dir: {control_dir}")
    valid = _collect_valid_samples(images_dir, control_dir)
    if not valid:
        _die("no sample has both prompt txt and control image")

    if sample_stem is not None:
        for i, v in enumerate(valid, start=1):
            if v.stem == sample_stem:
                return v, i
        _die(f"stem not found or no control counterpart: {sample_stem!r}")

    if sample_index_one_based is not None:
        if sample_index_one_based < 1:
            _die("--sample-index is 1-based")
        if sample_index_one_based > len(valid):
            _die(f"--sample-index out of range: have {len(valid)}")
        return valid[sample_index_one_based - 1], sample_index_one_based

    return valid[0], 1


def _shards(edit_root: Path, glob_pat: str) -> list[str]:
    paths = sorted(edit_root.glob(glob_pat))
    if not paths:
        _die(f"no weights under {edit_root}: {glob_pat}")
    return [str(p) for p in paths]


def _build_model_configs(backend: str, local_root: Path) -> list:
    from diffsynth.pipelines.qwen_image import ModelConfig

    te = local_root / "Qwen-Image"
    dit_dir = (
        "Qwen-Image-Edit-2511"
        if backend == "qwen2511"
        else "FireRed-Image-Edit-1.1"
    )
    edit = local_root / dit_dir
    vae_f = te / "vae" / "diffusion_pytorch_model.safetensors"
    if not vae_f.is_file():
        _die(f"missing VAE weights: {vae_f}")
    return [
        ModelConfig(path=_shards(edit, "transformer/diffusion_pytorch_model*.safetensors")),
        ModelConfig(path=_shards(te, "text_encoder/model*.safetensors")),
        ModelConfig(path=[str(te / "vae" / "diffusion_pytorch_model.safetensors")]),
    ]


def _processor_path(backend: str, local_root: Path) -> Path:
    if backend == "qwen2511":
        p = local_root / "Qwen-Image-Edit-2511" / "processor"
    else:
        p = local_root / "FireRed-Image-Edit-1.1" / "processor"
    return p


def _infer_once(
    pipe,
    *,
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    edit_image_rgb,
    height: int,
    width: int,
    seed: int,
    num_inference_steps: int,
    rand_device: str,
    zero_cond_t: bool,
):
    kw = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg_scale=cfg_scale,
        edit_image=[edit_image_rgb],
        seed=seed,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        edit_image_auto_resize=True,
        rand_device=rand_device,
    )
    if zero_cond_t:
        kw["zero_cond_t"] = True
    return pipe(**kw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiffSynth: fuse epoch-*.safetensors LoRA into DiT then one-sample edit inference."
    )
    parser.add_argument(
        "--backend",
        choices=("qwen2511", "firered11"),
        default="qwen2511",
        help="qwen2511 → Qwen-Image-Edit-2511 + zero_cond_t",
    )
    parser.add_argument(
        "--local-models-root",
        default=os.environ.get("LOCAL_MODELS_ROOT") or "",
        help="Base-model directory (must contain Qwen-Image / Qwen-Image-Edit-2511 or FireRed-Image-Edit-1.1). Defaults to $LOCAL_MODELS_ROOT.",
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="A .safetensors LoRA produced by DiffSynth training, e.g. epoch-0.safetensors",
    )
    parser.add_argument(
        "--data-root",
        default=str(
            Path(__file__).resolve().parents[2] / "data" / "qwen-edit-medbanana-10"
        ),
        help="Data root holding images/ and control_images/",
    )
    sample_grp = parser.add_mutually_exclusive_group()
    sample_grp.add_argument("--sample-stem", default=None)
    sample_grp.add_argument("--sample-index", type=int, default=None, metavar="N")
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "diffsynth_edit_merge_infer_one"
        ),
    )
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="cfg_scale as used by DiffSynth QwenImagePipeline (true_cfg_scale in the flymyai docs)",
    )
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run once without LoRA, then again with the LoRA fused into the DiT, for comparison",
    )
    args = parser.parse_args()

    if not args.local_models_root.strip():
        _die("set --local-models-root or LOCAL_MODELS_ROOT")
    local_root = Path(args.local_models_root).expanduser().resolve()
    if not local_root.is_dir():
        _die(f"local-models-root not a directory: {local_root}")

    lora_path = Path(args.lora_path).expanduser().resolve()
    if not lora_path.is_file():
        _die(f"missing LoRA file: {lora_path}")

    lp_lower = str(lora_path).lower().replace("\\", "/")
    if args.backend == "qwen2511" and "firered_edit" in lp_lower:
        _die(
            "--backend qwen2511 loads LOCAL **Qwen-Image-Edit-2511** DiT; your --lora-path is under "
            "**firered_edit_***. Use `--backend firered11` and `.../firered_edit_medbanana_*/epoch-*.safetensors`, "
            "or use a `qwen_edit_medbanana_*` LoRA with qwen2511."
        )
    if args.backend == "firered11" and (
        "qwen_edit_medbanana" in lp_lower or "/qwen_edit_" in lp_lower
    ):
        if "firered_edit" not in lp_lower:
            _die(
                "--backend firered11 loads LOCAL **FireRed-Image-Edit-1.1** DiT; your LoRA path looks like "
                "**qwen_edit_***. Use `.../firered_edit_medbanana_*/epoch-*.safetensors` or switch to "
                "`--backend qwen2511` for Qwen LoRA."
            )

    tokenizer_dir = local_root / "Qwen-Image" / "tokenizer"
    if not tokenizer_dir.is_dir():
        _die(f"missing tokenizer dir: {tokenizer_dir}")
    proc_dir = _processor_path(args.backend, local_root)
    if not proc_dir.is_dir():
        _die(f"missing processor dir: {proc_dir}")

    if args.sample_stem is not None and args.sample_stem.isdigit():
        probe = os.path.join(args.data_root, "images", f"{args.sample_stem}.txt")
        if not os.path.isfile(probe):
            idx = int(args.sample_stem)
            print(
                "NOTE: --sample-stem looks like a number and "
                f"{probe} does not exist; using --sample-index {idx} (1-based). "
                "Prefer: --sample-index N",
                file=sys.stderr,
            )
            args.sample_index = idx
            args.sample_stem = None

    sample, resolved_idx = _pick_sample(args.data_root, args.sample_stem, args.sample_index)

    os.makedirs(args.output_dir, exist_ok=True)
    ext = Path(sample.control_path).suffix
    input_copy = os.path.join(args.output_dir, f"{sample.stem}_input{ext}")
    prompt_copy = os.path.join(args.output_dir, f"{sample.stem}.txt")
    shutil.copy2(sample.control_path, input_copy)
    shutil.copy2(sample.prompt_path, prompt_copy)

    target_copy = None
    if sample.target_path is not None:
        target_copy = os.path.join(
            args.output_dir,
            f"{sample.stem}_target{Path(sample.target_path).suffix}",
        )
        shutil.copy2(sample.target_path, target_copy)

    with open(sample.prompt_path, encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        _die(f"empty prompt: {sample.prompt_path}")

    med_train = Path(__file__).resolve().parents[2]
    difroot = Path(os.environ.get("DIFFSYNTH_STUDIO_ROOT", med_train.parent / "DiffSynth-Studio")).resolve()
    if not difroot.is_dir():
        _die(f"DIFFSYNTH_STUDIO_ROOT not a directory: {difroot}")

    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

    sys.path.insert(0, str(difroot))

    from PIL import Image

    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    torch_dtype = _infer_dtype(args.dtype)
    model_cfgs = _build_model_configs(args.backend, local_root)
    tokenizer_conf = ModelConfig(path=str(tokenizer_dir))
    processor_conf = ModelConfig(path=str(proc_dir))

    rand_device = "cuda" if str(args.device).startswith("cuda") else "cpu"
    _dit_id = (
        "Qwen-Image-Edit-2511"
        if args.backend == "qwen2511"
        else "FireRed-Image-Edit-1.1"
    )
    print(
        f"[medbanana-merge] backend={args.backend} "
        f"base_dit_dir={_dit_id} processor={proc_dir} lora={lora_path}",
        file=sys.stderr,
    )
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=model_cfgs,
        tokenizer_config=tokenizer_conf,
        processor_config=processor_conf,
    )

    edit_rgb = Image.open(input_copy).convert("RGB")

    if args.height is not None and args.width is not None:
        hi, wi = args.height, args.width
        if hi % 16 != 0 or wi % 16 != 0:
            _die("height and width must be multiples of 16")
    else:
        iw, ih = edit_rgb.size
        hi = max(16, (ih // 16) * 16)
        wi = max(16, (iw // 16) * 16)

    zero_cond = args.backend == "qwen2511"

    baseline_path = None
    if args.compare:
        out_b = _infer_once(
            pipe,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            cfg_scale=args.cfg_scale,
            edit_image_rgb=edit_rgb,
            height=hi,
            width=wi,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            rand_device=rand_device,
            zero_cond_t=zero_cond,
        )
        baseline_path = os.path.join(args.output_dir, f"{sample.stem}_edited_baseline.png")
        out_b.save(baseline_path)

    pipe.load_lora(pipe.dit, str(lora_path))
    out_l = _infer_once(
        pipe,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        edit_image_rgb=edit_rgb,
        height=hi,
        width=wi,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        rand_device=rand_device,
        zero_cond_t=zero_cond,
    )
    out_path = os.path.join(args.output_dir, f"{sample.stem}_edited.png")
    out_l.save(out_path)

    info = {
        "stack": "diffsynth_qwen_pipeline",
        "backend": args.backend,
        "base_dit_weights_dir": _dit_id,
        "local_models_root": str(local_root),
        "lora_path": str(lora_path),
        "compare": args.compare,
        "sample_stem": sample.stem,
        "sample_index_cli": args.sample_index,
        "resolved_sample_index_one_based": resolved_idx,
        "input_image": input_copy,
        "prompt_file": prompt_copy,
        "target_image": target_copy,
        "output_image_baseline": baseline_path,
        "output_image": out_path,
        "inference_kwargs": {
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "height": hi,
            "width": wi,
            "rand_device": rand_device,
            "zero_cond_t": zero_cond,
        },
        "dtype": args.dtype,
        "device": args.device,
        "notes": (
            "LoRA fused into DiT via pipe.load_lora (see diffsynth.diffusion.base_pipeline.load_lora). "
            "Not compatible with merge_lora_infer_one.py / diffusers LoRA checkpoints."
        ),
    }
    info_path = os.path.join(args.output_dir, "run_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
