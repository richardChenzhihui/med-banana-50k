#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import importlib.util
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _load_diffsynth_helpers(script_path: Path):
    spec = importlib.util.spec_from_file_location("diffsynth_merge_lora_infer_one", script_path)
    if spec is None or spec.loader is None:
        _die(f"failed to load helper module: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _decode_b64_image(image_b64: str) -> Image.Image:
    if not image_b64:
        raise ValueError("empty image_b64")
    payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    raw = base64.b64decode(payload)
    return Image.open(BytesIO(raw)).convert("RGB")


def _encode_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    seed: int | None = None
    num_inference_steps: int | None = None
    cfg_scale: float | None = None
    negative_prompt: str | None = None
    height: int | None = None
    width: int | None = None


class EditServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.pipe = None
        self.lora_loaded = False
        self.rand_device = "cuda" if str(args.device).startswith("cuda") else "cpu"
        self.zero_cond_t = args.backend == "qwen2511"
        self.ready = False
        self._load()

    def _load(self) -> None:
        med_train = Path(__file__).resolve().parents[2] / "medbanana-train"
        script_path = med_train / "scripts" / "diffsynth" / "diffsynth_merge_lora_infer_one.py"
        if not script_path.is_file():
            _die(f"missing helper script: {script_path}")

        helper = _load_diffsynth_helpers(script_path)

        local_root = Path(self.args.local_models_root).expanduser().resolve()
        if not local_root.is_dir():
            _die(f"local-models-root not a directory: {local_root}")
        lora_raw = (self.args.lora_path or "").strip()
        lora_path: Path | None = None
        if lora_raw:
            lora_path = Path(lora_raw).expanduser().resolve()
            if not lora_path.is_file():
                _die(f"missing LoRA file: {lora_path}")
        self.lora_loaded = lora_path is not None

        tokenizer_dir = local_root / "Qwen-Image" / "tokenizer"
        if not tokenizer_dir.is_dir():
            _die(f"missing tokenizer dir: {tokenizer_dir}")

        proc_dir = helper._processor_path(self.args.backend, local_root)
        if not proc_dir.is_dir():
            _die(f"missing processor dir: {proc_dir}")

        med_root = Path(__file__).resolve().parents[2]
        difroot = Path(
            os.environ.get("DIFFSYNTH_STUDIO_ROOT", str(med_root / "DiffSynth-Studio"))
        ).resolve()
        if not difroot.is_dir():
            _die(f"DIFFSYNTH_STUDIO_ROOT not a directory: {difroot}")

        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        sys.path.insert(0, str(difroot))

        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        torch_dtype = helper._infer_dtype(self.args.dtype)
        model_cfgs = helper._build_model_configs(self.args.backend, local_root)
        tokenizer_conf = ModelConfig(path=str(tokenizer_dir))
        processor_conf = ModelConfig(path=str(proc_dir))

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=self.args.device,
            model_configs=model_cfgs,
            tokenizer_config=tokenizer_conf,
            processor_config=processor_conf,
        )
        if lora_path is not None:
            self.pipe.load_lora(self.pipe.dit, str(lora_path))
        self._infer_once = helper._infer_once
        self.ready = True

    def edit(self, req: EditRequest) -> dict[str, Any]:
        if not self.ready or self.pipe is None:
            raise RuntimeError("edit server not ready")
        if not req.prompt:
            raise ValueError("empty prompt")

        image = _decode_b64_image(req.image_b64)
        iw, ih = image.size
        height = req.height if req.height is not None else max(16, (ih // 16) * 16)
        width = req.width if req.width is not None else max(16, (iw // 16) * 16)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("height and width must be multiples of 16")

        seed = self.args.seed if req.seed is None else req.seed
        steps = (
            self.args.num_inference_steps
            if req.num_inference_steps is None
            else req.num_inference_steps
        )
        cfg = self.args.cfg_scale if req.cfg_scale is None else req.cfg_scale
        neg = self.args.negative_prompt if req.negative_prompt is None else req.negative_prompt

        start = time.time()
        with self.lock:
            out = self._infer_once(
                self.pipe,
                prompt=req.prompt,
                negative_prompt=neg,
                cfg_scale=cfg,
                edit_image_rgb=image,
                height=height,
                width=width,
                seed=seed,
                num_inference_steps=steps,
                rand_device=self.rand_device,
                zero_cond_t=self.zero_cond_t,
            )
        elapsed_ms = int((time.time() - start) * 1000)
        return {"image_b64": _encode_b64_png(out), "elapsed_ms": elapsed_ms}


def build_app(server: EditServer) -> FastAPI:
    app = FastAPI(title="medbanana-diffsynth-edit-server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": server.ready,
            "backend": server.args.backend,
            "device": server.args.device,
            "lora_loaded": server.lora_loaded,
            "lora_path": server.args.lora_path or None,
        }

    @app.post("/edit")
    def edit(req: EditRequest) -> dict[str, Any]:
        try:
            return server.edit(req)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffSynth edit server (optional LoRA)")
    parser.add_argument("--backend", choices=("qwen2511", "firered11"), required=True)
    parser.add_argument("--local-models-root", default=os.environ.get("LOCAL_MODELS_ROOT") or "")
    parser.add_argument(
        "--lora-path",
        default=os.environ.get("LORA_PATH", ""),
        help="omit or empty for base model only (no LoRA)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not args.local_models_root.strip():
        _die("set --local-models-root or LOCAL_MODELS_ROOT")
    return args


def main() -> None:
    args = parse_args()
    server = EditServer(args)
    app = build_app(server)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
