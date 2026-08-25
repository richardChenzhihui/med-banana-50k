#!/usr/bin/env python3
"""
Strip `meta` and rewrite `images` entries to absolute paths for a fixed NFS/root.

Input:  Med-Banana/finetune/vlm_{prompt_generation,verification}_dataset.json
Output: Med-Banana/finetune/data/  (same basenames)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rewrite_record(obj: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict record, got {type(obj)}")
    out = {k: v for k, v in obj.items() if k != "meta"}
    if "images" not in out:
        raise KeyError("record missing images")
    imgs = out["images"]
    if not isinstance(imgs, list):
        raise TypeError("images must be a list")
    root = prefix.rstrip("/")
    new_imgs: list[str] = []
    for p in imgs:
        if not isinstance(p, str):
            raise TypeError("each image path must be str")
        rel = p.strip().replace("\\", "/")
        if rel.startswith("/"):
            raise ValueError(f"unexpected absolute path in source: {p!r}")
        new_imgs.append(f"{root}/{rel}")
    out["images"] = new_imgs
    return out


def process_file(src: Path, dst: Path, prefix: str) -> int:
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{src} must be a JSON array")
    rewritten = [rewrite_record(item, prefix) for item in data]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rewritten)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--finetune-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing source vlm_*.json",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <finetune-dir>/data",
    )
    ap.add_argument(
        "--path-prefix",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Prepend this (no trailing slash required) to each relative image path",
    )
    args = ap.parse_args()

    finetune_dir = args.finetune_dir.resolve()
    out_dir = (args.output_dir or (finetune_dir / "data")).resolve()

    files = [
        "vlm_prompt_generation_dataset.json",
        "vlm_verification_dataset.json",
    ]
    for name in files:
        src = finetune_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        n = process_file(src, out_dir / name, args.path_prefix)
        print(f"{name}: {n} -> {out_dir / name}")


if __name__ == "__main__":
    main()
