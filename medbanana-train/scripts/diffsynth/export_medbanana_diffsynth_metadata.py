#!/usr/bin/env python3
"""Write DiffSynth UnifiedDataset metadata (JSONL) for qwen-edit-medbanana-* layout.

Maps (same semantics as flymyai control_dataset):
- image           -> edited/target file under images/ (paired with *.txt caption)
- edit_image      -> source/condition file under control_images/
- prompt          -> text from images/<stem>.txt

`image` and `edit_image` in each JSON row are resolved **absolute filesystem paths** (POSIX). DiffSynth loaders join them with dataset base_path; an absolute operand is unchanged."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
CTL_TRIES = (".jpg", ".jpeg", ".png", ".webp")


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def _find_control(rel_control_dir: Path, stem: str) -> Path:
    for suf in CTL_TRIES:
        p = rel_control_dir / f"{stem}{suf}"
        if p.is_file():
            return p
    _die(f"missing control image for stem {stem!r} under {rel_control_dir}")


_EPILOG = """
Examples (either way of naming the dataset root)::

  export ROOT=/path/to/medbanana-train/data/qwen-edit-medbanana-1000
  python export_medbanana_diffsynth_metadata.py --dataset-root "$ROOT"

  export MEDBANANA_EDIT_DATASET_ROOT=/path/to/medbanana-train/data/qwen-edit-medbanana-1000
  python export_medbanana_diffsynth_metadata.py

  python export_medbanana_diffsynth_metadata.py \\
    --dataset-root /path/to/medbanana-train/data/qwen-edit-medbanana-1000 \\
    --max-samples 64

When --dataset-root is omitted, MEDBANANA_EDIT_DATASET_ROOT is used instead."""
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    ap.add_argument(
        "--dataset-root",
        default=None,
        metavar="PATH",
        help="Dataset directory holding images/ and control_images/; falls back to $MEDBANANA_EDIT_DATASET_ROOT when omitted",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="JSONL output path (default: <dataset-root>/diffsynth_metadata.jsonl)",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Write at most N rows (sorted by stem). For smoke tests.",
    )
    args = ap.parse_args()

    root_str = args.dataset_root or os.environ.get("MEDBANANA_EDIT_DATASET_ROOT", "").strip()
    if not root_str:
        ap.error(
            "Missing --dataset-root, and MEDBANANA_EDIT_DATASET_ROOT is not set."
            " e.g. MEDBANANA_EDIT_DATASET_ROOT=$PWD/../medbanana-train/data/qwen-edit-medbanana-1000 python ..."
        )

    root = Path(root_str).resolve()
    img_dir = root / "images"
    ctl_dir = root / "control_images"
    if not img_dir.is_dir():
        _die(f"not a directory: {img_dir}")
    if not ctl_dir.is_dir():
        _die(f"not a directory: {ctl_dir}")

    stems: list[str] = []
    for p in sorted(img_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_SUFFIXES:
            continue
        stem = p.stem
        txt_path = img_dir / f"{stem}.txt"
        if not txt_path.is_file():
            continue
        stems.append(stem)

    if not stems:
        _die(f"no usable samples under {img_dir} (need image + matching .txt)")

    if args.max_samples is not None:
        stems = stems[: args.max_samples]

    out_path = Path(args.output) if args.output else root / "diffsynth_metadata.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for stem in stems:
        tgt_file = None
        for suf in IMG_SUFFIXES:
            cand = img_dir / f"{stem}{suf}"
            if cand.is_file():
                tgt_file = cand
                break
        if tgt_file is None:
            _die(f"internal error: missing target for stem {stem!r}")

        ctl_file = _find_control(ctl_dir, stem)
        txt_file = img_dir / f"{stem}.txt"
        prompt = txt_file.read_text(encoding="utf-8")
        if not prompt.strip():
            _die(f"empty prompt file: {txt_file}")

        abs_img = tgt_file.resolve().as_posix()
        abs_ctl = ctl_file.resolve().as_posix()
        rows.append({"image": abs_img, "edit_image": abs_ctl, "prompt": prompt})

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {"out": str(out_path), "count": len(rows), "dataset_root": str(root)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
