#!/usr/bin/env python3
"""Build the flymyai Qwen-Image-Edit layout from a Med-Banana-80K success_edit.jsonl
(3 modalities, stratified sampling).

Download the dataset from https://huggingface.co/datasets/RichardChenZH/Med-Banana-80K
and pass its root via --medbanana-root. For DiffSynth training see
medbanana-train/scripts/diffsynth/train_medbanana_edit.sh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Values as in success_edit.jsonl
STRAT_MODALITIES = ("brain_mri", "chest_xray", "fundus")


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def _safe_stem(parts: list[str]) -> str:
    raw = "__".join(p for p in parts if p)
    raw = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")
    if not raw:
        _die("empty stem after sanitize")
    return raw[:180]


def _resolve_under_medbanana_root(medbanana_root: Path, raw: str) -> Path:
    """Resolve a success_edit.jsonl path against the dataset root.

    Paths are relative to the root; ".." escapes are rejected. Already-absolute paths are
    accepted as-is.
    """
    s = raw.strip()
    if not s:
        _die("empty path in success_edit row")
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    root_r = medbanana_root.resolve()
    out = (root_r / p).resolve()
    try:
        out.relative_to(root_r)
    except ValueError:
        _die(f"path escapes medbanana root ({root_r}): {raw!r}")
    return out


def _load_rows(jsonl_path: Path, medbanana_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                _die(f"{jsonl_path}:{lineno}: JSON decode error: {e}")
            mod = o.get("modality")
            if mod not in STRAT_MODALITIES:
                continue
            if o.get("status") != "success":
                continue
            for k in ("edited_image", "source_image", "prompt"):
                if not o.get(k):
                    _die(f"{jsonl_path}:{lineno}: missing field {k!r}")
            src_p = _resolve_under_medbanana_root(medbanana_root, str(o["source_image"]))
            tgt_p = _resolve_under_medbanana_root(medbanana_root, str(o["edited_image"]))
            if not src_p.is_file():
                _die(
                    f"{jsonl_path}:{lineno}: source_image not a regular file: {src_p} "
                    f"(from {o['source_image']!r})"
                )
            if not tgt_p.is_file():
                _die(
                    f"{jsonl_path}:{lineno}: edited_image not a regular file: {tgt_p} "
                    f"(from {o['edited_image']!r})"
                )
            rows.append(o)
    return rows


def _verify_manifest_paths(manifest_path: Path) -> None:
    """Validate a written manifest: every source_image/edited_image must be an existing
    absolute path."""
    with open(manifest_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                _die(f"{manifest_path}:{lineno}: JSON decode error: {e}")
            for key in ("source_image", "edited_image"):
                v = o.get(key)
                if not isinstance(v, str) or not v.strip():
                    _die(f"{manifest_path}:{lineno}: missing {key}")
                p = Path(v)
                if not p.is_absolute():
                    _die(
                        f"{manifest_path}:{lineno}: {key} must be absolute path in manifest, "
                        f"got {v!r}"
                    )
                pr = p.resolve()
                if not pr.is_file():
                    _die(f"{manifest_path}:{lineno}: {key} not a regular file: {pr}")


def _stratified_sample(
    rows: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    if n > len(rows):
        _die(f"need {n} samples but only {len(rows)} eligible rows after filters")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        mod = r.get("modality")
        if mod not in STRAT_MODALITIES:
            continue
        buckets[mod].append(r)
    for k in STRAT_MODALITIES:
        random.Random(seed).shuffle(buckets[k])
    base, rem = divmod(n, len(STRAT_MODALITIES))
    quotas = {k: base + (1 if i < rem else 0) for i, k in enumerate(STRAT_MODALITIES)}
    for k in STRAT_MODALITIES:
        if len(buckets[k]) < quotas[k]:
            _die(f"modality {k!r}: need {quotas[k]} have {len(buckets[k])}")
    out: list[dict[str, Any]] = []
    for k in STRAT_MODALITIES:
        out.extend(buckets[k][: quotas[k]])
    random.Random(seed + 7).shuffle(out)
    return out


def _copy_triplet(
    medbanana_root: Path,
    row: dict[str, Any],
    images_dir: str,
    control_dir: str,
    used_stems: set[str],
) -> dict[str, Any]:
    src_p = _resolve_under_medbanana_root(medbanana_root, str(row["source_image"]))
    tgt_p = _resolve_under_medbanana_root(medbanana_root, str(row["edited_image"]))
    if not src_p.is_file():
        _die(f"missing source_image: {src_p}")
    if not tgt_p.is_file():
        _die(f"missing edited_image: {tgt_p}")

    modality = str(row.get("modality") or "unknown").strip()
    disease = (row.get("disease") or row.get("label") or "").strip().lower()
    edit_type = (row.get("edit_type") or "unknown").strip().lower()
    image_id = str(row.get("image_id") or "").strip()
    # Stem stability: hash the original path string from success_edit.jsonl so stems do not
    # drift when the dataset is re-run on a different machine.
    h = hashlib.sha1(
        f"{row['source_image']}|{row['edited_image']}".encode()
    ).hexdigest()[:10]
    stem = _safe_stem(["med", modality, disease, edit_type, image_id or h, h])
    if stem in used_stems:
        stem = _safe_stem(["med", modality, disease, edit_type, image_id or h, h, h])
    used_stems.add(stem)
    out_img = os.path.join(images_dir, f"{stem}.jpg")
    out_ctl = os.path.join(control_dir, f"{stem}.jpg")
    out_txt = os.path.join(images_dir, f"{stem}.txt")

    shutil.copy2(str(src_p), out_ctl)
    shutil.copy2(str(tgt_p), out_img)
    with open(out_txt, "w", encoding="utf-8") as tf:
        tf.write(str(row["prompt"]))

    return {
        "stem": stem,
        "modality": modality,
        "disease": disease,
        "edit_type": edit_type,
        "image_id": image_id,
        "source_image": src_p.resolve().as_posix(),
        "edited_image": tgt_p.resolve().as_posix(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--medbanana-root",
        required=True,
        help=(
            "Root of a downloaded Med-Banana-80K tree containing success_edit.jsonl "
            "and the data/ image directories "
            "(https://huggingface.co/datasets/RichardChenZH/Med-Banana-80K)"
        ),
    )
    ap.add_argument(
        "--out-root",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
        ),
        help="Directory for qwen-edit-medbanana-* outputs",
    )
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    medbanana_root = Path(args.medbanana_root).resolve()
    jsonl_path = medbanana_root / "success_edit.jsonl"
    if not jsonl_path.is_file():
        _die(f"missing {jsonl_path}")

    rows = _load_rows(jsonl_path, medbanana_root)
    if not rows:
        _die("no eligible rows after modality filter and file existence check")
    os.makedirs(args.out_root, exist_ok=True)

    for n in args.sizes:
        picked = _stratified_sample(rows, n, args.seed)
        slug = f"qwen-edit-medbanana-{n}"
        out_dir = os.path.join(args.out_root, slug)
        images_dir = os.path.join(out_dir, "images")
        control_dir = os.path.join(out_dir, "control_images")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(control_dir, exist_ok=True)

        used: set[str] = set()
        manifest: list[dict[str, Any]] = []
        for r in picked:
            manifest.append(
                _copy_triplet(medbanana_root, r, images_dir, control_dir, used)
            )

        summary: dict[str, Any] = {
            "n": n,
            "seed": args.seed,
            "stratify_modalities": list(STRAT_MODALITIES),
            "counts_by_modality": {
                k: sum(1 for m in manifest if m["modality"] == k) for k in STRAT_MODALITIES
            },
            "counts_by_edit_type": {
                k: sum(1 for m in manifest if m["edit_type"] == k)
                for k in sorted({m["edit_type"] for m in manifest})
            },
        }
        with open(os.path.join(out_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        manifest_path = Path(out_dir) / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        _verify_manifest_paths(manifest_path)

        print(json.dumps({"out_dir": out_dir, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
