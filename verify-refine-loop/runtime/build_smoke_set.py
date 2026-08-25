#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


def _die(msg: str) -> None:
    raise SystemExit(msg)


@dataclass
class PairMeta:
    stem: str
    task_type: str
    dataset_code: str
    disease: str
    image_src: Path
    prompt_src: Path


def _parse_stem(stem: str) -> tuple[str, str, str]:
    parts = stem.split("__")
    if len(parts) < 6 or parts[0] != "med":
        raise ValueError(f"invalid stem format: {stem}")
    dataset_code = parts[1]
    disease = parts[2]
    task_type = parts[3]
    if task_type not in {"addition", "removal"}:
        raise ValueError(f"invalid task_type in stem: {stem}")
    if dataset_code not in {"brain_mri", "chest_xray", "fundus"}:
        raise ValueError(f"invalid dataset code in stem: {stem}")
    return dataset_code, disease, task_type


def _collect_pairs(src_root: Path) -> list[PairMeta]:
    images_dir = src_root / "images"
    control_dir = src_root / "control_images"
    if not images_dir.is_dir():
        _die(f"missing images dir: {images_dir}")
    if not control_dir.is_dir():
        _die(f"missing control_images dir: {control_dir}")

    pairs: list[PairMeta] = []
    for prompt_path in sorted(images_dir.glob("*.txt")):
        stem = prompt_path.stem
        image_src = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = control_dir / f"{stem}{ext}"
            if p.is_file():
                image_src = p
                break
        if image_src is None:
            continue
        dataset_code, disease, task_type = _parse_stem(stem)
        pairs.append(
            PairMeta(
                stem=stem,
                task_type=task_type,
                dataset_code=dataset_code,
                disease=disease,
                image_src=image_src,
                prompt_src=prompt_path,
            )
        )
    if not pairs:
        _die("no valid pairs found")
    return pairs


def _pick_stratified(pairs: list[PairMeta], total: int) -> list[PairMeta]:
    if total != 10:
        raise ValueError("this smoke builder currently requires --n 10")

    by_bucket: dict[tuple[str, str], list[PairMeta]] = {}
    for p in pairs:
        by_bucket.setdefault((p.task_type, p.dataset_code), []).append(p)
    for v in by_bucket.values():
        v.sort(key=lambda x: x.stem)

    selected: list[PairMeta] = []
    used = set()

    # Ensure coverage: each (addition/removal) x (brain_mri/chest_xray/fundus)
    for task_type in ("addition", "removal"):
        for dataset_code in ("brain_mri", "chest_xray", "fundus"):
            bucket = by_bucket.get((task_type, dataset_code), [])
            if not bucket:
                raise ValueError(f"empty bucket {task_type}-{dataset_code}")
            pick = bucket[0]
            selected.append(pick)
            used.add(pick.stem)

    # Fill to 5 per task type, deterministic by stem.
    for task_type in ("addition", "removal"):
        need = 5 - sum(1 for x in selected if x.task_type == task_type)
        if need < 0:
            raise ValueError("internal selection error")
        pool = sorted(
            [p for p in pairs if p.task_type == task_type and p.stem not in used],
            key=lambda x: x.stem,
        )
        if len(pool) < need:
            raise ValueError(f"not enough pairs to fill {task_type} quota")
        for p in pool[:need]:
            selected.append(p)
            used.add(p.stem)

    if len(selected) != total:
        raise ValueError(f"selected count mismatch: {len(selected)} vs {total}")
    return sorted(selected, key=lambda x: x.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 10-pair smoke set for verify-refine runtime")
    parser.add_argument("--src", required=True, help="qwen-edit-medbanana-10000 root")
    parser.add_argument("--dst", required=True, help="destination directory")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(src_root)
    selected = _pick_stratified(pairs, args.n)

    manifest = []
    for item in selected:
        img_dst = dst / item.image_src.name
        txt_dst = dst / item.prompt_src.name
        shutil.copy2(item.image_src, img_dst)
        shutil.copy2(item.prompt_src, txt_dst)
        manifest.append(
            {
                "stem": item.stem,
                "task_type": item.task_type,
                "dataset_code": item.dataset_code,
                "disease": item.disease,
                "image_src": str(item.image_src),
                "prompt_src": str(item.prompt_src),
                "image_dst": str(img_dst),
                "prompt_dst": str(txt_dst),
            }
        )

    with open(dst / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({"count": len(manifest), "dst": str(dst)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
