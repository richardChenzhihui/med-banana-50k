#!/usr/bin/env python3
"""Verify every path in vlm_* dataset JSON `images` exists on disk."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def iter_image_paths(records: Any, *, src: Path) -> list[str]:
    if not isinstance(records, list):
        raise TypeError(f"{src} must be a JSON array")
    out: list[str] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"{src} record {i}: expected object")
        imgs = rec.get("images")
        if not isinstance(imgs, list):
            raise KeyError(f"{src} record {i}: missing or invalid images")
        for j, p in enumerate(imgs):
            if not isinstance(p, str) or not p.strip():
                raise TypeError(f"{src} record {i} images[{j}]: non-empty str required")
            rel = p.strip().replace("\\", "/")
            if rel.startswith("/"):
                raise ValueError(f"{src} record {i}: unexpected absolute path {p!r}")
            out.append(rel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    finetune_dir = Path(__file__).resolve().parent
    default_publish_root = finetune_dir.parent
    ap.add_argument(
        "--publish-root",
        type=Path,
        default=default_publish_root,
        help="Same directory as build_vlm_sft_from_conversations.py --medbanana-root "
        "(paths join as publish-root / images[i])",
    )
    ap.add_argument(
        "--prompt-json",
        type=Path,
        required=True,
        help="vlm_prompt_generation_dataset.json produced by build_vlm_sft_from_conversations.py",
    )
    ap.add_argument(
        "--verify-json",
        type=Path,
        required=True,
        help="vlm_verification_dataset.json produced by build_vlm_sft_from_conversations.py",
    )
    ap.add_argument(
        "--list-missing",
        type=int,
        default=20,
        metavar="N",
        help="Print at most N missing paths per dataset (0 = print all). Default: 20",
    )
    args = ap.parse_args()

    publish_root = args.publish_root.resolve()
    if not publish_root.is_dir():
        raise FileNotFoundError(f"--publish-root not a directory: {publish_root}")

    all_missing: list[tuple[str, str, int]] = []
    for label, path in (
        ("prompt", args.prompt_json.resolve()),
        ("verify", args.verify_json.resolve()),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        rels = iter_image_paths(json.loads(path.read_text(encoding="utf-8")), src=path)
        ctr = Counter(rels)
        missing: list[tuple[str, int]] = []
        missing_refs = 0
        for rel, n in ctr.items():
            abs_p = (publish_root / rel).resolve()
            if not abs_p.is_file():
                missing.append((rel, n))
                missing_refs += n
                all_missing.append((label, rel, n))
        ok_refs = len(rels) - missing_refs
        print(
            f"{label}: {path.name} path_refs={len(rels)} unique={len(ctr)} "
            f"ok_refs={ok_refs} missing_refs={missing_refs} missing_unique={len(missing)}"
        )
        lim = args.list_missing
        show = missing if lim == 0 else sorted(missing)[:lim]
        for rel, n in show:
            print(f"  MISSING x{n}: {rel}")
        if lim and len(missing) > lim:
            print(f"  ... and {len(missing) - lim} more distinct missing paths")

    if all_missing:
        distinct = len({(lb, r) for lb, r, _ in all_missing})
        refs = sum(t for _, _, t in all_missing)
        print(f"\nSummary: missing_unique={distinct} missing_path_refs={refs}")
        sys.exit(1)
    print("\nAll image paths exist under publish-root.")
    sys.exit(0)


if __name__ == "__main__":
    main()
