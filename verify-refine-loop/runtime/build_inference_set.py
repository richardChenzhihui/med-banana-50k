#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STRAT_MODALITIES = ("brain_mri", "chest_xray", "fundus")
SELECTED_DISEASES: dict[str, tuple[str, ...]] = {
    "brain_mri": ("glioma", "meningioma", "pituitary"),
    "chest_xray": ("pneumothorax", "cardiomegaly", "pneumonia"),
    "fundus": ("amd", "diabetes", "cataract"),
}


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


def _load_rows(jsonl_path: Path, medbanana_root: Path, edit_type: str) -> list[dict[str, Any]]:
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
            if o.get("edit_type") != edit_type:
                continue

            disease = (o.get("disease") or o.get("label") or "").strip().lower()
            allowed = SELECTED_DISEASES.get(str(mod), ())
            if disease not in allowed:
                continue

            source_raw = o.get("source_image")
            if not isinstance(source_raw, str) or not source_raw.strip():
                _die(f"{jsonl_path}:{lineno}: missing source_image")
            if "/images/real/" not in source_raw:
                continue

            edited_raw = o.get("edited_image")
            if not isinstance(edited_raw, str) or not edited_raw.strip():
                _die(f"{jsonl_path}:{lineno}: missing edited_image")

            src_p = _resolve_under_medbanana_root(medbanana_root, source_raw)
            if not src_p.is_file():
                continue

            rows.append(o)
    return rows


def _modality_quotas(n: int) -> dict[str, int]:
    base, rem = divmod(n, len(STRAT_MODALITIES))
    return {mod: base + (1 if i < rem else 0) for i, mod in enumerate(STRAT_MODALITIES)}


def _disease_quotas(total: int, diseases: tuple[str, ...]) -> dict[str, int]:
    base, rem = divmod(total, len(diseases))
    return {dis: base + (1 if i < rem else 0) for i, dis in enumerate(diseases)}


def _select_rows(rows: list[dict[str, Any]], edit_type: str, n: int, seed: int) -> list[dict[str, Any]]:
    mod_quotas = _modality_quotas(n)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mod = str(row["modality"])
        disease = (row.get("disease") or row.get("label") or "").strip().lower()
        pools[(mod, disease)].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for mod in STRAT_MODALITIES:
        mod_quota = mod_quotas[mod]
        dis_quotas = _disease_quotas(mod_quota, SELECTED_DISEASES[mod])
        used_sources: set[str] = set()

        for disease in SELECTED_DISEASES[mod]:
            need = dis_quotas[disease]
            bucket = pools.get((mod, disease), [])
            bucket.sort(key=lambda r: str(r.get("image_id") or r["source_image"]))
            rng.shuffle(bucket)

            picked: list[dict[str, Any]] = []
            for row in bucket:
                src = str(row["source_image"])
                if src in used_sources:
                    continue
                picked.append(row)
                used_sources.add(src)
                if len(picked) == need:
                    break
            if len(picked) < need:
                _die(
                    f"{edit_type} {mod}/{disease}: need {need} unique real sources, got {len(picked)}"
                )
            selected.extend(picked)

    if len(selected) != n:
        _die(f"{edit_type}: selected {len(selected)} rows, expected {n}")

    rng.shuffle(selected)
    return selected


def _make_stem(row: dict[str, Any], used_stems: set[str]) -> str:
    modality = str(row.get("modality") or "unknown").strip()
    disease = (row.get("disease") or row.get("label") or "").strip().lower()
    edit_type = str(row.get("edit_type") or "unknown").strip().lower()
    image_id = str(row.get("image_id") or "").strip()
    if not image_id:
        _die(f"missing image_id for source_image={row.get('source_image')!r}")
    stem = _safe_stem([modality, disease, edit_type, image_id])
    if stem in used_stems:
        _die(f"duplicate stem after sanitize: {stem}")
    used_stems.add(stem)
    return stem


def _short_prompt(edit_type: str, disease: str) -> str:
    label = disease.replace("_", " ").lower()
    if edit_type == "addition":
        return f"add {label} to this image"
    if edit_type == "removal":
        return f"remove {label} from this image"
    _die(f"unsupported edit_type: {edit_type}")
    raise AssertionError


def _write_dataset(
    rows: list[dict[str, Any]],
    edit_type: str,
    medbanana_root: Path,
    dst: Path,
) -> dict[str, Any]:
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.iterdir():
        if old.is_file() and (old.suffix in {".jpg", ".txt"} or old.name == "manifest.json"):
            old.unlink()

    source_kind = "real_healthy" if edit_type == "addition" else "real_diseased"
    used_stems: set[str] = set()
    manifest: list[dict[str, Any]] = []

    for row in rows:
        stem = _make_stem(row, used_stems)
        disease = (row.get("disease") or row.get("label") or "").strip().lower()
        prompt = _short_prompt(edit_type, disease)
        src_p = _resolve_under_medbanana_root(medbanana_root, str(row["source_image"]))

        img_dst = dst / f"{stem}.jpg"
        txt_dst = dst / f"{stem}.txt"
        shutil.copy2(src_p, img_dst)
        txt_dst.write_text(prompt, encoding="utf-8")

        manifest.append(
            {
                "stem": stem,
                "task_type": edit_type,
                "modality": row["modality"],
                "disease": disease,
                "image_id": row.get("image_id"),
                "source_kind": source_kind,
                "source_image": src_p.resolve().as_posix(),
                "prompt": prompt,
                "image_dst": str(img_dst.resolve()),
                "prompt_dst": str(txt_dst.resolve()),
            }
        )

    manifest_path = dst / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    summary = {
        "edit_type": edit_type,
        "count": len(manifest),
        "dst": str(dst.resolve()),
        "manifest": str(manifest_path.resolve()),
        "by_modality": dict(Counter(item["modality"] for item in manifest)),
        "by_disease": dict(Counter(item["disease"] for item in manifest)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verify-refine inference datasets from Med-Banana-80K")
    parser.add_argument(
        "--medbanana-root",
        required=True,
        help=(
            "Root of a downloaded Med-Banana-80K tree containing success_edit.jsonl "
            "and the data/ image directories "
            "(https://huggingface.co/datasets/RichardChenZH/Med-Banana-80K)"
        ),
    )
    parser.add_argument(
        "--out-root",
        default=str(Path(__file__).resolve().parents[2] / "inference" / "data"),
        help="Output root; writes addition/ and removal/ subdirectories",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n <= 0:
        _die(f"--n must be positive, got {args.n}")

    medbanana_root = Path(args.medbanana_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    jsonl_path = medbanana_root / "success_edit.jsonl"
    if not jsonl_path.is_file():
        _die(f"missing {jsonl_path}")

    summaries = []
    for edit_type in ("addition", "removal"):
        rows = _load_rows(jsonl_path, medbanana_root, edit_type)
        if not rows:
            _die(f"no eligible rows for edit_type={edit_type}")
        selected = _select_rows(rows, edit_type, args.n, args.seed + (0 if edit_type == "addition" else 1000))
        summary = _write_dataset(selected, edit_type, medbanana_root, out_root / edit_type)
        summaries.append(summary)

    print(json.dumps({"datasets": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
