#!/usr/bin/env python3
"""
Build two VLM SFT JSON datasets from Med-Banana edit/remove conversation logs.

Restores API-side instruction strings from remove_disease.py / add_disease.py
(system_instruction / verification_instruction) into `messages[0].role == system`.

Default outputs under Med-Banana/finetune:
  - vlm_prompt_generation_dataset.json
  - vlm_verification_dataset.json
  - unresolved_images.json (only if any path could not be resolved)

Paths in `images` are relative to the outer release folder passed as `--medbanana-root`.
Images are resolved under `--image-data-root` (default: `<root>/<release-dir>/data`).
Optional `metadata.jsonl` (default `<root>/<release-dir>/metadata.jsonl` if present) provides
(modality, task_key) -> real path per reorganize_to_hf when basename index misses.

This is a maintainer tool: it rebuilds the verifier / refiner SFT datasets from the raw
edit/remove conversation logs, which are not part of the public release. The published
dataset lives at https://huggingface.co/datasets/RichardChenZH/Med-Banana-80K.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Name of the release folder inside the raw publish tree. It is also the prefix written
# into every emitted `images` path, so it must match the on-disk layout. Override with
# --release-dir-name when the raw tree uses a different folder name.
RELEASE_DIR_NAME = "Med-Banana-50K"

# Short user triggers (API had no separate user role; Gemini round-1 prompt was text-only, but
# this builder attaches the original image for every prompt sample so VLM training always receives vision.)
USER_TRIGGER_PROMPT_WITH_IMAGE = (
    "<image>\nApply the system instruction and output only the editing prompt."
)
USER_TRIGGER_VERIFY_WITH_IMAGE = (
    "<image>\nApply the system instruction and output only the JSON evaluation object."
)


# --- Instruction strings: copied from remove_disease.py / add_disease.py (must stay in sync). ---


def make_remove_initial_prompt_instruction(disease_name: str, dataset_type: str) -> str:
    return f"""You are a medical image editing expert. Generate a clear, concise prompt to edit a {dataset_type} image showing {disease_name} to make it appear normal (healthy).

Requirements:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow common sense and medical plausibility
4. Be specific about what to remove/modify to make it normal

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image noise, grain texture, and contrast
- Do NOT alter device artifacts, watermarks, grid lines, or annotations
- Keep the same imaging characteristics (brightness, sharpness, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, or annotations
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures
- NO introducing new pathologies
- NO obvious deepfake artifacts

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify areas directly related to removing {disease_name}
- Keep all other anatomical structures, textures, and background UNCHANGED
- Minimal intervention: change as little as possible to make it appear healthy

Return ONLY the editing prompt in English, no explanations."""


def make_add_initial_prompt_instruction(disease_name: str, dataset_type: str) -> str:
    return f"""You are a medical image editing expert. Generate a clear, concise prompt to edit a normal {dataset_type} image to show {disease_name}.

Requirements:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow common sense and medical plausibility
4. Be specific about what to add/modify

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image noise, grain texture, and contrast
- Do NOT alter device artifacts, watermarks, grid lines, or annotations
- Keep the same imaging characteristics (brightness, sharpness, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, or annotations
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures
- NO non-target pathologies
- NO obvious deepfake artifacts

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify areas directly related to {disease_name}
- Keep all other anatomical structures, textures, and background UNCHANGED
- Minimal intervention: change as little as possible to show the disease

Return ONLY the editing prompt in English, no explanations."""


def make_remove_update_prompt_instruction(
    disease_name: str, dataset_type: str, history_text: str
) -> str:
    return f"""You are a medical image editing expert. Multiple previous editing attempts to make this image normal have failed. You need to analyze ALL previous attempts and generate a BETTER prompt.

HISTORY OF ALL PREVIOUS ATTEMPTS:
{history_text}

Looking at the ORIGINAL diseased image ({disease_name}) and analyzing the patterns of failures above, generate an IMPROVED editing prompt to make it appear normal and healthy.

ANALYSIS REQUIREMENTS:
1. Identify common issues across multiple attempts
2. Learn from what didn't work in previous rounds
3. Avoid repeating the same mistakes
4. Address ALL verification issues mentioned in the history

BASIC REQUIREMENTS:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow common sense and medical plausibility
4. Be specific about what to remove/modify to make it normal

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image noise, grain texture, and contrast
- Do NOT alter device artifacts, watermarks, grid lines, or annotations
- Keep the same imaging characteristics (brightness, sharpness, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, or annotations
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures
- NO introducing new pathologies
- NO obvious deepfake artifacts

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify areas directly related to removing {disease_name}
- Keep all other anatomical structures, textures, and background UNCHANGED
- Minimal intervention: change as little as possible to make it appear healthy

Return ONLY the improved editing prompt in English, no explanations."""


def make_add_update_prompt_instruction(
    disease_name: str, dataset_type: str, history_text: str
) -> str:
    return f"""You are a medical image editing expert. Multiple previous editing attempts have failed. You need to analyze ALL previous attempts and generate a BETTER prompt.

HISTORY OF ALL PREVIOUS ATTEMPTS:
{history_text}

Looking at the ORIGINAL image and analyzing the patterns of failures above, generate an IMPROVED editing prompt to add {disease_name} to this {dataset_type} image.

ANALYSIS REQUIREMENTS:
1. Identify common issues across multiple attempts
2. Learn from what didn't work in previous rounds
3. Avoid repeating the same mistakes
4. Address ALL verification issues mentioned in the history

BASIC REQUIREMENTS:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow common sense and medical plausibility
4. Be specific about what to add/modify

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image noise, grain texture, and contrast
- Do NOT alter device artifacts, watermarks, grid lines, or annotations
- Keep the same imaging characteristics (brightness, sharpness, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, or annotations
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures
- NO non-target pathologies
- NO obvious deepfake artifacts

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify areas directly related to {disease_name}
- Keep all other anatomical structures, textures, and background UNCHANGED
- Minimal intervention: change as little as possible to show the disease

Return ONLY the improved editing prompt in English, no explanations."""


def make_remove_verification_instruction(disease_name: str, dataset_type: str) -> str:
    return f"""You are a medical image verification expert. This {dataset_type} image was originally showing {disease_name} and has been edited to appear normal.

IMPORTANT: Take your time to think carefully. Medical image editing is challenging, and minor imperfections are acceptable as long as the overall goal is achieved. Be thoughtful and balanced in your evaluation - don't reject an image for trivial issues.

Check these aspects:
1. Has disease: Does the image still show signs of {disease_name} or any other disease? (should be False for normal) (Consider: Minor residual traces are acceptable if the image appears largely normal)
2. Structure reasonable: Are the anatomical structures reasonable and correct? (Consider: Minor artifacts are acceptable if the overall anatomy is preserved)
3. Looks realistic: Does it look like a real medical image? (Consider: Some editing traces are inevitable; focus on whether it could pass as a real medical image to non-experts)

Additional verification for image fidelity:
- Check if the image preserves natural noise/grain texture (minor changes are acceptable)
- Check if there are unnatural sharp edges or boundaries (slight artifacts are tolerable if not obvious)
- Check if there are added text, repetitive structures, or deepfake artifacts (focus on major issues, not minor imperfections)
- Check if modifications are minimal (only disease removal, other areas unchanged; some collateral changes are acceptable)

Before deciding, ask yourself:
- Would this image be useful for the intended purpose despite minor flaws?
- Are the issues critical or just cosmetic?
- Does the image achieve the main goal of appearing normal/healthy?

Return your evaluation in this JSON format:
{{
    "qualified": true/false,
    "has_disease": true/false,
    "structure_reasonable": true/false,
    "looks_realistic": true/false,
    "reason": "detailed explanation (mention both strengths and weaknesses; explain your reasoning for acceptance or rejection)"
}}

Only qualified if has_disease is FALSE and the other two are TRUE (normal, reasonable structure, realistic) AND no MAJOR fidelity issues detected. Minor imperfections are acceptable."""


def make_add_verification_instruction(disease_name: str, dataset_type: str) -> str:
    return f"""You are a medical image verification expert. Evaluate if this {dataset_type} image shows {disease_name}.

IMPORTANT: Take your time to think carefully. Medical image editing is challenging, and minor imperfections are acceptable as long as the overall goal is achieved. Be thoughtful and balanced in your evaluation - don't reject an image for trivial issues.

Check these aspects:
1. Has disease: Does the image show signs of {disease_name}? (Consider: Are the disease features visible and recognizable, even if not perfect?)
2. Structure reasonable: Are the anatomical structures reasonable and correct? (Consider: Minor artifacts are acceptable if the overall anatomy is preserved)
3. Looks realistic: Does it look like a real medical image? (Consider: Some editing traces are inevitable; focus on whether it could pass as a real medical image to non-experts)

Additional verification for image fidelity:
- Check if the image preserves natural noise/grain texture (minor changes are acceptable)
- Check if there are unnatural sharp edges or boundaries (slight artifacts are tolerable if not obvious)
- Check if there are added text, repetitive structures, or deepfake artifacts (focus on major issues, not minor imperfections)
- Check if modifications are minimal (only disease-related changes; some collateral changes are acceptable)

Before deciding, ask yourself:
- Would this image be useful for the intended purpose despite minor flaws?
- Are the issues critical or just cosmetic?
- Does the image achieve the main goal of showing {disease_name}?

Return your evaluation in this JSON format:
{{
    "qualified": true/false,
    "has_disease": true/false,
    "structure_reasonable": true/false,
    "looks_realistic": true/false,
    "reason": "detailed explanation (mention both strengths and weaknesses; explain your reasoning for acceptance or rejection)"
}}

Only qualified if ALL three aspects are true AND no MAJOR fidelity issues detected. Minor imperfections are acceptable."""


@dataclass(frozen=True)
class SourceLoc:
    dataset_base: str
    op: str  # "edit" | "remove"
    storage_folder: str  # e.g. brain-tumor-1024x1024-edit


def dataset_base_to_modality(dataset_base: str) -> str:
    n = dataset_base.lower()
    if "brain" in n or "tumor" in n:
        return "brain_mri"
    if "mimic" in n or "chest" in n:
        return "chest_xray"
    if "odir" in n or "fundus" in n:
        return "fundus"
    raise ValueError(f"Unknown dataset_base for modality: {dataset_base!r}")


def get_dataset_type(dataset_base: str) -> str:
    name = dataset_base.lower()
    if "mimic" in name or "chest" in name:
        return "chest X-ray"
    if "brain" in name or "tumor" in name:
        return "brain MRI"
    if "odir" in name or "fundus" in name:
        return "fundus"
    return "medical"


def parse_main_folder(folder_name: str) -> SourceLoc | None:
    if folder_name.endswith("-edit"):
        return SourceLoc(folder_name[: -len("-edit")], "edit", folder_name)
    if folder_name.endswith("-remove"):
        return SourceLoc(folder_name[: -len("-remove")], "remove", folder_name)
    return None


def parse_failed_folder(folder_name: str) -> SourceLoc | None:
    if folder_name.endswith("-edit-failed"):
        return SourceLoc(folder_name[: -len("-edit-failed")], "edit", folder_name)
    if folder_name.endswith("-remove-failed"):
        return SourceLoc(folder_name[: -len("-remove-failed")], "remove", folder_name)
    return None


def norm_basename(name: str) -> str:
    p = Path(name)
    return p.name


def parse_filename_info(stem: str, is_failed: bool, is_remove: bool) -> tuple[str, int | None]:
    """
    Same as the release-side scripts/reorganize_to_hf.parse_filename_info:
    strip -failed-<round>, then trailing -edit / -remove -> task_key.
    (is_remove is kept for signature parity; not used in stripping.)
    """
    clean_stem = stem
    round_num: int | None = None
    if is_failed and "-failed-" in clean_stem:
        parts = clean_stem.split("-failed-")
        if len(parts) > 1:
            try:
                round_num = int(parts[-1])
            except ValueError:
                pass
            clean_stem = parts[0]
    for suffix in ("-edit", "-remove"):
        if clean_stem.endswith(suffix):
            clean_stem = clean_stem[: -len(suffix)]
            break
    return clean_stem, round_num


def edited_row_in_metadata(file_name: str) -> bool:
    fn = file_name.replace("\\", "/")
    return (
        "/images/addition/" in fn
        or "/images/removal/" in fn
        or "/images/failed/" in fn
    )


def source_image_match_score(task_key: str, rel: str) -> int:
    """Prefer source_image whose basename aligns with task_key image id (before last '-disease')."""
    pid = task_key.rsplit("-", 1)[0]
    name = Path(rel).name
    stem = Path(name).stem
    if stem == pid:
        return 4
    if name.startswith(pid):
        return 3
    if pid in stem:
        return 2
    if stem in pid:
        return 1
    return 0


def load_task_source_real_from_metadata_jsonl(path: Path) -> dict[tuple[str, str], str]:
    """
    Invert reorganize_to_hf: for each edited-image row, map (modality, task_key) ->
    path relative to outer publish root (<release-dir>/...).
    """
    out: dict[tuple[str, str], str] = {}
    scores: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fn = obj.get("file_name")
            if not isinstance(fn, str) or not edited_row_in_metadata(fn):
                continue
            modality = obj.get("modality")
            edit_type = obj.get("edit_type")
            if not isinstance(modality, str) or edit_type not in ("addition", "removal"):
                continue
            stem = Path(fn).stem
            is_remove = edit_type == "removal"
            is_failed = "/images/failed/" in fn.replace("\\", "/")
            task_key, _ = parse_filename_info(stem, is_failed, is_remove)
            si = obj.get("source_image")
            if not isinstance(si, str) or not si.strip():
                continue
            rel = f"{RELEASE_DIR_NAME}/" + si.strip().lstrip("/")
            k = (modality, task_key)
            sc = source_image_match_score(task_key, rel)
            if k not in out:
                out[k] = rel
                scores[k] = sc
                continue
            if out[k] == rel:
                continue
            prev = scores[k]
            if sc > prev:
                out[k] = rel
                scores[k] = sc
            elif sc == prev:
                out[k] = min(out[k], rel)
            # sc < prev: keep existing
    return out


def build_basename_index(search_roots: Iterable[Path]) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = defaultdict(list)
    for root in search_roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in IMAGE_SUFFIXES:
                idx[p.name].append(p)
    return idx


def pick_with_hints(cands: list[Path], *, must_substrings: list[str]) -> Path | None:
    cur = cands
    for sub in must_substrings:
        if not sub:
            continue
        nxt = [p for p in cur if sub in str(p)]
        if nxt:
            cur = nxt
    if len(cur) == 1:
        return cur[0]
    return None


def rel_to_publish(publish_root: Path, abs_path: Path) -> str:
    return str(abs_path.resolve().relative_to(publish_root)).replace("\\", "/")


def candidate_original_names(loc: SourceLoc, disease: str, image_filename: str) -> list[str]:
    base = norm_basename(image_filename)
    stem = Path(base).stem
    names: list[str] = []
    for ext in (".jpeg", ".jpg", ".JPEG", ".JPG", ".png", ".PNG"):
        names.append(f"{stem}{ext}")
    if base not in names:
        names.insert(0, base)
    if loc.op == "edit":
        for ext in (".jpeg", ".jpg"):
            names.append(f"normal_{stem}{ext}")
    if loc.op == "remove":
        for ext in (".jpeg", ".jpg"):
            names.append(f"{disease}_{stem}{ext}")
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def collect_paths_for_names(idx: dict[str, list[Path]], names: Iterable[str]) -> list[Path]:
    acc: list[Path] = []
    for n in names:
        acc.extend(idx.get(n, []))
    return acc


def pick_best_original(
    cands: list[Path], loc: SourceLoc, disease: str, image_filename: str
) -> Path | None:
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    want = norm_basename(image_filename)

    def score(p: Path) -> tuple[int, int]:
        n = p.name
        s = 0
        if n == want:
            s += 200
        if loc.op == "remove" and n.startswith(f"{disease}_"):
            s += 100
        if want in n:
            s += 20
        return s, -len(str(p))

    return max(cands, key=score)


def resolve_original_hub(
    publish_root: Path,
    data_root: Path,
    idx: dict[str, list[Path]],
    modality: str,
    loc: SourceLoc,
    disease: str,
    image_filename: str,
    task_key: str,
    task_source_real: dict[tuple[str, str], str],
) -> tuple[str | None, list[str]]:
    tried: list[str] = []
    real_marker = f"{modality}/images/real"
    names = candidate_original_names(loc, disease, image_filename)
    # real/ is flat per reorganize_to_hf; remove rows may need disease in path; edit uses conversation basename only.
    hints: list[str] = [modality, "images/real"]
    if loc.op == "remove":
        hints.append(disease)
    cands = [
        p
        for p in collect_paths_for_names(idx, names)
        if real_marker in str(p).replace("\\", "/")
    ]
    if loc.op == "edit":
        prefer_plain = [p for p in cands if not p.name.startswith("normal_")]
        if prefer_plain:
            cands = prefer_plain
    tried.extend(names[:8])
    if cands:
        picked = pick_with_hints(cands, must_substrings=hints)
        chosen = picked if picked is not None else pick_best_original(cands, loc, disease, image_filename)
        return rel_to_publish(publish_root, chosen), tried

    meta_rel = task_source_real.get((modality, task_key))
    if meta_rel:
        meta_key = f"metadata:{meta_rel}"
        tried.append(meta_key)
        abs_p = publish_root / meta_rel.replace("\\", "/")
        if abs_p.is_file():
            return meta_rel.replace("\\", "/"), tried
    return None, tried


def edited_hub_filename(
    loc: SourceLoc, stem: str, disease: str, round_num: int, qualified: bool
) -> str:
    if loc.op == "edit":
        if qualified:
            return f"{stem}-{disease}-edit.jpeg"
        return f"{stem}-{disease}-edit-failed-{round_num}.jpeg"
    if qualified:
        return f"{stem}-{disease}-remove.jpeg"
    return f"{stem}-{disease}-remove-failed-{round_num}.jpeg"


def edited_hub_subdir(loc: SourceLoc, qualified: bool) -> str:
    if loc.op == "edit":
        return "addition" if qualified else "failed"
    return "removal" if qualified else "failed"


def expected_hub_edited_relpath(
    modality: str, loc: SourceLoc, stem: str, disease: str, round_num: int, qualified: bool
) -> str:
    sub = edited_hub_subdir(loc, qualified)
    fn = edited_hub_filename(loc, stem, disease, round_num, qualified)
    return f"{RELEASE_DIR_NAME}/data/{modality}/images/{sub}/{fn}"


def resolve_edited_hub(
    publish_root: Path,
    data_root: Path,
    idx: dict[str, list[Path]],
    modality: str,
    loc: SourceLoc,
    stem: str,
    disease: str,
    round_num: int,
    qualified: bool,
) -> tuple[str | None, list[str]]:
    rel = expected_hub_edited_relpath(modality, loc, stem, disease, round_num, qualified)
    tried: list[str] = [rel]
    abs_p = publish_root / rel
    if abs_p.is_file():
        return rel.replace("\\", "/"), tried

    fname = Path(rel).name
    hints = [
        modality,
        edited_hub_subdir(loc, qualified),
        f"{stem}-{disease}",
        str(round_num),
    ]
    cands = idx.get(fname, [])
    picked = pick_with_hints(cands, must_substrings=hints)
    if picked is None and len(cands) == 1:
        picked = cands[0]
    if picked is None:
        return None, tried + [rel_to_publish(publish_root, p) for p in cands[:12]]
    return rel_to_publish(publish_root, picked), tried


def wrap_thinking(body: str) -> str:
    text = (body or "").strip()
    return f"<think>\n{text}\n</think>"


def format_prompt_history_like_runtime(
    prev_rounds: list[dict[str, Any]], current_round: int
) -> str:
    """
    Match remove_disease.py / add_disease.py update_prompt(): enumerate(history, 1) for Attempt i,
    not conversation round numbers.
    """
    entries: list[tuple[str, dict[str, Any]]] = []
    for pr in prev_rounds:
        if not isinstance(pr, dict):
            continue
        r = pr.get("round")
        if r is None or int(r) >= int(current_round):
            continue
        gp = pr.get("generate_prompt") or {}
        prompt = gp.get("prompt")
        ver = pr.get("verification") or {}
        if prompt is None or "error" in ver:
            continue
        if not all(
            k in ver for k in ("has_disease", "structure_reasonable", "looks_realistic", "reason")
        ):
            continue
        entries.append((str(prompt), ver))
    history_text = ""
    for i, (prompt, ver) in enumerate(entries, 1):
        history_text += f"""
Attempt {i}:
  Prompt: {prompt}
  Verification Result:
    - Has disease: {ver['has_disease']}
    - Structure reasonable: {ver['structure_reasonable']}
    - Looks realistic: {ver['looks_realistic']}
    - Reason: {ver['reason']}
"""
    return history_text


def verification_json_blob(ver: dict[str, Any]) -> str:
    core = {
        "qualified": bool(ver.get("qualified")),
        "has_disease": bool(ver.get("has_disease")),
        "structure_reasonable": bool(ver.get("structure_reasonable")),
        "looks_realistic": bool(ver.get("looks_realistic")),
        "reason": ver.get("reason", ""),
    }
    return json.dumps(core)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    global RELEASE_DIR_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--medbanana-root",
        type=Path,
        required=True,
        help="Outer raw publish folder (contains <release-dir>-raw/ and <release-dir>/data/)",
    )
    parser.add_argument(
        "--release-dir-name",
        default=RELEASE_DIR_NAME,
        help="Release folder name inside --medbanana-root, also used as the prefix of "
        f"emitted image paths (default: {RELEASE_DIR_NAME})",
    )
    parser.add_argument(
        "--image-data-root",
        type=Path,
        default=None,
        help="Root that holds modality folders (brain_mri,chest_xray,fundus). "
        "Default: <medbanana-root>/<release-dir>/data",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "finetune",
        help="Output directory for dataset JSON files",
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        default=None,
        help="reorganize_to_hf metadata.jsonl. Default: <medbanana-root>/<release-dir>/metadata.jsonl if it exists.",
    )
    parser.add_argument(
        "--wrap-thinking",
        action="store_true",
        help="Prefix assistant with thinking_summary wrapped in <think> tags (legacy).",
    )
    args = parser.parse_args()

    RELEASE_DIR_NAME = args.release_dir_name

    publish_root: Path = args.medbanana_root.resolve()
    raw_root = publish_root / f"{RELEASE_DIR_NAME}-raw"
    data_root: Path = (
        args.image_data_root.resolve()
        if args.image_data_root is not None
        else (publish_root / RELEASE_DIR_NAME / "data")
    )
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Image data root not found: {data_root}. Set --image-data-root."
        )
    meta_path: Path | None = args.metadata_jsonl
    if meta_path is not None:
        meta_path = meta_path.resolve()
        if not meta_path.is_file():
            raise FileNotFoundError(f"--metadata-jsonl not found: {meta_path}")
    else:
        cand_meta = publish_root / RELEASE_DIR_NAME / "metadata.jsonl"
        meta_path = cand_meta if cand_meta.is_file() else None
    task_source_real: dict[tuple[str, str], str] = (
        load_task_source_real_from_metadata_jsonl(meta_path) if meta_path else {}
    )
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    main_dirs = [
        "brain-tumor-1024x1024-edit",
        "brain-tumor-1024x1024-remove",
        "MIMIC_single_disease_selection_dim1024_1k_per_class-edit",
        "MIMIC_single_disease_selection_dim1024_1k_per_class-remove",
        "ODIR-5K-classified-1024x1024-edit",
        "ODIR-5K-classified-1024x1024-remove",
    ]
    failed_dirs = [
        "brain-tumor-1024x1024-edit-failed",
        "brain-tumor-1024x1024-remove-failed",
        "MIMIC_single_disease_selection_dim1024_1k_per_class-edit-failed",
        "MIMIC_single_disease_selection_dim1024_1k_per_class-remove-failed",
        "ODIR-5K-classified-1024x1024-edit-failed",
        "ODIR-5K-classified-1024x1024-remove-failed",
    ]

    conversations: dict[tuple[str, str], dict[str, Any]] = {}
    for folder in main_dirs:
        loc = parse_main_folder(folder)
        if loc is None:
            raise RuntimeError(f"Bad main folder {folder}")
        path = raw_root / folder / "all_conversations.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = load_json(path)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in {path}")
        for task_key, conv in data.items():
            conversations[(folder, str(task_key))] = conv

    for folder in failed_dirs:
        loc = parse_failed_folder(folder)
        if loc is None:
            raise RuntimeError(f"Bad failed folder {folder}")
        path = raw_root / folder / "failed_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = load_json(path)
        if not isinstance(rows, list):
            raise TypeError(f"Expected list in {path}")
        main_folder = f"{loc.dataset_base}-{loc.op}"
        for row in rows:
            task_key = str(row.get("task", ""))
            conv = row.get("full_conversation")
            if not task_key or not isinstance(conv, dict):
                continue
            k = (main_folder, task_key)
            if k not in conversations:
                conversations[k] = conv

    idx = build_basename_index([data_root])
    if meta_path:
        print(
            f"metadata.jsonl: {meta_path} -> {len(task_source_real)} (modality, task_key) source_image entries"
        )

    prompt_rows: list[dict[str, Any]] = []
    verify_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for (storage_folder, task_key), conv in sorted(conversations.items()):
        loc = parse_main_folder(storage_folder)
        if loc is None:
            continue
        disease = str(conv.get("disease", ""))
        image_fn = norm_basename(str(conv.get("image", "")))
        stem = Path(image_fn).stem
        modality = dataset_base_to_modality(loc.dataset_base)
        ds_type = get_dataset_type(loc.dataset_base)
        rounds = conv.get("rounds") or []
        if not isinstance(rounds, list):
            continue

        for rd in rounds:
            if not isinstance(rd, dict):
                continue
            rnum = rd.get("round")
            if rnum is None:
                continue
            gp = rd.get("generate_prompt")
            if not isinstance(gp, dict):
                continue
            thinking_raw = gp.get("thinking_summary")
            prompt_out = gp.get("prompt")
            if prompt_out is None or not str(prompt_out).strip():
                continue
            prompt_out = str(prompt_out)
            if args.wrap_thinking:
                if thinking_raw is None or not str(thinking_raw).strip():
                    continue
                thinking = str(thinking_raw)

            is_initial = int(rnum) <= 1
            if is_initial:
                if loc.op == "remove":
                    system_prompt = make_remove_initial_prompt_instruction(disease, ds_type)
                else:
                    system_prompt = make_add_initial_prompt_instruction(disease, ds_type)
            else:
                hist = format_prompt_history_like_runtime(rounds, int(rnum))
                if not hist.strip():
                    continue
                if loc.op == "remove":
                    system_prompt = make_remove_update_prompt_instruction(
                        disease, ds_type, hist
                    )
                else:
                    system_prompt = make_add_update_prompt_instruction(disease, ds_type, hist)

            # Round-1 API call was text-only, but VLM SFT needs the same original image as update_prompt.
            resolved, tried = resolve_original_hub(
                publish_root,
                data_root,
                idx,
                modality,
                loc,
                disease,
                image_fn,
                task_key,
                task_source_real,
            )
            if resolved is None:
                unresolved.append(
                    {
                        "dataset": storage_folder,
                        "task_key": task_key,
                        "round": rnum,
                        "kind": "prompt_original_image",
                        "basename": image_fn,
                        "modality": modality,
                        "tried": tried,
                    }
                )
            else:
                user_txt = USER_TRIGGER_PROMPT_WITH_IMAGE
                images_list = [resolved.replace("\\", "/")]
                assistant = (
                    f"{wrap_thinking(thinking)}\n\n{prompt_out}"
                    if args.wrap_thinking
                    else prompt_out
                )
                prompt_rows.append(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_txt},
                            {"role": "assistant", "content": assistant},
                        ],
                        "images": images_list,
                        "meta": {
                            "task_key": task_key,
                            "storage_folder": storage_folder,
                            "dataset_base": loc.dataset_base,
                            "op": loc.op,
                            "round": int(rnum),
                            "conversation_status": conv.get("status"),
                            "prompt_stage": "initial" if is_initial else "update",
                        },
                    }
                )

            ver = rd.get("verification")
            if not isinstance(ver, dict) or "error" in ver:
                continue
            if "qualified" not in ver:
                continue
            v_think_raw = ver.get("thinking_summary")
            if args.wrap_thinking:
                if v_think_raw is None or not str(v_think_raw).strip():
                    continue
                v_think = str(v_think_raw)

            qualified = bool(ver.get("qualified"))
            expected_rel = expected_hub_edited_relpath(
                modality, loc, stem, disease, int(rnum), qualified
            )
            resolved_ed, tried_ed = resolve_edited_hub(
                publish_root,
                data_root,
                idx,
                modality,
                loc,
                stem,
                disease,
                int(rnum),
                qualified,
            )
            if resolved_ed is None:
                unresolved.append(
                    {
                        "dataset": storage_folder,
                        "task_key": task_key,
                        "round": rnum,
                        "kind": "verify_edited_image",
                        "basename": Path(expected_rel).name,
                        "expected": expected_rel.replace("\\", "/"),
                        "modality": modality,
                        "tried": tried_ed,
                    }
                )
                continue
            ed_use = resolved_ed

            if loc.op == "remove":
                v_system = make_remove_verification_instruction(disease, ds_type)
            else:
                v_system = make_add_verification_instruction(disease, ds_type)
            v_assistant = (
                f"{wrap_thinking(v_think)}\n\n{verification_json_blob(ver)}"
                if args.wrap_thinking
                else verification_json_blob(ver)
            )
            verify_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": v_system},
                        {"role": "user", "content": USER_TRIGGER_VERIFY_WITH_IMAGE},
                        {"role": "assistant", "content": v_assistant},
                    ],
                    "images": [ed_use.replace("\\", "/")],
                    "meta": {
                        "task_key": task_key,
                        "storage_folder": storage_folder,
                        "dataset_base": loc.dataset_base,
                        "op": loc.op,
                        "round": int(rnum),
                        "conversation_status": conv.get("status"),
                    },
                }
            )

    prompt_path = out_dir / "vlm_prompt_generation_dataset.json"
    verify_path = out_dir / "vlm_verification_dataset.json"
    unres_path = out_dir / "unresolved_images.json"

    with prompt_path.open("w", encoding="utf-8") as f:
        json.dump(prompt_rows, f, ensure_ascii=False, indent=2)
    with verify_path.open("w", encoding="utf-8") as f:
        json.dump(verify_rows, f, ensure_ascii=False, indent=2)
    if unresolved:
        with unres_path.open("w", encoding="utf-8") as f:
            json.dump(unresolved, f, ensure_ascii=False, indent=2)

    print(f"prompt samples: {len(prompt_rows)} -> {prompt_path}")
    print(f"verify samples: {len(verify_rows)} -> {verify_path}")
    print(f"unresolved: {len(unresolved)}")
    if unresolved:
        print(f"details: {unres_path}")


if __name__ == "__main__":
    main()
