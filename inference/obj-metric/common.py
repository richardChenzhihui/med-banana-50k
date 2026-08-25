from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImagePair:
    model: str
    disease: str
    stem: str
    original_path: Path
    edited_path: Path


DISEASE_ALIASES = {
    "pituitary": "pituitary-adenoma",
}


def default_paths() -> tuple[Path, Path, Path]:
    inference_dir = Path(__file__).resolve().parents[1]
    return inference_dir / "data" / "addition", inference_dir / "result", Path(__file__).resolve().parent


def default_medbanana_root() -> Path:
    """Root of a downloaded Med-Banana-80K tree, taken from $MEDBANANA_ROOT.

    The dataset is not bundled with this repository; download it from
    https://huggingface.co/datasets/RichardChenZH/Med-Banana-80K and point
    MEDBANANA_ROOT at the extracted directory.
    """
    root = os.environ.get("MEDBANANA_ROOT", "")
    if not root:
        raise RuntimeError(
            "MEDBANANA_ROOT is not set. Download Med-Banana-80K from "
            "https://huggingface.co/datasets/RichardChenZH/Med-Banana-80K and export "
            "MEDBANANA_ROOT=/path/to/Med-Banana-80K"
        )
    return Path(root).expanduser()


def metadata_disease_label(manifest_disease: str) -> str:
    disease = manifest_disease.strip().lower()
    return DISEASE_ALIASES.get(disease, disease)


def load_addition_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"addition manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"addition manifest must be a non-empty list: {manifest_path}")
    return manifest


def sample_real_pathology_images(
    medbanana_root: Path,
    addition_manifest_path: Path,
    expected_count: int = 100,
    seed: int = 82,
) -> list[Path]:
    manifest = load_addition_manifest(addition_manifest_path)
    if expected_count != len(manifest):
        raise ValueError(f"expected {expected_count} real pathology images, manifest has {len(manifest)}")

    metadata_path = medbanana_root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")

    quotas = Counter((item["modality"], item["disease"].strip().lower()) for item in manifest)
    exclude_image_ids = {str(item["image_id"]) for item in manifest if item.get("image_id")}

    pools: dict[tuple[str, str], list[Path]] = defaultdict(list)
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") != "original":
                continue
            modality = str(record["modality"])
            disease = str(record.get("disease") or record.get("label") or "").strip().lower()
            image_id = str(record.get("image_id") or "")
            if image_id in exclude_image_ids:
                continue
            for target_modality, target_disease in quotas:
                if modality != target_modality:
                    continue
                if disease != metadata_disease_label(target_disease):
                    continue
                file_name = record.get("file_name")
                if not isinstance(file_name, str) or not file_name.strip():
                    raise ValueError(f"missing file_name for image_id={image_id!r}")
                image_path = (medbanana_root / file_name).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"real pathology image not found: {image_path}")
                pools[(target_modality, target_disease)].append(image_path)
                break

    rng = random.Random(seed)
    selected: list[Path] = []
    seen_paths: set[Path] = set()
    for target_key, need in sorted(quotas.items()):
        bucket = sorted(set(pools.get(target_key, [])))
        if len(bucket) < need:
            raise ValueError(
                f"not enough real pathology images for {target_key}: need {need}, found {len(bucket)}"
            )
        rng.shuffle(bucket)
        picked = bucket[:need]
        for path in picked:
            if path in seen_paths:
                raise ValueError(f"duplicate real pathology image sampled: {path}")
            seen_paths.add(path)
            selected.append(path)

    if len(selected) != expected_count:
        raise ValueError(f"sampled {len(selected)} real pathology images, expected {expected_count}")
    rng.shuffle(selected)
    return selected


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def find_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {root}")
    return sorted(path for path in root.rglob("*") if is_image(path))


def load_originals(original_dir: Path, expected_count: int | None = 100) -> dict[str, Path]:
    originals: dict[str, Path] = {}
    for path in find_images(original_dir):
        if path.stem in originals:
            raise ValueError(f"duplicate original image stem: {path.stem}")
        originals[path.stem] = path
    if expected_count is not None and len(originals) != expected_count:
        raise ValueError(f"expected {expected_count} original images, found {len(originals)} in {original_dir}")
    return originals


def edited_to_original_stem(edited_path: Path, originals: dict[str, Path]) -> str:
    disease = edited_path.parent.name
    stem = edited_path.stem
    candidates = [
        stem,
        stem.removesuffix(f"-{disease}-edit"),
        stem.removesuffix("-edit"),
    ]
    for candidate in candidates:
        if candidate in originals:
            return candidate
    raise ValueError(f"cannot match edited image to original: {edited_path}")


def collect_model_pairs(
    original_dir: Path,
    result_dir: Path,
    expected_count: int | None = 100,
) -> dict[str, list[ImagePair]]:
    originals = load_originals(original_dir, expected_count=expected_count)
    model_pairs: dict[str, list[ImagePair]] = {}
    for model_dir in sorted(path for path in result_dir.iterdir() if path.is_dir()):
        edit_dir = model_dir / "edit"
        if not edit_dir.is_dir():
            continue
        edited_images = find_images(edit_dir)
        if expected_count is not None and len(edited_images) != expected_count:
            raise ValueError(f"expected {expected_count} edit images for {model_dir.name}, found {len(edited_images)}")
        seen: set[str] = set()
        pairs: list[ImagePair] = []
        for edited_path in edited_images:
            stem = edited_to_original_stem(edited_path, originals)
            if stem in seen:
                raise ValueError(f"duplicate edited image for original stem {stem} in model {model_dir.name}")
            seen.add(stem)
            pairs.append(
                ImagePair(
                    model=model_dir.name,
                    disease=edited_path.parent.name,
                    stem=stem,
                    original_path=originals[stem],
                    edited_path=edited_path,
                )
            )
        missing = sorted(set(originals) - seen)
        if missing:
            raise ValueError(f"model {model_dir.name} is missing {len(missing)} originals, first missing: {missing[:5]}")
        model_pairs[model_dir.name] = pairs
    if not model_pairs:
        raise ValueError(f"no model edit directories found under {result_dir}")
    return model_pairs
