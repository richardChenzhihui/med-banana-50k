#!/usr/bin/env python3
"""Simple Gradio app for blind human scoring of Med-Banana edit results."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import socket
import statistics
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr


APP_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = APP_DIR.parent
SOURCE_ADDITION_DIR = INFERENCE_DIR / "data" / "addition"
SOURCE_RESULT_DIR = INFERENCE_DIR / "result"
DATA_DIR = APP_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULT_JSON = APP_DIR / "result.json"
RESULT_JSONL = APP_DIR / "result.jsonl"

DEFAULT_MODELS = (
    "openai",
    "gemini",
    "qwen-ori",
    "firered-ori",
    "med-banana-qwen-r10",
    "med-banana-firered-r10",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCORE_KEYS = (
    "target_pathology_instruction",
    "biological_pathological_plausibility",
    "modality_anatomical_consistency",
    "imaging_physics_counterfactual_fidelity",
)
SCORE_LABELS = {
    "target_pathology_instruction": "Target lesion / instruction fulfilment",
    "biological_pathological_plausibility": "Biological and pathological plausibility",
    "modality_anatomical_consistency": "Modality and anatomical consistency",
    "imaging_physics_counterfactual_fidelity": "Imaging realism / counterfactual fidelity",
}
SCORE_HINTS = {
    "target_pathology_instruction": "Is the target disease clearly present and broadly consistent with the editing instruction?",
    "biological_pathological_plausibility": "Is the lesion appearance consistent with medical knowledge, with no obvious biological contradiction?",
    "modality_anatomical_consistency": "Are the major anatomical structures and modality characteristics preserved, with a plausible location?",
    "imaging_physics_counterfactual_fidelity": "Do texture, noise, edges, blending and the non-lesion regions look natural and real?",
}
SCORE_CHOICES = [str(i) for i in range(11)]
DISEASE_FEATURE_HINTS = {
    "amd": "AMD is judged mainly on macular changes: drusen, pigmentary disturbance and geographic atrophy may appear, while wet lesions commonly show exudate, haemorrhage or a greyish-white neovascular membrane. Check that the lesion centres on the macula, affects the central retinal layers, and has boundaries and colour that read as a fundus lesion rather than randomly scattered bright spots or surface blemishes.",
    "cataract": "In fundus images cataract typically shows as global darkening, haze and reduced contrast, with softened vessel and optic-disc boundaries; severe cases add central or diffuse opacity. A valid edit should look like optical degradation from lens-media opacity affecting global clarity, not a new focal lesion or hard-edged patch pasted onto the retinal surface.",
    "diabetes": "Diabetic retinopathy commonly shows microaneurysms, dot-and-blot haemorrhages, hard exudates and cotton-wool spots, distributed along the vascular arcades or the posterior pole; severe cases add neovascularisation and extensive exudation. Check that the lesions relate to the retinal vascular structure, that their number and size look natural, and that the scatter and depth are plausible.",
    "glioma": "On brain MRI, glioma typically appears as an intra-axial mass with abnormal signal and possibly infiltrative, indistinct margins, often with surrounding oedema, mass effect or effaced sulci; high-grade lesions may enhance heterogeneously or show necrosis. A valid edit sits within brain parenchyma, with signal intensity, edge transition and local anatomical compression consistent with the original image.",
    "meningioma": "Meningioma is usually an extra-axial, dura-based mass with fairly well-defined margins that may compress adjacent brain, commonly attached to the dura on a broad base, sometimes with a dural tail and fairly uniform enhancement. It should sit near the meninges, parafalcine region or skull base, look extrinsic, and relate plausibly to the inner table of the skull rather than being embedded deep in brain parenchyma.",
    "pituitary": "Pituitary lesions sit in the sellar region, presenting as an enlarged pituitary or an intrasellar/suprasellar mass that may compress the optic chiasm, deviate the stalk or involve the cavernous sinus. A valid edit stays near the sellar midline, with size, margins and signal consistent with the surrounding cisterns, sella turcica and cavernous sinus, and must not drift into ordinary brain parenchyma.",
    "cardiomegaly": "On chest X-ray, cardiomegaly shows mainly as an enlarged cardiac silhouette and raised cardiothoracic ratio, with the cardiac border expanding outward — especially the left border or the mediastinal silhouette overall. Lung fields, ribs and diaphragm should stay largely intact. A valid edit changes the contour and proportion of the cardiac shadow while keeping projection logic, rather than adding a focal intrapulmonary opacity or reorienting the whole film.",
    "pneumonia": "On chest X-ray, pneumonia commonly shows patchy or segmental consolidation, possibly with air bronchograms, distributed lobar, segmental or scattered across both lungs. Check that the opacity lies within the lung fields, that its density matches inflammatory infiltration, that it follows the lung markings and thoracic projection, and that the cardiac silhouette, ribs and diaphragm are preserved.",
    "pneumothorax": "On chest X-ray, pneumothorax typically shows a pleural line with absent lung markings peripheral to it and increased lucency on the affected side; severe cases collapse the lung toward the hilum with possible mediastinal shift. A valid edit renders the pleural line, the avascular peripheral zone and the retracted lung edge, usually near the chest wall or apex — not a plain white consolidation, a random transparent block, or an intrapulmonary patch.",
}
APP_CSS = """
body, .gradio-container {
    background: #ffffff !important;
    color: #111827 !important;
}
.gradio-container {
    max-width: 1380px !important;
    margin: 0 auto !important;
    padding: 5px 10px !important;
}
.app-title h2, .app-title p {
    margin: 0 !important;
}
.app-title h3 {
    margin: 0 !important;
    font-size: 16px !important;
    font-weight: 650 !important;
    line-height: 1.2 !important;
}
.topbar {
    align-items: center !important;
    gap: 8px !important;
    margin-bottom: 4px !important;
}
.meta-line, .progress-line, .status-line {
    font-size: 13px !important;
    color: #4b5563 !important;
}
.progress-line p {
    margin: 0 !important;
    text-align: right !important;
}
.topbar .compact-input {
    min-width: 150px !important;
}
.topbar .compact-input .block {
    min-height: 0 !important;
}
.main-panel {
    gap: 8px !important;
}
.image-card, .score-card {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    padding: 6px !important;
}
.image-stack {
    gap: 6px !important;
}
.image-stack .block {
    margin: 0 !important;
}
.image-stack img {
    object-fit: contain !important;
}
.feature-hint {
    font-size: 12px !important;
    line-height: 1.34 !important;
    color: #374151 !important;
    margin-top: 2px !important;
}
.feature-hint p {
    margin: 0 !important;
}
.score-card {
    max-height: calc(100vh - 92px) !important;
    overflow: hidden !important;
    min-width: 300px !important;
}
.score-title h3 {
    margin: 0 0 2px 0 !important;
    font-size: 14px !important;
    font-weight: 650 !important;
}
.score-row {
    margin: 0 0 2px 0 !important;
}
.score-row > label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #111827 !important;
    margin-bottom: 1px !important;
}
.score-row .wrap {
    gap: 1px !important;
}
.score-row label {
    min-width: 23px !important;
    min-height: 22px !important;
    padding: 1px 3px !important;
    text-align: center !important;
    border-radius: 5px !important;
    font-size: 12px !important;
}
.compact-input textarea, .compact-input input {
    min-height: 28px !important;
}
.compact-buttons button {
    min-height: 29px !important;
    padding: 4px 7px !important;
}
.gr-form, .block {
    border-color: #e5e7eb !important;
}
"""


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_source_image(stem: str) -> Path:
    matches = sorted(
        p for p in SOURCE_ADDITION_DIR.iterdir()
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source image for stem={stem}, found {len(matches)}")
    return matches[0]


def normalize_edited_stem(path: Path, disease: str) -> str:
    stem = path.stem
    suffix = f"-{disease}-edit"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def load_source_items() -> list[dict[str, Any]]:
    manifest_path = SOURCE_ADDITION_DIR / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list):
        raise TypeError(f"Expected list manifest: {manifest_path}")

    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for raw in manifest:
        if not isinstance(raw, dict):
            raise TypeError("Each source manifest item must be an object")
        stem = raw["stem"]
        if stem in seen:
            raise ValueError(f"Duplicate source stem: {stem}")
        seen.add(stem)

        original_path = find_source_image(stem)
        prompt_path = SOURCE_ADDITION_DIR / f"{stem}.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

        items.append(
            {
                "stem": stem,
                "task_type": raw.get("task_type", "addition"),
                "modality": raw["modality"],
                "disease": raw["disease"],
                "image_id": raw.get("image_id", ""),
                "instruction": prompt_path.read_text(encoding="utf-8").strip(),
                "source_original": str(original_path),
                "source_prompt": str(prompt_path),
            }
        )

    if not items:
        raise ValueError(f"No source items found in {manifest_path}")
    return sorted(items, key=lambda x: x["stem"])


def discover_edited_images(model_name: str) -> dict[str, Path]:
    edit_dir = SOURCE_RESULT_DIR / model_name / "edit"
    if not edit_dir.is_dir():
        raise FileNotFoundError(f"Missing edit directory for model={model_name}: {edit_dir}")

    by_stem: dict[str, Path] = {}
    duplicates: list[str] = []
    for image_path in sorted(p for p in edit_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        disease = image_path.parent.name
        stem = normalize_edited_stem(image_path, disease)
        if stem in by_stem:
            duplicates.append(stem)
        by_stem[stem] = image_path

    if duplicates:
        raise ValueError(f"Duplicate edited stems for model={model_name}: {sorted(set(duplicates))[:20]}")
    if not by_stem:
        raise ValueError(f"No edit images found for model={model_name}: {edit_dir}")
    return by_stem


def prepare_data(models: tuple[str, ...] = DEFAULT_MODELS) -> dict[str, Any]:
    if not SOURCE_ADDITION_DIR.is_dir():
        raise FileNotFoundError(f"Missing source addition directory: {SOURCE_ADDITION_DIR}")
    if not SOURCE_RESULT_DIR.is_dir():
        raise FileNotFoundError(f"Missing source result directory: {SOURCE_RESULT_DIR}")

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    original_dir = DATA_DIR / "original"
    prompt_dir = DATA_DIR / "prompts"
    edited_root = DATA_DIR / "edited"
    original_dir.mkdir(parents=True)
    prompt_dir.mkdir(parents=True)
    edited_root.mkdir(parents=True)

    source_items = load_source_items()
    sources_by_stem = {item["stem"]: item for item in source_items}
    local_sources: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []

    for item in source_items:
        src_img = Path(item["source_original"])
        dst_img = original_dir / f"{item['stem']}{src_img.suffix.lower()}"
        dst_prompt = prompt_dir / f"{item['stem']}.txt"
        shutil.copy2(src_img, dst_img)
        shutil.copy2(Path(item["source_prompt"]), dst_prompt)
        local_sources[item["stem"]] = {
            "stem": item["stem"],
            "task_type": item["task_type"],
            "modality": item["modality"],
            "disease": item["disease"],
            "image_id": item["image_id"],
            "instruction": item["instruction"],
            "original_image": str(dst_img.relative_to(APP_DIR)),
            "prompt_file": str(dst_prompt.relative_to(APP_DIR)),
        }

    for model_name in models:
        edited_by_stem = discover_edited_images(model_name)
        missing = sorted(set(sources_by_stem) - set(edited_by_stem))
        extra = sorted(set(edited_by_stem) - set(sources_by_stem))
        if missing:
            raise FileNotFoundError(f"Missing edited images for model={model_name}: {missing[:20]}")
        if extra:
            raise ValueError(f"Edited images without source for model={model_name}: {extra[:20]}")

        for stem, source in sources_by_stem.items():
            src_edit = edited_by_stem[stem]
            disease = source["disease"]
            dst_edit = edited_root / model_name / disease / src_edit.name
            dst_edit.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_edit, dst_edit)
            cases.append(
                {
                    "case_id": f"{model_name}::{stem}",
                    "model": model_name,
                    "stem": stem,
                    "task_type": source["task_type"],
                    "modality": source["modality"],
                    "disease": disease,
                    "image_id": source["image_id"],
                    "instruction": source["instruction"],
                    "original_image": local_sources[stem]["original_image"],
                    "edited_image": str(dst_edit.relative_to(APP_DIR)),
                    "source_edited_image": str(src_edit),
                }
            )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": "edit/addition",
        "models": list(models),
        "source_count": len(source_items),
        "case_count": len(cases),
        "sources": list(local_sources.values()),
        "cases": cases,
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def load_manifest() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise TypeError(f"Manifest must be an object: {MANIFEST_PATH}")
    if "cases" not in manifest or not isinstance(manifest["cases"], list):
        raise KeyError(f"Manifest missing list field: cases ({MANIFEST_PATH})")
    return refresh_manifest_cases(manifest)


def refresh_manifest_cases(manifest: dict[str, Any]) -> dict[str, Any]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise KeyError(f"Manifest missing non-empty list field: sources ({MANIFEST_PATH})")
    edited_root = DATA_DIR / "edited"
    if not edited_root.is_dir():
        raise FileNotFoundError(f"Missing edited data directory: {edited_root}")

    sources_by_stem = {}
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("Each manifest source must be an object")
        stem = source["stem"]
        if stem in sources_by_stem:
            raise ValueError(f"Duplicate source stem in manifest: {stem}")
        sources_by_stem[stem] = source

    cases: list[dict[str, Any]] = []
    model_names = sorted(p.name for p in edited_root.iterdir() if p.is_dir())
    if not model_names:
        raise ValueError(f"No model directories found in: {edited_root}")

    for model_name in model_names:
        model_dir = edited_root / model_name
        edited_by_stem: dict[str, Path] = {}
        duplicates: list[str] = []
        for image_path in sorted(p for p in model_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            disease = image_path.parent.name
            stem = normalize_edited_stem(image_path, disease)
            if stem in edited_by_stem:
                duplicates.append(stem)
            edited_by_stem[stem] = image_path
        if duplicates:
            raise ValueError(f"Duplicate edited stems for model={model_name}: {sorted(set(duplicates))[:20]}")
        if not edited_by_stem:
            raise ValueError(f"No edited images found for model={model_name}: {model_dir}")

        missing = sorted(set(sources_by_stem) - set(edited_by_stem))
        extra = sorted(set(edited_by_stem) - set(sources_by_stem))
        if missing:
            raise FileNotFoundError(f"Missing edited images for model={model_name}: {missing[:20]}")
        if extra:
            raise ValueError(f"Edited images without source for model={model_name}: {extra[:20]}")

        for stem, source in sorted(sources_by_stem.items()):
            edited_path = edited_by_stem[stem]
            cases.append(
                {
                    "case_id": f"{model_name}::{stem}",
                    "model": model_name,
                    "stem": stem,
                    "task_type": source.get("task_type", "addition"),
                    "modality": source["modality"],
                    "disease": source["disease"],
                    "image_id": source.get("image_id", ""),
                    "instruction": source["instruction"],
                    "original_image": source["original_image"],
                    "edited_image": str(edited_path.relative_to(APP_DIR)),
                }
            )

    old_models = manifest.get("models")
    old_case_count = manifest.get("case_count")
    manifest["models"] = model_names
    manifest["source_count"] = len(sources_by_stem)
    manifest["case_count"] = len(cases)
    manifest["cases"] = cases
    if old_models != model_names or old_case_count != len(cases):
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(MANIFEST_PATH, manifest)
    return manifest


def check_data() -> dict[str, Any]:
    manifest = load_manifest()
    cases = manifest["cases"]
    seen: set[str] = set()
    by_model: dict[str, int] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("Each manifest case must be an object")
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        original = APP_DIR / case["original_image"]
        edited = APP_DIR / case["edited_image"]
        if not original.is_file():
            raise FileNotFoundError(f"Missing original image: {original}")
        if not edited.is_file():
            raise FileNotFoundError(f"Missing edited image: {edited}")
        by_model[case["model"]] = by_model.get(case["model"], 0) + 1

    expected = manifest.get("source_count")
    if not isinstance(expected, int) or expected <= 0:
        raise ValueError("Manifest source_count must be a positive integer")
    short_models = {model: count for model, count in by_model.items() if count != expected}
    if short_models:
        raise ValueError(f"Model case count mismatch, expected {expected}: {short_models}")
    return {"case_count": len(cases), "source_count": expected, "by_model": dict(sorted(by_model.items()))}


def disease_feature_hint(disease: str) -> str:
    key = disease.strip().lower()
    if key not in DISEASE_FEATURE_HINTS:
        raise KeyError(f"Missing disease feature hint: {disease}")
    return f"Imaging hint: {DISEASE_FEATURE_HINTS[key]}"


class RatingManager:
    def __init__(self, seed: int = 82):
        self.seed = seed
        self.manifest = load_manifest()
        self.cases = list(self.manifest["cases"])
        random.Random(seed).shuffle(self.cases)
        for index, case in enumerate(self.cases):
            case["blind_order_index"] = index

        self.records: dict[str, dict[str, Any]] = {}
        self.current_index = 0
        self.evaluator_id = "evaluator"
        self.load_progress()
        self.jump_to_first_unrated()

    def load_progress(self) -> None:
        if not RESULT_JSON.is_file():
            return
        data = read_json(RESULT_JSON)
        if not isinstance(data, dict):
            raise TypeError(f"Result JSON must be an object: {RESULT_JSON}")
        self.evaluator_id = str(data.get("evaluator_id", self.evaluator_id))
        records = data.get("records", [])
        if not isinstance(records, list):
            raise TypeError("result.json field records must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("Each saved record must be an object")
            self.records[record["case_id"]] = record

    def jump_to_first_unrated(self) -> None:
        for index, case in enumerate(self.cases):
            if case["case_id"] not in self.records:
                self.current_index = index
                return
        self.current_index = 0

    def current_case(self) -> dict[str, Any] | None:
        if not self.cases:
            return None
        return self.cases[self.current_index]

    def validate_scores(self, scores: dict[str, Any]) -> dict[str, int]:
        parsed: dict[str, int] = {}
        for key in SCORE_KEYS:
            value = scores.get(key)
            if value is None:
                raise ValueError(f"Missing score: {key}")
            score = int(value)
            if score < 0 or score > 10:
                raise ValueError(f"Score out of range [0, 10]: {key}={score}")
            parsed[key] = score
        return parsed

    def save_current(self, evaluator_id: str, scores: dict[str, Any], skipped: bool = False) -> str:
        case = self.current_case()
        if case is None:
            raise ValueError("No case loaded")
        evaluator_id = evaluator_id.strip() or "evaluator"
        self.evaluator_id = evaluator_id

        parsed_scores = None if skipped else self.validate_scores(scores)
        average_score = None
        if parsed_scores is not None:
            average_score = statistics.fmean(parsed_scores.values())

        record = {
            "case_id": case["case_id"],
            "blind_order_index": case["blind_order_index"],
            "evaluator_id": self.evaluator_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "skipped": skipped,
            "model": case["model"],
            "stem": case["stem"],
            "task_type": case["task_type"],
            "modality": case["modality"],
            "disease": case["disease"],
            "image_id": case["image_id"],
            "instruction": case["instruction"],
            "original_image": case["original_image"],
            "edited_image": case["edited_image"],
            "scores": parsed_scores,
            "average_score": average_score,
        }
        self.records[case["case_id"]] = record
        self.write_results()
        return f"Saved: {self.done_count()} / {len(self.cases)}"

    def write_results(self) -> None:
        ordered_records = [self.records[k] for k in sorted(self.records)]
        result = {
            "evaluator_id": self.evaluator_id,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "seed": self.seed,
            "task": self.manifest.get("task", "edit/addition"),
            "total_cases": len(self.cases),
            "done_cases": self.done_count(),
            "remaining_cases": len(self.cases) - self.done_count(),
            "records": ordered_records,
            "summary": self.build_summary(ordered_records),
        }
        write_json(RESULT_JSON, result)
        with RESULT_JSONL.open("w", encoding="utf-8") as f:
            for record in ordered_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def build_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [r for r in records if not r.get("skipped") and isinstance(r.get("scores"), dict)]
        return {
            "overall": self.stats_for(scored),
            "by_model": self.group_stats(scored, "model"),
            "by_disease": self.group_stats(scored, "disease"),
            "by_modality": self.group_stats(scored, "modality"),
            "skipped_count": len([r for r in records if r.get("skipped")]),
        }

    def group_stats(self, records: list[dict[str, Any]], key: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record[key]), []).append(record)
        return {name: self.stats_for(items) for name, items in sorted(grouped.items())}

    def stats_for(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"count": len(records), "dimensions": {}, "average_score": None}
        averages = [float(r["average_score"]) for r in records if r.get("average_score") is not None]
        if averages:
            out["average_score"] = statistics.fmean(averages)
        for key in SCORE_KEYS:
            values = [float(r["scores"][key]) for r in records]
            out["dimensions"][key] = statistics.fmean(values) if values else None
        return out

    def done_count(self) -> int:
        return len(self.records)

    def next_case(self) -> None:
        if self.current_index < len(self.cases) - 1:
            self.current_index += 1

    def prev_case(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1

    def jump_to(self, index: int) -> None:
        if index < 0 or index >= len(self.cases):
            raise ValueError(f"Jump index out of range: {index}")
        self.current_index = index


def build_interface(seed: int) -> gr.Blocks:
    check_data()
    manager = RatingManager(seed=seed)

    with gr.Blocks(title="Med-Banana blind human evaluation") as demo:
        with gr.Row(elem_classes=["topbar"]):
            with gr.Column(scale=5, elem_classes=["app-title"]):
                gr.Markdown("### Med-Banana blind human evaluation &nbsp;|&nbsp; 0 = failure, 10 = excellent &nbsp;·&nbsp; score the edited image only")
            with gr.Column(scale=1):
                evaluator_id = gr.Textbox(
                    label="Annotator",
                    value=manager.evaluator_id,
                    show_label=False,
                    placeholder="Annotator",
                    elem_classes=["compact-input"],
                )
            with gr.Column(scale=2):
                progress_md = gr.Markdown(elem_classes=["progress-line"])

        with gr.Row(elem_classes=["main-panel"]):
            with gr.Column(scale=9, elem_classes=["image-card"]):
                with gr.Column(elem_classes=["image-stack"]):
                    original_img = gr.Image(label="Source", type="filepath", height=330)
                    edited_img = gr.Image(label="Edited (score this)", type="filepath", height=330)
                info_md = gr.Markdown(elem_classes=["meta-line"])
                disease_hint_md = gr.Markdown(elem_classes=["feature-hint"])

            with gr.Column(scale=3, elem_classes=["score-card"]):
                gr.Markdown("### Four dimensions", elem_classes=["score-title"])
                score_components: dict[str, gr.Radio] = {}
                for key in SCORE_KEYS:
                    score_components[key] = gr.Radio(
                        choices=SCORE_CHOICES,
                        label=SCORE_LABELS[key],
                        value=None,
                        elem_classes=["score-row"],
                    )

                quick_scores = gr.Textbox(
                    label="Quick entry",
                    placeholder="8 7 8 6",
                    lines=1,
                    elem_classes=["compact-input"],
                )
                status_md = gr.Markdown(elem_classes=["status-line"])

                with gr.Row(elem_classes=["compact-buttons"]):
                    btn_prev = gr.Button("Previous")
                    btn_save_next = gr.Button("Save and next", variant="primary")
                with gr.Row(elem_classes=["compact-buttons"]):
                    btn_skip = gr.Button("Skip")
                    btn_next = gr.Button("Next")

        def values_for_current(status: str = "") -> list[Any]:
            case = manager.current_case()
            if case is None:
                return [None, None, "No data", "No data", "No data", "", None, None, None, None, "", status]

            record = manager.records.get(case["case_id"])
            scores = record.get("scores") if isinstance(record, dict) else None
            score_values = []
            for key in SCORE_KEYS:
                value = scores.get(key) if isinstance(scores, dict) else None
                score_values.append(str(value) if value is not None else None)

            done = manager.done_count()
            progress = f"**{done}/{len(manager.cases)}** &nbsp; current **{manager.current_index + 1}**"
            info = (
                f"Target disease: **{case['disease']}** &nbsp; "
                f"Modality: **{case['modality']}** &nbsp; "
                f"Instruction: `{case['instruction']}` &nbsp; "
                f"{'scored' if record else 'not scored'}"
            )
            disease_hint = disease_feature_hint(case["disease"])
            return [
                str(APP_DIR / case["original_image"]),
                str(APP_DIR / case["edited_image"]),
                progress,
                info,
                disease_hint,
                manager.evaluator_id,
                *score_values,
                "",
                status,
            ]

        def parse_quick_scores(text: str, current_values: list[Any]) -> dict[str, Any]:
            text = text.strip()
            if not text:
                return dict(zip(SCORE_KEYS, current_values, strict=True))
            parts = text.replace(",", " ").replace("，", " ").split()
            if len(parts) != 4:
                raise ValueError("Quick entry must be four scores in 0-10, e.g. 8 7 8 6")
            return dict(zip(SCORE_KEYS, parts, strict=True))

        def save_common(evaluator: str, quick: str, skip: bool, move_next: bool, *score_values: Any) -> list[Any]:
            scores = parse_quick_scores(quick, list(score_values))
            status = manager.save_current(evaluator, scores, skipped=skip)
            if move_next:
                manager.next_case()
            return values_for_current(status)

        def on_save_next(evaluator: str, quick: str, *score_values: Any) -> list[Any]:
            return save_common(evaluator, quick, False, True, *score_values)

        def on_skip(evaluator: str) -> list[Any]:
            manager.save_current(evaluator, {key: 0 for key in SCORE_KEYS}, skipped=True)
            manager.next_case()
            return values_for_current("Skipped")

        def on_prev() -> list[Any]:
            manager.prev_case()
            return values_for_current()

        def on_next() -> list[Any]:
            manager.next_case()
            return values_for_current()

        outputs = [
            original_img,
            edited_img,
            progress_md,
            info_md,
            disease_hint_md,
            evaluator_id,
            *[score_components[key] for key in SCORE_KEYS],
            quick_scores,
            status_md,
        ]
        score_inputs = [score_components[key] for key in SCORE_KEYS]

        btn_save_next.click(on_save_next, [evaluator_id, quick_scores, *score_inputs], outputs)
        btn_skip.click(on_skip, [evaluator_id], outputs)
        btn_prev.click(on_prev, [], outputs)
        btn_next.click(on_next, [], outputs)
        demo.load(values_for_current, [], outputs)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Med-Banana edit human blind scoring GUI")
    parser.add_argument("--prepare", action="store_true", help="copy and organize source data into this folder")
    parser.add_argument("--check", action="store_true", help="validate organized data and exit")
    parser.add_argument("--seed", type=int, default=82, help="blind shuffle seed")
    parser.add_argument("--port", type=int, default=7861, help="Gradio port")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server host")
    parser.add_argument("--share", action="store_true", help="enable Gradio share link")
    parser.add_argument("--no-browser", action="store_true", help="do not open the default browser")
    return parser.parse_args()


def open_browser_later(url: str) -> None:
    def _open() -> None:
        webbrowser.open(url, new=2)

    threading.Timer(1.5, _open).start()


def find_available_port(preferred_port: int, host: str, max_tries: int = 50) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise OSError(f"Cannot find empty port in range: {preferred_port}-{preferred_port + max_tries - 1}")


def main() -> None:
    args = parse_args()
    if args.prepare:
        manifest = prepare_data()
        print(f"prepared {manifest['case_count']} cases from {manifest['source_count']} source images")
    if args.check:
        stats = check_data()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        if not args.prepare:
            return
    if args.prepare:
        return

    demo = build_interface(seed=args.seed)
    selected_port = find_available_port(args.port, args.server_name)
    if selected_port != args.port:
        print(f"Port {args.port} is busy, using {selected_port} instead.")
    browser_url = f"http://127.0.0.1:{selected_port}"
    print(f"Local URL: {browser_url}")
    if not args.no_browser:
        open_browser_later(browser_url)
    demo.launch(
        server_name=args.server_name,
        server_port=selected_port,
        share=args.share,
        show_error=True,
        inbrowser=False,
        allowed_paths=[str(DATA_DIR)],
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
