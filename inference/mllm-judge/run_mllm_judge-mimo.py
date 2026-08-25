#!/usr/bin/env python3
"""Run MLLM-based evaluation for Med-Banana Edit/addition outputs."""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import importlib.util
import json
import mimetypes
import os
import random
import re
import shlex
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError


DEFAULT_INFERENCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = DEFAULT_INFERENCE_ROOT / "data"
DEFAULT_RESULT_ROOT = DEFAULT_INFERENCE_ROOT / "result"
DEFAULT_OUTPUT_ROOT = DEFAULT_INFERENCE_ROOT / "mllm-judge" / "outputs-mimo"

# OpenAI-compatible endpoint that serves the MiMo-V2.5 judge. Set MIMO_SERVER_URL to
# whichever gateway or self-hosted server you have access to.
DEFAULT_SERVER_URL = os.environ.get("MIMO_SERVER_URL", "")
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_API_KEY_ENV = "MIMO_API_KEY"
MEDGEMMA_JUDGE_SCRIPT = Path(__file__).with_name("run_mllm_judge-medgemma.py")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SCORE_KEYS = (
    "target_pathology_instruction",
    "biological_pathological_plausibility",
    "modality_anatomical_consistency",
    "imaging_physics_counterfactual_fidelity",
)
QUALITY_FLAG_KEYS = (
    "target_visible",
    "biologically_integrated",
    "anatomically_consistent",
    "physics_matched",
    "global_image_preserved",
    "major_artifact",
)

def load_medgemma_prompt_module() -> Any:
    if not MEDGEMMA_JUDGE_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing MedGemma judge script: {MEDGEMMA_JUDGE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("medgemma_judge_prompts", MEDGEMMA_JUDGE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MedGemma judge script: {MEDGEMMA_JUDGE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not isinstance(getattr(module, "SYSTEM_PROMPT", None), str):
        raise TypeError("MedGemma judge script must define SYSTEM_PROMPT as str")
    if not callable(getattr(module, "build_user_prompt", None)):
        raise TypeError("MedGemma judge script must define callable build_user_prompt")
    return module


MEDGEMMA_PROMPTS = load_medgemma_prompt_module()
SYSTEM_PROMPT = MEDGEMMA_PROMPTS.SYSTEM_PROMPT


def build_user_prompt(sample: "JudgeSample") -> str:
    return MEDGEMMA_PROMPTS.build_user_prompt(sample)


@dataclass(frozen=True)
class SourceItem:
    stem: str
    modality: str
    disease: str
    instruction: str
    original_path: Path


@dataclass(frozen=True)
class JudgeSample:
    stem: str
    model_name: str
    modality: str
    disease: str
    instruction: str
    original_path: Path
    edited_path: Path


def image_to_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime = mimetypes.guess_type(str(path))[0]
    if mime is None:
        raise ValueError(f"Cannot infer MIME type for image: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources(data_root: Path) -> dict[str, SourceItem]:
    addition_dir = data_root / "addition"
    manifest_path = addition_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list):
        raise TypeError(f"Expected list manifest: {manifest_path}")

    sources: dict[str, SourceItem] = {}
    for item in manifest:
        stem = item["stem"]
        original_path = addition_dir / f"{stem}.jpg"
        prompt_path = addition_dir / f"{stem}.txt"
        if not original_path.is_file():
            raise FileNotFoundError(f"Missing original image: {original_path}")
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
        sources[stem] = SourceItem(
            stem=stem,
            modality=item["modality"],
            disease=item["disease"],
            instruction=prompt_path.read_text(encoding="utf-8").strip(),
            original_path=original_path,
        )

    if not sources:
        raise ValueError(f"No addition sources found in {manifest_path}")
    return sources


def normalize_result_stem(path: Path, disease: str) -> str:
    stem = path.stem
    suffix = f"-{disease}-edit"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def discover_model_dirs(result_root: Path, requested_models: list[str] | None) -> list[Path]:
    if requested_models:
        dirs = [result_root / name for name in requested_models]
        missing = [str(p) for p in dirs if not p.is_dir()]
        if missing:
            raise FileNotFoundError(f"Requested model dirs not found: {missing}")
        return dirs

    dirs = []
    for model_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        edit_dir = model_dir / "edit"
        if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in edit_dir.rglob("*")) if edit_dir.is_dir() else False:
            dirs.append(model_dir)
    if not dirs:
        raise ValueError(f"No model edit images found under {result_root}")
    return dirs


def build_samples(
    sources: dict[str, SourceItem],
    result_root: Path,
    requested_models: list[str] | None,
) -> tuple[list[JudgeSample], list[dict[str, Any]]]:
    samples: list[JudgeSample] = []
    missing_cases: list[dict[str, Any]] = []
    model_dirs = discover_model_dirs(result_root, requested_models)

    for model_dir in model_dirs:
        model_name = model_dir.name
        edit_dir = model_dir / "edit"
        by_stem: dict[str, Path] = {}
        duplicates: list[str] = []

        for image_path in sorted(p for p in edit_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            disease = image_path.parent.name
            norm_stem = normalize_result_stem(image_path, disease)
            if norm_stem in by_stem:
                duplicates.append(norm_stem)
            by_stem[norm_stem] = image_path

        if duplicates:
            raise ValueError(f"Duplicate normalized result stems for {model_name}: {sorted(set(duplicates))[:10]}")

        for stem, source in sorted(sources.items()):
            edited_path = by_stem.get(stem)
            if edited_path is None:
                missing_cases.append(
                    {
                        "type": "missing_edited_image",
                        "model": model_name,
                        "stem": stem,
                        "modality": source.modality,
                        "disease": source.disease,
                    }
                )
                continue
            samples.append(
                JudgeSample(
                    stem=stem,
                    model_name=model_name,
                    modality=source.modality,
                    disease=source.disease,
                    instruction=source.instruction,
                    original_path=source.original_path,
                    edited_path=edited_path,
                )
            )

    return samples, missing_cases


def extract_text_and_reasoning(message: Any) -> tuple[str, str]:
    content = getattr(message, "content", "") or ""
    reasoning = (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or ""
    )

    if not reasoning and "<think>" in content and "</think>" in content:
        match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            content = (content[: match.start()] + content[match.end() :]).strip()

    return str(content).strip(), str(reasoning).strip()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def find_balanced_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object start found", text, 0)

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise json.JSONDecodeError("No balanced JSON object found", text, start)


def parse_json_with_repairs(text: str) -> dict[str, Any]:
    candidates = [text]
    fenced = strip_code_fence(text)
    if fenced != text:
        candidates.append(fenced)

    try:
        balanced = find_balanced_json(fenced)
        candidates.append(balanced)
    except json.JSONDecodeError:
        pass

    tried: set[str] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in tried:
            continue
        tried.add(candidate)
        repairs = [
            candidate,
            re.sub(r",(\s*[}\]])", r"\1", candidate),
        ]
        for repaired in repairs:
            try:
                obj = json.loads(repaired)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if not isinstance(obj, dict):
                raise TypeError("Judge response JSON must be an object")
            return obj

        # Some models emit Python-style dicts. This repair is explicit and
        # deterministic; malformed content still fails below.
        try:
            obj = ast.literal_eval(candidate)
        except (ValueError, SyntaxError) as exc:
            last_error = exc
            continue
        if not isinstance(obj, dict):
            raise TypeError("Judge response JSON must be an object")
        return obj

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("Unable to parse JSON object", text, 0)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return parse_json_with_repairs(text)


def validate_scores(obj: dict[str, Any]) -> dict[str, float]:
    scores = obj.get("scores")
    if not isinstance(scores, dict):
        raise TypeError("Judge JSON missing object field: scores")

    parsed: dict[str, float] = {}
    for key in SCORE_KEYS:
        if key not in scores:
            raise KeyError(f"Judge scores missing key: {key}")
        value = scores[key]
        if not isinstance(value, (int, float)):
            raise TypeError(f"Score {key} must be numeric, got {type(value).__name__}")
        value = float(value)
        if value < 0 or value > 10:
            raise ValueError(f"Score {key} out of range [0, 10]: {value}")
        parsed[key] = value
    return parsed


def validate_quality_flags(obj: dict[str, Any]) -> dict[str, bool]:
    flags = obj.get("quality_flags")
    if not isinstance(flags, dict):
        raise TypeError("Judge JSON missing object field: quality_flags")

    parsed: dict[str, bool] = {}
    for key in QUALITY_FLAG_KEYS:
        if key not in flags:
            raise KeyError(f"Judge quality_flags missing key: {key}")
        value = flags[key]
        if not isinstance(value, bool):
            raise TypeError(f"quality_flags.{key} must be boolean, got {type(value).__name__}")
        parsed[key] = value
    return parsed


def dimension_reason_text(obj: dict[str, Any], key: str) -> str:
    reasons = obj.get("dimension_reasons")
    if not isinstance(reasons, dict):
        raise TypeError("Judge JSON missing object field: dimension_reasons")
    value = reasons.get(key)
    if not isinstance(value, str):
        raise TypeError(f"dimension_reasons.{key} must be a string")
    return value.lower()


def validate_score_consistency(obj: dict[str, Any], scores: dict[str, float], flags: dict[str, bool]) -> None:
    cap_errors: list[str] = []
    if not flags["target_visible"] and scores["target_pathology_instruction"] > 3.5:
        cap_errors.append("target_visible=false requires target_pathology_instruction <= 3.5")
    if not flags["biologically_integrated"] and scores["biological_pathological_plausibility"] > 6.0:
        cap_errors.append("biologically_integrated=false requires biological_pathological_plausibility <= 6.0")
    if not flags["anatomically_consistent"] and scores["modality_anatomical_consistency"] > 6.0:
        cap_errors.append("anatomically_consistent=false requires modality_anatomical_consistency <= 6.0")
    if not flags["physics_matched"] and scores["imaging_physics_counterfactual_fidelity"] > 5.5:
        cap_errors.append("physics_matched=false requires imaging_physics_counterfactual_fidelity <= 5.5")
    if not flags["global_image_preserved"] and scores["imaging_physics_counterfactual_fidelity"] > 6.0:
        cap_errors.append("global_image_preserved=false requires imaging_physics_counterfactual_fidelity <= 6.0")
    if flags["major_artifact"] and scores["imaging_physics_counterfactual_fidelity"] > 5.5:
        cap_errors.append("major_artifact=true requires imaging_physics_counterfactual_fidelity <= 5.5")

    failure_modes = obj.get("failure_modes")
    if not isinstance(failure_modes, list):
        raise TypeError("Judge JSON field failure_modes must be a list")
    if failure_modes and min(scores.values()) >= 7.0:
        cap_errors.append("failure_modes is non-empty but all scores are >= 7")

    bio_reason = dimension_reason_text(obj, "biological_pathological_plausibility")
    if any(term in bio_reason for term in ("isolated", "lack", "no local", "weak local")):
        if scores["biological_pathological_plausibility"] > 6.0:
            cap_errors.append("biological reason reports isolation or weak reaction but biology score > 6.0")

    anatomy_reason = dimension_reason_text(obj, "modality_anatomical_consistency")
    if any(term in anatomy_reason for term in ("distort", "impossible", "wrong", "poorly localized")):
        if scores["modality_anatomical_consistency"] > 6.0:
            cap_errors.append("anatomy reason reports distortion/wrong placement but anatomy score > 6.0")

    fidelity_reason = dimension_reason_text(obj, "imaging_physics_counterfactual_fidelity")
    if any(term in fidelity_reason for term in ("pasted", "sharp", "smudg", "texture mismatch", "mismatched", "noise mismatch", "abrupt")):
        if scores["imaging_physics_counterfactual_fidelity"] > 5.5:
            cap_errors.append("fidelity reason reports artifact/texture mismatch but fidelity score > 5.5")

    if cap_errors:
        raise ValueError("; ".join(cap_errors))


def is_retryable_openai_error(exc: OpenAIError) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc)
    return (
        "Too many requests" in message
        or "temporarily unavailable" in message.lower()
        or "Multimodal data is corrupted or cannot be processed" in message
    )


async def call_with_api_retries(
    client: AsyncOpenAI,
    request_kwargs: dict[str, Any],
    response_format_json: bool,
    api_retries: int,
    api_retry_base_delay: float,
    api_retry_max_delay: float,
    sem: asyncio.Semaphore,
) -> Any:
    kwargs = dict(request_kwargs)
    for attempt in range(api_retries + 1):
        async with sem:
            try:
                return await client.chat.completions.create(**kwargs)
            except OpenAIError as exc:
                if response_format_json and "response_format" in str(exc):
                    kwargs.pop("response_format", None)
                    return await client.chat.completions.create(**kwargs)
                if attempt >= api_retries or not is_retryable_openai_error(exc):
                    raise
                delay = min(api_retry_max_delay, api_retry_base_delay * (2 ** attempt))
                delay += random.uniform(0, delay * 0.25)
        await asyncio.sleep(delay)

    raise RuntimeError("unreachable api retry state")


async def judge_one(
    client: AsyncOpenAI,
    sample: JudgeSample,
    model: str,
    max_tokens: int,
    temperature: float,
    parse_retries: int,
    api_retries: int,
    api_retry_base_delay: float,
    api_retry_max_delay: float,
    response_format_json: bool,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(sample)
    last_error: Exception | None = None
    last_content = ""

    for attempt in range(parse_retries + 1):
        retry_suffix = ""
        if attempt:
            retry_suffix = (
                "\n\nYour previous answer was not valid parseable JSON or missed required fields. "
                f"Last validation error: {last_error}. "
                "Retry now with exactly one compact JSON object using the required schema. "
                "Make quality_flags, failure_modes, reasons, and scores consistent with the hard caps."
            )
        request_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image 1: Original image before editing."},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(sample.original_path)}},
                        {"type": "text", "text": "Image 2: Edited candidate image to evaluate."},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(sample.edited_path)}},
                        {"type": "text", "text": user_prompt + retry_suffix},
                    ],
                },
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if response_format_json:
            request_kwargs["response_format"] = {"type": "json_object"}

        response = await call_with_api_retries(
            client=client,
            request_kwargs=request_kwargs,
            response_format_json=response_format_json,
            api_retries=api_retries,
            api_retry_base_delay=api_retry_base_delay,
            api_retry_max_delay=api_retry_max_delay,
            sem=sem,
        )

        message = response.choices[0].message
        content, reasoning = extract_text_and_reasoning(message)
        last_content = content
        try:
            obj = extract_json_object(content)
            scores = validate_scores(obj)
            flags = validate_quality_flags(obj)
            validate_score_consistency(obj, scores, flags)
            break
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, SyntaxError) as exc:
            last_error = exc
    else:
        raise ValueError(
            f"Unable to parse valid judge JSON after {parse_retries + 1} attempts. "
            f"Last error: {last_error}. Last content prefix: {last_content[:500]}"
        )

    combined_output = f"<think>\n{reasoning}\n</think>\n{json.dumps(obj, ensure_ascii=False)}"

    return {
        "model": sample.model_name,
        "stem": sample.stem,
        "modality": sample.modality,
        "disease": sample.disease,
        "task_type": "addition",
        "instruction": sample.instruction,
        "original_path": str(sample.original_path),
        "edited_path": str(sample.edited_path),
        "think": reasoning,
        "judge_json": obj,
        "scores": scores,
        "raw_content": content,
        "combined_output": combined_output,
        "parse_attempts": attempt + 1,
        "response_format_json": response_format_json,
        "status": "success",
    }


def stats_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(records), "dimensions": {}}
    for key in SCORE_KEYS:
        values = [float(r["scores"][key]) for r in records]
        if values:
            out["dimensions"][key] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        else:
            out["dimensions"][key] = {"mean": None, "std": None, "min": None, "max": None}
    return out


def build_summary(
    records: list[dict[str, Any]],
    missing_cases: list[dict[str, Any]],
    expected_per_model: int,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_disease: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in records:
        by_model[rec["model"]].append(rec)
        by_model_modality[f"{rec['model']}::{rec['modality']}"].append(rec)
        by_model_disease[f"{rec['model']}::{rec['disease']}"].append(rec)

    missing_by_model: dict[str, int] = defaultdict(int)
    for case in missing_cases:
        if "model" in case:
            missing_by_model[case["model"]] += 1

    models = {}
    for model, model_records in sorted(by_model.items()):
        model_stats = stats_for(model_records)
        model_stats["expected_n"] = expected_per_model
        model_stats["effective_n"] = len(model_records)
        model_stats["missing_n"] = missing_by_model.get(model, 0)
        models[model] = model_stats

    return {
        "task": "edit/addition",
        "models": models,
        "by_model_modality": {k: stats_for(v) for k, v in sorted(by_model_modality.items())},
        "by_model_disease": {k: stats_for(v) for k, v in sorted(by_model_disease.items())},
        "missing_cases_count": len(missing_cases),
    }


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["model"]), str(record["stem"])


def sample_key(sample: JudgeSample) -> tuple[str, str]:
    return sample.model_name, sample.stem


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"JSONL line must be an object: {path}:{line_number}")
            records.append(obj)
    return records


def load_existing_success_records(out_dir: Path) -> list[dict[str, Any]]:
    records = read_jsonl(out_dir / "records.jsonl")
    seen: set[tuple[str, str]] = set()
    for record in records:
        validate_scores(record.get("judge_json", {}))
        key = record_key(record)
        if key in seen:
            raise ValueError(f"Duplicate successful record in resume file: model={key[0]} stem={key[1]}")
        seen.add(key)
    return records


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_models(value: str) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def read_api_key_from_zshrc() -> str | None:
    zshrc_path = Path.home() / ".zshrc"
    if not zshrc_path.is_file():
        return None

    prefix = f"{DEFAULT_API_KEY_ENV}="
    for raw_line in zshrc_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if not value:
            return None
        parts = shlex.split(value, comments=True)
        return parts[0] if parts else None
    return None


def resolve_api_key(cli_api_key: str | None) -> str:
    api_key = cli_api_key or os.getenv(DEFAULT_API_KEY_ENV) or read_api_key_from_zshrc()
    if not api_key:
        raise ValueError(f"Missing MiMo API key. Set {DEFAULT_API_KEY_ENV} or pass --api-key.")
    return api_key


def read_subset_stems(path: Path, sources: dict[str, SourceItem]) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Subset stems file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise TypeError(f"Subset stems file must be a JSON list of strings: {path}")
    missing = [stem for stem in data if stem not in sources]
    if missing:
        raise ValueError(f"Subset stems not found in source manifest: {missing[:10]}")
    duplicates = sorted(stem for stem, count in Counter(data).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate subset stems: {duplicates[:10]}")
    return data


def choose_subset_stems(
    sources: dict[str, SourceItem],
    subset_size: int,
    subset_seed: int,
    subset_stems_path: Path | None,
) -> list[str]:
    if subset_stems_path is not None:
        return read_subset_stems(subset_stems_path, sources)
    if subset_size < 0:
        raise ValueError("--subset-size must be >= 0")
    if subset_size == 0 or subset_size >= len(sources):
        return sorted(sources)

    rng = random.Random(subset_seed)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for stem, source in sources.items():
        groups[(source.modality, source.disease)].append(stem)

    for stems in groups.values():
        rng.shuffle(stems)

    selected: list[str] = []
    group_keys = sorted(groups)
    while len(selected) < subset_size:
        progressed = False
        for key in group_keys:
            stems = groups[key]
            if stems:
                selected.append(stems.pop())
                progressed = True
                if len(selected) == subset_size:
                    break
        if not progressed:
            break

    return sorted(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Med-Banana MLLM judge for Edit/addition outputs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None, help=f"MiMo API key. Default: ${DEFAULT_API_KEY_ENV}.")
    parser.add_argument("--models", default="", help="Comma-separated result model dirs. Default: all non-empty edit dirs.")
    parser.add_argument("--limit", type=int, default=0, help="Limit total evaluated samples after pairing.")
    parser.add_argument("--subset-size", type=int, default=0, help="Evaluate the same deterministic source subset for every model. 0 means all sources.")
    parser.add_argument("--subset-seed", type=int, default=82, help="Seed for deterministic stratified subset selection.")
    parser.add_argument("--subset-stems", type=Path, default=None, help="JSON list of source stems to evaluate for every model.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=2, help="Retry the same sample when JSON parsing/schema validation fails.")
    parser.add_argument("--api-retries", type=int, default=5, help="Retry transient MiMo/OpenAI API errors such as 429.")
    parser.add_argument("--api-retry-base-delay", type=float, default=2.0)
    parser.add_argument("--api-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--no-response-format-json", action="store_true", help="Do not request OpenAI JSON object response_format.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an existing outputs/<run_id> directory. Successful records are skipped; failures are retried.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only scan and write pairing summary; do not call VLM.")
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.parse_retries < 0:
        raise ValueError("--parse-retries must be >= 0")
    if args.api_retries < 0:
        raise ValueError("--api-retries must be >= 0")
    if args.api_retry_base_delay <= 0:
        raise ValueError("--api-retry-base-delay must be > 0")
    if args.api_retry_max_delay <= 0:
        raise ValueError("--api-retry-max-delay must be > 0")
    if args.limit and (args.subset_size or args.subset_stems is not None):
        raise ValueError("Use either --limit or subset options, not both. --subset-size is the fair model-comparison mode.")
    if args.subset_size and args.subset_stems is not None:
        raise ValueError("Use either --subset-size or --subset-stems, not both.")

    if args.resume is not None:
        out_dir = args.resume
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {out_dir}")
        run_id = out_dir.name
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out_dir = args.output_root / run_id

    existing_records = load_existing_success_records(out_dir) if args.resume is not None else []
    existing_keys = {record_key(record) for record in existing_records}
    all_sources = load_sources(args.data_root)
    subset_stems_path = args.subset_stems
    if (
        args.resume is not None
        and subset_stems_path is None
        and args.subset_size == 0
        and (out_dir / "subset_stems.json").is_file()
    ):
        subset_stems_path = out_dir / "subset_stems.json"
    selected_stems = choose_subset_stems(
        sources=all_sources,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        subset_stems_path=subset_stems_path,
    )
    outside_subset_records = [record for record in existing_records if record["stem"] not in set(selected_stems)]
    if outside_subset_records:
        raise ValueError(
            "Existing resume records contain stems outside the selected subset. "
            f"First mismatch: model={outside_subset_records[0]['model']} stem={outside_subset_records[0]['stem']}"
        )
    sources = {stem: all_sources[stem] for stem in selected_stems}
    samples, missing_cases = build_samples(sources, args.result_root, parse_models(args.models))
    pending_samples = [sample for sample in samples if sample_key(sample) not in existing_keys]
    if args.limit:
        pending_samples = pending_samples[: args.limit]

    run_config = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resume": str(args.resume) if args.resume is not None else None,
        "data_root": str(args.data_root),
        "result_root": str(args.result_root),
        "server_url": args.server_url,
        "judge_backend": "mimo",
        "judge_model": args.judge_model,
        "requested_models": parse_models(args.models),
        "limit": args.limit,
        "subset_size": args.subset_size,
        "subset_seed": args.subset_seed,
        "subset_stems_file": str(subset_stems_path) if subset_stems_path is not None else None,
        "source_count_total": len(all_sources),
        "selected_stems_count": len(selected_stems),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "parse_retries": args.parse_retries,
        "api_retries": args.api_retries,
        "api_retry_base_delay": args.api_retry_base_delay,
        "api_retry_max_delay": args.api_retry_max_delay,
        "response_format_json": not args.no_response_format_json,
        "dry_run": args.dry_run,
        "expected_per_model": len(sources),
        "paired_samples": len(samples),
        "existing_success_records": len(existing_records),
        "pending_samples": len(pending_samples),
        "missing_cases": len(missing_cases),
    }
    write_json(out_dir / "run_config.json", run_config)
    write_json(out_dir / "missing_cases.json", missing_cases)
    write_json(out_dir / "subset_stems.json", selected_stems)

    if args.dry_run:
        scan_records = [
            {
                "model": s.model_name,
                "stem": s.stem,
                "modality": s.modality,
                "disease": s.disease,
                "original_path": str(s.original_path),
                "edited_path": str(s.edited_path),
            }
            for s in samples
        ]
        write_jsonl(out_dir / "paired_samples.jsonl", scan_records)
        write_json(out_dir / "summary.json", build_summary(existing_records, missing_cases, len(sources)))
        print(f"dry_run_output={out_dir}")
        print(
            f"paired_samples={len(samples)} existing_success={len(existing_records)} "
            f"pending_samples={len(pending_samples)} missing_cases={len(missing_cases)} "
            f"expected_per_model={len(sources)}"
        )
        return

    if not args.server_url:
        raise RuntimeError(
            "MiMo judge server URL is missing. Set the MIMO_SERVER_URL environment "
            "variable or pass --server-url with an OpenAI-compatible endpoint."
        )
    client = AsyncOpenAI(base_url=args.server_url.rstrip("/"), api_key=resolve_api_key(args.api_key))
    sem = asyncio.Semaphore(args.concurrency)
    records: list[dict[str, Any]] = list(existing_records)
    failures: list[dict[str, Any]] = []
    completed_keys = set(existing_keys)
    write_lock = asyncio.Lock()
    start = time.monotonic()

    async def run_sample(sample: JudgeSample) -> None:
        try:
            record = await judge_one(
                client=client,
                sample=sample,
                model=args.judge_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                parse_retries=args.parse_retries,
                api_retries=args.api_retries,
                api_retry_base_delay=args.api_retry_base_delay,
                api_retry_max_delay=args.api_retry_max_delay,
                response_format_json=not args.no_response_format_json,
                sem=sem,
            )
            async with write_lock:
                key = sample_key(sample)
                if key not in completed_keys:
                    records.append(record)
                    completed_keys.add(key)
                    append_jsonl(out_dir / "records.jsonl", record)
        except (FileNotFoundError, ValueError, TypeError, KeyError, json.JSONDecodeError, OpenAIError) as exc:
            failure = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "model": sample.model_name,
                "stem": sample.stem,
                "modality": sample.modality,
                "disease": sample.disease,
                "original_path": str(sample.original_path),
                "edited_path": str(sample.edited_path),
                "attempted_parse_retries": args.parse_retries,
            }
            async with write_lock:
                failures.append(failure)
                append_jsonl(out_dir / "failures.jsonl", failure)

    tasks = [asyncio.create_task(run_sample(sample)) for sample in pending_samples]
    for i, task in enumerate(asyncio.as_completed(tasks), start=1):
        await task
        if i % 25 == 0 or i == len(tasks):
            print(
                f"progress={i}/{len(tasks)} existing_success={len(existing_records)} "
                f"total_success={len(records)} failures={len(failures)}"
            )

    elapsed = time.monotonic() - start
    all_missing = missing_cases + failures
    summary = build_summary(records, all_missing, len(sources))
    summary["elapsed_seconds"] = elapsed
    summary["success_records"] = len(records)
    summary["failure_records"] = len(failures)

    write_jsonl(out_dir / "records.jsonl", sorted(records, key=lambda r: (r["model"], r["stem"])))
    write_json(out_dir / "missing_cases.json", all_missing)
    write_json(out_dir / "summary.json", summary)
    print(f"output={out_dir}")
    print(
        f"existing_success={len(existing_records)} total_success={len(records)} "
        f"new_failures={len(failures)} missing={len(missing_cases)} elapsed={elapsed:.1f}s"
    )


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
