#!/usr/bin/env python3
"""Gemini batch image edit pipeline for med-banana inference data."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GEMINI_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = GEMINI_DIR.parent
DATA_DIR = INFERENCE_DIR / "data"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MODEL = "gemini-2.5-flash-image"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
ASPECT_RATIO = "1:1"
IMAGE_SIZE = "1K"
SYSTEM_INSTRUCTION = (
    "You assist with medical imaging education and research. "
    "Each request modifies a medical image for legitimate teaching purposes only: "
    "helping clinicians and researchers learn to recognize disease patterns on "
    "synthetic or edited images. These outputs are not for deception, harm, or "
    "misinformation. Follow the user instruction and return the edited image."
)
# Batch JSONL does not accept thinkingLevel; it causes JOB_STATE_FAILED (code=13).
# Use includeThoughts only. thinkingLevel applies to interactive generateContent, not batch file input.
TERMINAL_BATCH_STATUSES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
PENDING_BATCH_STATUSES = {
    "JOB_STATE_UNSPECIFIED",
    "JOB_STATE_QUEUED",
    "JOB_STATE_PENDING",
    "JOB_STATE_RUNNING",
    "JOB_STATE_CANCELLING",
    "JOB_STATE_PAUSED",
    "JOB_STATE_UPDATING",
}
FAILED_BATCH_STATUSES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
SUCCESS_BATCH_STATUSES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
MIME_TO_EXT = {
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpeg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class RunPaths:
    mode: str
    batch_dir: Path
    output_dir: Path
    thoughts_dir: Path

    @classmethod
    def from_flags(cls, test: bool, retry: bool) -> RunPaths:
        if test and retry:
            raise ValueError("Cannot use --test and --retry together.")
        if retry:
            return cls(
                mode="retry",
                batch_dir=GEMINI_DIR / "batch_retry",
                output_dir=GEMINI_DIR / "outputs",
                thoughts_dir=GEMINI_DIR / "thoughts",
            )
        if test:
            return cls(
                mode="test",
                batch_dir=GEMINI_DIR / "batch_test",
                output_dir=GEMINI_DIR / "outputs_test",
                thoughts_dir=GEMINI_DIR / "thoughts_test",
            )
        return cls(
            mode="full",
            batch_dir=GEMINI_DIR / "batch",
            output_dir=GEMINI_DIR / "outputs",
            thoughts_dir=GEMINI_DIR / "thoughts",
        )

    @classmethod
    def from_test_flag(cls, test: bool) -> RunPaths:
        return cls.from_flags(test=test, retry=False)


@dataclass(frozen=True)
class Sample:
    task_type: str
    stem: str
    prompt: str
    image_path: Path

    @property
    def custom_id(self) -> str:
        return f"{self.task_type}__{self.stem}"


def resolve_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY environment variable before running."
        )
    return key


def find_image_path(task_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        image_path = task_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Missing image for stem {stem!r} in {task_dir}")


def collect_samples(task_type: str) -> list[Sample]:
    task_dir = DATA_DIR / task_type
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    samples: list[Sample] = []
    for txt_path in sorted(task_dir.glob("*.txt")):
        stem = txt_path.stem
        image_path = find_image_path(task_dir, stem)
        prompt = txt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Empty prompt in {txt_path}")
        samples.append(Sample(task_type=task_type, stem=stem, prompt=prompt, image_path=image_path))
    return samples


def collect_all_samples() -> list[Sample]:
    return collect_samples("addition") + collect_samples("removal")


def find_sample_by_stem(stem: str) -> Sample:
    for task_type in ("addition", "removal"):
        task_dir = DATA_DIR / task_type
        txt_path = task_dir / f"{stem}.txt"
        if txt_path.exists():
            image_path = find_image_path(task_dir, stem)
            prompt = txt_path.read_text(encoding="utf-8").strip()
            if not prompt:
                raise ValueError(f"Empty prompt in {txt_path}")
            return Sample(task_type=task_type, stem=stem, prompt=prompt, image_path=image_path)
    raise FileNotFoundError(f"No sample found for stem {stem!r} under {DATA_DIR}")


def load_samples_from_stems_file(stems_file: Path) -> list[Sample]:
    if not stems_file.is_file():
        raise FileNotFoundError(f"Stems file not found: {stems_file}")

    samples: list[Sample] = []
    seen: set[str] = set()
    for line_no, raw_line in enumerate(stems_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)

        if line.startswith("addition__") or line.startswith("removal__"):
            _task_type, stem = line.split("__", 1)
            sample = find_sample_by_stem(stem)
            if sample.custom_id != line:
                raise ValueError(
                    f"Line {line_no}: custom_id {line!r} does not match sample {sample.custom_id!r}"
                )
        else:
            sample = find_sample_by_stem(line)
        samples.append(sample)

    if not samples:
        raise ValueError(f"No stems found in {stems_file}")
    return samples


def select_samples(test: bool, stem: str | None, stems_file: Path | None) -> list[Sample]:
    if stems_file is not None:
        return load_samples_from_stems_file(stems_file)
    if test:
        if stem:
            return [find_sample_by_stem(stem)]
        addition_samples = collect_samples("addition")
        if not addition_samples:
            raise RuntimeError(f"No addition samples found in {DATA_DIR / 'addition'}")
        return [addition_samples[0]]
    return collect_all_samples()


def image_mime_type(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0]
    if mime_type is None:
        raise ValueError(f"Cannot infer MIME type for {image_path}")
    return mime_type


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def build_request_line(sample: Sample) -> dict[str, Any]:
    mime_type = image_mime_type(sample.image_path)
    return {
        "key": sample.custom_id,
        "request": {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": sample.prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_to_base64(sample.image_path),
                            }
                        },
                    ],
                }
            ],
            "generation_config": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": ASPECT_RATIO,
                    "imageSize": IMAGE_SIZE,
                },
            },
        },
    }
    
                #     "thinkingConfig": {
                #     "includeThoughts": True,
                # },


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_failures_from_summary(summary_path: Path, out_path: Path) -> list[dict[str, Any]]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}. Run parse first.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures = summary.get("failures")
    if failures is None:
        raise KeyError(f"Missing failures in {summary_path}")
    if not isinstance(failures, list):
        raise TypeError(f"Expected failures list in {summary_path}")

    lines: list[str] = []
    records: list[dict[str, Any]] = []
    for failure in failures:
        custom_id = failure.get("custom_id")
        if custom_id is None:
            raise KeyError(f"Failure entry missing custom_id in {summary_path}")
        reason = failure.get("reason")
        task_type, stem = str(custom_id).split("__", 1)
        lines.append(str(custom_id))
        records.append(
            {
                "custom_id": custom_id,
                "task_type": task_type,
                "stem": stem,
                "reason": reason,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    write_json(out_path.with_suffix(".json"), {"count": len(records), "items": records})
    return records


def cmd_extract_failures(args: argparse.Namespace) -> None:
    paths = RunPaths.from_flags(test=args.test, retry=False)
    summary_path = paths.batch_dir / "parse_summary.json"
    out_path = paths.batch_dir / "failed_stems.txt"
    records = extract_failures_from_summary(summary_path, out_path)
    print(f"count={len(records)}")
    print(f"failed_stems={out_path}")
    print(f"failed_stems_json={out_path.with_suffix('.json')}")


def cmd_build(args: argparse.Namespace) -> None:
    paths = RunPaths.from_flags(test=args.test, retry=args.retry)
    stems_file = Path(args.stems_file) if args.stems_file else None
    if stems_file is not None and args.test:
        raise ValueError("Use --stems-file with --retry, not --test.")
    samples = select_samples(test=args.test, stem=args.stem, stems_file=stems_file)
    paths.batch_dir.mkdir(parents=True, exist_ok=True)

    requests_path = paths.batch_dir / "requests.jsonl"
    manifest_path = paths.batch_dir / "build_manifest.json"

    manifest = {
        "mode": paths.mode,
        "model": MODEL,
        "system_instruction": SYSTEM_INSTRUCTION,
        "generation_config": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": ASPECT_RATIO, "imageSize": IMAGE_SIZE},
            "thinkingConfig": {"includeThoughts": True},
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(samples),
        "items": [],
    }

    with requests_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            request_line = build_request_line(sample)
            f.write(json.dumps(request_line, ensure_ascii=False) + "\n")
            manifest["items"].append(
                {
                    "custom_id": sample.custom_id,
                    "key": sample.custom_id,
                    "task_type": sample.task_type,
                    "stem": sample.stem,
                    "prompt": sample.prompt,
                    "source_image": str(sample.image_path.relative_to(INFERENCE_DIR)),
                }
            )

    write_json(manifest_path, manifest)
    print(f"mode={paths.mode} requests={len(samples)}")
    print(f"requests_jsonl={requests_path}")
    print(f"build_manifest={manifest_path}")


def get_gemini_client():
    from google import genai

    return genai.Client(api_key=resolve_api_key())


def fetch_batch_operation(batch_id: str, api_key: str) -> dict[str, Any]:
    url = f"{GEMINI_API_BASE}/{batch_id}"
    request = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected batch operation payload type: {type(payload)!r}")
    return payload


def extract_operation_error(operation: dict[str, Any]) -> dict[str, Any] | None:
    error = operation.get("error")
    if error is None:
        return None
    if not isinstance(error, dict):
        raise TypeError(f"Unexpected batch operation error type: {type(error)!r}")
    return error


def extract_batch_stats(operation: dict[str, Any]) -> dict[str, Any] | None:
    metadata = operation.get("metadata")
    if not isinstance(metadata, dict):
        return None
    batch_stats = metadata.get("batchStats")
    if batch_stats is None:
        return None
    if not isinstance(batch_stats, dict):
        raise TypeError(f"Unexpected batchStats type: {type(batch_stats)!r}")
    return batch_stats


def extract_operation_dest_file_name(operation: dict[str, Any]) -> str | None:
    metadata = operation.get("metadata")
    if not isinstance(metadata, dict):
        return None
    output = metadata.get("output")
    if not isinstance(output, dict):
        return None
    for key in ("responsesFile", "fileName"):
        value = output.get(key)
        if value:
            return value
    return None


def normalize_batch_stats(batch_stats: dict[str, Any]) -> dict[str, int | None]:
    def to_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    return {
        "completed": to_int(
            batch_stats.get("successfulRequestCount")
            or batch_stats.get("succeededRequestCount")
        ),
        "failed": to_int(batch_stats.get("failedRequestCount")),
        "total": to_int(batch_stats.get("requestCount")),
        "pending": to_int(batch_stats.get("pendingRequestCount")),
    }


def format_batch_error(batch: Any, operation: dict[str, Any] | None = None) -> str | None:
    messages: list[str] = []
    if batch.error is not None:
        if batch.error.message:
            messages.append(f"batch.error.message={batch.error.message}")
        if batch.error.code is not None:
            messages.append(f"batch.error.code={batch.error.code}")
        if batch.error.details:
            messages.append(f"batch.error.details={batch.error.details}")
    if operation is not None:
        operation_error = extract_operation_error(operation)
        if operation_error is not None:
            code = operation_error.get("code")
            message = operation_error.get("message")
            if code is not None:
                messages.append(f"operation.error.code={code}")
            if message:
                messages.append(f"operation.error.message={message}")
            details = operation_error.get("details")
            if details:
                messages.append(f"operation.error.details={details}")
    if not messages:
        return None
    return "; ".join(messages)


def raise_batch_terminal_error(batch_id: str, status: str, batch: Any, operation: dict[str, Any] | None) -> None:
    error_text = format_batch_error(batch, operation)
    if error_text:
        raise RuntimeError(f"Batch {batch_id} finished with status={status}: {error_text}")
    raise RuntimeError(f"Batch {batch_id} finished with status={status}")


def batch_status_name(batch: Any) -> str:
    state = batch.state
    if state is None:
        raise RuntimeError("Batch has no state.")
    if hasattr(state, "name"):
        return state.name
    return str(state)


def extract_request_counts(batch: Any) -> dict[str, int | None]:
    counts: dict[str, int | None] = {"completed": None, "failed": None, "total": None}
    for attr_map in (
        ("successful_request_count", "failed_request_count", "total_request_count"),
        ("successfulRequestCount", "failedRequestCount", "totalRequestCount"),
    ):
        completed = getattr(batch, attr_map[0], None)
        failed = getattr(batch, attr_map[1], None)
        total = getattr(batch, attr_map[2], None)
        if completed is not None or failed is not None or total is not None:
            counts["completed"] = completed
            counts["failed"] = failed
            counts["total"] = total
            return counts
    return counts


def cmd_submit(args: argparse.Namespace) -> None:
    from google.genai import types

    paths = RunPaths.from_flags(test=args.test, retry=args.retry)
    requests_path = paths.batch_dir / "requests.jsonl"
    if not requests_path.exists():
        raise FileNotFoundError(f"Missing {requests_path}. Run build first.")

    client = get_gemini_client()
    uploaded = client.files.upload(
        file=str(requests_path),
        config=types.UploadFileConfig(
            display_name=f"med-banana-{paths.mode}",
            mime_type="jsonl",
        ),
    )

    batch = client.batches.create(
        model=MODEL,
        src=uploaded.name,
        config={"display_name": f"med-banana-inference-{paths.mode}"},
    )

    status = batch_status_name(batch)
    state = {
        "mode": paths.mode,
        "batch_id": batch.name,
        "input_file_id": uploaded.name,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "display_name": f"med-banana-inference-{paths.mode}",
    }
    write_json(paths.batch_dir / "batch_state.json", state)
    print(f"batch_id={batch.name}")
    print(f"status={status}")

    if args.wait:
        poll_batch(client, batch.name, paths.batch_dir)


def sync_batch_state(client, batch_id: str, batch_dir: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    batch = client.batches.get(name=batch_id)
    operation = fetch_batch_operation(batch_id, resolve_api_key())
    state = load_batch_state(batch_dir)
    status = batch_status_name(batch)
    dest_file_name = None
    if batch.dest is not None and batch.dest.file_name:
        dest_file_name = batch.dest.file_name
    if dest_file_name is None:
        dest_file_name = extract_operation_dest_file_name(operation)

    state["status"] = status
    state["dest_file_name"] = dest_file_name
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    batch_stats = extract_batch_stats(operation)
    if batch_stats is not None:
        state["batch_stats"] = batch_stats
        state["request_counts"] = normalize_batch_stats(batch_stats)
    else:
        if "batch_stats" in state:
            del state["batch_stats"]
        state["request_counts"] = extract_request_counts(batch)

    if batch.error is not None:
        state["error"] = batch.error.model_dump(mode="json")
    elif "error" in state:
        del state["error"]

    operation_error = extract_operation_error(operation)
    if operation_error is not None:
        state["operation_error"] = operation_error
    elif "operation_error" in state:
        del state["operation_error"]

    write_json(batch_dir / "batch_state.json", state)
    return batch, state, operation


def print_batch_progress(
    batch_id: str,
    batch: Any,
    batch_dir: Path,
    operation: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    if state is not None and state.get("request_counts") is not None:
        counts = state["request_counts"]
    else:
        counts = extract_request_counts(batch)
    completed = counts.get("completed")
    failed = counts.get("failed")
    total = counts.get("total")
    pending = counts.get("pending")
    status = batch_status_name(batch)
    if completed is None and failed is None and total is None and pending is None:
        print(f"batch_id={batch_id} status={status}")
    else:
        print(
            f"batch_id={batch_id} status={status} "
            f"completed={completed} failed={failed} pending={pending} total={total}"
        )
    error_text = format_batch_error(batch, operation)
    if error_text:
        print(f"error={error_text}")
    print(f"batch_state={batch_dir / 'batch_state.json'}")


def poll_batch(client, batch_id: str, batch_dir: Path, interval_sec: int = 30) -> None:
    while True:
        batch, state, operation = sync_batch_state(client, batch_id, batch_dir)
        print_batch_progress(batch_id, batch, batch_dir, operation, state)

        status = batch_status_name(batch)
        if status in TERMINAL_BATCH_STATUSES:
            if status not in SUCCESS_BATCH_STATUSES:
                raise_batch_terminal_error(batch_id, status, batch, operation)
            return
        time.sleep(interval_sec)


def load_batch_state(batch_dir: Path) -> dict[str, Any]:
    state_path = batch_dir / "batch_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing {state_path}. Run submit first.")
    return json.loads(state_path.read_text(encoding="utf-8"))


def download_result_file(client, file_name: str) -> str:
    file_content = client.files.download(file=file_name)
    if isinstance(file_content, bytes):
        return file_content.decode("utf-8")
    if isinstance(file_content, str):
        return file_content
    raise TypeError(f"Unexpected download result type: {type(file_content)!r}")


def part_field(part: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in part:
            return part[name]
    return None


def is_thought_part(part: dict[str, Any]) -> bool:
    return bool(part_field(part, "thought"))


def part_inline_data(part: dict[str, Any]) -> dict[str, Any] | None:
    inline = part_field(part, "inlineData", "inline_data")
    if inline is None:
        return None
    if not isinstance(inline, dict):
        raise TypeError(f"Expected inline data dict, got {type(inline)!r}")
    return inline


def ext_for_mime(mime_type: str) -> str:
    ext = MIME_TO_EXT.get(mime_type.lower())
    if ext is None:
        raise ValueError(f"Unsupported output mime type: {mime_type!r}")
    return ext


def record_finish_reason(record: dict[str, Any]) -> str | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    candidates = response.get("candidates")
    if not candidates:
        return None
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return None
    finish_reason = candidate.get("finishReason")
    if finish_reason is None:
        return None
    return str(finish_reason)


def parse_record_parts(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    candidates = response.get("candidates")
    if not candidates:
        return None
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return None
    content = candidate.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return None
    return parts


def response_parts(record: dict[str, Any]) -> list[dict[str, Any]]:
    parts = parse_record_parts(record)
    if parts is None:
        finish_reason = record_finish_reason(record)
        if finish_reason is not None:
            raise RuntimeError(f"Missing response parts (finishReason={finish_reason}).")
        raise RuntimeError("Missing response.candidates[0].content.parts in batch output record.")
    return parts


def save_thought_parts(
    parts: list[dict[str, Any]],
    thoughts_dir: Path,
) -> list[str]:
    saved_paths: list[str] = []
    thought_index = 0
    for part in parts:
        if not is_thought_part(part):
            continue

        inline = part_inline_data(part)
        text = part_field(part, "text")
        if inline is not None:
            mime_type = part_field(inline, "mimeType", "mime_type")
            data_b64 = part_field(inline, "data")
            if mime_type is None or data_b64 is None:
                raise RuntimeError("Thought inline data missing mime type or data.")
            out_path = thoughts_dir / f"thought_{thought_index}{ext_for_mime(mime_type)}"
            out_path.write_bytes(base64.b64decode(data_b64))
            saved_paths.append(str(out_path.name))
            thought_index += 1
        elif text is not None:
            out_path = thoughts_dir / f"thought_{thought_index}.txt"
            out_path.write_text(text, encoding="utf-8")
            saved_paths.append(str(out_path.name))
            thought_index += 1
    return saved_paths


def save_final_image(parts: list[dict[str, Any]], out_dir: Path, stem: str) -> Path:
    final_parts = [part for part in parts if part_inline_data(part) is not None and not is_thought_part(part)]
    if not final_parts:
        raise RuntimeError("No final image part found in response.")
    if len(final_parts) > 1:
        raise RuntimeError(f"Expected one final image part, found {len(final_parts)}.")

    inline = part_inline_data(final_parts[0])
    if inline is None:
        raise RuntimeError("Final image part has no inline data.")
    mime_type = part_field(inline, "mimeType", "mime_type")
    data_b64 = part_field(inline, "data")
    if mime_type is None or data_b64 is None:
        raise RuntimeError("Final image inline data missing mime type or data.")

    out_path = out_dir / f"{stem}{ext_for_mime(mime_type)}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(data_b64))
    return out_path


def cmd_parse(args: argparse.Namespace) -> None:
    paths = RunPaths.from_flags(test=args.test, retry=args.retry)
    client = get_gemini_client()

    state = load_batch_state(paths.batch_dir)
    batch_id = args.batch_id or state["batch_id"]

    batch, state, operation = sync_batch_state(client, batch_id, paths.batch_dir)
    print_batch_progress(batch_id, batch, paths.batch_dir, operation, state)

    status = batch_status_name(batch)
    if status in PENDING_BATCH_STATUSES:
        print("batch not ready; run parse again later")
        return

    if status in FAILED_BATCH_STATUSES:
        dest_file_name = state.get("dest_file_name")
        if dest_file_name:
            output_path = paths.batch_dir / "output.jsonl"
            output_path.write_text(download_result_file(client, dest_file_name), encoding="utf-8")
            print(f"output_jsonl={output_path}")
        raise_batch_terminal_error(batch_id, status, batch, operation)

    if status not in SUCCESS_BATCH_STATUSES:
        raise RuntimeError(f"Batch {batch_id} has unexpected status={status}")

    dest_file_name = None
    if batch.dest is not None and batch.dest.file_name:
        dest_file_name = batch.dest.file_name
    if not dest_file_name:
        raise RuntimeError(f"Batch {batch_id} has no dest.file_name.")

    output_path = paths.batch_dir / "output.jsonl"
    output_path.write_text(download_result_file(client, dest_file_name), encoding="utf-8")

    manifest_path = paths.batch_dir / "build_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}. Run build first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_key = {item["custom_id"]: item for item in manifest["items"]}

    saved = 0
    thoughts_saved = 0
    failed: list[dict[str, Any]] = []
    thought_records: list[dict[str, Any]] = []

    with output_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("key")
            if key is None:
                raise KeyError(f"Missing key at output line {line_no}")

            if record.get("error") is not None:
                failed.append({"custom_id": key, "line_no": line_no, "record": record})
                continue

            item = manifest_by_key.get(key)
            if item is None:
                raise KeyError(f"key {key!r} not found in build_manifest.json")

            parts = parse_record_parts(record)
            if parts is None:
                finish_reason = record_finish_reason(record)
                reason = f"finishReason={finish_reason}" if finish_reason else "missing response parts"
                failed.append(
                    {
                        "custom_id": key,
                        "line_no": line_no,
                        "reason": reason,
                        "record": record,
                    }
                )
                continue

            sample_thoughts_dir = paths.thoughts_dir / item["task_type"] / item["stem"]
            thought_files = save_thought_parts(parts, sample_thoughts_dir)
            if thought_files:
                thoughts_saved += len(thought_files)
                thought_records.append(
                    {
                        "custom_id": key,
                        "task_type": item["task_type"],
                        "stem": item["stem"],
                        "thought_dir": str(sample_thoughts_dir.relative_to(INFERENCE_DIR)),
                        "files": thought_files,
                    }
                )

            try:
                save_final_image(parts, paths.output_dir / item["task_type"], item["stem"])
            except RuntimeError as exc:
                failed.append(
                    {
                        "custom_id": key,
                        "line_no": line_no,
                        "reason": str(exc),
                        "record": record,
                    }
                )
                continue
            saved += 1

    summary = {
        "mode": paths.mode,
        "batch_id": batch_id,
        "saved": saved,
        "thoughts_saved": thoughts_saved,
        "failed": len(failed),
        "output_dir": str(paths.output_dir.relative_to(INFERENCE_DIR)),
        "thoughts_dir": str(paths.thoughts_dir.relative_to(INFERENCE_DIR)),
        "thought_records": thought_records,
        "failures": failed,
    }
    write_json(paths.batch_dir / "parse_summary.json", summary)

    print(f"saved={saved} thoughts_saved={thoughts_saved} failed={len(failed)}")
    print(f"output_dir={paths.output_dir}")
    print(f"thoughts_dir={paths.thoughts_dir}")
    print(f"parse_summary={paths.batch_dir / 'parse_summary.json'}")

    if failed:
        raise RuntimeError(f"Parse completed with {len(failed)} failed request(s). See parse_summary.json.")


def add_run_mode_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test", action="store_true", help="Use batch_test/ artifacts.")
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Use batch_retry/ for batch artifacts; outputs still go to outputs/.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini batch image edit pipeline for med-banana inference data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build", help="Build batch requests JSONL from local data.")
    add_run_mode_flags(build_cmd)
    build_cmd.add_argument("--stem", help="Sample stem for --test mode.")
    build_cmd.add_argument(
        "--stems-file",
        help="Build only listed custom_id/stem entries (one per line). Use with --retry.",
    )
    build_cmd.set_defaults(func=cmd_build)

    submit_cmd = subparsers.add_parser("submit", help="Upload JSONL and create Gemini batch job.")
    add_run_mode_flags(submit_cmd)
    submit_cmd.add_argument("--wait", action="store_true", help="Poll until batch completes.")
    submit_cmd.set_defaults(func=cmd_submit)

    parse_cmd = subparsers.add_parser(
        "parse",
        help="Sync batch state; parse images when completed (pending exits 0).",
    )
    add_run_mode_flags(parse_cmd)
    parse_cmd.add_argument("--batch-id", help="Optional batch id override.")
    parse_cmd.set_defaults(func=cmd_parse)

    extract_cmd = subparsers.add_parser(
        "extract-failures",
        help="Write failed_stems.txt from batch/parse_summary.json.",
    )
    extract_cmd.add_argument("--test", action="store_true", help="Read batch_test/parse_summary.json.")
    extract_cmd.set_defaults(func=cmd_extract_failures)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
