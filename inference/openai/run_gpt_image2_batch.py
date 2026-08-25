#!/usr/bin/env python3
"""GPT Image 2 batch image edit pipeline for med-banana inference data."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OPENAI_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = OPENAI_DIR.parent
DATA_DIR = INFERENCE_DIR / "data"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MODEL = "gpt-image-2"
BATCH_ENDPOINT = "/v1/images/edits"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
PENDING_BATCH_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}
FAILED_BATCH_STATUSES = {"failed", "expired", "cancelled"}


@dataclass(frozen=True)
class RunPaths:
    mode: str
    batch_dir: Path
    output_dir: Path

    @classmethod
    def from_test_flag(cls, test: bool) -> RunPaths:
        if test:
            return cls(mode="test", batch_dir=OPENAI_DIR / "batch_test", output_dir=OPENAI_DIR / "outputs_test")
        return cls(mode="full", batch_dir=OPENAI_DIR / "batch", output_dir=OPENAI_DIR / "outputs")


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
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OpenAI API key is missing. Set the OPENAI_API_KEY environment variable."
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


def select_samples(test: bool, stem: str | None) -> list[Sample]:
    if test:
        if stem:
            return [find_sample_by_stem(stem)]
        addition_samples = collect_samples("addition")
        if not addition_samples:
            raise RuntimeError(f"No addition samples found in {DATA_DIR / 'addition'}")
        return [addition_samples[0]]
    return collect_all_samples()


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0]
    if mime_type is None:
        raise ValueError(f"Cannot infer MIME type for {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_request_line(sample: Sample) -> dict[str, Any]:
    return {
        "custom_id": sample.custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": MODEL,
            "images": [{"image_url": image_to_data_url(sample.image_path)}],
            "prompt": sample.prompt,
            "quality": "low",
            "size": "1024x1024",
            "output_format": "jpeg",
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_build(args: argparse.Namespace) -> None:
    paths = RunPaths.from_test_flag(args.test)
    samples = select_samples(test=args.test, stem=args.stem)
    paths.batch_dir.mkdir(parents=True, exist_ok=True)

    requests_path = paths.batch_dir / "requests.jsonl"
    manifest_path = paths.batch_dir / "build_manifest.json"

    manifest = {
        "mode": paths.mode,
        "model": MODEL,
        "endpoint": BATCH_ENDPOINT,
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


def get_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=resolve_api_key())


def cmd_submit(args: argparse.Namespace) -> None:
    paths = RunPaths.from_test_flag(args.test)
    requests_path = paths.batch_dir / "requests.jsonl"
    if not requests_path.exists():
        raise FileNotFoundError(f"Missing {requests_path}. Run build first.")

    client = get_openai_client()
    with requests_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    metadata = {"project": "med-banana-inference", "mode": paths.mode}
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=BATCH_ENDPOINT,
        completion_window="24h",
        metadata=metadata,
    )

    state = {
        "mode": paths.mode,
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": BATCH_ENDPOINT,
        "metadata": metadata,
    }
    write_json(paths.batch_dir / "batch_state.json", state)
    print(f"batch_id={batch.id}")
    print(f"status={batch.status}")

    if args.wait:
        poll_batch(client, batch.id, paths.batch_dir)


def sync_batch_state(client, batch_id: str, batch_dir: Path) -> tuple[Any, dict[str, Any]]:
    batch = client.batches.retrieve(batch_id)
    state = load_batch_state(batch_dir)
    counts = batch.request_counts
    state["status"] = batch.status
    state["output_file_id"] = batch.output_file_id
    state["error_file_id"] = batch.error_file_id
    state["request_counts"] = {
        "completed": counts.completed,
        "failed": counts.failed,
        "total": counts.total,
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(batch_dir / "batch_state.json", state)
    return batch, state


def print_batch_progress(batch_id: str, batch: Any, batch_dir: Path) -> None:
    counts = batch.request_counts
    print(
        f"batch_id={batch_id} status={batch.status} "
        f"completed={counts.completed} failed={counts.failed} total={counts.total}"
    )
    print(f"batch_state={batch_dir / 'batch_state.json'}")


def poll_batch(client, batch_id: str, batch_dir: Path, interval_sec: int = 30) -> None:
    while True:
        batch, _ = sync_batch_state(client, batch_id, batch_dir)
        print_batch_progress(batch_id, batch, batch_dir)

        if batch.status in TERMINAL_BATCH_STATUSES:
            if batch.status != "completed":
                raise RuntimeError(f"Batch {batch_id} finished with status={batch.status}")
            return
        time.sleep(interval_sec)


def load_batch_state(batch_dir: Path) -> dict[str, Any]:
    state_path = batch_dir / "batch_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing {state_path}. Run submit first.")
    return json.loads(state_path.read_text(encoding="utf-8"))


def download_file_content(client, file_id: str) -> str:
    response = client.files.content(file_id)
    return response.text


def cmd_parse(args: argparse.Namespace) -> None:
    paths = RunPaths.from_test_flag(args.test)
    client = get_openai_client()

    state = load_batch_state(paths.batch_dir)
    batch_id = args.batch_id or state["batch_id"]

    batch, _ = sync_batch_state(client, batch_id, paths.batch_dir)
    print_batch_progress(batch_id, batch, paths.batch_dir)

    if batch.status in PENDING_BATCH_STATUSES:
        print("batch not ready; run parse again later")
        return

    if batch.status in FAILED_BATCH_STATUSES:
        raise RuntimeError(f"Batch {batch_id} finished with status={batch.status}")

    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} has unexpected status={batch.status}")

    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output_file_id.")

    output_path = paths.batch_dir / "output.jsonl"
    output_path.write_text(download_file_content(client, batch.output_file_id), encoding="utf-8")

    errors_path = paths.batch_dir / "errors.jsonl"
    if batch.error_file_id:
        errors_path.write_text(download_file_content(client, batch.error_file_id), encoding="utf-8")
    elif errors_path.exists():
        errors_path.unlink()

    manifest_path = paths.batch_dir / "build_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}. Run build first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_custom_id = {item["custom_id"]: item for item in manifest["items"]}

    saved = 0
    failed: list[dict[str, Any]] = []

    with output_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record["custom_id"]
            response = record.get("response")
            if response is None or response.get("status_code") != 200:
                failed.append({"custom_id": custom_id, "line_no": line_no, "record": record})
                continue

            body = response["body"]
            data = body.get("data")
            if not data:
                raise RuntimeError(f"Missing response.body.data for custom_id={custom_id!r} at line {line_no}")
            b64_json = data[0].get("b64_json")
            if not b64_json:
                raise RuntimeError(f"Missing b64_json for custom_id={custom_id!r} at line {line_no}")

            item = manifest_by_custom_id.get(custom_id)
            if item is None:
                raise KeyError(f"custom_id {custom_id!r} not found in build_manifest.json")

            out_path = paths.output_dir / item["task_type"] / f"{item['stem']}.jpeg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(base64.b64decode(b64_json))
            saved += 1

    if errors_path.exists():
        with errors_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                failed.append({"custom_id": record.get("custom_id"), "line_no": line_no, "record": record})

    summary = {
        "mode": paths.mode,
        "batch_id": batch_id,
        "saved": saved,
        "failed": len(failed),
        "output_dir": str(paths.output_dir.relative_to(INFERENCE_DIR)),
        "failures": failed,
    }
    write_json(paths.batch_dir / "parse_summary.json", summary)

    print(f"saved={saved} failed={len(failed)}")
    print(f"output_dir={paths.output_dir}")
    print(f"parse_summary={paths.batch_dir / 'parse_summary.json'}")

    if failed:
        raise RuntimeError(f"Parse completed with {len(failed)} failed request(s). See parse_summary.json.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT Image 2 batch image edit pipeline for med-banana inference data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build", help="Build batch requests JSONL from local data.")
    build_cmd.add_argument("--test", action="store_true", help="Build a single-request test batch.")
    build_cmd.add_argument("--stem", help="Sample stem for --test mode.")
    build_cmd.set_defaults(func=cmd_build)

    submit_cmd = subparsers.add_parser("submit", help="Upload JSONL and create OpenAI batch job.")
    submit_cmd.add_argument("--test", action="store_true", help="Use batch_test/ artifacts.")
    submit_cmd.add_argument("--wait", action="store_true", help="Poll until batch completes.")
    submit_cmd.set_defaults(func=cmd_submit)

    parse_cmd = subparsers.add_parser(
        "parse",
        help="Sync batch state; parse images when completed (pending exits 0).",
    )
    parse_cmd.add_argument("--test", action="store_true", help="Use batch_test/ artifacts.")
    parse_cmd.add_argument("--batch-id", help="Optional batch id override.")
    parse_cmd.set_defaults(func=cmd_parse)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
