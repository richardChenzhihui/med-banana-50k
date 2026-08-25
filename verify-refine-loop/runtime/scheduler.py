#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

from prompts import (
    add_update_prompt,
    add_verify_prompt,
    build_last_attempt_text,
    remove_update_prompt,
    remove_verify_prompt,
)

# Repository root, used to anchor relative paths coming from the config file. The
# launcher scripts cd into runtime/ before exec'ing this module, so resolving a relative
# output_root against the current working directory would bury run outputs at
# runtime/verify-refine-loop/outputs/ instead of the documented location.
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value: str) -> Path:
    """Expand `value` and, when it is relative, anchor it at the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _to_data_url_jpeg(image_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def _to_data_url_png(image: Image.Image) -> str:
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _decode_data_url_or_b64(data: str) -> bytes:
    payload = data.split(",", 1)[1] if "," in data else data
    return base64.b64decode(payload)


def _extract_json_block(content: str) -> str:
    if "```json" in content:
        return content.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in content:
        return content.split("```", 1)[1].split("```", 1)[0].strip()
    return content.strip()


def _contains_negative_rules(text: str) -> bool:
    lower = text.lower()
    hits = 0
    for token in (" no ", "no ", "do not", "don't", "never", "without "):
        if token in lower:
            hits += 1
    return hits >= 1


def _parse_refiner_dual_prompt_plain(block: str) -> tuple[str, str]:
    # Fallback for non-JSON legacy/refusal outputs:
    # paragraph-1 = main prompt, paragraph-2+ = negative prompt clauses.
    paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        prompt = paragraphs[0]
        neg = " ".join(paragraphs[1:]).strip()
    else:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            raise ValueError("non-JSON refiner output has insufficient lines for prompt/negative_prompt")
        prompt = lines[0]
        neg = " ".join(lines[1:]).strip()

    if neg.lower().startswith("negative_prompt:"):
        neg = neg.split(":", 1)[1].strip()
    elif neg.lower().startswith("negative prompt:"):
        neg = neg.split(":", 1)[1].strip()

    if not prompt:
        raise ValueError("empty prompt from non-JSON refiner output")
    if not neg:
        raise ValueError("empty negative_prompt from non-JSON refiner output")
    if not _contains_negative_rules(neg):
        raise ValueError("non-JSON fallback parsed negative_prompt but found no negative-rule phrasing")
    return prompt, neg


def _parse_refiner_dual_prompt(content: str) -> tuple[str, str]:
    block = _extract_json_block(content)
    try:
        data = json.loads(block)
    except json.JSONDecodeError as e:
        try:
            return _parse_refiner_dual_prompt_plain(block)
        except ValueError as e_plain:
            raise ValueError(
                f"refiner output is not valid JSON: {e}; fallback parse also failed: {e_plain}"
            ) from e_plain
    if not isinstance(data, dict):
        raise ValueError("refiner JSON root must be an object")
    for key in ("prompt", "negative_prompt"):
        if key not in data:
            raise ValueError(f'missing "{key}" in refiner JSON')
    prompt_raw = data["prompt"]
    neg_raw = data["negative_prompt"]
    if not isinstance(prompt_raw, str) or not isinstance(neg_raw, str):
        raise ValueError("prompt and negative_prompt must be JSON strings")
    prompt = prompt_raw.strip()
    neg = neg_raw.strip()
    if not prompt:
        raise ValueError("empty prompt in refiner JSON")
    if not neg:
        raise ValueError("empty negative_prompt in refiner JSON")
    return prompt, neg


def _safe_name_fragment(s: str, max_len: int = 80) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    return out[:max_len] if len(out) > max_len else out


@dataclass
class _ActiveTaskState:
    backend: str
    task_key: str
    task_type: str
    disease: str
    image_name: str
    round_num: int = 0
    step: str = "queued"
    attempt: int = 0
    max_attempts: int = 1
    qualified: bool | None = None
    message: str = ""
    preview_path: str | None = None
    updated_at: float = field(default_factory=time.time)


class RunProgress:
    """Thread-safe live progress: tqdm bar + status.json + preview JPEG."""

    def __init__(
        self,
        *,
        total_jobs: int,
        live_dir: Path,
        max_api_retries: int,
        max_rounds: int,
        enabled: bool = True,
    ):
        if total_jobs < 1:
            raise ValueError(f"total_jobs must be >= 1, got {total_jobs}")
        self.total_jobs = total_jobs
        self.live_dir = live_dir
        self.max_api_retries = max_api_retries
        self.max_rounds = max_rounds
        self.enabled = enabled
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.live_dir / "status.json"
        self.current_preview_path = self.live_dir / "current_preview.jpeg"
        self.lock = threading.Lock()
        self.completed = 0
        self.stats: dict[str, int] = {
            "success": 0,
            "failed": 0,
            "api_failed": 0,
            "skipped": 0,
            "crashed": 0,
        }
        self.active: dict[str, _ActiveTaskState] = {}
        self.recent_done: list[dict[str, Any]] = []
        self._closed = False
        self._pbar = None
        if self.enabled and tqdm is not None and sys.stderr.isatty():
            self._pbar = tqdm(
                total=total_jobs,
                desc="scheduler",
                unit="job",
                dynamic_ncols=True,
                file=sys.stderr,
            )
        self._flush_status()

    def _job_id(self, backend: str, task_key: str) -> str:
        return f"{backend}|{task_key}"

    def _save_preview(self, image: Image.Image, dest: Path) -> None:
        thumb = image.copy()
        thumb.thumbnail((512, 512), Image.Resampling.LANCZOS)
        thumb.save(dest, "JPEG", quality=85)

    def _flush_status(self) -> None:
        with self.lock:
            active_list = []
            for st in self.active.values():
                active_list.append(
                    {
                        "backend": st.backend,
                        "task": st.task_key,
                        "task_type": st.task_type,
                        "disease": st.disease,
                        "image": st.image_name,
                        "round": st.round_num,
                        "max_rounds": self.max_rounds,
                        "step": st.step,
                        "attempt": st.attempt,
                        "max_attempts": st.max_attempts,
                        "qualified": st.qualified,
                        "message": st.message,
                        "preview_path": st.preview_path,
                        "updated_at": st.updated_at,
                    }
                )
            active_list.sort(key=lambda x: x["updated_at"], reverse=True)
            payload = {
                "updated_at": time.time(),
                "total_jobs": self.total_jobs,
                "completed_jobs": self.completed,
                "max_api_retries": self.max_api_retries,
                "max_rounds": self.max_rounds,
                "stats": dict(self.stats),
                "active": active_list,
                "recent_completed": list(self.recent_done[-20:]),
                "current_preview": str(self.current_preview_path),
            }
            tmp = self.status_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp.replace(self.status_path)

        if self._pbar is not None:
            parts = []
            for st in active_list[:2]:
                retry = ""
                if st["attempt"] > 0:
                    retry = f" retry {st['attempt']}/{st['max_attempts']}"
                q = ""
                if st["qualified"] is not None:
                    q = f" q={'Y' if st['qualified'] else 'N'}"
                parts.append(
                    f"{st['backend']}|r{st['round']}/{st['max_rounds']}|{st['step']}{retry}{q}"
                )
            postfix = " | ".join(parts) if parts else f"done {self.completed}/{self.total_jobs}"
            self._pbar.set_postfix_str(postfix[:180], refresh=False)

    def task_start(
        self,
        *,
        backend: str,
        task_key: str,
        task_type: str,
        disease: str,
        image_path: Path,
    ) -> None:
        jid = self._job_id(backend, task_key)
        with self.lock:
            self.active[jid] = _ActiveTaskState(
                backend=backend,
                task_key=task_key,
                task_type=task_type,
                disease=disease,
                image_name=image_path.name,
                step="start",
                message="task started",
            )
        self._flush_status()

    def update(
        self,
        *,
        backend: str,
        task_key: str,
        round_num: int | None = None,
        step: str | None = None,
        attempt: int | None = None,
        max_attempts: int | None = None,
        qualified: bool | None = None,
        message: str | None = None,
        preview_image: Image.Image | None = None,
        preview_tag: str | None = None,
    ) -> None:
        jid = self._job_id(backend, task_key)
        with self.lock:
            st = self.active.get(jid)
            if st is None:
                return
            if round_num is not None:
                st.round_num = round_num
            if step is not None:
                st.step = step
            if attempt is not None:
                st.attempt = attempt
            if max_attempts is not None:
                st.max_attempts = max_attempts
            if qualified is not None:
                st.qualified = qualified
            if message is not None:
                st.message = message
            st.updated_at = time.time()

        if preview_image is not None:
            round_for_tag = round_num if round_num is not None else 0
            with self.lock:
                st_locked = self.active.get(jid)
                if st_locked is not None and round_num is None:
                    round_for_tag = st_locked.round_num
            tag = preview_tag or f"{backend}__{_safe_name_fragment(task_key)}__r{round_for_tag}"
            per_task = self.live_dir / f"{tag}.jpeg"
            self._save_preview(preview_image, per_task)
            self._save_preview(preview_image, self.current_preview_path)
            with self.lock:
                st = self.active.get(jid)
                if st is not None:
                    st.preview_path = str(per_task)
        self._flush_status()

    def task_done(
        self,
        *,
        backend: str,
        task_key: str,
        status: str,
        rounds: int | None = None,
        error: str | None = None,
    ) -> None:
        jid = self._job_id(backend, task_key)
        rec: dict[str, Any] = {
            "backend": backend,
            "task": task_key,
            "status": status,
            "ts": time.time(),
        }
        if rounds is not None:
            rec["rounds"] = rounds
        if error:
            rec["error"] = error
        with self.lock:
            self.active.pop(jid, None)
            self.completed += 1
            if status in self.stats:
                self.stats[status] += 1
            else:
                self.stats["crashed"] += 1
            self.recent_done.append(rec)
            if self._pbar is not None:
                self._pbar.update(1)
        self._flush_status()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pbar is not None:
            self._pbar.close()
        self._flush_status()


def _truncate_str_bytes(s: str, max_bytes: int) -> tuple[str, bool]:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s, False
    return b[:max_bytes].decode("utf-8", errors="replace"), True


@dataclass
class ChatVlmResult:
    status_code: int
    raw_response_text: str
    parsed_json: dict[str, Any] | None
    content_text: str


class ChatVlmFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        request_url: str,
        http_status: int | None = None,
        raw_response_text: str | None = None,
    ):
        super().__init__(message)
        self.request_url = request_url
        self.http_status = http_status
        self.raw_response_text = raw_response_text


class EditServerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        request_url: str,
        http_status: int | None,
        raw_response_text: str,
    ):
        super().__init__(message)
        self.request_url = request_url
        self.http_status = http_status
        self.raw_response_text = raw_response_text


def _extract_assistant_content_text(data: dict[str, Any], *, request_url: str, raw_response_text: str) -> str:
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    if not isinstance(content, str):
        raise ChatVlmFailure(
            "unexpected vllm content type",
            request_url=request_url,
            http_status=200,
            raw_response_text=raw_response_text,
        )
    return content


def _dataset_name_for_prompt(dataset_code: str) -> str:
    m = {
        "brain_mri": "brain MRI",
        "chest_xray": "chest X-ray",
        "fundus": "fundus",
    }
    if dataset_code not in m:
        raise ValueError(f"unsupported dataset code in stem: {dataset_code}")
    return m[dataset_code]


def _parse_task_stem(stem: str) -> tuple[str, str, str]:
    parts = stem.split("__")
    if len(parts) >= 6 and parts[0] == "med":
        dataset_code, disease, task_type = parts[1], parts[2], parts[3]
    elif len(parts) >= 4 and parts[0] in {"brain_mri", "chest_xray", "fundus"}:
        dataset_code, disease, task_type = parts[0], parts[1], parts[2]
    else:
        raise ValueError(
            f"invalid stem format (expected med__dataset__disease__task__... "
            f"or dataset__disease__task__id): {stem}"
        )
    if task_type not in {"addition", "removal"}:
        raise ValueError(f"unsupported task_type in stem: {stem}")
    return dataset_code, disease, task_type


@dataclass
class InputPair:
    stem: str
    image_path: Path
    prompt_path: Path
    task_type: str
    disease: str
    dataset_type: str
    initial_prompt: str


class EditServerPool:
    def __init__(self, urls: list[str], health_path: str = "/health", edit_path: str = "/edit"):
        if not urls:
            raise ValueError("empty edit server list")
        self.urls = [u.rstrip("/") for u in urls]
        self.health_path = health_path
        self.edit_path = edit_path
        self.active = {u: 0 for u in self.urls}
        self.lock = threading.Lock()

    def acquire(self) -> str:
        with self.lock:
            picked = min(self.urls, key=lambda u: self.active[u])
            self.active[picked] += 1
            return picked

    def release(self, url: str) -> None:
        with self.lock:
            self.active[url] -= 1
            if self.active[url] < 0:
                raise RuntimeError(f"negative active count for {url}")


class Store:
    def __init__(self, root: Path, suffix: str, raw_log_max_bytes: int = 65536):
        self.output_dir = root / suffix
        self.failed_dir = root / f"{suffix}-failed"
        self.trace_dir = root / f"{suffix}-traces"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.raw_log_max_bytes = max(1024, int(raw_log_max_bytes))
        self.progress_file = self.output_dir / "progress.json"
        self.api_failures_file = self.output_dir / "api_failures.json"
        self.failed_summary_file = self.failed_dir / "failed_summary.json"
        self.final_prompts_file = self.output_dir / "final_prompts.json"
        self.conversations_file = self.output_dir / "all_conversations.json"
        self.stats_file = self.output_dir / "stats.json"
        self.raw_events_file = self.output_dir / "raw_events.jsonl"
        self.progress = self._load_json(self.progress_file, {})
        self.api_failures = self._load_json(self.api_failures_file, [])
        self.failed_summary = self._load_json(self.failed_summary_file, [])
        self.final_prompts = self._load_json(self.final_prompts_file, {})
        self.all_conversations = self._load_json(self.conversations_file, {})
        self.lock = threading.RLock()

    def _load_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def _save_json(self, path: Path, value: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)

    def append_event(self, event: dict[str, Any]) -> None:
        evt = dict(event)
        evt["ts"] = time.time()
        for key in ("raw_text", "raw_response_text", "raw_llm_output", "json_block"):
            if key in evt and isinstance(evt[key], str):
                t, truncated = _truncate_str_bytes(evt[key], self.raw_log_max_bytes)
                evt[key] = t
                if truncated:
                    evt[f"{key}_bytes_truncated"] = True
        line = json.dumps(evt, ensure_ascii=False, default=str) + "\n"
        with self.lock:
            with open(self.raw_events_file, "a", encoding="utf-8") as f:
                f.write(line)

    def save_all(self) -> None:
        with self.lock:
            self._save_json(self.progress_file, self.progress)
            self._save_json(self.api_failures_file, self.api_failures)
            self._save_json(self.failed_summary_file, self.failed_summary)
            self._save_json(self.final_prompts_file, self.final_prompts)
            self._save_json(self.conversations_file, self.all_conversations)
            self._save_json(self.stats_file, self.build_stats())

    def save_trace(self, task_key: str, conversation: dict[str, Any]) -> None:
        trace_path = self.trace_dir / f"{task_key}.json"
        self._save_json(trace_path, conversation)

    def reset_task_for_retry(self, task_key: str) -> bool:
        with self.lock:
            had_progress = self.progress.pop(task_key, None) is not None
            old_summary_len = len(self.failed_summary)
            self.failed_summary = [x for x in self.failed_summary if x.get("task") != task_key]
            had_summary = len(self.failed_summary) != old_summary_len
            had_conversation = self.all_conversations.pop(task_key, None) is not None
            had_final_prompt = self.final_prompts.pop(task_key, None) is not None
            changed = had_progress or had_summary or had_conversation or had_final_prompt
            if changed:
                self.save_all()
            return changed

    def build_stats(self) -> dict[str, Any]:
        status_counter = {"success": 0, "failed": 0, "api_failed": 0, "skipped": 0}
        for v in self.progress.values():
            if v in status_counter:
                status_counter[v] += 1
        rounds_counter: dict[str, int] = {}
        for record in self.final_prompts.values():
            rounds = str(record.get("rounds", "unknown"))
            rounds_counter[rounds] = rounds_counter.get(rounds, 0) + 1
        return {
            "status_counter": status_counter,
            "rounds_counter": rounds_counter,
            "total_progress_items": len(self.progress),
            "api_failures_count": len(self.api_failures),
            "failed_summary_count": len(self.failed_summary),
        }


class Scheduler:
    def __init__(
        self,
        config: dict[str, Any],
        selected_backends: list[str] | None = None,
        progress: RunProgress | None = None,
    ):
        self.cfg = config
        self.progress = progress
        self.http_timeout = int(config.get("http_timeout_sec", 300))
        self.raw_log_max_bytes = int(config.get("raw_log_max_bytes", 65536))
        self.max_retries = int(config.get("max_retries", 3))
        _mar = config.get("max_api_retries")
        self.max_api_retries = int(_mar) if _mar is not None else self.max_retries
        if self.max_api_retries < 1:
            raise ValueError(f"max_api_retries must be >= 1, got {self.max_api_retries}")
        self.max_rounds = int(config.get("max_rounds", 5))
        if self.max_rounds < 2:
            raise ValueError(f"max_rounds must be >= 2, got {self.max_rounds}")
        self.refine_budget = int(config.get("refine_budget", self.max_rounds - 1))
        if self.refine_budget < 1:
            raise ValueError(f"refine_budget must be >= 1, got {self.refine_budget}")
        if self.refine_budget >= self.max_rounds:
            raise ValueError(
                f"refine_budget must be < max_rounds, got refine_budget={self.refine_budget}, max_rounds={self.max_rounds}"
            )
        self.refiner_url = config["refiner"]["base_url"].rstrip("/")
        self.refiner_model = config["refiner"]["model"]
        self.verifier_url = config["verifier"]["base_url"].rstrip("/")
        self.verifier_model = config["verifier"]["model"]
        self.refiner_key = config["refiner"].get("api_key", "EMPTY")
        self.verifier_key = config["verifier"].get("api_key", "EMPTY")
        default_vllm_extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
        self.vllm_chat_completions_extra: dict[str, Any] = dict(
            config.get("vllm_chat_completions_extra", default_vllm_extra)
        )
        self.output_root = resolve_repo_path(config["output_root"])
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._health_events_file = self.output_root / "health_check_events.jsonl"
        self._health_log_lock = threading.Lock()
        self.concurrent_per_backend = int(config.get("concurrent_per_backend", 2))
        self.default_edit_params = config.get("edit_defaults", {})

        self.backend_pools: dict[str, EditServerPool] = {}
        self.stores: dict[tuple[str, str], Store] = {}
        raw_edit_servers = config["edit_servers"]
        available_backends = list(raw_edit_servers.keys())
        if selected_backends is None:
            backends = available_backends
        else:
            backends = selected_backends
            for backend in backends:
                if backend not in raw_edit_servers:
                    raise ValueError(
                        f"unknown backend in --backends: {backend}, available={available_backends}"
                    )

        for backend in backends:
            server_conf = raw_edit_servers[backend]
            if isinstance(server_conf, list):
                urls = server_conf
                health_path = "/health"
                edit_path = "/edit"
            elif isinstance(server_conf, dict):
                urls = server_conf.get("urls", [])
                health_path = server_conf.get("health_path", "/health")
                edit_path = server_conf.get("edit_path", "/edit")
            else:
                raise ValueError(f"invalid edit_servers config for backend={backend}")
            self.backend_pools[backend] = EditServerPool(
                urls=urls,
                health_path=health_path,
                edit_path=edit_path,
            )
            self.stores[(backend, "addition")] = Store(self.output_root / backend, "edit", self.raw_log_max_bytes)
            self.stores[(backend, "removal")] = Store(self.output_root / backend, "remove", self.raw_log_max_bytes)

    def _prog(
        self,
        *,
        backend: str,
        task_key: str,
        round_num: int | None = None,
        step: str | None = None,
        attempt: int | None = None,
        max_attempts: int | None = None,
        qualified: bool | None = None,
        message: str | None = None,
        preview_image: Image.Image | None = None,
        preview_tag: str | None = None,
    ) -> None:
        if self.progress is None:
            return
        self.progress.update(
            backend=backend,
            task_key=task_key,
            round_num=round_num,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            qualified=qualified,
            message=message,
            preview_image=preview_image,
            preview_tag=preview_tag,
        )

    def health_check(self) -> None:
        health_timeout = min(30, max(1, self.http_timeout))
        for backend, pool in self.backend_pools.items():
            for url in pool.urls:
                full_url = f"{url}{pool.health_path}"
                try:
                    r = requests.get(full_url, timeout=health_timeout)
                except requests.RequestException as e:
                    self._append_health_event(
                        {
                            "event_type": "api_failure",
                            "step": "health_check_edit",
                            "backend": backend,
                            "request_url": full_url,
                            "error_type": "request_exception",
                            "error": str(e),
                            "timeout_sec": health_timeout,
                        }
                    )
                    raise RuntimeError(
                        f"edit server health request failed backend={backend} url={full_url}: {e}"
                    ) from e
                if r.status_code != 200:
                    self._append_health_event(
                        {
                            "event_type": "api_failure",
                            "step": "health_check_edit",
                            "backend": backend,
                            "request_url": full_url,
                            "error_type": "http_error",
                            "http_status": r.status_code,
                            "raw_response_text": r.text,
                            "error": f"unhealthy status {r.status_code}",
                            "timeout_sec": health_timeout,
                        }
                    )
                    raise RuntimeError(
                        f"edit server unhealthy backend={backend} url={full_url}: {r.text}"
                    )
        for name, base_url, api_key in (
            ("refiner", self.refiner_url, self.refiner_key),
            ("verifier", self.verifier_url, self.verifier_key),
        ):
            models_url = f"{base_url}/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            try:
                r = requests.get(models_url, headers=headers, timeout=health_timeout)
            except requests.RequestException as e:
                self._append_health_event(
                    {
                        "event_type": "api_failure",
                        "step": "health_check_vllm",
                        "service": name,
                        "request_url": models_url,
                        "error_type": "request_exception",
                        "error": str(e),
                        "timeout_sec": health_timeout,
                    }
                )
                raise RuntimeError(f"{name} vllm health request failed {base_url}: {e}") from e
            if r.status_code != 200:
                self._append_health_event(
                    {
                        "event_type": "api_failure",
                        "step": "health_check_vllm",
                        "service": name,
                        "request_url": models_url,
                        "error_type": "http_error",
                        "http_status": r.status_code,
                        "raw_response_text": r.text,
                        "error": f"unhealthy status {r.status_code}",
                        "timeout_sec": health_timeout,
                    }
                )
                raise RuntimeError(f"{name} vllm unhealthy {base_url}: {r.text}")

    def plan_jobs(
        self,
        pairs: list[InputPair],
        *,
        retry_failed: bool = False,
        reset_failed: bool = True,
    ) -> list[tuple[str, InputPair]]:
        jobs: list[tuple[str, InputPair]] = []
        for pair in pairs:
            for backend in self.backend_pools:
                store = self.stores[(backend, pair.task_type)]
                task_key = f"{pair.stem}-{pair.disease}"
                if retry_failed:
                    if store.progress.get(task_key) != "failed":
                        continue
                    if reset_failed:
                        store.reset_task_for_retry(task_key)
                jobs.append((backend, pair))
        return jobs

    def _append_health_event(self, event: dict[str, Any]) -> None:
        evt = dict(event)
        evt["ts"] = time.time()
        mb = self.raw_log_max_bytes
        for key in ("raw_text", "raw_response_text", "raw_llm_output", "json_block"):
            if key in evt and isinstance(evt[key], str):
                t, truncated = _truncate_str_bytes(evt[key], mb)
                evt[key] = t
                if truncated:
                    evt[f"{key}_bytes_truncated"] = True
        line = json.dumps(evt, ensure_ascii=False, default=str) + "\n"
        with self._health_log_lock:
            with open(self._health_events_file, "a", encoding="utf-8") as f:
                f.write(line)

    def _record_api_failure(
        self,
        store: Store,
        *,
        task_key: str,
        backend: str,
        task: InputPair,
        round_num: int,
        step: str,
        error_type: str,
        error: str,
        request_url: str | None = None,
        model: str | None = None,
        timeout_sec: int | None = None,
        http_status: int | None = None,
        raw_response_text: str | None = None,
        raw_llm_output: str | None = None,
        json_block: str | None = None,
        parse_error: str | None = None,
    ) -> None:
        mb = store.raw_log_max_bytes

        def clip(s: str | None) -> tuple[str | None, bool]:
            if s is None:
                return None, False
            return _truncate_str_bytes(s, mb)

        rrt, rrt_tr = clip(raw_response_text)
        rlo, rlo_tr = clip(raw_llm_output)
        jb, jb_tr = clip(json_block)
        rec: dict[str, Any] = {
            "task": task_key,
            "backend": backend,
            "task_type": task.task_type,
            "step": step,
            "round": round_num,
            "error_type": error_type,
            "error": error,
            "image": task.image_path.name,
            "disease": task.disease,
            "request_url": request_url,
            "model": model,
            "timeout_sec": timeout_sec,
            "http_status": http_status,
            "raw_response_text": rrt,
            "raw_response_truncated": rrt_tr,
            "raw_llm_output": rlo,
            "raw_llm_output_truncated": rlo_tr,
            "json_block": jb,
            "json_block_truncated": jb_tr,
            "parse_error": parse_error,
        }
        req_meta: dict[str, Any] = {}
        if request_url is not None:
            req_meta["url"] = request_url
        if model is not None:
            req_meta["model"] = model
        if timeout_sec is not None:
            req_meta["timeout_sec"] = timeout_sec
        resp_meta: dict[str, Any] = {}
        if http_status is not None:
            resp_meta["http_status"] = http_status
        evt: dict[str, Any] = {
            "event_type": "api_failure",
            "task": task_key,
            "backend": backend,
            "task_type": task.task_type,
            "round": round_num,
            "step": step,
            "error_type": error_type,
            "error": error,
            "request_meta": req_meta,
            "response_meta": resp_meta,
            "raw_response_text": rrt,
            "raw_llm_output": rlo,
            "json_block": jb,
            "parse_error": parse_error,
        }
        with store.lock:
            store.api_failures.append(rec)
        store.append_event(evt)

    def _log_llm_output_event(
        self,
        store: Store,
        *,
        task_key: str,
        backend: str,
        task: InputPair,
        round_num: int,
        step: str,
        model: str,
        base_url: str,
        raw_llm_content: str,
        raw_http_response_text: str,
        http_status: int,
    ) -> None:
        req_url = f"{base_url.rstrip('/')}/chat/completions"
        store.append_event(
            {
                "event_type": "llm_output",
                "task": task_key,
                "backend": backend,
                "task_type": task.task_type,
                "round": round_num,
                "step": step,
                "request_meta": {"url": req_url, "model": model},
                "response_meta": {"http_status": http_status},
                "raw_llm_output": raw_llm_content,
                "raw_response_text": raw_http_response_text,
            }
        )

    def _chat_vlm(
        self,
        *,
        store: Store | None,
        log_context: dict[str, Any],
        base_url: str,
        model: str,
        api_key: str,
        instruction: str,
        image_bytes: bytes,
        temperature: float = 0.0,
    ) -> ChatVlmResult:
        request_url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": _to_data_url_jpeg(image_bytes)}},
                    ],
                }
            ],
            "temperature": float(temperature),
        }
        payload.update(self.vllm_chat_completions_extra)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            r = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.http_timeout,
            )
        except requests.RequestException as e:
            if store is not None:
                store.append_event(
                    {
                        **log_context,
                        "event_type": "api_failure",
                        "error_type": "request_exception",
                        "error": str(e),
                        "request_meta": {"url": request_url, "model": model, "timeout_sec": self.http_timeout},
                    }
                )
            raise ChatVlmFailure(
                f"vllm request failed: {e}",
                request_url=request_url,
                http_status=None,
                raw_response_text=None,
            ) from e

        raw_text = r.text
        if r.status_code != 200:
            if store is not None:
                store.append_event(
                    {
                        **log_context,
                        "event_type": "api_failure",
                        "error_type": "http_error",
                        "error": f"vllm request failed ({r.status_code})",
                        "request_meta": {"url": request_url, "model": model, "timeout_sec": self.http_timeout},
                        "response_meta": {"http_status": r.status_code},
                        "raw_response_text": raw_text,
                    }
                )
            raise ChatVlmFailure(
                f"vllm request failed ({r.status_code}): {raw_text}",
                request_url=request_url,
                http_status=r.status_code,
                raw_response_text=raw_text,
            )

        try:
            data = r.json()
        except json.JSONDecodeError as e:
            if store is not None:
                store.append_event(
                    {
                        **log_context,
                        "event_type": "api_failure",
                        "error_type": "invalid_json_response",
                        "error": str(e),
                        "request_meta": {"url": request_url, "model": model, "timeout_sec": self.http_timeout},
                        "response_meta": {"http_status": r.status_code},
                        "raw_response_text": raw_text,
                    }
                )
            raise ChatVlmFailure(
                f"vllm response is not JSON: {e}",
                request_url=request_url,
                http_status=r.status_code,
                raw_response_text=raw_text,
            ) from e

        try:
            content_text = _extract_assistant_content_text(
                data, request_url=request_url, raw_response_text=raw_text
            )
        except ChatVlmFailure:
            raise
        except (KeyError, IndexError, TypeError) as e:
            if store is not None:
                store.append_event(
                    {
                        **log_context,
                        "event_type": "api_failure",
                        "error_type": "unexpected_response_shape",
                        "error": str(e),
                        "request_meta": {"url": request_url, "model": model, "timeout_sec": self.http_timeout},
                        "response_meta": {"http_status": r.status_code},
                        "raw_response_text": raw_text,
                    }
                )
            raise ChatVlmFailure(
                f"vllm response shape error: {e}",
                request_url=request_url,
                http_status=r.status_code,
                raw_response_text=raw_text,
            ) from e

        return ChatVlmResult(
            status_code=r.status_code,
            raw_response_text=raw_text,
            parsed_json=data,
            content_text=content_text,
        )

    def _call_edit_server(
        self,
        backend: str,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        store: Store | None = None,
        log_context: dict[str, Any] | None = None,
    ) -> Image.Image:
        pool = self.backend_pools[backend]
        picked = pool.acquire()
        req_url = f"{picked}{pool.edit_path}"
        try:
            neg_eff = negative_prompt
            if neg_eff is None:
                neg_eff = self.default_edit_params.get("negative_prompt", " ")
            if not isinstance(neg_eff, str):
                neg_eff = str(neg_eff)
            if not neg_eff.strip():
                neg_eff = " "
            payload = {
                "image_b64": _to_data_url_png(image),
                "prompt": prompt,
                "seed": self.default_edit_params.get("seed", 0),
                "num_inference_steps": self.default_edit_params.get("num_inference_steps", 40),
                "cfg_scale": self.default_edit_params.get("cfg_scale", 4.0),
                "negative_prompt": neg_eff,
            }
            try:
                r = requests.post(
                    req_url,
                    json=payload,
                    timeout=self.http_timeout,
                )
            except requests.RequestException as e:
                if store is not None and log_context is not None:
                    store.append_event(
                        {
                            **log_context,
                            "event_type": "api_failure",
                            "error_type": "request_exception",
                            "error": str(e),
                            "request_meta": {"url": req_url, "timeout_sec": self.http_timeout},
                        }
                    )
                raise EditServerError(
                    f"edit request failed: {e}",
                    request_url=req_url,
                    http_status=None,
                    raw_response_text="",
                ) from e

            raw_body = r.text
            if r.status_code != 200:
                if store is not None and log_context is not None:
                    store.append_event(
                        {
                            **log_context,
                            "event_type": "api_failure",
                            "error_type": "http_error",
                            "error": f"edit server failed ({r.status_code})",
                            "request_meta": {"url": req_url, "timeout_sec": self.http_timeout},
                            "response_meta": {"http_status": r.status_code},
                            "raw_response_text": raw_body,
                        }
                    )
                raise EditServerError(
                    f"edit server failed backend={backend} url={picked}: {raw_body}",
                    request_url=req_url,
                    http_status=r.status_code,
                    raw_response_text=raw_body,
                )

            try:
                data = r.json()
            except json.JSONDecodeError as e:
                if store is not None and log_context is not None:
                    store.append_event(
                        {
                            **log_context,
                            "event_type": "api_failure",
                            "error_type": "invalid_json_response",
                            "error": str(e),
                            "request_meta": {"url": req_url, "timeout_sec": self.http_timeout},
                            "response_meta": {"http_status": r.status_code},
                            "raw_response_text": raw_body,
                        }
                    )
                raise EditServerError(
                    f"edit server returned non-JSON: {e}",
                    request_url=req_url,
                    http_status=r.status_code,
                    raw_response_text=raw_body,
                ) from e

            if "image_b64" not in data:
                if store is not None and log_context is not None:
                    snippet = json.dumps(data, ensure_ascii=False)[:4096]
                    store.append_event(
                        {
                            **log_context,
                            "event_type": "api_failure",
                            "error_type": "missing_image_b64",
                            "error": "response JSON missing image_b64",
                            "request_meta": {"url": req_url, "timeout_sec": self.http_timeout},
                            "response_meta": {"http_status": r.status_code},
                            "raw_response_text": raw_body if raw_body else snippet,
                        }
                    )
                raise EditServerError(
                    "edit server response missing image_b64",
                    request_url=req_url,
                    http_status=r.status_code,
                    raw_response_text=raw_body,
                )

            raw = _decode_data_url_or_b64(data["image_b64"])
            from io import BytesIO

            return Image.open(BytesIO(raw)).convert("RGB")
        finally:
            pool.release(picked)

    def _task_suffix(self, task_type: str) -> str:
        if task_type == "addition":
            return "edit"
        if task_type == "removal":
            return "remove"
        raise ValueError(f"unsupported task_type: {task_type}")

    def _build_refiner_instruction(self, task: InputPair, last_attempt: dict) -> str:
        last_attempt_text = build_last_attempt_text(last_attempt)
        if task.task_type == "addition":
            return add_update_prompt(task.disease, task.dataset_type, last_attempt_text)
        if task.task_type == "removal":
            return remove_update_prompt(task.disease, last_attempt_text)
        raise ValueError(f"unsupported task_type: {task.task_type}")

    def _build_verify_instruction(self, task: InputPair, round_num: int) -> str:
        if task.task_type == "addition":
            base_instruction = add_verify_prompt(task.disease, task.dataset_type)
        elif task.task_type == "removal":
            base_instruction = remove_verify_prompt(task.disease, task.dataset_type)
        else:
            raise ValueError(f"unsupported task_type: {task.task_type}")
        force_unqualified = round_num <= self.refine_budget
        gating_instruction = (
            "MANDATORY GATING RULE (ACTIVE):\n"
            "- current_round is before refine_budget, so this round CANNOT pass.\n"
            '- You MUST set "qualified" to false and provide a concrete reason explaining why the current image does not meet the standard.\n'
       
        ) if force_unqualified else (
            ""
        )
        return (
            f"{base_instruction}\n\n"
            "Round control for this sample:\n"
            f"- max_rounds: {self.max_rounds}\n"
            f"- refine_budget: {self.refine_budget}\n"
            f"- current_round: {round_num}\n"
            f"- force_unqualified_now: {'true' if force_unqualified else 'false'}\n"
            f"\n{gating_instruction}"
        )

    def _process_one(self, backend: str, task: InputPair) -> dict[str, Any]:
        store = self.stores[(backend, task.task_type)]
        task_key = f"{task.stem}-{task.disease}"
        suffix = self._task_suffix(task.task_type)
        terminal = store.progress.get(task_key)
        if terminal in ("success", "failed"):
            result = {"status": "skipped", "task": task_key, "backend": backend}
            if self.progress is not None:
                self.progress.task_done(
                    backend=backend, task_key=task_key, status="skipped"
                )
            return result

        if self.progress is not None:
            self.progress.task_start(
                backend=backend,
                task_key=task_key,
                task_type=task.task_type,
                disease=task.disease,
                image_path=task.image_path,
            )

        try:
            return self._process_one_body(backend, task, store, task_key, suffix)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            tb_ev, tb_ev_tr = _truncate_str_bytes(tb, store.raw_log_max_bytes)
            self._record_api_failure(
                store,
                task_key=task_key,
                backend=backend,
                task=task,
                round_num=0,
                step="pipeline",
                error_type=type(e).__name__,
                error=err_msg,
            )
            ev_crash: dict[str, Any] = {
                "event_type": "pipeline_crash",
                "task": task_key,
                "backend": backend,
                "task_type": task.task_type,
                "error": err_msg,
                "traceback": tb_ev,
            }
            if tb_ev_tr:
                ev_crash["traceback_bytes_truncated"] = True
            store.append_event(ev_crash)
            conversation = {
                "image": task.image_path.name,
                "disease": task.disease,
                "status": "api_failed",
                "task_type": task.task_type,
                "backend": backend,
                "error": err_msg,
                "rounds": [],
            }
            store.save_trace(task_key, conversation)
            with store.lock:
                store.progress[task_key] = "api_failed"
            store.save_all()
            result = {"status": "api_failed", "task": task_key, "backend": backend, "error": err_msg}
            if self.progress is not None:
                self.progress.task_done(
                    backend=backend, task_key=task_key, status="api_failed", error=err_msg
                )
            return result

    def _process_one_body(
        self, backend: str, task: InputPair, store: Store, task_key: str, suffix: str
    ) -> dict[str, Any]:
        conversation = {
            "image": task.image_path.name,
            "disease": task.disease,
            "status": "failed",
            "task_type": task.task_type,
            "backend": backend,
            "rounds": [],
        }
        store.save_trace(task_key, conversation)

        current_prompt = task.initial_prompt
        _cfg_neg = self.default_edit_params.get("negative_prompt", " ")
        current_negative = _cfg_neg if isinstance(_cfg_neg, str) else str(_cfg_neg)
        prev_edited_image_bytes: bytes | None = None

        original_image = Image.open(task.image_path).convert("RGB")
        self._prog(
            backend=backend,
            task_key=task_key,
            round_num=0,
            step="load_input",
            message="loaded source image",
            preview_image=original_image,
            preview_tag=f"{backend}__{_safe_name_fragment(task_key)}__input",
        )

        for round_num in range(1, self.max_rounds + 1):
            self._prog(
                backend=backend,
                task_key=task_key,
                round_num=round_num,
                step="round_start",
                attempt=0,
                max_attempts=self.max_api_retries,
                message=f"round {round_num}/{self.max_rounds}",
            )
            round_data: dict[str, Any] = {"round": round_num}
            if round_num == 1:
                round_data["generate_prompt"] = {
                    "thinking_summary": "",
                    "prompt": current_prompt,
                    "negative_prompt": current_negative,
                }
            else:
                prev_round = conversation["rounds"][-1]
                if "generate_prompt" not in prev_round or "verification" not in prev_round:
                    raise RuntimeError(
                        f"missing generate_prompt/verification in previous round for {task_key}"
                    )
                gp = prev_round["generate_prompt"]
                last_attempt: dict[str, Any] = {
                    "prompt": gp["prompt"],
                    "verification": prev_round["verification"],
                }
                npv = gp.get("negative_prompt")
                if isinstance(npv, str) and npv.strip():
                    last_attempt["negative_prompt"] = npv.strip()
                instruction = self._build_refiner_instruction(task, last_attempt)
                base_log = {
                    "task": task_key,
                    "backend": backend,
                    "task_type": task.task_type,
                    "round": round_num,
                    "step": "update_prompt",
                }
                current_prompt_str: str | None = None
                current_negative_str: str | None = None
                last_err: str | None = None
                last_exc: BaseException | None = None
                if prev_edited_image_bytes is None:
                    raise RuntimeError("missing previous edited image bytes for refiner input")
                for attempt in range(self.max_api_retries):
                    att = attempt + 1
                    self._prog(
                        backend=backend,
                        task_key=task_key,
                        round_num=round_num,
                        step="update_prompt",
                        attempt=att,
                        max_attempts=self.max_api_retries,
                        message=f"refiner attempt {att}/{self.max_api_retries}",
                    )
                    try:
                        res = self._chat_vlm(
                            store=store,
                            log_context={**base_log, "attempt": att},
                            base_url=self.refiner_url,
                            model=self.refiner_model,
                            api_key=self.refiner_key,
                            instruction=instruction,
                            image_bytes=prev_edited_image_bytes,
                            temperature=0.9,
                        )
                        self._log_llm_output_event(
                            store,
                            task_key=task_key,
                            backend=backend,
                            task=task,
                            round_num=round_num,
                            step="update_prompt",
                            model=self.refiner_model,
                            base_url=self.refiner_url,
                            raw_llm_content=res.content_text,
                            raw_http_response_text=res.raw_response_text,
                            http_status=res.status_code,
                        )
                        raw_llm = res.content_text
                        if not raw_llm.strip():
                            if attempt < self.max_api_retries - 1:
                                time.sleep(2**attempt)
                                continue
                            trim, tr = _truncate_str_bytes(res.content_text, store.raw_log_max_bytes)
                            round_data["generate_prompt"] = {
                                "thinking_summary": "",
                                "prompt": "",
                                "negative_prompt": "",
                                "raw_llm_output": trim,
                                "raw_llm_output_bytes_truncated": tr,
                                "error": "refiner returned empty prompt after strip",
                            }
                            conversation["rounds"].append(round_data)
                            self._record_api_failure(
                                store,
                                task_key=task_key,
                                backend=backend,
                                task=task,
                                round_num=round_num,
                                step="update_prompt",
                                error_type="empty_refiner_output",
                                error="refiner returned empty prompt after strip",
                                request_url=f"{self.refiner_url}/chat/completions",
                                model=self.refiner_model,
                                timeout_sec=self.http_timeout,
                                http_status=res.status_code,
                                raw_llm_output=res.content_text,
                            )
                            with store.lock:
                                store.progress[task_key] = "api_failed"
                            store.save_trace(task_key, conversation)
                            store.save_all()
                            result = {
                                "status": "api_failed",
                                "task": task_key,
                                "backend": backend,
                                "error": "refiner returned empty prompt after strip",
                            }
                            if self.progress is not None:
                                self.progress.task_done(
                                    backend=backend,
                                    task_key=task_key,
                                    status="api_failed",
                                    error=result["error"],
                                )
                            return result
                        try:
                            p_parsed, n_parsed = _parse_refiner_dual_prompt(raw_llm)
                        except ValueError as parse_err:
                            parse_msg = str(parse_err)
                            if attempt < self.max_api_retries - 1:
                                time.sleep(2**attempt)
                                continue
                            trim, tr = _truncate_str_bytes(raw_llm, store.raw_log_max_bytes)
                            round_data["generate_prompt"] = {
                                "thinking_summary": "",
                                "prompt": "",
                                "negative_prompt": "",
                                "raw_llm_output": trim,
                                "raw_llm_output_bytes_truncated": tr,
                                "error": f"refiner JSON parse failed: {parse_msg}",
                            }
                            conversation["rounds"].append(round_data)
                            self._record_api_failure(
                                store,
                                task_key=task_key,
                                backend=backend,
                                task=task,
                                round_num=round_num,
                                step="update_prompt",
                                error_type="refiner_json_parse_error",
                                error=parse_msg,
                                request_url=f"{self.refiner_url}/chat/completions",
                                model=self.refiner_model,
                                timeout_sec=self.http_timeout,
                                http_status=res.status_code,
                                raw_llm_output=raw_llm,
                            )
                            with store.lock:
                                store.progress[task_key] = "api_failed"
                            store.save_trace(task_key, conversation)
                            store.save_all()
                            result = {
                                "status": "api_failed",
                                "task": task_key,
                                "backend": backend,
                                "error": parse_msg,
                            }
                            if self.progress is not None:
                                self.progress.task_done(
                                    backend=backend,
                                    task_key=task_key,
                                    status="api_failed",
                                    error=parse_msg,
                                )
                            return result
                        current_prompt_str = p_parsed
                        current_negative_str = n_parsed
                        last_err = None
                        last_exc = None
                        break
                    except Exception as e:
                        last_err = str(e)
                        last_exc = e
                        if attempt < self.max_api_retries - 1:
                            time.sleep(2**attempt)
                if last_err:
                    req_url = f"{self.refiner_url}/chat/completions"
                    http_status = None
                    raw_rt = None
                    if isinstance(last_exc, ChatVlmFailure):
                        http_status = last_exc.http_status
                        raw_rt = last_exc.raw_response_text
                    self._record_api_failure(
                        store,
                        task_key=task_key,
                        backend=backend,
                        task=task,
                        round_num=round_num,
                        step="update_prompt",
                        error_type=type(last_exc).__name__ if last_exc else "Error",
                        error=last_err,
                        request_url=req_url,
                        model=self.refiner_model,
                        timeout_sec=self.http_timeout,
                        http_status=http_status,
                        raw_response_text=raw_rt,
                    )
                    with store.lock:
                        store.progress[task_key] = "api_failed"
                    store.save_trace(task_key, conversation)
                    store.save_all()
                    result = {"status": "api_failed", "task": task_key, "backend": backend, "error": last_err}
                    if self.progress is not None:
                        self.progress.task_done(
                            backend=backend, task_key=task_key, status="api_failed", error=last_err
                        )
                    return result
                if current_prompt_str is None or current_negative_str is None:
                    raise RuntimeError("refiner loop exited without parsed prompt pair")
                round_data["generate_prompt"] = {
                    "thinking_summary": "",
                    "prompt": current_prompt_str,
                    "negative_prompt": current_negative_str,
                }
                current_prompt = current_prompt_str
                current_negative = current_negative_str

            edit_log = {
                "task": task_key,
                "backend": backend,
                "task_type": task.task_type,
                "round": round_num,
                "step": "edit_image",
            }
            edited_image: Image.Image | None = None
            last_err = None
            last_exc: BaseException | None = None
            for attempt in range(self.max_api_retries):
                att = attempt + 1
                self._prog(
                    backend=backend,
                    task_key=task_key,
                    round_num=round_num,
                    step="edit_image",
                    attempt=att,
                    max_attempts=self.max_api_retries,
                    message=f"edit attempt {att}/{self.max_api_retries}",
                )
                try:
                    edited_image = self._call_edit_server(
                        backend,
                        original_image,
                        current_prompt,
                        negative_prompt=current_negative,
                        store=store,
                        log_context={**edit_log, "attempt": att},
                    )
                    last_err = None
                    last_exc = None
                    break
                except Exception as e:
                    last_err = str(e)
                    last_exc = e
                    if attempt < self.max_api_retries - 1:
                        time.sleep(2**attempt)
            if last_err or edited_image is None:
                round_data["edit_result"] = {"success": False, "error": last_err}
                conversation["rounds"].append(round_data)
                req_url: str | None = None
                http_status = None
                raw_rt = None
                if isinstance(last_exc, EditServerError):
                    req_url = last_exc.request_url
                    http_status = last_exc.http_status
                    raw_rt = last_exc.raw_response_text or None
                self._record_api_failure(
                    store,
                    task_key=task_key,
                    backend=backend,
                    task=task,
                    round_num=round_num,
                    step="edit_image",
                    error_type=type(last_exc).__name__ if last_exc else "Error",
                    error=last_err or "edit failed",
                    request_url=req_url,
                    timeout_sec=self.http_timeout,
                    http_status=http_status,
                    raw_response_text=raw_rt,
                )
                with store.lock:
                    store.progress[task_key] = "api_failed"
                store.save_trace(task_key, conversation)
                store.save_all()
                result = {"status": "api_failed", "task": task_key, "backend": backend, "error": last_err}
                if self.progress is not None:
                    self.progress.task_done(
                        backend=backend, task_key=task_key, status="api_failed", error=last_err
                    )
                return result
            round_data["edit_result"] = {"success": True}
            self._prog(
                backend=backend,
                task_key=task_key,
                round_num=round_num,
                step="edit_done",
                attempt=0,
                message="edit succeeded",
                preview_image=edited_image,
                preview_tag=f"{backend}__{_safe_name_fragment(task_key)}__r{round_num}_edit",
            )
            verify_instruction = self._build_verify_instruction(task, round_num)

            byte_arr = BytesIO()
            edited_image.save(byte_arr, format="JPEG")
            edited_image_bytes = byte_arr.getvalue()
            prev_edited_image_bytes = edited_image_bytes

            verify_log = {
                "task": task_key,
                "backend": backend,
                "task_type": task.task_type,
                "round": round_num,
                "step": "verify_image",
            }
            verification: dict[str, Any] | None = None
            last_err = None
            last_exc: BaseException | None = None
            last_raw: str | None = None
            last_block: str | None = None
            last_parse: str | None = None
            for attempt in range(self.max_api_retries):
                att = attempt + 1
                self._prog(
                    backend=backend,
                    task_key=task_key,
                    round_num=round_num,
                    step="verify_image",
                    attempt=att,
                    max_attempts=self.max_api_retries,
                    message=f"verify attempt {att}/{self.max_api_retries}",
                    preview_image=edited_image,
                )
                res: ChatVlmResult | None = None
                try:
                    res = self._chat_vlm(
                        store=store,
                        log_context={**verify_log, "attempt": att},
                        base_url=self.verifier_url,
                        model=self.verifier_model,
                        api_key=self.verifier_key,
                        instruction=verify_instruction,
                        image_bytes=edited_image_bytes,
                    )
                    self._log_llm_output_event(
                        store,
                        task_key=task_key,
                        backend=backend,
                        task=task,
                        round_num=round_num,
                        step="verify_image",
                        model=self.verifier_model,
                        base_url=self.verifier_url,
                        raw_llm_content=res.content_text,
                        raw_http_response_text=res.raw_response_text,
                        http_status=res.status_code,
                    )
                    block = _extract_json_block(res.content_text)
                    verification = json.loads(block)
                    last_err = None
                    last_exc = None
                    break
                except json.JSONDecodeError as e:
                    last_parse = str(e)
                    last_err = str(e)
                    last_exc = e
                    if res is not None:
                        last_raw = res.content_text
                        last_block = block
                    if attempt < self.max_api_retries - 1:
                        time.sleep(2**attempt)
                except ChatVlmFailure as e:
                    last_err = str(e)
                    last_exc = e
                    if attempt < self.max_api_retries - 1:
                        time.sleep(2**attempt)
                except Exception as e:
                    last_err = str(e)
                    last_exc = e
                    if attempt < self.max_api_retries - 1:
                        time.sleep(2**attempt)
            if last_err or verification is None:
                v_err: dict[str, Any] = {"error": last_err or "verify failed"}
                if last_raw is not None:
                    t_raw, tr_raw = _truncate_str_bytes(last_raw, store.raw_log_max_bytes)
                    v_err["raw_llm_output"] = t_raw
                    v_err["raw_llm_output_bytes_truncated"] = tr_raw
                if last_block is not None:
                    t_b, tr_b = _truncate_str_bytes(last_block, store.raw_log_max_bytes)
                    v_err["json_block"] = t_b
                    v_err["json_block_bytes_truncated"] = tr_b
                if last_parse is not None:
                    v_err["parse_error"] = last_parse
                if isinstance(last_exc, ChatVlmFailure):
                    v_err["http_status"] = last_exc.http_status
                    if last_exc.raw_response_text:
                        t_rr, tr_rr = _truncate_str_bytes(
                            last_exc.raw_response_text, store.raw_log_max_bytes
                        )
                        v_err["raw_response_text"] = t_rr
                        v_err["raw_response_text_bytes_truncated"] = tr_rr
                round_data["verification"] = v_err
                conversation["rounds"].append(round_data)
                req_url = f"{self.verifier_url}/chat/completions"
                http_status = None
                raw_rt = None
                if isinstance(last_exc, ChatVlmFailure):
                    http_status = last_exc.http_status
                    raw_rt = last_exc.raw_response_text
                self._record_api_failure(
                    store,
                    task_key=task_key,
                    backend=backend,
                    task=task,
                    round_num=round_num,
                    step="verify_image",
                    error_type=type(last_exc).__name__ if last_exc else "Error",
                    error=last_err or "verify failed",
                    request_url=req_url,
                    model=self.verifier_model,
                    timeout_sec=self.http_timeout,
                    http_status=http_status,
                    raw_response_text=raw_rt,
                    raw_llm_output=last_raw,
                    json_block=last_block,
                    parse_error=last_parse,
                )
                with store.lock:
                    store.progress[task_key] = "api_failed"
                store.save_trace(task_key, conversation)
                store.save_all()
                result = {"status": "api_failed", "task": task_key, "backend": backend, "error": last_err}
                if self.progress is not None:
                    self.progress.task_done(
                        backend=backend, task_key=task_key, status="api_failed", error=last_err
                    )
                return result

            round_data["verification"] = verification
            qualified_now = bool(verification.get("qualified", False))
            self._prog(
                backend=backend,
                task_key=task_key,
                round_num=round_num,
                step="verify_done",
                attempt=0,
                qualified=qualified_now,
                message=f"qualified={qualified_now}",
                preview_image=edited_image,
            )
            if round_num < self.refine_budget:
                forced_prefix = (
                    f"round {round_num} is before refine_budget {self.refine_budget}; "
                    "must keep refining and reject this round."
                )
                existing_reason = verification.get("reason")
                if isinstance(existing_reason, str) and existing_reason.strip():
                    verification["reason"] = f"{forced_prefix} Original verifier reason: {existing_reason.strip()}"
                else:
                    verification["reason"] = forced_prefix
                verification["qualified"] = False
            round_data["verification"]["thinking_summary"] = ""
            conversation["rounds"].append(round_data)
            store.save_trace(task_key, conversation)

            if verification.get("qualified", False):
                output_subdir = store.output_dir / task.disease
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_path = output_subdir / f"{task.stem}-{task.disease}-{suffix}.jpeg"
                edited_image.save(output_path, "JPEG", quality=95)

                conversation["status"] = "success"
                conversation["final_prompt"] = current_prompt
                conversation["final_negative_prompt"] = current_negative
                conversation["final_image_path"] = str(output_path)
                with store.lock:
                    store.all_conversations[task_key] = conversation
                    store.final_prompts[task_key] = {
                        "image": task.image_path.name,
                        "disease": task.disease,
                        "task_type": task.task_type,
                        "status": "success",
                        "final_prompt": current_prompt,
                        "final_negative_prompt": current_negative,
                        "rounds": round_num,
                    }
                    store.progress[task_key] = "success"
                store.save_trace(task_key, conversation)
                store.save_all()
                result = {"status": "success", "task": task_key, "backend": backend, "rounds": round_num}
                if self.progress is not None:
                    self.progress.task_done(
                        backend=backend,
                        task_key=task_key,
                        status="success",
                        rounds=round_num,
                    )
                return result

            failed_subdir = store.failed_dir / task.disease
            failed_subdir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_subdir / f"{task.stem}-{task.disease}-{suffix}-failed-{round_num}.jpeg"
            edited_image.save(failed_path, "JPEG", quality=95)

        conversation["status"] = "failed"
        conversation["final_prompt"] = current_prompt
        conversation["final_negative_prompt"] = current_negative
        conversation["final_image_path"] = str(
            store.failed_dir / task.disease / f"{task.stem}-{task.disease}-{suffix}-failed-{self.max_rounds}.jpeg"
        )
        with store.lock:
            store.all_conversations[task_key] = conversation
            store.failed_summary.append(
                {
                    "task": task_key,
                    "backend": backend,
                    "image": task.image_path.name,
                    "disease": task.disease,
                    "task_type": task.task_type,
                    "final_prompt": current_prompt,
                    "final_negative_prompt": current_negative,
                    "rounds": self.max_rounds,
                    "last_verification": conversation["rounds"][-1]["verification"],
                    "final_image_path": conversation["final_image_path"],
                    "full_conversation": conversation,
                }
            )
            store.final_prompts[task_key] = {
                "image": task.image_path.name,
                "disease": task.disease,
                "task_type": task.task_type,
                "status": "failed",
                "final_prompt": current_prompt,
                "final_negative_prompt": current_negative,
                "rounds": self.max_rounds,
            }
            store.progress[task_key] = "failed"
        store.save_trace(task_key, conversation)
        store.save_all()
        result = {"status": "failed", "task": task_key, "backend": backend, "rounds": self.max_rounds}
        if self.progress is not None:
            self.progress.task_done(
                backend=backend,
                task_key=task_key,
                status="failed",
                rounds=self.max_rounds,
            )
        return result


def parse_input_pairs(input_dir: Path) -> list[InputPair]:
    txts = sorted(input_dir.glob("*.txt"))
    if not txts:
        raise ValueError(f"no txt prompt files found in {input_dir}")

    out: list[InputPair] = []
    for txt in txts:
        stem = txt.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = input_dir / f"{stem}{ext}"
            if p.is_file():
                image_path = p
                break
        if image_path is None:
            raise ValueError(f"missing paired image for {txt.name}")

        dataset_code, disease, task_type = _parse_task_stem(stem)
        dataset_type = _dataset_name_for_prompt(dataset_code)

        initial_prompt = txt.read_text(encoding="utf-8")
        if not initial_prompt.strip():
            raise ValueError(f"empty initial prompt file: {txt}")
        out.append(
            InputPair(
                stem=stem,
                image_path=image_path,
                prompt_path=txt,
                task_type=task_type,
                disease=disease,
                dataset_type=dataset_type,
                initial_prompt=initial_prompt,
            )
        )
    return out


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("refiner", "verifier", "edit_servers", "output_root"):
        if key not in cfg:
            raise ValueError(f"missing config key: {key}")
    if not cfg["edit_servers"]:
        raise ValueError("edit_servers is empty")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify-refine scheduler for DiffSynth + vLLM")
    parser.add_argument("--config", required=True, help="json config path")
    parser.add_argument("--input-dir", required=True, help="folder with <stem>.txt + <stem>.jpg pairs")
    parser.add_argument("--max-rounds", type=int, default=None, help="override config max_rounds")
    parser.add_argument("--refine-budget", type=int, default=None, help="override config refine_budget")
    parser.add_argument("--dry-run", action="store_true", help="only validate config/input and print task plan")
    parser.add_argument(
        "--backends",
        default=None,
        help="comma-separated backends to run, e.g. qwen2511 or qwen2511,firered11",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm bar (still writes scheduler_live/status.json and preview JPEGs)",
    )
    parser.add_argument(
        "--progress-dir",
        default=None,
        help="live status/preview directory (default: <output_root>/scheduler_live)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="override config output_root (e.g. outputs-original for base-model runs)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="only re-run tasks with progress status failed; clears their progress before retry",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config).expanduser().resolve())
    if args.max_rounds is not None:
        config["max_rounds"] = args.max_rounds
    if args.refine_budget is not None:
        config["refine_budget"] = args.refine_budget
    if args.output_root is not None:
        config["output_root"] = str(resolve_repo_path(args.output_root))
    selected_backends = None
    if args.backends:
        selected_backends = [x.strip() for x in args.backends.split(",") if x.strip()]
        if not selected_backends:
            _die("--backends is empty after parsing")

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        _die(f"input dir does not exist: {input_dir}")
    pairs = parse_input_pairs(input_dir)
    if args.dry_run:
        dry_scheduler = Scheduler(config, selected_backends=selected_backends)
        if args.retry_failed:
            retry_jobs = dry_scheduler.plan_jobs(pairs, retry_failed=True, reset_failed=False)
            retry_candidates = [
                {
                    "backend": backend,
                    "task_key": f"{pair.stem}-{pair.disease}",
                    "task_type": pair.task_type,
                    "image_path": str(pair.image_path),
                }
                for backend, pair in retry_jobs
            ]
            print(
                json.dumps(
                    {
                        "mode": "dry_run_retry_failed",
                        "pairs": len(pairs),
                        "backends": list(dry_scheduler.backend_pools.keys()),
                        "retry_count": len(retry_candidates),
                        "retry_candidates": retry_candidates,
                        "output_root": str(dry_scheduler.output_root),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        preview = []
        for p in pairs[:10]:
            preview.append(
                {
                    "stem": p.stem,
                    "task_type": p.task_type,
                    "dataset_type": p.dataset_type,
                    "disease": p.disease,
                    "image_path": str(p.image_path),
                    "prompt_path": str(p.prompt_path),
                }
            )
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "pairs": len(pairs),
                    "backends": list(dry_scheduler.backend_pools.keys()),
                    "preview_first_10": preview,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_root = Path(config["output_root"]).expanduser().resolve()
    progress_dir = (
        Path(args.progress_dir).expanduser().resolve()
        if args.progress_dir
        else output_root / "scheduler_live"
    )
    scheduler = Scheduler(config, selected_backends=selected_backends, progress=None)
    jobs = scheduler.plan_jobs(pairs, retry_failed=args.retry_failed, reset_failed=True)
    if args.retry_failed and not jobs:
        print(
            json.dumps(
                {
                    "mode": "retry_failed",
                    "retry_count": 0,
                    "total_pairs": len(pairs),
                    "backends": list(scheduler.backend_pools.keys()),
                    "output_root": str(output_root),
                    "message": "no failed tasks to retry for this input-dir and backends",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    total_jobs = len(jobs)
    max_rounds_eff = int(config.get("max_rounds", 5))
    _mar = config.get("max_api_retries")
    max_retries_eff = int(_mar) if _mar is not None else int(config.get("max_retries", 3))
    run_progress = RunProgress(
        total_jobs=total_jobs,
        live_dir=progress_dir,
        max_api_retries=max_retries_eff,
        max_rounds=max_rounds_eff,
        enabled=not args.no_progress,
    )
    scheduler.progress = run_progress
    print(
        f"Live progress: {progress_dir}/status.json  preview: {progress_dir}/current_preview.jpeg",
        file=sys.stderr,
    )
    if args.retry_failed:
        print(f"Retry failed: {total_jobs} job(s)", file=sys.stderr)

    scheduler.health_check()

    stats: dict[str, dict[str, int]] = {
        backend: {"success": 0, "failed": 0, "api_failed": 0, "skipped": 0, "crashed": 0}
        for backend in scheduler.backend_pools
    }
    futures: dict[Any, tuple[str, str]] = {}
    try:
        with ThreadPoolExecutor(
            max_workers=max(1, len(scheduler.backend_pools) * scheduler.concurrent_per_backend)
        ) as executor:
            for backend, pair in jobs:
                task_key = f"{pair.stem}-{pair.disease}"
                fut = executor.submit(scheduler._process_one, backend, pair)
                futures[fut] = (backend, task_key)
            for fut in as_completed(futures):
                backend_done, task_key_done = futures[fut]
                try:
                    result = fut.result()
                except Exception:
                    sys.stderr.write(traceback.format_exc())
                    stats[backend_done]["crashed"] += 1
                    run_progress.task_done(
                        backend=backend_done,
                        task_key=task_key_done,
                        status="crashed",
                        error="uncaught exception in worker",
                    )
                    continue
                stats[result["backend"]][result["status"]] += 1
    finally:
        run_progress.close()

    summary: dict[str, Any] = {
        "total_pairs": len(pairs),
        "jobs_run": total_jobs,
        "stats": stats,
        "progress_dir": str(progress_dir),
    }
    if args.retry_failed:
        summary["mode"] = "retry_failed"
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
