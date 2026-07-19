"""Structured runtime tracing for Gemma QA bottlenecks and model I/O.

Writes JSONL events to stderr and optionally ``.gemma_qa/events.jsonl``.
Enable verbose model bodies with ``GEMMA_QA_LOG_BODIES=1``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_LOCK = threading.RLock()
_CONFIGURED = False
_RUN_ID = uuid.uuid4().hex[:12]
_JSONL_PATH: Path | None = None
_LEVEL = "INFO"
_LOG_BODIES = False
_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

# Cap large fields so a single bad response cannot blow disk/console.
_MAX_TEXT = 4_000
_MAX_ROWS_SAMPLE = 12
_MAX_THOUGHT = 8_000


@dataclass(frozen=True)
class TraceConfig:
    run_id: str
    jsonl_path: Path | None
    level: str
    log_bodies: bool


def configure(
    *,
    root: Path | None = None,
    level: str | None = None,
    log_bodies: bool | None = None,
    jsonl_path: Path | None = None,
) -> TraceConfig:
    """Configure tracing once per process. Safe to call repeatedly."""
    global _CONFIGURED, _JSONL_PATH, _LEVEL, _LOG_BODIES
    with _LOCK:
        env_level = (level or os.environ.get("GEMMA_QA_LOG_LEVEL") or "INFO").upper()
        if env_level not in _LEVEL_ORDER:
            env_level = "INFO"
        bodies = log_bodies
        if bodies is None:
            bodies = os.environ.get("GEMMA_QA_LOG_BODIES", "").strip() in {
                "1",
                "true",
                "yes",
            }
        path = jsonl_path
        if path is None and root is not None:
            path = root / ".gemma_qa" / "events.jsonl"
        if path is None:
            env_path = os.environ.get("GEMMA_QA_LOG_PATH")
            if env_path:
                path = Path(env_path)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        _LEVEL = env_level
        _LOG_BODIES = bool(bodies)
        _JSONL_PATH = path
        _CONFIGURED = True
        return TraceConfig(
            run_id=_RUN_ID,
            jsonl_path=_JSONL_PATH,
            level=_LEVEL,
            log_bodies=_LOG_BODIES,
        )


def run_id() -> str:
    return _RUN_ID


def log_bodies_enabled() -> bool:
    return _LOG_BODIES


def event(
    kind: str,
    *,
    level: str = "INFO",
    **fields: Any,
) -> None:
    """Emit one structured event (JSONL + human stderr line)."""
    if not _CONFIGURED:
        configure()
    severity = level.upper()
    if _LEVEL_ORDER.get(severity, 20) < _LEVEL_ORDER.get(_LEVEL, 20):
        return
    payload: dict[str, Any] = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": _RUN_ID,
        "level": severity,
        "kind": kind,
    }
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = _jsonable(value)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    human = _human_line(payload)
    with _LOCK:
        print(human, file=sys.stderr, flush=True)
        if _JSONL_PATH is not None:
            with _JSONL_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def extract_thoughts(response_json: Mapping[str, object] | None) -> list[str]:
    """Pull model thought/reasoning parts when the API returns them."""
    if not response_json:
        return []
    thoughts: list[str] = []
    candidates = response_json.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                if part.get("thought") is True or part.get("thoughtSignature"):
                    thoughts.append(_clip(text, _MAX_THOUGHT))
    # Some payloads put thoughts at top-level or under usageMetadata notes.
    for key in ("thinking", "thoughts", "reasoning"):
        raw = response_json.get(key)
        if isinstance(raw, str) and raw.strip():
            thoughts.append(_clip(raw, _MAX_THOUGHT))
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    thoughts.append(_clip(item, _MAX_THOUGHT))
    return thoughts


def summarize_parsed(parsed: object) -> dict[str, Any]:
    """Compact summary of structured model output for logs."""
    if parsed is None:
        return {"type": "none"}
    dump: Any
    if hasattr(parsed, "model_dump"):
        dump = parsed.model_dump()
    elif isinstance(parsed, Mapping):
        dump = dict(parsed)
    else:
        return {"type": type(parsed).__name__, "repr": _clip(repr(parsed), 500)}
    rows = dump.get("rows") if isinstance(dump, dict) else None
    if not isinstance(rows, list):
        return {
            "type": type(parsed).__name__,
            "keys": sorted(dump.keys()) if isinstance(dump, dict) else [],
            "preview": _clip(json.dumps(dump, ensure_ascii=False, default=str), 800),
        }
    actions: dict[str, int] = {}
    sample: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        action = str(row.get("action") or row.get("upos") or "?")
        actions[action] = actions.get(action, 0) + 1
        if len(sample) < _MAX_ROWS_SAMPLE:
            sample.append(
                {
                    "id": row.get("id"),
                    "lemma": row.get("lemma"),
                    "english_lemma": row.get("english_lemma"),
                    "upos": row.get("upos"),
                    "action": row.get("action"),
                }
            )
    return {
        "type": type(parsed).__name__,
        "row_count": len(rows),
        "actions": actions,
        "sample": sample,
    }


def recent_events(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = lines[-limit:]
    events: list[dict[str, Any]] = []
    for line in selected:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _human_line(payload: Mapping[str, Any]) -> str:
    kind = payload.get("kind", "?")
    level = payload.get("level", "INFO")
    parts = [f"[{level}]", str(kind)]
    for key in (
        "phase",
        "model",
        "batch_id",
        "lang",
        "level_name",
        "status",
        "wait_s",
        "duration_ms",
        "http_status",
        "prompt_tokens",
        "candidate_tokens",
        "error",
        "reason",
        "checkpoint",
        "attempt",
        "task",
        "output",
        "gap",
        "accepted",
        "target",
    ):
        if key in payload and payload[key] not in (None, ""):
            parts.append(f"{key}={payload[key]}")
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary.get("row_count") is not None:
        parts.append(f"rows={summary['row_count']}")
        actions = summary.get("actions")
        if actions:
            parts.append(f"actions={actions}")
    thoughts = payload.get("thoughts")
    if isinstance(thoughts, list) and thoughts:
        parts.append(f"thoughts={len(thoughts)}")
    return " ".join(parts)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > _MAX_TEXT and not _LOG_BODIES:
            return _clip(value, _MAX_TEXT)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    return _clip(repr(value), 500)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + f"...<clipped n={len(text)}>"
