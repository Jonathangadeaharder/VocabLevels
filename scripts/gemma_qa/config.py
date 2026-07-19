from __future__ import annotations

import os
import threading

from .model_strategies import (
    ACTIVE_STRATEGIES,
    ALL_STRATEGIES,
    API_BASE_EXTERNAL,
    API_BASE_INTERNAL,
    KEY_GLM_52_EXTERNAL,
    KEY_GLM_52_INTERNAL,
    KEY_GLM_52_TEE,
    KEY_QWEN_35B,
    KEY_QWEN_397B,
    STRATEGY_BY_KEY,
    WIRE_GEMMA,
    WIRE_GLM_51,
    WIRE_GLM_52,
    active_strategy_keys,
    get_strategy,
)

# Re-export gateway defaults for older imports.
API_BASE = API_BASE_INTERNAL
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Public wire / key constants (tests + CLI).
MODEL_QWEN_397B = KEY_QWEN_397B
MODEL_QWEN_35B = KEY_QWEN_35B
MODEL_GEMMA = WIRE_GEMMA
MODEL_GLM_51 = WIRE_GLM_51
MODEL_GLM_52 = WIRE_GLM_52
MODEL_GLM_52_TEE = KEY_GLM_52_TEE

# Legacy dual aliases — any active model can fill these roles via routing.
MODEL_31B = MODEL_QWEN_397B
MODEL_26B = MODEL_QWEN_35B
# Handcraft fixed adj fallback (fast pool member, not Gemma).
MODEL_ADJUDICATION = MODEL_QWEN_35B

# All keys known to the client (including bottleneck / optional for explicit use).
MODEL_IDS: tuple[str, ...] = tuple(s.key() for s in ALL_STRATEGIES)
ALL_MODEL_IDS = MODEL_IDS

# Unified job pool: every key can dual or adjudicate (no role silos).
ACTIVE_POOL: tuple[str, ...] = active_strategy_keys()
DUAL_POOL: tuple[str, ...] = ACTIVE_POOL
ADJUDICATION_POOL: tuple[str, ...] = ACTIVE_POOL

# Back-compat registry shape for resolve_model_spec callers.
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    wire_id: str
    api_base: str
    optional: bool = False


def _spec_from_strategy(strategy_cls: type) -> ModelSpec:
    return ModelSpec(
        key=strategy_cls.key(),
        wire_id=strategy_cls.wire_id(),
        api_base=strategy_cls.api_base(),
        optional=strategy_cls.optional(),
    )


MODEL_SPECS: tuple[ModelSpec, ...] = tuple(
    _spec_from_strategy(s) for s in ALL_STRATEGIES
)
MODEL_REGISTRY: dict[str, ModelSpec] = {spec.key: spec for spec in MODEL_SPECS}

INPUT_BATCH_TOKEN_CAP = 28_000
REASONING_EFFORT = os.environ.get("TNG_REASONING_EFFORT", "none")

# Optional / disabled
_disabled_lock = threading.Lock()
_disabled_model_keys: set[str] = set()
_slot_lock = threading.Lock()
_model_semaphores: dict[str, threading.Semaphore] = {}
_model_inflight: dict[str, int] = {}


def mark_model_unavailable(model_key: str) -> None:
    with _disabled_lock:
        _disabled_model_keys.add(model_key)


def is_model_available(model_key: str) -> bool:
    with _disabled_lock:
        return model_key not in _disabled_model_keys


def per_model_max_inflight() -> int:
    raw = os.environ.get("GEMMA_QA_PER_MODEL_INFLIGHT", "2")
    try:
        value = int(raw)
    except ValueError:
        value = 2
    return max(1, min(value, 8))


def _semaphore_for(model_key: str) -> threading.Semaphore:
    with _slot_lock:
        sem = _model_semaphores.get(model_key)
        if sem is None:
            sem = threading.Semaphore(per_model_max_inflight())
            _model_semaphores[model_key] = sem
            _model_inflight[model_key] = 0
        return sem


def slot_acquire_timeout_s() -> float:
    """Max seconds to wait for a free per-model slot (default 120).

    Prevents indefinite block when all slots are held by hung HTTP that
    has not yet released (wall-clock failure path still frees in finally).
    """
    raw = os.environ.get("GEMMA_QA_SLOT_ACQUIRE_TIMEOUT_S", "120")
    try:
        value = float(raw)
    except ValueError:
        value = 120.0
    return max(5.0, min(value, 900.0))


def acquire_model_slot(model_key: str, *, timeout_s: float | None = None) -> None:
    timeout = slot_acquire_timeout_s() if timeout_s is None else timeout_s
    got = _semaphore_for(model_key).acquire(timeout=timeout)
    if not got:
        raise TimeoutError(
            f"model slot acquire timed out after {timeout:.0f}s: {model_key}"
        )
    with _slot_lock:
        _model_inflight[model_key] = _model_inflight.get(model_key, 0) + 1


def release_model_slot(model_key: str) -> None:
    with _slot_lock:
        current = _model_inflight.get(model_key, 0)
        _model_inflight[model_key] = max(0, current - 1)
    _semaphore_for(model_key).release()


def model_inflight(model_key: str) -> int:
    with _slot_lock:
        return int(_model_inflight.get(model_key, 0))


def model_free_slots(model_key: str) -> int:
    return max(0, per_model_max_inflight() - model_inflight(model_key))


def default_batch_concurrency() -> int:
    """Sweet spot ~4; cap by 2× active models so free-slot selection can fill."""
    available = sum(1 for key in ACTIVE_POOL if is_model_available(key))
    if available < 1:
        available = len(ACTIVE_POOL)
    # Measured optimum was ~4 workers; do not exceed 2*active or 8.
    auto = min(8, max(4, available * per_model_max_inflight() // 2))
    raw = os.environ.get("GEMMA_QA_BATCH_CONCURRENCY")
    if raw is None or raw == "":
        return auto
    try:
        value = int(raw)
    except ValueError:
        return auto
    return max(1, min(value, 16))


def resolve_model_spec(model: str) -> ModelSpec:
    strategy = get_strategy(model)
    return _spec_from_strategy(strategy)


def get_api_key() -> str:
    for name in ("API_KEY", "TNG_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        "API_KEY is required for chat.model.tngtech.com "
        "(or set TNG_API_KEY / GEMINI_API_KEY)"
    )


def probe_optional_models(*, timeout_s: float = 15.0) -> list[str]:
    """Disable optional strategies missing from /v1/models."""
    import json
    import urllib.error
    import urllib.request

    disabled: list[str] = []
    key = get_api_key()
    bases = {s.api_base() for s in ALL_STRATEGIES if s.optional()}
    listed: dict[str, set[str]] = {}
    for base in bases:
        url = f"{base.rstrip('/')}/v1/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode())
            listed[base] = {
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            listed[base] = set()
    for strategy in ALL_STRATEGIES:
        if not strategy.optional():
            continue
        if strategy.wire_id() not in listed.get(strategy.api_base(), set()):
            mark_model_unavailable(strategy.key())
            disabled.append(strategy.key())
    return disabled


__all__ = [
    "ACTIVE_POOL",
    "ACTIVE_STRATEGIES",
    "ADJUDICATION_POOL",
    "API_BASE",
    "API_BASE_EXTERNAL",
    "API_BASE_INTERNAL",
    "CHAT_COMPLETIONS_PATH",
    "DUAL_POOL",
    "INPUT_BATCH_TOKEN_CAP",
    "KEY_GLM_52_EXTERNAL",
    "KEY_GLM_52_INTERNAL",
    "KEY_GLM_52_TEE",
    "MODEL_26B",
    "MODEL_31B",
    "MODEL_ADJUDICATION",
    "MODEL_GEMMA",
    "MODEL_GLM_51",
    "MODEL_GLM_52",
    "MODEL_IDS",
    "MODEL_QWEN_35B",
    "MODEL_QWEN_397B",
    "MODEL_REGISTRY",
    "MODEL_SPECS",
    "ModelSpec",
    "REASONING_EFFORT",
    "STRATEGY_BY_KEY",
    "acquire_model_slot",
    "default_batch_concurrency",
    "get_api_key",
    "get_strategy",
    "is_model_available",
    "mark_model_unavailable",
    "model_free_slots",
    "model_inflight",
    "per_model_max_inflight",
    "probe_optional_models",
    "release_model_slot",
    "resolve_model_spec",
    "slot_acquire_timeout_s",
]
