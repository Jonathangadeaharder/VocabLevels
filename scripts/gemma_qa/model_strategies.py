"""Strategy pattern: every active model can run any CEFR job.

Jobs (dual review A/B, adjudication, handcraft gen/review/adj) share one
interface. Strategies only encode endpoint/wire-id/thinking subtleties.
Bottleneck models (Gemma-4, GLM-5.1) are registered but not in ACTIVE_POOL.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Final

# ---------------------------------------------------------------------------
# Gateways
# ---------------------------------------------------------------------------
API_BASE_INTERNAL = os.environ.get(
    "TNG_API_BASE",
    "https://chat.model.tngtech.com",
).rstrip("/")
API_BASE_EXTERNAL = os.environ.get(
    "TNG_API_BASE_EXTERNAL",
    "https://external.model.tngtech.com",
).rstrip("/")

# Wire ids
WIRE_QWEN_397B = "Qwen/Qwen3.5-397B-A17B-FP8"
WIRE_QWEN_35B = "Qwen/Qwen3.6-35B-A3B-FP8"
WIRE_GEMMA = "google/gemma-4-31B-it"
WIRE_GLM_51 = "zai-org/GLM-5.1-FP8"
WIRE_GLM_52 = "zai-org/GLM-5.2"
WIRE_GLM_52_TEE = "zai-org/GLM-5.2-TEE"

# Unique keys when wire id is shared across gateways
KEY_QWEN_397B = WIRE_QWEN_397B
KEY_QWEN_35B = WIRE_QWEN_35B
KEY_GLM_52_INTERNAL = "zai-org/GLM-5.2"
KEY_GLM_52_EXTERNAL = "external/zai-org/GLM-5.2"
KEY_GLM_52_TEE = "external/zai-org/GLM-5.2-TEE"
KEY_GEMMA = WIRE_GEMMA
KEY_GLM_51 = WIRE_GLM_51

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Job roles — every active strategy can do all of these.
ROLE_DUAL = "dual"
ROLE_ADJUDICATION = "adjudication"
ROLE_GENERATION = "generation"
ALL_ROLES: Final[frozenset[str]] = frozenset(
    {ROLE_DUAL, ROLE_ADJUDICATION, ROLE_GENERATION}
)


@dataclass(frozen=True)
class ModelEndpoint:
    key: str
    wire_id: str
    api_base: str
    optional: bool = False


class ModelStrategy(ABC):
    """Uniform surface for any structured CEFR/handcraft call."""

    endpoint: ClassVar[ModelEndpoint]
    # Subtleties
    prefers_reasoning_effort_none: ClassVar[bool] = True
    strips_think_tags: ClassVar[bool] = True

    @classmethod
    def key(cls) -> str:
        return cls.endpoint.key

    @classmethod
    def wire_id(cls) -> str:
        return cls.endpoint.wire_id

    @classmethod
    def api_base(cls) -> str:
        return cls.endpoint.api_base

    @classmethod
    def optional(cls) -> bool:
        return cls.endpoint.optional

    @classmethod
    def supports_role(cls, role: str) -> bool:
        """Active strategies support every job; bottlenecks override to False."""
        return role in ALL_ROLES

    @classmethod
    def request_extras(cls) -> dict[str, object]:
        """Model-specific chat.completions fields (thinking, etc.)."""
        extras: dict[str, object] = {}
        if cls.prefers_reasoning_effort_none:
            extras["reasoning_effort"] = os.environ.get(
                "TNG_REASONING_EFFORT", "none"
            )
        return extras

    @classmethod
    def strip_output(cls, text: str) -> str:
        """Normalize model text before JSON parse. Never treat think as answer."""
        if not cls.strips_think_tags:
            return text.strip()
        stripped = text.strip()
        stripped = _THINK_CLOSED.sub("", stripped).strip()
        lower = stripped.lower()
        if "<think>" in lower:
            stripped = stripped[: lower.find("<think>")].strip()
        stripped = re.sub(r"</think>", "", stripped, flags=re.IGNORECASE).strip()
        return stripped

    @classmethod
    def family(cls) -> str:
        return "generic"


class QwenStrategy(ModelStrategy):
    family_name: ClassVar[str] = "qwen"
    prefers_reasoning_effort_none = True
    strips_think_tags = True

    @classmethod
    def family(cls) -> str:
        return cls.family_name


class Qwen397Strategy(QwenStrategy):
    endpoint = ModelEndpoint(KEY_QWEN_397B, WIRE_QWEN_397B, API_BASE_INTERNAL)


class Qwen35Strategy(QwenStrategy):
    endpoint = ModelEndpoint(KEY_QWEN_35B, WIRE_QWEN_35B, API_BASE_INTERNAL)


class GlmStrategy(ModelStrategy):
    """GLM on TNG: thinking must be forced off; tags still stripped."""

    family_name: ClassVar[str] = "glm"
    prefers_reasoning_effort_none = True
    strips_think_tags = True

    @classmethod
    def family(cls) -> str:
        return cls.family_name


class Glm52InternalStrategy(GlmStrategy):
    endpoint = ModelEndpoint(KEY_GLM_52_INTERNAL, WIRE_GLM_52, API_BASE_INTERNAL)


class Glm52ExternalStrategy(GlmStrategy):
    # External pool flaked with 422 model_unavailable; optional so probe/422
    # can drop it without failing whole dual batches.
    endpoint = ModelEndpoint(
        KEY_GLM_52_EXTERNAL, WIRE_GLM_52, API_BASE_EXTERNAL, optional=True
    )


class Glm52TeeStrategy(GlmStrategy):
    endpoint = ModelEndpoint(
        KEY_GLM_52_TEE, WIRE_GLM_52_TEE, API_BASE_EXTERNAL, optional=True
    )


class GemmaStrategy(ModelStrategy):
    """Registered for completeness; not in ACTIVE_POOL (latency bottleneck)."""

    endpoint = ModelEndpoint(KEY_GEMMA, WIRE_GEMMA, API_BASE_INTERNAL)
    prefers_reasoning_effort_none = True
    strips_think_tags = True

    @classmethod
    def supports_role(cls, role: str) -> bool:
        return False  # stripped from production rotation

    @classmethod
    def family(cls) -> str:
        return "gemma"


class Glm51Strategy(GlmStrategy):
    """Registered; not in ACTIVE_POOL (slow under concurrent load)."""

    endpoint = ModelEndpoint(KEY_GLM_51, WIRE_GLM_51, API_BASE_INTERNAL)

    @classmethod
    def supports_role(cls, role: str) -> bool:
        return False


# All known strategies (including bottlenecks / optional).
ALL_STRATEGIES: tuple[type[ModelStrategy], ...] = (
    Qwen397Strategy,
    Qwen35Strategy,
    Glm52InternalStrategy,
    Glm52ExternalStrategy,
    Glm52TeeStrategy,
    GemmaStrategy,
    Glm51Strategy,
)

# Production rotation: every entry can do dual OR adjudication (any job).
ACTIVE_STRATEGIES: tuple[type[ModelStrategy], ...] = (
    Qwen397Strategy,
    Qwen35Strategy,
    Glm52InternalStrategy,
    Glm52ExternalStrategy,
    Glm52TeeStrategy,  # optional; probe may disable
)

STRATEGY_BY_KEY: dict[str, type[ModelStrategy]] = {
    s.key(): s for s in ALL_STRATEGIES
}


def get_strategy(model_key: str) -> type[ModelStrategy]:
    if model_key in STRATEGY_BY_KEY:
        return STRATEGY_BY_KEY[model_key]
    for strategy in ALL_STRATEGIES:
        if strategy.wire_id() == model_key and "external" not in strategy.key():
            return strategy
    raise ValueError(f"unsupported model: {model_key}")


def active_strategy_keys() -> tuple[str, ...]:
    return tuple(s.key() for s in ACTIVE_STRATEGIES)
