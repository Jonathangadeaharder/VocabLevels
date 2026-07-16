from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast

MODEL_31B = "gemma-4-31b-it"
MODEL_26B = "gemma-4-26b-a4b-it"
MODEL_ANTIGRAVITY = "antigravity-preview-05-2026"
MODEL_IDS = (MODEL_31B, MODEL_26B)
ALL_MODEL_IDS = (*MODEL_IDS, MODEL_ANTIGRAVITY)
API_BASE = "https://generativelanguage.googleapis.com"
INTERACTIONS_PATH = "/v1beta/interactions"
# Interactions API revision header required by Antigravity REST callers.
ANTIGRAVITY_API_REVISION = "2026-05-20"

HARD_REQUESTS_PER_MINUTE = 30
# Gemini Gemma TPM is input tokens / minute / model (not input+output).
HARD_TOKENS_PER_MINUTE = 16_000
HARD_REQUESTS_PER_DAY = 14_400
REQUESTS_PER_MINUTE = 30
TOKENS_PER_MINUTE = 16_000
REQUESTS_PER_DAY = 14_400

# Antigravity Agent UI quotas (separate bucket from Gemma).
# Interpreting shown 0/60, 0/100K, 0/100 as RPM / TPM / RPD.
ANTIGRAVITY_HARD_REQUESTS_PER_MINUTE = 60
ANTIGRAVITY_HARD_TOKENS_PER_MINUTE = 100_000
ANTIGRAVITY_HARD_REQUESTS_PER_DAY = 100
ANTIGRAVITY_REQUESTS_PER_MINUTE = 60
ANTIGRAVITY_TOKENS_PER_MINUTE = 100_000
ANTIGRAVITY_REQUESTS_PER_DAY = 100

# Leave modest headroom under 16k Gemma input TPM for schema/overhead per request.
INPUT_BATCH_TOKEN_CAP = 14_500
# Antigravity has higher TPM; keep CEFR prompts tight anyway (100 RPD budget).
ANTIGRAVITY_INPUT_TOKEN_CAP = 8_000
THINKING_CONFIG = {"thinkingLevel": "minimal"}
SchemaProperty = Literal["responseJsonSchema", "responseSchema"]
RpdPolicy = Literal["wait", "fail"]


@dataclass(frozen=True)
class ModelCeilings:
    requests_per_minute: int = REQUESTS_PER_MINUTE
    tokens_per_minute: int = TOKENS_PER_MINUTE
    requests_per_day: int = REQUESTS_PER_DAY
    hard_requests_per_minute: int = HARD_REQUESTS_PER_MINUTE
    hard_tokens_per_minute: int = HARD_TOKENS_PER_MINUTE
    hard_requests_per_day: int = HARD_REQUESTS_PER_DAY
    # "fail" = raise when daily ceiling would be exceeded (Antigravity).
    # "wait" = sleep until the oldest day window entry ages out (Gemma).
    rpd_policy: RpdPolicy = "wait"

    def __post_init__(self) -> None:
        values = (
            self.requests_per_minute,
            self.tokens_per_minute,
            self.requests_per_day,
        )
        hard_caps = (
            self.hard_requests_per_minute,
            self.hard_tokens_per_minute,
            self.hard_requests_per_day,
        )
        if min(values) <= 0:
            raise ValueError("model ceilings must be positive")
        if any(value > cap for value, cap in zip(values, hard_caps, strict=True)):
            raise ValueError("model ceilings must not exceed API hard caps")
        if self.rpd_policy not in {"wait", "fail"}:
            raise ValueError("rpd_policy must be wait or fail")


MODEL_CEILINGS = {
    MODEL_31B: ModelCeilings(),
    MODEL_26B: ModelCeilings(),
    MODEL_ANTIGRAVITY: ModelCeilings(
        requests_per_minute=ANTIGRAVITY_REQUESTS_PER_MINUTE,
        tokens_per_minute=ANTIGRAVITY_TOKENS_PER_MINUTE,
        requests_per_day=ANTIGRAVITY_REQUESTS_PER_DAY,
        hard_requests_per_minute=ANTIGRAVITY_HARD_REQUESTS_PER_MINUTE,
        hard_tokens_per_minute=ANTIGRAVITY_HARD_TOKENS_PER_MINUTE,
        hard_requests_per_day=ANTIGRAVITY_HARD_REQUESTS_PER_DAY,
        rpd_policy="fail",
    ),
}
_configured_schema_property = os.environ.get(
    "GEMMA_RESPONSE_SCHEMA_PROPERTY",
    "responseJsonSchema",
)
if _configured_schema_property not in {"responseJsonSchema", "responseSchema"}:
    raise RuntimeError(
        "GEMMA_RESPONSE_SCHEMA_PROPERTY must be responseJsonSchema or responseSchema"
    )
RESPONSE_SCHEMA_PROPERTY = cast(SchemaProperty, _configured_schema_property)


def get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required")
    return api_key


def ceilings_for(model: str) -> ModelCeilings:
    try:
        return MODEL_CEILINGS[model]
    except KeyError as error:
        raise ValueError(f"unsupported model for quota ceilings: {model}") from error
