"""Context-local hard limits for paid LLM execution.

The benchmark cost preflight is only meaningful if execution cannot silently
select another provider/model or submit a request larger than the priced bound.
This module contains no provider imports so providers can enforce it without a
circular dependency.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class LLMExecutionPolicy:
    provider: str
    model: str
    max_serialized_request_bytes: int
    max_output_tokens: int
    max_retries_per_request: int
    allow_provider_fallback: bool = False
    allow_output_token_retry: bool = False

    def __post_init__(self) -> None:
        if str(self.provider or "").strip().lower() != "openai":
            raise ValueError("bounded all-sector execution requires provider=openai")
        model = str(self.model or "").strip().lower()
        if model != "gpt-5.5" and not model.startswith("gpt-5.5-"):
            raise ValueError("bounded all-sector execution requires GPT-5.5")
        for name in (
            "max_serialized_request_bytes",
            "max_output_tokens",
            "max_retries_per_request",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.max_serialized_request_bytes or not self.max_output_tokens:
            raise ValueError("request and output bounds must be positive")


_ACTIVE_POLICY: ContextVar[LLMExecutionPolicy | None] = ContextVar(
    "active_llm_execution_policy",
    default=None,
)


def active_llm_execution_policy() -> LLMExecutionPolicy | None:
    return _ACTIVE_POLICY.get()


@contextmanager
def llm_execution_policy(policy: LLMExecutionPolicy) -> Iterator[None]:
    token = _ACTIVE_POLICY.set(policy)
    try:
        yield
    finally:
        _ACTIVE_POLICY.reset(token)


def enforce_openai_request_policy(
    *,
    model: str,
    request_payload: dict[str, Any],
    max_output_tokens: int,
) -> LLMExecutionPolicy | None:
    """Reject drift before the provider performs any network I/O."""

    policy = active_llm_execution_policy()
    if policy is None:
        return None
    resolved_model = str(model or "").strip().lower()
    if resolved_model != policy.model.lower():
        raise RuntimeError(
            "LLM execution policy rejected model drift: "
            f"expected={policy.model}, received={resolved_model or 'missing'}"
        )
    if int(max_output_tokens) > policy.max_output_tokens:
        raise RuntimeError(
            "LLM execution policy rejected output bound: "
            f"requested={int(max_output_tokens)}, max={policy.max_output_tokens}"
        )
    serialized_bytes = len(
        json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    )
    if serialized_bytes > policy.max_serialized_request_bytes:
        raise RuntimeError(
            "LLM execution policy rejected input bound: "
            f"serialized_request_bytes={serialized_bytes}, "
            f"max={policy.max_serialized_request_bytes}"
        )
    return policy


__all__ = [
    "LLMExecutionPolicy",
    "active_llm_execution_policy",
    "enforce_openai_request_policy",
    "llm_execution_policy",
]
