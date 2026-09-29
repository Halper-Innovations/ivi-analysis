"""Run-local provider usage capture for autonomous research lanes.

The capture is intentionally context-local: nested company or validation runs
receive their own ledger and restore the parent sector ledger on exit.  It has
no persistence or provider side effects.
"""

from __future__ import annotations

import json
import math
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.llm.providers.retry_guard import (
    LLMCostBudgetExceeded,
    llm_attempt_observer,
    llm_cost_budget,
)
from app.llm.synthesis_agent import _estimate_cost_usd


_USAGE_SINK: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "autonomous_provider_usage_sink",
    default=None,
)
_USAGE_LANE: ContextVar[str | None] = ContextVar(
    "autonomous_provider_usage_lane",
    default=None,
)

_PROVIDER_RESPONSE_ATTEMPTS_ATTR = "_provider_response_attempts"
_PROVIDER_USAGE_RECORDS_ATTR = "_provider_usage_records"


@dataclass
class ProviderUsageBudget:
    """Synchronous paid-call ceiling for one run stage.

    A request reserves its deterministic worst-case cost before transport
    entry. Successful physical-response records then consume the ceiling at
    their canonical cost. The lock makes the contract safe if a caller copies
    the context into worker threads.
    """

    max_cost_usd: float
    spent_cost_usd: float = 0.0
    reserved_cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self, *, estimated_cost_usd: float, provider: str, schema_name: str) -> float:
        estimate = float(estimated_cost_usd)
        if not math.isfinite(estimate) or estimate <= 0.0:
            raise ValueError("provider request cost estimate must be finite and positive")
        with self._lock:
            available = max(
                0.0,
                float(self.max_cost_usd)
                - float(self.spent_cost_usd)
                - float(self.reserved_cost_usd),
            )
            if estimate > available + 1e-12:
                raise LLMCostBudgetExceeded(
                    "LLM cost budget exceeded before physical provider call "
                    f"(request ceiling ${estimate:.6f} > remaining ${available:.6f}) "
                    f"for {provider}:{schema_name}"
                )
            self.reserved_cost_usd += estimate
            return estimate

    def release(self, reservation: float) -> None:
        with self._lock:
            self.reserved_cost_usd = max(
                0.0,
                float(self.reserved_cost_usd) - max(0.0, float(reservation)),
            )

    def record(self, record: dict[str, Any]) -> None:
        value = record.get("cost_estimate_usd")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("provider usage cost must be numeric")
        cost = float(value)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("provider usage cost must be finite and non-negative")
        with self._lock:
            self.spent_cost_usd += cost
            if self.spent_cost_usd > float(self.max_cost_usd) + 1e-12:
                raise LLMCostBudgetExceeded(
                    "physical provider response exceeded the authorized LLM "
                    f"cost ceiling (${self.spent_cost_usd:.6f} > "
                    f"${float(self.max_cost_usd):.6f})"
                )

    def remaining(self) -> float:
        with self._lock:
            return max(
                0.0,
                float(self.max_cost_usd)
                - float(self.spent_cost_usd)
                - float(self.reserved_cost_usd),
            )


_USAGE_BUDGET: ContextVar[ProviderUsageBudget | None] = ContextVar(
    "provider_usage_budget",
    default=None,
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def _provider_name(provider: Any) -> str:
    return str(getattr(provider, "provider_name", provider.__class__.__name__) or "unknown").lower()


def _provider_model(provider: Any, result: Any | None = None) -> str:
    model = str(getattr(result, "model", "") or getattr(provider, "model", "") or "")
    if model:
        return model
    cfg = getattr(provider, "cfg", None)
    provider_name = _provider_name(provider)
    if provider_name == "openai":
        return str(getattr(cfg, "openai_model", "") or "unknown")
    if provider_name == "anthropic":
        return str(getattr(cfg, "anthropic_model", "") or "unknown")
    if provider_name == "deepseek":
        return str(getattr(cfg, "deepseek_model", "") or "unknown")
    return "unknown"


def provider_usage_meta(
    *,
    provider: Any,
    result: Any,
    prompt: str,
    schema_name: str,
) -> dict[str, Any]:
    """Return one canonical successful provider-call usage record."""

    raw_text = str(getattr(result, "json_text", result) or "")
    input_tokens = getattr(result, "usage_input_tokens", None)
    output_tokens = getattr(result, "usage_output_tokens", None)
    cached_input_tokens = getattr(result, "usage_cached_input_tokens", None)
    estimated = not isinstance(input_tokens, int) or not isinstance(output_tokens, int)
    normalized_input = (
        int(input_tokens) if isinstance(input_tokens, int) else _estimate_tokens(prompt)
    )
    normalized_output = (
        int(output_tokens) if isinstance(output_tokens, int) else _estimate_tokens(raw_text)
    )
    normalized_cached = (
        min(max(0, int(cached_input_tokens)), normalized_input)
        if isinstance(cached_input_tokens, int)
        else 0
    )
    provider_name = _provider_name(provider)
    model = _provider_model(provider, result)
    return {
        "status": "OK",
        "lane": _USAGE_LANE.get() or "unclassified",
        "provider": provider_name,
        "model": model,
        "schema_name": str(schema_name or "structured_output"),
        "input_tokens": normalized_input,
        "cached_input_tokens": normalized_cached,
        "output_tokens": normalized_output,
        "reserved_output_tokens": 0,
        "estimated_tokens": estimated,
        "cost_estimate_usd": _estimate_cost_usd(
            model,
            normalized_input,
            normalized_output,
            provider_name=provider_name,
            cached_input_tokens=normalized_cached,
        ),
    }


def provider_usage_records(
    *,
    provider: Any,
    result: Any,
    prompt: str,
    schema_name: str,
) -> list[dict[str, Any]]:
    """Return one record per successful physical provider response.

    OpenAI can retry a completed HTTP request with a larger output allowance.
    Its result preserves those response payloads in ``_response_attempts``;
    split them here so benchmark call counts and costs are not collapsed into
    one logical synthesis call.  Providers without that detail retain the
    canonical single record.
    """

    raw = getattr(result, "raw", None)
    attempts = raw.get("_response_attempts") if isinstance(raw, dict) else None
    if not isinstance(attempts, list) or not attempts:
        return [
            provider_usage_meta(
                provider=provider,
                result=result,
                prompt=prompt,
                schema_name=schema_name,
            )
        ]

    return provider_usage_records_from_response_payloads(
        provider=provider,
        response_payloads=attempts,
        prompt=prompt,
        schema_name=schema_name,
        result=result,
    )


def provider_usage_records_from_response_payloads(
    *,
    provider: Any,
    response_payloads: list[dict[str, Any]],
    prompt: str,
    schema_name: str,
    result: Any | None = None,
) -> list[dict[str, Any]]:
    """Normalize successful billed response payloads into physical-call rows."""

    attempts = [payload for payload in response_payloads if isinstance(payload, dict)]
    if not attempts:
        return []
    provider_name = _provider_name(provider)
    default_model = _provider_model(provider, result)
    lane = _USAGE_LANE.get() or "unclassified"
    records: list[dict[str, Any]] = []
    for index, payload in enumerate(attempts, start=1):
        normalized_payload = payload if isinstance(payload, dict) else {}
        usage = normalized_payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        input_value = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_value = usage.get("output_tokens", usage.get("completion_tokens"))
        details = usage.get("input_tokens_details")
        cached_value = details.get("cached_tokens") if isinstance(details, dict) else None
        if cached_value is None:
            # DeepSeek reports cache hits at the top level under its own name.
            cached_value = usage.get("prompt_cache_hit_tokens")
        normalized_input = (
            int(input_value)
            if isinstance(input_value, int) and not isinstance(input_value, bool)
            else _estimate_tokens(prompt)
        )
        normalized_output = (
            int(output_value)
            if isinstance(output_value, int) and not isinstance(output_value, bool)
            else _estimate_tokens(json.dumps(normalized_payload, sort_keys=True, default=str))
        )
        normalized_cached = (
            min(max(0, int(cached_value)), normalized_input)
            if isinstance(cached_value, int) and not isinstance(cached_value, bool)
            else 0
        )
        response_status = str(normalized_payload.get("status") or "").upper()
        status = "INCOMPLETE" if response_status == "INCOMPLETE" else "OK"
        model = str(normalized_payload.get("model") or default_model)
        records.append(
            {
                "status": status,
                "lane": lane,
                "provider": provider_name,
                "model": model,
                "schema_name": str(schema_name or "structured_output"),
                "input_tokens": normalized_input,
                "cached_input_tokens": normalized_cached,
                "output_tokens": normalized_output,
                "reserved_output_tokens": 0,
                "estimated_tokens": not (
                    isinstance(input_value, int)
                    and not isinstance(input_value, bool)
                    and isinstance(output_value, int)
                    and not isinstance(output_value, bool)
                ),
                "cost_estimate_usd": _estimate_cost_usd(
                    model,
                    normalized_input,
                    normalized_output,
                    provider_name=provider_name,
                    cached_input_tokens=normalized_cached,
                ),
                "physical_response_sequence": index,
                "physical_response_count": len(attempts),
            }
        )
    return records


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        current = cause


def attached_provider_usage_records(error: BaseException) -> list[dict[str, Any]]:
    """Return normalized provider rows carried by an exception or its cause."""

    for item in _exception_chain(error):
        records = getattr(item, _PROVIDER_USAGE_RECORDS_ATTR, None)
        if isinstance(records, list):
            return [
                json.loads(json.dumps(record, sort_keys=True, default=str))
                for record in records
                if isinstance(record, dict)
            ]
    return []


def provider_usage_records_from_exception(
    *,
    provider: Any,
    error: BaseException,
    prompt: str,
    schema_name: str,
) -> list[dict[str, Any]]:
    """Recover usage when a provider raises after one or more billed responses."""

    attached = attached_provider_usage_records(error)
    if attached:
        return attached
    for item in _exception_chain(error):
        payloads = getattr(item, _PROVIDER_RESPONSE_ATTEMPTS_ATTR, None)
        if isinstance(payloads, list) and payloads:
            return provider_usage_records_from_response_payloads(
                provider=provider,
                response_payloads=payloads,
                prompt=prompt,
                schema_name=schema_name,
            )
    return []


@contextmanager
def provider_failed_attempt_capture(
    *,
    provider: Any,
    prompt: str,
    schema_name: str,
    estimated_output_tokens: int,
) -> Iterator[list[dict[str, Any]]]:
    """Record every definitively failed transport attempt conservatively."""

    failed_attempts: list[dict[str, Any]] = []

    def observe(event: dict[str, Any]) -> None:
        error = event.get("error")
        if not isinstance(error, BaseException):
            error = RuntimeError(str(error or "physical provider attempt failed"))
        record = failed_provider_usage_meta(
            provider=provider,
            prompt=prompt,
            schema_name=schema_name,
            estimated_output_tokens=max(1, int(estimated_output_tokens)),
            error=error,
        )
        record.update(
            {
                "physical_attempt": int(event.get("attempt") or 1),
                "retryable": bool(event.get("retryable")),
                "will_retry": bool(event.get("will_retry")),
            }
        )
        failed_attempts.append(record)
        record_provider_usage(record)

    with llm_attempt_observer(observe):
        yield failed_attempts


def attach_provider_usage_to_exception(
    error: BaseException,
    records: list[dict[str, Any]],
) -> None:
    """Carry paid-call accounting through a failed logical run.

    Wrappers may attach the same subset at multiple layers.  Prefer the larger
    capture when one side is a prefix, which preserves every physical call
    without duplicating the propagated tail.
    """

    incoming = [
        json.loads(json.dumps(record, sort_keys=True, default=str))
        for record in records
        if isinstance(record, dict)
    ]
    if not incoming:
        return
    existing = getattr(error, _PROVIDER_USAGE_RECORDS_ATTR, None)
    existing = (
        [record for record in existing if isinstance(record, dict)]
        if isinstance(existing, list)
        else []
    )

    def comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key != "provider_call_id"} for row in rows
        ]

    existing_comparable = comparable(existing)
    incoming_comparable = comparable(incoming)

    def contains(
        haystack: list[dict[str, Any]],
        needle: list[dict[str, Any]],
    ) -> bool:
        if not needle:
            return True
        return any(
            haystack[index : index + len(needle)] == needle
            for index in range(len(haystack) - len(needle) + 1)
        )

    if existing_comparable == incoming_comparable:
        merged = incoming
    elif contains(incoming_comparable, existing_comparable):
        merged = incoming
    elif contains(existing_comparable, incoming_comparable):
        merged = existing
    else:
        merged = [*existing, *incoming]
    try:
        setattr(error, _PROVIDER_USAGE_RECORDS_ATTR, merged)
    except Exception:  # pragma: no cover - exotic immutable exception doubles
        return


def merge_provider_usage_records(
    *record_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append independently captured ledgers and assign artifact-local IDs."""

    merged: list[dict[str, Any]] = []
    for group in record_groups:
        for record in group:
            if not isinstance(record, dict):
                continue
            normalized = json.loads(json.dumps(record, sort_keys=True, default=str))
            normalized["provider_call_id"] = f"P{len(merged) + 1}"
            merged.append(normalized)
    return merged


def failed_provider_usage_meta(
    *,
    provider: Any,
    prompt: str,
    schema_name: str,
    estimated_output_tokens: int,
    error: BaseException,
) -> dict[str, Any]:
    """Conservatively reserve a failed attempt when provider usage is absent."""

    provider_name = _provider_name(provider)
    model = _provider_model(provider)
    input_tokens = _estimate_tokens(prompt)
    output_tokens = max(1, int(estimated_output_tokens))
    return {
        "status": "ERROR",
        "lane": _USAGE_LANE.get() or "unclassified",
        "provider": provider_name,
        "model": model,
        "schema_name": str(schema_name or "structured_output"),
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reserved_output_tokens": output_tokens,
        "estimated_tokens": True,
        "cost_basis": "CONSERVATIVE_FAILED_CALL_RESERVE",
        "cost_estimate_usd": _estimate_cost_usd(
            model,
            input_tokens,
            output_tokens,
            provider_name=provider_name,
        ),
        "error": f"{type(error).__name__}: {error}",
    }


def record_provider_usage(record: dict[str, Any]) -> None:
    sink = _USAGE_SINK.get()
    normalized = json.loads(json.dumps(record, sort_keys=True, default=str))
    if sink is not None:
        normalized["provider_call_id"] = f"P{len(sink) + 1}"
        sink.append(normalized)
    budget = _USAGE_BUDGET.get()
    if budget is not None:
        budget.record(normalized)


@contextmanager
def provider_usage_budget(max_cost_usd: float) -> Iterator[ProviderUsageBudget]:
    """Authorize paid responses up to an exact run-stage cost ceiling."""

    ceiling = float(max_cost_usd)
    if not math.isfinite(ceiling) or ceiling < 0.0:
        raise ValueError("provider usage budget must be finite and non-negative")
    budget = ProviderUsageBudget(max_cost_usd=ceiling)
    token = _USAGE_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _USAGE_BUDGET.reset(token)


def _request_cost_ceiling(
    *,
    provider: Any,
    prompt: str,
    schema: dict[str, Any],
    max_output_tokens: int,
) -> float:
    provider_name = _provider_name(provider)
    model = _provider_model(provider)
    # Byte length is a deterministic upper bound on tokenizer output for the
    # submitted UTF-8 prompt/schema payload. Include the structured-output
    # schema because providers can bill those request tokens as well.
    request_text = f"{prompt}\n{json.dumps(schema, sort_keys=True, separators=(',', ':'))}"
    input_token_ceiling = max(1, len(request_text.encode("utf-8")))
    output_token_ceiling = max(1, int(max_output_tokens))
    estimate = _estimate_cost_usd(
        model,
        input_token_ceiling,
        output_token_ceiling,
        provider_name=provider_name,
        cached_input_tokens=0,
    )
    # Canonical cost rounds to six decimals. A real request must never obtain
    # zero-dollar authorization solely because its bound rounded down.
    return max(0.000001, float(estimate))


@contextmanager
def provider_usage_request(
    *,
    provider: Any,
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int,
) -> Iterator[dict[str, Any]]:
    """Preauthorize one definitive physical provider request.

    Outside an RLM ``provider_usage_budget`` this is accounting-only and does
    not change provider retry behavior. Inside one, it reserves a conservative
    request ceiling and disables non-cancellable transport/output retries for
    production providers so the authorized amount bounds the physical call.
    """

    budget = _USAGE_BUDGET.get()
    if budget is None:
        yield {}
        return

    provider_name = _provider_name(provider)
    estimate = _request_cost_ceiling(
        provider=provider,
        prompt=prompt,
        schema=schema,
        max_output_tokens=max_output_tokens,
    )
    reservation = budget.reserve(
        estimated_cost_usd=estimate,
        provider=provider_name,
        schema_name=str(schema_name or "structured_output"),
    )
    provider_kwargs: dict[str, Any] = {}
    if bool(getattr(provider, "_handles_retry_guard", False)):
        provider_kwargs["max_output_tokens"] = max(1, int(max_output_tokens))
        if provider_name == "openai":
            provider_kwargs["allow_output_token_retry"] = False
            provider_kwargs["max_retries_per_request"] = 0
    try:
        # ``strict_first_call`` makes the shared retry guard refuse transparent
        # retries for transports (such as Anthropic) without caller-level
        # retry-control parameters.
        with llm_cost_budget(budget.remaining(), strict_first_call=True):
            yield provider_kwargs
    finally:
        budget.release(reservation)


@contextmanager
def provider_usage_capture(lane: str) -> Iterator[list[dict[str, Any]]]:
    sink: list[dict[str, Any]] = []
    sink_token = _USAGE_SINK.set(sink)
    lane_token = _USAGE_LANE.set(str(lane).strip() or "unclassified")
    try:
        yield sink
    finally:
        _USAGE_LANE.reset(lane_token)
        _USAGE_SINK.reset(sink_token)


@contextmanager
def provider_usage_lane(lane: str) -> Iterator[None]:
    token = _USAGE_LANE.set(str(lane).strip() or "unclassified")
    try:
        yield
    finally:
        _USAGE_LANE.reset(token)


def current_provider_usage_lane() -> str | None:
    return _USAGE_LANE.get()


__all__ = [
    "attach_provider_usage_to_exception",
    "attached_provider_usage_records",
    "current_provider_usage_lane",
    "failed_provider_usage_meta",
    "merge_provider_usage_records",
    "provider_usage_capture",
    "provider_usage_budget",
    "provider_failed_attempt_capture",
    "provider_usage_lane",
    "provider_usage_meta",
    "provider_usage_request",
    "provider_usage_records",
    "provider_usage_records_from_exception",
    "provider_usage_records_from_response_payloads",
    "record_provider_usage",
]
