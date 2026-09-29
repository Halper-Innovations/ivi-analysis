from __future__ import annotations

import copy
import inspect
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping

from app.alpha.llm_tools import (
    AlphaToolContext,
    alpha_tool_schemas,
    dispatch_alpha_tool,
    dispatch_alpha_tool_json,
    suggested_tool_calls_for_gaps,
)
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
    require_unchanged_financial_integrity_scope,
)
from app.autonomous.v1_financial_context import (
    bind_v1_financial_scope,
    financial_input_scenario,
)
from app.config import get_config
from app.util.http import require_network
from app.llm.providers import get_anthropic_provider, get_llm_provider
from app.llm.providers.retry_guard import LLMCostBudgetExceeded, llm_physical_attempt_guard
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    failed_provider_usage_meta,
    provider_failed_attempt_capture,
    provider_usage_budget,
    provider_usage_capture,
    provider_usage_records,
    provider_usage_records_from_exception,
    provider_usage_request,
    record_provider_usage,
)

logger = logging.getLogger(__name__)

_DEFAULT_ALPHA_REQUEST_COST_CAP_USD = 1.25

try:
    import anthropic as _anthropic_sdk
except ImportError:  # pragma: no cover - exercised only when dependency missing
    _anthropic_sdk = None  # type: ignore[assignment]

_PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "selected_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "reason": {"type": "string"},
                    "evidence_gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["ticker", "reason", "evidence_gaps"],
                "additionalProperties": False,
            },
        },
        "skipped_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["ticker", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "selected_targets", "skipped_candidates"],
    "additionalProperties": False,
}

_FINAL_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "runner_up": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "winner_thesis": {"type": "string"},
        "winner_conviction": {
            "type": "string",
            "enum": ["HIGH", "MODERATE", "LOW"],
        },
        "runner_up_thesis": {"type": "string"},
        "key_risk": {"type": "string"},
        "falsification_trigger": {"type": "string"},
        "time_horizon": {"type": "string"},
        "decision_trace": {"type": "string"},
        "rejected_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["ticker", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "winner",
        "runner_up",
        "winner_thesis",
        "winner_conviction",
        "runner_up_thesis",
        "key_risk",
        "falsification_trigger",
        "time_horizon",
        "decision_trace",
        "rejected_candidates",
    ],
    "additionalProperties": False,
}

_INVESTIGATION_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["PROCEED", "WATCH", "AVOID", "NO_WINNER"],
        },
        "confidence": {
            "type": "string",
            "enum": ["HIGH", "MODERATE", "LOW"],
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "key_risk": {"type": "string"},
        "falsification_trigger": {"type": "string"},
        "reasoning_trace": {"type": "string"},
        "model_validity": {
            "type": "string",
            "enum": [
                "VALID",
                "INVALID_SECURITY_TYPE",
                "INVALID_VALUATION_MODEL",
                "INSUFFICIENT_SECURITY_IDENTITY",
                "UNKNOWN",
            ],
        },
        "selection_blockers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "eligible_for_selection": {"type": "boolean"},
    },
    "required": [
        "verdict",
        "confidence",
        "key_findings",
        "open_questions",
        "key_risk",
        "falsification_trigger",
        "reasoning_trace",
        "model_validity",
        "selection_blockers",
        "eligible_for_selection",
    ],
    "additionalProperties": False,
}

_ANTHROPIC_SYSTEM_PROMPT = """You are running an LLM-first alpha investigation on a single company candidate.

Your job is to decide whether this candidate merits PROCEED, WATCH, or AVOID based on deterministic tools.

Rules:
- Use tools selectively to close the most important evidence gaps.
- Prefer primary-source filing passages, deterministic KPI trends, peer comparisons, current events, dilution, and liquidity evidence.
- Do not restate the candidate prior as if it were proof.
- If a tool returns unavailable or thin evidence, note that uncertainty rather than pretending it was resolved.
- If hard-block reasons are present, you may still investigate the name, but your reasoning must acknowledge the block.
- If evidence shows the ticker is the wrong security type, the valuation model is invalid, or security identity is insufficient, set model_validity accordingly and add a selection blocker.
- Call finalize_candidate_investigation once further evidence would not materially change the verdict.
"""


@dataclass
class AlphaInvestigationConfig:
    max_turns: int = 8
    max_tool_calls: int = 8
    max_cost_usd: float = 1.25
    max_output_tokens: int = 4000
    input_usd_per_mtok: float = 15.0
    output_usd_per_mtok: float = 75.0


@dataclass
class AlphaCandidateLoopResult:
    ticker: str
    verdict: str = "WATCH"
    confidence: str = "LOW"
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    key_risk: str = ""
    falsification_trigger: str = ""
    reasoning_trace: str = ""
    model_validity: str = "UNKNOWN"
    selection_blockers: list[str] = field(default_factory=list)
    confidence_cap_reasons: list[str] = field(default_factory=list)
    eligible_for_selection: bool = False
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    tool_transcript: list[dict[str, Any]] = field(default_factory=list)
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    termination_reason: str = ""
    error: str | None = None


@dataclass(frozen=True)
class _AlphaAnthropicResponse:
    response: Any
    model: str

    @property
    def usage(self) -> Any:
        return getattr(self.response, "usage", None)

    @property
    def content(self) -> Any:
        return getattr(self.response, "content", ())

    @property
    def usage_input_tokens(self) -> int:
        return int(getattr(self.usage, "input_tokens", 0) or 0)

    @property
    def usage_output_tokens(self) -> int:
        return int(getattr(self.usage, "output_tokens", 0) or 0)

    @property
    def json_text(self) -> str:
        return ""


def _provider_enabled(provider: Any) -> bool:
    enabled = getattr(provider, "enabled", None)
    if callable(enabled):
        try:
            return bool(enabled())
        except Exception:
            return False
    return False


def get_alpha_llm_provider() -> Any:
    cfg = get_config()
    provider_name = (cfg.llm_provider or "disabled").strip().lower()
    if provider_name == "anthropic":
        anthropic_provider = get_anthropic_provider()
        if anthropic_provider is not None:
            return anthropic_provider
    return get_llm_provider()


def alpha_llm_available() -> bool:
    return _provider_enabled(get_alpha_llm_provider())


def _alpha_provider_request_envelope(
    provider: Any,
    request: Mapping[str, Any],
    *,
    transport: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the full output-semantic request without binding credentials."""

    provider_name = str(
        getattr(provider, "provider_name", provider.__class__.__name__) or "unknown"
    ).lower()
    cfg = getattr(provider, "cfg", None)
    explicit_model = str(request.get("model") or "").strip()
    bound_model = str(getattr(provider, "model", "") or "").strip()
    configured_model = ""
    configured_output_limit: Any = None
    configured_timeout: Any = None
    if cfg is not None:
        if provider_name == "openai":
            configured_model = str(getattr(cfg, "openai_model", "") or "").strip()
            configured_output_limit = getattr(cfg, "openai_max_output_tokens", None)
            configured_timeout = getattr(cfg, "openai_request_timeout", None)
        elif provider_name == "anthropic":
            configured_model = str(getattr(cfg, "anthropic_model", "") or "").strip()
            configured_output_limit = getattr(cfg, "anthropic_max_output_tokens", None)
            configured_timeout = getattr(cfg, "anthropic_request_timeout", None)
        else:
            configured_model = str(getattr(cfg, "llm_model", "") or "").strip()

    signature_defaults: dict[str, Any] = {}
    try:
        parameters = inspect.signature(provider.synthesize_json).parameters
    except (AttributeError, TypeError, ValueError):
        parameters = {}
    for name, parameter in parameters.items():
        if name == "self" or parameter.default is inspect.Parameter.empty:
            continue
        signature_defaults[name] = parameter.default

    effective_output_limit = request.get("max_output_tokens")
    if effective_output_limit is None:
        effective_output_limit = request.get("max_tokens")
    if effective_output_limit is None:
        effective_output_limit = configured_output_limit
    return {
        "provider": {
            "name": provider_name,
            "class": f"{provider.__class__.__module__}.{provider.__class__.__qualname__}",
            "model": explicit_model or bound_model or configured_model,
            "request_timeout_seconds": configured_timeout,
            "handles_retry_guard": bool(getattr(provider, "_handles_retry_guard", False)),
        },
        "effective_defaults": {
            "max_output_tokens": effective_output_limit,
            "signature": signature_defaults,
        },
        "transport": dict(transport or {}),
        "request": request,
    }


@contextmanager
def _alpha_paid_provider_request(
    provider: Any,
    request: Mapping[str, Any],
    *,
    raw_anthropic_sdk: bool = False,
) -> Iterator[dict[str, Any]]:
    """Reserve the exact request ceiling before binding or transport."""

    base_request = dict(request)
    schema = base_request.get("schema")
    prompt = base_request.get("prompt")
    if not isinstance(schema, dict):
        schema = {
            "type": "object",
            "transport_tools": base_request.get("tools") or [],
            "tool_choice": base_request.get("tool_choice"),
        }
        prompt = json.dumps(
            {
                "system": base_request.get("system"),
                "messages": base_request.get("messages") or [],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    max_output_tokens = int(
        base_request.get("max_output_tokens") or base_request.get("max_tokens") or 1
    )
    with provider_usage_request(
        provider=provider,
        prompt=str(prompt or ""),
        schema=schema,
        schema_name=str(base_request.get("schema_name") or "alpha_anthropic_tool_loop_v1"),
        max_output_tokens=max(1, max_output_tokens),
    ) as transport_controls:
        effective_controls = dict(transport_controls)
        if raw_anthropic_sdk:
            reserved_output_tokens = effective_controls.pop("max_output_tokens", None)
            if reserved_output_tokens is not None and int(reserved_output_tokens) != max(
                1,
                max_output_tokens,
            ):
                raise RuntimeError(
                    "Anthropic reservation output limit does not match max_tokens"
                )
            if effective_controls:
                raise RuntimeError(
                    "Unsupported generic transport controls for raw Anthropic SDK: "
                    + ", ".join(sorted(effective_controls))
                )
            yield base_request
        else:
            yield {**base_request, **effective_controls}


def _alpha_usage_summary(
    records: list[dict[str, Any]],
    *,
    budget_usd: float | None = None,
    budget_remaining_usd: float | None = None,
) -> dict[str, Any]:
    summary = {
        "provider_usage": list(records),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in records),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in records),
        "cost_usd": round(
            sum(float(item.get("cost_estimate_usd") or 0.0) for item in records),
            6,
        ),
        "physical_calls": len(records),
    }
    if budget_usd is not None:
        summary["budget_usd"] = float(budget_usd)
    if budget_remaining_usd is not None:
        summary["budget_remaining_usd"] = float(budget_remaining_usd)
    return summary


def _json_loads(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


def _alpha_physical_attempt_authorizer(
    integrity_scope: FinancialIntegrityScope | None,
    *,
    context: str,
    run_as_of_date: str = "",
):
    """Bind one exact scope fingerprint to every physical provider attempt."""

    scope = integrity_scope or FinancialIntegrityScope(
        context=context,
        run_as_of_date=run_as_of_date,
        packets=(),
    )
    initial = require_financial_integrity_scope(scope)

    def require_exact_scope(_attempt: dict[str, Any] | None = None) -> None:
        require_unchanged_financial_integrity_scope(
            scope,
            expected_scope_fingerprint=initial.scope_fingerprint,
        )

    return require_exact_scope


def _alpha_prompt_attempt_authorizer(
    integrity_scope: FinancialIntegrityScope | None,
    *,
    context: str,
    financial_inputs_factory: Callable[[], Mapping[str, Any]],
    binding_packet: Any | None = None,
    run_as_of_date: str = "",
):
    """Bind the exact mutable financial prompt payload to one paid attempt."""

    require_base_scope = _alpha_physical_attempt_authorizer(
        integrity_scope,
        context=context,
        run_as_of_date=run_as_of_date,
    )
    require_base_scope()
    assert integrity_scope is not None
    packet = binding_packet if binding_packet is not None else integrity_scope.packets[0]
    effective_as_of_date = run_as_of_date or integrity_scope.run_as_of_date
    packet_ticker = (
        str(packet.get("ticker") if isinstance(packet, Mapping) else getattr(packet, "ticker", ""))
        .strip()
        .upper()
    )
    matching_scope_packet = next(
        (
            scope_packet
            for scope_packet in integrity_scope.packets
            if str(
                scope_packet.get("ticker")
                if isinstance(scope_packet, Mapping)
                else getattr(scope_packet, "ticker", "")
            )
            .strip()
            .upper()
            == packet_ticker
        ),
        None,
    )
    expected_packet_fingerprint = ""
    if matching_scope_packet is not None:
        expected_packet_fingerprint = require_financial_integrity_scope(
            FinancialIntegrityScope(
                context=f"{context}:prompt_packet",
                run_as_of_date=effective_as_of_date,
                packets=(matching_scope_packet,),
            )
        ).scope_fingerprint
    prompt_packet_scope = FinancialIntegrityScope(
        context=f"{context}:prompt_packet",
        run_as_of_date=effective_as_of_date,
        packets=(packet,),
    )

    def current_scenarios() -> tuple[dict[str, Any], ...]:
        return (
            financial_input_scenario(
                packet,
                financial_inputs=financial_inputs_factory(),
            ),
        )

    bound_prompt_scope = bind_v1_financial_scope(
        context=context,
        run_as_of_date=effective_as_of_date,
        packets=integrity_scope.packets,
        scenarios=current_scenarios(),
    )

    def require_exact_prompt(attempt: dict[str, Any] | None = None) -> None:
        require_base_scope(attempt)
        require_unchanged_financial_integrity_scope(
            prompt_packet_scope,
            expected_scope_fingerprint=expected_packet_fingerprint,
        )
        bound_prompt_scope.require(scenarios=current_scenarios())

    require_exact_prompt()
    return require_exact_prompt


def _call_alpha_provider_attempt(
    call,
    *,
    provider: Any,
    request: Mapping[str, Any],
    require_exact_scope,
) -> Any:
    """Account for one logical request and revalidate every terminal path."""

    failed_attempts: list[dict[str, Any]]
    successful_attempts: list[dict[str, Any]] = []
    prompt = str(request.get("prompt") or request.get("messages") or "")
    schema_name = str(request.get("schema_name") or "alpha_anthropic_tool_loop_v1")
    max_output_tokens = int(request.get("max_output_tokens") or request.get("max_tokens") or 1)
    try:
        with provider_failed_attempt_capture(
            provider=provider,
            prompt=prompt,
            schema_name=schema_name,
            estimated_output_tokens=max(1, max_output_tokens),
        ) as failed_attempts:
            result = call()
    except Exception as exc:
        successful_attempts = provider_usage_records_from_exception(
            provider=provider,
            error=exc,
            prompt=prompt,
            schema_name=schema_name,
        )
        for record in successful_attempts:
            record_provider_usage(record)
        if not failed_attempts and not successful_attempts:
            failure = failed_provider_usage_meta(
                provider=provider,
                prompt=prompt,
                schema_name=schema_name,
                estimated_output_tokens=max(1, max_output_tokens),
                error=exc,
            )
            failed_attempts.append(failure)
            record_provider_usage(failure)
        attach_provider_usage_to_exception(
            exc,
            [*failed_attempts, *successful_attempts],
        )
        try:
            require_exact_scope()
        except InvalidFinancialInputError as integrity_exc:
            attach_provider_usage_to_exception(
                integrity_exc,
                [*failed_attempts, *successful_attempts],
            )
            raise integrity_exc from exc
        raise
    successful_attempts = provider_usage_records(
        provider=provider,
        result=result,
        prompt=prompt,
        schema_name=schema_name,
    )
    for record in successful_attempts:
        record_provider_usage(record)
    try:
        require_exact_scope()
    except InvalidFinancialInputError as exc:
        attach_provider_usage_to_exception(
            exc,
            [*failed_attempts, *successful_attempts],
        )
        raise
    return result


def plan_alpha_investigations(
    *,
    sector: str,
    prior_candidates: list[dict[str, Any]],
    hard_blocked_candidates: list[dict[str, Any]],
    max_targets: int,
    integrity_scope: FinancialIntegrityScope | None = None,
    max_cost_usd: float = _DEFAULT_ALPHA_REQUEST_COST_CAP_USD,
) -> dict[str, Any]:
    provider = get_alpha_llm_provider()
    allowed = {str(candidate.get("ticker")).upper() for candidate in prior_candidates}
    eligible = {
        str(candidate.get("ticker")).upper()
        for candidate in prior_candidates
        if not candidate.get("hard_block_reasons")
    }
    prompt = (
        f"You are planning an LLM-first alpha investigation for the {sector} sector.\n\n"
        "Deterministic consensus ranking is a prior, not the chooser. Pick only the names where extra work "
        "is most likely to change the final selection. Keep the shortlist tight.\n\n"
        f"Prior candidates:\n{json.dumps(prior_candidates, indent=2, default=str)}\n\n"
        f"Hard-blocked candidates:\n{json.dumps(hard_blocked_candidates, indent=2, default=str)}\n\n"
        f"Select up to {max_targets} candidates to investigate more deeply. Favor names with real upside, "
        "method tension, unresolved risk, or evidence gaps. A hard-blocked candidate can still be investigated "
        "for context, but it is not eligible to win."
    )
    base_request = {
        "prompt": prompt,
        "schema": _PLANNER_SCHEMA,
        "schema_name": "alpha_shortlist_plan_v1",
        "max_output_tokens": 2500,
    }
    require_exact_scope = None
    budget_ceiling = max(0.0, float(max_cost_usd))
    with (
        provider_usage_budget(budget_ceiling) as usage_budget,
        provider_usage_capture("alpha_planner") as provider_usage,
    ):
        try:
            with _alpha_paid_provider_request(provider, base_request) as provider_request:
                require_exact_scope = _alpha_prompt_attempt_authorizer(
                    integrity_scope,
                    context=f"alpha_shortlist_plan:{sector}",
                    financial_inputs_factory=lambda: {
                        "prompt": prompt,
                        "prior_candidates": prior_candidates,
                        "hard_blocked_candidates": hard_blocked_candidates,
                        "max_targets": max_targets,
                        "provider_request": _alpha_provider_request_envelope(
                            provider,
                            provider_request,
                        ),
                    },
                )
                with llm_physical_attempt_guard(require_exact_scope):
                    require_exact_scope()
                    payload = _json_loads(
                        _call_alpha_provider_attempt(
                            lambda: provider.synthesize_json(**provider_request),
                            provider=provider,
                            request=provider_request,
                            require_exact_scope=require_exact_scope,
                        ).json_text
                    )
                    require_exact_scope()
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            if require_exact_scope is not None:
                require_exact_scope()
            logger.warning("alpha planner failed for %s: %s", sector, exc)
            fallback_targets = []
            for candidate in prior_candidates:
                ticker = str(candidate.get("ticker") or "").upper()
                if ticker and ticker in eligible:
                    fallback_targets.append(
                        {
                            "ticker": ticker,
                            "reason": (
                                "Fallback to highest-ranked eligible prior because "
                                "shortlist planning failed."
                            ),
                            "evidence_gaps": [
                                "Validate current thesis with KPI, liquidity, and filing evidence."
                            ],
                        }
                    )
                if len(fallback_targets) >= min(3, max_targets):
                    break
            return {
                "summary": (
                    "Planner unavailable — using highest-ranked eligible priors as fallback."
                ),
                "selected_targets": fallback_targets,
                "skipped_candidates": [],
                "planner_mode": "fallback",
                **_alpha_usage_summary(
                    provider_usage,
                    budget_usd=budget_ceiling,
                    budget_remaining_usd=usage_budget.remaining(),
                ),
            }

        selected_targets: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in payload.get("selected_targets", []):
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker") or "").upper()
            if not ticker or ticker not in allowed or ticker in seen:
                continue
            seen.add(ticker)
            selected_targets.append(
                {
                    "ticker": ticker,
                    "reason": str(item.get("reason") or "").strip()
                    or "Close the most material evidence gaps.",
                    "evidence_gaps": [
                        str(gap).strip()
                        for gap in (item.get("evidence_gaps") or [])
                        if str(gap).strip()
                    ][:5],
                }
            )
            if len(selected_targets) >= max_targets:
                break

        result = {
            "summary": str(payload.get("summary") or "").strip(),
            "selected_targets": selected_targets,
            "skipped_candidates": [
                {
                    "ticker": str(item.get("ticker") or "").upper(),
                    "reason": str(item.get("reason") or "").strip(),
                }
                for item in (payload.get("skipped_candidates") or [])
                if isinstance(item, dict) and str(item.get("ticker") or "").strip()
            ],
            "planner_mode": "llm",
            **_alpha_usage_summary(
                provider_usage,
                budget_usd=budget_ceiling,
                budget_remaining_usd=usage_budget.remaining(),
            ),
        }
        require_exact_scope()
        return result


def _summarize_investigation_with_provider(
    *,
    provider: Any,
    ctx: AlphaToolContext,
    investigation_request: dict[str, Any],
    hard_block_reasons: list[str],
    tool_results: list[dict[str, Any]],
    integrity_scope: FinancialIntegrityScope | None,
) -> dict[str, Any]:
    prompt = (
        f"You are evaluating {ctx.ticker} as an alpha candidate in the {ctx.sector} sector.\n\n"
        f"Candidate prior context:\n{json.dumps({'ticker': ctx.ticker, 'consensus_rank': ctx.consensus_rank, 'consensus_score': ctx.consensus_score, 'packet': ctx.packet.to_summary_dict()}, indent=2, default=str)}\n\n"
        f"Investigation request:\n{json.dumps(investigation_request, indent=2, default=str)}\n\n"
        f"Hard-block reasons:\n{json.dumps(hard_block_reasons, indent=2)}\n\n"
        f"Deterministic tool results:\n{json.dumps(tool_results, indent=2, default=str)}\n\n"
        "Return an evidence-backed candidate verdict. A hard-blocked candidate may still look interesting, "
        "but your verdict must reflect that it is not eligible to win. If the security type, valuation model, "
        "or security identity is invalid or insufficient, make that machine-readable in model_validity and "
        "selection_blockers."
    )
    base_request = {
        "prompt": prompt,
        "schema": _INVESTIGATION_SUMMARY_SCHEMA,
        "schema_name": "alpha_candidate_summary_v1",
        "max_output_tokens": 2500,
    }
    with _alpha_paid_provider_request(provider, base_request) as provider_request:
        require_exact_scope = _alpha_prompt_attempt_authorizer(
            integrity_scope,
            context=f"alpha_candidate_summary:{ctx.sector}:{ctx.ticker}",
            financial_inputs_factory=lambda: {
                "prompt": prompt,
                "investigation_request": investigation_request,
                "hard_block_reasons": hard_block_reasons,
                "tool_results": tool_results,
                "provider_request": _alpha_provider_request_envelope(
                    provider,
                    provider_request,
                ),
            },
            binding_packet=ctx.packet,
            run_as_of_date=ctx.as_of_date,
        )
        with llm_physical_attempt_guard(require_exact_scope):
            require_exact_scope()
            payload = _json_loads(
                _call_alpha_provider_attempt(
                    lambda: provider.synthesize_json(**provider_request),
                    provider=provider,
                    request=provider_request,
                    require_exact_scope=require_exact_scope,
                ).json_text
            )
            require_exact_scope()
        require_exact_scope()
    require_exact_scope()
    return payload


def _fallback_investigation_summary(
    *,
    ctx: AlphaToolContext,
    hard_block_reasons: list[str],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    findings = [f"Consensus prior rank {ctx.consensus_rank} with score {ctx.consensus_score}."]
    if hard_block_reasons:
        findings.append(
            "Candidate carries hard-block reasons that make it ineligible without evidence reversal."
        )
        verdict = "AVOID"
        confidence = "MODERATE"
        key_risk = "Hard-block evidence already points to disqualifying risk."
    else:
        verdict = "WATCH"
        confidence = "LOW"
        key_risk = "Evidence is mixed and requires judgment beyond deterministic fallback."
    for result in tool_results[:3]:
        tool = str(result.get("tool") or "")
        payload = result.get("result") if isinstance(result.get("result"), dict) else {}
        if tool == "analyze_dilution":
            findings.append(
                f"Share-count direction: {payload.get('dilution_direction') or 'UNKNOWN'}."
            )
        elif tool == "analyze_liquidity_stress":
            findings.append(f"Solvency risk: {payload.get('solvency_risk') or 'UNKNOWN'}.")
        elif tool == "compare_peer_metric":
            findings.append(
                f"Peer position: {payload.get('relative_position') or 'UNKNOWN'} on {payload.get('metric') or 'metric'}."
            )
    return {
        "verdict": verdict,
        "confidence": confidence,
        "key_findings": findings,
        "open_questions": ["LLM summary unavailable; revisit candidate once provider is restored."],
        "key_risk": key_risk,
        "falsification_trigger": "Re-run with full provider access and refreshed evidence before committing capital.",
        "reasoning_trace": "Fallback candidate investigation summary due to provider failure.",
        "model_validity": "UNKNOWN" if not hard_block_reasons else "INSUFFICIENT_SECURITY_IDENTITY",
        "selection_blockers": list(hard_block_reasons),
        "eligible_for_selection": not hard_block_reasons and verdict != "AVOID",
    }


def _canonical_selection_blockers(
    summary: dict[str, Any], hard_block_reasons: list[str]
) -> tuple[str, list[str], bool]:
    validity = str(summary.get("model_validity") or "UNKNOWN").upper()
    if validity not in {
        "VALID",
        "INVALID_SECURITY_TYPE",
        "INVALID_VALUATION_MODEL",
        "INSUFFICIENT_SECURITY_IDENTITY",
        "UNKNOWN",
    }:
        validity = "UNKNOWN"
    blockers = [
        str(item).strip() for item in (summary.get("selection_blockers") or []) if str(item).strip()
    ]
    invalid_to_blocker = {
        "INVALID_SECURITY_TYPE": "INVALID_SECURITY_TYPE",
        "INVALID_VALUATION_MODEL": "INVALID_VALUATION_MODEL",
        "INSUFFICIENT_SECURITY_IDENTITY": "INSUFFICIENT_SECURITY_IDENTITY",
    }
    if validity in invalid_to_blocker and invalid_to_blocker[validity] not in blockers:
        blockers.append(invalid_to_blocker[validity])
    for reason in hard_block_reasons:
        if reason not in blockers:
            blockers.append(reason)
    verdict = str(summary.get("verdict") or "").upper()
    eligible_requested = bool(summary.get("eligible_for_selection", True))
    if not eligible_requested and not blockers:
        blockers.append("LLM_MARKED_INELIGIBLE")
    eligible = eligible_requested and not blockers and verdict != "AVOID"
    return validity, blockers, eligible


_CONFIDENCE_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


def _cap_confidence(confidence: str, cap: str) -> str:
    current = str(confidence or "LOW").upper()
    cap_upper = str(cap or "LOW").upper()
    if _CONFIDENCE_ORDER.get(current, 0) <= _CONFIDENCE_ORDER.get(cap_upper, 0):
        return current
    return cap_upper


def _apply_deterministic_evidence_gates(
    ctx: AlphaToolContext, result: AlphaCandidateLoopResult
) -> None:
    """Apply subtype-specific evidence sufficiency after LLM/fallback judgment."""
    packet = ctx.packet
    if packet.issuer_type != "insurance_underwriter":
        return
    insurance_packet = packet.insurance_packet if isinstance(packet.insurance_packet, dict) else {}
    operating_metrics = (
        insurance_packet.get("operating_metrics")
        if isinstance(insurance_packet.get("operating_metrics"), dict)
        else {}
    )
    missing = {str(item) for item in (operating_metrics.get("missing_components") or [])}
    metric_family = str(operating_metrics.get("metric_family") or "")
    warnings = {
        str(item)
        for item in (packet.model_fit_warnings or insurance_packet.get("model_fit_warnings") or [])
    }

    def _add_blocker(reason: str) -> None:
        if reason not in result.selection_blockers:
            result.selection_blockers.append(reason)

    def _add_cap(reason: str, cap: str = "MODERATE") -> None:
        if reason not in result.confidence_cap_reasons:
            result.confidence_cap_reasons.append(reason)
        result.confidence = _cap_confidence(result.confidence, cap)

    if metric_family == "pc_insurance":
        core_missing = missing & {"COMBINED_RATIO", "LOSS_RATIO", "EXPENSE_RATIO"}
        if operating_metrics.get("status") != "OK" or core_missing:
            _add_blocker("PC_CORE_OPERATING_METRICS_MISSING")
            _add_cap("PC_CORE_OPERATING_METRICS_MISSING", "LOW")
        if "RESERVE_DEVELOPMENT" in missing:
            _add_cap("PC_RESERVE_DEVELOPMENT_UNKNOWN")
    elif metric_family == "mortgage_insurance":
        core_missing = missing & {
            "PMIER_EXCESS_RATIO",
            "PRIMARY_IIF",
            "PRIMARY_RIF",
            "NIW",
            "DEFAULT_RATE",
        }
        if operating_metrics.get("status") != "OK" or core_missing:
            _add_blocker("MORTGAGE_CORE_OPERATING_METRICS_MISSING")
            _add_cap("MORTGAGE_CORE_OPERATING_METRICS_MISSING", "LOW")
        if operating_metrics.get("credit_capital_assessment") in {
            "MORTGAGE_CREDIT_STRESS",
            "PMIER_CAPITAL_THIN",
        }:
            _add_blocker("MORTGAGE_CREDIT_OR_CAPITAL_STRESS")
            _add_cap("MORTGAGE_CREDIT_OR_CAPITAL_STRESS", "LOW")
    elif packet.insurance_subtype in {"pc_insurer", "reinsurer", "title_mortgage_specialty"}:
        _add_blocker("INSURANCE_OPERATING_METRICS_NOT_AVAILABLE")
        _add_cap("INSURANCE_OPERATING_METRICS_NOT_AVAILABLE", "LOW")

    statutory_warnings = warnings & {
        "RBC_OR_BCAR",
        "STATUTORY_SURPLUS",
        "RATINGS_MODEL",
        "ALM_DETAIL",
    }
    if statutory_warnings:
        _add_cap("INSURANCE_STATUTORY_RATING_OR_ALM_EVIDENCE_MISSING")

    risk_signals = (
        packet.filing_risk_signals if isinstance(packet.filing_risk_signals, dict) else {}
    )
    tracked_signals = [
        str(risk_signals.get(key) or "UNKNOWN").upper()
        for key in (
            "competitive_disruption",
            "secular_decline",
            "regulatory_legal",
            "customer_concentration",
        )
    ]
    if tracked_signals and all(value == "UNKNOWN" for value in tracked_signals):
        _add_cap("FILING_RISK_SIGNALS_UNKNOWN")

    if result.selection_blockers:
        result.eligible_for_selection = False


def _run_fallback_candidate_investigation(
    *,
    ctx: AlphaToolContext,
    investigation_request: dict[str, Any],
    hard_block_reasons: list[str],
    cfg: AlphaInvestigationConfig,
    integrity_scope: FinancialIntegrityScope | None,
) -> AlphaCandidateLoopResult:
    provider = get_alpha_llm_provider()
    result = AlphaCandidateLoopResult(ticker=ctx.ticker)
    tool_results: list[dict[str, Any]] = []
    for tool_name, payload in suggested_tool_calls_for_gaps(
        [str(gap) for gap in (investigation_request.get("evidence_gaps") or [])]
    )[: cfg.max_tool_calls]:
        tool_output = dispatch_alpha_tool(tool_name, payload, ctx)
        result.tool_call_counts[tool_name] = result.tool_call_counts.get(tool_name, 0) + 1
        result.tool_transcript.append(
            {
                "turn": 1,
                "tool": tool_name,
                "input": payload,
                "output_preview": json.dumps(tool_output)[:500],
                "output_chars": len(json.dumps(tool_output)),
            }
        )
        tool_results.append({"tool": tool_name, "input": payload, "result": tool_output})
    result.num_turns = 1
    if not _provider_enabled(provider):
        summary = _fallback_investigation_summary(
            ctx=ctx,
            hard_block_reasons=hard_block_reasons,
            tool_results=tool_results,
        )
        result.termination_reason = "deterministic_summary_provider_unavailable"
    else:
        try:
            summary = _summarize_investigation_with_provider(
                provider=provider,
                ctx=ctx,
                investigation_request=investigation_request,
                hard_block_reasons=hard_block_reasons,
                tool_results=tool_results,
                integrity_scope=integrity_scope,
            )
            result.termination_reason = "fallback_summary"
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            logger.warning("alpha candidate summary failed for %s: %s", ctx.ticker, exc)
            summary = _fallback_investigation_summary(
                ctx=ctx,
                hard_block_reasons=hard_block_reasons,
                tool_results=tool_results,
            )
            result.error = str(exc)
            result.termination_reason = "fallback_summary_error"
    result.verdict = str(summary.get("verdict") or "WATCH")
    result.confidence = str(summary.get("confidence") or "LOW")
    result.key_findings = [
        str(item) for item in (summary.get("key_findings") or []) if str(item).strip()
    ]
    result.open_questions = [
        str(item) for item in (summary.get("open_questions") or []) if str(item).strip()
    ]
    result.key_risk = str(summary.get("key_risk") or "")
    result.falsification_trigger = str(summary.get("falsification_trigger") or "")
    result.reasoning_trace = str(summary.get("reasoning_trace") or "")
    result.model_validity, result.selection_blockers, result.eligible_for_selection = (
        _canonical_selection_blockers(
            summary,
            hard_block_reasons,
        )
    )
    _apply_deterministic_evidence_gates(ctx, result)
    return result


def _run_anthropic_candidate_investigation(
    *,
    provider: Any,
    ctx: AlphaToolContext,
    investigation_request: dict[str, Any],
    hard_block_reasons: list[str],
    cfg: AlphaInvestigationConfig,
    integrity_scope: FinancialIntegrityScope | None,
) -> AlphaCandidateLoopResult:
    if _anthropic_sdk is None:
        return _run_fallback_candidate_investigation(
            ctx=ctx,
            investigation_request=investigation_request,
            hard_block_reasons=hard_block_reasons,
            cfg=cfg,
            integrity_scope=integrity_scope,
        )
    app_cfg = get_config()
    client = _anthropic_sdk.Anthropic(
        api_key=app_cfg.anthropic_api_key,
        base_url=app_cfg.anthropic_base_url,
        timeout=float(app_cfg.anthropic_request_timeout),
        max_retries=0,
    )
    result = AlphaCandidateLoopResult(ticker=ctx.ticker)
    tools = alpha_tool_schemas()
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Investigate this alpha candidate and decide whether it merits PROCEED, WATCH, or AVOID.\n\n"
                f"Candidate prior context:\n{json.dumps({'ticker': ctx.ticker, 'sector': ctx.sector, 'consensus_rank': ctx.consensus_rank, 'consensus_score': ctx.consensus_score, 'packet': ctx.packet.to_summary_dict()}, indent=2, default=str)}\n\n"
                f"Investigation request:\n{json.dumps(investigation_request, indent=2, default=str)}\n\n"
                f"Hard-block reasons:\n{json.dumps(hard_block_reasons, indent=2)}"
            ),
        }
    ]
    tool_calls_used = 0
    start = time.perf_counter()
    transport = {
        "request_timeout_seconds": float(app_cfg.anthropic_request_timeout),
        "max_retries": 0,
    }
    for turn in range(1, cfg.max_turns + 1):
        if result.cost_usd >= max(0.0, float(cfg.max_cost_usd)):
            result.termination_reason = "budget_cap"
            break
        base_request = {
            "model": str(getattr(provider, "model", "") or app_cfg.anthropic_model),
            "max_tokens": cfg.max_output_tokens,
            "system": _ANTHROPIC_SYSTEM_PROMPT,
            "tools": copy.deepcopy(tools),
            "tool_choice": {"type": "auto"},
            "messages": copy.deepcopy(messages),
        }
        with _alpha_paid_provider_request(
            provider,
            base_request,
            raw_anthropic_sdk=True,
        ) as provider_request:
            require_exact_scope = _alpha_prompt_attempt_authorizer(
                integrity_scope,
                context=f"alpha_anthropic_tool_loop:{ctx.sector}:{ctx.ticker}",
                financial_inputs_factory=lambda provider_request=provider_request: {
                    "provider_request": _alpha_provider_request_envelope(
                        provider,
                        provider_request,
                        transport=transport,
                    ),
                    "investigation_request": investigation_request,
                    "hard_block_reasons": hard_block_reasons,
                    "conversation_state": messages,
                    "tool_schemas": tools,
                },
                binding_packet=ctx.packet,
                run_as_of_date=ctx.as_of_date,
            )
            try:
                with llm_physical_attempt_guard(require_exact_scope):
                    require_exact_scope(
                        {
                            "provider": "anthropic",
                            "schema_name": "alpha_anthropic_tool_loop_v1",
                            "attempt": turn,
                        }
                    )
                    # VOE_NET_PROVIDER is the single switch for all outbound traffic.
                    require_network(app_cfg.anthropic_base_url, app_cfg)
                    response = _call_alpha_provider_attempt(
                        lambda provider_request=provider_request: _AlphaAnthropicResponse(
                            client.messages.create(**provider_request),
                            model=str(provider_request["model"]),
                        ),
                        provider=provider,
                        request=provider_request,
                        require_exact_scope=require_exact_scope,
                    )
            except Exception as exc:
                try:
                    require_exact_scope()
                except InvalidFinancialInputError as integrity_exc:
                    raise integrity_exc from exc
                raise
            require_exact_scope()
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        result.input_tokens += int(input_tokens or 0)
        result.output_tokens += int(output_tokens or 0)
        result.cost_usd += (int(input_tokens or 0) / 1_000_000) * cfg.input_usd_per_mtok + (
            int(output_tokens or 0) / 1_000_000
        ) * cfg.output_usd_per_mtok
        result.num_turns = turn
        require_exact_scope()
        assistant_blocks: list[dict[str, Any]] = []
        tool_uses: list[tuple[str, str, dict[str, Any]]] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                assistant_blocks.append({"type": "text", "text": getattr(block, "text", "")})
            elif block_type == "tool_use":
                tool_input = block.input or {}
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": tool_input,
                    }
                )
                tool_uses.append((block.id, block.name, tool_input))
                result.tool_call_counts[block.name] = result.tool_call_counts.get(block.name, 0) + 1
        messages.append({"role": "assistant", "content": assistant_blocks})
        if not tool_uses:
            result.termination_reason = "no_tool_call"
            break
        finalize_call = next(
            (
                (tool_id, name, payload)
                for tool_id, name, payload in tool_uses
                if name == "finalize_candidate_investigation"
            ),
            None,
        )
        if finalize_call is not None:
            _, _, payload = finalize_call
            result.verdict = str(payload.get("verdict") or "WATCH")
            result.confidence = str(payload.get("confidence") or "LOW")
            result.key_findings = [
                str(item) for item in (payload.get("key_findings") or []) if str(item).strip()
            ]
            result.open_questions = [
                str(item) for item in (payload.get("open_questions") or []) if str(item).strip()
            ]
            result.key_risk = str(payload.get("key_risk") or "")
            result.falsification_trigger = str(payload.get("falsification_trigger") or "")
            result.reasoning_trace = str(payload.get("reasoning_trace") or "")
            result.model_validity, result.selection_blockers, result.eligible_for_selection = (
                _canonical_selection_blockers(
                    payload,
                    hard_block_reasons,
                )
            )
            _apply_deterministic_evidence_gates(ctx, result)
            result.termination_reason = "finalize_candidate_investigation"
            break
        tool_results: list[dict[str, Any]] = []
        for tool_id, tool_name, tool_input in tool_uses:
            if tool_calls_used >= cfg.max_tool_calls:
                break
            tool_calls_used += 1
            tool_output = dispatch_alpha_tool_json(tool_name, tool_input, ctx)
            result.tool_transcript.append(
                {
                    "turn": turn,
                    "tool": tool_name,
                    "input": tool_input,
                    "output_preview": tool_output[:500],
                    "output_chars": len(tool_output),
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tool_output,
                }
            )
        if not tool_results:
            result.termination_reason = "tool_cap_reached"
            break
        messages.append({"role": "user", "content": tool_results})
        if tool_calls_used >= cfg.max_tool_calls:
            result.termination_reason = "tool_cap_reached"
            break
        if result.cost_usd >= cfg.max_cost_usd:
            result.termination_reason = "budget_cap"
            break
    if not result.termination_reason:
        result.termination_reason = "max_turns"
    if not result.key_findings:
        fallback = _run_fallback_candidate_investigation(
            ctx=ctx,
            investigation_request=investigation_request,
            hard_block_reasons=hard_block_reasons,
            cfg=cfg,
            integrity_scope=integrity_scope,
        )
        fallback.tool_transcript = result.tool_transcript + fallback.tool_transcript
        fallback.tool_call_counts = {
            **result.tool_call_counts,
            **{
                name: result.tool_call_counts.get(name, 0) + fallback.tool_call_counts.get(name, 0)
                for name in set(result.tool_call_counts) | set(fallback.tool_call_counts)
            },
        }
        fallback.input_tokens += result.input_tokens
        fallback.output_tokens += result.output_tokens
        fallback.cost_usd += result.cost_usd
        fallback.num_turns = max(result.num_turns, fallback.num_turns)
        fallback.termination_reason = f"{result.termination_reason}_with_fallback"
        fallback.error = result.error
        return fallback
    result.reasoning_trace = (
        result.reasoning_trace or f"Anthropic tool loop finished in {result.num_turns} turns."
    )
    result.tool_transcript.append({"wall_seconds": round(time.perf_counter() - start, 3)})
    return result


def run_candidate_investigation(
    *,
    ctx: AlphaToolContext,
    investigation_request: dict[str, Any],
    hard_block_reasons: list[str],
    config: AlphaInvestigationConfig | None = None,
    integrity_scope: FinancialIntegrityScope | None = None,
) -> dict[str, Any]:
    cfg = config or AlphaInvestigationConfig()
    provider = get_alpha_llm_provider()
    provider_name = str(getattr(provider, "provider_name", "") or "").lower()
    with (
        provider_usage_budget(max(0.0, float(cfg.max_cost_usd))),
        provider_usage_capture(f"alpha_candidate:{ctx.ticker}") as provider_usage,
    ):
        if provider_name == "anthropic" and _provider_enabled(provider):
            try:
                loop_result = _run_anthropic_candidate_investigation(
                    provider=provider,
                    ctx=ctx,
                    investigation_request=investigation_request,
                    hard_block_reasons=hard_block_reasons,
                    cfg=cfg,
                    integrity_scope=integrity_scope,
                )
            except LLMCostBudgetExceeded as exc:
                loop_result = _run_fallback_candidate_investigation(
                    ctx=ctx,
                    investigation_request=investigation_request,
                    hard_block_reasons=hard_block_reasons,
                    cfg=cfg,
                    integrity_scope=integrity_scope,
                )
                loop_result.error = str(exc)
                loop_result.termination_reason = "budget_cap_with_deterministic_fallback"
            investigation_mode = "anthropic_tool_loop"
        else:
            loop_result = _run_fallback_candidate_investigation(
                ctx=ctx,
                investigation_request=investigation_request,
                hard_block_reasons=hard_block_reasons,
                cfg=cfg,
                integrity_scope=integrity_scope,
            )
            investigation_mode = (
                f"{provider_name}_tool_bundle"
                if provider_name and provider_name != "disabled" and _provider_enabled(provider)
                else "fallback_tool_bundle"
            )
        usage_summary = _alpha_usage_summary(provider_usage)
        loop_result.input_tokens = int(usage_summary["input_tokens"])
        loop_result.output_tokens = int(usage_summary["output_tokens"])
        loop_result.cost_usd = float(usage_summary["cost_usd"])
    return {
        "ticker": ctx.ticker,
        "verdict": loop_result.verdict,
        "confidence": loop_result.confidence,
        "key_findings": loop_result.key_findings,
        "open_questions": loop_result.open_questions,
        "key_risk": loop_result.key_risk,
        "falsification_trigger": loop_result.falsification_trigger,
        "reasoning_trace": loop_result.reasoning_trace,
        "model_validity": loop_result.model_validity,
        "selection_blockers": list(loop_result.selection_blockers),
        "confidence_cap_reasons": list(loop_result.confidence_cap_reasons),
        "tool_call_counts": dict(loop_result.tool_call_counts),
        "tool_transcript": list(loop_result.tool_transcript),
        "num_turns": loop_result.num_turns,
        "input_tokens": loop_result.input_tokens,
        "output_tokens": loop_result.output_tokens,
        "cost_usd": round(loop_result.cost_usd, 4),
        "provider_usage": usage_summary["provider_usage"],
        "physical_calls": usage_summary["physical_calls"],
        "termination_reason": loop_result.termination_reason,
        "error": loop_result.error,
        "investigation_mode": investigation_mode,
        "hard_block_reasons": list(hard_block_reasons),
        "eligible_for_selection": bool(loop_result.eligible_for_selection),
        "requested_focus": {
            "reason": str(investigation_request.get("reason") or ""),
            "evidence_gaps": [
                str(item)
                for item in (investigation_request.get("evidence_gaps") or [])
                if str(item).strip()
            ],
        },
    }


def decide_alpha_winner(
    *,
    sector: str,
    prior_ranking: list[dict[str, Any]],
    investigation_plan: dict[str, Any],
    candidate_investigations: list[dict[str, Any]],
    hard_blocked_candidates: list[dict[str, Any]],
    integrity_scope: FinancialIntegrityScope | None = None,
    max_cost_usd: float = _DEFAULT_ALPHA_REQUEST_COST_CAP_USD,
) -> dict[str, Any]:
    provider = get_alpha_llm_provider()
    eligible_tickers = [
        str(item.get("ticker") or "").upper()
        for item in candidate_investigations
        if item.get("eligible_for_selection") is True
    ]
    prompt = (
        f"You are making the canonical alpha decision for the {sector} sector.\n\n"
        "Deterministic consensus ranking is only a prior. The final winner must come from the investigated, "
        "eligible candidates based on the evidence gathered.\n\n"
        f"Prior ranking:\n{json.dumps(prior_ranking, indent=2, default=str)}\n\n"
        f"Investigation plan:\n{json.dumps(investigation_plan, indent=2, default=str)}\n\n"
        f"Candidate investigations:\n{json.dumps(candidate_investigations, indent=2, default=str)}\n\n"
        f"Hard-blocked candidates:\n{json.dumps(hard_blocked_candidates, indent=2, default=str)}\n\n"
        f"Eligible investigated tickers: {eligible_tickers}\n\n"
        "Choose the best eligible candidate if one genuinely stands out. If no candidate deserves selection, set winner to null."
    )
    base_request = {
        "prompt": prompt,
        "schema": _FINAL_DECISION_SCHEMA,
        "schema_name": "alpha_final_decision_v1",
        "max_output_tokens": 3000,
    }
    require_exact_scope = None
    budget_ceiling = max(0.0, float(max_cost_usd))
    with (
        provider_usage_budget(budget_ceiling) as usage_budget,
        provider_usage_capture("alpha_final_decision") as provider_usage,
    ):
        try:
            with _alpha_paid_provider_request(provider, base_request) as provider_request:
                require_exact_scope = _alpha_prompt_attempt_authorizer(
                    integrity_scope,
                    context=f"alpha_final_decision:{sector}",
                    financial_inputs_factory=lambda: {
                        "prompt": prompt,
                        "prior_ranking": prior_ranking,
                        "investigation_plan": investigation_plan,
                        "candidate_investigations": candidate_investigations,
                        "hard_blocked_candidates": hard_blocked_candidates,
                        "provider_request": _alpha_provider_request_envelope(
                            provider,
                            provider_request,
                        ),
                    },
                )
                with llm_physical_attempt_guard(require_exact_scope):
                    require_exact_scope()
                    payload = _json_loads(
                        _call_alpha_provider_attempt(
                            lambda: provider.synthesize_json(**provider_request),
                            provider=provider,
                            request=provider_request,
                            require_exact_scope=require_exact_scope,
                        ).json_text
                    )
                    require_exact_scope()
            decision_mode = "llm"
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            if require_exact_scope is not None:
                require_exact_scope()
            logger.warning("alpha final decision failed for %s: %s", sector, exc)
            payload = {
                "winner": None,
                "runner_up": None,
                "winner_thesis": "No winner selected because the final LLM decision failed.",
                "winner_conviction": "LOW",
                "runner_up_thesis": "",
                "key_risk": (
                    "Final LLM decision failed, so this result should be treated "
                    "as degraded and non-actionable."
                ),
                "falsification_trigger": (
                    "Re-run alpha-scan with a working provider before acting on this result."
                ),
                "time_horizon": "12 months",
                "decision_trace": f"Final decision unavailable due to provider error: {exc}",
                "rejected_candidates": [],
            }
            decision_mode = "fallback"
        winner = str(payload.get("winner") or "").upper() or None
        runner_up = str(payload.get("runner_up") or "").upper() or None
        eligible_set = set(eligible_tickers)
        if winner not in eligible_set:
            winner = None
        if runner_up not in eligible_set or runner_up == winner:
            runner_up = None
        payload["winner"] = winner
        payload["runner_up"] = runner_up
        payload["decision_mode"] = decision_mode
        payload.update(
            _alpha_usage_summary(
                provider_usage,
                budget_usd=budget_ceiling,
                budget_remaining_usd=usage_budget.remaining(),
            )
        )
        if require_exact_scope is not None:
            require_exact_scope()
        return payload
