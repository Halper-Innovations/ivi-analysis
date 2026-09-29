"""Runtime orchestration for the single-candidate autonomous analyst loop."""

from __future__ import annotations

import inspect
import json
import math
import secrets
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from typing import Any, Callable, Mapping

from app.alpha.llm_runtime import _provider_enabled, get_alpha_llm_provider
from app.alpha.llm_tools import AlphaToolContext, dispatch_alpha_tool
from app.alpha.schemas import TickerSignalPacket
from app.alpha.signal_assembler import assemble_signal_packet
from app.autonomous.financial_integrity import (
    FinancialIntegrityGateResult,
    FinancialIntegrityScope,
    FinancialIntegrityViolation,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
    require_unchanged_financial_integrity_scope,
)
from app.autonomous.run_contract import (
    AutonomousRunArtifact,
    AutonomousRunBudget,
    AutonomousRunRequest,
    BeliefUpdate,
    CandidateDecision,
    EvidenceReference,
    ResearchQuestion,
    ToolCallRecord,
)
from app.autonomous.sector_contract import (
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    canonical_v2_signal_packet_snapshot,
    v2_signal_packet_snapshot_fingerprint,
)
from app.autonomous.tool_input_guardrails import repair_alpha_tool_input
from app.llm.providers import get_anthropic_provider
from app.llm.providers.retry_guard import (
    current_cost_context,
    llm_attempt_observer,
    llm_physical_attempt_guard,
    require_llm_physical_attempt_authorization,
)
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    failed_provider_usage_meta,
    merge_provider_usage_records,
    provider_usage_capture,
    provider_usage_records,
    provider_usage_records_from_exception,
    record_provider_usage,
)


DEFAULT_AUTONOMOUS_OBJECTIVE = (
    "Determine whether this ticker is actionable, watchlist-only, avoid, or "
    "no-winner based on available evidence."
)

DEFAULT_ALLOWED_TOOLS = [
    "fetch_kpi_trends",
    "fetch_filing_section",
    "fetch_companyfacts_timeseries",
    "compare_peer_metric",
    "analyze_dilution",
    "analyze_liquidity_stress",
    "analyze_capital_structure_resolution",
    "analyze_capital_allocation",
    "fetch_current_events",
    "fetch_recent_filing_context",
]

INSURANCE_TOOL = "fetch_insurance_evidence_packet"

DEFAULT_BUDGET = AutonomousRunBudget(
    max_tool_calls=8,
    max_turns=4,
    max_cost_usd=1.25,
    timebox_seconds=None,
    max_candidates=1,
)

_FINANCIAL_PROMPT_BINDING_KWARG = "_financial_prompt_integrity_binding"

VALID_FINAL_VERDICTS = {"ACTIONABLE", "WATCHLIST_ONLY", "AVOID", "NO_WINNER"}
_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG = "_expected_integrity_scope_fingerprint"
FINAL_VERDICT_ALIASES = {
    "BUY": "ACTIONABLE",
    "SELECTED": "ACTIONABLE",
    "WATCH": "WATCHLIST_ONLY",
    "WATCHLIST": "WATCHLIST_ONLY",
    "NO_SELECTION": "NO_WINNER",
}

LLM_PROVIDER_QUOTA_EXHAUSTED = "LLM_PROVIDER_QUOTA_EXHAUSTED"

_ALPHA_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keywords": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "section_focus": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "max_chars": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "line_items": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "line_item": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "metric": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "metrics": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "years": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "max_items": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "quarters": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "material_event_window_days": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "max_documents": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    },
    "required": [
        "keywords",
        "section_focus",
        "max_chars",
        "line_items",
        "line_item",
        "metric",
        "metrics",
        "years",
        "max_items",
        "quarters",
        "material_event_window_days",
        "max_documents",
    ],
    "additionalProperties": False,
}

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "question": {"type": "string"},
                    "rationale": {"type": "string"},
                    "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "planned_tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "tool_input": _ALPHA_TOOL_INPUT_SCHEMA,
                                "rationale": {"type": "string"},
                            },
                            "required": ["tool_name", "tool_input", "rationale"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "question_id",
                    "question",
                    "rationale",
                    "priority",
                    "planned_tool_calls",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_FINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_verdict": {
            "type": "string",
            "enum": ["ACTIONABLE", "WATCHLIST_ONLY", "AVOID", "NO_WINNER"],
        },
        "selected_ticker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {
            "anyOf": [
                {"type": "string", "enum": ["HIGH", "MODERATE", "LOW"]},
                {"type": "null"},
            ]
        },
        "belief_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "ticker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "prior_belief": {"type": "string"},
                    "updated_belief": {"type": "string"},
                    "direction": {"type": "string"},
                    "confidence_after": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidence_ref_ids": {"type": "array", "items": {"type": "string"}},
                    "remaining_uncertainty": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "question_id",
                    "ticker",
                    "prior_belief",
                    "updated_belief",
                    "direction",
                    "confidence_after",
                    "summary",
                    "evidence_ref_ids",
                    "remaining_uncertainty",
                ],
                "additionalProperties": False,
            },
        },
        "candidate_decision": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "verdict": {"type": "string"},
                "confidence": {"type": "string"},
                "thesis": {"type": "string"},
                "key_risk": {"type": "string"},
                "eligible_for_selection": {"type": "boolean"},
                "selection_blockers": {"type": "array", "items": {"type": "string"}},
                "falsifiers": {"type": "array", "items": {"type": "string"}},
                "evidence_ref_ids": {"type": "array", "items": {"type": "string"}},
                "confidence_cap_reasons": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "ticker",
                "verdict",
                "confidence",
                "thesis",
                "key_risk",
                "eligible_for_selection",
                "selection_blockers",
                "falsifiers",
                "evidence_ref_ids",
                "confidence_cap_reasons",
            ],
            "additionalProperties": False,
        },
        "no_winner_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "degraded_states": {"type": "array", "items": {"type": "string"}},
        "audit_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "final_verdict",
        "selected_ticker",
        "confidence",
        "belief_updates",
        "candidate_decision",
        "no_winner_reason",
        "degraded_states",
        "audit_notes",
    ],
    "additionalProperties": False,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_run_id(ticker: str, created_at: str) -> str:
    day = created_at[:10].replace("-", "")
    return f"autonomous_{ticker.upper()}_{day}_{secrets.token_hex(3)}"


def _budget_or_default(budget: AutonomousRunBudget | None) -> AutonomousRunBudget:
    if budget is None:
        return AutonomousRunBudget.from_dict(DEFAULT_BUDGET.to_dict())
    return budget


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_jsonable(v) for v in value]
    return str(value)


def _detached_financial_integrity_value(value: Any) -> Any:
    """Return a full, JSON-safe copy of an authorized financial input."""

    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, dict):
        return {str(key): _detached_financial_integrity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detached_financial_integrity_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_detached_financial_integrity_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    return value


def _financial_integrity_run_binding(
    scope: FinancialIntegrityScope,
    *,
    scope_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": "financial_integrity_run_binding_v1",
        "context": scope.context,
        "run_as_of_date": scope.run_as_of_date,
        "scope_fingerprint": scope_fingerprint,
        "packets": [_detached_financial_integrity_value(packet) for packet in scope.packets],
        "scenarios": [
            _detached_financial_integrity_value(scenario) for scenario in scope.scenarios
        ],
    }


def _signal_packet_fingerprint(packet: TickerSignalPacket) -> str:
    """Stable identity for the exact packet supplied to a v2 child lane."""

    return v2_signal_packet_snapshot_fingerprint(canonical_v2_signal_packet_snapshot(packet))


def _financial_payload_fingerprint(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _apply_financial_integrity_result(
    scope: FinancialIntegrityScope,
    result: Any,
) -> None:
    violations = [item.to_dict() for item in getattr(result, "violations", ())]
    for item in (*scope.packets, *scope.scenarios):
        ticker = str(getattr(item, "ticker", "") or "").strip().upper()
        item_violations = [
            violation
            for violation in violations
            if not violation.get("ticker")
            or str(violation.get("ticker") or "").strip().upper() == ticker
        ]
        if hasattr(item, "financial_integrity_status"):
            item.financial_integrity_status = str(result.status)
        if hasattr(item, "financial_integrity_violations"):
            item.financial_integrity_violations = item_violations


def _require_applied_financial_integrity_scope(
    scope: FinancialIntegrityScope,
) -> Any:
    """Validate, apply the result, then bind the exact post-application bytes."""

    result = require_financial_integrity_scope(scope)
    _apply_financial_integrity_result(scope, result)
    return require_financial_integrity_scope(scope)


def _financial_integrity_result_payload(result: Any) -> dict[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {
        "status": str(getattr(result, "status", "PASS")),
        "passed": bool(getattr(result, "passed", True)),
        "violations": [
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in getattr(result, "violations", ())
        ],
    }


def _financial_scope_identity_fingerprint(scope: FinancialIntegrityScope) -> str:
    """Identify the exact quote-bearing packet/scenario bytes independent of call label."""

    return _financial_payload_fingerprint(
        {
            "run_as_of_date": scope.run_as_of_date,
            "packets": [_detached_financial_integrity_value(packet) for packet in scope.packets],
            "scenarios": [
                _detached_financial_integrity_value(scenario) for scenario in scope.scenarios
            ],
        }
    )


def _raise_financial_prompt_integrity_error(
    *,
    scope: FinancialIntegrityScope,
    code: str,
    field: str,
    expected: str,
    observed: str,
    reason: str,
) -> None:
    violation = FinancialIntegrityViolation(
        code=code,
        field=field,
        source_values={
            "expected_fingerprint": expected,
            "observed_fingerprint": observed,
        },
        expected_relationship="derived financial prompt state remains byte-for-byte unchanged",
        observed_relationship=f"{observed} != {expected}",
        reason=reason,
        terminal_status="INVALID_FINANCIAL_INPUT",
    )
    raise InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context=scope.context,
            run_as_of_date=scope.run_as_of_date,
            status="INVALID_FINANCIAL_INPUT",
            violations=(violation,),
            scope_fingerprint=_financial_scope_identity_fingerprint(scope),
            packet_count=len(scope.packets),
            scenario_count=len(scope.scenarios),
        )
    )


@dataclass(frozen=True)
class _FrozenFinancialPromptState:
    """One explicitly produced prompt-state epoch bound to canonical quote identities."""

    scope: FinancialIntegrityScope
    expected_scope_fingerprint: str
    expected_financial_identity_fingerprint: str
    expected_state_fingerprint: str
    state_getter: Callable[[], Any]
    scope_revalidator: Callable[[FinancialIntegrityScope, str], Any]

    def require_current(self) -> None:
        self.scope_revalidator(self.scope, self.expected_scope_fingerprint)
        observed_financial_identity = _financial_scope_identity_fingerprint(self.scope)
        if observed_financial_identity != self.expected_financial_identity_fingerprint:
            _raise_financial_prompt_integrity_error(
                scope=self.scope,
                code="BOUND_FINANCIAL_PROMPT_QUOTE_IDENTITY_MUTATED",
                field="financial_prompt_quote_identity",
                expected=self.expected_financial_identity_fingerprint,
                observed=observed_financial_identity,
                reason=(
                    "Canonical packet or scenario quote identities changed after the "
                    "derived provider prompt state was frozen."
                ),
            )
        observed_state = _financial_payload_fingerprint(
            _detached_financial_integrity_value(self.state_getter())
        )
        if observed_state != self.expected_state_fingerprint:
            _raise_financial_prompt_integrity_error(
                scope=self.scope,
                code="BOUND_FINANCIAL_PROMPT_STATE_MUTATED",
                field="financial_prompt_state",
                expected=self.expected_state_fingerprint,
                observed=observed_state,
                reason=(
                    "Derived research questions, tool results, evidence, or belief "
                    "updates changed after their prompt state was frozen."
                ),
            )

    def bind_request(
        self,
        provider: Any,
        request: Mapping[str, Any],
    ) -> _BoundFinancialPromptCall:
        self.require_current()
        return _BoundFinancialPromptCall(
            frozen_state=self,
            expected_request_fingerprint=_financial_provider_request_fingerprint(
                provider,
                request,
            ),
        )


@dataclass(frozen=True)
class _BoundFinancialPromptCall:
    frozen_state: _FrozenFinancialPromptState
    expected_request_fingerprint: str

    def require_current(
        self,
        provider: Any,
        request: Mapping[str, Any],
    ) -> None:
        self.frozen_state.require_current()
        observed_request = _financial_provider_request_fingerprint(provider, request)
        if observed_request != self.expected_request_fingerprint:
            _raise_financial_prompt_integrity_error(
                scope=self.frozen_state.scope,
                code="BOUND_FINANCIAL_PROVIDER_REQUEST_MUTATED",
                field="financial_provider_request",
                expected=self.expected_request_fingerprint,
                observed=observed_request,
                reason=(
                    "Provider identity, model, prompt, schema, tools, limits, or another "
                    "transport-semantic request field changed after authorization."
                ),
            )


def _freeze_financial_prompt_state(
    *,
    scope: FinancialIntegrityScope,
    expected_scope_fingerprint: str,
    state_getter: Callable[[], Any],
    scope_revalidator: Callable[[FinancialIntegrityScope, str], Any] | None = None,
) -> _FrozenFinancialPromptState:
    """Freeze a legitimate state transition before any later paid call can see it."""

    if scope_revalidator is None:

        def scope_revalidator(
            current_scope: FinancialIntegrityScope,
            expected_fingerprint: str,
        ) -> Any:
            return require_unchanged_financial_integrity_scope(
                current_scope,
                expected_scope_fingerprint=expected_fingerprint,
            )

    scope_revalidator(scope, expected_scope_fingerprint)
    financial_identity = _financial_scope_identity_fingerprint(scope)
    state_fingerprint = _financial_payload_fingerprint(
        _detached_financial_integrity_value(state_getter())
    )
    return _FrozenFinancialPromptState(
        scope=scope,
        expected_scope_fingerprint=expected_scope_fingerprint,
        expected_financial_identity_fingerprint=financial_identity,
        expected_state_fingerprint=state_fingerprint,
        state_getter=state_getter,
        scope_revalidator=scope_revalidator,
    )


def _json_preview(value: Any, limit: int = 1200) -> str:
    text = json.dumps(_jsonable(value), sort_keys=True, default=str)
    return text[:limit]


def _paths_dict_from_analysis_paths(paths: Any) -> dict[str, str | None]:
    return {
        "analysis_evidence_bundle": str(getattr(paths, "bundle_json", "") or "") or None,
        "analysis_report_json": str(getattr(paths, "report_json", "") or "") or None,
        "analysis_report_markdown": str(getattr(paths, "report_markdown", "") or "") or None,
    }


def _cached_analysis_paths(ticker: str, as_of_date: str) -> dict[str, str | None]:
    from app.analyst.output_store import latest_analysis_output_path

    return {
        "analysis_evidence_bundle": str(
            latest_analysis_output_path(ticker, "analysis_evidence_bundle", as_of_date=as_of_date)
            or ""
        )
        or None,
        "analysis_report_json": str(
            latest_analysis_output_path(ticker, "analysis_report", as_of_date=as_of_date) or ""
        )
        or None,
        "analysis_report_markdown": str(
            latest_analysis_output_path(ticker, "analysis_report_markdown", as_of_date=as_of_date)
            or ""
        )
        or None,
    }


def _compact_findings(items: Any, limit: int = 3) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in list(items or [])[:limit]:
        compact.append(
            {
                "claim": str(getattr(item, "claim", "") or ""),
                "direction": getattr(item, "direction", None),
                "severity": getattr(item, "severity", None),
                "citation_ids": list(getattr(item, "citation_ids", []) or []),
            }
        )
    return compact


def _compact_open_questions(items: Any, limit: int = 5) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in list(items or [])[:limit]:
        compact.append(
            {
                "question": str(getattr(item, "question", "") or ""),
                "importance": str(getattr(item, "importance", "") or ""),
                "next_step": getattr(item, "next_step", None),
            }
        )
    return compact


def _compact_citations(items: Any, limit: int = 8) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in list(items or [])[:limit]:
        compact.append(
            {
                "citation_id": str(getattr(item, "citation_id", "") or ""),
                "source_type": str(getattr(item, "source_type", "") or ""),
                "source_label": str(getattr(item, "source_label", "") or ""),
                "source_date": getattr(item, "source_date", None),
                "source_url": getattr(item, "source_url", None),
                "section": getattr(item, "section", None),
                "excerpt": str(getattr(item, "excerpt", "") or "")[:500],
            }
        )
    return compact


def _packet_identity(packet: TickerSignalPacket) -> dict[str, Any]:
    raw = packet.raw_valuation if isinstance(packet.raw_valuation, dict) else {}
    insurance_packet = packet.insurance_packet if isinstance(packet.insurance_packet, dict) else {}
    universe_identity: dict[str, Any] = {}
    try:
        from app.db import get_db

        with get_db() as conn:
            row = conn.execute(
                """
                SELECT cik, name, homepage_url
                FROM universe_members
                WHERE ticker = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (packet.ticker.upper(),),
            ).fetchone()
        if row:
            universe_identity = dict(row)
    except Exception:
        universe_identity = {}
    return {
        "ticker": packet.ticker.upper(),
        "company_name": raw.get("company_name")
        or raw.get("name")
        or insurance_packet.get("company_name")
        or universe_identity.get("name"),
        "cik": universe_identity.get("cik"),
        "homepage_url": universe_identity.get("homepage_url"),
        "sector": raw.get("sector") or raw.get("sector_id") or packet.issuer_type,
        "issuer_type": packet.issuer_type,
        "security_type": packet.security_type,
        "insurance_subtype": packet.insurance_subtype,
        "model_status": packet.model_status,
    }


def _analysis_context_from_report(
    *,
    report: Any,
    packet: TickerSignalPacket,
    paths: dict[str, str | None],
    source: str,
) -> dict[str, Any]:
    valuation = getattr(report, "valuation", None)
    research_quality = getattr(report, "research_quality", None)
    return {
        "source": source,
        "company_identity": _packet_identity(packet),
        "ticker": str(getattr(report, "ticker", packet.ticker) or packet.ticker).upper(),
        "as_of_date": str(getattr(report, "as_of_date", "") or ""),
        "verdict": str(getattr(report, "verdict", "") or ""),
        "confidence_label": str(getattr(report, "confidence_label", "") or ""),
        "confidence_score": getattr(report, "confidence_score", None),
        "thesis_summary": str(getattr(report, "thesis_summary", "") or ""),
        "latest_evidence_date": getattr(report, "latest_evidence_date", None),
        "latest_evidence_source_type": getattr(report, "latest_evidence_source_type", None),
        "valuation": {
            "price": getattr(valuation, "price", None),
            "base_case_value": getattr(valuation, "base_case_value", None),
            "bear_case_value": getattr(valuation, "bear_case_value", None),
            "bull_case_value": getattr(valuation, "bull_case_value", None),
            "margin_of_safety": getattr(valuation, "margin_of_safety", None),
        },
        "research_quality": (
            {
                "overall_score": getattr(research_quality, "overall_score", None),
                "coverage_score": getattr(research_quality, "coverage_score", None),
                "freshness_score": getattr(research_quality, "freshness_score", None),
                "gap_score": getattr(research_quality, "gap_score", None),
                "evidence_count": getattr(research_quality, "evidence_count", None),
            }
            if research_quality is not None
            else None
        ),
        "positives": _compact_findings(getattr(report, "positives", [])),
        "risks": _compact_findings(getattr(report, "risks", [])),
        "recent_events": _compact_findings(getattr(report, "recent_event_impacts", [])),
        "open_questions": _compact_open_questions(getattr(report, "open_questions", [])),
        "falsifiers": [
            {
                "description": str(getattr(item, "description", "") or ""),
                "trigger_type": str(getattr(item, "trigger_type", "") or ""),
                "monitoring_hint": getattr(item, "monitoring_hint", None),
            }
            for item in list(getattr(report, "falsifiers", []) or [])[:5]
        ],
        "citations": _compact_citations(getattr(report, "citations", [])),
        "sources_used": list(getattr(report, "sources_used", []) or []),
        "warnings": list(getattr(report, "warnings", []) or []),
        "paths": paths,
    }


def _load_or_refresh_analyst_context(
    *,
    ticker: str,
    as_of_date: str,
    packet: TickerSignalPacket,
    years: int,
    quarters: int,
    skip_analysis_refresh: bool,
) -> dict[str, Any]:
    upper = ticker.upper()
    if skip_analysis_refresh:
        from app.analyst.output_store import latest_analysis_report

        cached_report = latest_analysis_report(upper, as_of_date=as_of_date)
        if cached_report is None:
            raise RuntimeError("no cached analyst report was available")
        return _analysis_context_from_report(
            report=cached_report,
            packet=packet,
            paths=_cached_analysis_paths(upper, as_of_date),
            source="cached_analysis_report",
        )

    from app.analyst.materializer import materialize_analysis_outputs_from_research
    from app.research.deep_research import run_deep_research

    research_report = run_deep_research(
        upper,
        as_of_date=as_of_date,
        years=years,
        quarters=quarters,
    )
    materialized = materialize_analysis_outputs_from_research(
        research_report,
        years=years,
        quarters=quarters,
    )
    return _analysis_context_from_report(
        report=materialized.report,
        packet=packet,
        paths=_paths_dict_from_analysis_paths(materialized.paths),
        source="fresh_deep_research",
    )


def _analyst_context_summary(context: dict[str, Any]) -> str:
    ticker = str(context.get("ticker") or "").upper()
    verdict = str(context.get("verdict") or "UNKNOWN")
    confidence = str(context.get("confidence_label") or "UNKNOWN")
    thesis = str(context.get("thesis_summary") or "").strip()
    latest = context.get("latest_evidence_date") or "unknown evidence date"
    if len(thesis) > 260:
        thesis = thesis[:257].rstrip() + "..."
    return (
        f"Analyst report context for {ticker}: verdict {verdict}, confidence "
        f"{confidence}, latest evidence {latest}. {thesis}"
    ).strip()


def _build_analyst_context_evidence(
    *,
    ticker: str,
    as_of_date: str,
    packet: TickerSignalPacket,
    years: int,
    quarters: int,
    skip_analysis_refresh: bool,
) -> tuple[list[EvidenceReference], dict[str, Any] | None, list[str], list[str]]:
    try:
        context = _load_or_refresh_analyst_context(
            ticker=ticker,
            as_of_date=as_of_date,
            packet=packet,
            years=years,
            quarters=quarters,
            skip_analysis_refresh=skip_analysis_refresh,
        )
    except Exception as exc:
        return (
            [],
            None,
            ["ANALYST_CONTEXT_UNAVAILABLE"],
            [
                f"Analyst context unavailable; autonomous run continued without seeded analyst report: {exc}"
            ],
        )

    context_ticker = str(context.get("ticker") or "").strip().upper()
    context_as_of = str(context.get("as_of_date") or "").strip()[:10]
    valuation = context.get("valuation")
    context_price = valuation.get("price") if isinstance(valuation, dict) else None
    packet_price = packet.current_price
    quote_matches = bool(
        isinstance(context_price, (int, float))
        and not isinstance(context_price, bool)
        and isinstance(packet_price, (int, float))
        and not isinstance(packet_price, bool)
        and math.isfinite(float(context_price))
        and math.isfinite(float(packet_price))
        and math.isclose(
            float(context_price),
            float(packet_price),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )
    try:
        context_not_future = date.fromisoformat(context_as_of) <= date.fromisoformat(
            str(as_of_date)[:10]
        )
    except ValueError:
        context_not_future = False
    if context_ticker != ticker.upper() or not context_not_future or not quote_matches:
        return (
            [],
            None,
            ["ANALYST_CONTEXT_FINANCIAL_LINEAGE_MISMATCH"],
            [
                "Cached analyst context was excluded because its ticker, as-of date, or price "
                "did not match the gated autonomous packet."
            ],
        )

    evidence = EvidenceReference(
        evidence_id="E1",
        source_type="analysis_report",
        source_label="analyst_context",
        summary=_analyst_context_summary(context),
        ticker=ticker.upper(),
        source_date=context.get("latest_evidence_date") or context.get("as_of_date") or as_of_date,
        excerpt=json.dumps(_jsonable(context), sort_keys=True, default=str),
        confidence="MODERATE",
    )
    return (
        [evidence],
        context,
        [],
        ["Seeded autonomous run with analyst report context before provider planning."],
    )


def _parse_provider_json(result: Any) -> dict[str, Any]:
    raw = getattr(result, "json_text", result)
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw))


def _confidence_for_tool_output(output: dict[str, Any]) -> str:
    if not isinstance(output, dict):
        return "LOW"
    if output.get("usable_for_decision") is False:
        return "LOW"
    status = str(output.get("status") or "").lower()
    if status == "ok":
        return "MODERATE"
    return "LOW"


def _is_quota_exhaustion(exc: BaseException) -> bool:
    # Only genuine quota exhaustion should force the expensive Anthropic
    # fallback. Transient 429s ("rate limit") and the sticky process-global
    # breaker are handled by the retry guard; matching them here would force
    # an unnecessary, expensive provider switch. The breaker message that
    # signals real exhaustion still contains "insufficient_quota".
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "insufficient_quota" in text
        or "insufficient quota" in text
        # Our own canonical quota marker, re-raised when the fallback is
        # unavailable/failed, must still classify as exhaustion.
        or LLM_PROVIDER_QUOTA_EXHAUSTED.lower() in text
    )


def _provider_name(provider: Any) -> str:
    return str(getattr(provider, "provider_name", provider.__class__.__name__) or "unknown").lower()


def _estimated_provider_call_cost_usd(provider: Any, kwargs: dict[str, Any]) -> float:
    estimate = failed_provider_usage_meta(
        provider=provider,
        prompt=str(kwargs.get("prompt") or ""),
        schema_name=str(kwargs.get("schema_name") or "structured_output"),
        estimated_output_tokens=max(
            1,
            int(kwargs.get("max_output_tokens") or 1000),
        ),
        error=RuntimeError("pre-call cost reservation"),
    )
    return max(0.0, float(estimate.get("cost_estimate_usd") or 0.0))


def _reserve_provider_call_cost(provider: Any, kwargs: dict[str, Any]) -> Any:
    context = current_cost_context()
    if context is None:
        return None
    return context.reserve_call(
        provider=_provider_name(provider),
        schema_name=str(kwargs.get("schema_name") or "structured_output"),
        estimated_cost_usd=_estimated_provider_call_cost_usd(provider, kwargs),
    )


def _physical_usage_cost_usd(records: list[dict[str, Any]]) -> float:
    return sum(
        max(0.0, float(record.get("cost_estimate_usd") or 0.0))
        for record in records
        if isinstance(record, dict)
    )


def _settle_provider_call_cost(
    reservation: Any,
    records: list[dict[str, Any]],
    *,
    failed: bool,
) -> None:
    context = current_cost_context()
    if context is None:
        return
    if failed and not context.strict_first_call:
        context.cancel_call(reservation)
        return
    physical_cost = _physical_usage_cost_usd(records)
    if failed and context.strict_first_call:
        physical_cost = max(
            physical_cost,
            max(
                0.0,
                float(getattr(reservation, "estimated_cost_usd", 0.0) or 0.0),
            ),
        )
    context.complete_call(reservation, actual_cost_usd=physical_cost)


def _adapt_provider_synthesize_kwargs(
    provider: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Adapt legacy provider signatures before the only physical call.

    A ``TypeError`` raised by the provider is never evidence that transport did
    not occur, so compatibility must be resolved by capability/signature
    inspection before invocation.
    """

    adapted = dict(kwargs)
    if "max_output_tokens" not in adapted:
        return adapted
    capability = getattr(provider, "supports_max_output_tokens", None)
    if capability is True:
        return adapted
    if capability is False:
        adapted.pop("max_output_tokens", None)
        return adapted
    try:
        parameters = inspect.signature(provider.synthesize_json).parameters
    except (TypeError, ValueError):
        # Unknown signatures get one normal call. Never probe by calling twice.
        return adapted
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    parameter = parameters.get("max_output_tokens")
    accepts_keyword = parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if not accepts_kwargs and not accepts_keyword:
        adapted.pop("max_output_tokens", None)
    return adapted


def _financial_provider_request_envelope(
    provider: Any,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Return every output-semantic field used by one physical provider call."""

    adapted_request = _adapt_provider_synthesize_kwargs(provider, dict(request))
    provider_name = _provider_name(provider)
    cfg = getattr(provider, "cfg", None)
    explicit_model = str(adapted_request.get("model") or "").strip()
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
    except (TypeError, ValueError):
        parameters = {}
    for name, parameter in parameters.items():
        if name == "self" or parameter.default is inspect.Parameter.empty:
            continue
        signature_defaults[name] = parameter.default

    effective_output_limit = adapted_request.get("max_output_tokens")
    if effective_output_limit is None:
        effective_output_limit = configured_output_limit
    return {
        "provider": {
            "name": provider_name,
            "class": (
                f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
                if provider is not None
                else "none"
            ),
            "model": explicit_model or bound_model or configured_model,
            "base_url": str(getattr(provider, "base_url", "") or ""),
            "request_timeout_seconds": configured_timeout,
            "handles_retry_guard": bool(getattr(provider, "_handles_retry_guard", False)),
        },
        "effective_defaults": {
            "max_output_tokens": effective_output_limit,
            "signature": signature_defaults,
        },
        "request": adapted_request,
    }


def _financial_provider_request_fingerprint(
    provider: Any,
    request: Mapping[str, Any],
) -> str:
    return _financial_payload_fingerprint(
        _detached_financial_integrity_value(_financial_provider_request_envelope(provider, request))
    )


def _rebind_financial_provider_request(
    kwargs: Mapping[str, Any],
    *,
    current_provider: Any,
    next_provider: Any,
) -> dict[str, Any]:
    """Authorize a provider fallback without accepting intervening request drift."""

    rebound = dict(kwargs)
    binding = rebound.get(_FINANCIAL_PROMPT_BINDING_KWARG)
    if binding is None:
        return rebound
    if not isinstance(binding, _BoundFinancialPromptCall):
        raise TypeError("financial prompt integrity binding has an invalid type")
    request = dict(rebound)
    request.pop(_FINANCIAL_PROMPT_BINDING_KWARG, None)
    request.pop(_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG, None)
    request.pop("integrity_scope", None)
    binding.require_current(current_provider, request)
    rebound[_FINANCIAL_PROMPT_BINDING_KWARG] = binding.frozen_state.bind_request(
        next_provider,
        request,
    )
    return rebound


def _call_provider_json(provider: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Call provider JSON synthesis while tolerating older test doubles."""

    kwargs = dict(kwargs)
    prompt_binding = kwargs.pop(_FINANCIAL_PROMPT_BINDING_KWARG, None)
    if prompt_binding is not None and not isinstance(prompt_binding, _BoundFinancialPromptCall):
        raise TypeError("financial prompt integrity binding has an invalid type")
    expected_scope_fingerprint = str(
        kwargs.pop(_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG, "") or ""
    ).strip()
    integrity_scope = kwargs.pop("integrity_scope", None)
    if integrity_scope is None:
        integrity_scope = FinancialIntegrityScope(
            context=str(kwargs.get("schema_name") or "provider_call_missing_scope"),
            run_as_of_date="",
            packets=(),
        )
    try:
        gate_result = _require_applied_financial_integrity_scope(integrity_scope)
    except InvalidFinancialInputError as exc:
        _apply_financial_integrity_result(integrity_scope, exc.result)
        raise
    if expected_scope_fingerprint:
        gate_result = require_unchanged_financial_integrity_scope(
            integrity_scope,
            expected_scope_fingerprint=expected_scope_fingerprint,
        )
        _apply_financial_integrity_result(integrity_scope, gate_result)
    else:
        expected_scope_fingerprint = gate_result.scope_fingerprint

    call_kwargs = _adapt_provider_synthesize_kwargs(provider, kwargs)

    def require_exact_scope(_attempt: dict[str, Any]) -> None:
        current_result = require_unchanged_financial_integrity_scope(
            integrity_scope,
            expected_scope_fingerprint=expected_scope_fingerprint,
        )
        _apply_financial_integrity_result(integrity_scope, current_result)
        if prompt_binding is not None:
            prompt_binding.require_current(provider, call_kwargs)

    require_exact_scope({})
    reservation = _reserve_provider_call_cost(provider, kwargs)
    failed_attempts: list[dict[str, Any]] = []

    def observe_failed_attempt(event: dict[str, Any]) -> None:
        error = event.get("error")
        if not isinstance(error, BaseException):
            error = RuntimeError(str(error or "physical provider attempt failed"))
        failure_meta = failed_provider_usage_meta(
            provider=provider,
            prompt=str(kwargs.get("prompt") or ""),
            schema_name=str(kwargs.get("schema_name") or "structured_output"),
            estimated_output_tokens=max(1, int(kwargs.get("max_output_tokens") or 1000)),
            error=error,
        )
        failure_meta.update(
            {
                "physical_attempt": int(event.get("attempt") or 1),
                "retryable": bool(event.get("retryable")),
                "will_retry": bool(event.get("will_retry")),
            }
        )
        failed_attempts.append(failure_meta)
        record_provider_usage(failure_meta)

    try:
        with (
            llm_attempt_observer(observe_failed_attempt),
            llm_physical_attempt_guard(require_exact_scope),
        ):
            require_llm_physical_attempt_authorization(
                provider=_provider_name(provider),
                schema_name=str(call_kwargs.get("schema_name") or "structured_output"),
                attempt=1,
            )
            result = provider.synthesize_json(**call_kwargs)
    except Exception as exc:
        successful_attempts = provider_usage_records_from_exception(
            provider=provider,
            error=exc,
            prompt=str(kwargs.get("prompt") or ""),
            schema_name=str(kwargs.get("schema_name") or "structured_output"),
        )
        for usage in successful_attempts:
            record_provider_usage(usage)
        if not failed_attempts and not successful_attempts:
            failure_meta = failed_provider_usage_meta(
                provider=provider,
                prompt=str(kwargs.get("prompt") or ""),
                schema_name=str(kwargs.get("schema_name") or "structured_output"),
                estimated_output_tokens=max(1, int(kwargs.get("max_output_tokens") or 1000)),
                error=exc,
            )
            failed_attempts.append(failure_meta)
            record_provider_usage(failure_meta)
        attach_provider_usage_to_exception(
            exc,
            [*failed_attempts, *successful_attempts],
        )
        _settle_provider_call_cost(
            reservation,
            [*failed_attempts, *successful_attempts],
            failed=True,
        )
        try:
            require_exact_scope({})
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
        prompt=str(kwargs.get("prompt") or ""),
        schema_name=str(kwargs.get("schema_name") or "structured_output"),
    )
    for usage in successful_attempts:
        record_provider_usage(usage)
    _settle_provider_call_cost(
        reservation,
        [*failed_attempts, *successful_attempts],
        failed=False,
    )
    try:
        require_exact_scope({})
    except InvalidFinancialInputError as exc:
        attach_provider_usage_to_exception(
            exc,
            [*failed_attempts, *successful_attempts],
        )
        raise
    try:
        payload = _parse_provider_json(result)
    except Exception as exc:
        attach_provider_usage_to_exception(exc, successful_attempts)
        raise
    try:
        require_exact_scope({})
    except InvalidFinancialInputError as exc:
        attach_provider_usage_to_exception(exc, successful_attempts)
        raise
    return payload


def _synthesize_provider_json(provider: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs = dict(kwargs)
    integrity_scope = kwargs.get("integrity_scope")
    if integrity_scope is not None and not kwargs.get(_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG):
        gate_result = _require_applied_financial_integrity_scope(integrity_scope)
        kwargs[_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG] = gate_result.scope_fingerprint
    try:
        return _call_provider_json(provider, kwargs)
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        cost_context = current_cost_context()
        if cost_context is not None and cost_context.strict_first_call:
            raise
        if not _is_quota_exhaustion(exc) or _provider_name(provider) == "anthropic":
            raise
        fallback = get_anthropic_provider()
        if fallback is None or not _provider_enabled(fallback):
            raise RuntimeError(
                f"{LLM_PROVIDER_QUOTA_EXHAUSTED}: primary provider {_provider_name(provider)} "
                "exhausted quota and Anthropic fallback is unavailable."
            ) from exc
        fallback_kwargs = _rebind_financial_provider_request(
            kwargs,
            current_provider=provider,
            next_provider=fallback,
        )
        try:
            return _call_provider_json(fallback, fallback_kwargs)
        except InvalidFinancialInputError:
            raise
        except Exception as fallback_exc:
            raise RuntimeError(
                f"{LLM_PROVIDER_QUOTA_EXHAUSTED}: primary provider {_provider_name(provider)} "
                f"exhausted quota and Anthropic fallback failed: {fallback_exc}"
            ) from fallback_exc


def _packet_is_usable(packet: TickerSignalPacket) -> bool:
    has_value = any(
        value is not None
        for value in (
            packet.current_price,
            packet.dcf_value,
            packet.epv_value,
            packet.graham_value,
            packet.ncav_value,
            packet.insurance_value,
        )
    )
    return bool(
        has_value
        or packet.raw_valuation
        or packet.insurance_packet
        or packet.gate_verdict
        or packet.research_report
    )


def _assemble_deterministic_autonomous_signal_packet(
    ticker: str,
    *,
    as_of_date: str,
) -> TickerSignalPacket:
    """Build the pre-gate packet without a filing-risk provider side lane."""

    kwargs = {
        "filing_risk_use_llm": False,
        "as_of_date": as_of_date,
        "pipeline_version": "v1",
    }
    # Preserve narrow legacy-test doubles that predate keyword-aware assembly
    # without weakening the real production call contract.
    parameters = inspect.signature(assemble_signal_packet).parameters.values()
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    named = {parameter.name for parameter in parameters}
    if accepts_kwargs or set(kwargs).issubset(named):
        return assemble_signal_packet(ticker, **kwargs)
    return assemble_signal_packet(ticker)


def _allowed_tools_for_packet(
    packet: TickerSignalPacket, allowed_tools: list[str] | None
) -> list[str]:
    if allowed_tools is not None:
        return list(dict.fromkeys(str(tool).strip() for tool in allowed_tools if str(tool).strip()))
    tools = list(DEFAULT_ALLOWED_TOOLS)
    if isinstance(packet.insurance_packet, dict) and packet.insurance_packet:
        tools.append(INSURANCE_TOOL)
    return tools


def _sector_for_packet(packet: TickerSignalPacket) -> str:
    raw = packet.raw_valuation if isinstance(packet.raw_valuation, dict) else {}
    sector = raw.get("sector") or raw.get("sector_id") or packet.issuer_type
    return str(sector or "unknown")


def _empty_artifact(
    *,
    request: AutonomousRunRequest,
    started_at: str,
    degraded_state: str,
    reason: str,
    questions: list[ResearchQuestion] | None = None,
    tool_calls: list[ToolCallRecord] | None = None,
    evidence: list[EvidenceReference] | None = None,
    audit_notes: list[str] | None = None,
    additional_degraded_states: list[str] | None = None,
) -> AutonomousRunArtifact:
    degraded_states = list(dict.fromkeys([degraded_state, *(additional_degraded_states or [])]))
    return AutonomousRunArtifact(
        request=request,
        status="FAILED",
        started_at=started_at,
        completed_at=_utc_now_iso(),
        final_verdict="NO_WINNER",
        selected_ticker=None,
        confidence=None,
        questions=questions or [],
        tool_calls=tool_calls or [],
        evidence=evidence or [],
        no_winner_reason=reason,
        degraded_states=degraded_states,
        audit_notes=audit_notes or [reason],
    )


def _build_request(
    *,
    ticker: str,
    objective: str,
    as_of_date: str,
    budget: AutonomousRunBudget,
    allowed_tools: list[str],
    created_at: str,
) -> AutonomousRunRequest:
    return AutonomousRunRequest(
        run_id=_default_run_id(ticker, created_at),
        objective=objective,
        as_of_date=as_of_date,
        created_at=created_at,
        candidate_scope={"mode": "single_candidate", "tickers": [ticker.upper()]},
        allowed_tools=allowed_tools,
        budget=budget,
        stop_rules=["stop_if_budget_exhausted", "stop_if_no_actionable_decision"],
        user_constraints=["use_allowed_alpha_tools_only"],
    )


def _planner_prompt(
    *,
    ticker: str,
    objective: str,
    packet: TickerSignalPacket,
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    analyst_context: dict[str, Any] | None,
    financial_context: dict[str, Any] | None = None,
) -> str:
    return (
        "You are planning a bounded autonomous analyst run for one ticker.\n"
        "Choose 2-4 decision-relevant research questions and the smallest set "
        "of tool calls needed to answer them.\n"
        "Treat analyst_context as seeded research evidence, but use follow-up "
        "tools to close gaps that could change the verdict.\n"
        "Use only allowed tools. If evidence is likely insufficient, ask the "
        "question that would reveal that rather than forcing optimism.\n\n"
        f"Ticker: {ticker.upper()}\n"
        f"Objective: {objective}\n"
        f"Allowed tools: {allowed_tools}\n"
        f"Budget: {budget.to_dict()}\n"
        f"Initial packet summary: {json.dumps(_jsonable(packet.to_summary_dict()), sort_keys=True)}\n"
        f"Canonical financial context: {json.dumps(_jsonable(financial_context or {}), sort_keys=True)}\n"
        f"Analyst context: {json.dumps(_jsonable(analyst_context or {}), sort_keys=True)}"
    )


def _final_prompt(
    *,
    ticker: str,
    objective: str,
    packet: TickerSignalPacket,
    questions: list[ResearchQuestion],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    analyst_context: dict[str, Any] | None,
    financial_context: dict[str, Any] | None = None,
) -> str:
    payload = {
        "ticker": ticker.upper(),
        "objective": objective,
        "initial_packet_summary": packet.to_summary_dict(),
        "canonical_financial_context": financial_context or {},
        "analyst_context": analyst_context or {},
        "questions": [item.to_dict() for item in questions],
        "tool_calls": [item.to_dict() for item in tool_calls],
        "evidence": [item.to_dict() for item in evidence],
    }
    return (
        "You are finalizing a single-candidate autonomous analyst run.\n"
        "Return an audited verdict. Use ACTIONABLE only when the evidence "
        "supports action and there are no binding model-fit or data blockers. "
        "Use WATCHLIST_ONLY when the idea is interesting but incomplete, AVOID "
        "when evidence is negative or invalid, and NO_WINNER when the run cannot "
        "support a decision.\n\n"
        f"Run evidence: {json.dumps(_jsonable(payload), sort_keys=True)}"
    )


def _followup_prompt(
    *,
    ticker: str,
    objective: str,
    packet: TickerSignalPacket,
    questions: list[ResearchQuestion],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    analyst_context: dict[str, Any] | None,
    seen_tool_call_keys: set[str],
    financial_context: dict[str, Any] | None = None,
) -> str:
    payload = {
        "ticker": ticker.upper(),
        "objective": objective,
        "initial_packet_summary": packet.to_summary_dict(),
        "canonical_financial_context": financial_context or {},
        "analyst_context": analyst_context or {},
        "questions_so_far": [item.to_dict() for item in questions],
        "tool_calls_so_far": [item.to_dict() for item in tool_calls],
        "evidence_so_far": [item.to_dict() for item in evidence],
        "seen_tool_call_keys": sorted(seen_tool_call_keys),
    }
    return (
        "You are in a middle turn of a bounded autonomous analyst run.\n"
        "Return only follow-up research questions and tool calls that could "
        "change the final verdict. Do not repeat prior tool calls. If the "
        "current evidence is enough, return a question with no planned calls.\n\n"
        f"Allowed tools: {allowed_tools}\n"
        f"Budget: {budget.to_dict()}\n"
        f"Run state: {json.dumps(_jsonable(payload), sort_keys=True)}"
    )


def _questions_from_plan(
    ticker: str, plan: dict[str, Any]
) -> tuple[list[ResearchQuestion], list[dict[str, Any]]]:
    questions: list[ResearchQuestion] = []
    planned_calls: list[dict[str, Any]] = []
    raw_questions = plan.get("questions") if isinstance(plan.get("questions"), list) else []
    for idx, item in enumerate(raw_questions[:4], start=1):
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or f"Q{idx}")
        raw_calls = (
            item.get("planned_tool_calls")
            if isinstance(item.get("planned_tool_calls"), list)
            else []
        )
        selected_tools = [
            str(call.get("tool_name"))
            for call in raw_calls
            if isinstance(call, dict) and call.get("tool_name")
        ]
        question = ResearchQuestion(
            question_id=question_id,
            question=str(item.get("question") or "Unspecified research question"),
            rationale=str(item.get("rationale") or "Provider did not supply a rationale."),
            priority=str(item.get("priority") or "MEDIUM"),
            status="PLANNED",
            target_tickers=[ticker.upper()],
            selected_tools=selected_tools,
        )
        questions.append(question)
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            planned_calls.append(
                {
                    "question_id": question_id,
                    "tool_name": str(call.get("tool_name") or ""),
                    "tool_input": call.get("tool_input")
                    if isinstance(call.get("tool_input"), dict)
                    else {},
                    "rationale": str(call.get("rationale") or question.rationale),
                }
            )
    return questions, planned_calls


def _mark_question_statuses(
    questions: list[ResearchQuestion], tool_calls: list[ToolCallRecord]
) -> None:
    by_question: dict[str, list[str]] = {}
    for call in tool_calls:
        if call.question_id:
            by_question.setdefault(call.question_id, []).append(call.status)
    for question in questions:
        statuses = by_question.get(question.question_id, [])
        if not statuses:
            question.status = "OPEN"
        elif any(status == "OK" for status in statuses):
            question.status = "ANSWERED"
        elif all(status.startswith("SKIPPED") for status in statuses):
            question.status = "SKIPPED"
        else:
            question.status = "PARTIAL"


def _evidence_from_tool_output(
    *,
    ticker: str,
    call: ToolCallRecord,
    output: dict[str, Any],
    evidence_index: int,
) -> EvidenceReference:
    status = str(output.get("status") or "unknown") if isinstance(output, dict) else "unknown"
    summary = f"{call.tool_name} returned status {status} for {ticker.upper()}."
    if isinstance(output, dict) and output.get("summary"):
        summary = str(output["summary"])
    return EvidenceReference(
        evidence_id=f"E{evidence_index}",
        source_type="tool_output",
        source_label=call.tool_name,
        summary=summary,
        ticker=ticker.upper(),
        excerpt=_json_preview(output, limit=2000),
        tool_call_id=call.call_id,
        confidence=_confidence_for_tool_output(output),
    )


def _tool_call_key(tool_name: str, tool_input: dict[str, Any]) -> str:
    return f"{tool_name}|{json.dumps(_jsonable(tool_input), sort_keys=True, default=str)}"


def _execute_planned_calls(
    *,
    ticker: str,
    packet: TickerSignalPacket,
    as_of_date: str,
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    planned_calls: list[dict[str, Any]],
    evidence_start_index: int = 1,
    call_start_index: int = 1,
    executed_count: int = 0,
    seen_tool_call_keys: set[str] | None = None,
) -> tuple[list[ToolCallRecord], list[EvidenceReference], list[str], int, bool, set[str]]:
    ctx = AlphaToolContext(
        sector=_sector_for_packet(packet),
        ticker=ticker.upper(),
        packet=packet,
        as_of_date=as_of_date,
    )
    allowed = set(allowed_tools)
    tool_records: list[ToolCallRecord] = []
    evidence: list[EvidenceReference] = []
    degraded_states: list[str] = []
    budget_exhausted = False
    seen_keys = set(seen_tool_call_keys or set())

    for idx, planned in enumerate(planned_calls, start=1):
        tool_name = str(planned.get("tool_name") or "").strip()
        raw_tool_input = (
            planned.get("tool_input") if isinstance(planned.get("tool_input"), dict) else {}
        )
        call = ToolCallRecord(
            call_id=f"TC{call_start_index + idx - 1}",
            tool_name=tool_name,
            tool_input=raw_tool_input,
            rationale=str(planned.get("rationale") or ""),
            status="PLANNED",
            question_id=planned.get("question_id"),
        )
        if tool_name not in allowed:
            call.status = "SKIPPED_DISALLOWED_TOOL"
            call.error = f"Tool {tool_name} is not in the allowed toolset."
            degraded_states.append("DISALLOWED_TOOL_SKIPPED")
            tool_records.append(call)
            continue

        repair = repair_alpha_tool_input(tool_name, raw_tool_input)
        tool_input = repair.tool_input
        call.tool_input = tool_input
        if repair.repair_notes:
            call.rationale = (
                f"{call.rationale} Tool input guardrail: {'; '.join(repair.repair_notes)}."
            ).strip()
            degraded_states.extend(repair.degraded_states)

        call_key = _tool_call_key(tool_name, tool_input)
        if call_key in seen_keys:
            call.status = "SKIPPED_DUPLICATE_TOOL_CALL"
            call.error = "Tool call duplicated an earlier executed or planned call."
            degraded_states.append("DUPLICATE_TOOL_CALL_SKIPPED")
            tool_records.append(call)
            continue

        if executed_count >= max(0, int(budget.max_tool_calls)):
            call.status = "SKIPPED_BUDGET_EXHAUSTED"
            call.error = "Tool-call budget exhausted before execution."
            degraded_states.append("BUDGET_EXHAUSTED")
            budget_exhausted = True
            tool_records.append(call)
            continue

        seen_keys.add(call_key)
        call.started_at = _utc_now_iso()
        try:
            output = dispatch_alpha_tool(tool_name, tool_input, ctx)
            call.completed_at = _utc_now_iso()
            call.status = "OK" if output.get("status") != "error" else "ERROR"
            call.output_preview = _json_preview(output)
            if call.status == "ERROR":
                call.error = str(output.get("reason") or "tool_error")
            else:
                evidence_ref = _evidence_from_tool_output(
                    ticker=ticker,
                    call=call,
                    output=output,
                    evidence_index=evidence_start_index + len(evidence),
                )
                evidence.append(evidence_ref)
                call.evidence_ref_ids.append(evidence_ref.evidence_id)
        except Exception as exc:  # pragma: no cover - defensive guard for live tools
            call.completed_at = _utc_now_iso()
            call.status = "ERROR"
            call.error = str(exc)
        executed_count += 1
        tool_records.append(call)

    return (
        tool_records,
        evidence,
        list(dict.fromkeys(degraded_states)),
        executed_count,
        budget_exhausted,
        seen_keys,
    )


def _decision_from_payload(payload: dict[str, Any], ticker: str) -> CandidateDecision:
    raw = (
        payload.get("candidate_decision")
        if isinstance(payload.get("candidate_decision"), dict)
        else {}
    )
    return CandidateDecision(
        ticker=str(raw.get("ticker") or ticker.upper()),
        verdict=str(raw.get("verdict") or payload.get("final_verdict") or "NO_WINNER"),
        confidence=str(raw.get("confidence") or payload.get("confidence") or "LOW"),
        thesis=str(raw.get("thesis") or ""),
        key_risk=str(raw.get("key_risk") or ""),
        eligible_for_selection=bool(raw.get("eligible_for_selection", False)),
        selection_blockers=[str(item) for item in raw.get("selection_blockers") or []],
        falsifiers=[str(item) for item in raw.get("falsifiers") or []],
        evidence_ref_ids=[str(item) for item in raw.get("evidence_ref_ids") or []],
        confidence_cap_reasons=[str(item) for item in raw.get("confidence_cap_reasons") or []],
    )


def _normalize_final_verdict(raw_verdict: Any) -> tuple[str, list[str], str]:
    raw = str(raw_verdict or "").strip().upper()
    if raw in VALID_FINAL_VERDICTS:
        return raw, [], raw
    if raw in FINAL_VERDICT_ALIASES:
        return FINAL_VERDICT_ALIASES[raw], ["FINAL_VERDICT_ALIAS_NORMALIZED"], raw
    return "NO_WINNER", ["FINAL_VERDICT_INVALID"], raw or "MISSING"


def _replace_decision_verdict(decision: CandidateDecision, verdict: str) -> CandidateDecision:
    return CandidateDecision(
        ticker=decision.ticker,
        verdict=verdict,
        confidence=decision.confidence,
        thesis=decision.thesis,
        key_risk=decision.key_risk,
        eligible_for_selection=decision.eligible_for_selection,
        selection_blockers=list(decision.selection_blockers),
        falsifiers=list(decision.falsifiers),
        evidence_ref_ids=list(decision.evidence_ref_ids),
        confidence_cap_reasons=list(decision.confidence_cap_reasons),
    )


def _belief_updates_from_payload(payload: dict[str, Any], ticker: str) -> list[BeliefUpdate]:
    updates: list[BeliefUpdate] = []
    raw_updates = (
        payload.get("belief_updates") if isinstance(payload.get("belief_updates"), list) else []
    )
    for idx, raw in enumerate(raw_updates, start=1):
        if not isinstance(raw, dict):
            continue
        updates.append(
            BeliefUpdate(
                update_id=f"BU{idx}",
                question_id=raw.get("question_id"),
                ticker=raw.get("ticker") or ticker.upper(),
                prior_belief=str(raw.get("prior_belief") or ""),
                updated_belief=str(raw.get("updated_belief") or ""),
                direction=str(raw.get("direction") or "UNCHANGED"),
                confidence_after=str(raw.get("confidence_after") or "LOW"),
                summary=str(raw.get("summary") or ""),
                evidence_ref_ids=[str(item) for item in raw.get("evidence_ref_ids") or []],
                remaining_uncertainty=[
                    str(item) for item in raw.get("remaining_uncertainty") or []
                ],
            )
        )
    return updates


def _out_of_scope_final_decision(
    *,
    ticker: str,
    no_winner_reason: str,
) -> CandidateDecision:
    return CandidateDecision(
        ticker=ticker.upper(),
        verdict="NO_WINNER",
        confidence="LOW",
        thesis="Final provider decision referenced a ticker outside the single-candidate scope.",
        key_risk=no_winner_reason,
        eligible_for_selection=False,
        selection_blockers=["FINAL_DECISION_OUT_OF_SCOPE"],
    )


def _downgrade_inconsistent_actionable_decision(decision: CandidateDecision) -> CandidateDecision:
    blockers = list(decision.selection_blockers)
    if not blockers:
        blockers.append("PROVIDER_MARKED_CANDIDATE_INELIGIBLE")
    return CandidateDecision(
        ticker=decision.ticker,
        verdict="WATCHLIST_ONLY",
        confidence=decision.confidence or "LOW",
        thesis=decision.thesis,
        key_risk=decision.key_risk,
        eligible_for_selection=False,
        selection_blockers=blockers,
        falsifiers=list(decision.falsifiers),
        evidence_ref_ids=list(decision.evidence_ref_ids),
        confidence_cap_reasons=list(decision.confidence_cap_reasons),
    )


def _normalize_no_winner_decision(
    decision: CandidateDecision,
    *,
    no_winner_reason: str,
    blocker: str = "NO_WINNER_FINAL_DECISION_INCONSISTENT",
) -> CandidateDecision:
    blockers = list(decision.selection_blockers)
    if blocker not in blockers:
        blockers.append(blocker)
    return CandidateDecision(
        ticker=decision.ticker,
        verdict="NO_WINNER",
        confidence="LOW",
        thesis=decision.thesis,
        key_risk=no_winner_reason or decision.key_risk,
        eligible_for_selection=False,
        selection_blockers=blockers,
        falsifiers=list(decision.falsifiers),
        evidence_ref_ids=list(decision.evidence_ref_ids),
        confidence_cap_reasons=list(decision.confidence_cap_reasons),
    )


def _has_decision_usable_evidence(evidence: list[EvidenceReference]) -> bool:
    usable_confidence = {"MODERATE", "HIGH"}
    return any(str(item.confidence or "").upper() in usable_confidence for item in evidence)


def _has_decision_usable_tool_evidence(
    *,
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
) -> bool:
    """Return whether an OK tool call produced linked, usable evidence."""

    successful_refs_by_call = {
        call.call_id: set(call.evidence_ref_ids) for call in tool_calls if call.status == "OK"
    }
    usable_confidence = {"MODERATE", "HIGH"}
    return any(
        str(item.confidence or "").upper() in usable_confidence
        and bool(item.tool_call_id)
        and item.evidence_id in successful_refs_by_call.get(str(item.tool_call_id), set())
        for item in evidence
    )


def _downgrade_actionable_without_usable_evidence(decision: CandidateDecision) -> CandidateDecision:
    blockers = list(decision.selection_blockers)
    if "NO_DECISION_USABLE_EVIDENCE" not in blockers:
        blockers.append("NO_DECISION_USABLE_EVIDENCE")
    cap_reasons = list(decision.confidence_cap_reasons)
    if "NO_DECISION_USABLE_EVIDENCE" not in cap_reasons:
        cap_reasons.append("NO_DECISION_USABLE_EVIDENCE")
    return CandidateDecision(
        ticker=decision.ticker,
        verdict="WATCHLIST_ONLY",
        confidence=decision.confidence or "LOW",
        thesis=decision.thesis,
        key_risk=decision.key_risk,
        eligible_for_selection=False,
        selection_blockers=blockers,
        falsifiers=list(decision.falsifiers),
        evidence_ref_ids=list(decision.evidence_ref_ids),
        confidence_cap_reasons=cap_reasons,
    )


def _downgrade_incomplete_actionable_decision(decision: CandidateDecision) -> CandidateDecision:
    blockers = list(decision.selection_blockers)
    if "ACTIONABLE_FINAL_DECISION_INCOMPLETE" not in blockers:
        blockers.append("ACTIONABLE_FINAL_DECISION_INCOMPLETE")
    cap_reasons = list(decision.confidence_cap_reasons)
    if "ACTIONABLE_FINAL_DECISION_INCOMPLETE" not in cap_reasons:
        cap_reasons.append("ACTIONABLE_FINAL_DECISION_INCOMPLETE")
    return CandidateDecision(
        ticker=decision.ticker,
        verdict="WATCHLIST_ONLY",
        confidence=decision.confidence or "LOW",
        thesis=decision.thesis,
        key_risk=decision.key_risk,
        eligible_for_selection=False,
        selection_blockers=blockers,
        falsifiers=list(decision.falsifiers),
        evidence_ref_ids=list(decision.evidence_ref_ids),
        confidence_cap_reasons=cap_reasons,
    )


def _needs_followup_turn(
    *,
    questions: list[ResearchQuestion],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    budget: AutonomousRunBudget,
    executed_count: int,
    budget_exhausted: bool,
) -> bool:
    if budget_exhausted or executed_count >= max(0, int(budget.max_tool_calls)):
        return False
    if any(question.status in {"OPEN", "PARTIAL"} for question in questions):
        return True
    if any(call.status == "ERROR" for call in tool_calls):
        return True
    if not tool_calls and not evidence:
        return True
    return False


def _run_single_candidate_autonomous_analysis_impl(
    ticker: str,
    *,
    objective: str | None = None,
    as_of_date: str | None = None,
    budget: AutonomousRunBudget | None = None,
    allowed_tools: list[str] | None = None,
    analysis_years: int = 5,
    analysis_quarters: int = 4,
    skip_analysis_refresh: bool = False,
    initial_budget: AutonomousRunBudget | None = None,
    canonical_signal_packet: TickerSignalPacket | None = None,
    source_binding: dict[str, Any] | None = None,
) -> AutonomousRunArtifact:
    """Run a bounded autonomous analyst loop for one ticker."""

    upper = ticker.upper()
    run_objective = objective or DEFAULT_AUTONOMOUS_OBJECTIVE
    run_as_of = as_of_date or date.today().isoformat()
    run_budget = _budget_or_default(budget)
    starting_budget = initial_budget or run_budget
    if initial_budget is not None:
        if int(starting_budget.max_tool_calls) > int(run_budget.max_tool_calls) or int(
            starting_budget.max_turns
        ) > int(run_budget.max_turns):
            raise ValueError("initial autonomous budget cannot exceed the extended budget")
        if int(starting_budget.max_turns) < 2:
            raise ValueError("initial autonomous budget must allow planning and finalization")
    started_at = _utc_now_iso()
    bound_financial_packet: SectorCompanyFinancialPacket | None = None
    bound_financial_scenarios: tuple[SectorExpectedReturnScenario, ...] = ()

    if canonical_signal_packet is not None:
        if not isinstance(source_binding, dict):
            raise ValueError("canonical v2 child packet requires source_binding")
        if str(canonical_signal_packet.ticker or "").strip().upper() != upper:
            raise ValueError("canonical child packet ticker does not match requested ticker")
        if str(source_binding.get("ticker") or "").strip().upper() != upper:
            raise ValueError("canonical child source binding ticker mismatch")
        if str(source_binding.get("as_of_date") or "") != run_as_of:
            raise ValueError("canonical child source binding as-of mismatch")
        expected_fingerprint = (
            str(source_binding.get("signal_packet_fingerprint") or "").strip().lower()
        )
        actual_fingerprint = _signal_packet_fingerprint(canonical_signal_packet)
        if expected_fingerprint != actual_fingerprint:
            raise ValueError("canonical child signal packet fingerprint drift")
        if source_binding.get("artifact_type") == "v1_canonical_child_source_binding_v1":
            raw_financial_packet = source_binding.get("financial_integrity_packet")
            raw_financial_scenarios = source_binding.get("financial_integrity_scenarios")
            if not isinstance(raw_financial_packet, dict):
                raise ValueError("canonical v1 child financial packet is missing")
            if not isinstance(raw_financial_scenarios, list):
                raise ValueError("canonical v1 child financial scenarios are missing")
            expected_company_fingerprint = (
                str(source_binding.get("company_packet_fingerprint") or "").strip().lower()
            )
            if expected_company_fingerprint != _financial_payload_fingerprint(raw_financial_packet):
                raise ValueError("canonical v1 child company packet fingerprint drift")
            expected_scenario_fingerprint = (
                str(source_binding.get("financial_integrity_scenarios_fingerprint") or "")
                .strip()
                .lower()
            )
            if expected_scenario_fingerprint != _financial_payload_fingerprint(
                raw_financial_scenarios
            ):
                raise ValueError("canonical v1 child scenario fingerprint drift")
            bound_financial_packet = SectorCompanyFinancialPacket.from_dict(raw_financial_packet)
            bound_financial_scenarios = tuple(
                SectorExpectedReturnScenario.from_dict(item)
                for item in raw_financial_scenarios
                if isinstance(item, dict)
            )
            if str(bound_financial_packet.ticker).strip().upper() != upper:
                raise ValueError("canonical v1 child financial packet ticker mismatch")
            if any(
                str(scenario.ticker).strip().upper() != upper
                for scenario in bound_financial_scenarios
            ):
                raise ValueError("canonical v1 child financial scenario ticker mismatch")
        packet = canonical_signal_packet
    else:
        try:
            packet = _assemble_deterministic_autonomous_signal_packet(
                upper,
                as_of_date=run_as_of,
            )
        except Exception as exc:
            request = _build_request(
                ticker=upper,
                objective=run_objective,
                as_of_date=run_as_of,
                budget=run_budget,
                allowed_tools=list(allowed_tools or DEFAULT_ALLOWED_TOOLS),
                created_at=started_at,
            )
            return _empty_artifact(
                request=request,
                started_at=started_at,
                degraded_state="PACKET_ASSEMBLY_FAILED",
                reason=f"Could not assemble signal packet: {exc}",
            )

    def integrity_scope(context: str) -> FinancialIntegrityScope:
        return FinancialIntegrityScope(
            context=context,
            run_as_of_date=run_as_of,
            packets=(bound_financial_packet or packet,),
            scenarios=bound_financial_scenarios,
        )

    financial_context = (
        {
            "packet": bound_financial_packet.to_dict(),
            "scenarios": [item.to_dict() for item in bound_financial_scenarios],
        }
        if bound_financial_packet is not None
        else {"packet": packet.to_summary_dict(), "scenarios": []}
    )

    if source_binding is not None and canonical_signal_packet is None:
        raise ValueError("source_binding cannot be used without canonical_signal_packet")

    run_allowed_tools = _allowed_tools_for_packet(packet, allowed_tools)
    request = _build_request(
        ticker=upper,
        objective=run_objective,
        as_of_date=run_as_of,
        budget=run_budget,
        allowed_tools=run_allowed_tools,
        created_at=started_at,
    )
    if isinstance(source_binding, dict):
        request.candidate_scope["source_binding"] = json.loads(
            json.dumps(source_binding, sort_keys=True, default=str)
        )
        request.candidate_scope["signal_packet_snapshot"] = canonical_v2_signal_packet_snapshot(
            canonical_signal_packet
        )

    if not _packet_is_usable(packet):
        return _empty_artifact(
            request=request,
            started_at=started_at,
            degraded_state="DATA_QUALITY_INSUFFICIENT",
            reason="No usable signal packet data was available for the ticker.",
        )

    pre_context_scope = integrity_scope("autonomous_pre_analyst_context")
    try:
        pre_context_gate = _require_applied_financial_integrity_scope(pre_context_scope)
    except InvalidFinancialInputError as exc:
        _apply_financial_integrity_result(pre_context_scope, exc.result)
        request.candidate_scope["financial_integrity"] = exc.result.to_dict()
        return _empty_artifact(
            request=request,
            started_at=started_at,
            degraded_state=exc.status,
            reason=(
                "Deterministic financial inputs failed before analyst context or "
                f"provider work: {exc}"
            ),
            audit_notes=[
                "No analyst refresh or provider call ran because the financial-integrity "
                f"gate returned {exc.status}.",
                json.dumps(exc.result.to_dict(), sort_keys=True, default=str),
            ],
        )
    request.candidate_scope["financial_integrity"] = _financial_integrity_result_payload(
        pre_context_gate
    )
    request.candidate_scope["financial_integrity_binding"] = _financial_integrity_run_binding(
        pre_context_scope,
        scope_fingerprint=pre_context_gate.scope_fingerprint,
    )

    def require_run_scope_unchanged() -> None:
        current_result = require_unchanged_financial_integrity_scope(
            pre_context_scope,
            expected_scope_fingerprint=pre_context_gate.scope_fingerprint,
        )
        _apply_financial_integrity_result(pre_context_scope, current_result)

    def finalize_run_artifact(artifact: AutonomousRunArtifact) -> AutonomousRunArtifact:
        require_run_scope_unchanged()
        return artifact

    financial_context = (
        {
            "packet": bound_financial_packet.to_dict(),
            "scenarios": [item.to_dict() for item in bound_financial_scenarios],
        }
        if bound_financial_packet is not None
        else {"packet": packet.to_summary_dict(), "scenarios": []}
    )

    if canonical_signal_packet is not None:
        analyst_evidence, analyst_context, analyst_degraded, analyst_notes = (
            [],
            None,
            [],
            [
                "Canonical child analysis used the immutable sector packet and did not refresh global analyst context."
            ],
        )
    else:
        analyst_evidence, analyst_context, analyst_degraded, analyst_notes = (
            _build_analyst_context_evidence(
                ticker=upper,
                as_of_date=run_as_of,
                packet=packet,
                years=max(1, int(analysis_years)),
                quarters=max(0, int(analysis_quarters)),
                # Paid/deep refresh is not permitted before a separately scoped
                # provider gate. Reuse only already-persisted eligible evidence.
                skip_analysis_refresh=True,
            )
        )

    provider = get_alpha_llm_provider()
    if not _provider_enabled(provider):
        return finalize_run_artifact(
            _empty_artifact(
                request=request,
                started_at=started_at,
                degraded_state="LLM_PROVIDER_UNAVAILABLE",
                reason="LLM provider is unavailable; autonomous run stopped without forcing a verdict.",
                evidence=analyst_evidence,
                audit_notes=analyst_notes
                + [
                    "LLM provider is unavailable; autonomous run stopped without forcing a verdict."
                ],
                additional_degraded_states=analyst_degraded,
            )
        )

    planner_prompt_state = _freeze_financial_prompt_state(
        scope=pre_context_scope,
        expected_scope_fingerprint=pre_context_gate.scope_fingerprint,
        state_getter=lambda: {
            "ticker": upper,
            "objective": run_objective,
            "packet": packet.to_summary_dict(),
            "allowed_tools": run_allowed_tools,
            "budget": starting_budget.to_dict(),
            "analyst_context": analyst_context,
            "financial_context": financial_context,
        },
    )
    planner_prompt = _planner_prompt(
        ticker=upper,
        objective=run_objective,
        packet=packet,
        allowed_tools=run_allowed_tools,
        budget=starting_budget,
        analyst_context=analyst_context,
        financial_context=financial_context,
    )
    planner_request = {
        "prompt": planner_prompt,
        "schema": _PLAN_SCHEMA,
        "schema_name": "autonomous_question_plan",
        "max_output_tokens": 3000,
    }
    try:
        require_run_scope_unchanged()
        plan_payload = _synthesize_provider_json(
            provider,
            integrity_scope=integrity_scope("autonomous_question_plan"),
            _financial_prompt_integrity_binding=planner_prompt_state.bind_request(
                provider,
                planner_request,
            ),
            **planner_request,
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        failure_state = (
            LLM_PROVIDER_QUOTA_EXHAUSTED if _is_quota_exhaustion(exc) else "LLM_PROVIDER_ERROR"
        )
        return finalize_run_artifact(
            _empty_artifact(
                request=request,
                started_at=started_at,
                degraded_state=failure_state,
                reason=f"LLM provider failed while planning research questions: {exc}",
                evidence=analyst_evidence,
                audit_notes=analyst_notes
                + [f"LLM provider failed while planning research questions: {exc}"],
                additional_degraded_states=analyst_degraded,
            )
        )

    questions, planned_calls = _questions_from_plan(upper, plan_payload)
    if not questions:
        return finalize_run_artifact(
            _empty_artifact(
                request=request,
                started_at=started_at,
                degraded_state="NO_RESEARCH_PLAN",
                reason="LLM provider did not return any research questions.",
                evidence=analyst_evidence,
                audit_notes=analyst_notes + ["LLM provider did not return any research questions."],
                additional_degraded_states=analyst_degraded,
            )
        )

    evidence = list(analyst_evidence)
    (
        tool_calls,
        new_evidence,
        degraded_states,
        executed_count,
        budget_exhausted,
        seen_tool_call_keys,
    ) = _execute_planned_calls(
        ticker=upper,
        packet=packet,
        as_of_date=run_as_of,
        allowed_tools=run_allowed_tools,
        budget=starting_budget,
        planned_calls=planned_calls,
        evidence_start_index=len(evidence) + 1,
    )
    evidence.extend(new_evidence)
    _mark_question_statuses(questions, tool_calls)

    def current_derived_prompt_state() -> dict[str, Any]:
        return {
            "ticker": upper,
            "objective": run_objective,
            "packet": packet.to_summary_dict(),
            "questions": questions,
            "tool_calls": tool_calls,
            "evidence": evidence,
            "allowed_tools": run_allowed_tools,
            "budget": run_budget.to_dict(),
            "analyst_context": analyst_context,
            "seen_tool_call_keys": sorted(seen_tool_call_keys),
            "financial_context": financial_context,
        }

    prompt_state_epoch = _freeze_financial_prompt_state(
        scope=pre_context_scope,
        expected_scope_fingerprint=pre_context_gate.scope_fingerprint,
        state_getter=current_derived_prompt_state,
    )

    progressive_extension = initial_budget is not None and (
        int(starting_budget.max_tool_calls) < int(run_budget.max_tool_calls)
        or int(starting_budget.max_turns) < int(run_budget.max_turns)
    )
    extension_attempted = False
    # Exhausting only the starting envelope is the signal to consider the
    # larger envelope, not a terminal run failure.  The full-envelope guard in
    # the follow-up execution remains authoritative.
    if progressive_extension and budget_exhausted:
        budget_exhausted = False
    followup_turns = max(0, int(run_budget.max_turns) - 2)
    for _turn_index in range(followup_turns):
        needs_followup = _needs_followup_turn(
            questions=questions,
            tool_calls=tool_calls,
            evidence=evidence,
            budget=run_budget,
            executed_count=executed_count,
            budget_exhausted=budget_exhausted,
        )
        if (
            progressive_extension
            and not extension_attempted
            and not budget_exhausted
            and executed_count < max(0, int(run_budget.max_tool_calls))
            and not _has_decision_usable_tool_evidence(
                tool_calls=tool_calls,
                evidence=evidence,
            )
        ):
            needs_followup = True
        if not needs_followup:
            break
        extension_attempted = progressive_extension
        followup_prompt = _followup_prompt(
            ticker=upper,
            objective=run_objective,
            packet=packet,
            questions=questions,
            tool_calls=tool_calls,
            evidence=evidence,
            allowed_tools=run_allowed_tools,
            budget=run_budget,
            analyst_context=analyst_context,
            seen_tool_call_keys=seen_tool_call_keys,
            financial_context=financial_context,
        )
        followup_request = {
            "prompt": followup_prompt,
            "schema": _PLAN_SCHEMA,
            "schema_name": "autonomous_followup_question_plan",
            "max_output_tokens": 3000,
        }
        try:
            require_run_scope_unchanged()
            followup_payload = _synthesize_provider_json(
                provider,
                integrity_scope=integrity_scope("autonomous_followup_question_plan"),
                _financial_prompt_integrity_binding=prompt_state_epoch.bind_request(
                    provider,
                    followup_request,
                ),
                **followup_request,
            )
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            degraded_states.append("LLM_PROVIDER_ERROR")
            analyst_notes.append(
                f"LLM provider failed during follow-up planning; finalization continued: {exc}"
            )
            break
        followup_questions, followup_calls = _questions_from_plan(upper, followup_payload)
        if not followup_questions and not followup_calls:
            break
        questions.extend(followup_questions)
        if not followup_calls:
            _mark_question_statuses(questions, tool_calls)
            prompt_state_epoch = _freeze_financial_prompt_state(
                scope=pre_context_scope,
                expected_scope_fingerprint=pre_context_gate.scope_fingerprint,
                state_getter=current_derived_prompt_state,
            )
            break
        (
            records,
            new_evidence,
            turn_degraded,
            executed_count,
            budget_exhausted,
            seen_tool_call_keys,
        ) = _execute_planned_calls(
            ticker=upper,
            packet=packet,
            as_of_date=run_as_of,
            allowed_tools=run_allowed_tools,
            budget=run_budget,
            planned_calls=followup_calls,
            evidence_start_index=len(evidence) + 1,
            call_start_index=len(tool_calls) + 1,
            executed_count=executed_count,
            seen_tool_call_keys=seen_tool_call_keys,
        )
        tool_calls.extend(records)
        evidence.extend(new_evidence)
        degraded_states.extend(turn_degraded)
        _mark_question_statuses(questions, tool_calls)
        prompt_state_epoch = _freeze_financial_prompt_state(
            scope=pre_context_scope,
            expected_scope_fingerprint=pre_context_gate.scope_fingerprint,
            state_getter=current_derived_prompt_state,
        )
        if not new_evidence and all(
            record.status == "SKIPPED_DUPLICATE_TOOL_CALL" for record in records
        ):
            break
        if new_evidence:
            break

    if progressive_extension:
        extension_note = (
            "Progressive autonomous budget extended from "
            f"{starting_budget.max_tool_calls} tools/{starting_budget.max_turns} turns to "
            f"{run_budget.max_tool_calls} tools/{run_budget.max_turns} turns because "
            "decision-relevant evidence remained unresolved."
            if extension_attempted
            else "Progressive autonomous budget completed within the initial "
            f"{starting_budget.max_tool_calls}-tool/{starting_budget.max_turns}-turn envelope."
        )
        analyst_notes.append(extension_note)

    if budget_exhausted or run_budget.max_turns < 2:
        states = list(dict.fromkeys(analyst_degraded + degraded_states + ["BUDGET_EXHAUSTED"]))
        return finalize_run_artifact(
            AutonomousRunArtifact(
                request=request,
                status="COMPLETED",
                started_at=started_at,
                completed_at=_utc_now_iso(),
                final_verdict="NO_WINNER",
                selected_ticker=None,
                confidence=None,
                questions=questions,
                tool_calls=tool_calls,
                evidence=evidence,
                no_winner_reason="Budget exhausted before a candidate decision could be completed.",
                degraded_states=states,
                audit_notes=analyst_notes
                + [
                    "Stopped without forcing a verdict because the autonomous budget was exhausted."
                ],
            )
        )

    final_prompt = _final_prompt(
        ticker=upper,
        objective=run_objective,
        packet=packet,
        questions=questions,
        tool_calls=tool_calls,
        evidence=evidence,
        analyst_context=analyst_context,
        financial_context=financial_context,
    )
    final_request = {
        "prompt": final_prompt,
        "schema": _FINAL_SCHEMA,
        "schema_name": "autonomous_candidate_decision",
        "max_output_tokens": 4000,
    }
    try:
        require_run_scope_unchanged()
        final_payload = _synthesize_provider_json(
            provider,
            integrity_scope=integrity_scope("autonomous_candidate_decision"),
            _financial_prompt_integrity_binding=prompt_state_epoch.bind_request(
                provider,
                final_request,
            ),
            **final_request,
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        failure_state = (
            LLM_PROVIDER_QUOTA_EXHAUSTED if _is_quota_exhaustion(exc) else "LLM_PROVIDER_ERROR"
        )
        return finalize_run_artifact(
            AutonomousRunArtifact(
                request=request,
                status="FAILED",
                started_at=started_at,
                completed_at=_utc_now_iso(),
                final_verdict="NO_WINNER",
                selected_ticker=None,
                confidence=None,
                questions=questions,
                tool_calls=tool_calls,
                evidence=evidence,
                no_winner_reason=f"LLM provider failed while finalizing verdict: {exc}",
                degraded_states=list(
                    dict.fromkeys(analyst_degraded + degraded_states + [failure_state])
                ),
                audit_notes=analyst_notes
                + ["Stopped without forcing a verdict because final decision synthesis failed."],
            )
        )

    provider_degraded = [str(item) for item in final_payload.get("degraded_states") or []]
    decision = _decision_from_payload(final_payload, upper)
    belief_updates = _belief_updates_from_payload(final_payload, upper)
    final_verdict, verdict_states, raw_final_verdict = _normalize_final_verdict(
        final_payload.get("final_verdict") or decision.verdict
    )
    provider_degraded.extend(verdict_states)
    decision = _replace_decision_verdict(decision, final_verdict)
    raw_selected_ticker = final_payload.get("selected_ticker")
    selected_ticker = str(raw_selected_ticker).upper() if raw_selected_ticker else None
    confidence = str(final_payload.get("confidence")) if final_payload.get("confidence") else None
    no_winner_reason = final_payload.get("no_winner_reason")
    audit_notes = [str(item) for item in final_payload.get("audit_notes") or []]
    if "FINAL_VERDICT_INVALID" in verdict_states:
        selected_ticker = None
        confidence = None
        no_winner_reason = (
            str(no_winner_reason)
            if no_winner_reason
            else f"Provider returned invalid final_verdict {raw_final_verdict}."
        )
        audit_notes.append(
            "Deterministic final-decision guard rejected an invalid final verdict label."
        )
        decision = _normalize_no_winner_decision(
            decision,
            no_winner_reason=no_winner_reason,
            blocker="FINAL_VERDICT_INVALID",
        )
    decision_ticker = str(decision.ticker or "").upper()
    referenced_tickers = [
        item for item in (selected_ticker, decision_ticker) if item and item != upper
    ]
    if referenced_tickers:
        referenced = referenced_tickers[0]
        final_verdict = "NO_WINNER"
        selected_ticker = None
        confidence = None
        no_winner_reason = (
            "Provider final decision referenced out-of-scope ticker "
            f"{referenced} for single-candidate run {upper}."
        )
        provider_degraded.append("FINAL_DECISION_OUT_OF_SCOPE")
        audit_notes.append(
            "Deterministic final-decision guard rejected an out-of-scope ticker selection."
        )
        decision = _out_of_scope_final_decision(
            ticker=upper,
            no_winner_reason=no_winner_reason,
        )
    elif final_verdict == "ACTIONABLE" and (
        not decision.eligible_for_selection or decision.selection_blockers
    ):
        final_verdict = "WATCHLIST_ONLY"
        provider_degraded.append("ACTIONABLE_FINAL_DECISION_INCONSISTENT")
        audit_notes.append(
            "Deterministic final-decision guard downgraded an actionable verdict with binding selection blockers."
        )
        decision = _downgrade_inconsistent_actionable_decision(decision)
    elif final_verdict == "ACTIONABLE" and (selected_ticker != upper or confidence is None):
        final_verdict = "WATCHLIST_ONLY"
        provider_degraded.append("ACTIONABLE_FINAL_DECISION_INCOMPLETE")
        audit_notes.append(
            "Deterministic final-decision guard downgraded an actionable verdict missing selected ticker or confidence."
        )
        decision = _downgrade_incomplete_actionable_decision(decision)
    elif final_verdict == "ACTIONABLE" and not _has_decision_usable_evidence(evidence):
        final_verdict = "WATCHLIST_ONLY"
        provider_degraded.append("ACTIONABLE_WITHOUT_DECISION_USABLE_EVIDENCE")
        audit_notes.append(
            "Deterministic final-decision guard downgraded an actionable verdict without decision-usable evidence."
        )
        decision = _downgrade_actionable_without_usable_evidence(decision)
    elif final_verdict == "NO_WINNER" and (
        selected_ticker is not None or confidence is not None or decision.eligible_for_selection
    ):
        selected_ticker = None
        confidence = None
        no_winner_reason = (
            str(no_winner_reason)
            if no_winner_reason
            else "Provider returned NO_WINNER with selected ticker, confidence, or eligible candidate state."
        )
        provider_degraded.append("NO_WINNER_FINAL_DECISION_INCONSISTENT")
        audit_notes.append(
            "Deterministic final-decision guard cleared selected ticker/confidence from inconsistent NO_WINNER payload."
        )
        decision = _normalize_no_winner_decision(
            decision,
            no_winner_reason=no_winner_reason,
        )
    return finalize_run_artifact(
        AutonomousRunArtifact(
            request=request,
            status="COMPLETED",
            started_at=started_at,
            completed_at=_utc_now_iso(),
            final_verdict=final_verdict,
            selected_ticker=selected_ticker,
            confidence=confidence,
            questions=questions,
            tool_calls=tool_calls,
            evidence=evidence,
            belief_updates=belief_updates,
            candidate_decisions=[decision],
            no_winner_reason=no_winner_reason,
            degraded_states=list(
                dict.fromkeys(analyst_degraded + degraded_states + provider_degraded)
            ),
            audit_notes=list(dict.fromkeys(analyst_notes + audit_notes)),
        )
    )


def run_single_candidate_autonomous_analysis(
    ticker: str,
    *,
    objective: str | None = None,
    as_of_date: str | None = None,
    budget: AutonomousRunBudget | None = None,
    allowed_tools: list[str] | None = None,
    analysis_years: int = 5,
    analysis_quarters: int = 4,
    skip_analysis_refresh: bool = False,
    execution_lane: str = "company_underwriting",
    initial_budget: AutonomousRunBudget | None = None,
    canonical_signal_packet: TickerSignalPacket | None = None,
    source_binding: dict[str, Any] | None = None,
) -> AutonomousRunArtifact:
    """Run one company analysis while capturing every provider attempt by lane."""

    with provider_usage_capture(execution_lane) as provider_usage:
        try:
            artifact = _run_single_candidate_autonomous_analysis_impl(
                ticker,
                objective=objective,
                as_of_date=as_of_date,
                budget=budget,
                allowed_tools=allowed_tools,
                analysis_years=analysis_years,
                analysis_quarters=analysis_quarters,
                skip_analysis_refresh=skip_analysis_refresh,
                initial_budget=initial_budget,
                canonical_signal_packet=canonical_signal_packet,
                source_binding=source_binding,
            )
        except Exception as exc:
            attach_provider_usage_to_exception(exc, provider_usage)
            raise
    artifact.provider_usage = merge_provider_usage_records(
        artifact.provider_usage,
        provider_usage,
    )
    for call in artifact.tool_calls:
        if call.lane is None:
            call.lane = execution_lane
    return artifact


def artifact_summary(artifact: AutonomousRunArtifact) -> dict[str, Any]:
    """Small JSON-safe CLI summary for one autonomous run."""

    return {
        "run_id": artifact.request.run_id,
        "ticker": artifact.request.candidate_scope.get("tickers", [None])[0],
        "status": artifact.status,
        "final_verdict": artifact.final_verdict,
        "selected_ticker": artifact.selected_ticker,
        "confidence": artifact.confidence,
        "degraded_states": list(artifact.degraded_states),
        "no_winner_reason": artifact.no_winner_reason,
        "tool_calls": len([call for call in artifact.tool_calls if call.status == "OK"]),
    }


__all__ = [
    "DEFAULT_ALLOWED_TOOLS",
    "DEFAULT_AUTONOMOUS_OBJECTIVE",
    "DEFAULT_BUDGET",
    "artifact_summary",
    "run_single_candidate_autonomous_analysis",
]
