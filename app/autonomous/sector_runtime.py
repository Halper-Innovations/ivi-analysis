"""Iterative autonomous sector financial analyst runtime."""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Mapping

from app.alpha.cross_sectional_ranker import (
    CROSS_SECTIONAL_BUY_QUANTILE as _RANKER_BUY_QUANTILE,
    DEFAULT_FACTOR_WEIGHTS,
    FactorVector,
    rank_cross_sectional,
)
from app.alpha.llm_runtime import _provider_enabled, get_alpha_llm_provider
from app.alpha.llm_tools import AlphaToolContext, dispatch_alpha_tool
from app.alpha.schemas import TickerSignalPacket
from app.alpha.signal_assembler import assemble_sector_packets
from app.alpha.solvency_scanner import going_concern_asserted
from app.config import get_config, resolve_autonomous_sector_pipeline_version
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    require_financial_integrity_scope,
    require_unchanged_financial_integrity_scope,
)
from app.autonomous.run_contract import (
    AutonomousRunArtifact,
    AutonomousRunBudget,
    BeliefUpdate,
    EvidenceReference,
    ToolCallRecord,
)
from app.autonomous.competitive_frontier import build_competitive_frontier
from app.autonomous.runtime import (
    DEFAULT_ALLOWED_TOOLS,
    INSURANCE_TOOL,
    LLM_PROVIDER_QUOTA_EXHAUSTED,
    _BoundFinancialPromptCall,
    _FINANCIAL_PROMPT_BINDING_KWARG,
    _adapt_provider_synthesize_kwargs,
    _freeze_financial_prompt_state,
    _is_quota_exhaustion,
    _rebind_financial_provider_request,
    run_single_candidate_autonomous_analysis,
)
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    CandidateDisposition,
    GateEvaluation,
    SECTOR_CONTRACT_VERSION_V2,
    SECTOR_PIPELINE_VERSION_V2,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorFinalDecision,
    SectorFinancialFramework,
    SectorResearchQuestion,
    SectorSelectionValidation,
    ScreenResult,
    UnderwritingResult,
    _v2_blocking_run_states,
    build_v2_canonical_child_source_bindings,
    canonical_v2_signal_packet_snapshot,
    selection_validation_terminal_ledger_fingerprint,
)
from app.autonomous.candidate_review import (
    candidate_memo_is_substantive,
    candidate_memo_schema_name,
)
from app.autonomous.candidate_checkpoint import (
    CANDIDATE_MEMO_CHECKPOINT_CONTRACT_VERSION,
    CANDIDATE_MEMO_CONTEXT_CHECKPOINT_CONTRACT_VERSION,
    CANDIDATE_MEMO_CONTEXT_PROMPT_VERSION,
    CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION,
    CANDIDATE_MEMO_PROMPT_VERSION,
    CandidateMemoCheckpointError,
    bind_provider_usage_to_checkpoint,
    candidate_checkpoint_key_sha256,
    candidate_checkpoint_path,
    candidate_context_checkpoint_key_sha256,
    candidate_context_checkpoint_path,
    canonical_json_sha256,
    load_candidate_memo_checkpoint,
    load_candidate_memo_context_checkpoint,
    persist_candidate_memo_checkpoint,
    persist_candidate_memo_context_checkpoint,
)
from app.autonomous.sector_framework_templates import (
    augment_sector_framework_payload,
    sector_framework_contract_id,
    sector_framework_pipeline_version,
    sector_framework_screen_rule_ids,
    sector_framework_template_payload,
)
from app.autonomous.sector_lane_budget import (
    CANONICAL_LANES,
    EXTENDED_COMPANY_CHILD_BUDGET,
    INITIAL_COMPANY_CHILD_BUDGET,
)
from app.autonomous.sector_financial_packets import (
    build_sector_company_financial_packets_from_signal_packets,
)
from app.autonomous.sector_scenarios import build_expected_return_scenarios_for_packets
from app.autonomous.tool_input_guardrails import repair_alpha_tool_input
from app.autonomous.v1_financial_context import build_canonical_v1_financial_context
from app.llm.providers import DeepSeekOutputTruncatedError, get_anthropic_provider
from app.llm.providers.retry_guard import (
    DEFAULT_LLM_CALL_TIMEOUT_SECONDS,
    DEFAULT_LLM_RETRY_RUN_BUDGET,
    LLMCostBudgetExceeded,
    LLMMaxRetriesExceeded,
    LLMRetryBudgetExceeded,
    call_with_llm_retry_guard,
    current_cost_context,
    current_retry_context,
    llm_attempt_observer,
    llm_cost_budget,
    llm_physical_attempt_guard,
    llm_retry_budget,
)
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    attached_provider_usage_records,
    failed_provider_usage_meta,
    merge_provider_usage_records,
    provider_usage_capture,
    provider_usage_meta,
    provider_usage_records,
    provider_usage_records_from_exception,
    record_provider_usage,
)
from app.util.financial_data_access import companyfacts_rows


DEFAULT_SECTOR_OBJECTIVE = (
    "Determine which company has the strongest financially underwritten 5-10 year "
    "per-share return potential, or return no selection if the evidence does not clear the bar."
)

_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG = "_expected_integrity_scope_fingerprint"

DEFAULT_SECTOR_ALLOWED_TOOLS = [
    "summarize_financial_packets",
    "compare_expected_return_scenarios",
    "rank_expected_return_cases",
    *DEFAULT_ALLOWED_TOOLS,
]

DEFAULT_SECTOR_BUDGET = AutonomousRunBudget(
    max_tool_calls=16,
    max_turns=6,
    # Research quality is the default constraint. Operators can still pass an
    # explicit cost ceiling, but an implicit per-sector dollar cap must not
    # cut off finalist synthesis after the evidence/tool phase has completed.
    max_cost_usd=None,
    timebox_seconds=None,
    max_candidates=None,
)
SECTOR_LLM_RETRY_RUN_BUDGET = DEFAULT_LLM_RETRY_RUN_BUDGET
LLM_COST_BUDGET_EXCEEDED = "LLM_COST_BUDGET_EXCEEDED"

BASE_RETURN_HURDLE = 0.12
# Default mode for the 12% base-return hurdle. 'soft' demotes a sub-hurdle
# base return to a HURDLE-class confidence cap (WATCHLIST_ONLY); 'hard' preserves
# the legacy hard-block. The live value is read from get_config() so operators can
# flip it via the BASE_RETURN_HURDLE_MODE env var without code changes.
BASE_RETURN_HURDLE_MODE = "soft"
# Sector-relative cross-sectional ranking tunables. CROSS_SECTIONAL_BUY_QUANTILE
# is the top-quantile BUY-candidate cutoff; CROSS_SECTIONAL_FACTOR_WEIGHTS blends
# value/quality/gap within-sector z-scores. The pure ranker negates the gap factor
# (raw implied_growth, lower-is-better) internally, so callers pass RAW implied_growth.
CROSS_SECTIONAL_BUY_QUANTILE = _RANKER_BUY_QUANTILE
CROSS_SECTIONAL_FACTOR_WEIGHTS: dict[str, float] = dict(DEFAULT_FACTOR_WEIGHTS)
SELECTED_RETURN_CUSHION_HURDLE = 0.15
DOWNSIDE_RISK_CAP_THRESHOLD = -0.05
SECTOR_PROMPT_CANDIDATE_LIMIT = 25
SECTOR_MEMO_PROMPT_PREFLIGHT_MAX_TOKENS = 100_000
REPORTABLE_FINANCIAL_HISTORY_MIN_ROWS = 3
CANDIDATE_MEMO_TICKER_MISMATCH = "CANDIDATE_MEMO_TICKER_MISMATCH"
CANDIDATE_MEMO_CONTENT_INCOMPLETE = "CANDIDATE_MEMO_CONTENT_INCOMPLETE"
CANDIDATE_MEMO_MAX_OUTPUT_TOKENS = 2200
CANDIDATE_MEMO_MAX_WORKERS = 4
SHARED_MEMO_MAX_OUTPUT_TOKENS = 1800
SUSPICIOUS_VALUATION_ANCHOR_PRICE_MULTIPLE = 2.5
SUSPICIOUS_VALUATION_ANCHOR_CAP = "SUSPICIOUS_MAGNITUDE_DCF"
AUDIT_GAP_REPAIR_TARGET_LIMIT = 5
AUDIT_GAP_REPAIR_TOOLS_PER_TARGET = 2
REPORTABLE_FINANCIAL_HISTORY_LINE_ITEMS = (
    "revenue",
    "operating_income",
    "cfo",
    "capex",
    "cash",
    "total_debt",
    "shares_outstanding",
    "sbc",
)
AuditSignalClass = Literal["EVIDENCE_QUALITY", "BUSINESS_QUALITY", "HURDLE", "CATALYST"]
AUDIT_SIGNAL_CLASSIFICATION: dict[str, AuditSignalClass] = {
    # CATALYST is a strictly POSITIVE, non-penalizing timing class. It is
    # excluded from every binding set (hard blockers, hurdle caps,
    # business-quality caps) so a catalyst can never downgrade a grade; it only
    # surfaces in the audit's catalyst_signals list.
    "INSIDER_BUY_CLUSTER": "CATALYST",
    "BUYBACK_ACCELERATION": "CATALYST",
    "BASE_RETURN_BELOW_12PCT_HURDLE": "HURDLE",
    "BASE_RETURN_BELOW_HURDLE_SOFT": "HURDLE",
    "THIN_RETURN_CUSHION": "HURDLE",
    "NEGATIVE_MARGIN_OF_SAFETY": "HURDLE",
    "EXPENSIVE_VS_EXPECTATIONS": "HURDLE",
    SUSPICIOUS_VALUATION_ANCHOR_CAP: "BUSINESS_QUALITY",
    "ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK": "BUSINESS_QUALITY",
    "UNRESOLVED_HIGH_GROWTH_DEPENDENCY": "BUSINESS_QUALITY",
    "NO_ASSURANCE_FINANCING_UNRESOLVED": "BUSINESS_QUALITY",
    "FOLLOW_UP_NO_ASSURANCE_FINANCING": "BUSINESS_QUALITY",
    "STRUCTURALLY_WEAK_ECONOMICS": "BUSINESS_QUALITY",
    "CAPITAL_STRUCTURE_WATCHLIST": "BUSINESS_QUALITY",
    "FOLLOW_UP_CAPITAL_STRUCTURE_ACTIVE_DISTRESS": "BUSINESS_QUALITY",
    "PROBABLE_PERMANENT_CAPITAL_LOSS": "BUSINESS_QUALITY",
    "CURRENT_EVENTS_UNAVAILABLE": "EVIDENCE_QUALITY",
    "CAPITAL_LOSS_EVIDENCE_DEGRADED": "EVIDENCE_QUALITY",
    "NO_FILING": "EVIDENCE_QUALITY",
    "NO_READABLE_ANNUAL_FILING": "EVIDENCE_QUALITY",
    "FILING_RISK_NO_FILING": "EVIDENCE_QUALITY",
    "FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE": "EVIDENCE_QUALITY",
    "MISSING_KPI_QUALITY_EVIDENCE": "EVIDENCE_QUALITY",
    "NEAR_TERM_MATURITY_UNRESOLVED": "EVIDENCE_QUALITY",
    "RISK_SECTION_NOT_FOUND": "EVIDENCE_QUALITY",
    "FILING_RISK_SECTION_NOT_FOUND": "EVIDENCE_QUALITY",
    "QUARTERLY_REVENUE_TREND_UNKNOWN": "EVIDENCE_QUALITY",
    "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE": "EVIDENCE_QUALITY",
    "MISSING_COMPANY_SPECIFIC_EVIDENCE": "EVIDENCE_QUALITY",
    "INSUFFICIENT_COMPANY_SPECIFIC_EVIDENCE": "EVIDENCE_QUALITY",
    "INSUFFICIENT_EVIDENCE_PILLAR_COVERAGE": "EVIDENCE_QUALITY",
}
_V2_DATA_AVAILABILITY_BLOCKERS = frozenset(
    {
        "MISSING_PRICE",
        "MISSING_VALUATION",
        "MISSING_BASE_RETURN_CASE",
        "MODEL_FIT_BLOCKED",
    }
)
CAPITAL_LOSS_IMPAIRMENT_BLOCKERS = {
    "CLEAR_IMPAIRMENT": "PERMANENT_CAPITAL_LOSS_CONFIRMED",
    "PROBABLE_IMPAIRMENT": "PROBABLE_PERMANENT_CAPITAL_LOSS",
}
CAPITAL_LOSS_IMPAIRMENT_CAPS = {
    "STRUCTURALLY_WEAK_NOT_IMPAIRED": "STRUCTURALLY_WEAK_ECONOMICS",
    "EVIDENCE_DEGRADED_NOT_ASSESSABLE": "CAPITAL_LOSS_EVIDENCE_DEGRADED",
}
CAPITAL_LOSS_STATUS_BY_IMPAIRMENT_CLASS = {
    "CLEAR_IMPAIRMENT": "PERMANENT_LOSS_BLOCKED",
    "PROBABLE_IMPAIRMENT": "PROBABLE_PERMANENT_LOSS_BLOCKED",
    "TEMPORARY_WEAKNESS": "TEMPORARY_WEAKNESS_SUPPORTED",
    "STRUCTURALLY_WEAK_NOT_IMPAIRED": "STRUCTURALLY_WEAK_WATCHLIST",
    "EVIDENCE_DEGRADED_NOT_ASSESSABLE": "EVIDENCE_DEGRADED_WATCHLIST",
    "IMPAIRMENT_UNKNOWN": "UNKNOWN",
}
ALTERNATE_FINALIST_AUDIT_LIMIT = 3
COMPANY_AUTONOMY_MAX_CANDIDATES = 3
COMPANY_AUTONOMY_CHILD_MAX_TOOL_CALLS = 4
COMPANY_AUTONOMY_CHILD_EXTENDED_MAX_TOOL_CALLS = 8
FALLBACK_IMPAIRMENT_DEGRADED_CAPS = {
    "FILING_RISK_NO_FILING",
    "FILING_RISK_ERROR",
    "FILING_RISK_KEYWORD_FALLBACK",
    "FILING_RISK_SECTION_NOT_FOUND",
}
FALLBACK_STRUCTURAL_VALUATION_HEADWINDS = {
    "ACCOUNTING_QUALITY_HEADWIND",
    "CAPITAL_ALLOCATION_HEADWIND",
    "EARNINGS_QUALITY_HEADWIND",
    "INVENTORY_BUILDING_HEADWIND",
    "LEVERAGE_STRESS_HEADWIND",
    "LOW_ACCOUNTING_QUALITY_HEADWIND",
    "NET_DILUTION_HEADWIND",
    "RECEIVABLES_DETERIORATING_HEADWIND",
    "SBC_BURDEN_EXTREME_HEADWIND",
    "SBC_BURDEN_HEADWIND",
    "SECULAR_DECLINE_HEADWIND",
    "WORKING_CAPITAL_DRAG_HEADWIND",
}


def classify_audit_signal(code: str) -> AuditSignalClass:
    """Classify a deterministic audit signal by underwriting meaning."""
    normalized = str(code or "").strip().upper()
    return AUDIT_SIGNAL_CLASSIFICATION.get(normalized, "BUSINESS_QUALITY")


def _audit_signal_classification_rows(codes: list[str]) -> list[dict[str, str]]:
    return [
        {"code": str(code), "classification": classify_audit_signal(str(code))}
        for code in codes
        if str(code).strip()
    ]


def _audit_signal_class_counts(codes: list[str]) -> list[dict[str, Any]]:
    counts = Counter(classify_audit_signal(str(code)) for code in codes if str(code).strip())
    return [
        {"classification": classification, "count": counts[classification]}
        for classification in ("EVIDENCE_QUALITY", "BUSINESS_QUALITY", "HURDLE", "CATALYST")
        if counts[classification]
    ]


def _audit_signals_by_class(codes: list[str], classification: AuditSignalClass) -> list[str]:
    return list(
        dict.fromkeys(
            str(code)
            for code in codes
            if str(code).strip() and classify_audit_signal(str(code)) == classification
        )
    )


def _mergeable_confidence_caps(selection_audit: dict[str, Any]) -> list[str]:
    """Audit confidence-cap codes to merge into a DOWNGRADED decision's reasons.

    When the final-decision dispatch converts a candidate to WATCHLIST_ONLY (or
    sweeps a NO_SELECTION finalist), it merges the selection audit's
    ``confidence_caps`` into the user-facing ``confidence_cap_reasons``. CATALYST
    -class codes (INSIDER_BUY_CLUSTER / BUYBACK_ACCELERATION) are a strictly
    POSITIVE, non-penalizing timing axis and must NOT be displayed as a
    downgrade reason — they surface only via ``selection_audit['catalyst_signals']``.
    This drops them so the merge mirrors how ``binding_hurdle_caps`` /
    ``business_quality_confidence_caps`` already class-filter the catalyst out.
    """
    return [
        str(item)
        for item in selection_audit.get("confidence_caps") or []
        if classify_audit_signal(str(item)) != "CATALYST"
    ]


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        token = str(value).strip().replace("%", "")
        if not token:
            return None
        parsed = float(token)
    except (TypeError, ValueError):
        return None
    return parsed / 100.0 if abs(parsed) > 1.0 and "%" in str(value) else parsed


def _iter_margin_of_safety_values(payload: Any) -> list[float]:
    values: list[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() == "margin_of_safety":
                parsed = _float_or_none(value)
                if parsed is not None:
                    values.append(parsed)
            values.extend(_iter_margin_of_safety_values(value))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_iter_margin_of_safety_values(item))
    return values


def _margin_of_safety_from_text(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(
        r"(?:adjusted\s+)?margin\s+of\s+safety\s+(?:is|=|at)?\s*(?:negative\s+at\s*)?(-?\d+(?:\.\d+)?)\s*%",
        str(text),
        re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(1)) / 100.0


def _negative_margin_of_safety_profile(
    *,
    ticker: str,
    packet: SectorCompanyFinancialPacket,
    evidence: list[EvidenceReference],
) -> dict[str, Any]:
    valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    expected_return = packet.expected_return if isinstance(packet.expected_return, dict) else {}
    for source_key, value in (
        ("packet.valuation.margin_of_safety", valuation.get("margin_of_safety")),
        ("packet.valuation.discount_to_anchor", valuation.get("discount_to_anchor")),
        ("packet.expected_return.margin_of_safety", expected_return.get("margin_of_safety")),
    ):
        parsed = _float_or_none(value)
        if parsed is not None and parsed < 0:
            return {"status": "NEGATIVE", "margin_of_safety": parsed, "source": source_key}
    if str(valuation.get("margin_of_safety_verdict") or "").upper() == "OVERVALUED":
        return {
            "status": "NEGATIVE",
            "margin_of_safety": None,
            "source": "packet.valuation.margin_of_safety_verdict",
        }

    ticker_upper = str(ticker or "").upper()
    for item in evidence:
        if not _evidence_mentions_ticker(item, ticker_upper):
            continue
        for text_source, text in (("summary", item.summary), ("excerpt", item.excerpt)):
            parsed_text = _margin_of_safety_from_text(text)
            if parsed_text is not None and parsed_text < 0:
                return {
                    "status": "NEGATIVE",
                    "margin_of_safety": parsed_text,
                    "source": f"evidence.{item.evidence_id}.{text_source}",
                }
        if item.excerpt:
            try:
                payload = json.loads(item.excerpt)
            except (TypeError, json.JSONDecodeError):
                payload = None
            for parsed in _iter_margin_of_safety_values(payload):
                if parsed < 0:
                    return {
                        "status": "NEGATIVE",
                        "margin_of_safety": parsed,
                        "source": f"evidence.{item.evidence_id}.excerpt.margin_of_safety",
                    }
    return {"status": "NON_NEGATIVE_OR_UNAVAILABLE", "margin_of_safety": None, "source": None}


def _valuation_anchor_magnitude_profile(packet: SectorCompanyFinancialPacket) -> dict[str, Any]:
    valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    anchor = _float_or_none(valuation.get("valuation_anchor"))
    current_price = _float_or_none(
        packet.current_price if packet.current_price is not None else valuation.get("current_price")
    )
    if anchor is None or current_price is None or current_price <= 0:
        return {
            "status": "NOT_ASSESSABLE",
            "valuation_anchor_to_price_ratio": None,
            "threshold": SUSPICIOUS_VALUATION_ANCHOR_PRICE_MULTIPLE,
        }
    ratio = anchor / current_price
    return {
        "status": "SUSPICIOUS" if ratio > SUSPICIOUS_VALUATION_ANCHOR_PRICE_MULTIPLE else "OK",
        "valuation_anchor_to_price_ratio": round(ratio, 3),
        "threshold": SUSPICIOUS_VALUATION_ANCHOR_PRICE_MULTIPLE,
    }


def _audit_signal_classification_summary(
    *,
    hard_blockers: list[str],
    confidence_caps: list[str],
) -> dict[str, Any]:
    evidence_signals = _audit_signals_by_class(
        [*hard_blockers, *confidence_caps], "EVIDENCE_QUALITY"
    )
    return {
        "hard_blocker_classifications": _audit_signal_classification_rows(hard_blockers),
        "confidence_cap_classifications": _audit_signal_classification_rows(confidence_caps),
        "blockers_by_class": _audit_signal_class_counts(hard_blockers),
        "caps_by_class": _audit_signal_class_counts(confidence_caps),
        "needs_evidence_resolution": evidence_signals,
    }


COMPANY_AUTONOMY_CHILD_MAX_TURNS = 2
COMPANY_AUTONOMY_CHILD_EXTENDED_MAX_TURNS = 3
COMPANY_AUTONOMY_CHILD_MAX_COST_USD: float | None = None
GENERIC_VALUATION_ANCHOR_METHODS = {"dcf", "epv", "graham", "ncav"}
EXPECTED_RETURN_EVIDENCE_TOOLS = {"rank_expected_return_cases", "compare_expected_return_scenarios"}
FRESHNESS_EVIDENCE_TOOLS = {
    "fetch_filing_section",
    "fetch_current_events",
    "fetch_recent_filing_context",
}
COMPANY_EVIDENCE_PILLARS = {
    "fetch_kpi_trends": "quality",
    "compare_peer_metric": "quality",
    "analyze_liquidity_stress": "liquidity",
    "analyze_capital_structure_resolution": "liquidity",
    "analyze_dilution": "dilution",
    "analyze_capital_allocation": "capital_allocation",
    "fetch_filing_section": "filing_freshness",
    "fetch_recent_filing_context": "filing_freshness",
    "fetch_current_events": "current_events",
    "fetch_transcript_excerpt": "management_commentary",
    "fetch_companyfacts_timeseries": "companyfacts",
    "fetch_insurance_evidence_packet": "insurance",
}
SELECTED_RISK_EVIDENCE_TOOLS = {
    "analyze_liquidity_stress",
    "analyze_capital_structure_resolution",
    "analyze_dilution",
    "analyze_capital_allocation",
    "fetch_filing_section",
    "fetch_recent_filing_context",
    "fetch_current_events",
    "fetch_companyfacts_timeseries",
}
AUDIT_GAP_REPAIRABLE_BLOCKERS = {
    "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE",
    "MISSING_COMPANY_SPECIFIC_EVIDENCE",
    "FOLLOW_UP_NO_ASSURANCE_FINANCING",
    "FOLLOW_UP_GOING_CONCERN_LANGUAGE",
    "FOLLOW_UP_SOLVENCY_CRITICAL",
    "PACKET_GOING_CONCERN_LANGUAGE",
}

SECTOR_TOOL_NAMES = {
    "summarize_financial_packets",
    "compare_expected_return_scenarios",
    "rank_expected_return_cases",
}

GUARDRAILS_BINDING_AUDIT_NOTE = (
    "Runtime guardrails are binding over provider working notes; "
    "top-level final_verdict and final_decision fields are the source of truth."
)

_CODE_LIKE_STATE_RE = re.compile(r"^[A-Z][A-Z0-9_]*(?::.*)?$")


def _split_provider_degraded_states(items: list[Any]) -> tuple[list[str], list[str]]:
    """Keep machine-readable provider states separate from free-form provider guidance."""

    degraded: list[str] = []
    audit_notes: list[str] = []
    for item in items or []:
        text = str(item or "").strip()
        if not text:
            continue
        label = text.split(":", 1)[0].strip()
        if _CODE_LIKE_STATE_RE.match(text) and (
            "_" in label or (label.isupper() and ":" not in text)
        ):
            degraded.append(text)
        else:
            audit_notes.append(f"Provider degraded-state note moved to audit trail: {text}")
    return list(dict.fromkeys(degraded)), list(dict.fromkeys(audit_notes))


_JSON_OBJECT_EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

_POLICY_THRESHOLD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "string"},
        "units": {"type": "string"},
    },
    "required": ["name", "value", "units"],
    "additionalProperties": False,
}

_FRAMEWORK_POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approach": {"type": "string"},
        "rules": {"type": "array", "items": {"type": "string"}},
        "numeric_thresholds": {"type": "array", "items": _POLICY_THRESHOLD_SCHEMA},
    },
    "required": ["approach", "rules", "numeric_thresholds"],
    "additionalProperties": False,
}

_SECTOR_TOOL_INPUT_SCHEMA: dict[str, Any] = {
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
        "horizon_years": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "scenario_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "tickers": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
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
        "horizon_years",
        "scenario_name",
        "tickers",
    ],
    "additionalProperties": False,
}

_FRAMEWORK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sector": {"type": "string"},
        "market_cap_focus": {"type": "string"},
        "horizon_years": {"type": "array", "items": {"type": "integer"}},
        "economic_model": {"type": "string"},
        "selected_value_drivers": {"type": "array", "items": {"type": "string"}},
        "selected_metrics": {"type": "array", "items": {"type": "string"}},
        "valid_valuation_methods": {"type": "array", "items": {"type": "string"}},
        "invalid_valuation_methods": {"type": "array", "items": {"type": "string"}},
        "required_evidence": {"type": "array", "items": {"type": "string"}},
        "normalization_policy": _FRAMEWORK_POLICY_SCHEMA,
        "hurdle_rate_policy": _FRAMEWORK_POLICY_SCHEMA,
        "weighting_policy": _FRAMEWORK_POLICY_SCHEMA,
        "sector_specific_risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "sector",
        "market_cap_focus",
        "horizon_years",
        "economic_model",
        "selected_value_drivers",
        "selected_metrics",
        "valid_valuation_methods",
        "invalid_valuation_methods",
        "required_evidence",
        "normalization_policy",
        "hurdle_rate_policy",
        "weighting_policy",
        "sector_specific_risks",
    ],
    "additionalProperties": False,
}

_REJECTED_FINALIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["ticker", "reason"],
    "additionalProperties": False,
}

_SECTOR_FINAL_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SELECTED", "WATCHLIST", "NO_SELECTION", "NO_WINNER"],
        },
        "confidence": {
            "anyOf": [{"type": "string", "enum": ["HIGH", "MODERATE", "LOW"]}, {"type": "null"}]
        },
        "selected_ticker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "expected_annualized_return_range": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "thesis": {"type": "string"},
        "key_risk": {"type": "string"},
        "downside_case": {"type": "string"},
        "no_selection_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "falsifiers": {"type": "array", "items": {"type": "string"}},
        "why_selected_over_finalists": {"type": "array", "items": {"type": "string"}},
        "rejected_finalists": {"type": "array", "items": _REJECTED_FINALIST_SCHEMA},
        "selection_blockers": {"type": "array", "items": {"type": "string"}},
        "confidence_cap_reasons": {"type": "array", "items": {"type": "string"}},
        "evidence_ref_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "verdict",
        "confidence",
        "selected_ticker",
        "expected_annualized_return_range",
        "thesis",
        "key_risk",
        "downside_case",
        "no_selection_reason",
        "falsifiers",
        "why_selected_over_finalists",
        "rejected_finalists",
        "selection_blockers",
        "confidence_cap_reasons",
        "evidence_ref_ids",
    ],
    "additionalProperties": False,
}

_TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "framework": {
            "anyOf": [
                _FRAMEWORK_SCHEMA,
                {"type": "null"},
            ]
        },
        "research_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "question": {"type": "string"},
                    "financial_pillar": {"type": "string"},
                    "expected_decision_impact": {"type": "string"},
                    "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "target_tickers": {"type": "array", "items": {"type": "string"}},
                    "planned_tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "ticker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "tool_input": _SECTOR_TOOL_INPUT_SCHEMA,
                                "rationale": {"type": "string"},
                            },
                            "required": ["tool_name", "ticker", "tool_input", "rationale"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "question_id",
                    "question",
                    "financial_pillar",
                    "expected_decision_impact",
                    "priority",
                    "target_tickers",
                    "planned_tool_calls",
                ],
                "additionalProperties": False,
            },
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
        "continue_research": {"type": "boolean"},
        "final_decision": {
            "anyOf": [
                _SECTOR_FINAL_DECISION_SCHEMA,
                {"type": "null"},
            ]
        },
        "degraded_states": {"type": "array", "items": {"type": "string"}},
        "audit_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "framework",
        "research_questions",
        "belief_updates",
        "continue_research",
        "final_decision",
        "degraded_states",
        "audit_notes",
    ],
    "additionalProperties": False,
}

_INITIAL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "framework": _FRAMEWORK_SCHEMA,
        "research_questions": _TURN_SCHEMA["properties"]["research_questions"],
        "belief_updates": _TURN_SCHEMA["properties"]["belief_updates"],
        "degraded_states": {"type": "array", "items": {"type": "string"}},
        "audit_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "framework",
        "research_questions",
        "belief_updates",
        "degraded_states",
        "audit_notes",
    ],
    "additionalProperties": False,
}

_MINIMUM_TOOL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "research_questions": _TURN_SCHEMA["properties"]["research_questions"],
        "degraded_states": {"type": "array", "items": {"type": "string"}},
        "audit_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "research_questions",
        "degraded_states",
        "audit_notes",
    ],
    "additionalProperties": False,
}

_COHORT_COMPARISON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "paragraphs": {"type": "array", "items": {"type": "string"}},
        "generation_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["paragraphs", "generation_notes"],
    "additionalProperties": False,
}

_TRIAGE_SURPRISES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "string"}},
        "generation_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["items", "generation_notes"],
    "additionalProperties": False,
}

_MEMO_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "minLength": 1},
        "thesis": {"type": "string", "minLength": 1},
        "key_risks": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
        "falsifiers": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
    },
    "required": ["ticker", "thesis", "key_risks", "falsifiers", "open_questions"],
    "additionalProperties": False,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_id(sector: str, created_at: str) -> str:
    day = created_at[:10].replace("-", "")
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in sector.lower()).strip("_") or "sector"
    return f"autonomous_sector_{cleaned}_{day}_{secrets.token_hex(3)}"


def _budget_or_default(budget: AutonomousRunBudget | None) -> AutonomousRunBudget:
    if budget is None:
        return AutonomousRunBudget.from_dict(DEFAULT_SECTOR_BUDGET.to_dict())
    return budget


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_jsonable(item) for item in value]
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


def _require_candidate_financial_integrity_binding(
    candidate_selection: dict[str, Any] | None,
    *,
    packets: list[Any] | tuple[Any, ...],
    scenarios: list[Any] | tuple[Any, ...],
) -> None:
    binding = (
        candidate_selection.get("financial_integrity_binding")
        if isinstance(candidate_selection, dict)
        else None
    )
    if not isinstance(binding, dict):
        return
    expected = str(binding.get("scope_fingerprint") or "")
    context = str(binding.get("context") or "")
    run_as_of_date = str(binding.get("run_as_of_date") or "")
    stored_scope = FinancialIntegrityScope(
        context=context,
        run_as_of_date=run_as_of_date,
        packets=tuple(binding.get("packets") or ()),
        scenarios=tuple(binding.get("scenarios") or ()),
    )
    require_unchanged_financial_integrity_scope(
        stored_scope,
        expected_scope_fingerprint=expected,
    )
    require_unchanged_financial_integrity_scope(
        FinancialIntegrityScope(
            context=context,
            run_as_of_date=run_as_of_date,
            packets=tuple(packets),
            scenarios=tuple(scenarios),
        ),
        expected_scope_fingerprint=expected,
    )


def _json_preview(value: Any, limit: int = 1600) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, default=str)[:limit]


def _exception_metadata(exc: BaseException, *, limit: int = 500) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    value = f"{exc.__class__.__name__}: {text}"
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _parse_provider_json(result: Any) -> dict[str, Any]:
    raw = getattr(result, "json_text", result)
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw))


def _estimate_tokens_from_text(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def _estimate_llm_cost_usd(
    *,
    provider_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    # Single source of truth for per-model pricing: drive Anthropic cost from
    # the configured/bound model family (Haiku ~ $1/$5 per 1M), not a hardcoded
    # Opus constant that overestimated ~15x and tripped premature budget aborts.
    from app.llm.synthesis_agent import _pricing_per_1k

    input_per_1k, output_per_1k = _pricing_per_1k(model, provider_name=provider_name)
    return round(
        (input_tokens / 1000.0) * input_per_1k + (output_tokens / 1000.0) * output_per_1k, 6
    )


def _confidence_for_tool_output(output: dict[str, Any]) -> str:
    if not isinstance(output, dict):
        return "LOW"
    if output.get("usable_for_decision") is False:
        return "LOW"
    status = str(output.get("status") or "").lower()
    if status == "ok":
        return "MODERATE"
    return "LOW"


def _confidence_rank(value: str | None) -> int:
    return {"LOW": 1, "MODERATE": 2, "HIGH": 3}.get(str(value or "").upper(), 0)


def _confidence_at_least(value: str | None, minimum: str) -> bool:
    return _confidence_rank(value) >= _confidence_rank(minimum)


def _cap_confidence(value: str | None, ceiling: str | None) -> str | None:
    if ceiling is None:
        return None
    if not value:
        return ceiling
    return value if _confidence_rank(value) <= _confidence_rank(ceiling) else ceiling


def _provider_name(provider: Any) -> str:
    return str(getattr(provider, "provider_name", provider.__class__.__name__) or "unknown").lower()


def _provider_model_for_estimate(provider: Any) -> str:
    provider_name = _provider_name(provider)
    cfg = getattr(provider, "cfg", None) or get_config()
    if provider_name == "anthropic":
        # Prefer the provider's bound model (may be a cheap override that
        # differs from cfg.anthropic_model) so the cost estimate matches reality.
        provider_model = str(getattr(provider, "model", "") or "")
        return provider_model or str(getattr(cfg, "anthropic_model", "") or "")
    if provider_name == "openai":
        return str(getattr(cfg, "openai_model", "") or "")
    return str(getattr(cfg, "llm_model", "") or provider_name)


def _estimated_output_tokens_for_provider(provider: Any, kwargs: dict[str, Any]) -> int:
    explicit = kwargs.get("max_output_tokens")
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass
    cfg = getattr(provider, "cfg", None) or get_config()
    if _provider_name(provider) == "anthropic":
        return max(1, int(getattr(cfg, "anthropic_max_output_tokens", 1000) or 1000))
    return max(1, int(getattr(cfg, "openai_max_output_tokens", 1000) or 1000))


def _estimated_llm_call_cost_usd(provider: Any, kwargs: dict[str, Any]) -> float:
    prompt = str(kwargs.get("prompt") or "")
    return _estimate_llm_cost_usd(
        provider_name=_provider_name(provider),
        model=_provider_model_for_estimate(provider),
        # UTF-8 bytes are a conservative tokenizer upper bound; the fixed
        # margin covers provider envelope/special tokens.  A hard cost cap
        # cannot rely on the usual chars/4 planning estimate.
        input_tokens=max(1, len(prompt.encode("utf-8")) + 1024),
        output_tokens=_estimated_output_tokens_for_provider(provider, kwargs),
    )


def _reserve_llm_cost_budget(provider: Any, kwargs: dict[str, Any]) -> Any:
    context = current_cost_context()
    if context is None:
        return None
    return context.reserve_call(
        provider=_provider_name(provider),
        schema_name=str(kwargs.get("schema_name") or "structured_output"),
        estimated_cost_usd=_estimated_llm_call_cost_usd(provider, kwargs),
    )


def _usage_cost_usd(records: list[dict[str, Any]]) -> float:
    return sum(
        max(0.0, float(record.get("cost_estimate_usd") or 0.0))
        for record in records
        if isinstance(record, dict)
    )


def _complete_llm_cost_budget(
    reservation: Any,
    meta: dict[str, Any],
    *,
    physical_attempts: list[dict[str, Any]] | None = None,
) -> None:
    context = current_cost_context()
    if context is None:
        return
    actual_cost_usd = float(meta.get("cost_estimate_usd") or 0.0)
    if context.strict_first_call and physical_attempts:
        # Provider results can represent multiple billed responses (for
        # example output-expansion attempts).  Strict accounting charges the
        # complete physical ledger rather than the logical result summary.
        actual_cost_usd = max(actual_cost_usd, _usage_cost_usd(physical_attempts))
    context.complete_call(
        reservation,
        actual_cost_usd=actual_cost_usd,
    )


def _cancel_llm_cost_budget(reservation: Any) -> None:
    """Restore legacy per-sector enforcement after a failed logical call.

    Physical failures remain persisted in the v2 provider-usage ledger.  The
    old v1 cost guard, however, cancelled a failed logical reservation instead
    of consuming the next-call envelope; retaining that enforcement behavior
    keeps flag-gated v1 schedules stable while v2 relies on the whole-run
    preflight for authorization.
    """

    context = current_cost_context()
    if context is not None:
        context.cancel_call(reservation)


def _settle_failed_llm_cost_budget(
    reservation: Any,
    physical_attempts: list[dict[str, Any]],
) -> None:
    """Charge strict failed attempts; retain legacy cancellation otherwise."""

    context = current_cost_context()
    if context is None:
        return
    if not context.strict_first_call:
        context.cancel_call(reservation)
        return
    reserved_cost_usd = max(
        0.0,
        float(getattr(reservation, "estimated_cost_usd", 0.0) or 0.0),
    )
    context.complete_call(
        reservation,
        actual_cost_usd=max(
            reserved_cost_usd,
            _usage_cost_usd(physical_attempts),
        ),
    )


def _provider_synthesize_json_result(
    provider: Any,
    kwargs: dict[str, Any],
) -> tuple[Any, bool]:
    """Make the one physical call and report whether the token cap survived.

    Legacy-signature compatibility is resolved by capability/signature
    inspection *before* invocation — a ``TypeError`` from the provider is never
    evidence that transport did not occur, so it must never trigger a second
    call. When adaptation has to drop the cap the call still happens once, and
    the ``False`` flag is what stops that uncapped attempt from being reused as
    a checkpoint downstream.
    """

    call_kwargs = _adapt_provider_synthesize_kwargs(provider, kwargs)
    max_output_tokens_applied = (
        "max_output_tokens" not in kwargs or "max_output_tokens" in call_kwargs
    )
    return provider.synthesize_json(**call_kwargs), max_output_tokens_applied


def _financial_integrity_scope(
    *,
    context: str,
    as_of_date: str,
    packets: list[Any] | tuple[Any, ...],
    scenarios: list[Any] | tuple[Any, ...] = (),
) -> FinancialIntegrityScope:
    """Bind one paid call to the exact deterministic financial inputs it sees."""

    return FinancialIntegrityScope(
        context=context,
        run_as_of_date=as_of_date,
        packets=tuple(packets),
        scenarios=tuple(scenarios),
    )


def _apply_financial_integrity_result(
    *,
    packets: list[Any] | tuple[Any, ...],
    scenarios: list[Any] | tuple[Any, ...],
    result: Any,
) -> None:
    """Persist gate state on the exact packet/scenario objects being published."""

    # A few legacy unit tests replace the gate with a no-op. Production
    # ``require_financial_integrity_scope`` always returns a concrete result.
    if result is None:
        return
    violations = [item.to_dict() for item in getattr(result, "violations", ())]
    for item in (*tuple(packets), *tuple(scenarios)):
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
    _apply_financial_integrity_result(
        packets=scope.packets,
        scenarios=scope.scenarios,
        result=result,
    )
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


def _checkpointable_financial_integrity_payload(
    result: Any,
) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = _financial_integrity_result_payload(result)
    fingerprint = str(payload.get("scope_fingerprint") or "").strip().lower()
    snapshot_ids = payload.get("ticker_snapshot_ids")
    if (
        payload.get("status") != "PASS"
        or payload.get("passed") is not True
        or payload.get("violations")
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        or not isinstance(snapshot_ids, dict)
        or not snapshot_ids
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value or "").strip().lower())
            is None
            for value in snapshot_ids.values()
        )
    ):
        return None
    return payload


def _guarded_provider_synthesize_json_result(
    provider: Any,
    kwargs: dict[str, Any],
) -> tuple[Any, bool]:
    if bool(getattr(provider, "_handles_retry_guard", False)):
        return _provider_synthesize_json_result(provider, kwargs)
    return call_with_llm_retry_guard(
        provider_name=_provider_name(provider),
        schema_name=str(kwargs.get("schema_name") or "structured_output"),
        call=lambda: _provider_synthesize_json_result(provider, kwargs),
        timeout_seconds=DEFAULT_LLM_CALL_TIMEOUT_SECONDS,
    )


def _bind_logical_provider_integrity_scope(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Authorize one immutable scope for the primary/fallback logical call."""

    bound_kwargs = dict(kwargs)
    integrity_scope = bound_kwargs.get("integrity_scope")
    if integrity_scope is None or bound_kwargs.get(_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG):
        return bound_kwargs
    gate_result = _require_applied_financial_integrity_scope(integrity_scope)
    bound_kwargs[_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG] = gate_result.scope_fingerprint
    return bound_kwargs


def _call_provider_result_with_meta(
    provider: Any, kwargs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
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
        _apply_financial_integrity_result(
            packets=integrity_scope.packets,
            scenarios=integrity_scope.scenarios,
            result=exc.result,
        )
        raise
    if expected_scope_fingerprint:
        gate_result = require_unchanged_financial_integrity_scope(
            integrity_scope,
            expected_scope_fingerprint=expected_scope_fingerprint,
        )
        _apply_financial_integrity_result(
            packets=integrity_scope.packets,
            scenarios=integrity_scope.scenarios,
            result=gate_result,
        )
    else:
        expected_scope_fingerprint = gate_result.scope_fingerprint

    def require_exact_scope(_attempt: dict[str, Any]) -> None:
        current_result = require_unchanged_financial_integrity_scope(
            integrity_scope,
            expected_scope_fingerprint=expected_scope_fingerprint,
        )
        _apply_financial_integrity_result(
            packets=integrity_scope.packets,
            scenarios=integrity_scope.scenarios,
            result=current_result,
        )
        if prompt_binding is not None:
            prompt_binding.require_current(provider, kwargs)

    require_exact_scope({})
    reservation = _reserve_llm_cost_budget(provider, kwargs)
    failed_attempts: list[dict[str, Any]] = []

    def observe_failed_attempt(event: dict[str, Any]) -> None:
        error = event.get("error")
        if not isinstance(error, BaseException):
            error = RuntimeError(str(error or "physical provider attempt failed"))
        failure_meta = failed_provider_usage_meta(
            provider=provider,
            prompt=str(kwargs.get("prompt") or ""),
            schema_name=str(kwargs.get("schema_name") or "structured_output"),
            estimated_output_tokens=_estimated_output_tokens_for_provider(provider, kwargs),
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
            (
                result,
                max_output_tokens_applied,
            ) = _guarded_provider_synthesize_json_result(provider, kwargs)
    except Exception as exc:
        successful_attempts = provider_usage_records_from_exception(
            provider=provider,
            error=exc,
            prompt=str(kwargs.get("prompt") or ""),
            schema_name=str(kwargs.get("schema_name") or "structured_output"),
        )
        for attempt_meta in successful_attempts:
            record_provider_usage(attempt_meta)
        if not failed_attempts and not successful_attempts:
            failure_meta = failed_provider_usage_meta(
                provider=provider,
                prompt=str(kwargs.get("prompt") or ""),
                schema_name=str(kwargs.get("schema_name") or "structured_output"),
                estimated_output_tokens=_estimated_output_tokens_for_provider(provider, kwargs),
                error=exc,
            )
            failed_attempts.append(failure_meta)
            record_provider_usage(failure_meta)
        attach_provider_usage_to_exception(
            exc,
            [*failed_attempts, *successful_attempts],
        )
        _settle_failed_llm_cost_budget(
            reservation,
            [*failed_attempts, *successful_attempts],
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
    meta = provider_usage_meta(
        provider=provider,
        result=result,
        prompt=str(kwargs.get("prompt") or ""),
        schema_name=str(kwargs.get("schema_name") or "structured_output"),
    )
    meta.update(
        {
            "max_output_tokens_applied": max_output_tokens_applied,
            "requested_max_output_tokens": kwargs.get("max_output_tokens"),
        }
    )
    successful_attempts = provider_usage_records(
        provider=provider,
        result=result,
        prompt=str(kwargs.get("prompt") or ""),
        schema_name=str(kwargs.get("schema_name") or "structured_output"),
    )
    for attempt_meta in successful_attempts:
        attempt_meta.update(
            {
                "max_output_tokens_applied": max_output_tokens_applied,
                "requested_max_output_tokens": kwargs.get("max_output_tokens"),
            }
        )
    for attempt_meta in successful_attempts:
        record_provider_usage(attempt_meta)
    # Preserve the pre-v2 logical-call cost-guard semantics.  The returned
    # provider result already aggregates successful output-expansion attempts;
    # failed physical retries are captured for benchmark truth but do not
    # silently tighten a v1 sector's next-call allowance.
    _complete_llm_cost_budget(
        reservation,
        meta,
        physical_attempts=[*failed_attempts, *successful_attempts],
    )
    try:
        require_exact_scope({})
    except InvalidFinancialInputError as exc:
        attach_provider_usage_to_exception(
            exc,
            [*failed_attempts, *successful_attempts],
        )
        raise
    return result, meta


def _require_provider_request_binding_current(
    provider: Any,
    kwargs: dict[str, Any],
) -> None:
    binding = kwargs.get(_FINANCIAL_PROMPT_BINDING_KWARG)
    if binding is None:
        return
    if not isinstance(binding, _BoundFinancialPromptCall):
        raise TypeError("financial prompt integrity binding has an invalid type")
    request = dict(kwargs)
    request.pop(_FINANCIAL_PROMPT_BINDING_KWARG, None)
    request.pop(_EXPECTED_INTEGRITY_SCOPE_FINGERPRINT_KWARG, None)
    request.pop("integrity_scope", None)
    binding.require_current(provider, request)


def _call_provider_json(provider: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    result, _meta = _call_provider_result_with_meta(provider, kwargs)
    try:
        payload = _parse_provider_json(result)
    except Exception as exc:
        attach_provider_usage_to_exception(
            exc,
            provider_usage_records(
                provider=provider,
                result=result,
                prompt=str(kwargs.get("prompt") or ""),
                schema_name=str(kwargs.get("schema_name") or "structured_output"),
            ),
        )
        raise
    _require_provider_request_binding_current(provider, kwargs)
    return payload


def _call_provider_json_with_meta(
    provider: Any, kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    result, meta = _call_provider_result_with_meta(provider, kwargs)
    try:
        payload = _parse_provider_json(result)
    except Exception as exc:
        attach_provider_usage_to_exception(
            exc,
            provider_usage_records(
                provider=provider,
                result=result,
                prompt=str(kwargs.get("prompt") or ""),
                schema_name=str(kwargs.get("schema_name") or "structured_output"),
            ),
        )
        raise
    _require_provider_request_binding_current(provider, kwargs)
    return payload, meta


def _synthesize_provider_json(provider: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs = _bind_logical_provider_integrity_scope(kwargs)
    try:
        return _call_provider_json(provider, kwargs)
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        if isinstance(exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
            raise
        if bool(current_cost_context() is not None and current_cost_context().strict_first_call):
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
            if isinstance(fallback_exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
                raise
            raise RuntimeError(
                f"{LLM_PROVIDER_QUOTA_EXHAUSTED}: primary provider {_provider_name(provider)} "
                f"exhausted quota and Anthropic fallback failed: {fallback_exc}"
            ) from fallback_exc


def _synthesize_provider_json_with_meta(
    provider: Any, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs = _bind_logical_provider_integrity_scope(kwargs)
    try:
        return _call_provider_json_with_meta(provider, kwargs)
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        if isinstance(exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
            raise
        if bool(current_cost_context() is not None and current_cost_context().strict_first_call):
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
            payload, meta = _call_provider_json_with_meta(fallback, fallback_kwargs)
            meta = {
                **meta,
                "fallback_from_provider": _provider_name(provider),
                "original_failure": _exception_metadata(exc),
            }
            return payload, meta
        except InvalidFinancialInputError:
            raise
        except Exception as fallback_exc:
            if isinstance(fallback_exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
                raise
            raise RuntimeError(
                f"{LLM_PROVIDER_QUOTA_EXHAUSTED}: primary provider {_provider_name(provider)} "
                f"exhausted quota and Anthropic fallback failed: {fallback_exc}"
            ) from fallback_exc


def _synthesize_memo_provider_json_with_usage(
    provider: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return memo output plus an explicit thread-local physical-call ledger."""

    with provider_usage_capture("parent_research") as provider_usage:
        try:
            payload, meta = _synthesize_provider_json_with_meta(provider, **kwargs)
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            attach_provider_usage_to_exception(exc, provider_usage)
            raise
    return payload, meta, provider_usage


def _provider_failure_state(exc: Exception) -> str:
    if isinstance(exc, LLMCostBudgetExceeded):
        return LLM_COST_BUDGET_EXCEEDED
    if isinstance(exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
        return "LLM_RETRY_BUDGET_EXCEEDED"
    if isinstance(exc, LLMMaxRetriesExceeded):
        return "LLM_PROVIDER_MAX_RETRIES_EXCEEDED"
    if isinstance(exc, DeepSeekOutputTruncatedError):
        return "LLM_PROVIDER_TRUNCATED"
    if _is_quota_exhaustion(exc):
        return LLM_PROVIDER_QUOTA_EXHAUSTED
    msg = str(exc).lower()
    exc_name = exc.__class__.__name__.lower()
    if "timeout" in msg or "timed out" in msg or "timeout" in exc_name:
        return "LLM_PROVIDER_TIMEOUT"
    if "did not include output text" in msg or "empty output" in msg or "empty" in msg:
        return "LLM_PROVIDER_EMPTY_OUTPUT"
    if isinstance(exc, json.JSONDecodeError) or "not valid json" in msg or "json" in exc_name:
        return "LLM_PROVIDER_INVALID_JSON"
    return "LLM_PROVIDER_ERROR"


def _append_unique(items: list[str], *values: str) -> None:
    seen = set(items)
    for value in values:
        if value and value not in seen:
            items.append(value)
            seen.add(value)


def _append_llm_retry_audit_notes(
    artifact: AutonomousSectorFinancialRunArtifact,
    retry_context: Any,
) -> None:
    summary = retry_context.summary() if hasattr(retry_context, "summary") else {}
    retry_count = int(summary.get("retry_count") or 0)
    if retry_count <= 0:
        return
    artifact.audit_notes = [
        note
        for note in artifact.audit_notes
        if not str(note).startswith("LLM retry guard recorded ")
    ]
    events = summary.get("events") if isinstance(summary.get("events"), list) else []
    event_preview = "; ".join(
        f"{event.get('provider')}:{event.get('schema_name')}#{event.get('attempt')} {event.get('reason')}"
        for event in events[:5]
        if isinstance(event, dict)
    )
    if len(events) > 5:
        event_preview += f"; +{len(events) - 5} more"
    artifact.audit_notes.append(
        "LLM retry guard recorded "
        f"{retry_count}/{summary.get('retry_budget')} retries"
        + (f" ({event_preview})." if event_preview else ".")
    )


def _append_llm_cost_audit_notes(
    artifact: AutonomousSectorFinancialRunArtifact,
    cost_context: Any,
) -> None:
    if LLM_COST_BUDGET_EXCEEDED not in set(artifact.degraded_states or []):
        return
    summary = cost_context.summary() if hasattr(cost_context, "summary") else {}
    max_cost = summary.get("max_cost_usd")
    if max_cost is None:
        return
    call_count = int(summary.get("call_count") or 0)
    if call_count <= 0:
        return
    cumulative = float(summary.get("cumulative_cost_usd") or 0.0)
    artifact.audit_notes = [
        note
        for note in artifact.audit_notes
        if not str(note).startswith("LLM cost guard recorded ")
    ]
    artifact.audit_notes.append(
        "LLM cost guard recorded "
        f"${cumulative:.4f}/${float(max_cost):.4f} across {call_count} provider call(s)."
    )


def _recover_final_decision_after_provider_error(
    *,
    provider: Any,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    research_questions: list[SectorResearchQuestion],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    belief_updates: list[BeliefUpdate],
    provider_error: Exception,
    prompt_state_epoch: Any,
) -> SectorFinalDecision:
    recovery_prompt = _recovery_final_decision_prompt(
        sector=sector,
        market_cap_focus=market_cap_focus,
        objective=objective,
        as_of_date=as_of_date,
        company_packets=company_packets,
        scenarios=scenarios,
        research_questions=research_questions,
        tool_calls=tool_calls,
        evidence=evidence,
        belief_updates=belief_updates,
        provider_error=str(provider_error),
    )
    recovery_request = {
        "prompt": recovery_prompt,
        "schema": _SECTOR_FINAL_DECISION_SCHEMA,
        "schema_name": "autonomous_sector_final_decision_recovery",
        "max_output_tokens": 2500,
    }
    payload = _synthesize_provider_json(
        provider,
        integrity_scope=_financial_integrity_scope(
            context="autonomous_sector_final_decision_recovery",
            as_of_date=as_of_date,
            packets=company_packets,
            scenarios=scenarios,
        ),
        _financial_prompt_integrity_binding=prompt_state_epoch.bind_request(
            provider,
            recovery_request,
        ),
        **recovery_request,
    )
    return SectorFinalDecision.from_dict(payload)


def _fallback_rejected_finalists_from_evidence(
    *,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    research_questions: list[SectorResearchQuestion],
    limit: int = 8,
) -> list[dict[str, Any]]:
    packet_tickers = {str(packet.ticker).upper() for packet in company_packets}
    finalists: list[dict[str, Any]] = []
    seen: set[str] = set()

    base_returns = _scenario_base_return_by_ticker(scenarios)
    for ticker, base_return in sorted(base_returns.items(), key=lambda item: (-item[1], item[0])):
        if ticker not in packet_tickers or ticker in seen:
            continue
        seen.add(ticker)
        finalists.append(
            {
                "ticker": ticker,
                "reason": (
                    f"Scenario-ranked base case at {base_return * 100:.1f}% annualized; "
                    "provider final decision was unavailable, so deterministic audit is required."
                ),
            }
        )
        if len(finalists) >= limit:
            return finalists

    for question in research_questions:
        for ticker in question.target_tickers:
            normalized = str(ticker).upper()
            if normalized not in packet_tickers or normalized in seen:
                continue
            seen.add(normalized)
            finalists.append(
                {
                    "ticker": normalized,
                    "reason": (
                        "Provider-planned research target; provider final decision was unavailable, "
                        "so deterministic audit is required."
                    ),
                }
            )
            if len(finalists) >= limit:
                return finalists

    return finalists


def _deterministic_finalization_fallback_decision(
    *,
    provider_error: Exception,
    recovery_error: Exception,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    research_questions: list[SectorResearchQuestion],
    evidence: list[EvidenceReference],
) -> SectorFinalDecision:
    finalists = _fallback_rejected_finalists_from_evidence(
        company_packets=company_packets,
        scenarios=scenarios,
        research_questions=research_questions,
    )
    reason = (
        "LLM provider final decision was unavailable after deterministic evidence collection; "
        "the runtime will attempt deterministic finalist audit instead of forcing a selection."
    )
    error_summary = f"Initial failure: {provider_error}; compact recovery failure: {recovery_error}"
    return SectorFinalDecision(
        verdict="NO_SELECTION",
        confidence=None,
        selected_ticker=None,
        expected_annualized_return_range=None,
        thesis="No provider-authored final decision was available; deterministic audit will govern any outcome.",
        key_risk=(
            "Provider final-decision failure can leave evidence interpretation incomplete, "
            "so only deterministic audit pass/fail may change the result."
        ),
        downside_case="No provider-authored downside synthesis was available after evidence collection.",
        no_selection_reason=f"{reason} {error_summary}",
        falsifiers=[],
        why_selected_over_finalists=[],
        rejected_finalists=finalists,
        selection_blockers=["LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE"],
        confidence_cap_reasons=["LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE"],
        evidence_ref_ids=[item.evidence_id for item in evidence],
    )


def _empty_artifact(
    *,
    run_id: str,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    created_at: str,
    started_packets: list[SectorCompanyFinancialPacket] | None = None,
    scenarios: list[SectorExpectedReturnScenario] | None = None,
    degraded_state: str,
    reason: str,
    audit_notes: list[str] | None = None,
    candidate_selection: dict[str, Any] | None = None,
    status: str = "FAILED",
) -> AutonomousSectorFinancialRunArtifact:
    packets = started_packets or []
    expected_scenarios = scenarios or []
    return AutonomousSectorFinancialRunArtifact(
        run_id=run_id,
        sector=sector,
        market_cap_focus=market_cap_focus,
        objective=objective,
        as_of_date=as_of_date,
        created_at=created_at,
        completed_at=_utc_now_iso(),
        status=status,
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        candidate_selection=candidate_selection or {},
        candidate_dispositions=_valuation_anchor_dispositions(candidate_selection),
        company_packets=packets,
        expected_return_scenarios=expected_scenarios,
        relative_ranking=_relative_ranking(
            company_packets=packets,
            scenarios=expected_scenarios,
            tool_calls=[],
            evidence=[],
            degraded_states=[degraded_state],
            company_autonomy_runs=[],
        )
        if packets
        else [],
        no_selection_reason=reason,
        degraded_states=[degraded_state],
        audit_notes=audit_notes or [reason],
    )


def _allowed_tools(
    signal_packets: dict[str, TickerSignalPacket], allowed_tools: list[str] | None
) -> list[str]:
    if allowed_tools is not None:
        return list(dict.fromkeys(str(tool).strip() for tool in allowed_tools if str(tool).strip()))
    tools = list(DEFAULT_SECTOR_ALLOWED_TOOLS)
    if any(
        isinstance(packet.insurance_packet, dict) and packet.insurance_packet
        for packet in signal_packets.values()
    ):
        tools.append(INSURANCE_TOOL)
    return list(dict.fromkeys(tools))


def _company_autonomy_budget_capacity(
    *, budget: AutonomousRunBudget, executed_tool_count: int
) -> int:
    """Return how many child company runs fit inside the remaining sector tool budget."""

    if int(budget.max_tool_calls) < COMPANY_AUTONOMY_CHILD_MAX_TOOL_CALLS * 3:
        return 0
    remaining = max(0, int(budget.max_tool_calls) - int(executed_tool_count))
    return min(COMPANY_AUTONOMY_MAX_CANDIDATES, remaining // COMPANY_AUTONOMY_CHILD_MAX_TOOL_CALLS)


def _company_autonomy_child_budget(*, extended: bool = False) -> AutonomousRunBudget:
    return AutonomousRunBudget(
        max_tool_calls=(
            COMPANY_AUTONOMY_CHILD_EXTENDED_MAX_TOOL_CALLS
            if extended
            else COMPANY_AUTONOMY_CHILD_MAX_TOOL_CALLS
        ),
        max_turns=(
            COMPANY_AUTONOMY_CHILD_EXTENDED_MAX_TURNS
            if extended
            else COMPANY_AUTONOMY_CHILD_MAX_TURNS
        ),
        max_cost_usd=COMPANY_AUTONOMY_CHILD_MAX_COST_USD,
        timebox_seconds=None,
        max_candidates=1,
    )


def _company_autonomy_allowed_tools(allowed_tools: list[str]) -> list[str]:
    return [tool for tool in allowed_tools if tool not in SECTOR_TOOL_NAMES]


def _company_autonomy_candidates(
    *,
    research_questions: list[SectorResearchQuestion],
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    candidate_selection: dict[str, Any] | None,
    limit: int,
) -> list[str]:
    packets_by_ticker = {packet.ticker.upper(): packet for packet in company_packets}
    ordered: list[str] = []

    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    for question in sorted(
        research_questions,
        key=lambda item: priority_rank.get(str(item.priority or "").upper(), 3),
    ):
        for ticker in question.target_tickers:
            ordered.append(str(ticker).upper())

    base_returns = _scenario_base_return_by_ticker(scenarios)
    ordered.extend(
        ticker
        for ticker, _return in sorted(
            base_returns.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    if isinstance(candidate_selection, dict):
        ordered.extend(
            str(ticker).upper()
            for ticker in candidate_selection.get("selected_tickers") or []
            if str(ticker).strip()
        )
    ordered.extend(packet.ticker.upper() for packet in company_packets)

    selected: list[str] = []
    for ticker in ordered:
        if ticker in selected or ticker not in packets_by_ticker:
            continue
        packet = packets_by_ticker[ticker]
        blockers = set(_selection_blockers_for_packet(packet))
        if "MODEL_FIT_BLOCKED" in blockers or not _packet_has_price_and_valuation(packet):
            continue
        selected.append(ticker)
        if len(selected) >= limit:
            break
    return selected


def _company_autonomy_objective(*, sector: str, ticker: str) -> str:
    return (
        f"Underwrite {ticker} as a company-level candidate inside the {sector} sector run. "
        "Choose decision-relevant questions, use deterministic financial tools, and determine "
        "whether the company strengthens the sector-level ranking, remains watchlist-only, "
        "should be avoided, or lacks enough evidence."
    )


def _selected_company_challenge_objective(*, sector: str, ticker: str) -> str:
    return (
        f"Independently challenge the provisional selection of {ticker} in the {sector} "
        "sector run. Start from the strongest plausible bear case, actively try to falsify "
        "the selection thesis with fresh company-specific evidence, and test valuation, "
        "capital structure, filing risk, and downside durability. Return ACTIONABLE only "
        "when the provisional selection survives the challenge; otherwise return "
        "WATCHLIST_ONLY, AVOID, or NO_WINNER with explicit evidence-linked reasons."
    )


def _compact_company_autonomy_run(artifact: AutonomousRunArtifact) -> dict[str, Any]:
    ticker = artifact.request.candidate_scope.get("tickers", [artifact.selected_ticker])[0]
    physical_tool_calls = [call for call in artifact.tool_calls if call.status in {"OK", "ERROR"}]
    nonphysical_statuses = Counter(
        call.status for call in artifact.tool_calls if call.status not in {"OK", "ERROR"}
    )
    return {
        "run_id": artifact.request.run_id,
        "ticker": ticker,
        "status": artifact.status,
        "final_verdict": artifact.final_verdict,
        "selected_ticker": artifact.selected_ticker,
        "confidence": artifact.confidence,
        "degraded_states": list(artifact.degraded_states),
        "no_winner_reason": artifact.no_winner_reason,
        "questions": len(artifact.questions),
        "tool_calls": len([call for call in artifact.tool_calls if call.status == "OK"]),
        "tool_call_attempts": len(physical_tool_calls),
        "nonphysical_tool_call_count": sum(nonphysical_statuses.values()),
        "nonphysical_tool_call_status_counts": dict(sorted(nonphysical_statuses.items())),
        "evidence_references": len(artifact.evidence),
        "provider_usage": [dict(item) for item in artifact.provider_usage],
        "artifact": artifact.to_dict(),
    }


def _merge_company_autonomy_evidence(
    *,
    child_artifact: AutonomousRunArtifact,
    evidence_start_index: int,
) -> list[EvidenceReference]:
    ticker = str(
        child_artifact.request.candidate_scope.get("tickers", [child_artifact.selected_ticker])[0]
        or child_artifact.selected_ticker
        or ""
    ).upper()
    merged: list[EvidenceReference] = []
    for idx, item in enumerate(child_artifact.evidence, start=evidence_start_index):
        merged.append(
            EvidenceReference(
                evidence_id=f"E{idx}",
                source_type="company_autonomous_run",
                source_label=item.source_label,
                summary=(f"Company autonomous run for {ticker}: {item.summary}").strip(),
                ticker=str(item.ticker or ticker).upper() if (item.ticker or ticker) else None,
                source_date=item.source_date,
                source_url=item.source_url,
                excerpt=item.excerpt,
                tool_call_id=item.tool_call_id,
                confidence=item.confidence,
            )
        )
    if not merged:
        merged.append(
            EvidenceReference(
                evidence_id=f"E{evidence_start_index}",
                source_type="company_autonomous_run",
                source_label="company_autonomous_run",
                summary=(
                    f"Company autonomous run for {ticker} completed with verdict "
                    f"{child_artifact.final_verdict}; no decision-usable child evidence was produced."
                ),
                ticker=ticker or None,
                confidence="LOW",
            )
        )
    return merged


def _company_underwriting_child_requires_extension(run: dict[str, Any]) -> bool:
    verdict = _v2_normalized_underwriting_verdict(run.get("final_verdict"))
    return bool(_v2_child_underwriting_gap_reasons(run, verdict=verdict))


def _v2_json_fingerprint(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _v1_child_source_bindings(
    *,
    sector: str,
    as_of_date: str,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    signal_packets: dict[str, TickerSignalPacket],
) -> dict[str, dict[str, Any]]:
    """Bind v1 child analysis to the parent run's immutable packet cohort."""

    ordered_tickers = [str(packet.ticker).strip().upper() for packet in company_packets]
    normalized_signals = {
        str(ticker).strip().upper(): packet
        for ticker, packet in signal_packets.items()
        if str(ticker).strip()
    }
    signal_fingerprints = {
        ticker: _v2_json_fingerprint(
            canonical_v2_signal_packet_snapshot(normalized_signals[ticker])
        )
        for ticker in ordered_tickers
        if ticker in normalized_signals
    }
    company_fingerprints = {
        str(packet.ticker).strip().upper(): _v2_json_fingerprint(packet.to_dict())
        for packet in company_packets
    }
    scenario_rows = [scenario.to_dict() for scenario in scenarios]
    scenarios_by_ticker = {
        ticker: [
            scenario.to_dict()
            for scenario in scenarios
            if str(scenario.ticker).strip().upper() == ticker
        ]
        for ticker in ordered_tickers
    }
    cohort_fingerprint = _v2_json_fingerprint(
        {
            "sector": sector,
            "as_of_date": as_of_date,
            "ordered_tickers": ordered_tickers,
            "signal_packet_fingerprints": signal_fingerprints,
            "company_packet_fingerprints": company_fingerprints,
            "expected_return_scenarios": scenario_rows,
        }
    )
    return {
        ticker: {
            "artifact_type": "v1_canonical_child_source_binding_v1",
            "pipeline_version": "v1",
            "sector": sector,
            "ticker": ticker,
            "as_of_date": as_of_date,
            "signal_packet_fingerprint": signal_fingerprints[ticker],
            "cohort_signal_packet_fingerprints": dict(signal_fingerprints),
            "company_packet_fingerprint": company_fingerprints[ticker],
            # Carry the exact deterministic financial inputs into the child
            # lane.  Fingerprints alone prove identity only if the child can
            # independently retrieve the same immutable object; embedding the
            # payload lets the child validate and prompt from the parent run's
            # actual packet without consulting a mutable latest-artifact path.
            "financial_integrity_packet": next(
                packet.to_dict()
                for packet in company_packets
                if str(packet.ticker).strip().upper() == ticker
            ),
            "financial_integrity_scenarios": scenarios_by_ticker[ticker],
            "financial_integrity_scenarios_fingerprint": _v2_json_fingerprint(
                scenarios_by_ticker[ticker]
            ),
            "cohort_fingerprint": cohort_fingerprint,
            "cohort_tickers": list(ordered_tickers),
        }
        for ticker in ordered_tickers
        if ticker in signal_fingerprints and ticker in company_fingerprints
    }


def _v2_child_source_bindings(
    *,
    sector: str,
    as_of_date: str,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    signal_packets: dict[str, TickerSignalPacket],
    frontier_candidate_tickers: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Bind child lanes to the complete packet cohort and live frontier order."""

    normalized_signal_packets = {
        str(ticker).strip().upper(): packet
        for ticker, packet in signal_packets.items()
        if str(ticker).strip()
    }
    signal_snapshots = {
        ticker: canonical_v2_signal_packet_snapshot(normalized_signal_packets[ticker])
        for ticker in [str(packet.ticker).upper() for packet in company_packets]
        if ticker in normalized_signal_packets
    }
    return build_v2_canonical_child_source_bindings(
        sector=sector,
        as_of_date=as_of_date,
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packet_snapshots=signal_snapshots,
        frontier_candidate_tickers=frontier_candidate_tickers,
    )


def _run_v2_company_underwriting_child(
    *,
    sector: str,
    ticker: str,
    as_of_date: str,
    allowed_tools: list[str],
    objective: str | None = None,
    execution_lane: str = "company_underwriting",
    canonical_signal_packet: TickerSignalPacket,
    source_binding: dict[str, Any],
) -> tuple[dict[str, Any], AutonomousRunArtifact | None, list[str]]:
    """Run one progressive child: 4 tools/2 turns, then at most 8/3 total."""

    notes: list[str] = []
    try:
        if not isinstance(canonical_signal_packet, TickerSignalPacket) or not isinstance(
            source_binding, dict
        ):
            raise ValueError(
                "v2 company underwriting requires the canonical signal packet and source binding"
            )
        artifact = run_single_candidate_autonomous_analysis(
            ticker,
            objective=objective or _company_autonomy_objective(sector=sector, ticker=ticker),
            as_of_date=as_of_date,
            budget=_company_autonomy_child_budget(extended=True),
            initial_budget=_company_autonomy_child_budget(extended=False),
            allowed_tools=allowed_tools,
            execution_lane=execution_lane,
            canonical_signal_packet=canonical_signal_packet,
            source_binding=source_binding,
        )
    except Exception as exc:  # noqa: BLE001 - one child cannot abort the sector
        provider_usage = attached_provider_usage_records(exc)
        notes.append(
            f"Company underwriting attempt for {ticker} failed: {type(exc).__name__}: {exc}"
        )
        return (
            {
                "ticker": ticker,
                "status": "FAILED",
                "final_verdict": None,
                "selected_ticker": None,
                "confidence": None,
                "degraded_states": ["COMPANY_AUTONOMY_CHILD_FAILED"],
                "error": f"{type(exc).__name__}: {exc}",
                "extension_attempted": False,
                "attempts": [],
                "tool_call_attempts": 0,
                "nonphysical_tool_call_count": 0,
                "nonphysical_tool_call_status_counts": {},
                "provider_usage": provider_usage,
                "source_binding": dict(source_binding or {}),
            },
            None,
            notes,
        )
    compact = _compact_company_autonomy_run(artifact)
    extension_attempted = any(
        str(note).startswith("Progressive autonomous budget extended from ")
        for note in artifact.audit_notes
    )
    if extension_attempted:
        notes.append(
            f"Extended {ticker} company underwriting to eight tools/three turns because required evidence remained unresolved."
        )
    compact["extension_attempted"] = extension_attempted
    compact["attempts"] = [artifact.to_dict()]
    compact["tool_call_attempts"] = len(
        [call for call in artifact.tool_calls if call.status in {"OK", "ERROR"}]
    )
    compact["provider_usage"] = [dict(usage) for usage in artifact.provider_usage]
    compact["source_binding"] = dict(source_binding or {})
    return compact, artifact, notes


def _v2_completed_underwriting_review(run: dict[str, Any]) -> bool:
    verdict = _v2_normalized_underwriting_verdict(run.get("final_verdict"))
    return verdict in {"ACTIONABLE", "WATCHLIST_ONLY", "AVOID"} and not (
        _v2_child_underwriting_gap_reasons(run, verdict=verdict)
    )


def _v2_screen_passing_frontier_packets(
    *,
    company_packets: list[SectorCompanyFinancialPacket],
    candidate_selection: dict[str, Any] | None,
) -> list[SectorCompanyFinancialPacket]:
    gate_rows = (
        candidate_selection.get("structural_gate_results")
        if isinstance(candidate_selection, dict)
        and isinstance(candidate_selection.get("structural_gate_results"), dict)
        else None
    )
    eligible: list[SectorCompanyFinancialPacket] = []
    for packet in company_packets:
        ticker = packet.ticker.upper()
        if gate_rows is not None:
            gate_row = gate_rows.get(ticker)
            screen = gate_row.get("screen_result") if isinstance(gate_row, dict) else None
            if not isinstance(screen, dict) or str(screen.get("status") or "").upper() != "PASS":
                continue
        if not _packet_has_price_and_valuation(packet):
            continue
        eligible.append(packet)
    return eligible


def _v2_operational_frontier_payload(
    state: Any,
    *,
    attempted_tickers: set[str],
) -> dict[str, Any]:
    """Attach review-attempt truth to one independently rebuilt frontier state."""

    candidate_order = [row.ticker for row in state.candidates]
    candidate_set = set(candidate_order)
    reviewed = set(state.reviewed_tickers)
    attempted = (set(attempted_tickers) | reviewed) & candidate_set
    failed = attempted - reviewed
    minimum_required = min(3, len(candidate_order))
    payload = state.to_dict()
    closure = dict(payload["closure_certificate"])
    reason_codes = list(closure.get("reason_codes") or [])
    if len(reviewed) < minimum_required:
        reason_codes.append("MINIMUM_UNDERWRITING_REVIEWS_INCOMPLETE")
    if failed:
        reason_codes.append("COMPANY_UNDERWRITING_ATTEMPTS_INCOMPLETE")
    closure["reason_codes"] = list(dict.fromkeys(reason_codes))
    closure["closed"] = not closure["reason_codes"]
    closure["status"] = "CLOSED" if closure["closed"] else "OPEN"
    payload.update(
        {
            "status": closure["status"],
            "minimum_reviews_required": minimum_required,
            "successful_review_count": len(reviewed),
            "attempted_tickers": [ticker for ticker in candidate_order if ticker in attempted],
            "failed_review_tickers": [ticker for ticker in candidate_order if ticker in failed],
            "closure_certificate": closure,
        }
    )
    return payload


def _v2_completed_carried_review_tickers(
    candidate_selection: dict[str, Any],
) -> set[str]:
    delta_audit = candidate_selection.get("delta_audit")
    carried = (
        delta_audit.get("carried_verdicts")
        if isinstance(delta_audit, dict) and isinstance(delta_audit.get("carried_verdicts"), dict)
        else {}
    )
    completed: set[str] = set()
    for raw_ticker, raw in carried.items():
        if not isinstance(raw, dict):
            continue
        result = raw.get("underwriting_result")
        verdict = _v2_normalized_underwriting_verdict(raw.get("verdict"))
        if (
            isinstance(result, dict)
            and str(result.get("status") or "").strip().upper() == "COMPLETED"
            and _v2_normalized_underwriting_verdict(result.get("verdict")) == verdict
            and verdict in {"ACTIONABLE", "WATCHLIST_ONLY", "AVOID"}
        ):
            completed.add(str(raw_ticker).strip().upper())
    return completed


def _rebuild_v2_competitive_frontier_before_finalization(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> None:
    """Re-certify the frontier from final packet/scenario inputs before v2 projection."""

    selection = dict(artifact.candidate_selection or {})
    final_input_tickers = {
        *(str(packet.ticker).strip().upper() for packet in artifact.company_packets),
        *(str(scenario.ticker).strip().upper() for scenario in artifact.expected_return_scenarios),
    }
    authorized_execution = (
        set(_normalized_ticker_values(selection.get("execution_tickers")))
        if bool(selection.get("execution_bound_frozen"))
        else set(final_input_tickers)
    )
    if bool(selection.get("execution_bound_frozen")) and not final_input_tickers.issubset(
        authorized_execution
    ):
        raise ValueError("final v2 frontier inputs widened beyond the frozen execution set")
    prior_frontier = (
        artifact.competitive_frontier if isinstance(artifact.competitive_frontier, dict) else {}
    )
    eligible_packets = _v2_screen_passing_frontier_packets(
        company_packets=artifact.company_packets,
        candidate_selection=selection,
    )
    eligible_tickers = {packet.ticker.strip().upper() for packet in eligible_packets}
    eligible_scenarios = [
        scenario
        for scenario in artifact.expected_return_scenarios
        if scenario.ticker.strip().upper() in eligible_tickers
    ]
    attempted: set[str] = set()
    reviewed: set[str] = set()
    for run in artifact.company_autonomy_runs:
        ticker = str(run.get("ticker") or run.get("selected_ticker") or "").strip().upper()
        if not ticker or ticker not in eligible_tickers:
            continue
        attempted.add(ticker)
        if _v2_completed_underwriting_review(run):
            reviewed.add(ticker)
    carried_reviewed = _v2_completed_carried_review_tickers(selection) & eligible_tickers
    reviewed.update(carried_reviewed)
    attempted.update(carried_reviewed)
    rebuilt = build_competitive_frontier(
        eligible_packets,
        eligible_scenarios,
        reviewed_tickers=reviewed,
        top_n=SECTOR_PROMPT_CANDIDATE_LIMIT,
        batch_size=3,
    )
    payload = _v2_operational_frontier_payload(
        rebuilt,
        attempted_tickers=attempted,
    )
    payload["source_binding"] = "FINAL_ARTIFACT_COMPANY_PACKETS_AND_EXPECTED_RETURN_SCENARIOS"
    payload["source_bindings"] = dict(prior_frontier.get("source_bindings") or {})
    payload["cohort_fingerprint"] = prior_frontier.get("cohort_fingerprint")
    payload["execution_fingerprint"] = selection.get("execution_fingerprint")
    payload["authorized_execution_tickers"] = sorted(authorized_execution)
    artifact.competitive_frontier = payload
    selection["competitive_frontier"] = payload
    artifact.candidate_selection = selection
    artifact.company_autonomy_status = str(payload.get("status") or "OPEN")
    artifact.audit_notes = list(
        dict.fromkeys(
            [
                *artifact.audit_notes,
                "Rebuilt the v2 competitive frontier from final artifact packet and expected-return inputs before truth projection.",
            ]
        )
    )


def _run_v2_competitive_frontier(
    *,
    sector: str,
    as_of_date: str,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    candidate_selection: dict[str, Any] | None,
    signal_packets: dict[str, TickerSignalPacket],
    allowed_tools: list[str],
    evidence_start_index: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[EvidenceReference],
    list[str],
]:
    """Review the minimum cohort, then every unresolved nondominated candidate."""

    packet_tickers = {str(packet.ticker).strip().upper() for packet in company_packets}
    scenario_tickers = {str(item.ticker).strip().upper() for item in scenarios}
    signal_tickers = {
        str(ticker).strip().upper() for ticker in signal_packets if str(ticker).strip()
    }
    bound_is_frozen = bool((candidate_selection or {}).get("execution_bound_frozen"))
    authorized_execution = (
        set(_normalized_ticker_values((candidate_selection or {}).get("execution_tickers")))
        if bound_is_frozen
        else packet_tickers | scenario_tickers | signal_tickers
    )
    outside_authority = sorted(
        (packet_tickers | scenario_tickers | signal_tickers) - authorized_execution
    )
    if bound_is_frozen and outside_authority:
        raise ValueError(
            "v2 competitive frontier input widened beyond the frozen execution set: "
            + ",".join(outside_authority)
        )

    eligible_packets = _v2_screen_passing_frontier_packets(
        company_packets=company_packets,
        candidate_selection=candidate_selection,
    )
    eligible_tickers = {packet.ticker.upper() for packet in eligible_packets}
    eligible_scenarios = [item for item in scenarios if item.ticker.upper() in eligible_tickers]
    state = build_competitive_frontier(
        eligible_packets,
        eligible_scenarios,
        top_n=SECTOR_PROMPT_CANDIDATE_LIMIT,
        batch_size=3,
    )
    child_tools = _company_autonomy_allowed_tools(allowed_tools)
    runs: list[dict[str, Any]] = []
    evidence: list[EvidenceReference] = []
    notes: list[str] = []
    attempted: set[str] = set()
    reviewed: set[str] = set()
    next_evidence_index = evidence_start_index
    source_bindings = _v2_child_source_bindings(
        sector=sector,
        as_of_date=as_of_date,
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        frontier_candidate_tickers=[row.ticker for row in state.candidates],
    )
    canonical_signal_packets = {
        str(ticker).strip().upper(): packet
        for ticker, packet in signal_packets.items()
        if str(ticker).strip()
    }

    def run_batch(tickers: list[str]) -> None:
        nonlocal next_evidence_index
        for ticker in tickers:
            attempted.add(ticker)
            canonical_packet = canonical_signal_packets.get(ticker)
            source_binding = source_bindings.get(ticker)
            if canonical_packet is None or source_binding is None:
                runs.append(
                    {
                        "ticker": ticker,
                        "status": "FAILED",
                        "final_verdict": None,
                        "selected_ticker": None,
                        "confidence": None,
                        "degraded_states": ["CANONICAL_SIGNAL_PACKET_MISSING"],
                        "error": (
                            "V2 competitive-frontier underwriting refused to read a "
                            "global latest packet when its canonical packet binding was missing."
                        ),
                        "extension_attempted": False,
                        "attempts": [],
                        "tool_call_attempts": 0,
                        "nonphysical_tool_call_count": 0,
                        "nonphysical_tool_call_status_counts": {},
                        "provider_usage": [],
                        "source_binding": {},
                    }
                )
                notes.append(
                    f"Company underwriting attempt for {ticker} was not started because its canonical v2 packet binding was missing."
                )
                continue
            run, artifact, child_notes = _run_v2_company_underwriting_child(
                sector=sector,
                ticker=ticker,
                as_of_date=as_of_date,
                allowed_tools=child_tools,
                canonical_signal_packet=canonical_packet,
                source_binding=source_binding,
            )
            runs.append(run)
            notes.extend(child_notes)
            if artifact is not None:
                child_evidence = _merge_company_autonomy_evidence(
                    child_artifact=artifact,
                    evidence_start_index=next_evidence_index,
                )
                evidence.extend(child_evidence)
                next_evidence_index += len(child_evidence)
            if _v2_completed_underwriting_review(run):
                reviewed.add(ticker)

    minimum_required = min(3, len(state.candidates))
    if not child_tools and state.candidates:
        notes.append("No allowed company-underwriting tools were available.")
    while child_tools and len(reviewed) < minimum_required:
        backfill = [row.ticker for row in state.candidates if row.ticker not in attempted][:3]
        if not backfill:
            break
        run_batch(backfill)

    while child_tools:
        cursor = state.mark_reviewed(reviewed)
        pending = [ticker for ticker in cursor.pending_tickers if ticker not in attempted]
        if not pending:
            break
        run_batch(pending[:3])

    final_state = state.mark_reviewed(reviewed)
    payload = _v2_operational_frontier_payload(
        final_state,
        attempted_tickers=attempted,
    )
    payload["source_bindings"] = source_bindings
    payload["signal_packet_snapshots"] = {
        ticker: canonical_v2_signal_packet_snapshot(packet)
        for ticker, packet in canonical_signal_packets.items()
        if ticker in {str(item.ticker).strip().upper() for item in company_packets}
    }
    payload["cohort_fingerprint"] = (
        next(iter(source_bindings.values())).get("cohort_fingerprint") if source_bindings else None
    )
    payload["execution_fingerprint"] = (candidate_selection or {}).get("execution_fingerprint")
    payload["authorized_execution_tickers"] = sorted(authorized_execution)
    closure = dict(payload["closure_certificate"])
    notes.append(
        "Competitive frontier "
        f"{closure['status']}: {len(reviewed)}/{minimum_required} minimum reviews; "
        f"{len(final_state.pending_tickers)} nondominated candidate(s) pending."
    )
    return payload, runs, evidence, notes


def _validation_audit_from_attempts(
    run: dict[str, Any],
) -> tuple[list[ToolCallRecord], list[EvidenceReference], list[str]]:
    """Return self-contained, namespaced challenge calls and evidence."""

    attempts = run.get("attempts") if isinstance(run.get("attempts"), list) else []
    if not attempts and isinstance(run.get("artifact"), dict):
        attempts = [run["artifact"]]
    records: list[ToolCallRecord] = []
    evidence: list[EvidenceReference] = []
    evidence_id_maps: list[dict[str, str]] = []
    for attempt_index, raw_artifact in enumerate(attempts, start=1):
        if not isinstance(raw_artifact, dict):
            evidence_id_maps.append({})
            continue
        run_id = (
            str(
                (raw_artifact.get("request") or {}).get("run_id")
                if isinstance(raw_artifact.get("request"), dict)
                else ""
            ).strip()
            or str(run.get("run_id") or "validator").strip()
        )
        namespace = f"{run_id}:attempt{attempt_index}"
        tool_id_map = {
            str(raw_call.get("call_id")): f"{namespace}:{raw_call.get('call_id')}"
            for raw_call in raw_artifact.get("tool_calls") or []
            if isinstance(raw_call, dict) and raw_call.get("call_id")
        }
        evidence_id_map = {
            str(raw_evidence.get("evidence_id")): (f"{namespace}:{raw_evidence.get('evidence_id')}")
            for raw_evidence in raw_artifact.get("evidence") or []
            if isinstance(raw_evidence, dict) and raw_evidence.get("evidence_id")
        }
        evidence_id_maps.append(evidence_id_map)
        for raw_call in raw_artifact.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            call = ToolCallRecord.from_dict(raw_call)
            call.call_id = tool_id_map.get(call.call_id, f"{namespace}:{call.call_id}")
            call.evidence_ref_ids = [
                evidence_id_map[ref] for ref in call.evidence_ref_ids if ref in evidence_id_map
            ]
            if call.lane is None:
                call.lane = "selected_company_validation"
            records.append(call)
        for raw_evidence in raw_artifact.get("evidence") or []:
            if not isinstance(raw_evidence, dict):
                continue
            item = EvidenceReference.from_dict(raw_evidence)
            item.evidence_id = evidence_id_map[item.evidence_id]
            if item.tool_call_id:
                item.tool_call_id = tool_id_map.get(
                    item.tool_call_id,
                    f"{namespace}:{item.tool_call_id}",
                )
            evidence.append(item)

    final_refs: list[str] = []
    if evidence_id_maps:
        final_map = evidence_id_maps[-1]
        final_run_id = str(run.get("run_id") or "").strip()
        for raw_ref in _v2_child_decision_evidence_ids(run):
            prefix = f"{final_run_id}:" if final_run_id else ""
            source_id = raw_ref[len(prefix) :] if prefix and raw_ref.startswith(prefix) else raw_ref
            if source_id in final_map:
                final_refs.append(final_map[source_id])
    return records, evidence, list(dict.fromkeys(final_refs))


def _validation_provider_usage_from_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    usage: list[dict[str, Any]] = []
    for index, raw in enumerate(run.get("provider_usage") or [], start=1):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        original_id = str(row.get("provider_call_id") or f"P{index}")
        row["provider_call_id"] = f"{str(run.get('run_id') or 'validator')}:{original_id}:{index}"
        row["validator_run_id"] = str(run.get("run_id") or "validator")
        row["lane"] = "selected_company_validation"
        usage.append(row)
    return usage


def _run_v2_selected_company_validation(
    *,
    sector: str,
    ticker: str | None,
    as_of_date: str,
    allowed_tools: list[str],
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    signal_packets: dict[str, TickerSignalPacket],
    expected_source_binding: dict[str, Any] | None,
) -> SectorSelectionValidation:
    """Run a separate adversarial challenge for the provisional selected company."""

    selected = str(ticker or "").strip().upper() or None
    if selected is None:
        return SectorSelectionValidation(status="NOT_REQUIRED")
    current_bindings = _v2_child_source_bindings(
        sector=sector,
        as_of_date=as_of_date,
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        frontier_candidate_tickers=(
            list(expected_source_binding.get("frontier_candidate_tickers") or [])
            if isinstance(expected_source_binding, dict)
            else None
        ),
    )
    canonical_signal_packets = {
        str(candidate_ticker).strip().upper(): packet
        for candidate_ticker, packet in signal_packets.items()
        if str(candidate_ticker).strip()
    }
    current_source_binding = current_bindings.get(selected)
    if (
        not isinstance(expected_source_binding, dict)
        or not isinstance(current_source_binding, dict)
        or current_source_binding != expected_source_binding
    ):
        return SectorSelectionValidation(
            status="INCOMPLETE",
            selected_ticker=selected,
            reason_codes=["SELECTED_COMPANY_CANONICAL_SOURCE_BINDING_DRIFT"],
            notes=[
                "Selected-company validation refused packet, cohort, or as-of drift from the competitive frontier."
            ],
            source_binding=dict(current_source_binding or {}),
        )
    child_tools = _company_autonomy_allowed_tools(allowed_tools)
    if not child_tools:
        return SectorSelectionValidation(
            status="NOT_ATTEMPTED",
            selected_ticker=selected,
            reason_codes=["SELECTED_COMPANY_VALIDATION_TOOLS_UNAVAILABLE"],
            notes=["The selected-company challenge had no allowed company research tools."],
        )

    run, artifact, notes = _run_v2_company_underwriting_child(
        sector=sector,
        ticker=selected,
        as_of_date=as_of_date,
        allowed_tools=child_tools,
        objective=_selected_company_challenge_objective(
            sector=sector,
            ticker=selected,
        ),
        execution_lane="selected_company_validation",
        canonical_signal_packet=canonical_signal_packets[selected],
        source_binding=current_source_binding,
    )
    verdict = _v2_normalized_underwriting_verdict(run.get("final_verdict"))
    gaps = _v2_child_underwriting_gap_reasons(run, verdict=verdict)
    tool_calls, validation_evidence, evidence_ids = _validation_audit_from_attempts(run)
    provider_usage = _validation_provider_usage_from_run(run)
    validator_run_id = str(run.get("run_id") or "").strip() or (
        artifact.request.run_id if artifact is not None else None
    )

    if not gaps and verdict == "ACTIONABLE":
        status = "VALIDATED"
        reason_codes = ["SELECTED_COMPANY_CHALLENGE_PASSED"]
        validation_notes = [
            *notes,
            "Independent selected-company challenge affirmed the actionable underwriting.",
        ]
    elif not gaps and verdict in {"WATCHLIST_ONLY", "AVOID"}:
        status = "CONTRADICTED"
        reason_codes = [f"SELECTED_COMPANY_CHALLENGE_{verdict}"]
        validation_notes = [
            *notes,
            "Independent selected-company challenge did not affirm actionable underwriting.",
        ]
    else:
        status = "INCOMPLETE"
        reason_codes = gaps or ["SELECTED_COMPANY_VALIDATION_INCOMPLETE"]
        validation_notes = [
            *notes,
            "Independent selected-company challenge did not resolve all required evidence.",
        ]

    validation_payload = {
        "status": status,
        "selected_ticker": selected,
        "validator_run_id": validator_run_id,
        "validator_verdict": verdict,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "evidence_ref_ids": evidence_ids,
        "evidence": [item.to_dict() for item in validation_evidence],
        "tool_calls": [item.to_dict() for item in tool_calls],
        "provider_usage": provider_usage,
        "notes": list(dict.fromkeys(validation_notes)),
        "source_binding": current_source_binding,
    }
    return SectorSelectionValidation(
        **{
            **validation_payload,
            "evidence": validation_evidence,
            "tool_calls": tool_calls,
        },
        terminal_ledger_fingerprint=(
            selection_validation_terminal_ledger_fingerprint(validation_payload)
            if status in {"VALIDATED", "CONTRADICTED"}
            else None
        ),
    )


def _v2_lane_budget_payload(
    *,
    run_budget: AutonomousRunBudget,
    competitive_frontier: dict[str, Any],
) -> dict[str, Any]:
    """Persist independent operational envelopes without a per-sector dollar ceiling."""

    candidate_count = len(
        competitive_frontier.get("candidates")
        if isinstance(competitive_frontier.get("candidates"), list)
        else []
    )
    return {
        "artifact_type": "autonomous_sector_operational_lane_budget_v2",
        "dollar_ceiling_usd": None,
        "whole_run_cost_preflight_required": True,
        "lanes": {
            "provider_preflight": {
                "scope": "whole_run",
                "max_tool_calls": 0,
                "max_turns": 1,
            },
            "parent_research": {
                "scope": "per_sector",
                "max_tool_calls": max(0, int(run_budget.max_tool_calls)),
                "max_turns": max(0, int(run_budget.max_turns)),
            },
            "company_underwriting": {
                "scope": "per_sector",
                "candidate_count": candidate_count,
                "initial_child": INITIAL_COMPANY_CHILD_BUDGET.to_dict(),
                "extended_child": EXTENDED_COMPANY_CHILD_BUDGET.to_dict(),
                "max_tool_calls": candidate_count * EXTENDED_COMPANY_CHILD_BUDGET.max_tool_calls,
                "max_turns": candidate_count * EXTENDED_COMPANY_CHILD_BUDGET.max_turns,
            },
            "selected_company_validation": {
                "scope": "per_sector",
                "initial_child": INITIAL_COMPANY_CHILD_BUDGET.to_dict(),
                "extended_child": EXTENDED_COMPANY_CHILD_BUDGET.to_dict(),
                "max_tool_calls": EXTENDED_COMPANY_CHILD_BUDGET.max_tool_calls,
                "max_turns": EXTENDED_COMPANY_CHILD_BUDGET.max_turns,
            },
            "repair_fallback": {
                "scope": "per_sector",
                "max_tool_calls": max(0, int(run_budget.max_tool_calls)),
                "max_turns": max(0, int(run_budget.max_turns)),
            },
            "terminal_cap_search": {
                "scope": "whole_run",
                "checkpoint_batch_size": 25,
                "max_tool_calls": None,
                "max_turns": None,
            },
        },
    }


def _usage_microdollars(value: Any) -> int:
    try:
        amount = Decimal(str(value or 0))
    except (InvalidOperation, TypeError, ValueError):
        return 0
    if not amount.is_finite() or amount <= 0:
        return 0
    return int(
        (amount * Decimal(1_000_000)).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _empty_v2_lane_usage() -> dict[str, int]:
    return {
        "tool_call_attempts": 0,
        "tool_calls_ok": 0,
        "tool_calls_failed": 0,
        "provider_call_attempts": 0,
        "provider_calls_ok": 0,
        "provider_calls_failed": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reserved_output_tokens": 0,
        "cost_microdollars": 0,
    }


def _v2_lane_usage_payload(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> dict[str, Any]:
    """Reconcile every tool/provider attempt, including nested and failed calls."""

    lanes = {lane: _empty_v2_lane_usage() for lane in CANONICAL_LANES}
    nonphysical_statuses = {lane: Counter() for lane in CANONICAL_LANES}
    nonphysical_reserve_exposure = {lane: 0 for lane in CANONICAL_LANES}

    def add_tool(call: Any, *, default_lane: str) -> None:
        lane = str(
            call.get("lane")
            if isinstance(call, dict)
            else getattr(call, "lane", None) or default_lane
        )
        if lane not in lanes:
            lane = default_lane
        status = str(
            call.get("status")
            if isinstance(call, dict)
            else getattr(call, "status", None) or "UNKNOWN"
        ).upper()
        if status not in {"OK", "ERROR"}:
            nonphysical_statuses[lane][status] += 1
            return
        totals = lanes[lane]
        totals["tool_call_attempts"] += 1
        totals["tool_calls_ok" if status == "OK" else "tool_calls_failed"] += 1
        if isinstance(call, dict):
            totals["cost_microdollars"] += _usage_microdollars(call.get("cost_estimate_usd"))

    def add_provider(row: dict[str, Any], *, default_lane: str) -> None:
        lane = str(row.get("lane") or default_lane)
        if lane not in lanes:
            lane = default_lane
        totals = lanes[lane]
        status = str(row.get("status") or row.get("attempt_status") or "UNKNOWN").upper()
        totals["provider_call_attempts"] += 1
        totals[
            "provider_calls_ok"
            if status in {"OK", "RESOLVED", "INCOMPLETE"}
            else "provider_calls_failed"
        ] += 1
        for field_name in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reserved_output_tokens",
        ):
            value = row.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                totals[field_name] += value
        totals["cost_microdollars"] += _usage_microdollars(row.get("cost_estimate_usd"))

    for call in artifact.tool_calls:
        question_id = str(call.question_id or "").upper()
        add_tool(
            call,
            default_lane=(
                "repair_fallback" if question_id.startswith(("AGR", "WR")) else "parent_research"
            ),
        )
    for row in artifact.provider_usage:
        if isinstance(row, dict):
            add_provider(row, default_lane="parent_research")

    for run in artifact.company_autonomy_runs:
        if not isinstance(run, dict):
            continue
        attempts = run.get("attempts") if isinstance(run.get("attempts"), list) else None
        artifacts = attempts or ([run["artifact"]] if isinstance(run.get("artifact"), dict) else [])
        for nested in artifacts:
            if not isinstance(nested, dict):
                continue
            for call in nested.get("tool_calls") or []:
                if isinstance(call, dict):
                    add_tool(call, default_lane="company_underwriting")
        for row in run.get("provider_usage") or []:
            if isinstance(row, dict):
                add_provider(row, default_lane="company_underwriting")

    validation = artifact.selection_validation
    if validation is not None:
        for call in validation.tool_calls:
            add_tool(call, default_lane="selected_company_validation")
        for row in validation.provider_usage:
            if isinstance(row, dict):
                add_provider(row, default_lane="selected_company_validation")

    repair = (
        artifact.candidate_selection.get("data_gap_repair")
        if isinstance(artifact.candidate_selection, dict)
        else None
    )
    usage_records = (
        repair.get("terminal_cap_search_usage_records")
        if isinstance(repair, dict)
        and isinstance(repair.get("terminal_cap_search_usage_records"), list)
        else []
    )
    for row in usage_records:
        if not isinstance(row, dict):
            continue
        if row.get("call_type") == "web_search_call":
            add_tool(
                {
                    **row,
                    "status": (
                        "ERROR"
                        if str(row.get("attempt_status") or "").upper() == "FAILED"
                        else "OK"
                    ),
                },
                default_lane="terminal_cap_search",
            )
        elif row.get("call_type") == "web_search_call_reserve":
            # A held reserve is neither a dispatched attempt nor realized
            # cost.  Preserve it as nonphysical exposure without contaminating
            # actual benchmark call/cost totals.
            nonphysical_statuses["terminal_cap_search"]["RESERVED_NOT_DISPATCHED"] += 1
            nonphysical_reserve_exposure["terminal_cap_search"] += _usage_microdollars(
                row.get("cost_estimate_usd")
            )
        elif row.get("call_type") == "responses_model":
            add_provider(row, default_lane="terminal_cap_search")

    aggregate = _empty_v2_lane_usage()
    for totals in lanes.values():
        for field_name in aggregate:
            aggregate[field_name] += totals[field_name]

    def with_cost(totals: dict[str, int]) -> dict[str, Any]:
        return {
            **totals,
            "cost_estimate_usd": f"{Decimal(totals['cost_microdollars']) / Decimal(1_000_000):.6f}",
        }

    payload = {
        "artifact_type": "autonomous_sector_lane_usage_v2",
        "currency": "USD",
        "cost_unit": "microdollars",
        "lanes": {lane: with_cost(lanes[lane]) for lane in CANONICAL_LANES},
        "aggregate": with_cost(aggregate),
        "nonphysical_tool_diagnostics": {
            "lanes": {
                lane: {
                    "record_count": sum(nonphysical_statuses[lane].values()),
                    "status_counts": dict(sorted(nonphysical_statuses[lane].items())),
                    "reserved_cost_exposure_microdollars": nonphysical_reserve_exposure[lane],
                }
                for lane in CANONICAL_LANES
            },
            "aggregate": {
                "record_count": sum(
                    sum(nonphysical_statuses[lane].values()) for lane in CANONICAL_LANES
                ),
                "status_counts": dict(
                    sorted(
                        sum(
                            (nonphysical_statuses[lane] for lane in CANONICAL_LANES),
                            Counter(),
                        ).items()
                    )
                ),
                "reserved_cost_exposure_microdollars": sum(nonphysical_reserve_exposure.values()),
            },
        },
    }
    payload["aggregate_reconciles"] = all(
        aggregate[field_name] == sum(lanes[lane][field_name] for lane in CANONICAL_LANES)
        for field_name in aggregate
    )
    return payload


def _run_company_autonomy_pass(
    *,
    sector: str,
    as_of_date: str,
    research_questions: list[SectorResearchQuestion],
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    signal_packets: dict[str, TickerSignalPacket],
    candidate_selection: dict[str, Any] | None,
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    executed_tool_count: int,
    evidence_start_index: int,
) -> tuple[bool, str | None, list[str], list[dict[str, Any]], list[EvidenceReference], int]:
    capacity = _company_autonomy_budget_capacity(
        budget=budget,
        executed_tool_count=executed_tool_count,
    )
    if capacity <= 0:
        return False, None, [], [], [], executed_tool_count
    child_allowed_tools = _company_autonomy_allowed_tools(allowed_tools)
    if not child_allowed_tools:
        return (
            True,
            "NO_ALLOWED_COMPANY_TOOLS",
            ["No allowed company-level tools were available."],
            [],
            [],
            executed_tool_count,
        )

    tickers = _company_autonomy_candidates(
        research_questions=research_questions,
        company_packets=company_packets,
        scenarios=scenarios,
        candidate_selection=candidate_selection,
        limit=capacity,
    )
    if not tickers:
        return (
            True,
            "NO_ELIGIBLE_CANDIDATES",
            ["No eligible company candidates were available for nested autonomy."],
            [],
            [],
            executed_tool_count,
        )

    notes: list[str] = [
        f"Nested company autonomy selected {', '.join(tickers)} for bounded company-level underwriting."
    ]
    canonical_signal_packets = {
        str(ticker).strip().upper(): packet
        for ticker, packet in signal_packets.items()
        if str(ticker).strip()
    }
    source_bindings = _v1_child_source_bindings(
        sector=sector,
        as_of_date=as_of_date,
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=canonical_signal_packets,
    )
    child_runs: list[dict[str, Any]] = []
    merged_evidence: list[EvidenceReference] = []
    next_evidence_index = evidence_start_index
    for ticker in tickers:
        canonical_signal_packet = canonical_signal_packets.get(ticker)
        source_binding = source_bindings.get(ticker)
        if canonical_signal_packet is None or source_binding is None:
            notes.append(
                f"Nested company autonomy skipped {ticker}: canonical parent packet binding was unavailable."
            )
            child_runs.append(
                {
                    "ticker": ticker,
                    "status": "FAILED",
                    "final_verdict": "NO_WINNER",
                    "selected_ticker": None,
                    "confidence": None,
                    "degraded_states": ["NEEDS_DATA"],
                    "error": "canonical parent packet binding was unavailable",
                    "provider_usage": [],
                }
            )
            continue
        try:
            child_artifact = run_single_candidate_autonomous_analysis(
                ticker,
                objective=_company_autonomy_objective(sector=sector, ticker=ticker),
                as_of_date=as_of_date,
                budget=_company_autonomy_child_budget(),
                allowed_tools=child_allowed_tools,
                canonical_signal_packet=canonical_signal_packet,
                source_binding=source_binding,
            )
            compact = _compact_company_autonomy_run(child_artifact)
            child_runs.append(compact)
            child_evidence = _merge_company_autonomy_evidence(
                child_artifact=child_artifact,
                evidence_start_index=next_evidence_index,
            )
            merged_evidence.extend(child_evidence)
            next_evidence_index += len(child_evidence)
            child_tool_count = len(
                [call for call in child_artifact.tool_calls if call.status == "OK"]
            )
            executed_tool_count += min(COMPANY_AUTONOMY_CHILD_MAX_TOOL_CALLS, child_tool_count)
            notes.append(
                f"Company autonomy for {ticker} completed with verdict {child_artifact.final_verdict}."
            )
        except InvalidFinancialInputError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard for live child runs
            provider_usage = attached_provider_usage_records(exc)
            child_runs.append(
                {
                    "ticker": ticker,
                    "status": "FAILED",
                    "final_verdict": None,
                    "selected_ticker": None,
                    "confidence": None,
                    "degraded_states": ["COMPANY_AUTONOMY_CHILD_FAILED"],
                    "error": str(exc),
                    "provider_usage": provider_usage,
                }
            )
            notes.append(
                f"Company autonomy for {ticker} failed without aborting the sector run: {exc}"
            )

    completed = [run for run in child_runs if run.get("status") == "COMPLETED"]
    status = "COMPLETED" if len(completed) == len(child_runs) else "PARTIAL"
    return True, status, notes, child_runs, merged_evidence, executed_tool_count


def _financial_lineage_summary(
    packet: SectorCompanyFinancialPacket,
) -> dict[str, Any]:
    """Exact unit/snapshot/formula contract supplied to every LLM prompt."""

    return {
        "market_cap": {
            "value_mm": packet.market_cap_mm,
            "unit": packet.market_cap_unit,
            "source": packet.market_cap_source,
            "effective_as_of_date": packet.market_cap_effective_as_of_date,
            "source_kind": packet.market_cap_source_kind,
            "source_name": packet.market_cap_source_name,
            "source_url": packet.market_cap_source_url,
            "confidence": packet.market_cap_confidence,
            "method": packet.market_cap_method,
            "derivation": packet.market_cap_derivation,
        },
        "cap_stage_quote": {
            "price": packet.cap_stage_price,
            "as_of_date": packet.cap_stage_price_as_of_date,
            "currency": packet.cap_stage_price_currency,
            "source": packet.cap_stage_price_source,
            "source_url": packet.cap_stage_price_source_url,
            "confidence": packet.cap_stage_price_confidence,
            "quote_snapshot_id": packet.cap_stage_quote_snapshot_id,
        },
        "valuation_quote": {
            "price": packet.current_price,
            "unit": packet.current_price_unit,
            "as_of_date": packet.current_price_as_of_date,
            "currency": packet.current_price_currency,
            "source": packet.current_price_source,
            "source_url": packet.current_price_source_url,
            "confidence": packet.current_price_confidence,
            "quote_snapshot_id": packet.quote_snapshot_id,
            "basis": packet.price_basis,
            "raw_price": packet.raw_price,
        },
        "shares": {
            "value_mm": packet.shares_outstanding_mm,
            "unit": packet.shares_unit,
            "basis": packet.shares_basis,
            "as_of_date": packet.shares_as_of_date,
            "source": packet.shares_source,
            "source_url": packet.shares_source_url,
            "issuer_quote_ratio": packet.issuer_quote_ratio,
        },
        "split": {
            "adjustment_factor": packet.split_adjustment_factor,
            "effective_date": packet.split_effective_date,
        },
        "metric_traces": packet.metric_traces,
        "financial_integrity_status": packet.financial_integrity_status,
        "financial_integrity_violations": packet.financial_integrity_violations,
    }


def _compact_packet(packet: SectorCompanyFinancialPacket) -> dict[str, Any]:
    return {
        "ticker": packet.ticker,
        "financial_status": packet.financial_status,
        "model_fit_status": packet.model_fit_status,
        "data_quality_status": packet.data_quality_status,
        "current_price": packet.current_price,
        "business_quality": packet.business_quality,
        "reinvestment": packet.reinvestment,
        "returns_on_capital": packet.returns_on_capital,
        "cash_conversion": packet.cash_conversion,
        "balance_sheet": packet.balance_sheet,
        "capital_allocation": packet.capital_allocation,
        "valuation": packet.valuation,
        "expected_return": packet.expected_return,
        "blockers": packet.blockers,
        "confidence_caps": packet.confidence_caps,
        "financial_lineage": _financial_lineage_summary(packet),
    }


def _compact_scenario(scenario: SectorExpectedReturnScenario) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "ticker": scenario.ticker,
        "scenario_name": scenario.scenario_name,
        "horizon_years": scenario.horizon_years,
        "current_price": scenario.current_price,
        "current_price_unit": scenario.current_price_unit,
        "quote_snapshot_id": scenario.quote_snapshot_id,
        "price_basis": scenario.price_basis,
        "estimated_future_value_per_share": scenario.estimated_future_value_per_share,
        "annualized_return": scenario.annualized_return,
        "metric_trace": scenario.metric_trace,
        "financial_integrity_status": scenario.financial_integrity_status,
        "financial_integrity_violations": scenario.financial_integrity_violations,
        "unsupported_assumptions": scenario.unsupported_assumptions,
        "key_sensitivities": scenario.key_sensitivities,
    }


def _sector_prompt_packet_scope(
    company_packets: list[SectorCompanyFinancialPacket],
    *,
    scenarios: list[SectorExpectedReturnScenario] | None = None,
    candidate_limit: int = SECTOR_PROMPT_CANDIDATE_LIMIT,
) -> tuple[list[SectorCompanyFinancialPacket], dict[str, Any]]:
    # V2 deterministically reorders the complete pool once, immediately after
    # data repair and before any prompt is built.  This shared helper must
    # preserve its input order so rollout-gated v1 runs retain their historical
    # prompt semantics instead of being silently reranked.
    _ = scenarios
    included_packets = list(company_packets)[:candidate_limit]
    omitted_count = max(0, len(company_packets) - len(included_packets))
    return included_packets, {
        "total_candidates": len(company_packets),
        "included_candidates": len(included_packets),
        "omitted_candidates": omitted_count,
        "selection_rule": (
            "all candidates included"
            if omitted_count == 0
            else f"first {len(included_packets)} in runtime candidate order"
        ),
        "ranked_tickers": [packet.ticker.upper() for packet in included_packets],
        "tail_summary": (
            f"and {omitted_count} other candidates with similar profile" if omitted_count else None
        ),
    }


def _scenario_summaries_for_prompt(
    scenarios: list[SectorExpectedReturnScenario],
    *,
    tickers: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    filtered = [item for item in scenarios if item.ticker.upper() in tickers]
    ranked = sorted(
        filtered,
        key=lambda item: item.annualized_return if item.annualized_return is not None else -999.0,
        reverse=True,
    )
    return [_compact_scenario(item) for item in ranked[:limit]]


def _planning_packet_summary(packet: SectorCompanyFinancialPacket) -> dict[str, Any]:
    valuation = packet.valuation or {}
    expected_return = packet.expected_return or {}
    return {
        "ticker": packet.ticker,
        "financial_status": packet.financial_status,
        "model_fit_status": packet.model_fit_status,
        "data_quality_status": packet.data_quality_status,
        "current_price": packet.current_price,
        "market_cap_category": packet.market_cap_category,
        "market_cap_source": packet.market_cap_source,
        "valuation_anchor": valuation.get("valuation_anchor"),
        "anchor_method": valuation.get("anchor_method"),
        "discount_to_anchor": valuation.get("discount_to_anchor"),
        "expected_return": expected_return,
        "blockers": packet.blockers,
        "confidence_caps": packet.confidence_caps,
        "financial_lineage": _financial_lineage_summary(packet),
    }


def _planning_scenario_summaries(
    scenarios: list[SectorExpectedReturnScenario], limit: int = 16
) -> list[dict[str, Any]]:
    base_cases = [item for item in scenarios if str(item.scenario_name).lower() == "base"]
    ranked = sorted(
        base_cases or scenarios,
        key=lambda item: item.annualized_return if item.annualized_return is not None else -999.0,
        reverse=True,
    )
    return [_compact_scenario(item) for item in ranked[:limit]]


def _artifact_candidate_tickers(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    limit: int | None = None,
) -> list[str]:
    tickers: list[str] = []
    selection = (
        artifact.candidate_selection if isinstance(artifact.candidate_selection, dict) else {}
    )
    for ticker in selection.get("selected_tickers") or []:
        token = str(ticker or "").upper()
        if token and token not in tickers:
            tickers.append(token)
    for item in artifact.relative_ranking:
        token = str(item.get("ticker") or "").upper()
        if token and token not in tickers:
            tickers.append(token)
    for packet in artifact.company_packets:
        token = packet.ticker.upper()
        if token not in tickers:
            tickers.append(token)
    return tickers[:limit] if limit is not None else tickers


def _artifact_packet_tickers(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> list[str]:
    """Exact post-filter packet roster eligible for per-ticker memo review."""
    return list(
        dict.fromkeys(
            str(packet.ticker).strip().upper()
            for packet in artifact.company_packets
            if str(packet.ticker).strip()
        )
    )


def _require_unique_candidate_packets(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> list[str]:
    """Fail before provider work when a candidate has ambiguous packet identity."""

    tickers = _artifact_packet_tickers(artifact)
    counts = {
        ticker: sum(
            1
            for packet in artifact.company_packets
            if str(packet.ticker).strip().upper() == ticker
        )
        for ticker in tickers
    }
    ambiguous = {
        ticker: count for ticker, count in counts.items() if count != 1
    }
    if ambiguous:
        rendered = ", ".join(
            f"{ticker}={count}" for ticker, count in sorted(ambiguous.items())
        )
        raise CandidateMemoCheckpointError(
            "candidate memo requires exactly one packet per ticker before "
            f"provider execution: {rendered}"
        )
    return tickers


def _memo_candidate_fallback(ticker: str, state: str) -> dict[str, Any]:
    return {
        "source": "deterministic_fallback",
        "status": "DEGRADED_STATE",
        "degraded_state": state,
        "thesis": None,
        "key_risks": [],
        "falsifiers": [],
        "open_questions": [],
    }


def _fallback_memo_body(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    state: str,
    reason: str,
) -> dict[str, Any]:
    section = {
        "source": "deterministic_fallback",
        "status": "DEGRADED_STATE",
        "degraded_state": state,
        "reason": reason,
    }
    payload = {
        "status": "DEGRADED_FALLBACK",
        "source": "deterministic_fallback",
        "generated_at": _utc_now_iso(),
        "degraded_state": state,
        "degraded_states": [state],
        "reason": reason,
        "usage": {"calls": [], "input_tokens": 0, "output_tokens": 0, "cost_estimate_usd": 0.0},
        "cohort_comparison": {**section, "paragraphs": []},
        "triage_surprises": {**section, "items": []},
        "candidates": {
            ticker: _memo_candidate_fallback(ticker, state)
            for ticker in _artifact_packet_tickers(artifact)
        },
        "generation_notes": [reason],
    }
    return _memo_body_with_degraded_state_consistency(payload)


def _fallback_source(value: dict[str, Any]) -> bool:
    source = str(value.get("source") or "").lower()
    status = str(value.get("status") or "").upper()
    return "fallback" in source or "deterministic" in source or status == "DEGRADED_STATE"


def _memo_body_with_degraded_state_consistency(memo_body: dict[str, Any]) -> dict[str, Any]:
    """Ensure fallback memo-body sections always carry top-level degraded states."""

    normalized = dict(memo_body)
    raw_degraded_states = normalized.get("degraded_states")
    existing = (
        [str(item) for item in raw_degraded_states if str(item).strip()]
        if isinstance(raw_degraded_states, list)
        else []
    )
    if existing:
        normalized["degraded_states"] = list(dict.fromkeys(existing))
        return normalized

    fallback_states: list[str] = []
    default_state = str(normalized.get("degraded_state") or "SECTOR_MEMO_BODY_FALLBACK")

    def add(marker: str) -> None:
        if marker and marker not in fallback_states:
            fallback_states.append(marker)

    cohort = normalized.get("cohort_comparison")
    if isinstance(cohort, dict) and _fallback_source(cohort):
        add(f"cohort_comparison_fallback:{cohort.get('degraded_state') or default_state}")

    triage = normalized.get("triage_surprises")
    if isinstance(triage, dict) and _fallback_source(triage):
        add(f"triage_surprises_fallback:{triage.get('degraded_state') or default_state}")

    candidates = normalized.get("candidates")
    if isinstance(candidates, dict):
        for ticker, payload in candidates.items():
            if isinstance(payload, dict) and _fallback_source(payload):
                add(
                    "candidate_thesis_fallback:"
                    f"{str(ticker).upper()}:{payload.get('degraded_state') or default_state}"
                )

    if not fallback_states and _fallback_source(normalized):
        add(default_state)

    normalized["degraded_states"] = fallback_states
    return normalized


def _memo_usage_rollup(calls: list[dict[str, Any]]) -> dict[str, Any]:
    incremental_calls = [
        item for item in calls if item.get("reused_from_checkpoint") is not True
    ]
    return {
        "calls": calls,
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in calls),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in calls),
        "cost_estimate_usd": round(
            sum(float(item.get("cost_estimate_usd") or 0.0) for item in calls), 6
        ),
        "incremental_input_tokens": sum(
            int(item.get("input_tokens") or 0) for item in incremental_calls
        ),
        "incremental_output_tokens": sum(
            int(item.get("output_tokens") or 0) for item in incremental_calls
        ),
        "incremental_cost_estimate_usd": round(
            sum(
                float(item.get("cost_estimate_usd") or 0.0)
                for item in incremental_calls
            ),
            6,
        ),
        "reused_call_count": len(calls) - len(incremental_calls),
    }


def _memo_body_usage_summary(memo_body: dict[str, Any]) -> dict[str, Any]:
    usage = memo_body.get("usage") if isinstance(memo_body, dict) else {}
    if isinstance(usage, dict):
        canonical_cost = usage.get("cost_estimate_usd")
        if isinstance(canonical_cost, (int, float)) and not isinstance(canonical_cost, bool):
            return dict(usage)
        calls = usage.get("calls")
        if isinstance(calls, list):
            return _memo_usage_rollup([dict(item) for item in calls if isinstance(item, dict)])

    calls: list[dict[str, Any]] = []
    for section in ("cohort_comparison", "triage_surprises"):
        payload = memo_body.get(section)
        section_usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(section_usage, dict):
            calls.append({"section": section, **section_usage})
    candidates = memo_body.get("candidates")
    if isinstance(candidates, dict):
        for ticker, payload in candidates.items():
            section_usage = payload.get("usage") if isinstance(payload, dict) else None
            if isinstance(section_usage, dict):
                calls.append(
                    {"section": "candidate", "ticker": str(ticker).upper(), **section_usage}
                )
    return _memo_usage_rollup(calls)


def _memo_scenario_context(
    artifact: AutonomousSectorFinancialRunArtifact, ticker: str
) -> list[dict[str, Any]]:
    return [
        _compact_scenario(scenario)
        for scenario in artifact.expected_return_scenarios
        if scenario.ticker.upper() == ticker.upper()
    ]


def _memo_ranking_context(
    artifact: AutonomousSectorFinancialRunArtifact, ticker: str
) -> dict[str, Any]:
    for item in artifact.relative_ranking:
        if str(item.get("ticker") or "").upper() == ticker.upper():
            return dict(item)
    return {}


def _memo_candidate_context(
    artifact: AutonomousSectorFinancialRunArtifact, ticker: str
) -> dict[str, Any]:
    packet_by_ticker = {packet.ticker.upper(): packet for packet in artifact.company_packets}
    packet = packet_by_ticker.get(ticker.upper())
    return {
        "ticker": ticker.upper(),
        "packet": _compact_packet(packet) if packet else None,
        "expected_return_scenarios": _memo_scenario_context(artifact, ticker),
        "relative_ranking": _memo_ranking_context(artifact, ticker),
        "company_autonomy_run": next(
            (
                dict(item)
                for item in artifact.company_autonomy_runs
                if str(item.get("ticker") or "").upper() == ticker.upper()
            ),
            None,
        ),
        "evidence_summaries": [
            {
                "evidence_id": item.evidence_id,
                "source_label": item.source_label,
                "confidence": item.confidence,
                "summary": item.summary,
            }
            for item in artifact.evidence
            if str(item.ticker or "").upper() in {"", ticker.upper()}
        ][:10],
    }


def _short_text(value: Any, *, limit: int = 240) -> str | None:
    text = " ".join(str(value or "").split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _compact_string_list(values: Any, *, limit: int = 2) -> list[str]:
    if not isinstance(values, list):
        return []
    compacted: list[str] = []
    for value in values:
        text = _short_text(value, limit=160)
        if text:
            compacted.append(text)
        if len(compacted) >= limit:
            break
    return compacted


def _compact_mapping(mapping: Any, keys: list[str]) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {
        key: mapping.get(key) for key in keys if key in mapping and mapping.get(key) is not None
    }


def _compact_packet_for_shared_memo(
    packet: SectorCompanyFinancialPacket | None,
) -> dict[str, Any] | None:
    if packet is None:
        return None
    valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    return {
        "ticker": packet.ticker,
        "financial_status": packet.financial_status,
        "model_fit_status": packet.model_fit_status,
        "data_quality_status": packet.data_quality_status,
        "current_price": packet.current_price,
        "market_cap_category": packet.market_cap_category,
        "business_quality": _compact_mapping(
            packet.business_quality,
            [
                "piotroski_f_score",
                "beneish_m_score",
                "altman_z_score",
                "normalized_operating_margin",
                "latest_operating_margin",
            ],
        ),
        "returns_on_capital": _compact_mapping(
            packet.returns_on_capital,
            ["roic", "roic_wacc_spread", "incremental_roic_3y"],
        ),
        "cash_conversion": _compact_mapping(
            packet.cash_conversion,
            ["fcf_margin", "fcf_yield", "cfo_to_net_income"],
        ),
        "valuation": {
            "anchor_method": valuation.get("anchor_method"),
            "valuation_anchor": valuation.get("valuation_anchor"),
            "discount_to_anchor": valuation.get("discount_to_anchor"),
            "fcf_yield": valuation.get("fcf_yield"),
            "buy_price_target": valuation.get("buy_price_target"),
        },
        "expected_return": _compact_mapping(
            packet.expected_return,
            [
                "base_case_annualized_return",
                "downside_annualized_return",
                "upside_annualized_return",
            ],
        ),
        "blockers": _compact_string_list(packet.blockers, limit=2),
        "confidence_caps": _compact_string_list(packet.confidence_caps, limit=2),
        "financial_lineage": _financial_lineage_summary(packet),
    }


def _compact_relative_ranking_for_shared_memo(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "rank": item.get("rank"),
        "ticker": item.get("ticker"),
        "best_base_annualized_return": item.get("best_base_annualized_return"),
        "downside_annualized_return": item.get("downside_annualized_return"),
        "audit_status": item.get("audit_status"),
        "actionable": item.get("actionable"),
        "buy_candidate": item.get("buy_candidate"),
        "cross_sectional_percentile": item.get("cross_sectional_percentile"),
        "company_autonomy_verdict": item.get("company_autonomy_verdict"),
        "confidence": item.get("confidence"),
        "hard_blockers": _compact_string_list(item.get("hard_blockers"), limit=2),
        "confidence_caps": _compact_string_list(item.get("confidence_caps"), limit=2),
        "positioning_summary": _short_text(item.get("positioning_summary"), limit=220),
    }


def _compact_selection_audit_for_shared_memo(audit: Any) -> dict[str, Any]:
    if not isinstance(audit, dict):
        return {}
    compact = {
        key: audit.get(key)
        for key in (
            "status",
            "audit_status",
            "selected_ticker",
            "actionable",
            "final_verdict_after_audit",
            "confidence_ceiling",
            "confidence_after_caps",
            "base_return_hurdle",
            "best_base_annualized_return",
            "base_return_margin_over_hurdle",
            "downside_annualized_return",
            "return_cushion_status",
            "capital_loss_underwriting_status",
            "capital_structure_resolution_status",
            "expected_return_evidence_count",
            "company_specific_evidence_count",
            "framework_required_evidence_coverage_ratio",
            "blockers_by_class",
            "caps_by_class",
        )
        if audit.get(key) is not None
    }
    compact["hard_blockers"] = _compact_string_list(audit.get("hard_blockers"), limit=2)
    compact["confidence_caps"] = _compact_string_list(audit.get("confidence_caps"), limit=2)
    compact["needs_evidence_resolution"] = _compact_string_list(
        audit.get("needs_evidence_resolution"), limit=2
    )
    compact["notes"] = _compact_string_list(audit.get("notes"), limit=2)
    return compact


def _compact_final_decision_for_shared_memo(
    decision: SectorFinalDecision | None,
) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "verdict": decision.verdict,
        "selected_ticker": decision.selected_ticker,
        "confidence": decision.confidence,
        "expected_annualized_return_range": decision.expected_annualized_return_range,
        "no_selection_reason": _short_text(decision.no_selection_reason, limit=220),
        "selection_blockers": _compact_string_list(decision.selection_blockers, limit=2),
        "confidence_cap_reasons": _compact_string_list(decision.confidence_cap_reasons, limit=2),
    }


def _memo_shared_candidate_context(
    artifact: AutonomousSectorFinancialRunArtifact, ticker: str
) -> dict[str, Any]:
    packet_by_ticker = {packet.ticker.upper(): packet for packet in artifact.company_packets}
    packet = packet_by_ticker.get(ticker.upper())
    autonomy_run = next(
        (
            dict(item)
            for item in artifact.company_autonomy_runs
            if str(item.get("ticker") or "").upper() == ticker.upper()
        ),
        None,
    )
    return {
        "ticker": ticker.upper(),
        "packet": _compact_packet_for_shared_memo(packet),
        "expected_return_scenarios": _memo_scenario_context(artifact, ticker)[:3],
        "relative_ranking": _compact_relative_ranking_for_shared_memo(
            _memo_ranking_context(artifact, ticker)
        ),
        "company_autonomy_run": _compact_mapping(
            autonomy_run,
            [
                "ticker",
                "status",
                "final_verdict",
                "confidence",
                "tool_calls",
                "evidence_references",
            ],
        ),
        "evidence_summaries": [
            {
                "evidence_id": item.evidence_id,
                "source_label": item.source_label,
                "confidence": item.confidence,
                "summary": _short_text(item.summary, limit=220),
            }
            for item in artifact.evidence
            if str(item.ticker or "").upper() in {"", ticker.upper()}
        ][:3],
    }


def _memo_shared_context(
    artifact: AutonomousSectorFinancialRunArtifact, *, candidate_limit: int = 25
) -> dict[str, Any]:
    all_candidate_tickers = _artifact_candidate_tickers(artifact)
    candidate_tickers = all_candidate_tickers[:candidate_limit]
    included = set(candidate_tickers)
    relative_ranking = [
        _compact_relative_ranking_for_shared_memo(item)
        for item in artifact.relative_ranking
        if str(item.get("ticker") or "").upper() in included
    ]
    omitted_count = max(0, len(all_candidate_tickers) - len(candidate_tickers))
    payload = {
        "sector": artifact.sector,
        "market_cap_focus": artifact.market_cap_focus,
        "objective": artifact.objective,
        "as_of_date": artifact.as_of_date,
        "final_verdict": artifact.final_verdict,
        "selected_ticker": artifact.selected_ticker,
        "confidence": artifact.confidence,
        "deterministic_guardrails": {
            "base_return_hurdle": BASE_RETURN_HURDLE,
            "selected_return_cushion_hurdle": SELECTED_RETURN_CUSHION_HURDLE,
        },
        "candidate_context_scope": {
            "total_candidates": len(all_candidate_tickers),
            "included_candidates": len(candidate_tickers),
            "omitted_candidates": omitted_count,
            "selection_rule": (
                "all candidates included"
                if omitted_count == 0
                else f"top {len(candidate_tickers)} by deterministic pre-rank order"
            ),
            "tail_summary": (
                f"and {omitted_count} other candidates with lower deterministic pre-rank priority"
                if omitted_count
                else None
            ),
        },
        "candidate_order": candidate_tickers,
        "candidates": [
            _memo_shared_candidate_context(artifact, ticker) for ticker in candidate_tickers
        ],
        "relative_ranking": relative_ranking,
        "selection_audit": _compact_selection_audit_for_shared_memo(artifact.selection_audit),
        "final_decision": _compact_final_decision_for_shared_memo(artifact.final_decision),
    }
    return _jsonable(payload)


def _cohort_comparison_prompt(artifact: AutonomousSectorFinancialRunArtifact) -> str:
    return (
        "Write only the Cohort Comparison section for an investor-readable autonomous sector report. "
        "Return only JSON matching the schema. Do not change the final verdict, do not "
        "invent new calculations, and do not emit audit enum codes in the memo body; "
        "translate them into plain business language. Use specific numbers from the "
        "provided packets whenever possible.\n\n"
        "Write 2-3 short paragraphs total, 150-300 words, comparing valuation, quality, "
        "and asymmetry across candidates. Use the style of pre-April-19 sector memos: "
        "answer first, compare peers, and sound like an investment memo rather than an "
        "audit dashboard. Do not claim a market-beating edge or call any flag a buy "
        "signal: price triggers route analyst attention, and valuation gaps (including "
        "cheap-vs-expectations) are candidate signals, not established edges.\n\n"
        f"Sector memo state: {json.dumps(_memo_shared_context(artifact), sort_keys=True)}"
    )


def _triage_surprises_prompt(
    artifact: AutonomousSectorFinancialRunArtifact, cohort: dict[str, Any]
) -> str:
    context = {
        "cohort_comparison": cohort,
        "sector_memo_state": _memo_shared_context(artifact),
    }
    return (
        "Write only the Triage Surprises section for an investor-readable autonomous sector report. "
        "Return only JSON matching the schema. Each item should contain one sentence stating the "
        "divergence or anomaly and one sentence explaining why it matters. Include specific numbers. "
        "Flag deterministic pre-rank versus displayed order divergence, valuation-method disagreement, "
        "conviction versus base-return ranking tension, and candidates trading below DCF that are not "
        "top finalists. If everything aligned, return one explicit no-surprises line. Use the cohort "
        "comparison as binding context so the surprises do not contradict it.\n\n"
        f"Memo context: {json.dumps(_jsonable(context), sort_keys=True)}"
    )


def _candidate_memo_prompt(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    ticker: str,
    cohort: dict[str, Any],
    triage: dict[str, Any],
) -> str:
    context = {
        "cohort_comparison": cohort,
        "triage_surprises": triage,
        "sector_summary": {
            "sector": artifact.sector,
            "market_cap_focus": artifact.market_cap_focus,
            "as_of_date": artifact.as_of_date,
            "final_verdict": artifact.final_verdict,
            "selected_ticker": artifact.selected_ticker,
            "base_return_hurdle": BASE_RETURN_HURDLE,
        },
        "candidate": _memo_candidate_context(artifact, ticker),
    }
    return (
        f"Write only the per-candidate memo body for {ticker}. Return only JSON matching the schema. "
        "Do not change the final verdict. Do not emit audit enum codes in the memo body; translate "
        "them into plain business language. The thesis must start with a one-sentence company framing, "
        "state the central tension, name specific business facts with numbers from the packet, and end "
        "with verdict reasoning. Length: 4-8 sentences. Write 3-6 specific risk bullets with numbers, "
        "2-4 thesis-specific falsifiers, and 3-5 answerable open questions. Use the cohort and triage "
        "context so this candidate does not contradict the sector comparison. Do not claim a "
        "market-beating edge or call any flag a buy signal: price triggers route analyst attention, "
        "and valuation gaps (including cheap-vs-expectations) are candidate signals, not established "
        "edges. Never claim an input, metric, or disclosure is missing or unavailable unless the "
        "packet itself records that gap (a non-OK status, a blocker, a confidence cap, or a "
        "*_MISSING code); if the packet shows no data gap, do not cite missing data as a reason.\n\n"
        f"Candidate memo context: {json.dumps(_jsonable(context), sort_keys=True)}"
    )


def _section_from_payload(payload: dict[str, Any], key: str) -> list[str]:
    values = payload.get(key)
    return [str(item).strip() for item in values or [] if str(item).strip()]


def _candidate_from_payload(ticker: str, payload: dict[str, Any]) -> dict[str, Any]:
    expected_ticker = str(ticker).strip().upper()
    returned_ticker = str(payload.get("ticker") or "").strip().upper()
    if returned_ticker != expected_ticker:
        raise ValueError(
            f"{CANDIDATE_MEMO_TICKER_MISMATCH}: expected {expected_ticker}, "
            f"received {returned_ticker or '<missing>'}"
        )
    if not candidate_memo_is_substantive(payload):
        raise ValueError(
            f"{CANDIDATE_MEMO_CONTENT_INCOMPLETE}: {expected_ticker} response "
            "must contain a nonblank thesis, risk, falsifier, and open question"
        )
    return {
        "ticker": expected_ticker,
        "source": "llm",
        "status": "OK",
        "thesis": str(payload.get("thesis") or "").strip(),
        "key_risks": _section_from_payload(payload, "key_risks")[:6],
        "falsifiers": _section_from_payload(payload, "falsifiers")[:4],
        "open_questions": _section_from_payload(payload, "open_questions")[:5],
    }


def _candidate_checkpoint_namespace(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> dict[str, Any]:
    selection = (
        artifact.candidate_selection
        if isinstance(artifact.candidate_selection, dict)
        else {}
    )
    campaign_id = str(selection.get("coverage_campaign_id") or "").strip() or None
    execution_fingerprint = (
        str(selection.get("execution_fingerprint") or "").strip().lower() or None
    )
    if campaign_id is not None:
        run_identity = (
            f"coverage_campaign:{campaign_id}:"
            f"execution:{execution_fingerprint or 'not_available'}:"
            f"{artifact.sector}:{artifact.market_cap_focus}"
        )
    else:
        run_identity = str(artifact.run_id)
    return {
        "campaign_id": campaign_id,
        "run_identity": run_identity,
        "sector": str(artifact.sector),
        "market_cap_focus": str(artifact.market_cap_focus),
        "pipeline_version": str(artifact.pipeline_version or "v1"),
        "effective_as_of_date": str(artifact.as_of_date),
    }


def _candidate_checkpoint_root(
    checkpoint_root: str | Path | None,
) -> Path:
    if checkpoint_root is not None:
        return Path(checkpoint_root)
    return (
        get_config().outputs_dir
        / "checkpoints"
        / "autonomous_sector_candidate_memos"
    )


def _candidate_checkpoint_response_contract(
    provider: Any,
    *,
    max_output_tokens: int,
) -> dict[str, Any]:
    provider_name = _provider_name(provider)
    model = str(getattr(provider, "model", "") or "").strip()
    if not model:
        model = _provider_model_for_estimate(provider)
    cost_context = current_cost_context()
    strict_paid_execution = bool(
        cost_context is not None
        and bool(getattr(cost_context, "strict_first_call", False))
    )
    reasoning_level = (
        "low"
        if provider_name == "openai"
        else str(
            getattr(provider, "reasoning_effort", "")
            or "provider_default"
        )
    )
    response_config = {
        "max_output_tokens": int(max_output_tokens),
        "service_tier": str(getattr(provider, "service_tier", "") or "") or None,
        "strict_paid_execution": strict_paid_execution,
        "output_expansion_allowed": not strict_paid_execution,
        "provider_fallback_allowed": not strict_paid_execution,
        "provider_handles_retry_guard": bool(
            getattr(provider, "_handles_retry_guard", False)
        ),
        "retry_budget_contract": int(SECTOR_LLM_RETRY_RUN_BUDGET),
    }
    return {
        "provider": provider_name,
        "model": model,
        "reasoning_level": reasoning_level,
        "response_config": response_config,
        "response_config_sha256": canonical_json_sha256(response_config),
    }


def _candidate_context_source_binding(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> dict[str, Any]:
    artifact_payload = artifact.to_dict()
    for field_name in (
        "run_id",
        "created_at",
        "completed_at",
        "memo_body",
        "provider_usage",
        "lane_usage",
        "audit_notes",
    ):
        artifact_payload.pop(field_name, None)
    return {
        "binding_contract": "candidate_memo_context_source_binding_v1",
        "artifact_sha256": canonical_json_sha256(artifact_payload),
        "ordered_tickers": _artifact_packet_tickers(artifact),
        "ordered_packet_sha256": [
            canonical_json_sha256(packet.to_dict())
            for packet in artifact.company_packets
        ],
        "ordered_scenario_sha256": [
            canonical_json_sha256(scenario.to_dict())
            for scenario in artifact.expected_return_scenarios
        ],
        "evidence_sha256": canonical_json_sha256(
            [item.to_dict() for item in artifact.evidence]
        ),
    }


def _candidate_memo_context_checkpoint_plan(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    provider: Any,
    financial_integrity_results: Mapping[str, Any],
    cohort_prompt: str,
    checkpoint_root: str | Path | None,
) -> dict[str, Any] | None:
    if not isinstance(
        financial_integrity_results, Mapping
    ) or set(financial_integrity_results) != {"cohort", "triage"}:
        return None
    namespace = _candidate_checkpoint_namespace(artifact)
    source_binding = _candidate_context_source_binding(artifact)
    shared_response_contract = _candidate_checkpoint_response_contract(
        provider,
        max_output_tokens=SHARED_MEMO_MAX_OUTPUT_TOKENS,
    )
    response_config = dict(shared_response_contract.pop("response_config"))
    response_config_sha256 = str(
        shared_response_contract.pop("response_config_sha256")
    )
    integrity_scope_payloads: dict[str, dict[str, Any]] = {}
    for scope_name in ("cohort", "triage"):
        payload = _checkpointable_financial_integrity_payload(
            financial_integrity_results[scope_name]
        )
        if payload is None:
            return None
        integrity_scope_payloads[scope_name] = payload
    key = {
        **namespace,
        "source_binding": source_binding,
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "financial_integrity_contract_version": (
            CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION
        ),
        "financial_integrity_scope_fingerprints": {
            scope_name: scope_payload["scope_fingerprint"]
            for scope_name, scope_payload in integrity_scope_payloads.items()
        },
        "financial_integrity_scopes_sha256": canonical_json_sha256(
            integrity_scope_payloads
        ),
        "quote_snapshot_ids_by_scope": {
            scope_name: dict(scope_payload["ticker_snapshot_ids"])
            for scope_name, scope_payload in integrity_scope_payloads.items()
        },
        "prompt_version": CANDIDATE_MEMO_CONTEXT_PROMPT_VERSION,
        "cohort_prompt_sha256": canonical_json_sha256(cohort_prompt),
        "cohort_schema_sha256": canonical_json_sha256(
            _COHORT_COMPARISON_SCHEMA
        ),
        "triage_schema_sha256": canonical_json_sha256(
            _TRIAGE_SURPRISES_SCHEMA
        ),
        "schema_names": [
            "autonomous_sector_cohort_comparison",
            "autonomous_sector_triage_surprises",
        ],
        "cohort_response_config": response_config,
        "cohort_response_config_sha256": response_config_sha256,
        "triage_response_config": response_config,
        "triage_response_config_sha256": response_config_sha256,
        **shared_response_contract,
    }
    return {
        "checkpoint_key": key,
        "checkpoint_key_sha256": candidate_context_checkpoint_key_sha256(key),
        "checkpoint_path": candidate_context_checkpoint_path(
            _candidate_checkpoint_root(checkpoint_root),
            namespace=namespace,
            key=key,
        ),
    }


def _candidate_checkpoint_plan(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    ticker: str,
    cohort: dict[str, Any],
    triage: dict[str, Any],
    provider: Any,
    checkpoint_root: str | Path | None,
    shared_context_checkpoint_key_sha256: str | None,
    shared_context_source_binding_sha256: str | None,
) -> dict[str, Any]:
    expected_ticker = str(ticker).strip().upper()
    matching_packets = [
        item
        for item in artifact.company_packets
        if str(item.ticker).strip().upper() == expected_ticker
    ]
    if len(matching_packets) != 1:
        raise CandidateMemoCheckpointError(
            "candidate checkpoint requires exactly one packet for "
            f"{expected_ticker}; found {len(matching_packets)}"
        )
    packet = matching_packets[0]
    scenarios = [
        item
        for item in artifact.expected_return_scenarios
        if str(item.ticker).strip().upper() == expected_ticker
    ]
    integrity_scope = _financial_integrity_scope(
        context=candidate_memo_schema_name(expected_ticker),
        as_of_date=artifact.as_of_date,
        packets=[packet],
        scenarios=scenarios,
    )
    integrity_result = require_financial_integrity_scope(integrity_scope)
    _apply_financial_integrity_result(
        packets=[packet],
        scenarios=scenarios,
        result=integrity_result,
    )
    prompt = _candidate_memo_prompt(
        artifact,
        ticker=expected_ticker,
        cohort=cohort,
        triage=triage,
    )
    integrity_payload = _checkpointable_financial_integrity_payload(
        integrity_result
    )
    if (
        integrity_payload is None
        or not shared_context_checkpoint_key_sha256
        or not shared_context_source_binding_sha256
    ):
        return {
            "ticker": expected_ticker,
            "integrity_scope": integrity_scope,
            "integrity_result": None,
            "prompt": prompt,
            "checkpoint_key": None,
            "checkpoint_key_sha256": None,
            "checkpoint_path": None,
        }
    packet_payload = packet.to_dict()
    scenario_payloads = [item.to_dict() for item in scenarios]
    all_scenario_payloads = [
        item.to_dict() for item in artifact.expected_return_scenarios
    ]
    full_evidence_payloads = [item.to_dict() for item in artifact.evidence]
    candidate_context = _memo_candidate_context(artifact, expected_ticker)
    evidence_payload = {
        "full_evidence_references": full_evidence_payloads,
        "candidate_context_without_financial_payloads": {
            key: value
            for key, value in candidate_context.items()
            if key not in {"packet", "expected_return_scenarios"}
        },
        "cohort_comparison": cohort,
        "triage_surprises": triage,
    }
    source_binding = {
        "artifact_type": "candidate_memo_source_binding_v2",
        "sector": artifact.sector,
        "market_cap_focus": artifact.market_cap_focus,
        "pipeline_version": artifact.pipeline_version,
        "as_of_date": artifact.as_of_date,
        "ordered_tickers": _artifact_packet_tickers(artifact),
        "ordered_packet_fingerprints": [
            {
                "ticker": str(item.ticker).strip().upper(),
                "packet_sha256": canonical_json_sha256(item.to_dict()),
            }
            for item in artifact.company_packets
        ],
        "candidate_packet_sha256": canonical_json_sha256(packet_payload),
        "candidate_scenarios_sha256": canonical_json_sha256(
            scenario_payloads
        ),
        "all_scenarios_sha256": canonical_json_sha256(
            all_scenario_payloads
        ),
        "evidence_sha256": canonical_json_sha256(evidence_payload),
        "shared_context_checkpoint_key_sha256": (
            shared_context_checkpoint_key_sha256
        ),
        "shared_context_source_binding_sha256": (
            shared_context_source_binding_sha256
        ),
    }
    issuer_identity = {
        "ticker": expected_ticker,
        "issuer_cik": packet.issuer_cik,
        "issuer_primary_ticker": packet.issuer_primary_ticker,
        "issuer_listed_tickers": list(packet.issuer_listed_tickers),
        "security_role": packet.security_role,
        "is_secondary_class": packet.is_secondary_class,
        "is_adr": packet.is_adr,
        "adr_ratio": packet.adr_ratio,
        "share_class_ratio": packet.share_class_ratio,
        "identity_source": packet.identity_source,
        "identity_source_url": packet.identity_source_url,
        "identity_as_of_date": packet.identity_as_of_date,
    }
    price_share_basis = {
        "quote_snapshot_id": packet.quote_snapshot_id,
        "cap_stage_quote_snapshot_id": packet.cap_stage_quote_snapshot_id,
        "price_basis": packet.price_basis,
        "raw_price": packet.raw_price,
        "shares_outstanding_mm": packet.shares_outstanding_mm,
        "shares_unit": packet.shares_unit,
        "shares_basis": packet.shares_basis,
        "shares_as_of_date": packet.shares_as_of_date,
        "issuer_quote_ratio": packet.issuer_quote_ratio,
        "split_adjustment_factor": packet.split_adjustment_factor,
        "split_effective_date": packet.split_effective_date,
    }
    namespace = _candidate_checkpoint_namespace(artifact)
    response_contract = _candidate_checkpoint_response_contract(
        provider,
        max_output_tokens=CANDIDATE_MEMO_MAX_OUTPUT_TOKENS,
    )
    schema_name = candidate_memo_schema_name(expected_ticker)
    memo_context = {
        "cohort_comparison": cohort,
        "triage_surprises": triage,
    }
    key = {
        **namespace,
        "ticker": expected_ticker,
        "issuer_identity": issuer_identity,
        "issuer_identity_sha256": canonical_json_sha256(issuer_identity),
        "packet_sha256": canonical_json_sha256(packet_payload),
        "scenarios_sha256": canonical_json_sha256(scenario_payloads),
        "all_scenarios_sha256": canonical_json_sha256(
            all_scenario_payloads
        ),
        "evidence_sha256": canonical_json_sha256(evidence_payload),
        "memo_context": memo_context,
        "memo_context_sha256": canonical_json_sha256(memo_context),
        "source_binding": source_binding,
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "shared_context_checkpoint_key_sha256": (
            shared_context_checkpoint_key_sha256
        ),
        "shared_context_source_binding_sha256": (
            shared_context_source_binding_sha256
        ),
        "financial_integrity_contract_version": (
            CANDIDATE_MEMO_FINANCIAL_INTEGRITY_CONTRACT_VERSION
        ),
        "financial_integrity_scope_fingerprint": (
            integrity_payload["scope_fingerprint"]
        ),
        "quote_snapshot_ids": dict(
            integrity_payload["ticker_snapshot_ids"]
        ),
        "price_share_basis": price_share_basis,
        "price_share_basis_sha256": canonical_json_sha256(price_share_basis),
        "prompt_version": CANDIDATE_MEMO_PROMPT_VERSION,
        "prompt_sha256": canonical_json_sha256(prompt),
        "schema_name": schema_name,
        "schema_sha256": canonical_json_sha256(_MEMO_CANDIDATE_SCHEMA),
        **response_contract,
    }
    return {
        "ticker": expected_ticker,
        "integrity_scope": integrity_scope,
        "integrity_result": integrity_result,
        "prompt": prompt,
        "checkpoint_key": key,
        "checkpoint_key_sha256": candidate_checkpoint_key_sha256(key),
        "checkpoint_path": candidate_checkpoint_path(
            _candidate_checkpoint_root(checkpoint_root),
            namespace=namespace,
            key=key,
        ),
    }


def _provider_usage_matches_checkpoint_key(
    rows: list[dict[str, Any]],
    key: dict[str, Any],
) -> bool:
    allowed_schemas = (
        {str(item) for item in key.get("schema_names") or []}
        or {str(key.get("schema_name") or "")}
    )
    if not rows:
        return False
    rows_by_schema: dict[str, list[dict[str, Any]]] = {
        schema_name: [] for schema_name in allowed_schemas
    }
    for row in rows:
        schema_name = str(row.get("schema_name") or "")
        if schema_name not in allowed_schemas:
            return False
        response_config = (
            key.get("cohort_response_config")
            if schema_name == "autonomous_sector_cohort_comparison"
            else key.get("triage_response_config")
            if schema_name == "autonomous_sector_triage_surprises"
            else key.get("response_config")
        )
        if (
            not isinstance(response_config, dict)
            or str(row.get("provider") or "").strip().lower()
            != str(key.get("provider") or "").strip().lower()
            or str(row.get("model") or "").strip()
            != str(key.get("model") or "")
            or row.get("max_output_tokens_applied") is not True
            or row.get("requested_max_output_tokens")
            != response_config.get("max_output_tokens")
        ):
            return False
        rows_by_schema[schema_name].append(row)
    if any(
        not any(str(row.get("status") or "").upper() == "OK" for row in schema_rows)
        for schema_rows in rows_by_schema.values()
    ):
        return False
    strict_paid_execution = any(
        bool(
            (
                key.get("cohort_response_config")
                if schema_name == "autonomous_sector_cohort_comparison"
                else key.get("triage_response_config")
                if schema_name == "autonomous_sector_triage_surprises"
                else key.get("response_config")
            ).get("strict_paid_execution")
        )
        for schema_name in allowed_schemas
    )
    return not strict_paid_execution or all(
        len(schema_rows) == 1
        and str(schema_rows[0].get("status") or "").upper() == "OK"
        for schema_rows in rows_by_schema.values()
    )


def _notify_candidate_checkpoint_written(
    *,
    ticker: str,
    checkpoint_path: Path,
    checkpoint_key_sha256: str,
) -> None:
    """Test/telemetry seam invoked only after the atomic file is durable."""

    del ticker, checkpoint_path, checkpoint_key_sha256


def _memo_prompt_oversize_fallback(
    *,
    section: str,
    prompt: str,
    degraded_states: list[str],
    generation_notes: list[str],
) -> dict[str, Any] | None:
    estimated_tokens = _estimate_tokens_from_text(prompt)
    if estimated_tokens <= SECTOR_MEMO_PROMPT_PREFLIGHT_MAX_TOKENS:
        return None
    state = "COHORT_PROMPT_OVERSIZE"
    _append_unique(degraded_states, state, f"{section}_fallback:{state}")
    generation_notes.append(
        f"{section} memo-body call skipped before provider call because prompt estimate "
        f"{estimated_tokens} tokens exceeded {SECTOR_MEMO_PROMPT_PREFLIGHT_MAX_TOKENS}."
    )
    base = {
        "source": "deterministic_fallback",
        "status": "DEGRADED_STATE",
        "degraded_state": state,
        "reason": (
            f"Prompt estimate {estimated_tokens} tokens exceeded "
            f"{SECTOR_MEMO_PROMPT_PREFLIGHT_MAX_TOKENS}; skipped provider call."
        ),
        "prompt_estimated_tokens": estimated_tokens,
    }
    if section == "cohort_comparison":
        return {**base, "paragraphs": []}
    return {**base, "items": []}


def _persist_memo_provider_usage(
    artifact: AutonomousSectorFinancialRunArtifact,
    provider_usage: list[dict[str, Any]],
) -> None:
    """Merge memo calls and refresh v2 lane totals after thread-local capture."""

    artifact.provider_usage = merge_provider_usage_records(
        artifact.provider_usage,
        provider_usage,
    )
    if getattr(artifact, "pipeline_version", "v1") == SECTOR_PIPELINE_VERSION_V2:
        artifact.lane_usage = _v2_lane_usage_payload(artifact)


def _enrich_sector_artifact_memo_body_impl(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    provider: Any | None = None,
    checkpoint_root: str | Path | None = None,
) -> AutonomousSectorFinancialRunArtifact:
    """Attach LLM-authored memo-body sections, with renderer-safe fallback markers."""

    if artifact.status != "COMPLETED" or not artifact.company_packets:
        return artifact
    memo_integrity_scope = _financial_integrity_scope(
        context="autonomous_sector_memo_preflight",
        as_of_date=artifact.as_of_date,
        packets=artifact.company_packets,
        scenarios=artifact.expected_return_scenarios,
    )
    try:
        memo_integrity_result = _require_applied_financial_integrity_scope(memo_integrity_scope)
    except InvalidFinancialInputError as exc:
        _apply_financial_integrity_result(
            packets=artifact.company_packets,
            scenarios=artifact.expected_return_scenarios,
            result=exc.result,
        )
        raise
    active_provider = provider if provider is not None else get_alpha_llm_provider()
    if not _provider_enabled(active_provider):
        artifact.memo_body = _fallback_memo_body(
            artifact,
            state="SECTOR_MEMO_BODY_LLM_UNAVAILABLE",
            reason="LLM provider unavailable for memo-body prose; renderer will use deterministic packet prose.",
        )
        return artifact
    candidate_tickers = _require_unique_candidate_packets(artifact)
    shared_integrity_scopes = {
        "cohort": _financial_integrity_scope(
            context="autonomous_sector_cohort_comparison",
            as_of_date=artifact.as_of_date,
            packets=artifact.company_packets,
            scenarios=artifact.expected_return_scenarios,
        ),
        "triage": _financial_integrity_scope(
            context="autonomous_sector_triage_surprises",
            as_of_date=artifact.as_of_date,
            packets=artifact.company_packets,
            scenarios=artifact.expected_return_scenarios,
        ),
    }
    shared_integrity_results: dict[str, Any] = {}
    for scope_name, integrity_scope in shared_integrity_scopes.items():
        try:
            shared_integrity_results[scope_name] = (
                require_financial_integrity_scope(integrity_scope)
            )
        except InvalidFinancialInputError as exc:
            _apply_financial_integrity_result(
                packets=integrity_scope.packets,
                scenarios=integrity_scope.scenarios,
                result=exc.result,
            )
            raise
        _apply_financial_integrity_result(
            packets=integrity_scope.packets,
            scenarios=integrity_scope.scenarios,
            result=shared_integrity_results[scope_name],
        )

    def revalidate_memo_scope(
        scope: FinancialIntegrityScope,
        expected_fingerprint: str,
    ) -> Any:
        return require_unchanged_financial_integrity_scope(
            scope,
            expected_scope_fingerprint=expected_fingerprint,
        )

    def artifact_memo_prompt_state() -> dict[str, Any]:
        return {"artifact": artifact.to_dict()}

    generated_at = _utc_now_iso()
    degraded_states: list[str] = []
    generation_notes: list[str] = []
    usage_calls: list[dict[str, Any]] = []
    memo_provider_usage: list[dict[str, Any]] = []
    memo_budget_exhausted = False
    preserve_partial_v1 = (
        str(getattr(artifact, "pipeline_version", "v1") or "v1").lower()
        != SECTOR_PIPELINE_VERSION_V2
    )
    cohort_prompt_state = _freeze_financial_prompt_state(
        scope=memo_integrity_scope,
        expected_scope_fingerprint=memo_integrity_result.scope_fingerprint,
        state_getter=artifact_memo_prompt_state,
        scope_revalidator=revalidate_memo_scope,
    )
    cohort_prompt = _cohort_comparison_prompt(artifact)
    context_checkpoint_plan = _candidate_memo_context_checkpoint_plan(
        artifact,
        provider=active_provider,
        financial_integrity_results=shared_integrity_results,
        cohort_prompt=cohort_prompt,
        checkpoint_root=checkpoint_root,
    )
    context_checkpoint_hit: dict[str, Any] | None = None
    if context_checkpoint_plan is not None:
        try:
            context_checkpoint_hit = load_candidate_memo_context_checkpoint(
                context_checkpoint_plan["checkpoint_path"],
                expected_key=context_checkpoint_plan["checkpoint_key"],
                current_financial_integrity_scopes=(
                    shared_integrity_results
                ),
                triage_prompt_builder=lambda restored_cohort: (
                    _triage_surprises_prompt(artifact, restored_cohort)
                ),
            )
        except CandidateMemoCheckpointError as exc:
            attach_provider_usage_to_exception(exc, memo_provider_usage)
            raise
    if context_checkpoint_hit is not None:
        cohort = dict(context_checkpoint_hit["cohort"])
        usage_calls.extend(context_checkpoint_hit["usage_calls"])
        memo_provider_usage.extend(context_checkpoint_hit["provider_usage"])
        generation_notes.append(
            "Exact P0-authorized cohort/triage prompt context was restored "
            "before candidate checkpoint lookup."
        )
    else:
        cohort = _memo_prompt_oversize_fallback(
            section="cohort_comparison",
            prompt=cohort_prompt,
            degraded_states=degraded_states,
            generation_notes=generation_notes,
        )
    if cohort is None:
        cohort_request = {
            "prompt": cohort_prompt,
            "schema": _COHORT_COMPARISON_SCHEMA,
            "schema_name": "autonomous_sector_cohort_comparison",
            "max_output_tokens": SHARED_MEMO_MAX_OUTPUT_TOKENS,
        }
        try:
            (
                cohort_payload,
                cohort_usage,
                cohort_provider_usage,
            ) = _synthesize_memo_provider_json_with_usage(
                active_provider,
                integrity_scope=shared_integrity_scopes["cohort"],
                _financial_prompt_integrity_binding=cohort_prompt_state.bind_request(
                    active_provider,
                    cohort_request,
                ),
                **cohort_request,
            )
            memo_provider_usage.extend(cohort_provider_usage)
            cohort = {
                "source": "llm",
                "status": "OK",
                "paragraphs": _section_from_payload(cohort_payload, "paragraphs")[:3],
                "usage": cohort_usage,
            }
            usage_calls.append({"section": "cohort_comparison", **cohort_usage})
        except LLMRetryBudgetExceeded as exc:
            memo_provider_usage.extend(attached_provider_usage_records(exc))
            _persist_memo_provider_usage(artifact, memo_provider_usage)
            attach_provider_usage_to_exception(exc, memo_provider_usage)
            raise
        except LLMCostBudgetExceeded as exc:
            memo_provider_usage.extend(attached_provider_usage_records(exc))
            if not preserve_partial_v1:
                _persist_memo_provider_usage(artifact, memo_provider_usage)
                attach_provider_usage_to_exception(exc, memo_provider_usage)
                raise
            memo_budget_exhausted = True
            degraded_states.append(f"cohort_comparison_fallback:{LLM_COST_BUDGET_EXCEEDED}")
            generation_notes.append(f"Cohort comparison skipped after cost cap: {exc}")
            cohort = {
                "source": "deterministic_fallback",
                "status": "DEGRADED_STATE",
                "degraded_state": LLM_COST_BUDGET_EXCEEDED,
                "reason": str(exc),
                "paragraphs": [],
            }
        except InvalidFinancialInputError as exc:
            memo_provider_usage.extend(attached_provider_usage_records(exc))
            _persist_memo_provider_usage(artifact, memo_provider_usage)
            attach_provider_usage_to_exception(exc, memo_provider_usage)
            raise
        except Exception as exc:  # noqa: BLE001 - memo prose must degrade without failing a valid run
            memo_provider_usage.extend(attached_provider_usage_records(exc))
            state = _provider_failure_state(exc)
            degraded_states.append(f"cohort_comparison_fallback:{state}")
            generation_notes.append(f"Cohort comparison memo-body call failed ({state}): {exc}")
            cohort = {
                "source": "deterministic_fallback",
                "status": "DEGRADED_STATE",
                "degraded_state": state,
                "reason": str(exc),
                "paragraphs": [],
            }

    triage_prompt_state = _freeze_financial_prompt_state(
        scope=memo_integrity_scope,
        expected_scope_fingerprint=memo_integrity_result.scope_fingerprint,
        state_getter=lambda: {
            **artifact_memo_prompt_state(),
            "cohort_comparison": cohort,
        },
        scope_revalidator=revalidate_memo_scope,
    )
    triage_prompt = _triage_surprises_prompt(artifact, cohort)
    if context_checkpoint_hit is not None:
        triage = dict(context_checkpoint_hit["triage"])
    else:
        triage = _memo_prompt_oversize_fallback(
            section="triage_surprises",
            prompt=triage_prompt,
            degraded_states=degraded_states,
            generation_notes=generation_notes,
        )
    if memo_budget_exhausted and context_checkpoint_hit is None:
        triage = {
            "source": "deterministic_fallback",
            "status": "DEGRADED_STATE",
            "degraded_state": LLM_COST_BUDGET_EXCEEDED,
            "reason": "Campaign cost cap was exhausted before triage generation.",
            "items": [],
        }
        degraded_states.append(f"triage_surprises_fallback:{LLM_COST_BUDGET_EXCEEDED}")
    if triage is None:
        triage_request = {
            "prompt": triage_prompt,
            "schema": _TRIAGE_SURPRISES_SCHEMA,
            "schema_name": "autonomous_sector_triage_surprises",
            "max_output_tokens": SHARED_MEMO_MAX_OUTPUT_TOKENS,
        }
        try:
            (
                triage_payload,
                triage_usage,
                triage_provider_usage,
            ) = _synthesize_memo_provider_json_with_usage(
                active_provider,
                integrity_scope=shared_integrity_scopes["triage"],
                _financial_prompt_integrity_binding=triage_prompt_state.bind_request(
                    active_provider,
                    triage_request,
                ),
                **triage_request,
            )
            memo_provider_usage.extend(triage_provider_usage)
            triage = {
                "source": "llm",
                "status": "OK",
                "items": _section_from_payload(triage_payload, "items")[:8]
                or ["No material triage surprises surfaced in the LLM memo-body pass."],
                "usage": triage_usage,
            }
            usage_calls.append({"section": "triage_surprises", **triage_usage})
        except LLMRetryBudgetExceeded as exc:
            memo_provider_usage.extend(attached_provider_usage_records(exc))
            _persist_memo_provider_usage(artifact, memo_provider_usage)
            attach_provider_usage_to_exception(exc, memo_provider_usage)
            raise
        except LLMCostBudgetExceeded as exc:
            memo_provider_usage.extend(attached_provider_usage_records(exc))
            if not preserve_partial_v1:
                _persist_memo_provider_usage(artifact, memo_provider_usage)
                attach_provider_usage_to_exception(exc, memo_provider_usage)
                raise
            memo_budget_exhausted = True
            degraded_states.append(f"triage_surprises_fallback:{LLM_COST_BUDGET_EXCEEDED}")
            generation_notes.append(f"Triage skipped after cost cap: {exc}")
            triage = {
                "source": "deterministic_fallback",
                "status": "DEGRADED_STATE",
                "degraded_state": LLM_COST_BUDGET_EXCEEDED,
                "reason": str(exc),
                "items": [],
            }
        except InvalidFinancialInputError as exc:
            memo_provider_usage.extend(attached_provider_usage_records(exc))
            _persist_memo_provider_usage(artifact, memo_provider_usage)
            attach_provider_usage_to_exception(exc, memo_provider_usage)
            raise
        except Exception as exc:  # noqa: BLE001
            memo_provider_usage.extend(attached_provider_usage_records(exc))
            state = _provider_failure_state(exc)
            degraded_states.append(f"triage_surprises_fallback:{state}")
            generation_notes.append(f"Triage surprises memo-body call failed ({state}): {exc}")
            triage = {
                "source": "deterministic_fallback",
                "status": "DEGRADED_STATE",
                "degraded_state": state,
                "reason": str(exc),
                "items": [],
            }

    context_checkpoint_written = False
    context_checkpoint_skipped_binding = False
    if (
        context_checkpoint_hit is None
        and context_checkpoint_plan is not None
        and cohort.get("source") == "llm"
        and cohort.get("status") == "OK"
        and triage.get("source") == "llm"
        and triage.get("status") == "OK"
    ):
        if _provider_usage_matches_checkpoint_key(
            memo_provider_usage,
            context_checkpoint_plan["checkpoint_key"],
        ):
            try:
                persist_candidate_memo_context_checkpoint(
                    context_checkpoint_plan["checkpoint_path"],
                    checkpoint_key=context_checkpoint_plan["checkpoint_key"],
                    cohort=cohort,
                    triage=triage,
                    triage_prompt_sha256=canonical_json_sha256(triage_prompt),
                    usage_calls=usage_calls,
                    provider_usage=memo_provider_usage,
                    financial_integrity_scopes=shared_integrity_results,
                    producer_run_id=artifact.run_id,
                    created_at=_utc_now_iso(),
                )
            except Exception as exc:
                checkpoint_error = (
                    exc
                    if isinstance(exc, CandidateMemoCheckpointError)
                    else CandidateMemoCheckpointError(
                        f"candidate context checkpoint write failed: {exc}"
                    )
                )
                _persist_memo_provider_usage(artifact, memo_provider_usage)
                attach_provider_usage_to_exception(
                    checkpoint_error,
                    memo_provider_usage,
                )
                if checkpoint_error is exc:
                    raise
                raise checkpoint_error from exc
            memo_provider_usage = bind_provider_usage_to_checkpoint(
                memo_provider_usage,
                checkpoint_key_sha256=context_checkpoint_plan[
                    "checkpoint_key_sha256"
                ],
                reused=False,
            )
            context_checkpoint_written = True
        else:
            context_checkpoint_skipped_binding = True
            generation_notes.append(
                "Shared memo context was not checkpointed because the effective "
                "provider/model differed from the planned response binding."
            )

    candidate_plans: dict[str, dict[str, Any]] = {}
    candidate_results: dict[str, dict[str, Any]] = {}
    checkpoint_hit_tickers: list[str] = []
    checkpoint_miss_tickers: list[str] = []
    checkpoint_written_tickers: list[str] = []
    checkpoint_skipped_binding_tickers: list[str] = []
    candidate_checkpoint_context_authorized = bool(
        context_checkpoint_hit is not None or context_checkpoint_written
    )
    try:
        for ticker in candidate_tickers:
            plan = _candidate_checkpoint_plan(
                artifact,
                ticker=ticker,
                cohort=cohort,
                triage=triage,
                provider=active_provider,
                checkpoint_root=checkpoint_root,
                shared_context_checkpoint_key_sha256=(
                    context_checkpoint_plan["checkpoint_key_sha256"]
                    if context_checkpoint_plan is not None
                    else None
                ),
                shared_context_source_binding_sha256=(
                    context_checkpoint_plan["checkpoint_key"][
                        "source_binding_sha256"
                    ]
                    if context_checkpoint_plan is not None
                    else None
                ),
            )
            if not candidate_checkpoint_context_authorized:
                plan = {
                    **plan,
                    "checkpoint_key": None,
                    "checkpoint_key_sha256": None,
                    "checkpoint_path": None,
                }
            candidate_plans[ticker] = plan
            checkpoint_key = plan.get("checkpoint_key")
            checkpoint_path = plan.get("checkpoint_path")
            if checkpoint_key is None or checkpoint_path is None:
                checkpoint_miss_tickers.append(ticker)
                continue
            hydrated = load_candidate_memo_checkpoint(
                checkpoint_path,
                expected_key=checkpoint_key,
                current_financial_integrity=plan["integrity_result"],
            )
            if hydrated is None:
                checkpoint_miss_tickers.append(ticker)
                continue
            checkpoint_hit_tickers.append(ticker)
            candidate_results[ticker] = {
                "payload": hydrated["candidate"],
                "usage": hydrated["usage"],
                "degraded": None,
                "physical_usage": hydrated["provider_usage"],
                "fatal_error": None,
                "checkpoint_written": False,
                "checkpoint_skipped_binding": False,
            }
    except (CandidateMemoCheckpointError, InvalidFinancialInputError) as exc:
        _persist_memo_provider_usage(artifact, memo_provider_usage)
        attach_provider_usage_to_exception(exc, memo_provider_usage)
        raise

    candidate_prompt_state = _freeze_financial_prompt_state(
        scope=memo_integrity_scope,
        expected_scope_fingerprint=memo_integrity_result.scope_fingerprint,
        state_getter=lambda: {
            **artifact_memo_prompt_state(),
            "cohort_comparison": cohort,
            "triage_surprises": triage,
        },
        scope_revalidator=revalidate_memo_scope,
    )

    def candidate_task(
        ticker: str,
    ) -> tuple[
        str,
        dict[str, Any],
        dict[str, Any] | None,
        str | None,
        list[dict[str, Any]],
        Exception | None,
        bool,
        bool,
    ]:
        candidate_provider_usage: list[dict[str, Any]] = []
        plan = candidate_plans[ticker]
        if memo_budget_exhausted:
            fallback = _memo_candidate_fallback(ticker, LLM_COST_BUDGET_EXCEEDED)
            fallback["reason"] = "Campaign cost cap exhausted before candidate memo."
            return (
                ticker,
                fallback,
                None,
                f"candidate_thesis_fallback:{ticker}:{LLM_COST_BUDGET_EXCEEDED}",
                [],
                None,
                False,
                False,
            )
        candidate_request = {
            # The checkpoint plan already froze this exact prompt, so the
            # request, the checkpoint key and the paid call all bind to one
            # string rather than to two independently rebuilt ones.
            "prompt": plan["prompt"],
            "schema": _MEMO_CANDIDATE_SCHEMA,
            "schema_name": candidate_memo_schema_name(ticker),
            "max_output_tokens": CANDIDATE_MEMO_MAX_OUTPUT_TOKENS,
        }
        try:
            (
                candidate_payload,
                candidate_usage,
                candidate_provider_usage,
            ) = _synthesize_memo_provider_json_with_usage(
                active_provider,
                integrity_scope=plan["integrity_scope"],
                _financial_prompt_integrity_binding=candidate_prompt_state.bind_request(
                    active_provider,
                    candidate_request,
                ),
                **candidate_request,
            )
            candidate = _candidate_from_payload(ticker, candidate_payload)
            checkpoint_written = False
            checkpoint_skipped_binding = False
            checkpoint_key = plan.get("checkpoint_key")
            checkpoint_path = plan.get("checkpoint_path")
            if checkpoint_key is not None and checkpoint_path is not None:
                if _provider_usage_matches_checkpoint_key(
                    candidate_provider_usage,
                    checkpoint_key,
                ):
                    try:
                        persist_candidate_memo_checkpoint(
                            checkpoint_path,
                            checkpoint_key=checkpoint_key,
                            candidate=candidate,
                            usage=candidate_usage,
                            provider_usage=candidate_provider_usage,
                            financial_integrity=plan["integrity_result"],
                            producer_run_id=artifact.run_id,
                            created_at=_utc_now_iso(),
                        )
                    except Exception as exc:
                        if isinstance(exc, CandidateMemoCheckpointError):
                            raise
                        raise CandidateMemoCheckpointError(
                            f"candidate checkpoint write failed for {ticker}: {exc}"
                        ) from exc
                    candidate_provider_usage = (
                        bind_provider_usage_to_checkpoint(
                            candidate_provider_usage,
                            checkpoint_key_sha256=plan[
                                "checkpoint_key_sha256"
                            ],
                            reused=False,
                        )
                    )
                    checkpoint_written = True
                    _notify_candidate_checkpoint_written(
                        ticker=ticker,
                        checkpoint_path=checkpoint_path,
                        checkpoint_key_sha256=plan[
                            "checkpoint_key_sha256"
                        ],
                    )
                else:
                    checkpoint_skipped_binding = True
            return (
                ticker,
                candidate,
                candidate_usage,
                None,
                candidate_provider_usage,
                None,
                checkpoint_written,
                checkpoint_skipped_binding,
            )
        except LLMRetryBudgetExceeded as exc:
            return (
                ticker,
                {},
                None,
                None,
                candidate_provider_usage or attached_provider_usage_records(exc),
                exc,
                False,
                False,
            )
        except LLMCostBudgetExceeded as exc:
            if not preserve_partial_v1:
                return (
                    ticker,
                    {},
                    None,
                    None,
                    candidate_provider_usage or attached_provider_usage_records(exc),
                    exc,
                    False,
                    False,
                )
            fallback = _memo_candidate_fallback(ticker, LLM_COST_BUDGET_EXCEEDED)
            fallback["reason"] = str(exc)
            return (
                ticker,
                fallback,
                None,
                f"candidate_thesis_fallback:{ticker}:{LLM_COST_BUDGET_EXCEEDED}",
                candidate_provider_usage or attached_provider_usage_records(exc),
                None,
                False,
                False,
            )
        except InvalidFinancialInputError as exc:
            return (
                ticker,
                {},
                None,
                None,
                candidate_provider_usage or attached_provider_usage_records(exc),
                exc,
                False,
                False,
            )
        except CandidateMemoCheckpointError as exc:
            return (
                ticker,
                {},
                None,
                None,
                candidate_provider_usage or attached_provider_usage_records(exc),
                exc,
                False,
                False,
            )
        except Exception as exc:  # noqa: BLE001
            state = (
                CANDIDATE_MEMO_TICKER_MISMATCH
                if CANDIDATE_MEMO_TICKER_MISMATCH in str(exc)
                else CANDIDATE_MEMO_CONTENT_INCOMPLETE
                if CANDIDATE_MEMO_CONTENT_INCOMPLETE in str(exc)
                else _provider_failure_state(exc)
            )
            fallback = _memo_candidate_fallback(ticker, state)
            fallback["reason"] = str(exc)
            return (
                ticker,
                fallback,
                None,
                f"candidate_thesis_fallback:{ticker}:{state}",
                candidate_provider_usage or attached_provider_usage_records(exc),
                None,
                False,
                False,
            )

    candidate_payloads: dict[str, dict[str, Any]] = {}
    fatal_candidate_error: Exception | None = None
    candidate_worker_count = min(
        CANDIDATE_MEMO_MAX_WORKERS,
        max(1, len(checkpoint_miss_tickers)),
    )
    executor = ThreadPoolExecutor(max_workers=candidate_worker_count)
    future_by_ticker: dict[Any, str] = {}
    pending_tickers = iter(checkpoint_miss_tickers)

    def submit_next_candidate() -> bool:
        try:
            pending_ticker = next(pending_tickers)
        except StopIteration:
            return False
        future_by_ticker[
            executor.submit(candidate_task, pending_ticker)
        ] = pending_ticker
        return True

    try:
        for _worker_index in range(candidate_worker_count):
            if not submit_next_candidate():
                break
        while future_by_ticker:
            future = next(as_completed(tuple(future_by_ticker)))
            future_by_ticker.pop(future)
            (
                ticker,
                payload,
                usage,
                degraded,
                physical_usage,
                fatal_error,
                checkpoint_written,
                checkpoint_skipped_binding,
            ) = future.result()
            candidate_results[ticker] = {
                "payload": payload,
                "usage": usage,
                "degraded": degraded,
                "physical_usage": physical_usage,
                "fatal_error": fatal_error,
                "checkpoint_written": checkpoint_written,
                "checkpoint_skipped_binding": checkpoint_skipped_binding,
            }
            if fatal_error is not None:
                fatal_candidate_error = (
                    fatal_candidate_error or fatal_error
                )
                for pending_future in tuple(future_by_ticker):
                    if pending_future.cancel():
                        future_by_ticker.pop(pending_future, None)
            elif fatal_candidate_error is None:
                submit_next_candidate()
    except BaseException:
        for pending_future in future_by_ticker:
            pending_future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    for ticker in candidate_tickers:
        result = candidate_results.get(ticker)
        if result is None:
            continue
        memo_provider_usage.extend(result["physical_usage"])
        payload = result["payload"]
        usage = result["usage"]
        degraded = result["degraded"]
        fatal_error = result["fatal_error"]
        if result["checkpoint_written"]:
            checkpoint_written_tickers.append(ticker)
        if result["checkpoint_skipped_binding"]:
            checkpoint_skipped_binding_tickers.append(ticker)
        if fatal_error is not None:
            fatal_candidate_error = fatal_candidate_error or fatal_error
            continue
        candidate_payloads[ticker] = payload
        if usage is not None:
            usage_calls.append(
                {"section": "candidate", "ticker": ticker, **usage}
            )
        if degraded:
            degraded_states.append(degraded)
            generation_notes.append(
                f"Candidate memo-body call fell back for {ticker}: "
                f"{payload.get('reason')}"
            )

    if checkpoint_hit_tickers:
        generation_notes.append(
            "Exact P0-authorized candidate checkpoints reused for: "
            + ", ".join(checkpoint_hit_tickers)
            + "."
        )
    if checkpoint_skipped_binding_tickers:
        generation_notes.append(
            "Candidate checkpoints were not written after effective "
            "provider/model drift for: "
            + ", ".join(checkpoint_skipped_binding_tickers)
            + "."
        )

    if fatal_candidate_error is not None:
        _persist_memo_provider_usage(artifact, memo_provider_usage)
        attach_provider_usage_to_exception(
            fatal_candidate_error,
            memo_provider_usage,
        )
        raise fatal_candidate_error

    any_llm = (
        cohort.get("source") == "llm"
        or triage.get("source") == "llm"
        or any(item.get("source") == "llm" for item in candidate_payloads.values())
    )
    artifact.memo_body = _memo_body_with_degraded_state_consistency(
        {
            "status": "LLM_GENERATED"
            if any_llm and not degraded_states
            else ("PARTIAL_LLM_GENERATED" if any_llm else "DEGRADED_FALLBACK"),
            "source": "split_llm",
            "generated_at": generated_at,
            "degraded_states": list(dict.fromkeys(degraded_states)),
            "cohort_comparison": cohort,
            "triage_surprises": triage,
            "candidates": {
                ticker: candidate_payloads[ticker]
                for ticker in candidate_tickers
                if ticker in candidate_payloads
            },
            "generation_notes": generation_notes,
            "candidate_checkpoints": {
                "contract_version": CANDIDATE_MEMO_CHECKPOINT_CONTRACT_VERSION,
                "context_contract_version": (
                    CANDIDATE_MEMO_CONTEXT_CHECKPOINT_CONTRACT_VERSION
                ),
                "context_exact_hit": context_checkpoint_hit is not None,
                "context_written": context_checkpoint_written,
                "candidate_checkpoint_context_authorized": (
                    candidate_checkpoint_context_authorized
                ),
                "context_skipped_provider_binding": (
                    context_checkpoint_skipped_binding
                ),
                "total_candidates": len(candidate_tickers),
                "exact_hit_count": len(checkpoint_hit_tickers),
                "provider_miss_count": len(checkpoint_miss_tickers),
                "written_count": len(checkpoint_written_tickers),
                "skipped_provider_binding_count": len(
                    checkpoint_skipped_binding_tickers
                ),
                "exact_hit_tickers": checkpoint_hit_tickers,
                "provider_miss_tickers": checkpoint_miss_tickers,
                "written_tickers": checkpoint_written_tickers,
                "skipped_provider_binding_tickers": (
                    checkpoint_skipped_binding_tickers
                ),
                "checkpoint_key_sha256_by_ticker": {
                    ticker: candidate_plans[ticker][
                        "checkpoint_key_sha256"
                    ]
                    for ticker in candidate_tickers
                    if candidate_plans[ticker].get(
                        "checkpoint_key_sha256"
                    )
                },
            },
            "usage": _memo_usage_rollup(usage_calls),
        }
    )
    _persist_memo_provider_usage(artifact, memo_provider_usage)
    return artifact


def enrich_sector_artifact_memo_body(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    provider: Any | None = None,
    checkpoint_root: str | Path | None = None,
) -> AutonomousSectorFinancialRunArtifact:
    """Attach memo-body prose under the sector-run retry budget."""

    active_context = current_retry_context()
    active_cost_context = current_cost_context()
    if active_context is None:
        with llm_retry_budget(SECTOR_LLM_RETRY_RUN_BUDGET) as retry_context:
            return _enrich_sector_artifact_memo_body_with_retry_context(
                artifact,
                provider=provider,
                checkpoint_root=checkpoint_root,
                retry_context=retry_context,
                cost_context=active_cost_context,
            )
    return _enrich_sector_artifact_memo_body_with_retry_context(
        artifact,
        provider=provider,
        checkpoint_root=checkpoint_root,
        retry_context=active_context,
        cost_context=active_cost_context,
    )


def _enrich_sector_artifact_memo_body_with_retry_context(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    provider: Any | None,
    checkpoint_root: str | Path | None,
    retry_context: Any,
    cost_context: Any,
) -> AutonomousSectorFinancialRunArtifact:
    try:
        enriched = _enrich_sector_artifact_memo_body_impl(
            artifact,
            provider=provider,
            checkpoint_root=checkpoint_root,
        )
    except InvalidFinancialInputError as exc:
        integrity_usage = attached_provider_usage_records(exc)
        prior_attempt_count = len(integrity_usage)
        artifact.status = "FAILED"
        artifact.final_verdict = "NO_SELECTION"
        artifact.selected_ticker = None
        artifact.confidence = None
        artifact.no_selection_reason = (
            "Memo generation stopped because deterministic financial prompt inputs "
            f"returned {exc.status}"
            + (
                f" after {prior_attempt_count} prior physical provider attempt(s)."
                if prior_attempt_count
                else " before any physical provider attempt."
            )
        )
        _append_unique(artifact.degraded_states, exc.status)
        artifact.audit_notes.extend(
            [
                (
                    "Stopped all later memo provider calls and substantive fallback after "
                    f"the financial-integrity gate returned {exc.status}; "
                    f"{prior_attempt_count} prior physical attempt(s) remain accounted."
                    if prior_attempt_count
                    else "No memo provider call or substantive fallback ran after the "
                    f"financial-integrity gate returned {exc.status}."
                ),
                json.dumps(exc.result.to_dict(), sort_keys=True, default=str),
            ]
        )
        artifact.memo_body = {
            "status": exc.status,
            "source": "deterministic_financial_integrity_gate",
            "generated_at": _utc_now_iso(),
            "degraded_states": [exc.status],
            "cohort_comparison": {"paragraphs": []},
            "triage_surprises": {"items": []},
            "candidates": {},
            "generation_notes": [artifact.no_selection_reason],
            "usage": _memo_usage_rollup(integrity_usage),
            "financial_integrity": exc.result.to_dict(),
        }
        enriched = artifact
    except (LLMRetryBudgetExceeded, LLMCostBudgetExceeded) as exc:
        state = _provider_failure_state(exc)
        if getattr(artifact, "pipeline_version", "v1") == SECTOR_PIPELINE_VERSION_V2:
            _append_unique(artifact.degraded_states, f"MEMO_BODY_{state}")
        else:
            _append_unique(artifact.degraded_states, state)
            artifact.status = "DEGRADED"
            artifact.no_selection_reason = artifact.no_selection_reason or str(exc)
        artifact.audit_notes.append(
            f"Memo-body enrichment aborted because a run LLM budget was exceeded ({state}): {exc}"
        )
        enriched = artifact
    _append_llm_retry_audit_notes(enriched, retry_context)
    if cost_context is not None:
        _append_llm_cost_audit_notes(enriched, cost_context)
    integrity_terminal = (
        isinstance(enriched.memo_body, dict)
        and enriched.memo_body.get("source") == "deterministic_financial_integrity_gate"
    )
    if (
        getattr(enriched, "pipeline_version", "v1") == SECTOR_PIPELINE_VERSION_V2
        and not integrity_terminal
    ):
        enriched._validate_v2()
    return enriched


def _default_framework_payload(sector: str, market_cap_focus: str) -> dict[str, Any]:
    return sector_framework_template_payload(sector, market_cap_focus)


def _framework_required_evidence_tool_hints(
    required_evidence: list[str],
    allowed_tools: list[str],
) -> list[dict[str, Any]]:
    allowed = {str(tool) for tool in allowed_tools}
    rows: list[dict[str, Any]] = []

    def matching_tools(requirement: str) -> list[str]:
        text = requirement.lower()
        tools: list[str] = []

        def add(candidates: list[str]) -> None:
            for tool in candidates:
                if tool in allowed and tool not in tools:
                    tools.append(tool)

        if "expected_return" in text or "scenario" in text:
            add(["rank_expected_return_cases", "compare_expected_return_scenarios"])
        if any(
            term in text
            for term in (
                "payment",
                "transaction",
                "take_rate",
                "merchant",
                "fraud",
                "chargeback",
                "funding",
                "float",
                "partner",
                "compliance",
            )
        ):
            add(
                [
                    "fetch_kpi_trends",
                    "fetch_companyfacts_timeseries",
                    "fetch_filing_section",
                    "fetch_current_events",
                    "analyze_liquidity_stress",
                ]
            )
        if any(
            term in text
            for term in (
                "unit_volume",
                "volume_mix",
                "incentive",
                "inventory",
                "channel",
                "warranty",
                "recall",
                "powertrain",
                "residual",
                "vehicle",
            )
        ):
            add(
                [
                    "fetch_kpi_trends",
                    "fetch_companyfacts_timeseries",
                    "fetch_filing_section",
                    "fetch_current_events",
                    "analyze_liquidity_stress",
                ]
            )
        if any(
            term in text
            for term in (
                "subscriber",
                "audience",
                "arpu",
                "advertising",
                "affiliate",
                "spectrum",
                "distribution",
                "content",
                "network",
            )
        ):
            add(
                [
                    "fetch_kpi_trends",
                    "fetch_companyfacts_timeseries",
                    "fetch_filing_section",
                    "fetch_current_events",
                ]
            )
        if any(
            term in text
            for term in (
                "procedure",
                "utilization",
                "installed_base",
                "consumables",
                "site_of_care",
                "fda",
                "clearance",
                "quality_system",
                "recall",
                "hospital_capex",
            )
        ):
            add(
                [
                    "fetch_kpi_trends",
                    "fetch_companyfacts_timeseries",
                    "fetch_filing_section",
                    "fetch_current_events",
                    "fetch_recent_filing_context",
                ]
            )
        if any(
            term in text
            for term in (
                "rate_case",
                "rate_base",
                "allowed_roe",
                "equity_ratio",
                "affordability",
                "dividend",
                "customer_bill",
                "load_growth",
            )
        ):
            add(
                [
                    "fetch_filing_section",
                    "fetch_recent_filing_context",
                    "fetch_companyfacts_timeseries",
                    "analyze_liquidity_stress",
                ]
            )
        if any(
            term in text
            for term in (
                "commodity",
                "feedstock",
                "cost_curve",
                "cash_cost",
                "capacity_utilization",
                "reclamation",
                "environmental",
            )
        ):
            add(
                [
                    "fetch_kpi_trends",
                    "fetch_companyfacts_timeseries",
                    "fetch_filing_section",
                    "fetch_current_events",
                    "analyze_liquidity_stress",
                ]
            )
        if "kpi" in text or "quality" in text:
            add(["fetch_kpi_trends", "compare_peer_metric"])
        if any(
            term in text for term in ("latest", "current", "filing_update", "freshness", "filing")
        ):
            add(["fetch_current_events", "fetch_recent_filing_context", "fetch_filing_section"])
        if any(term in text for term in ("transcript", "management", "commentary")):
            add(["fetch_transcript_excerpt", "fetch_current_events"])
        if any(term in text for term in ("sbc", "share_count", "dilution", "capital_allocation")):
            add(["analyze_dilution", "analyze_capital_allocation", "fetch_companyfacts_timeseries"])
        if any(
            term in text
            for term in (
                "capital",
                "solvency",
                "debt",
                "liquidity",
                "refinancing",
                "credit",
                "reserve",
                "book_value",
                "security_type",
                "common_equity",
                "underwriting",
                "spread",
            )
        ):
            add(
                [
                    "analyze_liquidity_stress",
                    "analyze_capital_structure_resolution",
                    "fetch_companyfacts_timeseries",
                    "fetch_filing_section",
                    "fetch_insurance_evidence_packet",
                ]
            )
        if any(
            term in text
            for term in (
                "margin",
                "cash_conversion",
                "cash",
                "fcf",
                "gross",
                "backlog",
                "order",
                "pricing",
                "input_cost",
                "cycle",
                "working_capital",
                "capex",
                "retention",
                "churn",
                "concentration",
                "customer",
                "product",
                "payer",
                "reimbursement",
                "clinical",
                "regulatory",
                "milestone",
            )
        ):
            add(
                [
                    "fetch_kpi_trends",
                    "fetch_companyfacts_timeseries",
                    "fetch_filing_section",
                    "fetch_recent_filing_context",
                ]
            )
        if any(
            term in text
            for term in ("noi", "occupancy", "lease", "nav", "cap_rate", "property", "ffo", "affo")
        ):
            add(
                [
                    "fetch_filing_section",
                    "fetch_companyfacts_timeseries",
                    "fetch_kpi_trends",
                    "analyze_liquidity_stress",
                ]
            )
        if not tools:
            add(["fetch_kpi_trends", "fetch_filing_section", "fetch_companyfacts_timeseries"])
        return tools

    for item in required_evidence:
        requirement = str(item or "").strip()
        if not requirement:
            continue
        rows.append(
            {
                "required_evidence": requirement,
                "suggested_tools": matching_tools(requirement),
            }
        )
    return rows


def _packet_field_has_signal(payload: Any, keys: tuple[str, ...] | None = None) -> bool:
    if not isinstance(payload, dict):
        return False
    items = ((key, payload.get(key)) for key in keys) if keys else payload.items()
    for _key, value in items:
        if value is None:
            continue
        if isinstance(value, str) and value.strip().upper() in {"", "UNKNOWN", "N/A"}:
            continue
        if value == [] or value == {}:
            continue
        return True
    return False


def _is_medical_device_sector(sector: str | None) -> bool:
    return str(sector or "").strip().lower() in {
        "medical_devices",
        "medical device",
        "medical devices",
        "medtech",
    }


def _requirement_has_any(normalized_requirement: str, tokens: tuple[str, ...]) -> bool:
    return any(token in normalized_requirement for token in tokens)


def _medical_device_requirement_kind(
    normalized_requirement: str, *, sector: str | None = None
) -> str | None:
    if not normalized_requirement:
        return None
    medical_device_context = _is_medical_device_sector(sector) or _requirement_has_any(
        normalized_requirement,
        (
            "medical_device",
            "medical device",
            "medtech",
            "device",
            "fda",
            "510(k)",
            "quality_system",
            "quality system",
            "site_of_care",
            "site of care",
            "hospital_capex",
            "hospital capex",
            "hospital_purchasing",
            "hospital purchasing",
            "capital_equipment",
            "capital equipment",
        ),
    )
    if "reimbursement_site_of_care" in normalized_requirement or (
        medical_device_context
        and _requirement_has_any(
            normalized_requirement,
            (
                "reimbursement",
                "payer",
                "site_of_care",
                "site of care",
                "hospital_outpatient",
                "hospital outpatient",
            ),
        )
    ):
        return "reimbursement"
    if "channel_customer_concentration_or_hospital_capex" in normalized_requirement or (
        medical_device_context
        and _requirement_has_any(
            normalized_requirement,
            (
                "channel_customer_concentration",
                "customer_concentration",
                "customer concentration",
                "channel_concentration",
                "channel concentration",
                "hospital_capex",
                "hospital capex",
                "hospital_purchasing",
                "hospital purchasing",
                "purchasing_cycle",
                "capital_equipment",
                "capital equipment",
            ),
        )
    ):
        return "channel"
    if _requirement_has_any(
        normalized_requirement,
        (
            "fda",
            "510(k)",
            "clearance",
            "quality_system",
            "quality system",
            "warning_letter",
            "warning letter",
        ),
    ):
        return "regulatory"
    if "recall" in normalized_requirement and (
        medical_device_context
        or _requirement_has_any(
            normalized_requirement, ("medical_device", "medical device", "medtech", "device")
        )
    ):
        return "regulatory"
    if (
        _is_medical_device_sector(sector)
        and "regulatory" in normalized_requirement
        and not _requirement_has_any(
            normalized_requirement,
            (
                "regulatory_capital",
                "regulatory capital",
                "rating_agency",
                "rating agency",
                "rate_case",
                "rate case",
            ),
        )
    ):
        return "regulatory"
    return None


def _framework_requirement_packet_support(
    requirement: str,
    *,
    packet: SectorCompanyFinancialPacket,
    has_base_return_scenario: bool,
    sector: str | None = None,
) -> list[str]:
    normalized = str(requirement or "").strip().lower()
    supporting_fields: list[str] = []

    def add(field_name: str, payload: Any, keys: tuple[str, ...] | None = None) -> None:
        if _packet_field_has_signal(payload, keys) and field_name not in supporting_fields:
            supporting_fields.append(field_name)

    if normalized in {"expected_return_evidence", "expected_return_scenarios"}:
        if has_base_return_scenario:
            supporting_fields.append("expected_return_scenarios")
        add(
            "expected_return",
            packet.expected_return,
            ("base_anchor_discount", "base_anchor_method"),
        )
    if normalized in {"company_specific_evidence", "company_specific_evidence_packet"}:
        add("business_quality", packet.business_quality)
        add("balance_sheet", packet.balance_sheet)
        add("capital_allocation", packet.capital_allocation)
    if normalized in {"kpi_trends", "quality_metrics"}:
        add(
            "business_quality",
            packet.business_quality,
            ("revenue_cagr_5y", "earnings_quality", "moat_score"),
        )
        add(
            "returns_on_capital",
            packet.returns_on_capital,
            ("roic_vs_median", "operating_margin_vs_median"),
        )
        add("cash_conversion", packet.cash_conversion, ("cash_conversion_ratio", "fcf_margin"))
    medical_device_requirement_kind = _medical_device_requirement_kind(normalized, sector=sector)
    if medical_device_requirement_kind == "reimbursement":
        add(
            "business_quality",
            packet.business_quality,
            ("reimbursement_status", "site_of_care_status", "payer_mix_status"),
        )
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("reimbursement_status", "site_of_care_status", "payer_mix_status"),
        )
        return supporting_fields
    if medical_device_requirement_kind == "regulatory":
        add(
            "accounting_quality",
            packet.accounting_quality,
            (
                "fda_clearance_status",
                "quality_system_status",
                "recall_status",
                "regulatory_status",
                "regulatory_or_recall_event_status",
            ),
        )
        add(
            "business_quality",
            packet.business_quality,
            (
                "fda_clearance_status",
                "quality_system_status",
                "recall_status",
                "regulatory_status",
                "regulatory_or_recall_event_status",
            ),
        )
        return supporting_fields
    if medical_device_requirement_kind == "channel":
        add(
            "business_quality",
            packet.business_quality,
            (
                "customer_or_channel_concentration",
                "customer_concentration",
                "channel_concentration",
                "hospital_capex_cycle_status",
                "hospital_purchasing_status",
                "capital_equipment_cycle_status",
            ),
        )
        add(
            "cash_conversion",
            packet.cash_conversion,
            (
                "hospital_capex_cycle_status",
                "hospital_purchasing_status",
                "capital_equipment_cycle_status",
            ),
        )
        return supporting_fields
    if any(term in normalized for term in ("payment", "transaction", "take_rate", "merchant")):
        add(
            "business_quality",
            packet.business_quality,
            ("revenue_cagr_5y", "earnings_quality", "moat_score"),
        )
        add("cash_conversion", packet.cash_conversion, ("cash_conversion_ratio", "fcf_margin"))
    if any(
        term in normalized
        for term in ("fraud", "chargeback", "credit_loss", "funding", "float", "regulatory_capital")
    ):
        add(
            "balance_sheet",
            packet.balance_sheet,
            ("solvency_risk", "current_ratio", "debt_due_within_12mo"),
        )
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
    if any(term in normalized for term in ("partner", "compliance")):
        add("business_quality", packet.business_quality, ("moat_score", "earnings_quality"))
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
    if any(
        term in normalized
        for term in ("unit_volume", "volume_mix", "unit_sales", "incentive", "inventory", "channel")
    ):
        add(
            "business_quality",
            packet.business_quality,
            ("revenue_cagr_5y", "earnings_quality", "moat_score"),
        )
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
    if any(term in normalized for term in ("platform", "powertrain", "tooling")):
        add("business_quality", packet.business_quality, ("revenue_cagr_5y", "earnings_quality"))
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
    if any(term in normalized for term in ("warranty", "recall")):
        add("business_quality", packet.business_quality, ("earnings_quality",))
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
    if any(term in normalized for term in ("residual_value", "leverage", "finance")):
        add(
            "balance_sheet",
            packet.balance_sheet,
            ("solvency_risk", "current_ratio", "debt_due_within_12mo"),
        )
    if any(
        term in normalized
        for term in ("subscriber", "audience", "arpu", "advertising", "affiliate")
    ):
        add(
            "business_quality",
            packet.business_quality,
            ("revenue_cagr_5y", "earnings_quality", "moat_score"),
        )
        add("cash_conversion", packet.cash_conversion, ("cash_conversion_ratio", "fcf_margin"))
    if any(term in normalized for term in ("spectrum", "distribution", "content", "network")):
        add("business_quality", packet.business_quality, ("moat_score", "earnings_quality"))
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
    if any(
        term in normalized
        for term in (
            "procedure",
            "utilization",
            "installed_base",
            "consumables",
            "site_of_care",
            "hospital_capex",
        )
    ):
        add(
            "business_quality",
            packet.business_quality,
            ("revenue_cagr_5y", "earnings_quality", "moat_score"),
        )
        add("cash_conversion", packet.cash_conversion, ("cash_conversion_ratio", "fcf_margin"))
    if any(term in normalized for term in ("fda", "clearance", "quality_system", "recall")):
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
        add("business_quality", packet.business_quality, ("earnings_quality", "moat_score"))
    if any(
        term in normalized
        for term in (
            "rate_case",
            "rate_base",
            "allowed_roe",
            "equity_ratio",
            "affordability",
            "customer_bill",
            "load_growth",
            "dividend",
        )
    ):
        add(
            "balance_sheet",
            packet.balance_sheet,
            ("solvency_risk", "current_ratio", "debt_due_within_12mo"),
        )
        add("cash_conversion", packet.cash_conversion, ("cash_conversion_ratio", "fcf_margin"))
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
    if any(
        term in normalized
        for term in ("commodity", "feedstock", "cost_curve", "cash_cost", "capacity_utilization")
    ):
        add(
            "business_quality",
            packet.business_quality,
            ("revenue_cagr_5y", "earnings_quality", "moat_score"),
        )
        add("cash_conversion", packet.cash_conversion, ("cash_conversion_ratio", "fcf_margin"))
    if any(term in normalized for term in ("environmental", "reclamation")):
        add(
            "balance_sheet",
            packet.balance_sheet,
            ("solvency_risk", "current_ratio", "debt_due_within_12mo"),
        )
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
    if "security_type" in normalized or "common_equity" in normalized:
        if not _selection_blockers_for_packet(packet):
            supporting_fields.append("model_fit_status")
    if any(
        term in normalized
        for term in ("capital", "solvency", "debt", "liquidity", "refinancing", "rate")
    ):
        add(
            "balance_sheet",
            packet.balance_sheet,
            ("solvency_risk", "current_ratio", "debt_due_within_12mo"),
        )
    if any(
        term in normalized for term in ("book_value", "reserve", "credit", "underwriting", "spread")
    ):
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status"),
        )
        add("valuation", packet.valuation, ("insurance_value", "generic_valuation_valid"))
    if any(term in normalized for term in ("margin", "cash_conversion", "cash", "fcf", "gross")):
        add("cash_conversion", packet.cash_conversion, ("cash_conversion_ratio", "fcf_margin"))
        add("business_quality", packet.business_quality, ("earnings_quality",))
    if any(term in normalized for term in ("sbc", "share_count", "dilution")):
        add("capital_allocation", packet.capital_allocation, ("dilution_rate_shares_cagr",))
    if any(
        term in normalized for term in ("latest", "current", "filing_update", "freshness", "filing")
    ):
        add(
            "accounting_quality",
            packet.accounting_quality,
            ("filing_risk_status", "filing_risk_evidence_status", "filing_risk_source_filing_date"),
        )
    return supporting_fields


def _framework_evidence_preflight(
    *,
    required_evidence: list[str],
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    allowed_tools: list[str],
    sector: str | None = None,
) -> list[dict[str, Any]]:
    required = [str(item).strip() for item in required_evidence if str(item).strip()]
    if not required:
        return []
    hints_by_requirement = {
        item["required_evidence"]: item["suggested_tools"]
        for item in _framework_required_evidence_tool_hints(required, allowed_tools)
    }
    base_scenario_tickers = {
        scenario.ticker.upper()
        for scenario in scenarios
        if str(scenario.scenario_name).lower() == "base" and scenario.annualized_return is not None
    }
    rows: list[dict[str, Any]] = []
    for packet in company_packets:
        supported: list[str] = []
        needs_tools: list[str] = []
        support_fields: dict[str, list[str]] = {}
        for requirement in required:
            fields = _framework_requirement_packet_support(
                requirement,
                packet=packet,
                has_base_return_scenario=packet.ticker.upper() in base_scenario_tickers,
                sector=sector,
            )
            if fields:
                supported.append(requirement)
                support_fields[requirement] = fields
            else:
                needs_tools.append(requirement)
        rows.append(
            {
                "ticker": packet.ticker,
                "packet_support_status": "NEEDS_TOOL_EVIDENCE"
                if needs_tools
                else "PACKET_SUPPORT_PRESENT",
                "packet_supported_required_evidence": supported,
                "needs_tool_evidence": needs_tools,
                "packet_support_ratio": len(supported) / len(required),
                "supporting_packet_fields": support_fields,
                "suggested_tools_for_missing_evidence": {
                    item: hints_by_requirement.get(item, []) for item in needs_tools
                },
            }
        )
    return rows


def _artifact_framework_evidence_preflight(
    *,
    framework: SectorFinancialFramework | None,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    allowed_tools: list[str] | None,
) -> list[dict[str, Any]]:
    if framework is None:
        return []
    return _framework_evidence_preflight(
        required_evidence=list(framework.required_evidence),
        company_packets=company_packets,
        scenarios=scenarios,
        allowed_tools=allowed_tools or [],
        sector=framework.sector,
    )


def _normalize_initial_plan_payload(
    payload: dict[str, Any],
    *,
    sector: str,
    market_cap_focus: str,
) -> dict[str, Any]:
    normalized = dict(payload)
    if not isinstance(normalized.get("framework"), dict):
        normalized["framework"] = _default_framework_payload(sector, market_cap_focus)
    else:
        normalized["framework"] = augment_sector_framework_payload(
            normalized["framework"],
            sector=sector,
            market_cap_focus=market_cap_focus,
        )
    normalized["research_questions"] = (
        normalized.get("research_questions")
        if isinstance(normalized.get("research_questions"), list)
        else []
    )
    normalized["belief_updates"] = (
        normalized.get("belief_updates")
        if isinstance(normalized.get("belief_updates"), list)
        else []
    )
    normalized["degraded_states"] = (
        normalized.get("degraded_states")
        if isinstance(normalized.get("degraded_states"), list)
        else []
    )
    normalized["audit_notes"] = (
        normalized.get("audit_notes") if isinstance(normalized.get("audit_notes"), list) else []
    )
    if normalized.get("final_decision") is not None:
        normalized["degraded_states"] = [
            *[str(item) for item in normalized["degraded_states"]],
            "INITIAL_FINAL_DECISION_IGNORED",
        ]
        normalized["audit_notes"] = [
            *[str(item) for item in normalized["audit_notes"]],
            "Ignored provider final decision on compact first planning turn; deterministic tools must run before sector selection.",
        ]
    normalized["continue_research"] = True
    normalized["final_decision"] = None
    return normalized


def _initial_plan_prompt(
    *,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    budget: AutonomousRunBudget,
    allowed_tools: list[str],
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
) -> str:
    framework_hint = _default_framework_payload(sector, market_cap_focus)
    required_evidence = [str(item) for item in framework_hint.get("required_evidence") or []]
    prompt_packets, candidate_context_scope = _sector_prompt_packet_scope(
        company_packets,
        scenarios=scenarios,
    )
    prompt_tickers = {packet.ticker.upper() for packet in prompt_packets}
    payload = {
        "turn_index": 1,
        "sector": sector,
        "market_cap_focus": market_cap_focus,
        "objective": objective,
        "as_of_date": as_of_date,
        "budget": {
            "max_tool_calls": budget.max_tool_calls,
            "max_turns": budget.max_turns,
            "max_cost_usd": budget.max_cost_usd,
            "max_candidates": budget.max_candidates,
        },
        "deterministic_selection_guardrails": {
            "base_return_hurdle": BASE_RETURN_HURDLE,
            "selected_return_cushion_hurdle": SELECTED_RETURN_CUSHION_HURDLE,
            "guardrail_source": "runtime_selection_audit",
        },
        "allowed_tools": allowed_tools,
        "framework_template_hint": {
            "economic_model": framework_hint.get("economic_model"),
            "selected_metrics": framework_hint.get("selected_metrics") or [],
            "required_evidence": required_evidence,
            "invalid_valuation_methods": framework_hint.get("invalid_valuation_methods") or [],
        },
        "framework_required_evidence_tool_hints": _framework_required_evidence_tool_hints(
            required_evidence,
            allowed_tools,
        ),
        "framework_evidence_preflight": _framework_evidence_preflight(
            required_evidence=required_evidence,
            company_packets=company_packets,
            scenarios=scenarios,
            allowed_tools=allowed_tools,
            sector=sector,
        ),
        "candidate_context_scope": candidate_context_scope,
        "company_packets": [_planning_packet_summary(packet) for packet in prompt_packets],
        "top_base_expected_return_scenarios": _scenario_summaries_for_prompt(
            scenarios,
            tickers=prompt_tickers,
            limit=16,
        ),
    }
    return (
        "You are starting an autonomous sector financial analyst loop. "
        "This first turn is planning-only: choose the financial framework, then "
        "choose 1-3 decision-relevant research questions and deterministic tool "
        "calls. Do not select a winner, do not return a final decision, and do "
        "not use narrative filler. Use framework_required_evidence_tool_hints "
        "to choose sector-specific evidence tools rather than generic-only "
        "quality checks. Treat framework_evidence_preflight as packet-level "
        "guidance, not audit proof. Prefer expected-return and company-specific "
        "evidence tools that can unblock the deterministic selection audit. Use "
        "the exact deterministic_selection_guardrails values in any hurdle-rate "
        "discussion.\n\n"
        f"Compact planning state: {json.dumps(_jsonable(payload), sort_keys=True)}"
    )


def _minimum_tool_plan_prompt(
    *,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    allowed_tools: list[str],
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    provider_error: str,
) -> str:
    framework_hint = _default_framework_payload(sector, market_cap_focus)
    required_evidence = [str(item) for item in framework_hint.get("required_evidence") or []]
    prompt_packets, candidate_context_scope = _sector_prompt_packet_scope(
        company_packets,
        scenarios=scenarios,
    )
    prompt_tickers = {packet.ticker.upper() for packet in prompt_packets}
    payload = {
        "sector": sector,
        "market_cap_focus": market_cap_focus,
        "objective": objective,
        "as_of_date": as_of_date,
        "provider_error": provider_error[:600],
        "deterministic_selection_guardrails": {
            "base_return_hurdle": BASE_RETURN_HURDLE,
            "selected_return_cushion_hurdle": SELECTED_RETURN_CUSHION_HURDLE,
            "guardrail_source": "runtime_selection_audit",
        },
        "allowed_tools": allowed_tools,
        "framework_template_hint": {
            "economic_model": framework_hint.get("economic_model"),
            "selected_metrics": framework_hint.get("selected_metrics") or [],
            "required_evidence": required_evidence,
            "invalid_valuation_methods": framework_hint.get("invalid_valuation_methods") or [],
        },
        "framework_required_evidence_tool_hints": _framework_required_evidence_tool_hints(
            required_evidence,
            allowed_tools,
        ),
        "framework_evidence_preflight": _framework_evidence_preflight(
            required_evidence=required_evidence,
            company_packets=company_packets,
            scenarios=scenarios,
            allowed_tools=allowed_tools,
            sector=sector,
        ),
        "candidate_context_scope": candidate_context_scope,
        "company_packets": [_planning_packet_summary(packet) for packet in prompt_packets],
        "top_base_expected_return_scenarios": _scenario_summaries_for_prompt(
            scenarios,
            tickers=prompt_tickers,
            limit=8,
        ),
    }
    return (
        "The first autonomous sector planning turn failed. Return the smallest "
        "valid tool plan only: 1-3 research questions with deterministic tool "
        "calls. Do not select a winner. Do not include a final decision. Keep all "
        "text fields short. Use framework_required_evidence_tool_hints to prefer "
        "sector-specific evidence over generic-only quality checks. Treat "
        "framework_evidence_preflight as packet-level guidance, not audit proof. "
        "Use the exact deterministic_selection_guardrails values in any "
        "hurdle-rate discussion.\n\n"
        f"Minimum planning state: {json.dumps(_jsonable(payload), sort_keys=True)}"
    )


def _recover_initial_plan_after_provider_error(
    *,
    provider: Any,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    allowed_tools: list[str],
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    provider_error: Exception,
    prompt_state_epoch: Any,
) -> dict[str, Any]:
    recovery_prompt = _minimum_tool_plan_prompt(
        sector=sector,
        market_cap_focus=market_cap_focus,
        objective=objective,
        as_of_date=as_of_date,
        allowed_tools=allowed_tools,
        company_packets=company_packets,
        scenarios=scenarios,
        provider_error=str(provider_error),
    )
    recovery_request = {
        "prompt": recovery_prompt,
        "schema": _MINIMUM_TOOL_PLAN_SCHEMA,
        "schema_name": "autonomous_sector_minimum_tool_plan",
        "max_output_tokens": 4000,
    }
    payload = _synthesize_provider_json(
        provider,
        integrity_scope=_financial_integrity_scope(
            context="autonomous_sector_minimum_tool_plan",
            as_of_date=as_of_date,
            packets=company_packets,
            scenarios=scenarios,
        ),
        _financial_prompt_integrity_binding=prompt_state_epoch.bind_request(
            provider,
            recovery_request,
        ),
        **recovery_request,
    )
    return _normalize_initial_plan_payload(
        payload, sector=sector, market_cap_focus=market_cap_focus
    )


def _recovery_final_decision_prompt(
    *,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    research_questions: list[SectorResearchQuestion],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    belief_updates: list[BeliefUpdate],
    provider_error: str,
) -> str:
    prompt_packets, candidate_context_scope = _sector_prompt_packet_scope(
        company_packets,
        scenarios=scenarios,
    )
    prompt_tickers = {packet.ticker.upper() for packet in prompt_packets}
    payload = {
        "sector": sector,
        "market_cap_focus": market_cap_focus,
        "objective": objective,
        "as_of_date": as_of_date,
        "provider_error": provider_error[:600],
        "deterministic_selection_guardrails": {
            "base_return_hurdle": BASE_RETURN_HURDLE,
            "selected_return_cushion_hurdle": SELECTED_RETURN_CUSHION_HURDLE,
            "guardrail_source": "runtime_selection_audit",
        },
        "candidate_context_scope": candidate_context_scope,
        "company_packets": [_compact_packet(packet) for packet in prompt_packets],
        "top_expected_return_scenarios": _scenario_summaries_for_prompt(
            scenarios,
            tickers=prompt_tickers,
            limit=24,
        ),
        "research_questions": [item.to_dict() for item in research_questions],
        "tool_calls": [item.to_dict() for item in tool_calls],
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "source_label": item.source_label,
                "ticker": item.ticker,
                "confidence": item.confidence,
                "summary": item.summary,
            }
            for item in evidence
        ],
        "belief_updates": [item.to_dict() for item in belief_updates],
    }
    return (
        "The previous autonomous sector turn failed structured JSON parsing. "
        "Do not continue research in this recovery call. Return only one compact "
        "final decision JSON object matching the schema. Use NO_SELECTION if the "
        "existing deterministic evidence does not clear the financial underwriting bar. "
        "Use the exact deterministic_selection_guardrails values for hurdle-rate "
        "discussion; do not invent or lower the base-return hurdle in narrative text. "
        "Keep every narrative field to two concise sentences or fewer.\n\n"
        f"Recovery state: {json.dumps(_jsonable(payload), sort_keys=True)}"
    )


def _turn_prompt(
    *,
    turn_index: int,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    budget: AutonomousRunBudget,
    allowed_tools: list[str],
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    framework: SectorFinancialFramework | None,
    research_questions: list[SectorResearchQuestion],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    belief_updates: list[BeliefUpdate],
) -> str:
    required_evidence = list(framework.required_evidence if framework else [])
    prompt_packets, candidate_context_scope = _sector_prompt_packet_scope(
        company_packets,
        scenarios=scenarios,
    )
    prompt_tickers = {packet.ticker.upper() for packet in prompt_packets}
    payload = {
        "turn_index": turn_index,
        "sector": sector,
        "market_cap_focus": market_cap_focus,
        "objective": objective,
        "as_of_date": as_of_date,
        "budget": budget.to_dict(),
        "deterministic_selection_guardrails": {
            "base_return_hurdle": BASE_RETURN_HURDLE,
            "selected_return_cushion_hurdle": SELECTED_RETURN_CUSHION_HURDLE,
            "guardrail_source": "runtime_selection_audit",
        },
        "allowed_tools": allowed_tools,
        "framework": framework.to_dict() if framework else None,
        "framework_required_evidence_tool_hints": _framework_required_evidence_tool_hints(
            required_evidence,
            allowed_tools,
        ),
        "framework_evidence_preflight": _framework_evidence_preflight(
            required_evidence=required_evidence,
            company_packets=company_packets,
            scenarios=scenarios,
            allowed_tools=allowed_tools,
            sector=framework.sector if framework is not None else sector,
        ),
        "candidate_context_scope": candidate_context_scope,
        "company_packets": [_compact_packet(packet) for packet in prompt_packets],
        "expected_return_scenarios": _scenario_summaries_for_prompt(
            scenarios,
            tickers=prompt_tickers,
            limit=24,
        ),
        "prior_research_questions": [item.to_dict() for item in research_questions],
        "tool_calls": [item.to_dict() for item in tool_calls],
        "evidence": [item.to_dict() for item in evidence],
        "belief_updates": [item.to_dict() for item in belief_updates],
    }
    return (
        "You are running an autonomous sector financial analyst loop.\n"
        "Choose the next decision-relevant research questions and deterministic "
        "tools, or stop with a final sector decision. You may continue research "
        "when another deterministic tool result could change the decision. Do "
        "not force a selected company if model fit, evidence quality, or "
        "expected return is insufficient.\n\n"
        "Return one JSON object matching the schema. If this is the first turn, "
        "include the sector financial framework you are using. If you need more "
        "evidence, set continue_research=true and include planned tool calls. If "
        "you are done, set continue_research=false and include final_decision. "
        "For turn_index greater than 1, set framework to null rather than repeating "
        "the framework. Keep narrative fields concise so the structured output does "
        "not exceed the response budget. "
        "When continuing research, prioritize unresolved framework-required evidence "
        "using framework_required_evidence_tool_hints and framework_evidence_preflight "
        "before adding generic tool calls. Do not treat preflight support as final audit evidence. "
        "In final_decision, use selection_blockers only for binding blockers that "
        "should prevent selection; use confidence_cap_reasons for non-binding "
        "uncertainty or evidence limitations. Use the exact "
        "deterministic_selection_guardrails values for hurdle-rate discussion; "
        "do not invent or lower the base-return hurdle in narrative text.\n\n"
        f"Run state: {json.dumps(_jsonable(payload), sort_keys=True)}"
    )


def _framework_from_payload(
    payload: dict[str, Any],
    *,
    sector: str,
    market_cap_focus: str,
) -> SectorFinancialFramework | None:
    raw = payload.get("framework")
    if not isinstance(raw, dict):
        return None
    raw = augment_sector_framework_payload(raw, sector=sector, market_cap_focus=market_cap_focus)
    return SectorFinancialFramework(
        sector=str(raw.get("sector") or sector),
        market_cap_focus=str(raw.get("market_cap_focus") or market_cap_focus),
        horizon_years=[int(item) for item in raw.get("horizon_years") or [5, 10]],
        economic_model=str(
            raw.get("economic_model") or "Sector-specific financial return underwriting."
        ),
        framework_contract_id=(
            str(raw["framework_contract_id"])
            if raw.get("framework_contract_id") is not None
            else None
        ),
        selected_value_drivers=[str(item) for item in raw.get("selected_value_drivers") or []],
        selected_metrics=[str(item) for item in raw.get("selected_metrics") or []],
        valid_valuation_methods=[str(item) for item in raw.get("valid_valuation_methods") or []],
        invalid_valuation_methods=[
            str(item) for item in raw.get("invalid_valuation_methods") or []
        ],
        required_evidence=[str(item) for item in raw.get("required_evidence") or []],
        applicable_screen_rule_ids=[
            str(item) for item in raw.get("applicable_screen_rule_ids") or []
        ],
        normalization_policy=dict(raw.get("normalization_policy") or {}),
        hurdle_rate_policy=dict(raw.get("hurdle_rate_policy") or {}),
        weighting_policy=dict(raw.get("weighting_policy") or {}),
        sector_specific_risks=[str(item) for item in raw.get("sector_specific_risks") or []],
    )


def _v2_structural_gate_results(
    *,
    sector: str,
    as_of_date: str,
    company_packets: list[SectorCompanyFinancialPacket],
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate and persist every v2 sector-contract screen rule per packet."""

    from app.autonomous.structural_gate import STRUCTURAL_CODES, evaluate_structural_gate

    contract_id = sector_framework_contract_id(sector, pipeline_version="v2")
    applicable_rule_ids = set(sector_framework_screen_rule_ids(sector, pipeline_version="v2"))
    results: dict[str, Any] = {}
    for packet in company_packets:
        ticker = str(packet.ticker).upper()
        price = packet.cap_stage_price
        if not isinstance(price, (int, float)):
            price = packet.current_price
        price_date = packet.cap_stage_price_as_of_date or packet.current_price_as_of_date
        price_source = packet.cap_stage_price_source or packet.current_price_source
        price_url = packet.cap_stage_price_source_url or packet.current_price_source_url
        cap_date = packet.market_cap_effective_as_of_date
        cap_source = packet.market_cap_source_name or packet.market_cap_source
        issuer_aliases = tuple(
            dict.fromkeys(
                str(item).strip().upper()
                for item in [
                    packet.issuer_primary_ticker,
                    *packet.issuer_listed_tickers,
                ]
                if str(item or "").strip()
            )
        )
        try:
            result = evaluate_structural_gate(
                ticker,
                as_of_date=as_of_date,
                price=price,
                market_cap_mm=packet.market_cap_mm,
                db_path=db_path or get_config().db_path,
                pipeline_version="v2",
                issuer_cik=packet.issuer_cik,
                aliases=issuer_aliases,
                sector_contract_id=contract_id,
                applicable_rule_ids=applicable_rule_ids,
                price_evidence_ref=(
                    f"price:{ticker}:{price_source or 'unknown'}:{price_date or as_of_date}"
                ),
                price_evidence_url=price_url,
                cap_evidence_ref=(
                    f"market-cap:{ticker}:{cap_source or 'unknown'}:{cap_date or as_of_date}"
                ),
                cap_evidence_url=packet.market_cap_source_url,
            )
            results[ticker] = result.to_dict()
        except Exception as exc:  # noqa: BLE001 - one gate failure is a visible data state
            evaluations = [
                GateEvaluation(
                    contract_id=contract_id,
                    rule_id=rule_id,
                    status="INCOMPLETE" if rule_id in applicable_rule_ids else "NOT_APPLICABLE",
                    applicable=rule_id in applicable_rule_ids,
                    reason_code=(
                        "SCREEN_EVALUATION_ERROR"
                        if rule_id in applicable_rule_ids
                        else "SECTOR_RULE_NOT_APPLICABLE"
                    ),
                    notes=[f"{type(exc).__name__}: {exc}"],
                )
                for rule_id in STRUCTURAL_CODES
            ]
            screen = ScreenResult(
                contract_id=contract_id,
                status="INCOMPLETE",
                gate_evaluations=evaluations,
                required_rule_ids=list(STRUCTURAL_CODES),
                reason_codes=["SCREEN_EVALUATION_ERROR"],
            )
            results[ticker] = {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "quarantined": False,
                "excluded_error": True,
                "triggered_codes": [],
                "degraded_codes": ["SCREEN_EVALUATION_ERROR"],
                "advisory_codes": [],
                "reasons": ["EXCLUDED_ERROR:SCREEN_EVALUATION_ERROR"],
                "details": {"error": f"{type(exc).__name__}: {exc}"},
                "contract_id": contract_id,
                "gate_evaluations": [item.to_dict() for item in evaluations],
                "screen_result": screen.to_dict(),
            }
    return results


def _runtime_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _cached_report_financial_snapshot_years(
    ticker: str,
    *,
    as_of_date: str,
) -> int:
    try:
        path = get_config().db_path
    except Exception:
        return 0
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(str(path)) as conn:
            required = {
                "ticker",
                "fiscal_year",
                "period_type",
                "period_end",
                "filed_date",
                "line_item",
                "value",
            }
            if not required <= _runtime_table_columns(conn, "companyfacts_facts"):
                return 0
            rows = companyfacts_rows(
                conn,
                ticker.upper(),
                columns=("fiscal_year", "period_end", "filed_date"),
                period_types=("FY",),
                line_items=REPORTABLE_FINANCIAL_HISTORY_LINE_ITEMS,
                as_of_date=as_of_date,
                require_filed_asof=True,
                value_not_null=True,
                order_by="fiscal_year DESC",
            )
    except sqlite3.Error:
        return 0
    return len(
        {
            row[0]
            for row in rows
            if isinstance(row[0], int) and str(row[1] or "") <= str(row[2] or "")
        }
    )


def _candidate_selection_uses_uncapped_loaded_pool(
    candidate_selection: dict[str, Any] | None,
) -> bool:
    if not isinstance(candidate_selection, dict):
        return False
    if str(candidate_selection.get("source") or "") == "explicit_tickers":
        return False
    ranking_basis = str(candidate_selection.get("ranking_basis") or "")
    return ranking_basis in {
        "consensus_pre_rank_all_loaded",
        "candidate_pool_consensus_pre_rank_all_loaded",
    }


def _pre_provider_financial_history_filter(
    *,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    signal_packets: dict[str, TickerSignalPacket],
    candidate_selection: dict[str, Any] | None,
    run_as_of_date: str,
    pipeline_version: str = "v1",
) -> tuple[
    list[SectorCompanyFinancialPacket],
    list[SectorExpectedReturnScenario],
    dict[str, TickerSignalPacket],
    dict[str, Any] | None,
    list[str],
    list[str],
]:
    if not _candidate_selection_uses_uncapped_loaded_pool(candidate_selection):
        return company_packets, scenarios, signal_packets, candidate_selection, [], []

    updated_selection = dict(candidate_selection or {})
    packet_tickers = [packet.ticker.upper() for packet in company_packets]
    year_counts = {
        ticker: _cached_report_financial_snapshot_years(
            ticker,
            as_of_date=run_as_of_date,
        )
        for ticker in packet_tickers
    }
    keep_tickers = [
        ticker
        for ticker in packet_tickers
        if year_counts.get(ticker, 0) >= REPORTABLE_FINANCIAL_HISTORY_MIN_ROWS
    ]
    excluded = [ticker for ticker in packet_tickers if ticker not in set(keep_tickers)]
    metadata = {
        "status": "NO_FILTER_ALL_CANDIDATES_REPORTABLE"
        if not excluded
        else "FILTERED_SPARSE_FINANCIAL_HISTORY",
        "minimum_rows": REPORTABLE_FINANCIAL_HISTORY_MIN_ROWS,
        "input_tickers": packet_tickers,
        "selected_tickers_before_filter": [
            str(item).upper()
            for item in updated_selection.get("selected_tickers") or packet_tickers
            if str(item).strip()
        ],
        "selected_tickers_after_filter": keep_tickers,
        "excluded_tickers": excluded,
        "year_counts": {ticker: year_counts[ticker] for ticker in excluded},
    }
    updated_selection["financial_history_filter"] = metadata
    if not excluded:
        return company_packets, scenarios, signal_packets, updated_selection, [], []

    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        metadata["status"] = "V2_RETAINED_SPARSE_FINANCIAL_HISTORY"
        metadata["selected_tickers_after_filter"] = packet_tickers
        metadata["needs_data_tickers"] = excluded
        updated_selection["financial_history_filter"] = metadata
        warnings = [str(item) for item in updated_selection.get("warnings") or []]
        warnings.append(f"SPARSE_FINANCIAL_HISTORY_RETAINED_NEEDS_DATA:{len(excluded)}")
        updated_selection["warnings"] = list(dict.fromkeys(warnings))
        return (
            company_packets,
            scenarios,
            signal_packets,
            updated_selection,
            [],
            [
                (
                    "V2 retained "
                    f"{len(excluded)} sparse-history candidate(s) as visible research states: "
                    f"{', '.join(excluded)}."
                )
            ],
        )

    keep_set = set(keep_tickers)
    filtered_packets = [packet for packet in company_packets if packet.ticker.upper() in keep_set]
    filtered_scenarios = [scenario for scenario in scenarios if scenario.ticker.upper() in keep_set]
    filtered_signal_packets = {
        ticker: packet
        for ticker, packet in signal_packets.items()
        if str(ticker).upper() in keep_set
    }
    updated_selection["selected_tickers_before_financial_history_filter"] = list(
        metadata["selected_tickers_before_filter"]
    )
    updated_selection["selected_tickers"] = keep_tickers
    updated_selection["excluded_tickers"] = list(
        dict.fromkeys(
            [
                *[
                    str(item).upper()
                    for item in updated_selection.get("excluded_tickers") or []
                    if str(item).strip()
                ],
                *excluded,
            ]
        )
    )
    warnings = [str(item) for item in updated_selection.get("warnings") or []]
    warnings.append(f"SPARSE_FINANCIAL_HISTORY_FILTERED:{len(excluded)}")
    updated_selection["warnings"] = list(dict.fromkeys(warnings))
    return (
        filtered_packets,
        filtered_scenarios,
        filtered_signal_packets,
        updated_selection,
        ["SPARSE_FINANCIAL_HISTORY_FILTERED"],
        [
            (
                "Pre-provider financial-history filter excluded "
                f"{len(excluded)} candidate(s) with fewer than {REPORTABLE_FINANCIAL_HISTORY_MIN_ROWS} "
                f"FY table rows: {', '.join(excluded)}."
            )
        ],
    )


def _pre_provider_framework_evidence_filter(
    *,
    sector: str,
    market_cap_focus: str,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    signal_packets: dict[str, TickerSignalPacket],
    allowed_tools: list[str],
    candidate_selection: dict[str, Any] | None,
    pipeline_version: str = "v1",
) -> tuple[
    list[SectorCompanyFinancialPacket],
    list[SectorExpectedReturnScenario],
    dict[str, TickerSignalPacket],
    dict[str, Any] | None,
    list[str],
    list[str],
]:
    if not isinstance(candidate_selection, dict):
        return company_packets, scenarios, signal_packets, candidate_selection, [], []

    updated_selection = dict(candidate_selection)
    source = str(updated_selection.get("source") or "")
    if source == "explicit_tickers":
        updated_selection["framework_evidence_filter"] = {
            "status": "SKIPPED_EXPLICIT_TICKERS",
            "reason": "Explicit ticker inputs are preserved; framework readiness is audited later.",
        }
        return company_packets, scenarios, signal_packets, updated_selection, [], []
    if not source:
        updated_selection["framework_evidence_filter"] = {
            "status": "SKIPPED_UNKNOWN_SOURCE",
            "reason": "Candidate source was not declared, so pre-provider framework filtering was skipped.",
        }
        return company_packets, scenarios, signal_packets, updated_selection, [], []

    framework = _framework_from_payload(
        {"framework": _default_framework_payload(sector, market_cap_focus)},
        sector=sector,
        market_cap_focus=market_cap_focus,
    )
    required_evidence = list(framework.required_evidence if framework is not None else [])
    preflight_rows = _framework_evidence_preflight(
        required_evidence=required_evidence,
        company_packets=company_packets,
        scenarios=scenarios,
        allowed_tools=allowed_tools,
        sector=sector,
    )
    packet_tickers = [packet.ticker.upper() for packet in company_packets]
    metadata: dict[str, Any] = {
        "status": "NO_REQUIRED_EVIDENCE"
        if not required_evidence
        else "NO_FILTER_ALL_CANDIDATES_READY",
        "framework_economic_model": framework.economic_model if framework is not None else None,
        "input_tickers": packet_tickers,
        "selected_tickers_before_filter": [
            str(item).upper()
            for item in updated_selection.get("selected_tickers") or packet_tickers
            if str(item).strip()
        ],
        "selected_tickers_after_filter": packet_tickers,
        "excluded_tickers": [],
        "minimum_packet_support_ratio": 0.0,
        "preflight": preflight_rows,
    }
    if not required_evidence or not preflight_rows:
        updated_selection["framework_evidence_filter"] = metadata
        return company_packets, scenarios, signal_packets, updated_selection, [], []

    support_by_ticker = {
        str(row.get("ticker") or "").upper(): float(row.get("packet_support_ratio") or 0.0)
        for row in preflight_rows
    }
    keep_tickers = [ticker for ticker in packet_tickers if support_by_ticker.get(ticker, 0.0) > 0.0]
    if not keep_tickers:
        metadata["status"] = "NO_FILTER_ALL_CANDIDATES_UNSUPPORTED"
        metadata["reason"] = (
            "No candidate had preliminary packet support for the framework; provider must gather evidence or final audit will cap actionability."
        )
        updated_selection["framework_evidence_filter"] = metadata
        warnings = [str(item) for item in updated_selection.get("warnings") or []]
        warnings.append("FRAMEWORK_EVIDENCE_PREFLIGHT_ALL_CANDIDATES_UNSUPPORTED")
        updated_selection["warnings"] = list(dict.fromkeys(warnings))
        return (
            company_packets,
            scenarios,
            signal_packets,
            updated_selection,
            [],
            [
                "Pre-provider framework evidence filter found no packet-supported candidates; retained full pool for evidence gathering."
            ],
        )

    excluded = [ticker for ticker in packet_tickers if ticker not in set(keep_tickers)]
    if not excluded:
        updated_selection["framework_evidence_filter"] = metadata
        return company_packets, scenarios, signal_packets, updated_selection, [], []

    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        metadata["status"] = "V2_RETAINED_ZERO_PACKET_SUPPORT"
        metadata["selected_tickers_after_filter"] = packet_tickers
        metadata["needs_data_tickers"] = excluded
        updated_selection["framework_evidence_filter"] = metadata
        warnings = [str(item) for item in updated_selection.get("warnings") or []]
        warnings.append(f"FRAMEWORK_ZERO_SUPPORT_RETAINED_NEEDS_DATA:{len(excluded)}")
        updated_selection["warnings"] = list(dict.fromkeys(warnings))
        return (
            company_packets,
            scenarios,
            signal_packets,
            updated_selection,
            [],
            [
                (
                    "V2 retained "
                    f"{len(excluded)} zero-support candidate(s) as visible research states: "
                    f"{', '.join(excluded)}."
                )
            ],
        )

    keep_set = set(keep_tickers)
    filtered_packets = [packet for packet in company_packets if packet.ticker.upper() in keep_set]
    filtered_scenarios = [scenario for scenario in scenarios if scenario.ticker.upper() in keep_set]
    filtered_signal_packets = {
        ticker: packet
        for ticker, packet in signal_packets.items()
        if str(ticker).upper() in keep_set
    }
    metadata["status"] = "FILTERED_ZERO_PACKET_SUPPORT"
    metadata["selected_tickers_after_filter"] = keep_tickers
    metadata["excluded_tickers"] = excluded
    updated_selection["framework_evidence_filter"] = metadata
    updated_selection["selected_tickers_before_framework_filter"] = list(
        metadata["selected_tickers_before_filter"]
    )
    updated_selection["selected_tickers"] = keep_tickers
    updated_selection["excluded_tickers"] = list(
        dict.fromkeys(
            [
                *[
                    str(item).upper()
                    for item in updated_selection.get("excluded_tickers") or []
                    if str(item).strip()
                ],
                *excluded,
            ]
        )
    )
    warnings = [str(item) for item in updated_selection.get("warnings") or []]
    warnings.append(f"FRAMEWORK_EVIDENCE_PREFLIGHT_FILTERED:{len(excluded)}")
    updated_selection["warnings"] = list(dict.fromkeys(warnings))
    return (
        filtered_packets,
        filtered_scenarios,
        filtered_signal_packets,
        updated_selection,
        ["FRAMEWORK_EVIDENCE_PREFLIGHT_FILTERED"],
        [
            f"Pre-provider framework evidence filter excluded {len(excluded)} candidate(s) with zero packet support: {', '.join(excluded)}."
        ],
    )


def _packet_has_usable_valuation_anchor(
    packet: SectorCompanyFinancialPacket,
) -> bool:
    valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    method = str(valuation.get("anchor_method") or "").strip()
    anchor = _float_or_none(valuation.get("valuation_anchor"))
    methods = valuation.get("available_methods")
    return bool(
        method
        and anchor is not None
        and anchor > 0
        and isinstance(methods, list)
        and method in {str(item) for item in methods}
    )


def _valuation_anchor_dispositions(
    candidate_selection: dict[str, Any] | None,
) -> list[CandidateDisposition]:
    anchor_filter = (
        candidate_selection.get("valuation_anchor_filter")
        if isinstance(candidate_selection, dict)
        else None
    )
    rows = anchor_filter.get("dispositions") if isinstance(anchor_filter, dict) else None
    if not isinstance(rows, list):
        return []
    dispositions: list[CandidateDisposition] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dispositions.append(CandidateDisposition.from_dict(row))
    return dispositions


def _pre_provider_valuation_anchor_filter(
    *,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    signal_packets: dict[str, TickerSignalPacket],
    candidate_selection: dict[str, Any] | None,
) -> tuple[
    list[SectorCompanyFinancialPacket],
    list[SectorExpectedReturnScenario],
    dict[str, TickerSignalPacket],
    dict[str, Any],
    list[SectorCompanyFinancialPacket],
    list[SectorExpectedReturnScenario],
]:
    """Admit paid reasoning only for packets carrying a real valuation anchor."""

    input_packets = list(company_packets)
    input_scenarios = list(scenarios)
    input_tickers = [packet.ticker.upper() for packet in input_packets]
    ready_tickers = [
        packet.ticker.upper()
        for packet in input_packets
        if _packet_has_usable_valuation_anchor(packet)
    ]
    ready_set = set(ready_tickers)
    needs_data_tickers = [ticker for ticker in input_tickers if ticker not in ready_set]
    dispositions = [
        CandidateDisposition(
            ticker=ticker,
            terminal_state="NEEDS_DATA",
            scope_status="IN_SCOPE",
            screen_status="INCOMPLETE",
            review_status="NOT_STARTED",
            watchlist_eligible=False,
            reason_codes=["MISSING_VALUATION"],
            last_completed_stage="VALUATION_ANCHOR_PREFLIGHT",
        ).to_dict()
        for ticker in needs_data_tickers
    ]
    if not needs_data_tickers:
        status = "PASS_ALL_CANDIDATES_VALUED"
    elif ready_tickers:
        status = "MIXED_VALUED_SUBSET"
    else:
        status = "NEEDS_DATA_ALL_CANDIDATES_MISSING_VALUATION"
    updated_selection = {
        **(candidate_selection or {}),
        "valuation_anchor_filter": {
            "status": status,
            "input_tickers": input_tickers,
            "ready_tickers": ready_tickers,
            "needs_data_tickers": needs_data_tickers,
            "reason_code": "MISSING_VALUATION",
            "dispositions": dispositions,
        },
    }
    if not needs_data_tickers:
        return (
            company_packets,
            scenarios,
            signal_packets,
            updated_selection,
            input_packets,
            input_scenarios,
        )

    updated_selection["selected_tickers_before_valuation_anchor_filter"] = [
        str(item).strip().upper()
        for item in updated_selection.get("selected_tickers") or input_tickers
        if str(item).strip()
    ]
    updated_selection["selected_tickers"] = ready_tickers
    warnings = [str(item) for item in updated_selection.get("warnings") or []]
    warnings.append(f"MISSING_VALUATION_NEEDS_DATA:{len(needs_data_tickers)}")
    updated_selection["warnings"] = list(dict.fromkeys(warnings))
    return (
        [packet for packet in company_packets if packet.ticker.upper() in ready_set],
        [scenario for scenario in scenarios if scenario.ticker.upper() in ready_set],
        {
            ticker: packet
            for ticker, packet in signal_packets.items()
            if str(ticker).upper() in ready_set
        },
        updated_selection,
        input_packets,
        input_scenarios,
    )


def _questions_from_payload(
    payload: dict[str, Any], turn_index: int
) -> tuple[list[SectorResearchQuestion], list[dict[str, Any]]]:
    questions: list[SectorResearchQuestion] = []
    planned_calls: list[dict[str, Any]] = []
    raw_questions = (
        payload.get("research_questions")
        if isinstance(payload.get("research_questions"), list)
        else []
    )
    for idx, raw in enumerate(raw_questions, start=1):
        if not isinstance(raw, dict):
            continue
        question_id = str(raw.get("question_id") or f"T{turn_index}Q{idx}")
        raw_calls = (
            raw.get("planned_tool_calls") if isinstance(raw.get("planned_tool_calls"), list) else []
        )
        planned_tools = [
            str(call.get("tool_name"))
            for call in raw_calls
            if isinstance(call, dict) and call.get("tool_name")
        ]
        target_tickers = [
            str(item).upper() for item in raw.get("target_tickers") or [] if str(item).strip()
        ]
        question = SectorResearchQuestion(
            question_id=question_id,
            question=str(raw.get("question") or "Unspecified sector financial question."),
            financial_pillar=str(raw.get("financial_pillar") or "Financial underwriting"),
            expected_decision_impact=str(
                raw.get("expected_decision_impact") or "Could change the sector decision."
            ),
            priority=str(raw.get("priority") or "MEDIUM"),
            status="PLANNED",
            target_tickers=target_tickers,
            planned_tools=planned_tools,
        )
        questions.append(question)
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            call_ticker = call.get("ticker")
            planned_calls.append(
                {
                    "question_id": question_id,
                    "tool_name": str(call.get("tool_name") or ""),
                    "ticker": str(call_ticker).upper() if call_ticker else None,
                    "tool_input": call.get("tool_input")
                    if isinstance(call.get("tool_input"), dict)
                    else {},
                    "rationale": str(call.get("rationale") or question.expected_decision_impact),
                    "target_tickers": target_tickers,
                }
            )
    return questions, planned_calls


def _initial_fallback_focus_ticker(
    *,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
) -> tuple[str | None, int | None]:
    ranked: list[tuple[float, int, int, str, int | None]] = []
    for original_order, packet in enumerate(company_packets):
        ticker = str(packet.ticker or "").upper()
        if not ticker:
            continue
        base_return, horizon = _best_base_return(ticker=ticker, scenarios=scenarios)
        blocked = 1 if _selection_blockers_for_packet(packet) else 0
        ranked.append(
            (
                base_return if base_return is not None else -999.0,
                -blocked,
                -original_order,
                ticker,
                horizon,
            )
        )
    if not ranked:
        return None, None
    _base_return, _blocked, _order, ticker, horizon = max(ranked)
    return ticker, horizon


def _deterministic_tool_input(tool_name: str, *, horizon_years: int | None) -> dict[str, Any]:
    if tool_name == "rank_expected_return_cases":
        return {"horizon_years": horizon_years or 5, "scenario_name": "base"}
    if tool_name == "compare_expected_return_scenarios":
        return {"horizon_years": horizon_years or 5, "scenario_name": "base"}
    if tool_name == "fetch_recent_filing_context":
        return {
            "quarters": 2,
            "material_event_window_days": 180,
            "max_documents": 4,
            "max_chars": 2200,
        }
    if tool_name == "fetch_current_events":
        return {"max_items": 5}
    if tool_name == "fetch_companyfacts_timeseries":
        return {
            "line_items": [
                "revenue",
                "operating_income",
                "cfo",
                "capex",
                "shares_outstanding",
            ],
            "years": 5,
        }
    if tool_name == "fetch_filing_section":
        return {
            "keywords": ["risk factors", "liquidity", "competition", "revenue", "cash flow"],
            "max_chars": 3000,
        }
    if tool_name == "analyze_dilution":
        return {"years": 5}
    if tool_name == "analyze_capital_structure_resolution":
        return {"max_chars": 2200}
    if tool_name == "compare_peer_metric":
        return {"metric": "roic"}
    return {}


def _deterministic_initial_tool_plan(
    *,
    framework: SectorFinancialFramework | None,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    allowed_tools: list[str],
) -> tuple[list[SectorResearchQuestion], list[dict[str, Any]], list[str]]:
    """Build a small evidence plan when the first provider turn returns no tools."""

    allowed = {str(tool) for tool in allowed_tools}
    focus_ticker, horizon_years = _initial_fallback_focus_ticker(
        company_packets=company_packets,
        scenarios=scenarios,
    )
    if not focus_ticker:
        return [], [], ["Deterministic initial fallback could not choose a focus ticker."]

    all_tickers = [
        str(packet.ticker).upper() for packet in company_packets if str(packet.ticker).strip()
    ]
    required_evidence = list(framework.required_evidence if framework is not None else [])
    missing_required_evidence: list[str] = []
    if framework is not None:
        preflight_rows = _framework_evidence_preflight(
            required_evidence=required_evidence,
            company_packets=company_packets,
            scenarios=scenarios,
            allowed_tools=allowed_tools,
            sector=framework.sector,
        )
        for row in preflight_rows:
            if str(row.get("ticker") or "").upper() == focus_ticker:
                missing_required_evidence = [
                    str(item) for item in row.get("needs_tool_evidence") or [] if str(item).strip()
                ]
                break

    tool_candidates: list[str] = []

    def add_tool(tool_name: str) -> None:
        if tool_name in allowed and tool_name not in tool_candidates:
            tool_candidates.append(tool_name)

    add_tool("rank_expected_return_cases")
    evidence_hints = _framework_required_evidence_tool_hints(
        missing_required_evidence or required_evidence,
        allowed_tools,
    )
    hinted_tools = [
        str(tool)
        for row in evidence_hints
        for tool in row.get("suggested_tools") or []
        if str(tool).strip()
    ]
    priority = [
        "fetch_kpi_trends",
        "analyze_dilution",
        "fetch_recent_filing_context",
        "fetch_companyfacts_timeseries",
        "analyze_liquidity_stress",
        "analyze_capital_allocation",
        "fetch_current_events",
        "fetch_filing_section",
        "compare_peer_metric",
        "analyze_capital_structure_resolution",
        "compare_expected_return_scenarios",
        "summarize_financial_packets",
    ]
    for tool_name in priority:
        if tool_name in hinted_tools:
            add_tool(tool_name)
    for tool_name in ("fetch_kpi_trends", "analyze_liquidity_stress"):
        add_tool(tool_name)
    planned_tool_names = tool_candidates[:4]
    if not planned_tool_names:
        return [], [], ["Deterministic initial fallback found no allowed evidence tools."]

    question = SectorResearchQuestion(
        question_id="DTP1",
        question=(
            f"Deterministic fallback: rank sector candidates and gather decision evidence for {focus_ticker}."
        ),
        financial_pillar="Initial evidence fallback",
        expected_decision_impact=(
            "Prevents an empty first provider plan from stopping the autonomous sector run before evidence collection."
        ),
        priority="HIGH",
        status="PLANNED",
        target_tickers=all_tickers,
        planned_tools=planned_tool_names,
    )
    planned_calls: list[dict[str, Any]] = []
    for tool_name in planned_tool_names:
        planned_calls.append(
            {
                "question_id": question.question_id,
                "tool_name": tool_name,
                "ticker": None if tool_name in SECTOR_TOOL_NAMES else focus_ticker,
                "tool_input": _deterministic_tool_input(tool_name, horizon_years=horizon_years),
                "rationale": (
                    "Deterministic initial fallback after provider returned no executable sector tool plan."
                ),
                "target_tickers": all_tickers if tool_name in SECTOR_TOOL_NAMES else [focus_ticker],
            }
        )
    notes = [
        (
            "Provider returned no executable first-turn sector tool plan; "
            f"runtime created a deterministic fallback plan focused on {focus_ticker}."
        )
    ]
    if missing_required_evidence:
        notes.append(
            "Fallback targeted missing framework evidence: "
            + ", ".join(missing_required_evidence[:6])
            + (
                "."
                if len(missing_required_evidence) <= 6
                else f", plus {len(missing_required_evidence) - 6} more."
            )
        )
    return [question], planned_calls, notes


def _belief_updates_from_payload(payload: dict[str, Any], start_index: int) -> list[BeliefUpdate]:
    updates: list[BeliefUpdate] = []
    raw_updates = (
        payload.get("belief_updates") if isinstance(payload.get("belief_updates"), list) else []
    )
    for idx, raw in enumerate(raw_updates, start=start_index):
        if not isinstance(raw, dict):
            continue
        updates.append(
            BeliefUpdate(
                update_id=f"BU{idx}",
                question_id=raw.get("question_id"),
                ticker=str(raw.get("ticker")).upper() if raw.get("ticker") else None,
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


def _final_decision_from_payload(payload: dict[str, Any]) -> SectorFinalDecision | None:
    raw = payload.get("final_decision")
    if not isinstance(raw, dict):
        return None
    return SectorFinalDecision(
        verdict=str(raw.get("verdict") or "NO_SELECTION"),
        confidence=str(raw.get("confidence")) if raw.get("confidence") else None,
        selected_ticker=str(raw.get("selected_ticker")).upper()
        if raw.get("selected_ticker")
        else None,
        expected_annualized_return_range=(
            str(raw.get("expected_annualized_return_range"))
            if raw.get("expected_annualized_return_range")
            else None
        ),
        thesis=str(raw.get("thesis") or ""),
        key_risk=str(raw.get("key_risk") or ""),
        downside_case=str(raw.get("downside_case") or ""),
        no_selection_reason=str(raw.get("no_selection_reason"))
        if raw.get("no_selection_reason")
        else None,
        falsifiers=[str(item) for item in raw.get("falsifiers") or []],
        why_selected_over_finalists=[
            str(item) for item in raw.get("why_selected_over_finalists") or []
        ],
        rejected_finalists=[
            dict(item) for item in raw.get("rejected_finalists") or [] if isinstance(item, dict)
        ],
        selection_blockers=[str(item) for item in raw.get("selection_blockers") or []],
        confidence_cap_reasons=[str(item) for item in raw.get("confidence_cap_reasons") or []],
        evidence_ref_ids=[str(item) for item in raw.get("evidence_ref_ids") or []],
    )


def _mark_question_statuses(
    questions: list[SectorResearchQuestion], tool_calls: list[ToolCallRecord]
) -> None:
    by_question: dict[str, list[str]] = {}
    evidence_by_question: dict[str, list[str]] = {}
    for call in tool_calls:
        if call.question_id:
            by_question.setdefault(call.question_id, []).append(call.status)
            evidence_by_question.setdefault(call.question_id, []).extend(call.evidence_ref_ids)
    for question in questions:
        statuses = by_question.get(question.question_id, [])
        question.evidence_ref_ids = list(
            dict.fromkeys(evidence_by_question.get(question.question_id, []))
        )
        if not statuses:
            question.status = "OPEN"
        elif any(status == "OK" for status in statuses):
            question.status = "ANSWERED"
        elif all(status.startswith("SKIPPED") for status in statuses):
            question.status = "SKIPPED"
        else:
            question.status = "PARTIAL"


def _selection_blockers_for_packet(packet: SectorCompanyFinancialPacket) -> list[str]:
    """Return deterministic hard blockers that make a final selection invalid."""

    blockers = [str(item) for item in packet.blockers if str(item).strip()]
    if str(packet.model_fit_status or "").upper() == "BLOCKED":
        blockers.append("MODEL_FIT_BLOCKED")
    if packet.current_price is None:
        blockers.append("MISSING_PRICE")
    valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    anchor = valuation.get("valuation_anchor")
    if not isinstance(anchor, (int, float)) or isinstance(anchor, bool) or anchor <= 0:
        blockers.append("MISSING_VALUATION")
    data_quality_status = str(packet.data_quality_status or "").upper()
    if data_quality_status in {"MISSING_PRICE", "MISSING_VALUATION", "MODEL_BLOCKED"}:
        blockers.append(data_quality_status)
    return list(dict.fromkeys(blockers))


def _capital_loss_underwriting_for_packet(
    packet: SectorCompanyFinancialPacket,
    *,
    refinancing_timeline_status: str | None = None,
) -> dict[str, Any]:
    business_quality = packet.business_quality if isinstance(packet.business_quality, dict) else {}
    detail = business_quality.get("impairment_classification")
    if not isinstance(detail, dict):
        detail = {}

    impairment_class = (
        str(
            detail.get("impairment_class_primary")
            or business_quality.get("impairment_class_primary")
            or ""
        )
        .strip()
        .upper()
    )
    if not impairment_class:
        detail = _fallback_capital_loss_detail(
            packet,
            refinancing_timeline_status=refinancing_timeline_status,
        )
        impairment_class = str(detail.get("impairment_class_primary") or "").strip().upper()
        if not impairment_class:
            return {
                "status": "NOT_AVAILABLE",
                "impairment_class_primary": None,
                "primary_underwriting_caution": None,
                "reason_codes": [],
                "support_signals": [],
                "rebuttal_signals": [],
                "hard_blockers": [],
                "confidence_caps": [],
            }

    return {
        "status": CAPITAL_LOSS_STATUS_BY_IMPAIRMENT_CLASS.get(impairment_class, "UNKNOWN"),
        "impairment_class_primary": impairment_class,
        "primary_underwriting_caution": (
            detail.get("primary_underwriting_caution")
            or business_quality.get("primary_underwriting_caution")
        ),
        "reason_codes": [str(item) for item in (detail.get("impairment_class_reason_codes") or [])],
        "support_signals": [str(item) for item in (detail.get("impairment_support_signals") or [])],
        "rebuttal_signals": [
            str(item) for item in (detail.get("impairment_rebuttal_signals") or [])
        ],
        "hard_blockers": [CAPITAL_LOSS_IMPAIRMENT_BLOCKERS[impairment_class]]
        if impairment_class in CAPITAL_LOSS_IMPAIRMENT_BLOCKERS
        else [],
        "confidence_caps": [CAPITAL_LOSS_IMPAIRMENT_CAPS[impairment_class]]
        if impairment_class in CAPITAL_LOSS_IMPAIRMENT_CAPS
        else [],
    }


def _fallback_capital_loss_detail(
    packet: SectorCompanyFinancialPacket,
    *,
    refinancing_timeline_status: str | None = None,
) -> dict[str, Any]:
    """Infer a conservative capital-loss class when upstream classification is absent."""

    balance_sheet = packet.balance_sheet if isinstance(packet.balance_sheet, dict) else {}
    business_quality = packet.business_quality if isinstance(packet.business_quality, dict) else {}
    solvency_risk = str(balance_sheet.get("solvency_risk") or "").strip().upper()
    blockers = {str(item).strip().upper() for item in packet.blockers if str(item).strip()}
    caps = {str(item).strip().upper() for item in packet.confidence_caps if str(item).strip()}
    valuation_headwinds = {
        str(item).strip().upper()
        for item in (business_quality.get("valuation_headwinds") or [])
        if str(item).strip()
    }
    financial_status = str(packet.financial_status or "").strip().upper()
    data_quality_status = str(packet.data_quality_status or "").strip().upper()
    timeline_status = str(refinancing_timeline_status or "").strip().upper()

    if (
        solvency_risk == "CRITICAL"
        or "SOLVENCY_CRITICAL" in blockers
        or financial_status == "BALANCE-SHEET CONSTRAINED"
    ):
        return {
            "impairment_class_primary": "PROBABLE_IMPAIRMENT",
            "primary_underwriting_caution": "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT",
            "impairment_class_reason_codes": ["FALLBACK_SOLVENCY_CRITICAL"],
            "impairment_support_signals": ["CRITICAL_SOLVENCY_OR_BALANCE_SHEET_CONSTRAINT"],
            "impairment_rebuttal_signals": [],
        }

    if _optional_bool(balance_sheet.get("negative_equity")) is True:
        return {
            "impairment_class_primary": "PROBABLE_IMPAIRMENT",
            "primary_underwriting_caution": "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT",
            "impairment_class_reason_codes": ["FALLBACK_NEGATIVE_EQUITY"],
            "impairment_support_signals": ["NEGATIVE_EQUITY_PRESENT"],
            "impairment_rebuttal_signals": [],
        }

    current_ratio = _optional_number(balance_sheet.get("current_ratio"))
    if current_ratio is not None and current_ratio < 0.5:
        return {
            "impairment_class_primary": "PROBABLE_IMPAIRMENT",
            "primary_underwriting_caution": "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT",
            "impairment_class_reason_codes": ["FALLBACK_CURRENT_RATIO_LIQUIDITY_CRISIS"],
            "impairment_support_signals": ["CURRENT_RATIO_BELOW_HALF"],
            "impairment_rebuttal_signals": [],
        }

    # A going-concern block rests on a stored, filed, blockable assertion — not on the
    # bare flag, which an old cached payload or a summary that dropped the assertions
    # can still carry (a $100B issuer was blocked on one with no excerpt behind it).
    if _packet_going_concern_asserted(balance_sheet):
        return {
            "impairment_class_primary": "PROBABLE_IMPAIRMENT",
            "primary_underwriting_caution": "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT",
            "impairment_class_reason_codes": ["FALLBACK_GOING_CONCERN_LANGUAGE"],
            "impairment_support_signals": ["GOING_CONCERN_LANGUAGE_PRESENT"],
            "impairment_rebuttal_signals": [],
        }

    if _optional_bool(balance_sheet.get("no_assurance_financing")) is True:
        return {
            "impairment_class_primary": "EVIDENCE_DEGRADED_NOT_ASSESSABLE",
            "primary_underwriting_caution": "POSSIBLE_EVIDENCE_GAP",
            "impairment_class_reason_codes": ["FALLBACK_NO_ASSURANCE_FINANCING"],
            "impairment_support_signals": ["NO_ASSURANCE_FINANCING_UNRESOLVED"],
            "impairment_rebuttal_signals": [],
        }

    if timeline_status == "LIMITED_CASH_RUNWAY":
        return {
            "impairment_class_primary": "EVIDENCE_DEGRADED_NOT_ASSESSABLE",
            "primary_underwriting_caution": "POSSIBLE_EVIDENCE_GAP",
            "impairment_class_reason_codes": ["FALLBACK_LIMITED_CASH_RUNWAY"],
            "impairment_support_signals": ["LIMITED_CASH_RUNWAY_UNRESOLVED"],
            "impairment_rebuttal_signals": [],
        }

    if (
        data_quality_status in {"NO_FILING", "RISK_SECTION_NOT_FOUND"}
        or caps & FALLBACK_IMPAIRMENT_DEGRADED_CAPS
    ):
        return {
            "impairment_class_primary": "EVIDENCE_DEGRADED_NOT_ASSESSABLE",
            "primary_underwriting_caution": "POSSIBLE_EVIDENCE_GAP",
            "impairment_class_reason_codes": ["FALLBACK_IMPAIRMENT_EVIDENCE_GAP"],
            "impairment_support_signals": ["FILING_RISK_EVIDENCE_GAP"],
            "impairment_rebuttal_signals": [],
        }

    if (
        "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in valuation_headwinds
        or "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in caps
    ):
        return {
            "impairment_class_primary": "STRUCTURALLY_WEAK_NOT_IMPAIRED",
            "primary_underwriting_caution": "POSSIBLE_STRUCTURAL_ECONOMIC_WEAKNESS",
            "impairment_class_reason_codes": ["FALLBACK_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"],
            "impairment_support_signals": ["CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"],
            "impairment_rebuttal_signals": [],
        }

    structural_valuation_headwinds = [
        signal
        for signal in sorted(FALLBACK_STRUCTURAL_VALUATION_HEADWINDS)
        if signal in valuation_headwinds
    ]
    if len(structural_valuation_headwinds) >= 2:
        return {
            "impairment_class_primary": "STRUCTURALLY_WEAK_NOT_IMPAIRED",
            "primary_underwriting_caution": "POSSIBLE_STRUCTURAL_ECONOMIC_WEAKNESS",
            "impairment_class_reason_codes": ["FALLBACK_CLUSTERED_VALUATION_HEADWINDS"],
            "impairment_support_signals": structural_valuation_headwinds,
            "impairment_rebuttal_signals": [],
        }

    structural_weakness_signals = [
        signal
        for signal in (
            "FINANCIAL_ANOMALIES_PRESENT",
            "HIGH_GROWTH_DEPENDENCY",
            "METHOD_TENSION_GROWTH_VS_EARNINGS_POWER",
            "METHOD_TENSION_GROWTH_VS_INTRINSIC_VALUE",
            "METHOD_TENSION_QUALITY_VS_VALUATION",
        )
        if signal in caps
    ]
    if len(structural_weakness_signals) >= 2:
        return {
            "impairment_class_primary": "STRUCTURALLY_WEAK_NOT_IMPAIRED",
            "primary_underwriting_caution": "POSSIBLE_STRUCTURAL_ECONOMIC_WEAKNESS",
            "impairment_class_reason_codes": ["FALLBACK_STRUCTURAL_WEAKNESS_CAPS"],
            "impairment_support_signals": structural_weakness_signals,
            "impairment_rebuttal_signals": [],
        }

    return {}


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _packet_going_concern_asserted(balance_sheet: dict[str, Any]) -> bool:
    """Whether the packet's balance sheet carries a filed going-concern assertion."""
    return going_concern_asserted(
        {
            "going_concern_language": _optional_bool(balance_sheet.get("going_concern_language")),
            "going_concern_assertions": balance_sheet.get("going_concern_assertions"),
        }
    )


def _refinancing_timeline_for_packet(
    packet: SectorCompanyFinancialPacket,
    *,
    capital_structure_status: str | None,
    capital_structure_summary: str | None,
) -> dict[str, Any]:
    balance_sheet = packet.balance_sheet if isinstance(packet.balance_sheet, dict) else {}
    debt_due = _optional_bool(balance_sheet.get("debt_due_within_12mo"))
    going_concern = _optional_bool(balance_sheet.get("going_concern_language"))
    # Only a stored blockable assertion makes going-concern language a hard blocker; the
    # bare flag is reported as unsupported evidence below and blocks nothing.
    going_concern_supported = _packet_going_concern_asserted(balance_sheet)
    no_assurance = _optional_bool(balance_sheet.get("no_assurance_financing"))
    cash_runway = _optional_number(balance_sheet.get("cash_runway_quarters"))
    solvency_risk = str(balance_sheet.get("solvency_risk") or "").strip().upper() or None
    status = str(capital_structure_status or "").strip().upper()
    needs: list[str] = []
    hard_blockers: list[str] = []
    confidence_caps: list[str] = []

    if status == "CAPITAL_STRUCTURE_RESOLVED_CLEAR":
        timeline_status = "RESOLVED_CURRENT_EVIDENCE"
    elif status == "CAPITAL_STRUCTURE_ACTIVE_DISTRESS":
        timeline_status = "ACTIVE_DISTRESS"
    elif status == "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST":
        timeline_status = "WATCHLIST_CURRENT_EVIDENCE"
        needs.append("CURRENT_MATURITY_OR_COVENANT_FOLLOW_UP")
    elif going_concern_supported:
        timeline_status = "GOING_CONCERN_UNRESOLVED"
        needs.append("GOING_CONCERN_RESOLUTION")
        hard_blockers.append("PACKET_GOING_CONCERN_LANGUAGE")
    elif no_assurance is True:
        timeline_status = "NO_ASSURANCE_FINANCING_UNRESOLVED"
        needs.append("NO_ASSURANCE_FINANCING_RESOLUTION")
        confidence_caps.append("NO_ASSURANCE_FINANCING_UNRESOLVED")
    elif debt_due is True:
        timeline_status = "NEAR_TERM_MATURITY_UNRESOLVED"
        needs.append("NEAR_TERM_MATURITY_SCHEDULE")
        confidence_caps.append("NEAR_TERM_MATURITY_UNRESOLVED")
    elif cash_runway is not None and cash_runway < 8:
        timeline_status = "LIMITED_CASH_RUNWAY"
        needs.append("CASH_RUNWAY_BRIDGE")
        confidence_caps.append("LIMITED_CASH_RUNWAY")
    elif (
        any(value is not None for value in (debt_due, going_concern, no_assurance, cash_runway))
        or solvency_risk
    ):
        timeline_status = "NO_NEAR_TERM_REFINANCING_FLAG"
    else:
        timeline_status = "NOT_AVAILABLE"

    if going_concern is True and not going_concern_supported:
        needs.append("GOING_CONCERN_ASSERTION_EXCERPT")
        confidence_caps.append("GOING_CONCERN_LANGUAGE_UNSUPPORTED")

    if not capital_structure_summary:
        if timeline_status == "NOT_AVAILABLE":
            summary = "No packet-level refinancing, maturity, covenant, or cash-runway timeline was available."
        else:
            summary = (
                f"Packet timeline flags: debt_due_within_12mo={debt_due}, "
                f"going_concern_language={going_concern}, no_assurance_financing={no_assurance}, "
                f"cash_runway_quarters={cash_runway}, solvency_risk={solvency_risk or 'UNKNOWN'}."
            )
    else:
        summary = capital_structure_summary

    return {
        "status": timeline_status,
        "needs": needs,
        "summary": summary,
        "debt_due_within_12mo": debt_due,
        "going_concern_language": going_concern,
        "no_assurance_financing": no_assurance,
        "cash_runway_quarters": cash_runway,
        "hard_blockers": hard_blockers,
        "confidence_caps": confidence_caps,
    }


def _text_mentions_ticker(text: str | None, ticker: str) -> bool:
    return ticker.upper() in str(text or "").upper()


def _tool_call_mentions_ticker(call: ToolCallRecord, ticker: str) -> bool:
    return _text_mentions_ticker(
        json.dumps(_jsonable(call.tool_input), sort_keys=True), ticker
    ) or _text_mentions_ticker(call.output_preview, ticker)


def _tool_call_scope_includes_ticker(call: ToolCallRecord, ticker: str) -> bool:
    tool_input = call.tool_input if isinstance(call.tool_input, dict) else {}
    scoped_tickers = tool_input.get("tickers")
    if isinstance(scoped_tickers, list):
        return any(str(item).upper() == ticker.upper() for item in scoped_tickers)
    explicit_ticker = tool_input.get("ticker")
    if explicit_ticker:
        return str(explicit_ticker).upper() == ticker.upper()
    return _tool_call_mentions_ticker(call, ticker)


def _evidence_mentions_ticker(item: EvidenceReference, ticker: str) -> bool:
    return (
        str(item.ticker or "").upper() == ticker.upper()
        or _text_mentions_ticker(item.summary, ticker)
        or _text_mentions_ticker(item.excerpt, ticker)
    )


def _best_base_return(
    *,
    ticker: str,
    scenarios: list[SectorExpectedReturnScenario],
) -> tuple[float | None, int | None]:
    best_return: float | None = None
    best_horizon: int | None = None
    for scenario in scenarios:
        if scenario.ticker.upper() != ticker.upper():
            continue
        if scenario.scenario_name.lower() != "base":
            continue
        if scenario.horizon_years not in {5, 10}:
            continue
        if scenario.annualized_return is None:
            continue
        if best_return is None or scenario.annualized_return > best_return:
            best_return = float(scenario.annualized_return)
            best_horizon = int(scenario.horizon_years)
    return best_return, best_horizon


def _best_downside_return(
    *,
    ticker: str,
    scenarios: list[SectorExpectedReturnScenario],
) -> tuple[float | None, int | None]:
    worst_return: float | None = None
    worst_horizon: int | None = None
    for scenario in scenarios:
        if scenario.ticker.upper() != ticker.upper():
            continue
        if scenario.scenario_name.lower() != "downside":
            continue
        if scenario.horizon_years not in {5, 10}:
            continue
        if scenario.annualized_return is None:
            continue
        if worst_return is None or scenario.annualized_return < worst_return:
            worst_return = float(scenario.annualized_return)
            worst_horizon = int(scenario.horizon_years)
    return worst_return, worst_horizon


def _selected_evidence_profile(
    *,
    ticker: str,
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
) -> dict[str, Any]:
    expected_return_count = 0
    company_specific_count = 0
    has_freshness_evidence = False
    has_unavailable_current_events = False
    has_usable_current_events = False
    company_tools: list[str] = []
    evidence_pillars: list[str] = []
    risk_tools: list[str] = []
    evidence_text_by_pillar: dict[str, list[str]] = {}
    tool_calls_by_id = {call.call_id: call for call in tool_calls}
    for item in evidence:
        if not _confidence_at_least(item.confidence, "MODERATE"):
            if item.source_label == "fetch_current_events" and _evidence_mentions_ticker(
                item, ticker
            ):
                has_unavailable_current_events = True
            continue
        if item.source_label in EXPECTED_RETURN_EVIDENCE_TOOLS:
            linked_call = tool_calls_by_id.get(str(item.tool_call_id or ""))
            if (
                linked_call is not None
                and linked_call.status == "OK"
                and linked_call.tool_name in EXPECTED_RETURN_EVIDENCE_TOOLS
                and _tool_call_scope_includes_ticker(linked_call, ticker)
            ):
                expected_return_count += 1
            elif linked_call is None and _evidence_mentions_ticker(item, ticker):
                expected_return_count += 1
        if (
            item.source_label not in SECTOR_TOOL_NAMES
            and str(item.ticker or "").upper() == ticker.upper()
        ):
            company_specific_count += 1
            company_tools.append(item.source_label)
            pillar = COMPANY_EVIDENCE_PILLARS.get(item.source_label)
            if pillar:
                evidence_pillars.append(pillar)
                evidence_text_by_pillar.setdefault(pillar, []).append(
                    " ".join(
                        [
                            str(item.source_label or ""),
                            str(item.summary or ""),
                            str(item.excerpt or ""),
                        ]
                    ).lower()
                )
            if item.source_label in SELECTED_RISK_EVIDENCE_TOOLS:
                risk_tools.append(item.source_label)
            if item.source_label in FRESHNESS_EVIDENCE_TOOLS:
                has_freshness_evidence = True
            if item.source_label == "fetch_current_events":
                has_usable_current_events = True
    return {
        "expected_return_count": expected_return_count,
        "company_specific_count": company_specific_count,
        "has_freshness_evidence": has_freshness_evidence,
        "has_usable_current_events": has_usable_current_events,
        "current_events_low": has_unavailable_current_events and not has_usable_current_events,
        "selected_company_evidence_tools": sorted(set(company_tools)),
        "selected_evidence_pillars": sorted(set(evidence_pillars)),
        "selected_risk_evidence_tools": sorted(set(risk_tools)),
        "selected_evidence_text_by_pillar": evidence_text_by_pillar,
    }


def _profile_has_requirement_text(
    evidence_profile: dict[str, Any],
    *,
    pillars: set[str],
    tokens: tuple[str, ...],
) -> bool:
    text_by_pillar = evidence_profile.get("selected_evidence_text_by_pillar")
    if not isinstance(text_by_pillar, dict):
        return False
    wanted = tuple(token.lower() for token in tokens)
    for pillar in pillars:
        entries = text_by_pillar.get(pillar) or []
        for entry in entries:
            text = str(entry or "").lower()
            if any(token in text for token in wanted):
                return True
    return False


def _framework_requirement_covered(
    requirement: str,
    *,
    evidence_profile: dict[str, Any],
    packet: SectorCompanyFinancialPacket,
    sector: str | None = None,
) -> bool:
    normalized = str(requirement or "").strip().lower()
    if not normalized:
        return True

    tools = {str(item) for item in evidence_profile.get("selected_company_evidence_tools") or []}
    pillars = {str(item) for item in evidence_profile.get("selected_evidence_pillars") or []}
    expected_count = int(evidence_profile.get("expected_return_count") or 0)
    company_count = int(evidence_profile.get("company_specific_count") or 0)
    has_freshness = bool(
        evidence_profile.get("has_freshness_evidence")
        or evidence_profile.get("has_usable_current_events")
    )
    model_fit_clear = not _selection_blockers_for_packet(packet)

    if normalized in {"expected_return_evidence", "expected_return_scenarios"}:
        return expected_count >= 1
    if normalized in {"company_specific_evidence", "company_specific_evidence_packet"}:
        return company_count >= 1
    if normalized in {"filing_or_current_evidence", "latest_current_event_or_filing_update"}:
        return has_freshness or bool({"filing_freshness", "current_events"} & pillars)
    if normalized in {"kpi_trends", "quality_metrics"}:
        return "fetch_kpi_trends" in tools or "quality" in pillars

    if any(token in normalized for token in ("payment", "transaction", "take_rate", "merchant")):
        return bool({"quality", "companyfacts", "filing_freshness"} & pillars)
    if any(token in normalized for token in ("fraud", "chargeback", "credit_loss")):
        return bool({"liquidity", "filing_freshness", "current_events", "companyfacts"} & pillars)
    if any(token in normalized for token in ("funding", "float", "regulatory_capital")):
        return bool({"liquidity", "filing_freshness", "companyfacts"} & pillars)
    if any(token in normalized for token in ("partner", "compliance")):
        return bool(
            {"quality", "filing_freshness", "current_events", "management_commentary"} & pillars
        )
    medical_device_requirement_kind = _medical_device_requirement_kind(normalized, sector=sector)
    if medical_device_requirement_kind == "reimbursement":
        return _profile_has_requirement_text(
            evidence_profile,
            pillars={"filing_freshness", "current_events", "management_commentary"},
            tokens=(
                "reimbursement",
                "payer",
                "site of care",
                "site_of_care",
                "hospital outpatient",
                "hospital purchasing",
            ),
        )
    if medical_device_requirement_kind == "regulatory":
        return _profile_has_requirement_text(
            evidence_profile,
            pillars={"filing_freshness", "current_events", "management_commentary"},
            tokens=(
                "fda",
                "510(k)",
                "clearance",
                "quality system",
                "quality_system",
                "recall",
                "warning letter",
                "regulatory",
            ),
        )
    if medical_device_requirement_kind == "channel":
        return _profile_has_requirement_text(
            evidence_profile,
            pillars={"filing_freshness", "current_events", "management_commentary"},
            tokens=(
                "channel",
                "distributor",
                "customer concentration",
                "hospital capex",
                "hospital purchasing",
                "capital equipment",
            ),
        )
    if any(
        token in normalized
        for token in (
            "unit_volume",
            "volume_mix",
            "unit_sales",
            "incentive",
            "inventory",
            "channel",
        )
    ):
        return bool({"quality", "companyfacts", "filing_freshness"} & pillars)
    if any(token in normalized for token in ("platform", "powertrain", "tooling")):
        return bool(
            {"companyfacts", "filing_freshness", "current_events", "management_commentary"}
            & pillars
        )
    if any(token in normalized for token in ("warranty", "recall")):
        return bool({"quality", "filing_freshness", "current_events"} & pillars)
    if any(token in normalized for token in ("residual_value", "leverage", "finance")):
        return bool({"liquidity", "companyfacts", "filing_freshness"} & pillars)
    if any(
        token in normalized
        for token in ("subscriber", "audience", "arpu", "advertising", "affiliate")
    ):
        return bool({"quality", "companyfacts", "filing_freshness", "current_events"} & pillars)
    if any(token in normalized for token in ("spectrum", "distribution", "content", "network")):
        return bool(
            {"companyfacts", "filing_freshness", "current_events", "management_commentary"}
            & pillars
        )
    if any(
        token in normalized
        for token in (
            "procedure",
            "utilization",
            "installed_base",
            "consumables",
            "site_of_care",
            "hospital_capex",
        )
    ):
        return bool({"quality", "companyfacts", "filing_freshness", "current_events"} & pillars)
    if any(token in normalized for token in ("fda", "clearance", "quality_system", "recall")):
        return bool(
            {"quality", "filing_freshness", "current_events", "management_commentary"} & pillars
        )
    if any(
        token in normalized
        for token in (
            "rate_case",
            "rate_base",
            "allowed_roe",
            "equity_ratio",
            "affordability",
            "customer_bill",
            "load_growth",
            "dividend",
        )
    ):
        return bool({"filing_freshness", "companyfacts", "liquidity", "current_events"} & pillars)
    if any(
        token in normalized
        for token in ("commodity", "feedstock", "cost_curve", "cash_cost", "capacity_utilization")
    ):
        return bool({"quality", "companyfacts", "filing_freshness", "current_events"} & pillars)
    if any(token in normalized for token in ("environmental", "reclamation")):
        return bool({"filing_freshness", "current_events", "liquidity"} & pillars)
    if "security_type" in normalized or "common_equity" in normalized:
        return model_fit_clear
    if any(
        token in normalized
        for token in ("capital", "solvency", "debt", "liquidity", "refinancing", "rate")
    ):
        return bool({"liquidity", "insurance", "companyfacts"} & pillars)
    if any(
        token in normalized
        for token in ("book_value", "reserve", "credit", "underwriting", "spread")
    ):
        return bool({"insurance", "companyfacts", "quality", "filing_freshness"} & pillars)
    if any(
        token in normalized
        for token in ("noi", "occupancy", "lease", "nav", "cap_rate", "property", "ffo", "affo")
    ):
        return bool({"filing_freshness", "companyfacts", "quality", "liquidity"} & pillars)
    if any(
        token in normalized
        for token in ("reimbursement", "payer", "clinical", "regulatory", "milestone")
    ):
        return bool(
            {"filing_freshness", "current_events", "management_commentary", "quality"} & pillars
        )
    if any(token in normalized for token in ("margin", "cash_conversion", "cash", "fcf", "gross")):
        return bool({"quality", "companyfacts"} & pillars)
    if any(
        token in normalized
        for token in (
            "backlog",
            "order",
            "pricing",
            "input_cost",
            "cycle",
            "working_capital",
            "capex",
        )
    ):
        return bool({"quality", "companyfacts", "filing_freshness"} & pillars)
    if any(
        token in normalized
        for token in ("retention", "churn", "concentration", "customer", "product")
    ):
        return bool(
            {"quality", "filing_freshness", "current_events", "management_commentary"} & pillars
        )
    if any(token in normalized for token in ("sbc", "share_count", "dilution")):
        return bool({"dilution", "companyfacts"} & pillars)

    return company_count >= 1


def _framework_required_evidence_coverage(
    *,
    framework: SectorFinancialFramework | None,
    evidence_profile: dict[str, Any],
    packet: SectorCompanyFinancialPacket,
) -> dict[str, Any]:
    required = [
        str(item).strip()
        for item in ((framework.required_evidence if framework is not None else []) or [])
        if str(item).strip()
    ]
    covered = [
        item
        for item in required
        if _framework_requirement_covered(
            item,
            evidence_profile=evidence_profile,
            packet=packet,
            sector=framework.sector if framework is not None else None,
        )
    ]
    missing = [item for item in required if item not in covered]
    return {
        "required": required,
        "covered": covered,
        "missing": missing,
        "coverage_ratio": (len(covered) / len(required)) if required else None,
    }


def _current_events_unavailable_for_ticker(
    *,
    ticker: str,
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
) -> bool:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    saw_relevant_current_events_call = False
    saw_usable_current_events = False
    for call in tool_calls:
        if call.tool_name != "fetch_current_events":
            continue
        if not _tool_call_mentions_ticker(call, ticker):
            continue
        saw_relevant_current_events_call = True
        linked = [evidence_by_id[item] for item in call.evidence_ref_ids if item in evidence_by_id]
        if linked and any(_confidence_at_least(item.confidence, "MODERATE") for item in linked):
            saw_usable_current_events = True
            continue
        preview = str(call.output_preview or "").lower()
        if call.status == "OK" and "usable_for_decision" in preview and "true" in preview:
            saw_usable_current_events = True
    return saw_relevant_current_events_call and not saw_usable_current_events


def _capital_structure_resolution_for_ticker(
    *,
    ticker: str,
    evidence: list[EvidenceReference],
) -> dict[str, Any]:
    priority = {
        "CAPITAL_STRUCTURE_ACTIVE_DISTRESS": 4,
        "CAPITAL_STRUCTURE_RESOLVED_CLEAR": 3,
        "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST": 2,
        "CAPITAL_STRUCTURE_UNRESOLVED": 1,
    }
    selected_status: str | None = None
    selected_summary: str | None = None
    selected_terms: dict[str, Any] | None = None
    selected_maturity_schedule: list[dict[str, Any]] = []
    selected_covenant_status: str | None = None
    selected_covenant_terms: list[dict[str, Any]] = []
    for item in evidence:
        if item.source_label != "analyze_capital_structure_resolution":
            continue
        if not _confidence_at_least(item.confidence, "MODERATE") or not _evidence_mentions_ticker(
            item, ticker
        ):
            continue
        payload = f"{item.summary}\n{item.excerpt}".upper()
        status = next((candidate for candidate in priority if candidate in payload), None)
        if not status:
            continue
        if selected_status is None or priority[status] > priority.get(selected_status, 0):
            selected_status = status
            selected_summary = item.summary
            try:
                parsed = json.loads(item.excerpt) if item.excerpt else {}
            except json.JSONDecodeError:
                parsed = {}
            terms = parsed.get("capital_structure_terms") if isinstance(parsed, dict) else {}
            if isinstance(terms, dict):
                selected_terms = terms
                selected_covenant_status = (
                    str(terms.get("covenant_status")) if terms.get("covenant_status") else None
                )
                selected_maturity_schedule = [
                    dict(row)
                    for row in (terms.get("maturity_schedule") or [])
                    if isinstance(row, dict)
                ]
                selected_covenant_terms = [
                    dict(row)
                    for row in (terms.get("covenant_terms") or [])
                    if isinstance(row, dict)
                ]
            elif isinstance(parsed, dict):
                selected_covenant_status = (
                    str(parsed.get("covenant_status")) if parsed.get("covenant_status") else None
                )
                selected_maturity_schedule = [
                    dict(row)
                    for row in (parsed.get("maturity_schedule") or [])
                    if isinstance(row, dict)
                ]
                selected_covenant_terms = [
                    dict(row)
                    for row in (parsed.get("covenant_terms") or [])
                    if isinstance(row, dict)
                ]
    return {
        "status": selected_status,
        "summary": selected_summary,
        "terms_status": (selected_terms or {}).get("extraction_status"),
        "maturity_schedule_status": (selected_terms or {}).get("maturity_schedule_status"),
        "maturity_schedule": selected_maturity_schedule,
        "covenant_status": selected_covenant_status,
        "covenant_terms_status": (selected_terms or {}).get("covenant_terms_status"),
        "covenant_terms": selected_covenant_terms,
    }


def _evidence_hard_blockers_for_ticker(
    *, ticker: str, evidence: list[EvidenceReference]
) -> list[str]:
    blockers: list[str] = []
    capital_structure_resolution = _capital_structure_resolution_for_ticker(
        ticker=ticker, evidence=evidence
    )
    capital_structure_status = str(capital_structure_resolution.get("status") or "")
    if capital_structure_status == "CAPITAL_STRUCTURE_ACTIVE_DISTRESS":
        return ["FOLLOW_UP_CAPITAL_STRUCTURE_ACTIVE_DISTRESS"]
    suppress_raw_financing_flags = capital_structure_status in {
        "CAPITAL_STRUCTURE_RESOLVED_CLEAR",
        "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST",
    }
    for item in evidence:
        if not _confidence_at_least(item.confidence, "MODERATE") or not _evidence_mentions_ticker(
            item, ticker
        ):
            continue
        payload = f"{item.summary}\n{item.excerpt}".lower()
        if '"solvency_risk": "critical"' in payload or '"risk": "critical"' in payload:
            blockers.append("FOLLOW_UP_SOLVENCY_CRITICAL")
        # The tool payload's bare flag is not enough; it must say a filed, blockable
        # assertion stands behind it (going_concern_asserted, from going_concern_asserted()).
        if (
            not suppress_raw_financing_flags
            and '"going_concern_language": true' in payload
            and '"going_concern_asserted": true' in payload
        ):
            blockers.append("FOLLOW_UP_GOING_CONCERN_LANGUAGE")
        if not suppress_raw_financing_flags and '"no_assurance_financing": true' in payload:
            blockers.append("FOLLOW_UP_NO_ASSURANCE_FINANCING")
    return list(dict.fromkeys(blockers))


def _selection_audit_for_ticker(
    *,
    selected_ticker: str | None,
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    degraded_states: list[str],
    framework: SectorFinancialFramework | None = None,
) -> dict[str, Any]:
    if not selected_ticker:
        return {
            "status": "NOT_APPLICABLE",
            "selected_ticker": None,
            "actionable": False,
            "confidence_ceiling": None,
            "hard_blockers": [],
            "confidence_caps": [],
            "catalyst_signals": [],
            "expected_return_evidence_count": 0,
            "company_specific_evidence_count": 0,
            "base_return_hurdle": BASE_RETURN_HURDLE,
            "selected_return_cushion_hurdle": SELECTED_RETURN_CUSHION_HURDLE,
            "best_base_annualized_return": None,
            "base_return_margin_over_hurdle": None,
            "best_base_horizon_years": None,
            "downside_annualized_return": None,
            "downside_horizon_years": None,
            "downside_evidence_status": "NOT_APPLICABLE",
            "capital_loss_underwriting_status": "NOT_APPLICABLE",
            "capital_loss_impairment_class": None,
            "capital_loss_underwriting_caution": None,
            "capital_loss_reason_codes": [],
            "refinancing_timeline_status": "NOT_APPLICABLE",
            "refinancing_timeline_needs": [],
            "refinancing_timeline_summary": None,
            "debt_due_within_12mo": None,
            "going_concern_language": None,
            "no_assurance_financing": None,
            "cash_runway_quarters": None,
            "capital_structure_terms_status": None,
            "maturity_schedule_status": None,
            "maturity_schedule": [],
            "covenant_status": None,
            "covenant_terms_status": None,
            "covenant_terms": [],
            "selected_company_evidence_tools": [],
            "selected_evidence_pillars": [],
            "selected_risk_evidence_tools": [],
            "framework_required_evidence": [],
            "framework_required_evidence_covered": [],
            "framework_required_evidence_missing": [],
            "framework_required_evidence_coverage_ratio": None,
            "return_cushion_status": "MISSING",
            "notes": ["No selected ticker to audit."],
            **_audit_signal_classification_summary(hard_blockers=[], confidence_caps=[]),
        }

    ticker = selected_ticker.upper()
    packet = packets_by_ticker.get(ticker)
    if packet is None:
        return {
            "status": "BLOCKED",
            "selected_ticker": ticker,
            "actionable": False,
            "confidence_ceiling": None,
            "hard_blockers": ["SELECTED_TICKER_NOT_IN_SCOPE"],
            "confidence_caps": [],
            "catalyst_signals": [],
            "expected_return_evidence_count": 0,
            "company_specific_evidence_count": 0,
            "base_return_hurdle": BASE_RETURN_HURDLE,
            "selected_return_cushion_hurdle": SELECTED_RETURN_CUSHION_HURDLE,
            "best_base_annualized_return": None,
            "base_return_margin_over_hurdle": None,
            "best_base_horizon_years": None,
            "downside_annualized_return": None,
            "downside_horizon_years": None,
            "downside_evidence_status": "MISSING",
            "capital_loss_underwriting_status": "NOT_AVAILABLE",
            "capital_loss_impairment_class": None,
            "capital_loss_underwriting_caution": None,
            "capital_loss_reason_codes": [],
            "refinancing_timeline_status": "NOT_AVAILABLE",
            "refinancing_timeline_needs": [],
            "refinancing_timeline_summary": None,
            "debt_due_within_12mo": None,
            "going_concern_language": None,
            "no_assurance_financing": None,
            "cash_runway_quarters": None,
            "capital_structure_terms_status": None,
            "maturity_schedule_status": None,
            "maturity_schedule": [],
            "covenant_status": None,
            "covenant_terms_status": None,
            "covenant_terms": [],
            "selected_company_evidence_tools": [],
            "selected_evidence_pillars": [],
            "selected_risk_evidence_tools": [],
            "framework_required_evidence": [],
            "framework_required_evidence_covered": [],
            "framework_required_evidence_missing": [],
            "framework_required_evidence_coverage_ratio": None,
            "return_cushion_status": "MISSING",
            "notes": ["Selected ticker is outside the analyzed packet set."],
            **_audit_signal_classification_summary(
                hard_blockers=["SELECTED_TICKER_NOT_IN_SCOPE"],
                confidence_caps=[],
            ),
        }

    hard_blockers = _selection_blockers_for_packet(packet)
    packet_caps = [str(item) for item in packet.confidence_caps if str(item).strip()]
    accounting_quality = (
        packet.accounting_quality if isinstance(packet.accounting_quality, dict) else {}
    )
    filing_status = str(accounting_quality.get("filing_risk_evidence_status") or "").upper()
    data_quality_status = str(packet.data_quality_status or "").upper()
    if data_quality_status in {"NO_FILING", "RISK_SECTION_NOT_FOUND"}:
        hard_blockers.append(data_quality_status)
    if filing_status in {"NO_READABLE_ANNUAL_FILING", "RISK_SECTION_NOT_FOUND"}:
        hard_blockers.append(filing_status)
    for cap in packet_caps:
        if cap in {"FILING_RISK_NO_FILING", "FILING_RISK_SECTION_NOT_FOUND"}:
            hard_blockers.append(cap)

    # The 12% base-return hurdle is operator-tunable. In 'soft' mode a
    # sub-hurdle base return becomes a HURDLE-class confidence cap (routed to
    # WATCHLIST_ONLY below), not a hard blocker; 'hard' preserves the legacy
    # hard-block for rollback. A genuinely MISSING base case is unaffected — it
    # remains a legitimate hard blocker regardless of mode.
    hurdle_mode = (
        str(get_config().base_return_hurdle_mode or BASE_RETURN_HURDLE_MODE).strip().lower()
    )
    base_return_below_hurdle_soft = False
    best_base_return, best_base_horizon = _best_base_return(ticker=ticker, scenarios=scenarios)
    if best_base_return is None:
        hard_blockers.append("MISSING_BASE_RETURN_CASE")
        return_cushion_status = "MISSING"
    elif best_base_return < BASE_RETURN_HURDLE:
        if hurdle_mode == "hard":
            hard_blockers.append("BASE_RETURN_BELOW_12PCT_HURDLE")
        else:
            base_return_below_hurdle_soft = True
        return_cushion_status = "BELOW_HURDLE"
    elif best_base_return < SELECTED_RETURN_CUSHION_HURDLE:
        return_cushion_status = "THIN"
    else:
        return_cushion_status = "CLEAR"
    base_return_margin = (
        best_base_return - BASE_RETURN_HURDLE
        if isinstance(best_base_return, (int, float))
        else None
    )
    downside_return, downside_horizon = _best_downside_return(ticker=ticker, scenarios=scenarios)
    downside_evidence_status = "PRESENT" if downside_return is not None else "MISSING"

    evidence_profile = _selected_evidence_profile(
        ticker=ticker,
        tool_calls=tool_calls,
        evidence=evidence,
    )
    capital_structure_resolution = _capital_structure_resolution_for_ticker(
        ticker=ticker, evidence=evidence
    )
    capital_structure_status = capital_structure_resolution.get("status")
    capital_structure_summary = capital_structure_resolution.get("summary")
    refinancing_timeline = _refinancing_timeline_for_packet(
        packet,
        capital_structure_status=capital_structure_status,
        capital_structure_summary=capital_structure_summary,
    )
    expected_count = int(evidence_profile["expected_return_count"])
    company_count = int(evidence_profile["company_specific_count"])
    selected_company_tools = list(evidence_profile["selected_company_evidence_tools"])
    selected_pillars = list(evidence_profile["selected_evidence_pillars"])
    selected_risk_tools = list(evidence_profile["selected_risk_evidence_tools"])
    framework_evidence = _framework_required_evidence_coverage(
        framework=framework,
        evidence_profile=evidence_profile,
        packet=packet,
    )
    if expected_count < 1:
        hard_blockers.append("MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE")
    if company_count < 1:
        hard_blockers.append("MISSING_COMPANY_SPECIFIC_EVIDENCE")
    hard_blockers.extend(_evidence_hard_blockers_for_ticker(ticker=ticker, evidence=evidence))

    confidence_caps: list[str] = []
    # Surface any CATALYST-class codes carried on the packet (e.g.
    # INSIDER_BUY_CLUSTER, BUYBACK_ACCELERATION). These are an ORTHOGONAL
    # positive timing axis: they ride on confidence_caps purely so they appear
    # in catalyst_signals, and because binding_hurdle_caps /
    # business_quality_confidence_caps filter by their own class, a CATALYST
    # code is excluded from every binding set and can never downgrade a grade.
    confidence_caps.extend(_audit_signals_by_class(packet_caps, "CATALYST"))
    if base_return_below_hurdle_soft:
        confidence_caps.append("BASE_RETURN_BELOW_HURDLE_SOFT")
    if return_cushion_status == "THIN":
        confidence_caps.append("THIN_RETURN_CUSHION")
    if company_count >= 1 and len(selected_company_tools) < 2:
        confidence_caps.append("INSUFFICIENT_COMPANY_SPECIFIC_EVIDENCE")
    if company_count >= 1 and "quality" not in selected_pillars:
        confidence_caps.append("MISSING_KPI_QUALITY_EVIDENCE")
    if company_count >= 1 and len(selected_pillars) < 2:
        confidence_caps.append("INSUFFICIENT_EVIDENCE_PILLAR_COVERAGE")
    if downside_return is None:
        confidence_caps.append("MISSING_DOWNSIDE_RETURN_CASE")
    if (
        isinstance(downside_return, (int, float))
        and downside_return <= DOWNSIDE_RISK_CAP_THRESHOLD
        and not selected_risk_tools
    ):
        confidence_caps.append("DOWNSIDE_ASYMMETRY_UNRESOLVED")
        downside_evidence_status = "UNRESOLVED_ASYMMETRY"
    if (
        "FILING_RISK_STALE_ANNUAL_FILING" in packet_caps
        and not evidence_profile["has_freshness_evidence"]
    ):
        confidence_caps.append("STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE")
    if "HIGH_GROWTH_DEPENDENCY" in packet_caps:
        confidence_caps.append("UNRESOLVED_HIGH_GROWTH_DEPENDENCY")
    if "QUARTERLY_REVENUE_TREND_UNKNOWN" in packet_caps:
        confidence_caps.append("QUARTERLY_REVENUE_TREND_UNKNOWN")
    if str(packet.balance_sheet.get("solvency_risk") or "").upper() == "ELEVATED":
        confidence_caps.append("ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK")
    margin_of_safety_profile = _negative_margin_of_safety_profile(
        ticker=ticker,
        packet=packet,
        evidence=evidence,
    )
    if margin_of_safety_profile.get("status") == "NEGATIVE":
        confidence_caps.append("NEGATIVE_MARGIN_OF_SAFETY")
    valuation_anchor_magnitude = _valuation_anchor_magnitude_profile(packet)
    if valuation_anchor_magnitude.get("status") == "SUSPICIOUS":
        confidence_caps.append(SUSPICIOUS_VALUATION_ANCHOR_CAP)
    # The reverse-DCF expectations gap sits ALONGSIDE the existing
    # MOS/hurdle/solvency gates, never overriding them. An EXPENSIVE bucket
    # (price implies more growth than is supportable) is a binding HURDLE cap
    # that downgrades to WATCHLIST_ONLY; a CHEAP bucket (price implies less
    # growth than is supportable) is purely additive — a positive note (added
    # below, once notes exists), never a blocker; UNRELIABLE/None is silent.
    packet_valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
    expectations_gap_bucket = packet_valuation.get("expectations_gap_bucket")
    if expectations_gap_bucket == "EXPENSIVE_VS_EXPECTATIONS":
        confidence_caps.append("EXPENSIVE_VS_EXPECTATIONS")
    for state in degraded_states:
        state_text = str(state)
        if ticker in state_text.upper() and any(
            term in state_text.lower() for term in ("liquidity", "refinancing")
        ):
            confidence_caps.append("ELEVATED_LIQUIDITY_OR_REFINANCING_RISK")
    if evidence_profile["current_events_low"] or _current_events_unavailable_for_ticker(
        ticker=ticker,
        tool_calls=tool_calls,
        evidence=evidence,
    ):
        confidence_caps.append("CURRENT_EVENTS_UNAVAILABLE")
    if capital_structure_status == "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST":
        confidence_caps.append("CAPITAL_STRUCTURE_WATCHLIST")
    elif capital_structure_status == "CAPITAL_STRUCTURE_UNRESOLVED":
        confidence_caps.append("CAPITAL_STRUCTURE_UNRESOLVED")
    capital_loss = _capital_loss_underwriting_for_packet(
        packet,
        refinancing_timeline_status=refinancing_timeline["status"],
    )
    hard_blockers.extend(capital_loss["hard_blockers"])
    confidence_caps.extend(capital_loss["confidence_caps"])
    hard_blockers.extend(refinancing_timeline["hard_blockers"])
    confidence_caps.extend(refinancing_timeline["confidence_caps"])
    if framework_evidence["missing"]:
        confidence_caps.append("FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE")

    hard_blockers = list(dict.fromkeys(hard_blockers))
    confidence_caps = list(dict.fromkeys(confidence_caps))
    binding_hard_blockers = [
        code for code in hard_blockers if classify_audit_signal(code) != "EVIDENCE_QUALITY"
    ]
    binding_hurdle_caps = _audit_signals_by_class(confidence_caps, "HURDLE")
    business_quality_confidence_caps = _audit_signals_by_class(confidence_caps, "BUSINESS_QUALITY")
    # The positive timing axis. CATALYST codes are reported separately and
    # are NOT members of any binding set above, so they never affect status.
    catalyst_signals = _audit_signals_by_class([*hard_blockers, *confidence_caps], "CATALYST")
    # A data-availability gap is an EVIDENCE_QUALITY code that surfaced as a
    # hard_blocker (e.g. NO_FILING, MISSING_*_EVIDENCE) — the underlying filing
    # or company-specific evidence could not be fetched at all. These were made
    # non-binding by the EVIDENCE_QUALITY reclassification, but a candidate held
    # purely on such a gap must not read as a clean PASS — it is DATA_INCOMPLETE
    # (resolve-then-promote). A thin-coverage EVIDENCE_QUALITY *confidence cap*
    # (full filing present, just sparse follow-up evidence) does NOT trigger the
    # tier; it stays PASS with a resolution flag, preserving prior behavior.
    evidence_quality_hard_blockers = _audit_signals_by_class(hard_blockers, "EVIDENCE_QUALITY")
    needs_evidence_resolution = _audit_signals_by_class(
        [*hard_blockers, *confidence_caps],
        "EVIDENCE_QUALITY",
    )
    status = "PASS"
    confidence_ceiling = "HIGH"
    notes: list[str] = []
    if expectations_gap_bucket == "CHEAP_VS_EXPECTATIONS":
        notes.append(
            "Expectations gap is negative: market implies less growth than is supportable "
            "(cheap vs expectations — candidate signal under measurement, not an established edge)."
        )
    if binding_hard_blockers:
        status = "BLOCKED"
        confidence_ceiling = None
        notes.append("Selection audit found binding hard blockers.")
    elif binding_hurdle_caps:
        status = "WATCHLIST_ONLY"
        confidence_ceiling = "LOW" if len(binding_hurdle_caps) >= 2 else "MODERATE"
        notes.append("Selection audit found binding hurdle caps; candidate is watchlist-only.")
    elif evidence_quality_hard_blockers:
        # No binding hard blockers and no hurdle caps, but a fetchable
        # data-availability gap surfaced as a hard blocker (an EVIDENCE_QUALITY
        # code such as NO_FILING / MISSING_*_EVIDENCE): the filing or the
        # core expected-return / company-specific evidence could not be fetched
        # at all, so the candidate cannot be fully assessed. This is the
        # resolve-then-promote state — not a clean PASS, not a quality
        # reject. It outranks a mere business-quality confidence cap because an
        # unassessable name must never read as an actionable PASS, while real
        # hurdle/solvency HARD blocks (handled above) still dominate it.
        # Confidence stays None pending the fetch that unblocks it.
        status = "DATA_INCOMPLETE"
        confidence_ceiling = None
        notes.append(
            "Selection audit found fetchable data-availability gaps; candidate is data-incomplete pending resolution."
        )
    elif business_quality_confidence_caps:
        # A business-quality cap is a genuine (non-fetchable) risk signal: the
        # candidate stays actionable with a capped confidence ceiling.
        confidence_ceiling = (
            "LOW"
            if len(business_quality_confidence_caps) >= 2
            or SUSPICIOUS_VALUATION_ANCHOR_CAP in business_quality_confidence_caps
            else "MODERATE"
        )
        notes.append(
            "Selection audit passed with business-quality confidence caps; candidate remains actionable."
        )
        if needs_evidence_resolution:
            notes.append(
                "Selection audit passed with evidence-quality signals requiring follow-up resolution."
            )
    else:
        if needs_evidence_resolution:
            notes.append(
                "Selection audit passed with evidence-quality signals requiring follow-up resolution."
            )
        else:
            notes.append("Selection audit passed; candidate remains actionable.")

    return {
        "status": status,
        "selected_ticker": ticker,
        "actionable": status == "PASS",
        "data_incomplete": status == "DATA_INCOMPLETE",
        "data_resolution_needed": list(needs_evidence_resolution),
        "confidence_ceiling": confidence_ceiling,
        "hard_blockers": hard_blockers,
        "confidence_caps": confidence_caps,
        "catalyst_signals": catalyst_signals,
        "expected_return_evidence_count": expected_count,
        "company_specific_evidence_count": company_count,
        "base_return_hurdle": BASE_RETURN_HURDLE,
        "selected_return_cushion_hurdle": SELECTED_RETURN_CUSHION_HURDLE,
        "best_base_annualized_return": best_base_return,
        "base_return_margin_over_hurdle": base_return_margin,
        "best_base_horizon_years": best_base_horizon,
        "downside_annualized_return": downside_return,
        "downside_horizon_years": downside_horizon,
        "downside_evidence_status": downside_evidence_status,
        "capital_loss_underwriting_status": capital_loss["status"],
        "capital_loss_impairment_class": capital_loss["impairment_class_primary"],
        "capital_loss_underwriting_caution": capital_loss["primary_underwriting_caution"],
        "capital_loss_reason_codes": capital_loss["reason_codes"],
        "margin_of_safety_status": margin_of_safety_profile.get("status"),
        "margin_of_safety": margin_of_safety_profile.get("margin_of_safety"),
        "margin_of_safety_source": margin_of_safety_profile.get("source"),
        "valuation_anchor_magnitude_status": valuation_anchor_magnitude.get("status"),
        "valuation_anchor_to_price_ratio": valuation_anchor_magnitude.get(
            "valuation_anchor_to_price_ratio"
        ),
        "valuation_anchor_magnitude_threshold": valuation_anchor_magnitude.get("threshold"),
        "expectations_gap_bucket": expectations_gap_bucket,
        "expectations_gap": packet_valuation.get("expectations_gap"),
        "supportable_growth": packet_valuation.get("supportable_growth"),
        "implied_growth": packet_valuation.get("implied_growth"),
        "refinancing_timeline_status": refinancing_timeline["status"],
        "refinancing_timeline_needs": refinancing_timeline["needs"],
        "refinancing_timeline_summary": refinancing_timeline["summary"],
        "debt_due_within_12mo": refinancing_timeline["debt_due_within_12mo"],
        "going_concern_language": refinancing_timeline["going_concern_language"],
        "no_assurance_financing": refinancing_timeline["no_assurance_financing"],
        "cash_runway_quarters": refinancing_timeline["cash_runway_quarters"],
        "selected_company_evidence_tools": selected_company_tools,
        "selected_evidence_pillars": selected_pillars,
        "selected_risk_evidence_tools": selected_risk_tools,
        "framework_required_evidence": framework_evidence["required"],
        "framework_required_evidence_covered": framework_evidence["covered"],
        "framework_required_evidence_missing": framework_evidence["missing"],
        "framework_required_evidence_coverage_ratio": framework_evidence["coverage_ratio"],
        "capital_structure_resolution_status": capital_structure_status,
        "capital_structure_resolution_summary": capital_structure_summary,
        "capital_structure_terms_status": capital_structure_resolution.get("terms_status"),
        "maturity_schedule_status": capital_structure_resolution.get("maturity_schedule_status"),
        "maturity_schedule": capital_structure_resolution.get("maturity_schedule") or [],
        "covenant_status": capital_structure_resolution.get("covenant_status"),
        "covenant_terms_status": capital_structure_resolution.get("covenant_terms_status"),
        "covenant_terms": capital_structure_resolution.get("covenant_terms") or [],
        "return_cushion_status": return_cushion_status,
        "notes": notes,
        **_audit_signal_classification_summary(
            hard_blockers=hard_blockers, confidence_caps=confidence_caps
        ),
    }


def _audit_gap_is_repairable(selection_audit: dict[str, Any], allowed_tools: list[str]) -> bool:
    # A repairable gap is one where the only obstacle is fetchable missing
    # evidence. Previously the reclassified MISSING_*_EVIDENCE codes surfaced as a
    # BLOCKED status; they now surface as DATA_INCOMPLETE (the same
    # underlying hard_blockers, now non-binding because they classify
    # EVIDENCE_QUALITY). Both states are eligible for repair.
    if selection_audit.get("status") not in {"BLOCKED", "DATA_INCOMPLETE"}:
        return False
    hard_blockers = {str(item) for item in selection_audit.get("hard_blockers") or []}
    if not hard_blockers or not hard_blockers.issubset(AUDIT_GAP_REPAIRABLE_BLOCKERS):
        return False
    return bool(_company_autonomy_allowed_tools(allowed_tools))


def _expected_return_gap_only(selection_audit: dict[str, Any]) -> bool:
    # The cheap in-run repair (rank_expected_return_cases) applies only when the
    # sole hard blocker is the missing expected-return scenario evidence. Since
    # the EVIDENCE_QUALITY reclassification moved
    # MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE into the EVIDENCE_QUALITY class,
    # this check matches the literal hard-blocker list rather than filtering out
    # EVIDENCE_QUALITY codes (which would now drop the very code it targets).
    blockers = [
        str(item) for item in selection_audit.get("hard_blockers") or [] if str(item).strip()
    ]
    return blockers == ["MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE"]


def _scenario_stack_available(
    ticker: str,
    scenarios: list[SectorExpectedReturnScenario],
) -> tuple[bool, int | None]:
    normalized = str(ticker or "").upper()
    scenario_names: set[str] = set()
    best_base_horizon: int | None = None
    best_base_return: float | None = None
    for scenario in scenarios:
        if scenario.ticker.upper() != normalized:
            continue
        scenario_name = scenario.scenario_name.lower()
        if scenario_name in {"base", "downside", "upside"}:
            scenario_names.add(scenario_name)
        if scenario_name == "base" and scenario.annualized_return is not None:
            if best_base_return is None or scenario.annualized_return > best_base_return:
                best_base_return = float(scenario.annualized_return)
                best_base_horizon = int(scenario.horizon_years)
    return {"base", "downside", "upside"}.issubset(
        scenario_names
    ) and best_base_return is not None, best_base_horizon


def _expected_return_repair_targets(
    *,
    relative_ranking: list[dict[str, Any]],
    scenarios: list[SectorExpectedReturnScenario],
    limit: int = AUDIT_GAP_REPAIR_TARGET_LIMIT,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    eligible_verdicts = {"WATCHLIST_ONLY", "ACTIONABLE", "SELECTED"}
    for order, row in enumerate(relative_ranking):
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        verdict = str(row.get("company_autonomy_verdict") or "").upper()
        if verdict not in eligible_verdicts:
            continue
        blockers = {str(item) for item in row.get("hard_blockers") or []}
        if "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" not in blockers:
            continue
        has_stack, horizon = _scenario_stack_available(ticker, scenarios)
        if not has_stack:
            continue
        rank = row.get("rank")
        targets.append(
            {
                "ticker": ticker,
                "rank": int(rank) if isinstance(rank, int) else order + 1,
                "best_base_horizon_years": horizon or row.get("best_base_horizon_years") or 5,
            }
        )
    targets.sort(key=lambda item: int(item["rank"]))
    return targets[:limit]


def _final_decision_prompt_context(
    *,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    relative_ranking: list[dict[str, Any]],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    audit_gap_repair_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_packets, _scope = _sector_prompt_packet_scope(
        company_packets,
        scenarios=scenarios,
    )
    prompt_scoped_tickers = [packet.ticker.upper() for packet in prompt_packets]
    repair_target_tickers = {
        str(item.get("ticker") or "").upper()
        for item in audit_gap_repair_targets
        if isinstance(item, dict) and str(item.get("ticker") or "").strip()
    }
    finalist_tickers: list[str] = []
    for row in relative_ranking:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker or ticker in finalist_tickers:
            continue
        verdict = str(row.get("company_autonomy_verdict") or "").upper()
        audit_status = str(row.get("audit_status") or "").upper()
        if (
            verdict in {"WATCHLIST_ONLY", "ACTIONABLE", "SELECTED"}
            or audit_status in {"PASS", "WATCHLIST_ONLY"}
            or ticker in repair_target_tickers
        ):
            finalist_tickers.append(ticker)

    evidence_by_finalist: dict[str, dict[str, Any]] = {}
    for ticker in finalist_tickers:
        profile = _selected_evidence_profile(
            ticker=ticker, tool_calls=tool_calls, evidence=evidence
        )
        evidence_by_finalist[ticker] = {
            "evidence_count": int(profile["expected_return_count"]),
            "has_repair_target": ticker in repair_target_tickers,
        }

    return {
        "prompt_scoped_tickers": prompt_scoped_tickers,
        "expected_return_evidence_by_finalist": evidence_by_finalist,
        "audit_gap_repair_targets": [dict(item) for item in audit_gap_repair_targets],
    }


def _degraded_state_labels(states: list[str]) -> list[str]:
    labels: list[str] = []
    for state in states:
        label = str(state or "").split(":", 1)[0].strip()
        if not label or label == "STRUCTURED_DECISION_INCOMPLETE":
            continue
        labels.append(label)
    return list(dict.fromkeys(labels))


def _synthesized_no_selection_reason(
    *,
    decision: SectorFinalDecision,
    states: list[str],
) -> str:
    if decision.no_selection_reason:
        return decision.no_selection_reason
    if decision.selection_blockers:
        return "No company selected because selection blockers remained unresolved."
    labels = _degraded_state_labels(states)
    if labels:
        shown = ", ".join(labels[:4])
        suffix = "" if len(labels) <= 4 else f", plus {len(labels) - 4} more"
        return f"No company selected because unresolved evidence gaps remained: {shown}{suffix}."
    return "Autonomous sector analyst returned no selection."


def _scenario_base_return_by_ticker(
    scenarios: list[SectorExpectedReturnScenario],
) -> dict[str, float]:
    returns: dict[str, float] = {}
    for scenario in scenarios:
        if scenario.scenario_name.lower() != "base" or scenario.annualized_return is None:
            continue
        ticker = scenario.ticker.upper()
        prior = returns.get(ticker)
        if prior is None or scenario.annualized_return > prior:
            returns[ticker] = scenario.annualized_return
    return returns


def _no_selection_finalist_focus(
    *,
    decision: SectorFinalDecision,
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    candidate_selection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Choose the most financially relevant finalist to audit after provider no-selection."""

    if not packets_by_ticker:
        return None

    explicit_finalists: list[str] = []
    for item in decision.rejected_finalists:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").upper()
        if ticker and ticker in packets_by_ticker:
            explicit_finalists.append(ticker)

    ordered_candidates: list[tuple[str, str, int]] = []
    for order, ticker in enumerate(explicit_finalists):
        ordered_candidates.append((ticker, "provider_rejected_finalist", order))

    selected_tickers = []
    if isinstance(candidate_selection, dict):
        selected_tickers = [
            str(item).upper()
            for item in candidate_selection.get("selected_tickers") or []
            if str(item).strip()
        ]
    for order, ticker in enumerate(selected_tickers, start=len(ordered_candidates)):
        if ticker in packets_by_ticker:
            ordered_candidates.append((ticker, "candidate_selection", order))

    for order, ticker in enumerate(packets_by_ticker, start=len(ordered_candidates)):
        ordered_candidates.append((ticker, "packet_order", order))

    deduped: dict[str, tuple[str, int]] = {}
    for ticker, source, order in ordered_candidates:
        if ticker not in deduped:
            deduped[ticker] = (source, order)
    if not deduped:
        return None

    base_returns = _scenario_base_return_by_ticker(scenarios)

    def sort_key(item: tuple[str, tuple[str, int]]) -> tuple[float, int, int]:
        ticker, (_source, order) = item
        packet_blocked = 1 if _selection_blockers_for_packet(packets_by_ticker[ticker]) else 0
        base_return = base_returns.get(ticker, -999.0)
        return (base_return, -packet_blocked, -order)

    focus_ticker, (source, _order) = max(deduped.items(), key=sort_key)
    reason = "highest available base-case expected return among audited no-selection finalists"
    if source == "provider_rejected_finalist":
        reason = "provider finalist with the strongest available base-case expected return"
    elif source == "candidate_selection":
        reason = (
            "highest available base-case expected return from deterministic candidate selection"
        )

    return {
        "ticker": focus_ticker,
        "source": source,
        "base_annualized_return": base_returns.get(focus_ticker),
        "reason": reason,
    }


def _no_selection_reason_from_finalist_audit(
    *,
    focus_ticker: str,
    selection_audit: dict[str, Any],
) -> str:
    status = str(selection_audit.get("status") or "UNKNOWN")
    hard_blockers = [
        str(item) for item in selection_audit.get("hard_blockers") or [] if str(item).strip()
    ]
    confidence_caps = [
        str(item) for item in selection_audit.get("confidence_caps") or [] if str(item).strip()
    ]
    if status == "BLOCKED":
        blockers = ", ".join(hard_blockers) if hard_blockers else "unresolved hard blockers"
        return f"No company selected because finalist {focus_ticker} failed selection audit: {blockers}."
    if status == "WATCHLIST_ONLY":
        caps = ", ".join(confidence_caps) if confidence_caps else "unresolved evidence caps"
        return (
            f"No company selected because finalist {focus_ticker} remained watchlist-only: {caps}."
        )
    return f"No company selected after auditing finalist {focus_ticker}."


def _pct_text(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value * 100:.1f}%"
    return "N/A"


def _deterministic_no_selection_thesis(
    *,
    focus_ticker: str,
    selection_audit: dict[str, Any],
) -> str:
    hard_blockers = [
        str(item) for item in selection_audit.get("hard_blockers") or [] if str(item).strip()
    ]
    base_return = selection_audit.get("best_base_annualized_return")
    hurdle = selection_audit.get("base_return_hurdle")
    margin = selection_audit.get("base_return_margin_over_hurdle")
    horizon = selection_audit.get("best_base_horizon_years")
    below_hurdle_blocked = "BASE_RETURN_BELOW_12PCT_HURDLE" in hard_blockers
    if (
        below_hurdle_blocked
        and isinstance(base_return, (int, float))
        and isinstance(hurdle, (int, float))
    ):
        horizon_text = f" over {int(horizon)}Y" if isinstance(horizon, int) else ""
        if isinstance(margin, (int, float)) and margin < 0:
            gap_text = f", {abs(margin) * 100:.1f} percentage points below"
        elif isinstance(margin, (int, float)):
            gap_text = f", {margin * 100:.1f} percentage points above"
        else:
            gap_text = " versus"
        return (
            f"Deterministic audit made no selection because finalist {focus_ticker}'s "
            f"best base-case return was {_pct_text(base_return)} annualized{horizon_text}"
            f"{gap_text} the {_pct_text(hurdle)} base-return hurdle."
        )
    if hard_blockers:
        return (
            f"Deterministic audit made no selection because finalist {focus_ticker} "
            f"had binding blockers: {', '.join(hard_blockers)}."
        )
    return f"Deterministic audit made no selection after reviewing finalist {focus_ticker}."


def _deterministic_no_selection_key_risk(selection_audit: dict[str, Any]) -> str:
    hard_blockers = [
        str(item) for item in selection_audit.get("hard_blockers") or [] if str(item).strip()
    ]
    if hard_blockers:
        return f"Binding audit blockers: {', '.join(hard_blockers)}."
    confidence_caps = [
        str(item) for item in selection_audit.get("confidence_caps") or [] if str(item).strip()
    ]
    if confidence_caps:
        return f"Audit confidence caps: {', '.join(confidence_caps)}."
    return "The deterministic selection audit did not clear all guardrails."


def _deterministic_no_selection_downside(selection_audit: dict[str, Any]) -> str:
    downside = selection_audit.get("downside_annualized_return")
    status = str(selection_audit.get("downside_evidence_status") or "UNKNOWN")
    if isinstance(downside, (int, float)) and not isinstance(downside, bool):
        return f"Audited downside case was {_pct_text(downside)} annualized; downside evidence status: {status}."
    return f"Audited downside evidence status: {status}."


def _decision_from_finalist_audit_pass(
    *,
    original_decision: SectorFinalDecision,
    focus_ticker: str,
    selection_audit: dict[str, Any],
) -> SectorFinalDecision:
    base_return = selection_audit.get("best_base_annualized_return")
    if isinstance(base_return, (int, float)):
        expected_range = f"Audited base case: {base_return * 100:.1f}% annualized"
    else:
        expected_range = original_decision.expected_annualized_return_range
    return SectorFinalDecision(
        verdict="SELECTED",
        confidence=_cap_confidence(
            "MODERATE", str(selection_audit.get("confidence_ceiling") or "MODERATE")
        ),
        selected_ticker=focus_ticker,
        expected_annualized_return_range=expected_range,
        thesis=(
            original_decision.thesis
            or f"{focus_ticker} passed the deterministic no-selection finalist audit."
        ),
        key_risk=(
            original_decision.key_risk
            or "The selection depends on deterministic audit evidence remaining current and decision-usable."
        ),
        downside_case=(
            original_decision.downside_case
            or "Downside remains governed by the audited downside scenario and evidence quality caps."
        ),
        no_selection_reason=None,
        falsifiers=list(original_decision.falsifiers),
        why_selected_over_finalists=list(
            dict.fromkeys(
                [
                    *original_decision.why_selected_over_finalists,
                    "Runtime no-selection finalist audit found the candidate passed all binding selection guardrails.",
                ]
            )
        ),
        rejected_finalists=[dict(item) for item in original_decision.rejected_finalists],
        selection_blockers=[],
        confidence_cap_reasons=list(original_decision.confidence_cap_reasons),
        evidence_ref_ids=list(original_decision.evidence_ref_ids),
    )


_VERDICT_ACTIONABILITY_ORDER = {
    "AVOID": 0,
    "REJECTED": 0,
    "NO_WINNER": 1,
    "NO_SELECTION": 1,
    "WATCHLIST": 2,
    "WATCHLIST_ONLY": 2,
    "DATA_INCOMPLETE": 2,
    "ACTIONABLE": 3,
    "SELECTED": 3,
}


def _sector_verdict_from_audit_status(selection_audit: dict[str, Any]) -> str:
    status = str(selection_audit.get("status") or "").upper()
    if status == "PASS":
        return "SELECTED"
    if status == "WATCHLIST_ONLY":
        return "WATCHLIST"
    if status == "DATA_INCOMPLETE":
        return "DATA_INCOMPLETE"
    return "NO_SELECTION"


def _sector_verdict_from_actionability_level(level: int) -> str:
    if level >= _VERDICT_ACTIONABILITY_ORDER["SELECTED"]:
        return "SELECTED"
    if level >= _VERDICT_ACTIONABILITY_ORDER["WATCHLIST"]:
        return "WATCHLIST"
    return "NO_SELECTION"


def _llm_verdict_ceiling_payload(
    *,
    llm_verdict: str | None,
    audit_verdict: str,
    final_verdict: str,
    bound: bool,
) -> dict[str, Any]:
    return {
        "llm_verdict": llm_verdict,
        "audit_verdict": audit_verdict,
        "final_verdict": final_verdict,
        "bound": bound,
    }


def _apply_llm_verdict_ceiling(
    *,
    decision: SectorFinalDecision,
    selection_audit: dict[str, Any],
    llm_verdict: str | None,
) -> tuple[SectorFinalDecision, dict[str, Any], bool]:
    normalized_llm_verdict = str(llm_verdict or "").upper() or None
    audit_verdict = _sector_verdict_from_audit_status(selection_audit)
    current_verdict = str(decision.verdict or audit_verdict or "NO_SELECTION").upper()
    current_level = _VERDICT_ACTIONABILITY_ORDER.get(current_verdict)
    llm_level = _VERDICT_ACTIONABILITY_ORDER.get(str(normalized_llm_verdict or ""))
    if current_level is None or llm_level is None:
        final_verdict = current_verdict if current_level is not None else audit_verdict
        return (
            decision,
            {
                **selection_audit,
                "llm_verdict_ceiling_applied": _llm_verdict_ceiling_payload(
                    llm_verdict=normalized_llm_verdict,
                    audit_verdict=audit_verdict,
                    final_verdict=final_verdict,
                    bound=False,
                ),
            },
            False,
        )
    final_level = min(current_level, llm_level)
    final_verdict = _sector_verdict_from_actionability_level(final_level)
    bound = final_level < current_level
    if not bound:
        return (
            decision,
            {
                **selection_audit,
                "llm_verdict_ceiling_applied": _llm_verdict_ceiling_payload(
                    llm_verdict=normalized_llm_verdict,
                    audit_verdict=audit_verdict,
                    final_verdict=current_verdict,
                    bound=False,
                ),
            },
            False,
        )
    if final_verdict == "WATCHLIST":
        capped_decision = SectorFinalDecision(
            verdict="WATCHLIST",
            confidence=_cap_confidence(decision.confidence, "MODERATE"),
            selected_ticker=decision.selected_ticker,
            expected_annualized_return_range=decision.expected_annualized_return_range,
            thesis=decision.thesis,
            key_risk=decision.key_risk,
            downside_case=decision.downside_case,
            no_selection_reason=None,
            falsifiers=list(decision.falsifiers),
            why_selected_over_finalists=list(decision.why_selected_over_finalists),
            rejected_finalists=[dict(item) for item in decision.rejected_finalists],
            selection_blockers=[],
            confidence_cap_reasons=list(decision.confidence_cap_reasons),
            evidence_ref_ids=list(decision.evidence_ref_ids),
        )
    else:
        capped_decision = SectorFinalDecision(
            verdict="NO_SELECTION",
            confidence=None,
            selected_ticker=None,
            expected_annualized_return_range=None,
            thesis=decision.thesis,
            key_risk=decision.key_risk,
            downside_case=decision.downside_case,
            no_selection_reason=(
                f"No company selected because the company-level LLM verdict was {normalized_llm_verdict}."
            ),
            falsifiers=list(decision.falsifiers),
            why_selected_over_finalists=list(decision.why_selected_over_finalists),
            rejected_finalists=[dict(item) for item in decision.rejected_finalists],
            selection_blockers=["LLM_VERDICT_CEILING", str(normalized_llm_verdict)],
            confidence_cap_reasons=list(decision.confidence_cap_reasons),
            evidence_ref_ids=list(decision.evidence_ref_ids),
        )
    return (
        capped_decision,
        {
            **selection_audit,
            "status": "WATCHLIST_ONLY" if final_verdict == "WATCHLIST" else "BLOCKED",
            "actionable": final_verdict == "SELECTED",
            "final_verdict_after_audit": final_verdict,
            "llm_verdict_ceiling_applied": _llm_verdict_ceiling_payload(
                llm_verdict=normalized_llm_verdict,
                audit_verdict=audit_verdict,
                final_verdict=final_verdict,
                bound=True,
            ),
        },
        True,
    )


def _packet_has_price_and_valuation(packet: SectorCompanyFinancialPacket) -> bool:
    blockers = set(_selection_blockers_for_packet(packet))
    return not {"MISSING_PRICE", "MISSING_VALUATION"} & blockers


def _company_autonomy_run_by_ticker(
    company_autonomy_runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for item in company_autonomy_runs:
        ticker = str(item.get("ticker") or item.get("selected_ticker") or "").upper()
        if ticker:
            mapped[ticker] = item
    return mapped


def _cross_sectional_factor_vectors(
    company_packets: list[SectorCompanyFinancialPacket],
) -> list[FactorVector]:
    """Extract the value/quality/gap factor inputs for the cross-sectional ranker.

    - value   = ``valuation.discount_to_anchor`` (higher-is-better)
    - quality = ``returns_on_capital.roic`` -> fallback ``roic_wacc_spread`` ->
                fallback ``cash_conversion.fcf_margin`` (higher-is-better)
    - gap     = RAW ``valuation.implied_growth`` (the top-level scalar surfaced on
                the packet). It is lower-is-better; the pure ranker NEGATES it
                internally, so it is passed RAW here (NOT pre-negated, NOT a dict).
    """
    vectors: list[FactorVector] = []
    for packet in company_packets:
        valuation = packet.valuation or {}
        returns_on_capital = packet.returns_on_capital or {}
        cash_conversion = packet.cash_conversion or {}

        value = valuation.get("discount_to_anchor")
        quality = returns_on_capital.get("roic")
        if quality is None:
            quality = returns_on_capital.get("roic_wacc_spread")
        if quality is None:
            quality = cash_conversion.get("fcf_margin")
        gap = valuation.get("implied_growth")

        vectors.append(
            FactorVector(
                ticker=packet.ticker.upper(),
                value=value,
                quality=quality,
                gap=gap,
            )
        )
    return vectors


def _relative_ranking(
    *,
    company_packets: list[SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    degraded_states: list[str],
    company_autonomy_runs: list[dict[str, Any]],
    framework: SectorFinancialFramework | None = None,
) -> list[dict[str, Any]]:
    packets_by_ticker = {packet.ticker.upper(): packet for packet in company_packets}
    child_runs = _company_autonomy_run_by_ticker(company_autonomy_runs)
    cross_sectional_by_ticker = {
        result.ticker: result
        for result in rank_cross_sectional(
            _cross_sectional_factor_vectors(company_packets),
            weights=CROSS_SECTIONAL_FACTOR_WEIGHTS,
            buy_quantile=CROSS_SECTIONAL_BUY_QUANTILE,
        )
    }
    rows: list[dict[str, Any]] = []
    for original_order, packet in enumerate(company_packets):
        ticker = packet.ticker.upper()
        best_base, best_base_horizon = _best_base_return(ticker=ticker, scenarios=scenarios)
        downside, downside_horizon = _best_downside_return(ticker=ticker, scenarios=scenarios)
        downside_evidence_status = "PRESENT" if downside is not None else "MISSING"
        audit = _selection_audit_for_ticker(
            selected_ticker=ticker,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            tool_calls=tool_calls,
            evidence=evidence,
            degraded_states=degraded_states,
            framework=framework,
        )
        child = child_runs.get(ticker, {})
        status = str(audit.get("status") or "UNKNOWN")
        child_verdict = child.get("final_verdict")
        runtime_verdict = child_verdict
        row_actionable = bool(audit.get("actionable"))
        row_audit_status = status
        if status == "PASS":
            child_level = _VERDICT_ACTIONABILITY_ORDER.get(str(child_verdict or "").upper())
            if child_level is not None and child_level < _VERDICT_ACTIONABILITY_ORDER["ACTIONABLE"]:
                row_actionable = False
                row_audit_status = (
                    "WATCHLIST_ONLY"
                    if child_level >= _VERDICT_ACTIONABILITY_ORDER["WATCHLIST"]
                    else "BLOCKED"
                )
                if child_level >= _VERDICT_ACTIONABILITY_ORDER["WATCHLIST"]:
                    positioning = (
                        "Best-positioned but capped by company-level LLM verdict at WATCHLIST_ONLY."
                    )
                    runtime_verdict = "WATCHLIST_ONLY"
                else:
                    positioning = (
                        "Not actionable because the company-level LLM verdict rejected selection."
                    )
                    runtime_verdict = "AVOID"
            else:
                positioning = "Best-positioned and actionable under current guardrails."
                runtime_verdict = "ACTIONABLE"
        elif status == "WATCHLIST_ONLY":
            positioning = "Best-positioned but not actionable until evidence caps clear."
            runtime_verdict = "WATCHLIST_ONLY"
            row_actionable = False
        else:
            blockers = (
                ", ".join(str(item) for item in audit.get("hard_blockers") or [])
                or "unresolved blockers"
            )
            positioning = f"Not actionable under current guardrails: {blockers}."
            row_actionable = False
            if child_verdict in {"ACTIONABLE", "SELECTED"}:
                runtime_verdict = "AVOID"
        cross_sectional = cross_sectional_by_ticker.get(ticker)
        # Apply the composite>0 floor at THIS integration layer: the pure
        # ranker intentionally omits it. A top-quantile row whose
        # composite is at or below the sector mean (composite <= 0) is the
        # "best house in a bad neighborhood" — it must NOT be a buy_candidate.
        cs_composite = cross_sectional.composite if cross_sectional else None
        cs_buy_candidate = bool(cross_sectional.buy_candidate) if cross_sectional else False
        cs_buy_reason = (
            cross_sectional.buy_candidate_reason if cross_sectional else "NO_RANKABLE_FACTORS"
        )
        if cs_buy_candidate and not (cs_composite is not None and cs_composite > 0):
            cs_buy_candidate = False
            cs_buy_reason = "BELOW_SECTOR_MEAN"
        rows.append(
            {
                "ticker": ticker,
                "rank": 0,
                "best_base_annualized_return": best_base,
                "best_base_horizon_years": best_base_horizon,
                "downside_annualized_return": downside,
                "downside_horizon_years": downside_horizon,
                "downside_evidence_status": downside_evidence_status,
                "capital_loss_underwriting_status": audit.get("capital_loss_underwriting_status"),
                "capital_loss_impairment_class": audit.get("capital_loss_impairment_class"),
                "refinancing_timeline_status": audit.get("refinancing_timeline_status"),
                "audit_status": row_audit_status,
                "actionable": row_actionable,
                "hard_blockers": list(audit.get("hard_blockers") or []),
                "confidence_caps": list(audit.get("confidence_caps") or []),
                "company_autonomy_status": child.get("status"),
                "company_autonomy_verdict": runtime_verdict,
                "child_company_autonomy_verdict": child_verdict,
                "company_autonomy_confidence": child.get("confidence"),
                "positioning_summary": positioning,
                "cross_sectional_score": cs_composite,
                "cross_sectional_rank": cross_sectional.rank if cross_sectional else None,
                "cross_sectional_percentile": cross_sectional.percentile
                if cross_sectional
                else None,
                "factor_zscores": dict(cross_sectional.factor_zscores) if cross_sectional else {},
                "buy_candidate": cs_buy_candidate,
                "buy_candidate_reason": cs_buy_reason,
                "_original_order": original_order,
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[int, float, float, int]:
        composite = row.get("cross_sectional_score")
        has_composite = isinstance(composite, (int, float)) and not isinstance(composite, bool)
        base = row.get("best_base_annualized_return")
        has_base = isinstance(base, (int, float)) and not isinstance(base, bool)
        return (
            1 if has_composite else 0,
            float(composite) if has_composite else 0.0,
            float(base) if has_base else -999.0,
            -int(row.get("_original_order", 0)),
        )

    rows = sorted(rows, key=sort_key, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row.pop("_original_order", None)
    return rows


def _company_autonomy_child_verdict_by_ticker(
    company_autonomy_runs: list[dict[str, Any]],
) -> dict[str, str | None]:
    verdicts: dict[str, str | None] = {}
    for run in company_autonomy_runs:
        ticker = str(run.get("ticker") or run.get("selected_ticker") or "").upper()
        if ticker:
            verdicts[ticker] = run.get("final_verdict")
    return verdicts


def _company_autonomy_ticker_impact(
    *,
    ticker: str,
    child_run: dict[str, Any],
    before_audit: dict[str, Any],
    after_audit: dict[str, Any],
    selected_ticker: str | None,
) -> str:
    child_status = str(child_run.get("status") or "").upper()
    child_verdict = str(child_run.get("final_verdict") or "").upper()
    before_actionable = bool(before_audit.get("actionable"))
    after_actionable = bool(after_audit.get("actionable"))
    before_status = str(before_audit.get("status") or "")
    after_status = str(after_audit.get("status") or "")
    if child_run and child_status != "COMPLETED":
        return "CHILD_FAILED" if child_status == "FAILED" else "CHILD_NOT_COMPLETED"
    if selected_ticker == ticker and child_verdict in {
        "NO_WINNER",
        "NO_SELECTION",
        "AVOID",
        "REJECTED",
        "WATCHLIST",
        "WATCHLIST_ONLY",
    }:
        return "CONTRADICTED_SELECTED"
    if not before_actionable and after_actionable:
        return "CHANGED_TO_ACTIONABLE"
    if before_actionable and not after_actionable:
        return "CHANGED_TO_NON_ACTIONABLE"
    if (
        selected_ticker == ticker
        and after_actionable
        and child_verdict in {"ACTIONABLE", "SELECTED"}
    ):
        return "VALIDATED_SELECTED"
    if before_status != after_status:
        return "AUDIT_STATUS_CHANGED"
    return "NO_DECISION_CHANGE" if child_run else "NOT_REVIEWED"


def _company_autonomy_decision_trace(
    *,
    decision: SectorFinalDecision,
    selection_audit: dict[str, Any],
    relative_ranking: list[dict[str, Any]],
    company_autonomy_attempted: bool,
    company_autonomy_status: str | None,
    company_autonomy_runs: list[dict[str, Any]],
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    degraded_states: list[str],
    framework: SectorFinancialFramework | None,
) -> dict[str, Any]:
    status = str(
        company_autonomy_status or ("UNKNOWN" if company_autonomy_attempted else "NOT_ATTEMPTED")
    )
    if not company_autonomy_attempted and not company_autonomy_runs:
        return {
            "status": "NOT_ATTEMPTED",
            "changed_sector_finalist_decision": False,
            "validated_sector_finalist_decision": False,
            "contradicted_sector_finalist_decision": False,
            "selected_ticker_child_verdict": None,
            "top_ranked_child_verdict": None,
            "impact_counts": {},
            "ticker_impacts": [],
        }

    selected_ticker = str(decision.selected_ticker or "").upper() or None
    top_ranked_ticker = (
        str(relative_ranking[0].get("ticker") or "").upper() if relative_ranking else None
    )
    child_runs = _company_autonomy_run_by_ticker(company_autonomy_runs)
    child_verdicts = _company_autonomy_child_verdict_by_ticker(company_autonomy_runs)
    evidence_without_child = [
        item for item in evidence if str(item.source_type or "") != "company_autonomous_run"
    ]
    ticker_order: list[str] = []

    def add_ticker(value: Any) -> None:
        ticker = str(value or "").upper()
        if ticker and ticker not in ticker_order:
            ticker_order.append(ticker)

    for row in relative_ranking:
        add_ticker(row.get("ticker"))
    for run in company_autonomy_runs:
        add_ticker(run.get("ticker") or run.get("selected_ticker"))
    add_ticker(selected_ticker)
    add_ticker(top_ranked_ticker)

    impacts: list[dict[str, Any]] = []
    impact_counts: Counter[str] = Counter()
    for ticker in ticker_order:
        before_audit = _selection_audit_for_ticker(
            selected_ticker=ticker,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            tool_calls=tool_calls,
            evidence=evidence_without_child,
            degraded_states=degraded_states,
            framework=framework,
        )
        after_audit = (
            selection_audit
            if selected_ticker == ticker
            and str(selection_audit.get("selected_ticker") or "").upper() == ticker
            else _selection_audit_for_ticker(
                selected_ticker=ticker,
                packets_by_ticker=packets_by_ticker,
                scenarios=scenarios,
                tool_calls=tool_calls,
                evidence=evidence,
                degraded_states=degraded_states,
                framework=framework,
            )
        )
        child_run = child_runs.get(ticker, {})
        impact = _company_autonomy_ticker_impact(
            ticker=ticker,
            child_run=child_run,
            before_audit=before_audit,
            after_audit=after_audit,
            selected_ticker=selected_ticker,
        )
        impact_counts.update([impact])
        impacts.append(
            {
                "ticker": ticker,
                "child_status": child_run.get("status"),
                "child_verdict": child_run.get("final_verdict"),
                "audit_status_before": before_audit.get("status"),
                "audit_status_after": after_audit.get("status"),
                "actionable_before": bool(before_audit.get("actionable")),
                "actionable_after": bool(after_audit.get("actionable")),
                "hard_blockers_removed": sorted(
                    set(str(item) for item in before_audit.get("hard_blockers") or [])
                    - set(str(item) for item in after_audit.get("hard_blockers") or [])
                ),
                "confidence_caps_removed": sorted(
                    set(str(item) for item in before_audit.get("confidence_caps") or [])
                    - set(str(item) for item in after_audit.get("confidence_caps") or [])
                ),
                "impact": impact,
            }
        )

    return {
        "status": status,
        "changed_sector_finalist_decision": any(
            row.get("impact")
            in {"CHANGED_TO_ACTIONABLE", "CHANGED_TO_NON_ACTIONABLE", "AUDIT_STATUS_CHANGED"}
            for row in impacts
        ),
        "validated_sector_finalist_decision": any(
            row.get("impact") == "VALIDATED_SELECTED" for row in impacts
        ),
        "contradicted_sector_finalist_decision": any(
            row.get("impact") == "CONTRADICTED_SELECTED" for row in impacts
        ),
        "selected_ticker_child_verdict": child_verdicts.get(selected_ticker or ""),
        "top_ranked_child_verdict": child_verdicts.get(top_ranked_ticker or ""),
        "impact_counts": dict(sorted(impact_counts.items())),
        "ticker_impacts": impacts,
    }


def _alternate_finalist_candidates(
    *,
    decision: SectorFinalDecision,
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    already_audited: set[str],
    limit: int = ALTERNATE_FINALIST_AUDIT_LIMIT,
) -> list[dict[str, Any]]:
    """Return bounded alternate finalists after the primary no-selection finalist is blocked."""

    base_returns = _scenario_base_return_by_ticker(scenarios)
    seen = {str(item).upper() for item in already_audited if str(item).strip()}
    candidates: list[dict[str, Any]] = []

    def add_candidate(ticker: str, source: str) -> None:
        normalized = str(ticker or "").upper()
        if not normalized or normalized in seen:
            return
        packet = packets_by_ticker.get(normalized)
        if packet is None or not _packet_has_price_and_valuation(packet):
            return
        seen.add(normalized)
        candidates.append(
            {
                "ticker": normalized,
                "source": source,
                "best_base_annualized_return": base_returns.get(normalized),
            }
        )

    for item in decision.rejected_finalists:
        if isinstance(item, dict):
            add_candidate(str(item.get("ticker") or ""), "provider_rejected_finalist")

    for ticker, _base_return in sorted(base_returns.items(), key=lambda item: (-item[1], item[0])):
        add_candidate(ticker, "scenario_ranked")

    return candidates[:limit]


def _compact_alternate_audit_result(
    *,
    candidate: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ticker": candidate.get("ticker"),
        "source": candidate.get("source"),
        "audit_status": audit.get("status"),
        "hard_blockers": [str(item) for item in audit.get("hard_blockers") or []],
        "confidence_caps": [str(item) for item in audit.get("confidence_caps") or []],
        "best_base_annualized_return": audit.get(
            "best_base_annualized_return",
            candidate.get("best_base_annualized_return"),
        ),
    }


def _run_alternate_finalist_audit(
    *,
    decision: SectorFinalDecision,
    primary_focus_ticker: str,
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    degraded_states: list[str],
    framework: SectorFinancialFramework | None = None,
) -> tuple[str, list[str], list[dict[str, Any]], dict[str, Any] | None]:
    candidates = _alternate_finalist_candidates(
        decision=decision,
        packets_by_ticker=packets_by_ticker,
        scenarios=scenarios,
        already_audited={primary_focus_ticker},
    )
    if not candidates:
        return (
            "NO_ALTERNATES",
            ["No alternate finalists with price and valuation were available to audit."],
            [],
            None,
        )

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        ticker = str(candidate["ticker"]).upper()
        audit = _selection_audit_for_ticker(
            selected_ticker=ticker,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            tool_calls=tool_calls,
            evidence=evidence,
            degraded_states=degraded_states,
            framework=framework,
        )
        result = _compact_alternate_audit_result(candidate=candidate, audit=audit)
        results.append(result)
        if audit.get("status") == "PASS":
            return (
                "RESOLVED_SELECTED",
                [f"Alternate finalist {ticker} passed deterministic selection audit."],
                results,
                audit,
            )

    return (
        "NO_ALTERNATE_PASSED",
        ["No alternate finalist cleared the deterministic selection audit."],
        results,
        None,
    )


def _target_tickers(
    planned: dict[str, Any], packets_by_ticker: dict[str, SectorCompanyFinancialPacket]
) -> list[str]:
    explicit = planned.get("ticker")
    if explicit:
        return [str(explicit).upper()]
    from_input = planned.get("tool_input") if isinstance(planned.get("tool_input"), dict) else {}
    raw_tickers = from_input.get("tickers")
    if isinstance(raw_tickers, list) and raw_tickers:
        return [str(item).upper() for item in raw_tickers if str(item).strip()]
    target_tickers = planned.get("target_tickers")
    if isinstance(target_tickers, list) and target_tickers:
        return [str(item).upper() for item in target_tickers if str(item).strip()]
    return list(packets_by_ticker.keys())


def _planned_call_key(
    planned: dict[str, Any], packets_by_ticker: dict[str, SectorCompanyFinancialPacket]
) -> str:
    """Stable identity for a planned deterministic tool call across LLM turns."""

    raw_input = planned.get("tool_input") if isinstance(planned.get("tool_input"), dict) else {}
    normalized_input = {str(key): value for key, value in raw_input.items() if key != "tickers"}
    return json.dumps(
        {
            "tool_name": str(planned.get("tool_name") or "").strip(),
            "target_tickers": _target_tickers(planned, packets_by_ticker),
            "tool_input": _jsonable(normalized_input),
        },
        sort_keys=True,
        default=str,
    )


def _filter_scenarios(
    scenarios: list[SectorExpectedReturnScenario],
    *,
    tickers: list[str],
    tool_input: dict[str, Any],
) -> list[SectorExpectedReturnScenario]:
    ticker_set = set(tickers)
    horizon = tool_input.get("horizon_years")
    scenario_name = str(tool_input.get("scenario_name") or "").lower()
    filtered: list[SectorExpectedReturnScenario] = []
    for scenario in scenarios:
        if scenario.ticker not in ticker_set:
            continue
        if horizon is not None and scenario.horizon_years != int(horizon):
            continue
        if scenario_name and scenario.scenario_name.lower() != scenario_name:
            continue
        filtered.append(scenario)
    return filtered


def _dispatch_sector_tool(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tickers: list[str],
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
) -> dict[str, Any]:
    if tool_name == "summarize_financial_packets":
        packets = [packets_by_ticker[ticker] for ticker in tickers if ticker in packets_by_ticker]
        covered_tickers = [packet.ticker for packet in packets]
        return {
            "status": "ok",
            "tool": tool_name,
            "tickers": covered_tickers,
            "covered_tickers": covered_tickers,
            "usable_for_decision": bool(packets),
            "evidence_status": "FINANCIAL_PACKETS_AVAILABLE" if packets else "NO_FINANCIAL_PACKETS",
            "summary": (
                f"Summarized financial packets for {len(covered_tickers)} ticker(s): "
                f"{', '.join(covered_tickers) if covered_tickers else 'none'}."
            ),
            "packets": [_compact_packet(packet) for packet in packets],
        }
    if tool_name == "compare_expected_return_scenarios":
        filtered = _filter_scenarios(scenarios, tickers=tickers, tool_input=tool_input)
        covered_tickers = sorted({scenario.ticker for scenario in filtered})
        usable = bool(filtered)
        return {
            "status": "ok" if usable else "unavailable",
            "tool": tool_name,
            "tickers": tickers,
            "covered_tickers": covered_tickers,
            "scenario_count": len(filtered),
            "usable_for_decision": usable,
            "evidence_status": "EXPECTED_RETURN_SCENARIOS_AVAILABLE"
            if usable
            else "NO_EXPECTED_RETURN_SCENARIOS",
            "summary": (
                f"Compared {len(filtered)} expected-return scenario(s) for "
                f"{len(covered_tickers)} ticker(s): {', '.join(covered_tickers) if covered_tickers else 'none'}."
            ),
            "scenarios": [scenario.to_dict() for scenario in filtered],
        }
    if tool_name == "rank_expected_return_cases":
        horizon = int(tool_input.get("horizon_years") or 5)
        scenario_name = str(tool_input.get("scenario_name") or "base").lower()
        filtered = _filter_scenarios(
            scenarios,
            tickers=tickers,
            tool_input={"horizon_years": horizon, "scenario_name": scenario_name},
        )
        ranked = sorted(
            filtered,
            key=lambda item: (
                item.annualized_return if item.annualized_return is not None else -999.0
            ),
            reverse=True,
        )
        ranked_tickers = list(dict.fromkeys(scenario.ticker for scenario in ranked))
        usable = bool(ranked)
        return {
            "status": "ok" if usable else "unavailable",
            "tool": tool_name,
            "tickers": tickers,
            "covered_tickers": ranked_tickers,
            "ranked_tickers": ranked_tickers,
            "top_ticker": ranked_tickers[0] if ranked_tickers else None,
            "scenario_count": len(ranked),
            "horizon_years": horizon,
            "scenario_name": scenario_name,
            "usable_for_decision": usable,
            "evidence_status": "EXPECTED_RETURN_RANKING_AVAILABLE"
            if usable
            else "NO_EXPECTED_RETURN_RANKING",
            "summary": (
                f"Ranked {len(ranked)} {scenario_name} {horizon}Y expected-return case(s): "
                f"{', '.join(ranked_tickers[:8]) if ranked_tickers else 'none'}."
            ),
            "ranked": [scenario.to_dict() for scenario in ranked],
        }
    return {"status": "error", "reason": f"Unknown sector tool: {tool_name}"}


def _sector_for_alpha_packet(packet: TickerSignalPacket, fallback: str) -> str:
    raw = packet.raw_valuation if isinstance(packet.raw_valuation, dict) else {}
    sector = raw.get("sector") or raw.get("sector_id") or packet.issuer_type
    return str(sector or fallback)


def _evidence_from_output(
    *,
    call: ToolCallRecord,
    output: dict[str, Any],
    evidence_index: int,
    ticker: str | None,
) -> EvidenceReference:
    status = str(output.get("status") or "unknown") if isinstance(output, dict) else "unknown"
    summary = f"{call.tool_name} returned status {status}."
    if isinstance(output, dict) and output.get("summary"):
        summary = str(output["summary"])
    return EvidenceReference(
        evidence_id=f"E{evidence_index}",
        source_type="tool_output",
        source_label=call.tool_name,
        summary=summary,
        ticker=ticker,
        excerpt=_json_preview(output, limit=2200),
        tool_call_id=call.call_id,
        confidence=_confidence_for_tool_output(output),
    )


def _sector_tool_evidence_ticker(
    *,
    tool_name: str,
    output: dict[str, Any],
    fallback_ticker: str | None,
) -> str | None:
    if tool_name == "rank_expected_return_cases":
        top_ticker = str(output.get("top_ticker") or "").upper()
        return top_ticker or fallback_ticker
    covered = [
        str(item).upper() for item in output.get("covered_tickers") or [] if str(item).strip()
    ]
    if len(covered) == 1:
        return covered[0]
    return None


def _execute_planned_calls(
    *,
    sector: str,
    as_of_date: str,
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    planned_calls: list[dict[str, Any]],
    signal_packets: dict[str, TickerSignalPacket],
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    call_start_index: int,
    executed_so_far: int,
    evidence_start_index: int,
) -> tuple[list[ToolCallRecord], list[EvidenceReference], list[str], int, bool]:
    allowed = set(allowed_tools)
    tool_records: list[ToolCallRecord] = []
    evidence: list[EvidenceReference] = []
    degraded_states: list[str] = []
    executed_count = executed_so_far
    budget_exhausted = False

    for offset, planned in enumerate(planned_calls, start=0):
        tool_name = str(planned.get("tool_name") or "").strip()
        raw_tool_input = (
            planned.get("tool_input") if isinstance(planned.get("tool_input"), dict) else {}
        )
        target_tickers = _target_tickers(planned, packets_by_ticker)
        primary_ticker = target_tickers[0] if target_tickers else None
        call = ToolCallRecord(
            call_id=f"TC{call_start_index + offset}",
            tool_name=tool_name,
            tool_input={**raw_tool_input, "tickers": target_tickers}
            if tool_name in SECTOR_TOOL_NAMES
            else raw_tool_input,
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

        if executed_count >= max(0, int(budget.max_tool_calls)):
            call.status = "SKIPPED_BUDGET_EXHAUSTED"
            call.error = "Tool-call budget exhausted before execution."
            degraded_states.append("BUDGET_EXHAUSTED")
            budget_exhausted = True
            tool_records.append(call)
            continue

        if tool_name in SECTOR_TOOL_NAMES:
            tool_input = raw_tool_input
        else:
            repair = repair_alpha_tool_input(tool_name, raw_tool_input)
            tool_input = repair.tool_input
            call.tool_input = tool_input
            if repair.repair_notes:
                call.rationale = (
                    f"{call.rationale} Tool input guardrail: {'; '.join(repair.repair_notes)}."
                ).strip()
                degraded_states.extend(repair.degraded_states)

        call.started_at = _utc_now_iso()
        try:
            if tool_name in SECTOR_TOOL_NAMES:
                output = _dispatch_sector_tool(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tickers=target_tickers,
                    packets_by_ticker=packets_by_ticker,
                    scenarios=scenarios,
                )
            else:
                if primary_ticker is None or primary_ticker not in signal_packets:
                    output = {
                        "status": "error",
                        "reason": "No target ticker available for alpha tool.",
                    }
                else:
                    alpha_packet = signal_packets[primary_ticker]
                    ctx = AlphaToolContext(
                        sector=_sector_for_alpha_packet(alpha_packet, sector),
                        ticker=primary_ticker,
                        packet=alpha_packet,
                        as_of_date=as_of_date,
                    )
                    output = dispatch_alpha_tool(tool_name, tool_input, ctx)
            call.completed_at = _utc_now_iso()
            call.status = "OK" if output.get("status") != "error" else "ERROR"
            call.output_preview = _json_preview(output)
            if call.status == "ERROR":
                call.error = str(output.get("reason") or "tool_error")
            else:
                evidence_ticker = (
                    _sector_tool_evidence_ticker(
                        tool_name=tool_name,
                        output=output,
                        fallback_ticker=primary_ticker,
                    )
                    if tool_name in SECTOR_TOOL_NAMES
                    else primary_ticker
                )
                evidence_ref = _evidence_from_output(
                    call=call,
                    output=output,
                    evidence_index=evidence_start_index + len(evidence),
                    ticker=evidence_ticker,
                )
                evidence.append(evidence_ref)
                call.evidence_ref_ids.append(evidence_ref.evidence_id)
        except Exception as exc:  # pragma: no cover - live-tool guard
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
    )


def _focused_child_objective(
    *,
    mode_label: str,
    sector: str,
    market_cap_focus: str,
    objective: str,
    selected_ticker: str,
    selection_audit: dict[str, Any],
) -> str:
    return (
        f"{objective}\n\n"
        f"Sector finalist follow-up: run a single-company {mode_label} deep dive for "
        f"{selected_ticker} in {sector}/{market_cap_focus}. Use company-level tools to "
        "test the unresolved selection-audit issues; do not make a sector-level pick. "
        f"Selection audit: {_json_preview(selection_audit, limit=2400)}"
    )


def _merge_focused_child_artifact(
    *,
    child_artifact: AutonomousRunArtifact,
    question_id: str,
    call_start_index: int,
    evidence_start_index: int,
) -> tuple[list[ToolCallRecord], list[EvidenceReference], dict[str, str]]:
    call_id_map: dict[str, str] = {}
    merged_calls: list[ToolCallRecord] = []
    for idx, call in enumerate(child_artifact.tool_calls, start=call_start_index):
        new_call_id = f"TC{idx}"
        call_id_map[call.call_id] = new_call_id
        merged_calls.append(
            ToolCallRecord(
                call_id=new_call_id,
                tool_name=call.tool_name,
                tool_input=dict(call.tool_input),
                rationale=call.rationale,
                status=call.status,
                question_id=question_id,
                started_at=call.started_at,
                completed_at=call.completed_at,
                output_preview=call.output_preview,
                output_path=call.output_path,
                error=call.error,
                evidence_ref_ids=[],
                lane="repair_fallback",
            )
        )

    evidence_id_map: dict[str, str] = {}
    merged_evidence: list[EvidenceReference] = []
    ticker = str(
        child_artifact.request.candidate_scope.get("tickers", [child_artifact.selected_ticker])[0]
        or child_artifact.selected_ticker
        or ""
    ).upper()
    for idx, item in enumerate(child_artifact.evidence, start=evidence_start_index):
        new_evidence_id = f"E{idx}"
        evidence_id_map[item.evidence_id] = new_evidence_id
        merged_evidence.append(
            EvidenceReference(
                evidence_id=new_evidence_id,
                source_type="company_autonomous_run",
                source_label=item.source_label,
                summary=f"Company autonomous run for {ticker}: {item.summary}".strip(),
                ticker=str(item.ticker or ticker).upper() if (item.ticker or ticker) else None,
                source_date=item.source_date,
                source_url=item.source_url,
                excerpt=item.excerpt,
                tool_call_id=call_id_map.get(str(item.tool_call_id or ""), item.tool_call_id),
                confidence=item.confidence,
            )
        )
    if not merged_evidence:
        merged_evidence.append(
            EvidenceReference(
                evidence_id=f"E{evidence_start_index}",
                source_type="company_autonomous_run",
                source_label="company_autonomous_run",
                summary=(
                    f"Company autonomous run for {ticker} completed with verdict "
                    f"{child_artifact.final_verdict}; no decision-usable child evidence was produced."
                ),
                ticker=ticker or None,
                confidence="LOW",
            )
        )

    original_calls = {call.call_id: call for call in child_artifact.tool_calls}
    for old_call_id, new_call_id in call_id_map.items():
        original = original_calls.get(old_call_id)
        if original is None:
            continue
        for call in merged_calls:
            if call.call_id == new_call_id:
                call.evidence_ref_ids = [
                    evidence_id_map.get(str(evidence_id), str(evidence_id))
                    for evidence_id in original.evidence_ref_ids
                ]
                break
    return merged_calls, merged_evidence, evidence_id_map


def _run_finalist_child_research_pass(
    *,
    mode: str,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    selected_ticker: str,
    selection_audit: dict[str, Any],
    signal_packets: dict[str, TickerSignalPacket] | None = None,
    company_packets: list[SectorCompanyFinancialPacket] | None = None,
    scenarios: list[SectorExpectedReturnScenario] | None = None,
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    executed_tool_count: int,
    question_start_index: int,
    belief_update_start_index: int,
    call_start_index: int,
    evidence_start_index: int,
    separate_lane_budget: bool = False,
    pipeline_version: str = "v1",
) -> tuple[
    list[SectorResearchQuestion],
    list[ToolCallRecord],
    list[EvidenceReference],
    list[BeliefUpdate],
    list[str],
    list[str],
    int,
    dict[str, Any],
]:
    mode_label = "audit-gap repair" if mode == "audit_gap_repair" else "watchlist resolution"
    prefix = "AUDIT_GAP_REPAIR" if mode == "audit_gap_repair" else "WATCHLIST_RESOLUTION"
    question_prefix = "AGR" if mode == "audit_gap_repair" else "WR"
    no_tools_status = "NO_REPAIR_TOOLS" if mode == "audit_gap_repair" else "NO_FOLLOW_UP_TOOLS"
    remaining_budget = max(0, int(budget.max_tool_calls) - int(executed_tool_count))
    metadata: dict[str, Any] = {
        "attempted": True,
        "status": "SKIPPED_BUDGET_EXHAUSTED" if remaining_budget <= 0 else "PLANNED",
        "notes": [],
    }
    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        metadata.update(
            {
                "attempted": False,
                "status": "DISABLED_V2_CANONICAL_FRONTIER",
                "notes": [
                    "Legacy LLM repair child disabled in v2; canonical competitive-frontier underwriting and selected-company validation are the only paid child lanes."
                ],
            }
        )
        return (
            [],
            [],
            [],
            [],
            [],
            list(metadata["notes"]),
            executed_tool_count,
            metadata,
        )
    if remaining_budget <= 0:
        return (
            [],
            [],
            [],
            [],
            [f"{prefix}_BUDGET_EXHAUSTED"],
            [f"{mode_label.title()} skipped because no tool-call budget remained."],
            executed_tool_count,
            metadata,
        )

    child_allowed_tools = _company_autonomy_allowed_tools(allowed_tools)
    if not child_allowed_tools:
        metadata["status"] = "NO_ALLOWED_TOOLS"
        metadata["notes"] = [f"No allowed company-level tools were available for {mode_label}."]
        return (
            [],
            [],
            [],
            [],
            [f"{prefix}_NO_ALLOWED_TOOLS"],
            list(metadata["notes"]),
            executed_tool_count,
            metadata,
        )

    canonical_child_kwargs: dict[str, Any] = {}
    if signal_packets is not None and company_packets is not None and scenarios is not None:
        canonical_signal_packets = {
            str(ticker).strip().upper(): packet
            for ticker, packet in signal_packets.items()
            if str(ticker).strip()
        }
        source_binding = _v1_child_source_bindings(
            sector=sector,
            as_of_date=as_of_date,
            company_packets=company_packets,
            scenarios=scenarios,
            signal_packets=canonical_signal_packets,
        ).get(selected_ticker)
        canonical_signal_packet = canonical_signal_packets.get(selected_ticker)
        if canonical_signal_packet is None or source_binding is None:
            metadata["status"] = "NEEDS_DATA"
            metadata["notes"] = [
                f"Single-candidate {mode_label} child was suppressed because the canonical parent packet binding was unavailable."
            ]
            return (
                [],
                [],
                [],
                [],
                [f"{prefix}_NEEDS_DATA"],
                list(metadata["notes"]),
                executed_tool_count,
                metadata,
            )
        canonical_child_kwargs = {
            "canonical_signal_packet": canonical_signal_packet,
            "source_binding": source_binding,
        }

    child_budget = AutonomousRunBudget(
        max_tool_calls=min(
            (
                COMPANY_AUTONOMY_CHILD_EXTENDED_MAX_TOOL_CALLS
                if separate_lane_budget
                else COMPANY_AUTONOMY_CHILD_MAX_TOOL_CALLS
            ),
            remaining_budget,
        ),
        max_turns=(
            COMPANY_AUTONOMY_CHILD_EXTENDED_MAX_TURNS
            if separate_lane_budget
            else COMPANY_AUTONOMY_CHILD_MAX_TURNS
        ),
        max_cost_usd=COMPANY_AUTONOMY_CHILD_MAX_COST_USD,
        timebox_seconds=None,
        max_candidates=1,
    )
    question_id = f"{question_prefix}{question_start_index}"
    question = SectorResearchQuestion(
        question_id=question_id,
        question=f"Can a single-company deep dive resolve {mode_label} issues for {selected_ticker}?",
        financial_pillar=mode_label,
        expected_decision_impact="May clear audit caps/blockers or confirm no-selection.",
        priority="HIGH",
        status="PLANNED",
        target_tickers=[selected_ticker],
        planned_tools=child_allowed_tools,
    )
    try:
        child_artifact = run_single_candidate_autonomous_analysis(
            selected_ticker,
            objective=_focused_child_objective(
                mode_label=mode_label,
                sector=sector,
                market_cap_focus=market_cap_focus,
                objective=objective,
                selected_ticker=selected_ticker,
                selection_audit=selection_audit,
            ),
            as_of_date=as_of_date,
            budget=child_budget,
            initial_budget=(
                _company_autonomy_child_budget(extended=False)
                if separate_lane_budget
                and child_budget.max_tool_calls >= COMPANY_AUTONOMY_CHILD_MAX_TOOL_CALLS
                else None
            ),
            allowed_tools=child_allowed_tools,
            execution_lane="repair_fallback",
            **canonical_child_kwargs,
        )
    except InvalidFinancialInputError:
        raise
    except Exception as exc:
        for usage in attached_provider_usage_records(exc):
            record_provider_usage({**usage, "lane": "repair_fallback"})
        metadata["status"] = "PROVIDER_ERROR"
        metadata["notes"] = [f"Single-candidate {mode_label} child run failed: {exc}"]
        question.status = "ERROR"
        return (
            [question],
            [],
            [],
            [],
            [f"{prefix}_PROVIDER_ERROR"],
            list(metadata["notes"]),
            executed_tool_count,
            metadata,
        )

    merged_calls, merged_evidence, evidence_id_map = _merge_focused_child_artifact(
        child_artifact=child_artifact,
        question_id=question_id,
        call_start_index=call_start_index,
        evidence_start_index=evidence_start_index,
    )
    for usage in child_artifact.provider_usage:
        if isinstance(usage, dict):
            record_provider_usage({**usage, "lane": "repair_fallback"})
    ok_tool_count = len([call for call in merged_calls if call.status == "OK"])
    attempted_tool_count = len([call for call in merged_calls if call.status in {"OK", "ERROR"}])
    counted_tool_calls = attempted_tool_count if separate_lane_budget else ok_tool_count
    executed_tool_count += min(child_budget.max_tool_calls, counted_tool_calls)
    if child_artifact.status != "COMPLETED":
        metadata["status"] = "PROVIDER_ERROR"
        question.status = "ERROR"
    elif ok_tool_count <= 0:
        metadata["status"] = no_tools_status
        question.status = "SKIPPED"
    else:
        metadata["status"] = "EXECUTED"
        question.status = "ANSWERED"
    metadata["notes"] = [
        f"Single-candidate {mode_label} child run {child_artifact.request.run_id} "
        f"completed with status {child_artifact.status} and verdict {child_artifact.final_verdict}."
    ]
    belief_updates = [
        BeliefUpdate(
            update_id=f"BU{belief_update_start_index + idx}",
            question_id=question_id,
            ticker=selected_ticker,
            prior_belief=item.prior_belief,
            updated_belief=item.updated_belief,
            direction=item.direction,
            confidence_after=item.confidence_after,
            summary=f"Single-candidate {mode_label}: {item.summary}",
            evidence_ref_ids=[
                evidence_id_map.get(str(evidence_id), str(evidence_id))
                for evidence_id in item.evidence_ref_ids
            ],
            remaining_uncertainty=list(item.remaining_uncertainty),
        )
        for idx, item in enumerate(child_artifact.belief_updates)
    ]
    degraded = [f"{prefix}_PROVIDER_ERROR"] if child_artifact.status != "COMPLETED" else []
    if ok_tool_count <= 0:
        degraded.append(f"{prefix}_NO_TOOLS")
    return (
        [question],
        merged_calls,
        merged_evidence,
        belief_updates,
        degraded,
        list(metadata["notes"]),
        executed_tool_count,
        metadata,
    )


def _run_audit_gap_repair_pass(
    *,
    provider: Any,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    selected_ticker: str,
    selection_audit: dict[str, Any],
    signal_packets: dict[str, TickerSignalPacket],
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    executed_tool_count: int,
    question_start_index: int,
    belief_update_start_index: int,
    separate_lane_budget: bool = False,
    pipeline_version: str = "v1",
) -> tuple[
    list[SectorResearchQuestion],
    list[ToolCallRecord],
    list[EvidenceReference],
    list[BeliefUpdate],
    list[str],
    list[str],
    int,
    dict[str, Any],
]:
    _ = (provider, belief_update_start_index)
    if _expected_return_gap_only(selection_audit) and "rank_expected_return_cases" in {
        str(tool) for tool in allowed_tools
    }:
        remaining_budget = max(0, int(budget.max_tool_calls) - int(executed_tool_count))
        metadata: dict[str, Any] = {
            "attempted": True,
            "status": "SKIPPED_BUDGET_EXHAUSTED" if remaining_budget <= 0 else "PLANNED",
            "notes": [],
        }
        if remaining_budget <= 0:
            metadata["notes"] = [
                "Expected-return audit-gap repair skipped because no tool-call budget remained."
            ]
            return (
                [],
                [],
                [],
                [],
                ["AUDIT_GAP_REPAIR_BUDGET_EXHAUSTED"],
                list(metadata["notes"]),
                executed_tool_count,
                metadata,
            )

        selected = str(selected_ticker).upper()
        target_tickers = [
            str(ticker).upper() for ticker in packets_by_ticker if str(ticker).strip()
        ]
        question_id = f"AGR{question_start_index}"
        question = SectorResearchQuestion(
            question_id=question_id,
            question=f"Can deterministic expected-return ranking resolve the audit gap for {selected}?",
            financial_pillar="audit-gap repair",
            expected_decision_impact="May clear missing expected-return evidence without a slow child run.",
            priority="HIGH",
            status="PLANNED",
            target_tickers=target_tickers,
            planned_tools=["rank_expected_return_cases"],
        )
        horizon = selection_audit.get("best_base_horizon_years")
        planned_calls = [
            {
                "question_id": question_id,
                "tool_name": "rank_expected_return_cases",
                "ticker": None,
                "tool_input": {
                    "horizon_years": int(horizon) if isinstance(horizon, int) else 5,
                    "scenario_name": "base",
                },
                "rationale": (
                    "Deterministic audit-gap repair for missing expected-return evidence; "
                    f"the existing scenario table already contains a base case for {selected}."
                ),
                "target_tickers": target_tickers,
            }
        ]
        (
            repair_tool_calls,
            repair_evidence,
            repair_degraded,
            repaired_tool_count,
            budget_exhausted,
        ) = _execute_planned_calls(
            sector=sector,
            as_of_date=as_of_date,
            allowed_tools=allowed_tools,
            budget=budget,
            planned_calls=planned_calls,
            signal_packets=signal_packets,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            call_start_index=len(tool_calls) + 1,
            executed_so_far=executed_tool_count,
            evidence_start_index=len(evidence) + 1,
        )
        _mark_question_statuses([question], repair_tool_calls)
        ok_count = len([call for call in repair_tool_calls if call.status == "OK"])
        if budget_exhausted:
            metadata["status"] = "BUDGET_EXHAUSTED"
        elif ok_count > 0:
            metadata["status"] = "EXECUTED"
        else:
            metadata["status"] = "NO_REPAIR_TOOLS"
        metadata["notes"] = [
            (
                f"Deterministic expected-return audit-gap repair for {selected}: "
                f"{metadata['status']}."
            )
        ]
        return (
            [question],
            repair_tool_calls,
            repair_evidence,
            [],
            repair_degraded,
            list(metadata["notes"]),
            repaired_tool_count,
            metadata,
        )

    return _run_finalist_child_research_pass(
        mode="audit_gap_repair",
        sector=sector,
        market_cap_focus=market_cap_focus,
        objective=objective,
        as_of_date=as_of_date,
        selected_ticker=selected_ticker,
        selection_audit=selection_audit,
        signal_packets=signal_packets,
        company_packets=list(packets_by_ticker.values()),
        scenarios=scenarios,
        allowed_tools=allowed_tools,
        budget=budget,
        executed_tool_count=executed_tool_count,
        question_start_index=question_start_index,
        belief_update_start_index=belief_update_start_index,
        call_start_index=len(tool_calls) + 1,
        evidence_start_index=len(evidence) + 1,
        separate_lane_budget=separate_lane_budget,
        pipeline_version=pipeline_version,
    )


def _run_broad_expected_return_repair_pass(
    *,
    sector: str,
    as_of_date: str,
    repair_targets: list[dict[str, Any]],
    signal_packets: dict[str, TickerSignalPacket],
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    executed_tool_count: int,
    question_start_index: int,
) -> tuple[
    list[SectorResearchQuestion],
    list[ToolCallRecord],
    list[EvidenceReference],
    list[BeliefUpdate],
    list[str],
    list[str],
    int,
    dict[str, Any],
]:
    _ = signal_packets
    metadata: dict[str, Any] = {
        "attempted": bool(repair_targets),
        "status": "NO_OP" if not repair_targets else "PLANNED",
        "notes": [],
        "targets": [],
    }
    if not repair_targets:
        return [], [], [], [], [], [], executed_tool_count, metadata

    remaining_budget = max(0, int(budget.max_tool_calls) - int(executed_tool_count))
    executable_count = min(
        len(repair_targets),
        remaining_budget // AUDIT_GAP_REPAIR_TOOLS_PER_TARGET,
    )
    executable_targets = repair_targets[:executable_count]
    skipped_targets = repair_targets[executable_count:]
    degraded_states: list[str] = []
    if skipped_targets:
        degraded_states.append("AUDIT_GAP_REPAIR_BUDGET_PARTIAL")
    if not executable_targets:
        metadata["status"] = "SKIPPED_BUDGET_EXHAUSTED"
        metadata["targets"].extend(
            {"ticker": str(item["ticker"]), "status": "SKIPPED_BUDGET_EXHAUSTED"}
            for item in skipped_targets
        )
        metadata["notes"] = [
            "Expected-return audit-gap repair skipped because no two-tool budget remained."
        ]
        return (
            [],
            [],
            [],
            [],
            degraded_states,
            list(metadata["notes"]),
            executed_tool_count,
            metadata,
        )

    questions: list[SectorResearchQuestion] = []
    planned_calls: list[dict[str, Any]] = []
    for index, target in enumerate(executable_targets):
        ticker = str(target["ticker"]).upper()
        question_id = f"AGR{question_start_index + index}"
        horizon = target.get("best_base_horizon_years") or 5
        questions.append(
            SectorResearchQuestion(
                question_id=question_id,
                question=f"Can deterministic expected-return evidence resolve the audit gap for {ticker}?",
                financial_pillar="audit-gap repair",
                expected_decision_impact="May clear missing expected-return evidence before final decision.",
                priority="HIGH",
                status="PLANNED",
                target_tickers=[ticker],
                planned_tools=["compare_expected_return_scenarios", "rank_expected_return_cases"],
            )
        )
        planned_calls.extend(
            [
                {
                    "question_id": question_id,
                    "tool_name": "compare_expected_return_scenarios",
                    "ticker": None,
                    "tool_input": {},
                    "rationale": (
                        "Deterministic audit-gap repair for missing expected-return evidence; "
                        f"compare available scenarios for {ticker}."
                    ),
                    "target_tickers": [ticker],
                },
                {
                    "question_id": question_id,
                    "tool_name": "rank_expected_return_cases",
                    "ticker": None,
                    "tool_input": {
                        "horizon_years": int(horizon) if isinstance(horizon, int) else 5,
                        "scenario_name": "base",
                    },
                    "rationale": (
                        "Deterministic audit-gap repair for missing expected-return evidence; "
                        f"rank the base case for {ticker}."
                    ),
                    "target_tickers": [ticker],
                },
            ]
        )

    (
        repair_tool_calls,
        repair_evidence,
        repair_degraded,
        repaired_tool_count,
        budget_exhausted,
    ) = _execute_planned_calls(
        sector=sector,
        as_of_date=as_of_date,
        allowed_tools=allowed_tools,
        budget=budget,
        planned_calls=planned_calls,
        signal_packets=signal_packets,
        packets_by_ticker=packets_by_ticker,
        scenarios=scenarios,
        call_start_index=len(tool_calls) + 1,
        executed_so_far=executed_tool_count,
        evidence_start_index=len(evidence) + 1,
    )
    _mark_question_statuses(questions, repair_tool_calls)
    calls_by_question: dict[str, list[ToolCallRecord]] = {}
    for call in repair_tool_calls:
        if call.question_id:
            calls_by_question.setdefault(str(call.question_id), []).append(call)
    for question in questions:
        ticker = question.target_tickers[0] if question.target_tickers else None
        statuses = [call.status for call in calls_by_question.get(question.question_id, [])]
        metadata["targets"].append(
            {
                "ticker": ticker,
                "status": "OK"
                if statuses and all(status == "OK" for status in statuses)
                else "ERROR",
            }
        )
    metadata["targets"].extend(
        {"ticker": str(item["ticker"]), "status": "SKIPPED_BUDGET_EXHAUSTED"}
        for item in skipped_targets
    )
    degraded_states.extend(repair_degraded)
    if budget_exhausted and "AUDIT_GAP_REPAIR_BUDGET_PARTIAL" not in degraded_states:
        degraded_states.append("AUDIT_GAP_REPAIR_BUDGET_PARTIAL")
    ok_targets = [item for item in metadata["targets"] if item.get("status") == "OK"]
    metadata["status"] = (
        "EXECUTED"
        if len(ok_targets) == len(executable_targets) and not skipped_targets
        else "PARTIAL"
    )
    metadata["notes"] = [
        (
            "Deterministic expected-return audit-gap repair targeted "
            f"{len(executable_targets)} candidate(s); {len(skipped_targets)} skipped for budget."
        )
    ]
    return (
        questions,
        repair_tool_calls,
        repair_evidence,
        [],
        list(dict.fromkeys(degraded_states)),
        list(metadata["notes"]),
        repaired_tool_count,
        metadata,
    )


def _run_watchlist_resolution_pass(
    *,
    provider: Any,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    selected_ticker: str,
    selection_audit: dict[str, Any],
    signal_packets: dict[str, TickerSignalPacket],
    packets_by_ticker: dict[str, SectorCompanyFinancialPacket],
    scenarios: list[SectorExpectedReturnScenario],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    allowed_tools: list[str],
    budget: AutonomousRunBudget,
    executed_tool_count: int,
    question_start_index: int,
    belief_update_start_index: int,
    separate_lane_budget: bool = False,
    pipeline_version: str = "v1",
) -> tuple[
    list[SectorResearchQuestion],
    list[ToolCallRecord],
    list[EvidenceReference],
    list[BeliefUpdate],
    list[str],
    list[str],
    int,
    dict[str, Any],
]:
    _ = provider
    return _run_finalist_child_research_pass(
        mode="watchlist_resolution",
        sector=sector,
        market_cap_focus=market_cap_focus,
        objective=objective,
        as_of_date=as_of_date,
        selected_ticker=selected_ticker,
        selection_audit=selection_audit,
        signal_packets=signal_packets,
        company_packets=list(packets_by_ticker.values()),
        scenarios=scenarios,
        allowed_tools=allowed_tools,
        budget=budget,
        executed_tool_count=executed_tool_count,
        question_start_index=question_start_index,
        belief_update_start_index=belief_update_start_index,
        call_start_index=len(tool_calls) + 1,
        evidence_start_index=len(evidence) + 1,
        separate_lane_budget=separate_lane_budget,
        pipeline_version=pipeline_version,
    )


def _artifact_from_decision(
    *,
    run_id: str,
    sector: str,
    market_cap_focus: str,
    objective: str,
    as_of_date: str,
    created_at: str,
    framework: SectorFinancialFramework | None,
    company_packets: list[SectorCompanyFinancialPacket],
    research_questions: list[SectorResearchQuestion],
    scenarios: list[SectorExpectedReturnScenario],
    tool_calls: list[ToolCallRecord],
    evidence: list[EvidenceReference],
    belief_updates: list[BeliefUpdate],
    final_decision: SectorFinalDecision,
    degraded_states: list[str],
    audit_notes: list[str],
    candidate_selection: dict[str, Any] | None,
    provider: Any | None = None,
    signal_packets: dict[str, TickerSignalPacket] | None = None,
    allowed_tools: list[str] | None = None,
    budget: AutonomousRunBudget | None = None,
    executed_tool_count: int = 0,
    company_autonomy_attempted: bool = False,
    company_autonomy_status: str | None = None,
    company_autonomy_notes: list[str] | None = None,
    company_autonomy_runs: list[dict[str, Any]] | None = None,
    competitive_frontier: dict[str, Any] | None = None,
    pipeline_version: str = "v1",
) -> AutonomousSectorFinancialRunArtifact:
    packets_by_ticker = {str(packet.ticker).upper(): packet for packet in company_packets}

    def require_run_scope_unchanged() -> None:
        _require_candidate_financial_integrity_binding(
            candidate_selection,
            packets=company_packets,
            scenarios=scenarios,
        )

    require_run_scope_unchanged()
    packet_tickers = set(packets_by_ticker)
    decision = final_decision
    states = list(degraded_states)
    notes = list(audit_notes)
    audit_gap_repair_attempted = False
    audit_gap_repair_status: str | None = None
    audit_gap_repair_notes: list[str] = []
    watchlist_resolution_attempted = False
    watchlist_resolution_status: str | None = None
    watchlist_resolution_notes: list[str] = []
    no_selection_finalist_audit_attempted = False
    no_selection_finalist_audit_status: str | None = None
    no_selection_finalist_audit_focus_ticker: str | None = None
    no_selection_finalist_audit_notes: list[str] = []
    no_selection_finalist_data_incomplete_focus: str | None = None
    no_selection_finalist_data_incomplete_audit: dict[str, Any] | None = None
    no_selection_finalist_resolution_attempted = False
    no_selection_finalist_resolution_status: str | None = None
    no_selection_finalist_resolution_notes: list[str] = []
    alternate_finalist_audit_attempted = False
    alternate_finalist_audit_status: str | None = None
    alternate_finalist_audit_notes: list[str] = []
    alternate_finalist_audit_results: list[dict[str, Any]] = []
    audit_gap_repair_target_statuses: list[dict[str, Any]] = []
    separate_repair_lane = pipeline_version == SECTOR_PIPELINE_VERSION_V2
    repair_lane_tool_count = 0
    autonomy_notes = list(company_autonomy_notes or [])
    autonomy_runs = [dict(item) for item in company_autonomy_runs or []]
    selected_ticker = str(decision.selected_ticker or "").upper() or None
    audit_ticker = selected_ticker
    guardrail_normalized = False
    skip_no_selection_finalist_audit = False
    if selected_ticker and selected_ticker not in packet_tickers:
        states.append("SELECTED_TICKER_NOT_IN_SCOPE")
        notes.append(
            "Provider selected a ticker outside the analyzed sector scope; converted to no-selection."
        )
        guardrail_normalized = True
        skip_no_selection_finalist_audit = True
        decision = SectorFinalDecision(
            verdict="NO_SELECTION",
            confidence=None,
            selected_ticker=None,
            expected_annualized_return_range=None,
            thesis="No company selected because the proposed selection was outside the analyzed scope.",
            key_risk="Provider decision referenced an out-of-scope ticker.",
            downside_case="Out-of-scope selection cannot be underwritten.",
            no_selection_reason="Selected ticker was not in the analyzed packet set.",
            selection_blockers=["SELECTED_TICKER_NOT_IN_SCOPE"],
        )
    elif selected_ticker:
        packet_blockers = _selection_blockers_for_packet(packets_by_ticker[selected_ticker])
        if packet_blockers:
            states.append("SELECTED_TICKER_PACKET_BLOCKED")
            notes.append(
                "Provider selected an in-scope ticker with deterministic packet blockers; converted to no-selection."
            )
            guardrail_normalized = True
            skip_no_selection_finalist_audit = True
            decision = SectorFinalDecision(
                verdict="NO_SELECTION",
                confidence=None,
                selected_ticker=None,
                expected_annualized_return_range=None,
                thesis=decision.thesis,
                key_risk=decision.key_risk,
                downside_case=decision.downside_case,
                no_selection_reason=(
                    f"No company selected because {selected_ticker} has deterministic packet blockers: "
                    f"{', '.join(packet_blockers)}."
                ),
                falsifiers=list(decision.falsifiers),
                why_selected_over_finalists=list(decision.why_selected_over_finalists),
                rejected_finalists=[dict(item) for item in decision.rejected_finalists],
                selection_blockers=["SELECTED_TICKER_PACKET_BLOCKED", *packet_blockers],
                confidence_cap_reasons=list(decision.confidence_cap_reasons),
                evidence_ref_ids=list(decision.evidence_ref_ids),
            )
        elif decision.selection_blockers:
            provider_blockers = [
                str(item) for item in decision.selection_blockers if str(item).strip()
            ]
            states.append("SELECTED_TICKER_SELECTION_BLOCKED")
            notes.append(
                "Provider selected a ticker while also reporting binding selection blockers; "
                "converted to no-selection."
            )
            guardrail_normalized = True
            skip_no_selection_finalist_audit = True
            decision = SectorFinalDecision(
                verdict="NO_SELECTION",
                confidence=None,
                selected_ticker=None,
                expected_annualized_return_range=None,
                thesis=decision.thesis,
                key_risk=decision.key_risk,
                downside_case=decision.downside_case,
                no_selection_reason=(
                    f"No company selected because {selected_ticker} has unresolved selection blockers: "
                    f"{', '.join(provider_blockers)}."
                ),
                falsifiers=list(decision.falsifiers),
                why_selected_over_finalists=list(decision.why_selected_over_finalists),
                rejected_finalists=[dict(item) for item in decision.rejected_finalists],
                selection_blockers=["SELECTED_TICKER_SELECTION_BLOCKED", *provider_blockers],
                confidence_cap_reasons=list(decision.confidence_cap_reasons),
                evidence_ref_ids=list(decision.evidence_ref_ids),
            )
    if (decision.verdict or "").upper() in {
        "NO_SELECTION",
        "NO_WINNER",
    } or not decision.selected_ticker:
        original_confidence = decision.confidence
        original_reason = decision.no_selection_reason
        missing_decision_fields = [
            field_name
            for field_name, value in [
                ("thesis", decision.thesis),
                ("key_risk", decision.key_risk),
                ("downside_case", decision.downside_case),
            ]
            if not str(value or "").strip()
        ]
        if missing_decision_fields:
            states.append("STRUCTURED_DECISION_INCOMPLETE")
            notes.append(
                "Provider returned no selection without complete final-decision narrative fields; "
                "runtime synthesized conservative no-selection text from degraded states."
            )
            guardrail_normalized = True
            skip_no_selection_finalist_audit = True
        if original_confidence is not None or not original_reason:
            guardrail_normalized = True
        if not original_reason:
            skip_no_selection_finalist_audit = True
        no_selection_reason = _synthesized_no_selection_reason(decision=decision, states=states)
        decision = SectorFinalDecision(
            verdict="NO_SELECTION",
            confidence=None,
            selected_ticker=None,
            expected_annualized_return_range=None,
            thesis=decision.thesis or "No sector selection was made.",
            key_risk=decision.key_risk or no_selection_reason,
            downside_case=decision.downside_case or "No underwritten downside case was finalized.",
            no_selection_reason=no_selection_reason,
            falsifiers=list(decision.falsifiers),
            why_selected_over_finalists=list(decision.why_selected_over_finalists),
            rejected_finalists=[dict(item) for item in decision.rejected_finalists],
            selection_blockers=list(decision.selection_blockers),
            confidence_cap_reasons=list(decision.confidence_cap_reasons),
            evidence_ref_ids=list(decision.evidence_ref_ids),
        )
    selection_audit = _selection_audit_for_ticker(
        selected_ticker=audit_ticker or decision.selected_ticker,
        packets_by_ticker=packets_by_ticker,
        scenarios=scenarios,
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=states,
        framework=framework,
    )
    if (
        isinstance(allowed_tools, list)
        and budget is not None
        and {"compare_expected_return_scenarios", "rank_expected_return_cases"}.issubset(
            set(allowed_tools)
        )
    ):
        pre_repair_relative_ranking = _relative_ranking(
            company_packets=company_packets,
            scenarios=scenarios,
            tool_calls=tool_calls,
            evidence=evidence,
            degraded_states=states,
            company_autonomy_runs=autonomy_runs,
            framework=framework,
        )
        broad_repair_targets = _expected_return_repair_targets(
            relative_ranking=pre_repair_relative_ranking,
            scenarios=scenarios,
        )
        if broad_repair_targets:
            audit_gap_repair_attempted = True
            (
                repair_questions,
                repair_tool_calls,
                repair_evidence,
                repair_belief_updates,
                repair_degraded,
                repair_notes,
                repair_result_count,
                repair_metadata,
            ) = _run_broad_expected_return_repair_pass(
                sector=sector,
                as_of_date=as_of_date,
                repair_targets=broad_repair_targets,
                signal_packets=signal_packets or {},
                packets_by_ticker=packets_by_ticker,
                scenarios=scenarios,
                tool_calls=tool_calls,
                evidence=evidence,
                allowed_tools=allowed_tools,
                budget=budget,
                executed_tool_count=(
                    repair_lane_tool_count if separate_repair_lane else executed_tool_count
                ),
                question_start_index=len(research_questions) + 1,
            )
            if separate_repair_lane:
                repair_lane_tool_count = repair_result_count
            else:
                executed_tool_count = repair_result_count
            research_questions.extend(repair_questions)
            tool_calls.extend(repair_tool_calls)
            evidence.extend(repair_evidence)
            belief_updates.extend(repair_belief_updates)
            states.extend(repair_degraded)
            audit_gap_repair_status = str(repair_metadata.get("status") or "UNKNOWN")
            audit_gap_repair_notes.extend(str(item) for item in repair_notes if str(item).strip())
            audit_gap_repair_target_statuses.extend(
                dict(item)
                for item in repair_metadata.get("targets") or []
                if isinstance(item, dict)
            )
            notes.append(
                f"Broad expected-return audit-gap repair attempted: {audit_gap_repair_status}."
            )
            notes.extend(audit_gap_repair_notes)
            _mark_question_statuses(research_questions, tool_calls)
            selection_audit = _selection_audit_for_ticker(
                selected_ticker=audit_ticker or decision.selected_ticker,
                packets_by_ticker=packets_by_ticker,
                scenarios=scenarios,
                tool_calls=tool_calls,
                evidence=evidence,
                degraded_states=states,
                framework=framework,
            )
            if decision.selected_ticker:
                pre_audit_repair_status = audit_gap_repair_status
                if selection_audit.get("status") == "PASS":
                    audit_gap_repair_status = "RESOLVED_SELECTED"
                elif selection_audit.get("status") == "WATCHLIST_ONLY":
                    audit_gap_repair_status = "PARTIALLY_RESOLVED_WATCHLIST"
                elif pre_audit_repair_status in {
                    "BUDGET_EXHAUSTED",
                    "NO_REPAIR_TOOLS",
                    "NO_ALLOWED_TOOLS",
                    "PROVIDER_ERROR",
                    "SKIPPED_BUDGET_EXHAUSTED",
                    "PARTIAL",
                }:
                    audit_gap_repair_status = pre_audit_repair_status
                else:
                    audit_gap_repair_status = "UNRESOLVED_BLOCKED"
    if (
        not decision.selected_ticker
        and tool_calls
        and packets_by_ticker
        and not skip_no_selection_finalist_audit
        and "STRUCTURED_DECISION_INCOMPLETE" not in states
        and "BUDGET_EXHAUSTED" not in states
    ):
        finalist_focus = _no_selection_finalist_focus(
            decision=decision,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            candidate_selection=candidate_selection,
        )
        if finalist_focus:
            focus_ticker = str(finalist_focus["ticker"]).upper()
            no_selection_finalist_audit_attempted = True
            no_selection_finalist_audit_focus_ticker = focus_ticker
            no_selection_finalist_audit_notes.append(
                f"Audited {focus_ticker} after provider no-selection: {finalist_focus['reason']}."
            )
            notes.append(
                f"No-selection finalist audit focused on {focus_ticker}: {finalist_focus['reason']}."
            )
            selection_audit = _selection_audit_for_ticker(
                selected_ticker=focus_ticker,
                packets_by_ticker=packets_by_ticker,
                scenarios=scenarios,
                tool_calls=tool_calls,
                evidence=evidence,
                degraded_states=states,
                framework=framework,
            )
            no_selection_finalist_audit_status = str(selection_audit.get("status") or "UNKNOWN")
            if (
                _audit_gap_is_repairable(selection_audit, allowed_tools or [])
                and provider is not None
                and isinstance(signal_packets, dict)
                and isinstance(allowed_tools, list)
                and budget is not None
                and focus_ticker in packets_by_ticker
            ):
                require_run_scope_unchanged()
                audit_gap_repair_attempted = True
                (
                    repair_questions,
                    repair_tool_calls,
                    repair_evidence,
                    repair_belief_updates,
                    repair_degraded,
                    repair_notes,
                    repair_result_count,
                    repair_metadata,
                ) = _run_audit_gap_repair_pass(
                    provider=provider,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=objective,
                    as_of_date=as_of_date,
                    selected_ticker=focus_ticker,
                    selection_audit=selection_audit,
                    signal_packets=signal_packets,
                    packets_by_ticker=packets_by_ticker,
                    scenarios=scenarios,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    allowed_tools=allowed_tools,
                    budget=budget,
                    executed_tool_count=(
                        repair_lane_tool_count if separate_repair_lane else executed_tool_count
                    ),
                    question_start_index=len(research_questions) + 1,
                    belief_update_start_index=len(belief_updates) + 1,
                    separate_lane_budget=separate_repair_lane,
                    pipeline_version=pipeline_version,
                )
                if separate_repair_lane:
                    repair_lane_tool_count = repair_result_count
                else:
                    executed_tool_count = repair_result_count
                research_questions.extend(repair_questions)
                tool_calls.extend(repair_tool_calls)
                evidence.extend(repair_evidence)
                belief_updates.extend(repair_belief_updates)
                states.extend(repair_degraded)
                audit_gap_repair_status = str(repair_metadata.get("status") or "UNKNOWN")
                audit_gap_repair_notes = [str(item) for item in repair_notes if str(item).strip()]
                notes.append(
                    f"No-selection finalist audit-gap repair attempted for {focus_ticker}: "
                    f"{audit_gap_repair_status}."
                )
                notes.extend(audit_gap_repair_notes)
                no_selection_finalist_audit_notes.append(
                    f"Audit-gap repair for no-selection finalist {focus_ticker}: {audit_gap_repair_status}."
                )
                _mark_question_statuses(research_questions, tool_calls)
                selection_audit = _selection_audit_for_ticker(
                    selected_ticker=focus_ticker,
                    packets_by_ticker=packets_by_ticker,
                    scenarios=scenarios,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    degraded_states=states,
                    framework=framework,
                )
                pre_audit_repair_status = audit_gap_repair_status
                if selection_audit.get("status") == "PASS":
                    audit_gap_repair_status = "RESOLVED_SELECTED"
                    audit_gap_repair_notes.append(
                        "Repair evidence cleared the binding audit blockers."
                    )
                elif selection_audit.get("status") == "WATCHLIST_ONLY":
                    audit_gap_repair_status = "PARTIALLY_RESOLVED_WATCHLIST"
                    audit_gap_repair_notes.append(
                        "Repair evidence cleared hard blockers but left watchlist-only caps."
                    )
                elif pre_audit_repair_status in {
                    "BUDGET_EXHAUSTED",
                    "NO_REPAIR_TOOLS",
                    "NO_ALLOWED_TOOLS",
                    "PROVIDER_ERROR",
                    "SKIPPED_BUDGET_EXHAUSTED",
                }:
                    audit_gap_repair_status = pre_audit_repair_status
                    audit_gap_repair_notes.append(
                        "Audit-gap repair did not complete with decision-usable evidence."
                    )
                else:
                    audit_gap_repair_status = "UNRESOLVED_BLOCKED"
                    audit_gap_repair_notes.append(
                        "Repair evidence did not clear the binding audit blockers."
                    )
                audit_gap_repair_target_statuses.append(
                    {"ticker": focus_ticker, "status": audit_gap_repair_status}
                )
                no_selection_finalist_audit_status = str(selection_audit.get("status") or "UNKNOWN")

            if (
                selection_audit.get("status") == "WATCHLIST_ONLY"
                and provider is not None
                and isinstance(signal_packets, dict)
                and isinstance(allowed_tools, list)
                and budget is not None
                and focus_ticker in packets_by_ticker
            ):
                require_run_scope_unchanged()
                no_selection_finalist_resolution_attempted = True
                (
                    resolution_questions,
                    resolution_tool_calls,
                    resolution_evidence,
                    resolution_belief_updates,
                    resolution_degraded,
                    resolution_notes,
                    repair_result_count,
                    resolution_metadata,
                ) = _run_watchlist_resolution_pass(
                    provider=provider,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=objective,
                    as_of_date=as_of_date,
                    selected_ticker=focus_ticker,
                    selection_audit=selection_audit,
                    signal_packets=signal_packets,
                    packets_by_ticker=packets_by_ticker,
                    scenarios=scenarios,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    allowed_tools=allowed_tools,
                    budget=budget,
                    executed_tool_count=(
                        repair_lane_tool_count if separate_repair_lane else executed_tool_count
                    ),
                    question_start_index=len(research_questions) + 1,
                    belief_update_start_index=len(belief_updates) + 1,
                    separate_lane_budget=separate_repair_lane,
                    pipeline_version=pipeline_version,
                )
                if separate_repair_lane:
                    repair_lane_tool_count = repair_result_count
                else:
                    executed_tool_count = repair_result_count
                research_questions.extend(resolution_questions)
                tool_calls.extend(resolution_tool_calls)
                evidence.extend(resolution_evidence)
                belief_updates.extend(resolution_belief_updates)
                states.extend(resolution_degraded)
                pre_resolution_status = str(resolution_metadata.get("status") or "UNKNOWN")
                no_selection_finalist_resolution_status = (
                    "SKIPPED_BUDGET_EXHAUSTED"
                    if pre_resolution_status == "BUDGET_EXHAUSTED"
                    else pre_resolution_status
                )
                no_selection_finalist_resolution_notes = [
                    str(item) for item in resolution_notes if str(item).strip()
                ]
                notes.append(
                    f"No-selection finalist resolution attempted for {focus_ticker}: "
                    f"{no_selection_finalist_resolution_status}."
                )
                notes.extend(no_selection_finalist_resolution_notes)
                _mark_question_statuses(research_questions, tool_calls)
                selection_audit = _selection_audit_for_ticker(
                    selected_ticker=focus_ticker,
                    packets_by_ticker=packets_by_ticker,
                    scenarios=scenarios,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    degraded_states=states,
                    framework=framework,
                )
                no_selection_finalist_audit_status = str(selection_audit.get("status") or "UNKNOWN")
                if selection_audit.get("status") == "PASS":
                    no_selection_finalist_resolution_status = "RESOLVED_SELECTED"
                    no_selection_finalist_resolution_notes.append(
                        "Follow-up evidence cleared the no-selection finalist audit caps."
                    )
                elif selection_audit.get("status") == "BLOCKED":
                    no_selection_finalist_resolution_status = "RESOLVED_NO_SELECTION"
                    no_selection_finalist_resolution_notes.append(
                        "Follow-up evidence exposed binding selection blockers for the finalist."
                    )
                elif pre_resolution_status in {
                    "BUDGET_EXHAUSTED",
                    "SKIPPED_BUDGET_EXHAUSTED",
                    "NO_ALLOWED_TOOLS",
                    "NO_FOLLOW_UP_TOOLS",
                    "PROVIDER_ERROR",
                }:
                    no_selection_finalist_resolution_status = (
                        "SKIPPED_BUDGET_EXHAUSTED"
                        if pre_resolution_status in {"BUDGET_EXHAUSTED", "SKIPPED_BUDGET_EXHAUSTED"}
                        else pre_resolution_status
                    )
                    no_selection_finalist_resolution_notes.append(
                        "No-selection finalist resolution did not complete with decision-usable follow-up evidence."
                    )
                else:
                    no_selection_finalist_resolution_status = "UNRESOLVED_NO_SELECTION"
                    no_selection_finalist_resolution_notes.append(
                        "Follow-up evidence did not clear all no-selection finalist audit caps."
                    )

            if selection_audit.get("status") == "PASS":
                no_selection_finalist_audit_status = "PASS"
                no_selection_finalist_audit_notes.append(
                    "No-selection finalist audit passed; runtime upgraded the guarded result to SELECTED."
                )
                notes.append(
                    f"No-selection finalist audit upgraded {focus_ticker} because the deterministic selection audit passed."
                )
                guardrail_normalized = True
                decision = _decision_from_finalist_audit_pass(
                    original_decision=decision,
                    focus_ticker=focus_ticker,
                    selection_audit=selection_audit,
                )
            elif selection_audit.get("status") == "WATCHLIST_ONLY":
                no_selection_finalist_audit_status = "WATCHLIST_ONLY"
                reason = _no_selection_reason_from_finalist_audit(
                    focus_ticker=focus_ticker,
                    selection_audit=selection_audit,
                )
                no_selection_finalist_audit_notes.append(reason)
                decision = SectorFinalDecision(
                    verdict="NO_SELECTION",
                    confidence=None,
                    selected_ticker=None,
                    expected_annualized_return_range=None,
                    thesis=_deterministic_no_selection_thesis(
                        focus_ticker=focus_ticker,
                        selection_audit=selection_audit,
                    ),
                    key_risk=_deterministic_no_selection_key_risk(selection_audit),
                    downside_case=_deterministic_no_selection_downside(selection_audit),
                    no_selection_reason=reason,
                    falsifiers=list(decision.falsifiers),
                    why_selected_over_finalists=list(decision.why_selected_over_finalists),
                    rejected_finalists=[dict(item) for item in decision.rejected_finalists],
                    selection_blockers=list(decision.selection_blockers),
                    confidence_cap_reasons=list(
                        dict.fromkeys(
                            [
                                *decision.confidence_cap_reasons,
                                *_mergeable_confidence_caps(selection_audit),
                            ]
                        )
                    ),
                    evidence_ref_ids=list(decision.evidence_ref_ids),
                )
            elif selection_audit.get("status") == "DATA_INCOMPLETE":
                # The finalist's only obstacle is a fetchable data-availability
                # gap. Like BLOCKED, the finalist itself cannot be promoted here,
                # so we tentatively clear the selection (ticker=None) to let the
                # alternate-finalist audit prefer a clean alternate. If no
                # alternate passes, the primary is re-surfaced below as a
                # DATA_INCOMPLETE resolve-then-promote candidate rather than a
                # hard NO_SELECTION.
                no_selection_finalist_audit_status = "DATA_INCOMPLETE"
                no_selection_finalist_data_incomplete_focus = focus_ticker
                no_selection_finalist_data_incomplete_audit = selection_audit
                reason = _no_selection_reason_from_finalist_audit(
                    focus_ticker=focus_ticker,
                    selection_audit=selection_audit,
                )
                no_selection_finalist_audit_notes.append(reason)
                decision = SectorFinalDecision(
                    verdict="NO_SELECTION",
                    confidence=None,
                    selected_ticker=None,
                    expected_annualized_return_range=None,
                    thesis=_deterministic_no_selection_thesis(
                        focus_ticker=focus_ticker,
                        selection_audit=selection_audit,
                    ),
                    key_risk=_deterministic_no_selection_key_risk(selection_audit),
                    downside_case=_deterministic_no_selection_downside(selection_audit),
                    no_selection_reason=reason,
                    falsifiers=list(decision.falsifiers),
                    why_selected_over_finalists=list(decision.why_selected_over_finalists),
                    rejected_finalists=[dict(item) for item in decision.rejected_finalists],
                    selection_blockers=list(decision.selection_blockers),
                    confidence_cap_reasons=list(decision.confidence_cap_reasons),
                    evidence_ref_ids=list(decision.evidence_ref_ids),
                )
            elif selection_audit.get("status") == "BLOCKED":
                no_selection_finalist_audit_status = "BLOCKED"
                audit_blockers = [str(item) for item in selection_audit.get("hard_blockers") or []]
                reason = _no_selection_reason_from_finalist_audit(
                    focus_ticker=focus_ticker,
                    selection_audit=selection_audit,
                )
                no_selection_finalist_audit_notes.append(reason)
                decision = SectorFinalDecision(
                    verdict="NO_SELECTION",
                    confidence=None,
                    selected_ticker=None,
                    expected_annualized_return_range=None,
                    thesis=_deterministic_no_selection_thesis(
                        focus_ticker=focus_ticker,
                        selection_audit=selection_audit,
                    ),
                    key_risk=_deterministic_no_selection_key_risk(selection_audit),
                    downside_case=_deterministic_no_selection_downside(selection_audit),
                    no_selection_reason=reason,
                    falsifiers=list(decision.falsifiers),
                    why_selected_over_finalists=list(decision.why_selected_over_finalists),
                    rejected_finalists=[dict(item) for item in decision.rejected_finalists],
                    selection_blockers=list(
                        dict.fromkeys(
                            [
                                *decision.selection_blockers,
                                "NO_SELECTION_FINALIST_AUDIT_BLOCKED",
                                *audit_blockers,
                            ]
                        )
                    ),
                    confidence_cap_reasons=list(decision.confidence_cap_reasons),
                    evidence_ref_ids=list(decision.evidence_ref_ids),
                )

            if (
                selection_audit.get("status") in {"BLOCKED", "DATA_INCOMPLETE"}
                and not decision.selected_ticker
                and scenarios
            ):
                alternate_finalist_audit_attempted = True
                (
                    alternate_finalist_audit_status,
                    alternate_notes,
                    alternate_results,
                    alternate_pass_audit,
                ) = _run_alternate_finalist_audit(
                    decision=decision,
                    primary_focus_ticker=focus_ticker,
                    packets_by_ticker=packets_by_ticker,
                    scenarios=scenarios,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    degraded_states=states,
                    framework=framework,
                )
                alternate_finalist_audit_notes = [
                    str(item) for item in alternate_notes if str(item).strip()
                ]
                alternate_finalist_audit_results = alternate_results
                notes.append(
                    f"Alternate finalist audit after blocked no-selection finalist {focus_ticker}: "
                    f"{alternate_finalist_audit_status}."
                )
                notes.extend(alternate_finalist_audit_notes)
                if alternate_pass_audit is not None:
                    alternate_ticker = str(
                        alternate_pass_audit.get("selected_ticker") or ""
                    ).upper()
                    if alternate_ticker:
                        selection_audit = alternate_pass_audit
                        no_selection_finalist_audit_notes.append(
                            f"Alternate finalist {alternate_ticker} passed audit after {focus_ticker} was blocked."
                        )
                        guardrail_normalized = True
                        decision = _decision_from_finalist_audit_pass(
                            original_decision=decision,
                            focus_ticker=alternate_ticker,
                            selection_audit=selection_audit,
                        )

            # No alternate finalist cleared the audit, but the primary finalist's
            # only obstacle was a fetchable data-availability gap. Re-surface it
            # as a DATA_INCOMPLETE resolve-then-promote candidate (ticker
            # preserved) instead of discarding it as a hard NO_SELECTION.
            if (
                not decision.selected_ticker
                and no_selection_finalist_data_incomplete_focus
                and no_selection_finalist_data_incomplete_audit is not None
            ):
                selection_audit = no_selection_finalist_data_incomplete_audit
                no_selection_finalist_audit_status = "DATA_INCOMPLETE"
                guardrail_normalized = True
                decision = _decision_from_finalist_audit_pass(
                    original_decision=decision,
                    focus_ticker=no_selection_finalist_data_incomplete_focus,
                    selection_audit=selection_audit,
                )

    if (
        decision.selected_ticker
        and _audit_gap_is_repairable(selection_audit, allowed_tools or [])
        and provider is not None
        and isinstance(signal_packets, dict)
        and isinstance(allowed_tools, list)
        and budget is not None
        and str(decision.selected_ticker).upper() in packets_by_ticker
    ):
        require_run_scope_unchanged()
        selected_for_repair = str(decision.selected_ticker).upper()
        audit_gap_repair_attempted = True
        (
            repair_questions,
            repair_tool_calls,
            repair_evidence,
            repair_belief_updates,
            repair_degraded,
            repair_notes,
            repair_result_count,
            repair_metadata,
        ) = _run_audit_gap_repair_pass(
            provider=provider,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=objective,
            as_of_date=as_of_date,
            selected_ticker=selected_for_repair,
            selection_audit=selection_audit,
            signal_packets=signal_packets,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            tool_calls=tool_calls,
            evidence=evidence,
            allowed_tools=allowed_tools,
            budget=budget,
            executed_tool_count=(
                repair_lane_tool_count if separate_repair_lane else executed_tool_count
            ),
            question_start_index=len(research_questions) + 1,
            belief_update_start_index=len(belief_updates) + 1,
            separate_lane_budget=separate_repair_lane,
            pipeline_version=pipeline_version,
        )
        if separate_repair_lane:
            repair_lane_tool_count = repair_result_count
        else:
            executed_tool_count = repair_result_count
        research_questions.extend(repair_questions)
        tool_calls.extend(repair_tool_calls)
        evidence.extend(repair_evidence)
        belief_updates.extend(repair_belief_updates)
        states.extend(repair_degraded)
        audit_gap_repair_status = str(repair_metadata.get("status") or "UNKNOWN")
        audit_gap_repair_notes = [str(item) for item in repair_notes if str(item).strip()]
        notes.append(
            f"Audit-gap repair pass attempted for {selected_for_repair}: {audit_gap_repair_status}."
        )
        notes.extend(audit_gap_repair_notes)
        _mark_question_statuses(research_questions, tool_calls)
        selection_audit = _selection_audit_for_ticker(
            selected_ticker=selected_for_repair,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            tool_calls=tool_calls,
            evidence=evidence,
            degraded_states=states,
            framework=framework,
        )
        pre_audit_repair_status = audit_gap_repair_status
        if selection_audit.get("status") == "PASS":
            audit_gap_repair_status = "RESOLVED_SELECTED"
            audit_gap_repair_notes.append("Repair evidence cleared the binding audit blockers.")
        elif selection_audit.get("status") == "WATCHLIST_ONLY":
            audit_gap_repair_status = "PARTIALLY_RESOLVED_WATCHLIST"
            audit_gap_repair_notes.append(
                "Repair evidence cleared hard blockers but left watchlist-only caps."
            )
        elif pre_audit_repair_status in {
            "BUDGET_EXHAUSTED",
            "NO_REPAIR_TOOLS",
            "NO_ALLOWED_TOOLS",
            "PROVIDER_ERROR",
            "SKIPPED_BUDGET_EXHAUSTED",
        }:
            audit_gap_repair_status = pre_audit_repair_status
            audit_gap_repair_notes.append(
                "Audit-gap repair did not complete with decision-usable evidence."
            )
        else:
            audit_gap_repair_status = "UNRESOLVED_BLOCKED"
            audit_gap_repair_notes.append(
                "Repair evidence did not clear the binding audit blockers."
            )
        audit_gap_repair_target_statuses.append(
            {"ticker": selected_for_repair, "status": audit_gap_repair_status}
        )

    if (
        decision.selected_ticker
        and selection_audit.get("status") == "WATCHLIST_ONLY"
        and provider is not None
        and isinstance(signal_packets, dict)
        and isinstance(allowed_tools, list)
        and budget is not None
        and str(decision.selected_ticker).upper() in packets_by_ticker
    ):
        require_run_scope_unchanged()
        selected_for_resolution = str(decision.selected_ticker).upper()
        watchlist_resolution_attempted = True
        (
            resolution_questions,
            resolution_tool_calls,
            resolution_evidence,
            resolution_belief_updates,
            resolution_degraded,
            resolution_notes,
            repair_result_count,
            resolution_metadata,
        ) = _run_watchlist_resolution_pass(
            provider=provider,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=objective,
            as_of_date=as_of_date,
            selected_ticker=selected_for_resolution,
            selection_audit=selection_audit,
            signal_packets=signal_packets,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            tool_calls=tool_calls,
            evidence=evidence,
            allowed_tools=allowed_tools,
            budget=budget,
            executed_tool_count=(
                repair_lane_tool_count if separate_repair_lane else executed_tool_count
            ),
            question_start_index=len(research_questions) + 1,
            belief_update_start_index=len(belief_updates) + 1,
            separate_lane_budget=separate_repair_lane,
            pipeline_version=pipeline_version,
        )
        if separate_repair_lane:
            repair_lane_tool_count = repair_result_count
        else:
            executed_tool_count = repair_result_count
        research_questions.extend(resolution_questions)
        tool_calls.extend(resolution_tool_calls)
        evidence.extend(resolution_evidence)
        belief_updates.extend(resolution_belief_updates)
        states.extend(resolution_degraded)
        watchlist_resolution_status = str(resolution_metadata.get("status") or "UNKNOWN")
        watchlist_resolution_notes = [str(item) for item in resolution_notes if str(item).strip()]
        notes.append(
            f"Watchlist resolution pass attempted for {selected_for_resolution}: {watchlist_resolution_status}."
        )
        notes.extend(watchlist_resolution_notes)
        _mark_question_statuses(research_questions, tool_calls)
        selection_audit = _selection_audit_for_ticker(
            selected_ticker=selected_for_resolution,
            packets_by_ticker=packets_by_ticker,
            scenarios=scenarios,
            tool_calls=tool_calls,
            evidence=evidence,
            degraded_states=states,
            framework=framework,
        )
        pre_audit_resolution_status = watchlist_resolution_status
        if selection_audit.get("status") == "PASS":
            watchlist_resolution_status = "RESOLVED_SELECTED"
            watchlist_resolution_notes.append(
                "Follow-up evidence cleared the watchlist audit caps."
            )
        elif selection_audit.get("status") == "BLOCKED":
            watchlist_resolution_status = "RESOLVED_NO_SELECTION"
            watchlist_resolution_notes.append(
                "Follow-up evidence exposed binding selection blockers."
            )
        elif pre_audit_resolution_status in {
            "BUDGET_EXHAUSTED",
            "NO_FOLLOW_UP_TOOLS",
            "PROVIDER_ERROR",
            "SKIPPED_BUDGET_EXHAUSTED",
        }:
            watchlist_resolution_status = pre_audit_resolution_status
            watchlist_resolution_notes.append(
                "Watchlist resolution did not complete with decision-usable follow-up evidence."
            )
        else:
            watchlist_resolution_status = "UNRESOLVED_WATCHLIST"
            watchlist_resolution_notes.append(
                "Follow-up evidence did not clear all watchlist audit caps."
            )

    if decision.selected_ticker and selection_audit.get("status") == "BLOCKED":
        audit_blockers = [str(item) for item in selection_audit.get("hard_blockers") or []]
        states.append("SELECTION_AUDIT_BLOCKED")
        notes.append(
            "Selection audit found binding blockers after provider decision; converted to no-selection."
        )
        guardrail_normalized = True
        decision = SectorFinalDecision(
            verdict="NO_SELECTION",
            confidence=None,
            selected_ticker=None,
            expected_annualized_return_range=None,
            thesis=_deterministic_no_selection_thesis(
                focus_ticker=str(
                    selection_audit.get("selected_ticker") or decision.selected_ticker or "UNKNOWN"
                ),
                selection_audit=selection_audit,
            ),
            key_risk=_deterministic_no_selection_key_risk(selection_audit),
            downside_case=_deterministic_no_selection_downside(selection_audit),
            no_selection_reason=(
                f"No company selected because {selection_audit.get('selected_ticker')} failed selection audit: "
                f"{', '.join(audit_blockers)}."
            ),
            falsifiers=list(decision.falsifiers),
            why_selected_over_finalists=list(decision.why_selected_over_finalists),
            rejected_finalists=[dict(item) for item in decision.rejected_finalists],
            selection_blockers=["SELECTION_AUDIT_BLOCKED", *audit_blockers],
            confidence_cap_reasons=list(decision.confidence_cap_reasons),
            evidence_ref_ids=list(decision.evidence_ref_ids),
        )
        selection_audit = {
            **selection_audit,
            "actionable": False,
            "final_verdict_after_audit": "NO_SELECTION",
        }
    elif decision.selected_ticker and selection_audit.get("status") == "DATA_INCOMPLETE":
        data_resolution_needed = [
            str(item) for item in selection_audit.get("data_resolution_needed") or []
        ]
        states.append("SELECTION_AUDIT_DATA_INCOMPLETE")
        notes.append(
            "Selection audit found only fetchable data-availability gaps; selected candidate is "
            "data-incomplete pending resolution (resolve-then-promote)."
        )
        guardrail_normalized = True
        decision = SectorFinalDecision(
            verdict="DATA_INCOMPLETE",
            confidence=None,
            selected_ticker=decision.selected_ticker,
            expected_annualized_return_range=decision.expected_annualized_return_range,
            thesis=decision.thesis,
            key_risk=decision.key_risk,
            downside_case=decision.downside_case,
            no_selection_reason=None,
            falsifiers=list(decision.falsifiers),
            why_selected_over_finalists=list(decision.why_selected_over_finalists),
            rejected_finalists=[dict(item) for item in decision.rejected_finalists],
            selection_blockers=[],
            confidence_cap_reasons=list(decision.confidence_cap_reasons),
            evidence_ref_ids=list(decision.evidence_ref_ids),
            data_resolution_needed=data_resolution_needed,
        )
        selection_audit = {
            **selection_audit,
            "actionable": False,
            "final_verdict_after_audit": "DATA_INCOMPLETE",
        }
    elif decision.selected_ticker and selection_audit.get("status") == "WATCHLIST_ONLY":
        audit_caps = _mergeable_confidence_caps(selection_audit)
        states.append("SELECTION_AUDIT_WATCHLIST_ONLY")
        notes.append(
            "Selection audit found non-binding evidence caps; converted selected candidate to non-actionable WATCHLIST."
        )
        capped_confidence = _cap_confidence(
            decision.confidence, str(selection_audit.get("confidence_ceiling") or "MODERATE")
        )
        decision = SectorFinalDecision(
            verdict="WATCHLIST",
            confidence=capped_confidence,
            selected_ticker=decision.selected_ticker,
            expected_annualized_return_range=decision.expected_annualized_return_range,
            thesis=decision.thesis,
            key_risk=decision.key_risk,
            downside_case=decision.downside_case,
            no_selection_reason=None,
            falsifiers=list(decision.falsifiers),
            why_selected_over_finalists=list(decision.why_selected_over_finalists),
            rejected_finalists=[dict(item) for item in decision.rejected_finalists],
            selection_blockers=[],
            confidence_cap_reasons=list(
                dict.fromkeys([*decision.confidence_cap_reasons, *audit_caps])
            ),
            evidence_ref_ids=list(decision.evidence_ref_ids),
        )
        selection_audit = {
            **selection_audit,
            "actionable": False,
            "final_verdict_after_audit": "WATCHLIST",
        }
    else:
        selection_audit = {
            **selection_audit,
            "final_verdict_after_audit": decision.verdict or "NO_SELECTION",
        }
        if decision.selected_ticker and selection_audit.get("confidence_ceiling"):
            audit_business_caps = [
                str(item)
                for item in selection_audit.get("confidence_caps") or []
                if classify_audit_signal(str(item)) == "BUSINESS_QUALITY"
            ]
            decision.confidence = _cap_confidence(
                decision.confidence,
                str(selection_audit.get("confidence_ceiling")),
            )
            decision.confidence_cap_reasons = list(
                dict.fromkeys([*decision.confidence_cap_reasons, *audit_business_caps])
            )

    if decision.selected_ticker:
        child_verdicts = _company_autonomy_child_verdict_by_ticker(autonomy_runs)
        selected_child_verdict = child_verdicts.get(str(decision.selected_ticker).upper())
        decision, selection_audit, llm_ceiling_bound = _apply_llm_verdict_ceiling(
            decision=decision,
            selection_audit=selection_audit,
            llm_verdict=selected_child_verdict,
        )
        if llm_ceiling_bound:
            states.append("LLM_VERDICT_CEILING_BOUND")
            notes.append(
                "Company-level LLM verdict ceiling prevented deterministic audit from promoting the candidate."
            )
            guardrail_normalized = True

    selection_validation: SectorSelectionValidation | None = None
    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        selection_validation = _run_v2_selected_company_validation(
            sector=sector,
            ticker=decision.selected_ticker,
            as_of_date=as_of_date,
            allowed_tools=list(allowed_tools or []),
            company_packets=company_packets,
            scenarios=scenarios,
            signal_packets=dict(signal_packets or {}),
            expected_source_binding=(
                (competitive_frontier or {})
                .get("source_bindings", {})
                .get(str(decision.selected_ticker or "").upper())
                if isinstance((competitive_frontier or {}).get("source_bindings"), dict)
                else None
            ),
        )
        if selection_validation.status == "CONTRADICTED":
            states.append("SELECTED_COMPANY_VALIDATION_CONTRADICTED")
            notes.append(
                "The independent selected-company challenge contradicted the provisional selection."
            )
        elif selection_validation.status in {"NOT_ATTEMPTED", "INCOMPLETE"}:
            states.append("SELECTED_COMPANY_VALIDATION_INCOMPLETE")
            notes.extend(selection_validation.notes)

    if guardrail_normalized:
        notes.append(GUARDRAILS_BINDING_AUDIT_NOTE)
    final_verdict = decision.verdict or "NO_SELECTION"
    relative_ranking = _relative_ranking(
        company_packets=company_packets,
        scenarios=scenarios,
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=states,
        company_autonomy_runs=autonomy_runs,
        framework=framework,
    )
    company_autonomy_decision_trace = _company_autonomy_decision_trace(
        decision=decision,
        selection_audit=selection_audit,
        relative_ranking=relative_ranking,
        company_autonomy_attempted=company_autonomy_attempted,
        company_autonomy_status=company_autonomy_status,
        company_autonomy_runs=autonomy_runs,
        packets_by_ticker=packets_by_ticker,
        scenarios=scenarios,
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=states,
        framework=framework,
    )
    final_decision_prompt_context = _final_decision_prompt_context(
        company_packets=company_packets,
        scenarios=scenarios,
        relative_ranking=relative_ranking,
        tool_calls=tool_calls,
        evidence=evidence,
        audit_gap_repair_targets=audit_gap_repair_target_statuses,
    )
    require_run_scope_unchanged()
    return AutonomousSectorFinancialRunArtifact(
        run_id=run_id,
        sector=sector,
        market_cap_focus=market_cap_focus,
        objective=objective,
        as_of_date=as_of_date,
        created_at=created_at,
        completed_at=_utc_now_iso(),
        status="COMPLETED",
        final_verdict=final_verdict,
        selected_ticker=decision.selected_ticker,
        confidence=decision.confidence,
        candidate_selection=candidate_selection or {},
        candidate_dispositions=(
            _valuation_anchor_dispositions(candidate_selection)
            if pipeline_version != SECTOR_PIPELINE_VERSION_V2
            else []
        ),
        framework=framework,
        company_packets=company_packets,
        research_questions=research_questions,
        expected_return_scenarios=scenarios,
        tool_calls=tool_calls,
        evidence=evidence,
        belief_updates=belief_updates,
        final_decision=decision,
        selection_audit=selection_audit,
        framework_evidence_preflight=_artifact_framework_evidence_preflight(
            framework=framework,
            company_packets=company_packets,
            scenarios=scenarios,
            allowed_tools=allowed_tools,
        ),
        final_decision_prompt_context=final_decision_prompt_context,
        audit_gap_repair_attempted=audit_gap_repair_attempted,
        audit_gap_repair_status=audit_gap_repair_status,
        audit_gap_repair_notes=list(dict.fromkeys(audit_gap_repair_notes)),
        watchlist_resolution_attempted=watchlist_resolution_attempted,
        watchlist_resolution_status=watchlist_resolution_status,
        watchlist_resolution_notes=list(dict.fromkeys(watchlist_resolution_notes)),
        no_selection_finalist_audit_attempted=no_selection_finalist_audit_attempted,
        no_selection_finalist_audit_status=no_selection_finalist_audit_status,
        no_selection_finalist_audit_focus_ticker=no_selection_finalist_audit_focus_ticker,
        no_selection_finalist_audit_notes=list(dict.fromkeys(no_selection_finalist_audit_notes)),
        no_selection_finalist_resolution_attempted=no_selection_finalist_resolution_attempted,
        no_selection_finalist_resolution_status=no_selection_finalist_resolution_status,
        no_selection_finalist_resolution_notes=list(
            dict.fromkeys(no_selection_finalist_resolution_notes)
        ),
        alternate_finalist_audit_attempted=alternate_finalist_audit_attempted,
        alternate_finalist_audit_status=alternate_finalist_audit_status,
        alternate_finalist_audit_notes=list(dict.fromkeys(alternate_finalist_audit_notes)),
        alternate_finalist_audit_results=alternate_finalist_audit_results,
        company_autonomy_attempted=company_autonomy_attempted,
        company_autonomy_status=company_autonomy_status,
        company_autonomy_notes=list(dict.fromkeys(autonomy_notes)),
        company_autonomy_runs=autonomy_runs,
        company_autonomy_decision_trace=company_autonomy_decision_trace,
        competitive_frontier=dict(competitive_frontier or {}),
        selection_validation=selection_validation,
        relative_ranking=relative_ranking,
        no_selection_reason=decision.no_selection_reason,
        degraded_states=list(dict.fromkeys(states)),
        audit_notes=list(dict.fromkeys(notes)),
    )


def _validate_v2_frozen_execution_bound(
    *,
    tickers: list[str],
    as_of_date: str,
    budget: AutonomousRunBudget,
    candidate_selection: dict[str, Any] | None,
    terminal_cap_search: Any | None,
) -> None:
    if not isinstance(candidate_selection, dict) or not bool(
        candidate_selection.get("execution_bound_frozen")
    ):
        raise ValueError("v2 runtime requires a frozen execution-bound ledger")
    normalized = list(
        dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    )
    membership = _normalized_ticker_values(
        candidate_selection.get("membership_tickers") or candidate_selection.get("selected_tickers")
    )
    selected = _normalized_ticker_values(candidate_selection.get("selected_tickers"))
    execution = _normalized_ticker_values(candidate_selection.get("execution_tickers"))
    deferred = _normalized_ticker_values(candidate_selection.get("deferred_by_bound_tickers"))
    excluded = set(_normalized_ticker_values(candidate_selection.get("excluded_tickers")))
    if selected != membership or [*execution, *deferred] != membership:
        raise ValueError("v2 execution/deferred ledgers do not partition membership")
    if set(execution) & set(deferred) or excluded & set(membership):
        raise ValueError("v2 execution-bound ledgers overlap")
    if normalized != execution:
        raise ValueError("v2 runtime tickers do not match the frozen execution set")
    bound = candidate_selection.get("execution_bound")
    if bound is not None and len(execution) > int(bound):
        raise ValueError("v2 runtime execution set exceeds its candidate bound")
    if budget.max_candidates != bound:
        raise ValueError("v2 runtime budget candidate bound drifted after preflight")
    if str(candidate_selection.get("execution_as_of_date") or as_of_date) != str(as_of_date):
        raise ValueError("v2 runtime as-of date drifted after preflight")
    if str(candidate_selection.get("execution_fingerprint") or "").strip().lower() != (
        _v2_json_fingerprint(execution)
    ):
        raise ValueError("v2 runtime execution fingerprint drifted after preflight")
    if terminal_cap_search is not None:
        authorization = getattr(terminal_cap_search, "authorization", None)
        request_fingerprint = (
            str(candidate_selection.get("request_fingerprint") or "").strip().lower()
        )
        if (
            authorization is None
            or str(getattr(authorization, "request_fingerprint", "")).lower() != request_fingerprint
            or not set(execution).issubset(set(getattr(authorization, "allowed_tickers", ()) or ()))
        ):
            raise ValueError("terminal cap search authority drifted from execution set")


def _run_sector_autonomous_financial_analysis_impl(
    *,
    sector: str,
    tickers: list[str],
    objective: str | None = None,
    as_of_date: str | None = None,
    market_cap_focus: str = "small_cap",
    budget: AutonomousRunBudget | None = None,
    allowed_tools: list[str] | None = None,
    candidate_selection: dict[str, Any] | None = None,
    pipeline_version: str = "v1",
    terminal_cap_search: Any | None = None,
) -> AutonomousSectorFinancialRunArtifact:
    """Run an iterative autonomous financial analyst loop over sector candidates."""

    run_objective = objective or DEFAULT_SECTOR_OBJECTIVE
    run_as_of = as_of_date or date.today().isoformat()
    run_budget = _budget_or_default(budget)
    created_at = _utc_now_iso()
    run_id = _run_id(sector, created_at)
    normalized_tickers = list(
        dict.fromkeys(str(ticker).upper() for ticker in tickers if str(ticker).strip())
    )
    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        _validate_v2_frozen_execution_bound(
            tickers=normalized_tickers,
            as_of_date=run_as_of,
            budget=run_budget,
            candidate_selection=candidate_selection,
            terminal_cap_search=terminal_cap_search,
        )

    if not normalized_tickers:
        v2_empty_scope = pipeline_version == SECTOR_PIPELINE_VERSION_V2
        return _empty_artifact(
            run_id=run_id,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=run_objective,
            as_of_date=run_as_of,
            created_at=created_at,
            degraded_state=("NO_IN_SCOPE_TICKERS" if v2_empty_scope else "NO_TICKERS_PROVIDED"),
            reason=(
                "No securities were admitted after deterministic scope resolution."
                if v2_empty_scope
                else "No tickers were provided for the sector autonomous run."
            ),
            candidate_selection=candidate_selection,
            status="COMPLETED" if v2_empty_scope else "FAILED",
        )

    # Evidence auto-resolution (universe program Phase D): repair fetchable
    # packet inputs — stale/missing scorecards, uncached annual filings —
    # BEFORE assembly so candidates are reviewed instead of held on gaps
    # nobody tried to fetch. Deterministic, count-capped, no LLM spend.
    try:
        from app.autonomous.evidence_resolution import (
            V2ExecutionBoundDriftError,
            pre_assembly_data_gap_repair,
        )

        prior_repair = (
            candidate_selection.get("data_gap_repair")
            if isinstance(candidate_selection, dict)
            and isinstance(candidate_selection.get("data_gap_repair"), dict)
            else {}
        )
        data_gap_repair = pre_assembly_data_gap_repair(
            tickers=normalized_tickers,
            as_of_date=run_as_of,
            pipeline_version=pipeline_version,
            apply_repairs=pipeline_version == SECTOR_PIPELINE_VERSION_V2,
            checkpoint_path=(
                candidate_selection.get("data_gap_repair_checkpoint_path")
                if isinstance(candidate_selection, dict)
                else None
            )
            or prior_repair.get("checkpoint_path"),
            checkpoint_scope=f"{sector}:{market_cap_focus}",
            candidate_context=candidate_selection,
            run_id=run_id,
            evidence_revision=(
                candidate_selection.get("evidence_revision")
                if isinstance(candidate_selection, dict)
                else None
            ),
            candidate_context_revision=(
                candidate_selection.get("candidate_context_revision")
                if isinstance(candidate_selection, dict)
                else None
            ),
            terminal_cap_search=terminal_cap_search,
        )
    except V2ExecutionBoundDriftError:
        raise
    except Exception as exc:  # noqa: BLE001 - non-authority repair remains best-effort
        data_gap_repair = {"status": "REPAIR_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    if isinstance(candidate_selection, dict):
        candidate_selection = {**candidate_selection, "data_gap_repair": data_gap_repair}
    elif pipeline_version == "v2" or data_gap_repair.get("status") == "APPLIED":
        candidate_selection = {"data_gap_repair": data_gap_repair}

    try:
        if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
            classifications = (
                candidate_selection.get("cap_classifications")
                if isinstance(candidate_selection, dict)
                and isinstance(candidate_selection.get("cap_classifications"), dict)
                else {}
            )
            repair_states_by_ticker = {
                str(row.get("ticker") or "").upper(): row
                for row in data_gap_repair.get("candidate_states") or []
                if isinstance(row, dict) and str(row.get("ticker") or "").strip()
            }
            repaired_identity_by_ticker = {
                ticker: row.get("issuer_identity")
                for ticker, row in repair_states_by_ticker.items()
                if isinstance(row.get("issuer_identity"), dict)
            }
            repaired_price_by_ticker: dict[str, dict[str, Any]] = {}
            for ticker, row in repair_states_by_ticker.items():
                packet_inputs = (
                    row.get("packet_inputs") if isinstance(row.get("packet_inputs"), dict) else {}
                )
                price_input = (
                    packet_inputs.get("price")
                    if isinstance(packet_inputs.get("price"), dict)
                    else {}
                )
                snapshot = (
                    price_input.get("snapshot")
                    if isinstance(price_input.get("snapshot"), dict)
                    else {}
                )
                price = snapshot.get("price")
                if not isinstance(price, (int, float)):
                    price = price_input.get("price")
                if not isinstance(price, (int, float)):
                    continue
                repaired_price_by_ticker[ticker] = {
                    "current_price": float(price),
                    "current_price_as_of_date": (
                        snapshot.get("as_of_date") or price_input.get("as_of_date")
                    ),
                    "current_price_currency": (
                        snapshot.get("currency") or price_input.get("currency")
                    ),
                    "current_price_source": (
                        snapshot.get("source")
                        or price_input.get("provider")
                        or price_input.get("source")
                    ),
                    "current_price_source_url": (
                        snapshot.get("url") or price_input.get("source_url")
                    ),
                    "current_price_confidence": (
                        snapshot.get("confidence") or price_input.get("confidence")
                    ),
                }
            issuer_contexts = {
                ticker: {
                    **(
                        classifications.get(ticker, {})
                        if isinstance(classifications.get(ticker), dict)
                        else {}
                    ),
                    **repaired_identity_by_ticker.get(ticker, {}),
                    **repaired_price_by_ticker.get(ticker, {}),
                }
                for ticker in normalized_tickers
            }
            fixed_prices = {
                ticker: repaired_price_by_ticker.get(ticker, {}).get("current_price")
                for ticker in normalized_tickers
            }
            signal_packets = assemble_sector_packets(
                normalized_tickers,
                filing_risk_use_llm=False,
                as_of_date=run_as_of,
                pipeline_version=pipeline_version,
                current_prices=fixed_prices,
                issuer_contexts=issuer_contexts,
                db_path=get_config().db_path,
            )
        else:
            financial_context = build_canonical_v1_financial_context(
                tickers=normalized_tickers,
                as_of_date=run_as_of,
                db_path=get_config().db_path,
            )
            fixed_prices = financial_context.current_prices
            issuer_contexts = financial_context.issuer_contexts
            signal_packets = financial_context.packets
            candidate_selection = {
                **(candidate_selection or {}),
                "cap_classifications": issuer_contexts,
            }
    except Exception as exc:
        return _empty_artifact(
            run_id=run_id,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=run_objective,
            as_of_date=run_as_of,
            created_at=created_at,
            degraded_state="PACKET_ASSEMBLY_FAILED",
            reason=f"Could not assemble sector signal packets: {exc}",
            candidate_selection=candidate_selection,
        )

    company_packets = build_sector_company_financial_packets_from_signal_packets(
        signal_packets,
        sector=sector,
        as_of_date=run_as_of,
        cap_classifications=(
            issuer_contexts
            if pipeline_version == SECTOR_PIPELINE_VERSION_V2
            else candidate_selection.get("cap_classifications")
            if isinstance(candidate_selection, dict)
            else None
        ),
        pipeline_version=pipeline_version,
    )
    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        structural_gate_results = _v2_structural_gate_results(
            sector=sector,
            as_of_date=run_as_of,
            company_packets=company_packets,
        )
        candidate_selection = {
            **(candidate_selection or {}),
            "structural_gate_results": structural_gate_results,
        }
    packets_by_ticker = {packet.ticker: packet for packet in company_packets}
    scenarios_by_ticker = build_expected_return_scenarios_for_packets(company_packets)
    scenarios = [
        scenario
        for ticker in sorted(scenarios_by_ticker)
        for scenario in scenarios_by_ticker[ticker]
    ]
    run_allowed_tools = _allowed_tools(signal_packets, allowed_tools)

    if not company_packets:
        return _empty_artifact(
            run_id=run_id,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=run_objective,
            as_of_date=run_as_of,
            created_at=created_at,
            degraded_state="FINANCIAL_PACKET_DATA_INSUFFICIENT",
            reason="No company financial packets could be built for the sector run.",
            candidate_selection=candidate_selection,
        )

    (
        company_packets,
        scenarios,
        signal_packets,
        candidate_selection,
        financial_history_filter_states,
        financial_history_filter_notes,
    ) = _pre_provider_financial_history_filter(
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        candidate_selection=candidate_selection,
        run_as_of_date=run_as_of,
        pipeline_version=pipeline_version,
    )

    if not company_packets:
        return _empty_artifact(
            run_id=run_id,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=run_objective,
            as_of_date=run_as_of,
            created_at=created_at,
            degraded_state="FINANCIAL_HISTORY_DATA_INSUFFICIENT",
            reason="No sector candidates had enough cached FY history for a reportable financial table.",
            candidate_selection=candidate_selection,
        )

    (
        company_packets,
        scenarios,
        signal_packets,
        candidate_selection,
        pre_provider_filter_states,
        pre_provider_filter_notes,
    ) = _pre_provider_framework_evidence_filter(
        sector=sector,
        market_cap_focus=market_cap_focus,
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        allowed_tools=run_allowed_tools,
        candidate_selection=candidate_selection,
        pipeline_version=pipeline_version,
    )
    (
        company_packets,
        scenarios,
        signal_packets,
        candidate_selection,
        valuation_input_packets,
        valuation_input_scenarios,
    ) = _pre_provider_valuation_anchor_filter(
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        candidate_selection=candidate_selection,
    )
    valuation_filter = candidate_selection["valuation_anchor_filter"]
    missing_valuation_tickers = list(valuation_filter["needs_data_tickers"])
    if missing_valuation_tickers:
        pre_provider_filter_states.append("MISSING_VALUATION_NEEDS_DATA")
        pre_provider_filter_notes.append(
            "Pre-provider valuation-anchor gate withheld "
            f"{len(missing_valuation_tickers)} candidate(s) from paid reasoning: "
            f"{', '.join(missing_valuation_tickers)}."
        )
    if not company_packets:
        return _empty_artifact(
            run_id=run_id,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=run_objective,
            as_of_date=run_as_of,
            created_at=created_at,
            started_packets=valuation_input_packets,
            scenarios=valuation_input_scenarios,
            degraded_state="NEEDS_DATA",
            reason=(
                "Deterministic valuation-anchor gate stopped before any sector "
                "provider work: NEEDS_DATA: MISSING_VALUATION across all admitted companies."
            ),
            audit_notes=[
                "No provider call ran because every admitted company lacked a "
                "usable valuation anchor.",
                f"Withheld tickers: {', '.join(missing_valuation_tickers)}.",
            ],
            candidate_selection=candidate_selection,
        )
    integrity_scope = _financial_integrity_scope(
        context="autonomous_sector_pre_provider",
        as_of_date=run_as_of,
        packets=company_packets,
        scenarios=scenarios,
    )
    try:
        financial_integrity_result = _require_applied_financial_integrity_scope(integrity_scope)
    except InvalidFinancialInputError as exc:
        _apply_financial_integrity_result(
            packets=company_packets,
            scenarios=scenarios,
            result=exc.result,
        )
        candidate_selection = {
            **(candidate_selection or {}),
            "financial_integrity": exc.result.to_dict(),
        }
        return _empty_artifact(
            run_id=run_id,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=run_objective,
            as_of_date=run_as_of,
            created_at=created_at,
            started_packets=company_packets,
            scenarios=scenarios,
            degraded_state=exc.status,
            reason=(
                f"Deterministic financial inputs failed before any sector provider work: {exc}"
            ),
            audit_notes=[
                "No provider call or substantive fallback ran because the "
                f"financial-integrity gate returned {exc.status}.",
                json.dumps(exc.result.to_dict(), sort_keys=True, default=str),
            ],
            candidate_selection=candidate_selection,
        )
    candidate_selection = {
        **(candidate_selection or {}),
        "financial_integrity": _financial_integrity_result_payload(financial_integrity_result),
    }
    if pipeline_version == "v1":
        candidate_selection["financial_integrity_binding"] = _financial_integrity_run_binding(
            integrity_scope,
            scope_fingerprint=financial_integrity_result.scope_fingerprint,
        )

    def require_run_scope_unchanged() -> None:
        _require_candidate_financial_integrity_binding(
            candidate_selection,
            packets=company_packets,
            scenarios=scenarios,
        )

    provider = get_alpha_llm_provider()
    if not _provider_enabled(provider):
        require_run_scope_unchanged()
        return _empty_artifact(
            run_id=run_id,
            sector=sector,
            market_cap_focus=market_cap_focus,
            objective=run_objective,
            as_of_date=run_as_of,
            created_at=created_at,
            started_packets=company_packets,
            scenarios=scenarios,
            degraded_state="LLM_PROVIDER_UNAVAILABLE",
            reason="LLM provider is unavailable; sector autonomous run stopped without forcing a selection.",
            candidate_selection=candidate_selection,
        )
    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        deterministic_pre_rank = build_competitive_frontier(
            company_packets,
            scenarios,
            top_n=SECTOR_PROMPT_CANDIDATE_LIMIT,
            batch_size=3,
        )
        packets_by_rank = {packet.ticker.upper(): packet for packet in company_packets}
        company_packets = [packets_by_rank[row.ticker] for row in deterministic_pre_rank.candidates]
        candidate_selection = {
            **(candidate_selection or {}),
            "deterministic_pre_rank": deterministic_pre_rank.to_dict(),
            "deterministic_prompt_tickers": list(deterministic_pre_rank.top_tickers),
        }
    packets_by_ticker = {packet.ticker: packet for packet in company_packets}

    framework: SectorFinancialFramework | None = None
    research_questions: list[SectorResearchQuestion] = []
    tool_calls: list[ToolCallRecord] = []
    evidence: list[EvidenceReference] = []
    belief_updates: list[BeliefUpdate] = []
    degraded_states: list[str] = [*financial_history_filter_states, *pre_provider_filter_states]
    audit_notes: list[str] = [*financial_history_filter_notes, *pre_provider_filter_notes]
    executed_tool_count = 0
    executed_plan_keys: set[str] = set()
    company_autonomy_attempted = False
    company_autonomy_status: str | None = None
    company_autonomy_notes: list[str] = []
    company_autonomy_runs: list[dict[str, Any]] = []
    competitive_frontier_payload: dict[str, Any] = {}

    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        (
            competitive_frontier_payload,
            frontier_runs,
            frontier_evidence,
            frontier_notes,
        ) = _run_v2_competitive_frontier(
            sector=sector,
            as_of_date=run_as_of,
            company_packets=company_packets,
            scenarios=scenarios,
            candidate_selection=candidate_selection,
            signal_packets=signal_packets,
            allowed_tools=run_allowed_tools,
            evidence_start_index=len(evidence) + 1,
        )
        company_autonomy_attempted = True
        company_autonomy_status = str(competitive_frontier_payload.get("status") or "OPEN")
        company_autonomy_notes.extend(frontier_notes)
        company_autonomy_runs.extend(frontier_runs)
        evidence.extend(frontier_evidence)
        audit_notes.extend(frontier_notes)
        candidate_selection = {
            **(candidate_selection or {}),
            "competitive_frontier": competitive_frontier_payload,
        }

    def _company_autonomy_kwargs() -> dict[str, Any]:
        return {
            "company_autonomy_attempted": company_autonomy_attempted,
            "company_autonomy_status": company_autonomy_status,
            "company_autonomy_notes": company_autonomy_notes,
            "company_autonomy_runs": company_autonomy_runs,
            "competitive_frontier": competitive_frontier_payload,
            "pipeline_version": pipeline_version,
        }

    def current_sector_prompt_state() -> dict[str, Any]:
        return {
            "sector": sector,
            "market_cap_focus": market_cap_focus,
            "objective": run_objective,
            "as_of_date": run_as_of,
            "budget": run_budget.to_dict(),
            "allowed_tools": run_allowed_tools,
            "company_packets": company_packets,
            "scenarios": scenarios,
            "framework": framework,
            "research_questions": research_questions,
            "tool_calls": tool_calls,
            "evidence": evidence,
            "belief_updates": belief_updates,
        }

    def revalidate_prompt_scope(
        scope: FinancialIntegrityScope,
        expected_fingerprint: str,
    ) -> Any:
        return require_unchanged_financial_integrity_scope(
            scope,
            expected_scope_fingerprint=expected_fingerprint,
        )

    prompt_state_epoch = _freeze_financial_prompt_state(
        scope=integrity_scope,
        expected_scope_fingerprint=financial_integrity_result.scope_fingerprint,
        state_getter=current_sector_prompt_state,
        scope_revalidator=revalidate_prompt_scope,
    )

    for turn_index in range(1, max(1, int(run_budget.max_turns)) + 1):
        require_run_scope_unchanged()
        try:
            if (
                turn_index == 1
                and not research_questions
                and not tool_calls
                and (not evidence or pipeline_version == SECTOR_PIPELINE_VERSION_V2)
            ):
                initial_prompt = _initial_plan_prompt(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=run_objective,
                    as_of_date=run_as_of,
                    budget=run_budget,
                    allowed_tools=run_allowed_tools,
                    company_packets=company_packets,
                    scenarios=scenarios,
                )
                initial_request = {
                    "prompt": initial_prompt,
                    "schema": _INITIAL_PLAN_SCHEMA,
                    "schema_name": "autonomous_sector_initial_research_plan",
                    "max_output_tokens": 6000,
                }
                payload = _normalize_initial_plan_payload(
                    _synthesize_provider_json(
                        provider,
                        integrity_scope=_financial_integrity_scope(
                            context="autonomous_sector_initial_research_plan",
                            as_of_date=run_as_of,
                            packets=company_packets,
                            scenarios=scenarios,
                        ),
                        _financial_prompt_integrity_binding=prompt_state_epoch.bind_request(
                            provider,
                            initial_request,
                        ),
                        **initial_request,
                    ),
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                )
            else:
                turn_prompt = _turn_prompt(
                    turn_index=turn_index,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=run_objective,
                    as_of_date=run_as_of,
                    budget=run_budget,
                    allowed_tools=run_allowed_tools,
                    company_packets=company_packets,
                    scenarios=scenarios,
                    framework=framework,
                    research_questions=research_questions,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    belief_updates=belief_updates,
                )
                turn_request = {
                    "prompt": turn_prompt,
                    "schema": _TURN_SCHEMA,
                    "schema_name": "autonomous_sector_financial_turn",
                    "max_output_tokens": 5000,
                }
                payload = _synthesize_provider_json(
                    provider,
                    integrity_scope=_financial_integrity_scope(
                        context="autonomous_sector_financial_turn",
                        as_of_date=run_as_of,
                        packets=company_packets,
                        scenarios=scenarios,
                    ),
                    _financial_prompt_integrity_binding=prompt_state_epoch.bind_request(
                        provider,
                        turn_request,
                    ),
                    **turn_request,
                )
        except InvalidFinancialInputError:
            raise
        except Exception as exc:
            if isinstance(exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
                raise
            failure_state = _provider_failure_state(exc)
            recovery_notes = [
                f"Provider structured-output failure on sector turn {turn_index} ({failure_state}): {exc}",
            ]
            first_turn_recovered = False
            if turn_index == 1 and not tool_calls and not evidence and not research_questions:
                try:
                    require_run_scope_unchanged()
                    payload = _recover_initial_plan_after_provider_error(
                        provider=provider,
                        sector=sector,
                        market_cap_focus=market_cap_focus,
                        objective=run_objective,
                        as_of_date=run_as_of,
                        allowed_tools=run_allowed_tools,
                        company_packets=company_packets,
                        scenarios=scenarios,
                        provider_error=exc,
                        prompt_state_epoch=prompt_state_epoch,
                    )
                    _append_unique(degraded_states, failure_state, "LLM_PROVIDER_TURN_RECOVERED")
                    audit_notes.extend(
                        recovery_notes
                        + [
                            "Recovered from first-turn provider failure with compact minimum tool-plan request."
                        ]
                    )
                    first_turn_recovered = True
                except InvalidFinancialInputError:
                    raise
                except Exception as recovery_exc:  # noqa: BLE001
                    if isinstance(recovery_exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
                        raise
                    recovery_state = _provider_failure_state(recovery_exc)
                    failure_states: list[str] = []
                    _append_unique(failure_states, failure_state, recovery_state)
                    return AutonomousSectorFinancialRunArtifact(
                        run_id=run_id,
                        sector=sector,
                        market_cap_focus=market_cap_focus,
                        objective=run_objective,
                        as_of_date=run_as_of,
                        created_at=created_at,
                        completed_at=_utc_now_iso(),
                        status="FAILED",
                        final_verdict="NO_SELECTION",
                        selected_ticker=None,
                        confidence=None,
                        candidate_selection=candidate_selection or {},
                        framework=framework,
                        company_packets=company_packets,
                        research_questions=research_questions,
                        expected_return_scenarios=scenarios,
                        tool_calls=tool_calls,
                        evidence=evidence,
                        belief_updates=belief_updates,
                        company_autonomy_attempted=company_autonomy_attempted,
                        company_autonomy_status=company_autonomy_status,
                        company_autonomy_notes=company_autonomy_notes,
                        company_autonomy_runs=company_autonomy_runs,
                        relative_ranking=_relative_ranking(
                            company_packets=company_packets,
                            scenarios=scenarios,
                            tool_calls=tool_calls,
                            evidence=evidence,
                            degraded_states=list(dict.fromkeys(degraded_states + failure_states)),
                            company_autonomy_runs=company_autonomy_runs,
                            framework=framework,
                        ),
                        no_selection_reason=(
                            f"LLM provider failed during first sector planning turn: {exc}; "
                            f"compact recovery also failed: {recovery_exc}"
                        ),
                        degraded_states=list(dict.fromkeys(degraded_states + failure_states)),
                        audit_notes=audit_notes
                        + recovery_notes
                        + [
                            f"Compact first-turn tool-plan recovery also failed ({recovery_state}): {recovery_exc}"
                        ]
                        + ["Stopped before tool execution without forcing a sector selection."],
                    )
            elif tool_calls or evidence or research_questions:
                try:
                    require_run_scope_unchanged()
                    recovered_decision = _recover_final_decision_after_provider_error(
                        provider=provider,
                        sector=sector,
                        market_cap_focus=market_cap_focus,
                        objective=run_objective,
                        as_of_date=run_as_of,
                        company_packets=company_packets,
                        scenarios=scenarios,
                        research_questions=research_questions,
                        tool_calls=tool_calls,
                        evidence=evidence,
                        belief_updates=belief_updates,
                        provider_error=exc,
                        prompt_state_epoch=prompt_state_epoch,
                    )
                    _mark_question_statuses(research_questions, tool_calls)
                    return _artifact_from_decision(
                        run_id=run_id,
                        sector=sector,
                        market_cap_focus=market_cap_focus,
                        objective=run_objective,
                        as_of_date=run_as_of,
                        created_at=created_at,
                        framework=framework,
                        company_packets=company_packets,
                        research_questions=research_questions,
                        scenarios=scenarios,
                        tool_calls=tool_calls,
                        evidence=evidence,
                        belief_updates=belief_updates,
                        final_decision=recovered_decision,
                        degraded_states=degraded_states
                        + [failure_state, "LLM_PROVIDER_JSON_RECOVERED"],
                        audit_notes=audit_notes
                        + recovery_notes
                        + [
                            "Recovered from provider structured-output failure with a compact final-decision-only request."
                        ],
                        candidate_selection=candidate_selection,
                        provider=provider,
                        signal_packets=signal_packets,
                        allowed_tools=run_allowed_tools,
                        budget=run_budget,
                        executed_tool_count=executed_tool_count,
                        **_company_autonomy_kwargs(),
                    )
                except InvalidFinancialInputError:
                    raise
                except Exception as recovery_exc:  # noqa: BLE001
                    if isinstance(recovery_exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
                        raise
                    recovery_state = _provider_failure_state(recovery_exc)
                    recovery_notes.append(
                        f"Compact final-decision recovery also failed: {recovery_exc}"
                    )
                    has_deterministic_evidence = bool(evidence) or any(
                        call.status == "OK" for call in tool_calls
                    )
                    if has_deterministic_evidence and LLM_PROVIDER_QUOTA_EXHAUSTED not in {
                        failure_state,
                        recovery_state,
                    }:
                        _mark_question_statuses(research_questions, tool_calls)
                        fallback_decision = _deterministic_finalization_fallback_decision(
                            provider_error=exc,
                            recovery_error=recovery_exc,
                            company_packets=company_packets,
                            scenarios=scenarios,
                            research_questions=research_questions,
                            evidence=evidence,
                        )
                        return _artifact_from_decision(
                            run_id=run_id,
                            sector=sector,
                            market_cap_focus=market_cap_focus,
                            objective=run_objective,
                            as_of_date=run_as_of,
                            created_at=created_at,
                            framework=framework,
                            company_packets=company_packets,
                            research_questions=research_questions,
                            scenarios=scenarios,
                            tool_calls=tool_calls,
                            evidence=evidence,
                            belief_updates=belief_updates,
                            final_decision=fallback_decision,
                            degraded_states=degraded_states
                            + [
                                failure_state,
                                recovery_state,
                                "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
                            ],
                            audit_notes=audit_notes
                            + recovery_notes
                            + [
                                "Deterministic finalization fallback ran after provider final-decision failure."
                            ],
                            candidate_selection=candidate_selection,
                            provider=None,
                            signal_packets=signal_packets,
                            allowed_tools=run_allowed_tools,
                            budget=run_budget,
                            executed_tool_count=executed_tool_count,
                            **_company_autonomy_kwargs(),
                        )
            else:
                recovery_notes.append(
                    "No prior evidence existed, and first-turn recovery was not eligible."
                )
            if not first_turn_recovered:
                return AutonomousSectorFinancialRunArtifact(
                    run_id=run_id,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=run_objective,
                    as_of_date=run_as_of,
                    created_at=created_at,
                    completed_at=_utc_now_iso(),
                    status="FAILED",
                    final_verdict="NO_SELECTION",
                    selected_ticker=None,
                    confidence=None,
                    candidate_selection=candidate_selection or {},
                    framework=framework,
                    company_packets=company_packets,
                    research_questions=research_questions,
                    expected_return_scenarios=scenarios,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    belief_updates=belief_updates,
                    company_autonomy_attempted=company_autonomy_attempted,
                    company_autonomy_status=company_autonomy_status,
                    company_autonomy_notes=company_autonomy_notes,
                    company_autonomy_runs=company_autonomy_runs,
                    relative_ranking=_relative_ranking(
                        company_packets=company_packets,
                        scenarios=scenarios,
                        tool_calls=tool_calls,
                        evidence=evidence,
                        degraded_states=list(dict.fromkeys(degraded_states + [failure_state])),
                        company_autonomy_runs=company_autonomy_runs,
                        framework=framework,
                    ),
                    no_selection_reason=f"LLM provider failed during sector turn {turn_index}: {exc}",
                    degraded_states=list(dict.fromkeys(degraded_states + [failure_state])),
                    audit_notes=audit_notes
                    + recovery_notes
                    + [
                        "Stopped without forcing a sector selection because provider synthesis failed."
                    ],
                )

        if framework is None:
            framework = _framework_from_payload(
                payload, sector=sector, market_cap_focus=market_cap_focus
            )
        new_questions, planned_calls = _questions_from_payload(payload, turn_index)
        research_questions.extend(new_questions)
        belief_updates.extend(_belief_updates_from_payload(payload, len(belief_updates) + 1))
        provider_degraded, provider_degraded_notes = _split_provider_degraded_states(
            payload.get("degraded_states") or []
        )
        degraded_states.extend(provider_degraded)
        audit_notes.extend(provider_degraded_notes)
        audit_notes.extend(str(item) for item in payload.get("audit_notes") or [])
        fresh_planned_calls: list[dict[str, Any]] = []
        duplicate_planned_call_count = 0
        for planned in planned_calls:
            key = _planned_call_key(planned, packets_by_ticker)
            if key in executed_plan_keys:
                duplicate_planned_call_count += 1
                continue
            fresh_planned_calls.append(planned)
        planned_calls = fresh_planned_calls
        if duplicate_planned_call_count:
            degraded_states.append("DUPLICATE_PLANNED_TOOLS_SKIPPED")
            audit_notes.append(
                f"Skipped {duplicate_planned_call_count} duplicate planned tool call(s) already executed earlier in the run."
            )

        final_decision = _final_decision_from_payload(payload)
        continue_research = bool(payload.get("continue_research"))
        if final_decision is not None and not continue_research and planned_calls:
            degraded_states.append("FINAL_DECISION_WITH_UNEXECUTED_PLAN")
            audit_notes.append(
                "Provider returned a final decision while also planning deterministic tool calls; "
                "runtime executed the planned tools before accepting a final verdict."
            )
        elif final_decision is not None and not continue_research:
            _mark_question_statuses(research_questions, tool_calls)
            return _artifact_from_decision(
                run_id=run_id,
                sector=sector,
                market_cap_focus=market_cap_focus,
                objective=run_objective,
                as_of_date=run_as_of,
                created_at=created_at,
                framework=framework,
                company_packets=company_packets,
                research_questions=research_questions,
                scenarios=scenarios,
                tool_calls=tool_calls,
                evidence=evidence,
                belief_updates=belief_updates,
                final_decision=final_decision,
                degraded_states=degraded_states,
                audit_notes=audit_notes,
                candidate_selection=candidate_selection,
                provider=provider,
                signal_packets=signal_packets,
                allowed_tools=run_allowed_tools,
                budget=run_budget,
                executed_tool_count=executed_tool_count,
                **_company_autonomy_kwargs(),
            )

        if (
            not planned_calls
            and turn_index == 1
            and not tool_calls
            and (not evidence or pipeline_version == SECTOR_PIPELINE_VERSION_V2)
            and "INITIAL_FINAL_DECISION_IGNORED" not in degraded_states
        ):
            fallback_questions, fallback_calls, fallback_notes = _deterministic_initial_tool_plan(
                framework=framework,
                company_packets=company_packets,
                scenarios=scenarios,
                allowed_tools=run_allowed_tools,
            )
            if fallback_calls:
                research_questions.extend(fallback_questions)
                planned_calls = fallback_calls
                degraded_states.append("DETERMINISTIC_INITIAL_PLAN_FALLBACK")
            audit_notes.extend(fallback_notes)

        if not planned_calls:
            decision = SectorFinalDecision(
                verdict="NO_SELECTION",
                confidence=None,
                selected_ticker=None,
                expected_annualized_return_range=None,
                thesis="No sector selection was made because the provider did not plan further evidence or return a final decision.",
                key_risk="Stopping without a decision is safer than inventing a selection.",
                downside_case="No underwritten downside case was finalized.",
                no_selection_reason="No research plan or final decision was returned.",
                selection_blockers=["NO_RESEARCH_PLAN"],
            )
            return _artifact_from_decision(
                run_id=run_id,
                sector=sector,
                market_cap_focus=market_cap_focus,
                objective=run_objective,
                as_of_date=run_as_of,
                created_at=created_at,
                framework=framework,
                company_packets=company_packets,
                research_questions=research_questions,
                scenarios=scenarios,
                tool_calls=tool_calls,
                evidence=evidence,
                belief_updates=belief_updates,
                final_decision=decision,
                degraded_states=degraded_states + ["NO_RESEARCH_PLAN"],
                audit_notes=audit_notes,
                candidate_selection=candidate_selection,
                provider=provider,
                signal_packets=signal_packets,
                allowed_tools=run_allowed_tools,
                budget=run_budget,
                executed_tool_count=executed_tool_count,
                **_company_autonomy_kwargs(),
            )

        records, new_evidence, turn_degraded, executed_tool_count, budget_exhausted = (
            _execute_planned_calls(
                sector=sector,
                as_of_date=run_as_of,
                allowed_tools=run_allowed_tools,
                budget=run_budget,
                planned_calls=planned_calls,
                signal_packets=signal_packets,
                packets_by_ticker=packets_by_ticker,
                scenarios=scenarios,
                call_start_index=len(tool_calls) + 1,
                executed_so_far=executed_tool_count,
                evidence_start_index=len(evidence) + 1,
            )
        )
        tool_calls.extend(records)
        evidence.extend(new_evidence)
        for planned, record in zip(planned_calls, records, strict=False):
            if record.status != "SKIPPED_BUDGET_EXHAUSTED":
                executed_plan_keys.add(_planned_call_key(planned, packets_by_ticker))
        degraded_states.extend(turn_degraded)
        _mark_question_statuses(research_questions, tool_calls)

        if turn_index == 1 and not company_autonomy_attempted and not budget_exhausted:
            require_run_scope_unchanged()
            (
                autonomy_attempted,
                autonomy_status,
                autonomy_notes,
                autonomy_runs,
                autonomy_evidence,
                executed_tool_count,
            ) = _run_company_autonomy_pass(
                sector=sector,
                as_of_date=run_as_of,
                research_questions=research_questions,
                company_packets=company_packets,
                scenarios=scenarios,
                signal_packets=signal_packets,
                candidate_selection=candidate_selection,
                allowed_tools=run_allowed_tools,
                budget=run_budget,
                executed_tool_count=executed_tool_count,
                evidence_start_index=len(evidence) + 1,
            )
            if autonomy_attempted:
                company_autonomy_attempted = True
                company_autonomy_status = autonomy_status
                company_autonomy_notes.extend(autonomy_notes)
                company_autonomy_runs.extend(autonomy_runs)
                evidence.extend(autonomy_evidence)
                audit_notes.extend(autonomy_notes)
            require_run_scope_unchanged()

        prompt_state_epoch = _freeze_financial_prompt_state(
            scope=integrity_scope,
            expected_scope_fingerprint=financial_integrity_result.scope_fingerprint,
            state_getter=current_sector_prompt_state,
            scope_revalidator=revalidate_prompt_scope,
        )

        if budget_exhausted:
            budget_error = RuntimeError(
                "Tool-call budget exhausted after executing available evidence."
            )
            budget_notes = [
                "Tool-call budget exhausted; requesting compact final decision without additional tools."
            ]
            try:
                require_run_scope_unchanged()
                recovered_decision = _recover_final_decision_after_provider_error(
                    provider=provider,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=run_objective,
                    as_of_date=run_as_of,
                    company_packets=company_packets,
                    scenarios=scenarios,
                    research_questions=research_questions,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    belief_updates=belief_updates,
                    provider_error=budget_error,
                    prompt_state_epoch=prompt_state_epoch,
                )
                return _artifact_from_decision(
                    run_id=run_id,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=run_objective,
                    as_of_date=run_as_of,
                    created_at=created_at,
                    framework=framework,
                    company_packets=company_packets,
                    research_questions=research_questions,
                    scenarios=scenarios,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    belief_updates=belief_updates,
                    final_decision=recovered_decision,
                    degraded_states=degraded_states + ["BUDGET_EXHAUSTED"],
                    audit_notes=audit_notes + budget_notes,
                    candidate_selection=candidate_selection,
                    provider=provider,
                    signal_packets=signal_packets,
                    allowed_tools=run_allowed_tools,
                    budget=run_budget,
                    executed_tool_count=executed_tool_count,
                    **_company_autonomy_kwargs(),
                )
            except InvalidFinancialInputError:
                raise
            except Exception as recovery_exc:  # noqa: BLE001
                if isinstance(recovery_exc, (LLMRetryBudgetExceeded, LLMCostBudgetExceeded)):
                    raise
                fallback_decision = _deterministic_finalization_fallback_decision(
                    provider_error=budget_error,
                    recovery_error=recovery_exc,
                    company_packets=company_packets,
                    scenarios=scenarios,
                    research_questions=research_questions,
                    evidence=evidence,
                )
                return _artifact_from_decision(
                    run_id=run_id,
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=run_objective,
                    as_of_date=run_as_of,
                    created_at=created_at,
                    framework=framework,
                    company_packets=company_packets,
                    research_questions=research_questions,
                    scenarios=scenarios,
                    tool_calls=tool_calls,
                    evidence=evidence,
                    belief_updates=belief_updates,
                    final_decision=fallback_decision,
                    degraded_states=degraded_states
                    + ["BUDGET_EXHAUSTED", "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE"],
                    audit_notes=audit_notes
                    + budget_notes
                    + [
                        f"Compact final-decision recovery after tool-budget exhaustion failed: {recovery_exc}",
                        "Deterministic finalization fallback ran after tool-budget exhaustion.",
                    ],
                    candidate_selection=candidate_selection,
                    provider=None,
                    signal_packets=signal_packets,
                    allowed_tools=run_allowed_tools,
                    budget=run_budget,
                    executed_tool_count=executed_tool_count,
                    **_company_autonomy_kwargs(),
                )

    decision = SectorFinalDecision(
        verdict="NO_SELECTION",
        confidence=None,
        selected_ticker=None,
        expected_annualized_return_range=None,
        thesis="No sector selection was made before the autonomous turn budget expired.",
        key_risk="Forcing a winner after turn exhaustion would exceed the run's operating envelope.",
        downside_case="No final downside case was completed before turn exhaustion.",
        no_selection_reason="LLM turn budget exhausted before final sector decision.",
        selection_blockers=["TURN_BUDGET_EXHAUSTED"],
    )
    return _artifact_from_decision(
        run_id=run_id,
        sector=sector,
        market_cap_focus=market_cap_focus,
        objective=run_objective,
        as_of_date=run_as_of,
        created_at=created_at,
        framework=framework,
        company_packets=company_packets,
        research_questions=research_questions,
        scenarios=scenarios,
        tool_calls=tool_calls,
        evidence=evidence,
        belief_updates=belief_updates,
        final_decision=decision,
        degraded_states=degraded_states + ["TURN_BUDGET_EXHAUSTED"],
        audit_notes=audit_notes
        + ["Stopped without forcing a selection because LLM turn budget was exhausted."],
        candidate_selection=candidate_selection,
        provider=provider,
        signal_packets=signal_packets,
        allowed_tools=run_allowed_tools,
        budget=run_budget,
        executed_tool_count=executed_tool_count,
        **_company_autonomy_kwargs(),
    )


def _normalized_ticker_values(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return list(dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip()))


def _v2_discovered_tickers(
    *,
    admitted_tickers: list[str],
    candidate_selection: dict[str, Any],
) -> list[str]:
    """Return the stable security-slot denominator for a v2 artifact."""

    discovered: list[str] = []
    cap_classifications = candidate_selection.get("cap_classifications")
    if isinstance(cap_classifications, dict):
        discovered.extend(str(ticker).strip().upper() for ticker in cap_classifications)
    for field_name in (
        "requested_tickers",
        "loaded_tickers",
        "excluded_tickers",
        "selected_tickers_before_financial_history_filter",
        "selected_tickers_before_framework_filter",
        "selected_tickers",
        "membership_tickers",
        "execution_tickers",
        "deferred_by_bound_tickers",
    ):
        discovered.extend(_normalized_ticker_values(candidate_selection.get(field_name)))
    discovered.extend(admitted_tickers)
    return list(dict.fromkeys(ticker for ticker in discovered if ticker))


def _v2_missing_packet_reasons(
    ticker: str,
    candidate_selection: dict[str, Any],
    *,
    execution_tickers: set[str],
) -> list[str]:
    reasons: list[str] = []
    financial_filter = candidate_selection.get("financial_history_filter")
    if isinstance(financial_filter, dict) and ticker in _normalized_ticker_values(
        [
            *_normalized_ticker_values(financial_filter.get("excluded_tickers")),
            *_normalized_ticker_values(financial_filter.get("needs_data_tickers")),
        ]
    ):
        reasons.append("SPARSE_FINANCIAL_HISTORY")
    framework_filter = candidate_selection.get("framework_evidence_filter")
    if isinstance(framework_filter, dict) and ticker in _normalized_ticker_values(
        [
            *_normalized_ticker_values(framework_filter.get("excluded_tickers")),
            *_normalized_ticker_values(framework_filter.get("needs_data_tickers")),
        ]
    ):
        reasons.append("FRAMEWORK_EVIDENCE_MISSING")
    valuation_filter = candidate_selection.get("valuation_anchor_filter")
    if isinstance(valuation_filter, dict) and ticker in _normalized_ticker_values(
        valuation_filter.get("needs_data_tickers")
    ):
        reasons.append("MISSING_VALUATION")
    if ticker not in execution_tickers:
        reasons.append("EXECUTION_DEFERRED_NOT_RUN")
    if not reasons:
        reasons.append("PACKET_NOT_ASSEMBLED")
    return reasons


def _v2_foreign_facts_gap_reason(
    packet: SectorCompanyFinancialPacket,
) -> str | None:
    accounting_quality = (
        packet.accounting_quality if isinstance(packet.accounting_quality, dict) else {}
    )
    signals = accounting_quality.get("filing_risk_signals")
    if not isinstance(signals, dict):
        return None
    reason = str(signals.get("foreign_facts_gap_reason") or "").strip().upper()
    if reason in {
        "IFRS_FACTS_UNSUPPORTED",
        "NON_USD_FACTS_UNNORMALIZED",
        "FOREIGN_NORMALIZED_FACTS_UNAVAILABLE",
    }:
        return reason
    return None


def _v2_data_repair_gap(
    ticker: str,
    candidate_selection: dict[str, Any],
) -> tuple[list[str], str] | None:
    repair = candidate_selection.get("data_gap_repair")
    if not isinstance(repair, dict):
        return None
    if str(repair.get("status") or "").upper() == "REPAIR_ERROR":
        error = str(repair.get("error") or "UNKNOWN").strip()
        return ["DATA_REPAIR_ERROR", f"DATA_REPAIR_ERROR_DETAIL:{error}"], "DATA_REPAIR"
    states = repair.get("candidate_states")
    if not isinstance(states, list):
        return None
    state = next(
        (
            row
            for row in states
            if isinstance(row, dict)
            and str(row.get("ticker") or "").strip().upper() == ticker.upper()
        ),
        None,
    )
    if not isinstance(state, dict):
        if int(repair.get("examined") or 0) > 0:
            return ["DATA_REPAIR_STATE_MISSING"], "DATA_REPAIR"
        return None
    queue_status = str(state.get("queue_status") or "").strip().upper()
    last_completed = str(state.get("last_completed_stage") or "DATA_REPAIR")
    stage_states = state.get("stage_states")
    stage_states = stage_states if isinstance(stage_states, dict) else {}
    if queue_status != "COMPLETED":
        reasons = [f"DATA_REPAIR_{queue_status or 'INCOMPLETE'}"]
        reasons.extend(
            f"DATA_REPAIR_STAGE_{str(stage).upper()}_{str(values.get('status') or 'UNKNOWN').upper()}"
            for stage, values in stage_states.items()
            if isinstance(values, dict)
            and str(values.get("status") or "").upper() in {"FAILED", "INTERRUPTED", "NEEDS_RETRY"}
        )
        reasons.extend(
            str(values.get("reason_code"))
            for values in stage_states.values()
            if isinstance(values, dict)
            and str(values.get("status") or "").upper() in {"FAILED", "INTERRUPTED", "NEEDS_RETRY"}
            and str(values.get("reason_code") or "").strip()
        )
        return list(dict.fromkeys(reasons)), last_completed
    packet_outcome = str(state.get("packet_readiness_outcome") or "").strip().upper()
    if packet_outcome == "NEEDS_DATA":

        def _stage_reason(stage: str, fallback: str) -> str:
            values = stage_states.get(stage)
            reason = values.get("reason_code") if isinstance(values, dict) else None
            return str(reason or fallback)

        filing_reasons = [_stage_reason("FILINGS", "NO_READABLE_ANNUAL_FILING")]
        parsing_values = stage_states.get("PARSING")
        if (
            isinstance(parsing_values, dict)
            and str(parsing_values.get("reason_code") or "").strip()
        ):
            filing_reasons.append(str(parsing_values["reason_code"]))
        reason_by_input: dict[str, list[str]] = {
            "IDENTITY": [_stage_reason("IDENTITY", "MISSING_ISSUER_IDENTITY")],
            "CAP": [_stage_reason("CAP", "MISSING_MARKET_CAP")],
            "FACTS": [_stage_reason("FACTS_AVAILABILITY", "MISSING_NORMALIZED_FACTS")],
            # Preserve the causal order. An identity/source failure in FILINGS
            # must not be hidden by the downstream parser's generic record gap.
            "FILING": filing_reasons,
            "PRICE": [_stage_reason("PRICE", "MISSING_PRICE")],
            "VALUATION": [_stage_reason("VALUATION", "MISSING_VALUATION")],
        }
        missing = [
            str(item).strip().upper()
            for item in state.get("packet_missing_inputs") or []
            if str(item).strip()
        ]
        return (
            list(
                dict.fromkeys(
                    reason
                    for item in missing
                    for reason in reason_by_input.get(item, [f"MISSING_{item}"])
                )
            )
            or ["PACKET_READINESS_INCOMPLETE"],
            last_completed,
        )
    return None


def _v2_retained_financial_history_gap(
    ticker: str,
    candidate_selection: dict[str, Any],
) -> bool:
    financial_filter = candidate_selection.get("financial_history_filter")
    retained = isinstance(financial_filter, dict) and ticker in _normalized_ticker_values(
        financial_filter.get("needs_data_tickers")
    )
    if not retained:
        return False
    repair = candidate_selection.get("data_gap_repair")
    states = repair.get("candidate_states") if isinstance(repair, dict) else None
    state = next(
        (
            row
            for row in states or []
            if isinstance(row, dict)
            and str(row.get("ticker") or "").strip().upper() == ticker.upper()
        ),
        None,
    )
    return not (
        isinstance(state, dict)
        and str(state.get("queue_status") or "").strip().upper() == "COMPLETED"
        and str(state.get("packet_readiness_outcome") or "").strip().upper()
        in {"READY", "COMPLETED"}
    )


def _v2_framework_prefilter_needs_evidence(
    ticker: str,
    candidate_selection: dict[str, Any],
) -> bool:
    framework_filter = candidate_selection.get("framework_evidence_filter")
    return bool(
        isinstance(framework_filter, dict)
        and ticker in _normalized_ticker_values(framework_filter.get("needs_data_tickers"))
    )


def _v2_cap_identity(
    ticker: str,
    candidate_selection: dict[str, Any],
) -> dict[str, Any]:
    classifications = candidate_selection.get("cap_classifications")
    row = classifications.get(ticker) if isinstance(classifications, dict) else None
    row = row if isinstance(row, dict) else {}
    repair = candidate_selection.get("data_gap_repair")
    repair_states = repair.get("candidate_states") if isinstance(repair, dict) else None
    repair_state = next(
        (
            item
            for item in repair_states or []
            if isinstance(item, dict)
            and str(item.get("ticker") or "").strip().upper() == ticker.upper()
        ),
        {},
    )
    repaired_identity = (
        repair_state.get("issuer_identity")
        if isinstance(repair_state, dict) and isinstance(repair_state.get("issuer_identity"), dict)
        else {}
    )
    return {
        "issuer_key": repaired_identity.get("issuer_cik")
        or row.get("issuer_key")
        or row.get("issuer_cik")
        or row.get("cik"),
        "issuer_cik": repaired_identity.get("issuer_cik")
        or row.get("issuer_cik")
        or row.get("cik"),
        "primary_ticker": repaired_identity.get("issuer_primary_ticker")
        or row.get("issuer_primary_ticker")
        or row.get("primary_ticker"),
        "security_type": repaired_identity.get("security_role")
        or row.get("security_role")
        or row.get("security_type"),
        "is_secondary_class": repaired_identity.get(
            "is_secondary_class", row.get("is_secondary_class")
        ),
        "is_adr": repaired_identity.get("is_adr", row.get("is_adr")),
        "adr_ratio": repaired_identity.get("adr_ratio", row.get("adr_ratio")),
        "share_class_ratio": repaired_identity.get(
            "share_class_ratio", row.get("share_class_ratio")
        ),
        "identity_source_url": repaired_identity.get("identity_source_url")
        or row.get("identity_source_url"),
        "ratio_source_url": repaired_identity.get("ratio_source_url")
        or row.get("ratio_source_url"),
    }


def _v2_child_evidence_ids(child: dict[str, Any]) -> list[str]:
    nested = child.get("artifact")
    if not isinstance(nested, dict):
        return []
    evidence_rows = nested.get("evidence")
    if not isinstance(evidence_rows, list):
        return []
    run_id = str(child.get("run_id") or "UNIDENTIFIED_CHILD").strip()
    return list(
        dict.fromkeys(
            f"{run_id}:{row.get('evidence_id')}"
            for row in evidence_rows
            if isinstance(row, dict) and row.get("evidence_id")
        )
    )


def _v2_child_decision_evidence_ids(child: dict[str, Any]) -> list[str]:
    nested = child.get("artifact")
    if not isinstance(nested, dict):
        return []
    ticker = str(child.get("ticker") or "").strip().upper()
    decision = next(
        (
            row
            for row in nested.get("candidate_decisions") or []
            if isinstance(row, dict)
            and (not ticker or str(row.get("ticker") or "").strip().upper() == ticker)
        ),
        None,
    )
    if not isinstance(decision, dict):
        return []
    linked = {
        str(item).strip() for item in decision.get("evidence_ref_ids") or [] if str(item).strip()
    }
    successful_tool_ids = set(_v2_child_ok_tool_ids(child))
    usable = {
        str(row.get("evidence_id"))
        for row in nested.get("evidence") or []
        if isinstance(row, dict)
        and str(row.get("evidence_id") or "") in linked
        and str(row.get("confidence") or "").strip().upper() in {"MODERATE", "HIGH"}
        and str(row.get("tool_call_id") or "").strip() in successful_tool_ids
    }
    run_id = str(child.get("run_id") or "UNIDENTIFIED_CHILD").strip()
    return [f"{run_id}:{item}" for item in sorted(usable)]


def _v2_child_ok_tool_ids(child: dict[str, Any]) -> list[str]:
    nested = child.get("artifact")
    if not isinstance(nested, dict):
        return []
    return [
        str(item.get("call_id"))
        for item in nested.get("tool_calls") or []
        if isinstance(item, dict)
        and item.get("call_id")
        and str(item.get("status") or "").strip().upper() == "OK"
    ]


def _v2_normalized_underwriting_verdict(value: Any) -> str | None:
    verdict = str(value or "").strip().upper()
    aliases = {
        "SELECTED": "ACTIONABLE",
        "WATCHLIST": "WATCHLIST_ONLY",
        "NO_SELECTION": "NO_WINNER",
    }
    normalized = aliases.get(verdict, verdict)
    return normalized or None


def _v2_incomplete_screen_result(
    contract_id: str,
    reason_codes: list[str],
    *,
    rule_id: str = "SCREEN_DATA_AVAILABILITY",
) -> ScreenResult:
    reasons = list(
        dict.fromkeys(str(item).strip().upper() for item in reason_codes if str(item).strip())
    ) or ["SCREEN_DATA_INCOMPLETE"]
    return ScreenResult(
        contract_id=contract_id,
        status="INCOMPLETE",
        gate_evaluations=[
            GateEvaluation(
                contract_id=contract_id,
                rule_id=rule_id,
                status="INCOMPLETE",
                applicable=True,
                observed_value=None,
                threshold=None,
                reason_code=reasons[0],
                notes=reasons[1:],
            )
        ],
        reason_codes=reasons,
    )


def _v2_pass_screen_result(
    contract_id: str,
    *,
    rule_id: str = "SECTOR_SCREEN",
    evidence_ref_id: str | None = None,
) -> ScreenResult:
    return ScreenResult(
        contract_id=contract_id,
        status="PASS",
        gate_evaluations=[
            GateEvaluation(
                contract_id=contract_id,
                rule_id=rule_id,
                status="PASS",
                applicable=True,
                observed_value="PASS",
                threshold="PASS",
                evidence_ref_id=evidence_ref_id or f"screen:{contract_id}:{rule_id.lower()}:pass",
            )
        ],
    )


def _v2_screen_result_from_gate_row(
    gate_row: dict[str, Any] | None,
    *,
    contract_id: str,
) -> ScreenResult:
    nested = gate_row.get("screen_result") if isinstance(gate_row, dict) else None
    if isinstance(nested, dict):
        try:
            result = ScreenResult.from_dict(nested)
        except (KeyError, TypeError, ValueError) as exc:
            return _v2_incomplete_screen_result(
                contract_id,
                ["SCREEN_RESULT_INVALID", f"SCREEN_RESULT_INVALID:{type(exc).__name__}"],
                rule_id="SCREEN_RESULT_CONTRACT",
            )
        if result.contract_id != contract_id:
            return _v2_incomplete_screen_result(
                contract_id,
                ["SCREEN_CONTRACT_MISMATCH"],
                rule_id="SCREEN_RESULT_CONTRACT",
            )
        return result
    return _v2_incomplete_screen_result(
        contract_id,
        ["SCREEN_NOT_EVALUATED"],
        rule_id="SCREEN_EXECUTION",
    )


def _v2_screen_state(
    row: dict[str, Any],
    *,
    screen_result: ScreenResult | None = None,
) -> tuple[str, str, list[str]]:
    """Project only an explicit deterministic screen result into terminal state.

    ``row`` is retained for a stable private call signature, but legacy
    relative-ranking audit blockers no longer decide v2 screen state.  Those
    signals belong to underwriting; absent deterministic gate evidence is a
    visible data gap.
    """

    _ = row
    result = screen_result or _v2_incomplete_screen_result(
        "UNRECORDED_V2_SCREEN",
        ["SCREEN_NOT_EVALUATED"],
        rule_id="SCREEN_EXECUTION",
    )
    reasons = list(result.reason_codes)
    if not reasons:
        reasons = list(
            dict.fromkeys(
                item.reason_code or item.rule_id
                for item in result.gate_evaluations
                if item.status in {"FAIL", "INCOMPLETE"}
            )
        )
    if result.status == "PASS":
        return "PASS", "READY_FOR_UNDERWRITING", []
    if result.status == "FAIL":
        return "FAIL", "SCREENED_OUT", reasons or ["SECTOR_GATE_FAILED"]
    return "INCOMPLETE", "NEEDS_DATA", reasons or ["SCREEN_NOT_COMPLETED"]


_V2_CHILD_INCOMPLETE_STATES = frozenset(
    {
        "BUDGET_EXHAUSTED",
        "TURN_BUDGET_EXHAUSTED",
        "LLM_PROVIDER_UNAVAILABLE",
        "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
        "LLM_COST_BUDGET_EXCEEDED",
        "LLM_RETRY_BUDGET_EXCEEDED",
        "DISALLOWED_TOOL_SKIPPED",
        "TOOL_EXECUTION_ERROR",
    }
)


def _v2_child_underwriting_gap_reasons(
    child: dict[str, Any],
    *,
    verdict: str | None,
) -> list[str]:
    reasons: list[str] = []
    if str(child.get("status") or "").strip().upper() != "COMPLETED":
        reasons.append("UNDERWRITING_FAILED")
    degraded = {
        str(item).strip().upper()
        for item in child.get("degraded_states") or []
        if str(item).strip()
    }
    if degraded & _V2_CHILD_INCOMPLETE_STATES or any(
        item.startswith("LLM_PROVIDER_") and not item.endswith("_RECOVERED") for item in degraded
    ):
        reasons.append("UNDERWRITING_RUNTIME_INCOMPLETE")
    if not _v2_child_ok_tool_ids(child):
        reasons.append("UNDERWRITING_TOOL_OUTPUT_MISSING")
    if not _v2_child_decision_evidence_ids(child):
        reasons.append("UNDERWRITING_DECISION_EVIDENCE_MISSING")
    if verdict in {None, "DATA_INCOMPLETE", "NO_WINNER"}:
        reasons.append(
            "UNDERWRITING_DATA_INCOMPLETE"
            if verdict == "DATA_INCOMPLETE"
            else "UNDERWRITING_NO_DECISION"
            if verdict == "NO_WINNER"
            else "UNDERWRITING_VERDICT_MISSING"
        )
    if not str(child.get("run_id") or "").strip():
        reasons.append("UNDERWRITING_RUN_ID_MISSING")
    return list(dict.fromkeys(reasons))


def _v2_underwriting_source_binding_gap_reasons(
    *,
    child: dict[str, Any],
    expected_source_binding: dict[str, Any] | None,
    ticker: str,
    sector: str,
    as_of_date: str,
    expected_cohort_tickers: list[str],
    expected_frontier_tickers: list[str],
    company_packet: SectorCompanyFinancialPacket,
) -> list[str]:
    """Return precise reasons when a completed child is not canonically bound."""

    if not isinstance(expected_source_binding, dict):
        return ["UNDERWRITING_FRONTIER_SOURCE_BINDING_MISSING"]
    if (
        expected_source_binding.get("artifact_type") != "v2_canonical_child_source_binding_v1"
        or expected_source_binding.get("pipeline_version") != "v2"
        or str(expected_source_binding.get("ticker") or "").strip().upper() != ticker
        or str(expected_source_binding.get("sector") or "").strip().lower()
        != str(sector or "").strip().lower()
        or str(expected_source_binding.get("as_of_date") or "") != as_of_date
        or expected_source_binding.get("cohort_tickers") != expected_cohort_tickers
        or expected_source_binding.get("frontier_candidate_tickers") != expected_frontier_tickers
        or expected_source_binding.get("company_packet_fingerprint")
        != _v2_json_fingerprint(company_packet.to_dict())
        or not str(expected_source_binding.get("signal_packet_fingerprint") or "").strip()
        or not str(expected_source_binding.get("cohort_fingerprint") or "").strip()
    ):
        return ["UNDERWRITING_FRONTIER_SOURCE_BINDING_INVALID"]

    reasons: list[str] = []
    child_binding = child.get("source_binding")
    if not isinstance(child_binding, dict):
        reasons.append("UNDERWRITING_CHILD_SOURCE_BINDING_MISSING")
    elif child_binding != expected_source_binding:
        reasons.append("UNDERWRITING_CHILD_SOURCE_BINDING_MISMATCH")

    nested_artifacts: list[dict[str, Any]] = []
    raw_artifact = child.get("artifact")
    if isinstance(raw_artifact, dict):
        nested_artifacts.append(raw_artifact)
    attempts = child.get("attempts")
    if isinstance(attempts, list):
        nested_artifacts.extend(attempt for attempt in attempts if isinstance(attempt, dict))
    if not nested_artifacts:
        reasons.append("UNDERWRITING_NESTED_ARTIFACT_MISSING")
    for nested in nested_artifacts:
        request = nested.get("request")
        candidate_scope = request.get("candidate_scope") if isinstance(request, dict) else None
        nested_binding = (
            candidate_scope.get("source_binding") if isinstance(candidate_scope, dict) else None
        )
        if not isinstance(nested_binding, dict):
            reasons.append("UNDERWRITING_NESTED_SOURCE_BINDING_MISSING")
        elif nested_binding != expected_source_binding:
            reasons.append("UNDERWRITING_NESTED_SOURCE_BINDING_MISMATCH")
    return list(dict.fromkeys(reasons))


def _v2_underwriting_result(
    *,
    child: dict[str, Any] | None,
    verdict: str | None,
    screen_status: str,
    expected_source_binding: dict[str, Any] | None,
    ticker: str,
    sector: str,
    as_of_date: str,
    expected_cohort_tickers: list[str],
    expected_frontier_tickers: list[str],
    company_packet: SectorCompanyFinancialPacket,
) -> UnderwritingResult:
    if screen_status == "FAIL":
        return UnderwritingResult(status="NOT_REQUIRED", reason_codes=["SCREEN_FAILED"])
    if not child:
        return UnderwritingResult(status="NOT_STARTED")
    reasons = _v2_child_underwriting_gap_reasons(child, verdict=verdict)
    if not reasons:
        reasons.extend(
            _v2_underwriting_source_binding_gap_reasons(
                child=child,
                expected_source_binding=expected_source_binding,
                ticker=ticker,
                sector=sector,
                as_of_date=as_of_date,
                expected_cohort_tickers=expected_cohort_tickers,
                expected_frontier_tickers=expected_frontier_tickers,
                company_packet=company_packet,
            )
        )
    evidence_ids = _v2_child_decision_evidence_ids(child)
    tool_ids = _v2_child_ok_tool_ids(child)
    if reasons:
        return UnderwritingResult(
            status=(
                "FAILED"
                if str(child.get("status") or "").strip().upper() != "COMPLETED"
                else "INCOMPLETE"
            ),
            reason_codes=reasons,
            evidence_ref_ids=evidence_ids,
            tool_call_ids=tool_ids,
            child_run_id=child.get("run_id"),
        )
    return UnderwritingResult(
        status="COMPLETED",
        verdict=verdict,
        confidence=child.get("confidence"),
        evidence_ref_ids=evidence_ids,
        tool_call_ids=tool_ids,
        child_run_id=child.get("run_id"),
    )


def _v2_candidate_dispositions(
    *,
    artifact: AutonomousSectorFinancialRunArtifact,
    admitted_tickers: list[str],
    execution_tickers: list[str],
) -> list[CandidateDisposition]:
    selection = dict(artifact.candidate_selection or {})
    admitted_set = set(admitted_tickers)
    discovered = _v2_discovered_tickers(
        admitted_tickers=admitted_tickers,
        candidate_selection=selection,
    )
    packets = {str(packet.ticker).upper(): packet for packet in artifact.company_packets}
    rankings = {
        str(row.get("ticker") or "").upper(): row
        for row in artifact.relative_ranking
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    children = _company_autonomy_run_by_ticker(artifact.company_autonomy_runs)
    execution_set = set(execution_tickers)
    delta_audit = selection.get("delta_audit")
    carried_verdicts = (
        delta_audit.get("carried_verdicts")
        if isinstance(delta_audit, dict) and isinstance(delta_audit.get("carried_verdicts"), dict)
        else {}
    )
    provisional_selected = str(
        artifact.selected_ticker
        or (artifact.final_decision.selected_ticker if artifact.final_decision else None)
        or ""
    ).upper()
    classifications = selection.get("cap_classifications")
    structural_gate_results = selection.get("structural_gate_results")
    screen_contract_id = (
        str(artifact.framework.framework_contract_id).strip()
        if artifact.framework is not None and artifact.framework.framework_contract_id
        else sector_framework_contract_id(artifact.sector, pipeline_version="v2")
    )
    frontier = (
        artifact.competitive_frontier if isinstance(artifact.competitive_frontier, dict) else {}
    )
    frontier_source_bindings = (
        frontier.get("source_bindings") if isinstance(frontier.get("source_bindings"), dict) else {}
    )
    frontier_candidate_tickers = [
        str(item.get("ticker") or "").strip().upper()
        for item in frontier.get("candidates") or []
        if isinstance(item, dict) and str(item.get("ticker") or "").strip()
    ]
    cohort_tickers = [str(packet.ticker).strip().upper() for packet in artifact.company_packets]
    frontier_tickers = set(_normalized_ticker_values(frontier.get("frontier_tickers")))
    dominated_tickers = set(_normalized_ticker_values(frontier.get("dominated_tickers")))
    unresolved_frontier_tickers = set(_normalized_ticker_values(frontier.get("unresolved_tickers")))
    dominators_by_ticker = (
        frontier.get("dominators_by_ticker")
        if isinstance(frontier.get("dominators_by_ticker"), dict)
        else {}
    )

    def frontier_fields(
        ticker: str,
        *,
        terminal_state: str,
        review_status: str,
    ) -> dict[str, Any]:
        if terminal_state in {"OUT_OF_SCOPE", "DEFERRED_BY_BOUND", "SCREENED_OUT"}:
            return {"frontier_status": "NOT_ELIGIBLE"}
        if terminal_state == "NEEDS_DATA":
            return {"frontier_status": "UNRESOLVED"}
        if review_status == "COMPLETED" or terminal_state == "UNDERWRITTEN":
            return {"frontier_status": "REVIEWED"}
        if ticker in dominated_tickers:
            return {
                "frontier_status": "DOMINATED",
                "frontier_dominated_by": _normalized_ticker_values(
                    dominators_by_ticker.get(ticker)
                ),
            }
        if ticker in unresolved_frontier_tickers:
            return {"frontier_status": "UNRESOLVED"}
        if ticker in frontier_tickers or terminal_state == "READY_FOR_UNDERWRITING":
            return {"frontier_status": "LIVE"}
        return {"frontier_status": "NOT_ELIGIBLE"}

    dispositions: list[CandidateDisposition] = []
    for ticker in discovered:
        identity = _v2_cap_identity(ticker, selection)
        if ticker not in admitted_set:
            cap_row = classifications.get(ticker) if isinstance(classifications, dict) else None
            reason = None
            if isinstance(cap_row, dict):
                reason = (
                    cap_row.get("scope_reason")
                    or cap_row.get("reason_code")
                    or cap_row.get("reason")
                    or cap_row.get("terminal_status")
                )
            if isinstance(delta_audit, dict) and ticker in _normalized_ticker_values(
                delta_audit.get("swept_excluded")
            ):
                reason = "DELTA_ALREADY_SWEPT_OUTSIDE_CURRENT_ATTEMPT"
            dispositions.append(
                CandidateDisposition(
                    ticker=ticker,
                    terminal_state="OUT_OF_SCOPE",
                    scope_status="OUT_OF_SCOPE",
                    screen_status="NOT_RUN",
                    review_status="NOT_REQUIRED",
                    reason_codes=[str(reason or "NOT_ADMITTED_TO_ANALYSIS")],
                    last_completed_stage="SCOPE_RESOLUTION",
                    screen_result=ScreenResult(
                        contract_id=screen_contract_id,
                        status="NOT_RUN",
                        reason_codes=[str(reason or "NOT_ADMITTED_TO_ANALYSIS")],
                    ),
                    underwriting_result=UnderwritingResult(
                        status="NOT_REQUIRED",
                        reason_codes=["OUT_OF_SCOPE"],
                    ),
                    **frontier_fields(
                        ticker,
                        terminal_state="OUT_OF_SCOPE",
                        review_status="NOT_REQUIRED",
                    ),
                    **identity,
                )
            )
            continue

        if bool(selection.get("execution_bound_frozen")) and ticker not in execution_set:
            dispositions.append(
                CandidateDisposition(
                    ticker=ticker,
                    terminal_state="DEFERRED_BY_BOUND",
                    scope_status="IN_SCOPE",
                    screen_status="NOT_RUN",
                    review_status="NOT_REQUIRED",
                    watchlist_eligible=False,
                    reason_codes=["DEFERRED_BY_EXECUTION_BOUND"],
                    last_completed_stage="EXECUTION_BOUND",
                    screen_result=ScreenResult(
                        contract_id=screen_contract_id,
                        status="NOT_RUN",
                        reason_codes=["DEFERRED_BY_EXECUTION_BOUND"],
                    ),
                    underwriting_result=UnderwritingResult(
                        status="NOT_REQUIRED",
                        reason_codes=["DEFERRED_BY_EXECUTION_BOUND"],
                    ),
                    **frontier_fields(
                        ticker,
                        terminal_state="DEFERRED_BY_BOUND",
                        review_status="NOT_REQUIRED",
                    ),
                    **identity,
                )
            )
            continue

        gate_row = (
            structural_gate_results.get(ticker)
            if isinstance(structural_gate_results, dict)
            else None
        )
        screen_result = _v2_screen_result_from_gate_row(
            gate_row,
            contract_id=screen_contract_id,
        )
        screen_status, screen_terminal_state, screen_reasons = _v2_screen_state(
            {},
            screen_result=screen_result,
        )
        if screen_terminal_state == "SCREENED_OUT":
            dispositions.append(
                CandidateDisposition(
                    ticker=ticker,
                    terminal_state="SCREENED_OUT",
                    scope_status="IN_SCOPE",
                    screen_status="FAIL",
                    review_status="NOT_REQUIRED",
                    watchlist_eligible=False,
                    reason_codes=screen_reasons,
                    evidence_ref_ids=list(screen_result.evidence_ref_ids),
                    last_completed_stage="SCREENING",
                    screen_result=screen_result,
                    underwriting_result=UnderwritingResult(
                        status="NOT_REQUIRED",
                        reason_codes=["SCREEN_FAILED"],
                    ),
                    **frontier_fields(
                        ticker,
                        terminal_state="SCREENED_OUT",
                        review_status="NOT_REQUIRED",
                    ),
                    **identity,
                )
            )
            continue

        repair_gap = _v2_data_repair_gap(ticker, selection)
        if repair_gap is not None:
            repair_reasons, repair_stage = repair_gap
            repair_screen = screen_result
            dispositions.append(
                CandidateDisposition(
                    ticker=ticker,
                    terminal_state="NEEDS_DATA",
                    scope_status="IN_SCOPE",
                    screen_status=repair_screen.status,
                    review_status="NOT_STARTED",
                    watchlist_eligible=ticker == provisional_selected,
                    reason_codes=list(
                        dict.fromkeys([*repair_reasons, *repair_screen.reason_codes])
                    ),
                    last_completed_stage=repair_stage,
                    screen_result=repair_screen,
                    underwriting_result=UnderwritingResult(status="NOT_STARTED"),
                    **frontier_fields(
                        ticker,
                        terminal_state="NEEDS_DATA",
                        review_status="NOT_STARTED",
                    ),
                    **identity,
                )
            )
            continue

        if _v2_retained_financial_history_gap(ticker, selection):
            dispositions.append(
                CandidateDisposition(
                    ticker=ticker,
                    terminal_state="NEEDS_DATA",
                    scope_status="IN_SCOPE",
                    screen_status=screen_result.status,
                    review_status="NOT_STARTED",
                    watchlist_eligible=ticker == provisional_selected,
                    reason_codes=[
                        "SPARSE_FINANCIAL_HISTORY",
                        *screen_result.reason_codes,
                    ],
                    evidence_ref_ids=list(screen_result.evidence_ref_ids),
                    last_completed_stage="FACTS_AVAILABILITY",
                    screen_result=screen_result,
                    underwriting_result=UnderwritingResult(status="NOT_STARTED"),
                    **frontier_fields(
                        ticker,
                        terminal_state="NEEDS_DATA",
                        review_status="NOT_STARTED",
                    ),
                    **identity,
                )
            )
            continue

        carried = carried_verdicts.get(ticker)
        if isinstance(carried, dict):
            carried_verdict = _v2_normalized_underwriting_verdict(carried.get("verdict"))
            carried_result_data = carried.get("underwriting_result")
            try:
                carried_result = (
                    UnderwritingResult.from_dict(carried_result_data)
                    if isinstance(carried_result_data, dict)
                    else None
                )
            except (KeyError, TypeError, ValueError):
                carried_result = None
            carried_complete = (
                screen_status == "PASS"
                and carried_result is not None
                and carried_result.status == "COMPLETED"
                and carried_result.verdict == carried_verdict
            )
            if carried_complete:
                terminal_state = "UNDERWRITTEN"
                review_status = "COMPLETED"
                reason_codes = [
                    "CARRIED_PRIOR_UNDERWRITING",
                    f"SOURCE_RUN:{carried_result.child_run_id}",
                ]
                underwriting_verdict = carried_verdict
            else:
                terminal_state = "NEEDS_DATA"
                review_status = "INCOMPLETE"
                reason_codes = [
                    "CARRIED_UNDERWRITING_EVIDENCE_UNAVAILABLE",
                    *([f"SOURCE_RUN:{carried.get('run_id')}"] if carried.get("run_id") else []),
                ]
                underwriting_verdict = None
                carried_result = UnderwritingResult(
                    status="INCOMPLETE",
                    reason_codes=reason_codes,
                    child_run_id=carried.get("run_id"),
                )
            dispositions.append(
                CandidateDisposition(
                    ticker=ticker,
                    terminal_state=terminal_state,
                    scope_status="IN_SCOPE",
                    screen_status=screen_status,
                    review_status=review_status,
                    underwriting_verdict=underwriting_verdict,
                    underwriting_confidence=(
                        carried_result.confidence if carried_complete else None
                    ),
                    watchlist_eligible=(
                        carried_complete and carried_verdict in {"ACTIONABLE", "WATCHLIST_ONLY"}
                    )
                    or (not carried_complete and ticker == provisional_selected),
                    reason_codes=reason_codes,
                    evidence_ref_ids=list(carried_result.evidence_ref_ids),
                    last_completed_stage="UNDERWRITING",
                    screen_result=screen_result,
                    underwriting_result=carried_result,
                    **frontier_fields(
                        ticker,
                        terminal_state=terminal_state,
                        review_status=review_status,
                    ),
                    **identity,
                )
            )
            continue

        if ticker not in packets:
            missing_reasons = _v2_missing_packet_reasons(
                ticker,
                selection,
                execution_tickers=execution_set,
            )
            incomplete_screen = _v2_incomplete_screen_result(
                screen_contract_id,
                missing_reasons,
                rule_id="COMPANY_PACKET_READINESS",
            )
            dispositions.append(
                CandidateDisposition(
                    ticker=ticker,
                    terminal_state="NEEDS_DATA",
                    scope_status="IN_SCOPE",
                    screen_status="INCOMPLETE",
                    review_status="NOT_STARTED",
                    watchlist_eligible=ticker == provisional_selected,
                    reason_codes=missing_reasons,
                    last_completed_stage="DATA_REPAIR",
                    screen_result=incomplete_screen,
                    underwriting_result=UnderwritingResult(status="NOT_STARTED"),
                    **frontier_fields(
                        ticker,
                        terminal_state="NEEDS_DATA",
                        review_status="NOT_STARTED",
                    ),
                    **identity,
                )
            )
            continue

        foreign_facts_gap = _v2_foreign_facts_gap_reason(packets[ticker])
        if foreign_facts_gap is not None:
            dispositions.append(
                CandidateDisposition(
                    ticker=ticker,
                    terminal_state="NEEDS_DATA",
                    scope_status="IN_SCOPE",
                    screen_status=screen_result.status,
                    review_status="NOT_STARTED",
                    watchlist_eligible=ticker == provisional_selected,
                    reason_codes=list(
                        dict.fromkeys([foreign_facts_gap, *screen_result.reason_codes])
                    ),
                    last_completed_stage="FACTS_AVAILABILITY",
                    screen_result=screen_result,
                    underwriting_result=UnderwritingResult(status="NOT_STARTED"),
                    **frontier_fields(
                        ticker,
                        terminal_state="NEEDS_DATA",
                        review_status="NOT_STARTED",
                    ),
                    **identity,
                )
            )
            continue

        row = rankings.get(ticker, {})
        child = children.get(ticker, {})
        child_verdict = _v2_normalized_underwriting_verdict(child.get("final_verdict"))
        screen_status, terminal_state, reason_codes = _v2_screen_state(
            row,
            screen_result=screen_result,
        )
        underwriting_result = _v2_underwriting_result(
            child=child or None,
            verdict=child_verdict,
            screen_status=screen_status,
            expected_source_binding=frontier_source_bindings.get(ticker),
            ticker=ticker,
            sector=artifact.sector,
            as_of_date=artifact.as_of_date,
            expected_cohort_tickers=cohort_tickers,
            expected_frontier_tickers=frontier_candidate_tickers,
            company_packet=packets[ticker],
        )
        if screen_status == "INCOMPLETE":
            terminal_state = "NEEDS_DATA"
            review_status = "NOT_STARTED"
            underwriting_verdict = None
            reason_codes = list(screen_reasons)
            underwriting_result = UnderwritingResult(status="NOT_STARTED")
        elif underwriting_result.status == "COMPLETED":
            terminal_state = "UNDERWRITTEN"
            review_status = "COMPLETED"
            underwriting_verdict = underwriting_result.verdict
            reason_codes = []
        elif underwriting_result.status == "NOT_STARTED":
            terminal_state = (
                "NEEDS_DATA"
                if _v2_framework_prefilter_needs_evidence(ticker, selection)
                else "READY_FOR_UNDERWRITING"
            )
            review_status = "NOT_STARTED"
            underwriting_verdict = None
            reason_codes = ["FRAMEWORK_EVIDENCE_MISSING"] if terminal_state == "NEEDS_DATA" else []
        else:
            terminal_state = "NEEDS_DATA"
            review_status = underwriting_result.status
            underwriting_verdict = None
            reason_codes = list(underwriting_result.reason_codes)
        watchlist_eligible = (
            terminal_state == "READY_FOR_UNDERWRITING"
            or (terminal_state == "NEEDS_DATA" and ticker == provisional_selected)
            or (
                terminal_state == "UNDERWRITTEN"
                and underwriting_verdict in {"ACTIONABLE", "WATCHLIST_ONLY"}
            )
        )

        dispositions.append(
            CandidateDisposition(
                ticker=ticker,
                terminal_state=terminal_state,
                scope_status="IN_SCOPE",
                screen_status=screen_status,
                review_status=review_status,
                underwriting_verdict=underwriting_verdict,
                underwriting_confidence=underwriting_result.confidence,
                watchlist_eligible=watchlist_eligible,
                reason_codes=list(dict.fromkeys(reason_codes)),
                evidence_ref_ids=list(
                    dict.fromkeys(
                        [
                            *screen_result.evidence_ref_ids,
                            *underwriting_result.evidence_ref_ids,
                        ]
                    )
                ),
                last_completed_stage=(
                    "UNDERWRITING"
                    if child and underwriting_result.status in {"COMPLETED", "INCOMPLETE", "FAILED"}
                    else "SCREENING"
                    if child
                    else "SCREENING"
                    if terminal_state in {"SCREENED_OUT", "READY_FOR_UNDERWRITING"}
                    else "DATA_REPAIR"
                ),
                screen_result=screen_result,
                underwriting_result=underwriting_result,
                **frontier_fields(
                    ticker,
                    terminal_state=terminal_state,
                    review_status=review_status,
                ),
                **identity,
            )
        )
    return dispositions


def _v2_selection_validation(
    artifact: AutonomousSectorFinancialRunArtifact,
    provisional_selected: str | None,
) -> SectorSelectionValidation:
    if artifact.selection_validation is not None:
        return artifact.selection_validation
    if provisional_selected:
        return SectorSelectionValidation(
            status="NOT_ATTEMPTED",
            selected_ticker=provisional_selected,
            reason_codes=["SELECTED_COMPANY_VALIDATION_NOT_RUN"],
            notes=["A separate selected-company challenge pass has not completed."],
        )
    return SectorSelectionValidation(status="NOT_REQUIRED")


def _v2_clear_unreviewed_ranking_verdicts(
    rows: list[dict[str, Any]],
    dispositions: list[CandidateDisposition],
) -> list[dict[str, Any]]:
    by_ticker = {item.ticker: item for item in dispositions}
    normalized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        ticker = str(row.get("ticker") or "").upper()
        disposition = by_ticker.get(ticker)
        if disposition is not None and disposition.review_status != "COMPLETED":
            row["company_autonomy_verdict"] = None
            row["company_autonomy_confidence"] = None
            row["actionable"] = False
            if disposition.terminal_state == "READY_FOR_UNDERWRITING":
                row["positioning_summary"] = (
                    "Deterministic screen passed; company underwriting has not completed."
                )
        normalized.append(row)
    return normalized


def _v2_decision_failure_reasons(
    artifact: AutonomousSectorFinancialRunArtifact,
) -> list[str]:
    """Return run-level failures that make any v2 decision incomplete."""

    return _v2_blocking_run_states(artifact.degraded_states)


def _finalize_sector_artifact_v2(
    artifact: AutonomousSectorFinancialRunArtifact,
    *,
    admitted_tickers: list[str],
) -> AutonomousSectorFinancialRunArtifact:
    """Project the legacy execution into the truthful, rollout-gated v2 contract."""

    execution_tickers = _normalized_ticker_values(admitted_tickers)
    selection = dict(artifact.candidate_selection or {})
    admitted = _normalized_ticker_values(
        (selection.get("membership_tickers") or selection.get("selected_tickers"))
        if bool(selection.get("execution_bound_frozen"))
        else selection.get("loaded_tickers") or execution_tickers
    )
    if not admitted:
        admitted = list(execution_tickers)
    delta_audit = selection.get("delta_audit")
    if isinstance(delta_audit, dict):
        swept = set(_normalized_ticker_values(delta_audit.get("swept_excluded")))
        admitted = [ticker for ticker in admitted if ticker not in swept]
    dispositions = _v2_candidate_dispositions(
        artifact=artifact,
        admitted_tickers=admitted,
        execution_tickers=execution_tickers,
    )
    provisional_selected = (
        str(
            artifact.selected_ticker
            or (artifact.final_decision.selected_ticker if artifact.final_decision else None)
            or ""
        ).upper()
        or None
    )
    validation = _v2_selection_validation(artifact, provisional_selected)
    execution_status = "COMPLETED" if artifact.status == "COMPLETED" else "FAILED"
    frontier_status = str((artifact.competitive_frontier or {}).get("status") or "").upper()
    frontier_present = bool(artifact.competitive_frontier)
    unresolved = any(
        item.scope_status == "IN_SCOPE"
        and (
            item.terminal_state in {"NEEDS_DATA", "DEFERRED_BY_BOUND"}
            or (
                item.terminal_state == "READY_FOR_UNDERWRITING"
                and item.frontier_status != "DOMINATED"
            )
            or (
                item.terminal_state == "READY_FOR_UNDERWRITING"
                and item.frontier_status == "DOMINATED"
                and frontier_status != "CLOSED"
            )
        )
        for item in dispositions
    )
    selected_disposition = next(
        (item for item in dispositions if item.ticker == provisional_selected),
        None,
    )
    decision_failure_reasons = _v2_decision_failure_reasons(artifact)
    eligible_dispositions = [
        item
        for item in dispositions
        if item.scope_status == "IN_SCOPE"
        and item.screen_status == "PASS"
        and item.terminal_state in {"READY_FOR_UNDERWRITING", "UNDERWRITTEN"}
    ]
    source_binding_incomplete = [
        item
        for item in dispositions
        if item.scope_status == "IN_SCOPE"
        and item.screen_status == "PASS"
        and item.terminal_state == "NEEDS_DATA"
        and any("SOURCE_BINDING" in reason for reason in item.reason_codes)
    ]
    frontier_validation_dispositions = [
        *eligible_dispositions,
        *source_binding_incomplete,
    ]
    if frontier_validation_dispositions and not frontier_present:
        decision_failure_reasons.append("COMPETITIVE_FRONTIER_MISSING")
    elif frontier_validation_dispositions and frontier_status != "CLOSED":
        decision_failure_reasons.append("COMPETITIVE_FRONTIER_OPEN")
    elif frontier_validation_dispositions:
        from app.autonomous.competitive_frontier import (
            validate_closed_competitive_frontier_payload,
        )

        try:
            expected_candidate_tickers = [item.ticker for item in frontier_validation_dispositions]
            expected_candidate_set = set(expected_candidate_tickers)
            validate_closed_competitive_frontier_payload(
                artifact.competitive_frontier,
                expected_candidate_tickers=expected_candidate_tickers,
                expected_reviewed_tickers=[
                    item.ticker
                    for item in frontier_validation_dispositions
                    if item.review_status == "COMPLETED"
                ],
                source_company_packets=[
                    packet
                    for packet in artifact.company_packets
                    if packet.ticker.strip().upper() in expected_candidate_set
                ],
                source_scenarios=[
                    scenario
                    for scenario in artifact.expected_return_scenarios
                    if scenario.ticker.strip().upper() in expected_candidate_set
                ],
            )
        except ValueError:
            decision_failure_reasons.append("COMPETITIVE_FRONTIER_INVALID")
    elif frontier_present and frontier_status != "CLOSED":
        decision_failure_reasons.append("COMPETITIVE_FRONTIER_OPEN")

    final_verdict: str | None = None
    selected_ticker: str | None = None
    confidence: str | None = None
    decision_status = "INCOMPLETE"
    if execution_status == "COMPLETED" and not unresolved and not decision_failure_reasons:
        if (
            selected_disposition is not None
            and selected_disposition.terminal_state == "UNDERWRITTEN"
            and selected_disposition.underwriting_verdict == "ACTIONABLE"
            and validation.status == "VALIDATED"
            and validation.selected_ticker == provisional_selected
        ):
            decision_status = "COMPLETE"
            final_verdict = "SELECTED"
            selected_ticker = provisional_selected
            confidence = artifact.confidence
        elif (
            selected_disposition is not None
            and selected_disposition.terminal_state == "UNDERWRITTEN"
            and selected_disposition.underwriting_verdict == "WATCHLIST_ONLY"
            and not any(
                item.scope_status == "IN_SCOPE"
                and item.terminal_state == "UNDERWRITTEN"
                and item.underwriting_verdict == "ACTIONABLE"
                for item in dispositions
            )
        ):
            decision_status = "COMPLETE"
            final_verdict = "WATCHLIST"
            selected_ticker = provisional_selected
            confidence = artifact.confidence
        elif all(
            item.terminal_state == "SCREENED_OUT"
            or (
                item.terminal_state == "READY_FOR_UNDERWRITING"
                and item.frontier_status == "DOMINATED"
            )
            or (item.terminal_state == "UNDERWRITTEN" and item.underwriting_verdict == "AVOID")
            for item in dispositions
            if item.scope_status == "IN_SCOPE"
        ):
            decision_status = "COMPLETE"
            final_verdict = "NO_SELECTION"

    payload = artifact.to_dict()
    provisional_decision = payload.get("final_decision")
    payload.update(
        {
            "contract_version": SECTOR_CONTRACT_VERSION_V2,
            "pipeline_version": SECTOR_PIPELINE_VERSION_V2,
            "status": execution_status,
            "execution_status": execution_status,
            "decision_status": decision_status,
            "admitted_tickers": admitted,
            "candidate_dispositions": [item.to_dict() for item in dispositions],
            "selection_validation": validation.to_dict(),
            "provisional_final_decision": provisional_decision,
            "final_verdict": final_verdict,
            "selected_ticker": selected_ticker,
            "confidence": confidence,
            "final_decision": provisional_decision if decision_status == "COMPLETE" else None,
            "no_selection_reason": (
                artifact.no_selection_reason if final_verdict == "NO_SELECTION" else None
            ),
            "relative_ranking": _v2_clear_unreviewed_ranking_verdicts(
                list(artifact.relative_ranking),
                dispositions,
            ),
        }
    )
    audit = dict(payload.get("selection_audit") or {})
    audit["actionable"] = final_verdict == "SELECTED"
    audit["final_verdict_after_audit"] = final_verdict or "DECISION_INCOMPLETE"
    payload["selection_audit"] = audit
    if decision_failure_reasons:
        audit["v2_decision_failure_reasons"] = decision_failure_reasons
    degraded = list(payload.get("degraded_states") or [])
    if execution_status == "FAILED":
        degraded.append("V2_EXECUTION_FAILED")
    elif decision_status == "INCOMPLETE":
        degraded.append("V2_DECISION_INCOMPLETE")
    payload["degraded_states"] = list(dict.fromkeys(degraded))
    audit_notes = list(payload.get("audit_notes") or [])
    audit_notes.append(
        "V2 truth projection preserved the admitted roster and separated screen, underwriting, and validation state."
    )
    payload["audit_notes"] = list(dict.fromkeys(audit_notes))
    return AutonomousSectorFinancialRunArtifact.from_dict(payload)


def run_sector_autonomous_financial_analysis(
    *,
    sector: str,
    tickers: list[str],
    objective: str | None = None,
    as_of_date: str | None = None,
    market_cap_focus: str = "small_cap",
    budget: AutonomousRunBudget | None = None,
    allowed_tools: list[str] | None = None,
    candidate_selection: dict[str, Any] | None = None,
    pipeline_version: str | None = None,
    terminal_cap_search: Any | None = None,
) -> AutonomousSectorFinancialRunArtifact:
    """Run the sector analyst loop under a bounded run-wide LLM retry budget."""

    run_budget = _budget_or_default(budget)
    resolved_pipeline_version = resolve_autonomous_sector_pipeline_version(
        market_cap_focus,
        pipeline_version,
    )
    active_retry_context = current_retry_context()
    active_cost_context = current_cost_context()

    def _invoke(retry_context: Any, cost_context: Any) -> AutonomousSectorFinancialRunArtifact:
        return _run_sector_autonomous_financial_analysis_with_retry_context(
            sector=sector,
            tickers=tickers,
            objective=objective,
            as_of_date=as_of_date,
            market_cap_focus=market_cap_focus,
            budget=run_budget,
            allowed_tools=allowed_tools,
            candidate_selection=candidate_selection,
            pipeline_version=resolved_pipeline_version,
            terminal_cap_search=terminal_cap_search,
            retry_context=retry_context,
            cost_context=cost_context,
        )

    if active_retry_context is None and active_cost_context is None:
        with (
            llm_retry_budget(SECTOR_LLM_RETRY_RUN_BUDGET) as retry_context,
            llm_cost_budget(run_budget.max_cost_usd) as cost_context,
        ):
            return _invoke(retry_context, cost_context)
    if active_retry_context is None:
        with llm_retry_budget(SECTOR_LLM_RETRY_RUN_BUDGET) as retry_context:
            return _invoke(retry_context, active_cost_context)
    if active_cost_context is None:
        with llm_cost_budget(run_budget.max_cost_usd) as cost_context:
            return _run_sector_autonomous_financial_analysis_with_retry_context(
                sector=sector,
                tickers=tickers,
                objective=objective,
                as_of_date=as_of_date,
                market_cap_focus=market_cap_focus,
                budget=run_budget,
                allowed_tools=allowed_tools,
                candidate_selection=candidate_selection,
                pipeline_version=resolved_pipeline_version,
                terminal_cap_search=terminal_cap_search,
                retry_context=active_retry_context,
                cost_context=cost_context,
            )
    return _invoke(active_retry_context, active_cost_context)


def _run_sector_autonomous_financial_analysis_with_retry_context(
    *,
    sector: str,
    tickers: list[str],
    objective: str | None,
    as_of_date: str | None,
    market_cap_focus: str,
    budget: AutonomousRunBudget | None,
    allowed_tools: list[str] | None,
    candidate_selection: dict[str, Any] | None,
    pipeline_version: str,
    terminal_cap_search: Any | None,
    retry_context: Any,
    cost_context: Any,
) -> AutonomousSectorFinancialRunArtifact:
    with provider_usage_capture("parent_research") as provider_usage:
        try:
            with sector_framework_pipeline_version(pipeline_version):
                artifact = _run_sector_autonomous_financial_analysis_impl(
                    sector=sector,
                    tickers=tickers,
                    objective=objective,
                    as_of_date=as_of_date,
                    market_cap_focus=market_cap_focus,
                    budget=budget,
                    allowed_tools=allowed_tools,
                    candidate_selection=candidate_selection,
                    pipeline_version=pipeline_version,
                    terminal_cap_search=terminal_cap_search,
                )
        except (LLMRetryBudgetExceeded, LLMCostBudgetExceeded) as exc:
            state = _provider_failure_state(exc)
            created_at = _utc_now_iso()
            with sector_framework_pipeline_version(pipeline_version):
                artifact = _empty_artifact(
                    run_id=_run_id(sector, created_at),
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    objective=objective or DEFAULT_SECTOR_OBJECTIVE,
                    as_of_date=as_of_date or date.today().isoformat(),
                    created_at=created_at,
                    degraded_state=state,
                    reason=f"LLM budget exceeded before the sector run could complete: {exc}",
                    audit_notes=[
                        f"Aborted with DEGRADED status because an LLM budget was exceeded ({state}): {exc}"
                    ],
                    candidate_selection=candidate_selection,
                    status="DEGRADED",
                )
    artifact.provider_usage = merge_provider_usage_records(
        artifact.provider_usage,
        provider_usage,
    )
    for call in artifact.tool_calls:
        if call.lane is None:
            question_id = str(call.question_id or "").upper()
            call.lane = (
                "repair_fallback" if question_id.startswith(("AGR", "WR")) else "parent_research"
            )
    _append_llm_retry_audit_notes(artifact, retry_context)
    _append_llm_cost_audit_notes(artifact, cost_context)
    if pipeline_version != SECTOR_PIPELINE_VERSION_V2:
        artifact.candidate_dispositions = _valuation_anchor_dispositions(
            artifact.candidate_selection
        )
    if pipeline_version == SECTOR_PIPELINE_VERSION_V2:
        try:
            _rebuild_v2_competitive_frontier_before_finalization(artifact)
        except (TypeError, ValueError) as exc:
            failed_frontier = {
                "status": "OPEN",
                "source_binding": ("FINAL_ARTIFACT_COMPANY_PACKETS_AND_EXPECTED_RETURN_SCENARIOS"),
                "reason_codes": ["COMPETITIVE_FRONTIER_SOURCE_REBUILD_FAILED"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            artifact.competitive_frontier = failed_frontier
            artifact.candidate_selection = {
                **(artifact.candidate_selection or {}),
                "competitive_frontier": failed_frontier,
            }
            artifact.degraded_states = list(
                dict.fromkeys(
                    [
                        *artifact.degraded_states,
                        "COMPETITIVE_FRONTIER_SOURCE_REBUILD_FAILED",
                    ]
                )
            )
            artifact.audit_notes.append(
                "Final source-backed competitive-frontier rebuild failed; the v2 decision remains incomplete."
            )
        artifact.lane_budget = _v2_lane_budget_payload(
            run_budget=_budget_or_default(budget),
            competitive_frontier=artifact.competitive_frontier,
        )
        artifact.lane_usage = _v2_lane_usage_payload(artifact)
        artifact.audit_notes.append(
            "V2 lane accounting includes parent, repair, company-child, selected-company validation, and terminal-cap attempts."
        )
        artifact = _finalize_sector_artifact_v2(
            artifact,
            admitted_tickers=tickers,
        )
    return artifact


def sector_artifact_summary(artifact: AutonomousSectorFinancialRunArtifact) -> dict[str, Any]:
    """Small JSON-safe summary for sector autonomous runs."""

    audit = artifact.selection_audit if isinstance(artifact.selection_audit, dict) else {}
    memo_body = artifact.memo_body if isinstance(artifact.memo_body, dict) else {}
    valuation_anchor_summary = _valuation_anchor_summary(artifact.company_packets)
    disposition_counts = Counter(
        item.terminal_state for item in getattr(artifact, "candidate_dispositions", [])
    )
    memo_candidates = (
        memo_body.get("candidates") if isinstance(memo_body.get("candidates"), dict) else {}
    )
    successful_candidate_schemas = {
        str(row.get("schema_name") or "")
        for row in artifact.provider_usage
        if isinstance(row, dict) and str(row.get("status") or "").upper() == "OK"
    }
    packet_tickers = _artifact_packet_tickers(artifact)
    candidate_review_completed = sum(
        1
        for ticker in packet_tickers
        if isinstance(memo_candidates.get(ticker), dict)
        and str(memo_candidates[ticker].get("source") or "").lower() == "llm"
        and str(memo_candidates[ticker].get("status") or "").upper() == "OK"
        and str(memo_candidates[ticker].get("ticker") or ticker).strip().upper() == ticker
        and candidate_memo_is_substantive(memo_candidates[ticker])
        and candidate_memo_schema_name(ticker) in successful_candidate_schemas
    )
    final_prompt_context = (
        artifact.final_decision_prompt_context
        if isinstance(artifact.final_decision_prompt_context, dict)
        else {}
    )
    candidate_selection = (
        artifact.candidate_selection if isinstance(artifact.candidate_selection, dict) else {}
    )
    data_gap_repair = (
        candidate_selection.get("data_gap_repair")
        if isinstance(candidate_selection.get("data_gap_repair"), dict)
        else {}
    )
    selection_validation = getattr(artifact, "selection_validation", None)
    return {
        "scan_family": getattr(artifact, "scan_family", "normal") or "normal",
        "pipeline_version": getattr(artifact, "pipeline_version", "v1") or "v1",
        "execution_status": getattr(artifact, "execution_status", None),
        "decision_status": getattr(artifact, "decision_status", None),
        "admitted_ticker_count": len(getattr(artifact, "admitted_tickers", []) or []),
        "candidate_disposition_counts": dict(sorted(disposition_counts.items())),
        "selection_validation_status": (
            selection_validation.status if selection_validation is not None else None
        ),
        "run_id": artifact.run_id,
        "sector": artifact.sector,
        "market_cap_focus": artifact.market_cap_focus,
        "status": artifact.status,
        "final_verdict": artifact.final_verdict,
        "selected_ticker": artifact.selected_ticker,
        "confidence": artifact.confidence,
        "selection_audit_status": audit.get("status"),
        "actionable": audit.get("actionable"),
        "confidence_ceiling": audit.get("confidence_ceiling"),
        "return_cushion_status": audit.get("return_cushion_status"),
        "base_return_margin_over_hurdle": audit.get("base_return_margin_over_hurdle"),
        "downside_annualized_return": audit.get("downside_annualized_return"),
        "downside_evidence_status": audit.get("downside_evidence_status"),
        "capital_loss_underwriting_status": audit.get("capital_loss_underwriting_status"),
        "capital_loss_impairment_class": audit.get("capital_loss_impairment_class"),
        "refinancing_timeline_status": audit.get("refinancing_timeline_status"),
        "refinancing_timeline_needs": audit.get("refinancing_timeline_needs"),
        "selected_company_evidence_tools": audit.get("selected_company_evidence_tools"),
        "selected_evidence_pillars": audit.get("selected_evidence_pillars"),
        "capital_structure_resolution_status": audit.get("capital_structure_resolution_status"),
        "framework_evidence_preflight": artifact.framework_evidence_preflight,
        "audit_gap_repair_attempted": artifact.audit_gap_repair_attempted,
        "audit_gap_repair_status": artifact.audit_gap_repair_status,
        "watchlist_resolution_attempted": artifact.watchlist_resolution_attempted,
        "watchlist_resolution_status": artifact.watchlist_resolution_status,
        "no_selection_finalist_audit_attempted": artifact.no_selection_finalist_audit_attempted,
        "no_selection_finalist_audit_status": artifact.no_selection_finalist_audit_status,
        "no_selection_finalist_audit_focus_ticker": artifact.no_selection_finalist_audit_focus_ticker,
        "no_selection_finalist_resolution_attempted": artifact.no_selection_finalist_resolution_attempted,
        "no_selection_finalist_resolution_status": artifact.no_selection_finalist_resolution_status,
        "alternate_finalist_audit_attempted": artifact.alternate_finalist_audit_attempted,
        "alternate_finalist_audit_status": artifact.alternate_finalist_audit_status,
        "alternate_finalist_audit_results": artifact.alternate_finalist_audit_results,
        "company_autonomy_attempted": artifact.company_autonomy_attempted,
        "company_autonomy_status": artifact.company_autonomy_status,
        "company_autonomy_decision_trace": artifact.company_autonomy_decision_trace,
        "company_autonomy_runs": [
            {
                "ticker": run.get("ticker"),
                "status": run.get("status"),
                "final_verdict": run.get("final_verdict"),
                "confidence": run.get("confidence"),
                "tool_calls": run.get("tool_calls"),
                "evidence_references": run.get("evidence_references"),
            }
            for run in artifact.company_autonomy_runs
        ],
        "relative_ranking": artifact.relative_ranking,
        "memo_body_status": memo_body.get("status"),
        "memo_body_section_sources": {
            "cohort_comparison": (memo_body.get("cohort_comparison") or {}).get("source")
            if isinstance(memo_body.get("cohort_comparison"), dict)
            else None,
            "triage_surprises": (memo_body.get("triage_surprises") or {}).get("source")
            if isinstance(memo_body.get("triage_surprises"), dict)
            else None,
            "candidates": sorted(
                {
                    str((item or {}).get("source") or "")
                    for item in (memo_body.get("candidates") or {}).values()
                    if isinstance(item, dict)
                }
            )
            if isinstance(memo_body.get("candidates"), dict)
            else [],
        },
        "memo_body_degraded_states": list(memo_body.get("degraded_states") or []),
        "memo_body_usage": _memo_body_usage_summary(memo_body),
        "data_gap_repair_examined": int(data_gap_repair.get("examined") or 0),
        "sector_context_tickers": len(final_prompt_context.get("prompt_scoped_tickers") or []),
        "candidate_review_completed": candidate_review_completed,
        "candidate_review_failed": max(0, len(packet_tickers) - candidate_review_completed),
        "company_packets": len(artifact.company_packets),
        **valuation_anchor_summary,
        "candidate_source": candidate_selection.get("source"),
        "candidate_selection": candidate_selection,
        "expected_return_scenarios": len(artifact.expected_return_scenarios),
        "research_questions": len(artifact.research_questions),
        "tool_calls": len([call for call in artifact.tool_calls if call.status == "OK"]),
        "degraded_states": list(artifact.degraded_states),
        "no_selection_reason": artifact.no_selection_reason,
    }


def _valuation_anchor_summary(
    company_packets: list[SectorCompanyFinancialPacket],
) -> dict[str, Any]:
    valid_count = 0
    generic_count = 0
    sector_specific_count = 0
    invalid_generic_count = 0
    missing_count = 0
    unknown_method_count = 0
    method_counts: dict[str, int] = {}

    for packet in company_packets:
        valuation = packet.valuation if isinstance(packet.valuation, dict) else {}
        anchor = _optional_number(valuation.get("valuation_anchor"))
        method = str(valuation.get("anchor_method") or "").strip().lower()
        generic_valuation_valid = valuation.get("generic_valuation_valid")
        if anchor is None:
            missing_count += 1
            continue
        if method in GENERIC_VALUATION_ANCHOR_METHODS and generic_valuation_valid is False:
            invalid_generic_count += 1
            continue
        valid_count += 1
        method_key = method or "unknown"
        method_counts[method_key] = method_counts.get(method_key, 0) + 1
        if method in GENERIC_VALUATION_ANCHOR_METHODS:
            generic_count += 1
        elif method:
            sector_specific_count += 1
        else:
            unknown_method_count += 1

    return {
        "valuation_anchor_count": valid_count,
        "generic_valuation_anchor_count": generic_count,
        "sector_specific_valuation_anchor_count": sector_specific_count,
        "unknown_method_valuation_anchor_count": unknown_method_count,
        "invalid_generic_valuation_anchor_count": invalid_generic_count,
        "missing_valuation_anchor_count": missing_count,
        "valuation_anchor_method_counts": dict(sorted(method_counts.items())),
    }


__all__ = [
    "DEFAULT_SECTOR_ALLOWED_TOOLS",
    "DEFAULT_SECTOR_BUDGET",
    "DEFAULT_SECTOR_OBJECTIVE",
    "enrich_sector_artifact_memo_body",
    "run_sector_autonomous_financial_analysis",
    "sector_artifact_summary",
]
