from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.valuation.accounting_quality import ACCOUNTING_QUALITY_UNKNOWN, LOW_ACCOUNTING_QUALITY
from app.valuation.balance_sheet_stress import (
    BALANCE_SHEET_STRESS_UNKNOWN,
    HIGH_BALANCE_SHEET_STRESS,
    LOW_BALANCE_SHEET_STRESS,
)
from app.valuation.returns_persistence import (
    HIGH_RETURNS_PERSISTENCE,
    LOW_RETURNS_PERSISTENCE,
    RETURNS_PERSISTENCE_UNKNOWN,
)
from app.valuation.revenue_dependence import (
    HIGH_REVENUE_DEPENDENCE_RISK,
    LOW_REVENUE_DEPENDENCE_RISK,
    REASON_CHANNEL_DEPENDENCE_HEADWIND,
    REASON_CUSTOMER_CONCENTRATION_HEADWIND,
    REVENUE_DEPENDENCE_UNKNOWN,
)
from app.valuation.evidence_sufficiency import (
    MOS_CONFIRMED_ABSENT,
    MOS_CONFIRMED_PRESENT,
    MOS_UNASSESSABLE,
    MOS_UNKNOWN_STATUS,
    MOS_WEAK,
    REASON_MOS_BLOCKED_BY_MISSING_FACTS,
    REASON_MOS_BLOCKED_BY_MISSING_PRICE,
    REASON_MOS_BLOCKED_BY_MISSING_SHARES,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_UNKNOWN,
    compute_evidence_sufficiency,
)
from app.valuation.intrinsic_discipline import (
    MOS_UNKNOWN,
    REASON_CYCLICAL_NORMALIZATION_LOW_CONFIDENCE,
    SUPPORT_ASSET,
    SUPPORT_BALANCE_SHEET,
    SUPPORT_EARNINGS,
    SUPPORT_UNKNOWN,
)
from app.valuation.valuation_confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_UNKNOWN,
    FRAGILITY_HIGH,
    FRAGILITY_LOW,
    FRAGILITY_MODERATE,
    REASON_SINGLE_SUPPORT_ONLY,
)
from app.valuation.valuation_integrity import (
    INTEGRITY_OK,
    INTEGRITY_SUSPECT,
    INTEGRITY_UNKNOWN,
    INTEGRITY_WARNING,
)
from app.valuation.value_type import (
    VALUE_TYPE_CYCLICAL,
    VALUE_TYPE_FRAGILE,
    VALUE_TYPE_UNKNOWN,
)


UNKNOWN = "UNKNOWN"
OK = "OK"

READY_INVESTABLE_NOW = "INVESTABLE_NOW"
READY_RESEARCH_WORTHY = "RESEARCH_WORTHY_NOT_READY"
READY_WATCH_ONLY = "WATCH_ONLY"
READY_NOT_INVESTABLE = "NOT_INVESTABLE"
READY_UNKNOWN = "READINESS_UNKNOWN"

READINESS_ORDER = {
    READY_INVESTABLE_NOW: 0,
    READY_RESEARCH_WORTHY: 1,
    READY_WATCH_ONLY: 2,
    READY_NOT_INVESTABLE: 3,
    READY_UNKNOWN: 4,
}

NEXT_DEEPER_UNDERWRITING = "DEEPER_UNDERWRITING"
NEXT_CLEAR_EVIDENCE = "CLEAR_EVIDENCE_BLOCKERS"
NEXT_MONITOR_ONLY = "MONITOR_ONLY"
NEXT_DO_NOT_ADVANCE = "DO_NOT_ADVANCE"
NEXT_UNKNOWN = "NEXT_STEP_UNKNOWN"

BLOCKER_VALUATION_NO_MOS_CONFIRMED = "VALUATION_NO_MOS_CONFIRMED"
BLOCKER_EVIDENCE_INSUFFICIENT_FOR_MOS = "EVIDENCE_INSUFFICIENT_FOR_MOS"
BLOCKER_MOS_UNASSESSABLE_MISSING_PRICE = "MOS_UNASSESSABLE_MISSING_PRICE"
BLOCKER_MOS_UNASSESSABLE_MISSING_FACTS = "MOS_UNASSESSABLE_MISSING_FACTS"
BLOCKER_MOS_UNASSESSABLE_MISSING_SHARES = "MOS_UNASSESSABLE_MISSING_SHARES"
BLOCKER_VALUATION_FRAGILE = "VALUATION_FRAGILE"
BLOCKER_VALUATION_INTEGRITY_WARNING = "VALUATION_INTEGRITY_WARNING"
BLOCKER_VALUATION_INTEGRITY_SUSPECT = "VALUATION_INTEGRITY_SUSPECT"
BLOCKER_LOW_CONFIDENCE_VALUE = "LOW_CONFIDENCE_VALUE"
BLOCKER_MISSING_PRICE = "MISSING_PRICE"
BLOCKER_MISSING_FACTS = "MISSING_FACTS"
BLOCKER_MISSING_SHARES = "MISSING_SHARES"
BLOCKER_MISSING_FCF = "MISSING_FCF"
BLOCKER_STRUCTURAL_WEAK_ECONOMICS = "STRUCTURAL_WEAK_ECONOMICS"
BLOCKER_LOW_OWNER_EARNINGS_QUALITY = "LOW_OWNER_EARNINGS_QUALITY"
BLOCKER_IMPAIRMENT_CONFIRMED = "IMPAIRMENT_CONFIRMED"
BLOCKER_PROBABLE_IMPAIRMENT = "PROBABLE_IMPAIRMENT_HEADWIND"
BLOCKER_EVIDENCE_GAP_PREVENTS_IMPAIRMENT_JUDGMENT = "EVIDENCE_GAP_PREVENTS_IMPAIRMENT_JUDGMENT"
BLOCKER_STRUCTURALLY_WEAK_ECONOMICS = "STRUCTURALLY_WEAK_ECONOMICS"
BLOCKER_HIGH_DILUTION = "HIGH_DILUTION"
BLOCKER_HIGH_LEVERAGE = "HIGH_LEVERAGE"
BLOCKER_UNKNOWN_VALUE_TYPE = "UNKNOWN_VALUE_TYPE"
BLOCKER_CYCLICAL_LOW_CONFIDENCE = "CYCLICAL_LOW_CONFIDENCE"
BLOCKER_PEAK_EARNINGS_CYCLICAL_RISK = "PEAK_EARNINGS_CYCLICAL_RISK"
BLOCKER_LOW_REINVESTMENT_EFFICIENCY = "LOW_REINVESTMENT_EFFICIENCY"
BLOCKER_REINVESTMENT_EVIDENCE_THIN = "REINVESTMENT_EVIDENCE_THIN"
BLOCKER_LOW_ACCOUNTING_QUALITY = "LOW_ACCOUNTING_QUALITY"
BLOCKER_ACCOUNTING_EVIDENCE_THIN = "ACCOUNTING_EVIDENCE_THIN"
BLOCKER_HIGH_BALANCE_SHEET_STRESS = "HIGH_BALANCE_SHEET_STRESS"
BLOCKER_HIGH_REFINANCING_RISK = "HIGH_REFINANCING_RISK"
BLOCKER_BALANCE_SHEET_EVIDENCE_THIN = "BALANCE_SHEET_EVIDENCE_THIN"
BLOCKER_LOW_RETURNS_PERSISTENCE = "LOW_RETURNS_PERSISTENCE"
BLOCKER_RETURNS_DURABILITY_UNKNOWN = "RETURNS_DURABILITY_UNKNOWN"
BLOCKER_HIGH_REVENUE_DEPENDENCE_RISK = "HIGH_REVENUE_DEPENDENCE_RISK"
BLOCKER_REVENUE_DEPENDENCE_UNKNOWN = "REVENUE_DEPENDENCE_UNKNOWN"
BLOCKER_FACTS_RETRYABLE_TIMEOUT = "FACTS_RETRYABLE_TIMEOUT"
BLOCKER_FACTS_RETRYABLE_NETWORK_FAILURE = "FACTS_RETRYABLE_NETWORK_FAILURE"
BLOCKER_FACTS_RETRYABLE_RATE_LIMIT = "FACTS_RETRYABLE_RATE_LIMIT"
BLOCKER_FACTS_NO_CACHE_OFFLINE = "FACTS_NO_CACHE_OFFLINE"
BLOCKER_FACTS_PROVIDER_NO_DATA = "FACTS_PROVIDER_NO_DATA"
BLOCKER_FACTS_PARSE_FAILURE = "FACTS_PARSE_FAILURE"
BLOCKER_FACTS_PARTIAL_COVERAGE = "FACTS_PARTIAL_COVERAGE"
BLOCKER_FACTS_TERMINAL_MISSING_KEY_INPUTS = "FACTS_TERMINAL_MISSING_KEY_INPUTS"

SUPPORT_ADEQUATE_MOS = "ADEQUATE_MOS"
SUPPORT_MULTI_SUPPORT = "MULTI_SUPPORT_VALUE_CASE"
SUPPORT_LOW_FRAGILITY = "LOW_FRAGILITY"
SUPPORT_INTEGRITY_OK = "INTEGRITY_OK"
SUPPORT_EARNINGS_POWER = "EARNINGS_POWER_SUPPORT"
SUPPORT_BALANCE = "BALANCE_SHEET_SUPPORT"
SUPPORT_HIGH_OE_QUALITY = "HIGH_OE_QUALITY"
SUPPORT_HIGH_INTANGIBLE = "HIGH_INTANGIBLE_SUPPORT"
SUPPORT_TEMPORARY_WEAKNESS_WITH_SUPPORT = "TEMPORARY_WEAKNESS_WITH_SUPPORT"
SUPPORT_PRODUCTIVE_REINVESTMENT = "PRODUCTIVE_REINVESTMENT_SUPPORT"
SUPPORT_HIGH_ACCOUNTING_QUALITY = "HIGH_ACCOUNTING_QUALITY_SUPPORT"
SUPPORT_LOW_BALANCE_SHEET_STRESS = "LOW_BALANCE_SHEET_STRESS_SUPPORT"
SUPPORT_HIGH_RETURNS_PERSISTENCE = "HIGH_RETURNS_PERSISTENCE_SUPPORT"
SUPPORT_LOW_REVENUE_DEPENDENCE = "LOW_REVENUE_DEPENDENCE_SUPPORT"

HEADWIND_MOS_UNKNOWN = "MOS_UNKNOWN"
HEADWIND_IMPAIRMENT_CONFIRMED = "IMPAIRMENT_CONFIRMED_HEADWIND"
HEADWIND_PROBABLE_IMPAIRMENT = "PROBABLE_IMPAIRMENT_HEADWIND"
HEADWIND_STRUCTURALLY_WEAK = "STRUCTURALLY_WEAK_HEADWIND"
HEADWIND_EVIDENCE_GAP_IMPAIRMENT = "EVIDENCE_GAP_PREVENTS_IMPAIRMENT_JUDGMENT"
HEADWIND_MOS_UNASSESSABLE = "MOS_UNASSESSABLE_EVIDENCE_GAP"
HEADWIND_INTEGRITY_UNKNOWN = "INTEGRITY_UNKNOWN"
HEADWIND_FACTS_MISSING = "FACTS_MISSING"
HEADWIND_QUALITY = "QUALITY_HEADWIND"
HEADWIND_FRAGILE_VALUE = "FRAGILE_VALUE_HEADWIND"
HEADWIND_REINVESTMENT_MIXED = "REINVESTMENT_MIXED"
HEADWIND_CAPITAL_HUNGRY_GROWTH = "CAPITAL_HUNGRY_GROWTH_HEADWIND"
HEADWIND_GROWTH_WITHOUT_OWNER_OUTCOME = "GROWTH_WITHOUT_OWNER_OUTCOME"
HEADWIND_LOW_ACCOUNTING_QUALITY = "LOW_ACCOUNTING_QUALITY_HEADWIND"
HEADWIND_ACCRUAL_HEAVY = "ACCRUAL_HEAVY_EARNINGS_HEADWIND"
HEADWIND_WEAK_CASH_CONVERSION = "WEAK_CASH_CONVERSION_HEADWIND"
HEADWIND_REINVESTMENT_EVIDENCE_THIN = "REINVESTMENT_EVIDENCE_THIN"
HEADWIND_HIGH_BALANCE_SHEET_STRESS = "HIGH_BALANCE_SHEET_STRESS_HEADWIND"
HEADWIND_HIGH_REFINANCING_RISK = "HIGH_REFINANCING_RISK_HEADWIND"
HEADWIND_CAPITAL_STRUCTURE_DOMINATES = "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"
HEADWIND_LOW_RETURNS_PERSISTENCE = "LOW_RETURNS_PERSISTENCE_HEADWIND"
HEADWIND_INCREMENTAL_RETURNS_DETERIORATION = "INCREMENTAL_RETURNS_DETERIORATION"
HEADWIND_RETURNS_DURABILITY_MIXED = "RETURNS_DURABILITY_MIXED"
HEADWIND_RETURNS_DURABILITY_UNKNOWN = "RETURNS_DURABILITY_UNKNOWN"
HEADWIND_HIGH_REVENUE_DEPENDENCE = "HIGH_REVENUE_DEPENDENCE_HEADWIND"
HEADWIND_CUSTOMER_CONCENTRATION = "CUSTOMER_CONCENTRATION_HEADWIND"
HEADWIND_CHANNEL_DEPENDENCE = "CHANNEL_DEPENDENCE_HEADWIND"
HEADWIND_REVENUE_DEPENDENCE_UNKNOWN = "REVENUE_DEPENDENCE_UNKNOWN"

REASON_READY_WITH_REAL_SUPPORT = "READY_WITH_REAL_SUPPORT"
REASON_BLOCKED_BUT_RESEARCH_WORTHY = "BLOCKED_BUT_RESEARCH_WORTHY"
REASON_WATCH_WITH_THIN_EDGE = "WATCH_WITH_THIN_EDGE"
REASON_NOT_INVESTABLE_STRUCTURAL = "NOT_INVESTABLE_STRUCTURAL"
REASON_READINESS_UNKNOWN_EVIDENCE_GAP = "READINESS_UNKNOWN_EVIDENCE_GAP"

_BLOCKER_PRIORITY = {
    BLOCKER_IMPAIRMENT_CONFIRMED: 0,
    BLOCKER_PROBABLE_IMPAIRMENT: 1,
    BLOCKER_STRUCTURAL_WEAK_ECONOMICS: 2,
    BLOCKER_VALUATION_INTEGRITY_SUSPECT: 2,
    BLOCKER_VALUATION_NO_MOS_CONFIRMED: 2,
    BLOCKER_STRUCTURALLY_WEAK_ECONOMICS: 3,
    BLOCKER_HIGH_DILUTION: 3,
    BLOCKER_HIGH_LEVERAGE: 3,
    BLOCKER_VALUATION_FRAGILE: 5,
    BLOCKER_LOW_CONFIDENCE_VALUE: 6,
    BLOCKER_LOW_OWNER_EARNINGS_QUALITY: 7,
    BLOCKER_LOW_ACCOUNTING_QUALITY: 8,
    BLOCKER_HIGH_BALANCE_SHEET_STRESS: 9,
    BLOCKER_HIGH_REFINANCING_RISK: 10,
    BLOCKER_LOW_RETURNS_PERSISTENCE: 11,
    BLOCKER_HIGH_REVENUE_DEPENDENCE_RISK: 12,
    BLOCKER_MOS_UNASSESSABLE_MISSING_PRICE: 13,
    BLOCKER_MOS_UNASSESSABLE_MISSING_FACTS: 14,
    BLOCKER_MOS_UNASSESSABLE_MISSING_SHARES: 15,
    BLOCKER_EVIDENCE_INSUFFICIENT_FOR_MOS: 16,
    BLOCKER_FACTS_TERMINAL_MISSING_KEY_INPUTS: 17,
    BLOCKER_FACTS_PROVIDER_NO_DATA: 18,
    BLOCKER_FACTS_PARSE_FAILURE: 19,
    BLOCKER_FACTS_RETRYABLE_TIMEOUT: 20,
    BLOCKER_FACTS_RETRYABLE_NETWORK_FAILURE: 21,
    BLOCKER_FACTS_RETRYABLE_RATE_LIMIT: 22,
    BLOCKER_FACTS_NO_CACHE_OFFLINE: 23,
    BLOCKER_FACTS_PARTIAL_COVERAGE: 24,
    BLOCKER_MISSING_PRICE: 25,
    BLOCKER_MISSING_FACTS: 26,
    BLOCKER_MISSING_SHARES: 27,
    BLOCKER_MISSING_FCF: 28,
    BLOCKER_CYCLICAL_LOW_CONFIDENCE: 29,
    BLOCKER_PEAK_EARNINGS_CYCLICAL_RISK: 30,
    BLOCKER_LOW_REINVESTMENT_EFFICIENCY: 31,
    BLOCKER_REINVESTMENT_EVIDENCE_THIN: 32,
    BLOCKER_ACCOUNTING_EVIDENCE_THIN: 33,
    BLOCKER_BALANCE_SHEET_EVIDENCE_THIN: 34,
    BLOCKER_RETURNS_DURABILITY_UNKNOWN: 35,
    BLOCKER_REVENUE_DEPENDENCE_UNKNOWN: 36,
    BLOCKER_VALUATION_INTEGRITY_WARNING: 37,
    BLOCKER_UNKNOWN_VALUE_TYPE: 38,
    BLOCKER_EVIDENCE_GAP_PREVENTS_IMPAIRMENT_JUDGMENT: 39,
}

_RETRYABLE_BLOCKERS = {
    BLOCKER_FACTS_RETRYABLE_TIMEOUT,
    BLOCKER_FACTS_RETRYABLE_NETWORK_FAILURE,
    BLOCKER_FACTS_RETRYABLE_RATE_LIMIT,
    BLOCKER_FACTS_NO_CACHE_OFFLINE,
    BLOCKER_FACTS_PARTIAL_COVERAGE,
    BLOCKER_MISSING_PRICE,
    BLOCKER_MISSING_FACTS,
    BLOCKER_MISSING_SHARES,
    BLOCKER_MISSING_FCF,
    BLOCKER_EVIDENCE_GAP_PREVENTS_IMPAIRMENT_JUDGMENT,
    BLOCKER_REINVESTMENT_EVIDENCE_THIN,
    BLOCKER_ACCOUNTING_EVIDENCE_THIN,
    BLOCKER_BALANCE_SHEET_EVIDENCE_THIN,
    BLOCKER_RETURNS_DURABILITY_UNKNOWN,
    BLOCKER_REVENUE_DEPENDENCE_UNKNOWN,
}

_STRUCTURAL_BLOCKERS = {
    BLOCKER_IMPAIRMENT_CONFIRMED,
    BLOCKER_PROBABLE_IMPAIRMENT,
    BLOCKER_STRUCTURAL_WEAK_ECONOMICS,
    BLOCKER_STRUCTURALLY_WEAK_ECONOMICS,
    BLOCKER_VALUATION_NO_MOS_CONFIRMED,
    BLOCKER_VALUATION_INTEGRITY_SUSPECT,
    BLOCKER_HIGH_DILUTION,
    BLOCKER_HIGH_LEVERAGE,
    BLOCKER_LOW_OWNER_EARNINGS_QUALITY,
    BLOCKER_FACTS_TERMINAL_MISSING_KEY_INPUTS,
    BLOCKER_FACTS_PROVIDER_NO_DATA,
    BLOCKER_FACTS_PARSE_FAILURE,
}


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _claim(*, value: Any, refs: list[Any], reason_code: str, status: str | None = None) -> dict[str, Any]:
    derived = _dedupe(refs)
    if _is_num(value):
        return {
            "value": float(value),
            "status": str(status or OK).upper(),
            "reason_code": str(reason_code or OK),
            "derived_from": derived,
        }
    token = str(value or "").strip()
    if token and token.upper() != UNKNOWN:
        return {
            "value": token,
            "status": str(status or OK).upper(),
            "reason_code": str(reason_code or OK),
            "derived_from": derived,
        }
    return {
        "value": UNKNOWN,
        "status": str(status or UNKNOWN).upper(),
        "reason_code": str(reason_code or UNKNOWN),
        "derived_from": derived,
    }


def _claim_refs(payload: dict[str, Any], key: str) -> list[str]:
    claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
    claim = claims.get(key) if isinstance(claims, dict) else {}
    if not isinstance(claim, dict):
        return []
    return [str(ref) for ref in (claim.get("derived_from") or []) if str(ref).strip()]


def _coalesce_status(*values: Any, fallback: str = UNKNOWN) -> str:
    for value in values:
        token = str(value or "").strip().upper()
        if token:
            return token
    return fallback


def _payload_refs(*payloads: dict[str, Any], row_refs: list[Any] | None = None) -> list[str]:
    refs: list[Any] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        refs.extend(payload.get("derived_from") or [])
        claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
        for claim in claims.values():
            if isinstance(claim, dict):
                refs.extend(claim.get("derived_from") or [])
    refs.extend(row_refs or [])
    return _dedupe(refs)


def _add_blocker(
    blockers: list[dict[str, Any]],
    code: str,
    *,
    retryable: bool = False,
    structural: bool = False,
) -> None:
    token = str(code or "").strip().upper()
    if not token:
        return
    blockers.append(
        {
            "code": token,
            "retryable": bool(retryable),
            "structural": bool(structural),
            "priority": int(_BLOCKER_PRIORITY.get(token, 99)),
        }
    )


def _facts_blocker_code(
    blocker_class: str,
    *,
    partial_usable: bool,
    terminal: bool,
) -> str:
    token = str(blocker_class or "FACTS_OK").strip().upper()
    if token in {"", "FACTS_OK"}:
        return ""
    if token == BLOCKER_FACTS_PARTIAL_COVERAGE or partial_usable:
        return BLOCKER_FACTS_PARTIAL_COVERAGE
    if token == BLOCKER_FACTS_TERMINAL_MISSING_KEY_INPUTS or terminal:
        return BLOCKER_FACTS_TERMINAL_MISSING_KEY_INPUTS
    if token in {
        BLOCKER_FACTS_RETRYABLE_TIMEOUT,
        BLOCKER_FACTS_RETRYABLE_NETWORK_FAILURE,
        BLOCKER_FACTS_RETRYABLE_RATE_LIMIT,
        BLOCKER_FACTS_NO_CACHE_OFFLINE,
        BLOCKER_FACTS_PROVIDER_NO_DATA,
        BLOCKER_FACTS_PARSE_FAILURE,
    }:
        return token
    return BLOCKER_MISSING_FACTS


def compute_investment_readiness(
    ticker: str,
    as_of_date: str,
    *,
    intrinsic_payload: dict[str, Any] | None = None,
    evidence_sufficiency_payload: dict[str, Any] | None = None,
    valuation_confidence_payload: dict[str, Any] | None = None,
    valuation_integrity_payload: dict[str, Any] | None = None,
    value_type_payload: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    accounting_quality_payload: dict[str, Any] | None = None,
    balance_sheet_stress_payload: dict[str, Any] | None = None,
    returns_persistence_payload: dict[str, Any] | None = None,
    revenue_dependence_payload: dict[str, Any] | None = None,
    cyclical_normalization_payload: dict[str, Any] | None = None,
    impairment_classification_payload: dict[str, Any] | None = None,
    normalization_credibility_payload: dict[str, Any] | None = None,
    capital_allocation_discipline_payload: dict[str, Any] | None = None,
    reinvestment_efficiency_payload: dict[str, Any] | None = None,
    price_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    fcf_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    valuation_status: Any = UNKNOWN,
    facts_blocker_class: str = "FACTS_OK",
    facts_blocker_retryable: bool = False,
    facts_blocker_terminal: bool = False,
    facts_blocker_partial_usable: bool = False,
    facts_missing_key_inputs: list[Any] | None = None,
    facts_retry_recommended: bool = False,
    fail_due_to_missing_evidence: bool = False,
    fail_due_to_economic_weakness: bool = False,
    primary_fail_domain: str | None = None,
    value_gate_status: str | None = None,
    primary_blocker: str | None = None,
    row_derived_from: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg
    intrinsic_payload = intrinsic_payload if isinstance(intrinsic_payload, dict) else {}
    evidence_sufficiency_payload = (
        evidence_sufficiency_payload if isinstance(evidence_sufficiency_payload, dict) else {}
    )
    valuation_confidence_payload = (
        valuation_confidence_payload if isinstance(valuation_confidence_payload, dict) else {}
    )
    valuation_integrity_payload = (
        valuation_integrity_payload if isinstance(valuation_integrity_payload, dict) else {}
    )
    value_type_payload = value_type_payload if isinstance(value_type_payload, dict) else {}
    owner_quality_payload = owner_quality_payload if isinstance(owner_quality_payload, dict) else {}
    intangible_payload = intangible_payload if isinstance(intangible_payload, dict) else {}
    accounting_quality_payload = (
        accounting_quality_payload if isinstance(accounting_quality_payload, dict) else {}
    )
    balance_sheet_stress_payload = (
        balance_sheet_stress_payload if isinstance(balance_sheet_stress_payload, dict) else {}
    )
    returns_persistence_payload = (
        returns_persistence_payload if isinstance(returns_persistence_payload, dict) else {}
    )
    revenue_dependence_payload = (
        revenue_dependence_payload if isinstance(revenue_dependence_payload, dict) else {}
    )
    reinvestment_efficiency_payload = (
        reinvestment_efficiency_payload
        if isinstance(reinvestment_efficiency_payload, dict)
        else {}
    )

    ticker_norm = str(ticker or "").strip().upper()
    gate_status = _coalesce_status(value_gate_status, UNKNOWN)
    primary_fail_domain = _coalesce_status(primary_fail_domain, UNKNOWN)
    primary_blocker = str(primary_blocker or "").strip().upper() or UNKNOWN

    mos_to_floor = intrinsic_payload.get("mos_to_floor", UNKNOWN)
    mos_classification = _coalesce_status(intrinsic_payload.get("mos_classification"), MOS_UNKNOWN)
    downside_support_type = _coalesce_status(intrinsic_payload.get("downside_support_type"), SUPPORT_UNKNOWN)
    normalized_status = _coalesce_status(
        intrinsic_payload.get("normalized_earnings_power_status"),
        UNKNOWN,
    )
    normalized_reason_codes = {
        str(code)
        for code in (intrinsic_payload.get("normalized_earnings_power_reason_codes") or [])
        if str(code).strip()
    }

    support_count = int(valuation_confidence_payload.get("valuation_support_count") or 0)
    valuation_confidence_class = _coalesce_status(
        valuation_confidence_payload.get("valuation_confidence_class"),
        CONFIDENCE_UNKNOWN,
    )
    valuation_fragility_status = _coalesce_status(
        valuation_confidence_payload.get("valuation_fragility_status"),
        UNKNOWN,
    )
    valuation_fragility_reason_codes = {
        str(code)
        for code in (valuation_confidence_payload.get("valuation_fragility_reason_codes") or [])
        if str(code).strip()
    }
    valuation_integrity_class = _coalesce_status(
        valuation_integrity_payload.get("valuation_integrity_class"),
        INTEGRITY_UNKNOWN,
    )
    if not evidence_sufficiency_payload:
        evidence_sufficiency_payload = compute_evidence_sufficiency(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            intrinsic_payload=intrinsic_payload,
            valuation_confidence_payload=valuation_confidence_payload,
            valuation_integrity_payload=valuation_integrity_payload,
            price_status=price_status,
            shares_status=shares_status,
            fcf_status=fcf_status,
            facts_status=facts_status,
            valuation_status=valuation_status,
            facts_blocker_class=facts_blocker_class,
            fail_due_to_missing_evidence=fail_due_to_missing_evidence,
            primary_fail_domain=primary_fail_domain,
            row_derived_from=row_derived_from,
        )
    evidence_sufficiency_class = _coalesce_status(
        evidence_sufficiency_payload.get("evidence_sufficiency_class"),
        SUFFICIENCY_UNKNOWN,
    )
    evidence_sufficiency_reason_codes = {
        str(code)
        for code in (evidence_sufficiency_payload.get("evidence_sufficiency_reason_codes") or [])
        if str(code).strip()
    }
    mos_assessment_status = _coalesce_status(
        evidence_sufficiency_payload.get("mos_assessment_status"),
        MOS_UNKNOWN_STATUS,
    )
    mos_guardrail_reason_codes = {
        str(code)
        for code in (evidence_sufficiency_payload.get("mos_guardrail_reason_codes") or [])
        if str(code).strip()
    }

    value_type_primary = _coalesce_status(
        value_type_payload.get("value_type_primary"),
        VALUE_TYPE_UNKNOWN,
    )
    oe_quality_total = owner_quality_payload.get("oe_quality_total", UNKNOWN)
    oe_quality_reason_codes = {
        str(code)
        for code in (owner_quality_payload.get("oe_quality_reason_codes") or [])
        if str(code).strip()
    }
    accounting_payload_present = bool(accounting_quality_payload)
    accounting_quality_class = _coalesce_status(
        accounting_quality_payload.get("accounting_quality_class"),
        ACCOUNTING_QUALITY_UNKNOWN,
    )
    accounting_quality_reason_codes = {
        str(code)
        for code in (accounting_quality_payload.get("accounting_quality_reason_codes") or [])
        if str(code).strip()
    }
    accounting_support_signals = {
        str(code)
        for code in (accounting_quality_payload.get("cash_earnings_support_signals") or [])
        if str(code).strip()
    }
    accounting_headwind_signals = {
        str(code)
        for code in (accounting_quality_payload.get("cash_earnings_headwind_signals") or [])
        if str(code).strip()
    }
    capital_allocation_score = owner_quality_payload.get("capital_allocation_score", UNKNOWN)
    intangible_total = intangible_payload.get("intangible_economics_total", UNKNOWN)
    cycle_resilience_score = intangible_payload.get("cycle_resilience_score", UNKNOWN)
    owner_value_capture_score = intangible_payload.get("owner_value_capture_score", UNKNOWN)
    owner_value_capture_reason_codes = {
        str(code)
        for code in (intangible_payload.get("owner_value_capture_reason_codes") or [])
        if str(code).strip()
    }
    reinvestment_payload_present = bool(reinvestment_efficiency_payload)
    reinvestment_efficiency_class = _coalesce_status(
        reinvestment_efficiency_payload.get("reinvestment_efficiency_class"),
        "REINVESTMENT_EFFICIENCY_UNKNOWN",
    )
    reinvestment_reason_codes = {
        str(code)
        for code in (reinvestment_efficiency_payload.get("reinvestment_efficiency_reason_codes") or [])
        if str(code).strip()
    }
    reinvestment_support_signals = {
        str(code)
        for code in (reinvestment_efficiency_payload.get("reinvestment_support_signals") or [])
        if str(code).strip()
    }
    reinvestment_headwind_signals = {
        str(code)
        for code in (reinvestment_efficiency_payload.get("reinvestment_headwind_signals") or [])
        if str(code).strip()
    }

    mos_real = mos_assessment_status == MOS_CONFIRMED_PRESENT
    mos_thin = mos_assessment_status == MOS_WEAK
    no_mos_confirmed = mos_assessment_status == MOS_CONFIRMED_ABSENT
    mos_unassessable = mos_assessment_status == MOS_UNASSESSABLE
    multi_support = support_count >= 2
    evidence_gap = (
        fail_due_to_missing_evidence
        or primary_fail_domain == "EVIDENCE"
        or evidence_sufficiency_class == SUFFICIENCY_INSUFFICIENT
        or _coalesce_status(price_status) != OK
        or _coalesce_status(shares_status) != OK
        or _coalesce_status(fcf_status) != OK
        or _coalesce_status(facts_status) != OK
        or str(facts_blocker_class or "FACTS_OK").upper() != "FACTS_OK"
    )
    promising_support = (
        mos_real
        or multi_support
        or downside_support_type in {SUPPORT_ASSET, SUPPORT_EARNINGS, SUPPORT_BALANCE_SHEET}
    )
    quality_positive = (
        (_is_num(oe_quality_total) and float(oe_quality_total) >= 8.0)
        or (_is_num(intangible_total) and float(intangible_total) >= 7.0)
    )
    quality_headwind = (
        (_is_num(oe_quality_total) and float(oe_quality_total) < 5.0)
        or (_is_num(owner_value_capture_score) and float(owner_value_capture_score) <= 1.0)
        or "WEAK_PER_SHARE_CAPTURE" in owner_value_capture_reason_codes
    )
    productive_reinvestment = (
        reinvestment_efficiency_class == "HIGH_REINVESTMENT_EFFICIENCY"
        or "PRODUCTIVE_REINVESTMENT_SUPPORT" in reinvestment_reason_codes
    )
    reinvestment_mixed = (
        reinvestment_efficiency_class == "MODERATE_REINVESTMENT_EFFICIENCY"
        or "REINVESTMENT_MIXED" in reinvestment_reason_codes
    )
    low_reinvestment_efficiency = (
        reinvestment_efficiency_class == "LOW_REINVESTMENT_EFFICIENCY"
        or "CAPITAL_HUNGRY_GROWTH_HEADWIND" in reinvestment_reason_codes
        or "GROWTH_WITHOUT_OWNER_OUTCOME" in reinvestment_reason_codes
    )
    reinvestment_evidence_thin = reinvestment_payload_present and (
        reinvestment_efficiency_class == "REINVESTMENT_EFFICIENCY_UNKNOWN"
        or "REINVESTMENT_EVIDENCE_THIN" in reinvestment_reason_codes
    )
    high_accounting_quality = (
        accounting_payload_present and accounting_quality_class == "HIGH_ACCOUNTING_QUALITY"
    )
    low_accounting_quality = (
        accounting_payload_present and accounting_quality_class == LOW_ACCOUNTING_QUALITY
    )
    accounting_evidence_thin = (
        accounting_payload_present and accounting_quality_class == ACCOUNTING_QUALITY_UNKNOWN
    )
    balance_sheet_payload_present = bool(balance_sheet_stress_payload)
    balance_sheet_stress_class = _coalesce_status(
        balance_sheet_stress_payload.get("balance_sheet_stress_class"),
        BALANCE_SHEET_STRESS_UNKNOWN,
    )
    refinancing_risk_class = _coalesce_status(
        balance_sheet_stress_payload.get("refinancing_risk_class"),
        "REFINANCING_RISK_UNKNOWN",
    )
    balance_sheet_stress_reason_codes = {
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_stress_reason_codes") or [])
        if str(code).strip()
    }
    refinancing_risk_reason_codes = {
        str(code)
        for code in (balance_sheet_stress_payload.get("refinancing_risk_reason_codes") or [])
        if str(code).strip()
    }
    balance_sheet_support_signals = {
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_support_signals") or [])
        if str(code).strip()
    }
    balance_sheet_headwind_signals = {
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_headwind_signals") or [])
        if str(code).strip()
    }
    low_balance_sheet_stress = (
        balance_sheet_payload_present and balance_sheet_stress_class == LOW_BALANCE_SHEET_STRESS
    )
    high_balance_sheet_stress = (
        balance_sheet_payload_present and balance_sheet_stress_class == HIGH_BALANCE_SHEET_STRESS
    )
    high_refinancing_risk = (
        balance_sheet_payload_present and refinancing_risk_class == "HIGH_REFINANCING_RISK"
    )
    balance_sheet_evidence_thin = (
        balance_sheet_payload_present and balance_sheet_stress_class == BALANCE_SHEET_STRESS_UNKNOWN
    )
    returns_payload_present = bool(returns_persistence_payload)
    returns_persistence_class = _coalesce_status(
        returns_persistence_payload.get("returns_persistence_class"),
        RETURNS_PERSISTENCE_UNKNOWN,
    )
    returns_persistence_reason_codes = {
        str(code)
        for code in (returns_persistence_payload.get("returns_persistence_reason_codes") or [])
        if str(code).strip()
    }
    returns_support_signals = {
        str(code)
        for code in (returns_persistence_payload.get("returns_support_signals") or [])
        if str(code).strip()
    }
    returns_headwind_signals = {
        str(code)
        for code in (returns_persistence_payload.get("returns_headwind_signals") or [])
        if str(code).strip()
    }
    high_returns_persistence = (
        returns_payload_present and returns_persistence_class == HIGH_RETURNS_PERSISTENCE
    )
    low_returns_persistence = (
        returns_payload_present and returns_persistence_class == LOW_RETURNS_PERSISTENCE
    )
    returns_durability_unknown = (
        returns_payload_present and returns_persistence_class == RETURNS_PERSISTENCE_UNKNOWN
    )
    revenue_payload_present = bool(revenue_dependence_payload)
    revenue_dependence_risk_class = _coalesce_status(
        revenue_dependence_payload.get("revenue_dependence_risk_class"),
        REVENUE_DEPENDENCE_UNKNOWN,
    )
    revenue_dependence_reason_codes = {
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_risk_reason_codes") or [])
        if str(code).strip()
    }
    revenue_dependence_support_signals = {
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_support_signals") or [])
        if str(code).strip()
    }
    revenue_dependence_headwind_signals = {
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_headwind_signals") or [])
        if str(code).strip()
    }
    low_revenue_dependence = (
        revenue_payload_present and revenue_dependence_risk_class == LOW_REVENUE_DEPENDENCE_RISK
    )
    high_revenue_dependence = (
        revenue_payload_present and revenue_dependence_risk_class == HIGH_REVENUE_DEPENDENCE_RISK
    )
    revenue_dependence_unknown = (
        revenue_payload_present and revenue_dependence_risk_class == REVENUE_DEPENDENCE_UNKNOWN
    )
    high_dilution = (
        "EXCESS_DILUTION" in oe_quality_reason_codes
        or "EXCESS_DILUTION" in owner_value_capture_reason_codes
    )
    high_leverage = (
        "DEBT_ACCUMULATION" in oe_quality_reason_codes
        or (_is_num(capital_allocation_score) and float(capital_allocation_score) <= 1.0)
    )
    cyclical_low_confidence = (
        value_type_primary == VALUE_TYPE_CYCLICAL
        and (
            normalized_status in {"LOW_CONFIDENCE", UNKNOWN}
            or REASON_CYCLICAL_NORMALIZATION_LOW_CONFIDENCE in normalized_reason_codes
            or (_is_num(cycle_resilience_score) and float(cycle_resilience_score) >= 3.0)
        )
    )
    _cyclical_payload = cyclical_normalization_payload if isinstance(cyclical_normalization_payload, dict) else {}
    _cyclical_risk = str(_cyclical_payload.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN")
    peak_earnings_risk = _cyclical_risk == "PEAK_EARNINGS_RISK"

    blockers: list[dict[str, Any]] = []
    if fail_due_to_economic_weakness or primary_fail_domain == "ECONOMICS":
        _add_blocker(blockers, BLOCKER_STRUCTURAL_WEAK_ECONOMICS, structural=True)
    if valuation_integrity_class == INTEGRITY_SUSPECT:
        _add_blocker(blockers, BLOCKER_VALUATION_INTEGRITY_SUSPECT, structural=True)
    elif valuation_integrity_class == INTEGRITY_WARNING:
        _add_blocker(blockers, BLOCKER_VALUATION_INTEGRITY_WARNING)
    if no_mos_confirmed:
        _add_blocker(blockers, BLOCKER_VALUATION_NO_MOS_CONFIRMED, structural=True)
    elif mos_unassessable:
        mos_gap_retryable = (
            evidence_sufficiency_class == SUFFICIENCY_INSUFFICIENT
            or REASON_MOS_BLOCKED_BY_MISSING_PRICE in mos_guardrail_reason_codes
            or REASON_MOS_BLOCKED_BY_MISSING_FACTS in mos_guardrail_reason_codes
            or REASON_MOS_BLOCKED_BY_MISSING_SHARES in mos_guardrail_reason_codes
        )
        if REASON_MOS_BLOCKED_BY_MISSING_PRICE in mos_guardrail_reason_codes:
            _add_blocker(blockers, BLOCKER_MOS_UNASSESSABLE_MISSING_PRICE, retryable=True)
        if REASON_MOS_BLOCKED_BY_MISSING_FACTS in mos_guardrail_reason_codes:
            _add_blocker(blockers, BLOCKER_MOS_UNASSESSABLE_MISSING_FACTS, retryable=True)
        if REASON_MOS_BLOCKED_BY_MISSING_SHARES in mos_guardrail_reason_codes:
            _add_blocker(blockers, BLOCKER_MOS_UNASSESSABLE_MISSING_SHARES, retryable=True)
        _add_blocker(
            blockers,
            BLOCKER_EVIDENCE_INSUFFICIENT_FOR_MOS,
            retryable=mos_gap_retryable,
            structural=False,
        )
    if valuation_fragility_status == FRAGILITY_HIGH or REASON_SINGLE_SUPPORT_ONLY in valuation_fragility_reason_codes:
        _add_blocker(blockers, BLOCKER_VALUATION_FRAGILE)
    if valuation_confidence_class in {CONFIDENCE_LOW, CONFIDENCE_UNKNOWN}:
        _add_blocker(blockers, BLOCKER_LOW_CONFIDENCE_VALUE)
    if _coalesce_status(price_status) != OK:
        _add_blocker(blockers, BLOCKER_MISSING_PRICE, retryable=True)
    facts_code = _facts_blocker_code(
        str(facts_blocker_class or "FACTS_OK"),
        partial_usable=bool(facts_blocker_partial_usable),
        terminal=bool(facts_blocker_terminal),
    )
    if facts_code:
        _add_blocker(
            blockers,
            facts_code,
            retryable=bool(facts_blocker_retryable or facts_retry_recommended or facts_code in _RETRYABLE_BLOCKERS),
            structural=bool(facts_blocker_terminal or facts_code in _STRUCTURAL_BLOCKERS),
        )
    elif _coalesce_status(facts_status) != OK:
        _add_blocker(blockers, BLOCKER_MISSING_FACTS, retryable=True)
    if _coalesce_status(shares_status) != OK:
        _add_blocker(blockers, BLOCKER_MISSING_SHARES, retryable=True)
    if _coalesce_status(fcf_status) != OK:
        _add_blocker(blockers, BLOCKER_MISSING_FCF, retryable=True)
    if quality_headwind:
        _add_blocker(blockers, BLOCKER_LOW_OWNER_EARNINGS_QUALITY, structural=True)
    if high_dilution:
        _add_blocker(blockers, BLOCKER_HIGH_DILUTION, structural=True)
    if high_leverage:
        _add_blocker(blockers, BLOCKER_HIGH_LEVERAGE, structural=True)
    if low_reinvestment_efficiency:
        _add_blocker(blockers, BLOCKER_LOW_REINVESTMENT_EFFICIENCY)
    if reinvestment_evidence_thin:
        _add_blocker(blockers, BLOCKER_REINVESTMENT_EVIDENCE_THIN, retryable=True)
    if low_accounting_quality:
        _add_blocker(blockers, BLOCKER_LOW_ACCOUNTING_QUALITY)
    if accounting_evidence_thin:
        _add_blocker(blockers, BLOCKER_ACCOUNTING_EVIDENCE_THIN, retryable=True)
    if high_balance_sheet_stress:
        _add_blocker(
            blockers,
            BLOCKER_HIGH_BALANCE_SHEET_STRESS,
            structural="CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in balance_sheet_headwind_signals,
        )
    if high_refinancing_risk:
        _add_blocker(blockers, BLOCKER_HIGH_REFINANCING_RISK)
    if balance_sheet_evidence_thin:
        _add_blocker(blockers, BLOCKER_BALANCE_SHEET_EVIDENCE_THIN, retryable=True)
    if low_returns_persistence:
        _add_blocker(blockers, BLOCKER_LOW_RETURNS_PERSISTENCE)
    if returns_durability_unknown:
        _add_blocker(blockers, BLOCKER_RETURNS_DURABILITY_UNKNOWN, retryable=True)
    if high_revenue_dependence:
        _add_blocker(blockers, BLOCKER_HIGH_REVENUE_DEPENDENCE_RISK)
    if revenue_dependence_unknown:
        _add_blocker(blockers, BLOCKER_REVENUE_DEPENDENCE_UNKNOWN, retryable=True)
    if value_type_primary == VALUE_TYPE_UNKNOWN:
        _add_blocker(blockers, BLOCKER_UNKNOWN_VALUE_TYPE)
    if cyclical_low_confidence:
        _add_blocker(blockers, BLOCKER_CYCLICAL_LOW_CONFIDENCE)
    if peak_earnings_risk:
        _add_blocker(blockers, BLOCKER_PEAK_EARNINGS_CYCLICAL_RISK)

    # Impairment classification integration
    _impairment_payload = (
        impairment_classification_payload
        if isinstance(impairment_classification_payload, dict)
        else {}
    )
    _impairment_class = str(_impairment_payload.get("impairment_class_primary") or "").upper()
    _impairment_is_clear = _impairment_class == "CLEAR_IMPAIRMENT"
    _impairment_is_probable = _impairment_class == "PROBABLE_IMPAIRMENT"
    _impairment_is_temp_weakness = _impairment_class == "TEMPORARY_WEAKNESS"
    _impairment_is_evidence_degraded = _impairment_class == "EVIDENCE_DEGRADED_NOT_ASSESSABLE"
    _impairment_is_structurally_weak = _impairment_class == "STRUCTURALLY_WEAK_NOT_IMPAIRED"
    if _impairment_is_clear:
        _add_blocker(blockers, BLOCKER_IMPAIRMENT_CONFIRMED, structural=True)
    elif _impairment_is_probable:
        _add_blocker(blockers, BLOCKER_PROBABLE_IMPAIRMENT, structural=True)
    elif _impairment_is_evidence_degraded:
        _add_blocker(blockers, BLOCKER_EVIDENCE_GAP_PREVENTS_IMPAIRMENT_JUDGMENT, retryable=True)
    elif _impairment_is_structurally_weak:
        _add_blocker(blockers, BLOCKER_STRUCTURALLY_WEAK_ECONOMICS, structural=True)

    blocker_map: dict[str, dict[str, Any]] = {}
    for blocker in blockers:
        code = blocker["code"]
        existing = blocker_map.get(code)
        if existing is None:
            blocker_map[code] = blocker
            continue
        existing["retryable"] = bool(existing["retryable"] or blocker["retryable"])
        existing["structural"] = bool(existing["structural"] or blocker["structural"])
        existing["priority"] = min(int(existing["priority"]), int(blocker["priority"]))
    blocker_rows = sorted(
        blocker_map.values(),
        key=lambda item: (int(item["priority"]), str(item["code"])),
    )
    blocker_stack_all = [str(item["code"]) for item in blocker_rows]
    blocker_stack_primary = blocker_stack_all[0] if blocker_stack_all else UNKNOWN
    blocker_stack_secondary = blocker_stack_all[1] if len(blocker_stack_all) > 1 else None
    blocker_stack_retryable = any(bool(item["retryable"]) for item in blocker_rows)
    blocker_stack_structural = any(bool(item["structural"]) for item in blocker_rows)

    # Normalization credibility integration
    _norm_cred_payload = (
        normalization_credibility_payload
        if isinstance(normalization_credibility_payload, dict)
        else {}
    )
    _norm_cred_class = str(_norm_cred_payload.get("normalization_credibility_class") or "").upper()
    _norm_cred_high = _norm_cred_class == "HIGH_NORMALIZATION_CREDIBILITY"
    _norm_cred_moderate = _norm_cred_class == "MODERATE_NORMALIZATION_CREDIBILITY"
    _norm_cred_low = _norm_cred_class == "LOW_NORMALIZATION_CREDIBILITY"
    _norm_cred_blocked = str(_norm_cred_payload.get("primary_normalization_caution") or "").upper() in {
        "NORMALIZATION_BLOCKED_BY_EVIDENCE",
    }

    # Capital allocation discipline integration
    _cap_alloc_payload = (
        capital_allocation_discipline_payload
        if isinstance(capital_allocation_discipline_payload, dict)
        else {}
    )
    _cap_alloc_class = str(_cap_alloc_payload.get("capital_allocation_discipline_class") or "").upper()
    _cap_alloc_friendly = _cap_alloc_class == "OWNER_FRIENDLY_DISCIPLINED"
    _cap_alloc_dilutive = _cap_alloc_class == "OWNER_DILUTIVE_OR_DESTRUCTIVE"
    _cap_alloc_mixed = _cap_alloc_class == "MIXED_CAPITAL_ALLOCATION"

    readiness_support_present = _dedupe(
        [
            SUPPORT_ADEQUATE_MOS if mos_real else "",
            SUPPORT_MULTI_SUPPORT if multi_support else "",
            SUPPORT_LOW_FRAGILITY if valuation_fragility_status == FRAGILITY_LOW else "",
            SUPPORT_INTEGRITY_OK if valuation_integrity_class == INTEGRITY_OK else "",
            SUPPORT_EARNINGS_POWER if downside_support_type == SUPPORT_EARNINGS else "",
            SUPPORT_ASSET if downside_support_type == SUPPORT_ASSET else "",
            SUPPORT_BALANCE if downside_support_type == SUPPORT_BALANCE_SHEET else "",
            SUPPORT_HIGH_OE_QUALITY if (_is_num(oe_quality_total) and float(oe_quality_total) >= 8.0) else "",
            SUPPORT_HIGH_INTANGIBLE if (_is_num(intangible_total) and float(intangible_total) >= 7.0) else "",
            SUPPORT_TEMPORARY_WEAKNESS_WITH_SUPPORT if _impairment_is_temp_weakness else "",
            SUPPORT_PRODUCTIVE_REINVESTMENT if productive_reinvestment else "",
            SUPPORT_HIGH_ACCOUNTING_QUALITY if high_accounting_quality else "",
            SUPPORT_LOW_BALANCE_SHEET_STRESS if low_balance_sheet_stress else "",
            SUPPORT_HIGH_RETURNS_PERSISTENCE if high_returns_persistence else "",
            SUPPORT_LOW_REVENUE_DEPENDENCE if low_revenue_dependence else "",
            "HIGH_NORMALIZATION_CREDIBILITY_SUPPORT" if _norm_cred_high else "",
            "OWNER_FRIENDLY_CAPITAL_ALLOCATION_SUPPORT" if _cap_alloc_friendly else "",
        ]
    )
    readiness_support_missing = _dedupe(
        [
            HEADWIND_MOS_UNKNOWN if mos_assessment_status == MOS_UNKNOWN_STATUS else "",
            HEADWIND_MOS_UNASSESSABLE if mos_unassessable else "",
            REASON_SINGLE_SUPPORT_ONLY if support_count <= 1 else "",
            HEADWIND_FACTS_MISSING if evidence_gap else "",
            HEADWIND_INTEGRITY_UNKNOWN if valuation_integrity_class == INTEGRITY_UNKNOWN else "",
            BLOCKER_UNKNOWN_VALUE_TYPE if value_type_primary == VALUE_TYPE_UNKNOWN else "",
            HEADWIND_REINVESTMENT_EVIDENCE_THIN if reinvestment_evidence_thin else "",
            BLOCKER_ACCOUNTING_EVIDENCE_THIN if accounting_evidence_thin else "",
            BLOCKER_BALANCE_SHEET_EVIDENCE_THIN if balance_sheet_evidence_thin else "",
            BLOCKER_RETURNS_DURABILITY_UNKNOWN if returns_durability_unknown else "",
            BLOCKER_REVENUE_DEPENDENCE_UNKNOWN if revenue_dependence_unknown else "",
        ]
    )
    readiness_support_headwinds = _dedupe(
        [
            BLOCKER_VALUATION_NO_MOS_CONFIRMED if no_mos_confirmed else "",
            BLOCKER_EVIDENCE_INSUFFICIENT_FOR_MOS if mos_unassessable else "",
            BLOCKER_LOW_CONFIDENCE_VALUE if valuation_confidence_class == CONFIDENCE_LOW else "",
            BLOCKER_VALUATION_INTEGRITY_WARNING if valuation_integrity_class == INTEGRITY_WARNING else "",
            BLOCKER_VALUATION_INTEGRITY_SUSPECT if valuation_integrity_class == INTEGRITY_SUSPECT else "",
            HEADWIND_QUALITY if quality_headwind else "",
            HEADWIND_FRAGILE_VALUE if value_type_primary == VALUE_TYPE_FRAGILE else "",
            BLOCKER_CYCLICAL_LOW_CONFIDENCE if cyclical_low_confidence else "",
            BLOCKER_PEAK_EARNINGS_CYCLICAL_RISK if peak_earnings_risk else "",
            HEADWIND_IMPAIRMENT_CONFIRMED if _impairment_is_clear else "",
            HEADWIND_PROBABLE_IMPAIRMENT if _impairment_is_probable else "",
            HEADWIND_STRUCTURALLY_WEAK if _impairment_is_structurally_weak else "",
            HEADWIND_EVIDENCE_GAP_IMPAIRMENT if _impairment_is_evidence_degraded else "",
            "LOW_NORMALIZATION_CREDIBILITY_HEADWIND" if _norm_cred_low and _impairment_is_temp_weakness else "",
            "NORMALIZATION_BLOCKED_BY_EVIDENCE" if _norm_cred_blocked else "",
            "DILUTION_DESTROYS_OWNER_VALUE" if _cap_alloc_dilutive else "",
            "PER_SHARE_VALUE_CAPTURE_WEAK" if _cap_alloc_mixed else "",
            HEADWIND_REINVESTMENT_MIXED if reinvestment_mixed else "",
            HEADWIND_CAPITAL_HUNGRY_GROWTH
            if low_reinvestment_efficiency or "CAPITAL_HUNGRY_GROWTH" in reinvestment_headwind_signals
            else "",
            HEADWIND_GROWTH_WITHOUT_OWNER_OUTCOME
            if (
                low_reinvestment_efficiency
                or "REVENUE_GROWTH_WITHOUT_OWNER_OUTCOME" in reinvestment_headwind_signals
            )
            else "",
            HEADWIND_LOW_ACCOUNTING_QUALITY if low_accounting_quality else "",
            HEADWIND_ACCRUAL_HEAVY
            if (
                low_accounting_quality
                and (
                    "ACCRUAL_HEAVY_EARNINGS" in accounting_headwind_signals
                    or "ACCRUAL_HEAVY_EARNINGS_HEADWIND" in accounting_quality_reason_codes
                )
            )
            else "",
            HEADWIND_WEAK_CASH_CONVERSION
            if (
                low_accounting_quality
                and (
                    {
                        "WEAK_CFO_TO_EARNINGS_CONVERSION",
                        "WEAK_FCF_TO_EARNINGS_CONVERSION",
                        "CASH_EARNINGS_DIVERGENCE",
                    }
                    & accounting_headwind_signals
                    or "WEAK_CASH_CONVERSION_HEADWIND" in accounting_quality_reason_codes
                )
            )
            else "",
            HEADWIND_HIGH_BALANCE_SHEET_STRESS if high_balance_sheet_stress else "",
            HEADWIND_HIGH_REFINANCING_RISK if high_refinancing_risk else "",
            HEADWIND_CAPITAL_STRUCTURE_DOMINATES
            if "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in balance_sheet_headwind_signals
            else "",
            HEADWIND_LOW_RETURNS_PERSISTENCE if low_returns_persistence else "",
            HEADWIND_INCREMENTAL_RETURNS_DETERIORATION
            if "INCREMENTAL_RETURNS_DETERIORATING" in returns_headwind_signals
            else "",
            HEADWIND_RETURNS_DURABILITY_MIXED
            if (
                returns_payload_present
                and returns_persistence_class == "MODERATE_RETURNS_PERSISTENCE"
            )
            else "",
            HEADWIND_RETURNS_DURABILITY_UNKNOWN if returns_durability_unknown else "",
            HEADWIND_HIGH_REVENUE_DEPENDENCE if high_revenue_dependence else "",
            HEADWIND_CUSTOMER_CONCENTRATION
            if (
                high_revenue_dependence
                and (
                    "SINGLE_CUSTOMER_CONCENTRATION" in revenue_dependence_headwind_signals
                    or "TOP_CUSTOMER_DOMINANCE" in revenue_dependence_headwind_signals
                    or REASON_CUSTOMER_CONCENTRATION_HEADWIND in revenue_dependence_reason_codes
                )
            )
            else "",
            HEADWIND_CHANNEL_DEPENDENCE
            if (
                high_revenue_dependence
                and (
                    "NARROW_CHANNEL_DEPENDENCE" in revenue_dependence_headwind_signals
                    or REASON_CHANNEL_DEPENDENCE_HEADWIND in revenue_dependence_reason_codes
                )
            )
            else "",
            HEADWIND_REVENUE_DEPENDENCE_UNKNOWN if revenue_dependence_unknown else "",
        ]
    )

    ready_now = (
        gate_status in {"PASS", "WATCH"}
        and mos_real
        and multi_support
        and valuation_confidence_class in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}
        and valuation_fragility_status in {FRAGILITY_LOW, FRAGILITY_MODERATE}
        and valuation_integrity_class == INTEGRITY_OK
        and not low_accounting_quality
        and not high_balance_sheet_stress
        and not high_refinancing_risk
        and not low_returns_persistence
        and not high_revenue_dependence
        and not blocker_stack_structural
        and not blocker_stack_retryable
    )
    structural_not_investable = (
        blocker_stack_structural
        or valuation_integrity_class == INTEGRITY_SUSPECT
        or fail_due_to_economic_weakness
        or primary_fail_domain == "ECONOMICS"
        or (no_mos_confirmed and valuation_confidence_class in {CONFIDENCE_LOW, CONFIDENCE_UNKNOWN})
        or _impairment_is_clear
        or _impairment_is_probable
        or (
            high_balance_sheet_stress
            and "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in balance_sheet_headwind_signals
        )
        or high_revenue_dependence
    )

    if ready_now:
        readiness_class = READY_INVESTABLE_NOW
    elif structural_not_investable:
        readiness_class = READY_NOT_INVESTABLE
    elif evidence_gap and not promising_support and not fail_due_to_economic_weakness:
        readiness_class = READY_UNKNOWN
    elif promising_support and (
        blocker_stack_retryable
        or facts_blocker_partial_usable
        or gate_status in {"PASS", "WATCH"}
        and (
            valuation_integrity_class in {INTEGRITY_WARNING, INTEGRITY_UNKNOWN}
            or valuation_confidence_class in {CONFIDENCE_MEDIUM, CONFIDENCE_UNKNOWN}
            or valuation_fragility_status == FRAGILITY_MODERATE
            or mos_thin
        )
    ):
        readiness_class = READY_RESEARCH_WORTHY
    elif gate_status in {"PASS", "WATCH"} or promising_support or quality_positive:
        readiness_class = READY_WATCH_ONLY
    elif evidence_gap:
        readiness_class = READY_UNKNOWN
    else:
        readiness_class = READY_NOT_INVESTABLE

    if readiness_class == READY_INVESTABLE_NOW:
        readiness_reason_codes = [REASON_READY_WITH_REAL_SUPPORT]
    elif readiness_class == READY_RESEARCH_WORTHY:
        readiness_reason_codes = [REASON_BLOCKED_BUT_RESEARCH_WORTHY]
    elif readiness_class == READY_WATCH_ONLY:
        readiness_reason_codes = [REASON_WATCH_WITH_THIN_EDGE]
    elif readiness_class == READY_NOT_INVESTABLE:
        readiness_reason_codes = [REASON_NOT_INVESTABLE_STRUCTURAL]
    else:
        readiness_reason_codes = [REASON_READINESS_UNKNOWN_EVIDENCE_GAP]
    readiness_reason_codes = _dedupe(
        readiness_reason_codes
        + ([blocker_stack_primary] if blocker_stack_primary != UNKNOWN else [])
        + readiness_support_present[:2]
        + readiness_support_missing[:2]
        + readiness_support_headwinds[:2]
    )

    if readiness_class == READY_INVESTABLE_NOW:
        primary_next_step = NEXT_DEEPER_UNDERWRITING
        primary_next_step_reason = REASON_READY_WITH_REAL_SUPPORT
    elif readiness_class == READY_NOT_INVESTABLE:
        primary_next_step = NEXT_DO_NOT_ADVANCE
        primary_next_step_reason = blocker_stack_primary if blocker_stack_primary != UNKNOWN else REASON_NOT_INVESTABLE_STRUCTURAL
    elif blocker_stack_retryable:
        primary_next_step = NEXT_CLEAR_EVIDENCE
        primary_next_step_reason = blocker_stack_primary
    elif readiness_class == READY_RESEARCH_WORTHY:
        primary_next_step = NEXT_DEEPER_UNDERWRITING
        primary_next_step_reason = REASON_BLOCKED_BUT_RESEARCH_WORTHY
    elif readiness_class == READY_WATCH_ONLY:
        primary_next_step = NEXT_MONITOR_ONLY
        primary_next_step_reason = blocker_stack_primary if blocker_stack_primary != UNKNOWN else REASON_WATCH_WITH_THIN_EDGE
    else:
        primary_next_step = NEXT_UNKNOWN
        primary_next_step_reason = (
            blocker_stack_primary if blocker_stack_primary != UNKNOWN else REASON_READINESS_UNKNOWN_EVIDENCE_GAP
        )

    derived_from = _payload_refs(
        intrinsic_payload,
        evidence_sufficiency_payload,
        valuation_confidence_payload,
        valuation_integrity_payload,
        value_type_payload,
        owner_quality_payload,
        intangible_payload,
        accounting_quality_payload,
        balance_sheet_stress_payload,
        returns_persistence_payload,
        revenue_dependence_payload,
        reinvestment_efficiency_payload,
        row_refs=row_derived_from,
    )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "investment_readiness_class": readiness_class,
        "investment_readiness_reason_codes": readiness_reason_codes,
        "evidence_sufficiency_class": evidence_sufficiency_class,
        "evidence_sufficiency_reason_codes": sorted(evidence_sufficiency_reason_codes),
        "mos_assessment_status": mos_assessment_status,
        "mos_guardrail_reason_codes": sorted(mos_guardrail_reason_codes),
        "blocker_stack_primary": blocker_stack_primary,
        "blocker_stack_secondary": blocker_stack_secondary,
        "blocker_stack_all": blocker_stack_all,
        "blocker_stack_retryable": blocker_stack_retryable,
        "blocker_stack_structural": blocker_stack_structural,
        "readiness_support_present": readiness_support_present,
        "readiness_support_missing": readiness_support_missing,
        "readiness_support_headwinds": readiness_support_headwinds,
        "reinvestment_efficiency_class": reinvestment_efficiency_class,
        "reinvestment_efficiency_reason_codes": sorted(reinvestment_reason_codes),
        "reinvestment_support_signals": sorted(reinvestment_support_signals),
        "reinvestment_headwind_signals": sorted(reinvestment_headwind_signals),
        "primary_reinvestment_caution": str(
            reinvestment_efficiency_payload.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
        ),
        "accounting_quality_class": accounting_quality_class,
        "accounting_quality_reason_codes": sorted(accounting_quality_reason_codes),
        "cash_earnings_support_signals": sorted(accounting_support_signals),
        "cash_earnings_headwind_signals": sorted(accounting_headwind_signals),
        "primary_accounting_caution": str(
            accounting_quality_payload.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
        ),
        "cash_earnings_discipline_summary": str(
            accounting_quality_payload.get("cash_earnings_discipline_summary") or ""
        ),
        "balance_sheet_stress_class": balance_sheet_stress_class,
        "balance_sheet_stress_reason_codes": sorted(balance_sheet_stress_reason_codes),
        "refinancing_risk_class": refinancing_risk_class,
        "refinancing_risk_reason_codes": sorted(refinancing_risk_reason_codes),
        "balance_sheet_support_signals": sorted(balance_sheet_support_signals),
        "balance_sheet_headwind_signals": sorted(balance_sheet_headwind_signals),
        "primary_balance_sheet_caution": str(
            balance_sheet_stress_payload.get("primary_balance_sheet_caution") or "BALANCE_SHEET_UNCLEAR"
        ),
        "balance_sheet_discipline_summary": str(
            balance_sheet_stress_payload.get("balance_sheet_discipline_summary") or ""
        ),
        "returns_persistence_class": returns_persistence_class,
        "returns_persistence_reason_codes": sorted(returns_persistence_reason_codes),
        "returns_support_signals": sorted(returns_support_signals),
        "returns_headwind_signals": sorted(returns_headwind_signals),
        "primary_returns_caution": str(
            returns_persistence_payload.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
        ),
        "economic_durability_summary": str(
            returns_persistence_payload.get("economic_durability_summary") or ""
        ),
        "revenue_dependence_risk_class": revenue_dependence_risk_class,
        "revenue_dependence_risk_reason_codes": sorted(revenue_dependence_reason_codes),
        "revenue_dependence_support_signals": sorted(revenue_dependence_support_signals),
        "revenue_dependence_headwind_signals": sorted(revenue_dependence_headwind_signals),
        "primary_revenue_dependence_caution": str(
            revenue_dependence_payload.get("primary_revenue_dependence_caution") or "REVENUE_BASE_UNCLEAR"
        ),
        "revenue_fragility_summary": str(
            revenue_dependence_payload.get("revenue_fragility_summary") or ""
        ),
        "primary_next_step": primary_next_step,
        "primary_next_step_reason": primary_next_step_reason,
        "facts_missing_key_inputs": [
            str(value) for value in (facts_missing_key_inputs or []) if str(value).strip()
        ],
        "derived_from": derived_from,
        "claims": {
            "investment_readiness_class": _claim(
                value=readiness_class,
                refs=derived_from,
                reason_code=readiness_reason_codes[0] if readiness_reason_codes else UNKNOWN,
                status=OK if readiness_class != READY_UNKNOWN else UNKNOWN,
            ),
            "primary_next_step": _claim(
                value=primary_next_step,
                refs=derived_from,
                reason_code=primary_next_step_reason,
                status=OK if primary_next_step != NEXT_UNKNOWN else UNKNOWN,
            ),
        },
        "generated_at": utc_now_iso(),
    }


def write_investment_readiness_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("investment_readiness_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("investment_readiness_detail"), dict)
    }
    rows: list[dict[str, Any]] = []

    for ticker in sorted({str(token or "").strip().upper() for token in tickers if str(token or "").strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            score_row = next(
                (
                    row
                    for row in (scoreboard_rows or [])
                    if isinstance(row, dict) and str(row.get("ticker") or "").strip().upper() == ticker
                ),
                {},
            )
            detail = compute_investment_readiness(
                ticker=ticker,
                as_of_date=as_of_date,
                intrinsic_payload=score_row.get("intrinsic_discipline_detail")
                if isinstance(score_row.get("intrinsic_discipline_detail"), dict)
                else {},
                valuation_confidence_payload=score_row.get("valuation_confidence_detail")
                if isinstance(score_row.get("valuation_confidence_detail"), dict)
                else {},
                valuation_integrity_payload=score_row.get("valuation_integrity_detail")
                if isinstance(score_row.get("valuation_integrity_detail"), dict)
                else {},
                evidence_sufficiency_payload=score_row.get("evidence_sufficiency_detail")
                if isinstance(score_row.get("evidence_sufficiency_detail"), dict)
                else {},
                value_type_payload=score_row.get("value_type_detail")
                if isinstance(score_row.get("value_type_detail"), dict)
                else {},
                owner_quality_payload=score_row.get("owner_earnings_quality_detail")
                if isinstance(score_row.get("owner_earnings_quality_detail"), dict)
                else {},
                intangible_payload=score_row.get("intangible_economics_detail")
                if isinstance(score_row.get("intangible_economics_detail"), dict)
                else {},
                accounting_quality_payload=score_row.get("accounting_quality_detail")
                if isinstance(score_row.get("accounting_quality_detail"), dict)
                else {},
                balance_sheet_stress_payload=score_row.get("balance_sheet_stress_detail")
                if isinstance(score_row.get("balance_sheet_stress_detail"), dict)
                else {},
                returns_persistence_payload=score_row.get("returns_persistence_detail")
                if isinstance(score_row.get("returns_persistence_detail"), dict)
                else {},
                revenue_dependence_payload=score_row.get("revenue_dependence_detail")
                if isinstance(score_row.get("revenue_dependence_detail"), dict)
                else {},
                impairment_classification_payload=score_row.get("impairment_classification_detail")
                if isinstance(score_row.get("impairment_classification_detail"), dict)
                else {},
                normalization_credibility_payload=score_row.get("normalization_credibility_detail")
                if isinstance(score_row.get("normalization_credibility_detail"), dict)
                else {},
                capital_allocation_discipline_payload=score_row.get("capital_allocation_discipline_detail")
                if isinstance(score_row.get("capital_allocation_discipline_detail"), dict)
                else {},
                reinvestment_efficiency_payload=score_row.get("reinvestment_efficiency_detail")
                if isinstance(score_row.get("reinvestment_efficiency_detail"), dict)
                else {},
                price_status=score_row.get("price_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                fcf_status=score_row.get("fcf_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                valuation_status=score_row.get("valuation_status", UNKNOWN),
                facts_blocker_class=str(score_row.get("facts_blocker_class") or "FACTS_OK"),
                facts_blocker_retryable=bool(score_row.get("facts_blocker_retryable", False)),
                facts_blocker_terminal=bool(score_row.get("facts_blocker_terminal", False)),
                facts_blocker_partial_usable=bool(score_row.get("facts_blocker_partial_usable", False)),
                facts_missing_key_inputs=list(score_row.get("facts_missing_key_inputs") or []),
                facts_retry_recommended=bool(score_row.get("facts_retry_recommended", False)),
                fail_due_to_missing_evidence=bool(score_row.get("fail_due_to_missing_evidence", False)),
                fail_due_to_economic_weakness=bool(score_row.get("fail_due_to_economic_weakness", False)),
                primary_fail_domain=str(score_row.get("primary_fail_domain") or UNKNOWN),
                value_gate_status=str(score_row.get("scout_status") or score_row.get("value_gate_status") or UNKNOWN),
                primary_blocker=str(score_row.get("primary_blocker") or UNKNOWN),
                row_derived_from=list(score_row.get("derived_from") or []),
            )
        rows.append(detail)

    counts_by_class: dict[str, int] = {}
    primary_blocker_counts: dict[str, int] = {}
    next_step_counts: dict[str, int] = {}
    retryable_blocker_count = 0
    structural_blocker_count = 0
    for row in rows:
        readiness_class = str(row.get("investment_readiness_class") or READY_UNKNOWN)
        counts_by_class[readiness_class] = counts_by_class.get(readiness_class, 0) + 1
        primary_blocker = str(row.get("blocker_stack_primary") or UNKNOWN)
        if primary_blocker != UNKNOWN:
            primary_blocker_counts[primary_blocker] = primary_blocker_counts.get(primary_blocker, 0) + 1
        next_step = str(row.get("primary_next_step") or NEXT_UNKNOWN)
        next_step_counts[next_step] = next_step_counts.get(next_step, 0) + 1
        if bool(row.get("blocker_stack_retryable", False)):
            retryable_blocker_count += 1
        if bool(row.get("blocker_stack_structural", False)):
            structural_blocker_count += 1

    def _rows_for(readiness_class: str) -> list[dict[str, Any]]:
        subset = [
            row
            for row in rows
            if str(row.get("investment_readiness_class") or READY_UNKNOWN) == readiness_class
        ]
        subset.sort(key=lambda row: (str(row.get("blocker_stack_primary") or UNKNOWN), str(row.get("ticker") or "")))
        return subset

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "counts_by_readiness_class": dict(
            sorted(counts_by_class.items(), key=lambda item: (READINESS_ORDER.get(item[0], 99), item[0]))
        ),
        "top_10_investable_now": [
            {
                "ticker": str(row.get("ticker") or ""),
                "investment_readiness_class": str(row.get("investment_readiness_class") or READY_UNKNOWN),
                "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
                "primary_next_step": str(row.get("primary_next_step") or NEXT_UNKNOWN),
            }
            for row in _rows_for(READY_INVESTABLE_NOW)[:10]
        ],
        "top_10_research_worthy_not_ready": [
            {
                "ticker": str(row.get("ticker") or ""),
                "investment_readiness_class": str(row.get("investment_readiness_class") or READY_UNKNOWN),
                "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
                "primary_next_step": str(row.get("primary_next_step") or NEXT_UNKNOWN),
            }
            for row in _rows_for(READY_RESEARCH_WORTHY)[:10]
        ],
        "primary_blocker_counts": dict(
            sorted(primary_blocker_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ),
        "retryable_blocker_count": int(retryable_blocker_count),
        "structural_blocker_count": int(structural_blocker_count),
        "primary_next_step_counts": dict(
            sorted(next_step_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["investment_readiness_path"] = str(output_path)
    return payload


def _investment_readiness_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "investment_readiness.json",
        cfg.sectors_dir / run_id / "investment_readiness.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_investment_readiness(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _investment_readiness_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "investment_readiness_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_readiness_class": payload.get("counts_by_readiness_class")
        if isinstance(payload.get("counts_by_readiness_class"), dict)
        else {},
        "top_10_investable_now": [
            row for row in (payload.get("top_10_investable_now") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_research_worthy_not_ready": [
            row for row in (payload.get("top_10_research_worthy_not_ready") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "primary_blocker_counts": payload.get("primary_blocker_counts")
        if isinstance(payload.get("primary_blocker_counts"), dict)
        else {},
        "retryable_blocker_count": int(payload.get("retryable_blocker_count") or 0),
        "structural_blocker_count": int(payload.get("structural_blocker_count") or 0),
        "primary_next_step_counts": payload.get("primary_next_step_counts")
        if isinstance(payload.get("primary_next_step_counts"), dict)
        else {},
        "investment_readiness_path": str(path),
    }
