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
    REVENUE_DEPENDENCE_UNKNOWN,
)
from app.valuation.intrinsic_discipline import (
    MOS_ADEQUATE,
    MOS_DEEP_VALUE_SUPPORT,
    MOS_MODEST,
    REASON_CYCLICAL_NORMALIZATION_LOW_CONFIDENCE,
    SUPPORT_ASSET,
    SUPPORT_BALANCE_SHEET,
    SUPPORT_EARNINGS,
    SUPPORT_LIMITED,
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
    SUPPORT_EPV,
    SUPPORT_NETNET,
    SUPPORT_NORMALIZED,
    SUPPORT_OWNER,
)


UNKNOWN = "UNKNOWN"
OK = "OK"

VALUE_TYPE_ASSET_BACKED = "ASSET_BACKED_VALUE"
VALUE_TYPE_EARNINGS_POWER = "EARNINGS_POWER_VALUE"
VALUE_TYPE_QUALITY = "QUALITY_VALUE"
VALUE_TYPE_CYCLICAL = "CYCLICAL_VALUE"
VALUE_TYPE_FRAGILE = "FRAGILE_VALUE"
VALUE_TYPE_UNKNOWN = "UNKNOWN_VALUE_TYPE"

VALUE_TYPE_ORDER = {
    VALUE_TYPE_ASSET_BACKED: 0,
    VALUE_TYPE_EARNINGS_POWER: 1,
    VALUE_TYPE_QUALITY: 2,
    VALUE_TYPE_CYCLICAL: 3,
    VALUE_TYPE_FRAGILE: 4,
    VALUE_TYPE_UNKNOWN: 5,
}

REASON_NETNET_DRIVEN = "NETNET_DRIVEN"
REASON_BALANCE_SHEET_SUPPORT_PRESENT = "BALANCE_SHEET_SUPPORT_PRESENT"
REASON_EPV_DRIVEN = "EPV_DRIVEN"
REASON_NORMALIZED_EARNINGS_DRIVEN = "NORMALIZED_EARNINGS_DRIVEN"
REASON_QUALITY_OVERLAY_SUPPORTED = "QUALITY_OVERLAY_SUPPORTED"
REASON_CYCLICAL_NORMALIZATION_CASE = "CYCLICAL_NORMALIZATION_CASE"
REASON_HIGH_FRAGILITY_CASE = "HIGH_FRAGILITY_CASE"
REASON_SINGLE_SUPPORT_CASE = "SINGLE_SUPPORT_CASE"
REASON_INSUFFICIENT_VALUE_TYPE_EVIDENCE = "INSUFFICIENT_VALUE_TYPE_EVIDENCE"
REASON_EARNINGS_POWER_SUPPORT_PRESENT = "EARNINGS_POWER_SUPPORT_PRESENT"
REASON_ASSET_SUPPORT_PRESENT = "ASSET_SUPPORT_PRESENT"
REASON_MULTI_SUPPORT_UNDERWRITTEN = "MULTI_SUPPORT_UNDERWRITTEN"
REASON_EVIDENCE_DEGRADED = "EVIDENCE_DEGRADED"
REASON_QUALITY_SUPPORT_WEAK = "QUALITY_SUPPORT_WEAK"

_NEGATIVE_REASONS = {
    REASON_HIGH_FRAGILITY_CASE,
    REASON_SINGLE_SUPPORT_CASE,
    REASON_INSUFFICIENT_VALUE_TYPE_EVIDENCE,
    REASON_EVIDENCE_DEGRADED,
    REASON_QUALITY_SUPPORT_WEAK,
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


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


def _dedupe_refs(values: list[Any]) -> list[str]:
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
    derived = _dedupe_refs(refs)
    token = str(value or "").strip()
    if token and token.upper() not in {"", UNKNOWN}:
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


def _status_snapshot(payload: dict[str, Any], key: str, fallback: str = UNKNOWN) -> str:
    value = payload.get(key, fallback)
    return str(value or fallback).upper()


def _support_summary(primary: str) -> str:
    if primary == VALUE_TYPE_ASSET_BACKED:
        return "Anchored to asset or balance-sheet support."
    if primary == VALUE_TYPE_EARNINGS_POWER:
        return "Anchored to normalized earnings power or EPV support."
    if primary == VALUE_TYPE_QUALITY:
        return "Anchored to valuation support with strong owner-economics overlays."
    if primary == VALUE_TYPE_CYCLICAL:
        return "Anchored to cyclical normalization with resilience evidence."
    if primary == VALUE_TYPE_FRAGILE:
        return "Appears cheap, but support is thin, conflicted, or fragile."
    return "Insufficient support to classify the value case responsibly."


def compute_value_type(
    ticker: str,
    as_of_date: str,
    *,
    intrinsic_payload: dict[str, Any] | None = None,
    valuation_confidence_payload: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    accounting_quality_payload: dict[str, Any] | None = None,
    balance_sheet_stress_payload: dict[str, Any] | None = None,
    returns_persistence_payload: dict[str, Any] | None = None,
    revenue_dependence_payload: dict[str, Any] | None = None,
    reinvestment_efficiency_payload: dict[str, Any] | None = None,
    cyclical_normalization_payload: dict[str, Any] | None = None,
    capital_allocation_discipline_payload: dict[str, Any] | None = None,
    fail_due_to_missing_evidence: bool = False,
    fail_due_to_economic_weakness: bool = False,
    primary_fail_domain: str | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg
    intrinsic_payload = intrinsic_payload if isinstance(intrinsic_payload, dict) else {}
    valuation_confidence_payload = (
        valuation_confidence_payload if isinstance(valuation_confidence_payload, dict) else {}
    )
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
        reinvestment_efficiency_payload if isinstance(reinvestment_efficiency_payload, dict) else {}
    )

    ticker_norm = str(ticker or "").strip().upper()
    support_types = {
        str(value)
        for value in (valuation_confidence_payload.get("valuation_support_types_present") or [])
        if str(value).strip()
    }
    support_count = int(valuation_confidence_payload.get("valuation_support_count") or 0)
    confidence_class = _status_snapshot(
        valuation_confidence_payload,
        "valuation_confidence_class",
        CONFIDENCE_UNKNOWN,
    )
    fragility_status = _status_snapshot(
        valuation_confidence_payload,
        "valuation_fragility_status",
        "FRAGILITY_UNKNOWN",
    )
    downside_support_type = _status_snapshot(intrinsic_payload, "downside_support_type", SUPPORT_UNKNOWN)
    mos_classification = _status_snapshot(intrinsic_payload, "mos_classification", UNKNOWN)
    normalized_method = _status_snapshot(
        intrinsic_payload,
        "normalized_earnings_power_method_used",
        UNKNOWN,
    )
    normalized_status = _status_snapshot(
        intrinsic_payload,
        "normalized_earnings_power_status",
        UNKNOWN,
    )
    normalized_reason_codes = {
        str(code)
        for code in (intrinsic_payload.get("normalized_earnings_power_reason_codes") or [])
        if str(code).strip()
    }
    oe_quality_total = owner_quality_payload.get("oe_quality_total", UNKNOWN)
    intangible_total = intangible_payload.get("intangible_economics_total", UNKNOWN)
    cycle_resilience_score = intangible_payload.get("cycle_resilience_score", UNKNOWN)
    owner_value_capture_score = intangible_payload.get("owner_value_capture_score", UNKNOWN)
    accounting_quality_class = str(
        accounting_quality_payload.get("accounting_quality_class") or ACCOUNTING_QUALITY_UNKNOWN
    ).upper()
    accounting_quality_reason_codes = {
        str(code)
        for code in (accounting_quality_payload.get("accounting_quality_reason_codes") or [])
        if str(code).strip()
    }
    balance_sheet_stress_class = str(
        balance_sheet_stress_payload.get("balance_sheet_stress_class") or BALANCE_SHEET_STRESS_UNKNOWN
    ).upper()
    refinancing_risk_class = str(
        balance_sheet_stress_payload.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
    ).upper()
    balance_sheet_headwind_signals = {
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_headwind_signals") or [])
        if str(code).strip()
    }
    returns_persistence_class = str(
        returns_persistence_payload.get("returns_persistence_class") or RETURNS_PERSISTENCE_UNKNOWN
    ).upper()
    returns_reason_codes = {
        str(code)
        for code in (returns_persistence_payload.get("returns_persistence_reason_codes") or [])
        if str(code).strip()
    }
    returns_headwind_signals = {
        str(code)
        for code in (returns_persistence_payload.get("returns_headwind_signals") or [])
        if str(code).strip()
    }
    revenue_dependence_risk_class = str(
        revenue_dependence_payload.get("revenue_dependence_risk_class") or REVENUE_DEPENDENCE_UNKNOWN
    ).upper()
    revenue_dependence_reason_codes = {
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_risk_reason_codes") or [])
        if str(code).strip()
    }
    revenue_dependence_headwind_signals = {
        str(code)
        for code in (revenue_dependence_payload.get("revenue_dependence_headwind_signals") or [])
        if str(code).strip()
    }
    reinvestment_efficiency_class = str(
        reinvestment_efficiency_payload.get("reinvestment_efficiency_class")
        or "REINVESTMENT_EFFICIENCY_UNKNOWN"
    ).upper()
    reinvestment_reason_codes = {
        str(code)
        for code in (reinvestment_efficiency_payload.get("reinvestment_efficiency_reason_codes") or [])
        if str(code).strip()
    }
    primary_fail_domain = str(primary_fail_domain or UNKNOWN).upper()

    asset_support_present = (
        SUPPORT_NETNET in support_types
        or downside_support_type in {SUPPORT_ASSET, SUPPORT_BALANCE_SHEET}
    )
    earnings_support_present = (
        downside_support_type == SUPPORT_EARNINGS
        or bool(support_types & {SUPPORT_EPV, SUPPORT_NORMALIZED, SUPPORT_OWNER})
    )
    real_discount_present = mos_classification in {
        MOS_DEEP_VALUE_SUPPORT,
        MOS_ADEQUATE,
        MOS_MODEST,
    }
    multi_support = support_count >= 2
    quality_overlay_supported = (
        multi_support
        and confidence_class in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}
        and fragility_status in {FRAGILITY_LOW, FRAGILITY_MODERATE}
        and real_discount_present
        and _is_num(oe_quality_total)
        and float(oe_quality_total) >= 8.0
        and _is_num(intangible_total)
        and float(intangible_total) >= 7.0
        and _is_num(owner_value_capture_score)
        and float(owner_value_capture_score) >= 3.0
        and returns_persistence_class == HIGH_RETURNS_PERSISTENCE
        and revenue_dependence_risk_class != HIGH_REVENUE_DEPENDENCE_RISK
        and accounting_quality_class != LOW_ACCOUNTING_QUALITY
        and balance_sheet_stress_class != HIGH_BALANCE_SHEET_STRESS
        and refinancing_risk_class != "HIGH_REFINANCING_RISK"
        and reinvestment_efficiency_class != "LOW_REINVESTMENT_EFFICIENCY"
    )
    # Improved cyclical detection: prefer explicit cyclical_normalization_payload signals
    _cyclical_payload = cyclical_normalization_payload if isinstance(cyclical_normalization_payload, dict) else {}
    _cyclical_profile = str(_cyclical_payload.get("cyclical_profile_class") or "CYCLICALITY_UNKNOWN")
    _module_cyclical_case = (
        _cyclical_profile in {"CLEARLY_CYCLICAL", "MODERATELY_CYCLICAL"}
        and earnings_support_present
        and support_count >= 2
    )
    # Fallback: legacy detection via cycle_resilience_score + reason code
    _legacy_cyclical_case = (
        _is_num(cycle_resilience_score)
        and float(cycle_resilience_score) >= 3.0
        and REASON_CYCLICAL_NORMALIZATION_LOW_CONFIDENCE in normalized_reason_codes
        and earnings_support_present
        and support_count >= 2
    )
    cyclical_normalization_case = (
        (_module_cyclical_case or _legacy_cyclical_case)
        and confidence_class in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW}
        and fragility_status in {FRAGILITY_LOW, FRAGILITY_MODERATE}
    )
    fragile_case = (
        support_count > 0
        and (
            fragility_status == FRAGILITY_HIGH
            or REASON_SINGLE_SUPPORT_ONLY
            in {
                str(code)
                for code in (valuation_confidence_payload.get("valuation_fragility_reason_codes") or [])
                if str(code).strip()
            }
            or confidence_class == CONFIDENCE_LOW
        )
    )
    evidence_degraded_case = bool(fail_due_to_missing_evidence) or primary_fail_domain == "EVIDENCE"
    insufficient_case = (
        support_count == 0
        or downside_support_type in {SUPPORT_UNKNOWN, SUPPORT_LIMITED}
        or confidence_class == CONFIDENCE_UNKNOWN
    )

    primary = VALUE_TYPE_UNKNOWN
    secondary: str | None = None
    reasons: list[str] = []

    if insufficient_case and (support_count == 0 or evidence_degraded_case):
        primary = VALUE_TYPE_UNKNOWN
        reasons.extend([REASON_INSUFFICIENT_VALUE_TYPE_EVIDENCE])
        if evidence_degraded_case:
            reasons.append(REASON_EVIDENCE_DEGRADED)
    elif fragile_case and not (multi_support and confidence_class in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}):
        primary = VALUE_TYPE_FRAGILE
        reasons.append(REASON_HIGH_FRAGILITY_CASE)
        if support_count <= 1:
            reasons.append(REASON_SINGLE_SUPPORT_CASE)
    elif asset_support_present:
        primary = VALUE_TYPE_ASSET_BACKED
        reasons.append(REASON_ASSET_SUPPORT_PRESENT)
        if SUPPORT_NETNET in support_types:
            reasons.append(REASON_NETNET_DRIVEN)
        if downside_support_type in {SUPPORT_ASSET, SUPPORT_BALANCE_SHEET}:
            reasons.append(REASON_BALANCE_SHEET_SUPPORT_PRESENT)
        if multi_support:
            reasons.append(REASON_MULTI_SUPPORT_UNDERWRITTEN)
    elif cyclical_normalization_case:
        primary = VALUE_TYPE_CYCLICAL
        reasons.extend([REASON_CYCLICAL_NORMALIZATION_CASE, REASON_EARNINGS_POWER_SUPPORT_PRESENT])
        if SUPPORT_EPV in support_types:
            reasons.append(REASON_EPV_DRIVEN)
        if SUPPORT_NORMALIZED in support_types or normalized_method != UNKNOWN:
            reasons.append(REASON_NORMALIZED_EARNINGS_DRIVEN)
        if multi_support:
            reasons.append(REASON_MULTI_SUPPORT_UNDERWRITTEN)
        secondary = VALUE_TYPE_EARNINGS_POWER
    elif quality_overlay_supported:
        primary = VALUE_TYPE_QUALITY
        reasons.append(REASON_QUALITY_OVERLAY_SUPPORTED)
        if earnings_support_present:
            reasons.append(REASON_EARNINGS_POWER_SUPPORT_PRESENT)
            secondary = VALUE_TYPE_EARNINGS_POWER
        elif asset_support_present:
            reasons.append(REASON_ASSET_SUPPORT_PRESENT)
            secondary = VALUE_TYPE_ASSET_BACKED
        if multi_support:
            reasons.append(REASON_MULTI_SUPPORT_UNDERWRITTEN)
    elif earnings_support_present and not insufficient_case:
        primary = VALUE_TYPE_EARNINGS_POWER
        reasons.append(REASON_EARNINGS_POWER_SUPPORT_PRESENT)
        if SUPPORT_EPV in support_types:
            reasons.append(REASON_EPV_DRIVEN)
        if SUPPORT_NORMALIZED in support_types or normalized_method != UNKNOWN:
            reasons.append(REASON_NORMALIZED_EARNINGS_DRIVEN)
        if multi_support:
            reasons.append(REASON_MULTI_SUPPORT_UNDERWRITTEN)
    elif fragile_case:
        primary = VALUE_TYPE_FRAGILE
        reasons.append(REASON_HIGH_FRAGILITY_CASE)
        if support_count <= 1:
            reasons.append(REASON_SINGLE_SUPPORT_CASE)
    else:
        primary = VALUE_TYPE_UNKNOWN
        reasons.append(REASON_INSUFFICIENT_VALUE_TYPE_EVIDENCE)
        if evidence_degraded_case:
            reasons.append(REASON_EVIDENCE_DEGRADED)
        if not quality_overlay_supported and (
            _is_num(oe_quality_total) or _is_num(intangible_total) or _is_num(owner_value_capture_score)
        ):
            reasons.append(REASON_QUALITY_SUPPORT_WEAK)

    if primary == VALUE_TYPE_UNKNOWN and fail_due_to_economic_weakness:
        reasons.append(REASON_QUALITY_SUPPORT_WEAK)

    # Capital allocation discipline light integration — auxiliary reason codes only
    # Does NOT change the primary value_type classification
    _cap_alloc = capital_allocation_discipline_payload if isinstance(capital_allocation_discipline_payload, dict) else {}
    _cap_alloc_class = str(_cap_alloc.get("capital_allocation_discipline_class") or "").upper()
    if _cap_alloc_class == "OWNER_FRIENDLY_DISCIPLINED" and primary in {VALUE_TYPE_QUALITY, VALUE_TYPE_EARNINGS_POWER}:
        reasons.append("OWNER_FRIENDLY_CAPITAL_ALLOCATION_SUPPORT")
    elif _cap_alloc_class == "OWNER_DILUTIVE_OR_DESTRUCTIVE":
        reasons.append("DILUTIVE_CAPITAL_ALLOCATION_HEADWIND")
    if balance_sheet_stress_class == LOW_BALANCE_SHEET_STRESS and primary in {
        VALUE_TYPE_QUALITY,
        VALUE_TYPE_EARNINGS_POWER,
    }:
        reasons.append("LOW_BALANCE_SHEET_STRESS_SUPPORT")
    elif balance_sheet_stress_class == HIGH_BALANCE_SHEET_STRESS:
        reasons.append("HIGH_BALANCE_SHEET_STRESS_HEADWIND")
    elif balance_sheet_stress_class == BALANCE_SHEET_STRESS_UNKNOWN:
        reasons.append("BALANCE_SHEET_EVIDENCE_THIN")
    if refinancing_risk_class == "HIGH_REFINANCING_RISK":
        reasons.append("HIGH_REFINANCING_RISK_HEADWIND")
    elif refinancing_risk_class == "REFINANCING_RISK_UNKNOWN":
        reasons.append("BALANCE_SHEET_EVIDENCE_THIN")
    if "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in balance_sheet_headwind_signals:
        reasons.append("CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE")
        if primary in {VALUE_TYPE_QUALITY, VALUE_TYPE_EARNINGS_POWER}:
            primary = VALUE_TYPE_FRAGILE
            secondary = None
    if returns_persistence_class == HIGH_RETURNS_PERSISTENCE and primary in {
        VALUE_TYPE_QUALITY,
        VALUE_TYPE_EARNINGS_POWER,
    }:
        reasons.append("HIGH_RETURNS_PERSISTENCE_SUPPORT")
    elif returns_persistence_class == LOW_RETURNS_PERSISTENCE:
        reasons.append("LOW_RETURNS_PERSISTENCE_HEADWIND")
        if "INCREMENTAL_RETURNS_DETERIORATING" in returns_headwind_signals:
            reasons.append("INCREMENTAL_RETURNS_DETERIORATION")
        if primary in {VALUE_TYPE_QUALITY, VALUE_TYPE_CYCLICAL}:
            primary = VALUE_TYPE_FRAGILE
            secondary = None
    elif returns_persistence_class == RETURNS_PERSISTENCE_UNKNOWN:
        reasons.append("RETURNS_DURABILITY_UNKNOWN")
    if revenue_dependence_risk_class == LOW_REVENUE_DEPENDENCE_RISK and primary in {
        VALUE_TYPE_QUALITY,
        VALUE_TYPE_EARNINGS_POWER,
    }:
        reasons.append("LOW_REVENUE_DEPENDENCE_SUPPORT")
    elif revenue_dependence_risk_class == HIGH_REVENUE_DEPENDENCE_RISK:
        reasons.append("HIGH_REVENUE_DEPENDENCE_HEADWIND")
        if {
            "SINGLE_CUSTOMER_CONCENTRATION",
            "TOP_CUSTOMER_DOMINANCE",
        } & revenue_dependence_headwind_signals:
            reasons.append("CUSTOMER_CONCENTRATION_HEADWIND")
        if "NARROW_CHANNEL_DEPENDENCE" in revenue_dependence_headwind_signals:
            reasons.append("CHANNEL_DEPENDENCE_HEADWIND")
        if primary in {VALUE_TYPE_QUALITY, VALUE_TYPE_CYCLICAL}:
            primary = VALUE_TYPE_FRAGILE
            secondary = None
    elif revenue_dependence_risk_class == REVENUE_DEPENDENCE_UNKNOWN:
        reasons.append("REVENUE_DEPENDENCE_UNKNOWN")
    if "RETURNS_DEPEND_ON_FAVORABLE_CYCLE" in returns_headwind_signals:
        reasons.append("RETURNS_DEPEND_ON_FAVORABLE_CYCLE")
        if primary == VALUE_TYPE_CYCLICAL:
            primary = VALUE_TYPE_FRAGILE
            secondary = None
    if accounting_quality_class == "HIGH_ACCOUNTING_QUALITY" and primary in {
        VALUE_TYPE_QUALITY,
        VALUE_TYPE_EARNINGS_POWER,
    }:
        reasons.append("HIGH_ACCOUNTING_QUALITY_SUPPORT")
    elif accounting_quality_class == LOW_ACCOUNTING_QUALITY:
        reasons.append("LOW_ACCOUNTING_QUALITY_HEADWIND")
        if "ACCRUAL_HEAVY_EARNINGS_HEADWIND" in accounting_quality_reason_codes:
            reasons.append("ACCRUAL_HEAVY_EARNINGS_HEADWIND")
    elif accounting_quality_class == ACCOUNTING_QUALITY_UNKNOWN:
        reasons.append("ACCOUNTING_EVIDENCE_THIN")
    if (
        reinvestment_efficiency_class == "HIGH_REINVESTMENT_EFFICIENCY"
        and primary in {VALUE_TYPE_QUALITY, VALUE_TYPE_EARNINGS_POWER}
    ):
        reasons.append("PRODUCTIVE_REINVESTMENT_SUPPORT")
    elif reinvestment_efficiency_class == "LOW_REINVESTMENT_EFFICIENCY":
        reasons.append("CAPITAL_HUNGRY_GROWTH_HEADWIND")
    elif reinvestment_efficiency_class == "REINVESTMENT_EFFICIENCY_UNKNOWN":
        reasons.append("REINVESTMENT_EVIDENCE_THIN")
    if "GROWTH_WITHOUT_OWNER_OUTCOME" in reinvestment_reason_codes:
        reasons.append("GROWTH_WITHOUT_OWNER_OUTCOME")

    reasons = _dedupe_refs(reasons)
    derived_from = _dedupe_refs(
        _claim_refs(intrinsic_payload, "normalized_earnings_power_value")
        + _claim_refs(intrinsic_payload, "intrinsic_floor")
        + _claim_refs(intrinsic_payload, "intrinsic_base")
        + _claim_refs(valuation_confidence_payload, "valuation_support_count")
        + _claim_refs(valuation_confidence_payload, "valuation_confidence_class")
        + [str(ref) for ref in (owner_quality_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (intangible_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (accounting_quality_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (balance_sheet_stress_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (returns_persistence_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (revenue_dependence_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (reinvestment_efficiency_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (intrinsic_payload.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (valuation_confidence_payload.get("derived_from") or []) if str(ref).strip()]
    )
    claims = {
        "value_type_primary": _claim(
            value=primary,
            refs=derived_from,
            reason_code=reasons[0] if reasons else UNKNOWN,
            status=OK if primary != VALUE_TYPE_UNKNOWN else UNKNOWN,
        ),
        "value_type_secondary": _claim(
            value=secondary or UNKNOWN,
            refs=derived_from,
            reason_code=reasons[1] if len(reasons) > 1 else (reasons[0] if reasons else UNKNOWN),
            status=OK if secondary else UNKNOWN,
        ),
    }
    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "value_type_primary": primary,
        "value_type_secondary": secondary,
        "value_type_reason_codes": reasons,
        "value_type_support_summary": _support_summary(primary),
        "value_type_derived_from": derived_from,
        "claims": claims,
        "generated_at": utc_now_iso(),
    }


def write_value_type_for_run(
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
        str(row.get("ticker") or "").upper(): row.get("value_type_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("value_type_detail"), dict)
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
            detail = compute_value_type(
                ticker=ticker,
                as_of_date=as_of_date,
                intrinsic_payload=score_row.get("intrinsic_discipline_detail")
                if isinstance(score_row.get("intrinsic_discipline_detail"), dict)
                else {},
                valuation_confidence_payload=score_row.get("valuation_confidence_detail")
                if isinstance(score_row.get("valuation_confidence_detail"), dict)
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
                reinvestment_efficiency_payload=score_row.get("reinvestment_efficiency_detail")
                if isinstance(score_row.get("reinvestment_efficiency_detail"), dict)
                else {},
                fail_due_to_missing_evidence=bool(score_row.get("fail_due_to_missing_evidence", False)),
                fail_due_to_economic_weakness=bool(score_row.get("fail_due_to_economic_weakness", False)),
                primary_fail_domain=str(score_row.get("primary_fail_domain") or UNKNOWN),
            )
        rows.append(detail)

    counts_by_type: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        primary = str(row.get("value_type_primary") or VALUE_TYPE_UNKNOWN)
        counts_by_type[primary] = counts_by_type.get(primary, 0) + 1
        for code in (row.get("value_type_reason_codes") or []):
            token = str(code or "").strip()
            if not token:
                continue
            reason_counts[token] = reason_counts.get(token, 0) + 1

    def _typed_rows(value_type: str) -> list[dict[str, Any]]:
        subset = [
            row
            for row in rows
            if str(row.get("value_type_primary") or VALUE_TYPE_UNKNOWN) == value_type
        ]
        subset.sort(key=lambda row: (VALUE_TYPE_ORDER.get(value_type, 99), str(row.get("ticker") or "")))
        return subset

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "counts_by_primary_value_type": dict(
            sorted(counts_by_type.items(), key=lambda item: (VALUE_TYPE_ORDER.get(item[0], 99), item[0]))
        ),
        "top_10_asset_backed_value": [
            {
                "ticker": str(row.get("ticker") or ""),
                "value_type_primary": str(row.get("value_type_primary") or VALUE_TYPE_UNKNOWN),
                "value_type_reason_codes": [
                    str(code) for code in (row.get("value_type_reason_codes") or []) if str(code).strip()
                ],
            }
            for row in _typed_rows(VALUE_TYPE_ASSET_BACKED)[:10]
        ],
        "top_10_earnings_power_value": [
            {
                "ticker": str(row.get("ticker") or ""),
                "value_type_primary": str(row.get("value_type_primary") or VALUE_TYPE_UNKNOWN),
                "value_type_reason_codes": [
                    str(code) for code in (row.get("value_type_reason_codes") or []) if str(code).strip()
                ],
            }
            for row in _typed_rows(VALUE_TYPE_EARNINGS_POWER)[:10]
        ],
        "fragile_value_count": int(counts_by_type.get(VALUE_TYPE_FRAGILE, 0)),
        "unknown_value_type_count": int(counts_by_type.get(VALUE_TYPE_UNKNOWN, 0)),
        "value_type_reason_counts": dict(
            sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["value_type_path"] = str(output_path)
    return payload


def _value_type_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "value_type.json",
        cfg.sectors_dir / run_id / "value_type.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_value_type(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _value_type_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "value_type_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_primary_value_type": payload.get("counts_by_primary_value_type")
        if isinstance(payload.get("counts_by_primary_value_type"), dict)
        else {},
        "top_10_asset_backed_value": [
            row for row in (payload.get("top_10_asset_backed_value") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_earnings_power_value": [
            row for row in (payload.get("top_10_earnings_power_value") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "fragile_value_count": int(payload.get("fragile_value_count") or 0),
        "unknown_value_type_count": int(payload.get("unknown_value_type_count") or 0),
        "value_type_reason_counts": payload.get("value_type_reason_counts")
        if isinstance(payload.get("value_type_reason_counts"), dict)
        else {},
        "value_type_path": str(path),
    }
