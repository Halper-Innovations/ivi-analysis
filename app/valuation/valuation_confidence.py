from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.valuation.accounting_quality import (
    ACCOUNTING_QUALITY_UNKNOWN,
    HIGH_ACCOUNTING_QUALITY,
    LOW_ACCOUNTING_QUALITY,
)
from app.valuation.balance_sheet_stress import (
    BALANCE_SHEET_STRESS_UNKNOWN,
    HIGH_BALANCE_SHEET_STRESS,
    LOW_BALANCE_SHEET_STRESS,
)
from app.valuation.intrinsic_discipline import REASON_OWNER_EARNINGS_SELECTED
from app.valuation.maintenance_capex_discipline import (
    ASSET_INTENSITY_UNKNOWN,
    HIGH_ASSET_INTENSITY,
    LOW_ASSET_INTENSITY,
    LOW_MAINTENANCE_CAPEX_CREDIBILITY,
    MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN,
)


UNKNOWN = "UNKNOWN"
OK = "OK"

SUPPORT_NETNET = "NETNET_ASSET_SUPPORT"
SUPPORT_EPV = "EPV_SUPPORT"
SUPPORT_NORMALIZED = "NORMALIZED_EARNINGS_POWER_SUPPORT"
SUPPORT_OWNER = "OWNER_EARNINGS_VALUE_SUPPORT"

CONVERGENCE_STRONG = "STRONG_CONVERGENCE"
CONVERGENCE_MODERATE = "MODERATE_CONVERGENCE"
CONVERGENCE_WEAK = "WEAK_CONVERGENCE"
CONVERGENCE_UNKNOWN = "CONVERGENCE_UNKNOWN"

FRAGILITY_LOW = "LOW_FRAGILITY"
FRAGILITY_MODERATE = "MODERATE_FRAGILITY"
FRAGILITY_HIGH = "HIGH_FRAGILITY"
FRAGILITY_UNKNOWN = "FRAGILITY_UNKNOWN"

CONFIDENCE_HIGH = "HIGH_CONFIDENCE"
CONFIDENCE_MEDIUM = "MEDIUM_CONFIDENCE"
CONFIDENCE_LOW = "LOW_CONFIDENCE"
CONFIDENCE_UNKNOWN = "CONFIDENCE_UNKNOWN"

REASON_SINGLE_SUPPORT_ONLY = "SINGLE_SUPPORT_ONLY"
REASON_MISSING_PRICE = "MISSING_PRICE"
REASON_MISSING_SHARES = "MISSING_SHARES"
REASON_MISSING_FCF = "MISSING_FCF"
REASON_MISSING_FACTS = "MISSING_FACTS"
REASON_NORMALIZATION_LOW_CONFIDENCE = "NORMALIZATION_LOW_CONFIDENCE"
REASON_SUPPORTS_CONFLICT = "SUPPORTS_CONFLICT"
REASON_FRAGILITY_REDUCED_BY_MULTI_SUPPORT = "FRAGILITY_REDUCED_BY_MULTI_SUPPORT"
REASON_NO_VALUATION_SUPPORTS_PRESENT = "NO_VALUATION_SUPPORTS_PRESENT"
REASON_INSUFFICIENT_SUPPORTS_FOR_CONVERGENCE = "INSUFFICIENT_SUPPORTS_FOR_CONVERGENCE"
REASON_MULTI_SUPPORT_VALUE_CASE = "MULTI_SUPPORT_VALUE_CASE"
REASON_LOW_FRAGILITY_UNDERWRITING = "LOW_FRAGILITY_UNDERWRITING"
REASON_SINGLE_SUPPORT_FRAGILE = "SINGLE_SUPPORT_FRAGILE"
REASON_SUPPORT_CONFLICT_HEADWIND = "SUPPORT_CONFLICT_HEADWIND"
REASON_HIGH_CONFIDENCE_UNDERWRITING = "HIGH_CONFIDENCE_UNDERWRITING"
REASON_MEDIUM_CONFIDENCE_UNDERWRITING = "MEDIUM_CONFIDENCE_UNDERWRITING"
REASON_LOW_CONFIDENCE_UNDERWRITING = "LOW_CONFIDENCE_UNDERWRITING"
REASON_INTEGRITY_WARNING_HEADWIND = "INTEGRITY_WARNING_HEADWIND"
REASON_INTEGRITY_SUSPECT_HEADWIND = "INTEGRITY_SUSPECT_HEADWIND"
REASON_HIGH_ACCOUNTING_QUALITY_SUPPORT = "HIGH_ACCOUNTING_QUALITY_SUPPORT"
REASON_LOW_ACCOUNTING_QUALITY_HEADWIND = "LOW_ACCOUNTING_QUALITY_HEADWIND"
REASON_ACCRUAL_HEAVY_EARNINGS_HEADWIND = "ACCRUAL_HEAVY_EARNINGS_HEADWIND"
REASON_WEAK_CASH_CONVERSION_HEADWIND = "WEAK_CASH_CONVERSION_HEADWIND"
REASON_ACCOUNTING_EVIDENCE_THIN = "ACCOUNTING_EVIDENCE_THIN"
REASON_LOW_BALANCE_SHEET_STRESS_SUPPORT = "LOW_BALANCE_SHEET_STRESS_SUPPORT"
REASON_HIGH_BALANCE_SHEET_STRESS_HEADWIND = "HIGH_BALANCE_SHEET_STRESS_HEADWIND"
REASON_HIGH_REFINANCING_RISK_HEADWIND = "HIGH_REFINANCING_RISK_HEADWIND"
REASON_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE = "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"
REASON_BALANCE_SHEET_EVIDENCE_THIN = "BALANCE_SHEET_EVIDENCE_THIN"
REASON_LOW_ASSET_INTENSITY_SUPPORT = "LOW_ASSET_INTENSITY_SUPPORT"
REASON_HIGH_ASSET_INTENSITY_HEADWIND = "HIGH_ASSET_INTENSITY_HEADWIND"
REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND = "LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND"
REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE = "OWNER_EARNINGS_DENOMINATOR_SENSITIVE"
REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN = "MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN"

_POSITIVE_FRAGILITY_REASONS = {
    REASON_FRAGILITY_REDUCED_BY_MULTI_SUPPORT,
}
_POSITIVE_CONFIDENCE_REASONS = {
    REASON_MULTI_SUPPORT_VALUE_CASE,
    REASON_LOW_FRAGILITY_UNDERWRITING,
    REASON_HIGH_CONFIDENCE_UNDERWRITING,
    REASON_MEDIUM_CONFIDENCE_UNDERWRITING,
}


def apply_valuation_integrity_headwind(
    payload: dict[str, Any] | None,
    *,
    integrity_class: str,
    integrity_reason_codes: list[str] | None = None,
    derived_from: list[Any] | None = None,
) -> dict[str, Any]:
    detail = dict(payload or {})
    integrity_token = str(integrity_class or "").upper()
    if integrity_token not in {"INTEGRITY_WARNING", "INTEGRITY_SUSPECT"}:
        return detail

    current = str(detail.get("valuation_confidence_class") or CONFIDENCE_UNKNOWN).upper()
    downgraded = current
    headwind_reason = REASON_INTEGRITY_WARNING_HEADWIND
    if integrity_token == "INTEGRITY_WARNING":
        if current == CONFIDENCE_HIGH:
            downgraded = CONFIDENCE_MEDIUM
    else:
        headwind_reason = REASON_INTEGRITY_SUSPECT_HEADWIND
        if current in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}:
            downgraded = CONFIDENCE_LOW
        elif current == CONFIDENCE_UNKNOWN:
            downgraded = CONFIDENCE_UNKNOWN
        else:
            downgraded = CONFIDENCE_LOW

    detail["valuation_confidence_class"] = downgraded
    detail["valuation_confidence_reason_codes"] = _dedupe_refs(
        list(detail.get("valuation_confidence_reason_codes") or [])
        + [headwind_reason]
        + [str(code) for code in (integrity_reason_codes or []) if str(code).strip()]
    )
    detail["derived_from"] = _dedupe_refs(
        list(detail.get("derived_from") or []) + [str(ref) for ref in (derived_from or []) if str(ref).strip()]
    )
    claims = detail.get("claims") if isinstance(detail.get("claims"), dict) else {}
    claims["valuation_confidence_class"] = _claim(
        value=downgraded,
        refs=list((claims.get("valuation_confidence_class") or {}).get("derived_from") or [])
        + list(detail.get("derived_from") or []),
        reason_code=headwind_reason,
        status=OK if downgraded != CONFIDENCE_UNKNOWN else UNKNOWN,
    )
    detail["claims"] = claims
    return detail


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


def _normalized_reason_codes(intrinsic_payload: dict[str, Any]) -> list[str]:
    return [
        str(code)
        for code in (intrinsic_payload.get("normalized_earnings_power_reason_codes") or [])
        if str(code).strip()
    ]


def _support_entries(
    *,
    intrinsic_payload: dict[str, Any],
    epv_per_share: Any,
    epv_refs: list[Any],
    netnet_per_share: Any,
    netnet_refs: list[Any],
    existing_intrinsic_base: Any,
    existing_intrinsic_base_refs: list[Any],
    existing_intrinsic_conservative: Any,
    existing_intrinsic_conservative_refs: list[Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if _is_num(netnet_per_share) and float(netnet_per_share) > 0.0:
        entries.append(
            {
                "support_type": SUPPORT_NETNET,
                "representative_value": float(netnet_per_share),
                "derived_from": _dedupe_refs(netnet_refs or _claim_refs(intrinsic_payload, "intrinsic_floor")),
            }
        )
    if _is_num(epv_per_share) and float(epv_per_share) > 0.0:
        entries.append(
            {
                "support_type": SUPPORT_EPV,
                "representative_value": float(epv_per_share),
                "derived_from": _dedupe_refs(epv_refs or _claim_refs(intrinsic_payload, "intrinsic_base")),
            }
        )

    normalized_value = intrinsic_payload.get("normalized_earnings_power_value", UNKNOWN)
    intrinsic_base = intrinsic_payload.get("intrinsic_base", UNKNOWN)
    if _is_num(normalized_value) and float(normalized_value) > 0.0 and _is_num(intrinsic_base) and float(intrinsic_base) > 0.0:
        entries.append(
            {
                "support_type": SUPPORT_NORMALIZED,
                "representative_value": float(intrinsic_base),
                "derived_from": _dedupe_refs(
                    _claim_refs(intrinsic_payload, "normalized_earnings_power_value")
                    + _claim_refs(intrinsic_payload, "intrinsic_base")
                ),
            }
        )

    normalized_method = str(intrinsic_payload.get("normalized_earnings_power_method_used") or UNKNOWN)
    if (
        _is_num(existing_intrinsic_conservative)
        and float(existing_intrinsic_conservative) > 0.0
        and normalized_method.upper() not in {REASON_OWNER_EARNINGS_SELECTED}
    ):
        entries.append(
            {
                "support_type": SUPPORT_OWNER,
                "representative_value": float(existing_intrinsic_conservative),
                "derived_from": _dedupe_refs(existing_intrinsic_conservative_refs),
            }
        )
    elif (
        _is_num(existing_intrinsic_base)
        and float(existing_intrinsic_base) > 0.0
        and normalized_method.upper() not in {REASON_OWNER_EARNINGS_SELECTED}
        and not any(item["support_type"] == SUPPORT_NORMALIZED for item in entries)
    ):
        entries.append(
            {
                "support_type": SUPPORT_OWNER,
                "representative_value": float(existing_intrinsic_base),
                "derived_from": _dedupe_refs(existing_intrinsic_base_refs),
            }
        )
    return entries


def compute_valuation_confidence(
    ticker: str,
    as_of_date: str,
    *,
    intrinsic_payload: dict[str, Any] | None = None,
    accounting_quality_payload: dict[str, Any] | None = None,
    balance_sheet_stress_payload: dict[str, Any] | None = None,
    maintenance_capex_payload: dict[str, Any] | None = None,
    price_status: str | None = None,
    shares_status: str | None = None,
    fcf_status: str | None = None,
    facts_status: str | None = None,
    valuation_status: str | None = None,
    epv_per_share: Any = UNKNOWN,
    epv_refs: list[Any] | None = None,
    netnet_per_share: Any = UNKNOWN,
    netnet_refs: list[Any] | None = None,
    existing_intrinsic_base: Any = UNKNOWN,
    existing_intrinsic_base_refs: list[Any] | None = None,
    existing_intrinsic_conservative: Any = UNKNOWN,
    existing_intrinsic_conservative_refs: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg
    ticker_norm = str(ticker or "").strip().upper()
    intrinsic_payload = intrinsic_payload if isinstance(intrinsic_payload, dict) else {}
    accounting_quality_payload = (
        accounting_quality_payload if isinstance(accounting_quality_payload, dict) else {}
    )
    balance_sheet_stress_payload = (
        balance_sheet_stress_payload if isinstance(balance_sheet_stress_payload, dict) else {}
    )
    maintenance_capex_payload = (
        maintenance_capex_payload if isinstance(maintenance_capex_payload, dict) else {}
    )
    epv_refs = list(epv_refs or [])
    netnet_refs = list(netnet_refs or [])
    existing_intrinsic_base_refs = list(existing_intrinsic_base_refs or [])
    existing_intrinsic_conservative_refs = list(existing_intrinsic_conservative_refs or [])

    support_entries = _support_entries(
        intrinsic_payload=intrinsic_payload,
        epv_per_share=epv_per_share,
        epv_refs=epv_refs,
        netnet_per_share=netnet_per_share,
        netnet_refs=netnet_refs,
        existing_intrinsic_base=existing_intrinsic_base,
        existing_intrinsic_base_refs=existing_intrinsic_base_refs,
        existing_intrinsic_conservative=existing_intrinsic_conservative,
        existing_intrinsic_conservative_refs=existing_intrinsic_conservative_refs,
    )
    support_types_present = [str(item["support_type"]) for item in support_entries]
    support_values = [float(item["representative_value"]) for item in support_entries if _is_num(item.get("representative_value"))]
    support_refs = _dedupe_refs([ref for item in support_entries for ref in (item.get("derived_from") or [])])
    support_count = len(support_types_present)
    support_count_reason_codes = support_types_present or [REASON_NO_VALUATION_SUPPORTS_PRESENT]

    convergence_reason_codes: list[str] = []
    convergence_status = CONVERGENCE_UNKNOWN
    convergence_band_pct: float | str = UNKNOWN
    if support_count < 2:
        convergence_reason_codes.append(REASON_SINGLE_SUPPORT_ONLY if support_count == 1 else REASON_INSUFFICIENT_SUPPORTS_FOR_CONVERGENCE)
    else:
        # Scale the spread by the MIDPOINT of the range, not the median: an
        # estimate that lands inside the range corroborates it, but it moves
        # the median and could widen the reported band. With two
        # supports the midpoint is the median, so those bands are unchanged.
        midpoint_value = (max(support_values) + min(support_values)) / 2.0 if support_values else 0.0
        if midpoint_value > 0.0:
            convergence_band_pct = round((max(support_values) - min(support_values)) / float(midpoint_value), 6)
            if float(convergence_band_pct) <= 0.25:
                convergence_status = CONVERGENCE_STRONG
            elif float(convergence_band_pct) <= 0.50:
                convergence_status = CONVERGENCE_MODERATE
            else:
                convergence_status = CONVERGENCE_WEAK
                convergence_reason_codes.append(REASON_SUPPORTS_CONFLICT)
        else:
            convergence_reason_codes.append(REASON_INSUFFICIENT_SUPPORTS_FOR_CONVERGENCE)
    if not convergence_reason_codes and convergence_status == CONVERGENCE_UNKNOWN:
        convergence_reason_codes.append(REASON_INSUFFICIENT_SUPPORTS_FOR_CONVERGENCE)

    fragility_reason_codes: list[str] = []
    if support_count == 1:
        fragility_reason_codes.append(REASON_SINGLE_SUPPORT_ONLY)
    if str(price_status or UNKNOWN).upper() != OK:
        fragility_reason_codes.append(REASON_MISSING_PRICE)
    if str(shares_status or UNKNOWN).upper() != OK:
        fragility_reason_codes.append(REASON_MISSING_SHARES)
    if str(fcf_status or UNKNOWN).upper() != OK:
        fragility_reason_codes.append(REASON_MISSING_FCF)
    if str(facts_status or UNKNOWN).upper() != OK:
        fragility_reason_codes.append(REASON_MISSING_FACTS)
    if str(intrinsic_payload.get("normalized_earnings_power_status") or UNKNOWN).upper() == "LOW_CONFIDENCE":
        fragility_reason_codes.append(REASON_NORMALIZATION_LOW_CONFIDENCE)
    if convergence_status == CONVERGENCE_WEAK:
        fragility_reason_codes.append(REASON_SUPPORTS_CONFLICT)

    fragility_status = FRAGILITY_UNKNOWN
    if support_count == 0:
        fragility_status = FRAGILITY_UNKNOWN
    elif any(
        code in {
            REASON_SINGLE_SUPPORT_ONLY,
            REASON_MISSING_PRICE,
            REASON_MISSING_SHARES,
            REASON_MISSING_FACTS,
            REASON_NORMALIZATION_LOW_CONFIDENCE,
            REASON_SUPPORTS_CONFLICT,
        }
        for code in fragility_reason_codes
    ):
        fragility_status = FRAGILITY_HIGH
    elif REASON_MISSING_FCF in fragility_reason_codes or convergence_status == CONVERGENCE_MODERATE:
        fragility_status = FRAGILITY_MODERATE
    elif support_count >= 2 and convergence_status == CONVERGENCE_STRONG and str(valuation_status or UNKNOWN).upper() == OK:
        fragility_status = FRAGILITY_LOW
    elif support_count >= 2 and convergence_status in {CONVERGENCE_STRONG, CONVERGENCE_MODERATE}:
        fragility_status = FRAGILITY_MODERATE

    if fragility_status == FRAGILITY_LOW:
        fragility_reason_codes.append(REASON_FRAGILITY_REDUCED_BY_MULTI_SUPPORT)
    fragility_reason_codes = _dedupe_refs(fragility_reason_codes)

    confidence_reason_codes: list[str] = []
    confidence_class = CONFIDENCE_UNKNOWN
    if support_count == 0:
        confidence_reason_codes.append(REASON_NO_VALUATION_SUPPORTS_PRESENT)
    else:
        if support_count >= 2:
            confidence_reason_codes.append(REASON_MULTI_SUPPORT_VALUE_CASE)
        if fragility_status == FRAGILITY_LOW:
            confidence_reason_codes.append(REASON_LOW_FRAGILITY_UNDERWRITING)
        if support_count >= 3 and convergence_status == CONVERGENCE_STRONG and fragility_status == FRAGILITY_LOW:
            confidence_class = CONFIDENCE_HIGH
            confidence_reason_codes.append(REASON_HIGH_CONFIDENCE_UNDERWRITING)
        elif (
            support_count >= 2
            and convergence_status in {CONVERGENCE_STRONG, CONVERGENCE_MODERATE}
            and fragility_status in {FRAGILITY_LOW, FRAGILITY_MODERATE}
        ):
            confidence_class = CONFIDENCE_MEDIUM
            confidence_reason_codes.append(REASON_MEDIUM_CONFIDENCE_UNDERWRITING)
        elif fragility_status in {FRAGILITY_HIGH, FRAGILITY_MODERATE, FRAGILITY_LOW}:
            confidence_class = CONFIDENCE_LOW
            confidence_reason_codes.append(REASON_LOW_CONFIDENCE_UNDERWRITING)

    if REASON_SINGLE_SUPPORT_ONLY in fragility_reason_codes:
        confidence_reason_codes.append(REASON_SINGLE_SUPPORT_FRAGILE)
    if REASON_SUPPORTS_CONFLICT in fragility_reason_codes:
        confidence_reason_codes.append(REASON_SUPPORT_CONFLICT_HEADWIND)

    accounting_payload_present = bool(accounting_quality_payload)
    accounting_quality_class = str(
        accounting_quality_payload.get("accounting_quality_class") or ACCOUNTING_QUALITY_UNKNOWN
    ).upper()
    accounting_reasons = {
        str(code)
        for code in (accounting_quality_payload.get("accounting_quality_reason_codes") or [])
        if str(code).strip()
    }
    accounting_headwinds = {
        str(code)
        for code in (accounting_quality_payload.get("cash_earnings_headwind_signals") or [])
        if str(code).strip()
    }
    accounting_refs = [
        str(ref) for ref in (accounting_quality_payload.get("derived_from") or []) if str(ref).strip()
    ]
    if accounting_quality_class == LOW_ACCOUNTING_QUALITY:
        fragility_reason_codes.append(REASON_LOW_ACCOUNTING_QUALITY_HEADWIND)
        confidence_reason_codes.append(REASON_LOW_ACCOUNTING_QUALITY_HEADWIND)
        if {
            "ACCRUAL_HEAVY_EARNINGS",
            "REPORTED_EARNINGS_NOT_OWNER_RELEVANT",
        } & accounting_headwinds:
            fragility_reason_codes.append(REASON_ACCRUAL_HEAVY_EARNINGS_HEADWIND)
            confidence_reason_codes.append(REASON_ACCRUAL_HEAVY_EARNINGS_HEADWIND)
        if {
            "WEAK_CFO_TO_EARNINGS_CONVERSION",
            "WEAK_FCF_TO_EARNINGS_CONVERSION",
            "CASH_EARNINGS_DIVERGENCE",
        } & accounting_headwinds:
            fragility_reason_codes.append(REASON_WEAK_CASH_CONVERSION_HEADWIND)
            confidence_reason_codes.append(REASON_WEAK_CASH_CONVERSION_HEADWIND)

        if fragility_status == FRAGILITY_LOW:
            fragility_status = FRAGILITY_MODERATE
        else:
            fragility_status = FRAGILITY_HIGH

        if confidence_class == CONFIDENCE_HIGH:
            confidence_class = CONFIDENCE_MEDIUM
        elif confidence_class in {CONFIDENCE_MEDIUM, CONFIDENCE_LOW}:
            confidence_class = CONFIDENCE_LOW
    elif accounting_payload_present and accounting_quality_class == HIGH_ACCOUNTING_QUALITY:
        confidence_reason_codes.append(REASON_HIGH_ACCOUNTING_QUALITY_SUPPORT)
    elif accounting_payload_present and accounting_quality_class == ACCOUNTING_QUALITY_UNKNOWN:
        thin = REASON_ACCOUNTING_EVIDENCE_THIN in accounting_reasons or "ACCOUNTING_EVIDENCE_THIN" in accounting_reasons
        if thin:
            fragility_reason_codes.append(REASON_ACCOUNTING_EVIDENCE_THIN)
            confidence_reason_codes.append(REASON_ACCOUNTING_EVIDENCE_THIN)

    balance_sheet_payload_present = bool(balance_sheet_stress_payload)
    balance_sheet_stress_class = str(
        balance_sheet_stress_payload.get("balance_sheet_stress_class") or BALANCE_SHEET_STRESS_UNKNOWN
    ).upper()
    refinancing_risk_class = str(
        balance_sheet_stress_payload.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
    ).upper()
    balance_sheet_reasons = {
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_stress_reason_codes") or [])
        if str(code).strip()
    }
    balance_sheet_headwinds = {
        str(code)
        for code in (balance_sheet_stress_payload.get("balance_sheet_headwind_signals") or [])
        if str(code).strip()
    }
    balance_sheet_refs = [
        str(ref) for ref in (balance_sheet_stress_payload.get("derived_from") or []) if str(ref).strip()
    ]
    if balance_sheet_stress_class == HIGH_BALANCE_SHEET_STRESS:
        fragility_reason_codes.append(REASON_HIGH_BALANCE_SHEET_STRESS_HEADWIND)
        confidence_reason_codes.append(REASON_HIGH_BALANCE_SHEET_STRESS_HEADWIND)
        if fragility_status == FRAGILITY_LOW:
            fragility_status = FRAGILITY_MODERATE
        else:
            fragility_status = FRAGILITY_HIGH
        if confidence_class == CONFIDENCE_HIGH:
            confidence_class = CONFIDENCE_MEDIUM
        elif confidence_class in {CONFIDENCE_MEDIUM, CONFIDENCE_LOW}:
            confidence_class = CONFIDENCE_LOW
    elif balance_sheet_payload_present and balance_sheet_stress_class == LOW_BALANCE_SHEET_STRESS:
        confidence_reason_codes.append(REASON_LOW_BALANCE_SHEET_STRESS_SUPPORT)
    elif balance_sheet_payload_present and balance_sheet_stress_class == BALANCE_SHEET_STRESS_UNKNOWN:
        if REASON_BALANCE_SHEET_EVIDENCE_THIN in balance_sheet_reasons:
            fragility_reason_codes.append(REASON_BALANCE_SHEET_EVIDENCE_THIN)
            confidence_reason_codes.append(REASON_BALANCE_SHEET_EVIDENCE_THIN)

    if refinancing_risk_class == "HIGH_REFINANCING_RISK":
        fragility_reason_codes.append(REASON_HIGH_REFINANCING_RISK_HEADWIND)
        confidence_reason_codes.append(REASON_HIGH_REFINANCING_RISK_HEADWIND)
        fragility_status = FRAGILITY_HIGH
        if confidence_class != CONFIDENCE_UNKNOWN:
            confidence_class = CONFIDENCE_LOW
    if "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in balance_sheet_headwinds:
        fragility_reason_codes.append(REASON_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE)
        confidence_reason_codes.append(REASON_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE)
        fragility_status = FRAGILITY_HIGH
        if confidence_class != CONFIDENCE_UNKNOWN:
            confidence_class = CONFIDENCE_LOW

    maintenance_payload_present = bool(maintenance_capex_payload)
    maintenance_capex_class = str(
        maintenance_capex_payload.get("maintenance_capex_credibility_class")
        or MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
    ).upper()
    asset_intensity_class = str(
        maintenance_capex_payload.get("asset_intensity_class") or ASSET_INTENSITY_UNKNOWN
    ).upper()
    maintenance_reasons = {
        str(code)
        for code in (
            list(maintenance_capex_payload.get("asset_intensity_reason_codes") or [])
            + list(maintenance_capex_payload.get("maintenance_capex_credibility_reason_codes") or [])
        )
        if str(code).strip()
    }
    maintenance_headwinds = {
        str(code)
        for code in (maintenance_capex_payload.get("maintenance_capex_headwind_signals") or [])
        if str(code).strip()
    }
    maintenance_refs = [
        str(ref) for ref in (maintenance_capex_payload.get("derived_from") or []) if str(ref).strip()
    ]
    normalized_reasons = set(_normalized_reason_codes(intrinsic_payload))
    owner_earnings_selected = REASON_OWNER_EARNINGS_SELECTED in normalized_reasons

    if maintenance_capex_class == LOW_MAINTENANCE_CAPEX_CREDIBILITY:
        fragility_reason_codes.append(REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND)
        confidence_reason_codes.append(REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND)
        if owner_earnings_selected or "OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING" in maintenance_headwinds:
            fragility_reason_codes.append(REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE)
            confidence_reason_codes.append(REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE)
        if fragility_status == FRAGILITY_LOW:
            fragility_status = FRAGILITY_MODERATE
        else:
            fragility_status = FRAGILITY_HIGH
        if confidence_class == CONFIDENCE_HIGH:
            confidence_class = CONFIDENCE_MEDIUM
        elif confidence_class in {CONFIDENCE_MEDIUM, CONFIDENCE_LOW}:
            confidence_class = CONFIDENCE_LOW
    elif maintenance_payload_present and maintenance_capex_class == MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN:
        if REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN in maintenance_reasons:
            fragility_reason_codes.append(REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN)
            confidence_reason_codes.append(REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN)

    if asset_intensity_class == HIGH_ASSET_INTENSITY:
        fragility_reason_codes.append(REASON_HIGH_ASSET_INTENSITY_HEADWIND)
        confidence_reason_codes.append(REASON_HIGH_ASSET_INTENSITY_HEADWIND)
        if owner_earnings_selected:
            fragility_reason_codes.append(REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE)
            confidence_reason_codes.append(REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE)
        if fragility_status == FRAGILITY_LOW:
            fragility_status = FRAGILITY_MODERATE
        elif fragility_status != FRAGILITY_UNKNOWN:
            fragility_status = FRAGILITY_HIGH
        if confidence_class == CONFIDENCE_HIGH:
            confidence_class = CONFIDENCE_MEDIUM
        elif confidence_class in {CONFIDENCE_MEDIUM, CONFIDENCE_LOW}:
            confidence_class = CONFIDENCE_LOW
    elif maintenance_payload_present and asset_intensity_class == LOW_ASSET_INTENSITY:
        confidence_reason_codes.append(REASON_LOW_ASSET_INTENSITY_SUPPORT)

    fragility_reason_codes = _dedupe_refs(fragility_reason_codes)
    confidence_reason_codes = _dedupe_refs(confidence_reason_codes)

    claims = {
        "valuation_support_count": _claim(
            value=float(support_count),
            refs=support_refs,
            reason_code=support_count_reason_codes[0],
            status=OK,
        ),
        "valuation_convergence_band_pct": _claim(
            value=convergence_band_pct,
            refs=support_refs,
            reason_code=convergence_reason_codes[0] if convergence_reason_codes else convergence_status,
            status=OK if _is_num(convergence_band_pct) else UNKNOWN,
        ),
        "valuation_fragility_status": _claim(
            value=fragility_status,
            refs=support_refs
            + _claim_refs(intrinsic_payload, "normalized_earnings_power_value")
            + accounting_refs
            + balance_sheet_refs
            + maintenance_refs,
            reason_code=fragility_reason_codes[0] if fragility_reason_codes else UNKNOWN,
            status=OK if fragility_status != FRAGILITY_UNKNOWN else UNKNOWN,
        ),
        "valuation_confidence_class": _claim(
            value=confidence_class,
            refs=support_refs
            + _claim_refs(intrinsic_payload, "intrinsic_base")
            + accounting_refs
            + balance_sheet_refs
            + maintenance_refs,
            reason_code=confidence_reason_codes[0] if confidence_reason_codes else UNKNOWN,
            status=OK if confidence_class != CONFIDENCE_UNKNOWN else UNKNOWN,
        ),
    }

    derived_from = _dedupe_refs(
        support_refs
        + _claim_refs(intrinsic_payload, "normalized_earnings_power_value")
        + _claim_refs(intrinsic_payload, "intrinsic_floor")
        + _claim_refs(intrinsic_payload, "intrinsic_base")
        + _claim_refs(intrinsic_payload, "intrinsic_ceiling")
        + accounting_refs
        + balance_sheet_refs
        + maintenance_refs
        + [str(ref) for ref in (intrinsic_payload.get("derived_from") or []) if str(ref).strip()]
    )
    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "valuation_support_count": int(support_count),
        "valuation_support_types_present": support_types_present,
        "valuation_support_count_reason_codes": support_count_reason_codes,
        "valuation_convergence_status": convergence_status,
        "valuation_convergence_band_pct": _to_num(convergence_band_pct),
        "valuation_convergence_reason_codes": convergence_reason_codes,
        "valuation_fragility_status": fragility_status,
        "valuation_fragility_reason_codes": fragility_reason_codes,
        "valuation_confidence_class": confidence_class,
        "valuation_confidence_reason_codes": confidence_reason_codes,
        "valuation_status_snapshot": {
            "price_status": str(price_status or UNKNOWN).upper(),
            "shares_status": str(shares_status or UNKNOWN).upper(),
            "fcf_status": str(fcf_status or UNKNOWN).upper(),
            "facts_status": str(facts_status or UNKNOWN).upper(),
            "valuation_status": str(valuation_status or UNKNOWN).upper(),
        },
        "claims": claims,
        "derived_from": derived_from,
        "generated_at": utc_now_iso(),
    }


def write_valuation_confidence_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    coverage_rows_by_ticker: dict[str, dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("valuation_confidence_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("valuation_confidence_detail"), dict)
    }
    coverage_rows_by_ticker = dict(coverage_rows_by_ticker or {})
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
            coverage_row = coverage_rows_by_ticker.get(ticker, {})
            metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
            detail = compute_valuation_confidence(
                ticker=ticker,
                as_of_date=as_of_date,
                intrinsic_payload=score_row.get("intrinsic_discipline_detail")
                if isinstance(score_row.get("intrinsic_discipline_detail"), dict)
                else {},
                accounting_quality_payload=score_row.get("accounting_quality_detail")
                if isinstance(score_row.get("accounting_quality_detail"), dict)
                else {},
                balance_sheet_stress_payload=score_row.get("balance_sheet_stress_detail")
                if isinstance(score_row.get("balance_sheet_stress_detail"), dict)
                else {},
                maintenance_capex_payload=score_row.get("maintenance_capex_discipline_detail")
                if isinstance(score_row.get("maintenance_capex_discipline_detail"), dict)
                else {},
                price_status=coverage_row.get("price_status", UNKNOWN),
                shares_status=coverage_row.get("shares_status", UNKNOWN),
                fcf_status=coverage_row.get("fcf_status", UNKNOWN),
                facts_status=coverage_row.get("facts_status", UNKNOWN),
                valuation_status=coverage_row.get("valuation_gap_status", coverage_row.get("valuation_status", UNKNOWN)),
                epv_per_share=score_row.get("epv_per_share", metric_values.get("epv_per_share", UNKNOWN)),
                epv_refs=list(score_row.get("derived_from") or []),
                netnet_per_share=score_row.get("netnet_per_share", metric_values.get("netnet_per_share", UNKNOWN)),
                netnet_refs=list(score_row.get("derived_from") or []),
                existing_intrinsic_base=metric_values.get("intrinsic_per_share_base", metric_values.get("intrinsic_per_share_proxy", UNKNOWN)),
                existing_intrinsic_base_refs=list(score_row.get("derived_from") or []),
                existing_intrinsic_conservative=metric_values.get("intrinsic_per_share_conservative", UNKNOWN),
                existing_intrinsic_conservative_refs=list(score_row.get("derived_from") or []),
                cfg=cfg,
            )
        rows.append(detail)

    known_count = len(
        [
            row
            for row in rows
            if str(row.get("valuation_confidence_class") or CONFIDENCE_UNKNOWN) != CONFIDENCE_UNKNOWN
        ]
    )
    unknown_count = len(rows) - known_count
    high_confidence_rows = [
        row for row in rows if str(row.get("valuation_confidence_class") or "") == CONFIDENCE_HIGH
    ]
    high_fragility_rows = [
        row for row in rows if str(row.get("valuation_fragility_status") or "") == FRAGILITY_HIGH
    ]
    high_confidence_rows.sort(
        key=lambda row: (
            -int(row.get("valuation_support_count") or 0),
            float(row.get("valuation_convergence_band_pct"))
            if _is_num(row.get("valuation_convergence_band_pct"))
            else float("inf"),
            str(row.get("ticker") or ""),
        )
    )
    high_fragility_rows.sort(
        key=lambda row: (
            int(row.get("valuation_support_count") or 0),
            str(row.get("ticker") or ""),
        )
    )
    fragility_reason_counts: dict[str, int] = {}
    for row in rows:
        for code in (row.get("valuation_fragility_reason_codes") or []):
            token = str(code or "").strip()
            if not token or token in _POSITIVE_FRAGILITY_REASONS:
                continue
            fragility_reason_counts[token] = fragility_reason_counts.get(token, 0) + 1
    single_support_only_count = len(
        [
            row
            for row in rows
            if REASON_SINGLE_SUPPORT_ONLY in {str(code) for code in (row.get("valuation_fragility_reason_codes") or [])}
        ]
    )
    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "known_count": known_count,
        "unknown_count": unknown_count,
        "high_confidence_count": len(high_confidence_rows),
        "high_fragility_count": len(high_fragility_rows),
        "top_10_high_confidence": [
            {
                "ticker": str(row.get("ticker") or ""),
                "valuation_support_count": int(row.get("valuation_support_count") or 0),
                "valuation_support_types_present": [
                    str(value) for value in (row.get("valuation_support_types_present") or []) if str(value).strip()
                ],
                "valuation_convergence_status": str(row.get("valuation_convergence_status") or CONVERGENCE_UNKNOWN),
                "valuation_fragility_status": str(row.get("valuation_fragility_status") or FRAGILITY_UNKNOWN),
                "valuation_confidence_class": str(row.get("valuation_confidence_class") or CONFIDENCE_UNKNOWN),
            }
            for row in high_confidence_rows[:10]
        ],
        "top_10_high_fragility": [
            {
                "ticker": str(row.get("ticker") or ""),
                "valuation_support_count": int(row.get("valuation_support_count") or 0),
                "valuation_convergence_status": str(row.get("valuation_convergence_status") or CONVERGENCE_UNKNOWN),
                "valuation_fragility_status": str(row.get("valuation_fragility_status") or FRAGILITY_UNKNOWN),
                "valuation_fragility_reason_codes": [
                    str(code)
                    for code in (row.get("valuation_fragility_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in high_fragility_rows[:10]
        ],
        "fragility_reason_counts": dict(
            sorted(fragility_reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ),
        "single_support_only_count": int(single_support_only_count),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["valuation_confidence_path"] = str(output_path)
    return payload


def _valuation_confidence_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "valuation_confidence.json",
        cfg.sectors_dir / run_id / "valuation_confidence.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_valuation_confidence(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _valuation_confidence_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "valuation_confidence_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or len(rows)),
        "known_count": int(payload.get("known_count") or 0),
        "unknown_count": int(payload.get("unknown_count") or 0),
        "high_confidence_count": int(payload.get("high_confidence_count") or 0),
        "high_fragility_count": int(payload.get("high_fragility_count") or 0),
        "top_10_high_confidence": [
            row for row in (payload.get("top_10_high_confidence") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_high_fragility": [
            row for row in (payload.get("top_10_high_fragility") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "fragility_reason_counts": payload.get("fragility_reason_counts")
        if isinstance(payload.get("fragility_reason_counts"), dict)
        else {},
        "single_support_only_count": int(payload.get("single_support_only_count") or 0),
        "valuation_confidence_path": str(path),
    }
