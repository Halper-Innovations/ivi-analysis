from __future__ import annotations

from typing import Any

from app.config import AppConfig, get_config


UNKNOWN = "UNKNOWN"
_CONFIDENCE_ORDER = {
    "HIGH_CONFIDENCE": 0,
    "MEDIUM_CONFIDENCE": 1,
    "LOW_CONFIDENCE": 2,
    "CONFIDENCE_UNKNOWN": 3,
}
_INTEGRITY_ORDER = {
    "INTEGRITY_OK": 0,
    "INTEGRITY_WARNING": 1,
    "INTEGRITY_SUSPECT": 2,
    "INTEGRITY_UNKNOWN": 3,
}
_VALUE_TYPE_ORDER = {
    "ASSET_BACKED_VALUE": 0,
    "EARNINGS_POWER_VALUE": 1,
    "QUALITY_VALUE": 2,
    "CYCLICAL_VALUE": 3,
    "FRAGILE_VALUE": 4,
    "UNKNOWN_VALUE_TYPE": 5,
}
_READINESS_ORDER = {
    "INVESTABLE_NOW": 0,
    "RESEARCH_WORTHY_NOT_READY": 1,
    "WATCH_ONLY": 2,
    "NOT_INVESTABLE": 3,
    "READINESS_UNKNOWN": 4,
}
_ACCOUNTING_ORDER = {
    "HIGH_ACCOUNTING_QUALITY": 0,
    "MODERATE_ACCOUNTING_QUALITY": 1,
    "ACCOUNTING_QUALITY_UNKNOWN": 2,
    "LOW_ACCOUNTING_QUALITY": 3,
}
_BALANCE_SHEET_STRESS_ORDER = {
    "LOW_BALANCE_SHEET_STRESS": 0,
    "MODERATE_BALANCE_SHEET_STRESS": 1,
    "BALANCE_SHEET_STRESS_UNKNOWN": 2,
    "HIGH_BALANCE_SHEET_STRESS": 3,
}
_REFINANCING_RISK_ORDER = {
    "LOW_REFINANCING_RISK": 0,
    "MODERATE_REFINANCING_RISK": 1,
    "REFINANCING_RISK_UNKNOWN": 2,
    "HIGH_REFINANCING_RISK": 3,
}
_RETURNS_PERSISTENCE_ORDER = {
    "HIGH_RETURNS_PERSISTENCE": 0,
    "MODERATE_RETURNS_PERSISTENCE": 1,
    "RETURNS_PERSISTENCE_UNKNOWN": 2,
    "LOW_RETURNS_PERSISTENCE": 3,
}
_REVENUE_DEPENDENCE_ORDER = {
    "LOW_REVENUE_DEPENDENCE_RISK": 0,
    "MODERATE_REVENUE_DEPENDENCE_RISK": 1,
    "REVENUE_DEPENDENCE_UNKNOWN": 2,
    "HIGH_REVENUE_DEPENDENCE_RISK": 3,
}
_MAINTENANCE_CAPEX_CREDIBILITY_ORDER = {
    "HIGH_MAINTENANCE_CAPEX_CREDIBILITY": 0,
    "MODERATE_MAINTENANCE_CAPEX_CREDIBILITY": 1,
    "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN": 2,
    "LOW_MAINTENANCE_CAPEX_CREDIBILITY": 3,
}
_ASSET_INTENSITY_ORDER = {
    "LOW_ASSET_INTENSITY": 0,
    "MODERATE_ASSET_INTENSITY": 1,
    "ASSET_INTENSITY_UNKNOWN": 2,
    "HIGH_ASSET_INTENSITY": 3,
}
# Impairment order: less-impaired → more-impaired
_IMPAIRMENT_ORDER = {
    "TEMPORARY_WEAKNESS": 0,
    "EVIDENCE_DEGRADED_NOT_ASSESSABLE": 1,
    "STRUCTURALLY_WEAK_NOT_IMPAIRED": 2,
    "PROBABLE_IMPAIRMENT": 3,
    "CLEAR_IMPAIRMENT": 4,
    "IMPAIRMENT_UNKNOWN": 5,
}
_NORMALIZATION_CREDIBILITY_ORDER = {
    "HIGH_NORMALIZATION_CREDIBILITY": 0,
    "MODERATE_NORMALIZATION_CREDIBILITY": 1,
    "NORMALIZATION_CREDIBILITY_UNKNOWN": 2,
    "LOW_NORMALIZATION_CREDIBILITY": 3,
}
_CAPITAL_ALLOCATION_ORDER = {
    "OWNER_FRIENDLY_DISCIPLINED": 0,
    "MIXED_CAPITAL_ALLOCATION": 1,
    "CAPITAL_ALLOCATION_UNKNOWN": 2,
    "OWNER_DILUTIVE_OR_DESTRUCTIVE": 3,
}
_REINVESTMENT_ORDER = {
    "HIGH_REINVESTMENT_EFFICIENCY": 0,
    "MODERATE_REINVESTMENT_EFFICIENCY": 1,
    "REINVESTMENT_EFFICIENCY_UNKNOWN": 2,
    "LOW_REINVESTMENT_EFFICIENCY": 3,
}


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


def _dedupe_refs(refs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        token = str(ref).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _status_rank(status: str) -> int:
    token = str(status or "").upper()
    if token == "PASS":
        return 0
    if token == "WATCH":
        return 1
    return 2


def _confidence_rank(value: Any) -> int:
    token = str(value or "CONFIDENCE_UNKNOWN").upper()
    return _CONFIDENCE_ORDER.get(token, len(_CONFIDENCE_ORDER))


def _value_type_rank(value: Any) -> int:
    token = str(value or "UNKNOWN_VALUE_TYPE").upper()
    return _VALUE_TYPE_ORDER.get(token, len(_VALUE_TYPE_ORDER))


def _integrity_rank(value: Any) -> int:
    token = str(value or "INTEGRITY_UNKNOWN").upper()
    return _INTEGRITY_ORDER.get(token, len(_INTEGRITY_ORDER))


def _readiness_rank(value: Any) -> int:
    token = str(value or "READINESS_UNKNOWN").upper()
    return _READINESS_ORDER.get(token, len(_READINESS_ORDER))


def _accounting_quality_rank(value: Any) -> int:
    token = str(value or "ACCOUNTING_QUALITY_UNKNOWN").upper()
    return _ACCOUNTING_ORDER.get(token, len(_ACCOUNTING_ORDER))


def _balance_sheet_stress_rank(value: Any) -> int:
    token = str(value or "BALANCE_SHEET_STRESS_UNKNOWN").upper()
    return _BALANCE_SHEET_STRESS_ORDER.get(token, len(_BALANCE_SHEET_STRESS_ORDER))


def _refinancing_risk_rank(value: Any) -> int:
    token = str(value or "REFINANCING_RISK_UNKNOWN").upper()
    return _REFINANCING_RISK_ORDER.get(token, len(_REFINANCING_RISK_ORDER))


def _impairment_rank(value: Any) -> int:
    token = str(value or "IMPAIRMENT_UNKNOWN").upper()
    return _IMPAIRMENT_ORDER.get(token, len(_IMPAIRMENT_ORDER))


def _normalization_credibility_rank(value: Any) -> int:
    token = str(value or "NORMALIZATION_CREDIBILITY_UNKNOWN").upper()
    return _NORMALIZATION_CREDIBILITY_ORDER.get(token, len(_NORMALIZATION_CREDIBILITY_ORDER))


def _capital_allocation_rank(value: Any) -> int:
    token = str(value or "CAPITAL_ALLOCATION_UNKNOWN").upper()
    return _CAPITAL_ALLOCATION_ORDER.get(token, len(_CAPITAL_ALLOCATION_ORDER))


def _reinvestment_rank(value: Any) -> int:
    token = str(value or "REINVESTMENT_EFFICIENCY_UNKNOWN").upper()
    return _REINVESTMENT_ORDER.get(token, len(_REINVESTMENT_ORDER))


def _returns_persistence_rank(value: Any) -> int:
    token = str(value or "RETURNS_PERSISTENCE_UNKNOWN").upper()
    return _RETURNS_PERSISTENCE_ORDER.get(token, len(_RETURNS_PERSISTENCE_ORDER))


def _revenue_dependence_rank(value: Any) -> int:
    token = str(value or "REVENUE_DEPENDENCE_UNKNOWN").upper()
    return _REVENUE_DEPENDENCE_ORDER.get(token, len(_REVENUE_DEPENDENCE_ORDER))


def _maintenance_capex_credibility_rank(value: Any) -> int:
    token = str(value or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN").upper()
    return _MAINTENANCE_CAPEX_CREDIBILITY_ORDER.get(token, len(_MAINTENANCE_CAPEX_CREDIBILITY_ORDER))


def _asset_intensity_rank(value: Any) -> int:
    token = str(value or "ASSET_INTENSITY_UNKNOWN").upper()
    return _ASSET_INTENSITY_ORDER.get(token, len(_ASSET_INTENSITY_ORDER))


def ranking_sort_key(
    row: dict[str, Any],
    *,
    policy: str = "composite_first",
) -> tuple[Any, ...]:
    policy_norm = str(policy or "composite_first").strip().lower()
    if policy_norm == "value_first_not_impaired":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        readiness_class = row.get(
            "investment_readiness_class",
            metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
        )
        impairment_class = row.get(
            "impairment_class_primary",
            metric_values.get("impairment_class_primary", "IMPAIRMENT_UNKNOWN"),
        )
        confidence_class = row.get(
            "valuation_confidence_class",
            metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        )
        integrity_class = row.get(
            "valuation_integrity_class",
            metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
        )
        value_type_primary = row.get(
            "value_type_primary",
            metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"),
        )
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(readiness_class),
            _impairment_rank(impairment_class),
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            _confidence_rank(confidence_class),
            _integrity_rank(integrity_class),
            _value_type_rank(value_type_primary),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_normalization_aware":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        readiness_class = row.get(
            "investment_readiness_class",
            metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
        )
        impairment_class = row.get(
            "impairment_class_primary",
            metric_values.get("impairment_class_primary", "IMPAIRMENT_UNKNOWN"),
        )
        normalization_credibility_class = row.get(
            "normalization_credibility_class",
            metric_values.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN"),
        )
        mos_to_floor = row.get("mos_to_floor", metric_values.get("mos_to_floor", UNKNOWN))
        confidence_class = row.get(
            "valuation_confidence_class",
            metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        )
        integrity_class = row.get(
            "valuation_integrity_class",
            metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
        )
        value_type_primary = row.get(
            "value_type_primary",
            metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"),
        )
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(readiness_class),
            _impairment_rank(impairment_class),
            _normalization_credibility_rank(normalization_credibility_class),
            -float(mos_to_floor) if _is_num(mos_to_floor) else float("inf"),
            _confidence_rank(confidence_class),
            _integrity_rank(integrity_class),
            _value_type_rank(value_type_primary),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_owner_capture":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        readiness_class = row.get(
            "investment_readiness_class",
            metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
        )
        impairment_class = row.get(
            "impairment_class_primary",
            metric_values.get("impairment_class_primary", "IMPAIRMENT_UNKNOWN"),
        )
        capital_allocation_discipline_class = row.get(
            "capital_allocation_discipline_class",
            metric_values.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN"),
        )
        mos_to_floor = row.get("mos_to_floor", metric_values.get("mos_to_floor", UNKNOWN))
        confidence_class = row.get(
            "valuation_confidence_class",
            metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        )
        integrity_class = row.get(
            "valuation_integrity_class",
            metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
        )
        value_type_primary = row.get(
            "value_type_primary",
            metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"),
        )
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(readiness_class),
            _impairment_rank(impairment_class),
            _capital_allocation_rank(capital_allocation_discipline_class),
            -float(mos_to_floor) if _is_num(mos_to_floor) else float("inf"),
            _confidence_rank(confidence_class),
            _integrity_rank(integrity_class),
            _value_type_rank(value_type_primary),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_reinvestment":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        readiness_class = row.get(
            "investment_readiness_class",
            metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
        )
        confidence_class = row.get(
            "valuation_confidence_class",
            metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        )
        integrity_class = row.get(
            "valuation_integrity_class",
            metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
        )
        reinvestment_class = row.get(
            "reinvestment_efficiency_class",
            metric_values.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN"),
        )
        capital_allocation_discipline_class = row.get(
            "capital_allocation_discipline_class",
            metric_values.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN"),
        )
        value_type_primary = row.get(
            "value_type_primary",
            metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"),
        )
        normalization_credibility_class = row.get(
            "normalization_credibility_class",
            metric_values.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN"),
        )
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(readiness_class),
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            _confidence_rank(confidence_class),
            _integrity_rank(integrity_class),
            _reinvestment_rank(reinvestment_class),
            _capital_allocation_rank(capital_allocation_discipline_class),
            _value_type_rank(value_type_primary),
            _normalization_credibility_rank(normalization_credibility_class),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_cash_earnings":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(row.get("investment_readiness_class", metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"))),
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            _confidence_rank(
                row.get("valuation_confidence_class", metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"))
            ),
            _integrity_rank(
                row.get("valuation_integrity_class", metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"))
            ),
            _accounting_quality_rank(
                row.get("accounting_quality_class", metric_values.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN"))
            ),
            _reinvestment_rank(
                row.get("reinvestment_efficiency_class", metric_values.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN"))
            ),
            _capital_allocation_rank(
                row.get("capital_allocation_discipline_class", metric_values.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN"))
            ),
            _value_type_rank(row.get("value_type_primary", metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"))),
            _normalization_credibility_rank(
                row.get("normalization_credibility_class", metric_values.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN"))
            ),
            -float(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            if _is_num(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            else float("inf"),
            -float(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            if _is_num(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            else float("inf"),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_balance_sheet":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(
                row.get(
                    "investment_readiness_class",
                    metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
                )
            ),
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            _confidence_rank(
                row.get(
                    "valuation_confidence_class",
                    metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
                )
            ),
            _integrity_rank(
                row.get(
                    "valuation_integrity_class",
                    metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
                )
            ),
            _balance_sheet_stress_rank(
                row.get(
                    "balance_sheet_stress_class",
                    metric_values.get("balance_sheet_stress_class", "BALANCE_SHEET_STRESS_UNKNOWN"),
                )
            ),
            _refinancing_risk_rank(
                row.get(
                    "refinancing_risk_class",
                    metric_values.get("refinancing_risk_class", "REFINANCING_RISK_UNKNOWN"),
                )
            ),
            _accounting_quality_rank(
                row.get(
                    "accounting_quality_class",
                    metric_values.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN"),
                )
            ),
            _reinvestment_rank(
                row.get(
                    "reinvestment_efficiency_class",
                    metric_values.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN"),
                )
            ),
            _capital_allocation_rank(
                row.get(
                    "capital_allocation_discipline_class",
                    metric_values.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN"),
                )
            ),
            _value_type_rank(
                row.get("value_type_primary", metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"))
            ),
            _normalization_credibility_rank(
                row.get(
                    "normalization_credibility_class",
                    metric_values.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN"),
                )
            ),
            -float(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            if _is_num(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            else float("inf"),
            -float(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            if _is_num(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            else float("inf"),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_durable_returns":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(
                row.get(
                    "investment_readiness_class",
                    metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
                )
            ),
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            _confidence_rank(
                row.get(
                    "valuation_confidence_class",
                    metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
                )
            ),
            _integrity_rank(
                row.get(
                    "valuation_integrity_class",
                    metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
                )
            ),
            _returns_persistence_rank(
                row.get(
                    "returns_persistence_class",
                    metric_values.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN"),
                )
            ),
            _reinvestment_rank(
                row.get(
                    "reinvestment_efficiency_class",
                    metric_values.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN"),
                )
            ),
            _capital_allocation_rank(
                row.get(
                    "capital_allocation_discipline_class",
                    metric_values.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN"),
                )
            ),
            _accounting_quality_rank(
                row.get(
                    "accounting_quality_class",
                    metric_values.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN"),
                )
            ),
            _value_type_rank(
                row.get("value_type_primary", metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"))
            ),
            _normalization_credibility_rank(
                row.get(
                    "normalization_credibility_class",
                    metric_values.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN"),
                )
            ),
            -float(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            if _is_num(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            else float("inf"),
            -float(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            if _is_num(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            else float("inf"),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_revenue_resilience":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(
                row.get(
                    "investment_readiness_class",
                    metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
                )
            ),
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            _confidence_rank(
                row.get(
                    "valuation_confidence_class",
                    metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
                )
            ),
            _integrity_rank(
                row.get(
                    "valuation_integrity_class",
                    metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
                )
            ),
            _returns_persistence_rank(
                row.get(
                    "returns_persistence_class",
                    metric_values.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN"),
                )
            ),
            _revenue_dependence_rank(
                row.get(
                    "revenue_dependence_risk_class",
                    metric_values.get("revenue_dependence_risk_class", "REVENUE_DEPENDENCE_UNKNOWN"),
                )
            ),
            _accounting_quality_rank(
                row.get(
                    "accounting_quality_class",
                    metric_values.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN"),
                )
            ),
            _reinvestment_rank(
                row.get(
                    "reinvestment_efficiency_class",
                    metric_values.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN"),
                )
            ),
            _capital_allocation_rank(
                row.get(
                    "capital_allocation_discipline_class",
                    metric_values.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN"),
                )
            ),
            _value_type_rank(
                row.get("value_type_primary", metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"))
            ),
            _normalization_credibility_rank(
                row.get(
                    "normalization_credibility_class",
                    metric_values.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN"),
                )
            ),
            -float(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            if _is_num(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            else float("inf"),
            -float(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            if _is_num(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            else float("inf"),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_owner_earnings_hardness":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(
                row.get(
                    "investment_readiness_class",
                    metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
                )
            ),
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            _confidence_rank(
                row.get(
                    "valuation_confidence_class",
                    metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
                )
            ),
            _integrity_rank(
                row.get(
                    "valuation_integrity_class",
                    metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
                )
            ),
            _maintenance_capex_credibility_rank(
                row.get(
                    "maintenance_capex_credibility_class",
                    metric_values.get(
                        "maintenance_capex_credibility_class",
                        "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN",
                    ),
                )
            ),
            _asset_intensity_rank(
                row.get(
                    "asset_intensity_class",
                    metric_values.get("asset_intensity_class", "ASSET_INTENSITY_UNKNOWN"),
                )
            ),
            _accounting_quality_rank(
                row.get(
                    "accounting_quality_class",
                    metric_values.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN"),
                )
            ),
            _reinvestment_rank(
                row.get(
                    "reinvestment_efficiency_class",
                    metric_values.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN"),
                )
            ),
            _capital_allocation_rank(
                row.get(
                    "capital_allocation_discipline_class",
                    metric_values.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN"),
                )
            ),
            _returns_persistence_rank(
                row.get(
                    "returns_persistence_class",
                    metric_values.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN"),
                )
            ),
            _value_type_rank(
                row.get("value_type_primary", metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"))
            ),
            _normalization_credibility_rank(
                row.get(
                    "normalization_credibility_class",
                    metric_values.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN"),
                )
            ),
            -float(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            if _is_num(row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN)))
            else float("inf"),
            -float(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            if _is_num(row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN)))
            else float("inf"),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_ready":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        readiness_class = row.get(
            "investment_readiness_class",
            metric_values.get("investment_readiness_class", "READINESS_UNKNOWN"),
        )
        confidence_class = row.get(
            "valuation_confidence_class",
            metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        )
        integrity_class = row.get(
            "valuation_integrity_class",
            metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
        )
        value_type_primary = row.get(
            "value_type_primary",
            metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"),
        )
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            _readiness_rank(readiness_class),
            -float(row.get("mos_to_floor")) if _is_num(row.get("mos_to_floor")) else float("inf"),
            _confidence_rank(confidence_class),
            _integrity_rank(integrity_class),
            _value_type_rank(value_type_primary),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_trustworthy":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        implied_return = row.get("implied_return_base", metric_values.get("implied_return_base", UNKNOWN))
        mos_to_floor = row.get("mos_to_floor", metric_values.get("mos_to_floor", UNKNOWN))
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        confidence_class = row.get(
            "valuation_confidence_class",
            metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        )
        integrity_class = row.get(
            "valuation_integrity_class",
            metric_values.get("valuation_integrity_class", "INTEGRITY_UNKNOWN"),
        )
        value_type_primary = row.get(
            "value_type_primary",
            metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"),
        )
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            -float(mos_to_floor) if _is_num(mos_to_floor) else float("inf"),
            -float(implied_return) if _is_num(implied_return) else float("inf"),
            _confidence_rank(confidence_class),
            _integrity_rank(integrity_class),
            _value_type_rank(value_type_primary),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_typed":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        implied_return = row.get("implied_return_base", metric_values.get("implied_return_base", UNKNOWN))
        mos_to_floor = row.get("mos_to_floor", metric_values.get("mos_to_floor", UNKNOWN))
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        confidence_class = row.get(
            "valuation_confidence_class",
            metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        )
        value_type_primary = row.get(
            "value_type_primary",
            metric_values.get("value_type_primary", "UNKNOWN_VALUE_TYPE"),
        )
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            -float(mos_to_floor) if _is_num(mos_to_floor) else float("inf"),
            -float(implied_return) if _is_num(implied_return) else float("inf"),
            _confidence_rank(confidence_class),
            _value_type_rank(value_type_primary),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_confidence":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        implied_return = row.get("implied_return_base", metric_values.get("implied_return_base", UNKNOWN))
        mos_to_floor = row.get("mos_to_floor", metric_values.get("mos_to_floor", UNKNOWN))
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        confidence_class = row.get(
            "valuation_confidence_class",
            metric_values.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN"),
        )
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            -float(mos_to_floor) if _is_num(mos_to_floor) else float("inf"),
            -float(implied_return) if _is_num(implied_return) else float("inf"),
            _confidence_rank(confidence_class),
            -float(row.get("mos_epv")) if _is_num(row.get("mos_epv")) else float("inf"),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_discipline":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        implied_return = row.get("implied_return_base", metric_values.get("implied_return_base", UNKNOWN))
        mos_to_floor = row.get("mos_to_floor", metric_values.get("mos_to_floor", UNKNOWN))
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            -float(mos_to_floor) if _is_num(mos_to_floor) else float("inf"),
            -float(implied_return) if _is_num(implied_return) else float("inf"),
            -float(row.get("mos_epv")) if _is_num(row.get("mos_epv")) else float("inf"),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_intangible":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        implied_return = row.get("implied_return_base", metric_values.get("implied_return_base", UNKNOWN))
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            -float(implied_return) if _is_num(implied_return) else float("inf"),
            -float(row.get("mos_epv")) if _is_num(row.get("mos_epv")) else float("inf"),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -float(intangible_total) if _is_num(intangible_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_quality":
        metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
        implied_return = row.get("implied_return_base", metric_values.get("implied_return_base", UNKNOWN))
        owner_yield = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
        oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        return (
            _status_rank(str(row.get("scout_status") or row.get("value_gate_status") or "")),
            -float(implied_return) if _is_num(implied_return) else float("inf"),
            -float(row.get("mos_epv")) if _is_num(row.get("mos_epv")) else float("inf"),
            -float(owner_yield) if _is_num(owner_yield) else float("inf"),
            -float(oe_quality_total) if _is_num(oe_quality_total) else float("inf"),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )

    composite = row.get("composite_score_total", UNKNOWN)
    mos_epv = row.get("mos_epv", UNKNOWN)
    mos_netnet = row.get("mos_netnet", UNKNOWN)
    yield_gate = row.get("yield_gate_value_used", UNKNOWN)
    return (
        _status_rank(str(row.get("scout_status") or "")),
        -float(composite) if _is_num(composite) else float("inf"),
        -float(mos_epv) if _is_num(mos_epv) else float("inf"),
        -float(mos_netnet) if _is_num(mos_netnet) else float("inf"),
        -float(yield_gate) if _is_num(yield_gate) else float("inf"),
        str(row.get("ticker") or ""),
    )


def composite_thresholds_from_config(cfg: AppConfig | None = None) -> dict[str, float]:
    cfg = cfg or get_config()
    return {
        "gd_mos_high": float(getattr(cfg, "composite_gd_mos_high", 0.50)),
        "gd_mos_mid": float(getattr(cfg, "composite_gd_mos_mid", 0.30)),
        "gd_mos_low": float(getattr(cfg, "composite_gd_mos_low", 0.10)),
        "yield_high": float(getattr(cfg, "composite_yield_high", 0.08)),
        "yield_mid": float(getattr(cfg, "composite_yield_mid", 0.05)),
        "yield_low": float(getattr(cfg, "composite_yield_low", 0.03)),
        "quality_roic_high": float(getattr(cfg, "composite_quality_roic_high", 0.15)),
        "quality_roic_mid": float(getattr(cfg, "composite_quality_roic_mid", 0.10)),
        "quality_roic_low": float(getattr(cfg, "composite_quality_roic_low", 0.05)),
        "risk_dilution_warn": float(getattr(cfg, "composite_risk_dilution_warn", 0.06)),
        "risk_leverage_warn": float(getattr(cfg, "composite_risk_leverage_warn", 2.5)),
        "risk_keyword_warn": float(getattr(cfg, "composite_risk_keyword_warn", 2.0)),
    }


def compute_composite_score(
    row: dict[str, Any],
    gd: dict[str, Any] | None = None,
    yield_data: dict[str, Any] | None = None,
    fundamentals: dict[str, Any] | None = None,
    valuation_cov: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or composite_thresholds_from_config()
    gd = gd or {}
    yield_data = yield_data or {}
    fundamentals = fundamentals or {}
    valuation_cov = valuation_cov or {}
    risk = risk or {}

    metric_values = row.get("metric_values") if isinstance(row.get("metric_values"), dict) else {}
    inputs_used = row.get("inputs_used") if isinstance(row.get("inputs_used"), dict) else {}
    row_refs = [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()]

    mos_epv = gd.get("mos_epv", row.get("mos_epv", metric_values.get("mos_epv", UNKNOWN)))
    mos_netnet = gd.get("mos_netnet", row.get("mos_netnet", metric_values.get("mos_netnet", UNKNOWN)))
    best_mos = (
        max(float(mos_epv), float(mos_netnet))
        if _is_num(mos_epv) and _is_num(mos_netnet)
        else (float(mos_epv) if _is_num(mos_epv) else (float(mos_netnet) if _is_num(mos_netnet) else UNKNOWN))
    )

    gd_score = 0.0
    gd_reason = "MISSING_GD_MOS"
    if _is_num(best_mos):
        value = float(best_mos)
        if value >= thresholds["gd_mos_high"] + 0.50:
            gd_score = 40.0
        elif value >= thresholds["gd_mos_high"]:
            gd_score = 34.0
        elif value >= thresholds["gd_mos_mid"]:
            gd_score = 28.0
        elif value >= thresholds["gd_mos_low"]:
            gd_score = 20.0
        elif value >= 0.0:
            gd_score = 12.0
        elif value >= -0.20:
            gd_score = 6.0
        gd_reason = "OK"

    yield_gate = yield_data.get("yield_gate_value_used", row.get("yield_gate_value_used", UNKNOWN))
    yield_score = 0.0
    yield_reason = "MISSING_YIELD"
    if _is_num(yield_gate):
        y = float(yield_gate)
        if y >= thresholds["yield_high"] + 0.02:
            yield_score = 25.0
        elif y >= thresholds["yield_high"]:
            yield_score = 22.0
        elif y >= thresholds["yield_mid"]:
            yield_score = 18.0
        elif y >= thresholds["yield_low"]:
            yield_score = 12.0
        elif y >= 0.01:
            yield_score = 6.0
        elif y > 0:
            yield_score = 3.0
        yield_reason = "OK"

    roic_proxy = fundamentals.get("roic_proxy", metric_values.get("roic_proxy", UNKNOWN))
    margin_trend = fundamentals.get(
        "fcf_margin_trend_slope",
        valuation_cov.get("fcf_margin_trend_slope", metric_values.get("fcf_margin_trend_slope", UNKNOWN)),
    )
    cfo_value = fundamentals.get("cfo_value", metric_values.get("cfo_value", UNKNOWN))
    fcf_value = fundamentals.get("fcf_value", metric_values.get("fcf_value", UNKNOWN))
    cash_conversion = float(fcf_value) / float(cfo_value) if _is_num(fcf_value) and _is_num(cfo_value) and float(cfo_value) > 0 else UNKNOWN
    oe_quality_total = row.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
    intangible_total = row.get("intangible_economics_total", metric_values.get("intangible_economics_total", UNKNOWN))

    quality_score = 0.0
    quality_known = False
    if _is_num(roic_proxy):
        quality_known = True
        r = float(roic_proxy)
        if r >= thresholds["quality_roic_high"]:
            quality_score += 8.0
        elif r >= thresholds["quality_roic_mid"]:
            quality_score += 6.0
        elif r >= thresholds["quality_roic_low"]:
            quality_score += 4.0
        elif r > 0:
            quality_score += 2.0
    if _is_num(margin_trend):
        quality_known = True
        m = float(margin_trend)
        if m >= 0.02:
            quality_score += 6.0
        elif m > 0.0:
            quality_score += 4.0
        elif m >= -0.02:
            quality_score += 2.0
    if _is_num(cash_conversion):
        quality_known = True
        c = float(cash_conversion)
        if c >= 0.80:
            quality_score += 6.0
        elif c >= 0.50:
            quality_score += 4.0
        elif c > 0.0:
            quality_score += 2.0
    if _is_num(oe_quality_total):
        quality_known = True
        oe = float(oe_quality_total)
        if oe >= 12.0:
            quality_score += 4.0
        elif oe >= 9.0:
            quality_score += 3.0
        elif oe >= 6.0:
            quality_score += 2.0
        elif oe >= 3.0:
            quality_score += 1.0
    if _is_num(intangible_total):
        quality_known = True
        intangible = float(intangible_total)
        if intangible >= 12.0:
            quality_score += 3.0
        elif intangible >= 9.0:
            quality_score += 2.0
        elif intangible >= 6.0:
            quality_score += 1.0
        elif intangible >= 3.0:
            quality_score += 0.5
    quality_score = max(0.0, min(20.0, quality_score))
    quality_reason = "OK" if quality_known else "MISSING_QUALITY_INPUTS"

    dilution_rate = risk.get("dilution_rate", metric_values.get("dilution_rate", UNKNOWN))
    net_debt_to_cfo = risk.get("net_debt_to_cfo", metric_values.get("net_debt_to_cfo", UNKNOWN))
    risk_keyword_delta = risk.get("risk_factor_keyword_delta", row.get("risk_factor_keyword_delta", UNKNOWN))
    risk_penalty = 0.0
    if _is_num(dilution_rate):
        d = float(dilution_rate)
        if d >= 0.10:
            risk_penalty -= 5.0
        elif d >= thresholds["risk_dilution_warn"]:
            risk_penalty -= 3.0
        elif d >= 0.03:
            risk_penalty -= 2.0
    if _is_num(net_debt_to_cfo):
        l = float(net_debt_to_cfo)
        if l > 4.0:
            risk_penalty -= 5.0
        elif l > thresholds["risk_leverage_warn"]:
            risk_penalty -= 3.0
        elif l > 1.5:
            risk_penalty -= 2.0
    if _is_num(risk_keyword_delta):
        rk = float(risk_keyword_delta)
        if rk > 5.0:
            risk_penalty -= 5.0
        elif rk > thresholds["risk_keyword_warn"]:
            risk_penalty -= 3.0
        elif rk > 0:
            risk_penalty -= 1.0
    risk_penalty = max(-15.0, min(0.0, risk_penalty))

    composite = float(gd_score + yield_score + quality_score + risk_penalty)
    composite = max(0.0, min(100.0, round(composite, 6)))
    known_count = len(
        [
            item
            for item in [best_mos, yield_gate, roic_proxy, margin_trend, cash_conversion, oe_quality_total, intangible_total]
            if _is_num(item)
        ]
    )
    composite_status = "OK" if known_count >= 2 else "UNKNOWN"
    composite_reason = "OK" if composite_status == "OK" else "MISSING_COMPONENT_INPUTS"

    gd_refs = [str(ref) for ref in ((gd.get("derived_from") or []) if isinstance(gd, dict) else []) if str(ref).strip()]
    yield_refs = [str(ref) for ref in ((yield_data.get("derived_from") or []) if isinstance(yield_data, dict) else []) if str(ref).strip()]
    quality_refs = []
    if isinstance(inputs_used.get("cfo_value"), dict):
        quality_refs.extend([str(ref) for ref in (inputs_used.get("cfo_value", {}).get("derived_from") or []) if str(ref).strip()])
    if isinstance(inputs_used.get("fcf_value"), dict):
        quality_refs.extend([str(ref) for ref in (inputs_used.get("fcf_value", {}).get("derived_from") or []) if str(ref).strip()])
    risk_refs = []
    if isinstance(inputs_used.get("dilution_rate_shares_cagr"), dict):
        risk_refs.extend(
            [str(ref) for ref in (inputs_used.get("dilution_rate_shares_cagr", {}).get("derived_from") or []) if str(ref).strip()]
        )
    if isinstance(inputs_used.get("net_debt_proxy"), dict):
        risk_refs.extend([str(ref) for ref in (inputs_used.get("net_debt_proxy", {}).get("derived_from") or []) if str(ref).strip()])

    return {
        "composite_score_total": composite,
        "components": {
            "gd_score": round(float(gd_score), 6),
            "yield_score": round(float(yield_score), 6),
            "quality_score": round(float(quality_score), 6),
            "risk_penalty": round(float(risk_penalty), 6),
        },
        "status": composite_status,
        "primary_reason_code": composite_reason,
        "component_reason_codes": {
            "gd": gd_reason,
            "yield": yield_reason,
            "quality": quality_reason,
        },
        "inputs": {
            "best_mos": _to_num(best_mos),
            "yield_gate_value_used": _to_num(yield_gate),
            "roic_proxy": _to_num(roic_proxy),
            "fcf_margin_trend_slope": _to_num(margin_trend),
            "cash_conversion_proxy": _to_num(cash_conversion),
            "oe_quality_total": _to_num(oe_quality_total),
            "intangible_economics_total": _to_num(intangible_total),
            "dilution_rate": _to_num(dilution_rate),
            "net_debt_to_cfo": _to_num(net_debt_to_cfo),
            "risk_factor_keyword_delta": _to_num(risk_keyword_delta),
        },
        "thresholds_used": thresholds,
        "derived_from": _dedupe_refs(gd_refs + yield_refs + quality_refs + risk_refs + row_refs),
    }
