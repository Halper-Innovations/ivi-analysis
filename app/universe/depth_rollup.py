from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso
from app.valuation.evidence_sufficiency import compute_evidence_sufficiency
from app.valuation.investment_readiness import compute_investment_readiness
from app.valuation.intrinsic_discipline import compute_intrinsic_discipline
from app.valuation.intangible_economics import compute_intangible_economics
from app.valuation.owner_earnings_quality import compute_owner_earnings_quality
from app.valuation.accounting_quality import compute_accounting_quality
from app.valuation.balance_sheet_stress import compute_balance_sheet_stress
from app.valuation.maintenance_capex_discipline import compute_maintenance_capex_discipline
from app.valuation.returns_persistence import compute_returns_persistence
from app.valuation.valuation_confidence import compute_valuation_confidence
from app.valuation.valuation_integrity import compute_valuation_integrity
from app.valuation.value_type import compute_value_type
from app.valuation.impairment_classification import compute_impairment_classification
from app.valuation.normalization_credibility import (
    compute_normalization_credibility,
    write_normalization_credibility_for_run,
)
from app.valuation.capital_allocation_discipline import (
    compute_capital_allocation_discipline,
    write_capital_allocation_discipline_for_run,
)
from app.valuation.reinvestment_efficiency import compute_reinvestment_efficiency


UNKNOWN = "UNKNOWN"
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
_NORMALIZATION_CREDIBILITY_ORDER = {
    "HIGH_NORMALIZATION_CREDIBILITY": 0,
    "MODERATE_NORMALIZATION_CREDIBILITY": 1,
    "NORMALIZATION_CREDIBILITY_UNKNOWN": 2,
    "LOW_NORMALIZATION_CREDIBILITY": 3,
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
    return isinstance(value, (int, float))


def _status_rank(value_gate_status: str) -> int:
    token = str(value_gate_status or "").upper()
    if token == "PASS":
        return 0
    if token == "WATCH":
        return 1
    if token == "FAIL":
        return 2
    return 3


def _desc_key(value: Any) -> tuple[int, float]:
    if _is_num(value):
        return (0, -float(value))
    return (1, 0.0)


def _asc_key(value: Any) -> tuple[int, float]:
    if _is_num(value):
        return (0, float(value))
    return (1, 0.0)


def _sort_key_value_first(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
        _desc_key(row.get("mos_epv", UNKNOWN)),
        _desc_key(row.get("mos_netnet", UNKNOWN)),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("fcf_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("quality_score", UNKNOWN)),
        _asc_key(row.get("risk_penalty", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_quality(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
        _desc_key(row.get("mos_epv", UNKNOWN)),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("fcf_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("quality_score", UNKNOWN)),
        _asc_key(row.get("risk_penalty", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_intangible(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
        _desc_key(row.get("mos_epv", UNKNOWN)),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("fcf_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("quality_score", UNKNOWN)),
        _asc_key(row.get("risk_penalty", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_discipline(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
        _desc_key(row.get("mos_epv", UNKNOWN)),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _asc_key(row.get("risk_penalty", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _confidence_rank(value: Any) -> int:
    token = str(value or "CONFIDENCE_UNKNOWN").upper()
    if token == "HIGH_CONFIDENCE":
        return 0
    if token == "MEDIUM_CONFIDENCE":
        return 1
    if token == "LOW_CONFIDENCE":
        return 2
    return 3


def _value_type_rank(value: Any) -> int:
    token = str(value or "UNKNOWN_VALUE_TYPE").upper()
    if token == "ASSET_BACKED_VALUE":
        return 0
    if token == "EARNINGS_POWER_VALUE":
        return 1
    if token == "QUALITY_VALUE":
        return 2
    if token == "CYCLICAL_VALUE":
        return 3
    if token == "FRAGILE_VALUE":
        return 4
    return 5


def _integrity_rank(value: Any) -> int:
    token = str(value or "INTEGRITY_UNKNOWN").upper()
    if token == "INTEGRITY_OK":
        return 0
    if token == "INTEGRITY_WARNING":
        return 1
    if token == "INTEGRITY_SUSPECT":
        return 2
    return 3


def _readiness_rank(value: Any) -> int:
    token = str(value or "READINESS_UNKNOWN").upper()
    if token == "INVESTABLE_NOW":
        return 0
    if token == "RESEARCH_WORTHY_NOT_READY":
        return 1
    if token == "WATCH_ONLY":
        return 2
    if token == "NOT_INVESTABLE":
        return 3
    return 4


def _reinvestment_rank(value: Any) -> int:
    token = str(value or "REINVESTMENT_EFFICIENCY_UNKNOWN").upper()
    if token == "HIGH_REINVESTMENT_EFFICIENCY":
        return 0
    if token == "MODERATE_REINVESTMENT_EFFICIENCY":
        return 1
    if token == "REINVESTMENT_EFFICIENCY_UNKNOWN":
        return 2
    if token == "LOW_REINVESTMENT_EFFICIENCY":
        return 3
    return 4


def _accounting_quality_rank(value: Any) -> int:
    token = str(value or "ACCOUNTING_QUALITY_UNKNOWN").upper()
    return _ACCOUNTING_ORDER.get(token, len(_ACCOUNTING_ORDER))


def _capital_allocation_discipline_rank(value: Any) -> int:
    token = str(value or "CAPITAL_ALLOCATION_UNKNOWN").upper()
    if token == "OWNER_FRIENDLY_DISCIPLINED":
        return 0
    if token == "MIXED_CAPITAL_ALLOCATION":
        return 1
    if token == "CAPITAL_ALLOCATION_UNKNOWN":
        return 2
    if token == "OWNER_DILUTIVE_OR_DESTRUCTIVE":
        return 3
    return 4


def _balance_sheet_stress_rank(value: Any) -> int:
    token = str(value or "BALANCE_SHEET_STRESS_UNKNOWN").upper()
    return _BALANCE_SHEET_STRESS_ORDER.get(token, len(_BALANCE_SHEET_STRESS_ORDER))


def _refinancing_risk_rank(value: Any) -> int:
    token = str(value or "REFINANCING_RISK_UNKNOWN").upper()
    return _REFINANCING_RISK_ORDER.get(token, len(_REFINANCING_RISK_ORDER))


def _normalization_credibility_rank(value: Any) -> int:
    token = str(value or "NORMALIZATION_CREDIBILITY_UNKNOWN").upper()
    return _NORMALIZATION_CREDIBILITY_ORDER.get(token, len(_NORMALIZATION_CREDIBILITY_ORDER))


def _returns_persistence_rank(value: Any) -> int:
    token = str(value or "RETURNS_PERSISTENCE_UNKNOWN").upper()
    return _RETURNS_PERSISTENCE_ORDER.get(token, len(_RETURNS_PERSISTENCE_ORDER))


def _revenue_dependence_rank(value: Any) -> int:
    token = str(value or "REVENUE_DEPENDENCE_UNKNOWN").upper()
    return {
        "LOW_REVENUE_DEPENDENCE_RISK": 0,
        "MODERATE_REVENUE_DEPENDENCE_RISK": 1,
        "REVENUE_DEPENDENCE_UNKNOWN": 2,
        "HIGH_REVENUE_DEPENDENCE_RISK": 3,
    }.get(token, 4)


def _maintenance_capex_credibility_rank(value: Any) -> int:
    token = str(value or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN").upper()
    return _MAINTENANCE_CAPEX_CREDIBILITY_ORDER.get(token, len(_MAINTENANCE_CAPEX_CREDIBILITY_ORDER))


def _asset_intensity_rank(value: Any) -> int:
    token = str(value or "ASSET_INTENSITY_UNKNOWN").upper()
    return _ASSET_INTENSITY_ORDER.get(token, len(_ASSET_INTENSITY_ORDER))


def _sort_key_value_first_confidence(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _desc_key(row.get("mos_epv", UNKNOWN)),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("memory_priority_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_typed(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("memory_priority_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_trustworthy(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("memory_priority_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _cyclical_risk_rank(value: Any) -> int:
    token = str(value or "CYCLE_RISK_UNKNOWN").upper()
    if token == "MID_CYCLE_REASONABLE":
        return 0
    if token == "TROUGH_EARNINGS_RISK":
        return 1
    if token == "CYCLE_RISK_UNKNOWN":
        return 2
    if token == "PEAK_EARNINGS_RISK":
        return 3
    return 4


def _sort_key_value_first_cycle_aware(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _cyclical_risk_rank(row.get("cyclical_valuation_risk_class", "CYCLE_RISK_UNKNOWN")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("memory_priority_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_ready(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("memory_priority_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_reinvestment(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
        _desc_key(row.get("capital_allocation_score", UNKNOWN)),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_cash_earnings(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
        _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
        _desc_key(row.get("capital_allocation_score", UNKNOWN)),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _normalization_credibility_rank(
            row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
        ),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_balance_sheet(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _balance_sheet_stress_rank(
            row.get("balance_sheet_stress_class", "BALANCE_SHEET_STRESS_UNKNOWN")
        ),
        _refinancing_risk_rank(row.get("refinancing_risk_class", "REFINANCING_RISK_UNKNOWN")),
        _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
        _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _normalization_credibility_rank(
            row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
        ),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_durable_returns(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _returns_persistence_rank(row.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")),
        _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
        _desc_key(row.get("capital_allocation_score", UNKNOWN)),
        _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _normalization_credibility_rank(
            row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
        ),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_revenue_resilience(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _returns_persistence_rank(row.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")),
        _revenue_dependence_rank(row.get("revenue_dependence_risk_class", "REVENUE_DEPENDENCE_UNKNOWN")),
        _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
        _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _normalization_credibility_rank(
            row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
        ),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


def _sort_key_value_first_owner_earnings_hardness(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("value_gate_status") or "")),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _maintenance_capex_credibility_rank(
            row.get("maintenance_capex_credibility_class", "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN")
        ),
        _asset_intensity_rank(row.get("asset_intensity_class", "ASSET_INTENSITY_UNKNOWN")),
        _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
        _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
        _capital_allocation_discipline_rank(
            row.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN")
        ),
        _returns_persistence_rank(row.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _normalization_credibility_rank(
            row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
        ),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        str(row.get("ticker") or ""),
    )


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


def _reason_counts_from_entries(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in entries:
        code = str(row.get(key) or "").strip()
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))))


def _top_counts(counts: dict[str, int], *, top_n: int = 10) -> list[dict[str, Any]]:
    rows = [
        {"reason_code": str(name), "count": int(count)}
        for name, count in sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    ]
    return rows[: max(1, int(top_n))]


def _oe_claim_refs(payload: dict[str, Any], key: str) -> list[str]:
    claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
    claim = claims.get(key) if isinstance(claims, dict) else {}
    if not isinstance(claim, dict):
        return []
    return [str(ref) for ref in (claim.get("derived_from") or []) if str(ref).strip()]


def _owner_earnings_quality_payload(
    *,
    run_dir: Path,
    ticker: str,
    as_of_date: str,
    maintenance_capex_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fundamentals = _safe_json(run_dir / f"fundamentals_{ticker}.json")
    if not fundamentals:
        return {}
    return compute_owner_earnings_quality(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=fundamentals,
        maintenance_capex_payload=(
            maintenance_capex_payload if isinstance(maintenance_capex_payload, dict) else {}
        ),
    )


def _intangible_economics_payload(
    *,
    run_dir: Path,
    ticker: str,
    as_of_date: str,
    owner_quality_payload: dict[str, Any],
) -> dict[str, Any]:
    fundamentals = _safe_json(run_dir / f"fundamentals_{ticker}.json")
    if not fundamentals:
        return {}
    return compute_intangible_economics(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=fundamentals,
        owner_quality_payload=owner_quality_payload,
    )


def _intrinsic_discipline_payload(
    *,
    run_dir: Path,
    ticker: str,
    as_of_date: str,
    score_row: dict[str, Any],
    valuation_row: dict[str, Any],
    oe_quality_payload: dict[str, Any],
    intangible_payload: dict[str, Any],
    maintenance_capex_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fundamentals = _safe_json(run_dir / f"fundamentals_{ticker}.json")
    if not fundamentals:
        return {}
    metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
    metric_traces = score_row.get("metric_traces") if isinstance(score_row.get("metric_traces"), dict) else {}
    price_refs = (
        list(((metric_traces.get("current_price") or {}).get("derived_from") or []))
        if isinstance(metric_traces.get("current_price"), dict)
        else list(valuation_row.get("derived_from") or [])
    )
    intrinsic_base_refs = (
        list(((metric_traces.get("intrinsic_per_share_base") or {}).get("derived_from") or []))
        if isinstance(metric_traces.get("intrinsic_per_share_base"), dict)
        else []
    )
    intrinsic_conservative_refs = (
        list(((metric_traces.get("intrinsic_per_share_conservative") or {}).get("derived_from") or []))
        if isinstance(metric_traces.get("intrinsic_per_share_conservative"), dict)
        else []
    )
    gd_refs = list(score_row.get("derived_from") or [])
    return compute_intrinsic_discipline(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=fundamentals,
        owner_quality_payload=oe_quality_payload,
        intangible_payload=intangible_payload,
        price_value=metric_values.get("current_price", valuation_row.get("current_price", UNKNOWN)),
        price_refs=price_refs,
        epv_per_share=metric_values.get("epv_per_share", UNKNOWN),
        epv_refs=gd_refs,
        netnet_per_share=metric_values.get("netnet_per_share", UNKNOWN),
        netnet_refs=gd_refs,
        existing_intrinsic_base=metric_values.get("intrinsic_per_share_base", valuation_row.get("intrinsic_per_share_base", UNKNOWN)),
        existing_intrinsic_base_refs=intrinsic_base_refs + list(valuation_row.get("derived_from") or []),
        existing_intrinsic_conservative=metric_values.get(
            "intrinsic_per_share_conservative",
            valuation_row.get("intrinsic_per_share_conservative", UNKNOWN),
        ),
        existing_intrinsic_conservative_refs=intrinsic_conservative_refs + list(valuation_row.get("derived_from") or []),
        existing_intrinsic_ceiling=metric_values.get("intrinsic_per_share_base", valuation_row.get("intrinsic_per_share_base", UNKNOWN)),
        existing_intrinsic_ceiling_refs=intrinsic_base_refs + list(valuation_row.get("derived_from") or []),
        maintenance_capex_payload=(
            maintenance_capex_payload if isinstance(maintenance_capex_payload, dict) else {}
        ),
    )


def _valuation_confidence_payload(
    *,
    as_of_date: str,
    score_row: dict[str, Any],
    valuation_row: dict[str, Any],
    shares_row: dict[str, Any],
    fcf_row: dict[str, Any],
    facts_row: dict[str, Any],
    intrinsic_payload: dict[str, Any],
    accounting_quality_payload: dict[str, Any] | None = None,
    balance_sheet_stress_payload: dict[str, Any] | None = None,
    maintenance_capex_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
    return compute_valuation_confidence(
        ticker=str(score_row.get("ticker") or valuation_row.get("ticker") or facts_row.get("ticker") or ""),
        as_of_date=as_of_date,
        intrinsic_payload=intrinsic_payload,
        accounting_quality_payload=accounting_quality_payload if isinstance(accounting_quality_payload, dict) else {},
        balance_sheet_stress_payload=(
            balance_sheet_stress_payload if isinstance(balance_sheet_stress_payload, dict) else {}
        ),
        maintenance_capex_payload=(
            maintenance_capex_payload if isinstance(maintenance_capex_payload, dict) else {}
        ),
        price_status=str(valuation_row.get("price_status", metric_values.get("price_status", UNKNOWN))),
        shares_status=str(shares_row.get("shares_status", valuation_row.get("shares_status", metric_values.get("shares_status", UNKNOWN)))),
        fcf_status=str(fcf_row.get("fcf_status", valuation_row.get("fcf_status", metric_values.get("fcf_status", UNKNOWN)))),
        facts_status=str(facts_row.get("status", UNKNOWN)),
        valuation_status=str(valuation_row.get("valuation_status", metric_values.get("valuation_status", UNKNOWN))),
        epv_per_share=metric_values.get("epv_per_share", UNKNOWN),
        epv_refs=list(score_row.get("derived_from") or []),
        netnet_per_share=metric_values.get("netnet_per_share", UNKNOWN),
        netnet_refs=list(score_row.get("derived_from") or []),
        existing_intrinsic_base=metric_values.get("intrinsic_per_share_base", valuation_row.get("intrinsic_per_share_base", UNKNOWN)),
        existing_intrinsic_base_refs=list(valuation_row.get("derived_from") or []),
        existing_intrinsic_conservative=metric_values.get("intrinsic_per_share_conservative", valuation_row.get("intrinsic_per_share_conservative", UNKNOWN)),
        existing_intrinsic_conservative_refs=list(valuation_row.get("derived_from") or []),
    )


def _value_type_payload(
    *,
    ticker: str,
    as_of_date: str,
    score_row: dict[str, Any],
    intrinsic_payload: dict[str, Any],
    valuation_confidence_payload: dict[str, Any],
    oe_quality_payload: dict[str, Any],
    intangible_payload: dict[str, Any],
    accounting_quality_payload: dict[str, Any],
    balance_sheet_stress_payload: dict[str, Any],
    returns_persistence_payload: dict[str, Any],
    reinvestment_efficiency_payload: dict[str, Any],
    facts_row: dict[str, Any],
) -> dict[str, Any]:
    return compute_value_type(
        ticker=ticker,
        as_of_date=as_of_date,
        intrinsic_payload=intrinsic_payload,
        valuation_confidence_payload=valuation_confidence_payload,
        owner_quality_payload=oe_quality_payload,
        intangible_payload=intangible_payload,
        accounting_quality_payload=accounting_quality_payload,
        balance_sheet_stress_payload=balance_sheet_stress_payload,
        returns_persistence_payload=returns_persistence_payload,
        reinvestment_efficiency_payload=reinvestment_efficiency_payload,
        fail_due_to_missing_evidence=bool(score_row.get("fail_due_to_missing_evidence", False)),
        fail_due_to_economic_weakness=bool(score_row.get("fail_due_to_economic_weakness", False)),
        primary_fail_domain=str(
            score_row.get("primary_fail_domain")
            or facts_row.get("primary_fail_domain")
            or "NONE"
        ),
    )


def _valuation_integrity_payload(
    *,
    ticker: str,
    as_of_date: str,
    score_row: dict[str, Any],
    valuation_row: dict[str, Any],
    shares_row: dict[str, Any],
    fcf_row: dict[str, Any],
    facts_row: dict[str, Any],
    intrinsic_payload: dict[str, Any],
    valuation_confidence_payload: dict[str, Any],
    value_type_payload: dict[str, Any],
) -> dict[str, Any]:
    metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
    return compute_valuation_integrity(
        ticker=ticker,
        as_of_date=as_of_date,
        intrinsic_payload=intrinsic_payload,
        valuation_confidence_payload=valuation_confidence_payload,
        value_type_payload=value_type_payload,
        price_status=str(valuation_row.get("price_status", metric_values.get("price_status", UNKNOWN))),
        shares_status=str(
            shares_row.get("shares_status", valuation_row.get("shares_status", metric_values.get("shares_status", UNKNOWN)))
        ),
        fcf_status=str(fcf_row.get("fcf_status", valuation_row.get("fcf_status", metric_values.get("fcf_status", UNKNOWN)))),
        facts_status=str(facts_row.get("status", UNKNOWN)),
        valuation_status=str(valuation_row.get("valuation_status", metric_values.get("valuation_status", UNKNOWN))),
    )


def _investment_readiness_payload(
    *,
    ticker: str,
    as_of_date: str,
    score_row: dict[str, Any],
    gate_row: dict[str, Any],
    valuation_row: dict[str, Any],
    shares_row: dict[str, Any],
    fcf_row: dict[str, Any],
    facts_row: dict[str, Any],
    intrinsic_payload: dict[str, Any],
    evidence_sufficiency_payload: dict[str, Any],
    valuation_confidence_payload: dict[str, Any],
    valuation_integrity_payload: dict[str, Any],
    value_type_payload: dict[str, Any],
    oe_quality_payload: dict[str, Any],
    intangible_payload: dict[str, Any],
    accounting_quality_payload: dict[str, Any] | None = None,
    balance_sheet_stress_payload: dict[str, Any] | None = None,
    returns_persistence_payload: dict[str, Any] | None = None,
    impairment_classification_payload: dict[str, Any] | None = None,
    reinvestment_efficiency_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
    return compute_investment_readiness(
        ticker=ticker,
        as_of_date=as_of_date,
        intrinsic_payload=intrinsic_payload,
        evidence_sufficiency_payload=evidence_sufficiency_payload,
        valuation_confidence_payload=valuation_confidence_payload,
        valuation_integrity_payload=valuation_integrity_payload,
        value_type_payload=value_type_payload,
        owner_quality_payload=oe_quality_payload,
        intangible_payload=intangible_payload,
        accounting_quality_payload=accounting_quality_payload if isinstance(accounting_quality_payload, dict) else {},
        balance_sheet_stress_payload=(
            balance_sheet_stress_payload if isinstance(balance_sheet_stress_payload, dict) else {}
        ),
        returns_persistence_payload=(
            returns_persistence_payload if isinstance(returns_persistence_payload, dict) else {}
        ),
        reinvestment_efficiency_payload=(
            reinvestment_efficiency_payload if isinstance(reinvestment_efficiency_payload, dict) else {}
        ),
        price_status=str(valuation_row.get("price_status", metric_values.get("price_status", UNKNOWN))),
        shares_status=str(
            shares_row.get("shares_status", valuation_row.get("shares_status", metric_values.get("shares_status", UNKNOWN)))
        ),
        fcf_status=str(
            fcf_row.get("fcf_status", valuation_row.get("fcf_status", metric_values.get("fcf_status", UNKNOWN)))
        ),
        facts_status=str(facts_row.get("status", UNKNOWN)),
        valuation_status=str(valuation_row.get("valuation_status", metric_values.get("valuation_status", UNKNOWN))),
        facts_blocker_class=str(score_row.get("facts_blocker_class") or "FACTS_OK"),
        facts_blocker_retryable=bool(score_row.get("facts_blocker_retryable", False)),
        facts_blocker_terminal=bool(score_row.get("facts_blocker_terminal", False)),
        facts_blocker_partial_usable=bool(score_row.get("facts_blocker_partial_usable", False)),
        facts_missing_key_inputs=list(score_row.get("facts_missing_key_inputs") or []),
        facts_retry_recommended=bool(score_row.get("facts_retry_recommended", False)),
        fail_due_to_missing_evidence=bool(score_row.get("fail_due_to_missing_evidence", False)),
        fail_due_to_economic_weakness=bool(score_row.get("fail_due_to_economic_weakness", False)),
        primary_fail_domain=str(
            score_row.get("primary_fail_domain")
            or facts_row.get("primary_fail_domain")
            or "NONE"
        ),
        value_gate_status=str(gate_row.get("gate_status") or "UNKNOWN"),
        primary_blocker=str(gate_row.get("primary_blocker") or "NONE"),
        row_derived_from=list(score_row.get("derived_from") or []),
        impairment_classification_payload=impairment_classification_payload if isinstance(impairment_classification_payload, dict) else {},
    )


def _evidence_sufficiency_payload(
    *,
    ticker: str,
    as_of_date: str,
    score_row: dict[str, Any],
    valuation_row: dict[str, Any],
    shares_row: dict[str, Any],
    fcf_row: dict[str, Any],
    facts_row: dict[str, Any],
    intrinsic_payload: dict[str, Any],
    valuation_confidence_payload: dict[str, Any],
    valuation_integrity_payload: dict[str, Any],
) -> dict[str, Any]:
    metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
    return compute_evidence_sufficiency(
        ticker=ticker,
        as_of_date=as_of_date,
        intrinsic_payload=intrinsic_payload,
        valuation_confidence_payload=valuation_confidence_payload,
        valuation_integrity_payload=valuation_integrity_payload,
        price_status=str(valuation_row.get("price_status", metric_values.get("price_status", UNKNOWN))),
        shares_status=str(
            shares_row.get("shares_status", valuation_row.get("shares_status", metric_values.get("shares_status", UNKNOWN)))
        ),
        fcf_status=str(
            fcf_row.get("fcf_status", valuation_row.get("fcf_status", metric_values.get("fcf_status", UNKNOWN)))
        ),
        facts_status=str(facts_row.get("status", UNKNOWN)),
        valuation_status=str(valuation_row.get("valuation_status", metric_values.get("valuation_status", UNKNOWN))),
        facts_blocker_class=str(score_row.get("facts_blocker_class") or "FACTS_OK"),
        fail_due_to_missing_evidence=bool(score_row.get("fail_due_to_missing_evidence", False)),
        primary_fail_domain=str(
            score_row.get("primary_fail_domain")
            or facts_row.get("primary_fail_domain")
            or "NONE"
        ),
        row_derived_from=list(score_row.get("derived_from") or []),
    )


def _batch_dir(universe_run_id: str, batch_run_id: str) -> Path:
    cfg = get_config()
    return cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id


def load_batch_summary(universe_run_id: str, batch_run_id: str) -> dict[str, Any]:
    batch_dir = _batch_dir(universe_run_id, batch_run_id)
    state_path = batch_dir / "batch_state.json"
    summary_path = batch_dir / "batch_summary.json"
    state_payload = _safe_json(state_path)
    summary_payload = _safe_json(summary_path)
    if not state_payload:
        raise ValueError(f"Missing batch_state.json for universe_run_id={universe_run_id} batch_run_id={batch_run_id}")
    completed_runs = [row for row in (state_payload.get("completed_runs") or []) if isinstance(row, dict)]
    planned_runs = [row for row in (state_payload.get("planned_runs") or []) if isinstance(row, dict)]
    return {
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "batch_dir": str(batch_dir),
        "state_path": str(state_path),
        "summary_path": str(summary_path),
        "batch_state": state_payload,
        "batch_summary": summary_payload if summary_payload else {},
        "completed_runs": completed_runs,
        "planned_runs": planned_runs,
    }


def _collect_run_candidates(run_meta: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_id = str(run_meta.get("run_id") or "")
    sector = str(run_meta.get("sector") or "UNKNOWN_SECTOR")
    as_of_date = str(run_meta.get("as_of_date") or "")
    tickers_seed = [str(token).strip().upper() for token in (run_meta.get("tickers") or []) if str(token).strip()]

    artifact_paths = run_meta.get("artifacts_paths") if isinstance(run_meta.get("artifacts_paths"), dict) else {}
    run_dir = Path(str(artifact_paths.get("run_dir") or ""))
    if not str(run_dir).strip():
        cfg = get_config()
        run_dir = cfg.sectors_dir / run_id

    scoreboard_payload = _safe_json(run_dir / "peer_scoreboard.json")
    valuation_payload = _safe_json(run_dir / "valuation_coverage.json")
    value_gates_payload = _safe_json(run_dir / "value_gates.json")
    shares_payload = _safe_json(run_dir / "shares_coverage.json")
    fcf_payload = _safe_json(run_dir / "fcf_coverage.json")
    facts_payload = _safe_json(run_dir / "facts_coverage.json")
    rlm_state_payload = _safe_json(run_dir / "rlm_state.json")

    scoreboard_rows = [row for row in (scoreboard_payload.get("rows") or []) if isinstance(row, dict)]
    valuation_rows = [row for row in (valuation_payload.get("entries") or []) if isinstance(row, dict)]
    gate_rows = [row for row in (value_gates_payload.get("entries") or []) if isinstance(row, dict)]
    shares_rows = [row for row in (shares_payload.get("entries") or []) if isinstance(row, dict)]
    fcf_rows = [row for row in (fcf_payload.get("entries") or []) if isinstance(row, dict)]
    facts_rows = [row for row in (facts_payload.get("entries") or []) if isinstance(row, dict)]

    scoreboard_by_ticker = {str(row.get("ticker") or "").upper(): row for row in scoreboard_rows if str(row.get("ticker") or "").strip()}
    valuation_by_ticker = {str(row.get("ticker") or "").upper(): row for row in valuation_rows if str(row.get("ticker") or "").strip()}
    gates_by_ticker = {str(row.get("ticker") or "").upper(): row for row in gate_rows if str(row.get("ticker") or "").strip()}
    shares_by_ticker = {str(row.get("ticker") or "").upper(): row for row in shares_rows if str(row.get("ticker") or "").strip()}
    fcf_by_ticker = {str(row.get("ticker") or "").upper(): row for row in fcf_rows if str(row.get("ticker") or "").strip()}
    facts_by_ticker = {str(row.get("ticker") or "").upper(): row for row in facts_rows if str(row.get("ticker") or "").strip()}

    ticker_set = set(tickers_seed)
    ticker_set.update(scoreboard_by_ticker.keys())
    ticker_set.update(valuation_by_ticker.keys())
    ticker_set.update(gates_by_ticker.keys())
    ticker_set.update(shares_by_ticker.keys())
    ticker_set.update(fcf_by_ticker.keys())
    ticker_set.update(facts_by_ticker.keys())

    candidates: list[dict[str, Any]] = []
    for ticker in sorted(ticker_set):
        score_row = scoreboard_by_ticker.get(ticker, {})
        valuation_row = valuation_by_ticker.get(ticker, {})
        gate_row = gates_by_ticker.get(ticker, {})
        shares_row = shares_by_ticker.get(ticker, {})
        fcf_row = fcf_by_ticker.get(ticker, {})
        facts_row = facts_by_ticker.get(ticker, {})
        metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
        metric_traces = score_row.get("metric_traces") if isinstance(score_row.get("metric_traces"), dict) else {}
        fundamentals_payload = _safe_json(run_dir / f"fundamentals_{ticker}.json")
        maintenance_capex_payload = compute_maintenance_capex_discipline(
            ticker=ticker,
            as_of_date=as_of_date,
            fundamentals=fundamentals_payload if isinstance(fundamentals_payload, dict) else {},
            price_status=str(valuation_row.get("price_status", score_row.get("price_status", UNKNOWN))),
            facts_status=str(facts_row.get("status", UNKNOWN)),
            shares_status=str(score_row.get("shares_status", UNKNOWN)),
            fcf_status=str(score_row.get("fcf_status", valuation_row.get("fcf_status", UNKNOWN))),
            row_derived_from=list(score_row.get("derived_from") or []),
        )
        oe_quality_payload = _owner_earnings_quality_payload(
            run_dir=run_dir,
            ticker=ticker,
            as_of_date=as_of_date,
            maintenance_capex_payload=maintenance_capex_payload,
        )
        intangible_payload = _intangible_economics_payload(
            run_dir=run_dir,
            ticker=ticker,
            as_of_date=as_of_date,
            owner_quality_payload=oe_quality_payload,
        )
        intrinsic_payload = _intrinsic_discipline_payload(
            run_dir=run_dir,
            ticker=ticker,
            as_of_date=as_of_date,
            score_row=score_row,
            valuation_row=valuation_row,
            oe_quality_payload=oe_quality_payload,
            intangible_payload=intangible_payload,
            maintenance_capex_payload=maintenance_capex_payload,
        )
        valuation_confidence_payload = _valuation_confidence_payload(
            as_of_date=as_of_date,
            score_row=score_row,
            valuation_row=valuation_row,
            shares_row=shares_row,
            fcf_row=fcf_row,
            facts_row=facts_row,
            intrinsic_payload=intrinsic_payload,
            maintenance_capex_payload=maintenance_capex_payload,
        )
        capital_allocation_discipline_payload = compute_capital_allocation_discipline(
            ticker=ticker,
            as_of_date=as_of_date,
            owner_quality_payload=oe_quality_payload,
            intrinsic_payload=intrinsic_payload,
            evidence_sufficiency_payload={},
            valuation_confidence_payload=valuation_confidence_payload,
            price_status=str(valuation_row.get("price_status", score_row.get("price_status", UNKNOWN))),
            facts_status=str(facts_row.get("status", UNKNOWN)),
            shares_status=str(score_row.get("shares_status", UNKNOWN)),
        )
        reinvestment_efficiency_payload = compute_reinvestment_efficiency(
            ticker=ticker,
            as_of_date=as_of_date,
            fundamentals=fundamentals_payload if isinstance(fundamentals_payload, dict) else {},
            owner_quality_payload=oe_quality_payload,
            intangible_payload=intangible_payload,
            capital_allocation_discipline_payload=capital_allocation_discipline_payload,
            maintenance_capex_payload=maintenance_capex_payload,
            evidence_sufficiency_payload={},
            price_status=str(valuation_row.get("price_status", score_row.get("price_status", UNKNOWN))),
            facts_status=str(facts_row.get("status", UNKNOWN)),
            shares_status=str(score_row.get("shares_status", UNKNOWN)),
            fcf_status=str(score_row.get("fcf_status", valuation_row.get("fcf_status", UNKNOWN))),
            row_derived_from=list(score_row.get("derived_from") or []),
        )
        accounting_quality_payload = compute_accounting_quality(
            ticker=ticker,
            as_of_date=as_of_date,
            fundamentals=fundamentals_payload if isinstance(fundamentals_payload, dict) else {},
            owner_quality_payload=oe_quality_payload,
            capital_allocation_discipline_payload=capital_allocation_discipline_payload,
            reinvestment_efficiency_payload=reinvestment_efficiency_payload,
            price_status=str(valuation_row.get("price_status", score_row.get("price_status", UNKNOWN))),
            facts_status=str(facts_row.get("status", UNKNOWN)),
            shares_status=str(score_row.get("shares_status", UNKNOWN)),
            fcf_status=str(score_row.get("fcf_status", valuation_row.get("fcf_status", UNKNOWN))),
            row_derived_from=list(score_row.get("derived_from") or []),
        )
        balance_sheet_stress_payload = compute_balance_sheet_stress(
            ticker=ticker,
            as_of_date=as_of_date,
            fundamentals=fundamentals_payload if isinstance(fundamentals_payload, dict) else {},
            owner_quality_payload=oe_quality_payload,
            intangible_payload=intangible_payload,
            evidence_sufficiency_payload={},
            net_debt_proxy=metric_values.get("net_debt_proxy", UNKNOWN),
            total_debt=metric_values.get("total_debt", UNKNOWN),
            cash_equivalents=metric_values.get("cash_equivalents", UNKNOWN),
            net_debt_to_cfo=metric_values.get("net_debt_to_cfo", UNKNOWN),
            row_derived_from=list(score_row.get("derived_from") or []),
        )
        returns_persistence_payload = compute_returns_persistence(
            ticker=ticker,
            as_of_date=as_of_date,
            fundamentals=fundamentals_payload if isinstance(fundamentals_payload, dict) else {},
            owner_quality_payload=oe_quality_payload,
            intangible_payload=intangible_payload,
            reinvestment_efficiency_payload=reinvestment_efficiency_payload,
            capital_allocation_discipline_payload=capital_allocation_discipline_payload,
            roic_proxy=metric_values.get("roic_proxy", UNKNOWN),
            roe_proxy=metric_values.get("roe_proxy", UNKNOWN),
            roa_proxy=metric_values.get("roa_proxy", UNKNOWN),
            return_on_retained_earnings=metric_values.get("return_on_retained_earnings", UNKNOWN),
            revenue_cagr_proxy=metric_values.get("revenue_cagr_5y", UNKNOWN),
            invested_capital_cagr_proxy=metric_values.get("invested_capital_cagr_5y", UNKNOWN),
            price_status=str(valuation_row.get("price_status", score_row.get("price_status", UNKNOWN))),
            facts_status=str(facts_row.get("status", UNKNOWN)),
            shares_status=str(score_row.get("shares_status", UNKNOWN)),
            fcf_status=str(score_row.get("fcf_status", valuation_row.get("fcf_status", UNKNOWN))),
            row_derived_from=list(score_row.get("derived_from") or []),
        )
        valuation_confidence_payload = _valuation_confidence_payload(
            as_of_date=as_of_date,
            score_row=score_row,
            valuation_row=valuation_row,
            shares_row=shares_row,
            fcf_row=fcf_row,
            facts_row=facts_row,
            intrinsic_payload=intrinsic_payload,
            accounting_quality_payload=accounting_quality_payload,
            balance_sheet_stress_payload=balance_sheet_stress_payload,
            maintenance_capex_payload=maintenance_capex_payload,
        )
        value_type_payload = _value_type_payload(
            ticker=ticker,
            as_of_date=as_of_date,
            score_row=score_row,
            intrinsic_payload=intrinsic_payload,
            valuation_confidence_payload=valuation_confidence_payload,
            oe_quality_payload=oe_quality_payload,
            intangible_payload=intangible_payload,
            accounting_quality_payload=accounting_quality_payload,
            balance_sheet_stress_payload=balance_sheet_stress_payload,
            returns_persistence_payload=returns_persistence_payload,
            reinvestment_efficiency_payload=reinvestment_efficiency_payload,
            facts_row=facts_row,
        )
        valuation_integrity_payload = _valuation_integrity_payload(
            ticker=ticker,
            as_of_date=as_of_date,
            score_row=score_row,
            valuation_row=valuation_row,
            shares_row=shares_row,
            fcf_row=fcf_row,
            facts_row=facts_row,
            intrinsic_payload=intrinsic_payload,
            valuation_confidence_payload=valuation_confidence_payload,
            value_type_payload=value_type_payload,
        )
        evidence_sufficiency_payload = _evidence_sufficiency_payload(
            ticker=ticker,
            as_of_date=as_of_date,
            score_row=score_row,
            valuation_row=valuation_row,
            shares_row=shares_row,
            fcf_row=fcf_row,
            facts_row=facts_row,
            intrinsic_payload=intrinsic_payload,
            valuation_confidence_payload=valuation_confidence_payload,
            valuation_integrity_payload=valuation_integrity_payload,
        )
        impairment_classification_payload = compute_impairment_classification(
            ticker=ticker,
            as_of_date=as_of_date,
            intrinsic_payload=intrinsic_payload,
            evidence_sufficiency_payload=evidence_sufficiency_payload,
            valuation_confidence_payload=valuation_confidence_payload,
            valuation_integrity_payload=valuation_integrity_payload,
            owner_quality_payload=oe_quality_payload,
            intangible_payload=intangible_payload,
            balance_sheet_stress_payload=balance_sheet_stress_payload,
            cyclical_normalization_payload=score_row.get("cyclical_normalization_detail")
            if isinstance(score_row.get("cyclical_normalization_detail"), dict)
            else score_row,
            price_status=str(valuation_row.get("price_status", score_row.get("price_status", UNKNOWN))),
            shares_status=str(score_row.get("shares_status", UNKNOWN)),
            fcf_status=str(score_row.get("fcf_status", UNKNOWN)),
            facts_status=str(facts_row.get("status", UNKNOWN)),
            fail_due_to_missing_evidence=bool(score_row.get("fail_due_to_missing_evidence", False)),
            fail_due_to_economic_weakness=bool(score_row.get("fail_due_to_economic_weakness", False)),
            primary_fail_domain=str(score_row.get("primary_fail_domain") or facts_row.get("primary_fail_domain") or UNKNOWN),
            row_derived_from=list(score_row.get("derived_from") or []),
        )
        normalization_credibility_payload = compute_normalization_credibility(
            ticker=ticker,
            as_of_date=as_of_date,
            cyclical_normalization_payload=score_row.get("cyclical_normalization_detail")
            if isinstance(score_row.get("cyclical_normalization_detail"), dict)
            else score_row,
            impairment_classification_payload=impairment_classification_payload,
            evidence_sufficiency_payload=evidence_sufficiency_payload,
            valuation_confidence_payload=valuation_confidence_payload,
            valuation_integrity_payload=valuation_integrity_payload,
            intrinsic_payload=intrinsic_payload,
            owner_quality_payload=oe_quality_payload,
            intangible_payload=intangible_payload,
            value_type_payload=value_type_payload,
            revenue_dependence_payload=score_row.get("revenue_dependence_detail")
            if isinstance(score_row.get("revenue_dependence_detail"), dict)
            else score_row,
            price_status=str(valuation_row.get("price_status", score_row.get("price_status", UNKNOWN))),
            facts_status=str(facts_row.get("status", UNKNOWN)),
            shares_status=str(score_row.get("shares_status", UNKNOWN)),
        )
        investment_readiness_payload = _investment_readiness_payload(
            ticker=ticker,
            as_of_date=as_of_date,
            score_row=score_row,
            gate_row=gate_row,
            valuation_row=valuation_row,
            shares_row=shares_row,
            fcf_row=fcf_row,
            facts_row=facts_row,
            intrinsic_payload=intrinsic_payload,
            evidence_sufficiency_payload=evidence_sufficiency_payload,
            valuation_confidence_payload=valuation_confidence_payload,
            valuation_integrity_payload=valuation_integrity_payload,
            value_type_payload=value_type_payload,
            oe_quality_payload=oe_quality_payload,
            intangible_payload=intangible_payload,
            accounting_quality_payload=accounting_quality_payload,
            balance_sheet_stress_payload=balance_sheet_stress_payload,
            returns_persistence_payload=returns_persistence_payload,
            impairment_classification_payload=impairment_classification_payload,
            reinvestment_efficiency_payload=reinvestment_efficiency_payload,
        )

        value_gate_status = str(gate_row.get("gate_status") or "UNKNOWN").upper()
        value_gate_reasons = [str(reason) for reason in (gate_row.get("gate_reasons") or []) if str(reason).strip()]
        primary_blocker = str(gate_row.get("primary_blocker") or "NONE")

        implied_return = metric_values.get("implied_return_base", valuation_row.get("implied_return_base", UNKNOWN))
        intrinsic_base = metric_values.get("intrinsic_per_share_base", valuation_row.get("intrinsic_per_share_base", UNKNOWN))
        mos_epv = metric_values.get("mos_epv", UNKNOWN)
        mos_netnet = metric_values.get("mos_netnet", UNKNOWN)
        owner_yield_ev = metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN)
        fcf_yield_ev = metric_values.get("fcf_yield_ev_3y", UNKNOWN)
        quality_score = metric_values.get("quality_score", UNKNOWN)
        risk_penalty = metric_values.get("risk_penalty", UNKNOWN)
        oe_quality_total = oe_quality_payload.get("oe_quality_total", metric_values.get("oe_quality_total", UNKNOWN))
        owner_earnings_stability_score = oe_quality_payload.get(
            "owner_earnings_stability_score",
            metric_values.get("owner_earnings_stability_score", UNKNOWN),
        )
        capital_allocation_score = oe_quality_payload.get(
            "capital_allocation_score",
            metric_values.get("capital_allocation_score", UNKNOWN),
        )
        cash_conversion_score = oe_quality_payload.get(
            "cash_conversion_score",
            metric_values.get("cash_conversion_score", UNKNOWN),
        )
        oe_quality_reason_codes = [
            str(code)
            for code in (oe_quality_payload.get("oe_quality_reason_codes") or [])
            if str(code).strip()
        ]
        gross_margin_durability_score = intangible_payload.get(
            "gross_margin_durability_score",
            metric_values.get("gross_margin_durability_score", UNKNOWN),
        )
        balance_sheet_optionality_score = intangible_payload.get(
            "balance_sheet_optionality_score",
            metric_values.get("balance_sheet_optionality_score", UNKNOWN),
        )
        cycle_resilience_score = intangible_payload.get(
            "cycle_resilience_score",
            metric_values.get("cycle_resilience_score", UNKNOWN),
        )
        rnd_productivity_score = intangible_payload.get(
            "rnd_productivity_score",
            metric_values.get("rnd_productivity_score", UNKNOWN),
        )
        sga_leverage_score = intangible_payload.get(
            "sga_leverage_score",
            metric_values.get("sga_leverage_score", UNKNOWN),
        )
        owner_value_capture_score = intangible_payload.get(
            "owner_value_capture_score",
            metric_values.get("owner_value_capture_score", UNKNOWN),
        )
        intangible_economics_total = intangible_payload.get(
            "intangible_economics_total",
            metric_values.get("intangible_economics_total", UNKNOWN),
        )
        rnd_productivity_reason_codes = [
            str(code)
            for code in (intangible_payload.get("rnd_productivity_reason_codes") or [])
            if str(code).strip()
        ]
        sga_leverage_reason_codes = [
            str(code)
            for code in (intangible_payload.get("sga_leverage_reason_codes") or [])
            if str(code).strip()
        ]
        owner_value_capture_reason_codes = [
            str(code)
            for code in (intangible_payload.get("owner_value_capture_reason_codes") or [])
            if str(code).strip()
        ]
        intangible_economics_reason_codes = [
            str(code)
            for code in (intangible_payload.get("intangible_economics_reason_codes") or [])
            if str(code).strip()
        ]
        reinvestment_efficiency_class = str(
            reinvestment_efficiency_payload.get("reinvestment_efficiency_class")
            or "REINVESTMENT_EFFICIENCY_UNKNOWN"
        )
        reinvestment_efficiency_reason_codes = [
            str(code)
            for code in (reinvestment_efficiency_payload.get("reinvestment_efficiency_reason_codes") or [])
            if str(code).strip()
        ]
        reinvestment_support_signals = [
            str(code)
            for code in (reinvestment_efficiency_payload.get("reinvestment_support_signals") or [])
            if str(code).strip()
        ]
        reinvestment_headwind_signals = [
            str(code)
            for code in (reinvestment_efficiency_payload.get("reinvestment_headwind_signals") or [])
            if str(code).strip()
        ]
        primary_reinvestment_caution = str(
            reinvestment_efficiency_payload.get("primary_reinvestment_caution")
            or "REINVESTMENT_UNCLEAR"
        )
        reinvestment_efficiency_summary = str(
            reinvestment_efficiency_payload.get("reinvestment_efficiency_summary") or ""
        )
        accounting_quality_class = str(
            accounting_quality_payload.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
        )
        accounting_quality_reason_codes = [
            str(code)
            for code in (accounting_quality_payload.get("accounting_quality_reason_codes") or [])
            if str(code).strip()
        ]
        cash_earnings_support_signals = [
            str(code)
            for code in (accounting_quality_payload.get("cash_earnings_support_signals") or [])
            if str(code).strip()
        ]
        cash_earnings_headwind_signals = [
            str(code)
            for code in (accounting_quality_payload.get("cash_earnings_headwind_signals") or [])
            if str(code).strip()
        ]
        primary_accounting_caution = str(
            accounting_quality_payload.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
        )
        cash_earnings_discipline_summary = str(
            accounting_quality_payload.get("cash_earnings_discipline_summary") or ""
        )
        balance_sheet_stress_class = str(
            balance_sheet_stress_payload.get("balance_sheet_stress_class")
            or "BALANCE_SHEET_STRESS_UNKNOWN"
        )
        balance_sheet_stress_reason_codes = [
            str(code)
            for code in (balance_sheet_stress_payload.get("balance_sheet_stress_reason_codes") or [])
            if str(code).strip()
        ]
        refinancing_risk_class = str(
            balance_sheet_stress_payload.get("refinancing_risk_class")
            or "REFINANCING_RISK_UNKNOWN"
        )
        refinancing_risk_reason_codes = [
            str(code)
            for code in (balance_sheet_stress_payload.get("refinancing_risk_reason_codes") or [])
            if str(code).strip()
        ]
        balance_sheet_support_signals = [
            str(code)
            for code in (balance_sheet_stress_payload.get("balance_sheet_support_signals") or [])
            if str(code).strip()
        ]
        balance_sheet_headwind_signals = [
            str(code)
            for code in (balance_sheet_stress_payload.get("balance_sheet_headwind_signals") or [])
            if str(code).strip()
        ]
        primary_balance_sheet_caution = str(
            balance_sheet_stress_payload.get("primary_balance_sheet_caution")
            or "BALANCE_SHEET_UNCLEAR"
        )
        balance_sheet_discipline_summary = str(
            balance_sheet_stress_payload.get("balance_sheet_discipline_summary") or ""
        )
        returns_persistence_class = str(
            returns_persistence_payload.get("returns_persistence_class")
            or "RETURNS_PERSISTENCE_UNKNOWN"
        )
        returns_persistence_reason_codes = [
            str(code)
            for code in (returns_persistence_payload.get("returns_persistence_reason_codes") or [])
            if str(code).strip()
        ]
        returns_support_signals = [
            str(code)
            for code in (returns_persistence_payload.get("returns_support_signals") or [])
            if str(code).strip()
        ]
        returns_headwind_signals = [
            str(code)
            for code in (returns_persistence_payload.get("returns_headwind_signals") or [])
            if str(code).strip()
        ]
        primary_returns_caution = str(
            returns_persistence_payload.get("primary_returns_caution")
            or "RETURNS_DURABILITY_UNCLEAR"
        )
        economic_durability_summary = str(
            returns_persistence_payload.get("economic_durability_summary") or ""
        )
        revenue_dependence_payload = score_row.get("revenue_dependence_detail")
        if not isinstance(revenue_dependence_payload, dict):
            revenue_dependence_payload = score_row
        revenue_dependence_risk_class = str(
            revenue_dependence_payload.get("revenue_dependence_risk_class")
            or "REVENUE_DEPENDENCE_UNKNOWN"
        )
        revenue_dependence_risk_reason_codes = [
            str(code)
            for code in (revenue_dependence_payload.get("revenue_dependence_risk_reason_codes") or [])
            if str(code).strip()
        ]
        revenue_dependence_support_signals = [
            str(code)
            for code in (revenue_dependence_payload.get("revenue_dependence_support_signals") or [])
            if str(code).strip()
        ]
        revenue_dependence_headwind_signals = [
            str(code)
            for code in (revenue_dependence_payload.get("revenue_dependence_headwind_signals") or [])
            if str(code).strip()
        ]
        primary_revenue_dependence_caution = str(
            revenue_dependence_payload.get("primary_revenue_dependence_caution")
            or "REVENUE_BASE_UNCLEAR"
        )
        revenue_fragility_summary = str(
            revenue_dependence_payload.get("revenue_fragility_summary") or ""
        )
        asset_intensity_class = str(
            maintenance_capex_payload.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
        )
        asset_intensity_reason_codes = [
            str(code)
            for code in (maintenance_capex_payload.get("asset_intensity_reason_codes") or [])
            if str(code).strip()
        ]
        maintenance_capex_credibility_class = str(
            maintenance_capex_payload.get("maintenance_capex_credibility_class")
            or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
        )
        maintenance_capex_credibility_reason_codes = [
            str(code)
            for code in (maintenance_capex_payload.get("maintenance_capex_credibility_reason_codes") or [])
            if str(code).strip()
        ]
        maintenance_capex_support_signals = [
            str(code)
            for code in (maintenance_capex_payload.get("maintenance_capex_support_signals") or [])
            if str(code).strip()
        ]
        maintenance_capex_headwind_signals = [
            str(code)
            for code in (maintenance_capex_payload.get("maintenance_capex_headwind_signals") or [])
            if str(code).strip()
        ]
        primary_maintenance_capex_caution = str(
            maintenance_capex_payload.get("primary_maintenance_capex_caution")
            or "OWNER_EARNINGS_UNCLEAR"
        )
        maintenance_capex_discipline_summary = str(
            maintenance_capex_payload.get("maintenance_capex_discipline_summary") or ""
        )
        normalized_earnings_power_value = intrinsic_payload.get(
            "normalized_earnings_power_value",
            metric_values.get("normalized_earnings_power_value", UNKNOWN),
        )
        normalized_earnings_power_method_used = str(
            intrinsic_payload.get("normalized_earnings_power_method_used")
            or metric_values.get("normalized_earnings_power_method_used")
            or UNKNOWN
        )
        normalized_earnings_power_status = str(
            intrinsic_payload.get("normalized_earnings_power_status")
            or metric_values.get("normalized_earnings_power_status")
            or UNKNOWN
        )
        intrinsic_floor = intrinsic_payload.get("intrinsic_floor", metric_values.get("intrinsic_floor", UNKNOWN))
        intrinsic_base_discipline = intrinsic_payload.get("intrinsic_base", metric_values.get("intrinsic_base", UNKNOWN))
        intrinsic_ceiling = intrinsic_payload.get("intrinsic_ceiling", metric_values.get("intrinsic_ceiling", UNKNOWN))
        mos_to_floor = intrinsic_payload.get("mos_to_floor", metric_values.get("mos_to_floor", UNKNOWN))
        mos_to_base = intrinsic_payload.get("mos_to_base", metric_values.get("mos_to_base", UNKNOWN))
        mos_classification = str(intrinsic_payload.get("mos_classification") or UNKNOWN)
        downside_support_type = str(intrinsic_payload.get("downside_support_type") or UNKNOWN)
        downside_support_status = str(intrinsic_payload.get("downside_support_status") or UNKNOWN)
        normalized_earnings_power_reason_codes = [
            str(code)
            for code in (intrinsic_payload.get("normalized_earnings_power_reason_codes") or [])
            if str(code).strip()
        ]
        valuation_range_reason_codes = [
            str(code)
            for code in (intrinsic_payload.get("valuation_range_reason_codes") or [])
            if str(code).strip()
        ]
        downside_support_reason_codes = [
            str(code)
            for code in (intrinsic_payload.get("downside_support_reason_codes") or [])
            if str(code).strip()
        ]
        valuation_support_count = valuation_confidence_payload.get("valuation_support_count", 0)
        valuation_support_types_present = [
            str(value)
            for value in (valuation_confidence_payload.get("valuation_support_types_present") or [])
            if str(value).strip()
        ]
        valuation_support_count_reason_codes = [
            str(code)
            for code in (valuation_confidence_payload.get("valuation_support_count_reason_codes") or [])
            if str(code).strip()
        ]
        valuation_convergence_status = str(
            valuation_confidence_payload.get("valuation_convergence_status") or UNKNOWN
        )
        valuation_convergence_band_pct = valuation_confidence_payload.get(
            "valuation_convergence_band_pct",
            UNKNOWN,
        )
        valuation_convergence_reason_codes = [
            str(code)
            for code in (valuation_confidence_payload.get("valuation_convergence_reason_codes") or [])
            if str(code).strip()
        ]
        valuation_fragility_status = str(
            valuation_confidence_payload.get("valuation_fragility_status") or UNKNOWN
        )
        valuation_fragility_reason_codes = [
            str(code)
            for code in (valuation_confidence_payload.get("valuation_fragility_reason_codes") or [])
            if str(code).strip()
        ]
        valuation_confidence_class = str(
            valuation_confidence_payload.get("valuation_confidence_class") or UNKNOWN
        )
        valuation_confidence_reason_codes = [
            str(code)
            for code in (valuation_confidence_payload.get("valuation_confidence_reason_codes") or [])
            if str(code).strip()
        ]
        valuation_integrity_class = str(
            valuation_integrity_payload.get("valuation_integrity_class") or UNKNOWN
        )
        valuation_integrity_reason_codes = [
            str(code)
            for code in (valuation_integrity_payload.get("valuation_integrity_reason_codes") or [])
            if str(code).strip()
        ]
        evidence_sufficiency_class = str(
            evidence_sufficiency_payload.get("evidence_sufficiency_class") or UNKNOWN
        )
        evidence_sufficiency_reason_codes = [
            str(code)
            for code in (evidence_sufficiency_payload.get("evidence_sufficiency_reason_codes") or [])
            if str(code).strip()
        ]
        mos_assessment_status = str(
            evidence_sufficiency_payload.get("mos_assessment_status") or UNKNOWN
        )
        mos_guardrail_reason_codes = [
            str(code)
            for code in (evidence_sufficiency_payload.get("mos_guardrail_reason_codes") or [])
            if str(code).strip()
        ]
        valuation_consistency_status = str(
            valuation_integrity_payload.get("valuation_consistency_status") or UNKNOWN
        )
        valuation_consistency_reason_codes = [
            str(code)
            for code in (valuation_integrity_payload.get("valuation_consistency_reason_codes") or [])
            if str(code).strip()
        ]
        valuation_uniformity_group_id = (
            str(valuation_integrity_payload.get("valuation_uniformity_group_id"))
            if str(valuation_integrity_payload.get("valuation_uniformity_group_id") or "").strip()
            else None
        )
        valuation_uniformity_reason_codes = [
            str(code)
            for code in (valuation_integrity_payload.get("valuation_uniformity_reason_codes") or [])
            if str(code).strip()
        ]
        value_type_primary = str(value_type_payload.get("value_type_primary") or UNKNOWN)
        value_type_secondary = (
            str(value_type_payload.get("value_type_secondary"))
            if str(value_type_payload.get("value_type_secondary") or "").strip()
            else None
        )
        value_type_reason_codes = [
            str(code)
            for code in (value_type_payload.get("value_type_reason_codes") or [])
            if str(code).strip()
        ]
        value_type_support_summary = str(value_type_payload.get("value_type_support_summary") or "")
        investment_readiness_class = str(
            investment_readiness_payload.get("investment_readiness_class") or UNKNOWN
        )
        investment_readiness_reason_codes = [
            str(code)
            for code in (investment_readiness_payload.get("investment_readiness_reason_codes") or [])
            if str(code).strip()
        ]
        blocker_stack_primary = (
            str(investment_readiness_payload.get("blocker_stack_primary"))
            if str(investment_readiness_payload.get("blocker_stack_primary") or "").strip()
            else UNKNOWN
        )
        blocker_stack_secondary = (
            str(investment_readiness_payload.get("blocker_stack_secondary"))
            if str(investment_readiness_payload.get("blocker_stack_secondary") or "").strip()
            else None
        )
        blocker_stack_all = [
            str(code)
            for code in (investment_readiness_payload.get("blocker_stack_all") or [])
            if str(code).strip()
        ]
        blocker_stack_retryable = bool(
            investment_readiness_payload.get("blocker_stack_retryable", False)
        )
        blocker_stack_structural = bool(
            investment_readiness_payload.get("blocker_stack_structural", False)
        )
        readiness_support_present = [
            str(code)
            for code in (investment_readiness_payload.get("readiness_support_present") or [])
            if str(code).strip()
        ]
        readiness_support_missing = [
            str(code)
            for code in (investment_readiness_payload.get("readiness_support_missing") or [])
            if str(code).strip()
        ]
        readiness_support_headwinds = [
            str(code)
            for code in (investment_readiness_payload.get("readiness_support_headwinds") or [])
            if str(code).strip()
        ]
        primary_next_step = str(investment_readiness_payload.get("primary_next_step") or UNKNOWN)
        primary_next_step_reason = str(
            investment_readiness_payload.get("primary_next_step_reason") or UNKNOWN
        )

        implied_trace = (
            (metric_traces.get("implied_return_base") or {}).get("derived_from")
            if isinstance(metric_traces.get("implied_return_base"), dict)
            else []
        )
        intrinsic_trace = (
            (metric_traces.get("intrinsic_per_share_base") or {}).get("derived_from")
            if isinstance(metric_traces.get("intrinsic_per_share_base"), dict)
            else []
        )
        mos_epv_trace = (
            (metric_traces.get("mos_epv") or {}).get("derived_from")
            if isinstance(metric_traces.get("mos_epv"), dict)
            else []
        )
        mos_netnet_trace = (
            (metric_traces.get("mos_netnet") or {}).get("derived_from")
            if isinstance(metric_traces.get("mos_netnet"), dict)
            else []
        )
        owner_yield_trace = (
            (metric_traces.get("owner_earnings_yield_ev_3y") or {}).get("derived_from")
            if isinstance(metric_traces.get("owner_earnings_yield_ev_3y"), dict)
            else []
        )
        fcf_yield_trace = (
            (metric_traces.get("fcf_yield_ev_3y") or {}).get("derived_from")
            if isinstance(metric_traces.get("fcf_yield_ev_3y"), dict)
            else []
        )
        oe_quality_trace = _oe_claim_refs(oe_quality_payload, "oe_quality_total")
        owner_stability_trace = _oe_claim_refs(oe_quality_payload, "owner_earnings_stability_score")
        capital_allocation_trace = _oe_claim_refs(oe_quality_payload, "capital_allocation_score")
        cash_conversion_trace = _oe_claim_refs(oe_quality_payload, "cash_conversion_score")
        gross_margin_trace = _oe_claim_refs(intangible_payload, "gross_margin_durability_score")
        balance_sheet_optionality_trace = _oe_claim_refs(intangible_payload, "balance_sheet_optionality_score")
        cycle_resilience_trace = _oe_claim_refs(intangible_payload, "cycle_resilience_score")
        rnd_productivity_trace = _oe_claim_refs(intangible_payload, "rnd_productivity_score")
        sga_leverage_trace = _oe_claim_refs(intangible_payload, "sga_leverage_score")
        owner_value_capture_trace = _oe_claim_refs(intangible_payload, "owner_value_capture_score")
        intangible_total_trace = _oe_claim_refs(intangible_payload, "intangible_economics_total")
        reinvestment_class_trace = _oe_claim_refs(
            reinvestment_efficiency_payload,
            "reinvestment_efficiency_class",
        )
        accounting_quality_trace = _oe_claim_refs(
            accounting_quality_payload,
            "accounting_quality_class",
        )
        balance_sheet_stress_trace = _oe_claim_refs(
            balance_sheet_stress_payload,
            "balance_sheet_stress_class",
        )
        refinancing_risk_trace = _oe_claim_refs(
            balance_sheet_stress_payload,
            "refinancing_risk_class",
        )
        returns_persistence_trace = _oe_claim_refs(
            returns_persistence_payload,
            "returns_persistence_class",
        )
        normalized_power_trace = _oe_claim_refs(intrinsic_payload, "normalized_earnings_power_value")
        intrinsic_floor_trace = _oe_claim_refs(intrinsic_payload, "intrinsic_floor")
        intrinsic_base_trace = _oe_claim_refs(intrinsic_payload, "intrinsic_base")
        intrinsic_ceiling_trace = _oe_claim_refs(intrinsic_payload, "intrinsic_ceiling")
        mos_to_floor_trace = _oe_claim_refs(intrinsic_payload, "mos_to_floor")
        mos_to_base_trace = _oe_claim_refs(intrinsic_payload, "mos_to_base")
        valuation_support_trace = _oe_claim_refs(valuation_confidence_payload, "valuation_support_count")
        valuation_convergence_trace = _oe_claim_refs(valuation_confidence_payload, "valuation_convergence_band_pct")
        valuation_fragility_trace = _oe_claim_refs(valuation_confidence_payload, "valuation_fragility_status")
        valuation_confidence_trace = _oe_claim_refs(valuation_confidence_payload, "valuation_confidence_class")
        valuation_integrity_trace = _oe_claim_refs(valuation_integrity_payload, "valuation_integrity_class")
        evidence_sufficiency_trace = _oe_claim_refs(
            evidence_sufficiency_payload,
            "evidence_sufficiency_class",
        )
        value_type_trace = _oe_claim_refs(value_type_payload, "value_type_primary")
        investment_readiness_trace = _oe_claim_refs(
            investment_readiness_payload,
            "investment_readiness_class",
        )

        price_status = str(
            valuation_row.get("price_status", metric_values.get("price_status", UNKNOWN))
        ).upper()
        valuation_status = str(
            valuation_row.get("valuation_status", metric_values.get("valuation_status", UNKNOWN))
        ).upper()
        shares_status = str(
            shares_row.get("shares_status", valuation_row.get("shares_status", metric_values.get("shares_status", UNKNOWN)))
        ).upper()
        fcf_status = str(
            fcf_row.get("fcf_status", valuation_row.get("fcf_status", metric_values.get("fcf_status", UNKNOWN)))
        ).upper()
        facts_status = str(facts_row.get("status", UNKNOWN)).upper()

        price_reason_code = str(valuation_row.get("price_reason_code") or UNKNOWN)
        valuation_reason_code = str(valuation_row.get("valuation_reason_code") or metric_values.get("valuation_reason_code") or UNKNOWN)
        shares_reason_code = str(shares_row.get("shares_reason_code") or valuation_row.get("shares_reason_code") or metric_values.get("shares_reason_code") or UNKNOWN)
        fcf_reason_code = str(fcf_row.get("fcf_reason_code") or valuation_row.get("fcf_reason_code") or metric_values.get("fcf_reason_code") or UNKNOWN)
        facts_reason_code = str(facts_row.get("fetch_reason_code") or UNKNOWN)
        if facts_reason_code == UNKNOWN and facts_status != "OK":
            facts_reason_code = str(facts_row.get("status") or UNKNOWN)

        yield_denominator_used = str(
            metric_values.get("yield_denominator_used")
            or metric_values.get("denominator_used")
            or UNKNOWN
        )

        notes = (
            f"{value_gate_status} | implied_return_base="
            f"{implied_return if _is_num(implied_return) else UNKNOWN} | blocker={primary_blocker}"
        )

        candidates.append(
            {
                "ticker": ticker,
                "run_id": run_id,
                "sector": sector,
                "as_of_date": as_of_date,
                "source_depth_runs": [{"run_id": run_id, "sector": sector, "as_of_date": as_of_date}],
                "value_gate_status": value_gate_status,
                "value_gate_reasons": value_gate_reasons,
                "primary_blocker": primary_blocker,
                "primary_blocker_reason_code": primary_blocker,
                "implied_return_base": implied_return if _is_num(implied_return) else UNKNOWN,
                "implied_return_base_derived_from": _dedupe_refs(list(implied_trace) + list(valuation_row.get("derived_from") or [])),
                "intrinsic_per_share_base": intrinsic_base if _is_num(intrinsic_base) else UNKNOWN,
                "intrinsic_per_share_base_derived_from": _dedupe_refs(list(intrinsic_trace) + list(valuation_row.get("derived_from") or [])),
                "mos_epv": mos_epv if _is_num(mos_epv) else UNKNOWN,
                "mos_epv_derived_from": _dedupe_refs(list(mos_epv_trace)),
                "mos_netnet": mos_netnet if _is_num(mos_netnet) else UNKNOWN,
                "mos_netnet_derived_from": _dedupe_refs(list(mos_netnet_trace)),
                "owner_earnings_yield_ev_3y": owner_yield_ev if _is_num(owner_yield_ev) else UNKNOWN,
                "owner_earnings_yield_ev_3y_derived_from": _dedupe_refs(list(owner_yield_trace)),
                "fcf_yield_ev_3y": fcf_yield_ev if _is_num(fcf_yield_ev) else UNKNOWN,
                "fcf_yield_ev_3y_derived_from": _dedupe_refs(list(fcf_yield_trace)),
                "yield_denominator_used": yield_denominator_used,
                "owner_earnings_stability_score": owner_earnings_stability_score if _is_num(owner_earnings_stability_score) else UNKNOWN,
                "owner_earnings_stability_score_derived_from": _dedupe_refs(list(owner_stability_trace)),
                "capital_allocation_score": capital_allocation_score if _is_num(capital_allocation_score) else UNKNOWN,
                "capital_allocation_score_derived_from": _dedupe_refs(list(capital_allocation_trace)),
                "cash_conversion_score": cash_conversion_score if _is_num(cash_conversion_score) else UNKNOWN,
                "cash_conversion_score_derived_from": _dedupe_refs(list(cash_conversion_trace)),
                "oe_quality_total": oe_quality_total if _is_num(oe_quality_total) else UNKNOWN,
                "oe_quality_total_derived_from": _dedupe_refs(list(oe_quality_trace)),
                "oe_quality_reason_codes": oe_quality_reason_codes,
                "gross_margin_durability_score": gross_margin_durability_score if _is_num(gross_margin_durability_score) else UNKNOWN,
                "gross_margin_durability_score_derived_from": _dedupe_refs(list(gross_margin_trace)),
                "balance_sheet_optionality_score": balance_sheet_optionality_score if _is_num(balance_sheet_optionality_score) else UNKNOWN,
                "balance_sheet_optionality_score_derived_from": _dedupe_refs(list(balance_sheet_optionality_trace)),
                "cycle_resilience_score": cycle_resilience_score if _is_num(cycle_resilience_score) else UNKNOWN,
                "cycle_resilience_score_derived_from": _dedupe_refs(list(cycle_resilience_trace)),
                "rnd_productivity_score": rnd_productivity_score if _is_num(rnd_productivity_score) else UNKNOWN,
                "rnd_productivity_score_derived_from": _dedupe_refs(list(rnd_productivity_trace)),
                "sga_leverage_score": sga_leverage_score if _is_num(sga_leverage_score) else UNKNOWN,
                "sga_leverage_score_derived_from": _dedupe_refs(list(sga_leverage_trace)),
                "owner_value_capture_score": owner_value_capture_score if _is_num(owner_value_capture_score) else UNKNOWN,
                "owner_value_capture_score_derived_from": _dedupe_refs(list(owner_value_capture_trace)),
                "intangible_economics_total": intangible_economics_total if _is_num(intangible_economics_total) else UNKNOWN,
                "intangible_economics_total_derived_from": _dedupe_refs(list(intangible_total_trace)),
                "rnd_productivity_reason_codes": rnd_productivity_reason_codes,
                "sga_leverage_reason_codes": sga_leverage_reason_codes,
                "owner_value_capture_reason_codes": owner_value_capture_reason_codes,
                "intangible_economics_reason_codes": intangible_economics_reason_codes,
                "reinvestment_efficiency_class": reinvestment_efficiency_class,
                "reinvestment_efficiency_class_derived_from": _dedupe_refs(list(reinvestment_class_trace)),
                "reinvestment_efficiency_reason_codes": reinvestment_efficiency_reason_codes,
                "reinvestment_support_signals": reinvestment_support_signals,
                "reinvestment_headwind_signals": reinvestment_headwind_signals,
                "primary_reinvestment_caution": primary_reinvestment_caution,
                "reinvestment_efficiency_summary": reinvestment_efficiency_summary,
                "accounting_quality_class": accounting_quality_class,
                "accounting_quality_class_derived_from": _dedupe_refs(list(accounting_quality_trace)),
                "accounting_quality_reason_codes": accounting_quality_reason_codes,
                "cash_earnings_support_signals": cash_earnings_support_signals,
                "cash_earnings_headwind_signals": cash_earnings_headwind_signals,
                "primary_accounting_caution": primary_accounting_caution,
                "cash_earnings_discipline_summary": cash_earnings_discipline_summary,
                "balance_sheet_stress_class": balance_sheet_stress_class,
                "balance_sheet_stress_class_derived_from": _dedupe_refs(list(balance_sheet_stress_trace)),
                "balance_sheet_stress_reason_codes": balance_sheet_stress_reason_codes,
                "refinancing_risk_class": refinancing_risk_class,
                "refinancing_risk_class_derived_from": _dedupe_refs(list(refinancing_risk_trace)),
                "refinancing_risk_reason_codes": refinancing_risk_reason_codes,
                "balance_sheet_support_signals": balance_sheet_support_signals,
                "balance_sheet_headwind_signals": balance_sheet_headwind_signals,
                "primary_balance_sheet_caution": primary_balance_sheet_caution,
                "balance_sheet_discipline_summary": balance_sheet_discipline_summary,
                "returns_persistence_class": returns_persistence_class,
                "returns_persistence_class_derived_from": _dedupe_refs(list(returns_persistence_trace)),
                "returns_persistence_reason_codes": returns_persistence_reason_codes,
                "returns_support_signals": returns_support_signals,
                "returns_headwind_signals": returns_headwind_signals,
                "primary_returns_caution": primary_returns_caution,
                "economic_durability_summary": economic_durability_summary,
                "revenue_dependence_risk_class": revenue_dependence_risk_class,
                "revenue_dependence_risk_reason_codes": revenue_dependence_risk_reason_codes,
                "revenue_dependence_support_signals": revenue_dependence_support_signals,
                "revenue_dependence_headwind_signals": revenue_dependence_headwind_signals,
                "primary_revenue_dependence_caution": primary_revenue_dependence_caution,
                "revenue_fragility_summary": revenue_fragility_summary,
                "maintenance_capex_discipline_detail": maintenance_capex_payload,
                "asset_intensity_class": asset_intensity_class,
                "asset_intensity_reason_codes": asset_intensity_reason_codes,
                "maintenance_capex_credibility_class": maintenance_capex_credibility_class,
                "maintenance_capex_credibility_reason_codes": maintenance_capex_credibility_reason_codes,
                "maintenance_capex_support_signals": maintenance_capex_support_signals,
                "maintenance_capex_headwind_signals": maintenance_capex_headwind_signals,
                "primary_maintenance_capex_caution": primary_maintenance_capex_caution,
                "maintenance_capex_discipline_summary": maintenance_capex_discipline_summary,
                "normalized_earnings_power_value": normalized_earnings_power_value if _is_num(normalized_earnings_power_value) else UNKNOWN,
                "normalized_earnings_power_derived_from": _dedupe_refs(list(normalized_power_trace)),
                "normalized_earnings_power_value_derived_from": _dedupe_refs(list(normalized_power_trace)),
                "normalized_earnings_power_method_used": normalized_earnings_power_method_used,
                "normalized_earnings_power_status": normalized_earnings_power_status,
                "normalized_earnings_power_reason_codes": normalized_earnings_power_reason_codes,
                "intrinsic_floor": intrinsic_floor if _is_num(intrinsic_floor) else UNKNOWN,
                "intrinsic_floor_derived_from": _dedupe_refs(list(intrinsic_floor_trace)),
                "intrinsic_base": intrinsic_base_discipline if _is_num(intrinsic_base_discipline) else UNKNOWN,
                "intrinsic_base_derived_from": _dedupe_refs(list(intrinsic_base_trace)),
                "intrinsic_ceiling": intrinsic_ceiling if _is_num(intrinsic_ceiling) else UNKNOWN,
                "intrinsic_ceiling_derived_from": _dedupe_refs(list(intrinsic_ceiling_trace)),
                "mos_to_floor": mos_to_floor if _is_num(mos_to_floor) else UNKNOWN,
                "mos_to_floor_derived_from": _dedupe_refs(list(mos_to_floor_trace)),
                "mos_to_base": mos_to_base if _is_num(mos_to_base) else UNKNOWN,
                "mos_to_base_derived_from": _dedupe_refs(list(mos_to_base_trace)),
                "mos_classification": mos_classification,
                "downside_support_type": downside_support_type,
                "downside_support_status": downside_support_status,
                "valuation_range_reason_codes": valuation_range_reason_codes,
                "downside_support_reason_codes": downside_support_reason_codes,
                "valuation_support_count": int(valuation_support_count or 0),
                "valuation_support_count_derived_from": _dedupe_refs(list(valuation_support_trace)),
                "valuation_support_types_present": valuation_support_types_present,
                "valuation_support_count_reason_codes": valuation_support_count_reason_codes,
                "valuation_convergence_status": valuation_convergence_status,
                "valuation_convergence_band_pct": valuation_convergence_band_pct if _is_num(valuation_convergence_band_pct) else UNKNOWN,
                "valuation_convergence_band_pct_derived_from": _dedupe_refs(list(valuation_convergence_trace)),
                "valuation_convergence_reason_codes": valuation_convergence_reason_codes,
                "valuation_fragility_status": valuation_fragility_status,
                "valuation_fragility_status_derived_from": _dedupe_refs(list(valuation_fragility_trace)),
                "valuation_fragility_reason_codes": valuation_fragility_reason_codes,
                "valuation_confidence_class": valuation_confidence_class,
                "valuation_confidence_class_derived_from": _dedupe_refs(list(valuation_confidence_trace)),
                "valuation_confidence_reason_codes": valuation_confidence_reason_codes,
                "valuation_integrity_class": valuation_integrity_class,
                "valuation_integrity_class_derived_from": _dedupe_refs(list(valuation_integrity_trace)),
                "valuation_integrity_reason_codes": valuation_integrity_reason_codes,
                "evidence_sufficiency_class": evidence_sufficiency_class,
                "evidence_sufficiency_class_derived_from": _dedupe_refs(
                    list(evidence_sufficiency_trace)
                ),
                "evidence_sufficiency_reason_codes": evidence_sufficiency_reason_codes,
                "mos_assessment_status": mos_assessment_status,
                "mos_guardrail_reason_codes": mos_guardrail_reason_codes,
                "valuation_consistency_status": valuation_consistency_status,
                "valuation_consistency_reason_codes": valuation_consistency_reason_codes,
                "valuation_uniformity_group_id": valuation_uniformity_group_id,
                "valuation_uniformity_reason_codes": valuation_uniformity_reason_codes,
                "investment_readiness_class": investment_readiness_class,
                "investment_readiness_class_derived_from": _dedupe_refs(
                    list(investment_readiness_trace)
                ),
                "investment_readiness_reason_codes": investment_readiness_reason_codes,
                "blocker_stack_primary": blocker_stack_primary,
                "blocker_stack_secondary": blocker_stack_secondary,
                "blocker_stack_all": blocker_stack_all,
                "blocker_stack_retryable": blocker_stack_retryable,
                "blocker_stack_structural": blocker_stack_structural,
                "readiness_support_present": readiness_support_present,
                "readiness_support_missing": readiness_support_missing,
                "readiness_support_headwinds": readiness_support_headwinds,
                "primary_next_step": primary_next_step,
                "primary_next_step_reason": primary_next_step_reason,
                "value_type_primary": value_type_primary,
                "value_type_secondary": value_type_secondary,
                "value_type_reason_codes": value_type_reason_codes,
                "value_type_support_summary": value_type_support_summary,
                "value_type_derived_from": _dedupe_refs(list(value_type_trace)),
                "impairment_class_primary": str(impairment_classification_payload.get("impairment_class_primary") or "IMPAIRMENT_UNKNOWN"),
                "primary_underwriting_caution": str(impairment_classification_payload.get("primary_underwriting_caution") or "CAUTION_UNKNOWN"),
                "impairment_class_reason_codes": [
                    str(code)
                    for code in (impairment_classification_payload.get("impairment_class_reason_codes") or [])
                    if str(code).strip()
                ],
                "impairment_classification_detail": impairment_classification_payload,
                "normalization_credibility_detail": normalization_credibility_payload,
                "normalization_credibility_class": str(normalization_credibility_payload.get("normalization_credibility_class") or "NORMALIZATION_CREDIBILITY_UNKNOWN"),
                "primary_normalization_caution": str(normalization_credibility_payload.get("primary_normalization_caution") or "NORMALIZATION_UNCLEAR"),
                "capital_allocation_discipline_detail": capital_allocation_discipline_payload,
                "capital_allocation_discipline_class": str(capital_allocation_discipline_payload.get("capital_allocation_discipline_class") or "CAPITAL_ALLOCATION_UNKNOWN"),
                "primary_capital_allocation_caution": str(capital_allocation_discipline_payload.get("primary_capital_allocation_caution") or "CAPITAL_ALLOCATION_UNCLEAR"),
                "maintenance_capex_discipline_detail": maintenance_capex_payload,
                "accounting_quality_detail": accounting_quality_payload,
                "balance_sheet_stress_detail": balance_sheet_stress_payload,
                "returns_persistence_detail": returns_persistence_payload,
                "quality_score": quality_score if _is_num(quality_score) else UNKNOWN,
                "risk_penalty": risk_penalty if _is_num(risk_penalty) else UNKNOWN,
                "price_status": price_status,
                "valuation_status": valuation_status,
                "shares_status": shares_status,
                "fcf_status": fcf_status,
                "facts_status": facts_status,
                "price_reason_code": price_reason_code,
                "valuation_reason_code": valuation_reason_code,
                "shares_reason_code": shares_reason_code,
                "fcf_reason_code": fcf_reason_code,
                "facts_reason_code": facts_reason_code,
                "notes": notes,
                "derived_from": _dedupe_refs(
                    [f"run:{run_id}"]
                    + list(score_row.get("metric_values", {}).keys())
                    + list(score_row.get("metric_traces", {}).keys())
                    + list(gate_row.get("gate_reasons") or [])
                    + list(valuation_row.get("derived_from") or [])
                    + list(shares_row.get("derived_from") or [])
                    + list(fcf_row.get("derived_from") or [])
                    + list(facts_row.get("derived_from") or [])
                    + oe_quality_trace
                    + owner_stability_trace
                    + capital_allocation_trace
                    + cash_conversion_trace
                    + gross_margin_trace
                    + balance_sheet_optionality_trace
                    + cycle_resilience_trace
                    + rnd_productivity_trace
                    + sga_leverage_trace
                    + owner_value_capture_trace
                    + intangible_total_trace
                    + reinvestment_class_trace
                    + accounting_quality_trace
                    + balance_sheet_stress_trace
                    + refinancing_risk_trace
                    + returns_persistence_trace
                    + list(maintenance_capex_payload.get("derived_from") or [])
                    + normalized_power_trace
                    + intrinsic_floor_trace
                    + intrinsic_base_trace
                    + intrinsic_ceiling_trace
                    + mos_to_floor_trace
                    + mos_to_base_trace
                    + valuation_support_trace
                    + valuation_convergence_trace
                    + valuation_fragility_trace
                    + valuation_confidence_trace
                    + valuation_integrity_trace
                    + evidence_sufficiency_trace
                    + investment_readiness_trace
                    + value_type_trace
                ),
            }
        )

    metadata = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "sector": sector,
        "as_of_date": as_of_date,
        "thresholds_effective": (
            rlm_state_payload.get("gate_thresholds_effective")
            if isinstance(rlm_state_payload.get("gate_thresholds_effective"), dict)
            else {}
        ),
        "coverage_counts": {
            "price_reason_counts": (
                valuation_payload.get("reason_counts")
                if isinstance(valuation_payload.get("reason_counts"), dict)
                else {}
            ),
            "shares_reason_counts": (
                shares_payload.get("reason_counts")
                if isinstance(shares_payload.get("reason_counts"), dict)
                else {}
            ),
            "fcf_reason_counts": (
                fcf_payload.get("reason_counts")
                if isinstance(fcf_payload.get("reason_counts"), dict)
                else {}
            ),
            "facts_reason_counts": (
                facts_payload.get("reason_counts", {}).get("shares_reason")
                if isinstance(facts_payload.get("reason_counts"), dict)
                and isinstance(facts_payload.get("reason_counts", {}).get("shares_reason"), dict)
                else {}
            ),
            "valuation_reason_counts": _reason_counts_from_entries(valuation_rows, "valuation_reason_code"),
        },
        "value_gates_counts": (
            (value_gates_payload.get("summary") or {}).get("counts")
            if isinstance(value_gates_payload.get("summary"), dict)
            and isinstance((value_gates_payload.get("summary") or {}).get("counts"), dict)
            else {}
        ),
    }
    return candidates, metadata


def collect_depth_run_artifacts(run_dir_paths: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    run_meta_out: list[dict[str, Any]] = []
    for run_meta in run_dir_paths:
        if not isinstance(run_meta, dict):
            continue
        candidates, metadata = _collect_run_candidates(run_meta)
        rows.extend(candidates)
        run_meta_out.append(metadata)
    return {
        "rows": rows,
        "run_metadata": run_meta_out,
    }


def build_global_shortlist(rows: list[dict[str, Any]], top_n: int, policy: str) -> dict[str, Any]:
    policy_norm = str(policy or "value_first").strip().lower()
    if policy_norm not in {"value_first", "value_first_quality", "value_first_intangible", "value_first_discipline", "value_first_confidence", "value_first_typed", "value_first_trustworthy", "value_first_ready", "value_first_cycle_aware", "value_first_reinvestment", "value_first_cash_earnings", "value_first_balance_sheet", "value_first_durable_returns", "value_first_revenue_resilience", "value_first_owner_earnings_hardness"}:
        raise ValueError(f"Unsupported policy={policy}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        grouped.setdefault(ticker, []).append(row)

    merged_rows: list[dict[str, Any]] = []
    sort_key = _sort_key_value_first
    if policy_norm == "value_first_quality":
        sort_key = _sort_key_value_first_quality
    elif policy_norm == "value_first_intangible":
        sort_key = _sort_key_value_first_intangible
    elif policy_norm == "value_first_discipline":
        sort_key = _sort_key_value_first_discipline
    elif policy_norm == "value_first_confidence":
        sort_key = _sort_key_value_first_confidence
    elif policy_norm == "value_first_typed":
        sort_key = _sort_key_value_first_typed
    elif policy_norm == "value_first_trustworthy":
        sort_key = _sort_key_value_first_trustworthy
    elif policy_norm == "value_first_ready":
        sort_key = _sort_key_value_first_ready
    elif policy_norm == "value_first_cycle_aware":
        sort_key = _sort_key_value_first_cycle_aware
    elif policy_norm == "value_first_reinvestment":
        sort_key = _sort_key_value_first_reinvestment
    elif policy_norm == "value_first_cash_earnings":
        sort_key = _sort_key_value_first_cash_earnings
    elif policy_norm == "value_first_balance_sheet":
        sort_key = _sort_key_value_first_balance_sheet
    elif policy_norm == "value_first_durable_returns":
        sort_key = _sort_key_value_first_durable_returns
    elif policy_norm == "value_first_revenue_resilience":
        sort_key = _sort_key_value_first_revenue_resilience
    elif policy_norm == "value_first_owner_earnings_hardness":
        sort_key = _sort_key_value_first_owner_earnings_hardness
    for ticker in sorted(grouped.keys()):
        ticker_rows = grouped[ticker]
        representative = sorted(ticker_rows, key=sort_key)[0]
        source_depth_runs = []
        source_seen: set[tuple[str, str, str]] = set()
        for row in sorted(ticker_rows, key=lambda item: (str(item.get("run_id") or ""), str(item.get("sector") or ""), str(item.get("as_of_date") or ""))):
            key = (
                str(row.get("run_id") or ""),
                str(row.get("sector") or ""),
                str(row.get("as_of_date") or ""),
            )
            if key in source_seen:
                continue
            source_seen.add(key)
            source_depth_runs.append(
                {"run_id": key[0], "sector": key[1], "as_of_date": key[2]}
            )

        merged = dict(representative)
        merged["ticker"] = ticker
        merged["source_depth_runs"] = source_depth_runs
        merged["derived_from"] = _dedupe_refs(
            [token for row in ticker_rows for token in list(row.get("derived_from") or [])]
        )
        merged_rows.append(merged)

    ranked = sorted(merged_rows, key=sort_key)
    top_n_eff = max(1, int(top_n))
    shortlist_rows = ranked[:top_n_eff]
    return {
        "policy": policy_norm,
        "rows_all": ranked,
        "rows_top_n": shortlist_rows,
    }


def _aggregate_rollup(
    *,
    universe_run_id: str,
    batch_run_id: str,
    batch_state: dict[str, Any],
    batch_summary: dict[str, Any],
    run_metadata: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    planned_runs = [row for row in (batch_state.get("planned_runs") or []) if isinstance(row, dict)]
    completed_runs = [row for row in (batch_state.get("completed_runs") or []) if isinstance(row, dict)]

    run_count_total = len(planned_runs)
    run_count_done = len([row for row in completed_runs if str(row.get("status") or "").upper() in {"DONE", "COMPLETED"}])
    run_count_failed = len([row for row in completed_runs if str(row.get("status") or "").upper() == "FAILED"])
    run_count_cancelled = len([row for row in completed_runs if str(row.get("status") or "").upper() == "CANCELLED"])

    price_reason_counts: dict[str, int] = {}
    valuation_reason_counts: dict[str, int] = {}
    shares_reason_counts: dict[str, int] = {}
    fcf_reason_counts: dict[str, int] = {}
    facts_reason_counts: dict[str, int] = {}
    value_gate_counts: dict[str, int] = {"PASS": 0, "WATCH": 0, "FAIL": 0}
    blocker_counts: dict[str, int] = {}
    thresholds_effective_by_run: dict[str, Any] = {}

    for row in ranked_rows:
        gate_status = str(row.get("value_gate_status") or "UNKNOWN").upper()
        if gate_status in value_gate_counts:
            value_gate_counts[gate_status] += 1
        blocker = str(row.get("primary_blocker") or "NONE")
        if blocker and blocker not in {"NONE", UNKNOWN}:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1

        for bucket, key in [
            (price_reason_counts, "price_reason_code"),
            (valuation_reason_counts, "valuation_reason_code"),
            (shares_reason_counts, "shares_reason_code"),
            (fcf_reason_counts, "fcf_reason_code"),
            (facts_reason_counts, "facts_reason_code"),
        ]:
            code = str(row.get(key) or "").strip()
            if not code or code == UNKNOWN:
                continue
            bucket[code] = bucket.get(code, 0) + 1

    for meta in run_metadata:
        run_id = str(meta.get("run_id") or "")
        thresholds = meta.get("thresholds_effective") if isinstance(meta.get("thresholds_effective"), dict) else {}
        if run_id and thresholds:
            thresholds_effective_by_run[run_id] = thresholds

    thresholds_snapshot = {
        "batch_execution_defaults": (
            batch_state.get("execution_defaults")
            if isinstance(batch_state.get("execution_defaults"), dict)
            else {}
        ),
        "gate_thresholds_effective_by_run": thresholds_effective_by_run,
    }

    return {
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "generated_at": utc_now_iso(),
        "run_count_total": run_count_total,
        "run_count_done": run_count_done,
        "run_count_failed": run_count_failed,
        "run_count_cancelled": run_count_cancelled,
        "candidate_count_total": len(ranked_rows),
        "candidate_count_ranked": len(ranked_rows),
        "coverage_breakdown": {
            "price_reason_counts": dict(sorted(price_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "valuation_reason_counts": dict(sorted(valuation_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "shares_reason_counts": dict(sorted(shares_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "fcf_reason_counts": dict(sorted(fcf_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "facts_reason_counts": dict(sorted(facts_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "value_gate_counts": value_gate_counts,
        "top_blocker_categories": _top_counts(blocker_counts, top_n=10),
        "thresholds_effective_snapshot": thresholds_snapshot,
        "batch_state_status": str(batch_state.get("status") or UNKNOWN),
        "batch_summary_status": str(batch_summary.get("status") or UNKNOWN) if isinstance(batch_summary, dict) else UNKNOWN,
    }


def _shortlist_markdown(*, rollup: dict[str, Any], shortlist_rows: list[dict[str, Any]], top_n: int) -> str:
    lines: list[str] = []
    lines.append("# Depth Batch Global Shortlist")
    lines.append("")
    lines.append(f"- Universe Run: `{rollup.get('universe_run_id')}`")
    lines.append(f"- Batch Run: `{rollup.get('batch_run_id')}`")
    lines.append(f"- Generated At: `{rollup.get('generated_at')}`")
    lines.append(f"- Top N: `{int(top_n)}`")
    lines.append("")
    lines.append("| Rank | Ticker | Gate | Ready | MOS Floor | Implied Return | Valuation Conf. | Integrity | Returns | Accounting | Value Type | Fragility | MOS EPV | OE Yield EV 3Y | OE Quality | Intangible | Reinvestment | Downside Support | Blocker | Next Step |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for idx, row in enumerate(shortlist_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("ticker") or ""),
                    str(row.get("value_gate_status") or UNKNOWN),
                    str(row.get("investment_readiness_class") or UNKNOWN),
                    str(row.get("mos_to_floor") if _is_num(row.get("mos_to_floor")) else UNKNOWN),
                    str(row.get("implied_return_base") if _is_num(row.get("implied_return_base")) else UNKNOWN),
                    str(row.get("valuation_confidence_class") or UNKNOWN),
                    str(row.get("valuation_integrity_class") or UNKNOWN),
                    str(row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"),
                    str(row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"),
                    str(row.get("value_type_primary") or UNKNOWN),
                    str(row.get("valuation_fragility_status") or UNKNOWN),
                    str(row.get("mos_epv") if _is_num(row.get("mos_epv")) else UNKNOWN),
                    str(row.get("owner_earnings_yield_ev_3y") if _is_num(row.get("owner_earnings_yield_ev_3y")) else UNKNOWN),
                    str(row.get("oe_quality_total") if _is_num(row.get("oe_quality_total")) else UNKNOWN),
                    str(row.get("intangible_economics_total") if _is_num(row.get("intangible_economics_total")) else UNKNOWN),
                    str(row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"),
                    str(row.get("downside_support_type") or UNKNOWN),
                    str(row.get("blocker_stack_primary") or row.get("primary_blocker") or "NONE"),
                    str(row.get("primary_next_step") or UNKNOWN),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Coverage Breakdown")
    cov = rollup.get("coverage_breakdown") if isinstance(rollup.get("coverage_breakdown"), dict) else {}
    for key in ["price_reason_counts", "valuation_reason_counts", "shares_reason_counts", "fcf_reason_counts", "facts_reason_counts"]:
        lines.append(f"- `{key}`: `{json.dumps(cov.get(key, {}), sort_keys=True)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_depth_batch_rollup(
    universe_run_id: str,
    batch_run_id: str,
    top_n: int = 25,
    policy: str = "value_first",
) -> dict[str, Any]:
    batch_payload = load_batch_summary(universe_run_id, batch_run_id)
    batch_state = batch_payload["batch_state"]
    batch_summary = batch_payload["batch_summary"]

    planned_runs = [row for row in (batch_state.get("planned_runs") or []) if isinstance(row, dict)]
    completed_runs = [row for row in (batch_state.get("completed_runs") or []) if isinstance(row, dict)]
    completed_by_id = {str(row.get("run_id") or ""): row for row in completed_runs}

    run_items: list[dict[str, Any]] = []
    for planned in planned_runs:
        run_id = str(planned.get("run_id") or "")
        completed = completed_by_id.get(run_id, {})
        run_items.append(
            {
                "run_id": run_id,
                "sector": str(planned.get("sector") or completed.get("sector") or "UNKNOWN_SECTOR"),
                "as_of_date": str(planned.get("as_of_date") or ""),
                "tickers": [str(token).upper() for token in (planned.get("tickers") or []) if str(token).strip()],
                "status": str(completed.get("status") or planned.get("status") or UNKNOWN),
                "artifacts_paths": completed.get("artifacts_paths") if isinstance(completed.get("artifacts_paths"), dict) else {},
            }
        )

    collected = collect_depth_run_artifacts(run_items)
    rows = [row for row in (collected.get("rows") or []) if isinstance(row, dict)]
    run_metadata = [row for row in (collected.get("run_metadata") or []) if isinstance(row, dict)]
    shortlist_payload = build_global_shortlist(rows, top_n=max(1, int(top_n)), policy=policy)
    ranked_rows = shortlist_payload["rows_all"]
    shortlist_rows = shortlist_payload["rows_top_n"]

    rollup = _aggregate_rollup(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        batch_state=batch_state,
        batch_summary=batch_summary,
        run_metadata=run_metadata,
        ranked_rows=ranked_rows,
    )

    batch_dir = _batch_dir(universe_run_id, batch_run_id)
    shortlist_json_path = batch_dir / "global_shortlist.json"
    shortlist_md_path = batch_dir / "global_shortlist.md"
    rollup_json_path = batch_dir / "global_rollup.json"

    shortlist_json = {
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "policy": shortlist_payload["policy"],
        "top_n": int(top_n),
        "candidate_count_total": len(ranked_rows),
        "candidate_count_selected": len(shortlist_rows),
        "rows": shortlist_rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(shortlist_json_path, shortlist_json)
    shortlist_md_path.write_text(
        _shortlist_markdown(rollup=rollup, shortlist_rows=shortlist_rows, top_n=max(1, int(top_n))),
        encoding="utf-8",
    )
    _json_write(rollup_json_path, rollup)
    _batch_as_of_date = str(
        next(
            (str(r.get("as_of_date") or "") for r in ranked_rows if str(r.get("as_of_date") or "").strip()),
            "",
        )
    )
    write_normalization_credibility_for_run(
        run_id=batch_run_id,
        as_of_date=_batch_as_of_date,
        tickers=[str(r.get("ticker") or "") for r in ranked_rows if str(r.get("ticker") or "").strip()],
        output_path=batch_dir / "normalization_credibility.json",
        scoreboard_rows=ranked_rows,
    )
    write_capital_allocation_discipline_for_run(
        run_id=batch_run_id,
        as_of_date=_batch_as_of_date,
        tickers=[str(r.get("ticker") or "") for r in ranked_rows if str(r.get("ticker") or "").strip()],
        output_path=batch_dir / "capital_allocation_discipline.json",
        scoreboard_rows=ranked_rows,
    )

    return {
        "status": "OK",
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "policy": shortlist_payload["policy"],
        "top_n": int(top_n),
        "candidate_count_total": len(ranked_rows),
        "candidate_count_selected": len(shortlist_rows),
        "global_shortlist_json_path": str(shortlist_json_path),
        "global_shortlist_md_path": str(shortlist_md_path),
        "global_rollup_json_path": str(rollup_json_path),
        "top_rows": shortlist_rows[:10],
    }


def open_depth_batch_rollup(
    universe_run_id: str,
    batch_run_id: str,
    top_n: int = 10,
) -> dict[str, Any]:
    batch_dir = _batch_dir(universe_run_id, batch_run_id)
    shortlist_path = batch_dir / "global_shortlist.json"
    rollup_path = batch_dir / "global_rollup.json"
    if not shortlist_path.exists() or not rollup_path.exists():
        return {
            "status": "MISSING",
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "global_shortlist_json_path": str(shortlist_path),
            "global_rollup_json_path": str(rollup_path),
        }

    shortlist_payload = _safe_json(shortlist_path)
    rollup_payload = _safe_json(rollup_path)
    rows = [row for row in (shortlist_payload.get("rows") or []) if isinstance(row, dict)]
    top_rows = rows[: max(1, int(top_n))]
    preview = [
        {
            "ticker": str(row.get("ticker") or ""),
            "value_gate_status": str(row.get("value_gate_status") or UNKNOWN),
            "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
            "implied_return_base": row.get("implied_return_base", UNKNOWN),
            "mos_epv": row.get("mos_epv", UNKNOWN),
            "valuation_support_count": int(row.get("valuation_support_count") or 0),
            "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
            "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
            "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
            "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
            "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
            "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
            "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
            "value_type_primary": str(row.get("value_type_primary") or UNKNOWN),
            "yield_metric_used": "owner_earnings_yield_ev_3y"
            if _is_num(row.get("owner_earnings_yield_ev_3y"))
            else ("fcf_yield_ev_3y" if _is_num(row.get("fcf_yield_ev_3y")) else UNKNOWN),
            "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
            "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
            "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
            "reinvestment_efficiency_class": str(
                row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
            ),
            "primary_reinvestment_caution": str(
                row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
            ),
            "returns_persistence_class": str(
                row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
            ),
            "primary_returns_caution": str(
                row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
            ),
            "revenue_dependence_risk_class": str(
                row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
            ),
            "primary_revenue_dependence_caution": str(
                row.get("primary_revenue_dependence_caution") or "REVENUE_BASE_UNCLEAR"
            ),
            "asset_intensity_class": str(
                row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
            ),
            "maintenance_capex_credibility_class": str(
                row.get("maintenance_capex_credibility_class") or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
            ),
            "primary_maintenance_capex_caution": str(
                row.get("primary_maintenance_capex_caution") or "OWNER_EARNINGS_UNCLEAR"
            ),
            "accounting_quality_class": str(
                row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
            ),
            "primary_accounting_caution": str(
                row.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
            ),
        }
        for row in top_rows
    ]
    return {
        "status": "OK",
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "run_counts": {
            "run_count_total": int(rollup_payload.get("run_count_total") or 0),
            "run_count_done": int(rollup_payload.get("run_count_done") or 0),
            "run_count_failed": int(rollup_payload.get("run_count_failed") or 0),
            "run_count_cancelled": int(rollup_payload.get("run_count_cancelled") or 0),
        },
        "candidate_counts": {
            "candidate_count_total": int(rollup_payload.get("candidate_count_total") or 0),
            "candidate_count_ranked": int(rollup_payload.get("candidate_count_ranked") or 0),
        },
        "coverage_breakdown": (
            rollup_payload.get("coverage_breakdown")
            if isinstance(rollup_payload.get("coverage_breakdown"), dict)
            else {}
        ),
        "top_shortlist": preview,
        "global_shortlist_json_path": str(shortlist_path),
        "global_shortlist_md_path": str(batch_dir / "global_shortlist.md"),
        "global_rollup_json_path": str(rollup_path),
    }
