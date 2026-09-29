from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"
OK = "OK"

# Units of the cash-flow series a caller hands in "fundamentals". Net debt is
# $millions everywhere in the valuation spine (app/valuation/net_debt.py), but
# the row series is not: on the universe-scout path it comes from
# app/valuation/owner_earnings.py, which uses companyfacts "val" unscaled —
# whole dollars. The caller declares which it holds and both operands are
# brought to $millions before any ratio is formed.
CASHFLOW_UNITS_USD_MILLIONS = "USD_MILLIONS"
CASHFLOW_UNITS_USD = "USD"
_USD_TO_MUSD = 1_000_000.0
_DOLLAR_UNIT_TOKENS = {"USD", "USD_DOLLARS", "DOLLARS", "WHOLE_DOLLARS"}

LOW_BALANCE_SHEET_STRESS = "LOW_BALANCE_SHEET_STRESS"
MODERATE_BALANCE_SHEET_STRESS = "MODERATE_BALANCE_SHEET_STRESS"
HIGH_BALANCE_SHEET_STRESS = "HIGH_BALANCE_SHEET_STRESS"
BALANCE_SHEET_STRESS_UNKNOWN = "BALANCE_SHEET_STRESS_UNKNOWN"

LOW_REFINANCING_RISK = "LOW_REFINANCING_RISK"
MODERATE_REFINANCING_RISK = "MODERATE_REFINANCING_RISK"
HIGH_REFINANCING_RISK = "HIGH_REFINANCING_RISK"
REFINANCING_RISK_UNKNOWN = "REFINANCING_RISK_UNKNOWN"

BALANCE_SHEET_SUPPORTIVE = "BALANCE_SHEET_SUPPORTIVE"
BALANCE_SHEET_MIXED = "BALANCE_SHEET_MIXED"
BALANCE_SHEET_HEADWIND = "BALANCE_SHEET_HEADWIND"
BALANCE_SHEET_UNCLEAR = "BALANCE_SHEET_UNCLEAR"

SIG_NET_CASH_OR_LOW_NET_DEBT = "NET_CASH_OR_LOW_NET_DEBT"
SIG_STRONG_CASH_TO_DEBT = "STRONG_CASH_TO_DEBT"
SIG_LOW_NET_DEBT_TO_CFO = "LOW_NET_DEBT_TO_CFO"
SIG_BALANCE_SHEET_OPTIONALITY_PRESENT = "BALANCE_SHEET_OPTIONALITY_PRESENT"
SIG_LIQUIDITY_SUPPORT_PRESENT = "LIQUIDITY_SUPPORT_PRESENT"
SIG_CAPITAL_STRUCTURE_NOT_DOMINANT = "CAPITAL_STRUCTURE_NOT_DOMINANT"

SIG_HIGH_NET_DEBT_TO_CFO = "HIGH_NET_DEBT_TO_CFO"
SIG_HIGH_LEVERAGE_TO_OWNER_EARNINGS = "HIGH_LEVERAGE_TO_OWNER_EARNINGS"
SIG_WEAK_CASH_CUSHION = "WEAK_CASH_CUSHION"
SIG_REFINANCING_DEPENDENCE_HEADWIND = "REFINANCING_DEPENDENCE_HEADWIND"
SIG_POSSIBLE_DILUTION_RISK = "POSSIBLE_DILUTION_RISK"
SIG_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE = "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"
SIG_DEBT_INPUTS_THIN = "DEBT_INPUTS_THIN"
SIG_NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW = "NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW"

REASON_STRONG_LIQUIDITY_SUPPORT = "STRONG_LIQUIDITY_SUPPORT"
REASON_LOW_NET_DEBT_TO_CFO = "LOW_NET_DEBT_TO_CFO"
REASON_MODERATE_LEVERAGE = "MODERATE_LEVERAGE"
REASON_HIGH_NET_DEBT_TO_CFO = "HIGH_NET_DEBT_TO_CFO"
REASON_NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW = "NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW"
REASON_WEAK_CASH_TO_DEBT = "WEAK_CASH_TO_DEBT"
REASON_REFINANCING_DEPENDENCE_POSSIBLE = "REFINANCING_DEPENDENCE_POSSIBLE"
REASON_BALANCE_SHEET_OPTIONALITY_STRONG = "BALANCE_SHEET_OPTIONALITY_STRONG"
REASON_BALANCE_SHEET_OPTIONALITY_WEAK = "BALANCE_SHEET_OPTIONALITY_WEAK"
REASON_INSUFFICIENT_DEBT_INPUTS = "INSUFFICIENT_DEBT_INPUTS"
REASON_LOW_BALANCE_SHEET_STRESS_SUPPORT = "LOW_BALANCE_SHEET_STRESS_SUPPORT"
REASON_HIGH_BALANCE_SHEET_STRESS_HEADWIND = "HIGH_BALANCE_SHEET_STRESS_HEADWIND"
REASON_HIGH_REFINANCING_RISK_HEADWIND = "HIGH_REFINANCING_RISK_HEADWIND"
REASON_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE = "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"
REASON_BALANCE_SHEET_EVIDENCE_THIN = "BALANCE_SHEET_EVIDENCE_THIN"

_STRESS_ORDER = {
    LOW_BALANCE_SHEET_STRESS: 0,
    MODERATE_BALANCE_SHEET_STRESS: 1,
    BALANCE_SHEET_STRESS_UNKNOWN: 2,
    HIGH_BALANCE_SHEET_STRESS: 3,
}
_REFINANCING_ORDER = {
    LOW_REFINANCING_RISK: 0,
    MODERATE_REFINANCING_RISK: 1,
    REFINANCING_RISK_UNKNOWN: 2,
    HIGH_REFINANCING_RISK: 3,
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


def _coalesce_num(*values: Any) -> float | str:
    for value in values:
        if _is_num(value):
            return float(value)
    return UNKNOWN


def _series_rows(fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("year") or 0))
    return rows


def _trace_refs_from_row(row: dict[str, Any], key: str) -> list[str]:
    year = int(row.get("year") or 0)
    metric = str(key).strip()
    if year <= 0 or not metric:
        return []
    return [f"fundamentals.rows[{year}].{metric}"]


def _latest_metric(rows: list[dict[str, Any]], *keys: str) -> tuple[float | str, list[str]]:
    for row in sorted(rows, key=lambda item: int(item.get("year") or 0), reverse=True):
        for key in keys:
            value = row.get(key, UNKNOWN)
            if _is_num(value):
                return float(value), _trace_refs_from_row(row, key)
    return UNKNOWN, []


def _latest_metric_year(rows: list[dict[str, Any]], *keys: str) -> int | None:
    for row in sorted(rows, key=lambda item: int(item.get("year") or 0), reverse=True):
        if any(_is_num(row.get(key, UNKNOWN)) for key in keys):
            return int(row.get("year") or 0) or None
    return None


def _balance_sheet_year(net_debt_payload: dict[str, Any]) -> int | None:
    years: list[int] = []
    for key in ("total_debt", "cash_equivalents"):
        part = net_debt_payload.get(key)
        end = str(part.get("period_end") or "") if isinstance(part, dict) else ""
        if len(end) >= 4 and end[:4].isdigit():
            years.append(int(end[:4]))
    return max(years) if years else None


def _latest_positive_ratio(
    numerator: Any,
    denominator: Any,
    *,
    refs: list[Any] | None = None,
) -> tuple[float | str, list[str]]:
    derived = _dedupe(list(refs or []))
    if not _is_num(numerator) or not _is_num(denominator):
        return UNKNOWN, derived
    if float(denominator) <= 0.0:
        return UNKNOWN, derived
    return float(numerator) / float(denominator), derived


def _collect_derived_from(*payloads: dict[str, Any], row_refs: list[Any] | None = None) -> list[str]:
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


def _summary_for_profile(
    *,
    stress_class: str,
    refinancing_class: str,
    support_signals: list[str],
    headwind_signals: list[str],
) -> str:
    if stress_class == LOW_BALANCE_SHEET_STRESS and refinancing_class == LOW_REFINANCING_RISK:
        return "capital structure appears supportive and non-dominant; leverage and liquidity look manageable"
    if stress_class == HIGH_BALANCE_SHEET_STRESS:
        if SIG_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE in headwind_signals:
            return "downside appears capital-structure dominated; leverage and liquidity could break value realization"
        if refinancing_class == HIGH_REFINANCING_RISK:
            return "refinancing dependence is a real headwind; leverage and liquidity leave little room for error"
        return "balance-sheet stress is elevated; capital structure weakens owner protection materially"
    if stress_class == MODERATE_BALANCE_SHEET_STRESS or refinancing_class == MODERATE_REFINANCING_RISK:
        return "leverage is manageable but not strong; capital structure support is mixed and depends on steady economics"
    if headwind_signals:
        return "balance-sheet risk is unclear, but visible leverage or liquidity headwinds prevent confidence"
    if support_signals:
        return "some balance-sheet support is present, but evidence remains too thin to judge conservatively"
    return "evidence too thin to judge whether the capital structure supports value realization"


def compute_balance_sheet_stress(
    ticker: str,
    as_of_date: str,
    *,
    fundamentals: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    intangible_payload: dict[str, Any] | None = None,
    evidence_sufficiency_payload: dict[str, Any] | None = None,
    net_debt_payload: dict[str, Any] | None = None,
    net_debt_proxy: Any = UNKNOWN,
    total_debt: Any = UNKNOWN,
    cash_equivalents: Any = UNKNOWN,
    net_debt_to_cfo: Any = UNKNOWN,
    net_debt_to_owner_earnings: Any = UNKNOWN,
    fundamentals_cashflow_units: str = CASHFLOW_UNITS_USD_MILLIONS,
    price_status: Any = UNKNOWN,
    facts_status: Any = UNKNOWN,
    shares_status: Any = UNKNOWN,
    fcf_status: Any = UNKNOWN,
    row_derived_from: list[Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    del cfg

    ticker_norm = str(ticker or "").strip().upper()
    fundamentals = fundamentals if isinstance(fundamentals, dict) else {}
    owner_quality_payload = owner_quality_payload if isinstance(owner_quality_payload, dict) else {}
    intangible_payload = intangible_payload if isinstance(intangible_payload, dict) else {}
    evidence_sufficiency_payload = (
        evidence_sufficiency_payload if isinstance(evidence_sufficiency_payload, dict) else {}
    )
    net_debt_payload = net_debt_payload if isinstance(net_debt_payload, dict) else {}
    rows = _series_rows(fundamentals)

    latest_net_debt, latest_net_debt_refs = _latest_metric(rows, "net_debt")
    latest_cfo, latest_cfo_refs = _latest_metric(rows, "cfo")
    # A cash flow more than one fiscal year older than the net-debt balance sheet says
    # nothing about servicing that debt: no ratio (UNKNOWN), and no signal from it.
    _cfo_year = _latest_metric_year(rows, "cfo")
    _bs_year = _balance_sheet_year(net_debt_payload)
    if _cfo_year is not None and _bs_year is not None and _cfo_year < _bs_year - 1:
        latest_cfo, latest_cfo_refs = UNKNOWN, []
    latest_owner_earnings, latest_owner_refs = _latest_metric(rows, "owner_earnings", "fcf")
    latest_total_debt, latest_total_debt_refs = _latest_metric(rows, "total_debt", "debt")
    latest_cash, latest_cash_refs = _latest_metric(
        rows,
        "cash_equivalents",
        "cash_and_equivalents",
        "cash",
    )

    net_debt_value = _coalesce_num(
        net_debt_proxy,
        net_debt_payload.get("net_debt_proxy"),
        latest_net_debt,
    )
    total_debt_value = _coalesce_num(
        total_debt,
        (net_debt_payload.get("total_debt") or {}).get("value")
        if isinstance(net_debt_payload.get("total_debt"), dict)
        else UNKNOWN,
        latest_total_debt,
    )
    cash_value = _coalesce_num(
        cash_equivalents,
        (net_debt_payload.get("cash_equivalents") or {}).get("value")
        if isinstance(net_debt_payload.get("cash_equivalents"), dict)
        else UNKNOWN,
        latest_cash,
    )

    owner_reason_codes = {
        str(code)
        for code in (owner_quality_payload.get("oe_quality_reason_codes") or [])
        if str(code).strip()
    }
    optionality_score = intangible_payload.get("balance_sheet_optionality_score", UNKNOWN)
    optionality_reason_codes = {
        str(code)
        for code in (intangible_payload.get("balance_sheet_optionality_reason_codes") or [])
        if str(code).strip()
    }

    debt_refs = _dedupe(
        list(net_debt_payload.get("derived_from") or [])
        + latest_net_debt_refs
        + latest_total_debt_refs
        + latest_cash_refs
    )
    # Bring the row-derived cash flows onto net debt's scale before dividing.
    # Whole dollars against $millions net debt made a plainly 4x-levered issuer
    # read 0.000004 and collect the LOW_NET_DEBT_TO_CFO *support* signal — the
    # error flatters exactly the companies it should flag.
    if str(fundamentals_cashflow_units or "").strip().upper() in _DOLLAR_UNIT_TOKENS:
        if _is_num(latest_cfo):
            latest_cfo = float(latest_cfo) / _USD_TO_MUSD
        if _is_num(latest_owner_earnings):
            latest_owner_earnings = float(latest_owner_earnings) / _USD_TO_MUSD

    cfo_ratio_value, cfo_ratio_refs = _latest_positive_ratio(
        _coalesce_num(net_debt_to_cfo, net_debt_value),
        latest_cfo if not _is_num(net_debt_to_cfo) else 1.0,
        refs=debt_refs + latest_cfo_refs,
    )
    if _is_num(net_debt_to_cfo):
        cfo_ratio_value = float(net_debt_to_cfo)
        cfo_ratio_refs = _dedupe(debt_refs + latest_cfo_refs)

    owner_ratio_value, owner_ratio_refs = _latest_positive_ratio(
        _coalesce_num(net_debt_to_owner_earnings, net_debt_value),
        latest_owner_earnings if not _is_num(net_debt_to_owner_earnings) else 1.0,
        refs=debt_refs + latest_owner_refs,
    )
    if _is_num(net_debt_to_owner_earnings):
        owner_ratio_value = float(net_debt_to_owner_earnings)
        owner_ratio_refs = _dedupe(debt_refs + latest_owner_refs)

    # Net debt against a cash flow that is zero or negative has no ratio, but it is the
    # opposite of unformable: nothing is generated to service the debt, so it reads as
    # leverage above every threshold, never as low or moderate stress. A negative ratio handed in by a caller is the same case, not a small
    # multiple.
    net_debt_positive = _is_num(net_debt_value) and float(net_debt_value) > 0.0

    def _cannot_service(ratio_in: Any, latest_flow: Any) -> bool:
        if not net_debt_positive:
            return False
        if _is_num(ratio_in):
            return float(ratio_in) < 0.0
        return _is_num(latest_flow) and float(latest_flow) <= 0.0

    cfo_cannot_service_net_debt = _cannot_service(net_debt_to_cfo, latest_cfo)
    owner_cannot_service_net_debt = _cannot_service(net_debt_to_owner_earnings, latest_owner_earnings)
    if cfo_cannot_service_net_debt:
        cfo_ratio_value = UNKNOWN
    if owner_cannot_service_net_debt:
        owner_ratio_value = UNKNOWN

    cash_to_debt = UNKNOWN
    cash_to_debt_refs: list[str] = []
    if _is_num(cash_value) and _is_num(total_debt_value):
        cash_to_debt_refs = _dedupe(debt_refs)
        if float(total_debt_value) > 0.0:
            cash_to_debt = float(cash_value) / float(total_debt_value)
        elif float(total_debt_value) == 0.0:
            cash_to_debt = UNKNOWN

    mos_assessment_status = str(
        evidence_sufficiency_payload.get("mos_assessment_status") or UNKNOWN
    ).upper()
    evidence_class = str(
        evidence_sufficiency_payload.get("evidence_sufficiency_class") or UNKNOWN
    ).upper()

    support_signals: list[str] = []
    headwind_signals: list[str] = []
    stress_reason_codes: list[str] = []
    refinancing_reason_codes: list[str] = []

    if _is_num(net_debt_value) and float(net_debt_value) <= 0.0:
        support_signals.extend([SIG_NET_CASH_OR_LOW_NET_DEBT, SIG_LIQUIDITY_SUPPORT_PRESENT])
        stress_reason_codes.append(REASON_STRONG_LIQUIDITY_SUPPORT)
        refinancing_reason_codes.extend(
            [REASON_STRONG_LIQUIDITY_SUPPORT, REASON_LOW_NET_DEBT_TO_CFO]
        )
    if cfo_cannot_service_net_debt:
        headwind_signals.extend(
            [SIG_NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW, SIG_HIGH_NET_DEBT_TO_CFO]
        )
        stress_reason_codes.extend(
            [REASON_NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW, REASON_HIGH_NET_DEBT_TO_CFO]
        )
        refinancing_reason_codes.extend(
            [REASON_NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW, REASON_HIGH_NET_DEBT_TO_CFO]
        )
    elif _is_num(cfo_ratio_value) and float(cfo_ratio_value) <= 1.5:
        support_signals.append(SIG_LOW_NET_DEBT_TO_CFO)
        stress_reason_codes.append(REASON_LOW_NET_DEBT_TO_CFO)
        refinancing_reason_codes.append(REASON_LOW_NET_DEBT_TO_CFO)
        if float(cfo_ratio_value) <= 1.0:
            support_signals.append(SIG_NET_CASH_OR_LOW_NET_DEBT)
    elif _is_num(cfo_ratio_value) and float(cfo_ratio_value) > 3.5:
        headwind_signals.append(SIG_HIGH_NET_DEBT_TO_CFO)
        stress_reason_codes.append(REASON_HIGH_NET_DEBT_TO_CFO)
        refinancing_reason_codes.append(REASON_HIGH_NET_DEBT_TO_CFO)
    elif _is_num(cfo_ratio_value):
        stress_reason_codes.append(REASON_MODERATE_LEVERAGE)
        refinancing_reason_codes.append(REASON_MODERATE_LEVERAGE)

    if owner_cannot_service_net_debt or (
        _is_num(owner_ratio_value) and float(owner_ratio_value) > 4.0
    ):
        headwind_signals.append(SIG_HIGH_LEVERAGE_TO_OWNER_EARNINGS)
        stress_reason_codes.append(SIG_HIGH_LEVERAGE_TO_OWNER_EARNINGS)

    if _is_num(cash_to_debt) and float(cash_to_debt) >= 0.50:
        support_signals.extend([SIG_STRONG_CASH_TO_DEBT, SIG_LIQUIDITY_SUPPORT_PRESENT])
        stress_reason_codes.append(REASON_STRONG_LIQUIDITY_SUPPORT)
        refinancing_reason_codes.append(REASON_STRONG_LIQUIDITY_SUPPORT)
    elif _is_num(cash_to_debt) and float(cash_to_debt) < 0.15:
        headwind_signals.append(SIG_WEAK_CASH_CUSHION)
        stress_reason_codes.append(REASON_WEAK_CASH_TO_DEBT)
        refinancing_reason_codes.append(REASON_WEAK_CASH_TO_DEBT)

    if (
        _is_num(optionality_score)
        and float(optionality_score) >= 4.0
    ) or "BALANCE_SHEET_OPTIONALITY_STRONG" in optionality_reason_codes:
        support_signals.append(SIG_BALANCE_SHEET_OPTIONALITY_PRESENT)
        stress_reason_codes.append(REASON_BALANCE_SHEET_OPTIONALITY_STRONG)
        refinancing_reason_codes.append(REASON_BALANCE_SHEET_OPTIONALITY_STRONG)
    elif (
        _is_num(optionality_score)
        and float(optionality_score) <= 2.0
    ) or "BALANCE_SHEET_OPTIONALITY_UNKNOWN" in optionality_reason_codes:
        stress_reason_codes.append(REASON_BALANCE_SHEET_OPTIONALITY_WEAK)
        refinancing_reason_codes.append(REASON_BALANCE_SHEET_OPTIONALITY_WEAK)

    refinancing_dependence = (
        (SIG_HIGH_NET_DEBT_TO_CFO in headwind_signals and SIG_WEAK_CASH_CUSHION in headwind_signals)
        or (
            SIG_HIGH_NET_DEBT_TO_CFO in headwind_signals
            and REASON_BALANCE_SHEET_OPTIONALITY_WEAK in stress_reason_codes
        )
        or (
            _is_num(total_debt_value)
            and float(total_debt_value) > 0.0
            and not _is_num(cash_to_debt)
            and SIG_HIGH_NET_DEBT_TO_CFO in headwind_signals
        )
    )
    if refinancing_dependence:
        headwind_signals.append(SIG_REFINANCING_DEPENDENCE_HEADWIND)
        stress_reason_codes.append(REASON_REFINANCING_DEPENDENCE_POSSIBLE)
        refinancing_reason_codes.append(REASON_REFINANCING_DEPENDENCE_POSSIBLE)

    if refinancing_dependence and (
        "EXCESS_DILUTION" in owner_reason_codes
        or REASON_BALANCE_SHEET_OPTIONALITY_WEAK in stress_reason_codes
        or SIG_HIGH_LEVERAGE_TO_OWNER_EARNINGS in headwind_signals
    ):
        headwind_signals.append(SIG_POSSIBLE_DILUTION_RISK)

    capital_structure_dominates = (
        refinancing_dependence
        and (
            SIG_HIGH_LEVERAGE_TO_OWNER_EARNINGS in headwind_signals
            or mos_assessment_status == "MOS_CONFIRMED_ABSENT"
            or evidence_class in {"PARTIAL_FOR_MOS", "SUFFICIENT_FOR_MOS"}
        )
    )
    if capital_structure_dominates:
        headwind_signals.append(SIG_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE)
        stress_reason_codes.append(REASON_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE)
        refinancing_reason_codes.append(REASON_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE)

    if (
        SIG_HIGH_NET_DEBT_TO_CFO not in headwind_signals
        and SIG_HIGH_LEVERAGE_TO_OWNER_EARNINGS not in headwind_signals
        and SIG_WEAK_CASH_CUSHION not in headwind_signals
        and (
            SIG_LOW_NET_DEBT_TO_CFO in support_signals
            or SIG_STRONG_CASH_TO_DEBT in support_signals
            or SIG_NET_CASH_OR_LOW_NET_DEBT in support_signals
        )
    ):
        support_signals.append(SIG_CAPITAL_STRUCTURE_NOT_DOMINANT)

    evidence_known = any(
        _is_num(value)
        for value in [
            net_debt_value,
            total_debt_value,
            cash_value,
            cfo_ratio_value,
            owner_ratio_value,
            cash_to_debt,
        ]
    )
    evidence_blocked = (
        str(facts_status or UNKNOWN).upper() != OK
        or (not _is_num(net_debt_value) and not _is_num(total_debt_value))
    )
    if not evidence_known:
        headwind_signals.append(SIG_DEBT_INPUTS_THIN)
        stress_reason_codes.extend([REASON_INSUFFICIENT_DEBT_INPUTS, REASON_BALANCE_SHEET_EVIDENCE_THIN])
        refinancing_reason_codes.extend([REASON_INSUFFICIENT_DEBT_INPUTS, REASON_BALANCE_SHEET_EVIDENCE_THIN])
        stress_class = BALANCE_SHEET_STRESS_UNKNOWN
        refinancing_class = REFINANCING_RISK_UNKNOWN
        caution = BALANCE_SHEET_UNCLEAR
    else:
        support_strength = 0
        headwind_strength = 0
        if SIG_NET_CASH_OR_LOW_NET_DEBT in support_signals:
            support_strength += 2
        if SIG_LOW_NET_DEBT_TO_CFO in support_signals:
            support_strength += 2
        if SIG_STRONG_CASH_TO_DEBT in support_signals:
            support_strength += 2
        if SIG_BALANCE_SHEET_OPTIONALITY_PRESENT in support_signals:
            support_strength += 1
        if SIG_LIQUIDITY_SUPPORT_PRESENT in support_signals:
            support_strength += 1
        if SIG_CAPITAL_STRUCTURE_NOT_DOMINANT in support_signals:
            support_strength += 1

        if SIG_HIGH_NET_DEBT_TO_CFO in headwind_signals:
            headwind_strength += 2
        if SIG_HIGH_LEVERAGE_TO_OWNER_EARNINGS in headwind_signals:
            headwind_strength += 1
        if SIG_WEAK_CASH_CUSHION in headwind_signals:
            headwind_strength += 2
        if SIG_REFINANCING_DEPENDENCE_HEADWIND in headwind_signals:
            headwind_strength += 1
        if SIG_POSSIBLE_DILUTION_RISK in headwind_signals:
            headwind_strength += 1
        if SIG_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE in headwind_signals:
            headwind_strength += 2

        if support_strength >= 4 and headwind_strength == 0:
            stress_class = LOW_BALANCE_SHEET_STRESS
            stress_reason_codes.append(REASON_LOW_BALANCE_SHEET_STRESS_SUPPORT)
        elif (
            SIG_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE in headwind_signals
            or SIG_NET_DEBT_WITH_NON_POSITIVE_CASH_FLOW in headwind_signals
            or headwind_strength >= 4
        ):
            stress_class = HIGH_BALANCE_SHEET_STRESS
            stress_reason_codes.append(REASON_HIGH_BALANCE_SHEET_STRESS_HEADWIND)
        else:
            stress_class = MODERATE_BALANCE_SHEET_STRESS

        if stress_class == LOW_BALANCE_SHEET_STRESS and not refinancing_dependence:
            refinancing_class = LOW_REFINANCING_RISK
        elif refinancing_dependence or SIG_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE in headwind_signals:
            refinancing_class = HIGH_REFINANCING_RISK
            refinancing_reason_codes.append(REASON_HIGH_REFINANCING_RISK_HEADWIND)
        elif evidence_blocked and not support_signals and not headwind_signals:
            refinancing_class = REFINANCING_RISK_UNKNOWN
        else:
            refinancing_class = MODERATE_REFINANCING_RISK

        if stress_class == LOW_BALANCE_SHEET_STRESS and refinancing_class == LOW_REFINANCING_RISK:
            caution = BALANCE_SHEET_SUPPORTIVE
        elif stress_class == HIGH_BALANCE_SHEET_STRESS or refinancing_class == HIGH_REFINANCING_RISK:
            caution = BALANCE_SHEET_HEADWIND
        else:
            caution = BALANCE_SHEET_MIXED

    if evidence_blocked:
        # Blocked evidence on an input this module reads withholds the class whatever
        # fired (it used to apply only when no signal fired). Signals stay listed.
        stress_class = BALANCE_SHEET_STRESS_UNKNOWN
        refinancing_class = REFINANCING_RISK_UNKNOWN
        caution = BALANCE_SHEET_UNCLEAR
    if evidence_blocked and REASON_BALANCE_SHEET_EVIDENCE_THIN not in stress_reason_codes:
        stress_reason_codes.append(REASON_BALANCE_SHEET_EVIDENCE_THIN)
        refinancing_reason_codes.append(REASON_BALANCE_SHEET_EVIDENCE_THIN)

    support_signals = _dedupe(support_signals)
    headwind_signals = _dedupe(headwind_signals)
    stress_reason_codes = _dedupe(stress_reason_codes)
    refinancing_reason_codes = _dedupe(refinancing_reason_codes)
    derived_from = _dedupe(
        _collect_derived_from(
            fundamentals,
            owner_quality_payload,
            intangible_payload,
            evidence_sufficiency_payload,
            net_debt_payload,
            row_refs=row_derived_from,
        )
        + debt_refs
        + cfo_ratio_refs
        + owner_ratio_refs
        + cash_to_debt_refs
    )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "net_debt_proxy": _to_num(net_debt_value),
        "total_debt": _to_num(total_debt_value),
        "cash_equivalents": _to_num(cash_value),
        "net_debt_to_cfo": _to_num(cfo_ratio_value),
        "net_debt_to_owner_earnings": _to_num(owner_ratio_value),
        "cash_to_debt": _to_num(cash_to_debt),
        "balance_sheet_optionality_score": _to_num(optionality_score),
        "balance_sheet_stress_class": stress_class,
        "balance_sheet_stress_reason_codes": stress_reason_codes,
        "refinancing_risk_class": refinancing_class,
        "refinancing_risk_reason_codes": refinancing_reason_codes,
        "balance_sheet_support_signals": support_signals,
        "balance_sheet_headwind_signals": headwind_signals,
        "primary_balance_sheet_caution": caution,
        "balance_sheet_discipline_summary": _summary_for_profile(
            stress_class=stress_class,
            refinancing_class=refinancing_class,
            support_signals=support_signals,
            headwind_signals=headwind_signals,
        ),
        "derived_from": derived_from,
        "claims": {
            "balance_sheet_stress_class": {
                "value": stress_class,
                "status": OK if stress_class != BALANCE_SHEET_STRESS_UNKNOWN else UNKNOWN,
                "reason_code": stress_reason_codes[0] if stress_reason_codes else UNKNOWN,
                "derived_from": derived_from,
            },
            "refinancing_risk_class": {
                "value": refinancing_class,
                "status": OK if refinancing_class != REFINANCING_RISK_UNKNOWN else UNKNOWN,
                "reason_code": refinancing_reason_codes[0] if refinancing_reason_codes else UNKNOWN,
                "derived_from": derived_from,
            },
            "primary_balance_sheet_caution": {
                "value": caution,
                "status": OK if caution != BALANCE_SHEET_UNCLEAR else UNKNOWN,
                "reason_code": (
                    stress_reason_codes[0]
                    if stress_reason_codes
                    else (refinancing_reason_codes[0] if refinancing_reason_codes else UNKNOWN)
                ),
                "derived_from": derived_from,
            },
        },
        "generated_at": utc_now_iso(),
    }


def write_balance_sheet_stress_for_run(
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
        str(row.get("ticker") or "").upper(): row.get("balance_sheet_stress_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("balance_sheet_stress_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    stress_counts: dict[str, int] = {}
    refinancing_counts: dict[str, int] = {}
    caution_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

    for ticker in sorted({str(value or "").strip().upper() for value in tickers if str(value or "").strip()}):
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
            detail = compute_balance_sheet_stress(
                ticker=ticker,
                as_of_date=as_of_date,
                fundamentals=(
                    score_row.get("fundamentals_detail")
                    if isinstance(score_row.get("fundamentals_detail"), dict)
                    else {}
                ),
                owner_quality_payload=(
                    score_row.get("owner_earnings_quality_detail")
                    if isinstance(score_row.get("owner_earnings_quality_detail"), dict)
                    else {}
                ),
                intangible_payload=(
                    score_row.get("intangible_economics_detail")
                    if isinstance(score_row.get("intangible_economics_detail"), dict)
                    else {}
                ),
                evidence_sufficiency_payload=(
                    score_row.get("evidence_sufficiency_detail")
                    if isinstance(score_row.get("evidence_sufficiency_detail"), dict)
                    else {}
                ),
                net_debt_payload=(
                    score_row.get("net_debt_detail")
                    if isinstance(score_row.get("net_debt_detail"), dict)
                    else {}
                ),
                net_debt_proxy=score_row.get("net_debt_proxy", UNKNOWN),
                total_debt=score_row.get("total_debt", UNKNOWN),
                cash_equivalents=score_row.get("cash_equivalents", UNKNOWN),
                net_debt_to_cfo=score_row.get("net_debt_to_cfo", UNKNOWN),
                price_status=score_row.get("price_status", UNKNOWN),
                facts_status=score_row.get("facts_status", UNKNOWN),
                shares_status=score_row.get("shares_status", UNKNOWN),
                fcf_status=score_row.get("fcf_status", UNKNOWN),
                row_derived_from=list(score_row.get("derived_from") or []),
            )

        stress_class = str(detail.get("balance_sheet_stress_class") or BALANCE_SHEET_STRESS_UNKNOWN)
        refinancing_class = str(detail.get("refinancing_risk_class") or REFINANCING_RISK_UNKNOWN)
        caution = str(detail.get("primary_balance_sheet_caution") or BALANCE_SHEET_UNCLEAR)
        stress_counts[stress_class] = stress_counts.get(stress_class, 0) + 1
        refinancing_counts[refinancing_class] = refinancing_counts.get(refinancing_class, 0) + 1
        caution_counts[caution] = caution_counts.get(caution, 0) + 1
        for code in detail.get("balance_sheet_stress_reason_codes") or []:
            token = str(code or "").strip()
            if token:
                reason_counts[token] = reason_counts.get(token, 0) + 1
        rows.append(
            {
                "ticker": ticker,
                "balance_sheet_stress_class": stress_class,
                "balance_sheet_stress_reason_codes": [
                    str(code)
                    for code in (detail.get("balance_sheet_stress_reason_codes") or [])
                    if str(code).strip()
                ],
                "refinancing_risk_class": refinancing_class,
                "refinancing_risk_reason_codes": [
                    str(code)
                    for code in (detail.get("refinancing_risk_reason_codes") or [])
                    if str(code).strip()
                ],
                "balance_sheet_support_signals": [
                    str(code)
                    for code in (detail.get("balance_sheet_support_signals") or [])
                    if str(code).strip()
                ],
                "balance_sheet_headwind_signals": [
                    str(code)
                    for code in (detail.get("balance_sheet_headwind_signals") or [])
                    if str(code).strip()
                ],
                "primary_balance_sheet_caution": caution,
                "balance_sheet_discipline_summary": str(
                    detail.get("balance_sheet_discipline_summary") or ""
                ),
                "derived_from": [
                    str(ref) for ref in (detail.get("derived_from") or []) if str(ref).strip()
                ],
            }
        )

    def _class_rows(key: str, value: str) -> list[dict[str, Any]]:
        subset = [row for row in rows if str(row.get(key) or "") == value]
        subset.sort(key=lambda row: str(row.get("ticker") or ""))
        return subset

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "counts_by_balance_sheet_stress_class": dict(
            sorted(stress_counts.items(), key=lambda item: (_STRESS_ORDER.get(item[0], 99), item[0]))
        ),
        "counts_by_refinancing_risk_class": dict(
            sorted(refinancing_counts.items(), key=lambda item: (_REFINANCING_ORDER.get(item[0], 99), item[0]))
        ),
        "counts_by_primary_balance_sheet_caution": dict(
            sorted(caution_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ),
        "top_10_low_balance_sheet_stress": [
            {
                "ticker": str(row.get("ticker") or ""),
                "balance_sheet_stress_reason_codes": [
                    str(code)
                    for code in (row.get("balance_sheet_stress_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows("balance_sheet_stress_class", LOW_BALANCE_SHEET_STRESS)[:10]
        ],
        "top_10_high_balance_sheet_stress": [
            {
                "ticker": str(row.get("ticker") or ""),
                "balance_sheet_stress_reason_codes": [
                    str(code)
                    for code in (row.get("balance_sheet_stress_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows("balance_sheet_stress_class", HIGH_BALANCE_SHEET_STRESS)[:10]
        ],
        "top_10_high_refinancing_risk": [
            {
                "ticker": str(row.get("ticker") or ""),
                "refinancing_risk_reason_codes": [
                    str(code)
                    for code in (row.get("refinancing_risk_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in _class_rows("refinancing_risk_class", HIGH_REFINANCING_RISK)[:10]
        ],
        "most_common_balance_sheet_stress_reason_codes": [
            {"reason_code": str(name), "count": int(count)}
            for name, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ][:20],
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["balance_sheet_stress_path"] = str(output_path)
    return payload


def _balance_sheet_stress_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "balance_sheet_stress.json",
        cfg.sectors_dir / run_id / "balance_sheet_stress.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_balance_sheet_stress(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _balance_sheet_stress_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "balance_sheet_stress_path": str(path) if path is not None else "",
        }

    payload = _safe_json(path)
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_balance_sheet_stress_class": (
            payload.get("counts_by_balance_sheet_stress_class")
            if isinstance(payload.get("counts_by_balance_sheet_stress_class"), dict)
            else {}
        ),
        "counts_by_refinancing_risk_class": (
            payload.get("counts_by_refinancing_risk_class")
            if isinstance(payload.get("counts_by_refinancing_risk_class"), dict)
            else {}
        ),
        "counts_by_primary_balance_sheet_caution": (
            payload.get("counts_by_primary_balance_sheet_caution")
            if isinstance(payload.get("counts_by_primary_balance_sheet_caution"), dict)
            else {}
        ),
        "top_10_low_balance_sheet_stress": [
            row for row in (payload.get("top_10_low_balance_sheet_stress") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_high_balance_sheet_stress": [
            row for row in (payload.get("top_10_high_balance_sheet_stress") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_10_high_refinancing_risk": [
            row for row in (payload.get("top_10_high_refinancing_risk") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "most_common_balance_sheet_stress_reason_codes": [
            row
            for row in (payload.get("most_common_balance_sheet_stress_reason_codes") or [])
            if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "balance_sheet_stress_path": str(path),
    }
