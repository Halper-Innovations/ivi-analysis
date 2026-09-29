from __future__ import annotations

import json
from pathlib import Path
from statistics import median, pstdev
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.market.company_facts_extract import (
    CASH_EQUIVALENTS_TAG_PRIORITY,
    DEBT_CURRENT_TAG,
    FCF_DIRECT_TAG_PRIORITY,
    LONG_TERM_DEBT_NONCURRENT_TAG,
    SHARES_TAG_PRIORITY,
    TOTAL_DEBT_FALLBACK_TAG_PRIORITY,
)
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.owner_earnings import (
    DEFAULT_MAINT_CAPEX_RATIO,
    _annual_series_for_priority,
    _dedupe_refs,
    _load_companyfacts_payload,
    compute_owner_earnings_series,
)
from app.valuation.pre_valuation_gate import _DILUTION_WINDOW_YEARS
from app.valuation.share_splits import (
    REASON_SHARE_COUNT_BREAK_UNCORROBORATED,
    split_adjust_share_series,
    split_ratio_rows_from_companyfacts,
)
from app.valuation.maintenance_capex_discipline import (
    ASSET_INTENSITY_UNKNOWN,
    HIGH_ASSET_INTENSITY,
    LOW_ASSET_INTENSITY,
    LOW_MAINTENANCE_CAPEX_CREDIBILITY,
    MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN,
    REASON_HIGH_ASSET_INTENSITY_HEADWIND,
    REASON_LOW_ASSET_INTENSITY_SUPPORT,
    REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND,
    REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN,
    REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE,
)


UNKNOWN = "UNKNOWN"
OK = "OK"

REASON_INSUFFICIENT_OWNER_EARNINGS_HISTORY = "INSUFFICIENT_OWNER_EARNINGS_HISTORY"
REASON_NEGATIVE_OWNER_EARNINGS_SERIES = "NEGATIVE_OWNER_EARNINGS_SERIES"
REASON_OWNER_EARNINGS_UNKNOWN = "OWNER_EARNINGS_UNKNOWN"

REASON_EXCESS_DILUTION = "EXCESS_DILUTION"
REASON_HIGH_CAPEX_BURDEN = "HIGH_CAPEX_BURDEN"
REASON_DEBT_ACCUMULATION = "DEBT_ACCUMULATION"
REASON_SHAREHOLDER_FRIENDLY = "SHAREHOLDER_FRIENDLY"
REASON_UNKNOWN_CAPITAL_ALLOCATION = "UNKNOWN_CAPITAL_ALLOCATION"

REASON_MISSING_REVENUE = "MISSING_REVENUE"
REASON_MISSING_CFO = "MISSING_CFO"
REASON_MISSING_FCF = "MISSING_FCF"
REASON_CASH_CONVERSION_UNKNOWN = "CASH_CONVERSION_UNKNOWN"

POSITIVE_REASON_CODES = {REASON_SHAREHOLDER_FRIENDLY}

_CFO_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]
_CAPEX_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ("us-gaap", "CapitalExpendituresIncurredButNotYetPaid"),
]
_REVENUE_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    ("us-gaap", "SalesRevenueNet"),
    ("us-gaap", "SalesRevenueGoodsNet"),
    ("us-gaap", "Revenues"),
]


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


def _to_num(value: Any) -> float | int | str:
    if isinstance(value, bool):
        return UNKNOWN
    return float(value) if _is_num(value) else UNKNOWN


def _claim(*, value: Any, refs: list[Any], reason_code: str) -> dict[str, Any]:
    derived = _dedupe_refs([str(ref) for ref in refs if str(ref).strip()])
    if _is_num(value):
        return {
            "value": float(value),
            "status": OK,
            "reason_code": OK,
            "derived_from": derived,
        }
    return {
        "value": UNKNOWN,
        "status": UNKNOWN,
        "reason_code": str(reason_code or UNKNOWN),
        "derived_from": derived,
    }


def _cagr(start_value: Any, end_value: Any, periods: int) -> float | str:
    if not _is_num(start_value) or not _is_num(end_value):
        return UNKNOWN
    if float(start_value) <= 0 or float(end_value) <= 0 or int(periods) <= 0:
        return UNKNOWN
    return (float(end_value) / float(start_value)) ** (1.0 / float(periods)) - 1.0


def _series_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        value = row.get("value", row.get("owner_earnings", UNKNOWN))
        refs = [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()]
        if not refs and str(row.get("ref") or "").strip():
            refs = [str(row.get("ref"))]
        out[year] = {
            "year": year,
            "value": float(value) if _is_num(value) else UNKNOWN,
            "derived_from": refs,
        }
    return out


def _owner_series_from_owner_payload(owner_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in [row for row in (owner_payload.get("series") or []) if isinstance(row, dict)]:
        owner_value = raw.get("owner_earnings", UNKNOWN)
        refs = [str(ref) for ref in (raw.get("derived_from") or []) if str(ref).strip()]
        if not refs:
            refs = [str(ref) for ref in (owner_payload.get("derived_from") or []) if str(ref).strip()]
        rows.append(
            {
                "year": int(raw.get("year") or 0),
                "value": float(owner_value) if _is_num(owner_value) else UNKNOWN,
                "derived_from": refs,
            }
        )
    return [row for row in rows if int(row.get("year") or 0) > 0]


def _series_from_companyfacts(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
    expected_unit_exact: tuple[str, ...],
) -> list[dict[str, Any]]:
    series = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=priority,
        expected_unit_exact=expected_unit_exact,
    )
    return [
        {
            "year": int(row.get("year") or 0),
            "value": float(row.get("value")) if _is_num(row.get("value")) else UNKNOWN,
            "derived_from": [str(row.get("ref"))] if str(row.get("ref") or "").strip() else [],
        }
        for row in series
        if int(row.get("year") or 0) > 0
    ]


def _net_debt_series_from_companyfacts(*, companyfacts: dict[str, Any], as_of_date: str) -> list[dict[str, Any]]:
    current = _series_map(
        _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=[DEBT_CURRENT_TAG],
            expected_unit_exact=("usd",),
        )
    )
    noncurrent = _series_map(
        _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=[LONG_TERM_DEBT_NONCURRENT_TAG],
            expected_unit_exact=("usd",),
        )
    )
    fallback = _series_map(
        _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=TOTAL_DEBT_FALLBACK_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
    )
    cash = _series_map(
        _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=CASH_EQUIVALENTS_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
    )

    years = sorted(set(current) | set(noncurrent) | set(fallback) | set(cash))
    rows: list[dict[str, Any]] = []
    for year in years:
        debt_value = UNKNOWN
        debt_refs: list[str] = []
        current_row = current.get(year)
        noncurrent_row = noncurrent.get(year)
        fallback_row = fallback.get(year)
        if (
            isinstance(current_row, dict)
            and isinstance(noncurrent_row, dict)
            and _is_num(current_row.get("value"))
            and _is_num(noncurrent_row.get("value"))
        ):
            debt_value = float(current_row["value"]) + float(noncurrent_row["value"])
            debt_refs = _dedupe_refs(
                list(current_row.get("derived_from") or []) + list(noncurrent_row.get("derived_from") or [])
            )
        elif isinstance(fallback_row, dict) and _is_num(fallback_row.get("value")):
            debt_value = float(fallback_row["value"])
            debt_refs = [str(ref) for ref in (fallback_row.get("derived_from") or []) if str(ref).strip()]

        cash_row = cash.get(year)
        if not _is_num(debt_value) or not isinstance(cash_row, dict) or not _is_num(cash_row.get("value")):
            continue
        rows.append(
            {
                "year": year,
                "value": float(debt_value) - float(cash_row["value"]),
                "derived_from": _dedupe_refs(debt_refs + list(cash_row.get("derived_from") or [])),
            }
        )
    return rows


def _fundamentals_rows(fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    return sorted(rows, key=lambda row: int(row.get("year") or 0))


def _fundamentals_trace_map(fundamentals: dict[str, Any]) -> dict[str, Any]:
    return fundamentals.get("row_traces") if isinstance(fundamentals.get("row_traces"), dict) else {}


def _fundamentals_signal_map(fundamentals: dict[str, Any]) -> dict[str, Any]:
    return fundamentals.get("derived_signals") if isinstance(fundamentals.get("derived_signals"), dict) else {}


def _trace_refs(trace_map: dict[str, Any], year: int, metric: str, fallback: str) -> list[str]:
    year_bucket = trace_map.get(str(year)) if isinstance(trace_map, dict) else {}
    metric_bucket = year_bucket.get(metric) if isinstance(year_bucket, dict) else {}
    if isinstance(metric_bucket, dict):
        refs = [str(ref) for ref in (metric_bucket.get("derived_from") or []) if str(ref).strip()]
        if refs:
            return refs
    return [fallback]


def _signal_refs(signal_map: dict[str, Any], key: str, fallback: str) -> list[str]:
    signal = signal_map.get(key) if isinstance(signal_map, dict) else {}
    if isinstance(signal, dict):
        refs = [str(ref) for ref in (signal.get("derived_from") or []) if str(ref).strip()]
        if refs:
            return refs
    return [fallback]


def _build_series_from_fundamentals(fundamentals: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = _fundamentals_rows(fundamentals)
    traces = _fundamentals_trace_map(fundamentals)
    signals = _fundamentals_signal_map(fundamentals)

    owner_rows: list[dict[str, Any]] = []
    revenue_rows: list[dict[str, Any]] = []
    cfo_rows: list[dict[str, Any]] = []
    capex_rows: list[dict[str, Any]] = []
    fcf_rows: list[dict[str, Any]] = []
    shares_rows: list[dict[str, Any]] = []
    net_debt_rows: list[dict[str, Any]] = []

    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        revenue = row.get("revenue", UNKNOWN)
        cfo = row.get("cfo", UNKNOWN)
        capex = row.get("capex", UNKNOWN)
        fcf = row.get("fcf", UNKNOWN)
        shares = row.get("shares_outstanding", UNKNOWN)
        net_debt = row.get("net_debt", UNKNOWN)
        if _is_num(cfo) and _is_num(capex):
            owner_rows.append(
                {
                    "year": year,
                    "value": float(cfo) - (float(DEFAULT_MAINT_CAPEX_RATIO) * float(capex)),
                    "derived_from": _dedupe_refs(
                        _trace_refs(traces, year, "cfo", f"fundamentals.rows[{year}].cfo")
                        + _trace_refs(traces, year, "capex", f"fundamentals.rows[{year}].capex")
                    ),
                }
            )
        if _is_num(revenue):
            revenue_rows.append(
                {
                    "year": year,
                    "value": float(revenue),
                    "derived_from": _trace_refs(traces, year, "revenue", f"fundamentals.rows[{year}].revenue"),
                }
            )
        if _is_num(cfo):
            cfo_rows.append(
                {
                    "year": year,
                    "value": float(cfo),
                    "derived_from": _trace_refs(traces, year, "cfo", f"fundamentals.rows[{year}].cfo"),
                }
            )
        if _is_num(capex):
            capex_rows.append(
                {
                    "year": year,
                    "value": float(capex),
                    "derived_from": _trace_refs(traces, year, "capex", f"fundamentals.rows[{year}].capex"),
                }
            )
        if _is_num(fcf):
            fcf_rows.append(
                {
                    "year": year,
                    "value": float(fcf),
                    "derived_from": _trace_refs(traces, year, "fcf", f"fundamentals.rows[{year}].fcf"),
                }
            )
        if _is_num(shares):
            shares_rows.append(
                {
                    "year": year,
                    "value": float(shares),
                    "derived_from": _trace_refs(
                        traces,
                        year,
                        "shares_outstanding",
                        f"fundamentals.rows[{year}].shares_outstanding",
                    ),
                }
            )
        if _is_num(net_debt):
            net_debt_rows.append(
                {
                    "year": year,
                    "value": float(net_debt),
                    "derived_from": _trace_refs(traces, year, "net_debt", f"fundamentals.rows[{year}].net_debt"),
                }
            )

    if not shares_rows and isinstance(signals.get("dilution_rate_shares_cagr"), dict):
        latest_row = rows[-1] if rows else {}
        latest_year = int(latest_row.get("year") or 0)
        latest_shares = latest_row.get("shares_outstanding", UNKNOWN)
        if latest_year > 0 and _is_num(latest_shares):
            shares_rows.append(
                {
                    "year": latest_year,
                    "value": float(latest_shares),
                    "derived_from": _signal_refs(
                        signals,
                        "dilution_rate_shares_cagr",
                        "fundamentals.derived_signals.dilution_rate_shares_cagr",
                    ),
                }
            )

    return {
        "owner_earnings": owner_rows,
        "revenue": revenue_rows,
        "cfo": cfo_rows,
        "capex": capex_rows,
        "fcf": fcf_rows,
        "shares": shares_rows,
        "net_debt": net_debt_rows,
    }


def _latest_numeric_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    numeric = [row for row in rows if _is_num(row.get("value"))]
    return numeric[-1] if numeric else None


def _latest_common_pair(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    second_by_year = {int(row.get("year") or 0): row for row in second if int(row.get("year") or 0) > 0}
    pairs = [
        (row, second_by_year[int(row.get("year") or 0)])
        for row in first
        if int(row.get("year") or 0) in second_by_year and _is_num(row.get("value")) and _is_num(second_by_year[int(row.get("year") or 0)].get("value"))
    ]
    return pairs[-1] if pairs else (None, None)


def _recent_common_ratios(
    numerator: list[dict[str, Any]],
    denominator: list[dict[str, Any]],
    *,
    require_positive_denominator: bool = False,
    window: int = 3,
) -> list[tuple[float, list[str]]]:
    denominator_by_year = {int(row.get("year") or 0): row for row in denominator if int(row.get("year") or 0) > 0}
    ratios: list[tuple[int, float, list[str]]] = []
    for row in numerator:
        year = int(row.get("year") or 0)
        denom = denominator_by_year.get(year)
        if year <= 0 or not isinstance(denom, dict):
            continue
        num_value = row.get("value")
        denom_value = denom.get("value")
        if not _is_num(num_value) or not _is_num(denom_value):
            continue
        if float(denom_value) == 0.0:
            continue
        if require_positive_denominator and float(denom_value) <= 0.0:
            continue
        ratios.append(
            (
                year,
                float(num_value) / float(denom_value),
                _dedupe_refs(list(row.get("derived_from") or []) + list(denom.get("derived_from") or [])),
            )
        )
    # The newest ``window`` FISCAL YEARS, calendar-adjacent: a gap shrinks the
    # sample instead of pulling an older year in.
    ratios = sorted(ratios, key=lambda item: item[0])
    if ratios:
        floor_year = ratios[-1][0] - (max(1, int(window)) - 1)
        ratios = [item for item in ratios if item[0] >= floor_year]
    return [(value, refs) for _year, value, refs in ratios]


def _recent_dilution_rate(
    rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]] | None = None,
) -> tuple[float | str, list[str], list[int], list[int]]:
    """Annual change in the share count over a recent window, split-aware.

    The window is the newest ``_DILUTION_WINDOW_YEARS + 1`` fiscal years with a
    positive count (three intervals, the pre-valuation gate's window). A
    year-over-year move outside the gate's [2/3, 3/2] guard, or within 3% of a
    clean split factor, is a BREAK. A break that a filed split ratio
    (``split_rows``) corroborates is a split: earlier counts are multiplied by
    the filed ratio and the rate is measured across the whole window. Any other
    break — a 50% equity raise reads exactly like a 3-for-2 split — leaves the
    rate UNKNOWN; before 2026-09-29 every break was taken for a split and the
    rate measured after it, so a heavy diluter read as flat. Returns the rate,
    its source refs, the uncorroborated break years and the split years.
    """
    numeric = sorted(
        (
            row
            for row in rows
            if _is_num(row.get("value"))
            and not isinstance(row.get("value"), bool)
            and float(row["value"]) > 0
            and int(row.get("year") or 0) > 0
        ),
        key=lambda row: int(row.get("year") or 0),
    )[-(_DILUTION_WINDOW_YEARS + 1) :]
    ends = [numeric[0], numeric[-1]] if len(numeric) >= 2 else numeric
    refs = _dedupe_refs([ref for row in ends for ref in list(row.get("derived_from") or [])])
    adjusted, split_years, breaks = split_adjust_share_series(
        [(int(row["year"]), float(row["value"])) for row in numeric], split_rows
    )
    if split_years:
        refs = _dedupe_refs(
            refs
            + [
                ref
                for split in split_rows or []
                for ref in list(split.get("derived_from") or [])
            ]
        )
    if breaks or len(adjusted) < 2:
        return UNKNOWN, refs, breaks, split_years
    (start_year, start_value), (end_year, end_value) = adjusted[0], adjusted[-1]
    periods = max(1, int(end_year) - int(start_year))
    return _cagr(start_value, end_value, periods), refs, breaks, split_years


def _net_debt_change(rows: list[dict[str, Any]], *, window: int = 3) -> tuple[float | str, list[str]]:
    numeric = [row for row in rows if _is_num(row.get("value"))]
    numeric = sorted(numeric, key=lambda row: int(row.get("year") or 0))[-max(2, int(window)) :]
    if len(numeric) < 2:
        return UNKNOWN, _dedupe_refs([ref for row in numeric for ref in list(row.get("derived_from") or [])])
    start = numeric[0]
    end = numeric[-1]
    denom = max(abs(float(start.get("value") or 0.0)), abs(float(end.get("value") or 0.0)), 1.0)
    return (
        (float(end["value"]) - float(start["value"])) / denom,
        _dedupe_refs(list(start.get("derived_from") or []) + list(end.get("derived_from") or [])),
    )


def _owner_stability(series: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = sorted(
        [row for row in series if _is_num(row.get("value"))],
        key=lambda row: int(row.get("year") or 0),
    )[-5:]
    refs = _dedupe_refs([ref for row in numeric for ref in list(row.get("derived_from") or [])])
    if not numeric:
        reason_codes = [REASON_OWNER_EARNINGS_UNKNOWN]
        return {
            "owner_earnings_positive_years_5y": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_OWNER_EARNINGS_UNKNOWN),
            "owner_earnings_volatility_5y": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_OWNER_EARNINGS_UNKNOWN),
            "owner_earnings_stability_score": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_OWNER_EARNINGS_UNKNOWN),
            "reason_codes": reason_codes,
        }

    values = [float(row["value"]) for row in numeric]
    positive_years = len([value for value in values if value > 0.0])
    positive_claim = _claim(value=positive_years, refs=refs, reason_code=OK)
    volatility = UNKNOWN
    if len(values) >= 2:
        mean_abs = max(abs(sum(values) / float(len(values))), 1.0)
        volatility = pstdev(values) / mean_abs
    volatility_reason = OK if _is_num(volatility) else REASON_INSUFFICIENT_OWNER_EARNINGS_HISTORY
    volatility_claim = _claim(value=volatility, refs=refs, reason_code=volatility_reason)

    reason_codes: list[str] = []
    score: float | str = UNKNOWN
    if len(values) < 3:
        reason_codes.append(REASON_INSUFFICIENT_OWNER_EARNINGS_HISTORY)
    else:
        if positive_years == 0:
            reason_codes.append(REASON_NEGATIVE_OWNER_EARNINGS_SERIES)
        if _is_num(volatility):
            vol = float(volatility)
            if positive_years >= 5 and vol <= 0.30:
                score = 5.0
            elif positive_years >= 4 and vol <= 0.50:
                score = 4.0
            elif positive_years >= 3 and vol <= 0.75:
                score = 3.0
            elif positive_years >= 2 and vol <= 1.00:
                score = 2.0
            elif positive_years >= 1:
                score = 1.0
            else:
                score = 0.0
        else:
            reason_codes.append(REASON_INSUFFICIENT_OWNER_EARNINGS_HISTORY)
    if not reason_codes and not _is_num(score):
        reason_codes.append(REASON_OWNER_EARNINGS_UNKNOWN)

    stability_reason = reason_codes[0] if reason_codes else OK
    return {
        "owner_earnings_positive_years_5y": positive_claim,
        "owner_earnings_volatility_5y": volatility_claim,
        "owner_earnings_stability_score": _claim(value=score, refs=refs, reason_code=stability_reason),
        "reason_codes": _dedupe_refs(reason_codes),
    }


def _capital_allocation(series_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    dilution_rate, dilution_refs, dilution_breaks, dilution_splits = _recent_dilution_rate(
        series_map.get("shares") or [], series_map.get("share_splits") or []
    )
    dilution_unknown_reason = (
        REASON_SHARE_COUNT_BREAK_UNCORROBORATED
        if dilution_breaks
        else REASON_UNKNOWN_CAPITAL_ALLOCATION
    )
    capex_burden_rows = _recent_common_ratios(
        series_map.get("capex") or [],
        series_map.get("cfo") or [],
        require_positive_denominator=True,
        window=3,
    )
    capex_burden = median([value for value, _refs in capex_burden_rows]) if capex_burden_rows else UNKNOWN
    capex_refs = _dedupe_refs([ref for _value, refs in capex_burden_rows for ref in refs])
    net_debt_change_proxy, debt_refs = _net_debt_change(series_map.get("net_debt") or [], window=3)

    reason_codes: list[str] = []
    known = any(_is_num(value) for value in [dilution_rate, capex_burden, net_debt_change_proxy])
    if not known:
        reason_codes.append(REASON_UNKNOWN_CAPITAL_ALLOCATION)
        return {
            "dilution_rate_shares_cagr": _claim(value=UNKNOWN, refs=dilution_refs, reason_code=dilution_unknown_reason),
            "capex_burden_vs_cfo": _claim(value=UNKNOWN, refs=capex_refs, reason_code=REASON_UNKNOWN_CAPITAL_ALLOCATION),
            "net_debt_change_proxy_3y": _claim(value=UNKNOWN, refs=debt_refs, reason_code=REASON_UNKNOWN_CAPITAL_ALLOCATION),
            "capital_allocation_score": _claim(value=UNKNOWN, refs=dilution_refs + capex_refs + debt_refs, reason_code=REASON_UNKNOWN_CAPITAL_ALLOCATION),
            "dilution_share_count_breaks": dilution_breaks,
            "dilution_share_count_splits": dilution_splits,
            "reason_codes": reason_codes,
        }

    score = 2.0
    if _is_num(dilution_rate):
        dilution = float(dilution_rate)
        if dilution <= -0.01:
            score += 2.0
        elif dilution <= 0.02:
            score += 1.0
        elif dilution > 0.06:
            score -= 2.0
            reason_codes.append(REASON_EXCESS_DILUTION)

    if _is_num(capex_burden):
        burden = float(capex_burden)
        if burden <= 0.25:
            score += 2.0
        elif burden <= 0.45:
            score += 1.0
        elif burden > 0.60:
            score -= 1.0
            reason_codes.append(REASON_HIGH_CAPEX_BURDEN)

    if _is_num(net_debt_change_proxy):
        debt_change = float(net_debt_change_proxy)
        if debt_change <= -0.10:
            score += 1.0
        elif debt_change > 0.15:
            score -= 1.0
            reason_codes.append(REASON_DEBT_ACCUMULATION)

    score = max(0.0, min(5.0, round(score, 6)))
    if not reason_codes and score >= 4.0:
        reason_codes.append(REASON_SHAREHOLDER_FRIENDLY)
    score_reason = reason_codes[0] if reason_codes else OK
    return {
        "dilution_rate_shares_cagr": _claim(
            value=dilution_rate,
            refs=dilution_refs,
            reason_code=dilution_unknown_reason if not _is_num(dilution_rate) else OK,
        ),
        "capex_burden_vs_cfo": _claim(
            value=capex_burden,
            refs=capex_refs,
            reason_code=REASON_UNKNOWN_CAPITAL_ALLOCATION if not _is_num(capex_burden) else OK,
        ),
        "net_debt_change_proxy_3y": _claim(
            value=net_debt_change_proxy,
            refs=debt_refs,
            reason_code=REASON_UNKNOWN_CAPITAL_ALLOCATION if not _is_num(net_debt_change_proxy) else OK,
        ),
        "capital_allocation_score": _claim(
            value=score,
            refs=dilution_refs + capex_refs + debt_refs,
            reason_code=score_reason,
        ),
        "dilution_share_count_breaks": dilution_breaks,
        "dilution_share_count_splits": dilution_splits,
        "reason_codes": _dedupe_refs(reason_codes),
    }


def _cash_conversion(series_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    revenue_row, cfo_row = _latest_common_pair(series_map.get("revenue") or [], series_map.get("cfo") or [])
    fcf_row, cfo_for_fcf = _latest_common_pair(series_map.get("fcf") or [], series_map.get("cfo") or [])
    reason_codes: list[str] = []

    if not isinstance(revenue_row, dict):
        reason_codes.append(REASON_MISSING_REVENUE)
    if not isinstance(cfo_row, dict):
        reason_codes.append(REASON_MISSING_CFO)
    if not isinstance(fcf_row, dict):
        reason_codes.append(REASON_MISSING_FCF)

    if reason_codes:
        reason_codes.append(REASON_CASH_CONVERSION_UNKNOWN)
        refs = []
        if isinstance(revenue_row, dict):
            refs.extend(list(revenue_row.get("derived_from") or []))
        if isinstance(cfo_row, dict):
            refs.extend(list(cfo_row.get("derived_from") or []))
        if isinstance(fcf_row, dict):
            refs.extend(list(fcf_row.get("derived_from") or []))
        return {
            "cfo_margin_proxy": _claim(value=UNKNOWN, refs=refs, reason_code=reason_codes[0]),
            "fcf_conversion_proxy": _claim(value=UNKNOWN, refs=refs, reason_code=reason_codes[0]),
            "cash_conversion_score": _claim(value=UNKNOWN, refs=refs, reason_code=reason_codes[0]),
            "reason_codes": _dedupe_refs(reason_codes),
        }

    revenue_value = revenue_row.get("value", UNKNOWN) if isinstance(revenue_row, dict) else UNKNOWN
    cfo_value = cfo_row.get("value", UNKNOWN) if isinstance(cfo_row, dict) else UNKNOWN
    fcf_value = fcf_row.get("value", UNKNOWN) if isinstance(fcf_row, dict) else UNKNOWN
    cfo_margin = (
        float(cfo_value) / float(revenue_value)
        if _is_num(cfo_value) and _is_num(revenue_value) and float(revenue_value) != 0.0
        else UNKNOWN
    )
    fcf_conversion = (
        float(fcf_value) / float(cfo_for_fcf.get("value"))
        if isinstance(cfo_for_fcf, dict)
        and _is_num(fcf_value)
        and _is_num(cfo_for_fcf.get("value"))
        and float(cfo_for_fcf.get("value")) > 0.0
        else UNKNOWN
    )
    refs = _dedupe_refs(
        list(revenue_row.get("derived_from") or [])
        + list(cfo_row.get("derived_from") or [])
        + list(fcf_row.get("derived_from") or [])
        + (list(cfo_for_fcf.get("derived_from") or []) if isinstance(cfo_for_fcf, dict) else [])
    )

    score = 0.0
    if _is_num(cfo_margin):
        margin = float(cfo_margin)
        if margin >= 0.20:
            score += 2.0
        elif margin >= 0.10:
            score += 1.0
    if _is_num(fcf_conversion):
        conversion = float(fcf_conversion)
        if conversion >= 0.80:
            score += 3.0
        elif conversion >= 0.60:
            score += 2.0
        elif conversion >= 0.30:
            score += 1.0
    score = max(0.0, min(5.0, round(score, 6)))
    return {
        "cfo_margin_proxy": _claim(
            value=cfo_margin,
            refs=list(revenue_row.get("derived_from") or []) + list(cfo_row.get("derived_from") or []),
            reason_code=REASON_CASH_CONVERSION_UNKNOWN if not _is_num(cfo_margin) else OK,
        ),
        "fcf_conversion_proxy": _claim(
            value=fcf_conversion,
            refs=list(fcf_row.get("derived_from") or [])
            + (list(cfo_for_fcf.get("derived_from") or []) if isinstance(cfo_for_fcf, dict) else []),
            reason_code=REASON_CASH_CONVERSION_UNKNOWN if not _is_num(fcf_conversion) else OK,
        ),
        "cash_conversion_score": _claim(value=score, refs=refs, reason_code=OK),
        "reason_codes": [],
    }


def _build_owner_earnings_quality_payload(
    *,
    ticker: str,
    as_of_date: str,
    source_kind: str,
    series_map: dict[str, list[dict[str, Any]]],
    maintenance_capex_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    owner = _owner_stability(series_map.get("owner_earnings") or [])
    capital = _capital_allocation(series_map)
    cash = _cash_conversion(series_map)
    all_refs = _dedupe_refs(
        list(owner["owner_earnings_positive_years_5y"].get("derived_from") or [])
        + list(owner["owner_earnings_volatility_5y"].get("derived_from") or [])
        + list(owner["owner_earnings_stability_score"].get("derived_from") or [])
        + list(capital["dilution_rate_shares_cagr"].get("derived_from") or [])
        + list(capital["capex_burden_vs_cfo"].get("derived_from") or [])
        + list(capital["net_debt_change_proxy_3y"].get("derived_from") or [])
        + list(capital["capital_allocation_score"].get("derived_from") or [])
        + list(cash["cfo_margin_proxy"].get("derived_from") or [])
        + list(cash["fcf_conversion_proxy"].get("derived_from") or [])
        + list(cash["cash_conversion_score"].get("derived_from") or [])
    )

    component_scores = [
        owner["owner_earnings_stability_score"].get("value"),
        capital["capital_allocation_score"].get("value"),
        cash["cash_conversion_score"].get("value"),
    ]
    known_scores = [float(value) for value in component_scores if _is_num(value)]
    oe_quality_total = round(sum(known_scores), 6) if known_scores else UNKNOWN
    oe_quality_reason_codes = _dedupe_refs(
        list(owner.get("reason_codes") or [])
        + list(capital.get("reason_codes") or [])
        + list(cash.get("reason_codes") or [])
    )
    maintenance_capex_payload = (
        maintenance_capex_payload if isinstance(maintenance_capex_payload, dict) else {}
    )
    maintenance_capex_class = str(
        maintenance_capex_payload.get("maintenance_capex_credibility_class")
        or MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
    ).upper()
    asset_intensity_class = str(
        maintenance_capex_payload.get("asset_intensity_class") or ASSET_INTENSITY_UNKNOWN
    ).upper()
    maintenance_refs = [
        str(ref) for ref in (maintenance_capex_payload.get("derived_from") or []) if str(ref).strip()
    ]
    if maintenance_capex_class == LOW_MAINTENANCE_CAPEX_CREDIBILITY:
        oe_quality_reason_codes = _dedupe_refs(
            oe_quality_reason_codes
            + [
                REASON_LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND,
                REASON_OWNER_EARNINGS_DENOMINATOR_SENSITIVE,
            ]
            + ([REASON_HIGH_ASSET_INTENSITY_HEADWIND] if asset_intensity_class == HIGH_ASSET_INTENSITY else [])
        )
        if _is_num(oe_quality_total):
            oe_quality_total = max(0.0, round(float(oe_quality_total) - 1.5, 6))
    elif maintenance_capex_class == MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN:
        oe_quality_reason_codes = _dedupe_refs(
            oe_quality_reason_codes + [REASON_MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN]
        )
    elif maintenance_capex_class == "HIGH_MAINTENANCE_CAPEX_CREDIBILITY" and asset_intensity_class == LOW_ASSET_INTENSITY:
        oe_quality_reason_codes = _dedupe_refs(
            oe_quality_reason_codes + [REASON_LOW_ASSET_INTENSITY_SUPPORT]
        )

    claims = {
        "owner_earnings_positive_years_5y": owner["owner_earnings_positive_years_5y"],
        "owner_earnings_volatility_5y": owner["owner_earnings_volatility_5y"],
        "owner_earnings_stability_score": owner["owner_earnings_stability_score"],
        "dilution_rate_shares_cagr": capital["dilution_rate_shares_cagr"],
        "capex_burden_vs_cfo": capital["capex_burden_vs_cfo"],
        "net_debt_change_proxy_3y": capital["net_debt_change_proxy_3y"],
        "capital_allocation_score": capital["capital_allocation_score"],
        "cfo_margin_proxy": cash["cfo_margin_proxy"],
        "fcf_conversion_proxy": cash["fcf_conversion_proxy"],
        "cash_conversion_score": cash["cash_conversion_score"],
        "oe_quality_total": _claim(
            value=oe_quality_total,
            refs=all_refs + maintenance_refs,
            reason_code=oe_quality_reason_codes[0] if oe_quality_reason_codes else OK,
        ),
    }

    return {
        "ticker": str(ticker or "").strip().upper(),
        "as_of_date": str(as_of_date or ""),
        "source_kind": source_kind,
        "owner_earnings_positive_years_5y": claims["owner_earnings_positive_years_5y"].get("value", UNKNOWN),
        "owner_earnings_volatility_5y": claims["owner_earnings_volatility_5y"].get("value", UNKNOWN),
        "owner_earnings_stability_score": claims["owner_earnings_stability_score"].get("value", UNKNOWN),
        "owner_earnings_stability_reason_codes": list(owner.get("reason_codes") or []),
        "dilution_rate_shares_cagr": claims["dilution_rate_shares_cagr"].get("value", UNKNOWN),
        "capex_burden_vs_cfo": claims["capex_burden_vs_cfo"].get("value", UNKNOWN),
        "net_debt_change_proxy_3y": claims["net_debt_change_proxy_3y"].get("value", UNKNOWN),
        "capital_allocation_score": claims["capital_allocation_score"].get("value", UNKNOWN),
        "capital_allocation_reason_codes": list(capital.get("reason_codes") or []),
        "dilution_share_count_breaks": list(capital.get("dilution_share_count_breaks") or []),
        "dilution_share_count_splits": list(capital.get("dilution_share_count_splits") or []),
        "dilution_rate_reason_code": str(
            claims["dilution_rate_shares_cagr"].get("reason_code") or UNKNOWN
        ),
        "cfo_margin_proxy": claims["cfo_margin_proxy"].get("value", UNKNOWN),
        "fcf_conversion_proxy": claims["fcf_conversion_proxy"].get("value", UNKNOWN),
        "cash_conversion_score": claims["cash_conversion_score"].get("value", UNKNOWN),
        "cash_conversion_reason_codes": list(cash.get("reason_codes") or []),
        "oe_quality_total": claims["oe_quality_total"].get("value", UNKNOWN),
        "oe_quality_reason_codes": oe_quality_reason_codes,
        "claims": claims,
        "derived_from": _dedupe_refs(all_refs + maintenance_refs),
        "generated_at": utc_now_iso(),
    }


def compute_owner_earnings_quality(
    ticker: str,
    as_of_date: str,
    *,
    run_id: str | None = None,
    facts_row: dict[str, Any] | None = None,
    owner_payload: dict[str, Any] | None = None,
    fundamentals: dict[str, Any] | None = None,
    maintenance_capex_payload: dict[str, Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    ticker_norm = str(ticker or "").strip().upper()
    if isinstance(fundamentals, dict) and fundamentals:
        return _build_owner_earnings_quality_payload(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            source_kind="FUNDAMENTALS",
            series_map=_build_series_from_fundamentals(fundamentals),
            maintenance_capex_payload=maintenance_capex_payload,
        )

    cfg = cfg or get_config()
    base_facts_row = facts_row if isinstance(facts_row, dict) else resolve_financial_facts_asof(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        run_id=run_id,
        refresh=False,
        cfg=cfg,
    )
    owner = owner_payload if isinstance(owner_payload, dict) else compute_owner_earnings_series(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        years_back=5,
        run_id=run_id,
        maintenance_capex_ratio=DEFAULT_MAINT_CAPEX_RATIO,
        cfg=cfg,
    )
    companyfacts = _load_companyfacts_payload(base_facts_row)
    series_map = {
        "owner_earnings": _owner_series_from_owner_payload(owner),
        "revenue": [],
        "cfo": [],
        "capex": [],
        "fcf": [],
        "shares": [],
        "net_debt": [],
    }
    if companyfacts:
        series_map["revenue"] = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=_REVENUE_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
        series_map["cfo"] = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=_CFO_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
        series_map["capex"] = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=_CAPEX_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
        fcf_series = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=FCF_DIRECT_TAG_PRIORITY,
            expected_unit_exact=("usd",),
        )
        if fcf_series:
            series_map["fcf"] = fcf_series
        else:
            cfo_by_year = _series_map(series_map["cfo"])
            capex_by_year = _series_map(series_map["capex"])
            derived_fcf: list[dict[str, Any]] = []
            for year in sorted(set(cfo_by_year) & set(capex_by_year)):
                cfo_row = cfo_by_year.get(year)
                capex_row = capex_by_year.get(year)
                if not isinstance(cfo_row, dict) or not isinstance(capex_row, dict):
                    continue
                if not _is_num(cfo_row.get("value")) or not _is_num(capex_row.get("value")):
                    continue
                derived_fcf.append(
                    {
                        "year": year,
                        "value": float(cfo_row["value"]) - float(capex_row["value"]),
                        "derived_from": _dedupe_refs(
                            list(cfo_row.get("derived_from") or []) + list(capex_row.get("derived_from") or [])
                        ),
                    }
                )
            series_map["fcf"] = derived_fcf
        series_map["shares"] = _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=SHARES_TAG_PRIORITY,
            expected_unit_exact=("shares",),
        )
        series_map["net_debt"] = _net_debt_series_from_companyfacts(companyfacts=companyfacts, as_of_date=as_of_date)
        series_map["share_splits"] = split_ratio_rows_from_companyfacts(companyfacts, as_of_date)

    return _build_owner_earnings_quality_payload(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        source_kind="COMPANYFACTS",
        series_map=series_map,
        maintenance_capex_payload=maintenance_capex_payload,
    )


def write_owner_earnings_quality_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    facts_rows_by_ticker: dict[str, dict[str, Any]] | None = None,
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    maintenance_capex_by_ticker: dict[str, dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("owner_earnings_quality_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("owner_earnings_quality_detail"), dict)
    }
    rows: list[dict[str, Any]] = []
    for ticker in sorted({str(token or "").strip().upper() for token in tickers if str(token or "").strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            detail = compute_owner_earnings_quality(
                ticker,
                as_of_date,
                run_id=run_id,
                facts_row=(facts_rows_by_ticker or {}).get(ticker),
                fundamentals=(fundamentals_by_ticker or {}).get(ticker),
                maintenance_capex_payload=(maintenance_capex_by_ticker or {}).get(ticker),
                cfg=cfg,
            )
        rows.append(detail)

    known_count = len([row for row in rows if _is_num(row.get("oe_quality_total"))])
    unknown_count = len(rows) - known_count
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("oe_quality_total")) if _is_num(row.get("oe_quality_total")) else float("inf"),
            str(row.get("ticker") or ""),
        ),
    )
    negative_reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in [str(code) for code in (row.get("oe_quality_reason_codes") or []) if str(code).strip()]:
            if reason in POSITIVE_REASON_CODES:
                continue
            negative_reason_counts[reason] = negative_reason_counts.get(reason, 0) + 1
    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "known_count": known_count,
        "unknown_count": unknown_count,
        "top_10_by_oe_quality_total": [
            {
                "ticker": str(row.get("ticker") or ""),
                "oe_quality_total": row.get("oe_quality_total", UNKNOWN),
                "owner_earnings_stability_score": row.get("owner_earnings_stability_score", UNKNOWN),
                "capital_allocation_score": row.get("capital_allocation_score", UNKNOWN),
                "cash_conversion_score": row.get("cash_conversion_score", UNKNOWN),
                "oe_quality_reason_codes": [str(code) for code in (row.get("oe_quality_reason_codes") or []) if str(code).strip()],
            }
            for row in ranked[:10]
        ],
        "negative_reason_counts": dict(sorted(negative_reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["owner_earnings_quality_path"] = str(output_path)
    return payload


def _quality_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "owner_earnings_quality.json",
        cfg.sectors_dir / run_id / "owner_earnings_quality.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_owner_earnings_quality(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _quality_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "owner_earnings_quality_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    top_rows = [row for row in (payload.get("top_10_by_oe_quality_total") or []) if isinstance(row, dict)]
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or len(rows)),
        "known_count": int(payload.get("known_count") or 0),
        "unknown_count": int(payload.get("unknown_count") or 0),
        "top_10_by_oe_quality_total": top_rows[: max(1, int(top_n))],
        "negative_reason_counts": payload.get("negative_reason_counts") if isinstance(payload.get("negative_reason_counts"), dict) else {},
        "owner_earnings_quality_path": str(path),
    }
