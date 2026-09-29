from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.market.company_facts_extract import CASH_EQUIVALENTS_TAG_PRIORITY, FCF_DIRECT_TAG_PRIORITY, SHARES_TAG_PRIORITY
from app.valuation.adjacent_years import trailing_adjacent_run
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.owner_earnings import (
    DEFAULT_MAINT_CAPEX_RATIO,
    _annual_series_for_priority,
    _dedupe_refs,
    _load_companyfacts_payload,
    compute_owner_earnings_series,
)
from app.valuation.owner_earnings_quality import compute_owner_earnings_quality


UNKNOWN = "UNKNOWN"
OK = "OK"

REASON_MISSING_GROSS_MARGIN_HISTORY = "MISSING_GROSS_MARGIN_HISTORY"
REASON_INSUFFICIENT_GROSS_MARGIN_HISTORY = "INSUFFICIENT_GROSS_MARGIN_HISTORY"
REASON_VOLATILE_GROSS_MARGIN = "VOLATILE_GROSS_MARGIN"
REASON_GROSS_MARGIN_DURABILITY_UNKNOWN = "GROSS_MARGIN_DURABILITY_UNKNOWN"

REASON_NET_DEBT_PRESSURE = "NET_DEBT_PRESSURE"
REASON_WEAK_LIQUIDITY_SUPPORT = "WEAK_LIQUIDITY_SUPPORT"
REASON_BALANCE_SHEET_OPTIONALITY_STRONG = "BALANCE_SHEET_OPTIONALITY_STRONG"
REASON_BALANCE_SHEET_OPTIONALITY_UNKNOWN = "BALANCE_SHEET_OPTIONALITY_UNKNOWN"
REASON_MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS = "MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS"

REASON_HIGH_OWNER_EARNINGS_VOLATILITY = "HIGH_OWNER_EARNINGS_VOLATILITY"
REASON_HIGH_MARGIN_VOLATILITY = "HIGH_MARGIN_VOLATILITY"
REASON_RESILIENT_CYCLICAL_PROFILE = "RESILIENT_CYCLICAL_PROFILE"
REASON_CYCLE_RESILIENCE_UNKNOWN = "CYCLE_RESILIENCE_UNKNOWN"
REASON_INSUFFICIENT_CYCLE_HISTORY = "INSUFFICIENT_CYCLE_HISTORY"

REASON_MISSING_RND_HISTORY = "MISSING_RND_HISTORY"
REASON_MISSING_RND_DENOMINATOR = "MISSING_RND_DENOMINATOR"
REASON_INSUFFICIENT_RND_HISTORY = "INSUFFICIENT_RND_HISTORY"
REASON_LOW_RND_PRODUCTIVITY = "LOW_RND_PRODUCTIVITY"
REASON_RND_PRODUCTIVITY_UNKNOWN = "RND_PRODUCTIVITY_UNKNOWN"

REASON_MISSING_SGA_HISTORY = "MISSING_SGA_HISTORY"
REASON_MISSING_OPERATING_LEVERAGE_INPUTS = "MISSING_OPERATING_LEVERAGE_INPUTS"
REASON_INSUFFICIENT_SGA_HISTORY = "INSUFFICIENT_SGA_HISTORY"
REASON_WEAK_SGA_LEVERAGE = "WEAK_SGA_LEVERAGE"
REASON_SGA_LEVERAGE_UNKNOWN = "SGA_LEVERAGE_UNKNOWN"

REASON_EXCESS_DILUTION = "EXCESS_DILUTION"
REASON_WEAK_PER_SHARE_CAPTURE = "WEAK_PER_SHARE_CAPTURE"
REASON_STRONG_OWNER_VALUE_CAPTURE = "STRONG_OWNER_VALUE_CAPTURE"
REASON_OWNER_VALUE_CAPTURE_UNKNOWN = "OWNER_VALUE_CAPTURE_UNKNOWN"
REASON_MISSING_PER_SHARE_INPUTS = "MISSING_PER_SHARE_INPUTS"

POSITIVE_REASON_CODES = {
    REASON_BALANCE_SHEET_OPTIONALITY_STRONG,
    REASON_RESILIENT_CYCLICAL_PROFILE,
    REASON_STRONG_OWNER_VALUE_CAPTURE,
}

_REVENUE_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    ("us-gaap", "SalesRevenueNet"),
    ("us-gaap", "SalesRevenueGoodsNet"),
    ("us-gaap", "Revenues"),
]
_GROSS_PROFIT_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "GrossProfit"),
]
_OPERATING_INCOME_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "OperatingIncomeLoss"),
]
_CFO_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]
_CAPEX_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ("us-gaap", "CapitalExpendituresIncurredButNotYetPaid"),
]
_TOTAL_DEBT_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "Debt"),
    ("us-gaap", "DebtAndCapitalLeaseObligations"),
]
_RND_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "ResearchAndDevelopmentExpense"),
    ("us-gaap", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
    ("us-gaap", "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost"),
]
_SGA_DIRECT_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "SellingGeneralAndAdministrativeExpense"),
]
_SALES_MARKETING_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "SellingAndMarketingExpense"),
    ("us-gaap", "SalesAndMarketingExpense"),
]
_G_AND_A_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "GeneralAndAdministrativeExpense"),
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


def _to_num(value: Any) -> float | str:
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


def _series_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        out[year] = {
            "year": year,
            "value": float(row.get("value")) if _is_num(row.get("value")) else UNKNOWN,
            "derived_from": [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()],
        }
    return out


def _trace_refs(trace_map: dict[str, Any], year: int, metric: str, fallback: str) -> list[str]:
    year_bucket = trace_map.get(str(year)) if isinstance(trace_map, dict) else {}
    metric_bucket = year_bucket.get(metric) if isinstance(year_bucket, dict) else {}
    if isinstance(metric_bucket, dict):
        refs = [str(ref) for ref in (metric_bucket.get("derived_from") or []) if str(ref).strip()]
        if refs:
            return refs
    return [fallback]


def _owner_series_from_owner_payload(owner_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in [row for row in (owner_payload.get("series") or []) if isinstance(row, dict)]:
        year = int(raw.get("year") or 0)
        value = raw.get("owner_earnings", UNKNOWN)
        if year <= 0 or not _is_num(value):
            continue
        refs = [str(ref) for ref in (raw.get("derived_from") or []) if str(ref).strip()]
        if not refs:
            refs = [str(ref) for ref in (owner_payload.get("derived_from") or []) if str(ref).strip()]
        rows.append(
            {
                "year": year,
                "value": float(value),
                "derived_from": refs,
            }
        )
    return rows


def _latest_numeric_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    numeric = [row for row in rows if _is_num(row.get("value"))]
    return numeric[-1] if numeric else None


def _latest_common_pair(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    second_by_year = {int(row.get("year") or 0): row for row in second if int(row.get("year") or 0) > 0}
    pairs = [
        (row, second_by_year[int(row.get("year") or 0)])
        for row in first
        if int(row.get("year") or 0) in second_by_year
        and _is_num(row.get("value"))
        and _is_num(second_by_year[int(row.get("year") or 0)].get("value"))
    ]
    return pairs[-1] if pairs else (None, None)


def _recent_adjacent(rows: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
    """The newest ``n`` rows, but only the calendar-adjacent run ending at the latest year.

    A gap inside the window used to let an old year through as if it were recent.
    """
    ordered = sorted(rows, key=lambda row: int(row.get("year") or 0))
    return trailing_adjacent_run(ordered, lambda row: row.get("year"))[-n:]


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
    ratios = trailing_adjacent_run(sorted(ratios, key=lambda item: item[0]))[-max(1, int(window)) :]
    return [(value, refs) for _year, value, refs in ratios]


def _cagr(start_value: Any, end_value: Any, periods: int) -> float | str:
    if not _is_num(start_value) or not _is_num(end_value):
        return UNKNOWN
    if float(start_value) <= 0.0 or float(end_value) <= 0.0 or int(periods) <= 0:
        return UNKNOWN
    return (float(end_value) / float(start_value)) ** (1.0 / float(periods)) - 1.0


def _series_cagr(rows: list[dict[str, Any]]) -> tuple[float | str, list[str]]:
    # Keep loss/zero endpoints; _cagr must decide whether growth is defined.
    numeric = [row for row in rows if _is_num(row.get("value"))]
    if len(numeric) < 2:
        return UNKNOWN, _dedupe_refs([ref for row in numeric for ref in list(row.get("derived_from") or [])])
    start = numeric[0]
    end = numeric[-1]
    periods = max(1, int(end.get("year") or 0) - int(start.get("year") or 0))
    return (
        _cagr(start.get("value"), end.get("value"), periods),
        _dedupe_refs(list(start.get("derived_from") or []) + list(end.get("derived_from") or [])),
    )


def _ratio_series(
    numerator: list[dict[str, Any]],
    denominator: list[dict[str, Any]],
    *,
    require_positive_denominator: bool = False,
) -> list[dict[str, Any]]:
    denominator_by_year = {int(row.get("year") or 0): row for row in denominator if int(row.get("year") or 0) > 0}
    out: list[dict[str, Any]] = []
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
        out.append(
            {
                "year": year,
                "value": float(num_value) / float(denom_value),
                "derived_from": _dedupe_refs(list(row.get("derived_from") or []) + list(denom.get("derived_from") or [])),
            }
        )
    return sorted(out, key=lambda row: int(row.get("year") or 0))


def _claim_refs(payload: dict[str, Any], key: str) -> list[str]:
    claims = payload.get("claims") if isinstance(payload.get("claims"), dict) else {}
    claim = claims.get(key) if isinstance(claims, dict) else {}
    return [str(ref) for ref in (claim.get("derived_from") or []) if str(ref).strip()] if isinstance(claim, dict) else []


def _net_debt_series_from_companyfacts(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_debt = _series_map(
        _series_from_companyfacts(
            companyfacts=companyfacts,
            as_of_date=as_of_date,
            priority=_TOTAL_DEBT_TAG_PRIORITY,
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
    years = sorted(set(total_debt) | set(cash))
    net_debt_rows: list[dict[str, Any]] = []
    cash_rows: list[dict[str, Any]] = []
    for year in years:
        cash_row = cash.get(year)
        debt_row = total_debt.get(year)
        if isinstance(cash_row, dict) and _is_num(cash_row.get("value")):
            cash_rows.append(
                {
                    "year": year,
                    "value": float(cash_row["value"]),
                    "derived_from": list(cash_row.get("derived_from") or []),
                }
            )
        if isinstance(debt_row, dict) and isinstance(cash_row, dict) and _is_num(debt_row.get("value")) and _is_num(cash_row.get("value")):
            net_debt_rows.append(
                {
                    "year": year,
                    "value": float(debt_row["value"]) - float(cash_row["value"]),
                    "derived_from": _dedupe_refs(list(debt_row.get("derived_from") or []) + list(cash_row.get("derived_from") or [])),
                }
            )
    return net_debt_rows, cash_rows


def _build_series_from_fundamentals(fundamentals: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = sorted([row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)], key=lambda row: int(row.get("year") or 0))
    traces = fundamentals.get("row_traces") if isinstance(fundamentals.get("row_traces"), dict) else {}

    gross_margin_rows: list[dict[str, Any]] = []
    gross_profit_rows: list[dict[str, Any]] = []
    operating_income_rows: list[dict[str, Any]] = []
    revenue_rows: list[dict[str, Any]] = []
    cfo_rows: list[dict[str, Any]] = []
    capex_rows: list[dict[str, Any]] = []
    fcf_rows: list[dict[str, Any]] = []
    owner_earnings_rows: list[dict[str, Any]] = []
    shares_rows: list[dict[str, Any]] = []
    rnd_rows: list[dict[str, Any]] = []
    sga_rows: list[dict[str, Any]] = []
    net_debt_rows: list[dict[str, Any]] = []
    cash_rows: list[dict[str, Any]] = []

    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0:
            continue
        revenue = row.get("revenue", UNKNOWN)
        gross_margin = row.get("gross_margin", UNKNOWN)
        gross_profit = row.get("gross_profit", UNKNOWN)
        operating_income = row.get("operating_income", UNKNOWN)
        cfo = row.get("cfo", UNKNOWN)
        capex = row.get("capex", UNKNOWN)
        fcf = row.get("fcf", UNKNOWN)
        shares = row.get("shares_outstanding", UNKNOWN)
        rnd = row.get("r_and_d_total", row.get("r_and_d", UNKNOWN))
        sales_marketing_total = row.get("sales_marketing_total", UNKNOWN)
        g_and_a_total = row.get("g_and_a_total", UNKNOWN)
        sga_total = row.get("sga_total", UNKNOWN)
        net_debt = row.get("net_debt", UNKNOWN)
        cash = row.get("cash", UNKNOWN)
        if _is_num(cash):
            cash_rows.append(
                {
                    "year": year,
                    "value": float(cash),
                    "derived_from": _trace_refs(traces, year, "cash", f"fundamentals.rows[{year}].cash"),
                }
            )
        if not _is_num(sga_total) and _is_num(sales_marketing_total) and _is_num(g_and_a_total):
            sga_total = float(sales_marketing_total) + float(g_and_a_total)
        if not _is_num(gross_margin) and _is_num(gross_profit) and _is_num(revenue) and float(revenue) != 0.0:
            gross_margin = float(gross_profit) / float(revenue)
        if _is_num(gross_margin):
            gross_margin_rows.append(
                {
                    "year": year,
                    "value": float(gross_margin),
                    "derived_from": _dedupe_refs(
                        _trace_refs(traces, year, "gross_margin", f"fundamentals.rows[{year}].gross_margin")
                        + _trace_refs(traces, year, "gross_profit", f"fundamentals.rows[{year}].gross_profit")
                        + _trace_refs(traces, year, "revenue", f"fundamentals.rows[{year}].revenue")
                    ),
                }
            )
        if _is_num(gross_profit):
            gross_profit_rows.append(
                {
                    "year": year,
                    "value": float(gross_profit),
                    "derived_from": _trace_refs(traces, year, "gross_profit", f"fundamentals.rows[{year}].gross_profit"),
                }
            )
        if _is_num(operating_income):
            operating_income_rows.append(
                {
                    "year": year,
                    "value": float(operating_income),
                    "derived_from": _trace_refs(
                        traces,
                        year,
                        "operating_income",
                        f"fundamentals.rows[{year}].operating_income",
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
        if _is_num(cfo) and _is_num(capex):
            owner_earnings_rows.append(
                {
                    "year": year,
                    "value": float(cfo) - (float(DEFAULT_MAINT_CAPEX_RATIO) * abs(float(capex))),
                    "derived_from": _dedupe_refs(
                        _trace_refs(traces, year, "cfo", f"fundamentals.rows[{year}].cfo")
                        + _trace_refs(traces, year, "capex", f"fundamentals.rows[{year}].capex")
                    ),
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
        if _is_num(rnd):
            rnd_rows.append(
                {
                    "year": year,
                    "value": float(rnd),
                    "derived_from": _trace_refs(traces, year, "r_and_d_total", f"fundamentals.rows[{year}].r_and_d_total"),
                }
            )
        if _is_num(sga_total):
            sga_rows.append(
                {
                    "year": year,
                    "value": float(sga_total),
                    "derived_from": _dedupe_refs(
                        _trace_refs(traces, year, "sga_total", f"fundamentals.rows[{year}].sga_total")
                        + _trace_refs(
                            traces,
                            year,
                            "sales_marketing_total",
                            f"fundamentals.rows[{year}].sales_marketing_total",
                        )
                        + _trace_refs(traces, year, "g_and_a_total", f"fundamentals.rows[{year}].g_and_a_total")
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

    return {
        "gross_margin": gross_margin_rows,
        "gross_profit": gross_profit_rows,
        "operating_income": operating_income_rows,
        "revenue": revenue_rows,
        "cfo": cfo_rows,
        "capex": capex_rows,
        "fcf": fcf_rows,
        "owner_earnings": owner_earnings_rows,
        "shares": shares_rows,
        "rnd": rnd_rows,
        "sga": sga_rows,
        "net_debt": net_debt_rows,
        "cash": cash_rows,
    }


def _build_series_from_companyfacts(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
) -> dict[str, list[dict[str, Any]]]:
    revenue_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_REVENUE_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    gross_profit_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_GROSS_PROFIT_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    operating_income_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_OPERATING_INCOME_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    cfo_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CFO_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    fcf_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=FCF_DIRECT_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    if not fcf_rows:
        cfo_by_year = _series_map(cfo_rows)
        capex_by_year = _series_map(
            _series_from_companyfacts(
                companyfacts=companyfacts,
                as_of_date=as_of_date,
                priority=_CAPEX_TAG_PRIORITY,
                expected_unit_exact=("usd",),
            )
        )
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
                    "value": float(cfo_row["value"]) - abs(float(capex_row["value"])),
                    "derived_from": _dedupe_refs(list(cfo_row.get("derived_from") or []) + list(capex_row.get("derived_from") or [])),
                }
            )
        fcf_rows = derived_fcf
    capex_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_CAPEX_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    owner_earnings_rows: list[dict[str, Any]] = []
    cfo_by_year = _series_map(cfo_rows)
    capex_by_year = _series_map(capex_rows)
    for year in sorted(set(cfo_by_year) & set(capex_by_year)):
        cfo_row = cfo_by_year.get(year)
        capex_row = capex_by_year.get(year)
        if not isinstance(cfo_row, dict) or not isinstance(capex_row, dict):
            continue
        if not _is_num(cfo_row.get("value")) or not _is_num(capex_row.get("value")):
            continue
        owner_earnings_rows.append(
            {
                "year": year,
                "value": float(cfo_row["value"]) - (float(DEFAULT_MAINT_CAPEX_RATIO) * abs(float(capex_row["value"]))),
                "derived_from": _dedupe_refs(list(cfo_row.get("derived_from") or []) + list(capex_row.get("derived_from") or [])),
            }
        )

    rnd_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_RND_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    sga_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=_SGA_DIRECT_TAG_PRIORITY,
        expected_unit_exact=("usd",),
    )
    if not sga_rows:
        sales_marketing = _series_map(
            _series_from_companyfacts(
                companyfacts=companyfacts,
                as_of_date=as_of_date,
                priority=_SALES_MARKETING_TAG_PRIORITY,
                expected_unit_exact=("usd",),
            )
        )
        g_and_a = _series_map(
            _series_from_companyfacts(
                companyfacts=companyfacts,
                as_of_date=as_of_date,
                priority=_G_AND_A_TAG_PRIORITY,
                expected_unit_exact=("usd",),
            )
        )
        derived_sga: list[dict[str, Any]] = []
        for year in sorted(set(sales_marketing) & set(g_and_a)):
            sales_row = sales_marketing.get(year)
            gna_row = g_and_a.get(year)
            if not isinstance(sales_row, dict) or not isinstance(gna_row, dict):
                continue
            if not _is_num(sales_row.get("value")) or not _is_num(gna_row.get("value")):
                continue
            derived_sga.append(
                {
                    "year": year,
                    "value": float(sales_row["value"]) + float(gna_row["value"]),
                    "derived_from": _dedupe_refs(list(sales_row.get("derived_from") or []) + list(gna_row.get("derived_from") or [])),
                }
            )
        sga_rows = derived_sga
    shares_rows = _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=SHARES_TAG_PRIORITY,
        expected_unit_exact=("shares",),
    )

    revenue_by_year = _series_map(revenue_rows)
    gross_profit_by_year = _series_map(gross_profit_rows)
    gross_margin_rows: list[dict[str, Any]] = []
    for year in sorted(set(revenue_by_year) & set(gross_profit_by_year)):
        revenue_row = revenue_by_year.get(year)
        gross_profit_row = gross_profit_by_year.get(year)
        if not isinstance(revenue_row, dict) or not isinstance(gross_profit_row, dict):
            continue
        if not _is_num(revenue_row.get("value")) or not _is_num(gross_profit_row.get("value")):
            continue
        if float(revenue_row["value"]) == 0.0:
            continue
        gross_margin_rows.append(
            {
                "year": year,
                "value": float(gross_profit_row["value"]) / float(revenue_row["value"]),
                "derived_from": _dedupe_refs(list(gross_profit_row.get("derived_from") or []) + list(revenue_row.get("derived_from") or [])),
            }
        )

    net_debt_rows, cash_rows = _net_debt_series_from_companyfacts(companyfacts=companyfacts, as_of_date=as_of_date)
    return {
        "gross_margin": gross_margin_rows,
        "gross_profit": gross_profit_rows,
        "operating_income": operating_income_rows,
        "revenue": revenue_rows,
        "cfo": cfo_rows,
        "capex": capex_rows,
        "fcf": fcf_rows,
        "owner_earnings": owner_earnings_rows,
        "shares": shares_rows,
        "rnd": rnd_rows,
        "sga": sga_rows,
        "net_debt": net_debt_rows,
        "cash": cash_rows,
    }


def _gross_margin_durability(series_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    numeric = _recent_adjacent([row for row in (series_map.get("gross_margin") or []) if _is_num(row.get("value"))])
    refs = _dedupe_refs([ref for row in numeric for ref in list(row.get("derived_from") or [])])
    values = [float(row.get("value")) for row in numeric]
    avg_value = round(mean(values), 6) if values else UNKNOWN
    floor_value = round(min(values), 6) if values else UNKNOWN
    volatility = round(pstdev(values), 6) if len(values) >= 2 else UNKNOWN

    reason_codes: list[str] = []
    if not values:
        reason_codes.append(REASON_MISSING_GROSS_MARGIN_HISTORY)
    elif len(values) < 3:
        reason_codes.append(REASON_INSUFFICIENT_GROSS_MARGIN_HISTORY)
    if _is_num(volatility) and float(volatility) > 0.12:
        reason_codes.append(REASON_VOLATILE_GROSS_MARGIN)

    score: float | str = UNKNOWN
    if len(values) >= 3 and _is_num(avg_value) and _is_num(floor_value) and _is_num(volatility):
        avg_float = float(avg_value)
        floor_float = float(floor_value)
        vol_float = float(volatility)
        if avg_float >= 0.60 and floor_float >= 0.55 and vol_float <= 0.05:
            score = 5.0
        elif avg_float >= 0.50 and floor_float >= 0.42 and vol_float <= 0.08:
            score = 4.0
        elif avg_float >= 0.40 and floor_float >= 0.32 and vol_float <= 0.10:
            score = 3.0
        elif avg_float >= 0.25 and floor_float >= 0.20 and vol_float <= 0.14:
            score = 2.0
        elif avg_float > 0.0 and floor_float > 0.0 and vol_float <= 0.18:
            score = 1.0
        else:
            score = 0.0

    if not _is_num(score):
        reason_codes.append(REASON_GROSS_MARGIN_DURABILITY_UNKNOWN)

    primary_reason = reason_codes[0] if reason_codes else OK
    return {
        "gross_margin_avg_5y": _claim(value=avg_value, refs=refs, reason_code=primary_reason if not _is_num(avg_value) else OK),
        "gross_margin_volatility_5y": _claim(value=volatility, refs=refs, reason_code=primary_reason if not _is_num(volatility) else OK),
        "gross_margin_floor_5y": _claim(value=floor_value, refs=refs, reason_code=primary_reason if not _is_num(floor_value) else OK),
        "gross_margin_durability_score": _claim(value=score, refs=refs, reason_code=primary_reason),
        "reason_codes": _dedupe_refs(reason_codes),
    }


def _balance_sheet_optionality(
    series_map: dict[str, list[dict[str, Any]]],
    *,
    net_debt_resolved: dict[str, Any] | None = None,
    source_kind: str | None = None,
) -> dict[str, Any]:
    net_debt_rows = list(series_map.get("net_debt") or [])
    if isinstance(net_debt_resolved, dict) and _is_num(net_debt_resolved.get("net_debt_proxy")):
        refs = [str(ref) for ref in (net_debt_resolved.get("derived_from") or []) if str(ref).strip()]
        net_debt_rows = [
            {
                "year": 9999,
                "value": float(net_debt_resolved["net_debt_proxy"]),
                "derived_from": refs,
            }
        ]
    net_debt_row = _latest_numeric_row(net_debt_rows)
    cfo_row = _latest_numeric_row(series_map.get("cfo") or [])
    cash_row, revenue_row = _latest_common_pair(series_map.get("cash") or [], series_map.get("revenue") or [])

    net_debt_value = net_debt_row.get("value", UNKNOWN) if isinstance(net_debt_row, dict) else UNKNOWN
    cfo_value = cfo_row.get("value", UNKNOWN) if isinstance(cfo_row, dict) else UNKNOWN
    debt_for_ratio = net_debt_value
    if (
        source_kind == "COMPANYFACTS"
        and isinstance(net_debt_resolved, dict)
        and _is_num(net_debt_resolved.get("net_debt_proxy"))
    ):
        # Resolver debt is millions; raw companyfacts cash flows are whole USD.
        debt_for_ratio = float(net_debt_value) * 1_000_000.0
    net_debt_to_cfo = (
        float(debt_for_ratio) / float(cfo_value)
        if _is_num(net_debt_value) and _is_num(cfo_value) and float(cfo_value) > 0.0
        else UNKNOWN
    )
    cash_pct_revenue = (
        float(cash_row.get("value")) / float(revenue_row.get("value"))
        if isinstance(cash_row, dict)
        and isinstance(revenue_row, dict)
        and _is_num(cash_row.get("value"))
        and _is_num(revenue_row.get("value"))
        and float(revenue_row.get("value")) > 0.0
        else UNKNOWN
    )

    net_debt_refs = list(net_debt_row.get("derived_from") or []) if isinstance(net_debt_row, dict) else []
    cfo_refs = list(cfo_row.get("derived_from") or []) if isinstance(cfo_row, dict) else []
    cash_refs = list(cash_row.get("derived_from") or []) if isinstance(cash_row, dict) else []
    revenue_refs = list(revenue_row.get("derived_from") or []) if isinstance(revenue_row, dict) else []

    known = any(_is_num(value) for value in [net_debt_value, net_debt_to_cfo, cash_pct_revenue])
    if not known:
        reason_codes = [REASON_MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS, REASON_BALANCE_SHEET_OPTIONALITY_UNKNOWN]
        refs = _dedupe_refs(net_debt_refs + cfo_refs + cash_refs + revenue_refs)
        return {
            "net_debt_proxy": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS),
            "net_debt_to_cfo_proxy": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS),
            "cash_pct_revenue": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS),
            "balance_sheet_optionality_score": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS),
            "reason_codes": reason_codes,
        }

    score = 0.0
    reason_codes: list[str] = []
    if _is_num(net_debt_value):
        if float(net_debt_value) <= 0.0:
            score += 3.0
    if _is_num(net_debt_to_cfo):
        ratio = float(net_debt_to_cfo)
        if ratio <= 0.0:
            score += 2.0
        elif ratio <= 1.0:
            score += 2.0
        elif ratio <= 2.5:
            score += 1.0
        elif ratio > 3.0:
            score -= 2.0
            reason_codes.append(REASON_NET_DEBT_PRESSURE)
    elif _is_num(net_debt_value) and float(net_debt_value) > 0.0:
        reason_codes.append(REASON_MISSING_BALANCE_SHEET_OPTIONALITY_INPUTS)

    if _is_num(cash_pct_revenue):
        cash_ratio = float(cash_pct_revenue)
        if cash_ratio >= 0.25:
            score += 2.0
        elif cash_ratio >= 0.10:
            score += 1.0
        elif cash_ratio < 0.05:
            score -= 1.0
            reason_codes.append(REASON_WEAK_LIQUIDITY_SUPPORT)

    score = max(0.0, min(5.0, round(score, 6)))
    if not reason_codes and score >= 4.0:
        reason_codes.append(REASON_BALANCE_SHEET_OPTIONALITY_STRONG)
    score_reason = reason_codes[0] if reason_codes else OK
    refs = _dedupe_refs(net_debt_refs + cfo_refs + cash_refs + revenue_refs)
    return {
        "net_debt_proxy": _claim(value=net_debt_value, refs=net_debt_refs, reason_code=REASON_BALANCE_SHEET_OPTIONALITY_UNKNOWN if not _is_num(net_debt_value) else OK),
        "net_debt_to_cfo_proxy": _claim(value=net_debt_to_cfo, refs=net_debt_refs + cfo_refs, reason_code=REASON_BALANCE_SHEET_OPTIONALITY_UNKNOWN if not _is_num(net_debt_to_cfo) else OK),
        "cash_pct_revenue": _claim(value=cash_pct_revenue, refs=cash_refs + revenue_refs, reason_code=REASON_BALANCE_SHEET_OPTIONALITY_UNKNOWN if not _is_num(cash_pct_revenue) else OK),
        "balance_sheet_optionality_score": _claim(value=score, refs=refs, reason_code=score_reason),
        "reason_codes": _dedupe_refs(reason_codes),
    }


def _cycle_resilience(
    series_map: dict[str, list[dict[str, Any]]],
    *,
    owner_quality_payload: dict[str, Any],
) -> dict[str, Any]:
    owner_volatility = owner_quality_payload.get("owner_earnings_volatility_5y", UNKNOWN)
    owner_refs = []
    claims = owner_quality_payload.get("claims") if isinstance(owner_quality_payload.get("claims"), dict) else {}
    owner_claim = claims.get("owner_earnings_volatility_5y") if isinstance(claims, dict) else {}
    if isinstance(owner_claim, dict):
        owner_refs = [str(ref) for ref in (owner_claim.get("derived_from") or []) if str(ref).strip()]

    gross_margin_rows = _recent_adjacent([row for row in (series_map.get("gross_margin") or []) if _is_num(row.get("value"))])
    gross_margin_values = [float(row.get("value")) for row in gross_margin_rows]
    margin_volatility = round(pstdev(gross_margin_values), 6) if len(gross_margin_values) >= 2 else UNKNOWN
    margin_refs = _dedupe_refs([ref for row in gross_margin_rows for ref in list(row.get("derived_from") or [])])

    conversion_rows = _recent_common_ratios(series_map.get("fcf") or [], series_map.get("cfo") or [], require_positive_denominator=True, window=3)
    conversion_values = [float(value) for value, _refs in conversion_rows]
    conversion_median = round(median(conversion_values), 6) if conversion_values else UNKNOWN
    conversion_volatility = round(pstdev(conversion_values), 6) if len(conversion_values) >= 2 else UNKNOWN
    conversion_refs = _dedupe_refs([ref for _value, refs in conversion_rows for ref in refs])

    sufficient = bool(_is_num(owner_volatility) or len(gross_margin_values) >= 3 or len(conversion_values) >= 2)
    if not sufficient:
        reason_codes = [REASON_INSUFFICIENT_CYCLE_HISTORY, REASON_CYCLE_RESILIENCE_UNKNOWN]
        refs = _dedupe_refs(owner_refs + margin_refs + conversion_refs)
        return {
            "margin_volatility_5y": _claim(value=margin_volatility, refs=margin_refs, reason_code=REASON_INSUFFICIENT_CYCLE_HISTORY),
            "fcf_cfo_conversion_median_3y": _claim(value=conversion_median, refs=conversion_refs, reason_code=REASON_INSUFFICIENT_CYCLE_HISTORY),
            "fcf_cfo_conversion_volatility_3y": _claim(value=conversion_volatility, refs=conversion_refs, reason_code=REASON_INSUFFICIENT_CYCLE_HISTORY),
            "cycle_resilience_score": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_INSUFFICIENT_CYCLE_HISTORY),
            "reason_codes": reason_codes,
        }

    score = 0.0
    reason_codes: list[str] = []
    if _is_num(owner_volatility):
        owner_vol = float(owner_volatility)
        if owner_vol <= 0.40:
            score += 2.0
        elif owner_vol <= 0.80:
            score += 1.0
        elif owner_vol > 1.20:
            reason_codes.append(REASON_HIGH_OWNER_EARNINGS_VOLATILITY)
    if _is_num(margin_volatility):
        gm_vol = float(margin_volatility)
        if gm_vol <= 0.05:
            score += 2.0
        elif gm_vol <= 0.10:
            score += 1.0
        elif gm_vol > 0.15:
            reason_codes.append(REASON_HIGH_MARGIN_VOLATILITY)
    if _is_num(conversion_median) and _is_num(conversion_volatility):
        if float(conversion_median) >= 0.50 and float(conversion_volatility) <= 0.25:
            score += 1.0
    if not reason_codes and score >= 4.0:
        reason_codes.append(REASON_RESILIENT_CYCLICAL_PROFILE)
    elif (
        _is_num(owner_volatility)
        and _is_num(margin_volatility)
        and _is_num(conversion_median)
        and float(owner_volatility) > 0.80
        and float(margin_volatility) <= 0.08
        and float(conversion_median) >= 0.60
    ):
        score = max(score, 3.0)
        reason_codes.append(REASON_RESILIENT_CYCLICAL_PROFILE)

    score = max(0.0, min(5.0, round(score, 6)))
    refs = _dedupe_refs(owner_refs + margin_refs + conversion_refs)
    score_reason = reason_codes[0] if reason_codes else OK
    return {
        "margin_volatility_5y": _claim(value=margin_volatility, refs=margin_refs, reason_code=REASON_CYCLE_RESILIENCE_UNKNOWN if not _is_num(margin_volatility) else OK),
        "fcf_cfo_conversion_median_3y": _claim(value=conversion_median, refs=conversion_refs, reason_code=REASON_CYCLE_RESILIENCE_UNKNOWN if not _is_num(conversion_median) else OK),
        "fcf_cfo_conversion_volatility_3y": _claim(value=conversion_volatility, refs=conversion_refs, reason_code=REASON_CYCLE_RESILIENCE_UNKNOWN if not _is_num(conversion_volatility) else OK),
        "cycle_resilience_score": _claim(value=score, refs=refs, reason_code=score_reason),
        "reason_codes": _dedupe_refs(reason_codes),
    }


def _rnd_productivity(
    series_map: dict[str, list[dict[str, Any]]],
    *,
    owner_quality_payload: dict[str, Any],
) -> dict[str, Any]:
    rnd_rows = _recent_adjacent([row for row in (series_map.get("rnd") or []) if _is_num(row.get("value"))])
    rnd_refs = _dedupe_refs([ref for row in rnd_rows for ref in list(row.get("derived_from") or [])])
    if not rnd_rows:
        reason_codes = [REASON_MISSING_RND_HISTORY, REASON_RND_PRODUCTIVITY_UNKNOWN]
        return {
            "rnd_to_revenue_avg_5y": _claim(value=UNKNOWN, refs=rnd_refs, reason_code=REASON_MISSING_RND_HISTORY),
            "revenue_per_rnd_proxy": _claim(value=UNKNOWN, refs=rnd_refs, reason_code=REASON_MISSING_RND_HISTORY),
            "gross_profit_per_rnd_proxy": _claim(value=UNKNOWN, refs=rnd_refs, reason_code=REASON_MISSING_RND_HISTORY),
            "owner_earnings_per_rnd_proxy": _claim(value=UNKNOWN, refs=rnd_refs, reason_code=REASON_MISSING_RND_HISTORY),
            "rnd_productivity_score": _claim(value=UNKNOWN, refs=rnd_refs, reason_code=REASON_MISSING_RND_HISTORY),
            "reason_codes": reason_codes,
        }

    rnd_to_revenue_rows = _recent_adjacent(_ratio_series(rnd_rows, series_map.get("revenue") or [], require_positive_denominator=True))
    revenue_per_rnd_rows = _recent_adjacent(_ratio_series(series_map.get("revenue") or [], rnd_rows, require_positive_denominator=True))
    gross_profit_per_rnd_rows = _recent_adjacent(_ratio_series(series_map.get("gross_profit") or [], rnd_rows, require_positive_denominator=True))
    owner_rows = series_map.get("owner_earnings") or []
    if not owner_rows and isinstance(owner_quality_payload.get("series"), list):
        owner_rows = _owner_series_from_owner_payload(owner_quality_payload)
    owner_per_rnd_rows = _recent_adjacent(_ratio_series(owner_rows, rnd_rows, require_positive_denominator=True))

    rnd_to_revenue = round(mean([float(row["value"]) for row in rnd_to_revenue_rows]), 6) if rnd_to_revenue_rows else UNKNOWN
    revenue_per_rnd = round(median([float(row["value"]) for row in revenue_per_rnd_rows]), 6) if revenue_per_rnd_rows else UNKNOWN
    gross_profit_per_rnd = round(median([float(row["value"]) for row in gross_profit_per_rnd_rows]), 6) if gross_profit_per_rnd_rows else UNKNOWN
    owner_per_rnd = round(median([float(row["value"]) for row in owner_per_rnd_rows]), 6) if owner_per_rnd_rows else UNKNOWN

    refs = _dedupe_refs(
        rnd_refs
        + [ref for row in rnd_to_revenue_rows for ref in list(row.get("derived_from") or [])]
        + [ref for row in revenue_per_rnd_rows for ref in list(row.get("derived_from") or [])]
        + [ref for row in gross_profit_per_rnd_rows for ref in list(row.get("derived_from") or [])]
        + [ref for row in owner_per_rnd_rows for ref in list(row.get("derived_from") or [])]
    )

    reason_codes: list[str] = []
    if len(rnd_rows) < 3:
        reason_codes.append(REASON_INSUFFICIENT_RND_HISTORY)
    if not any(_is_num(value) for value in [revenue_per_rnd, gross_profit_per_rnd, owner_per_rnd]):
        reason_codes.append(REASON_MISSING_RND_DENOMINATOR)

    score: float | str = UNKNOWN
    if len(rnd_rows) >= 3 and any(_is_num(value) for value in [revenue_per_rnd, gross_profit_per_rnd, owner_per_rnd]):
        score_value = 0.0
        if _is_num(revenue_per_rnd):
            ratio = float(revenue_per_rnd)
            if ratio >= 10.0:
                score_value += 2.0
            elif ratio >= 6.0:
                score_value += 1.5
            elif ratio >= 4.0:
                score_value += 1.0
            elif ratio > 0.0:
                score_value += 0.5
        if _is_num(gross_profit_per_rnd):
            ratio = float(gross_profit_per_rnd)
            if ratio >= 6.0:
                score_value += 2.0
            elif ratio >= 3.0:
                score_value += 1.5
            elif ratio >= 1.5:
                score_value += 1.0
            elif ratio > 0.0:
                score_value += 0.5
        if _is_num(owner_per_rnd):
            ratio = float(owner_per_rnd)
            if ratio >= 1.0:
                score_value += 1.5
            elif ratio >= 0.4:
                score_value += 1.0
            elif ratio >= 0.1:
                score_value += 0.5
        score = max(0.0, min(5.0, round(score_value, 6)))
        if float(score) <= 1.0 or (_is_num(rnd_to_revenue) and float(rnd_to_revenue) > 0.30 and float(score) < 2.0):
            reason_codes.append(REASON_LOW_RND_PRODUCTIVITY)
    if not _is_num(score):
        reason_codes.append(REASON_RND_PRODUCTIVITY_UNKNOWN)

    reason_codes = _dedupe_refs(reason_codes)
    score_reason = reason_codes[0] if reason_codes else OK
    return {
        "rnd_to_revenue_avg_5y": _claim(value=rnd_to_revenue, refs=refs, reason_code=score_reason if not _is_num(rnd_to_revenue) else OK),
        "revenue_per_rnd_proxy": _claim(value=revenue_per_rnd, refs=refs, reason_code=score_reason if not _is_num(revenue_per_rnd) else OK),
        "gross_profit_per_rnd_proxy": _claim(value=gross_profit_per_rnd, refs=refs, reason_code=score_reason if not _is_num(gross_profit_per_rnd) else OK),
        "owner_earnings_per_rnd_proxy": _claim(value=owner_per_rnd, refs=refs, reason_code=score_reason if not _is_num(owner_per_rnd) else OK),
        "rnd_productivity_score": _claim(value=score, refs=refs, reason_code=score_reason),
        "reason_codes": reason_codes,
    }


def _sga_leverage(series_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sga_rows = _recent_adjacent([row for row in (series_map.get("sga") or []) if _is_num(row.get("value"))])
    sga_refs = _dedupe_refs([ref for row in sga_rows for ref in list(row.get("derived_from") or [])])
    if not sga_rows:
        reason_codes = [REASON_MISSING_SGA_HISTORY, REASON_SGA_LEVERAGE_UNKNOWN]
        return {
            "sga_to_revenue_avg_5y": _claim(value=UNKNOWN, refs=sga_refs, reason_code=REASON_MISSING_SGA_HISTORY),
            "sga_growth_vs_revenue_growth_proxy": _claim(value=UNKNOWN, refs=sga_refs, reason_code=REASON_MISSING_SGA_HISTORY),
            "operating_leverage_proxy": _claim(value=UNKNOWN, refs=sga_refs, reason_code=REASON_MISSING_SGA_HISTORY),
            "sga_leverage_score": _claim(value=UNKNOWN, refs=sga_refs, reason_code=REASON_MISSING_SGA_HISTORY),
            "reason_codes": reason_codes,
        }

    sga_to_revenue_rows = _recent_adjacent(_ratio_series(sga_rows, series_map.get("revenue") or [], require_positive_denominator=True))
    sga_to_revenue = round(mean([float(row["value"]) for row in sga_to_revenue_rows]), 6) if sga_to_revenue_rows else UNKNOWN
    sga_cagr, sga_growth_refs = _series_cagr(sga_rows)
    revenue_cagr, revenue_growth_refs = _series_cagr(series_map.get("revenue") or [])
    growth_spread = (
        round(float(revenue_cagr) - float(sga_cagr), 6)
        if _is_num(revenue_cagr) and _is_num(sga_cagr)
        else UNKNOWN
    )

    operating_leverage = UNKNOWN
    operating_refs: list[str] = []
    op_margin_rows = _ratio_series(series_map.get("operating_income") or [], series_map.get("revenue") or [], require_positive_denominator=True)
    if len(op_margin_rows) >= 3:
        operating_leverage = round(float(op_margin_rows[-1]["value"]) - float(op_margin_rows[0]["value"]), 6)
        operating_refs = _dedupe_refs(list(op_margin_rows[0].get("derived_from") or []) + list(op_margin_rows[-1].get("derived_from") or []))
    else:
        revenue_by_year = {int(row.get("year") or 0): row for row in (series_map.get("revenue") or []) if int(row.get("year") or 0) > 0}
        gross_profit_by_year = {int(row.get("year") or 0): row for row in (series_map.get("gross_profit") or []) if int(row.get("year") or 0) > 0}
        coverage_rows: list[dict[str, Any]] = []
        for sga_row in sga_rows:
            year = int(sga_row.get("year") or 0)
            revenue_row = revenue_by_year.get(year)
            gross_profit_row = gross_profit_by_year.get(year)
            if year <= 0 or not isinstance(revenue_row, dict) or not isinstance(gross_profit_row, dict):
                continue
            if not _is_num(revenue_row.get("value")) or not _is_num(gross_profit_row.get("value")) or not _is_num(sga_row.get("value")):
                continue
            if float(revenue_row["value"]) <= 0.0:
                continue
            coverage_rows.append(
                {
                    "year": year,
                    "value": (float(gross_profit_row["value"]) - float(sga_row["value"])) / float(revenue_row["value"]),
                    "derived_from": _dedupe_refs(
                        list(gross_profit_row.get("derived_from") or [])
                        + list(sga_row.get("derived_from") or [])
                        + list(revenue_row.get("derived_from") or [])
                    ),
                }
            )
        if len(coverage_rows) >= 3:
            operating_leverage = round(float(coverage_rows[-1]["value"]) - float(coverage_rows[0]["value"]), 6)
            operating_refs = _dedupe_refs(list(coverage_rows[0].get("derived_from") or []) + list(coverage_rows[-1].get("derived_from") or []))

    refs = _dedupe_refs(
        sga_refs
        + [ref for row in sga_to_revenue_rows for ref in list(row.get("derived_from") or [])]
        + sga_growth_refs
        + revenue_growth_refs
        + operating_refs
    )
    reason_codes: list[str] = []
    if len(sga_rows) < 3:
        reason_codes.append(REASON_INSUFFICIENT_SGA_HISTORY)
    if not any(_is_num(value) for value in [sga_to_revenue, growth_spread, operating_leverage]):
        reason_codes.append(REASON_MISSING_OPERATING_LEVERAGE_INPUTS)

    score: float | str = UNKNOWN
    if len(sga_rows) >= 3 and any(_is_num(value) for value in [sga_to_revenue, growth_spread, operating_leverage]):
        score_value = 0.0
        if _is_num(sga_to_revenue):
            ratio = float(sga_to_revenue)
            if ratio <= 0.20:
                score_value += 2.0
            elif ratio <= 0.30:
                score_value += 1.5
            elif ratio <= 0.40:
                score_value += 1.0
            elif ratio <= 0.55:
                score_value += 0.5
        if _is_num(growth_spread):
            spread = float(growth_spread)
            if spread >= 0.05:
                score_value += 2.0
            elif spread >= 0.02:
                score_value += 1.5
            elif spread >= 0.0:
                score_value += 1.0
            elif spread > -0.03:
                score_value += 0.5
            elif spread < -0.05:
                reason_codes.append(REASON_WEAK_SGA_LEVERAGE)
        if _is_num(operating_leverage):
            leverage = float(operating_leverage)
            if leverage >= 0.05:
                score_value += 1.5
            elif leverage >= 0.02:
                score_value += 1.0
            elif leverage >= 0.0:
                score_value += 0.5
            elif leverage < -0.03:
                reason_codes.append(REASON_WEAK_SGA_LEVERAGE)
        score = max(0.0, min(5.0, round(score_value, 6)))
        if float(score) <= 1.0:
            reason_codes.append(REASON_WEAK_SGA_LEVERAGE)
    if not _is_num(score):
        reason_codes.append(REASON_SGA_LEVERAGE_UNKNOWN)

    reason_codes = _dedupe_refs(reason_codes)
    score_reason = reason_codes[0] if reason_codes else OK
    return {
        "sga_to_revenue_avg_5y": _claim(value=sga_to_revenue, refs=refs, reason_code=score_reason if not _is_num(sga_to_revenue) else OK),
        "sga_growth_vs_revenue_growth_proxy": _claim(value=growth_spread, refs=refs, reason_code=score_reason if not _is_num(growth_spread) else OK),
        "operating_leverage_proxy": _claim(value=operating_leverage, refs=refs, reason_code=score_reason if not _is_num(operating_leverage) else OK),
        "sga_leverage_score": _claim(value=score, refs=refs, reason_code=score_reason),
        "reason_codes": reason_codes,
    }


def _owner_value_capture(
    series_map: dict[str, list[dict[str, Any]]],
    *,
    owner_quality_payload: dict[str, Any],
) -> dict[str, Any]:
    shares_rows = list(series_map.get("shares") or [])
    dilution_rate = owner_quality_payload.get("dilution_rate_shares_cagr", UNKNOWN)
    dilution_refs = _claim_refs(owner_quality_payload, "dilution_rate_shares_cagr")
    if not _is_num(dilution_rate):
        dilution_rate, dilution_refs = _series_cagr(shares_rows)

    revenue_per_share_rows = _ratio_series(series_map.get("revenue") or [], shares_rows, require_positive_denominator=True)
    fcf_per_share_rows = _ratio_series(series_map.get("fcf") or [], shares_rows, require_positive_denominator=True)
    owner_rows = series_map.get("owner_earnings") or []
    if not owner_rows and isinstance(owner_quality_payload.get("series"), list):
        owner_rows = _owner_series_from_owner_payload(owner_quality_payload)
    owner_per_share_rows = _ratio_series(owner_rows, shares_rows, require_positive_denominator=True)

    revenue_per_share_cagr, revenue_per_share_refs = _series_cagr(revenue_per_share_rows)
    fcf_per_share_cagr, fcf_per_share_refs = _series_cagr(fcf_per_share_rows)
    owner_per_share_cagr, owner_per_share_refs = _series_cagr(owner_per_share_rows)
    refs = _dedupe_refs(dilution_refs + revenue_per_share_refs + fcf_per_share_refs + owner_per_share_refs)

    reason_codes: list[str] = []
    if not any(
        _is_num(value)
        for value in [dilution_rate, revenue_per_share_cagr, fcf_per_share_cagr, owner_per_share_cagr]
    ):
        reason_codes.extend([REASON_MISSING_PER_SHARE_INPUTS, REASON_OWNER_VALUE_CAPTURE_UNKNOWN])
        return {
            "dilution_rate_shares_cagr": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_PER_SHARE_INPUTS),
            "revenue_per_share_cagr_proxy": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_PER_SHARE_INPUTS),
            "fcf_per_share_cagr_proxy": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_PER_SHARE_INPUTS),
            "owner_earnings_per_share_cagr_proxy": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_PER_SHARE_INPUTS),
            "owner_value_capture_score": _claim(value=UNKNOWN, refs=refs, reason_code=REASON_MISSING_PER_SHARE_INPUTS),
            "reason_codes": reason_codes,
        }

    score_value = 0.0
    if _is_num(dilution_rate):
        dilution = float(dilution_rate)
        if dilution <= -0.02:
            score_value += 2.0
        elif dilution <= 0.01:
            score_value += 1.5
        elif dilution <= 0.03:
            score_value += 1.0
        elif dilution > 0.06:
            reason_codes.append(REASON_EXCESS_DILUTION)
        else:
            reason_codes.append(REASON_WEAK_PER_SHARE_CAPTURE)
    if _is_num(revenue_per_share_cagr):
        growth = float(revenue_per_share_cagr)
        if growth >= 0.08:
            score_value += 1.0
        elif growth >= 0.03:
            score_value += 0.5
    if _is_num(fcf_per_share_cagr):
        growth = float(fcf_per_share_cagr)
        if growth >= 0.08:
            score_value += 1.5
        elif growth >= 0.03:
            score_value += 1.0
        elif growth > 0.0:
            score_value += 0.5
    if _is_num(owner_per_share_cagr):
        growth = float(owner_per_share_cagr)
        if growth >= 0.08:
            score_value += 1.5
        elif growth >= 0.03:
            score_value += 1.0
        elif growth > 0.0:
            score_value += 0.5

    if (
        _is_num(revenue_per_share_cagr)
        and float(revenue_per_share_cagr) > 0.03
        and (
            (_is_num(fcf_per_share_cagr) and float(fcf_per_share_cagr) <= 0.0)
            or (_is_num(owner_per_share_cagr) and float(owner_per_share_cagr) <= 0.0)
        )
    ):
        reason_codes.append(REASON_WEAK_PER_SHARE_CAPTURE)

    score = max(0.0, min(5.0, round(score_value, 6)))
    if REASON_EXCESS_DILUTION in reason_codes:
        score = min(score, 2.0)
        reason_codes.append(REASON_WEAK_PER_SHARE_CAPTURE)
    if REASON_WEAK_PER_SHARE_CAPTURE in reason_codes:
        score = min(score, 2.0)
    if not reason_codes and score >= 4.0:
        reason_codes.append(REASON_STRONG_OWNER_VALUE_CAPTURE)
    elif score <= 1.0 and REASON_EXCESS_DILUTION not in reason_codes:
        reason_codes.append(REASON_WEAK_PER_SHARE_CAPTURE)

    reason_codes = _dedupe_refs(reason_codes)
    score_reason = reason_codes[0] if reason_codes else OK
    return {
        "dilution_rate_shares_cagr": _claim(value=dilution_rate, refs=dilution_refs or refs, reason_code=score_reason if not _is_num(dilution_rate) else OK),
        "revenue_per_share_cagr_proxy": _claim(value=revenue_per_share_cagr, refs=revenue_per_share_refs or refs, reason_code=score_reason if not _is_num(revenue_per_share_cagr) else OK),
        "fcf_per_share_cagr_proxy": _claim(value=fcf_per_share_cagr, refs=fcf_per_share_refs or refs, reason_code=score_reason if not _is_num(fcf_per_share_cagr) else OK),
        "owner_earnings_per_share_cagr_proxy": _claim(value=owner_per_share_cagr, refs=owner_per_share_refs or refs, reason_code=score_reason if not _is_num(owner_per_share_cagr) else OK),
        "owner_value_capture_score": _claim(value=score, refs=refs, reason_code=score_reason),
        "reason_codes": reason_codes,
    }


def _build_intangible_economics_payload(
    *,
    ticker: str,
    as_of_date: str,
    source_kind: str,
    series_map: dict[str, list[dict[str, Any]]],
    owner_quality_payload: dict[str, Any],
    net_debt_resolved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gross_margin = _gross_margin_durability(series_map)
    balance_sheet = _balance_sheet_optionality(
        series_map, net_debt_resolved=net_debt_resolved, source_kind=source_kind
    )
    cycle = _cycle_resilience(series_map, owner_quality_payload=owner_quality_payload)
    rnd_productivity = _rnd_productivity(series_map, owner_quality_payload=owner_quality_payload)
    sga_leverage = _sga_leverage(series_map)
    owner_value_capture = _owner_value_capture(series_map, owner_quality_payload=owner_quality_payload)

    oe_quality_total = owner_quality_payload.get("oe_quality_total", UNKNOWN)
    oe_support = 0.0
    if _is_num(oe_quality_total):
        if float(oe_quality_total) >= 12.0:
            oe_support = 2.0
        elif float(oe_quality_total) >= 8.0:
            oe_support = 1.0

    known_component_scores = [
        gross_margin["gross_margin_durability_score"].get("value"),
        balance_sheet["balance_sheet_optionality_score"].get("value"),
        cycle["cycle_resilience_score"].get("value"),
        rnd_productivity["rnd_productivity_score"].get("value"),
        sga_leverage["sga_leverage_score"].get("value"),
        owner_value_capture["owner_value_capture_score"].get("value"),
    ]
    known_component_scores = [float(value) for value in known_component_scores if _is_num(value)]
    intangible_total = max(0.0, min(20.0, round(sum(known_component_scores) + oe_support, 6))) if known_component_scores else UNKNOWN

    reason_codes = _dedupe_refs(
        list(gross_margin.get("reason_codes") or [])
        + list(balance_sheet.get("reason_codes") or [])
        + list(cycle.get("reason_codes") or [])
        + list(rnd_productivity.get("reason_codes") or [])
        + list(sga_leverage.get("reason_codes") or [])
        + list(owner_value_capture.get("reason_codes") or [])
    )
    all_refs = _dedupe_refs(
        list(gross_margin["gross_margin_avg_5y"].get("derived_from") or [])
        + list(gross_margin["gross_margin_volatility_5y"].get("derived_from") or [])
        + list(gross_margin["gross_margin_floor_5y"].get("derived_from") or [])
        + list(gross_margin["gross_margin_durability_score"].get("derived_from") or [])
        + list(balance_sheet["net_debt_proxy"].get("derived_from") or [])
        + list(balance_sheet["net_debt_to_cfo_proxy"].get("derived_from") or [])
        + list(balance_sheet["cash_pct_revenue"].get("derived_from") or [])
        + list(balance_sheet["balance_sheet_optionality_score"].get("derived_from") or [])
        + list(cycle["margin_volatility_5y"].get("derived_from") or [])
        + list(cycle["fcf_cfo_conversion_median_3y"].get("derived_from") or [])
        + list(cycle["fcf_cfo_conversion_volatility_3y"].get("derived_from") or [])
        + list(cycle["cycle_resilience_score"].get("derived_from") or [])
        + list(rnd_productivity["rnd_to_revenue_avg_5y"].get("derived_from") or [])
        + list(rnd_productivity["revenue_per_rnd_proxy"].get("derived_from") or [])
        + list(rnd_productivity["gross_profit_per_rnd_proxy"].get("derived_from") or [])
        + list(rnd_productivity["owner_earnings_per_rnd_proxy"].get("derived_from") or [])
        + list(rnd_productivity["rnd_productivity_score"].get("derived_from") or [])
        + list(sga_leverage["sga_to_revenue_avg_5y"].get("derived_from") or [])
        + list(sga_leverage["sga_growth_vs_revenue_growth_proxy"].get("derived_from") or [])
        + list(sga_leverage["operating_leverage_proxy"].get("derived_from") or [])
        + list(sga_leverage["sga_leverage_score"].get("derived_from") or [])
        + list(owner_value_capture["dilution_rate_shares_cagr"].get("derived_from") or [])
        + list(owner_value_capture["revenue_per_share_cagr_proxy"].get("derived_from") or [])
        + list(owner_value_capture["fcf_per_share_cagr_proxy"].get("derived_from") or [])
        + list(owner_value_capture["owner_earnings_per_share_cagr_proxy"].get("derived_from") or [])
        + list(owner_value_capture["owner_value_capture_score"].get("derived_from") or [])
        + [str(ref) for ref in (owner_quality_payload.get("derived_from") or []) if str(ref).strip()]
    )

    claims = {
        "gross_margin_avg_5y": gross_margin["gross_margin_avg_5y"],
        "gross_margin_volatility_5y": gross_margin["gross_margin_volatility_5y"],
        "gross_margin_floor_5y": gross_margin["gross_margin_floor_5y"],
        "gross_margin_durability_score": gross_margin["gross_margin_durability_score"],
        "net_debt_proxy": balance_sheet["net_debt_proxy"],
        "net_debt_to_cfo_proxy": balance_sheet["net_debt_to_cfo_proxy"],
        "cash_pct_revenue": balance_sheet["cash_pct_revenue"],
        "balance_sheet_optionality_score": balance_sheet["balance_sheet_optionality_score"],
        "margin_volatility_5y": cycle["margin_volatility_5y"],
        "fcf_cfo_conversion_median_3y": cycle["fcf_cfo_conversion_median_3y"],
        "fcf_cfo_conversion_volatility_3y": cycle["fcf_cfo_conversion_volatility_3y"],
        "cycle_resilience_score": cycle["cycle_resilience_score"],
        "rnd_to_revenue_avg_5y": rnd_productivity["rnd_to_revenue_avg_5y"],
        "revenue_per_rnd_proxy": rnd_productivity["revenue_per_rnd_proxy"],
        "gross_profit_per_rnd_proxy": rnd_productivity["gross_profit_per_rnd_proxy"],
        "owner_earnings_per_rnd_proxy": rnd_productivity["owner_earnings_per_rnd_proxy"],
        "rnd_productivity_score": rnd_productivity["rnd_productivity_score"],
        "sga_to_revenue_avg_5y": sga_leverage["sga_to_revenue_avg_5y"],
        "sga_growth_vs_revenue_growth_proxy": sga_leverage["sga_growth_vs_revenue_growth_proxy"],
        "operating_leverage_proxy": sga_leverage["operating_leverage_proxy"],
        "sga_leverage_score": sga_leverage["sga_leverage_score"],
        "dilution_rate_shares_cagr": owner_value_capture["dilution_rate_shares_cagr"],
        "revenue_per_share_cagr_proxy": owner_value_capture["revenue_per_share_cagr_proxy"],
        "fcf_per_share_cagr_proxy": owner_value_capture["fcf_per_share_cagr_proxy"],
        "owner_earnings_per_share_cagr_proxy": owner_value_capture["owner_earnings_per_share_cagr_proxy"],
        "owner_value_capture_score": owner_value_capture["owner_value_capture_score"],
        "intangible_economics_total": _claim(
            value=intangible_total,
            refs=all_refs,
            reason_code=reason_codes[0] if reason_codes else OK,
        ),
    }

    return {
        "ticker": str(ticker or "").strip().upper(),
        "as_of_date": str(as_of_date or ""),
        "source_kind": source_kind,
        "gross_margin_avg_5y": claims["gross_margin_avg_5y"].get("value", UNKNOWN),
        "gross_margin_volatility_5y": claims["gross_margin_volatility_5y"].get("value", UNKNOWN),
        "gross_margin_floor_5y": claims["gross_margin_floor_5y"].get("value", UNKNOWN),
        "gross_margin_durability_score": claims["gross_margin_durability_score"].get("value", UNKNOWN),
        "gross_margin_reason_codes": list(gross_margin.get("reason_codes") or []),
        "net_debt_proxy": claims["net_debt_proxy"].get("value", UNKNOWN),
        "net_debt_to_cfo_proxy": claims["net_debt_to_cfo_proxy"].get("value", UNKNOWN),
        "cash_pct_revenue": claims["cash_pct_revenue"].get("value", UNKNOWN),
        "balance_sheet_optionality_score": claims["balance_sheet_optionality_score"].get("value", UNKNOWN),
        "balance_sheet_optionality_reason_codes": list(balance_sheet.get("reason_codes") or []),
        "margin_volatility_5y": claims["margin_volatility_5y"].get("value", UNKNOWN),
        "fcf_cfo_conversion_median_3y": claims["fcf_cfo_conversion_median_3y"].get("value", UNKNOWN),
        "fcf_cfo_conversion_volatility_3y": claims["fcf_cfo_conversion_volatility_3y"].get("value", UNKNOWN),
        "cycle_resilience_score": claims["cycle_resilience_score"].get("value", UNKNOWN),
        "cycle_resilience_reason_codes": list(cycle.get("reason_codes") or []),
        "rnd_to_revenue_avg_5y": claims["rnd_to_revenue_avg_5y"].get("value", UNKNOWN),
        "revenue_per_rnd_proxy": claims["revenue_per_rnd_proxy"].get("value", UNKNOWN),
        "gross_profit_per_rnd_proxy": claims["gross_profit_per_rnd_proxy"].get("value", UNKNOWN),
        "owner_earnings_per_rnd_proxy": claims["owner_earnings_per_rnd_proxy"].get("value", UNKNOWN),
        "rnd_productivity_score": claims["rnd_productivity_score"].get("value", UNKNOWN),
        "rnd_productivity_reason_codes": list(rnd_productivity.get("reason_codes") or []),
        "sga_to_revenue_avg_5y": claims["sga_to_revenue_avg_5y"].get("value", UNKNOWN),
        "sga_growth_vs_revenue_growth_proxy": claims["sga_growth_vs_revenue_growth_proxy"].get("value", UNKNOWN),
        "operating_leverage_proxy": claims["operating_leverage_proxy"].get("value", UNKNOWN),
        "sga_leverage_score": claims["sga_leverage_score"].get("value", UNKNOWN),
        "sga_leverage_reason_codes": list(sga_leverage.get("reason_codes") or []),
        "dilution_rate_shares_cagr": claims["dilution_rate_shares_cagr"].get("value", UNKNOWN),
        "revenue_per_share_cagr_proxy": claims["revenue_per_share_cagr_proxy"].get("value", UNKNOWN),
        "fcf_per_share_cagr_proxy": claims["fcf_per_share_cagr_proxy"].get("value", UNKNOWN),
        "owner_earnings_per_share_cagr_proxy": claims["owner_earnings_per_share_cagr_proxy"].get("value", UNKNOWN),
        "owner_value_capture_score": claims["owner_value_capture_score"].get("value", UNKNOWN),
        "owner_value_capture_reason_codes": list(owner_value_capture.get("reason_codes") or []),
        "intangible_economics_total": claims["intangible_economics_total"].get("value", UNKNOWN),
        "intangible_economics_reason_codes": reason_codes,
        "claims": claims,
        "derived_from": all_refs,
        "generated_at": utc_now_iso(),
    }


def compute_intangible_economics(
    ticker: str,
    as_of_date: str,
    *,
    run_id: str | None = None,
    facts_row: dict[str, Any] | None = None,
    owner_payload: dict[str, Any] | None = None,
    owner_quality_payload: dict[str, Any] | None = None,
    net_debt_resolved: dict[str, Any] | None = None,
    fundamentals: dict[str, Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    ticker_norm = str(ticker or "").strip().upper()
    resolved_owner_quality = owner_quality_payload if isinstance(owner_quality_payload, dict) else {}
    if isinstance(fundamentals, dict) and fundamentals:
        if not resolved_owner_quality:
            resolved_owner_quality = compute_owner_earnings_quality(
                ticker=ticker_norm,
                as_of_date=as_of_date,
                fundamentals=fundamentals,
                cfg=cfg,
            )
        return _build_intangible_economics_payload(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            source_kind="FUNDAMENTALS",
            series_map=_build_series_from_fundamentals(fundamentals),
            owner_quality_payload=resolved_owner_quality,
            net_debt_resolved=net_debt_resolved,
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
    if not resolved_owner_quality:
        resolved_owner_quality = compute_owner_earnings_quality(
            ticker=ticker_norm,
            as_of_date=as_of_date,
            run_id=run_id,
            facts_row=base_facts_row,
            owner_payload=owner,
            cfg=cfg,
        )
    companyfacts = _load_companyfacts_payload(base_facts_row)
    series_map = _build_series_from_companyfacts(companyfacts=companyfacts, as_of_date=as_of_date) if companyfacts else {
        "gross_margin": [],
        "revenue": [],
        "cfo": [],
        "fcf": [],
        "net_debt": [],
        "cash": [],
    }
    if not (series_map.get("gross_margin") or []) and owner:
        owner_refs = [str(ref) for ref in (owner.get("derived_from") or []) if str(ref).strip()]
        series_map["gross_margin"] = []
        if owner_refs:
            series_map["gross_margin"] = []
    return _build_intangible_economics_payload(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        source_kind="COMPANYFACTS",
        series_map=series_map,
        owner_quality_payload=resolved_owner_quality,
        net_debt_resolved=net_debt_resolved,
    )


def write_intangible_economics_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    facts_rows_by_ticker: dict[str, dict[str, Any]] | None = None,
    fundamentals_by_ticker: dict[str, dict[str, Any]] | None = None,
    owner_earnings_quality_by_ticker: dict[str, dict[str, Any]] | None = None,
    net_debt_by_ticker: dict[str, dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("intangible_economics_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("intangible_economics_detail"), dict)
    }
    owner_quality_lookup = dict(owner_earnings_quality_by_ticker or {})
    for row in (scoreboard_rows or []):
        if not isinstance(row, dict):
            continue
        ticker_norm = str(row.get("ticker") or "").upper()
        owner_quality_detail = row.get("owner_earnings_quality_detail")
        if ticker_norm and isinstance(owner_quality_detail, dict) and ticker_norm not in owner_quality_lookup:
            owner_quality_lookup[ticker_norm] = owner_quality_detail

    rows: list[dict[str, Any]] = []
    for ticker in sorted({str(token or "").strip().upper() for token in tickers if str(token or "").strip()}):
        detail = detail_by_ticker.get(ticker)
        if not isinstance(detail, dict):
            detail = compute_intangible_economics(
                ticker=ticker,
                as_of_date=as_of_date,
                run_id=run_id,
                facts_row=(facts_rows_by_ticker or {}).get(ticker),
                owner_quality_payload=owner_quality_lookup.get(ticker),
                net_debt_resolved=(net_debt_by_ticker or {}).get(ticker),
                fundamentals=(fundamentals_by_ticker or {}).get(ticker),
                cfg=cfg,
            )
        rows.append(detail)

    known_count = len([row for row in rows if _is_num(row.get("intangible_economics_total"))])
    unknown_count = len(rows) - known_count
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("intangible_economics_total")) if _is_num(row.get("intangible_economics_total")) else float("inf"),
            str(row.get("ticker") or ""),
        ),
    )
    negative_reason_counts: dict[str, int] = {}
    rnd_unknown_reason_counts: dict[str, int] = {}
    sga_unknown_reason_counts: dict[str, int] = {}
    for row in rows:
        for code in [str(reason) for reason in (row.get("intangible_economics_reason_codes") or []) if str(reason).strip()]:
            if code in POSITIVE_REASON_CODES:
                continue
            negative_reason_counts[code] = negative_reason_counts.get(code, 0) + 1
        for code in [str(reason) for reason in (row.get("rnd_productivity_reason_codes") or []) if str(reason).strip()]:
            if code in {REASON_MISSING_RND_HISTORY, REASON_MISSING_RND_DENOMINATOR, REASON_INSUFFICIENT_RND_HISTORY, REASON_RND_PRODUCTIVITY_UNKNOWN}:
                rnd_unknown_reason_counts[code] = rnd_unknown_reason_counts.get(code, 0) + 1
        for code in [str(reason) for reason in (row.get("sga_leverage_reason_codes") or []) if str(reason).strip()]:
            if code in {REASON_MISSING_SGA_HISTORY, REASON_MISSING_OPERATING_LEVERAGE_INPUTS, REASON_INSUFFICIENT_SGA_HISTORY, REASON_SGA_LEVERAGE_UNKNOWN}:
                sga_unknown_reason_counts[code] = sga_unknown_reason_counts.get(code, 0) + 1
    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "known_count": known_count,
        "unknown_count": unknown_count,
        "top_10_by_intangible_economics_total": [
            {
                "ticker": str(row.get("ticker") or ""),
                "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
                "gross_margin_durability_score": row.get("gross_margin_durability_score", UNKNOWN),
                "balance_sheet_optionality_score": row.get("balance_sheet_optionality_score", UNKNOWN),
                "cycle_resilience_score": row.get("cycle_resilience_score", UNKNOWN),
                "rnd_productivity_score": row.get("rnd_productivity_score", UNKNOWN),
                "sga_leverage_score": row.get("sga_leverage_score", UNKNOWN),
                "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
                "intangible_economics_reason_codes": [str(code) for code in (row.get("intangible_economics_reason_codes") or []) if str(code).strip()],
            }
            for row in ranked[:10]
        ],
        "top_gross_margin_durability": [
            {
                "ticker": str(row.get("ticker") or ""),
                "gross_margin_durability_score": row.get("gross_margin_durability_score", UNKNOWN),
                "gross_margin_avg_5y": row.get("gross_margin_avg_5y", UNKNOWN),
                "gross_margin_floor_5y": row.get("gross_margin_floor_5y", UNKNOWN),
            }
            for row in sorted(
                rows,
                key=lambda row: (
                    -float(row.get("gross_margin_durability_score")) if _is_num(row.get("gross_margin_durability_score")) else float("inf"),
                    str(row.get("ticker") or ""),
                ),
            )[:10]
        ],
        "top_balance_sheet_optionality": [
            {
                "ticker": str(row.get("ticker") or ""),
                "balance_sheet_optionality_score": row.get("balance_sheet_optionality_score", UNKNOWN),
                "net_debt_to_cfo_proxy": row.get("net_debt_to_cfo_proxy", UNKNOWN),
                "cash_pct_revenue": row.get("cash_pct_revenue", UNKNOWN),
            }
            for row in sorted(
                rows,
                key=lambda row: (
                    -float(row.get("balance_sheet_optionality_score")) if _is_num(row.get("balance_sheet_optionality_score")) else float("inf"),
                    str(row.get("ticker") or ""),
                ),
            )[:10]
        ],
        "top_owner_value_capture": [
            {
                "ticker": str(row.get("ticker") or ""),
                "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
                "dilution_rate_shares_cagr": row.get("dilution_rate_shares_cagr", UNKNOWN),
                "revenue_per_share_cagr_proxy": row.get("revenue_per_share_cagr_proxy", UNKNOWN),
                "fcf_per_share_cagr_proxy": row.get("fcf_per_share_cagr_proxy", UNKNOWN),
                "owner_earnings_per_share_cagr_proxy": row.get("owner_earnings_per_share_cagr_proxy", UNKNOWN),
                "owner_value_capture_reason_codes": [str(code) for code in (row.get("owner_value_capture_reason_codes") or []) if str(code).strip()],
            }
            for row in sorted(
                rows,
                key=lambda row: (
                    -float(row.get("owner_value_capture_score")) if _is_num(row.get("owner_value_capture_score")) else float("inf"),
                    str(row.get("ticker") or ""),
                ),
            )[:10]
        ],
        "rnd_unknown_reason_counts": dict(sorted(rnd_unknown_reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))),
        "sga_unknown_reason_counts": dict(sorted(sga_unknown_reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))),
        "strong_gross_margin_weak_owner_capture": [
            {
                "ticker": str(row.get("ticker") or ""),
                "gross_margin_durability_score": row.get("gross_margin_durability_score", UNKNOWN),
                "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
                "owner_value_capture_reason_codes": [str(code) for code in (row.get("owner_value_capture_reason_codes") or []) if str(code).strip()],
            }
            for row in sorted(
                [
                    row
                    for row in rows
                    if _is_num(row.get("gross_margin_durability_score"))
                    and float(row.get("gross_margin_durability_score")) >= 4.0
                    and (
                        (not _is_num(row.get("owner_value_capture_score")))
                        or float(row.get("owner_value_capture_score")) <= 2.0
                        or any(
                            code in {REASON_EXCESS_DILUTION, REASON_WEAK_PER_SHARE_CAPTURE}
                            for code in (row.get("owner_value_capture_reason_codes") or [])
                        )
                    )
                ],
                key=lambda row: (
                    -float(row.get("gross_margin_durability_score")) if _is_num(row.get("gross_margin_durability_score")) else float("inf"),
                    str(row.get("ticker") or ""),
                ),
            )[:10]
        ],
        "strong_total_acceptable_dilution": [
            {
                "ticker": str(row.get("ticker") or ""),
                "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
                "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
                "dilution_rate_shares_cagr": row.get("dilution_rate_shares_cagr", UNKNOWN),
            }
            for row in sorted(
                [
                    row
                    for row in rows
                    if _is_num(row.get("intangible_economics_total"))
                    and float(row.get("intangible_economics_total")) >= 12.0
                    and not any(
                        code == REASON_EXCESS_DILUTION
                        for code in (row.get("owner_value_capture_reason_codes") or [])
                    )
                ],
                key=lambda row: (
                    -float(row.get("intangible_economics_total")) if _is_num(row.get("intangible_economics_total")) else float("inf"),
                    -float(row.get("owner_value_capture_score")) if _is_num(row.get("owner_value_capture_score")) else float("inf"),
                    str(row.get("ticker") or ""),
                ),
            )[:10]
        ],
        "negative_reason_counts": dict(sorted(negative_reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))),
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    payload["intangible_economics_path"] = str(output_path)
    return payload


def _intangible_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "intangible_economics.json",
        cfg.sectors_dir / run_id / "intangible_economics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_intangible_economics(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    path = _intangible_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "intangible_economics_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or len(rows)),
        "known_count": int(payload.get("known_count") or 0),
        "unknown_count": int(payload.get("unknown_count") or 0),
        "top_10_by_intangible_economics_total": [
            row for row in (payload.get("top_10_by_intangible_economics_total") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_gross_margin_durability": [
            row for row in (payload.get("top_gross_margin_durability") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_balance_sheet_optionality": [
            row for row in (payload.get("top_balance_sheet_optionality") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "top_owner_value_capture": [
            row for row in (payload.get("top_owner_value_capture") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "rnd_unknown_reason_counts": payload.get("rnd_unknown_reason_counts") if isinstance(payload.get("rnd_unknown_reason_counts"), dict) else {},
        "sga_unknown_reason_counts": payload.get("sga_unknown_reason_counts") if isinstance(payload.get("sga_unknown_reason_counts"), dict) else {},
        "strong_gross_margin_weak_owner_capture": [
            row for row in (payload.get("strong_gross_margin_weak_owner_capture") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "strong_total_acceptable_dilution": [
            row for row in (payload.get("strong_total_acceptable_dilution") or []) if isinstance(row, dict)
        ][: max(1, int(top_n))],
        "negative_reason_counts": payload.get("negative_reason_counts") if isinstance(payload.get("negative_reason_counts"), dict) else {},
        "intangible_economics_path": str(path),
    }
