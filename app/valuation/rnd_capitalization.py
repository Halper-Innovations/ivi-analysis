from __future__ import annotations

import logging
import math
from statistics import mean
from typing import Any

from app.config import AppConfig, get_config
from app.valuation.facts import resolve_financial_facts_asof
from app.valuation.intangible_economics import (
    UNKNOWN,
    _CAPEX_TAG_PRIORITY,
    _GROSS_PROFIT_TAG_PRIORITY,
    _OPERATING_INCOME_TAG_PRIORITY,
    _RND_TAG_PRIORITY,
    _REVENUE_TAG_PRIORITY,
    _dedupe_refs,
    _is_num,
    _series_from_companyfacts,
    compute_intangible_economics,
)
from app.valuation.owner_earnings import _CFO_TAG_PRIORITY, _load_companyfacts_payload
from app.valuation.tech_category import (
    CONSUMER_HARDWARE,
    ENTERPRISE_SOFTWARE,
    INDUSTRIAL_TECH,
    NETWORK_INFRA,
    PLATFORM_HYBRID,
    SEMICONDUCTOR,
    TRADITIONAL_OPERATING,
)


STATUS_OK = "OK"
STATUS_NO_FACTS = "NO_COMPANYFACTS"
STATUS_INSUFFICIENT_RND_HISTORY = "INSUFFICIENT_RND_HISTORY"
STATUS_SKIPPED_LOW_RND_INTENSITY = "SKIPPED_LOW_RND_INTENSITY"
STATUS_SKIPPED_LOW_GROSS_MARGIN = "SKIPPED_LOW_GROSS_MARGIN"

FLAG_LOW_RND_PRODUCTIVITY_WARNING = "LOW_RND_PRODUCTIVITY_WARNING"
FLAG_RND_ADJUSTMENT_CAPPED = "RND_ADJUSTMENT_CAPPED"
FLAG_INPUTS_SCALED_TO_MILLIONS = "INPUTS_SCALED_TO_MILLIONS"

_AMORTIZATION_LIVES = {
    ENTERPRISE_SOFTWARE: 3,
    SEMICONDUCTOR: 4,
    CONSUMER_HARDWARE: 2,
    NETWORK_INFRA: 3,
    PLATFORM_HYBRID: 3,
    INDUSTRIAL_TECH: 4,
}
_SBC_TAG_PRIORITY: list[tuple[str, str]] = [
    ("us-gaap", "ShareBasedCompensation"),
    ("us-gaap", "AllocatedShareBasedCompensationExpense"),
]

logger = logging.getLogger(__name__)


def _resolve_companyfacts(
    *,
    ticker: str,
    as_of_date: str,
    companyfacts: dict[str, Any] | None,
    facts_row: dict[str, Any] | None,
    cfg: AppConfig | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    refs = [str(ref) for ref in (facts_row or {}).get("derived_from", []) if str(ref).strip()]
    if isinstance(companyfacts, dict) and isinstance(companyfacts.get("companyfacts"), dict):
        return companyfacts["companyfacts"], facts_row, _dedupe_refs(refs)
    if isinstance(companyfacts, dict) and isinstance(companyfacts.get("facts"), dict):
        return companyfacts, facts_row, _dedupe_refs(refs)

    cfg = cfg or get_config()
    resolved_row = (
        facts_row
        if isinstance(facts_row, dict)
        else resolve_financial_facts_asof(
            ticker=str(ticker or "").strip().upper(),
            as_of_date=as_of_date,
            refresh=False,
            cfg=cfg,
        )
    )
    loaded = _load_companyfacts_payload(resolved_row)
    refs = refs + [str(ref) for ref in resolved_row.get("derived_from", []) if str(ref).strip()]
    return loaded, resolved_row, _dedupe_refs(refs)


def _metric_series(
    *,
    companyfacts: dict[str, Any],
    as_of_date: str,
    priority: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    return _series_from_companyfacts(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=priority,
        expected_unit_exact=("usd",),
    )


def _divide_rows_by_millions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scaled: list[dict[str, Any]] = []
    for row in rows:
        scaled_row = dict(row)
        if _is_num(scaled_row.get("value")):
            scaled_row["value"] = float(scaled_row["value"]) / 1_000_000.0
        scaled.append(scaled_row)
    return scaled


def _scale_series_groups_to_millions(
    groups: tuple[list[dict[str, Any]], ...],
) -> tuple[list[list[dict[str, Any]]], float]:
    """Convert exact raw-USD CompanyFacts series to USD millions once.

    ``_metric_series`` accepts only the literal ``USD`` unit.  Magnitude is
    therefore irrelevant: a $900,000 series and a $900,000,000 series cross
    the same deterministic ingestion boundary.
    """

    return [_divide_rows_by_millions(rows) for rows in groups], 1_000_000.0


def _series_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        year = int(row.get("year") or 0)
        if year <= 0 or not _is_num(row.get("value")):
            continue
        out[year] = {
            "value": float(row["value"]),
            "derived_from": [
                str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()
            ],
        }
    return out


def _recent_ratio_average(
    numerator: list[dict[str, Any]],
    denominator: list[dict[str, Any]],
    *,
    window: int = 3,
) -> tuple[float | str, list[str]]:
    denom_by_year = _series_map(denominator)
    ratios: list[tuple[int, float, list[str]]] = []
    for row in numerator:
        year = int(row.get("year") or 0)
        denom = denom_by_year.get(year)
        if year <= 0 or not isinstance(denom, dict):
            continue
        if not _is_num(row.get("value")) or not _is_num(denom.get("value")):
            continue
        denom_value = float(denom["value"])
        if denom_value <= 0.0:
            continue
        ratios.append(
            (
                year,
                float(row["value"]) / denom_value,
                _dedupe_refs(
                    list(row.get("derived_from") or []) + list(denom.get("derived_from") or [])
                ),
            )
        )
    ratios = sorted(ratios, key=lambda item: item[0])[-max(1, int(window)) :]
    if not ratios:
        return UNKNOWN, []
    return round(mean(value for _year, value, _refs in ratios), 6), _dedupe_refs(
        [ref for _year, _value, refs in ratios for ref in refs]
    )


def _normalize_capex(series: list[tuple[int, float]], n: int = 5) -> float:
    # Capex as a MAGNITUDE (same convention as valuation_writer._normalize_capex):
    # a filer-negated series would otherwise invert the spike cap and turn the
    # owner-earnings deduction below into an addition.
    values = [
        abs(value) for _year, value in sorted(series, key=lambda item: item[0], reverse=True)[:n]
    ]
    if not values:
        return 0.0
    uncapped_mean = sum(values) / len(values)
    threshold = 2.0 * uncapped_mean
    capped = [min(value, uncapped_mean) if value > threshold else value for value in values]
    return sum(capped) / len(capped)


def _owner_earnings_by_year(
    *,
    cfo_rows: list[dict[str, Any]],
    capex_rows: list[dict[str, Any]],
    sbc_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    cfo_by_year = _series_map(cfo_rows)
    capex_by_year = _series_map(capex_rows)
    sbc_by_year = _series_map(sbc_rows)
    years = sorted(set(cfo_by_year) | set(capex_by_year) | set(sbc_by_year))
    out: dict[int, dict[str, Any]] = {}
    for year in years:
        cfo = cfo_by_year.get(year)
        if not isinstance(cfo, dict) or not _is_num(cfo.get("value")):
            continue
        hist_capex = [
            (candidate_year, float(row["value"]))
            for candidate_year, row in capex_by_year.items()
            if candidate_year <= year and _is_num(row.get("value"))
        ]
        normalized_capex = _normalize_capex(hist_capex) if hist_capex else 0.0
        sbc = sbc_by_year.get(year)
        sbc_value = (
            float(sbc["value"]) if isinstance(sbc, dict) and _is_num(sbc.get("value")) else 0.0
        )
        refs = list(cfo.get("derived_from") or [])
        refs.extend(
            ref
            for _hist_year, row in capex_by_year.items()
            if _hist_year <= year
            for ref in list(row.get("derived_from") or [])
        )
        if isinstance(sbc, dict):
            refs.extend(list(sbc.get("derived_from") or []))
        out[year] = {
            "value": float(cfo["value"]) - float(normalized_capex) - sbc_value,
            "normalized_capex": normalized_capex,
            "sbc_used": sbc_value,
            "derived_from": _dedupe_refs(refs),
        }
    return out


def _fundamentals_payload_for_productivity(
    *,
    ticker: str,
    as_of_date: str,
    revenue_rows: list[dict[str, Any]],
    gross_profit_rows: list[dict[str, Any]],
    cfo_rows: list[dict[str, Any]],
    capex_rows: list[dict[str, Any]],
    rnd_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    revenue_by_year = _series_map(revenue_rows)
    gross_profit_by_year = _series_map(gross_profit_rows)
    cfo_by_year = _series_map(cfo_rows)
    capex_by_year = _series_map(capex_rows)
    rnd_by_year = _series_map(rnd_rows)
    years = sorted(
        set(revenue_by_year)
        | set(gross_profit_by_year)
        | set(cfo_by_year)
        | set(capex_by_year)
        | set(rnd_by_year)
    )
    rows: list[dict[str, Any]] = []
    row_traces: dict[str, dict[str, dict[str, list[str]]]] = {}
    for year in years:
        row: dict[str, Any] = {"year": year}
        traces: dict[str, dict[str, list[str]]] = {}
        for field_name, series_map in (
            ("revenue", revenue_by_year),
            ("gross_profit", gross_profit_by_year),
            ("cfo", cfo_by_year),
            ("capex", capex_by_year),
            ("r_and_d_total", rnd_by_year),
        ):
            metric_row = series_map.get(year)
            if not isinstance(metric_row, dict) or not _is_num(metric_row.get("value")):
                continue
            row[field_name] = float(metric_row["value"])
            traces[field_name] = {"derived_from": list(metric_row.get("derived_from") or [])}
        if len(row) > 1:
            rows.append(row)
            row_traces[str(year)] = traces
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "rows": rows,
        "row_traces": row_traces,
        "derived_signals": {},
    }


def _rnd_productivity_score(
    *,
    ticker: str,
    as_of_date: str,
    revenue_rows: list[dict[str, Any]],
    gross_profit_rows: list[dict[str, Any]],
    cfo_rows: list[dict[str, Any]],
    capex_rows: list[dict[str, Any]],
    rnd_rows: list[dict[str, Any]],
    cfg: AppConfig | None,
) -> tuple[float | str, list[str]]:
    fundamentals = _fundamentals_payload_for_productivity(
        ticker=ticker,
        as_of_date=as_of_date,
        revenue_rows=revenue_rows,
        gross_profit_rows=gross_profit_rows,
        cfo_rows=cfo_rows,
        capex_rows=capex_rows,
        rnd_rows=rnd_rows,
    )
    if not fundamentals.get("rows"):
        return UNKNOWN, []
    payload = compute_intangible_economics(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamentals=fundamentals,
        cfg=cfg,
    )
    refs = [
        str(ref)
        for ref in (payload.get("rnd_productivity_score_derived_from") or [])
        if str(ref).strip()
    ]
    return payload.get("rnd_productivity_score", UNKNOWN), _dedupe_refs(refs)


def compute_rnd_adjusted_earnings(
    ticker: str,
    as_of_date: str,
    *,
    category: str,
    companyfacts: dict[str, Any] | None = None,
    facts_row: dict[str, Any] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any] | None:
    ticker_norm = str(ticker or "").strip().upper()
    logger.info(
        "rnd_adjustment start ticker=%s as_of=%s category=%s",
        ticker_norm,
        as_of_date,
        category,
    )
    if category == TRADITIONAL_OPERATING:
        logger.info("rnd_adjustment skipped ticker=%s reason=TRADITIONAL_OPERATING", ticker_norm)
        return None

    amortization_life = _AMORTIZATION_LIVES.get(str(category or "").strip().upper())
    if amortization_life is None:
        logger.info("rnd_adjustment skipped ticker=%s reason=UNKNOWN_CATEGORY", ticker_norm)
        return None

    resolved_companyfacts, resolved_facts_row, base_refs = _resolve_companyfacts(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        companyfacts=companyfacts,
        facts_row=facts_row,
        cfg=cfg,
    )
    if not resolved_companyfacts:
        logger.info("rnd_adjustment ticker=%s status=%s", ticker_norm, STATUS_NO_FACTS)
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "category": category,
            "status": STATUS_NO_FACTS,
            "amortization_life": amortization_life,
            "flags": [],
            "time_series": [],
            "derived_from": base_refs,
        }

    rnd_rows = _metric_series(
        companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_RND_TAG_PRIORITY
    )
    revenue_rows = _metric_series(
        companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_REVENUE_TAG_PRIORITY
    )
    gross_profit_rows = _metric_series(
        companyfacts=resolved_companyfacts,
        as_of_date=as_of_date,
        priority=_GROSS_PROFIT_TAG_PRIORITY,
    )
    operating_income_rows = _metric_series(
        companyfacts=resolved_companyfacts,
        as_of_date=as_of_date,
        priority=_OPERATING_INCOME_TAG_PRIORITY,
    )
    cfo_rows = _metric_series(
        companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_CFO_TAG_PRIORITY
    )
    capex_rows = _metric_series(
        companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_CAPEX_TAG_PRIORITY
    )
    sbc_rows = _metric_series(
        companyfacts=resolved_companyfacts, as_of_date=as_of_date, priority=_SBC_TAG_PRIORITY
    )

    scaled_groups, unit_scale = _scale_series_groups_to_millions(
        (
            rnd_rows,
            revenue_rows,
            gross_profit_rows,
            operating_income_rows,
            cfo_rows,
            capex_rows,
            sbc_rows,
        )
    )
    (
        rnd_rows,
        revenue_rows,
        gross_profit_rows,
        operating_income_rows,
        cfo_rows,
        capex_rows,
        sbc_rows,
    ) = scaled_groups

    avg_rnd_to_revenue, rnd_revenue_refs = _recent_ratio_average(rnd_rows, revenue_rows, window=3)
    avg_gross_margin, gross_margin_refs = _recent_ratio_average(
        gross_profit_rows, revenue_rows, window=3
    )

    guardrail_refs = _dedupe_refs(base_refs + rnd_revenue_refs + gross_margin_refs)
    if _is_num(avg_rnd_to_revenue) and float(avg_rnd_to_revenue) < 0.03:
        logger.info(
            "rnd_adjustment ticker=%s status=%s avg_rnd_to_revenue=%s",
            ticker_norm,
            STATUS_SKIPPED_LOW_RND_INTENSITY,
            avg_rnd_to_revenue,
        )
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "category": category,
            "status": STATUS_SKIPPED_LOW_RND_INTENSITY,
            "amortization_life": amortization_life,
            "flags": [],
            "guardrails": {
                "avg_rnd_to_revenue_3y": float(avg_rnd_to_revenue),
                "avg_gross_margin_3y": avg_gross_margin,
            },
            "time_series": [],
            "derived_from": guardrail_refs,
        }
    if _is_num(avg_gross_margin) and float(avg_gross_margin) < 0.30:
        logger.info(
            "rnd_adjustment ticker=%s status=%s avg_gross_margin=%s",
            ticker_norm,
            STATUS_SKIPPED_LOW_GROSS_MARGIN,
            avg_gross_margin,
        )
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "category": category,
            "status": STATUS_SKIPPED_LOW_GROSS_MARGIN,
            "amortization_life": amortization_life,
            "flags": [],
            "guardrails": {
                "avg_rnd_to_revenue_3y": avg_rnd_to_revenue,
                "avg_gross_margin_3y": float(avg_gross_margin),
            },
            "time_series": [],
            "derived_from": guardrail_refs,
        }

    rnd_by_year = _series_map(rnd_rows)
    if len(rnd_by_year) < (amortization_life + 1):
        logger.info(
            "rnd_adjustment ticker=%s status=%s rnd_history_years=%s amortization_life=%s",
            ticker_norm,
            STATUS_INSUFFICIENT_RND_HISTORY,
            len(rnd_by_year),
            amortization_life,
        )
        return {
            "ticker": ticker_norm,
            "as_of_date": as_of_date,
            "category": category,
            "status": STATUS_INSUFFICIENT_RND_HISTORY,
            "amortization_life": amortization_life,
            "flags": [],
            "guardrails": {
                "avg_rnd_to_revenue_3y": avg_rnd_to_revenue,
                "avg_gross_margin_3y": avg_gross_margin,
                "rnd_history_years": len(rnd_by_year),
            },
            "time_series": [],
            "derived_from": guardrail_refs,
        }

    operating_income_by_year = _series_map(operating_income_rows)
    owner_earnings_by_year = _owner_earnings_by_year(
        cfo_rows=cfo_rows, capex_rows=capex_rows, sbc_rows=sbc_rows
    )
    productivity_score, productivity_refs = _rnd_productivity_score(
        ticker=ticker_norm,
        as_of_date=as_of_date,
        revenue_rows=revenue_rows,
        gross_profit_rows=gross_profit_rows,
        cfo_rows=cfo_rows,
        capex_rows=capex_rows,
        rnd_rows=rnd_rows,
        cfg=cfg,
    )

    years = sorted(rnd_by_year)
    time_series: list[dict[str, Any]] = []
    overall_flags: list[str] = []
    if unit_scale > 1.0:
        overall_flags.append(FLAG_INPUTS_SCALED_TO_MILLIONS)
    if _is_num(productivity_score) and float(productivity_score) < 1.0:
        overall_flags.append(FLAG_LOW_RND_PRODUCTIVITY_WARNING)

    for year in years:
        rnd_row = rnd_by_year.get(year)
        if not isinstance(rnd_row, dict) or not _is_num(rnd_row.get("value")):
            continue
        current_rnd = float(rnd_row["value"])
        amortization_raw = 0.0
        capitalized_asset = 0.0
        for vintage_year, vintage_row in rnd_by_year.items():
            if not _is_num(vintage_row.get("value")) or vintage_year > year:
                continue
            age = year - vintage_year
            vintage_value = float(vintage_row["value"])
            if age == 0:
                capitalized_asset += vintage_value
                continue
            if age <= amortization_life:
                amortization_raw += vintage_value / float(amortization_life)
                capitalized_asset += vintage_value * max(
                    0.0, 1.0 - (age / float(amortization_life))
                )
        raw_adjustment = current_rnd - amortization_raw

        operating_income_row = operating_income_by_year.get(year)
        gaap_operating_income = (
            float(operating_income_row["value"])
            if isinstance(operating_income_row, dict) and _is_num(operating_income_row.get("value"))
            else UNKNOWN
        )
        applied_adjustment = raw_adjustment
        row_flags: list[str] = []
        if _is_num(gaap_operating_income):
            cap_limit = 0.40 * abs(float(gaap_operating_income))
            if abs(raw_adjustment) > cap_limit:
                applied_adjustment = math.copysign(cap_limit, raw_adjustment)
                row_flags.append(FLAG_RND_ADJUSTMENT_CAPPED)

        owner_row = owner_earnings_by_year.get(year)
        gaap_owner_earnings = (
            float(owner_row["value"])
            if isinstance(owner_row, dict) and _is_num(owner_row.get("value"))
            else UNKNOWN
        )
        adjusted_operating_income = (
            float(gaap_operating_income) + float(applied_adjustment)
            if _is_num(gaap_operating_income)
            else UNKNOWN
        )
        adjusted_owner_earnings = (
            float(gaap_owner_earnings) + float(applied_adjustment)
            if _is_num(gaap_owner_earnings)
            else UNKNOWN
        )

        refs = list(rnd_row.get("derived_from") or [])
        if isinstance(operating_income_row, dict):
            refs.extend(list(operating_income_row.get("derived_from") or []))
        if isinstance(owner_row, dict):
            refs.extend(list(owner_row.get("derived_from") or []))
        refs.extend(productivity_refs)
        # Vintage completeness: a year whose amortization stack would need
        # R&D vintages predating the available history carries NO charge for
        # the missing vintages, overstating its adjustment (audit:
        # rnd-vintage-boundary-overstates-early-years). Consumers (the
        # valuation_writer merge) skip incomplete years.
        vintage_complete = all(
            vintage_year in rnd_by_year
            for vintage_year in range(year - amortization_life, year + 1)
        )
        time_series.append(
            {
                "year": year,
                "vintage_complete": vintage_complete,
                "rnd_expense": current_rnd,
                "rnd_capitalized_asset": round(capitalized_asset, 6) if vintage_complete else UNKNOWN,
                "rnd_amortization_current": round(amortization_raw, 6) if vintage_complete else UNKNOWN,
                "rnd_adjustment_pre_cap": round(raw_adjustment, 6) if vintage_complete else UNKNOWN,
                "rnd_adjustment": round(applied_adjustment, 6) if vintage_complete else UNKNOWN,
                "gaap_operating_income": gaap_operating_income,
                "adjusted_operating_income": round(adjusted_operating_income, 6)
                if vintage_complete and _is_num(adjusted_operating_income)
                else UNKNOWN,
                "gaap_owner_earnings": gaap_owner_earnings,
                "adjusted_owner_earnings": round(adjusted_owner_earnings, 6)
                if vintage_complete and _is_num(adjusted_owner_earnings)
                else UNKNOWN,
                "flags": _dedupe_refs(row_flags + overall_flags),
                "derived_from": _dedupe_refs(refs),
            }
        )

    # Current means the latest observed annual financial year, not merely the
    # last year with R&D. Keep older complete stacks in time_series, but do not
    # expose their adjustment as current when the latest R&D expense is absent.
    latest_year = max(int(row["year"]) for rows in scaled_groups for row in rows)
    latest_row = next(
        (row for row in reversed(time_series) if row["year"] == latest_year),
        {
            "year": latest_year,
            "gaap_operating_income": operating_income_by_year.get(latest_year, {}).get(
                "value", UNKNOWN
            ),
            "gaap_owner_earnings": owner_earnings_by_year.get(latest_year, {}).get("value", UNKNOWN),
        },
    )
    # The writer admits this payload only when status is OK. An older complete
    # stack must not supply the missing latest adjustment through its fallback.
    result_status = (
        STATUS_OK if latest_row.get("vintage_complete") else STATUS_INSUFFICIENT_RND_HISTORY
    )
    logger.info(
        "rnd_adjustment ticker=%s status=%s amortization_life=%s unit_scale=%s latest_year=%s rnd_adjustment=%s capitalized_asset=%s amortization=%s",
        ticker_norm,
        result_status,
        amortization_life,
        unit_scale,
        latest_row.get("year"),
        latest_row.get("rnd_adjustment", UNKNOWN),
        latest_row.get("rnd_capitalized_asset", UNKNOWN),
        latest_row.get("rnd_amortization_current", UNKNOWN),
    )
    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "category": category,
        "status": result_status,
        "amortization_life": amortization_life,
        "latest_year": latest_row.get("year"),
        "gaap_operating_income": latest_row.get("gaap_operating_income", UNKNOWN),
        "adjusted_operating_income": latest_row.get("adjusted_operating_income", UNKNOWN),
        "gaap_owner_earnings": latest_row.get("gaap_owner_earnings", UNKNOWN),
        "adjusted_owner_earnings": latest_row.get("adjusted_owner_earnings", UNKNOWN),
        "rnd_adjustment": latest_row.get("rnd_adjustment", UNKNOWN),
        "rnd_capitalized_asset": latest_row.get("rnd_capitalized_asset", UNKNOWN),
        "rnd_amortization_current": latest_row.get("rnd_amortization_current", UNKNOWN),
        "flags": _dedupe_refs(
            overall_flags + [flag for row in time_series for flag in row.get("flags", [])]
        ),
        "guardrails": {
            "avg_rnd_to_revenue_3y": avg_rnd_to_revenue,
            "avg_gross_margin_3y": avg_gross_margin,
            "rnd_productivity_score": productivity_score,
            "input_unit": "USD",
            "output_unit": "USD_millions",
            "input_unit_scale": unit_scale,
        },
        "time_series": time_series,
        "derived_from": _dedupe_refs(
            guardrail_refs
            + productivity_refs
            + [ref for row in time_series for ref in row.get("derived_from", [])]
            + [
                str(ref)
                for ref in (resolved_facts_row or {}).get("derived_from", [])
                if str(ref).strip()
            ]
        ),
    }
