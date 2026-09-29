from __future__ import annotations

import json
from typing import Any

from app.config import get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"
WHALE_SIGNALS_VERSION = "v1.1"

WHALE_SIGNAL_WEIGHTS: dict[str, float] = {
    "growth_persistence": 22.0,
    "margin_expansion": 16.0,
    "fcf_inflection": 18.0,
    "dilution_discipline": 14.0,
    "reinvestment_capacity": 15.0,
    "balance_sheet_resilience": 15.0,
}

MISSING_METRIC_PENALTY = 2.0
CRITICAL_SIGNAL_KEYS = {"growth_persistence", "fcf_inflection", "balance_sheet_resilience"}

SIGNAL_SPECS: dict[str, dict[str, Any]] = {
    "growth_persistence": {
        "why_it_matters": "Durable multi-year growth with no recent collapse is a core prerequisite for future mega-compounders.",
        "thresholds": {
            "pass": {
                "revenue_cagr_5y_min": 0.10,
                "revenue_cagr_10y_min": 0.08,
                "revenue_cagr_3y_min": 0.04,
            },
            "partial": {
                "revenue_cagr_5y_min": 0.06,
                "revenue_cagr_10y_min": 0.04,
                "revenue_cagr_3y_min": 0.00,
            },
        },
    },
    "margin_expansion": {
        "why_it_matters": "Sustained margin expansion often indicates improving moat, mix, and operating leverage over time.",
        "thresholds": {
            "pass": {"gross_margin_trend_slope_gt": 0.0, "operating_margin_trend_slope_gt": 0.0},
            "partial": {"any_margin_trend_slope_gt": 0.0},
        },
    },
    "fcf_inflection": {
        "why_it_matters": "FCF trend plus stable cash-conversion quality helps separate accounting growth from compounding economics.",
        "thresholds": {
            "pass": {"fcf_margin_trend_slope_gt": 0.0, "cfo_to_net_income_quality": "STABLE"},
            "partial": {"fcf_margin_trend_slope_gte": 0.0, "cfo_to_net_income_quality": ["STABLE", "MIXED"]},
        },
    },
    "dilution_discipline": {
        "why_it_matters": "Per-share compounding is hard when share count expands too quickly for long periods.",
        "thresholds": {
            "pass": {"shares_cagr_10y_max": 0.02, "or_shares_cagr_5y_lte": 0.0},
            "partial": {"shares_cagr_10y_max": 0.04},
        },
    },
    "reinvestment_capacity": {
        "why_it_matters": "Growing internally funded reinvestment capacity supports long runway without balance-sheet strain.",
        "thresholds": {
            "pass": {"cfo_cagr_5y_min": 0.05, "capex_to_revenue_ratio_growth_max": 0.35},
            "partial": {"cfo_cagr_5y_min": 0.0, "capex_to_revenue_ratio_growth_max": 0.80},
        },
    },
    "balance_sheet_resilience": {
        "why_it_matters": "Balance-sheet deterioration can break compounding trajectories even when growth appears strong.",
        "thresholds": {
            "pass": {"net_debt_delta_max": 0.0},
            "partial": {"net_debt_delta_max": "25% of starting net debt"},
        },
    },
}


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _fmt_pct(value: float | str) -> str:
    if not _is_num(value):
        return "UNKNOWN"
    return f"{float(value) * 100:.2f}%"


def _safe_div(numerator: Any, denominator: Any) -> float | str:
    if not _is_num(numerator) or not _is_num(denominator) or float(denominator) == 0.0:
        return UNKNOWN
    return float(numerator) / float(denominator)


def _derived_signal_lookup(dossier: dict[str, Any], signal_name: str) -> tuple[float | str, list[str]]:
    derived = (dossier.get("time_series") or {}).get("derived_signals") or []
    for row in derived:
        if row.get("signal") != signal_name:
            continue
        value = row.get("value")
        trace = [str(x) for x in (row.get("derived_from") or []) if str(x).strip()]
        if not trace:
            trace = [f"dossier.time_series.derived_signals.{signal_name}"]
        return (float(value) if _is_num(value) else UNKNOWN), trace
    return UNKNOWN, [f"dossier.time_series.derived_signals.{signal_name}"]


def _std_rows(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (dossier.get("time_series") or {}).get("standardized_rows") or []
    normalized = [row for row in rows if isinstance(row, dict)]
    return sorted(normalized, key=lambda row: int(row.get("year", 0)))


def _metric_refs(dossier: dict[str, Any], metric: str, years: list[int] | None = None) -> list[str]:
    rows = _std_rows(dossier)
    trace_map = (dossier.get("time_series") or {}).get("standardized_row_traces") or {}
    refs: list[str] = []
    years_set = set(years or [])
    for row in rows:
        year = int(row.get("year", 0))
        if years and year not in years_set:
            continue
        trace_bucket = trace_map.get(str(year), {}).get(metric, {})
        trace_derived = [str(x) for x in (trace_bucket.get("derived_from") or []) if str(x).strip()]
        if trace_derived:
            refs.extend(trace_derived)
        else:
            refs.append(f"dossier.time_series.standardized_rows[{year}].{metric}")
    if not refs:
        refs.append(f"dossier.time_series.standardized_rows[*].{metric}")
    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        deduped.append(ref)
    return deduped


def _metric_series(dossier: dict[str, Any], metric: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for row in _std_rows(dossier):
        year = row.get("year")
        value = row.get(metric)
        if isinstance(year, int) and _is_num(value):
            out.append((year, float(value)))
    return out


def _cagr_from_series(series: list[tuple[int, float]], years_window: int) -> tuple[float | str, tuple[int, int] | None]:
    if len(series) < 2:
        return UNKNOWN, None
    end_year, end_value = series[-1]
    if end_value <= 0:
        return UNKNOWN, None
    start_target = end_year - years_window
    start_candidates = [(year, value) for year, value in series if year <= start_target and value > 0]
    if start_candidates:
        start_year, start_value = start_candidates[-1]
    else:
        start_year, start_value = series[0]
    if start_value <= 0:
        return UNKNOWN, None
    span = max(1, end_year - start_year)
    return (end_value / start_value) ** (1.0 / span) - 1.0, (start_year, end_year)


def _years_for_recent(rows: list[dict[str, Any]], count: int) -> list[int]:
    return [int(row["year"]) for row in rows[-count:] if isinstance(row.get("year"), int)]


def _quality_cfo_vs_ni(dossier: dict[str, Any]) -> tuple[str, list[str]]:
    rows = _std_rows(dossier)
    ratios: list[float] = []
    years: list[int] = []
    for row in rows:
        cfo = row.get("cfo")
        ni = row.get("net_income")
        year = row.get("year")
        ratio = _safe_div(cfo, ni)
        if isinstance(year, int) and _is_num(ratio) and float(ni) > 0:
            ratios.append(float(ratio))
            years.append(int(year))
    refs = _metric_refs(dossier, "cfo", years) + _metric_refs(dossier, "net_income", years)
    if len(ratios) < 3:
        return "UNKNOWN", refs
    ratio_min = min(ratios)
    ratio_max = max(ratios)
    ratio_avg = sum(ratios) / len(ratios)
    if ratio_min >= 0.8 and ratio_max <= 1.8 and ratio_avg >= 0.9:
        return "STABLE", refs
    if ratio_min > 0 and ratio_avg >= 0.75:
        return "MIXED", refs
    return "WEAK", refs


def _signal_result(
    *,
    name: str,
    title: str,
    contribution: float,
    status: str,
    signal_confidence: str,
    explanation: str,
    derived_from: list[str],
) -> dict[str, Any]:
    max_score = float(WHALE_SIGNAL_WEIGHTS[name])
    value = max(0.0, min(max_score, float(contribution)))
    return {
        "signal": name,
        "title": title,
        "score_contribution": round(value, 4),
        "max_score": round(max_score, 4),
        "status": status,
        "signal_confidence": signal_confidence,
        "why_it_matters": SIGNAL_SPECS.get(name, {}).get("why_it_matters", ""),
        "thresholds": SIGNAL_SPECS.get(name, {}).get("thresholds", {}),
        "explanation": explanation,
        "derived_from": derived_from,
    }


def _confidence(*, status: str, complete_data: bool, stability: str) -> str:
    status_norm = str(status or "").upper()
    stability_norm = str(stability or "").upper()
    if not complete_data or status_norm == "GAP":
        return "LOW"
    if stability_norm == "HIGH":
        return "HIGH"
    if stability_norm == "MED":
        return "MED"
    if status_norm == "PASS":
        return "HIGH"
    return "MED"


def build_whale_signals(dossier: dict[str, Any]) -> dict[str, Any]:
    ticker = str(dossier.get("ticker") or "").upper()
    run_id = str(dossier.get("run_id") or "")
    as_of_date = str(dossier.get("as_of_date") or "")
    rows = _std_rows(dossier)

    whale_signals: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    total_penalty = 0.0
    critical_unknowns: list[str] = []

    # 1) Growth persistence.
    rev_5y, rev_5y_refs = _derived_signal_lookup(dossier, "revenue_cagr_5y")
    rev_10y, rev_10y_refs = _derived_signal_lookup(dossier, "revenue_cagr_10y")
    rev_3y, rev_3y_refs = _derived_signal_lookup(dossier, "revenue_cagr_3y")
    growth_refs = rev_5y_refs + rev_10y_refs + rev_3y_refs
    growth_weight = WHALE_SIGNAL_WEIGHTS["growth_persistence"]
    if not (_is_num(rev_5y) and _is_num(rev_10y) and _is_num(rev_3y)):
        growth_contribution = growth_weight * 0.40
        total_penalty += MISSING_METRIC_PENALTY
        missing = ["revenue_cagr_5y", "revenue_cagr_10y", "revenue_cagr_3y"]
        gaps.append(
            {
                "signal": "growth_persistence",
                "missing_metrics": missing,
                "penalty": MISSING_METRIC_PENALTY,
                "derived_from": growth_refs,
            }
        )
        critical_unknowns.extend(missing)
        whale_signals.append(
            _signal_result(
                name="growth_persistence",
                title="Growth Persistence",
                contribution=growth_contribution,
                status="GAP",
                signal_confidence=_confidence(status="GAP", complete_data=False, stability="LOW"),
                explanation="Revenue CAGR metrics are incomplete; applied conservative partial credit.",
                derived_from=growth_refs,
            )
        )
    else:
        if float(rev_5y) >= 0.10 and float(rev_10y) >= 0.08 and float(rev_3y) >= 0.04:
            growth_contribution = growth_weight
            growth_status = "PASS"
        elif float(rev_5y) >= 0.06 and float(rev_10y) >= 0.04 and float(rev_3y) >= 0.0:
            growth_contribution = growth_weight * 0.65
            growth_status = "PARTIAL"
        else:
            growth_contribution = growth_weight * 0.20
            growth_status = "FAIL"
        whale_signals.append(
            _signal_result(
                name="growth_persistence",
                title="Growth Persistence",
                contribution=growth_contribution,
                status=growth_status,
                signal_confidence=_confidence(
                    status=growth_status,
                    complete_data=True,
                    stability="HIGH" if growth_status == "PASS" else ("MED" if growth_status == "PARTIAL" else "LOW"),
                ),
                explanation=(
                    f"5Y CAGR {_fmt_pct(rev_5y)}, 10Y CAGR {_fmt_pct(rev_10y)}, "
                    f"last-3Y CAGR {_fmt_pct(rev_3y)}."
                ),
                derived_from=growth_refs,
            )
        )

    # 2) Margin expansion.
    gm_slope, gm_refs = _derived_signal_lookup(dossier, "gross_margin_trend_slope")
    om_slope, om_refs = _derived_signal_lookup(dossier, "operating_margin_trend_slope")
    margin_refs = gm_refs + om_refs
    margin_weight = WHALE_SIGNAL_WEIGHTS["margin_expansion"]
    if not (_is_num(gm_slope) and _is_num(om_slope)):
        margin_contribution = margin_weight * 0.40
        total_penalty += MISSING_METRIC_PENALTY
        missing = ["gross_margin_trend_slope", "operating_margin_trend_slope"]
        gaps.append(
            {
                "signal": "margin_expansion",
                "missing_metrics": missing,
                "penalty": MISSING_METRIC_PENALTY,
                "derived_from": margin_refs,
            }
        )
        whale_signals.append(
            _signal_result(
                name="margin_expansion",
                title="Margin Expansion",
                contribution=margin_contribution,
                status="GAP",
                signal_confidence=_confidence(status="GAP", complete_data=False, stability="LOW"),
                explanation="Gross/operating margin trend signals are incomplete; conservative partial credit applied.",
                derived_from=margin_refs,
            )
        )
    else:
        gm = float(gm_slope)
        om = float(om_slope)
        if gm > 0 and om > 0:
            margin_contribution = margin_weight
            margin_status = "PASS"
        elif gm > 0 or om > 0:
            margin_contribution = margin_weight * 0.60
            margin_status = "PARTIAL"
        else:
            margin_contribution = margin_weight * 0.15
            margin_status = "FAIL"
        whale_signals.append(
            _signal_result(
                name="margin_expansion",
                title="Margin Expansion",
                contribution=margin_contribution,
                status=margin_status,
                signal_confidence=_confidence(
                    status=margin_status,
                    complete_data=True,
                    stability="HIGH" if margin_status == "PASS" else ("MED" if margin_status == "PARTIAL" else "LOW"),
                ),
                explanation=f"Gross-margin slope {gm:.4f}; operating-margin slope {om:.4f}.",
                derived_from=margin_refs,
            )
        )

    # 3) FCF inflection.
    fcf_slope, fcf_refs = _derived_signal_lookup(dossier, "fcf_margin_trend_slope")
    quality_state, quality_refs = _quality_cfo_vs_ni(dossier)
    fcf_weight = WHALE_SIGNAL_WEIGHTS["fcf_inflection"]
    fcf_trace = fcf_refs + quality_refs
    if not _is_num(fcf_slope) or quality_state == "UNKNOWN":
        fcf_contribution = fcf_weight * 0.35
        total_penalty += MISSING_METRIC_PENALTY
        missing = ["fcf_margin_trend_slope", "cfo_vs_net_income_quality"]
        gaps.append(
            {
                "signal": "fcf_inflection",
                "missing_metrics": missing,
                "penalty": MISSING_METRIC_PENALTY,
                "derived_from": fcf_trace,
            }
        )
        critical_unknowns.extend(missing)
        whale_signals.append(
            _signal_result(
                name="fcf_inflection",
                title="FCF Inflection",
                contribution=fcf_contribution,
                status="GAP",
                signal_confidence=_confidence(status="GAP", complete_data=False, stability="LOW"),
                explanation="FCF trend or CFO/NI quality history is incomplete; conservative partial credit applied.",
                derived_from=fcf_trace,
            )
        )
    else:
        slope = float(fcf_slope)
        if slope > 0 and quality_state == "STABLE":
            fcf_contribution = fcf_weight
            fcf_status = "PASS"
        elif slope >= 0 and quality_state in {"STABLE", "MIXED"}:
            fcf_contribution = fcf_weight * 0.65
            fcf_status = "PARTIAL"
        else:
            fcf_contribution = fcf_weight * 0.20
            fcf_status = "FAIL"
        whale_signals.append(
            _signal_result(
                name="fcf_inflection",
                title="FCF Inflection",
                contribution=fcf_contribution,
                status=fcf_status,
                signal_confidence=_confidence(
                    status=fcf_status,
                    complete_data=True,
                    stability="HIGH" if (fcf_status == "PASS" and quality_state == "STABLE") else ("MED" if fcf_status == "PARTIAL" else "LOW"),
                ),
                explanation=f"FCF-margin slope {slope:.4f}; CFO/NI quality={quality_state}.",
                derived_from=fcf_trace,
            )
        )

    # 4) Dilution discipline.
    dilution_cagr, dilution_refs = _derived_signal_lookup(dossier, "dilution_rate_shares_cagr")
    shares_series = _metric_series(dossier, "shares_outstanding")
    shares_5y_cagr, shares_span = _cagr_from_series(shares_series, 5)
    share_years = list(range(shares_span[0], shares_span[1] + 1)) if shares_span else _years_for_recent(rows, 6)
    shares_refs = _metric_refs(dossier, "shares_outstanding", share_years)
    dilution_weight = WHALE_SIGNAL_WEIGHTS["dilution_discipline"]
    dilution_trace = dilution_refs + shares_refs
    if not _is_num(dilution_cagr):
        dilution_contribution = dilution_weight * 0.40
        total_penalty += 1.5
        missing = ["dilution_rate_shares_cagr"]
        gaps.append(
            {
                "signal": "dilution_discipline",
                "missing_metrics": missing,
                "penalty": 1.5,
                "derived_from": dilution_trace,
            }
        )
        whale_signals.append(
            _signal_result(
                name="dilution_discipline",
                title="Dilution Discipline",
                contribution=dilution_contribution,
                status="GAP",
                signal_confidence=_confidence(status="GAP", complete_data=False, stability="LOW"),
                explanation="Share-count CAGR signal missing; conservative partial credit applied.",
                derived_from=dilution_trace,
            )
        )
    else:
        dil = float(dilution_cagr)
        shares_5y = float(shares_5y_cagr) if _is_num(shares_5y_cagr) else None
        if dil <= 0.02 or (shares_5y is not None and shares_5y <= 0):
            dilution_contribution = dilution_weight
            dilution_status = "PASS"
        elif dil <= 0.04:
            dilution_contribution = dilution_weight * 0.60
            dilution_status = "PARTIAL"
        else:
            dilution_contribution = dilution_weight * 0.15
            dilution_status = "FAIL"
        shares_5y_text = _fmt_pct(shares_5y) if shares_5y is not None else "UNKNOWN"
        whale_signals.append(
            _signal_result(
                name="dilution_discipline",
                title="Dilution Discipline",
                contribution=dilution_contribution,
                status=dilution_status,
                signal_confidence=_confidence(
                    status=dilution_status,
                    complete_data=True,
                    stability="HIGH" if dilution_status == "PASS" else ("MED" if dilution_status == "PARTIAL" else "LOW"),
                ),
                explanation=f"Shares CAGR (10Y proxy) {_fmt_pct(dil)}; shares CAGR (5Y) {shares_5y_text}.",
                derived_from=dilution_trace,
            )
        )

    # 5) Reinvestment capacity.
    cfo_series = _metric_series(dossier, "cfo")
    capex_series = _metric_series(dossier, "capex")
    revenue_series = _metric_series(dossier, "revenue")
    cfo_5y_cagr, _ = _cagr_from_series(cfo_series, 5)
    reinvest_weight = WHALE_SIGNAL_WEIGHTS["reinvestment_capacity"]
    recent_years = _years_for_recent(rows, 6)
    reinvest_refs = (
        _metric_refs(dossier, "cfo", recent_years)
        + _metric_refs(dossier, "capex", recent_years)
        + _metric_refs(dossier, "revenue", recent_years)
    )
    if not cfo_series or not capex_series or not revenue_series:
        reinvest_contribution = reinvest_weight * 0.35
        total_penalty += MISSING_METRIC_PENALTY
        missing = ["cfo", "capex", "revenue"]
        gaps.append(
            {
                "signal": "reinvestment_capacity",
                "missing_metrics": missing,
                "penalty": MISSING_METRIC_PENALTY,
                "derived_from": reinvest_refs,
            }
        )
        whale_signals.append(
            _signal_result(
                name="reinvestment_capacity",
                title="Reinvestment Capacity",
                contribution=reinvest_contribution,
                status="GAP",
                signal_confidence=_confidence(status="GAP", complete_data=False, stability="LOW"),
                explanation="CFO/capex/revenue history is incomplete; conservative partial credit applied.",
                derived_from=reinvest_refs,
            )
        )
    else:
        latest_cfo = cfo_series[-1][1]
        first_capex = capex_series[0][1]
        first_revenue = revenue_series[0][1]
        latest_capex = capex_series[-1][1]
        latest_revenue = revenue_series[-1][1]
        capex_ratio_first = _safe_div(first_capex, first_revenue)
        capex_ratio_latest = _safe_div(latest_capex, latest_revenue)
        capex_ratio_growth = (
            (float(capex_ratio_latest) / float(capex_ratio_first) - 1.0)
            if _is_num(capex_ratio_first) and _is_num(capex_ratio_latest) and float(capex_ratio_first) > 0
            else UNKNOWN
        )
        cfo_growth = float(cfo_5y_cagr) if _is_num(cfo_5y_cagr) else UNKNOWN
        if _is_num(cfo_growth) and _is_num(capex_ratio_growth) and latest_cfo > 0 and cfo_growth >= 0.05 and capex_ratio_growth <= 0.35:
            reinvest_contribution = reinvest_weight
            reinvest_status = "PASS"
        elif _is_num(cfo_growth) and latest_cfo > 0 and cfo_growth >= 0 and (not _is_num(capex_ratio_growth) or capex_ratio_growth <= 0.8):
            reinvest_contribution = reinvest_weight * 0.62
            reinvest_status = "PARTIAL"
        else:
            reinvest_contribution = reinvest_weight * 0.20
            reinvest_status = "FAIL"
        capex_growth_text = _fmt_pct(capex_ratio_growth) if _is_num(capex_ratio_growth) else "UNKNOWN"
        whale_signals.append(
            _signal_result(
                name="reinvestment_capacity",
                title="Reinvestment Capacity",
                contribution=reinvest_contribution,
                status=reinvest_status,
                signal_confidence=_confidence(
                    status=reinvest_status,
                    complete_data=True,
                    stability="HIGH" if reinvest_status == "PASS" else ("MED" if reinvest_status == "PARTIAL" else "LOW"),
                ),
                explanation=f"CFO 5Y CAGR {_fmt_pct(cfo_growth)}; capex/revenue ratio change {capex_growth_text}.",
                derived_from=reinvest_refs,
            )
        )

    # 6) Balance sheet resilience.
    net_debt_series = _metric_series(dossier, "net_debt")
    balance_weight = WHALE_SIGNAL_WEIGHTS["balance_sheet_resilience"]
    if len(net_debt_series) < 2:
        balance_contribution = balance_weight * 0.35
        total_penalty += MISSING_METRIC_PENALTY
        balance_refs = _metric_refs(dossier, "net_debt")
        missing = ["net_debt"]
        gaps.append(
            {
                "signal": "balance_sheet_resilience",
                "missing_metrics": missing,
                "penalty": MISSING_METRIC_PENALTY,
                "derived_from": balance_refs,
            }
        )
        critical_unknowns.extend(missing)
        whale_signals.append(
            _signal_result(
                name="balance_sheet_resilience",
                title="Balance Sheet Resilience",
                contribution=balance_contribution,
                status="GAP",
                signal_confidence=_confidence(status="GAP", complete_data=False, stability="LOW"),
                explanation="Net-debt trend is incomplete; conservative partial credit applied.",
                derived_from=balance_refs,
            )
        )
    else:
        start_year, start_debt = net_debt_series[0]
        end_year, end_debt = net_debt_series[-1]
        debt_delta = end_debt - start_debt
        tolerance = abs(start_debt) * 0.25
        balance_refs = _metric_refs(dossier, "net_debt", [start_year, end_year])
        if debt_delta <= 0:
            balance_contribution = balance_weight
            balance_status = "PASS"
        elif debt_delta <= tolerance:
            balance_contribution = balance_weight * 0.60
            balance_status = "PARTIAL"
        else:
            balance_contribution = balance_weight * 0.15
            balance_status = "FAIL"
        whale_signals.append(
            _signal_result(
                name="balance_sheet_resilience",
                title="Balance Sheet Resilience",
                contribution=balance_contribution,
                status=balance_status,
                signal_confidence=_confidence(
                    status=balance_status,
                    complete_data=True,
                    stability="HIGH" if balance_status == "PASS" else ("MED" if balance_status == "PARTIAL" else "LOW"),
                ),
                explanation=f"Net debt moved from {start_debt:.2f} to {end_debt:.2f} across {start_year}-{end_year}.",
                derived_from=balance_refs,
            )
        )

    raw_total = sum(float(row.get("score_contribution") or 0.0) for row in whale_signals)
    critical_unknowns = sorted(set([metric for metric in critical_unknowns if metric]))
    critical_penalty = float(len(critical_unknowns)) * 1.0
    raw_penalty = float(total_penalty) + critical_penalty
    penalty_cap = float(raw_total) * 0.8 if critical_unknowns else float(raw_total)
    applied_penalty = min(raw_penalty, penalty_cap)
    whale_signature_score = max(0.0, min(100.0, raw_total - applied_penalty))
    payload = {
        "whale_signals_version": WHALE_SIGNALS_VERSION,
        "ticker": ticker,
        "run_id": run_id,
        "as_of_date": as_of_date,
        "whale_signature_score": round(float(whale_signature_score), 4),
        "whale_signals_json": whale_signals,
        "signals": whale_signals,
        "gaps": gaps,
        "critical_unknowns": critical_unknowns,
        "weights": WHALE_SIGNAL_WEIGHTS,
        "raw_penalty": round(float(raw_penalty), 4),
        "total_penalty": round(float(applied_penalty), 4),
        "penalty_rule": {
            "description": "Apply missing-metric penalties plus critical-unknown penalty, capped at 80% of raw signal total when critical unknowns exist.",
            "critical_unknowns": critical_unknowns,
            "critical_unknown_penalty_per_metric": 1.0,
            "raw_penalty": round(float(raw_penalty), 4),
            "penalty_cap": round(float(penalty_cap), 4),
            "applied_penalty": round(float(applied_penalty), 4),
        },
    }
    return payload


def _top_signal_names(whale_payload: dict[str, Any], limit: int = 3) -> list[str]:
    rows = whale_payload.get("signals") or whale_payload.get("whale_signals_json") or []
    if not isinstance(rows, list):
        return []
    ranked = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: (-float(row.get("score_contribution") or 0.0), str(row.get("signal") or "")),
    )
    return [str(row.get("signal") or "") for row in ranked[: max(1, int(limit))] if str(row.get("signal") or "")]


def run_whale_signals_for_run(*, run_id: str) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.dossiers_dir / run_id
    if not run_dir.exists():
        raise ValueError(f"dossier run not found: {run_id}")

    rows: list[dict[str, Any]] = []
    artifacts: list[str] = []
    for ticker_dir in sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.isupper()], key=lambda p: p.name):
        dossier_path = ticker_dir / "dossier.json"
        if not dossier_path.exists():
            continue
        dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
        if not isinstance(dossier, dict):
            continue
        whale_payload = build_whale_signals(dossier)
        ticker = str(whale_payload.get("ticker") or ticker_dir.name).upper()
        out_path = run_dir / f"whale_signals_{ticker}.json"
        out_path.write_text(json.dumps(whale_payload, indent=2), encoding="utf-8")
        artifacts.append(str(out_path))
        rows.append(
            {
                "ticker": ticker,
                "whale_signature_score": whale_payload.get("whale_signature_score", 0.0),
                "gap_count": len(whale_payload.get("gaps") or []),
                "top_signals": _top_signal_names(whale_payload, limit=3),
                "path": str(out_path),
            }
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (-float(row.get("whale_signature_score") or 0.0), str(row.get("ticker") or "")),
    )
    summary = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "tickers": [row["ticker"] for row in ranked_rows],
        "whale_signature_rank": [row["ticker"] for row in ranked_rows],
        "rows": ranked_rows,
        "artifacts": artifacts,
    }
    summary_path = run_dir / "whale_signals_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
