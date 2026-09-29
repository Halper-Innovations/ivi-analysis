from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable

from app.valuation.intangible_economics import _dedupe_refs, _is_num
from app.valuation.tech_category import (
    CONSUMER_HARDWARE,
    ENTERPRISE_SOFTWARE,
    INDUSTRIAL_TECH,
    NETWORK_INFRA,
    PLATFORM_HYBRID,
    SEMICONDUCTOR,
    TRADITIONAL_OPERATING,
)

TimeSeriesData = dict[str, Any]
PatternCallable = Callable[[TimeSeriesData], dict[str, Any]]

ALL_CATEGORIES = (
    ENTERPRISE_SOFTWARE,
    SEMICONDUCTOR,
    CONSUMER_HARDWARE,
    INDUSTRIAL_TECH,
    NETWORK_INFRA,
    PLATFORM_HYBRID,
    TRADITIONAL_OPERATING,
)
TECH_CATEGORIES = (
    ENTERPRISE_SOFTWARE,
    SEMICONDUCTOR,
    CONSUMER_HARDWARE,
    INDUSTRIAL_TECH,
    NETWORK_INFRA,
    PLATFORM_HYBRID,
)


@dataclass(frozen=True)
class PatternDefinition:
    pattern_id: str
    name: str
    hypothesis: str
    category: tuple[str, ...]
    required_metrics: tuple[str, ...]
    lookback_years: int
    detection_fn: PatternCallable
    outcome_fn: Callable[[TimeSeriesData, dict[str, Any]], dict[str, Any]]


def _series_rows(data: TimeSeriesData, metric: str) -> list[dict[str, Any]]:
    rows = data.get(metric) if isinstance(data, dict) else []
    if not isinstance(rows, list):
        return []
    return sorted(
        [
            row
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("year"), int)
            and _is_num(row.get("value"))
        ],
        key=lambda row: int(row.get("year") or 0),
    )


def _series_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["year"]): row for row in rows if isinstance(row.get("year"), int)}


def _ratio_series(data: TimeSeriesData, numerator_metric: str, denominator_metric: str) -> list[dict[str, Any]]:
    numerator = _series_rows(data, numerator_metric)
    denominator_by_year = _series_map(_series_rows(data, denominator_metric))
    out: list[dict[str, Any]] = []
    for row in numerator:
        denom = denominator_by_year.get(int(row["year"]))
        if not isinstance(denom, dict) or not _is_num(denom.get("value")):
            continue
        denom_value = float(denom["value"])
        if denom_value == 0.0:
            continue
        out.append(
            {
                "year": int(row["year"]),
                "value": float(row["value"]) / denom_value,
                "derived_from": _dedupe_refs(list(row.get("derived_from") or []) + list(denom.get("derived_from") or [])),
            }
        )
    return out


def _growth_series(data: TimeSeriesData, metric: str) -> list[dict[str, Any]]:
    rows = _series_rows(data, metric)
    out: list[dict[str, Any]] = []
    for prev, curr in zip(rows, rows[1:]):
        prev_value = float(prev["value"])
        curr_value = float(curr["value"])
        if prev_value == 0.0:
            continue
        out.append(
            {
                "year": int(curr["year"]),
                "value": (curr_value - prev_value) / abs(prev_value),
                "derived_from": _dedupe_refs(list(prev.get("derived_from") or []) + list(curr.get("derived_from") or [])),
            }
        )
    return out


def _consecutive_runs(rows: list[dict[str, Any]], *, min_len: int = 1) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: int(row.get("year") or 0))
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [ordered[0]]
    for row in ordered[1:]:
        prev_year = int(current[-1].get("year") or 0)
        year = int(row.get("year") or 0)
        if year == prev_year + 1:
            current.append(row)
        else:
            if len(current) >= min_len:
                runs.append(current)
            current = [row]
    if len(current) >= min_len:
        runs.append(current)
    return runs


def _select_best_run(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not runs:
        return []
    return sorted(
        runs,
        key=lambda run: (
            mean(float(row.get("score", 0.0)) for row in run),
            len(run),
            int(run[-1].get("year") or 0),
        ),
    )[-1]


def _missing_required(data: TimeSeriesData, required_metrics: tuple[str, ...], lookback_years: int) -> bool:
    for metric in required_metrics:
        rows = _series_rows(data, metric)
        if len(rows) < int(lookback_years):
            return True
    return False


def _empty_detection(required_metrics: tuple[str, ...], lookback_years: int, data: TimeSeriesData) -> dict[str, Any]:
    return {
        "present": False,
        "years_detected": [],
        "detection_strength": 0.0,
        "derived_from": [],
        "reason": "INSUFFICIENT_HISTORY" if _missing_required(data, required_metrics, lookback_years) else "NO_PATTERN",
    }


def _future_rows(rows: list[dict[str, Any]], after_year: int, years_ahead: tuple[int, ...]) -> list[dict[str, Any]]:
    row_map = _series_map(rows)
    return [row_map[target_year] for target_year in [after_year + offset for offset in years_ahead] if target_year in row_map]


def _deferred_revenue_leading_indicator_detect(data: TimeSeriesData) -> dict[str, Any]:
    required = ("revenue", "deferred_revenue")
    if _missing_required(data, required, 4):
        return _empty_detection(required, 4, data)
    revenue_growth = _series_map(_growth_series(data, "revenue"))
    deferred_growth = _growth_series(data, "deferred_revenue")
    qualifying: list[dict[str, Any]] = []
    for row in deferred_growth:
        year = int(row["year"])
        rev = revenue_growth.get(year)
        if not isinstance(rev, dict):
            continue
        gap = float(row["value"]) - float(rev["value"])
        if gap > 0.15:
            qualifying.append(
                {
                    "year": year,
                    "score": gap,
                    "derived_from": _dedupe_refs(list(row.get("derived_from") or []) + list(rev.get("derived_from") or [])),
                    "baseline_revenue_growth": float(rev["value"]),
                }
            )
    best_run = _select_best_run(_consecutive_runs(qualifying, min_len=2))
    if not best_run:
        return _empty_detection(required, 4, data)
    return {
        "present": True,
        "years_detected": [int(row["year"]) for row in best_run],
        "detection_strength": min(1.0, max(0.0, mean(float(row["score"]) for row in best_run) / 0.30)),
        "baseline_revenue_growth": float(best_run[-1]["baseline_revenue_growth"]),
        "derived_from": _dedupe_refs([ref for row in best_run for ref in list(row.get("derived_from") or [])]),
    }


def _deferred_revenue_leading_indicator_outcome(data: TimeSeriesData, detection: dict[str, Any]) -> dict[str, Any]:
    if not detection.get("present"):
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": None, "derived_from": []}
    future_growth = _future_rows(_growth_series(data, "revenue"), max(detection["years_detected"]), (1, 2))
    if not future_growth:
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "FOLLOW_UP_TOO_RECENT", "derived_from": []}
    baseline = float(detection.get("baseline_revenue_growth") or 0.0)
    best_future = max(float(row["value"]) for row in future_growth)
    return {
        "outcome_confirmed": best_future > baseline,
        "outcome_value": best_future - baseline,
        "outcome_details": "Revenue growth accelerated after deferred revenue outpaced recognized revenue.",
        "derived_from": _dedupe_refs([ref for row in future_growth for ref in list(row.get("derived_from") or [])]),
    }


def _capex_to_depreciation_divergence_detect(data: TimeSeriesData) -> dict[str, Any]:
    required = ("capex", "depreciation", "operating_income", "revenue")
    if _missing_required(data, required, 4):
        return _empty_detection(required, 4, data)
    ratio_rows = _ratio_series(data, "capex", "depreciation")
    qualifying = [
        {
            "year": int(row["year"]),
            "score": float(row["value"]) - 1.5,
            "derived_from": list(row.get("derived_from") or []),
        }
        for row in ratio_rows
        if float(row["value"]) > 1.5
    ]
    best_run = _select_best_run(_consecutive_runs(qualifying, min_len=2))
    if not best_run:
        return _empty_detection(required, 4, data)
    return {
        "present": True,
        "years_detected": [int(row["year"]) for row in best_run],
        "detection_strength": min(1.0, max(0.0, mean(float(row["score"]) for row in best_run) / 1.0)),
        "derived_from": _dedupe_refs([ref for row in best_run for ref in list(row.get("derived_from") or [])]),
    }


def _capex_to_depreciation_divergence_outcome(data: TimeSeriesData, detection: dict[str, Any]) -> dict[str, Any]:
    if not detection.get("present"):
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": None, "derived_from": []}
    op_margin_rows = _ratio_series(data, "operating_income", "revenue")
    op_margin_map = _series_map(op_margin_rows)
    last_year = max(detection["years_detected"])
    baseline = op_margin_map.get(last_year)
    if not isinstance(baseline, dict):
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "BASELINE_MARGIN_MISSING", "derived_from": []}
    future_rows = _future_rows(op_margin_rows, last_year, (2, 3))
    if not future_rows:
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "FOLLOW_UP_TOO_RECENT", "derived_from": []}
    best_delta = max(float(row["value"]) - float(baseline["value"]) for row in future_rows)
    return {
        "outcome_confirmed": best_delta > 0.01,
        "outcome_value": best_delta,
        "outcome_details": "Operating margin expanded after capex materially exceeded depreciation.",
        "derived_from": _dedupe_refs(list(baseline.get("derived_from") or []) + [ref for row in future_rows for ref in list(row.get("derived_from") or [])]),
    }


def _rnd_intensity_inflection_detect(data: TimeSeriesData) -> dict[str, Any]:
    required = ("r_and_d_total", "revenue", "operating_income")
    if _missing_required(data, required, 5):
        return _empty_detection(required, 5, data)
    ratio_rows = _ratio_series(data, "r_and_d_total", "revenue")
    if len(ratio_rows) < 4:
        return _empty_detection(required, 5, data)
    down_run: list[dict[str, Any]] = []
    up_streak = 0
    best_run: list[dict[str, Any]] = []
    for prev, curr in zip(ratio_rows, ratio_rows[1:]):
        delta = float(curr["value"]) - float(prev["value"])
        if delta > 0:
            up_streak += 1
            if down_run:
                if len(down_run) >= 1:
                    best_run = down_run
                down_run = []
        elif delta < 0 and up_streak >= 2:
            down_run.append(
                {
                    "year": int(curr["year"]),
                    "score": abs(delta),
                    "derived_from": _dedupe_refs(list(prev.get("derived_from") or []) + list(curr.get("derived_from") or [])),
                }
            )
        else:
            if down_run:
                best_run = down_run
            down_run = []
            up_streak = 0
    if down_run:
        best_run = down_run
    if not best_run:
        return _empty_detection(required, 5, data)
    return {
        "present": True,
        "years_detected": [int(row["year"]) for row in best_run],
        "detection_strength": min(1.0, max(0.0, mean(float(row["score"]) for row in best_run) / 0.05)),
        "derived_from": _dedupe_refs([ref for row in best_run for ref in list(row.get("derived_from") or [])]),
    }


def _rnd_intensity_inflection_outcome(data: TimeSeriesData, detection: dict[str, Any]) -> dict[str, Any]:
    if not detection.get("present"):
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": None, "derived_from": []}
    op_margin_rows = _ratio_series(data, "operating_income", "revenue")
    revenue_growth_rows = _growth_series(data, "revenue")
    last_year = max(detection["years_detected"])
    op_margin_map = _series_map(op_margin_rows)
    baseline_margin = op_margin_map.get(last_year)
    if not isinstance(baseline_margin, dict):
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "BASELINE_MARGIN_MISSING", "derived_from": []}
    future_margin_rows = _future_rows(op_margin_rows, last_year, (1, 2))
    future_rev_growth = _future_rows(revenue_growth_rows, last_year, (1, 2))
    if not future_margin_rows or not future_rev_growth:
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "FOLLOW_UP_TOO_RECENT", "derived_from": []}
    margin_delta = max(float(row["value"]) - float(baseline_margin["value"]) for row in future_margin_rows)
    revenue_continues = any(float(row["value"]) > 0.0 for row in future_rev_growth)
    return {
        "outcome_confirmed": revenue_continues and margin_delta > 0.01,
        "outcome_value": margin_delta,
        "outcome_details": "Operating leverage followed the R&D intensity inflection while revenue continued to grow.",
        "derived_from": _dedupe_refs(
            list(baseline_margin.get("derived_from") or [])
            + [ref for row in future_margin_rows for ref in list(row.get("derived_from") or [])]
            + [ref for row in future_rev_growth for ref in list(row.get("derived_from") or [])]
        ),
    }


def _cash_conversion_quality_divergence_detect(data: TimeSeriesData) -> dict[str, Any]:
    required = ("cfo", "net_income")
    if _missing_required(data, required, 4):
        return _empty_detection(required, 4, data)
    ratio_rows = _ratio_series(data, "cfo", "net_income")
    high_rows = [
        {
            "year": int(row["year"]),
            "score": float(row["value"]) - 1.3,
            "direction": "HIGH",
            "derived_from": list(row.get("derived_from") or []),
        }
        for row in ratio_rows
        if float(row["value"]) > 1.3
    ]
    low_rows = [
        {
            "year": int(row["year"]),
            "score": 0.7 - float(row["value"]),
            "direction": "LOW",
            "derived_from": list(row.get("derived_from") or []),
        }
        for row in ratio_rows
        if float(row["value"]) < 0.7
    ]
    best_high = _select_best_run(_consecutive_runs(high_rows, min_len=2))
    best_low = _select_best_run(_consecutive_runs(low_rows, min_len=2))
    chosen = sorted(
        [run for run in (best_high, best_low) if run],
        key=lambda run: (mean(float(row["score"]) for row in run), int(run[-1]["year"])),
    )[-1:] or []
    if not chosen:
        return _empty_detection(required, 4, data)
    run = chosen[0]
    return {
        "present": True,
        "years_detected": [int(row["year"]) for row in run],
        "direction": str(run[-1]["direction"]),
        "detection_strength": min(1.0, max(0.0, mean(float(row["score"]) for row in run) / 0.6)),
        "derived_from": _dedupe_refs([ref for row in run for ref in list(row.get("derived_from") or [])]),
    }


def _cash_conversion_quality_divergence_outcome(data: TimeSeriesData, detection: dict[str, Any]) -> dict[str, Any]:
    if not detection.get("present"):
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": None, "derived_from": []}
    net_income_rows = _series_rows(data, "net_income")
    net_income_map = _series_map(net_income_rows)
    last_year = max(detection["years_detected"])
    future_rows = _future_rows(net_income_rows, last_year, (1, 2))
    baseline_rows = [net_income_map[year] for year in detection["years_detected"] if year in net_income_map]
    if not future_rows or not baseline_rows:
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "FOLLOW_UP_TOO_RECENT", "derived_from": []}
    baseline_avg = mean(float(row["value"]) for row in baseline_rows)
    future_avg = mean(float(row["value"]) for row in future_rows)
    change_ratio = None
    if baseline_avg != 0.0:
        change_ratio = (future_avg - baseline_avg) / abs(baseline_avg)
    direction = str(detection.get("direction") or "")
    if direction == "HIGH":
        confirmed = min(float(row["value"]) for row in future_rows) >= baseline_avg * 0.90 and future_avg >= baseline_avg
        detail = "High CFO-to-net-income conversion preceded more durable earnings."
    else:
        confirmed = any(float(row["value"]) < 0.0 for row in future_rows) or future_avg <= baseline_avg * 0.85
        detail = "Low CFO-to-net-income conversion preceded weaker subsequent earnings."
    return {
        "outcome_confirmed": confirmed,
        "outcome_value": change_ratio,
        "outcome_details": detail,
        "derived_from": _dedupe_refs([ref for row in baseline_rows + future_rows for ref in list(row.get("derived_from") or [])]),
    }


def _capital_return_inflection_detect(data: TimeSeriesData) -> dict[str, Any]:
    required = ("shares_outstanding", "fcf")
    if _missing_required(data, required, 4):
        return _empty_detection(required, 4, data)
    shares_rows = _series_rows(data, "shares_outstanding")
    down_run: list[dict[str, Any]] = []
    up_streak = 0
    best_run: list[dict[str, Any]] = []
    for prev, curr in zip(shares_rows, shares_rows[1:]):
        delta = float(curr["value"]) - float(prev["value"])
        if delta > 0:
            up_streak += 1
            if down_run:
                best_run = down_run
                down_run = []
        elif delta < 0 and up_streak >= 2:
            down_run.append(
                {
                    "year": int(curr["year"]),
                    "score": abs(delta) / max(abs(float(prev["value"])), 1.0),
                    "derived_from": _dedupe_refs(list(prev.get("derived_from") or []) + list(curr.get("derived_from") or [])),
                }
            )
        else:
            if down_run:
                best_run = down_run
            down_run = []
            up_streak = 0
    if down_run:
        best_run = down_run
    if not best_run:
        return _empty_detection(required, 4, data)
    return {
        "present": True,
        "years_detected": [int(row["year"]) for row in best_run],
        "detection_strength": min(1.0, max(0.0, mean(float(row["score"]) for row in best_run) / 0.05)),
        "derived_from": _dedupe_refs([ref for row in best_run for ref in list(row.get("derived_from") or [])]),
    }


def _capital_return_inflection_outcome(data: TimeSeriesData, detection: dict[str, Any]) -> dict[str, Any]:
    if not detection.get("present"):
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": None, "derived_from": []}
    fcf_rows = _series_rows(data, "fcf")
    shares_rows = _series_rows(data, "shares_outstanding")
    fcf_map = _series_map(fcf_rows)
    shares_map = _series_map(shares_rows)
    last_year = max(detection["years_detected"])
    if last_year not in fcf_map or last_year not in shares_map:
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "BASELINE_PER_SHARE_MISSING", "derived_from": []}
    baseline_share = float(shares_map[last_year]["value"])
    baseline_fcf = float(fcf_map[last_year]["value"])
    if baseline_share == 0.0:
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "BASELINE_PER_SHARE_MISSING", "derived_from": []}
    baseline_per_share = baseline_fcf / baseline_share
    future_years = [year for year in (last_year + 1, last_year + 2) if year in fcf_map and year in shares_map and float(shares_map[year]["value"]) != 0.0]
    if not future_years:
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "FOLLOW_UP_TOO_RECENT", "derived_from": []}
    future_per_share = [
        {
            "value": float(fcf_map[year]["value"]) / float(shares_map[year]["value"]),
            "derived_from": _dedupe_refs(list(fcf_map[year].get("derived_from") or []) + list(shares_map[year].get("derived_from") or [])),
        }
        for year in future_years
    ]
    best_improvement = max(row["value"] - baseline_per_share for row in future_per_share)
    return {
        "outcome_confirmed": best_improvement > 0.0,
        "outcome_value": best_improvement,
        "outcome_details": "Per-share cash generation improved after share count inflected from dilution to buybacks.",
        "derived_from": _dedupe_refs(
            list(fcf_map[last_year].get("derived_from") or [])
            + list(shares_map[last_year].get("derived_from") or [])
            + [ref for row in future_per_share for ref in list(row.get("derived_from") or [])]
        ),
    }


def _gross_margin_regime_change_detect(data: TimeSeriesData) -> dict[str, Any]:
    required = ("gross_profit", "revenue", "operating_income")
    if _missing_required(data, required, 5):
        return _empty_detection(required, 5, data)
    gm_rows = _ratio_series(data, "gross_profit", "revenue")
    qualifying: list[dict[str, Any]] = []
    for idx in range(3, len(gm_rows)):
        prior = gm_rows[idx - 3:idx]
        current = gm_rows[idx]
        prior_avg = mean(float(row["value"]) for row in prior)
        delta = float(current["value"]) - prior_avg
        if abs(delta) > 0.03:
            qualifying.append(
                {
                    "year": int(current["year"]),
                    "score": abs(delta),
                    "direction": "UP" if delta > 0 else "DOWN",
                    "derived_from": _dedupe_refs([ref for row in prior + [current] for ref in list(row.get("derived_from") or [])]),
                }
            )
    same_direction_runs = []
    for run in _consecutive_runs(qualifying, min_len=2):
        directions = {str(row.get("direction") or "") for row in run}
        if len(directions) == 1:
            same_direction_runs.append(run)
    best_run = _select_best_run(same_direction_runs)
    if not best_run:
        return _empty_detection(required, 5, data)
    return {
        "present": True,
        "years_detected": [int(row["year"]) for row in best_run],
        "direction": str(best_run[-1]["direction"]),
        "detection_strength": min(1.0, max(0.0, mean(float(row["score"]) for row in best_run) / 0.10)),
        "derived_from": _dedupe_refs([ref for row in best_run for ref in list(row.get("derived_from") or [])]),
    }


def _gross_margin_regime_change_outcome(data: TimeSeriesData, detection: dict[str, Any]) -> dict[str, Any]:
    if not detection.get("present"):
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": None, "derived_from": []}
    op_margin_rows = _ratio_series(data, "operating_income", "revenue")
    op_margin_map = _series_map(op_margin_rows)
    last_year = max(detection["years_detected"])
    baseline = op_margin_map.get(last_year)
    future_rows = _future_rows(op_margin_rows, last_year, (1, 2))
    if not isinstance(baseline, dict) or not future_rows:
        return {"outcome_confirmed": None, "outcome_value": None, "outcome_details": "FOLLOW_UP_TOO_RECENT", "derived_from": []}
    best_delta = max(float(row["value"]) - float(baseline["value"]) for row in future_rows)
    worst_delta = min(float(row["value"]) - float(baseline["value"]) for row in future_rows)
    direction = str(detection.get("direction") or "")
    if direction == "UP":
        confirmed = best_delta > 0.01
        outcome_value = best_delta
        detail = "Operating margin improved after the gross margin regime shifted upward."
    else:
        confirmed = worst_delta < -0.01
        outcome_value = worst_delta
        detail = "Operating margin deteriorated after the gross margin regime shifted downward."
    return {
        "outcome_confirmed": confirmed,
        "outcome_value": outcome_value,
        "outcome_details": detail,
        "derived_from": _dedupe_refs(list(baseline.get("derived_from") or []) + [ref for row in future_rows for ref in list(row.get("derived_from") or [])]),
    }


PATTERN_CATALOG: tuple[PatternDefinition, ...] = (
    PatternDefinition(
        pattern_id="deferred_revenue_leading_indicator",
        name="Deferred Revenue Leading Indicator",
        hypothesis="Deferred revenue growth exceeding recognized revenue growth by more than 15 percentage points for 2+ consecutive years predicts revenue acceleration in the following 1-2 years.",
        category=(ENTERPRISE_SOFTWARE, PLATFORM_HYBRID),
        required_metrics=("revenue", "deferred_revenue"),
        lookback_years=4,
        detection_fn=_deferred_revenue_leading_indicator_detect,
        outcome_fn=_deferred_revenue_leading_indicator_outcome,
    ),
    PatternDefinition(
        pattern_id="capex_to_depreciation_divergence",
        name="Capex-to-Depreciation Divergence",
        hypothesis="Capex-to-depreciation above 1.5x for 2+ consecutive years predicts operating-margin expansion within 2-3 years.",
        category=(INDUSTRIAL_TECH, SEMICONDUCTOR, TRADITIONAL_OPERATING),
        required_metrics=("capex", "depreciation", "operating_income", "revenue"),
        lookback_years=4,
        detection_fn=_capex_to_depreciation_divergence_detect,
        outcome_fn=_capex_to_depreciation_divergence_outcome,
    ),
    PatternDefinition(
        pattern_id="rnd_intensity_inflection",
        name="R&D Intensity Inflection",
        hypothesis="R&D-to-revenue declining after a multi-year rise signals operating leverage if revenue continues to grow.",
        category=TECH_CATEGORIES,
        required_metrics=("r_and_d_total", "revenue", "operating_income"),
        lookback_years=5,
        detection_fn=_rnd_intensity_inflection_detect,
        outcome_fn=_rnd_intensity_inflection_outcome,
    ),
    PatternDefinition(
        pattern_id="cash_conversion_quality_divergence",
        name="Cash Conversion Quality Divergence",
        hypothesis="CFO-to-net-income diverging above 1.3x or below 0.7x for 2+ years foreshadows stronger or weaker subsequent earnings durability.",
        category=ALL_CATEGORIES,
        required_metrics=("cfo", "net_income"),
        lookback_years=4,
        detection_fn=_cash_conversion_quality_divergence_detect,
        outcome_fn=_cash_conversion_quality_divergence_outcome,
    ),
    PatternDefinition(
        pattern_id="capital_return_inflection",
        name="Capital Return Inflection",
        hypothesis="Share count declining after 2+ years of dilution signals a transition toward shareholder returns and stronger per-share cash generation.",
        category=ALL_CATEGORIES,
        required_metrics=("shares_outstanding", "fcf"),
        lookback_years=4,
        detection_fn=_capital_return_inflection_detect,
        outcome_fn=_capital_return_inflection_outcome,
    ),
    PatternDefinition(
        pattern_id="gross_margin_regime_change",
        name="Gross Margin Regime Change",
        hypothesis="A sustained gross-margin shift of more than 3 percentage points from the prior 3-year average predicts a strengthening or weakening business model.",
        category=ALL_CATEGORIES,
        required_metrics=("gross_profit", "revenue", "operating_income"),
        lookback_years=5,
        detection_fn=_gross_margin_regime_change_detect,
        outcome_fn=_gross_margin_regime_change_outcome,
    ),
)


def get_pattern_definitions(pattern_ids: list[str] | None = None) -> list[PatternDefinition]:
    if not pattern_ids:
        return list(PATTERN_CATALOG)
    wanted = {str(pattern_id).strip() for pattern_id in pattern_ids if str(pattern_id).strip()}
    return [definition for definition in PATTERN_CATALOG if definition.pattern_id in wanted]


def get_pattern_definition(pattern_id: str) -> PatternDefinition | None:
    for definition in PATTERN_CATALOG:
        if definition.pattern_id == pattern_id:
            return definition
    return None
