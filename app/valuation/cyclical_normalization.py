"""
Cyclical Normalization Discipline v1

Detects whether a business has cyclical earnings patterns and provides
conservative cycle-normalized denominators for valuation.

Doctrine:
- Cyclicality detected from earnings variability only (no sector labels, no ML)
- Conservative: prefer multi-year median to avoid peak-earnings mispricing
- UNKNOWN remains UNKNOWN with explicit reason codes
- Missing evidence is not treated as cyclical or non-cyclical
- Peak earnings risk is flagged explicitly when current earnings elevated vs cycle normal
- Loss years are part of the cycle: variability and the normalized denominator
  count them rather than looking at profitable years only
- The denominator normalizes MARGINS and applies them to current revenue;
  averaging past earnings levels understates a grower and overstates a shrinker
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median, stdev
from typing import Any

from app.config import AppConfig, get_config
from app.db import utc_now_iso
from app.ingest.companyfacts import TAG_MAP
from app.valuation.adjacent_years import trailing_adjacent_run
from app.valuation.owner_earnings import _annual_series_for_priority, _is_annual_row


UNKNOWN = "UNKNOWN"
OK = "OK"

# Cyclical profile classes
CLEARLY_CYCLICAL = "CLEARLY_CYCLICAL"
MODERATELY_CYCLICAL = "MODERATELY_CYCLICAL"
LOW_CYCLICALITY = "LOW_CYCLICALITY"
CYCLICALITY_UNKNOWN = "CYCLICALITY_UNKNOWN"

# Cycle position classes
DEPRESSED_RELATIVE_TO_NORMAL = "DEPRESSED_RELATIVE_TO_NORMAL"
NEAR_NORMAL = "NEAR_NORMAL"
ELEVATED_RELATIVE_TO_NORMAL = "ELEVATED_RELATIVE_TO_NORMAL"
CYCLE_POSITION_UNKNOWN = "CYCLE_POSITION_UNKNOWN"

# Cyclical valuation risk classes
PEAK_EARNINGS_RISK = "PEAK_EARNINGS_RISK"
TROUGH_EARNINGS_RISK = "TROUGH_EARNINGS_RISK"
MID_CYCLE_REASONABLE = "MID_CYCLE_REASONABLE"
CYCLE_RISK_UNKNOWN = "CYCLE_RISK_UNKNOWN"

# Reason codes
REASON_INSUFFICIENT_SERIES = "INSUFFICIENT_SERIES_FOR_CYCLICALITY"
REASON_SERIES_ALL_NEGATIVE = "SERIES_ALL_NEGATIVE"
REASON_DENOMINATOR_5Y_MEDIAN = "DENOMINATOR_5Y_MEDIAN"
REASON_DENOMINATOR_3Y_MEDIAN = "DENOMINATOR_3Y_MEDIAN"
REASON_DENOMINATOR_INSUFFICIENT = "DENOMINATOR_INSUFFICIENT"
REASON_NON_ADJACENT_YEARS_DROPPED = "NON_ADJACENT_YEARS_DROPPED"
REASON_CYCLE_POSITION_NO_LATEST = "CYCLE_POSITION_NO_LATEST"
REASON_CYCLE_POSITION_NO_MEDIAN = "CYCLE_POSITION_NO_MEDIAN"
REASON_HIGH_COV = "HIGH_COV"
REASON_MODERATE_COV = "MODERATE_COV"
REASON_LOW_COV = "LOW_COV"
REASON_OWNER_EARNINGS_SERIES_USED = "OWNER_EARNINGS_SERIES_USED"
REASON_FCF_SERIES_USED = "FCF_SERIES_USED"
REASON_CFO_SERIES_USED = "CFO_SERIES_USED"
REASON_SINGLE_POINT_SERIES = "SINGLE_POINT_SERIES"
REASON_DENOMINATOR_5Y_MARGIN = "DENOMINATOR_5Y_MARGIN_X_LATEST_REVENUE"
REASON_DENOMINATOR_3Y_MARGIN = "DENOMINATOR_3Y_MARGIN_X_LATEST_REVENUE"
REASON_DENOMINATOR_LATEST_MARGIN = "DENOMINATOR_LATEST_MARGIN_X_LATEST_REVENUE"
REASON_DENOMINATOR_REVENUE_MISSING = "DENOMINATOR_REVENUE_MISSING"
REASON_DENOMINATOR_NONPOSITIVE = "DENOMINATOR_NONPOSITIVE"
REASON_ROWS_WITHOUT_YEAR = "SERIES_ROWS_WITHOUT_YEAR_DROPPED"
REASON_CONFLICTING_DUPLICATE_YEAR = "SERIES_CONFLICTING_DUPLICATE_YEAR_DROPPED"
REASON_LOSS_YEARS_INCLUDED = "LOSS_YEARS_IN_VARIABILITY"
REASON_LOSSES_OFFSET_PROFITS = "LOSSES_OFFSET_PROFITS"

# CoV thresholds for cyclicality detection
_COV_CLEARLY_CYCLICAL = 0.50
_COV_MODERATELY_CYCLICAL = 0.25

# Cycle position thresholds
_DEPRESSED_THRESHOLD = 0.75   # latest < median * 0.75
_ELEVATED_THRESHOLD = 1.25    # latest > median * 1.25


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dedupe_refs(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        token = str(v or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


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


def _row_year(row: dict[str, Any]) -> int | None:
    """The fiscal year a row belongs to, or None when it states no usable year."""
    raw = row.get("year")
    if isinstance(raw, bool):
        return None
    try:
        year = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return year if year > 0 else None


def _usable_rows(
    series: list[dict[str, Any]], *, adjacent_only: bool = True
) -> tuple[list[tuple[int, float]], list[str]]:
    """One (year, value) per fiscal year, oldest first, and why rows were dropped.

    With ``adjacent_only`` (the default) only the run of calendar-adjacent
    fiscal years ending at the latest year is kept: a gap used to let an old
    year sit in the "recent" window (2016 beside 2021-2024). Years cut off by a
    gap are reported as NON_ADJACENT_YEARS_DROPPED. The revenue series passes
    False: it is only looked up by year.

    A row with no usable year used to sort as year zero and count as the oldest
    observation; it is dropped. A fiscal year that appears twice used to count
    twice; a repeat with the same value counts once, and a repeat with a
    DIFFERENT value is dropped altogether, because nothing here says which of
    the two the filer meant.
    """
    reasons: list[str] = []
    by_year: dict[int, set[float]] = {}
    for row in series:
        if not isinstance(row, dict) or not _is_num(row.get("value")):
            continue
        year = _row_year(row)
        if year is None:
            reasons.append(REASON_ROWS_WITHOUT_YEAR)
            continue
        by_year.setdefault(year, set()).add(float(row["value"]))
    rows: list[tuple[int, float]] = []
    for year in sorted(by_year):
        values = by_year[year]
        if len(values) > 1:
            reasons.append(REASON_CONFLICTING_DUPLICATE_YEAR)
            continue
        rows.append((year, next(iter(values))))
    if adjacent_only:
        run = trailing_adjacent_run(rows)
        if len(run) < len(rows):
            reasons.append(REASON_NON_ADJACENT_YEARS_DROPPED)
        rows = run
    return rows, _dedupe_refs(reasons)


def _extract_positive_values(series: list[dict[str, Any]]) -> list[float]:
    """Return positive numeric values from a series, one per fiscal year, oldest first."""
    return [value for _, value in _usable_rows(series)[0] if value > 0.0]


def _extract_all_numeric(series: list[dict[str, Any]]) -> list[float]:
    """Return all numeric values from a series, one per fiscal year, oldest first."""
    return [value for _, value in _usable_rows(series)[0]]


def _cov(values: list[float]) -> float | None:
    """Coefficient of variation (stdev / mean) for a list of positive values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean <= 0.0:
        return None
    try:
        sd = stdev(values)
    except Exception:
        return None
    return sd / mean


def _select_best_series(
    *,
    owner_earnings_series: list[dict[str, Any]],
    fcf_series: list[dict[str, Any]],
    cfo_series: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str]:
    """Select the series with the most usable data points (5+ positive preferred).
    Priority: owner_earnings > fcf > cfo.

    Below five positive years the series with more POSITIVE years wins, then
    the one with more usable years, then priority. Counting all points first
    let a series with no positive year beat one with three.
    """
    candidates = [
        (owner_earnings_series, REASON_OWNER_EARNINGS_SERIES_USED, "owner_earnings"),
        (fcf_series, REASON_FCF_SERIES_USED, "fcf"),
        (cfo_series, REASON_CFO_SERIES_USED, "cfo"),
    ]
    # First pass: prefer series with 5+ positive data points
    for series, reason, name in candidates:
        if len(_extract_positive_values(series)) >= 5:
            return series, reason, name
    best: tuple[int, int, int] | None = None
    chosen: tuple[list[dict[str, Any]], str, str] | None = None
    for index, (series, reason, name) in enumerate(candidates):
        usable = len(_extract_all_numeric(series))
        if usable == 0:
            continue
        key = (len(_extract_positive_values(series)), usable, -index)
        if best is None or key > best:
            best = key
            chosen = (series, reason, name)
    if chosen is not None:
        return chosen
    return [], REASON_INSUFFICIENT_SERIES, "none"


def _detect_cyclicality(
    positive_values: list[float],
    *,
    all_values: list[float],
) -> tuple[str, list[str]]:
    """Detect cyclical profile from earnings variability (CoV-based, no sector labels)."""
    reason_codes: list[str] = []
    if len(positive_values) < 3:
        if len(all_values) >= 3:
            # This named shortcut is supported only when every observation is
            # negative. Mixed or zero-only histories still lack three positive
            # observations for the coefficient-of-variation model.
            if all(value < 0.0 for value in all_values):
                reason_codes.append(REASON_SERIES_ALL_NEGATIVE)
                return CLEARLY_CYCLICAL, reason_codes
        return CYCLICALITY_UNKNOWN, [REASON_INSUFFICIENT_SERIES]

    # Loss years are the swings this measure exists to see. Measured over the
    # profitable years alone, 200, -100, 200, -100, 200 read as LOW cyclicality.
    if any(value <= 0.0 for value in all_values):
        reason_codes.append(REASON_LOSS_YEARS_INCLUDED)
        if sum(all_values) <= 0.0:
            # The losses cancel the profits: no mean to scale by, and no
            # steadier profile than the most cyclical one.
            reason_codes.append(REASON_LOSSES_OFFSET_PROFITS)
            return CLEARLY_CYCLICAL, reason_codes
        cov = _cov(all_values)
    else:
        cov = _cov(positive_values)
    if cov is None:
        return CYCLICALITY_UNKNOWN, [REASON_INSUFFICIENT_SERIES]

    if cov >= _COV_CLEARLY_CYCLICAL:
        reason_codes.append(REASON_HIGH_COV)
        return CLEARLY_CYCLICAL, reason_codes
    elif cov >= _COV_MODERATELY_CYCLICAL:
        reason_codes.append(REASON_MODERATE_COV)
        return MODERATELY_CYCLICAL, reason_codes
    else:
        reason_codes.append(REASON_LOW_COV)
        return LOW_CYCLICALITY, reason_codes


def _compute_cycle_position(
    series: list[dict[str, Any]],
    *,
    positive_values: list[float],
) -> tuple[str, float | str, list[str]]:
    """Assess cycle position by comparing latest value to multi-year median."""
    if not positive_values:
        return CYCLE_POSITION_UNKNOWN, UNKNOWN, [REASON_CYCLE_POSITION_NO_MEDIAN]
    all_rows = _usable_rows(series)[0]
    if not all_rows:
        return CYCLE_POSITION_UNKNOWN, UNKNOWN, [REASON_CYCLE_POSITION_NO_LATEST]

    latest_value = float(all_rows[-1][1])
    window = positive_values[-5:]
    cycle_median = float(median(window))
    if cycle_median <= 0.0:
        return CYCLE_POSITION_UNKNOWN, UNKNOWN, [REASON_CYCLE_POSITION_NO_MEDIAN]

    ratio = latest_value / cycle_median
    if ratio < _DEPRESSED_THRESHOLD:
        cycle_position = DEPRESSED_RELATIVE_TO_NORMAL
    elif ratio > _ELEVATED_THRESHOLD:
        cycle_position = ELEVATED_RELATIVE_TO_NORMAL
    else:
        cycle_position = NEAR_NORMAL

    return cycle_position, round(ratio, 4), []


def _compute_conservative_denominator(
    rows: list[tuple[int, float]],
    *,
    revenue_rows: list[tuple[int, float]],
    cyclical_profile_class: str,
) -> tuple[float | str, str, list[str]]:
    """Conservative cycle-normalized earnings: a normalized margin times current revenue.

    The window is the last five usable years for a cyclical profile (three when
    only three or four exist) and the last three otherwise, loss years
    included. Each year's earnings are divided by that year's revenue; the
    normalized margin is the LOWER of the window's median and mean margin (the
    median guards against a peak year, the mean against loss years the median
    would step over), applied to the latest revenue. Averaging earnings levels
    instead understated a grower and overstated a shrinker. A year in the
    window without positive revenue, or a latest revenue older than the latest
    earnings year, leaves the denominator UNKNOWN; so does a normalized result
    at or below zero, which is no denominator at all.
    """
    positives = [value for _, value in rows if value > 0.0]
    if not positives:
        return UNKNOWN, REASON_DENOMINATOR_INSUFFICIENT, [REASON_DENOMINATOR_INSUFFICIENT]

    if cyclical_profile_class in {CLEARLY_CYCLICAL, MODERATELY_CYCLICAL}:
        if len(rows) >= 5 and len(positives) >= 3:
            window, method = rows[-5:], REASON_DENOMINATOR_5Y_MARGIN
        elif len(rows) >= 3 and len(positives) >= 3:
            window, method = rows[-3:], REASON_DENOMINATOR_3Y_MARGIN
        else:
            return UNKNOWN, REASON_DENOMINATOR_INSUFFICIENT, [REASON_DENOMINATOR_INSUFFICIENT]
    elif len(rows) >= 3:
        window, method = rows[-3:], REASON_DENOMINATOR_3Y_MARGIN
    else:
        window, method = rows[-1:], REASON_DENOMINATOR_LATEST_MARGIN

    revenue_by_year = {year: value for year, value in revenue_rows if value > 0.0}
    if not revenue_by_year or any(year not in revenue_by_year for year, _ in window):
        return UNKNOWN, REASON_DENOMINATOR_REVENUE_MISSING, [REASON_DENOMINATOR_REVENUE_MISSING]
    latest_revenue_year = max(revenue_by_year)
    if latest_revenue_year < window[-1][0]:
        return UNKNOWN, REASON_DENOMINATOR_REVENUE_MISSING, [REASON_DENOMINATOR_REVENUE_MISSING]
    margins = [value / revenue_by_year[year] for year, value in window]
    margin = min(float(median(margins)), sum(margins) / len(margins))
    denominator = margin * revenue_by_year[latest_revenue_year]
    if denominator <= 0.0:
        return UNKNOWN, REASON_DENOMINATOR_NONPOSITIVE, [REASON_DENOMINATOR_NONPOSITIVE]
    reasons = [method] if method != REASON_DENOMINATOR_LATEST_MARGIN else [
        method,
        REASON_SINGLE_POINT_SERIES,
    ]
    return float(denominator), method, reasons


def _compute_valuation_risk(
    *,
    cyclical_profile_class: str,
    cycle_position_class: str,
) -> tuple[str, str]:
    """Compute cyclical valuation risk class and cycle-aware value support summary."""
    if cyclical_profile_class == CYCLICALITY_UNKNOWN or cycle_position_class == CYCLE_POSITION_UNKNOWN:
        return (
            CYCLE_RISK_UNKNOWN,
            "Cyclical valuation risk cannot be assessed without sufficient earnings history.",
        )

    if cyclical_profile_class in {CLEARLY_CYCLICAL, MODERATELY_CYCLICAL}:
        if cycle_position_class == ELEVATED_RELATIVE_TO_NORMAL:
            return (
                PEAK_EARNINGS_RISK,
                "Current earnings appear elevated relative to cycle normal. "
                "Valuation based on current earnings may overstate sustainable earning power. "
                "Use cycle-normalized denominator rather than current period metrics.",
            )
        elif cycle_position_class == DEPRESSED_RELATIVE_TO_NORMAL:
            return (
                TROUGH_EARNINGS_RISK,
                "Current earnings appear depressed relative to cycle normal. "
                "Current yield may understate normalized earning power. "
                "Apparent richness may be cyclical rather than structural.",
            )
        else:
            return (
                MID_CYCLE_REASONABLE,
                "Current earnings appear near cycle-normal range. "
                "Conservative multi-year median denominator preferred for cyclical discipline.",
            )
    else:
        if cycle_position_class == ELEVATED_RELATIVE_TO_NORMAL:
            return (
                PEAK_EARNINGS_RISK,
                "Earnings elevated vs recent history though cyclicality appears low. "
                "Conservative 3Y median denominator preferred.",
            )
        else:
            return (
                MID_CYCLE_REASONABLE,
                "Earnings cyclicality appears low. "
                "Standard 3Y median normalization is appropriate.",
            )


def compute_cyclical_normalization(
    ticker: str,
    as_of_date: str,
    *,
    owner_earnings_series: list[dict[str, Any]] | None = None,
    fcf_series: list[dict[str, Any]] | None = None,
    cfo_series: list[dict[str, Any]] | None = None,
    revenue_series: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compute cyclical normalization discipline for a single ticker.

    ``revenue_series`` ({"year", "value"} rows on the same fiscal-year basis as
    the earnings series) is what the conservative denominator normalizes
    margins against; without it the denominator is UNKNOWN.

    All detection is from earnings series variability only — no sector labels, no ML.
    UNKNOWN remains UNKNOWN with explicit reason codes.
    Missing evidence is not treated as cyclical or non-cyclical.
    """
    ticker_norm = str(ticker or "").strip().upper()
    owner_series = list(owner_earnings_series or [])
    fcf = list(fcf_series or [])
    cfo = list(cfo_series or [])

    selected_series, series_reason, series_name = _select_best_series(
        owner_earnings_series=owner_series,
        fcf_series=fcf,
        cfo_series=cfo,
    )

    selected_rows, row_reasons = _usable_rows(selected_series)
    revenue_rows, _ = _usable_rows(list(revenue_series or []), adjacent_only=False)
    reason_codes: list[str] = [series_reason, *row_reasons]
    derived_from: list[str] = []
    for row in selected_series:
        for ref in (row.get("derived_from") or []):
            if str(ref or "").strip():
                derived_from.append(str(ref))

    positive_values = _extract_positive_values(selected_series)
    all_values = _extract_all_numeric(selected_series)

    # 1. Cyclicality detection
    cyclical_profile_class, profile_reasons = _detect_cyclicality(
        positive_values, all_values=all_values
    )
    reason_codes.extend(profile_reasons)

    # 2. Cycle position assessment
    cycle_position_class, cycle_position_ratio, position_reasons = _compute_cycle_position(
        selected_series, positive_values=positive_values
    )
    reason_codes.extend(position_reasons)

    # 3. Conservative cyclical denominator
    conservative_denominator, denominator_method, denominator_reasons = _compute_conservative_denominator(
        selected_rows,
        revenue_rows=revenue_rows,
        cyclical_profile_class=cyclical_profile_class,
    )
    reason_codes.extend(denominator_reasons)

    # 4. Peak/trough valuation risk and support summary
    cyclical_valuation_risk_class, cycle_aware_value_support_summary = _compute_valuation_risk(
        cyclical_profile_class=cyclical_profile_class,
        cycle_position_class=cycle_position_class,
    )

    return {
        "ticker": ticker_norm,
        "as_of_date": as_of_date,
        "cyclical_profile_class": cyclical_profile_class,
        "cycle_position_class": cycle_position_class,
        "cycle_position_ratio": cycle_position_ratio,
        "conservative_cyclical_denominator": conservative_denominator,
        "conservative_cyclical_denominator_method": denominator_method,
        "cyclical_valuation_risk_class": cyclical_valuation_risk_class,
        "cycle_aware_value_support_summary": cycle_aware_value_support_summary,
        "series_used": series_name,
        "series_points_available": len(all_values),
        "positive_series_points": len(positive_values),
        "cyclical_normalization_reason_codes": _dedupe_refs(reason_codes),
        "derived_from": _dedupe_refs(derived_from),
        "generated_at": utc_now_iso(),
    }


def annual_revenue_series(
    *, companyfacts: dict[str, Any], as_of_date: str
) -> list[dict[str, Any]]:
    """Full-year revenue by fiscal year, on the owner-earnings series' year basis.

    Only rows that cover a full year are kept: a margin of annual earnings over
    a quarter's revenue would be four times too high.
    """
    if not isinstance(companyfacts, dict) or not companyfacts:
        return []
    rows = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date=as_of_date,
        priority=[("us-gaap", tag) for tag in TAG_MAP["revenue"]],
        expected_unit_exact=("usd",),
    )
    return [
        {
            "year": int(row["year"]),
            "value": float(row["value"]),
            "derived_from": [str(row.get("ref") or "")] if row.get("ref") else [],
        }
        for row in rows
        if _is_annual_row(row) and _is_num(row.get("value"))
    ]


def write_cyclical_normalization_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    output_path: Path,
    scoreboard_rows: list[dict[str, Any]] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Write cyclical_normalization.json artifact for a universe run."""
    cfg = cfg or get_config()
    del cfg

    detail_by_ticker = {
        str(row.get("ticker") or "").upper(): row.get("cyclical_normalization_detail")
        for row in (scoreboard_rows or [])
        if isinstance(row, dict) and isinstance(row.get("cyclical_normalization_detail"), dict)
    }

    rows: list[dict[str, Any]] = []
    counts_by_profile: dict[str, int] = {}
    counts_by_position: dict[str, int] = {}
    counts_by_risk: dict[str, int] = {}

    for ticker in tickers:
        ticker_norm = str(ticker).upper()
        detail = detail_by_ticker.get(ticker_norm)
        if not isinstance(detail, dict):
            detail = {
                "ticker": ticker_norm,
                "cyclical_profile_class": CYCLICALITY_UNKNOWN,
                "cycle_position_class": CYCLE_POSITION_UNKNOWN,
                "cyclical_valuation_risk_class": CYCLE_RISK_UNKNOWN,
                "conservative_cyclical_denominator": UNKNOWN,
                "conservative_cyclical_denominator_method": REASON_DENOMINATOR_INSUFFICIENT,
                "cycle_position_ratio": UNKNOWN,
                "cycle_aware_value_support_summary": "No cyclical normalization detail available",
                "series_used": UNKNOWN,
                "series_points_available": 0,
                "cyclical_normalization_reason_codes": [REASON_INSUFFICIENT_SERIES],
                "derived_from": [],
            }
        profile = str(detail.get("cyclical_profile_class") or CYCLICALITY_UNKNOWN)
        position = str(detail.get("cycle_position_class") or CYCLE_POSITION_UNKNOWN)
        risk = str(detail.get("cyclical_valuation_risk_class") or CYCLE_RISK_UNKNOWN)
        counts_by_profile[profile] = counts_by_profile.get(profile, 0) + 1
        counts_by_position[position] = counts_by_position.get(position, 0) + 1
        counts_by_risk[risk] = counts_by_risk.get(risk, 0) + 1
        rows.append({
            "ticker": ticker_norm,
            "cyclical_profile_class": profile,
            "cycle_position_class": position,
            "cyclical_valuation_risk_class": risk,
            "conservative_cyclical_denominator": detail.get("conservative_cyclical_denominator", UNKNOWN),
            "conservative_cyclical_denominator_method": detail.get("conservative_cyclical_denominator_method", REASON_DENOMINATOR_INSUFFICIENT),
            "cycle_position_ratio": detail.get("cycle_position_ratio", UNKNOWN),
            "cycle_aware_value_support_summary": detail.get("cycle_aware_value_support_summary", ""),
            "series_used": detail.get("series_used", UNKNOWN),
            "series_points_available": int(detail.get("series_points_available") or 0),
            "cyclical_normalization_reason_codes": list(detail.get("cyclical_normalization_reason_codes") or []),
        })

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(tickers),
        "counts_by_cyclical_profile_class": counts_by_profile,
        "counts_by_cycle_position_class": counts_by_position,
        "counts_by_cyclical_valuation_risk_class": counts_by_risk,
        "rows": rows,
        "generated_at": utc_now_iso(),
    }
    _json_write(output_path, payload)
    return payload


def _cyclical_normalization_path(run_id: str) -> Path | None:
    cfg = get_config()
    candidates = [
        cfg.outputs_dir / "universe" / run_id / "cyclical_normalization.json",
        cfg.sectors_dir / run_id / "cyclical_normalization.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_cyclical_normalization(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    """Open and summarize the cyclical_normalization.json artifact for a run."""
    path = _cyclical_normalization_path(run_id)
    if path is None or not path.exists():
        return {
            "status": "MISSING",
            "run_id": run_id,
            "cyclical_normalization_path": str(path) if path is not None else "",
        }
    payload = _safe_json(path)
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    return {
        "status": "OK",
        "run_id": run_id,
        "ticker_count": int(payload.get("ticker_count") or 0),
        "counts_by_cyclical_profile_class": (
            payload.get("counts_by_cyclical_profile_class")
            if isinstance(payload.get("counts_by_cyclical_profile_class"), dict)
            else {}
        ),
        "counts_by_cycle_position_class": (
            payload.get("counts_by_cycle_position_class")
            if isinstance(payload.get("counts_by_cycle_position_class"), dict)
            else {}
        ),
        "counts_by_cyclical_valuation_risk_class": (
            payload.get("counts_by_cyclical_valuation_risk_class")
            if isinstance(payload.get("counts_by_cyclical_valuation_risk_class"), dict)
            else {}
        ),
        "top_rows": rows[:top_n],
        "cyclical_normalization_path": str(path),
    }
