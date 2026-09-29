"""Tests for app/valuation/cyclical_normalization.py — fully offline, no real data."""
from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

import pytest

from app.valuation.cyclical_normalization import (
    CLEARLY_CYCLICAL,
    CYCLE_POSITION_UNKNOWN,
    CYCLE_RISK_UNKNOWN,
    CYCLICALITY_UNKNOWN,
    DEPRESSED_RELATIVE_TO_NORMAL,
    ELEVATED_RELATIVE_TO_NORMAL,
    LOW_CYCLICALITY,
    MID_CYCLE_REASONABLE,
    MODERATELY_CYCLICAL,
    NEAR_NORMAL,
    PEAK_EARNINGS_RISK,
    TROUGH_EARNINGS_RISK,
    compute_cyclical_normalization,
    open_cyclical_normalization,
    write_cyclical_normalization_for_run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _oe_series(values: list[float], start_year: int = 2019) -> list[dict]:
    return [
        {"year": start_year + i, "value": v, "derived_from": []}
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# 1. Clearly cyclical — high CoV
# ---------------------------------------------------------------------------

def test_clearly_cyclical_detection():
    # Wide swings → CoV well above 0.50
    oe = _oe_series([100.0, 10.0, 200.0, 5.0, 150.0, 8.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cyclical_profile_class"] == CLEARLY_CYCLICAL


# ---------------------------------------------------------------------------
# 2. Moderately cyclical — CoV between 0.25 and 0.50
# ---------------------------------------------------------------------------

def test_moderately_cyclical_detection():
    # Moderate swings → CoV ~0.36 (between 0.25 and 0.50)
    # Mean ≈ 86.7, stdev ≈ 31.6 → CoV ≈ 0.36
    oe = _oe_series([100.0, 50.0, 130.0, 55.0, 120.0, 65.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cyclical_profile_class"] == MODERATELY_CYCLICAL


# ---------------------------------------------------------------------------
# 3. Low cyclicality — CoV below 0.25
# ---------------------------------------------------------------------------

def test_low_cyclicality_detection():
    # Stable earnings → CoV well below 0.25
    oe = _oe_series([100.0, 102.0, 104.0, 103.0, 101.0, 105.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cyclical_profile_class"] == LOW_CYCLICALITY


# ---------------------------------------------------------------------------
# 4. Cyclicality unknown — insufficient series
# ---------------------------------------------------------------------------

def test_cyclicality_unknown_insufficient_data():
    # Only 2 positive points — insufficient for CoV-based cyclicality detection
    oe = _oe_series([100.0, 90.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cyclical_profile_class"] == CYCLICALITY_UNKNOWN
    # Note: cycle_position may still be computed from available values;
    # risk class will be CYCLE_RISK_UNKNOWN due to CYCLICALITY_UNKNOWN
    assert result["cyclical_valuation_risk_class"] == CYCLE_RISK_UNKNOWN


def test_cyclicality_unknown_empty_series():
    # No data at all — all classes must be UNKNOWN
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=[], fcf_series=[], cfo_series=[],
    )
    assert result["cyclical_profile_class"] == CYCLICALITY_UNKNOWN
    assert result["cycle_position_class"] == CYCLE_POSITION_UNKNOWN
    assert result["cyclical_valuation_risk_class"] == CYCLE_RISK_UNKNOWN


# ---------------------------------------------------------------------------
# 5. Cycle position: DEPRESSED (latest well below median)
# ---------------------------------------------------------------------------

def test_cycle_position_depressed():
    # Latest value is 40 — well below the 5Y median of ~100
    oe = _oe_series([100.0, 110.0, 95.0, 105.0, 100.0, 40.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cycle_position_class"] == DEPRESSED_RELATIVE_TO_NORMAL


# ---------------------------------------------------------------------------
# 6. Cycle position: NEAR_NORMAL
# ---------------------------------------------------------------------------

def test_cycle_position_near_normal():
    # Latest value close to median
    oe = _oe_series([100.0, 102.0, 98.0, 101.0, 99.0, 100.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cycle_position_class"] == NEAR_NORMAL


# ---------------------------------------------------------------------------
# 7. Cycle position: ELEVATED (latest well above median)
# ---------------------------------------------------------------------------

def test_cycle_position_elevated():
    # Latest value is 200 — well above median of ~100
    oe = _oe_series([100.0, 95.0, 105.0, 98.0, 100.0, 200.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cycle_position_class"] == ELEVATED_RELATIVE_TO_NORMAL


# ---------------------------------------------------------------------------
# 8. PEAK_EARNINGS_RISK — clearly/moderately cyclical + elevated
# ---------------------------------------------------------------------------

def test_peak_earnings_risk():
    # High CoV (clearly cyclical) + elevated latest
    oe = _oe_series([50.0, 10.0, 80.0, 5.0, 60.0, 300.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cyclical_valuation_risk_class"] == PEAK_EARNINGS_RISK


# ---------------------------------------------------------------------------
# 9. TROUGH_EARNINGS_RISK — clearly/moderately cyclical + depressed
# ---------------------------------------------------------------------------

def test_trough_earnings_risk():
    # High CoV (clearly cyclical) + depressed latest
    oe = _oe_series([100.0, 10.0, 150.0, 8.0, 120.0, 5.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cyclical_valuation_risk_class"] == TROUGH_EARNINGS_RISK


# ---------------------------------------------------------------------------
# 10. MID_CYCLE_REASONABLE — clearly cyclical + near normal
# ---------------------------------------------------------------------------

def test_mid_cycle_reasonable():
    # High CoV but latest near median
    oe = _oe_series([200.0, 10.0, 200.0, 10.0, 105.0, 100.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    assert result["cyclical_valuation_risk_class"] == MID_CYCLE_REASONABLE


# ---------------------------------------------------------------------------
# 11. Conservative denominator — 5Y median for clearly cyclical
# ---------------------------------------------------------------------------

def test_conservative_denominator_5y_for_cyclical():
    # Migrated 2026-09-28: the denominator normalizes MARGINS against
    # revenue, so the fixture now supplies revenue. Flat revenue of 1,000: the
    # last five margins are 1%, 20%, 0.5%, 15%, 0.8%; the lower of their median
    # (1%) and mean (7.46%) times the latest revenue is 10.
    oe = _oe_series([100.0, 10.0, 200.0, 5.0, 150.0, 8.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
        revenue_series=_oe_series([1000.0] * 6),
    )
    assert result["cyclical_profile_class"] == CLEARLY_CYCLICAL
    assert result["conservative_cyclical_denominator"] == pytest.approx(10.0, abs=1e-9)
    assert (
        result["conservative_cyclical_denominator_method"]
        == "DENOMINATOR_5Y_MARGIN_X_LATEST_REVENUE"
    )


# ---------------------------------------------------------------------------
# 12. Conservative denominator — 3Y median for low cyclicality
# ---------------------------------------------------------------------------

def test_conservative_denominator_3y_for_low_cyclicality():
    # Migrated 2026-09-28: revenue supplied; margins 10.3%, 10.1%,
    # 10.5% on flat revenue of 1,000 give 103.
    oe = _oe_series([100.0, 102.0, 104.0, 103.0, 101.0, 105.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
        revenue_series=_oe_series([1000.0] * 6),
    )
    assert result["cyclical_profile_class"] == LOW_CYCLICALITY
    assert result["conservative_cyclical_denominator"] == pytest.approx(103.0, abs=1e-9)
    assert (
        result["conservative_cyclical_denominator_method"]
        == "DENOMINATOR_3Y_MARGIN_X_LATEST_REVENUE"
    )


# ---------------------------------------------------------------------------
# 13. FCF fallback when OE not available
# ---------------------------------------------------------------------------

def test_fcf_fallback_series():
    fcf = _oe_series([80.0, 10.0, 160.0, 5.0, 120.0, 8.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=[], fcf_series=fcf, cfo_series=[],
    )
    assert result["series_used"] in {"fcf", "owner_earnings", "cfo"}
    assert result["cyclical_profile_class"] != CYCLICALITY_UNKNOWN


# ---------------------------------------------------------------------------
# 14. write_cyclical_normalization_for_run — writes valid JSON artifact
# ---------------------------------------------------------------------------

def test_write_cyclical_normalization_for_run(tmp_path):
    oe = _oe_series([100.0, 10.0, 200.0, 5.0, 150.0, 100.0])
    payload = compute_cyclical_normalization(
        ticker="AAPL", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    rows = [{"ticker": "AAPL", "cyclical_normalization_detail": payload}]
    out_file = tmp_path / "cyclical_normalization.json"

    write_cyclical_normalization_for_run(
        run_id="test_run",
        as_of_date="2025-01-01",
        tickers=["AAPL"],
        output_path=out_file,
        scoreboard_rows=rows,
    )

    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "rows" in data
    tickers_in_rows = {r["ticker"] for r in data["rows"] if isinstance(r, dict)}
    assert "AAPL" in tickers_in_rows


# ---------------------------------------------------------------------------
# 15. open_cyclical_normalization — returns MISSING when file not present
# ---------------------------------------------------------------------------

def test_open_cyclical_normalization_missing():
    with patch("app.valuation.cyclical_normalization._cyclical_normalization_path") as mock_path:
        mock_path.return_value = pathlib.Path("/nonexistent/path/cyclical_normalization.json")
        result = open_cyclical_normalization(run_id="nonexistent_run", top_n=5)
    assert result["status"] == "MISSING"


# ---------------------------------------------------------------------------
# 16. Output fields completeness
# ---------------------------------------------------------------------------

def test_output_fields_completeness():
    oe = _oe_series([100.0, 70.0, 110.0, 75.0, 105.0, 80.0])
    result = compute_cyclical_normalization(
        ticker="TEST", as_of_date="2025-01-01",
        owner_earnings_series=oe, fcf_series=[], cfo_series=[],
    )
    required_fields = {
        "cyclical_profile_class",
        "cycle_position_class",
        "cyclical_valuation_risk_class",
        "conservative_cyclical_denominator",
        "cycle_position_ratio",
        "cycle_aware_value_support_summary",
        "series_used",
        "series_points_available",
        "positive_series_points",
        "cyclical_normalization_reason_codes",
        "derived_from",
        "generated_at",
    }
    for field in required_fields:
        assert field in result, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Cyclical-normalization fixes
# ---------------------------------------------------------------------------

def _grow(values: list[float], start_year: int = 2020) -> list[dict]:
    return [{"year": start_year + i, "value": v} for i, v in enumerate(values)]


def test_denominator_normalizes_margins_not_levels_for_a_grower_and_a_shrinker():
    """A steady 10% margin on revenue growing 1,000 -> 2,000 is 200 of
    normalized earnings today; the median of past earnings LEVELS said 150.
    The shrinker mirrors it: 100 today, not 150."""
    grower = compute_cyclical_normalization(
        "GROW", "2025-06-30",
        owner_earnings_series=_grow([100.0, 125.0, 150.0, 175.0, 200.0]),
        revenue_series=_grow([1000.0, 1250.0, 1500.0, 1750.0, 2000.0]),
    )
    assert grower["cyclical_profile_class"] == MODERATELY_CYCLICAL
    assert grower["conservative_cyclical_denominator"] == pytest.approx(200.0, abs=1e-9)
    shrinker = compute_cyclical_normalization(
        "SHRK", "2025-06-30",
        owner_earnings_series=_grow([200.0, 175.0, 150.0, 125.0, 100.0]),
        revenue_series=_grow([2000.0, 1750.0, 1500.0, 1250.0, 1000.0]),
    )
    assert shrinker["conservative_cyclical_denominator"] == pytest.approx(100.0, abs=1e-9)


def test_denominator_without_revenue_is_unknown_not_a_level_average():
    """With no revenue there is no margin to normalize; the level
    median is the biased number the row reports, so the answer is UNKNOWN."""
    result = compute_cyclical_normalization(
        "NOREV", "2025-06-30",
        owner_earnings_series=_grow([100.0, 125.0, 150.0, 175.0, 200.0]),
    )
    assert result["conservative_cyclical_denominator"] == "UNKNOWN"
    assert result["conservative_cyclical_denominator_method"] == "DENOMINATOR_REVENUE_MISSING"


def test_loss_years_pull_the_denominator_down_and_a_net_loss_window_is_unknown():
    """The denominator half: 200, -100, 200, -100, 200 on flat revenue of
    1,000 is 80 of through-cycle earnings (the mean margin, 8%), not 200."""
    result = compute_cyclical_normalization(
        "LOSS", "2026-01-01",
        owner_earnings_series=_grow([200.0, -100.0, 200.0, -100.0, 200.0], 2021),
        revenue_series=_grow([1000.0] * 5, 2021),
    )
    assert result["cyclical_profile_class"] == CLEARLY_CYCLICAL
    assert "LOSS_YEARS_IN_VARIABILITY" in result["cyclical_normalization_reason_codes"]
    assert result["conservative_cyclical_denominator"] == pytest.approx(80.0, abs=1e-9)
    net_loss = compute_cyclical_normalization(
        "NETL", "2026-01-01",
        owner_earnings_series=_grow([100.0, -300.0, 100.0, -300.0, 100.0], 2021),
        revenue_series=_grow([1000.0] * 5, 2021),
    )
    assert net_loss["cyclical_profile_class"] == CLEARLY_CYCLICAL
    assert "LOSSES_OFFSET_PROFITS" in net_loss["cyclical_normalization_reason_codes"]
    assert net_loss["conservative_cyclical_denominator"] == "UNKNOWN"
    assert net_loss["conservative_cyclical_denominator_method"] == "DENOMINATOR_NONPOSITIVE"


def test_rows_without_a_year_are_dropped_not_counted_as_the_oldest_year():
    """Two yearless rows used to sort as year zero, count as the two
    oldest observations and turn a flat 100, 100, 100 into CLEARLY_CYCLICAL."""
    result = compute_cyclical_normalization(
        "NOYR", "2025-06-30",
        owner_earnings_series=[
            {"year": None, "value": 500.0},
            {"value": 400.0},
            *_grow([100.0, 100.0, 100.0], 2022),
        ],
    )
    assert result["cyclical_profile_class"] == LOW_CYCLICALITY
    assert result["series_points_available"] == 3
    assert result["positive_series_points"] == 3
    assert "SERIES_ROWS_WITHOUT_YEAR_DROPPED" in result["cyclical_normalization_reason_codes"]


def test_a_repeated_fiscal_year_counts_once_and_a_conflicting_repeat_not_at_all():
    """The same year twice was weighted twice."""
    repeated = compute_cyclical_normalization(
        "DUP", "2025-06-30",
        owner_earnings_series=[*_grow([100.0, 100.0, 300.0], 2021), {"year": 2023, "value": 300.0}],
    )
    assert repeated["series_points_available"] == 3
    assert repeated["positive_series_points"] == 3
    conflicting = compute_cyclical_normalization(
        "CONF", "2025-06-30",
        owner_earnings_series=[*_grow([100.0, 100.0, 300.0], 2021), {"year": 2023, "value": 50.0}],
    )
    assert conflicting["series_points_available"] == 2
    assert conflicting["cyclical_profile_class"] == CYCLICALITY_UNKNOWN
    assert (
        "SERIES_CONFLICTING_DUPLICATE_YEAR_DROPPED"
        in conflicting["cyclical_normalization_reason_codes"]
    )


def test_series_selection_prefers_positive_evidence_over_point_count():
    """Three loss years of owner earnings used to beat three profitable
    years of free cash flow because only the point count was compared."""
    result = compute_cyclical_normalization(
        "SEL", "2025-06-30",
        owner_earnings_series=_grow([-1.0, -2.0, -3.0], 2022),
        fcf_series=_grow([100.0, 110.0, 105.0], 2022),
    )
    assert result["series_used"] == "fcf"
    assert result["positive_series_points"] == 3


def test_annual_revenue_series_keeps_full_years_only():
    """The revenue the denominator divides by must cover a full year: a
    quarter's revenue under an annual margin would inflate it fourfold."""
    from app.valuation.cyclical_normalization import annual_revenue_series

    def fact(start, end, value, form="10-K", fp="FY"):
        return {"start": start, "end": end, "val": value, "filed": "2026-02-15",
                "form": form, "fp": fp}

    companyfacts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        fact("2024-01-01", "2024-12-31", 1000.0),
        fact("2025-07-01", "2025-09-30", 300.0, form="10-Q", fp="Q3"),
    ]}}}}}
    rows = annual_revenue_series(companyfacts=companyfacts, as_of_date="2026-03-01")
    assert [(row["year"], row["value"]) for row in rows] == [(2024, 1000.0)]
