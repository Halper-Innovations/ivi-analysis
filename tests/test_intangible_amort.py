"""Tests for app.valuation.intangible_amort — EPV add-back for serial acquirers."""

from __future__ import annotations

from app.valuation.intangible_amort import (
    IntangibleAmortAddBack,
    IntangibleAmortSeries,
    adjust_operating_income_series,
    compute_series,
    compute_single_year_addback,
)


# ---------------------------------------------------------------------------
# compute_single_year_addback — pure math on a single year
# ---------------------------------------------------------------------------


def test_zero_addback_when_no_excess_da():
    """When D&A == capex, there's no excess to attribute to intangible amort."""
    r = compute_single_year_addback(
        fiscal_year=2025,
        d_and_a=100.0,
        capex=100.0,
        intangible_assets=5000.0,
    )
    assert r.addback == 0.0
    assert r.reason == "NO_EXCESS_DA_OVER_CAPEX"


def test_zero_addback_when_capex_exceeds_da():
    """Growth-mode companies where capex > D&A: no add-back."""
    r = compute_single_year_addback(
        fiscal_year=2025,
        d_and_a=50.0,
        capex=200.0,
        intangible_assets=1000.0,
    )
    assert r.addback == 0.0
    assert r.reason == "NO_EXCESS_DA_OVER_CAPEX"


def test_zero_addback_when_no_intangibles():
    """Excess D&A without intangibles on the balance sheet → don't add back.
    The excess is probably software/accelerated depreciation, not M&A amort."""
    r = compute_single_year_addback(
        fiscal_year=2025,
        d_and_a=300.0,
        capex=100.0,
        intangible_assets=0.0,
    )
    assert r.addback == 0.0
    assert r.reason == "NO_MATERIAL_INTANGIBLES"


def test_addback_capped_by_intangible_useful_life():
    """Excess $500 but intangibles only $3000 → cap at 10% = $300."""
    r = compute_single_year_addback(
        fiscal_year=2025,
        d_and_a=600.0,
        capex=100.0,
        intangible_assets=3000.0,
    )
    assert r.excess_da_over_capex == 500.0
    assert r.intangible_cap == 300.0
    assert r.addback == 300.0
    assert r.reason == "CAPPED_BY_INTANGIBLE_USEFUL_LIFE"


def test_addback_when_excess_below_cap():
    """Excess $80 but intangibles $5000 (cap $500) → use full excess."""
    r = compute_single_year_addback(
        fiscal_year=2025,
        d_and_a=180.0,
        capex=100.0,
        intangible_assets=5000.0,
    )
    assert r.excess_da_over_capex == 80.0
    assert r.addback == 80.0
    assert r.reason == "EXCESS_DA_BELOW_INTANGIBLE_CAP"


def test_missing_data_returns_zero_addback():
    """Any missing required input → zero add-back with clear reason."""
    r = compute_single_year_addback(
        fiscal_year=2025,
        d_and_a=None,
        capex=100.0,
        intangible_assets=1000.0,
    )
    assert r.addback == 0.0
    assert r.reason == "INSUFFICIENT_DATA_DA_OR_CAPEX"


def test_handles_negative_capex_convention():
    """Capex often arrives as negative (cash outflow). The math should use the
    absolute value: D&A=200, capex=-100 → abs(capex)=100 → excess=100."""
    r = compute_single_year_addback(
        fiscal_year=2025,
        d_and_a=200.0,
        capex=-100.0,
        intangible_assets=5000.0,
    )
    assert r.excess_da_over_capex == 100.0
    assert r.addback == 100.0


# ---------------------------------------------------------------------------
# compute_series — multi-year aggregation + materiality flag
# ---------------------------------------------------------------------------


def test_compute_series_coll_like_scenario():
    """COLL-style: $222M/yr intangible amort on ~$800M revenue = material distortion."""
    years = [
        {"fiscal_year": 2023, "d_and_a": 280.0, "capex": 50.0, "intangible_assets": 2500.0},
        {"fiscal_year": 2024, "d_and_a": 300.0, "capex": 55.0, "intangible_assets": 2400.0},
        {"fiscal_year": 2025, "d_and_a": 310.0, "capex": 60.0, "intangible_assets": 2300.0},
    ]
    revenue_by_year = {2023: 800.0, 2024: 850.0, 2025: 900.0}
    s = compute_series(ticker="COLL", years_data=years, revenue_by_year=revenue_by_year)

    assert len(s.years) == 3
    # Each year's addback is capped at 10% of intangibles ($240M, $230M)
    # For 2024: excess = 245, cap = 240 → addback = 240
    y2024 = next(y for y in s.years if y.fiscal_year == 2024)
    assert y2024.addback == 240.0
    # Average addback on ~$850M avg revenue is >5% → MATERIAL
    assert s.is_materially_distorted is True
    assert s.average_addback > 0
    assert s.average_addback_pct_of_revenue is not None
    assert s.average_addback_pct_of_revenue >= 0.20  # very high distortion


def test_compute_series_modest_distortion_not_material():
    """Small intangible amort below 5% of revenue → NOT flagged as distorted."""
    years = [
        {"fiscal_year": 2023, "d_and_a": 120.0, "capex": 100.0, "intangible_assets": 500.0},
        {"fiscal_year": 2024, "d_and_a": 122.0, "capex": 100.0, "intangible_assets": 480.0},
        {"fiscal_year": 2025, "d_and_a": 125.0, "capex": 100.0, "intangible_assets": 460.0},
    ]
    revenue_by_year = {2023: 5000.0, 2024: 5200.0, 2025: 5500.0}
    s = compute_series(ticker="MODEST", years_data=years, revenue_by_year=revenue_by_year)

    # Per-year add-back ~$20-25M on ~$5.2B revenue = 0.5% — not material
    assert s.is_materially_distorted is False
    assert s.average_addback_pct_of_revenue is not None
    assert s.average_addback_pct_of_revenue < 0.05


def test_compute_series_no_revenue_defaults_to_not_material():
    """Without revenue data we can't assess materiality → default to False."""
    years = [
        {"fiscal_year": 2025, "d_and_a": 100.0, "capex": 10.0, "intangible_assets": 800.0},
    ]
    s = compute_series(ticker="X", years_data=years)
    assert s.is_materially_distorted is False
    assert s.average_addback_pct_of_revenue is None
    # But the addback itself is still computed
    assert s.average_addback > 0


def test_compute_series_no_data_years_gracefully():
    """Years with missing data produce zero add-backs; aggregate stays clean."""
    years = [
        {"fiscal_year": 2023, "d_and_a": None, "capex": None, "intangible_assets": None},
        {"fiscal_year": 2024, "d_and_a": 100.0, "capex": 100.0, "intangible_assets": 0.0},
    ]
    s = compute_series(ticker="EMPTY", years_data=years)
    assert all(y.addback == 0.0 for y in s.years)
    assert s.average_addback == 0.0


# ---------------------------------------------------------------------------
# adjust_operating_income_series — applies the add-back to OI series
# ---------------------------------------------------------------------------


def test_adjust_operating_income_series_applies_per_year():
    addback_series = IntangibleAmortSeries(
        ticker="X",
        years=[
            IntangibleAmortAddBack(
                fiscal_year=2024, d_and_a=300, capex=50, intangible_assets=3000,
                excess_da_over_capex=250, intangible_cap=300, addback=250,
                reason="EXCESS_DA_BELOW_INTANGIBLE_CAP",
            ),
            IntangibleAmortAddBack(
                fiscal_year=2025, d_and_a=310, capex=60, intangible_assets=2800,
                excess_da_over_capex=250, intangible_cap=280, addback=250,
                reason="EXCESS_DA_BELOW_INTANGIBLE_CAP",
            ),
        ],
    )
    oi_series = [(2024, 100.0), (2025, 120.0)]
    adjusted = adjust_operating_income_series(oi_series, addback_series)
    assert adjusted == [(2024, 350.0), (2025, 370.0)]


def test_adjust_operating_income_series_passes_through_missing_years():
    """Years without a matching add-back entry are preserved unchanged."""
    addback_series = IntangibleAmortSeries(
        ticker="X",
        years=[
            IntangibleAmortAddBack(
                fiscal_year=2025, d_and_a=100, capex=10, intangible_assets=500,
                excess_da_over_capex=90, intangible_cap=50, addback=50,
                reason="CAPPED_BY_INTANGIBLE_USEFUL_LIFE",
            ),
        ],
    )
    oi_series = [(2023, 200.0), (2024, 210.0), (2025, 220.0)]
    adjusted = adjust_operating_income_series(oi_series, addback_series)
    # 2023 and 2024 have no addback entry → unchanged; 2025 gets +50
    assert adjusted == [(2023, 200.0), (2024, 210.0), (2025, 270.0)]


# ---------------------------------------------------------------------------
# Real-world scenario: negative-EPV acquirer gets lifted to positive
# ---------------------------------------------------------------------------


def test_coll_style_negative_epv_recovers_after_addback():
    """COLL-style: GAAP OI barely positive; add-back brings it to clearly
    profitable territory. Verifies the chain end-to-end.

    Simulating a company like COLL: $900M revenue, $30M GAAP OI (low),
    heavy $250M/yr intangible amort. After add-back, normalized OI ~$270M.
    """
    years_data = [
        {"fiscal_year": 2023, "d_and_a": 280.0, "capex": 30.0, "intangible_assets": 2500.0},
        {"fiscal_year": 2024, "d_and_a": 290.0, "capex": 35.0, "intangible_assets": 2400.0},
        {"fiscal_year": 2025, "d_and_a": 300.0, "capex": 40.0, "intangible_assets": 2300.0},
    ]
    revenue_by_year = {2023: 800.0, 2024: 850.0, 2025: 900.0}

    # GAAP operating income is suppressed by intangible amort → barely positive
    gaap_oi_series = [(2023, 20.0), (2024, 25.0), (2025, 30.0)]

    # Compute add-back series
    addback = compute_series(
        ticker="COLL", years_data=years_data, revenue_by_year=revenue_by_year,
    )
    assert addback.is_materially_distorted

    adjusted = adjust_operating_income_series(gaap_oi_series, addback)

    # Compute average OI before and after
    gaap_avg = sum(v for _, v in gaap_oi_series) / 3
    adjusted_avg = sum(v for _, v in adjusted) / 3

    # Before: ~$25M avg → NOPAT ~$20M → at 10% WACC, EPV ~$200M
    # After: ~$25M + $240M = $265M avg → NOPAT ~$210M → EPV ~$2.1B
    # That's a 10x increase in EPV — matches the critique's "completely changes
    # the triage signal" prediction.
    assert adjusted_avg > gaap_avg * 5


# ---------------------------------------------------------------------------
# Signatures / schema guard
# ---------------------------------------------------------------------------


def test_result_fields_stay_stable():
    """Snapshot test: IntangibleAmortAddBack has the fields consumers expect."""
    r = compute_single_year_addback(
        fiscal_year=2025, d_and_a=100, capex=50, intangible_assets=500,
    )
    expected_fields = {
        "fiscal_year", "d_and_a", "capex", "intangible_assets",
        "excess_da_over_capex", "intangible_cap", "addback", "reason",
    }
    assert set(r.__dict__.keys()) == expected_fields
