"""Tests for app.valuation.nonrecurring_revenue — detects one-time licensing spikes."""

from __future__ import annotations

from app.valuation.nonrecurring_revenue import (
    detect,
)


# ---------------------------------------------------------------------------
# Real-world scenarios from the healthcare_pharma_2026-04-14 scan
# ---------------------------------------------------------------------------


def test_kros_takeda_spike_detected():
    """KROS FY2024 $3.5M → FY2025 $244M (~$200M Takeda upfront).

    Both ratio AND absolute tests fire loudly; durable base should step
    back to the pre-spike median, not the ballooned $244M.
    """
    series = [(2021, 20.1), (2023, 0.151), (2024, 3.55), (2025, 244.061)]
    r = detect(ticker="KROS", revenue_series=series)
    assert r.has_suspected_nonrecurring is True
    assert r.spike_year == 2025
    assert r.spike_revenue == 244.061
    assert r.prior_year_revenue == 3.55
    assert r.dollar_delta > 200.0
    # Durable base must NOT be the $244M spike
    assert r.durable_revenue_base is not None
    assert r.durable_revenue_base < 30.0
    assert "TRIGGERED" in r.reason


def test_ptct_novartis_spike_detected():
    """PTCT with a ~$998M collaboration revenue year.

    Even with a moderate prior year (say ~$700M), a $700M jump trips the
    dollar threshold even if the ratio is sub-2x.
    """
    series = [(2022, 650.0), (2023, 720.0), (2024, 740.0), (2025, 1720.0)]
    r = detect(ticker="PTCT", revenue_series=series)
    # $740M → $1720M = 2.3x ratio AND $980M delta → both trigger
    assert r.has_suspected_nonrecurring is True
    assert r.spike_year == 2025
    # Durable base should be the prior year (not thin), not the spike
    assert r.durable_revenue_base == 740.0


# ---------------------------------------------------------------------------
# No-spike cases (regression guard — must NOT flag these)
# ---------------------------------------------------------------------------


def test_steady_growth_not_flagged():
    """15-20% annual growth is normal business — not a spike."""
    series = [(2022, 500.0), (2023, 580.0), (2024, 675.0), (2025, 780.0)]
    r = detect(ticker="STEADY", revenue_series=series)
    assert r.has_suspected_nonrecurring is False
    assert r.durable_revenue_base == 780.0


def test_modest_decline_not_flagged():
    """Revenue declining slightly is not a spike."""
    series = [(2023, 1000.0), (2024, 920.0), (2025, 880.0)]
    r = detect(ticker="DECLINE", revenue_series=series)
    assert r.has_suspected_nonrecurring is False


def test_big_company_modest_growth_not_flagged():
    """A $10B company growing 8% YoY = $800M delta, 1.08x ratio — not a spike."""
    series = [(2024, 10_000.0), (2025, 10_800.0)]
    r = detect(ticker="BIG", revenue_series=series)
    assert r.has_suspected_nonrecurring is False
    # Delta is $800M which EXCEEDS the absolute threshold, BUT ratio is only
    # 1.08x (below 2.0x). Need BOTH to fire for non-thin-prior case.
    assert r.dollar_delta == 800.0
    assert r.spike_ratio is not None
    assert 1.0 < r.spike_ratio < 1.2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_to_positive_is_always_flagged():
    """Zero revenue one year, material revenue the next — always suspect."""
    series = [(2024, 0.0), (2025, 100.0)]
    r = detect(ticker="NEW", revenue_series=series)
    assert r.has_suspected_nonrecurring is True
    assert r.reason == "ZERO_TO_POSITIVE_REVENUE"


def test_insufficient_history():
    series = [(2025, 100.0)]
    r = detect(ticker="X", revenue_series=series)
    assert r.has_suspected_nonrecurring is False
    assert r.reason == "INSUFFICIENT_HISTORY"


def test_empty_series():
    r = detect(ticker="Y", revenue_series=[])
    assert r.has_suspected_nonrecurring is False


def test_thin_prior_year_sensitive_trigger():
    """Thin prior ($30M) + doubling but small absolute (<$50M) → still flags.
    This catches small-cap pharma with a modest licensing event that's
    material relative to their base but not absolute-threshold large.
    """
    series = [(2023, 20.0), (2024, 30.0), (2025, 80.0)]
    r = detect(ticker="TINY", revenue_series=series)
    # 30 → 80 = 2.67x ratio AND $50M delta — both trigger
    assert r.has_suspected_nonrecurring is True


def test_thin_prior_moderate_jump_triggers():
    """Thin prior ($10M) + $30M jump (not enough for $50M absolute trigger,
    and ratio is 4x which passes) — should still fire via the thin-prior path.
    """
    series = [(2023, 5.0), (2024, 10.0), (2025, 40.0)]
    r = detect(ticker="TINY2", revenue_series=series)
    # 10 → 40 = 4x ratio (passes), delta = $30M (below $50M abs threshold)
    # But since prior is thin, the thin-prior gate fires.
    assert r.has_suspected_nonrecurring is True


def test_durable_base_uses_earlier_median_when_prior_is_thin():
    """When prior year is thin (licensing-dependent), durable base should
    step back to the median of ALL positive earlier years (including the
    thin one), which gives a more conservative baseline."""
    series = [(2021, 15.0), (2022, 18.0), (2023, 25.0), (2024, 3.0), (2025, 200.0)]
    r = detect(ticker="KROSLIKE", revenue_series=series)
    assert r.has_suspected_nonrecurring is True
    # Earlier positive revenues are [3, 15, 18, 25]; sorted median of 4 values
    # = (15 + 18) / 2 = 16.5. Deliberately conservative — including the thin
    # prior pulls the durable base down, guarding against DCF inflation.
    assert r.durable_revenue_base == 16.5


# ---------------------------------------------------------------------------
# Schema guard
# ---------------------------------------------------------------------------


def test_detection_fields_complete():
    r = detect(ticker="X", revenue_series=[(2024, 100.0), (2025, 500.0)])
    expected = {
        "ticker", "has_suspected_nonrecurring", "spike_year",
        "spike_revenue", "prior_year_revenue", "spike_ratio",
        "dollar_delta", "durable_revenue_base", "reason",
    }
    assert set(r.__dict__.keys()) == expected
