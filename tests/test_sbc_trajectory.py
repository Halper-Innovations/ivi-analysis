"""Tests for app.valuation.sbc_trajectory."""

from __future__ import annotations


def test_sbc_accelerating():
    """SBC/revenue ratio increased >1 pct point over 3yr → SBC_ACCELERATING."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory
    facts = {
        "sbc": [(2023, 30.0), (2024, 40.0), (2025, 55.0)],
        "revenue": [(2023, 1000.0), (2024, 1020.0), (2025, 1050.0)],
        "shares_outstanding": [(2023, 100.0), (2024, 100.0), (2025, 100.0)],
    }
    result = compute_sbc_trajectory(facts)
    # Ratio: 3.0% → 3.9% → 5.2% — increase of 2.2 pct points
    assert "SBC_ACCELERATING" in result["sbc_flags"]
    assert result["sbc_revenue_trend"] == "INCREASING"


def test_net_dilution_despite_buybacks():
    """Shares rising for 2+ years despite buybacks → NET_DILUTION_DESPITE_BUYBACKS."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory
    facts = {
        "sbc": [(2023, 20.0), (2024, 22.0), (2025, 25.0)],
        "revenue": [(2023, 1000.0), (2024, 1050.0), (2025, 1100.0)],
        "shares_outstanding": [(2023, 100.0), (2024, 101.0), (2025, 102.0)],
        "share_repurchases_amount": [(2023, 10.0), (2024, 12.0), (2025, 15.0)],
    }
    result = compute_sbc_trajectory(facts)
    assert "NET_DILUTION_DESPITE_BUYBACKS" in result["sbc_flags"]


def test_sbc_improving():
    """SBC/revenue ratio declining over 3yr → SBC_IMPROVING."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory
    facts = {
        "sbc": [(2023, 60.0), (2024, 45.0), (2025, 30.0)],
        "revenue": [(2023, 1000.0), (2024, 1050.0), (2025, 1100.0)],
        "shares_outstanding": [(2023, 100.0), (2024, 98.0), (2025, 96.0)],
    }
    result = compute_sbc_trajectory(facts)
    # Ratio: 6.0% → 4.3% → 2.7% — declining
    assert "SBC_IMPROVING" in result["sbc_flags"]
    assert result["sbc_revenue_trend"] == "DECREASING"
    assert "SBC_ACCELERATING" not in result["sbc_flags"]


def test_clean_company_no_flags():
    """Stable low SBC, shares declining → no flags."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory
    facts = {
        "sbc": [(2023, 20.0), (2024, 21.0), (2025, 22.0)],
        "revenue": [(2023, 1000.0), (2024, 1050.0), (2025, 1100.0)],
        "shares_outstanding": [(2023, 100.0), (2024, 98.0), (2025, 96.0)],
    }
    result = compute_sbc_trajectory(facts)
    # Ratio: 2.0% → 2.0% → 2.0% — stable, no acceleration
    assert "SBC_ACCELERATING" not in result["sbc_flags"]
    assert "NET_DILUTION_DESPITE_BUYBACKS" not in result["sbc_flags"]
    assert result["sbc_revenue_trend"] == "STABLE"


def test_missing_data_graceful():
    """Empty facts → no crash, empty result."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory
    result = compute_sbc_trajectory({})
    assert result["sbc_flags"] == []
    assert result["sbc_revenue_ratios"] == []
    assert result["sbc_revenue_trend"] == "UNKNOWN"
    assert result["latest_sbc_revenue_ratio"] is None


def test_old_high_point_does_not_mask_a_recent_ramp():
    """The old rule was latest minus oldest: a 9% ratio in 2005 made a 2.0% -> 3.5% ramp
    over the last three years read DECREASING/SBC_IMPROVING. The window is the recent
    five fiscal years, so 2005 is out and the ramp reads INCREASING."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory
    facts = {
        "sbc": [(2005, 90.0), (2023, 20.0), (2024, 27.5), (2025, 35.0)],
        "revenue": [(2005, 1000.0), (2023, 1000.0), (2024, 1000.0), (2025, 1000.0)],
        "shares_outstanding": [(2023, 100.0), (2024, 100.0), (2025, 100.0)],
    }
    result = compute_sbc_trajectory(facts)
    assert result["sbc_revenue_trend"] == "INCREASING"
    assert result["sbc_flags"] == ["SBC_ACCELERATING"]
    assert result["sbc_revenue_trend_window"] == [2023, 2025]
    assert result["sbc_revenue_trend_change"] == 0.015


def test_trend_is_a_fit_over_the_window_not_two_endpoints():
    """Five years at 3.0%, 3.0%, 3.0%, 3.0%, 4.2%: a single last-year jump of 1.2 pts is
    0.72 pts on the fitted line across the window, below the 1 pt threshold."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory
    ratios = [0.030, 0.030, 0.030, 0.030, 0.042]
    facts = {
        "sbc": [(2021 + i, r * 1000.0) for i, r in enumerate(ratios)],
        "revenue": [(2021 + i, 1000.0) for i in range(5)],
    }
    result = compute_sbc_trajectory(facts)
    assert result["sbc_revenue_trend"] == "STABLE"
    assert result["sbc_flags"] == []
    assert result["sbc_revenue_trend_change"] == 0.0096


def test_fewer_than_three_recent_ratio_years_leaves_the_trend_unknown():
    """Ratios in 2025, 2024 and a decade-old 2015: only two inside the window, so no trend
    is claimed; the current level is still reported and still flags a heavy burden."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory
    facts = {
        "sbc": [(2025, 90.0), (2024, 50.0), (2015, 10.0)],
        "revenue": [(2025, 1000.0), (2024, 1000.0), (2015, 1000.0)],
    }
    result = compute_sbc_trajectory(facts)
    assert result["sbc_revenue_trend"] == "UNKNOWN"
    assert result["sbc_revenue_trend_change"] is None
    assert result["sbc_revenue_trend_window"] == []
    assert result["latest_sbc_revenue_ratio"] == 0.09
    assert result["sbc_flags"] == ["SBC_BURDEN_EXTREME"]


def test_a_filed_split_year_is_not_counted_as_dilution():
    """Review M5, 2026-09-29. A 2-for-1 split in 2024 read as +100% share
    growth, and with buybacks in 2024 and 2025 the two rising years made a
    NET_DILUTION_DESPITE_BUYBACKS streak. With the split filed, 2024 is
    measured on the split-adjusted count (100 -> 200 is 0%), no streak."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory

    facts = {
        "shares_outstanding": [(2023, 100.0), (2024, 200.0), (2025, 202.0)],
        "share_repurchases_amount": [(2024, 10.0), (2025, 10.0)],
    }
    split = [{"year": 2024, "value": 2.0, "derived_from": ["split.2024"]}]
    result = compute_sbc_trajectory(facts, split_rows=split)

    assert result["shares_yoy_changes"] == [
        {"year": 2024, "shares": 200.0, "split_factor": 2.0, "change_pct": 0.0},
        {"year": 2025, "shares": 202.0, "change_pct": 0.01},
    ]
    assert result["shares_count_breaks"] == []
    assert "NET_DILUTION_DESPITE_BUYBACKS" not in result["sbc_flags"]


def test_an_unfiled_break_year_is_flagged_not_counted():
    """Review M5, 2026-09-29. The same doubling with no filed split may be a split
    or an equity raise; it is neither counted as ordinary dilution nor dropped
    silently: the year leaves the changes and is named in shares_count_breaks
    with the SHARE_COUNT_BREAK_UNCORROBORATED flag."""
    from app.valuation.sbc_trajectory import compute_sbc_trajectory

    facts = {
        "shares_outstanding": [(2023, 100.0), (2024, 200.0), (2025, 202.0)],
        "share_repurchases_amount": [(2024, 10.0), (2025, 10.0)],
    }
    result = compute_sbc_trajectory(facts)

    assert result["shares_yoy_changes"] == [{"year": 2025, "shares": 202.0, "change_pct": 0.01}]
    assert result["shares_count_breaks"] == [2024]
    assert "SHARE_COUNT_BREAK_UNCORROBORATED" in result["sbc_flags"]
    assert "NET_DILUTION_DESPITE_BUYBACKS" not in result["sbc_flags"]
