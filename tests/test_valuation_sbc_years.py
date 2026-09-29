"""Exact-literal regressions for known valuation edge cases."""

from app.valuation.sbc_trajectory import compute_sbc_trajectory

def test_sbc_share_changes_need_adjacent_fiscal_years():
    """2020->2022 and 2022->2024 are two-year changes; neither is a YoY pair."""
    result = compute_sbc_trajectory({
        "shares_outstanding": [(2020, 100.0), (2022, 110.0), (2024, 121.0)],
        "share_repurchases_amount": [(2022, 10.0), (2024, 10.0)],
    })
    assert result["shares_yoy_changes"] == []
    assert result["sbc_flags"] == []


def test_sbc_consecutive_dilution_streak_breaks_at_missing_year():
    """2021 and 2025 have observed one-year increases; they are not consecutive years."""
    result = compute_sbc_trajectory({
        "shares_outstanding": [(2020, 100.0), (2021, 110.0), (2024, 120.0), (2025, 132.0)],
        "share_repurchases_amount": [(2021, 10.0), (2025, 10.0)],
    })
    assert result["sbc_flags"] == []

