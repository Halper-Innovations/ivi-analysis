"""Exact-literal regressions for the SBC burden calculations."""

import pytest

from app.valuation.sbc_trajectory import compute_sbc_trajectory

@pytest.mark.parametrize("years", [[2024], [2023, 2024]])
def test_sbc_extreme_burden_needs_one_known_ratio(years):
    """10 / 100 = 10%, exceeding the existing 8% extreme threshold in each year."""
    result = compute_sbc_trajectory({
        "sbc": [(year, 10.0) for year in years],
        "revenue": [(year, 100.0) for year in years],
    })
    assert result["latest_sbc_revenue_ratio"] == 0.1
    assert result["sbc_flags"] == ["SBC_BURDEN_EXTREME"]

