"""Exact-literal regressions for the valuation math."""

import pytest

from app.valuation.cyclical_normalization import compute_cyclical_normalization

def test_known_cyclical_mixed_years_are_not_all_negative():
    """Two positive years and one loss are not an all-negative series; 2 < 3 positives."""
    result = compute_cyclical_normalization(
        "FIXTURE", "2025-03-01",
        owner_earnings_series=[
            {"year": 2022, "value": 100.0},
            {"year": 2023, "value": 100.0},
            {"year": 2024, "value": -1.0},
        ],
    )
    assert result["cyclical_profile_class"] == "CYCLICALITY_UNKNOWN"
    assert "SERIES_ALL_NEGATIVE" not in result["cyclical_normalization_reason_codes"]


@pytest.mark.parametrize("values", [[0.0, 0.0, 0.0], [-1.0, 0.0, -2.0]])
def test_zero_observations_do_not_prove_an_all_negative_series(values):
    result = compute_cyclical_normalization(
        "FIXTURE", "2025-03-01",
        owner_earnings_series=[{"year": 2022 + i, "value": value} for i, value in enumerate(values)],
    )
    assert result["cyclical_profile_class"] == "CYCLICALITY_UNKNOWN"
    assert "SERIES_ALL_NEGATIVE" not in result["cyclical_normalization_reason_codes"]


def test_three_negative_observations_keep_the_existing_negative_shortcut():
    result = compute_cyclical_normalization(
        "FIXTURE", "2025-03-01",
        owner_earnings_series=[{"year": 2022 + i, "value": value}
                               for i, value in enumerate([-1.0, -2.0, -3.0])],
    )
    assert result["cyclical_profile_class"] == "CLEARLY_CYCLICAL"
    assert "SERIES_ALL_NEGATIVE" in result["cyclical_normalization_reason_codes"]
