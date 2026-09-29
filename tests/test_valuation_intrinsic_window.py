"""Recent means recent fiscal rows, not the last profitable observations."""

import pytest

from app.valuation.intrinsic_discipline import _median_positive


@pytest.mark.parametrize("latest", [[-10.0, -20.0, -30.0], ["UNKNOWN"] * 3])
def test_intrinsic_positive_median_does_not_reach_past_recent_window(latest):
    rows = [
        {"year": year, "value": value, "derived_from": [f"year:{year}"]}
        for year, value in zip(range(2019, 2025), [30.0, 40.0, 50.0, *latest], strict=True)
    ]
    value, refs, points = _median_positive(rows, window=3)
    assert value == "UNKNOWN"
    assert points == 0
    assert refs == ["year:2022", "year:2023", "year:2024"]


def test_intrinsic_median_runs_over_every_recent_observation_losses_included():
    """Migrated 2026-09-29: the old answer (20.0 from the two profitable years)
    dropped the 2023 loss before taking the median. The median of 10, -20, 30 is 10."""
    rows = [
        {"year": year, "value": value}
        for year, value in [(2021, 1000.0), (2022, 10.0), (2023, -20.0), (2024, 30.0)]
    ]
    assert _median_positive(rows, window=3) == (10.0, [], 3)


def test_intrinsic_median_needs_two_profitable_years_in_the_window():
    """[-500, -400, 100] is two losses and one profit: no normalized level, not 100."""
    rows = [
        {"year": year, "value": value, "derived_from": [f"year:{year}"]}
        for year, value in [(2022, -500.0), (2023, -400.0), (2024, 100.0)]
    ]
    assert _median_positive(rows, window=3) == (
        "UNKNOWN", ["year:2022", "year:2023", "year:2024"], 0
    )
    lone = [{"year": 9999, "value": 100.0, "derived_from": []}]
    assert _median_positive(lone, window=3) == ("UNKNOWN", [], 0)
