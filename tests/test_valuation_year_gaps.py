"""Fiscal-year gaps must not be read as adjacent years.

Residual cases that no other test covered. Each docstring derives the correct answer by hand; none is copied from
implementation output.
"""

from __future__ import annotations

from app.valuation import intrinsic_discipline as intrinsic
from app.valuation.nonrecurring_filter import detect_nonrecurring_items
from app.valuation.sbc_trajectory import compute_sbc_trajectory


# ── SBC: buyback credit must come from the dilution streak itself ─────────────


def test_a_buyback_in_a_disconnected_rising_year_does_not_credit_a_later_streak():
    """Shares 100 -> 110 (2020 -> 2021), then 120 -> 132 -> 145.2 (2023 -> 2025).

    2021 is a rising year but stands alone (2022 is missing, so 2021 -> 2023 is not
    a year-over-year change). The only two-year dilution streak is 2024-2025, and
    no shares were repurchased in either year; the only repurchase is in 2021.
    "Net dilution despite buybacks" needs the buybacks inside the streak, so the
    flag must not fire.
    """
    result = compute_sbc_trajectory(
        {
            "shares_outstanding": [
                (2020, 100.0),
                (2021, 110.0),
                (2023, 120.0),
                (2024, 132.0),
                (2025, 145.2),
            ],
            "share_repurchases_amount": [(2021, 50.0)],
        }
    )
    assert "NET_DILUTION_DESPITE_BUYBACKS" not in result["sbc_flags"]


def test_a_buyback_inside_the_dilution_streak_still_fires():
    """Control: the same streak with a 2025 repurchase is net dilution despite buybacks."""
    result = compute_sbc_trajectory(
        {
            "shares_outstanding": [(2023, 120.0), (2024, 132.0), (2025, 145.2)],
            "share_repurchases_amount": [(2025, 50.0)],
        }
    )
    assert "NET_DILUTION_DESPITE_BUYBACKS" in result["sbc_flags"]


# ── Nonrecurring: a goodwill fall is a one-year event only between adjacent years ─


def test_a_goodwill_fall_across_missing_years_is_not_a_one_year_impairment():
    """Goodwill 100 at FY2021, 80 at FY2024, nothing filed for 2022-2023.

    The 20% fall happened somewhere across three years (a disposal, currency, an
    impairment, or several). Calling FY2024 the impairment year has no support, so
    no GOODWILL_IMPAIRMENT_LIKELY flag and no impairment year.
    """
    result = detect_nonrecurring_items({"goodwill": [(2021, 100.0), (2024, 80.0)]})
    assert result["goodwill_impairment_years"] == []
    assert "GOODWILL_IMPAIRMENT_LIKELY" not in result["nonrecurring_flags"]


def test_an_adjacent_year_goodwill_fall_still_flags():
    """Control: 100 at FY2023 to 80 at FY2024 is a >10% one-year fall."""
    result = detect_nonrecurring_items({"goodwill": [(2023, 100.0), (2024, 80.0)]})
    assert result["goodwill_impairment_years"] == [2024]


# ── Intrinsic: "last three years" means three fiscal years, not three rows ──────


def test_intrinsic_three_year_window_does_not_reach_across_missing_years():
    """Owner-earnings rows at FY2018 (50), FY2021 (60) and FY2024 (-10) only.

    The latest fiscal year is 2024, so the three-year window is 2022-2024. The only
    observation in it is the 2024 loss; the 2018 and 2021 profits are outside it.
    With no positive observation in the window the normalized value is UNKNOWN,
    not the median 55 of two old profitable years.
    """
    rows = [
        {"year": 2018, "value": 50.0, "derived_from": ["oe:2018"]},
        {"year": 2021, "value": 60.0, "derived_from": ["oe:2021"]},
        {"year": 2024, "value": -10.0, "derived_from": ["oe:2024"]},
    ]
    value, refs, points = intrinsic._median_positive(rows, window=3)
    assert value == "UNKNOWN"
    assert points == 0
    assert refs == ["oe:2024"]


def test_intrinsic_three_year_window_keeps_contiguous_recent_years():
    """Control: FY2022-2024 of 30, 40, 50 give the median 40 from three points."""
    rows = [
        {"year": y, "value": v, "derived_from": []}
        for y, v in [(2022, 30.0), (2023, 40.0), (2024, 50.0)]
    ]
    assert intrinsic._median_positive(rows, window=3) == (40.0, [], 3)


def test_an_undated_owner_summary_cannot_override_dated_losses():
    """The summary says 40 but carries neither a series year nor an as-of date.

    FY2022-2024 filed cash flows are -10 each. A positive summary with no date proof
    cannot be shown to cover those years, so it must not replace them: normalized
    earnings power is UNKNOWN with the negative-earnings reason, not 40.
    """
    undated = {
        "summary": {
            "owner_earnings_normalized_3y": 40.0,
            "owner_earnings_normalized_method": "MEDIAN_3Y",
            "owner_earnings_points": 3,
        },
        "derived_from": ["owner-summary"],
    }
    rows = [{"year": y, "cfo": -10.0, "capex": 0.0, "fcf": -10.0} for y in (2022, 2023, 2024)]
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE",
        "2025-03-01",
        fundamentals={"rows": rows},
        owner_payload=undated,
        price_value=10.0,
        shares_value=10.0,
        net_debt_value=0.0,
    )
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["NEGATIVE_NORMALIZED_EARNINGS"]
    assert result["intrinsic_base"] == "UNKNOWN"
