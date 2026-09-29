"""Math defects in compute_intrinsic_range_from_multiples, pinned to exact literals.

Every test here fails on the pre-2026-08-29 implementation, which indexed the caller's
multiple history by list position, pooled FCF- and EBIT-derived enterprise values into a
single sorted list, multiplied negative drivers by positive multiples, accepted
non-positive multiples, and used enterprise value as an equity proxy when net debt was
unavailable.
"""

from __future__ import annotations

from app.fundamentals.normalize import UNKNOWN
from app.valuation.multiples import compute_intrinsic_range_from_multiples

UNKNOWN_TRIPLE = {"low": UNKNOWN, "base": UNKNOWN, "high": UNKNOWN}


def test_band_comes_from_order_statistics_not_list_position() -> None:
    """A chronological history is a sample, not an ordered (low, base, high) band.

    History below is oldest-first: the company traded at 20x and 18x years ago and at 6x,
    9x and 12x more recently. The old code read band[0]=20x as the low, band[1]=18x as the
    mid and band[-1]=12x as the high, then sorted the three products, reporting
    1200/1800/2000 -- a band whose *low* sat above the 25th percentile of the real history
    and which never saw the 6x and 9x years at all. The band is now the 25th percentile,
    median and 75th percentile of the sample: 9x, 12x, 18x.
    """
    outputs, warnings = compute_intrinsic_range_from_multiples(
        fcf=100.0,
        ebit_proxy=None,
        historical_ev_fcf=[20.0, 18.0, 6.0, 9.0, 12.0],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )

    assert outputs["ev_range"] == {"low": 900.0, "base": 1200.0, "high": 1800.0}
    assert outputs["equity_value_range"] == {"low": 900.0, "base": 1200.0, "high": 1800.0}
    assert outputs["per_share_range"] == {"low": 90.0, "base": 120.0, "high": 180.0}
    assert outputs["confidence"] == "MEDIUM"
    assert warnings == []

    # Order of the same observations cannot change the answer.
    shuffled, _ = compute_intrinsic_range_from_multiples(
        fcf=100.0,
        ebit_proxy=None,
        historical_ev_fcf=[6.0, 12.0, 20.0, 9.0, 18.0],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )
    assert shuffled["ev_range"] == outputs["ev_range"]


def test_methods_blend_statistic_by_statistic_and_base_is_a_real_median() -> None:
    """Two methods are blended low-with-low, not pooled into one sorted list.

    FCF says a flat 10x, EBIT says a flat 30x, on the same 100 of driver. The old code
    appended all six enterprise values to one list and sorted it, reporting low=1000 (an
    FCF number) and high=3000 (an EBIT number) -- a "range" that was really the two models
    disagreeing threefold, and a base of 3000 because ev_values[6 // 2] takes the
    upper-middle element rather than the median. Each method now yields its own band and
    the bands are averaged statistic by statistic, with the disagreement said out loud.
    """
    outputs, warnings = compute_intrinsic_range_from_multiples(
        fcf=100.0,
        ebit_proxy=100.0,
        historical_ev_fcf=[10.0, 10.0, 10.0],
        historical_ev_ebit=[30.0, 30.0, 30.0],
        net_debt=0.0,
        shares_outstanding=10.0,
    )

    assert outputs["ev_range"] == {"low": 2000.0, "base": 2000.0, "high": 2000.0}
    assert outputs["equity_value_range"] == {"low": 2000.0, "base": 2000.0, "high": 2000.0}
    assert outputs["per_share_range"] == {"low": 200.0, "base": 200.0, "high": 200.0}
    assert warnings == [
        "FCF and EBIT multiples disagree by 3.0x (base EV 1,000 vs 3,000); "
        "blended band is uncertain"
    ]

    # The base is a real median of the history. For [8, 10, 14, 20] that is 12.0; the old
    # code took band[1] by position -- 10x, the second year in the series -- and valued the
    # business at 1000. (Its ev_values[len // 2] upper-middle bug is the one the two-method
    # case above pins: six pooled values made the base 3000 when the median was 2000.)
    even, even_warnings = compute_intrinsic_range_from_multiples(
        fcf=100.0,
        ebit_proxy=None,
        historical_ev_fcf=[8.0, 10.0, 14.0, 20.0],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )
    assert even["ev_range"] == {"low": 950.0, "base": 1200.0, "high": 1550.0}
    assert even_warnings == []


def test_negative_driver_refuses_instead_of_reporting_negative_value() -> None:
    """A negative FCF times a positive multiple is not an intrinsic value.

    The old code multiplied -50 by the default 8x/12x/16x band and reported a confident
    equity range of -800/-600/-400 and a per-share range of -80/-60/-40, with the ordering
    inverted (the highest multiple gave the lowest value). The method now refuses, the way
    dcf_lite and reverse_dcf refuse when their inputs cannot support the model.
    """
    outputs, warnings = compute_intrinsic_range_from_multiples(
        fcf=-50.0,
        ebit_proxy=None,
        historical_ev_fcf=[],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )

    assert outputs["ev_range"] == UNKNOWN_TRIPLE
    assert outputs["equity_value_range"] == UNKNOWN_TRIPLE
    assert outputs["per_share_range"] == UNKNOWN_TRIPLE
    assert outputs["confidence"] == "LOW"
    assert warnings == [
        "FCF is non-positive (-50.00); EV/FCF multiples valuation is not meaningful",
        "No computable multiples method; intrinsic range UNKNOWN",
    ]

    # A zero driver is refused for the same reason: it values the business at zero
    # whatever the band says.
    zero, zero_warnings = compute_intrinsic_range_from_multiples(
        fcf=0.0,
        ebit_proxy=None,
        historical_ev_fcf=[10.0, 14.0],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )
    assert zero["per_share_range"] == UNKNOWN_TRIPLE
    assert zero_warnings == [
        "FCF is non-positive (0.00); EV/FCF multiples valuation is not meaningful",
        "No computable multiples method; intrinsic range UNKNOWN",
    ]

    # One dead method does not kill a live one: EBIT still values the business.
    mixed, mixed_warnings = compute_intrinsic_range_from_multiples(
        fcf=-50.0,
        ebit_proxy=100.0,
        historical_ev_fcf=[],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )
    assert mixed["ev_range"] == {"low": 700.0, "base": 1000.0, "high": 1400.0}
    assert mixed_warnings == [
        "FCF is non-positive (-50.00); EV/FCF multiples valuation is not meaningful",
        "Using wide default EV/EBIT band due to insufficient history",
    ]


def test_non_positive_historical_multiples_are_dropped() -> None:
    """A loss year prints a negative or zero EV/FCF multiple; it is not a valuation input.

    The old code passed [-5, 0, 12, 16] straight through: band[0] = -5x produced an
    enterprise value of -500, band[1] = 0x produced 0, and after sorting the reported range
    was -500/0/1600. Only the two positive observations survive now, giving 13x/14x/15x.
    """
    outputs, warnings = compute_intrinsic_range_from_multiples(
        fcf=100.0,
        ebit_proxy=None,
        historical_ev_fcf=[-5.0, 0.0, 12.0, 16.0],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )

    assert outputs["ev_range"] == {"low": 1300.0, "base": 1400.0, "high": 1500.0}
    assert outputs["per_share_range"] == {"low": 130.0, "base": 140.0, "high": 150.0}
    assert warnings == ["Dropped 2 non-positive EV/FCF multiple(s) from history"]

    # Filtering down to fewer than two observations is insufficient history, not a
    # zero-width band: fall back to the wide default 8x/12x/16x and say so.
    thin, thin_warnings = compute_intrinsic_range_from_multiples(
        fcf=100.0,
        ebit_proxy=None,
        historical_ev_fcf=[-5.0, 0.0, 12.0],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )
    assert thin["ev_range"] == {"low": 800.0, "base": 1200.0, "high": 1600.0}
    assert thin["confidence"] == "LOW"
    assert thin_warnings == [
        "Dropped 2 non-positive EV/FCF multiple(s) from history",
        "Using wide default EV/FCF band due to insufficient history",
    ]


def test_unknown_net_debt_refuses_equity_and_per_share() -> None:
    """Enterprise value is not an equity proxy: net debt is exactly the difference.

    The old code warned and then handed back the enterprise value as the equity value,
    reporting 800/1200/1600 of equity and 80/120/160 per share for a company whose net debt
    was unknown -- overstating equity by the whole of net debt for any levered issuer. The
    enterprise band is still published (it is honest), but equity and per-share are now
    UNKNOWN, matching dcf_lite's refusal.
    """
    outputs, warnings = compute_intrinsic_range_from_multiples(
        fcf=100.0,
        ebit_proxy=None,
        historical_ev_fcf=[],
        historical_ev_ebit=[],
        net_debt=None,
        shares_outstanding=10.0,
    )

    assert outputs["ev_range"] == {"low": 800.0, "base": 1200.0, "high": 1600.0}
    assert outputs["equity_value_range"] == UNKNOWN_TRIPLE
    assert outputs["per_share_range"] == UNKNOWN_TRIPLE
    assert outputs["confidence"] == "LOW"
    assert warnings == [
        "Using wide default EV/FCF band due to insufficient history",
        "Net debt UNKNOWN for equity-value multiples",
    ]

    # Known net debt still produces equity and per-share numbers.
    known, known_warnings = compute_intrinsic_range_from_multiples(
        fcf=100.0,
        ebit_proxy=None,
        historical_ev_fcf=[],
        historical_ev_ebit=[],
        net_debt=200.0,
        shares_outstanding=10.0,
    )
    assert known["equity_value_range"] == {"low": 600.0, "base": 1000.0, "high": 1400.0}
    assert known["per_share_range"] == {"low": 60.0, "base": 100.0, "high": 140.0}
    assert known_warnings == ["Using wide default EV/FCF band due to insufficient history"]


def test_missing_both_drivers_keeps_its_original_refusal() -> None:
    """Unchanged contract: no FCF and no EBIT is still the same single warning."""
    outputs, warnings = compute_intrinsic_range_from_multiples(
        fcf=None,
        ebit_proxy=None,
        historical_ev_fcf=[],
        historical_ev_ebit=[],
        net_debt=0.0,
        shares_outstanding=10.0,
    )

    assert outputs["equity_value_range"] == UNKNOWN_TRIPLE
    assert outputs["per_share_range"] == UNKNOWN_TRIPLE
    assert outputs["confidence"] == "LOW"
    assert warnings == ["Insufficient FCF/EBIT inputs for multiples valuation"]
