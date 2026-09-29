"""Arithmetic pins for app/valuation/dcf_lite.py.

One test per defect fixed on 2026-08-29. Every number below is an exact literal
produced by the fixed module; the "before" value recorded in each test is what the
old code returned (or the exception it raised), so each test fails on the old code.
"""

import pytest

from app.fundamentals.normalize import UNKNOWN
from app.valuation.dcf_lite import DCFInputs, run_dcf_lite

TERMINAL_REINVEST_WARNING_35 = (
    "Terminal reinvestment held at the explicit 35.0%, which implies a steady-state "
    "return on capital of 5.7%, below the 10.0% discount rate, so terminal value is "
    "understated"
)
TERMINAL_REINVEST_WARNING_30 = (
    "Terminal reinvestment held at the explicit 30.0%, which implies a steady-state "
    "return on capital of 6.7%, below the 10.0% discount rate, so terminal value is "
    "understated"
)


def test_invalid_discount_versus_terminal_growth_short_circuits():
    """disc <= terminal_growth must refuse to value, not drop the terminal value.

    Before: the guard returned (finite explicit-period PV, inf, inf), so the caller's
    `base_ev == inf` test was False and the run produced ev_range base
    405.48338392654614 and per-share base 3.5548338392654615 at confidence MEDIUM,
    with no "Invalid discount/terminal growth assumptions" warning anywhere.
    """
    outputs, warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.25,
            reinvestment_rate=0.30,
            discount_rate=0.02,
            terminal_growth=0.05,
            shares_outstanding=100.0,
            net_debt=50.0,
        )
    )

    assert warnings == ["Invalid discount/terminal growth assumptions"]
    assert outputs == {
        "ev_range": {"low": UNKNOWN, "base": UNKNOWN, "high": UNKNOWN},
        "equity_value_range": {"low": UNKNOWN, "base": UNKNOWN, "high": UNKNOWN},
        "per_share_range": {"low": UNKNOWN, "base": UNKNOWN, "high": UNKNOWN},
        "confidence": "LOW",
        "terminal_value_share": UNKNOWN,
        "implied_terminal_ebit_multiple": UNKNOWN,
    }


def test_valid_spread_still_values_normally():
    """The short-circuit must not fire one basis point early: disc > g still values."""
    outputs, warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.25,
            reinvestment_rate=0.30,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=50.0,
        )
    )

    assert outputs["ev_range"]["base"] == 1047.4031543621081
    assert outputs["per_share_range"]["base"] == 9.97403154362108
    assert outputs["confidence"] == "MEDIUM"
    assert "Invalid discount/terminal growth assumptions" not in warnings


def test_negative_margin_band_is_not_inverted():
    """A negative base margin must still give low <= base <= high.

    Before: margin_low = max(0.0, -0.10 * 0.8) = 0.0 and margin_high = -0.115, so the
    run returned ev_range low 0.0, base -997.526813678198, high -1300.1906370124989 —
    the "optimistic" case was the deepest loss and the "pessimistic" case was breakeven.
    """
    outputs, warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=-0.10,
            tax_rate=0.25,
            reinvestment_rate=0.35,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=0.0,
        )
    )

    assert outputs["ev_range"] == {
        "low": -1053.6780274571406,
        "base": -997.526813678198,
        "high": -961.0104708353255,
    }
    assert outputs["ev_range"]["low"] < outputs["ev_range"]["base"] < outputs["ev_range"]["high"]
    assert outputs["per_share_range"] == {
        "low": -10.536780274571406,
        "base": -9.97526813678198,
        "high": -9.610104708353255,
    }
    assert warnings == [
        TERMINAL_REINVEST_WARNING_35,
        "Operating margin negative; pessimistic case widens the loss instead of "
        "turning it positive",
    ]


def test_positive_margin_band_is_unchanged():
    """The negative-margin branch must not move the ordinary positive-margin band.

    These three numbers are what the old module returned for this input, byte for
    byte: the band fix is confined to margins below zero.
    """
    outputs, _ = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.25,
            reinvestment_rate=0.30,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=50.0,
        )
    )

    assert outputs["ev_range"] == {
        "low": 737.5746192199983,
        "base": 1047.4031543621081,
        "high": 1365.200168863124,
    }


def test_zero_operating_margin_does_not_raise_zero_division():
    """A 0.0 base margin makes base_ev exactly 0.0.

    Before: `base_term / base_ev > 0.75` ran before any zero check and raised
    ZeroDivisionError, even though the outputs dict below it already guarded with
    `if base_ev else UNKNOWN`.
    """
    outputs, warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.0,
            tax_rate=0.25,
            reinvestment_rate=0.35,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=0.0,
        )
    )

    assert outputs["ev_range"] == {"low": 0.0, "base": 0.0, "high": 0.0}
    assert outputs["equity_value_range"] == {"low": 0.0, "base": 0.0, "high": 0.0}
    assert outputs["per_share_range"] == {"low": 0.0, "base": 0.0, "high": 0.0}
    assert outputs["terminal_value_share"] == UNKNOWN
    assert outputs["confidence"] == "MEDIUM"
    assert warnings == [
        TERMINAL_REINVEST_WARNING_35,
        "Base enterprise value is zero; terminal-value share unavailable",
    ]
    assert "DCF dominated by terminal value (>75%)" not in warnings


def test_tax_rate_outside_defensible_range_is_clamped_and_warned():
    """An effective rate above 1.0 or below 0.0 must not flip NOPAT's sign.

    Before: tax_rate=1.2 gave ev_range base -399.01072547127916 and per-share base
    -4.490107254712791 at confidence MEDIUM with an empty warnings list; tax_rate=-0.30
    gave ev_range base 1815.4988008943205, also unwarned.
    """
    over, over_warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=1.2,
            reinvestment_rate=0.30,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=50.0,
        )
    )

    assert over["ev_range"]["base"] == 698.2687695747387
    assert over["per_share_range"]["base"] == 6.482687695747387
    assert over_warnings == [
        "Tax rate 120.0% outside defensible range [0.0%, 50.0%]; clamped to 50.0%",
        TERMINAL_REINVEST_WARNING_30,
    ]

    under, under_warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=-0.30,
            reinvestment_rate=0.30,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=50.0,
        )
    )

    assert under["ev_range"]["base"] == 1396.5375391494774
    assert under_warnings == [
        "Tax rate -30.0% outside defensible range [0.0%, 50.0%]; clamped to 0.0%",
        TERMINAL_REINVEST_WARNING_30,
    ]

    inside, inside_warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.50,
            reinvestment_rate=0.30,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=50.0,
        )
    )

    assert inside["ev_range"]["base"] == over["ev_range"]["base"]
    assert inside_warnings == [TERMINAL_REINVEST_WARNING_30]


def test_terminal_reinvestment_roic_is_exposed_not_silently_changed():
    """Default terminal reinvestment is unchanged; the ROIC it implies is now visible.

    Before: the terminal FCF reused the explicit reinvestment rate with no way to see
    or override it — reinvesting 35% forever to fund 3% growth asserts an 8.6% return
    on capital — and the outputs carried neither key, while passing steady_state_roic
    raised TypeError.
    """
    default_out, default_warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.25,
            reinvestment_rate=0.35,
            discount_rate=0.10,
            terminal_growth=0.03,
            shares_outstanding=100.0,
            net_debt=0.0,
        )
    )

    # Default behaviour is byte-for-byte what the old module produced.
    assert default_out["ev_range"]["base"] == 1075.9821428571427
    assert default_out["terminal_reinvestment_rate"] == 0.35
    assert default_out["implied_steady_state_roic"] == 0.08571428571428572
    assert default_warnings == [
        "Terminal reinvestment held at the explicit 35.0%, which implies a steady-state "
        "return on capital of 8.6%, below the 10.0% discount rate, so terminal value is "
        "understated"
    ]

    steady_out, steady_warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.25,
            reinvestment_rate=0.35,
            discount_rate=0.10,
            terminal_growth=0.03,
            shares_outstanding=100.0,
            net_debt=0.0,
        ),
        steady_state_roic=0.15,
    )

    assert steady_out["terminal_reinvestment_rate"] == 0.2
    assert steady_out["implied_steady_state_roic"] == 0.15
    assert steady_out["ev_range"]["base"] == 1254.7155210499145
    assert steady_out["per_share_range"]["base"] == 12.547155210499145
    assert steady_warnings[0] == (
        "Terminal reinvestment set to 20.0% from steady-state return on capital 15.0%; "
        "explicit period reinvests 35.0%"
    )


@pytest.mark.parametrize("bad_roic", [0.0, -0.10, float("nan"), float("inf")])
def test_unusable_steady_state_roic_falls_back_and_warns(bad_roic):
    outputs, warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.25,
            reinvestment_rate=0.35,
            discount_rate=0.10,
            terminal_growth=0.03,
            shares_outstanding=100.0,
            net_debt=0.0,
        ),
        steady_state_roic=bad_roic,
    )

    assert outputs["terminal_reinvestment_rate"] == 0.35
    assert outputs["ev_range"]["base"] == 1075.9821428571427
    assert warnings[0] == (
        "Steady-state return on capital must be a positive number; terminal "
        "reinvestment held at the explicit-period rate"
    )
