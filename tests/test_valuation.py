from app.fundamentals.normalize import UNKNOWN
from app.valuation.dcf_lite import DCFInputs, run_dcf_lite
from app.valuation.reverse_dcf import implied_growth_from_price


def test_dcf_lite_outputs_range_ordering():
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

    rng = outputs["per_share_range"]
    assert rng["low"] < rng["base"] < rng["high"]
    assert outputs["confidence"] in {"LOW", "MEDIUM"}


def test_dcf_lite_unknown_when_key_inputs_missing():
    outputs, warnings = run_dcf_lite(
        DCFInputs(
            revenue=None,
            operating_margin=None,
            tax_rate=None,
            reinvestment_rate=None,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=None,
            net_debt=None,
        )
    )
    assert outputs["ev_range"]["base"] == UNKNOWN


def test_dcf_lite_missing_net_debt_keeps_ev_but_blocks_equity_and_per_share():
    outputs, warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.25,
            reinvestment_rate=0.30,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=None,
        )
    )

    assert isinstance(outputs["ev_range"]["base"], float)
    assert outputs["equity_value_range"] == {
        "low": UNKNOWN,
        "base": UNKNOWN,
        "high": UNKNOWN,
    }
    assert outputs["per_share_range"] == {
        "low": UNKNOWN,
        "base": UNKNOWN,
        "high": UNKNOWN,
    }
    assert outputs["confidence"] == "LOW"
    assert warnings[-1] == "Net debt UNKNOWN for equity-value DCF"


def test_dcf_lite_explicit_zero_net_debt_is_valid():
    outputs, warnings = run_dcf_lite(
        DCFInputs(
            revenue=1000.0,
            operating_margin=0.15,
            tax_rate=0.25,
            reinvestment_rate=0.30,
            discount_rate=0.10,
            terminal_growth=0.02,
            shares_outstanding=100.0,
            net_debt=0.0,
        )
    )

    assert outputs["equity_value_range"]["base"] == outputs["ev_range"]["base"]
    assert isinstance(outputs["per_share_range"]["base"], float)
    assert outputs["confidence"] == "MEDIUM"
    assert "Net debt UNKNOWN for equity-value DCF" not in warnings


def test_reverse_dcf_unknown_without_market_price():
    outputs, warnings = implied_growth_from_price(
        market_price=None,
        shares_outstanding=100,
        net_debt=0,
        base_revenue=1000,
        margin=0.1,
    )
    assert outputs["implied_growth"] == UNKNOWN
