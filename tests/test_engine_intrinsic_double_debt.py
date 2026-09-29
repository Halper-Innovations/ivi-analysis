"""app/valuation/engine.py::_owner_earnings_intrinsic must not charge the debt twice.

The engine's owner-earnings stack
capitalises a LEVERED free cash flow (CFO − capex, with cash interest already paid
inside CFO) at an equity-style multiple and then subtracts net debt again
(engine.py:1177-1181). The definitional block in
app/valuation/fcf.py says so in as many words: "Capitalising it and then
subtracting net debt counts the debt twice ... engine.py::_owner_earnings_intrinsic
does exactly that." Observed: FCF 100 × 10 − net debt 500 = 500 / 10 shares =
$50 a share. A levered stream capitalised is already an equity value: $100 a share.
"""

from __future__ import annotations

from app.valuation.engine import _owner_earnings_intrinsic


def test_levered_fcf_multiple_is_an_equity_value_and_debt_is_not_subtracted_again():
    base, conservative, refs = _owner_earnings_intrinsic(
        fcf_values=[100.0],
        shares=10.0,
        net_debt=500.0,
        base_multiple=10.0,
        conservative_multiple=8.0,
    )
    assert base == 100.0
    assert conservative == 80.0
    assert "fundamentals.rows[*].fcf" in refs


def test_net_cash_is_not_added_a_second_time_either():
    # The interest income on the cash is already inside CFO, so a net-cash balance
    # must not be added on top of the capitalised stream.
    base, conservative, _ = _owner_earnings_intrinsic(
        fcf_values=[100.0],
        shares=10.0,
        net_debt=-500.0,
        base_multiple=10.0,
        conservative_multiple=8.0,
    )
    assert base == 100.0
    assert conservative == 80.0


def test_the_median_of_positive_observations_is_still_the_stream():
    base, _, _ = _owner_earnings_intrinsic(
        fcf_values=[50.0, 100.0, 150.0],
        shares=10.0,
        net_debt=0.0,
        base_multiple=10.0,
        conservative_multiple=8.0,
    )
    assert base == 100.0
