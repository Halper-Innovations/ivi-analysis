"""app/valuation/engine.py::_owner_earnings_intrinsic — unknown net debt is refused.

The helper
substituted 0.0 for an unresolved net debt and carried on, which publishes the
ENTERPRISE value as the EQUITY value — overstating the shares by the whole debt
load. The multiples model had the same defect, on a different path.

``compute_valuation_for_ticker`` refuses first (its preflight sets
MISSING_NET_DEBT and ``can_compute`` False), so today nothing reaches the
substitution; these tests pin the helper itself so the next caller cannot walk
into it. The second test is the control: a resolved net debt is still valued.
"""

from __future__ import annotations

from app.valuation.engine import _owner_earnings_intrinsic

UNKNOWN = "UNKNOWN"


def test_unknown_net_debt_is_refused_rather_than_treated_as_zero():
    base, cons, refs = _owner_earnings_intrinsic(
        fcf_values=[100.0, 100.0, 100.0],
        shares=10.0,
        net_debt=UNKNOWN,
        base_multiple=15.0,
        conservative_multiple=12.0,
    )
    assert base == UNKNOWN
    assert cons == UNKNOWN
    assert "fundamentals.rows[-1].net_debt" in refs


def test_resolved_net_debt_still_values_the_levered_stream():
    # Control: a resolved net debt passes the refusal. The stream is levered free
    # cash flow, so its capitalised value is already an equity value and net debt
    # is not subtracted from it again.
    base, cons, _refs = _owner_earnings_intrinsic(
        fcf_values=[100.0, 100.0, 100.0],
        shares=10.0,
        net_debt=500.0,
        base_multiple=15.0,
        conservative_multiple=12.0,
    )
    assert base == 150.0
    assert cons == 120.0
