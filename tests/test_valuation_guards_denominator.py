"""app/valuation/guards.py::validate_denominators — the two type exclusions.

Two defects: `_is_num` was
a bare isinstance, so NaN validated as a share count (every ordered comparison
below it is False for NaN) and `True` validated as one share. Both tests below
fail on the code before the fix and pass with the exclusions in place.

The ceiling is ten trillion shares in the caller's unit, MILLIONS: the only
caller (app/valuation/engine.py) forwards the value resolve_shares_asof returns,
and that resolver stamps shares_unit "shares_millions" (H&R Block resolves to
123.27 million). An earlier version pinned the
ceiling in absolute shares on the premise that the caller passes absolute
shares, which the resolver's unit stamp contradicts; the literals below are the
same companies in the unit the guard actually sees.
"""

from __future__ import annotations

import math

import pytest

from app.valuation.guards import validate_denominators


def test_nan_share_count_is_refused_rather_than_dividing_through():
    ok, reason, details = validate_denominators({"shares_outstanding": float("nan")})
    assert ok is False
    assert reason == "NON_FINITE_SHARES"
    assert math.isnan(details["shares_outstanding"])


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinite_share_count_is_refused(value):
    ok, reason, _details = validate_denominators({"shares_outstanding": value})
    assert ok is False
    assert reason == "NON_FINITE_SHARES"


@pytest.mark.parametrize("value", [True, False])
def test_a_bool_is_not_a_share_count(value):
    ok, reason, details = validate_denominators({"shares_outstanding": value})
    assert ok is False
    assert reason == "MISSING_SHARES_DENOMINATOR"
    # The bool is echoed back untouched, never coerced to 1.0 or 0.0.
    assert details["shares_outstanding"] is value


@pytest.mark.parametrize(
    "shares",
    [
        0.55,  # Berkshire A, the smallest real cap table, in millions
        145.461,  # ResMed FY2021 in millions, the share count the scale defect mangled
        15_000.0,  # the largest listed counts, in millions
    ],
)
def test_real_share_counts_in_millions_still_validate(shares):
    """The unit the caller actually passes is millions of shares.

    app/valuation/engine.py forwards the value resolve_shares_asof returns, which
    is stamped shares_unit "shares_millions"; every real cap table sits far
    below the ten-trillion-share ceiling in that unit.
    """
    ok, reason, details = validate_denominators({"shares_outstanding": shares})
    assert ok is True
    assert reason is None
    assert details["shares_outstanding"] == shares


def test_the_ceiling_in_millions_and_the_zero_and_negative_reasons():
    # Ten trillion shares is 10,000,000 million shares.
    assert validate_denominators({"shares_outstanding": 10_000_000.0})[:2] == (True, None)
    assert validate_denominators({"shares_outstanding": 10_000_001.0})[:2] == (
        False,
        "ABSURD_SHARES",
    )
    assert validate_denominators({"shares_outstanding": 0.0})[:2] == (False, "ZERO_SHARES")
    assert validate_denominators({"shares_outstanding": -1.0})[:2] == (False, "NEGATIVE_SHARES")
    assert validate_denominators({})[:2] == (False, "MISSING_SHARES_DENOMINATOR")
