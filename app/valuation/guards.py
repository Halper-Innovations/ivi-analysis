from __future__ import annotations

import math
from typing import Any


def _is_num(value: Any) -> bool:
    # bool is a subclass of int: without the exclusion True validates as one share and
    # False as zero shares. Matches app/valuation/shares.py:_is_num.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_denominators(inputs: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
    shares = inputs.get("shares_outstanding")
    details: dict[str, Any] = {
        "shares_outstanding": shares,
    }
    if not _is_num(shares):
        return False, "MISSING_SHARES_DENOMINATOR", details

    shares_value = float(shares)
    details["shares_outstanding"] = shares_value
    # Every ordered comparison below is False for NaN, so NaN would validate as a share
    # count and divide straight through to a NaN intrinsic value. Reject it first.
    if not math.isfinite(shares_value):
        return False, "NON_FINITE_SHARES", details
    if shares_value == 0.0:
        return False, "ZERO_SHARES", details
    if shares_value < 0.0:
        return False, "NEGATIVE_SHARES", details
    # The ceiling is ten trillion shares expressed in the caller's unit, which is
    # MILLIONS: app/valuation/engine.py forwards the value resolve_shares_asof
    # returns, and that resolver stamps shares_unit "shares_millions" (H&R Block
    # resolves to 123.27, not 123,265,922). An earlier version kept the
    # ceiling in absolute shares on the premise that the caller passes absolute
    # shares; the resolver's own unit stamp shows it does not, so the absolute
    # ceiling could never fire. Same ceiling, right unit.
    # A scale check that CAN see a real unit slip in a filed count lives with the
    # share chooser, in app/market/shares_guard.py (it replaced the power-of-ten
    # check once in app/valuation/shares.py).
    if shares_value > 10_000_000.0:
        return False, "ABSURD_SHARES", details
    return True, None, details
