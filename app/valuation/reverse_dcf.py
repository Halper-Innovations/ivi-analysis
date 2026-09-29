from __future__ import annotations

import math
from typing import Any, TypeGuard

from app.fundamentals.normalize import UNKNOWN


# The growth bracket the solver searches. Named once so that the value returned
# on a saturated solve and the label describing it are derived from the same
# two numbers rather than from independent reasoning about signs.
GROWTH_BOUND_LOW = -0.25
GROWTH_BOUND_HIGH = 0.60


def _finite(value: Any) -> TypeGuard[float]:
    """True only for a real, finite number. Bools are not numbers here."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _not_computable(reason: str) -> tuple[dict[str, Any], list[str]]:
    """The module's single non-computable return: UNKNOWN legs plus a reason.

    Every existing failure exit in this module already takes this shape, and
    the new ones join it. The sibling convention splits on whether the model is
    merely awkward or actually undefined: dcf_lite repairs a nuisance parameter
    in place (tax clamped to a defensible band, scenario margins re-ordered,
    each with a warning) but returns UNKNOWN for the one input that leaves the
    model undefined, an impossible discount/terminal-growth pair; sanity_checks
    skips a ticker whose share count it cannot prove rather than substituting
    one. A rate at or above 100% belongs to the undefined class here, not the
    nuisance class; see the guards below.
    """
    return ({"implied_growth": UNKNOWN, "feasibility_gap_score": UNKNOWN}, [reason])


def implied_growth_from_price(
    market_price: float | None,
    shares_outstanding: float | None,
    net_debt: float | None,
    base_revenue: float | None,
    margin: float | None,
    tax_rate: float = 0.25,
    reinvestment_rate: float = 0.35,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.02,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if market_price is None:
        return _not_computable("Market price UNKNOWN")
    if shares_outstanding in (None, 0) or base_revenue is None or margin is None:
        return _not_computable("Insufficient inputs for reverse DCF")
    if _finite(net_debt):
        net_debt_value = float(net_debt)
    else:
        return _not_computable("Net debt UNKNOWN")
    if not all(
        _finite(value) for value in (market_price, shares_outstanding, base_revenue, margin)
    ):
        # NaN propagates through every comparison as False, so an unguarded NaN
        # margin used to walk the bisection to a bracket end and return it as a
        # genuine interior solve with a feasibility score attached.
        return _not_computable("Non-finite price, share count, revenue or margin")

    # Rate inputs that real filings do produce out of range: a 150% effective
    # tax rate off a tiny pre-tax base, a reinvestment rate above 100%. Both
    # enter the model as (1 - rate), so a value at or above 1.0 flips the sign
    # of every modeled cash flow, reverses the direction of the solve, and
    # makes the implied growth it returns meaningless. dcf_lite clamps its own
    # tax rate because it has no solver whose direction can invert; here the
    # inversion IS the failure, so this is the undefined class and the answer
    # is a refusal. The accepted domain is the one the model stays sign-stable
    # over, [0, 1), which is also the bound LIVE_DESK_SPEC §20.2 already
    # refuses on.
    if not _finite(tax_rate) or not 0.0 <= float(tax_rate) < 1.0:
        return _not_computable("Effective tax rate outside 0-100%")
    if not _finite(reinvestment_rate) or not 0.0 <= float(reinvestment_rate) < 1.0:
        return _not_computable("Reinvestment rate outside 0-100%")
    # (1 + discount_rate) is the compounding base: at or below -1 it is zero or
    # negative and the discount factors blow up (a ZeroDivisionError escaped the
    # module at discount_rate=-1.0).
    if not _finite(discount_rate) or float(discount_rate) <= -1.0:
        return _not_computable("Invalid discount rate")
    if not _finite(terminal_growth):
        return _not_computable("Invalid terminal growth")

    # Guard the Gordon-growth terminal denominator. If discount_rate <= terminal_growth
    # the perpetuity is undefined (non-positive / infinite present value). Mirror
    # dcf_lite (dcf_lite.py): surface the bad assumptions as UNKNOWN + a warning
    # rather than silently clamping the spread to a tiny positive number, which would
    # fabricate a finite-but-meaningless implied growth (FIX 3).
    if discount_rate <= terminal_growth:
        return _not_computable("Invalid discount/terminal growth assumptions")

    target_equity = market_price * shares_outstanding
    target_ev = target_equity + net_debt_value

    def dcf_ev(growth: float) -> float:
        years = 5
        revenue = base_revenue
        pv = 0.0
        for year in range(1, years + 1):
            revenue *= 1 + growth
            ebit = revenue * margin
            nopat = ebit * (1 - tax_rate)
            fcf = nopat * (1 - reinvestment_rate)
            pv += fcf / ((1 + discount_rate) ** year)
        terminal_fcf = (
            revenue * (1 + terminal_growth) * margin * (1 - tax_rate) * (1 - reinvestment_rate)
        )
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv += terminal_value / ((1 + discount_rate) ** years)
        return pv

    bound_low, bound_high = GROWTH_BOUND_LOW, GROWTH_BOUND_HIGH
    margin_sign = "POSITIVE" if margin > 0 else ("NEGATIVE" if margin < 0 else "ZERO")

    # Bracket pre-check + direction awareness (audit findings
    # bisection-monotonicity-negative-margin and
    # sat-conflates-deep-cheap-with-unsolvable): every term of dcf_ev is a
    # positive multiple of (1 + growth) ** year, so the curve is strictly
    # monotone and its direction is fixed by the sign of the cash-flow
    # constant — DECREASING when that constant is negative. The two endpoint
    # evaluations therefore recover the direction exactly.
    f_low = dcf_ev(bound_low)
    f_high = dcf_ev(bound_high)
    if not (_finite(f_low) and _finite(f_high)):
        return _not_computable("Model produced non-finite values")
    if f_low == f_high:
        # A flat objective has no unique root: every growth rate yields the same
        # enterprise value, so no growth rate is the implied one. Bisection used
        # to run anyway and hand back the low bracket end as a genuine interior
        # solve, feasibility score included.
        return _not_computable("Modeled cash flow does not respond to growth")

    increasing = f_high > f_low
    ev_min, ev_max = (f_low, f_high) if increasing else (f_high, f_low)
    if target_ev < ev_min or target_ev > ev_max:
        # ONE boolean fixes both the returned growth and its label, so the two
        # cannot disagree. at_low_bound is true when the nearest achievable
        # endpoint is the LOW growth bound: for an increasing curve that is a
        # price below the worst-case-growth value; for a decreasing curve the
        # growth endpoints swap, so a price above everything the model can
        # produce also lands on the LOW growth bound.
        #
        # The label names the bound the returned value SITS AT. Deriving it
        # from the price side instead is what returned bound "HIGH" alongside
        # implied_growth -0.25 (a recorded defect). The price
        # side is not lost: it is recoverable from (bound_label, margin_sign),
        # which is the pair compute_expectations_gap already consumes.
        at_low_bound = (target_ev < ev_min) == increasing
        implied_growth = bound_low if at_low_bound else bound_high
        bound_label = "LOW" if at_low_bound else "HIGH"
        warnings.append("Implied growth solve saturated at bound")
        return (
            {
                "implied_growth": implied_growth,
                # A score computed off a clipped bound is fabricated; nulled
                # for saturated solves (no live consumer).
                "feasibility_gap_score": None,
                "target_ev": target_ev,
                "implied_growth_saturated": True,
                "implied_growth_saturated_bound": bound_label,
                "margin_sign": margin_sign,
            },
            warnings,
        )

    low, high = bound_low, bound_high
    for _ in range(50):
        mid = (low + high) / 2
        ev = dcf_ev(mid)
        if (ev < target_ev) == increasing:
            low = mid
        else:
            high = mid

    implied_growth = (low + high) / 2
    feasibility_gap = max(0.0, implied_growth - 0.10) * 100
    if implied_growth > 0.20:
        warnings.append("Implied growth exceeds 20% for five years")

    return (
        {
            "implied_growth": implied_growth,
            "feasibility_gap_score": feasibility_gap,
            "target_ev": target_ev,
            # Bracketed interior solves are genuine even within 1e-3 of a
            # bound (the old tolerance discarded legitimate solves).
            "implied_growth_saturated": False,
            "implied_growth_saturated_bound": None,
            "margin_sign": margin_sign,
        },
        warnings,
    )
