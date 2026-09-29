from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.fundamentals.normalize import UNKNOWN

DEFAULT_TAX_RATE = 0.25
DEFAULT_REINVESTMENT_RATE = 0.35

# Effective tax rates read off real filings are occasionally nonsense: above 100% when the
# pre-tax base is tiny, below 0% on a valuation-allowance release. Unclamped they flip the
# sign of NOPAT and hand back a negative (or inflated) intrinsic value that reads as valid.
# 0%-50% brackets every plausible combined statutory rate; outside it the number is an
# artifact of the denominator, not a tax rate (2026-08-29).
TAX_RATE_FLOOR = 0.0
TAX_RATE_CEILING = 0.50


@dataclass
class DCFInputs:
    revenue: float | None
    operating_margin: float | None
    tax_rate: float | None
    reinvestment_rate: float | None
    discount_rate: float
    terminal_growth: float
    shares_outstanding: float | None
    net_debt: float | None


def _project_fcf(revenue: float, margin: float, tax_rate: float, reinvestment_rate: float) -> float:
    ebit = revenue * margin
    nopat = ebit * (1 - tax_rate)
    reinvestment = max(0.0, nopat * reinvestment_rate)
    return nopat - reinvestment


def _unknown_range() -> dict[str, Any]:
    return {"low": UNKNOWN, "base": UNKNOWN, "high": UNKNOWN}


def _invalid_assumption_outputs() -> dict[str, Any]:
    return {
        "ev_range": _unknown_range(),
        "equity_value_range": _unknown_range(),
        "per_share_range": _unknown_range(),
        "confidence": "LOW",
        "terminal_value_share": UNKNOWN,
        "implied_terminal_ebit_multiple": UNKNOWN,
    }


def _resolve_tax_rate(raw: float | None, warnings: list[str]) -> float:
    """Return a usable tax rate, warning on default or clamp (never silently sign-flipping)."""
    if raw is None:
        warnings.append("Tax rate defaulted to conservative 25%")
        return DEFAULT_TAX_RATE
    rate = float(raw)
    if not math.isfinite(rate):
        warnings.append("Tax rate not a finite number; defaulted to conservative 25%")
        return DEFAULT_TAX_RATE
    clamped = min(max(rate, TAX_RATE_FLOOR), TAX_RATE_CEILING)
    if clamped != rate:
        warnings.append(
            f"Tax rate {rate:.1%} outside defensible range [0.0%, 50.0%]; clamped to {clamped:.1%}"
        )
    return clamped


def _resolve_terminal_reinvestment(
    reinvest: float,
    terminal_growth: float,
    discount_rate: float,
    steady_state_roic: float | None,
    warnings: list[str],
) -> tuple[float, float | str]:
    """Terminal reinvestment rate plus the steady-state ROIC it implies.

    In steady state reinvestment = terminal_growth / ROIC. Carrying the explicit-period
    rate into perpetuity (the default here, unchanged) therefore asserts a return on
    capital of terminal_growth / reinvestment: 3% growth funded by 35% reinvestment
    implies 8.6%, usually below the discount rate, which understates terminal value.
    The default is left alone; callers who want the steady-state identity pass
    ``steady_state_roic`` explicitly. Either way the assumption is warned about.
    """
    if steady_state_roic is not None:
        roic = float(steady_state_roic)
        if math.isfinite(roic) and roic > 0:
            terminal_reinvest = min(max(terminal_growth / roic, 0.0), 1.0)
            warnings.append(
                f"Terminal reinvestment set to {terminal_reinvest:.1%} from steady-state return "
                f"on capital {roic:.1%}; explicit period reinvests {reinvest:.1%}"
            )
            return terminal_reinvest, roic
        warnings.append(
            "Steady-state return on capital must be a positive number; terminal reinvestment "
            "held at the explicit-period rate"
        )
    if reinvest > 0 and terminal_growth > 0:
        implied = terminal_growth / reinvest
        if implied < discount_rate:
            warnings.append(
                f"Terminal reinvestment held at the explicit {reinvest:.1%}, which implies a "
                f"steady-state return on capital of {implied:.1%}, below the "
                f"{discount_rate:.1%} discount rate, so terminal value is understated"
            )
        return reinvest, implied
    return reinvest, UNKNOWN


def run_dcf_lite(
    inputs: DCFInputs, *, steady_state_roic: float | None = None
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if inputs.revenue is None or inputs.operating_margin is None:
        return (
            {
                "ev_range": _unknown_range(),
                "equity_value_range": _unknown_range(),
                "per_share_range": _unknown_range(),
                "confidence": "LOW",
            },
            ["Revenue or operating margin UNKNOWN"],
        )

    tax = _resolve_tax_rate(inputs.tax_rate, warnings)
    reinvest = (
        inputs.reinvestment_rate
        if inputs.reinvestment_rate is not None
        else DEFAULT_REINVESTMENT_RATE
    )
    if inputs.reinvestment_rate is None:
        warnings.append("Reinvestment rate defaulted to conservative 35%")

    # The Gordon denominator (disc - g) is non-positive here, so no scenario has a defined
    # terminal value. This used to be caught only inside discounted_ev, which returned the
    # finite explicit-period PV in the first slot; the caller's `base_ev == inf` test was
    # therefore never true and an invalid WACC/g pair produced a finite enterprise value
    # with the whole terminal value dropped. Short-circuit before any math (2026-08-29).
    if inputs.discount_rate <= inputs.terminal_growth:
        warnings.append("Invalid discount/terminal growth assumptions")
        return _invalid_assumption_outputs(), warnings

    terminal_reinvest, implied_roic = _resolve_terminal_reinvestment(
        reinvest, inputs.terminal_growth, inputs.discount_rate, steady_state_roic, warnings
    )

    growth_low, growth_base, growth_high = 0.00, 0.03, 0.06
    margin_base = inputs.operating_margin
    # Scenario margins must stay ordered low <= base <= high. Scaling a negative margin by
    # 0.8/1.15 inverted the band (m=-0.10 gave low 0.0, high -0.115: the "optimistic" case
    # was the deepest loss), so a negative margin widens downward for the pessimistic edge
    # and shrinks toward zero for the optimistic edge (2026-08-29).
    if margin_base < 0:
        margin_low = margin_base * 1.20
        margin_high = margin_base * 0.85
        warnings.append(
            "Operating margin negative; pessimistic case widens the loss instead of "
            "turning it positive"
        )
    else:
        margin_low = margin_base * 0.80
        margin_high = min(0.45, margin_base * 1.15)
    margin_low = min(margin_low, margin_base)
    margin_high = max(margin_high, margin_base)

    base_revenue = float(inputs.revenue)

    def discounted_ev(growth: float, margin: float) -> tuple[float, float, float]:
        revenue = base_revenue
        disc = inputs.discount_rate
        years = 5
        pv = 0.0
        for year in range(1, years + 1):
            revenue = revenue * (1 + growth)
            fcf = _project_fcf(revenue, margin, tax, reinvest)
            pv += fcf / ((1 + disc) ** year)
        ebit_year5 = revenue * margin
        terminal_fcf = _project_fcf(
            revenue * (1 + inputs.terminal_growth), margin, tax, terminal_reinvest
        )
        if disc <= inputs.terminal_growth:
            # Unreachable behind the guard above; every slot is inf so that any future
            # caller testing one element sees the failure rather than a finite PV.
            return float("inf"), float("inf"), float("inf")
        terminal_value = terminal_fcf / (disc - inputs.terminal_growth)
        pv_terminal = terminal_value / ((1 + disc) ** years)
        implied_terminal_multiple = terminal_value / ebit_year5 if ebit_year5 else float("inf")
        return pv + pv_terminal, pv_terminal, implied_terminal_multiple

    low_ev, low_term, low_mult = discounted_ev(growth_low, margin_low)
    base_ev, base_term, base_mult = discounted_ev(growth_base, margin_base)
    high_ev, high_term, high_mult = discounted_ev(growth_high, margin_high)

    if base_ev == float("inf"):
        warnings.append("Invalid discount/terminal growth assumptions")
        return _invalid_assumption_outputs(), warnings

    # base_ev is exactly 0.0 whenever the base margin is 0.0 (a real filing outcome); the
    # ratio used to be taken before any zero check and raised ZeroDivisionError (2026-08-29).
    terminal_share: float | str = base_term / base_ev if base_ev else UNKNOWN
    if isinstance(terminal_share, float) and terminal_share > 0.75:
        warnings.append("DCF dominated by terminal value (>75%)")
    elif not isinstance(terminal_share, float):
        warnings.append("Base enterprise value is zero; terminal-value share unavailable")

    net_debt_available = (
        isinstance(inputs.net_debt, (int, float))
        and not isinstance(inputs.net_debt, bool)
        and math.isfinite(float(inputs.net_debt))
    )
    if net_debt_available:
        net_debt = float(inputs.net_debt)
        equity_value_range = {
            "low": low_ev - net_debt,
            "base": base_ev - net_debt,
            "high": high_ev - net_debt,
        }
    else:
        equity_value_range = _unknown_range()
        warnings.append("Net debt UNKNOWN for equity-value DCF")

    if inputs.shares_outstanding in (None, 0) or not net_debt_available:
        per_share = _unknown_range()
        if inputs.shares_outstanding in (None, 0):
            warnings.append("Shares outstanding UNKNOWN for per-share DCF")
    else:
        per_share = {
            "low": equity_value_range["low"] / inputs.shares_outstanding,
            "base": equity_value_range["base"] / inputs.shares_outstanding,
            "high": equity_value_range["high"] / inputs.shares_outstanding,
        }

    outputs = {
        "ev_range": {"low": low_ev, "base": base_ev, "high": high_ev},
        "equity_value_range": equity_value_range,
        "per_share_range": per_share,
        "terminal_value_share": terminal_share,
        "implied_terminal_ebit_multiple": {
            "low": low_mult,
            "base": base_mult,
            "high": high_mult,
        },
        "terminal_reinvestment_rate": terminal_reinvest,
        "implied_steady_state_roic": implied_roic,
        "confidence": "MEDIUM" if net_debt_available else "LOW",
    }
    return outputs, warnings
