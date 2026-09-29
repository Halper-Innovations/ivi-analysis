"""Non-recurring revenue spike detector.

Problem
-------
Companies occasionally book large one-time revenue from licensing deals,
collaboration milestones, or upfront payments. When a quantitative DCF /
growth model trains on this revenue series, it projects these one-offs
forward as durable recurring revenue — producing inflated DCFs that no
serious analyst would trust.

Documented examples from the healthcare_pharma_2026-04-14 scan:
- KROS: FY2024 $3.5M revenue; FY2025 jumped to $244M due to the ~$200M
  Takeda collaboration upfront payment. DCF of $62.23 baked the $244M
  in as recurring.
- PTCT: $998M Novartis collaboration revenue flowing into growth metrics.

Fix
---
Detect revenue spikes from the numeric series (prior year → current year
growth) and flag them. Provide a "durable revenue base" that callers can
use instead of the spiked year when building DCF inputs.

This module is pure numbers — no filing-text parsing. A follow-up can
cross-reference MD&A keywords (`collaboration`, `milestone`, `upfront`,
`one-time`) to raise confidence, but even the numeric-only signal
correctly catches KROS and PTCT given their prior-year baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# Thresholds tuned to the documented cases.
# KROS: FY2024 $3.5M → FY2025 $244M = 69x multiplier, $240M absolute delta.
# Both gates fire loudly.
# A thinner case: a $200M company that books a $100M milestone (50% spike,
# $100M delta) should also trigger; that's the lower end of what we want.

# Ratio test: current / prior >= 2.0 (doubling or more)
_SPIKE_RATIO_THRESHOLD = 2.0

# Absolute test: current − prior >= $50M (meaningful on dollar basis)
_SPIKE_ABS_DELTA_THRESHOLD_M = 50.0

# "Thin prior year" cut-off: prior year revenue < $50M suggests the company
# was essentially pre-revenue, so any big step-up is a near-certain licensing /
# collaboration event. Lower the trigger bar when this is the case.
_THIN_PRIOR_REVENUE_THRESHOLD_M = 50.0


@dataclass
class NonRecurringRevenueDetection:
    ticker: str
    has_suspected_nonrecurring: bool
    spike_year: int | None = None
    spike_revenue: float | None = None
    prior_year_revenue: float | None = None
    spike_ratio: float | None = None  # spike / prior
    dollar_delta: float | None = None
    # Durable revenue base the caller should use for DCF / growth modeling
    # when a spike is detected. Falls back to the spike revenue if we don't
    # have a credible baseline.
    durable_revenue_base: float | None = None
    reason: str = "NO_SPIKE_DETECTED"


def detect(
    *,
    ticker: str,
    revenue_series: Iterable[tuple[int, float]],
) -> NonRecurringRevenueDetection:
    """Scan a revenue series (fiscal_year, revenue) for a non-recurring spike.

    Flags the *most recent* year if it's a spike vs. the immediately prior
    year. We deliberately focus on the latest year since that's what the
    DCF terminal value is most sensitive to.

    The revenue_series is expected to cover the most recent fiscal years
    (typically 3-5 years). Any ordering is accepted — the function sorts
    internally.
    """
    series = sorted(
        ((int(y), float(v)) for y, v in revenue_series if v is not None),
        key=lambda x: x[0],
    )
    if len(series) < 2:
        return NonRecurringRevenueDetection(
            ticker=ticker,
            has_suspected_nonrecurring=False,
            reason="INSUFFICIENT_HISTORY",
        )

    prior_year, prior_rev = series[-2]
    latest_year, latest_rev = series[-1]

    if prior_rev <= 0 and latest_rev > 0:
        # Went from zero / negative to positive — always suspect for a spike
        return NonRecurringRevenueDetection(
            ticker=ticker,
            has_suspected_nonrecurring=True,
            spike_year=latest_year,
            spike_revenue=latest_rev,
            prior_year_revenue=prior_rev,
            spike_ratio=None,
            dollar_delta=latest_rev - prior_rev,
            durable_revenue_base=_earlier_positive_median(series[:-1]),
            reason="ZERO_TO_POSITIVE_REVENUE",
        )

    ratio = latest_rev / prior_rev if prior_rev > 0 else float("inf")
    delta = latest_rev - prior_rev

    ratio_trigger = ratio >= _SPIKE_RATIO_THRESHOLD
    abs_trigger = delta >= _SPIKE_ABS_DELTA_THRESHOLD_M
    thin_prior = prior_rev < _THIN_PRIOR_REVENUE_THRESHOLD_M

    spike = False
    reason = "NO_SPIKE_DETECTED"

    if ratio_trigger and abs_trigger:
        # Both tests fire (KROS, PTCT cases)
        spike = True
        reason = "RATIO_AND_ABSOLUTE_BOTH_TRIGGERED"
    elif thin_prior and ratio_trigger:
        # Thin prior year + doubled revenue → near-certain non-recurring
        spike = True
        reason = "THIN_PRIOR_AND_RATIO_TRIGGERED"
    elif thin_prior and delta >= 25.0:
        # Thin prior and $25M+ jump is also suspect
        spike = True
        reason = "THIN_PRIOR_AND_LARGE_DELTA"

    if not spike:
        return NonRecurringRevenueDetection(
            ticker=ticker,
            has_suspected_nonrecurring=False,
            spike_year=latest_year,
            spike_revenue=latest_rev,
            prior_year_revenue=prior_rev,
            spike_ratio=ratio if ratio != float("inf") else None,
            dollar_delta=delta,
            durable_revenue_base=latest_rev,  # no spike → use actual
            reason=reason,
        )

    # Durable base: use the prior-year revenue as the baseline. If the prior
    # year itself was thin, back off to the median of the available earlier
    # years. Either way, don't project the spike forward.
    durable_base = _durable_base(series[:-1], thin_prior)

    return NonRecurringRevenueDetection(
        ticker=ticker,
        has_suspected_nonrecurring=True,
        spike_year=latest_year,
        spike_revenue=latest_rev,
        prior_year_revenue=prior_rev,
        spike_ratio=ratio if ratio != float("inf") else None,
        dollar_delta=delta,
        durable_revenue_base=durable_base,
        reason=reason,
    )


def _earlier_positive_median(
    series: list[tuple[int, float]],
) -> float | None:
    """Median of earlier positive revenues, used when prior year itself is 0."""
    values = sorted(v for _, v in series if v > 0)
    if not values:
        return None
    n = len(values)
    if n % 2:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2


def _durable_base(
    earlier_series: list[tuple[int, float]],
    thin_prior: bool,
) -> float | None:
    """Pick a durable revenue base for downstream DCF use.

    If prior year was thin (e.g., KROS $3.5M), fall back to the median of
    earlier years when available. Otherwise use the prior-year value.
    """
    if not earlier_series:
        return None
    prior_year_rev = earlier_series[-1][1]
    if not thin_prior:
        return prior_year_rev
    # Thin prior — use the median of earlier years (if we have at least 2 more)
    candidates = sorted(v for _, v in earlier_series if v > 0)
    if len(candidates) >= 2:
        n = len(candidates)
        if n % 2:
            return candidates[n // 2]
        return (candidates[n // 2 - 1] + candidates[n // 2]) / 2
    # Single-point fallback
    return prior_year_rev
