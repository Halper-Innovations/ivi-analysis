"""Pure catalyst overlay scorer with deterministic labels.

Turns a catalyst context dict (built from the Form-4 insider-buy
cluster signal and the buyback-acceleration signal) into a single deterministic
``CatalystSignal``. This is a pure function -- no I/O, no network, no LLM -- so
the catalyst overlay stays reproducible.

The catalyst is an orthogonal TIMING axis: a CONFIRMED catalyst upgrades only
the price-trigger status (e.g. DEPLOY_READY -> BUY_CONFIRMED); it never changes
the conviction grade in v1.

Deterministic v1 rules (owner-tunable):
  * score starts at 0.0
  * +2.0 per distinct insider open-market buyer, capped at +6.0 (3 buyers)
  * +1.0 when ``buyback_label == 'ACCELERATING'`` (secondary, WEAK-only signal)
  * label:
      - ``CONFIRMED`` -- ``insider_distinct_buyers >= 2`` (a genuine cluster,
        not one routine buy) AND at least one confirmed open-market purchase
      - ``WEAK`` -- exactly one open-market buyer, OR an accelerating buyback
        alone, OR Form-4 filings are present but the transaction codes are
        unverified (net/parse disabled) so open-market buys cannot be confirmed
      - ``NONE`` -- otherwise

Signature/shape mirrors ``app/score/change_momentum.py`` so the catalyst overlay
is consistent with the rest of the deterministic scoring layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# +2 per distinct open-market buyer, capped at three buyers (+6).
PER_BUYER_POINTS = 2.0
MAX_BUYER_POINTS = 6.0
ACCELERATING_BUYBACK_POINTS = 1.0

# A genuine cluster requires at least two distinct open-market buyers.
CLUSTER_MIN_DISTINCT_BUYERS = 2


@dataclass
class CatalystSignal:
    label: str = "NONE"
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def compute_catalyst_signal(ctx: dict[str, Any] | None) -> CatalystSignal:
    """Score a catalyst context into a deterministic ``CatalystSignal``.

    Recognized ``ctx`` keys (all optional; absent keys are treated as zero/None):
      * ``insider_open_market_purchase_count`` (int) -- confirmed P-coded Form-4
        open-market purchases by officers/directors within the lookback window.
      * ``insider_distinct_buyers`` (int) -- distinct reporting owners with a
        confirmed open-market purchase in the window.
      * ``buyback_label`` (str) -- 'ACCELERATING' / 'STEADY' / 'NONE' from the
        buyback-acceleration detector.
      * ``codes_unverified`` (bool) -- True when Form-4 filings exist in the
        window but transaction codes could not be confirmed (net/parse disabled).
      * ``insider_filing_count`` (int) -- count of in-window Form-4 filings,
        used only to surface the "codes unverified" WEAK path.
    """
    if not isinstance(ctx, dict):
        return CatalystSignal()

    distinct_buyers = int(ctx.get("insider_distinct_buyers") or 0)
    open_market_count = int(ctx.get("insider_open_market_purchase_count") or 0)
    buyback_label = ctx.get("buyback_label") or "NONE"
    codes_unverified = bool(ctx.get("codes_unverified"))
    filing_count = int(ctx.get("insider_filing_count") or 0)

    score = 0.0
    reasons: list[str] = []

    # Buyer points require at least one CONFIRMED open-market purchase. distinct_buyers
    # without any open-market purchase is not a buy signal (e.g. a hand-crafted ctx, or
    # filers whose codes are unverified), so it must score 0 rather than award points
    # for a non-buy. The real context builder only sets distinct_buyers>=2 alongside
    # open_market_count>=2, so this gate is a no-op for the production path.
    has_open_market = open_market_count > 0
    buyer_points = (
        min(MAX_BUYER_POINTS, float(distinct_buyers) * PER_BUYER_POINTS)
        if has_open_market
        else 0.0
    )
    if buyer_points > 0:
        score += buyer_points
        reasons.append(
            f"{distinct_buyers} distinct insider open-market buyer(s) "
            f"({open_market_count} purchase(s)) within the catalyst window."
        )

    accelerating = buyback_label == "ACCELERATING"
    if accelerating:
        score += ACCELERATING_BUYBACK_POINTS
        reasons.append("Buyback acceleration vs. prior fiscal year.")

    if distinct_buyers >= CLUSTER_MIN_DISTINCT_BUYERS and has_open_market:
        label = "CONFIRMED"
    elif distinct_buyers == 1 and has_open_market:
        label = "WEAK"
    elif accelerating:
        label = "WEAK"
    elif codes_unverified and filing_count > 0:
        label = "WEAK"
        reasons.append(
            f"{filing_count} in-window Form-4 filing(s) present but transaction "
            "codes unverified (net/parse disabled); cannot confirm open-market buys."
        )
    else:
        label = "NONE"

    return CatalystSignal(label=label, score=round(score, 2), reasons=reasons)
