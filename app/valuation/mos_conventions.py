"""Margin-of-safety convention helpers.

Two incompatible conventions ship in this codebase under similar names
(audit finding dual-mos-convention-same-name). They are NOT interchangeable:

  TEXTBOOK margin of safety (the scorecard convention):
      mos = (intrinsic - price) / intrinsic
      "the fraction of intrinsic value NOT paid for at the current price"
      Range (-inf, 1). Stored by valuation_writer in scorecard["discounts"]
      and in pricing_zone_detail margin_of_safety_* fields.

  UPSIDE RATIO (graham_dodd / intrinsic_discipline / value_gates / scout):
      upside = intrinsic / price - 1
      "how far price would have to rise to reach intrinsic value"
      Range (-1, inf). For intrinsic=150, price=100: textbook = 0.333,
      upside = 0.50 — same position, different number.

Surfaces rendering either number must label the convention explicitly.

This module is the ONE place that inverts the textbook discount back to an
intrinsic value. The wrong inversion price/(1+d) — which sign-inverts the
cheap/expensive relationship on every input — shipped at five independent
call sites before this helper existed (audit: graham-discount-inversion).
"""
from __future__ import annotations

# valuation_writer stores discounts[name] = -1.0 when intrinsic value <= 0
# (non-computable sentinel). A legitimate d == -1.0 (price exactly 2x the
# intrinsic value) is indistinguishable and also treated as non-computable —
# a documented, acceptable collision.
TEXTBOOK_DISCOUNT_SENTINEL = -1.0


def graham_value_from_textbook_discount(
    price: float | None,
    discount: float | None,
) -> float | None:
    """Invert a TEXTBOOK margin-of-safety discount back to intrinsic value.

    d = (iv - price) / iv  =>  iv = price / (1 - d)

    Returns None for missing inputs, the -1.0 sentinel, non-positive price,
    or d >= 1 (impossible for a finite positive intrinsic value).
    """
    if not isinstance(price, (int, float)) or not isinstance(discount, (int, float)):
        return None
    if float(price) <= 0:
        return None
    d = float(discount)
    if d == TEXTBOOK_DISCOUNT_SENTINEL or d >= 1.0:
        return None
    return float(price) / (1.0 - d)
