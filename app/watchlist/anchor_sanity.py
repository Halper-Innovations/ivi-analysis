"""Per-share anchor sanity band.

Pure helper implementing the per-share sanity gate: given a per-share anchor,
a trusted trailing reference price, and a configurable band, return a verdict
plus the ratio. No DB or network access.

Callers (``store.py`` at add time, ``triggers.py`` at recheck time) supply the
trusted reference price (e.g. a trailing fiscal-year median price) so an anchor
that is an order-of-magnitude/units artifact is QUARANTINEd rather than shipped
as a buy target.
"""

from __future__ import annotations

from dataclasses import dataclass

# A per-share anchor must sit within [LOW, HIGH] multiples of the trusted
# trailing reference price, else it is quarantined as a magnitude/units error.
PER_SHARE_ANCHOR_LOW_MULTIPLE = 0.2
PER_SHARE_ANCHOR_HIGH_MULTIPLE = 5.0


@dataclass(frozen=True)
class AnchorSanityResult:
    """Outcome of the per-share anchor sanity check.

    ``verdict`` is one of:
      - ``'OK'`` — anchor within the [0.2x, 5.0x] band of the reference price
      - ``'QUARANTINE_ANCHOR_ABOVE_BAND'`` — anchor exceeds 5.0x the reference
      - ``'QUARANTINE_ANCHOR_BELOW_BAND'`` — anchor below 0.2x the reference
      - ``'UNASSESSABLE_NO_REFERENCE'`` — anchor or reference missing/non-positive

    ``ratio`` is anchor / reference_price rounded to 3 decimal places, or
    ``None`` when the verdict is ``'UNASSESSABLE_NO_REFERENCE'``.
    """

    verdict: str
    ratio: float | None


def evaluate_anchor(anchor, reference_price) -> AnchorSanityResult:
    """Evaluate a per-share anchor against a trusted trailing reference price."""
    if anchor is None or reference_price is None or reference_price <= 0 or anchor <= 0:
        return AnchorSanityResult("UNASSESSABLE_NO_REFERENCE", None)

    ratio = round(anchor / reference_price, 3)
    if ratio > PER_SHARE_ANCHOR_HIGH_MULTIPLE:
        return AnchorSanityResult("QUARANTINE_ANCHOR_ABOVE_BAND", ratio)
    if ratio < PER_SHARE_ANCHOR_LOW_MULTIPLE:
        return AnchorSanityResult("QUARANTINE_ANCHOR_BELOW_BAND", ratio)
    return AnchorSanityResult("OK", ratio)
