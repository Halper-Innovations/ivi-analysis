"""Per-name margin-of-safety discount.

Pure function. No DB. No file I/O. No LLM calls. Mirrors the purity of
``app/valuation/conviction.py``.

Replaces the legacy flat ``BUY_PRICE_DISCOUNT_TO_ANCHOR = 0.25`` haircut
(``buy_target = anchor * 0.75`` for every name) with a bounded per-name
discount that scales with conviction grade, intrinsic-method dispersion, and
(eventually) realized volatility:

    discount = clamp(base[grade, confidence]
                     + w_disp * dispersion
                     + w_vol  * vol_excess,
                     MIN, MAX)

where:
  * ``dispersion = (intrinsic_high - intrinsic_low) / anchor`` when ``anchor > 0``
    and both intrinsic bounds are present; otherwise ``NEUTRAL_DISPERSION``.
  * ``vol_excess`` is gated on the price-history backfill. Until that lands,
    ``realized_volatility`` is ``None`` in production, so the volatility term
    contributes nothing (``w_vol = 0.0`` and the term is forced to 0 when vol
    is ``None``). The seam is wired so the term activates automatically once a
    volatility distribution exists.

Public API:
  * ``compute_buy_discount(...) -> BuyDiscountResult``
  * ``compute_buy_target(anchor, **kwargs) -> float | None``

Tuning surface: all constants live in ``DISCOUNT_CONFIG`` (a frozen
``DiscountConfig`` dataclass) so the operator can adjust weights/bands without a
code change. The defaults are chosen so that:
  * (ACTIONABLE, HIGH, 10% intrinsic range) -> exactly 0.12
  * (WATCHLIST_ONLY, LOW, 100% intrinsic range) -> exactly 0.40
  * (WATCHLIST_ONLY, MODERATE, no range) -> exactly 0.25 (legacy-equivalent),
    so the ~137 existing range-less watchlist rows do not shift on migration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Base discount per (conviction_grade, confidence). The grade is the dominant
#: axis (ACTIONABLE names earn a tighter haircut because the tool is most
#: confident in them and should not stay silent on exactly those names);
#: confidence tightens/loosens within a grade. Any grade/confidence not listed
#: falls back to ``DEFAULT_BASE`` (the neutral WATCHLIST_ONLY base).
#:
#: Chosen so the two anchored literals hold exactly with w_disp = 0.20:
#:   base[ACTIONABLE, HIGH] = 0.10  ->  0.10 + 0.20*0.10 = 0.12
#:   base[WATCHLIST_ONLY, *] = 0.20 ->  0.20 + 0.20*1.00 = 0.40
#:                                   ->  0.20 + 0.20*0.25 = 0.25 (neutral)
_DEFAULT_BASE_TABLE: dict[tuple[str, str], float] = {
    ("ACTIONABLE", "HIGH"): 0.10,
    ("ACTIONABLE", "MODERATE"): 0.12,
    ("ACTIONABLE", "LOW"): 0.15,
    ("WATCHLIST_ONLY", "HIGH"): 0.18,
    ("WATCHLIST_ONLY", "MODERATE"): 0.20,
    ("WATCHLIST_ONLY", "LOW"): 0.20,
    ("AVOID", "HIGH"): 0.30,
    ("AVOID", "MODERATE"): 0.30,
    ("AVOID", "LOW"): 0.30,
}


@dataclass(frozen=True)
class DiscountConfig:
    """Tunable constants for the discount function (single source of truth)."""

    #: Base discount lookup keyed by (grade, confidence). Copied so the frozen
    #: default below cannot be mutated by callers.
    base_table: dict[tuple[str, str], float] = field(
        default_factory=lambda: dict(_DEFAULT_BASE_TABLE)
    )
    #: Fallback base when (grade, confidence) is unknown — neutral WATCHLIST.
    default_base: float = 0.20
    #: Weight on intrinsic-method dispersion (range / anchor).
    w_disp: float = 0.20
    #: Weight on realized-volatility excess. ZERO until the price-history backfill lands.
    w_vol: float = 0.0
    #: Dispersion used when the intrinsic range or anchor is unavailable.
    #: Set so a WATCHLIST_ONLY/MODERATE/no-range name lands at the legacy 0.25:
    #:   0.20 (default_base) + 0.20 (w_disp) * 0.25 = 0.25
    neutral_dispersion: float = 0.25
    #: Reference volatility above which vol_excess becomes positive. The seam is
    #: wired but inert (w_vol = 0.0) until a real vol distribution exists.
    neutral_volatility: float = 0.0
    #: Band bounds — every output is clamped into [min_discount, max_discount].
    min_discount: float = 0.10
    max_discount: float = 0.45

    def base_for(self, grade: str | None, confidence: str | None) -> float:
        key = (str(grade or "").upper(), str(confidence or "").upper())
        return self.base_table.get(key, self.default_base)


#: Module-level default config. Callers may pass a custom ``defaults=`` to tune.
DISCOUNT_CONFIG = DiscountConfig()


@dataclass
class BuyDiscountResult:
    """Result of :func:`compute_buy_discount`.

    Mirrors ``ConvictionResult`` — carries the chosen value plus a
    human-readable component breakdown for auditability.
    """

    discount: float          # final clamped discount in [MIN, MAX]
    clamped: bool            # True iff the raw value was outside the band
    breakdown: dict          # component contributions, for logging/audit


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _dispersion(
    intrinsic_low: float | None,
    intrinsic_high: float | None,
    anchor: float | None,
    config: DiscountConfig,
) -> tuple[float, bool]:
    """Return (dispersion, is_neutral).

    dispersion = (high - low) / anchor when anchor > 0 and both bounds present and the
    range is not inverted; otherwise the configured NEUTRAL_DISPERSION.

    An inverted range (low > high) or a non-finite bound is bad data, not a tight
    valuation: it used to yield a NEGATIVE dispersion that shaved the discount below the
    grade's own base and raised the buy target. It is treated as unavailable.
    """
    if (
        intrinsic_low is not None
        and intrinsic_high is not None
        and anchor is not None
        and anchor > 0
        and math.isfinite(intrinsic_low)
        and math.isfinite(intrinsic_high)
        and math.isfinite(anchor)
        and intrinsic_high >= intrinsic_low
    ):
        return (intrinsic_high - intrinsic_low) / anchor, False
    return config.neutral_dispersion, True


def _volatility_excess(
    realized_volatility: float | None,
    config: DiscountConfig,
) -> float:
    """Return vol_excess = max(0, realized_volatility - neutral_volatility).

    Returns 0.0 when volatility is unavailable (None) — that gate. Combined
    with ``w_vol = 0.0`` this makes the volatility term inert until price
    history exists, after which raising ``w_vol`` activates it with no further
    code change.
    """
    if realized_volatility is None:
        return 0.0
    return max(0.0, realized_volatility - config.neutral_volatility)


def compute_buy_discount(
    *,
    conviction_grade: str | None,
    confidence: str | None,
    intrinsic_low: float | None,
    intrinsic_high: float | None,
    anchor: float | None,
    realized_volatility: float | None,
    defaults: DiscountConfig = DISCOUNT_CONFIG,
) -> BuyDiscountResult:
    """Compute the per-name buy discount, bounded into the configured band.

    Pure: no DB, no file I/O, no LLM. All inputs are scalars.
    """
    config = defaults

    base = config.base_for(conviction_grade, confidence)
    dispersion, dispersion_is_neutral = _dispersion(
        intrinsic_low, intrinsic_high, anchor, config
    )
    vol_excess = _volatility_excess(realized_volatility, config)

    dispersion_term = config.w_disp * dispersion
    volatility_term = config.w_vol * vol_excess

    raw = base + dispersion_term + volatility_term

    clamped_value = min(max(raw, config.min_discount), config.max_discount)
    clamped = clamped_value != raw

    breakdown = {
        "base_grade": base,
        "dispersion": dispersion,
        "dispersion_is_neutral": dispersion_is_neutral,
        "dispersion_term": dispersion_term,
        "vol_excess": vol_excess,
        "volatility_term": volatility_term,
        "raw": raw,
        "clamped": clamped,
        "min_discount": config.min_discount,
        "max_discount": config.max_discount,
    }

    return BuyDiscountResult(
        discount=clamped_value,
        clamped=clamped,
        breakdown=breakdown,
    )


def compute_buy_target(
    anchor: float | None,
    *,
    conviction_grade: str | None = None,
    confidence: str | None = None,
    intrinsic_low: float | None = None,
    intrinsic_high: float | None = None,
    realized_volatility: float | None = None,
    defaults: DiscountConfig = DISCOUNT_CONFIG,
) -> float | None:
    """Return ``anchor * (1 - discount)`` or ``None`` when anchor is unavailable."""
    if anchor is None:
        return None
    result = compute_buy_discount(
        conviction_grade=conviction_grade,
        confidence=confidence,
        intrinsic_low=intrinsic_low,
        intrinsic_high=intrinsic_high,
        anchor=anchor,
        realized_volatility=realized_volatility,
        defaults=defaults,
    )
    return anchor * (1.0 - result.discount)
