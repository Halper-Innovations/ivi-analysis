"""Tests for the pure per-name margin-of-safety discount function.

All assertions use EXACT LITERAL values, hand-computed from the documented
formula, NOT recomputed from the function under test:

    discount = clamp(base[grade,confidence] + w_disp*dispersion + w_vol*vol_excess,
                     MIN=0.10, MAX=0.45)
    dispersion = (intrinsic_high - intrinsic_low) / anchor   (when anchor>0 & both present)
               = NEUTRAL_DISPERSION                          (otherwise)

Plan defaults: MIN=0.10, MAX=0.45, w_disp=0.20, w_vol=0.0,
base[ACTIONABLE,HIGH]=0.10, base[WATCHLIST_ONLY,*]=0.20, base[AVOID]=0.30,
NEUTRAL_DISPERSION=0.25 (so WATCHLIST_ONLY/MODERATE/no-range -> 0.20+0.20*0.25=0.25).
"""
from __future__ import annotations

from app.watchlist.margin_of_safety import (
    compute_buy_discount,
    compute_buy_target,
)


def test_actionable_high_low_dispersion_yields_012():
    # base 0.10 + 0.20 * (105-95)/100=0.10 -> 0.10 + 0.02 = 0.12
    result = compute_buy_discount(
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        intrinsic_low=95.0,
        intrinsic_high=105.0,
        anchor=100.0,
        realized_volatility=None,
    )
    assert round(result.discount, 4) == 0.12
    assert result.clamped is False


def test_watchlist_low_wide_dispersion_yields_040():
    # base 0.20 + 0.20 * (120-40)/80=1.0 -> 0.20 + 0.20 = 0.40
    result = compute_buy_discount(
        conviction_grade="WATCHLIST_ONLY",
        confidence="LOW",
        intrinsic_low=40.0,
        intrinsic_high=120.0,
        anchor=80.0,
        realized_volatility=None,
    )
    assert round(result.discount, 4) == 0.40
    assert result.clamped is False


def test_band_clamp_ceiling_045():
    # AVOID base 0.30 + 0.20 * (1000-1)/1=999*... -> far above 0.45 -> clamps to 0.45
    result = compute_buy_discount(
        conviction_grade="AVOID",
        confidence="LOW",
        intrinsic_low=1.0,
        intrinsic_high=1000.0,
        anchor=1.0,
        realized_volatility=None,
    )
    assert round(result.discount, 4) == 0.45
    assert result.clamped is True


def test_band_clamp_floor_010():
    # ACTIONABLE+HIGH base 0.10 + 0.20 * 0 dispersion = 0.10 -> floor 0.10
    result = compute_buy_discount(
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        intrinsic_low=100.0,
        intrinsic_high=100.0,
        anchor=100.0,
        realized_volatility=None,
    )
    assert round(result.discount, 4) == 0.10
    # 0.10 is the raw value AND the floor; it is not produced by clamping.
    assert result.clamped is False


def test_compute_buy_target_actionable_high():
    # 100 * (1 - 0.12) = 88.0
    target = compute_buy_target(
        anchor=100.0,
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        intrinsic_low=95.0,
        intrinsic_high=105.0,
        realized_volatility=None,
    )
    assert round(target, 4) == 88.0


def test_compute_buy_target_none_anchor_returns_none():
    target = compute_buy_target(
        anchor=None,
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        intrinsic_low=95.0,
        intrinsic_high=105.0,
        realized_volatility=None,
    )
    assert target is None


def test_neutral_fallback_no_range_reproduces_legacy_025():
    # WATCHLIST_ONLY/MODERATE/no-range -> base 0.20 + 0.20 * NEUTRAL_DISPERSION(0.25) = 0.25
    result = compute_buy_discount(
        conviction_grade="WATCHLIST_ONLY",
        confidence="MODERATE",
        intrinsic_low=None,
        intrinsic_high=None,
        anchor=106.67,
        realized_volatility=None,
    )
    assert round(result.discount, 4) == 0.25
    assert result.clamped is False


def test_breakdown_keys_present():
    result = compute_buy_discount(
        conviction_grade="ACTIONABLE",
        confidence="HIGH",
        intrinsic_low=95.0,
        intrinsic_high=105.0,
        anchor=100.0,
        realized_volatility=None,
    )
    assert isinstance(result.breakdown, dict)
    for key in ("base_grade", "dispersion_term", "volatility_term", "clamped"):
        assert key in result.breakdown
