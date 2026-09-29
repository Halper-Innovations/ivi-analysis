"""Unit tests for the per-share anchor sanity band.

Literal verdict/ratio assertions for the quarantine gate helper. Values are
exact literals derived from real watchlist anchors
(BKNG / BTM / CALM) against trusted trailing reference prices.
"""

from __future__ import annotations

from app.watchlist.anchor_sanity import evaluate_anchor


def test_bkng_anchor_in_band_against_true_trailing_price_ok():
    # BKNG: anchor in the 0.2-5.0x band against the TRUE trailing price ->
    # not quarantined.
    result = evaluate_anchor(anchor=4435.46, reference_price=5000.0)
    assert result.verdict == "OK"
    assert result.ratio == 0.887


def test_btm_anchor_above_band_quarantined():
    # BTM: 12.4x > 5.0x high band.
    result = evaluate_anchor(anchor=36.20, reference_price=2.93)
    assert result.verdict == "QUARANTINE_ANCHOR_ABOVE_BAND"
    assert result.ratio == 12.355


def test_calm_borderline_cyclical_passes_hard_gate():
    # CALM: 4.92x is BELOW the 5.0x band so it is OK; the 5.0x band
    # intentionally lets a borderline cyclical name through to be caught by
    # the soft 2.5x BUSINESS_QUALITY cap, not the hard gate.
    result = evaluate_anchor(anchor=370.9, reference_price=75.45)
    assert result.verdict == "OK"
    assert round(result.ratio, 3) == 4.916


def test_anchor_below_band_quarantined():
    # 0.1x < 0.2x low band.
    result = evaluate_anchor(anchor=10.0, reference_price=100.0)
    assert result.verdict == "QUARANTINE_ANCHOR_BELOW_BAND"
    assert result.ratio == 0.1


def test_no_reference_price_unassessable():
    result = evaluate_anchor(anchor=4435.46, reference_price=None)
    assert result.verdict == "UNASSESSABLE_NO_REFERENCE"
    assert result.ratio is None


def test_no_anchor_unassessable():
    result = evaluate_anchor(anchor=None, reference_price=100.0)
    assert result.verdict == "UNASSESSABLE_NO_REFERENCE"
    assert result.ratio is None
