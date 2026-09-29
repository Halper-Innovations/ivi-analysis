"""Decline-cap anchor policy (Part-0c): a decline-class name must not anchor
on a growth-bearing basis — the shared select_anchor caps at the no-growth
(EPV-class) basis, falling to the most conservative positive anchor-eligible
method when EPV is uncomputable/anomalous. Lenses never anchor, so they never
cap. The conviction-widened MoS stacks on top of the capped anchor downstream.
"""
from __future__ import annotations

from app.valuation.anchor_policy import is_decline_class, select_anchor


def test_decline_dcf_high_capped_to_epv():
    """Decline + DCF-high fixture: the growth-bearing DCF (100) must not set
    the anchor for a shrinking business; capped at EPV (60)."""
    s = select_anchor(dcf=100.0, epv=60.0, decline_class=True)
    assert s.value == 60.0
    assert s.method == "epv"
    assert s.reason == "DECLINE_CAPPED_NO_GROWTH"


def test_no_decline_keeps_max_positive_rule():
    s = select_anchor(dcf=100.0, epv=60.0, decline_class=False)
    assert s.value == 100.0
    assert s.method == "dcf"
    assert s.reason == "MAX_POSITIVE_DCF_EPV"


def test_decline_epv_already_selected_unchanged():
    """EPV >= DCF: the selection IS the no-growth basis — cap is a no-op and
    provenance keeps the canonical reason."""
    s = select_anchor(dcf=60.0, epv=100.0, decline_class=True)
    assert s.value == 100.0
    assert s.method == "epv"
    assert s.reason == "MAX_POSITIVE_DCF_EPV"


def test_cyclical_trough_is_not_a_decline_class():
    """Cyclical-trough fixture: VOLATILE (cyclical) is handled by the EPV
    OI-median normalization, not by capping — the predicate excludes it."""
    assert is_decline_class("VOLATILE") is False
    assert is_decline_class("GROWING") is False
    assert is_decline_class("FLAT") is False
    assert is_decline_class("UNKNOWN") is False
    assert is_decline_class(None) is False
    assert is_decline_class("DECLINING") is True
    assert is_decline_class("SECULAR_DECLINE") is True


def test_decline_epv_uncomputable_falls_to_most_conservative_eligible():
    """EPV-uncomputable fixture: negative EPV (anomalous) cannot cap; the cap
    falls to the most conservative POSITIVE anchor-eligible method."""
    s = select_anchor(dcf=100.0, epv=-5.0, graham=40.0, ncav=25.0, decline_class=True)
    assert s.value == 25.0
    assert s.method == "ncav"
    assert s.reason == "DECLINE_CAPPED_NO_GROWTH"


def test_decline_lenses_never_cap():
    """Provenance lenses are not anchor-eligible: a tangible_floor below every
    method must not become the cap."""
    s = select_anchor(
        dcf=100.0,
        epv=None,
        graham=40.0,
        decline_class=True,
        extra_candidates={"tangible_floor": 10.0, "ev_ebit": 15.0},
    )
    assert s.value == 40.0
    assert s.method == "graham"
    assert s.reason == "DECLINE_CAPPED_NO_GROWTH"


def test_decline_caps_sector_specific_anchor_too():
    """The policy applies to the final selection regardless of branch — a
    sector-specific anchor on a decline-class name is capped the same way."""
    s = select_anchor(
        dcf=50.0,
        epv=60.0,
        sector_specific=("technology_adjusted_dcf", 120.0),
        decline_class=True,
    )
    assert s.value == 60.0
    assert s.method == "epv"
    assert s.reason == "DECLINE_CAPPED_NO_GROWTH"


def test_decline_no_positive_methods_stays_none():
    s = select_anchor(dcf=-10.0, epv=-5.0, decline_class=True)
    assert s.value is None
    assert s.reason == "NO_POSITIVE_ANCHOR"
