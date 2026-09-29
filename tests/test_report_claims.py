"""Tests for app.report.claims — numeric claim extraction guards."""
from __future__ import annotations

from app.report.claims import build_numeric_claims


def _claim_ids(claims):
    return {c["claim_id"] for c in claims}


def test_reverse_dcf_claim_emitted_for_interior_solve():
    packet = {
        "fundamentals": {},
        "financials": [],
        "valuations": {
            "reverse_dcf": {"outputs": {"implied_growth": 0.07, "implied_growth_saturated": False}},
        },
    }
    claims = build_numeric_claims(packet)
    assert "valuation_reverse_dcf_implied_growth" in _claim_ids(claims)


def test_reverse_dcf_claim_skipped_when_saturated_top_level():
    """Review RDCF-2: the memo claim must not present a clipped bound as a
    point estimate — post-fix, saturated-HIGH negative-margin names would
    report implied growth -25% (deep-value-looking) for exactly the
    unsolvable money-loser population."""
    packet = {
        "fundamentals": {},
        "financials": [],
        "valuations": {
            "reverse_dcf": {"outputs": {"implied_growth": -0.25, "implied_growth_saturated": True}},
        },
    }
    claims = build_numeric_claims(packet)
    assert "valuation_reverse_dcf_implied_growth" not in _claim_ids(claims)


def test_reverse_dcf_claim_skipped_when_saturated_nested():
    """The double-nested legacy shape (outputs.outputs) must honor the flag
    at the same level the value is read from."""
    packet = {
        "fundamentals": {},
        "financials": [],
        "valuations": {
            "reverse_dcf": {
                "outputs": {
                    "outputs": {"implied_growth": -0.25, "implied_growth_saturated": True},
                }
            },
        },
    }
    claims = build_numeric_claims(packet)
    assert "valuation_reverse_dcf_implied_growth" not in _claim_ids(claims)
