from __future__ import annotations

from app.autonomous.sector_contract import SectorCompanyFinancialPacket
from app.autonomous.semantic_consistency import (
    grade_narrative_mismatch,
    missing_data_claims,
    recommended_verdicts,
    ungrounded_missing_data_claims,
)


def _packet(**overrides) -> SectorCompanyFinancialPacket:
    fields = {
        "ticker": "AAA",
        "financial_status": "OK",
        "model_fit_status": "OK",
        "data_quality_status": "OK",
        "current_price": 50.0,
        "valuation": {"anchor_method": "DCF", "valuation_anchor": 80.0},
    }
    fields.update(overrides)
    return SectorCompanyFinancialPacket(**fields)


def test_explicit_recommendation_is_parsed():
    assert recommended_verdicts("The correct verdict is WATCHLIST_ONLY here.") == [
        "WATCHLIST_ONLY"
    ]
    assert recommended_verdicts("The recommendation is avoid.") == ["AVOID"]


def test_watchlist_argument_phrasing_is_parsed():
    thesis = (
        "Returns sit below the sector's hurdle, which caps conviction and "
        "justifies watchlist positioning until margins recover."
    )
    assert recommended_verdicts(thesis) == ["WATCHLIST_ONLY"]


def test_grade_narrative_mismatch_actionable_vs_watchlist_argument():
    thesis = "The weak hurdle spread justifies watchlist positioning for now."
    assert (
        grade_narrative_mismatch(thesis, "ACTIONABLE")
        == "thesis argues WATCHLIST_ONLY but structured grade is ACTIONABLE"
    )
    assert grade_narrative_mismatch(thesis, "WATCHLIST_ONLY") is None


def test_grade_narrative_mismatch_avoid():
    thesis = "Evidence is negative and the final decision should be avoid."
    assert grade_narrative_mismatch(thesis, "WATCHLIST_ONLY") == (
        "thesis argues AVOID but structured grade is WATCHLIST_ONLY"
    )
    assert grade_narrative_mismatch(thesis, "AVOID") is None


def test_no_mismatch_without_explicit_recommendation():
    thesis = "A durable franchise trading below intrinsic value with clean data."
    assert grade_narrative_mismatch(thesis, "ACTIONABLE") is None


def test_missing_data_claims_detected():
    thesis = (
        "The case is clouded by missing cash-flow and working-capital inputs. "
        "Margins are otherwise stable."
    )
    claims = missing_data_claims(thesis)
    assert len(claims) == 1
    assert "missing cash-flow" in claims[0]


def test_missing_data_claim_ungrounded_when_packet_is_clean():
    thesis = "Valuation is hampered by missing cash-flow inputs."
    assert ungrounded_missing_data_claims(thesis, _packet()) != []


def test_missing_data_claim_grounded_by_matching_code():
    thesis = "Valuation is hampered by missing cash-flow inputs."
    packet = _packet(
        cash_conversion={"not_computable_reasons": ["CFO_MISSING"]},
    )
    assert ungrounded_missing_data_claims(thesis, packet) == []


def test_specific_claim_not_grounded_by_unrelated_gap():
    # A no-filing blocker must not license a false missing-cash-flow claim.
    thesis = "The case is clouded by missing cash-flow and working-capital inputs."
    packet = _packet(blockers=["NO_READABLE_ANNUAL_FILING"])
    assert ungrounded_missing_data_claims(thesis, packet) != []


def test_filing_claim_grounded_by_filing_gap():
    thesis = "Analysis is limited because narrative filings are unavailable."
    packet = _packet(blockers=["NO_READABLE_ANNUAL_FILING"])
    assert ungrounded_missing_data_claims(thesis, packet) == []


def test_generic_claim_grounded_by_any_gap():
    thesis = "Several inputs are missing, so scoring confidence is capped."
    packet = _packet(financial_status="DEGRADED")
    assert ungrounded_missing_data_claims(thesis, packet) == []
