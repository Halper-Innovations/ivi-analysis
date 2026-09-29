"""Tests for app.evidence.enrichment quality assessment wiring."""

from __future__ import annotations


def _make_packet_with_scorecard(**scorecard_overrides) -> dict:
    """Build a minimal evidence packet with a scorecard."""
    scorecard_outputs = {
        "signal": "PASS",
        "signal_context": "OVERVALUED",
        "quality_context": {
            "gate_action": "PROCEED",
            "confidence_class": "MODERATE",
            "gate_reason_codes": [],
            "valuation_headwinds": ["SBC_ACCELERATING_HEADWIND"],
            "valuation_supports": ["GROWING_REVENUE_SUPPORT", "STRONG_EARNINGS_SUPPORT"],
            "nonrecurring_detection": {"nonrecurring_flags": []},
            "sbc_trajectory": {"sbc_flags": ["SBC_ACCELERATING"]},
            "depreciation_audit": {"depreciation_flags": []},
        },
        "moat_strength": {
            "moat_class": "MODERATE_MOAT",
            "moat_score": 5,
        },
        "downside_scenario": {
            "downside_risk_class": "LIMITED",
            "bear_case_intrinsic": 50.94,
        },
    }
    scorecard_outputs.update(scorecard_overrides)
    return {
        "valuations": {
            "scorecard": {"outputs": scorecard_outputs},
            "owner_earnings": {"outputs": {}},
        },
    }


def test_enrichment_includes_quality_assessment():
    from app.evidence.enrichment import enrich_evidence_packet
    packet = _make_packet_with_scorecard()
    enriched = enrich_evidence_packet(packet)
    qa = enriched.get("enrichment", {}).get("quality_assessment", {})
    assert qa["gate_verdict"] == "PROCEED"
    assert qa["confidence_class"] == "MODERATE"
    assert qa["moat_class"] == "MODERATE_MOAT"
    assert qa["moat_score"] == 5
    assert qa["signal_context"] == "OVERVALUED"
    assert qa["downside_risk_class"] == "LIMITED"
    assert qa["bear_case_intrinsic"] == 50.94
    assert "SBC_ACCELERATING_HEADWIND" in qa["valuation_headwinds"]
    assert "GROWING_REVENUE_SUPPORT" in qa["valuation_supports"]
    assert "SBC_ACCELERATING" in qa["sbc_flags"]
    assert qa["nonrecurring_flags"] == []
    assert qa["depreciation_flags"] == []


def test_enrichment_quality_assessment_empty_when_no_scorecard():
    from app.evidence.enrichment import enrich_evidence_packet
    packet = {"valuations": {}}
    enriched = enrich_evidence_packet(packet)
    qa = enriched.get("enrichment", {}).get("quality_assessment", {})
    assert qa["gate_verdict"] is None
    assert qa["confidence_class"] is None
    assert qa["moat_class"] is None
    assert qa["valuation_headwinds"] == []
    assert qa["valuation_supports"] == []


def test_enrichment_quality_assessment_handles_missing_nested():
    from app.evidence.enrichment import enrich_evidence_packet
    packet = {
        "valuations": {
            "scorecard": {"outputs": {"signal": "PASS"}},
            "owner_earnings": {"outputs": {}},
        },
    }
    enriched = enrich_evidence_packet(packet)
    qa = enriched.get("enrichment", {}).get("quality_assessment", {})
    assert qa["gate_verdict"] is None
    assert qa["moat_class"] is None
    assert qa["nonrecurring_flags"] == []


def test_prompt_contains_quality_guardrails():
    """Built prompt includes quality gate guardrail instructions."""
    from app.llm.synthesis_agent import _build_prompt
    inputs = {
        "ticker": "TEST",
        "as_of_date": "2026-03-26",
        "evidence_packet": {"enrichment": {"quality_assessment": {"gate_verdict": "BLOCK"}}},
        "analysis_context": {"status": "MISSING"},
        "delta_context": {},
    }
    prompt = _build_prompt(inputs, {})
    assert "gate_verdict is BLOCK" in prompt
    assert "VALUE_TRAP_RISK" in prompt
    assert "PREMIUM_JUSTIFIED" in prompt
    assert "downside_risk_class is SEVERE" in prompt


def test_compact_evidence_passes_quality_assessment():
    """_compact_evidence output includes enrichment.quality_assessment."""
    from app.llm.synthesis_agent import _compact_evidence
    packet = _make_packet_with_scorecard()
    compact = _compact_evidence(packet)
    qa = compact.get("enrichment", {}).get("quality_assessment", {})
    assert qa.get("gate_verdict") == "PROCEED"
    assert qa.get("moat_class") == "MODERATE_MOAT"
