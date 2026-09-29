"""Tests for app.research.hypothesis_generator."""
from __future__ import annotations

from app.alpha.schemas import Anomaly


def test_growth_vs_earnings_power_hypothesis():
    """GROWTH_VS_EARNINGS_POWER tension with high growth dependency should produce BEARISH hypothesis."""
    from app.research.hypothesis_generator import generate_hypotheses, EvidenceNeed

    tensions = {
        "tension_type": "GROWTH_VS_EARNINGS_POWER",
        "method_count": 3,
        "consensus_direction": "MIXED",
        "undervalued_count": 1,
        "overvalued_count": 1,
        "growth_value_pct": 0.81,
        "sensitivity": {"growth_dependency_ratio": 0.81, "growth_value_per_share": 171.0, "earnings_power_per_share": 39.0},
        "assumption_sensitivity": {"dcf_to_growth": "HIGH — 1pp growth change ≈ $24.71/share DCF impact"},
        "method_values": {"dcf": 210.0, "epv": 39.0, "graham": 50.0},
        "intrinsic_range": {"low": 39.0, "mid": 99.7, "high": 210.0},
    }
    valuation = {"pricing_zone_detail": {"current_price": 190.0, "dcf_base": 210.0}}

    result = generate_hypotheses(
        ticker="TEST", valuation=valuation, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    assert len(result) >= 1
    growth_hyp = next(h for h in result if h.source == "GROWTH_VS_EARNINGS_POWER")
    assert growth_hyp.direction == "BEARISH"
    assert growth_hyp.priority == "HIGH"
    assert "$210" in growth_hyp.claim
    assert "$39" in growth_hyp.claim
    assert growth_hyp.impact_estimate == 171.0
    assert len(growth_hyp.evidence_needed) >= 3
    assert all(isinstance(n, EvidenceNeed) for n in growth_hyp.evidence_needed)
    assert any(n.importance == "REQUIRED" for n in growth_hyp.evidence_needed)
    assert len(growth_hyp.falsification) > 0


def test_asset_vs_earnings_hypothesis():
    """ASSET_VS_EARNINGS tension should produce hypothesis about liquidation value."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {
        "tension_type": "ASSET_VS_EARNINGS",
        "method_count": 3,
        "consensus_direction": "MIXED",
        "undervalued_count": 1,
        "overvalued_count": 1,
        "growth_value_pct": 0,
        "sensitivity": {},
        "method_values": {"dcf": 5.0, "epv": 3.0, "ncav": 12.0},
        "intrinsic_range": {"low": 3.0, "mid": 6.7, "high": 12.0},
    }
    valuation = {"pricing_zone_detail": {"current_price": 8.0}}

    result = generate_hypotheses(
        ticker="TEST", valuation=valuation, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    asset_hyp = next(h for h in result if h.source == "ASSET_VS_EARNINGS")
    assert asset_hyp.direction == "BEARISH"
    assert asset_hyp.priority == "MODERATE"
    assert "$12" in asset_hyp.claim
    assert asset_hyp.impact_estimate == 8.0


def test_unanimous_undervaluation_hypothesis():
    """All methods agree UNDERVALUED should produce BULLISH hypothesis."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {
        "tension_type": "NONE",
        "method_count": 3,
        "consensus_direction": "UNDERVALUED",
        "undervalued_count": 3,
        "overvalued_count": 0,
        "growth_value_pct": 0.2,
        "sensitivity": {},
        "method_values": {"dcf": 150.0, "epv": 120.0, "graham": 100.0},
        "intrinsic_range": {"low": 100.0, "mid": 123.3, "high": 150.0},
    }
    valuation = {"pricing_zone_detail": {"current_price": 80.0}}

    result = generate_hypotheses(
        ticker="TEST", valuation=valuation, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    bull_hyp = next(h for h in result if h.source == "UNANIMOUS_UNDERVALUATION")
    assert bull_hyp.direction == "BULLISH"
    assert bull_hyp.priority == "MODERATE"
    assert "3" in bull_hyp.claim
    assert bull_hyp.impact_estimate is None


def test_unanimous_undervaluation_requires_two_methods():
    """UNANIMOUS_UNDERVALUATION should not trigger with only 1 method."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {
        "tension_type": "INSUFFICIENT_METHODS",
        "method_count": 1,
        "consensus_direction": "UNDERVALUED",
        "undervalued_count": 1,
        "overvalued_count": 0,
        "growth_value_pct": 0,
        "sensitivity": {},
        "method_values": {"dcf": 150.0},
        "intrinsic_range": {"low": 150.0, "mid": 150.0, "high": 150.0},
    }
    valuation = {"pricing_zone_detail": {"current_price": 80.0}}

    result = generate_hypotheses(
        ticker="TEST", valuation=valuation, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    sources = [h.source for h in result]
    assert "UNANIMOUS_UNDERVALUATION" not in sources


def test_tier1_skipped_for_insufficient_methods():
    """Tier 1 should produce no hypotheses when tension_type is INSUFFICIENT_METHODS."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {
        "tension_type": "INSUFFICIENT_METHODS",
        "method_count": 1,
        "consensus_direction": "UNKNOWN",
        "undervalued_count": 0,
        "overvalued_count": 0,
        "growth_value_pct": 0,
        "sensitivity": {},
        "method_values": {"dcf": 100.0},
        "intrinsic_range": {"low": 100.0, "mid": 100.0, "high": 100.0},
    }
    valuation = {"pricing_zone_detail": {"current_price": 90.0}}

    result = generate_hypotheses(
        ticker="TEST", valuation=valuation, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    tier1_sources = {"GROWTH_VS_EARNINGS_POWER", "ASSET_VS_EARNINGS", "UNANIMOUS_UNDERVALUATION", "MARKET_PREMIUM"}
    assert not any(h.source in tier1_sources for h in result)


def test_q4_earnings_bomb_hypothesis():
    """Q4_EARNINGS_BOMB anomaly should produce HIGH-priority BEARISH hypothesis."""
    from app.research.hypothesis_generator import generate_hypotheses

    anomalies = [Anomaly(
        anomaly_type="Q4_EARNINGS_BOMB",
        severity="HIGH",
        description="Q1-Q3 net income totaled $500.0M, but FY was $200.0M — implying Q4 was $-300.0M.",
        question="What caused the $300M Q4 loss?",
        data={"fy_ni": 200.0, "q123_total": 500.0, "q4_implied": -300.0},
    )]

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    result = generate_hypotheses(
        ticker="TEST", valuation={}, tensions=tensions,
        anomalies=anomalies, quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    bomb_hyp = next(h for h in result if h.source == "Q4_EARNINGS_BOMB")
    assert bomb_hyp.direction == "BEARISH"
    assert bomb_hyp.priority == "HIGH"
    assert "$-300.0M" in bomb_hyp.claim or "Q4" in bomb_hyp.claim
    assert bomb_hyp.impact_estimate == -300.0
    descs = [n.description for n in bomb_hyp.evidence_needed]
    assert "restructuring" in " ".join(descs).lower() or "writedown" in " ".join(descs).lower()


def test_margin_collapse_hypothesis():
    """MARGIN_COLLAPSE should include impact estimate."""
    from app.research.hypothesis_generator import generate_hypotheses

    anomalies = [Anomaly(
        anomaly_type="MARGIN_COLLAPSE",
        severity="HIGH",
        description="Operating margin fell from 25.0% to 10.0% — a 15.0pp decline.",
        question="What is causing margin compression?",
        data={"newest": 10.0, "oldest": 25.0, "decline_pp": 15.0},
    )]

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    result = generate_hypotheses(
        ticker="TEST", valuation={}, tensions=tensions,
        anomalies=anomalies, quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    margin_hyp = next(h for h in result if h.source == "MARGIN_COLLAPSE")
    assert margin_hyp.direction == "BEARISH"
    assert margin_hyp.priority == "HIGH"
    assert "15.0" in margin_hyp.claim or "25.0" in margin_hyp.claim or "10.0" in margin_hyp.claim


def test_all_8_anomaly_types_have_templates():
    """Every anomaly type from anomaly_detector should produce a hypothesis."""
    from app.research.hypothesis_generator import generate_hypotheses

    all_types = [
        "Q4_EARNINGS_BOMB", "NEGATIVE_EQUITY", "MARGIN_COLLAPSE",
        "INTANGIBLE_ASSET_JUMP", "REVENUE_DECLINE_FROM_PEAK",
        "PERSISTENT_CASH_BURN", "WORKING_CAPITAL_CRISIS", "DEBT_SPIKE",
    ]

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    for atype in all_types:
        anomalies = [Anomaly(
            anomaly_type=atype, severity="HIGH",
            description=f"Test anomaly of type {atype}.",
            question=f"Test question for {atype}?",
            data={},
        )]
        result = generate_hypotheses(
            ticker="TEST", valuation={}, tensions=tensions,
            anomalies=anomalies, quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
        )
        sources = [h.source for h in result]
        assert atype in sources, f"No hypothesis generated for anomaly type {atype}"


def test_competitive_disruption_hypothesis():
    """HIGH competitive_disruption in filing risk should produce BEARISH hypothesis."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    filing_risk = {"competitive_disruption": "HIGH", "secular_decline": "LOW"}

    result = generate_hypotheses(
        ticker="TEST", valuation={}, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=filing_risk,
    )

    cd_hyp = next(h for h in result if h.source == "COMPETITIVE_DISRUPTION")
    assert cd_hyp.direction == "BEARISH"
    assert cd_hyp.priority == "HIGH"
    assert cd_hyp.impact_estimate is None


def test_proceed_severe_downside_hypothesis():
    """PROCEED gate + SEVERE downside should produce hypothesis about tail risk."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    quality_ctx = {"gate_action": "PROCEED", "valuation_headwinds": [], "valuation_supports": []}

    result = generate_hypotheses(
        ticker="TEST", valuation={"pricing_zone_detail": {"downside_risk_class": "SEVERE"}},
        tensions=tensions, anomalies=[], quality_ctx=quality_ctx, solvency=None, filing_risk=None,
    )

    ds_hyp = next(h for h in result if h.source == "PROCEED_SEVERE_DOWNSIDE")
    assert ds_hyp.direction == "BEARISH"
    assert ds_hyp.priority == "MODERATE"


def test_negative_owner_earnings_hypothesis():
    """Negative owner earnings headwind should produce hypothesis."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    quality_ctx = {
        "gate_action": "BLOCK",
        "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
        "valuation_headwinds": [],
    }

    result = generate_hypotheses(
        ticker="TEST", valuation={}, tensions=tensions,
        anomalies=[], quality_ctx=quality_ctx, solvency=None, filing_risk=None,
    )

    oe_hyp = next(h for h in result if h.source == "NEGATIVE_OWNER_EARNINGS")
    assert oe_hyp.direction == "BEARISH"
    assert oe_hyp.priority == "HIGH"


def test_no_filing_risk_skips_tier3_filing():
    """When filing_risk is None, filing-dependent templates should be skipped."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    result = generate_hypotheses(
        ticker="TEST", valuation={}, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    filing_sources = {"COMPETITIVE_DISRUPTION", "SECULAR_DECLINE"}
    assert not any(h.source in filing_sources for h in result)


def test_output_sorted_by_priority_then_direction():
    """Hypotheses should be sorted: HIGH > MODERATE > LOW, BEARISH > BULLISH > NEUTRAL."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {
        "tension_type": "GROWTH_VS_EARNINGS_POWER",
        "method_count": 3,
        "consensus_direction": "UNDERVALUED",
        "undervalued_count": 3,
        "overvalued_count": 0,
        "growth_value_pct": 0.7,
        "sensitivity": {"growth_dependency_ratio": 0.7},
        "method_values": {"dcf": 200.0, "epv": 60.0, "graham": 80.0},
        "intrinsic_range": {"low": 60.0, "mid": 113.3, "high": 200.0},
    }
    valuation = {"pricing_zone_detail": {"current_price": 150.0, "dcf_base": 200.0}}

    anomalies = [Anomaly(
        anomaly_type="INTANGIBLE_ASSET_JUMP", severity="MODERATE",
        description="Test.", question="Test?", data={},
    )]

    result = generate_hypotheses(
        ticker="TEST", valuation=valuation, tensions=tensions,
        anomalies=anomalies, quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    assert len(result) >= 3
    priorities = [h.priority for h in result]
    first_high = next(i for i, p in enumerate(priorities) if p == "HIGH")
    first_low = next(i for i, p in enumerate(priorities) if p == "LOW")
    assert first_high < first_low


def test_empty_inputs_returns_empty():
    """No tensions, anomalies, or signals should return empty list."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    result = generate_hypotheses(
        ticker="TEST", valuation={}, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    assert result == []


def test_dedup_merges_evidence_by_source():
    """If two hypotheses share the same source, keep higher priority and merge evidence."""
    from app.research.hypothesis_generator import _deduplicate, Hypothesis, EvidenceNeed

    h1 = Hypothesis(
        claim="Claim A", direction="BEARISH", priority="HIGH", source="SAME_SOURCE",
        evidence_needed=[
            EvidenceNeed("s_001_a", "evidence_a", "REQUIRED"),
            EvidenceNeed("s_002_b", "evidence_b", "IMPORTANT"),
        ],
        falsification="F1", impact_estimate=10.0,
    )
    h2 = Hypothesis(
        claim="Claim B", direction="BEARISH", priority="MODERATE", source="SAME_SOURCE",
        evidence_needed=[
            EvidenceNeed("s_003_b", "evidence_b", "REQUIRED"),
            EvidenceNeed("s_004_c", "evidence_c", "IMPORTANT"),
        ],
        falsification="F2", impact_estimate=5.0,
    )

    result = _deduplicate([h1, h2])
    assert len(result) == 1
    assert result[0].priority == "HIGH"
    assert result[0].claim == "Claim A"
    descs = [n.description for n in result[0].evidence_needed]
    assert "evidence_c" in descs
    assert descs.count("evidence_b") == 1
    # evidence_b should have been upgraded to REQUIRED (from h2)
    ev_b = next(n for n in result[0].evidence_needed if n.description == "evidence_b")
    assert ev_b.importance == "REQUIRED"


def test_blocked_gate_still_generates_hypotheses():
    """BLOCKED tickers should still get hypotheses — caller decides whether to invoke."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {"tension_type": "NONE", "method_count": 2, "consensus_direction": "MIXED",
                "undervalued_count": 0, "overvalued_count": 0, "growth_value_pct": 0,
                "sensitivity": {}, "method_values": {}, "intrinsic_range": {}}

    quality_ctx = {"gate_action": "BLOCK", "gate_reason_codes": ["ZERO_OWNER_EARNINGS"]}

    result = generate_hypotheses(
        ticker="TEST", valuation={}, tensions=tensions,
        anomalies=[], quality_ctx=quality_ctx, solvency=None, filing_risk=None,
    )

    assert any(h.source == "NEGATIVE_OWNER_EARNINGS" for h in result)


# ── calibration_context ──────────────────────────────────────────────────────

def test_growth_tension_has_calibration_context():
    """GROWTH_VS_EARNINGS_POWER should include calibration_context with growth_dependency_ratio and dcf_epv_gap."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {
        "tension_type": "GROWTH_VS_EARNINGS_POWER",
        "method_count": 3,
        "consensus_direction": "MIXED",
        "undervalued_count": 1,
        "overvalued_count": 1,
        "growth_value_pct": 0.81,
        "sensitivity": {"growth_dependency_ratio": 0.81, "growth_value_per_share": 171.0, "earnings_power_per_share": 39.0},
        "assumption_sensitivity": {"dcf_to_growth": "HIGH — 1pp growth change ≈ $24.71/share DCF impact"},
        "method_values": {"dcf": 210.0, "epv": 39.0, "graham": 50.0},
        "intrinsic_range": {"low": 39.0, "mid": 99.7, "high": 210.0},
    }
    valuation = {"pricing_zone_detail": {"current_price": 190.0, "dcf_base": 210.0}}

    result = generate_hypotheses(
        ticker="TEST", valuation=valuation, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )
    growth_hyp = next(h for h in result if h.source == "GROWTH_VS_EARNINGS_POWER")
    assert growth_hyp.calibration_context is not None
    assert growth_hyp.calibration_context["growth_dependency_ratio"] == 0.81
    assert growth_hyp.calibration_context["dcf_epv_gap"] == 171.0


def test_margin_collapse_has_calibration_context():
    """MARGIN_COLLAPSE anomaly should include margin_decline_pp in calibration_context."""
    from app.research.hypothesis_generator import generate_hypotheses
    from app.alpha.schemas import Anomaly

    anomaly = Anomaly(
        anomaly_type="MARGIN_COLLAPSE",
        severity="HIGH",
        description="Operating margin fell from 25% to 13% (12pp decline).",
        question="Is this margin decline structural or temporary?",
        data={"latest_margin": 0.13, "prior_margin": 0.25, "decline_pp": 12.0},
    )
    result = generate_hypotheses(
        ticker="TEST",
        valuation={"pricing_zone_detail": {"current_price": 50.0}},
        tensions={"tension_type": "NONE", "method_values": {}},
        anomalies=[anomaly],
        quality_ctx={"gate_action": "PROCEED"},
    )
    mc_hyp = next(h for h in result if h.source == "MARGIN_COLLAPSE")
    assert mc_hyp.calibration_context is not None
    assert mc_hyp.calibration_context["margin_decline_pp"] == 12.0


def test_revenue_decline_has_calibration_context():
    """REVENUE_DECLINE_FROM_PEAK should include peak_decline_pct in calibration_context."""
    from app.research.hypothesis_generator import generate_hypotheses
    from app.alpha.schemas import Anomaly

    anomaly = Anomaly(
        anomaly_type="REVENUE_DECLINE_FROM_PEAK",
        severity="MODERATE",
        description="Revenue peaked at $1000M in FY2021, now $680M (32% decline).",
        question="Is the revenue decline structural or cyclical?",
        data={"peak_yr": 2021, "peak_rev": 1000.0, "current_rev": 680.0},
    )
    result = generate_hypotheses(
        ticker="TEST",
        valuation={"pricing_zone_detail": {"current_price": 30.0}},
        tensions={"tension_type": "NONE", "method_values": {}},
        anomalies=[anomaly],
        quality_ctx={"gate_action": "PROCEED"},
    )
    rd_hyp = next(h for h in result if h.source == "REVENUE_DECLINE_FROM_PEAK")
    assert rd_hyp.calibration_context is not None
    assert abs(rd_hyp.calibration_context["peak_decline_pct"] - 0.32) < 0.01


def test_persistent_cash_burn_has_calibration_context():
    """PERSISTENT_CASH_BURN should include burn_years in calibration_context."""
    from app.research.hypothesis_generator import generate_hypotheses
    from app.alpha.schemas import Anomaly

    anomaly = Anomaly(
        anomaly_type="PERSISTENT_CASH_BURN",
        severity="HIGH",
        description="Negative CFO for 3 of last 3 years.",
        question="How is the company funding operations?",
        data={"neg_years": 3, "cash": 500.0, "avg_burn": -200.0, "quarters_left": 10.0},
    )
    result = generate_hypotheses(
        ticker="TEST",
        valuation={"pricing_zone_detail": {"current_price": 20.0}},
        tensions={"tension_type": "NONE", "method_values": {}},
        anomalies=[anomaly],
        quality_ctx={"gate_action": "PROCEED"},
    )
    pcb_hyp = next(h for h in result if h.source == "PERSISTENT_CASH_BURN")
    assert pcb_hyp.calibration_context is not None
    assert pcb_hyp.calibration_context["burn_years"] == 3


def test_negative_owner_earnings_has_calibration_context():
    """NEGATIVE_OWNER_EARNINGS should include negative_oe_years from quality_ctx."""
    from app.research.hypothesis_generator import generate_hypotheses

    result = generate_hypotheses(
        ticker="TEST",
        valuation={"pricing_zone_detail": {"current_price": 15.0}},
        tensions={"tension_type": "NONE", "method_values": {}},
        anomalies=[],
        quality_ctx={
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
            "negative_oe_years": 4,
        },
    )
    noe_hyp = next(h for h in result if h.source == "NEGATIVE_OWNER_EARNINGS")
    assert noe_hyp.calibration_context is not None
    assert noe_hyp.calibration_context["negative_oe_years"] == 4


def test_negative_owner_earnings_missing_years_has_none_context():
    """NEGATIVE_OWNER_EARNINGS without negative_oe_years should have calibration_context=None."""
    from app.research.hypothesis_generator import generate_hypotheses

    result = generate_hypotheses(
        ticker="TEST",
        valuation={"pricing_zone_detail": {"current_price": 15.0}},
        tensions={"tension_type": "NONE", "method_values": {}},
        anomalies=[],
        quality_ctx={
            "gate_action": "BLOCK",
            "gate_reason_codes": ["ZERO_OWNER_EARNINGS"],
        },
    )
    noe_hyp = next(h for h in result if h.source == "NEGATIVE_OWNER_EARNINGS")
    assert noe_hyp.calibration_context is None


def test_dedup_carries_calibration_context_from_loser():
    """When winner has no calibration_context but loser does, carry loser's context."""
    from app.research.hypothesis_generator import _deduplicate, Hypothesis, EvidenceNeed

    winner = Hypothesis(
        claim="Winner claim", direction="BEARISH",
        evidence_needed=[EvidenceNeed("w_001_test", "test", "REQUIRED")],
        falsification="test", priority="HIGH", source="TEST_SOURCE",
        impact_estimate=10.0, calibration_context=None,
    )
    loser = Hypothesis(
        claim="Loser claim", direction="BEARISH",
        evidence_needed=[EvidenceNeed("l_001_test", "other", "IMPORTANT")],
        falsification="test", priority="MODERATE", source="TEST_SOURCE",
        impact_estimate=5.0, calibration_context={"some_fact": 42.0},
    )
    result = _deduplicate([winner, loser])
    assert len(result) == 1
    assert result[0].priority == "HIGH"
    assert result[0].calibration_context == {"some_fact": 42.0}


def test_plausible_undervaluation_fallback():
    """When DCF and EPV are above price but consensus is MIXED, generate fallback hypothesis."""
    from app.research.hypothesis_generator import generate_hypotheses

    # CCSI-like scenario: DCF and EPV above price, Graham and NCAV below
    tensions = {
        "tension_type": "NONE",
        "method_count": 4,
        "consensus_direction": "MIXED",
        "undervalued_count": 2,
        "overvalued_count": 2,
        "growth_value_pct": 0,
        "sensitivity": {"growth_dependency_ratio": -0.204},
        "method_values": {"dcf": 31.85, "epv": 38.34, "graham": 8.64, "ncav": 1.67},
        "intrinsic_range": {"low": 1.67, "mid": 20.12, "high": 38.34},
    }
    valuation = {"pricing_zone_detail": {"current_price": 23.96, "dcf_base": 31.85}}

    result = generate_hypotheses(
        ticker="CCSI", valuation=valuation, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    sources = [h.source for h in result]
    assert "PLAUSIBLE_UNDERVALUATION" in sources
    plausible = next(h for h in result if h.source == "PLAUSIBLE_UNDERVALUATION")
    assert plausible.direction == "BULLISH"
    assert plausible.priority == "HIGH"
    assert len(plausible.evidence_needed) >= 3


def test_plausible_undervaluation_skipped_when_unanimous_exists():
    """No fallback when UNANIMOUS_UNDERVALUATION already covers it."""
    from app.research.hypothesis_generator import generate_hypotheses

    tensions = {
        "tension_type": "NONE",
        "method_count": 3,
        "consensus_direction": "UNDERVALUED",
        "undervalued_count": 3,
        "overvalued_count": 0,
        "growth_value_pct": 0,
        "sensitivity": {"growth_dependency_ratio": 0.1},
        "method_values": {"dcf": 50.0, "epv": 45.0, "graham": 40.0},
        "intrinsic_range": {"low": 40.0, "mid": 45.0, "high": 50.0},
    }
    valuation = {"pricing_zone_detail": {"current_price": 30.0, "dcf_base": 50.0}}

    result = generate_hypotheses(
        ticker="TEST", valuation=valuation, tensions=tensions,
        anomalies=[], quality_ctx={"gate_action": "PROCEED"}, solvency=None, filing_risk=None,
    )

    sources = [h.source for h in result]
    assert "UNANIMOUS_UNDERVALUATION" in sources
    assert "PLAUSIBLE_UNDERVALUATION" not in sources
