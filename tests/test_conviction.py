"""Tests for app.valuation.conviction."""
from __future__ import annotations

from app.research.deep_research import ResearchReport
from app.research.thesis_updater import ThesisResult


def _make_report(
    *,
    status="OK",
    gate_action="PROCEED",
    investigation_ran=True,
    methods_agree=True,
    consensus_strength=2,
    method_count=2,
    tension_type="NONE",
    thesis=None,
) -> ResearchReport:
    """Build a ResearchReport with controlled fields for conviction testing."""
    return ResearchReport(
        ticker="TEST",
        as_of_date="2026-03-25",
        status=status,
        started_at="2026-03-25T00:00:00+00:00",
        completed_at="2026-03-25T00:01:00+00:00",
        scorecard_present=status != "NO_SCORECARD",
        filing_present=True,
        filing_date="2025-11-15",
        form_type="10-K",
        anomaly_count=0,
        solvency_status=None,
        filing_risk_status=None,
        gate_action=gate_action,
        investigation_ran=investigation_ran,
        hypotheses_generated=3,
        thesis=thesis,
        researchable_items=[],
        not_researchable_items=[],
        total_adjustments=0,
        fact_calibrated_count=0,
        heuristic_count=0,
        methods_agree=methods_agree,
        consensus_strength=consensus_strength,
        method_count=method_count,
        tension_type=tension_type,
    )


def _make_thesis(
    *,
    confirmed=2,
    contradicted=1,
    partially=1,
    inconclusive=1,
    unclassified=0,
    average_coverage=0.8,
) -> ThesisResult:
    """Build a ThesisResult with controlled hypothesis counts."""
    return ThesisResult(
        ticker="TEST",
        iteration=0,
        status="OK",
        original_dcf=100.0,
        original_epv=60.0,
        original_graham=80.0,
        current_price=75.0,
        adjustments=[],
        adjusted_dcf=100.0,
        adjusted_epv=60.0,
        adjusted_intrinsic_mid=80.0,
        adjusted_margin_of_safety=0.067,
        hypotheses_confirmed=confirmed,
        hypotheses_contradicted=contradicted,
        hypotheses_partially_confirmed=partially,
        hypotheses_inconclusive=inconclusive,
        hypotheses_unclassified=unclassified,
        average_coverage=average_coverage,
        unresolved=[],
        high_priority_unresolved=0,
    )


class TestNoScorecardShortCircuit:

    def test_all_scores_zero(self):
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            status="NO_SCORECARD",
            investigation_ran=False,
            methods_agree=None,
            consensus_strength=None,
            method_count=None,
            tension_type=None,
            gate_action=None,
        )
        result = compute_conviction(report)
        assert result.conviction_score == 0
        assert result.conviction_class == "INSUFFICIENT"
        assert result.method_agreement_score == 0
        assert result.evidence_coverage_score == 0
        assert result.gate_quality_score == 0
        assert result.investigation_resolution_score == 0
        assert "NO_SCORECARD" in result.detail


class TestMethodAgreementScore:

    def test_all_methods_agree(self):
        """All methods agree (2+ methods) -> 25."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            methods_agree=True, method_count=3, consensus_strength=3,
            thesis=_make_thesis(),
        )
        result = compute_conviction(report)
        assert result.method_agreement_score == 25

    def test_three_of_four_agree(self):
        """3 of 4 agree -> round(5 + 20 * 0.75) = 20."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            methods_agree=False, method_count=4, consensus_strength=3,
            thesis=_make_thesis(),
        )
        result = compute_conviction(report)
        assert result.method_agreement_score == 20

    def test_two_of_four_agree(self):
        """2 of 4 agree -> round(5 + 20 * 0.5) = 15."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            methods_agree=False, method_count=4, consensus_strength=2,
            thesis=_make_thesis(),
        )
        result = compute_conviction(report)
        assert result.method_agreement_score == 15

    def test_two_of_three_agree(self):
        """2 of 3 agree -> round(5 + 20 * 0.667) = 18."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            methods_agree=False, method_count=3, consensus_strength=2,
            thesis=_make_thesis(),
        )
        result = compute_conviction(report)
        assert result.method_agreement_score == 18

    def test_single_method(self):
        """Only 1 method -> 5."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            methods_agree=True, method_count=1, consensus_strength=1,
            thesis=_make_thesis(),
        )
        result = compute_conviction(report)
        assert result.method_agreement_score == 5

    def test_none_method_count(self):
        """method_count=None -> 5 (insufficient methods)."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            methods_agree=None, method_count=None, consensus_strength=None,
            thesis=_make_thesis(),
        )
        result = compute_conviction(report)
        assert result.method_agreement_score == 5

    def test_none_consensus_strength_with_multiple_methods(self):
        """consensus_strength=None with method_count>1 -> 5 (defensive guard)."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            methods_agree=False, method_count=3, consensus_strength=None,
            thesis=_make_thesis(),
        )
        result = compute_conviction(report)
        assert result.method_agreement_score == 5


class TestEvidenceCoverageScore:

    def test_full_coverage(self):
        """average_coverage=1.0 -> 25."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(thesis=_make_thesis(average_coverage=1.0))
        result = compute_conviction(report)
        assert result.evidence_coverage_score == 25

    def test_partial_coverage(self):
        """average_coverage=0.8 -> 20."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(thesis=_make_thesis(average_coverage=0.8))
        result = compute_conviction(report)
        assert result.evidence_coverage_score == 20

    def test_half_coverage(self):
        """average_coverage=0.5 -> round(12.5) = 12 (banker's rounding)."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(thesis=_make_thesis(average_coverage=0.5))
        result = compute_conviction(report)
        assert result.evidence_coverage_score == 12

    def test_zero_coverage(self):
        """average_coverage=0.0 -> 0."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(thesis=_make_thesis(average_coverage=0.0))
        result = compute_conviction(report)
        assert result.evidence_coverage_score == 0

    def test_no_investigation(self):
        """investigation_ran=False -> 0."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(investigation_ran=False, thesis=None)
        result = compute_conviction(report)
        assert result.evidence_coverage_score == 0

    def test_no_thesis(self):
        """thesis=None with investigation_ran=True -> 0."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(investigation_ran=True, thesis=None)
        result = compute_conviction(report)
        assert result.evidence_coverage_score == 0


class TestGateQualityScore:

    def test_proceed(self):
        """PROCEED -> 25."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(gate_action="PROCEED", thesis=_make_thesis())
        result = compute_conviction(report)
        assert result.gate_quality_score == 25

    def test_adjust(self):
        """ADJUST -> 15."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(gate_action="ADJUST", thesis=_make_thesis())
        result = compute_conviction(report)
        assert result.gate_quality_score == 15

    def test_block(self):
        """BLOCK -> 0, no better than an unknown gate (was 5 before 2026-09-28)."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(gate_action="BLOCK", thesis=_make_thesis())
        result = compute_conviction(report)
        assert result.gate_quality_score == 0

    def test_block_caps_an_otherwise_perfect_report_at_low(self):
        """Full agreement, coverage and resolution with a BLOCKED gate:
        25 + 25 + 0 + 25 = 75 would read HIGH; the gate caps it at 49, LOW."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            gate_action="BLOCK",
            methods_agree=True, method_count=3, consensus_strength=3,
            tension_type="NONE",
            thesis=_make_thesis(confirmed=4, contradicted=0, partially=0,
                                inconclusive=0, unclassified=0, average_coverage=1.0),
        )
        result = compute_conviction(report)
        assert result.conviction_score == 49
        assert result.conviction_class == "LOW"
        assert result.detail.endswith("(capped: gate blocked)")

    def test_none(self):
        """gate_action=None -> 0."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(gate_action=None, thesis=_make_thesis())
        result = compute_conviction(report)
        assert result.gate_quality_score == 0


class TestInvestigationResolutionScore:

    def test_all_resolved_no_tension(self):
        """All resolved + no tension -> 20 + 5 bonus = 25."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            tension_type="NONE",
            thesis=_make_thesis(confirmed=3, contradicted=1, partially=1,
                                inconclusive=0, unclassified=0),
        )
        result = compute_conviction(report)
        assert result.investigation_resolution_score == 25

    def test_all_resolved_with_tension(self):
        """All resolved + tension exists -> 20 (no bonus)."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            tension_type="GROWTH_VS_EARNINGS_POWER",
            methods_agree=False,
            thesis=_make_thesis(confirmed=3, contradicted=2, partially=0,
                                inconclusive=0, unclassified=0),
        )
        result = compute_conviction(report)
        assert result.investigation_resolution_score == 20

    def test_eighty_percent_resolved_no_tension(self):
        """80% resolved + no tension -> round(20*0.8) + 5 = 21."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            tension_type="NONE",
            thesis=_make_thesis(confirmed=2, contradicted=1, partially=1,
                                inconclusive=1, unclassified=0),
        )
        result = compute_conviction(report)
        assert result.investigation_resolution_score == 21

    def test_eighty_percent_resolved_with_tension(self):
        """80% resolved + tension -> round(20*0.8) = 16."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            tension_type="GROWTH_VS_EARNINGS_POWER",
            methods_agree=False,
            thesis=_make_thesis(confirmed=2, contradicted=1, partially=1,
                                inconclusive=1, unclassified=0),
        )
        result = compute_conviction(report)
        assert result.investigation_resolution_score == 16

    def test_fifty_percent_resolved(self):
        """50% resolved -> round(20*0.5) = 10."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            tension_type="GROWTH_VS_EARNINGS_POWER",
            methods_agree=False,
            thesis=_make_thesis(confirmed=1, contradicted=0, partially=0,
                                inconclusive=1, unclassified=0),
        )
        result = compute_conviction(report)
        assert result.investigation_resolution_score == 10

    def test_nothing_resolved(self):
        """0% resolved with tension -> 0 (no bonus)."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            tension_type="GROWTH_VS_EARNINGS_POWER",
            methods_agree=False,
            thesis=_make_thesis(confirmed=0, contradicted=0, partially=0,
                                inconclusive=3, unclassified=0),
        )
        result = compute_conviction(report)
        assert result.investigation_resolution_score == 0

    def test_no_investigation(self):
        """investigation_ran=False -> 0."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(investigation_ran=False, thesis=None)
        result = compute_conviction(report)
        assert result.investigation_resolution_score == 0

    def test_zero_total_hypotheses(self):
        """All hypothesis counts are 0 -> 0."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            thesis=_make_thesis(confirmed=0, contradicted=0, partially=0,
                                inconclusive=0, unclassified=0),
        )
        result = compute_conviction(report)
        assert result.investigation_resolution_score == 0


class TestConvictionClassThresholds:

    def test_high_class(self):
        """Score >= 75 -> HIGH. All components maxed: 25+25+25+25=100."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            gate_action="PROCEED",
            methods_agree=True, method_count=3, consensus_strength=3,
            tension_type="NONE",
            thesis=_make_thesis(confirmed=5, contradicted=0, partially=0,
                                inconclusive=0, unclassified=0, average_coverage=1.0),
        )
        result = compute_conviction(report)
        assert result.conviction_class == "HIGH"
        assert result.conviction_score == 100

    def test_moderate_class(self):
        """Score >= 50, < 75 -> MODERATE.
        method=25, evidence=12, gate=15, resolution=15 -> 67."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            gate_action="ADJUST",
            methods_agree=True, method_count=3, consensus_strength=3,
            tension_type="NONE",
            thesis=_make_thesis(confirmed=2, contradicted=0, partially=0,
                                inconclusive=2, unclassified=0, average_coverage=0.5),
        )
        result = compute_conviction(report)
        # method=25, evidence=round(25*0.5)=12, gate=15, resolution=round(20*0.5)+5=15
        assert result.conviction_score == 67
        assert result.conviction_class == "MODERATE"

    def test_low_class(self):
        """Score >= 25, < 50 -> LOW.
        method=15, evidence=5, gate=15, resolution=4 -> 39."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            gate_action="ADJUST",
            methods_agree=False, method_count=4, consensus_strength=2,
            tension_type="GROWTH_VS_EARNINGS_POWER",
            thesis=_make_thesis(confirmed=1, contradicted=0, partially=0,
                                inconclusive=3, unclassified=1, average_coverage=0.2),
        )
        result = compute_conviction(report)
        # method=round(5+20*0.5)=15, evidence=round(25*0.2)=5, gate=15, resolution=round(20*0.2)=4
        assert result.conviction_score == 39
        assert result.conviction_class == "LOW"

    def test_insufficient_class(self):
        """Score < 25 -> INSUFFICIENT.
        method=5, evidence=0, gate=0 (BLOCK), resolution=0 -> 5."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            gate_action="BLOCK",
            methods_agree=True, method_count=1, consensus_strength=1,
            investigation_ran=False,
            thesis=None,
        )
        result = compute_conviction(report)
        # method=5, evidence=0, gate=0, resolution=0
        assert result.conviction_score == 5
        assert result.conviction_class == "INSUFFICIENT"

    def test_detail_string_format(self):
        """Detail string contains all 4 component scores."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            gate_action="PROCEED",
            methods_agree=True, method_count=2, consensus_strength=2,
            thesis=_make_thesis(average_coverage=0.8),
        )
        result = compute_conviction(report)
        assert "method_agreement=" in result.detail
        assert "evidence_coverage=" in result.detail
        assert "gate_quality=" in result.detail
        assert "investigation_resolution=" in result.detail
        assert "/25" in result.detail

    def test_boundary_75(self):
        """Score exactly 75 -> HIGH."""
        from app.valuation.conviction import compute_conviction
        # method=25, evidence=20, gate=25, resolution=5 -> 75
        report = _make_report(
            gate_action="PROCEED",
            methods_agree=True, method_count=2, consensus_strength=2,
            tension_type="NONE",
            thesis=_make_thesis(confirmed=0, contradicted=0, partially=0,
                                inconclusive=3, unclassified=0, average_coverage=0.8),
        )
        result = compute_conviction(report)
        # method=25, evidence=20, gate=25, resolution=0+5=5 -> 75
        assert result.conviction_score == 75
        assert result.conviction_class == "HIGH"

    def test_boundary_50(self):
        """Score exactly 50 -> MODERATE (requires investigation to have run)."""
        from app.valuation.conviction import compute_conviction
        # With investigation: method=25, evidence=0, gate=25, resolution=0 -> 50
        report = _make_report(
            gate_action="PROCEED",
            methods_agree=True, method_count=2, consensus_strength=2,
            investigation_ran=True,
            thesis=_make_thesis(
                average_coverage=0.0,
                confirmed=0, contradicted=0, partially=0, inconclusive=0, unclassified=0,
            ),
        )
        result = compute_conviction(report)
        assert result.conviction_score == 50
        assert result.conviction_class == "MODERATE"

    def test_uninvestigated_capped_at_25(self):
        """Uninvestigated report capped at 25 regardless of method/gate scores."""
        from app.valuation.conviction import compute_conviction
        report = _make_report(
            gate_action="PROCEED",
            methods_agree=True, method_count=2, consensus_strength=2,
            investigation_ran=False,
            thesis=None,
        )
        result = compute_conviction(report)
        assert result.conviction_score == 25
        assert result.conviction_class == "LOW"
        assert "capped" in result.detail

    def test_boundary_25(self):
        """Score exactly 25 -> LOW."""
        from app.valuation.conviction import compute_conviction
        # method=25, evidence=0, gate=0, resolution=0 -> 25
        report = _make_report(
            gate_action=None,
            methods_agree=True, method_count=2, consensus_strength=2,
            investigation_ran=False,
            thesis=None,
        )
        result = compute_conviction(report)
        # method=25, evidence=0, gate=0, resolution=0 -> 25
        assert result.conviction_score == 25
        assert result.conviction_class == "LOW"


class TestRescoring:
    """Round-trip: persist → deserialize → rescore must produce identical results."""

    def test_round_trip_rescore(self):
        from dataclasses import asdict
        from app.valuation.conviction import compute_conviction
        from app.research.deep_research import ResearchReport
        from app.research.thesis_updater import ThesisAdjustment, UnresolvedEvidence

        thesis = _make_thesis(confirmed=3, contradicted=1, partially=0,
                              inconclusive=1, unclassified=0, average_coverage=0.8)
        thesis.adjustments = [ThesisAdjustment(
            hypothesis_source="GATE_SIGNAL",
            hypothesis_claim="test claim",
            hypothesis_direction="BEARISH",
            hypothesis_status="CONFIRMED",
            affected_method="dcf",
            adjustment_magnitude=-5.0,
            adjustment_confidence="FACT_CALIBRATED",
            calibration_detail="test detail",
            evidence_item_ids=["ev1"],
            structured_facts_used=["fact1"],
        )]
        thesis.unresolved = [UnresolvedEvidence(
            need_id="n1",
            description="test need",
            importance="HIGH",
            unresolved_reason="INCONCLUSIVE",
            hypothesis_source="ANOMALY",
            hypothesis_priority="P1",
            hypothesis_direction="BEARISH",
        )]

        report = _make_report(
            gate_action="ADJUST",
            methods_agree=True, method_count=3, consensus_strength=3,
            tension_type="NONE",
            thesis=thesis,
        )
        report.researchable_items = [thesis.unresolved[0]]

        # Score the original
        original = compute_conviction(report)

        # Round-trip through dict (simulates DB persist + load)
        data = asdict(report)
        reconstructed = ResearchReport.from_dict(data)

        # Rescore the reconstructed report
        rescored = compute_conviction(reconstructed)

        assert rescored.conviction_score == original.conviction_score
        assert rescored.conviction_class == original.conviction_class
        assert rescored.method_agreement_score == original.method_agreement_score
        assert rescored.evidence_coverage_score == original.evidence_coverage_score
        assert rescored.gate_quality_score == original.gate_quality_score
        assert rescored.investigation_resolution_score == original.investigation_resolution_score

    def test_round_trip_no_thesis(self):
        """Reports with thesis=None also round-trip correctly."""
        from dataclasses import asdict
        from app.valuation.conviction import compute_conviction
        from app.research.deep_research import ResearchReport

        report = _make_report(
            status="NO_SCORECARD",
            investigation_ran=False,
            methods_agree=None, consensus_strength=None,
            method_count=None, tension_type=None,
            gate_action=None, thesis=None,
        )

        original = compute_conviction(report)
        data = asdict(report)
        reconstructed = ResearchReport.from_dict(data)
        rescored = compute_conviction(reconstructed)

        assert rescored.conviction_score == original.conviction_score
        assert rescored.conviction_class == original.conviction_class
