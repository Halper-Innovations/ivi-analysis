"""Tests for app.research.report_renderer."""
from __future__ import annotations

from app.research.report_renderer import build_view, render_report


def _make_minimal_report(**overrides):
    """Build a minimal ResearchReport for testing build_view."""
    from app.research.deep_research import ResearchReport
    defaults = dict(
        ticker="TEST", as_of_date="2026-04-04", status="OK",
        started_at="t0", completed_at="t1",
        scorecard_present=True, filing_present=True,
        filing_date="2025-11-15", form_type="10-K",
        anomaly_count=0, solvency_status="LOW",
        filing_risk_status=None, gate_action="PROCEED",
        investigation_ran=True, hypotheses_generated=3,
        thesis=None,
        researchable_items=[], not_researchable_items=[],
        total_adjustments=0, fact_calibrated_count=0, heuristic_count=0,
        methods_agree=True, consensus_strength=2,
        method_count=2, tension_type="NONE",
        citations=[],
        conviction_score=60, conviction_class="MODERATE",
        report_path=None,
    )
    defaults.update(overrides)
    return ResearchReport(**defaults)


def _make_thesis(**overrides):
    from app.research.thesis_updater import ThesisResult
    defaults = dict(
        ticker="TEST", iteration=0, status="OK",
        original_dcf=42.0, original_epv=18.0, original_graham=22.0,
        current_price=30.0,
        adjustments=[],
        adjusted_dcf=42.0, adjusted_epv=18.0,
        adjusted_intrinsic_mid=30.0, adjusted_margin_of_safety=0.0,
        hypotheses_confirmed=2, hypotheses_contradicted=0,
        hypotheses_partially_confirmed=1, hypotheses_inconclusive=0,
        hypotheses_unclassified=0, average_coverage=0.8,
        unresolved=[], high_priority_unresolved=0,
    )
    defaults.update(overrides)
    return ThesisResult(**defaults)


def _make_adjustment(**overrides):
    from app.research.thesis_updater import ThesisAdjustment
    defaults = dict(
        hypothesis_source="GROWTH_VS_EARNINGS_POWER",
        hypothesis_claim="Growth may be overstated",
        hypothesis_direction="BEARISH",
        hypothesis_status="CONFIRMED",
        affected_method="dcf",
        adjustment_magnitude=-5.0,
        adjustment_confidence="FACT_CALIBRATED",
        calibration_detail="concentration 35% -> 2pp growth haircut -> $5.00",
        evidence_item_ids=["n1"],
        structured_facts_used=["35%"],
    )
    defaults.update(overrides)
    return ThesisAdjustment(**defaults)


def _make_unresolved(**overrides):
    from app.research.thesis_updater import UnresolvedEvidence
    defaults = dict(
        need_id="n99", description="retention rate",
        importance="SUPPORTING", unresolved_reason="NOT_FOUND",
        hypothesis_source="GROWTH_VS_EARNINGS_POWER",
        hypothesis_priority="P2", hypothesis_direction="BEARISH",
    )
    defaults.update(overrides)
    return UnresolvedEvidence(**defaults)


class TestVerdictDerivation:
    """Verdict precedence: BLOCK gate > low conviction > blockers > ADJUST gate > PROCEED."""

    def test_do_not_act_block_gate(self):
        from app.research.report_renderer import build_view
        report = _make_minimal_report(gate_action="BLOCK", conviction_score=80)
        view = build_view(report)
        assert view.verdict == "DO NOT ACT"

    def test_do_not_act_low_conviction(self):
        from app.research.report_renderer import build_view
        report = _make_minimal_report(conviction_score=20, conviction_class="INSUFFICIENT")
        view = build_view(report)
        assert view.verdict == "DO NOT ACT"

    def test_watch_hard_blockers(self):
        from app.research.report_renderer import build_view
        blocker = _make_unresolved(importance="REQUIRED")
        report = _make_minimal_report(
            conviction_score=60,
            researchable_items=[blocker], not_researchable_items=[],
        )
        view = build_view(report)
        assert view.verdict == "WATCH"

    def test_watch_adjust_gate(self):
        from app.research.report_renderer import build_view
        report = _make_minimal_report(gate_action="ADJUST", conviction_score=60)
        view = build_view(report)
        assert view.verdict == "WATCH"

    def test_proceed_default(self):
        from app.research.report_renderer import build_view
        report = _make_minimal_report(conviction_score=60)
        view = build_view(report)
        assert view.verdict == "PROCEED"

    def test_block_gate_overrides_high_conviction(self):
        from app.research.report_renderer import build_view
        report = _make_minimal_report(gate_action="BLOCK", conviction_score=95)
        view = build_view(report)
        assert view.verdict == "DO NOT ACT"

    def test_conviction_none_treated_as_insufficient(self):
        from app.research.report_renderer import build_view
        report = _make_minimal_report(conviction_score=None, conviction_class=None)
        view = build_view(report)
        assert view.verdict == "DO NOT ACT"

    def test_uninvestigated_never_proceed(self):
        """Even with PROCEED gate and high conviction, uninvestigated = WATCH."""
        from app.research.report_renderer import build_view
        report = _make_minimal_report(
            gate_action="PROCEED", conviction_score=60,
            investigation_ran=False,
        )
        view = build_view(report)
        assert view.verdict == "WATCH"
        assert view.verdict != "PROCEED"


class TestBlockerClassification:

    def test_required_importance_is_blocker(self):
        from app.research.report_renderer import build_view
        blocker = _make_unresolved(importance="REQUIRED")
        report = _make_minimal_report(
            researchable_items=[blocker], not_researchable_items=[],
        )
        view = build_view(report)
        assert len(view.hard_blockers) == 1
        assert view.hard_blockers[0].blocking_reason == "Required evidence for hypothesis resolution"

    def test_thesis_critical_source_is_blocker(self):
        from app.research.report_renderer import build_view
        adj = _make_adjustment(
            hypothesis_source="MARGIN_COLLAPSE",
            adjustment_magnitude=-10.0,
        )
        thesis = _make_thesis(adjustments=[adj])
        item = _make_unresolved(
            importance="SUPPORTING",
            hypothesis_source="MARGIN_COLLAPSE",
        )
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=1,
            researchable_items=[item], not_researchable_items=[],
        )
        view = build_view(report)
        assert len(view.hard_blockers) == 1
        assert "MARGIN_COLLAPSE" in view.hard_blockers[0].blocking_reason

    def test_important_p1_bearish_is_blocker(self):
        from app.research.report_renderer import build_view
        item = _make_unresolved(
            importance="IMPORTANT",
            hypothesis_priority="P1",
            hypothesis_direction="BEARISH",
        )
        report = _make_minimal_report(
            researchable_items=[item], not_researchable_items=[],
        )
        view = build_view(report)
        assert len(view.hard_blockers) == 1

    def test_supporting_p2_is_open_question(self):
        from app.research.report_renderer import build_view
        item = _make_unresolved(
            importance="SUPPORTING",
            hypothesis_priority="P2",
            hypothesis_direction="BEARISH",
        )
        report = _make_minimal_report(
            researchable_items=[item], not_researchable_items=[],
        )
        view = build_view(report)
        assert len(view.hard_blockers) == 0
        assert len(view.open_questions) == 1

    def test_missing_importance_falls_back_to_rule2(self):
        from app.research.report_renderer import build_view
        adj = _make_adjustment(
            hypothesis_source="SECULAR_DECLINE",
            adjustment_magnitude=-8.0,
        )
        thesis = _make_thesis(adjustments=[adj])
        item = _make_unresolved(
            importance="",
            hypothesis_source="SECULAR_DECLINE",
        )
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=1,
            researchable_items=[item], not_researchable_items=[],
        )
        view = build_view(report)
        assert len(view.hard_blockers) == 1

    def test_all_fields_missing_fails_safe(self):
        from app.research.report_renderer import build_view
        item = _make_unresolved(
            importance="",
            hypothesis_source="",
            hypothesis_priority="",
            hypothesis_direction="",
        )
        report = _make_minimal_report(
            researchable_items=[item], not_researchable_items=[],
        )
        view = build_view(report)
        assert len(view.hard_blockers) == 0
        assert len(view.open_questions) == 1


class TestMethodUsability:

    def test_na_when_no_value(self):
        from app.research.report_renderer import build_view
        thesis = _make_thesis(original_dcf=None, adjusted_dcf=None)
        report = _make_minimal_report(thesis=thesis)
        view = build_view(report)
        dcf = next(m for m in view.methods if m.name == "DCF")
        assert dcf.usability == "N/A"

    def test_ok_when_no_adjustments(self):
        from app.research.report_renderer import build_view
        thesis = _make_thesis(original_dcf=42.0, adjusted_dcf=42.0)
        report = _make_minimal_report(thesis=thesis)
        view = build_view(report)
        dcf = next(m for m in view.methods if m.name == "DCF")
        assert dcf.usability == "OK"

    def test_overridden_when_adjustments_exceed_20pct(self):
        from app.research.report_renderer import build_view
        adj = _make_adjustment(affected_method="dcf", adjustment_magnitude=-12.0)
        thesis = _make_thesis(
            original_dcf=42.0, adjusted_dcf=30.0,
            adjustments=[adj],
        )
        report = _make_minimal_report(thesis=thesis, total_adjustments=1)
        view = build_view(report)
        dcf = next(m for m in view.methods if m.name == "DCF")
        assert dcf.usability == "OVERRIDDEN"

    def test_ok_when_adjustments_under_20pct(self):
        from app.research.report_renderer import build_view
        adj = _make_adjustment(affected_method="dcf", adjustment_magnitude=-2.0)
        thesis = _make_thesis(
            original_dcf=42.0, adjusted_dcf=40.0,
            adjustments=[adj],
        )
        report = _make_minimal_report(thesis=thesis, total_adjustments=1)
        view = build_view(report)
        dcf = next(m for m in view.methods if m.name == "DCF")
        assert dcf.usability == "OK"


class TestWhyWrong:

    def test_capped_at_4(self):
        from app.research.report_renderer import build_view
        adjs = [
            _make_adjustment(
                hypothesis_claim=f"Claim {i}",
                adjustment_magnitude=float(-i),
            )
            for i in range(1, 6)
        ]
        blocker = _make_unresolved(importance="REQUIRED")
        thesis = _make_thesis(adjustments=adjs)
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=5,
            researchable_items=[blocker], not_researchable_items=[],
        )
        view = build_view(report)
        assert len(view.why_wrong) <= 4

    def test_includes_adjustments_and_blockers(self):
        from app.research.report_renderer import build_view
        adj = _make_adjustment(adjustment_magnitude=-10.0)
        blocker = _make_unresolved(importance="REQUIRED")
        thesis = _make_thesis(adjustments=[adj])
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=1,
            researchable_items=[blocker], not_researchable_items=[],
        )
        view = build_view(report)
        assert len(view.why_wrong) == 2
        assumptions = [w.assumption for w in view.why_wrong]
        assert any("Growth" in a for a in assumptions)
        assert any("Unknown" in a for a in assumptions)

    def test_empty_when_no_adjustments_no_blockers(self):
        from app.research.report_renderer import build_view
        report = _make_minimal_report()
        view = build_view(report)
        assert view.why_wrong == []


class TestRenderReport:

    def test_full_report_has_all_sections(self):
        from app.research.report_renderer import build_view, render_report
        from app.research.deep_research import ReportCitation
        adj = _make_adjustment(adjustment_magnitude=-10.0)
        blocker = _make_unresolved(importance="REQUIRED")
        question = _make_unresolved(importance="SUPPORTING", need_id="n100", hypothesis_source="OTHER_SOURCE")
        thesis = _make_thesis(adjustments=[adj])
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=1,
            researchable_items=[blocker, question], not_researchable_items=[],
            citations=[ReportCitation(
                citation_id="C1", need_id="n1", section="Item 1A",
                excerpt="35% revenue concentration",
                relevance="customer concentration", hypothesis_source="GROWTH_VS_EARNINGS_POWER",
                source_quality={
                    "source_family": "company_controlled",
                    "freshness_bucket": "recent_7d",
                    "source_quality_score": 0.85,
                    "calibration_status": "deterministic_heuristic",
                },
            )],
        )
        view = build_view(report)
        md = render_report(view)
        assert "# TEST" in md
        assert "WATCH" in md  # blocker forces WATCH
        assert "| DCF" in md
        assert "Thesis Adjustments" in md
        assert "Why This Could Be Wrong" in md
        assert "Hard Blockers" in md
        assert "Open Questions" in md
        assert "Conviction Breakdown" in md
        assert "[C1]" in md
        assert "Citations" in md
        assert "Source Quality: company_controlled; freshness=recent_7d; score=0.85; calibration=deterministic_heuristic" in md

    def test_no_investigation_omits_sections(self):
        from app.research.report_renderer import build_view, render_report
        report = _make_minimal_report(
            investigation_ran=False, status="NO_FILING",
            conviction_score=None, conviction_class=None,
        )
        view = build_view(report)
        md = render_report(view)
        assert "# TEST" in md
        assert "Thesis Adjustments" not in md
        assert "Why This Could Be Wrong" not in md
        assert "Hard Blockers" not in md
        assert "Open Questions" not in md
        assert "Citations" not in md
        assert "did not run" in md.lower() or "NO_FILING" in md

    def test_citation_handles_in_adjustments(self):
        from app.research.report_renderer import build_view, render_report
        from app.research.deep_research import ReportCitation
        adj = _make_adjustment(evidence_item_ids=["n1"])
        thesis = _make_thesis(adjustments=[adj])
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=1,
            citations=[ReportCitation(
                citation_id="C1", need_id="n1", section="Item 1A",
                excerpt="text", relevance="rel",
                hypothesis_source="GROWTH_VS_EARNINGS_POWER",
            )],
        )
        view = build_view(report)
        md = render_report(view)
        assert "[C1]" in md

    def test_renders_material_event_item_provenance_in_citations(self):
        from app.research.report_renderer import build_view, render_report
        from app.research.deep_research import ReportCitation

        report = _make_minimal_report(
            citations=[
                ReportCitation(
                    citation_id="C1",
                    need_id="n1",
                    section="leadership_governance",
                    excerpt="The board appointed a new chief executive officer.",
                    relevance="leadership transition",
                    hypothesis_source="OTHER_MATERIAL",
                    source_form_type="8-K",
                    source_filing_date="2026-04-10",
                    source_accession="0000000000-26-000101",
                    source_role="material_event",
                    item_code="5.02",
                    event_category="leadership_governance",
                )
            ],
        )

        view = build_view(report)
        md = render_report(view)

        assert "8-K / Item 5.02 / 2026-04-10 - leadership_governance" in md


class TestRenderSummary:

    def test_six_line_format(self):
        from app.research.report_renderer import build_view, render_summary
        adj = _make_adjustment(adjustment_magnitude=-12.0)
        blocker = _make_unresolved(importance="REQUIRED")
        question = _make_unresolved(importance="SUPPORTING", need_id="n100", hypothesis_source="OTHER")
        thesis = _make_thesis(
            adjustments=[adj],
            adjusted_dcf=30.0, adjusted_epv=17.0,
            adjusted_margin_of_safety=0.362,
            hypotheses_confirmed=3,
        )
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=1,
            conviction_score=58, conviction_class="MODERATE",
            tension_type="GROWTH_VS_EARNINGS_POWER",
            researchable_items=[blocker, question], not_researchable_items=[],
            report_path="data/outputs/research/TEST_report.md",
        )
        view = build_view(report)
        summary = render_summary(view)
        lines = [l for l in summary.strip().split("\n") if l.strip()]
        assert len(lines) == 6
        assert "TEST" in lines[0]
        assert "WATCH" in lines[0]
        assert "58" in lines[0]
        assert "MOS" in lines[1]
        assert "$" in lines[2]
        assert "GROWTH_VS_EARNINGS_POWER" in lines[3]
        assert "confirmed" in lines[4].lower()
        assert "report.md" in lines[5]

    def test_no_investigation_summary(self):
        from app.research.report_renderer import build_view, render_summary
        report = _make_minimal_report(
            investigation_ran=False, status="NO_FILING",
            conviction_score=None, conviction_class=None,
            tension_type=None,
        )
        view = build_view(report)
        summary = render_summary(view)
        assert "TEST" in summary
        assert "DO NOT ACT" in summary

    def test_path_always_last_line(self):
        from app.research.report_renderer import build_view, render_summary
        report = _make_minimal_report(report_path="/tmp/test.md")
        view = build_view(report)
        summary = render_summary(view)
        last_line = summary.strip().split("\n")[-1]
        assert "/tmp/test.md" in last_line


class TestScorecardFallback:
    """Bug 2: NO_HYPOTHESES reports should show scorecard values."""

    def test_scorecard_values_shown_when_no_thesis(self):
        from app.research.report_renderer import build_view, render_report
        report = _make_minimal_report(
            investigation_ran=False, status="NO_HYPOTHESES",
            thesis=None,
            conviction_score=25, conviction_class="LOW",
            scorecard_dcf=145.27, scorecard_epv=99.28,
            scorecard_graham=110.0, scorecard_price=138.00,
        )
        view = build_view(report)
        assert view.current_price == 138.00
        md = render_report(view)
        assert "$138.00" in md
        assert "$145.27" in md  # DCF should appear
        assert "$99.28" in md   # EPV should appear
        assert "N/A" not in md.split("Current Price")[1].split("\n")[0]

    def test_thesis_values_take_precedence(self):
        from app.research.report_renderer import build_view
        thesis = _make_thesis(original_dcf=50.0, current_price=30.0)
        report = _make_minimal_report(
            thesis=thesis,
            scorecard_dcf=999.0, scorecard_price=999.0,
        )
        view = build_view(report)
        assert view.current_price == 30.0  # thesis wins
        dcf = next(m for m in view.methods if m.name == "DCF")
        assert dcf.original == 50.0  # thesis wins


class TestSharedCitationHandles:
    """Fix for finding 1: shared citations must map handles for all need_ids."""

    def test_shared_citation_maps_to_both_adjustments(self):
        from app.research.report_renderer import build_view
        from app.research.deep_research import ReportCitation
        # Two adjustments reference different need_ids, but the filing excerpt is the same
        adj1 = _make_adjustment(
            hypothesis_source="SRC_A", hypothesis_claim="Claim A",
            evidence_item_ids=["n1"], adjustment_magnitude=-5.0,
        )
        adj2 = _make_adjustment(
            hypothesis_source="SRC_B", hypothesis_claim="Claim B",
            evidence_item_ids=["n2"], adjustment_magnitude=-3.0,
        )
        thesis = _make_thesis(adjustments=[adj1, adj2])
        # Citation was deduped: primary need_id is n1, additional is n2
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=2,
            citations=[ReportCitation(
                citation_id="C1", need_id="n1", section="Item 1A",
                excerpt="shared text", relevance="rel",
                hypothesis_source="SRC_A",
                additional_need_ids=["n2"],
            )],
        )
        view = build_view(report)
        # Both adjustments should get the [C1] handle
        assert view.adjustments[0].citation_handles == ["C1"]
        assert view.adjustments[1].citation_handles == ["C1"]


class TestZeroValueHandling:
    """Fix for finding 3: 0.0 values must not be treated as missing."""

    def test_zero_price_renders_as_dollar_zero(self):
        from app.research.report_renderer import build_view, render_report
        thesis = _make_thesis(current_price=0.0, original_dcf=10.0, adjusted_dcf=10.0)
        report = _make_minimal_report(thesis=thesis)
        view = build_view(report)
        md = render_report(view)
        assert "$0.00" in md
        assert "N/A" not in md.split("Current Price")[1].split("\n")[0]

    def test_zero_adjusted_value_used_in_range(self):
        from app.research.report_renderer import build_view, render_summary
        thesis = _make_thesis(
            original_dcf=10.0, adjusted_dcf=0.0,
            original_epv=5.0, adjusted_epv=5.0,
        )
        report = _make_minimal_report(
            thesis=thesis, conviction_score=60,
            report_path="/tmp/test.md",
        )
        view = build_view(report)
        summary = render_summary(view)
        # Adjusted range should include 0.0, not fall back to original
        assert "$0.00" in summary


class TestTensionExplanation:
    """Fix for finding 2: tension section must include explanation."""

    def test_growth_tension_has_explanation(self):
        from app.research.report_renderer import build_view, render_report
        thesis = _make_thesis(original_dcf=42.0, adjusted_dcf=42.0, original_epv=18.0, adjusted_epv=18.0)
        report = _make_minimal_report(
            thesis=thesis, tension_type="GROWTH_VS_EARNINGS_POWER",
        )
        view = build_view(report)
        md = render_report(view)
        assert "GROWTH_VS_EARNINGS_POWER" in md
        assert "growth premium" in md.lower() or "growth" in md.lower()
        assert view.tension_explanation is not None

    def test_no_tension_no_explanation(self):
        from app.research.report_renderer import build_view
        report = _make_minimal_report(tension_type="NONE")
        view = build_view(report)
        assert view.tension_explanation is None


class TestConvictionRationale:
    """Fix for finding 2: conviction components must include rationale."""

    def test_conviction_has_rationales(self):
        from app.research.report_renderer import build_view, render_report
        thesis = _make_thesis()
        report = _make_minimal_report(
            thesis=thesis, conviction_score=60, conviction_class="MODERATE",
        )
        view = build_view(report)
        md = render_report(view)
        # Should have explanatory text, not just bare numbers
        assert "/25 —" in md or "/25 —" in md  # rationale separator


# ---------------------------------------------------------------------------
# LOGIC TESTS: verify the verdict is *correct*, not just well-shaped
# ---------------------------------------------------------------------------

class TestValueTrapDetection:
    """A company with severe revenue decline should not get a favorable verdict."""

    def test_severe_decline_gets_do_not_act(self):
        """48% revenue decline + BLOCK gate = DO NOT ACT regardless of discount size."""
        from app.research.report_renderer import build_view
        adj = _make_adjustment(
            hypothesis_source="REVENUE_DECLINE_FROM_PEAK",
            hypothesis_claim="Revenue declined 48% from peak",
            hypothesis_direction="BEARISH",
            hypothesis_status="CONFIRMED",
            affected_method="dcf",
            adjustment_magnitude=-20.0,
        )
        thesis = _make_thesis(
            original_dcf=42.0, adjusted_dcf=22.0,
            original_epv=18.0, adjusted_epv=18.0,
            current_price=12.0,
            adjustments=[adj],
            adjusted_margin_of_safety=0.45,  # looks cheap!
        )
        report = _make_minimal_report(
            gate_action="BLOCK", conviction_score=10, conviction_class="INSUFFICIENT",
            thesis=thesis, total_adjustments=1,
        )
        view = build_view(report)
        assert view.verdict == "DO NOT ACT"
        # Must not say PROCEED even with large margin of safety
        assert view.verdict != "PROCEED"

    def test_high_discount_low_conviction_is_not_proceed(self):
        """Big discount + INSUFFICIENT conviction = not investable."""
        from app.research.report_renderer import build_view
        thesis = _make_thesis(
            original_dcf=100.0, adjusted_dcf=100.0,
            current_price=30.0,
            adjusted_margin_of_safety=0.70,
        )
        report = _make_minimal_report(
            gate_action="PROCEED",
            conviction_score=15, conviction_class="INSUFFICIENT",
            thesis=thesis,
        )
        view = build_view(report)
        assert view.verdict == "DO NOT ACT"

    def test_unresolved_required_evidence_prevents_proceed(self):
        """Required evidence missing = WATCH, never PROCEED."""
        from app.research.report_renderer import build_view
        blocker = _make_unresolved(importance="REQUIRED")
        thesis = _make_thesis(
            original_dcf=50.0, adjusted_dcf=50.0,
            current_price=30.0,
            adjusted_margin_of_safety=0.40,
        )
        report = _make_minimal_report(
            gate_action="PROCEED", conviction_score=70, conviction_class="MODERATE",
            thesis=thesis,
            researchable_items=[blocker], not_researchable_items=[],
        )
        view = build_view(report)
        assert view.verdict == "WATCH"
        assert view.verdict != "PROCEED"


class TestLegitimateUndervaluation:
    """A genuinely undervalued company with strong evidence should get PROCEED."""

    def test_confirmed_thesis_high_conviction_gets_proceed(self):
        """Methods agree, evidence confirms, gate passes = PROCEED."""
        from app.research.report_renderer import build_view
        adj = _make_adjustment(
            hypothesis_source="GROWTH_VS_EARNINGS_POWER",
            hypothesis_claim="Growth supported by retention data",
            hypothesis_direction="BULLISH",
            hypothesis_status="CONFIRMED",
            affected_method="dcf",
            adjustment_magnitude=5.0,
        )
        thesis = _make_thesis(
            original_dcf=50.0, adjusted_dcf=55.0,
            original_epv=45.0, adjusted_epv=45.0,
            current_price=30.0,
            adjustments=[adj],
            adjusted_margin_of_safety=0.40,
            hypotheses_confirmed=3,
            hypotheses_inconclusive=0,
            average_coverage=0.9,
        )
        report = _make_minimal_report(
            gate_action="PROCEED", conviction_score=80, conviction_class="HIGH",
            thesis=thesis, total_adjustments=1,
            methods_agree=True, consensus_strength=3, method_count=3,
            tension_type="NONE",
        )
        view = build_view(report)
        assert view.verdict == "PROCEED"
        assert len(view.hard_blockers) == 0

    def test_moderate_conviction_no_blockers_gets_proceed(self):
        """Moderate conviction with no blockers and PROCEED gate = PROCEED."""
        from app.research.report_renderer import build_view
        thesis = _make_thesis(
            original_dcf=40.0, adjusted_dcf=38.0,
            current_price=25.0,
            adjusted_margin_of_safety=0.34,
        )
        report = _make_minimal_report(
            gate_action="PROCEED", conviction_score=55, conviction_class="MODERATE",
            thesis=thesis,
        )
        view = build_view(report)
        assert view.verdict == "PROCEED"


class TestWhyWrongCorrectness:
    """Why This Could Be Wrong must surface the actual thesis-breaking risks."""

    def test_largest_adjustment_appears_first(self):
        """The biggest valuation impact should be the first thing listed."""
        from app.research.report_renderer import build_view
        small = _make_adjustment(
            hypothesis_claim="Minor issue",
            adjustment_magnitude=-2.0,
        )
        large = _make_adjustment(
            hypothesis_claim="Revenue is structurally declining",
            adjustment_magnitude=-15.0,
            hypothesis_source="REVENUE_DECLINE_FROM_PEAK",
        )
        thesis = _make_thesis(adjustments=[small, large])
        report = _make_minimal_report(thesis=thesis, total_adjustments=2)
        view = build_view(report)
        assert len(view.why_wrong) >= 2
        # Largest impact should be first
        assert "structurally declining" in view.why_wrong[0].assumption.lower()

    def test_blocker_surfaces_in_why_wrong(self):
        """A hard blocker represents unknown risk — must appear in why-wrong."""
        from app.research.report_renderer import build_view
        adj = _make_adjustment(adjustment_magnitude=-5.0)
        blocker = _make_unresolved(
            importance="REQUIRED",
            description="Customer retention rate undisclosed",
        )
        thesis = _make_thesis(adjustments=[adj])
        report = _make_minimal_report(
            thesis=thesis, total_adjustments=1,
            researchable_items=[blocker], not_researchable_items=[],
        )
        view = build_view(report)
        why_texts = " ".join(w.assumption for w in view.why_wrong)
        assert "retention" in why_texts.lower()


class TestAnalystReadView:
    def _report_with_analyst_notes(self):
        from app.research.deep_research import ResearchReport
        from app.research.analyst_notes import AnalystCitation, AnalystNote, AnalystNotes
        from app.research.findings_reconciliation import (
            PipelineFinding, MergedFindings,
        )

        citation = AnalystCitation(section="mda", excerpt="Revenue grew 15%", block_id="mda_p0")
        positive = AnalystNote(
            category="POSITIVE", claim="Strong revenue growth",
            direction="BULLISH", severity="HIGH",
            citations=[citation], suggested_adjustment=None,
            validation_status="VERIFIED",
        )
        risk = AnalystNote(
            category="RISK", claim="Customer concentration above 40%",
            direction="BEARISH", severity="HIGH",
            citations=[AnalystCitation(section="risk_factors", excerpt="top customers", block_id=None)],
            suggested_adjustment="Growth haircut 2pp",
            validation_status="UNVERIFIED",
        )
        notes = AnalystNotes(
            ticker="TEST", positives=[positive], risks=[risk],
            surprises=[], adjustment_triggers=[],
            overall_assessment="Solid company with concentration risk.",
            filing_sections_read=["mda", "risk_factors"],
        )

        pf = PipelineFinding(
            claim="Revenue decline structural concern", direction="BEARISH",
            section="mda", content_words={"revenue", "decline", "structural", "concern"},
            source_hypothesis="REV_DECLINE", evidence_status="CONFIRMED",
        )
        merged = MergedFindings(
            both_paths=[], llm_only=[risk], pipeline_only=[pf],
            agreement_score=0.0,
        )

        return ResearchReport(
            ticker="TEST", as_of_date="2026-01-01", status="OK",
            started_at="t0", completed_at="t1",
            scorecard_present=True, filing_present=True,
            filing_date=None, form_type="10-K",
            anomaly_count=0, solvency_status=None,
            filing_risk_status=None, gate_action="PROCEED",
            investigation_ran=True, hypotheses_generated=1,
            thesis=None,
            researchable_items=[], not_researchable_items=[],
            total_adjustments=0, fact_calibrated_count=0, heuristic_count=0,
            methods_agree=True, consensus_strength=3,
            method_count=3, tension_type="NONE",
            analyst_notes=notes,
            merged_findings=merged,
        )

    def test_build_view_maps_analyst_notes(self):
        from app.research.report_renderer import build_view
        report = self._report_with_analyst_notes()
        view = build_view(report)
        assert view.analyst_read is not None
        assert view.analyst_read.overall_assessment == "Solid company with concentration risk."
        assert len(view.analyst_read.positives) == 1
        assert len(view.analyst_read.risks) == 1
        assert view.analyst_read.positives[0].claim == "Strong revenue growth"
        assert view.analyst_read.positives[0].validation_status == "VERIFIED"
        assert "[A1]" in view.analyst_read.positives[0].citation_handles

    def test_build_view_maps_reconciliation(self):
        from app.research.report_renderer import build_view
        report = self._report_with_analyst_notes()
        view = build_view(report)
        assert view.reconciliation is not None
        assert view.reconciliation.agreement_score == 0.0
        assert len(view.reconciliation.llm_only) == 1
        assert len(view.reconciliation.pipeline_only) == 1

    def test_build_view_none_when_no_analyst_notes(self):
        from app.research.deep_research import ResearchReport
        from app.research.report_renderer import build_view
        report = ResearchReport(
            ticker="TEST", as_of_date="2026-01-01", status="OK",
            started_at="t0", completed_at="t1",
            scorecard_present=True, filing_present=True,
            filing_date=None, form_type="10-K",
            anomaly_count=0, solvency_status=None,
            filing_risk_status=None, gate_action="PROCEED",
            investigation_ran=True, hypotheses_generated=0,
            thesis=None,
            researchable_items=[], not_researchable_items=[],
            total_adjustments=0, fact_calibrated_count=0, heuristic_count=0,
            methods_agree=True, consensus_strength=3,
            method_count=3, tension_type="NONE",
        )
        view = build_view(report)
        assert view.analyst_read is None
        assert view.reconciliation is None

    def test_render_report_includes_analyst_sections(self):
        from app.research.report_renderer import build_view, render_report
        report = self._report_with_analyst_notes()
        view = build_view(report)
        md = render_report(view)
        assert "## Analyst Read" in md
        assert "Strong revenue growth" in md
        assert "## Findings Reconciliation" in md
        assert "[A1]" in md

    def test_render_report_omits_analyst_sections_when_none(self):
        from app.research.deep_research import ResearchReport
        from app.research.report_renderer import build_view, render_report
        report = ResearchReport(
            ticker="TEST", as_of_date="2026-01-01", status="OK",
            started_at="t0", completed_at="t1",
            scorecard_present=True, filing_present=True,
            filing_date=None, form_type="10-K",
            anomaly_count=0, solvency_status=None,
            filing_risk_status=None, gate_action="PROCEED",
            investigation_ran=True, hypotheses_generated=0,
            thesis=None,
            researchable_items=[], not_researchable_items=[],
            total_adjustments=0, fact_calibrated_count=0, heuristic_count=0,
            methods_agree=True, consensus_strength=3,
            method_count=3, tension_type="NONE",
        )
        view = build_view(report)
        md = render_report(view)
        assert "## Analyst Read" not in md
        assert "## Findings Reconciliation" not in md


class TestDeriveVerdict:
    def _make_report(self, **overrides):
        from app.research.deep_research import ResearchReport
        defaults = dict(
            ticker="TEST", as_of_date="2026-01-01", status="OK",
            started_at="t0", completed_at="t1",
            scorecard_present=True, filing_present=True,
            filing_date=None, form_type="10-K",
            anomaly_count=0, solvency_status=None,
            filing_risk_status=None, gate_action="PROCEED",
            investigation_ran=True, hypotheses_generated=1,
            thesis=None,
            researchable_items=[], not_researchable_items=[],
            total_adjustments=0, fact_calibrated_count=0, heuristic_count=0,
            methods_agree=True, consensus_strength=3,
            method_count=3, tension_type="NONE",
            conviction_score=80, conviction_class="HIGH",
        )
        defaults.update(overrides)
        return ResearchReport(**defaults)

    def test_proceed_high_conviction(self):
        from app.research.report_renderer import derive_verdict
        report = self._make_report(gate_action="PROCEED", conviction_score=80)
        assert derive_verdict(report) == "PROCEED"

    def test_block_returns_do_not_act(self):
        from app.research.report_renderer import derive_verdict
        report = self._make_report(gate_action="BLOCK", conviction_score=80)
        assert derive_verdict(report) == "DO NOT ACT"

    def test_low_conviction_returns_do_not_act(self):
        from app.research.report_renderer import derive_verdict
        report = self._make_report(conviction_score=20)
        assert derive_verdict(report) == "DO NOT ACT"

    def test_no_investigation_returns_watch(self):
        from app.research.report_renderer import derive_verdict
        report = self._make_report(conviction_score=80, investigation_ran=False)
        assert derive_verdict(report) == "WATCH"

    def test_adjust_returns_watch(self):
        from app.research.report_renderer import derive_verdict
        report = self._make_report(gate_action="ADJUST", conviction_score=80)
        assert derive_verdict(report) == "WATCH"

    def test_none_conviction_returns_do_not_act(self):
        from app.research.report_renderer import derive_verdict
        report = self._make_report(conviction_score=None)
        assert derive_verdict(report) == "DO NOT ACT"


class TestLLMDisabledWarning:
    """Verify the report surfaces a clear warning when LLM provider was unavailable."""

    def _make_report(self, warnings=None, **overrides):
        from app.research.deep_research import ResearchReport
        defaults = dict(
            ticker="TEST", as_of_date="2026-01-01", status="OK",
            started_at="t0", completed_at="t1",
            scorecard_present=True, filing_present=True,
            filing_date=None, form_type="10-K",
            anomaly_count=0, solvency_status=None,
            filing_risk_status=None, gate_action="PROCEED",
            investigation_ran=False, hypotheses_generated=1,
            thesis=None,
            researchable_items=[], not_researchable_items=[],
            total_adjustments=0, fact_calibrated_count=0, heuristic_count=0,
            methods_agree=True, consensus_strength=3,
            method_count=3, tension_type="NONE",
            conviction_score=45, conviction_class="LOW",
            warnings=warnings or [],
        )
        defaults.update(overrides)
        return ResearchReport(**defaults)

    def test_warning_in_verdict_block(self):
        report = self._make_report(warnings=["llm_provider_disabled"])
        view = build_view(report)
        md = render_report(view)
        assert "# TEST — Deep Research Diagnostic Report" in md
        assert "WARNING: LLM provider was unavailable" in md
        assert "**Artifact Quality:** DEGRADED_DIAGNOSTIC" in md
        assert "pipeline diagnostic, not an investment research output" in md

    def test_no_warning_when_llm_available(self):
        report = self._make_report(warnings=[])
        view = build_view(report)
        md = render_report(view)
        assert "WARNING: LLM provider was unavailable" not in md

    def test_complete_report_has_no_degraded_artifact_quality_label(self):
        report = self._make_report(warnings=[], investigation_ran=True)
        view = build_view(report)
        md = render_report(view)
        assert "# TEST — Deep Research Report" in md
        assert "Deep Research Diagnostic Report" not in md
        assert "Artifact Quality" not in md

    def test_no_investigation_without_llm_warning_is_still_diagnostic(self):
        report = self._make_report(warnings=[], investigation_ran=False, status="NO_HYPOTHESES")
        view = build_view(report)
        md = render_report(view)
        assert "**Artifact Quality:** DEGRADED_DIAGNOSTIC" in md
        assert "Investigation did not run" in md

    def test_conviction_breakdown_shows_llm_unavailable(self):
        report = self._make_report(warnings=["llm_provider_disabled"])
        view = build_view(report)
        md = render_report(view)
        assert "LLM UNAVAILABLE" in md
        assert "not weak fundamentals" in md

    def test_conviction_breakdown_normal_when_llm_available(self):
        report = self._make_report(warnings=[])
        view = build_view(report)
        md = render_report(view)
        assert "LLM UNAVAILABLE" not in md

    def test_warnings_propagate_to_view(self):
        report = self._make_report(warnings=["llm_provider_disabled", "artifact_write_failed"])
        view = build_view(report)
        assert "llm_provider_disabled" in view.warnings
        assert "artifact_write_failed" in view.warnings


class TestCurrentEventRendering:
    def test_build_view_maps_current_event_citation_and_latest_evidence(self):
        from app.research.deep_research import ResearchReport, ReportCitation

        report = ResearchReport(
            ticker="TEST",
            as_of_date="2026-04-18",
            status="OK",
            started_at="t0",
            completed_at="t1",
            scorecard_present=True,
            filing_present=True,
            filing_date="2026-04-01",
            form_type="10-Q",
            latest_evidence_date="2026-04-17T10:00:00+00:00",
            latest_evidence_source_type="ir_press",
            anomaly_count=0,
            solvency_status=None,
            filing_risk_status=None,
            gate_action="PROCEED",
            investigation_ran=True,
            hypotheses_generated=1,
            thesis=None,
            researchable_items=[],
            not_researchable_items=[],
            total_adjustments=0,
            fact_calibrated_count=0,
            heuristic_count=0,
            methods_agree=True,
            consensus_strength=3,
            method_count=3,
            tension_type="NONE",
            conviction_score=80,
            conviction_class="HIGH",
            citations=[
                ReportCitation(
                    citation_id="C1",
                    need_id="n1",
                    section="ir_press",
                    excerpt="Management raised revenue guidance for the year.",
                    relevance="guidance update",
                    hypothesis_source="SRC",
                    source_role="current_event",
                    source_type="ir_press",
                    source_title="Guidance update",
                    source_url="https://example.com/press/guidance",
                    source_published_at="2026-04-17T10:00:00+00:00",
                )
            ],
        )

        view = build_view(report)

        assert view.latest_evidence_date == "2026-04-17T10:00:00+00:00"
        assert view.latest_evidence_source_type == "ir_press"
        assert view.citations[0].source_label == "Guidance update"
        assert view.citations[0].source_url == "https://example.com/press/guidance"

        md = render_report(view)
        assert "**Latest Evidence:** ir_press (2026-04-17T10:00:00+00:00)" in md
        assert "Guidance update / 2026-04-17T10:00:00+00:00 - ir_press" in md
        assert "https://example.com/press/guidance" in md


class TestExpectationsGapSection:
    """The expectations gap leads the memo when the bucket is reliable."""

    def test_expectations_gap_section_renders_before_valuation_summary(self):
        from app.research.report_renderer import ReportView, render_report

        view = ReportView(
            ticker="TEST",
            verdict="PROCEED",
            expectations_gap_bucket="CHEAP_VS_EXPECTATIONS",
            implied_growth=0.04,
            supportable_growth=0.10,
            expectations_gap=-0.06,
        )
        md = render_report(view)
        gap_idx = md.index("## Expectations Gap")
        val_idx = md.index("## Valuation Summary")
        assert gap_idx < val_idx

    def test_expectations_gap_section_contains_exact_line(self):
        from app.research.report_renderer import ReportView, render_report

        view = ReportView(
            ticker="TEST",
            verdict="PROCEED",
            expectations_gap_bucket="CHEAP_VS_EXPECTATIONS",
            implied_growth=0.04,
            supportable_growth=0.10,
            expectations_gap=-0.06,
        )
        md = render_report(view)
        assert (
            "Market implies 4% growth; supportable 10%; gap -6% (CHEAP_VS_EXPECTATIONS)"
            in md
        )

    def test_expensive_bucket_renders_expensive_not_cheap_interpretation(self):
        # The EXPENSIVE bucket (positive gap) must NOT render the
        # negative-gap (cheap) sentence — that inverts the signal.
        from app.research.report_renderer import ReportView, render_report

        view = ReportView(
            ticker="TEST",
            verdict="PROCEED",
            expectations_gap_bucket="EXPENSIVE_VS_EXPECTATIONS",
            implied_growth=0.12,
            supportable_growth=0.06,
            expectations_gap=0.06,
        )
        md = render_report(view)
        assert "paying for MORE growth" in md
        assert "paying for LESS growth" not in md

    def test_cheap_bucket_renders_candidate_not_established_label(self):
        # Rule: gap-CHEAP is a candidate signal and
        # must be labeled as such, never as an established edge.
        from app.research.report_renderer import ReportView, render_report

        view = ReportView(
            ticker="TEST",
            verdict="PROCEED",
            expectations_gap_bucket="CHEAP_VS_EXPECTATIONS",
            implied_growth=0.04,
            supportable_growth=0.10,
            expectations_gap=-0.06,
        )
        md = render_report(view)
        assert "paying for LESS growth" in md
        assert "candidate signal under measurement, not an established edge" in md

    def test_unreliable_bucket_omits_section(self):
        from app.research.report_renderer import ReportView, render_report

        view = ReportView(
            ticker="TEST",
            verdict="PROCEED",
            expectations_gap_bucket="EXPECTATIONS_GAP_UNRELIABLE",
            implied_growth=0.60,
            supportable_growth=0.06,
            expectations_gap=None,
        )
        md = render_report(view)
        assert "## Expectations Gap" not in md

    def test_none_bucket_omits_section_and_does_not_raise(self):
        from app.research.report_renderer import ReportView, render_report

        view = ReportView(ticker="TEST", verdict="PROCEED")
        md = render_report(view)
        assert "## Expectations Gap" not in md

    def test_build_view_populates_expectations_fields_from_scorecard(self):
        report = _make_minimal_report(
            scorecard_implied_growth=0.04,
            scorecard_supportable_growth=0.10,
            scorecard_expectations_gap=-0.06,
            scorecard_expectations_gap_bucket="CHEAP_VS_EXPECTATIONS",
        )
        view = build_view(report)
        assert view.implied_growth == 0.04
        assert view.supportable_growth == 0.10
        assert view.expectations_gap == -0.06
        assert view.expectations_gap_bucket == "CHEAP_VS_EXPECTATIONS"

    def test_build_view_threads_canonical_line_from_report(self):
        # When the report carries the canonical one-liner (threaded from the
        # expectations_gap authority), build_view uses it verbatim — the renderer
        # does not reconstruct a divergent phrasing.
        report = _make_minimal_report(
            scorecard_implied_growth=0.04,
            scorecard_supportable_growth=0.10,
            scorecard_expectations_gap=-0.06,
            scorecard_expectations_gap_bucket="CHEAP_VS_EXPECTATIONS",
            scorecard_expectations_gap_line=(
                "Market implies 4% growth; supportable 10%; gap -6% (CHEAP_VS_EXPECTATIONS)"
            ),
        )
        view = build_view(report)
        assert view.expectations_gap_line == (
            "Market implies 4% growth; supportable 10%; gap -6% (CHEAP_VS_EXPECTATIONS)"
        )
        md = render_report(view)
        assert (
            "Market implies 4% growth; supportable 10%; gap -6% (CHEAP_VS_EXPECTATIONS)"
            in md
        )

    def test_build_view_recomputes_line_via_authority_when_absent(self):
        # Legacy report: line field unset but legs present -> build_view
        # recomputes the line via the canonical authority (single-authority),
        # never an ad-hoc renderer string.
        report = _make_minimal_report(
            scorecard_implied_growth=0.12,
            scorecard_supportable_growth=0.06,
            scorecard_expectations_gap=0.06,
            scorecard_expectations_gap_bucket="EXPENSIVE_VS_EXPECTATIONS",
        )
        view = build_view(report)
        assert view.expectations_gap_line == (
            "Market implies 12% growth; supportable 6%; gap 6% (EXPENSIVE_VS_EXPECTATIONS)"
        )

    def test_build_view_defaults_expectations_fields_to_none_for_legacy_report(self):
        report = _make_minimal_report()
        view = build_view(report)
        assert view.implied_growth is None
        assert view.supportable_growth is None
        assert view.expectations_gap is None
        assert view.expectations_gap_bucket is None
        assert view.expectations_gap_line is None
