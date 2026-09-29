"""Tests for findings reconciliation (Stage C)."""
from app.research.findings_reconciliation import (
    extract_content_words,
    project_pipeline_findings,
    reconcile_findings,
)
from app.research.hypothesis_generator import Hypothesis, EvidenceNeed
from app.research.evidence_searcher import (
    EvidenceResult, EvidenceItemResult, Citation,
)
from app.research.analyst_notes import AnalystNote, AnalystNotes, AnalystCitation


def _make_hypothesis(claim="Revenue decline is structural", direction="BEARISH", source="REVENUE_DECLINE_FROM_PEAK"):
    return Hypothesis(
        claim=claim, direction=direction,
        evidence_needed=[EvidenceNeed(need_id="n1", description="revenue trend data", importance="REQUIRED")],
        falsification="Revenue recovers", priority="HIGH", source=source,
    )


def _make_evidence_result(hypothesis, hypothesis_status="CONFIRMED", item_status="CONFIRMS", section="mda"):
    """Build an EvidenceResult with correct item-level and hypothesis-level statuses."""
    citation = Citation(
        block_id=f"{section}_p0", section=section, ordinal=0,
        excerpt="Revenue declined 8%",
    )
    item = EvidenceItemResult(
        need_id="n1", needed="revenue trend data", importance="REQUIRED",
        status=item_status, classification_method="LLM",
        citations=[citation], excerpt="Revenue declined 8%",
        structured_fact=None, reasoning_short="Confirmed decline",
        candidates_considered=3, top_candidate_score=0.9,
        candidate_rankings=[],
    )
    return EvidenceResult(
        hypothesis=hypothesis,
        evidence_item_results=[item],
        hypothesis_status=hypothesis_status,
        coverage_score=1.0,
        classification_method="LLM",
    )


class TestContentWords:
    def test_basic_extraction(self):
        words = extract_content_words("Revenue grew 15% year over year")
        assert "revenue" in words
        assert "grew" in words
        assert "year" in words

    def test_excludes_short_words(self):
        words = extract_content_words("A big tax on the old car")
        assert "big" not in words
        assert "tax" not in words
        assert "the" not in words

    def test_excludes_section_labels(self):
        words = extract_content_words("mda risk_factors fin_notes analysis")
        assert "risk_factors" not in words
        assert "fin_notes" not in words
        assert "analysis" in words

    def test_empty_input(self):
        assert extract_content_words("") == set()


class TestProjection:
    def test_projects_confirmed_hypothesis(self):
        hyp = _make_hypothesis()
        er = _make_evidence_result(hyp, hypothesis_status="CONFIRMED", item_status="CONFIRMS", section="mda")
        findings = project_pipeline_findings([hyp], [er])
        assert len(findings) == 1
        assert findings[0].direction == "BEARISH"
        assert findings[0].section == "mda"
        assert findings[0].source_hypothesis == "REVENUE_DECLINE_FROM_PEAK"

    def test_projects_partially_confirmed(self):
        hyp = _make_hypothesis()
        er = _make_evidence_result(hyp, hypothesis_status="PARTIALLY_CONFIRMED", item_status="CONFIRMS", section="risk_factors")
        findings = project_pipeline_findings([hyp], [er])
        assert len(findings) == 1
        assert findings[0].section == "risk_factors"

    def test_excludes_inconclusive(self):
        hyp = _make_hypothesis()
        er = _make_evidence_result(hyp, hypothesis_status="INCONCLUSIVE", item_status="INCONCLUSIVE")
        findings = project_pipeline_findings([hyp], [er])
        assert len(findings) == 0

    def test_excludes_not_found(self):
        hyp = _make_hypothesis()
        er = _make_evidence_result(hyp, hypothesis_status="INCONCLUSIVE", item_status="NOT_FOUND")
        findings = project_pipeline_findings([hyp], [er])
        assert len(findings) == 0

    def test_section_collapse_most_frequent(self):
        hyp = _make_hypothesis()
        c1 = Citation(block_id="mda_p0", section="mda", ordinal=0, excerpt="a")
        c2 = Citation(block_id="mda_p1", section="mda", ordinal=1, excerpt="b")
        c3 = Citation(block_id="risk_factors_p0", section="risk_factors", ordinal=0, excerpt="c")
        item1 = EvidenceItemResult(
            need_id="n1", needed="q", importance="REQUIRED",
            status="CONFIRMS", classification_method="LLM",
            citations=[c1, c2, c3], excerpt="x",
            structured_fact=None, reasoning_short="ok",
            candidates_considered=3, top_candidate_score=0.9,
            candidate_rankings=[],
        )
        er = EvidenceResult(
            hypothesis=hyp, evidence_item_results=[item1],
            hypothesis_status="CONFIRMED", coverage_score=1.0,
            classification_method="LLM",
        )
        findings = project_pipeline_findings([hyp], [er])
        assert len(findings) == 1
        assert findings[0].section == "mda"

    def test_empty_inputs(self):
        assert project_pipeline_findings([], []) == []


def _make_analyst_note(claim, direction="BEARISH", section="mda", severity="HIGH"):
    return AnalystNote(
        category="RISK", claim=claim, direction=direction, severity=severity,
        citations=[AnalystCitation(section=section, excerpt="test excerpt", block_id=None)],
        suggested_adjustment=None, validation_status="VERIFIED",
    )


class TestReconciliation:
    def _make_er(self, claim, direction="BEARISH", section="mda", source="TEST_SRC"):
        hyp = _make_hypothesis(claim=claim, direction=direction, source=source)
        return _make_evidence_result(hyp, hypothesis_status="CONFIRMED", item_status="CONFIRMS", section=section)

    def test_matching_same_section_direction_shared_words(self):
        llm_note = _make_analyst_note("Revenue decline from customer concentration", direction="BEARISH", section="mda")
        er = self._make_er("Revenue decline is structural from peak", section="mda")
        analyst_notes = AnalystNotes(
            ticker="ACME", positives=[], risks=[llm_note], surprises=[],
            adjustment_triggers=[], overall_assessment="test", filing_sections_read=["mda"],
        )
        result = reconcile_findings(analyst_notes, [er.hypothesis], [er])
        assert len(result.both_paths) == 1
        assert len(result.llm_only) == 0
        assert len(result.pipeline_only) == 0

    def test_no_match_different_section(self):
        llm_note = _make_analyst_note("Revenue decline risk", section="risk_factors")
        er = self._make_er("Revenue decline structural", section="mda")
        analyst_notes = AnalystNotes(
            ticker="ACME", positives=[], risks=[llm_note], surprises=[],
            adjustment_triggers=[], overall_assessment="test", filing_sections_read=["mda", "risk_factors"],
        )
        result = reconcile_findings(analyst_notes, [er.hypothesis], [er])
        assert len(result.both_paths) == 0
        assert len(result.llm_only) == 1
        assert len(result.pipeline_only) == 1

    def test_no_match_different_direction(self):
        llm_note = _make_analyst_note("Revenue growth strong", direction="BULLISH", section="mda")
        er = self._make_er("Revenue decline structural", direction="BEARISH", section="mda")
        analyst_notes = AnalystNotes(
            ticker="ACME", positives=[llm_note], risks=[], surprises=[],
            adjustment_triggers=[], overall_assessment="test", filing_sections_read=["mda"],
        )
        result = reconcile_findings(analyst_notes, [er.hypothesis], [er])
        assert len(result.both_paths) == 0

    def test_no_match_insufficient_shared_words(self):
        llm_note = _make_analyst_note("Acquisition plans announced", section="mda")
        er = self._make_er("Revenue decline structural", section="mda")
        analyst_notes = AnalystNotes(
            ticker="ACME", positives=[], risks=[llm_note], surprises=[],
            adjustment_triggers=[], overall_assessment="test", filing_sections_read=["mda"],
        )
        result = reconcile_findings(analyst_notes, [er.hypothesis], [er])
        assert len(result.both_paths) == 0

    def test_one_to_one_matching(self):
        llm_note = _make_analyst_note("Revenue decline customer concentration risk", section="mda")
        er1 = self._make_er("Revenue decline from peak structural", section="mda", source="SRC1")
        er2 = self._make_er("Customer concentration revenue risk material", section="mda", source="SRC2")
        analyst_notes = AnalystNotes(
            ticker="ACME", positives=[], risks=[llm_note], surprises=[],
            adjustment_triggers=[], overall_assessment="test", filing_sections_read=["mda"],
        )
        result = reconcile_findings(analyst_notes, [er1.hypothesis, er2.hypothesis], [er1, er2])
        assert len(result.both_paths) == 1
        assert len(result.pipeline_only) == 1

    def test_agreement_score_calculation(self):
        llm_note = _make_analyst_note("Revenue decline structural ongoing", section="mda")
        er = self._make_er("Revenue decline from structural peak", section="mda")
        unmatched_llm = _make_analyst_note("Acquisition plans announced", section="risk_factors")
        analyst_notes = AnalystNotes(
            ticker="ACME", positives=[], risks=[llm_note, unmatched_llm], surprises=[],
            adjustment_triggers=[], overall_assessment="test", filing_sections_read=["mda", "risk_factors"],
        )
        result = reconcile_findings(analyst_notes, [er.hypothesis], [er])
        assert result.agreement_score == 0.5

    def test_no_pipeline_findings(self):
        analyst_notes = AnalystNotes(
            ticker="ACME", positives=[], risks=[], surprises=[],
            adjustment_triggers=[], overall_assessment="test", filing_sections_read=["mda"],
        )
        result = reconcile_findings(analyst_notes, [], [])
        assert result.agreement_score == 0.0

    def test_returns_none_when_analyst_notes_none(self):
        result = reconcile_findings(None, [], [])
        assert result is None
