"""Tests for app.analyst contract layer (Redirect Task 1).

Covers construction and round-trip serialization for:
- AnalysisEvidenceBundle and nested types (input contract)
- AnalysisReport and nested types (output contract)
"""

from app.analyst.evidence_bundle import (
    AnalysisEvidenceBundle,
    BundleEvent,
    BundleFiling,
    PriorThesisSnapshot,
    ValuationSnapshot,
)
from app.analyst.thesis_contract import (
    AnalysisCitation,
    AnalysisFinding,
    AnalysisReport,
    ExpectationsGap,
    Falsifier,
    OpenQuestion,
    ResearchGapSummary,
    ResearchQualitySummary,
    ValuationConclusion,
)


# ---------------------------------------------------------------------------
# ValuationSnapshot
# ---------------------------------------------------------------------------


def test_valuation_snapshot_all_none_fields():
    snapshot = ValuationSnapshot(
        current_price=None,
        market_cap=None,
        dcf_base=None,
        epv_adjusted=None,
        graham_value=None,
        methods_agree=None,
        tension_type=None,
        gate_action=None,
        solvency_status=None,
        filing_risk_status=None,
    )
    assert snapshot.current_price is None
    assert snapshot.dcf_base is None
    assert snapshot.methods_agree is None


def test_valuation_snapshot_populated_fields():
    snapshot = ValuationSnapshot(
        current_price=45.50,
        market_cap=12000.0,
        dcf_base=62.10,
        epv_adjusted=48.75,
        graham_value=55.00,
        methods_agree=True,
        tension_type="NONE",
        gate_action="PROCEED",
        solvency_status="OK",
        filing_risk_status="OK",
    )
    assert snapshot.current_price == 45.50
    assert snapshot.methods_agree is True
    assert snapshot.gate_action == "PROCEED"


def test_valuation_snapshot_round_trip():
    snapshot = ValuationSnapshot(
        current_price=45.50,
        market_cap=12000.0,
        dcf_base=62.10,
        epv_adjusted=48.75,
        graham_value=55.00,
        methods_agree=True,
        tension_type="NONE",
        gate_action="PROCEED",
        solvency_status="OK",
        filing_risk_status="OK",
    )
    restored = ValuationSnapshot.from_dict(snapshot.to_dict())
    assert restored == snapshot


def test_valuation_snapshot_none_round_trip():
    snapshot = ValuationSnapshot(
        current_price=None,
        market_cap=None,
        dcf_base=None,
        epv_adjusted=None,
        graham_value=None,
        methods_agree=None,
        tension_type=None,
        gate_action=None,
        solvency_status=None,
        filing_risk_status=None,
    )
    restored = ValuationSnapshot.from_dict(snapshot.to_dict())
    assert restored == snapshot


# ---------------------------------------------------------------------------
# BundleFiling
# ---------------------------------------------------------------------------


def test_bundle_filing_constructs_with_required_fields():
    filing = BundleFiling(
        form_type="10-K",
        filing_date="2025-02-14",
        accession="0001234567-25-000001",
        role="annual",
        sections_included=["mda", "risk_factors"],
        section_text={"mda": "Management discussion...", "risk_factors": "Risk..."},
    )
    assert filing.form_type == "10-K"
    assert filing.role == "annual"
    assert filing.sections_included == ["mda", "risk_factors"]
    assert filing.section_text["mda"].startswith("Management")


def test_bundle_filing_round_trip():
    filing = BundleFiling(
        form_type="10-Q",
        filing_date="2025-08-07",
        accession=None,
        role="quarterly",
        sections_included=["mda"],
        section_text={"mda": "Quarterly update..."},
    )
    restored = BundleFiling.from_dict(filing.to_dict())
    assert restored == filing


# ---------------------------------------------------------------------------
# BundleEvent
# ---------------------------------------------------------------------------


def test_bundle_event_constructs_with_required_fields():
    event = BundleEvent(
        source_type="8-K",
        published_at="2025-09-12",
        title="Material agreement filed",
        summary="Company signed a supply agreement.",
        source_url="https://www.sec.gov/archive/example.htm",
        materiality="HIGH",
        accession="0001234567-25-000101",
        item_code="1.01",
        event_category="strategic_transaction",
    )
    assert event.source_type == "8-K"
    assert event.materiality == "HIGH"
    assert event.item_code == "1.01"
    assert event.event_category == "strategic_transaction"


def test_bundle_event_round_trip():
    event = BundleEvent(
        source_type="8-K",
        published_at=None,
        title="Earnings release",
        summary="Q2 results.",
        source_url=None,
        materiality=None,
        accession="0001234567-25-000202",
        item_code="2.02",
        event_category="results_guidance",
        source_quality={
            "source_family": "regulatory_filing",
            "freshness_bucket": "current_30d",
            "source_quality_score": 1.0,
        },
    )
    restored = BundleEvent.from_dict(event.to_dict())
    assert restored == event


# ---------------------------------------------------------------------------
# PriorThesisSnapshot
# ---------------------------------------------------------------------------


def test_prior_thesis_snapshot_constructs_with_required_fields():
    prior = PriorThesisSnapshot(
        as_of_date="2025-11-01",
        verdict="WATCH",
        thesis_summary="Market was pricing in too much risk...",
        key_risks=["cyclical exposure", "capex overhang"],
        falsifiers=["gross margin dropping below 30%"],
        report_path="data/outputs/thesis/EXAMPLE-2025-11-01.md",
    )
    assert prior.verdict == "WATCH"
    assert "cyclical exposure" in prior.key_risks


def test_prior_thesis_snapshot_round_trip():
    prior = PriorThesisSnapshot(
        as_of_date="2025-11-01",
        verdict="BUY",
        thesis_summary="Thesis summary",
        key_risks=["risk a"],
        falsifiers=["falsifier b"],
        report_path=None,
    )
    restored = PriorThesisSnapshot.from_dict(prior.to_dict())
    assert restored == prior


# ---------------------------------------------------------------------------
# AnalysisEvidenceBundle
# ---------------------------------------------------------------------------


def _valuation_snapshot_for_tests() -> ValuationSnapshot:
    return ValuationSnapshot(
        current_price=45.50,
        market_cap=12000.0,
        dcf_base=62.10,
        epv_adjusted=48.75,
        graham_value=55.00,
        methods_agree=True,
        tension_type="NONE",
        gate_action="PROCEED",
        solvency_status="OK",
        filing_risk_status="OK",
    )


def test_evidence_bundle_constructs_with_minimal_fields():
    bundle = AnalysisEvidenceBundle(
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        built_at="2026-04-10T12:00:00Z",
        analysis_years=5,
        analysis_quarters=4,
        freshness_window_days=90,
        valuation=_valuation_snapshot_for_tests(),
    )
    assert bundle.ticker == "EXAMPLE"
    assert bundle.filings == []
    assert bundle.recent_events == []
    assert bundle.prior_thesis is None
    assert bundle.warnings == []


def test_evidence_bundle_repeated_fields_default_to_empty_lists():
    bundle = AnalysisEvidenceBundle(
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        built_at="2026-04-10T12:00:00Z",
        analysis_years=5,
        analysis_quarters=4,
        freshness_window_days=90,
        valuation=_valuation_snapshot_for_tests(),
    )
    assert isinstance(bundle.filings, list) and bundle.filings == []
    assert isinstance(bundle.recent_events, list) and bundle.recent_events == []
    assert isinstance(bundle.warnings, list) and bundle.warnings == []


def test_evidence_bundle_minimal_round_trip():
    bundle = AnalysisEvidenceBundle(
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        built_at="2026-04-10T12:00:00Z",
        analysis_years=5,
        analysis_quarters=4,
        freshness_window_days=90,
        valuation=_valuation_snapshot_for_tests(),
    )
    restored = AnalysisEvidenceBundle.from_dict(bundle.to_dict())
    assert restored == bundle
    assert restored.prior_thesis is None
    assert restored.filings == []
    assert restored.recent_events == []
    assert restored.warnings == []


def test_evidence_bundle_full_round_trip():
    filing = BundleFiling(
        form_type="10-K",
        filing_date="2025-02-14",
        accession="0001234567-25-000001",
        role="annual",
        sections_included=["mda", "risk_factors"],
        section_text={"mda": "mda text", "risk_factors": "risks text"},
    )
    event = BundleEvent(
        source_type="8-K",
        published_at="2025-09-12",
        title="Material agreement filed",
        summary="Company signed a supply agreement.",
        source_url="https://www.sec.gov/archive/example.htm",
        materiality="HIGH",
        accession="0001234567-25-000101",
        item_code="1.01",
        event_category="strategic_transaction",
    )
    prior = PriorThesisSnapshot(
        as_of_date="2025-11-01",
        verdict="WATCH",
        thesis_summary="Prior thesis summary",
        key_risks=["cyclical exposure"],
        falsifiers=["gross margin drop"],
        report_path="data/outputs/thesis/EXAMPLE-2025-11-01.md",
    )
    bundle = AnalysisEvidenceBundle(
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        built_at="2026-04-10T12:00:00Z",
        analysis_years=5,
        analysis_quarters=4,
        freshness_window_days=90,
        valuation=_valuation_snapshot_for_tests(),
        filings=[filing],
        recent_events=[event],
        prior_thesis=prior,
        warnings=["missing quarter data"],
    )
    restored = AnalysisEvidenceBundle.from_dict(bundle.to_dict())
    assert restored == bundle
    assert len(restored.filings) == 1
    assert restored.filings[0].section_text["mda"] == "mda text"
    assert restored.recent_events[0].materiality == "HIGH"
    assert restored.prior_thesis is not None
    assert restored.prior_thesis.verdict == "WATCH"
    assert restored.warnings == ["missing quarter data"]


def test_evidence_bundle_prior_thesis_none_survives_round_trip():
    bundle = AnalysisEvidenceBundle(
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        built_at="2026-04-10T12:00:00Z",
        analysis_years=5,
        analysis_quarters=4,
        freshness_window_days=90,
        valuation=_valuation_snapshot_for_tests(),
        prior_thesis=None,
    )
    restored = AnalysisEvidenceBundle.from_dict(bundle.to_dict())
    assert restored.prior_thesis is None


# ---------------------------------------------------------------------------
# AnalysisCitation
# ---------------------------------------------------------------------------


def test_analysis_citation_constructs_with_required_fields():
    citation = AnalysisCitation(
        citation_id="C1",
        source_type="filing",
        source_label="10-K FY2025",
        source_date="2025-02-14",
        section="mda",
        excerpt="Revenue grew 12% year over year.",
        source_url=None,
    )
    assert citation.citation_id == "C1"
    assert citation.section == "mda"


def test_analysis_citation_round_trip():
    citation = AnalysisCitation(
        citation_id="C1",
        source_type="filing",
        source_label="10-K FY2025",
        source_date="2025-02-14",
        section="mda",
        excerpt="Revenue grew 12% year over year.",
        source_form_type="8-K",
        source_accession="0001234567-26-000301",
        source_role="material_event",
        item_code="5.02",
        event_category="leadership_governance",
        source_quality={
            "source_family": "regulatory_filing",
            "freshness_bucket": "current_30d",
            "source_quality_score": 1.0,
        },
    )
    restored = AnalysisCitation.from_dict(citation.to_dict())
    assert restored == citation


# ---------------------------------------------------------------------------
# AnalysisFinding
# ---------------------------------------------------------------------------


def test_analysis_finding_constructs_with_required_fields():
    finding = AnalysisFinding(
        finding_id="F1",
        category="RISK",
        claim="Customer concentration at 35% of revenue",
        direction="BEARISH",
        severity="HIGH",
        source_basis="filings",
        citation_ids=["C1", "C2"],
    )
    assert finding.category == "RISK"
    assert finding.citation_ids == ["C1", "C2"]


def test_analysis_finding_citation_ids_default_to_empty_list():
    finding = AnalysisFinding(
        finding_id="F1",
        category="POSITIVE",
        claim="Operating margin expanded",
        direction="BULLISH",
        severity="MODERATE",
        source_basis="filings",
    )
    assert finding.citation_ids == []


def test_analysis_finding_round_trip():
    finding = AnalysisFinding(
        finding_id="F1",
        category="RECENT_EVENT",
        claim="New contract signed",
        direction=None,
        severity=None,
        source_basis="recent_events",
        citation_ids=["C3"],
    )
    restored = AnalysisFinding.from_dict(finding.to_dict())
    assert restored == finding


# ---------------------------------------------------------------------------
# ExpectationsGap / OpenQuestion / Falsifier
# ---------------------------------------------------------------------------


def test_expectations_gap_constructs_with_required_fields():
    gap = ExpectationsGap(
        market_implied_view="Market pricing in permanent margin compression",
        analyst_view="Margin compression is temporary and cyclical",
        key_mismatch="Structural vs cyclical interpretation",
    )
    assert gap.market_implied_view.startswith("Market")
    assert gap.key_mismatch == "Structural vs cyclical interpretation"


def test_expectations_gap_round_trip():
    gap = ExpectationsGap(
        market_implied_view="a",
        analyst_view="b",
        key_mismatch="c",
    )
    restored = ExpectationsGap.from_dict(gap.to_dict())
    assert restored == gap


def test_open_question_constructs_with_required_fields():
    q = OpenQuestion(
        question="Will backlog convert at historical rates?",
        importance="REQUIRED",
        next_step="Review Q3 conversion rate",
    )
    assert q.importance == "REQUIRED"


def test_open_question_default_next_step_is_none():
    q = OpenQuestion(
        question="Will backlog convert?",
        importance="IMPORTANT",
    )
    assert q.next_step is None


def test_open_question_round_trip():
    q = OpenQuestion(
        question="Will backlog convert?",
        importance="IMPORTANT",
        next_step=None,
    )
    restored = OpenQuestion.from_dict(q.to_dict())
    assert restored == q


def test_falsifier_constructs_with_required_fields():
    f = Falsifier(
        description="Gross margin drops below 30%",
        trigger_type="metric",
        monitoring_hint="Check each 10-Q",
    )
    assert f.trigger_type == "metric"


def test_falsifier_default_monitoring_hint_is_none():
    f = Falsifier(
        description="Major customer lost",
        trigger_type="event",
    )
    assert f.monitoring_hint is None


def test_falsifier_round_trip():
    f = Falsifier(
        description="Major customer lost",
        trigger_type="event",
        monitoring_hint=None,
    )
    restored = Falsifier.from_dict(f.to_dict())
    assert restored == f


# ---------------------------------------------------------------------------
# ValuationConclusion
# ---------------------------------------------------------------------------


def test_valuation_conclusion_constructs_with_minimal_fields():
    conclusion = ValuationConclusion(
        price=45.50,
        base_case_value=60.00,
        bear_case_value=35.00,
        bull_case_value=80.00,
        margin_of_safety=0.32,
    )
    assert conclusion.price == 45.50
    assert conclusion.expectations_gap is None


def test_valuation_conclusion_round_trip_with_gap():
    gap = ExpectationsGap(
        market_implied_view="mv",
        analyst_view="av",
        key_mismatch="km",
    )
    conclusion = ValuationConclusion(
        price=45.50,
        base_case_value=60.00,
        bear_case_value=35.00,
        bull_case_value=80.00,
        margin_of_safety=0.32,
        expectations_gap=gap,
    )
    restored = ValuationConclusion.from_dict(conclusion.to_dict())
    assert restored == conclusion
    assert restored.expectations_gap == gap


def test_valuation_conclusion_round_trip_without_gap():
    conclusion = ValuationConclusion(
        price=None,
        base_case_value=None,
        bear_case_value=None,
        bull_case_value=None,
        margin_of_safety=None,
    )
    restored = ValuationConclusion.from_dict(conclusion.to_dict())
    assert restored == conclusion
    assert restored.expectations_gap is None


# ---------------------------------------------------------------------------
# AnalysisReport
# ---------------------------------------------------------------------------


def _minimal_valuation_conclusion() -> ValuationConclusion:
    return ValuationConclusion(
        price=45.50,
        base_case_value=60.00,
        bear_case_value=35.00,
        bull_case_value=80.00,
        margin_of_safety=0.32,
    )


def test_analysis_report_constructs_with_minimal_fields():
    report = AnalysisReport(
        analysis_id="A-EXAMPLE-20260410",
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        generated_at="2026-04-10T12:05:00Z",
        verdict="BUY",
        confidence_label="HIGH",
        confidence_score=80,
        thesis_summary="The market is mispricing the cyclical recovery.",
        valuation=_minimal_valuation_conclusion(),
    )
    assert report.verdict == "BUY"
    assert report.positives == []
    assert report.risks == []
    assert report.recent_event_impacts == []
    assert report.open_questions == []
    assert report.falsifiers == []
    assert report.citations == []
    assert report.sources_used == []
    assert report.warnings == []
    assert report.prior_thesis_change_summary is None
    assert report.latest_evidence_date is None
    assert report.latest_evidence_source_type is None


def test_analysis_report_repeated_fields_default_to_empty_lists():
    report = AnalysisReport(
        analysis_id="A1",
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        generated_at="2026-04-10T12:05:00Z",
        verdict="WATCH",
        confidence_label="MODERATE",
        confidence_score=55,
        thesis_summary="Summary",
        valuation=_minimal_valuation_conclusion(),
    )
    for repeated in (
        report.positives,
        report.risks,
        report.recent_event_impacts,
        report.open_questions,
        report.falsifiers,
        report.citations,
        report.sources_used,
        report.warnings,
    ):
        assert isinstance(repeated, list) and repeated == []
    assert report.research_quality is None


def test_research_quality_summary_round_trip():
    quality = ResearchQualitySummary(
        coverage_score=25.0,
        freshness_score=20.0,
        gap_score=10.0,
        overall_score=18.0,
        incomplete=True,
        evidence_count=4,
        top_gaps=[
            ResearchGapSummary(
                severity="high",
                summary="Missing recent filings",
                recommended_action="Review the latest filing set.",
            )
        ],
    )
    restored = ResearchQualitySummary.from_dict(quality.to_dict())
    assert restored == quality
    assert restored.top_gaps[0].recommended_action == "Review the latest filing set."


def test_analysis_report_minimal_round_trip():
    report = AnalysisReport(
        analysis_id="A1",
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        generated_at="2026-04-10T12:05:00Z",
        verdict="STAY_AWAY",
        confidence_label="LOW",
        confidence_score=20,
        thesis_summary="Summary",
        valuation=_minimal_valuation_conclusion(),
    )
    restored = AnalysisReport.from_dict(report.to_dict())
    assert restored == report
    assert restored.valuation.expectations_gap is None


def test_analysis_report_full_round_trip():
    citation = AnalysisCitation(
        citation_id="C1",
        source_type="filing",
        source_label="10-K FY2025",
        source_date="2025-02-14",
        section="mda",
        excerpt="Revenue grew 12%.",
    )
    positive = AnalysisFinding(
        finding_id="F1",
        category="POSITIVE",
        claim="Revenue growth",
        direction="BULLISH",
        severity="MODERATE",
        source_basis="filings",
        citation_ids=["C1"],
    )
    risk = AnalysisFinding(
        finding_id="F2",
        category="RISK",
        claim="Customer concentration",
        direction="BEARISH",
        severity="HIGH",
        source_basis="filings",
        citation_ids=["C1"],
    )
    open_q = OpenQuestion(
        question="Will backlog convert?",
        importance="REQUIRED",
        next_step="Review Q3 conversion",
    )
    falsifier = Falsifier(
        description="Gross margin drops below 30%",
        trigger_type="metric",
        monitoring_hint="Check each 10-Q",
    )
    gap = ExpectationsGap(
        market_implied_view="Market pricing permanent decline",
        analyst_view="Decline is cyclical",
        key_mismatch="Structural vs cyclical",
    )
    valuation = ValuationConclusion(
        price=45.50,
        base_case_value=60.00,
        bear_case_value=35.00,
        bull_case_value=80.00,
        margin_of_safety=0.32,
        expectations_gap=gap,
    )
    report = AnalysisReport(
        analysis_id="A-EXAMPLE-20260410",
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        generated_at="2026-04-10T12:05:00Z",
        verdict="BUY",
        confidence_label="HIGH",
        confidence_score=80,
        thesis_summary="Cyclical mispricing thesis.",
        valuation=valuation,
        research_quality=ResearchQualitySummary(
            coverage_score=25.0,
            freshness_score=20.0,
            gap_score=10.0,
            overall_score=18.0,
            incomplete=True,
            evidence_count=4,
            top_gaps=[
                ResearchGapSummary(
                    severity="high",
                    summary="Missing recent filings",
                    recommended_action="Review the latest filing set.",
                )
            ],
        ),
        positives=[positive],
        risks=[risk],
        recent_event_impacts=[],
        open_questions=[open_q],
        falsifiers=[falsifier],
        citations=[citation],
        prior_thesis_change_summary="Thesis unchanged from prior quarter.",
        latest_evidence_date="2026-04-17",
        latest_evidence_source_type="8-K",
        sources_used=["10-K FY2025", "8-K 2025-09-12"],
        warnings=[],
    )
    restored = AnalysisReport.from_dict(report.to_dict())
    assert restored == report
    assert len(restored.positives) == 1
    assert restored.positives[0].citation_ids == ["C1"]
    assert restored.risks[0].severity == "HIGH"
    assert restored.open_questions[0].question == "Will backlog convert?"
    assert restored.falsifiers[0].trigger_type == "metric"
    assert restored.citations[0].section == "mda"
    assert restored.valuation.expectations_gap is not None
    assert restored.valuation.expectations_gap.key_mismatch == "Structural vs cyclical"
    assert restored.prior_thesis_change_summary == "Thesis unchanged from prior quarter."
    assert restored.latest_evidence_date == "2026-04-17"
    assert restored.latest_evidence_source_type == "8-K"
    assert restored.research_quality is not None
    assert restored.research_quality.overall_score == 18.0
    assert restored.research_quality.top_gaps[0].summary == "Missing recent filings"


def test_analysis_report_expectations_gap_none_survives_round_trip():
    report = AnalysisReport(
        analysis_id="A1",
        ticker="EXAMPLE",
        as_of_date="2026-04-10",
        generated_at="2026-04-10T12:05:00Z",
        verdict="WATCH",
        confidence_label="MODERATE",
        confidence_score=55,
        thesis_summary="Summary",
        valuation=_minimal_valuation_conclusion(),
    )
    restored = AnalysisReport.from_dict(report.to_dict())
    assert restored.valuation.expectations_gap is None
    assert restored.prior_thesis_change_summary is None
