from __future__ import annotations

from types import SimpleNamespace

from app.analyst.evidence_bundle import AnalysisEvidenceBundle, BundleEvent, BundleFiling, ValuationSnapshot
from app.analyst.report_adapter import build_analysis_report_from_research
from app.analyst.report_renderer import render_summary


def test_build_analysis_report_from_research_maps_bundle_and_view(monkeypatch):
    bundle = AnalysisEvidenceBundle(
        ticker="AAA",
        as_of_date="2026-04-17",
        built_at="2026-04-17T12:00:00+00:00",
        analysis_years=5,
        analysis_quarters=0,
        freshness_window_days=90,
        valuation=ValuationSnapshot(
            current_price=100.0,
            market_cap=5000.0,
            dcf_base=140.0,
            epv_adjusted=120.0,
            graham_value=90.0,
            methods_agree=True,
            tension_type="GROWTH_VS_EARNINGS_POWER",
            gate_action="PROCEED",
            solvency_status="LOW",
            filing_risk_status="OK",
        ),
        filings=[
            BundleFiling(
                form_type="10-K",
                filing_date="2026-02-10",
                accession="0000000000-26-000001",
                role="annual",
                sections_included=["mda"],
                section_text={"mda": "Revenue accelerated."},
            )
        ],
        recent_events=[
            BundleEvent(
                source_type="8-K",
                published_at="2026-04-15",
                title="Product launch",
                summary="Management announced a new product launch.",
                source_url="https://example.com/event",
                materiality="HIGH",
                accession="0000000000-26-000999",
                item_code="5.02",
                event_category="leadership_governance",
            )
        ],
        warnings=["bundle_warning"],
    )

    fake_view = SimpleNamespace(
        methods=[
            SimpleNamespace(adjusted=150.0, original=140.0),
            SimpleNamespace(adjusted=None, original=115.0),
        ],
        current_price=100.0,
        adjusted_margin_of_safety=0.18,
        tension_type="GROWTH_VS_EARNINGS_POWER",
        tension_explanation="Market over-weights near-term weakness.",
            adjustments=[
                SimpleNamespace(
                    direction="BULLISH",
                    claim="Margin expansion is durable",
                    magnitude=6.0,
                    citation_handles=["[C1]"],
                ),
                SimpleNamespace(
                    direction="BEARISH",
                    claim="Debt refinancing could pressure free cash flow",
                    magnitude=4.0,
                    citation_handles=[],
                ),
            ],
            hard_blockers=[],
            open_questions=[
                SimpleNamespace(
                    description="Pricing durability in the enterprise segment",
                    importance="MEDIUM",
                )
            ],
        why_wrong=[
            SimpleNamespace(
                assumption="Margin expansion persists.",
                invalidation="Margins compress back below prior-year levels.",
                impact="$10/share downside",
            )
        ],
        citations=[
            SimpleNamespace(
                citation_id="C1",
                section="MD&A",
                excerpt="Operating margin improved year over year.",
                source_form_type="10-Q",
                source_filing_date="2026-02-10",
                source_accession="0000000000-26-000001",
                source_role="quarterly",
            )
        ],
        filing_form="10-K",
        filing_date="2026-02-10",
    )

    monkeypatch.setattr("app.analyst.report_adapter.build_view", lambda report: fake_view)
    monkeypatch.setattr("app.analyst.report_adapter.derive_verdict", lambda report: "PROCEED")

    report = SimpleNamespace(
        completed_at="2026-04-17T12:00:00+00:00",
        conviction_class="HIGH",
        conviction_score=82,
        warnings=["report_warning"],
        thesis=SimpleNamespace(adjusted_intrinsic_mid=140.0),
    )

    analysis = build_analysis_report_from_research(bundle, report)

    assert analysis.analysis_id == "AAA_2026-04-17_20260417T120000_0000"
    assert analysis.verdict == "BUY"
    assert analysis.research_gate == "PROCEED"
    assert analysis.confidence_label == "HIGH"
    assert analysis.confidence_score == 82
    assert analysis.thesis_summary == (
        "AAA screens as BUY. Adjusted margin of safety is 18.0%. "
        "Primary support: Margin expansion is durable. "
        "Main uncertainty: Pricing durability in the enterprise segment."
    )
    assert analysis.valuation.base_case_value == 140.0
    assert analysis.valuation.bear_case_value == 115.0
    assert analysis.valuation.bull_case_value == 150.0
    assert analysis.valuation.expectations_gap is not None
    assert analysis.valuation.expectations_gap.key_mismatch == "Market over-weights near-term weakness."
    assert [finding.claim for finding in analysis.positives] == ["Margin expansion is durable"]
    assert [finding.claim for finding in analysis.risks] == [
        "Debt refinancing could pressure free cash flow",
    ]
    assert [finding.claim for finding in analysis.recent_event_impacts] == ["Product launch"]
    assert [question.question for question in analysis.open_questions] == [
        "Pricing durability in the enterprise segment",
    ]
    assert analysis.falsifiers[0].description == "Margins compress back below prior-year levels."
    assert [citation.citation_id for citation in analysis.citations] == ["C1", "E1"]
    assert analysis.citations[0].source_form_type == "10-Q"
    assert analysis.citations[0].source_accession == "0000000000-26-000001"
    assert analysis.citations[0].source_role == "quarterly"
    assert analysis.citations[1].source_form_type == "8-K"
    assert analysis.citations[1].source_accession == "0000000000-26-000999"
    assert analysis.citations[1].source_role == "material_event"
    assert analysis.citations[1].item_code == "5.02"
    assert analysis.citations[1].event_category == "leadership_governance"
    assert analysis.latest_evidence_date == "2026-04-15"
    assert analysis.latest_evidence_source_type == "8-K"
    assert analysis.sources_used == ["event:8-K", "filing:10-K", "valuation_snapshot"]
    assert analysis.warnings == ["bundle_warning", "report_warning"]


def test_build_analysis_report_from_research_preserves_current_event_citation_from_report(monkeypatch):
    bundle = AnalysisEvidenceBundle(
        ticker="AAA",
        as_of_date="2026-04-17",
        built_at="2026-04-17T12:00:00+00:00",
        analysis_years=5,
        analysis_quarters=0,
        freshness_window_days=90,
        valuation=ValuationSnapshot(
            current_price=100.0,
            market_cap=5000.0,
            dcf_base=140.0,
            epv_adjusted=120.0,
            graham_value=90.0,
            methods_agree=True,
            tension_type="NONE",
            gate_action="PROCEED",
            solvency_status="LOW",
            filing_risk_status="OK",
        ),
        recent_events=[
            BundleEvent(
                source_type="ir_press",
                published_at="2026-04-17T10:00:00+00:00",
                title="Guidance update",
                summary="Management raised revenue guidance.",
                source_url="https://example.com/press/guidance",
                materiality="HIGH",
                source_quality={
                    "source_family": "company_controlled",
                    "source_origin": "primary_company_controlled",
                    "source_independence": "company_controlled",
                    "source_domain": "example.com",
                    "freshness_days": 0,
                    "freshness_bucket": "same_day",
                    "source_quality_score": 0.86,
                    "calibration_status": "deterministic_heuristic",
                    "reason_codes": [
                        "SOURCE_COMPANY_CONTROLLED",
                        "SOURCE_ISSUER_BIAS_POSSIBLE",
                        "FRESHNESS_SAME_DAY",
                    ],
                },
            )
        ],
    )

    fake_view = SimpleNamespace(
        methods=[SimpleNamespace(adjusted=150.0, original=140.0)],
        current_price=100.0,
        adjusted_margin_of_safety=0.18,
        tension_type="NONE",
        tension_explanation=None,
        adjustments=[],
        hard_blockers=[],
        open_questions=[],
        why_wrong=[],
        citations=[
            SimpleNamespace(
                citation_id="C1",
                section="ir_press",
                excerpt="Management raised revenue guidance for the year.",
                source_type="ir_press",
                source_title="Guidance update",
                source_url="https://example.com/press/guidance",
                source_published_at="2026-04-17T10:00:00+00:00",
                source_quality={
                    "source_family": "company_controlled",
                    "source_origin": "primary_company_controlled",
                    "source_independence": "company_controlled",
                    "source_domain": "example.com",
                    "freshness_days": 0,
                    "freshness_bucket": "same_day",
                    "source_quality_score": 0.86,
                    "calibration_status": "deterministic_heuristic",
                    "reason_codes": [
                        "SOURCE_COMPANY_CONTROLLED",
                        "SOURCE_ISSUER_BIAS_POSSIBLE",
                        "FRESHNESS_SAME_DAY",
                    ],
                },
            )
        ],
        filing_form="10-Q",
        filing_date="2026-04-01",
    )

    monkeypatch.setattr("app.analyst.report_adapter.build_view", lambda report: fake_view)
    monkeypatch.setattr("app.analyst.report_adapter.derive_verdict", lambda report: "PROCEED")

    report = SimpleNamespace(
        completed_at="2026-04-17T12:00:00+00:00",
        conviction_class="HIGH",
        conviction_score=82,
        warnings=[],
        thesis=SimpleNamespace(adjusted_intrinsic_mid=140.0),
        citations=fake_view.citations,
    )

    analysis = build_analysis_report_from_research(bundle, report)

    assert analysis.citations[0].source_type == "ir_press"
    assert analysis.citations[0].source_label == "Guidance update"
    assert analysis.citations[0].source_date == "2026-04-17T10:00:00+00:00"
    assert analysis.citations[0].source_url == "https://example.com/press/guidance"
    assert analysis.citations[0].source_quality["source_family"] == "company_controlled"
    assert "event:ir_press" in analysis.sources_used


def test_wes_shaped_incomplete_proceed_report_fails_closed_to_watch(monkeypatch):
    bundle = AnalysisEvidenceBundle(
        ticker="WES",
        as_of_date="2026-07-21",
        built_at="2026-07-21T12:00:00+00:00",
        analysis_years=5,
        analysis_quarters=4,
        freshness_window_days=90,
        valuation=ValuationSnapshot(
            current_price=None,
            market_cap=None,
            dcf_base=None,
            epv_adjusted=None,
            graham_value=None,
            methods_agree=None,
            tension_type=None,
            gate_action="PROCEED",
            solvency_status="LOW",
            filing_risk_status="OK",
        ),
        warnings=[
            "quarterly_filings_partial:0/4",
            "current_event_source_disabled:ir_press",
            "current_event_source_disabled:company_news",
            "research_quality_unavailable",
        ],
    )
    fake_view = SimpleNamespace(
        methods=[],
        current_price=None,
        adjusted_margin_of_safety=None,
        tension_type="NONE",
        tension_explanation=None,
        adjustments=[],
        hard_blockers=[],
        open_questions=[],
        why_wrong=[],
        citations=[],
        filing_form="10-K",
        filing_date="2026-02-20",
    )
    monkeypatch.setattr("app.analyst.report_adapter.build_view", lambda report: fake_view)
    monkeypatch.setattr("app.analyst.report_adapter.derive_verdict", lambda report: "PROCEED")

    analysis = build_analysis_report_from_research(
        bundle,
        SimpleNamespace(
            completed_at="2026-07-21T12:00:00+00:00",
            conviction_class="HIGH",
            conviction_score=75,
            warnings=[],
            thesis=None,
        ),
    )

    assert analysis.verdict == "WATCH"
    assert analysis.research_gate == "PROCEED"
    assert render_summary(analysis).splitlines()[1] == "Action: Wait"
    assert [
        warning
        for warning in analysis.warnings
        if warning.startswith("BUY_POSTURE_BLOCKED:")
    ] == [
        "BUY_POSTURE_BLOCKED:MISSING_PRICE",
        "BUY_POSTURE_BLOCKED:MISSING_MARGIN_OF_SAFETY",
        "BUY_POSTURE_BLOCKED:MISSING_RISKS",
        "BUY_POSTURE_BLOCKED:MISSING_FALSIFIERS",
        "BUY_POSTURE_BLOCKED:RESEARCH_QUALITY_UNAVAILABLE",
        "BUY_POSTURE_BLOCKED:CURRENT_CONTEXT_INCOMPLETE",
    ]
