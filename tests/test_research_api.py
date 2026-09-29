from __future__ import annotations

import json
from types import SimpleNamespace

from app.analyst.evidence_bundle import AnalysisEvidenceBundle, ValuationSnapshot
from app.analyst.thesis_contract import AnalysisCitation, AnalysisFinding, AnalysisReport, ValuationConclusion
from app.db import get_db, init_db
from app.research import run_research_agent_for_ticker


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_report_path_fallback_handles_diagnostic_report_suffix():
    from app.research import _artifact_path_from_report_path

    assert (
        _artifact_path_from_report_path("data/outputs/research/AAA_2026-04-17_abc_diagnostic_report.md")
        == "data/outputs/research/AAA_2026-04-17_abc.json"
    )
    assert (
        _artifact_path_from_report_path("data/outputs/research/AAA_2026-04-17_abc_report.md")
        == "data/outputs/research/AAA_2026-04-17_abc.json"
    )


def test_run_research_agent_for_ticker_materializes_analyst_outputs(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    research_path = cfg.outputs_dir / "research" / "AAA_2026-04-17.json"
    research_path.parent.mkdir(parents=True, exist_ok=True)
    research_path.write_text("{}", encoding="utf-8")
    legacy_packet_path = cfg.research_dir / "AAA_run_legacy.json"

    report = SimpleNamespace(
        ticker="AAA",
        as_of_date="2026-04-17",
        artifact_path=str(research_path),
        warnings=[],
        scorecard_dcf=140.0,
        scorecard_epv=120.0,
        scorecard_graham=90.0,
        scorecard_price=100.0,
        methods_agree=True,
        tension_type="NONE",
        gate_action="PROCEED",
        solvency_status="LOW",
        filing_risk_status="OK",
        completed_at="2026-04-17T12:00:00+00:00",
        conviction_class="HIGH",
        conviction_score=81,
        thesis=SimpleNamespace(adjusted_intrinsic_mid=140.0),
    )
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
    )
    analysis_report = AnalysisReport(
        analysis_id="AAA_2026-04-17_20260417T1200000000",
        ticker="AAA",
        as_of_date="2026-04-17",
        generated_at="2026-04-17T12:00:00+00:00",
        verdict="BUY",
        confidence_label="HIGH",
        confidence_score=81,
        thesis_summary="AAA screens as BUY with resilient economics.",
        valuation=ValuationConclusion(
            price=100.0,
            base_case_value=140.0,
            bear_case_value=110.0,
            bull_case_value=170.0,
            margin_of_safety=0.29,
        ),
        positives=[
            AnalysisFinding(
                finding_id="P1",
                category="POSITIVE",
                claim="Retention remains strong.",
                direction="BULLISH",
                severity="HIGH",
                source_basis="filings",
                citation_ids=["C1"],
            )
        ],
        citations=[
            AnalysisCitation(
                citation_id="C1",
                source_type="event",
                source_label="8-K Item 5.02",
                source_date="2026-04-10",
                section="leadership_governance",
                excerpt="The board appointed a new chief executive officer.",
                source_form_type="8-K",
                source_accession="0000000000-26-000101",
                source_role="material_event",
                item_code="5.02",
                event_category="leadership_governance",
            )
        ],
    )

    monkeypatch.setattr(
        "app.research.deep_research.run_deep_research",
        lambda ticker, as_of_date=None, years=5, quarters=0: report,
    )
    monkeypatch.setattr(
        "app.analyst.materializer._load_scorecard",
        lambda ticker, as_of_date=None: (
            {
                "pricing_zone_detail": {"current_price": 100.0, "dcf_base": 140.0, "epv_adjusted": 120.0},
                "discounts": {"graham": 0.1111111111},
                "quality_context": {"gate_action": "PROCEED"},
            },
            "2026-04-17",
        ),
    )
    monkeypatch.setattr(
        "app.analyst.materializer.build_analysis_evidence_bundle_from_cached_scorecard",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        "app.analyst.materializer.build_analysis_report_from_research",
        lambda built_bundle, built_report: analysis_report,
    )

    def _seed_legacy_packet(*args, **kwargs):
        payload = {
            "quality": {
                "coverage_score": 25.0,
                "freshness_score": 20.0,
                "gap_score": 10.0,
                "overall_research_score": 18.0,
                "incomplete": True,
                "top_gaps": [
                    {
                        "severity": "high",
                        "summary": "Missing recent filings",
                        "recommended_action": "Review the latest filing set.",
                    }
                ],
            },
            "evidence_items": [
                {"source_type": "filing", "source_url": "https://example.com/1", "excerpt_text": "Excerpt 1"},
                {"source_type": "filing", "source_url": "https://example.com/2", "excerpt_text": "Excerpt 2"},
            ],
        }
        legacy_packet_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_packet_path.write_text(json.dumps(payload), encoding="utf-8")
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO research_packets(ticker, as_of_date, run_id, packet_path, packet_hash, created_at)
                VALUES(?, ?, ?, ?, 'hash', datetime('now'))
                """,
                ("AAA", "2026-04-17", "run_legacy", str(legacy_packet_path)),
            )
        return legacy_packet_path

    monkeypatch.setattr("app.research.run_legacy_research_packet_for_ticker", _seed_legacy_packet)

    path = run_research_agent_for_ticker("AAA", as_of_date="2026-04-17")

    assert path == research_path

    out_dir = cfg.analyst_outputs_dir / "AAA_2026-04-17"
    bundle_path = out_dir / "analysis_evidence_bundle.json"
    report_json_path = out_dir / "analysis_report.json"
    report_md_path = out_dir / "analysis_report.md"

    assert bundle_path.exists()
    assert report_json_path.exists()
    assert report_md_path.exists()
    assert json.loads(bundle_path.read_text(encoding="utf-8"))["ticker"] == "AAA"
    assert json.loads(report_json_path.read_text(encoding="utf-8"))["verdict"] == "BUY"
    saved_report = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert saved_report["research_quality"]["overall_score"] == 18.0
    assert saved_report["research_quality"]["evidence_count"] == 2
    assert saved_report["research_quality"]["top_gaps"][0]["recommended_action"] == "Review the latest filing set."
    assert saved_report["citations"][0]["source_form_type"] == "8-K"
    assert saved_report["citations"][0]["item_code"] == "5.02"
    assert saved_report["citations"][0]["event_category"] == "leadership_governance"

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT output_type, output_path
            FROM analyst_outputs
            WHERE ticker = ? AND as_of_date = ?
            ORDER BY output_type
            """,
            ("AAA", "2026-04-17"),
        ).fetchall()

    assert [row["output_type"] for row in rows] == [
        "analysis_evidence_bundle",
        "analysis_report",
        "analysis_report_markdown",
    ]
    assert rows[1]["output_path"].endswith("analysis_report.json")


def test_run_research_agent_for_ticker_warns_when_quality_unavailable(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    research_path = cfg.outputs_dir / "research" / "BBB_2026-04-17.json"
    research_path.parent.mkdir(parents=True, exist_ok=True)
    research_path.write_text("{}", encoding="utf-8")

    report = SimpleNamespace(
        ticker="BBB",
        as_of_date="2026-04-17",
        artifact_path=str(research_path),
        warnings=[],
        scorecard_dcf=140.0,
        scorecard_epv=120.0,
        scorecard_graham=90.0,
        scorecard_price=100.0,
        methods_agree=True,
        tension_type="NONE",
        gate_action="PROCEED",
        solvency_status="LOW",
        filing_risk_status="OK",
        completed_at="2026-04-17T12:00:00+00:00",
        conviction_class="HIGH",
        conviction_score=81,
        thesis=SimpleNamespace(adjusted_intrinsic_mid=140.0),
    )
    bundle = AnalysisEvidenceBundle(
        ticker="BBB",
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
    )
    analysis_report = AnalysisReport(
        analysis_id="BBB_2026-04-17_20260417T1200000000",
        ticker="BBB",
        as_of_date="2026-04-17",
        generated_at="2026-04-17T12:00:00+00:00",
        verdict="BUY",
        confidence_label="HIGH",
        confidence_score=81,
        thesis_summary="BBB screens as BUY with resilient economics.",
        valuation=ValuationConclusion(
            price=100.0,
            base_case_value=140.0,
            bear_case_value=110.0,
            bull_case_value=170.0,
            margin_of_safety=0.29,
        ),
    )

    monkeypatch.setattr(
        "app.research.deep_research.run_deep_research",
        lambda ticker, as_of_date=None, years=5, quarters=0: report,
    )
    monkeypatch.setattr(
        "app.analyst.materializer._load_scorecard",
        lambda ticker, as_of_date=None: (
            {
                "pricing_zone_detail": {"current_price": 100.0, "dcf_base": 140.0, "epv_adjusted": 120.0},
                "discounts": {"graham": 0.1111111111},
                "quality_context": {"gate_action": "PROCEED"},
            },
            "2026-04-17",
        ),
    )
    monkeypatch.setattr(
        "app.analyst.materializer.build_analysis_evidence_bundle_from_cached_scorecard",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        "app.analyst.materializer.build_analysis_report_from_research",
        lambda built_bundle, built_report: analysis_report,
    )
    monkeypatch.setattr("app.research.run_legacy_research_packet_for_ticker", lambda *args, **kwargs: None)

    path = run_research_agent_for_ticker("BBB", as_of_date="2026-04-17")

    assert path == research_path
    report_json_path = cfg.analyst_outputs_dir / "BBB_2026-04-17" / "analysis_report.json"
    saved_report = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert saved_report["research_quality"] is None
    assert "research_quality_unavailable" in saved_report["warnings"]


def test_run_research_agent_for_ticker_passes_quarter_scope_to_runtime(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    calls: list[tuple[int, int]] = []
    research_path = tmp_path / "research.json"
    research_path.write_text("{}", encoding="utf-8")

    report = SimpleNamespace(
        ticker="CCC",
        as_of_date="2026-04-17",
        artifact_path=str(research_path),
        warnings=[],
        scorecard_dcf=140.0,
        scorecard_epv=120.0,
        scorecard_graham=90.0,
        scorecard_price=100.0,
        methods_agree=True,
        tension_type="NONE",
        gate_action="PROCEED",
        solvency_status="LOW",
        filing_risk_status="OK",
        completed_at="2026-04-17T12:00:00+00:00",
        conviction_class="HIGH",
        conviction_score=81,
        thesis=SimpleNamespace(adjusted_intrinsic_mid=140.0),
    )

    monkeypatch.setattr(
        "app.research.deep_research.run_deep_research",
        lambda ticker, as_of_date=None, years=5, quarters=0: calls.append((years, quarters)) or report,
    )
    monkeypatch.setattr(
        "app.analyst.materializer.materialize_analysis_outputs_from_research",
        lambda report, years=5, quarters=0: calls.append((years, quarters)),
    )
    monkeypatch.setattr("app.research.run_legacy_research_packet_for_ticker", lambda *args, **kwargs: None)

    run_research_agent_for_ticker("CCC", as_of_date="2026-04-17", years=6, quarters=2)

    assert calls == [(6, 2), (6, 2)]
