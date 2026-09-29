from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app.analyst.evidence_bundle import AnalysisEvidenceBundle, ValuationSnapshot
from app.analyst.thesis_contract import (
    AnalysisCitation,
    AnalysisFinding,
    AnalysisReport,
    ValuationConclusion,
)
from app.cli import app
from app.db import init_db


runner = CliRunner()


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


def test_analyze_cli_writes_analyst_artifacts(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    research_report = SimpleNamespace(
        as_of_date="2026-04-17",
        status="OK",
        conviction_class="HIGH",
        conviction_score=82,
        artifact_path="data/outputs/research/AAA_2026-04-17.json",
        report_path="data/outputs/research/AAA_2026-04-17.md",
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
        confidence_score=82,
        thesis_summary="AAA screens as BUY with margin expansion support.",
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
                claim="Margin expansion is holding.",
                direction="BULLISH",
                severity="HIGH",
                source_basis="filings",
                citation_ids=["C1"],
            )
        ],
        risks=[
            AnalysisFinding(
                finding_id="R1",
                category="RISK",
                claim="Customer concentration remains elevated.",
                direction="BEARISH",
                severity="MODERATE",
                source_basis="filings",
                citation_ids=["C2"],
            )
        ],
        citations=[
            AnalysisCitation(
                citation_id="C2",
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
    out_dir = cfg.analyst_outputs_dir / "AAA_2026-04-17"
    bundle_path = out_dir / "analysis_evidence_bundle.json"
    report_json_path = out_dir / "analysis_report.json"
    report_md_path = out_dir / "analysis_report.md"
    materialized = SimpleNamespace(
        bundle=bundle,
        report=analysis_report,
        paths=SimpleNamespace(
            bundle_json=bundle_path,
            report_json=report_json_path,
            report_markdown=report_md_path,
        ),
    )

    monkeypatch.setattr(
        "app.research.deep_research.run_deep_research",
        lambda ticker, as_of_date=None, years=5, quarters=0: research_report,
    )
    monkeypatch.setattr(
        "app.analyst.materializer.materialize_analysis_outputs_from_research",
        lambda report, years=5, quarters=0: materialized,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    report_json_path.write_text(json.dumps(analysis_report.to_dict(), indent=2), encoding="utf-8")
    report_md_path.write_text("# AAA - Analyst Report\n\n## Positives\n", encoding="utf-8")

    result = runner.invoke(app, ["analyze", "AAA"])

    assert result.exit_code == 0, result.output
    assert "AAA Analyst Report (2026-04-17)" in result.output
    assert "Verdict: BUY | Confidence: HIGH (82/100)" in result.output
    assert "Analyst report JSON:" in result.output

    assert bundle_path.exists()
    assert report_json_path.exists()
    assert report_md_path.exists()

    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    report_payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    report_markdown = report_md_path.read_text(encoding="utf-8")

    assert bundle_payload["ticker"] == "AAA"
    assert report_payload["verdict"] == "BUY"
    assert report_payload["valuation"]["base_case_value"] == 140.0
    assert report_payload["citations"][0]["source_form_type"] == "8-K"
    assert report_payload["citations"][0]["item_code"] == "5.02"
    assert "# AAA - Analyst Report" in report_markdown
    assert "## Positives" in report_markdown


def test_analyze_cli_passes_quarter_scope(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    calls: list[tuple[int, int]] = []

    research_report = SimpleNamespace(
        as_of_date="2026-04-17",
        status="OK",
        conviction_class="HIGH",
        conviction_score=82,
        artifact_path="data/outputs/research/AAA_2026-04-17.json",
        report_path="data/outputs/research/AAA_2026-04-17.md",
    )

    monkeypatch.setattr(
        "app.research.deep_research.run_deep_research",
        lambda ticker, as_of_date=None, years=5, quarters=0: calls.append((years, quarters)) or research_report,
    )
    monkeypatch.setattr(
        "app.analyst.materializer.materialize_analysis_outputs_from_research",
        lambda report, years=5, quarters=0: calls.append((years, quarters)) or SimpleNamespace(
            report=SimpleNamespace(
                ticker="AAA",
                as_of_date="2026-04-17",
                verdict="BUY",
                confidence_label="HIGH",
                confidence_score=82,
                thesis_summary="Quarter-aware thesis.",
                valuation=SimpleNamespace(price=100.0, base_case_value=140.0, margin_of_safety=0.29),
                research_quality=None,
                positives=[],
                risks=[],
                recent_event_impacts=[],
                warnings=[],
            ),
            paths=SimpleNamespace(report_json=None, report_markdown=None),
        ),
    )

    result = runner.invoke(app, ["analyze", "AAA", "--quarters", "2"])

    assert result.exit_code == 0, result.output
    assert calls == [(5, 2), (5, 2)]


def test_analyze_cli_exits_nonzero_on_no_scorecard(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    research_report = SimpleNamespace(
        ticker="AAA",
        as_of_date="2026-04-17",
        status="NO_SCORECARD",
        conviction_class=None,
        conviction_score=None,
        artifact_path=None,
        report_path=None,
    )

    monkeypatch.setattr(
        "app.research.deep_research.run_deep_research",
        lambda ticker, as_of_date=None, years=5, quarters=0: research_report,
    )

    def _raise(report, years=5, quarters=0):
        raise RuntimeError("no scorecard to materialize")

    monkeypatch.setattr(
        "app.analyst.materializer.materialize_analysis_outputs_from_research",
        _raise,
    )

    result = runner.invoke(app, ["analyze", "AAA"])

    assert result.exit_code == 1, result.output
    assert "Status: NO_SCORECARD" in result.output


# Words that make a header read as an investment call.
_VERDICT_WORDS = ("Action:", "Verdict:", "STAY_AWAY", "Confidence:", "INSUFFICIENT", "Thesis:")


def test_analyze_cli_no_scorecard_says_no_valuation_not_a_verdict(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    research_report = SimpleNamespace(
        ticker="AAA",
        as_of_date="2026-04-17",
        status="NO_SCORECARD",
        conviction_class="INSUFFICIENT",
        conviction_score=0,
        artifact_path=None,
        report_path=None,
        warnings=["llm_provider_disabled", "facts_ingestion_failed: SEC_USER_AGENT_INVALID"],
    )
    monkeypatch.setattr(
        "app.research.deep_research.run_deep_research",
        lambda ticker, as_of_date=None, years=5, quarters=0: research_report,
    )

    def _forbidden(report, years=5, quarters=0):
        raise AssertionError("no analyst report may be materialized without a scorecard")

    monkeypatch.setattr(
        "app.analyst.materializer.materialize_analysis_outputs_from_research", _forbidden
    )

    result = runner.invoke(app, ["analyze", "aaa"])

    assert result.exit_code == 1, result.output
    assert result.output.splitlines()[0] == "No valuation yet for AAA: nothing to analyze."
    assert "For a valuation from free SEC data, run: ivi value AAA" in result.output
    assert "Status: NO_SCORECARD (as of 2026-04-17)" in result.output
    assert result.output.splitlines()[-3:] == [
        "Warnings:",
        "  - llm_provider_disabled",
        "  - facts_ingestion_failed: SEC_USER_AGENT_INVALID",
    ]
    for word in _VERDICT_WORDS:
        assert word not in result.output, word
    assert not cfg.analyst_outputs_dir.exists() or not any(cfg.analyst_outputs_dir.iterdir())


@pytest.mark.financial_integrity_contract
def test_analyze_cli_on_a_fresh_data_dir_reports_no_valuation(monkeypatch, tmp_path):
    # The real pipeline, nothing stubbed: an initialized data dir with no
    # filings, no scorecard and no network. This is what a new user gets. The
    # marker keeps the suite's "everything is audited" shims off, so the
    # valuation the pipeline writes for itself is (correctly) not eligible.
    cfg = _init_temp_db(monkeypatch, tmp_path)

    result = runner.invoke(app, ["analyze", "KO"])

    assert result.exit_code == 1, result.output
    assert "No valuation yet for KO: nothing to analyze." in result.output
    assert "For a valuation from free SEC data, run: ivi value KO" in result.output
    assert "Status: NO_SCORECARD" in result.output
    for word in _VERDICT_WORDS:
        assert word not in result.output, word
    assert not cfg.analyst_outputs_dir.exists() or not any(cfg.analyst_outputs_dir.iterdir())


def test_deep_research_cli_exits_nonzero_on_no_hypotheses(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    research_report = SimpleNamespace(
        ticker="AAA",
        as_of_date="2026-04-17",
        status="NO_HYPOTHESES",
        thesis=None,
        hypotheses_generated=0,
        total_adjustments=0,
        fact_calibrated_count=0,
        heuristic_count=0,
        researchable_items=[],
        not_researchable_items=[],
        conviction_class=None,
        conviction_score=None,
    )

    monkeypatch.setattr(
        "app.research.deep_research.run_deep_research",
        lambda ticker, as_of_date=None: research_report,
    )

    result = runner.invoke(app, ["deep-research", "--ticker", "AAA"])

    assert result.exit_code == 1, result.output
    assert "Status: NO_HYPOTHESES" in result.output
