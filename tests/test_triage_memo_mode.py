from __future__ import annotations

import json
from pathlib import Path

from app.analyst.thesis_contract import (
    AnalysisCitation,
    AnalysisFinding,
    AnalysisReport,
    Falsifier,
    OpenQuestion,
    ResearchGapSummary,
    ResearchQualitySummary,
    ValuationConclusion,
)
from app.db import get_db, init_db, utc_now_iso
from app.report.memo_builder import build_top_memos


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _seed_packet_and_score(cfg, ticker: str, as_of_date: str, score: float = 55.0, run_id: str = "seed_run"):
    packet = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "filings_used": [
            {
                "accession": "0000320193-26-000001",
                "form_type": "10-Q",
                "filing_date": as_of_date,
                "period_end": "2025-12-31",
                "primary_doc_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
            }
        ],
        "extracted_facts": [
            {
                "fact_type": "shares_outstanding",
                "value": {"value": 15000000000},
                "citation": {
                    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    "snippet": "shares outstanding",
                    "section_label": "cover_page",
                },
            }
        ],
        "financials": [
            {
                "statement_type": "income",
                "line_item": "revenue",
                "value": 1000.0,
                "units": "USD",
                "period": "2025Q4",
                "citation": {
                    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    "snippet": "revenue snippet",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "income",
                "line_item": "operating_income",
                "value": 200.0,
                "units": "USD",
                "period": "2025Q4",
                "citation": {
                    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    "snippet": "operating income snippet",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "income",
                "line_item": "net_income",
                "value": 120.0,
                "units": "USD",
                "period": "2025Q4",
                "citation": {
                    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    "snippet": "net income snippet",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "cashflow",
                "line_item": "cfo",
                "value": 250.0,
                "units": "USD",
                "period": "2025Q4",
                "citation": {
                    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    "snippet": "cfo snippet",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "cashflow",
                "line_item": "capex",
                "value": 50.0,
                "units": "USD",
                "period": "2025Q4",
                "citation": {
                    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    "snippet": "capex snippet",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "balance",
                "line_item": "cash",
                "value": 600.0,
                "units": "USD",
                "period": "2025Q4",
                "citation": {
                    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    "snippet": "cash snippet",
                    "section_label": "financials",
                },
            },
            {
                "statement_type": "balance",
                "line_item": "total_debt",
                "value": 300.0,
                "units": "USD",
                "period": "2025Q4",
                "citation": {
                    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    "snippet": "debt snippet",
                    "section_label": "financials",
                },
            },
        ],
        "fundamentals": {
            "revenue": 1000.0,
            "revenue_growth": "UNKNOWN",
            "gross_margin": 0.4,
            "operating_margin": 0.2,
            "net_income": 120.0,
            "cfo": 250.0,
            "capex": 50.0,
            "fcf": 200.0,
            "fcf_margin": 0.2,
            "net_debt": -300.0,
            "liquidity_stress_score": 2,
            "sbc_proxy_flag": "UNKNOWN",
        },
        "valuations": {
            "multiples": {
                "inputs": {},
                "outputs": {"per_share_range": {"low": 10.0, "base": 20.0, "high": 30.0}, "confidence": "MEDIUM"},
                "warnings": [],
            },
            "dcf_lite": {
                "inputs": {},
                "outputs": {"per_share_range": {"low": 8.0, "base": 18.0, "high": 28.0}, "confidence": "LOW"},
                "warnings": [],
            },
            "reverse_dcf": {
                "inputs": {"market_price": "UNKNOWN", "price_status": "UNKNOWN"},
                "outputs": {"implied_growth": "UNKNOWN", "confidence": "LOW"},
                "warnings": [],
            },
        },
        "deltas_vs_prior_period": {},
    }

    packet_path = cfg.evidence_dir / f"{ticker}_{as_of_date}.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO evidence_packets(ticker, as_of_date, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, 'hash', ?)
            """,
            (ticker, as_of_date, str(packet_path), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            ) VALUES(?, ?, ?, '{}', ?, 'Watchlist', 1, 0, ?, '[]', ?)
            """,
            (ticker, as_of_date, run_id, score, run_id, utc_now_iso()),
        )


def _seed_analysis_report(
    cfg,
    ticker: str,
    as_of_date: str,
    *,
    verdict: str = "BUY",
    thesis_summary: str = "Analyst contract thesis.",
    overall_score: float = 18.0,
    incomplete: bool = True,
) -> Path:
    report = AnalysisReport(
        analysis_id=f"{ticker}_{as_of_date}_analysis",
        ticker=ticker,
        as_of_date=as_of_date,
        generated_at="2026-02-13T12:00:00+00:00",
        verdict=verdict,
        confidence_label="HIGH",
        confidence_score=81,
        thesis_summary=thesis_summary,
        valuation=ValuationConclusion(
            price=100.0,
            base_case_value=140.0,
            bear_case_value=110.0,
            bull_case_value=170.0,
            margin_of_safety=0.29,
        ),
        research_quality=ResearchQualitySummary(
            coverage_score=25.0,
            freshness_score=20.0,
            gap_score=10.0,
            overall_score=overall_score,
            incomplete=incomplete,
            evidence_count=2,
            top_gaps=[
                ResearchGapSummary(
                    severity="high",
                    summary="Coverage is still thin.",
                    recommended_action="Review the next filing in full.",
                )
            ],
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
                citation_ids=["C1"],
            )
        ],
        recent_event_impacts=[
            AnalysisFinding(
                finding_id="E1",
                category="RECENT_EVENT",
                claim="Management reiterated near-term guidance.",
                direction=None,
                severity="MODERATE",
                source_basis="recent_events",
                citation_ids=["C1"],
            )
        ],
        open_questions=[
            OpenQuestion(
                question="Will backlog convert into revenue this quarter?",
                importance="HIGH",
                next_step="Track the next quarterly filing.",
            )
        ],
        falsifiers=[
            Falsifier(
                description="Margins compress below recent trend.",
                trigger_type="THESIS_BREAK",
                monitoring_hint="Check the next quarterly filing.",
            )
        ],
        citations=[
            AnalysisCitation(
                citation_id="C1",
                source_type="filing",
                source_label="10-Q",
                source_date=as_of_date,
                section="mda",
                excerpt="Excerpt from the filing.",
                source_url="https://www.sec.gov/example",
            )
        ],
    )
    out_dir = cfg.analyst_outputs_dir / f"{ticker}_{as_of_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "analysis_report.json"
    report_md_path = out_dir / "analysis_report.md"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    report_md_path.write_text("# analyst report\n", encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, 'analysis_report', ?, 'hash', ?)
            """,
            (ticker, as_of_date, str(report_path), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, 'analysis_report_markdown', ?, 'hash', ?)
            """,
            (ticker, as_of_date, str(report_md_path), utc_now_iso()),
        )
    return report_path


def _seed_legacy_research_packet(cfg, ticker: str, as_of_date: str, run_id: str) -> Path:
    research_payload = {
        "findings": [{"entry_id": "F1", "summary": "Legacy finding should be ignored when report exists."}],
        "risks": [{"entry_id": "R1", "summary": "Legacy risk."}],
        "catalysts": [{"entry_id": "C1", "summary": "Legacy catalyst."}],
        "key_questions": [{"question_id": "Q1", "question": "Legacy question?"}],
        "disconfirming_evidence": [{"entry_id": "D1", "summary": "Legacy disconfirming evidence."}],
        "quality": {
            "coverage_score": 5.0,
            "freshness_score": 5.0,
            "gap_score": 30.0,
            "overall_research_score": 5.0,
            "incomplete": True,
            "top_gaps": [{"severity": "high", "summary": "Legacy gap", "recommended_action": "Legacy fix"}],
        },
        "evidence_items": [{"source_type": "filing", "source_url": "https://legacy.example", "excerpt_text": "Legacy excerpt"}],
    }
    research_path = cfg.research_dir / f"{ticker}_{run_id}.json"
    research_path.parent.mkdir(parents=True, exist_ok=True)
    research_path.write_text(json.dumps(research_payload), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO research_packets(ticker, as_of_date, run_id, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, ?, 'hash', ?)
            """,
            (ticker, as_of_date, run_id, str(research_path), utc_now_iso()),
        )
    return research_path


def test_triage_mode_builds_memo_and_gap_when_research_incomplete(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_packet_and_score(cfg, "AAPL", "2026-02-13", run_id="run_triage_test")

    built = build_top_memos(top_n=1, memo_mode="triage", run_id="run_triage_test")
    assert built >= 1

    memo_json = cfg.memos_dir / "AAPL_2026-02-13" / "memo.json"
    assert memo_json.exists()
    payload = json.loads(memo_json.read_text(encoding="utf-8"))
    assert payload["memo_mode"] == "triage"
    assert "RESEARCH_INCOMPLETE" in payload["status_flags"]

    gaps_path = cfg.gaps_dir / "AAPL_run_triage_test.json"
    assert gaps_path.exists()


def test_triage_memo_prefers_analysis_report_inputs(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_packet_and_score(cfg, "AAPL", "2026-02-13", score=55.0, run_id="run_claim_test")
    report_path = _seed_analysis_report(
        cfg,
        "AAPL",
        "2026-02-13",
        thesis_summary="Analysis report thesis wins.",
    )

    built = build_top_memos(top_n=1, memo_mode="triage", run_id="run_claim_test")
    assert built >= 1

    memo_md = (cfg.memos_dir / "AAPL_2026-02-13" / "memo.md").read_text(encoding="utf-8")
    memo_json = json.loads((cfg.memos_dir / "AAPL_2026-02-13" / "memo.json").read_text(encoding="utf-8"))
    assert "## Analyst View" in memo_md
    assert "Analysis report thesis wins." in memo_md
    assert "Margin expansion is holding." in memo_md
    assert "Track the next quarterly filing." in memo_md
    assert memo_json["analysis_source"] == "analysis_report"
    assert memo_json["analysis_report_path"] == str(report_path)
    assert memo_json["research_path"] is None


def test_triage_memo_falls_back_to_legacy_research_packet(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_packet_and_score(cfg, "AAPL", "2026-02-13", score=55.0, run_id="run_fallback_test")
    research_path = _seed_legacy_research_packet(cfg, "AAPL", "2026-02-13", "run_fallback_test")

    built = build_top_memos(top_n=1, memo_mode="triage", run_id="run_fallback_test")
    assert built >= 1

    memo_md = (cfg.memos_dir / "AAPL_2026-02-13" / "memo.md").read_text(encoding="utf-8")
    memo_json = json.loads((cfg.memos_dir / "AAPL_2026-02-13" / "memo.json").read_text(encoding="utf-8"))
    assert "legacy research fallback" in memo_md.lower()
    assert memo_json["analysis_source"] == "legacy_research_packet_fallback"
    assert memo_json["analysis_report_path"] is None
    assert memo_json["research_path"] == str(research_path)


def test_triage_memo_prefers_analysis_report_over_legacy_fallback(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_packet_and_score(cfg, "AAPL", "2026-02-13", score=55.0, run_id="run_precedence_test")
    _seed_analysis_report(
        cfg,
        "AAPL",
        "2026-02-13",
        thesis_summary="Canonical analysis report thesis.",
        incomplete=False,
    )
    _seed_legacy_research_packet(cfg, "AAPL", "2026-02-13", "run_precedence_test")

    built = build_top_memos(top_n=1, memo_mode="triage", run_id="run_precedence_test")
    assert built >= 1

    memo_md = (cfg.memos_dir / "AAPL_2026-02-13" / "memo.md").read_text(encoding="utf-8")
    memo_json = json.loads((cfg.memos_dir / "AAPL_2026-02-13" / "memo.json").read_text(encoding="utf-8"))
    assert "Canonical analysis report thesis." in memo_md
    assert "Legacy finding should be ignored when report exists." not in memo_md
    assert memo_json["analysis_source"] == "analysis_report"
    assert memo_json["research_path"] is None
