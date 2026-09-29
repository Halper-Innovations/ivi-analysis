from __future__ import annotations

import json

from app.analyst.thesis_contract import AnalysisReport, ResearchGapSummary, ResearchQualitySummary, ValuationConclusion
from app.config import get_config
from app.db import get_db, init_db, utc_now_iso
from app.score.ranker import score_ticker


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


def _sample_packet(ticker: str, as_of_date: str) -> dict:
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "fundamentals": {
            "revenue": 1000,
            "operating_margin": 0.12,
            "fcf": 100,
            "fcf_margin": 0.10,
            "net_debt": 200,
            "liquidity_stress_score": 3,
        },
        "valuations": {
            "dcf_lite": {"outputs": {"confidence": "MEDIUM"}},
            "reverse_dcf": {"inputs": {"market_price": "UNKNOWN"}},
        },
        "deltas_vs_prior_period": {"revenue": 50, "fcf": 10},
        "extracted_facts": [],
    }


def _seed_evidence_packet(cfg, ticker: str, as_of_date: str):
    packet = _sample_packet(ticker, as_of_date)
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


def _seed_decision_output(cfg, ticker: str, as_of_date: str):
    decision_path = cfg.analyst_outputs_dir / f"{ticker}_{as_of_date}" / "decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps({"decision": {"classification": "WATCHLIST"}}), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, 'decision', ?, 'hash', ?)
            """,
            (ticker, as_of_date, str(decision_path), utc_now_iso()),
        )


def _seed_analysis_report(
    cfg,
    ticker: str,
    as_of_date: str,
    *,
    overall_score: float,
    incomplete: bool,
):
    report = AnalysisReport(
        analysis_id=f"{ticker}_{as_of_date}_analysis",
        ticker=ticker,
        as_of_date=as_of_date,
        generated_at="2026-02-13T12:00:00+00:00",
        verdict="BUY",
        confidence_label="HIGH",
        confidence_score=81,
        thesis_summary="Ranker test report.",
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
                    summary="Missing recent filings",
                    recommended_action="Review the next filing set.",
                )
            ],
        ),
    )
    report_path = cfg.analyst_outputs_dir / f"{ticker}_{as_of_date}" / "analysis_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, 'analysis_report', ?, 'hash', ?)
            """,
            (ticker, as_of_date, str(report_path), utc_now_iso()),
        )


def _seed_legacy_packet(cfg, ticker: str, as_of_date: str, run_id: str, *, overall_score: float, incomplete: bool):
    payload = {
        "quality": {
            "coverage_score": 20.0,
            "freshness_score": 15.0,
            "gap_score": 12.0,
            "overall_research_score": overall_score,
            "incomplete": incomplete,
            "top_gaps": [
                {
                    "severity": "high",
                    "summary": "Legacy gap",
                    "recommended_action": "Legacy action",
                }
            ],
        },
        "evidence_items": [{"source_type": "filing", "source_url": "https://legacy.example", "excerpt_text": "Legacy"}],
    }
    research_path = cfg.research_dir / f"{ticker}_{run_id}.json"
    research_path.parent.mkdir(parents=True, exist_ok=True)
    research_path.write_text(json.dumps(payload), encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO research_packets(ticker, as_of_date, run_id, packet_path, packet_hash, created_at)
            VALUES(?, ?, ?, ?, 'hash', ?)
            """,
            (ticker, as_of_date, run_id, str(research_path), utc_now_iso()),
        )


def _latest_score_row(ticker: str):
    with get_db() as conn:
        return conn.execute(
            """
            SELECT subscores_json, reasons_json
            FROM scores
            WHERE ticker = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()


def test_score_ticker_uses_analysis_report_without_research_packet(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _ = get_config()
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    _seed_decision_output(cfg, "AAPL", "2026-02-13")
    _seed_analysis_report(cfg, "AAPL", "2026-02-13", overall_score=80.0, incomplete=False)

    assert score_ticker("AAPL", run_id="run_analysis_only") is True

    row = _latest_score_row("AAPL")
    subscores = json.loads(row["subscores_json"])
    reasons = json.loads(row["reasons_json"])
    assert subscores["research_overall"] == 80.0
    assert "research_penalty" not in subscores
    assert all("Research incomplete" not in reason for reason in reasons)


def test_score_ticker_falls_back_to_legacy_research_packet(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    _seed_decision_output(cfg, "AAPL", "2026-02-13")
    _seed_legacy_packet(cfg, "AAPL", "2026-02-13", "run_legacy_only", overall_score=40.0, incomplete=True)

    assert score_ticker("AAPL", run_id="run_legacy_only") is True

    row = _latest_score_row("AAPL")
    subscores = json.loads(row["subscores_json"])
    reasons = json.loads(row["reasons_json"])
    assert subscores["research_overall"] == 40.0
    assert subscores["research_penalty"] == -12.0
    assert any("Research incomplete" in reason for reason in reasons)


def test_score_ticker_prefers_analysis_report_over_legacy_packet(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_evidence_packet(cfg, "AAPL", "2026-02-13")
    _seed_decision_output(cfg, "AAPL", "2026-02-13")
    _seed_analysis_report(cfg, "AAPL", "2026-02-13", overall_score=80.0, incomplete=False)
    _seed_legacy_packet(cfg, "AAPL", "2026-02-13", "run_precedence", overall_score=10.0, incomplete=True)

    assert score_ticker("AAPL", run_id="run_precedence") is True

    row = _latest_score_row("AAPL")
    subscores = json.loads(row["subscores_json"])
    reasons = json.loads(row["reasons_json"])
    assert subscores["research_overall"] == 80.0
    assert "research_penalty" not in subscores
    assert all("Research incomplete" not in reason for reason in reasons)
