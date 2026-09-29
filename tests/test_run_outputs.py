from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.analyst.thesis_contract import (
    AnalysisReport,
    ResearchGapSummary,
    ResearchQualitySummary,
    ValuationConclusion,
)
from app.cli import app
from app.db import get_db, init_db, utc_now_iso
from app.ops.runs import finalize_run_outputs


runner = CliRunner()


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


def _seed_minimal_ticker(cfg, ticker: str = "AAPL"):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik, name=excluded.name
            """,
            (ticker, "320193", "Apple Inc.", utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'parsed', ?, ?)
            ON CONFLICT(cik, accession) DO UPDATE SET
                ticker=excluded.ticker,
                filing_date=excluded.filing_date,
                status='parsed'
            """,
            (
                "320193",
                ticker,
                "0000320193-26-000001",
                "10-Q",
                "2026-02-10",
                "2025-12-31",
                "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                str(Path(cfg.raw_filings_dir) / "dummy.htm"),
                "hash",
                "2026-02-13",
                utc_now_iso(),
                utc_now_iso(),
            ),
        )
        filing_id = int(
            conn.execute(
                "SELECT id FROM filings WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()["id"]
        )

        financial_rows = [
            ("income", "revenue", 1000.0),
            ("income", "operating_income", 250.0),
            ("cashflow", "cfo", 300.0),
            ("cashflow", "capex", 60.0),
            ("balance", "cash", 500.0),
            ("balance", "total_debt", 200.0),
        ]
        for statement_type, line_item, value in financial_rows:
            conn.execute(
                """
                INSERT INTO financials(
                    filing_id, statement_type, line_item, value, units, period, source_url, snippet, created_at
                ) VALUES(?, ?, ?, ?, 'USD', '2025Q4', ?, ?, ?)
                """,
                (
                    filing_id,
                    statement_type,
                    line_item,
                    value,
                    "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                    f"{line_item} snippet",
                    utc_now_iso(),
                ),
            )

        conn.execute(
            """
            INSERT INTO extracted_facts(filing_id, fact_type, value_json, source_url, snippet, section_label, created_at)
            VALUES(?, 'shares_outstanding', ?, ?, ?, 'cover_page', ?)
            """,
            (
                filing_id,
                json.dumps({"value": 15000000000}),
                "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
                "shares outstanding snippet",
                utc_now_iso(),
            ),
        )


def _seed_packet_fundamentals_and_valuations(cfg, ticker: str, as_of_date: str, run_id: str):
    packet = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "filings_used": [
            {
                "accession": "0000320193-26-000001",
                "form_type": "10-Q",
                "filing_date": "2026-02-10",
                "period_end": "2025-12-31",
                "primary_doc_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
            }
        ],
        "financials": [
            {"line_item": "revenue"},
            {"line_item": "operating_income"},
            {"line_item": "cfo"},
            {"line_item": "capex"},
            {"line_item": "cash"},
            {"line_item": "total_debt"},
        ],
        "extracted_facts": [],
        "fundamentals": {
            "revenue": 1000.0,
            "operating_margin": 0.20,
            "fcf": 240.0,
            "net_debt": -300.0,
            "liquidity_stress_score": 2,
        },
        "valuations": {
            "dcf": {"outputs": {"status": "OK", "confidence": "MEDIUM"}},
            "reverse_dcf": {"inputs": {"price_status": "OK", "market_price": 100.0}},
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
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                ticker,
                as_of_date,
                json.dumps(packet["fundamentals"]),
                utc_now_iso(),
            ),
        )
        conn.execute(
            """
            INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
            VALUES
                (?, ?, 'dcf', '{}', ?, '[]', ?),
                (?, ?, 'reverse_dcf', ?, '{}', '[]', ?)
            """,
            (
                ticker,
                as_of_date,
                json.dumps({"status": "OK", "confidence": "MEDIUM"}),
                utc_now_iso(),
                ticker,
                as_of_date,
                json.dumps({"price_status": "OK", "market_price": 100.0}),
                utc_now_iso(),
            ),
        )
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            ) VALUES(?, ?, ?, '{}', 55, 'Watchlist', 1, 0, ?, '[]', ?)
            """,
            (ticker, as_of_date, run_id, run_id, utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES
                (?, ?, 'hypotheses', ?, 'hash', ?),
                (?, ?, 'red_team', ?, 'hash', ?)
            """,
            (
                ticker,
                as_of_date,
                str(cfg.outputs_dir / "hypotheses.json"),
                utc_now_iso(),
                ticker,
                as_of_date,
                str(cfg.outputs_dir / "red_team.json"),
                utc_now_iso(),
            ),
        )

    (cfg.outputs_dir / "hypotheses.json").write_text(json.dumps({"hypotheses": [{"claim": "Claim", "citations": ["C1"]}]}), encoding="utf-8")
    (cfg.outputs_dir / "red_team.json").write_text(json.dumps({"red_team": [{"claim": "Risk"}]}), encoding="utf-8")


def _seed_analysis_report(cfg, ticker: str, as_of_date: str):
    report = AnalysisReport(
        analysis_id=f"{ticker}_{as_of_date}_analysis",
        ticker=ticker,
        as_of_date=as_of_date,
        generated_at="2026-02-13T12:00:00+00:00",
        verdict="BUY",
        confidence_label="HIGH",
        confidence_score=81,
        thesis_summary="Run output analysis report.",
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
            overall_score=72.0,
            incomplete=True,
            evidence_count=2,
            top_gaps=[
                ResearchGapSummary(
                    severity="high",
                    summary="Coverage is still thin.",
                    recommended_action="Review the next filing set.",
                )
            ],
        ),
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
            VALUES
                (?, ?, 'analysis_report', ?, 'hash', ?),
                (?, ?, 'analysis_report_markdown', ?, 'hash', ?)
            """,
            (
                ticker,
                as_of_date,
                str(report_path),
                utc_now_iso(),
                ticker,
                as_of_date,
                str(report_md_path),
                utc_now_iso(),
            ),
        )
        conn.execute(
            """
            INSERT INTO evidence_items(
                evidence_id, ticker, as_of_date, run_id, source_type, source_url,
                retrieved_at, excerpt_text, excerpt_hash, citations_json, derived_from_json, item_hash, created_at
            )
            VALUES(?, ?, ?, ?, 'filing', 'https://example.com', ?, 'excerpt', 'hash1', '[]', '[]', 'item1', ?)
            """,
            ("ev1", ticker, as_of_date, "run_analysis", utc_now_iso(), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO evidence_items(
                evidence_id, ticker, as_of_date, run_id, source_type, source_url,
                retrieved_at, excerpt_text, excerpt_hash, citations_json, derived_from_json, item_hash, created_at
            )
            VALUES(?, ?, ?, ?, 'filing', 'https://example.com/2', ?, 'excerpt', 'hash2', '[]', '[]', 'item2', ?)
            """,
            ("ev2", ticker, as_of_date, "run_analysis", utc_now_iso(), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO evidence_items(
                evidence_id, ticker, as_of_date, run_id, source_type, source_url,
                retrieved_at, excerpt_text, excerpt_hash, citations_json, derived_from_json, item_hash, created_at
            )
            VALUES(?, ?, ?, ?, 'filing', 'https://example.com/3', ?, 'excerpt', 'hash3', '[]', '[]', 'item3', ?)
            """,
            ("ev3", ticker, as_of_date, "run_analysis", utc_now_iso(), utc_now_iso()),
        )


def test_run_all_twice_creates_two_run_dirs_and_index_entries(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_minimal_ticker(cfg, ticker="AAPL")

    first = runner.invoke(
        app,
        [
            "run-all",
            "--as-of",
            "2026-02-13",
            "--tickers",
            "AAPL",
            "--phase",
            "valuation",
        ],
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        app,
        [
            "run-all",
            "--as-of",
            "2026-02-14",
            "--tickers",
            "AAPL",
            "--phase",
            "valuation",
        ],
    )
    assert second.exit_code == 0, second.output

    index_path = cfg.outputs_dir / "runs" / "index.json"
    assert index_path.exists()
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    assert isinstance(entries, list)
    assert len(entries) == 2
    assert entries[0]["run_id"] != entries[1]["run_id"]

    for entry in entries:
        run_dir = cfg.outputs_dir / "runs" / entry["run_id"]
        assert run_dir.exists()
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "gating_report.json").exists()
        assert (run_dir / "gating_report.csv").exists()


def test_gating_report_contains_reason_codes_and_fix_hint(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_minimal_ticker(cfg, ticker="AAPL")

    result = runner.invoke(
        app,
        [
            "run-all",
            "--as-of",
            "2026-02-13",
            "--tickers",
            "AAPL",
            "--phase",
            "valuation",
        ],
    )
    assert result.exit_code == 0, result.output

    entries = json.loads((cfg.outputs_dir / "runs" / "index.json").read_text(encoding="utf-8"))
    run_id = entries[-1]["run_id"]
    gating_path = cfg.outputs_dir / "runs" / run_id / "gating_report.json"
    payload = json.loads(gating_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    assert rows

    row = rows[0]
    assert "ticker" in row
    assert "gate_statuses" in row
    assert row.get("minimal_fix")
    first_gate = row["gate_statuses"][0]
    assert "reason_code" in first_gate
    assert "fix_hint" in first_gate


def test_finalize_run_outputs_scores_are_scoped_to_run_and_as_of(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    manifest_path = cfg.outputs_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run_scope",
                "as_of_date": "2026-02-13",
                "universe": {"universe_id": "u1", "snapshot_hash": "h1"},
            }
        ),
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            ) VALUES
                ('AAPL', '2026-02-13', 'run_scope', '{}', 55, 'Watchlist', 1, 0, 'run_scope', '[]', ?),
                ('AAPL', '2026-02-12', 'run_scope', '{}', 99, 'Watchlist', 0, 0, NULL, '[]', ?),
                ('AAPL', '2026-02-11', 'run_other', '{}', 88, 'Watchlist', 1, 0, 'run_other', '[]', ?)
            """,
            (now, now, now),
        )

    summary = finalize_run_outputs(
        run_id="run_scope",
        as_of_date="2026-02-13",
        tickers_targeted=["AAPL"],
        with_research=False,
        dead_letter_before=0,
        manifest_path=manifest_path,
    )
    assert summary["run_id"] == "run_scope"

    scores_payload = json.loads((cfg.outputs_dir / "runs" / "run_scope" / "rankings" / "scores.json").read_text(encoding="utf-8"))
    assert len(scores_payload["scores"]) == 1
    only = scores_payload["scores"][0]
    assert only["ticker"] == "AAPL"
    assert only["run_id"] == "run_scope"
    assert only["as_of_date"] == "2026-02-13"


def test_finalize_run_outputs_copies_analysis_report_and_uses_quality_for_gating(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_minimal_ticker(cfg, ticker="AAPL")
    _seed_packet_fundamentals_and_valuations(cfg, ticker="AAPL", as_of_date="2026-02-13", run_id="run_analysis")
    _seed_analysis_report(cfg, ticker="AAPL", as_of_date="2026-02-13")

    manifest_path = cfg.outputs_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"run_id": "run_analysis", "as_of_date": "2026-02-13"}),
        encoding="utf-8",
    )

    summary = finalize_run_outputs(
        run_id="run_analysis",
        as_of_date="2026-02-13",
        tickers_targeted=["AAPL"],
        with_research=True,
        dead_letter_before=0,
        manifest_path=manifest_path,
    )

    assert summary["run_id"] == "run_analysis"
    gating_path = cfg.outputs_dir / "runs" / "run_analysis" / "gating_report.json"
    payload = json.loads(gating_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["research"] is True
    assert row["key_metrics"]["overall_research_score"] == 72.0
    assert row["counts"]["evidence_items_count"] == 3
    research_gate = next(gate for gate in row["gate_statuses"] if gate["gate"] == "research")
    assert research_gate["status"] == "FAIL"
    assert research_gate["fix_hint"] == "Review the next filing set."
    assert row["artifacts"]["analysis_report_path"].endswith("analysis_report.json")
    assert row["artifacts"]["analysis_report_md_path"].endswith("analysis_report.md")
