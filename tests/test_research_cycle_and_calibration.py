from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import get_db, init_db, utc_now_iso
from app.discovery.calibration import run_calibration
from app.research.cycle import run_research_cycle


runner = CliRunner()


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_outcome_cli_add_list_close(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    add = runner.invoke(
        app,
        [
            "outcome-add",
            "--ticker",
            "AAPL",
            "--run-id",
            "run_test_outcome",
            "--decision",
            "BUY",
            "--conviction",
            "4",
            "--horizon-days",
            "90",
            "--as-of",
            "2026-02-13",
            "--notes",
            "evidence-backed setup",
        ],
    )
    assert add.exit_code == 0, add.output
    payload = json.loads(add.output)
    assert payload["ticker"] == "AAPL"
    assert payload["outcome_status"] == "OPEN"

    listed = runner.invoke(app, ["outcome-list", "--run-id", "run_test_outcome"])
    assert listed.exit_code == 0, listed.output
    rows = json.loads(listed.output)
    assert rows["count"] == 1
    assert rows["rows"][0]["decision"] == "BUY"

    close = runner.invoke(
        app,
        [
            "outcome-close",
            "--ticker",
            "AAPL",
            "--run-id",
            "run_test_outcome",
            "--realized-return",
            "12.5",
            "--close-date",
            "2026-05-15",
            "--max-dd",
            "-8.0",
        ],
    )
    assert close.exit_code == 0, close.output
    closed = json.loads(close.output)
    assert closed["outcome_status"] == "CLOSED"
    assert closed["realized_return_pct"] == 12.5


def test_research_cycle_stops_on_no_new_evidence_and_persists_summary(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO discovery_runs(
                run_id, run_as_of_date, seed_hash, config_hash, seed_path, status, processed_count,
                phase, tickers_targeted_json, processed_effective_dates_json, stats_json, created_at, updated_at
            ) VALUES('discovery_test', '2026-02-13', 'h', 'c', 'seed.csv', 'COMPLETED', 1,
                     'full', '["AAPL"]', '{}', '{}', ?, ?)
            """,
            (utc_now_iso(), utc_now_iso()),
        )

    monkeypatch.setattr(
        "app.research.cycle._load_advance_tickers",
        lambda discovery_run_id, top_k: [{"ticker": "AAPL", "stage": "ADVANCE_TO_DEEP"}],
    )
    monkeypatch.setattr(
        "app.research.cycle._baseline_deep_stage", lambda **kwargs: {"baseline_completed": True}
    )
    monkeypatch.setattr(
        "app.research.cycle.run_research_gap_closer", lambda **kwargs: {"processed_count": 1}
    )
    monkeypatch.setattr("app.research.cycle.score_ticker", lambda *args, **kwargs: True)
    monkeypatch.setattr("app.research.cycle.build_memo_for_ticker", lambda *args, **kwargs: True)
    monkeypatch.setattr("app.research.cycle._latest_score", lambda *args, **kwargs: 50.0)
    monkeypatch.setattr(
        "app.research.cycle._latest_research_quality", lambda *args, **kwargs: {"gap_score": 10.0}
    )
    monkeypatch.setattr("app.research.cycle._evidence_count", lambda *args, **kwargs: 5)
    monkeypatch.setattr("app.research.cycle.run_synthesis_for_ticker", lambda *args, **kwargs: None)

    summary = run_research_cycle(
        discovery_run_id="discovery_test",
        as_of_date="2026-02-13",
        max_iterations=2,
        top_k=1,
        budget_usd=0.0,
        sources={"news", "exhibits"},
        run_id="cycle_test",
    )
    assert summary["processed_count"] == 1
    assert summary["summaries"][0]["stop_reason"] == "NO_NEW_EVIDENCE"
    assert summary["summaries"][0]["iterations_run"] == 1

    with get_db() as conn:
        row = conn.execute(
            "SELECT summary_json FROM research_cycles WHERE ticker = 'AAPL' AND run_id = 'cycle_test'"
        ).fetchone()
    assert row is not None
    persisted = json.loads(row["summary_json"])
    assert persisted["completed"] is True
    assert persisted["stop_reason"] == "NO_NEW_EVIDENCE"


def test_calibration_excludes_unproven_rows_and_empty_report_is_stable(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO discovery_runs(
                run_id, run_as_of_date, seed_hash, config_hash, seed_path, status, processed_count,
                phase, tickers_targeted_json, processed_effective_dates_json, stats_json, created_at, updated_at
            ) VALUES('discovery_cal_1', '2026-02-13', 'seed', 'cfg', 'seed.csv', 'COMPLETED', 2,
                     'full', '["AAPL","MSFT"]', '{}', '{}', ?, ?)
            """,
            (now, now),
        )
        payload_aapl = {
            "ticker": "AAPL",
            "run_id": "discovery_cal_1",
            "stage": "ADVANCE_TO_DEEP",
            "whale_fit_score": 19.0,
            "evidence_strength_score": 7.0,
            "key_reasons": ["Revenue growth acceleration"],
            "gaps": ["MARKET_CAP_UNKNOWN"],
        }
        payload_msft = {
            "ticker": "MSFT",
            "run_id": "discovery_cal_1",
            "stage": "WATCHLIST_ONLY",
            "whale_fit_score": 12.0,
            "evidence_strength_score": 5.0,
            "key_reasons": ["Margin durability pending"],
            "gaps": ["MISSING_FCF"],
        }
        conn.execute(
            """
            INSERT INTO discovery_candidates(ticker, run_id, discovery_score, payload_json, created_at)
            VALUES('AAPL', 'discovery_cal_1', 75.0, ?, ?)
            """,
            (json.dumps(payload_aapl), now),
        )
        conn.execute(
            """
            INSERT INTO discovery_candidates(ticker, run_id, discovery_score, payload_json, created_at)
            VALUES('MSFT', 'discovery_cal_1', 58.0, ?, ?)
            """,
            (json.dumps(payload_msft), now),
        )
        conn.execute(
            """
            INSERT INTO ticker_outcomes(
                ticker, as_of_date, run_id, discovery_run_id, deep_run_id, decision, conviction, horizon_days,
                thesis_tags_json, notes, outcome_status, close_date, realized_return_pct, max_drawdown_pct, created_at, updated_at
            ) VALUES
            ('AAPL', '2026-02-13', 'deep_1', 'discovery_cal_1', 'deep_1', 'BUY', 4, 90, '[]', '', 'CLOSED', '2026-05-10', 18.0, -6.0, ?, ?),
            ('MSFT', '2026-02-13', 'deep_1', 'discovery_cal_1', 'deep_1', 'WATCH', 3, 90, '[]', '', 'CLOSED', '2026-05-10', -4.0, -9.0, ?, ?)
            """,
            (now, now, now, now),
        )

    report = run_calibration(run_id="discovery_cal_1", last_n=1)
    assert report["candidate_count"] == 0
    assert report["outcomes_total"] == 0
    assert report["closed_outcomes_total"] == 0
    assert report["hit_rate_by_stage"] == {}
    assert Path(report["json_path"]).exists()
    assert Path(report["md_path"]).exists()

    report_again = run_calibration(run_id="discovery_cal_1", last_n=1)
    assert report_again["hit_rate_by_stage"] == report["hit_rate_by_stage"]
    assert report_again["top_reason_counts"] == report["top_reason_counts"]
    assert str(cfg.calibration_dir) in report_again["json_path"]
