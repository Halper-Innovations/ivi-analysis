"""Ops-deck read model + /api/ops/*: heartbeats, backups, costs, logs, consoles."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import init_db
from app.ops.heartbeat_ledger import heartbeat_status, record_step
from app.web.main import app
from app.web.readmodel import ops_deck
from app.web.readmodel.runs_index import open_ui_db, refresh_index

client = TestClient(app)

NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    # The cheap-mode health check probes the watchlist table.
    from app.watchlist.schema import ensure_watchlist_schema

    ensure_watchlist_schema(db_path)
    return cfg


def _conn(cfg) -> sqlite3.Connection:
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _cron_dir(cfg) -> Path:
    directory = Path(cfg.outputs_dir) / "cron"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_heartbeat_ledger_matches_production_derivation(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    # 07-21: a failed step but a clean completion afterwards -> COMPLETE.
    record_step(heartbeat="events_daily", step="poll", exit_code=0, run_date="2026-07-21")
    record_step(heartbeat="events_daily", step="news_scan", exit_code=1, run_date="2026-07-21")
    record_step(heartbeat="events_daily", step="_complete", exit_code=0, run_date="2026-07-21")
    # 07-20: a failed step, never completed -> FAILED.
    record_step(heartbeat="events_daily", step="poll", exit_code=1, run_date="2026-07-20")
    # 07-19: started only -> STARTED.
    record_step(heartbeat="events_daily", step="poll", exit_code=0, run_date="2026-07-19")
    cron = _cron_dir(cfg)
    (cron / "events_heartbeat_20260721.log").write_text("ran\n", encoding="utf-8")

    conn = _conn(cfg)
    ledger = ops_deck.heartbeat_ledger(conn, days=4, now=NOW)
    conn.close()

    assert ledger["dates"] == ["2026-07-22", "2026-07-21", "2026-07-20", "2026-07-19"]
    assert len(ledger["heartbeats"]) == 1
    row = ledger["heartbeats"][0]
    assert row["heartbeat"] == "events_daily"
    cells = {cell["run_date"]: cell for cell in row["days"]}
    assert cells["2026-07-22"]["status"] == "MISSING"
    assert cells["2026-07-21"]["status"] == "COMPLETE"
    assert cells["2026-07-21"]["failed_steps"] == ["news_scan"]
    assert cells["2026-07-21"]["log_name"] == "events_heartbeat_20260721.log"
    assert cells["2026-07-20"]["status"] == "FAILED"
    assert cells["2026-07-20"]["log_name"] is None
    assert cells["2026-07-19"]["status"] == "STARTED"

    # The web cells agree with the production heartbeat_status derivation.
    for run_date in ("2026-07-21", "2026-07-20", "2026-07-19"):
        production = heartbeat_status(heartbeat="events_daily", run_date=run_date)
        assert cells[run_date]["status"] == production["status"]
        assert cells[run_date]["failed_steps"] == production["failed_steps"]


def test_backups_reads_status_file_and_history(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    cron = _cron_dir(cfg)
    (cron / "backup_status_latest.json").write_text(
        json.dumps(
            {
                "date": "20260721",
                "status": "OK",
                "detail": "rsync clean",
                "duration_s": 42.5,
                "finished_at": "2026-07-21T14:20:00Z",
            }
        ),
        encoding="utf-8",
    )
    (cron / "nightly_backup_20260721.log").write_text("done\n", encoding="utf-8")
    (cron / "nightly_backup_20260720.log").write_text("done\n", encoding="utf-8")

    payload = ops_deck.backups(now=NOW)
    assert payload["ok"] is True
    assert payload["detail"] == "backup status=OK (1d old)"
    assert payload["ceiling_days"] == 4
    assert payload["age_days"] == 1
    assert payload["status"] == "OK"
    assert payload["status_detail"] == "rsync clean"
    assert payload["duration_s"] == 42.5
    assert [h["log_name"] for h in payload["history"]] == [
        "nightly_backup_20260721.log",
        "nightly_backup_20260720.log",
    ]
    assert payload["history"][0]["date"] == "20260721"


def test_backups_failed_status_is_not_ok(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    cron = _cron_dir(cfg)
    (cron / "backup_status_latest.json").write_text(
        json.dumps({"date": "20260721", "status": "FAILED", "detail": "root not writable"}),
        encoding="utf-8",
    )
    payload = ops_deck.backups(now=NOW)
    assert payload["ok"] is False
    assert payload["detail"] == "backup status=FAILED (1d old)"
    assert payload["status_detail"] == "root not writable"


def _seed_run_artifact(cfg, *, with_cost: bool) -> None:
    run_dir = Path(cfg.runs_dir) / "autonomous_sector" / (
        "run_costed" if with_cost else "run_uncosted"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_dir.name,
        "sector": "biotech" if with_cost else "energy",
        "market_cap_focus": "mid_cap",
        "as_of_date": "2026-01-01",
        "created_at": "2026-01-01T10:00:00Z",
        "status": "COMPLETED",
    }
    if with_cost:
        payload["pipeline_version"] = "v2"
        payload["lane_usage"] = {"aggregate": {"cost_microdollars": 2_500_000}}
    (run_dir / "autonomous_sector_run.json").write_text(json.dumps(payload), encoding="utf-8")


def test_cost_ledger_weeks_models_and_uncosted_runs(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run_artifact(cfg, with_cost=True)
    _seed_run_artifact(cfg, with_cost=False)
    engine = _conn(cfg)
    engine.execute(
        """
        INSERT INTO synthesis_packets(
            ticker, as_of_date, run_id, packet_path, packet_hash, packet_json,
            prompt_hash, input_hash, provider, model, cost_estimate_usd,
            from_cache, created_at)
        VALUES ('AAA', '2026-01-08', 'r1', 'p1', 'h1', '{}', 'ph1', 'ih1',
                'openai', 'gpt-5-mini', 0.75, 0, '2026-01-08T10:00:00Z')
        """
    )
    engine.execute(
        """
        INSERT INTO synthesis_packets(
            ticker, as_of_date, run_id, packet_path, packet_hash, packet_json,
            prompt_hash, input_hash, provider, model, cost_estimate_usd,
            from_cache, created_at)
        VALUES ('BBB', '2026-01-08', 'r2', 'p2', 'h2', '{}', 'ph2', 'ih2',
                'openai', 'gpt-5-mini', 0.30, 1, '2026-01-08T11:00:00Z')
        """
    )
    engine.commit()

    ui_conn = open_ui_db()
    refresh_index(ui_conn)
    ledger = ops_deck.cost_ledger(ui_conn, engine)
    ui_conn.close()
    engine.close()

    weeks = {w["week"]: w for w in ledger["weeks"]}
    # 2026-01-01 is ISO 2026-W01; 2026-01-08 is ISO 2026-W02.
    assert weeks["2026-W01"]["scan_cost_usd"] == 2.5
    assert weeks["2026-W01"]["scan_runs"] == 1
    assert weeks["2026-W01"]["total_usd"] == 2.5
    assert weeks["2026-W02"]["research_cost_usd"] == 0.75
    assert weeks["2026-W02"]["research_calls"] == 2
    assert ledger["runs_with_cost"] == 1
    assert ledger["runs_without_cost"] == 1
    assert ledger["by_sector"] == [{"key": "biotech", "cost_usd": 2.5, "runs": 1}]
    assert ledger["by_band"] == [{"key": "mid_cap", "cost_usd": 2.5, "runs": 1}]
    assert ledger["by_model"] == [
        {"key": "openai/gpt-5-mini", "cost_usd": 0.75, "calls": 2, "cached_calls": 1}
    ]
    assert ledger["total_scan_usd"] == 2.5
    assert ledger["total_research_usd"] == 0.75


def test_read_log_tail_allowlist_and_tail(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    cron = _cron_dir(cfg)
    (cron / "events_heartbeat_20260721.log").write_text(
        "line1\nline2\nline3\nline4\nline5\n", encoding="utf-8"
    )
    payload = ops_deck.read_log_tail("events_heartbeat_20260721.log", lines=3)
    assert payload["lines"] == ["line3", "line4", "line5"]
    assert payload["truncated"] is True
    full = ops_deck.read_log_tail("events_heartbeat_20260721.log", lines=10)
    assert full["lines"] == ["line1", "line2", "line3", "line4", "line5"]
    assert full["truncated"] is False

    with pytest.raises(ValueError):
        ops_deck.read_log_tail("../evil.log")
    with pytest.raises(ValueError):
        ops_deck.read_log_tail("notes.txt")
    with pytest.raises(FileNotFoundError):
        ops_deck.read_log_tail("missing.log")

    listing = ops_deck.list_cron_logs()
    assert [f["name"] for f in listing] == ["events_heartbeat_20260721.log"]


def test_consoles_lists_legacy_tables(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    conn = _conn(cfg)
    conn.execute(
        """
        INSERT INTO dead_letter_jobs(job_id, job_type, payload_json, attempts,
                                     error_type, error_message, moved_at)
        VALUES (7, 'research', '{}', 3, 'TimeoutError', 'took too long',
                '2026-07-20T10:00:00Z')
        """
    )
    conn.execute(
        """
        INSERT INTO jobs(job_type, payload_json, scheduled_at, status,
                         created_at, updated_at)
        VALUES ('research', '{}', '2026-07-20T09:00:00Z', 'pending',
                '2026-07-20T09:00:00Z', '2026-07-20T09:00:00Z')
        """
    )
    conn.execute(
        """
        INSERT INTO research_signals(ticker, as_of_date, run_id, recency_days_min,
                                     item_count_30d, sentiment_flags_json, summary_json,
                                     created_at)
        VALUES ('AAA', '2026-07-01', 'run9', 45, 2, '["STALE"]',
                '{"freshness_bucket": "STALE", "flow_bucket": "LOW"}',
                '2026-07-01T09:00:00Z')
        """
    )
    conn.commit()

    payload = ops_deck.consoles(conn)
    conn.close()
    assert payload["backlog_size"] == 1
    assert payload["deadletters"] == [
        {
            "id": 1,
            "job_id": 7,
            "job_type": "research",
            "attempts": 3,
            "error_type": "TimeoutError",
            "error_message": "took too long",
            "moved_at": "2026-07-20T10:00:00Z",
        }
    ]
    assert payload["research_gaps"] == [
        {
            "ticker": "AAA",
            "as_of_date": "2026-07-01",
            "run_id": "run9",
            "recency_days_min": 45,
            "item_count_30d": 2,
            "risk_flags": ["STALE"],
            "freshness_bucket": "STALE",
            "flow_bucket": "LOW",
            "created_at": "2026-07-01T09:00:00Z",
        }
    ]
    assert payload["legacy_runs"] == []


def test_api_ops_endpoints(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    # The ledger reads a frozen clock rather than the machine's date, so the
    # recorded run date is a literal instead of "today".
    api_run_date = "2026-07-21"
    api_date_token = api_run_date.replace("-", "")
    heartbeat_log = f"events_heartbeat_{api_date_token}.log"
    record_step(
        heartbeat="events_daily",
        step="_complete",
        exit_code=0,
        run_date=api_run_date,
    )
    production_heartbeat_ledger = ops_deck.heartbeat_ledger
    monkeypatch.setattr(
        ops_deck,
        "heartbeat_ledger",
        lambda conn, *, days=14: production_heartbeat_ledger(
            conn,
            days=days,
            now=NOW,
        ),
    )
    cron = _cron_dir(cfg)
    (cron / "backup_status_latest.json").write_text(
        json.dumps({"date": "20260721", "status": "OK"}), encoding="utf-8"
    )
    (cron / heartbeat_log).write_text("ok\n", encoding="utf-8")

    health = client.get("/api/ops/health")
    assert health.status_code == 200
    # VOE_DB_PATH override -> cheap mode: readable books only.
    assert health.json()["health"]["state"] == "GREEN"
    assert health.json()["health"]["checks"][0]["name"] == "db_readable"

    heartbeats = client.get("/api/ops/heartbeats", params={"days": 3})
    assert heartbeats.status_code == 200
    assert heartbeats.json()["heartbeats"][0]["heartbeat"] == "events_daily"

    backups = client.get("/api/ops/backups")
    assert backups.status_code == 200
    assert backups.json()["status"] == "OK"

    costs = client.get("/api/ops/costs")
    assert costs.status_code == 200
    assert costs.json()["weeks"] == []

    logs = client.get("/api/ops/logs")
    assert logs.status_code == 200
    names = [f["name"] for f in logs.json()["files"]]
    assert heartbeat_log in names

    tail = client.get(f"/api/ops/logs/{heartbeat_log}")
    assert tail.status_code == 200
    assert tail.json()["lines"] == ["ok"]

    assert client.get("/api/ops/logs/missing.log").status_code == 404
    assert client.get("/api/ops/logs/..%2Fevil.log").status_code in (400, 404)

    consoles = client.get("/api/ops/consoles")
    assert consoles.status_code == 200
    assert consoles.json()["backlog_size"] == 0
