"""Measurement-scope routing + append-preserving history archival."""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.config import get_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def _env(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    from app.db import init_db

    init_db()
    return db_path


def test_valuations_table_switches_inside_scope():
    from app.valuation.measurement import (
        in_measurement_scope,
        measurement_scope,
        valuations_table,
    )

    assert valuations_table() == "valuations"
    assert not in_measurement_scope()
    with measurement_scope():
        assert valuations_table() == "valuations_measurement"
        assert in_measurement_scope()
    assert valuations_table() == "valuations"


def test_measurement_write_never_touches_live_table(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.db import get_db
    from app.valuation.measurement import measurement_scope
    from app.valuation.sanity_checks import _insert_valuation

    with get_db() as conn:
        with measurement_scope():
            _insert_valuation(
                conn, "AAA", "2026-07-01", "scorecard", {"k": 1}, {"v": 2}, []
            )
        live = conn.execute("SELECT COUNT(*) FROM valuations").fetchone()[0]
        measurement = conn.execute(
            "SELECT COUNT(*) FROM valuations_measurement"
        ).fetchone()[0]
    assert live == 0
    assert measurement == 1


def test_live_overwrite_archives_prior_row(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.db import get_db
    from app.valuation.sanity_checks import _insert_valuation

    with get_db() as conn:
        _insert_valuation(conn, "AAA", "2026-07-01", "dcf", {"k": 1}, {"value": 10}, [])
        _insert_valuation(conn, "AAA", "2026-07-01", "dcf", {"k": 2}, {"value": 20}, [])
        # Identical re-write must NOT add a second history copy.
        _insert_valuation(conn, "AAA", "2026-07-01", "dcf", {"k": 2}, {"value": 20}, [])
        live = conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker='AAA' AND method='dcf'"
        ).fetchone()
        history = conn.execute(
            "SELECT outputs_json, archived_at FROM valuations_history "
            "WHERE ticker='AAA' AND method='dcf' ORDER BY id"
        ).fetchall()
    assert json.loads(live[0]) == {"value": 20}
    assert len(history) == 1
    assert json.loads(history[0][0]) == {"value": 10}


def test_add_outcome_overwrite_archives_prior_row(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    from app.db import get_db
    from app.outcomes.store import add_outcome

    add_outcome(
        ticker="AAA",
        as_of_date="2026-07-01",
        run_id="run1",
        decision="BUY",
        conviction=4,
        horizon_days=180,
    )
    add_outcome(
        ticker="AAA",
        as_of_date="2026-07-01",
        run_id="run1",
        decision="PASS",
        conviction=2,
        horizon_days=180,
    )
    with get_db() as conn:
        history = conn.execute(
            "SELECT row_json FROM ticker_outcomes_history WHERE ticker='AAA'"
        ).fetchall()
        live = conn.execute(
            "SELECT decision FROM ticker_outcomes WHERE ticker='AAA' AND run_id='run1'"
        ).fetchone()
    assert live["decision"] == "PASS"
    assert len(history) == 1
    assert json.loads(history[0]["row_json"])["decision"] == "BUY"


def test_close_outcome_archives_open_row(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    from app.db import get_db
    from app.outcomes.store import add_outcome, close_outcome

    add_outcome(
        ticker="BBB",
        as_of_date="2026-07-01",
        run_id="run1",
        decision="BUY",
        conviction=4,
        horizon_days=180,
    )
    close_outcome(
        ticker="BBB",
        run_id="run1",
        close_date="2026-07-10",
        realized_return_pct=5.0,
    )
    with get_db() as conn:
        history = conn.execute(
            "SELECT row_json FROM ticker_outcomes_history WHERE ticker='BBB'"
        ).fetchall()
    assert len(history) == 1
    assert json.loads(history[0]["row_json"])["outcome_status"] == "OPEN"


def test_record_scan_overwrite_archives_prior(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.events import store

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    store.record_scan(conn, scan_date="2026-07-01", mode="daily", status="ERROR")
    store.record_scan(conn, scan_date="2026-07-01", mode="daily", status="OK")
    history = conn.execute(
        "SELECT row_json FROM corporate_event_scans_history WHERE scan_date='2026-07-01'"
    ).fetchall()
    conn.close()
    assert len(history) == 1
    assert json.loads(history[0]["row_json"])["status"] == "ERROR"
