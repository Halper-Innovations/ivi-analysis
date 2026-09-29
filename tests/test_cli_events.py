from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.flags import sync_event_pending_flags
from app.events.poller import ScanSummary
from app.watchlist.schema import ensure_watchlist_schema

runner = CliRunner()


def _init(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


def test_events_registered_in_root_help():
    result = runner.invoke(app, ["events", "--help"])
    assert result.exit_code == 0
    for command in ("poll", "backfill", "surface", "coverage", "list", "dispose", "cheapness"):
        assert command in result.output


def test_events_poll_passes_date(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    captured = {}

    def fake_poll_day(scan_date, **kwargs):
        captured["scan_date"] = scan_date
        return ScanSummary(scan_date=scan_date, mode="daily", status="OK", index_rows=5)

    import app.events.poller as poller_module

    monkeypatch.setattr(poller_module, "poll_day", fake_poll_day)
    result = runner.invoke(app, ["events", "poll", "--date", "2026-06-08"])
    assert result.exit_code == 0, result.output
    assert captured["scan_date"] == "2026-06-08"
    assert '"status": "OK"' in result.output


def test_events_poll_guard_exits_nonzero(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    def fake_poll_day(scan_date, **kwargs):
        raise ValueError("poll_day refuses scan_date")

    import app.events.poller as poller_module

    monkeypatch.setattr(poller_module, "poll_day", fake_poll_day)
    result = runner.invoke(app, ["events", "poll", "--date", "2099-01-01"])
    assert result.exit_code != 0


def test_events_list_and_dispose_clear_flag(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO watchlist(ticker, status, source_run_id, added_at) "
            "VALUES('NCSM', 'DEPLOY_READY', 'run_NCSM', '2026-06-01T00:00:00Z')"
        )
        eid = store.upsert_event(
            conn, cik="0001692427", event_type="merger",
            anchor_accession="0001104659-26-061001", company_name="NCS MULTISTAGE",
            detection_date="2026-06-01", source_mode="daily",
        )
        store.set_ticker(conn, event_id=eid, ticker="NCSM")
        sync_event_pending_flags(conn)

    result = runner.invoke(app, ["events", "list", "--open"])
    assert result.exit_code == 0, result.output
    assert "merger" in result.output
    assert "NCSM" in result.output

    result = runner.invoke(
        app, ["events", "dispose", str(eid), "--note", "analyst pass: merger arb only"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["disposed"] == eid

    with get_db() as conn:
        row = conn.execute("SELECT event_pending FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] is None
        event = conn.execute("SELECT status FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert event["status"] == "DECIDED"

    # Second dispose is a refused no-op.
    result = runner.invoke(app, ["events", "dispose", str(eid), "--note", "again"])
    assert result.exit_code == 1


def test_events_coverage_runs_on_seed_csv(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    result = runner.invoke(app, ["events", "coverage"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total"] >= 1
    assert "coverage" in payload


def test_events_cheapness_uses_queue_when_no_tickers(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO watchlist(ticker, status, source_run_id, added_at) "
            "VALUES('KEQU', 'DEPLOY_READY', 'run_KEQU', '2026-06-01T00:00:00Z')"
        )

    captured = {}

    def fake_build(ticker, **kwargs):
        captured.setdefault("tickers", []).append(ticker)
        return {"ticker": ticker, "status": "OK", "from_cache": False, "flags": [],
                "llm_verdict": "NO_KNOWN_EVENT", "llm_bullets": [], "deterministic": {},
                "as_of": "2026-06-11"}

    import app.events.cheapness as cheapness_module

    monkeypatch.setattr(cheapness_module, "build_cheapness_report", fake_build)
    result = runner.invoke(app, ["events", "cheapness", "--as-of", "2026-06-11"])
    assert result.exit_code == 0, result.output
    assert captured["tickers"] == ["KEQU"]
    assert "NO_KNOWN_EVENT" in result.output


def test_events_cheapness_does_not_resurrect_superseded_deploy_row(
    monkeypatch, tmp_path
):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO watchlist(ticker, status, source_run_id, added_at) "
            "VALUES('OLD', 'DEPLOY_READY', 'run_old', '2026-06-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO watchlist(ticker, status, source_run_id, added_at) "
            "VALUES('OLD', 'REMOVED', 'run_current', '2026-06-02T00:00:00Z')"
        )

    calls = []
    monkeypatch.setattr(
        "app.events.cheapness.build_cheapness_report",
        lambda ticker, **_kwargs: calls.append(ticker),
    )

    result = runner.invoke(app, ["events", "cheapness", "--as-of", "2026-06-11"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "No targets."
    assert calls == []


def test_events_cheapness_isolates_integrity_refusal_and_exits_not_clean(
    monkeypatch, tmp_path
):
    from app.autonomous.financial_integrity import (
        FinancialIntegrityGateResult,
        FinancialIntegrityViolation,
        InvalidFinancialInputError,
    )

    _init(monkeypatch, tmp_path)
    calls: list[str] = []

    def fake_build(ticker, **kwargs):
        calls.append(ticker)
        if ticker == "AAA":
            raise InvalidFinancialInputError(
                FinancialIntegrityGateResult(
                    context="cheapness:AAA:2026-06-11",
                    run_as_of_date="2026-06-11",
                    status="NEEDS_DATA",
                    violations=(
                        FinancialIntegrityViolation(
                            code="SHARES_BASIS_MISSING",
                            ticker="AAA",
                            terminal_status="NEEDS_DATA",
                        ),
                    ),
                )
            )
        return {"status": "NEEDS_REFRESH"}

    import app.events.cheapness as cheapness_module

    monkeypatch.setattr(cheapness_module, "build_cheapness_report", fake_build)
    result = runner.invoke(
        app,
        ["events", "cheapness", "AAA", "BBB", "CCC", "--as-of", "2026-06-11"],
    )

    assert result.exit_code == 1
    assert calls == ["AAA", "BBB", "CCC"]
    assert "## AAA\n- status: NEEDS_DATA\n- refusal_codes: SHARES_BASIS_MISSING" in result.output
    assert '"integrity_refusals": 1' in result.output
    assert '"ticker": "AAA"' in result.output
