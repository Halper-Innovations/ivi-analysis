from __future__ import annotations

from typer.testing import CliRunner

from app.cli import app
from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.flags import sync_event_pending_flags
from app.watchlist.contract import WatchlistEntry
from app.watchlist.digest import render_digest
from app.watchlist.store import add_or_update, add_price_snapshot, get_latest, watchlist_queue

runner = CliRunner()


def _init(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    init_db()


def _add_row(ticker, *, buy_target, price, status="DEPLOY_READY"):
    add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="sector_final_decision",
            valuation_anchor_method="DCF",
            valuation_anchor_value=buy_target / 0.75,
            buy_price_target=buy_target,
            current_price_at_addition=price,
            thesis_text=f"{ticker} thesis",
            source_run_id=f"run_{ticker}",
            source_sector="energy_services",
            added_at="2026-06-01T00:00:00Z",
        )
    )
    entry = get_latest(ticker)
    add_price_snapshot(entry.id, price=price, checked_at="2026-06-10T00:00:00Z", source="test")


def _flag_merger(ticker, cik):
    with get_db() as conn:
        eid = store.upsert_event(
            conn, cik=cik, event_type="merger",
            anchor_accession=f"{cik}-26-061001", company_name=ticker,
            detection_date="2026-06-01", source_mode="daily",
        )
        store.set_ticker(conn, event_id=eid, ticker=ticker)
        sync_event_pending_flags(conn)
        return eid


def test_queue_presents_event_pending_and_demotes(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_row("NCSM", buy_target=57.18, price=24.95)
    _add_row("TDW", buy_target=60.00, price=55.00)
    _flag_merger("NCSM", "0001692427")
    rows = {row["ticker"]: row for row in watchlist_queue(limit=25)}
    assert rows["NCSM"]["status"] == "DEPLOY_READY"
    assert rows["NCSM"]["presented_status"] == "EVENT_PENDING"
    assert rows["NCSM"]["event_pending"] == "EVENT_PENDING:MERGER"
    assert rows["TDW"]["presented_status"] == "DEPLOY_READY"
    ordered = [row["ticker"] for row in watchlist_queue(limit=25)]
    # The unflagged DEPLOY_READY name outranks the blocked one.
    assert ordered.index("TDW") < ordered.index("NCSM")


def test_digest_blocks_deploy_ready_presentation(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_row("NCSM", buy_target=57.18, price=24.95)
    _add_row("TDW", buy_target=60.00, price=55.00)
    _flag_merger("NCSM", "0001692427")
    digest = render_digest(days_back=30)
    at_target_section = digest.split("## At Buy Target (Review)")[1].split("##")[0]
    assert "TDW" in at_target_section
    assert "NCSM" not in at_target_section
    blocked_section = digest.split("## Blocked Pending Event Review")[1].split("##")[0]
    assert "NCSM" in blocked_section
    assert "EVENT_PENDING:MERGER" in blocked_section


def test_buy_now_cli_blocks_flagged_rows(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_row("NCSM", buy_target=57.18, price=24.95)
    _flag_merger("NCSM", "0001692427")
    result = runner.invoke(app, ["investor", "buy-now"])
    assert result.exit_code == 0, result.output
    assert "AT TARGET NCSM" not in result.output
    assert "Blocked Pending Event Review" in result.output
    assert "EVENT_PENDING:MERGER" in result.output


def test_watchlist_show_renders_flag(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_row("NCSM", buy_target=57.18, price=24.95)
    _flag_merger("NCSM", "0001692427")
    result = runner.invoke(app, ["watchlist", "show", "NCSM"])
    assert result.exit_code == 0, result.output
    assert "EVENT_PENDING (stored: DEPLOY_READY)" in result.output
    assert "Events pending: EVENT_PENDING:MERGER" in result.output


def test_watchlist_queue_cli_renders_events_column(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_row("NCSM", buy_target=57.18, price=24.95)
    _flag_merger("NCSM", "0001692427")
    result = runner.invoke(app, ["watchlist", "queue"])
    assert result.exit_code == 0, result.output
    assert "EVENT_PENDING:MERGER" in result.output


def test_disposal_restores_presentation(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_row("NCSM", buy_target=57.18, price=24.95)
    eid = _flag_merger("NCSM", "0001692427")
    with get_db() as conn:
        store.mark_decided(conn, event_id=eid, note="analyst pass: definitive merger, exit")
        sync_event_pending_flags(conn)
    rows = {row["ticker"]: row for row in watchlist_queue(limit=25)}
    assert rows["NCSM"]["presented_status"] == "DEPLOY_READY"
    assert rows["NCSM"]["event_pending"] is None
