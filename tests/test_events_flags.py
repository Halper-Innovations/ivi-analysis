from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.flags import open_flags_by_ticker, sync_event_pending_flags
from app.watchlist.schema import ensure_watchlist_schema


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


def _add_watch_row(conn, ticker, status="DEPLOY_READY"):
    conn.execute(
        "INSERT INTO watchlist(ticker, status, source_run_id, added_at) VALUES(?, ?, ?, ?)",
        (ticker, status, f"run_{ticker}", "2026-06-01T00:00:00Z"),
    )


def _add_merger_event(conn, cik="0001692427", ticker="NCSM"):
    eid = store.upsert_event(
        conn, cik=cik, event_type="merger",
        anchor_accession="0001104659-26-061001", company_name="NCS MULTISTAGE",
        detection_date="2026-06-01", source_mode="daily",
        detail={"watchlist_ticker": ticker},
    )
    store.set_ticker(conn, event_id=eid, ticker=ticker)
    return eid


def test_sync_sets_flag_and_writes_history(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        _add_merger_event(conn)
        result = sync_event_pending_flags(conn)
        assert result == {"flagged": 1, "cleared": 0, "unchanged": 0}
        row = conn.execute("SELECT * FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] == "EVENT_PENDING:MERGER"
        history = conn.execute(
            "SELECT * FROM watchlist_history WHERE field_name='event_pending'"
        ).fetchone()
        assert history["new_value"] == "EVENT_PENDING:MERGER"
        assert history["source"] == "events_feed"
        # Idempotent second sync.
        assert sync_event_pending_flags(conn) == {"flagged": 0, "cleared": 0, "unchanged": 1}


def test_sync_clears_flag_after_disposal(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        eid = _add_merger_event(conn)
        sync_event_pending_flags(conn)
        store.mark_decided(conn, event_id=eid, note="analyst pass: merger arb, not value")
        result = sync_event_pending_flags(conn)
        assert result == {"flagged": 0, "cleared": 1, "unchanged": 0}
        row = conn.execute("SELECT * FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] is None


def test_multiple_flags_sorted_and_joined(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        _add_merger_event(conn)
        eid = store.upsert_event(
            conn, cik="0001692427", event_type="material_agreement",
            anchor_accession="0001692427-26-000020", company_name="NCS MULTISTAGE",
            detection_date="2026-06-02", source_mode="daily",
        )
        store.set_ticker(conn, event_id=eid, ticker="NCSM")
        sync_event_pending_flags(conn)
        row = conn.execute("SELECT * FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] == "EVENT_PENDING:MATERIAL_AGREEMENT,EVENT_PENDING:MERGER"
        assert open_flags_by_ticker(conn) == {
            "NCSM": ["EVENT_PENDING:MATERIAL_AGREEMENT", "EVENT_PENDING:MERGER"],
        }


def test_opportunity_events_do_not_flag(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "SPNC")
        eid = store.upsert_event(
            conn, cik="0000123456", event_type="spinoff",
            anchor_accession="0001234567-26-000001", company_name="SPINCO",
            detection_date="2026-01-05", source_mode="daily",
        )
        store.set_ticker(conn, event_id=eid, ticker="SPNC")
        sync_event_pending_flags(conn)
        row = conn.execute("SELECT * FROM watchlist WHERE ticker='SPNC'").fetchone()
        assert row["event_pending"] is None


def test_sync_tolerates_missing_watchlist_table(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    with get_db() as conn:
        assert sync_event_pending_flags(conn) == {"flagged": 0, "cleared": 0, "unchanged": 0}
