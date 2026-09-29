from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.intake import surface_qualified_events
from app.watchlist.schema import ensure_watchlist_schema


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


def _seed(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        qualified_resolved = store.upsert_event(
            conn, cik="0000123456", event_type="spinoff",
            anchor_accession="0001234567-26-000001", company_name="SPINCO CORP",
            detection_date="2026-03-01", source_mode="daily",
        )
        store.mark_qualified(conn, event_id=qualified_resolved, qualification_date="2026-05-20")
        store.set_ticker(conn, event_id=qualified_resolved, ticker="SPNC")

        qualified_unknown = store.upsert_event(
            conn, cik="0000777777", event_type="ch11_emergence",
            anchor_accession="0000777777-26-000001", company_name="EMERGECO",
            detection_date="2026-02-01", source_mode="daily",
        )
        store.mark_qualified(conn, event_id=qualified_unknown, qualification_date="2026-05-01")

        store.upsert_event(
            conn, cik="0000888888", event_type="spinoff",
            anchor_accession="0000888888-26-000001", company_name="DETECTEDCO",
            detection_date="2026-04-01", source_mode="daily",
        )
        # A queue-protection event must never route to intake even if qualified.
        merger = store.upsert_event(
            conn, cik="0001692427", event_type="merger",
            anchor_accession="0001104659-26-061001", company_name="NCS MULTISTAGE",
            detection_date="2026-06-01", source_mode="daily",
        )
        store.set_ticker(conn, event_id=merger, ticker="NCSM")
    return qualified_resolved, qualified_unknown


def test_surface_qualified_events(monkeypatch, tmp_path):
    surfaced_id, unknown_id = _seed(monkeypatch, tmp_path)
    result = surface_qualified_events(as_of="2026-06-01")
    assert result == {"surfaced": 1, "awaiting_ticker": 1}
    with get_db() as conn:
        watch = conn.execute("SELECT * FROM watchlist WHERE ticker='SPNC'").fetchone()
        assert watch["conviction_grade"] == "DATA_INCOMPLETE"
        assert watch["conviction_source"] == "corporate_event"
        assert watch["source_run_id"] == "events_spinoff_0000123456_2026-05-20"
        assert watch["status_reason"] == "spinoff qualified 2026-05-20"
        assert watch["status"] == "ACTIVE"

        surfaced = conn.execute(
            "SELECT * FROM corporate_events WHERE id=?", (surfaced_id,)
        ).fetchone()
        assert surfaced["status"] == "SURFACED"
        assert surfaced["surfaced_at"] is not None
        unknown = conn.execute(
            "SELECT * FROM corporate_events WHERE id=?", (unknown_id,)
        ).fetchone()
        assert unknown["status"] == "QUALIFIED"

    # Idempotent second call.
    result2 = surface_qualified_events(as_of="2026-06-02")
    assert result2 == {"surfaced": 0, "awaiting_ticker": 1}
    with get_db() as conn:
        n = conn.execute("SELECT COUNT(*) c FROM watchlist").fetchone()["c"]
        assert n == 1
