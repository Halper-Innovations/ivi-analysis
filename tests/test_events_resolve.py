from __future__ import annotations


from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.resolve import invert_ticker_map, resolve_unknown_tickers


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def test_invert_ticker_map_shortest_then_alphabetical():
    assert invert_ticker_map({"AAPL": "320193", "AAPLW": "320193"}) == {"0000320193": "AAPL"}
    assert invert_ticker_map({"BB": "1", "AA": "1"}) == {"0000000001": "AA"}


def test_resolve_unknown_tickers(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        store.upsert_event(
            conn, cik="0000320193", event_type="spinoff",
            anchor_accession="0000320193-26-000001", company_name="APPLE SPINCO",
            detection_date="2026-01-05", source_mode="daily",
        )
        store.upsert_event(
            conn, cik="0000777777", event_type="spinoff",
            anchor_accession="0000777777-26-000001", company_name="UNMAPPED CO",
            detection_date="2026-01-05", source_mode="daily",
        )
        result = resolve_unknown_tickers(conn, mapping={"AAPL": "320193"})
        assert result == {"resolved": 1, "still_unknown": 1}
        rows = {
            r["cik"]: r for r in conn.execute("SELECT * FROM corporate_events").fetchall()
        }
        assert rows["0000320193"]["ticker"] == "AAPL"
        assert rows["0000320193"]["ticker_state"] == "RESOLVED"
        assert rows["0000777777"]["ticker"] is None
        assert rows["0000777777"]["ticker_state"] == "UNKNOWN_TICKER"


def test_resolution_prefers_relisting_cik(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid = store.upsert_event(
            conn, cik="0000111111", event_type="ch11_emergence",
            anchor_accession="0000111111-24-000001", company_name="ACME HOLDINGS",
            detection_date="2024-01-05", source_mode="daily",
        )
        store.merge_event_detail(conn, event_id=eid, detail={"relisting_cik": "0000320193"})
        result = resolve_unknown_tickers(conn, mapping={"AAPL": "320193"})
        assert result == {"resolved": 1, "still_unknown": 0}
        row = conn.execute("SELECT * FROM corporate_events WHERE id=?", (eid,)).fetchone()
        # Resolved via the relisting CIK, never the cancelled debtor CIK.
        assert row["ticker"] == "AAPL"
