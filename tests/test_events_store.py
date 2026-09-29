from __future__ import annotations

import json

import pytest

from app.config import get_config
from app.db import get_db, init_db
from app.events import store


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def _seed_event(conn, **overrides):
    kwargs = dict(
        cik="0000123456",
        event_type="spinoff",
        anchor_accession="0001234567-26-000001",
        company_name="SPINCO CORP",
        detection_date="2026-01-05",
        source_mode="daily",
    )
    kwargs.update(overrides)
    return store.upsert_event(conn, **kwargs)


def test_upsert_event_idempotent(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid1 = _seed_event(conn)
        eid2 = _seed_event(conn)
        assert eid1 == eid2
        n = conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"]
        assert n == 1


def test_attach_filing_idempotent(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid = _seed_event(conn)
        first = store.attach_filing(
            conn, event_id=eid, cik="0000123456", accession="0001234567-26-000001",
            form_type="10-12B", filing_date="2026-01-05", role="REGISTRATION",
        )
        second = store.attach_filing(
            conn, event_id=eid, cik="0000123456", accession="0001234567-26-000001",
            form_type="10-12B", filing_date="2026-01-05", role="REGISTRATION",
        )
        assert first is True
        assert second is False
        n = conn.execute("SELECT COUNT(*) c FROM corporate_event_filings").fetchone()["c"]
        assert n == 1


def test_lifecycle_forward_only(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid = _seed_event(conn)
        assert store.mark_qualified(conn, event_id=eid, qualification_date="2026-02-01") is True
        row = conn.execute("SELECT * FROM corporate_events WHERE id=?", (eid,)).fetchone()
        qualified_at = row["qualified_at"]
        assert row["status"] == "QUALIFIED"
        assert row["qualification_date"] == "2026-02-01"
        # Second qualify is a no-op and does not re-stamp.
        assert store.mark_qualified(conn, event_id=eid, qualification_date="2026-03-01") is False
        row = conn.execute("SELECT * FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert row["qualified_at"] == qualified_at
        assert row["qualification_date"] == "2026-02-01"

        assert store.mark_surfaced(conn, event_id=eid) is True
        # Backward refusal.
        assert store.mark_qualified(conn, event_id=eid, qualification_date="2026-04-01") is False
        row = conn.execute("SELECT * FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert row["status"] == "SURFACED"


def test_mark_expired_rules(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid = _seed_event(conn)
        store.mark_qualified(conn, event_id=eid, qualification_date="2026-02-01")
        store.mark_surfaced(conn, event_id=eid)
        assert store.mark_expired(conn, event_id=eid, expiry_reason="NO_EFFECTIVENESS_270D") is True
        row = conn.execute("SELECT * FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert row["status"] == "EXPIRED"
        assert row["expiry_reason"] == "NO_EFFECTIVENESS_270D"

        eid2 = _seed_event(conn, anchor_accession="0001234567-26-000002")
        store.mark_decided(conn, event_id=eid2, note="reviewed")
        assert store.mark_expired(conn, event_id=eid2, expiry_reason="NO_EFFECTIVENESS_270D") is False


def test_mark_decided_disposes_open_event(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid = _seed_event(conn, event_type="merger")
        assert store.mark_decided(conn, event_id=eid, note="analyst reviewed: hold") is True
        row = conn.execute("SELECT * FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert row["status"] == "DECIDED"
        assert row["decided_at"] is not None
        assert json.loads(row["detail_json"])["decision_note"] == "analyst reviewed: hold"
        # Second call is a no-op.
        assert store.mark_decided(conn, event_id=eid, note="again") is False


def test_record_skip_validates_reason(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        with pytest.raises(ValueError):
            store.record_skip(
                conn, scan_date="2026-01-05", cik="0000123456",
                accession="0001234567-26-000001", form_type="S-1",
                detector="busted_ipo", reason_code="NOT_A_REAL_REASON",
            )
        first = store.record_skip(
            conn, scan_date="2026-01-05", cik="0000123456",
            accession="0001234567-26-000001", form_type="S-1",
            detector="busted_ipo", reason_code="PRIOR_REPORTING_S1",
        )
        second = store.record_skip(
            conn, scan_date="2026-01-05", cik="0000123456",
            accession="0001234567-26-000001", form_type="S-1",
            detector="busted_ipo", reason_code="PRIOR_REPORTING_S1",
        )
        assert first is True
        assert second is False
        n = conn.execute("SELECT COUNT(*) c FROM corporate_event_skips").fetchone()["c"]
        assert n == 1


def test_record_scan_upserts_on_scan_date(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        store.record_scan(
            conn, scan_date="2026-01-05", mode="daily", status="OK",
            counters={"index_rows": 10, "candidate_rows": 2},
        )
        store.record_scan(
            conn, scan_date="2026-01-05", mode="daily", status="OK",
            counters={"index_rows": 12, "candidate_rows": 3},
        )
        rows = conn.execute("SELECT * FROM corporate_event_scans").fetchall()
        assert len(rows) == 1
        assert rows[0]["index_rows"] == 12
        assert rows[0]["candidate_rows"] == 3


def test_merge_event_detail(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid = _seed_event(conn, detail={"listing_track": "otc"})
        store.merge_event_detail(conn, event_id=eid, detail={"relisting_cik": "0000999999"})
        row = conn.execute("SELECT detail_json FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert json.loads(row["detail_json"]) == {
            "listing_track": "otc",
            "relisting_cik": "0000999999",
        }
        store.merge_event_detail(conn, event_id=eid, detail={"relisting_cik": "0000999999"})
        row = conn.execute("SELECT detail_json FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert json.loads(row["detail_json"]) == {
            "listing_track": "otc",
            "relisting_cik": "0000999999",
        }


def test_set_and_reset_ticker(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid = _seed_event(conn)
        store.set_ticker(conn, event_id=eid, ticker="OLDT")
        row = conn.execute("SELECT * FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert row["ticker"] == "OLDT"
        assert row["ticker_state"] == "RESOLVED"
        store.reset_ticker(conn, event_id=eid)
        row = conn.execute("SELECT * FROM corporate_events WHERE id=?", (eid,)).fetchone()
        assert row["ticker"] is None
        assert row["ticker_state"] == "UNKNOWN_TICKER"


def test_find_active_event_skips_terminal(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        eid = _seed_event(conn)
        found = store.find_active_event(conn, cik="0000123456", event_type="spinoff")
        assert found is not None and found["id"] == eid
        store.mark_expired(conn, event_id=eid, expiry_reason="NO_EFFECTIVENESS_270D")
        assert store.find_active_event(conn, cik="0000123456", event_type="spinoff") is None


def test_open_queue_protection_events_for_ciks(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        merger_id = _seed_event(
            conn, event_type="merger", cik="0001692427",
            anchor_accession="0001104659-26-061001", company_name="NCS MULTISTAGE",
        )
        _seed_event(conn, event_type="spinoff", cik="0001692427",
                    anchor_accession="0001104659-26-061002", company_name="NCS MULTISTAGE")
        disposed = _seed_event(
            conn, event_type="activist", cik="0000034067",
            anchor_accession="0001104659-26-061003", company_name="DMC GLOBAL",
        )
        store.mark_decided(conn, event_id=disposed, note="done")
        by_cik = store.open_queue_protection_events(conn, ciks=["0001692427", "0000034067"])
        # Spinoff is an opportunity event, not queue-protection; disposed activist excluded.
        assert list(by_cik.keys()) == ["0001692427"]
        assert by_cik["0001692427"][0]["id"] == merger_id
        assert by_cik["0001692427"][0]["event_type"] == "merger"


def test_event_pending_flag_string():
    assert store.event_pending_flag("merger") == "EVENT_PENDING:MERGER"
    assert store.event_pending_flag("delisting_notice") == "EVENT_PENDING:DELISTING_NOTICE"
