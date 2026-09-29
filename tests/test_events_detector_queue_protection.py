from __future__ import annotations


from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.detectors import detect_queue_protection
from app.events.index_feed import IndexRow


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def _row(cik, form_type, date_filed, accession, name="NCS Multistage Holdings, Inc."):
    return IndexRow(
        cik=cik, company_name=name, form_type=form_type, date_filed=date_filed,
        file_name=f"edgar/data/{int(cik)}/{accession}.txt",
    )


class FakeSubmissions:
    def __init__(self, items=None):
        self.items = items or {}

    def items_for(self, cik, accession):
        return self.items.get((cik, accession))

    def prior_item103(self, cik, *, before):
        return None


WATCHLIST = {"0001692427": "NCSM", "0000034067": "BOOM"}


def test_425_creates_merger_event_with_ticker(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [_row("0001692427", "425", "2026-06-01", "0001104659-26-061001")]
    with get_db() as conn:
        counters = detect_queue_protection(
            conn, rows, scan_date="2026-06-01", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=FakeSubmissions(),
        )
        assert counters.events_created == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["event_type"] == "merger"
        assert event["ticker"] == "NCSM"
        assert event["ticker_state"] == "RESOLVED"
        assert event["status"] == "DETECTED"
        assert event["detection_date"] == "2026-06-01"
        flags = store.open_queue_protection_events(conn, ciks=["0001692427"])
        assert store.event_pending_flag(flags["0001692427"][0]["event_type"]) == "EVENT_PENDING:MERGER"


def test_non_watchlist_cik_ignored(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [_row("0009999999", "425", "2026-06-01", "0001104659-26-061001")]
    with get_db() as conn:
        counters = detect_queue_protection(
            conn, rows, scan_date="2026-06-01", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=FakeSubmissions(),
        )
        assert counters.events_created == 0
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 0


def test_multiple_425s_attach_to_one_event(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        for day, acc in (
            ("2026-06-01", "0001104659-26-061001"),
            ("2026-06-01", "0001104659-26-061002"),
            ("2026-06-02", "0001104659-26-061101"),
        ):
            detect_queue_protection(
                conn, [_row("0001692427", "425", day, acc)],
                scan_date=day, source_mode="daily",
                watchlist_ciks=WATCHLIST, submissions=FakeSubmissions(),
            )
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 1
        assert conn.execute("SELECT COUNT(*) c FROM corporate_event_filings").fetchone()["c"] == 3


def test_merger_proxy_and_tender_forms(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [
        _row("0001692427", "DEFM14A", "2026-06-03", "0001104659-26-062001"),
        _row("0000034067", "SC TO-T", "2026-06-03", "0001104659-26-062002", name="DMC GLOBAL"),
    ]
    with get_db() as conn:
        detect_queue_protection(
            conn, rows, scan_date="2026-06-03", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=FakeSubmissions(),
        )
        types = {
            r["cik"]: r["event_type"]
            for r in conn.execute("SELECT cik, event_type FROM corporate_events").fetchall()
        }
        assert types == {"0001692427": "merger", "0000034067": "merger"}


def test_8k_items_map_to_event_types(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={
        ("0001692427", "0001692427-26-000020"): "1.01,2.05,9.01",
        ("0000034067", "0000034067-26-000021"): "3.01",
    })
    rows = [
        _row("0001692427", "8-K", "2026-06-02", "0001692427-26-000020"),
        _row("0000034067", "8-K", "2026-06-02", "0000034067-26-000021", name="DMC GLOBAL"),
    ]
    with get_db() as conn:
        counters = detect_queue_protection(
            conn, rows, scan_date="2026-06-02", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=subs,
        )
        assert counters.events_created == 3
        rows_db = conn.execute(
            "SELECT cik, event_type FROM corporate_events ORDER BY cik, event_type"
        ).fetchall()
        assert [(r["cik"], r["event_type"]) for r in rows_db] == [
            ("0000034067", "delisting_notice"),
            ("0001692427", "material_agreement"),
            ("0001692427", "restructuring"),
        ]


def test_8k_item_204_opens_obligation_acceleration(monkeypatch, tmp_path):
    """The BTM class: items 1.03 + 2.04 on one 8-K open two distinct events."""
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={
        ("0000034067", "0001193125-26-227832"): "1.03,2.04,9.01",
    })
    rows = [_row("0000034067", "8-K", "2026-05-18", "0001193125-26-227832", name="DMC GLOBAL")]
    with get_db() as conn:
        counters = detect_queue_protection(
            conn, rows, scan_date="2026-05-18", source_mode="backfill",
            watchlist_ciks=WATCHLIST, submissions=subs,
        )
        assert counters.events_created == 2
        types = sorted(
            r["event_type"] for r in conn.execute(
                "SELECT event_type FROM corporate_events"
            ).fetchall()
        )
        assert types == ["bankruptcy", "obligation_acceleration"]
        assert store.event_pending_flag("obligation_acceleration") == (
            "EVENT_PENDING:OBLIGATION_ACCELERATION"
        )


def test_8k_fetch_failure_records_skip(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [_row("0001692427", "8-K", "2026-06-02", "0001692427-26-000020")]
    with get_db() as conn:
        detect_queue_protection(
            conn, rows, scan_date="2026-06-02", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=FakeSubmissions(),
        )
        skip = conn.execute("SELECT * FROM corporate_event_skips").fetchone()
        assert skip["reason_code"] == "SUBMISSIONS_FETCH_FAILED"
        assert skip["detector"] == "queue_protection"


def test_activist_and_dilution_forms(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [
        _row("0001692427", "SC 13D", "2026-06-04", "0001104659-26-063001"),
        _row("0000034067", "424B5", "2026-06-04", "0001104659-26-063002", name="DMC GLOBAL"),
        _row("0000034067", "S-1", "2026-06-05", "0001104659-26-063003", name="DMC GLOBAL"),
    ]
    with get_db() as conn:
        detect_queue_protection(
            conn, rows, scan_date="2026-06-05", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=FakeSubmissions(),
        )
        rows_db = conn.execute(
            "SELECT cik, event_type FROM corporate_events ORDER BY id"
        ).fetchall()
        assert [(r["cik"], r["event_type"]) for r in rows_db] == [
            ("0001692427", "activist"),
            ("0000034067", "dilution"),
        ]
        # The S-1 attaches to the open dilution event rather than duplicating it.
        n_filings = conn.execute(
            "SELECT COUNT(*) c FROM corporate_event_filings WHERE event_id = 2"
        ).fetchone()["c"]
        assert n_filings == 2


def test_rerun_zero_deltas(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={("0001692427", "0001692427-26-000020"): "1.01"})
    rows = [
        _row("0001692427", "425", "2026-06-01", "0001104659-26-061001"),
        _row("0001692427", "8-K", "2026-06-01", "0001692427-26-000020"),
    ]
    with get_db() as conn:
        detect_queue_protection(
            conn, rows, scan_date="2026-06-01", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=subs,
        )
        counters = detect_queue_protection(
            conn, rows, scan_date="2026-06-01", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=subs,
        )
        assert counters.events_created == 0
        assert counters.events_updated == 0
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 2


def test_disposed_event_reopens_on_new_filing(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        detect_queue_protection(
            conn, [_row("0001692427", "425", "2026-06-01", "0001104659-26-061001")],
            scan_date="2026-06-01", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=FakeSubmissions(),
        )
        eid = conn.execute("SELECT id FROM corporate_events").fetchone()["id"]
        store.mark_decided(conn, event_id=eid, note="reviewed")
        assert store.open_queue_protection_events(conn, ciks=["0001692427"]) == {}
        # A NEW filing after disposal opens a fresh event (new anchor).
        detect_queue_protection(
            conn, [_row("0001692427", "425", "2026-06-08", "0001104659-26-068001")],
            scan_date="2026-06-08", source_mode="daily",
            watchlist_ciks=WATCHLIST, submissions=FakeSubmissions(),
        )
        open_events = store.open_queue_protection_events(conn, ciks=["0001692427"])
        assert len(open_events["0001692427"]) == 1
        assert open_events["0001692427"][0]["anchor_accession"] == "0001104659-26-068001"
