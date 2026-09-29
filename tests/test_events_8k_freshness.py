"""At-target 8-K freshness gate: any fresh 8-K by an at-target name opens an
unreviewed_8k queue-protection event unless the accession is already covered
by another event."""

from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.detectors import detect_atarget_8k_freshness, detect_queue_protection
from app.events.flags import sync_event_pending_flags
from app.events.index_feed import IndexRow
from app.events.poller import atarget_cik_map
from app.watchlist.schema import ensure_watchlist_schema


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


def _add_watch_row(conn, ticker, status="DEPLOY_READY", conviction_grade="ACTIONABLE"):
    conn.execute(
        "INSERT INTO watchlist(ticker, status, conviction_grade, source_run_id, added_at) "
        "VALUES(?, ?, ?, ?, ?)",
        (ticker, status, conviction_grade, f"run_{ticker}", "2026-06-01T00:00:00Z"),
    )


class FakeFilings:
    """FilingsWindowReader over a fixed per-CIK filing list."""

    def __init__(self, by_cik=None, fail_ciks=()):
        self.by_cik = by_cik or {}
        self.fail_ciks = set(fail_ciks)

    def filings_window(self, cik, *, start, end):
        if cik in self.fail_ciks:
            return None
        return [
            f for f in self.by_cik.get(cik, [])
            if start <= f["filing_date"] <= end
        ]


def _8k(accession, filing_date, items=""):
    return {
        "accession": accession,
        "form": "8-K",
        "filing_date": filing_date,
        "items": items,
        "primary_document": "doc.htm",
    }


ATARGET = {"0001692427": "NCSM"}


def test_fresh_8k_opens_unreviewed_event_and_blocks_queue(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    filings = FakeFilings(by_cik={
        "0001692427": [_8k("0001692427-26-000030", "2026-06-25", items="8.01,9.01")],
    })
    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        counters = detect_atarget_8k_freshness(
            conn, ATARGET, scan_date="2026-07-01", lookback_days=21,
            source_mode="daily", filings=filings,
        )
        assert counters.events_created == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["event_type"] == "unreviewed_8k"
        assert event["anchor_accession"] == "0001692427-26-000030"
        assert event["ticker"] == "NCSM"
        assert event["status"] == "DETECTED"
        assert event["detection_date"] == "2026-06-25"
        sync_event_pending_flags(conn)
        row = conn.execute("SELECT * FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] == "EVENT_PENDING:UNREVIEWED_8K"


def test_accession_covered_by_item_detector_is_skipped(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    class Subs:
        def items_for(self, cik, accession):
            return "1.01,9.01"

        def prior_item103(self, cik, *, before):
            return None

    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        rows = [IndexRow(
            cik="0001692427", company_name="NCS MULTISTAGE", form_type="8-K",
            date_filed="2026-06-25",
            file_name="edgar/data/1692427/0001692427-26-000030.txt",
        )]
        detect_queue_protection(
            conn, rows, scan_date="2026-06-25", source_mode="daily",
            watchlist_ciks=ATARGET, submissions=Subs(),
        )
        filings = FakeFilings(by_cik={
            "0001692427": [_8k("0001692427-26-000030", "2026-06-25", items="1.01,9.01")],
        })
        counters = detect_atarget_8k_freshness(
            conn, ATARGET, scan_date="2026-07-01", lookback_days=21,
            source_mode="daily", filings=filings,
        )
        assert counters.events_created == 0
        types = [r["event_type"] for r in conn.execute(
            "SELECT event_type FROM corporate_events"
        ).fetchall()]
        assert types == ["material_agreement"]


def test_stale_8k_outside_lookback_ignored(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    filings = FakeFilings(by_cik={
        "0001692427": [_8k("0001692427-26-000010", "2026-05-01")],
    })
    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        counters = detect_atarget_8k_freshness(
            conn, ATARGET, scan_date="2026-07-01", lookback_days=21,
            source_mode="daily", filings=filings,
        )
        assert counters.events_created == 0


def test_rerun_is_idempotent(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    filings = FakeFilings(by_cik={
        "0001692427": [_8k("0001692427-26-000030", "2026-06-25", items="8.01")],
    })
    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        for _ in range(2):
            counters = detect_atarget_8k_freshness(
                conn, ATARGET, scan_date="2026-07-01", lookback_days=21,
                source_mode="daily", filings=filings,
            )
        assert counters.events_created == 0
        n = conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"]
        assert n == 1


def test_disposed_event_not_recreated(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    filings = FakeFilings(by_cik={
        "0001692427": [_8k("0001692427-26-000030", "2026-06-25", items="8.01")],
    })
    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        detect_atarget_8k_freshness(
            conn, ATARGET, scan_date="2026-07-01", lookback_days=21,
            source_mode="daily", filings=filings,
        )
        eid = conn.execute("SELECT id FROM corporate_events").fetchone()["id"]
        store.mark_decided(conn, event_id=eid, note="read it: routine earnings 8-K")
        sync_event_pending_flags(conn)
        counters = detect_atarget_8k_freshness(
            conn, ATARGET, scan_date="2026-07-02", lookback_days=21,
            source_mode="daily", filings=filings,
        )
        assert counters.events_created == 0
        row = conn.execute("SELECT event_pending FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] is None


def test_fetch_failure_counted_not_fatal(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "NCSM")
        counters = detect_atarget_8k_freshness(
            conn, ATARGET, scan_date="2026-07-01", lookback_days=21,
            source_mode="daily", filings=FakeFilings(fail_ciks={"0001692427"}),
        )
        assert counters.events_created == 0
        assert counters.counts == {"n_8k_freshness_fetch_failed": 1}


def test_atarget_cik_map_filters_status_and_grade(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "NCSM", status="DEPLOY_READY")
        _add_watch_row(conn, "BOOM", status="ACTIVE")
        _add_watch_row(conn, "AVDT", status="DEPLOY_READY", conviction_grade="AVOID")
        watchlist_ciks = {
            "0001692427": "NCSM", "0000034067": "BOOM", "0000099999": "AVDT",
        }
        assert atarget_cik_map(conn, watchlist_ciks) == {"0001692427": "NCSM"}


def test_item_402_maps_to_restatement(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    class Subs:
        def items_for(self, cik, accession):
            return "4.02"

        def prior_item103(self, cik, *, before):
            return None

    with get_db() as conn:
        rows = [IndexRow(
            cik="0001692427", company_name="NCS MULTISTAGE", form_type="8-K",
            date_filed="2026-06-25",
            file_name="edgar/data/1692427/0001692427-26-000031.txt",
        )]
        counters = detect_queue_protection(
            conn, rows, scan_date="2026-06-25", source_mode="daily",
            watchlist_ciks=ATARGET, submissions=Subs(),
        )
        assert counters.events_created == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["event_type"] == "restatement"
        assert store.event_pending_flag("restatement") == "EVENT_PENDING:RESTATEMENT"
