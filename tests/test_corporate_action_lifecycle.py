"""Corporate-action lifecycle: Form 25 protection, append-preserving scan
rows, gap self-healing, CIK flag matching, removal propagation, filing-watch
retirement, sweep-scope exclusion."""

from __future__ import annotations

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


def _conn(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Form 25 protection + append-preserving scan rows
# ---------------------------------------------------------------------------


def test_form_25_maps_to_delisting_notice_protection():
    from app.events.detectors import queue_protection_type_for_form

    assert queue_protection_type_for_form("25") == "delisting_notice"
    assert queue_protection_type_for_form("25-NSE") == "delisting_notice"
    assert queue_protection_type_for_form("10-K") is None


def test_record_scan_never_downgrades_ok_row(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.events import store

    conn = _conn(db_path)
    store.record_scan(conn, scan_date="2026-07-01", mode="daily", status="OK")
    # A later failed re-poll must NOT replace the OK record.
    store.record_scan(conn, scan_date="2026-07-01", mode="daily", status="ERROR")
    row = conn.execute(
        "SELECT status FROM corporate_event_scans WHERE scan_date='2026-07-01'"
    ).fetchone()
    assert row["status"] == "OK"
    # An OK re-poll may refresh an OK row, and OK replaces ERROR.
    store.record_scan(conn, scan_date="2026-07-02", mode="daily", status="ERROR")
    store.record_scan(conn, scan_date="2026-07-02", mode="daily", status="OK")
    row = conn.execute(
        "SELECT status FROM corporate_event_scans WHERE scan_date='2026-07-02'"
    ).fetchone()
    assert row["status"] == "OK"
    conn.close()


# ---------------------------------------------------------------------------
# Scan-gap self-healing
# ---------------------------------------------------------------------------


def test_find_scan_gaps_lists_business_days_without_ok_rows(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.events import store
    from app.events.poller import find_scan_gaps

    conn = _conn(db_path)
    # today = Wednesday 2026-07-15; window covers the prior week.
    store.record_scan(conn, scan_date="2026-07-13", mode="daily", status="OK")
    store.record_scan(conn, scan_date="2026-07-10", mode="daily", status="ERROR")
    conn.commit()

    gaps = find_scan_gaps(conn, window_days=7, today_et_str="2026-07-15")
    conn.close()
    # 7/8 Wed, 7/9 Thu, 7/10 Fri (ERROR counts as gap), 7/14 Tue are missing;
    # 7/11-12 is a weekend; 7/13 Mon is OK.
    assert gaps == ["2026-07-08", "2026-07-09", "2026-07-10", "2026-07-14"]


def test_run_scan_gap_backfill_closes_no_index_days(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.events import poller, store

    # Only 7/14 is missing; the (mocked) backfill recovers nothing — the day
    # had no index rows (holiday) and must be closed with a zero-count OK row.
    conn = _conn(db_path)
    day = "2026-07-13"
    store.record_scan(conn, scan_date=day, mode="daily", status="OK")
    conn.commit()
    conn.close()

    calls = {}

    def fake_backfill(start, end, **kwargs):
        calls["range"] = (start, end)
        return []

    monkeypatch.setattr(poller, "backfill", fake_backfill)
    result = poller.run_scan_gap_backfill(window_days=7, today_et="2026-07-15")

    assert calls["range"] == ("2026-07-08", "2026-07-14")
    assert result["unrecovered"] == []
    assert "2026-07-14" in result["closed_no_index"]
    conn = _conn(db_path)
    row = conn.execute(
        "SELECT status, mode FROM corporate_event_scans WHERE scan_date='2026-07-14'"
    ).fetchone()
    conn.close()
    assert row["status"] == "OK"
    assert row["mode"] == "gap_backfill_no_index"


# ---------------------------------------------------------------------------
# CIK-based flag matching
# ---------------------------------------------------------------------------


def test_event_pending_matches_by_cik_despite_unknown_ticker(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.events import store
    from app.events.flags import sync_event_pending_flags
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update, get_latest

    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            buy_price_target=10.0,
            source_run_id="run1",
            added_at="2026-07-01T00:00:00+00:00",
        ),
        db_path=db_path,
    )

    conn = _conn(db_path)
    # Open merger event by CIK with NO resolved ticker (UNKNOWN_TICKER).
    store.upsert_event(
        conn,
        cik="1234567",
        event_type="merger",
        anchor_accession="0001-26-000001",
        company_name="Alpha Co",
        detection_date="2026-07-01",
        source_mode="daily",
    )
    conn.commit()

    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda refresh_if_missing=False: {"AAA": "0001234567"},
    )
    summary = sync_event_pending_flags(conn)
    conn.commit()
    conn.close()

    assert summary["flagged"] == 1
    entry = get_latest("AAA", db_path=db_path)
    assert entry is not None
    # Flag present via CIK even though the event's ticker is unresolved.
    row_flags = entry.event_pending if hasattr(entry, "event_pending") else None
    if row_flags is None:
        conn = _conn(db_path)
        row = conn.execute(
            "SELECT event_pending FROM watchlist WHERE ticker='AAA'"
        ).fetchone()
        conn.close()
        row_flags = row["event_pending"]
    assert row_flags == "EVENT_PENDING:MERGER"


# ---------------------------------------------------------------------------
# Retroactive intake protection
# ---------------------------------------------------------------------------


def test_retroactive_queue_protection_opens_events_from_cached_filings(
    monkeypatch, tmp_path
):
    db_path = _env(monkeypatch, tmp_path)
    from app.events.grade_time import retroactive_queue_protection

    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda refresh_if_missing=False: {"AAA": "0001234567"},
    )
    recent = {
        "form": ["25", "10-Q", "425"],
        "filingDate": ["2026-07-01", "2026-06-20", "2025-01-05"],
        "accessionNumber": ["0001-26-000025", "0001-26-000010", "0001-25-000425"],
    }
    monkeypatch.setattr(
        "app.universe.sector_universe.load_company_submissions",
        lambda cik, refresh_if_missing=True, **kw: {
            "name": "Alpha Co",
            "filings": {"recent": recent},
        },
    )

    conn = _conn(db_path)
    summary = retroactive_queue_protection(
        conn, ["AAA"], scan_date="2026-07-15", lookback_days=270
    )
    rows = conn.execute(
        "SELECT event_type, ticker, cik FROM corporate_events ORDER BY event_type"
    ).fetchall()
    conn.close()

    # Form 25 within the lookback opens delisting_notice; the 425 from 2025
    # is outside the window; the 10-Q maps to nothing.
    assert summary["events_created"] == 1
    assert len(rows) == 1
    assert rows[0]["event_type"] == "delisting_notice"
    assert rows[0]["ticker"] == "AAA"


# ---------------------------------------------------------------------------
# Removal propagation + filing-watch retirement
# ---------------------------------------------------------------------------


def test_propagate_removals_marks_watchlist_removed(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.universe.registrant_intake import _propagate_removals
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update, get_latest

    add_or_update(
        WatchlistEntry(
            ticker="GONE",
            status="DEPLOY_READY",
            buy_price_target=5.0,
            source_run_id="run1",
            added_at="2026-07-01T00:00:00+00:00",
        ),
        db_path=db_path,
    )

    counts = _propagate_removals(
        [{"cik": "0001234567", "ticker": "GONE"}], db_path=db_path
    )

    assert counts["watchlist_removed"] == 1
    entry = get_latest("GONE", db_path=db_path)
    # get_latest hides REMOVED rows on some paths; read raw for certainty.
    conn = _conn(db_path)
    row = conn.execute(
        "SELECT status, status_reason FROM watchlist WHERE ticker='GONE'"
    ).fetchone()
    conn.close()
    assert row["status"] == "REMOVED"
    assert row["status_reason"].startswith("UNIVERSE_EXIT")


# ---------------------------------------------------------------------------
# Sweep scope excludes removed registrants
# ---------------------------------------------------------------------------


def _seed_sector_name(conn, ticker: str) -> None:
    conn.execute(
        "INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at) "
        "VALUES (?, '2026-07-01', 'energy', 1.0, '[]', '2026-07-01T00:00:00+00:00')",
        (ticker,),
    )
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
        "VALUES (?, '2026-07-01', 'scorecard', '{}', '{}', '[]', '2026-07-01T00:00:00+00:00')",
        (ticker,),
    )


def test_sweep_scope_excludes_removed_registrants(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.sector.scan import load_sector_tickers_classified

    conn = _conn(db_path)
    _seed_sector_name(conn, "LIVE")
    _seed_sector_name(conn, "DEAD")
    now = "2026-07-15T00:00:00+00:00"
    conn.execute(
        "INSERT INTO sec_registrants(cik, primary_ticker, all_tickers, exchange_scope, operating_status, in_scope, first_seen_at, last_seen_at, removed_at) "
        "VALUES ('1', 'DEAD', '[]', 'IN_SCOPE', 'OPERATING', 1, ?, ?, ?)",
        (now, now, now),
    )
    conn.commit()
    conn.close()

    rows, _ = load_sector_tickers_classified(sector="energy", db_path=db_path)
    tickers = [row[0] for row in rows]
    assert tickers == ["LIVE"]


def test_sweep_scope_tolerates_missing_registrant_table(monkeypatch, tmp_path):
    # A DB without sec_registrants (minimal test DBs) stays sweepable.
    db_path = tmp_path / "mini.db"
    conn = _conn(db_path)
    conn.execute(
        "CREATE TABLE sector_inference (ticker TEXT, as_of_date TEXT, inferred_sector TEXT, score REAL, derived_from TEXT, created_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE valuations (ticker TEXT, as_of_date TEXT, method TEXT, inputs_json TEXT, outputs_json TEXT, warnings_json TEXT, created_at TEXT)"
    )
    _seed_sector_name(conn, "AAA")
    conn.commit()
    conn.close()
    from app.sector.scan import load_sector_tickers_classified

    rows, _ = load_sector_tickers_classified(sector="energy", db_path=db_path)
    assert [row[0] for row in rows] == ["AAA"]


# ---------------------------------------------------------------------------
# Digest universe-exit section
# ---------------------------------------------------------------------------


def test_digest_suppresses_unbound_universe_exit_history(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.digest import render_digest
    from app.watchlist.store import add_or_update, mark_status

    add_or_update(
        WatchlistEntry(
            ticker="GONE",
            status="ACTIVE",
            buy_price_target=5.0,
            source_run_id="run1",
            added_at="2026-07-01T00:00:00+00:00",
        ),
        db_path=db_path,
    )
    mark_status(
        "GONE",
        "REMOVED",
        "UNIVERSE_EXIT:left SEC exchange registry",
        source="universe_sync",
        db_path=db_path,
    )

    digest = render_digest(days_back=30, db_path=db_path)
    assert "## Newly Removed (Universe Exits)" in digest
    exits_section = digest.split("## Newly Removed (Universe Exits)", 1)[1].split("##", 1)[0]
    # watchlist_history is mutable auxiliary state with no independently
    # verifiable source binding. The transition remains in the database ledger
    # but must not become a current decision claim in a product digest.
    assert "GONE" not in exits_section
    assert "UNIVERSE_EXIT" not in exits_section
    assert "(none in this window)" in exits_section

    conn = _conn(db_path)
    history = conn.execute(
        """
        SELECT h.field_name, h.new_value
        FROM watchlist_history h
        JOIN watchlist w ON w.id = h.watchlist_id
        WHERE w.ticker = 'GONE'
        ORDER BY h.id
        """
    ).fetchall()
    conn.close()
    assert ("status", "REMOVED") in {
        (row["field_name"], row["new_value"]) for row in history
    }
