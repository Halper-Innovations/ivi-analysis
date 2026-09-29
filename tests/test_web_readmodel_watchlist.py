from __future__ import annotations

import sqlite3
from pathlib import Path

from app.watchlist.contract import WatchlistEntry
from app.watchlist.store import add_or_update, add_price_snapshot, watchlist_queue
from app.web.readmodel.db import open_readonly
from app.web.readmodel.watchlist import queue_rows


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    return db_path


def _entry(
    *,
    ticker: str,
    status: str = "ACTIVE",
    conviction_grade: str = "WATCHLIST_ONLY",
    confidence: str | None = "MODERATE",
    buy_price_target: float = 80.0,
    source_run_id: str = "sector_run_1",
) -> WatchlistEntry:
    return WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade=conviction_grade,
        confidence=confidence,
        conviction_source="company_autonomy",
        scan_family="normal",
        valuation_anchor_method="DCF",
        valuation_anchor_value=106.67,
        buy_price_target=buy_price_target,
        current_price_at_addition=100.0,
        thesis_text="Durable candidate with a buy-price anchor.",
        key_risks=["Margin compression"],
        falsifiers=["Revenue decline persists"],
        open_questions=["Can margins normalize?"],
        source_run_id=source_run_id,
        source_sector="industrial_tech",
        added_at="2026-05-08T12:00:00+00:00",
    )


def _seed(db_path: Path) -> None:
    active_id = add_or_update(_entry(ticker="AAA"), db_path=db_path)
    deploy_id = add_or_update(
        _entry(ticker="BBB", status="DEPLOY_READY", conviction_grade="ACTIONABLE", confidence="HIGH"),
        db_path=db_path,
    )
    pending_id = add_or_update(
        _entry(ticker="CCC", status="DEPLOY_READY", conviction_grade="ACTIONABLE", confidence="HIGH"),
        db_path=db_path,
    )
    suspect_id = add_or_update(_entry(ticker="DDD", status="PRICE_DATA_SUSPECT"), db_path=db_path)
    add_price_snapshot(active_id, price=95.0, checked_at="2026-07-20T12:00:00+00:00", db_path=db_path)
    add_price_snapshot(deploy_id, price=78.0, checked_at="2026-07-20T12:00:00+00:00", db_path=db_path)
    add_price_snapshot(pending_id, price=77.0, checked_at="2026-07-20T12:00:00+00:00", db_path=db_path)
    add_price_snapshot(suspect_id, price=50.0, checked_at="2026-07-01T12:00:00+00:00", db_path=db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE watchlist SET event_pending = 'unreviewed_8k' WHERE id = ?", (pending_id,)
    )
    conn.commit()
    conn.close()


def test_queue_rows_matches_store_watchlist_queue(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed(db_path)
    expected = watchlist_queue(limit=500, include_price_suspect=True, db_path=db_path)
    conn = open_readonly(db_path)
    try:
        actual = queue_rows(conn)
    finally:
        conn.close()
    assert actual == expected
    assert len(actual) == 4


def test_queue_rows_over_readonly_connection_derives_presented_statuses(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed(db_path)
    conn = open_readonly(db_path)
    try:
        rows = queue_rows(conn)
    finally:
        conn.close()
    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["BBB"]["presented_status"] == "DEPLOY_READY"
    assert by_ticker["CCC"]["presented_status"] == "EVENT_PENDING"
    assert by_ticker["CCC"]["status"] == "DEPLOY_READY"
    assert by_ticker["CCC"]["event_pending"] == "unreviewed_8k"
    assert by_ticker["AAA"]["presented_status"] == "ACTIVE"
    assert by_ticker["DDD"]["presented_status"] == "PRICE_DATA_SUSPECT"
    assert by_ticker["BBB"]["latest_price"] == 78.0
    assert by_ticker["BBB"]["buy_price_target"] == 80.0
    assert by_ticker["BBB"]["distance_from_buy_pct"] == -2.5


def test_queue_rows_filters_pass_through(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed(db_path)
    conn = open_readonly(db_path)
    try:
        without_suspect = queue_rows(conn, include_price_suspect=False)
        limited = queue_rows(conn, limit=1)
    finally:
        conn.close()
    assert [row["ticker"] for row in without_suspect] == ["BBB", "CCC", "AAA"]
    assert [row["ticker"] for row in limited] == ["BBB"]
