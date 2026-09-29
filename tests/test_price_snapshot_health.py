from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.autonomous.financial_integrity import PRICE_BASIS_UNADJUSTED
from app.config import get_config
from app.db import get_db, init_db
from app.market.price_provider import PriceSnapshot
from app.ops.data_health import check_price_snapshot_coverage
from app.watchlist.market_data import refresh_current_watchlist_prices
from app.watchlist.schema import ensure_watchlist_schema


class StubPriceProvider:
    provider_name = "stub"

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot:
        return PriceSnapshot(
            ticker=ticker,
            as_of_date=as_of_date,
            price=25.0,
            currency="USD",
            source="yahoo",
            retrieved_at="2026-07-28T14:00:00+00:00",
            confidence="HIGH",
        )

    def get_last_diagnostic(self, ticker: str, as_of_date: str) -> dict[str, str]:
        return {"ticker": ticker, "as_of_date": as_of_date, "status": "OK"}


def _init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()
    return db_path


def _insert_watchlist_row(
    conn: sqlite3.Connection, *, ticker: str, run_id: str
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO watchlist(ticker, status, source_run_id, added_at)
        VALUES (?, 'ACTIVE', ?, '2026-07-01T00:00:00+00:00')
        """,
        (ticker, run_id),
    )
    return int(cursor.lastrowid)


def test_data_health_coverage_uses_canonical_rows_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with get_db() as conn:
        old_id = _insert_watchlist_row(conn, ticker="TDW", run_id="old")
        current_id = _insert_watchlist_row(conn, ticker="TDW", run_id="current")
        conn.execute(
            """
            INSERT INTO watchlist_price_snapshots(
                watchlist_id, price, checked_at, source, quote_as_of_date,
                currency, price_basis, quote_snapshot_id
            ) VALUES (?, 20.0, ?, 'yahoo', '2026-07-28', 'USD', ?, 'old-row')
            """,
            (
                old_id,
                datetime(2026, 7, 28, 14, tzinfo=timezone.utc).isoformat(),
                PRICE_BASIS_UNADJUSTED,
            ),
        )
    ok, detail = check_price_snapshot_coverage(
        db_path,
        now=datetime(2026, 7, 28, 15, tzinfo=timezone.utc),
    )
    assert not ok
    assert "0/1" in detail

    refresh_current_watchlist_prices(
        as_of_date="2026-07-28",
        db_path=db_path,
        provider=StubPriceProvider(),
    )
    ok, detail = check_price_snapshot_coverage(
        db_path,
        now=datetime(2026, 7, 28, 15, tzinfo=timezone.utc),
    )
    assert ok
    assert "1/1" in detail
    assert old_id != current_id


def test_coverage_not_activated_until_provenance_migration_applied(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE watchlist_price_snapshots(
            id INTEGER PRIMARY KEY,
            watchlist_id INTEGER,
            price REAL,
            checked_at TEXT,
            source TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE watchlist(
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            status TEXT,
            source_run_id TEXT,
            added_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    ok, detail = check_price_snapshot_coverage(
        db_path,
        now=datetime(2026, 7, 28, 15, tzinfo=timezone.utc),
    )
    assert ok
    assert detail == (
        "price snapshot coverage: not activated (provenance migration unapplied)"
    )
