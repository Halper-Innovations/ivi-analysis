from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.autonomous.financial_integrity import (
    PRICE_BASIS_UNADJUSTED,
    stable_quote_hash,
)
from app.config import get_config
from app.db import get_db, init_db
from app.market.price_provider import PriceSnapshot
from app.watchlist.market_data import (
    PriceRefreshMigrationRequired,
    build_cached_price_lookup,
    refresh_current_watchlist_prices,
)
from app.watchlist.schema import ensure_watchlist_schema
from app.watchlist.store import list_current_for_market_data


class StubPriceProvider:
    provider_name = "stub"

    def __init__(self, snapshots: dict[str, PriceSnapshot | None]) -> None:
        self.snapshots = snapshots
        self.calls: list[tuple[str, str]] = []

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        self.calls.append((ticker, as_of_date))
        return self.snapshots.get(ticker)

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
    conn: sqlite3.Connection,
    *,
    ticker: str,
    run_id: str,
    status: str = "ACTIVE",
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO watchlist(ticker, status, source_run_id, added_at)
        VALUES (?, ?, ?, '2026-07-01T00:00:00+00:00')
        """,
        (ticker, status, run_id),
    )
    return int(cursor.lastrowid)


def _snapshot(ticker: str, price: float = 25.0) -> PriceSnapshot:
    return PriceSnapshot(
        ticker=ticker,
        as_of_date="2026-07-28",
        price=price,
        currency="USD",
        source="yahoo",
        retrieved_at="2026-07-28T14:00:00+00:00",
        confidence="HIGH",
    )


def test_market_data_selector_returns_only_authoritative_current_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with get_db() as conn:
        old_id = _insert_watchlist_row(conn, ticker="TDW", run_id="run_old")
        current_id = _insert_watchlist_row(conn, ticker="TDW", run_id="run_current")
        _insert_watchlist_row(conn, ticker="GONE", run_id="run_removed", status="REMOVED")
        conn.execute(
            """
            INSERT INTO watchlist_price_snapshots(
                watchlist_id, price, checked_at, source
            ) VALUES (?, 18.0, '2026-07-20T00:00:00+00:00', 'legacy')
            """,
            (old_id,),
        )

    rows = list_current_for_market_data(db_path=db_path)

    assert [(row.ticker, row.id) for row in rows] == [("TDW", current_id)]


def test_refresh_writes_quote_only_to_canonical_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with get_db() as conn:
        old_id = _insert_watchlist_row(conn, ticker="TDW", run_id="run_old")
        current_id = _insert_watchlist_row(conn, ticker="TDW", run_id="run_current")
    provider = StubPriceProvider({"TDW": _snapshot("TDW", 26.5)})

    summary = refresh_current_watchlist_prices(
        as_of_date="2026-07-28", db_path=db_path, provider=provider
    )

    assert summary.to_dict() == {
        "as_of_date": "2026-07-28",
        "candidates": 1,
        "attempts": 1,
        "written": 1,
        "unavailable": [],
        "failed": {},
        "unavailable_count": 0,
        "failed_count": 0,
        "status": "OK",
    }
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT watchlist_id, price, quote_as_of_date, currency, price_basis,
                   quote_snapshot_id, price_quote_id
            FROM watchlist_price_snapshots
            ORDER BY id
            """
        ).fetchall()
        quote = conn.execute(
            """
            SELECT ticker, provider, as_of_date, price, status, price_basis,
                   split_adjustment_factor, quote_hash
            FROM price_quotes
            """
        ).fetchone()
    assert [int(row["watchlist_id"]) for row in rows] == [current_id]
    assert old_id != current_id
    assert rows[0]["price"] == 26.5
    assert rows[0]["quote_as_of_date"] == "2026-07-28"
    assert rows[0]["currency"] == "USD"
    assert rows[0]["price_basis"] == PRICE_BASIS_UNADJUSTED
    assert rows[0]["quote_snapshot_id"] == quote["quote_hash"]
    assert int(rows[0]["price_quote_id"]) > 0
    assert tuple(quote[:7]) == (
        "TDW",
        "yahoo",
        "2026-07-28",
        26.5,
        "OK",
        PRICE_BASIS_UNADJUSTED,
        1.0,
    )


def test_refresh_refuses_provider_calls_until_migration_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_watchlist_row(conn, ticker="ABG", run_id="run_abg")
        conn.execute("ALTER TABLE watchlist_price_snapshots RENAME TO migrated_snapshots")
        conn.execute(
            """
            CREATE TABLE watchlist_price_snapshots(
                id INTEGER PRIMARY KEY,
                watchlist_id INTEGER NOT NULL,
                price REAL NOT NULL,
                checked_at TEXT NOT NULL,
                source TEXT
            )
            """
        )
        conn.execute("DROP TABLE migrated_snapshots")
    provider = StubPriceProvider({"ABG": _snapshot("ABG")})

    with pytest.raises(PriceRefreshMigrationRequired, match="migration required"):
        refresh_current_watchlist_prices(
            as_of_date="2026-07-28", db_path=db_path, provider=provider
        )

    assert provider.calls == []


def test_cached_price_lookup_is_read_only_and_hash_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    quote = {
        "ticker": "ABG",
        "as_of_date": "2026-07-28",
        "price": 193.0,
        "currency": "USD",
        "source": "yahoo",
        "source_url": "",
        "price_basis": PRICE_BASIS_UNADJUSTED,
        "raw_price": 193.0,
        "split_adjustment_factor": 1.0,
        "split_effective_date": None,
    }
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO price_quotes(
                ticker, provider, as_of_date, price, currency, price_basis,
                split_adjustment_factor, source_url, status, fetched_at,
                expires_at, raw_json, quote_hash
            ) VALUES(
                'ABG', 'yahoo', '2026-07-28', 193.0, 'USD', ?, 1.0, '',
                'OK', '2026-07-28T14:00:00+00:00',
                '2026-07-29T14:00:00+00:00', '{}', ?
            )
            """,
            (PRICE_BASIS_UNADJUSTED, stable_quote_hash(quote)),
        )

    resolved = build_cached_price_lookup(db_path)("ABG", "2026-07-28")

    assert resolved == {**quote, "confidence": "HIGH"}


def _quote_row(
    ticker: str, as_of: str, price: float, source: str = "yahoo"
) -> dict:
    return {
        "ticker": ticker,
        "as_of_date": as_of,
        "price": price,
        "currency": "USD",
        "source": source,
        "source_url": "",
        "price_basis": PRICE_BASIS_UNADJUSTED,
        "raw_price": price,
        "split_adjustment_factor": 1.0,
        "split_effective_date": None,
    }


def _insert_quote(conn: sqlite3.Connection, quote: dict) -> None:
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, price_basis,
            split_adjustment_factor, source_url, status, fetched_at,
            expires_at, raw_json, quote_hash
        ) VALUES(?, ?, ?, ?, 'USD', ?, 1.0, '', 'OK',
            '2026-07-28T14:00:00+00:00', '2026-07-29T14:00:00+00:00',
            '{}', ?)
        """,
        (
            quote["ticker"],
            quote["source"],
            quote["as_of_date"],
            quote["price"],
            PRICE_BASIS_UNADJUSTED,
            stable_quote_hash(quote),
        ),
    )


def test_cached_price_lookup_refuses_tampered_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_quote(conn, _quote_row("ABG", "2026-07-28", 193.0))
        conn.execute(
            "UPDATE price_quotes SET price = 999.0 "
            "WHERE ticker = 'ABG' AND provider = 'yahoo'"
        )

    resolved = build_cached_price_lookup(db_path)("ABG", "2026-07-28")

    assert resolved is None


def test_cached_price_lookup_ignores_non_refresh_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_quote(
            conn, _quote_row("LIN", "2026-07-28", 50.0, source="eodhd_split_lineage")
        )
        _insert_quote(conn, _quote_row("ABG", "2026-07-27", 193.0))
        _insert_quote(
            conn, _quote_row("ABG", "2026-07-28", 999.0, source="eodhd_split_lineage")
        )

    lookup = build_cached_price_lookup(db_path)

    assert lookup("LIN", "2026-07-28") is None
    resolved = lookup("ABG", "2026-07-28")
    assert resolved == {
        **_quote_row("ABG", "2026-07-27", 193.0),
        "confidence": "MEDIUM",
    }


def test_cached_price_lookup_rejects_empty_provider_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="providers must be non-empty"):
        build_cached_price_lookup(db_path, providers=())
