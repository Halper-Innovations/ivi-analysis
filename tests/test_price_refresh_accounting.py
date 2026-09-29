from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.config import get_config
from app.db import get_db, init_db
from app.market.price_provider import PriceSnapshot
from app.watchlist.market_data import (
    PriceRefreshSummary,
    refresh_current_watchlist_prices,
)
from app.watchlist.schema import ensure_watchlist_schema

runner = CliRunner()


class StubPriceProvider:
    provider_name = "stub"

    def get_price_asof(
        self, ticker: str, as_of_date: str
    ) -> PriceSnapshot | None:
        if ticker == "NOPE":
            return None
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
) -> None:
    conn.execute(
        """
        INSERT INTO watchlist(ticker, status, source_run_id, added_at)
        VALUES (?, 'ACTIVE', ?, '2026-07-01T00:00:00+00:00')
        """,
        (ticker, run_id),
    )


def test_refresh_counts_unavailable_without_treating_it_as_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_watchlist_row(conn, ticker="ABG", run_id="run_abg")
        _insert_watchlist_row(conn, ticker="NOPE", run_id="run_nope")
    summary = refresh_current_watchlist_prices(
        as_of_date="2026-07-28",
        db_path=db_path,
        provider=StubPriceProvider(),
    )
    assert summary.candidates == 2
    assert summary.attempts == 2
    assert summary.written == 1
    assert summary.unavailable == ["NOPE"]
    assert summary.exit_code == 0


@pytest.mark.parametrize(
    ("candidates", "attempts", "written", "unavailable", "expected"),
    [
        (3, 0, 0, 0, 1),
        (3, 3, 0, 0, 1),
        (0, 0, 0, 0, 2),
        (3, 3, 2, 0, 0),
        (3, 3, 2, 1, 0),
        (2, 2, 1, 1, 0),
        (3, 3, 1, 2, 1),
        (4, 4, 1, 3, 1),
    ],
)
def test_refresh_summary_fails_loud_on_successful_noop(
    candidates: int,
    attempts: int,
    written: int,
    unavailable: int,
    expected: int,
) -> None:
    summary = PriceRefreshSummary(
        as_of_date="2026-07-28",
        candidates=candidates,
        attempts=attempts,
        written=written,
        unavailable=[f"T{i}" for i in range(unavailable)],
    )
    assert summary.exit_code == expected


def test_price_refresh_cli_rejects_backdated_as_of(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _init(monkeypatch, tmp_path)
    result = runner.invoke(
        app, ["watchlist", "price-refresh", "--as-of", "2026-07-28"]
    )
    assert result.exit_code == 2
    assert "--as-of must be today" in result.output
