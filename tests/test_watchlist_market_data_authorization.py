from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from app.config import get_config
from app.db import get_db, init_db
from app.market.price_provider import PriceSnapshot
from app.watchlist.market_data import refresh_current_watchlist_prices
from app.watchlist.schema import ensure_watchlist_schema
from app.watchlist.store import list_active, watchlist_queue


class StubPriceProvider:
    provider_name = "stub"

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot:
        return PriceSnapshot(
            ticker=ticker,
            as_of_date=as_of_date,
            price=193.0,
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


def _insert_watchlist_row(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        """
        INSERT INTO watchlist(ticker, status, source_run_id, added_at)
        VALUES ('ABG', 'ACTIVE', 'audited_invalid', '2026-07-01T00:00:00+00:00')
        """
    )
    return int(cursor.lastrowid)


def test_audited_invalid_row_receives_fact_but_stays_out_of_decisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = _init(monkeypatch, tmp_path)
    with get_db() as conn:
        current_id = _insert_watchlist_row(conn)
    monkeypatch.setattr(
        "app.watchlist.store.financial_integrity_manifest_is_usable",
        lambda *_args, **_kwargs: False,
    )

    summary = refresh_current_watchlist_prices(
        as_of_date="2026-07-28",
        db_path=db_path,
        provider=StubPriceProvider(),
    )

    assert summary.written == 1
    assert list_active(db_path=db_path) == []
    assert watchlist_queue(db_path=db_path) == []
    with get_db() as conn:
        row = conn.execute(
            "SELECT status FROM watchlist WHERE id = ?", (current_id,)
        ).fetchone()
        snapshot_count = conn.execute(
            "SELECT COUNT(*) FROM watchlist_price_snapshots WHERE watchlist_id = ?",
            (current_id,),
        ).fetchone()[0]
        history_count = conn.execute(
            "SELECT COUNT(*) FROM watchlist_history WHERE watchlist_id = ?",
            (current_id,),
        ).fetchone()[0]
    assert row["status"] == "ACTIVE"
    assert snapshot_count == 1
    assert history_count == 0


def test_decision_and_presentation_consumers_do_not_import_ungated_selector() -> None:
    repo = Path(__file__).resolve().parents[1]
    guarded = [
        repo / "app/watchlist/triggers.py",
        repo / "app/watchlist/digest.py",
        repo / "app/cli_investor.py",
        repo / "app/web/readmodel/today.py",
    ]
    for path in guarded:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "list_current_for_market_data" not in imported, path
