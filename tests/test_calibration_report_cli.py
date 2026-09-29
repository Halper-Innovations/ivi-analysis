# tests/test_calibration_report_cli.py
"""CLI wiring for the grade/status calibration loop.

Covers the ``calibration-report`` and ``calibration-resolve`` Typer commands.
Tests run against an env-redirected tmp db (VOE_DB_PATH) and, for resolution,
an injected FakeProvider patched over ``get_default_provider`` — no network, no
LLM, no live price fetch. Expected values are hand-computed literals.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from app.cli import app
from app.db import get_db, init_db, utc_now_iso
from app.market.price_provider import PriceSnapshot
from app.outcomes.store import add_outcome
from app.watchlist.contract import WatchlistEntry
from app.watchlist.store import add_or_update

runner = CliRunner()


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


class FakeProvider:
    """Returns a PriceSnapshot for known (ticker, date) keys, else None."""

    provider_name = "fake"

    def __init__(self, prices: dict[tuple[str, str], float]) -> None:
        self.prices = prices

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        price = self.prices.get((ticker, as_of_date))
        if price is None:
            return None
        return PriceSnapshot(
            ticker=ticker,
            as_of_date=as_of_date,
            price=price,
            currency="USD",
            source="fake",
        )


def _insert_closed(
    *,
    ticker: str,
    run_id: str,
    grade: str,
    status: str,
    realized: float,
) -> None:
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO ticker_outcomes(
                ticker, as_of_date, run_id, decision, conviction, horizon_days,
                thesis_tags_json, notes, outcome_status, close_date,
                realized_return_pct, benchmark_return_pct, excess_return_pct,
                reached_buy_target, entry_price, entry_date, grade, status,
                benchmark_symbol, created_at, updated_at
            ) VALUES(?, ?, ?, 'BUY', 4, 365, '[]', '', 'CLOSED', '2026-05-17',
                ?, NULL, NULL, NULL, 100.0, '2025-05-17', ?, ?, 'SPY', ?, ?)
            """,
            (
                ticker.upper(),
                "2025-05-17",
                run_id,
                realized,
                grade,
                status,
                now,
                now,
            ),
        )


def _open_outcome(
    *,
    ticker: str,
    entry_price: float,
    entry_date: str,
    horizon_days: int,
    benchmark_symbol: str = "SPY",
    run_id: str = "r1",
) -> None:
    add_outcome(
        ticker=ticker,
        as_of_date=entry_date,
        run_id=run_id,
        decision="BUY",
        conviction=4,
        horizon_days=horizon_days,
        entry_price=entry_price,
        entry_price_source="watchlist_population",
        entry_date=entry_date,
        grade="ACTIONABLE",
        status="DEPLOY_READY",
        benchmark_symbol=benchmark_symbol,
    )


def test_calibration_report_cmd_prints_segmented_json(monkeypatch, tmp_path) -> None:
    _init_temp_db(monkeypatch, tmp_path)
    _insert_closed(
        ticker="AAA", run_id="r1", grade="ACTIONABLE", status="DEPLOY_READY", realized=20.0
    )
    _insert_closed(
        ticker="BBB", run_id="r1", grade="WATCHLIST_ONLY", status="ACTIVE", realized=-5.0
    )

    result = runner.invoke(app, ["calibration-report", "--as-of", "2026-05-30"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "by_grade" in payload
    assert "by_status" in payload
    assert "overall" in payload
    assert payload["overall"]["n"] == 2


def test_calibration_resolve_cmd_closes_matured_row(monkeypatch, tmp_path) -> None:
    _init_temp_db(monkeypatch, tmp_path)
    _open_outcome(
        ticker="AAA",
        entry_price=100.0,
        entry_date="2026-01-01",
        horizon_days=90,
    )
    provider = FakeProvider(
        {
            ("AAA", "2026-04-01"): 120.0,
            ("SPY", "2026-01-01"): 400.0,
            ("SPY", "2026-04-01"): 440.0,
        }
    )
    monkeypatch.setattr(
        "app.market.price_provider.get_default_provider",
        lambda *a, **k: provider,
    )

    result = runner.invoke(app, ["calibration-resolve", "--as-of", "2026-05-01"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["closed"] == 1


def test_calibration_backfill_excludes_unauthorized_watchlist_and_outcome_rows_before_fetch(
    monkeypatch,
    tmp_path,
) -> None:
    _init_temp_db(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="FORGED",
            status="ACTIVE",
            source_run_id="unauthorized_watchlist_run",
            added_at="2026-01-01T00:00:00Z",
            conviction_grade="WATCHLIST_ONLY",
        )
    )
    _open_outcome(
        ticker="FORGED",
        entry_price=100.0,
        entry_date="2099-01-02",
        horizon_days=7,
        run_id="unauthorized_outcome_run",
    )
    observed: dict[str, object] = {}

    def fake_backfill_daily_history(*, tickers, anchor_dates, benchmark_symbols):
        observed.update(
            {
                "tickers": list(tickers),
                "anchor_dates": list(anchor_dates),
                "benchmark_symbols": tuple(benchmark_symbols),
            }
        )
        return SimpleNamespace(fetched=0, rows_written=0, failed_tickers=[])

    monkeypatch.setattr(
        "app.market.price_history_backfill.backfill_daily_history",
        fake_backfill_daily_history,
    )

    result = runner.invoke(app, ["calibration-backfill-prices"])

    assert result.exit_code == 0, result.output
    assert observed == {
        "tickers": [],
        "anchor_dates": [],
        "benchmark_symbols": ("SPY",),
    }
    payload = json.loads(result.output)
    assert payload["ticker_count"] == 0
    assert payload["anchor_date_count"] == 0
