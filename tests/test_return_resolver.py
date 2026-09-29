# tests/test_return_resolver.py
"""return_resolver computes realized + benchmark + excess returns and
closes matured ticker_outcomes rows.

Pure helpers are tested with literal arithmetic; resolution is tested with an
env-redirected tmp db (VOE_DB_PATH) and an injected FakeProvider only — no
network, no LLM, no live price fetch.
"""
from __future__ import annotations

from pathlib import Path

from app.calibration.return_resolver import (
    _excess_pct,
    _realized_pct,
    resolve_open_outcomes,
)
from app.db import get_db, init_db
from app.market.price_provider import PriceSnapshot
from app.outcomes.store import add_outcome


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    return db_path


class FakeProvider:
    """Returns a PriceSnapshot for known (ticker, date) keys, else None.

    ``prices`` maps (ticker, as_of_date) -> price. Keys absent from the map
    resolve to None (symbol-not-found / no price).
    """

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


def _open_outcome(
    *,
    ticker: str,
    entry_price: float,
    entry_date: str,
    horizon_days: int,
    benchmark_symbol: str = "SPY",
    run_id: str = "r1",
    buy_price_target: float | None = None,
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
        buy_price_target=buy_price_target,
    )


def _row(ticker: str) -> dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM ticker_outcomes WHERE ticker = ? LIMIT 1",
            (ticker.upper(),),
        ).fetchone()
    return dict(row)


def test_realized_pct_gain():
    assert _realized_pct(100.0, 115.0) == 15.0


def test_realized_pct_loss():
    assert _realized_pct(50.0, 40.0) == -20.0


def test_excess_pct():
    assert _excess_pct(15.0, 9.0) == 6.0


def test_resolve_closes_matured_row(monkeypatch, tmp_path):
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

    summary = resolve_open_outcomes("2026-05-01", provider=provider)

    assert summary.closed == 1
    row = _row("AAA")
    assert row["realized_return_pct"] == 20.0
    assert row["benchmark_return_pct"] == 10.0
    assert row["excess_return_pct"] == 10.0
    assert row["outcome_status"] == "CLOSED"


def test_resolve_skips_not_matured_row(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _open_outcome(
        ticker="BBB",
        entry_price=100.0,
        entry_date="2026-04-15",
        horizon_days=90,
    )
    provider = FakeProvider({})

    summary = resolve_open_outcomes("2026-05-01", provider=provider)

    assert summary.skipped_not_matured == 1
    assert summary.closed == 0
    assert _row("BBB")["outcome_status"] == "OPEN"


def test_resolve_skips_when_no_exit_price(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _open_outcome(
        ticker="CCC",
        entry_price=100.0,
        entry_date="2026-01-01",
        horizon_days=90,
    )
    provider = FakeProvider(
        {
            ("SPY", "2026-01-01"): 400.0,
            ("SPY", "2026-04-01"): 440.0,
        }
    )

    summary = resolve_open_outcomes("2026-05-01", provider=provider)

    assert summary.skipped_no_price == 1
    assert summary.closed == 0
    row = _row("CCC")
    assert row["realized_return_pct"] is None
    assert row["outcome_status"] == "OPEN"


def test_resolve_reached_buy_target_set_when_exit_at_or_below(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _open_outcome(
        ticker="DDD",
        entry_price=100.0,
        entry_date="2026-01-01",
        horizon_days=90,
        buy_price_target=80.0,
    )
    provider = FakeProvider(
        {
            ("DDD", "2026-04-01"): 75.0,  # below the 80 target -> correction reached
            ("SPY", "2026-01-01"): 400.0,
            ("SPY", "2026-04-01"): 380.0,
        }
    )

    summary = resolve_open_outcomes("2026-05-01", provider=provider)

    assert summary.closed == 1
    row = _row("DDD")
    assert row["reached_buy_target"] == 1
    assert row["realized_return_pct"] == -25.0


def test_resolve_reached_buy_target_zero_when_exit_above_target(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _open_outcome(
        ticker="EEE",
        entry_price=100.0,
        entry_date="2026-01-01",
        horizon_days=90,
        buy_price_target=80.0,
    )
    provider = FakeProvider({("EEE", "2026-04-01"): 90.0})

    summary = resolve_open_outcomes("2026-05-01", provider=provider)

    assert summary.closed == 1
    assert _row("EEE")["reached_buy_target"] == 0


def test_resolve_reached_buy_target_null_when_no_target(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _open_outcome(
        ticker="FFF",
        entry_price=100.0,
        entry_date="2026-01-01",
        horizon_days=90,
    )
    provider = FakeProvider({("FFF", "2026-04-01"): 90.0})

    summary = resolve_open_outcomes("2026-05-01", provider=provider)

    assert summary.closed == 1
    assert _row("FFF")["reached_buy_target"] is None
