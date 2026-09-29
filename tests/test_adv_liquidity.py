"""Liquidity layer: volume retention, dollar-ADV, capacity bands,
CAPACITY_LIMITED surfaces, MIN_ADV advisory flag."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

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


def _seed_quotes(
    db_path,
    ticker: str,
    *,
    days: int,
    price: float = 10.0,
    volume: float | None = 50_000.0,
    end: str = "2026-07-15",
) -> None:
    conn = sqlite3.connect(str(db_path))
    end_date = date.fromisoformat(end)
    written = 0
    offset = 0
    while written < days:
        day = end_date - timedelta(days=offset)
        offset += 1
        if day.weekday() >= 5:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO price_quotes(
                ticker, provider, as_of_date, price, currency, status,
                fetched_at, expires_at, raw_json, quote_hash, volume
            ) VALUES (?, 'test', ?, ?, 'USD', 'OK', ?, ?, '{}', ?, ?)
            """,
            (
                ticker.upper(),
                day.isoformat(),
                price,
                f"{day.isoformat()}T21:00:00+00:00",
                f"{day.isoformat()}T21:00:00+00:00",
                f"h-{ticker}-{day.isoformat()}",
                volume,
            ),
        )
        written += 1
    conn.commit()
    conn.close()


def test_stooq_csv_parse_retains_volume():
    from app.market.price_provider import StooqProvider

    provider = StooqProvider.__new__(StooqProvider)
    provider._volumes_by_day = {}
    csv_text = "Date,Open,High,Low,Close,Volume\n2026-07-14,9,11,8,10,12345\n2026-07-15,10,12,9,11,23456\n"
    rows, raw = provider._parse_history(csv_text)
    assert raw == 2
    assert rows[date(2026, 7, 15)] == 11.0
    assert provider._volumes_by_day[date(2026, 7, 15)] == 23456.0


def test_capacity_class_bands():
    from app.market.adv import ADV_UNKNOWN, capacity_class_for

    assert capacity_class_for(None) == ADV_UNKNOWN
    assert capacity_class_for(50_000) == "MICRO_LIQUIDITY"
    assert capacity_class_for(500_000) == "THIN"
    assert capacity_class_for(5_000_000) == "MODERATE"
    assert capacity_class_for(50_000_000) == "DEEP"


def test_compute_adv_uses_trading_day_window(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    _seed_quotes(db_path, "AAA", days=70, price=10.0, volume=50_000.0)
    from app.market.adv import compute_adv

    result = compute_adv("AAA", as_of_date="2026-07-15", db_path=db_path)
    assert result.adv_dollar_20d == pytest.approx(500_000.0)
    assert result.adv_dollar_60d == pytest.approx(500_000.0)
    assert result.capacity_class == "THIN"


def test_compute_adv_returns_unknown_below_coverage(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    # Only 5 volume days: below the 50% coverage floor for a 20d window.
    _seed_quotes(db_path, "BBB", days=5, price=10.0, volume=50_000.0)
    from app.market.adv import ADV_UNKNOWN, compute_adv

    result = compute_adv("BBB", as_of_date="2026-07-15", db_path=db_path)
    assert result.adv_dollar_20d is None
    assert result.capacity_class == ADV_UNKNOWN


def test_compute_adv_ignores_volumeless_rows(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    _seed_quotes(db_path, "CCC", days=30, price=10.0, volume=None)
    from app.market.adv import ADV_UNKNOWN, compute_adv

    result = compute_adv("CCC", as_of_date="2026-07-15", db_path=db_path)
    assert result.adv_dollar_20d is None
    assert result.capacity_class == ADV_UNKNOWN


def test_persist_watchlist_adv_updates_live_rows(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.market.adv import persist_watchlist_adv
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update

    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            buy_price_target=9.0,
            source_run_id="run1",
            added_at="2026-07-01T00:00:00+00:00",
        ),
        db_path=db_path,
    )
    _seed_quotes(db_path, "AAA", days=70, price=10.0, volume=200_000.0)

    result = persist_watchlist_adv("AAA", as_of_date="2026-07-15", db_path=db_path)

    assert result.adv_dollar_20d == pytest.approx(2_000_000.0)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT adv_dollar_20d, adv_dollar_60d, capacity_class, adv_asof FROM watchlist WHERE ticker='AAA'"
    ).fetchone()
    conn.close()
    assert row["adv_dollar_20d"] == pytest.approx(2_000_000.0)
    assert row["capacity_class"] == "MODERATE"
    assert row["adv_asof"] == "2026-07-15"


def test_digest_renders_capacity_limited_banner(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.market.adv import persist_watchlist_adv
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.digest import render_digest
    from app.watchlist.store import add_or_update

    add_or_update(
        WatchlistEntry(
            ticker="TINY",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            confidence="HIGH",
            conviction_source="sector_final_decision",
            buy_price_target=12.0,
            current_price_at_addition=10.0,
            source_run_id="run1",
            added_at="2026-07-01T00:00:00+00:00",
        ),
        db_path=db_path,
    )
    # $2 * 20k shares = $40k dollar-ADV — far below the $250k floor.
    _seed_quotes(db_path, "TINY", days=30, price=2.0, volume=20_000.0)
    persist_watchlist_adv("TINY", as_of_date="2026-07-15", db_path=db_path)

    digest = render_digest(days_back=1, db_path=db_path)
    assert "\n  - (**CAPACITY_LIMITED**" in digest
    assert "below the" in digest


def test_gate_emits_min_adv_advisory_flag_not_quarantine(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    _seed_quotes(db_path, "TINY", days=30, price=2.0, volume=20_000.0)
    from app.autonomous.structural_gate import evaluate_structural_gate

    result = evaluate_structural_gate(
        "TINY", as_of_date="2026-07-15", price=2.0, db_path=db_path
    )
    assert any(code.startswith("MIN_ADV:") for code in result.advisory_codes)
    # Advisory only: the flag never quarantines on its own.
    assert "MIN_ADV" not in ";".join(result.triggered_codes)


def test_gate_silent_when_no_volume_history(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.autonomous.structural_gate import evaluate_structural_gate

    result = evaluate_structural_gate(
        "NOVOL", as_of_date="2026-07-15", price=50.0, db_path=db_path
    )
    assert result.advisory_codes == []
