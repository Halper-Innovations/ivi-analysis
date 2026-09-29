"""Holdings + exit signals: registry, held-book pass, tripwires,
monitoring registration, digest surfacing."""

from __future__ import annotations

import sqlite3

import pytest

from app.config import get_config
from app.market.price_provider import PriceSnapshot


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


def _snap(ticker: str, price: float, as_of: str = "2026-07-15") -> PriceSnapshot:
    return PriceSnapshot(
        ticker=ticker,
        as_of_date=as_of,
        price=price,
        currency="USD",
        source="test",
        retrieved_at=f"{as_of}T12:00:00+00:00",
        confidence="HIGH",
    )


def _add_watchlist_row(db_path, ticker, status="ACTIVE", anchor=None, reason=None):
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update

    return add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            buy_price_target=10.0,
            valuation_anchor_method="EPV" if anchor else None,
            valuation_anchor_value=anchor,
            status_reason=reason,
            source_run_id="run1",
            added_at="2026-07-01T00:00:00+00:00",
        ),
        db_path=db_path,
    )


def test_add_holding_links_thesis(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, list_holdings

    wl_id = _add_watchlist_row(db_path, "HLD")
    row = add_holding(
        ticker="HLD",
        entry_date="2026-07-01",
        entry_price=10.0,
        size="2% NAV",
        db_path=db_path,
    )
    assert row["thesis_watchlist_id"] == wl_id
    assert row["status"] == "OPEN"
    assert row["peak_price"] == 10.0
    assert list_holdings(db_path=db_path)[0]["ticker"] == "HLD"


def test_held_cik_map_extends_event_protection_scope(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.events.poller import watchlist_cik_map
    from app.holdings import add_holding

    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda refresh_if_missing=True: {"HLD": "7654321"},
    )
    add_holding(ticker="HLD", entry_date="2026-07-01", entry_price=10.0, db_path=db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    scope = watchlist_cik_map(conn)
    conn.close()
    assert scope.get("0007654321") == "HLD"


def test_drawdown_tripwire_and_peak_tracking(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, run_held_book_pass

    add_holding(
        ticker="DD",
        entry_date="2026-07-01",
        entry_price=100.0,
        drawdown_alert_pct=25.0,
        db_path=db_path,
    )
    report = run_held_book_pass(
        as_of_date="2026-07-15",
        db_path=db_path,
        price_lookup=lambda ticker: _snap(ticker, 70.0),
    )
    types = [signal["signal_type"] for signal in report["new_signals"]]
    assert "DRAWDOWN_TRIPWIRE" in types
    assert any(s in report["critical"] for s in report["new_signals"])
    # 30% below entry AND 30% below the (entry) peak — both bases fire but
    # they land as one row (unique per holding/type/day).
    tripwires = [s for s in report["new_signals"] if s["signal_type"] == "DRAWDOWN_TRIPWIRE"]
    assert len(tripwires) == 1


def test_no_price_is_a_monitoring_failure(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, run_held_book_pass

    add_holding(ticker="NOPX", entry_date="2026-07-01", entry_price=10.0, db_path=db_path)
    report = run_held_book_pass(
        as_of_date="2026-07-15", db_path=db_path, price_lookup=lambda ticker: None
    )
    types = [signal["signal_type"] for signal in report["new_signals"]]
    assert "MONITORING_FAILURE" in types


def test_contradicted_held_name_signals_review_required(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, run_held_book_pass

    _add_watchlist_row(db_path, "CON", status="CONTRADICTED", reason="thesis broke")
    add_holding(ticker="CON", entry_date="2026-07-01", entry_price=10.0, db_path=db_path)
    report = run_held_book_pass(
        as_of_date="2026-07-15",
        db_path=db_path,
        price_lookup=lambda ticker: _snap(ticker, 10.0),
    )
    contradicted = [s for s in report["new_signals"] if s["signal_type"] == "THESIS_CONTRADICTED"]
    assert len(contradicted) == 1
    assert contradicted[0]["evidence"]["action"] == "REVIEW_REQUIRED"


def test_universe_exit_signal_from_registry_removal(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, run_held_book_pass

    add_holding(ticker="GONE", entry_date="2026-07-01", entry_price=10.0, db_path=db_path)
    conn = sqlite3.connect(str(db_path))
    now = "2026-07-10T00:00:00+00:00"
    conn.execute(
        "INSERT INTO sec_registrants(cik, primary_ticker, all_tickers, exchange_scope, "
        "operating_status, in_scope, first_seen_at, last_seen_at, removed_at) "
        "VALUES ('9', 'GONE', '[]', 'IN_SCOPE', 'OPERATING', 1, ?, ?, ?)",
        (now, now, now),
    )
    conn.commit()
    conn.close()

    report = run_held_book_pass(
        as_of_date="2026-07-15",
        db_path=db_path,
        price_lookup=lambda ticker: _snap(ticker, 10.0),
    )
    types = [signal["signal_type"] for signal in report["new_signals"]]
    assert "UNIVERSE_EXIT" in types


def test_fair_value_exit_is_config_gated_off(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, run_held_book_pass

    _add_watchlist_row(db_path, "FV", status="ACTIVE", anchor=10.0)
    add_holding(ticker="FV", entry_date="2026-07-01", entry_price=8.0, db_path=db_path)

    # Price above the anchor, flag OFF -> no signal (default: off).
    report = run_held_book_pass(
        as_of_date="2026-07-15",
        db_path=db_path,
        price_lookup=lambda ticker: _snap(ticker, 12.0),
    )
    assert "PRICE_AT_FAIR_VALUE" not in [s["signal_type"] for s in report["new_signals"]]

    monkeypatch.setenv("VOE_EXIT_FAIR_VALUE_ENABLED", "true")
    report = run_held_book_pass(
        as_of_date="2026-07-16",
        db_path=db_path,
        price_lookup=lambda ticker: _snap(ticker, 12.0, as_of="2026-07-16"),
    )
    assert "PRICE_AT_FAIR_VALUE" in [s["signal_type"] for s in report["new_signals"]]


def test_pass_is_idempotent_per_day(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, run_held_book_pass

    add_holding(ticker="DD", entry_date="2026-07-01", entry_price=100.0, db_path=db_path)
    first = run_held_book_pass(
        as_of_date="2026-07-15",
        db_path=db_path,
        price_lookup=lambda ticker: _snap(ticker, 60.0),
    )
    second = run_held_book_pass(
        as_of_date="2026-07-15",
        db_path=db_path,
        price_lookup=lambda ticker: _snap(ticker, 60.0),
    )
    assert first["new_signals"]
    assert second["new_signals"] == []


def test_close_holding_populates_max_drawdown(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, close_holding

    row = add_holding(ticker="MDD", entry_date="2026-07-01", entry_price=100.0, db_path=db_path)
    conn = sqlite3.connect(str(db_path))
    for day, price in (("2026-07-02", 110.0), ("2026-07-08", 66.0), ("2026-07-10", 90.0)):
        conn.execute(
            "INSERT INTO price_quotes(ticker, provider, as_of_date, price, currency, status, "
            "fetched_at, expires_at, raw_json, quote_hash) "
            "VALUES ('MDD', 'test', ?, ?, 'USD', 'OK', ?, ?, '{}', ?)",
            (day, price, f"{day}T21:00:00+00:00", f"{day}T21:00:00+00:00", f"h-{day}"),
        )
    conn.commit()
    conn.close()

    closed = close_holding(
        holding_id=int(row["id"]),
        close_price=90.0,
        closed_at="2026-07-15T00:00:00+00:00",
        db_path=db_path,
    )
    assert closed["status"] == "CLOSED"
    # Peak 110 -> trough 66 = -40%.
    assert closed["max_drawdown_pct"] == pytest.approx(-40.0)


def test_digest_suppresses_unbound_held_book_exits_but_preserves_ledger(monkeypatch, tmp_path):
    db_path = _env(monkeypatch, tmp_path)
    from app.holdings import add_holding, run_held_book_pass
    from app.watchlist.digest import render_digest

    add_holding(ticker="DD", entry_date="2026-07-01", entry_price=100.0, db_path=db_path)
    run_held_book_pass(
        as_of_date="2026-07-15",
        db_path=db_path,
        price_lookup=lambda ticker: _snap(ticker, 60.0),
    )
    monkeypatch.setattr(
        "app.holdings.open_exit_signals",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("digest must not open a second held-exit connection")
        ),
    )
    digest = render_digest(days_back=1, db_path=db_path)
    assert "## Exits (Held Book)" not in digest
    assert "DRAWDOWN_TRIPWIRE" not in digest

    conn = sqlite3.connect(str(db_path))
    try:
        assert (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM exit_signals
                WHERE ticker = 'DD' AND signal_type = 'DRAWDOWN_TRIPWIRE'
                """
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()
