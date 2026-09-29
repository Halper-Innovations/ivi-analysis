from __future__ import annotations

from app.db import get_db, init_db
from app.market.price_history_backfill import backfill_daily_history
from app.market.price_provider import PriceSnapshot


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


class FakeProvider:
    """Returns a PriceSnapshot(price=10.0) for every (ticker, date) it knows.

    Tickers in ``missing`` always resolve to None (symbol-not-found / disabled).
    """

    provider_name = "fake"

    def __init__(self, *, known: set[str], missing: set[str] | None = None) -> None:
        self.known = known
        self.missing = missing or set()
        self.calls: list[tuple[str, str]] = []

    def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot | None:
        self.calls.append((ticker, as_of_date))
        if ticker in self.missing or ticker not in self.known:
            return None
        return PriceSnapshot(
            ticker=ticker,
            as_of_date=as_of_date,
            price=10.0,
            currency="USD",
            source="fake",
        )


def _count_rows(ticker: str) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM price_quotes WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    return int(row["n"])


def test_backfill_writes_one_row_per_anchor_date(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    provider = FakeProvider(known={"AAA"})

    summary = backfill_daily_history(
        tickers=["AAA"],
        anchor_dates=["2025-01-02", "2026-01-02"],
        benchmark_symbols=(),
        provider=provider,
    )

    assert summary.rows_written == 2
    assert summary.fetched == 2
    assert summary.failed_tickers == []
    assert _count_rows("AAA") == 2


def test_backfill_persists_benchmark_symbol(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    provider = FakeProvider(known={"AAA", "SPY"})

    backfill_daily_history(
        tickers=["AAA"],
        anchor_dates=["2025-01-02"],
        benchmark_symbols=("SPY",),
        provider=provider,
    )

    with get_db() as conn:
        row = conn.execute(
            "SELECT ticker FROM price_quotes WHERE ticker = 'SPY'"
        ).fetchone()
    assert row is not None
    assert row["ticker"] == "SPY"


def test_backfill_tracks_failed_ticker_and_writes_no_rows(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    provider = FakeProvider(known=set(), missing={"BAD"})

    summary = backfill_daily_history(
        tickers=["BAD"],
        anchor_dates=["2025-01-02"],
        benchmark_symbols=(),
        provider=provider,
    )

    assert summary.failed_tickers == ["BAD"]
    assert _count_rows("BAD") == 0


def test_backfill_is_idempotent(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    provider = FakeProvider(known={"AAA"})

    backfill_daily_history(
        tickers=["AAA"],
        anchor_dates=["2025-01-02", "2026-01-02"],
        benchmark_symbols=(),
        provider=provider,
    )
    first_count = _count_rows("AAA")

    backfill_daily_history(
        tickers=["AAA"],
        anchor_dates=["2025-01-02", "2026-01-02"],
        benchmark_symbols=(),
        provider=provider,
    )
    second_count = _count_rows("AAA")

    assert first_count == 2
    assert second_count == 2


def test_backfill_rows_written_dedupes_same_resolved_trading_day(monkeypatch, tmp_path):
    # Two anchor dates (e.g. a Saturday and Sunday) can resolve to the SAME
    # trading day; the ON CONFLICT(ticker, provider, as_of_date) upsert collapses
    # them to one row, so rows_written must count DISTINCT persisted keys (fetched
    # still counts both provider calls).
    _init_temp_db(monkeypatch, tmp_path)

    class _SameDayProvider:
        provider_name = "sameday"

        def get_price_asof(self, ticker: str, as_of_date: str) -> PriceSnapshot:
            return PriceSnapshot(
                ticker=ticker,
                as_of_date="2026-05-15",  # both anchors resolve to this Friday close
                price=100.0,
                currency="USD",
                source="sameday",
            )

    summary = backfill_daily_history(
        tickers=["AAA"],
        anchor_dates=["2026-05-16", "2026-05-17"],
        benchmark_symbols=(),
        provider=_SameDayProvider(),
    )

    assert summary.fetched == 2
    assert summary.rows_written == 1
    assert _count_rows("AAA") == 1
