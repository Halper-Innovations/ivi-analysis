"""Dollar-ADV liquidity layer.

The platform's first tradeability signal: 20d/60d average daily dollar
volume computed from the daily quote rows in ``price_quotes`` (volume
retained by the Stooq/EODHD parsers, seeded by the 90-day watchlist
backfill, refreshed by the daily trigger pass). Persisted per watchlist row
as ``adv_dollar_20d`` / ``adv_dollar_60d`` with a ``capacity_class`` band;
names without enough volume history carry the explicit ADV_UNKNOWN state
rather than a fabricated number.

Deliberately NOT here: position sizing or days-to-build — position size is
the portfolio layer's variable; it computes those parametrically from the
persisted ADV (portfolio-interface contract §4.2).

The volatility-weight term (w_vol=0.0) stays untouched; this
layer only unblocks that decision.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from app.config import get_config
from app.logging import get_logger

logger = get_logger(__name__)

ADV_UNKNOWN = "ADV_UNKNOWN"

# Capacity bands on 20d dollar-ADV. Bands, not verdicts: the gate flags,
# never rejects, and the portfolio layer sizes against the raw numbers.
CAPACITY_BANDS = (
    ("MICRO_LIQUIDITY", 100_000.0),
    ("THIN", 1_000_000.0),
    ("MODERATE", 10_000_000.0),
    ("DEEP", None),
)

# At least this fraction of the window must have volume rows before an ADV
# is asserted — 8 prints do not make a 20-day average.
MIN_WINDOW_COVERAGE = 0.5

DEFAULT_BACKFILL_DAYS = 90


def adv_dollar_floor() -> float:
    """CAPACITY_LIMITED banner floor (20d dollar-ADV), env-overridable."""
    raw = os.getenv("VOE_ADV_DOLLAR_FLOOR", "")
    try:
        return float(raw) if raw.strip() else 250_000.0
    except ValueError:
        return 250_000.0


@dataclass(frozen=True)
class AdvResult:
    ticker: str
    as_of_date: str
    adv_dollar_20d: float | None
    adv_dollar_60d: float | None
    capacity_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "adv_dollar_20d": self.adv_dollar_20d,
            "adv_dollar_60d": self.adv_dollar_60d,
            "capacity_class": self.capacity_class,
        }


def capacity_class_for(adv_dollar_20d: float | None) -> str:
    if adv_dollar_20d is None or adv_dollar_20d < 0:
        return ADV_UNKNOWN
    for label, upper in CAPACITY_BANDS:
        if upper is None or adv_dollar_20d < upper:
            return label
    return ADV_UNKNOWN  # pragma: no cover - the DEEP band is unbounded


def _dollar_adv_window(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    as_of_date: str,
    window: int,
) -> float | None:
    """Mean of price*volume over the newest ``window`` distinct trading days
    at/before as_of_date that carry volume; None below minimum coverage."""
    try:
        rows = conn.execute(
            """
            SELECT price * volume AS dollar_volume
            FROM (
                SELECT as_of_date, price, volume,
                       ROW_NUMBER() OVER (
                           PARTITION BY as_of_date
                           ORDER BY fetched_at DESC, id DESC
                       ) AS rn
                FROM price_quotes
                WHERE ticker = ?
                  AND as_of_date <= ?
                  AND status = 'OK'
                  AND price IS NOT NULL AND price > 0
                  AND volume IS NOT NULL AND volume >= 0
            )
            WHERE rn = 1
            ORDER BY as_of_date DESC
            LIMIT ?
            """,
            (ticker.upper(), as_of_date, int(window)),
        ).fetchall()
    except sqlite3.Error:
        return None
    values = [float(row[0]) for row in rows if isinstance(row[0], (int, float))]
    if len(values) < max(1, int(window * MIN_WINDOW_COVERAGE)):
        return None
    return sum(values) / len(values)


def compute_adv(
    ticker: str,
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> AdvResult:
    from app.db import connect

    asof = str(as_of_date or date.today().isoformat())
    owns_conn = conn is None
    if conn is None:
        path = Path(db_path) if db_path is not None else Path(get_config().db_path)
        if not path.exists():
            return AdvResult(ticker.upper(), asof, None, None, ADV_UNKNOWN)
        conn = connect(path)
    try:
        adv20 = _dollar_adv_window(conn, ticker, as_of_date=asof, window=20)
        adv60 = _dollar_adv_window(conn, ticker, as_of_date=asof, window=60)
    finally:
        if owns_conn:
            conn.close()
    return AdvResult(
        ticker=ticker.upper(),
        as_of_date=asof,
        adv_dollar_20d=adv20,
        adv_dollar_60d=adv60,
        capacity_class=capacity_class_for(adv20),
    )


def persist_watchlist_adv(
    ticker: str,
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
) -> AdvResult:
    """Compute and persist ADV fields on the ticker's live watchlist rows.

    Market data, not judgment — column updates only, no watchlist_history
    rows (mirroring price snapshots).
    """
    from app.db import connect
    from app.watchlist.schema import ensure_watchlist_schema, resolve_db_path

    ensure_watchlist_schema(db_path)
    result = compute_adv(ticker, as_of_date=as_of_date, db_path=db_path)
    conn = connect(resolve_db_path(db_path))
    try:
        conn.execute(
            """
            UPDATE watchlist
            SET adv_dollar_20d = ?, adv_dollar_60d = ?, adv_asof = ?, capacity_class = ?
            WHERE ticker = ? AND status != 'REMOVED'
            """,
            (
                result.adv_dollar_20d,
                result.adv_dollar_60d,
                result.as_of_date,
                result.capacity_class,
                ticker.upper(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return result


def _bulk_history_provider() -> Any:
    """Provider chain ordered for bulk history seeding.

    The default quote chain (EODHD → Yahoo → Stooq) optimizes single fresh
    quotes; a 60-anchor backfill wants full-history providers whose download
    memo serves every anchor from ONE request per symbol. Yahoo has no
    full-history path (per-anchor window fetches), so it goes last —
    with it second, a quota-starved EODHD dropped the whole fleet
    onto Yahoo at ~40s/name (~17h projected). Honors the disabled-provider
    config exactly like the default chain.
    """
    from app.market.price_provider import (
        ChainedPriceProvider,
        EODHDProvider,
        StooqProvider,
        StooqSecondaryProvider,
        YahooFinanceProvider,
        get_default_provider,
    )

    cfg = get_config()
    provider_name = str(cfg.price_provider or "").strip().lower()
    if provider_name in {"disabled", "off", "none"}:
        return get_default_provider(cfg)
    providers: list[Any] = []
    if cfg.eodhd_apikey:
        providers.append(EODHDProvider(cfg))
    providers.extend(
        [
            StooqProvider(cfg),
            StooqSecondaryProvider(cfg),
            YahooFinanceProvider(cfg),
        ]
    )
    return ChainedPriceProvider(providers)


def backfill_watchlist_price_history(
    *,
    days: int = DEFAULT_BACKFILL_DAYS,
    tickers: Iterable[str] | None = None,
    db_path: str | Path | None = None,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Seed ~``days`` of daily close+volume rows for watchlist names.

    Cheap by construction: the provider downloads full daily history once
    per symbol and serves every anchor date from that cache; rows land in
    price_quotes via the existing idempotent upsert. After the backfill,
    ADV fields are persisted for every covered name.
    """
    from app.market.price_history_backfill import backfill_daily_history
    from app.watchlist.schema import resolve_db_path
    from app.db import connect

    if provider is None:
        provider = _bulk_history_provider()

    if tickers is None:
        conn = connect(resolve_db_path(db_path))
        try:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM watchlist WHERE status != 'REMOVED' ORDER BY ticker"
            ).fetchall()
        finally:
            conn.close()
        tickers = [str(row[0]).upper() for row in rows]
    tickers = list(tickers)

    today = date.today()
    anchors = [
        (today - timedelta(days=offset)).isoformat()
        for offset in range(days, -1, -1)
        if (today - timedelta(days=offset)).weekday() < 5
    ]

    # Per-ticker: seed history, then persist that ticker's ADV immediately.
    # A fleet run through rate-limited providers spans hours — persisting
    # everything at the end meant an interrupted run banked price rows but
    # ZERO capacity classes. Interrupt/resume now loses only the name in
    # flight; the provider's download memo carries across calls.
    rows_written = 0
    failed_tickers: list[str] = []
    adv_results: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        summary = backfill_daily_history(
            tickers=[ticker],
            anchor_dates=anchors,
            benchmark_symbols=(),
            provider=provider,
            db_path=db_path,
        )
        rows_written += summary.rows_written
        failed_tickers.extend(summary.failed_tickers)
        adv_results[ticker] = persist_watchlist_adv(ticker, db_path=db_path).to_dict()

    return {
        "tickers": len(tickers),
        "anchor_days": len(anchors),
        "rows_written": rows_written,
        "failed_tickers": sorted(failed_tickers),
        "adv": adv_results,
    }


__all__ = [
    "ADV_UNKNOWN",
    "AdvResult",
    "adv_dollar_floor",
    "backfill_watchlist_price_history",
    "capacity_class_for",
    "compute_adv",
    "persist_watchlist_adv",
]
