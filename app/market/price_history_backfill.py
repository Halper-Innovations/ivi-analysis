"""Persist daily price history for watchlist tickers + a benchmark.

For each ticker (and each benchmark symbol) and each anchor date, fetch a
resolved snapshot from the date-aware price provider and persist it into the
``price_quotes`` table via an idempotent INSERT ON CONFLICT upsert. Because
the underlying StooqProvider downloads full daily history on first touch, a
single ``get_price_asof`` call per anchor seeds the provider's cache; we
persist one ``price_quotes`` row per resolved snapshot so resolution can read
realized prices without re-fetching.

Snapshots that resolve to ``None`` (symbol-not-found, or provider disabled via
``VOE_NET_PROVIDER``) write nothing and mark the ticker as failed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config import AppConfig, get_config
from app.db import get_db, utc_now_iso
from app.util.credential_hygiene import sanitize_url_credentials
from app.util.hashing import sha256_text

logger = logging.getLogger(__name__)


@dataclass
class BackfillSummary:
    rows_written: int = 0
    fetched: int = 0
    failed_tickers: list[str] = field(default_factory=list)


def _persist_snapshot(
    conn: Any,
    *,
    ticker: str,
    provider: str,
    as_of_date: str,
    price: float,
    currency: str,
    source_url: str,
    ttl_seconds: int,
    volume: float | None = None,
) -> None:
    source_url = sanitize_url_credentials(source_url) or ""
    now = utc_now_iso()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    ).isoformat()
    provenance = {"mode": "price_history_backfill", "source": provider}
    raw_json = json.dumps(provenance, sort_keys=True)
    quote_hash = sha256_text(
        json.dumps(
            {
                "ticker": ticker,
                "provider": provider,
                "as_of_date": as_of_date,
                "price": price,
                "status": "OK",
                "fetched_at": now,
            },
            sort_keys=True,
        )
    )
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, source_url,
            status, fetched_at, expires_at, raw_json, quote_hash, volume
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, provider, as_of_date) DO UPDATE SET
            price=excluded.price,
            currency=excluded.currency,
            source_url=excluded.source_url,
            status=excluded.status,
            fetched_at=excluded.fetched_at,
            expires_at=excluded.expires_at,
            raw_json=excluded.raw_json,
            quote_hash=excluded.quote_hash,
            volume=COALESCE(excluded.volume, price_quotes.volume)
        """,
        (
            ticker,
            provider,
            as_of_date,
            price,
            currency,
            source_url,
            "OK",
            now,
            expires_at,
            raw_json,
            quote_hash,
            volume,
        ),
    )


def backfill_daily_history(
    *,
    tickers: Iterable[str],
    anchor_dates: Iterable[str],
    benchmark_symbols: tuple[str, ...] = ("SPY",),
    provider: Any | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> BackfillSummary:
    """Persist daily price history for watchlist tickers + benchmark symbols.

    For each symbol (tickers followed by benchmark symbols) and each anchor
    date, call ``provider.get_price_asof(symbol, date)``; persist a
    ``price_quotes`` row per non-None snapshot. A symbol that yields a None
    snapshot for any anchor date is recorded in ``failed_tickers`` and writes
    no rows for that resolution.
    """
    cfg = cfg or get_config()
    ttl_seconds = int(getattr(cfg, "quote_ttl_seconds", 86400) or 86400)

    if provider is None:
        from app.market.price_provider import get_default_provider

        provider = get_default_provider(cfg)
    # One full-history download per symbol instead of one per (symbol, anchor)
    # — anchor-major re-downloads burned provider rate limits and hours of
    # wall clock (2026-07-16). No-op for providers without the memo.
    enable_memo = getattr(provider, "enable_history_memo", None)
    if callable(enable_memo):
        enable_memo()

    anchors = list(anchor_dates)
    symbols = list(tickers) + list(benchmark_symbols)

    summary = BackfillSummary()
    failed: set[str] = set()
    # Rows_written counts DISTINCT persisted (ticker, provider, as_of_date)
    # keys. Two anchor dates can resolve to the same trading day (weekend /
    # holiday), which the ON CONFLICT upsert collapses to one row — don't
    # double-count it. fetched still counts every provider call.
    seen_keys: set[tuple[str, str, str]] = set()

    with get_db(cfg) as conn:
        for symbol in symbols:
            for anchor in anchors:
                try:
                    snapshot = provider.get_price_asof(symbol, anchor)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "price_history_backfill %s @ %s: fetch failed: %s",
                        symbol,
                        anchor,
                        exc,
                    )
                    snapshot = None

                price = (
                    snapshot.price
                    if snapshot is not None
                    and isinstance(snapshot.price, (int, float))
                    else None
                )
                if snapshot is None or price is None:
                    failed.add(symbol)
                    continue

                summary.fetched += 1
                _persist_snapshot(
                    conn,
                    ticker=symbol,
                    provider=snapshot.source,
                    as_of_date=snapshot.as_of_date,
                    price=float(price),
                    currency=snapshot.currency,
                    source_url=snapshot.url or "",
                    ttl_seconds=ttl_seconds,
                    volume=getattr(snapshot, "volume", None),
                )
                # Commit per row: provider fetches run inside this loop, so
                # one run-wide transaction would hold the engine.db write
                # lock for the entire multi-hour backfill (2026-07-16).
                conn.commit()
                key = (symbol, snapshot.source, snapshot.as_of_date)
                if key not in seen_keys:
                    seen_keys.add(key)
                    summary.rows_written += 1

    summary.failed_tickers = sorted(failed)
    return summary
