"""Ungated factual quote refresh for authoritative current watchlist rows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from app.autonomous.financial_integrity import (
    PRICE_BASIS_UNADJUSTED,
    stable_quote_hash,
)
from app.config import AppConfig, get_config
from app.db import connect
from app.market.price_provider import (
    ChainedPriceProvider,
    PriceProvider,
    PriceSnapshot,
    StooqProvider,
    YahooFinanceProvider,
)
from app.util.credential_hygiene import sanitize_json_value, sanitize_url_credentials
from app.watchlist.schema import resolve_db_path
from app.watchlist.store import list_current_for_market_data


REQUIRED_WATCHLIST_SNAPSHOT_COLUMNS = {
    "quote_as_of_date",
    "currency",
    "price_basis",
    "quote_snapshot_id",
    "price_quote_id",
}
CACHED_QUOTE_MAX_AGE_DAYS = 5


class PriceRefreshMigrationRequired(RuntimeError):
    """Raised before provider calls when the additive migration is unapplied."""


@dataclass(frozen=True)
class PriceRefreshSummary:
    as_of_date: str
    candidates: int
    attempts: int
    written: int
    unavailable: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        if self.candidates <= 0:
            return 2
        if self.attempts <= 0 or self.written <= 0 or self.failed:
            return 1
        # Providers swallow outages into per-ticker None results, so a mass
        # outage arrives as `unavailable` with `failed` empty. A majority-dark
        # run is a failure even when the few survivors wrote cleanly.
        if len(self.unavailable) * 2 > self.attempts:
            return 1
        return 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unavailable_count"] = len(self.unavailable)
        payload["failed_count"] = len(self.failed)
        payload["status"] = "OK" if self.exit_code == 0 else "FAILED"
        return payload


def _free_unadjusted_provider(cfg: AppConfig) -> PriceProvider:
    """Build the existing free Yahoo -> Stooq chain with no paid-provider lane."""

    return ChainedPriceProvider(
        [
            YahooFinanceProvider(
                cfg,
                fallback_days=cfg.price_fallback_days,
                auto_adjust=False,
            ),
            StooqProvider(
                cfg,
                fallback_days=cfg.price_fallback_days,
            ),
        ]
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _require_migration(conn: sqlite3.Connection) -> None:
    missing = REQUIRED_WATCHLIST_SNAPSHOT_COLUMNS - _table_columns(
        conn, "watchlist_price_snapshots"
    )
    if missing:
        raise PriceRefreshMigrationRequired(
            "watchlist_price_snapshots migration required; missing columns: "
            + ", ".join(sorted(missing))
        )


def _quote_payload(snapshot: PriceSnapshot) -> dict[str, Any]:
    return {
        "ticker": snapshot.ticker.upper(),
        "as_of_date": snapshot.as_of_date,
        "price": float(snapshot.price),
        "currency": snapshot.currency.upper(),
        "source": snapshot.source,
        "source_url": sanitize_url_credentials(snapshot.url) or "",
        "price_basis": PRICE_BASIS_UNADJUSTED,
        "raw_price": float(snapshot.price),
        "split_adjustment_factor": 1.0,
        "split_effective_date": None,
    }


def _persist_snapshot(
    conn: sqlite3.Connection,
    *,
    watchlist_id: int,
    snapshot: PriceSnapshot,
    checked_at: str,
    expires_at: str,
    diagnostic: dict[str, Any] | None,
) -> None:
    quote = _quote_payload(snapshot)
    quote_snapshot_id = stable_quote_hash(quote)
    raw_json = json.dumps(
        sanitize_json_value(
            {
                "quote": quote,
                "confidence": snapshot.confidence,
                "diagnostic": diagnostic or {},
            }
        ),
        sort_keys=True,
    )
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, price_basis,
            split_adjustment_factor, split_effective_date, source_url,
            status, fetched_at, expires_at, raw_json, quote_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OK', ?, ?, ?, ?)
        ON CONFLICT(ticker, provider, as_of_date) DO UPDATE SET
            price=excluded.price,
            currency=excluded.currency,
            price_basis=excluded.price_basis,
            split_adjustment_factor=excluded.split_adjustment_factor,
            split_effective_date=excluded.split_effective_date,
            source_url=excluded.source_url,
            status=excluded.status,
            fetched_at=excluded.fetched_at,
            expires_at=excluded.expires_at,
            raw_json=excluded.raw_json,
            quote_hash=excluded.quote_hash
        """,
        (
            quote["ticker"],
            quote["source"],
            quote["as_of_date"],
            quote["price"],
            quote["currency"],
            quote["price_basis"],
            quote["split_adjustment_factor"],
            quote["split_effective_date"],
            quote["source_url"],
            checked_at,
            expires_at,
            raw_json,
            quote_snapshot_id,
        ),
    )
    price_quote_row = conn.execute(
        """
        SELECT id
        FROM price_quotes
        WHERE ticker = ? AND provider = ? AND as_of_date = ?
        """,
        (quote["ticker"], quote["source"], quote["as_of_date"]),
    ).fetchone()
    if price_quote_row is None:
        raise RuntimeError("price quote upsert did not produce a row")
    conn.execute(
        """
        INSERT INTO watchlist_price_snapshots(
            watchlist_id, price, checked_at, source, quote_as_of_date,
            currency, price_basis, quote_snapshot_id, price_quote_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(watchlist_id),
            quote["price"],
            checked_at,
            quote["source"],
            quote["as_of_date"],
            quote["currency"],
            quote["price_basis"],
            quote_snapshot_id,
            int(price_quote_row[0]),
        ),
    )


def refresh_current_watchlist_prices(
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    provider: PriceProvider | None = None,
) -> PriceRefreshSummary:
    """Fetch and persist facts only; never evaluate or mutate trigger status."""

    effective_as_of = str(as_of_date or date.today().isoformat()).strip()[:10]
    date.fromisoformat(effective_as_of)
    resolved_cfg = cfg or get_config()
    resolved_path = resolve_db_path(db_path)
    resolved_cfg = resolved_cfg.model_copy(update={"db_path": resolved_path})
    candidates = list_current_for_market_data(db_path=resolved_path)

    conn = connect(resolved_path, cfg=resolved_cfg)
    try:
        _require_migration(conn)
        price_provider = provider or _free_unadjusted_provider(resolved_cfg)
        attempts = 0
        written = 0
        unavailable: list[str] = []
        failed: dict[str, str] = {}
        for candidate in candidates:
            attempts += 1
            try:
                snapshot = price_provider.get_price_asof(
                    candidate.ticker, effective_as_of
                )
            except Exception as exc:  # noqa: BLE001 - counted and surfaced
                failed[candidate.ticker] = type(exc).__name__
                continue
            if snapshot is None:
                unavailable.append(candidate.ticker)
                continue
            checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=max(1, int(resolved_cfg.quote_ttl_seconds)))
            ).replace(microsecond=0).isoformat()
            try:
                diagnostic = price_provider.get_last_diagnostic(
                    candidate.ticker, effective_as_of
                )
                _persist_snapshot(
                    conn,
                    watchlist_id=int(candidate.id),
                    snapshot=snapshot,
                    checked_at=checked_at,
                    expires_at=expires_at,
                    diagnostic=diagnostic,
                )
                conn.commit()
                written += 1
            except Exception as exc:  # noqa: BLE001 - isolate per-ticker write
                conn.rollback()
                failed[candidate.ticker] = type(exc).__name__
        return PriceRefreshSummary(
            as_of_date=effective_as_of,
            candidates=len(candidates),
            attempts=attempts,
            written=written,
            unavailable=unavailable,
            failed=failed,
        )
    finally:
        conn.close()


def build_cached_price_lookup(
    db_path: str | Path,
    *,
    max_age_days: int = CACHED_QUOTE_MAX_AGE_DAYS,
    providers: tuple[str, ...] = ("yahoo", "stooq"),
) -> Callable[[str, str], dict[str, Any] | None]:
    """Return a read-only, provenance-preserving lookup over ``price_quotes``.

    Only rows from the refresh's own free providers are eligible: other
    writers (e.g. ``eodhd_split_lineage``) answer to a different lineage
    contract, and selecting them here would surface only as a silent
    hash-rejection.
    """

    resolved = Path(db_path)
    provider_filter = tuple(str(p) for p in providers)
    if not provider_filter:
        raise ValueError("providers must be non-empty")
    placeholders = ", ".join("?" for _ in provider_filter)

    def lookup(ticker: str, as_of_date: str) -> dict[str, Any] | None:
        try:
            conn = sqlite3.connect(
                f"file:{resolved}?mode=ro", uri=True, timeout=10.0
            )
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    f"""
                    SELECT ticker, provider, as_of_date, price, currency,
                           price_basis, split_adjustment_factor,
                           split_effective_date, source_url, quote_hash
                    FROM price_quotes
                    WHERE ticker = ?
                      AND as_of_date <= ?
                      AND status = 'OK'
                      AND price IS NOT NULL
                      AND provider IN ({placeholders})
                    ORDER BY as_of_date DESC, fetched_at DESC
                    LIMIT 1
                    """,
                    (str(ticker).upper(), str(as_of_date)[:10], *provider_filter),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        used = date.fromisoformat(str(row["as_of_date"])[:10])
        requested = date.fromisoformat(str(as_of_date)[:10])
        if (requested - used).days > max(0, int(max_age_days)):
            return None
        payload = {
            "ticker": str(row["ticker"]).upper(),
            "as_of_date": used.isoformat(),
            "price": float(row["price"]),
            "currency": str(row["currency"] or "").upper(),
            "source": str(row["provider"] or ""),
            "source_url": str(row["source_url"] or ""),
            "price_basis": str(row["price_basis"] or "").upper(),
            "raw_price": (
                float(row["price"])
                if str(row["price_basis"] or "").upper()
                == PRICE_BASIS_UNADJUSTED
                else None
            ),
            "split_adjustment_factor": row["split_adjustment_factor"],
            "split_effective_date": row["split_effective_date"],
            "confidence": "HIGH" if used == requested else "MEDIUM",
        }
        if stable_quote_hash(payload) != str(row["quote_hash"] or ""):
            return None
        return payload

    return lookup


__all__ = [
    "PriceRefreshMigrationRequired",
    "PriceRefreshSummary",
    "build_cached_price_lookup",
    "refresh_current_watchlist_prices",
]
