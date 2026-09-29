from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.autonomous.financial_integrity import (
    PRICE_BASIS_SPLIT_ADJUSTED,
    PRICE_BASIS_UNADJUSTED,
    stable_quote_hash,
)
from app.config import AppConfig, get_config
from app.db import get_db, utc_now_iso
from app.util.credential_hygiene import sanitize_json_value, sanitize_url_credentials
from app.util.http import HttpClient


UNKNOWN = "UNKNOWN"


@dataclass
class PriceQuote:
    ticker: str
    as_of_date: str
    price: float | None
    currency: str
    provider: str
    status: str
    source_url: str
    fetched_at: str
    expires_at: str
    provenance: dict[str, Any]
    price_basis: str = PRICE_BASIS_UNADJUSTED
    split_adjustment_factor: float = 1.0
    split_effective_date: str | None = None
    quote_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        self.source_url = sanitize_url_credentials(self.source_url) or ""
        self.provenance = sanitize_json_value(self.provenance)
        basis_aliases = {
            "raw_close": PRICE_BASIS_UNADJUSTED,
            "adjusted_close": PRICE_BASIS_SPLIT_ADJUSTED,
            "split_adjusted_close": PRICE_BASIS_SPLIT_ADJUSTED,
        }
        self.price_basis = basis_aliases.get(self.price_basis, self.price_basis)
        if self.price_basis not in {
            PRICE_BASIS_UNADJUSTED,
            PRICE_BASIS_SPLIT_ADJUSTED,
        }:
            raise ValueError(f"unsupported price basis: {self.price_basis!r}")
        if not isinstance(self.split_adjustment_factor, (int, float)) or float(
            self.split_adjustment_factor
        ) <= 0:
            raise ValueError("split_adjustment_factor must be positive")
        snapshot_payload = {
            "ticker": self.ticker.upper(),
            "value": self.price,
            "currency": self.currency.upper(),
            "as_of_date": self.as_of_date,
            "source": self.provider,
            "source_url": self.source_url,
            "price_basis": self.price_basis,
            "split_adjustment_factor": float(self.split_adjustment_factor),
            "split_effective_date": self.split_effective_date,
        }
        stable_id = stable_quote_hash(snapshot_payload)
        if self.quote_snapshot_id is not None and self.quote_snapshot_id != stable_id:
            raise ValueError("quote_snapshot_id does not match the quote payload")
        self.quote_snapshot_id = stable_id


class PriceProvider:
    provider_name = "base"

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()

    def get_quote(self, ticker: str, as_of_date: str) -> PriceQuote:
        raise NotImplementedError


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(days=3650)


def _read_cached_quote(ticker: str, provider: str, as_of_date: str) -> PriceQuote | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT ticker, as_of_date, price, currency, price_basis,
                   split_adjustment_factor, split_effective_date, quote_hash,
                   provider, status, source_url, fetched_at, expires_at, raw_json
            FROM price_quotes
            WHERE ticker = ? AND provider = ? AND as_of_date = ?
            """,
            (ticker, provider, as_of_date),
        ).fetchone()
    if not row:
        return None
    expires_at = row["expires_at"]
    if _parse_iso(expires_at) < datetime.now(timezone.utc):
        return None
    provenance = json.loads(row["raw_json"])
    return PriceQuote(
        ticker=row["ticker"],
        as_of_date=row["as_of_date"],
        price=row["price"],
        currency=row["currency"] or "USD",
        provider=row["provider"],
        status=row["status"],
        source_url=row["source_url"] or "",
        fetched_at=row["fetched_at"],
        expires_at=row["expires_at"],
        provenance=provenance,
        price_basis=row["price_basis"] or PRICE_BASIS_UNADJUSTED,
        split_adjustment_factor=(
            float(row["split_adjustment_factor"])
            if row["split_adjustment_factor"] is not None
            else 1.0
        ),
        split_effective_date=row["split_effective_date"],
        quote_snapshot_id=row["quote_hash"],
    )


def _cache_quote(quote: PriceQuote) -> None:
    raw_json = json.dumps(quote.provenance, sort_keys=True)
    quote_hash = str(quote.quote_snapshot_id)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO price_quotes(
                ticker, provider, as_of_date, price, currency, price_basis,
                split_adjustment_factor, split_effective_date, source_url,
                status, fetched_at, expires_at, raw_json, quote_hash
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                quote.ticker,
                quote.provider,
                quote.as_of_date,
                quote.price,
                quote.currency,
                quote.price_basis,
                quote.split_adjustment_factor,
                quote.split_effective_date,
                quote.source_url,
                quote.status,
                quote.fetched_at,
                quote.expires_at,
                raw_json,
                quote_hash,
            ),
        )


class DisabledPriceProvider(PriceProvider):
    provider_name = "disabled"

    def __init__(self, cfg: AppConfig | None = None, *, reason: str = "price_provider_disabled") -> None:
        super().__init__(cfg)
        self.reason = reason

    def get_quote(self, ticker: str, as_of_date: str) -> PriceQuote:
        now = utc_now_iso()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=self.cfg.quote_ttl_seconds)).isoformat()
        quote = PriceQuote(
            ticker=ticker,
            as_of_date=as_of_date,
            price=None,
            currency="USD",
            provider=self.provider_name,
            status="UNKNOWN",
            source_url="",
            fetched_at=now,
            expires_at=expires,
            provenance={"mode": "offline_disabled", "reason": self.reason},
        )
        _cache_quote(quote)
        return quote


class StooqPriceProvider(PriceProvider):
    provider_name = "stooq"

    def __init__(self, cfg: AppConfig | None = None) -> None:
        super().__init__(cfg)
        self.http = HttpClient(self.cfg)

    @staticmethod
    def _ticker_symbol(ticker: str) -> str:
        # Stooq uses .US suffix for US equities.
        return f"{ticker.lower().replace('.', '-')}.us"

    def _fetch_quote(self, ticker: str, as_of_date: str) -> PriceQuote:
        params = {
            "s": self._ticker_symbol(ticker),
            "i": "d",
            "f": "sd2t2ohlcv",
            "h": "",
            "e": "csv",
        }
        url = self.cfg.stooq_base_url
        now = utc_now_iso()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=self.cfg.quote_ttl_seconds)).isoformat()
        try:
            payload = self.http.get_bytes(url, params=params, use_cache=False, cache_ttl_seconds=0)
            text = payload.decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            row = next(reader, None)
            if not row:
                quote = PriceQuote(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    price=None,
                    currency="USD",
                    provider=self.provider_name,
                    status="UNKNOWN",
                    source_url=url,
                    fetched_at=now,
                    expires_at=expires,
                    provenance={"params": params, "raw": text[:500], "reason": "empty_csv"},
                )
                _cache_quote(quote)
                return quote

            close_raw = row.get("Close") or row.get("close")
            date_raw = row.get("Date") or row.get("date") or as_of_date
            try:
                price = float(close_raw) if close_raw and close_raw not in {"N/D", ""} else None
            except ValueError:
                price = None

            quote = PriceQuote(
                ticker=ticker,
                as_of_date=as_of_date,
                price=price,
                currency="USD",
                provider=self.provider_name,
                status="OK" if price is not None else "UNKNOWN",
                source_url=url,
                fetched_at=now,
                expires_at=expires,
                provenance={"params": params, "quote_date": date_raw, "row": row},
            )
            _cache_quote(quote)
            return quote
        except Exception as exc:  # noqa: BLE001
            quote = PriceQuote(
                ticker=ticker,
                as_of_date=as_of_date,
                price=None,
                currency="USD",
                provider=self.provider_name,
                status="UNKNOWN",
                source_url=url,
                fetched_at=now,
                expires_at=expires,
                provenance={"params": params, "error": str(exc)},
            )
            _cache_quote(quote)
            return quote

    def get_quote(self, ticker: str, as_of_date: str) -> PriceQuote:
        cached = _read_cached_quote(ticker=ticker, provider=self.provider_name, as_of_date=as_of_date)
        if cached:
            return cached
        return self._fetch_quote(ticker, as_of_date)


class MarketQuoteProvider(PriceProvider):
    """Valuation quote from one ``app.market.price_provider`` source.

    The HTTP code lives in the market module (Yahoo, Stooq, EODHD); this
    adapter only turns a snapshot into a cached ``PriceQuote`` whose price
    basis is stated, never assumed. A source that cannot show an unadjusted
    close yields an UNKNOWN quote with the reason, not a relabeled price.
    """

    def __init__(self, cfg: AppConfig | None, source: Any) -> None:
        super().__init__(cfg)
        self.source = source
        self.provider_name = str(source.provider_name)

    def _raw_close(self, snapshot: Any) -> tuple[float | None, str]:
        if self.provider_name == "eodhd":
            # EODHD's `price` is the adjusted close; the raw close rides beside it.
            raw = getattr(snapshot, "raw_price", None)
            if isinstance(raw, (int, float)) and raw > 0:
                return float(raw), ""
            return None, "EODHD_RAW_CLOSE_UNAVAILABLE"
        # Yahoo (auto_adjust=False) and Stooq daily history are not dividend
        # adjusted, but both sources restate history for later splits. A close
        # is therefore the raw close only while no later split can have
        # restated it: accept it as unadjusted only when the bar is recent.
        quote_day = _parse_iso_date(getattr(snapshot, "as_of_date", ""))
        window = max(7, int(self.cfg.price_fallback_days))
        today = datetime.now(timezone.utc).date()
        if quote_day is None or (today - quote_day).days > window:
            return None, "RAW_CLOSE_UNPROVEN_FOR_HISTORICAL_DATE"
        return float(snapshot.price), ""

    def get_quote(self, ticker: str, as_of_date: str) -> PriceQuote:
        cached = _read_cached_quote(ticker=ticker, provider=self.provider_name, as_of_date=as_of_date)
        if cached and cached.status == "OK":
            return cached
        now = utc_now_iso()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=self.cfg.quote_ttl_seconds)).isoformat()
        snapshot = self.source.get_price_asof(ticker, as_of_date)
        diagnostic = self.source.get_last_diagnostic(ticker, as_of_date) or {}
        provenance: dict[str, Any] = {
            "mode": "live",
            "source": self.provider_name,
            "requested_as_of_date": as_of_date,
            "diagnostic": diagnostic,
        }
        if self.provider_name == "yahoo":
            provenance["history_auto_adjust"] = bool(getattr(self.source, "auto_adjust", True))
        price: float | None = None
        source_url = ""
        if snapshot is None:
            raw_result = diagnostic.get("result")
            result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
            provenance["reason"] = str(
                result.get("reason_code") or diagnostic.get("status") or "PROVIDER_NO_DATA"
            )
        else:
            price, problem = self._raw_close(snapshot)
            source_url = str(getattr(snapshot, "url", None) or "")
            provenance.update(
                {
                    "quote_date": snapshot.as_of_date,
                    "confidence": snapshot.confidence,
                    "retrieved_at": snapshot.retrieved_at,
                    "provider_price": float(snapshot.price),
                }
            )
            if problem:
                provenance["reason"] = problem
        quote = PriceQuote(
            ticker=ticker,
            as_of_date=as_of_date,
            price=price,
            currency=str(getattr(snapshot, "currency", None) or "USD"),
            provider=self.provider_name,
            status="OK" if price is not None else "UNKNOWN",
            source_url=source_url,
            fetched_at=now,
            expires_at=expires,
            provenance=provenance,
            price_basis=PRICE_BASIS_UNADJUSTED,
        )
        _cache_quote(quote)
        return quote


class ChainedQuoteProvider(PriceProvider):
    """Try each source in order; the first OK quote wins.

    When none succeeds the result is an explicit UNKNOWN quote whose
    provenance lists every attempt and its reason.
    """

    def __init__(self, cfg: AppConfig | None, providers: list[PriceProvider]) -> None:
        super().__init__(cfg)
        self.providers = providers
        self.provider_name = "+".join(p.provider_name for p in providers)

    def get_quote(self, ticker: str, as_of_date: str) -> PriceQuote:
        attempts: list[dict[str, Any]] = []
        last: PriceQuote | None = None
        for provider in self.providers:
            quote = provider.get_quote(ticker, as_of_date)
            if quote.status == "OK" and quote.price is not None:
                return quote
            attempts.append(
                {
                    "provider": provider.provider_name,
                    "status": quote.status,
                    "reason": quote.provenance.get("reason"),
                }
            )
            last = quote
        assert last is not None
        return PriceQuote(
            ticker=ticker,
            as_of_date=as_of_date,
            price=None,
            currency="USD",
            provider=self.provider_name,
            status="UNKNOWN",
            source_url="",
            fetched_at=last.fetched_at,
            expires_at=last.expires_at,
            provenance={"mode": "live", "reason": "NO_SOURCE_RETURNED_A_QUOTE", "attempts": attempts},
        )


def _parse_iso_date(value: Any) -> Any:
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def get_default_provider(cfg: AppConfig | None = None) -> PriceProvider:
    """Valuation quotes, governed by ``price_provider`` and ``VOE_NET_PROVIDER``.

    ``safe_mode`` is not consulted: it governs the research web adapters, not
    market quotes. "auto" chains EODHD (only when a key is set), then Yahoo
    (free, keyless), then Stooq (only when a Stooq key is set). "yahoo",
    "stooq" and "eodhd" pin one source; "disabled" turns quotes off.
    """
    from app.market import price_provider as market

    cfg = cfg or get_config()
    provider = str(cfg.price_provider or "").strip().lower()
    if provider in {"disabled", "off", "none"}:
        return DisabledPriceProvider(cfg, reason="price_provider_disabled")
    if market._network_disabled(cfg):  # noqa: SLF001
        return DisabledPriceProvider(cfg, reason="network_disabled")

    def yahoo() -> PriceProvider:
        return MarketQuoteProvider(cfg, market.YahooFinanceProvider(cfg, auto_adjust=False))

    def stooq() -> PriceProvider:
        return MarketQuoteProvider(cfg, market.StooqProvider(cfg))

    def eodhd() -> PriceProvider:
        return MarketQuoteProvider(cfg, market.EODHDProvider(cfg))

    if provider == "yahoo":
        return yahoo()
    if provider == "stooq":
        return stooq()
    if provider == "eodhd":
        if not cfg.eodhd_apikey:
            return DisabledPriceProvider(cfg, reason="eodhd_selected_without_api_key")
        return eodhd()
    chain: list[PriceProvider] = []
    if cfg.eodhd_apikey:
        chain.append(eodhd())
    chain.append(yahoo())
    if cfg.stooq_apikey:
        chain.append(stooq())
    return ChainedQuoteProvider(cfg, chain)
