from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.db import utc_now_iso
from app.fundamentals.normalize import UNKNOWN
from app.valuation.price_provider import PriceProvider


@dataclass
class MarketCapResult:
    ticker: str
    run_id: str
    run_as_of_date: str
    effective_as_of_date: str
    market_cap: float | str
    market_cap_unit: str
    market_cap_status: str
    market_cap_in_band: bool
    price: float | None
    price_currency: str
    price_basis: str
    quote_snapshot_id: str
    shares_outstanding: float | None
    shares_unit: str
    shares_basis: str
    split_adjustment_factor: float
    split_effective_date: str | None
    provider: str
    source_url: str
    fetched_at: str
    flags: list[str]
    payload: dict[str, Any]


def compute_market_cap(
    *,
    ticker: str,
    run_id: str,
    run_as_of_date: str,
    effective_as_of_date: str,
    shares_outstanding: float | str,
    price_provider: PriceProvider,
    cap_min: float,
    cap_max: float,
) -> MarketCapResult:
    quote = price_provider.get_quote(ticker.upper(), run_as_of_date)
    flags: list[str] = []
    shares = shares_outstanding if isinstance(shares_outstanding, (int, float)) else None
    price = quote.price if isinstance(quote.price, (int, float)) else None

    if shares is None:
        flags.append("SHARES_UNKNOWN")
    if price is None:
        flags.append("PRICE_UNKNOWN")

    market_cap: float | str = UNKNOWN
    in_band = False
    status = "UNKNOWN"
    if shares is not None and price is not None:
        market_cap = float(shares) * float(price)
        status = "OK"
        in_band = cap_min <= market_cap <= cap_max
        if not in_band:
            flags.append("MARKET_CAP_OUT_OF_BAND")
    else:
        flags.append("MARKET_CAP_UNKNOWN")

    payload = {
        "ticker": ticker.upper(),
        "run_id": run_id,
        "run_as_of_date": run_as_of_date,
        "effective_as_of_date": effective_as_of_date,
        "market_cap": market_cap,
        "market_cap_unit": "USD",
        "market_cap_status": status,
        "market_cap_in_band": in_band,
        "price": price,
        "price_currency": quote.currency,
        "price_basis": quote.price_basis,
        "quote_snapshot_id": quote.quote_snapshot_id,
        "price_status": quote.status,
        "price_provider": quote.provider,
        "price_source_url": quote.source_url,
        "price_fetched_at": quote.fetched_at,
        "shares_outstanding": shares,
        "shares_unit": "shares",
        "shares_basis": "raw",
        "split_adjustment_factor": quote.split_adjustment_factor,
        "split_effective_date": quote.split_effective_date,
    }
    return MarketCapResult(
        ticker=ticker.upper(),
        run_id=run_id,
        run_as_of_date=run_as_of_date,
        effective_as_of_date=effective_as_of_date,
        market_cap=market_cap,
        market_cap_unit="USD",
        market_cap_status=status,
        market_cap_in_band=in_band,
        price=price,
        price_currency=quote.currency,
        price_basis=quote.price_basis,
        quote_snapshot_id=str(quote.quote_snapshot_id),
        shares_outstanding=shares,
        shares_unit="shares",
        shares_basis="raw",
        split_adjustment_factor=float(quote.split_adjustment_factor),
        split_effective_date=quote.split_effective_date,
        provider=quote.provider,
        source_url=quote.source_url,
        fetched_at=quote.fetched_at,
        flags=sorted(set(flags)),
        payload=payload,
    )


def persist_market_cap(conn, result: MarketCapResult) -> None:
    conn.execute(
        """
        INSERT INTO market_caps(
            ticker, effective_as_of_date, run_id, run_as_of_date,
            market_cap, market_cap_unit, market_cap_status,
            price, price_currency, price_basis, quote_snapshot_id,
            shares_outstanding, shares_unit, shares_basis,
            split_adjustment_factor, split_effective_date,
            provider, source_url, fetched_at, payload_json, created_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, effective_as_of_date, run_id) DO UPDATE SET
            run_as_of_date=excluded.run_as_of_date,
            market_cap=excluded.market_cap,
            market_cap_unit=excluded.market_cap_unit,
            market_cap_status=excluded.market_cap_status,
            price=excluded.price,
            price_currency=excluded.price_currency,
            price_basis=excluded.price_basis,
            quote_snapshot_id=excluded.quote_snapshot_id,
            shares_outstanding=excluded.shares_outstanding,
            shares_unit=excluded.shares_unit,
            shares_basis=excluded.shares_basis,
            split_adjustment_factor=excluded.split_adjustment_factor,
            split_effective_date=excluded.split_effective_date,
            provider=excluded.provider,
            source_url=excluded.source_url,
            fetched_at=excluded.fetched_at,
            payload_json=excluded.payload_json,
            created_at=excluded.created_at
        """,
        (
            result.ticker,
            result.effective_as_of_date,
            result.run_id,
            result.run_as_of_date,
            result.market_cap if isinstance(result.market_cap, (int, float)) else None,
            result.market_cap_unit,
            result.market_cap_status,
            result.price,
            result.price_currency,
            result.price_basis,
            result.quote_snapshot_id,
            result.shares_outstanding,
            result.shares_unit,
            result.shares_basis,
            result.split_adjustment_factor,
            result.split_effective_date,
            result.provider,
            result.source_url,
            result.fetched_at,
            json.dumps(result.payload),
            utc_now_iso(),
        ),
    )
