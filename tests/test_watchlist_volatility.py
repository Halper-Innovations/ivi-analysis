"""Tests for realized-volatility accessor with graceful degradation.

The accessor reads ``price_quotes`` and returns the annualized stdev of daily
log returns, or ``None`` when there is insufficient history. Today production
``price_quotes`` rows are all ``status='UNKNOWN'`` (no ``'OK'`` quotes
have been backfilled), so production calls return ``None`` and the discount
function falls back to its neutral volatility term. These tests seed
``status='OK'`` rows to exercise the math directly.

NOTE ON THE 0.158745 LITERAL (spec inconsistency, resolved here):
The original spec cited both "30 quotes" and ``ddof=1`` AND the literal
``round(result, 6) == 0.158745`` (= ``0.01 * sqrt(252)``). Those three are
mutually inconsistent:
  * ``0.158745`` requires the *daily* stdev to be exactly ``0.01``.
  * A balanced alternating +0.01/-0.01 series has daily stdev exactly ``0.01``
    only under POPULATION stdev (ddof=0) AND an EVEN number of returns (mean
    exactly 0). ``ddof=1`` over the same series yields ``0.161459``; an odd
    29-return series (30 quotes) yields a non-zero mean and ``0.158651``.
The authoritative value to assert is the LITERAL ``0.158745``,
which also matches the codebase's established population-stdev convention
(``app/alpha/cross_sectional_ranker.py`` uses ``statistics.pstdev`` / ddof=0).
We therefore implement population stdev (ddof=0) and build a balanced
30-return (31-quote) series so the literal holds EXACTLY. The "ddof=1"/"30
quotes" prose is the erroneous detail, not the literal.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from app.db import init_db
from app.watchlist.volatility import realized_volatility


def _seed_quote(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    as_of_date: str,
    price: float,
    status: str = "OK",
    provider: str = "test",
    fetched_at: str = "2026-01-01T00:00:00Z",
) -> None:
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, source_url,
            status, fetched_at, expires_at, raw_json, quote_hash
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            provider,
            as_of_date,
            price,
            "USD",
            None,
            status,
            fetched_at,
            "2099-01-01T00:00:00Z",
            "{}",
            f"{ticker}-{provider}-{as_of_date}",
        ),
    )


def _date(n: int) -> str:
    """Return a trade date n days after 2026-01-01 (within a 180d window)."""
    day = 1 + n
    month = 4
    while day > 28:
        day -= 28
        month += 1
    return f"2026-{month:02d}-{day:02d}"


@pytest.fixture()
def tmp_db(tmp_path) -> str:
    db_path = tmp_path / "engine.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        init_db(conn=conn)
        conn.commit()
    finally:
        conn.close()
    return str(db_path)


def test_no_quotes_returns_none(tmp_db: str) -> None:
    assert realized_volatility("NOPRICE", db_path=tmp_db) is None


def test_fewer_than_min_observations_returns_none(tmp_db: str) -> None:
    conn = sqlite3.connect(tmp_db)
    try:
        for i in range(4):  # 4 distinct days < min_observations=10
            _seed_quote(conn, ticker="FEW", as_of_date=_date(i), price=100.0 + i)
        conn.commit()
    finally:
        conn.close()
    assert realized_volatility("FEW", db_path=tmp_db) is None


def test_zero_variance_returns_zero(tmp_db: str) -> None:
    conn = sqlite3.connect(tmp_db)
    try:
        for i in range(30):  # 30 distinct days, all price 100.0
            _seed_quote(conn, ticker="STEADY", as_of_date=_date(i), price=100.0)
        conn.commit()
    finally:
        conn.close()
    assert realized_volatility("STEADY", db_path=tmp_db) == 0.0


def test_alternating_log_returns_exact_annualized_stdev(tmp_db: str) -> None:
    # 31 quotes -> 30 balanced log returns alternating +0.01 / -0.01.
    # Daily population stdev == 0.01 exactly; annualized = 0.01 * sqrt(252).
    prices = [100.0]
    sign = 1
    for _ in range(30):
        prices.append(prices[-1] * math.exp(0.01 * sign))
        sign *= -1
    conn = sqlite3.connect(tmp_db)
    try:
        for i, px in enumerate(prices):
            _seed_quote(conn, ticker="VOL", as_of_date=_date(i), price=px)
        conn.commit()
    finally:
        conn.close()
    result = realized_volatility("VOL", db_path=tmp_db)
    assert result is not None
    assert round(result, 6) == 0.158745


def test_non_cached_status_rows_are_ignored(tmp_db: str) -> None:
    conn = sqlite3.connect(tmp_db)
    try:
        for i in range(30):  # 30 UNKNOWN-status rows -> ignored -> 0 usable -> None
            _seed_quote(
                conn,
                ticker="JUNK",
                as_of_date=_date(i),
                price=100.0 + i,
                status="UNKNOWN",
            )
        conn.commit()
    finally:
        conn.close()
    assert realized_volatility("JUNK", db_path=tmp_db) is None
