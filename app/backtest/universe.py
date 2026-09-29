"""Point-in-time, survivorship-aware universe for the scale backtest.

Membership = names that had filed FY fundamentals as-of T (their old filings
persist in companyfacts_facts even after delisting, so then-listed-now-delisted
names are retained). Cap band as-of T = price_T x shares_outstanding(<=cutoff),
in millions, mapped through the production CAP_TIERS boundaries.
"""

from __future__ import annotations

import random
from contextlib import closing
from dataclasses import dataclass

from app.backtest.asof import effective_asof_cutoff
from app.db import connect, get_db
from app.valuation.shares import resolve_market_cap_from_price_asof


# CAP_TIERS upper bounds in millions USD (mirrors app/sector/scan.py:105).
# Lower-inclusive, upper-exclusive.
def cap_category_from_market_cap(mc_millions: float | None) -> str | None:
    """Map a market cap in millions to a canonical cap-category token.

    Band thresholds come from the ONE canonical definition
    (app.autonomous.sector_candidates.MARKET_CAP_FOCUS_TIERS) so backtest
    strata and sweep filters can never diverge. Tokens are chosen so
    select_benchmark_symbol routes large_cap/mega_cap -> SPY and
    micro/small/mid -> IWM (see _LARGE_CAP_TOKENS).
    """
    from app.autonomous.sector_candidates import MARKET_CAP_FOCUS_TIERS

    if mc_millions is None or not isinstance(mc_millions, (int, float)) or mc_millions <= 0:
        return None
    mc = float(mc_millions)
    if mc < float(MARKET_CAP_FOCUS_TIERS["micro"][1]):
        return "micro"
    if mc < float(MARKET_CAP_FOCUS_TIERS["small"][1]):
        return "small"
    if mc < float(MARKET_CAP_FOCUS_TIERS["mid"][1]):
        return "mid"
    if mc < 200_000:
        return "large_cap"
    return "mega_cap"


def sampling_band(cap_category: str | None) -> str | None:
    """Collapse cap-category tokens into the four sampling bands.

    The 'large' band absorbs mega (both route to SPY, so folding is lossless for
    sampling). UNKNOWN_CAP / None are not sampled.
    """
    if cap_category in ("micro", "small", "mid"):
        return cap_category
    if cap_category in ("large_cap", "mega_cap"):
        return "large"
    return None


@dataclass(frozen=True)
class EligibleName:
    ticker: str
    market_cap_asof: float | None
    cap_category_asof: str | None  # canonical token, or "UNKNOWN_CAP"
    price_asof: float | None


def cap_category_asof(ticker: str, as_of_cutoff: str, price: float | None) -> str | None:
    """Cap-category token as-of the filing-lag cutoff, from price_T x shares(<=cutoff).

    Returns None when price is missing/invalid or shares cannot be resolved from a
    non-circular as-of source (resolve_market_cap_from_price_asof rejects shares
    derived from market_cap/price).
    """
    if price is None or not isinstance(price, (int, float)) or price <= 0:
        return None
    mc, _coverage = resolve_market_cap_from_price_asof(
        ticker=ticker, as_of_date=as_of_cutoff, price=float(price)
    )
    return cap_category_from_market_cap(mc)


def eligible_universe_asof(
    as_of_date: str,
    *,
    filing_lag_days: int = 90,
    provider,
    db_path=None,
) -> list[EligibleName]:
    """Names that existed (had filed FY fundamentals) as-of T, with cap band as-of T.

    Membership requires at least one fully sourced FY companyfacts row satisfying
    ``period_end <= filed_date <= cutoff``. The row must carry a canonical
    normalized unit, exact source URL, and accession. This retains
    then-listed-now-delisted names while excluding both not-yet-filed names and
    rows whose filing visibility cannot be proved. A caller-supplied ``db_path``
    is authoritative for membership reads.

    Cap band: price_asof(T) x shares(<=cutoff) in millions. Names without a
    computable cap get cap_category_asof='UNKNOWN_CAP' (counted, not sampled).
    """
    cutoff = effective_asof_cutoff(as_of_date, filing_lag_days=filing_lag_days)

    def _membership_rows(conn):
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
        }
        required_columns = {
            "ticker",
            "period_type",
            "period_end",
            "filed_date",
            "units",
            "source_url",
            "accession",
        }
        if not required_columns <= columns:
            return []
        return conn.execute(
            "SELECT DISTINCT ticker FROM companyfacts_facts "
            "WHERE period_type = 'FY' "
            "AND date(period_end) IS NOT NULL "
            "AND date(filed_date) IS NOT NULL "
            "AND date(period_end) <= date(filed_date) "
            "AND date(period_end) <= date(?) "
            "AND date(filed_date) <= date(?) "
            "AND units IN ('USD_millions', 'shares_millions') "
            "AND source_url IS NOT NULL AND TRIM(source_url) != '' "
            "AND accession IS NOT NULL AND TRIM(accession) != '' "
            "ORDER BY ticker",
            (cutoff, cutoff),
        ).fetchall()

    if db_path is None:
        with get_db() as conn:
            rows = _membership_rows(conn)
    else:
        with closing(connect(db_path)) as conn:
            rows = _membership_rows(conn)

    out: list[EligibleName] = []
    for row in rows:
        ticker = str(row["ticker"]).upper()
        snap = provider.get_price_asof(ticker, as_of_date)
        price = (
            float(snap.price)
            if snap is not None and isinstance(snap.price, (int, float)) and snap.price > 0
            else None
        )
        mc = None
        category = None
        if price is not None:
            mc, _cov = resolve_market_cap_from_price_asof(
                ticker=ticker, as_of_date=cutoff, price=price
            )
            category = cap_category_from_market_cap(mc)
        out.append(
            EligibleName(
                ticker=ticker,
                market_cap_asof=mc,
                cap_category_asof=category or "UNKNOWN_CAP",
                price_asof=price,
            )
        )
    return out


def stratified_sample(eligible, *, per_band: int, seed: int = 42) -> list[str]:
    """Deterministically draw up to `per_band` names from each of the four bands.

    Bands: micro, small, mid, large (large = large_cap + mega_cap). UNKNOWN_CAP is
    never sampled. Under-full bands contribute all their names (no padding). The
    result is sorted for stable downstream run_ids/reports.
    """
    rng = random.Random(seed)
    bands: dict[str, list[str]] = {}
    for name in eligible:
        band = sampling_band(name.cap_category_asof)
        if band is None:
            continue
        bands.setdefault(band, []).append(name.ticker)

    picked: list[str] = []
    for band in ("micro", "small", "mid", "large"):
        names = sorted(set(bands.get(band, [])))
        rng.shuffle(names)
        picked.extend(names[: int(per_band)])
    return sorted(picked)
