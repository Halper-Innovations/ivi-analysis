"""
Data Foundation acceptance tests.
These make real SEC API calls. Run with: pytest tests/test_data_foundation.py -v -s
Do NOT include in the standard CI unit suite (--ignore=tests/test_data_foundation.py).
"""
from __future__ import annotations
import os
import pytest
from app.ingest.cik_registry import resolve, CIKNotFoundError
from app.ingest.companyfacts import fetch_annual_facts

pytestmark = [
    pytest.mark.sec_live,
    pytest.mark.skipif(
        os.environ.get("VOE_RUN_SEC_LIVE_TESTS") != "1",
        reason="live SEC data-foundation acceptance tests require VOE_RUN_SEC_LIVE_TESTS=1",
    ),
]

KNOWN_TICKERS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}

CORE_LINE_ITEMS = ["revenue", "operating_income", "net_income", "cfo"]
ALL_LINE_ITEMS = [
    "revenue", "gross_profit", "operating_income", "net_income",
    "cfo", "capex", "cash", "total_debt", "equity", "shares_outstanding",
]


@pytest.mark.parametrize("ticker,expected_cik", KNOWN_TICKERS.items())
def test_cik_resolves(ticker, expected_cik):
    cik = resolve(ticker)
    assert cik == expected_cik, f"{ticker}: expected {expected_cik}, got {cik}"


def test_unknown_ticker_raises():
    with pytest.raises(CIKNotFoundError):
        resolve("ZZZZZZ_FAKE")


@pytest.mark.parametrize("ticker", KNOWN_TICKERS.keys())
def test_core_line_items_populated(ticker):
    cik = resolve(ticker)
    facts = fetch_annual_facts(cik, years_back=10)
    by_year_item = {(f["line_item"], f["fiscal_year"]): f["value"] for f in facts}
    years_present = sorted({f["fiscal_year"] for f in facts}, reverse=True)
    assert len(years_present) >= 8, f"{ticker}: only {len(years_present)} years of data"
    for item in CORE_LINE_ITEMS:
        item_years = [y for (li, y) in by_year_item if li == item]
        assert len(item_years) >= 8, f"{ticker}: '{item}' only has {len(item_years)} years"


def test_aapl_fy2021_revenue_plausible():
    """AAPL FY2021 revenue was $365.8B. Validates unit normalization."""
    cik = resolve("AAPL")
    facts = fetch_annual_facts(cik, years_back=10)
    aapl_2021_rev = next(
        (f["value"] for f in facts if f["line_item"] == "revenue" and f["fiscal_year"] == 2021),
        None,
    )
    assert aapl_2021_rev is not None, "AAPL FY2021 revenue missing"
    assert 340_000 <= aapl_2021_rev <= 400_000, \
        f"AAPL FY2021 revenue out of range: {aapl_2021_rev} (expected ~365800)"


def test_aapl_fy2016_revenue_plausible():
    """AAPL FY2016 revenue was ~$215B. Validates historical unit normalization."""
    cik = resolve("AAPL")
    facts = fetch_annual_facts(cik, years_back=12)
    aapl_2016_rev = next(
        (f["value"] for f in facts if f["line_item"] == "revenue" and f["fiscal_year"] == 2016),
        None,
    )
    assert aapl_2016_rev is not None, "AAPL FY2016 revenue missing"
    assert 210_000 <= aapl_2016_rev <= 220_000, \
        f"AAPL FY2016 revenue out of range: {aapl_2016_rev} (expected ~215000)"


def test_no_implausible_values():
    """No revenue value should be negative or impossibly small for AAPL/MSFT/NVDA."""
    for ticker in KNOWN_TICKERS:
        cik = resolve(ticker)
        facts = fetch_annual_facts(cik, years_back=10)
        for f in facts:
            if f["line_item"] == "revenue":
                assert f["value"] > 100, \
                    f"{ticker} FY{f['fiscal_year']} revenue suspiciously small: {f['value']}"


def test_smaller_cap_ticker_resolves_and_plausible():
    """
    A smaller-cap company should resolve and return plausible values.
    Uses CBRE Group as a mid-cap with well-covered XBRL filings.
    """
    ticker = "CBRE"
    cik = resolve(ticker)
    facts = fetch_annual_facts(cik, years_back=5)
    rev_facts = [f for f in facts if f["line_item"] == "revenue"]
    assert len(rev_facts) >= 3, f"{ticker}: fewer than 3 years of revenue data"
    for f in rev_facts:
        assert 100 < f["value"] < 500_000, \
            f"{ticker} FY{f['fiscal_year']} revenue out of plausible range: {f['value']}"


def test_no_duplicate_rows_on_rerun(monkeypatch, tmp_path):
    """Re-running ensure_facts must not create duplicate rows."""
    import os
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    os.environ["VOE_DATA_DIR"] = str(data_dir)
    os.environ["VOE_DB_PATH"] = str(db_path)
    os.environ["VOE_SAFE_MODE"] = "true"
    from app.config import get_config as _gc
    _gc.cache_clear()
    from app.db import init_db, get_db
    init_db(_gc())

    from app.ingest.facts_writer import ensure_facts
    ensure_facts("MSFT", years_back=3)

    with get_db() as conn:
        count_after_first = conn.execute(
            "SELECT COUNT(*) as n FROM companyfacts_facts WHERE ticker='MSFT' AND line_item='revenue'"
        ).fetchone()["n"]

    # Force second run by clearing the TTL check: reset fetched_at to the past
    from datetime import timezone, timedelta
    past_ts = (
        __import__("datetime").datetime.now(timezone.utc) - timedelta(hours=25)
    ).isoformat()
    with get_db() as conn:
        conn.execute(
            "UPDATE companyfacts_facts SET fetched_at=? WHERE ticker='MSFT'",
            (past_ts,),
        )

    ensure_facts("MSFT", years_back=3)  # second run after TTL expiry

    with get_db() as conn:
        count_after_second = conn.execute(
            "SELECT COUNT(*) as n FROM companyfacts_facts WHERE ticker='MSFT' AND line_item='revenue'"
        ).fetchone()["n"]

    assert count_after_first >= 1, "Expected at least 1 revenue row for MSFT after first run"
    assert count_after_second == count_after_first, (
        f"Duplicate rows detected: {count_after_first} rows after run 1, "
        f"{count_after_second} rows after run 2"
    )
