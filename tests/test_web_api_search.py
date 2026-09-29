from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.db import init_db
from app.watchlist.contract import WatchlistEntry
from app.watchlist.schema import ensure_watchlist_schema
from app.watchlist.store import add_or_update
from app.web.main import app

client = TestClient(app)


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    ensure_watchlist_schema(db_path)
    return cfg


def _entry(*, ticker: str, status: str = "ACTIVE") -> WatchlistEntry:
    return WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade="WATCHLIST_ONLY",
        confidence="HIGH",
        conviction_source="company_autonomy",
        scan_family="normal",
        valuation_anchor_method="DCF",
        valuation_anchor_value=100.0,
        buy_price_target=80.0,
        current_price_at_addition=95.0,
        thesis_text="Search fixture row.",
        key_risks=[],
        falsifiers=[],
        open_questions=[],
        source_run_id="sector_run_search",
        source_sector="industrial_tech",
        added_at="2026-07-01T12:00:00+00:00",
    )


def _seed_registrant(
    conn: sqlite3.Connection,
    *,
    cik: str,
    ticker: str,
    name: str,
    sector: str | None = None,
    sic_description: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sec_registrants
            (cik, primary_ticker, all_tickers, name, sector, sic_description,
             exchange_scope, operating_status, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, 'US_LISTED', 'OPERATING',
                '2026-07-01T00:00:00Z', '2026-07-01T00:00:00Z')
        """,
        (cik, ticker, ticker, name, sector, sic_description),
    )


def _seed(cfg) -> None:
    add_or_update(_entry(ticker="BYD"), db_path=cfg.db_path)
    add_or_update(_entry(ticker="ZZOFF"), db_path=cfg.db_path)  # not in census
    conn = sqlite3.connect(cfg.db_path)
    _seed_registrant(
        conn, cik="0000001", ticker="BYD", name="BOYD GAMING CORP",
        sector="consumer_gaming",
    )
    _seed_registrant(
        conn, cik="0000002", ticker="BGSI", name="Boyd Group Services Inc.",
        sic_description="AUTO REPAIR SERVICES",
    )
    _seed_registrant(
        conn, cik="0000003", ticker="BY", name="Byline Bancorp Inc",
    )
    conn.execute(
        """
        INSERT INTO companyfacts_facts
            (ticker, fiscal_year, period_type, period_end, line_item, value,
             units, source_url, fetched_at)
        VALUES ('BGSI', 2025, 'FY', '2025-12-31', 'revenue', 1000.0,
                'USD', 'https://example.test/facts', '2026-07-01T00:00:00Z')
        """
    )
    conn.commit()
    conn.close()


def test_search_matches_company_name(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed(cfg)
    response = client.get("/api/search", params={"q": "boyd"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "boyd"
    tickers = [r["ticker"] for r in payload["results"]]
    assert tickers == ["BGSI", "BYD"]
    byd = next(r for r in payload["results"] if r["ticker"] == "BYD")
    assert byd["name"] == "BOYD GAMING CORP"
    assert byd["sector"] == "consumer_gaming"
    assert byd["covered"] is True
    assert byd["presented_status"] == "ACTIVE"


def test_search_exact_ticker_outranks_prefix_and_name(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed(cfg)
    response = client.get("/api/search", params={"q": "BY"})
    assert response.status_code == 200
    tickers = [r["ticker"] for r in response.json()["results"]]
    # Exact ticker match first — even uncovered — then ticker-prefix matches.
    assert tickers == ["BY", "BYD"]


def test_search_census_only_row_is_uncovered(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed(cfg)
    response = client.get("/api/search", params={"q": "byline"})
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["ticker"] == "BY"
    assert results[0]["covered"] is False
    assert results[0]["presented_status"] is None


def test_search_facts_only_ticker_is_covered(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed(cfg)
    response = client.get("/api/search", params={"q": "BGSI"})
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["covered"] is True
    assert results[0]["sector"] == "AUTO REPAIR SERVICES"


def test_search_includes_watchlist_ticker_missing_from_census(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed(cfg)
    response = client.get("/api/search", params={"q": "zzoff"})
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["ticker"] == "ZZOFF"
    assert results[0]["name"] is None
    assert results[0]["sector"] == "industrial_tech"
    assert results[0]["covered"] is True


def test_search_escapes_like_wildcards(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed(cfg)
    response = client.get("/api/search", params={"q": "%"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_respects_limit(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed(cfg)
    response = client.get("/api/search", params={"q": "b", "limit": 2})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_search_rejects_empty_query(monkeypatch, tmp_path):
    _init_temp_env(monkeypatch, tmp_path)
    response = client.get("/api/search", params={"q": ""})
    assert response.status_code == 422
