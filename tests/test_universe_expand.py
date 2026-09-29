from __future__ import annotations

import csv


def test_load_russell_3000_tickers_normalizes_class_share_symbols(monkeypatch, tmp_path):
    from app.universe.expand import _load_russell_3000_tickers

    path = tmp_path / "russell_3000.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "name", "weight", "sector_hint"])
        writer.writeheader()
        writer.writerow({"ticker": "BRKB", "name": "Berkshire Hathaway Class B", "weight": "1.24", "sector_hint": "Financials"})
        writer.writerow({"ticker": "ABC", "name": "ABC Corp", "weight": "0.01", "sector_hint": "Health Care"})
        writer.writerow({"ticker": "ABC", "name": "ABC Corp duplicate", "weight": "0.01", "sector_hint": "Health Care"})

    monkeypatch.setattr("app.universe.ticker_cik_map.load_ticker_cik_map", lambda: {"BRK-B": "1067983", "ABC": "123456"})

    assert _load_russell_3000_tickers(path) == ["BRK-B", "ABC"]


def test_load_russell_3000_tickers_preserves_unresolved_symbol(monkeypatch, tmp_path):
    from app.universe.expand import _load_russell_3000_tickers

    path = tmp_path / "russell_3000.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "name", "weight", "sector_hint"])
        writer.writeheader()
        writer.writerow({"ticker": "MISSING", "name": "Missing Inc", "weight": "0.01", "sector_hint": "Industrials"})

    monkeypatch.setattr("app.universe.ticker_cik_map.load_ticker_cik_map", lambda: {})

    assert _load_russell_3000_tickers(path) == ["MISSING"]
