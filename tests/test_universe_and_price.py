from __future__ import annotations

import json

from app.config import get_config
from app.db import get_db, init_db
from app.universe.universe import compute_universe_snapshot_hash, load_universe_to_db
from app.valuation.price_provider import DisabledPriceProvider


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_universe_snapshot_hash_and_persistence(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    rows = [
        {
            "ticker": "AAPL",
            "cik": "320193",
            "name": "Apple Inc.",
            "ir_rss_url": "",
            "homepage_url": "https://www.apple.com",
            "notes": "",
        },
        {
            "ticker": "MSFT",
            "cik": "789019",
            "name": "Microsoft Corporation",
            "ir_rss_url": "https://news.microsoft.com/feed/",
            "homepage_url": "https://www.microsoft.com",
            "notes": "feed",
        },
    ]

    expected_hash = compute_universe_snapshot_hash(rows)
    with get_db() as conn:
        universe_id, snapshot_hash, count = load_universe_to_db(conn, rows, cfg.universe_path)
        assert count == 2
        assert snapshot_hash == expected_hash
        state = conn.execute("SELECT value_json FROM state WHERE key='active_universe'").fetchone()
        payload = json.loads(state["value_json"])
        assert payload["universe_id"] == universe_id
        row = conn.execute(
            """
            SELECT ir_rss_url, homepage_url, notes
            FROM universe_members
            WHERE universe_id = ? AND ticker = 'MSFT'
            """,
            (universe_id,),
        ).fetchone()
        assert row["ir_rss_url"] == "https://news.microsoft.com/feed/"
        assert row["homepage_url"] == "https://www.microsoft.com"
        assert row["notes"] == "feed"


def test_disabled_price_provider_caches_unknown_quote(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    provider = DisabledPriceProvider(get_config())
    quote = provider.get_quote("AAPL", "2026-02-13")

    assert quote.status == "UNKNOWN"
    assert quote.price is None

    with get_db() as conn:
        row = conn.execute(
            "SELECT status, provider FROM price_quotes WHERE ticker='AAPL' AND as_of_date='2026-02-13'"
        ).fetchone()
    assert row is not None
    assert row["status"] == "UNKNOWN"
    assert row["provider"] == "disabled"
