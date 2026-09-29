from __future__ import annotations

from pathlib import Path

from app.db import get_db, init_db
from app.universe.universe import (
    export_universe_snapshot,
    load_universe_to_db,
    merge_universe_rows,
    read_base_universe_csv,
    read_metadata_overrides_csv,
)


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


def test_universe_merge_load_and_export(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    base = tmp_path / "base.csv"
    overrides = tmp_path / "overrides.csv"
    base.write_text(
        "ticker,cik,name\nAAPL,320193,Apple Inc.\nMSFT,789019,Microsoft Corporation\n",
        encoding="utf-8",
    )
    overrides.write_text(
        "ticker,homepage_url,ir_rss_url,allowlist_domains,notes\n"
        "AAPL,https://www.apple.com,,*.apple.com,apple note\n"
        "MSFT,https://www.microsoft.com,https://news.microsoft.com/feed/,news.microsoft.com,msft note\n",
        encoding="utf-8",
    )

    merged = merge_universe_rows(
        read_base_universe_csv(base),
        read_metadata_overrides_csv(overrides),
    )
    with get_db() as conn:
        universe_id, _, count = load_universe_to_db(conn, merged, Path("merged.csv"), set_active=True)
        assert count == 2
        row = conn.execute(
            """
            SELECT homepage_url, ir_rss_url, allowlist_domains, notes
            FROM universe_members
            WHERE universe_id = ? AND ticker = 'AAPL'
            """,
            (universe_id,),
        ).fetchone()
        assert row["homepage_url"] == "https://www.apple.com"
        assert row["allowlist_domains"] == "*.apple.com"

        out = tmp_path / "snapshot_export.csv"
        export_universe_snapshot(conn, universe_id, out)

    payload = out.read_text(encoding="utf-8")
    assert "allowlist_domains" in payload
    assert "AAPL,320193,Apple Inc.,https://www.apple.com,,*.apple.com,apple note" in payload
    assert "MSFT,789019,Microsoft Corporation,https://www.microsoft.com,https://news.microsoft.com/feed/,news.microsoft.com,msft note" in payload
    assert cfg.db_path.exists()

