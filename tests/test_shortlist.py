from __future__ import annotations

import json

from app.db import get_db, init_db, utc_now_iso
from app.delta.shortlist import build_shortlist


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


def test_shortlist_filters_and_ordering(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    run_id = "run_shortlist"

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision, reasons_json, created_at
            ) VALUES
            ('AAPL', '2026-02-13', ?, '{}', 90, 'Watchlist', '[]', ?),
            ('MSFT', '2026-02-13', ?, '{}', 85, 'Watchlist', '[]', ?),
            ('GOOG', '2026-02-13', ?, '{}', 80, 'Watchlist', '[]', ?)
            """,
            (run_id, now, run_id, now, run_id, now),
        )
        conn.execute(
            """
            INSERT INTO research_signals(
                ticker, as_of_date, run_id, recency_days_min, item_count_30d, has_earnings_release, has_investor_presentation,
                sentiment_flags_json, key_topics_json, evidence_item_ids_json, summary_json, created_at
            ) VALUES
            ('AAPL', '2026-02-13', ?, 10, 4, 0, 0, '[]', '[]', '[]', '{}', ?),
            ('MSFT', '2026-02-13', ?, 220, 1, 0, 0, '[]', '[]', '[]', '{}', ?)
            """,
            (run_id, now, run_id, now),
        )
        conn.execute(
            """
            INSERT INTO ticker_deltas(ticker, run_id, prev_run_id, as_of_date, changed, delta_path, delta_hash, created_at)
            VALUES
            ('AAPL', ?, 'run_prev', '2026-02-13', 1, '/tmp/AAPL_delta.json', 'h1', ?),
            ('MSFT', ?, 'run_prev', '2026-02-13', 0, '/tmp/MSFT_delta.json', 'h2', ?),
            ('GOOG', ?, 'run_prev', '2026-02-13', 1, '/tmp/GOOG_delta.json', 'h3', ?)
            """,
            (run_id, now, run_id, now, run_id, now),
        )

    filtered = build_shortlist(
        run_id=run_id,
        top_n=5,
        min_score=82,
        require_recent_research=True,
    )
    assert filtered["count"] == 1
    payload = json.loads((cfg.shortlists_dir / f"shortlist_{run_id}.json").read_text(encoding="utf-8"))
    assert [row["ticker"] for row in payload["rows"]] == ["AAPL"]

    unfiltered = build_shortlist(
        run_id=run_id,
        top_n=3,
        min_score=80,
        require_recent_research=False,
    )
    assert unfiltered["count"] == 3
    payload = json.loads((cfg.shortlists_dir / f"shortlist_{run_id}.json").read_text(encoding="utf-8"))
    assert [row["ticker"] for row in payload["rows"]] == ["AAPL", "GOOG", "MSFT"]

