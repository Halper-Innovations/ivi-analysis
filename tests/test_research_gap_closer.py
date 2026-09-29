from __future__ import annotations

import json
from pathlib import Path

from app.db import get_db, init_db, utc_now_iso
from app.research.engine import run_research_gap_closer


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_research_close_gaps_processes_subset(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, '1', ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET name=excluded.name
                """,
                (ticker, ticker, utc_now_iso()),
            )
        conn.execute(
            """
            INSERT INTO research_signals(
                ticker, as_of_date, run_id, recency_days_min, item_count_30d,
                has_earnings_release, has_investor_presentation,
                sentiment_flags_json, key_topics_json, evidence_item_ids_json, summary_json, created_at
            ) VALUES
            ('AAPL', '2026-02-13', 'run_test', 250, 0, 0, 0, '[]', '[]', '[]', '{}', ?),
            ('MSFT', '2026-02-13', 'run_test', 10, 4, 1, 1, '[]', '[]', '[]', '{}', ?),
            ('GOOGL', '2026-02-13', 'run_test', 12, 3, 0, 0, '[]', '[]', '[]', '{}', ?)
            """,
            (utc_now_iso(), utc_now_iso(), utc_now_iso()),
        )

    cfg.gaps_dir.mkdir(parents=True, exist_ok=True)
    (cfg.gaps_dir / "MSFT_run_test.json").write_text(
        json.dumps(
            {
                "ticker": "MSFT",
                "run_id": "run_test",
                "research_warnings": ["Research packet missing for this run/ticker."],
                "missing_research_sources": ["company_news"],
            }
        ),
        encoding="utf-8",
    )

    processed: list[str] = []

    def _fake_run(ticker: str, **kwargs):
        processed.append(ticker)
        return Path(cfg.research_dir / f"{ticker}_{kwargs.get('run_id')}.json")

    rescored: list[str] = []

    def _fake_score(ticker: str, **kwargs):
        rescored.append(ticker)
        return True

    monkeypatch.setattr("app.research.engine.run_research_agent_for_ticker", _fake_run)
    monkeypatch.setattr("app.research.engine.score_ticker", _fake_score)

    summary = run_research_gap_closer(
        as_of_date="2026-02-13",
        run_id="run_test",
        source_filters={"news", "exhibits"},
    )

    assert summary["processed_count"] == 2
    assert sorted(processed) == ["AAPL", "MSFT"]
    assert sorted(rescored) == ["AAPL", "MSFT"]
