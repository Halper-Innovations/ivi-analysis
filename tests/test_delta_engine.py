from __future__ import annotations

import json

from app.db import get_db, init_db, utc_now_iso
from app.delta.engine import build_deltas


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


def test_delta_computation_detects_changes(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    packet_path = cfg.evidence_dir / "AAPL_2026-02-13.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "as_of_date": "2026-02-13",
                "financials": [
                    {
                        "line_item": "revenue",
                        "citation": {
                            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/doc.htm",
                            "snippet": "revenue snippet",
                            "section_label": "financials",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scores(ticker, as_of_date, run_id, subscores_json, total_score, decision, reasons_json, created_at)
            VALUES
            ('AAPL', '2026-02-10', 'run_a', '{}', 50, 'Watchlist', '[]', ?),
            ('AAPL', '2026-02-13', 'run_b', '{}', 60, 'Watchlist', '[]', ?)
            """,
            (utc_now_iso(), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO filing_coverage(
              ticker, run_id, as_of_date, forms_included_json, accession_numbers_json, coverage_score, missing_required_json, created_at
            ) VALUES
            ('AAPL', 'run_a', '2026-02-10', '["10-K","10-Q"]', '["0001"]', 100, '[]', ?),
            ('AAPL', 'run_b', '2026-02-13', '["10-K","10-Q","8-K"]', '["0001","0002"]', 100, '[]', ?)
            """,
            (utc_now_iso(), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO filings(
              cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url, status, created_at, updated_at
            ) VALUES(
              '320193', 'AAPL', '0002', '8-K', '2026-02-12', '2025-12-31',
              'https://www.sec.gov/Archives/edgar/data/320193/8k.htm', 'parsed', ?, ?
            )
            """,
            (utc_now_iso(), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES
            ('AAPL', '2026-02-10', '{"revenue": 1000, "fcf": 100, "net_debt": 10, "operating_margin": 0.2}', '{}', ?),
            ('AAPL', '2026-02-13', '{"revenue": 1100, "fcf": 120, "net_debt": 5, "operating_margin": 0.25}', '{}', ?)
            """,
            (utc_now_iso(), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO research_signals(
              ticker, as_of_date, run_id, recency_days_min, item_count_30d, has_earnings_release, has_investor_presentation,
              sentiment_flags_json, key_topics_json, evidence_item_ids_json, summary_json, created_at
            ) VALUES
            ('AAPL', '2026-02-10', 'run_a', 20, 2, 0, 0, '["guidance_raised"]', '[]', '[]', '{}', ?),
            ('AAPL', '2026-02-13', 'run_b', 5, 5, 1, 0, '["guidance_raised","investigation"]', '[]', '[]', '{}', ?)
            """,
            (utc_now_iso(), utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO evidence_packets(ticker, as_of_date, packet_path, packet_hash, created_at)
            VALUES('AAPL', '2026-02-13', ?, 'h', ?)
            """,
            (str(packet_path), utc_now_iso()),
        )

    summary = build_deltas(run_id="run_b", prev_run_id="run_a")
    assert summary["deltas_written"] >= 1
    delta_path = cfg.deltas_dir / "AAPL_run_b.json"
    payload = json.loads(delta_path.read_text(encoding="utf-8"))
    assert payload["changed"] is True
    assert payload["new_filings"]
    assert "investigation" in payload["research_signal_changes"]["added_flags"]
