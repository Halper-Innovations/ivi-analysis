from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import get_db, init_db, utc_now_iso
from app.report.memo_builder import build_top_memos
from app.score.ranker import score_and_rank


runner = CliRunner()


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_score_and_rank_uses_run_id_scope_only(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            ) VALUES
                ('AAPL', '2026-02-12', 'run_old', '{}', 99, 'Watchlist', 0, 0, NULL, '[]', ?),
                ('AAPL', '2026-02-13', 'run_new', '{}', 20, 'Watchlist', 0, 0, NULL, '[]', ?),
                ('MSFT', '2026-02-13', 'run_new', '{}', 50, 'Watchlist', 0, 0, NULL, '[]', ?)
            """,
            (now, now, now),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(
                '789019', 'MSFT', '0000789019-26-000001', '10-Q', '2026-02-01', '2025-12-31',
                'https://www.sec.gov/Archives/edgar/data/789019/doc.htm', NULL, NULL, '2026-02-13', 'parsed', ?, ?
            )
            """,
            (now, now),
        )
        conn.execute(
            """
            INSERT INTO research_signals(
                ticker, as_of_date, run_id, recency_days_min, item_count_30d, has_earnings_release, has_investor_presentation,
                sentiment_flags_json, key_topics_json, evidence_item_ids_json, summary_json, created_at
            ) VALUES(
                'MSFT', '2026-02-13', 'run_new', 12, 3, 0, 0, '[]', '[]', '[]', '{}', ?
            )
            """,
            (now,),
        )

    summary = score_and_rank(
        top_n=1,
        memo_mode="triage",
        run_id="run_new",
        as_of_date="2026-02-13",
        candidate_scope=["AAPL", "MSFT"],
        rescore=False,
    )

    assert summary["selection_mode"] == "run_id"
    assert summary["score_rows_considered"] == 2
    assert summary["candidate_count"] == 1

    with get_db() as conn:
        candidate_rows = conn.execute(
            """
            SELECT ticker, as_of_date
            FROM scores
            WHERE is_candidate = 1 AND candidate_run_id = 'run_new'
            ORDER BY total_score DESC
            """
        ).fetchall()
    assert len(candidate_rows) == 1
    assert candidate_rows[0]["ticker"] == "MSFT"
    assert candidate_rows[0]["as_of_date"] == "2026-02-13"


def test_score_and_rank_run_id_with_as_of_ignores_other_dates_same_run(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            ) VALUES
                ('AAPL', '2026-02-13', 'run_same', '{}', 10, 'Watchlist', 0, 0, NULL, '[]', ?),
                ('AAPL', '2026-02-14', 'run_same', '{}', 95, 'Watchlist', 0, 0, NULL, '[]', ?),
                ('MSFT', '2026-02-13', 'run_same', '{}', 40, 'Watchlist', 0, 0, NULL, '[]', ?)
            """,
            (now, now, now),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES
                ('320193', 'AAPL', '0000320193-26-000011', '10-Q', '2026-02-10', '2025-12-31',
                 'https://www.sec.gov/Archives/edgar/data/320193/doc.htm', NULL, NULL, '2026-02-13', 'parsed', ?, ?),
                ('789019', 'MSFT', '0000789019-26-000011', '10-Q', '2026-02-10', '2025-12-31',
                 'https://www.sec.gov/Archives/edgar/data/789019/doc.htm', NULL, NULL, '2026-02-13', 'parsed', ?, ?)
            """,
            (now, now, now, now),
        )
        conn.execute(
            """
            INSERT INTO research_signals(
                ticker, as_of_date, run_id, recency_days_min, item_count_30d, has_earnings_release, has_investor_presentation,
                sentiment_flags_json, key_topics_json, evidence_item_ids_json, summary_json, created_at
            ) VALUES
                ('AAPL', '2026-02-13', 'run_same', 5, 4, 0, 0, '[]', '[]', '[]', '{}', ?),
                ('MSFT', '2026-02-13', 'run_same', 8, 3, 0, 0, '[]', '[]', '[]', '{}', ?)
            """,
            (now, now),
        )

    summary = score_and_rank(
        top_n=1,
        memo_mode="triage",
        run_id="run_same",
        as_of_date="2026-02-13",
        candidate_scope=["AAPL", "MSFT"],
        rescore=False,
    )

    assert summary["selection_mode"] == "run_id"
    assert summary["score_rows_considered"] == 2
    assert summary["tickers_ranked"] == 2
    assert summary["candidate_count"] == 1

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT ticker, as_of_date, is_candidate, candidate_run_id
            FROM scores
            WHERE run_id = 'run_same'
            ORDER BY ticker, as_of_date
            """
        ).fetchall()

    scoped = {(row["ticker"], row["as_of_date"]): (int(row["is_candidate"]), row["candidate_run_id"]) for row in rows}
    # Run-scoped ranking chooses latest-per-ticker regardless of run_as_of_date.
    assert scoped[("AAPL", "2026-02-14")][0] == 1
    assert scoped[("AAPL", "2026-02-14")][1] == "run_same"
    assert scoped[("AAPL", "2026-02-13")][0] == 0
    assert scoped[("MSFT", "2026-02-13")][0] == 0


def test_build_report_triage_without_run_id_uses_latest(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    (cfg.runs_dir).mkdir(parents=True, exist_ok=True)
    (cfg.runs_dir / "index.json").write_text(
        json.dumps(
            [
                {"run_id": "run_old", "generated_at": "2026-02-13T00:00:00+00:00"},
                {"run_id": "run_new", "generated_at": "2026-02-14T00:00:00+00:00"},
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_build_top_memos(*, top_n: int, memo_mode: str, run_id: str | None):
        captured["top_n"] = top_n
        captured["memo_mode"] = memo_mode
        captured["run_id"] = run_id
        return 0

    monkeypatch.setattr("app.report.memo_builder.build_top_memos", _fake_build_top_memos)

    result = runner.invoke(app, ["build-report", "--memo-mode", "triage", "--top", "5"])
    assert result.exit_code == 0, result.output
    assert "Using latest run_id: run_new" in result.output
    assert captured["memo_mode"] == "triage"
    assert captured["top_n"] == 5
    assert captured["run_id"] == "run_new"


def test_triage_memo_creates_gaps_with_same_run_id_and_packet_missing_flag(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            ) VALUES('AAPL', '2026-02-13', 'run_gap_test', '{}', 40, 'Watchlist', 1, 0, 'run_gap_test', '[]', ?)
            """,
            (utc_now_iso(),),
        )

    built = build_top_memos(top_n=1, memo_mode="triage", run_id="run_gap_test")
    assert built == 1

    memo_payload = json.loads((cfg.memos_dir / "AAPL_2026-02-13" / "memo.json").read_text(encoding="utf-8"))
    assert "PACKET_MISSING" in memo_payload.get("status_flags", [])

    gaps_path = cfg.gaps_dir / "AAPL_run_gap_test.json"
    assert gaps_path.exists()
    gaps_payload = json.loads(gaps_path.read_text(encoding="utf-8"))
    assert gaps_payload["run_id"] == "run_gap_test"
    assert "PACKET_MISSING" in gaps_payload.get("status_flags", [])
