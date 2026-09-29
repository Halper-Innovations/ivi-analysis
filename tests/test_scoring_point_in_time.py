"""Point-in-time bounds for the ranker, the research-scope ranking and evidence age
(the analyst decision and previous-run helpers, plus edge cases around the
surrounding scoring fixes and the calendar-day as-of date). Hermetic: temp DB via ``init_db``, no network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.db import get_db, init_db, utc_now_iso
from app.research.engine import _top_ranked_tickers
from app.research.signals import _age_in_days
from app.score import ranker

TICKER = "TEST"


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


# ── the analyst decision is bounded to the scoring date ─────────────


def test_analyst_decision_lookup_ignores_decisions_written_after_the_scoring_date(
    monkeypatch, tmp_path
):
    """A score dated 2025-03-31 is classified by the 2025-02-15 decision (LONG), not the
    2025-09-15 one (SHORT); a date before every decision has no decision at all; the
    unbounded call still returns the newest."""
    _init_temp_db(monkeypatch, tmp_path)
    import app.analyst.output_store as store

    monkeypatch.setattr(store, "financial_integrity_manifest_is_usable", lambda: True)
    monkeypatch.setattr(
        store,
        "authorized_artifact_bytes",
        lambda path: (store.FINANCIAL_INTEGRITY_PASS, Path(path).read_bytes()),
    )
    with get_db() as conn:
        for as_of, classification in (("2025-02-15", "LONG"), ("2025-09-15", "SHORT")):
            out = tmp_path / f"decision_{as_of}.json"
            out.write_text(json.dumps({"decision": {"classification": classification}}))
            conn.execute(
                "INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path,"
                " output_hash, created_at) VALUES(?, ?, 'decision', ?, 'h', ?)",
                (TICKER, as_of, str(out), utc_now_iso()),
            )
        assert ranker._latest_analyst_decision(conn, TICKER, "2025-03-31") == {
            "classification": "LONG"
        }
        assert ranker._latest_analyst_decision(conn, TICKER, "2025-01-01") is None
        assert ranker._latest_analyst_decision(conn, TICKER) == {"classification": "SHORT"}


# ── previous-run helpers order by the business date, not created_at ──


def _insert_coverage(conn, *, run_id: str, as_of: str, created_at: str, accessions: list[str]):
    conn.execute(
        "INSERT INTO filing_coverage(ticker, run_id, as_of_date, forms_included_json,"
        " accession_numbers_json, coverage_score, missing_required_json, created_at)"
        " VALUES(?, ?, ?, '[]', ?, 100, '[]', ?)",
        (TICKER, run_id, as_of, json.dumps(accessions), created_at),
    )


def _insert_signals(conn, *, run_id: str, as_of: str, created_at: str, flags: list[str]):
    conn.execute(
        "INSERT INTO research_signals(ticker, as_of_date, run_id, recency_days_min,"
        " item_count_30d, sentiment_flags_json, created_at) VALUES(?, ?, ?, 5, 1, ?, ?)",
        (TICKER, as_of, run_id, json.dumps(flags), created_at),
    )


def test_previous_coverage_accessions_come_from_the_latest_date_not_the_latest_write(
    monkeypatch, tmp_path
):
    """The 2025-06-30 row (written 2025-06-30) is the previous coverage; a December row
    backfilled on 2025-07-15 has the newer created_at but the older business date."""
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_coverage(
            conn, run_id="jun", as_of="2025-06-30", created_at="2025-06-30T12:00:00Z",
            accessions=["A-JUN"],
        )
        _insert_coverage(
            conn, run_id="backfill", as_of="2024-12-31", created_at="2025-07-15T12:00:00Z",
            accessions=["A-DEC"],
        )
        got = ranker._previous_filing_coverage_accessions(conn, TICKER, "2025-06-30", "current")
    assert got == {"A-JUN"}


def test_previous_signal_flags_come_from_the_latest_date_not_the_latest_write(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_signals(
            conn, run_id="jun", as_of="2025-06-30", created_at="2025-06-30T12:00:00Z",
            flags=["guidance_lowered"],
        )
        _insert_signals(
            conn, run_id="backfill", as_of="2024-12-31", created_at="2025-07-15T12:00:00Z",
            flags=["bankruptcy"],
        )
        got = ranker._latest_previous_signal_flags(conn, TICKER, "2025-06-30", "current")
    assert got == {"guidance_lowered"}


def test_previous_run_helpers_never_read_rows_dated_after_the_scoring_date(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_coverage(
            conn, run_id="later", as_of="2025-09-30", created_at="2025-09-30T12:00:00Z",
            accessions=["A-SEP"],
        )
        _insert_signals(
            conn, run_id="later", as_of="2025-09-30", created_at="2025-09-30T12:00:00Z",
            flags=["bankruptcy"],
        )
        assert ranker._previous_filing_coverage_accessions(
            conn, TICKER, "2025-06-30", "current"
        ) == set()
        assert ranker._latest_previous_signal_flags(conn, TICKER, "2025-06-30", "current") == set()


def test_a_filing_made_after_the_scoring_date_is_not_a_new_recent_filing(monkeypatch, tmp_path):
    """Coverage lists a filing filed 2025-07-15 for a 2025-06-30 score: 15 days AFTER the
    date, which the old ``(as_of - filed).days <= 30`` counted as recent (-15 <= 30)."""
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        for accession, filed in (("A-FUTURE", "2025-07-15"), ("A-RECENT", "2025-06-20")):
            conn.execute(
                "INSERT INTO filings(cik, ticker, accession, form_type, filing_date,"
                " primary_doc_url, status, created_at, updated_at)"
                " VALUES('0', ?, ?, '8-K', ?, 'https://www.sec.gov/x', 'parsed', ?, ?)",
                (TICKER, accession, filed, now, now),
            )
        ctx = ranker._change_context(
            conn,
            ticker=TICKER,
            as_of_date="2025-06-30",
            run_id="current",
            filing_coverage={"accession_numbers": ["A-FUTURE", "A-RECENT"]},
            research_signals=None,
        )
    assert ctx["new_filing_recent_count"] == 1


# ── edge cases ───────────────────────────────────────────────


def _insert_score(conn, ticker: str, as_of: str, total: float, created_at: str | None = None):
    conn.execute(
        "INSERT INTO scores(ticker, as_of_date, run_id, subscores_json, total_score, decision,"
        " reasons_json, created_at) VALUES(?, ?, 'r', '{}', ?, 'Watchlist', '[]', ?)",
        (ticker, as_of, total, created_at or utc_now_iso()),
    )


def test_top_ranked_ignores_scores_dated_after_the_as_of_date_and_breaks_ties_by_ticker(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_score(conn, "ZZZ", "2024-06-30", 50.0)
        _insert_score(conn, "AAA", "2024-06-30", 50.0)
        _insert_score(conn, "FUT", "2025-12-31", 99.0)  # first scored after the as-of date
        _insert_score(conn, "ZZZ", "2025-12-31", 99.0)  # ZZZ's later score is not visible either
        assert _top_ranked_tickers(conn, top_n=10, as_of_date="2024-06-30") == ["AAA", "ZZZ"]


def test_top_ranked_without_a_date_uses_each_tickers_newest_score_once(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        _insert_score(conn, "AAA", "2024-01-31", 90.0)
        _insert_score(conn, "AAA", "2024-06-30", 40.0)
        _insert_score(conn, "BBB", "2024-06-30", 70.0)
        assert _top_ranked_tickers(conn, top_n=10, as_of_date=None) == ["BBB", "AAA"]


# ── the as-of date is a calendar day ───────────────────────────────────


def _midnight(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def test_an_item_published_any_time_on_the_asof_day_is_zero_days_old():
    as_of = _midnight("2024-06-30")
    assert _age_in_days("2024-06-30T00:00:00Z", None, as_of) == 0
    assert _age_in_days("2024-06-30T12:00:00Z", None, as_of) == 0
    assert _age_in_days("2024-06-30T23:59:59Z", None, as_of) == 0


def test_an_item_published_the_next_day_or_undated_carries_no_age():
    as_of = _midnight("2024-06-30")
    assert _age_in_days("2024-07-01T00:00:00Z", None, as_of) is None
    assert _age_in_days("2025-01-15T00:00:00Z", None, as_of) is None
    assert _age_in_days(None, "2024-06-30T00:00:00Z", as_of) is None


def test_ages_before_the_asof_day_are_unchanged():
    as_of = _midnight("2024-06-30")
    assert _age_in_days("2024-06-29T18:00:00Z", None, as_of) == 0
    assert _age_in_days("2024-06-29T00:00:00Z", None, as_of) == 1
    assert _age_in_days("2023-12-01T00:00:00+00:00", None, as_of) == 212
