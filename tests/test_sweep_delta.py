from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.autonomous.sweep_delta import (
    CANONICAL_SWEEP_SECTORS,
    V1_ATOMIC_BANDS,
    backfill_loaded_sets_from_artifacts,
    band_delta_report,
    carried_verdicts,
    full_universe_delta_report,
    record_loaded_set,
    record_unknown_cap_cross_band_coverage,
    split_unswept,
    swept_tickers_for_band,
    v1_terminal_coverage_from_artifact,
)
from app.cli import app
from app.config import get_config
from app.db import get_db, init_db

runner = CliRunner()


def test_loaded_set_migration_preserves_legacy_rows(tmp_path):
    import sqlite3

    from app.db import _migrate_sector_run_loaded_sets

    conn = sqlite3.connect(tmp_path / "legacy.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE sector_run_loaded_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            sector TEXT NOT NULL,
            market_cap_focus TEXT NOT NULL,
            source TEXT NOT NULL,
            ticker TEXT NOT NULL,
            loaded_at TEXT NOT NULL,
            UNIQUE(run_id, ticker)
        )
        """
    )
    conn.execute(
        "INSERT INTO sector_run_loaded_sets(run_id, sector, market_cap_focus, "
        "source, ticker, loaded_at) VALUES('r1', 'energy', 'micro_cap', "
        "'sector_scan_db', 'AAA', '2026-07-15T00:00:00Z')"
    )

    _migrate_sector_run_loaded_sets(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(sector_run_loaded_sets)")}
    row = conn.execute(
        "SELECT ticker, pipeline_version, candidate_disposition, coverage_complete "
        "FROM sector_run_loaded_sets"
    ).fetchone()
    conn.close()
    assert columns.issuperset(
        {
            "pipeline_version",
            "candidate_disposition",
            "coverage_campaign_id",
            "coverage_evidence_json",
            "coverage_evidence_sha256",
            "coverage_authority_kind",
            "coverage_authority_path",
            "coverage_authority_sha256",
            "coverage_source_run_id",
            "coverage_complete",
        }
    )
    assert tuple(row) == ("AAA", None, None, 1)


def _init(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    init_db()

    def legacy_ledger_terminal_dispositions(
        run_id: str,
        *,
        pipeline_version: str,
        conn=None,
    ) -> dict[str, str]:
        # These ledger-unit tests predate persisted run artifacts and model
        # their rows as already authorized. Exact artifact/row agreement and
        # forged-category rejection run unshimmed in the Classic contract
        # suite.
        del pipeline_version
        if conn is not None:
            rows = conn.execute(
                """
                SELECT ticker, candidate_disposition
                FROM sector_run_loaded_sets
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
        else:
            with get_db() as opened_conn:
                rows = opened_conn.execute(
                    """
                    SELECT ticker, candidate_disposition
                    FROM sector_run_loaded_sets
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchall()
        return {
            str(row["ticker"]).strip().upper(): str(row["candidate_disposition"] or "")
            .strip()
            .upper()
            for row in rows
            if str(row["ticker"]).strip() and str(row["candidate_disposition"] or "").strip()
        }

    monkeypatch.setattr(
        "app.autonomous.sweep_delta._authorized_terminal_dispositions",
        legacy_ledger_terminal_dispositions,
    )


def _record_v1_completed(
    conn,
    *,
    run_id,
    sector,
    market_cap_focus,
    tickers,
    source="sector_scan_db",
    loaded_at=None,
):
    normalized = [str(ticker).upper() for ticker in tickers]
    return record_loaded_set(
        conn,
        run_id=run_id,
        sector=sector,
        market_cap_focus=market_cap_focus,
        source=source,
        tickers=normalized,
        loaded_at=loaded_at,
        pipeline_version="v1",
        candidate_dispositions={ticker: "LLM_CANDIDATE_REVIEW_COMPLETED" for ticker in normalized},
    )


def test_record_loaded_set_idempotent(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        first = _record_v1_completed(
            conn,
            run_id="run1",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["aaa", "BBB", "AAA"],
        )
        second = _record_v1_completed(
            conn,
            run_id="run1",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA", "BBB"],
        )
        assert first == 2  # AAA deduped within the call
        assert second == 0
        n = conn.execute("SELECT COUNT(*) c FROM sector_run_loaded_sets").fetchone()["c"]
        assert n == 2


def test_swept_tickers_filters_band_and_probe_source(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="r1",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
        )
        _record_v1_completed(
            conn,
            run_id="r2",
            sector="energy",
            market_cap_focus="mid_cap",
            source="sector_scan_db",
            tickers=["BBB"],
        )
        # Probe runs never count as coverage.
        _record_v1_completed(
            conn,
            run_id="r3",
            sector="payments_fintech",
            market_cap_focus="micro_cap",
            source="explicit_tickers",
            tickers=["RSSS"],
        )
        record_unknown_cap_cross_band_coverage(
            conn,
            run_id="r3",
            sector="payments_fintech",
            source="explicit_tickers",
            primary_band="micro_cap",
            unknown_cap_tickers=["RSSS"],
            candidate_dispositions={"RSSS": "LLM_CANDIDATE_REVIEW_COMPLETED"},
        )
        assert swept_tickers_for_band(conn, "micro_cap") == {"AAA"}
        assert swept_tickers_for_band(conn, "MID_CAP") == {"BBB"}


def _seed_outcome(conn, ticker, run_id, grade, updated_at):
    conn.execute(
        """
        INSERT INTO ticker_outcomes(
            ticker, as_of_date, run_id, decision, conviction, horizon_days,
            thesis_tags_json, outcome_status, grade, created_at, updated_at)
        VALUES(?, '2026-06-11', ?, 'PASS', 2, 365, '[]', 'OPEN', ?, ?, ?)
        """,
        (ticker, run_id, grade, updated_at, updated_at),
    )


def test_carried_verdicts_latest_autonomous_grade(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_outcome(
            conn, "RSSS", "autonomous_sector_payments_fintech_x", "AVOID", "2026-06-11T01:00:00Z"
        )
        # Backtest rows never carry.
        _seed_outcome(conn, "QLYS", "h1_postingest", "AVOID", "2026-06-11T01:00:00Z")
        # Later autonomous verdict wins.
        _seed_outcome(conn, "CCC", "autonomous_sector_energy_a", "AVOID", "2026-06-01T00:00:00Z")
        _seed_outcome(
            conn, "CCC", "autonomous_sector_energy_b", "WATCHLIST_ONLY", "2026-06-10T00:00:00Z"
        )
        carried = carried_verdicts(conn, ["RSSS", "QLYS", "CCC", "DDD"])
        assert carried["RSSS"]["verdict"] == "AVOID"
        assert "QLYS" not in carried
        assert carried["CCC"]["verdict"] == "WATCHLIST_ONLY"
        assert "DDD" not in carried


def test_carried_verdicts_latest_unaudited_row_clears_older_authorized_row(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    old_run = "autonomous_sector_energy_old"
    unaudited_new_run = "autonomous_sector_energy_new"
    monkeypatch.setattr(
        "app.autonomous.sweep_delta.outcome_row_is_decision_eligible",
        lambda row: str(row["run_id"]) == old_run,
    )
    with get_db() as conn:
        _seed_outcome(conn, "AAA", old_run, "AVOID", "2026-06-01T00:00:00Z")
        _seed_outcome(
            conn,
            "AAA",
            unaudited_new_run,
            "WATCHLIST_ONLY",
            "2026-06-10T00:00:00Z",
        )

        assert carried_verdicts(conn, ["AAA"]) == {}
        split = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["AAA"],
        )
        assert split["carried"] == {}
        assert split["to_review"] == ["AAA"]


def test_split_unswept_exact(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="r1",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
        )
        _seed_outcome(
            conn, "RSSS", "autonomous_sector_payments_fintech_x", "AVOID", "2026-06-11T01:00:00Z"
        )
        split = split_unswept(
            conn, band="micro_cap", selected_tickers=["AAA", "RSSS", "NEW1", "NEW2"]
        )
        assert split["swept"] == ["AAA"]
        assert split["unswept"] == ["RSSS", "NEW1", "NEW2"]
        assert split["carried"]["RSSS"]["verdict"] == "AVOID"
        assert split["to_review"] == ["NEW1", "NEW2"]


def test_split_unswept_is_sector_scoped(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id="payments-run",
            sector="payments_fintech",
            market_cap_focus="large_and_mega",
            source="sector_scan_db",
            tickers=["AAA"],
            pipeline_version="v2",
            candidate_dispositions={"AAA": "UNDERWRITTEN"},
        )

        energy = split_unswept(
            conn,
            band="large_and_mega",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v2",
        )
        payments = split_unswept(
            conn,
            band="large_and_mega",
            sector="payments_fintech",
            selected_tickers=["AAA"],
            pipeline_version="v2",
        )

    assert energy["swept"] == []
    assert energy["to_review"] == ["AAA"]
    assert payments["swept"] == ["AAA"]
    assert payments["to_review"] == []


def test_latest_v2_incomplete_coverage_reopens_name(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="old-v1",
            sector="energy",
            market_cap_focus="large_and_mega",
            source="sector_scan_db",
            tickers=["AAA"],
            loaded_at="2026-07-14T00:00:00Z",
        )
        record_loaded_set(
            conn,
            run_id="new-v2-ready",
            sector="energy",
            market_cap_focus="large_and_mega",
            source="sector_scan_db",
            tickers=["AAA"],
            loaded_at="2026-07-15T00:00:00Z",
            pipeline_version="v2",
            candidate_dispositions={"AAA": "READY_FOR_UNDERWRITING"},
        )

        assert (
            swept_tickers_for_band(conn, "large_and_mega", sector="energy", pipeline_version="v2")
            == set()
        )

        record_loaded_set(
            conn,
            run_id="newest-v2-underwritten",
            sector="energy",
            market_cap_focus="large_and_mega",
            source="sector_scan_db",
            tickers=["AAA"],
            loaded_at="2026-07-16T00:00:00Z",
            pipeline_version="v2",
            candidate_dispositions={"AAA": "UNDERWRITTEN"},
        )

        assert swept_tickers_for_band(
            conn, "large_and_mega", sector="energy", pipeline_version="v2"
        ) == {"AAA"}


def test_v1_coverage_is_sector_scoped_and_ignores_newer_v2_incomplete_row(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id="old-v1",
            sector="payments_fintech",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
            loaded_at="2026-07-14T00:00:00Z",
        )
        record_loaded_set(
            conn,
            run_id="new-v2-ready",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
            loaded_at="2026-07-15T00:00:00Z",
            pipeline_version="v2",
            candidate_dispositions={"AAA": "READY_FOR_UNDERWRITING"},
        )

        v1 = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v1",
        )
        v2 = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v2",
        )

    assert v1["swept"] == []
    assert v1["to_review"] == ["AAA"]
    assert v2["swept"] == []
    assert v2["to_review"] == ["AAA"]


def test_latest_v1_failed_candidate_review_reopens_name(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="old-success",
            sector="energy",
            market_cap_focus="micro_cap",
            tickers=["AAA"],
            loaded_at="2026-07-14T00:00:00Z",
        )
        record_loaded_set(
            conn,
            run_id="new-failure",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
            loaded_at="2026-07-15T00:00:00Z",
            pipeline_version="v1",
            candidate_dispositions={"AAA": "LLM_CANDIDATE_REVIEW_FAILED"},
        )
        # Watchlist population may snapshot a new outcome after the memo
        # fallback.  The explicit same-cell failed-review state still wins.
        _seed_outcome(
            conn,
            "AAA",
            "autonomous_sector_energy_new_failure",
            "WATCHLIST_ONLY",
            "2026-07-15T00:01:00Z",
        )

        assert (
            swept_tickers_for_band(conn, "micro_cap", sector="energy", pipeline_version="v1")
            == set()
        )
        retry = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v1",
        )
        other_band = split_unswept(
            conn,
            band="mid_cap",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v1",
        )

    assert retry["carried"] == {}
    assert retry["to_review"] == ["AAA"]
    assert other_band["carried"]["AAA"]["verdict"] == "WATCHLIST_ONLY"
    assert other_band["to_review"] == []


def test_old_v1_coverage_does_not_suppress_first_v2_sweep(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="old-v1-only",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
        )

        v1 = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v1",
        )
        v2 = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v2",
        )

    assert v1["swept"] == ["AAA"]
    assert v2["swept"] == []
    assert v2["to_review"] == ["AAA"]


def test_latest_v2_screen_outcome_is_not_carried_as_underwriting(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        for run_id, grade, basis, disposition, updated_at in (
            (
                "autonomous_sector_old_underwriting",
                "WATCHLIST_ONLY",
                "UNDERWRITING",
                "UNDERWRITTEN",
                "2026-07-14T00:00:00Z",
            ),
            (
                "autonomous_sector_new_screen",
                "DATA_INCOMPLETE",
                "SCREEN",
                "NEEDS_DATA",
                "2026-07-15T00:00:00Z",
            ),
        ):
            conn.execute(
                """
                INSERT INTO ticker_outcomes(
                    ticker, as_of_date, run_id, decision, conviction,
                    horizon_days, thesis_tags_json, outcome_status, grade,
                    pipeline_version, candidate_disposition, decision_basis,
                    source_sector, created_at, updated_at)
                VALUES(
                    'AAA', '2026-07-16', ?, 'WATCH', 2, 365, '[]', 'OPEN',
                    ?, 'v2', ?, ?, 'energy', ?, ?)
                """,
                (run_id, grade, disposition, basis, updated_at, updated_at),
            )

        carried = carried_verdicts(
            conn,
            ["AAA"],
            pipeline_version="v2",
            sector="energy",
        )

    assert carried == {}


def test_v2_data_incomplete_underwriting_remains_unswept(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO ticker_outcomes(
                ticker, as_of_date, run_id, decision, conviction, horizon_days,
                thesis_tags_json, outcome_status, grade, pipeline_version,
                candidate_disposition, decision_basis, source_sector,
                created_at, updated_at)
            VALUES(
                'AAA', '2026-07-16', 'autonomous_sector_gap', 'WATCH', 2,
                365, '[]', 'OPEN', 'DATA_INCOMPLETE', 'v2', 'NEEDS_DATA',
                'UNDERWRITING', 'energy', '2026-07-16T00:00:00Z',
                '2026-07-16T00:00:00Z')
            """
        )

        split = split_unswept(
            conn,
            band="large_and_mega",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v2",
        )

    assert split["carried"] == {}
    assert split["to_review"] == ["AAA"]


def test_backfill_from_artifacts(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    runs_dir = tmp_path / "runs"
    for run_id, source, focus, loaded, status in (
        ("autonomous_sector_energy_1", "sector_scan_db", "micro_cap", ["AAA", "BBB"], "COMPLETED"),
        ("autonomous_sector_pay_2", "explicit_tickers", "micro_cap", ["RSSS"], "COMPLETED"),
        ("autonomous_sector_energy_3", "sector_scan_db", "micro_cap", ["CCC"], "FAILED"),
    ):
        run_dir = runs_dir / "autonomous_sector" / run_id
        run_dir.mkdir(parents=True)
        memo_candidates = {
            ticker: {
                "ticker": ticker,
                "source": ("deterministic_fallback" if ticker == "BBB" else "llm"),
                "status": "DEGRADED_STATE" if ticker == "BBB" else "OK",
                "thesis": f"{ticker} has a review thesis.",
                "key_risks": [f"{ticker} risk."],
                "falsifiers": [f"{ticker} falsifier."],
                "open_questions": [f"{ticker} question?"],
            }
            for ticker in loaded
        }
        (run_dir / "autonomous_sector_run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": status,
                    "market_cap_focus": focus,
                    "created_at": "2026-06-11T00:00:00Z",
                    "candidate_selection": {
                        "sector": "energy",
                        "market_cap_focus": focus,
                        "source": source,
                        "loaded_tickers": loaded,
                    },
                    "company_packets": [{"ticker": ticker} for ticker in loaded],
                    "memo_body": {"candidates": memo_candidates},
                    "provider_usage": [
                        {
                            "schema_name": f"autonomous_sector_candidate_memo_{ticker.lower()}",
                            "status": "OK",
                        }
                        for ticker in loaded
                        if ticker != "BBB"
                    ],
                }
            )
        )
    # This legacy fixture tests ledger reconstruction, not financial-contract
    # auditing. Invalid/unaudited backfill behavior has dedicated coverage.
    monkeypatch.setattr(
        "app.autonomous.sweep_delta.authorized_artifact_bytes",
        lambda path, _manifest: ("PASS", Path(path).read_bytes()),
    )
    result = backfill_loaded_sets_from_artifacts(runs_dir)
    assert result["artifacts"] == 3
    assert result["runs_recorded"] == 2  # FAILED skipped
    assert result["rows_inserted"] == 3
    # Idempotent.
    again = backfill_loaded_sets_from_artifacts(runs_dir)
    assert again["rows_inserted"] == 0
    with get_db() as conn:
        assert swept_tickers_for_band(conn, "micro_cap", sector="energy") == {"AAA"}
        row = conn.execute(
            "SELECT candidate_disposition, coverage_complete "
            "FROM sector_run_loaded_sets WHERE ticker = 'BBB'"
        ).fetchone()
        assert tuple(row) == ("LLM_CANDIDATE_REVIEW_FAILED", 0)


def test_v1_terminal_coverage_requires_bound_successful_candidate_review():
    artifact = {
        "candidate_selection": {
            "loaded_tickers": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
                "GATE",
                "BROKEN",
                "CARRY",
                "SPARSE",
                "ZERO",
                "GENERIC_DROP",
            ],
            "excluded_tickers": ["GENERIC_DROP"],
            "structural_gate_results": {
                "GATE": {"quarantined": True, "excluded_error": False},
                "BROKEN": {"quarantined": False, "excluded_error": True},
            },
            "delta_audit": {"carried_verdicts": {"CARRY": {"verdict": "AVOID"}}},
            "financial_history_filter": {
                "status": "FILTERED_SPARSE_FINANCIAL_HISTORY",
                "excluded_tickers": ["SPARSE"],
            },
            "framework_evidence_filter": {
                "status": "FILTERED_ZERO_PACKET_SUPPORT",
                "excluded_tickers": ["ZERO"],
            },
        },
        "company_packets": [
            {"ticker": "AAA"},
            {"ticker": "BBB"},
            {"ticker": "CCC"},
            {"ticker": "DDD"},
            {"ticker": "EEE"},
        ],
        "memo_body": {
            "candidates": {
                "AAA": {
                    "ticker": "AAA",
                    "source": "llm",
                    "status": "OK",
                    "thesis": "AAA has a review thesis.",
                    "key_risks": ["AAA risk."],
                    "falsifiers": ["AAA falsifier."],
                    "open_questions": ["AAA question?"],
                },
                "BBB": {
                    "ticker": "BBB",
                    "source": "deterministic_fallback",
                    "status": "DEGRADED_STATE",
                },
                # Looks successful in prose, but no successful provider-use
                # record binds it to the requested candidate.
                "CCC": {
                    "ticker": "CCC",
                    "source": "llm",
                    "status": "OK",
                    "thesis": "CCC has a review thesis.",
                    "key_risks": ["CCC risk."],
                    "falsifiers": ["CCC falsifier."],
                    "open_questions": ["CCC question?"],
                },
                "DDD": {
                    "ticker": "DDD",
                    "source": "llm",
                    "status": "OK",
                    "thesis": "DDD has a review thesis.",
                    "key_risks": ["DDD risk."],
                    "falsifiers": ["DDD falsifier."],
                    "open_questions": ["DDD question?"],
                },
                # Correct identity and provider binding are insufficient when
                # the structured response contains no actual review content.
                "EEE": {
                    "ticker": "EEE",
                    "source": "llm",
                    "status": "OK",
                    "thesis": "",
                    "key_risks": [],
                    "falsifiers": [],
                    "open_questions": [],
                },
                "EXTRA": {
                    "ticker": "EXTRA",
                    "source": "llm",
                    "status": "OK",
                    "thesis": "EXTRA has a review thesis.",
                    "key_risks": ["EXTRA risk."],
                    "falsifiers": ["EXTRA falsifier."],
                    "open_questions": ["EXTRA question?"],
                },
            }
        },
        "provider_usage": [
            {
                "schema_name": "autonomous_sector_candidate_memo_aaa",
                "status": "OK",
            },
            {
                "schema_name": "autonomous_sector_candidate_memo_ddd",
                "status": "OK",
            },
            {
                "schema_name": "autonomous_sector_candidate_memo_eee",
                "status": "OK",
            },
            {
                "schema_name": "autonomous_sector_candidate_memo_extra",
                "status": "OK",
            },
        ],
    }

    projected = v1_terminal_coverage_from_artifact(artifact)

    assert projected["llm_candidate_review_completed"] == ["AAA", "DDD"]
    assert projected["llm_candidate_review_failed"] == ["BBB", "CCC", "EEE"]
    assert projected["terminal_candidate_dispositions"] == {
        "AAA": "LLM_CANDIDATE_REVIEW_COMPLETED",
        "CARRY": "CARRIED_VERDICT",
        "DDD": "LLM_CANDIDATE_REVIEW_COMPLETED",
        "GATE": "STRUCTURAL_SCREENED",
        "SPARSE": "NEEDS_DATA_SPARSE_HISTORY",
        "ZERO": "NEEDS_DATA_FRAMEWORK_EVIDENCE",
    }
    assert projected["candidate_dispositions"]["BBB"] == ("LLM_CANDIDATE_REVIEW_FAILED")
    assert "GENERIC_DROP" not in projected["candidate_dispositions"]
    assert "EXTRA" not in projected["candidate_dispositions"]
    assert "BROKEN" not in projected["candidate_dispositions"]


def test_campaign_candidate_review_rejects_mixed_or_fallback_provider_usage():
    candidate = {
        "ticker": "AAA",
        "source": "llm",
        "status": "OK",
        "thesis": "AAA has a review thesis.",
        "key_risks": ["AAA risk."],
        "falsifiers": ["AAA falsifier."],
        "open_questions": ["AAA question?"],
    }
    base = {
        "candidate_selection": {
            "loaded_tickers": ["AAA"],
            "coverage_expected_provider": "anthropic",
            "coverage_expected_model": "claude-sonnet-test",
        },
        "company_packets": [{"ticker": "AAA"}],
        "memo_body": {"candidates": {"AAA": candidate}},
        "provider_usage": [
            {
                "schema_name": "autonomous_sector_candidate_memo_aaa",
                "status": "OK",
                "provider": "anthropic",
                "model": "claude-sonnet-test",
                "cost_estimate_usd": 0.1,
            }
        ],
    }

    mixed = json.loads(json.dumps(base))
    mixed["provider_usage"].append(
        {
            "schema_name": "parent",
            "status": "OK",
            "provider": "openai",
            "model": "wrong",
            "cost_estimate_usd": 0.1,
        }
    )
    fallback = json.loads(json.dumps(base))
    fallback["provider_usage"][0]["fallback_from_provider"] = "openai"

    assert v1_terminal_coverage_from_artifact(mixed)["llm_candidate_review_completed"] == []
    assert v1_terminal_coverage_from_artifact(fallback)["llm_candidate_review_completed"] == []


class _FakeSelection:
    def __init__(
        self,
        selected,
        *,
        loaded=None,
        structural_gate_results=None,
        warnings=None,
    ):
        self.sector = "energy"
        self.market_cap_focus = "micro_cap"
        self.selected_tickers = list(selected)
        self.loaded_tickers = list(selected if loaded is None else loaded)
        self.source = "sector_scan_db"
        self.requested_tickers = []
        self.excluded_tickers = []
        self.warnings = list(warnings or [])
        self.cap_classifications = {}
        self.structural_gate_results = dict(structural_gate_results or {})

    def to_dict(self):
        return {
            "sector": self.sector,
            "market_cap_focus": self.market_cap_focus,
            "selected_tickers": self.selected_tickers,
            "loaded_tickers": self.loaded_tickers,
            "source": self.source,
            "structural_gate_results": self.structural_gate_results,
        }


def test_band_delta_report(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="r1",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
        )
        _seed_outcome(
            conn, "RSSS", "autonomous_sector_payments_fintech_x", "AVOID", "2026-06-11T01:00:00Z"
        )

    def fake_resolver(**kwargs):
        return _FakeSelection(["AAA", "RSSS", "NEW1"])

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers", fake_resolver
    )
    report = band_delta_report("micro_cap", sectors=["energy"])
    assert report["pipeline_version"] == "v1"
    assert report["sectors"]["energy"] == {
        "source": "sector_scan_db",
        "source_status": "OK",
        "gate_scope": "coverage_uncovered_only",
        "gate_evaluated_tickers": ["RSSS", "NEW1"],
        "loaded": 3,
        "loaded_tickers": ["AAA", "RSSS", "NEW1"],
        "selected": 3,
        "selected_semantics": "previously_covered_or_gate_admitted_uncovered",
        "selected_tickers": ["AAA", "RSSS", "NEW1"],
        "structural_screened": 0,
        "structural_screened_tickers": [],
        "structural_quarantined_tickers": [],
        "gate_errors": 0,
        "gate_error_tickers": [],
        "unknown_cap_tickers": [],
        "covered_loaded": 1,
        "coverage_uncovered": 2,
        "coverage_uncovered_tickers": ["RSSS", "NEW1"],
        "swept": 1,
        "swept_tickers": ["AAA"],
        "carried": {"RSSS": "AVOID"},
        "to_review": 1,
        "to_review_tickers": ["NEW1"],
        "source_errors": [],
        "warnings": [],
    }
    assert report["totals"] == {
        "loaded": 3,
        "selected": 3,
        "structural_screened": 0,
        "gate_errors": 0,
        "covered_loaded": 1,
        "coverage_uncovered": 2,
        "swept": 1,
        "carried": 1,
        "to_review": 1,
    }


def test_band_delta_report_uses_cap_band_v2_rollout(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_AUTONOMOUS_SECTOR_V2_CAP_BANDS", "large_and_mega")
    get_config.cache_clear()
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id="v2-energy",
            sector="energy",
            market_cap_focus="large_and_mega",
            source="sector_scan_db",
            tickers=["AAA"],
            pipeline_version="v2",
            candidate_dispositions={"AAA": "UNDERWRITTEN"},
        )

    resolution_kwargs: list[dict] = []

    def fake_resolver(**kwargs):
        resolution_kwargs.append(kwargs)
        return _FakeSelection(["AAA", "BBB"])

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        fake_resolver,
    )

    report = band_delta_report("large_and_mega", sectors=["energy"])

    assert report["pipeline_version"] == "v2"
    assert resolution_kwargs[0]["require_accepted_census"] is True
    assert resolution_kwargs[0]["accepted_census_authority"] is not None
    assert report["sectors"]["energy"]["swept"] == 1
    assert report["sectors"]["energy"]["to_review_tickers"] == ["BBB"]


def _full_membership(tickers):
    normalized = sorted(tickers)
    return {
        "current_sector_tagged": len(normalized),
        "current_membership_fingerprint": "test-fingerprint",
        "current_membership_tickers": normalized,
        "eligible_common_equity": len(normalized),
        "eligible_common_equity_tickers": normalized,
        "security_type_non_common_equity": 0,
        "security_type_non_common_equity_tickers": [],
        "registry_removed": 0,
        "registry_removed_tickers": [],
        "membership_partition_complete": True,
        "membership_unaccounted_tickers": [],
        "missing_scorecard": 0,
        "missing_scorecard_tickers": [],
    }


def test_current_v1_membership_reconciles_independent_denominator(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        for ticker, as_of_date, sector in (
            ("GOOD", "2026-07-20", "energy"),
            ("NOSC", "2026-07-20", "energy"),
            ("PREF", "2026-07-19", "energy"),
            ("PREF", "2026-07-20", None),
            ("OLDNULL", "2026-07-19", "energy"),
            ("OLDNULL", "2026-07-20", None),
            ("REMOVED", "2026-07-19", "energy"),
            ("REMOVED", "2026-07-20", None),
            ("NEVER", "2026-07-20", None),
        ):
            conn.execute(
                """
                INSERT INTO sector_inference(
                    ticker, as_of_date, inferred_sector, created_at
                ) VALUES(?, ?, ?, '2026-07-20T00:00:00Z')
                """,
                (ticker, as_of_date, sector),
            )
        for ticker in ("GOOD", "PREF"):
            conn.execute(
                """
                INSERT INTO valuations(
                    ticker, as_of_date, method, inputs_json, outputs_json,
                    warnings_json, created_at
                ) VALUES(?, '2026-07-20', 'scorecard', '{}', '{}', '[]',
                         '2026-07-20T00:00:00Z')
                """,
                (ticker,),
            )
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, exchange_scope, operating_status,
                first_seen_at, last_seen_at, removed_at
            ) VALUES(
                '1', 'REMOVED', 'IN_SCOPE', 'NON_OPERATING',
                '2026-07-19T00:00:00Z', '2026-07-20T00:00:00Z',
                '2026-07-20T00:00:00Z'
            )
            """
        )

    from app.autonomous import sector_candidates as candidate_module
    from app.autonomous import sweep_delta as sweep_delta_module

    monkeypatch.setattr(
        candidate_module,
        "_security_filter_for_ticker",
        lambda ticker: candidate_module.SecurityTypeFilterResult(
            ticker=ticker,
            is_common_equity=ticker != "PREF",
            reason=(candidate_module.SECURITY_TYPE_NON_COMMON_EQUITY if ticker == "PREF" else None),
        ),
    )

    membership = sweep_delta_module._current_v1_membership()

    assert membership["all_sector_inference_tickers"] == 6
    assert membership["ever_sector_tagged"] == 5
    assert membership["current_sector_tagged"] == 5
    assert membership["current_membership_tickers"] == [
        "GOOD",
        "NOSC",
        "OLDNULL",
        "PREF",
        "REMOVED",
    ]
    assert membership["latest_null_previously_tagged_tickers"] == [
        "OLDNULL",
        "PREF",
        "REMOVED",
    ]
    assert membership["latest_null_carried_last_valid_sector_tickers"] == [
        "OLDNULL",
        "PREF",
        "REMOVED",
    ]
    assert membership["eligible_common_equity_tickers"] == [
        "GOOD",
        "NOSC",
        "OLDNULL",
    ]
    assert membership["security_type_non_common_equity_tickers"] == ["PREF"]
    assert membership["registry_removed_tickers"] == ["REMOVED"]
    assert membership["missing_scorecard_tickers"] == ["NOSC", "OLDNULL"]
    assert membership["membership_partition_complete"] is True
    assert membership["membership_unaccounted_tickers"] == []


def test_full_universe_plan_enumerates_canonical_atomic_grid(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    calls: list[tuple[str, str, str, object, bool, bool, bool]] = []

    def fake_resolver(**kwargs):
        calls.append(
            (
                kwargs["sector"],
                kwargs["market_cap_focus"],
                kwargs["pipeline_version"],
                kwargs["max_candidates"],
                kwargs["filing_risk_use_llm"],
                kwargs["allow_live_market_data"],
                kwargs["coverage_only"],
            )
        )
        return _FakeSelection([])

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        fake_resolver,
    )
    monkeypatch.setattr(
        "app.autonomous.sweep_delta._current_v1_membership",
        lambda: _full_membership([]),
    )

    report = full_universe_delta_report()

    expected = {
        (sector, band, "v1", None, False, False, True)
        for band in V1_ATOMIC_BANDS
        for sector in CANONICAL_SWEEP_SECTORS
    }
    assert len(calls) == 170
    assert set(calls) == expected
    assert report["expected_cells"] == 170
    assert report["cells_resolved"] == 170
    assert report["atomic_bands"] == [
        "micro_cap",
        "small_cap",
        "mid_cap",
        "large_cap",
        "mega_cap",
    ]
    assert report["status"] == "COMPLETE"


def test_full_universe_completion_rechecks_current_membership(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="old-energy-micro",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["OLD"],
        )

    def fake_resolver(**kwargs):
        if kwargs["sector"] == "energy" and kwargs["market_cap_focus"] == "micro_cap":
            return _FakeSelection(["OLD", "NEW"])
        return _FakeSelection([])

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        fake_resolver,
    )
    monkeypatch.setattr(
        "app.autonomous.sweep_delta._current_v1_membership",
        lambda: _full_membership(["OLD", "NEW"]),
    )

    report = full_universe_delta_report()

    assert report["status"] == "INCOMPLETE"
    assert report["coverage_residual_tickers"] == ["NEW"]
    assert report["loader_residual_tickers"] == []
    assert report["pending_cells"] == [
        {
            "cell_id": "energy:micro_cap",
            "sector": "energy",
            "band": "micro_cap",
            "coverage_uncovered": 1,
            "coverage_uncovered_tickers": ["NEW"],
            "to_review": 1,
            "to_review_tickers": ["NEW"],
            "sector_context_limit": 25,
            "command": (
                ".venv/bin/ivi autonomous-sector-run --sector energy "
                "--market-cap-focus micro_cap --only-unswept --pipeline-version v1"
            ),
        }
    ]

    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="new-energy-micro",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["NEW"],
        )
    closed = full_universe_delta_report()
    assert closed["status"] == "COMPLETE"
    assert closed["coverage_residual_tickers"] == []
    assert closed["pending_cells"] == []


def test_full_universe_reopens_current_band_after_cap_migration(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id="prior-large",
            sector="energy",
            market_cap_focus="large_cap",
            source="sector_scan_db",
            tickers=["MOVER"],
        )

    def fake_resolver(**kwargs):
        if kwargs["sector"] == "energy" and kwargs["market_cap_focus"] == "micro_cap":
            return _FakeSelection(["MOVER"])
        return _FakeSelection([])

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        fake_resolver,
    )
    monkeypatch.setattr(
        "app.autonomous.sweep_delta._current_v1_membership",
        lambda: _full_membership(["MOVER"]),
    )

    report = full_universe_delta_report()

    assert report["coverage_residual_tickers"] == ["MOVER"]
    assert [cell["cell_id"] for cell in report["pending_cells"]] == ["energy:micro_cap"]


def _sector_artifact_stub():
    from tests.test_autonomous_cli import _sector_artifact

    return _sector_artifact()


def test_only_unswept_passes_filtered_tickers_and_records_loaded_set(monkeypatch, tmp_path):
    from app.autonomous import sector_runtime
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        require_financial_integrity_scope,
    )
    from app.autonomous.sector_contract import SectorCompanyFinancialPacket
    from tests.financial_integrity_helpers import canonicalize_financial_packet
    from tests.test_classic_postwrite_authorization import _baseline_manifest

    _init(monkeypatch, tmp_path)
    _baseline_manifest(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="r1",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
        )
        _seed_outcome(
            conn, "RSSS", "autonomous_sector_payments_fintech_x", "AVOID", "2026-06-11T01:00:00Z"
        )
    captured: dict = {}

    def fake_resolver(**kwargs):
        return _FakeSelection(["AAA", "RSSS", "NEW1", "NEW2"])

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        artifact = _sector_artifact_stub()
        artifact.final_verdict = "NO_SELECTION"
        artifact.selected_ticker = None
        artifact.confidence = None
        artifact.no_selection_reason = "One candidate remained incomplete."
        artifact.company_packets = [
            canonicalize_financial_packet(
                SectorCompanyFinancialPacket(
                    ticker=ticker,
                    financial_status="Financially Viable",
                    model_fit_status="VALID_GENERIC",
                    data_quality_status="OK",
                    current_price=50.0,
                    current_price_source="fixture_quote",
                    current_price_source_url=f"https://example.test/quotes/{ticker}",
                    financial_integrity_status="PASS",
                    financial_integrity_violations=[],
                    metric_traces={},
                ),
                as_of_date=artifact.as_of_date,
                shares_mm=10.0,
            )
            for ticker in ("NEW1", "NEW2")
        ]
        scope = FinancialIntegrityScope(
            context="autonomous_sector_pre_provider",
            run_as_of_date=artifact.as_of_date,
            packets=tuple(artifact.company_packets),
        )
        gate = require_financial_integrity_scope(scope)
        artifact.candidate_selection = {
            **dict(kwargs["candidate_selection"]),
            "financial_integrity": gate.to_dict(),
            "financial_integrity_binding": sector_runtime._financial_integrity_run_binding(
                scope,
                scope_fingerprint=gate.scope_fingerprint,
            ),
        }
        artifact.memo_body = {
            "candidates": {
                "NEW1": {
                    "ticker": "NEW1",
                    "source": "llm",
                    "status": "OK",
                    "thesis": "NEW1 has a review thesis.",
                    "key_risks": ["NEW1 risk."],
                    "falsifiers": ["NEW1 falsifier."],
                    "open_questions": ["NEW1 question?"],
                },
                "NEW2": {
                    "ticker": "NEW2",
                    "source": "deterministic_fallback",
                    "status": "DEGRADED_STATE",
                },
            }
        }
        artifact.provider_usage = [
            {
                "schema_name": "autonomous_sector_candidate_memo_new1",
                "status": "OK",
            }
        ]
        return artifact

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers", fake_resolver
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis", fake_runtime
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.enrich_sector_artifact_memo_body",
        lambda artifact: artifact,
    )
    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--no-watchlist",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["tickers"] == ["NEW1", "NEW2"]  # swept + carried excluded
    assert (
        captured["candidate_selection"]["delta_audit"]["carried_verdicts"]["RSSS"]["verdict"]
        == "AVOID"
    )
    payload = json.loads(result.output)
    assert payload["delta_audit"] == {
        "swept_excluded": 1,
        "carried_verdicts": {"RSSS": "AVOID"},
        "to_review": 2,
        "llm_candidate_review_completed": 1,
        "llm_candidate_review_failed": 1,
        "remaining_uncovered": 1,
    }
    assert payload["coverage_accounting"]["remaining_uncovered_tickers"] == ["NEW2"]
    # Exact terminal and retry states replace the old all-loaded credit.
    with get_db() as conn:
        rows = conn.execute(
            "SELECT ticker, candidate_disposition, coverage_complete "
            "FROM sector_run_loaded_sets WHERE run_id != 'r1' ORDER BY ticker"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("NEW1", "LLM_CANDIDATE_REVIEW_COMPLETED", 1),
            ("NEW2", "LLM_CANDIDATE_REVIEW_FAILED", 0),
            ("RSSS", "CARRIED_VERDICT", 1),
        ]
        split = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["NEW1", "NEW2"],
            pipeline_version="v1",
        )
        assert split["swept"] == ["NEW1"]
        assert split["to_review"] == ["NEW2"]


def test_failed_v1_run_records_exact_sparse_history_needs_data(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: _FakeSelection(["SPARSE"]),
    )

    def fake_runtime(**kwargs):
        from tests.test_autonomous_cli import _sector_artifact

        artifact = _sector_artifact(
            verdict="NO_SELECTION",
            selected_ticker=None,
            confidence=None,
            status="FAILED",
            no_selection_reason="No reportable financial history.",
        )
        artifact.candidate_selection = {
            **kwargs["candidate_selection"],
            "financial_history_filter": {
                "status": "FILTERED_SPARSE_FINANCIAL_HISTORY",
                "excluded_tickers": ["SPARSE"],
                "year_counts": {"SPARSE": 1},
            },
        }
        return artifact

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        fake_runtime,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
            "--no-watchlist",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["coverage_accounting"] == {
        "semantics": "failed_run_explicit_terminal_and_retry_states_only",
        "terminal_tickers_recorded": ["SPARSE"],
        "incomplete_tickers_recorded": [],
        "rows_recorded_or_upgraded": 1,
        "remaining_uncovered_tickers": [],
        "rerun_required": False,
    }
    with get_db() as conn:
        row = conn.execute(
            "SELECT candidate_disposition, coverage_complete "
            "FROM sector_run_loaded_sets WHERE ticker = 'SPARSE'"
        ).fetchone()
        assert tuple(row) == ("NEEDS_DATA_SPARSE_HISTORY", 1)
        assert swept_tickers_for_band(
            conn, "micro_cap", sector="energy", pipeline_version="v1"
        ) == {"SPARSE"}


def test_failed_v1_run_keeps_unaccounted_loaded_name_open(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: _FakeSelection(["UNRESOLVED"]),
    )

    def fake_runtime(**kwargs):
        from tests.test_autonomous_cli import _sector_artifact

        artifact = _sector_artifact(
            verdict="NO_SELECTION",
            selected_ticker=None,
            confidence=None,
            status="FAILED",
            no_selection_reason="Packet assembly failed before a terminal state.",
        )
        artifact.candidate_selection = dict(kwargs["candidate_selection"])
        return artifact

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        fake_runtime,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
            "--no-watchlist",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["coverage_accounting"] == {
        "semantics": "failed_run_explicit_terminal_and_retry_states_only",
        "terminal_tickers_recorded": [],
        "incomplete_tickers_recorded": [],
        "rows_recorded_or_upgraded": 0,
        "remaining_uncovered_tickers": ["UNRESOLVED"],
        "rerun_required": True,
    }
    with get_db() as conn:
        assert (
            swept_tickers_for_band(conn, "micro_cap", sector="energy", pipeline_version="v1")
            == set()
        )


def test_failed_v1_run_cannot_close_unpersisted_candidate_review(monkeypatch, tmp_path):
    from app.autonomous.sector_contract import SectorCompanyFinancialPacket

    _init(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: _FakeSelection(["UNAUDITED"]),
    )

    def fake_runtime(**kwargs):
        from tests.test_autonomous_cli import _sector_artifact

        artifact = _sector_artifact(
            verdict="NO_SELECTION",
            selected_ticker=None,
            confidence=None,
            status="FAILED",
            no_selection_reason="Sector decision failed after candidate review.",
        )
        artifact.candidate_selection = dict(kwargs["candidate_selection"])
        artifact.company_packets = [
            SectorCompanyFinancialPacket(
                ticker="UNAUDITED",
                financial_status="Financially Viable",
                model_fit_status="VALID_GENERIC",
                data_quality_status="OK",
            )
        ]
        artifact.memo_body = {
            "candidates": {
                "UNAUDITED": {
                    "ticker": "UNAUDITED",
                    "source": "llm",
                    "status": "OK",
                    "thesis": "UNAUDITED has an in-memory review thesis.",
                    "key_risks": ["UNAUDITED risk."],
                    "falsifiers": ["UNAUDITED falsifier."],
                    "open_questions": ["UNAUDITED question?"],
                }
            }
        }
        artifact.provider_usage = [
            {
                "schema_name": "autonomous_sector_candidate_memo_unaudited",
                "status": "OK",
            }
        ]
        return artifact

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        fake_runtime,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.enrich_sector_artifact_memo_body",
        lambda artifact: artifact,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
            "--no-watchlist",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["artifact_path"] is None
    assert payload["coverage_accounting"] == {
        "semantics": "failed_run_explicit_terminal_and_retry_states_only",
        "terminal_tickers_recorded": [],
        "incomplete_tickers_recorded": ["UNAUDITED"],
        "rows_recorded_or_upgraded": 1,
        "remaining_uncovered_tickers": ["UNAUDITED"],
        "rerun_required": True,
    }
    with get_db() as conn:
        row = conn.execute(
            "SELECT candidate_disposition, coverage_complete "
            "FROM sector_run_loaded_sets WHERE ticker = 'UNAUDITED'"
        ).fetchone()
        assert tuple(row) == ("LLM_CANDIDATE_REVIEW_UNAUDITABLE", 0)
        assert (
            swept_tickers_for_band(conn, "micro_cap", sector="energy", pipeline_version="v1")
            == set()
        )


def test_only_unswept_empty_delta_zero_cost(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _record_v1_completed(
            conn,
            run_id="r1",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA", "BBB"],
        )

    def fake_resolver(**kwargs):
        return _FakeSelection(["AAA", "BBB"])

    def fail_runtime(**kwargs):
        raise AssertionError("empty delta must never reach the LLM runtime")

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers", fake_resolver
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis", fail_runtime
    )
    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "EMPTY_DELTA"
    assert payload["to_review"] == 0
    assert payload["llm_cost"] == {"cumulative_cost_usd": 0.0, "call_count": 0}


def test_only_unswept_source_failure_is_incomplete_not_empty(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: _FakeSelection(
            [],
            warnings=["SECTOR_CANDIDATE_SOURCE_FAILED:OperationalError:boom"],
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("source failure must never reach the LLM runtime")
        ),
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
            "--no-watchlist",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload == {
        "scan_family": "normal",
        "pipeline_version": "v1",
        "sector": "energy",
        "market_cap_focus": "micro_cap",
        "status": "INCOMPLETE_DELTA",
        "loaded": 0,
        "selected": 0,
        "source_errors": ["SECTOR_CANDIDATE_SOURCE_FAILED:OperationalError:boom"],
        "coverage_unresolved": "candidate_source_failed",
        "llm_cost": {"cumulative_cost_usd": 0.0, "call_count": 0},
        "provider_usage_attestation": {
            "valid": True,
            "physical_attempt_count": 0,
            "cost_estimate_usd": 0.0,
            "provider_models": [],
            "usage_records_sha256": (
                "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
            ),
        },
        "provider_usage_incremental_attestation": {
            "valid": True,
            "physical_attempt_count": 0,
            "cost_estimate_usd": 0.0,
            "provider_models": [],
            "usage_records_sha256": (
                "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
            ),
        },
    }


def test_strict_cost_exhaustion_emits_auditable_resumable_v1_envelope(monkeypatch, tmp_path):
    from app.llm.providers.retry_guard import LLMCostBudgetExceeded

    _init(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: _FakeSelection(["AAA"]),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(
            LLMCostBudgetExceeded("projected call exceeds strict cap")
        ),
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
            "--strict-cost-cap",
            "--max-cost-usd",
            "0.01",
            "--no-watchlist",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "FAILED"
    assert payload["provider_usage_attestation"]["valid"] is True
    assert payload["provider_usage_attestation"]["cost_estimate_usd"] == 0.0
    assert payload["coverage_accounting"] == {
        "semantics": "failed_v1_attempt_remains_open",
        "remaining_uncovered_tickers": ["AAA"],
        "rerun_required": True,
    }


def test_only_unswept_rejects_max_candidates(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
            "--max-candidates",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "requires the complete loaded cell" in result.output
    with get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sector_run_loaded_sets").fetchone()[0] == 0


def test_only_unswept_empty_delta_records_new_structural_coverage(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    gate_result = {
        "GATED": {
            "ticker": "GATED",
            "quarantined": True,
            "excluded_error": False,
            "triggered_codes": ["PENNY_FLOOR"],
        }
    }

    def fake_resolver(**kwargs):
        return _FakeSelection(
            [],
            loaded=["GATED"],
            structural_gate_results=gate_result,
        )

    def fail_runtime(**kwargs):
        raise AssertionError("structural-only delta must never reach the LLM runtime")

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        fake_resolver,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        fail_runtime,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "EMPTY_DELTA"
    assert payload["coverage_uncovered_before_record"] == 1
    assert payload["coverage_record"]["rows_inserted"] == 1
    assert payload["coverage_record"]["tickers"] == ["GATED"]
    assert set(payload["coverage_record"]["coverage_evidence_sha256"]) == {"GATED"}
    with get_db() as conn:
        assert swept_tickers_for_band(conn, "micro_cap") == {"GATED"}
        evidence_row = conn.execute(
            "SELECT coverage_evidence_json, coverage_evidence_sha256 "
            "FROM sector_run_loaded_sets WHERE ticker = 'GATED'"
        ).fetchone()
        evidence = json.loads(evidence_row["coverage_evidence_json"])
        assert evidence["disposition"] == "STRUCTURAL_SCREENED"
        assert evidence["structural_gate_result"]["triggered_codes"] == ["PENNY_FLOOR"]
        assert len(evidence_row["coverage_evidence_sha256"]) == 64

    again = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
        ],
    )
    assert again.exit_code == 0, again.output
    assert json.loads(again.output)["coverage_uncovered_before_record"] == 0


def test_only_unswept_gate_error_remains_uncovered(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    gate_result = {
        "BROKEN": {
            "ticker": "BROKEN",
            "quarantined": False,
            "excluded_error": True,
            "triggered_codes": [],
            "degraded_codes": ["GATE_EVAL_FAILED"],
        }
    }

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: _FakeSelection(
            [],
            loaded=["BROKEN"],
            structural_gate_results=gate_result,
        ),
    )
    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "INCOMPLETE_DELTA"
    assert payload["coverage_uncovered_before_record"] == 1
    assert payload["coverage_record"]["rows_inserted"] == 0
    assert payload["coverage_blocked_by_gate_error"] == ["BROKEN"]
    with get_db() as conn:
        assert swept_tickers_for_band(conn, "micro_cap") == set()


def test_sweep_delta_report_cli(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    def fake_resolver(**kwargs):
        return _FakeSelection(["NEW1"])

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers", fake_resolver
    )
    result = runner.invoke(
        app, ["sweep-delta-report", "--band", "micro_cap", "--sectors", "energy"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["report"]["totals"]["to_review"] == 1


def test_sweep_delta_report_full_universe_require_complete(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.autonomous.sweep_delta.full_universe_delta_report",
        lambda **kwargs: {"complete": False, "status": "INCOMPLETE"},
    )

    result = runner.invoke(
        app,
        [
            "sweep-delta-report",
            "--full-universe-v1",
            "--pipeline-version",
            "v1",
            "--require-complete",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["report"]["status"] == "INCOMPLETE"


def test_sweep_delta_report_full_universe_rejects_partial_grid(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "sweep-delta-report",
            "--full-universe-v1",
            "--sectors",
            "energy",
        ],
    )

    assert result.exit_code == 2


def test_campaign_scoped_coverage_ignores_lifetime_rows(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id="lifetime",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
            pipeline_version="v1",
            candidate_dispositions={"AAA": "LLM_CANDIDATE_REVIEW_COMPLETED"},
        )

        assert swept_tickers_for_band(conn, "micro_cap", "energy", pipeline_version="v1") == {"AAA"}
        assert (
            swept_tickers_for_band(
                conn,
                "micro_cap",
                "energy",
                pipeline_version="v1",
                coverage_campaign_id="fresh-campaign",
            )
            == set()
        )

        record_loaded_set(
            conn,
            run_id="fresh",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
            pipeline_version="v1",
            candidate_dispositions={"AAA": "LLM_CANDIDATE_REVIEW_COMPLETED"},
            coverage_campaign_id="fresh-campaign",
        )
        assert swept_tickers_for_band(
            conn,
            "micro_cap",
            "energy",
            pipeline_version="v1",
            coverage_campaign_id="fresh-campaign",
        ) == {"AAA"}


def test_full_campaign_can_disable_carried_verdicts(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_outcome(
            conn,
            "AAA",
            "autonomous_sector_energy_prior",
            "WATCHLIST_ONLY",
            "2026-07-01T00:00:00Z",
        )
        split = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v1",
            coverage_campaign_id="fresh-campaign",
            allow_carried_verdicts=False,
        )

    assert split["carried"] == {}
    assert split["to_review"] == ["AAA"]


def test_campaign_scope_ignores_preexisting_carried_coverage_row(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id="poisoned-carry",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["AAA"],
            pipeline_version="v1",
            candidate_dispositions={"AAA": "CARRIED_VERDICT"},
            coverage_campaign_id="fresh-campaign",
        )
        split = split_unswept(
            conn,
            band="micro_cap",
            sector="energy",
            selected_tickers=["AAA"],
            pipeline_version="v1",
            coverage_campaign_id="fresh-campaign",
            allow_carried_verdicts=False,
        )

    assert split["swept"] == []
    assert split["to_review"] == ["AAA"]


def test_unknown_cap_terminal_state_projects_to_all_atomic_bands(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id="primary",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["UNKNOWN"],
            pipeline_version="v1",
            candidate_dispositions={"UNKNOWN": "LLM_CANDIDATE_REVIEW_COMPLETED"},
            coverage_campaign_id="fresh-campaign",
        )
        result = record_unknown_cap_cross_band_coverage(
            conn,
            run_id="primary",
            sector="energy",
            source="sector_scan_db",
            primary_band="micro_cap",
            unknown_cap_tickers=["UNKNOWN"],
            candidate_dispositions={"UNKNOWN": "LLM_CANDIDATE_REVIEW_COMPLETED"},
            coverage_campaign_id="fresh-campaign",
        )

        assert result == {
            "tickers": ["UNKNOWN"],
            "bands": ["small_cap", "mid_cap", "large_cap", "mega_cap"],
            "rows_inserted_or_upgraded": 4,
        }
        for band in V1_ATOMIC_BANDS:
            assert swept_tickers_for_band(
                conn,
                band,
                "energy",
                pipeline_version="v1",
                coverage_campaign_id="fresh-campaign",
            ) == {"UNKNOWN"}
