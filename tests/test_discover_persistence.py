"""Tests for app.discover.persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def tmp_session_db(tmp_path) -> Path:
    """Return a Path to a temp session DB file; caller opens it."""
    return tmp_path / "discover_sessions.db"


def test_ensure_schema_creates_tables(tmp_session_db):
    from app.discover.persistence import ensure_schema, SESSION_TABLES

    ensure_schema(tmp_session_db)
    conn = sqlite3.connect(tmp_session_db)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert set(SESSION_TABLES).issubset(tables)
    conn.close()


def test_ensure_schema_migrates_financial_scope_columns(tmp_session_db):
    from app.discover.persistence import ensure_schema

    conn = sqlite3.connect(tmp_session_db)
    try:
        for table_name in (
            "stage2_results",
            "stage3_results",
            "stage4_results",
        ):
            conn.execute(
                f"""
                CREATE TABLE {table_name} (
                    sweep_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    PRIMARY KEY (sweep_id, ticker)
                )
                """
            )
        conn.commit()
    finally:
        conn.close()

    ensure_schema(tmp_session_db)

    conn = sqlite3.connect(tmp_session_db)
    try:
        for table_name in (
            "stage2_results",
            "stage3_results",
            "stage4_results",
        ):
            columns = {
                str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            assert "financial_scope_fingerprint" in columns
            assert "financial_scope_publication_fingerprint" in columns
            assert "publication_evidence_json" in columns
            assert "publication_row_sha256" in columns
            if table_name == "stage4_results":
                assert "financial_scope_manifest_json" in columns
    finally:
        conn.close()


def test_ensure_schema_adds_append_only_publication_authorization_ledger(
    tmp_session_db,
):
    from app.discover.persistence import ensure_schema

    ensure_schema(tmp_session_db)

    conn = sqlite3.connect(tmp_session_db)
    try:
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(discover_publication_authorizations)"
            ).fetchall()
        }
        triggers = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'trigger'
                  AND tbl_name = 'discover_publication_authorizations'
                """
            ).fetchall()
        }
    finally:
        conn.close()

    assert {
        "stage",
        "sweep_id",
        "ticker",
        "publication_row_sha256",
        "financial_scope_fingerprint",
        "publication_evidence_sha256",
        "authorized_at",
    }.issubset(columns)
    assert triggers == {
        "discover_publication_authorizations_no_update",
        "discover_publication_authorizations_no_delete",
    }


def test_create_sweep_and_fetch(tmp_session_db):
    from app.discover.persistence import ensure_schema, create_sweep, get_sweep

    ensure_schema(tmp_session_db)
    sweep_id = create_sweep(
        tmp_session_db,
        sweep_id="20260411-150000",
        universe_size=100,
        limit_applied=10,
        budget_usd=25.0,
    )
    assert sweep_id == "20260411-150000"

    sweep = get_sweep(tmp_session_db, sweep_id)
    assert sweep["sweep_id"] == "20260411-150000"
    assert sweep["universe_size"] == 100
    assert sweep["limit_applied"] == 10
    assert sweep["budget_usd"] == 25.0
    assert sweep["status"] == "running"
    assert sweep["total_cost_usd"] == 0.0


def test_paid_attempt_reservation_enforces_sweep_hard_cap_before_insert(tmp_session_db):
    from app.discover.persistence import (
        DiscoverCostBudgetExceeded,
        _reserve_discover_paid_attempt,
        create_sweep,
        ensure_schema,
        get_sweep,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "hard-cap", 1, None, 0.0)

    with pytest.raises(DiscoverCostBudgetExceeded, match="hard cost cap"):
        _reserve_discover_paid_attempt(
            tmp_session_db,
            stage=2,
            sweep_id="hard-cap",
            ticker="AAA",
            execution_id="execution-1",
            request_payload={"prompt": "literal request"},
            provider="anthropic",
            model="literal-model",
            estimated_cost_usd=0.01,
        )

    conn = sqlite3.connect(tmp_session_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM discover_paid_attempts").fetchone()[0] == 0
    finally:
        conn.close()
    assert get_sweep(tmp_session_db, "hard-cap")["total_cost_usd"] == 0.0


def test_paid_attempt_reservation_enforces_stage4_ticker_cap_before_insert(
    tmp_session_db,
):
    from app.discover.persistence import (
        DiscoverCostBudgetExceeded,
        _reserve_discover_paid_attempt,
        create_sweep,
        ensure_schema,
        get_sweep,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "ticker-cap", 1, None, 1.0)

    with pytest.raises(DiscoverCostBudgetExceeded, match="Stage 4"):
        _reserve_discover_paid_attempt(
            tmp_session_db,
            stage=4,
            sweep_id="ticker-cap",
            ticker="AAA",
            execution_id="execution-1",
            request_payload={"prompt": "literal request"},
            provider="anthropic",
            model="literal-model",
            estimated_cost_usd=0.04,
            hard_ticker_cost_limit_usd=0.03,
        )

    conn = sqlite3.connect(tmp_session_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM discover_paid_attempts").fetchone()[0] == 0
    finally:
        conn.close()
    assert get_sweep(tmp_session_db, "ticker-cap")["total_cost_usd"] == 0.0


def test_paid_attempt_cost_overrun_is_durably_nonrepeatable_and_nonpublishable(
    tmp_session_db,
):
    from app.discover.persistence import (
        DiscoverCostIntegrityError,
        DiscoverPaidAttemptAmbiguousError,
        _complete_discover_paid_attempt,
        _reserve_discover_paid_attempt,
        create_sweep,
        discover_paid_attempt_summary,
        ensure_schema,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "cost-overrun", 1, None, 1.0)
    attempt_number = _reserve_discover_paid_attempt(
        tmp_session_db,
        stage=2,
        sweep_id="cost-overrun",
        ticker="AAA",
        execution_id="execution-1",
        request_payload={"prompt": "literal request"},
        provider="anthropic",
        model="literal-model",
        estimated_cost_usd=0.01,
    )

    with pytest.raises(DiscoverCostIntegrityError, match="actual"):
        _complete_discover_paid_attempt(
            tmp_session_db,
            stage=2,
            sweep_id="cost-overrun",
            ticker="AAA",
            execution_id="execution-1",
            attempt_number=attempt_number,
            outcome="SUCCESS",
            input_tokens=10,
            output_tokens=10,
            accounted_cost_usd=0.02,
            error=None,
        )

    conn = sqlite3.connect(tmp_session_db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM discover_paid_attempts").fetchone()
    finally:
        conn.close()
    assert row["status"] == "FAILED"
    assert row["outcome"] == "ERROR"
    assert row["cost_integrity_violation"] == 1
    assert row["accounted_cost_usd"] == 0.02

    with pytest.raises(DiscoverPaidAttemptAmbiguousError):
        _reserve_discover_paid_attempt(
            tmp_session_db,
            stage=2,
            sweep_id="cost-overrun",
            ticker="AAA",
            execution_id="execution-1",
            request_payload={"prompt": "literal request"},
            provider="anthropic",
            model="literal-model",
            estimated_cost_usd=0.01,
        )
    with pytest.raises(DiscoverCostIntegrityError, match="cannot authorize publication"):
        discover_paid_attempt_summary(
            tmp_session_db,
            stage=2,
            sweep_id="cost-overrun",
            ticker="AAA",
        )


def test_insert_and_fetch_stage2_result(tmp_session_db):
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_stage2_result,
        list_stage2_keeps,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "20260411-150000", 10, 5, 10.0)

    insert_stage2_result(
        tmp_session_db,
        sweep_id="20260411-150000",
        ticker="CCSI",
        decision="KEEP",
        confidence="MODERATE",
        reason="Deep value setup with real MOS",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.0025,
        wall_ms=2200,
    )
    insert_stage2_result(
        tmp_session_db,
        sweep_id="20260411-150000",
        ticker="HBIO",
        decision="DROP",
        confidence="HIGH",
        reason="Structural decline",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.0025,
        wall_ms=2200,
    )

    keeps = list_stage2_keeps(tmp_session_db, "20260411-150000")
    assert len(keeps) == 1
    assert keeps[0]["ticker"] == "CCSI"


def test_insert_and_fetch_stage3_result(tmp_session_db):
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_stage3_result,
        list_stage3_survivors,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "20260411-150000", 10, 5, 10.0)

    insert_stage3_result(
        tmp_session_db,
        sweep_id="20260411-150000",
        ticker="GILD",
        verdict="WATCH",
        confidence="MODERATE",
        thesis_summary="DCF/EPV gap worth investigating",
        key_numbers=["DCF $70.76", "EPV $30.58"],
        positives=["Pipeline"],
        risks=["Solvency"],
        open_questions=["Current price?"],
        reasoning_trace="Weighted toward risk",
        input_tokens=4000,
        output_tokens=1500,
        cost_usd=0.04,
        wall_ms=40000,
    )
    insert_stage3_result(
        tmp_session_db,
        sweep_id="20260411-150000",
        ticker="HBIO",
        verdict="PASS",
        confidence="HIGH",
        thesis_summary="Structurally impaired",
        key_numbers=["negative EPV"],
        positives=[],
        risks=["decline"],
        open_questions=[],
        reasoning_trace="Clear pass",
        input_tokens=4000,
        output_tokens=1500,
        cost_usd=0.04,
        wall_ms=40000,
    )

    survivors = list_stage3_survivors(tmp_session_db, "20260411-150000")
    assert len(survivors) == 1
    assert survivors[0]["ticker"] == "GILD"


def test_update_sweep_total_cost(tmp_session_db):
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        add_sweep_cost,
        get_sweep,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "20260411-150000", 10, 5, 10.0)
    add_sweep_cost(tmp_session_db, "20260411-150000", 1.25)
    add_sweep_cost(tmp_session_db, "20260411-150000", 0.75)

    sweep = get_sweep(tmp_session_db, "20260411-150000")
    assert sweep["total_cost_usd"] == 2.0


def test_finalize_sweep(tmp_session_db):
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        finalize_sweep,
        get_sweep,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "20260411-150000", 10, 5, 10.0)
    finalize_sweep(tmp_session_db, "20260411-150000", status="completed")

    sweep = get_sweep(tmp_session_db, "20260411-150000")
    assert sweep["status"] == "completed"
    assert sweep["finished_at"] is not None


def test_insert_and_get_sweep_universe(tmp_session_db):
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_sweep_universe,
        get_sweep_universe,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "20260411-150000", 5, None, 25.0)
    insert_sweep_universe(
        tmp_session_db,
        "20260411-150000",
        [("AAPL", "2026-04-10"), ("MSFT", "2026-04-10"), ("GOOG", "2026-04-09")],
    )

    result = get_sweep_universe(tmp_session_db, "20260411-150000")
    assert len(result) == 3
    tickers = {t for t, _ in result}
    assert tickers == {"AAPL", "MSFT", "GOOG"}
    # Verify (ticker, as_of_date) round-trip
    lookup = {t: d for t, d in result}
    assert lookup["GOOG"] == "2026-04-09"


def test_sweep_universe_empty_for_legacy_sweep(tmp_session_db):
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        get_sweep_universe,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "20260411-150000", 5, None, 25.0)
    # No insert_sweep_universe call — simulates a pre-snapshot sweep
    result = get_sweep_universe(tmp_session_db, "20260411-150000")
    assert result == []


def test_update_sweep_status_and_budget(tmp_session_db):
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        get_sweep,
        update_sweep_status,
        update_sweep_budget,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "20260411-150000", 10, None, 25.0)

    update_sweep_status(tmp_session_db, "20260411-150000", "aborted")
    sweep = get_sweep(tmp_session_db, "20260411-150000")
    assert sweep["status"] == "aborted"

    update_sweep_budget(tmp_session_db, "20260411-150000", 50.0)
    sweep = get_sweep(tmp_session_db, "20260411-150000")
    assert sweep["budget_usd"] == 50.0


def test_list_stage_tickers(tmp_session_db):
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_stage2_result,
        insert_stage3_result,
        insert_stage4_result,
        list_stage2_tickers,
        list_stage3_tickers,
        list_stage4_tickers,
    )

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 5, None, 10.0)

    insert_stage2_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        decision="KEEP",
        confidence="HIGH",
        reason="ok",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_ms=100,
    )
    insert_stage2_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="BBB",
        decision="DROP",
        confidence="HIGH",
        reason="no",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_ms=100,
    )

    assert list_stage2_tickers(tmp_session_db, "s1") == {"AAA", "BBB"}
    assert list_stage3_tickers(tmp_session_db, "s1") == set()
    assert list_stage4_tickers(tmp_session_db, "s1") == set()

    insert_stage3_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        verdict="WATCH",
        confidence="MODERATE",
        thesis_summary="ok",
        key_numbers=[],
        positives=[],
        risks=[],
        open_questions=[],
        reasoning_trace="ok",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        wall_ms=1000,
    )

    assert list_stage3_tickers(tmp_session_db, "s1") == {"AAA"}

    insert_stage4_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        verdict="BUY",
        confidence="HIGH",
        thesis="deep",
        key_findings=[],
        open_questions=[],
        falsifiers=[],
        reasoning_trace="ok",
        num_turns=3,
        tool_call_counts={},
        termination_reason="finalize_analysis",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.05,
        wall_seconds=30.0,
    )

    assert list_stage4_tickers(tmp_session_db, "s1") == {"AAA"}


def test_stage2_and_stage3_authorized_reads_reject_tampered_rows(tmp_session_db):
    from app.discover.persistence import (
        build_discover_publication_evidence,
        create_sweep,
        ensure_schema,
        insert_stage2_result,
        insert_stage3_result,
        list_stage2_results,
        list_stage3_results,
    )

    scope = "a" * 64
    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "tampered", 1, None, 10.0)
    insert_stage2_result(
        tmp_session_db,
        sweep_id="tampered",
        ticker="AAA",
        decision="KEEP",
        confidence="HIGH",
        reason="bound reason",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_ms=100,
        financial_scope_fingerprint=scope,
        financial_scope_publication_fingerprint=scope,
        publication_evidence=build_discover_publication_evidence(
            stage=2,
            ticker="AAA",
            scope_fingerprint=scope,
            primary_evidence={"fixture": "stage2"},
        ),
    )
    insert_stage3_result(
        tmp_session_db,
        sweep_id="tampered",
        ticker="AAA",
        verdict="WATCH",
        confidence="MODERATE",
        thesis_summary="bound thesis",
        key_numbers=[],
        positives=[],
        risks=[],
        open_questions=[],
        reasoning_trace="bound trace",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        wall_ms=1000,
        financial_scope_fingerprint=scope,
        financial_scope_publication_fingerprint=scope,
        publication_evidence=build_discover_publication_evidence(
            stage=3,
            ticker="AAA",
            scope_fingerprint=scope,
            primary_evidence={"fixture": "stage3"},
        ),
    )

    conn = sqlite3.connect(tmp_session_db)
    try:
        conn.execute(
            "UPDATE stage2_results SET reason = 'tampered reason' WHERE sweep_id = 'tampered'"
        )
        conn.execute(
            "UPDATE stage3_results "
            "SET thesis_summary = 'tampered thesis' WHERE sweep_id = 'tampered'"
        )
        conn.commit()
    finally:
        conn.close()

    assert list_stage2_results(tmp_session_db, "tampered", require_financial_scope=True) == []
    assert list_stage3_results(tmp_session_db, "tampered", require_financial_scope=True) == []
