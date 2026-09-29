"""Tests for app.discover.report."""

from __future__ import annotations

from pathlib import Path

import pytest


def _publication_evidence(stage: int, ticker: str, scope: str) -> dict:
    from app.discover.persistence import build_discover_publication_evidence

    return build_discover_publication_evidence(
        stage=stage,
        ticker=ticker,
        scope_fingerprint=scope,
        primary_evidence={"fixture": f"stage{stage}:{ticker}"},
        stage4_tool_manifest=[] if stage == 4 else None,
    )


@pytest.fixture
def seeded_db(tmp_path, monkeypatch) -> tuple[Path, str]:
    from app.discover import persistence as persistence_module
    from app.discover.persistence import (
        add_sweep_cost,
        create_sweep,
        ensure_schema,
        finalize_sweep,
        insert_stage2_result,
        insert_stage3_result,
        insert_stage4_result,
    )

    # These rendering fixtures isolate exact-row rendering/tamper behavior.
    # Paid-attempt and append-only authorization-ledger behavior has dedicated
    # production-runner regressions, so this seam validates every immutable
    # publication field except that separate ledger membership.
    def fixture_row_is_authorized(_conn, stage, row):
        paid = str(row["financial_scope_fingerprint"] or "").strip().lower()
        publication = str(row["financial_scope_publication_fingerprint"] or "").strip().lower()
        evidence = persistence_module._strict_publication_evidence(
            row["publication_evidence_json"],
            stage=stage,
            ticker=row["ticker"],
            scope_fingerprint=paid,
        )
        if len(paid) != 64 or publication != paid or evidence is None:
            return False
        if int(stage) == 4:
            import json

            try:
                raw_manifest = row["financial_scope_manifest_json"]
                manifest = (
                    json.loads(raw_manifest) if isinstance(raw_manifest, str) else raw_manifest
                )
            except json.JSONDecodeError:
                return False
            strict_manifest = persistence_module._strict_stage4_tool_manifest(
                manifest,
                ticker=str(row["ticker"]),
            )
            if strict_manifest is None or len(evidence["entries"]) != len(strict_manifest) + 1:
                return False
        return (
            persistence_module._publication_row_sha256(stage, row) == row["publication_row_sha256"]
        )

    monkeypatch.setattr(
        persistence_module,
        "_row_publication_is_authorized",
        fixture_row_is_authorized,
    )

    db = tmp_path / "discover_sessions.db"
    ensure_schema(db)
    create_sweep(db, "20260411-150000", universe_size=3, limit_applied=3, budget_usd=25.0)

    insert_stage2_result(
        db,
        sweep_id="20260411-150000",
        ticker="CCSI",
        decision="KEEP",
        confidence="MODERATE",
        reason="good spread",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.0025,
        wall_ms=2200,
        financial_scope_fingerprint="a" * 64,
        financial_scope_publication_fingerprint="a" * 64,
        publication_evidence=_publication_evidence(2, "CCSI", "a" * 64),
    )
    insert_stage2_result(
        db,
        sweep_id="20260411-150000",
        ticker="HBIO",
        decision="DROP",
        confidence="HIGH",
        reason="decline",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.0025,
        wall_ms=2200,
        financial_scope_fingerprint="a" * 64,
        financial_scope_publication_fingerprint="a" * 64,
        publication_evidence=_publication_evidence(2, "HBIO", "a" * 64),
    )
    insert_stage2_result(
        db,
        sweep_id="20260411-150000",
        ticker="GILD",
        decision="KEEP",
        confidence="MODERATE",
        reason="pipeline",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.0025,
        wall_ms=2200,
        financial_scope_fingerprint="a" * 64,
        financial_scope_publication_fingerprint="a" * 64,
        publication_evidence=_publication_evidence(2, "GILD", "a" * 64),
    )

    insert_stage3_result(
        db,
        sweep_id="20260411-150000",
        ticker="CCSI",
        verdict="BUY_CANDIDATE",
        confidence="HIGH",
        thesis_summary="Clear value",
        key_numbers=["MOS 32%"],
        positives=["Cash flow"],
        risks=["Cyclical"],
        open_questions=["Trend?"],
        reasoning_trace="Strong case",
        input_tokens=4200,
        output_tokens=1750,
        cost_usd=0.0386,
        wall_ms=40000,
        financial_scope_fingerprint="b" * 64,
        financial_scope_publication_fingerprint="b" * 64,
        publication_evidence=_publication_evidence(3, "CCSI", "b" * 64),
    )
    insert_stage3_result(
        db,
        sweep_id="20260411-150000",
        ticker="GILD",
        verdict="WATCH",
        confidence="MODERATE",
        thesis_summary="Pipeline bet",
        key_numbers=["DCF $70.76"],
        positives=["Lenacapavir"],
        risks=["Leverage"],
        open_questions=["Price?"],
        reasoning_trace="Needs more data",
        input_tokens=4200,
        output_tokens=1750,
        cost_usd=0.0386,
        wall_ms=40000,
        financial_scope_fingerprint="b" * 64,
        financial_scope_publication_fingerprint="b" * 64,
        publication_evidence=_publication_evidence(3, "GILD", "b" * 64),
    )

    insert_stage4_result(
        db,
        sweep_id="20260411-150000",
        ticker="CCSI",
        verdict="BUY",
        confidence="HIGH",
        thesis="Strong fundamental setup with 32% MOS",
        key_findings=["DCF $31.85 vs price $25.33", "Stable EPV"],
        open_questions=["Cycle turn timing"],
        falsifiers=["Revenue decline >5%"],
        reasoning_trace="All signals align",
        num_turns=3,
        tool_call_counts={"fetch_historical_scorecards": 1, "finalize_analysis": 1},
        termination_reason="finalize_analysis",
        input_tokens=3000,
        output_tokens=800,
        cost_usd=0.1360,
        wall_seconds=51.3,
        financial_scope_fingerprint="c" * 64,
        financial_scope_publication_fingerprint="c" * 64,
        financial_scope_manifest=[],
        publication_evidence=_publication_evidence(4, "CCSI", "c" * 64),
    )
    insert_stage4_result(
        db,
        sweep_id="20260411-150000",
        ticker="GILD",
        verdict="PASS",
        confidence="HIGH",
        thesis="Price at 2x DCF",
        key_findings=["Price $139.71", "DCF $70.76"],
        open_questions=[],
        falsifiers=["Price below $70"],
        reasoning_trace="Clear premium",
        num_turns=3,
        tool_call_counts={"fetch_historical_scorecards": 1, "finalize_analysis": 1},
        termination_reason="finalize_analysis",
        input_tokens=3000,
        output_tokens=800,
        cost_usd=0.1360,
        wall_seconds=51.3,
        financial_scope_fingerprint="c" * 64,
        financial_scope_publication_fingerprint="c" * 64,
        financial_scope_manifest=[],
        publication_evidence=_publication_evidence(4, "GILD", "c" * 64),
    )

    add_sweep_cost(db, "20260411-150000", 0.0025 * 3 + 0.0386 * 2 + 0.1360 * 2)
    finalize_sweep(db, "20260411-150000", status="completed")

    return db, "20260411-150000"


def test_render_sweep_report_contains_buy_verdicts(seeded_db):
    from app.discover.report import render_sweep_report

    db_path, sweep_id = seeded_db
    md = render_sweep_report(db_path, sweep_id)

    assert "# Discover Sweep Report" in md
    assert sweep_id in md
    assert "BUY" in md
    assert "CCSI" in md
    assert "Strong fundamental setup with 32% MOS" in md
    assert "Total cost" in md
    assert "GILD" in md
    # HBIO is in stage 2 DROP section
    assert "HBIO" in md
    assert "decline" in md


def test_render_sweep_report_handles_no_stage4_results(tmp_path):
    from app.discover.persistence import (
        create_sweep,
        ensure_schema,
        finalize_sweep,
        insert_stage2_result,
    )
    from app.discover.report import render_sweep_report

    db = tmp_path / "discover_sessions.db"
    ensure_schema(db)
    create_sweep(db, "20260411-150001", universe_size=1, limit_applied=1, budget_usd=5.0)
    insert_stage2_result(
        db,
        sweep_id="20260411-150001",
        ticker="HBIO",
        decision="DROP",
        confidence="HIGH",
        reason="decline",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.0025,
        wall_ms=2200,
        financial_scope_fingerprint="a" * 64,
        financial_scope_publication_fingerprint="a" * 64,
        publication_evidence=_publication_evidence(2, "HBIO", "a" * 64),
    )
    finalize_sweep(db, "20260411-150001", status="completed")

    md = render_sweep_report(db, "20260411-150001")
    assert "No Stage 4 results" in md or "No BUY candidates" in md


def test_render_sweep_report_suppresses_tampered_stage4_decision(seeded_db):
    import sqlite3

    from app.discover.report import render_sweep_report

    db_path, sweep_id = seeded_db
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            UPDATE stage4_results
            SET thesis = 'tampered BUY thesis'
            WHERE sweep_id = ? AND ticker = 'CCSI'
            """,
            (sweep_id,),
        )
        conn.commit()
    finally:
        conn.close()

    report = render_sweep_report(db_path, sweep_id)
    assert "tampered BUY thesis" not in report
    assert "No BUY candidates surfaced in this sweep." in report
    assert "Stage 4: 1" in report


def test_render_sweep_report_suppresses_malformed_stage4_manifest(seeded_db):
    import sqlite3

    from app.discover.report import render_sweep_report

    db_path, sweep_id = seeded_db
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            UPDATE stage4_results
            SET financial_scope_manifest_json = '[1]'
            WHERE sweep_id = ? AND ticker = 'CCSI'
            """,
            (sweep_id,),
        )
        conn.commit()
    finally:
        conn.close()

    report = render_sweep_report(db_path, sweep_id)
    assert "Strong fundamental setup with 32% MOS" not in report
    assert "No BUY candidates surfaced in this sweep." in report
    assert "Stage 4: 1" in report
