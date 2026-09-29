"""Tests for stage-level skip-if-present (resume support).

Each stage's run_* function must skip tickers that already have a result
row for the given sweep_id. These tests verify the idempotency mechanism
that makes sweep resume work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import pytest

_BOUND_SCOPE = "a" * 64


def _publication_evidence(stage: int, ticker: str) -> dict[str, object]:
    from app.discover.persistence import build_discover_publication_evidence

    return build_discover_publication_evidence(
        stage=stage,
        ticker=ticker,
        scope_fingerprint=_BOUND_SCOPE,
        primary_evidence={"fixture": f"resume-stage-{stage}:{ticker}"},
        stage4_tool_manifest=[] if stage == 4 else None,
    )


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """Legacy resume fixtures isolate idempotency from source authorization."""

    from app.autonomous.v1_financial_context import BoundV1FinancialScope
    from app.discover import persistence as persistence_module

    class _Scope(BoundV1FinancialScope):
        def require(self, **_kwargs):
            return SimpleNamespace(scope_fingerprint=self.expected_scope_fingerprint)

    def _scope(**kwargs):
        ticker = str(kwargs["context"]).split(":")[1].upper()
        return _Scope(
            context=kwargs["context"],
            run_as_of_date=kwargs["run_as_of_date"],
            packets=({"ticker": ticker},),
            scenarios=tuple(kwargs["scenarios"]),
            expected_scope_fingerprint=_BOUND_SCOPE,
        )

    for module in (
        "app.discover.stage2",
        "app.discover.stage3",
        "app.discover.stage4",
    ):
        monkeypatch.setattr(
            f"{module}.bind_v1_financial_scope",
            _scope,
        )

    def _legacy_fixture_authorized(_conn, stage, row):
        return (
            str(row["financial_scope_fingerprint"] or "").lower() == _BOUND_SCOPE
            and str(row["financial_scope_publication_fingerprint"] or "").lower() == _BOUND_SCOPE
            and str(row["publication_row_sha256"] or "").lower()
            == persistence_module._publication_row_sha256(stage, row)
        )

    monkeypatch.setattr(
        persistence_module,
        "_row_publication_is_authorized",
        _legacy_fixture_authorized,
    )

    def _legacy_authorized_row(row, *, engine_db_path):
        import json

        ticker = str(row["ticker"]).upper()
        as_of_date = str(row["as_of_date"])
        scorecard = json.loads(row["outputs_json"] or "{}")
        packet = {
            "ticker": ticker,
            "raw_valuation": scorecard,
            "current_price": None,
            "dcf_value": None,
            "epv_value": None,
            "graham_value": None,
        }
        return ticker, as_of_date, scorecard, packet

    monkeypatch.setattr(
        "app.discover.sweep._authorized_scorecard_input",
        _legacy_authorized_row,
    )

    def _legacy_selected_row(
        conn,
        *,
        ticker,
        as_of_date=None,
        exact_as_of_date=False,
    ):
        clauses = ["ticker = ?", "method = 'scorecard'"]
        params = [ticker]
        if as_of_date is not None:
            clauses.append(f"as_of_date {'=' if exact_as_of_date else '<='} ?")
            params.append(as_of_date)
        return conn.execute(
            "SELECT * FROM valuations WHERE "
            + " AND ".join(clauses)
            + " ORDER BY as_of_date DESC LIMIT 1",
            params,
        ).fetchone()

    monkeypatch.setattr(
        "app.discover.sweep._select_authorized_scorecard_row",
        _legacy_selected_row,
    )


# ---------------------------------------------------------------------------
# Shared fakes (copied from test_discover_sweep.py)
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int = 1500
    output_tokens: int = 200


@dataclass
class _FakeToolUseBlock:
    type: str = "tool_use"
    name: str = ""
    id: str = "tu"
    input: dict = field(default_factory=dict)


@dataclass
class _FakeMessage:
    content: list
    usage: _FakeUsage
    stop_reason: str = "tool_use"


class _ScriptedClient:
    """Returns canned responses in order; records every call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls.append(kwargs)
            return self._outer._responses.pop(0)

    @property
    def messages(self):
        return _ScriptedClient._Messages(self)


def _stage2_keep() -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeToolUseBlock(
                name="classify_ticker",
                input={"decision": "KEEP", "confidence": "MODERATE", "reason": "ok"},
            )
        ],
        usage=_FakeUsage(1500, 200),
    )


def _stage2_drop() -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeToolUseBlock(
                name="classify_ticker",
                input={"decision": "DROP", "confidence": "HIGH", "reason": "decline"},
            )
        ],
        usage=_FakeUsage(1500, 200),
    )


def _stage3_watch() -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeToolUseBlock(
                name="stage3_analysis",
                input={
                    "verdict": "WATCH",
                    "confidence": "MODERATE",
                    "thesis_summary": "worth a look",
                    "key_numbers": [],
                    "positives": [],
                    "risks": [],
                    "open_questions": [],
                    "reasoning_trace": "ok",
                },
            )
        ],
        usage=_FakeUsage(4200, 1750),
    )


def _stage3_pass() -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeToolUseBlock(
                name="stage3_analysis",
                input={
                    "verdict": "PASS",
                    "confidence": "HIGH",
                    "thesis_summary": "clear pass",
                    "key_numbers": [],
                    "positives": [],
                    "risks": [],
                    "open_questions": [],
                    "reasoning_trace": "clear",
                },
            )
        ],
        usage=_FakeUsage(4200, 1750),
    )


def _stage4_finalize() -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeToolUseBlock(
                name="finalize_analysis",
                id="tu_final",
                input={
                    "verdict": "BUY",
                    "confidence": "HIGH",
                    "thesis": "Deep conviction",
                    "key_findings": [],
                    "open_questions": [],
                    "falsifiers": [],
                    "reasoning_trace": "ok",
                },
            )
        ],
        usage=_FakeUsage(3000, 800),
    )


def _sample_bundle(ticker: str):
    from app.analyst.evidence_bundle import AnalysisEvidenceBundle, ValuationSnapshot

    return AnalysisEvidenceBundle(
        ticker=ticker,
        as_of_date="2026-04-10",
        built_at="2026-04-10T12:00:00+00:00",
        analysis_years=5,
        analysis_quarters=0,
        freshness_window_days=90,
        valuation=ValuationSnapshot(
            current_price=25.0,
            market_cap=1000.0,
            dcf_base=32.0,
            epv_adjusted=38.0,
            graham_value=None,
            methods_agree=True,
            tension_type=None,
            gate_action="PROCEED",
            solvency_status="LOW",
            filing_risk_status="OK",
        ),
    )


def _create_fake_engine_db(tmp_path: Path, tickers: list[str]) -> Path:
    """Create a minimal engine.db with scorecard rows for the given tickers."""
    import json
    import sqlite3

    engine_db = tmp_path / "engine.db"
    conn = sqlite3.connect(str(engine_db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS valuations (
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            method TEXT NOT NULL,
            outputs_json TEXT
        )
    """)
    for ticker in tickers:
        conn.execute(
            "INSERT INTO valuations (ticker, as_of_date, method, outputs_json) VALUES (?, ?, ?, ?)",
            (ticker, "2026-04-10", "scorecard", json.dumps({"pricing_zone": "MOS"})),
        )
    conn.commit()
    conn.close()
    return engine_db


@pytest.fixture
def tmp_session_db(tmp_path) -> Path:
    return tmp_path / "discover_sessions.db"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage2_skip_existing(tmp_session_db):
    """Pre-insert AAA with decision=KEEP; run_stage2 with [AAA, BBB].
    Only 1 API call should be made (for BBB); AAA's row stays untouched.
    """
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_stage2_result,
        list_stage2_results,
    )
    from app.discover.stage2 import run_stage2

    ensure_schema(tmp_session_db)
    sweep_id = "sweep-resume-s2"
    create_sweep(tmp_session_db, sweep_id, universe_size=2, limit_applied=None, budget_usd=100.0)

    # Pre-insert AAA with decision=KEEP (simulating a prior run)
    insert_stage2_result(
        tmp_session_db,
        sweep_id=sweep_id,
        ticker="AAA",
        decision="KEEP",
        confidence="HIGH",
        reason="pre-existing",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_ms=42,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        publication_evidence=_publication_evidence(2, "AAA"),
    )

    # Client only has one response — for BBB
    client = _ScriptedClient([_stage2_drop()])

    tickers_with_scorecards = [
        ("AAA", "2026-04-10", {"pricing_zone": "MOS", "pricing_zone_detail": {}}),
        ("BBB", "2026-04-10", {"pricing_zone": "BLOCKED", "pricing_zone_detail": {}}),
    ]

    results = run_stage2(
        tmp_session_db,
        sweep_id,
        tickers_with_scorecards,
        client,
    )

    # Only 1 API call (BBB); AAA was skipped
    assert len(client.calls) == 1, f"Expected 1 API call, got {len(client.calls)}"

    # run_stage2 returns only the newly processed results (not the skipped one)
    assert len(results) == 1
    assert results[0].ticker == "BBB"

    # AAA's original row is untouched
    all_rows = {r["ticker"]: r for r in list_stage2_results(tmp_session_db, sweep_id)}
    assert all_rows["AAA"]["decision"] == "KEEP"
    assert all_rows["AAA"]["reason"] == "pre-existing"
    assert all_rows["AAA"]["confidence"] == "HIGH"


def test_stage3_skip_existing(tmp_session_db):
    """Pre-insert AAA with verdict=WATCH; run_stage3 with [AAA, BBB].
    Only 1 API call should be made (for BBB); AAA's row stays untouched.
    """
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_stage3_result,
        list_stage3_results,
    )
    from app.discover.stage3 import run_stage3

    ensure_schema(tmp_session_db)
    sweep_id = "sweep-resume-s3"
    create_sweep(tmp_session_db, sweep_id, universe_size=2, limit_applied=None, budget_usd=100.0)

    # Pre-insert AAA with verdict=WATCH (simulating a prior run)
    insert_stage3_result(
        tmp_session_db,
        sweep_id=sweep_id,
        ticker="AAA",
        verdict="WATCH",
        confidence="MODERATE",
        thesis_summary="pre-existing thesis",
        key_numbers=[],
        positives=[],
        risks=[],
        open_questions=[],
        reasoning_trace="pre-existing trace",
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.005,
        wall_ms=150,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        publication_evidence=_publication_evidence(3, "AAA"),
    )

    # Client only has one response — for BBB
    client = _ScriptedClient([_stage3_pass()])

    results = run_stage3(
        tmp_session_db,
        sweep_id,
        ["AAA", "BBB"],
        client,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
    )

    # Only 1 API call (BBB); AAA was skipped
    assert len(client.calls) == 1, f"Expected 1 API call, got {len(client.calls)}"

    # run_stage3 returns only the newly processed results
    assert len(results) == 1
    assert results[0].ticker == "BBB"

    # AAA's original row is untouched
    all_rows = {r["ticker"]: r for r in list_stage3_results(tmp_session_db, sweep_id)}
    assert all_rows["AAA"]["verdict"] == "WATCH"
    assert all_rows["AAA"]["thesis_summary"] == "pre-existing thesis"


def test_stage4_skip_existing(tmp_session_db):
    """Pre-insert AAA with verdict=BUY; run_stage4 with [AAA, BBB].
    Only 1 API call should be made (for BBB); AAA's row stays untouched.
    """
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_stage4_result,
        list_stage4_results,
    )
    from app.discover.stage4 import run_stage4

    ensure_schema(tmp_session_db)
    sweep_id = "sweep-resume-s4"
    create_sweep(tmp_session_db, sweep_id, universe_size=2, limit_applied=None, budget_usd=100.0)

    # Pre-insert AAA with verdict=BUY (simulating a prior run)
    insert_stage4_result(
        tmp_session_db,
        sweep_id=sweep_id,
        ticker="AAA",
        verdict="BUY",
        confidence="HIGH",
        thesis="pre-existing thesis",
        key_findings=[],
        open_questions=[],
        falsifiers=[],
        reasoning_trace="pre-existing trace",
        num_turns=1,
        tool_call_counts={},
        termination_reason="finalize_analysis",
        input_tokens=300,
        output_tokens=150,
        cost_usd=0.01,
        wall_seconds=0.5,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        financial_scope_manifest=[],
        publication_evidence=_publication_evidence(4, "AAA"),
    )

    # Client only has one response — for BBB
    client = _ScriptedClient([_stage4_finalize()])

    results = run_stage4(
        tmp_session_db,
        sweep_id,
        ["AAA", "BBB"],
        client,
        context_builder=lambda t: {"ticker": t, "user_message": f"=== {t} ==="},
        tool_dispatcher=lambda n, i: "tool output",
    )

    # Only 1 API call (BBB); AAA was skipped
    assert len(client.calls) == 1, f"Expected 1 API call, got {len(client.calls)}"

    # run_stage4 returns only the newly processed results
    assert len(results) == 1
    assert results[0].ticker == "BBB"

    # AAA's original row is untouched
    all_rows = {r["ticker"]: r for r in list_stage4_results(tmp_session_db, sweep_id)}
    assert all_rows["AAA"]["verdict"] == "BUY"
    assert all_rows["AAA"]["thesis"] == "pre-existing thesis"


# --- Resume tests ---


def test_resume_completes_remaining_stage3(tmp_session_db, tmp_path):
    """Sweep killed mid-Stage-3: resume completes remaining Stage 3 + Stage 4."""
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_sweep_universe,
        insert_stage2_result,
        insert_stage3_result,
        get_sweep,
        list_stage3_results,
        list_stage4_results,
    )
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 3, None, 100.0)
    insert_sweep_universe(
        tmp_session_db,
        "s1",
        [
            ("AAA", "2026-04-10"),
            ("BBB", "2026-04-10"),
            ("CCC", "2026-04-10"),
        ],
    )

    # Stage 2 fully complete: AAA=KEEP, BBB=KEEP, CCC=DROP
    for t, d in [("AAA", "KEEP"), ("BBB", "KEEP"), ("CCC", "DROP")]:
        insert_stage2_result(
            tmp_session_db,
            sweep_id="s1",
            ticker=t,
            decision=d,
            confidence="HIGH",
            reason="ok",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            wall_ms=100,
            financial_scope_fingerprint=_BOUND_SCOPE,
            financial_scope_publication_fingerprint=_BOUND_SCOPE,
            publication_evidence=_publication_evidence(2, t),
        )

    # Stage 3 partial: AAA done, BBB not done
    insert_stage3_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        verdict="WATCH",
        confidence="MODERATE",
        thesis_summary="pre",
        key_numbers=[],
        positives=[],
        risks=[],
        open_questions=[],
        reasoning_trace="pre",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        wall_ms=1000,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        publication_evidence=_publication_evidence(3, "AAA"),
    )

    # Mark as aborted (simulating crash)
    from app.discover.persistence import finalize_sweep

    finalize_sweep(tmp_session_db, "s1", "aborted")

    engine_db = _create_fake_engine_db(tmp_path, ["AAA", "BBB", "CCC"])

    # BBB needs Stage 3 (WATCH) + Stage 4
    # AAA needs Stage 4 only (already WATCH in Stage 3)
    client = _ScriptedClient(
        [
            _stage3_watch(),  # BBB Stage 3
            _stage4_finalize(),  # AAA Stage 4
            _stage4_finalize(),  # BBB Stage 4
        ]
    )

    resume_sweep(
        db_path=tmp_session_db,
        engine_db_path=engine_db,
        sweep_id="s1",
        client=client,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
        stage4_tool_dispatcher=lambda n, i: "output",
    )

    sweep = get_sweep(tmp_session_db, "s1")
    assert sweep["status"] == "completed"

    s3 = {r["ticker"]: r for r in list_stage3_results(tmp_session_db, "s1")}
    assert len(s3) == 2  # AAA + BBB
    assert s3["AAA"]["thesis_summary"] == "pre"  # Original untouched
    assert s3["BBB"]["verdict"] == "WATCH"

    s4 = {r["ticker"]: r for r in list_stage4_results(tmp_session_db, "s1")}
    assert len(s4) == 2  # AAA + BBB


def test_resume_completes_remaining_stage2(tmp_session_db, tmp_path):
    """Sweep killed mid-Stage-2: resume completes remaining Stage 2, then 3, then 4."""
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_sweep_universe,
        insert_stage2_result,
        get_sweep,
        list_stage2_results,
    )
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 3, None, 100.0)
    insert_sweep_universe(
        tmp_session_db,
        "s1",
        [
            ("AAA", "2026-04-10"),
            ("BBB", "2026-04-10"),
            ("CCC", "2026-04-10"),
        ],
    )

    # Stage 2 partial: only AAA done
    insert_stage2_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        decision="DROP",
        confidence="HIGH",
        reason="ok",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_ms=100,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        publication_evidence=_publication_evidence(2, "AAA"),
    )

    from app.discover.persistence import finalize_sweep

    finalize_sweep(tmp_session_db, "s1", "aborted")

    engine_db = _create_fake_engine_db(tmp_path, ["AAA", "BBB", "CCC"])

    # BBB -> KEEP, CCC -> DROP in Stage 2. BBB -> PASS in Stage 3.
    client = _ScriptedClient(
        [
            _stage2_keep(),  # BBB
            _stage2_drop(),  # CCC
            _stage3_pass(),  # BBB (only KEEP)
        ]
    )

    resume_sweep(
        db_path=tmp_session_db,
        engine_db_path=engine_db,
        sweep_id="s1",
        client=client,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
        stage4_tool_dispatcher=lambda n, i: "output",
    )

    sweep = get_sweep(tmp_session_db, "s1")
    assert sweep["status"] == "completed"
    s2 = {r["ticker"]: r for r in list_stage2_results(tmp_session_db, "s1")}
    assert len(s2) == 3
    assert s2["AAA"]["decision"] == "DROP"  # Original
    assert s2["BBB"]["decision"] == "KEEP"
    assert s2["CCC"]["decision"] == "DROP"


def test_resume_noop_when_all_done(tmp_session_db, tmp_path):
    """Resume on a completed sweep with all rows present is a no-op."""
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        get_sweep,
        insert_sweep_universe,
        insert_stage2_result,
    )
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 1, None, 100.0)
    insert_sweep_universe(tmp_session_db, "s1", [("AAA", "2026-04-10")])
    insert_stage2_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        decision="DROP",
        confidence="HIGH",
        reason="ok",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_ms=100,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        publication_evidence=_publication_evidence(2, "AAA"),
    )
    from app.discover.persistence import finalize_sweep

    finalize_sweep(tmp_session_db, "s1", "completed")

    engine_db = _create_fake_engine_db(tmp_path, ["AAA"])

    # No API calls expected
    client = _ScriptedClient([])

    resume_sweep(
        db_path=tmp_session_db,
        engine_db_path=engine_db,
        sweep_id="s1",
        client=client,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
        stage4_tool_dispatcher=lambda n, i: "output",
    )

    sweep = get_sweep(tmp_session_db, "s1")
    assert sweep["status"] == "completed"
    assert len(client.calls) == 0


def test_resume_lazy_backfill(tmp_session_db, tmp_path):
    """Pre-snapshot sweep resumes with backfilled universe from stage2_results."""
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_stage2_result,
        get_sweep_universe,
    )
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 2, None, 100.0)
    # NO insert_sweep_universe — simulates pre-snapshot sweep

    for t in ["AAA", "BBB"]:
        insert_stage2_result(
            tmp_session_db,
            sweep_id="s1",
            ticker=t,
            decision="DROP",
            confidence="HIGH",
            reason="ok",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            wall_ms=100,
            financial_scope_fingerprint=_BOUND_SCOPE,
            financial_scope_publication_fingerprint=_BOUND_SCOPE,
            publication_evidence=_publication_evidence(2, t),
        )
    from app.discover.persistence import finalize_sweep

    finalize_sweep(tmp_session_db, "s1", "aborted")

    engine_db = _create_fake_engine_db(tmp_path, ["AAA", "BBB"])

    client = _ScriptedClient([])  # All drops, no Stage 3

    resume_sweep(
        db_path=tmp_session_db,
        engine_db_path=engine_db,
        sweep_id="s1",
        client=client,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
        stage4_tool_dispatcher=lambda n, i: "output",
    )

    snapshot = get_sweep_universe(tmp_session_db, "s1")
    assert len(snapshot) == 2
    assert {t for t, _ in snapshot} == {"AAA", "BBB"}


def test_resume_running_sweep_raises(tmp_session_db, tmp_path):
    """Resuming a sweep with status 'running' raises RuntimeError."""
    from app.discover.persistence import ensure_schema, create_sweep
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 2, None, 10.0)

    engine_db = _create_fake_engine_db(tmp_path, [])
    client = _ScriptedClient([])

    with pytest.raises(RuntimeError, match="running"):
        resume_sweep(
            db_path=tmp_session_db,
            engine_db_path=engine_db,
            sweep_id="s1",
            client=client,
        )


def test_resume_missing_sweep_raises(tmp_session_db, tmp_path):
    """Resuming a non-existent sweep_id raises ValueError."""
    from app.discover.persistence import ensure_schema
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)

    engine_db = _create_fake_engine_db(tmp_path, [])
    client = _ScriptedClient([])

    with pytest.raises(ValueError, match="not found"):
        resume_sweep(
            db_path=tmp_session_db,
            engine_db_path=engine_db,
            sweep_id="nonexistent",
            client=client,
        )


def test_resume_budget_override(tmp_session_db, tmp_path):
    """budget_override updates sweeps.budget_usd to spent + override."""
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_sweep_universe,
        insert_stage2_result,
        add_sweep_cost,
        get_sweep,
    )
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 1, None, 25.0)
    insert_sweep_universe(tmp_session_db, "s1", [("AAA", "2026-04-10")])
    insert_stage2_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        decision="DROP",
        confidence="HIGH",
        reason="ok",
        input_tokens=100,
        output_tokens=50,
        cost_usd=5.0,
        wall_ms=100,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        publication_evidence=_publication_evidence(2, "AAA"),
    )
    add_sweep_cost(tmp_session_db, "s1", 5.0)

    from app.discover.persistence import finalize_sweep

    finalize_sweep(tmp_session_db, "s1", "aborted")

    engine_db = _create_fake_engine_db(tmp_path, ["AAA"])
    client = _ScriptedClient([])

    resume_sweep(
        db_path=tmp_session_db,
        engine_db_path=engine_db,
        sweep_id="s1",
        client=client,
        budget_override=50.0,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
        stage4_tool_dispatcher=lambda n, i: "output",
    )

    sweep = get_sweep(tmp_session_db, "s1")
    assert sweep["budget_usd"] == 55.0  # 5.0 spent + 50.0 override


def test_resume_budget_continuation(tmp_session_db, tmp_path):
    """No budget_override uses the existing sweeps.budget_usd."""
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_sweep_universe,
        insert_stage2_result,
        get_sweep,
    )
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 1, None, 25.0)
    insert_sweep_universe(tmp_session_db, "s1", [("AAA", "2026-04-10")])
    insert_stage2_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        decision="DROP",
        confidence="HIGH",
        reason="ok",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_ms=100,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        publication_evidence=_publication_evidence(2, "AAA"),
    )

    from app.discover.persistence import finalize_sweep

    finalize_sweep(tmp_session_db, "s1", "aborted")

    engine_db = _create_fake_engine_db(tmp_path, ["AAA"])
    client = _ScriptedClient([])

    resume_sweep(
        db_path=tmp_session_db,
        engine_db_path=engine_db,
        sweep_id="s1",
        client=client,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
        stage4_tool_dispatcher=lambda n, i: "output",
    )

    sweep = get_sweep(tmp_session_db, "s1")
    assert sweep["budget_usd"] == 25.0  # Unchanged


def test_resume_idempotent(tmp_session_db, tmp_path):
    """Calling resume twice on the same sweep is safe (second call is no-op)."""
    from app.discover.persistence import (
        ensure_schema,
        create_sweep,
        insert_sweep_universe,
        insert_stage2_result,
        get_sweep,
    )
    from app.discover.sweep import resume_sweep

    ensure_schema(tmp_session_db)
    create_sweep(tmp_session_db, "s1", 1, None, 100.0)
    insert_sweep_universe(tmp_session_db, "s1", [("AAA", "2026-04-10")])
    insert_stage2_result(
        tmp_session_db,
        sweep_id="s1",
        ticker="AAA",
        decision="DROP",
        confidence="HIGH",
        reason="ok",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_ms=100,
        financial_scope_fingerprint=_BOUND_SCOPE,
        financial_scope_publication_fingerprint=_BOUND_SCOPE,
        publication_evidence=_publication_evidence(2, "AAA"),
    )

    from app.discover.persistence import finalize_sweep

    finalize_sweep(tmp_session_db, "s1", "aborted")

    engine_db = _create_fake_engine_db(tmp_path, ["AAA"])

    for _ in range(2):
        client = _ScriptedClient([])
        resume_sweep(
            db_path=tmp_session_db,
            engine_db_path=engine_db,
            sweep_id="s1",
            client=client,
            bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
            stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
            stage4_tool_dispatcher=lambda n, i: "output",
        )

    sweep = get_sweep(tmp_session_db, "s1")
    assert sweep["status"] == "completed"
