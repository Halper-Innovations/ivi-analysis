"""Tests for app.discover.sweep — four-stage orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """Orchestrator tests use synthetic scorecards; gate behavior is separate."""

    from app.autonomous.v1_financial_context import BoundV1FinancialScope

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
            expected_scope_fingerprint="a" * 64,
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


def test_fast_bundle_builder_threads_canonical_parent_packet(monkeypatch):
    from app.discover.sweep import _make_fast_bundle_builder

    packet = object()
    captured: dict[str, Any] = {}

    def fake_cached_builder(**kwargs):
        captured.update(kwargs)
        return "bundle"

    monkeypatch.setattr(
        "app.analyst.bundle_builder.build_analysis_evidence_bundle_from_cached_scorecard",
        fake_cached_builder,
    )
    builder = _make_fast_bundle_builder(
        {"TEST": ("2026-04-10", {"pricing_zone": "MOS"})},
        {"TEST": packet},
    )

    assert builder("test", as_of_date="2026-04-10") == "bundle"
    assert captured["ticker"] == "test"
    assert captured["financial_packet"] is packet


@pytest.fixture
def tmp_session_db(tmp_path) -> Path:
    return tmp_path / "discover_sessions.db"


def test_full_sweep_runs_all_four_stages(tmp_session_db):
    from app.discover.persistence import (
        get_sweep,
        list_stage2_results,
        list_stage3_results,
        list_stage4_results,
    )
    from app.discover.sweep import run_sweep

    universe = [
        ("CCSI", "2026-04-08", {"pricing_zone": "MOS", "pricing_zone_detail": {}}),
        ("HBIO", "2026-04-08", {"pricing_zone": "BLOCKED", "pricing_zone_detail": {}}),
    ]

    responses = [
        _stage2_keep(),  # CCSI
        _stage2_drop(),  # HBIO
        _stage3_watch(),  # CCSI
        _stage4_finalize(),  # CCSI
    ]
    client = _ScriptedClient(responses)

    def fake_bundle_builder(ticker: str, as_of_date: str | None = None):
        return _sample_bundle(ticker)

    def fake_context_builder(ticker: str) -> dict[str, Any]:
        return {"ticker": ticker, "user_message": f"=== {ticker} ==="}

    def fake_tool_dispatcher(name: str, tool_input: dict) -> str:
        return "tool output"

    sweep_id = run_sweep(
        db_path=tmp_session_db,
        universe=universe,
        limit=None,
        budget_usd=100.0,
        client=client,
        bundle_builder=fake_bundle_builder,
        stage4_context_builder=fake_context_builder,
        stage4_tool_dispatcher=fake_tool_dispatcher,
    )

    sweep = get_sweep(tmp_session_db, sweep_id)
    assert sweep is not None
    assert sweep["status"] == "completed"
    assert sweep["universe_size"] == 2
    assert sweep["total_cost_usd"] > 0

    s2 = {r["ticker"]: r for r in list_stage2_results(tmp_session_db, sweep_id)}
    assert s2["CCSI"]["decision"] == "KEEP"
    assert s2["HBIO"]["decision"] == "DROP"

    s3 = {r["ticker"]: r for r in list_stage3_results(tmp_session_db, sweep_id)}
    assert "CCSI" in s3
    assert "HBIO" not in s3
    assert s3["CCSI"]["verdict"] == "WATCH"

    s4 = {r["ticker"]: r for r in list_stage4_results(tmp_session_db, sweep_id)}
    assert "CCSI" in s4
    assert s4["CCSI"]["verdict"] == "BUY"


def test_sweep_with_limit_flag(tmp_session_db):
    from app.discover.persistence import list_stage2_results
    from app.discover.sweep import run_sweep

    universe = [
        ("A", "2026-04-08", {}),
        ("B", "2026-04-08", {}),
        ("C", "2026-04-08", {}),
        ("D", "2026-04-08", {}),
        ("E", "2026-04-08", {}),
    ]

    responses = [_stage2_drop(), _stage2_drop()]
    client = _ScriptedClient(responses)

    sweep_id = run_sweep(
        db_path=tmp_session_db,
        universe=universe,
        limit=2,
        budget_usd=100.0,
        client=client,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
        stage4_tool_dispatcher=lambda n, i: "",
    )

    s2_rows = list_stage2_results(tmp_session_db, sweep_id)
    assert len(s2_rows) == 2


def test_sweep_skips_stages_3_4_when_no_keeps(tmp_session_db):
    from app.discover.persistence import (
        list_stage2_results,
        list_stage3_results,
        list_stage4_results,
    )
    from app.discover.sweep import run_sweep

    universe = [("HBIO", "2026-04-08", {})]
    responses = [_stage2_drop()]
    client = _ScriptedClient(responses)

    sweep_id = run_sweep(
        db_path=tmp_session_db,
        universe=universe,
        limit=None,
        budget_usd=100.0,
        client=client,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        stage4_context_builder=lambda t: {"ticker": t, "user_message": t},
        stage4_tool_dispatcher=lambda n, i: "",
    )

    assert len(list_stage2_results(tmp_session_db, sweep_id)) == 1
    assert list_stage3_results(tmp_session_db, sweep_id) == []
    assert list_stage4_results(tmp_session_db, sweep_id) == []


def test_sweep_fast_path_skips_slow_ensure_chain(tmp_session_db, monkeypatch):
    """When no bundle_builder override is passed, the sweep must derive a
    fast-path builder that calls build_analysis_evidence_bundle_from_cached_scorecard
    and NOT the slow ensure_all_facts + ensure_valuation + _load_scorecard chain.

    Regression test for the 772-ticker sweep performance bug: the default
    bundle_builder was calling build_analysis_evidence_bundle which re-ran
    ensure_valuation per ticker, taking 5-6 minutes of CPU each.
    """
    from app.discover.persistence import list_stage3_results
    from app.discover.sweep import run_sweep

    # Trap the slow chain — if any of these get called the sweep has
    # regressed back to the pre-fix behavior.
    def trap_ensure_all_facts(*args, **kwargs):
        raise AssertionError("sweep fast path must NOT call ensure_all_facts per Stage 3 ticker")

    def trap_ensure_valuation(*args, **kwargs):
        raise AssertionError("sweep fast path must NOT call ensure_valuation per Stage 3 ticker")

    def trap_load_scorecard(*args, **kwargs):
        raise AssertionError("sweep fast path must NOT call _load_scorecard per Stage 3 ticker")

    monkeypatch.setattr("app.analyst.bundle_builder.ensure_all_facts", trap_ensure_all_facts)
    monkeypatch.setattr("app.analyst.bundle_builder.ensure_valuation", trap_ensure_valuation)
    monkeypatch.setattr("app.analyst.bundle_builder._load_scorecard", trap_load_scorecard)

    # Stub out the remaining IO that the fast path still runs (docket,
    # scanners, tensions, filings, adapter).
    from dataclasses import dataclass as _dc

    @_dc
    class _Span:
        section_label: str
        text: str

    monkeypatch.setattr(
        "app.analyst.bundle_builder.collect_10k_docket",
        lambda **kw: [],
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.assess_solvency",
        lambda t: None,
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.scan_filing_risks",
        lambda t, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder._compute_tensions_from_scorecard",
        lambda sc, qc: {},
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.read_filing_text",
        lambda f: None,
    )
    monkeypatch.setattr(
        "app.analyst.bundle_builder.segment_10k_sections",
        lambda t: [],
    )

    # SecExhibitsAdapter needs an enabled() that returns False so we skip
    # the whole adapter path without touching the network.
    class _DisabledAdapter:
        def enabled(self):
            return False

        def collect(self, ctx):
            return None

    monkeypatch.setattr(
        "app.analyst.bundle_builder.SecExhibitsAdapter",
        lambda cfg=None: _DisabledAdapter(),
    )

    universe = [
        ("CCSI", "2026-04-08", {"pricing_zone": "MOS", "pricing_zone_detail": {}}),
    ]
    responses = [
        _stage2_keep(),  # CCSI Stage 2 → KEEP
        _stage3_pass(),  # CCSI Stage 3 → PASS (so Stage 4 is skipped)
    ]
    client = _ScriptedClient(responses)

    # DO NOT pass bundle_builder — force the sweep to derive the fast path
    sweep_id = run_sweep(
        db_path=tmp_session_db,
        universe=universe,
        limit=None,
        budget_usd=100.0,
        client=client,
    )

    # If we got here, none of the traps fired. Verify Stage 3 still ran.
    rows = list_stage3_results(tmp_session_db, sweep_id)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "CCSI"
    assert rows[0]["verdict"] == "PASS"
