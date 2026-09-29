"""Tests for app.discover.stage4 — Sonnet deep loop production module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """These loop tests isolate tool orchestration from the integrity contract."""

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

    monkeypatch.setattr(
        "app.discover.stage4.bind_v1_financial_scope",
        _scope,
    )


@dataclass
class _FakeUsage:
    input_tokens: int = 3000
    output_tokens: int = 800


@dataclass
class _FakeTextBlock:
    type: str = "text"
    text: str = "reasoning text"


@dataclass
class _FakeToolUseBlock:
    type: str = "tool_use"
    id: str = "tu_1"
    name: str = "fetch_filing_section"
    input: dict = field(default_factory=dict)


@dataclass
class _FakeMessage:
    content: list
    usage: _FakeUsage
    stop_reason: str = "tool_use"


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


@pytest.fixture
def tmp_session_db(tmp_path) -> Path:
    from app.discover.persistence import ensure_schema, create_sweep

    db = tmp_path / "discover_sessions.db"
    ensure_schema(db)
    create_sweep(db, "20260411-150000", universe_size=1, limit_applied=1, budget_usd=5.0)
    return db


def _finalize_block(
    verdict: str = "BUY",
    confidence: str = "HIGH",
    thesis: str = "Clear value setup",
) -> _FakeToolUseBlock:
    return _FakeToolUseBlock(
        id="tu_final",
        name="finalize_analysis",
        input={
            "verdict": verdict,
            "confidence": confidence,
            "thesis": thesis,
            "key_findings": ["MOS 32%"],
            "open_questions": ["FY ARR trajectory?"],
            "falsifiers": ["Revenue decline >10%"],
            "reasoning_trace": "Strong conviction",
        },
    )


def test_deep_loop_terminates_on_finalize(tmp_session_db):
    from app.discover.stage4 import deep_research_ticker, Stage4Config

    responses = [
        _FakeMessage(
            content=[_finalize_block()],
            usage=_FakeUsage(3000, 800),
            stop_reason="tool_use",
        ),
    ]
    client = _FakeClient(responses)

    def fake_context_builder(ticker: str) -> dict[str, Any]:
        return {"ticker": ticker, "user_message": f"=== {ticker} ==="}

    def fake_tool_dispatcher(name: str, tool_input: dict) -> str:
        return "tool result"

    result = deep_research_ticker(
        client=client,
        ticker="CCSI",
        config=Stage4Config(),
        context_builder=fake_context_builder,
        tool_dispatcher=fake_tool_dispatcher,
    )

    assert result.verdict == "BUY"
    assert result.confidence == "HIGH"
    assert result.num_turns == 1
    assert result.termination_reason == "finalize_analysis"


def test_deep_loop_dispatches_tool_then_finalizes(tmp_session_db):
    from app.discover.stage4 import deep_research_ticker, Stage4Config

    tool_call_block = _FakeToolUseBlock(
        id="tu_1",
        name="fetch_historical_scorecards",
        input={"ticker": "CCSI", "n_years": 3},
    )
    responses = [
        _FakeMessage(content=[tool_call_block], usage=_FakeUsage(3000, 300)),
        _FakeMessage(content=[_finalize_block(verdict="WATCH")], usage=_FakeUsage(5000, 700)),
    ]
    client = _FakeClient(responses)

    dispatched: list[tuple[str, dict]] = []

    def fake_context_builder(ticker: str) -> dict[str, Any]:
        return {"ticker": ticker, "user_message": f"=== {ticker} ==="}

    def fake_tool_dispatcher(name: str, tool_input: dict) -> str:
        dispatched.append((name, dict(tool_input)))
        return "2026-04-08: price=25.33"

    result = deep_research_ticker(
        client=client,
        ticker="CCSI",
        config=Stage4Config(),
        context_builder=fake_context_builder,
        tool_dispatcher=fake_tool_dispatcher,
    )

    assert result.verdict == "WATCH"
    assert result.num_turns == 2
    assert dispatched == [("fetch_historical_scorecards", {"ticker": "CCSI", "n_years": 3})]
    assert result.tool_call_counts == {
        "fetch_historical_scorecards": 1,
        "finalize_analysis": 1,
    }


def test_deep_loop_respects_budget_cap(tmp_session_db):
    from app.discover.stage4 import deep_research_ticker, Stage4Config

    tool_block = _FakeToolUseBlock(
        id="tu_1",
        name="fetch_filing_section",
        input={"ticker": "CCSI", "section_key": "mda"},
    )
    responses = [
        _FakeMessage(
            content=[tool_block],
            usage=_FakeUsage(input_tokens=100_000_000, output_tokens=1),
        ),
    ]
    client = _FakeClient(responses)

    def fake_context_builder(ticker: str) -> dict[str, Any]:
        return {"ticker": ticker, "user_message": f"=== {ticker} ==="}

    def fake_tool_dispatcher(name: str, tool_input: dict) -> str:
        return ""

    result = deep_research_ticker(
        client=client,
        ticker="CCSI",
        config=Stage4Config(max_cost_usd=0.50),
        context_builder=fake_context_builder,
        tool_dispatcher=fake_tool_dispatcher,
    )

    assert result.termination_reason == "budget_cap"


def test_run_stage4_writes_results_to_db(tmp_session_db):
    from app.discover.persistence import list_stage4_results
    from app.discover.stage4 import run_stage4, Stage4Config

    responses = [
        _FakeMessage(content=[_finalize_block(verdict="BUY")], usage=_FakeUsage(3000, 800)),
        _FakeMessage(
            content=[_finalize_block(verdict="PASS", confidence="HIGH", thesis="Clear pass")],
            usage=_FakeUsage(3000, 800),
        ),
    ]
    client = _FakeClient(responses)

    def fake_context_builder(ticker: str) -> dict[str, Any]:
        return {"ticker": ticker, "user_message": f"=== {ticker} ==="}

    def fake_tool_dispatcher(name: str, tool_input: dict) -> str:
        return ""

    run_stage4(
        db_path=tmp_session_db,
        sweep_id="20260411-150000",
        tickers=["CCSI", "HBIO"],
        client=client,
        config=Stage4Config(),
        context_builder=fake_context_builder,
        tool_dispatcher=fake_tool_dispatcher,
    )

    rows = list_stage4_results(tmp_session_db, "20260411-150000")
    assert len(rows) == 2
    verdicts = {r["ticker"]: r["verdict"] for r in rows}
    assert verdicts == {"CCSI": "BUY", "HBIO": "PASS"}
