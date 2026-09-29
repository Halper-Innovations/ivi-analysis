"""Tests for app.discover.stage2 — Haiku classifier production module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """These classifier tests isolate API parsing from the integrity contract."""

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
        "app.discover.stage2.bind_v1_financial_scope",
        _scope,
    )


# --- Fake Anthropic client ---


@dataclass
class _FakeUsage:
    input_tokens: int = 1500
    output_tokens: int = 200


@dataclass
class _FakeToolUseBlock:
    type: str = "tool_use"
    name: str = "classify_ticker"
    input: dict = field(
        default_factory=lambda: {
            "decision": "KEEP",
            "confidence": "MODERATE",
            "reason": "Meaningful valuation spread worth investigating",
        }
    )


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
        if not self._responses:
            raise AssertionError("FakeMessages ran out of canned responses")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


# --- Fixtures ---


@pytest.fixture
def tmp_session_db(tmp_path) -> Path:
    from app.discover.persistence import ensure_schema, create_sweep

    db = tmp_path / "discover_sessions.db"
    ensure_schema(db)
    create_sweep(db, "20260411-150000", universe_size=2, limit_applied=2, budget_usd=5.0)
    return db


def _sample_scorecard() -> dict[str, Any]:
    return {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "signal": "PROCEED",
        "pricing_zone_detail": {
            "current_price": 25.33,
            "market_cap": 1200.0,
            "dcf_base": 31.85,
            "epv_adjusted": 38.34,
            "margin_of_safety_vs_epv_adjusted": 0.3393,
            "gate_action": "PROCEED",
            "earnings_quality": "MODERATE",
            "epv_quality": "STABLE",
            "cycle_position": "MID",
            "revenue_cagr_5y": -0.0021,
            "revenue_cagr_3y": -0.0179,
        },
        "quality_context": {
            "confidence_class": "MODERATE",
            "allocation_grade": "C",
            "gate_action": "PROCEED",
            "leverage_stress": "MODERATE",
        },
    }


# --- Tests ---


def test_compact_scorecard_is_concise():
    from app.discover.stage2 import compact_scorecard

    text = compact_scorecard(_sample_scorecard())
    assert "MARGIN_OF_SAFETY" in text
    assert "current_price" in text
    assert "gate_action: PROCEED" in text
    # Must be short — target < 800 chars for Haiku budget
    assert len(text) < 1200


def test_classify_ticker_returns_decision_from_tool_use(tmp_session_db):
    from app.discover.stage2 import classify_ticker, Stage2Config

    responses = [
        _FakeMessage(content=[_FakeToolUseBlock()], usage=_FakeUsage(1500, 200)),
    ]
    client = _FakeClient(responses)

    result = classify_ticker(
        client=client,
        ticker="CCSI",
        as_of_date="2026-04-08",
        scorecard=_sample_scorecard(),
        config=Stage2Config(),
    )

    assert result.ticker == "CCSI"
    assert result.decision == "KEEP"
    assert result.confidence == "MODERATE"
    assert result.input_tokens == 1500
    assert result.output_tokens == 200
    assert result.cost_usd > 0
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5-20251001"


def test_classify_ticker_handles_api_error(tmp_session_db):
    from app.discover.stage2 import classify_ticker, Stage2Config

    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError("api boom")

    class _BoomClient:
        messages = _BoomMessages()

    result = classify_ticker(
        client=_BoomClient(),
        ticker="FAIL",
        as_of_date="2026-04-08",
        scorecard=_sample_scorecard(),
        config=Stage2Config(),
    )

    assert result.decision == "ERROR"
    assert "api boom" in (result.error or "")


def test_run_stage2_writes_results_to_db(tmp_session_db):
    from app.discover.persistence import list_stage2_results, get_sweep
    from app.discover.stage2 import run_stage2, Stage2Config

    # Two tickers, two canned responses
    responses = [
        _FakeMessage(
            content=[
                _FakeToolUseBlock(
                    input={
                        "decision": "KEEP",
                        "confidence": "MODERATE",
                        "reason": "Real MOS",
                    }
                )
            ],
            usage=_FakeUsage(1500, 200),
        ),
        _FakeMessage(
            content=[
                _FakeToolUseBlock(
                    input={
                        "decision": "DROP",
                        "confidence": "HIGH",
                        "reason": "Structural decline",
                    }
                )
            ],
            usage=_FakeUsage(1500, 200),
        ),
    ]
    client = _FakeClient(responses)

    tickers_with_scorecards = [
        ("CCSI", "2026-04-08", _sample_scorecard()),
        ("HBIO", "2026-04-08", _sample_scorecard()),
    ]

    run_stage2(
        db_path=tmp_session_db,
        sweep_id="20260411-150000",
        tickers_with_scorecards=tickers_with_scorecards,
        client=client,
        config=Stage2Config(),
    )

    rows = list_stage2_results(tmp_session_db, "20260411-150000")
    assert len(rows) == 2
    decisions = {r["ticker"]: r["decision"] for r in rows}
    assert decisions == {"CCSI": "KEEP", "HBIO": "DROP"}

    sweep = get_sweep(tmp_session_db, "20260411-150000")
    assert sweep["total_cost_usd"] == pytest.approx(sum(float(row["cost_usd"]) for row in rows))
    assert sweep["paid_attempt_count"] == 2
    assert sweep["reserved_attempt_cost_usd"] == 0.0
    assert sweep["unpublished_attempt_cost_usd"] == 0.0


def test_crash_after_provider_before_attempt_settlement_keeps_reserved_cost_and_blocks_resume(
    monkeypatch,
    tmp_session_db,
):
    import sqlite3

    import app.discover.persistence as persistence
    from app.discover.persistence import (
        DiscoverPaidAttemptAmbiguousError,
        get_sweep,
        list_stage2_results,
    )
    from app.discover.stage2 import Stage2Config, run_stage2

    response = _FakeMessage(
        content=[_FakeToolUseBlock()],
        usage=_FakeUsage(1500, 200),
    )
    client = _FakeClient([response])
    real_complete = persistence._complete_discover_paid_attempt

    def crash_before_settlement(*_args, **_kwargs):
        raise RuntimeError("simulated crash before paid-attempt settlement")

    monkeypatch.setattr(
        persistence,
        "_complete_discover_paid_attempt",
        crash_before_settlement,
    )

    with pytest.raises(DiscoverPaidAttemptAmbiguousError, match="still RESERVED"):
        run_stage2(
            db_path=tmp_session_db,
            sweep_id="20260411-150000",
            tickers_with_scorecards=[
                ("CCSI", "2026-04-08", _sample_scorecard()),
            ],
            client=client,
            config=Stage2Config(),
        )

    assert len(client.messages.calls) == 1
    conn = sqlite3.connect(tmp_session_db)
    conn.row_factory = sqlite3.Row
    try:
        attempt = conn.execute(
            """
            SELECT status, estimated_cost_usd, accounted_cost_usd
            FROM discover_paid_attempts
            WHERE stage = 2 AND sweep_id = '20260411-150000' AND ticker = 'CCSI'
            """
        ).fetchone()
    finally:
        conn.close()
    assert attempt is not None
    assert attempt["status"] == "RESERVED"
    assert attempt["accounted_cost_usd"] == 0.0
    assert attempt["estimated_cost_usd"] > 0.0
    assert list_stage2_results(tmp_session_db, "20260411-150000") == []

    sweep = get_sweep(tmp_session_db, "20260411-150000")
    assert sweep is not None
    assert sweep["total_cost_usd"] == pytest.approx(attempt["estimated_cost_usd"])
    assert sweep["reserved_attempt_cost_usd"] == pytest.approx(attempt["estimated_cost_usd"])
    assert sweep["unpublished_attempt_cost_usd"] == pytest.approx(attempt["estimated_cost_usd"])

    monkeypatch.setattr(
        persistence,
        "_complete_discover_paid_attempt",
        real_complete,
    )
    resume_client = _FakeClient([response])
    with pytest.raises(
        DiscoverPaidAttemptAmbiguousError,
        match="durable paid-attempt history",
    ):
        run_stage2(
            db_path=tmp_session_db,
            sweep_id="20260411-150000",
            tickers_with_scorecards=[
                ("CCSI", "2026-04-08", _sample_scorecard()),
            ],
            client=resume_client,
            config=Stage2Config(),
        )
    assert resume_client.messages.calls == []


def test_durable_discover_client_rejects_sdk_retry_configuration(tmp_session_db):
    from app.discover.persistence import DurableDiscoverClient

    client = _FakeClient([])
    client.max_retries = 2

    with pytest.raises(ValueError, match="max_retries must be zero"):
        DurableDiscoverClient(
            client,
            db_path=tmp_session_db,
            stage=2,
            sweep_id="20260411-150000",
            ticker="CCSI",
            input_usd_per_mtok=1.0,
            output_usd_per_mtok=5.0,
        )


def test_failed_discover_attempt_is_charged_once_and_published(tmp_session_db):
    import sqlite3

    from app.discover.persistence import get_sweep, list_stage2_results
    from app.discover.stage2 import Stage2Config, run_stage2

    class FailingMessages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("provider failed after dispatch")

    class FailingClient:
        def __init__(self):
            self.messages = FailingMessages()

    client = FailingClient()
    results = run_stage2(
        db_path=tmp_session_db,
        sweep_id="20260411-150000",
        tickers_with_scorecards=[
            ("CCSI", "2026-04-08", _sample_scorecard()),
        ],
        client=client,
        config=Stage2Config(),
    )

    assert len(client.messages.calls) == 1
    assert len(results) == 1
    assert results[0].decision == "ERROR"
    assert results[0].cost_usd > 0.0
    rows = list_stage2_results(tmp_session_db, "20260411-150000")
    assert len(rows) == 1
    assert rows[0]["decision"] == "ERROR"
    assert rows[0]["cost_usd"] == results[0].cost_usd

    conn = sqlite3.connect(tmp_session_db)
    conn.row_factory = sqlite3.Row
    try:
        attempt = conn.execute(
            """
            SELECT status, outcome, estimated_cost_usd, accounted_cost_usd
            FROM discover_paid_attempts
            WHERE stage = 2 AND sweep_id = '20260411-150000' AND ticker = 'CCSI'
            """
        ).fetchone()
    finally:
        conn.close()
    assert attempt is not None
    assert attempt["status"] == "PUBLISHED"
    assert attempt["outcome"] == "ERROR"
    assert attempt["accounted_cost_usd"] == attempt["estimated_cost_usd"]

    sweep = get_sweep(tmp_session_db, "20260411-150000")
    assert sweep is not None
    assert sweep["total_cost_usd"] == rows[0]["cost_usd"]
    assert sweep["reserved_attempt_cost_usd"] == 0.0
    assert sweep["unpublished_attempt_cost_usd"] == 0.0


def test_overvaluation_sanity_check_overrides_keep():
    """AAON-style case: price $83 >> DCF $30 and EPV $15. KEEP should be overridden to DROP."""
    from app.discover.stage2 import _overvaluation_sanity_check, Stage2Result

    result = Stage2Result(
        ticker="AAON",
        as_of_date="2026-04-10",
        decision="KEEP",
        confidence="MODERATE",
        reason="deep valuation discount",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.002,
        wall_ms=2000,
    )
    scorecard = {
        "pricing_zone_detail": {
            "current_price": 83.40,
            "dcf_base": 30.0,
            "epv_adjusted": 15.06,
        }
    }

    checked = _overvaluation_sanity_check(result, scorecard)
    assert checked.decision == "DROP"
    assert "SANITY_OVERRIDE" in checked.reason
    assert checked.ticker == "AAON"


def test_overvaluation_sanity_check_passes_genuine_undervaluation():
    """Price below intrinsic values: KEEP should NOT be overridden."""
    from app.discover.stage2 import _overvaluation_sanity_check, Stage2Result

    result = Stage2Result(
        ticker="INTC",
        as_of_date="2026-04-10",
        decision="KEEP",
        confidence="MODERATE",
        reason="genuine discount",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.002,
        wall_ms=2000,
    )
    scorecard = {
        "pricing_zone_detail": {
            "current_price": 25.0,
            "dcf_base": 40.0,
            "epv_adjusted": 35.0,
        }
    }

    checked = _overvaluation_sanity_check(result, scorecard)
    assert checked.decision == "KEEP"
    assert checked.reason == "genuine discount"


def test_overvaluation_sanity_check_ignores_drops():
    """DROP decisions should pass through unchanged."""
    from app.discover.stage2 import _overvaluation_sanity_check, Stage2Result

    result = Stage2Result(
        ticker="BAD",
        as_of_date="2026-04-10",
        decision="DROP",
        confidence="HIGH",
        reason="decline",
        input_tokens=1500,
        output_tokens=200,
        cost_usd=0.002,
        wall_ms=2000,
    )
    scorecard = {
        "pricing_zone_detail": {"current_price": 100.0, "dcf_base": 10.0, "epv_adjusted": 5.0}
    }

    checked = _overvaluation_sanity_check(result, scorecard)
    assert checked.decision == "DROP"
    assert checked.reason == "decline"
