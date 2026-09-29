"""Tests for app.discover.stage3 — Sonnet mid-depth researcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import pytest

from app.analyst.evidence_bundle import (
    AnalysisEvidenceBundle,
    BundleFiling,
    ValuationSnapshot,
)


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """These researcher tests isolate API parsing from the integrity contract."""

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
        "app.discover.stage3.bind_v1_financial_scope",
        _scope,
    )


@dataclass
class _FakeUsage:
    input_tokens: int = 4200
    output_tokens: int = 1750


@dataclass
class _FakeToolUseBlock:
    type: str = "tool_use"
    name: str = "stage3_analysis"
    input: dict = field(
        default_factory=lambda: {
            "verdict": "WATCH",
            "confidence": "MODERATE",
            "thesis_summary": "Classic value setup worth investigating",
            "key_numbers": ["DCF $70.76", "EPV $30.58"],
            "positives": ["Strong cash flow"],
            "risks": ["Elevated debt"],
            "open_questions": ["Current price?"],
            "reasoning_trace": "Weighted evidence favorably",
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
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


@pytest.fixture
def tmp_session_db(tmp_path) -> Path:
    from app.discover.persistence import ensure_schema, create_sweep

    db = tmp_path / "discover_sessions.db"
    ensure_schema(db)
    create_sweep(db, "20260411-150000", universe_size=2, limit_applied=2, budget_usd=5.0)
    return db


def _sample_bundle(ticker: str) -> AnalysisEvidenceBundle:
    snapshot = ValuationSnapshot(
        current_price=45.50,
        market_cap=12000.0,
        dcf_base=62.10,
        epv_adjusted=48.75,
        graham_value=37.90,
        methods_agree=True,
        tension_type="NONE",
        gate_action="PROCEED",
        solvency_status="LOW",
        filing_risk_status="OK",
    )
    filing = BundleFiling(
        form_type="10-K",
        filing_date="2025-02-14",
        accession="0001234567-25-000001",
        role="annual",
        sections_included=["mda", "risk_factors"],
        section_text={"mda": "MDA body text", "risk_factors": "Risk body text"},
    )
    return AnalysisEvidenceBundle(
        ticker=ticker,
        as_of_date="2026-04-10",
        built_at="2026-04-10T12:00:00+00:00",
        analysis_years=5,
        analysis_quarters=0,
        freshness_window_days=90,
        valuation=snapshot,
        filings=[filing],
    )


def test_build_stage3_prompt_includes_key_fields():
    from app.discover.stage3 import build_stage3_prompt

    bundle = _sample_bundle("CCSI")
    prompt = build_stage3_prompt(bundle)
    assert "CCSI" in prompt
    assert "current_price: 45.5" in prompt
    assert "dcf_base: 62.1" in prompt
    assert "10-K" in prompt
    assert "MDA body text" in prompt


def test_research_ticker_writes_structured_verdict(tmp_session_db):
    from app.discover.stage3 import research_ticker, Stage3Config

    responses = [
        _FakeMessage(content=[_FakeToolUseBlock()], usage=_FakeUsage(4200, 1750)),
    ]
    client = _FakeClient(responses)

    def fake_bundle_builder(ticker: str, as_of_date: str | None = None):
        return _sample_bundle(ticker)

    result = research_ticker(
        client=client,
        ticker="CCSI",
        config=Stage3Config(),
        bundle_builder=fake_bundle_builder,
    )

    assert result.ticker == "CCSI"
    assert result.verdict == "WATCH"
    assert result.confidence == "MODERATE"
    assert result.input_tokens == 4200
    assert result.output_tokens == 1750
    assert result.cost_usd > 0
    assert client.messages.calls[0]["model"] == "claude-sonnet-4-6"


def test_research_ticker_handles_bundle_build_failure(tmp_session_db):
    from app.discover.stage3 import research_ticker, Stage3Config

    def broken_builder(ticker: str, as_of_date: str | None = None):
        raise RuntimeError("bundle broke")

    class _NoClient:
        pass

    result = research_ticker(
        client=_NoClient(),
        ticker="FAIL",
        config=Stage3Config(),
        bundle_builder=broken_builder,
    )
    assert result.verdict == "ERROR"
    assert "bundle broke" in (result.error or "")


def test_run_stage3_writes_results_to_db(tmp_session_db):
    from app.discover.persistence import list_stage3_results
    from app.discover.stage3 import run_stage3, Stage3Config

    responses = [
        _FakeMessage(
            content=[
                _FakeToolUseBlock(
                    input={
                        "verdict": "WATCH",
                        "confidence": "MODERATE",
                        "thesis_summary": "Worth a look",
                        "key_numbers": ["MOS 32%"],
                        "positives": ["Strong cash flow"],
                        "risks": ["Elevated debt"],
                        "open_questions": ["current price?"],
                        "reasoning_trace": "Balanced",
                    }
                )
            ],
            usage=_FakeUsage(4200, 1750),
        ),
        _FakeMessage(
            content=[
                _FakeToolUseBlock(
                    input={
                        "verdict": "PASS",
                        "confidence": "HIGH",
                        "thesis_summary": "Clear value trap",
                        "key_numbers": ["negative EPV"],
                        "positives": [],
                        "risks": ["structural decline"],
                        "open_questions": [],
                        "reasoning_trace": "Strong pass signal",
                    }
                )
            ],
            usage=_FakeUsage(4200, 1750),
        ),
    ]
    client = _FakeClient(responses)

    def fake_bundle_builder(ticker: str, as_of_date: str | None = None):
        return _sample_bundle(ticker)

    run_stage3(
        db_path=tmp_session_db,
        sweep_id="20260411-150000",
        tickers=["CCSI", "HBIO"],
        client=client,
        config=Stage3Config(),
        bundle_builder=fake_bundle_builder,
    )

    rows = list_stage3_results(tmp_session_db, "20260411-150000")
    assert len(rows) == 2
    verdicts = {r["ticker"]: r["verdict"] for r in rows}
    assert verdicts == {"CCSI": "WATCH", "HBIO": "PASS"}
