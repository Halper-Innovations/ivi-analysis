from __future__ import annotations

from app.autonomous.run_contract import EvidenceReference, ToolCallRecord
from app.autonomous.sector_contract import (
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
)
from app.autonomous.sector_runtime import (
    AUDIT_SIGNAL_CLASSIFICATION,
    _selection_audit_for_ticker,
)


# --- Test harness mirroring tests/test_data_incomplete_status.py. ---


def _packet(
    ticker: str = "AAA",
    *,
    expectations_gap_bucket: str | None = None,
) -> SectorCompanyFinancialPacket:
    valuation: dict[str, object] = {"valuation_anchor": 100.0, "anchor_method": "dcf"}
    if expectations_gap_bucket is not None:
        valuation["expectations_gap_bucket"] = expectations_gap_bucket
        valuation["expectations_gap"] = -0.06
        valuation["implied_growth"] = 0.04
        valuation["supportable_growth"] = 0.10
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status="OK",
        current_price=50.0,
        valuation=valuation,
    )


def _base_scenario(
    ticker: str = "AAA",
    annualized_return: float = 0.18,
) -> SectorExpectedReturnScenario:
    return SectorExpectedReturnScenario(
        scenario_id=f"{ticker}_base_5Y",
        ticker=ticker,
        scenario_name="base",
        horizon_years=5,
        current_price=50.0,
        estimated_future_value_per_share=100.0,
        annualized_return=annualized_return,
    )


def _downside_scenario(
    ticker: str = "AAA",
    annualized_return: float = -0.02,
) -> SectorExpectedReturnScenario:
    return SectorExpectedReturnScenario(
        scenario_id=f"{ticker}_downside_5Y",
        ticker=ticker,
        scenario_name="downside",
        horizon_years=5,
        current_price=50.0,
        estimated_future_value_per_share=45.0,
        annualized_return=annualized_return,
    )


def _passing_tool_evidence() -> tuple[list[ToolCallRecord], list[EvidenceReference]]:
    tool_calls = [
        ToolCallRecord("TC1", "rank_expected_return_cases", {"tickers": ["AAA"], "horizon_years": 5}, "Rank.", status="OK", evidence_ref_ids=["E1"]),
        ToolCallRecord("TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]),
        ToolCallRecord("TC3", "analyze_liquidity_stress", {}, "Validate liquidity.", status="OK", evidence_ref_ids=["E3"]),
    ]
    evidence = [
        EvidenceReference("E1", "tool_output", "rank_expected_return_cases", "AAA ranked first.", "AAA", tool_call_id="TC1", confidence="MODERATE"),
        EvidenceReference("E2", "tool_output", "fetch_kpi_trends", "AAA KPI evidence.", "AAA", tool_call_id="TC2", confidence="MODERATE"),
        EvidenceReference("E3", "tool_output", "analyze_liquidity_stress", "AAA liquidity evidence.", "AAA", tool_call_id="TC3", confidence="MODERATE"),
    ]
    return tool_calls, evidence


def _audit(packet: SectorCompanyFinancialPacket) -> dict[str, object]:
    tool_calls, evidence = _passing_tool_evidence()
    return _selection_audit_for_ticker(
        selected_ticker=packet.ticker,
        packets_by_ticker={packet.ticker: packet},
        scenarios=[_base_scenario(packet.ticker, 0.18), _downside_scenario(packet.ticker, -0.02)],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )


# --- sector selection tests. ---


def test_expensive_vs_expectations_is_classified_hurdle():
    assert AUDIT_SIGNAL_CLASSIFICATION["EXPENSIVE_VS_EXPECTATIONS"] == "HURDLE"


def test_expensive_vs_expectations_is_a_binding_hurdle_cap():
    # An EXPENSIVE_VS_EXPECTATIONS bucket on an otherwise-clean packet is a
    # binding HURDLE-class confidence cap that downgrades to WATCHLIST_ONLY.
    audit = _audit(_packet("AAA", expectations_gap_bucket="EXPENSIVE_VS_EXPECTATIONS"))

    assert audit["status"] == "WATCHLIST_ONLY"
    assert "EXPENSIVE_VS_EXPECTATIONS" in audit["confidence_caps"]


def test_cheap_vs_expectations_passes_with_positive_note():
    # CHEAP_VS_EXPECTATIONS is never a blocker: the candidate stays an actionable
    # PASS, the returned dict carries the bucket, and a note flags the cheap-vs-
    # expectations framing.
    audit = _audit(_packet("AAA", expectations_gap_bucket="CHEAP_VS_EXPECTATIONS"))

    assert audit["status"] == "PASS"
    assert audit["expectations_gap_bucket"] == "CHEAP_VS_EXPECTATIONS"
    assert any("cheap vs expectations" in str(note).lower() for note in audit["notes"])


def test_unreliable_expectations_gap_adds_no_cap():
    # An EXPECTATIONS_GAP_UNRELIABLE bucket is silent: it adds no expectations
    # cap and leaves the status identical to the same packet without any bucket.
    unreliable = _audit(_packet("AAA", expectations_gap_bucket="EXPECTATIONS_GAP_UNRELIABLE"))
    no_bucket = _audit(_packet("AAA"))

    assert "EXPENSIVE_VS_EXPECTATIONS" not in unreliable["confidence_caps"]
    assert unreliable["status"] == no_bucket["status"]
    assert unreliable["status"] == "PASS"


def test_expensive_vs_expectations_does_not_override_a_hard_blocker(monkeypatch):
    # EXPENSIVE_VS_EXPECTATIONS is a HURDLE cap that downgrades to
    # WATCHLIST_ONLY, but it sits ALONGSIDE the safety gates and must never soften
    # a stronger binding hard blocker. Here a below-hurdle base return in 'hard'
    # mode is the hard blocker (-> BLOCKED); the EXPENSIVE bucket must NOT relax
    # BLOCKED to WATCHLIST_ONLY (the precedence is hard-blocker > hurdle-cap).
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    tool_calls, evidence = _passing_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={
            "AAA": _packet("AAA", expectations_gap_bucket="EXPENSIVE_VS_EXPECTATIONS")
        },
        scenarios=[_base_scenario("AAA", 0.05), _downside_scenario("AAA", -0.02)],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )
    get_config.cache_clear()

    assert audit["status"] == "BLOCKED"
