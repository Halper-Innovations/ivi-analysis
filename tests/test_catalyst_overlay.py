from __future__ import annotations

from app.autonomous.run_contract import EvidenceReference, ToolCallRecord
from app.autonomous.sector_contract import (
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
)
from app.autonomous.sector_runtime import (
    _audit_signals_by_class,
    _mergeable_confidence_caps,
    _selection_audit_for_ticker,
    classify_audit_signal,
)
from app.catalyst.overlay import compute_catalyst_signal


def test_cluster_two_buyers_open_market_no_buyback_confirmed():
    signal = compute_catalyst_signal(
        {
            "insider_distinct_buyers": 2,
            "insider_open_market_purchase_count": 3,
            "buyback_label": "NONE",
        }
    )
    assert signal.label == "CONFIRMED"
    assert signal.score == 4.0


def test_cluster_two_buyers_open_market_with_accelerating_buyback_confirmed():
    signal = compute_catalyst_signal(
        {
            "insider_distinct_buyers": 2,
            "insider_open_market_purchase_count": 3,
            "buyback_label": "ACCELERATING",
        }
    )
    assert signal.label == "CONFIRMED"
    assert signal.score == 5.0


def test_single_buyer_open_market_weak():
    signal = compute_catalyst_signal(
        {
            "insider_distinct_buyers": 1,
            "insider_open_market_purchase_count": 1,
            "buyback_label": "NONE",
        }
    )
    assert signal.label == "WEAK"
    assert signal.score == 2.0


def test_accelerating_buyback_alone_weak():
    signal = compute_catalyst_signal(
        {
            "insider_distinct_buyers": 0,
            "insider_open_market_purchase_count": 0,
            "buyback_label": "ACCELERATING",
        }
    )
    assert signal.label == "WEAK"
    assert signal.score == 1.0


def test_no_catalyst_none():
    signal = compute_catalyst_signal(
        {
            "insider_distinct_buyers": 0,
            "insider_open_market_purchase_count": 0,
            "buyback_label": "NONE",
        }
    )
    assert signal.label == "NONE"
    assert signal.score == 0.0


def test_empty_ctx_none_with_no_reasons():
    signal = compute_catalyst_signal({})
    assert signal.label == "NONE"
    assert signal.score == 0.0
    assert signal.reasons == []


def test_distinct_buyers_without_open_market_scores_zero_none():
    # Buyer points require at least one confirmed open-market purchase. A ctx with
    # distinct_buyers>=2 but zero open-market purchases (a non-buy) must NOT award
    # buyer points: score stays 0.0 and label stays NONE so a non-buy cannot be
    # mislabeled or carry a buy score.
    signal = compute_catalyst_signal(
        {
            "insider_distinct_buyers": 2,
            "insider_open_market_purchase_count": 0,
            "buyback_label": "NONE",
        }
    )
    assert signal.label == "NONE"
    assert signal.score == 0.0
    assert signal.reasons == []


# --- CATALYST audit class (positive, non-penalizing). ---


def test_insider_buy_cluster_classifies_catalyst():
    assert classify_audit_signal("INSIDER_BUY_CLUSTER") == "CATALYST"


def test_buyback_acceleration_classifies_catalyst():
    assert classify_audit_signal("BUYBACK_ACCELERATION") == "CATALYST"


def test_unknown_code_defaults_to_business_quality():
    # Regression guard: the default classification must stay BUSINESS_QUALITY so
    # the new CATALYST class never broadens to swallow unmapped codes.
    assert classify_audit_signal("SOME_UNKNOWN_CODE") == "BUSINESS_QUALITY"


def test_catalyst_excluded_from_hurdle_class_filter():
    # A CATALYST code mixed with a HURDLE code returns only the HURDLE code when
    # filtering for the HURDLE class -- the positive catalyst is never counted
    # as a hurdle.
    assert _audit_signals_by_class(
        ["INSIDER_BUY_CLUSTER", "BASE_RETURN_BELOW_12PCT_HURDLE"], "HURDLE"
    ) == ["BASE_RETURN_BELOW_12PCT_HURDLE"]


def _catalyst_packet() -> SectorCompanyFinancialPacket:
    # A clean, actionable packet carrying a CATALYST code on confidence_caps.
    return SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status="OK",
        current_price=50.0,
        valuation={"valuation_anchor": 100.0, "anchor_method": "dcf"},
        confidence_caps=["INSIDER_BUY_CLUSTER"],
    )


def _catalyst_audit() -> dict[str, object]:
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
    scenarios = [
        SectorExpectedReturnScenario(
            scenario_id="AAA_base_5Y",
            ticker="AAA",
            scenario_name="base",
            horizon_years=5,
            current_price=50.0,
            estimated_future_value_per_share=100.0,
            annualized_return=0.18,
        ),
        SectorExpectedReturnScenario(
            scenario_id="AAA_downside_5Y",
            ticker="AAA",
            scenario_name="downside",
            horizon_years=5,
            current_price=50.0,
            estimated_future_value_per_share=45.0,
            annualized_return=-0.02,
        ),
    ]
    packet = _catalyst_packet()
    return _selection_audit_for_ticker(
        selected_ticker=packet.ticker,
        packets_by_ticker={packet.ticker: packet},
        scenarios=scenarios,
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )


def test_mergeable_confidence_caps_excludes_catalyst_class():
    # On a downgraded decision (WATCHLIST_ONLY / NO_SELECTION) the audit's
    # confidence_caps are merged into the user-facing confidence_cap_reasons.
    # CATALYST-class codes are strictly positive/non-penalizing and must
    # be excluded so a positive insider-buy signal is never displayed as a
    # downgrade reason (it still surfaces via selection_audit['catalyst_signals']).
    audit = {
        "confidence_caps": [
            "INSIDER_BUY_CLUSTER",
            "EXPENSIVE_VS_EXPECTATIONS",
            "BUYBACK_ACCELERATION",
        ]
    }
    assert _mergeable_confidence_caps(audit) == ["EXPENSIVE_VS_EXPECTATIONS"]


def test_catalyst_code_passes_and_surfaces_in_catalyst_signals():
    # A CATALYST code on an otherwise-clean packet (zero binding blockers/caps)
    # keeps status PASS and surfaces in catalyst_signals; it never enters any of
    # the binding sets so it can never downgrade the grade.
    audit = _catalyst_audit()

    assert audit["status"] == "PASS"
    assert audit["catalyst_signals"] == ["INSIDER_BUY_CLUSTER"]
