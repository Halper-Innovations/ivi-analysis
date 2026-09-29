from __future__ import annotations

import pytest

from app.autonomous import (
    AutonomousSectorFinancialRunArtifact,
    BeliefUpdate,
    EvidenceReference,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorFinalDecision,
    SectorFinancialFramework,
    SectorResearchQuestion,
    ToolCallRecord,
)
from app.autonomous.sector_contract import (
    CandidateDisposition,
    GateEvaluation,
    SECTOR_CONTRACT_VERSION,
    SECTOR_CONTRACT_VERSION_V2,
    ScreenResult,
    SectorSelectionValidation,
    UnderwritingResult,
    build_v2_canonical_child_source_bindings,
)
from app.autonomous.competitive_frontier import build_competitive_frontier


def _frontier_inputs(
    tickers: list[str],
) -> tuple[list[SectorCompanyFinancialPacket], list[SectorExpectedReturnScenario]]:
    packets = [
        SectorCompanyFinancialPacket(
            ticker=ticker,
            financial_status="READY",
            model_fit_status="SUPPORTED",
            data_quality_status="COMPLETE",
            score_components={"deterministic_score": float(len(tickers) - index)},
        )
        for index, ticker in enumerate(tickers)
    ]
    scenarios = [
        SectorExpectedReturnScenario(
            scenario_id=f"{ticker}-base",
            ticker=ticker,
            scenario_name="base",
            horizon_years=5,
            current_price=10.0,
            estimated_future_value_per_share=20.0,
            annualized_return=0.15 - index / 100,
        )
        for index, ticker in enumerate(tickers)
    ]
    return packets, scenarios


def _signal_snapshots(tickers: list[str]) -> dict[str, dict[str, str]]:
    return {
        ticker: {
            "ticker": ticker,
            "sector": "energy",
            "as_of_date": "2026-07-16",
        }
        for ticker in tickers
    }


def _source_binding(
    ticker: str,
    tickers: list[str],
    packets: list[SectorCompanyFinancialPacket] | None = None,
    frontier_tickers: list[str] | None = None,
) -> dict[str, object]:
    normalized_ticker = ticker.strip().upper()
    default_packets, scenarios = _frontier_inputs(tickers)
    source_packets = packets if packets is not None else default_packets
    return build_v2_canonical_child_source_bindings(
        sector="energy",
        as_of_date="2026-07-16",
        company_packets=source_packets,
        scenarios=scenarios,
        signal_packet_snapshots=_signal_snapshots(tickers),
        frontier_candidate_tickers=list(frontier_tickers or tickers),
    )[normalized_ticker]


def _closed_frontier_payload(
    tickers: list[str],
    *,
    reviewed_tickers: list[str] | None = None,
) -> dict:
    packets, scenarios = _frontier_inputs(tickers)
    state = build_competitive_frontier(
        packets,
        scenarios,
        reviewed_tickers=reviewed_tickers or tickers,
    )
    payload = state.to_dict()
    payload.update(
        {
            "status": "CLOSED",
            "minimum_reviews_required": min(3, len(tickers)),
            "successful_review_count": len(state.reviewed_tickers),
            "attempted_tickers": list(state.reviewed_tickers),
            "failed_review_tickers": [],
            "source_bindings": {
                ticker: _source_binding(ticker, tickers, packets) for ticker in tickers
            },
            "signal_packet_snapshots": _signal_snapshots(tickers),
        }
    )
    return payload


def _validation_tool_call(
    evidence_id: str = "E-VALIDATION",
    *,
    validator_run_id: str | None = None,
) -> ToolCallRecord:
    return ToolCallRecord(
        call_id=(
            f"{validator_run_id}:VALIDATION-TC1"
            if validator_run_id
            else "VALIDATION-TC1"
        ),
        tool_name="challenge_selected_company",
        tool_input={"ticker": "AAA"},
        rationale="Challenge the provisional selection.",
        status="OK",
        evidence_ref_ids=[evidence_id],
        lane="selected_company_validation",
    )


def _validator_provider_usage(validator_run_id: str) -> list[dict[str, object]]:
    return [
        {
            "provider_call_id": f"{validator_run_id}:P1",
            "validator_run_id": validator_run_id,
            "lane": "selected_company_validation",
        }
    ]


def _bound_underwriting(
    ticker: str,
    verdict: str,
    source_binding: dict[str, object],
) -> tuple[UnderwritingResult, dict[str, object]]:
    normalized = ticker.strip().upper()
    run_id = f"child-{normalized.lower()}"
    evidence_id = f"{run_id}:E1"
    nested = {
        "request": {
            "run_id": run_id,
            "as_of_date": "2026-07-16",
            "candidate_scope": {
                "mode": "single_candidate",
                "tickers": [normalized],
                "source_binding": dict(source_binding),
                "signal_packet_snapshot": _signal_snapshots([normalized])[normalized],
            },
        }
    }
    return (
        UnderwritingResult(
            status="COMPLETED",
            verdict=verdict,
            confidence="HIGH",
            evidence_ref_ids=[evidence_id],
            tool_call_ids=["TC1"],
            child_run_id=run_id,
        ),
        {
            "ticker": normalized,
            "run_id": run_id,
            "source_binding": dict(source_binding),
            "artifact": nested,
            "attempts": [nested],
        },
    )


def _single_bound_underwritten_values() -> dict[str, object]:
    packets, scenarios = _frontier_inputs(["AAA"])
    frontier = _closed_frontier_payload(["AAA"])
    underwriting, child_run = _bound_underwriting(
        "AAA",
        "AVOID",
        frontier["source_bindings"]["AAA"],
    )
    return {
        "run_id": "ASFR-V2-BOUND-UNDERWRITING",
        "sector": "energy",
        "market_cap_focus": "large_and_mega",
        "objective": "Verify bound underwriting.",
        "as_of_date": "2026-07-16",
        "created_at": "2026-07-16T15:00:00Z",
        "status": "COMPLETED",
        "final_verdict": "NO_SELECTION",
        "selected_ticker": None,
        "confidence": None,
        "pipeline_version": "v2",
        "execution_status": "COMPLETED",
        "decision_status": "COMPLETE",
        "admitted_tickers": ["AAA"],
        "candidate_dispositions": [
            CandidateDisposition(
                ticker="AAA",
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="AVOID",
                watchlist_eligible=False,
                frontier_status="REVIEWED",
                underwriting_result=underwriting,
            )
        ],
        "selection_validation": SectorSelectionValidation(status="NOT_REQUIRED"),
        "competitive_frontier": frontier,
        "company_packets": packets,
        "expected_return_scenarios": scenarios,
        "company_autonomy_runs": [child_run],
        "contract_version": SECTOR_CONTRACT_VERSION_V2,
    }


def _framework() -> SectorFinancialFramework:
    return SectorFinancialFramework(
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        horizon_years=[5, 10],
        economic_model="Asset-light production with pricing-power and reinvestment constraints.",
        selected_value_drivers=["organic_revenue_growth", "incremental_margin", "roic_spread"],
        selected_metrics=["revenue_cagr_5y", "incremental_operating_margin", "roic"],
        valid_valuation_methods=["owner_earnings_multiple", "dcf", "epv"],
        invalid_valuation_methods=["generic_book_value_anchor"],
        required_evidence=["organic_growth_bridge", "maintenance_capex", "share_count_history"],
        normalization_policy={"remove_nonrecurring_items": True, "cycle_years": 7},
        hurdle_rate_policy={"base_case_minimum_annualized_return": 0.15},
        weighting_policy={"business_quality": 0.15, "expected_return": 0.20},
        sector_specific_risks=["customer_concentration", "input_cost_volatility"],
    )


def _company_packet() -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="Financially Viable",
        model_fit_status="VALID",
        data_quality_status="OK",
        market_cap_category="small_cap",
        current_price=25.50,
        business_quality={"gross_margin": 0.42, "operating_margin_stability": "HIGH"},
        reinvestment={"reinvestment_rate": 0.36, "incremental_roic": 0.18},
        returns_on_capital={"roic": 0.16, "cost_of_capital": 0.10},
        cash_conversion={"fcf_margin": 0.11, "cash_conversion_ratio": 0.92},
        balance_sheet={"net_debt_to_ebit": 0.8, "interest_coverage": 9.2},
        capital_allocation={"share_count_cagr": -0.01, "acquisition_spend_5y": 125.0},
        accounting_quality={"status": "CLEAN", "nonrecurring_items": 1},
        valuation={"owner_earnings_yield": 0.08, "base_case_value": 43.0},
        expected_return={"base_case_annualized_return": 0.145},
        score_components={"business_quality": 82, "expected_return": 76},
        blockers=[],
        confidence_caps=["CUSTOMER_CONCENTRATION_UNRESOLVED"],
        evidence_ref_ids=["E1"],
    )


def _scenario() -> SectorExpectedReturnScenario:
    return SectorExpectedReturnScenario(
        scenario_id="S1",
        ticker="AAA",
        scenario_name="BASE",
        horizon_years=5,
        current_price=25.50,
        estimated_future_value_per_share=52.75,
        annualized_return=0.156,
        revenue_cagr=0.08,
        normalized_operating_margin=0.18,
        owner_earnings_per_share=2.45,
        terminal_multiple=18.0,
        share_count_cagr=-0.01,
        downside_value_per_share=18.25,
        assumptions={"terminal_multiple_basis": "normalized owner earnings"},
        key_sensitivities=["terminal_multiple", "owner_earnings_margin"],
        unsupported_assumptions=[],
        evidence_ref_ids=["E1", "E2"],
    )


def _research_question() -> SectorResearchQuestion:
    return SectorResearchQuestion(
        question_id="Q1",
        question="Is the apparent margin expansion supported by sustainable incremental margins?",
        financial_pillar="Business quality",
        expected_decision_impact="Could determine whether the base-case return clears the hurdle.",
        priority="HIGH",
        status="ANSWERED",
        target_tickers=["AAA", "BBB"],
        planned_tools=["fetch_kpi_trends", "fetch_filing_section"],
        depends_on=["framework_selected"],
        evidence_ref_ids=["E1"],
    )


def _final_decision() -> SectorFinalDecision:
    return SectorFinalDecision(
        verdict="SELECTED",
        confidence="MODERATE",
        selected_ticker="AAA",
        expected_annualized_return_range="14-18%",
        thesis="AAA has the strongest underwritten per-share compounding profile.",
        key_risk="Margin expansion could reverse if input costs normalize poorly.",
        downside_case="Downside value is $18.25 if margins revert and the exit multiple compresses.",
        falsifiers=["Owner earnings per share declines for two consecutive years."],
        why_selected_over_finalists=["Higher cash conversion and lower leverage than BBB."],
        rejected_finalists=[{"ticker": "BBB", "reason": "Higher leverage and weaker cash conversion."}],
        selection_blockers=[],
        confidence_cap_reasons=["CUSTOMER_CONCENTRATION_UNRESOLVED"],
        evidence_ref_ids=["E1", "E2"],
    )


def test_sector_financial_framework_defaults_and_round_trip():
    framework = SectorFinancialFramework(
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        horizon_years=[5, 10],
        economic_model="Production economics with cyclical demand exposure.",
    )

    restored = SectorFinancialFramework.from_dict(framework.to_dict())

    assert restored == framework
    assert restored.selected_value_drivers == []
    assert restored.selected_metrics == []
    assert restored.valid_valuation_methods == []
    assert restored.invalid_valuation_methods == []
    assert restored.required_evidence == []
    assert restored.normalization_policy == {}
    assert restored.hurdle_rate_policy == {}
    assert restored.weighting_policy == {}
    assert restored.sector_specific_risks == []
    assert restored.contract_version == SECTOR_CONTRACT_VERSION


def test_sector_company_financial_packet_defaults_keep_missing_values_missing():
    packet = SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="Data Insufficient",
        model_fit_status="UNKNOWN",
        data_quality_status="MISSING_PRICE",
    )

    restored = SectorCompanyFinancialPacket.from_dict(packet.to_dict())

    assert restored == packet
    assert restored.current_price is None
    assert restored.current_price_currency is None
    assert restored.cap_stage_price_currency is None
    assert restored.market_cap_category is None
    assert restored.business_quality == {}
    assert restored.reinvestment == {}
    assert restored.returns_on_capital == {}
    assert restored.cash_conversion == {}
    assert restored.balance_sheet == {}
    assert restored.capital_allocation == {}
    assert restored.accounting_quality == {}
    assert restored.valuation == {}
    assert restored.expected_return == {}
    assert restored.score_components == {}
    assert restored.blockers == []
    assert restored.confidence_caps == []
    assert restored.evidence_ref_ids == []


def test_sector_expected_return_scenario_round_trip():
    scenario = _scenario()

    restored = SectorExpectedReturnScenario.from_dict(scenario.to_dict())

    assert restored == scenario
    assert restored.scenario_name == "BASE"
    assert restored.horizon_years == 5
    assert restored.current_price == 25.50
    assert restored.estimated_future_value_per_share == 52.75
    assert restored.annualized_return == 0.156
    assert restored.evidence_ref_ids == ["E1", "E2"]


def test_sector_expected_return_scenario_missing_values_stay_none():
    scenario = SectorExpectedReturnScenario(
        scenario_id="S0",
        ticker="AAA",
        scenario_name="DOWNSIDE",
        horizon_years=10,
        current_price=None,
        estimated_future_value_per_share=None,
        annualized_return=None,
    )

    restored = SectorExpectedReturnScenario.from_dict(scenario.to_dict())

    assert restored == scenario
    assert restored.current_price is None
    assert restored.estimated_future_value_per_share is None
    assert restored.annualized_return is None
    assert restored.revenue_cagr is None
    assert restored.key_sensitivities == []
    assert restored.unsupported_assumptions == []


def test_sector_research_question_defaults_and_round_trip():
    question = SectorResearchQuestion(
        question_id="Q1",
        question="Does base-case expected return clear the hurdle without unsupported assumptions?",
        financial_pillar="Valuation and expected return",
        expected_decision_impact="Could force no selection.",
        priority="HIGH",
        status="OPEN",
    )

    restored = SectorResearchQuestion.from_dict(question.to_dict())

    assert restored == question
    assert restored.target_tickers == []
    assert restored.planned_tools == []
    assert restored.depends_on == []
    assert restored.evidence_ref_ids == []


def test_sector_final_decision_no_selection_defaults():
    decision = SectorFinalDecision(
        verdict="NO_SELECTION",
        confidence=None,
        selected_ticker=None,
        expected_annualized_return_range=None,
        thesis="No company cleared the long-term expected-return hurdle.",
        key_risk="Forcing a winner would overstate unsupported valuation assumptions.",
        downside_case="Downside cases did not provide enough capital protection.",
        no_selection_reason="All finalists failed the base-case return hurdle.",
        selection_blockers=["BASE_CASE_RETURN_BELOW_HURDLE"],
    )

    restored = SectorFinalDecision.from_dict(decision.to_dict())

    assert restored == decision
    assert restored.verdict == "NO_SELECTION"
    assert restored.selected_ticker is None
    assert restored.confidence is None
    assert restored.falsifiers == []
    assert restored.why_selected_over_finalists == []
    assert restored.rejected_finalists == []
    assert restored.confidence_cap_reasons == []
    assert restored.evidence_ref_ids == []


def test_autonomous_sector_financial_run_artifact_minimal_round_trip():
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="ASFR-1",
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        objective="Find the best 5-10 year financial return candidate or stop.",
        as_of_date="2026-04-26",
        created_at="2026-04-26T15:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        no_selection_reason="Insufficient financial evidence.",
        degraded_states=["FINANCIAL_DATA_INSUFFICIENT"],
        audit_notes=["Stopped without forcing a sector winner."],
    )

    restored = AutonomousSectorFinancialRunArtifact.from_dict(artifact.to_dict())
    restored_legacy = AutonomousSectorFinancialRunArtifact.from_dict(
        {k: v for k, v in artifact.to_dict().items() if k != "scan_family"}
    )

    assert restored == artifact
    assert restored.scan_family == "normal"
    assert restored_legacy.scan_family == "normal"
    assert restored.framework is None
    assert restored.candidate_selection == {}
    assert restored.company_packets == []
    assert restored.research_questions == []
    assert restored.expected_return_scenarios == []
    assert restored.tool_calls == []
    assert restored.evidence == []
    assert restored.belief_updates == []
    assert restored.final_decision is None
    assert restored.selection_audit == {}
    assert restored.framework_evidence_preflight == []
    assert restored.final_decision_prompt_context == {}
    assert restored.audit_gap_repair_attempted is False
    assert restored.audit_gap_repair_status is None
    assert restored.audit_gap_repair_notes == []
    assert restored.watchlist_resolution_attempted is False
    assert restored.watchlist_resolution_status is None
    assert restored.watchlist_resolution_notes == []
    assert restored.no_selection_finalist_audit_attempted is False
    assert restored.no_selection_finalist_audit_status is None
    assert restored.no_selection_finalist_audit_focus_ticker is None
    assert restored.no_selection_finalist_audit_notes == []
    assert restored.no_selection_finalist_resolution_attempted is False
    assert restored.no_selection_finalist_resolution_status is None
    assert restored.no_selection_finalist_resolution_notes == []
    assert restored.alternate_finalist_audit_attempted is False
    assert restored.alternate_finalist_audit_status is None
    assert restored.alternate_finalist_audit_notes == []
    assert restored.alternate_finalist_audit_results == []
    assert restored.company_autonomy_attempted is False
    assert restored.company_autonomy_status is None
    assert restored.company_autonomy_notes == []
    assert restored.company_autonomy_runs == []
    assert restored.company_autonomy_decision_trace == {}
    assert restored.relative_ranking == []
    assert restored.memo_body == {}
    assert restored.contract_version == SECTOR_CONTRACT_VERSION


def test_v1_artifact_without_v2_fields_remains_legacy_and_unrelabelled():
    legacy = {
        "run_id": "ASFR-V1-LEGACY",
        "sector": "energy",
        "market_cap_focus": "mid_cap",
        "objective": "Legacy replay.",
        "as_of_date": "2026-04-26",
        "created_at": "2026-04-26T15:00:00Z",
        "status": "COMPLETED",
        "selected_ticker": None,
        "confidence": None,
    }

    restored = AutonomousSectorFinancialRunArtifact.from_dict(legacy)

    assert restored.pipeline_version == "v1"
    assert restored.contract_version == SECTOR_CONTRACT_VERSION
    assert restored.execution_status is None
    assert restored.decision_status is None
    assert restored.final_verdict is None
    assert restored.admitted_tickers == ()
    assert restored.candidate_dispositions == []
    assert restored.selection_validation is None


def test_v2_selected_artifact_round_trip_requires_underwriting_and_validation():
    packets, scenarios = _frontier_inputs(["AAA", "BBB"])
    frontier = _closed_frontier_payload(["AAA", "BBB"])
    aaa_underwriting, aaa_run = _bound_underwriting(
        "AAA", "ACTIONABLE", frontier["source_bindings"]["AAA"]
    )
    bbb_underwriting, bbb_run = _bound_underwriting(
        "BBB", "AVOID", frontier["source_bindings"]["BBB"]
    )
    dispositions = [
        CandidateDisposition(
            ticker="aaa",
            terminal_state="UNDERWRITTEN",
            scope_status="IN_SCOPE",
            screen_status="PASS",
            review_status="COMPLETED",
            underwriting_verdict="ACTIONABLE",
            underwriting_confidence="HIGH",
            watchlist_eligible=True,
            issuer_key="CIK:0000000001",
            issuer_cik="0000000001",
            primary_ticker="AAA",
            security_type="COMMON_STOCK",
            reason_codes=["UNDERWRITING_COMPLETE"],
            evidence_ref_ids=["E1"],
            last_completed_stage="validation",
            frontier_status="REVIEWED",
            underwriting_result=aaa_underwriting,
        ),
        CandidateDisposition(
            ticker="BBB",
            terminal_state="UNDERWRITTEN",
            scope_status="IN_SCOPE",
            screen_status="PASS",
            review_status="COMPLETED",
            underwriting_verdict="AVOID",
            watchlist_eligible=False,
            last_completed_stage="underwriting",
            frontier_status="REVIEWED",
            underwriting_result=bbb_underwriting,
        ),
        CandidateDisposition(
            ticker="CCC",
            terminal_state="OUT_OF_SCOPE",
            scope_status="OUT_OF_SCOPE",
            screen_status="NOT_RUN",
            review_status="NOT_REQUIRED",
            reason_codes=["SECURITY_TYPE_NON_COMMON_EQUITY"],
            last_completed_stage="scope",
        ),
    ]
    validation = SectorSelectionValidation(
        status="VALIDATED",
        selected_ticker="aaa",
        validator_run_id="validator-1",
        validator_verdict="CONFIRMED_ACTIONABLE",
        reason_codes=["NO_BINDING_CONTRADICTION"],
        evidence_ref_ids=["validator-1:E2"],
        evidence=[
            EvidenceReference(
                evidence_id="validator-1:E2",
                source_type="tool_output",
                source_label="challenge",
                summary="Independent challenge evidence.",
                ticker="AAA",
                tool_call_id="validator-1:VALIDATION-TC1",
                confidence="HIGH",
            )
        ],
        tool_calls=[
            _validation_tool_call(
                "validator-1:E2", validator_run_id="validator-1"
            )
        ],
        provider_usage=_validator_provider_usage("validator-1"),
        notes=["Independent challenge passed."],
        source_binding=frontier["source_bindings"]["AAA"],
    )
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="ASFR-V2-SELECTED",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Run truthful v2 underwriting.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T15:00:00Z",
        status="COMPLETED",
        final_verdict="SELECTED",
        selected_ticker="aaa",
        confidence="HIGH",
        pipeline_version="v2",
        execution_status="COMPLETED",
        decision_status="COMPLETE",
        admitted_tickers=["aaa", "BBB"],
        candidate_selection={
            "loaded_tickers": ["AAA", "BBB", "CCC"],
            "selected_tickers": ["AAA", "BBB"],
            "excluded_tickers": ["CCC"],
        },
        candidate_dispositions=dispositions,
        selection_validation=validation,
        competitive_frontier=frontier,
        company_packets=packets,
        expected_return_scenarios=scenarios,
        company_autonomy_runs=[aaa_run, bbb_run],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    restored = AutonomousSectorFinancialRunArtifact.from_dict(artifact.to_dict())

    assert restored == artifact
    assert restored.admitted_tickers == ("AAA", "BBB")
    assert restored.candidate_dispositions[0].ticker == "AAA"
    assert restored.candidate_dispositions[2].terminal_state == "OUT_OF_SCOPE"
    assert restored.selection_validation is not None
    assert restored.selection_validation.status == "VALIDATED"


def test_v2_selected_binding_separates_complete_packet_cohort_from_frontier() -> None:
    packets, scenarios = _frontier_inputs(["AAA", "BBB"])
    frontier_state = build_competitive_frontier(
        [packets[0]],
        [scenarios[0]],
        reviewed_tickers=["AAA"],
    )
    binding = _source_binding(
        "AAA",
        ["AAA", "BBB"],
        packets,
        frontier_tickers=["AAA"],
    )
    frontier = frontier_state.to_dict()
    frontier.update(
        {
            "status": "CLOSED",
            "minimum_reviews_required": 1,
            "successful_review_count": 1,
            "attempted_tickers": ["AAA"],
            "failed_review_tickers": [],
            "source_bindings": {"AAA": binding},
            "signal_packet_snapshots": _signal_snapshots(["AAA", "BBB"]),
        }
    )
    failed_screen = ScreenResult(
        contract_id="energy",
        status="FAIL",
        gate_evaluations=[
            GateEvaluation(
                contract_id="energy",
                rule_id="SOURCE_BACKED_GATE",
                status="FAIL",
                applicable=True,
                observed_value=0,
                threshold=1,
                evidence_ref_id="filing:BBB:2026",
                reason_code="SOURCE_BACKED_GATE_FAILURE",
            )
        ],
        reason_codes=["SOURCE_BACKED_GATE_FAILURE"],
    )
    validation = SectorSelectionValidation(
        status="VALIDATED",
        selected_ticker="AAA",
        validator_run_id="validator-mixed-cohort",
        validator_verdict="CONFIRMED_ACTIONABLE",
        evidence_ref_ids=["validator-mixed-cohort:E-VALIDATION"],
        evidence=[
            EvidenceReference(
                evidence_id="validator-mixed-cohort:E-VALIDATION",
                source_type="tool_output",
                source_label="challenge",
                summary="Independent challenge evidence.",
                ticker="AAA",
                tool_call_id="validator-mixed-cohort:VALIDATION-TC1",
                confidence="HIGH",
            )
        ],
        tool_calls=[
            _validation_tool_call(
                "validator-mixed-cohort:E-VALIDATION",
                validator_run_id="validator-mixed-cohort",
            )
        ],
        provider_usage=_validator_provider_usage("validator-mixed-cohort"),
        source_binding=binding,
    )
    underwriting, child_run = _bound_underwriting("AAA", "ACTIONABLE", binding)

    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="ASFR-V2-MIXED-SCREEN",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Bind the complete packet cohort and eligible frontier separately.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T15:00:00Z",
        status="COMPLETED",
        final_verdict="SELECTED",
        selected_ticker="AAA",
        confidence="HIGH",
        pipeline_version="v2",
        execution_status="COMPLETED",
        decision_status="COMPLETE",
        admitted_tickers=["AAA", "BBB"],
        candidate_dispositions=[
            CandidateDisposition(
                ticker="AAA",
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="ACTIONABLE",
                watchlist_eligible=True,
                frontier_status="REVIEWED",
                underwriting_result=underwriting,
            ),
            CandidateDisposition(
                ticker="BBB",
                terminal_state="SCREENED_OUT",
                scope_status="IN_SCOPE",
                screen_status="FAIL",
                review_status="NOT_REQUIRED",
                reason_codes=["SOURCE_BACKED_GATE_FAILURE"],
                frontier_status="NOT_ELIGIBLE",
                screen_result=failed_screen,
                underwriting_result=UnderwritingResult(
                    status="NOT_REQUIRED",
                    reason_codes=["SCREEN_FAILED"],
                ),
            ),
        ],
        selection_validation=validation,
        competitive_frontier=frontier,
        company_packets=packets,
        expected_return_scenarios=scenarios,
        company_autonomy_runs=[child_run],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    assert artifact.selection_validation is not None
    assert artifact.selection_validation.source_binding["cohort_tickers"] == [
        "AAA",
        "BBB",
    ]
    assert artifact.selection_validation.source_binding[
        "frontier_candidate_tickers"
    ] == ["AAA"]


def test_v2_failed_attempt_is_incomplete_and_has_no_false_verdict():
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="ASFR-V2-FAILED",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Interrupted v2 attempt.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T15:00:00Z",
        status="FAILED",
        final_verdict=None,
        selected_ticker=None,
        confidence=None,
        pipeline_version="v2",
        execution_status="FAILED",
        decision_status="INCOMPLETE",
        admitted_tickers=["AAA"],
        candidate_dispositions=[
            CandidateDisposition(
                ticker="AAA",
                terminal_state="NEEDS_DATA",
                scope_status="IN_SCOPE",
                screen_status="INCOMPLETE",
                review_status="NOT_STARTED",
                watchlist_eligible=True,
                reason_codes=["PROVIDER_FAILURE"],
                last_completed_stage="filings",
            )
        ],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    assert artifact.final_verdict is None
    assert artifact.decision_status == "INCOMPLETE"


@pytest.mark.parametrize(
    ("review_status", "underwriting_verdict"),
    [
        ("FAILED", None),
        ("COMPLETED", "DATA_INCOMPLETE"),
    ],
)
def test_needs_data_disposition_preserves_successful_screen_truth(
    review_status, underwriting_verdict
):
    disposition = CandidateDisposition(
        ticker="AAA",
        terminal_state="NEEDS_DATA",
        scope_status="IN_SCOPE",
        screen_status="PASS",
        review_status=review_status,
        underwriting_verdict=underwriting_verdict,
        watchlist_eligible=True,
        reason_codes=["UNDERWRITING_INCOMPLETE"],
        last_completed_stage="SCREENING",
    )

    assert disposition.screen_status == "PASS"
    assert disposition.review_status == review_status


def test_candidate_disposition_round_trip_preserves_security_identity():
    disposition = CandidateDisposition(
        ticker="adrx",
        terminal_state="READY_FOR_UNDERWRITING",
        scope_status="IN_SCOPE",
        screen_status="PASS",
        review_status="NOT_STARTED",
        watchlist_eligible=True,
        issuer_key="CIK:0001234567",
        issuer_cik="0001234567",
        primary_ticker="ADRX",
        security_type="ADR",
        is_secondary_class=False,
        is_adr=True,
        adr_ratio=2.0,
        share_class_ratio=None,
        identity_source_url="https://www.sec.gov/Archives/example-20f.htm",
        ratio_source_url="https://www.sec.gov/Archives/example-20f.htm",
        last_completed_stage="SCREENING",
    )

    restored = CandidateDisposition.from_dict(disposition.to_dict())

    assert restored == disposition
    assert restored.ticker == "ADRX"
    assert restored.is_adr is True
    assert restored.is_secondary_class is False
    assert restored.adr_ratio == 2.0
    assert restored.identity_source_url == (
        "https://www.sec.gov/Archives/example-20f.htm"
    )
    assert restored.ratio_source_url == (
        "https://www.sec.gov/Archives/example-20f.htm"
    )


def test_deferred_by_bound_is_in_scope_but_never_screened_or_reviewed():
    disposition = CandidateDisposition(
        ticker="BBB",
        terminal_state="DEFERRED_BY_BOUND",
        scope_status="IN_SCOPE",
        screen_status="NOT_RUN",
        review_status="NOT_REQUIRED",
        watchlist_eligible=False,
        reason_codes=["DEFERRED_BY_EXECUTION_BOUND"],
        frontier_status="NOT_ELIGIBLE",
        screen_result=ScreenResult(
            contract_id="energy",
            status="NOT_RUN",
            reason_codes=["DEFERRED_BY_EXECUTION_BOUND"],
        ),
        underwriting_result=UnderwritingResult(
            status="NOT_REQUIRED",
            reason_codes=["DEFERRED_BY_EXECUTION_BOUND"],
        ),
    )

    assert CandidateDisposition.from_dict(disposition.to_dict()) == disposition


def test_failed_gate_requires_concrete_evidence_and_threshold():
    with pytest.raises(ValueError, match="completed gate requires"):
        GateEvaluation(
            contract_id="energy",
            rule_id="GOING_CONCERN",
            status="FAIL",
            applicable=True,
            observed_value="registrant substantial doubt",
            threshold="no corroborated registrant assertion",
        )


def test_passing_gate_requires_concrete_evidence_and_threshold():
    with pytest.raises(ValueError, match="completed gate requires"):
        GateEvaluation(
            contract_id="energy",
            rule_id="PENNY_FLOOR",
            status="PASS",
            applicable=True,
            observed_value=12.5,
            threshold=1.0,
        )


def test_screen_result_requires_exact_rule_coverage():
    with pytest.raises(ValueError, match="cover every required gate rule"):
        ScreenResult(
            contract_id="energy",
            status="PASS",
            required_rule_ids=["PENNY_FLOOR", "GOING_CONCERN"],
            gate_evaluations=[
                GateEvaluation(
                    contract_id="energy",
                    rule_id="PENNY_FLOOR",
                    status="PASS",
                    applicable=True,
                    observed_value=12.5,
                    threshold=1.0,
                    evidence_ref_id="price:AAA:2026-07-15",
                )
            ],
        )


def test_incomplete_underwriting_round_trips_without_a_verdict():
    result = UnderwritingResult(
        status="INCOMPLETE",
        reason_codes=["UNDERWRITING_RUNTIME_INCOMPLETE"],
        evidence_ref_ids=["child-aaa:E1"],
        tool_call_ids=["TC1"],
        child_run_id="child-aaa",
    )

    assert UnderwritingResult.from_dict(result.to_dict()) == result
    assert result.verdict is None


@pytest.mark.parametrize("verdict", ["NO_WINNER", "DATA_INCOMPLETE"])
def test_completed_underwriting_rejects_non_decisions(verdict):
    with pytest.raises(ValueError, match="completed underwriting requires"):
        UnderwritingResult(
            status="COMPLETED",
            verdict=verdict,
            confidence="MODERATE",
            evidence_ref_ids=["child-aaa:E1"],
            tool_call_ids=["TC1"],
            child_run_id="child-aaa",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"evidence_ref_ids": []},
        {"tool_call_ids": []},
        {"child_run_id": None},
    ],
)
def test_completed_underwriting_requires_provenance(overrides):
    values = {
        "status": "COMPLETED",
        "verdict": "AVOID",
        "confidence": "MODERATE",
        "evidence_ref_ids": ["child-aaa:E1"],
        "tool_call_ids": ["TC1"],
        "child_run_id": "child-aaa",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="completed underwriting requires"):
        UnderwritingResult(**values)


def test_screen_and_underwriting_results_round_trip_separately():
    screen = ScreenResult(
        contract_id="energy",
        status="FAIL",
        gate_evaluations=[
            GateEvaluation(
                contract_id="energy",
                rule_id="GOING_CONCERN",
                status="FAIL",
                applicable=True,
                observed_value="registrant substantial doubt",
                threshold="no corroborated registrant assertion",
                evidence_ref_id="filing:0001:10-k",
                evidence_url="https://www.sec.gov/Archives/example.htm",
                reason_code="ATTRIBUTED_GOING_CONCERN",
            )
        ],
        reason_codes=["ATTRIBUTED_GOING_CONCERN"],
    )
    underwriting = UnderwritingResult(
        status="NOT_REQUIRED",
        reason_codes=["SCREEN_FAILED"],
    )

    assert ScreenResult.from_dict(screen.to_dict()) == screen
    assert UnderwritingResult.from_dict(underwriting.to_dict()) == underwriting


def test_screened_out_disposition_requires_failed_screen_result():
    with pytest.raises(ValueError, match="source-backed failed ScreenResult"):
        CandidateDisposition(
            ticker="AAA",
            terminal_state="SCREENED_OUT",
            scope_status="IN_SCOPE",
            screen_status="FAIL",
            review_status="NOT_REQUIRED",
            reason_codes=["CLEAR_IMPAIRMENT"],
        )


@pytest.mark.parametrize(
    "values",
    [
        {
            "terminal_state": "OUT_OF_SCOPE",
            "scope_status": "OUT_OF_SCOPE",
            "screen_status": "NOT_RUN",
            "review_status": "NOT_REQUIRED",
        },
        {
            "terminal_state": "SCREENED_OUT",
            "scope_status": "IN_SCOPE",
            "screen_status": "PASS",
            "review_status": "NOT_REQUIRED",
            "reason_codes": ["GATE_FAILED"],
        },
        {
            "terminal_state": "READY_FOR_UNDERWRITING",
            "scope_status": "IN_SCOPE",
            "screen_status": "FAIL",
            "review_status": "COMPLETED",
            "underwriting_verdict": "ACTIONABLE",
            "watchlist_eligible": True,
        },
        {
            "terminal_state": "UNDERWRITTEN",
            "scope_status": "IN_SCOPE",
            "screen_status": "PASS",
            "review_status": "COMPLETED",
        },
        {
            "terminal_state": "NEEDS_DATA",
            "scope_status": "IN_SCOPE",
            "screen_status": "FAIL",
            "review_status": "NOT_STARTED",
            "reason_codes": ["SCREEN_NOT_COMPLETE"],
        },
    ],
)
def test_candidate_disposition_rejects_impossible_state_combinations(values):
    with pytest.raises(ValueError, match="incompatible pipeline state"):
        CandidateDisposition(ticker="AAA", **values)


def test_needs_data_can_preserve_a_completed_passing_screen():
    screen = ScreenResult(
        contract_id="energy",
        status="PASS",
        gate_evaluations=[
            GateEvaluation(
                contract_id="energy",
                rule_id="PENNY_FLOOR",
                status="PASS",
                applicable=True,
                observed_value=12.5,
                threshold=1.0,
                evidence_ref_id="price:AAA:2026-07-15",
            )
        ],
    )
    disposition = CandidateDisposition(
        ticker="AAA",
        terminal_state="NEEDS_DATA",
        scope_status="IN_SCOPE",
        screen_status="PASS",
        review_status="NOT_STARTED",
        reason_codes=["VALUATION_INPUTS_INCOMPLETE"],
        screen_result=screen,
        underwriting_result=UnderwritingResult(status="NOT_STARTED"),
    )

    assert disposition.screen_result == screen
    assert disposition.underwriting_result == UnderwritingResult(status="NOT_STARTED")


@pytest.mark.parametrize(
    "overrides",
    [
        {"validator_run_id": None},
        {"validator_verdict": "CONTRADICTED"},
        {"validator_verdict": None},
        {"evidence_ref_ids": []},
    ],
)
def test_validated_selection_requires_affirmative_challenge_proof(overrides):
    values = {
        "status": "VALIDATED",
        "selected_ticker": "AAA",
        "validator_run_id": "validator-aaa",
        "validator_verdict": "CONFIRMED_ACTIONABLE",
        "evidence_ref_ids": ["E-VALIDATION"],
        "evidence": [
            EvidenceReference(
                evidence_id="E-VALIDATION",
                source_type="tool_output",
                source_label="challenge",
                summary="Independent validation evidence.",
                tool_call_id="VALIDATION-TC1",
            )
        ],
        "tool_calls": [_validation_tool_call()],
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="successful challenge evidence"):
        SectorSelectionValidation(**values)


@pytest.mark.parametrize(
    ("evidence_ticker", "confidence", "lane"),
    [
        ("BBB", "HIGH", "selected_company_validation"),
        ("AAA", "LOW", "selected_company_validation"),
        ("AAA", "HIGH", "company_underwriting"),
    ],
)
def test_validated_selection_requires_selected_ticker_decision_usable_lane_evidence(
    evidence_ticker,
    confidence,
    lane,
):
    with pytest.raises(ValueError, match="successful challenge evidence"):
        SectorSelectionValidation(
            status="VALIDATED",
            selected_ticker="AAA",
            validator_run_id="validator-aaa",
            validator_verdict="CONFIRMED_ACTIONABLE",
            evidence_ref_ids=["E-VALIDATION"],
            evidence=[
                EvidenceReference(
                    evidence_id="E-VALIDATION",
                    source_type="tool_output",
                    source_label="challenge",
                    summary="Evidence must concern the selected security.",
                    ticker=evidence_ticker,
                    tool_call_id="VALIDATION-TC1",
                    confidence=confidence,
                )
            ],
            tool_calls=[
                ToolCallRecord(
                    call_id="VALIDATION-TC1",
                    tool_name="challenge_selected_company",
                    tool_input={"ticker": "AAA"},
                    rationale="Challenge the provisional selection.",
                    status="OK",
                    evidence_ref_ids=["E-VALIDATION"],
                    lane=lane,
                )
            ],
        )


def test_validated_selection_rejects_evidence_backed_only_by_failed_challenge_call():
    with pytest.raises(ValueError, match="successful challenge evidence"):
        SectorSelectionValidation(
            status="VALIDATED",
            selected_ticker="AAA",
            validator_run_id="validator-aaa",
            validator_verdict="CONFIRMED_ACTIONABLE",
            evidence_ref_ids=["E-VALIDATION"],
            evidence=[
                EvidenceReference(
                    evidence_id="E-VALIDATION",
                    source_type="tool_output",
                    source_label="challenge",
                    summary="The failed challenge call cannot validate a selection.",
                    tool_call_id="VALIDATION-TC1",
                )
            ],
            tool_calls=[
                ToolCallRecord(
                    call_id="VALIDATION-TC1",
                    tool_name="challenge_selected_company",
                    tool_input={"ticker": "AAA"},
                    rationale="Challenge the provisional selection.",
                    status="ERROR",
                    evidence_ref_ids=["E-VALIDATION"],
                )
            ],
        )


def test_validated_selection_requires_every_cited_ref_from_ok_challenge_call():
    with pytest.raises(ValueError, match="successful challenge evidence"):
        SectorSelectionValidation(
            status="VALIDATED",
            selected_ticker="AAA",
            validator_run_id="validator-aaa",
            validator_verdict="CONFIRMED_ACTIONABLE",
            evidence_ref_ids=["E-OK", "E-UNPROVEN"],
            evidence=[
                EvidenceReference(
                    evidence_id="E-OK",
                    source_type="tool_output",
                    source_label="challenge",
                    summary="Evidence from the successful challenge.",
                    tool_call_id="VALIDATION-TC1",
                ),
                EvidenceReference(
                    evidence_id="E-UNPROVEN",
                    source_type="tool_output",
                    source_label="challenge",
                    summary="Cited evidence without a successful challenge call.",
                    tool_call_id="VALIDATION-TC2",
                ),
            ],
            tool_calls=[_validation_tool_call("E-OK")],
        )


def test_validated_selection_rejects_ok_call_claiming_evidence_from_failed_call():
    with pytest.raises(ValueError, match="successful challenge evidence"):
        SectorSelectionValidation(
            status="VALIDATED",
            selected_ticker="AAA",
            validator_run_id="validator-aaa",
            validator_verdict="CONFIRMED_ACTIONABLE",
            evidence_ref_ids=["E-VALIDATION"],
            evidence=[
                EvidenceReference(
                    evidence_id="E-VALIDATION",
                    source_type="tool_output",
                    source_label="challenge",
                    summary="The evidence was actually produced by a failed call.",
                    tool_call_id="FAILED-TC",
                )
            ],
            tool_calls=[
                ToolCallRecord(
                    call_id="FAILED-TC",
                    tool_name="challenge_selected_company",
                    tool_input={"ticker": "AAA"},
                    rationale="Failed challenge attempt.",
                    status="ERROR",
                    evidence_ref_ids=[],
                ),
                ToolCallRecord(
                    call_id="OK-TC",
                    tool_name="challenge_selected_company",
                    tool_input={"ticker": "AAA"},
                    rationale="Crafted call tries to claim another call's evidence.",
                    status="OK",
                    evidence_ref_ids=["E-VALIDATION"],
                ),
            ],
        )


def test_contradicted_selection_requires_canonical_source_binding() -> None:
    with pytest.raises(ValueError, match="canonical selected-company source_binding"):
        SectorSelectionValidation(
            status="CONTRADICTED",
            selected_ticker="AAA",
            validator_run_id="validator-aaa",
            validator_verdict="AVOID",
            source_binding={},
        )


def test_v2_underwritten_child_requires_top_level_frontier_source_binding() -> None:
    values = _single_bound_underwritten_values()
    child_run = values["company_autonomy_runs"][0]
    child_run.pop("source_binding")

    with pytest.raises(ValueError, match="child source_binding"):
        AutonomousSectorFinancialRunArtifact(**values)


def test_v2_underwritten_child_requires_nested_frontier_source_binding() -> None:
    values = _single_bound_underwritten_values()
    child_run = values["company_autonomy_runs"][0]
    child_run["artifact"]["request"]["candidate_scope"]["source_binding"] = {
        **child_run["source_binding"],
        "signal_packet_fingerprint": "0" * 64,
    }

    with pytest.raises(ValueError, match="nested child source_binding"):
        AutonomousSectorFinancialRunArtifact(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"decision_status": "INCOMPLETE", "final_verdict": "NO_SELECTION"}, "incomplete"),
        ({"decision_status": "COMPLETE", "final_verdict": None, "selected_ticker": None}, "requires a final verdict"),
        ({"candidate_dispositions": []}, "admitted_tickers"),
        ({"final_verdict": "WATCHLIST", "selected_ticker": "ZZZ"}, "must be admitted"),
        (
            {
                "final_verdict": "NO_SELECTION",
                "selected_ticker": None,
            },
            "decisively screened out or underwritten negative",
        ),
        (
            {
                "candidate_selection": {
                    "loaded_tickers": ["AAA", "BBB"],
                    "selected_tickers": ["AAA"],
                    "excluded_tickers": ["BBB"],
                }
            },
            "every discovered security",
        ),
        (
            {"degraded_states": ["LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE"]},
            "unresolved run failures",
        ),
        (
            {
                "admitted_tickers": ["AAA", "BBB"],
                "candidate_dispositions": [
                    CandidateDisposition(
                        ticker="AAA",
                        terminal_state="UNDERWRITTEN",
                        scope_status="IN_SCOPE",
                        screen_status="PASS",
                        review_status="COMPLETED",
                        underwriting_verdict="ACTIONABLE",
                        watchlist_eligible=True,
                    ),
                    CandidateDisposition(
                        ticker="BBB",
                        terminal_state="READY_FOR_UNDERWRITING",
                        scope_status="IN_SCOPE",
                        screen_status="PASS",
                        review_status="NOT_STARTED",
                        watchlist_eligible=True,
                    ),
                ],
            },
            "closed competitive frontier",
        ),
        ({"selection_validation": None}, "VALIDATED"),
    ],
)
def test_v2_artifact_rejects_false_completion_states(overrides, message):
    values = {
        "run_id": "ASFR-V2-INVALID",
        "sector": "energy",
        "market_cap_focus": "large_and_mega",
        "objective": "Reject false completion.",
        "as_of_date": "2026-07-16",
        "created_at": "2026-07-16T15:00:00Z",
        "status": "COMPLETED",
        "final_verdict": "SELECTED",
        "selected_ticker": "AAA",
        "confidence": "MODERATE",
        "pipeline_version": "v2",
        "execution_status": "COMPLETED",
        "decision_status": "COMPLETE",
        "admitted_tickers": ["AAA"],
        "candidate_dispositions": [
            CandidateDisposition(
                ticker="AAA",
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="ACTIONABLE",
                watchlist_eligible=True,
                frontier_status="REVIEWED",
            )
        ],
        "selection_validation": SectorSelectionValidation(
            status="VALIDATED",
            selected_ticker="AAA",
            validator_run_id="validator-default",
            validator_verdict="CONFIRMED_ACTIONABLE",
            evidence_ref_ids=["validator-default:E-VALIDATION"],
            evidence=[
                EvidenceReference(
                    evidence_id="validator-default:E-VALIDATION",
                    source_type="tool_output",
                    source_label="challenge",
                    summary="Independent validation evidence.",
                    ticker="AAA",
                    tool_call_id="validator-default:VALIDATION-TC1",
                    confidence="HIGH",
                )
            ],
            tool_calls=[
                _validation_tool_call(
                    "validator-default:E-VALIDATION",
                    validator_run_id="validator-default",
                )
            ],
            provider_usage=_validator_provider_usage("validator-default"),
            source_binding=_source_binding("AAA", ["AAA"]),
        ),
        "competitive_frontier": _closed_frontier_payload(["AAA"]),
        "company_packets": _frontier_inputs(["AAA"])[0],
        "expected_return_scenarios": _frontier_inputs(["AAA"])[1],
        "contract_version": SECTOR_CONTRACT_VERSION_V2,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        AutonomousSectorFinancialRunArtifact(**values)


def test_v2_watchlist_rejects_actionable_unvalidated_competitor():
    with pytest.raises(ValueError, match="actionable competitor"):
        AutonomousSectorFinancialRunArtifact(
            run_id="ASFR-V2-WATCHLIST-COMPETITOR",
            sector="energy",
            market_cap_focus="large_and_mega",
            objective="Do not finalize beneath an actionable competitor.",
            as_of_date="2026-07-16",
            created_at="2026-07-16T15:00:00Z",
            status="COMPLETED",
            final_verdict="WATCHLIST",
            selected_ticker="AAA",
            confidence="MODERATE",
            pipeline_version="v2",
            execution_status="COMPLETED",
            decision_status="COMPLETE",
            admitted_tickers=["AAA", "BBB"],
            candidate_dispositions=[
                CandidateDisposition(
                    ticker="AAA",
                    terminal_state="UNDERWRITTEN",
                    scope_status="IN_SCOPE",
                    screen_status="PASS",
                    review_status="COMPLETED",
                    underwriting_verdict="WATCHLIST_ONLY",
                    watchlist_eligible=True,
                    frontier_status="REVIEWED",
                ),
                CandidateDisposition(
                    ticker="BBB",
                    terminal_state="UNDERWRITTEN",
                    scope_status="IN_SCOPE",
                    screen_status="PASS",
                    review_status="COMPLETED",
                    underwriting_verdict="ACTIONABLE",
                    watchlist_eligible=True,
                    frontier_status="REVIEWED",
                ),
            ],
            selection_validation=SectorSelectionValidation(
                status="NOT_ATTEMPTED",
                selected_ticker="BBB",
            ),
            competitive_frontier=_closed_frontier_payload(["AAA", "BBB"]),
            company_packets=_frontier_inputs(["AAA", "BBB"])[0],
            expected_return_scenarios=_frontier_inputs(["AAA", "BBB"])[1],
            contract_version=SECTOR_CONTRACT_VERSION_V2,
        )


def test_v2_no_selection_allows_unreviewed_dominated_queue_name_only_with_closed_frontier():
    tickers = ["AAA", "CCC", "DDD", "BBB"]
    packets, scenarios = _frontier_inputs(tickers)
    frontier = _closed_frontier_payload(
        tickers,
        reviewed_tickers=["AAA", "CCC", "DDD"],
    )
    proofs = {
        ticker: _bound_underwriting(
            ticker,
            "AVOID",
            frontier["source_bindings"][ticker],
        )
        for ticker in ("AAA", "CCC", "DDD")
    }
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="ASFR-V2-CLOSED-FRONTIER",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Close the live frontier without hiding dominated queue names.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T15:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        pipeline_version="v2",
        execution_status="COMPLETED",
        decision_status="COMPLETE",
        admitted_tickers=["AAA", "CCC", "DDD", "BBB"],
        candidate_dispositions=[
            CandidateDisposition(
                ticker="AAA",
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="AVOID",
                watchlist_eligible=False,
                frontier_status="REVIEWED",
                underwriting_result=proofs["AAA"][0],
            ),
            CandidateDisposition(
                ticker="BBB",
                terminal_state="READY_FOR_UNDERWRITING",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="NOT_STARTED",
                watchlist_eligible=True,
                frontier_status="DOMINATED",
                frontier_dominated_by=["AAA", "CCC", "DDD"],
            ),
            CandidateDisposition(
                ticker="CCC",
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="AVOID",
                watchlist_eligible=False,
                frontier_status="REVIEWED",
                underwriting_result=proofs["CCC"][0],
            ),
            CandidateDisposition(
                ticker="DDD",
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="AVOID",
                watchlist_eligible=False,
                frontier_status="REVIEWED",
                underwriting_result=proofs["DDD"][0],
            ),
        ],
        selection_validation=SectorSelectionValidation(status="NOT_REQUIRED"),
        competitive_frontier=frontier,
        company_packets=packets,
        expected_return_scenarios=scenarios,
        company_autonomy_runs=[proofs[ticker][1] for ticker in ("AAA", "CCC", "DDD")],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    assert artifact.decision_status == "COMPLETE"
    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.candidate_dispositions[1].frontier_status == "DOMINATED"


def test_v2_complete_decision_rejects_dominated_queue_name_when_frontier_is_open():
    open_frontier = _closed_frontier_payload(
        ["AAA", "CCC", "DDD", "BBB"],
        reviewed_tickers=["AAA", "CCC", "DDD"],
    )
    open_frontier["status"] = "OPEN"
    with pytest.raises(ValueError, match="closed competitive frontier"):
        AutonomousSectorFinancialRunArtifact(
            run_id="ASFR-V2-OPEN-FRONTIER",
            sector="energy",
            market_cap_focus="large_and_mega",
            objective="Do not close an open frontier.",
            as_of_date="2026-07-16",
            created_at="2026-07-16T15:00:00Z",
            status="COMPLETED",
            final_verdict="NO_SELECTION",
            selected_ticker=None,
            confidence=None,
            pipeline_version="v2",
            execution_status="COMPLETED",
            decision_status="COMPLETE",
            admitted_tickers=["AAA", "CCC", "DDD", "BBB"],
            candidate_dispositions=[
                CandidateDisposition(
                    ticker="AAA",
                    terminal_state="UNDERWRITTEN",
                    scope_status="IN_SCOPE",
                    screen_status="PASS",
                    review_status="COMPLETED",
                    underwriting_verdict="AVOID",
                    watchlist_eligible=False,
                    frontier_status="REVIEWED",
                ),
                CandidateDisposition(
                    ticker="BBB",
                    terminal_state="READY_FOR_UNDERWRITING",
                    scope_status="IN_SCOPE",
                    screen_status="PASS",
                    review_status="NOT_STARTED",
                    watchlist_eligible=True,
                    frontier_status="DOMINATED",
                    frontier_dominated_by=["AAA", "CCC", "DDD"],
                ),
                CandidateDisposition(
                    ticker="CCC",
                    terminal_state="UNDERWRITTEN",
                    scope_status="IN_SCOPE",
                    screen_status="PASS",
                    review_status="COMPLETED",
                    underwriting_verdict="AVOID",
                    watchlist_eligible=False,
                    frontier_status="REVIEWED",
                ),
                CandidateDisposition(
                    ticker="DDD",
                    terminal_state="UNDERWRITTEN",
                    scope_status="IN_SCOPE",
                    screen_status="PASS",
                    review_status="COMPLETED",
                    underwriting_verdict="AVOID",
                    watchlist_eligible=False,
                    frontier_status="REVIEWED",
                ),
            ],
            selection_validation=SectorSelectionValidation(status="NOT_REQUIRED"),
            competitive_frontier=open_frontier,
            contract_version=SECTOR_CONTRACT_VERSION_V2,
        )


def test_v2_complete_decision_rejects_omitted_competitive_frontier():
    with pytest.raises(ValueError, match="closed competitive frontier"):
        AutonomousSectorFinancialRunArtifact(
            run_id="ASFR-V2-MISSING-FRONTIER",
            sector="energy",
            market_cap_focus="large_and_mega",
            objective="Reject completion without a frontier proof.",
            as_of_date="2026-07-16",
            created_at="2026-07-16T15:00:00Z",
            status="COMPLETED",
            final_verdict="SELECTED",
            selected_ticker="AAA",
            confidence="HIGH",
            pipeline_version="v2",
            execution_status="COMPLETED",
            decision_status="COMPLETE",
            admitted_tickers=["AAA"],
            candidate_dispositions=[
                CandidateDisposition(
                    ticker="AAA",
                    terminal_state="UNDERWRITTEN",
                    scope_status="IN_SCOPE",
                    screen_status="PASS",
                    review_status="COMPLETED",
                    underwriting_verdict="ACTIONABLE",
                    watchlist_eligible=True,
                    frontier_status="REVIEWED",
                )
            ],
            selection_validation=SectorSelectionValidation(
                status="VALIDATED",
                selected_ticker="AAA",
                validator_run_id="validator-aaa",
                validator_verdict="CONFIRMED_ACTIONABLE",
                evidence_ref_ids=["validator-aaa:E-VALIDATION"],
                evidence=[
                    EvidenceReference(
                        evidence_id="validator-aaa:E-VALIDATION",
                        source_type="tool_output",
                        source_label="challenge",
                        summary="Independent validation evidence.",
                        ticker="AAA",
                        tool_call_id="validator-aaa:VALIDATION-TC1",
                        confidence="HIGH",
                    )
                ],
                tool_calls=[
                    _validation_tool_call(
                        "validator-aaa:E-VALIDATION",
                        validator_run_id="validator-aaa",
                    )
                ],
                provider_usage=_validator_provider_usage("validator-aaa"),
                source_binding=_source_binding("AAA", ["AAA"]),
            ),
            contract_version=SECTOR_CONTRACT_VERSION_V2,
        )


def test_autonomous_sector_financial_run_artifact_memo_body_round_trip():
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="ASFR-MEMO",
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        objective="Render memo body prose.",
        as_of_date="2026-05-07",
        created_at="2026-05-07T20:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        memo_body={
            "status": "PARTIAL_LLM_GENERATED",
            "degraded_states": ["candidate_thesis_fallback:BBB"],
            "cohort_comparison": {
                "source": "llm",
                "paragraphs": ["AAA has the strongest valuation asymmetry."],
            },
            "triage_surprises": {
                "source": "deterministic_fallback",
                "items": ["BBB fell from pre-rank 1 to final rank 3."],
            },
            "candidates": {
                "AAA": {
                    "source": "llm",
                    "thesis": "AAA is cheap but cyclically exposed.",
                    "key_risks": ["Margins may normalize lower."],
                    "falsifiers": ["Base return falls below 12.0%."],
                    "open_questions": ["What evidence supports backlog conversion?"],
                }
            },
            "usage": {
                "input_tokens": 1234,
                "output_tokens": 321,
                "cost_estimate_usd": 0.042,
            },
        },
    )

    restored = AutonomousSectorFinancialRunArtifact.from_dict(artifact.to_dict())

    assert restored.memo_body == artifact.memo_body
    assert restored.memo_body["status"] == "PARTIAL_LLM_GENERATED"
    assert restored.memo_body["degraded_states"] == ["candidate_thesis_fallback:BBB"]
    assert restored.memo_body["cohort_comparison"]["paragraphs"] == ["AAA has the strongest valuation asymmetry."]
    assert restored.memo_body["triage_surprises"]["items"] == ["BBB fell from pre-rank 1 to final rank 3."]
    assert restored.memo_body["candidates"]["AAA"]["open_questions"] == ["What evidence supports backlog conversion?"]
    assert restored.memo_body["usage"]["cost_estimate_usd"] == 0.042


def test_autonomous_sector_financial_run_artifact_full_round_trip():
    tool_call = ToolCallRecord(
        call_id="TC1",
        tool_name="fetch_kpi_trends",
        tool_input={},
        rationale="Need comparable KPI trend evidence.",
        status="OK",
        question_id="Q1",
        evidence_ref_ids=["E1"],
    )
    evidence = EvidenceReference(
        evidence_id="E1",
        source_type="tool_output",
        source_label="fetch_kpi_trends",
        summary="AAA showed stronger cash conversion than BBB.",
        ticker="AAA",
        tool_call_id="TC1",
        confidence="HIGH",
    )
    belief_update = BeliefUpdate(
        update_id="BU1",
        question_id="Q1",
        ticker="AAA",
        prior_belief="AAA may not clear the return hurdle.",
        updated_belief="AAA clears the base-case return hurdle with moderate confidence.",
        direction="BULLISH",
        confidence_after="MODERATE",
        summary="Expected return is supported by cash conversion and lower leverage.",
        evidence_ref_ids=["E1"],
        remaining_uncertainty=["Customer concentration still caps confidence."],
    )
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="ASFR-2",
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        objective="Select the best financially underwritten 5-10 year return candidate.",
        as_of_date="2026-04-26",
        created_at="2026-04-26T15:00:00Z",
        completed_at="2026-04-26T15:20:00Z",
        status="COMPLETED",
        final_verdict="SELECTED",
        selected_ticker="AAA",
        confidence="MODERATE",
        framework=_framework(),
        candidate_selection={"source": "sector_scan_db", "selected_tickers": ["AAA"]},
        company_packets=[_company_packet()],
        research_questions=[_research_question()],
        expected_return_scenarios=[_scenario()],
        tool_calls=[tool_call],
        evidence=[evidence],
        belief_updates=[belief_update],
        final_decision=_final_decision(),
        selection_audit={
            "status": "PASS",
            "selected_ticker": "AAA",
            "actionable": True,
            "confidence_ceiling": "HIGH",
            "hard_blockers": [],
            "confidence_caps": [],
            "expected_return_evidence_count": 1,
            "company_specific_evidence_count": 1,
            "base_return_hurdle": 0.12,
            "best_base_annualized_return": 0.156,
            "best_base_horizon_years": 5,
            "notes": ["Selection audit passed; candidate remains actionable."],
            "final_verdict_after_audit": "SELECTED",
        },
        framework_evidence_preflight=[
            {
                "ticker": "AAA",
                "packet_support_status": "PACKET_SUPPORT_PRESENT",
                "packet_supported_required_evidence": ["expected_return_scenarios"],
                "needs_tool_evidence": [],
                "packet_support_ratio": 1.0,
                "supporting_packet_fields": {"expected_return_scenarios": ["expected_return_scenarios"]},
                "suggested_tools_for_missing_evidence": {},
            }
        ],
        final_decision_prompt_context={
            "prompt_scoped_tickers": ["AAA"],
            "expected_return_evidence_by_finalist": {
                "AAA": {"evidence_count": 1, "has_repair_target": True}
            },
            "audit_gap_repair_targets": [{"ticker": "AAA", "status": "RESOLVED_SELECTED"}],
        },
        audit_gap_repair_attempted=True,
        audit_gap_repair_status="RESOLVED_SELECTED",
        audit_gap_repair_notes=["Repair evidence cleared the binding audit blockers."],
        watchlist_resolution_attempted=True,
        watchlist_resolution_status="RESOLVED_SELECTED",
        watchlist_resolution_notes=["Follow-up evidence cleared the watchlist audit caps."],
        no_selection_finalist_audit_attempted=True,
        no_selection_finalist_audit_status="PASS",
        no_selection_finalist_audit_focus_ticker="AAA",
        no_selection_finalist_audit_notes=["No-selection finalist audit passed."],
        no_selection_finalist_resolution_attempted=True,
        no_selection_finalist_resolution_status="RESOLVED_SELECTED",
        no_selection_finalist_resolution_notes=["Follow-up evidence cleared the audit caps."],
        alternate_finalist_audit_attempted=True,
        alternate_finalist_audit_status="NO_ALTERNATE_PASSED",
        alternate_finalist_audit_notes=["No alternate finalist cleared audit."],
        alternate_finalist_audit_results=[
            {
                "ticker": "BBB",
                "source": "provider_rejected_finalist",
                "audit_status": "BLOCKED",
                "hard_blockers": ["BASE_RETURN_BELOW_12PCT_HURDLE"],
                "confidence_caps": [],
                "best_base_annualized_return": 0.08,
            }
        ],
        company_autonomy_attempted=True,
        company_autonomy_status="COMPLETED",
        company_autonomy_notes=["Nested company autonomy completed for AAA."],
        company_autonomy_runs=[
            {
                "run_id": "autonomous_AAA_child",
                "ticker": "AAA",
                "status": "COMPLETED",
                "final_verdict": "ACTIONABLE",
                "confidence": "MODERATE",
            }
        ],
        company_autonomy_decision_trace={
            "status": "COMPLETED",
            "changed_sector_finalist_decision": False,
            "validated_sector_finalist_decision": True,
            "contradicted_sector_finalist_decision": False,
            "selected_ticker_child_verdict": "ACTIONABLE",
            "top_ranked_child_verdict": "ACTIONABLE",
            "impact_counts": {"VALIDATED_SELECTED": 1},
            "ticker_impacts": [
                {
                    "ticker": "AAA",
                    "child_status": "COMPLETED",
                    "child_verdict": "ACTIONABLE",
                    "audit_status_before": "PASS",
                    "audit_status_after": "PASS",
                    "actionable_before": True,
                    "actionable_after": True,
                    "hard_blockers_removed": [],
                    "confidence_caps_removed": [],
                    "impact": "VALIDATED_SELECTED",
                }
            ],
        },
        relative_ranking=[
            {
                "rank": 1,
                "ticker": "AAA",
                "best_base_annualized_return": 0.156,
                "audit_status": "PASS",
                "actionable": True,
                "company_autonomy_verdict": "ACTIONABLE",
                "positioning_summary": "Best-positioned and actionable under current guardrails.",
            }
        ],
        memo_body={
            "status": "LLM_GENERATED",
            "cohort_comparison": {"source": "llm", "paragraphs": ["AAA leads the cohort."]},
            "triage_surprises": {"source": "llm", "items": ["No surprises."]},
            "candidates": {
                "AAA": {
                    "source": "llm",
                    "thesis": "AAA has the strongest memo-body case.",
                    "key_risks": ["Margins normalize lower."],
                    "falsifiers": ["Base return falls below 12.0%."],
                    "open_questions": ["What sustains cash conversion?"],
                }
            },
        },
        audit_notes=["Sector framework and final decision are reconstructable."],
    )

    restored = AutonomousSectorFinancialRunArtifact.from_dict(artifact.to_dict())

    assert restored == artifact
    assert restored.framework is not None
    assert restored.framework.valid_valuation_methods == ["owner_earnings_multiple", "dcf", "epv"]
    assert restored.final_decision_prompt_context == {
        "prompt_scoped_tickers": ["AAA"],
        "expected_return_evidence_by_finalist": {
            "AAA": {"evidence_count": 1, "has_repair_target": True}
        },
        "audit_gap_repair_targets": [{"ticker": "AAA", "status": "RESOLVED_SELECTED"}],
    }
    assert restored.candidate_selection == {"source": "sector_scan_db", "selected_tickers": ["AAA"]}
    assert restored.company_packets[0].valuation == {"owner_earnings_yield": 0.08, "base_case_value": 43.0}
    assert restored.expected_return_scenarios[0].annualized_return == 0.156
    assert restored.research_questions[0].planned_tools == ["fetch_kpi_trends", "fetch_filing_section"]
    assert restored.tool_calls[0].evidence_ref_ids == ["E1"]
    assert restored.evidence[0].summary == "AAA showed stronger cash conversion than BBB."
    assert restored.belief_updates[0].confidence_after == "MODERATE"
    assert restored.final_decision is not None
    assert restored.final_decision.expected_annualized_return_range == "14-18%"
    assert restored.selection_audit["status"] == "PASS"
    assert restored.selection_audit["best_base_annualized_return"] == 0.156
    assert restored.framework_evidence_preflight[0]["packet_support_status"] == "PACKET_SUPPORT_PRESENT"
    assert restored.framework_evidence_preflight[0]["packet_support_ratio"] == 1.0
    assert restored.audit_gap_repair_attempted is True
    assert restored.audit_gap_repair_status == "RESOLVED_SELECTED"
    assert restored.audit_gap_repair_notes == ["Repair evidence cleared the binding audit blockers."]
    assert restored.watchlist_resolution_attempted is True
    assert restored.watchlist_resolution_status == "RESOLVED_SELECTED"
    assert restored.watchlist_resolution_notes == ["Follow-up evidence cleared the watchlist audit caps."]
    assert restored.no_selection_finalist_audit_attempted is True
    assert restored.no_selection_finalist_audit_status == "PASS"
    assert restored.no_selection_finalist_audit_focus_ticker == "AAA"
    assert restored.no_selection_finalist_audit_notes == ["No-selection finalist audit passed."]
    assert restored.no_selection_finalist_resolution_attempted is True
    assert restored.no_selection_finalist_resolution_status == "RESOLVED_SELECTED"
    assert restored.no_selection_finalist_resolution_notes == ["Follow-up evidence cleared the audit caps."]
    assert restored.alternate_finalist_audit_attempted is True
    assert restored.alternate_finalist_audit_status == "NO_ALTERNATE_PASSED"
    assert restored.alternate_finalist_audit_notes == ["No alternate finalist cleared audit."]
    assert restored.alternate_finalist_audit_results == [
        {
            "ticker": "BBB",
            "source": "provider_rejected_finalist",
            "audit_status": "BLOCKED",
            "hard_blockers": ["BASE_RETURN_BELOW_12PCT_HURDLE"],
            "confidence_caps": [],
            "best_base_annualized_return": 0.08,
        }
    ]
    assert restored.company_autonomy_attempted is True
    assert restored.company_autonomy_status == "COMPLETED"
    assert restored.company_autonomy_notes == ["Nested company autonomy completed for AAA."]
    assert restored.company_autonomy_runs[0]["final_verdict"] == "ACTIONABLE"
    assert restored.company_autonomy_decision_trace["impact_counts"] == {"VALIDATED_SELECTED": 1}
    assert restored.relative_ranking[0]["ticker"] == "AAA"
    assert restored.relative_ranking[0]["company_autonomy_verdict"] == "ACTIONABLE"
    assert restored.memo_body["status"] == "LLM_GENERATED"
    assert restored.memo_body["candidates"]["AAA"]["open_questions"] == ["What sustains cash conversion?"]
