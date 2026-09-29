from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.alpha.schemas import TickerSignalPacket
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    validate_financial_integrity_scope,
)
from app.autonomous.run_contract import (
    AutonomousRunArtifact,
    AutonomousRunBudget,
    AutonomousRunRequest,
    CandidateDecision,
    EvidenceReference,
    ToolCallRecord,
)
from app.autonomous.runtime import _PLAN_SCHEMA
from app.autonomous.competitive_frontier import build_competitive_frontier
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorFinalDecision,
    SectorFinancialFramework,
    SectorSelectionValidation,
    canonical_v2_signal_packet_snapshot,
)
from app.autonomous.sector_framework_templates import (
    CANONICAL_SECTOR_FRAMEWORK_CONTRACTS,
    sector_framework_pipeline_version,
    sector_framework_template_payload,
)
from app.autonomous.sector_runtime import (
    DEFAULT_SECTOR_BUDGET,
    _INITIAL_PLAN_SCHEMA,
    _MINIMUM_TOOL_PLAN_SCHEMA,
    _TURN_SCHEMA,
    _artifact_candidate_tickers,
    _call_provider_result_with_meta,
    _candidate_memo_prompt,
    _cohort_comparison_prompt,
    _company_autonomy_child_budget,
    _dispatch_sector_tool,
    _estimate_llm_cost_usd,
    _framework_requirement_packet_support,
    _framework_required_evidence_tool_hints,
    _expected_return_repair_targets,
    _finalize_sector_artifact_v2,
    _framework_from_payload,
    _initial_plan_prompt,
    _is_quota_exhaustion,
    _v2_lane_budget_payload,
    _v2_lane_usage_payload,
    _memo_shared_context,
    _minimum_tool_plan_prompt,
    _persist_memo_provider_usage,
    _pre_provider_financial_history_filter,
    _pre_provider_framework_evidence_filter,
    _rebuild_v2_competitive_frontier_before_finalization,
    _recovery_final_decision_prompt,
    _run_broad_expected_return_repair_pass,
    _run_finalist_child_research_pass,
    _run_v2_company_underwriting_child,
    _run_v2_competitive_frontier,
    _run_v2_selected_company_validation,
    _selection_audit_for_ticker,
    _synthesize_provider_json_with_meta,
    _turn_prompt,
    _v2_child_source_bindings,
    _v2_structural_gate_results,
    _v2_screen_state,
    LLM_COST_BUDGET_EXCEEDED,
    classify_audit_signal,
    enrich_sector_artifact_memo_body,
    run_sector_autonomous_financial_analysis,
    sector_artifact_summary,
)
from app.autonomous.sector_runtime import (
    _cached_report_financial_snapshot_years as _real_cached_report_financial_snapshot_years,
)
from app.llm.providers.retry_guard import (
    LLMCostBudgetExceeded,
    LLMMaxRetriesExceeded,
    llm_cost_budget,
)
from app.llm.providers import DeepSeekOutputTruncatedError
from app.llm.usage_capture import (
    attached_provider_usage_records,
    attach_provider_usage_to_exception,
    provider_usage_capture,
)


@pytest.fixture(autouse=True)
def _bypass_financial_integrity_for_legacy_sector_runtime_tests(monkeypatch):
    """Keep legacy runtime fixtures isolated from the dedicated gate contract tests."""

    from app.autonomous import sector_runtime as sector_runtime_module

    gate_result = SimpleNamespace(
        status="PASS",
        passed=True,
        is_valid=True,
        scope_fingerprint="legacy-sector-runtime-fixture",
        violations=(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.require_financial_integrity_scope",
        lambda scope: gate_result,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.require_unchanged_financial_integrity_scope",
        lambda scope, *, expected_scope_fingerprint: gate_result,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._cached_report_financial_snapshot_years",
        lambda ticker, *, as_of_date: 3,
    )

    def legacy_financial_context(*, tickers, as_of_date, db_path, **_kwargs):
        normalized = [str(ticker).strip().upper() for ticker in tickers]
        issuer_contexts = {ticker: {} for ticker in normalized}
        return SimpleNamespace(
            as_of_date=as_of_date,
            packets=sector_runtime_module.assemble_sector_packets(
                normalized,
                filing_risk_use_llm=False,
                as_of_date=as_of_date,
                pipeline_version="v1",
                current_prices={ticker: None for ticker in normalized},
                issuer_contexts=issuer_contexts,
                db_path=db_path,
            ),
            current_prices={ticker: None for ticker in normalized},
            issuer_contexts=issuer_contexts,
        )

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_canonical_v1_financial_context",
        legacy_financial_context,
    )


def test_default_sector_budget_has_no_cost_ceiling():
    assert DEFAULT_SECTOR_BUDGET.max_cost_usd is None
    assert _company_autonomy_child_budget().max_cost_usd is None


@pytest.mark.parametrize(
    "sector",
    sorted(CANONICAL_SECTOR_FRAMEWORK_CONTRACTS),
)
def test_v2_runtime_framework_persists_explicit_contract_id_for_every_sector(sector):
    payload = sector_framework_template_payload(
        sector,
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
    )

    with sector_framework_pipeline_version("v2"):
        framework = _framework_from_payload(
            {"framework": payload},
            sector=sector,
            market_cap_focus="large_and_mega",
        )

    assert framework is not None
    assert framework.framework_contract_id == CANONICAL_SECTOR_FRAMEWORK_CONTRACTS[sector]


def test_v2_structural_gate_receives_packet_issuer_identity(monkeypatch, tmp_path):
    calls: list[tuple[str, dict]] = []

    def fake_evaluate(ticker: str, **kwargs):
        calls.append((ticker, kwargs))
        return SimpleNamespace(to_dict=lambda: {"ticker": ticker, "status": "PASS"})

    monkeypatch.setattr(
        "app.autonomous.structural_gate.evaluate_structural_gate",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.get_config",
        lambda: SimpleNamespace(db_path=tmp_path / "engine.db"),
    )
    packet = SectorCompanyFinancialPacket(
        ticker="issuer.adr",
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        issuer_cik="0000123456",
        issuer_primary_ticker="issuer",
        issuer_listed_tickers=["issuer", "issuer.adr", "issuer.b"],
        cap_stage_price=25.0,
        market_cap_mm=50_000.0,
    )

    results = _v2_structural_gate_results(
        sector="energy",
        as_of_date="2026-06-11",
        company_packets=[packet],
    )

    assert results == {"ISSUER.ADR": {"ticker": "ISSUER.ADR", "status": "PASS"}}
    assert len(calls) == 1
    ticker, kwargs = calls[0]
    assert ticker == "ISSUER.ADR"
    assert kwargs["pipeline_version"] == "v2"
    assert kwargs["issuer_cik"] == "0000123456"
    assert kwargs["aliases"] == ("ISSUER", "ISSUER.ADR", "ISSUER.B")


class TestSectorRuntimeCostEstimate:
    def test_anthropic_haiku_uses_real_haiku_pricing(self):
        # claude-haiku-4-5 real pricing (0.001, 0.005) per 1k.
        # 1000 in + 1000 out -> 0.001 + 0.005 = 0.006, NOT the ~15x opus figure (0.09).
        cost = _estimate_llm_cost_usd(
            provider_name="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=1000,
        )
        assert cost == 0.006

    def test_anthropic_opus_uses_opus_pricing(self):
        cost = _estimate_llm_cost_usd(
            provider_name="anthropic",
            model="claude-opus-4-6",
            input_tokens=1000,
            output_tokens=1000,
        )
        assert cost == 0.09

    def test_openai_gpt_5_4_mini_uses_gpt5_family_pricing(self):
        cost = _estimate_llm_cost_usd(
            provider_name="openai",
            model="gpt-5.4-mini",
            input_tokens=1000,
            output_tokens=1000,
        )
        assert cost == 0.00525


class TestQuotaExhaustionDetection:
    def test_insufficient_quota_is_exhaustion(self):
        assert _is_quota_exhaustion(RuntimeError("status=429 insufficient_quota")) is True

    def test_breaker_open_with_insufficient_quota_is_exhaustion(self):
        exc = RuntimeError("OpenAI provider circuit breaker open: status=429 insufficient_quota")
        assert _is_quota_exhaustion(exc) is True

    def test_transient_rate_limit_is_not_exhaustion(self):
        # A bare transient 429 / rate limit must be left to the retry guard,
        # NOT treated as quota exhaustion that forces the expensive fallback.
        assert _is_quota_exhaustion(RuntimeError("status=429 rate limit")) is False

    def test_bare_circuit_breaker_open_is_not_exhaustion(self):
        assert _is_quota_exhaustion(RuntimeError("circuit breaker open")) is False


class FakeProvider:
    provider_name = "test"

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return SimpleNamespace(json_text=json.dumps(payload))


class DisabledProvider:
    def enabled(self) -> bool:
        return False


class QuotaProvider:
    provider_name = "openai"

    def __init__(self):
        self.calls: list[dict] = []

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("OpenAI provider circuit breaker open: status=429 insufficient_quota")


class MetadataFallbackProvider:
    provider_name = "anthropic"

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        return SimpleNamespace(
            json_text=json.dumps({"status": "ok"}),
            model="claude-haiku-4-5",
            usage_input_tokens=10,
            usage_output_tokens=2,
        )


class AlwaysRateLimitedProvider:
    provider_name = "anthropic"

    def __init__(self):
        self.calls: list[dict] = []

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("status=429 rate limit")


class RecoveringProvider:
    def __init__(self):
        self.calls: list[dict] = []

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(json_text=json.dumps(_planning_turn()))
        if len(self.calls) == 2:
            raise RuntimeError("OpenAI output is not valid JSON")
        return SimpleNamespace(json_text=json.dumps(_final_turn()["final_decision"]))


class FinalTurnFailingProvider:
    def __init__(
        self,
        planning_payload: dict | None = None,
        turn_exc: Exception | None = None,
        recovery_exc: Exception | None = None,
    ):
        self.planning_payload = planning_payload or _planning_turn()
        self.turn_exc = turn_exc or TimeoutError("sector turn timed out")
        self.recovery_exc = recovery_exc or TimeoutError("final recovery timed out")
        self.calls: list[dict] = []

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(json_text=json.dumps(self.planning_payload))
        if len(self.calls) == 2:
            raise self.turn_exc
        raise self.recovery_exc


class FirstTurnRecoveringProvider:
    def __init__(
        self,
        first_exc: Exception,
        recovery_payload: dict | None = None,
        second_exc: Exception | None = None,
    ):
        self.first_exc = first_exc
        self.recovery_payload = recovery_payload or _minimum_plan_turn()
        self.second_exc = second_exc
        self.calls: list[dict] = []

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise self.first_exc
        if len(self.calls) == 2:
            if self.second_exc:
                raise self.second_exc
            return SimpleNamespace(json_text=json.dumps(self.recovery_payload))
        return SimpleNamespace(json_text=json.dumps(_final_turn()))


def _budget(max_tool_calls: int = 6, max_turns: int = 4) -> AutonomousRunBudget:
    return AutonomousRunBudget(
        max_tool_calls=max_tool_calls,
        max_turns=max_turns,
        max_cost_usd=3.0,
        timebox_seconds=None,
        max_candidates=3,
    )


def _raise_invalid_financial_input_at_recovery(scope) -> SimpleNamespace:
    if scope.context == "autonomous_sector_final_decision_recovery":
        invalid_result = validate_financial_integrity_scope(
            FinancialIntegrityScope(
                context=scope.context,
                run_as_of_date="2026-04-26",
                packets=(),
            )
        )
        raise InvalidFinancialInputError(invalid_result)
    return SimpleNamespace(
        status="PASS",
        passed=True,
        is_valid=True,
        scope_fingerprint="legacy-sector-runtime-fixture",
        violations=(),
    )


def _prompt_state(prompt: str, marker: str) -> dict:
    return json.loads(prompt.split(marker, 1)[1])


def _signal_packet(
    ticker: str, dcf_value: float, current_price: float, revenue_cagr: float
) -> TickerSignalPacket:
    return TickerSignalPacket(
        ticker=ticker,
        dcf_value=dcf_value,
        epv_value=dcf_value * 0.80,
        current_price=current_price,
        margin_of_safety_verdict="UNDERVALUED",
        gate_verdict="PROCEED",
        confidence_class="MODERATE",
        moat_score=4,
        filing_risk_status="OK",
        research_status="OK",
        solvency_risk="LOW",
        method_tension_type="NONE",
        growth_dependency_ratio=0.30,
        consensus_direction="UNDERVALUED",
        quarterly_revenue_trend="STABLE",
        raw_quality_ctx={
            "revenue_cagr_5y": revenue_cagr,
            "earnings_quality": "HIGH",
            "cash_conversion_ratio": 0.90,
            "dilution_rate_shares_cagr": 0.01,
        },
        raw_valuation={"sector": "specialty_manufacturing"},
    )


def _packets() -> dict[str, TickerSignalPacket]:
    return {
        "AAA": _signal_packet("AAA", 100.0, 50.0, 0.10),
        "BBB": _signal_packet("BBB", 80.0, 70.0, 0.04),
    }


def test_all_missing_valuation_anchors_stop_before_provider(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])
    packets = _packets()
    for packet in packets.values():
        packet.dcf_value = None
        packet.epv_value = None
    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.evidence_resolution.pre_assembly_data_gap_repair",
        lambda **_kwargs: {"status": "SKIPPED_PRE_AUTHORIZATION", "candidate_states": []},
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_canonical_v1_financial_context",
        lambda **_kwargs: SimpleNamespace(
            packets=packets,
            current_prices={"AAA": 50.0, "BBB": 70.0},
            issuer_contexts={"AAA": {}, "BBB": {}},
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
        candidate_selection={
            "source": "explicit_tickers",
            "selected_tickers": ["AAA", "BBB"],
            "cap_classifications": {},
        },
    )

    assert provider.calls == []
    assert artifact.status == "FAILED"
    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.degraded_states == ["NEEDS_DATA"]
    assert artifact.no_selection_reason == (
        "Deterministic valuation-anchor gate stopped before any sector provider "
        "work: NEEDS_DATA: MISSING_VALUATION across all admitted companies."
    )
    assert artifact.candidate_selection["valuation_anchor_filter"] == {
        "status": "NEEDS_DATA_ALL_CANDIDATES_MISSING_VALUATION",
        "input_tickers": ["AAA", "BBB"],
        "ready_tickers": [],
        "needs_data_tickers": ["AAA", "BBB"],
        "reason_code": "MISSING_VALUATION",
        "dispositions": [
            {
                "ticker": "AAA",
                "terminal_state": "NEEDS_DATA",
                "scope_status": "IN_SCOPE",
                "screen_status": "INCOMPLETE",
                "review_status": "NOT_STARTED",
                "underwriting_verdict": None,
                "underwriting_confidence": None,
                "watchlist_eligible": False,
                "issuer_key": None,
                "issuer_cik": None,
                "primary_ticker": None,
                "security_type": None,
                "is_secondary_class": None,
                "is_adr": None,
                "adr_ratio": None,
                "share_class_ratio": None,
                "identity_source_url": None,
                "ratio_source_url": None,
                "reason_codes": ["MISSING_VALUATION"],
                "evidence_ref_ids": [],
                "last_completed_stage": "VALUATION_ANCHOR_PREFLIGHT",
                "frontier_status": None,
                "frontier_dominated_by": [],
                "screen_result": None,
                "underwriting_result": None,
            },
            {
                "ticker": "BBB",
                "terminal_state": "NEEDS_DATA",
                "scope_status": "IN_SCOPE",
                "screen_status": "INCOMPLETE",
                "review_status": "NOT_STARTED",
                "underwriting_verdict": None,
                "underwriting_confidence": None,
                "watchlist_eligible": False,
                "issuer_key": None,
                "issuer_cik": None,
                "primary_ticker": None,
                "security_type": None,
                "is_secondary_class": None,
                "is_adr": None,
                "adr_ratio": None,
                "share_class_ratio": None,
                "identity_source_url": None,
                "ratio_source_url": None,
                "reason_codes": ["MISSING_VALUATION"],
                "evidence_ref_ids": [],
                "last_completed_stage": "VALUATION_ANCHOR_PREFLIGHT",
                "frontier_status": None,
                "frontier_dominated_by": [],
                "screen_result": None,
                "underwriting_result": None,
            },
        ],
    }
    assert [row.ticker for row in artifact.candidate_dispositions] == ["AAA", "BBB"]
    assert [row.reason_codes for row in artifact.candidate_dispositions] == [
        ["MISSING_VALUATION"],
        ["MISSING_VALUATION"],
    ]


def test_mixed_valuation_anchors_run_valued_subset_and_record_disposition(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])
    packets = _packets()
    packets["BBB"].dcf_value = None
    packets["BBB"].epv_value = None
    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.evidence_resolution.pre_assembly_data_gap_repair",
        lambda **_kwargs: {"status": "SKIPPED_PRE_AUTHORIZATION", "candidate_states": []},
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_canonical_v1_financial_context",
        lambda **_kwargs: SimpleNamespace(
            packets=packets,
            current_prices={"AAA": 50.0, "BBB": 70.0},
            issuer_contexts={"AAA": {}, "BBB": {}},
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} completed for {ctx.ticker}.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
        candidate_selection={
            "source": "explicit_tickers",
            "selected_tickers": ["AAA", "BBB"],
            "cap_classifications": {},
        },
    )

    assert len(provider.calls) == 2
    assert [packet.ticker for packet in artifact.company_packets] == ["AAA"]
    assert {scenario.ticker for scenario in artifact.expected_return_scenarios} == {"AAA"}
    assert artifact.candidate_selection["valuation_anchor_filter"]["status"] == (
        "MIXED_VALUED_SUBSET"
    )
    assert artifact.candidate_selection["valuation_anchor_filter"]["ready_tickers"] == ["AAA"]
    assert artifact.candidate_selection["valuation_anchor_filter"]["needs_data_tickers"] == [
        "BBB"
    ]
    assert artifact.candidate_selection["selected_tickers"] == ["AAA"]
    assert len(artifact.candidate_dispositions) == 1
    disposition = artifact.candidate_dispositions[0]
    assert disposition.ticker == "BBB"
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.reason_codes == ["MISSING_VALUATION"]
    assert disposition.last_completed_stage == "VALUATION_ANCHOR_PREFLIGHT"


def _audit_packet(ticker: str = "AAA") -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status="OK",
        current_price=50.0,
        valuation={
            "valuation_anchor": 100.0,
            "anchor_method": "dcf",
            "available_methods": ["dcf"],
        },
    )


def _audit_packet_with_impairment(
    impairment_class: str,
    *,
    caution: str = "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT",
    reason_codes: list[str] | None = None,
) -> SectorCompanyFinancialPacket:
    packet = _audit_packet("AAA")
    packet.business_quality = {
        "impairment_classification": {
            "impairment_class_primary": impairment_class,
            "primary_underwriting_caution": caution,
            "impairment_class_reason_codes": reason_codes or ["NO_MOS_CONFIRMED"],
            "impairment_support_signals": ["NO_MOS_CONFIRMED"],
            "impairment_rebuttal_signals": [],
        },
        "impairment_class_primary": impairment_class,
        "primary_underwriting_caution": caution,
    }
    return packet


def _custom_sector_packet(
    ticker: str,
    *,
    data_quality_status: str = "OK",
    valuation_anchor: float = 100.0,
    current_price: float = 50.0,
) -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status=data_quality_status,
        current_price=current_price,
        valuation={
            "valuation_anchor": valuation_anchor,
            "anchor_method": "dcf",
            "available_methods": ["dcf"],
        },
    )


def _audit_base_scenario(
    ticker: str = "AAA", annualized_return: float = 0.18
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


def _audit_downside_scenario(
    ticker: str = "AAA", annualized_return: float = -0.02
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


def _audit_upside_scenario(
    ticker: str = "AAA", annualized_return: float = 0.28
) -> SectorExpectedReturnScenario:
    return SectorExpectedReturnScenario(
        scenario_id=f"{ticker}_upside_5Y",
        ticker=ticker,
        scenario_name="upside",
        horizon_years=5,
        current_price=50.0,
        estimated_future_value_per_share=125.0,
        annualized_return=annualized_return,
    )


def _audit_scenario_stack(
    ticker: str, base_return: float = 0.18
) -> list[SectorExpectedReturnScenario]:
    return [
        _audit_base_scenario(ticker, base_return),
        _audit_downside_scenario(ticker, -0.02),
        _audit_upside_scenario(ticker, 0.28),
    ]


def test_expected_return_repair_targets_top_watchlist_and_actionable_rows():
    relative_ranking = [
        {
            "ticker": "AAA",
            "rank": 1,
            "company_autonomy_verdict": "WATCHLIST_ONLY",
            "hard_blockers": ["MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE"],
        },
        {
            "ticker": "BBB",
            "rank": 2,
            "company_autonomy_verdict": "ACTIONABLE",
            "hard_blockers": ["MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE"],
        },
        {
            "ticker": "CCC",
            "rank": 3,
            "company_autonomy_verdict": "AVOID",
            "hard_blockers": ["MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE"],
        },
        {
            "ticker": "DDD",
            "rank": 4,
            "company_autonomy_verdict": "WATCHLIST_ONLY",
            "hard_blockers": ["MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE"],
        },
    ]
    scenarios = [
        *_audit_scenario_stack("AAA", 0.18),
        *_audit_scenario_stack("BBB", 0.16),
        _audit_base_scenario("DDD", 0.20),
        _audit_upside_scenario("DDD", 0.30),
    ]

    targets = _expected_return_repair_targets(
        relative_ranking=relative_ranking, scenarios=scenarios
    )

    assert targets == [
        {"ticker": "AAA", "rank": 1, "best_base_horizon_years": 5},
        {"ticker": "BBB", "rank": 2, "best_base_horizon_years": 5},
    ]


def test_broad_expected_return_repair_runs_two_scoped_tools_per_target():
    packets_by_ticker = {"AAA": _custom_sector_packet("AAA"), "BBB": _custom_sector_packet("BBB")}
    scenarios = [*_audit_scenario_stack("AAA", 0.18), *_audit_scenario_stack("BBB", 0.16)]

    (
        questions,
        tool_calls,
        evidence,
        belief_updates,
        degraded,
        notes,
        executed_count,
        metadata,
    ) = _run_broad_expected_return_repair_pass(
        sector="enterprise_software",
        as_of_date="2026-05-17",
        repair_targets=[
            {"ticker": "AAA", "rank": 1, "best_base_horizon_years": 5},
            {"ticker": "BBB", "rank": 2, "best_base_horizon_years": 5},
        ],
        signal_packets={},
        packets_by_ticker=packets_by_ticker,
        scenarios=scenarios,
        tool_calls=[],
        evidence=[],
        allowed_tools=["compare_expected_return_scenarios", "rank_expected_return_cases"],
        budget=_budget(max_tool_calls=4),
        executed_tool_count=0,
        question_start_index=1,
    )

    assert [question.target_tickers for question in questions] == [["AAA"], ["BBB"]]
    assert [call.tool_name for call in tool_calls] == [
        "compare_expected_return_scenarios",
        "rank_expected_return_cases",
        "compare_expected_return_scenarios",
        "rank_expected_return_cases",
    ]
    assert [call.tool_input["tickers"] for call in tool_calls] == [
        ["AAA"],
        ["AAA"],
        ["BBB"],
        ["BBB"],
    ]
    assert [item.source_label for item in evidence] == [
        "compare_expected_return_scenarios",
        "rank_expected_return_cases",
        "compare_expected_return_scenarios",
        "rank_expected_return_cases",
    ]
    assert belief_updates == []
    assert degraded == []
    assert notes == [
        "Deterministic expected-return audit-gap repair targeted 2 candidate(s); 0 skipped for budget."
    ]
    assert executed_count == 4
    assert metadata["status"] == "EXECUTED"
    assert metadata["targets"] == [
        {"ticker": "AAA", "status": "OK"},
        {"ticker": "BBB", "status": "OK"},
    ]


def test_broad_expected_return_repair_skips_low_ranked_tail_when_budget_constrained():
    packets_by_ticker = {
        "AAA": _custom_sector_packet("AAA"),
        "BBB": _custom_sector_packet("BBB"),
        "CCC": _custom_sector_packet("CCC"),
    }
    scenarios = [
        *_audit_scenario_stack("AAA", 0.18),
        *_audit_scenario_stack("BBB", 0.16),
        *_audit_scenario_stack("CCC", 0.15),
    ]

    (
        _questions,
        tool_calls,
        _evidence,
        _belief_updates,
        degraded,
        notes,
        executed_count,
        metadata,
    ) = _run_broad_expected_return_repair_pass(
        sector="enterprise_software",
        as_of_date="2026-05-17",
        repair_targets=[
            {"ticker": "AAA", "rank": 1, "best_base_horizon_years": 5},
            {"ticker": "BBB", "rank": 2, "best_base_horizon_years": 5},
            {"ticker": "CCC", "rank": 3, "best_base_horizon_years": 5},
        ],
        signal_packets={},
        packets_by_ticker=packets_by_ticker,
        scenarios=scenarios,
        tool_calls=[],
        evidence=[],
        allowed_tools=["compare_expected_return_scenarios", "rank_expected_return_cases"],
        budget=_budget(max_tool_calls=3),
        executed_tool_count=0,
        question_start_index=1,
    )

    assert [call.tool_input["tickers"] for call in tool_calls] == [["AAA"], ["AAA"]]
    assert degraded == ["AUDIT_GAP_REPAIR_BUDGET_PARTIAL"]
    assert notes == [
        "Deterministic expected-return audit-gap repair targeted 1 candidate(s); 2 skipped for budget."
    ]
    assert executed_count == 2
    assert metadata["status"] == "PARTIAL"
    assert metadata["targets"] == [
        {"ticker": "AAA", "status": "OK"},
        {"ticker": "BBB", "status": "SKIPPED_BUDGET_EXHAUSTED"},
        {"ticker": "CCC", "status": "SKIPPED_BUDGET_EXHAUSTED"},
    ]


def test_broad_expected_return_repair_no_targets_is_noop():
    (
        questions,
        tool_calls,
        evidence,
        belief_updates,
        degraded,
        notes,
        executed_count,
        metadata,
    ) = _run_broad_expected_return_repair_pass(
        sector="enterprise_software",
        as_of_date="2026-05-17",
        repair_targets=[],
        signal_packets={},
        packets_by_ticker={"AAA": _custom_sector_packet("AAA")},
        scenarios=_audit_scenario_stack("AAA", 0.18),
        tool_calls=[],
        evidence=[],
        allowed_tools=["compare_expected_return_scenarios", "rank_expected_return_cases"],
        budget=_budget(max_tool_calls=4),
        executed_tool_count=0,
        question_start_index=1,
    )

    assert questions == []
    assert tool_calls == []
    assert evidence == []
    assert belief_updates == []
    assert degraded == []
    assert notes == []
    assert executed_count == 0
    assert metadata == {"attempted": False, "status": "NO_OP", "notes": [], "targets": []}


def test_memo_shared_context_truncates_large_cohort_to_top_25():
    tickers = [f"T{i:02d}" for i in range(30)]
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_large_context_test",
        sector="enterprise_software",
        market_cap_focus="mid_cap",
        objective="Review every loaded candidate.",
        as_of_date="2026-05-12",
        created_at="2026-05-12T00:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        candidate_selection={"selected_tickers": tickers},
        company_packets=[_custom_sector_packet(ticker) for ticker in tickers],
        relative_ranking=[
            {
                "ticker": ticker,
                "rank": index + 1,
                "best_base_annualized_return": 0.30 - index * 0.01,
            }
            for index, ticker in enumerate(tickers)
        ],
    )

    context = _memo_shared_context(artifact)

    assert _artifact_candidate_tickers(artifact) == tickers
    assert context["candidate_order"] == tickers[:25]
    assert [item["ticker"] for item in context["candidates"]] == tickers[:25]
    assert [item["ticker"] for item in context["relative_ranking"]] == tickers[:25]
    assert context["candidate_context_scope"] == {
        "total_candidates": 30,
        "included_candidates": 25,
        "omitted_candidates": 5,
        "selection_rule": "top 25 by deterministic pre-rank order",
        "tail_summary": "and 5 other candidates with lower deterministic pre-rank priority",
    }


def test_memo_shared_context_compacts_audit_and_final_decision_payloads():
    tickers = [f"T{i:02d}" for i in range(25)]
    huge_evidence = [{"evidence_id": f"E{i}", "summary": "x" * 500} for i in range(100)]
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_compact_context_test",
        sector="media_entertainment",
        market_cap_focus="small_cap",
        objective="Review every loaded candidate.",
        as_of_date="2026-05-16",
        created_at="2026-05-16T00:00:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker="T00",
        confidence="MODERATE",
        candidate_selection={"selected_tickers": tickers},
        company_packets=[_custom_sector_packet(ticker) for ticker in tickers],
        expected_return_scenarios=[_audit_base_scenario(ticker, 0.20) for ticker in tickers],
        selection_audit={
            "status": "PASS",
            "final_verdict_after_audit": "WATCHLIST",
            "hard_blockers": ["ONE", "TWO", "THREE"],
            "confidence_caps": ["CAP_ONE", "CAP_TWO", "CAP_THREE"],
            "evidence_reference_details": huge_evidence,
            "evidence_long_form_notes": ["y" * 1000],
        },
        final_decision=SectorFinalDecision(
            verdict="WATCHLIST",
            confidence="MODERATE",
            selected_ticker="T00",
            expected_annualized_return_range="12-15%",
            thesis="z" * 5000,
            key_risk="k" * 5000,
            downside_case="d" * 5000,
            no_selection_reason=None,
            selection_blockers=["BLOCK_ONE", "BLOCK_TWO", "BLOCK_THREE"],
            confidence_cap_reasons=["CAP_ONE", "CAP_TWO", "CAP_THREE"],
        ),
        relative_ranking=[
            {
                "ticker": ticker,
                "rank": index + 1,
                "best_base_annualized_return": 0.30 - index * 0.01,
                "confidence_caps": ["CAP_ONE", "CAP_TWO", "CAP_THREE"],
                "hard_blockers": ["BLOCK_ONE", "BLOCK_TWO", "BLOCK_THREE"],
                "positioning_summary": "p" * 2000,
            }
            for index, ticker in enumerate(tickers)
        ],
    )

    context = _memo_shared_context(artifact)
    serialized = json.dumps(context, sort_keys=True)

    assert len(serialized) // 4 < 30000
    assert "evidence_reference_details" not in serialized
    assert "evidence_long_form_notes" not in serialized
    assert "z" * 100 not in serialized
    assert context["selection_audit"]["confidence_caps"] == ["CAP_ONE", "CAP_TWO"]
    assert context["final_decision"] == {
        "verdict": "WATCHLIST",
        "selected_ticker": "T00",
        "confidence": "MODERATE",
        "expected_annualized_return_range": "12-15%",
        "no_selection_reason": None,
        "selection_blockers": ["BLOCK_ONE", "BLOCK_TWO"],
        "confidence_cap_reasons": ["CAP_ONE", "CAP_TWO"],
    }


def test_shared_context_compaction_does_not_trim_single_candidate_memo_context():
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_candidate_context_test",
        sector="media_entertainment",
        market_cap_focus="small_cap",
        objective="Review candidates.",
        as_of_date="2026-05-16",
        created_at="2026-05-16T00:00:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker="AAA",
        confidence="MODERATE",
        candidate_selection={"selected_tickers": ["AAA"]},
        company_packets=[_custom_sector_packet("AAA")],
        expected_return_scenarios=[_audit_base_scenario("AAA", 0.20)],
        relative_ranking=[
            {"ticker": "AAA", "rank": 1, "positioning_summary": "full candidate context"}
        ],
    )

    cohort_prompt = _cohort_comparison_prompt(artifact)
    candidate_prompt = _candidate_memo_prompt(
        artifact,
        ticker="AAA",
        cohort={"source": "llm", "paragraphs": ["AAA leads."]},
        triage={"source": "llm", "items": ["No surprises."]},
    )

    assert '"balance_sheet"' not in cohort_prompt
    assert '"balance_sheet"' in candidate_prompt


def test_memo_relative_ranking_mutation_blocks_next_physical_call(monkeypatch):
    from app.autonomous import sector_runtime as sector_runtime_module

    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_memo_prompt_integrity",
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        objective="Preserve exact deterministic memo inputs.",
        as_of_date="2026-05-16",
        created_at="2026-05-16T00:00:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker="AAA",
        confidence="MODERATE",
        candidate_selection={"selected_tickers": ["AAA"]},
        company_packets=[_custom_sector_packet("AAA")],
        expected_return_scenarios=[_audit_base_scenario("AAA", 0.10)],
        relative_ranking=[
            {
                "ticker": "AAA",
                "rank": 1,
                "best_base_annualized_return": 0.10,
            }
        ],
    )
    provider = FakeProvider(
        [
            {"paragraphs": ["AAA leads the one-company cohort at a 10% base return."]},
            {"items": ["This response must never be requested."]},
        ]
    )
    real_triage_prompt = sector_runtime_module._triage_surprises_prompt

    def mutate_after_cohort_production(current_artifact, cohort):
        current_artifact.relative_ranking[0]["best_base_annualized_return"] = 0.99
        return real_triage_prompt(current_artifact, cohort)

    monkeypatch.setattr(
        "app.autonomous.sector_runtime._triage_surprises_prompt",
        mutate_after_cohort_production,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        sector_runtime_module._enrich_sector_artifact_memo_body_impl(
            artifact,
            provider=provider,
        )

    assert [item.code for item in exc_info.value.violations] == [
        "BOUND_FINANCIAL_PROMPT_STATE_MUTATED"
    ]
    assert len(provider.calls) == 1
    assert len(attached_provider_usage_records(exc_info.value)) == 1
    assert len(artifact.provider_usage) == 1


def _large_prompt_packets(
    count: int = 100,
) -> tuple[list[SectorCompanyFinancialPacket], list[SectorExpectedReturnScenario]]:
    tickers = [f"T{i:03d}" for i in range(count)]
    packets = [_custom_sector_packet(ticker) for ticker in tickers]
    scenarios = [
        _audit_base_scenario(ticker, annualized_return=0.30 - index * 0.001)
        for index, ticker in enumerate(tickers)
    ]
    return packets, scenarios


def _estimated_prompt_tokens(prompt: str) -> int:
    return len(prompt) // 4


def test_sector_loop_prompts_truncate_large_candidate_context_to_top_25():
    packets, scenarios = _large_prompt_packets(100)
    expected_scope = {
        "total_candidates": 100,
        "included_candidates": 25,
        "omitted_candidates": 75,
        "selection_rule": "first 25 in runtime candidate order",
        "ranked_tickers": [f"T{i:03d}" for i in range(25)],
        "tail_summary": "and 75 other candidates with similar profile",
    }

    initial_state = _prompt_state(
        _initial_plan_prompt(
            sector="enterprise_software",
            market_cap_focus="mid_cap",
            objective="Review the expanded universe.",
            as_of_date="2026-05-15",
            budget=_budget(),
            allowed_tools=["rank_expected_return_cases"],
            company_packets=packets,
            scenarios=scenarios,
        ),
        "Compact planning state: ",
    )
    minimum_state = _prompt_state(
        _minimum_tool_plan_prompt(
            sector="enterprise_software",
            market_cap_focus="mid_cap",
            objective="Review the expanded universe.",
            as_of_date="2026-05-15",
            allowed_tools=["rank_expected_return_cases"],
            company_packets=packets,
            scenarios=scenarios,
            provider_error="context_length_exceeded",
        ),
        "Minimum planning state: ",
    )
    turn_state = _prompt_state(
        _turn_prompt(
            turn_index=2,
            sector="enterprise_software",
            market_cap_focus="mid_cap",
            objective="Review the expanded universe.",
            as_of_date="2026-05-15",
            budget=_budget(),
            allowed_tools=["rank_expected_return_cases"],
            company_packets=packets,
            scenarios=scenarios,
            framework=None,
            research_questions=[],
            tool_calls=[],
            evidence=[],
            belief_updates=[],
        ),
        "Run state: ",
    )
    recovery_state = _prompt_state(
        _recovery_final_decision_prompt(
            sector="enterprise_software",
            market_cap_focus="mid_cap",
            objective="Review the expanded universe.",
            as_of_date="2026-05-15",
            company_packets=packets,
            scenarios=scenarios,
            research_questions=[],
            tool_calls=[],
            evidence=[],
            belief_updates=[],
            provider_error="context_length_exceeded",
        ),
        "Recovery state: ",
    )

    for state in (initial_state, minimum_state, turn_state, recovery_state):
        assert state["candidate_context_scope"] == expected_scope
        assert [item["ticker"] for item in state["company_packets"]] == [
            f"T{i:03d}" for i in range(25)
        ]

    shuffled_state = _prompt_state(
        _initial_plan_prompt(
            sector="enterprise_software",
            market_cap_focus="mid_cap",
            objective="Review the expanded universe.",
            as_of_date="2026-05-15",
            budget=_budget(),
            allowed_tools=["rank_expected_return_cases"],
            company_packets=list(reversed(packets)),
            scenarios=list(reversed(scenarios)),
        ),
        "Compact planning state: ",
    )
    assert shuffled_state["candidate_context_scope"] == {
        **expected_scope,
        "ranked_tickers": [f"T{i:03d}" for i in range(99, 74, -1)],
    }
    assert [item["ticker"] for item in shuffled_state["company_packets"]] == [
        f"T{i:03d}" for i in range(99, 74, -1)
    ]


def test_high_n_sector_prompts_stay_under_50k_estimated_tokens():
    packets, scenarios = _large_prompt_packets(100)
    prompts = [
        _initial_plan_prompt(
            sector="enterprise_software",
            market_cap_focus="mid_cap",
            objective="Review the expanded universe.",
            as_of_date="2026-05-15",
            budget=_budget(),
            allowed_tools=["rank_expected_return_cases"],
            company_packets=packets,
            scenarios=scenarios,
        ),
        _turn_prompt(
            turn_index=2,
            sector="enterprise_software",
            market_cap_focus="mid_cap",
            objective="Review the expanded universe.",
            as_of_date="2026-05-15",
            budget=_budget(),
            allowed_tools=["rank_expected_return_cases"],
            company_packets=packets,
            scenarios=scenarios,
            framework=None,
            research_questions=[],
            tool_calls=[],
            evidence=[],
            belief_updates=[],
        ),
        _recovery_final_decision_prompt(
            sector="enterprise_software",
            market_cap_focus="mid_cap",
            objective="Review the expanded universe.",
            as_of_date="2026-05-15",
            company_packets=packets,
            scenarios=scenarios,
            research_questions=[],
            tool_calls=[],
            evidence=[],
            belief_updates=[],
            provider_error="context_length_exceeded",
        ),
    ]

    assert [_estimated_prompt_tokens(prompt) < 50000 for prompt in prompts] == [True, True, True]


def test_pre_provider_financial_history_filter_excludes_sparse_uncapped_candidates(monkeypatch):
    packets = [
        _custom_sector_packet("AAA"),
        _custom_sector_packet("BBB"),
        _custom_sector_packet("CCC"),
    ]
    scenarios = [
        _audit_base_scenario("AAA", 0.18),
        _audit_base_scenario("BBB", 0.20),
        _audit_base_scenario("CCC", 0.16),
    ]
    signal_packets = {
        "AAA": _signal_packet("AAA", 100.0, 50.0, 0.10),
        "BBB": _signal_packet("BBB", 100.0, 50.0, 0.10),
        "CCC": _signal_packet("CCC", 100.0, 50.0, 0.10),
    }
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._cached_report_financial_snapshot_years",
        lambda ticker, *, as_of_date: {"AAA": 5, "BBB": 2, "CCC": 3}[ticker],
    )

    filtered_packets, filtered_scenarios, filtered_signal_packets, selection, states, notes = (
        _pre_provider_financial_history_filter(
            company_packets=packets,
            scenarios=scenarios,
            signal_packets=signal_packets,
            candidate_selection={
                "source": "sector_scan_db",
                "selected_tickers": ["AAA", "BBB", "CCC"],
                "excluded_tickers": [],
                "warnings": ["UNUSABLE_CANDIDATES_INCLUDED_FOR_FULL_REVIEW:1"],
                "ranking_basis": "consensus_pre_rank_all_loaded",
            },
            run_as_of_date="2026-05-15",
        )
    )

    assert [packet.ticker for packet in filtered_packets] == ["AAA", "CCC"]
    assert [scenario.ticker for scenario in filtered_scenarios] == ["AAA", "CCC"]
    assert sorted(filtered_signal_packets) == ["AAA", "CCC"]
    assert selection["selected_tickers"] == ["AAA", "CCC"]
    assert selection["excluded_tickers"] == ["BBB"]
    assert selection["warnings"] == [
        "UNUSABLE_CANDIDATES_INCLUDED_FOR_FULL_REVIEW:1",
        "SPARSE_FINANCIAL_HISTORY_FILTERED:1",
    ]
    assert selection["financial_history_filter"] == {
        "status": "FILTERED_SPARSE_FINANCIAL_HISTORY",
        "minimum_rows": 3,
        "input_tickers": ["AAA", "BBB", "CCC"],
        "selected_tickers_before_filter": ["AAA", "BBB", "CCC"],
        "selected_tickers_after_filter": ["AAA", "CCC"],
        "excluded_tickers": ["BBB"],
        "year_counts": {"BBB": 2},
    }
    assert states == ["SPARSE_FINANCIAL_HISTORY_FILTERED"]
    assert notes == [
        "Pre-provider financial-history filter excluded 1 candidate(s) with fewer than 3 FY table rows: BBB."
    ]


def test_cached_report_history_excludes_post_asof_and_undated_fy_rows(
    monkeypatch,
    tmp_path,
):
    import sqlite3

    from app.autonomous import sector_runtime

    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE companyfacts_facts(
                ticker TEXT,
                fiscal_year INTEGER,
                period_type TEXT,
                period_end TEXT,
                filed_date TEXT,
                accession TEXT,
                source_url TEXT,
                line_item TEXT,
                value REAL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO companyfacts_facts VALUES(
                'AAA', ?, 'FY', ?, ?, ?, ?, 'revenue', ?
            )
            """,
            [
                (
                    2023,
                    "2023-12-31",
                    "2024-02-01",
                    "0000000001-24-000001",
                    "https://www.sec.gov/Archives/edgar/data/1/filing-2023.htm",
                    100.0,
                ),
                (
                    2024,
                    "2024-12-31",
                    "2026-03-01",
                    "0000000001-26-000001",
                    "https://www.sec.gov/Archives/edgar/data/1/filing-2024.htm",
                    200.0,
                ),
                (
                    2025,
                    "2025-12-31",
                    None,
                    "0000000001-26-000002",
                    "https://www.sec.gov/Archives/edgar/data/1/filing-2025.htm",
                    300.0,
                ),
            ],
        )
    monkeypatch.setattr(
        sector_runtime,
        "get_config",
        lambda: SimpleNamespace(db_path=db_path),
    )

    assert (
        _real_cached_report_financial_snapshot_years(
            "AAA",
            as_of_date="2026-02-13",
        )
        == 1
    )


def test_pre_provider_financial_history_filter_skips_explicit_or_capped_candidate_sets(monkeypatch):
    packets = [_custom_sector_packet("AAA")]
    scenarios = [_audit_base_scenario("AAA", 0.18)]
    signal_packets = {"AAA": _signal_packet("AAA", 100.0, 50.0, 0.10)}
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._cached_report_financial_snapshot_years",
        lambda ticker, *, as_of_date: 0,
    )

    explicit = _pre_provider_financial_history_filter(
        company_packets=packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        candidate_selection={
            "source": "explicit_tickers",
            "selected_tickers": ["AAA"],
            "ranking_basis": "input_order",
        },
        run_as_of_date="2026-05-15",
    )
    capped = _pre_provider_financial_history_filter(
        company_packets=packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        candidate_selection={
            "source": "sector_scan_db",
            "selected_tickers": ["AAA"],
            "ranking_basis": "consensus_pre_rank_selectable",
        },
        run_as_of_date="2026-05-15",
    )

    assert explicit[0] == packets
    assert explicit[1] == scenarios
    assert explicit[2] == signal_packets
    assert explicit[3]["selected_tickers"] == ["AAA"]
    assert explicit[4] == []
    assert explicit[5] == []
    assert capped[0] == packets
    assert capped[1] == scenarios
    assert capped[2] == signal_packets
    assert capped[3]["selected_tickers"] == ["AAA"]
    assert capped[4] == []
    assert capped[5] == []


def test_v2_financial_history_filter_retains_sparse_candidates_as_needs_data(monkeypatch):
    packets = [_custom_sector_packet("AAA"), _custom_sector_packet("BBB")]
    scenarios = [
        _audit_base_scenario("AAA", 0.18),
        _audit_base_scenario("BBB", 0.20),
    ]
    signal_packets = {
        "AAA": _signal_packet("AAA", 100.0, 50.0, 0.10),
        "BBB": _signal_packet("BBB", 100.0, 50.0, 0.10),
    }
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._cached_report_financial_snapshot_years",
        lambda ticker, *, as_of_date: {"AAA": 5, "BBB": 1}[ticker],
    )

    result = _pre_provider_financial_history_filter(
        company_packets=packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        candidate_selection={
            "source": "sector_scan_db",
            "selected_tickers": ["AAA", "BBB"],
            "excluded_tickers": [],
            "warnings": [],
            "ranking_basis": "consensus_pre_rank_all_loaded",
        },
        run_as_of_date="2026-05-15",
        pipeline_version="v2",
    )

    assert result[0] == packets
    assert result[1] == scenarios
    assert result[2] == signal_packets
    assert result[3]["selected_tickers"] == ["AAA", "BBB"]
    assert result[3]["excluded_tickers"] == []
    assert result[3]["financial_history_filter"]["status"] == (
        "V2_RETAINED_SPARSE_FINANCIAL_HISTORY"
    )
    assert result[3]["financial_history_filter"]["needs_data_tickers"] == ["BBB"]
    assert result[3]["warnings"] == ["SPARSE_FINANCIAL_HISTORY_RETAINED_NEEDS_DATA:1"]
    assert result[4] == []
    assert result[5] == [
        "V2 retained 1 sparse-history candidate(s) as visible research states: BBB."
    ]


def test_v2_framework_filter_retains_zero_support_candidate(monkeypatch):
    packets = [_custom_sector_packet("AAA"), _custom_sector_packet("BBB")]
    scenarios = [
        _audit_base_scenario("AAA", 0.18),
        _audit_base_scenario("BBB", 0.20),
    ]
    signal_packets = {
        "AAA": _signal_packet("AAA", 100.0, 50.0, 0.10),
        "BBB": _signal_packet("BBB", 100.0, 50.0, 0.10),
    }
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._framework_evidence_preflight",
        lambda **kwargs: [
            {"ticker": "AAA", "packet_support_ratio": 0.5},
            {"ticker": "BBB", "packet_support_ratio": 0.0},
        ],
    )

    result = _pre_provider_framework_evidence_filter(
        sector="enterprise_software",
        market_cap_focus="large_and_mega",
        company_packets=packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        allowed_tools=["fetch_kpi_trends"],
        candidate_selection={
            "source": "sector_scan_db",
            "selected_tickers": ["AAA", "BBB"],
            "excluded_tickers": [],
            "warnings": [],
        },
        pipeline_version="v2",
    )

    assert result[0] == packets
    assert result[1] == scenarios
    assert result[2] == signal_packets
    assert result[3]["selected_tickers"] == ["AAA", "BBB"]
    assert result[3]["excluded_tickers"] == []
    assert result[3]["framework_evidence_filter"]["status"] == ("V2_RETAINED_ZERO_PACKET_SUPPORT")
    assert result[3]["framework_evidence_filter"]["needs_data_tickers"] == ["BBB"]
    assert result[3]["warnings"] == ["FRAMEWORK_ZERO_SUPPORT_RETAINED_NEEDS_DATA:1"]
    assert result[4] == []


def _passing_audit_tool_evidence() -> tuple[list[ToolCallRecord], list[EvidenceReference]]:
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]
    return tool_calls, evidence


def _child_company_artifact(
    ticker: str,
    *,
    labels: tuple[str, ...] = ("fetch_kpi_trends", "analyze_dilution"),
    final_verdict: str = "ACTIONABLE",
    status: str = "COMPLETED",
    evidence_confidence: str = "MODERATE",
    summaries: dict[str, str] | None = None,
    excerpts: dict[str, str] | None = None,
) -> AutonomousRunArtifact:
    budget = AutonomousRunBudget(
        max_tool_calls=4,
        max_turns=2,
        max_cost_usd=0.75,
        timebox_seconds=None,
        max_candidates=1,
    )
    request = AutonomousRunRequest(
        run_id=f"autonomous_{ticker}_child",
        objective="Underwrite the company candidate.",
        as_of_date="2026-04-26",
        created_at="2026-04-26T12:00:00Z",
        candidate_scope={"tickers": [ticker]},
        allowed_tools=list(labels),
        budget=budget,
    )
    tool_calls = [
        ToolCallRecord(
            call_id=f"CTC{idx}",
            tool_name=label,
            tool_input={},
            rationale=f"Gather {label} evidence.",
            status="OK",
            output_preview=summaries.get(label) if summaries else None,
            evidence_ref_ids=[f"CE{idx}"],
        )
        for idx, label in enumerate(labels, start=1)
    ]
    evidence = [
        EvidenceReference(
            evidence_id=f"CE{idx}",
            source_type="tool_output",
            source_label=label,
            summary=(summaries or {}).get(label, f"{label} evidence supports {ticker}."),
            ticker=ticker,
            tool_call_id=f"CTC{idx}",
            excerpt=(excerpts or {}).get(label),
            confidence=evidence_confidence,
        )
        for idx, label in enumerate(labels, start=1)
    ]
    return AutonomousRunArtifact(
        request=request,
        status=status,
        started_at="2026-04-26T12:00:00Z",
        completed_at="2026-04-26T12:01:00Z",
        final_verdict=final_verdict,
        selected_ticker=ticker if final_verdict == "ACTIONABLE" else None,
        confidence="MODERATE",
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
        audit_notes=["Child company autonomy completed."],
    )


def _scenario_map(
    scenarios: list[SectorExpectedReturnScenario],
) -> dict[str, list[SectorExpectedReturnScenario]]:
    mapped: dict[str, list[SectorExpectedReturnScenario]] = {}
    for scenario in scenarios:
        mapped.setdefault(scenario.ticker, []).append(scenario)
    return mapped


def _framework_payload() -> dict:
    return {
        "sector": "specialty_manufacturing",
        "market_cap_focus": "small_cap",
        "horizon_years": [5, 10],
        "economic_model": "Small-cap industrial financial underwriting.",
        "selected_value_drivers": ["per_share_growth", "cash_conversion"],
        "selected_metrics": ["revenue_cagr_5y", "discount_to_anchor"],
        "valid_valuation_methods": ["dcf", "epv"],
        "invalid_valuation_methods": ["unsupported_story_multiple"],
        "required_evidence": ["expected_return_scenarios", "kpi_trends"],
        "normalization_policy": {"normalize_cycle": True},
        "hurdle_rate_policy": {"base_case_minimum_annualized_return": 0.15},
        "weighting_policy": {"expected_return": 0.40},
        "sector_specific_risks": ["cyclical_margin_pressure"],
    }


def _planning_turn(
    *, continue_research: bool = True, include_second_company_tool: bool = True
) -> dict:
    planned_tool_calls = [
        {
            "tool_name": "rank_expected_return_cases",
            "ticker": None,
            "tool_input": {"horizon_years": 5, "scenario_name": "base"},
            "rationale": "Rank base-case expected returns.",
        },
        {
            "tool_name": "fetch_kpi_trends",
            "ticker": "AAA",
            "tool_input": {},
            "rationale": "Check whether AAA's financial quality supports the scenario.",
        },
    ]
    if include_second_company_tool:
        planned_tool_calls.append(
            {
                "tool_name": "analyze_liquidity_stress",
                "ticker": "AAA",
                "tool_input": {},
                "rationale": "Check whether liquidity or solvency risk offsets the selected return case.",
            }
        )
    return {
        "framework": _framework_payload(),
        "research_questions": [
            {
                "question_id": "Q1",
                "question": "Which candidate clears the expected-return hurdle on audited scenario math?",
                "financial_pillar": "Expected return",
                "expected_decision_impact": "Determines whether a selection can be underwritten.",
                "priority": "HIGH",
                "target_tickers": ["AAA", "BBB"],
                "planned_tool_calls": planned_tool_calls,
            }
        ],
        "belief_updates": [],
        "continue_research": continue_research,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": ["Initial sector framework selected."],
    }


def _alternate_audit_planning_turn(*, extra_alternate_tools: bool = False) -> dict:
    planned_tool_calls = [
        {
            "tool_name": "rank_expected_return_cases",
            "ticker": None,
            "tool_input": {"horizon_years": 5, "scenario_name": "base"},
            "rationale": "Rank base-case expected returns for all candidates.",
        },
        {
            "tool_name": "fetch_kpi_trends",
            "ticker": "BBB",
            "tool_input": {},
            "rationale": "Check whether BBB has quality evidence if the top finalist fails audit.",
        },
        {
            "tool_name": "analyze_liquidity_stress",
            "ticker": "BBB",
            "tool_input": {},
            "rationale": "Check whether BBB has enough balance-sheet evidence if audited.",
        },
    ]
    if extra_alternate_tools:
        planned_tool_calls.extend(
            [
                {
                    "tool_name": "fetch_kpi_trends",
                    "ticker": "CCC",
                    "tool_input": {},
                    "rationale": "Gather alternate evidence for CCC.",
                },
                {
                    "tool_name": "fetch_kpi_trends",
                    "ticker": "DDD",
                    "tool_input": {},
                    "rationale": "Gather alternate evidence for DDD.",
                },
                {
                    "tool_name": "fetch_kpi_trends",
                    "ticker": "EEE",
                    "tool_input": {},
                    "rationale": "Gather alternate evidence for EEE.",
                },
            ]
        )
    return {
        "framework": _framework_payload(),
        "research_questions": [
            {
                "question_id": "Q1",
                "question": "Which alternate finalist clears audit if the highest-return finalist is blocked?",
                "financial_pillar": "Expected return and audit fallback",
                "expected_decision_impact": "Determines whether no-selection has a viable alternate.",
                "priority": "HIGH",
                "target_tickers": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "planned_tool_calls": planned_tool_calls,
            }
        ],
        "belief_updates": [],
        "continue_research": True,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": ["Initial sector framework selected."],
    }


def _minimum_plan_turn() -> dict:
    return {
        "research_questions": [
            {
                "question_id": "R1",
                "question": "Which candidate clears the base expected-return hurdle and has usable company evidence?",
                "financial_pillar": "Provider recovery planning",
                "expected_decision_impact": "Restarts the autonomous loop after a provider planning failure.",
                "priority": "HIGH",
                "target_tickers": ["AAA", "BBB"],
                "planned_tool_calls": [
                    {
                        "tool_name": "rank_expected_return_cases",
                        "ticker": None,
                        "tool_input": {"horizon_years": 5, "scenario_name": "base"},
                        "rationale": "Recover by gathering expected-return evidence.",
                    },
                    {
                        "tool_name": "fetch_kpi_trends",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Recover by gathering company-specific evidence.",
                    },
                    {
                        "tool_name": "analyze_liquidity_stress",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Recover by gathering risk evidence for the selected company.",
                    },
                ],
            }
        ],
        "degraded_states": [],
        "audit_notes": ["Minimum tool plan recovered after first-turn provider failure."],
    }


def _final_turn(selected_ticker: str | None = "AAA", verdict: str = "SELECTED") -> dict:
    return {
        "framework": None,
        "research_questions": [],
        "belief_updates": [
            {
                "question_id": "Q1",
                "ticker": selected_ticker,
                "prior_belief": "The expected-return ranking needed quality confirmation.",
                "updated_belief": "AAA clears the base-case hurdle with better supporting KPI evidence than BBB.",
                "direction": "BULLISH",
                "confidence_after": "MODERATE",
                "summary": "Scenario and KPI evidence support AAA with remaining uncertainty.",
                "evidence_ref_ids": ["E1", "E2"],
                "remaining_uncertainty": ["Cycle durability still caps conviction."],
            }
        ],
        "continue_research": False,
        "final_decision": {
            "verdict": verdict,
            "confidence": "MODERATE" if selected_ticker else None,
            "selected_ticker": selected_ticker,
            "expected_annualized_return_range": "20-25%" if selected_ticker else None,
            "thesis": "AAA has the strongest underwritten per-share return case.",
            "key_risk": "Cyclical margin pressure could reduce the base-case return.",
            "downside_case": "Downside case still depends on the valuation anchor holding.",
            "no_selection_reason": None if selected_ticker else "No company cleared the hurdle.",
            "falsifiers": ["Base-case expected return falls below the hurdle."],
            "why_selected_over_finalists": ["AAA has stronger scenario-adjusted upside than BBB."],
            "rejected_finalists": [{"ticker": "BBB", "reason": "Lower expected return."}],
            "selection_blockers": [],
            "confidence_cap_reasons": ["CYCLE_DURABILITY_UNRESOLVED"],
            "evidence_ref_ids": ["E1", "E2"],
        },
        "degraded_states": [],
        "audit_notes": ["Final decision made after reviewing deterministic tool output."],
    }


def _high_confidence_final_turn() -> dict:
    turn = _final_turn()
    turn["final_decision"]["confidence"] = "HIGH"
    return turn


def test_sector_runtime_executes_ai_planned_tools_then_finalizes(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])
    dispatched: list[tuple[str, str]] = []
    canonical_context_calls: list[dict[str, object]] = []
    pre_assembly_repair_calls: list[dict[str, object]] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.evidence_resolution.pre_assembly_data_gap_repair",
        lambda **kwargs: (
            pre_assembly_repair_calls.append(kwargs)
            or {
                "status": "SKIPPED_PRE_AUTHORIZATION",
                "repaired": [],
                "repaired_count": 0,
            }
        ),
    )

    def canonical_context(*, tickers, as_of_date, db_path, **_kwargs):
        normalized = [str(ticker).upper() for ticker in tickers]
        canonical_context_calls.append(
            {
                "tickers": normalized,
                "as_of_date": as_of_date,
                "db_path": db_path,
            }
        )
        return SimpleNamespace(
            packets=_packets(),
            current_prices={"AAA": 50.0, "BBB": 70.0},
            issuer_contexts={"AAA": {}, "BBB": {}},
        )

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_canonical_v1_financial_context",
        canonical_context,
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append((name, ctx.ticker))
        return {"status": "ok", "ticker": ctx.ticker, "summary": "KPI trends support the scenario."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        objective="Pick the strongest financially underwritten compounder or stop.",
        as_of_date="2026-04-26",
        budget=_budget(),
        candidate_selection={
            "source": "explicit_tickers",
            "selected_tickers": ["AAA", "BBB"],
            "cap_classifications": {},
        },
    )

    assert len(canonical_context_calls) == 1
    assert canonical_context_calls[0]["tickers"] == ["AAA", "BBB"]
    assert canonical_context_calls[0]["as_of_date"] == "2026-04-26"
    assert len(pre_assembly_repair_calls) == 1
    assert pre_assembly_repair_calls[0]["tickers"] == ["AAA", "BBB"]
    assert pre_assembly_repair_calls[0]["apply_repairs"] is False
    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.confidence == "MODERATE"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["actionable"] is True
    assert artifact.selection_audit["expected_return_evidence_count"] == 1
    assert artifact.selection_audit["company_specific_evidence_count"] == 2
    assert artifact.selection_audit["selected_company_evidence_tools"] == [
        "analyze_liquidity_stress",
        "fetch_kpi_trends",
    ]
    assert artifact.selection_audit["selected_evidence_pillars"] == ["liquidity", "quality"]
    assert artifact.selection_audit["return_cushion_status"] == "CLEAR"
    assert artifact.framework is not None
    assert artifact.framework.economic_model == "Small-cap industrial financial underwriting."
    assert artifact.framework_evidence_preflight[0]["packet_supported_required_evidence"][:2] == [
        "expected_return_scenarios",
        "kpi_trends",
    ]
    assert artifact.framework_evidence_preflight[0]["needs_tool_evidence"] == [
        "backlog_or_order_trend_evidence",
        "pricing_and_input_cost_evidence",
        "maintenance_capex_evidence",
        "cycle_normalization_evidence",
    ]
    assert artifact.framework_evidence_preflight[0]["packet_support_ratio"] == 3 / 7
    assert [packet.ticker for packet in artifact.company_packets] == ["AAA", "BBB"]
    assert len(artifact.expected_return_scenarios) == 12
    assert artifact.research_questions[0].status == "ANSWERED"
    assert artifact.research_questions[0].planned_tools == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]
    assert [call.status for call in artifact.tool_calls] == ["OK", "OK", "OK"]
    ranking_preview = artifact.tool_calls[0].output_preview or ""
    assert '"evidence_status": "EXPECTED_RETURN_RANKING_AVAILABLE"' in ranking_preview
    assert artifact.evidence[0].summary == "Ranked 2 base 5Y expected-return case(s): AAA, BBB."
    assert [evidence.source_label for evidence in artifact.evidence] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]
    assert (
        artifact.belief_updates[0].updated_belief
        == "AAA clears the base-case hurdle with better supporting KPI evidence than BBB."
    )
    assert artifact.final_decision is not None
    assert artifact.final_decision.why_selected_over_finalists == [
        "AAA has stronger scenario-adjusted upside than BBB."
    ]
    assert dispatched == [("fetch_kpi_trends", "AAA"), ("analyze_liquidity_stress", "AAA")]
    assert len(provider.calls) == 2
    assert provider.calls[0]["schema_name"] == "autonomous_sector_initial_research_plan"
    assert provider.calls[0]["schema"] == _INITIAL_PLAN_SCHEMA
    assert "final_decision" not in provider.calls[0]["schema"]["properties"]
    assert "Compact planning state:" in provider.calls[0]["prompt"]
    assert "prior_research_questions" not in provider.calls[0]["prompt"]
    planning_state = _prompt_state(provider.calls[0]["prompt"], "Compact planning state: ")
    assert planning_state["deterministic_selection_guardrails"] == {
        "base_return_hurdle": 0.12,
        "selected_return_cushion_hurdle": 0.15,
        "guardrail_source": "runtime_selection_audit",
    }
    assert planning_state["framework_template_hint"]["required_evidence"] == [
        "backlog_or_order_trend_evidence",
        "pricing_and_input_cost_evidence",
        "maintenance_capex_evidence",
        "working_capital_evidence",
        "cycle_normalization_evidence",
    ]
    hints_by_requirement = {
        item["required_evidence"]: item["suggested_tools"]
        for item in planning_state["framework_required_evidence_tool_hints"]
    }
    assert hints_by_requirement["backlog_or_order_trend_evidence"] == [
        "fetch_kpi_trends",
        "fetch_companyfacts_timeseries",
        "fetch_filing_section",
        "fetch_recent_filing_context",
    ]
    preflight_by_ticker = {
        item["ticker"]: item for item in planning_state["framework_evidence_preflight"]
    }
    assert preflight_by_ticker["AAA"]["packet_support_status"] == "NEEDS_TOOL_EVIDENCE"
    assert preflight_by_ticker["AAA"]["packet_supported_required_evidence"] == [
        "working_capital_evidence",
    ]
    assert preflight_by_ticker["AAA"]["needs_tool_evidence"] == [
        "backlog_or_order_trend_evidence",
        "pricing_and_input_cost_evidence",
        "maintenance_capex_evidence",
        "cycle_normalization_evidence",
    ]
    assert preflight_by_ticker["AAA"]["packet_support_ratio"] == 0.2
    turn_state = _prompt_state(provider.calls[1]["prompt"], "Run state: ")
    assert {
        "required_evidence": "expected_return_scenarios",
        "suggested_tools": ["rank_expected_return_cases", "compare_expected_return_scenarios"],
    } in turn_state["framework_required_evidence_tool_hints"]
    assert turn_state["framework_evidence_preflight"][0]["packet_supported_required_evidence"][
        :2
    ] == [
        "expected_return_scenarios",
        "kpi_trends",
    ]


def test_sector_runtime_runs_nested_company_autonomy_for_prioritized_candidates(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])
    child_tickers: list[str] = []
    assembled_packets = _packets()

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: assembled_packets,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    def fake_child_run(ticker, **kwargs):
        child_tickers.append(ticker)
        assert kwargs["budget"].max_tool_calls == 4
        assert kwargs["budget"].max_turns == 2
        assert "rank_expected_return_cases" not in kwargs["allowed_tools"]
        assert kwargs["canonical_signal_packet"] is assembled_packets[ticker]
        assert kwargs["source_binding"]["artifact_type"] == ("v1_canonical_child_source_binding_v1")
        assert kwargs["source_binding"]["ticker"] == ticker
        assert kwargs["source_binding"]["as_of_date"] == "2026-04-26"
        assert len(kwargs["source_binding"]["signal_packet_fingerprint"]) == 64
        assert len(kwargs["source_binding"]["cohort_fingerprint"]) == 64
        assert kwargs["source_binding"]["financial_integrity_packet"]["ticker"] == ticker
        assert all(
            item["ticker"] == ticker
            for item in kwargs["source_binding"]["financial_integrity_scenarios"]
        )
        assert len(kwargs["source_binding"]["financial_integrity_scenarios_fingerprint"]) == 64
        return _child_company_artifact(ticker)

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=16, max_turns=6),
    )

    assert child_tickers == ["AAA", "BBB"]
    assert artifact.company_autonomy_attempted is True
    assert artifact.company_autonomy_status == "COMPLETED"
    assert [run["ticker"] for run in artifact.company_autonomy_runs] == ["AAA", "BBB"]
    assert artifact.company_autonomy_runs[0]["final_verdict"] == "ACTIONABLE"
    child_evidence = [
        item for item in artifact.evidence if item.source_type == "company_autonomous_run"
    ]
    assert [item.source_label for item in child_evidence[:2]] == [
        "fetch_kpi_trends",
        "analyze_dilution",
    ]
    assert all(item.ticker in {"AAA", "BBB"} for item in child_evidence)
    assert artifact.company_autonomy_decision_trace["status"] == "COMPLETED"
    assert artifact.company_autonomy_decision_trace["selected_ticker_child_verdict"] == "ACTIONABLE"
    assert artifact.company_autonomy_decision_trace["top_ranked_child_verdict"] == "ACTIONABLE"
    assert artifact.company_autonomy_decision_trace["impact_counts"]["VALIDATED_SELECTED"] == 1
    assert artifact.company_autonomy_decision_trace["validated_sector_finalist_decision"] is True
    assert artifact.relative_ranking[0]["ticker"] == "AAA"
    assert artifact.relative_ranking[0]["company_autonomy_verdict"] == "ACTIONABLE"
    assert len(provider.calls) == 2


def test_sector_runtime_applies_child_verdict_ceiling_when_audit_would_upgrade(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(ticker, final_verdict="WATCHLIST_ONLY"),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=16, max_turns=6),
    )

    selected_row = next(row for row in artifact.relative_ranking if row["ticker"] == "AAA")
    assert artifact.final_verdict == "WATCHLIST"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "WATCHLIST_ONLY"
    assert artifact.selection_audit["actionable"] is False
    assert artifact.selection_audit["final_verdict_after_audit"] == "WATCHLIST"
    assert artifact.selection_audit["llm_verdict_ceiling_applied"] == {
        "llm_verdict": "WATCHLIST_ONLY",
        "audit_verdict": "SELECTED",
        "final_verdict": "WATCHLIST",
        "bound": True,
    }
    assert "LLM_VERDICT_CEILING_BOUND" in artifact.degraded_states
    assert selected_row["audit_status"] == "WATCHLIST_ONLY"
    assert selected_row["actionable"] is False
    assert selected_row["company_autonomy_verdict"] == "WATCHLIST_ONLY"
    assert selected_row["child_company_autonomy_verdict"] == "WATCHLIST_ONLY"


def test_sector_runtime_keeps_actionable_when_child_and_audit_agree(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(ticker, final_verdict="ACTIONABLE"),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=16, max_turns=6),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["actionable"] is True
    assert artifact.selection_audit["llm_verdict_ceiling_applied"] == {
        "llm_verdict": "ACTIONABLE",
        "audit_verdict": "SELECTED",
        "final_verdict": "SELECTED",
        "bound": False,
    }


def test_sector_runtime_child_company_evidence_can_satisfy_company_specific_audit(monkeypatch):
    rank_only_plan = _planning_turn()
    rank_only_plan["research_questions"][0]["planned_tool_calls"] = [
        {
            "tool_name": "rank_expected_return_cases",
            "ticker": None,
            "tool_input": {"horizon_years": 5, "scenario_name": "base"},
            "rationale": "Rank base-case expected returns.",
        }
    ]
    provider = FakeProvider([rank_only_plan, _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Expected returns ranked.",
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=(
                "fetch_kpi_trends",
                "analyze_dilution",
                "fetch_companyfacts_timeseries",
                "fetch_filing_section",
            ),
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=16, max_turns=6),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["company_specific_evidence_count"] == 4
    assert artifact.selection_audit["selected_company_evidence_tools"] == [
        "analyze_dilution",
        "fetch_companyfacts_timeseries",
        "fetch_filing_section",
        "fetch_kpi_trends",
    ]
    assert artifact.selection_audit["selected_evidence_pillars"] == [
        "companyfacts",
        "dilution",
        "filing_freshness",
        "quality",
    ]
    assert artifact.selection_audit["framework_required_evidence_missing"] == []
    assert artifact.company_autonomy_attempted is True
    assert artifact.company_autonomy_decision_trace["changed_sector_finalist_decision"] is True
    assert artifact.company_autonomy_decision_trace["impact_counts"]["CHANGED_TO_ACTIONABLE"] == 1
    aaa_impact = next(
        item
        for item in artifact.company_autonomy_decision_trace["ticker_impacts"]
        if item["ticker"] == "AAA"
    )
    # AAA's missing company-specific evidence is a fetchable gap (EVIDENCE_QUALITY
    # ), so its pre-child status is DATA_INCOMPLETE rather than BLOCKED; the
    # child run supplies the evidence and lifts it to an actionable PASS.
    assert aaa_impact["audit_status_before"] == "DATA_INCOMPLETE"
    assert aaa_impact["audit_status_after"] == "PASS"
    assert aaa_impact["impact"] == "CHANGED_TO_ACTIONABLE"


def test_sector_runtime_records_child_company_autonomy_failure_without_aborting(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])
    calls: list[str] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    def fake_child_run(ticker, **kwargs):
        calls.append(ticker)
        if ticker == "BBB":
            raise RuntimeError("child provider failed")
        return _child_company_artifact(ticker)

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=16, max_turns=6),
    )

    assert calls == ["AAA", "BBB"]
    assert artifact.final_verdict == "SELECTED"
    assert artifact.company_autonomy_attempted is True
    assert artifact.company_autonomy_status == "PARTIAL"
    assert artifact.company_autonomy_runs[1]["ticker"] == "BBB"
    assert artifact.company_autonomy_runs[1]["status"] == "FAILED"
    assert artifact.company_autonomy_runs[1]["error"] == "child provider failed"
    bbb_impact = next(
        item
        for item in artifact.company_autonomy_decision_trace["ticker_impacts"]
        if item["ticker"] == "BBB"
    )
    assert bbb_impact["child_status"] == "FAILED"
    assert bbb_impact["impact"] == "CHILD_FAILED"
    assert artifact.company_autonomy_decision_trace["impact_counts"]["CHILD_FAILED"] == 1


def test_sector_runtime_filters_zero_framework_preflight_candidates_before_provider(monkeypatch):
    planning = _planning_turn()
    planning.pop("framework")
    provider = FakeProvider([planning, _final_turn()])
    aaa = _audit_packet("AAA")
    aaa.business_quality = {"revenue_cagr_5y": 0.12, "earnings_quality": "HIGH"}
    bbb = _audit_packet("BBB")
    bbb.business_quality = {}
    bbb.cash_conversion = {}
    bbb.balance_sheet = {}
    bbb.accounting_quality = {}

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_sector_company_financial_packets_from_signal_packets",
        lambda signal_packets, **kwargs: [aaa, bbb],
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_expected_return_scenarios_for_packets",
        lambda packets: {
            "AAA": [
                _audit_base_scenario("AAA"),
                _audit_downside_scenario("AAA", annualized_return=-0.02),
            ],
            "BBB": [
                _audit_base_scenario("BBB"),
                _audit_downside_scenario("BBB", annualized_return=-0.02),
            ],
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="automotive",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=4, max_turns=3),
        candidate_selection={
            "source": "sector_scan_db",
            "selected_tickers": ["AAA", "BBB"],
            "excluded_tickers": [],
            "warnings": [],
        },
    )

    planning_state = _prompt_state(provider.calls[0]["prompt"], "Compact planning state: ")
    assert [packet["ticker"] for packet in planning_state["company_packets"]] == ["AAA"]
    assert [packet.ticker for packet in artifact.company_packets] == ["AAA"]
    assert [scenario.ticker for scenario in artifact.expected_return_scenarios] == ["AAA", "AAA"]
    assert artifact.candidate_selection["selected_tickers_before_framework_filter"] == [
        "AAA",
        "BBB",
    ]
    assert artifact.candidate_selection["selected_tickers"] == ["AAA"]
    assert artifact.candidate_selection["excluded_tickers"] == ["BBB"]
    assert artifact.candidate_selection["warnings"] == ["FRAMEWORK_EVIDENCE_PREFLIGHT_FILTERED:1"]
    assert (
        artifact.candidate_selection["framework_evidence_filter"]["status"]
        == "FILTERED_ZERO_PACKET_SUPPORT"
    )
    assert artifact.candidate_selection["framework_evidence_filter"]["excluded_tickers"] == ["BBB"]
    assert "FRAMEWORK_EVIDENCE_PREFLIGHT_FILTERED" in artifact.degraded_states
    assert (
        "Pre-provider framework evidence filter excluded 1 candidate(s) with zero packet support: BBB."
        in artifact.audit_notes
    )


def test_sector_runtime_aborts_degraded_when_run_retry_budget_exceeded(monkeypatch):
    provider = AlwaysRateLimitedProvider()

    monkeypatch.setattr("app.autonomous.sector_runtime.SECTOR_LLM_RETRY_RUN_BUDGET", 1)
    monkeypatch.setattr("app.llm.providers.retry_guard.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=4, max_turns=3),
    )

    assert artifact.status == "DEGRADED"
    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.degraded_states == ["LLM_RETRY_BUDGET_EXCEEDED"]
    assert "LLM retry budget exceeded" in str(artifact.no_selection_reason)
    assert provider.calls[0]["schema_name"] == "autonomous_sector_initial_research_plan"
    assert len(provider.calls) == 2
    assert len(artifact.provider_usage) == 2
    assert [row["status"] for row in artifact.provider_usage] == ["ERROR", "ERROR"]
    assert [row["physical_attempt"] for row in artifact.provider_usage] == [1, 2]
    assert any("LLM retry guard recorded 2/1 retries" in note for note in artifact.audit_notes)


def test_strict_sector_failed_attempt_consumes_cost_reservation():
    class StrictFailureProvider:
        provider_name = "anthropic"
        model = "claude-sonnet-4-6"

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, **kwargs):
            _ = kwargs
            self.calls += 1
            raise RuntimeError("status=429 rate limit")

    provider = StrictFailureProvider()
    with (
        provider_usage_capture("parent_research") as usage,
        llm_cost_budget(max_cost_usd=1.0, strict_first_call=True) as cost_context,
        pytest.raises(LLMMaxRetriesExceeded, match="max retries \\(0\\)"),
    ):
        _call_provider_result_with_meta(
            provider,
            {
                "prompt": "Return one strict JSON result.",
                "schema": {"type": "object"},
                "schema_name": "strict_failed_attempt",
                "max_output_tokens": 100,
            },
        )

    assert provider.calls == 1
    assert len(usage) == 1
    assert usage[0]["status"] == "ERROR"
    assert usage[0]["will_retry"] is False
    assert cost_context.call_count == 1
    assert cost_context.reserved_cost_usd == 0.0
    assert cost_context.cumulative_cost_usd > 0.0
    assert cost_context.cumulative_cost_usd == cost_context.events[0].estimated_cost_usd


def test_strict_v1_sector_packet_assembly_disables_filing_risk_llm(monkeypatch):
    assembly_calls: list[tuple[list[str], dict]] = []

    def assemble(tickers, **kwargs):
        assembly_calls.append((list(tickers), dict(kwargs)))
        return _packets()

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.get_alpha_llm_provider",
        lambda: DisabledProvider(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        assemble,
    )

    with llm_cost_budget(max_cost_usd=1.0, strict_first_call=True):
        artifact = run_sector_autonomous_financial_analysis(
            sector="specialty_manufacturing",
            tickers=["AAA", "BBB"],
            as_of_date="2026-04-26",
            budget=_budget(max_tool_calls=4, max_turns=3),
            pipeline_version="v1",
        )

    assert artifact.degraded_states == ["LLM_PROVIDER_UNAVAILABLE"]
    assert len(assembly_calls) == 1
    tickers, kwargs = assembly_calls[0]
    assert tickers == ["AAA", "BBB"]
    assert kwargs["filing_risk_use_llm"] is False
    assert kwargs["as_of_date"] == "2026-04-26"
    assert kwargs["pipeline_version"] == "v1"
    assert kwargs["current_prices"] == {"AAA": None, "BBB": None}
    assert kwargs["issuer_contexts"] == {"AAA": {}, "BBB": {}}
    assert kwargs["db_path"].name == "engine.db"


def test_sector_runtime_cost_budget_allows_under_budget_run(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(name)
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence supports the candidate.",
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=4, max_turns=3),
    )

    assert artifact.status == "COMPLETED"
    assert artifact.final_verdict == "SELECTED"
    assert len(provider.calls) == 2
    assert not any("LLM cost guard recorded" in note for note in artifact.audit_notes)


def test_sector_runtime_aborts_degraded_when_cost_budget_exceeded(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence.",
        },
    )
    budget = _budget(max_tool_calls=4, max_turns=3)
    budget.max_cost_usd = 0.000001

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=budget,
    )

    assert artifact.status == "DEGRADED"
    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.degraded_states == [LLM_COST_BUDGET_EXCEEDED]
    assert len(provider.calls) == 1
    assert "LLM cost budget exceeded" in str(artifact.no_selection_reason)
    assert any("LLM cost guard recorded $" in note for note in artifact.audit_notes)


def test_sector_runtime_cost_budget_none_disables_enforcement(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence.",
        },
    )
    budget = _budget(max_tool_calls=4, max_turns=3)
    budget.max_cost_usd = None

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=budget,
    )

    assert artifact.status == "COMPLETED"
    assert artifact.final_verdict == "SELECTED"
    assert len(provider.calls) == 2
    assert not any("LLM cost guard recorded" in note for note in artifact.audit_notes)


def test_provider_fallback_usage_records_original_failure(monkeypatch):
    primary = QuotaProvider()
    fallback = MetadataFallbackProvider()
    monkeypatch.setattr("app.autonomous.sector_runtime.get_anthropic_provider", lambda: fallback)

    payload, meta = _synthesize_provider_json_with_meta(
        primary,
        prompt="Return JSON.",
        schema={"type": "object"},
        schema_name="fallback_metadata_test",
        max_output_tokens=100,
    )

    assert payload == {"status": "ok"}
    assert meta["fallback_from_provider"] == "openai"
    assert meta["original_failure"] == (
        "RuntimeError: OpenAI provider circuit breaker open: status=429 insufficient_quota"
    )
    assert meta["provider"] == "anthropic"
    assert meta["model"] == "claude-haiku-4-5"


def test_strict_cost_context_forbids_cross_provider_quota_fallback(monkeypatch):
    primary = QuotaProvider()
    fallback_calls: list[dict] = []
    fallback = MetadataFallbackProvider()
    original = fallback.synthesize_json

    def tracked_fallback(**kwargs):
        fallback_calls.append(kwargs)
        return original(**kwargs)

    fallback.synthesize_json = tracked_fallback
    monkeypatch.setattr("app.autonomous.sector_runtime.get_anthropic_provider", lambda: fallback)

    with llm_cost_budget(10.0, strict_first_call=True):
        with pytest.raises(RuntimeError, match="insufficient_quota"):
            _synthesize_provider_json_with_meta(
                primary,
                prompt="Return JSON.",
                schema={"type": "object"},
                schema_name="strict_no_fallback",
                max_output_tokens=100,
            )

    assert fallback_calls == []


def test_sector_runtime_no_selection_preserves_relative_ranking(monkeypatch):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = []
    provider = FakeProvider([_planning_turn(), no_selection_turn])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.relative_ranking[0]["ticker"] == "AAA"
    assert artifact.relative_ranking[0]["rank"] == 1
    assert artifact.relative_ranking[1]["ticker"] == "BBB"
    assert "positioning_summary" in artifact.relative_ranking[0]


def test_sector_runtime_audits_no_selection_finalist_and_can_upgrade(monkeypatch):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = []
    provider = FakeProvider([_planning_turn(), no_selection_turn])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.no_selection_finalist_audit_attempted is True
    assert artifact.no_selection_finalist_audit_status == "PASS"
    assert artifact.no_selection_finalist_audit_focus_ticker == "AAA"
    assert artifact.no_selection_finalist_audit_notes == [
        "Audited AAA after provider no-selection: highest available base-case expected return among audited no-selection finalists.",
        "No-selection finalist audit passed; runtime upgraded the guarded result to SELECTED.",
    ]
    assert artifact.final_decision is not None
    assert (
        artifact.final_decision.expected_annualized_return_range
        == "Audited base case: 25.2% annualized"
    )
    assert any("No-selection finalist audit upgraded AAA" in note for note in artifact.audit_notes)


def test_sector_runtime_no_selection_finalist_repairs_expected_return_gap_without_child_run(
    monkeypatch,
):
    # Pin legacy hard-block hurdle behavior (production default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    company_only_plan = _planning_turn()
    company_only_plan["research_questions"][0]["planned_tool_calls"] = [
        {
            "tool_name": "fetch_kpi_trends",
            "ticker": "AAA",
            "tool_input": {},
            "rationale": "Check whether AAA's financial quality supports selection.",
        },
        {
            "tool_name": "analyze_liquidity_stress",
            "ticker": "AAA",
            "tool_input": {},
            "rationale": "Check whether AAA's risk evidence supports selection.",
        },
    ]
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = []
    provider = FakeProvider([company_only_plan, no_selection_turn])
    child_calls: list[dict] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Company evidence is usable.",
        },
    )

    def fake_child_run(ticker, **kwargs):
        child_calls.append({"ticker": ticker, **kwargs})
        return _child_company_artifact(ticker, labels=("fetch_kpi_trends",))

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.no_selection_finalist_audit_attempted is True
    assert artifact.no_selection_finalist_audit_status == "PASS"
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "RESOLVED_SELECTED"
    assert artifact.selection_audit["expected_return_evidence_count"] == 1
    assert artifact.selection_audit["company_specific_evidence_count"] == 2
    assert [call.tool_name for call in artifact.tool_calls] == [
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
        "rank_expected_return_cases",
    ]
    assert child_calls == []


def test_sector_runtime_no_selection_finalist_audit_keeps_blocked_result_non_actionable(
    monkeypatch,
):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = [
        {"ticker": "BBB", "reason": "Provider considered it."}
    ]
    no_selection_turn["final_decision"]["thesis"] = "AAA is below the 7% minimum hurdle."
    rank_only_plan = _planning_turn()
    rank_only_plan["research_questions"][0]["planned_tool_calls"] = [
        {
            "tool_name": "rank_expected_return_cases",
            "ticker": None,
            "tool_input": {"horizon_years": 5, "scenario_name": "base"},
            "rationale": "Rank base-case expected returns.",
        }
    ]
    provider = FakeProvider([rank_only_plan, no_selection_turn])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=1),
    )

    # The finalist AAA's only obstacle is MISSING_COMPANY_SPECIFIC_EVIDENCE,
    # which now classifies EVIDENCE_QUALITY (a fetchable data gap) . With
    # the repair budget exhausted the gap cannot be resolved in-run, so AAA is
    # surfaced as a DATA_INCOMPLETE resolve-then-promote candidate (ticker
    # preserved, confidence None) rather than discarded as a hard NO_SELECTION.
    assert artifact.final_verdict == "DATA_INCOMPLETE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.confidence is None
    assert artifact.no_selection_finalist_audit_attempted is True
    assert artifact.no_selection_finalist_audit_status == "DATA_INCOMPLETE"
    assert artifact.no_selection_finalist_audit_focus_ticker == "AAA"
    assert artifact.selection_audit["status"] == "DATA_INCOMPLETE"
    assert "MISSING_COMPANY_SPECIFIC_EVIDENCE" in artifact.selection_audit["hard_blockers"]
    assert "SELECTION_AUDIT_DATA_INCOMPLETE" in artifact.degraded_states
    assert artifact.final_decision is not None
    assert "MISSING_COMPANY_SPECIFIC_EVIDENCE" in artifact.final_decision.data_resolution_needed
    assert artifact.final_decision.selection_blockers == []


def test_sector_runtime_no_selection_finalist_audit_can_repair_and_upgrade(monkeypatch):
    rank_only_plan = _planning_turn()
    rank_only_plan["research_questions"][0]["planned_tool_calls"] = [
        {
            "tool_name": "rank_expected_return_cases",
            "ticker": None,
            "tool_input": {"horizon_years": 5, "scenario_name": "base"},
            "rationale": "Rank base-case expected returns.",
        }
    ]
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = []
    provider = FakeProvider([rank_only_plan, no_selection_turn])
    child_calls: list[dict] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Company evidence is usable.",
        },
    )

    def fake_child_run(ticker, **kwargs):
        child_calls.append({"ticker": ticker, **kwargs})
        return _child_company_artifact(
            ticker, labels=("fetch_kpi_trends", "analyze_liquidity_stress")
        )

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.no_selection_finalist_audit_attempted is True
    assert artifact.no_selection_finalist_audit_status == "PASS"
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "RESOLVED_SELECTED"
    assert artifact.selection_audit["status"] == "PASS"
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]
    assert len(provider.calls) == 2
    assert child_calls[0]["ticker"] == "AAA"
    assert "rank_expected_return_cases" not in child_calls[0]["allowed_tools"]


def test_sector_runtime_no_selection_finalist_audit_can_select_when_caps_do_not_block(monkeypatch):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = []
    provider = FakeProvider(
        [
            _planning_turn(include_second_company_tool=False),
            no_selection_turn,
        ]
    )
    child_calls: list[dict] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Follow-up evidence is usable.",
        },
    )

    def fake_child_run(ticker, **kwargs):
        child_calls.append({"ticker": ticker, **kwargs})
        return _child_company_artifact(ticker, labels=("analyze_liquidity_stress",))

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.no_selection_finalist_audit_attempted is True
    assert artifact.no_selection_finalist_audit_status == "PASS"
    assert artifact.no_selection_finalist_resolution_attempted is False
    assert artifact.no_selection_finalist_resolution_status is None
    assert artifact.watchlist_resolution_attempted is False
    assert artifact.watchlist_resolution_status is None
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
    ]
    assert len(provider.calls) == 2
    assert child_calls == []
    assert artifact.no_selection_finalist_resolution_notes == []


def test_sector_runtime_no_selection_finalist_audit_selects_despite_unresolved_evidence_caps(
    monkeypatch,
):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = []
    provider = FakeProvider(
        [
            _planning_turn(include_second_company_tool=False),
            no_selection_turn,
        ]
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "analyze_liquidity_stress":
            return {
                "status": "unavailable",
                "ticker": ctx.ticker,
                "usable_for_decision": False,
                "summary": "Follow-up liquidity evidence was unavailable.",
            }
        return {"status": "ok", "ticker": ctx.ticker, "summary": "KPI evidence is usable."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=("analyze_liquidity_stress",),
            evidence_confidence="LOW",
            summaries={"analyze_liquidity_stress": "Follow-up liquidity evidence was unavailable."},
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["actionable"] is True
    assert artifact.no_selection_finalist_resolution_attempted is False
    assert artifact.no_selection_finalist_resolution_status is None
    assert artifact.no_selection_reason is None
    assert artifact.no_selection_finalist_resolution_notes == []


def test_sector_runtime_no_selection_finalist_audit_does_not_fetch_blocker_evidence_for_cap_only_case(
    monkeypatch,
):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = []
    provider = FakeProvider(
        [
            _planning_turn(include_second_company_tool=False),
            no_selection_turn,
        ]
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "analyze_liquidity_stress":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "solvency_risk": "CRITICAL",
                "summary": "Follow-up liquidity stress exposed critical solvency risk.",
            }
        return {"status": "ok", "ticker": ctx.ticker, "summary": "KPI evidence is usable."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=("analyze_liquidity_stress",),
            summaries={
                "analyze_liquidity_stress": "Follow-up liquidity stress exposed critical solvency risk."
            },
            excerpts={"analyze_liquidity_stress": '{"solvency_risk": "critical"}'},
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert "FOLLOW_UP_SOLVENCY_CRITICAL" not in artifact.selection_audit["hard_blockers"]
    assert artifact.no_selection_finalist_resolution_attempted is False
    assert artifact.no_selection_finalist_resolution_status is None
    assert artifact.final_decision is not None
    assert artifact.final_decision.selection_blockers == []
    assert artifact.no_selection_finalist_resolution_notes == []


def test_sector_runtime_no_selection_finalist_audit_selects_without_resolution_budget(monkeypatch):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = []
    provider = FakeProvider([_planning_turn(include_second_company_tool=False), no_selection_turn])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Initial evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=2),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.no_selection_finalist_resolution_attempted is False
    assert artifact.no_selection_finalist_resolution_status is None
    assert "WATCHLIST_RESOLUTION_BUDGET_EXHAUSTED" not in artifact.degraded_states
    assert len(provider.calls) == 2
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
    ]


def test_sector_runtime_alternate_finalist_audit_can_upgrade_after_blocked_finalist(monkeypatch):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = [
        {"ticker": "AAA", "reason": "Highest return but filing blocked."},
        {"ticker": "BBB", "reason": "Second-best return."},
    ]
    provider = FakeProvider([_alternate_audit_planning_turn(), no_selection_turn])
    packets = [
        _custom_sector_packet("AAA", data_quality_status="NO_FILING", valuation_anchor=130.0),
        _custom_sector_packet("BBB", valuation_anchor=110.0),
    ]
    scenarios = [
        _audit_base_scenario("AAA", 0.30),
        _audit_base_scenario("BBB", 0.20),
        _audit_downside_scenario("BBB", -0.01),
    ]

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_sector_company_financial_packets_from_signal_packets",
        lambda signal_packets, market_cap_focus=None, **kwargs: packets,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_expected_return_scenarios_for_packets",
        lambda company_packets, horizon_years=None: _scenario_map(scenarios),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "BBB"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.no_selection_finalist_audit_focus_ticker == "AAA"
    assert artifact.alternate_finalist_audit_attempted is True
    assert artifact.alternate_finalist_audit_status == "RESOLVED_SELECTED"
    assert artifact.alternate_finalist_audit_results == [
        {
            "ticker": "BBB",
            "source": "provider_rejected_finalist",
            "audit_status": "PASS",
            "hard_blockers": [],
            "confidence_caps": [],
            "best_base_annualized_return": 0.20,
        }
    ]
    assert (
        "Alternate finalist BBB passed audit after AAA was blocked."
        in artifact.no_selection_finalist_audit_notes
    )


def test_sector_runtime_alternate_finalist_audit_preserves_no_selection_when_below_hurdle(
    monkeypatch,
):
    # Pin legacy hard-block hurdle behavior (production default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = [
        {"ticker": "AAA", "reason": "Highest return but filing blocked."},
        {"ticker": "BBB", "reason": "Second-best return."},
    ]
    provider = FakeProvider([_alternate_audit_planning_turn(), no_selection_turn])
    packets = [
        _custom_sector_packet("AAA", data_quality_status="NO_FILING", valuation_anchor=130.0),
        _custom_sector_packet("BBB", valuation_anchor=110.0),
    ]
    scenarios = [
        _audit_base_scenario("AAA", 0.30),
        _audit_base_scenario("BBB", 0.08),
        _audit_downside_scenario("BBB", -0.01),
    ]

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_sector_company_financial_packets_from_signal_packets",
        lambda signal_packets, market_cap_focus=None, **kwargs: packets,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_expected_return_scenarios_for_packets",
        lambda company_packets, horizon_years=None: _scenario_map(scenarios),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    # The alternate finalist (BBB) is below the 12% hurdle and is BLOCKED, so no
    # alternate clears the audit. The primary finalist AAA's only obstacle is a
    # fetchable NO_FILING gap (now EVIDENCE_QUALITY ), so rather than being
    # discarded as a hard NO_SELECTION it is re-surfaced as a DATA_INCOMPLETE
    # resolve-then-promote candidate with the ticker preserved.
    assert artifact.final_verdict == "DATA_INCOMPLETE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.alternate_finalist_audit_attempted is True
    assert artifact.alternate_finalist_audit_status == "NO_ALTERNATE_PASSED"
    assert artifact.alternate_finalist_audit_results[0]["ticker"] == "BBB"
    assert artifact.alternate_finalist_audit_results[0]["audit_status"] == "BLOCKED"
    assert artifact.alternate_finalist_audit_results[0]["hard_blockers"] == [
        "BASE_RETURN_BELOW_12PCT_HURDLE"
    ]
    assert (
        "No alternate finalist cleared the deterministic selection audit."
        in artifact.alternate_finalist_audit_notes
    )


def test_sector_runtime_alternate_finalist_audit_preserves_no_selection_on_hard_blocker(
    monkeypatch,
):
    # Pin legacy hard-block hurdle behavior (production default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = [
        {"ticker": "AAA", "reason": "Highest return but return hurdle blocked."},
        {"ticker": "BBB", "reason": "Second-best return but return hurdle blocked."},
    ]
    provider = FakeProvider([_alternate_audit_planning_turn(), no_selection_turn])
    packets = [
        _custom_sector_packet("AAA", valuation_anchor=130.0),
        _custom_sector_packet("BBB", valuation_anchor=110.0),
    ]
    scenarios = [
        _audit_base_scenario("AAA", 0.08),
        _audit_base_scenario("BBB", 0.07),
        _audit_downside_scenario("BBB", -0.01),
    ]

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_sector_company_financial_packets_from_signal_packets",
        lambda signal_packets, market_cap_focus=None, **kwargs: packets,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_expected_return_scenarios_for_packets",
        lambda company_packets, horizon_years=None: _scenario_map(scenarios),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.alternate_finalist_audit_status == "NO_ALTERNATE_PASSED"
    assert artifact.alternate_finalist_audit_results[0]["ticker"] == "BBB"
    assert artifact.alternate_finalist_audit_results[0]["hard_blockers"] == [
        "BASE_RETURN_BELOW_12PCT_HURDLE"
    ]


def test_sector_runtime_alternate_finalist_audit_skips_primary_and_caps_at_three(monkeypatch):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["rejected_finalists"] = [
        {"ticker": "AAA", "reason": "Primary blocked finalist."},
        {"ticker": "BBB", "reason": "Alternate one."},
        {"ticker": "CCC", "reason": "Alternate two."},
        {"ticker": "DDD", "reason": "Alternate three."},
        {"ticker": "EEE", "reason": "Alternate four."},
    ]
    provider = FakeProvider(
        [_alternate_audit_planning_turn(extra_alternate_tools=True), no_selection_turn]
    )
    packets = [
        _custom_sector_packet("AAA", data_quality_status="NO_FILING", valuation_anchor=130.0),
        _custom_sector_packet("BBB", valuation_anchor=110.0),
        _custom_sector_packet("CCC", valuation_anchor=110.0),
        _custom_sector_packet("DDD", valuation_anchor=110.0),
        _custom_sector_packet("EEE", valuation_anchor=110.0),
    ]
    scenarios = [
        _audit_base_scenario("AAA", 0.30),
        _audit_base_scenario("BBB", 0.08),
        _audit_base_scenario("CCC", 0.07),
        _audit_base_scenario("DDD", 0.06),
        _audit_base_scenario("EEE", 0.05),
    ]

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: {
            ticker: _signal_packet(ticker, 100.0, 50.0, 0.05)
            for ticker in ["AAA", "BBB", "CCC", "DDD", "EEE"]
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_sector_company_financial_packets_from_signal_packets",
        lambda signal_packets, market_cap_focus=None, **kwargs: packets,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_expected_return_scenarios_for_packets",
        lambda company_packets, horizon_years=None: _scenario_map(scenarios),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB", "CCC", "DDD", "EEE"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=6),
    )

    audited = [item["ticker"] for item in artifact.alternate_finalist_audit_results]
    assert audited == ["BBB", "CCC", "DDD"]
    assert "AAA" not in audited
    assert "EEE" not in audited


def test_sector_runtime_recovers_first_turn_invalid_json_with_minimum_plan(monkeypatch):
    provider = FirstTurnRecoveringProvider(RuntimeError("OpenAI output is not valid JSON"))
    dispatched: list[tuple[str, str]] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append((name, ctx.ticker))
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Recovered tool evidence is usable.",
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.degraded_states == ["LLM_PROVIDER_INVALID_JSON", "LLM_PROVIDER_TURN_RECOVERED"]
    assert artifact.framework is not None
    assert artifact.framework.economic_model == (
        "Industrial compounding model driven by backlog conversion, pricing versus input costs, operating leverage, "
        "maintenance capex, working-capital turns, and cycle-normalized margins."
    )
    assert "book_to_bill_or_backlog_growth" in artifact.framework.selected_metrics
    assert "unadjusted_peak_earnings_multiple" in artifact.framework.invalid_valuation_methods
    assert [question.question_id for question in artifact.research_questions] == ["R1"]
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]
    assert dispatched == [("fetch_kpi_trends", "AAA"), ("analyze_liquidity_stress", "AAA")]
    assert provider.calls[0]["schema_name"] == "autonomous_sector_initial_research_plan"
    assert provider.calls[0]["max_output_tokens"] == 6000
    assert provider.calls[1]["schema_name"] == "autonomous_sector_minimum_tool_plan"
    assert provider.calls[1]["max_output_tokens"] == 4000
    assert provider.calls[1]["schema"] == _MINIMUM_TOOL_PLAN_SCHEMA
    assert "Minimum planning state:" in provider.calls[1]["prompt"]
    minimum_state = _prompt_state(provider.calls[1]["prompt"], "Minimum planning state: ")
    assert minimum_state["framework_template_hint"]["selected_metrics"] == [
        "revenue_cagr_5y",
        "book_to_bill_or_backlog_growth",
        "gross_margin",
        "normalized_operating_margin",
        "maintenance_capex_to_sales",
        "cash_conversion_cycle",
        "share_count_cagr",
    ]
    assert minimum_state["framework_required_evidence_tool_hints"][0] == {
        "required_evidence": "backlog_or_order_trend_evidence",
        "suggested_tools": [
            "fetch_kpi_trends",
            "fetch_companyfacts_timeseries",
            "fetch_filing_section",
            "fetch_recent_filing_context",
        ],
    }
    assert minimum_state["framework_evidence_preflight"][0]["needs_tool_evidence"] == [
        "backlog_or_order_trend_evidence",
        "pricing_and_input_cost_evidence",
        "maintenance_capex_evidence",
        "cycle_normalization_evidence",
    ]
    assert any(
        "Recovered from first-turn provider failure" in note for note in artifact.audit_notes
    )


def test_sector_runtime_preserves_first_class_deepseek_truncation_state(monkeypatch):
    provider = FirstTurnRecoveringProvider(
        DeepSeekOutputTruncatedError(
            "LLM_PROVIDER_TRUNCATED: DeepSeek response finish_reason=length "
            "at requested max_output_tokens=6000"
        )
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Recovered evidence.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.degraded_states == [
        "LLM_PROVIDER_TRUNCATED",
        "LLM_PROVIDER_TURN_RECOVERED",
    ]
    assert provider.calls[0]["max_output_tokens"] == 6000
    assert provider.calls[1]["max_output_tokens"] == 4000


def test_sector_runtime_recovers_first_turn_timeout_with_minimum_plan(monkeypatch):
    provider = FirstTurnRecoveringProvider(TimeoutError("OpenAI request timed out"))

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Recovered evidence.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.degraded_states == ["LLM_PROVIDER_TIMEOUT", "LLM_PROVIDER_TURN_RECOVERED"]
    assert provider.calls[1]["schema_name"] == "autonomous_sector_minimum_tool_plan"


def test_sector_runtime_stops_precisely_when_first_turn_and_recovery_fail(monkeypatch):
    provider = FirstTurnRecoveringProvider(
        RuntimeError("OpenAI output is not valid JSON"),
        second_exc=TimeoutError("read timed out"),
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.tool_calls == []
    assert artifact.degraded_states == ["LLM_PROVIDER_INVALID_JSON", "LLM_PROVIDER_TIMEOUT"]
    assert "LLM provider failed during first sector planning turn" in artifact.no_selection_reason
    assert any("Stopped before tool execution" in note for note in artifact.audit_notes)
    summary = sector_artifact_summary(artifact)
    assert summary["degraded_states"] == ["LLM_PROVIDER_INVALID_JSON", "LLM_PROVIDER_TIMEOUT"]
    assert summary["no_selection_reason"] == artifact.no_selection_reason
    assert len(provider.calls) == 2


def test_sector_runtime_routes_sentence_like_provider_degraded_states_to_audit_notes(monkeypatch):
    planning_turn = _planning_turn()
    planning_turn["degraded_states"] = [
        "NO_FILING_FOR_SOME_CANDIDATES",
        "Do not select a winner in this turn.",
        "INTC: GATE_BLOCK + MISSING_VALUATION — excluded from scenario analysis",
    ]
    provider = FakeProvider([planning_turn, _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert "NO_FILING_FOR_SOME_CANDIDATES" in artifact.degraded_states
    assert "Do not select a winner in this turn." not in artifact.degraded_states
    assert (
        "INTC: GATE_BLOCK + MISSING_VALUATION — excluded from scenario analysis"
        not in artifact.degraded_states
    )
    assert any(
        note
        == "Provider degraded-state note moved to audit trail: Do not select a winner in this turn."
        for note in artifact.audit_notes
    )
    assert any(
        note
        == "Provider degraded-state note moved to audit trail: INTC: GATE_BLOCK + MISSING_VALUATION — excluded from scenario analysis"
        for note in artifact.audit_notes
    )


def test_selection_audit_credits_sector_expected_return_evidence_by_tool_scope():
    tool_calls = [
        ToolCallRecord(
            call_id="TC1",
            tool_name="rank_expected_return_cases",
            tool_input={"tickers": ["BBB", "AAA"], "horizon_years": 5, "scenario_name": "base"},
            rationale="Rank scoped expected-return cases.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            call_id="TC2",
            tool_name="fetch_kpi_trends",
            tool_input={},
            rationale="Validate selected ticker quality.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
        ToolCallRecord(
            call_id="TC3",
            tool_name="analyze_liquidity_stress",
            tool_input={},
            rationale="Validate selected ticker liquidity risk.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            evidence_id="E1",
            source_type="tool_output",
            source_label="rank_expected_return_cases",
            summary="Ranking output is attached to the first scoped ticker only.",
            ticker="BBB",
            excerpt='{"ranked_tickers": ["BBB"]}',
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            evidence_id="E2",
            source_type="tool_output",
            source_label="fetch_kpi_trends",
            summary="AAA company-specific KPI evidence is usable.",
            ticker="AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            evidence_id="E3",
            source_type="tool_output",
            source_label="analyze_liquidity_stress",
            summary="AAA company-specific liquidity evidence is usable.",
            ticker="AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["expected_return_evidence_count"] == 1
    assert audit["company_specific_evidence_count"] == 2
    assert audit["downside_evidence_status"] == "PRESENT"
    assert audit["selected_evidence_pillars"] == ["liquidity", "quality"]
    assert "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" not in audit["hard_blockers"]


def test_audit_signal_classification_uses_conservative_default():
    assert classify_audit_signal("CURRENT_EVENTS_UNAVAILABLE") == "EVIDENCE_QUALITY"
    assert classify_audit_signal("ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK") == "BUSINESS_QUALITY"
    assert classify_audit_signal("BASE_RETURN_BELOW_12PCT_HURDLE") == "HURDLE"
    assert classify_audit_signal("NEGATIVE_MARGIN_OF_SAFETY") == "HURDLE"
    assert classify_audit_signal("SUSPICIOUS_MAGNITUDE_DCF") == "BUSINESS_QUALITY"
    assert classify_audit_signal("SURPRISE_NEW_SIGNAL") == "BUSINESS_QUALITY"


@pytest.mark.parametrize(
    "code",
    [
        "MISSING_PRICE",
        "MISSING_VALUATION",
        "MISSING_BASE_RETURN_CASE",
        "MODEL_FIT_BLOCKED",
    ],
)
def test_v1_missing_data_blockers_keep_legacy_business_quality_classification(code):
    assert classify_audit_signal(code) == "BUSINESS_QUALITY"


@pytest.mark.parametrize(
    "code",
    [
        "MISSING_PRICE",
        "MISSING_VALUATION",
        "MISSING_BASE_RETURN_CASE",
        "MODEL_FIT_BLOCKED",
    ],
)
def test_v2_does_not_project_legacy_missing_data_blockers_as_screen_results(code):
    assert _v2_screen_state({"audit_status": "BLOCKED", "hard_blockers": [code]}) == (
        "INCOMPLETE",
        "NEEDS_DATA",
        ["SCREEN_NOT_EVALUATED"],
    )


def test_v2_does_not_project_legacy_business_quality_blocker_as_screened_out():
    assert _v2_screen_state({"audit_status": "BLOCKED", "hard_blockers": ["CLEAR_IMPAIRMENT"]}) == (
        "INCOMPLETE",
        "NEEDS_DATA",
        ["SCREEN_NOT_EVALUATED"],
    )


def test_selection_audit_includes_signal_classification_annotations(monkeypatch):
    # This test pins the legacy hard-block hurdle behavior, which is the
    # 'hard' rollback mode (the production default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _custom_sector_packet("AAA", data_quality_status="NO_FILING")},
        scenarios=[_audit_base_scenario("AAA", annualized_return=0.11)],
        tool_calls=[],
        evidence=[],
        degraded_states=[],
    )

    # Base return 0.11 is below the 12% hurdle, so BASE_RETURN_BELOW_12PCT_HURDLE
    # (classified HURDLE) remains a binding hard blocker and the status stays BLOCKED.
    # The two MISSING_*_EVIDENCE codes now classify EVIDENCE_QUALITY
    # (fetchable data gaps) and surface in needs_evidence_resolution rather than
    # binding as BUSINESS_QUALITY hard blocks.
    assert audit["status"] == "BLOCKED"
    assert audit["hard_blocker_classifications"] == [
        {"code": "NO_FILING", "classification": "EVIDENCE_QUALITY"},
        {"code": "BASE_RETURN_BELOW_12PCT_HURDLE", "classification": "HURDLE"},
        {"code": "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE", "classification": "EVIDENCE_QUALITY"},
        {"code": "MISSING_COMPANY_SPECIFIC_EVIDENCE", "classification": "EVIDENCE_QUALITY"},
    ]
    assert audit["blockers_by_class"] == [
        {"classification": "EVIDENCE_QUALITY", "count": 3},
        {"classification": "HURDLE", "count": 1},
    ]
    assert audit["confidence_cap_classifications"] == [
        {"code": "MISSING_DOWNSIDE_RETURN_CASE", "classification": "BUSINESS_QUALITY"},
        {"code": "CAPITAL_LOSS_EVIDENCE_DEGRADED", "classification": "EVIDENCE_QUALITY"},
    ]
    assert audit["caps_by_class"] == [
        {"classification": "EVIDENCE_QUALITY", "count": 1},
        {"classification": "BUSINESS_QUALITY", "count": 1},
    ]
    assert audit["needs_evidence_resolution"] == [
        "NO_FILING",
        "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE",
        "MISSING_COMPANY_SPECIFIC_EVIDENCE",
        "CAPITAL_LOSS_EVIDENCE_DEGRADED",
    ]


def test_selection_audit_allows_evidence_quality_caps_with_resolution_flag():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.confidence_caps = ["QUARTERLY_REVENUE_TREND_UNKNOWN"]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "HIGH"
    assert audit["confidence_caps"] == ["QUARTERLY_REVENUE_TREND_UNKNOWN"]
    assert audit["needs_evidence_resolution"] == ["QUARTERLY_REVENUE_TREND_UNKNOWN"]


def test_selection_audit_allows_evidence_quality_hard_blockers_with_resolution_flag():
    tool_calls, evidence = _passing_audit_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _custom_sector_packet("AAA", data_quality_status="NO_FILING")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    # NO_FILING is an EVIDENCE_QUALITY hard blocker (the annual filing could not
    # be fetched at all), so the candidate is DATA_INCOMPLETE — a
    # resolve-then-promote state, not a clean actionable PASS and not a quality
    # reject. The data-availability gaps surface in data_resolution_needed.
    assert audit["status"] == "DATA_INCOMPLETE"
    assert audit["actionable"] is False
    assert audit["data_incomplete"] is True
    assert "NO_FILING" in audit["hard_blockers"]
    assert audit["needs_evidence_resolution"] == ["NO_FILING", "CAPITAL_LOSS_EVIDENCE_DEGRADED"]
    assert audit["data_resolution_needed"] == ["NO_FILING", "CAPITAL_LOSS_EVIDENCE_DEGRADED"]


def test_selection_audit_business_quality_caps_confidence_without_blocking_actionable():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {"solvency_risk": "ELEVATED"}

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "MODERATE"
    assert audit["confidence_caps"] == ["ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK"]
    assert audit["needs_evidence_resolution"] == []


def test_selection_audit_two_business_quality_caps_lower_confidence_to_low():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {"solvency_risk": "ELEVATED"}
    packet.confidence_caps = ["HIGH_GROWTH_DEPENDENCY"]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "LOW"
    assert audit["confidence_caps"] == [
        "UNRESOLVED_HIGH_GROWTH_DEPENDENCY",
        "ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK",
    ]
    assert audit["needs_evidence_resolution"] == []


def test_selection_audit_business_quality_hard_blocker_still_blocks_actionable():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.blockers = ["FOLLOW_UP_NO_ASSURANCE_FINANCING"]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "BLOCKED"
    assert audit["actionable"] is False
    assert audit["confidence_ceiling"] is None
    assert audit["hard_blockers"] == ["FOLLOW_UP_NO_ASSURANCE_FINANCING"]


def test_selection_audit_hurdle_still_blocks_actionable(monkeypatch):
    # Validates the legacy hard-block path, the 'hard' rollback mode.
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()

    tool_calls, evidence = _passing_audit_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[_audit_base_scenario("AAA", annualized_return=0.11)],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "BLOCKED"
    assert audit["actionable"] is False
    assert audit["hard_blockers"] == ["BASE_RETURN_BELOW_12PCT_HURDLE"]
    assert audit["needs_evidence_resolution"] == []


def test_selection_audit_thin_return_hurdle_cap_still_blocks_actionable():
    tool_calls, evidence = _passing_audit_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA", annualized_return=0.13),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "WATCHLIST_ONLY"
    assert audit["actionable"] is False
    assert audit["confidence_ceiling"] == "MODERATE"
    assert audit["confidence_caps"] == ["THIN_RETURN_CUSHION"]
    assert audit["needs_evidence_resolution"] == []


def test_selection_audit_negative_margin_of_safety_blocks_actionable():
    tool_calls, evidence = _passing_audit_tool_evidence()
    evidence.append(
        EvidenceReference(
            evidence_id="E4",
            source_type="analysis_report",
            source_label="analyst_context",
            summary="AAA screens as WATCH. Adjusted margin of safety is -39.7%.",
            ticker="AAA",
            excerpt=json.dumps({"valuation": {"margin_of_safety": -0.397486008765985}}),
            confidence="HIGH",
        )
    )

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "WATCHLIST_ONLY"
    assert audit["actionable"] is False
    assert audit["confidence_ceiling"] == "MODERATE"
    assert audit["confidence_caps"] == ["NEGATIVE_MARGIN_OF_SAFETY"]
    assert audit["margin_of_safety_status"] == "NEGATIVE"
    assert audit["margin_of_safety"] == -0.397
    assert audit["margin_of_safety_source"] == "evidence.E4.summary"


def test_selection_audit_positive_margin_of_safety_allows_actionable():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.valuation = {
        "valuation_anchor": 100.0,
        "anchor_method": "dcf",
        "margin_of_safety": 0.215,
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "HIGH"
    assert audit["confidence_caps"] == []
    assert audit["margin_of_safety_status"] == "NON_NEGATIVE_OR_UNAVAILABLE"
    assert audit["margin_of_safety"] is None


def test_selection_audit_implausible_dcf_magnitude_caps_confidence_to_low():
    tool_calls, evidence = _passing_audit_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={
            "AAA": _custom_sector_packet("AAA", valuation_anchor=314.0, current_price=100.0)
        },
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "LOW"
    assert audit["confidence_caps"] == ["SUSPICIOUS_MAGNITUDE_DCF"]
    assert audit["valuation_anchor_magnitude_status"] == "SUSPICIOUS"
    assert audit["valuation_anchor_to_price_ratio"] == 3.14


def test_selection_audit_reasonable_dcf_magnitude_does_not_add_cap():
    tool_calls, evidence = _passing_audit_tool_evidence()

    qlys_audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={
            "AAA": _custom_sector_packet("AAA", valuation_anchor=142.0, current_price=100.0)
        },
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )
    two_x_audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={
            "AAA": _custom_sector_packet("AAA", valuation_anchor=200.0, current_price=100.0)
        },
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert qlys_audit["status"] == "PASS"
    assert qlys_audit["confidence_ceiling"] == "HIGH"
    assert qlys_audit["confidence_caps"] == []
    assert qlys_audit["valuation_anchor_magnitude_status"] == "OK"
    assert qlys_audit["valuation_anchor_to_price_ratio"] == 1.42
    assert two_x_audit["status"] == "PASS"
    assert two_x_audit["confidence_ceiling"] == "HIGH"
    assert two_x_audit["confidence_caps"] == []
    assert two_x_audit["valuation_anchor_magnitude_status"] == "OK"
    assert two_x_audit["valuation_anchor_to_price_ratio"] == 2.0


def test_selection_audit_rejects_sector_expected_return_evidence_outside_tool_scope():
    tool_calls = [
        ToolCallRecord(
            call_id="TC1",
            tool_name="rank_expected_return_cases",
            tool_input={"tickers": ["BBB"], "horizon_years": 5, "scenario_name": "base"},
            rationale="Rank expected-return cases without the selected ticker in scope.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            call_id="TC2",
            tool_name="fetch_kpi_trends",
            tool_input={},
            rationale="Validate selected ticker quality.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
    ]
    evidence = [
        EvidenceReference(
            evidence_id="E1",
            source_type="tool_output",
            source_label="rank_expected_return_cases",
            summary="Ranking output covered only BBB.",
            ticker="BBB",
            excerpt='{"ranked_tickers": ["BBB"]}',
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            evidence_id="E2",
            source_type="tool_output",
            source_label="fetch_kpi_trends",
            summary="AAA company-specific KPI evidence is usable.",
            ticker="AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[_audit_base_scenario("AAA")],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    # The expected-return evidence was not usable for the selected ticker, so
    # MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE is recorded as a hard blocker.
    # That code now classifies EVIDENCE_QUALITY (a fetchable data gap),
    # so it no longer hard-BLOCKS; the candidate is DATA_INCOMPLETE
    # (resolve-then-promote) rather than a permanent quality reject.
    assert audit["status"] == "DATA_INCOMPLETE"
    assert audit["expected_return_evidence_count"] == 0
    assert "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" in audit["hard_blockers"]
    assert "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" in audit["data_resolution_needed"]


def test_selection_audit_rejects_low_confidence_sector_expected_return_evidence():
    tool_calls = [
        ToolCallRecord(
            call_id="TC1",
            tool_name="rank_expected_return_cases",
            tool_input={"tickers": ["BBB", "AAA"], "horizon_years": 5, "scenario_name": "base"},
            rationale="Rank scoped expected-return cases.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            call_id="TC2",
            tool_name="fetch_kpi_trends",
            tool_input={},
            rationale="Validate selected ticker quality.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
    ]
    evidence = [
        EvidenceReference(
            evidence_id="E1",
            source_type="tool_output",
            source_label="rank_expected_return_cases",
            summary="Ranking output was non-usable and low confidence.",
            ticker="BBB",
            tool_call_id="TC1",
            confidence="LOW",
        ),
        EvidenceReference(
            evidence_id="E2",
            source_type="tool_output",
            source_label="fetch_kpi_trends",
            summary="AAA company-specific KPI evidence is usable.",
            ticker="AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[_audit_base_scenario("AAA")],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    # The expected-return evidence was not usable for the selected ticker, so
    # MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE is recorded as a hard blocker.
    # That code now classifies EVIDENCE_QUALITY (a fetchable data gap),
    # so it no longer hard-BLOCKS; the candidate is DATA_INCOMPLETE
    # (resolve-then-promote) rather than a permanent quality reject.
    assert audit["status"] == "DATA_INCOMPLETE"
    assert audit["expected_return_evidence_count"] == 0
    assert "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" in audit["hard_blockers"]
    assert "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" in audit["data_resolution_needed"]


def test_selection_audit_caps_thin_base_return_cushion():
    tool_calls = [
        ToolCallRecord(
            call_id="TC1",
            tool_name="rank_expected_return_cases",
            tool_input={"tickers": ["AAA"], "horizon_years": 5, "scenario_name": "base"},
            rationale="Rank expected-return cases.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            call_id="TC2",
            tool_name="fetch_kpi_trends",
            tool_input={},
            rationale="Validate KPI quality.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
        ToolCallRecord(
            call_id="TC3",
            tool_name="analyze_liquidity_stress",
            tool_input={},
            rationale="Validate risk evidence.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA", annualized_return=0.13),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "WATCHLIST_ONLY"
    assert audit["return_cushion_status"] == "THIN"
    assert audit["base_return_margin_over_hurdle"] == 0.010000000000000009
    assert "THIN_RETURN_CUSHION" in audit["confidence_caps"]


def test_selection_audit_caps_low_confidence_for_one_company_specific_evidence_tool():
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    # INSUFFICIENT_COMPANY_SPECIFIC_EVIDENCE and
    # INSUFFICIENT_EVIDENCE_PILLAR_COVERAGE now classify EVIDENCE_QUALITY
    # (fetchable thin-coverage gaps), so they no longer lower the business-
    # quality confidence ceiling. With no business-quality cap present the
    # candidate is a clean PASS at the HIGH ceiling, and the thin-coverage caps
    # surface as evidence-resolution needs instead.
    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "HIGH"
    assert audit["company_specific_evidence_count"] == 1
    assert audit["selected_company_evidence_tools"] == ["fetch_kpi_trends"]
    assert "INSUFFICIENT_COMPANY_SPECIFIC_EVIDENCE" in audit["confidence_caps"]
    assert "INSUFFICIENT_EVIDENCE_PILLAR_COVERAGE" in audit["confidence_caps"]
    assert audit["needs_evidence_resolution"] == [
        "INSUFFICIENT_COMPANY_SPECIFIC_EVIDENCE",
        "INSUFFICIENT_EVIDENCE_PILLAR_COVERAGE",
    ]


def test_selection_audit_requires_quality_plus_second_evidence_pillar():
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
        ToolCallRecord(
            "TC3",
            "analyze_dilution",
            {"years": 5},
            "Validate dilution.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_dilution",
            "AAA dilution evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["selected_evidence_pillars"] == ["dilution", "liquidity"]
    assert "MISSING_KPI_QUALITY_EVIDENCE" in audit["confidence_caps"]
    assert audit["needs_evidence_resolution"] == ["MISSING_KPI_QUALITY_EVIDENCE"]


def test_selection_audit_caps_confidence_for_negative_downside_without_risk_evidence():
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "compare_peer_metric",
            {"metric": "roic"},
            "Validate quality peer evidence.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "compare_peer_metric",
            "AAA peer evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.08),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    # The only business-quality cap is DOWNSIDE_ASYMMETRY_UNRESOLVED (the
    # INSUFFICIENT_* coverage caps are now EVIDENCE_QUALITY and no longer
    # count toward the ceiling), so the single business-quality cap caps the
    # ceiling at MODERATE rather than the prior LOW.
    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "MODERATE"
    assert audit["downside_annualized_return"] == -0.08
    assert audit["downside_evidence_status"] == "UNRESOLVED_ASYMMETRY"
    assert "DOWNSIDE_ASYMMETRY_UNRESOLVED" in audit["confidence_caps"]


def test_selection_audit_caps_confidence_for_missing_downside_return_case():
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[_audit_base_scenario("AAA")],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "MODERATE"
    assert audit["hard_blockers"] == []
    assert audit["downside_annualized_return"] is None
    assert audit["downside_evidence_status"] == "MISSING"
    assert audit["confidence_caps"] == ["MISSING_DOWNSIDE_RETURN_CASE"]


def test_selection_audit_blocks_permanent_capital_loss_classification():
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={
            "AAA": _audit_packet_with_impairment(
                "CLEAR_IMPAIRMENT",
                reason_codes=["ECONOMIC_WEAKNESS_CONFIRMED", "NO_MOS_CONFIRMED"],
            )
        },
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "BLOCKED"
    assert audit["capital_loss_underwriting_status"] == "PERMANENT_LOSS_BLOCKED"
    assert audit["capital_loss_impairment_class"] == "CLEAR_IMPAIRMENT"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT"
    assert audit["capital_loss_reason_codes"] == ["ECONOMIC_WEAKNESS_CONFIRMED", "NO_MOS_CONFIRMED"]
    assert "PERMANENT_CAPITAL_LOSS_CONFIRMED" in audit["hard_blockers"]


def test_selection_audit_keeps_temporary_weakness_actionable_when_other_evidence_passes():
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={
            "AAA": _audit_packet_with_impairment(
                "TEMPORARY_WEAKNESS",
                caution="POSSIBLE_CYCLE_DISTORTION",
                reason_codes=["CYCLICAL_TROUGH", "CYCLICAL_SUPPORT"],
            )
        },
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["capital_loss_underwriting_status"] == "TEMPORARY_WEAKNESS_SUPPORTED"
    assert audit["capital_loss_impairment_class"] == "TEMPORARY_WEAKNESS"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_CYCLE_DISTORTION"
    assert audit["capital_loss_reason_codes"] == ["CYCLICAL_TROUGH", "CYCLICAL_SUPPORT"]
    assert "PERMANENT_CAPITAL_LOSS_CONFIRMED" not in audit["hard_blockers"]


def test_selection_audit_fallback_blocks_probable_impairment_from_critical_solvency_without_upstream_classification():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {"solvency_risk": "CRITICAL"}

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "BLOCKED"
    assert audit["capital_loss_underwriting_status"] == "PROBABLE_PERMANENT_LOSS_BLOCKED"
    assert audit["capital_loss_impairment_class"] == "PROBABLE_IMPAIRMENT"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_SOLVENCY_CRITICAL"]
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" in audit["hard_blockers"]


def test_selection_audit_fallback_blocks_probable_impairment_from_negative_equity_without_upstream_classification():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "negative_equity": True,
        "debt_due_within_12mo": False,
        "going_concern_language": False,
        "no_assurance_financing": False,
        "cash_runway_quarters": 12.0,
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "BLOCKED"
    assert audit["capital_loss_underwriting_status"] == "PROBABLE_PERMANENT_LOSS_BLOCKED"
    assert audit["capital_loss_impairment_class"] == "PROBABLE_IMPAIRMENT"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_NEGATIVE_EQUITY"]
    assert audit["refinancing_timeline_status"] == "NO_NEAR_TERM_REFINANCING_FLAG"
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" in audit["hard_blockers"]


def test_selection_audit_fallback_blocks_probable_impairment_from_current_ratio_liquidity_crisis():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "negative_equity": False,
        "current_ratio": 0.42,
        "debt_due_within_12mo": False,
        "going_concern_language": False,
        "no_assurance_financing": False,
        "cash_runway_quarters": 12.0,
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "BLOCKED"
    assert audit["capital_loss_underwriting_status"] == "PROBABLE_PERMANENT_LOSS_BLOCKED"
    assert audit["capital_loss_impairment_class"] == "PROBABLE_IMPAIRMENT"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_CURRENT_RATIO_LIQUIDITY_CRISIS"]
    assert audit["refinancing_timeline_status"] == "NO_NEAR_TERM_REFINANCING_FLAG"
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" in audit["hard_blockers"]


_FILED_GOING_CONCERN_ASSERTION = {
    "subject": "REGISTRANT",
    "assertion_mode": "AFFIRMATIVE_CURRENT",
    "blockable": True,
    "accession": "0000000001-26-000001",
    "form_type": "10-K",
    "excerpt": "There is substantial doubt about our ability to continue as a going concern.",
}


def test_selection_audit_fallback_blocks_probable_impairment_from_going_concern_without_upstream_classification():
    """Migrated: the flag alone used to block. A going-concern block now needs the
    stored, filed, blockable assertion behind the flag (see the bare-flag test below)."""
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "debt_due_within_12mo": False,
        "going_concern_language": True,
        "going_concern_assertions": [dict(_FILED_GOING_CONCERN_ASSERTION)],
        "no_assurance_financing": False,
        "cash_runway_quarters": 6.0,
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "BLOCKED"
    assert audit["capital_loss_underwriting_status"] == "PROBABLE_PERMANENT_LOSS_BLOCKED"
    assert audit["capital_loss_impairment_class"] == "PROBABLE_IMPAIRMENT"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_PERMANENT_CAPITAL_IMPAIRMENT"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_GOING_CONCERN_LANGUAGE"]
    assert audit["refinancing_timeline_status"] == "GOING_CONCERN_UNRESOLVED"
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" in audit["hard_blockers"]
    assert "PACKET_GOING_CONCERN_LANGUAGE" in audit["hard_blockers"]


@pytest.mark.parametrize(
    "assertions",
    [
        pytest.param(None, id="flag_only"),
        pytest.param([], id="no_assertions"),
        pytest.param(
            [{**_FILED_GOING_CONCERN_ASSERTION, "blockable": False, "assertion_mode": "NEGATED"}],
            id="only_non_blockable_assertions",
        ),
        pytest.param(
            [{**_FILED_GOING_CONCERN_ASSERTION, "excerpt": ""}], id="blockable_but_no_excerpt"
        ),
    ],
)
def test_bare_going_concern_flag_without_a_filed_assertion_blocks_nothing(assertions):
    """A text flag with no stored filed assertion behind it must not block (a $100B
    issuer was blocked this way). It is reported as unsupported evidence instead."""
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "debt_due_within_12mo": False,
        "going_concern_language": True,
        "no_assurance_financing": False,
        "cash_runway_quarters": 20.0,
        **({} if assertions is None else {"going_concern_assertions": assertions}),
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert "PACKET_GOING_CONCERN_LANGUAGE" not in audit["hard_blockers"]
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" not in audit["hard_blockers"]
    assert audit["capital_loss_reason_codes"] != ["FALLBACK_GOING_CONCERN_LANGUAGE"]
    assert audit["refinancing_timeline_status"] != "GOING_CONCERN_UNRESOLVED"
    assert "GOING_CONCERN_LANGUAGE_UNSUPPORTED" in audit["confidence_caps"]


def test_follow_up_evidence_flag_needs_the_asserted_marker_to_block():
    from app.autonomous.sector_runtime import _evidence_hard_blockers_for_ticker

    def _evidence(payload: str):
        return [
            EvidenceReference(
                "E9",
                "tool_output",
                "analyze_liquidity_stress",
                f"AAA {payload}",
                "AAA",
                tool_call_id="TC3",
                confidence="MODERATE",
            )
        ]

    bare = '{"going_concern_language": true}'
    backed = '{"going_concern_language": true, "going_concern_asserted": true}'
    assert _evidence_hard_blockers_for_ticker(ticker="AAA", evidence=_evidence(bare)) == []
    assert _evidence_hard_blockers_for_ticker(ticker="AAA", evidence=_evidence(backed)) == [
        "FOLLOW_UP_GOING_CONCERN_LANGUAGE"
    ]


def test_selection_audit_fallback_caps_degraded_impairment_evidence_without_upstream_classification():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.confidence_caps = ["FILING_RISK_KEYWORD_FALLBACK"]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["capital_loss_underwriting_status"] == "EVIDENCE_DEGRADED_WATCHLIST"
    assert audit["capital_loss_impairment_class"] == "EVIDENCE_DEGRADED_NOT_ASSESSABLE"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_EVIDENCE_GAP"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_IMPAIRMENT_EVIDENCE_GAP"]
    assert "CAPITAL_LOSS_EVIDENCE_DEGRADED" in audit["confidence_caps"]
    assert audit["needs_evidence_resolution"] == ["CAPITAL_LOSS_EVIDENCE_DEGRADED"]


def test_selection_audit_fallback_structural_weakness_caps_confidence_without_blocking():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.confidence_caps = [
        "FINANCIAL_ANOMALIES_PRESENT",
        "HIGH_GROWTH_DEPENDENCY",
        "METHOD_TENSION_GROWTH_VS_EARNINGS_POWER",
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "LOW"
    assert audit["capital_loss_underwriting_status"] == "STRUCTURALLY_WEAK_WATCHLIST"
    assert audit["capital_loss_impairment_class"] == "STRUCTURALLY_WEAK_NOT_IMPAIRED"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_STRUCTURAL_ECONOMIC_WEAKNESS"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_STRUCTURAL_WEAKNESS_CAPS"]
    assert "STRUCTURALLY_WEAK_ECONOMICS" in audit["confidence_caps"]
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" not in audit["hard_blockers"]
    assert audit["needs_evidence_resolution"] == []


def test_selection_audit_fallback_capital_structure_dominates_downside_caps_confidence():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.business_quality = {
        "valuation_headwinds": ["CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"],
        "downside_risk_class": "SEVERE",
    }
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "negative_equity": False,
        "current_ratio": 1.3,
        "debt_due_within_12mo": False,
        "going_concern_language": False,
        "no_assurance_financing": False,
        "cash_runway_quarters": 12.0,
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "LOW"
    assert audit["capital_loss_underwriting_status"] == "STRUCTURALLY_WEAK_WATCHLIST"
    assert audit["capital_loss_impairment_class"] == "STRUCTURALLY_WEAK_NOT_IMPAIRED"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_STRUCTURAL_ECONOMIC_WEAKNESS"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"]
    assert "STRUCTURALLY_WEAK_ECONOMICS" in audit["confidence_caps"]
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" not in audit["hard_blockers"]


def test_selection_audit_fallback_clustered_valuation_headwinds_caps_confidence():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.business_quality = {
        "valuation_headwinds": ["LEVERAGE_STRESS_HEADWIND", "ACCOUNTING_QUALITY_HEADWIND"],
    }
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "negative_equity": False,
        "current_ratio": 1.3,
        "debt_due_within_12mo": False,
        "going_concern_language": False,
        "no_assurance_financing": False,
        "cash_runway_quarters": 12.0,
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "LOW"
    assert audit["capital_loss_underwriting_status"] == "STRUCTURALLY_WEAK_WATCHLIST"
    assert audit["capital_loss_impairment_class"] == "STRUCTURALLY_WEAK_NOT_IMPAIRED"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_STRUCTURAL_ECONOMIC_WEAKNESS"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_CLUSTERED_VALUATION_HEADWINDS"]
    assert "STRUCTURALLY_WEAK_ECONOMICS" in audit["confidence_caps"]
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" not in audit["hard_blockers"]


def test_selection_audit_caps_confidence_for_unresolved_no_assurance_refinancing_timeline():
    packet = _audit_packet("AAA")
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "debt_due_within_12mo": False,
        "going_concern_language": False,
        "no_assurance_financing": True,
        "cash_runway_quarters": 10.0,
    }
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "LOW"
    assert audit["refinancing_timeline_status"] == "NO_ASSURANCE_FINANCING_UNRESOLVED"
    assert audit["refinancing_timeline_needs"] == ["NO_ASSURANCE_FINANCING_RESOLUTION"]
    assert audit["no_assurance_financing"] is True
    assert audit["debt_due_within_12mo"] is False
    assert "NO_ASSURANCE_FINANCING_UNRESOLVED" in audit["confidence_caps"]


def test_selection_audit_fallback_no_assurance_caps_confidence_with_evidence_resolution():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "debt_due_within_12mo": False,
        "going_concern_language": False,
        "no_assurance_financing": True,
        "cash_runway_quarters": 10.0,
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "LOW"
    assert audit["capital_loss_underwriting_status"] == "EVIDENCE_DEGRADED_WATCHLIST"
    assert audit["capital_loss_impairment_class"] == "EVIDENCE_DEGRADED_NOT_ASSESSABLE"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_EVIDENCE_GAP"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_NO_ASSURANCE_FINANCING"]
    assert audit["refinancing_timeline_status"] == "NO_ASSURANCE_FINANCING_UNRESOLVED"
    assert "CAPITAL_LOSS_EVIDENCE_DEGRADED" in audit["confidence_caps"]
    assert "NO_ASSURANCE_FINANCING_UNRESOLVED" in audit["confidence_caps"]
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" not in audit["hard_blockers"]


def test_selection_audit_fallback_limited_cash_runway_caps_confidence_with_evidence_resolution():
    tool_calls, evidence = _passing_audit_tool_evidence()
    packet = _audit_packet("AAA")
    packet.balance_sheet = {
        "solvency_risk": "ELEVATED",
        "debt_due_within_12mo": False,
        "going_concern_language": False,
        "no_assurance_financing": False,
        "cash_runway_quarters": 5.0,
    }

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["actionable"] is True
    assert audit["confidence_ceiling"] == "LOW"
    assert audit["capital_loss_underwriting_status"] == "EVIDENCE_DEGRADED_WATCHLIST"
    assert audit["capital_loss_impairment_class"] == "EVIDENCE_DEGRADED_NOT_ASSESSABLE"
    assert audit["capital_loss_underwriting_caution"] == "POSSIBLE_EVIDENCE_GAP"
    assert audit["capital_loss_reason_codes"] == ["FALLBACK_LIMITED_CASH_RUNWAY"]
    assert audit["refinancing_timeline_status"] == "LIMITED_CASH_RUNWAY"
    assert audit["refinancing_timeline_needs"] == ["CASH_RUNWAY_BRIDGE"]
    assert "CAPITAL_LOSS_EVIDENCE_DEGRADED" in audit["confidence_caps"]
    assert "LIMITED_CASH_RUNWAY" in audit["confidence_caps"]
    assert "PROBABLE_PERMANENT_CAPITAL_LOSS" not in audit["hard_blockers"]


def test_selection_audit_resolves_near_term_maturity_with_capital_structure_evidence():
    packet = _audit_packet("AAA")
    packet.balance_sheet = {
        "solvency_risk": "LOW",
        "debt_due_within_12mo": True,
        "going_concern_language": False,
        "no_assurance_financing": False,
        "cash_runway_quarters": 6.0,
    }
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_capital_structure_resolution",
            {},
            "Validate maturity schedule.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            evidence_id="E3",
            source_type="tool_output",
            source_label="analyze_capital_structure_resolution",
            summary="AAA CAPITAL_STRUCTURE_RESOLVED_CLEAR: refinancing was completed and covenant compliance is current.",
            ticker="AAA",
            excerpt=json.dumps(
                {
                    "capital_structure_terms": {
                        "extraction_status": "STRUCTURED_TERMS_EXTRACTED",
                        "maturity_schedule_status": "MATURITY_SCHEDULE_EXTRACTED",
                        "maturity_schedule": [
                            {
                                "year": 2027,
                                "amount": 25.0,
                                "unit": "million",
                                "context": "$25 million of term debt is due in 2027.",
                            }
                        ],
                        "covenant_status": "COMPLIANCE_EVIDENCED",
                        "covenant_terms_status": "COVENANT_TERMS_EXTRACTED",
                        "covenant_terms": [
                            {
                                "metric": "consolidated net leverage ratio",
                                "condition": "MAXIMUM",
                                "threshold": 3.5,
                                "unit": "ratio",
                                "context": "Maximum consolidated net leverage ratio of 3.50 to 1.00.",
                            }
                        ],
                    }
                }
            ),
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["refinancing_timeline_status"] == "RESOLVED_CURRENT_EVIDENCE"
    assert audit["refinancing_timeline_needs"] == []
    assert audit["capital_structure_terms_status"] == "STRUCTURED_TERMS_EXTRACTED"
    assert audit["maturity_schedule_status"] == "MATURITY_SCHEDULE_EXTRACTED"
    assert audit["maturity_schedule"] == [
        {
            "year": 2027,
            "amount": 25.0,
            "unit": "million",
            "context": "$25 million of term debt is due in 2027.",
        }
    ]
    assert audit["covenant_status"] == "COMPLIANCE_EVIDENCED"
    assert audit["covenant_terms_status"] == "COVENANT_TERMS_EXTRACTED"
    assert audit["covenant_terms"] == [
        {
            "metric": "consolidated net leverage ratio",
            "condition": "MAXIMUM",
            "threshold": 3.5,
            "unit": "ratio",
            "context": "Maximum consolidated net leverage ratio of 3.50 to 1.00.",
        }
    ]
    assert audit["debt_due_within_12mo"] is True
    assert audit["cash_runway_quarters"] == 6.0
    assert "NEAR_TERM_MATURITY_UNRESOLVED" not in audit["confidence_caps"]


def test_selection_audit_allows_selected_with_expected_return_and_two_pillars():
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.08),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["downside_evidence_status"] == "PRESENT"
    assert audit["selected_company_evidence_tools"] == [
        "analyze_liquidity_stress",
        "fetch_kpi_trends",
    ]
    assert audit["selected_evidence_pillars"] == ["liquidity", "quality"]
    assert audit["selected_risk_evidence_tools"] == ["analyze_liquidity_stress"]


def test_selection_audit_caps_missing_framework_required_evidence():
    framework = SectorFinancialFramework(
        sector="software",
        market_cap_focus="small_cap",
        horizon_years=[5, 10],
        economic_model="Technology compounding model.",
        required_evidence=[
            "expected_return_scenarios",
            "kpi_trends",
            "sbc_and_share_count_evidence",
        ],
    )
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
        framework=framework,
    )

    assert audit["status"] == "PASS"
    assert audit["framework_required_evidence"] == [
        "expected_return_scenarios",
        "kpi_trends",
        "sbc_and_share_count_evidence",
    ]
    assert audit["framework_required_evidence_covered"] == [
        "expected_return_scenarios",
        "kpi_trends",
    ]
    assert audit["framework_required_evidence_missing"] == ["sbc_and_share_count_evidence"]
    assert audit["framework_required_evidence_coverage_ratio"] == 2 / 3
    assert "FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE" in audit["confidence_caps"]
    assert audit["needs_evidence_resolution"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]


def test_selection_audit_does_not_credit_generic_evidence_for_automotive_residual_risk():
    framework = SectorFinancialFramework(
        sector="automotive",
        market_cap_focus="small_cap",
        horizon_years=[5, 10],
        economic_model="Automotive and mobility cycle model.",
        required_evidence=[
            "unit_volume_mix_and_pricing_evidence",
            "warranty_recall_or_quality_evidence",
            "finance_residual_value_or_leverage_evidence",
        ],
    )
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2",
            "fetch_kpi_trends",
            {},
            "Validate unit volume, pricing, and quality.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
        ToolCallRecord(
            "TC3",
            "analyze_dilution",
            {},
            "Validate share-count discipline.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA unit, pricing, and quality evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_dilution",
            "AAA dilution evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
        framework=framework,
    )

    assert audit["status"] == "PASS"
    assert audit["framework_required_evidence_covered"] == [
        "unit_volume_mix_and_pricing_evidence",
        "warranty_recall_or_quality_evidence",
    ]
    assert audit["framework_required_evidence_missing"] == [
        "finance_residual_value_or_leverage_evidence",
    ]
    assert audit["framework_required_evidence_coverage_ratio"] == 2 / 3
    assert audit["confidence_caps"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]
    assert audit["needs_evidence_resolution"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]


def test_selection_audit_does_not_credit_generic_evidence_for_payments_losses_and_capital():
    framework = SectorFinancialFramework(
        sector="payments_fintech",
        market_cap_focus="small_cap",
        horizon_years=[5, 10],
        economic_model="Payments and fintech network model.",
        required_evidence=[
            "payment_volume_transaction_and_take_rate_evidence",
            "fraud_credit_loss_and_chargeback_evidence",
            "funding_float_or_regulatory_capital_evidence",
        ],
    )
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2",
            "fetch_kpi_trends",
            {},
            "Validate payment volume and take rate.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
        ToolCallRecord(
            "TC3",
            "analyze_dilution",
            {},
            "Validate dilution.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA payment volume and take-rate evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_dilution",
            "AAA dilution evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
        framework=framework,
    )

    assert audit["status"] == "PASS"
    assert audit["framework_required_evidence_covered"] == [
        "payment_volume_transaction_and_take_rate_evidence",
    ]
    assert audit["framework_required_evidence_missing"] == [
        "fraud_credit_loss_and_chargeback_evidence",
        "funding_float_or_regulatory_capital_evidence",
    ]
    assert audit["framework_required_evidence_coverage_ratio"] == 1 / 3
    assert audit["confidence_caps"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]
    assert audit["needs_evidence_resolution"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]


def test_selection_audit_does_not_credit_generic_evidence_for_medical_device_regulatory_reimbursement_and_channel_needs():
    framework = SectorFinancialFramework(
        sector="medical_devices",
        market_cap_focus="small_cap",
        horizon_years=[5, 10],
        economic_model="Medical-device compounding model.",
        required_evidence=[
            "procedure_volume_utilization_or_installed_base_evidence",
            "reimbursement_site_of_care_or_payer_evidence",
            "fda_clearance_quality_system_or_recall_evidence",
            "channel_customer_concentration_or_hospital_capex_evidence",
        ],
    )
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2",
            "fetch_kpi_trends",
            {},
            "Validate procedure and installed-base trends.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
        ToolCallRecord(
            "TC3",
            "fetch_current_events",
            {},
            "Validate recent product momentum.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA procedure volume and installed base evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "fetch_current_events",
            "AAA recent device launch momentum.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
        framework=framework,
    )

    assert audit["status"] == "PASS"
    assert audit["framework_required_evidence_covered"] == [
        "procedure_volume_utilization_or_installed_base_evidence",
    ]
    assert audit["framework_required_evidence_missing"] == [
        "reimbursement_site_of_care_or_payer_evidence",
        "fda_clearance_quality_system_or_recall_evidence",
        "channel_customer_concentration_or_hospital_capex_evidence",
    ]
    assert audit["framework_required_evidence_coverage_ratio"] == 1 / 4
    assert audit["confidence_caps"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]
    assert audit["needs_evidence_resolution"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]


def test_selection_audit_applies_medical_device_gates_to_variant_framework_requirement_names():
    framework = SectorFinancialFramework(
        sector="medical_devices",
        market_cap_focus="small_cap",
        horizon_years=[5, 10],
        economic_model="Medical-device compounding model.",
        required_evidence=[
            "procedure_volume_utilization_or_installed_base_evidence",
            "site_of_care_or_payer_evidence",
            "device_regulatory_status_evidence",
            "customer_concentration_or_hospital_capex_evidence",
        ],
    )
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2",
            "fetch_kpi_trends",
            {},
            "Validate procedure and installed-base trends.",
            status="OK",
            evidence_ref_ids=["E2"],
        ),
        ToolCallRecord(
            "TC3",
            "fetch_current_events",
            {},
            "Validate product momentum.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA procedure utilization and installed base evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "fetch_current_events",
            "AAA revenue growth, cash conversion, product launch, and business quality momentum.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
        framework=framework,
    )

    assert audit["status"] == "PASS"
    assert audit["framework_required_evidence_covered"] == [
        "procedure_volume_utilization_or_installed_base_evidence",
    ]
    assert audit["framework_required_evidence_missing"] == [
        "site_of_care_or_payer_evidence",
        "device_regulatory_status_evidence",
        "customer_concentration_or_hospital_capex_evidence",
    ]
    assert audit["framework_required_evidence_coverage_ratio"] == 1 / 4
    assert audit["confidence_caps"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]
    assert audit["needs_evidence_resolution"] == ["FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE"]


def test_framework_packet_preflight_does_not_credit_generic_medical_device_fields():
    packet = _audit_packet("AAA")
    packet.business_quality = {
        "revenue_cagr_5y": 0.18,
        "earnings_quality": "HIGH",
        "moat_score": "STRONG",
    }
    packet.cash_conversion = {
        "cash_conversion_ratio": 0.90,
        "fcf_margin": 0.22,
    }
    packet.accounting_quality = {
        "filing_risk_status": "LOW",
        "filing_risk_evidence_status": "CURRENT",
    }

    assert _framework_requirement_packet_support(
        "procedure_volume_utilization_or_installed_base_evidence",
        packet=packet,
        has_base_return_scenario=False,
    ) == ["business_quality", "cash_conversion"]
    assert (
        _framework_requirement_packet_support(
            "reimbursement_site_of_care_or_payer_evidence",
            packet=packet,
            has_base_return_scenario=False,
        )
        == []
    )
    assert (
        _framework_requirement_packet_support(
            "fda_clearance_quality_system_or_recall_evidence",
            packet=packet,
            has_base_return_scenario=False,
        )
        == []
    )
    assert (
        _framework_requirement_packet_support(
            "channel_customer_concentration_or_hospital_capex_evidence",
            packet=packet,
            has_base_return_scenario=False,
        )
        == []
    )

    packet.accounting_quality["fda_clearance_status"] = "CLEARED"
    packet.business_quality["site_of_care_status"] = "SUPPORTED"
    packet.business_quality["customer_or_channel_concentration"] = "LOW"

    assert _framework_requirement_packet_support(
        "reimbursement_site_of_care_or_payer_evidence",
        packet=packet,
        has_base_return_scenario=False,
    ) == ["business_quality"]
    assert _framework_requirement_packet_support(
        "fda_clearance_quality_system_or_recall_evidence",
        packet=packet,
        has_base_return_scenario=False,
    ) == ["accounting_quality"]
    assert _framework_requirement_packet_support(
        "channel_customer_concentration_or_hospital_capex_evidence",
        packet=packet,
        has_base_return_scenario=False,
    ) == ["business_quality"]


def test_framework_packet_preflight_applies_medical_device_gates_to_variant_requirement_names():
    packet = _audit_packet("AAA")
    packet.business_quality = {
        "revenue_cagr_5y": 0.18,
        "earnings_quality": "HIGH",
        "moat_score": "STRONG",
    }
    packet.cash_conversion = {
        "cash_conversion_ratio": 0.90,
        "fcf_margin": 0.22,
    }
    packet.accounting_quality = {
        "filing_risk_status": "LOW",
        "filing_risk_evidence_status": "CURRENT",
    }

    assert _framework_requirement_packet_support(
        "procedure_volume_utilization_or_installed_base_evidence",
        packet=packet,
        has_base_return_scenario=False,
        sector="medical_devices",
    ) == ["business_quality", "cash_conversion"]
    assert (
        _framework_requirement_packet_support(
            "site_of_care_or_payer_evidence",
            packet=packet,
            has_base_return_scenario=False,
            sector="medical_devices",
        )
        == []
    )
    assert (
        _framework_requirement_packet_support(
            "device_regulatory_status_evidence",
            packet=packet,
            has_base_return_scenario=False,
            sector="medical_devices",
        )
        == []
    )
    assert (
        _framework_requirement_packet_support(
            "customer_concentration_or_hospital_capex_evidence",
            packet=packet,
            has_base_return_scenario=False,
            sector="medical_devices",
        )
        == []
    )

    packet.business_quality["payer_mix_status"] = "SUPPORTED"
    packet.accounting_quality["regulatory_status"] = "CURRENT"
    packet.cash_conversion["hospital_capex_cycle_status"] = "NORMALIZED"

    assert _framework_requirement_packet_support(
        "site_of_care_or_payer_evidence",
        packet=packet,
        has_base_return_scenario=False,
        sector="medical_devices",
    ) == ["business_quality"]
    assert _framework_requirement_packet_support(
        "device_regulatory_status_evidence",
        packet=packet,
        has_base_return_scenario=False,
        sector="medical_devices",
    ) == ["accounting_quality"]
    assert _framework_requirement_packet_support(
        "customer_concentration_or_hospital_capex_evidence",
        packet=packet,
        has_base_return_scenario=False,
        sector="medical_devices",
    ) == ["cash_conversion"]


def test_framework_required_evidence_tool_hints_cover_specialized_requirements():
    hints = _framework_required_evidence_tool_hints(
        [
            "finance_residual_value_or_leverage_evidence",
            "fraud_credit_loss_and_chargeback_evidence",
        ],
        [
            "fetch_kpi_trends",
            "fetch_companyfacts_timeseries",
            "fetch_filing_section",
            "fetch_current_events",
            "analyze_liquidity_stress",
        ],
    )

    assert hints == [
        {
            "required_evidence": "finance_residual_value_or_leverage_evidence",
            "suggested_tools": [
                "fetch_kpi_trends",
                "fetch_companyfacts_timeseries",
                "fetch_filing_section",
                "fetch_current_events",
                "analyze_liquidity_stress",
            ],
        },
        {
            "required_evidence": "fraud_credit_loss_and_chargeback_evidence",
            "suggested_tools": [
                "fetch_kpi_trends",
                "fetch_companyfacts_timeseries",
                "fetch_filing_section",
                "fetch_current_events",
                "analyze_liquidity_stress",
            ],
        },
    ]


def test_framework_required_evidence_tool_hints_cover_medical_device_requirements():
    hints = _framework_required_evidence_tool_hints(
        [
            "procedure_volume_utilization_or_installed_base_evidence",
            "fda_clearance_quality_system_or_recall_evidence",
        ],
        [
            "fetch_kpi_trends",
            "fetch_companyfacts_timeseries",
            "fetch_filing_section",
            "fetch_current_events",
            "fetch_recent_filing_context",
        ],
    )

    assert hints == [
        {
            "required_evidence": "procedure_volume_utilization_or_installed_base_evidence",
            "suggested_tools": [
                "fetch_kpi_trends",
                "fetch_companyfacts_timeseries",
                "fetch_filing_section",
                "fetch_current_events",
                "fetch_recent_filing_context",
            ],
        },
        {
            "required_evidence": "fda_clearance_quality_system_or_recall_evidence",
            "suggested_tools": [
                "fetch_kpi_trends",
                "fetch_companyfacts_timeseries",
                "fetch_filing_section",
                "fetch_current_events",
                "fetch_recent_filing_context",
            ],
        },
    ]


def test_selection_audit_passes_when_framework_required_evidence_is_covered():
    framework = SectorFinancialFramework(
        sector="software",
        market_cap_focus="small_cap",
        horizon_years=[5, 10],
        economic_model="Technology compounding model.",
        required_evidence=[
            "expected_return_scenarios",
            "kpi_trends",
            "sbc_and_share_count_evidence",
        ],
    )
    tool_calls = [
        ToolCallRecord(
            "TC1",
            "rank_expected_return_cases",
            {"tickers": ["AAA"], "horizon_years": 5},
            "Rank.",
            status="OK",
            evidence_ref_ids=["E1"],
        ),
        ToolCallRecord(
            "TC2", "fetch_kpi_trends", {}, "Validate KPI.", status="OK", evidence_ref_ids=["E2"]
        ),
        ToolCallRecord(
            "TC3",
            "analyze_liquidity_stress",
            {},
            "Validate liquidity.",
            status="OK",
            evidence_ref_ids=["E3"],
        ),
        ToolCallRecord(
            "TC4",
            "analyze_dilution",
            {"years": 5},
            "Validate share-count dilution.",
            status="OK",
            evidence_ref_ids=["E4"],
        ),
    ]
    evidence = [
        EvidenceReference(
            "E1",
            "tool_output",
            "rank_expected_return_cases",
            "AAA ranked first.",
            "AAA",
            tool_call_id="TC1",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E2",
            "tool_output",
            "fetch_kpi_trends",
            "AAA KPI evidence.",
            "AAA",
            tool_call_id="TC2",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E3",
            "tool_output",
            "analyze_liquidity_stress",
            "AAA liquidity evidence.",
            "AAA",
            tool_call_id="TC3",
            confidence="MODERATE",
        ),
        EvidenceReference(
            "E4",
            "tool_output",
            "analyze_dilution",
            "AAA share-count dilution evidence.",
            "AAA",
            tool_call_id="TC4",
            confidence="MODERATE",
        ),
    ]

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _audit_packet("AAA")},
        scenarios=[
            _audit_base_scenario("AAA"),
            _audit_downside_scenario("AAA", annualized_return=-0.02),
        ],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
        framework=framework,
    )

    assert audit["status"] == "PASS"
    assert audit["downside_evidence_status"] == "PRESENT"
    assert audit["framework_required_evidence_missing"] == []
    assert audit["framework_required_evidence_coverage_ratio"] == 1.0
    assert "FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE" not in audit["confidence_caps"]


def test_sector_expected_return_tools_return_decision_usability_metadata():
    packets_by_ticker = {"AAA": _audit_packet("AAA"), "BBB": _audit_packet("BBB")}
    scenarios = [_audit_base_scenario("AAA"), _audit_base_scenario("BBB")]

    ranked = _dispatch_sector_tool(
        tool_name="rank_expected_return_cases",
        tool_input={"horizon_years": 5, "scenario_name": "base"},
        tickers=["BBB", "AAA"],
        packets_by_ticker=packets_by_ticker,
        scenarios=scenarios,
    )
    compared = _dispatch_sector_tool(
        tool_name="compare_expected_return_scenarios",
        tool_input={"horizon_years": 5, "scenario_name": "base"},
        tickers=["AAA", "BBB"],
        packets_by_ticker=packets_by_ticker,
        scenarios=scenarios,
    )

    assert ranked["status"] == "ok"
    assert ranked["usable_for_decision"] is True
    assert ranked["evidence_status"] == "EXPECTED_RETURN_RANKING_AVAILABLE"
    assert ranked["tickers"] == ["BBB", "AAA"]
    assert ranked["covered_tickers"] == ["AAA", "BBB"]
    assert ranked["ranked_tickers"] == ["AAA", "BBB"]
    assert ranked["summary"] == "Ranked 2 base 5Y expected-return case(s): AAA, BBB."
    assert compared["status"] == "ok"
    assert compared["usable_for_decision"] is True
    assert compared["evidence_status"] == "EXPECTED_RETURN_SCENARIOS_AVAILABLE"
    assert compared["tickers"] == ["AAA", "BBB"]
    assert compared["covered_tickers"] == ["AAA", "BBB"]
    assert (
        compared["summary"] == "Compared 2 expected-return scenario(s) for 2 ticker(s): AAA, BBB."
    )


def test_sector_runtime_compare_expected_return_tool_outputs_evidence_metadata(monkeypatch):
    plan = {
        "framework": _framework_payload(),
        "research_questions": [
            {
                "question_id": "Q1",
                "question": "Compare base-case expected returns for the finalists.",
                "financial_pillar": "Expected return",
                "expected_decision_impact": "Determines whether the selected ticker clears scenario support.",
                "priority": "HIGH",
                "target_tickers": ["AAA", "BBB"],
                "planned_tool_calls": [
                    {
                        "tool_name": "compare_expected_return_scenarios",
                        "ticker": None,
                        "tool_input": {"horizon_years": 5, "scenario_name": "base"},
                        "rationale": "Compare base-case expected-return scenarios.",
                    },
                    {
                        "tool_name": "fetch_kpi_trends",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Check whether AAA's financial quality supports the scenario.",
                    },
                    {
                        "tool_name": "analyze_liquidity_stress",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Check whether AAA's balance-sheet risk offsets the scenario.",
                    },
                ],
            }
        ],
        "belief_updates": [],
        "continue_research": True,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": [],
    }
    provider = FakeProvider([plan, _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI trends support the scenario.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    compare_preview = artifact.tool_calls[0].output_preview or ""
    assert artifact.final_verdict == "SELECTED"
    assert artifact.selection_audit["expected_return_evidence_count"] == 1
    assert '"evidence_status": "EXPECTED_RETURN_SCENARIOS_AVAILABLE"' in compare_preview
    assert '"covered_tickers": ["AAA", "BBB"]' in compare_preview
    assert '"scenario_count": 2' in compare_preview
    assert (
        artifact.evidence[0].summary
        == "Compared 2 expected-return scenario(s) for 2 ticker(s): AAA, BBB."
    )


def test_sector_runtime_downgrades_stale_filing_selection_to_watchlist(monkeypatch):
    packets = _packets()
    packets["AAA"].filing_risk_status = "OK"
    packets["AAA"].filing_risk_metadata = {
        "evidence_status": "STALE_READABLE_RISK_SECTION",
        "source_filing_date": "2021-02-18",
        "source_filing_age_days": 1893,
        "risk_text_chars": 30000,
        "warnings": ["stale_annual_filing:1893d"],
    }
    provider = FakeProvider([_planning_turn(), _high_confidence_final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "fetch_recent_filing_context":
            return {
                "status": "unavailable",
                "ticker": ctx.ticker,
                "usable_for_decision": False,
                "evidence_status": "NO_RECENT_FILING_CONTEXT",
                "summary": "No readable recent quarterly or material-event filing context was available.",
            }
        return {"status": "ok", "ticker": ctx.ticker, "summary": "KPI evidence is usable."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=("fetch_recent_filing_context",),
            evidence_confidence="LOW",
            summaries={
                "fetch_recent_filing_context": (
                    "No readable recent quarterly or material-event filing context was available."
                )
            },
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.confidence == "MODERATE"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["actionable"] is True
    assert artifact.selection_audit["confidence_ceiling"] == "MODERATE"
    assert artifact.selection_audit["confidence_caps"] == [
        "STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE"
    ]
    assert artifact.final_decision is not None
    assert artifact.final_decision.confidence_cap_reasons == [
        "CYCLE_DURABILITY_UNRESOLVED",
        "STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE",
    ]
    assert artifact.watchlist_resolution_attempted is False
    assert artifact.watchlist_resolution_status is None
    assert artifact.no_selection_finalist_resolution_attempted is False
    assert artifact.no_selection_finalist_resolution_status is None
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]
    assert "SELECTION_AUDIT_WATCHLIST_ONLY" not in artifact.degraded_states


def test_sector_runtime_watchlist_resolution_can_upgrade_to_selected(monkeypatch):
    packets = _packets()
    packets["AAA"].filing_risk_status = "OK"
    packets["AAA"].filing_risk_metadata = {
        "evidence_status": "STALE_READABLE_RISK_SECTION",
        "source_filing_date": "2021-02-18",
        "source_filing_age_days": 1893,
        "risk_text_chars": 30000,
        "warnings": ["stale_annual_filing:1893d"],
    }
    provider = FakeProvider([_planning_turn(), _high_confidence_final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "fetch_recent_filing_context":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "evidence_status": "RECENT_FILING_CONTEXT_AVAILABLE",
                "summary": "1 recent quarterly/material-event filing document(s) available.",
            }
        return {"status": "ok", "ticker": ctx.ticker, "summary": "KPI evidence is usable."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker, labels=("fetch_recent_filing_context",)
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.confidence == "MODERATE"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["confidence_caps"] == [
        "STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE"
    ]
    assert artifact.final_decision is not None
    assert artifact.final_decision.confidence_cap_reasons == [
        "CYCLE_DURABILITY_UNRESOLVED",
        "STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE",
    ]
    assert artifact.watchlist_resolution_attempted is False
    assert artifact.watchlist_resolution_status is None
    assert "SELECTION_AUDIT_WATCHLIST_ONLY" not in artifact.degraded_states
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]


def test_sector_runtime_watchlist_resolution_current_events_can_clear_freshness_cap(monkeypatch):
    packets = _packets()
    packets["AAA"].filing_risk_status = "OK"
    packets["AAA"].filing_risk_metadata = {
        "evidence_status": "STALE_READABLE_RISK_SECTION",
        "source_filing_date": "2021-02-18",
        "source_filing_age_days": 1893,
        "risk_text_chars": 30000,
        "warnings": ["stale_annual_filing:1893d"],
    }
    provider = FakeProvider([_planning_turn(), _high_confidence_final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "fetch_current_events":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "evidence_status": "CURRENT_EVENTS_AVAILABLE",
                "summary": (
                    "1 current-event document(s) available from configured company-controlled sources "
                    "(metadata=universe_members, homepage=present, ir_rss=missing)."
                ),
            }
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Company-specific evidence is usable.",
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=("fetch_current_events",),
            summaries={
                "fetch_current_events": (
                    "1 current-event document(s) available from configured company-controlled sources "
                    "(metadata=universe_members, homepage=present, ir_rss=missing)."
                )
            },
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["confidence_caps"] == [
        "STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE"
    ]
    assert artifact.watchlist_resolution_attempted is False
    assert artifact.watchlist_resolution_status is None
    assert artifact.tool_calls[-1].tool_name == "analyze_liquidity_stress"
    assert artifact.evidence[-1].confidence == "MODERATE"


def test_sector_runtime_watchlist_prompt_prioritizes_missing_framework_evidence(monkeypatch):
    provider = FakeProvider(
        [
            _planning_turn(include_second_company_tool=False),
            _high_confidence_final_turn(),
        ]
    )
    child_calls: list[dict] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence is usable.",
        },
    )

    def fake_child_run(ticker, **kwargs):
        child_calls.append({"ticker": ticker, **kwargs})
        return _child_company_artifact(ticker, labels=("analyze_dilution",))

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="enterprise_software",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=8, max_turns=4),
    )

    assert artifact.watchlist_resolution_attempted is False
    assert artifact.watchlist_resolution_status is None
    # Status stays an actionable PASS (these are evidence-quality confidence caps,
    # not hard blockers). The INSUFFICIENT_* coverage caps now classify
    # EVIDENCE_QUALITY and surface alongside the framework gap as resolution needs.
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["needs_evidence_resolution"] == [
        "INSUFFICIENT_COMPANY_SPECIFIC_EVIDENCE",
        "INSUFFICIENT_EVIDENCE_PILLAR_COVERAGE",
        "FRAMEWORK_REQUIRED_EVIDENCE_INCOMPLETE",
    ]
    assert len(provider.calls) == 2
    assert child_calls == []


def test_sector_runtime_watchlist_resolution_can_downgrade_to_no_selection(monkeypatch):
    packets = _packets()
    packets["AAA"].filing_risk_status = "OK"
    packets["AAA"].filing_risk_metadata = {
        "evidence_status": "STALE_READABLE_RISK_SECTION",
        "source_filing_date": "2021-02-18",
        "source_filing_age_days": 1893,
        "risk_text_chars": 30000,
        "warnings": ["stale_annual_filing:1893d"],
    }
    provider = FakeProvider(
        [
            _planning_turn(include_second_company_tool=False),
            _high_confidence_final_turn(),
        ]
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "analyze_liquidity_stress":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "solvency_risk": "CRITICAL",
                "summary": "Follow-up liquidity stress exposed critical solvency risk.",
            }
        return {"status": "ok", "ticker": ctx.ticker, "summary": "KPI evidence is usable."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=("analyze_liquidity_stress",),
            summaries={
                "analyze_liquidity_stress": "Follow-up liquidity stress exposed critical solvency risk."
            },
            excerpts={"analyze_liquidity_stress": '{"solvency_risk": "critical"}'},
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    # Only one business-quality cap (STALE_ANNUAL_FILING_WITHOUT_FRESHER_DECISION_EVIDENCE)
    # now drives the ceiling — the INSUFFICIENT_* coverage caps are EVIDENCE_QUALITY
    # and no longer count toward it — so the ceiling is MODERATE, not LOW.
    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.confidence == "MODERATE"
    assert artifact.selection_audit["status"] == "PASS"
    assert "FOLLOW_UP_SOLVENCY_CRITICAL" not in artifact.selection_audit["hard_blockers"]
    assert artifact.watchlist_resolution_attempted is False
    assert artifact.watchlist_resolution_status is None
    assert artifact.final_decision is not None
    assert artifact.final_decision.selection_blockers == []


def test_sector_runtime_capital_structure_repair_can_clear_no_assurance_blocker(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "analyze_liquidity_stress":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "solvency_risk": "ELEVATED",
                "solvency": {
                    "risk": "ELEVATED",
                    "signals": ["NO_ASSURANCE_FINANCING"],
                    "no_assurance_financing": True,
                    "going_concern_language": False,
                },
                "summary": "AAA liquidity evidence includes a no-assurance financing flag.",
            }
        if name == "analyze_capital_structure_resolution":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "evidence_status": "CAPITAL_STRUCTURE_RESOLVED_CLEAR",
                "summary": "AAA refinancing was completed and covenant compliance is current.",
            }
        return {"status": "ok", "ticker": ctx.ticker, "summary": "Company evidence is usable."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=("analyze_capital_structure_resolution",),
            summaries={
                "analyze_capital_structure_resolution": (
                    "CAPITAL_STRUCTURE_RESOLVED_CLEAR: AAA refinancing was completed and covenant compliance is current."
                )
            },
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert (
        artifact.selection_audit["capital_structure_resolution_status"]
        == "CAPITAL_STRUCTURE_RESOLVED_CLEAR"
    )
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "RESOLVED_SELECTED"
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
        "analyze_capital_structure_resolution",
    ]


def test_sector_runtime_capital_structure_repair_preserves_active_distress_blocker(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "analyze_liquidity_stress":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "solvency": {"risk": "ELEVATED", "no_assurance_financing": True},
                "summary": "AAA liquidity evidence includes a no-assurance financing flag.",
            }
        if name == "analyze_capital_structure_resolution":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "evidence_status": "CAPITAL_STRUCTURE_ACTIVE_DISTRESS",
                "summary": "AAA capital-structure evidence indicates active distress.",
            }
        return {"status": "ok", "ticker": ctx.ticker, "summary": "Company evidence is usable."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=("analyze_capital_structure_resolution",),
            summaries={
                "analyze_capital_structure_resolution": (
                    "CAPITAL_STRUCTURE_ACTIVE_DISTRESS: AAA capital-structure evidence indicates active distress."
                )
            },
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selection_audit["status"] == "BLOCKED"
    assert (
        "FOLLOW_UP_CAPITAL_STRUCTURE_ACTIVE_DISTRESS" in artifact.selection_audit["hard_blockers"]
    )
    assert (
        artifact.selection_audit["capital_structure_resolution_status"]
        == "CAPITAL_STRUCTURE_ACTIVE_DISTRESS"
    )
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "UNRESOLVED_BLOCKED"


def test_sector_runtime_capital_structure_repair_can_downgrade_to_watchlist(monkeypatch):
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "analyze_liquidity_stress":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "solvency": {"risk": "ELEVATED", "no_assurance_financing": True},
                "summary": "AAA liquidity evidence includes a no-assurance financing flag.",
            }
        if name == "analyze_capital_structure_resolution":
            return {
                "status": "ok",
                "ticker": ctx.ticker,
                "usable_for_decision": True,
                "evidence_status": "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST",
                "summary": "AAA refinancing evidence is decision-usable but maturity risk remains a watchlist cap.",
            }
        return {"status": "ok", "ticker": ctx.ticker, "summary": "Company evidence is usable."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _child_company_artifact(
            ticker,
            labels=("analyze_capital_structure_resolution",),
            summaries={
                "analyze_capital_structure_resolution": (
                    "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST: AAA refinancing evidence is decision-usable "
                    "but maturity risk remains a watchlist cap."
                )
            },
        ),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=5),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert "CAPITAL_STRUCTURE_WATCHLIST" in artifact.selection_audit["confidence_caps"]
    assert (
        artifact.selection_audit["capital_structure_resolution_status"]
        == "CAPITAL_STRUCTURE_RESOLVED_WATCHLIST"
    )
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "RESOLVED_SELECTED"


def test_sector_runtime_watchlist_resolution_respects_remaining_tool_budget(monkeypatch):
    packets = _packets()
    packets["AAA"].filing_risk_status = "OK"
    packets["AAA"].filing_risk_metadata = {
        "evidence_status": "STALE_READABLE_RISK_SECTION",
        "source_filing_date": "2021-02-18",
        "source_filing_age_days": 1893,
        "risk_text_chars": 30000,
        "warnings": ["stale_annual_filing:1893d"],
    }
    provider = FakeProvider(
        [_planning_turn(include_second_company_tool=False), _high_confidence_final_turn()]
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=2),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.watchlist_resolution_attempted is False
    assert artifact.watchlist_resolution_status is None
    assert "WATCHLIST_RESOLUTION_BUDGET_EXHAUSTED" not in artifact.degraded_states
    assert len(provider.calls) == 2
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
    ]


def test_sector_runtime_blocks_selected_candidate_without_readable_filing(monkeypatch):
    packets = _packets()
    packets["AAA"].filing_risk_status = "NO_FILING"
    packets["AAA"].filing_risk_metadata = {
        "evidence_status": "NO_READABLE_ANNUAL_FILING",
        "risk_text_chars": 0,
        "warnings": ["annual_filing_missing"],
    }
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    # A selected candidate whose annual filing cannot be read is no longer a
    # clean actionable PASS: the NO_FILING-class EVIDENCE_QUALITY hard blockers
    # surface it as DATA_INCOMPLETE (resolve-then-promote, ticker preserved,
    # not actionable) rather than silently passing.
    assert artifact.final_verdict == "DATA_INCOMPLETE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "DATA_INCOMPLETE"
    assert artifact.selection_audit["actionable"] is False
    assert "NO_FILING" in artifact.selection_audit["hard_blockers"]
    assert "NO_READABLE_ANNUAL_FILING" in artifact.selection_audit["hard_blockers"]
    assert artifact.selection_audit["needs_evidence_resolution"] == [
        "NO_FILING",
        "NO_READABLE_ANNUAL_FILING",
        "FILING_RISK_NO_FILING",
        "CAPITAL_LOSS_EVIDENCE_DEGRADED",
    ]
    assert artifact.final_decision is not None
    assert artifact.final_decision.selection_blockers == []
    assert artifact.final_decision.data_resolution_needed == [
        "NO_FILING",
        "NO_READABLE_ANNUAL_FILING",
        "FILING_RISK_NO_FILING",
        "CAPITAL_LOSS_EVIDENCE_DEGRADED",
    ]


def test_sector_runtime_ignores_first_turn_final_decision_without_tool_plan(monkeypatch):
    # Pin legacy hard-block hurdle behavior (production default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    provider = FakeProvider([{**_final_turn(), "framework": _framework_payload()}])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.no_selection_reason == "No research plan or final decision was returned."
    assert artifact.degraded_states == ["INITIAL_FINAL_DECISION_IGNORED", "NO_RESEARCH_PLAN"]
    assert any(
        "Ignored provider final decision on compact first planning turn" in note
        for note in artifact.audit_notes
    )


def test_sector_runtime_falls_back_when_initial_provider_plan_has_no_tools(monkeypatch):
    framework_only_turn = {
        "framework": _framework_payload(),
        "research_questions": [],
        "belief_updates": [],
        "continue_research": True,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": ["Framework returned without an executable plan."],
    }
    provider = FakeProvider([framework_only_turn, _final_turn()])
    dispatched: list[tuple[str, str]] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    reverse_order_packets = _packets()
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: {
            "BBB": reverse_order_packets["BBB"],
            "AAA": reverse_order_packets["AAA"],
        },
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append((name, ctx.ticker))
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert "DETERMINISTIC_INITIAL_PLAN_FALLBACK" in artifact.degraded_states
    assert "NO_RESEARCH_PLAN" not in artifact.degraded_states
    assert artifact.research_questions[0].question_id == "DTP1"
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "fetch_recent_filing_context",
        "fetch_companyfacts_timeseries",
    ]
    assert artifact.evidence[0].ticker == "AAA"
    assert dispatched == [
        ("fetch_kpi_trends", "AAA"),
        ("fetch_recent_filing_context", "AAA"),
        ("fetch_companyfacts_timeseries", "AAA"),
    ]
    assert any(
        "created a deterministic fallback plan focused on AAA" in note
        for note in artifact.audit_notes
    )
    assert len(provider.calls) == 2


def test_sector_runtime_audit_gap_repair_can_add_expected_return_evidence_and_select(monkeypatch):
    # Pin legacy hard-block hurdle behavior (production default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    company_only_plan = _planning_turn()
    company_only_plan["research_questions"][0]["planned_tool_calls"] = [
        {
            "tool_name": "fetch_kpi_trends",
            "ticker": "AAA",
            "tool_input": {},
            "rationale": "Check whether AAA's financial quality supports selection.",
        },
        {
            "tool_name": "analyze_liquidity_stress",
            "ticker": "AAA",
            "tool_input": {},
            "rationale": "Check whether AAA's risk evidence supports selection.",
        },
    ]
    provider = FakeProvider([company_only_plan, _final_turn()])
    child_calls: list[dict] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI evidence is usable.",
        },
    )

    def fake_child_run(ticker, **kwargs):
        child_calls.append({"ticker": ticker, **kwargs})
        return _child_company_artifact(ticker, labels=("fetch_companyfacts_timeseries",))

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["expected_return_evidence_count"] == 1
    assert artifact.selection_audit["company_specific_evidence_count"] == 2
    assert (
        "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" not in artifact.selection_audit["hard_blockers"]
    )
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "RESOLVED_SELECTED"
    assert artifact.research_questions[-1].question_id == "AGR2"
    assert [call.tool_name for call in artifact.tool_calls] == [
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
        "rank_expected_return_cases",
    ]
    assert len(provider.calls) == 2
    assert child_calls == []


def test_sector_runtime_audit_gap_repair_can_move_missing_company_evidence_to_watchlist(
    monkeypatch,
):
    packets = _packets()
    packets["AAA"].filing_risk_status = "OK"
    packets["AAA"].filing_risk_metadata = {
        "evidence_status": "STALE_READABLE_RISK_SECTION",
        "source_filing_date": "2021-02-18",
        "source_filing_age_days": 1893,
        "risk_text_chars": 30000,
        "warnings": ["stale_annual_filing:1893d"],
    }
    expected_only_plan = _planning_turn()
    expected_only_plan["research_questions"][0]["planned_tool_calls"] = [
        {
            "tool_name": "rank_expected_return_cases",
            "ticker": None,
            "tool_input": {"horizon_years": 5, "scenario_name": "base"},
            "rationale": "Rank base-case expected returns.",
        }
    ]
    provider = FakeProvider([expected_only_plan, _high_confidence_final_turn()])
    child_calls: list[str] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )

    def fake_dispatch(name, tool_input, ctx):
        if name == "fetch_recent_filing_context":
            return {
                "status": "unavailable",
                "ticker": ctx.ticker,
                "usable_for_decision": False,
                "evidence_status": "NO_RECENT_FILING_CONTEXT",
                "summary": "No readable recent quarterly or material-event filing context was available.",
            }
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Company-specific KPI evidence is usable.",
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    def fake_child_run(ticker, **kwargs):
        child_calls.append(ticker)
        if len(child_calls) == 1:
            return _child_company_artifact(ticker, labels=("fetch_kpi_trends",))
        return _child_company_artifact(
            ticker,
            labels=("fetch_recent_filing_context",),
            evidence_confidence="LOW",
            summaries={
                "fetch_recent_filing_context": (
                    "No readable recent quarterly or material-event filing context was available."
                )
            },
        )

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=6),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["company_specific_evidence_count"] == 1
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "RESOLVED_SELECTED"
    assert artifact.watchlist_resolution_attempted is False
    assert artifact.watchlist_resolution_status is None
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
    ]
    assert child_calls == ["AAA"]


def test_sector_runtime_does_not_repair_non_repairable_audit_blockers(monkeypatch):
    packets = _packets()
    packets["AAA"].filing_risk_status = "NO_FILING"
    packets["AAA"].filing_risk_metadata = {
        "evidence_status": "NO_READABLE_ANNUAL_FILING",
        "risk_text_chars": 0,
        "warnings": ["annual_filing_missing"],
    }
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    # NO_FILING-class blockers are EVIDENCE_QUALITY data-availability gaps that
    # are NOT in AUDIT_GAP_REPAIRABLE_BLOCKERS, so no in-run repair is attempted.
    # The candidate surfaces as DATA_INCOMPLETE (resolve-then-promote,
    # ticker preserved) rather than a clean actionable PASS.
    assert artifact.final_verdict == "DATA_INCOMPLETE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "DATA_INCOMPLETE"
    assert "NO_FILING" in artifact.selection_audit["hard_blockers"]
    assert "NO_FILING" in artifact.selection_audit["needs_evidence_resolution"]
    assert artifact.audit_gap_repair_attempted is False
    assert artifact.audit_gap_repair_status is None
    assert len(provider.calls) == 2


def test_sector_runtime_audit_gap_repair_respects_remaining_tool_budget(monkeypatch):
    company_only_plan = _planning_turn()
    company_only_plan["research_questions"][0]["planned_tool_calls"] = [
        {
            "tool_name": "fetch_kpi_trends",
            "ticker": "AAA",
            "tool_input": {},
            "rationale": "Check whether AAA's financial quality supports selection.",
        }
    ]
    provider = FakeProvider([company_only_plan, _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=1),
    )

    # The expected-return gap is repairable in principle, but the tool budget is
    # exhausted so the repair is skipped. The unrepaired fetchable gap
    # surfaces the candidate as DATA_INCOMPLETE (ticker preserved) rather than
    # discarding it as NO_SELECTION.
    assert artifact.final_verdict == "DATA_INCOMPLETE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "SKIPPED_BUDGET_EXHAUSTED"
    assert "AUDIT_GAP_REPAIR_BUDGET_EXHAUSTED" in artifact.degraded_states
    assert [call.tool_name for call in artifact.tool_calls] == ["fetch_kpi_trends"]
    assert len(provider.calls) == 2


def test_sector_runtime_audit_gap_repair_requires_company_level_tools(monkeypatch):
    company_only_plan = _planning_turn()
    company_only_plan["research_questions"][0]["planned_tool_calls"] = [
        {
            "tool_name": "fetch_kpi_trends",
            "ticker": "AAA",
            "tool_input": {},
            "rationale": "Check whether AAA's financial quality supports selection.",
        }
    ]
    provider = FakeProvider([company_only_plan, _final_turn()])
    child_calls: list[str] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI evidence is usable.",
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: child_calls.append(ticker) or _child_company_artifact(ticker),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
        allowed_tools=["rank_expected_return_cases"],
    )

    # Only sector-level tools are allowed, so the company-level audit-gap repair
    # cannot run. The candidate's remaining obstacles are fetchable
    # MISSING_*_EVIDENCE gaps (EVIDENCE_QUALITY ), so it surfaces as
    # DATA_INCOMPLETE (ticker preserved) rather than NO_SELECTION.
    assert artifact.final_verdict == "DATA_INCOMPLETE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.audit_gap_repair_attempted is False
    assert artifact.audit_gap_repair_status is None
    assert child_calls == []
    assert "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" in artifact.selection_audit["hard_blockers"]


def test_sector_runtime_ignores_first_turn_final_decision_and_executes_tools(monkeypatch):
    premature_turn = _planning_turn(continue_research=False)
    premature_turn["final_decision"] = _final_turn()["final_decision"]
    provider = FakeProvider([premature_turn, _final_turn()])
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(name)
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Deterministic tool evidence was gathered.",
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]
    assert dispatched == ["fetch_kpi_trends", "analyze_liquidity_stress"]
    assert artifact.degraded_states == ["INITIAL_FINAL_DECISION_IGNORED"]
    assert artifact.audit_notes == [
        "Initial sector framework selected.",
        "Ignored provider final decision on compact first planning turn; deterministic tools must run before sector selection.",
        "Final decision made after reviewing deterministic tool output.",
    ]
    assert len(provider.calls) == 2


def test_sector_runtime_skips_duplicate_planned_tools_across_turns(monkeypatch):
    duplicate_final_turn = _planning_turn(continue_research=False)
    duplicate_final_turn["framework"] = None
    duplicate_final_turn["final_decision"] = _final_turn()["final_decision"]
    duplicate_final_turn["audit_notes"] = [
        "Final decision returned with an already-executed tool plan."
    ]
    provider = FakeProvider([_planning_turn(), duplicate_final_turn])
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(name)
        return {"status": "ok", "ticker": ctx.ticker, "summary": "KPI trends support the scenario."}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=6, max_turns=3),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
    ]
    assert dispatched == ["fetch_kpi_trends", "analyze_liquidity_stress"]
    assert artifact.degraded_states == ["DUPLICATE_PLANNED_TOOLS_SKIPPED"]
    assert artifact.audit_notes == [
        "Initial sector framework selected.",
        "Final decision returned with an already-executed tool plan.",
        "Skipped 3 duplicate planned tool call(s) already executed earlier in the run.",
    ]
    assert len(provider.calls) == 2


def test_sector_runtime_can_continue_for_multiple_research_turns(monkeypatch):
    second_turn = {
        "framework": None,
        "research_questions": [
            {
                "question_id": "Q2",
                "question": "Does dilution undermine the per-share return case?",
                "financial_pillar": "Capital allocation",
                "expected_decision_impact": "Could reduce confidence or force no selection.",
                "priority": "MEDIUM",
                "target_tickers": ["AAA"],
                "planned_tool_calls": [
                    {
                        "tool_name": "analyze_dilution",
                        "ticker": "AAA",
                        "tool_input": {"years": 5},
                        "rationale": "Check share-count trend.",
                    }
                ],
            }
        ],
        "belief_updates": [
            {
                "question_id": "Q1",
                "ticker": "AAA",
                "prior_belief": "AAA looked strongest on scenario math.",
                "updated_belief": "AAA still leads, but dilution needs a direct check.",
                "direction": "UNCHANGED",
                "confidence_after": "LOW",
                "summary": "Scenario ranking was positive but incomplete.",
                "evidence_ref_ids": ["E1"],
                "remaining_uncertainty": ["Dilution trend unresolved."],
            }
        ],
        "continue_research": True,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": ["Requested another deterministic check before selection."],
    }
    provider = FakeProvider([_planning_turn(), second_turn, _final_turn()])
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(name)
        return {"status": "ok", "ticker": ctx.ticker}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=4, max_turns=4),
    )

    assert artifact.final_verdict == "SELECTED"
    assert [question.question_id for question in artifact.research_questions] == ["Q1", "Q2"]
    assert [question.status for question in artifact.research_questions] == ["ANSWERED", "ANSWERED"]
    assert [call.tool_name for call in artifact.tool_calls] == [
        "rank_expected_return_cases",
        "fetch_kpi_trends",
        "analyze_liquidity_stress",
        "analyze_dilution",
    ]
    assert dispatched == ["fetch_kpi_trends", "analyze_liquidity_stress", "analyze_dilution"]
    assert len(artifact.belief_updates) == 2
    assert len(provider.calls) == 3


def test_sector_runtime_recovers_compact_final_decision_after_provider_json_failure(monkeypatch):
    provider = RecoveringProvider()
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(name)
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI evidence supports the finalist.",
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=4, max_turns=4),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.degraded_states == ["LLM_PROVIDER_INVALID_JSON", "LLM_PROVIDER_JSON_RECOVERED"]
    assert artifact.research_questions[0].status == "ANSWERED"
    assert dispatched == ["fetch_kpi_trends", "analyze_liquidity_stress"]
    assert provider.calls[2]["schema_name"] == "autonomous_sector_final_decision_recovery"
    assert any("Provider structured-output failure" in note for note in artifact.audit_notes)
    assert any("compact final-decision-only request" in note for note in artifact.audit_notes)
    assert "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE" not in artifact.degraded_states


def test_sector_mutated_tool_evidence_blocks_next_turn_provider_call(monkeypatch):
    from app.autonomous import sector_runtime as sector_runtime_module

    provider = FakeProvider([_planning_turn(), _final_turn()])
    real_turn_prompt = sector_runtime_module._turn_prompt

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} retained its deterministic result.",
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._run_company_autonomy_pass",
        lambda **kwargs: (False, None, [], [], [], kwargs["executed_tool_count"]),
    )

    def mutate_after_tool_production(**kwargs):
        kwargs["evidence"][0].summary = "forged annualized_return=9.99 after tool production"
        return real_turn_prompt(**kwargs)

    monkeypatch.setattr(
        "app.autonomous.sector_runtime._turn_prompt",
        mutate_after_tool_production,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_autonomous_financial_analysis(
            sector="specialty_manufacturing",
            tickers=["AAA", "BBB"],
            as_of_date="2026-04-26",
            budget=_budget(max_tool_calls=4, max_turns=4),
        )

    assert [item.code for item in exc_info.value.violations] == [
        "BOUND_FINANCIAL_PROMPT_STATE_MUTATED"
    ]
    assert len(provider.calls) == 1


def test_sector_schema_mutation_after_physical_call_fails_closed(monkeypatch):
    class SchemaMutatingProvider:
        provider_name = "test"

        def __init__(self):
            self.calls: list[dict] = []

        def enabled(self):
            return True

        def synthesize_json(self, **kwargs):
            self.calls.append(kwargs)
            monkeypatch.setitem(
                kwargs["schema"]["properties"]["degraded_states"]["items"],
                "type",
                "integer",
            )
            return SimpleNamespace(json_text=json.dumps(_planning_turn()))

    provider = SchemaMutatingProvider()
    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._run_company_autonomy_pass",
        lambda **kwargs: (False, None, [], [], [], kwargs["executed_tool_count"]),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_autonomous_financial_analysis(
            sector="specialty_manufacturing",
            tickers=["AAA", "BBB"],
            as_of_date="2026-04-26",
            budget=_budget(max_tool_calls=4, max_turns=4),
        )

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_PROVIDER_REQUEST_MUTATED"
    assert len(provider.calls) == 1


def test_sector_mutated_tool_evidence_blocks_budget_recovery_and_fallback(monkeypatch):
    from app.autonomous import sector_runtime as sector_runtime_module

    provider = FakeProvider([_planning_turn(), _final_turn()["final_decision"]])
    real_recovery_prompt = sector_runtime_module._recovery_final_decision_prompt

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} retained its deterministic result.",
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._deterministic_finalization_fallback_decision",
        lambda **_kwargs: pytest.fail(
            "mutated prompt state must not become a deterministic fallback"
        ),
    )

    def mutate_before_recovery(**kwargs):
        kwargs["evidence"][0].summary = "forged recovery evidence after tool production"
        return real_recovery_prompt(**kwargs)

    monkeypatch.setattr(
        "app.autonomous.sector_runtime._recovery_final_decision_prompt",
        mutate_before_recovery,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_autonomous_financial_analysis(
            sector="specialty_manufacturing",
            tickers=["AAA", "BBB"],
            as_of_date="2026-04-26",
            budget=_budget(max_tool_calls=1, max_turns=4),
        )

    assert [item.code for item in exc_info.value.violations] == [
        "BOUND_FINANCIAL_PROMPT_STATE_MUTATED"
    ]
    assert len(provider.calls) == 1


def test_sector_runtime_deterministic_finalization_fallback_upgrades_passed_finalist(monkeypatch):
    provider = FinalTurnFailingProvider()

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=4, max_turns=4),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.no_selection_finalist_audit_attempted is True
    assert artifact.no_selection_finalist_audit_status == "PASS"
    assert artifact.degraded_states == [
        "LLM_PROVIDER_TIMEOUT",
        "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
    ]
    assert provider.calls[1]["schema_name"] == "autonomous_sector_financial_turn"
    assert provider.calls[2]["schema_name"] == "autonomous_sector_final_decision_recovery"
    assert any("Deterministic finalization fallback ran" in note for note in artifact.audit_notes)


def test_sector_runtime_re_raises_financial_drift_during_error_recovery(monkeypatch):
    provider = FinalTurnFailingProvider()

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        },
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.require_financial_integrity_scope",
        _raise_invalid_financial_input_at_recovery,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._deterministic_finalization_fallback_decision",
        lambda **_kwargs: pytest.fail(
            "financial drift at recovery must not become a deterministic fallback"
        ),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_autonomous_financial_analysis(
            sector="specialty_manufacturing",
            tickers=["AAA", "BBB"],
            as_of_date="2026-04-26",
            budget=_budget(max_tool_calls=4, max_turns=4),
        )

    assert exc_info.value.status == "NEEDS_DATA"
    assert [call["schema_name"] for call in provider.calls] == [
        "autonomous_sector_initial_research_plan",
        "autonomous_sector_financial_turn",
    ]


def test_sector_runtime_deterministic_finalization_fallback_blocked_finalist_runs_alternate_audit(
    monkeypatch,
):
    provider = FinalTurnFailingProvider(planning_payload=_alternate_audit_planning_turn())
    packets = [
        _custom_sector_packet("AAA", data_quality_status="NO_FILING", valuation_anchor=130.0),
        _custom_sector_packet("BBB", valuation_anchor=110.0),
    ]
    scenarios = [
        _audit_base_scenario("AAA", 0.30),
        _audit_base_scenario("BBB", 0.20),
        _audit_downside_scenario("BBB", -0.01),
    ]

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_sector_company_financial_packets_from_signal_packets",
        lambda signal_packets, market_cap_focus=None, **kwargs: packets,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_expected_return_scenarios_for_packets",
        lambda company_packets, horizon_years=None: _scenario_map(scenarios),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=4, max_turns=4),
    )

    # Primary finalist AAA has a fetchable NO_FILING gap (DATA_INCOMPLETE ,
    # previously BLOCKED), which still triggers the alternate-finalist audit. The
    # clean alternate BBB passes and is selected.
    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "BBB"
    assert artifact.no_selection_finalist_audit_focus_ticker == "AAA"
    assert artifact.no_selection_finalist_audit_status == "DATA_INCOMPLETE"
    assert artifact.alternate_finalist_audit_attempted is True
    assert artifact.alternate_finalist_audit_status == "RESOLVED_SELECTED"
    assert artifact.alternate_finalist_audit_results[0]["ticker"] == "BBB"
    assert artifact.degraded_states == [
        "LLM_PROVIDER_TIMEOUT",
        "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
    ]


def test_sector_runtime_deterministic_finalization_fallback_preserves_watchlist_only(monkeypatch):
    provider = FinalTurnFailingProvider(
        planning_payload=_planning_turn(include_second_company_tool=False)
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": True,
            "summary": f"{ctx.ticker} {name} evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=4, max_turns=4),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.no_selection_finalist_audit_attempted is True
    assert artifact.no_selection_finalist_audit_status == "PASS"
    assert artifact.no_selection_finalist_resolution_attempted is False
    assert artifact.degraded_states == [
        "LLM_PROVIDER_TIMEOUT",
        "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
    ]


def test_sector_runtime_recovers_final_decision_when_tool_budget_exhausts(monkeypatch):
    provider = FakeProvider([_planning_turn(), _high_confidence_final_turn()["final_decision"]])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=2, max_turns=4),
    )

    # The thin-coverage caps (INSUFFICIENT_*) are EVIDENCE_QUALITY and no
    # longer lower the ceiling; with no business-quality cap present the recovered
    # decision keeps its HIGH confidence rather than the prior LOW.
    assert artifact.final_verdict == "SELECTED"
    assert artifact.selected_ticker == "AAA"
    assert artifact.confidence == "HIGH"
    assert artifact.no_selection_reason is None
    assert artifact.degraded_states == ["BUDGET_EXHAUSTED"]
    assert artifact.selection_audit["status"] == "PASS"
    assert artifact.selection_audit["expected_return_evidence_count"] == 1
    assert artifact.selection_audit["company_specific_evidence_count"] == 1
    assert [call.status for call in artifact.tool_calls] == [
        "OK",
        "OK",
        "SKIPPED_BUDGET_EXHAUSTED",
    ]
    assert len(provider.calls) == 2


def test_sector_runtime_falls_back_when_budget_exhaustion_recovery_fails(monkeypatch):
    provider = FakeProvider([_planning_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker},
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(max_tool_calls=1, max_turns=4),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.no_selection_reason == (
        "LLM provider final decision was unavailable after deterministic evidence collection; "
        "the runtime will attempt deterministic finalist audit instead of forcing a selection. "
        "Initial failure: Tool-call budget exhausted after executing available evidence.; "
        "compact recovery failure: pop from empty list"
    )
    assert artifact.degraded_states == [
        "BUDGET_EXHAUSTED",
        "LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE",
    ]
    assert [call.status for call in artifact.tool_calls] == [
        "OK",
        "SKIPPED_BUDGET_EXHAUSTED",
        "SKIPPED_BUDGET_EXHAUSTED",
    ]
    assert len(provider.calls) == 2


def test_sector_runtime_re_raises_financial_drift_during_budget_recovery(monkeypatch):
    provider = FakeProvider([_planning_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker},
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.require_financial_integrity_scope",
        _raise_invalid_financial_input_at_recovery,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._deterministic_finalization_fallback_decision",
        lambda **_kwargs: pytest.fail(
            "financial drift at budget recovery must not become a deterministic fallback"
        ),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_autonomous_financial_analysis(
            sector="specialty_manufacturing",
            tickers=["AAA", "BBB"],
            as_of_date="2026-04-26",
            budget=_budget(max_tool_calls=1, max_turns=4),
        )

    assert exc_info.value.status == "NEEDS_DATA"
    assert [call["schema_name"] for call in provider.calls] == [
        "autonomous_sector_initial_research_plan"
    ]


def test_sector_runtime_records_disallowed_tools_and_continues(monkeypatch):
    # Pin legacy hard-block hurdle behavior (production default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    disallowed_turn = {
        "framework": _framework_payload(),
        "research_questions": [
            {
                "question_id": "Q1",
                "question": "Can unsupported external data change the decision?",
                "financial_pillar": "Evidence quality",
                "expected_decision_impact": "Should be skipped because the tool is not allowed.",
                "priority": "LOW",
                "target_tickers": ["AAA"],
                "planned_tool_calls": [
                    {
                        "tool_name": "browse_social_media",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Not allowed.",
                    },
                    {
                        "tool_name": "rank_expected_return_cases",
                        "ticker": None,
                        "tool_input": {"horizon_years": 5, "scenario_name": "base"},
                        "rationale": "Allowed scenario evidence.",
                    },
                    {
                        "tool_name": "fetch_kpi_trends",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Allowed company-specific evidence.",
                    },
                    {
                        "tool_name": "analyze_liquidity_stress",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Allowed company-specific risk evidence.",
                    },
                    {
                        "tool_name": "summarize_financial_packets",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Allowed fallback.",
                    },
                ],
            }
        ],
        "belief_updates": [],
        "continue_research": True,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": [],
    }
    provider = FakeProvider([disallowed_turn, _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "KPI evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "SELECTED"
    assert artifact.degraded_states == ["DISALLOWED_TOOL_SKIPPED"]
    assert [call.status for call in artifact.tool_calls] == [
        "SKIPPED_DISALLOWED_TOOL",
        "OK",
        "OK",
        "OK",
        "OK",
    ]
    assert artifact.research_questions[0].status == "ANSWERED"
    assert artifact.selection_audit["status"] == "PASS"


def test_sector_runtime_repairs_missing_companyfacts_line_items(monkeypatch):
    plan = {
        "framework": _framework_payload(),
        "research_questions": [
            {
                "question_id": "Q1",
                "question": "Can the financial history support the sector selection?",
                "financial_pillar": "Financial history",
                "expected_decision_impact": "Determines whether companyfacts evidence can be used.",
                "priority": "HIGH",
                "target_tickers": ["AAA"],
                "planned_tool_calls": [
                    {
                        "tool_name": "rank_expected_return_cases",
                        "ticker": None,
                        "tool_input": {"horizon_years": 5, "scenario_name": "base"},
                        "rationale": "Provider checks scenario evidence before financial history.",
                    },
                    {
                        "tool_name": "fetch_companyfacts_timeseries",
                        "ticker": "AAA",
                        "tool_input": {},
                        "rationale": "Provider omitted required line_items.",
                    },
                ],
            }
        ],
        "belief_updates": [],
        "continue_research": True,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": [],
    }
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["no_selection_reason"] = None
    provider = FakeProvider([plan, no_selection_turn])
    dispatched: list[dict] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(tool_input)
        return {"status": "ok", "ticker": ctx.ticker, "series": {}}

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert dispatched == [
        {
            "line_items": [
                "revenue",
                "operating_income",
                "net_income",
                "cfo",
                "capex",
                "cash",
                "total_debt",
                "shares_outstanding",
            ],
            "years": 5,
        }
    ]
    assert artifact.degraded_states == ["TOOL_INPUT_DEFAULTED"]
    assert artifact.tool_calls[1].tool_input == dispatched[0]
    assert "Tool input guardrail" in artifact.tool_calls[1].rationale
    assert artifact.tool_calls[1].status == "OK"


def test_sector_runtime_translates_companyfacts_metric_aliases(monkeypatch):
    plan = {
        "framework": _framework_payload(),
        "research_questions": [
            {
                "question_id": "Q1",
                "question": "Can SEC-named equity and EPS fields validate the financial history?",
                "financial_pillar": "Financial history",
                "expected_decision_impact": "Determines whether companyfacts aliases are usable.",
                "priority": "HIGH",
                "target_tickers": ["AAA"],
                "planned_tool_calls": [
                    {
                        "tool_name": "rank_expected_return_cases",
                        "ticker": None,
                        "tool_input": {"horizon_years": 5, "scenario_name": "base"},
                        "rationale": "Provider checks scenario evidence before SEC aliases.",
                    },
                    {
                        "tool_name": "fetch_companyfacts_timeseries",
                        "ticker": "AAA",
                        "tool_input": {"metrics": ["StockholdersEquity", "EarningsPerShareBasic"]},
                        "rationale": "Provider used SEC-style metric names.",
                    },
                ],
            }
        ],
        "belief_updates": [],
        "continue_research": True,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": [],
    }
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["no_selection_reason"] = None
    provider = FakeProvider([plan, no_selection_turn])
    dispatched: list[dict] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(tool_input)
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "series": {
                "equity": [{"fiscal_year": 2025, "value": 200.0}],
                "net_income": [{"fiscal_year": 2025, "value": 20.0}],
                "shares_outstanding": [{"fiscal_year": 2025, "value": 10.0}],
            },
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert dispatched == [
        {
            "metrics": ["StockholdersEquity", "EarningsPerShareBasic"],
            "line_items": ["equity", "net_income", "shares_outstanding"],
            "years": 5,
        }
    ]
    assert artifact.degraded_states == ["TOOL_INPUT_REPAIRED"]
    assert "translated companyfacts metric aliases" in artifact.tool_calls[1].rationale


def test_sector_runtime_does_not_substitute_unsupported_peer_metric(monkeypatch):
    plan = {
        "framework": _framework_payload(),
        "research_questions": [
            {
                "question_id": "Q1",
                "question": "Can combined ratio peer comparison validate underwriting quality?",
                "financial_pillar": "Peer comparison",
                "expected_decision_impact": "Should expose unsupported peer metrics honestly.",
                "priority": "HIGH",
                "target_tickers": ["AAA"],
                "planned_tool_calls": [
                    {
                        "tool_name": "compare_peer_metric",
                        "ticker": "AAA",
                        "tool_input": {"metric": "combined_ratio"},
                        "rationale": "Provider requested an unsupported peer metric.",
                    }
                ],
            }
        ],
        "belief_updates": [],
        "continue_research": True,
        "final_decision": None,
        "degraded_states": [],
        "audit_notes": [],
    }
    provider = FakeProvider([plan, _final_turn()])
    dispatched: list[dict] = []

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(tool_input)
        return {
            "status": "unavailable",
            "ticker": ctx.ticker,
            "metric": tool_input["metric"],
            "usable_for_decision": False,
            "summary": "Unsupported metric.",
        }

    monkeypatch.setattr("app.autonomous.sector_runtime.dispatch_alpha_tool", fake_dispatch)

    def fake_child_run(ticker, **kwargs):
        raise RuntimeError("child provider failed")

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis", fake_child_run
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert dispatched == [{"metric": "combined_ratio"}]
    # The child repair fails (PROVIDER_ERROR) and the remaining obstacles are
    # fetchable MISSING_*_EVIDENCE gaps, so the candidate surfaces as
    # DATA_INCOMPLETE (ticker preserved) rather than a hard NO_SELECTION.
    assert artifact.degraded_states == [
        "TOOL_INPUT_UNSUPPORTED",
        "AUDIT_GAP_REPAIR_PROVIDER_ERROR",
        "SELECTION_AUDIT_DATA_INCOMPLETE",
    ]
    assert artifact.audit_gap_repair_attempted is True
    assert artifact.audit_gap_repair_status == "PROVIDER_ERROR"
    assert artifact.evidence[0].confidence == "LOW"
    assert artifact.final_verdict == "DATA_INCOMPLETE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.selection_audit["status"] == "DATA_INCOMPLETE"
    assert "MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE" in artifact.selection_audit["hard_blockers"]


def test_sector_runtime_provider_unavailable_returns_no_selection(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: DisabledProvider()
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.degraded_states == ["LLM_PROVIDER_UNAVAILABLE"]
    assert "without forcing a selection" in artifact.no_selection_reason
    assert len(artifact.company_packets) == 2
    assert len(artifact.expected_return_scenarios) == 12


def test_sector_runtime_quota_exhausted_primary_provider_falls_back_to_anthropic(monkeypatch):
    primary = QuotaProvider()
    fallback = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: primary)
    monkeypatch.setattr("app.autonomous.sector_runtime.get_anthropic_provider", lambda: fallback)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.status == "COMPLETED"
    assert artifact.final_verdict == "SELECTED"
    assert len(primary.calls) == 2
    assert len(fallback.calls) == 2
    assert [call.status for call in artifact.tool_calls] == ["OK", "OK", "OK"]


def test_sector_runtime_converts_out_of_scope_selection_to_no_selection(monkeypatch):
    provider = FakeProvider(
        [
            _planning_turn(),
            {
                **_final_turn(selected_ticker="ZZZ"),
                "framework": _framework_payload(),
            },
        ]
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.degraded_states == ["SELECTED_TICKER_NOT_IN_SCOPE"]
    assert artifact.final_decision is not None
    assert artifact.final_decision.selection_blockers == ["SELECTED_TICKER_NOT_IN_SCOPE"]
    assert artifact.audit_notes[-1] == (
        "Runtime guardrails are binding over provider working notes; "
        "top-level final_verdict and final_decision fields are the source of truth."
    )


def test_sector_runtime_converts_blocked_in_scope_selection_to_no_selection(monkeypatch):
    provider = FakeProvider(
        [
            _planning_turn(),
            {
                **_final_turn(selected_ticker="AAA"),
                "framework": _framework_payload(),
            },
        ]
    )
    packets = _packets()
    packets["AAA"].model_status = "MODEL_BLOCKED"
    packets["AAA"].model_blockers = ["SECURITY_IDENTITY_UNVERIFIED"]

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets", lambda tickers, **_kwargs: packets
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.confidence is None
    assert artifact.degraded_states == ["SELECTED_TICKER_PACKET_BLOCKED"]
    assert artifact.no_selection_reason == (
        "No company selected because AAA has deterministic packet blockers: "
        "SECURITY_IDENTITY_UNVERIFIED, MODEL_FIT_BLOCKED, MODEL_BLOCKED."
    )
    assert artifact.final_decision is not None
    assert artifact.final_decision.selection_blockers == [
        "SELECTED_TICKER_PACKET_BLOCKED",
        "SECURITY_IDENTITY_UNVERIFIED",
        "MODEL_FIT_BLOCKED",
        "MODEL_BLOCKED",
    ]
    assert artifact.audit_notes[-1] == (
        "Runtime guardrails are binding over provider working notes; "
        "top-level final_verdict and final_decision fields are the source of truth."
    )


def test_sector_runtime_converts_selected_candidate_with_provider_blockers_to_no_selection(
    monkeypatch,
):
    final_turn = _final_turn(selected_ticker="AAA")
    final_turn["final_decision"]["selection_blockers"] = [
        "Capital adequacy evidence is missing",
        "Reserve development remains unresolved",
    ]
    provider = FakeProvider(
        [
            _planning_turn(),
            {
                **final_turn,
                "framework": _framework_payload(),
            },
        ]
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.confidence is None
    assert artifact.degraded_states == ["SELECTED_TICKER_SELECTION_BLOCKED"]
    assert artifact.no_selection_reason == (
        "No company selected because AAA has unresolved selection blockers: "
        "Capital adequacy evidence is missing, Reserve development remains unresolved."
    )
    assert artifact.final_decision is not None
    assert artifact.final_decision.selection_blockers == [
        "SELECTED_TICKER_SELECTION_BLOCKED",
        "Capital adequacy evidence is missing",
        "Reserve development remains unresolved",
    ]
    assert artifact.audit_notes[-1] == (
        "Runtime guardrails are binding over provider working notes; "
        "top-level final_verdict and final_decision fields are the source of truth."
    )


def test_sector_runtime_normalizes_no_selection_without_reason(monkeypatch):
    no_selection_turn = _final_turn(selected_ticker=None, verdict="NO_SELECTION")
    no_selection_turn["final_decision"]["confidence"] = "MODERATE"
    no_selection_turn["final_decision"]["no_selection_reason"] = None
    provider = FakeProvider(
        [
            _planning_turn(),
            {
                **no_selection_turn,
                "framework": _framework_payload(),
            },
        ]
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.selected_ticker is None
    assert artifact.confidence is None
    assert artifact.no_selection_reason == "Autonomous sector analyst returned no selection."
    assert artifact.final_decision is not None
    assert artifact.final_decision.confidence is None
    assert (
        artifact.final_decision.no_selection_reason
        == "Autonomous sector analyst returned no selection."
    )
    assert artifact.audit_notes[-1] == (
        "Runtime guardrails are binding over provider working notes; "
        "top-level final_verdict and final_decision fields are the source of truth."
    )


def test_sector_runtime_synthesizes_incomplete_no_selection_from_degraded_states(monkeypatch):
    provider = FakeProvider(
        [
            _planning_turn(),
            {
                "framework": _framework_payload(),
                "research_questions": [],
                "belief_updates": [],
                "continue_research": False,
                "final_decision": {
                    "verdict": "NO_SELECTION",
                    "confidence": None,
                    "selected_ticker": None,
                    "expected_annualized_return_range": None,
                    "thesis": "",
                    "key_risk": "",
                    "downside_case": "",
                    "no_selection_reason": None,
                    "falsifiers": [],
                    "why_selected_over_finalists": [],
                    "rejected_finalists": [],
                    "selection_blockers": [],
                    "confidence_cap_reasons": [],
                    "evidence_ref_ids": [],
                },
                "degraded_states": [
                    "NO_FILING: filings unavailable",
                    "ANCHOR_UNVALIDATED: valuation anchor cannot be validated",
                    "CURRENT_EVENTS_GAP: current events unavailable",
                    "RESERVE_DEVELOPMENT_UNKNOWN",
                    "DILUTION_ROOT_CAUSE_UNKNOWN",
                ],
                "audit_notes": ["Provider stopped without complete structured narrative fields."],
            },
        ]
    )

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Tool evidence is usable.",
        },
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.no_selection_reason == (
        "No company selected because unresolved evidence gaps remained: "
        "NO_FILING, ANCHOR_UNVALIDATED, CURRENT_EVENTS_GAP, RESERVE_DEVELOPMENT_UNKNOWN, plus 1 more."
    )
    assert artifact.final_decision is not None
    assert artifact.final_decision.thesis == "No sector selection was made."
    assert artifact.final_decision.key_risk == artifact.no_selection_reason
    assert artifact.final_decision.downside_case == "No underwritten downside case was finalized."
    assert artifact.degraded_states == [
        "NO_FILING: filings unavailable",
        "ANCHOR_UNVALIDATED: valuation anchor cannot be validated",
        "CURRENT_EVENTS_GAP: current events unavailable",
        "RESERVE_DEVELOPMENT_UNKNOWN",
        "DILUTION_ROOT_CAUSE_UNKNOWN",
        "STRUCTURED_DECISION_INCOMPLETE",
    ]
    assert any(
        "runtime synthesized conservative no-selection text" in note
        for note in artifact.audit_notes
    )
    assert artifact.audit_notes[-1] == (
        "Runtime guardrails are binding over provider working notes; "
        "top-level final_verdict and final_decision fields are the source of truth."
    )


def test_sector_runtime_artifact_round_trips_and_summary_is_compact(monkeypatch):
    # Pin legacy hard-block hurdle behavior (production default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    provider = FakeProvider([_planning_turn(), _final_turn()])

    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _packets(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker},
    )

    artifact = run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )
    restored = AutonomousSectorFinancialRunArtifact.from_dict(artifact.to_dict())
    summary = sector_artifact_summary(restored)

    assert restored == artifact
    assert restored.final_decision_prompt_context == {
        "prompt_scoped_tickers": ["AAA", "BBB"],
        "expected_return_evidence_by_finalist": {
            "AAA": {"evidence_count": 1, "has_repair_target": False},
        },
        "audit_gap_repair_targets": [],
    }
    assert summary["final_verdict"] == "SELECTED"
    assert summary["selected_ticker"] == "AAA"
    assert summary["selection_audit_status"] == "PASS"
    assert summary["actionable"] is True
    assert summary["confidence_ceiling"] == "HIGH"
    assert summary["audit_gap_repair_attempted"] is False
    assert summary["audit_gap_repair_status"] is None
    assert summary["watchlist_resolution_attempted"] is False
    assert summary["watchlist_resolution_status"] is None
    assert summary["no_selection_finalist_audit_attempted"] is False
    assert summary["no_selection_finalist_audit_status"] is None
    assert summary["no_selection_finalist_audit_focus_ticker"] is None
    assert summary["no_selection_finalist_resolution_attempted"] is False
    assert summary["no_selection_finalist_resolution_status"] is None
    assert summary["company_packets"] == 2
    assert summary["expected_return_scenarios"] == 12
    assert summary["research_questions"] == 1
    assert summary["return_cushion_status"] == "CLEAR"
    assert summary["selected_company_evidence_tools"] == [
        "analyze_liquidity_stress",
        "fetch_kpi_trends",
    ]
    assert summary["selected_evidence_pillars"] == ["liquidity", "quality"]
    assert summary["tool_calls"] == 3
    assert summary["valuation_anchor_count"] == 2
    assert summary["generic_valuation_anchor_count"] == 2
    assert summary["sector_specific_valuation_anchor_count"] == 0
    assert summary["invalid_generic_valuation_anchor_count"] == 0
    assert summary["missing_valuation_anchor_count"] == 0
    assert summary["valuation_anchor_method_counts"] == {"dcf": 2}


def test_sector_artifact_summary_uses_canonical_memo_body_cost_rollup():
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_cost_summary_test",
        sector="technology",
        market_cap_focus="mid_cap",
        objective="Summarize memo-body cost.",
        as_of_date="2026-05-10",
        created_at="2026-05-10T20:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        memo_body={
            "status": "LLM_GENERATED",
            "cohort_comparison": {
                "source": "llm",
                "usage": {"input_tokens": 1000, "output_tokens": 100, "cost_estimate_usd": 0.10},
            },
            "triage_surprises": {
                "source": "llm",
                "usage": {"input_tokens": 2000, "output_tokens": 200, "cost_estimate_usd": 0.20},
            },
            "usage": {
                "calls": [
                    {
                        "section": "cohort_comparison",
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "cost_estimate_usd": 0.10,
                    },
                    {
                        "section": "triage_surprises",
                        "input_tokens": 2000,
                        "output_tokens": 200,
                        "cost_estimate_usd": 0.20,
                    },
                    {
                        "section": "candidate",
                        "ticker": "AAA",
                        "input_tokens": 500,
                        "output_tokens": 50,
                        "cost_estimate_usd": 0.05,
                    },
                ],
                "input_tokens": 3500,
                "output_tokens": 350,
                "cost_estimate_usd": 0.35,
            },
        },
    )

    summary = sector_artifact_summary(artifact)

    assert summary["memo_body_usage"]["cost_estimate_usd"] == 0.35
    assert summary["memo_body_usage"]["input_tokens"] == 3500
    assert len(summary["memo_body_usage"]["calls"]) == 3


def test_sector_artifact_summary_falls_back_to_single_pass_memo_usage_sum():
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_cost_fallback_test",
        sector="technology",
        market_cap_focus="mid_cap",
        objective="Summarize memo-body section cost.",
        as_of_date="2026-05-10",
        created_at="2026-05-10T20:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        memo_body={
            "status": "LLM_GENERATED",
            "cohort_comparison": {
                "source": "llm",
                "usage": {"input_tokens": 1000, "output_tokens": 100, "cost_estimate_usd": 0.10},
            },
            "triage_surprises": {
                "source": "llm",
                "usage": {"input_tokens": 2000, "output_tokens": 200, "cost_estimate_usd": 0.20},
            },
            "candidates": {
                "AAA": {
                    "source": "llm",
                    "usage": {"input_tokens": 500, "output_tokens": 50, "cost_estimate_usd": 0.05},
                }
            },
        },
    )

    summary = sector_artifact_summary(artifact)

    assert summary["memo_body_usage"]["cost_estimate_usd"] == 0.35
    assert summary["memo_body_usage"]["input_tokens"] == 3500
    assert summary["memo_body_usage"]["output_tokens"] == 350


def test_sector_artifact_summary_counts_sector_specific_and_invalid_generic_anchors():
    generic_packet = _custom_sector_packet("AAA", valuation_anchor=100.0)
    sector_specific_packet = _custom_sector_packet("INS", valuation_anchor=80.0)
    sector_specific_packet.model_fit_status = "VALID_SECTOR_SPECIFIC"
    sector_specific_packet.valuation = {
        "valuation_anchor": 80.0,
        "anchor_method": "insurance_common",
        "generic_valuation_valid": False,
    }
    technology_packet = _custom_sector_packet("TECH", valuation_anchor=120.0)
    technology_packet.model_fit_status = "VALID_SECTOR_SPECIFIC"
    technology_packet.valuation = {
        "valuation_anchor": 120.0,
        "anchor_method": "technology_adjusted_dcf",
        "generic_anchor_method": "dcf",
        "generic_anchor_value": 100.0,
        "sector_specific_anchor_method": "technology_adjusted_dcf",
        "sector_specific_anchor_value": 120.0,
    }
    invalid_generic_packet = _custom_sector_packet("BAD", valuation_anchor=90.0)
    invalid_generic_packet.model_fit_status = "BLOCKED"
    invalid_generic_packet.valuation = {
        "valuation_anchor": 90.0,
        "anchor_method": "dcf",
        "generic_valuation_valid": False,
    }
    missing_packet = _custom_sector_packet("MISS", valuation_anchor=None)
    missing_packet.valuation = {"valuation_anchor": None, "anchor_method": None}

    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_anchor_summary_test",
        sector="financial_services",
        market_cap_focus="small_cap",
        objective="Count valid valuation anchors.",
        as_of_date="2026-05-03",
        created_at="2026-05-03T20:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        company_packets=[
            generic_packet,
            sector_specific_packet,
            technology_packet,
            invalid_generic_packet,
            missing_packet,
        ],
    )

    summary = sector_artifact_summary(artifact)

    assert summary["company_packets"] == 5
    assert summary["valuation_anchor_count"] == 3
    assert summary["generic_valuation_anchor_count"] == 1
    assert summary["sector_specific_valuation_anchor_count"] == 2
    assert summary["unknown_method_valuation_anchor_count"] == 0
    assert summary["invalid_generic_valuation_anchor_count"] == 1
    assert summary["missing_valuation_anchor_count"] == 1
    assert summary["valuation_anchor_method_counts"] == {
        "dcf": 1,
        "insurance_common": 1,
        "technology_adjusted_dcf": 1,
    }


def test_autonomous_structured_output_schemas_are_openai_strict():
    def assert_strict_objects(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                properties = node.get("properties")
                assert isinstance(properties, dict)
                assert node.get("additionalProperties") is False
                assert node.get("required") == list(properties.keys())
            for value in node.values():
                assert_strict_objects(value)
        elif isinstance(node, list):
            for item in node:
                assert_strict_objects(item)

    assert_strict_objects(_PLAN_SCHEMA)
    assert_strict_objects(_INITIAL_PLAN_SCHEMA)
    assert_strict_objects(_MINIMUM_TOOL_PLAN_SCHEMA)
    assert_strict_objects(_TURN_SCHEMA)


def _v2_projection_artifact(
    *,
    selected_ticker: str | None = None,
    child_runs: list[dict] | None = None,
    candidate_selection: dict | None = None,
    status: str = "COMPLETED",
) -> AutonomousSectorFinancialRunArtifact:
    runs = json.loads(json.dumps(child_runs or []))
    selection = dict(candidate_selection or {})
    gate_results = dict(selection.get("structural_gate_results") or {})
    gate_results.setdefault(
        "AAA",
        {
            "screen_result": {
                "contract_id": "technology",
                "status": "PASS",
                "gate_evaluations": [
                    {
                        "contract_id": "technology",
                        "rule_id": "TEST_SOURCE_BACKED_SCREEN",
                        "status": "PASS",
                        "applicable": True,
                        "observed_value": "PASS",
                        "threshold": "PASS",
                        "evidence_ref_id": "test:screen:AAA",
                        "evidence_url": None,
                        "reason_code": None,
                        "notes": [],
                    }
                ],
                "reason_codes": [],
                "evidence_ref_ids": ["test:screen:AAA"],
            }
        },
    )
    selection["structural_gate_results"] = gate_results
    packet = SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="OK",
        model_fit_status="SUPPORTED",
        data_quality_status="OK",
        current_price=50.0,
        valuation={"valuation_anchor": 75.0},
        score_components={"deterministic_score": 1.0},
    )
    scenario = SectorExpectedReturnScenario(
        scenario_id="AAA-base",
        ticker="AAA",
        scenario_name="base",
        horizon_years=5,
        current_price=50.0,
        estimated_future_value_per_share=75.0,
        annualized_return=0.084,
    )
    reviewed_tickers = [
        "AAA"
        for run in runs
        if str(run.get("ticker") or "").upper() == "AAA"
        and str(run.get("final_verdict") or "").upper() in {"ACTIONABLE", "WATCHLIST_ONLY", "AVOID"}
    ]
    frontier_state = build_competitive_frontier(
        [packet],
        [scenario],
        reviewed_tickers=reviewed_tickers,
    )
    competitive_frontier = frontier_state.to_dict()
    closure = frontier_state.closure_certificate
    competitive_frontier.update(
        {
            "status": closure.status,
            "minimum_reviews_required": 1,
            "successful_review_count": len(frontier_state.reviewed_tickers),
            "attempted_tickers": list(frontier_state.reviewed_tickers),
            "failed_review_tickers": [],
        }
    )
    frontier_candidate_tickers = [
        str(row.get("ticker") or "").upper() for row in competitive_frontier["candidates"]
    ]
    source_bindings = _v2_child_source_bindings(
        sector="enterprise_software",
        as_of_date="2026-07-15",
        company_packets=[packet],
        scenarios=[scenario],
        signal_packets={"AAA": _packets()["AAA"]},
        frontier_candidate_tickers=frontier_candidate_tickers,
    )
    competitive_frontier["source_bindings"] = source_bindings
    competitive_frontier["signal_packet_snapshots"] = {
        "AAA": canonical_v2_signal_packet_snapshot(_packets()["AAA"]),
    }
    competitive_frontier["cohort_fingerprint"] = source_bindings["AAA"]["cohort_fingerprint"]
    for run in runs:
        ticker = str(run.get("ticker") or "").strip().upper()
        binding = source_bindings.get(ticker)
        if (
            binding is None
            or str(run.get("status") or "").upper() != "COMPLETED"
            or str(run.get("final_verdict") or "").upper()
            not in {"ACTIONABLE", "WATCHLIST_ONLY", "AVOID"}
        ):
            continue
        run["source_binding"] = dict(binding)
        raw_artifact = run.get("artifact")
        nested_artifacts = []
        if isinstance(raw_artifact, dict):
            nested_artifacts.append(raw_artifact)
        nested_artifacts.extend(
            attempt for attempt in run.get("attempts") or [] if isinstance(attempt, dict)
        )
        for nested_artifact in nested_artifacts:
            request = dict(nested_artifact.get("request") or {})
            request.setdefault("run_id", run.get("run_id"))
            request["as_of_date"] = "2026-07-15"
            candidate_scope = dict(request.get("candidate_scope") or {})
            candidate_scope["mode"] = "single_candidate"
            candidate_scope["tickers"] = [ticker]
            candidate_scope["source_binding"] = dict(binding)
            candidate_scope["signal_packet_snapshot"] = json.loads(
                json.dumps(competitive_frontier["signal_packet_snapshots"][ticker])
            )
            request["candidate_scope"] = candidate_scope
            nested_artifact["request"] = request
        if isinstance(raw_artifact, dict) and not run.get("attempts"):
            run["attempts"] = [json.loads(json.dumps(raw_artifact))]
    decision = SectorFinalDecision(
        verdict="SELECTED" if selected_ticker else "NO_SELECTION",
        confidence="MODERATE" if selected_ticker else None,
        selected_ticker=selected_ticker,
        expected_annualized_return_range="12%-18%" if selected_ticker else None,
        thesis="AAA has a supportable return case." if selected_ticker else "No selection.",
        key_risk="Execution risk.",
        downside_case="Downside remains material.",
        no_selection_reason=None if selected_ticker else "No candidate cleared the bar.",
    )
    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_v2_projection",
        sector="enterprise_software",
        market_cap_focus="large_and_mega",
        objective="Rank the sector.",
        as_of_date="2026-07-15",
        created_at="2026-07-16T00:00:00Z",
        status=status,
        final_verdict=decision.verdict,
        selected_ticker=selected_ticker,
        confidence=decision.confidence,
        company_packets=[packet],
        expected_return_scenarios=[scenario],
        candidate_selection=selection,
        final_decision=decision,
        relative_ranking=[
            {
                "ticker": "AAA",
                "audit_status": "PASS",
                "actionable": True,
                "company_autonomy_verdict": "ACTIONABLE",
                "hard_blockers": [],
            }
        ],
        company_autonomy_runs=runs,
        competitive_frontier=competitive_frontier,
    )


def _v2_evidenced_child(
    verdict: str,
    *,
    ticker: str = "AAA",
    confidence: str = "HIGH",
    degraded_states: list[str] | None = None,
) -> dict:
    run_id = f"child-{ticker.lower()}-{verdict.lower()}"
    return {
        "run_id": run_id,
        "ticker": ticker,
        "status": "COMPLETED",
        "final_verdict": verdict,
        "confidence": confidence,
        "degraded_states": list(degraded_states or []),
        "tool_calls": 1,
        "evidence_references": 1,
        "artifact": {
            "tool_calls": [{"call_id": "TC1", "status": "OK"}],
            "evidence": [
                {
                    "evidence_id": "E1",
                    "confidence": "HIGH",
                    "tool_call_id": "TC1",
                }
            ],
            "candidate_decisions": [
                {
                    "ticker": ticker,
                    "verdict": verdict,
                    "evidence_ref_ids": ["E1"],
                }
            ],
        },
    }


def test_v2_projection_preserves_admitted_sparse_name_and_clears_synthetic_actionability():
    artifact = _v2_projection_artifact(
        candidate_selection={
            "financial_history_filter": {"excluded_tickers": ["BBB"]},
            "selected_tickers": ["AAA"],
        }
    )

    projected = _finalize_sector_artifact_v2(
        artifact,
        admitted_tickers=["AAA", "BBB"],
    )

    assert projected.pipeline_version == "v2"
    assert projected.admitted_tickers == ("AAA", "BBB")
    dispositions = {item.ticker: item for item in projected.candidate_dispositions}
    assert dispositions["AAA"].terminal_state == "READY_FOR_UNDERWRITING"
    assert dispositions["BBB"].terminal_state == "NEEDS_DATA"
    assert dispositions["BBB"].reason_codes == ["SPARSE_FINANCIAL_HISTORY"]
    assert projected.relative_ranking[0]["company_autonomy_verdict"] is None
    assert projected.relative_ranking[0]["actionable"] is False
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_keeps_retained_sparse_packet_needs_data_despite_child_verdict():
    artifact = _v2_projection_artifact(
        child_runs=[_v2_evidenced_child("ACTIONABLE")],
        candidate_selection={
            "financial_history_filter": {
                "status": "V2_RETAINED_SPARSE_FINANCIAL_HISTORY",
                "needs_data_tickers": ["AAA"],
            }
        },
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.review_status == "NOT_STARTED"
    assert disposition.reason_codes == ["SPARSE_FINANCIAL_HISTORY"]
    assert disposition.frontier_status == "UNRESOLVED"
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_clears_retained_sparse_gap_after_ready_data_repair():
    artifact = _v2_projection_artifact(
        child_runs=[_v2_evidenced_child("AVOID")],
        candidate_selection={
            "financial_history_filter": {
                "status": "V2_RETAINED_SPARSE_FINANCIAL_HISTORY",
                "needs_data_tickers": ["AAA"],
            },
            "data_gap_repair": {
                "examined": 1,
                "candidate_states": [
                    {
                        "ticker": "AAA",
                        "queue_status": "COMPLETED",
                        "packet_readiness_outcome": "READY",
                        "last_completed_stage": "PACKET",
                        "stage_states": {},
                    }
                ],
            },
        },
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "UNDERWRITTEN"
    assert disposition.underwriting_verdict == "AVOID"
    assert projected.decision_status == "COMPLETE"
    assert projected.final_verdict == "NO_SELECTION"


@pytest.mark.parametrize(
    "reason",
    ["IFRS_FACTS_UNSUPPORTED", "NON_USD_FACTS_UNNORMALIZED"],
)
def test_v2_projection_marks_foreign_normalized_facts_gap_needs_data(reason):
    artifact = _v2_projection_artifact()
    artifact.company_packets[0].accounting_quality = {
        "filing_risk_signals": {"foreign_facts_gap_reason": reason}
    }
    source_binding = _v2_child_source_bindings(
        sector=artifact.sector,
        as_of_date=artifact.as_of_date,
        company_packets=artifact.company_packets,
        scenarios=artifact.expected_return_scenarios,
        signal_packets={"AAA": _packets()["AAA"]},
        frontier_candidate_tickers=["AAA"],
    )["AAA"]
    artifact.competitive_frontier["source_bindings"] = {"AAA": source_binding}
    artifact.competitive_frontier["cohort_fingerprint"] = source_binding["cohort_fingerprint"]

    projected = _finalize_sector_artifact_v2(
        artifact,
        admitted_tickers=["AAA"],
    )

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.screen_status == "PASS"
    assert disposition.review_status == "NOT_STARTED"
    assert disposition.reason_codes == [reason]
    assert disposition.last_completed_stage == "FACTS_AVAILABILITY"
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_preserves_screen_result_while_repair_inputs_block_underwriting():
    artifact = _v2_projection_artifact(
        selected_ticker="AAA",
        child_runs=[
            {
                "ticker": "AAA",
                "status": "COMPLETED",
                "final_verdict": "ACTIONABLE",
                "confidence": "HIGH",
            }
        ],
        candidate_selection={
            "data_gap_repair": {
                "status": "COMPLETED",
                "examined": 1,
                "candidate_states": [
                    {
                        "ticker": "AAA",
                        "queue_status": "COMPLETED",
                        "last_completed_stage": "PACKET",
                        "packet_readiness_outcome": "NEEDS_DATA",
                        "packet_missing_inputs": ["CAP", "FILING"],
                        "stage_states": {},
                    }
                ],
            }
        },
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.screen_status == "PASS"
    assert disposition.review_status == "NOT_STARTED"
    assert disposition.reason_codes == [
        "MISSING_MARKET_CAP",
        "NO_READABLE_ANNUAL_FILING",
    ]
    assert disposition.last_completed_stage == "PACKET"
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_preserves_filing_identity_and_parser_gap_reasons():
    artifact = _v2_projection_artifact(
        candidate_selection={
            "data_gap_repair": {
                "status": "COMPLETED",
                "examined": 1,
                "candidate_states": [
                    {
                        "ticker": "AAA",
                        "queue_status": "COMPLETED",
                        "last_completed_stage": "PACKET",
                        "packet_readiness_outcome": "NEEDS_DATA",
                        "packet_missing_inputs": ["FILING"],
                        "stage_states": {
                            "FILINGS": {"reason_code": "ISSUER_CIK_CONFLICT"},
                            "PARSING": {"reason_code": "ANNUAL_FILING_RECORD_MISSING"},
                        },
                    }
                ],
            }
        },
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    assert projected.candidate_dispositions[0].reason_codes == [
        "ISSUER_CIK_CONFLICT",
        "ANNUAL_FILING_RECORD_MISSING",
    ]


def test_v2_projection_keeps_max_candidate_deferred_names_in_scope():
    artifact = _v2_projection_artifact(
        candidate_selection={
            "loaded_tickers": ["AAA", "BBB"],
            "selected_tickers": ["AAA", "BBB"],
            "membership_tickers": ["AAA", "BBB"],
            "execution_tickers": ["AAA"],
            "deferred_by_bound_tickers": ["BBB"],
            "excluded_tickers": [],
            "execution_bound": 1,
            "membership_fingerprint": (
                "518d1d0ec11d9c4a85cbc86de03771a3cc2b7c35eb40e6b70c08b0c736552490"
            ),
            "execution_fingerprint": (
                "48e6f6a04a3a679b4e2bdb382af448541fff6a9ae6b13a5ea16146559adaa3f5"
            ),
            "execution_bound_frozen": True,
            "ranking_basis": "v2_discovery_membership_order_pending_repaired_rank",
        }
    )

    projected = _finalize_sector_artifact_v2(
        artifact,
        admitted_tickers=["AAA"],
    )

    dispositions = {item.ticker: item for item in projected.candidate_dispositions}
    assert projected.admitted_tickers == ("AAA", "BBB")
    assert dispositions["BBB"].scope_status == "IN_SCOPE"
    assert dispositions["BBB"].terminal_state == "DEFERRED_BY_BOUND"
    assert dispositions["BBB"].screen_status == "NOT_RUN"
    assert dispositions["BBB"].review_status == "NOT_REQUIRED"
    assert dispositions["BBB"].frontier_status == "NOT_ELIGIBLE"
    assert dispositions["BBB"].reason_codes == ["DEFERRED_BY_EXECUTION_BOUND"]


def test_v2_zero_admitted_sector_is_complete_with_out_of_scope_dispositions():
    artifact = run_sector_autonomous_financial_analysis(
        sector="education_services",
        tickers=[],
        as_of_date="2026-07-15",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        candidate_selection={
            "loaded_tickers": [],
            "selected_tickers": [],
            "membership_tickers": [],
            "execution_tickers": [],
            "deferred_by_bound_tickers": [],
            "execution_bound": None,
            "membership_fingerprint": (
                "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
            ),
            "execution_fingerprint": (
                "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
            ),
            "execution_bound_frozen": True,
            "execution_as_of_date": "2026-07-15",
            "cap_classifications": {
                "LOW": {
                    "market_cap_mm": 5_000.0,
                    "reason_code": "BELOW_LARGE_CAP_THRESHOLD",
                }
            },
        },
    )

    assert artifact.pipeline_version == "v2"
    assert artifact.status == "COMPLETED"
    assert artifact.execution_status == "COMPLETED"
    assert artifact.decision_status == "COMPLETE"
    assert artifact.final_verdict == "NO_SELECTION"
    assert artifact.admitted_tickers == ()
    assert len(artifact.candidate_dispositions) == 1
    disposition = artifact.candidate_dispositions[0]
    assert disposition.ticker == "LOW"
    assert disposition.terminal_state == "OUT_OF_SCOPE"
    assert disposition.reason_codes == ["BELOW_LARGE_CAP_THRESHOLD"]


def test_v2_runtime_refuses_tickers_beyond_frozen_execution_bound(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "app.autonomous.evidence_resolution.pre_assembly_data_gap_repair",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("repair must not start after execution-set drift")
        ),
    )
    selection = {
        "selected_tickers": ["AAA", "BBB"],
        "membership_tickers": ["AAA", "BBB"],
        "execution_tickers": ["AAA"],
        "deferred_by_bound_tickers": ["BBB"],
        "excluded_tickers": [],
        "execution_bound": 1,
        "membership_fingerprint": (
            "518d1d0ec11d9c4a85cbc86de03771a3cc2b7c35eb40e6b70c08b0c736552490"
        ),
        "execution_fingerprint": (
            "48e6f6a04a3a679b4e2bdb382af448541fff6a9ae6b13a5ea16146559adaa3f5"
        ),
        "execution_bound_frozen": True,
        "execution_as_of_date": "2026-07-15",
    }

    with pytest.raises(
        ValueError,
        match="runtime tickers do not match the frozen execution set",
    ):
        run_sector_autonomous_financial_analysis(
            sector="energy",
            tickers=["AAA", "BBB"],
            as_of_date="2026-07-15",
            market_cap_focus="large_and_mega",
            pipeline_version="v2",
            candidate_selection=selection,
            budget=AutonomousRunBudget(
                max_tool_calls=8,
                max_turns=1,
                max_cost_usd=None,
                timebox_seconds=None,
                max_candidates=1,
            ),
        )


def test_v2_projection_preserves_source_backed_structural_screen_out():
    artifact = _v2_projection_artifact(
        candidate_selection={
            "loaded_tickers": ["AAA", "BBB"],
            "selected_tickers": ["AAA"],
            "excluded_tickers": ["BBB"],
            "structural_gate_results": {
                "BBB": {
                    "ticker": "BBB",
                    "as_of_date": "2026-07-15",
                    "quarantined": True,
                    "excluded_error": False,
                    "triggered_codes": ["DELISTING_NOTICE"],
                    "reasons": ["QUARANTINE_STRUCTURAL:DELISTING_NOTICE"],
                    "details": {"filing_date": "2026-07-10"},
                    "screen_result": {
                        "contract_id": "technology",
                        "status": "FAIL",
                        "gate_evaluations": [
                            {
                                "contract_id": "technology",
                                "rule_id": "DELISTING_NOTICE",
                                "status": "FAIL",
                                "applicable": True,
                                "observed_value": "8-K item 3.01 filed 2026-07-10",
                                "threshold": "no active delisting notice",
                                "evidence_ref_id": "sec:BBB:8-k:2026-07-10",
                                "evidence_url": "https://www.sec.gov/Archives/bbb-8k.htm",
                                "reason_code": "QUARANTINE_STRUCTURAL:DELISTING_NOTICE",
                                "notes": [],
                            }
                        ],
                        "reason_codes": ["QUARANTINE_STRUCTURAL:DELISTING_NOTICE"],
                        "evidence_ref_ids": ["sec:BBB:8-k:2026-07-10"],
                    },
                }
            },
        }
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = {item.ticker: item for item in projected.candidate_dispositions}["BBB"]
    assert disposition.scope_status == "IN_SCOPE"
    assert disposition.terminal_state == "SCREENED_OUT"
    assert disposition.reason_codes == ["QUARANTINE_STRUCTURAL:DELISTING_NOTICE"]


def test_v2_delta_requires_evidence_before_accepting_carried_underwriting():
    artifact = _v2_projection_artifact(
        candidate_selection={
            "loaded_tickers": ["AAA", "BBB", "CCC"],
            "selected_tickers": ["AAA"],
            "excluded_tickers": ["BBB", "CCC"],
            "delta_audit": {
                "swept_excluded": ["CCC"],
                "carried_verdicts": {
                    "BBB": {
                        "verdict": "WATCHLIST_ONLY",
                        "run_id": "autonomous_sector_prior_bbb",
                    }
                },
                "to_review": ["AAA"],
            },
        }
    )

    projected = _finalize_sector_artifact_v2(
        artifact,
        admitted_tickers=["AAA"],
    )

    dispositions = {item.ticker: item for item in projected.candidate_dispositions}
    assert projected.admitted_tickers == ("AAA", "BBB")
    assert dispositions["AAA"].scope_status == "IN_SCOPE"
    assert dispositions["BBB"].terminal_state == "NEEDS_DATA"
    assert dispositions["BBB"].underwriting_verdict is None
    assert dispositions["BBB"].reason_codes == [
        "CARRIED_UNDERWRITING_EVIDENCE_UNAVAILABLE",
        "SOURCE_RUN:autonomous_sector_prior_bbb",
    ]
    assert dispositions["CCC"].terminal_state == "OUT_OF_SCOPE"
    assert dispositions["CCC"].reason_codes == ["DELTA_ALREADY_SWEPT_OUTSIDE_CURRENT_ATTEMPT"]


def test_v2_delta_rejects_carried_no_winner_as_completed_underwriting():
    artifact = _v2_projection_artifact(
        candidate_selection={
            "loaded_tickers": ["AAA"],
            "selected_tickers": ["AAA"],
            "delta_audit": {
                "carried_verdicts": {
                    "AAA": {
                        "verdict": "NO_WINNER",
                        "run_id": "autonomous_sector_prior_aaa",
                        "underwriting_result": {
                            "status": "COMPLETED",
                            "verdict": "NO_WINNER",
                            "confidence": "MODERATE",
                            "evidence_ref_ids": ["prior:E1"],
                            "tool_call_ids": ["TC1"],
                            "child_run_id": "autonomous_sector_prior_aaa",
                        },
                    }
                }
            },
        }
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.underwriting_verdict is None
    assert disposition.reason_codes == [
        "CARRIED_UNDERWRITING_EVIDENCE_UNAVAILABLE",
        "SOURCE_RUN:autonomous_sector_prior_aaa",
    ]
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_caps_unvalidated_actionable_underwriting():
    artifact = _v2_projection_artifact(
        selected_ticker="AAA",
        child_runs=[_v2_evidenced_child("ACTIONABLE")],
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "UNDERWRITTEN"
    assert disposition.underwriting_verdict == "ACTIONABLE"
    assert projected.selection_validation.status == "NOT_ATTEMPTED"
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None
    assert projected.selected_ticker is None


def test_v2_projection_treats_completed_child_without_verdict_as_needs_data():
    artifact = _v2_projection_artifact(
        child_runs=[{"ticker": "AAA", "status": "COMPLETED"}],
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.review_status == "INCOMPLETE"
    assert disposition.reason_codes == [
        "UNDERWRITING_TOOL_OUTPUT_MISSING",
        "UNDERWRITING_DECISION_EVIDENCE_MISSING",
        "UNDERWRITING_VERDICT_MISSING",
        "UNDERWRITING_RUN_ID_MISSING",
    ]
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_treats_no_winner_as_incomplete_underwriting():
    artifact = _v2_projection_artifact(
        child_runs=[_v2_evidenced_child("NO_WINNER")],
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.review_status == "INCOMPLETE"
    assert disposition.underwriting_verdict is None
    assert disposition.reason_codes == ["UNDERWRITING_NO_DECISION"]
    assert disposition.underwriting_result is not None
    assert disposition.underwriting_result.status == "INCOMPLETE"
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_preserves_budget_failure_on_no_winner_child():
    child = _v2_evidenced_child(
        "NO_WINNER",
        degraded_states=["BUDGET_EXHAUSTED"],
    )
    artifact = _v2_projection_artifact(child_runs=[child])

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.reason_codes == [
        "UNDERWRITING_RUNTIME_INCOMPLETE",
        "UNDERWRITING_NO_DECISION",
    ]
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_rejects_avoid_without_decision_linked_evidence():
    child = _v2_evidenced_child("AVOID")
    child["artifact"]["candidate_decisions"][0]["evidence_ref_ids"] = []
    artifact = _v2_projection_artifact(child_runs=[child])

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.underwriting_verdict is None
    assert disposition.reason_codes == ["UNDERWRITING_DECISION_EVIDENCE_MISSING"]
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_rejects_evidence_from_failed_child_tool_call():
    child = _v2_evidenced_child("AVOID")
    child["artifact"]["tool_calls"] = [
        {"call_id": "TC1", "status": "ERROR"},
        {"call_id": "TC2", "status": "OK"},
    ]
    artifact = _v2_projection_artifact(child_runs=[child])

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.underwriting_verdict is None
    assert disposition.reason_codes == ["UNDERWRITING_DECISION_EVIDENCE_MISSING"]
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_allows_complete_no_selection_only_after_evidenced_avoid():
    artifact = _v2_projection_artifact(
        child_runs=[_v2_evidenced_child("AVOID")],
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "UNDERWRITTEN"
    assert disposition.review_status == "COMPLETED"
    assert disposition.underwriting_verdict == "AVOID"
    assert disposition.underwriting_result is not None
    assert disposition.underwriting_result.status == "COMPLETED"
    assert disposition.evidence_ref_ids == [
        "test:screen:AAA",
        "child-aaa-avoid:E1",
    ]
    assert projected.decision_status == "COMPLETE"
    assert projected.final_verdict == "NO_SELECTION"


@pytest.mark.parametrize(
    ("frontier_mutation", "expected_reason"),
    [
        ("MISSING", "COMPETITIVE_FRONTIER_MISSING"),
        ("OPEN", "COMPETITIVE_FRONTIER_OPEN"),
        ("FORGED", "COMPETITIVE_FRONTIER_INVALID"),
    ],
)
def test_v2_projection_cannot_complete_with_missing_open_or_forged_frontier(
    frontier_mutation,
    expected_reason,
):
    artifact = _v2_projection_artifact(
        child_runs=[_v2_evidenced_child("AVOID")],
    )
    if frontier_mutation == "MISSING":
        artifact.competitive_frontier = {}
    elif frontier_mutation == "OPEN":
        artifact.competitive_frontier["status"] = "OPEN"
    else:
        artifact.competitive_frontier["frontier_tickers"] = []

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    if frontier_mutation == "MISSING":
        assert disposition.terminal_state == "NEEDS_DATA"
        assert disposition.review_status == "INCOMPLETE"
        assert disposition.underwriting_verdict is None
        assert disposition.reason_codes == ["UNDERWRITING_FRONTIER_SOURCE_BINDING_MISSING"]
    else:
        assert disposition.terminal_state == "UNDERWRITTEN"
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None
    assert expected_reason in projected.selection_audit["v2_decision_failure_reasons"]


def test_v2_projection_rejects_coherent_frontier_metrics_not_bound_to_packets():
    artifact = _v2_projection_artifact(
        child_runs=[_v2_evidenced_child("AVOID")],
    )
    forged_packet = _frontier_packet("AAA", 99.0)
    forged_scenario = _frontier_scenario("AAA", 0.90)
    forged_state = build_competitive_frontier(
        [forged_packet],
        [forged_scenario],
        reviewed_tickers=["AAA"],
    )
    artifact.competitive_frontier = forged_state.to_dict()
    artifact.competitive_frontier.update(
        {
            "status": "CLOSED",
            "minimum_reviews_required": 1,
            "successful_review_count": 1,
            "attempted_tickers": ["AAA"],
            "failed_review_tickers": [],
        }
    )

    with pytest.raises(
        ValueError,
        match="v2 competitive frontier requires candidates and source bindings",
    ):
        _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])


@pytest.mark.parametrize(
    ("child", "expected_review_status", "expected_reasons", "expected_stage"),
    [
        (
            {"ticker": "AAA", "status": "FAILED"},
            "FAILED",
            [
                "UNDERWRITING_FAILED",
                "UNDERWRITING_TOOL_OUTPUT_MISSING",
                "UNDERWRITING_DECISION_EVIDENCE_MISSING",
                "UNDERWRITING_VERDICT_MISSING",
                "UNDERWRITING_RUN_ID_MISSING",
            ],
            "UNDERWRITING",
        ),
        (
            _v2_evidenced_child("DATA_INCOMPLETE"),
            "INCOMPLETE",
            ["UNDERWRITING_DATA_INCOMPLETE"],
            "UNDERWRITING",
        ),
    ],
)
def test_v2_projection_preserves_passed_screen_when_underwriting_is_incomplete(
    child, expected_review_status, expected_reasons, expected_stage
):
    artifact = _v2_projection_artifact(child_runs=[child])

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "NEEDS_DATA"
    assert disposition.screen_status == "PASS"
    assert disposition.review_status == expected_review_status
    assert disposition.reason_codes == expected_reasons
    assert disposition.last_completed_stage == expected_stage
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


def test_v2_projection_does_not_let_actionable_child_override_failed_screen():
    artifact = _v2_projection_artifact(
        selected_ticker="AAA",
        child_runs=[_v2_evidenced_child("ACTIONABLE")],
        candidate_selection={
            "structural_gate_results": {
                "AAA": {
                    "screen_result": {
                        "contract_id": "technology",
                        "status": "FAIL",
                        "gate_evaluations": [
                            {
                                "contract_id": "technology",
                                "rule_id": "DELISTING_NOTICE",
                                "status": "FAIL",
                                "applicable": True,
                                "observed_value": "8-K item 3.01",
                                "threshold": "no active delisting notice",
                                "evidence_ref_id": "sec:AAA:8-k:2026-07-10",
                                "evidence_url": ("https://www.sec.gov/Archives/aaa-8k.htm"),
                                "reason_code": ("QUARANTINE_STRUCTURAL:DELISTING_NOTICE"),
                                "notes": [],
                            }
                        ],
                        "reason_codes": ["QUARANTINE_STRUCTURAL:DELISTING_NOTICE"],
                        "evidence_ref_ids": ["sec:AAA:8-k:2026-07-10"],
                    }
                }
            }
        },
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    disposition = projected.candidate_dispositions[0]
    assert disposition.terminal_state == "SCREENED_OUT"
    assert disposition.review_status == "NOT_REQUIRED"
    assert disposition.underwriting_verdict is None
    assert disposition.underwriting_result is not None
    assert disposition.underwriting_result.status == "NOT_REQUIRED"
    assert projected.decision_status == "COMPLETE"
    assert projected.final_verdict == "NO_SELECTION"


def test_v2_projection_does_not_publish_no_selection_over_actionable_child():
    artifact = _v2_projection_artifact(
        selected_ticker=None,
        child_runs=[_v2_evidenced_child("ACTIONABLE")],
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    assert projected.candidate_dispositions[0].terminal_state == "UNDERWRITTEN"
    assert projected.candidate_dispositions[0].underwriting_verdict == "ACTIONABLE"
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None


@pytest.mark.parametrize(
    "failure_state",
    ["LLM_PROVIDER_FINAL_DECISION_UNAVAILABLE", "BUDGET_EXHAUSTED"],
)
def test_v2_projection_does_not_publish_no_selection_from_failed_run_state(
    failure_state,
):
    artifact = _v2_projection_artifact(
        selected_ticker=None,
        child_runs=[_v2_evidenced_child("AVOID")],
    )
    artifact.degraded_states = [failure_state]

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    assert projected.candidate_dispositions[0].underwriting_verdict == "AVOID"
    assert projected.decision_status == "INCOMPLETE"
    assert projected.final_verdict is None
    assert projected.selection_audit["v2_decision_failure_reasons"] == [failure_state]


def test_v2_projection_publishes_only_matching_validated_actionable_selection():
    artifact = _v2_projection_artifact(
        selected_ticker="AAA",
        child_runs=[_v2_evidenced_child("ACTIONABLE")],
    )
    source_binding = _v2_child_source_bindings(
        sector=artifact.sector,
        as_of_date=artifact.as_of_date,
        company_packets=artifact.company_packets,
        scenarios=artifact.expected_return_scenarios,
        signal_packets={"AAA": _packets()["AAA"]},
    )["AAA"]
    artifact.competitive_frontier["source_bindings"] = {"AAA": source_binding}
    validator_run_id = "validator_aaa"
    validation_evidence_id = f"{validator_run_id}:E-VALIDATION"
    validation_tool_id = f"{validator_run_id}:VTC1"
    artifact.selection_validation = SectorSelectionValidation(
        status="VALIDATED",
        selected_ticker="AAA",
        validator_run_id=validator_run_id,
        validator_verdict="ACTIONABLE",
        evidence_ref_ids=[validation_evidence_id],
        evidence=[
            EvidenceReference(
                evidence_id=validation_evidence_id,
                source_type="tool_output",
                source_label="challenge",
                summary="Independent validation evidence.",
                ticker="AAA",
                tool_call_id=validation_tool_id,
                confidence="HIGH",
            )
        ],
        tool_calls=[
            ToolCallRecord(
                call_id=validation_tool_id,
                tool_name="challenge_selected_company",
                tool_input={"ticker": "AAA"},
                rationale="Challenge the provisional selection.",
                status="OK",
                evidence_ref_ids=[validation_evidence_id],
                lane="selected_company_validation",
            )
        ],
        provider_usage=[
            {
                "provider_call_id": f"{validator_run_id}:P1",
                "validator_run_id": validator_run_id,
                "lane": "selected_company_validation",
                "provider": "openai",
                "model": "gpt-5.5",
                "schema_name": "autonomous_candidate_decision",
                "status": "OK",
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 25,
                "estimated_tokens": False,
                "cost_estimate_usd": 0.00125,
            }
        ],
        source_binding=source_binding,
    )

    projected = _finalize_sector_artifact_v2(artifact, admitted_tickers=["AAA"])

    assert projected.execution_status == "COMPLETED"
    assert projected.decision_status == "COMPLETE"
    assert projected.final_verdict == "SELECTED"
    assert projected.selected_ticker == "AAA"


def test_v2_memo_enrichment_budget_failure_preserves_execution_truth(monkeypatch):
    projected = _finalize_sector_artifact_v2(
        _v2_projection_artifact(),
        admitted_tickers=["AAA"],
    )

    def fail_enrichment(*args, **kwargs):
        raise LLMCostBudgetExceeded("memo budget exhausted")

    monkeypatch.setattr(
        "app.autonomous.sector_runtime._enrich_sector_artifact_memo_body_impl",
        fail_enrichment,
    )

    enriched = enrich_sector_artifact_memo_body(projected)

    assert enriched.status == "COMPLETED"
    assert enriched.execution_status == "COMPLETED"
    assert enriched.decision_status == "INCOMPLETE"
    assert enriched.final_verdict is None
    assert "MEMO_BODY_LLM_COST_BUDGET_EXCEEDED" in enriched.degraded_states


def test_v1_memo_cost_exhaustion_persists_partial_retryable_artifact():
    from tests.test_autonomous_cli import _sector_artifact

    class BudgetExhaustedProvider:
        provider_name = "openai"

        def __init__(self):
            self.calls = 0

        def enabled(self):
            return True

        def synthesize_json(self, **kwargs):
            self.calls += 1
            raise LLMCostBudgetExceeded("memo budget exhausted")

    artifact = _sector_artifact()
    artifact.pipeline_version = "v1"
    artifact.company_packets = [
        SectorCompanyFinancialPacket(
            ticker="AAA",
            financial_status="Financially Viable",
            model_fit_status="VALID_GENERIC",
            data_quality_status="OK",
        )
    ]
    provider = BudgetExhaustedProvider()

    enriched = enrich_sector_artifact_memo_body(artifact, provider=provider)

    assert enriched.status == "COMPLETED"
    assert provider.calls == 1
    assert enriched.memo_body["candidates"]["AAA"]["status"] == "DEGRADED_STATE"
    assert "LLM_COST_BUDGET_EXCEEDED" in enriched.memo_body["candidates"]["AAA"]["degraded_state"]


def _completed_v2_child_artifact(
    ticker: str,
    *,
    verdict: str = "ACTIONABLE",
) -> AutonomousRunArtifact:
    artifact = _child_company_artifact(ticker, final_verdict=verdict)
    artifact.candidate_decisions = [
        CandidateDecision(
            ticker=ticker,
            verdict=verdict,
            confidence="MODERATE",
            thesis="Evidence-linked test underwriting.",
            key_risk="Execution risk.",
            eligible_for_selection=verdict == "ACTIONABLE",
            selection_blockers=[],
            evidence_ref_ids=[item.evidence_id for item in artifact.evidence],
        )
    ]
    artifact.provider_usage = [
        {
            "provider_call_id": "P1",
            "lane": "company_underwriting",
            "provider": "openai",
            "model": "gpt-5.5",
            "schema_name": "autonomous_candidate_decision",
            "status": "OK",
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 25,
            "estimated_tokens": False,
            "cost_estimate_usd": 0.00125,
        }
    ]
    return artifact


def _v2_validation_source_inputs(
    sector: str,
) -> tuple[
    list[SectorCompanyFinancialPacket],
    list[SectorExpectedReturnScenario],
    dict[str, TickerSignalPacket],
    dict[str, object],
]:
    company_packets = [_audit_packet("AAA")]
    scenarios = [_audit_base_scenario("AAA")]
    signal_packets = {"AAA": _packets()["AAA"]}
    source_binding = _v2_child_source_bindings(
        sector=sector,
        as_of_date="2026-07-15",
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
    )["AAA"]
    return company_packets, scenarios, signal_packets, source_binding


def test_v2_selected_company_challenge_is_separate_evidence_linked_and_lane_tagged(
    monkeypatch,
):
    calls: list[dict] = []
    company_packets, scenarios, signal_packets, source_binding = _v2_validation_source_inputs(
        "enterprise_software"
    )

    def fake_child(ticker, **kwargs):
        calls.append({"ticker": ticker, **kwargs})
        return _completed_v2_child_artifact(ticker)

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        fake_child,
    )

    validation = _run_v2_selected_company_validation(
        sector="enterprise_software",
        ticker="AAA",
        as_of_date="2026-07-15",
        allowed_tools=["fetch_kpi_trends", "analyze_dilution"],
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        expected_source_binding=source_binding,
    )

    assert len(calls) == 1
    assert calls[0]["execution_lane"] == "selected_company_validation"
    assert calls[0]["initial_budget"].max_tool_calls == 4
    assert calls[0]["initial_budget"].max_turns == 2
    assert calls[0]["budget"].max_tool_calls == 8
    assert calls[0]["budget"].max_turns == 3
    assert calls[0]["canonical_signal_packet"] is signal_packets["AAA"]
    assert calls[0]["source_binding"] == source_binding
    assert "Independently challenge" in calls[0]["objective"]
    assert validation.status == "VALIDATED"
    assert validation.selected_ticker == "AAA"
    assert validation.validator_verdict == "ACTIONABLE"
    assert validation.evidence_ref_ids == [
        "autonomous_AAA_child:attempt1:CE1",
        "autonomous_AAA_child:attempt1:CE2",
    ]
    assert [item.evidence_id for item in validation.evidence] == validation.evidence_ref_ids
    assert all(call.lane == "selected_company_validation" for call in validation.tool_calls)
    assert all(":attempt1:" in call.call_id for call in validation.tool_calls)
    assert validation.provider_usage[0]["lane"] == "selected_company_validation"
    assert validation.source_binding == source_binding


def test_v2_selected_company_challenge_contradiction_cannot_validate(monkeypatch):
    company_packets, scenarios, signal_packets, source_binding = _v2_validation_source_inputs(
        "energy"
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: _completed_v2_child_artifact(
            ticker,
            verdict="WATCHLIST_ONLY",
        ),
    )

    validation = _run_v2_selected_company_validation(
        sector="energy",
        ticker="AAA",
        as_of_date="2026-07-15",
        allowed_tools=["fetch_kpi_trends"],
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        expected_source_binding=source_binding,
    )

    assert validation.status == "CONTRADICTED"
    assert validation.validator_verdict == "WATCHLIST_ONLY"
    assert validation.reason_codes == ["SELECTED_COMPANY_CHALLENGE_WATCHLIST_ONLY"]


def test_v2_selected_company_validation_refuses_canonical_packet_drift(monkeypatch):
    company_packets, scenarios, signal_packets, source_binding = _v2_validation_source_inputs(
        "energy"
    )
    signal_packets["AAA"].current_price = 51.0
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("drifted canonical packet must not reach the paid child")
        ),
    )

    validation = _run_v2_selected_company_validation(
        sector="energy",
        ticker="AAA",
        as_of_date="2026-07-15",
        allowed_tools=["fetch_kpi_trends"],
        company_packets=company_packets,
        scenarios=scenarios,
        signal_packets=signal_packets,
        expected_source_binding=source_binding,
    )

    assert validation.status == "INCOMPLETE"
    assert validation.reason_codes == ["SELECTED_COMPANY_CANONICAL_SOURCE_BINDING_DRIFT"]
    assert validation.source_binding != source_binding


def test_v2_company_child_autoextends_from_four_two_to_eight_three(monkeypatch):
    budgets: list[tuple[int, int, int, int, str]] = []
    _, _, signal_packets, source_binding = _v2_validation_source_inputs("energy")

    def fake_child(ticker, **kwargs):
        budgets.append(
            (
                kwargs["budget"].max_tool_calls,
                kwargs["budget"].max_turns,
                kwargs["initial_budget"].max_tool_calls,
                kwargs["initial_budget"].max_turns,
                kwargs["execution_lane"],
            )
        )
        child = _completed_v2_child_artifact(ticker)
        child.audit_notes.append(
            "Progressive autonomous budget extended from 4 tools/2 turns to 8 tools/3 turns because decision-relevant evidence remained unresolved."
        )
        return child

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        fake_child,
    )

    run, artifact, notes = _run_v2_company_underwriting_child(
        sector="energy",
        ticker="AAA",
        as_of_date="2026-07-15",
        allowed_tools=["fetch_kpi_trends"],
        canonical_signal_packet=signal_packets["AAA"],
        source_binding=source_binding,
    )

    assert budgets == [
        (8, 3, 4, 2, "company_underwriting"),
    ]
    assert artifact is not None
    assert run["extension_attempted"] is True
    assert len(run["attempts"]) == 1
    assert run["tool_call_attempts"] == 2
    assert run["final_verdict"] == "ACTIONABLE"
    assert notes == [
        "Extended AAA company underwriting to eight tools/three turns because required evidence remained unresolved."
    ]


def test_v2_failed_company_child_preserves_paid_provider_usage(monkeypatch):
    _, _, signal_packets, source_binding = _v2_validation_source_inputs("energy")
    error = RuntimeError("child synthesis failed after a paid plan")
    attach_provider_usage_to_exception(
        error,
        [
            {
                "status": "OK",
                "lane": "company_underwriting",
                "provider": "openai",
                "model": "gpt-5.5",
                "schema_name": "child_plan",
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 20,
                "estimated_tokens": False,
                "cost_estimate_usd": 0.0011,
            }
        ],
    )

    def fail_child(*args, **kwargs):
        _ = (args, kwargs)
        raise error

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        fail_child,
    )

    run, artifact, notes = _run_v2_company_underwriting_child(
        sector="energy",
        ticker="AAA",
        as_of_date="2026-07-15",
        allowed_tools=["fetch_kpi_trends"],
        canonical_signal_packet=signal_packets["AAA"],
        source_binding=source_binding,
    )

    assert artifact is None
    assert run["status"] == "FAILED"
    assert run["tool_call_attempts"] == 0
    assert len(run["provider_usage"]) == 1
    assert run["provider_usage"][0]["schema_name"] == "child_plan"
    assert notes == [
        "Company underwriting attempt for AAA failed: RuntimeError: "
        "child synthesis failed after a paid plan"
    ]


def _frontier_packet(ticker: str, score: float) -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="OK",
        model_fit_status="SUPPORTED",
        data_quality_status="OK",
        current_price=10.0,
        valuation={"valuation_anchor": 20.0},
        score_components={"deterministic_score": score},
    )


def _frontier_scenario(ticker: str, annualized_return: float) -> SectorExpectedReturnScenario:
    return SectorExpectedReturnScenario(
        scenario_id=f"{ticker}-base",
        ticker=ticker,
        scenario_name="base",
        horizon_years=5,
        current_price=10.0,
        estimated_future_value_per_share=20.0,
        annualized_return=annualized_return,
    )


def test_v2_frontier_reviews_three_then_continues_every_live_nondominated_name(
    monkeypatch,
):
    reviewed: list[str] = []

    def fake_underwriting(**kwargs):
        ticker = kwargs["ticker"]
        reviewed.append(ticker)
        artifact = _completed_v2_child_artifact(ticker)
        return (
            {
                **_v2_evidenced_child("ACTIONABLE", ticker=ticker),
                "run_id": artifact.request.run_id,
                "artifact": artifact.to_dict(),
            },
            artifact,
            [],
        )

    monkeypatch.setattr(
        "app.autonomous.sector_runtime._run_v2_company_underwriting_child",
        fake_underwriting,
    )
    packets = [
        _frontier_packet("AAA", 10.0),
        _frontier_packet("BBB", 9.0),
        _frontier_packet("CCC", 8.0),
        _frontier_packet("DDD", 7.0),
    ]
    scenarios = [
        _frontier_scenario("AAA", 0.10),
        _frontier_scenario("BBB", 0.09),
        _frontier_scenario("CCC", 0.08),
        _frontier_scenario("DDD", 0.20),
    ]
    gate_results = {packet.ticker: {"screen_result": {"status": "PASS"}} for packet in packets}
    signal_packets = {
        packet.ticker: _signal_packet(packet.ticker, 20.0, 10.0, 0.05) for packet in packets
    }

    payload, runs, evidence, notes = _run_v2_competitive_frontier(
        sector="energy",
        as_of_date="2026-07-15",
        company_packets=packets,
        scenarios=scenarios,
        candidate_selection={"structural_gate_results": gate_results},
        signal_packets=signal_packets,
        allowed_tools=["fetch_kpi_trends", "analyze_dilution"],
        evidence_start_index=1,
    )

    assert reviewed == ["AAA", "BBB", "CCC", "DDD"]
    assert len(runs) == 4
    assert len(evidence) == 8
    assert payload["minimum_reviews_required"] == 3
    assert payload["successful_review_count"] == 4
    assert payload["frontier_tickers"] == ["AAA", "DDD"]
    assert payload["closure_certificate"]["pending_tickers"] == []
    assert payload["status"] == "CLOSED"
    assert set(payload["source_bindings"]) == {"AAA", "BBB", "CCC", "DDD"}
    assert notes[-1].startswith("Competitive frontier CLOSED")


def test_v2_frontier_refuses_packet_outside_frozen_execution_set(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._run_v2_company_underwriting_child",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unauthorized child must not start")),
    )
    packets = [_frontier_packet("AAA", 10.0), _frontier_packet("BBB", 9.0)]
    scenarios = [
        _frontier_scenario("AAA", 0.10),
        _frontier_scenario("BBB", 0.09),
    ]
    selection = {
        "selected_tickers": ["AAA"],
        "membership_tickers": ["AAA"],
        "execution_tickers": ["AAA"],
        "deferred_by_bound_tickers": [],
        "excluded_tickers": [],
        "execution_bound": 1,
        "membership_fingerprint": (
            "48e6f6a04a3a679b4e2bdb382af448541fff6a9ae6b13a5ea16146559adaa3f5"
        ),
        "execution_fingerprint": (
            "48e6f6a04a3a679b4e2bdb382af448541fff6a9ae6b13a5ea16146559adaa3f5"
        ),
        "execution_bound_frozen": True,
        "structural_gate_results": {
            "AAA": {"screen_result": {"status": "PASS"}},
        },
    }

    with pytest.raises(
        ValueError,
        match="frontier input widened beyond the frozen execution set: BBB",
    ):
        _run_v2_competitive_frontier(
            sector="energy",
            as_of_date="2026-07-15",
            company_packets=packets,
            scenarios=scenarios,
            candidate_selection=selection,
            signal_packets={
                ticker: _signal_packet(ticker, 20.0, 10.0, 0.05) for ticker in ("AAA", "BBB")
            },
            allowed_tools=["fetch_kpi_trends"],
            evidence_start_index=1,
        )


def test_v2_frontier_missing_packet_binding_fails_closed_before_child_start(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._run_v2_company_underwriting_child",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("missing canonical binding must not start a child")
        ),
    )
    packets = [_frontier_packet("AAA", 10.0)]
    scenarios = [_frontier_scenario("AAA", 0.10)]

    with pytest.raises(
        ValueError,
        match=("v2 source binding requires one persisted signal snapshot per company packet"),
    ):
        _run_v2_competitive_frontier(
            sector="energy",
            as_of_date="2026-07-15",
            company_packets=packets,
            scenarios=scenarios,
            candidate_selection={
                "structural_gate_results": {"AAA": {"screen_result": {"status": "PASS"}}}
            },
            signal_packets={},
            allowed_tools=["fetch_kpi_trends"],
            evidence_start_index=1,
        )


def test_v2_final_frontier_rebuild_exposes_new_nondominated_packet():
    artifact = _v2_projection_artifact(
        child_runs=[_v2_evidenced_child("AVOID")],
    )
    artifact.company_packets.append(_frontier_packet("BBB", 2.0))
    artifact.expected_return_scenarios.append(_frontier_scenario("BBB", 0.20))
    artifact.candidate_selection["structural_gate_results"]["BBB"] = {
        "screen_result": {"status": "PASS"}
    }

    _rebuild_v2_competitive_frontier_before_finalization(artifact)

    assert [row["ticker"] for row in artifact.competitive_frontier["candidates"]] == ["BBB", "AAA"]
    assert artifact.competitive_frontier["reviewed_tickers"] == ["AAA"]
    assert artifact.competitive_frontier["frontier_tickers"] == ["BBB"]
    assert artifact.competitive_frontier["pending_tickers"] == ["BBB"]
    assert artifact.competitive_frontier["status"] == "OPEN"
    assert artifact.competitive_frontier["source_binding"] == (
        "FINAL_ARTIFACT_COMPANY_PACKETS_AND_EXPECTED_RETURN_SCENARIOS"
    )
    assert artifact.candidate_selection["competitive_frontier"] == (artifact.competitive_frontier)


def test_v2_frontier_binding_separates_full_packet_cohort_from_eligible_order():
    packets = [
        _frontier_packet("AAA", 10.0),
        _frontier_packet("BBB", 9.0),
    ]
    scenarios = [
        _frontier_scenario("AAA", 0.10),
        _frontier_scenario("BBB", 0.09),
    ]
    signal_packets = {
        packet.ticker: _signal_packet(packet.ticker, 20.0, 10.0, 0.05) for packet in packets
    }

    payload, runs, evidence, notes = _run_v2_competitive_frontier(
        sector="energy",
        as_of_date="2026-07-15",
        company_packets=packets,
        scenarios=scenarios,
        candidate_selection={
            "structural_gate_results": {
                "AAA": {"screen_result": {"status": "PASS"}},
                "BBB": {"screen_result": {"status": "FAIL"}},
            }
        },
        signal_packets=signal_packets,
        allowed_tools=[],
        evidence_start_index=1,
    )

    assert [row["ticker"] for row in payload["candidates"]] == ["AAA"]
    assert set(payload["source_bindings"]) == {"AAA"}
    binding = payload["source_bindings"]["AAA"]
    assert binding["cohort_tickers"] == ["AAA", "BBB"]
    assert binding["frontier_candidate_tickers"] == ["AAA"]
    assert runs == []
    assert evidence == []
    assert any("No allowed company-underwriting tools" in note for note in notes)


def test_v2_operational_lane_budget_has_independent_envelopes_and_no_sector_dollar_cap():
    payload = _v2_lane_budget_payload(
        run_budget=AutonomousRunBudget(16, 6, None, None),
        competitive_frontier={"candidates": [{"ticker": "AAA"}, {"ticker": "BBB"}]},
    )

    assert payload["dollar_ceiling_usd"] is None
    assert payload["whole_run_cost_preflight_required"] is True
    assert payload["lanes"]["parent_research"] == {
        "scope": "per_sector",
        "max_tool_calls": 16,
        "max_turns": 6,
    }
    assert payload["lanes"]["company_underwriting"]["initial_child"] == {
        "profile": "initial",
        "max_tool_calls": 4,
        "max_turns": 2,
    }
    assert payload["lanes"]["company_underwriting"]["extended_child"] == {
        "profile": "extended",
        "max_tool_calls": 8,
        "max_turns": 3,
    }
    assert payload["lanes"]["company_underwriting"]["max_tool_calls"] == 16
    assert payload["lanes"]["selected_company_validation"]["max_turns"] == 3


def test_v2_legacy_repair_child_is_disabled_and_cannot_observe_global_latest(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("v2 legacy repair must not assemble or observe a global latest packet")
        ),
    )

    with provider_usage_capture("parent_research") as captured_usage:
        result = _run_finalist_child_research_pass(
            mode="audit_gap_repair",
            sector="enterprise_software",
            market_cap_focus="large_and_mega",
            objective="Find the strongest evidence-backed candidate.",
            as_of_date="2026-07-15",
            selected_ticker="AAA",
            selection_audit={"status": "WATCHLIST_ONLY"},
            allowed_tools=["fetch_kpi_trends", "analyze_dilution"],
            budget=_budget(max_tool_calls=16, max_turns=6),
            executed_tool_count=0,
            question_start_index=1,
            belief_update_start_index=1,
            call_start_index=1,
            evidence_start_index=1,
            separate_lane_budget=True,
            pipeline_version="v2",
        )

    assert result[0:5] == ([], [], [], [], [])
    assert result[6] == 0
    assert result[7]["attempted"] is False
    assert result[7]["status"] == "DISABLED_V2_CANONICAL_FRONTIER"
    assert captured_usage == []


def test_v1_repair_child_keeps_legacy_success_only_budget_accounting(monkeypatch):
    child = _child_company_artifact("AAA")
    for call in child.tool_calls:
        call.status = "ERROR"
        call.error = "fixture failure"

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        lambda ticker, **kwargs: child,
    )

    result = _run_finalist_child_research_pass(
        mode="audit_gap_repair",
        sector="enterprise_software",
        market_cap_focus="large_and_mega",
        objective="Preserve the v1 repair contract.",
        as_of_date="2026-07-15",
        selected_ticker="AAA",
        selection_audit={"status": "WATCHLIST_ONLY"},
        allowed_tools=["fetch_kpi_trends", "analyze_dilution"],
        budget=_budget(max_tool_calls=4, max_turns=2),
        executed_tool_count=0,
        question_start_index=1,
        belief_update_start_index=1,
        call_start_index=1,
        evidence_start_index=1,
        separate_lane_budget=False,
    )

    assert result[6] == 0
    assert result[7]["status"] == "NO_REPAIR_TOOLS"


def test_v1_repair_child_uses_canonical_parent_packet_binding(monkeypatch):
    signal_packets = _packets()
    company_packets = [_custom_sector_packet("AAA")]
    scenarios = [_audit_base_scenario("AAA", 0.18)]
    captured: dict = {}

    def fake_child(ticker, **kwargs):
        captured.update(kwargs)
        return _child_company_artifact(ticker)

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_single_candidate_autonomous_analysis",
        fake_child,
    )

    _run_finalist_child_research_pass(
        mode="audit_gap_repair",
        sector="enterprise_software",
        market_cap_focus="large_and_mega",
        objective="Bind repair to the parent packet.",
        as_of_date="2026-07-15",
        selected_ticker="AAA",
        selection_audit={"status": "WATCHLIST_ONLY"},
        signal_packets=signal_packets,
        company_packets=company_packets,
        scenarios=scenarios,
        allowed_tools=["fetch_kpi_trends", "analyze_dilution"],
        budget=_budget(max_tool_calls=4, max_turns=2),
        executed_tool_count=0,
        question_start_index=1,
        belief_update_start_index=1,
        call_start_index=1,
        evidence_start_index=1,
        separate_lane_budget=False,
    )

    assert captured["canonical_signal_packet"] is signal_packets["AAA"]
    assert captured["source_binding"]["artifact_type"] == ("v1_canonical_child_source_binding_v1")
    assert captured["source_binding"]["ticker"] == "AAA"
    assert captured["source_binding"]["as_of_date"] == "2026-07-15"
    assert len(captured["source_binding"]["cohort_fingerprint"]) == 64
    assert captured["source_binding"]["financial_integrity_packet"] == (
        company_packets[0].to_dict()
    )
    assert captured["source_binding"]["financial_integrity_scenarios"] == [scenarios[0].to_dict()]


def test_v2_lane_usage_reconciles_parent_child_validation_terminal_and_failures():
    artifact = _v2_projection_artifact()
    artifact.tool_calls = [
        ToolCallRecord("P1", "parent", {}, "parent", "OK", lane="parent_research"),
        ToolCallRecord("R1", "repair", {}, "repair", "ERROR", lane="repair_fallback"),
    ]
    artifact.provider_usage = [
        {
            "status": "OK",
            "lane": "parent_research",
            "input_tokens": 100,
            "cached_input_tokens": 20,
            "output_tokens": 10,
            "cost_estimate_usd": 0.1,
        },
        {
            "status": "INCOMPLETE",
            "lane": "parent_research",
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "cost_estimate_usd": 0,
        },
    ]
    child = _completed_v2_child_artifact("AAA")
    child.provider_usage[0].update(
        {
            "input_tokens": 200,
            "cached_input_tokens": 40,
            "output_tokens": 20,
            "cost_estimate_usd": 0.2,
        }
    )
    artifact.company_autonomy_runs = [
        {
            "ticker": "AAA",
            "artifact": child.to_dict(),
            "attempts": [child.to_dict()],
            "provider_usage": child.provider_usage,
        }
    ]
    artifact.company_autonomy_runs[0]["attempts"][0]["tool_calls"].extend(
        [
            {
                "call_id": "NP1",
                "status": "PLANNED",
                "lane": "company_underwriting",
                "cost_estimate_usd": 99.0,
            },
            {
                "call_id": "NP2",
                "status": "SKIPPED_BUDGET_EXHAUSTED",
                "lane": "company_underwriting",
                "cost_estimate_usd": 99.0,
            },
        ]
    )
    artifact.selection_validation = SectorSelectionValidation(
        status="INCOMPLETE",
        selected_ticker="AAA",
        tool_calls=[
            ToolCallRecord(
                "V1", "challenge", {}, "challenge", "OK", lane="selected_company_validation"
            ),
            ToolCallRecord(
                "V2", "challenge", {}, "challenge", "ERROR", lane="selected_company_validation"
            ),
        ],
        provider_usage=[
            {
                "status": "ERROR",
                "lane": "selected_company_validation",
                "input_tokens": 300,
                "cached_input_tokens": 60,
                "output_tokens": 0,
                "reserved_output_tokens": 30,
                "cost_estimate_usd": 0.3,
            }
        ],
    )
    artifact.candidate_selection = {
        "data_gap_repair": {
            "terminal_cap_search_usage_records": [
                {
                    "call_type": "responses_model",
                    "attempt_status": "RESOLVED",
                    "input_tokens": 400,
                    "cached_input_tokens": 80,
                    "output_tokens": 40,
                    "cost_estimate_usd": 0.4,
                },
                {
                    "call_type": "web_search_call",
                    "attempt_status": "RESOLVED",
                    "cost_estimate_usd": 0.01,
                },
                {
                    "call_type": "web_search_call_reserve",
                    "attempt_status": "PLANNED",
                    "cost_estimate_usd": 9.99,
                },
            ]
        }
    }

    usage = _v2_lane_usage_payload(artifact)

    assert usage["aggregate_reconciles"] is True
    assert usage["aggregate"] == {
        "tool_call_attempts": 7,
        "tool_calls_ok": 5,
        "tool_calls_failed": 2,
        "provider_call_attempts": 5,
        "provider_calls_ok": 4,
        "provider_calls_failed": 1,
        "input_tokens": 1_000,
        "cached_input_tokens": 200,
        "output_tokens": 70,
        "reserved_output_tokens": 30,
        "cost_microdollars": 1_010_000,
        "cost_estimate_usd": "1.010000",
    }
    assert usage["lanes"]["company_underwriting"]["tool_call_attempts"] == 2
    assert usage["lanes"]["selected_company_validation"]["provider_calls_failed"] == 1
    assert usage["lanes"]["terminal_cap_search"]["cost_estimate_usd"] == "0.410000"
    assert usage["nonphysical_tool_diagnostics"]["aggregate"] == {
        "record_count": 3,
        "status_counts": {
            "PLANNED": 1,
            "RESERVED_NOT_DISPATCHED": 1,
            "SKIPPED_BUDGET_EXHAUSTED": 1,
        },
        "reserved_cost_exposure_microdollars": 9_990_000,
    }
    assert usage["nonphysical_tool_diagnostics"]["lanes"]["company_underwriting"] == {
        "record_count": 2,
        "status_counts": {
            "PLANNED": 1,
            "SKIPPED_BUDGET_EXHAUSTED": 1,
        },
        "reserved_cost_exposure_microdollars": 0,
    }
    assert usage["nonphysical_tool_diagnostics"]["lanes"]["terminal_cap_search"] == {
        "record_count": 1,
        "status_counts": {"RESERVED_NOT_DISPATCHED": 1},
        "reserved_cost_exposure_microdollars": 9_990_000,
    }


def test_v2_memo_usage_merge_refreshes_parent_lane_totals():
    artifact = _finalize_sector_artifact_v2(
        _v2_projection_artifact(),
        admitted_tickers=["AAA"],
    )

    _persist_memo_provider_usage(
        artifact,
        [
            {
                "status": "OK",
                "lane": "parent_research",
                "provider": "openai",
                "model": "gpt-5.5",
                "schema_name": "autonomous_sector_candidate_memo_aaa",
                "input_tokens": 300,
                "cached_input_tokens": 50,
                "output_tokens": 60,
                "cost_estimate_usd": 0.00305,
            }
        ],
    )

    assert artifact.provider_usage[0]["provider_call_id"] == "P1"
    assert artifact.lane_usage["lanes"]["parent_research"]["provider_call_attempts"] == 1
    assert artifact.lane_usage["lanes"]["parent_research"]["input_tokens"] == 300
    assert artifact.lane_usage["aggregate"]["cost_estimate_usd"] == "0.003050"
