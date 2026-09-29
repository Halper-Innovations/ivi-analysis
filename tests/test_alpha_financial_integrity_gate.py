from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app.alpha.llm_runtime import (
    AlphaInvestigationConfig,
    AlphaToolContext,
    decide_alpha_winner,
    plan_alpha_investigations,
    run_candidate_investigation,
)
from app.alpha.publication import prepare_alpha_publication
from app.alpha.schemas import SectorAlphaReport, TickerSignalPacket
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
    stable_quote_hash,
    validate_financial_integrity_scope,
)
from app.cli import app
from app.db import get_db, init_db, utc_now_iso
from tests.financial_integrity_helpers import canonicalize_financial_packet


class _CountingProvider:
    provider_name = "openai"

    def __init__(self) -> None:
        self.calls = 0

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **kwargs):
        self.calls += 1
        raise AssertionError(f"provider must be suppressed: {kwargs}")


def _valid_packet(
    ticker: str = "AAA",
    *,
    as_of_date: str = "2026-04-19",
) -> TickerSignalPacket:
    price = 100.0
    snapshot_id = stable_quote_hash(
        ticker=ticker,
        price=price,
        as_of_date=as_of_date,
        currency="USD",
        source="test_quote",
        price_basis="UNADJUSTED",
        raw_price=price,
        split_adjustment_factor=1.0,
    )
    packet = TickerSignalPacket(
        ticker=ticker,
        dcf_value=150.0,
        epv_value=120.0,
        current_price=price,
        current_price_unit="USD_per_share",
        current_price_as_of_date=as_of_date,
        current_price_currency="USD",
        current_price_source="test_quote",
        quote_snapshot_id=snapshot_id,
        price_basis="UNADJUSTED",
        raw_price=price,
        split_adjustment_factor=1.0,
        split_lineage_proof={
            "status": "PASS",
            "period_start": as_of_date,
            "period_end": as_of_date,
            "verified_as_of": as_of_date,
            "source": "fixture_corporate_actions_ledger",
            "source_reference": f"fixture://corporate-actions/{ticker}",
        },
        market_cap_mm=1000.0,
        market_cap_unit="USD_millions",
        market_cap_source="derived_from_quote_and_shares",
        market_cap_effective_as_of_date=as_of_date,
        market_cap_method="price_times_shares",
        shares_outstanding_mm=10.0,
        raw_shares_outstanding_mm=10.0,
        shares_unit="shares_millions",
        shares_basis="UNADJUSTED",
        shares_as_of_date=as_of_date,
        shares_source="test_filing",
        issuer_quote_ratio=1.0,
        cap_stage_price=price,
        cap_stage_price_as_of_date=as_of_date,
        cap_stage_price_currency="USD",
        cap_stage_price_source="test_quote",
        cap_stage_quote_snapshot_id=snapshot_id,
        gate_verdict="PROCEED",
        confidence_class="HIGH",
        moat_score=4,
        moat_classification="MODERATE_MOAT",
    )
    return canonicalize_financial_packet(packet, as_of_date=as_of_date, shares_mm=10.0)


def _valid_scope(packet: TickerSignalPacket) -> FinancialIntegrityScope:
    return FinancialIntegrityScope(
        context="alpha_test",
        run_as_of_date=str(packet.current_price_as_of_date),
        packets=(packet,),
    )


def _invalid_scope() -> FinancialIntegrityScope:
    return FinancialIntegrityScope(
        context="alpha_test_invalid",
        run_as_of_date="2026-04-19",
        packets=(TickerSignalPacket(ticker="AAA", current_price=100.0),),
    )


@pytest.mark.parametrize("integrity_scope", [None, _invalid_scope()])
def test_alpha_planner_missing_or_invalid_scope_suppresses_provider(
    monkeypatch,
    integrity_scope,
):
    provider = _CountingProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    with pytest.raises(InvalidFinancialInputError):
        plan_alpha_investigations(
            sector="software",
            prior_candidates=[{"ticker": "AAA", "hard_block_reasons": []}],
            hard_blocked_candidates=[],
            max_targets=1,
            integrity_scope=integrity_scope,
        )

    assert provider.calls == 0


def test_alpha_planner_schema_mutation_after_call_fails_closed(monkeypatch):
    packet = _valid_packet()

    class _SchemaMutatingProvider:
        provider_name = "openai"

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, **kwargs):
            self.calls += 1
            monkeypatch.setitem(
                kwargs["schema"]["properties"]["summary"],
                "type",
                "integer",
            )
            return SimpleNamespace(
                json_text=('{"summary":"review","selected_targets":[],"skipped_candidates":[]}')
            )

    provider = _SchemaMutatingProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    with pytest.raises(InvalidFinancialInputError):
        plan_alpha_investigations(
            sector="software",
            prior_candidates=[{"ticker": "AAA", "hard_block_reasons": []}],
            hard_blocked_candidates=[],
            max_targets=1,
            integrity_scope=_valid_scope(packet),
        )

    assert provider.calls == 1


def test_alpha_planner_schema_mutation_before_call_has_zero_provider_calls(
    monkeypatch,
):
    packet = _valid_packet()
    provider = _CountingProvider()

    @contextmanager
    def _mutating_attempt_guard(_guard):
        from app.alpha import llm_runtime

        monkeypatch.setitem(
            llm_runtime._PLANNER_SCHEMA["properties"]["summary"],
            "type",
            "integer",
        )
        yield

    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.alpha.llm_runtime.llm_physical_attempt_guard",
        _mutating_attempt_guard,
    )

    with pytest.raises(InvalidFinancialInputError):
        plan_alpha_investigations(
            sector="software",
            prior_candidates=[{"ticker": "AAA", "hard_block_reasons": []}],
            hard_blocked_candidates=[],
            max_targets=1,
            integrity_scope=_valid_scope(packet),
        )

    assert provider.calls == 0


def test_alpha_planner_model_mutation_after_call_fails_closed(monkeypatch):
    packet = _valid_packet()

    class _ModelMutatingProvider:
        provider_name = "openai"

        def __init__(self):
            self.calls = 0
            self.cfg = SimpleNamespace(
                openai_model="gpt-test-a",
                openai_max_output_tokens=4000,
                openai_request_timeout=30.0,
            )

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            self.cfg.openai_model = "gpt-test-b"
            return SimpleNamespace(
                json_text=('{"summary":"review","selected_targets":[],"skipped_candidates":[]}')
            )

    provider = _ModelMutatingProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    with pytest.raises(InvalidFinancialInputError):
        plan_alpha_investigations(
            sector="software",
            prior_candidates=[{"ticker": "AAA", "hard_block_reasons": []}],
            hard_blocked_candidates=[],
            max_targets=1,
            integrity_scope=_valid_scope(packet),
        )

    assert provider.calls == 1


def test_alpha_planner_zero_budget_suppresses_provider(monkeypatch):
    packet = _valid_packet()
    provider = _CountingProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    result = plan_alpha_investigations(
        sector="software",
        prior_candidates=[{"ticker": "AAA", "hard_block_reasons": []}],
        hard_blocked_candidates=[],
        max_targets=1,
        integrity_scope=_valid_scope(packet),
        max_cost_usd=0.0,
    )

    assert provider.calls == 0
    assert result["planner_mode"] == "fallback"
    assert result["physical_calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["budget_usd"] == 0.0
    assert result["budget_remaining_usd"] == 0.0


def test_alpha_candidate_summary_invalid_scope_has_no_substantive_fallback(monkeypatch):
    provider = _CountingProvider()
    packet = _valid_packet()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps",
        lambda evidence_gaps: [],
    )

    with pytest.raises(InvalidFinancialInputError):
        run_candidate_investigation(
            ctx=AlphaToolContext(
                sector="software",
                ticker="AAA",
                packet=packet,
                as_of_date="2026-04-19",
            ),
            investigation_request={"ticker": "AAA", "evidence_gaps": []},
            hard_block_reasons=[],
            integrity_scope=_invalid_scope(),
        )

    assert provider.calls == 0


def test_alpha_candidate_summary_rejects_detached_packet_from_valid_scope(monkeypatch):
    provider = _CountingProvider()
    scoped_packet = _valid_packet("AAA")
    prompt_packet = _valid_packet("AAA")
    prompt_packet.dcf_value = 999.0
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps",
        lambda evidence_gaps: [],
    )

    with pytest.raises(InvalidFinancialInputError) as caught:
        run_candidate_investigation(
            ctx=AlphaToolContext(
                sector="software",
                ticker="AAA",
                packet=prompt_packet,
                as_of_date="2026-04-19",
            ),
            investigation_request={"ticker": "AAA", "evidence_gaps": []},
            hard_block_reasons=[],
            integrity_scope=_valid_scope(scoped_packet),
        )

    assert provider.calls == 0
    assert caught.value.violations[0].code == "BOUND_FINANCIAL_INPUT_MUTATED"


def test_alpha_final_decision_invalid_scope_suppresses_provider(monkeypatch):
    provider = _CountingProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    with pytest.raises(InvalidFinancialInputError):
        decide_alpha_winner(
            sector="software",
            prior_ranking=[{"ticker": "AAA", "consensus_rank": 1}],
            investigation_plan={"selected_targets": [{"ticker": "AAA"}]},
            candidate_investigations=[{"ticker": "AAA", "eligible_for_selection": True}],
            hard_blocked_candidates=[],
            integrity_scope=_invalid_scope(),
        )

    assert provider.calls == 0


def test_alpha_final_decision_tiny_budget_suppresses_provider(monkeypatch):
    packet = _valid_packet()
    provider = _CountingProvider()
    provider.model = "gpt-5-mini"
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    result = decide_alpha_winner(
        sector="software",
        prior_ranking=[{"ticker": "AAA", "consensus_rank": 1}],
        investigation_plan={"selected_targets": [{"ticker": "AAA"}]},
        candidate_investigations=[{"ticker": "AAA", "eligible_for_selection": True}],
        hard_blocked_candidates=[],
        integrity_scope=_valid_scope(packet),
        max_cost_usd=0.0000005,
    )

    assert provider.calls == 0
    assert result["decision_mode"] == "fallback"
    assert result["physical_calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["budget_usd"] == 0.0000005
    assert result["budget_remaining_usd"] == 0.0000005


def test_alpha_anthropic_tool_loop_gates_before_messages_create(monkeypatch):
    packet = _valid_packet()
    provider = _CountingProvider()
    provider.provider_name = "anthropic"
    calls = {"messages_create": 0}

    class _Messages:
        def create(self, **kwargs):
            calls["messages_create"] += 1
            raise AssertionError(f"Anthropic call must be suppressed: {kwargs}")

    client = SimpleNamespace(messages=_Messages())
    sdk = SimpleNamespace(Anthropic=lambda **kwargs: client)
    monkeypatch.setattr("app.alpha.llm_runtime._anthropic_sdk", sdk)
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    with pytest.raises(InvalidFinancialInputError):
        run_candidate_investigation(
            ctx=AlphaToolContext(
                sector="software",
                ticker="AAA",
                packet=packet,
                as_of_date="2026-04-19",
            ),
            investigation_request={"ticker": "AAA", "evidence_gaps": []},
            hard_block_reasons=[],
            integrity_scope=_invalid_scope(),
        )

    assert calls["messages_create"] == 0
    assert provider.calls == 0


def test_alpha_anthropic_raw_sdk_receives_only_messages_api_kwargs(monkeypatch):
    # The SDK is faked below; the network switch is on for this call.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    from app.config import get_config

    get_config.cache_clear()
    packet = _valid_packet()
    provider = _CountingProvider()
    provider.provider_name = "anthropic"
    provider.model = "claude-sonnet-4-6"
    provider._handles_retry_guard = True
    calls = {"messages_create": 0}

    class _Messages:
        def create(
            self,
            *,
            model,
            max_tokens,
            system,
            tools,
            tool_choice,
            messages,
        ):
            calls["messages_create"] += 1
            assert model == provider.model
            assert max_tokens == 4000
            assert system
            assert tools
            assert tool_choice == {"type": "auto"}
            assert messages
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=20),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="final-1",
                        name="finalize_candidate_investigation",
                        input={
                            "verdict": "WATCH",
                            "confidence": "LOW",
                            "key_findings": ["Measured finding."],
                            "open_questions": [],
                            "key_risk": "Measured risk.",
                            "falsification_trigger": "Measured trigger.",
                            "reasoning_trace": "Measured trace.",
                            "model_validity": "VALID",
                            "selection_blockers": [],
                            "eligible_for_selection": True,
                        },
                    )
                ],
            )

    client = SimpleNamespace(messages=_Messages())
    sdk = SimpleNamespace(Anthropic=lambda **kwargs: client)
    monkeypatch.setattr("app.alpha.llm_runtime._anthropic_sdk", sdk)
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    result = run_candidate_investigation(
        ctx=AlphaToolContext(
            sector="software",
            ticker="AAA",
            packet=packet,
            as_of_date="2026-04-19",
        ),
        investigation_request={"ticker": "AAA", "evidence_gaps": []},
        hard_block_reasons=[],
        integrity_scope=_valid_scope(packet),
    )

    assert calls["messages_create"] == 1
    assert provider.calls == 0
    assert result["physical_calls"] == 1
    assert result["termination_reason"] == "finalize_candidate_investigation"


def test_alpha_anthropic_zero_budget_suppresses_every_paid_path(monkeypatch):
    packet = _valid_packet()
    provider = _CountingProvider()
    provider.provider_name = "anthropic"
    provider.model = "claude-test"
    calls = {"messages_create": 0}

    class _Messages:
        def create(self, **kwargs):
            calls["messages_create"] += 1
            raise AssertionError(f"zero budget must suppress Anthropic: {kwargs}")

    client = SimpleNamespace(messages=_Messages())
    sdk = SimpleNamespace(Anthropic=lambda **kwargs: client)
    monkeypatch.setattr("app.alpha.llm_runtime._anthropic_sdk", sdk)
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps",
        lambda evidence_gaps: [],
    )

    result = run_candidate_investigation(
        ctx=AlphaToolContext(
            sector="software",
            ticker="AAA",
            packet=packet,
            as_of_date="2026-04-19",
        ),
        investigation_request={"ticker": "AAA", "evidence_gaps": []},
        hard_block_reasons=[],
        config=AlphaInvestigationConfig(max_cost_usd=0.0),
        integrity_scope=_valid_scope(packet),
    )

    assert calls["messages_create"] == 0
    assert provider.calls == 0
    assert result["physical_calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["provider_usage"] == []


def test_alpha_candidate_tiny_positive_budget_prevents_overshoot(monkeypatch):
    packet = _valid_packet()
    provider = _CountingProvider()
    provider.model = "gpt-5-mini"
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps",
        lambda evidence_gaps: [],
    )

    result = run_candidate_investigation(
        ctx=AlphaToolContext(
            sector="software",
            ticker="AAA",
            packet=packet,
            as_of_date="2026-04-19",
        ),
        investigation_request={"ticker": "AAA", "evidence_gaps": []},
        hard_block_reasons=[],
        config=AlphaInvestigationConfig(max_cost_usd=0.0000005),
        integrity_scope=_valid_scope(packet),
    )

    assert provider.calls == 0
    assert result["physical_calls"] == 0
    assert result["cost_usd"] == 0.0


def test_alpha_generic_candidate_reports_successful_provider_usage(monkeypatch):
    packet = _valid_packet()

    class _UsageProvider:
        provider_name = "openai"
        model = "gpt-5-mini"

        def __init__(self):
            self.calls = 0

        def enabled(self):
            return True

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                json_text=(
                    '{"verdict":"WATCH","confidence":"LOW",'
                    '"key_findings":["Measured finding."],"open_questions":[],'
                    '"key_risk":"Measured risk.","falsification_trigger":"Measured trigger.",'
                    '"reasoning_trace":"Measured trace.","model_validity":"VALID",'
                    '"selection_blockers":[],"eligible_for_selection":true}'
                ),
                usage_input_tokens=120,
                usage_output_tokens=40,
                model=self.model,
            )

    provider = _UsageProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps",
        lambda evidence_gaps: [],
    )

    result = run_candidate_investigation(
        ctx=AlphaToolContext(
            sector="software",
            ticker="AAA",
            packet=packet,
            as_of_date="2026-04-19",
        ),
        investigation_request={"ticker": "AAA", "evidence_gaps": []},
        hard_block_reasons=[],
        integrity_scope=_valid_scope(packet),
    )

    assert provider.calls == 1
    assert result["physical_calls"] == 1
    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 40
    assert result["cost_usd"] > 0.0
    assert result["provider_usage"][0]["status"] == "OK"
    assert result["provider_usage"][0]["model"] == "gpt-5-mini"


def test_alpha_failed_billed_response_is_preserved_in_fallback(monkeypatch):
    packet = _valid_packet()

    class _BilledFailureProvider:
        provider_name = "openai"
        model = "gpt-5-mini"

        def __init__(self):
            self.calls = 0

        def enabled(self):
            return True

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            error = RuntimeError("parse failed after billed response")
            error._provider_response_attempts = [
                {
                    "status": "completed",
                    "model": self.model,
                    "usage": {"input_tokens": 90, "output_tokens": 30},
                }
            ]
            raise error

    provider = _BilledFailureProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.alpha.llm_runtime.suggested_tool_calls_for_gaps",
        lambda evidence_gaps: [],
    )

    result = run_candidate_investigation(
        ctx=AlphaToolContext(
            sector="software",
            ticker="AAA",
            packet=packet,
            as_of_date="2026-04-19",
        ),
        investigation_request={"ticker": "AAA", "evidence_gaps": []},
        hard_block_reasons=[],
        integrity_scope=_valid_scope(packet),
    )

    assert provider.calls == 1
    assert result["physical_calls"] == 1
    assert result["input_tokens"] == 90
    assert result["output_tokens"] == 30
    assert result["cost_usd"] > 0.0
    assert result["provider_usage"][0]["status"] == "OK"
    assert result["termination_reason"] == "fallback_summary_error"


def test_alpha_planner_and_final_decision_report_paid_usage(monkeypatch):
    packet = _valid_packet()

    class _SequencedUsageProvider:
        provider_name = "openai"
        model = "gpt-5-mini"
        _handles_retry_guard = True

        def __init__(self):
            self.calls = 0
            self.requests = []

        def enabled(self):
            return True

        def synthesize_json(self, **kwargs):
            self.calls += 1
            self.requests.append(dict(kwargs))
            if self.calls == 1:
                payload = (
                    '{"summary":"Investigate AAA.","selected_targets":'
                    '[{"ticker":"AAA","reason":"Check it.","evidence_gaps":[]}],'
                    '"skipped_candidates":[]}'
                )
                usage = (70, 20)
            else:
                payload = (
                    '{"winner":"AAA","runner_up":null,"winner_thesis":"Best evidence.",'
                    '"winner_conviction":"HIGH","runner_up_thesis":"",'
                    '"key_risk":"Risk.","falsification_trigger":"Trigger.",'
                    '"time_horizon":"12 months","decision_trace":"Selected AAA.",'
                    '"rejected_candidates":[]}'
                )
                usage = (80, 25)
            return SimpleNamespace(
                json_text=payload,
                usage_input_tokens=usage[0],
                usage_output_tokens=usage[1],
                model=self.model,
            )

    provider = _SequencedUsageProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    plan = plan_alpha_investigations(
        sector="software",
        prior_candidates=[{"ticker": "AAA", "hard_block_reasons": []}],
        hard_blocked_candidates=[],
        max_targets=1,
        integrity_scope=_valid_scope(packet),
    )
    decision = decide_alpha_winner(
        sector="software",
        prior_ranking=[{"ticker": "AAA", "consensus_rank": 1}],
        investigation_plan=plan,
        candidate_investigations=[{"ticker": "AAA", "eligible_for_selection": True}],
        hard_blocked_candidates=[],
        integrity_scope=_valid_scope(packet),
    )

    assert provider.calls == 2
    assert plan["physical_calls"] == 1
    assert plan["input_tokens"] == 70
    assert plan["output_tokens"] == 20
    assert plan["cost_usd"] > 0.0
    assert decision["physical_calls"] == 1
    assert decision["input_tokens"] == 80
    assert decision["output_tokens"] == 25
    assert decision["cost_usd"] > 0.0
    for request in provider.requests:
        assert request["allow_output_token_retry"] is False
        assert request["max_retries_per_request"] == 0


def test_alpha_anthropic_loop_rejects_changed_tool_payload_before_next_call(
    monkeypatch,
):
    # The SDK is faked below; the network switch is on for this call.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    from app.config import get_config

    get_config.cache_clear()
    packet = _valid_packet()
    provider = _CountingProvider()
    provider.provider_name = "anthropic"
    calls = {"messages_create": 0, "guard_entries": 0}
    mutable_tool_output = [{"metric": 1.0, "unit": "ratio"}]

    class _Messages:
        def create(self, **_kwargs):
            calls["messages_create"] += 1
            if calls["messages_create"] != 1:
                raise AssertionError("changed tool payload must suppress the next call")
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=0, output_tokens=0),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="tool-1",
                        name="compare_peer_metric",
                        input={"metric": "operating_margin"},
                    )
                ],
            )

    @contextmanager
    def _mutating_attempt_guard(_guard):
        calls["guard_entries"] += 1
        if calls["guard_entries"] == 2:
            mutable_tool_output[0]["metric"] = 2.0
        yield

    client = SimpleNamespace(messages=_Messages())
    sdk = SimpleNamespace(Anthropic=lambda **kwargs: client)
    monkeypatch.setattr("app.alpha.llm_runtime._anthropic_sdk", sdk)
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.alpha.llm_runtime.dispatch_alpha_tool_json",
        lambda *_args, **_kwargs: mutable_tool_output,
    )
    monkeypatch.setattr(
        "app.alpha.llm_runtime.llm_physical_attempt_guard",
        _mutating_attempt_guard,
    )

    with pytest.raises(InvalidFinancialInputError) as caught:
        run_candidate_investigation(
            ctx=AlphaToolContext(
                sector="software",
                ticker="AAA",
                packet=packet,
                as_of_date="2026-04-19",
            ),
            investigation_request={
                "ticker": "AAA",
                "evidence_gaps": ["Compare operating margin."],
            },
            hard_block_reasons=[],
            integrity_scope=_valid_scope(packet),
        )

    assert calls == {"messages_create": 1, "guard_entries": 2}
    assert caught.value.violations[0].code == "BOUND_FINANCIAL_INPUT_MUTATED"


def test_alpha_scan_invalid_integrity_returns_state_and_writes_no_report(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())
    assembly_kwargs: dict[str, object] = {}

    def _assemble(tickers, **kwargs):
        assembly_kwargs.update(kwargs)
        return {"AAA": TickerSignalPacket(ticker="AAA", current_price=100.0)}

    monkeypatch.setattr(
        "app.valuation.peer_context._load_sector_tickers",
        lambda sector: ["AAA"],
    )
    monkeypatch.setattr("app.alpha.signal_assembler.assemble_sector_packets", _assemble)

    result = CliRunner().invoke(
        app,
        ["alpha-scan", "--sector", "software", "--skip-quarterly", "--deterministic"],
    )

    assert result.exit_code == 1
    payload = result.output[result.output.index("{") :]
    assert '"financial_integrity_status": "NEEDS_DATA"' in payload
    assert '"report_path": null' in payload
    assert assembly_kwargs["filing_risk_use_llm"] is False
    assert assembly_kwargs["as_of_date"] == date.today().isoformat()
    assert assembly_kwargs["pipeline_version"] == "v1"
    assert assembly_kwargs["current_prices"] == {"AAA": None}
    assert assembly_kwargs["db_path"] == data_dir / "engine.db"
    issuer_context = assembly_kwargs["issuer_contexts"]["AAA"]
    assert issuer_context["ticker"] == "AAA"
    assert issuer_context["price_used"] is None
    assert issuer_context["market_cap_mm"] is None
    output_dir = data_dir / "outputs" / "alpha"
    assert not (output_dir / "alpha_software.json").exists()
    assert not (output_dir / "alpha_software_report.md").exists()
    get_config.cache_clear()


def test_alpha_scan_revalidates_before_publication(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())
    packet = _valid_packet(as_of_date=date.today().isoformat())
    report = SectorAlphaReport(
        sector="software",
        total_candidates=1,
        rounds=[],
        winner="AAA",
        winner_thesis="Deterministic test winner.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Test risk.",
        falsification_trigger="Test trigger.",
        time_horizon="12 months",
        signal_packets={"AAA": packet.to_summary_dict()},
        selection_basis="consensus",
    )

    def _compare(sector, packets, **kwargs):
        # The new value is still individually valid, so only comparison with
        # the exact originally authorized fingerprint can catch this drift.
        packets["AAA"].dcf_value = 151.0
        return report

    monkeypatch.setattr(
        "app.valuation.peer_context._load_sector_tickers",
        lambda sector: ["AAA"],
    )
    monkeypatch.setattr(
        "app.alpha.signal_assembler.assemble_sector_packets",
        lambda tickers, **kwargs: {"AAA": packet},
    )
    monkeypatch.setattr("app.alpha.sector_comparator.run_sector_comparison", _compare)

    result = CliRunner().invoke(
        app,
        ["alpha-scan", "--sector", "software", "--skip-quarterly", "--deterministic"],
    )

    assert result.exit_code == 1
    payload = result.output[result.output.index("{") :]
    assert '"financial_integrity_status": "INVALID_FINANCIAL_INPUT"' in payload
    output_dir = data_dir / "outputs" / "alpha"
    assert not (output_dir / "alpha_software.json").exists()
    assert not (output_dir / "alpha_software_report.md").exists()
    get_config.cache_clear()


def test_alpha_scan_suppresses_post_gate_supplemental_fact_mutation_without_egress(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())
    run_as_of_date = date.today().isoformat()
    fiscal_year = date.today().year - 1
    filed_date = run_as_of_date
    period_end = f"{fiscal_year}-12-31"
    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                'AAA', ?, 'FY', ?, ?, ?, 'USD_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                ?, ?, '10-K', '0000000001-26-000001'
            )
            """,
            [
                (
                    fiscal_year,
                    period_end,
                    line_item,
                    value,
                    utc_now_iso(),
                    filed_date,
                )
                for line_item, value in (
                    ("revenue", 100.0),
                    ("operating_income", 20.0),
                    ("net_income", 15.0),
                    ("cfo", 30.0),
                    ("capex", 5.0),
                )
            ],
        )

    packet = _valid_packet(as_of_date=run_as_of_date)
    report = SectorAlphaReport(
        sector="software",
        total_candidates=1,
        rounds=[],
        winner="AAA",
        winner_thesis="Deterministic test winner.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Test risk.",
        falsification_trigger="Test trigger.",
        time_horizon="12 months",
        signal_packets={"AAA": packet.to_summary_dict()},
        selection_basis="consensus",
    )
    provider = _CountingProvider()
    calls = {"network": 0}

    def _network_call(*args, **kwargs):
        calls["network"] += 1
        raise AssertionError(f"network call must be suppressed: {args}, {kwargs}")

    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.util.http.HttpClient.get_bytes", _network_call)
    monkeypatch.setattr(
        "app.valuation.peer_context._load_sector_tickers",
        lambda sector: ["AAA"],
    )
    monkeypatch.setattr(
        "app.alpha.signal_assembler.assemble_sector_packets",
        lambda tickers, **kwargs: {"AAA": packet},
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_sector_comparison",
        lambda sector, packets, **kwargs: report,
    )

    from app.alpha import publication as publication_module

    original_render = publication_module.render_alpha_report

    def _render_then_mutate(*args, **kwargs):
        markdown = original_render(*args, **kwargs)
        with get_db() as conn:
            conn.execute(
                """
                UPDATE companyfacts_facts
                SET value = 101.0
                WHERE ticker = 'AAA' AND line_item = 'revenue'
                """
            )
        return markdown

    monkeypatch.setattr(publication_module, "render_alpha_report", _render_then_mutate)

    result = CliRunner().invoke(
        app,
        ["alpha-scan", "--sector", "software", "--skip-quarterly", "--deterministic"],
    )

    assert result.exit_code == 1
    payload = result.output[result.output.index("{") :]
    assert '"financial_integrity_status": "INVALID_FINANCIAL_INPUT"' in payload
    assert '"BOUND_FINANCIAL_INPUT_MUTATED"' in payload
    assert provider.calls == 0
    assert calls["network"] == 0
    output_dir = data_dir / "outputs" / "alpha"
    assert not (output_dir / "alpha_software.json").exists()
    assert not (output_dir / "alpha_software_report.md").exists()
    assert not (output_dir / "alpha_software.financial_integrity_authorization.json").exists()
    get_config.cache_clear()


@pytest.mark.parametrize(
    ("units", "value"),
    [
        ("USD", 100_000_000.0),
        ("", 100.0),
        (None, 100.0),
    ],
)
def test_alpha_publication_rejects_non_normalized_supplemental_units(
    monkeypatch,
    tmp_path,
    units,
    value,
):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())
    run_as_of_date = date.today().isoformat()
    fiscal_year = date.today().year - 1
    packet = _valid_packet(as_of_date=run_as_of_date)
    report = SectorAlphaReport(
        sector="software",
        total_candidates=1,
        rounds=[],
        winner="AAA",
        winner_thesis="Unit validation fixture.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Test risk.",
        falsification_trigger="Test trigger.",
        time_horizon="12 months",
        signal_packets={"AAA": packet.to_summary_dict()},
        selection_basis="consensus",
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                'AAA', ?, 'FY', ?, 'revenue', ?, ?,
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                ?, ?, '10-K', '0000000001-26-000001'
            )
            """,
            (
                fiscal_year,
                f"{fiscal_year}-12-31",
                value,
                units,
                utc_now_iso(),
                run_as_of_date,
            ),
        )
    decision_scope = FinancialIntegrityScope(
        context="alpha_scan:software",
        run_as_of_date=run_as_of_date,
        packets=(packet,),
    )
    decision_result = validate_financial_integrity_scope(decision_scope)
    assert decision_result.passed

    with pytest.raises(InvalidFinancialInputError) as caught:
        prepare_alpha_publication(
            report=report,
            packets={"AAA": packet},
            run_as_of_date=run_as_of_date,
            decision_scope=decision_scope,
        )

    assert caught.value.status == "NEEDS_DATA"
    assert caught.value.violations[0].code == "ALPHA_REPORT_FINANCIAL_UNIT_INVALID"
    assert caught.value.violations[0].source_values["units"] == units
    assert not (data_dir / "outputs" / "alpha").exists()
    get_config.cache_clear()


def test_alpha_publication_rejects_decision_packet_scope_from_another_report(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())
    run_as_of_date = "2026-04-19"
    decision_packet = _valid_packet("AAA", as_of_date=run_as_of_date)
    publication_packet = _valid_packet("BBB", as_of_date=run_as_of_date)
    report = SectorAlphaReport(
        sector="software",
        total_candidates=1,
        rounds=[],
        winner="BBB",
        winner_thesis="Cross-scope rejection fixture.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Test risk.",
        falsification_trigger="Test trigger.",
        time_horizon="12 months",
        signal_packets={"BBB": publication_packet.to_summary_dict()},
        selection_basis="consensus",
    )
    decision_scope = FinancialIntegrityScope(
        context="alpha_scan:software",
        run_as_of_date=run_as_of_date,
        packets=(decision_packet,),
    )
    decision_result = validate_financial_integrity_scope(decision_scope)
    assert decision_result.passed

    with pytest.raises(InvalidFinancialInputError) as caught:
        prepare_alpha_publication(
            report=report,
            packets={"BBB": publication_packet},
            run_as_of_date=run_as_of_date,
            decision_scope=decision_scope,
        )

    assert caught.value.status == "INVALID_FINANCIAL_INPUT"
    assert caught.value.violations[0].code == "ALPHA_DECISION_PUBLICATION_SCOPE_MISMATCH"
    assert caught.value.violations[0].source_values["decision_tickers"] == ["AAA"]
    assert caught.value.violations[0].source_values["publication_tickers"] == ["BBB"]
    assert not (data_dir / "outputs" / "alpha").exists()
    get_config.cache_clear()


def test_alpha_scan_rejects_detached_report_financials_without_provider_call(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())
    run_as_of_date = date.today().isoformat()
    packet = _valid_packet(as_of_date=run_as_of_date)
    report = SectorAlphaReport(
        sector="software",
        total_candidates=1,
        rounds=[],
        winner="AAA",
        winner_thesis="Detached financial fixture.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Test risk.",
        falsification_trigger="Test trigger.",
        time_horizon="12 months",
        signal_packets={"AAA": {"ticker": "AAA", "current_price": 101.0}},
        selection_basis="consensus",
    )
    provider = _CountingProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.valuation.peer_context._load_sector_tickers",
        lambda sector: ["AAA"],
    )
    monkeypatch.setattr(
        "app.alpha.signal_assembler.assemble_sector_packets",
        lambda tickers, **kwargs: {"AAA": packet},
    )
    monkeypatch.setattr(
        "app.alpha.sector_comparator.run_sector_comparison",
        lambda sector, packets, **kwargs: report,
    )

    result = CliRunner().invoke(
        app,
        ["alpha-scan", "--sector", "software", "--skip-quarterly", "--deterministic"],
    )

    assert result.exit_code == 1
    payload = result.output[result.output.index("{") :]
    assert '"ALPHA_REPORT_PACKET_MISMATCH"' in payload
    assert provider.calls == 0
    output_dir = data_dir / "outputs" / "alpha"
    assert not (output_dir / "alpha_software.json").exists()
    assert not (output_dir / "alpha_software_report.md").exists()
    assert not (output_dir / "alpha_software.financial_integrity_authorization.json").exists()
    get_config.cache_clear()


def test_valid_scope_allows_provider_boundary(monkeypatch):
    packet = _valid_packet()

    class _PlannerProvider:
        calls = 0

        def synthesize_json(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                json_text=('{"summary":"ok","selected_targets":[],"skipped_candidates":[]}')
            )

    provider = _PlannerProvider()
    monkeypatch.setattr("app.alpha.llm_runtime.get_alpha_llm_provider", lambda: provider)

    result = plan_alpha_investigations(
        sector="software",
        prior_candidates=[{"ticker": "AAA", "hard_block_reasons": []}],
        hard_blocked_candidates=[],
        max_targets=1,
        integrity_scope=_valid_scope(packet),
    )

    assert result["planner_mode"] == "llm"
    assert provider.calls == 1
