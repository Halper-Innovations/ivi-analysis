from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.alpha.schemas import TickerSignalPacket
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    InvalidFinancialInputError,
)
from app.autonomous.run_contract import (
    AutonomousRunArtifact,
    AutonomousRunBudget,
    EvidenceReference,
)
from app.autonomous.run_report import render_autonomous_run_report
from app.autonomous.runtime import (
    _financial_payload_fingerprint,
    _is_quota_exhaustion,
    _signal_packet_fingerprint,
    run_single_candidate_autonomous_analysis,
)
from app.llm.providers.retry_guard import llm_cost_budget
from app.llm.usage_capture import (
    attached_provider_usage_records,
    record_provider_usage,
)


class TestRuntimeQuotaExhaustionDetection:
    def test_insufficient_quota_is_exhaustion(self):
        assert _is_quota_exhaustion(RuntimeError("status=429 insufficient_quota")) is True

    def test_breaker_open_with_insufficient_quota_is_exhaustion(self):
        exc = RuntimeError("OpenAI provider circuit breaker open: status=429 insufficient_quota")
        assert _is_quota_exhaustion(exc) is True

    def test_transient_rate_limit_is_not_exhaustion(self):
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


@pytest.fixture(autouse=True)
def _disable_default_analyst_context(monkeypatch):
    gate_result = SimpleNamespace(
        status="PASS",
        passed=True,
        is_valid=True,
        scope_fingerprint="legacy-autonomous-runtime-fixture",
    )
    monkeypatch.setattr(
        "app.autonomous.runtime.require_financial_integrity_scope",
        lambda scope: gate_result,
    )
    monkeypatch.setattr(
        "app.autonomous.runtime.require_unchanged_financial_integrity_scope",
        lambda scope, *, expected_scope_fingerprint: gate_result,
    )
    monkeypatch.setattr(
        "app.autonomous.runtime._build_analyst_context_evidence",
        lambda **kwargs: ([], None, [], []),
    )


def _packet() -> TickerSignalPacket:
    return TickerSignalPacket(
        ticker="AAA",
        dcf_value=140.0,
        epv_value=110.0,
        current_price=100.0,
        gate_verdict="PROCEED",
        raw_valuation={"sector": "test_sector"},
    )


def _bound_provider_request(runtime_module, provider):
    packet = _packet()
    scope = FinancialIntegrityScope(
        context="autonomous_request_envelope_test",
        run_as_of_date="2026-04-25",
        packets=(packet,),
    )
    gate_result = runtime_module._require_applied_financial_integrity_scope(scope)
    state = {"packet": packet.to_summary_dict(), "epoch": 1}
    frozen_state = runtime_module._freeze_financial_prompt_state(
        scope=scope,
        expected_scope_fingerprint=gate_result.scope_fingerprint,
        state_getter=lambda: state,
    )
    request = {
        "prompt": "Use the exact authorized financial packet.",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "schema_name": "autonomous_request_envelope_test",
        "max_output_tokens": 200,
    }
    kwargs = {
        "integrity_scope": scope,
        "_financial_prompt_integrity_binding": frozen_state.bind_request(
            provider,
            request,
        ),
        **request,
    }
    return request, kwargs


def _budget(max_tool_calls: int = 4, max_turns: int = 4) -> AutonomousRunBudget:
    return AutonomousRunBudget(
        max_tool_calls=max_tool_calls,
        max_turns=max_turns,
        max_cost_usd=1.25,
        timebox_seconds=None,
        max_candidates=1,
    )


def _plan_payload() -> dict:
    return {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Is the valuation supported by current KPIs?",
                "rationale": "The prior needs deterministic confirmation.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "fetch_kpi_trends",
                        "tool_input": {},
                        "rationale": "Check valuation and quality context.",
                    }
                ],
            }
        ]
    }


def _final_payload(verdict: str = "ACTIONABLE") -> dict:
    return {
        "final_verdict": verdict,
        "selected_ticker": "AAA" if verdict != "NO_WINNER" else None,
        "confidence": "MODERATE" if verdict != "NO_WINNER" else None,
        "belief_updates": [
            {
                "question_id": "Q1",
                "ticker": "AAA",
                "prior_belief": "The valuation may be unsupported.",
                "updated_belief": "KPI evidence supports the initial valuation but does not eliminate risk.",
                "direction": "BULLISH",
                "confidence_after": "MODERATE",
                "summary": "KPI tool evidence was directionally supportive.",
                "evidence_ref_ids": ["E1"],
                "remaining_uncertainty": ["Filing detail still limited."],
            }
        ],
        "candidate_decision": {
            "ticker": "AAA",
            "verdict": verdict,
            "confidence": "MODERATE",
            "thesis": "AAA is actionable with a capped confidence label.",
            "key_risk": "Evidence is still incomplete.",
            "eligible_for_selection": verdict == "ACTIONABLE",
            "selection_blockers": [],
            "falsifiers": ["Revenue quality weakens."],
            "evidence_ref_ids": ["E1"],
            "confidence_cap_reasons": ["FILING_DETAIL_LIMITED"],
        },
        "no_winner_reason": None,
        "degraded_states": [],
        "audit_notes": ["Question selection and tool use are reconstructable."],
    }


def test_bound_provider_request_rejects_pre_call_schema_mutation_at_zero_cost():
    from app.autonomous import runtime as runtime_module

    provider = FakeProvider([{"ok": True}])
    request, kwargs = _bound_provider_request(runtime_module, provider)
    request["schema"]["properties"]["ok"]["type"] = "integer"

    with (
        llm_cost_budget(max_cost_usd=10.0) as cost_context,
        pytest.raises(InvalidFinancialInputError) as exc_info,
    ):
        runtime_module._call_provider_json(provider, kwargs)

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_PROVIDER_REQUEST_MUTATED"
    assert provider.calls == []
    assert cost_context.call_count == 0
    assert cost_context.cumulative_cost_usd == 0.0


def test_bound_provider_request_rejects_schema_mutation_after_physical_call():
    from app.autonomous import runtime as runtime_module

    class MutatingProvider(FakeProvider):
        def synthesize_json(self, **kwargs):
            self.calls.append(kwargs)
            kwargs["schema"]["properties"]["ok"]["type"] = "integer"
            return SimpleNamespace(json_text=json.dumps({"ok": True}))

    provider = MutatingProvider([])
    _request, kwargs = _bound_provider_request(runtime_module, provider)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        runtime_module._call_provider_json(provider, kwargs)

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_PROVIDER_REQUEST_MUTATED"
    assert len(provider.calls) == 1


def test_bound_provider_request_mutation_suppresses_physical_retry():
    from app.autonomous import runtime as runtime_module
    from app.llm.providers.retry_guard import call_with_llm_retry_guard

    class RetryingProvider:
        provider_name = "openai"
        _handles_retry_guard = True

        def __init__(self):
            self.attempts = 0

        def synthesize_json(self, **kwargs):
            def attempt():
                self.attempts += 1
                kwargs["schema"]["properties"]["ok"]["type"] = "integer"
                raise RuntimeError("status=503 temporary server error")

            return call_with_llm_retry_guard(
                provider_name=self.provider_name,
                schema_name=kwargs["schema_name"],
                call=attempt,
                max_retries=1,
                backoff_seconds=(),
            )

    provider = RetryingProvider()
    _request, kwargs = _bound_provider_request(runtime_module, provider)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        runtime_module._call_provider_json(provider, kwargs)

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_PROVIDER_REQUEST_MUTATED"
    assert provider.attempts == 1


def test_bound_provider_request_mutation_suppresses_cross_provider_fallback(
    monkeypatch,
):
    from app.autonomous import runtime as runtime_module

    class MutatingQuotaProvider:
        provider_name = "openai"

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, **kwargs):
            self.calls += 1
            kwargs["schema"]["properties"]["ok"]["type"] = "integer"
            raise RuntimeError("status=429 insufficient_quota")

    primary = MutatingQuotaProvider()
    fallback = FakeProvider([{"ok": True}])
    monkeypatch.setattr(runtime_module, "get_anthropic_provider", lambda: fallback)
    _request, kwargs = _bound_provider_request(runtime_module, primary)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        runtime_module._synthesize_provider_json(primary, **kwargs)

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_PROVIDER_REQUEST_MUTATED"
    assert primary.calls == 1
    assert fallback.calls == []


def test_provider_questions_become_records_and_tools_execute(monkeypatch):
    provider = FakeProvider([_plan_payload(), _final_payload()])
    dispatched: list[tuple[str, dict]] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append((name, tool_input))
        return {"status": "ok", "ticker": ctx.ticker, "quality": "supportive"}

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        objective="Decide if AAA is actionable.",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "ACTIONABLE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.questions[0].question_id == "Q1"
    assert artifact.questions[0].status == "ANSWERED"
    assert artifact.tool_calls[0].tool_name == "fetch_kpi_trends"
    assert artifact.tool_calls[0].status == "OK"
    assert artifact.evidence[0].evidence_id == "E1"
    assert (
        artifact.belief_updates[0].updated_belief
        == "KPI evidence supports the initial valuation but does not eliminate risk."
    )
    assert artifact.candidate_decisions[0].eligible_for_selection is True
    assert dispatched == [("fetch_kpi_trends", {})]


def test_strict_nested_runtime_uses_shared_cap_and_suppresses_untracked_llm_lanes(
    monkeypatch,
):
    class StrictAnthropicProvider:
        provider_name = "anthropic"
        model = "claude-sonnet-4-6"

        def __init__(self):
            self.calls: list[dict] = []

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(json_text=json.dumps(_plan_payload()))

    provider = StrictAnthropicProvider()
    assembly_calls: list[tuple[str, dict]] = []
    analyst_calls: list[dict] = []

    def assemble(ticker, **kwargs):
        assembly_calls.append((ticker, dict(kwargs)))
        return _packet()

    def build_analyst_context(**kwargs):
        analyst_calls.append(dict(kwargs))
        return [], None, [], []

    monkeypatch.setattr(
        "app.autonomous.runtime.get_alpha_llm_provider",
        lambda: provider,
    )
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", assemble)
    monkeypatch.setattr(
        "app.autonomous.runtime._build_analyst_context_evidence",
        build_analyst_context,
    )

    with llm_cost_budget(
        max_cost_usd=0.0,
        strict_first_call=True,
    ) as cost_context:
        artifact = run_single_candidate_autonomous_analysis(
            ticker="AAA",
            as_of_date="2026-04-25",
            budget=_budget(max_tool_calls=4, max_turns=2),
        )

    assert assembly_calls == [
        (
            "AAA",
            {
                "filing_risk_use_llm": False,
                "as_of_date": "2026-04-25",
                "pipeline_version": "v1",
            },
        )
    ]
    assert len(analyst_calls) == 1
    assert analyst_calls[0]["skip_analysis_refresh"] is True
    assert provider.calls == []
    assert cost_context.call_count == 0
    assert artifact.status == "FAILED"
    assert artifact.degraded_states == ["LLM_PROVIDER_ERROR"]
    assert "LLM cost budget exceeded" in str(artifact.no_winner_reason)


def test_quota_exhausted_primary_provider_falls_back_to_anthropic(monkeypatch):
    primary = QuotaProvider()
    fallback = FakeProvider([_plan_payload(), _final_payload()])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: primary)
    monkeypatch.setattr("app.autonomous.runtime.get_anthropic_provider", lambda: fallback)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "quality": "supportive",
        },
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        objective="Decide if AAA is actionable.",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.status == "COMPLETED"
    assert artifact.final_verdict == "ACTIONABLE"
    assert len(primary.calls) == 2
    assert len(fallback.calls) == 2
    assert artifact.tool_calls[0].status == "OK"


def test_strict_single_candidate_run_forbids_cross_provider_fallback(monkeypatch):
    primary = QuotaProvider()
    fallback = FakeProvider([_plan_payload(), _final_payload()])
    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: primary)
    monkeypatch.setattr("app.autonomous.runtime.get_anthropic_provider", lambda: fallback)
    monkeypatch.setattr(
        "app.autonomous.runtime.assemble_signal_packet",
        lambda ticker, **kwargs: _packet(),
    )

    with llm_cost_budget(max_cost_usd=10.0, strict_first_call=True):
        artifact = run_single_candidate_autonomous_analysis(
            ticker="AAA",
            as_of_date="2026-04-25",
            budget=_budget(),
        )

    assert artifact.status == "FAILED"
    assert fallback.calls == []


def test_both_providers_quota_exhausted_fails_without_success_artifact(monkeypatch):
    primary = QuotaProvider()

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: primary)
    monkeypatch.setattr("app.autonomous.runtime.get_anthropic_provider", lambda: None)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.status == "FAILED"
    assert artifact.final_verdict == "NO_WINNER"
    assert artifact.degraded_states == ["LLM_PROVIDER_QUOTA_EXHAUSTED"]
    assert artifact.tool_calls == []


def test_analyst_context_is_seeded_before_provider_planning(monkeypatch):
    context = {
        "ticker": "AAA",
        "company_identity": {
            "ticker": "AAA",
            "company_name": "AAA Corporation",
            "cik": "123456",
            "homepage_url": "https://aaa.example",
        },
        "verdict": "WATCH",
        "confidence_label": "MODERATE",
        "thesis_summary": "AAA has a plausible setup but needs KPI confirmation.",
        "latest_evidence_date": "2026-04-24",
        "paths": {
            "analysis_report_markdown": "/tmp/analysis_report.md",
            "analysis_report_json": "/tmp/analysis_report.json",
            "analysis_evidence_bundle": "/tmp/analysis_evidence_bundle.json",
        },
    }
    provider = FakeProvider([_plan_payload(), _final_payload()])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime._build_analyst_context_evidence",
        lambda **kwargs: (
            [
                EvidenceReference(
                    evidence_id="E1",
                    source_type="analysis_report",
                    source_label="analyst_context",
                    summary="Analyst report context for AAA.",
                    ticker="AAA",
                    source_date="2026-04-24",
                    excerpt=json.dumps(context),
                    confidence="MODERATE",
                )
            ],
            context,
            [],
            ["Seeded autonomous run with analyst report context before provider planning."],
        ),
    )

    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "quality": "supportive",
        },
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        objective="Decide if AAA is actionable.",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.evidence[0].source_type == "analysis_report"
    assert artifact.evidence[0].source_label == "analyst_context"
    assert artifact.evidence[1].evidence_id == "E2"
    assert "analysis_report_markdown" in provider.calls[0]["prompt"]
    assert "Analyst context" in provider.calls[0]["prompt"]
    assert "analyst_context" in provider.calls[1]["prompt"]
    assert (
        artifact.audit_notes[0]
        == "Seeded autonomous run with analyst report context before provider planning."
    )
    report_markdown = render_autonomous_run_report(artifact)
    assert "| Company | AAA Corporation |" in report_markdown
    assert "| Analyst Markdown | /tmp/analysis_report.md |" in report_markdown


def test_followup_turn_executes_new_tools_and_skips_duplicates(monkeypatch):
    initial_plan = _plan_payload()
    followup_plan = {
        "questions": [
            {
                "question_id": "Q2",
                "question": "Can a second tool close the failed KPI evidence gap?",
                "rationale": "The first tool failed and a different deterministic check may still help.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "fetch_kpi_trends",
                        "tool_input": {},
                        "rationale": "Duplicate request should be skipped.",
                    },
                    {
                        "tool_name": "analyze_dilution",
                        "tool_input": {"years": 5},
                        "rationale": "Check owner dilution instead.",
                    },
                ],
            }
        ]
    }
    provider = FakeProvider([initial_plan, followup_plan, _final_payload("WATCHLIST_ONLY")])
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(name)
        if name == "fetch_kpi_trends":
            return {"status": "error", "reason": "temporary unavailable"}
        return {"status": "ok", "ticker": ctx.ticker, "summary": "Dilution was stable."}

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=4, max_turns=3),
    )

    assert dispatched == ["fetch_kpi_trends", "analyze_dilution"]
    assert [call.status for call in artifact.tool_calls] == [
        "ERROR",
        "SKIPPED_DUPLICATE_TOOL_CALL",
        "OK",
    ]
    assert artifact.tool_calls[2].call_id == "TC3"
    assert artifact.evidence[0].source_label == "analyze_dilution"
    assert "DUPLICATE_TOOL_CALL_SKIPPED" in artifact.degraded_states
    assert len(provider.calls) == 3
    assert "middle turn" in provider.calls[1]["prompt"]


def test_mutated_tool_evidence_blocks_next_final_provider_call(monkeypatch):
    from app.autonomous import runtime as runtime_module

    provider = FakeProvider([_plan_payload(), _final_payload("WATCHLIST_ONLY")])
    real_final_prompt = runtime_module._final_prompt

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "Deterministic KPI evidence retained its original value.",
        },
    )

    def mutate_after_tool_production(**kwargs):
        kwargs["evidence"][0].summary = "forged current_price=0 after tool production"
        return real_final_prompt(**kwargs)

    monkeypatch.setattr(
        "app.autonomous.runtime._final_prompt",
        mutate_after_tool_production,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_single_candidate_autonomous_analysis(
            ticker="AAA",
            as_of_date="2026-04-25",
            budget=_budget(max_tool_calls=4, max_turns=2),
        )

    assert [item.code for item in exc_info.value.violations] == [
        "BOUND_FINANCIAL_PROMPT_STATE_MUTATED"
    ]
    assert len(provider.calls) == 1


def test_progressive_child_starts_four_two_and_finishes_without_extension(monkeypatch):
    provider = FakeProvider([_plan_payload(), _final_payload("WATCHLIST_ONLY")])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": "The initial evidence resolved the question.",
        },
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=8, max_turns=3),
        initial_budget=_budget(max_tool_calls=4, max_turns=2),
    )

    assert len(provider.calls) == 2
    assert "'max_tool_calls': 4" in provider.calls[0]["prompt"]
    assert "'max_turns': 2" in provider.calls[0]["prompt"]
    assert any(
        note == "Progressive autonomous budget completed within the initial 4-tool/2-turn envelope."
        for note in artifact.audit_notes
    )


def test_progressive_child_extends_once_to_eight_three_when_evidence_is_open(monkeypatch):
    followup_plan = {
        "questions": [
            {
                "question_id": "Q2",
                "question": "Can dilution evidence resolve the failed KPI check?",
                "rationale": "A different deterministic source can close the gap.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "analyze_dilution",
                        "tool_input": {"years": 5},
                        "rationale": "Resolve the remaining evidence gap.",
                    }
                ],
            }
        ]
    }
    provider = FakeProvider([_plan_payload(), followup_plan, _final_payload("WATCHLIST_ONLY")])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: (
            {"status": "error", "reason": "temporary unavailable"}
            if name == "fetch_kpi_trends"
            else {
                "status": "ok",
                "ticker": ctx.ticker,
                "summary": "Dilution evidence resolved the gap.",
            }
        ),
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=8, max_turns=3),
        initial_budget=_budget(max_tool_calls=4, max_turns=2),
    )

    assert len(provider.calls) == 3
    assert "'max_tool_calls': 8" in provider.calls[1]["prompt"]
    assert "'max_turns': 3" in provider.calls[1]["prompt"]
    assert any(
        note.startswith(
            "Progressive autonomous budget extended from 4 tools/2 turns to 8 tools/3 turns"
        )
        for note in artifact.audit_notes
    )


def test_progressive_child_extends_when_answered_evidence_is_low_confidence(monkeypatch):
    followup_plan = {
        "questions": [
            {
                "question_id": "Q2",
                "question": "Can another source produce decision-usable evidence?",
                "rationale": "The answered first question produced only low-confidence evidence.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "analyze_dilution",
                        "tool_input": {"years": 5},
                        "rationale": "Seek linked evidence with adequate confidence.",
                    }
                ],
            }
        ]
    }
    provider = FakeProvider([_plan_payload(), followup_plan, _final_payload("WATCHLIST_ONLY")])
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def dispatch(name, tool_input, ctx):
        dispatched.append(name)
        if name == "fetch_kpi_trends":
            return {
                "status": "ok",
                "usable_for_decision": False,
                "summary": "The first source was readable but decision-inadequate.",
            }
        return {
            "status": "ok",
            "summary": "The follow-up source produced usable dilution evidence.",
        }

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=8, max_turns=3),
        initial_budget=_budget(max_tool_calls=4, max_turns=2),
    )

    assert dispatched == ["fetch_kpi_trends", "analyze_dilution"]
    assert len(provider.calls) == 3
    assert artifact.evidence[0].confidence == "LOW"
    assert any("Progressive autonomous budget extended" in note for note in artifact.audit_notes)


def test_progressive_child_extends_when_answered_evidence_is_not_tool_linked(monkeypatch):
    followup_plan = {
        "questions": [
            {
                "question_id": "Q2",
                "question": "Can the evidence be tied to its producing tool?",
                "rationale": "Unlinked evidence cannot support an audited decision.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "analyze_dilution",
                        "tool_input": {"years": 5},
                        "rationale": "Produce an auditable evidence link.",
                    }
                ],
            }
        ]
    }
    provider = FakeProvider([_plan_payload(), followup_plan, _final_payload("WATCHLIST_ONLY")])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "summary": "A source returned moderate-confidence evidence.",
        },
    )
    monkeypatch.setattr(
        "app.autonomous.runtime._evidence_from_tool_output",
        lambda *, ticker, call, output, evidence_index: EvidenceReference(
            evidence_id=f"E{evidence_index}",
            source_type="tool_output",
            source_label=call.tool_name,
            summary="Moderate evidence without a producing-tool link.",
            ticker=ticker,
            tool_call_id=None,
            confidence="MODERATE",
        ),
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=8, max_turns=3),
        initial_budget=_budget(max_tool_calls=4, max_turns=2),
    )

    assert len(provider.calls) == 3
    assert artifact.evidence[0].tool_call_id is None
    assert any("Progressive autonomous budget extended" in note for note in artifact.audit_notes)


def test_provider_usage_splits_openai_physical_response_attempts(monkeypatch):
    class PhysicalResponseProvider(FakeProvider):
        provider_name = "openai"
        model = "gpt-5.5"

        def synthesize_json(self, **kwargs):
            self.calls.append(kwargs)
            payload = self.payloads.pop(0)
            first = {
                "status": "incomplete",
                "model": self.model,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
            second = {
                "status": "completed",
                "model": self.model,
                "usage": {"input_tokens": 100, "output_tokens": 30},
            }
            return SimpleNamespace(
                json_text=json.dumps(payload),
                model=self.model,
                usage_input_tokens=100,
                usage_output_tokens=30,
                usage_cached_input_tokens=0,
                raw={"_response_attempts": [first, second]},
            )

    provider = PhysicalResponseProvider([_plan_payload(), _final_payload("WATCHLIST_ONLY")])
    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "summary": "Decision-usable evidence.",
        },
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=4, max_turns=2),
    )

    assert [item["status"] for item in artifact.provider_usage] == [
        "INCOMPLETE",
        "OK",
        "INCOMPLETE",
        "OK",
    ]
    assert [item["physical_response_sequence"] for item in artifact.provider_usage] == [
        1,
        2,
        1,
        2,
    ]


def test_provider_usage_records_each_observed_failed_physical_attempt(monkeypatch):
    from app.llm.providers.retry_guard import _notify_failed_attempt

    class ObservedRetryProvider(FakeProvider):
        provider_name = "openai"
        model = "gpt-5.5"

        def synthesize_json(self, **kwargs):
            self.calls.append(kwargs)
            _notify_failed_attempt(
                provider=self.provider_name,
                schema_name=str(kwargs.get("schema_name") or "structured_output"),
                attempt=1,
                exc=RuntimeError("transient physical failure"),
                retryable=True,
                will_retry=True,
            )
            payload = self.payloads.pop(0)
            return SimpleNamespace(json_text=json.dumps(payload), model=self.model)

    provider = ObservedRetryProvider([_plan_payload(), _final_payload("WATCHLIST_ONLY")])
    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "summary": "Decision-usable evidence.",
        },
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=4, max_turns=2),
    )

    assert [item["status"] for item in artifact.provider_usage] == [
        "ERROR",
        "OK",
        "ERROR",
        "OK",
    ]
    failed = [item for item in artifact.provider_usage if item["status"] == "ERROR"]
    assert [item["physical_attempt"] for item in failed] == [1, 1]
    assert all(item["will_retry"] is True for item in failed)


def test_billed_success_before_provider_parse_failure_is_not_recast_as_failed_call(
    monkeypatch,
):
    class BilledThenParseFailureProvider:
        provider_name = "openai"
        model = "gpt-5.5"

        def enabled(self):
            return True

        def synthesize_json(self, **kwargs):
            _ = kwargs
            error = RuntimeError("provider JSON expansion failed")
            error._provider_response_attempts = [
                {
                    "status": "completed",
                    "model": self.model,
                    "usage": {"input_tokens": 80, "output_tokens": 20},
                }
            ]
            raise error

    monkeypatch.setattr(
        "app.autonomous.runtime.get_alpha_llm_provider",
        lambda: BilledThenParseFailureProvider(),
    )
    monkeypatch.setattr(
        "app.autonomous.runtime.assemble_signal_packet",
        lambda ticker: _packet(),
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=4, max_turns=2),
    )

    assert artifact.status == "FAILED"
    assert len(artifact.provider_usage) == 1
    assert artifact.provider_usage[0]["status"] == "OK"
    assert artifact.provider_usage[0]["input_tokens"] == 80
    assert artifact.provider_usage[0]["output_tokens"] == 20
    assert artifact.provider_usage[0]["reserved_output_tokens"] == 0


def test_failed_single_child_carries_paid_provider_usage_on_exception(monkeypatch):
    def fail_after_paid_call(*args, **kwargs):
        _ = (args, kwargs)
        record_provider_usage(
            {
                "status": "OK",
                "lane": "company_underwriting",
                "provider": "openai",
                "model": "gpt-5.5",
                "schema_name": "child_plan",
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 25,
                "estimated_tokens": False,
                "cost_estimate_usd": 0.00125,
            }
        )
        raise RuntimeError("child failed after planning")

    monkeypatch.setattr(
        "app.autonomous.runtime._run_single_candidate_autonomous_analysis_impl",
        fail_after_paid_call,
    )

    with pytest.raises(RuntimeError, match="failed after planning") as exc_info:
        run_single_candidate_autonomous_analysis("AAA")

    usage = attached_provider_usage_records(exc_info.value)
    assert len(usage) == 1
    assert usage[0]["provider_call_id"] == "P1"
    assert usage[0]["status"] == "OK"
    assert usage[0]["lane"] == "company_underwriting"


def test_non_usable_tool_output_records_low_confidence_evidence(monkeypatch):
    provider = FakeProvider([_plan_payload(), _final_payload()])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "usable_for_decision": False,
            "evidence_status": "NO_READABLE_NARRATIVE",
            "summary": "Tool ran but did not produce decision-usable evidence.",
        },
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        objective="Decide if AAA is actionable.",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.evidence[0].confidence == "LOW"
    assert artifact.evidence[0].summary == "Tool ran but did not produce decision-usable evidence."
    assert artifact.final_verdict == "WATCHLIST_ONLY"
    assert artifact.degraded_states == ["ACTIONABLE_WITHOUT_DECISION_USABLE_EVIDENCE"]
    assert artifact.candidate_decisions[0].eligible_for_selection is False
    assert artifact.candidate_decisions[0].selection_blockers == ["NO_DECISION_USABLE_EVIDENCE"]


def test_tool_calls_are_capped_by_max_tool_calls(monkeypatch):
    plan = {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Which evidence should change the verdict?",
                "rationale": "Two calls are planned but only one is allowed by budget.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "fetch_kpi_trends",
                        "tool_input": {},
                        "rationale": "First check.",
                    },
                    {
                        "tool_name": "analyze_dilution",
                        "tool_input": {"years": 5},
                        "rationale": "Second check.",
                    },
                ],
            }
        ]
    }
    provider = FakeProvider([plan, _final_payload()])
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(name)
        return {"status": "ok", "ticker": ctx.ticker}

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=1),
    )

    assert dispatched == ["fetch_kpi_trends"]
    assert artifact.final_verdict == "NO_WINNER"
    assert artifact.degraded_states == ["BUDGET_EXHAUSTED"]
    assert artifact.tool_calls[1].status == "SKIPPED_BUDGET_EXHAUSTED"
    assert len(provider.calls) == 1


def test_disallowed_tools_are_skipped_and_recorded(monkeypatch):
    plan = {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Does forbidden evidence alter the thesis?",
                "rationale": "The provider proposed a disallowed tool.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "fetch_transcript_excerpt",
                        "tool_input": {"focus": "guidance"},
                        "rationale": "Not allowed.",
                    },
                    {
                        "tool_name": "fetch_kpi_trends",
                        "tool_input": {},
                        "rationale": "Allowed fallback.",
                    },
                ],
            }
        ]
    }
    provider = FakeProvider([plan, _final_payload("WATCHLIST_ONLY")])
    dispatched: list[str] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(name)
        return {"status": "ok", "ticker": ctx.ticker}

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(max_tool_calls=2),
    )

    assert dispatched == ["fetch_kpi_trends"]
    assert artifact.tool_calls[0].status == "SKIPPED_DISALLOWED_TOOL"
    assert artifact.tool_calls[1].status == "OK"
    assert artifact.degraded_states == ["DISALLOWED_TOOL_SKIPPED"]
    assert artifact.final_verdict == "WATCHLIST_ONLY"


def test_missing_companyfacts_line_items_are_defaulted_before_dispatch(monkeypatch):
    plan = {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Does the companyfacts trend support the thesis?",
                "rationale": "The provider omitted required line items.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "fetch_companyfacts_timeseries",
                        "tool_input": {},
                        "rationale": "Fetch financial history.",
                    }
                ],
            }
        ]
    }
    provider = FakeProvider([plan, _final_payload()])
    dispatched: list[dict] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(tool_input)
        return {"status": "ok", "ticker": ctx.ticker, "series": {}}

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
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
    assert artifact.tool_calls[0].tool_input == dispatched[0]
    assert "Tool input guardrail" in artifact.tool_calls[0].rationale
    assert artifact.tool_calls[0].status == "OK"


def test_companyfacts_metric_alias_is_translated_before_dispatch(monkeypatch):
    plan = {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Does SEC equity history support the balance-sheet thesis?",
                "rationale": "The provider supplied a SEC-style metric alias.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "fetch_companyfacts_timeseries",
                        "tool_input": {"metric": "StockholdersEquity"},
                        "rationale": "Fetch equity history.",
                    }
                ],
            }
        ]
    }
    provider = FakeProvider([plan, _final_payload()])
    dispatched: list[dict] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(tool_input)
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "series": {"equity": [{"fiscal_year": 2025, "value": 123.0}]},
        }

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert dispatched == [{"metric": "StockholdersEquity", "line_items": ["equity"], "years": 5}]
    assert artifact.degraded_states == ["TOOL_INPUT_REPAIRED"]
    assert "translated companyfacts metric aliases" in artifact.tool_calls[0].rationale


def test_explicit_companyfacts_line_items_take_precedence_over_metric_alias(monkeypatch):
    plan = {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Does cash history support liquidity?",
                "rationale": "Explicit line_items should not be replaced by metric intent.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "fetch_companyfacts_timeseries",
                        "tool_input": {"metric": "EarningsPerShareBasic", "line_items": ["cash"]},
                        "rationale": "Fetch cash history.",
                    }
                ],
            }
        ]
    }
    provider = FakeProvider([plan, _final_payload()])
    dispatched: list[dict] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(tool_input)
        return {
            "status": "ok",
            "ticker": ctx.ticker,
            "series": {"cash": [{"fiscal_year": 2025, "value": 50.0}]},
        }

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert dispatched == [{"metric": "EarningsPerShareBasic", "line_items": ["cash"], "years": 5}]
    assert artifact.degraded_states == []
    assert artifact.tool_calls[0].tool_input["line_items"] == ["cash"]


def test_companyfacts_common_gpt_line_item_aliases_are_translated_before_dispatch(monkeypatch):
    plan = {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Do free cash flow and dilution trends support the thesis?",
                "rationale": "The provider supplied common financial aliases.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "fetch_companyfacts_timeseries",
                        "tool_input": {
                            "line_items": [
                                "free_cash_flow",
                                "operating_cash_flow",
                                "diluted_shares",
                                "stock_based_compensation",
                            ]
                        },
                        "rationale": "Fetch financial history.",
                    }
                ],
            }
        ]
    }
    provider = FakeProvider([plan, _final_payload()])
    dispatched: list[dict] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(tool_input)
        return {"status": "ok", "ticker": ctx.ticker, "series": {}}

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert dispatched == [
        {
            "line_items": ["cfo", "capex", "shares_outstanding", "sbc"],
            "years": 5,
        }
    ]
    assert artifact.degraded_states == ["TOOL_INPUT_REPAIRED"]
    assert artifact.tool_calls[0].tool_input["line_items"] == [
        "cfo",
        "capex",
        "shares_outstanding",
        "sbc",
    ]


def test_unsupported_peer_metric_is_not_defaulted_before_dispatch(monkeypatch):
    plan = {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Does combined ratio beat peers?",
                "rationale": "The provider requested an unsupported peer metric.",
                "priority": "HIGH",
                "planned_tool_calls": [
                    {
                        "tool_name": "compare_peer_metric",
                        "tool_input": {"metric": "combined_ratio"},
                        "rationale": "Compare underwriting ratio.",
                    }
                ],
            }
        ]
    }
    provider = FakeProvider([plan, _final_payload()])
    dispatched: list[dict] = []

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    def fake_dispatch(name, tool_input, ctx):
        dispatched.append(tool_input)
        return {
            "status": "unavailable",
            "ticker": ctx.ticker,
            "metric": tool_input["metric"],
            "usable_for_decision": False,
            "summary": "Unsupported metric.",
        }

    monkeypatch.setattr("app.autonomous.runtime.dispatch_alpha_tool", fake_dispatch)

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert dispatched == [{"metric": "combined_ratio"}]
    assert artifact.degraded_states == [
        "TOOL_INPUT_UNSUPPORTED",
        "ACTIONABLE_WITHOUT_DECISION_USABLE_EVIDENCE",
    ]
    assert artifact.final_verdict == "WATCHLIST_ONLY"
    assert artifact.candidate_decisions[0].eligible_for_selection is False
    assert artifact.candidate_decisions[0].selection_blockers == ["NO_DECISION_USABLE_EVIDENCE"]
    assert artifact.evidence[0].confidence == "LOW"


def test_provider_unavailable_returns_no_winner(monkeypatch):
    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: DisabledProvider())
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_WINNER"
    assert artifact.selected_ticker is None
    assert artifact.degraded_states == ["LLM_PROVIDER_UNAVAILABLE"]
    assert "without forcing a verdict" in artifact.no_winner_reason


def test_provider_unavailable_preserves_analyst_context_failure_state(monkeypatch):
    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: DisabledProvider())
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime._build_analyst_context_evidence",
        lambda **kwargs: (
            [],
            None,
            ["ANALYST_CONTEXT_UNAVAILABLE"],
            [
                "Analyst context unavailable; autonomous run continued without seeded analyst report."
            ],
        ),
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_WINNER"
    assert artifact.degraded_states == ["LLM_PROVIDER_UNAVAILABLE", "ANALYST_CONTEXT_UNAVAILABLE"]
    assert artifact.audit_notes[0].startswith("Analyst context unavailable")


def test_out_of_scope_final_selection_is_forced_to_no_winner(monkeypatch):
    final_payload = _final_payload()
    final_payload["selected_ticker"] = "BBB"
    final_payload["candidate_decision"]["ticker"] = "BBB"
    provider = FakeProvider([_plan_payload(), final_payload])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker, "metric": "stable"},
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_WINNER"
    assert artifact.selected_ticker is None
    assert artifact.confidence is None
    assert artifact.degraded_states == ["FINAL_DECISION_OUT_OF_SCOPE"]
    assert artifact.candidate_decisions[0].ticker == "AAA"
    assert artifact.candidate_decisions[0].eligible_for_selection is False
    assert artifact.candidate_decisions[0].selection_blockers == ["FINAL_DECISION_OUT_OF_SCOPE"]
    assert "out-of-scope ticker BBB" in artifact.no_winner_reason


def test_actionable_final_with_selection_blockers_is_downgraded_to_watchlist(monkeypatch):
    final_payload = _final_payload()
    final_payload["candidate_decision"]["eligible_for_selection"] = False
    final_payload["candidate_decision"]["selection_blockers"] = ["MODEL_FIT_BLOCKED"]
    provider = FakeProvider([_plan_payload(), final_payload])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker, "metric": "stable"},
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "WATCHLIST_ONLY"
    assert artifact.selected_ticker == "AAA"
    assert artifact.degraded_states == ["ACTIONABLE_FINAL_DECISION_INCONSISTENT"]
    assert artifact.candidate_decisions[0].verdict == "WATCHLIST_ONLY"
    assert artifact.candidate_decisions[0].eligible_for_selection is False
    assert artifact.candidate_decisions[0].selection_blockers == ["MODEL_FIT_BLOCKED"]


def test_actionable_final_missing_selected_ticker_is_downgraded_to_watchlist(monkeypatch):
    final_payload = _final_payload()
    final_payload["selected_ticker"] = None
    final_payload["confidence"] = None
    provider = FakeProvider([_plan_payload(), final_payload])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker, "metric": "stable"},
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "WATCHLIST_ONLY"
    assert artifact.selected_ticker is None
    assert artifact.confidence is None
    assert artifact.degraded_states == ["ACTIONABLE_FINAL_DECISION_INCOMPLETE"]
    assert artifact.candidate_decisions[0].verdict == "WATCHLIST_ONLY"
    assert artifact.candidate_decisions[0].eligible_for_selection is False
    assert artifact.candidate_decisions[0].selection_blockers == [
        "ACTIONABLE_FINAL_DECISION_INCOMPLETE"
    ]


def test_legacy_buy_final_verdict_is_normalized_to_actionable(monkeypatch):
    final_payload = _final_payload()
    final_payload["final_verdict"] = "BUY"
    final_payload["candidate_decision"]["verdict"] = "BUY"
    provider = FakeProvider([_plan_payload(), final_payload])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker, "metric": "stable"},
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "ACTIONABLE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.degraded_states == ["FINAL_VERDICT_ALIAS_NORMALIZED"]
    assert artifact.candidate_decisions[0].verdict == "ACTIONABLE"
    assert artifact.candidate_decisions[0].eligible_for_selection is True


def test_invalid_final_verdict_is_forced_to_no_winner(monkeypatch):
    final_payload = _final_payload()
    final_payload["final_verdict"] = "MAYBE"
    final_payload["candidate_decision"]["verdict"] = "MAYBE"
    provider = FakeProvider([_plan_payload(), final_payload])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker, "metric": "stable"},
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_WINNER"
    assert artifact.selected_ticker is None
    assert artifact.confidence is None
    assert artifact.degraded_states == ["FINAL_VERDICT_INVALID"]
    assert artifact.candidate_decisions[0].verdict == "NO_WINNER"
    assert artifact.candidate_decisions[0].eligible_for_selection is False
    assert artifact.candidate_decisions[0].selection_blockers == ["FINAL_VERDICT_INVALID"]
    assert artifact.no_winner_reason == "Provider returned invalid final_verdict MAYBE."


def test_no_winner_final_clears_selected_ticker_and_confidence(monkeypatch):
    final_payload = _final_payload("NO_WINNER")
    final_payload["selected_ticker"] = "AAA"
    final_payload["confidence"] = "LOW"
    final_payload["candidate_decision"]["eligible_for_selection"] = True
    final_payload["candidate_decision"]["selection_blockers"] = []
    provider = FakeProvider([_plan_payload(), final_payload])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker, "metric": "stable"},
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_WINNER"
    assert artifact.selected_ticker is None
    assert artifact.confidence is None
    assert artifact.degraded_states == ["NO_WINNER_FINAL_DECISION_INCONSISTENT"]
    assert artifact.candidate_decisions[0].verdict == "NO_WINNER"
    assert artifact.candidate_decisions[0].eligible_for_selection is False
    assert artifact.candidate_decisions[0].selection_blockers == [
        "NO_WINNER_FINAL_DECISION_INCONSISTENT"
    ]


def test_packet_assembly_failure_returns_no_winner(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.runtime.assemble_signal_packet",
        lambda ticker: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )

    assert artifact.final_verdict == "NO_WINNER"
    assert artifact.degraded_states == ["PACKET_ASSEMBLY_FAILED"]
    assert "boom" in artifact.no_winner_reason


def test_canonical_v2_child_uses_bound_packet_without_global_latest_refresh(
    monkeypatch,
):
    packet = _packet()
    source_binding = {
        "artifact_type": "v2_canonical_child_source_binding_v1",
        "pipeline_version": "v2",
        "sector": "energy",
        "ticker": "AAA",
        "as_of_date": "2026-04-25",
        "signal_packet_fingerprint": _signal_packet_fingerprint(packet),
        "company_packet_fingerprint": "a" * 64,
        "cohort_fingerprint": "b" * 64,
    }
    monkeypatch.setattr(
        "app.autonomous.runtime.assemble_signal_packet",
        lambda ticker: (_ for _ in ()).throw(
            AssertionError("canonical child must not read the global latest packet")
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.runtime._build_analyst_context_evidence",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("canonical child must not refresh global analyst context")
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.runtime.get_alpha_llm_provider",
        lambda: DisabledProvider(),
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
        canonical_signal_packet=packet,
        source_binding=source_binding,
    )

    assert artifact.degraded_states == ["LLM_PROVIDER_UNAVAILABLE"]
    assert artifact.request.candidate_scope["source_binding"] == source_binding
    assert artifact.audit_notes[0] == (
        "Canonical child analysis used the immutable sector packet and did not refresh "
        "global analyst context."
    )


def test_canonical_v1_child_rejects_embedded_company_packet_drift_before_provider(
    monkeypatch,
):
    packet = _packet()
    provider = FakeProvider([_plan_payload(), _final_payload()])
    financial_packet = {
        "ticker": "AAA",
        "financial_status": "COMPLETE",
        "model_fit_status": "VALID_GENERIC",
        "data_quality_status": "OK",
    }
    scenarios: list[dict] = []
    source_binding = {
        "artifact_type": "v1_canonical_child_source_binding_v1",
        "pipeline_version": "v1",
        "sector": "energy",
        "ticker": "AAA",
        "as_of_date": "2026-04-25",
        "signal_packet_fingerprint": _signal_packet_fingerprint(packet),
        "company_packet_fingerprint": _financial_payload_fingerprint(financial_packet),
        "financial_integrity_packet": {**financial_packet, "current_price": 999.0},
        "financial_integrity_scenarios": scenarios,
        "financial_integrity_scenarios_fingerprint": _financial_payload_fingerprint(scenarios),
        "cohort_fingerprint": "b" * 64,
    }
    monkeypatch.setattr(
        "app.autonomous.runtime.get_alpha_llm_provider",
        lambda: provider,
    )

    with pytest.raises(ValueError, match="company packet fingerprint drift"):
        run_single_candidate_autonomous_analysis(
            ticker="AAA",
            as_of_date="2026-04-25",
            budget=_budget(),
            canonical_signal_packet=packet,
            source_binding=source_binding,
        )

    assert provider.calls == []


def test_canonical_v2_child_rejects_signal_fingerprint_drift(monkeypatch):
    packet = _packet()
    monkeypatch.setattr(
        "app.autonomous.runtime.get_alpha_llm_provider",
        lambda: DisabledProvider(),
    )

    with pytest.raises(ValueError, match="signal packet fingerprint drift"):
        run_single_candidate_autonomous_analysis(
            ticker="AAA",
            as_of_date="2026-04-25",
            budget=_budget(),
            canonical_signal_packet=packet,
            source_binding={
                "ticker": "AAA",
                "as_of_date": "2026-04-25",
                "signal_packet_fingerprint": "0" * 64,
            },
        )


def test_final_artifact_round_trips(monkeypatch):
    provider = FakeProvider([_plan_payload(), _final_payload()])

    monkeypatch.setattr("app.autonomous.runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr("app.autonomous.runtime.assemble_signal_packet", lambda ticker: _packet())
    monkeypatch.setattr(
        "app.autonomous.runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {"status": "ok", "ticker": ctx.ticker, "metric": "stable"},
    )

    artifact = run_single_candidate_autonomous_analysis(
        ticker="AAA",
        as_of_date="2026-04-25",
        budget=_budget(),
    )
    restored = AutonomousRunArtifact.from_dict(artifact.to_dict())

    assert restored == artifact
    assert restored.contract_version == "autonomous_analyst_run_v1"
