from __future__ import annotations

from app.autonomous import (
    AutonomousRunArtifact,
    AutonomousRunBudget,
    AutonomousRunRequest,
    BeliefUpdate,
    CandidateDecision,
    EvidenceReference,
    ResearchQuestion,
    ToolCallRecord,
)
from app.autonomous.run_contract import CONTRACT_VERSION


def _budget() -> AutonomousRunBudget:
    return AutonomousRunBudget(
        max_tool_calls=24,
        max_turns=8,
        max_cost_usd=3.50,
        timebox_seconds=900,
        max_candidates=5,
    )


def _request() -> AutonomousRunRequest:
    return AutonomousRunRequest(
        run_id="AR-20260425-insurance",
        objective="Find one actionable long idea or stop with no winner.",
        as_of_date="2026-04-25",
        created_at="2026-04-25T22:30:00Z",
        candidate_scope={"sector": "insurance", "tickers": ["AAA", "BBB"]},
        allowed_tools=["fetch_kpi_trends", "fetch_filing_section", "compare_peer_metric"],
        budget=_budget(),
        stop_rules=["stop_if_all_candidates_blocked", "stop_if_budget_exhausted"],
        user_constraints=["primary_sources_only"],
    )


def test_autonomous_run_budget_round_trip():
    budget = _budget()

    restored = AutonomousRunBudget.from_dict(budget.to_dict())

    assert restored == budget
    assert restored.max_tool_calls == 24
    assert restored.max_turns == 8
    assert restored.max_cost_usd == 3.50
    assert restored.timebox_seconds == 900
    assert restored.max_candidates == 5


def test_autonomous_run_request_defaults_and_round_trip():
    request = AutonomousRunRequest(
        run_id="AR-1",
        objective="Investigate a candidate set.",
        as_of_date="2026-04-25",
        created_at="2026-04-25T22:30:00Z",
        candidate_scope={"sector": "insurance"},
        allowed_tools=["fetch_kpi_trends"],
        budget=AutonomousRunBudget(
            max_tool_calls=10,
            max_turns=4,
            max_cost_usd=None,
            timebox_seconds=None,
        ),
    )

    restored = AutonomousRunRequest.from_dict(request.to_dict())

    assert restored == request
    assert restored.stop_rules == []
    assert restored.user_constraints == []
    assert restored.contract_version == CONTRACT_VERSION


def test_research_question_defaults_and_round_trip():
    question = ResearchQuestion(
        question_id="Q1",
        question="Is the apparent discount real or a model-fit artifact?",
        rationale="The prior valuation depends on one sector-specific model.",
        priority="HIGH",
        status="OPEN",
    )

    restored = ResearchQuestion.from_dict(question.to_dict())

    assert restored == question
    assert restored.target_tickers == []
    assert restored.selected_tools == []
    assert restored.depends_on == []


def test_tool_call_record_round_trip():
    record = ToolCallRecord(
        call_id="TC1",
        question_id="Q1",
        tool_name="fetch_filing_section",
        tool_input={"keywords": ["reserve development"], "section_focus": "risk"},
        rationale="Reserve evidence could change candidate eligibility.",
        status="OK",
        started_at="2026-04-25T22:31:00Z",
        completed_at="2026-04-25T22:31:05Z",
        output_preview="Reserve development was favorable in the latest year.",
        output_path=None,
        evidence_ref_ids=["E1"],
    )

    restored = ToolCallRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.tool_input == {"keywords": ["reserve development"], "section_focus": "risk"}
    assert restored.evidence_ref_ids == ["E1"]


def test_evidence_reference_round_trip():
    evidence = EvidenceReference(
        evidence_id="E1",
        source_type="filing",
        source_label="FY2025 10-K Item 7",
        summary="Combined ratio remained below underwriting breakeven.",
        ticker="AAA",
        source_date="2026-02-15",
        source_url="https://example.test/filing",
        excerpt="The combined ratio was 92.9%.",
        tool_call_id="TC1",
        confidence="HIGH",
    )

    restored = EvidenceReference.from_dict(evidence.to_dict())

    assert restored == evidence
    assert restored.excerpt == "The combined ratio was 92.9%."


def test_belief_update_round_trip():
    update = BeliefUpdate(
        update_id="BU1",
        question_id="Q1",
        ticker="AAA",
        prior_belief="Valuation may be overstated because reserve quality is unknown.",
        updated_belief="Reserve evidence reduces, but does not eliminate, the concern.",
        direction="BULLISH",
        confidence_after="MODERATE",
        summary="Primary filing evidence showed no adverse reserve development.",
        evidence_ref_ids=["E1"],
        remaining_uncertainty=["statutory capital still unavailable"],
    )

    restored = BeliefUpdate.from_dict(update.to_dict())

    assert restored == update
    assert restored.remaining_uncertainty == ["statutory capital still unavailable"]


def test_candidate_decision_defaults_and_round_trip():
    decision = CandidateDecision(
        ticker="AAA",
        verdict="WATCH",
        confidence="MODERATE",
        thesis="Discount is real but evidence is incomplete.",
        key_risk="Missing statutory capital evidence.",
        eligible_for_selection=False,
        selection_blockers=["STATUTORY_CAPITAL_UNKNOWN"],
    )

    restored = CandidateDecision.from_dict(decision.to_dict())

    assert restored == decision
    assert restored.falsifiers == []
    assert restored.evidence_ref_ids == []
    assert restored.confidence_cap_reasons == []


def test_autonomous_run_artifact_minimal_round_trip():
    artifact = AutonomousRunArtifact(
        request=_request(),
        status="NO_WINNER",
        started_at="2026-04-25T22:30:00Z",
        completed_at="2026-04-25T22:40:00Z",
        final_verdict="NO_WINNER",
        selected_ticker=None,
        confidence=None,
        no_winner_reason="All candidates lacked required source evidence.",
        degraded_states=["STATUTORY_DATA_UNAVAILABLE"],
        audit_notes=["Run stopped correctly rather than forcing a pick."],
    )

    restored = AutonomousRunArtifact.from_dict(artifact.to_dict())

    assert restored == artifact
    assert restored.questions == []
    assert restored.tool_calls == []
    assert restored.evidence == []
    assert restored.belief_updates == []
    assert restored.candidate_decisions == []
    assert restored.contract_version == CONTRACT_VERSION


def test_autonomous_run_artifact_full_round_trip():
    question = ResearchQuestion(
        question_id="Q1",
        question="Does the candidate have enough reserve evidence to remain eligible?",
        rationale="Reserve gaps should block or cap insurance conviction.",
        priority="HIGH",
        status="ANSWERED",
        target_tickers=["AAA"],
        selected_tools=["fetch_filing_section"],
    )
    tool_call = ToolCallRecord(
        call_id="TC1",
        question_id="Q1",
        tool_name="fetch_filing_section",
        tool_input={"keywords": ["reserve development"]},
        rationale="Need primary-source reserve evidence.",
        status="OK",
        evidence_ref_ids=["E1"],
    )
    evidence = EvidenceReference(
        evidence_id="E1",
        source_type="filing",
        source_label="FY2025 10-K",
        summary="No adverse reserve development was identified.",
        ticker="AAA",
        tool_call_id="TC1",
    )
    belief_update = BeliefUpdate(
        update_id="BU1",
        question_id="Q1",
        ticker="AAA",
        prior_belief="Reserve quality is unknown.",
        updated_belief="Reserve quality is acceptable but statutory capital remains missing.",
        direction="BULLISH",
        confidence_after="MODERATE",
        summary="Reserve evidence no longer blocks the candidate.",
        evidence_ref_ids=["E1"],
    )
    decision = CandidateDecision(
        ticker="AAA",
        verdict="BUY",
        confidence="MODERATE",
        thesis="Evidence supports an actionable but capped idea.",
        key_risk="Statutory capital remains unavailable.",
        eligible_for_selection=True,
        falsifiers=["Combined ratio approaches underwriting breakeven."],
        evidence_ref_ids=["E1"],
        confidence_cap_reasons=["STATUTORY_CAPITAL_UNKNOWN"],
    )
    artifact = AutonomousRunArtifact(
        request=_request(),
        status="COMPLETED",
        started_at="2026-04-25T22:30:00Z",
        completed_at="2026-04-25T22:40:00Z",
        final_verdict="BUY",
        selected_ticker="AAA",
        confidence="MODERATE",
        questions=[question],
        tool_calls=[tool_call],
        evidence=[evidence],
        belief_updates=[belief_update],
        candidate_decisions=[decision],
        audit_notes=["Question selection and tool calls are reconstructable."],
    )

    restored = AutonomousRunArtifact.from_dict(artifact.to_dict())

    assert restored == artifact
    assert restored.questions[0].question_id == "Q1"
    assert restored.tool_calls[0].evidence_ref_ids == ["E1"]
    assert restored.evidence[0].tool_call_id == "TC1"
    assert restored.belief_updates[0].updated_belief.startswith("Reserve quality is acceptable")
    assert restored.candidate_decisions[0].confidence_cap_reasons == ["STATUTORY_CAPITAL_UNKNOWN"]
