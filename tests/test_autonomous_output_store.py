from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _legacy_packet_fixture_bypasses_new_integrity_gate(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.output_store.artifact_decision_eligibility", lambda payload: "PASS"
    )
    monkeypatch.setattr(
        "app.autonomous.output_store.write_run_financial_authorization",
        lambda artifact_path, _report_path, **_kwargs: (
            artifact_path.parent / "financial_integrity_authorization.json"
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.output_store.bind_authorized_valuation_rows",
        lambda **_kwargs: 0,
    )
    monkeypatch.setattr(
        "app.autonomous.output_store._require_publication_financial_integrity_binding",
        lambda *args, **_kwargs: None,
    )


from app.autonomous.output_store import (
    persist_autonomous_run,
    persist_autonomous_sector_diagnostic,
    persist_autonomous_sector_run,
)
from app.autonomous.competitive_frontier import build_competitive_frontier
from app.autonomous.run_contract import (
    AutonomousRunArtifact,
    AutonomousRunBudget,
    AutonomousRunRequest,
    EvidenceReference,
    ToolCallRecord,
)
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    CandidateDisposition,
    SECTOR_CONTRACT_VERSION_V2,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorSelectionValidation,
    UnderwritingResult,
    build_v2_canonical_child_source_bindings,
)


def _frontier_packet() -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="READY",
        model_fit_status="SUPPORTED",
        data_quality_status="COMPLETE",
        score_components={"deterministic_score": 1.0},
    )


def _frontier_scenario() -> SectorExpectedReturnScenario:
    return SectorExpectedReturnScenario(
        scenario_id="AAA-base",
        ticker="AAA",
        scenario_name="base",
        horizon_years=5,
        current_price=10.0,
        estimated_future_value_per_share=20.0,
        annualized_return=0.15,
    )


def _selected_signal_snapshot() -> dict[str, str]:
    return {"sector": "energy", "as_of_date": "2026-07-16", "ticker": "AAA"}


def _selected_source_binding() -> dict[str, object]:
    packet = _frontier_packet()
    return build_v2_canonical_child_source_bindings(
        sector="energy",
        as_of_date="2026-07-16",
        company_packets=[packet],
        scenarios=[_frontier_scenario()],
        signal_packet_snapshots={"AAA": _selected_signal_snapshot()},
        frontier_candidate_tickers=["AAA"],
    )["AAA"]


def _closed_frontier_payload() -> dict:
    packet = _frontier_packet()
    scenario = _frontier_scenario()
    state = build_competitive_frontier(
        [packet],
        [scenario],
        reviewed_tickers=["AAA"],
    )
    payload = state.to_dict()
    payload.update(
        {
            "status": "CLOSED",
            "minimum_reviews_required": 1,
            "successful_review_count": 1,
            "attempted_tickers": ["AAA"],
            "failed_review_tickers": [],
            "source_bindings": {"AAA": _selected_source_binding()},
            "signal_packet_snapshots": {"AAA": _selected_signal_snapshot()},
        }
    )
    return payload


def _bound_underwriting_fixture(
    *, tool_calls: list[ToolCallRecord] | None = None
) -> tuple[UnderwritingResult, dict[str, object]]:
    binding = _selected_source_binding()
    nested = {
        "request": {
            "run_id": "child-aaa",
            "as_of_date": "2026-07-16",
            "candidate_scope": {
                "mode": "single_candidate",
                "tickers": ["AAA"],
                "source_binding": binding,
                "signal_packet_snapshot": _selected_signal_snapshot(),
            },
        },
        "tool_calls": [call.to_dict() for call in tool_calls or []],
    }
    return (
        UnderwritingResult(
            status="COMPLETED",
            verdict="ACTIONABLE",
            confidence="HIGH",
            evidence_ref_ids=["child-aaa:E1"],
            tool_call_ids=["TC1"],
            child_run_id="child-aaa",
        ),
        {
            "ticker": "AAA",
            "run_id": "child-aaa",
            "status": "COMPLETED",
            "final_verdict": "ACTIONABLE",
            "source_binding": binding,
            "artifact": nested,
            "attempts": [nested],
        },
    )


def _request() -> AutonomousRunRequest:
    return AutonomousRunRequest(
        run_id="autonomous_AAA_20260506_test",
        objective="Research AAA.",
        as_of_date="2026-05-06",
        created_at="2026-05-06T20:00:00Z",
        candidate_scope={"mode": "single_candidate", "tickers": ["AAA"]},
        allowed_tools=["fetch_kpi_trends"],
        budget=AutonomousRunBudget(
            max_tool_calls=4,
            max_turns=3,
            max_cost_usd=1.0,
            timebox_seconds=None,
            max_candidates=1,
        ),
    )


def _ok_tool() -> ToolCallRecord:
    return ToolCallRecord(
        call_id="TC1",
        tool_name="fetch_kpi_trends",
        tool_input={},
        rationale="Validate decision-useful facts.",
        status="OK",
        evidence_ref_ids=["E1"],
    )


def test_persist_autonomous_run_refuses_zero_successful_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    artifact = AutonomousRunArtifact(
        request=_request(),
        status="COMPLETED",
        started_at="2026-05-06T20:00:00Z",
        completed_at="2026-05-06T20:01:00Z",
        final_verdict="NO_WINNER",
        selected_ticker=None,
        confidence=None,
        tool_calls=[],
    )

    with pytest.raises(RuntimeError, match="without successful tool calls"):
        persist_autonomous_run(artifact)

    assert not (tmp_path / "data" / "outputs" / "runs" / "autonomous").exists()


def test_persist_autonomous_run_writes_json_and_markdown_after_tool_call(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    artifact = AutonomousRunArtifact(
        request=_request(),
        status="COMPLETED",
        started_at="2026-05-06T20:00:00Z",
        completed_at="2026-05-06T20:01:00Z",
        final_verdict="WATCHLIST_ONLY",
        selected_ticker=None,
        confidence="LOW",
        tool_calls=[_ok_tool()],
    )

    paths = persist_autonomous_run(artifact)

    assert paths.artifact_json.exists()
    assert paths.report_md.exists()


def test_persist_autonomous_sector_run_refuses_failed_or_zero_tool_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    failed = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_test_20260506_test",
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        objective="Research sector.",
        as_of_date="2026-05-06",
        created_at="2026-05-06T20:00:00Z",
        status="FAILED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        tool_calls=[_ok_tool()],
    )
    zero_tool = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_test_20260506_zero",
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        objective="Research sector.",
        as_of_date="2026-05-06",
        created_at="2026-05-06T20:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        tool_calls=[],
    )

    with pytest.raises(RuntimeError, match="non-success status"):
        persist_autonomous_sector_run(failed)
    with pytest.raises(RuntimeError, match="without successful tool calls"):
        persist_autonomous_sector_run(zero_tool)


def test_persist_autonomous_sector_diagnostic_writes_json_only(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_v2_interrupted",
        sector="hospitality_gaming",
        market_cap_focus="large_and_mega",
        objective="Preserve a truthful interrupted attempt.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T20:00:00Z",
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
                review_status="FAILED",
                watchlist_eligible=True,
                reason_codes=["PROVIDER_FAILURE"],
                last_completed_stage="filings",
            )
        ],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    paths = persist_autonomous_sector_diagnostic(artifact)
    payload = json.loads(paths.artifact_json.read_text(encoding="utf-8"))

    assert paths.artifact_json == (
        tmp_path
        / "data"
        / "outputs"
        / "runs"
        / "autonomous_sector_diagnostics"
        / "autonomous_sector_v2_interrupted"
        / "autonomous_sector_diagnostic.json"
    )
    assert payload["pipeline_version"] == "v2"
    assert payload["execution_status"] == "FAILED"
    assert payload["decision_status"] == "INCOMPLETE"
    assert payload["final_verdict"] is None
    assert list(paths.artifact_json.parent.glob("*.md")) == []


def test_completed_sector_decision_cannot_be_written_as_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_completed",
        sector="energy",
        market_cap_focus="mid_cap",
        objective="Completed v1 run.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T20:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
    )

    with pytest.raises(RuntimeError, match="completed sector decision"):
        persist_autonomous_sector_diagnostic(artifact)


def test_completed_execution_with_incomplete_v2_decision_is_diagnostic_only(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_v2_incomplete",
        sector="enterprise_software",
        market_cap_focus="large_and_mega",
        objective="Preserve unresolved frontier state.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T20:00:00Z",
        status="COMPLETED",
        final_verdict=None,
        selected_ticker=None,
        confidence=None,
        pipeline_version="v2",
        execution_status="COMPLETED",
        decision_status="INCOMPLETE",
        admitted_tickers=["AAA"],
        candidate_dispositions=[
            CandidateDisposition(
                ticker="AAA",
                terminal_state="READY_FOR_UNDERWRITING",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="NOT_STARTED",
                watchlist_eligible=True,
                last_completed_stage="screen",
            )
        ],
        tool_calls=[_ok_tool()],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    with pytest.raises(RuntimeError, match="incomplete v2 sector decision"):
        persist_autonomous_sector_run(artifact)
    paths = persist_autonomous_sector_diagnostic(artifact)

    assert paths.artifact_json.exists()
    assert list(paths.artifact_json.parent.glob("*.md")) == []


def test_v2_persistence_revalidates_truth_fields_after_construction(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    underwriting, child_run = _bound_underwriting_fixture()
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_v2_mutated",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Reject post-construction truth corruption.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T20:00:00Z",
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
                underwriting_result=underwriting,
            )
        ],
        selection_validation=SectorSelectionValidation(
            status="VALIDATED",
            selected_ticker="AAA",
            source_binding=_selected_source_binding(),
            validator_run_id="validator-aaa",
            validator_verdict="CONFIRMED_ACTIONABLE",
            evidence_ref_ids=["validator-aaa:E-VALIDATION"],
            evidence=[
                EvidenceReference(
                    evidence_id="validator-aaa:E-VALIDATION",
                    source_type="tool_output",
                    source_label="challenge_selected_company",
                    summary="Independent validation evidence.",
                    ticker="AAA",
                    tool_call_id="validator-aaa:V1",
                    confidence="HIGH",
                )
            ],
            tool_calls=[
                ToolCallRecord(
                    call_id="validator-aaa:V1",
                    tool_name="challenge_selected_company",
                    tool_input={"ticker": "AAA"},
                    rationale="Challenge the provisional selection.",
                    status="OK",
                    evidence_ref_ids=["validator-aaa:E-VALIDATION"],
                    lane="selected_company_validation",
                )
            ],
            provider_usage=[
                {
                    "provider_call_id": "validator-aaa:P1",
                    "validator_run_id": "validator-aaa",
                    "lane": "selected_company_validation",
                }
            ],
        ),
        competitive_frontier=_closed_frontier_payload(),
        company_packets=[_frontier_packet()],
        expected_return_scenarios=[_frontier_scenario()],
        tool_calls=[_ok_tool()],
        company_autonomy_runs=[child_run],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )
    artifact.selection_validation.status = "CONTRADICTED"

    with pytest.raises(ValueError, match="matching VALIDATED"):
        persist_autonomous_sector_run(artifact)

    assert not (
        tmp_path / "data" / "outputs" / "runs" / "autonomous_sector" / artifact.run_id
    ).exists()


def test_v2_product_persistence_counts_child_and_validation_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    child_call = _ok_tool()
    validation_call = ToolCallRecord(
        call_id="validator-aaa:V1",
        tool_name="challenge_selected_company",
        tool_input={"ticker": "AAA"},
        rationale="Challenge the provisional selection.",
        status="OK",
        evidence_ref_ids=["validator-aaa:E-VALIDATION"],
        lane="selected_company_validation",
    )
    underwriting, child_run = _bound_underwriting_fixture(tool_calls=[child_call])
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_v2_nested_calls",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Count nested research before product persistence.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T20:00:00Z",
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
                underwriting_result=underwriting,
            )
        ],
        selection_validation=SectorSelectionValidation(
            status="VALIDATED",
            selected_ticker="AAA",
            source_binding=_selected_source_binding(),
            validator_run_id="validator-aaa",
            validator_verdict="CONFIRMED_ACTIONABLE",
            evidence_ref_ids=["validator-aaa:E-VALIDATION"],
            evidence=[
                EvidenceReference(
                    evidence_id="validator-aaa:E-VALIDATION",
                    source_type="tool_output",
                    source_label="challenge_selected_company",
                    summary="Independent validation evidence.",
                    ticker="AAA",
                    tool_call_id="validator-aaa:V1",
                    confidence="HIGH",
                )
            ],
            tool_calls=[validation_call],
            provider_usage=[
                {
                    "provider_call_id": "validator-aaa:P1",
                    "validator_run_id": "validator-aaa",
                    "lane": "selected_company_validation",
                }
            ],
        ),
        competitive_frontier=_closed_frontier_payload(),
        company_packets=[_frontier_packet()],
        expected_return_scenarios=[_frontier_scenario()],
        tool_calls=[],
        company_autonomy_runs=[child_run],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    paths = persist_autonomous_sector_run(artifact)

    assert paths.artifact_json.exists()
    assert paths.report_md.exists()


def test_v1_product_persistence_still_requires_parent_tool_call(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_v1_child_only",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Preserve the v1 product-persistence gate.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T20:00:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        tool_calls=[],
        company_autonomy_runs=[
            {
                "ticker": "AAA",
                "status": "COMPLETED",
                "artifact": {"tool_calls": [_ok_tool().to_dict()]},
            }
        ],
    )

    with pytest.raises(RuntimeError, match="without successful tool calls"):
        persist_autonomous_sector_run(artifact)
