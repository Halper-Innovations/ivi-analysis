import hashlib
import json
from pathlib import Path

import pytest

from app.autonomous.run_contract import AutonomousRunBudget, ToolCallRecord
from app.autonomous.competitive_frontier import build_competitive_frontier
from app.autonomous.sector_benchmark import (
    _bind_sector_artifact_digests,
    _can_reuse_resume_result,
    _resume_artifact_compatibility,
    _resume_artifact_pipeline_version,
    _resume_realized_cost_microdollars,
    _lane_usage_totals,
    _build_rollups,
    _fixed_cohort_from_results,
    _run_provider_preflight,
    _run_provider_preflight_for_provider,
    _sector_result_from_artifact,
    _sector_result_skipped_cache,
    _sector_result_skipped_provider,
    _terminal_cap_search_totals,
    _tool_call_totals,
    persist_autonomous_sector_benchmark,
    run_autonomous_sector_benchmark,
)
from app.llm.providers.disabled_provider import LLMResult
from app.llm.providers.retry_guard import call_with_llm_retry_guard
from app.autonomous.sector_candidates import (
    AcceptedCensusRunAuthority,
    SectorCandidateSelection,
)
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    CandidateDisposition,
    GateEvaluation,
    SECTOR_CONTRACT_VERSION_V2,
    ScreenResult,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorSelectionValidation,
    UnderwritingResult,
    build_v2_canonical_child_source_bindings,
)


def _call(call_id: str, status: str, *, question_id: str | None = None) -> ToolCallRecord:
    return ToolCallRecord(
        call_id=call_id,
        tool_name="test_tool",
        tool_input={},
        rationale="Exercise exact benchmark accounting.",
        status=status,
        question_id=question_id,
    )


def _provider_call(
    call_id: str,
    lane: str,
    *,
    status: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    cost_estimate_usd: float,
    reserved_output_tokens: int = 0,
) -> dict:
    return {
        "provider_call_id": call_id,
        "lane": lane,
        "provider": "openai",
        "model": "gpt-5.5",
        "status": status,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "cost_estimate_usd": cost_estimate_usd,
    }


def _lane_usage_with_parent_cost(cost_microdollars: int) -> dict:
    zero = {
        "tool_call_attempts": 0,
        "tool_calls_ok": 0,
        "tool_calls_failed": 0,
        "provider_call_attempts": 0,
        "provider_calls_ok": 0,
        "provider_calls_failed": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reserved_output_tokens": 0,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }
    lanes = {
        lane: dict(zero)
        for lane in (
            "provider_preflight",
            "parent_research",
            "company_underwriting",
            "selected_company_validation",
            "repair_fallback",
            "terminal_cap_search",
        )
    }
    lanes["parent_research"]["cost_microdollars"] = cost_microdollars
    lanes["parent_research"]["cost_usd"] = f"{cost_microdollars / 1_000_000:.6f}"
    aggregate = dict(zero)
    aggregate["cost_microdollars"] = cost_microdollars
    aggregate["cost_usd"] = f"{cost_microdollars / 1_000_000:.6f}"
    return {
        "currency": "USD",
        "cost_unit": "microdollars",
        "lanes": lanes,
        "aggregate": aggregate,
        "aggregate_reconciles": True,
    }


def test_v2_benchmark_binds_referenced_artifact_bytes_and_persists_fixed_cohort(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "sector.json"
    artifact_path.write_text(
        json.dumps(
            {
                "run_id": "sector-run-1",
                "sector": "energy",
                "candidate_dispositions": [
                    {"ticker": "AAA", "issuer_cik": "0000000001"},
                    {"ticker": "AAA.B", "issuer_cik": "0000000001"},
                    {"ticker": "BBB", "issuer_key": "issuer-bbb"},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    results = [
        {
            "sector": "energy",
            "run_id": "sector-run-1",
            "artifact_path": str(artifact_path),
        }
    ]

    _bind_sector_artifact_digests(results)
    fixed = _fixed_cohort_from_results(results)

    assert results[0]["artifact_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert fixed == {
        "schema_version": "all_sector_v2_fixed_cohort_v1",
        "security_count": 3,
        "issuer_count": 2,
        "unresolved_issuer_identity": [],
        "sector_artifacts": [
            {
                "sector": "energy",
                "run_id": "sector-run-1",
                "artifact_path": str(artifact_path),
                "artifact_sha256": results[0]["artifact_sha256"],
            }
        ],
        "trusted_manifest_policy": ("CALLER_OR_CONFIG_PINNED_NOT_BENCHMARK_SELF_DECLARED"),
    }


def test_benchmark_totals_include_parent_repair_child_and_validation_calls_exactly() -> None:
    child_calls = [_call("C1", "OK"), _call("C2", "OK"), _call("C3", "ERROR")]
    validation_calls = [_call("V1", "OK"), _call("V2", "ERROR")]
    packet = SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="READY",
        model_fit_status="SUPPORTED",
        data_quality_status="COMPLETE",
        score_components={"deterministic_score": 1.0},
    )
    scenario = SectorExpectedReturnScenario(
        scenario_id="AAA-base",
        ticker="AAA",
        scenario_name="base",
        horizon_years=5,
        current_price=10.0,
        estimated_future_value_per_share=20.0,
        annualized_return=0.15,
    )
    frontier = build_competitive_frontier([packet], [scenario], reviewed_tickers=["AAA"]).to_dict()
    frontier.update(
        {
            "status": "CLOSED",
            "minimum_reviews_required": 1,
            "successful_review_count": 1,
            "attempted_tickers": ["AAA"],
            "failed_review_tickers": [],
        }
    )
    signal_snapshot = {
        "ticker": "AAA",
        "sector": "energy",
        "as_of_date": "2026-07-15",
    }
    source_binding = build_v2_canonical_child_source_bindings(
        sector="energy",
        as_of_date="2026-07-15",
        company_packets=[packet],
        scenarios=[scenario],
        signal_packet_snapshots={"AAA": signal_snapshot},
        frontier_candidate_tickers=["AAA"],
    )["AAA"]
    frontier["source_bindings"] = {"AAA": source_binding}
    frontier["signal_packet_snapshots"] = {"AAA": signal_snapshot}
    child_request = {
        "run_id": "child-aaa",
        "as_of_date": "2026-07-15",
        "candidate_scope": {
            "mode": "single_candidate",
            "tickers": ["AAA"],
            "source_binding": source_binding,
            "signal_packet_snapshot": signal_snapshot,
        },
    }
    child_attempt_1 = {
        "request": child_request,
        "tool_calls": [child_calls[0].to_dict()],
        "provider_usage": [
            _provider_call(
                "CP1",
                "company_underwriting",
                status="OK",
                input_tokens=300,
                cached_input_tokens=30,
                output_tokens=30,
                cost_estimate_usd=0.01,
            )
        ],
    }
    child_attempt_2 = {
        "request": child_request,
        "tool_calls": [call.to_dict() for call in child_calls[1:]],
        "provider_usage": [
            _provider_call(
                "CP2",
                "company_underwriting",
                status="ERROR",
                input_tokens=400,
                cached_input_tokens=40,
                output_tokens=40,
                cost_estimate_usd=0.02,
            )
        ],
    }
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_benchmark_accounting",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Count every research lane.",
        as_of_date="2026-07-15",
        created_at="2026-07-15T20:00:00Z",
        completed_at="2026-07-15T20:01:00Z",
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
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="ACTIONABLE",
                watchlist_eligible=True,
                last_completed_stage="UNDERWRITING",
                frontier_status="REVIEWED",
                underwriting_result=UnderwritingResult(
                    status="COMPLETED",
                    verdict="ACTIONABLE",
                    confidence="HIGH",
                    evidence_ref_ids=["child-aaa:E1"],
                    tool_call_ids=["C1"],
                    child_run_id="child-aaa",
                ),
            )
        ],
        selection_validation=SectorSelectionValidation(
            status="INCOMPLETE",
            selected_ticker="AAA",
            tool_calls=validation_calls,
            provider_usage=[
                _provider_call(
                    "VP1",
                    "selected_company_validation",
                    status="OK",
                    input_tokens=500,
                    cached_input_tokens=50,
                    output_tokens=50,
                    cost_estimate_usd=0.003,
                )
            ],
        ),
        tool_calls=[
            _call("P1", "OK", question_id="Q1"),
            _call("R1", "ERROR", question_id="AGR1"),
            _call("P2", "PLANNED", question_id="Q2"),
            _call("P3", "SKIPPED_BUDGET_EXHAUSTED", question_id="Q3"),
        ],
        company_autonomy_runs=[
            {
                "ticker": "AAA",
                "run_id": "child-aaa",
                "status": "COMPLETED",
                "final_verdict": "ACTIONABLE",
                "source_binding": source_binding,
                "tool_calls": 2,
                "tool_call_attempts": 3,
                "attempts": [child_attempt_1, child_attempt_2],
                # The compact final artifact duplicates attempt two. Accounting
                # must use the attempts list and count the final call only once.
                "artifact": child_attempt_2,
                "provider_usage": [
                    *child_attempt_1["provider_usage"],
                    *child_attempt_2["provider_usage"],
                ],
            }
        ],
        competitive_frontier=frontier,
        company_packets=[packet],
        expected_return_scenarios=[scenario],
        provider_usage=[
            _provider_call(
                "PP1",
                "parent_research",
                status="OK",
                input_tokens=100,
                cached_input_tokens=20,
                output_tokens=10,
                cost_estimate_usd=0.001,
            ),
            _provider_call(
                "RP1",
                "repair_fallback",
                status="ERROR",
                input_tokens=200,
                cached_input_tokens=0,
                output_tokens=20,
                cost_estimate_usd=0.002,
            ),
        ],
        candidate_selection={
            "data_gap_repair": {
                "terminal_cap_search_usage_records": [
                    {
                        "lane": "terminal_cap_search",
                        "call_type": "responses_model",
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "ticker": "AAA",
                        "attempt_status": "RESOLVED",
                        "input_tokens": 2000,
                        "cached_input_tokens": 1000,
                        "output_tokens": 1000,
                        "cost_estimate_usd": 0.0355,
                    },
                    {
                        "lane": "terminal_cap_search",
                        "call_type": "web_search_call",
                        "provider": "openai",
                        "ticker": "AAA",
                        "attempt_status": "RESOLVED",
                        "call_id": "WS1",
                        "cost_estimate_usd": 0.01,
                    },
                ],
                "terminal_cap_search_accounting": {
                    "lane_totals": {
                        "terminal_cap_search": {
                            "response_calls": 1,
                            "web_search_calls": 1,
                            "input_tokens": 2000,
                            "cached_input_tokens": 1000,
                            "output_tokens": 1000,
                            "cost_estimate_usd": 0.0455,
                        }
                    },
                    "aggregate": {
                        "response_calls": 1,
                        "web_search_calls": 1,
                        "input_tokens": 2000,
                        "cached_input_tokens": 1000,
                        "output_tokens": 1000,
                        "cost_estimate_usd": 0.0455,
                    },
                    "aggregate_reconciles": True,
                    "ledger_path": "/tmp/terminal-cap-ledger.json",
                },
            }
        },
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    row = _sector_result_from_artifact(
        artifact=artifact,
        artifact_path=Path("diagnostic.json"),
        report_path=None,
    )

    assert row["tool_call_lane_counts"] == {
        "provider_preflight": 0,
        "parent_research": 1,
        "company_underwriting": 3,
        "selected_company_validation": 2,
        "repair_fallback": 1,
        "terminal_cap_search": 1,
        "aggregate": 8,
    }
    assert row["tool_call_status_counts"] == {"OK": 5, "ERROR": 3}
    assert row["tool_call_diagnostic_status_counts"] == {
        "OK": 5,
        "ERROR": 3,
        "PLANNED": 1,
        "SKIPPED_BUDGET_EXHAUSTED": 1,
    }
    assert row["company_underwriting_tool_call_status_counts"] == {
        "OK": 2,
        "ERROR": 1,
    }
    assert row["tool_call_counts"] == {"test_tool": 7, "web_search": 1}
    assert row["tool_failure_counts"] == {"test_tool:ERROR": 3}
    assert {(item["call_id"], item["lane"]) for item in row["failed_tool_calls"]} == {
        ("R1", "repair_fallback"),
        ("C3", "company_underwriting"),
        ("V2", "selected_company_validation"),
    }
    assert all(item["call_id"] not in {"P2", "P3"} for item in row["failed_tool_calls"])
    assert row["provider_call_status_counts"] == {"OK": 4, "ERROR": 2}
    assert row["provider_call_counts"] == {"openai": 6}
    assert row["selected_company_validation_tool_call_status_counts"] == {
        "OK": 1,
        "ERROR": 1,
    }
    assert "selected_validation_tool_call_status_counts" not in row
    assert row["provider_call_lane_counts"] == {
        "provider_preflight": 0,
        "parent_research": 1,
        "company_underwriting": 2,
        "selected_company_validation": 1,
        "repair_fallback": 1,
        "terminal_cap_search": 1,
        "aggregate": 6,
    }
    assert row["lane_accounting_source"] == "artifact_records_rebuilt"
    assert row["lane_usage"] == {
        "currency": "USD",
        "cost_unit": "microdollars",
        "lanes": {
            "provider_preflight": {
                "tool_call_attempts": 0,
                "tool_calls_ok": 0,
                "tool_calls_failed": 0,
                "provider_call_attempts": 0,
                "provider_calls_ok": 0,
                "provider_calls_failed": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reserved_output_tokens": 0,
                "cost_microdollars": 0,
                "cost_usd": "0.000000",
            },
            "parent_research": {
                "tool_call_attempts": 1,
                "tool_calls_ok": 1,
                "tool_calls_failed": 0,
                "provider_call_attempts": 1,
                "provider_calls_ok": 1,
                "provider_calls_failed": 0,
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "reserved_output_tokens": 0,
                "cost_microdollars": 1000,
                "cost_usd": "0.001000",
            },
            "company_underwriting": {
                "tool_call_attempts": 3,
                "tool_calls_ok": 2,
                "tool_calls_failed": 1,
                "provider_call_attempts": 2,
                "provider_calls_ok": 1,
                "provider_calls_failed": 1,
                "input_tokens": 700,
                "cached_input_tokens": 70,
                "output_tokens": 70,
                "reserved_output_tokens": 0,
                "cost_microdollars": 30000,
                "cost_usd": "0.030000",
            },
            "selected_company_validation": {
                "tool_call_attempts": 2,
                "tool_calls_ok": 1,
                "tool_calls_failed": 1,
                "provider_call_attempts": 1,
                "provider_calls_ok": 1,
                "provider_calls_failed": 0,
                "input_tokens": 500,
                "cached_input_tokens": 50,
                "output_tokens": 50,
                "reserved_output_tokens": 0,
                "cost_microdollars": 3000,
                "cost_usd": "0.003000",
            },
            "repair_fallback": {
                "tool_call_attempts": 1,
                "tool_calls_ok": 0,
                "tool_calls_failed": 1,
                "provider_call_attempts": 1,
                "provider_calls_ok": 0,
                "provider_calls_failed": 1,
                "input_tokens": 200,
                "cached_input_tokens": 0,
                "output_tokens": 20,
                "reserved_output_tokens": 0,
                "cost_microdollars": 2000,
                "cost_usd": "0.002000",
            },
            "terminal_cap_search": {
                "tool_call_attempts": 1,
                "tool_calls_ok": 1,
                "tool_calls_failed": 0,
                "provider_call_attempts": 1,
                "provider_calls_ok": 1,
                "provider_calls_failed": 0,
                "input_tokens": 2000,
                "cached_input_tokens": 1000,
                "output_tokens": 1000,
                "reserved_output_tokens": 0,
                "cost_microdollars": 45500,
                "cost_usd": "0.045500",
            },
        },
        "aggregate": {
            "tool_call_attempts": 8,
            "tool_calls_ok": 5,
            "tool_calls_failed": 3,
            "provider_call_attempts": 6,
            "provider_calls_ok": 4,
            "provider_calls_failed": 2,
            "input_tokens": 3500,
            "cached_input_tokens": 1140,
            "output_tokens": 1150,
            "reserved_output_tokens": 0,
            "cost_microdollars": 81500,
            "cost_usd": "0.081500",
        },
        "aggregate_reconciles": True,
    }
    totals = _tool_call_totals([row])
    assert totals == {
        "tool_calls_total": 8,
        "tool_calls_ok": 5,
        "tool_calls_failed": 3,
        "tool_call_failure_rate": 0.375,
        "lane_totals": {
            "provider_preflight": 0,
            "company_underwriting": 3,
            "parent_research": 1,
            "repair_fallback": 1,
            "selected_company_validation": 2,
            "terminal_cap_search": 1,
        },
        "lane_sum": 8,
        "aggregate_reconciles": True,
    }
    usage_rollup = _lane_usage_totals([row])
    assert usage_rollup["accounted_sector_count"] == 1
    assert usage_rollup["aggregate"] == {
        "tool_call_attempts": 8,
        "tool_calls_ok": 5,
        "tool_calls_failed": 3,
        "provider_call_attempts": 6,
        "provider_calls_ok": 4,
        "provider_calls_failed": 2,
        "input_tokens": 3500,
        "cached_input_tokens": 1140,
        "output_tokens": 1150,
        "reserved_output_tokens": 0,
        "cost_microdollars": 81500,
        "cost_usd": "0.081500",
    }
    assert usage_rollup["aggregate_reconciles"] is True
    two_sector_rollup = _lane_usage_totals([row, row])
    assert two_sector_rollup["accounted_sector_count"] == 2
    assert two_sector_rollup["aggregate"] == {
        "tool_call_attempts": 16,
        "tool_calls_ok": 10,
        "tool_calls_failed": 6,
        "provider_call_attempts": 12,
        "provider_calls_ok": 8,
        "provider_calls_failed": 4,
        "input_tokens": 7000,
        "cached_input_tokens": 2280,
        "output_tokens": 2300,
        "reserved_output_tokens": 0,
        "cost_microdollars": 163000,
        "cost_usd": "0.163000",
    }
    assert two_sector_rollup["aggregate_reconciles"] is True
    assert _terminal_cap_search_totals([row]) == {
        "lane_totals": {
            "terminal_cap_search": {
                "response_calls": 1,
                "web_search_calls": 1,
                "input_tokens": 2000,
                "cached_input_tokens": 1000,
                "output_tokens": 1000,
                "cost_estimate_usd": 0.0455,
            }
        },
        "aggregate": {
            "response_calls": 1,
            "web_search_calls": 1,
            "input_tokens": 2000,
            "cached_input_tokens": 1000,
            "output_tokens": 1000,
            "cost_estimate_usd": 0.0455,
        },
        "aggregate_reconciles": True,
        "ledger_paths": ["/tmp/terminal-cap-ledger.json"],
    }


def test_v2_benchmark_rebuilds_when_persisted_lane_usage_drifts_from_records() -> None:
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_persisted_lane_accounting",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Prefer the runtime ledger over benchmark reconstruction.",
        as_of_date="2026-07-15",
        created_at="2026-07-15T20:00:00Z",
        completed_at="2026-07-15T20:01:00Z",
        status="COMPLETED",
        final_verdict=None,
        selected_ticker=None,
        confidence=None,
        pipeline_version="v2",
        execution_status="COMPLETED",
        decision_status="INCOMPLETE",
        tool_calls=[_call("FALLBACK_ONLY", "OK")],
        lane_usage={
            "summary": {
                "lanes": {
                    "parent_research": {
                        "tool_call_attempts": 4,
                        "tool_calls_ok": 3,
                        "tool_calls_failed": 1,
                        "provider_call_attempts": 2,
                        "provider_calls_ok": 1,
                        "provider_calls_failed": 1,
                        "input_tokens": 1000,
                        "cached_input_tokens": 100,
                        "output_tokens": 200,
                        "cost_microdollars": 125000,
                    },
                    "selected_validation": {
                        "tool_call_attempts": 2,
                        "tool_calls_ok": 2,
                        "tool_calls_failed": 0,
                        "provider_call_attempts": 1,
                        "provider_calls_ok": 1,
                        "provider_calls_failed": 0,
                        "input_tokens": 200,
                        "cached_input_tokens": 20,
                        "output_tokens": 50,
                        "cost_microdollars": 25000,
                    },
                },
                "aggregate": {"tool_call_attempts": 999},
                "aggregate_reconciles": False,
            }
        },
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    row = _sector_result_from_artifact(
        artifact=artifact,
        artifact_path=None,
        report_path=None,
    )

    assert row["lane_accounting_source"] == "artifact_lane_usage_mismatch_rebuilt"
    assert row["tool_call_lane_counts"] == {
        "provider_preflight": 0,
        "parent_research": 1,
        "company_underwriting": 0,
        "selected_company_validation": 0,
        "repair_fallback": 0,
        "terminal_cap_search": 0,
        "aggregate": 1,
    }
    assert row["provider_call_lane_counts"] == {
        "provider_preflight": 0,
        "parent_research": 0,
        "company_underwriting": 0,
        "selected_company_validation": 0,
        "repair_fallback": 0,
        "terminal_cap_search": 0,
        "aggregate": 0,
    }
    assert row["lane_usage"]["aggregate"] == {
        "tool_call_attempts": 1,
        "tool_calls_ok": 1,
        "tool_calls_failed": 0,
        "provider_call_attempts": 0,
        "provider_calls_ok": 0,
        "provider_calls_failed": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reserved_output_tokens": 0,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }
    assert row["lane_usage"]["aggregate_reconciles"] is True
    assert _tool_call_totals([row]) == {
        "tool_calls_total": 1,
        "tool_calls_ok": 1,
        "tool_calls_failed": 0,
        "tool_call_failure_rate": 0.0,
        "lane_totals": {
            "company_underwriting": 0,
            "parent_research": 1,
            "provider_preflight": 0,
            "repair_fallback": 0,
            "selected_company_validation": 0,
            "terminal_cap_search": 0,
        },
        "lane_sum": 1,
        "aggregate_reconciles": True,
    }


def test_v1_benchmark_preserves_legacy_nonphysical_tool_reporting() -> None:
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="legacy-v1-tool-reporting",
        sector="energy",
        market_cap_focus="smid_cap",
        objective="Preserve v1 reporting while v2 accounting becomes physical-only.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T20:00:00Z",
        completed_at="2026-07-16T20:01:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        tool_calls=[
            _call("V1-OK", "OK"),
            _call("V1-PLANNED", "PLANNED"),
            _call("V1-SKIPPED", "SKIPPED_BUDGET_EXHAUSTED"),
        ],
    )

    row = _sector_result_from_artifact(
        artifact=artifact,
        artifact_path=None,
        report_path=None,
    )

    assert row["tool_call_status_counts"] == {
        "OK": 1,
        "PLANNED": 1,
        "SKIPPED_BUDGET_EXHAUSTED": 1,
    }
    assert row["tool_call_counts"] == {"test_tool": 3}
    assert row["evidence_counts"]["tool_calls_total"] == 3
    assert row["evidence_counts"]["tool_calls_failed"] == 2
    assert {item["call_id"] for item in row["failed_tool_calls"]} == {
        "V1-PLANNED",
        "V1-SKIPPED",
    }
    assert row["tool_call_lane_counts"] == {
        "parent_research": 3,
        "repair_fallback": 0,
        "company_underwriting": 0,
        "selected_validation": 0,
        "aggregate": 3,
    }
    assert "tool_call_diagnostic_status_counts" not in row


def test_terminal_cap_reserve_cost_is_counted_without_fabricating_tool_calls() -> None:
    common = {
        "lane": "terminal_cap_search",
        "provider": "openai",
        "authorization_run_id": "cap-auth",
        "attempt_number": 1,
        "ticker": "AAA",
        "attempt_status": "PENDING",
        "billing_status": "WORST_CASE_RESERVED",
    }
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_terminal_reserve_accounting",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Count permanently consumed terminal-search authority.",
        as_of_date="2026-07-15",
        created_at="2026-07-15T20:00:00Z",
        completed_at="2026-07-15T20:01:00Z",
        status="FAILED",
        final_verdict=None,
        selected_ticker=None,
        confidence=None,
        pipeline_version="v2",
        execution_status="FAILED",
        decision_status="INCOMPLETE",
        candidate_selection={
            "data_gap_repair": {
                "terminal_cap_search_usage_records": [
                    {
                        **common,
                        "call_type": "responses_model",
                        "model": "gpt-5.5",
                        "input_tokens": 100,
                        "cached_input_tokens": 0,
                        "output_tokens": 10,
                        "reserved_output_tokens": 30,
                        "cost_estimate_usd": 0.02,
                    },
                    {
                        **common,
                        "call_type": "web_search_call_reserve",
                        "call_id": "RESERVE1",
                        "cost_estimate_usd": 0.01,
                    },
                    {
                        **common,
                        "call_type": "web_search_call_reserve",
                        "call_id": "RESERVE2",
                        "cost_estimate_usd": 0.01,
                    },
                ]
            }
        },
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )

    row = _sector_result_from_artifact(
        artifact=artifact,
        artifact_path=None,
        report_path=None,
    )

    assert row["tool_call_lane_counts"]["terminal_cap_search"] == 0
    assert row["provider_call_lane_counts"]["terminal_cap_search"] == 1
    assert row["provider_call_status_counts"] == {"ERROR": 1}
    assert row["lane_usage"]["lanes"]["terminal_cap_search"] == {
        "tool_call_attempts": 0,
        "tool_calls_ok": 0,
        "tool_calls_failed": 0,
        "provider_call_attempts": 1,
        "provider_calls_ok": 0,
        "provider_calls_failed": 1,
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "output_tokens": 10,
        "reserved_output_tokens": 30,
        "cost_microdollars": 40000,
        "cost_usd": "0.040000",
    }


def test_v2_benchmark_persists_exception_attempt_diagnostic(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA"],
            loaded_tickers=["AAA", "BBB"],
            source="sector_scan_db",
        ),
    )

    def fail_runtime(**kwargs):
        raise RuntimeError("benchmark child crashed before artifact")

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        fail_runtime,
    )

    benchmark = run_autonomous_sector_benchmark(
        sectors=["energy"],
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        provider_preflight=False,
    )

    assert benchmark["pipeline_version"] == "v2"
    assert benchmark["status"] == "COMPLETED_WITH_FAILURES"
    row = benchmark["sector_results"][0]
    assert row["execution_status"] == "FAILED"
    assert row["decision_status"] == "INCOMPLETE"
    assert row["final_verdict"] is None
    assert row["artifact_path"].endswith("autonomous_sector_attempt_diagnostic.json")
    assert Path(row["artifact_path"]).exists()


def test_v2_benchmark_passes_fixed_as_of_to_candidate_resolution(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    captured = {}

    def capture_selection(**kwargs):
        captured.update(kwargs)
        return SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA"],
            loaded_tickers=["AAA"],
            source="sector_scan_db",
        )

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        capture_selection,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop after resolution")),
    )

    run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-06-30",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        provider_preflight=False,
    )

    assert captured["as_of_date"] == "2026-06-30"
    assert captured["allow_live_market_data"] is False


def test_v2_benchmark_keyboard_interrupt_persists_attempt_then_reraises(
    monkeypatch, tmp_path
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA"],
            loaded_tickers=["AAA"],
            source="sector_scan_db",
        ),
    )

    def completed_no_selection(sector: str) -> AutonomousSectorFinancialRunArtifact:
        return AutonomousSectorFinancialRunArtifact(
            run_id=f"autonomous_sector_{sector}_complete",
            sector=sector,
            market_cap_focus="large_and_mega",
            objective="Preserve completed benchmark work across interruption.",
            as_of_date="2026-06-30",
            created_at="2026-06-30T12:00:00Z",
            completed_at="2026-06-30T12:01:00Z",
            status="COMPLETED",
            final_verdict="NO_SELECTION",
            selected_ticker=None,
            confidence=None,
            pipeline_version="v2",
            execution_status="COMPLETED",
            decision_status="COMPLETE",
            admitted_tickers=["AAA"],
            candidate_selection={
                "loaded_tickers": ["AAA"],
                "selected_tickers": ["AAA"],
            },
            candidate_dispositions=[
                CandidateDisposition(
                    ticker="AAA",
                    terminal_state="SCREENED_OUT",
                    scope_status="IN_SCOPE",
                    screen_status="FAIL",
                    review_status="NOT_REQUIRED",
                    reason_codes=["SOURCE_BACKED_SCREEN_FAILURE"],
                    last_completed_stage="SCREENING",
                    screen_result=ScreenResult(
                        contract_id="energy",
                        status="FAIL",
                        gate_evaluations=[
                            GateEvaluation(
                                contract_id="energy",
                                rule_id="DELISTING_NOTICE",
                                status="FAIL",
                                applicable=True,
                                observed_value="8-K item 3.01",
                                threshold="no active delisting notice",
                                evidence_ref_id="sec:AAA:8-k:2026-06-30",
                                reason_code="SOURCE_BACKED_SCREEN_FAILURE",
                            )
                        ],
                        reason_codes=["SOURCE_BACKED_SCREEN_FAILURE"],
                    ),
                )
            ],
            selection_validation=SectorSelectionValidation(status="NOT_REQUIRED"),
            tool_calls=[_call("P1", "OK", question_id="Q1")],
            contract_version=SECTOR_CONTRACT_VERSION_V2,
        )

    first_pass_sectors: list[str] = []

    def interrupt_second_sector(**kwargs):
        first_pass_sectors.append(kwargs["sector"])
        if kwargs["sector"] == "energy":
            return completed_no_selection("energy")
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        interrupt_second_sector,
    )

    with pytest.raises(KeyboardInterrupt):
        run_autonomous_sector_benchmark(
            sectors=["energy", "hospitality_gaming"],
            objective="Preserve completed benchmark work across interruption.",
            as_of_date="2026-06-30",
            market_cap_focus="large_and_mega",
            pipeline_version="v2",
            provider_preflight=False,
        )

    diagnostics = list(
        (data_dir / "outputs" / "runs" / "autonomous_sector_diagnostics").glob(
            "*/autonomous_sector_attempt_diagnostic.json"
        )
    )
    assert len(diagnostics) == 1
    summaries = list(
        (data_dir / "outputs" / "runs" / "autonomous_sector_benchmark").glob(
            "*/benchmark_summary.json"
        )
    )
    assert len(summaries) == 1
    checkpoint = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert checkpoint["status"] == "INTERRUPTED"
    assert checkpoint["artifact_class"] == "checkpoint"
    assert checkpoint["run_id"] == summaries[0].parent.name
    assert checkpoint["interrupted_sector"] == "hospitality_gaming"
    assert checkpoint["interrupted_diagnostic_path"] == str(diagnostics[0])
    assert [row["sector"] for row in checkpoint["sector_results"]] == ["energy"]
    assert checkpoint["sector_results"][0]["decision_status"] == "COMPLETE"
    assert first_pass_sectors == ["energy", "hospitality_gaming"]

    resumed_execution: list[str] = []

    def finish_resume(**kwargs):
        resumed_execution.append(kwargs["sector"])
        return completed_no_selection(kwargs["sector"])

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        finish_resume,
    )
    resumed = run_autonomous_sector_benchmark(
        sectors=["energy", "hospitality_gaming"],
        objective="Preserve completed benchmark work across interruption.",
        as_of_date="2026-06-30",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        provider_preflight=False,
        resume_benchmark_run_id=checkpoint["run_id"],
    )

    assert resumed_execution == ["hospitality_gaming"]
    assert resumed["sector_results"][0]["sector"] == "energy"
    assert resumed["sector_results"][0]["benchmark_execution_status"] == "REUSED_FROM_RESUME"
    assert resumed["sector_results"][1]["sector"] == "hospitality_gaming"


def test_v2_cache_prewarm_fails_before_candidate_or_refresh_work(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("candidate resolution must not run")),
    )

    with pytest.raises(ValueError, match="v2 cache prewarming is disabled"):
        run_autonomous_sector_benchmark(
            sectors=["energy"],
            market_cap_focus="large_and_mega",
            pipeline_version="v2",
            prewarm_cache=True,
            cache_refresh_run_id="cache-run",
            provider_preflight=False,
        )


def test_v2_readiness_preflight_uses_frozen_bound_and_skips_cost_provider_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    resolution_kwargs: list[dict] = []
    readiness_kwargs: list[dict] = []

    def resolve(**kwargs):
        resolution_kwargs.append(dict(kwargs))
        return SectorCandidateSelection(
            sector="energy",
            market_cap_focus="large_and_mega",
            selected_tickers=["AAA", "BBB"],
            loaded_tickers=["AAA", "BBB"],
            source="accepted_census",
            cap_classifications={
                "AAA": {"market_cap_mm": 25_000.0, "cap_source": "accepted_census"},
                "BBB": {"market_cap_mm": 30_000.0, "cap_source": "accepted_census"},
            },
            census_lineage={"run_id": "accepted-run-1", "as_of_date": "2026-07-17"},
        )

    def readiness(**kwargs):
        readiness_kwargs.append(dict(kwargs))
        assert kwargs["candidate_payloads"]["energy"]["execution_tickers"] == ["AAA"]
        assert kwargs["candidate_payloads"]["energy"]["deferred_by_bound_tickers"] == ["BBB"]
        return {
            "status": "COMPLETED",
            "readiness_status": "NEEDS_DATA",
            "census_lineage": {
                "run_id": "accepted-run-1",
                "as_of_date": "2026-07-17",
            },
            "counts": {
                "sector_count": 1,
                "membership_candidates": 2,
                "execution_candidates": 1,
                "deferred_by_bound": 1,
                "excluded_candidates": 0,
                "ready": 0,
                "needs_data": 1,
                "incomplete": 0,
            },
            "actual_usage": {
                "model_calls": 0,
                "search_calls": 0,
                "network_calls": 0,
                "cost_microdollars": 0,
                "cost_usd": "0.000000",
            },
            "production_sector_scan_exercised": False,
        }

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        resolve,
    )
    monkeypatch.setattr(
        "app.autonomous.readiness_preflight.run_v2_readiness_preflight",
        readiness,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._v2_execution_preflight_payload",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cost authority must not be built")),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._run_provider_preflight_in_v2_policy",
        lambda: (_ for _ in ()).throw(AssertionError("provider preflight forbidden")),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("sector runtime forbidden")),
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-07-17",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        max_candidates=1,
        provider_preflight=True,
        readiness_preflight_only=True,
    )

    assert len(resolution_kwargs) == 1
    assert resolution_kwargs[0]["allow_live_market_data"] is False
    assert resolution_kwargs[0]["require_accepted_census"] is True
    assert isinstance(
        resolution_kwargs[0]["accepted_census_authority"],
        AcceptedCensusRunAuthority,
    )
    assert len(readiness_kwargs) == 1
    assert readiness_kwargs[0]["as_of_date"] == "2026-07-17"
    assert readiness_kwargs[0]["execution_set_fingerprint"] == (
        "1553680d350cb3f680c661c3f83528d9791f1346e9fd3f8594f92d3c16066178"
    )
    assert artifact["status"] == "READINESS_PREFLIGHT_ONLY"
    assert artifact["execution_mode"] == "READINESS_PREFLIGHT_ONLY"
    assert artifact["diagnostic_only"] is True
    assert artifact["sector_results"] == []
    assert artifact["all_sector_cost_preflight"] == {}
    assert artifact["terminal_cap_search_authorization"] is None
    assert artifact["provider_preflight"] == {
        "enabled": False,
        "status": "SKIPPED",
        "reason": "readiness_preflight_only",
        "provider_calls": [],
    }
    assert artifact["candidate_execution_ledger"]["energy"]["membership_tickers"] == ["AAA", "BBB"]
    assert artifact["candidate_execution_ledger"]["energy"]["execution_tickers"] == ["AAA"]
    assert artifact["candidate_execution_ledger"]["energy"]["deferred_by_bound_tickers"] == ["BBB"]
    assert artifact["actual_usage"]["cost_usd"] == "0.000000"
    assert artifact["production_sector_scan_exercised"] is False


def test_v2_free_data_repair_uses_exact_bound_and_skips_cost_provider_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    resolution_kwargs: list[dict] = []
    repair_kwargs: list[dict] = []

    def resolve(**kwargs):
        resolution_kwargs.append(dict(kwargs))
        return SectorCandidateSelection(
            sector="energy",
            market_cap_focus="large_and_mega",
            selected_tickers=["AAA", "BBB"],
            loaded_tickers=["AAA", "BBB"],
            source="accepted_census",
            cap_classifications={
                "AAA": {
                    "issuer_cik": "0000000001",
                    "market_cap_mm": 25_000.0,
                    "cap_source": "accepted_census",
                },
                "BBB": {
                    "issuer_cik": "0000000002",
                    "market_cap_mm": 30_000.0,
                    "cap_source": "accepted_census",
                },
            },
            census_lineage={
                "run_id": "accepted-run-1",
                "as_of_date": "2026-07-17",
            },
        )

    def repair(**kwargs):
        repair_kwargs.append(dict(kwargs))
        payload = kwargs["candidate_payloads"]["energy"]
        assert payload["membership_tickers"] == ["AAA", "BBB"]
        assert payload["execution_tickers"] == ["AAA"]
        assert payload["deferred_by_bound_tickers"] == ["BBB"]
        assert payload["excluded_tickers"] == []
        return {
            "status": "COMPLETED",
            "after_readiness": {
                "status": "COMPLETED",
                "readiness_status": "READY",
                "census_lineage": {
                    "run_id": "accepted-run-1",
                    "as_of_date": "2026-07-17",
                },
                "counts": {
                    "sector_count": 1,
                    "membership_candidates": 2,
                    "execution_candidates": 1,
                    "deferred_by_bound": 1,
                    "excluded_candidates": 0,
                    "ready": 1,
                    "needs_data": 0,
                    "incomplete": 0,
                },
                "actual_usage": {
                    "model_calls": 0,
                    "search_calls": 0,
                    "network_calls": 0,
                    "cost_microdollars": 0,
                    "cost_usd": "0.000000",
                },
            },
            "readiness_transition_counts": {"NEEDS_DATA_TO_READY": 1},
            "network": {
                "allowed_domains": ["data.sec.gov", "stooq.com"],
                "calls_by_domain": {"data.sec.gov": 2, "stooq.com": 1},
                "free_network_calls": 3,
                "unexpected_domains": [],
            },
            "actual_usage": {
                "model_calls": 0,
                "search_calls": 0,
                "paid_provider_calls": 0,
                "network_calls": 3,
                "cost_microdollars": 0,
                "cost_usd": "0.000000",
            },
            "production_sector_scan_exercised": False,
            "packet_materialized": False,
            "watchlist_mutated": False,
        }

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        resolve,
    )
    monkeypatch.setattr(
        "app.autonomous.accepted_census_free_data_repair.run_accepted_census_free_data_repair",
        repair,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._v2_execution_preflight_payload",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cost authority must not be built")),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._run_provider_preflight_in_v2_policy",
        lambda: (_ for _ in ()).throw(AssertionError("provider preflight forbidden")),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("sector runtime forbidden")),
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-07-17",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        max_candidates=1,
        provider_preflight=True,
        free_data_repair_only=True,
    )

    assert len(resolution_kwargs) == 1
    assert resolution_kwargs[0]["allow_live_market_data"] is False
    assert resolution_kwargs[0]["require_accepted_census"] is True
    assert isinstance(
        resolution_kwargs[0]["accepted_census_authority"],
        AcceptedCensusRunAuthority,
    )
    assert len(repair_kwargs) == 1
    assert repair_kwargs[0]["sectors"] == ["energy"]
    assert repair_kwargs[0]["as_of_date"] == "2026-07-17"
    assert isinstance(
        repair_kwargs[0]["accepted_census_authority"],
        AcceptedCensusRunAuthority,
    )
    assert artifact["status"] == "FREE_DATA_REPAIR_ONLY"
    assert artifact["execution_mode"] == "FREE_DATA_REPAIR_ONLY"
    assert artifact["diagnostic_only"] is False
    assert artifact["maintenance_only"] is True
    assert artifact["spend_authorized"] is False
    assert artifact["sector_results"] == []
    assert artifact["all_sector_cost_preflight"] == {}
    assert artifact["terminal_cap_search_authorization"] is None
    assert artifact["provider_preflight"] == {
        "enabled": False,
        "status": "SKIPPED",
        "reason": "free_data_repair_only",
        "provider_calls": [],
    }
    assert artifact["candidate_execution_ledger"]["energy"]["membership_tickers"] == ["AAA", "BBB"]
    assert artifact["candidate_execution_ledger"]["energy"]["execution_tickers"] == ["AAA"]
    assert artifact["candidate_execution_ledger"]["energy"]["deferred_by_bound_tickers"] == ["BBB"]
    assert artifact["fixed_cohort"]["membership_security_count"] == 2
    assert artifact["fixed_cohort"]["execution_security_count"] == 1
    assert artifact["actual_usage"] == {
        "model_calls": 0,
        "search_calls": 0,
        "paid_provider_calls": 0,
        "network_calls": 3,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }
    assert artifact["production_sector_scan_exercised"] is False
    assert artifact["screened_candidate_count"] == 0
    assert artifact["underwritten_candidate_count"] == 0
    assert artifact["actionable_candidate_count"] == 0

    paths = persist_autonomous_sector_benchmark(artifact)
    assert paths.cost_preflight_json is None
    assert paths.readiness_preflight_json is not None
    assert paths.free_data_repair_json is not None
    persisted_repair = json.loads(paths.free_data_repair_json.read_text(encoding="utf-8"))
    persisted_readiness = json.loads(paths.readiness_preflight_json.read_text(encoding="utf-8"))
    assert persisted_repair["actual_usage"]["cost_usd"] == "0.000000"
    assert persisted_repair["network"]["free_network_calls"] == 3
    assert persisted_readiness["readiness_status"] == "READY"


def test_v1_resume_row_is_not_reused_by_v2() -> None:
    resume_artifact = {
        "artifact_type": "autonomous_sector_benchmark_v1",
        "sector_results": [],
    }
    row = {
        "sector": "energy",
        "status": "COMPLETED",
        "benchmark_execution_status": "RAN",
        "degraded_states": [],
    }
    resume_pipeline_version = _resume_artifact_pipeline_version(resume_artifact)

    assert resume_pipeline_version == "v1"
    assert (
        _can_reuse_resume_result(
            row,
            pipeline_version="v1",
            resume_pipeline_version=resume_pipeline_version,
            rerun_completed_sectors=False,
        )
        is True
    )
    assert (
        _can_reuse_resume_result(
            row,
            pipeline_version="v2",
            resume_pipeline_version=resume_pipeline_version,
            rerun_completed_sectors=False,
        )
        is False
    )


def test_v2_resume_reuse_requires_completed_execution_and_complete_decision() -> None:
    row = {
        "sector": "energy",
        "status": "COMPLETED",
        "pipeline_version": "v2",
        "benchmark_execution_status": "RAN",
        "execution_status": "COMPLETED",
        "decision_status": "INCOMPLETE",
        "degraded_states": [],
    }

    assert (
        _can_reuse_resume_result(
            row,
            pipeline_version="v2",
            resume_pipeline_version="v2",
            rerun_completed_sectors=False,
        )
        is False
    )
    assert (
        _can_reuse_resume_result(
            {**row, "execution_status": "FAILED", "decision_status": "COMPLETE"},
            pipeline_version="v2",
            resume_pipeline_version="v2",
            rerun_completed_sectors=False,
        )
        is False
    )
    assert (
        _can_reuse_resume_result(
            {**row, "decision_status": "COMPLETE"},
            pipeline_version="v2",
            resume_pipeline_version="v2",
            rerun_completed_sectors=False,
        )
        is True
    )


def test_v2_resume_compatibility_pins_all_reliable_run_inputs() -> None:
    budget = AutonomousRunBudget(
        max_tool_calls=8,
        max_turns=3,
        max_cost_usd=None,
        timebox_seconds=None,
        max_candidates=25,
    )
    resume_artifact = {
        "artifact_type": "autonomous_sector_benchmark_v2",
        "pipeline_version": "v2",
        "as_of_date": "2026-07-15",
        "market_cap_focus": "large_and_mega",
        "objective": "Find the best underwritten large-cap candidate.",
        "max_candidates": 25,
        "budget": budget.to_dict(),
        "request_fingerprint": "a" * 64,
    }
    kwargs = {
        "pipeline_version": "v2",
        "as_of_date": "2026-07-15",
        "market_cap_focus": "large_and_mega",
        "objective": "Find the best underwritten large-cap candidate.",
        "max_candidates": 25,
        "budget": budget,
        "expected_request_fingerprint": "a" * 64,
    }

    assert _resume_artifact_compatibility(resume_artifact, **kwargs) == {
        "compatible": True,
        "strict_input_match": True,
        "resume_pipeline_version": "v2",
        "reasons": [],
    }
    assert _resume_artifact_compatibility(
        {
            **resume_artifact,
            "pipeline_version": "v1",
            "artifact_type": "autonomous_sector_benchmark_v1",
        },
        **kwargs,
    )["reasons"] == ["PIPELINE_VERSION_MISMATCH"]
    assert _resume_artifact_compatibility(
        {**resume_artifact, "as_of_date": "2026-07-14"},
        **kwargs,
    )["reasons"] == ["AS_OF_DATE_MISMATCH"]
    assert _resume_artifact_compatibility(
        {**resume_artifact, "as_of_date": None},
        **kwargs,
    )["reasons"] == ["AS_OF_DATE_UNPINNED"]
    assert _resume_artifact_compatibility(
        {**resume_artifact, "market_cap_focus": "mega_cap"},
        **kwargs,
    )["reasons"] == ["MARKET_CAP_FOCUS_MISMATCH"]
    assert _resume_artifact_compatibility(
        {**resume_artifact, "objective": "Use a different objective."},
        **kwargs,
    )["reasons"] == ["OBJECTIVE_MISMATCH"]
    assert _resume_artifact_compatibility(
        {**resume_artifact, "max_candidates": 24},
        **kwargs,
    )["reasons"] == ["MAX_CANDIDATES_MISMATCH"]
    assert _resume_artifact_compatibility(
        {
            **resume_artifact,
            "budget": {**budget.to_dict(), "max_tool_calls": 7},
        },
        **kwargs,
    )["reasons"] == ["BUDGET_MISMATCH"]
    assert _resume_artifact_compatibility(
        {**resume_artifact, "request_fingerprint": "b" * 64},
        **kwargs,
    )["reasons"] == ["REQUEST_FINGERPRINT_MISMATCH"]
    assert _resume_artifact_compatibility(
        {key: value for key, value in resume_artifact.items() if key != "request_fingerprint"},
        **kwargs,
    )["reasons"] == ["REQUEST_FINGERPRINT_UNPINNED"]


def test_v1_resume_compatibility_keeps_legacy_unpinned_behavior() -> None:
    budget = AutonomousRunBudget(
        max_tool_calls=8,
        max_turns=3,
        max_cost_usd=None,
        timebox_seconds=None,
        max_candidates=None,
    )

    assert _resume_artifact_compatibility(
        {"artifact_type": "autonomous_sector_benchmark_v1"},
        pipeline_version="v1",
        as_of_date=None,
        market_cap_focus="smid_cap",
        objective="Legacy objective.",
        max_candidates=None,
        budget=budget,
    ) == {
        "compatible": True,
        "strict_input_match": False,
        "resume_pipeline_version": "v1",
        "reasons": [],
    }


def test_v2_benchmark_stops_before_spend_when_resume_as_of_is_incompatible(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    budget = AutonomousRunBudget(
        max_tool_calls=8,
        max_turns=3,
        max_cost_usd=None,
        timebox_seconds=None,
        max_candidates=1,
    )
    resume_artifact = {
        "artifact_type": "autonomous_sector_benchmark_v2",
        "pipeline_version": "v2",
        "as_of_date": "2026-07-14",
        "market_cap_focus": "large_and_mega",
        "objective": "Find the best underwritten large-cap candidate.",
        "max_candidates": 1,
        "budget": budget.to_dict(),
        "sector_results": [
            {
                "sector": "energy",
                "status": "COMPLETED",
                "pipeline_version": "v2",
                "execution_status": "COMPLETED",
                "decision_status": "COMPLETE",
                "benchmark_execution_status": "RAN",
                "degraded_states": [],
            }
        ],
    }
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._load_benchmark_resume",
        lambda _run_id: resume_artifact,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA"],
            loaded_tickers=["AAA"],
            source="sector_scan_db",
        ),
    )

    def fail_if_rerun(**kwargs):
        runtime_calls.append(kwargs["sector"])
        raise RuntimeError("proof that the incompatible row was rerun")

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        fail_if_rerun,
    )

    benchmark = run_autonomous_sector_benchmark(
        sectors=["energy"],
        objective="Find the best underwritten large-cap candidate.",
        as_of_date="2026-07-15",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        budget=budget,
        max_candidates=1,
        provider_preflight=False,
        resume_benchmark_run_id="prior-v2-run",
    )

    assert runtime_calls == []
    assert benchmark["resume_compatibility"] == {
        "requested": True,
        "compatible": False,
        "strict_input_match": True,
        "resume_pipeline_version": "v2",
        "reasons": ["AS_OF_DATE_MISMATCH", "REQUEST_FINGERPRINT_UNPINNED"],
    }
    assert benchmark["reused_sector_count"] == 0
    assert benchmark["status"] == "STOPPED_BEFORE_SPEND"
    assert benchmark["all_sector_cost_preflight"]["reason_codes"] == [
        "RESUME_EXECUTION_FINGERPRINT_MISMATCH"
    ]
    assert benchmark["sector_results"][0]["benchmark_execution_status"] == ("STOPPED_BEFORE_SPEND")


def test_provider_skip_is_truthful_for_v2_and_unchanged_for_v1() -> None:
    provider_failure = {
        "failure_code": "LLM_PROVIDER_UNAVAILABLE",
        "error": "provider disabled",
    }

    v1_row = _sector_result_skipped_provider(
        sector="energy",
        provider_failure=provider_failure,
        pipeline_version="v1",
    )
    v2_row = _sector_result_skipped_provider(
        sector="energy",
        provider_failure=provider_failure,
        pipeline_version="v2",
    )

    assert v1_row["final_verdict"] == "NO_SELECTION"
    assert "pipeline_version" not in v1_row
    assert "execution_status" not in v1_row
    assert "decision_status" not in v1_row
    assert v2_row["pipeline_version"] == "v2"
    assert v2_row["execution_status"] == "FAILED"
    assert v2_row["decision_status"] == "INCOMPLETE"
    assert v2_row["final_verdict"] is None


def test_cache_skip_is_truthful_for_v2_and_unchanged_for_v1() -> None:
    cache_coverage = {
        "cache_limited_reasons": ["NO_CACHE_READY_CANDIDATES"],
        "cache_readiness_warnings": ["ANNUAL_FACTS_MISSING"],
    }

    v1_row = _sector_result_skipped_cache(
        sector="energy",
        cache_coverage=cache_coverage,
        pipeline_version="v1",
    )
    v2_row = _sector_result_skipped_cache(
        sector="energy",
        cache_coverage=cache_coverage,
        pipeline_version="v2",
    )

    assert v1_row["final_verdict"] == "NO_SELECTION"
    assert "pipeline_version" not in v1_row
    assert "execution_status" not in v1_row
    assert "decision_status" not in v1_row
    assert v2_row["pipeline_version"] == "v2"
    assert v2_row["execution_status"] == "FAILED"
    assert v2_row["decision_status"] == "INCOMPLETE"
    assert v2_row["final_verdict"] is None


def test_v2_freezes_candidate_count_before_provider_health_preflight(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    events: list[str] = []

    def resolve(**kwargs):
        events.append(f"resolve:{kwargs['sector']}")
        return SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA"],
            loaded_tickers=["AAA"],
            source="sector_scan_db",
        )

    def provider_preflight():
        events.append("provider_preflight")
        return {
            "enabled": True,
            "status": "FAILED",
            "failure_code": "LLM_PROVIDER_UNAVAILABLE",
            "error": "test stop",
            "provider_calls": [],
        }

    monkeypatch.setattr("app.autonomous.sector_benchmark.resolve_sector_candidate_tickers", resolve)
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._run_provider_preflight_in_v2_policy",
        provider_preflight,
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-07-16",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        budget=AutonomousRunBudget(
            max_tool_calls=8,
            max_turns=1,
            max_cost_usd=None,
            timebox_seconds=None,
            max_candidates=1,
        ),
        max_candidates=1,
        provider_preflight=True,
    )

    assert events == ["resolve:energy", "provider_preflight"]
    assert artifact["all_sector_cost_preflight"]["status"] == "AUTHORIZED"
    assert artifact["all_sector_cost_preflight"]["frozen_candidate_counts"] == {"energy": 1}


def test_v1_invalid_canonical_financial_scope_prevents_provider_preflight(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.autonomous.financial_integrity import FinancialIntegrityScope
    from app.config import get_config

    get_config.cache_clear()
    events: list[str] = []

    def _resolve_candidates(**kwargs):
        events.append(f"resolve:{kwargs['sector']}")
        return SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["BAD"],
            loaded_tickers=["BAD", "DROP"],
            source="sector_scan_db",
        )

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        _resolve_candidates,
    )

    class _InvalidFinancialContext:
        def scope(self, *, context):
            events.append(f"scope:{context}")
            return FinancialIntegrityScope(
                context=context,
                run_as_of_date="2026-07-22",
                packets=(),
            )

    def _build_financial_context(**kwargs):
        events.append(f"context:{','.join(kwargs['tickers'])}")
        assert kwargs["tickers"] == ["BAD"]
        return _InvalidFinancialContext()

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.build_canonical_v1_financial_context",
        _build_financial_context,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._run_provider_preflight",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("provider preflight is forbidden for invalid financial scope")
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("sector execution is forbidden for invalid financial scope")
        ),
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["software"],
        as_of_date="2026-07-22",
        pipeline_version="v1",
        provider_preflight=True,
    )

    assert events == [
        "resolve:software",
        "context:BAD",
        "scope:benchmark_provider_preflight:software:2026-07-22",
    ]
    assert artifact["provider_preflight"] == {
        "enabled": False,
        "status": "SKIPPED",
        "reason": "no_sector_execution_required",
        "provider_calls": [],
    }
    assert artifact["sector_results"][0]["sector"] == "software"
    assert artifact["sector_results"][0]["financial_integrity_status"] == "NEEDS_DATA"


def _configure_v1_failed_provider_preflight(
    monkeypatch,
    tmp_path,
    *,
    mutate_scope: bool,
) -> dict[str, int]:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.autonomous.financial_integrity import FinancialIntegrityScope
    from app.config import get_config
    from tests.test_financial_integrity import _valid_packet

    get_config.cache_clear()
    packet = _valid_packet()
    calls = {"provider": 0, "runtime": 0}

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["MEGA"],
            loaded_tickers=["MEGA"],
            source="sector_scan_db",
        ),
    )

    class _FinancialContext:
        def scope(self, *, context):
            return FinancialIntegrityScope(
                context=context,
                run_as_of_date="2026-07-22",
                packets=(packet,),
            )

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.build_canonical_v1_financial_context",
        lambda **_kwargs: _FinancialContext(),
    )

    class _FailingProvider:
        provider_name = "openai"
        model = "gpt-5.5"

        @staticmethod
        def enabled() -> bool:
            return True

        def synthesize_json(self, **_kwargs):
            calls["provider"] += 1
            if mutate_scope:
                packet["current_price"] = 101.0
            raise RuntimeError("provider preflight failed")

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.get_llm_provider",
        lambda: _FailingProvider(),
    )

    def _unexpected_runtime(**_kwargs):
        calls["runtime"] += 1
        raise AssertionError("sector execution is forbidden after failed provider preflight")

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        _unexpected_runtime,
    )
    return calls


def test_v1_failed_provider_preflight_promotes_financial_scope_mutation(
    monkeypatch,
    tmp_path,
) -> None:
    from app.autonomous.financial_integrity import InvalidFinancialInputError

    calls = _configure_v1_failed_provider_preflight(
        monkeypatch,
        tmp_path,
        mutate_scope=True,
    )
    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_autonomous_sector_benchmark(
            sectors=["software"],
            as_of_date="2026-07-22",
            pipeline_version="v1",
            provider_preflight=True,
        )

    assert exc_info.value.status == "INVALID_FINANCIAL_INPUT"
    assert calls == {"provider": 1, "runtime": 0}


def test_v1_failed_provider_preflight_without_scope_mutation_keeps_failed_semantics(
    monkeypatch,
    tmp_path,
) -> None:
    calls = _configure_v1_failed_provider_preflight(
        monkeypatch,
        tmp_path,
        mutate_scope=False,
    )
    artifact = run_autonomous_sector_benchmark(
        sectors=["software"],
        as_of_date="2026-07-22",
        pipeline_version="v1",
        provider_preflight=True,
    )

    assert calls == {"provider": 1, "runtime": 0}
    assert artifact["provider_preflight"]["status"] == "FAILED"
    assert artifact["provider_preflight"]["failure_code"] == "LLM_PROVIDER_ERROR"
    assert artifact["sector_results"][0]["benchmark_execution_status"] == (
        "SKIPPED_PROVIDER_UNAVAILABLE"
    )


def test_v2_bound_prices_and_passes_only_frozen_execution_subset(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA", "BBB", "CCC"],
            loaded_tickers=["AAA", "BBB", "CCC"],
            source="sector_scan_db",
            census_lineage={"run_id": "accepted-census"},
        ),
    )
    captured: dict[str, object] = {}

    def stop_after_capture(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after bounded runtime capture")

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        stop_after_capture,
    )
    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-07-16",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        max_candidates=2,
        budget=AutonomousRunBudget(
            max_tool_calls=8,
            max_turns=1,
            max_cost_usd=None,
            timebox_seconds=None,
            max_candidates=2,
        ),
        provider_preflight=False,
    )

    assert captured["tickers"] == ["AAA", "BBB"]
    assert captured["candidate_selection"]["selected_tickers"] == [
        "AAA",
        "BBB",
        "CCC",
    ]
    assert captured["candidate_selection"]["execution_tickers"] == ["AAA", "BBB"]
    assert captured["candidate_selection"]["deferred_by_bound_tickers"] == ["CCC"]
    preflight = artifact["all_sector_cost_preflight"]
    assert preflight["membership_candidate_counts"] == {"energy": 3}
    assert preflight["frozen_candidate_counts"] == {"energy": 2}
    assert preflight["deferred_by_bound_counts"] == {"energy": 1}
    assert preflight["census_lineage_by_sector"] == {"energy": {"run_id": "accepted-census"}}
    assert preflight["estimate"]["aggregate"]["candidate_count"] == 2


def test_v2_cost_stop_never_calls_provider_or_sector_runtime(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    tickers = [f"T{index:03d}" for index in range(100)]
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=tickers,
            loaded_tickers=tickers,
            source="sector_scan_db",
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._run_provider_preflight_in_v2_policy",
        lambda: (_ for _ in ()).throw(AssertionError("provider spend must not start")),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("runtime must not start")),
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-07-16",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        provider_preflight=True,
    )

    assert artifact["status"] == "STOPPED_BEFORE_SPEND"
    assert artifact["all_sector_cost_preflight"]["status"] == "STOP_BEFORE_SPEND"
    assert artifact["provider_preflight"]["status"] == "SKIPPED"
    row = artifact["sector_results"][0]
    assert row["benchmark_execution_status"] == "STOPPED_BEFORE_SPEND"
    assert row["execution_status"] == "FAILED"
    assert row["decision_status"] == "INCOMPLETE"
    assert row["final_verdict"] is None


def test_v2_wrong_provider_stops_even_when_health_preflight_is_disabled(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA"],
            loaded_tickers=["AAA"],
            source="sector_scan_db",
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("runtime must not start")),
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-07-16",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        provider_preflight=False,
        max_candidates=1,
    )

    assert artifact["status"] == "STOPPED_BEFORE_SPEND"
    assert artifact["all_sector_cost_preflight"]["reason_codes"] == [
        "V2_PROVIDER_MUST_BE_OPENAI:configured=anthropic"
    ]


def test_all_reused_v2_resume_makes_no_new_calls_and_preserves_prior_lineage(
    monkeypatch,
) -> None:
    objective = "Resume without new spend."
    budget = AutonomousRunBudget(
        max_tool_calls=8,
        max_turns=1,
        max_cost_usd=None,
        timebox_seconds=None,
        max_candidates=1,
    )
    prior_row = {
        "sector": "energy",
        "status": "COMPLETED",
        "pipeline_version": "v2",
        "execution_status": "COMPLETED",
        "decision_status": "COMPLETE",
        "benchmark_execution_status": "RAN",
        "final_verdict": "NO_SELECTION",
        "degraded_states": [],
        "lane_usage": _lane_usage_with_parent_cost(1_000_000),
    }
    resume = {
        "artifact_type": "autonomous_sector_benchmark_v2",
        "pipeline_version": "v2",
        "as_of_date": "2026-07-16",
        "market_cap_focus": "large_and_mega",
        "objective": objective,
        "max_candidates": 1,
        "budget": budget.to_dict(),
        "request_fingerprint": "f" * 64,
        "sector_results": [prior_row],
        "rollups": {
            "lane_usage_totals": {
                "aggregate": {
                    "cost_microdollars": 2_000_000,
                    "cost_usd": "2.000000",
                }
            }
        },
    }
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._load_benchmark_resume", lambda _run_id: resume
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA"],
            loaded_tickers=["AAA"],
            source="sector_scan_db",
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.terminal_cap_search.whole_run_preflight_request_fingerprint",
        lambda **kwargs: "f" * 64,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._run_provider_preflight_in_v2_policy",
        lambda: (_ for _ in ()).throw(AssertionError("no provider preflight")),
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        objective=objective,
        as_of_date="2026-07-16",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        budget=budget,
        max_candidates=1,
        provider_preflight=True,
        resume_benchmark_run_id="prior-v2",
    )

    assert artifact["reused_sector_count"] == 1
    assert artifact["provider_preflight"]["reason"] == "no_sector_execution_required"
    assert artifact["all_sector_cost_preflight"]["status"] == ("SKIPPED_NO_EXECUTION_REQUIRED")
    assert artifact["all_sector_cost_preflight"]["prior_realized_cost_usd"] == ("2.000000")
    assert artifact["rollups"]["lane_usage_totals"]["aggregate"]["cost_microdollars"] == 0
    assert (
        artifact["rollups"]["prior_lineage_lane_usage_totals"]["aggregate"]["cost_microdollars"]
        == 1_000_000
    )
    assert artifact["spend_reconciliation"] == {
        "currency": "USD",
        "cost_unit": "microdollars",
        "current_run_actual_cost_microdollars": 0,
        "current_run_actual_cost_usd": "0.000000",
        "prior_lineage_realized_cost_microdollars": 2_000_000,
        "prior_lineage_realized_cost_usd": "2.000000",
        "combined_realized_cost_microdollars": 2_000_000,
        "combined_realized_cost_usd": "2.000000",
        "combined_realized_reconciles": True,
        "preflight_total_worst_case_cost_microdollars": None,
        "preflight_total_worst_case_cost_usd": None,
        "combined_realized_within_preflight_total_worst_case": None,
    }


def test_partial_v2_resume_authorizes_prior_plus_remaining_under_100(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    objective = "Resume one sector under the whole-run ceiling."
    budget = AutonomousRunBudget(
        max_tool_calls=8,
        max_turns=1,
        max_cost_usd=None,
        timebox_seconds=None,
        max_candidates=1,
    )
    resume = {
        "artifact_type": "autonomous_sector_benchmark_v2",
        "pipeline_version": "v2",
        "as_of_date": "2026-07-16",
        "market_cap_focus": "large_and_mega",
        "objective": objective,
        "max_candidates": 1,
        "budget": budget.to_dict(),
        "request_fingerprint": "e" * 64,
        "sector_results": [
            {
                "sector": "energy",
                "status": "COMPLETED",
                "pipeline_version": "v2",
                "execution_status": "COMPLETED",
                "decision_status": "COMPLETE",
                "benchmark_execution_status": "RAN",
                "final_verdict": "NO_SELECTION",
                "degraded_states": [],
                "lane_usage": _lane_usage_with_parent_cost(1_000_000),
            }
        ],
        "rollups": {
            "lane_usage_totals": {
                "aggregate": {
                    "cost_microdollars": 70_000_000,
                    "cost_usd": "70.000000",
                }
            }
        },
    }
    resolved: list[str] = []
    runtime: list[str] = []
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._load_benchmark_resume", lambda _run_id: resume
    )
    monkeypatch.setattr(
        "app.autonomous.terminal_cap_search.whole_run_preflight_request_fingerprint",
        lambda **kwargs: "e" * 64,
    )

    def resolve(**kwargs):
        resolved.append(kwargs["sector"])
        return SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA"],
            loaded_tickers=["AAA"],
            source="sector_scan_db",
        )

    def fail_runtime(**kwargs):
        runtime.append(kwargs["sector"])
        raise RuntimeError("test stops after authorization")

    monkeypatch.setattr("app.autonomous.sector_benchmark.resolve_sector_candidate_tickers", resolve)
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        fail_runtime,
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy", "hospitality_gaming"],
        objective=objective,
        as_of_date="2026-07-16",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        budget=budget,
        max_candidates=1,
        provider_preflight=False,
        resume_benchmark_run_id="prior-v2",
    )

    assert resolved == ["energy", "hospitality_gaming"]
    assert runtime == ["hospitality_gaming"]
    assert artifact["all_sector_cost_preflight"]["status"] == "AUTHORIZED"
    aggregate = artifact["all_sector_cost_preflight"]["estimate"]["aggregate"]
    assert aggregate["prior_realized_cost_usd"] == "70.000000"
    assert aggregate["remaining_worst_case_cost_usd"] == "27.000000"
    assert aggregate["cost_usd"] == "97.000000"
    assert artifact["spend_reconciliation"]["prior_lineage_realized_cost_usd"] == "70.000000"
    assert artifact["spend_reconciliation"]["current_run_actual_cost_usd"] == ("0.000000")
    assert artifact["spend_reconciliation"]["combined_realized_cost_usd"] == ("70.000000")
    assert artifact["spend_reconciliation"]["preflight_total_worst_case_cost_usd"] == "97.000000"


def test_provider_health_physical_retry_is_reconciled_in_preflight_lane() -> None:
    class RetryProvider:
        provider_name = "openai"
        model = "gpt-5.5"

        @staticmethod
        def enabled() -> bool:
            return True

        def synthesize_json(self, **kwargs):
            physical_attempts = 0

            def call():
                nonlocal physical_attempts
                physical_attempts += 1
                if physical_attempts == 1:
                    raise RuntimeError("status=503 temporary provider failure")
                return LLMResult(
                    json_text='{"ok": true}',
                    model="gpt-5.5",
                    usage_input_tokens=10,
                    usage_output_tokens=5,
                    raw={},
                )

            return call_with_llm_retry_guard(
                provider_name="openai",
                schema_name="benchmark_provider_preflight_v1",
                call=call,
                max_retries=1,
                backoff_seconds=(),
            )

    preflight = _run_provider_preflight_for_provider(RetryProvider())
    rollups = _build_rollups([], preflight)

    assert preflight["status"] == "OK"
    assert [row["status"] for row in preflight["provider_calls"]] == [
        "ERROR",
        "OK",
    ]
    assert [row["provider_call_id"] for row in preflight["provider_calls"]] == [
        "provider-preflight-1",
        "provider-preflight-2",
    ]
    lane = rollups["lane_usage_totals"]["lanes"]["provider_preflight"]
    assert lane["provider_call_attempts"] == 2
    assert lane["provider_calls_ok"] == 1
    assert lane["provider_calls_failed"] == 1
    assert lane["reserved_output_tokens"] == 64
    assert lane["cost_microdollars"] > 0
    assert rollups["lane_usage_totals"]["aggregate_reconciles"] is True


def test_provider_preflight_fallback_ids_are_unique_and_v2_disables_fallback(
    monkeypatch,
) -> None:
    class Primary:
        provider_name = "openai"
        model = "gpt-5.5"

        @staticmethod
        def enabled() -> bool:
            return True

        @staticmethod
        def synthesize_json(**kwargs):
            raise RuntimeError("status=429 insufficient_quota")

    class Fallback:
        provider_name = "anthropic"
        model = "claude-haiku-4-5"

        @staticmethod
        def enabled() -> bool:
            return True

        @staticmethod
        def synthesize_json(**kwargs):
            return LLMResult(
                json_text='{"ok": true}',
                model="claude-haiku-4-5",
                usage_input_tokens=10,
                usage_output_tokens=5,
                raw={},
            )

    monkeypatch.setattr("app.autonomous.sector_benchmark.get_llm_provider", lambda: Primary())
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.get_anthropic_provider", lambda: Fallback()
    )

    with_fallback = _run_provider_preflight(allow_fallback=True)
    without_fallback = _run_provider_preflight(allow_fallback=False)

    assert with_fallback["status"] == "OK"
    assert [row["provider_call_id"] for row in with_fallback["provider_calls"]] == [
        "provider-preflight-1",
        "provider-preflight-2",
    ]
    assert [row["provider"] for row in with_fallback["provider_calls"]] == [
        "openai",
        "anthropic",
    ]
    assert without_fallback["status"] == "FAILED"
    assert len(without_fallback["provider_calls"]) == 1


def test_resume_cost_adds_legacy_top_level_preflight_but_not_reconciled_v2_lane() -> None:
    provider_call = _provider_call(
        "PRE1",
        "provider_preflight",
        status="OK",
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        cost_estimate_usd=0.5,
    )
    legacy = {
        "rollups": {
            "lane_usage_totals": {
                "aggregate": {
                    "cost_microdollars": 2_000_000,
                    "cost_usd": "2.000000",
                }
            }
        },
        "provider_preflight": {"provider_calls": [provider_call]},
    }
    current = {
        "rollups": {
            "lane_usage_totals": {
                "lanes": {
                    "provider_preflight": {
                        "provider_call_attempts": 1,
                        "cost_microdollars": 500_000,
                    }
                },
                "aggregate": {
                    "cost_microdollars": 2_500_000,
                    "cost_usd": "2.500000",
                },
            }
        },
        "provider_preflight": {"provider_calls": [provider_call]},
    }

    assert _resume_realized_cost_microdollars(legacy) == 2_500_000
    assert _resume_realized_cost_microdollars(current) == 2_500_000


def test_third_generation_resume_uses_cumulative_spend_without_double_counting() -> None:
    artifact = {
        "spend_reconciliation": {
            "current_run_actual_cost_microdollars": 10_000_000,
            "prior_lineage_realized_cost_microdollars": 20_000_000,
            "combined_realized_cost_microdollars": 30_000_000,
            "combined_realized_reconciles": True,
        },
        "rollups": {
            "lane_usage_totals": {
                "aggregate": {
                    "cost_microdollars": 10_000_000,
                    "cost_usd": "10.000000",
                }
            }
        },
        # This diagnostic copy of the health call is already part of current
        # actual spend and must not be added again.
        "provider_preflight": {
            "provider_calls": [
                _provider_call(
                    "PRE1",
                    "provider_preflight",
                    status="OK",
                    input_tokens=10,
                    cached_input_tokens=0,
                    output_tokens=5,
                    cost_estimate_usd=1.0,
                )
            ]
        },
    }

    assert _resume_realized_cost_microdollars(artifact) == 30_000_000


def test_v2_cost_preflight_only_is_zero_spend_and_never_persists_spend_authority(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config
    from app.autonomous.terminal_cap_search import (
        authorization_from_whole_run_preflight,
    )

    get_config.cache_clear()
    events: list[str] = []
    resolution_kwargs: list[dict[str, object]] = []

    def resolve(**kwargs):
        events.append("candidate_freeze")
        resolution_kwargs.append(dict(kwargs))
        return SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["UNK"],
            loaded_tickers=["UNK"],
            source="sector_scan_db",
            cap_classifications={"UNK": {"market_cap_mm": None, "cap_source": "unknown"}},
        )

    monkeypatch.setattr("app.autonomous.sector_benchmark.resolve_sector_candidate_tickers", resolve)
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._run_provider_preflight_in_v2_policy",
        lambda: (_ for _ in ()).throw(AssertionError("provider call forbidden")),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("sector call forbidden")),
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-07-16",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        provider_preflight=True,
        cost_preflight_only=True,
        terminal_cap_search_ledger_path=tmp_path / "terminal-ledger.json",
    )

    assert events == ["candidate_freeze"]
    assert resolution_kwargs[0]["allow_live_market_data"] is False
    assert resolution_kwargs[0]["require_accepted_census"] is True
    assert isinstance(
        resolution_kwargs[0]["accepted_census_authority"],
        AcceptedCensusRunAuthority,
    )
    assert artifact["status"] == "COST_PREFLIGHT_ONLY"
    assert artifact["diagnostic_only"] is True
    assert artifact["sector_results"] == []
    assert artifact["provider_preflight"] == {
        "enabled": False,
        "status": "SKIPPED",
        "reason": "cost_preflight_only",
        "provider_calls": [],
    }
    preflight = artifact["all_sector_cost_preflight"]
    assert preflight["status"] == "WITHIN_CEILING_AWAITING_USER_AUTHORIZATION"
    assert preflight["spend_authorized"] is False
    assert preflight["reason_codes"] == ["COST_PREFLIGHT_ONLY_REQUIRES_SEPARATE_EXECUTION_REQUEST"]
    assert preflight["execution_bounds"]["filing_risk_llm_enabled"] is False
    assert preflight["execution_bounds"]["filing_risk_classification"] == (
        "DETERMINISTIC_KEYWORD_ONLY"
    )
    assert preflight["terminal_cap_search_attempts"] == 1
    assert preflight["unknown_cap_candidates"] == [{"sector": "energy", "ticker": "UNK"}]
    assert preflight["estimate"]["lane_costs"]["terminal_cap_search"]["model_calls"] == 1
    assert "terminal_cap_search_binding" not in preflight
    assert preflight["actual_usage"] == {
        "model_calls": 0,
        "search_calls": 0,
        "network_calls": 0,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }
    assert preflight["production_sector_scan_exercised"] is False
    assert preflight["screened_candidate_count"] == 0
    assert preflight["underwritten_candidate_count"] == 0
    assert preflight["actionable_candidate_count"] == 0
    assert preflight["membership_candidate_tickers"] == {"energy": ["UNK"]}
    assert preflight["execution_candidate_tickers"] == {"energy": ["UNK"]}
    assert preflight["deferred_by_bound_tickers"] == {"energy": []}
    assert preflight["excluded_candidate_tickers"] == {"energy": []}
    assert artifact["spend_reconciliation"]["current_run_actual_cost_usd"] == ("0.000000")
    assert not (tmp_path / "terminal-ledger.json").exists()

    paths = persist_autonomous_sector_benchmark(artifact)
    assert paths.cost_preflight_json is not None
    persisted = json.loads(paths.cost_preflight_json.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="diagnostic-only cost preflight"):
        authorization_from_whole_run_preflight(persisted)


def test_v2_cost_preflight_only_attaches_non_authorizing_mini_reprice(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["ONE"],
            loaded_tickers=["ONE"],
            source="accepted_census",
            cap_classifications={
                "ONE": {"market_cap_mm": 25_000.0, "cap_source": "accepted_census"}
            },
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark._run_provider_preflight_in_v2_policy",
        lambda: (_ for _ in ()).throw(AssertionError("provider call forbidden")),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("sector call forbidden")),
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["consumer_services"],
        as_of_date="2026-07-17",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        provider_preflight=False,
        cost_preflight_only=True,
        terminal_cap_search_max_attempts=0,
        diagnostic_reprice_model="gpt-5.4-mini",
    )

    preflight = artifact["all_sector_cost_preflight"]
    diagnostic = preflight["diagnostic_model_reprice"]
    assert artifact["status"] == "COST_PREFLIGHT_ONLY"
    assert preflight["provider_binding"] == {
        "provider": "openai",
        "model": "gpt-5.5",
        "fallback_allowed": False,
    }
    assert preflight["spend_authorized"] is False
    assert diagnostic["status"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["spend_authorized"] is False
    assert diagnostic["execution_compatible_with_current_v2_policy"] is False
    assert diagnostic["target_pricing_binding"]["model"] == "gpt-5.4-mini"
    assert diagnostic["cost_envelope"]["aggregate"]["model_calls"] == 46
    assert diagnostic["cost_envelope"]["aggregate"]["remaining_worst_case_cost_usd"] == "5.175000"
    assert diagnostic["actual_usage"]["cost_usd"] == "0.000000"
    assert preflight["actual_usage"]["cost_usd"] == "0.000000"


def test_v2_normal_run_builds_terminal_callback_from_same_preflight(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.config import get_config

    get_config.cache_clear()
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["UNK"],
            loaded_tickers=["UNK"],
            source="sector_scan_db",
            cap_classifications={"UNK": {"market_cap_mm": None, "cap_source": "unknown"}},
        ),
    )
    captured: dict = {}

    def runtime(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after callback construction")

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        runtime,
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date="2026-07-16",
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        provider_preflight=False,
        terminal_cap_search_ledger_path=tmp_path / "terminal-ledger.json",
    )

    callback = captured["terminal_cap_search"]
    assert callback.authorization.max_attempts == 1
    assert callback.authorization.run_id == artifact["run_id"]
    assert artifact["all_sector_cost_preflight"]["terminal_cap_search_attempts"] == 1
    assert artifact["terminal_cap_search_authorization"]["max_attempts"] == 1
    assert callback.attempt_count == 0


def test_v2_supplied_terminal_callback_is_rebound_to_exact_frozen_authority(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    from app.autonomous.all_sector_cost_preflight import (
        build_production_v2_cost_preflight,
    )
    from app.autonomous.sector_runtime import DEFAULT_SECTOR_OBJECTIVE
    from app.autonomous.terminal_cap_search import (
        authorization_from_whole_run_preflight,
        bind_v2_cost_preflight_for_terminal_cap_search,
        build_authorized_terminal_cap_search,
        whole_run_preflight_request_fingerprint,
    )
    from app.config import get_config

    get_config.cache_clear()
    as_of_date = "2026-07-16"
    ledger_path = tmp_path / "terminal-ledger.json"
    budget = AutonomousRunBudget(
        max_tool_calls=16,
        max_turns=6,
        max_cost_usd=None,
        timebox_seconds=None,
        max_candidates=None,
    )
    request_fingerprint = whole_run_preflight_request_fingerprint(
        sectors=["energy"],
        objective=DEFAULT_SECTOR_OBJECTIVE,
        as_of_date=as_of_date,
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        budget=budget.to_dict(),
        max_candidates=None,
    )
    stale_estimate = build_production_v2_cost_preflight(
        sector_candidate_counts={"energy": 2},
        terminal_cap_search_attempts=2,
        parent_max_turns=6,
    ).to_dict()
    stale_bound = bind_v2_cost_preflight_for_terminal_cap_search(
        {
            "artifact_type": "all_sector_execution_authorization_v1",
            "status": "AUTHORIZED",
            "spend_authorized": True,
            "reason_codes": [],
            "estimate": stale_estimate,
        },
        run_id="pre-freeze-external-run",
        authorized_at="2026-07-16T12:00:00+00:00",
        request_fingerprint=request_fingerprint,
        ledger_path=ledger_path,
    )
    supplied = build_authorized_terminal_cap_search(
        preflight=stale_bound,
        provider=object(),
        ledger_path=ledger_path,
        expected_request_fingerprint=request_fingerprint,
    )
    stale_authorization = supplied.authorization.to_dict()
    assert stale_authorization["max_attempts"] == 2

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["UNK"],
            loaded_tickers=["UNK"],
            source="sector_scan_db",
            cap_classifications={"UNK": {"market_cap_mm": None, "cap_source": "unknown"}},
        ),
    )
    captured: dict = {}

    def runtime(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after callback authority capture")

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_sector_autonomous_financial_analysis",
        runtime,
    )

    artifact = run_autonomous_sector_benchmark(
        sectors=["energy"],
        as_of_date=as_of_date,
        market_cap_focus="large_and_mega",
        pipeline_version="v2",
        budget=budget,
        provider_preflight=False,
        terminal_cap_search=supplied,
    )

    callback = captured["terminal_cap_search"]
    canonical = authorization_from_whole_run_preflight(
        artifact["all_sector_cost_preflight"]
    ).to_dict()
    artifact_authorization = dict(artifact["terminal_cap_search_authorization"])
    assert artifact_authorization.pop("preflight_artifact_path") is None
    assert artifact_authorization.pop("ledger_path") == str(ledger_path)
    assert callback is supplied
    assert callback.authorization.to_dict() == canonical
    assert artifact_authorization == canonical
    assert canonical != stale_authorization
    assert canonical["run_id"] == artifact["run_id"]
    assert canonical["request_fingerprint"] == artifact["request_fingerprint"]
    assert canonical["request_fingerprint"] != request_fingerprint
    assert canonical["allowed_tickers"] == ["UNK"]
    assert canonical["execution_fingerprint"] == artifact["execution_set_fingerprint"]
    assert canonical["max_attempts"] == 1
    assert canonical["max_tool_calls_per_attempt"] == 4
    assert canonical["terminal_cap_search_reserved_cost_usd"] == float(
        artifact["all_sector_cost_preflight"]["estimate"]["lane_costs"]["terminal_cap_search"][
            "cost_usd"
        ]
    )
    assert artifact["all_sector_cost_preflight"]["terminal_cap_search_ledger_path"] == str(
        ledger_path
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["authorization"] == canonical
    assert ledger["attempt_count"] == 0
