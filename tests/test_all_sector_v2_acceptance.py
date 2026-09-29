from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from app.autonomous.all_sector_v2_acceptance import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    AcceptanceThresholds,
    _parser,
    build_trusted_postmortem_manifest_hashes,
    evaluate_all_sector_v2_acceptance as _evaluate_all_sector_v2_acceptance,
    going_concern_false_provenance_manifest_sha256,
    load_trusted_manifest_hashes,
    load_trusted_semantic_manifest_hashes,
    run_all_sector_v2_acceptance,
)
from app.autonomous.competitive_frontier import build_competitive_frontier
from app.autonomous.run_contract import EvidenceReference, ToolCallRecord
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    CandidateDisposition,
    GateEvaluation,
    SECTOR_CONTRACT_VERSION_V2,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorSelectionValidation,
    ScreenResult,
    UnderwritingResult,
    build_v2_canonical_child_source_bindings,
    selection_validation_terminal_ledger_fingerprint,
)
from app.llm.synthesis_agent import _estimate_cost_usd
from app.sector.canonical_taxonomy import ACTIVE_SECTOR_LABELS


def _manifest_sha256(rows: list[dict]) -> str:
    identifiers = sorted(str(row["canonical_id"]) for row in rows)
    payload = json.dumps(identifiers, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_FIXTURE_CANONICAL_IDS = {
    "sparse_history": [
        "sparse_history:energy:AAA",
        "sparse_history:energy:BBB",
    ],
    "cap_price_losses": [
        "cap_price_loss:energy:AAA",
        "cap_price_loss:energy:BBB",
    ],
    "cached_annual_filings": [
        "cached_annual_filing:energy:AAA:0000001-26-000001",
        "cached_annual_filing:energy:BBB:0000002-26-000001",
    ],
    "extraction_gaps": [
        "extraction_gap:energy:BBB:0000002-26-000001",
    ],
    "previously_unreviewed": [
        "previously_unreviewed:enterprise_software:DDD",
        "previously_unreviewed:enterprise_software:EEE",
    ],
    "omitted_top_ten": [
        "omitted_top_ten:energy:FFF:1",
        "omitted_top_ten:energy:GGG:2",
    ],
    "going_concern": [
        "going_concern:false:energy:AAA:2026-07-15",
        "going_concern:false:energy:BBB:2026-07-15",
        "going_concern:true:registrant:REGP:4bd39566c08c23262daff27c1d01364a4485f604489482f78a59ae5c7024e1f4",
        "going_concern:true:consolidated_subsidiary:SUBP:a6a1106fc03011133695ef4b8defb5d77837d000cd87b80dce6abbd8a1307ff3",
    ],
}


_FALSE_GC_FIXTURES = [
    {
        "canonical_id": "going_concern:false:energy:AAA:2026-07-15",
        "fixture_kind": "false_positive",
        "sector": "energy",
        "ticker": "AAA",
        "gate_scorecard_as_of": "2026-07-15",
        "matched_context": "fixture false positive AAA",
        "content_revision": hashlib.sha256(b"fixture false positive AAA").hexdigest(),
        "matched_context_revision": hashlib.sha256(
            b"fixture false positive AAA"
        ).hexdigest(),
        "filing_content_revision": hashlib.sha256(
            b"fixture false positive AAA"
        ).hexdigest(),
        "accession": "acc-aaa",
        "form_type": "10-K",
        "filing_date": "2026-02-15",
        "issuer_cik": "1",
        "source_url": "test://filing/AAA",
        "evidence_ref_id": "sec-filing:AAA:acc-aaa",
        "blocked": False,
    },
    {
        "canonical_id": "going_concern:false:energy:BBB:2026-07-15",
        "fixture_kind": "false_positive",
        "sector": "energy",
        "ticker": "BBB",
        "gate_scorecard_as_of": "2026-07-15",
        "matched_context": "fixture false positive BBB",
        "content_revision": hashlib.sha256(b"fixture false positive BBB").hexdigest(),
        "matched_context_revision": hashlib.sha256(
            b"fixture false positive BBB"
        ).hexdigest(),
        "filing_content_revision": hashlib.sha256(
            b"fixture false positive BBB"
        ).hexdigest(),
        "accession": "acc-bbb",
        "form_type": "10-K",
        "filing_date": "2026-02-15",
        "issuer_cik": "2",
        "source_url": "test://filing/BBB",
        "evidence_ref_id": "sec-filing:BBB:acc-bbb",
        "blocked": False,
    },
]


def _identifier_manifest(identifiers: list[str]) -> str:
    rows = [{"canonical_id": identifier} for identifier in identifiers]
    return _manifest_sha256(rows)


THRESHOLDS = AcceptanceThresholds(
    sector_count=2,
    expected_sector_labels=("energy", "enterprise_software"),
    security_slots=5,
    sparse_history=2,
    cap_price_losses=2,
    cached_annual_filings=2,
    extraction_gaps=1,
    previously_unreviewed=2,
    omitted_top_ten=2,
    going_concern_false_positives=("AAA", "BBB"),
    trusted_evidence_manifest_sha256=tuple(
        (family, _identifier_manifest(identifiers))
        for family, identifiers in _FIXTURE_CANONICAL_IDS.items()
    ),
    trusted_semantic_manifest_sha256=(
        (
            "going_concern_false_provenance",
            str(going_concern_false_provenance_manifest_sha256(_FALSE_GC_FIXTURES)),
        ),
    ),
)


def evaluate_all_sector_v2_acceptance(
    payload: dict,
    *,
    evidence: dict | None = None,
    data_plane_report: dict | None = None,
    thresholds: AcceptanceThresholds | None = None,
) -> dict:
    """Persist synthetic product artifacts before exercising acceptance."""

    if not isinstance(payload.get("sector_artifacts"), list):
        return _evaluate_all_sector_v2_acceptance(
            payload,
            evidence=evidence,
            data_plane_report=data_plane_report,
            thresholds=thresholds,
        )
    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        benchmark = copy.deepcopy(payload)
        artifacts = benchmark.pop("sector_artifacts")
        indexed: list[dict] = []
        result_rows = [
            dict(row)
            for row in benchmark.get("sector_results") or []
            if isinstance(row, dict)
        ]
        by_sector = {str(row.get("sector") or ""): row for row in result_rows}
        for artifact in artifacts:
            sector = str(artifact.get("sector") or "unknown")
            relative = f"artifacts/{sector}.json"
            artifact_path = root / relative
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            row = by_sector.get(sector)
            if row is None:
                row = {"sector": sector}
                result_rows.append(row)
                by_sector[sector] = row
            row["artifact_path"] = relative
            row["artifact_sha256"] = digest
            indexed.append(
                {
                    "sector": sector,
                    "run_id": artifact.get("run_id"),
                    "artifact_path": relative,
                    "artifact_sha256": digest,
                }
            )
        benchmark["sector_results"] = result_rows
        fixed = benchmark.setdefault("fixed_cohort", {})
        fixed.update(
            {
                "schema_version": "all_sector_v2_fixed_cohort_v1",
                "sector_artifacts": indexed,
                "trusted_manifest_policy": (
                    "CALLER_OR_CONFIG_PINNED_NOT_BENCHMARK_SELF_DECLARED"
                ),
            }
        )
        if data_plane_report is not None:
            benchmark["data_plane_report"] = data_plane_report
        benchmark_path = root / "benchmark_summary.json"
        evidence_path = root / "acceptance_evidence.json"
        benchmark_path.write_text(json.dumps(benchmark, sort_keys=True), encoding="utf-8")
        evidence_path.write_text(
            json.dumps({"acceptance_evidence": evidence or {}}, sort_keys=True),
            encoding="utf-8",
        )
        return run_all_sector_v2_acceptance(
            benchmark_path,
            evidence_path=evidence_path,
            thresholds=thresholds,
        )


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _lane(
    tool_calls: int,
    provider_calls: int,
    input_tokens: int,
    output_tokens: int,
    cost_microdollars: int,
    *,
    cached_input_tokens: int = 0,
) -> dict[str, int]:
    return {
        "tool_call_attempts": tool_calls,
        "tool_calls_ok": tool_calls,
        "tool_calls_failed": 0,
        "provider_call_attempts": provider_calls,
        "provider_calls_ok": provider_calls,
        "provider_calls_failed": 0,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reserved_output_tokens": 0,
        "cost_microdollars": cost_microdollars,
    }


def _lane_usage() -> dict:
    lanes = {
        "parent_research": _lane(1, 1, 100, 10, 1_000, cached_input_tokens=20),
        "repair_fallback": _lane(1, 0, 0, 0, 0),
        "company_underwriting": {
            **_lane(2, 1, 200, 20, 1_420, cached_input_tokens=40),
            "tool_calls_ok": 1,
            "tool_calls_failed": 1,
        },
        "selected_company_validation": _lane(
            1, 1, 75, 7, 517, cached_input_tokens=15
        ),
        "provider_preflight": _lane(0, 1, 25, 0, 250),
        "terminal_cap_search": _lane(0, 0, 0, 0, 0),
    }
    aggregate = {
        key: sum(values[key] for values in lanes.values()) for key in next(iter(lanes.values()))
    }
    return {
        "lanes": lanes,
        "aggregate": aggregate,
        "aggregate_reconciles": True,
    }


def _reconcile_fixture_lane_totals(payload: dict) -> None:
    usage = payload["rollups"]["lane_usage_totals"]
    usage["aggregate"] = {
        field: sum(lane[field] for lane in usage["lanes"].values())
        for field in next(iter(usage["lanes"].values()))
    }


def _zero_fixture_provider_lane(payload: dict, lane_name: str) -> None:
    lane = payload["rollups"]["lane_usage_totals"]["lanes"][lane_name]
    for field in (
        "provider_call_attempts",
        "provider_calls_ok",
        "provider_calls_failed",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reserved_output_tokens",
        "cost_microdollars",
    ):
        lane[field] = 0
    _reconcile_fixture_lane_totals(payload)


def _pass_screen(contract_id: str, ticker: str) -> ScreenResult:
    content_revision = hashlib.sha256(
        f"fixture false positive {ticker}".encode("utf-8")
    ).hexdigest()
    issuer_cik = {"AAA": "1", "BBB": "2"}.get(ticker, "999")
    return ScreenResult(
        contract_id=contract_id,
        status="PASS",
        gate_evaluations=[
            GateEvaluation(
                contract_id=contract_id,
                rule_id="FIXTURE_GATE",
                status="PASS",
                applicable=True,
                observed_value=1,
                threshold=0,
                evidence_ref_id=f"screen:{ticker}",
            ),
            GateEvaluation(
                contract_id=contract_id,
                rule_id="GOING_CONCERN",
                status="PASS",
                applicable=True,
                observed_value={
                    "assertion": "NO_BLOCKABLE_ATTRIBUTED_ASSERTION",
                    "assertions": [],
                    "accession": f"acc-{ticker.lower()}",
                    "form_type": "10-K",
                    "filing_date": "2026-02-15",
                    "issuer_cik": issuer_cik,
                    "source_url": f"test://filing/{ticker}",
                    "content_revision": content_revision,
                },
                threshold="no current attributed going-concern assertion",
                evidence_ref_id=f"sec-filing:{ticker}:acc-{ticker.lower()}",
                evidence_url=f"test://filing/{ticker}",
            ),
        ],
    )


def _out_of_scope(ticker: str, issuer_cik: str) -> CandidateDisposition:
    reason = "SECONDARY_SECURITY"
    return CandidateDisposition(
        ticker=ticker,
        issuer_cik=issuer_cik,
        terminal_state="OUT_OF_SCOPE",
        scope_status="OUT_OF_SCOPE",
        screen_status="NOT_RUN",
        review_status="NOT_REQUIRED",
        reason_codes=[reason],
        frontier_status="NOT_ELIGIBLE",
        screen_result=ScreenResult(
            contract_id="fixture_v2",
            status="NOT_RUN",
            reason_codes=[reason],
        ),
        underwriting_result=UnderwritingResult(
            status="NOT_REQUIRED",
            reason_codes=["OUT_OF_SCOPE"],
        ),
    )


def _screened_out(ticker: str, issuer_cik: str) -> CandidateDisposition:
    contract_id = "fixture_energy_v2"
    reason = "SOURCE_BACKED_GATE_FAILURE"
    content_revision = hashlib.sha256(
        f"fixture false positive {ticker}".encode("utf-8")
    ).hexdigest()
    return CandidateDisposition(
        ticker=ticker,
        issuer_cik=issuer_cik,
        terminal_state="SCREENED_OUT",
        scope_status="IN_SCOPE",
        screen_status="FAIL",
        review_status="NOT_REQUIRED",
        reason_codes=[reason],
        frontier_status="NOT_ELIGIBLE",
        screen_result=ScreenResult(
            contract_id=contract_id,
            status="FAIL",
            gate_evaluations=[
                GateEvaluation(
                    contract_id=contract_id,
                    rule_id="FIXTURE_GATE",
                    status="FAIL",
                    applicable=True,
                    observed_value=0,
                    threshold=1,
                    evidence_ref_id=f"screen:{ticker}",
                    reason_code=reason,
                ),
                GateEvaluation(
                    contract_id=contract_id,
                    rule_id="GOING_CONCERN",
                    status="PASS",
                    applicable=True,
                    observed_value={
                        "assertion": "NO_BLOCKABLE_ATTRIBUTED_ASSERTION",
                        "assertions": [],
                        "accession": f"acc-{ticker.lower()}",
                        "form_type": "10-K",
                        "filing_date": "2026-02-15",
                        "issuer_cik": issuer_cik,
                        "source_url": f"test://filing/{ticker}",
                        "content_revision": content_revision,
                    },
                    threshold="no current attributed going-concern assertion",
                    evidence_ref_id=f"sec-filing:{ticker}:acc-{ticker.lower()}",
                    evidence_url=f"test://filing/{ticker}",
                ),
            ],
            reason_codes=[reason],
        ),
        underwriting_result=UnderwritingResult(
            status="NOT_REQUIRED",
            reason_codes=["SCREEN_FAILED"],
        ),
    )


def _selected_energy_artifact() -> dict:
    packet = SectorCompanyFinancialPacket(
        ticker="AAA",
        financial_status="COMPLETE",
        model_fit_status="FIT",
        data_quality_status="GOOD",
        score_components={"deterministic_score": 0.8},
    )
    scenario = SectorExpectedReturnScenario(
        scenario_id="AAA-base",
        ticker="AAA",
        scenario_name="base",
        horizon_years=3,
        current_price=20.0,
        estimated_future_value_per_share=30.0,
        annualized_return=0.1447,
    )
    prompt_packets = [
        SectorCompanyFinancialPacket(
            ticker="FFF",
            financial_status="COMPLETE",
            model_fit_status="FIT",
            data_quality_status="GOOD",
            score_components={"deterministic_score": 1.0},
        ),
        SectorCompanyFinancialPacket(
            ticker="GGG",
            financial_status="COMPLETE",
            model_fit_status="FIT",
            data_quality_status="GOOD",
            score_components={"deterministic_score": 0.9},
        ),
        packet,
    ]
    prompt_scenarios = [
        SectorExpectedReturnScenario(
            scenario_id="FFF-base",
            ticker="FFF",
            scenario_name="base",
            horizon_years=3,
            current_price=20.0,
            estimated_future_value_per_share=40.0,
            annualized_return=0.2599,
        ),
        SectorExpectedReturnScenario(
            scenario_id="GGG-base",
            ticker="GGG",
            scenario_name="base",
            horizon_years=3,
            current_price=20.0,
            estimated_future_value_per_share=35.0,
            annualized_return=0.2051,
        ),
        scenario,
    ]
    prompt_frontier = build_competitive_frontier(
        prompt_packets,
        prompt_scenarios,
        top_n=25,
        batch_size=3,
    )
    signal_snapshots = {
        ticker: {
            "ticker": ticker,
            "sector": "energy",
            "as_of_date": "2026-07-15",
        }
        for ticker in ("FFF", "GGG", "AAA")
    }
    source_binding = build_v2_canonical_child_source_bindings(
        sector="energy",
        as_of_date="2026-07-15",
        company_packets=prompt_packets,
        scenarios=prompt_scenarios,
        signal_packet_snapshots=signal_snapshots,
        frontier_candidate_tickers=["AAA"],
    )["AAA"]
    child_tools = [
        {
            "call_id": "C1",
            "tool_name": "filing_check",
            "tool_input": {},
            "rationale": "underwrite",
            "status": "OK",
            "evidence_ref_ids": ["CE1"],
            "lane": "company_underwriting",
        },
        {
            "call_id": "C2",
            "tool_name": "peer_check",
            "tool_input": {},
            "rationale": "underwrite",
            "status": "ERROR",
            "evidence_ref_ids": [],
            "lane": "company_underwriting",
        },
    ]
    child_evidence = [
        {
            "evidence_id": "CE1",
            "source_type": "filing",
            "source_label": "annual filing",
            "summary": "company-specific evidence",
            "ticker": "AAA",
            "tool_call_id": "C1",
            "confidence": "HIGH",
        }
    ]
    child_provider_usage = [
        {
            "provider_call_id": "CP1",
            "status": "OK",
            "lane": "company_underwriting",
            "provider": "openai",
            "model": "gpt-5.5",
            "schema_name": "autonomous_candidate_decision",
            "input_tokens": 200,
            "cached_input_tokens": 40,
            "output_tokens": 20,
            "reserved_output_tokens": 0,
            "estimated_tokens": False,
            "cost_estimate_usd": str(
                _estimate_cost_usd("gpt-5.5", 200, 20, cached_input_tokens=40)
            ),
        }
    ]
    child_artifact = {
        "request": {
            "run_id": "child-aaa",
            "as_of_date": "2026-07-15",
            "candidate_scope": {
                "mode": "single_candidate",
            "tickers": ["AAA"],
            "source_binding": source_binding,
            "signal_packet_snapshot": signal_snapshots["AAA"],
            },
        },
        "tool_calls": child_tools,
        "evidence": child_evidence,
        "candidate_decisions": [
            {"ticker": "AAA", "evidence_ref_ids": ["CE1"]}
        ],
        "provider_usage": child_provider_usage,
    }
    underwriting = UnderwritingResult(
        status="COMPLETED",
        verdict="ACTIONABLE",
        confidence="HIGH",
        evidence_ref_ids=["child-aaa:CE1"],
        tool_call_ids=["C1"],
        child_run_id="child-aaa",
    )
    disposition = CandidateDisposition(
        ticker="AAA",
        issuer_cik="1",
        terminal_state="UNDERWRITTEN",
        scope_status="IN_SCOPE",
        screen_status="PASS",
        review_status="COMPLETED",
        underwriting_verdict="ACTIONABLE",
        underwriting_confidence="HIGH",
        watchlist_eligible=True,
        frontier_status="REVIEWED",
        screen_result=_pass_screen("fixture_energy_v2", "AAA"),
        underwriting_result=underwriting,
    )
    screened_bbb = _screened_out("BBB", "2")
    validation_call = ToolCallRecord(
        call_id="validation-aaa:V1",
        tool_name="challenge_selection",
        tool_input={"ticker": "AAA"},
        rationale="independent selected-company challenge",
        status="OK",
        evidence_ref_ids=["validation-aaa:VE1"],
        lane="selected_company_validation",
    )
    validation_evidence = EvidenceReference(
        evidence_id="validation-aaa:VE1",
        source_type="filing",
        source_label="challenge evidence",
        summary="challenge passed",
        ticker="AAA",
        tool_call_id="validation-aaa:V1",
        confidence="HIGH",
    )
    validation = SectorSelectionValidation(
        status="VALIDATED",
        selected_ticker="AAA",
        validator_run_id="validation-aaa",
        validator_verdict="ACTIONABLE",
        evidence_ref_ids=["validation-aaa:VE1"],
        evidence=[validation_evidence],
        tool_calls=[validation_call],
        provider_usage=[
            {
                "provider_call_id": "validation-aaa:VP1",
                "validator_run_id": "validation-aaa",
                "status": "OK",
                "lane": "selected_company_validation",
                "provider": "openai",
                "model": "gpt-5.5",
                "schema_name": "sector_selection_validation",
                "input_tokens": 75,
                "cached_input_tokens": 15,
                "output_tokens": 7,
                "reserved_output_tokens": 0,
                "estimated_tokens": False,
                "cost_estimate_usd": str(
                    _estimate_cost_usd("gpt-5.5", 75, 7, cached_input_tokens=15)
                ),
            }
        ],
        source_binding=source_binding,
    )
    frontier = build_competitive_frontier(
        [packet],
        [scenario],
        reviewed_tickers=["AAA"],
    )
    frontier_payload = frontier.to_dict()
    frontier_payload.update(
        {
            "status": "CLOSED",
            "minimum_reviews_required": 1,
            "successful_review_count": 1,
            "attempted_tickers": ["AAA"],
            "failed_review_tickers": [],
        }
    )
    frontier_payload["source_bindings"] = {"AAA": source_binding}
    frontier_payload["signal_packet_snapshots"] = signal_snapshots
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="energy-v2",
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="fixture objective",
        as_of_date="2026-07-15",
        created_at="2026-07-16T00:00:00Z",
        status="COMPLETED",
        final_verdict="SELECTED",
        selected_ticker="AAA",
        confidence="HIGH",
        pipeline_version="v2",
        execution_status="COMPLETED",
        decision_status="COMPLETE",
        admitted_tickers=("AAA", "BBB"),
        candidate_dispositions=[disposition, screened_bbb],
        selection_validation=validation,
        competitive_frontier=frontier_payload,
        candidate_selection={
            "requested_tickers": ["AAA", "BBB"],
            "deterministic_pre_rank": prompt_frontier.to_dict(),
            "deterministic_prompt_tickers": list(prompt_frontier.top_tickers),
            "structural_gate_results": {
                "AAA": {"screen_result": disposition.screen_result.to_dict()},
                "BBB": {"screen_result": screened_bbb.screen_result.to_dict()},
            },
            "data_gap_repair": {
                "candidate_states": [
                    {
                        "ticker": "AAA",
                        "stage_states": {
                            "FACTS_AVAILABILITY": {"outcome": "AVAILABLE"},
                            "FILINGS": {"outcome": "AVAILABLE"},
                            "PARSING": {"outcome": "READABLE"},
                            "PRICE": {"outcome": "AVAILABLE"},
                        },
                        "packet_inputs": {
                            "facts": {"outcome": "AVAILABLE"},
                            "price": {
                                "outcome": "AVAILABLE",
                                "snapshot": {
                                    "price": 25.0,
                                    "source": "exchange",
                                    "as_of_date": "2026-07-15",
                                },
                            },
                        },
                    },
                    {
                        "ticker": "BBB",
                        "stage_states": {
                            "FACTS_AVAILABILITY": {
                                "outcome": "NEEDS_DATA",
                                "reason_code": "IFRS_FACTS_UNSUPPORTED",
                            },
                            "FILINGS": {"outcome": "AVAILABLE"},
                            "PARSING": {
                                "outcome": "NEEDS_DATA",
                                "reason_code": "ANNUAL_FILING_EXTRACTION_GAP",
                            },
                            "PRICE": {
                                "outcome": "NEEDS_DATA",
                                "reason_code": "ADR_RATIO_UNRESOLVED",
                            },
                        },
                        "packet_inputs": {
                            "facts": {
                                "outcome": "NEEDS_DATA",
                                "reason_code": "IFRS_FACTS_UNSUPPORTED",
                            },
                            "price": {
                                "outcome": "NEEDS_DATA",
                                "reason_code": "ADR_RATIO_UNRESOLVED",
                                "source_attempts": [
                                    {"source": "exchange", "outcome": "MISSING"}
                                ],
                            },
                        },
                    },
                ]
            },
        },
        company_packets=prompt_packets,
        expected_return_scenarios=prompt_scenarios,
        final_decision_prompt_context={"prompt_scoped_tickers": ["AAA"]},
        tool_calls=[
            ToolCallRecord("P1", "sector_research", {}, "research", "OK", lane="parent_research"),
            ToolCallRecord("R1", "repair", {}, "repair", "OK", lane="repair_fallback"),
        ],
        provider_usage=[
            {
                "provider_call_id": "PP1",
                "status": "OK",
                "lane": "parent_research",
                "provider": "openai",
                "model": "gpt-5.5",
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "reserved_output_tokens": 0,
                "cost_estimate_usd": "0.001",
            }
        ],
        company_autonomy_runs=[
            {
                "ticker": "AAA",
                "run_id": "child-aaa",
                "source_binding": source_binding,
                "artifact": child_artifact,
                "attempts": [child_artifact],
            }
        ],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )
    return artifact.to_dict()


def _failed_software_artifact() -> dict:
    ready = CandidateDisposition(
        ticker="DDD",
        issuer_cik="4",
        terminal_state="READY_FOR_UNDERWRITING",
        scope_status="IN_SCOPE",
        screen_status="PASS",
        review_status="NOT_STARTED",
        watchlist_eligible=True,
        frontier_status="LIVE",
        screen_result=_pass_screen("fixture_software_v2", "DDD"),
    )
    needs_data = CandidateDisposition(
        ticker="EEE",
        issuer_cik="5",
        terminal_state="NEEDS_DATA",
        scope_status="IN_SCOPE",
        screen_status="INCOMPLETE",
        review_status="NOT_STARTED",
        watchlist_eligible=True,
        reason_codes=["IFRS_FACTS_UNSUPPORTED"],
        frontier_status="UNRESOLVED",
        screen_result=ScreenResult(
            contract_id="fixture_software_v2",
            status="INCOMPLETE",
            gate_evaluations=[
                GateEvaluation(
                    contract_id="fixture_software_v2",
                    rule_id="FACTS_AVAILABILITY",
                    status="INCOMPLETE",
                    applicable=True,
                    reason_code="IFRS_FACTS_UNSUPPORTED",
                )
            ],
            reason_codes=["IFRS_FACTS_UNSUPPORTED"],
        ),
    )
    artifact = AutonomousSectorFinancialRunArtifact(
        run_id="software-v2",
        sector="enterprise_software",
        market_cap_focus="large_and_mega",
        objective="fixture objective",
        as_of_date="2026-07-15",
        created_at="2026-07-16T00:00:00Z",
        status="FAILED",
        final_verdict=None,
        selected_ticker=None,
        confidence=None,
        pipeline_version="v2",
        execution_status="FAILED",
        decision_status="INCOMPLETE",
        admitted_tickers=("DDD", "EEE"),
        candidate_dispositions=[_out_of_scope("CCC", "3"), ready, needs_data],
        candidate_selection={"requested_tickers": ["CCC", "DDD", "EEE"]},
        degraded_states=["RUNTIME_FAILED"],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )
    return artifact.to_dict()


def _passing_payload() -> tuple[dict, dict]:
    energy = _selected_energy_artifact()
    software = _failed_software_artifact()
    payload = {
        "artifact_type": "autonomous_sector_benchmark_v2",
        "pipeline_version": "v2",
        "run_id": "benchmark-v2",
        "as_of_date": "2026-07-15",
        "market_cap_focus": "large_and_mega",
        "objective": "fixture objective",
        "sectors": ["energy", "enterprise_software"],
        "fixed_cohort": {"security_count": 5, "issuer_count": 5},
        "sector_artifacts": [energy, software],
        "sector_results": [
            {
                "sector": artifact["sector"],
                "run_id": artifact["run_id"],
                "pipeline_version": artifact["pipeline_version"],
                "status": artifact["status"],
                "execution_status": artifact["execution_status"],
                "decision_status": artifact["decision_status"],
                "final_verdict": artifact["final_verdict"],
                "selected_ticker": artifact["selected_ticker"],
            }
            for artifact in (energy, software)
        ],
        "provider_preflight": {
            "provider_calls": [
                {
                    "provider_call_id": "PF1",
                    "status": "OK",
                    "lane": "provider_preflight",
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 25,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reserved_output_tokens": 0,
                    "cost_estimate_usd": "0.00025",
                }
            ]
        },
        "rollups": {"lane_usage_totals": _lane_usage()},
    }
    evidence = {
        "sparse_history": [
            {
                "canonical_id": "sparse_history:energy:AAA",
                "sector": "energy",
                "ticker": "AAA",
                "outcome": "AVAILABLE",
            },
            {
                "canonical_id": "sparse_history:energy:BBB",
                "sector": "energy",
                "ticker": "BBB",
                "outcome": "NEEDS_DATA",
                "reason_code": "IFRS_FACTS_UNSUPPORTED",
            },
        ],
        "cap_price_losses": [
            {
                "canonical_id": "cap_price_loss:energy:AAA",
                "sector": "energy",
                "ticker": "AAA",
                "outcome": "AVAILABLE",
                "price": 25.0,
                "source": "exchange",
                "as_of_date": "2026-07-15",
            },
            {
                "canonical_id": "cap_price_loss:energy:BBB",
                "sector": "energy",
                "ticker": "BBB",
                "outcome": "NEEDS_DATA",
                "reason_code": "ADR_RATIO_UNRESOLVED",
                "source_attempts": [{"source": "exchange", "outcome": "MISSING"}],
            },
        ],
        "cached_annual_filings": [
            {
                "canonical_id": "cached_annual_filing:energy:AAA:0000001-26-000001",
                "sector": "energy",
                "ticker": "AAA",
                "accession": "0000001-26-000001",
                "recognized": True,
            },
            {
                "canonical_id": "cached_annual_filing:energy:BBB:0000002-26-000001",
                "sector": "energy",
                "ticker": "BBB",
                "accession": "0000002-26-000001",
                "outcome": "AVAILABLE",
            },
        ],
        "extraction_gaps": [
            {
                "canonical_id": "extraction_gap:energy:BBB:0000002-26-000001",
                "sector": "energy",
                "ticker": "BBB",
                "accession": "0000002-26-000001",
                "outcome": "NEEDS_DATA",
                "reason_code": "ANNUAL_FILING_EXTRACTION_GAP",
            }
        ],
        "previously_unreviewed": [
            {
                "canonical_id": "previously_unreviewed:enterprise_software:DDD",
                "sector": "enterprise_software",
                "ticker": "DDD",
                "visible": True,
                "terminal_state": "READY_FOR_UNDERWRITING",
            },
            {
                "canonical_id": "previously_unreviewed:enterprise_software:EEE",
                "sector": "enterprise_software",
                "ticker": "EEE",
                "visible": True,
                "terminal_state": "NEEDS_DATA",
            },
        ],
        "omitted_top_ten": [
            {
                "canonical_id": "omitted_top_ten:energy:FFF:1",
                "sector": "energy",
                "ticker": "FFF",
                "rank": 1,
                "included_in_ranked_prompt_context": True,
            },
            {
                "canonical_id": "omitted_top_ten:energy:GGG:2",
                "sector": "energy",
                "ticker": "GGG",
                "rank": 2,
                "included_in_ranked_prompt_context": True,
            },
        ],
        "going_concern": {
            "false_positive_fixtures": copy.deepcopy(_FALSE_GC_FIXTURES),
            "true_positive_controls": [
                {
                    "canonical_id": "going_concern:true:registrant:REGP:4bd39566c08c23262daff27c1d01364a4485f604489482f78a59ae5c7024e1f4",
                    "fixture_kind": "true_positive",
                    "ticker": "REGP",
                    "accession": "acc-0",
                    "form_type": "10-K",
                    "filing_date": "2026-02-15",
                    "subject": "REGISTRANT",
                    "content_revision": "4bd39566c08c23262daff27c1d01364a4485f604489482f78a59ae5c7024e1f4",
                    "assertion_mode": "AFFIRMATIVE_CURRENT",
                    "evidence_ref_id": "sec-filing:REGP:acc-0",
                    "status": "FAIL",
                    "reason_code": "QUARANTINE_STRUCTURAL:GOING_CONCERN",
                    "blockable": True,
                    "blocked": True,
                },
                {
                    "canonical_id": "going_concern:true:consolidated_subsidiary:SUBP:a6a1106fc03011133695ef4b8defb5d77837d000cd87b80dce6abbd8a1307ff3",
                    "fixture_kind": "true_positive",
                    "ticker": "SUBP",
                    "accession": "acc-0",
                    "form_type": "10-K",
                    "filing_date": "2026-02-15",
                    "subject": "CONSOLIDATED_SUBSIDIARY",
                    "content_revision": "a6a1106fc03011133695ef4b8defb5d77837d000cd87b80dce6abbd8a1307ff3",
                    "assertion_mode": "AFFIRMATIVE_CURRENT",
                    "evidence_ref_id": "sec-filing:SUBP:acc-0",
                    "status": "FAIL",
                    "reason_code": "QUARANTINE_STRUCTURAL:GOING_CONCERN",
                    "blockable": True,
                    "blocked": True,
                },
            ],
        },
    }
    manifest_rows = {
        "sparse_history": evidence["sparse_history"],
        "cap_price_losses": evidence["cap_price_losses"],
        "cached_annual_filings": evidence["cached_annual_filings"],
        "extraction_gaps": evidence["extraction_gaps"],
        "previously_unreviewed": evidence["previously_unreviewed"],
        "omitted_top_ten": evidence["omitted_top_ten"],
        "going_concern": [
            *evidence["going_concern"]["false_positive_fixtures"],
            *evidence["going_concern"]["true_positive_controls"],
        ],
    }
    payload["fixed_cohort"]["evidence_manifest_sha256"] = {
        family: _manifest_sha256(rows) for family, rows in manifest_rows.items()
    }
    return payload, evidence


def _checks(report: dict) -> dict[str, dict]:
    return {row["id"]: row for row in report["checks"]}


def test_complete_evidence_passes_every_release_acceptance_check() -> None:
    payload, evidence = _passing_payload()

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )

    assert report["status"] == PASS
    assert report["network_calls"] == 0
    assert report["llm_calls"] == 0
    assert report["status_counts"] == {PASS: 17, FAIL: 0, NOT_EVALUABLE: 0}
    assert {row["status"] for row in report["checks"]} == {PASS}
    assert _checks(report)["security_and_issuer_counts"]["actual"] == {
        "security_count": 5,
        "issuer_count": 5,
        "derived_security_count": 5,
        "derived_issuer_count": 5,
        "unresolved_issuer_identity": 0,
        "declared_derived_mismatch": False,
    }


def test_default_acceptance_requires_the_exact_34_active_sector_labels() -> None:
    thresholds = AcceptanceThresholds()

    assert thresholds.sector_count == 34
    assert thresholds.expected_sector_labels == ACTIVE_SECTOR_LABELS
    assert len(thresholds.expected_sector_labels) == 34


def test_matching_sector_count_with_the_wrong_exact_set_fails_acceptance() -> None:
    payload, evidence = _passing_payload()
    payload["sectors"][1] = "utilities"
    payload["sector_artifacts"][1]["sector"] = "utilities"
    payload["sector_results"][1]["sector"] = "utilities"

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    coverage = _checks(report)["sector_coverage"]

    assert report["status"] == FAIL
    assert coverage["status"] == FAIL
    assert coverage["expected"] == {
        "sector_count": 2,
        "sector_labels": ["energy", "enterprise_software"],
    }
    assert coverage["actual"] == {
        "sector_count": 2,
        "sector_labels": ["energy", "utilities"],
    }
    assert coverage["details"] == {
        "missing_sector_labels": ["enterprise_software"],
        "unexpected_sector_labels": ["utilities"],
    }


def test_explicit_custom_thresholds_can_preserve_historical_count_only_coverage() -> None:
    payload, evidence = _passing_payload()
    payload["sectors"][1] = "utilities"
    payload["sector_artifacts"][1]["sector"] = "utilities"
    payload["sector_results"][1]["sector"] = "utilities"
    count_only_thresholds = replace(THRESHOLDS, expected_sector_labels=None)

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=count_only_thresholds,
    )
    coverage = _checks(report)["sector_coverage"]

    assert coverage["status"] == PASS
    assert coverage["expected"] == 2
    assert coverage["actual"] == 2
    assert coverage["details"] == {"sectors": ["energy", "utilities"]}


def test_absent_evidence_is_not_evaluable_never_implicit_pass() -> None:
    report = evaluate_all_sector_v2_acceptance({}, thresholds=THRESHOLDS)
    checks = _checks(report)

    assert report["status"] == NOT_EVALUABLE
    assert checks["input_integrity"]["status"] == PASS
    assert checks["security_slot_reconciliation"]["status"] == NOT_EVALUABLE
    assert checks["sparse_history_reconciliation"]["status"] == NOT_EVALUABLE
    assert checks["lane_and_cost_reconciliation"]["status"] == NOT_EVALUABLE


def test_present_but_incomplete_evidence_fails_instead_of_becoming_not_evaluable() -> None:
    payload, evidence = _passing_payload()
    evidence["sparse_history"] = []
    evidence["going_concern"]["false_positive_fixtures"][0]["source_url"] = (
        "test://forged-source"
    )
    payload["sector_artifacts"][1]["final_verdict"] = "NO_SELECTION"
    payload["rollups"]["lane_usage_totals"]["aggregate"]["tool_call_attempts"] += 1

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    checks = _checks(report)

    assert report["status"] == FAIL
    assert checks["sparse_history_reconciliation"]["status"] == FAIL
    assert checks["going_concern_regressions"]["status"] == FAIL
    assert checks["no_false_no_selection_on_execution_failure"]["status"] == FAIL
    assert checks["lane_and_cost_reconciliation"]["status"] == NOT_EVALUABLE


def test_duplicate_admitted_issuer_and_unproved_underwriting_both_fail() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    energy["candidate_dispositions"][1]["issuer_cik"] = "1"
    child = energy["company_autonomy_runs"][0]
    child["artifact"]["tool_calls"][0]["status"] = "ERROR"
    child["artifact"]["tool_calls"].pop()
    child["attempts"][0]["tool_calls"][0]["status"] = "ERROR"
    child["attempts"][0]["tool_calls"].pop()

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    checks = _checks(report)

    assert report["status"] == FAIL
    assert checks["one_terminal_disposition_per_admitted_issuer"]["status"] == FAIL
    assert checks["actionable_underwriting_and_validation"]["status"] == FAIL
    assert checks["company_child_call_reconciliation"]["status"] == FAIL
    assert checks["company_child_call_reconciliation"]["actual"] == {
        "company_child_runs": 1,
        "raw_tool_call_attempts": 1,
        "benchmark_company_underwriting_tool_call_attempts": 2,
        "unresolved_child_runs": 0,
    }
    assert checks["actionable_underwriting_and_validation"]["actual"] == {
        "completed_underwriting": 1,
        "completed_validation": 1,
        "completed_proofs": 2,
        "compliant": 1,
        "violations": 1,
    }


def test_data_plane_count_summaries_cannot_produce_release_pass() -> None:
    payload, evidence = _passing_payload()
    evidence.pop("sparse_history")
    evidence.pop("cap_price_losses")
    evidence.pop("cached_annual_filings")
    evidence.pop("extraction_gaps")
    data_plane = {
        "artifact_type": "all_sector_v2_data_plane_reconciliation_report",
        "status": "IGNORED",
        "summary": {
            "sparse_history": {"total": 2, "repaired": 1, "precise_needs_data": 1},
            "cap_price_losses": {
                "total": 2,
                "recovered_into_usd_packets": 1,
                "source_exhausted_precise_needs_data": 1,
            },
            "cached_annual_filings": {"total": 2, "recognized": 2},
            "remaining_extraction_gaps": {
                "total": 1,
                "repaired": 0,
                "precise_extraction_gap": 1,
            },
        },
    }

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        data_plane_report=data_plane,
        thresholds=THRESHOLDS,
    )

    assert report["status"] == NOT_EVALUABLE
    checks = _checks(report)
    assert checks["sparse_history_reconciliation"]["evidence_source"] == (
        "data_plane_report.summary.sparse_history"
    )
    assert checks["sparse_history_reconciliation"]["status"] == NOT_EVALUABLE
    assert checks["cap_price_loss_reconciliation"]["status"] == NOT_EVALUABLE
    assert checks["cached_annual_filing_recognition"]["status"] == NOT_EVALUABLE
    assert checks["extraction_gap_classification"]["status"] == NOT_EVALUABLE


def test_self_reconciling_declared_lane_forgery_fails_raw_recomputation() -> None:
    payload, evidence = _passing_payload()
    usage = payload["rollups"]["lane_usage_totals"]
    usage["lanes"]["company_underwriting"]["tool_call_attempts"] += 1
    usage["lanes"]["company_underwriting"]["tool_calls_ok"] += 1
    usage["aggregate"]["tool_call_attempts"] += 1
    usage["aggregate"]["tool_calls_ok"] += 1

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["lane_and_cost_reconciliation"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert check["actual"]["producer_aggregate_reconciles"] is True
    assert check["actual"]["mismatches"] == {
        "company_underwriting.tool_call_attempts": {"recomputed": 2, "declared": 3},
        "company_underwriting.tool_calls_ok": {"recomputed": 1, "declared": 2},
        "aggregate.tool_call_attempts": {"recomputed": 5, "declared": 6},
        "aggregate.tool_calls_ok": {"recomputed": 4, "declared": 5},
    }


def test_terminal_search_reserve_is_folded_once_into_provider_cost() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    energy["candidate_selection"]["data_gap_repair"][
        "terminal_cap_search_usage_records"
    ] = [
            {
                "call_type": "web_search_call_reserve",
                "authorization_run_id": "terminal-1",
                "attempt_number": 1,
                    "ticker": "AAA",
                    "attempt_status": "PENDING",
                    "call_id": "reserved-web-1",
                    "cost_estimate_usd": "0.0004",
            },
            {
                "call_type": "responses_model",
                "provider_call_id": "terminal-response:test:1",
                "provider": "openai",
                "model": "gpt-5.5",
                "authorization_run_id": "terminal-1",
                "attempt_number": 1,
                "ticker": "AAA",
                "attempt_status": "PENDING",
                "input_tokens": 30,
                "cached_input_tokens": 0,
                "output_tokens": 3,
                "reserved_output_tokens": 0,
                "cost_estimate_usd": "0.0003",
            },
        ]
    usage = payload["rollups"]["lane_usage_totals"]
    usage["lanes"]["terminal_cap_search"] = {
        **_lane(0, 1, 30, 3, 700),
        "provider_calls_ok": 0,
        "provider_calls_failed": 1,
    }
    usage["aggregate"] = {
        field: sum(lane[field] for lane in usage["lanes"].values())
        for field in next(iter(usage["lanes"].values()))
    }

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["lane_and_cost_reconciliation"]

    assert report["status"] == PASS
    assert check["status"] == PASS
    assert check["actual"]["recomputed_aggregate"]["tool_call_attempts"] == 5
    assert check["actual"]["recomputed_aggregate"]["provider_call_attempts"] == 5
    assert check["actual"]["recomputed_aggregate"]["cost_microdollars"] == 3_887


def test_orphan_terminal_search_reserve_cannot_disappear_from_accounting() -> None:
    payload, evidence = _passing_payload()
    payload["sector_artifacts"][0]["candidate_selection"]["data_gap_repair"][
        "terminal_cap_search_usage_records"
    ] = [
        {
            "call_type": "web_search_call_reserve",
            "authorization_run_id": "orphan-terminal",
            "attempt_number": 1,
            "ticker": "AAA",
            "attempt_status": "PENDING",
            "call_id": "orphan-reserved-web-1",
            "cost_estimate_usd": "0.75",
        }
    ]

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["lane_and_cost_reconciliation"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert any(
        "orphan-reserve-group:orphan-terminal:1:AAA" in item
        for item in check["details"]["raw_record_error_examples"]
    )


def test_handwritten_minimal_v2_forgery_cannot_pass() -> None:
    payload, evidence = _passing_payload()
    payload["sector_artifacts"] = [
        {
            "run_id": "energy-v2",
            "sector": "energy",
            "pipeline_version": "v2",
            "execution_status": "COMPLETED",
            "decision_status": "COMPLETE",
            "final_verdict": "SELECTED",
            "selected_ticker": "AAA",
            "selection_validation": {
                "status": "VALIDATED",
                "selected_ticker": "AAA",
            },
            "company_autonomy_runs": [{"ticker": "AAA", "tool_call_attempts": 2}],
            "candidate_dispositions": [
                {
                    "ticker": "AAA",
                    "issuer_cik": "1",
                    "scope_status": "IN_SCOPE",
                    "screen_status": "PASS",
                    "review_status": "COMPLETED",
                    "terminal_state": "UNDERWRITTEN",
                    "underwriting_verdict": "ACTIONABLE",
                },
                {
                    "ticker": "BBB",
                    "issuer_cik": "2",
                    "scope_status": "IN_SCOPE",
                    "screen_status": "FAIL",
                    "review_status": "NOT_REQUIRED",
                    "terminal_state": "SCREENED_OUT",
                    "reason_codes": ["FORGED_GATE"],
                },
            ],
        },
        {
            "run_id": "software-v2",
            "sector": "enterprise_software",
            "pipeline_version": "v2",
            "execution_status": "FAILED",
            "decision_status": "INCOMPLETE",
            "final_verdict": None,
            "selected_ticker": None,
            "candidate_dispositions": [
                {
                    "ticker": "CCC",
                    "issuer_cik": "3",
                    "scope_status": "OUT_OF_SCOPE",
                    "terminal_state": "OUT_OF_SCOPE",
                    "reason_codes": ["SECONDARY_SECURITY"],
                }
            ],
        },
    ]
    evidence["security_slots"] = [
        {"sector": artifact["sector"], **row}
        for artifact in payload["sector_artifacts"]
        for row in artifact["candidate_dispositions"]
    ]

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    checks = _checks(report)

    assert report["status"] == FAIL
    assert checks["artifact_contract_and_binding"]["status"] == FAIL
    assert checks["security_slot_reconciliation"]["status"] == NOT_EVALUABLE
    assert checks["actionable_underwriting_and_validation"]["status"] == FAIL
    assert checks["lane_and_cost_reconciliation"]["status"] == NOT_EVALUABLE


def test_historical_boolean_rows_without_canonical_cohort_ids_fail() -> None:
    payload, evidence = _passing_payload()
    evidence["sparse_history"][0].pop("canonical_id")

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["sparse_history_reconciliation"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert check["actual"]["cohort_manifest"]["missing_canonical_id_rows"] == [0]


def test_fabricated_rows_and_matching_self_declared_hash_fail_default_trust() -> None:
    payload, evidence = _passing_payload()
    forged_rows = [
        {
            "canonical_id": f"sparse_history:forged:FORGED{index:03d}",
            "ticker": f"FORGED{index:03d}",
            "outcome": "AVAILABLE",
        }
        for index in range(49)
    ]
    evidence["sparse_history"] = forged_rows
    forged_hash = _manifest_sha256(forged_rows)
    payload["fixed_cohort"]["evidence_manifest_sha256"]["sparse_history"] = forged_hash

    report = evaluate_all_sector_v2_acceptance(payload, evidence=evidence)
    check = _checks(report)["sparse_history_reconciliation"]
    binding = check["actual"]["cohort_manifest"]

    assert check["status"] == FAIL
    assert binding["declared_manifest_sha256"] == forged_hash
    assert binding["actual_manifest_sha256"] != forged_hash
    assert binding["trusted_manifest_sha256"] != forged_hash
    assert binding["declared_trusted_mismatch"] is True
    assert binding["unresolvable_identity_rows"] == list(range(30))


def test_trusted_postmortem_manifest_builder_uses_source_ledger_identity(
    tmp_path: Path,
) -> None:
    (tmp_path / "company_funnel.csv").write_text(
        "as_of_date,sector,ticker,financial_history_excluded,data_quality_status,"
        "packet_created,company_autonomy_reviewed\n"
        "2026-07-15,energy,aaa,True,MISSING_PRICE,True,False\n",
        encoding="utf-8",
    )
    (tmp_path / "filing_availability_cache_audit.csv").write_text(
        "as_of_date,sector,ticker,availability_bucket,audit_disposition,"
        "fallback_full_10k_risk_text_chars,available_source_accession\n"
        "2026-07-15,energy,aaa,READABLE_RISK_SECTION,"
        "FILING_EXISTS_EXTRACTION_GAP,,0000001-26-000001\n",
        encoding="utf-8",
    )
    thresholds = AcceptanceThresholds(
        sparse_history=1,
        cap_price_losses=1,
        cached_annual_filings=1,
        extraction_gaps=1,
        previously_unreviewed=1,
        trusted_evidence_manifest_sha256=(),
    )

    manifests = build_trusted_postmortem_manifest_hashes(
        tmp_path,
        thresholds=thresholds,
    )

    assert manifests == {
        "sparse_history": _identifier_manifest(["sparse_history:energy:AAA"]),
        "cap_price_losses": _identifier_manifest(["cap_price_loss:energy:AAA"]),
        "cached_annual_filings": _identifier_manifest(
            ["cached_annual_filing:energy:AAA:0000001-26-000001"]
        ),
        "extraction_gaps": _identifier_manifest(
            ["extraction_gap:energy:AAA:0000001-26-000001"]
        ),
        "previously_unreviewed": _identifier_manifest(
            ["previously_unreviewed:energy:AAA"]
        ),
    }


def test_selected_validation_cannot_share_a_forged_frontier_binding() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    forged = copy.deepcopy(energy["selection_validation"]["source_binding"])
    forged["company_packet_fingerprint"] = "c" * 64
    energy["selection_validation"]["source_binding"] = forged
    energy["competitive_frontier"]["source_bindings"]["AAA"] = forged

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "terminal ledger fingerprint" in " ".join(check["details"]["error_examples"])


def test_caller_supplied_fake_signal_fingerprint_map_cannot_replace_snapshot_proof() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    forged = copy.deepcopy(energy["competitive_frontier"]["source_bindings"]["AAA"])
    forged["signal_packet_fingerprint"] = "0" * 64
    forged["cohort_signal_packet_fingerprints"]["AAA"] = "0" * 64
    energy["competitive_frontier"]["source_bindings"]["AAA"] = forged
    validation = energy["selection_validation"]
    validation["source_binding"] = copy.deepcopy(forged)
    validation["terminal_ledger_fingerprint"] = (
        selection_validation_terminal_ledger_fingerprint(validation)
    )
    child = energy["company_autonomy_runs"][0]
    child["source_binding"] = copy.deepcopy(forged)
    for ledger in (child["artifact"], child["attempts"][0]):
        ledger["request"]["candidate_scope"]["source_binding"] = copy.deepcopy(forged)

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "derive from persisted signal snapshots" in " ".join(
        check["details"]["error_examples"]
    )


def test_validation_cannot_borrow_underwriting_child_run_identity() -> None:
    payload, evidence = _passing_payload()
    validation = payload["sector_artifacts"][0]["selection_validation"]
    validation["validator_run_id"] = "child-aaa"
    validation["evidence_ref_ids"] = ["child-aaa:VE1"]
    validation["evidence"][0]["evidence_id"] = "child-aaa:VE1"
    validation["evidence"][0]["tool_call_id"] = "child-aaa:V1"
    validation["tool_calls"][0]["call_id"] = "child-aaa:V1"
    validation["tool_calls"][0]["evidence_ref_ids"] = ["child-aaa:VE1"]
    validation["provider_usage"][0]["provider_call_id"] = "child-aaa:VP1"
    validation["provider_usage"][0]["validator_run_id"] = "child-aaa"
    validation["terminal_ledger_fingerprint"] = (
        selection_validation_terminal_ledger_fingerprint(validation)
    )

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert "must differ from every underwriting child_run_id" in " ".join(
        check["details"]["error_examples"]
    )


def test_validation_cannot_borrow_unlinked_company_child_run_identity() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    orphan = copy.deepcopy(energy["company_autonomy_runs"][0])
    orphan["run_id"] = "orphan-child"
    for ledger in (orphan["artifact"], *orphan["attempts"]):
        ledger["request"]["run_id"] = "orphan-child"
    energy["company_autonomy_runs"].append(orphan)

    validation = energy["selection_validation"]
    validation["validator_run_id"] = "orphan-child"
    validation["evidence_ref_ids"] = ["orphan-child:VE1"]
    validation["evidence"][0]["evidence_id"] = "orphan-child:VE1"
    validation["evidence"][0]["tool_call_id"] = "orphan-child:V1"
    validation["tool_calls"][0]["call_id"] = "orphan-child:V1"
    validation["tool_calls"][0]["evidence_ref_ids"] = ["orphan-child:VE1"]
    validation["provider_usage"][0]["provider_call_id"] = "orphan-child:VP1"
    validation["provider_usage"][0]["validator_run_id"] = "orphan-child"
    validation["terminal_ledger_fingerprint"] = (
        selection_validation_terminal_ledger_fingerprint(validation)
    )

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "must differ from every underwriting child_run_id" in " ".join(
        check["details"]["error_examples"]
    )


def test_validation_provider_row_cannot_borrow_a_foreign_run_identity() -> None:
    payload, evidence = _passing_payload()
    validation = payload["sector_artifacts"][0]["selection_validation"]
    validation["provider_usage"][0]["validator_run_id"] = "child-aaa"
    validation["terminal_ledger_fingerprint"] = (
        selection_validation_terminal_ledger_fingerprint(validation)
    )

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert "validator-bound provider usage" in " ".join(
        check["details"]["error_examples"]
    )


def test_selected_underwriting_requires_top_level_frontier_source_binding() -> None:
    payload, evidence = _passing_payload()
    payload["sector_artifacts"][0]["company_autonomy_runs"][0].pop(
        "source_binding"
    )

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "child source_binding" in " ".join(check["details"]["error_examples"])


def test_selected_underwriting_requires_nested_frontier_source_binding() -> None:
    payload, evidence = _passing_payload()
    child = payload["sector_artifacts"][0]["company_autonomy_runs"][0]
    nested_binding = child["artifact"]["request"]["candidate_scope"][
        "source_binding"
    ]
    nested_binding["signal_packet_fingerprint"] = "0" * 64
    child["attempts"][0]["request"]["candidate_scope"]["source_binding"][
        "signal_packet_fingerprint"
    ] = "0" * 64

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "nested child source_binding" in " ".join(
        check["details"]["error_examples"]
    )


def test_sector_result_missing_identity_field_breaks_source_binding() -> None:
    payload, evidence = _passing_payload()
    payload["sector_results"][0].pop("decision_status")

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "energy:sector_results.decision_status:missing" in check["details"][
        "error_examples"
    ]


def test_malformed_raw_usage_numbers_are_errors_not_zero_cost() -> None:
    payload, evidence = _passing_payload()
    provider = payload["sector_artifacts"][0]["provider_usage"][0]
    provider["input_tokens"] = -1
    provider["cost_estimate_usd"] = "NaN"

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["lane_and_cost_reconciliation"]
    errors = check["details"]["raw_record_error_examples"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert any("input_tokens:must-be-non-negative-integer" in item for item in errors)
    assert any("cost_estimate_usd:must-be-finite-non-negative" in item for item in errors)


def test_loader_follows_explicit_relative_artifact_and_preserves_inputs(tmp_path: Path) -> None:
    payload, evidence = _passing_payload()
    artifacts = payload.pop("sector_artifacts")
    for artifact in artifacts:
        artifact_path = tmp_path / artifact["sector"] / "autonomous_sector_run.json"
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    payload["sector_results"] = [
        {
            "sector": artifact["sector"],
            "artifact_path": f"{artifact['sector']}/autonomous_sector_run.json",
            "artifact_sha256": hashlib.sha256(
                (tmp_path / artifact["sector"] / "autonomous_sector_run.json").read_bytes()
            ).hexdigest(),
            "run_id": artifact["run_id"],
            "pipeline_version": artifact["pipeline_version"],
            "status": artifact["status"],
            "execution_status": artifact["execution_status"],
            "decision_status": artifact["decision_status"],
            "final_verdict": artifact["final_verdict"],
            "selected_ticker": artifact["selected_ticker"],
        }
        for artifact in artifacts
    ]
    payload["fixed_cohort"].update(
        {
            "schema_version": "all_sector_v2_fixed_cohort_v1",
            "sector_artifacts": [
                {
                    "sector": row["sector"],
                    "run_id": row["run_id"],
                    "artifact_path": row["artifact_path"],
                    "artifact_sha256": row["artifact_sha256"],
                }
                for row in payload["sector_results"]
            ],
            "trusted_manifest_policy": (
                "CALLER_OR_CONFIG_PINNED_NOT_BENCHMARK_SELF_DECLARED"
            ),
        }
    )
    benchmark_path = tmp_path / "benchmark_summary.json"
    evidence_path = tmp_path / "acceptance_evidence.json"
    benchmark_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    evidence_path.write_text(
        json.dumps({"acceptance_evidence": evidence}, sort_keys=True),
        encoding="utf-8",
    )
    before = {
        benchmark_path: benchmark_path.read_bytes(),
        evidence_path: evidence_path.read_bytes(),
    }

    report = run_all_sector_v2_acceptance(
        benchmark_path,
        evidence_path=evidence_path,
        thresholds=THRESHOLDS,
    )

    assert report["status"] == PASS
    assert report["source_path"] == str(benchmark_path)
    assert {path: path.read_bytes() for path in before} == before


def test_loader_rejects_declared_artifact_digest_mismatch(tmp_path: Path) -> None:
    payload, evidence = _passing_payload()
    artifacts = payload.pop("sector_artifacts")
    rows = []
    for index, artifact in enumerate(artifacts):
        artifact_path = tmp_path / artifact["sector"] / "autonomous_sector_run.json"
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        rows.append(
            {
                "sector": artifact["sector"],
                "artifact_path": f"{artifact['sector']}/autonomous_sector_run.json",
                "artifact_sha256": (
                    "0" * 64
                    if index == 0
                    else hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                ),
                "run_id": artifact["run_id"],
                "pipeline_version": artifact["pipeline_version"],
                "status": artifact["status"],
                "execution_status": artifact["execution_status"],
                "decision_status": artifact["decision_status"],
                "final_verdict": artifact["final_verdict"],
                "selected_ticker": artifact["selected_ticker"],
            }
        )
    payload["sector_results"] = rows
    payload["fixed_cohort"].update(
        {
            "schema_version": "all_sector_v2_fixed_cohort_v1",
            "sector_artifacts": [
                {
                    "sector": row["sector"],
                    "run_id": row["run_id"],
                    "artifact_path": row["artifact_path"],
                    "artifact_sha256": row["artifact_sha256"],
                }
                for row in rows
            ],
            "trusted_manifest_policy": (
                "CALLER_OR_CONFIG_PINNED_NOT_BENCHMARK_SELF_DECLARED"
            ),
        }
    )
    benchmark_path = tmp_path / "benchmark_summary.json"
    evidence_path = tmp_path / "acceptance_evidence.json"
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")
    evidence_path.write_text(json.dumps({"acceptance_evidence": evidence}), encoding="utf-8")

    report = run_all_sector_v2_acceptance(
        benchmark_path,
        evidence_path=evidence_path,
        thresholds=THRESHOLDS,
    )

    assert report["status"] == FAIL
    assert _checks(report)["input_integrity"]["status"] == FAIL


def test_persisted_inline_sector_artifacts_are_never_accepted(tmp_path: Path) -> None:
    payload, evidence = _passing_payload()
    benchmark_path = tmp_path / "benchmark_summary.json"
    evidence_path = tmp_path / "acceptance_evidence.json"
    benchmark_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    evidence_path.write_text(
        json.dumps({"acceptance_evidence": evidence}, sort_keys=True),
        encoding="utf-8",
    )

    report = run_all_sector_v2_acceptance(
        benchmark_path,
        evidence_path=evidence_path,
        thresholds=THRESHOLDS,
    )
    integrity = _checks(report)["input_integrity"]

    assert report["status"] == FAIL
    assert integrity["status"] == FAIL
    assert any(
        "INLINE_ARTIFACT_FORBIDDEN" in item
        for item in integrity["details"]["error_examples"]
    )


def test_direct_artifact_injection_cannot_claim_loader_verification(tmp_path: Path) -> None:
    payload, evidence = _passing_payload()
    artifacts = payload.pop("sector_artifacts")

    report = _evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        artifacts=artifacts,
        source_path=tmp_path / "nonexistent-benchmark.json",
        thresholds=THRESHOLDS,
    )
    integrity = _checks(report)["input_integrity"]

    assert report["status"] == FAIL
    assert integrity["status"] == FAIL
    assert "artifacts:DIRECT_UNBOUND_ARTIFACTS_FORBIDDEN" in integrity["details"][
        "error_examples"
    ]


def test_fixed_cohort_digest_index_must_exactly_match_sector_results(
    tmp_path: Path,
) -> None:
    payload, evidence = _passing_payload()
    artifacts = payload.pop("sector_artifacts")
    index_rows = []
    for position, artifact in enumerate(artifacts):
        relative = f"artifacts/{artifact['sector']}.json"
        artifact_path = tmp_path / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        payload["sector_results"][position].update(
            {"artifact_path": relative, "artifact_sha256": digest}
        )
        index_rows.append(
            {
                "sector": artifact["sector"],
                "run_id": artifact["run_id"],
                "artifact_path": relative,
                "artifact_sha256": "f" * 64 if position == 0 else digest,
            }
        )
    payload["fixed_cohort"].update(
        {
            "schema_version": "all_sector_v2_fixed_cohort_v1",
            "sector_artifacts": index_rows,
            "trusted_manifest_policy": (
                "CALLER_OR_CONFIG_PINNED_NOT_BENCHMARK_SELF_DECLARED"
            ),
        }
    )
    benchmark_path = tmp_path / "benchmark_summary.json"
    evidence_path = tmp_path / "acceptance_evidence.json"
    benchmark_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    evidence_path.write_text(
        json.dumps({"acceptance_evidence": evidence}, sort_keys=True),
        encoding="utf-8",
    )

    report = run_all_sector_v2_acceptance(
        benchmark_path,
        evidence_path=evidence_path,
        thresholds=THRESHOLDS,
    )
    binding = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert binding["status"] == FAIL
    assert any(
        "fixed_cohort.sector_artifacts:digest-mismatch" in item
        for item in binding["details"]["error_examples"]
    )


def test_cli_help_documents_bounded_input_and_evidence_paths() -> None:
    help_text = _parser().format_help()

    assert "--input" in help_text
    assert "--evidence" in help_text
    assert "--trusted-manifest" in help_text
    assert "--output" in help_text
    assert "benchmark_summary.json" in help_text


def test_mutating_caller_payload_after_evaluation_does_not_change_report() -> None:
    payload, evidence = _passing_payload()
    original = copy.deepcopy(payload)
    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )

    payload["sector_artifacts"][0]["candidate_dispositions"].clear()

    assert report["status"] == PASS
    assert original != payload
    assert _checks(report)["security_slot_reconciliation"]["actual"]["unique_security_slots"] == 5


def test_canonical_identifier_is_recomputed_instead_of_trusting_supplied_text() -> None:
    payload, evidence = _passing_payload()
    evidence["sparse_history"][0]["canonical_id"] = "sparse_history:energy:BBB"

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    binding = _checks(report)["sparse_history_reconciliation"]["actual"][
        "cohort_manifest"
    ]

    assert report["status"] == FAIL
    assert binding["canonical_id_mismatches"] == [
        {
            "row_index": 0,
            "supplied": "sparse_history:energy:BBB",
            "derived": "sparse_history:energy:AAA",
        }
    ]


def test_omitted_top_ten_true_boolean_cannot_override_ranked_artifact_context() -> None:
    payload, evidence = _passing_payload()
    payload["sector_artifacts"][0]["candidate_selection"][
        "deterministic_prompt_tickers"
    ].remove("FFF")

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["omitted_top_ten_prompt_context"]

    assert check["status"] == FAIL
    assert check["actual"]["artifact_proof"]["failed"] >= 1


def test_historical_omitted_rank_may_change_when_current_ranked_context_is_valid() -> None:
    payload, evidence = _passing_payload()
    row = evidence["omitted_top_ten"][0]
    row["rank"] = 10
    row["canonical_id"] = "omitted_top_ten:energy:FFF:10"
    row["included_in_ranked_prompt_context"] = False
    omitted_hash = _manifest_sha256(evidence["omitted_top_ten"])
    payload["fixed_cohort"]["evidence_manifest_sha256"]["omitted_top_ten"] = (
        omitted_hash
    )
    trusted = dict(THRESHOLDS.trusted_evidence_manifest_sha256)
    trusted["omitted_top_ten"] = omitted_hash

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=replace(
            THRESHOLDS,
            trusted_evidence_manifest_sha256=tuple(trusted.items()),
        ),
    )
    check = _checks(report)["omitted_top_ten_prompt_context"]

    assert report["status"] == PASS
    assert check["status"] == PASS
    assert check["actual"]["artifact_proof"]["proved"] == 2
    assert any(
        "current-rank:1:included" in item
        for item in check["actual"]["artifact_proof"]["examples"]
    )


def test_true_control_caller_outcome_fields_are_ignored_and_gate_is_reexecuted() -> None:
    payload, evidence = _passing_payload()
    for row in evidence["going_concern"]["true_positive_controls"]:
        row.update(
            {
                "status": "PASS",
                "reason_code": None,
                "blockable": False,
                "blocked": False,
            }
        )

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["going_concern_regressions"]

    assert report["status"] == PASS
    assert check["status"] == PASS
    assert any(
        "REGP:current-detector-gate-blocked" in item
        for item in check["actual"]["frozen_fixture_execution"]["examples"]
    )


def test_false_fixture_binds_historical_excerpt_and_current_filing_revisions() -> None:
    payload, evidence = _passing_payload()
    filing_revision = "a" * 64
    evidence["going_concern"]["false_positive_fixtures"][0][
        "filing_content_revision"
    ] = filing_revision
    energy = payload["sector_artifacts"][0]
    disposition_gate = energy["candidate_dispositions"][0]["screen_result"][
        "gate_evaluations"
    ][1]
    structural_gate = energy["candidate_selection"]["structural_gate_results"]["AAA"][
        "screen_result"
    ]["gate_evaluations"][1]
    disposition_gate["observed_value"]["content_revision"] = filing_revision
    structural_gate["observed_value"]["content_revision"] = filing_revision

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )

    assert report["status"] == FAIL
    check = _checks(report)["going_concern_regressions"]
    assert check["status"] == FAIL
    assert (
        check["actual"]["semantic_provenance_manifest"][
            "actual_semantic_manifest_sha256"
        ]
        != check["actual"]["semantic_provenance_manifest"][
            "trusted_semantic_manifest_sha256"
        ]
    )


def test_false_fixture_requires_explicit_assertion_list_in_both_gate_copies() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    energy["candidate_dispositions"][0]["screen_result"]["gate_evaluations"][1][
        "observed_value"
    ].pop("assertions")
    energy["candidate_selection"]["structural_gate_results"]["AAA"][
        "screen_result"
    ]["gate_evaluations"][1]["observed_value"].pop("assertions")

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["going_concern_regressions"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "assertions-list-missing" in str(
        check["actual"]["current_artifact_gate_proof"]
    )


def test_false_fixture_rejects_malformed_assertion_attribution_rows() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    malformed = [{"blockable": False}]
    energy["candidate_dispositions"][0]["screen_result"]["gate_evaluations"][1][
        "observed_value"
    ]["assertions"] = malformed
    energy["candidate_selection"]["structural_gate_results"]["AAA"][
        "screen_result"
    ]["gate_evaluations"][1]["observed_value"]["assertions"] = malformed

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["going_concern_regressions"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "assertion-schema-incomplete" in str(
        check["actual"]["current_artifact_gate_proof"]
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [("assertion_mode", "CALLER_INVENTED_MODE"), ("section", "")],
)
def test_false_fixture_rejects_unsupported_assertion_enum_or_blank_section(
    field_name: str,
    invalid_value: str,
) -> None:
    payload, evidence = _passing_payload()
    fixture = evidence["going_concern"]["false_positive_fixtures"][0]
    assertion = {
        "subject": "UNATTRIBUTED",
        "subject_detail": None,
        "assertion_mode": "HYPOTHETICAL",
        "blockable": False,
        "accession": fixture["accession"],
        "form_type": fixture["form_type"],
        "filing_date": fixture["filing_date"],
        "section": "ANNUAL_FILING_OTHER",
        "excerpt": "If liquidity declined, going concern risk could arise.",
        "corroborating_distress": [],
        "issuer_cik": fixture["issuer_cik"],
        "source_url": fixture["source_url"],
        "content_revision": fixture["filing_content_revision"],
    }
    assertion[field_name] = invalid_value
    energy = payload["sector_artifacts"][0]
    for observed in (
        energy["candidate_dispositions"][0]["screen_result"]["gate_evaluations"][1][
            "observed_value"
        ],
        energy["candidate_selection"]["structural_gate_results"]["AAA"][
            "screen_result"
        ]["gate_evaluations"][1]["observed_value"],
    ):
        observed["assertions"] = [copy.deepcopy(assertion)]

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["going_concern_regressions"]

    assert report["status"] == FAIL
    assert "assertion-attribution-invalid" in str(
        check["actual"]["current_artifact_gate_proof"]
    )


def test_false_fixture_rejects_supported_mode_that_contradicts_detector_excerpt() -> None:
    payload, evidence = _passing_payload()
    fixture = evidence["going_concern"]["false_positive_fixtures"][0]
    forged_assertion = {
        "subject": "REGISTRANT",
        "subject_detail": "AAA",
        "assertion_mode": "NEGATED",
        "blockable": False,
        "accession": fixture["accession"],
        "form_type": fixture["form_type"],
        "filing_date": fixture["filing_date"],
        "section": "AUDITOR_REPORT",
        "excerpt": (
            "These conditions raise substantial doubt about our ability to continue "
            "as a going concern."
        ),
        "corroborating_distress": [],
        "issuer_cik": fixture["issuer_cik"],
        "source_url": fixture["source_url"],
        "content_revision": fixture["filing_content_revision"],
    }
    energy = payload["sector_artifacts"][0]
    for observed in (
        energy["candidate_dispositions"][0]["screen_result"]["gate_evaluations"][1][
            "observed_value"
        ],
        energy["candidate_selection"]["structural_gate_results"]["AAA"][
            "screen_result"
        ]["gate_evaluations"][1]["observed_value"],
    ):
        observed["assertions"] = [copy.deepcopy(forged_assertion)]

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["going_concern_regressions"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "detector-replay-mismatch" in str(
        check["actual"]["current_artifact_gate_proof"]
    )


def test_identity_manifest_cannot_certify_caller_forged_filing_provenance() -> None:
    payload, evidence = _passing_payload()
    fixture = evidence["going_concern"]["false_positive_fixtures"][0]
    fixture.update(
        {
            "accession": "forged-accession",
            "form_type": "20-F",
            "filing_date": "2025-12-31",
            "issuer_cik": "9999999999",
            "source_url": "test://forged/full-filing",
            "evidence_ref_id": "sec-filing:AAA:forged-accession",
            "filing_content_revision": "b" * 64,
        }
    )
    energy = payload["sector_artifacts"][0]
    for evaluation in (
        energy["candidate_dispositions"][0]["screen_result"]["gate_evaluations"][1],
        energy["candidate_selection"]["structural_gate_results"]["AAA"][
            "screen_result"
        ]["gate_evaluations"][1],
    ):
        evaluation["evidence_ref_id"] = fixture["evidence_ref_id"]
        evaluation["evidence_url"] = fixture["source_url"]
        evaluation["observed_value"].update(
            {
                "accession": fixture["accession"],
                "form_type": fixture["form_type"],
                "filing_date": fixture["filing_date"],
                "issuer_cik": fixture["issuer_cik"],
                "source_url": fixture["source_url"],
                "content_revision": fixture["filing_content_revision"],
            }
        )

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["going_concern_regressions"]

    assert report["status"] == FAIL
    assert check["actual"]["cohort_manifest"]["actual_manifest_sha256"] == (
        check["actual"]["cohort_manifest"]["trusted_manifest_sha256"]
    )
    assert check["actual"]["semantic_provenance_manifest"][
        "actual_semantic_manifest_sha256"
    ] != check["actual"]["semantic_provenance_manifest"][
        "trusted_semantic_manifest_sha256"
    ]


def test_missing_independent_gc_semantic_manifest_is_not_evaluable() -> None:
    payload, evidence = _passing_payload()
    thresholds = replace(THRESHOLDS, trusted_semantic_manifest_sha256=())

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=thresholds,
    )

    assert report["status"] == NOT_EVALUABLE
    assert _checks(report)["going_concern_regressions"]["status"] == NOT_EVALUABLE


def test_provider_rows_require_unique_id_identity_tokens_and_cost() -> None:
    payload, evidence = _passing_payload()
    parent = payload["sector_artifacts"][0]["provider_usage"][0]
    parent.pop("model")
    parent.pop("reserved_output_tokens")
    parent.pop("cost_estimate_usd")
    payload["sector_artifacts"][0]["provider_usage"].append(dict(parent))

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    errors = _checks(report)["lane_and_cost_reconciliation"]["details"][
        "raw_record_error_examples"
    ]

    assert any("provider_call_id:duplicate" in item for item in errors)
    assert any("model:missing" in item for item in errors)
    assert any("reserved_output_tokens:missing" in item for item in errors)
    assert any("cost:missing" in item for item in errors)


def test_completed_underwriting_cannot_hide_behind_zero_provider_totals() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    child = energy["company_autonomy_runs"][0]
    child["artifact"]["provider_usage"] = []
    child["attempts"][0]["provider_usage"] = []
    lane = payload["rollups"]["lane_usage_totals"]["lanes"]["company_underwriting"]
    for field in (
        "provider_call_attempts",
        "provider_calls_ok",
        "provider_calls_failed",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reserved_output_tokens",
        "cost_microdollars",
    ):
        lane[field] = 0
    payload["rollups"]["lane_usage_totals"]["aggregate"] = {
        field: sum(
            item[field]
            for item in payload["rollups"]["lane_usage_totals"]["lanes"].values()
        )
        for field in lane
    }

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["actionable_underwriting_and_validation"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "selected-company underwriting has no provider execution records" in str(
        check["details"]["violation_examples"]
    )


def test_completed_validation_cannot_hide_behind_zero_provider_totals() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    energy["selection_validation"]["provider_usage"] = []
    lane = payload["rollups"]["lane_usage_totals"]["lanes"][
        "selected_company_validation"
    ]
    for field in (
        "provider_call_attempts",
        "provider_calls_ok",
        "provider_calls_failed",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reserved_output_tokens",
        "cost_microdollars",
    ):
        lane[field] = 0
    payload["rollups"]["lane_usage_totals"]["aggregate"] = {
        field: sum(
            item[field]
            for item in payload["rollups"]["lane_usage_totals"]["lanes"].values()
        )
        for field in lane
    }

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["actionable_underwriting_and_validation"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "validator-bound provider usage" in str(
        check["details"]["violation_examples"]
    )


def test_watchlist_only_underwriting_requires_physical_provider_proof() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    disposition = energy["candidate_dispositions"][0]
    disposition["underwriting_verdict"] = "WATCHLIST_ONLY"
    disposition["underwriting_result"]["verdict"] = "WATCHLIST_ONLY"
    energy["final_verdict"] = "WATCHLIST"
    energy["selection_validation"] = SectorSelectionValidation(
        status="NOT_REQUIRED"
    ).to_dict()
    child = energy["company_autonomy_runs"][0]
    child["artifact"]["provider_usage"] = []
    child["attempts"][0]["provider_usage"] = []
    _zero_fixture_provider_lane(payload, "company_underwriting")

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["actionable_underwriting_and_validation"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert check["actual"]["completed_underwriting"] == 1
    assert "no provider execution records" in str(check["details"])


def test_contradicted_validation_requires_complete_independent_challenge_proof() -> None:
    payload, evidence = _passing_payload()
    energy = payload["sector_artifacts"][0]
    disposition = energy["candidate_dispositions"][0]
    disposition["underwriting_verdict"] = "WATCHLIST_ONLY"
    disposition["underwriting_result"]["verdict"] = "WATCHLIST_ONLY"
    energy["final_verdict"] = "WATCHLIST"
    validation = copy.deepcopy(energy["selection_validation"])
    validation.update(
        {
            "status": "CONTRADICTED",
            "validator_verdict": "AVOID",
            "evidence_ref_ids": [],
            "evidence": [],
            "tool_calls": [],
        }
    )
    validation["terminal_ledger_fingerprint"] = (
        selection_validation_terminal_ledger_fingerprint(validation)
    )
    energy["selection_validation"] = validation

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["actionable_underwriting_and_validation"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert check["actual"]["completed_validation"] == 1
    assert "validation lacks challenge evidence references" in str(check["details"])


def test_zero_value_success_provider_row_cannot_prove_physical_execution() -> None:
    payload, evidence = _passing_payload()
    child = payload["sector_artifacts"][0]["company_autonomy_runs"][0]
    for ledger in (child["artifact"], child["attempts"][0]):
        provider = ledger["provider_usage"][0]
        provider.update(
            {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "cost_estimate_usd": "0",
            }
        )
    _zero_fixture_provider_lane(payload, "company_underwriting")

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["actionable_underwriting_and_validation"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "successful input_tokens must be positive" in str(check["details"])
    assert "successful output_tokens must be positive" in str(check["details"])


def test_scenario_economics_mutation_breaks_canonical_cohort_fingerprint() -> None:
    payload, evidence = _passing_payload()
    scenarios = payload["sector_artifacts"][0]["expected_return_scenarios"]
    aaa = next(row for row in scenarios if row["ticker"] == "AAA")
    aaa["estimated_future_value_per_share"] = 3_000.0

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "source bindings must derive from persisted signal snapshots" in " ".join(
        check["details"]["error_examples"]
    )


def test_nested_foreign_run_id_breaks_underwriting_source_binding() -> None:
    payload, evidence = _passing_payload()
    child = payload["sector_artifacts"][0]["company_autonomy_runs"][0]
    child["artifact"]["request"]["run_id"] = "foreign-child"
    child["attempts"][0]["request"]["run_id"] = "foreign-child"

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "nested request identity mismatch" in " ".join(
        check["details"]["error_examples"]
    )


def test_child_artifact_and_terminal_attempt_must_be_the_same_canonical_ledger() -> None:
    payload, evidence = _passing_payload()
    child = payload["sector_artifacts"][0]["company_autonomy_runs"][0]
    child["attempts"][0]["tool_calls"] = []
    child["attempts"][0]["provider_usage"] = []
    lane = payload["rollups"]["lane_usage_totals"]["lanes"]["company_underwriting"]
    for field in lane:
        lane[field] = 0
    _reconcile_fixture_lane_totals(payload)

    report = evaluate_all_sector_v2_acceptance(
        payload,
        evidence=evidence,
        thresholds=THRESHOLDS,
    )
    check = _checks(report)["artifact_contract_and_binding"]

    assert report["status"] == FAIL
    assert check["status"] == FAIL
    assert "terminal attempt must match child artifact ledger" in " ".join(
        check["details"]["error_examples"]
    )


def test_loader_requires_digest_and_forbids_evidence_artifact_overlay(
    tmp_path: Path,
) -> None:
    payload, evidence = _passing_payload()
    artifacts = payload.pop("sector_artifacts")
    path = tmp_path / "energy.json"
    path.write_text(json.dumps(artifacts[0]), encoding="utf-8")
    payload["sector_results"] = [
        {
            **payload["sector_results"][0],
            "artifact_path": path.name,
        }
    ]
    payload["fixed_cohort"].update(
        {
            "schema_version": "all_sector_v2_fixed_cohort_v1",
            "sector_artifacts": [
                {
                    "sector": "energy",
                    "run_id": artifacts[0]["run_id"],
                    "artifact_path": path.name,
                    "artifact_sha256": "",
                }
            ],
            "trusted_manifest_policy": (
                "CALLER_OR_CONFIG_PINNED_NOT_BENCHMARK_SELF_DECLARED"
            ),
        }
    )
    benchmark = tmp_path / "benchmark_summary.json"
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "acceptance_evidence": evidence,
                "sector_artifacts": [artifacts[1]],
            }
        ),
        encoding="utf-8",
    )

    report = run_all_sector_v2_acceptance(
        benchmark,
        evidence_path=evidence_path,
        thresholds=THRESHOLDS,
    )
    errors = _checks(report)["input_integrity"]["details"]["error_examples"]

    assert report["status"] == FAIL
    assert any("MISSING_SHA256" in item for item in errors)
    assert "evidence:sector_artifacts:FORBIDDEN_ARTIFACT_REPLACEMENT" in errors


def test_trusted_manifest_loader_accepts_only_versioned_sha256_map(tmp_path: Path) -> None:
    path = tmp_path / "trusted.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "all_sector_v2_trusted_acceptance_manifest_v1",
                "evidence_manifest_sha256": {"sparse_history": "a" * 64},
                "semantic_manifest_sha256": {
                    "going_concern_false_provenance": "b" * 64
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_trusted_manifest_hashes(path) == (("sparse_history", "a" * 64),)
    assert load_trusted_semantic_manifest_hashes(path) == (
        ("going_concern_false_provenance", "b" * 64),
    )
