from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _legacy_packet_fixture_bypasses_new_integrity_gate(monkeypatch):
    monkeypatch.setattr(
        "app.watchlist.store.artifact_decision_eligibility", lambda payload: "PASS"
    )

from app.autonomous.competitive_frontier import build_competitive_frontier
from app.autonomous.sector_contract import (
    SECTOR_CONTRACT_VERSION_V2,
    AutonomousSectorFinancialRunArtifact,
    CandidateDisposition,
    GateEvaluation,
    ScreenResult,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorSelectionValidation,
    UnderwritingResult,
    build_v2_canonical_child_source_bindings,
)
from app.autonomous.run_contract import EvidenceReference, ToolCallRecord
from app.config import get_config
from app.db import get_db, init_db
from app.market.price_provider import PriceSnapshot
from app.watchlist.contract import WatchlistEntry, is_price_trigger_eligible
from app.watchlist.reevaluation import _grade_after_resolution
from app.watchlist.store import (
    add_or_update,
    get_latest,
    get_latest_price,
    mark_status,
    populate_from_sector_artifact,
    record_reevaluation_result,
    record_trigger_status_change,
)
from app.watchlist.triggers import check_entry_trigger


def _init(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    init_db()
    monkeypatch.setattr(
        "app.watchlist.store._utc_now_iso", lambda: "2026-07-16T12:00:00+00:00"
    )
    return db_path


def _packet(ticker: str, *, price: float, target: float | None = None):
    valuation = {}
    if target is not None:
        valuation = {
            "anchor_method": "DCF",
            "valuation_anchor": 100.0,
            "buy_price_target": target,
        }
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        market_cap_category="large_cap",
        market_cap_mm=12_000.0,
        market_cap_source="fixture_authoritative",
        current_price=price,
        valuation=valuation,
    )


def _canonical_json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _source_binding(
    packet: SectorCompanyFinancialPacket,
    cohort_tickers: list[str],
    frontier_candidate_tickers: list[str] | None = None,
) -> dict[str, object]:
    ticker = packet.ticker.strip().upper()
    normalized_cohort = [item.strip().upper() for item in cohort_tickers]
    signal_fingerprints = {
        item: _canonical_json_fingerprint(
            {"sector": "energy", "as_of_date": "2026-07-16", "ticker": item}
        )
        for item in normalized_cohort
    }
    return {
        "artifact_type": "v2_canonical_child_source_binding_v1",
        "pipeline_version": "v2",
        "sector": "energy",
        "ticker": ticker,
        "as_of_date": "2026-07-16",
        "signal_packet_fingerprint": signal_fingerprints[ticker],
        "cohort_signal_packet_fingerprints": signal_fingerprints,
        "company_packet_fingerprint": _canonical_json_fingerprint(packet.to_dict()),
        "cohort_fingerprint": _canonical_json_fingerprint(normalized_cohort),
        "cohort_tickers": normalized_cohort,
        "frontier_candidate_tickers": [
            item.strip().upper()
            for item in (frontier_candidate_tickers or cohort_tickers)
        ],
    }


def _failed_screen() -> ScreenResult:
    return ScreenResult(
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
                evidence_ref_id="sec:DDD:8-k:2026-07-16",
                reason_code="SECTOR_GATE_FAILED",
            )
        ],
        reason_codes=["SECTOR_GATE_FAILED"],
    )


def _v2_artifact(
    *,
    run_id: str,
    dispositions: list[CandidateDisposition],
    packets: list[SectorCompanyFinancialPacket],
    final_verdict: str | None = None,
    selected_ticker: str | None = None,
    decision_status: str = "INCOMPLETE",
    validation: SectorSelectionValidation | None = None,
) -> AutonomousSectorFinancialRunArtifact:
    in_scope_tickers = [
        disposition.ticker
        for disposition in dispositions
        if disposition.scope_status == "IN_SCOPE"
    ]
    out_of_scope_tickers = [
        disposition.ticker
        for disposition in dispositions
        if disposition.scope_status == "OUT_OF_SCOPE"
    ]
    competitive_frontier: dict = {}
    frontier_scenarios: list[SectorExpectedReturnScenario] = []
    eligible = [
        disposition
        for disposition in dispositions
        if disposition.scope_status == "IN_SCOPE"
        and disposition.screen_status == "PASS"
        and disposition.terminal_state
        in {"READY_FOR_UNDERWRITING", "UNDERWRITTEN"}
    ]
    packets_by_ticker = {packet.ticker.upper(): packet for packet in packets}
    frontier_eligible = [
        disposition
        for disposition in eligible
        if disposition.ticker in packets_by_ticker
    ]
    if frontier_eligible:
        reviewed = [
            disposition.ticker
            for disposition in frontier_eligible
            if disposition.review_status == "COMPLETED"
        ]
        for disposition in frontier_eligible:
            if disposition.ticker in reviewed and disposition.frontier_status is None:
                disposition.frontier_status = "REVIEWED"
        frontier_packets = []
        for index, disposition in enumerate(frontier_eligible):
            packet = packets_by_ticker[disposition.ticker]
            packet.score_components = {
                **packet.score_components,
                "deterministic_score": float(len(frontier_eligible) - index),
            }
            frontier_packets.append(packet)
        frontier_scenarios = [
            SectorExpectedReturnScenario(
                scenario_id=f"{disposition.ticker}-base",
                ticker=disposition.ticker,
                scenario_name="base",
                horizon_years=5,
                current_price=10.0,
                estimated_future_value_per_share=20.0,
                annualized_return=0.15 - index / 100,
            )
            for index, disposition in enumerate(frontier_eligible)
        ]
        state = build_competitive_frontier(
            frontier_packets,
            frontier_scenarios,
            reviewed_tickers=reviewed,
        )
        competitive_frontier = state.to_dict()
        competitive_frontier.update(
            {
                "status": state.closure_certificate.status,
                "minimum_reviews_required": min(3, len(frontier_eligible)),
                "successful_review_count": len(reviewed),
                "attempted_tickers": reviewed,
                "failed_review_tickers": [],
            }
        )
        frontier_tickers = [
            str(item["ticker"]).strip().upper()
            for item in competitive_frontier.get("candidates", [])
        ]
        signal_snapshots = {
            packet.ticker.upper(): {
                "sector": "energy",
                "as_of_date": "2026-07-16",
                "ticker": packet.ticker.upper(),
            }
            for packet in packets
        }
        competitive_frontier["signal_packet_snapshots"] = signal_snapshots
        competitive_frontier["source_bindings"] = (
            build_v2_canonical_child_source_bindings(
                sector="energy",
                as_of_date="2026-07-16",
                company_packets=packets,
                scenarios=frontier_scenarios,
                signal_packet_snapshots=signal_snapshots,
                frontier_candidate_tickers=frontier_tickers,
            )
        )
        if validation is not None and selected_ticker in competitive_frontier[
            "source_bindings"
        ]:
            validation.source_binding = competitive_frontier["source_bindings"][
                selected_ticker
            ]
    company_runs: list[dict] = []
    for disposition in dispositions:
        if disposition.terminal_state != "UNDERWRITTEN":
            continue
        binding = competitive_frontier["source_bindings"][disposition.ticker]
        run_id_for_ticker = f"child-{disposition.ticker.lower()}"
        disposition.underwriting_result = UnderwritingResult(
            status="COMPLETED",
            verdict=disposition.underwriting_verdict,
            confidence=disposition.underwriting_confidence,
            evidence_ref_ids=[f"{run_id_for_ticker}:E1"],
            tool_call_ids=["TC1"],
            child_run_id=run_id_for_ticker,
        )
        nested = {
            "request": {
                "run_id": run_id_for_ticker,
                "as_of_date": "2026-07-16",
                "candidate_scope": {
                    "mode": "single_candidate",
                    "tickers": [disposition.ticker],
                    "source_binding": binding,
                    "signal_packet_snapshot": competitive_frontier[
                        "signal_packet_snapshots"
                    ][disposition.ticker],
                },
            }
        }
        company_runs.append(
            {
                "ticker": disposition.ticker,
                "run_id": run_id_for_ticker,
                "source_binding": binding,
                "artifact": nested,
                "attempts": [nested],
            }
        )
    return AutonomousSectorFinancialRunArtifact(
        run_id=run_id,
        sector="energy",
        market_cap_focus="large_and_mega",
        objective="Exercise hybrid watchlist semantics.",
        as_of_date="2026-07-16",
        created_at="2026-07-16T11:00:00Z",
        completed_at="2026-07-16T12:00:00Z",
        status="COMPLETED",
        final_verdict=final_verdict,
        selected_ticker=selected_ticker,
        confidence="HIGH",
        pipeline_version="v2",
        execution_status="COMPLETED",
        decision_status=decision_status,
        admitted_tickers=in_scope_tickers,
        candidate_selection={
            "loaded_tickers": [disposition.ticker for disposition in dispositions],
            "selected_tickers": in_scope_tickers,
            "excluded_tickers": out_of_scope_tickers,
        },
        candidate_dispositions=dispositions,
        selection_validation=validation,
        competitive_frontier=competitive_frontier,
        company_packets=packets,
        expected_return_scenarios=frontier_scenarios,
        company_autonomy_runs=company_runs,
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )


def test_v2_hybrid_population_keeps_research_states_and_literal_outcomes(
    monkeypatch, tmp_path
):
    db_path = _init(monkeypatch, tmp_path)
    artifact = _v2_artifact(
        run_id="autonomous_sector_v2_hybrid",
        dispositions=[
            CandidateDisposition(
                ticker="AAA",
                terminal_state="READY_FOR_UNDERWRITING",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="NOT_STARTED",
                watchlist_eligible=True,
                reason_codes=["UNDERWRITING_PENDING"],
            ),
            CandidateDisposition(
                ticker="BBB",
                terminal_state="NEEDS_DATA",
                scope_status="IN_SCOPE",
                screen_status="INCOMPLETE",
                review_status="NOT_STARTED",
                watchlist_eligible=True,
                reason_codes=["IFRS_FACTS_UNSUPPORTED"],
            ),
            CandidateDisposition(
                ticker="CCC",
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="WATCHLIST_ONLY",
                underwriting_confidence="MODERATE",
                watchlist_eligible=True,
            ),
            CandidateDisposition(
                ticker="DDD",
                terminal_state="SCREENED_OUT",
                scope_status="IN_SCOPE",
                screen_status="FAIL",
                review_status="NOT_REQUIRED",
                watchlist_eligible=False,
                reason_codes=["SECTOR_GATE_FAILED"],
                screen_result=_failed_screen(),
            ),
            CandidateDisposition(
                ticker="EEE",
                terminal_state="OUT_OF_SCOPE",
                scope_status="OUT_OF_SCOPE",
                screen_status="NOT_RUN",
                review_status="NOT_REQUIRED",
                watchlist_eligible=False,
                reason_codes=["NON_COMMON_SECURITY"],
            ),
        ],
        packets=[
            _packet("AAA", price=90.0),
            _packet("BBB", price=60.0),
            _packet("CCC", price=70.0, target=65.0),
            _packet("DDD", price=80.0),
        ],
    )

    result = populate_from_sector_artifact(artifact, db_path=db_path)

    assert result.added_or_updated == 3
    assert result.skipped_reasons == {
        "DDD": "DISPOSITION_SCREENED_OUT",
        "EEE": "DISPOSITION_OUT_OF_SCOPE",
    }
    ready = get_latest("AAA", db_path=db_path)
    needs_data = get_latest("BBB", db_path=db_path)
    underwritten_watch = get_latest("CCC", db_path=db_path)
    assert ready is not None
    assert ready.status == "ACTIVE"
    assert ready.conviction_grade == "WATCHLIST_ONLY"
    assert ready.conviction_source == "sector_screen"
    assert ready.buy_price_target is None
    assert ready.pipeline_version == "v2"
    assert ready.candidate_disposition == "READY_FOR_UNDERWRITING"
    assert ready.decision_basis == "SCREEN"
    assert "UNDERWRITING_PENDING" in ready.open_questions
    assert needs_data is not None
    assert needs_data.status == "UNCERTAIN"
    assert needs_data.conviction_grade == "DATA_INCOMPLETE"
    assert needs_data.decision_basis == "SCREEN"
    assert underwritten_watch is not None
    assert underwritten_watch.status == "ACTIVE"
    assert underwritten_watch.conviction_grade == "WATCHLIST_ONLY"
    assert underwritten_watch.conviction_source == "company_autonomy"
    assert underwritten_watch.decision_basis == "UNDERWRITING"

    with get_db() as conn:
        outcome_rows = conn.execute(
            "SELECT ticker, grade, decision_basis, candidate_disposition "
            "FROM ticker_outcomes ORDER BY ticker"
        ).fetchall()
    assert [tuple(row) for row in outcome_rows] == [
        ("AAA", "WATCHLIST_ONLY", "SCREEN", "READY_FOR_UNDERWRITING"),
        ("BBB", "DATA_INCOMPLETE", "SCREEN", "NEEDS_DATA"),
        ("CCC", "WATCHLIST_ONLY", "UNDERWRITING", "UNDERWRITTEN"),
        ("DDD", "AVOID", "SCREEN", "SCREENED_OUT"),
    ]


def test_v2_ready_without_packet_still_enters_research_queue(monkeypatch, tmp_path):
    db_path = _init(monkeypatch, tmp_path)
    artifact = _v2_artifact(
        run_id="autonomous_sector_v2_missing_packet",
        dispositions=[
            CandidateDisposition(
                ticker="AAA",
                terminal_state="READY_FOR_UNDERWRITING",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="NOT_STARTED",
                watchlist_eligible=True,
                reason_codes=["PACKET_BUILD_INCOMPLETE"],
            )
        ],
        packets=[],
    )

    result = populate_from_sector_artifact(artifact, db_path=db_path)
    entry = get_latest("AAA", db_path=db_path)

    assert result.added_or_updated == 1
    assert result.skipped_reasons == {}
    assert entry is not None
    assert entry.status == "ACTIVE"
    assert entry.buy_price_target is None
    assert entry.current_price_at_addition is None
    assert entry.candidate_disposition == "READY_FOR_UNDERWRITING"


def test_v2_actionable_is_capped_until_matching_validation(monkeypatch, tmp_path):
    db_path = _init(monkeypatch, tmp_path)
    disposition = CandidateDisposition(
        ticker="AAA",
        terminal_state="UNDERWRITTEN",
        scope_status="IN_SCOPE",
        screen_status="PASS",
        review_status="COMPLETED",
        underwriting_verdict="ACTIONABLE",
        underwriting_confidence="HIGH",
        watchlist_eligible=True,
    )
    artifact = _v2_artifact(
        run_id="autonomous_sector_v2_unvalidated",
        dispositions=[disposition],
        packets=[_packet("AAA", price=70.0, target=80.0)],
        validation=SectorSelectionValidation(
            status="NOT_ATTEMPTED", selected_ticker="AAA"
        ),
    )

    populate_from_sector_artifact(artifact, db_path=db_path)
    entry = get_latest("AAA", db_path=db_path)

    assert entry is not None
    assert entry.conviction_grade == "WATCHLIST_ONLY"
    assert entry.status == "ACTIVE"
    assert entry.decision_basis == "UNDERWRITING"
    assert entry.selection_validation_status == "NOT_ATTEMPTED"
    assert is_price_trigger_eligible(entry) is False


def test_v2_underwriting_data_gap_keeps_underwriting_provenance(monkeypatch, tmp_path):
    db_path = _init(monkeypatch, tmp_path)
    artifact = _v2_artifact(
        run_id="autonomous_sector_v2_underwriting_gap",
        dispositions=[
            CandidateDisposition(
                ticker="AAA",
                terminal_state="NEEDS_DATA",
                scope_status="IN_SCOPE",
                screen_status="INCOMPLETE",
                review_status="COMPLETED",
                underwriting_verdict="DATA_INCOMPLETE",
                underwriting_confidence="MODERATE",
                watchlist_eligible=True,
                reason_codes=["UNDERWRITING_DATA_INCOMPLETE"],
            )
        ],
        packets=[_packet("AAA", price=70.0)],
    )

    populate_from_sector_artifact(artifact, db_path=db_path)
    entry = get_latest("AAA", db_path=db_path)

    assert entry is not None
    assert entry.conviction_grade == "DATA_INCOMPLETE"
    assert entry.conviction_source == "company_autonomy"
    assert entry.decision_basis == "UNDERWRITING"
    assert entry.confidence == "MODERATE"
    with get_db() as conn:
        row = conn.execute(
            "SELECT decision_basis FROM ticker_outcomes WHERE ticker = 'AAA'"
        ).fetchone()
    assert tuple(row) == ("UNDERWRITING",)


def test_v2_validated_actionable_can_be_deploy_ready(monkeypatch, tmp_path):
    db_path = _init(monkeypatch, tmp_path)
    packet = _packet("AAA", price=70.0, target=80.0)
    packet.score_components = {"deterministic_score": 1.0}
    artifact = _v2_artifact(
        run_id="autonomous_sector_v2_validated",
        dispositions=[
            CandidateDisposition(
                ticker="AAA",
                terminal_state="UNDERWRITTEN",
                scope_status="IN_SCOPE",
                screen_status="PASS",
                review_status="COMPLETED",
                underwriting_verdict="ACTIONABLE",
                underwriting_confidence="HIGH",
                watchlist_eligible=True,
            )
        ],
        packets=[packet],
        final_verdict="SELECTED",
        selected_ticker="AAA",
        decision_status="COMPLETE",
        validation=SectorSelectionValidation(
            status="VALIDATED",
            selected_ticker="AAA",
            source_binding=_source_binding(packet, ["AAA"]),
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
                    tool_call_id="validator-aaa:VTC1",
                    confidence="HIGH",
                )
            ],
            tool_calls=[
                ToolCallRecord(
                    call_id="validator-aaa:VTC1",
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
    )
    monkeypatch.setattr(
        "app.events.grade_time.grade_time_adverse_check",
        lambda conn, tickers: {"checked": list(tickers)},
    )

    populate_from_sector_artifact(artifact, db_path=db_path)
    entry = get_latest("AAA", db_path=db_path)

    assert entry is not None
    assert entry.conviction_grade == "ACTIONABLE"
    assert entry.status == "DEPLOY_READY"
    assert entry.decision_basis == "VALIDATED_UNDERWRITING"
    assert entry.selection_validation_status == "VALIDATED"
    assert is_price_trigger_eligible(entry) is True


def test_store_rejects_unvalidated_v2_actionable(monkeypatch, tmp_path):
    db_path = _init(monkeypatch, tmp_path)
    entry = WatchlistEntry(
        ticker="AAA",
        status="ACTIVE",
        conviction_grade="ACTIONABLE",
        source_run_id="autonomous_sector_v2_invalid",
        source_sector="energy",
        added_at="2026-07-16T12:00:00Z",
        pipeline_version="v2",
        candidate_disposition="UNDERWRITTEN",
        decision_basis="UNDERWRITING",
        selection_validation_status="NOT_ATTEMPTED",
    )

    with pytest.raises(ValueError, match="v2 ACTIONABLE requires"):
        add_or_update(entry, db_path=db_path)


def test_v2_screen_row_can_be_safety_quarantined_without_becoming_investable(
    monkeypatch, tmp_path
):
    db_path = _init(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="QUARANTINE",
            conviction_grade="WATCHLIST_ONLY",
            conviction_source="sector_screen",
            source_run_id="autonomous_sector_v2_quarantine",
            source_sector="energy",
            added_at="2026-07-16T12:00:00Z",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
            buy_price_target=80.0,
        ),
        db_path=db_path,
    )

    entry = get_latest("AAA", db_path=db_path)
    assert entry is not None
    assert entry.status == "QUARANTINE"
    assert is_price_trigger_eligible(entry) is False


def test_population_revalidates_mutated_v2_artifact(monkeypatch, tmp_path):
    db_path = _init(monkeypatch, tmp_path)
    packet_aaa = _packet("AAA", price=70.0, target=80.0)
    packet_bbb = _packet("BBB", price=90.0)
    artifact = _v2_artifact(
        run_id="autonomous_sector_v2_mutated",
        dispositions=[
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
        packets=[packet_aaa, packet_bbb],
    )
    artifact.decision_status = "COMPLETE"
    artifact.final_verdict = "SELECTED"
    artifact.selected_ticker = "AAA"
    artifact.selection_validation = SectorSelectionValidation(
        status="VALIDATED",
        selected_ticker="AAA",
        source_binding=_source_binding(packet_aaa, ["AAA", "BBB"]),
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
                tool_call_id="validator-aaa:VTC1",
                confidence="HIGH",
            )
        ],
        tool_calls=[
            ToolCallRecord(
                call_id="validator-aaa:VTC1",
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
    )

    with pytest.raises(ValueError, match="closed competitive frontier"):
        populate_from_sector_artifact(artifact, db_path=db_path)

    assert get_latest("AAA", db_path=db_path) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_disposition": None, "decision_basis": None},
        {
            "candidate_disposition": "READY_FOR_UNDERWRITING",
            "decision_basis": "UNDERWRITING",
            "conviction_source": "company_autonomy",
        },
        {
            "candidate_disposition": "UNDERWRITTEN",
            "decision_basis": "SCREEN",
            "conviction_source": "sector_screen",
        },
        {
            "candidate_disposition": "SCREENED_OUT",
            "decision_basis": "SCREEN",
            "conviction_source": "sector_screen",
        },
    ],
)
def test_store_rejects_inconsistent_v2_provenance(
    monkeypatch, tmp_path, overrides
):
    db_path = _init(monkeypatch, tmp_path)
    values = {
        "ticker": "AAA",
        "status": "ACTIVE",
        "conviction_grade": "WATCHLIST_ONLY",
        "conviction_source": "sector_screen",
        "source_run_id": "autonomous_sector_v2_invalid_provenance",
        "source_sector": "energy",
        "added_at": "2026-07-16T12:00:00Z",
        "pipeline_version": "v2",
        "candidate_disposition": "READY_FOR_UNDERWRITING",
        "decision_basis": "SCREEN",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="v2"):
        add_or_update(WatchlistEntry(**values), db_path=db_path)


def test_v2_current_row_supersedes_older_classic_actionable(monkeypatch, tmp_path):
    db_path = _init(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            conviction_source="company_autonomy",
            source_run_id="autonomous_sector_old_v1",
            source_sector="energy",
            added_at="2026-07-15T12:00:00Z",
        ),
        db_path=db_path,
    )
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            conviction_source="sector_screen",
            source_run_id="autonomous_sector_new_v2",
            source_sector="energy",
            added_at="2026-07-16T12:00:00Z",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )

    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.source_run_id == "autonomous_sector_new_v2"
    assert latest.pipeline_version == "v2"
    assert latest.conviction_grade == "WATCHLIST_ONLY"


def test_newer_v1_row_supersedes_older_v2_after_rollout_rollback(
    monkeypatch, tmp_path
):
    db_path = _init(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            conviction_source="sector_screen",
            source_run_id="autonomous_sector_old_v2",
            source_sector="energy",
            added_at="2026-07-15T12:00:00Z",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="DEPLOY_READY",
            conviction_grade="ACTIONABLE",
            conviction_source="sector_final_decision",
            source_run_id="autonomous_sector_new_v1",
            source_sector="energy",
            added_at="2026-07-16T12:00:00Z",
        ),
        db_path=db_path,
    )

    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.source_run_id == "autonomous_sector_new_v1"
    assert latest.pipeline_version is None
    assert latest.conviction_grade == "ACTIONABLE"


def test_v2_screen_row_never_arms_price_or_catalyst_trigger(monkeypatch, tmp_path):
    db_path = _init(monkeypatch, tmp_path)
    entry_id = add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            conviction_source="sector_screen",
            valuation_anchor_method="DCF",
            valuation_anchor_value=100.0,
            buy_price_target=75.0,
            current_price_at_addition=90.0,
            source_run_id="autonomous_sector_v2_screen",
            source_sector="energy",
            added_at="2026-07-16T12:00:00Z",
            pipeline_version="v2",
            candidate_disposition="READY_FOR_UNDERWRITING",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )
    entry = get_latest("AAA", db_path=db_path)
    assert entry is not None
    catalyst_calls = []
    price_calls = []

    def forbidden_price_lookup(ticker):
        price_calls.append(ticker)
        return PriceSnapshot(
            ticker=ticker,
            as_of_date="2026-07-16",
            price=70.0,
            currency="USD",
            source="fixture",
            retrieved_at="2026-07-16T13:00:00+00:00",
            confidence="HIGH",
        )

    result = check_entry_trigger(
        entry,
        db_path=db_path,
        checked_at="2026-07-16T13:00:00+00:00",
        price_lookup=forbidden_price_lookup,
        historical_median_lookup=lambda ticker: 80.0,
        catalyst_lookup=lambda ticker: catalyst_calls.append(ticker) or "INSIDER_BUY",
    )
    latest = get_latest("AAA", db_path=db_path)

    assert entry_id == 1
    assert result.new_status == "ACTIVE"
    assert result.transition == "v2-investment-gate-refused"
    assert result.latest_price is None
    assert result.buy_price_target is None
    assert result.mutated is False
    assert "requires validated underwriting" in str(result.warning)
    assert price_calls == []
    assert catalyst_calls == []
    assert get_latest_price(entry_id, db_path=db_path) is None
    assert latest is not None
    assert latest.status == "ACTIVE"


def test_v2_watchlist_mutators_preserve_disposition_grade_and_investment_gate(
    monkeypatch, tmp_path
):
    db_path = _init(monkeypatch, tmp_path)
    add_or_update(
        WatchlistEntry(
            ticker="AAA",
            status="UNCERTAIN",
            conviction_grade="DATA_INCOMPLETE",
            conviction_source="sector_screen",
            buy_price_target=75.0,
            current_price_at_addition=90.0,
            open_questions=["NO_FILING"],
            source_run_id="autonomous_sector_v2_needs_data",
            source_sector="energy",
            added_at="2026-07-16T12:00:00Z",
            pipeline_version="v2",
            candidate_disposition="NEEDS_DATA",
            decision_basis="SCREEN",
        ),
        db_path=db_path,
    )

    with pytest.raises(ValueError, match="at-target status requires validated underwriting"):
        mark_status(
            "AAA",
            "DEPLOY_READY",
            "invalid manual promotion",
            "manual",
            db_path=db_path,
        )
    with pytest.raises(ValueError, match="at-target status requires validated underwriting"):
        record_trigger_status_change(
            "AAA",
            status="BUY_CONFIRMED",
            reason="invalid trigger promotion",
            db_path=db_path,
        )
    with pytest.raises(ValueError, match="NEEDS_DATA provenance is inconsistent"):
        record_reevaluation_result(
            "AAA",
            status="UNCERTAIN",
            reason="evidence arrived",
            evaluation="CONFIRMED",
            source_run_id="reevaluation_v2",
            conviction_grade="WATCHLIST_ONLY",
            db_path=db_path,
        )

    entry = get_latest("AAA", db_path=db_path)
    assert entry is not None
    assert entry.status == "UNCERTAIN"
    assert entry.conviction_grade == "DATA_INCOMPLETE"
    assert entry.candidate_disposition == "NEEDS_DATA"
    assert entry.decision_basis == "SCREEN"

    # Protective states remain legal without changing investment provenance.
    mark_status("AAA", "QUARANTINE", "protective hold", "test", db_path=db_path)
    protected = get_latest("AAA", db_path=db_path)
    assert protected is not None
    assert protected.status == "QUARANTINE"
    assert protected.conviction_grade == "DATA_INCOMPLETE"
    assert protected.candidate_disposition == "NEEDS_DATA"


def test_v2_reevaluation_never_promotes_needs_data_grade_inline():
    entry = WatchlistEntry(
        ticker="AAA",
        status="UNCERTAIN",
        conviction_grade="DATA_INCOMPLETE",
        conviction_source="sector_screen",
        source_run_id="autonomous_sector_v2_needs_data",
        source_sector="energy",
        added_at="2026-07-16T12:00:00Z",
        open_questions=["NO_FILING"],
        pipeline_version="v2",
        candidate_disposition="NEEDS_DATA",
        decision_basis="SCREEN",
    )

    assert _grade_after_resolution(entry, []) is None
