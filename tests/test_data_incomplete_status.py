from __future__ import annotations

from app.autonomous.financial_integrity import FinancialIntegrityGateResult
from app.autonomous.run_contract import EvidenceReference, ToolCallRecord
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorFinalDecision,
)
from app.autonomous.sector_runtime import (
    _VERDICT_ACTIONABILITY_ORDER,
    _apply_llm_verdict_ceiling,
    _selection_audit_for_ticker,
    classify_audit_signal,
    run_sector_autonomous_financial_analysis,
)
from tests.test_autonomous_sector_runtime import (
    FakeProvider,
    _budget,
    _final_turn,
    _packets,
    _planning_turn,
)


def test_classify_missing_evidence_signals_as_evidence_quality():
    assert classify_audit_signal("MISSING_RELEVANT_EXPECTED_RETURN_EVIDENCE") == "EVIDENCE_QUALITY"
    assert classify_audit_signal("MISSING_COMPANY_SPECIFIC_EVIDENCE") == "EVIDENCE_QUALITY"
    assert classify_audit_signal("INSUFFICIENT_COMPANY_SPECIFIC_EVIDENCE") == "EVIDENCE_QUALITY"
    assert classify_audit_signal("INSUFFICIENT_EVIDENCE_PILLAR_COVERAGE") == "EVIDENCE_QUALITY"


def test_classify_business_quality_signals_unchanged():
    assert classify_audit_signal("ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK") == "BUSINESS_QUALITY"
    assert classify_audit_signal("SURPRISE_NEW_SIGNAL") == "BUSINESS_QUALITY"


# --- fixtures: a self-contained mirror of the audit harness helpers. ---


def _packet(
    ticker: str = "AAA",
    *,
    data_quality_status: str = "OK",
) -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="Financially Viable",
        model_fit_status="VALID_GENERIC",
        data_quality_status=data_quality_status,
        current_price=50.0,
        valuation={"valuation_anchor": 100.0, "anchor_method": "dcf"},
    )


def _base_scenario(
    ticker: str = "AAA",
    annualized_return: float = 0.18,
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


def _downside_scenario(
    ticker: str = "AAA",
    annualized_return: float = -0.02,
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


def _passing_tool_evidence() -> tuple[list[ToolCallRecord], list[EvidenceReference]]:
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


# --- status-tier tests. ---


def test_no_filing_data_gap_is_data_incomplete_not_pass_or_blocked():
    # Clearing base return (0.18) + NO_FILING + passing tool evidence: the only
    # obstacle is a fetchable data-availability gap, so the candidate reads as
    # DATA_INCOMPLETE (resolve-then-promote), never PASS and never BLOCKED.
    tool_calls, evidence = _passing_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _packet("AAA", data_quality_status="NO_FILING")},
        scenarios=[_base_scenario("AAA", 0.18), _downside_scenario("AAA", -0.02)],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "DATA_INCOMPLETE"
    assert audit["actionable"] is False
    assert audit["data_incomplete"] is True
    assert audit["data_resolution_needed"] == [
        "NO_FILING",
        "CAPITAL_LOSS_EVIDENCE_DEGRADED",
    ]
    assert audit["confidence_ceiling"] is None


def test_clean_candidate_with_full_evidence_passes():
    # No signals at all: clean packet + full passing evidence + base 0.18 +
    # downside -0.02 => a clean actionable PASS, data_incomplete False.
    tool_calls, evidence = _passing_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _packet("AAA")},
        scenarios=[_base_scenario("AAA", 0.18), _downside_scenario("AAA", -0.02)],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["data_incomplete"] is False


def test_business_quality_cap_does_not_trigger_data_incomplete():
    # An ELEVATED solvency_risk is a (non-fetchable) business-quality cap, not a
    # data gap. With full evidence and a clearing base return the candidate stays
    # an actionable PASS — the business-quality cap does NOT trigger
    # DATA_INCOMPLETE.
    tool_calls, evidence = _passing_tool_evidence()
    packet = _packet("AAA")
    packet.balance_sheet = {"solvency_risk": "ELEVATED"}

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": packet},
        scenarios=[_base_scenario("AAA", 0.18), _downside_scenario("AAA", -0.02)],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "PASS"
    assert audit["data_incomplete"] is False


def test_thin_return_cushion_hurdle_cap_dominates_data_incomplete():
    # A thin base return (0.13, in [0.12, 0.15)) raises THIN_RETURN_CUSHION, a
    # HURDLE-class confidence cap. Even alongside a NO_FILING data gap the hurdle
    # cap dominates: the candidate is WATCHLIST_ONLY, not DATA_INCOMPLETE.
    tool_calls, evidence = _passing_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _packet("AAA", data_quality_status="NO_FILING")},
        scenarios=[_base_scenario("AAA", 0.13), _downside_scenario("AAA", -0.02)],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "WATCHLIST_ONLY"
    assert audit["data_incomplete"] is False


def test_below_hurdle_base_return_hard_blocks_over_data_incomplete(monkeypatch):
    # A base return of 0.05 is below the 0.12 hurdle, raising
    # BASE_RETURN_BELOW_12PCT_HURDLE — a binding HARD blocker that dominates the
    # NO_FILING data gap: the candidate is BLOCKED, not DATA_INCOMPLETE.
    # This hard-block path is now the 'hard' rollback mode (default is 'soft').
    from app.config import get_config

    monkeypatch.setenv("BASE_RETURN_HURDLE_MODE", "hard")
    get_config.cache_clear()
    tool_calls, evidence = _passing_tool_evidence()

    audit = _selection_audit_for_ticker(
        selected_ticker="AAA",
        packets_by_ticker={"AAA": _packet("AAA", data_quality_status="NO_FILING")},
        scenarios=[_base_scenario("AAA", 0.05), _downside_scenario("AAA", -0.02)],
        tool_calls=tool_calls,
        evidence=evidence,
        degraded_states=[],
    )

    assert audit["status"] == "BLOCKED"
    assert audit["data_incomplete"] is False


# --- DATA_INCOMPLETE through the final-decision dispatch + verdict order. ---


_CANONICAL_DATA_INCOMPLETE_AUDIT = {
    "status": "DATA_INCOMPLETE",
    "selected_ticker": "AAA",
    "actionable": False,
    "data_incomplete": True,
    "data_resolution_needed": ["NO_FILING", "CAPITAL_LOSS_EVIDENCE_DEGRADED"],
    "confidence_ceiling": None,
    "hard_blockers": ["NO_FILING"],
    "confidence_caps": [],
}


def _canonical_signal_packets(
    *,
    tickers: list[str],
    as_of_date: str,
):
    from dataclasses import fields

    from app.alpha.schemas import TickerSignalPacket
    from tests.test_events_acceptance import _canonical_packet_for

    seed_packets = _packets()
    allowed = {item.name for item in fields(TickerSignalPacket)}
    packets: dict[str, TickerSignalPacket] = {}
    for ticker in tickers:
        payload = _canonical_packet_for(ticker, as_of_date)
        packet = TickerSignalPacket(
            **{key: value for key, value in payload.items() if key in allowed}
        )
        seed = seed_packets.get(ticker)
        if seed is not None:
            packet.dcf_value = seed.dcf_value
            packet.epv_value = seed.epv_value
            packet.raw_quality_ctx = dict(seed.raw_quality_ctx)
            packet.raw_valuation = dict(seed.raw_valuation)
        packets[ticker] = packet
    return packets


def _run_data_incomplete_dispatch(monkeypatch):
    """Drive a full run_sector flow whose selected candidate audits to a
    canonical DATA_INCOMPLETE state (clearing base return + only a fetchable
    NO_FILING data gap) so the final-decision dispatch branch is exercised."""
    provider = FakeProvider([_planning_turn(), _final_turn()])
    monkeypatch.setattr("app.autonomous.sector_runtime.get_alpha_llm_provider", lambda: provider)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.require_financial_integrity_scope",
        lambda scope: FinancialIntegrityGateResult(
            context=scope.context,
            run_as_of_date=scope.run_as_of_date,
            status="PASS",
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.require_unchanged_financial_integrity_scope",
        lambda scope, **_kwargs: FinancialIntegrityGateResult(
            context=scope.context,
            run_as_of_date=scope.run_as_of_date,
            status="PASS",
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.assemble_sector_packets",
        lambda tickers, **_kwargs: _canonical_signal_packets(
            tickers=list(tickers),
            as_of_date="2026-04-26",
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.build_canonical_v1_financial_context",
        lambda *, tickers, as_of_date, **_kwargs: type(
            "FinancialContext",
            (),
            {
                "packets": _canonical_signal_packets(
                    tickers=list(tickers),
                    as_of_date=as_of_date,
                ),
                "current_prices": {str(ticker).upper(): 50.0 for ticker in tickers},
                "issuer_contexts": {str(ticker).upper(): {} for ticker in tickers},
            },
        )(),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.dispatch_alpha_tool",
        lambda name, tool_input, ctx: {
            "status": "ok",
            "ticker": ctx.ticker,
            "summary": f"{name} evidence supports {ctx.ticker}.",
        },
    )

    def _fake_audit(*, selected_ticker, **_kwargs):
        return dict(
            _CANONICAL_DATA_INCOMPLETE_AUDIT,
            selected_ticker=str(selected_ticker).upper(),
        )

    monkeypatch.setattr("app.autonomous.sector_runtime._selection_audit_for_ticker", _fake_audit)

    return run_sector_autonomous_financial_analysis(
        sector="specialty_manufacturing",
        tickers=["AAA", "BBB"],
        as_of_date="2026-04-26",
        budget=_budget(),
    )


def test_run_sector_dispatch_emits_data_incomplete_and_preserves_ticker(monkeypatch):
    # A selected candidate whose only obstacle is a fetchable NO_FILING data gap
    # is dispatched as DATA_INCOMPLETE: the audit status, the artifact final
    # verdict, and the SELECTION_AUDIT_DATA_INCOMPLETE state all agree, and the
    # selected ticker is preserved (NOT nulled out like a hard NO_SELECTION).
    artifact = _run_data_incomplete_dispatch(monkeypatch)

    assert artifact.selection_audit["status"] == "DATA_INCOMPLETE"
    assert artifact.final_verdict == "DATA_INCOMPLETE"
    assert artifact.selected_ticker == "AAA"
    assert artifact.confidence is None
    assert artifact.selection_audit["final_verdict_after_audit"] == "DATA_INCOMPLETE"
    assert "SELECTION_AUDIT_DATA_INCOMPLETE" in artifact.degraded_states


def test_run_sector_dispatch_data_incomplete_resolution_list_not_a_blocker(monkeypatch):
    # DATA_INCOMPLETE is a resolve-then-promote state, not a selection blocker:
    # the final decision carries the named fetchable codes in
    # data_resolution_needed and an EMPTY selection_blockers list.
    artifact = _run_data_incomplete_dispatch(monkeypatch)

    assert artifact.final_decision is not None
    assert artifact.final_decision.selection_blockers == []
    assert artifact.final_decision.data_resolution_needed == [
        "NO_FILING",
        "CAPITAL_LOSS_EVIDENCE_DEGRADED",
    ]


def test_verdict_actionability_order_data_incomplete_equals_watchlist():
    # DATA_INCOMPLETE ranks at level 2 — equal to WATCHLIST — never as a hard
    # reject (0/1) and never as an actionable selection (3).
    assert _VERDICT_ACTIONABILITY_ORDER["DATA_INCOMPLETE"] == 2
    assert (
        _VERDICT_ACTIONABILITY_ORDER["DATA_INCOMPLETE"] == _VERDICT_ACTIONABILITY_ORDER["WATCHLIST"]
    )


def test_llm_verdict_ceiling_does_not_raise_data_incomplete_above_level_2():
    # A company-autonomy child verdict of WATCHLIST_ONLY (level 2) cannot raise a
    # DATA_INCOMPLETE audit (level 2) above level 2: the ceiling takes the min of
    # the two, leaves the verdict DATA_INCOMPLETE, and reports bound=False.
    decision = SectorFinalDecision(
        verdict="DATA_INCOMPLETE",
        confidence=None,
        selected_ticker="AAA",
        expected_annualized_return_range="20-25%",
        thesis="AAA is data-incomplete pending the filing fetch.",
        key_risk="The unfetched filing could surface a binding blocker.",
        downside_case="Downside is unassessable until the filing is fetched.",
        no_selection_reason=None,
        selection_blockers=[],
        data_resolution_needed=["NO_FILING", "CAPITAL_LOSS_EVIDENCE_DEGRADED"],
    )

    capped, audit_after, bound = _apply_llm_verdict_ceiling(
        decision=decision,
        selection_audit=dict(_CANONICAL_DATA_INCOMPLETE_AUDIT),
        llm_verdict="WATCHLIST_ONLY",
    )

    assert capped.verdict == "DATA_INCOMPLETE"
    assert _VERDICT_ACTIONABILITY_ORDER[capped.verdict] == 2
    assert capped.selected_ticker == "AAA"
    assert bound is False
    assert audit_after["llm_verdict_ceiling_applied"]["bound"] is False
    assert audit_after["llm_verdict_ceiling_applied"]["final_verdict"] == "DATA_INCOMPLETE"


# --- persist the DATA_INCOMPLETE grade + resolution list onto the watchlist. ---


def _data_incomplete_artifact(
    *,
    selected_ticker: str = "AAA",
    current_price: float = 50.0,
    data_resolution_needed: list[str] | None = None,
) -> "AutonomousSectorFinancialRunArtifact":
    """A COMPLETED sector artifact whose selected candidate finished
    DATA_INCOMPLETE: final_verdict DATA_INCOMPLETE, selected ticker preserved,
    and a final_decision carrying the fetchable resolution codes."""
    from app.autonomous.sector_contract import (
        AutonomousSectorFinancialRunArtifact,
        SectorCompanyFinancialPacket,
        SectorFinalDecision,
    )

    codes = list(data_resolution_needed or ["NO_FILING"])
    packet = SectorCompanyFinancialPacket(
        ticker=selected_ticker,
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="NO_FILING",
        current_price=current_price,
        valuation={"valuation_anchor": 100.0, "anchor_method": "dcf"},
    )
    decision = SectorFinalDecision(
        verdict="DATA_INCOMPLETE",
        confidence=None,
        selected_ticker=selected_ticker,
        expected_annualized_return_range="18-22%",
        thesis=f"{selected_ticker} clears the base-return hurdle but is data-incomplete.",
        key_risk="The unfetched filing could surface a binding blocker.",
        downside_case="Downside is unassessable until the filing is fetched.",
        no_selection_reason=None,
        selection_blockers=[],
        data_resolution_needed=codes,
    )
    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_specialty_manufacturing_data_incomplete",
        sector="specialty_manufacturing",
        market_cap_focus="mid_cap",
        objective="Populate a watchlist from finalist candidates.",
        as_of_date="2026-04-26",
        created_at="2026-04-26T12:00:00Z",
        completed_at="2026-04-26T12:01:00Z",
        status="COMPLETED",
        final_verdict="DATA_INCOMPLETE",
        selected_ticker=selected_ticker,
        confidence=None,
        candidate_selection={"selected_tickers": [selected_ticker]},
        company_packets=[packet],
        relative_ranking=[{"ticker": selected_ticker}],
        final_decision=decision,
    )


def test_watchlist_conviction_grades_contains_data_incomplete():
    from app.watchlist.contract import WATCHLIST_CONVICTION_GRADES

    assert "DATA_INCOMPLETE" in WATCHLIST_CONVICTION_GRADES


def test_candidate_verdict_maps_data_incomplete_for_selected_ticker():
    from app.watchlist.store import _candidate_verdict

    artifact = _data_incomplete_artifact(selected_ticker="AAA")

    assert _candidate_verdict(artifact, "AAA", None) == "DATA_INCOMPLETE"


def test_entry_from_candidate_grade_is_data_incomplete():
    from app.watchlist.store import _entry_from_candidate

    artifact = _data_incomplete_artifact(selected_ticker="AAA", current_price=50.0)
    packet = artifact.company_packets[0]

    entry = _entry_from_candidate(
        artifact,
        ticker="AAA",
        packet=packet,
        ranking={"ticker": "AAA"},
        added_at="2026-04-26T12:00:00+00:00",
        db_path=None,
    )

    assert entry is not None
    assert entry.conviction_grade == "DATA_INCOMPLETE"
    assert entry.buy_price_target == 75.0


def test_entry_from_candidate_deploy_ready_when_price_at_or_below_buy_target():
    from app.watchlist.store import _entry_from_candidate

    # current_price 50.0 <= buy_price_target 75.0 (anchor 100.0 * 0.75): the
    # price STATUS is DEPLOY_READY and is unchanged by the DATA_INCOMPLETE grade.
    artifact = _data_incomplete_artifact(selected_ticker="AAA", current_price=50.0)
    packet = artifact.company_packets[0]

    entry = _entry_from_candidate(
        artifact,
        ticker="AAA",
        packet=packet,
        ranking={"ticker": "AAA"},
        added_at="2026-04-26T12:00:00+00:00",
        db_path=None,
    )

    assert entry is not None
    assert entry.status == "DEPLOY_READY"
    assert entry.conviction_grade == "DATA_INCOMPLETE"


def test_entry_from_candidate_active_when_price_above_buy_target():
    from app.watchlist.store import _entry_from_candidate

    # current_price 90.0 > buy_price_target 75.0: the price STATUS is ACTIVE and
    # is unchanged by the DATA_INCOMPLETE grade.
    artifact = _data_incomplete_artifact(selected_ticker="AAA", current_price=90.0)
    packet = artifact.company_packets[0]

    entry = _entry_from_candidate(
        artifact,
        ticker="AAA",
        packet=packet,
        ranking={"ticker": "AAA"},
        added_at="2026-04-26T12:00:00+00:00",
        db_path=None,
    )

    assert entry is not None
    assert entry.status == "ACTIVE"
    assert entry.conviction_grade == "DATA_INCOMPLETE"


def test_entry_from_candidate_surfaces_resolution_codes_in_open_questions():
    from app.watchlist.store import _entry_from_candidate

    artifact = _data_incomplete_artifact(
        selected_ticker="AAA",
        current_price=50.0,
        data_resolution_needed=["NO_FILING"],
    )
    packet = artifact.company_packets[0]

    entry = _entry_from_candidate(
        artifact,
        ticker="AAA",
        packet=packet,
        ranking={"ticker": "AAA"},
        added_at="2026-04-26T12:00:00+00:00",
        db_path=None,
    )

    assert entry is not None
    assert "NO_FILING" in entry.open_questions


def test_add_or_update_accepts_data_incomplete_grade(monkeypatch, tmp_path):
    from app.config import get_config
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update, get_latest

    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()

    entry = WatchlistEntry(
        ticker="AAA",
        status="DEPLOY_READY",
        conviction_grade="DATA_INCOMPLETE",
        confidence=None,
        conviction_source="sector_final_decision",
        valuation_anchor_method="dcf",
        valuation_anchor_value=100.0,
        buy_price_target=75.0,
        current_price_at_addition=50.0,
        thesis_text="AAA is data-incomplete pending the filing fetch.",
        open_questions=["NO_FILING"],
        source_run_id="autonomous_sector_specialty_manufacturing_data_incomplete",
        source_sector="specialty_manufacturing",
        added_at="2026-04-26T12:00:00+00:00",
    )

    # Must not raise ValueError on the unsupported-grade validation.
    row_id = add_or_update(entry, db_path=db_path)
    latest = get_latest("AAA", db_path=db_path)

    assert row_id == 1
    assert latest is not None
    assert latest.conviction_grade == "DATA_INCOMPLETE"
    assert latest.status == "DEPLOY_READY"
    assert latest.open_questions == ["NO_FILING"]


# --- memo _conviction_grade + CLI surfacing of DATA_INCOMPLETE. ---


def _conviction_grade_artifact(
    *,
    ticker: str,
    audit_status: str,
) -> "AutonomousSectorFinancialRunArtifact":
    """A COMPLETED artifact whose ranking row for ``ticker`` carries the given
    deterministic audit status. The ticker is intentionally NOT the SELECTED
    selected_ticker so _conviction_grade falls through to the status ladder."""
    from app.autonomous.sector_contract import AutonomousSectorFinancialRunArtifact

    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_specialty_manufacturing_conviction_grade",
        sector="specialty_manufacturing",
        market_cap_focus="mid_cap",
        objective="Grade a finalist candidate from a cached artifact.",
        as_of_date="2026-04-26",
        created_at="2026-04-26T12:00:00Z",
        completed_at="2026-04-26T12:01:00Z",
        status="COMPLETED",
        final_verdict="NO_SELECTION",
        selected_ticker=None,
        confidence=None,
        relative_ranking=[{"ticker": ticker, "audit_status": audit_status}],
    )


def test_memo_conviction_grade_data_incomplete_is_not_avoid():
    from app.autonomous.sector_report import _conviction_grade

    artifact = _conviction_grade_artifact(ticker="AAA", audit_status="DATA_INCOMPLETE")

    grade = _conviction_grade(artifact, "AAA")

    assert grade.startswith("DATA_INCOMPLETE")
    assert grade != "AVOID"


def test_memo_conviction_grade_blocked_still_avoid():
    from app.autonomous.sector_report import _conviction_grade

    artifact = _conviction_grade_artifact(ticker="BBB", audit_status="BLOCKED")

    assert _conviction_grade(artifact, "BBB") == "AVOID"


def test_memo_conviction_grade_pass_still_moderate():
    from app.autonomous.sector_report import _conviction_grade

    artifact = _conviction_grade_artifact(ticker="CCC", audit_status="PASS")

    assert _conviction_grade(artifact, "CCC") == "MODERATE"


def test_cli_classify_audit_signals_groups_evidence_and_business_quality():
    from app.cli import _counterfactual_signal_rows

    rows = _counterfactual_signal_rows(["NO_FILING", "ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK"])

    classification = {row["code"]: row["classification"] for row in rows}

    assert classification["NO_FILING"] == "EVIDENCE_QUALITY"
    assert classification["ELEVATED_LIQUIDITY_OR_SOLVENCY_RISK"] == "BUSINESS_QUALITY"


# --- resolved DATA_INCOMPLETE rows re-grade to WATCHLIST_ONLY on re-eval. ---


def _init_temp_db_b3t6(monkeypatch, tmp_path):
    from app.config import get_config
    from app.db import init_db

    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()
    init_db()
    return db_path


def _seed_data_incomplete_entry(
    db_path,
    *,
    ticker: str = "AAA",
    status: str = "DEPLOY_READY",
    open_questions: list[str] | None = None,
):
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update, get_latest

    entry = WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade="DATA_INCOMPLETE",
        conviction_source="sector_final_decision",
        valuation_anchor_method="dcf",
        valuation_anchor_value=100.0,
        buy_price_target=75.0,
        current_price_at_addition=50.0,
        thesis_text=f"{ticker} clears the hurdle but is data-incomplete pending the filing fetch.",
        open_questions=list(open_questions or ["NO_FILING"]),
        source_run_id="autonomous_sector_specialty_manufacturing_data_incomplete",
        source_sector="specialty_manufacturing",
        added_at="2026-05-01T12:00:00+00:00",
    )
    add_or_update(entry, db_path=db_path)
    latest = get_latest(ticker, db_path=db_path)
    assert latest is not None
    return latest


def _insert_filing_b3t6(
    db_path,
    *,
    ticker: str = "AAA",
    accession: str = "0000000000-26-000010",
    form_type: str = "10-Q",
    filing_date: str = "2026-05-03",
) -> None:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1234567890",
                ticker,
                accession,
                form_type,
                filing_date,
                "2026-03-31",
                f"https://example.com/{accession}.htm",
                None,
                "parsed",
                "2026-05-03T12:00:00Z",
                "2026-05-03T12:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _confirming_llm(prompt, **_kwargs):
    return (
        {
            "evaluation": "CONFIRMED",
            "summary": "The new 10-Q confirms the thesis and resolves the missing filing.",
            "evidence_references": ["0000000000-26-000010"],
            "thesis_components": [
                {
                    "component": "filing availability",
                    "verdict": "strengthened",
                    "evidence": "The 10-Q is now on file.",
                }
            ],
        },
        {"cost_estimate_usd": 0.10, "provider": "anthropic", "model": "claude-opus-4-6"},
    )


def _stub_reevaluation_financial_context(
    monkeypatch,
    *,
    ticker: str = "AAA",
) -> None:
    from types import SimpleNamespace

    as_of_date = "2026-07-24"
    packets = _canonical_signal_packets(
        tickers=[ticker],
        as_of_date=as_of_date,
    )
    monkeypatch.setattr(
        "app.watchlist.reevaluation.build_canonical_v1_financial_context",
        lambda **_kwargs: SimpleNamespace(
            as_of_date=as_of_date,
            packets=packets,
        ),
    )


def test_data_incomplete_promotes_to_watchlist_only_when_filing_detected(monkeypatch, tmp_path):
    # A DATA_INCOMPLETE row whose only open question is NO_FILING re-grades to
    # WATCHLIST_ONLY once detect_new_evidence surfaces a filing-section reference.
    from app.watchlist.reevaluation import reevaluate_entry
    from app.watchlist.store import get_latest

    db_path = _init_temp_db_b3t6(monkeypatch, tmp_path)
    entry = _seed_data_incomplete_entry(db_path, open_questions=["NO_FILING"])
    _insert_filing_b3t6(db_path)
    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", _confirming_llm)
    _stub_reevaluation_financial_context(monkeypatch)

    reevaluate_entry(entry, since="2026-05-01", db_path=db_path, include_current_events=False)
    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.conviction_grade == "WATCHLIST_ONLY"


def test_data_incomplete_no_promotion_when_no_new_evidence(monkeypatch, tmp_path):
    # No new evidence at all: the row must NOT be spuriously promoted out of
    # DATA_INCOMPLETE.
    from app.watchlist.reevaluation import reevaluate_entry
    from app.watchlist.store import get_latest

    db_path = _init_temp_db_b3t6(monkeypatch, tmp_path)
    entry = _seed_data_incomplete_entry(db_path, open_questions=["NO_FILING"])

    def fail_llm(prompt):
        raise AssertionError("LLM should not be called when no new evidence exists")

    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", fail_llm)

    reevaluate_entry(entry, since="2026-05-01", db_path=db_path, include_current_events=False)
    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.conviction_grade == "DATA_INCOMPLETE"


def test_data_incomplete_promotion_leaves_price_status_unchanged(monkeypatch, tmp_path):
    # Promoting the GRADE out of DATA_INCOMPLETE must not touch the price STATUS:
    # a DEPLOY_READY row stays DEPLOY_READY.
    from app.watchlist.reevaluation import reevaluate_entry
    from app.watchlist.store import get_latest

    db_path = _init_temp_db_b3t6(monkeypatch, tmp_path)
    entry = _seed_data_incomplete_entry(
        db_path, status="DEPLOY_READY", open_questions=["NO_FILING"]
    )
    _insert_filing_b3t6(db_path)
    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", _confirming_llm)
    _stub_reevaluation_financial_context(monkeypatch)

    reevaluate_entry(entry, since="2026-05-01", db_path=db_path, include_current_events=False)
    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.conviction_grade == "WATCHLIST_ONLY"
    assert latest.status == "DEPLOY_READY"


def test_data_incomplete_no_filing_resolution_never_jumps_to_actionable(monkeypatch, tmp_path):
    # A row whose only resolution code is NO_FILING, with no other business caps,
    # promotes to WATCHLIST_ONLY — never inline-jumps to ACTIONABLE (the full
    # re-grade to ACTIONABLE only happens on the next full sector run).
    from app.watchlist.reevaluation import reevaluate_entry
    from app.watchlist.store import get_latest

    db_path = _init_temp_db_b3t6(monkeypatch, tmp_path)
    entry = _seed_data_incomplete_entry(db_path, open_questions=["NO_FILING"])
    _insert_filing_b3t6(db_path)
    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", _confirming_llm)
    _stub_reevaluation_financial_context(monkeypatch)

    reevaluate_entry(entry, since="2026-05-01", db_path=db_path, include_current_events=False)
    latest = get_latest("AAA", db_path=db_path)

    assert latest is not None
    assert latest.conviction_grade == "WATCHLIST_ONLY"
    assert latest.conviction_grade != "ACTIONABLE"


def test_grade_after_resolution_reevaluation_helper_literals():
    # Direct unit coverage of the pure grade-resolution reevaluation helper.
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.reevaluation import (
        WatchlistEvidenceReference,
        _grade_after_resolution,
    )

    base_entry = WatchlistEntry(
        ticker="AAA",
        status="DEPLOY_READY",
        conviction_grade="DATA_INCOMPLETE",
        open_questions=["NO_FILING"],
        source_run_id="r1",
        added_at="2026-05-01T12:00:00+00:00",
    )
    filing_ref = WatchlistEvidenceReference(
        evidence_id="F1",
        source_type="filing",
        ticker="AAA",
        title="10-Q filed 2026-05-03",
        date="2026-05-03",
        source_ref="0000000000-26-000010",
        form_type="10-Q",
    )

    # Covered NO_FILING with a filing reference => WATCHLIST_ONLY.
    assert _grade_after_resolution(base_entry, [filing_ref]) == "WATCHLIST_ONLY"
    # No evidence => unchanged (None signals "do not re-grade").
    assert _grade_after_resolution(base_entry, []) is None
    # A non-DATA_INCOMPLETE grade is never touched.
    watchlist_entry = WatchlistEntry(
        ticker="AAA",
        status="DEPLOY_READY",
        conviction_grade="WATCHLIST_ONLY",
        open_questions=["NO_FILING"],
        source_run_id="r1",
        added_at="2026-05-01T12:00:00+00:00",
    )
    assert _grade_after_resolution(watchlist_entry, [filing_ref]) is None
