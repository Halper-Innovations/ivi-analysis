from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app.autonomous import runtime as autonomous_runtime
from app.autonomous import sector_runtime
from app.autonomous.financial_integrity import (
    FinancialIntegrityScope,
    require_financial_integrity_scope,
)
from app.autonomous.sector_candidates import (
    SectorCandidateSelection,
    freeze_v2_execution_bound,
)
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    CandidateDisposition,
    SECTOR_CONTRACT_VERSION_V2,
    SectorCompanyFinancialPacket,
    SectorSelectionValidation,
)
from app.autonomous.run_contract import (
    AutonomousRunArtifact,
    AutonomousRunBudget,
    AutonomousRunRequest,
    ToolCallRecord,
)
from app.cli import _campaign_provider_model, app
from app.watchlist.store import WatchlistPopulationResult
from tests.financial_integrity_helpers import canonicalize_financial_packet


runner = CliRunner()


@pytest.fixture(autouse=True)
def _authorize_synthetic_sector_artifacts(monkeypatch):
    monkeypatch.setattr(
        "app.autonomous.artifact_financial_audit.run_id_is_decision_eligible",
        lambda _run_id: True,
    )
    monkeypatch.setattr(
        "app.autonomous.output_store.bind_authorized_valuation_rows",
        lambda **_kwargs: 0,
    )


def test_campaign_provider_model_binds_exact_deepseek_model():
    config = SimpleNamespace(
        anthropic_model="claude-sonnet-test",
        openai_model="gpt-test",
        deepseek_model="deepseek-v4-pro",
    )

    assert _campaign_provider_model(config, "deepseek") == "deepseek-v4-pro"


def _stub_frozen_v2_candidate_selection(**kwargs) -> SectorCandidateSelection:
    membership = [str(ticker).strip().upper() for ticker in kwargs["explicit_tickers"]]
    return freeze_v2_execution_bound(
        SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=membership,
            requested_tickers=membership,
            loaded_tickers=membership,
            source="accepted_census_test_double",
        ),
        kwargs["max_candidates"],
    )


def _init_temp_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def _artifact(
    *,
    verdict: str = "ACTIONABLE",
    selected_ticker: str | None = "AAA",
    confidence: str | None = "MODERATE",
    degraded_states: list[str] | None = None,
    no_winner_reason: str | None = None,
    budget: AutonomousRunBudget | None = None,
    status: str = "COMPLETED",
    tool_calls: list[ToolCallRecord] | None = None,
) -> AutonomousRunArtifact:
    packet = canonicalize_financial_packet(
        {
            "ticker": "AAA",
            "current_price": 50.0,
            "current_price_source": "fixture_quote",
            "current_price_source_url": "https://example.test/quotes/AAA",
            "financial_integrity_status": "PASS",
            "financial_integrity_violations": [],
            "metric_traces": {},
        },
        as_of_date="2026-04-25",
        shares_mm=10.0,
    )
    scope = FinancialIntegrityScope(
        context="autonomous_pre_analyst_context",
        run_as_of_date="2026-04-25",
        packets=(packet,),
    )
    gate = require_financial_integrity_scope(scope)
    request = AutonomousRunRequest(
        run_id="autonomous_AAA_20260425_test",
        objective="Decide if AAA is actionable.",
        as_of_date="2026-04-25",
        created_at="2026-04-25T20:00:00Z",
        candidate_scope={
            "mode": "single_candidate",
            "tickers": ["AAA"],
            "financial_integrity": gate.to_dict(),
            "financial_integrity_binding": autonomous_runtime._financial_integrity_run_binding(
                scope,
                scope_fingerprint=gate.scope_fingerprint,
            ),
        },
        allowed_tools=["fetch_kpi_trends"],
        budget=budget
        or AutonomousRunBudget(
            max_tool_calls=8,
            max_turns=4,
            max_cost_usd=1.25,
            timebox_seconds=None,
            max_candidates=1,
        ),
    )
    return AutonomousRunArtifact(
        request=request,
        status=status,
        started_at="2026-04-25T20:00:00Z",
        completed_at="2026-04-25T20:01:00Z",
        final_verdict=verdict,
        selected_ticker=selected_ticker,
        confidence=confidence,
        tool_calls=tool_calls
        if tool_calls is not None
        else [
            ToolCallRecord(
                call_id="TC1",
                tool_name="fetch_kpi_trends",
                tool_input={},
                rationale="Validate the investment memo with a deterministic tool.",
                status="OK",
                evidence_ref_ids=["E1"],
            )
        ],
        degraded_states=degraded_states or [],
        no_winner_reason=no_winner_reason,
    )


def _sector_artifact(
    *,
    verdict: str = "SELECTED",
    selected_ticker: str | None = "AAA",
    confidence: str | None = "MODERATE",
    degraded_states: list[str] | None = None,
    no_selection_reason: str | None = None,
    status: str = "COMPLETED",
    tool_calls: list[ToolCallRecord] | None = None,
) -> AutonomousSectorFinancialRunArtifact:
    packet = canonicalize_financial_packet(
        SectorCompanyFinancialPacket(
            ticker="AAA",
            financial_status="READY",
            model_fit_status="SUPPORTED",
            data_quality_status="COMPLETE",
            current_price=50.0,
            current_price_source="fixture_quote",
            current_price_source_url="https://example.test/quotes/AAA",
            financial_integrity_status="PASS",
            financial_integrity_violations=[],
            metric_traces={},
        ),
        as_of_date="2026-04-26",
        shares_mm=10.0,
    )
    scope = FinancialIntegrityScope(
        context="autonomous_sector_pre_provider",
        run_as_of_date="2026-04-26",
        packets=(packet,),
    )
    gate = require_financial_integrity_scope(scope)
    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_test_20260426_test",
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        objective="Pick the strongest financially underwritten candidate.",
        as_of_date="2026-04-26",
        created_at="2026-04-26T20:00:00Z",
        completed_at="2026-04-26T20:01:00Z",
        status=status,
        final_verdict=verdict,
        selected_ticker=selected_ticker,
        confidence=confidence,
        candidate_selection={
            "financial_integrity": gate.to_dict(),
            "financial_integrity_binding": sector_runtime._financial_integrity_run_binding(
                scope,
                scope_fingerprint=gate.scope_fingerprint,
            ),
        },
        company_packets=[packet],
        tool_calls=tool_calls
        if tool_calls is not None
        else [
            ToolCallRecord(
                call_id="TC1",
                tool_name="rank_expected_return_cases",
                tool_input={"tickers": ["AAA"]},
                rationale="Rank finalist expected-return cases.",
                status="OK",
                evidence_ref_ids=["E1"],
            )
        ],
        no_selection_reason=no_selection_reason,
        degraded_states=degraded_states or [],
    )


def _v2_incomplete_sector_artifact() -> AutonomousSectorFinancialRunArtifact:
    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_test_20260426_v2_incomplete",
        sector="hospitality_gaming",
        market_cap_focus="large_and_mega",
        objective="Preserve the research queue without publishing an investment call.",
        as_of_date="2026-04-26",
        created_at="2026-04-26T20:00:00Z",
        completed_at="2026-04-26T20:01:00Z",
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
                last_completed_stage="SCREENING",
            )
        ],
        selection_validation=SectorSelectionValidation(status="NOT_REQUIRED"),
        tool_calls=[
            ToolCallRecord(
                call_id="TC1",
                tool_name="rank_expected_return_cases",
                tool_input={"tickers": ["AAA"]},
                rationale="Complete the deterministic screen.",
                status="OK",
                evidence_ref_ids=["E1"],
            )
        ],
        degraded_states=["V2_DECISION_INCOMPLETE"],
        contract_version=SECTOR_CONTRACT_VERSION_V2,
    )


def test_autonomous_run_cli_invokes_runtime_and_prints_artifact_path(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return _artifact()

    monkeypatch.setattr(
        "app.autonomous.runtime.run_single_candidate_autonomous_analysis", fake_runtime
    )

    result = runner.invoke(
        app, ["autonomous-run", "AAA", "--objective", "Decide if AAA is actionable."]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["ticker"] == "AAA"
    assert captured["objective"] == "Decide if AAA is actionable."
    assert payload["final_verdict"] == "ACTIONABLE"
    assert payload["selected_ticker"] == "AAA"
    assert payload["artifact_path"].endswith("autonomous_run.json")
    assert payload["report_path"].endswith("autonomous_research_report.md")
    report_text = Path(payload["report_path"]).read_text(encoding="utf-8")
    assert "# AAA - Autonomous Research Report" in report_text
    assert "## Research Questions" in report_text


def test_autonomous_run_cli_passes_budget_flags(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return _artifact(budget=kwargs["budget"])

    monkeypatch.setattr(
        "app.autonomous.runtime.run_single_candidate_autonomous_analysis", fake_runtime
    )

    result = runner.invoke(
        app,
        [
            "autonomous-run",
            "AAA",
            "--as-of",
            "2026-04-25",
            "--max-tool-calls",
            "3",
            "--max-turns",
            "2",
            "--max-cost-usd",
            "0.5",
            "--years",
            "7",
            "--quarters",
            "3",
            "--skip-analysis-refresh",
        ],
    )

    assert result.exit_code == 0, result.output
    budget = captured["budget"]
    assert captured["as_of_date"] == "2026-04-25"
    assert budget.max_tool_calls == 3
    assert budget.max_turns == 2
    assert budget.max_cost_usd == 0.5
    assert captured["analysis_years"] == 7
    assert captured["analysis_quarters"] == 3
    assert captured["skip_analysis_refresh"] is True


def test_autonomous_run_cli_provider_unavailable_does_not_write_success_artifact(
    monkeypatch, tmp_path
):
    _init_temp_data_dir(monkeypatch, tmp_path)

    def fake_runtime(**kwargs):
        return _artifact(
            verdict="NO_WINNER",
            selected_ticker=None,
            confidence=None,
            degraded_states=["LLM_PROVIDER_UNAVAILABLE"],
            no_winner_reason="LLM provider unavailable.",
            budget=kwargs["budget"],
            status="FAILED",
            tool_calls=[],
        )

    monkeypatch.setattr(
        "app.autonomous.runtime.run_single_candidate_autonomous_analysis", fake_runtime
    )

    result = runner.invoke(app, ["autonomous-run", "AAA"])

    assert result.exit_code != 0
    assert "Refusing to persist autonomous research artifact" in str(result.exception)


def test_autonomous_sector_run_cli_invokes_runtime_and_prints_artifact_path(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return _sector_artifact()

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis", fake_runtime
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--tickers",
            "AAA,BBB",
            "--objective",
            "Pick the best financially underwritten candidate.",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["sector"] == "specialty_manufacturing"
    assert captured["tickers"] == ["AAA", "BBB"]
    assert captured["objective"] == "Pick the best financially underwritten candidate."
    assert captured["market_cap_focus"] == "small_cap"
    assert captured["budget"].max_candidates is None
    assert payload["final_verdict"] == "SELECTED"
    assert payload["selected_ticker"] == "AAA"
    assert payload["scan_family"] == "normal"
    assert payload["artifact_path"].endswith("autonomous_sector_run.json")
    assert payload["report_path"].endswith("autonomous_sector_report.md")
    artifact_payload = json.loads(Path(payload["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact_payload["scan_family"] == "normal"


def test_autonomous_sector_run_cli_refuses_coverage_after_failed_postwrite_authorization(
    monkeypatch,
    tmp_path,
):
    _init_temp_data_dir(monkeypatch, tmp_path)
    coverage_writes: list[str] = []

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        lambda **_kwargs: _sector_artifact(),
    )
    monkeypatch.setattr(
        "app.autonomous.artifact_financial_audit.run_id_is_decision_eligible",
        lambda _run_id: False,
    )

    def _unexpected_coverage_write(*_args, **_kwargs):
        coverage_writes.append("attempted")
        raise AssertionError("coverage writes must remain unreachable")

    monkeypatch.setattr(
        "app.autonomous.sweep_delta.record_loaded_set",
        _unexpected_coverage_write,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--tickers",
            "AAA,BBB",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "refusing coverage writes" in str(result.exception)
    assert coverage_writes == []


def test_autonomous_sector_run_cli_failed_runtime_does_not_call_success_persister(
    monkeypatch, tmp_path
):
    _init_temp_data_dir(monkeypatch, tmp_path)

    def fake_runtime(**kwargs):
        return _sector_artifact(
            verdict="NO_SELECTION",
            selected_ticker=None,
            confidence=None,
            status="FAILED",
            no_selection_reason="Provider failed before a sector decision.",
            degraded_states=["LLM_PROVIDER_TURN_FAILED"],
            tool_calls=[],
        )

    def fail_persist(artifact):
        raise AssertionError("failed sector artifacts must not use the success persister")

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis", fake_runtime
    )
    monkeypatch.setattr("app.autonomous.output_store.persist_autonomous_sector_run", fail_persist)

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--tickers",
            "AAA,BBB",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "FAILED"
    assert payload["final_verdict"] == "NO_SELECTION"
    assert payload["artifact_path"] is None
    assert payload["report_path"] is None
    assert payload["watchlist_population"] == {
        "status": "skipped_failed_artifact",
        "added_or_updated": 0,
        "skipped": 0,
        "entry_ids": [],
        "skipped_reasons": {"run": "STATUS_FAILED"},
    }


def test_autonomous_sector_run_cli_persists_incomplete_v2_as_diagnostic_and_populates_research_queue(
    monkeypatch, tmp_path
):
    cfg = _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_runtime(**kwargs):
        captured["pipeline_version"] = kwargs["pipeline_version"]
        return _v2_incomplete_sector_artifact()

    def fake_populate(artifact):
        captured["watchlist_run_id"] = artifact.run_id
        return WatchlistPopulationResult(
            added_or_updated=1,
            skipped=0,
            entry_ids=[7],
            skipped_reasons={},
        )

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        fake_runtime,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        _stub_frozen_v2_candidate_selection,
    )
    monkeypatch.setattr("app.watchlist.store.populate_from_sector_artifact", fake_populate)

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "hospitality_gaming",
            "--tickers",
            "AAA",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured == {
        "pipeline_version": "v2",
        "watchlist_run_id": "autonomous_sector_test_20260426_v2_incomplete",
    }
    assert payload["decision_status"] == "INCOMPLETE"
    assert payload["final_verdict"] is None
    assert payload["artifact_class"] == "diagnostic"
    assert payload["artifact_path"].endswith("autonomous_sector_diagnostic.json")
    assert payload["report_path"] is None
    assert Path(payload["artifact_path"]).exists()
    assert payload["watchlist_population"]["status"] == "populated"

    import sqlite3

    with sqlite3.connect(cfg.db_path) as conn:
        row = conn.execute(
            """
            SELECT pipeline_version, candidate_disposition, coverage_complete
            FROM sector_run_loaded_sets
            WHERE run_id = ? AND ticker = ?
            """,
            ("autonomous_sector_test_20260426_v2_incomplete", "AAA"),
        ).fetchone()
    assert row == ("v2", "READY_FOR_UNDERWRITING", 0)


def test_autonomous_sector_run_cli_persists_v2_exception_attempt(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)

    def fail_runtime(**kwargs):
        raise RuntimeError("provider transport interrupted before artifact")

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        fail_runtime,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        _stub_frozen_v2_candidate_selection,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--tickers",
            "AAA,BBB",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["execution_status"] == "FAILED"
    assert payload["decision_status"] == "INCOMPLETE"
    assert payload["final_verdict"] is None
    assert payload["artifact_class"] == "diagnostic"
    assert payload["artifact_path"].endswith("autonomous_sector_attempt_diagnostic.json")
    diagnostic = json.loads(Path(payload["artifact_path"]).read_text(encoding="utf-8"))
    assert diagnostic["admitted_tickers"] == ["AAA", "BBB"]
    assert diagnostic["error"] == {
        "type": "RuntimeError",
        "message": "provider transport interrupted before artifact",
    }


def test_paid_v2_single_sector_cli_fails_closed_before_candidate_resolution(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_MODEL", "gpt-5.5")
    monkeypatch.setenv("VOE_OPENAI_API_KEY", "test-key-must-not-be-used")
    from app.config import get_config

    get_config.cache_clear()
    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("candidate resolution must not run on the blocked paid path")
        ),
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--pipeline-version",
            "v2",
        ],
    )

    assert result.exit_code == 2
    assert "Paid v2 single-sector execution is disabled" in result.output
    assert "autonomous-sector-benchmark" in result.output


def test_autonomous_sector_benchmark_help_defines_max_candidates_execution_bound():
    result = runner.invoke(
        app,
        ["autonomous-sector-benchmark", "--help"],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 0, result.output
    assert "Optional per-sector v2 execution bound" in result.output
    assert "complete membership is preserved" in result.output
    assert "overflow receives DEFERRED_BY_BOUND" in result.output


def test_autonomous_sector_run_cli_persists_candidate_resolution_failure(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cap resolver unavailable")),
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    diagnostic = json.loads(Path(payload["artifact_path"]).read_text(encoding="utf-8"))
    assert diagnostic["last_completed_stage"] == "PIPELINE_CONFIGURATION"
    assert diagnostic["artifact_snapshot"] == {}
    assert diagnostic["error"]["message"] == "cap resolver unavailable"


def test_autonomous_sector_run_cli_retains_artifact_when_enrichment_fails(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    artifact = _v2_incomplete_sector_artifact()
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        lambda **kwargs: artifact,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        _stub_frozen_v2_candidate_selection,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.enrich_sector_artifact_memo_body",
        lambda artifact: (_ for _ in ()).throw(RuntimeError("memo rendering failed")),
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "hospitality_gaming",
            "--tickers",
            "AAA",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    diagnostic = json.loads(Path(payload["artifact_path"]).read_text(encoding="utf-8"))
    assert diagnostic["last_completed_stage"] == "SECTOR_ANALYSIS"
    assert diagnostic["artifact_snapshot"]["run_id"] == artifact.run_id
    assert diagnostic["artifact_snapshot"]["decision_status"] == "INCOMPLETE"


def test_autonomous_sector_run_cli_keyboard_interrupt_persists_then_reraises(monkeypatch, tmp_path):
    cfg = _init_temp_data_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        _stub_frozen_v2_candidate_selection,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--tickers",
            "AAA",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
        ],
    )

    # Click converts a re-raised KeyboardInterrupt to the conventional SIGINT
    # exit status while the diagnostic has already been flushed.
    assert result.exit_code == 130
    diagnostics = list(
        (cfg.runs_dir / "autonomous_sector_diagnostics").glob(
            "*/autonomous_sector_attempt_diagnostic.json"
        )
    )
    assert len(diagnostics) == 1
    diagnostic = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert diagnostic["error"]["type"] == "KeyboardInterrupt"


def test_autonomous_sector_run_cli_passes_budget_and_candidate_flags(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return _sector_artifact()

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis", fake_runtime
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--tickers",
            "AAA,BBB,CCC",
            "--as-of",
            "2026-04-26",
            "--market-cap-focus",
            "micro_cap",
            "--max-tool-calls",
            "5",
            "--max-turns",
            "3",
            "--max-cost-usd",
            "1.5",
            "--max-candidates",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    budget = captured["budget"]
    assert captured["tickers"] == ["AAA", "BBB"]
    assert captured["as_of_date"] == "2026-04-26"
    assert captured["market_cap_focus"] == "micro_cap"
    assert budget.max_tool_calls == 5
    assert budget.max_turns == 3
    assert budget.max_cost_usd == 1.5
    assert budget.max_candidates == 2


def test_v2_single_sector_cli_passes_only_frozen_execution_tickers(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}
    candidate_kwargs: dict = {}

    def fake_candidate_resolution(**kwargs):
        candidate_kwargs.update(kwargs)
        return SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA", "BBB", "CCC"],
            membership_tickers=["AAA", "BBB", "CCC"],
            execution_tickers=["AAA", "BBB"],
            deferred_by_bound_tickers=["CCC"],
            execution_bound=2,
            membership_fingerprint=(
                "7ef16e7a59c5fa66fc95eb3ea8a35ae230a52937dada7db2b0e900201584b5ff"
            ),
            execution_fingerprint=(
                "518d1d0ec11d9c4a85cbc86de03771a3cc2b7c35eb40e6b70c08b0c736552490"
            ),
            execution_bound_frozen=True,
            source="explicit_tickers",
        )

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        fake_candidate_resolution,
    )

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return _v2_incomplete_sector_artifact()

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        fake_runtime,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "hospitality_gaming",
            "--tickers",
            "AAA,BBB,CCC",
            "--as-of",
            "2026-04-26",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
            "--max-candidates",
            "2",
            "--no-watchlist",
        ],
    )

    assert result.exit_code == 0, result.output
    assert candidate_kwargs["require_accepted_census"] is True
    assert captured["tickers"] == ["AAA", "BBB"]
    assert captured["candidate_selection"]["selected_tickers"] == [
        "AAA",
        "BBB",
        "CCC",
    ]
    assert captured["candidate_selection"]["execution_tickers"] == ["AAA", "BBB"]
    assert captured["candidate_selection"]["deferred_by_bound_tickers"] == ["CCC"]


def test_v2_single_sector_cli_rejects_only_unswept_before_candidate_resolution(
    monkeypatch, tmp_path
):
    _init_temp_data_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("candidate resolution must not run")),
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "hospitality_gaming",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
            "--only-unswept",
        ],
    )

    assert result.exit_code == 2
    assert "--only-unswept" in result.output


def test_autonomous_sector_run_cli_default_has_no_cost_ceiling(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return _sector_artifact()

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis",
        fake_runtime,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--tickers",
            "AAA",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["budget"].max_cost_usd is None


def test_autonomous_sector_run_cli_can_source_tickers_from_sector(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_resolver(**kwargs):
        captured["resolver"] = kwargs
        return SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["AAA", "BBB"],
            source="sector_scan_db",
            loaded_tickers=["AAA", "BBB", "CCC"],
            excluded_tickers=["CCC"],
            ranking_basis="consensus_pre_rank",
        )

    def fake_runtime(**kwargs):
        captured["runtime"] = kwargs
        return _sector_artifact()

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers", fake_resolver
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis", fake_runtime
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--market-cap-focus",
            "small_cap",
            "--max-candidates",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["resolver"]["explicit_tickers"] == []
    assert captured["resolver"]["market_cap_focus"] == "small_cap"
    assert captured["resolver"]["max_candidates"] == 2
    assert captured["runtime"]["tickers"] == ["AAA", "BBB"]
    assert captured["runtime"]["candidate_selection"] == {
        "sector": "specialty_manufacturing",
        "market_cap_focus": "small_cap",
        "selected_tickers": ["AAA", "BBB"],
        "source": "sector_scan_db",
        "requested_tickers": [],
        "loaded_tickers": ["AAA", "BBB", "CCC"],
        "excluded_tickers": ["CCC"],
        "warnings": [],
        "ranking_basis": "consensus_pre_rank",
        "cap_classifications": {},
        "structural_gate_results": {},
    }
    assert payload["final_verdict"] == "SELECTED"


def test_autonomous_sector_run_cli_rejects_zero_max_candidates(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--max-candidates",
            "0",
        ],
    )

    # A non-zero exit is the behavior under test (the CLI rejects a non-positive
    # --max-candidates). The exact error text is Typer/Click-version-dependent
    # (older versions print a custom message; newer render a Rich Usage panel),
    # so asserting the message would be flaky across versions.
    assert result.exit_code != 0


def test_autonomous_sector_run_cli_populates_watchlist_by_default(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_runtime(**kwargs):
        return _sector_artifact()

    def fake_populate(artifact):
        captured["run_id"] = artifact.run_id
        return WatchlistPopulationResult(
            added_or_updated=2,
            skipped=1,
            entry_ids=[11, 12],
            skipped_reasons={"CCC": "VERDICT_AVOID"},
        )

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis", fake_runtime
    )
    monkeypatch.setattr("app.watchlist.store.populate_from_sector_artifact", fake_populate)

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--tickers",
            "AAA,BBB,CCC",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["run_id"] == "autonomous_sector_test_20260426_test"
    assert payload["watchlist_population"] == {
        "status": "populated",
        "added_or_updated": 2,
        "skipped": 1,
        "entry_ids": [11, 12],
        "skipped_reasons": {"CCC": "VERDICT_AVOID"},
        "adverse_check": None,
    }


def test_autonomous_sector_run_cli_no_watchlist_bypasses_population(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)

    def fake_runtime(**kwargs):
        return _sector_artifact()

    def fail_populate(artifact):
        raise AssertionError("watchlist population should be skipped")

    monkeypatch.setattr(
        "app.autonomous.sector_runtime.run_sector_autonomous_financial_analysis", fake_runtime
    )
    monkeypatch.setattr("app.watchlist.store.populate_from_sector_artifact", fail_populate)

    result = runner.invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "specialty_manufacturing",
            "--tickers",
            "AAA,BBB",
            "--no-watchlist",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["watchlist_population"] == {
        "status": "skipped",
        "added_or_updated": 0,
        "skipped": 0,
        "entry_ids": [],
        "skipped_reasons": {},
    }


def test_autonomous_sector_benchmark_cli_passes_flags_and_prints_paths(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "autonomous_sector_benchmark_20260501_test",
            "status": "COMPLETED",
            "rollups": {
                "sector_count": 2,
                "completed_sector_count": 2,
                "failed_sector_count": 0,
                "skipped_sector_count": 0,
                "benchmark_execution_status_counts": {"RAN": 2},
                "verdict_counts": {"SELECTED": 1, "NO_SELECTION": 1},
                "selection_audit_status_counts": {"PASS": 1, "BLOCKED": 1},
                "actionable_selection_count": 1,
                "actionable_sectors": ["enterprise_software"],
                "selected_tickers": [{"sector": "enterprise_software", "ticker": "AAA"}],
                "watchlist_count": 0,
                "no_selection_count": 1,
                "provider_failure_counts": {},
                "cache_coverage_status_counts": {"CACHE_READY": 1},
                "cache_limited_sectors": [],
                "top_blocker_counts": {"MISSING_COMPANY_SPECIFIC_EVIDENCE": 1},
                "evidence_gap_counts": {"MISSING_COMPANY_SPECIFIC_EVIDENCE": 1},
                "sectors_needing_follow_up": [
                    {
                        "sector": "diversified_industrials",
                        "reason": "MISSING_COMPANY_SPECIFIC_EVIDENCE",
                    }
                ],
            },
            "provider_preflight": {"enabled": True, "status": "OK", "provider": "openai"},
            "resume_source_run_id": "autonomous_sector_benchmark_prior",
            "resumed_sector_count": 1,
            "reused_sector_count": 0,
            "rerun_sector_count": 1,
            "skipped_sector_count": 0,
            "sector_results": [
                {
                    "sector": "enterprise_software",
                    "status": "COMPLETED",
                    "benchmark_execution_status": "RAN",
                    "final_verdict": "SELECTED",
                    "selected_ticker": "AAA",
                    "selection_audit_status": "PASS",
                    "actionable": True,
                    "artifact_path": "sector_a.json",
                    "report_path": "sector_a.md",
                }
            ],
            "cache_refresh": {
                "run_id": "financial_cache_refresh_20260501_test",
                "status": "COMPLETED",
                "summary_path": "refresh_summary.json",
            },
        }

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_autonomous_sector_benchmark", fake_benchmark
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-benchmark",
            "--sectors",
            "enterprise_software,diversified_industrials",
            "--as-of",
            "2026-05-01",
            "--market-cap-focus",
            "smid_cap",
            "--pipeline-version",
            "v2",
            "--cost-preflight-only",
            "--diagnostic-reprice-model",
            "gpt-5.4-mini",
            "--terminal-cap-search-max-attempts",
            "3",
            "--max-candidates",
            "6",
            "--max-tool-calls",
            "9",
            "--max-turns",
            "4",
            "--max-cost-usd",
            "2.25",
            "--prewarm-cache",
            "--cache-refresh-run-id",
            "financial_cache_refresh_20260501_old",
            "--cache-years",
            "9",
            "--force-cache-refresh",
            "--cache-max-tickers-per-sector",
            "12",
            "--provider-preflight",
            "--continue-on-provider-unavailable",
            "--resume-benchmark-run-id",
            "autonomous_sector_benchmark_prior",
            "--rerun-completed-sectors",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["sectors"] == ["enterprise_software", "diversified_industrials"]
    assert captured["as_of_date"] == "2026-05-01"
    assert captured["market_cap_focus"] == "smid_cap"
    assert captured["pipeline_version"] == "v2"
    assert captured["cost_preflight_only"] is True
    assert captured["diagnostic_reprice_model"] == "gpt-5.4-mini"
    assert captured["terminal_cap_search_max_attempts"] == 3
    assert captured["max_candidates"] == 6
    assert captured["budget"].max_tool_calls == 9
    assert captured["budget"].max_turns == 4
    assert captured["budget"].max_cost_usd == 2.25
    assert captured["budget"].max_candidates == 6
    assert captured["prewarm_cache"] is True
    assert captured["cache_refresh_run_id"] == "financial_cache_refresh_20260501_old"
    assert captured["cache_years"] == 9
    assert captured["force_cache_refresh"] is True
    assert captured["cache_max_tickers_per_sector"] == 12
    assert captured["provider_preflight"] is True
    assert captured["continue_on_provider_unavailable"] is True
    assert captured["resume_benchmark_run_id"] == "autonomous_sector_benchmark_prior"
    assert captured["rerun_completed_sectors"] is True
    assert payload["run_id"] == "autonomous_sector_benchmark_20260501_test"
    assert payload["verdict_counts"] == {"SELECTED": 1, "NO_SELECTION": 1}
    assert payload["provider_preflight"]["status"] == "OK"
    assert payload["resume_source_run_id"] == "autonomous_sector_benchmark_prior"
    assert payload["benchmark_path"].endswith("benchmark_summary.json")
    assert payload["report_path"].endswith("benchmark_report.md")


def test_autonomous_sector_benchmark_cli_wires_readiness_preflight_only(
    monkeypatch,
    tmp_path,
):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "autonomous_sector_benchmark_20260717_ready",
            "status": "READINESS_PREFLIGHT_ONLY",
            "execution_mode": "READINESS_PREFLIGHT_ONLY",
            "diagnostic_only": True,
            "readiness_preflight": {
                "readiness_status": "NEEDS_DATA",
                "counts": {
                    "sector_count": 1,
                    "membership_candidates": 4,
                    "execution_candidates": 3,
                    "deferred_by_bound": 1,
                    "excluded_candidates": 0,
                    "ready": 0,
                    "needs_data": 3,
                    "incomplete": 0,
                },
                "missing_input_counts": {"VALUATION": 3},
                "actual_usage": {
                    "model_calls": 0,
                    "search_calls": 0,
                    "network_calls": 0,
                    "cost_microdollars": 0,
                    "cost_usd": "0.000000",
                },
            },
            "rollups": {},
            "sector_results": [],
        }

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_autonomous_sector_benchmark",
        fake_benchmark,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.persist_autonomous_sector_benchmark",
        lambda _artifact: SimpleNamespace(
            summary_json=tmp_path / "benchmark_summary.json",
            report_md=tmp_path / "benchmark_report.md",
            cost_preflight_json=None,
            readiness_preflight_json=tmp_path / "readiness_preflight.json",
        ),
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-benchmark",
            "--sectors",
            "construction_machinery",
            "--as-of",
            "2026-07-17",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
            "--max-candidates",
            "3",
            "--readiness-preflight-only",
            "--no-provider-preflight",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["readiness_preflight_only"] is True
    assert captured["cost_preflight_only"] is False
    assert captured["prewarm_cache"] is False
    assert captured["provider_preflight"] is False
    assert captured["terminal_cap_search"] is None
    assert captured["terminal_cap_search_max_attempts"] is None
    assert captured["max_candidates"] == 3
    assert payload["readiness_status"] == "NEEDS_DATA"
    assert payload["readiness_counts"]["execution_candidates"] == 3
    assert payload["cost_preflight_path"] is None
    assert payload["readiness_preflight_path"].endswith("readiness_preflight.json")


def test_readiness_preflight_cli_rejects_cost_preflight_combination() -> None:
    result = runner.invoke(
        app,
        [
            "autonomous-sector-benchmark",
            "--sectors",
            "energy",
            "--as-of",
            "2026-07-17",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
            "--readiness-preflight-only",
            "--cost-preflight-only",
        ],
    )

    assert result.exit_code == 2
    assert "--readiness-preflight-only cannot be combined with" in result.output
    assert "--cost-preflight-only" in result.output


def test_autonomous_sector_benchmark_cli_wires_free_data_repair_only(
    monkeypatch,
    tmp_path,
):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "autonomous_sector_benchmark_20260717_repair",
            "status": "FREE_DATA_REPAIR_ONLY",
            "execution_mode": "FREE_DATA_REPAIR_ONLY",
            "diagnostic_only": False,
            "maintenance_only": True,
            "free_data_repair": {
                "status": "COMPLETED",
                "readiness_transition_counts": {"NEEDS_DATA_TO_READY": 1},
                "network": {
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
            },
            "readiness_preflight": {
                "readiness_status": "READY",
                "counts": {
                    "sector_count": 1,
                    "membership_candidates": 4,
                    "execution_candidates": 1,
                    "deferred_by_bound": 3,
                    "excluded_candidates": 0,
                    "ready": 1,
                    "needs_data": 0,
                    "incomplete": 0,
                },
                "missing_input_counts": {},
                "actual_usage": {
                    "model_calls": 0,
                    "search_calls": 0,
                    "network_calls": 0,
                    "cost_microdollars": 0,
                    "cost_usd": "0.000000",
                },
            },
            "rollups": {},
            "sector_results": [],
        }

    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.run_autonomous_sector_benchmark",
        fake_benchmark,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_benchmark.persist_autonomous_sector_benchmark",
        lambda _artifact: SimpleNamespace(
            summary_json=tmp_path / "benchmark_summary.json",
            report_md=tmp_path / "benchmark_report.md",
            cost_preflight_json=None,
            readiness_preflight_json=tmp_path / "readiness_preflight.json",
            free_data_repair_json=tmp_path / "accepted_census_free_data_repair.json",
        ),
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-benchmark",
            "--sectors",
            "construction_machinery",
            "--as-of",
            "2026-07-17",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
            "--max-candidates",
            "1",
            "--free-data-repair-only",
            "--no-provider-preflight",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["free_data_repair_only"] is True
    assert captured["readiness_preflight_only"] is False
    assert captured["cost_preflight_only"] is False
    assert captured["prewarm_cache"] is False
    assert captured["provider_preflight"] is False
    assert captured["terminal_cap_search"] is None
    assert captured["terminal_cap_search_max_attempts"] is None
    assert captured["max_candidates"] == 1
    assert payload["free_data_repair_status"] == "COMPLETED"
    assert payload["free_data_repair_usage"]["cost_usd"] == "0.000000"
    assert payload["free_data_repair_network"]["free_network_calls"] == 3
    assert payload["readiness_status"] == "READY"
    assert payload["cost_preflight_path"] is None
    assert payload["readiness_preflight_path"].endswith("readiness_preflight.json")
    assert payload["free_data_repair_path"].endswith("accepted_census_free_data_repair.json")


def test_free_data_repair_cli_requires_a_bounded_execution_subset() -> None:
    result = runner.invoke(
        app,
        [
            "autonomous-sector-benchmark",
            "--sectors",
            "energy",
            "--as-of",
            "2026-07-17",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
            "--free-data-repair-only",
        ],
    )

    assert result.exit_code == 2
    assert "--free-data-repair-only requires --max-candidates between 1" in result.output
    assert "and 3" in result.output


def test_free_data_repair_cli_rejects_other_preflight_only_modes() -> None:
    result = runner.invoke(
        app,
        [
            "autonomous-sector-benchmark",
            "--sectors",
            "energy",
            "--as-of",
            "2026-07-17",
            "--market-cap-focus",
            "large_and_mega",
            "--pipeline-version",
            "v2",
            "--max-candidates",
            "1",
            "--free-data-repair-only",
            "--readiness-preflight-only",
        ],
    )

    assert result.exit_code == 2
    assert "--free-data-repair-only cannot be combined with" in result.output
    assert "preflight-only mode" in result.output


def test_financial_cache_refresh_cli_passes_scope_and_flags(monkeypatch, tmp_path):
    _init_temp_data_dir(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_refresh(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": "financial_cache_refresh_20260501_test",
            "status": "COMPLETED",
            "as_of_date": "2026-05-01",
            "ticker_count": 2,
            "ticker_status_counts": {"OK": 2},
            "step_status_counts": {
                "facts": {"OK": 2},
                "filings": {"DISABLED": 2},
                "price": {"DISABLED": 2},
            },
            "error_count": 0,
            "warnings": [],
            "summary_path": "data/outputs/runs/financial_cache_refresh/x/refresh_summary.json",
            "manifest_path": "data/outputs/runs/financial_cache_refresh/x/refresh_manifest.json",
            "report_path": "data/outputs/runs/financial_cache_refresh/x/refresh_report.md",
        }

    monkeypatch.setattr(
        "app.ingest.financial_cache_refresh.run_financial_cache_refresh", fake_refresh
    )

    result = runner.invoke(
        app,
        [
            "financial-cache-refresh",
            "--tickers",
            "AAA,BBB",
            "--sectors",
            "enterprise_software,semiconductors",
            "--all-known",
            "--market-cap-focus",
            "smid_cap",
            "--max-tickers",
            "7",
            "--max-tickers-per-sector",
            "4",
            "--sector-selection-mode",
            "autonomous_candidates",
            "--years",
            "10",
            "--as-of",
            "2026-05-01",
            "--weekly",
            "--force",
            "--resume-run-id",
            "financial_cache_refresh_20260501_old",
            "--no-with-prices",
            "--no-with-filings",
            "--fallback-days",
            "9",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["tickers"] == ["AAA", "BBB"]
    assert captured["sectors"] == ["enterprise_software", "semiconductors"]
    assert captured["all_known"] is True
    assert captured["market_cap_focus"] == "smid_cap"
    assert captured["max_tickers"] == 7
    assert captured["max_tickers_per_sector"] == 4
    assert captured["sector_selection_mode"] == "autonomous_candidates"
    assert captured["years"] == 10
    assert captured["as_of_date"] == "2026-05-01"
    assert captured["weekly"] is True
    assert captured["force"] is True
    assert captured["resume_run_id"] == "financial_cache_refresh_20260501_old"
    assert captured["with_prices"] is False
    assert captured["with_filings"] is False
    assert captured["fallback_days"] == 9
    assert payload["run_id"] == "financial_cache_refresh_20260501_test"
    assert payload["summary_path"].endswith("refresh_summary.json")
    assert payload["manifest_path"].endswith("refresh_manifest.json")
    assert payload["report_path"].endswith("refresh_report.md")
