from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.autonomous.financial_integrity import (
    FinancialIntegrityGateResult,
    InvalidFinancialInputError,
)
from app.rlm.executor import execute_actions
from app.rlm.schemas import PlannerOutput
from app.rlm.state import LoopBudgets, LoopState


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    return cfg


def test_executor_dispatches_actions(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_dir = cfg.sectors_dir / "rlm_exec_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    rankings_path = run_dir / "peer_rankings.json"
    rankings_path.write_text(
        json.dumps(
            {
                "future_whale_rank": ["AAPL", "MSFT"],
                "rankings": [
                    {"ticker": "AAPL", "whale_signature": {"score": 85.0}},
                    {"ticker": "MSFT", "whale_signature": {"score": 70.0}},
                ],
            }
        ),
        encoding="utf-8",
    )

    src_rankings = tmp_path / "peer_rankings_src.json"
    src_rankings.write_text(json.dumps({"future_whale_rank": ["AAPL"]}), encoding="utf-8")
    src_scoreboard = tmp_path / "peer_scoreboard_src.json"
    src_scoreboard.write_text(json.dumps({"rows": [{"ticker": "AAPL"}]}), encoding="utf-8")
    src_report = tmp_path / "peer_report_src.md"
    src_report.write_text("# peer report\n", encoding="utf-8")

    whale_summary_path = tmp_path / "whale_signals_summary_src.json"
    whale_summary_path.write_text(json.dumps({"rows": [{"ticker": "AAPL"}]}), encoding="utf-8")
    decision_json = tmp_path / "decision_pack_src.json"
    decision_md = tmp_path / "decision_pack_src.md"
    decision_json.write_text(json.dumps({"run_id": "rlm_exec_test"}), encoding="utf-8")
    decision_md.write_text("# decision pack\n", encoding="utf-8")

    calls = {
        "close_gaps": 0,
        "deepen": 0,
        "rebuild": 0,
        "synth_sector": 0,
        "synth_ticker": 0,
    }

    def _fake_close_gaps(**kwargs):
        calls["close_gaps"] += 1
        return {"run_id": kwargs["run_id"], "processed_count": len(kwargs.get("tickers") or [])}

    def _fake_deepen(**kwargs):
        calls["deepen"] += 1
        return {"run_id": kwargs["run_id"], "tickers_requested": kwargs["tickers"]}

    def _fake_rebuild(**kwargs):
        calls["rebuild"] += 1
        return {
            "peer_rankings_path": str(src_rankings),
            "peer_scoreboard_path": str(src_scoreboard),
            "peer_report_path": str(src_report),
        }

    def _fake_synth_sector(**kwargs):
        calls["synth_sector"] += 1
        return {"path": str(run_dir / "sector_synthesis.json")}

    def _fake_synth_ticker(**kwargs):
        calls["synth_ticker"] += 1
        return run_dir / f"{kwargs['ticker']}_synth.json"

    monkeypatch.setattr("app.rlm.executor.run_research_gap_closer", _fake_close_gaps)
    monkeypatch.setattr("app.rlm.executor.run_dossier_for_peer_set", _fake_deepen)
    monkeypatch.setattr("app.rlm.executor.build_peer_report_from_run", _fake_rebuild)
    monkeypatch.setattr("app.rlm.executor.run_sector_synthesis", _fake_synth_sector)
    monkeypatch.setattr("app.rlm.executor.run_synthesis_for_ticker", _fake_synth_ticker)
    monkeypatch.setattr(
        "app.rlm.executor.run_whale_signals_for_run",
        lambda **kwargs: {"summary_path": str(whale_summary_path)},
    )
    monkeypatch.setattr(
        "app.rlm.executor.build_sector_decision_pack",
        lambda **kwargs: {
            "decision_pack_path": str(decision_json),
            "decision_pack_md_path": str(decision_md),
        },
    )

    state = LoopState(
        run_id="rlm_exec_test",
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=["AAPL", "MSFT"],
        top_k_current=["AAPL", "MSFT"],
        artifacts={"peer_rankings_path": str(rankings_path)},
        budgets_remaining=LoopBudgets(sec_budget_remaining=100, llm_budget_remaining=2.0, max_iterations=2),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 0,
            "objective": "dispatch test",
            "actions": [
                {"action_type": "CLOSE_GAPS", "tickers": ["AAPL"], "limit": 1},
                {"action_type": "DEEPEN_DOSSIER", "tickers": ["AAPL"], "years_back": 5},
                {"action_type": "REBUILD_SCOREBOARD"},
                {"action_type": "RUN_SYNTHESIS", "target": {"scope": "sector", "value": "Software"}},
                {"action_type": "RUN_SYNTHESIS", "target": {"scope": "ticker", "value": "MSFT"}},
                {"action_type": "NARROW_PEER_SET", "filter_rules": {"keep_top_n": 1}},
            ],
        }
    )

    result = execute_actions(
        state=state,
        planner_output=planner,
        top_k=2,
        years_back_default=10,
        workers=1,
        with_research=True,
        with_synthesis=True,
    )

    assert result["executed_count"] == 6
    assert calls["close_gaps"] == 1
    assert calls["deepen"] == 1
    assert calls["rebuild"] == 1
    assert calls["synth_sector"] == 1
    assert calls["synth_ticker"] == 1
    assert state.peer_set == ["AAPL"]
    assert Path(state.artifacts["peer_rankings_path"]).exists()
    assert Path(state.artifacts["peer_scoreboard_path"]).exists()
    assert Path(state.artifacts["peer_report_path"]).exists()


def test_refine_peer_set_uses_historical_seed(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "rlm_exec_refine_seed"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_run_id = "seed_run"
    seed_dir = cfg.sectors_dir / seed_run_id
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "sector_summary.json").write_text(
        json.dumps(
            {
                "run_id": seed_run_id,
                "sector": "Software",
                "as_of_date": "2026-02-13",
                "updated_at": "2026-02-14T00:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (seed_dir / "sector_peers.json").write_text(
        json.dumps(
            {
                "selected_tickers": ["MSFT", "AAPL", "ORCL"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "app.rlm.executor.select_sector_peers",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("network should not be called")),
    )

    state = LoopState(
        run_id=run_id,
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=[],
        top_k_current=[],
        artifacts={},
        budgets_remaining=LoopBudgets(sec_budget_remaining=100, llm_budget_remaining=2.0, max_iterations=2),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 0,
            "objective": "refine seed fallback",
            "actions": [
                {
                    "action_type": "REFINE_PEER_SET",
                    "peer_mode": "hybrid",
                    "min_peers_dossierable": 2,
                    "min_annual_filings": 2,
                }
            ],
        }
    )

    result = execute_actions(
        state=state,
        planner_output=planner,
        top_k=2,
        years_back_default=10,
        workers=1,
        with_research=False,
        with_synthesis=False,
    )

    assert result["executed_count"] == 1
    details = result["results"][0]["details"]
    assert details["peer_mode"] == "historical_seed"
    assert details["peer_selection_summary"]["seed_run_id"] == seed_run_id
    assert state.peer_set == ["MSFT", "AAPL", "ORCL"]
    assert state.top_k_current == ["MSFT", "AAPL"]
    assert (run_dir / "sector_peers.json").exists()


def test_execute_actions_does_not_abandon_slow_stage_worker(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    calls = {"fast": 0}

    def _slow_dossiers(**kwargs):
        time.sleep(0.05)
        return {"run_id": kwargs["run_id"], "tickers_requested": kwargs["tickers"]}

    def _fast_synth(**kwargs):
        calls["fast"] += 1
        return {"path": "data/outputs/sectors/rlm_exec_timeout/sector_synthesis.json"}

    monkeypatch.setattr("app.rlm.executor.run_dossier_for_peer_set", _slow_dossiers)
    monkeypatch.setattr("app.rlm.executor.run_sector_synthesis", _fast_synth)

    state = LoopState(
        run_id="rlm_exec_timeout",
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=["MSFT"],
        top_k_current=["MSFT"],
        artifacts={},
        budgets_remaining=LoopBudgets(sec_budget_remaining=100, llm_budget_remaining=2.0, max_iterations=2),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 0,
            "objective": "timeout test",
            "actions": [
                {"action_type": "BUILD_DOSSIERS", "tickers": ["MSFT"], "years_back": 5},
                {"action_type": "RUN_SYNTHESIS", "target": {"scope": "sector", "value": "Software"}},
            ],
        }
    )

    result = execute_actions(
        state=state,
        planner_output=planner,
        top_k=1,
        years_back_default=10,
        workers=1,
        with_research=False,
        with_synthesis=True,
        timeout_per_stage=0.01,
    )

    assert result["executed_count"] == 2
    assert result["results"][0]["status"] == "OK"
    assert result["results"][1]["status"] == "OK"
    assert calls["fast"] == 1


def test_execute_actions_propagates_financial_integrity_failure(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    integrity_error = InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context="rlm_executor_test",
            run_as_of_date="2026-02-13",
            status="INVALID_FINANCIAL_INPUT",
        )
    )
    monkeypatch.setattr(
        "app.rlm.executor.run_sector_synthesis",
        lambda **_kwargs: (_ for _ in ()).throw(integrity_error),
    )

    state = LoopState(
        run_id="rlm_exec_integrity",
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=["MSFT"],
        top_k_current=["MSFT"],
        artifacts={},
        budgets_remaining=LoopBudgets(
            sec_budget_remaining=100,
            llm_budget_remaining=2.0,
            max_iterations=2,
        ),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 0,
            "objective": "integrity propagation",
            "actions": [
                {
                    "action_type": "RUN_SYNTHESIS",
                    "target": {"scope": "sector", "value": "Software"},
                }
            ],
        }
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        execute_actions(
            state=state,
            planner_output=planner,
            top_k=1,
            years_back_default=10,
            workers=1,
            with_research=False,
            with_synthesis=True,
        )

    assert exc_info.value is integrity_error


def test_build_dossiers_copies_from_historical_seed(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "rlm_exec_dossier_copy"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "peer_selection_summary.json").write_text(
        json.dumps(
            {
                "mode": "historical_seed",
                "seed_run_id": "seed_dossier_run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seed_dossier_dir = cfg.dossiers_dir / "seed_dossier_run"
    for ticker in ["AAPL", "MSFT"]:
        ticker_dir = seed_dossier_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        (ticker_dir / "dossier.json").write_text(
            json.dumps({"ticker": ticker, "as_of_date": "2026-02-13"}),
            encoding="utf-8",
        )

    summary_path = cfg.dossiers_dir / run_id / "dossier_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "tickers_requested": ["AAPL", "MSFT"],
                "tickers_built": [],
                "tickers_failed": ["AAPL", "MSFT"],
                "tickers_skipped": [],
                "tickers_skipped_budget": [],
                "tickers_pending": [],
                "ticker_results": {
                    "AAPL": {"status": "FAILED", "error": "dns"},
                    "MSFT": {"status": "FAILED", "error": "dns"},
                },
                "summary_path": str(summary_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "app.rlm.executor.run_dossier_for_peer_set",
        lambda **_kwargs: json.loads(summary_path.read_text(encoding="utf-8")),
    )

    state = LoopState(
        run_id=run_id,
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=["AAPL", "MSFT"],
        top_k_current=["AAPL", "MSFT"],
        artifacts={},
        budgets_remaining=LoopBudgets(sec_budget_remaining=100, llm_budget_remaining=2.0, max_iterations=2),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 0,
            "objective": "copy dossiers",
            "actions": [
                {"action_type": "BUILD_DOSSIERS", "tickers": ["AAPL", "MSFT"], "limit": 2},
            ],
        }
    )

    result = execute_actions(
        state=state,
        planner_output=planner,
        top_k=2,
        years_back_default=10,
        workers=1,
        with_research=False,
        with_synthesis=False,
    )

    details = result["results"][0]["details"]
    copied = details["copied_from_history"]
    assert copied == [
        {"ticker": "AAPL", "source_run_id": "seed_dossier_run"},
        {"ticker": "MSFT", "source_run_id": "seed_dossier_run"},
    ]
    assert (cfg.dossiers_dir / run_id / "AAPL" / "dossier.json").exists()
    assert (cfg.dossiers_dir / run_id / "MSFT" / "dossier.json").exists()
    updated_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert sorted(updated_summary["tickers_built"]) == ["AAPL", "MSFT"]
    assert updated_summary["tickers_failed"] == []


def test_executor_handles_prewarm_prices_action(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_dir = cfg.sectors_dir / "rlm_exec_prewarm"
    run_dir.mkdir(parents=True, exist_ok=True)
    prewarm_path = run_dir / "prices_prewarm.json"
    prewarm_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "app.rlm.executor.write_prices_prewarm_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "as_of_date": kwargs["as_of_date"],
            "tickers_requested": kwargs["tickers"],
            "tickers_ok": ["AAPL"],
            "tickers_unknown": [],
            "ok_count": 1,
            "unknown_count": 0,
            "reason_counts": {"PROVIDER_OK": 1},
            "prices_summary_path": "data/outputs/prices/rlm_exec_prewarm/prices_summary.json",
            "prices_prewarm_path": str(prewarm_path),
        },
    )

    state = LoopState(
        run_id="rlm_exec_prewarm",
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=["AAPL"],
        top_k_current=["AAPL"],
        artifacts={},
        budgets_remaining=LoopBudgets(sec_budget_remaining=100, llm_budget_remaining=2.0, max_iterations=2),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 0,
            "objective": "prewarm",
            "actions": [
                {"action_type": "PREWARM_PRICES", "tickers": ["AAPL"], "fallback_days": 5},
            ],
        }
    )

    result = execute_actions(
        state=state,
        planner_output=planner,
        top_k=1,
        years_back_default=10,
        workers=1,
        with_research=False,
        with_synthesis=False,
    )

    assert result["executed_count"] == 1
    assert result["results"][0]["status"] == "OK"
    assert state.artifacts["prices_prewarm_path"] == str(prewarm_path)


def test_executor_handles_hydrate_price_snapshot_action(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_dir = cfg.sectors_dir / "rlm_exec_hydrate_price"
    run_dir.mkdir(parents=True, exist_ok=True)
    prewarm_path = run_dir / "prices_prewarm.json"
    prewarm_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "app.rlm.executor.write_prices_prewarm_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "as_of_date": kwargs["as_of_date"],
            "tickers_requested": kwargs["tickers"],
            "tickers_ok": ["AAPL"],
            "tickers_unknown": ["MSFT"],
            "ok_count": 1,
            "unknown_count": 1,
            "reason_counts": {"CACHE_HIT": 1, "OFFLINE_NO_CACHE": 1},
            "prices_summary_path": "data/outputs/prices/rlm_exec_hydrate_price/prices_summary.json",
            "prices_prewarm_path": str(prewarm_path),
        },
    )

    state = LoopState(
        run_id="rlm_exec_hydrate_price",
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=["AAPL", "MSFT"],
        top_k_current=["AAPL", "MSFT"],
        artifacts={},
        budgets_remaining=LoopBudgets(sec_budget_remaining=100, llm_budget_remaining=2.0, max_iterations=2),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 1,
            "objective": "hydrate missing prices",
            "actions": [
                {"action_type": "HYDRATE_PRICE_SNAPSHOT", "tickers": ["AAPL", "MSFT"], "fallback_days": 5},
            ],
        }
    )

    result = execute_actions(
        state=state,
        planner_output=planner,
        top_k=2,
        years_back_default=10,
        workers=1,
        with_research=False,
        with_synthesis=False,
    )

    assert result["executed_count"] == 1
    assert result["results"][0]["status"] == "OK"
    assert state.artifacts["prices_prewarm_path"] == str(prewarm_path)
    assert result["results"][0]["details"]["hydration_action"] == "HYDRATE_PRICE_SNAPSHOT"


def test_executor_handles_hydrate_financial_facts_action(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_dir = cfg.sectors_dir / "rlm_exec_facts"
    run_dir.mkdir(parents=True, exist_ok=True)
    facts_path = run_dir / "facts_coverage.json"
    facts_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "app.rlm.executor.write_facts_coverage_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "as_of_date": kwargs["as_of_date"],
            "ticker_count": len(kwargs["tickers"]),
            "status_counts": {"OK": 1, "PARTIAL": 0, "UNKNOWN": 0},
            "facts_coverage_path": str(facts_path),
        },
    )

    state = LoopState(
        run_id="rlm_exec_facts",
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=["AAPL"],
        top_k_current=["AAPL"],
        artifacts={},
        budgets_remaining=LoopBudgets(sec_budget_remaining=100, llm_budget_remaining=2.0, max_iterations=2),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 0,
            "objective": "facts hydrate",
            "actions": [
                {"action_type": "HYDRATE_FINANCIAL_FACTS", "tickers": ["AAPL"], "limit": 1},
            ],
        }
    )

    result = execute_actions(
        state=state,
        planner_output=planner,
        top_k=1,
        years_back_default=10,
        workers=1,
        with_research=False,
        with_synthesis=False,
    )

    assert result["executed_count"] == 1
    assert result["results"][0]["status"] == "OK"
    assert state.artifacts["facts_coverage_path"] == str(facts_path)
