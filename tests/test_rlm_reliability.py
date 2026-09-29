from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.autonomous.financial_integrity import (
    FinancialIntegrityGateResult,
    InvalidFinancialInputError,
)
from app.config import get_config
from app.db import get_db, init_db
from app.rlm.loop import (
    cancel_sector_rlm_run,
    resume_sector_rlm_loop,
    run_sector_rlm_loop,
    sector_rlm_status,
)
from app.rlm.planner import generate_plan
from app.rlm.schemas import Action, PlannerOutput
from app.rlm.state import (
    get_max_iteration_for_run,
    init_loop_state,
    insert_rlm_iteration_row,
    load_loop_state,
    save_loop_state,
    upsert_rlm_run_row,
)


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """Planner policy tests use synthetic value-gate artifacts."""

    class _Scope:
        def require(self, **_kwargs):
            return None

    class _Context:
        def __init__(self, *, tickers, as_of_date):
            self.as_of_date = as_of_date
            self.packets = {
                str(ticker).upper(): {
                    "ticker": str(ticker).upper(),
                    "quote_snapshot_id": "fixture",
                    "current_price": 10.0,
                    "current_price_unit": "USD_per_share",
                    "price_basis": "UNADJUSTED",
                }
                for ticker in tickers
            }

    monkeypatch.setattr(
        "app.rlm.planner.bind_v1_financial_scope",
        lambda **_kwargs: _Scope(),
    )
    monkeypatch.setattr(
        "app.rlm.planner.build_canonical_v1_financial_context",
        lambda **kwargs: _Context(
            tickers=kwargs["tickers"],
            as_of_date=kwargs["as_of_date"],
        ),
    )


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _seed_running_state(run_id: str, *, sector: str = "Software", as_of_date: str = "2026-02-13"):
    state = init_loop_state(
        run_id=run_id,
        sector=sector,
        as_of_date=as_of_date,
        max_iterations=3,
        llm_budget_usd=1.0,
        sec_budget_count=100,
    )
    state.status = "RUNNING"
    save_loop_state(state)
    upsert_rlm_run_row(state)
    return state


def _stop_planner(iteration: int, objective: str = "stop") -> PlannerOutput:
    return PlannerOutput(
        iteration=iteration,
        objective=objective,
        actions=[
            Action(
                action_type="STOP",
                reason_code="TEST_STOP",
                summary="deterministic test stop",
            )
        ],
    )


def test_idempotent_insert_updates_existing_iteration(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_insert_idempotent")

    insert_rlm_iteration_row(
        run_id="rlm_insert_idempotent",
        iteration=0,
        planner_json={"iteration": 0, "actions": [{"a": 1}]},
        critic_json={"iteration": 0, "confidence": "LOW"},
        progress_json={"step": "first"},
    )
    insert_rlm_iteration_row(
        run_id="rlm_insert_idempotent",
        iteration=0,
        planner_json={"iteration": 0, "actions": [{"a": 2}]},
        critic_json={"iteration": 0, "confidence": "HIGH"},
        progress_json={"step": "second"},
    )

    with get_db() as conn:
        row = conn.execute(
            "SELECT planner_json, critic_json, progress_json FROM rlm_iterations WHERE run_id = ? AND iteration = 0",
            ("rlm_insert_idempotent",),
        ).fetchone()
    assert row is not None
    assert json.loads(row["planner_json"])["actions"][0]["a"] == 2
    assert json.loads(row["critic_json"])["confidence"] == "HIGH"
    assert json.loads(row["progress_json"])["step"] == "second"


def test_exception_marks_run_failed(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_exception")

    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom planner")),
    )

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_exception",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=2,
        top_k=3,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=True,
    )
    assert payload["status"] == "FAILED"
    assert payload["stop_reason_code"] == "EXCEPTION"

    with get_db() as conn:
        row = conn.execute(
            "SELECT status, stop_reason FROM rlm_runs WHERE run_id = ?", ("rlm_exception",)
        ).fetchone()
    assert row is not None
    assert row["status"] == "FAILED"
    assert "EXCEPTION" in str(row["stop_reason"] or "")


def test_financial_integrity_failure_escapes_loop_failure_boundary(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_integrity_failure")

    integrity_error = InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context="rlm_planner_test",
            run_as_of_date="2026-02-13",
            status="INVALID_FINANCIAL_INPUT",
        )
    )
    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **_kwargs: (_ for _ in ()).throw(integrity_error),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_rlm_loop(
            sector="Software",
            as_of_date="2026-02-13",
            run_id="rlm_integrity_failure",
            peer_limit=10,
            min_peers=5,
            limit_dossiers=5,
            years_back=10,
            workers=1,
            iterations=2,
            top_k=3,
            with_research=False,
            with_synthesis=False,
            budget_usd=1.0,
            resume=True,
        )

    assert exc_info.value is integrity_error
    with get_db() as conn:
        row = conn.execute(
            "SELECT status, stop_reason FROM rlm_runs WHERE run_id = ?",
            ("rlm_integrity_failure",),
        ).fetchone()
    assert row is not None
    assert row["status"] == "FAILED"
    assert "INVALID_FINANCIAL_INPUT" in str(row["stop_reason"] or "")
    assert not (
        Path(get_config().outputs_dir)
        / "sectors"
        / "rlm_integrity_failure"
        / "rlm"
        / "final_decision_pack.json"
    ).exists()


def test_successful_executor_provider_usage_is_debited_once(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_executor_cost")

    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **kwargs: (
            _stop_planner(int(kwargs["state"].iteration), "cost-step"),
            {"cost_estimate_usd": 0.0},
        ),
    )

    def _execute_with_paid_usage(**kwargs):
        from app.llm.usage_capture import record_provider_usage

        record_provider_usage(
            {
                "status": "OK",
                "lane": "rlm_executor",
                "provider": "openai",
                "model": "gpt-5-mini",
                "schema_name": "sector_synthesis_packet_v1",
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 20,
                "reserved_output_tokens": 0,
                "estimated_tokens": False,
                "cost_estimate_usd": 0.125,
            }
        )
        return {
            "planner_requested_stop": True,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": list(kwargs["state"].peer_set),
        }

    monkeypatch.setattr("app.rlm.loop.execute_actions", _execute_with_paid_usage)

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_executor_cost",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=1,
        top_k=3,
        with_research=False,
        with_synthesis=True,
        budget_usd=1.0,
        resume=True,
    )

    assert payload["status"] in {"DONE", "STOPPED"}
    state = load_loop_state("rlm_executor_cost")
    assert state is not None
    assert state.llm_cost_used == 0.125
    assert state.budgets_remaining.llm_budget_remaining == 0.875
    execution = state.action_history[0]["execution"]
    assert execution["cost_estimate_usd"] == 0.125
    assert len(execution["provider_usage"]) == 1


def test_post_response_integrity_failure_persists_cost_and_no_final_pack(
    monkeypatch,
    tmp_path,
):
    _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_executor_integrity_cost")

    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **kwargs: (
            _stop_planner(int(kwargs["state"].iteration), "integrity-cost-step"),
            {"cost_estimate_usd": 0.0},
        ),
    )
    integrity_error = InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context="rlm_executor_post_response",
            run_as_of_date="2026-02-13",
            status="INVALID_FINANCIAL_INPUT",
        )
    )

    def _execute_then_fail(**_kwargs):
        from app.llm.usage_capture import record_provider_usage

        record_provider_usage(
            {
                "status": "OK",
                "lane": "rlm_executor",
                "provider": "openai",
                "model": "gpt-5-mini",
                "schema_name": "filing_change_set_v1",
                "input_tokens": 120,
                "cached_input_tokens": 0,
                "output_tokens": 30,
                "reserved_output_tokens": 0,
                "estimated_tokens": False,
                "cost_estimate_usd": 0.2,
            }
        )
        raise integrity_error

    monkeypatch.setattr("app.rlm.loop.execute_actions", _execute_then_fail)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_rlm_loop(
            sector="Software",
            as_of_date="2026-02-13",
            run_id="rlm_executor_integrity_cost",
            peer_limit=10,
            min_peers=5,
            limit_dossiers=5,
            years_back=10,
            workers=1,
            iterations=1,
            top_k=3,
            with_research=False,
            with_synthesis=True,
            budget_usd=1.0,
            resume=True,
        )

    assert exc_info.value is integrity_error
    state = load_loop_state("rlm_executor_integrity_cost")
    assert state is not None
    assert state.status == "FAILED"
    assert state.llm_cost_used == 0.2
    assert state.budgets_remaining.llm_budget_remaining == 0.8
    assert not (
        get_config().sectors_dir
        / "rlm_executor_integrity_cost"
        / "rlm"
        / "final_decision_pack.json"
    ).exists()


def test_tiny_budget_refuses_planner_before_provider_call(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_tiny_budget")

    class _CountingProvider:
        provider_name = "openai"

        def __init__(self):
            self.calls = 0

        def enabled(self) -> bool:
            return True

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            raise AssertionError("tiny budget must suppress physical provider call")

    provider = _CountingProvider()
    monkeypatch.setattr("app.rlm.planner.get_llm_provider", lambda: provider)

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_tiny_budget",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=1,
        top_k=3,
        with_research=False,
        with_synthesis=False,
        budget_usd=0.0000001,
        resume=True,
    )

    assert payload["status"] == "FAILED"
    assert provider.calls == 0
    state = load_loop_state("rlm_tiny_budget")
    assert state is not None
    assert state.llm_cost_used == 0.0


def test_resume_continues_from_next_iteration(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    state = _seed_running_state("rlm_resume")
    state.iteration = 0
    save_loop_state(state)
    upsert_rlm_run_row(state)
    insert_rlm_iteration_row(
        run_id="rlm_resume",
        iteration=0,
        planner_json={"iteration": 0, "actions": []},
        critic_json={"iteration": 0},
        progress_json={"already": True},
    )

    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **kwargs: (
            _stop_planner(int(kwargs["state"].iteration), "resume-step"),
            {"cost_estimate_usd": 0.0},
        ),
    )
    monkeypatch.setattr(
        "app.rlm.loop.execute_actions",
        lambda **kwargs: {
            "planner_requested_stop": True,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": [],
        },
    )

    payload = resume_sector_rlm_loop(run_id="rlm_resume", iterations=2, top_k=3, workers=1)
    assert payload["status"] == "STOPPED"
    assert payload["iteration"] >= 2
    assert get_max_iteration_for_run("rlm_resume") == 1


def test_cancel_requested_stops_loop_at_checkpoint(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_cancel")

    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **kwargs: (
            _stop_planner(int(kwargs["state"].iteration), "cancel-step"),
            {"cost_estimate_usd": 0.0},
        ),
    )

    def _cancel_during_execute(**kwargs):
        cancel_sector_rlm_run(run_id="rlm_cancel", reason="test checkpoint cancel")
        return {
            "planner_requested_stop": False,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": [],
        }

    monkeypatch.setattr("app.rlm.loop.execute_actions", _cancel_during_execute)

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_cancel",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=2,
        top_k=3,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=True,
    )
    assert payload["status"] == "CANCELLED"
    assert payload["stop_reason_code"] == "CANCELLED"


def test_planner_timeout_setting_does_not_orphan_paid_work(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_planner_timeout")
    planner_finished = {"value": False}

    def _slow_plan(**kwargs):
        time.sleep(0.05)
        planner_finished["value"] = True
        return _stop_planner(int(kwargs["state"].iteration), "slow-step"), {
            "cost_estimate_usd": 0.0
        }

    monkeypatch.setattr("app.rlm.loop.generate_plan", _slow_plan)
    monkeypatch.setattr(
        "app.rlm.loop.execute_actions",
        lambda **kwargs: {
            "planner_requested_stop": True,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": list(kwargs["state"].peer_set),
        },
    )

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_planner_timeout",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=1,
        top_k=3,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=True,
        timeout_per_stage=0.01,
    )
    assert payload["status"] in {"DONE", "STOPPED"}
    assert planner_finished["value"] is True

    log_path = Path(payload["run_log_path"])
    assert log_path.exists()
    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not any(entry.get("event") == "STAGE_TIMEOUT" for entry in entries)
    assert any(entry.get("event") == "planner_complete" for entry in entries)


def test_heartbeat_written_and_status_reports_stale(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_running_state("rlm_heartbeat")

    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **kwargs: (
            _stop_planner(int(kwargs["state"].iteration), "heartbeat-step"),
            {"cost_estimate_usd": 0.0},
        ),
    )
    monkeypatch.setattr(
        "app.rlm.loop.execute_actions",
        lambda **kwargs: {
            "planner_requested_stop": True,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": [],
        },
    )

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_heartbeat",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=1,
        top_k=3,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=True,
    )
    assert payload["heartbeat_path"].endswith("rlm_heartbeat.json")

    status = sector_rlm_status(run_id="rlm_heartbeat")
    assert status["heartbeat_path"].endswith("rlm_heartbeat.json")
    assert status["heartbeat_age_seconds"] is not None

    # Simulate stale RUNNING heartbeat with dead pid.
    with get_db() as conn:
        conn.execute(
            "UPDATE rlm_runs SET status = 'RUNNING', stop_reason = NULL, updated_at = ? WHERE run_id = ?",
            (datetime.now(timezone.utc).isoformat(), "rlm_heartbeat"),
        )

    heartbeat_path = cfg.sectors_dir / "rlm_heartbeat" / "rlm_heartbeat.json"
    heartbeat_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    heartbeat_payload["updated_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=30)
    ).isoformat()
    heartbeat_payload["pid"] = 999999
    heartbeat_path.write_text(json.dumps(heartbeat_payload, indent=2), encoding="utf-8")

    stale = sector_rlm_status(run_id="rlm_heartbeat")
    assert stale["health"] == "STALE_RUN_DETECTED"
    assert stale["heartbeat_stale"] is True
    assert stale["heartbeat_pid_active"] is False


def test_sector_rlm_disabled_baseline_stop(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    state = _seed_running_state("rlm_disabled_baseline")
    state.peer_set = ["AAPL", "MSFT"]
    state.top_k_current = ["AAPL", "MSFT"]
    save_loop_state(state)
    upsert_rlm_run_row(state)

    def _fake_execute(**kwargs):
        state_obj = kwargs["state"]
        run_dir = cfg.sectors_dir / state_obj.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rankings_path = run_dir / "peer_rankings.json"
        scoreboard_path = run_dir / "peer_scoreboard.json"
        decision_json = run_dir / "decision_pack.json"
        decision_md = run_dir / "decision_pack.md"
        rankings_path.write_text(
            json.dumps(
                {
                    "future_whale_rank": ["AAPL", "MSFT"],
                    "whale_signature_rank": ["AAPL", "MSFT"],
                    "rankings": [
                        {
                            "ticker": "AAPL",
                            "metric_ranks": {
                                "future_whale_rank": 1,
                                "whale_signature_rank": 1,
                                "quality_rank": 1,
                                "valuation_rank": 1,
                                "risk_rank": 1,
                            },
                            "overall_score": 10.0,
                            "whale_signature": {"score": 80.0, "top_signals": [], "gaps": []},
                        },
                        {
                            "ticker": "MSFT",
                            "metric_ranks": {
                                "future_whale_rank": 2,
                                "whale_signature_rank": 2,
                                "quality_rank": 2,
                                "valuation_rank": 2,
                                "risk_rank": 2,
                            },
                            "overall_score": 9.0,
                            "whale_signature": {"score": 70.0, "top_signals": [], "gaps": []},
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        scoreboard_path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "ticker": "AAPL",
                            "metric_values": {
                                "whale_signature_score": 80.0,
                                "revenue_cagr_10y": 0.2,
                            },
                            "metric_traces": {
                                "whale_signature_score": {"derived_from": ["dossier.whale"]}
                            },
                            "whale_summary": {"gaps": [], "top_signals": []},
                        },
                        {
                            "ticker": "MSFT",
                            "metric_values": {
                                "whale_signature_score": 70.0,
                                "revenue_cagr_10y": 0.1,
                            },
                            "metric_traces": {
                                "whale_signature_score": {"derived_from": ["dossier.whale"]}
                            },
                            "whale_summary": {"gaps": [], "top_signals": []},
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        decision_json.write_text(
            json.dumps(
                {
                    "run_id": state_obj.run_id,
                    "sector": state_obj.sector,
                    "as_of_date": state_obj.as_of_date,
                    "top_peers": [],
                    "top_candidates_to_deepen": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        decision_md.write_text("# Decision Pack\n", encoding="utf-8")
        state_obj.artifacts.update(
            {
                "peer_rankings_path": str(rankings_path),
                "peer_scoreboard_path": str(scoreboard_path),
                "decision_pack_path": str(decision_json),
                "decision_pack_md_path": str(decision_md),
            }
        )
        return {
            "planner_requested_stop": False,
            "executed_count": 5,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": list(state_obj.peer_set),
        }

    monkeypatch.setattr("app.rlm.loop.execute_actions", _fake_execute)

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_disabled_baseline",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=2,
        top_k=3,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=True,
        mode="depth",
    )
    assert payload["status"] == "DONE"
    assert payload["stop_reason_code"] == "DISABLED_PROVIDER_COMPLETED_BASELINE"
    assert payload["mode"] == "depth"
    assert Path(payload["artifacts"]["peer_scoreboard_path"]).exists()
    assert Path(payload["artifacts"]["decision_pack_path"]).exists()


def test_rlm_depth_planner_picks_resolve_shares_when_missing(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "rlm_depth_resolve_shares"
    state = _seed_running_state(run_id)
    state.mode = "depth"
    state.rlm_version = "v1.1"
    state.iteration = 1
    state.peer_set = ["AAA", "BBB"]
    state.top_k_current = ["AAA", "BBB"]

    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    valuation_cov_path = run_dir / "valuation_coverage.json"
    valuation_cov_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "reason_counts": {"MISSING_SHARES": 1, "OK": 1},
                "entries": [
                    {
                        "ticker": "AAA",
                        "price_status": "OK",
                        "shares_status": "UNKNOWN",
                        "shares_reason_code": "NO_FILINGS",
                        "valuation_status": "UNKNOWN",
                        "valuation_reason_code": "MISSING_SHARES",
                    },
                    {
                        "ticker": "BBB",
                        "price_status": "OK",
                        "shares_status": "OK",
                        "shares_reason_code": "PROVIDER_OK",
                        "valuation_status": "OK",
                        "valuation_reason_code": None,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state.artifacts["valuation_coverage_path"] = str(valuation_cov_path)

    plan, meta = generate_plan(state=state, top_k=2, mode="depth")
    assert meta["fallback"] is True
    action_types = [action.action_type for action in plan.actions]
    assert action_types[:3] == [
        "HYDRATE_FINANCIAL_FACTS",
        "RECOMPUTE_VALUATION",
        "UPDATE_SCOREBOARD",
    ]
    assert plan.actions[-1].action_type == "STOP"


def test_rlm_depth_planner_bootstraps_without_baseline_artifacts(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    state = init_loop_state(
        run_id="rlm_depth_bootstrap",
        sector="Software",
        as_of_date="2026-02-13",
        max_iterations=2,
        llm_budget_usd=1.0,
        sec_budget_count=100,
    )
    state.mode = "depth"
    state.rlm_version = "v1.1"
    state.iteration = 0
    state.peer_set = []
    state.top_k_current = []

    plan, meta = generate_plan(state=state, top_k=3, mode="depth")
    assert meta["fallback"] is True
    action_types = [action.action_type for action in plan.actions]
    assert action_types == [
        "REFINE_PEER_SET",
        "HYDRATE_PRICE_SNAPSHOT",
        "BUILD_DOSSIERS",
        "BUILD_FUNDAMENTALS",
        "VALUE_TICKER",
        "UPDATE_SCOREBOARD",
    ]
    assert plan.actions[2].tickers == []
    assert plan.actions[2].limit == 3


def test_rlm_depth_planner_schedules_price_prewarm_when_unknown_dominates(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "rlm_depth_price_prewarm"
    state = _seed_running_state(run_id)
    state.mode = "depth"
    state.rlm_version = "v1.1"
    state.iteration = 1
    state.peer_set = ["AAA", "BBB", "CCC"]
    state.top_k_current = ["AAA", "BBB"]

    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    price_cov_path = run_dir / "price_coverage.json"
    price_cov_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "entries": [
                    {
                        "ticker": "AAA",
                        "result": {"status": "UNKNOWN", "reason_code": "PROVIDER_NO_DATA"},
                    },
                    {
                        "ticker": "BBB",
                        "result": {
                            "status": "UNKNOWN",
                            "reason_code": "NON_TRADING_DAY_NO_FALLBACK",
                        },
                    },
                    {"ticker": "CCC", "result": {"status": "OK", "reason_code": "PROVIDER_OK"}},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state.artifacts["price_coverage_path"] = str(price_cov_path)

    plan, meta = generate_plan(state=state, top_k=2, mode="depth")
    assert meta["fallback"] is True
    action_types = [action.action_type for action in plan.actions]
    assert action_types[:2] == ["HYDRATE_PRICE_SNAPSHOT", "RECOMPUTE_VALUATION"]


def test_rlm_depth_planner_deprioritizes_fail_for_expensive_actions(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "rlm_depth_value_gates_policy"
    state = _seed_running_state(run_id)
    state.mode = "depth"
    state.rlm_version = "v1.1"
    state.iteration = 1
    state.peer_set = ["AAA", "BBB"]
    state.top_k_current = ["AAA", "BBB"]

    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    value_gates_path = run_dir / "value_gates.json"
    value_gates_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "summary": {"counts": {"PASS": 1, "WATCH": 0, "FAIL": 1}},
                "entries": [
                    {
                        "ticker": "AAA",
                        "gate_status": "FAIL",
                        "gate_reasons": ["MOS_FAIL", "DILUTION_FAIL"],
                    },
                    {
                        "ticker": "BBB",
                        "gate_status": "PASS",
                        "gate_reasons": ["MOS_PASS", "FCF_POSITIVE_TREND_OK"],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    valuation_cov_path = run_dir / "valuation_coverage.json"
    valuation_cov_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "ticker": "AAA",
                        "price_status": "OK",
                        "shares_status": "OK",
                        "fcf_status": "OK",
                        "valuation_reason_code": None,
                    },
                    {
                        "ticker": "BBB",
                        "price_status": "OK",
                        "shares_status": "OK",
                        "fcf_status": "OK",
                        "valuation_reason_code": None,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state.artifacts["value_gates_path"] = str(value_gates_path)
    state.artifacts["valuation_coverage_path"] = str(valuation_cov_path)

    plan, meta = generate_plan(state=state, top_k=2, mode="depth")
    assert meta["fallback"] is True
    action_types = [action.action_type for action in plan.actions]
    assert action_types[:3] == ["BUILD_DOSSIERS", "RUN_RESEARCH_GAP_CLOSER", "RUN_SYNTHESIS"]
    assert plan.actions[0].tickers == ["BBB"]
    assert plan.actions[1].tickers == ["BBB"]
    assert (plan.actions[2].target or {}).get("value") == "BBB"
    expensive_tickers = []
    for action in plan.actions:
        if action.action_type in {"BUILD_DOSSIERS", "RUN_RESEARCH_GAP_CLOSER"}:
            expensive_tickers.extend(action.tickers)
    assert "AAA" not in expensive_tickers


def test_rlm_depth_planner_emits_calibration_required_when_all_fail(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "rlm_depth_calibration_required"
    state = _seed_running_state(run_id)
    state.mode = "depth"
    state.rlm_version = "v1.1"
    state.iteration = 1
    state.peer_set = ["AAA", "BBB"]
    state.top_k_current = ["AAA", "BBB"]

    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    value_gates_path = run_dir / "value_gates.json"
    value_gates_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "summary": {"counts": {"PASS": 0, "WATCH": 0, "FAIL": 2}},
                "entries": [
                    {
                        "ticker": "AAA",
                        "gate_status": "FAIL",
                        "gate_reasons": ["MOS_FAIL", "BALANCE_FAIL"],
                        "primary_blocker": "MOS_FAIL",
                    },
                    {
                        "ticker": "BBB",
                        "gate_status": "FAIL",
                        "gate_reasons": ["MOS_FAIL"],
                        "primary_blocker": "MOS_FAIL",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    valuation_cov_path = run_dir / "valuation_coverage.json"
    valuation_cov_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "ticker": "AAA",
                        "price_status": "OK",
                        "shares_status": "OK",
                        "fcf_status": "OK",
                    },
                    {
                        "ticker": "BBB",
                        "price_status": "OK",
                        "shares_status": "OK",
                        "fcf_status": "OK",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state.artifacts["value_gates_path"] = str(value_gates_path)
    state.artifacts["valuation_coverage_path"] = str(valuation_cov_path)

    plan, meta = generate_plan(state=state, top_k=2, mode="depth")
    assert meta["fallback"] is True
    assert len(plan.actions) == 1
    assert plan.actions[0].action_type == "STOP"
    assert plan.actions[0].reason_code == "CALIBRATION_REQUIRED"
    assert "CALIBRATION_REQUIRED" in str(plan.notes or "")


def test_rlm_depth_planner_avoids_repeat_price_hydration_for_same_tickers(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "rlm_depth_price_hydration_guard"
    state = _seed_running_state(run_id)
    state.mode = "depth"
    state.rlm_version = "v1.1"
    state.iteration = 2
    state.peer_set = ["AAA", "BBB"]
    state.top_k_current = ["AAA", "BBB"]
    state.action_history.append(
        {
            "iteration": 1,
            "planner": {
                "actions": [{"action_type": "HYDRATE_PRICE_SNAPSHOT", "tickers": ["AAA", "BBB"]}]
            },
            "execution": {
                "results": [
                    {
                        "effective_action_type": "HYDRATE_PRICE_SNAPSHOT",
                        "details": {"target_tickers": ["AAA", "BBB"]},
                    }
                ]
            },
        }
    )

    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    valuation_cov_path = run_dir / "valuation_coverage.json"
    valuation_cov_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "entries": [
                    {
                        "ticker": "AAA",
                        "price_status": "UNKNOWN",
                        "price_reason_code": "OFFLINE_NO_CACHE",
                        "valuation_status": "UNKNOWN",
                        "valuation_reason_code": "PRICE_UNKNOWN",
                    },
                    {
                        "ticker": "BBB",
                        "price_status": "UNKNOWN",
                        "price_reason_code": "OFFLINE_NO_CACHE",
                        "valuation_status": "UNKNOWN",
                        "valuation_reason_code": "PRICE_UNKNOWN",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state.artifacts["valuation_coverage_path"] = str(valuation_cov_path)
    price_cov_path = run_dir / "price_coverage.json"
    price_cov_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "entries": [
                    {
                        "ticker": "AAA",
                        "result": {
                            "status": "UNKNOWN",
                            "reason_code": "OFFLINE_NO_CACHE",
                            "terminal": True,
                        },
                        "local_fallbacks": {
                            "run_scoped_output_checked": True,
                            "disk_cache_checked": True,
                            "db_quote_cache_checked": True,
                            "historical_run_artifacts_checked": True,
                            "any_hit": False,
                        },
                    },
                    {
                        "ticker": "BBB",
                        "result": {
                            "status": "UNKNOWN",
                            "reason_code": "OFFLINE_NO_CACHE",
                            "terminal": True,
                        },
                        "local_fallbacks": {
                            "run_scoped_output_checked": True,
                            "disk_cache_checked": True,
                            "db_quote_cache_checked": True,
                            "historical_run_artifacts_checked": True,
                            "any_hit": False,
                        },
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state.artifacts["price_coverage_path"] = str(price_cov_path)

    plan, meta = generate_plan(state=state, top_k=2, mode="depth")
    assert meta["fallback"] is True
    assert meta["price_terminal_offline_tickers"] == ["AAA", "BBB"]
    action_types = [action.action_type for action in plan.actions]
    assert action_types == ["STOP"]
    assert "price_terminal_offline:AAA,BBB" in str(plan.notes or "")


def test_depth_run_skips_sector_cycle_baseline(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    def _fail_cycle(**_kwargs):
        raise AssertionError("run_sector_cycle should not be called in depth mode bootstrap")

    monkeypatch.setattr("app.rlm.loop.run_sector_cycle", _fail_cycle)
    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **kwargs: (
            _stop_planner(int(kwargs["state"].iteration), "depth-bootstrap"),
            {"cost_estimate_usd": 0.0},
        ),
    )
    monkeypatch.setattr(
        "app.rlm.loop.execute_actions",
        lambda **kwargs: {
            "planner_requested_stop": True,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": [],
        },
    )

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_depth_bootstrap_runtime",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=1,
        top_k=3,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=False,
        mode="depth",
    )
    assert payload["mode"] == "depth"
    assert payload["status"] == "STOPPED"
    assert payload["stop_reason_code"] == "TEST_STOP"


def test_progress_history_records_price_terminal_offline_note(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)

    def _fail_cycle(**_kwargs):
        raise AssertionError("run_sector_cycle should not be called in depth mode bootstrap")

    monkeypatch.setattr("app.rlm.loop.run_sector_cycle", _fail_cycle)
    monkeypatch.setattr(
        "app.rlm.loop.generate_plan",
        lambda **kwargs: (
            _stop_planner(int(kwargs["state"].iteration), "depth-terminal-offline"),
            {"cost_estimate_usd": 0.0, "price_terminal_offline_tickers": ["NVDA"]},
        ),
    )
    monkeypatch.setattr(
        "app.rlm.loop.execute_actions",
        lambda **kwargs: {
            "planner_requested_stop": True,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": [],
        },
    )

    run_id = "rlm_depth_offline_note"
    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id=run_id,
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=1,
        top_k=3,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=False,
        mode="depth",
    )
    assert payload["run_id"] == run_id
    state_payload = json.loads(
        (cfg.sectors_dir / run_id / "rlm_state.json").read_text(encoding="utf-8")
    )
    progress = [
        row for row in (state_payload.get("progress_history") or []) if isinstance(row, dict)
    ]
    assert progress
    assert progress[-1]["planner_note"] == "price_terminal_offline"
    assert progress[-1]["planner_note_tickers"] == ["NVDA"]


def test_sector_rlm_iteration_delta_written(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    state = _seed_running_state("rlm_delta")
    state.peer_set = ["AAPL", "MSFT"]
    state.top_k_current = ["AAPL", "MSFT"]
    save_loop_state(state)
    upsert_rlm_run_row(state)

    def _fake_plan(**kwargs):
        state_obj = kwargs["state"]
        if int(state_obj.iteration) == 0:
            plan = PlannerOutput.model_validate(
                {
                    "iteration": 0,
                    "objective": "iter0",
                    "actions": [{"action_type": "NO_OP", "summary": "baseline noop"}],
                }
            )
        else:
            plan = PlannerOutput.model_validate(
                {
                    "iteration": 1,
                    "objective": "iter1",
                    "actions": [
                        {"action_type": "NO_OP", "summary": "second noop"},
                        {
                            "action_type": "STOP",
                            "reason_code": "TEST_STOP",
                            "summary": "stop on second iteration",
                        },
                    ],
                }
            )
        return plan, {"cost_estimate_usd": 0.0, "provider": "disabled", "model": "disabled"}

    def _fake_execute(**kwargs):
        state_obj = kwargs["state"]
        run_dir = cfg.sectors_dir / state_obj.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        value = 10.0 + float(state_obj.iteration)
        rankings_path = run_dir / "peer_rankings.json"
        scoreboard_path = run_dir / "peer_scoreboard.json"
        rankings_path.write_text(
            json.dumps(
                {
                    "future_whale_rank": ["AAPL", "MSFT"],
                    "rankings": [
                        {"ticker": "AAPL", "metric_ranks": {"future_whale_rank": 1}},
                        {"ticker": "MSFT", "metric_ranks": {"future_whale_rank": 2}},
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        scoreboard_path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "ticker": "AAPL",
                            "metric_values": {
                                "whale_signature_score": value,
                                "revenue_cagr_5y": value / 100.0,
                                "revenue_cagr_10y": value / 100.0,
                                "dilution_rate_shares_cagr": 0.01,
                                "net_debt_latest": 100.0,
                                "risk_factor_keyword_delta": 0.0,
                                "valuation_gap": 0.0,
                                "score_total": value,
                            },
                            "metric_traces": {
                                "whale_signature_score": {"derived_from": ["trace.aapl.whale"]},
                            },
                            "whale_summary": {"gaps": [], "top_signals": []},
                        },
                        {
                            "ticker": "MSFT",
                            "metric_values": {
                                "whale_signature_score": value - 1.0,
                                "revenue_cagr_5y": (value - 1.0) / 100.0,
                                "revenue_cagr_10y": (value - 1.0) / 100.0,
                                "dilution_rate_shares_cagr": 0.02,
                                "net_debt_latest": 200.0,
                                "risk_factor_keyword_delta": 0.1,
                                "valuation_gap": 0.0,
                                "score_total": value - 1.0,
                            },
                            "metric_traces": {
                                "whale_signature_score": {"derived_from": ["trace.msft.whale"]},
                            },
                            "whale_summary": {"gaps": [], "top_signals": []},
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state_obj.artifacts.update(
            {
                "peer_rankings_path": str(rankings_path),
                "peer_scoreboard_path": str(scoreboard_path),
            }
        )
        should_stop = int(state_obj.iteration) >= 1
        return {
            "planner_requested_stop": should_stop,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": list(state_obj.peer_set),
        }

    monkeypatch.setattr("app.rlm.loop.generate_plan", _fake_plan)
    monkeypatch.setattr("app.rlm.loop.execute_actions", _fake_execute)

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id="rlm_delta",
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=2,
        top_k=2,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=True,
    )
    assert payload["status"] == "STOPPED"

    delta_path = cfg.sectors_dir / "rlm_delta" / "rlm_iterations" / "iter_001_scoreboard_delta.json"
    critic_path = cfg.sectors_dir / "rlm_delta" / "rlm_iterations" / "iter_001_critic.json"
    assert delta_path.exists()
    assert critic_path.exists()
    critic_payload = json.loads(critic_path.read_text(encoding="utf-8"))
    derived = [str(item) for item in (critic_payload.get("derived_from") or [])]
    assert any("iter_001_scoreboard_delta.json" in item for item in derived)


def test_sector_rlm_calibration_required_note_written_to_delta_and_critic(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "rlm_calibration_note"
    state = _seed_running_state(run_id)
    state.mode = "depth"
    state.rlm_version = "v1.1"
    state.peer_set = ["AAPL", "MSFT"]
    state.top_k_current = ["AAPL", "MSFT"]
    save_loop_state(state)
    upsert_rlm_run_row(state)

    def _fake_plan(**kwargs):
        state_obj = kwargs["state"]
        return (
            PlannerOutput.model_validate(
                {
                    "iteration": int(state_obj.iteration),
                    "objective": "calibration note test",
                    "actions": [
                        {
                            "action_type": "STOP",
                            "reason_code": "CALIBRATION_REQUIRED",
                            "summary": "all fail in top-k",
                        }
                    ],
                }
            ),
            {"cost_estimate_usd": 0.0, "provider": "disabled", "model": "disabled"},
        )

    def _fake_execute(**kwargs):
        state_obj = kwargs["state"]
        run_dir = cfg.sectors_dir / state_obj.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rankings_path = run_dir / "peer_rankings.json"
        scoreboard_path = run_dir / "peer_scoreboard.json"
        valuation_cov_path = run_dir / "valuation_coverage.json"
        rankings_path.write_text(
            json.dumps(
                {
                    "future_whale_rank": ["AAPL", "MSFT"],
                    "value_first_rank": ["AAPL", "MSFT"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        scoreboard_path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "ticker": "AAPL",
                            "metric_values": {"whale_signature_score": 50.0, "score_total": 50.0},
                            "metric_traces": {
                                "whale_signature_score": {"derived_from": ["trace.aapl"]}
                            },
                            "whale_summary": {"gaps": [], "top_signals": []},
                        },
                        {
                            "ticker": "MSFT",
                            "metric_values": {"whale_signature_score": 49.0, "score_total": 49.0},
                            "metric_traces": {
                                "whale_signature_score": {"derived_from": ["trace.msft"]}
                            },
                            "whale_summary": {"gaps": [], "top_signals": []},
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        valuation_cov_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "ticker": "AAPL",
                            "price_status": "OK",
                            "shares_status": "OK",
                            "fcf_status": "OK",
                        },
                        {
                            "ticker": "MSFT",
                            "price_status": "OK",
                            "shares_status": "OK",
                            "fcf_status": "OK",
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state_obj.artifacts.update(
            {
                "peer_rankings_path": str(rankings_path),
                "peer_scoreboard_path": str(scoreboard_path),
                "valuation_coverage_path": str(valuation_cov_path),
            }
        )
        return {
            "planner_requested_stop": True,
            "executed_count": 1,
            "results": [],
            "updated_artifacts": {},
            "peer_set_after": list(state_obj.peer_set),
        }

    def _fake_write_value_gates_for_run(**kwargs):
        run_dir = kwargs["output_dir"]
        gates_path = run_dir / "value_gates.json"
        calibration_path = run_dir / "value_gates_calibration.json"
        gates_payload = {
            "run_id": kwargs["run_id"],
            "summary": {"counts": {"PASS": 0, "WATCH": 0, "FAIL": 2}},
            "entries": [
                {
                    "ticker": "AAPL",
                    "gate_status": "FAIL",
                    "gate_reasons": ["MOS_FAIL", "BALANCE_FAIL"],
                    "primary_blocker": "MOS_FAIL",
                },
                {
                    "ticker": "MSFT",
                    "gate_status": "FAIL",
                    "gate_reasons": ["MOS_FAIL", "DILUTION_FAIL"],
                    "primary_blocker": "MOS_FAIL",
                },
            ],
        }
        calibration_payload = {
            "run_id": kwargs["run_id"],
            "counts": {"PASS": 0, "WATCH": 0, "FAIL": 2},
            "blocker_histogram": {"MOS_FAIL": 2},
            "missing_input_breakdown": {"price": 0, "shares": 0, "fcf": 0},
            "threshold_summary": {
                "mos_min": 0.3,
                "valuation_gap_min": 0.1,
                "net_debt_to_cfo_max": 2.5,
                "dilution_max": 0.02,
            },
            "what_would_flip": [],
        }
        gates_path.write_text(json.dumps(gates_payload, indent=2), encoding="utf-8")
        calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
        out = dict(gates_payload)
        out["value_gates_path"] = str(gates_path)
        out["value_gates_calibration_path"] = str(calibration_path)
        return out

    monkeypatch.setattr("app.rlm.loop.generate_plan", _fake_plan)
    monkeypatch.setattr("app.rlm.loop.execute_actions", _fake_execute)
    monkeypatch.setattr("app.rlm.loop.write_value_gates_for_run", _fake_write_value_gates_for_run)

    payload = run_sector_rlm_loop(
        sector="Software",
        as_of_date="2026-02-13",
        run_id=run_id,
        peer_limit=10,
        min_peers=5,
        limit_dossiers=5,
        years_back=10,
        workers=1,
        iterations=1,
        top_k=2,
        with_research=False,
        with_synthesis=False,
        budget_usd=1.0,
        resume=True,
        mode="depth",
    )
    assert payload["status"] == "STOPPED"

    delta_path = cfg.sectors_dir / run_id / "rlm_iterations" / "iter_000_scoreboard_delta.json"
    critic_path = cfg.sectors_dir / run_id / "rlm_iterations" / "iter_000_critic.json"
    assert delta_path.exists()
    assert critic_path.exists()

    delta_payload = json.loads(delta_path.read_text(encoding="utf-8"))
    critic_payload = json.loads(critic_path.read_text(encoding="utf-8"))
    assert (delta_payload.get("calibration_required") or {}).get(
        "note_code"
    ) == "CALIBRATION_REQUIRED"
    assert (critic_payload.get("calibration_required") or {}).get(
        "note_code"
    ) == "CALIBRATION_REQUIRED"
