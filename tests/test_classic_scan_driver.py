from __future__ import annotations

import copy
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app.autonomous import classic_scan_driver as driver
from app.autonomous.candidate_review import provider_usage_attestation
from app.autonomous.sweep_delta import (
    CANONICAL_SWEEP_SECTORS,
    V1_ATOMIC_BANDS,
    record_loaded_set,
)
from app.cli import app
from app.config import get_config
from app.db import get_db, init_db


def _binding() -> dict[str, object]:
    return {
        "provider": "anthropic",
        "model": "claude-sonnet-test",
        "credential_present": True,
        "ready": True,
        "max_output_tokens": 8000,
        "campaign_provider_policy": "exact_provider_model_strict_no_fallback",
    }


def test_configured_provider_binding_supports_exact_deepseek_v4_pro(monkeypatch):
    monkeypatch.setenv("VOE_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("VOE_DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("VOE_DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("VOE_DEEPSEEK_MAX_OUTPUT_TOKENS", "1200")
    from app.config import get_config

    get_config.cache_clear()
    binding = driver._configured_provider_binding()
    assert binding == {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "credential_present": True,
        "ready": True,
        "max_output_tokens": 1200,
        "campaign_provider_policy": "exact_provider_model_strict_no_fallback",
    }
    get_config.cache_clear()


def test_deepseek_cost_history_is_repriced_from_physical_token_rows(
    monkeypatch, tmp_path
):
    data_dir = tmp_path / "data"
    artifact_path = (
        data_dir
        / "outputs"
        / "runs"
        / "autonomous_sector"
        / "completed"
        / "autonomous_sector_run.json"
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "pipeline_version": "v1",
                "status": "COMPLETED",
                "company_packets": [{"ticker": "AAA"}, {"ticker": "BBB"}],
                "provider_usage": [
                    {
                        "provider": "openai",
                        "model": "gpt-5.4-mini",
                        "input_tokens": 1000,
                        "cached_input_tokens": 500,
                        "output_tokens": 100,
                        "reserved_output_tokens": 0,
                        "cost_estimate_usd": 0.001,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    from app.config import get_config

    get_config.cache_clear()
    samples = driver._artifact_cost_samples(
        provider="deepseek", model="deepseek-v4-pro"
    )
    assert samples == [(0.000522, 2)]
    get_config.cache_clear()


def _attestation(
    cost: float = 0.0,
    *,
    provider: str = "anthropic",
    model: str = "claude-sonnet-test",
) -> dict[str, object]:
    if cost == 0:
        return provider_usage_attestation({})
    return {
        "valid": True,
        "physical_attempt_count": 1,
        "cost_estimate_usd": cost,
        "provider_models": [{"provider": provider, "model": model}],
        "usage_records_sha256": "a" * 64,
    }


def _report(
    *,
    pending: dict[str, list[str]] | None = None,
    unknown: dict[str, list[str]] | None = None,
    loaded: dict[str, list[str]] | None = None,
    eligible: list[str] | None = None,
    fingerprint: str = "membership-fingerprint",
    target_tickers: list[str] | None = None,
) -> dict:
    pending = pending or {}
    unknown = unknown or {}
    loaded = loaded or pending
    cells = []
    pending_cells = []
    loaded_distinct: set[str] = set()
    uncovered_distinct: set[str] = set()
    review_distinct: set[str] = set()
    for band in V1_ATOMIC_BANDS:
        for sector in CANONICAL_SWEEP_SECTORS:
            cell_id = f"{sector}:{band}"
            loaded_tickers = list(loaded.get(cell_id, []))
            tickers = list(pending.get(cell_id, []))
            loaded_distinct.update(loaded_tickers)
            uncovered_distinct.update(tickers)
            review_distinct.update(tickers)
            cell = {
                "cell_id": cell_id,
                "sector": sector,
                "band": band,
                "loaded": len(loaded_tickers),
                "loaded_tickers": loaded_tickers,
                "coverage_uncovered": len(tickers),
                "coverage_uncovered_tickers": tickers,
                "to_review": len(tickers),
                "to_review_tickers": tickers,
                "unknown_cap_tickers": list(unknown.get(cell_id, [])),
                "source_errors": [],
                "gate_error_tickers": [],
            }
            cells.append(cell)
            if tickers:
                pending_cells.append(
                    {
                        "cell_id": cell_id,
                        "sector": sector,
                        "band": band,
                        "coverage_uncovered": len(tickers),
                        "coverage_uncovered_tickers": tickers,
                        "to_review": len(tickers),
                        "to_review_tickers": tickers,
                    }
                )
    eligible_tickers = sorted(set(eligible or loaded_distinct))
    target = sorted(set(target_tickers if target_tickers is not None else eligible_tickers))
    return {
        "schema_version": "classic_full_universe_delta_v1",
        "generated_at": "2026-07-22T12:00:00Z",
        "as_of": "2026-07-22",
        "pipeline_version": "v1",
        "coverage_campaign_id": None,
        "allow_carried_verdicts": True,
        "target_scope": "all_eligible",
        "complete": not pending_cells,
        "status": "COMPLETE" if not pending_cells else "INCOMPLETE",
        "expected_cells": 170,
        "cells_resolved": 170,
        "cells": cells,
        "pending_cells": pending_cells,
        "loader_residual_tickers": [],
        "coverage_residual_tickers": sorted(uncovered_distinct),
        "source_error_cells": [],
        "gate_error_cells": [],
        "newly_synced_cross_check_violations": [],
        "target_ticker_count": len(target),
        "target_tickers_sha256": driver._sha256(target),
        "membership": {
            "current_membership_fingerprint": fingerprint,
            "eligible_common_equity": len(eligible_tickers),
            "eligible_common_equity_tickers": eligible_tickers,
            "membership_unaccounted_tickers": [],
        },
        "distinct_ticker_totals": {
            "coverage_uncovered": len(uncovered_distinct),
            "fresh_review_queue": len(review_distinct),
        },
    }


def _scope_report(report: dict, kwargs: dict) -> dict:
    scoped = copy.deepcopy(report)
    scoped["as_of"] = kwargs.get("as_of")
    scoped["coverage_campaign_id"] = kwargs.get("coverage_campaign_id")
    scoped["allow_carried_verdicts"] = bool(kwargs.get("allow_carried_verdicts", True))
    if kwargs.get("target_tickers") is not None:
        target = sorted(set(kwargs["target_tickers"]))
        scoped["target_scope"] = "frozen_ticker_set"
        scoped["target_ticker_count"] = len(target)
        scoped["target_tickers_sha256"] = driver._sha256(target)
    else:
        scoped["target_scope"] = "all_eligible"
    return scoped


def _patch_dependencies(monkeypatch, report: dict) -> None:
    monkeypatch.setattr(driver, "_configured_provider_binding", _binding)
    monkeypatch.setattr(driver, "_artifact_cost_samples", lambda **kwargs: [])
    monkeypatch.setattr(driver, "backfill_loaded_sets_from_artifacts", lambda: {})
    monkeypatch.setattr(driver, "_quick_membership_fingerprint", lambda: "membership-fingerprint")
    monkeypatch.setattr(
        driver,
        "full_universe_delta_report",
        lambda **kwargs: _scope_report(report, kwargs),
    )
    monkeypatch.setattr(
        driver,
        "_live_uncovered_for_cell",
        lambda _state, cell: list(cell["initial_uncovered_tickers"]),
    )


def _child_result(
    cost: float = 0.1, *, returncode: int = 0, **extra
) -> subprocess.CompletedProcess:
    payload = {
        "status": "COMPLETED" if returncode == 0 else "FAILED",
        "artifact_path": None,
        "provider_usage_attestation": _attestation(cost),
        **extra,
    }
    return subprocess.CompletedProcess([], returncode, stdout=json.dumps(payload), stderr="")


def _terminal_data_gap_payload(
    state: dict,
    *,
    cell_id: str,
    missing_valuation: list[str],
    sparse_history: list[str],
    structural_screened: list[str],
    projected_tickers: list[str] | None = None,
    projected_bands: list[str] | None = None,
) -> dict:
    cell = next(item for item in state["cells"] if item["cell_id"] == cell_id)
    projected_tickers = list(projected_tickers or [])
    projected_bands = list(projected_bands or [])
    zero_attestation = _attestation()
    valuation_dispositions = [
        {
            "ticker": ticker,
            "terminal_state": "NEEDS_DATA",
            "scope_status": "IN_SCOPE",
            "screen_status": "INCOMPLETE",
            "review_status": "NOT_STARTED",
            "underwriting_verdict": None,
            "underwriting_confidence": None,
            "watchlist_eligible": False,
            "reason_codes": ["MISSING_VALUATION"],
            "last_completed_stage": "VALUATION_ANCHOR_PREFLIGHT",
        }
        for ticker in missing_valuation
    ]
    return {
        "status": "FAILED",
        "pipeline_version": "v1",
        "scan_family": "normal",
        "sector": cell["sector"],
        "market_cap_focus": cell["band"],
        "run_id": "autonomous_sector_test_terminal_data_gap",
        "final_verdict": "NO_SELECTION",
        "no_selection_reason": (
            "Deterministic valuation-anchor gate stopped before any sector provider "
            "work: NEEDS_DATA: MISSING_VALUATION across all admitted companies."
        ),
        "degraded_states": ["NEEDS_DATA"],
        "company_packets": len(missing_valuation),
        "missing_valuation_anchor_count": len(missing_valuation),
        "valuation_anchor_count": 0,
        "tool_calls": 0,
        "research_questions": 0,
        "expected_return_scenarios": 0,
        "artifact_path": None,
        "provider_usage_attestation": zero_attestation,
        "provider_usage_incremental_attestation": zero_attestation,
        "candidate_selection": {
            "loaded_tickers": list(cell["loaded_tickers"]),
            "coverage_campaign_id": state["campaign_id"],
            "coverage_target_file": state["target_file"],
            "coverage_expected_provider": state["provider_binding"]["provider"],
            "coverage_expected_model": state["provider_binding"]["model"],
            "financial_history_filter": {
                "status": (
                    "FILTERED_SPARSE_FINANCIAL_HISTORY"
                    if sparse_history
                    else "NO_FILTER_ALL_CANDIDATES_REPORTABLE"
                ),
                "minimum_rows": 3,
                "excluded_tickers": list(sparse_history),
                "year_counts": {ticker: 0 for ticker in sparse_history},
            },
            "valuation_anchor_filter": {
                "status": "NEEDS_DATA_ALL_CANDIDATES_MISSING_VALUATION",
                "input_tickers": list(missing_valuation),
                "ready_tickers": [],
                "needs_data_tickers": list(missing_valuation),
                "reason_code": "MISSING_VALUATION",
                "dispositions": valuation_dispositions,
            },
            "delta_audit": {
                "zero_cost_terminal_coverage": {
                    "tickers": list(structural_screened),
                    "candidate_dispositions": {
                        ticker: "STRUCTURAL_SCREENED" for ticker in structural_screened
                    },
                    "unknown_cap_cross_band_projection": {
                        "tickers": [],
                        "bands": [],
                    },
                }
            },
        },
        "coverage_accounting": {
            "rerun_required": True,
            "unknown_cap_cross_band_projection": {
                "tickers": projected_tickers,
                "bands": projected_bands,
            },
        },
    }


def _attach_terminal_data_gap_attempt(
    summary: dict,
    *,
    output_root: Path,
    payload: dict,
) -> tuple[Path, dict]:
    state_path = Path(summary["state_path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    cell_id = f"{payload['sector']}:{payload['market_cap_focus']}"
    cell = next(item for item in state["cells"] if item["cell_id"] == cell_id)
    attempt_number = len(state["attempts"]) + 1
    attempt = {
        "attempt_number": attempt_number,
        "wave_number": attempt_number,
        "cell_workers": 1,
        "cell_id": cell_id,
        "sector": cell["sector"],
        "band": cell["band"],
        "started_at": "2026-07-22T12:00:00Z",
        "status": "RUNNING",
        "live_uncovered_before": list(cell["initial_uncovered_tickers"]),
        "claimed_tickers_sha256": driver._sha256(cell["initial_uncovered_tickers"]),
        "reserved_cost_usd": 0.0,
        "cost_reconciled": False,
        "argv": driver._child_argv(state, cell, 0.0),
    }
    state["attempts"].append(attempt)
    result = subprocess.CompletedProcess([], 1, stdout=json.dumps(payload), stderr="")
    interrupted = driver._settle_child_outcome(
        state=state,
        attempt=attempt,
        outcome=result,
        campaign_dir=output_root / state["campaign_id"],
        ceiling=1.0,
    )
    assert interrupted is None
    assert attempt["status"] == "SAFE_RETRY_REQUIRED"
    state["status"] = "INCOMPLETE_RETRY_REQUIRED"
    state["stop_reason"] = attempt["stop_reason"]
    state["updated_at"] = "2026-07-22T12:01:00Z"
    driver._atomic_write_json(state_path, state)
    driver._validate_state_integrity(state)
    return state_path, state


def _cell_id_from_argv(argv: list[str]) -> str:
    sector = argv[argv.index("--sector") + 1]
    band = argv[argv.index("--market-cap-focus") + 1]
    return f"{sector}:{band}"


def _plan_dynamic_campaign(
    monkeypatch,
    output_root: Path,
    pending: dict[str, list[str]],
    completed: set[str],
) -> dict:
    eligible = sorted({ticker for tickers in pending.values() for ticker in tickers})
    initial = _report(pending=pending, loaded=pending, eligible=eligible)
    _patch_dependencies(monkeypatch, initial)
    summary = driver.plan_campaign(
        mode="full-rescan",
        populate_watchlist=False,
        output_root=output_root,
    )
    campaign_dir = output_root / summary["campaign_id"]
    (campaign_dir / ".campaign.lock").touch(mode=0o600, exist_ok=True)

    def dynamic_report(**kwargs):
        remaining = {
            cell_id: list(tickers)
            for cell_id, tickers in pending.items()
            if cell_id not in completed
        }
        return _scope_report(
            _report(
                pending=remaining,
                loaded=pending,
                eligible=eligible,
                target_tickers=eligible,
            ),
            kwargs,
        )

    monkeypatch.setattr(driver, "full_universe_delta_report", dynamic_report)
    monkeypatch.setattr(
        driver,
        "_live_uncovered_for_cell",
        lambda _state, cell: (
            [] if str(cell["cell_id"]) in completed else list(cell["initial_uncovered_tickers"])
        ),
    )
    return summary


def test_missed_plan_freezes_only_lifetime_gaps_and_requires_fresh_campaign(monkeypatch, tmp_path):
    report = _report(
        pending={"energy:micro_cap": ["MISS"]},
        loaded={"energy:micro_cap": ["DONE", "MISS"]},
        eligible=["DONE", "MISS"],
    )
    _patch_dependencies(monkeypatch, report)

    summary = driver.plan_campaign(mode="missed-only", as_of="2026-07-22", output_root=tmp_path)
    state = json.loads(Path(summary["state_path"]).read_text())
    target = json.loads(Path(summary["target_file"]).read_text())

    assert summary["status"] == "PLANNED"
    assert summary["coverage_campaign_id"] == summary["campaign_id"]
    assert summary["allow_carried_verdicts"] is False
    assert summary["effective_as_of"] == "2026-07-22"
    assert target["tickers"] == ["MISS"]
    assert state["target_ticker_count"] == 1


def test_resume_plan_reuses_prior_target_and_excludes_only_terminal_rows(
    monkeypatch, tmp_path
):
    report = _report(
        pending={
            "energy:micro_cap": ["MISS"],
            "energy:small_cap": ["PARTIAL"],
        },
        loaded={
            "energy:micro_cap": ["DONE", "MISS", "NEW", "PARTIAL"],
            "energy:small_cap": ["PARTIAL"],
        },
        eligible=["DONE", "MISS", "NEW", "PARTIAL"],
    )
    _patch_dependencies(monkeypatch, report)
    prior_id = "classic_missed_20260722T120000Z_deadbeef"
    prior_state = {
        "campaign_id": prior_id,
        "coverage_campaign_id": prior_id,
        "mode": "missed-only",
        "effective_as_of": "2026-07-22",
        "plan_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        driver,
        "_load_state",
        lambda campaign_id, output_root=None: (Path("prior.json"), prior_state),
    )
    monkeypatch.setattr(
        driver,
        "_target_tickers",
        lambda state: ["DONE", "MISS", "OLD", "PARTIAL"],
    )

    summary = driver.plan_campaign(
        mode="missed-only",
        as_of="2026-07-22",
        resume_from_campaign_id=prior_id,
        output_root=tmp_path,
    )
    state = json.loads(Path(summary["state_path"]).read_text())
    target = json.loads(Path(summary["target_file"]).read_text())

    assert target["tickers"] == ["MISS", "PARTIAL"]
    assert summary["resume_contract"] == {
        "campaign_id": prior_id,
        "plan_sha256": "b" * 64,
        "target_ticker_count": 4,
        "target_tickers_sha256": driver._sha256(["DONE", "MISS", "OLD", "PARTIAL"]),
        "current_scope_ticker_count": 3,
        "current_scope_tickers_sha256": driver._sha256(["DONE", "MISS", "PARTIAL"]),
        "dropped_current_ineligible_ticker_count": 1,
        "dropped_current_ineligible_tickers_sha256": driver._sha256(["OLD"]),
        "completed_ticker_count": 1,
        "completed_tickers_sha256": driver._sha256(["DONE"]),
    }
    state["resume_contract"]["completed_ticker_count"] = 2
    with pytest.raises(driver.ClassicScanPlanError, match="plan hash"):
        driver._validate_state_integrity(state)


def test_resume_completion_includes_loader_residual_and_rejects_scope_drift(monkeypatch):
    prior_id = "classic_missed_20260722T120000Z_deadbeef"
    prior_state = {
        "campaign_id": prior_id,
        "coverage_campaign_id": prior_id,
        "effective_as_of": "2026-07-22",
    }
    target = ["DONE", "LOADER"]
    report = _scope_report(
        _report(eligible=target, target_tickers=target),
        {
            "coverage_campaign_id": prior_id,
            "allow_carried_verdicts": False,
            "target_tickers": target,
        },
    )
    report["loader_residual_tickers"] = ["LOADER"]
    monkeypatch.setattr(driver, "_current_report", lambda *args, **kwargs: report)

    assert driver._completed_tickers_for_campaign(
        prior_state, target_tickers=target
    ) == ["DONE"]

    report["target_tickers_sha256"] = "0" * 64
    with pytest.raises(driver.ClassicScanPlanError, match="report contract"):
        driver._completed_tickers_for_campaign(prior_state, target_tickers=target)


def test_resume_plan_ignores_lifetime_loader_residual_outside_frozen_target(monkeypatch, tmp_path):
    lifetime = _report(eligible=["FROZEN", "POST_AS_OF"])
    lifetime["loader_residual_tickers"] = ["POST_AS_OF"]
    target_report = _report(
        pending={"energy:micro_cap": ["FROZEN"]},
        loaded={"energy:micro_cap": ["FROZEN"]},
        eligible=["FROZEN", "POST_AS_OF"],
        target_tickers=["FROZEN"],
    )
    _patch_dependencies(monkeypatch, lifetime)
    prior_id = "classic_missed_20260722T120000Z_deadbeef"
    prior_state = {
        "campaign_id": prior_id,
        "coverage_campaign_id": prior_id,
        "mode": "missed-only",
        "effective_as_of": "2026-07-22",
        "plan_sha256": "b" * 64,
        "status": "INCOMPLETE_RETRY_REQUIRED",
        "unresolved_cost_reservation_usd": 0.0,
        "attempts": [],
    }
    original_load_state = driver._load_state
    monkeypatch.setattr(
        driver,
        "_load_state",
        lambda campaign_id, output_root=None: (Path("prior.json"), prior_state),
    )
    monkeypatch.setattr(driver, "_target_tickers", lambda state: ["FROZEN"])

    def dynamic_report(**kwargs):
        base = lifetime if kwargs.get("coverage_campaign_id") is None else target_report
        return _scope_report(base, kwargs)

    monkeypatch.setattr(driver, "full_universe_delta_report", dynamic_report)
    summary = driver.plan_campaign(
        mode="missed-only",
        as_of="2026-07-22",
        resume_from_campaign_id=prior_id,
        output_root=tmp_path,
    )
    state = json.loads(Path(summary["state_path"]).read_text())
    lifetime_evidence, target_evidence = state["preflight_evidence"]["reports"]

    assert summary["status"] == "PLANNED"
    assert summary["preflight_blockers"] == []
    assert state["preflight_evidence"]["loader_residual_policy"] == (
        "frozen_missed_only_target_scope"
    )
    assert lifetime_evidence["observed_blockers"] == ["LOADER_RESIDUAL"]
    assert lifetime_evidence["effective_blockers"] == []
    assert lifetime_evidence["ignored_blockers"] == ["LOADER_RESIDUAL"]
    assert lifetime_evidence["loader_residual"] == {
        "applied_to_gate": False,
        "ticker_count": 1,
        "tickers": ["POST_AS_OF"],
        "tickers_sha256": "ea61ba96c7a1a713af986270ada945e29d01bc8052e87b1523b9fa0a2c5993ac",
    }
    assert target_evidence["observed_blockers"] == []
    assert target_evidence["loader_residual"] == {
        "applied_to_gate": True,
        "ticker_count": 0,
        "tickers": [],
        "tickers_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    }

    monkeypatch.setattr(driver, "_load_state", original_load_state)
    status = driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)
    assert status["preflight_evidence"] == state["preflight_evidence"]


def test_resume_plan_blocks_and_persists_in_target_loader_residual(monkeypatch, tmp_path):
    target = ["FROZEN", "IN_TARGET"]
    lifetime = _report(eligible=target)
    source_report = _report(
        pending={"energy:micro_cap": target},
        loaded={"energy:micro_cap": target},
        eligible=target,
        target_tickers=target,
    )
    child_report = _report(
        pending={"energy:micro_cap": ["FROZEN"]},
        loaded={"energy:micro_cap": ["FROZEN"]},
        eligible=target,
        target_tickers=target,
    )
    child_report["loader_residual_tickers"] = ["IN_TARGET"]
    _patch_dependencies(monkeypatch, lifetime)
    prior_id = "classic_missed_20260722T120000Z_deadbeef"
    prior_state = {
        "campaign_id": prior_id,
        "coverage_campaign_id": prior_id,
        "mode": "missed-only",
        "effective_as_of": "2026-07-22",
        "plan_sha256": "b" * 64,
        "status": "INCOMPLETE_RETRY_REQUIRED",
        "unresolved_cost_reservation_usd": 0.0,
        "attempts": [],
    }
    original_load_state = driver._load_state
    monkeypatch.setattr(
        driver,
        "_load_state",
        lambda campaign_id, output_root=None: (Path("prior.json"), prior_state),
    )
    monkeypatch.setattr(driver, "_target_tickers", lambda state: target)

    def dynamic_report(**kwargs):
        coverage_id = kwargs.get("coverage_campaign_id")
        if coverage_id is None:
            base = lifetime
        elif coverage_id == prior_id:
            base = source_report
        else:
            base = child_report
        return _scope_report(base, kwargs)

    monkeypatch.setattr(driver, "full_universe_delta_report", dynamic_report)
    summary = driver.plan_campaign(
        mode="missed-only",
        as_of="2026-07-22",
        resume_from_campaign_id=prior_id,
        output_root=tmp_path,
    )
    state_path = Path(summary["state_path"])
    state = json.loads(state_path.read_text())
    target_evidence = state["preflight_evidence"]["reports"][1]

    assert summary["status"] == "BLOCKED"
    assert summary["stop_reason"] == "LOADER_RESIDUAL"
    assert summary["preflight_blockers"] == ["LOADER_RESIDUAL"]
    assert target_evidence["observed_blockers"] == ["LOADER_RESIDUAL"]
    assert target_evidence["effective_blockers"] == ["LOADER_RESIDUAL"]
    assert target_evidence["ignored_blockers"] == []
    assert target_evidence["loader_residual"] == {
        "applied_to_gate": True,
        "ticker_count": 1,
        "tickers": ["IN_TARGET"],
        "tickers_sha256": "05a2b2dfcd9652364c4f7467fb20eb580705d5ee70390d0f292a43ac6745b3f0",
    }
    assert target_evidence["blocker_inputs"]["LOADER_RESIDUAL"] == {
        "ticker_count": 1,
        "tickers": ["IN_TARGET"],
        "tickers_sha256": "05a2b2dfcd9652364c4f7467fb20eb580705d5ee70390d0f292a43ac6745b3f0",
    }
    assert target_evidence["report_identity"]["coverage_campaign_id"] == (summary["campaign_id"])
    assert target_evidence["report_identity"]["target_tickers_sha256"] == (driver._sha256(target))

    monkeypatch.setattr(driver, "_load_state", original_load_state)
    status = driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)
    assert status["preflight_evidence"] == state["preflight_evidence"]

    state["preflight_evidence"]["reports"][1]["loader_residual"]["ticker_count"] = 2
    with pytest.raises(driver.ClassicScanPlanError, match="ticker count"):
        driver._validate_state_integrity(state)


def test_full_rescan_retains_lifetime_loader_residual_gate(monkeypatch, tmp_path):
    target = ["LOADED", "MISSING"]
    lifetime = _report(eligible=target)
    lifetime["loader_residual_tickers"] = ["MISSING"]
    target_report = _report(eligible=target, target_tickers=target)
    _patch_dependencies(monkeypatch, lifetime)

    def dynamic_report(**kwargs):
        base = lifetime if kwargs.get("coverage_campaign_id") is None else target_report
        return _scope_report(base, kwargs)

    monkeypatch.setattr(driver, "full_universe_delta_report", dynamic_report)
    summary = driver.plan_campaign(mode="full-rescan", as_of="2026-07-22", output_root=tmp_path)
    state = json.loads(Path(summary["state_path"]).read_text())
    lifetime_evidence, target_evidence = state["preflight_evidence"]["reports"]

    assert summary["status"] == "BLOCKED"
    assert summary["stop_reason"] == "LOADER_RESIDUAL"
    assert summary["preflight_blockers"] == ["LOADER_RESIDUAL"]
    assert state["preflight_evidence"]["loader_residual_policy"] == (
        "full_universe_lifetime_and_target"
    )
    assert lifetime_evidence["effective_blockers"] == ["LOADER_RESIDUAL"]
    assert lifetime_evidence["ignored_blockers"] == []
    assert lifetime_evidence["loader_residual"] == {
        "applied_to_gate": True,
        "ticker_count": 1,
        "tickers": ["MISSING"],
        "tickers_sha256": "b5688483ba5f9215a707ee77583433cd0a6ea16f78af991b8fdcb701a35f32fb",
    }
    assert target_evidence["effective_blockers"] == []


def test_provider_not_ready_is_persisted_as_exact_preflight_evidence(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    disabled_binding = {
        "provider": "disabled",
        "model": "disabled",
        "credential_present": False,
        "ready": False,
        "max_output_tokens": None,
        "campaign_provider_policy": "exact_provider_model_strict_no_fallback",
    }
    monkeypatch.setattr(
        driver,
        "_configured_provider_binding",
        lambda: copy.deepcopy(disabled_binding),
    )

    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    state = json.loads(Path(summary["state_path"]).read_text())

    assert summary["status"] == "BLOCKED"
    assert summary["stop_reason"] == "LLM_PROVIDER_NOT_READY"
    assert summary["preflight_blockers"] == ["LLM_PROVIDER_NOT_READY"]
    assert state["preflight_evidence"]["provider_binding_evaluation"] == {
        "effective_blockers": ["LLM_PROVIDER_NOT_READY"],
        "inputs": disabled_binding,
    }
    status = driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)
    assert status["preflight_evidence"] == state["preflight_evidence"]


def test_resume_plan_rejects_unresolved_source_reservation(monkeypatch, tmp_path):
    report = _report(eligible=["MISS"])
    _patch_dependencies(monkeypatch, report)
    prior_id = "classic_missed_20260722T120000Z_deadbeef"
    prior_state = {
        "campaign_id": prior_id,
        "coverage_campaign_id": prior_id,
        "mode": "missed-only",
        "effective_as_of": "2026-07-22",
        "status": "INCOMPLETE_RETRY_REQUIRED",
        "unresolved_cost_reservation_usd": 0.25,
    }
    monkeypatch.setattr(
        driver,
        "_load_state",
        lambda campaign_id, output_root=None: (Path("prior.json"), prior_state),
    )

    with pytest.raises(driver.ClassicScanPlanError, match="unresolved paid work"):
        driver.plan_campaign(
            mode="missed-only",
            as_of="2026-07-22",
            resume_from_campaign_id=prior_id,
            output_root=tmp_path,
        )


def test_resume_plan_rejects_full_rescan(monkeypatch, tmp_path):
    report = _report(eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)

    with pytest.raises(ValueError, match="requires --mode missed-only"):
        driver.plan_campaign(
            mode="full-rescan",
            resume_from_campaign_id="classic_missed_20260722T120000Z_deadbeef",
            output_root=tmp_path,
        )


def test_full_plan_freezes_all_eligible_and_dedupes_unknown_cap_cost(
    monkeypatch, tmp_path
):
    pending = {f"energy:{band}": ["UNKNOWN"] for band in V1_ATOMIC_BANDS}
    report = _report(pending=pending, unknown=pending, eligible=["UNKNOWN"])
    _patch_dependencies(monkeypatch, report)

    summary = driver.plan_campaign(mode="full-rescan", output_root=tmp_path)
    target = json.loads(Path(summary["target_file"]).read_text())

    assert target["tickers"] == ["UNKNOWN"]
    assert summary["cost_estimate"]["review_occurrences"] == 5
    assert summary["cost_estimate"]["payable_review_units"] == 1
    assert summary["cost_estimate"]["unknown_cap_duplicate_reviews_avoided"] == 4


def test_plan_hash_binds_scope_carries_watchlist_and_target(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="full-rescan", output_root=tmp_path)
    state_path = Path(summary["state_path"])

    for field, value in (
        ("coverage_campaign_id", "classic_full_20260722T120000Z_deadbeef"),
        ("allow_carried_verdicts", True),
        ("populate_watchlist", False),
        ("target_ticker_count", 2),
    ):
        state = json.loads(state_path.read_text())
        state[field] = value
        with pytest.raises(driver.ClassicScanPlanError):
            driver._validate_state_integrity(state)


def test_v4_requires_integrity_bound_preflight_evidence(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    state_path = Path(summary["state_path"])
    state = json.loads(state_path.read_text())

    missing = copy.deepcopy(state)
    del missing["preflight_evidence"]
    missing["plan_sha256"] = driver._sha256(driver._plan_contract(missing))
    with pytest.raises(driver.ClassicScanPlanError, match="preflight evidence"):
        driver._validate_state_integrity(missing)

    malformed = copy.deepcopy(state)
    malformed["preflight_evidence"]["reports"] = [0, 1]
    with pytest.raises(driver.ClassicScanPlanError, match="report roles"):
        driver._validate_state_integrity(malformed)

    plan_hash_tamper = copy.deepcopy(state)
    plan_hash_tamper["preflight_evidence"]["reports"][0]["report_identity"]["generated_at"] = (
        "2026-07-22T12:00:01Z"
    )
    with pytest.raises(driver.ClassicScanPlanError, match="plan hash"):
        driver._validate_state_integrity(plan_hash_tamper)


def test_fixed_v3_campaign_loads_and_resumes_into_v4_child(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    monkeypatch.setattr(driver, "_utc_now", lambda: "2026-07-22T12:00:00Z")
    monkeypatch.setattr(
        driver,
        "_source_contract",
        lambda: {
            "schema_version": "classic_scan_source_contract_test",
            "sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(driver, "uuid4", lambda: SimpleNamespace(hex="11111111"))
    source = driver.plan_campaign(
        mode="missed-only", as_of="2026-08-07", output_root=tmp_path
    )
    source_path = Path(source["state_path"])
    legacy = json.loads(source_path.read_text())
    legacy["schema_version"] = "classic_scan_campaign_v3"
    del legacy["preflight_evidence"]
    del legacy["preflight_blockers"]
    legacy["plan_sha256"] = "23fcd120e48a35771c3c22e6103e5269145a8f4c90088bfbddf3067d5d4de727"
    source_path.write_text(json.dumps(legacy), encoding="utf-8")

    status = driver.campaign_status(campaign_id=source["campaign_id"], output_root=tmp_path)
    assert status["schema_version"] == "classic_scan_campaign_v3"
    assert status["preflight_blockers"] == []
    assert status["preflight_evidence"] is None

    monkeypatch.setattr(driver, "uuid4", lambda: SimpleNamespace(hex="22222222"))
    child = driver.plan_campaign(
        mode="missed-only",
        as_of="2026-08-07",
        resume_from_campaign_id=source["campaign_id"],
        output_root=tmp_path,
    )
    assert child["schema_version"] == "classic_scan_campaign_v4"
    assert child["resume_contract"]["campaign_id"] == source["campaign_id"]
    assert child["resume_contract"]["plan_sha256"] == (
        "23fcd120e48a35771c3c22e6103e5269145a8f4c90088bfbddf3067d5d4de727"
    )
    assert child["target_ticker_count"] == 1


def test_runtime_preflight_blocker_persists_exact_inputs_before_provider_dispatch(
    monkeypatch, tmp_path
):
    planned = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    blocked = _report(eligible=["AAA"], target_tickers=["AAA"])
    blocked["loader_residual_tickers"] = ["AAA"]
    _patch_dependencies(monkeypatch, planned)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        driver,
        "full_universe_delta_report",
        lambda **kwargs: _scope_report(blocked, kwargs),
    )

    with pytest.raises(driver.ClassicScanPlanError, match="current manifest is blocked"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            command_runner=lambda *args, **kwargs: calls.append("provider-called"),
        )

    status = driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)
    failure = status["latest_preflight_failure"]
    assert calls == []
    assert status["status"] == "STALE_PLAN"
    assert status["authorized_max_cost_usd"] is None
    assert failure["error"] == "current manifest is blocked: LOADER_RESIDUAL"
    assert failure["report"]["observed_blockers"] == ["LOADER_RESIDUAL"]
    assert failure["report"]["loader_residual"] == {
        "applied_to_gate": True,
        "ticker_count": 1,
        "tickers": ["AAA"],
        "tickers_sha256": "48e6f6a04a3a679b4e2bdb382af448541fff6a9ae6b13a5ea16146559adaa3f5",
    }
    persisted = json.loads(Path(summary["state_path"]).read_text())
    assert len(failure["record_sha256"]) == 64

    report_tamper = copy.deepcopy(persisted)
    report_tamper["latest_report"]["generated_at"] = "2026-07-22T12:00:01Z"
    Path(summary["state_path"]).write_text(json.dumps(report_tamper), encoding="utf-8")
    with pytest.raises(driver.ClassicScanPlanError, match="failure report"):
        driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)

    record_tamper = copy.deepcopy(persisted)
    record_tamper["latest_preflight_failure"]["error"] = "tampered"
    Path(summary["state_path"]).write_text(json.dumps(record_tamper), encoding="utf-8")
    with pytest.raises(driver.ClassicScanPlanError, match="evidence hash"):
        driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)


def test_resumed_campaign_persists_manifest_blocker_before_provider_dispatch(monkeypatch, tmp_path):
    planned = _report(
        pending={"energy:micro_cap": ["AAA", "BBB"]},
        eligible=["AAA", "BBB"],
    )
    blocked = _report(
        pending={"energy:micro_cap": ["AAA"]},
        loaded={"energy:micro_cap": ["AAA"]},
        eligible=["AAA", "BBB"],
        target_tickers=["AAA", "BBB"],
    )
    blocked["loader_residual_tickers"] = ["BBB"]
    _patch_dependencies(monkeypatch, planned)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    state_path = Path(summary["state_path"])
    state = json.loads(state_path.read_text())
    state["status"] = "INCOMPLETE_RETRY_REQUIRED"
    state["stop_reason"] = "CELL_RETRY_REQUIRED:energy:micro_cap"
    state["cumulative_cost_usd"] = 0.1
    state["attempts"] = [
        {
            "attempt_number": 1,
            "cell_id": "energy:micro_cap",
            "sector": "energy",
            "band": "micro_cap",
            "status": "OK",
            "live_uncovered_before": ["AAA"],
            "reserved_cost_usd": 0.1,
            "charged_cost_usd": 0.1,
            "cost_reconciled": True,
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        driver,
        "full_universe_delta_report",
        lambda **kwargs: _scope_report(blocked, kwargs),
    )

    with pytest.raises(driver.ClassicScanPlanError, match="current manifest is blocked"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            command_runner=lambda *args, **kwargs: calls.append("provider-called"),
        )

    status = driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)
    failure = status["latest_preflight_failure"]
    assert calls == []
    assert status["status"] == "STALE_PLAN"
    assert status["attempt_count"] == 1
    assert status["authorized_max_cost_usd"] is None
    assert failure["report"]["loader_residual"] == {
        "applied_to_gate": True,
        "ticker_count": 1,
        "tickers": ["BBB"],
        "tickers_sha256": "5c94ace24677f0e750ece39788f4c5b0559a3b64d1ff605eda2e9784146cc3a6",
    }


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("target", "live target scope does not match frozen target"),
        ("cell", "candidate membership changed for energy:micro_cap before spend"),
    ),
)
def test_resumed_campaign_persists_frozen_scope_drift_before_provider_dispatch(
    monkeypatch, tmp_path, drift, message
):
    planned = _report(
        pending={"energy:micro_cap": ["AAA", "BBB"]},
        eligible=["AAA", "BBB"],
    )
    _patch_dependencies(monkeypatch, planned)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    state_path = Path(summary["state_path"])
    state = json.loads(state_path.read_text())
    state["status"] = "INCOMPLETE_RETRY_REQUIRED"
    state["stop_reason"] = "CELL_RETRY_REQUIRED:energy:micro_cap"
    state["cumulative_cost_usd"] = 0.1
    state["attempts"] = [
        {
            "attempt_number": 1,
            "cell_id": "energy:micro_cap",
            "sector": "energy",
            "band": "micro_cap",
            "status": "OK",
            "live_uncovered_before": ["AAA"],
            "reserved_cost_usd": 0.1,
            "charged_cost_usd": 0.1,
            "cost_reconciled": True,
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    calls: list[str] = []

    def drifted_report(**kwargs):
        if drift == "cell":
            base = _report(
                pending={"energy:micro_cap": ["AAA"]},
                loaded={"energy:micro_cap": ["AAA"]},
                eligible=["AAA", "BBB"],
                target_tickers=["AAA", "BBB"],
            )
        else:
            base = planned
        current = _scope_report(base, kwargs)
        if drift == "target":
            current["target_tickers_sha256"] = "0" * 64
        return current

    monkeypatch.setattr(driver, "full_universe_delta_report", drifted_report)

    with pytest.raises(driver.ClassicScanPlanError, match=message):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            command_runner=lambda *args, **kwargs: calls.append("provider-called"),
        )

    status = driver.campaign_status(
        campaign_id=summary["campaign_id"], output_root=tmp_path
    )
    assert calls == []
    assert status["status"] == "STALE_PLAN"
    assert status["stop_reason"] == message
    assert status["attempt_count"] == 1
    assert status["authorized_max_cost_usd"] is None
    assert status["latest_preflight_failure"]["error"] == message


def test_run_is_serial_campaign_scoped_targeted_and_recomputes_only_at_boundaries(
    monkeypatch, tmp_path
):
    pending_report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    complete_report = _report(loaded={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, pending_report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    done = {"value": False}
    report_calls = {"count": 0}
    calls: list[list[str]] = []

    def fake_report(**kwargs):
        report_calls["count"] += 1
        base = complete_report if done["value"] else pending_report
        return _scope_report(base, kwargs)

    def fake_runner(argv, **kwargs):
        calls.append(list(argv))
        done["value"] = True
        return _child_result(0.1)

    monkeypatch.setattr(driver, "full_universe_delta_report", fake_report)
    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=fake_runner,
    )

    assert result["status"] == "COMPLETE"
    assert result["cumulative_cost_usd"] == 0.1
    assert report_calls["count"] == 2
    assert len(calls) == 1
    argv = calls[0]
    assert argv[argv.index("--coverage-campaign-id") + 1] == summary["campaign_id"]
    assert argv[argv.index("--coverage-target-file") + 1] == summary["target_file"]
    assert argv[argv.index("--as-of") + 1] == summary["effective_as_of"]
    assert "--no-carry-prior-verdicts" in argv
    assert "--strict-cost-cap" in argv
    assert "--max-candidates" not in argv


def test_campaign_charges_incremental_checkpoint_usage_but_attests_total(
    monkeypatch,
    tmp_path,
):
    pending_report = _report(
        pending={"energy:micro_cap": ["AAA"]},
        eligible=["AAA"],
    )
    complete_report = _report(
        loaded={"energy:micro_cap": ["AAA"]},
        eligible=["AAA"],
    )
    _patch_dependencies(monkeypatch, pending_report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    done = False
    provider_usage = [
        {
            "provider": "anthropic",
            "model": "claude-sonnet-test",
            "schema_name": "autonomous_sector_candidate_memo_aaa",
            "status": "OK",
            "cost_estimate_usd": 0.08,
            "reused_from_checkpoint": True,
            "checkpoint_physical_id": "1" * 64,
        },
        {
            "provider": "anthropic",
            "model": "claude-sonnet-test",
            "schema_name": "autonomous_sector_candidate_memo_bbb",
            "status": "OK",
            "cost_estimate_usd": 0.03,
            "reused_from_checkpoint": False,
            "checkpoint_physical_id": "2" * 64,
        },
    ]
    artifact_path = tmp_path / "child_artifact.json"
    artifact_path.write_text(
        json.dumps({"provider_usage": provider_usage}),
        encoding="utf-8",
    )

    def report_after_child(**kwargs):
        return _scope_report(complete_report if done else pending_report, kwargs)

    def runner(_argv, **_kwargs):
        nonlocal done
        done = True
        artifact = {"provider_usage": provider_usage}
        payload = {
            "status": "COMPLETED",
            "artifact_path": str(artifact_path),
            "provider_usage_attestation": provider_usage_attestation(artifact),
            "provider_usage_incremental_attestation": (
                provider_usage_attestation(
                    artifact,
                    include_reused=False,
                )
            ),
        }
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(driver, "full_universe_delta_report", report_after_child)
    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=runner,
    )

    state = json.loads(Path(summary["state_path"]).read_text())
    attempt = state["attempts"][0]
    assert result["status"] == "COMPLETE"
    assert result["cumulative_cost_usd"] == pytest.approx(0.03)
    assert attempt["charged_cost_usd"] == pytest.approx(0.03)
    assert attempt["provider_usage_attestation"]["cost_estimate_usd"] == (pytest.approx(0.03))
    assert attempt["provider_usage_total_attestation"]["cost_estimate_usd"] == pytest.approx(0.11)
    assert attempt["provider_usage_attestation"]["physical_attempt_count"] == 1
    assert attempt["provider_usage_total_attestation"]["physical_attempt_count"] == 2


@pytest.mark.parametrize("value", [0, 5, True, "2.0", "many"])
def test_cell_worker_bounds_fail_closed(value):
    with pytest.raises(ValueError, match="integer from 1 through 4"):
        driver._resolve_cell_workers(value)


def test_cell_worker_env_defaults_to_one_and_accepts_bounded_override(monkeypatch):
    monkeypatch.delenv("VOE_CLASSIC_CELL_WORKERS", raising=False)
    assert driver._resolve_cell_workers(None) == 1
    monkeypatch.setenv("VOE_CLASSIC_CELL_WORKERS", "3")
    assert driver._resolve_cell_workers(None) == 3


def test_parallel_campaign_rejects_watchlist_mutation_before_dispatch(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(
        mode="missed-only",
        populate_watchlist=True,
        output_root=tmp_path,
    )
    calls: list[str] = []

    with pytest.raises(driver.ClassicScanAuthorizationError, match="--no-watchlist"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            cell_workers=2,
            command_runner=lambda *args, **kwargs: calls.append("called"),
        )

    state = json.loads(Path(summary["state_path"]).read_text())
    assert calls == []
    assert state["attempts"] == []
    assert state["authorization_history"] == []


@pytest.mark.slow  # wall-clock speed-up assertion; unreliable on shared CI runners
def test_three_workers_reserve_before_dispatch_speed_up_and_preserve_sqlite_rows(
    monkeypatch, tmp_path
):
    cells = [f"{sector}:{band}" for band in V1_ATOMIC_BANDS for sector in CANONICAL_SWEEP_SECTORS][
        :9
    ]
    pending = {cell_id: [f"T{index:02d}"] for index, cell_id in enumerate(cells)}

    def execute(workers: int, output_root: Path):
        data_dir = output_root / "data"
        monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
        monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
        get_config.cache_clear()
        init_db()
        with get_db() as conn:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(sector_run_loaded_sets)")
            }
        assert {
            "run_id",
            "sector",
            "market_cap_focus",
            "ticker",
            "candidate_disposition",
            "coverage_campaign_id",
            "coverage_complete",
        }.issubset(columns)
        completed: set[str] = set()
        summary = _plan_dynamic_campaign(
            monkeypatch,
            output_root,
            pending,
            completed,
        )
        lock = threading.Lock()
        active = 0
        peak_active = 0
        snapshots: list[dict] = []

        def runner(argv, **kwargs):
            nonlocal active, peak_active
            cell_id = _cell_id_from_argv(list(argv))
            state = json.loads(Path(summary["state_path"]).read_text())
            with lock:
                snapshots.append({"cell_id": cell_id, "state": state})
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.18)
            sector, band = cell_id.split(":", 1)
            ticker = pending[cell_id][0]
            with get_db() as conn:
                inserted = record_loaded_set(
                    conn,
                    run_id=f"phase-a-{cell_id}",
                    sector=sector,
                    market_cap_focus=band,
                    source="phase_a_concurrency_test",
                    tickers=[ticker],
                    pipeline_version="v1",
                    candidate_dispositions={ticker: "LLM_CANDIDATE_REVIEW_COMPLETED"},
                    coverage_campaign_id=summary["campaign_id"],
                )
            assert inserted == 1
            with lock:
                completed.add(cell_id)
                active -= 1
            return _child_result(0.01)

        started = time.perf_counter()
        result = driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=output_root,
            cell_workers=workers,
            command_runner=runner,
        )
        elapsed = time.perf_counter() - started
        with get_db() as conn:
            persisted = {
                f"{row['sector']}:{row['market_cap_focus']}"
                for row in conn.execute(
                    "SELECT sector, market_cap_focus "
                    "FROM sector_run_loaded_sets WHERE coverage_campaign_id = ?",
                    (summary["campaign_id"],),
                ).fetchall()
            }
        return result, elapsed, completed, peak_active, snapshots, persisted

    serial = execute(1, tmp_path / "serial")
    parallel = execute(3, tmp_path / "parallel")

    assert serial[0]["status"] == parallel[0]["status"] == "COMPLETE"
    assert serial[2] == parallel[2] == set(cells)
    assert serial[5] == parallel[5] == set(cells)
    assert serial[3] == 1
    assert parallel[3] == 3
    assert serial[1] / parallel[1] >= 2.2
    for snapshot in parallel[4]:
        state = snapshot["state"]
        assert (
            float(state["cumulative_cost_usd"]) + float(state["unresolved_cost_reservation_usd"])
            <= 10.0
        )
        current_attempt = next(
            attempt for attempt in state["attempts"] if attempt["cell_id"] == snapshot["cell_id"]
        )
        wave_attempts = [
            attempt
            for attempt in state["attempts"]
            if attempt.get("wave_number") == current_attempt["wave_number"]
        ]
        assert len(wave_attempts) == 3
        assert all(not attempt["cost_reconciled"] for attempt in wave_attempts)
    parallel_state = json.loads(Path(parallel[0]["state_path"]).read_text())
    assert [attempt["cell_id"] for attempt in parallel_state["attempts"]] == cells
    assert all(attempt["cell_workers"] == 3 for attempt in parallel_state["attempts"])


def test_unknown_cap_overlap_is_claimed_once_across_parallel_bands(monkeypatch, tmp_path):
    pending = {f"energy:{band}": ["UNKNOWN"] for band in V1_ATOMIC_BANDS}
    initial = _report(
        pending=pending,
        unknown=pending,
        loaded=pending,
        eligible=["UNKNOWN"],
    )
    _patch_dependencies(monkeypatch, initial)
    summary = driver.plan_campaign(
        mode="full-rescan",
        populate_watchlist=False,
        output_root=tmp_path,
    )
    covered = {"value": False}
    calls: list[str] = []

    def dynamic_report(**kwargs):
        report = _report(
            pending={} if covered["value"] else pending,
            unknown={} if covered["value"] else pending,
            loaded=pending,
            eligible=["UNKNOWN"],
            target_tickers=["UNKNOWN"],
        )
        return _scope_report(report, kwargs)

    monkeypatch.setattr(driver, "full_universe_delta_report", dynamic_report)
    monkeypatch.setattr(
        driver,
        "_live_uncovered_for_cell",
        lambda _state, cell: [] if covered["value"] else ["UNKNOWN"],
    )

    def runner(argv, **kwargs):
        calls.append(_cell_id_from_argv(list(argv)))
        covered["value"] = True
        return _child_result(0.05)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        cell_workers=4,
        command_runner=runner,
    )

    assert result["status"] == "COMPLETE"
    assert calls == ["energy:micro_cap"]
    assert result["known_physical_provider_attempts"] == 1


def test_parallel_aggregate_reservation_never_exceeds_owner_ceiling(monkeypatch, tmp_path):
    cells = [f"{sector}:micro_cap" for sector in CANONICAL_SWEEP_SECTORS[:3]]
    pending = {cell_id: [f"B{index}"] for index, cell_id in enumerate(cells)}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)
    backfill_calls = 0

    def flaky_final_backfill():
        nonlocal backfill_calls
        backfill_calls += 1
        if backfill_calls == 1:
            return {}
        raise RuntimeError("diagnostic refresh unavailable")

    monkeypatch.setattr(driver, "backfill_loaded_sets_from_artifacts", flaky_final_backfill)
    calls: list[str] = []
    observed_reserved: list[float] = []

    def runner(argv, **kwargs):
        state = json.loads(Path(summary["state_path"]).read_text())
        observed_reserved.append(float(state["unresolved_cost_reservation_usd"]))
        cell_id = _cell_id_from_argv(list(argv))
        calls.append(cell_id)
        completed.add(cell_id)
        return _child_result(0.15)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=0.15,
        accept_estimate_shortfall=True,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        cell_workers=3,
        command_runner=runner,
    )

    assert result["status"] == "BUDGET_EXHAUSTED"
    assert calls == [cells[0]]
    assert observed_reserved == [pytest.approx(0.15)]
    assert result["cumulative_cost_usd"] == pytest.approx(0.15)
    assert result["unresolved_cost_reservation_usd"] == 0
    state = json.loads(Path(summary["state_path"]).read_text())
    assert state["status"] == "BUDGET_EXHAUSTED"
    assert state["stop_reason"].startswith("NEXT_CELL_RESERVATION_")
    assert state["diagnostic_refresh_error"]["error"] == (
        "RuntimeError: diagnostic refresh unavailable"
    )


def test_rate_limit_stops_later_waves_but_drains_successful_sibling(monkeypatch, tmp_path):
    cells = [f"{sector}:micro_cap" for sector in CANONICAL_SWEEP_SECTORS[:3]]
    pending = {cell_id: [f"R{index}"] for index, cell_id in enumerate(cells)}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)
    calls: list[str] = []

    def runner(argv, **kwargs):
        cell_id = _cell_id_from_argv(list(argv))
        calls.append(cell_id)
        if cell_id == cells[0]:
            return _child_result(
                0.05,
                returncode=1,
                error="status=429 rate limit",
                coverage_accounting={"rerun_required": True},
            )
        completed.add(cell_id)
        return _child_result(0.05)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        cell_workers=2,
        command_runner=runner,
    )
    state = json.loads(Path(summary["state_path"]).read_text())

    assert calls == cells[:2]
    assert [attempt["status"] for attempt in state["attempts"]] == [
        "SAFE_RETRY_REQUIRED",
        "OK",
    ]
    assert result["status"] == "INCOMPLETE_RETRY_REQUIRED"
    assert result["stop_reason"] == f"RATE_LIMIT_BACKPRESSURE:{cells[0]}"
    assert result["cumulative_cost_usd"] == pytest.approx(0.10)
    assert result["unresolved_cost_reservation_usd"] == 0


def test_business_text_cannot_create_false_rate_limit_backpressure(monkeypatch, tmp_path):
    cell_id = "energy:micro_cap"
    pending = {cell_id: ["TEXT"]}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=lambda *args, **kwargs: _child_result(
            0.05,
            returncode=1,
            narrative="The issuer compares its service to a rate limit policy.",
            coverage_accounting={"rerun_required": True},
        ),
    )

    assert result["status"] == "INCOMPLETE_CELLS_STALLED"
    assert result["stop_reason"] == f"CELLS_STALLED_NO_PROGRESS:{cell_id}"
    assert result["n_stalled_cells"] == 1


def test_progress_callback_failures_are_nonfatal_and_audited(monkeypatch, tmp_path):
    cells = [f"{sector}:micro_cap" for sector in CANONICAL_SWEEP_SECTORS[:2]]
    pending = {cell_id: [f"P{index}"] for index, cell_id in enumerate(cells)}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)
    calls: list[str] = []

    def runner(argv, **kwargs):
        cell_id = _cell_id_from_argv(list(argv))
        calls.append(cell_id)
        completed.add(cell_id)
        return _child_result(0.05)

    def broken_progress(event):
        raise RuntimeError(f"telemetry offline for {event['event']}")

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        cell_workers=2,
        command_runner=runner,
        progress=broken_progress,
    )
    state = json.loads(Path(summary["state_path"]).read_text())

    assert result["status"] == "COMPLETE"
    assert set(calls) == set(cells)
    assert [attempt["status"] for attempt in state["attempts"]] == ["OK", "OK"]
    assert [row["event"] for row in state["telemetry_errors"]].count("CELL_START") == 2
    assert [row["event"] for row in state["telemetry_errors"]].count("CELL_COMPLETE") == 2


def test_unresolved_worker_drains_sibling_and_resume_skips_success(monkeypatch, tmp_path):
    cells = [f"{sector}:micro_cap" for sector in CANONICAL_SWEEP_SECTORS[:3]]
    pending = {cell_id: [f"U{index}"] for index, cell_id in enumerate(cells)}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)
    first_calls: list[str] = []

    def first_runner(argv, **kwargs):
        cell_id = _cell_id_from_argv(list(argv))
        first_calls.append(cell_id)
        if cell_id == cells[0]:
            return subprocess.CompletedProcess([], 1, stdout="not-json", stderr="ambiguous failure")
        completed.add(cell_id)
        return _child_result(0.05)

    first = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        cell_workers=2,
        command_runner=first_runner,
    )
    first_state = json.loads(Path(summary["state_path"]).read_text())

    assert set(first_calls) == set(cells[:2])
    assert [attempt["status"] for attempt in first_state["attempts"]] == [
        "UNRESOLVED_ATTEMPT",
        "OK",
    ]
    assert first["status"] == "RECONCILIATION_REQUIRED"
    assert first["cumulative_cost_usd"] == pytest.approx(0.05)
    assert first["unresolved_cost_reservation_usd"] == pytest.approx(0.15)

    reconciled = driver.reconcile_campaign(
        campaign_id=summary["campaign_id"],
        assume_reserved_spent=True,
        confirm_child_stopped=True,
        output_root=tmp_path,
    )
    assert reconciled["cumulative_cost_usd"] == pytest.approx(0.20)
    second_calls: list[str] = []

    def second_runner(argv, **kwargs):
        cell_id = _cell_id_from_argv(list(argv))
        second_calls.append(cell_id)
        completed.add(cell_id)
        return _child_result(0.05)

    second = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        cell_workers=2,
        command_runner=second_runner,
    )

    assert second["status"] == "COMPLETE"
    assert set(second_calls) == {cells[0], cells[2]}
    assert cells[1] not in second_calls
    final_state = json.loads(Path(summary["state_path"]).read_text())
    assert [attempt["cell_id"] for attempt in final_state["attempts"]] == [
        cells[0],
        cells[1],
        cells[0],
        cells[2],
    ]


def test_zero_cost_physical_attempt_still_enforces_provider_binding(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(
        mode="missed-only",
        populate_watchlist=False,
        output_root=tmp_path,
    )
    payload = {
        "status": "COMPLETED",
        "artifact_path": None,
        "provider_usage_attestation": {
            "valid": True,
            "physical_attempt_count": 1,
            "cost_estimate_usd": 0.0,
            "provider_models": [{"provider": "openai", "model": "wrong"}],
            "usage_records_sha256": "b" * 64,
        },
    }

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    assert result["status"] == "PROVIDER_MODEL_DRIFT"
    assert result["cumulative_cost_usd"] == 0
    assert result["known_physical_provider_attempts"] == 1


def test_reconcile_preserves_hard_sibling_blocker(monkeypatch, tmp_path):
    cells = [f"{sector}:micro_cap" for sector in CANONICAL_SWEEP_SECTORS[:2]]
    pending = {cell_id: [f"H{index}"] for index, cell_id in enumerate(cells)}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)

    def runner(argv, **kwargs):
        cell_id = _cell_id_from_argv(list(argv))
        if cell_id == cells[1]:
            return subprocess.CompletedProcess([], 1, stdout="not-json", stderr="")
        result = _child_result(0.05)
        payload = json.loads(result.stdout)
        payload["provider_usage_attestation"] = _attestation(
            0.05,
            provider="openai",
            model="wrong",
        )
        result.stdout = json.dumps(payload)
        return result

    first = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        cell_workers=2,
        command_runner=runner,
    )
    assert first["status"] == "RECONCILIATION_REQUIRED"

    reconciled = driver.reconcile_campaign(
        campaign_id=summary["campaign_id"],
        assume_reserved_spent=True,
        confirm_child_stopped=True,
        output_root=tmp_path,
    )
    assert reconciled["status"] == "PROVIDER_MODEL_DRIFT"
    assert reconciled["stop_reason"] == f"PROVIDER_MODEL_DRIFT:{cells[0]}"


def test_keyboard_interrupt_is_deferred_until_launched_sibling_is_persisted(monkeypatch, tmp_path):
    cells = [f"{sector}:micro_cap" for sector in CANONICAL_SWEEP_SECTORS[:2]]
    pending = {cell_id: [f"K{index}"] for index, cell_id in enumerate(cells)}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)

    def runner(argv, **kwargs):
        cell_id = _cell_id_from_argv(list(argv))
        if cell_id == cells[0]:
            raise KeyboardInterrupt("operator interrupted coordinator")
        completed.add(cell_id)
        return _child_result(0.05)

    with pytest.raises(KeyboardInterrupt, match="operator interrupted"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            cell_workers=2,
            command_runner=runner,
        )

    state = json.loads(Path(summary["state_path"]).read_text())
    assert state["status"] == "RECONCILIATION_REQUIRED"
    assert [attempt["status"] for attempt in state["attempts"]] == [
        "UNRESOLVED_INTERRUPTED",
        "OK",
    ]
    assert state["cumulative_cost_usd"] == pytest.approx(0.05)
    assert state["unresolved_cost_reservation_usd"] == pytest.approx(0.15)


def test_coordinator_interrupt_checkpoints_completed_sibling_in_completion_order(
    monkeypatch, tmp_path
):
    cells = [f"{sector}:micro_cap" for sector in CANONICAL_SWEEP_SECTORS[:2]]
    pending = {cell_id: [f"C{index}"] for index, cell_id in enumerate(cells)}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)
    release_slow = threading.Event()
    real_as_completed = driver.as_completed

    def runner(argv, **kwargs):
        cell_id = _cell_id_from_argv(list(argv))
        if cell_id == cells[0]:
            release_slow.wait(timeout=5)
        completed.add(cell_id)
        return _child_result(0.05)

    def interrupt_after_first_completion(futures):
        for future in real_as_completed(futures):
            yield future
            raise KeyboardInterrupt("operator interrupted coordinator wait")

    monkeypatch.setattr(driver, "as_completed", interrupt_after_first_completion)
    try:
        with pytest.raises(KeyboardInterrupt, match="coordinator wait"):
            driver.run_campaign(
                campaign_id=summary["campaign_id"],
                authorized_max_cost_usd=10,
                expect_plan_sha256=summary["plan_sha256"],
                expect_provider="anthropic",
                expect_model="claude-sonnet-test",
                output_root=tmp_path,
                cell_workers=2,
                command_runner=runner,
            )
    finally:
        release_slow.set()

    state = json.loads(Path(summary["state_path"]).read_text())
    assert state["status"] == "RECONCILIATION_REQUIRED"
    assert [attempt["status"] for attempt in state["attempts"]] == [
        "UNRESOLVED_INTERRUPTED",
        "OK",
    ]
    assert state["cumulative_cost_usd"] == pytest.approx(0.05)
    assert state["unresolved_cost_reservation_usd"] == pytest.approx(0.15)


def test_submit_window_interrupt_keeps_ambiguous_paid_reservation(monkeypatch, tmp_path):
    cells = [f"{sector}:micro_cap" for sector in CANONICAL_SWEEP_SECTORS[:2]]
    pending = {cell_id: [f"S{index}"] for index, cell_id in enumerate(cells)}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)
    calls: list[str] = []
    shutdown_wait: list[bool] = []

    def runner(argv, **kwargs):
        cell_id = _cell_id_from_argv(list(argv))
        calls.append(cell_id)
        completed.add(cell_id)
        return _child_result(0.05)

    class InterruptingExecutor:
        def __init__(self, **kwargs):
            pass

        def submit(self, function, *args, **kwargs):
            function(*args, **kwargs)
            raise KeyboardInterrupt("interrupted after paid dispatch")

        def shutdown(self, *, wait, cancel_futures):
            shutdown_wait.append(wait)

    monkeypatch.setattr(driver, "ThreadPoolExecutor", InterruptingExecutor)
    with pytest.raises(KeyboardInterrupt, match="after paid dispatch"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            cell_workers=2,
            command_runner=runner,
        )

    state = json.loads(Path(summary["state_path"]).read_text())
    assert calls == [cells[0]]
    assert shutdown_wait == [False]
    assert [attempt["status"] for attempt in state["attempts"]] == [
        "UNRESOLVED_INTERRUPTED",
        "SAFE_RETRY_REQUIRED",
    ]
    assert state["cumulative_cost_usd"] == 0
    assert state["unresolved_cost_reservation_usd"] == pytest.approx(0.15)


def test_completion_progress_interrupt_does_not_double_settle_cost(monkeypatch, tmp_path):
    cell_id = "energy:micro_cap"
    pending = {cell_id: ["DONE"]}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)

    def runner(argv, **kwargs):
        completed.add(_cell_id_from_argv(list(argv)))
        return _child_result(0.05)

    def interrupt_completion(event):
        if event["event"] == "CELL_COMPLETE":
            raise KeyboardInterrupt("interrupted during completion telemetry")

    with pytest.raises(KeyboardInterrupt, match="completion telemetry"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            command_runner=runner,
            progress=interrupt_completion,
        )

    state = json.loads(Path(summary["state_path"]).read_text())
    assert state["status"] == "INCOMPLETE_RETRY_REQUIRED"
    assert state["stop_reason"] == "COORDINATOR_INTERRUPTED_AFTER_SETTLEMENT"
    assert state["attempts"][0]["status"] == "OK"
    assert state["attempts"][0]["charged_cost_usd"] == pytest.approx(0.05)
    assert state["cumulative_cost_usd"] == pytest.approx(0.05)
    assert state["unresolved_cost_reservation_usd"] == 0


def test_interrupt_between_cost_reconciliation_and_status_is_idempotent(monkeypatch, tmp_path):
    cell_id = "energy:micro_cap"
    pending = {cell_id: ["COST"]}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)
    real_resolve = driver._resolve_attempt_cost
    interrupted = False

    def runner(argv, **kwargs):
        completed.add(_cell_id_from_argv(list(argv)))
        return _child_result(0.05)

    def interrupt_after_resolve(state, attempt, charged_cost):
        nonlocal interrupted
        real_resolve(state, attempt, charged_cost)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("interrupted after exact cost reconciliation")

    monkeypatch.setattr(driver, "_resolve_attempt_cost", interrupt_after_resolve)
    with pytest.raises(KeyboardInterrupt, match="exact cost reconciliation"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            command_runner=runner,
        )

    state = json.loads(Path(summary["state_path"]).read_text())
    assert state["attempts"][0]["status"] == "OK"
    assert state["attempts"][0]["charged_cost_usd"] == pytest.approx(0.05)
    assert state["cumulative_cost_usd"] == pytest.approx(0.05)
    assert state["unresolved_cost_reservation_usd"] == 0


def test_authorization_below_plan_estimate_stops_before_child(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    calls = []

    with pytest.raises(driver.ClassicScanAuthorizationError, match="below the plan estimate"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=0.01,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            command_runner=lambda *args, **kwargs: calls.append(args),
        )
    assert calls == []


def test_explicit_owner_override_accepts_below_estimate_hard_cap(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    _state_path, state = driver._load_state(summary["campaign_id"], tmp_path)

    ceiling = driver._validate_authorization(
        state,
        authorized_max_cost_usd=0.01,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        accept_estimate_shortfall=True,
    )

    assert ceiling == 0.01


def test_membership_drift_stops_before_child(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    monkeypatch.setattr(driver, "_quick_membership_fingerprint", lambda: "changed")

    with pytest.raises(driver.ClassicScanPlanError, match="membership fingerprint changed"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
            command_runner=lambda *args, **kwargs: pytest.fail("child launched"),
        )


def test_unparseable_child_reserves_cost_blocks_resume_and_requires_reconcile(
    monkeypatch, tmp_path
):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, stdout="not-json", stderr="failed"
        ),
    )
    assert result["status"] == "RECONCILIATION_REQUIRED"
    assert result["unresolved_cost_reservation_usd"] > 0

    with pytest.raises(driver.ClassicScanAuthorizationError, match="unresolved"):
        driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=tmp_path,
        )
    with pytest.raises(driver.ClassicScanAuthorizationError):
        driver.reconcile_campaign(
            campaign_id=summary["campaign_id"],
            assume_reserved_spent=True,
            confirm_child_stopped=False,
            output_root=tmp_path,
        )
    reconciled = driver.reconcile_campaign(
        campaign_id=summary["campaign_id"],
        assume_reserved_spent=True,
        confirm_child_stopped=True,
        output_root=tmp_path,
    )
    assert reconciled["unresolved_cost_reservation_usd"] == 0
    assert reconciled["cumulative_cost_usd"] > 0


def test_known_cost_above_cell_reservation_is_charged_then_blocks(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=lambda *args, **kwargs: _child_result(0.5),
    )

    assert result["status"] == "CELL_COST_RESERVATION_BREACH"
    assert result["cumulative_cost_usd"] == 0.5
    assert result["unresolved_cost_reservation_usd"] == 0


def test_provider_model_drift_is_charged_and_blocks(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    bad = _child_result(0.1)
    payload = json.loads(bad.stdout)
    payload["provider_usage_attestation"] = _attestation(0.1, provider="openai", model="wrong")
    bad.stdout = json.dumps(payload)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=lambda *args, **kwargs: bad,
    )
    assert result["status"] == "PROVIDER_MODEL_DRIFT"
    assert result["cumulative_cost_usd"] == 0.1


def test_safe_failed_child_is_resumable_not_unauditable(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=lambda *args, **kwargs: _child_result(
            0.1,
            returncode=1,
            coverage_accounting={"rerun_required": True},
        ),
    )
    assert result["status"] == "INCOMPLETE_CELLS_STALLED"
    assert result["stop_reason"] == "CELLS_STALLED_NO_PROGRESS:energy:micro_cap"
    assert result["attempt_count"] == 1
    assert result["n_stalled_cells"] == 1
    assert result["unresolved_cost_reservation_usd"] == 0


def test_cell_needing_multiple_passes_is_requeued_until_covered(
    monkeypatch, tmp_path
):
    cell_id = "energy:micro_cap"
    all_tickers = ["AAA", "BBB", "CCC"]
    pending = {cell_id: list(all_tickers)}
    _patch_dependencies(
        monkeypatch,
        _report(pending=pending, loaded=pending, eligible=all_tickers),
    )
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    remaining = list(all_tickers)

    def dynamic_report(**kwargs):
        live_pending = {cell_id: list(remaining)} if remaining else {}
        return _scope_report(
            _report(
                pending=live_pending,
                loaded=pending,
                eligible=all_tickers,
                target_tickers=all_tickers,
            ),
            kwargs,
        )

    monkeypatch.setattr(driver, "full_universe_delta_report", dynamic_report)
    monkeypatch.setattr(
        driver,
        "_live_uncovered_for_cell",
        lambda _state, _cell: list(remaining),
    )

    def runner(*args, **kwargs):
        remaining.pop(0)
        return _child_result(
            0.1,
            coverage_accounting={"rerun_required": bool(remaining)},
        )

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=runner,
    )
    state = json.loads(Path(summary["state_path"]).read_text())

    assert result["status"] == "COMPLETE"
    assert result["attempt_count"] == 3
    assert result["n_stalled_cells"] == 0
    assert result["cumulative_cost_usd"] == pytest.approx(0.3)
    assert [attempt["live_uncovered_before"] for attempt in state["attempts"]] == [
        ["AAA", "BBB", "CCC"],
        ["BBB", "CCC"],
        ["CCC"],
    ]


def test_stalled_cell_is_abandoned_without_a_second_charge(monkeypatch, tmp_path):
    report = _report(
        pending={"energy:micro_cap": ["AAA", "BBB"]},
        eligible=["AAA", "BBB"],
    )
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=lambda *args, **kwargs: _child_result(
            0.1,
            coverage_accounting={"rerun_required": True},
        ),
    )

    assert result["status"] == "INCOMPLETE_CELLS_STALLED"
    assert result["attempt_count"] == 1
    assert result["cumulative_cost_usd"] == pytest.approx(0.1)
    assert result["stalled_cells"] == [
        {
            "cell_id": "energy:micro_cap",
            "sector": "energy",
            "band": "micro_cap",
            "uncovered_before": 2,
            "uncovered_after": 2,
            "residual_tickers": ["AAA", "BBB"],
        }
    ]


def test_one_cell_needing_a_rerun_does_not_halt_the_other_cells(
    monkeypatch, tmp_path
):
    report = _report(
        pending={
            "energy:micro_cap": ["AAA"],
            "biotech:micro_cap": ["BBB"],
        },
        eligible=["AAA", "BBB"],
    )
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    attempted: list[str] = []

    def runner(argv, **kwargs):
        sector = argv[argv.index("--sector") + 1]
        attempted.append(sector)
        return _child_result(
            0.1,
            coverage_accounting={"rerun_required": sector == "biotech"},
        )

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=runner,
    )

    assert attempted == ["biotech", "energy"]
    assert result["attempt_count"] == 2
    assert result["n_stalled_cells"] == 1
    assert result["stalled_cells"][0]["cell_id"] == "biotech:micro_cap"


def test_requeue_semantics_match_default_and_two_workers_with_fixed_order(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("VOE_CLASSIC_CELL_WORKERS", raising=False)
    cells = ["biotech:micro_cap", "energy:micro_cap"]
    pending = {
        cells[0]: ["AAA", "BBB"],
        cells[1]: ["CCC"],
    }
    expected_attempt_cells = [cells[0], cells[1], cells[0]]
    observed: dict[str, dict[str, object]] = {}

    for mode, workers in (("default", None), ("two", 2)):
        output_root = tmp_path / mode
        remaining = {cell_id: list(tickers) for cell_id, tickers in pending.items()}
        all_tickers = ["AAA", "BBB", "CCC"]
        _patch_dependencies(
            monkeypatch,
            _report(
                pending=pending,
                loaded=pending,
                eligible=all_tickers,
            ),
        )
        summary = driver.plan_campaign(
            mode="full-rescan",
            populate_watchlist=False,
            output_root=output_root,
        )
        lock = threading.Lock()
        progress_events: list[dict] = []

        def dynamic_report(
            _remaining=remaining,
            _all_tickers=all_tickers,
            **kwargs,
        ):
            live_pending = {
                cell_id: list(tickers)
                for cell_id, tickers in _remaining.items()
                if tickers
            }
            return _scope_report(
                _report(
                    pending=live_pending,
                    loaded=pending,
                    eligible=_all_tickers,
                    target_tickers=_all_tickers,
                ),
                kwargs,
            )

        monkeypatch.setattr(driver, "full_universe_delta_report", dynamic_report)
        monkeypatch.setattr(
            driver,
            "_live_uncovered_for_cell",
            lambda _state, cell, _remaining=remaining: list(
                _remaining[str(cell["cell_id"])]
            ),
        )

        def runner(argv, _remaining=remaining, _lock=lock, **kwargs):
            cell_id = _cell_id_from_argv(list(argv))
            with _lock:
                _remaining[cell_id].pop(0)
                rerun_required = bool(_remaining[cell_id])
            return _child_result(
                0.05,
                coverage_accounting={"rerun_required": rerun_required},
            )

        result = driver.run_campaign(
            campaign_id=summary["campaign_id"],
            authorized_max_cost_usd=10,
            expect_plan_sha256=summary["plan_sha256"],
            expect_provider="anthropic",
            expect_model="claude-sonnet-test",
            output_root=output_root,
            cell_workers=workers,
            command_runner=runner,
            progress=progress_events.append,
        )
        state = json.loads(Path(summary["state_path"]).read_text())
        observed[mode] = {
            "status": result["status"],
            "attempt_cells": [attempt["cell_id"] for attempt in state["attempts"]],
            "biotech_claims": [
                attempt["live_uncovered_before"]
                for attempt in state["attempts"]
                if attempt["cell_id"] == cells[0]
            ],
            "requeues": [
                event["cell_id"]
                for event in progress_events
                if event["event"] == "CELL_REQUEUED"
            ],
            "cost": result["cumulative_cost_usd"],
        }

    assert observed == {
        "default": {
            "status": "COMPLETE",
            "attempt_cells": expected_attempt_cells,
            "biotech_claims": [["AAA", "BBB"], ["BBB"]],
            "requeues": [cells[0]],
            "cost": pytest.approx(0.15),
        },
        "two": {
            "status": "COMPLETE",
            "attempt_cells": expected_attempt_cells,
            "biotech_claims": [["AAA", "BBB"], ["BBB"]],
            "requeues": [cells[0]],
            "cost": pytest.approx(0.15),
        },
    }


def test_verify_requires_campaign_state_complete(monkeypatch, tmp_path):
    pending = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    complete = _report(loaded={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, pending)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    monkeypatch.setattr(
        driver,
        "full_universe_delta_report",
        lambda **kwargs: _scope_report(complete, kwargs),
    )

    verified = driver.verify_campaign(campaign_id=summary["campaign_id"], output_root=tmp_path)
    assert verified["complete"] is False
    assert verified["campaign_state_status"] == "PLANNED"


def test_registered_target_validation_rejects_tampering(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["AAA"]}, eligible=["AAA"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)

    assert driver.validate_coverage_target_file(
        summary["target_file"], campaign_id=summary["campaign_id"]
    ) == ["AAA"]
    target_path = Path(summary["target_file"])
    target = json.loads(target_path.read_text())
    target["tickers"] = ["AAA", "EXTRA"]
    target_path.write_text(json.dumps(target))
    with pytest.raises(driver.ClassicScanPlanError):
        driver.validate_coverage_target_file(target_path, campaign_id=summary["campaign_id"])


def test_campaign_state_validation_rejects_ticker_claim_hash_tampering(monkeypatch, tmp_path):
    cell_id = "energy:micro_cap"
    pending = {cell_id: ["CLAIM"]}
    completed: set[str] = set()
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, completed)

    def runner(argv, **kwargs):
        completed.add(_cell_id_from_argv(list(argv)))
        return _child_result(0.05)

    result = driver.run_campaign(
        campaign_id=summary["campaign_id"],
        authorized_max_cost_usd=10,
        expect_plan_sha256=summary["plan_sha256"],
        expect_provider="anthropic",
        expect_model="claude-sonnet-test",
        output_root=tmp_path,
        command_runner=runner,
    )
    state_path = Path(result["state_path"])
    state = json.loads(state_path.read_text())
    state["attempts"][0]["claimed_tickers_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state))

    with pytest.raises(driver.ClassicScanPlanError, match="ticker claim hash"):
        driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)


def test_legacy_serial_attempt_without_claim_hash_can_load_and_reconcile(monkeypatch, tmp_path):
    report = _report(pending={"energy:micro_cap": ["LEGACY"]}, eligible=["LEGACY"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path)
    state_path = Path(summary["state_path"])
    state = json.loads(state_path.read_text())
    state["status"] = "RECONCILIATION_REQUIRED"
    state["stop_reason"] = "legacy interrupted child"
    state["unresolved_cost_reservation_usd"] = 0.15
    state["attempts"] = [
        {
            "attempt_number": 1,
            "cell_id": "energy:micro_cap",
            "sector": "energy",
            "band": "micro_cap",
            "status": "UNRESOLVED_ATTEMPT",
            "live_uncovered_before": ["LEGACY"],
            "reserved_cost_usd": 0.15,
            "cost_reconciled": False,
        }
    ]
    state_path.write_text(json.dumps(state))

    loaded = driver.campaign_status(campaign_id=summary["campaign_id"], output_root=tmp_path)
    assert loaded["running_attempts"] == 1
    reconciled = driver.reconcile_campaign(
        campaign_id=summary["campaign_id"],
        assume_reserved_spent=True,
        confirm_child_stopped=True,
        output_root=tmp_path,
    )
    assert reconciled["cumulative_cost_usd"] == pytest.approx(0.15)
    assert reconciled["unresolved_cost_reservation_usd"] == 0


def test_child_cli_intersects_normal_loader_with_registered_target(monkeypatch, tmp_path):
    from app.autonomous.sector_candidates import SectorCandidateSelection
    from app.autonomous.sweep_delta import record_loaded_set
    from app.config import get_config
    from app.db import get_db, init_db

    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    init_db()
    report = _report(pending={"energy:micro_cap": ["TARGET"]}, eligible=["TARGET"])
    _patch_dependencies(monkeypatch, report)
    summary = driver.plan_campaign(mode="missed-only", output_root=tmp_path / "campaigns")
    with get_db() as conn:
        record_loaded_set(
            conn,
            run_id="already-done",
            sector="energy",
            market_cap_focus="micro_cap",
            source="sector_scan_db",
            tickers=["TARGET"],
            pipeline_version="v1",
            candidate_dispositions={"TARGET": "LLM_CANDIDATE_REVIEW_COMPLETED"},
            coverage_campaign_id=summary["campaign_id"],
        )

    monkeypatch.setattr(
        "app.autonomous.sector_candidates.resolve_sector_candidate_tickers",
        lambda **kwargs: SectorCandidateSelection(
            sector="energy",
            market_cap_focus="micro_cap",
            selected_tickers=["TARGET", "EXTRA"],
            loaded_tickers=["TARGET", "EXTRA"],
            source="sector_scan_db",
            cap_classifications={"TARGET": {}, "EXTRA": {}},
            structural_gate_results={},
        ),
    )
    result = CliRunner().invoke(
        app,
        [
            "autonomous-sector-run",
            "--sector",
            "energy",
            "--market-cap-focus",
            "micro_cap",
            "--only-unswept",
            "--pipeline-version",
            "v1",
            "--coverage-campaign-id",
            summary["campaign_id"],
            "--coverage-target-file",
            summary["target_file"],
            "--no-carry-prior-verdicts",
            "--no-watchlist",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "EMPTY_DELTA"
    assert payload["loaded"] == 1
    assert payload["swept_excluded"] == 1


def test_recursive_usage_attestation_deduplicates_and_rejects_bad_cost():
    row = {
        "provider": "anthropic",
        "model": "claude-sonnet-test",
        "status": "OK",
        "cost_estimate_usd": 0.25,
    }
    attested = provider_usage_attestation(
        {"provider_usage": [row], "nested": {"provider_usage": [dict(row)]}}
    )
    assert attested["physical_attempt_count"] == 1
    assert attested["cost_estimate_usd"] == 0.25
    assert (
        provider_usage_attestation(
            {"provider_usage": [{**row, "cost_estimate_usd": float("nan")}]}
        )["valid"]
        is False
    )


def test_terminal_data_gap_reconciliation_accounts_without_coverage(
    monkeypatch, tmp_path
):
    pending = {"energy:micro_cap": ["AAA", "BBB", "CCC"]}
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, set())
    initial_state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    payload = _terminal_data_gap_payload(
        initial_state,
        cell_id="energy:micro_cap",
        missing_valuation=["AAA"],
        sparse_history=["BBB"],
        structural_screened=["CCC"],
    )
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=payload,
    )
    state_bytes = state_path.read_bytes()
    target_path = Path(state["target_file"])
    target_bytes = target_path.read_bytes()
    state_sha256 = driver._raw_file_sha256(state_path)
    artifact_path = tmp_path / "terminal-data-gaps.json"
    monkeypatch.setattr(
        driver,
        "get_db",
        lambda: (_ for _ in ()).throw(AssertionError("reconciliation must not open the DB")),
    )

    result = driver.reconcile_terminal_data_gaps(
        campaign_id=state["campaign_id"],
        expect_plan_sha256=state["plan_sha256"],
        expect_state_sha256=state_sha256,
        expect_target_tickers_sha256=state["target_tickers_sha256"],
        artifact_path=artifact_path,
        output_root=tmp_path,
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert result["status"] == "TERMINAL_ACCOUNTING_COMPLETE_COVERAGE_INCOMPLETE"
    assert result["terminal_execution_accounting_complete"] is True
    assert result["investment_coverage_complete"] is False
    assert result["needs_data_ticker_count"] == 2
    assert artifact_path.stat().st_mode & 0o777 == 0o600
    assert state_path.read_bytes() == state_bytes
    assert target_path.read_bytes() == target_bytes
    assert report["campaign"]["charged_cost_usd"] == 0
    assert report["campaign"]["unresolved_cost_reservation_usd"] == 0
    assert report["reconciliation_source_contract"]["sha256"] == driver._source_contract()[
        "sha256"
    ]
    assert report["classification"]["ticker_count"] == 3
    assert report["classification"]["tickers_sha256"] == (
        "7ef16e7a59c5fa66fc95eb3ea8a35ae230a52937dada7db2b0e900201584b5ff"
    )
    assert report["classification"]["needs_data_tickers_sha256"] == (
        "518d1d0ec11d9c4a85cbc86de03771a3cc2b7c35eb40e6b70c08b0c736552490"
    )
    assert {
        key: value["ticker_count"]
        for key, value in report["classification"]["buckets"].items()
    } == {
        "NEEDS_DATA_MISSING_VALUATION": 1,
        "NEEDS_DATA_SPARSE_HISTORY": 1,
        "STRUCTURAL_SCREENED": 1,
    }
    assert report["terminal_execution_accounting"]["expected_cell_count"] == 170
    assert report["terminal_execution_accounting"]["accounted_cell_count"] == 170
    assert report["terminal_execution_accounting"]["ticker_occurrence_count"] == 3
    assert report["investment_coverage"] == {
        "status": "INCOMPLETE_NEEDS_DATA",
        "complete": False,
        "needs_data_ticker_count": 2,
        "needs_data_coverage_credit": 0,
        "decision_eligible_claims_created": 0,
        "semantics": (
            "NEEDS_DATA is a terminal execution explanation, not investment "
            "coverage or decision eligibility."
        ),
    }
    with pytest.raises(driver.ClassicScanPlanError, match="refusing to replace"):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=state_sha256,
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=artifact_path,
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_binds_unknown_cap_projections(
    monkeypatch, tmp_path
):
    pending = {f"energy:{band}": ["UNKNOWN"] for band in V1_ATOMIC_BANDS}
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, set())
    initial_state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    payload = _terminal_data_gap_payload(
        initial_state,
        cell_id="energy:micro_cap",
        missing_valuation=["UNKNOWN"],
        sparse_history=[],
        structural_screened=[],
        projected_tickers=["UNKNOWN"],
        projected_bands=["small_cap", "mid_cap", "large_cap", "mega_cap"],
    )
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=payload,
    )
    artifact_path = tmp_path / "projected-terminal-data-gaps.json"

    driver.reconcile_terminal_data_gaps(
        campaign_id=state["campaign_id"],
        expect_plan_sha256=state["plan_sha256"],
        expect_state_sha256=driver._raw_file_sha256(state_path),
        expect_target_tickers_sha256=state["target_tickers_sha256"],
        artifact_path=artifact_path,
        output_root=tmp_path,
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))
    accounting = report["terminal_execution_accounting"]

    assert accounting["ticker_occurrence_count"] == 5
    assert accounting["accounted_ticker_occurrence_count"] == 5
    assert accounting["attempted_cell_count"] == 1
    assert accounting["projected_only_nonempty_cell_count"] == 4
    assert report["classification"]["tickers_sha256"] == (
        "09247cbf234f183759f2fd54028274f628f813e9d2f04ae69a7ee391ce896089"
    )
    projected = [
        cell
        for cell in accounting["cells"]
        if cell["resolution_source"] == "EXPLICIT_UNKNOWN_CAP_CROSS_BAND_PROJECTION"
    ]
    assert [cell["cell_id"] for cell in projected] == [
        "energy:small_cap",
        "energy:mid_cap",
        "energy:large_cap",
        "energy:mega_cap",
    ]


def test_terminal_data_gap_reconciliation_allows_only_prior_projection_for_omitted_direct(
    monkeypatch, tmp_path
):
    pending = {
        "energy:micro_cap": ["UNKNOWN"],
        "energy:small_cap": ["UNKNOWN"],
    }
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, set())
    state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    first = _terminal_data_gap_payload(
        state,
        cell_id="energy:micro_cap",
        missing_valuation=["UNKNOWN"],
        sparse_history=[],
        structural_screened=[],
        projected_tickers=["UNKNOWN"],
        projected_bands=["small_cap"],
    )
    _state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=first,
    )
    second = _terminal_data_gap_payload(
        state,
        cell_id="energy:small_cap",
        missing_valuation=[],
        sparse_history=[],
        structural_screened=[],
    )
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=second,
    )
    artifact_path = tmp_path / "prior-projection-terminal-data-gaps.json"

    driver.reconcile_terminal_data_gaps(
        campaign_id=state["campaign_id"],
        expect_plan_sha256=state["plan_sha256"],
        expect_state_sha256=driver._raw_file_sha256(state_path),
        expect_target_tickers_sha256=state["target_tickers_sha256"],
        artifact_path=artifact_path,
        output_root=tmp_path,
    )
    report = json.loads(artifact_path.read_text(encoding="utf-8"))
    small = next(
        cell
        for cell in report["terminal_execution_accounting"]["cells"]
        if cell["cell_id"] == "energy:small_cap"
    )

    assert small["attempted_in_source_campaign"] is True
    assert small["reason_counts"]["NEEDS_DATA_MISSING_VALUATION"] == 1
    assert small["evidence_sources"] == [
        {
            "attempt_number": 1,
            "source_kind": "EXPLICIT_UNKNOWN_CAP_CROSS_BAND_PROJECTION",
            "source_cell_id": "energy:micro_cap",
        }
    ]


def test_terminal_data_gap_reconciliation_rejects_unproven_cell_occurrences(
    monkeypatch, tmp_path
):
    pending = {f"energy:{band}": ["UNKNOWN"] for band in V1_ATOMIC_BANDS}
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, set())
    initial_state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    payload = _terminal_data_gap_payload(
        initial_state,
        cell_id="energy:micro_cap",
        missing_valuation=["UNKNOWN"],
        sparse_history=[],
        structural_screened=[],
    )
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=payload,
    )

    with pytest.raises(driver.ClassicScanPlanError, match="every frozen cell occurrence"):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_rejects_conflicting_evidence(monkeypatch, tmp_path):
    pending = {"energy:micro_cap": ["AAA"]}
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, set())
    initial_state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    payload = _terminal_data_gap_payload(
        initial_state,
        cell_id="energy:micro_cap",
        missing_valuation=["AAA"],
        sparse_history=[],
        structural_screened=["AAA"],
    )
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=payload,
    )

    with pytest.raises(
        driver.ClassicScanPlanError,
        match="conflicting terminal reasons",
    ):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_rejects_fallback_scan_family(
    monkeypatch, tmp_path
):
    pending = {"energy:micro_cap": ["AAA"]}
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, set())
    state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    payload = _terminal_data_gap_payload(
        state,
        cell_id="energy:micro_cap",
        missing_valuation=["AAA"],
        sparse_history=[],
        structural_screened=[],
    )
    payload["scan_family"] = "fallback"
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=payload,
    )

    with pytest.raises(driver.ClassicScanPlanError, match="exact pre-provider refusal"):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_rejects_wrong_child_as_of(
    monkeypatch, tmp_path
):
    summary = _plan_dynamic_campaign(
        monkeypatch,
        tmp_path,
        {"energy:micro_cap": ["AAA"]},
        set(),
    )
    state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    payload = _terminal_data_gap_payload(
        state,
        cell_id="energy:micro_cap",
        missing_valuation=["AAA"],
        sparse_history=[],
        structural_screened=[],
    )
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=payload,
    )
    as_of_index = state["attempts"][0]["argv"].index("--as-of") + 1
    state["attempts"][0]["argv"][as_of_index] = "2026-07-21"
    driver._atomic_write_json(state_path, state)
    driver._validate_state_integrity(state)

    with pytest.raises(driver.ClassicScanPlanError, match="invocation contract is not exact"):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_rejects_incomplete_direct_partition(
    monkeypatch, tmp_path
):
    pending = {
        "energy:micro_cap": ["UNKNOWN"],
        "energy:small_cap": ["UNKNOWN", "UNPROVEN"],
    }
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, set())
    state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    first = _terminal_data_gap_payload(
        state,
        cell_id="energy:micro_cap",
        missing_valuation=["UNKNOWN"],
        sparse_history=[],
        structural_screened=[],
        projected_tickers=["UNKNOWN"],
        projected_bands=["small_cap"],
    )
    _state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=first,
    )
    second = _terminal_data_gap_payload(
        state,
        cell_id="energy:small_cap",
        missing_valuation=[],
        sparse_history=[],
        structural_screened=[],
    )
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=second,
    )

    with pytest.raises(driver.ClassicScanPlanError, match="exactly partition its frozen cell"):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_rejects_projection_borrowed_from_prior_cell(
    monkeypatch, tmp_path
):
    pending = {
        "energy:micro_cap": ["PRIOR"],
        "metals_mining:micro_cap": ["CURRENT"],
        "metals_mining:small_cap": ["PRIOR"],
    }
    summary = _plan_dynamic_campaign(monkeypatch, tmp_path, pending, set())
    state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    first = _terminal_data_gap_payload(
        state,
        cell_id="energy:micro_cap",
        missing_valuation=["PRIOR"],
        sparse_history=[],
        structural_screened=[],
    )
    _state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=first,
    )
    second = _terminal_data_gap_payload(
        state,
        cell_id="metals_mining:micro_cap",
        missing_valuation=["CURRENT"],
        sparse_history=[],
        structural_screened=[],
        projected_tickers=["PRIOR"],
        projected_bands=["small_cap"],
    )
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=second,
    )

    with pytest.raises(driver.ClassicScanPlanError, match="projection evidence is malformed"):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_rejects_decision_claim_in_needs_data(
    monkeypatch, tmp_path
):
    summary = _plan_dynamic_campaign(
        monkeypatch,
        tmp_path,
        {"energy:micro_cap": ["AAA"]},
        set(),
    )
    state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    payload = _terminal_data_gap_payload(
        state,
        cell_id="energy:micro_cap",
        missing_valuation=["AAA"],
        sparse_history=[],
        structural_screened=[],
    )
    payload["candidate_selection"]["valuation_anchor_filter"]["dispositions"][0][
        "underwriting_verdict"
    ] = "BUY"
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=payload,
    )

    with pytest.raises(driver.ClassicScanPlanError, match="terminal evidence is not exact"):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_rejects_boolean_count_literal(
    monkeypatch, tmp_path
):
    summary = _plan_dynamic_campaign(
        monkeypatch,
        tmp_path,
        {"energy:micro_cap": ["AAA"]},
        set(),
    )
    state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    payload = _terminal_data_gap_payload(
        state,
        cell_id="energy:micro_cap",
        missing_valuation=["AAA"],
        sparse_history=[],
        structural_screened=[],
    )
    payload["tool_calls"] = False
    state_path, state = _attach_terminal_data_gap_attempt(
        summary,
        output_root=tmp_path,
        payload=payload,
    )

    with pytest.raises(driver.ClassicScanPlanError, match="exact pre-provider refusal"):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_terminal_data_gap_reconciliation_rejects_held_campaign_lock(
    monkeypatch, tmp_path
):
    summary = _plan_dynamic_campaign(
        monkeypatch,
        tmp_path,
        {"energy:micro_cap": ["AAA"]},
        set(),
    )
    lock_path = tmp_path / summary["campaign_id"] / ".campaign.lock"

    with driver._exclusive_lock(lock_path):
        with pytest.raises(driver.ClassicScanLocked, match="lock is already held"):
            driver.reconcile_terminal_data_gaps(
                campaign_id=summary["campaign_id"],
                expect_plan_sha256=summary["plan_sha256"],
                expect_state_sha256="0" * 64,
                expect_target_tickers_sha256=summary["target_tickers_sha256"],
                artifact_path=tmp_path / "must-not-exist.json",
                output_root=tmp_path,
            )


@pytest.mark.parametrize(
    ("state_status", "unresolved_cost", "expected_error"),
    [
        ("RUNNING", 0.0, "running campaign evidence"),
        ("RECONCILIATION_REQUIRED", 0.25, "unresolved paid work"),
    ],
)
def test_terminal_data_gap_reconciliation_rejects_active_or_unsettled_campaign(
    monkeypatch,
    tmp_path,
    state_status,
    unresolved_cost,
    expected_error,
):
    summary = _plan_dynamic_campaign(
        monkeypatch,
        tmp_path,
        {"energy:micro_cap": ["AAA"]},
        set(),
    )
    state_path = Path(summary["state_path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = state_status
    if unresolved_cost:
        state["attempts"] = [
            {
                "attempt_number": 1,
                "cell_id": "energy:micro_cap",
                "sector": "energy",
                "band": "micro_cap",
                "status": "UNRESOLVED_ATTEMPT",
                "live_uncovered_before": ["AAA"],
                "reserved_cost_usd": unresolved_cost,
                "cost_reconciled": False,
            }
        ]
        state["unresolved_cost_reservation_usd"] = unresolved_cost
    driver._atomic_write_json(state_path, state)
    driver._validate_state_integrity(state)

    with pytest.raises(driver.ClassicScanPlanError, match=expected_error):
        driver.reconcile_terminal_data_gaps(
            campaign_id=state["campaign_id"],
            expect_plan_sha256=state["plan_sha256"],
            expect_state_sha256=driver._raw_file_sha256(state_path),
            expect_target_tickers_sha256=state["target_tickers_sha256"],
            artifact_path=tmp_path / "must-not-exist.json",
            output_root=tmp_path,
        )


def test_cli_registers_terminal_data_gap_reconciliation(monkeypatch, tmp_path):
    captured = {}
    artifact_path = tmp_path / "terminal-data-gaps.json"
    monkeypatch.setattr(
        driver,
        "reconcile_terminal_data_gaps",
        lambda **kwargs: captured.update(kwargs)
        or {"status": "TERMINAL_ACCOUNTING_COMPLETE_COVERAGE_INCOMPLETE"},
    )

    result = CliRunner().invoke(
        app,
        [
            "classic-scan",
            "reconcile-data-gaps",
            "--campaign-id",
            "classic_missed_20260804T022926Z_f70ba737",
            "--expect-plan-sha256",
            "1" * 64,
            "--expect-state-sha256",
            "2" * 64,
            "--expect-target-tickers-sha256",
            "3" * 64,
            "--output",
            str(artifact_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "status": "TERMINAL_ACCOUNTING_COMPLETE_COVERAGE_INCOMPLETE"
    }
    assert captured == {
        "campaign_id": "classic_missed_20260804T022926Z_f70ba737",
        "expect_plan_sha256": "1" * 64,
        "expect_state_sha256": "2" * 64,
        "expect_target_tickers_sha256": "3" * 64,
        "artifact_path": artifact_path,
    }


def test_cli_registers_classic_scan_plan(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        driver,
        "plan_campaign",
        lambda **kwargs: captured.update(kwargs) or {
            "status": "PLANNED",
            "campaign_id": "classic_missed_20260722T120000Z_deadbeef",
        },
    )
    result = CliRunner().invoke(
        app,
        [
            "classic-scan",
            "plan",
            "--mode",
            "missed-only",
            "--resume-from-campaign-id",
            "classic_missed_20260722T120000Z_deadbeef",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "PLANNED"
    assert (
        captured["resume_from_campaign_id"]
        == "classic_missed_20260722T120000Z_deadbeef"
    )
