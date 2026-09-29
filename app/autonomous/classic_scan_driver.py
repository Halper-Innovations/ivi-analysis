"""Crash-safe orchestration for classic v1 missed-only and full scans.

Planning performs no LLM call.  It freezes an exact ticker target, date,
provider/model binding, scanner source contract, and 170-cell membership
manifest.  Paid execution is campaign-scoped, fail-closed on drift, and uses
coordinator-owned bounded waves with persisted pre-launch cost reservations.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterator, Sequence
from uuid import uuid4

from app.autonomous.candidate_review import provider_usage_attestation
from app.autonomous.sweep_delta import (
    CANONICAL_SWEEP_SECTORS,
    V1_ATOMIC_BANDS,
    backfill_loaded_sets_from_artifacts,
    full_universe_delta_report,
    split_loaded_coverage,
)
from app.config import get_config
from app.db import get_db
from app.llm.synthesis_agent import _estimate_cost_usd


CLASSIC_SCAN_SCHEMA_VERSION = "classic_scan_campaign_v4"
_SUPPORTED_CLASSIC_SCAN_SCHEMA_VERSIONS = {
    "classic_scan_campaign_v2",
    "classic_scan_campaign_v3",
    CLASSIC_SCAN_SCHEMA_VERSION,
}
CLASSIC_SCAN_MODES = ("missed-only", "full-rescan")
_CAMPAIGN_ID_RE = re.compile(r"^classic_(?:missed|full)_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{8}$")
_DEFAULT_CELL_TIMEOUT_SECONDS = 7_200
_CELL_WORKERS_ENV = "VOE_CLASSIC_CELL_WORKERS"
_MIN_CELL_WORKERS = 1
_MAX_CELL_WORKERS = 4
_COST_SAFETY_MULTIPLIER = 1.50
_FALLBACK_BASE_COST_PER_CELL_USD = 0.05
_FALLBACK_COST_PER_REVIEW_USD = 0.05
_SOURCE_CONTRACT_VERSION = "classic_scan_source_contract_v4"
_TERMINAL_DATA_GAP_RECONCILIATION_SCHEMA_VERSION = (
    "classic_terminal_data_gap_reconciliation_v1"
)
_SOURCE_CONTRACT_FILES = (
    "app/autonomous/classic_scan_driver.py",
    "app/autonomous/candidate_review.py",
    "app/autonomous/runtime.py",
    "app/autonomous/sector_candidates.py",
    "app/autonomous/sector_runtime.py",
    "app/autonomous/sweep_delta.py",
    "app/cli.py",
    "app/llm/providers/retry_guard.py",
    "app/llm/providers/deepseek_provider.py",
    "app/llm/providers/__init__.py",
    "app/llm/synthesis_agent.py",
    "app/llm/usage_capture.py",
    "app/sector/scan.py",
)
_BLOCKED_RUN_STATUSES = {
    "BLOCKED",
    "AUTHORIZED_COST_BREACH",
    "CELL_COST_RESERVATION_BREACH",
    "PROVIDER_MODEL_DRIFT",
    "SOURCE_CONTRACT_DRIFT",
    "STALE_PLAN",
}


class ClassicScanError(RuntimeError):
    """Base fail-closed campaign error."""


class ClassicScanLocked(ClassicScanError):
    """Another paid classic campaign is already running."""


class ClassicScanAuthorizationError(ClassicScanError):
    """Paid execution was not authorized against the exact plan contract."""


class ClassicScanPlanError(ClassicScanError):
    """The current manifest cannot safely authorize paid execution."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalized_tickers(values: Sequence[Any]) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower().replace("_", "-")
    normalized = {
        "missed": "missed-only",
        "catch-up": "missed-only",
        "catchup": "missed-only",
        "full": "full-rescan",
    }.get(normalized, normalized)
    if normalized not in CLASSIC_SCAN_MODES:
        raise ValueError("mode must be missed-only or full-rescan")
    return normalized


def _effective_as_of(value: str | None) -> str:
    normalized = str(value or "").strip() or datetime.now(timezone.utc).date().isoformat()
    try:
        return date.fromisoformat(normalized).isoformat()
    except ValueError as exc:
        raise ValueError("as-of must be YYYY-MM-DD") from exc


def _validate_campaign_id(campaign_id: str) -> str:
    normalized = str(campaign_id or "").strip()
    if not _CAMPAIGN_ID_RE.fullmatch(normalized):
        raise ValueError(f"invalid classic scan campaign id: {campaign_id!r}")
    return normalized


def _configured_provider_binding() -> dict[str, Any]:
    cfg = get_config()
    provider = str(cfg.llm_provider or "disabled").strip().lower()
    if provider == "anthropic":
        model = str(cfg.anthropic_model or "").strip()
        credential_present = bool(cfg.anthropic_api_key)
        max_output_tokens = int(cfg.anthropic_max_output_tokens)
    elif provider == "openai":
        model = str(cfg.openai_model or "").strip()
        credential_present = bool(cfg.openai_api_key)
        max_output_tokens = int(cfg.openai_max_output_tokens)
    elif provider == "deepseek":
        model = str(cfg.deepseek_model or "").strip()
        credential_present = bool(cfg.deepseek_api_key)
        max_output_tokens = int(cfg.deepseek_max_output_tokens)
    else:
        model = ""
        credential_present = False
        max_output_tokens = None
    return {
        "provider": provider,
        "model": model or "disabled",
        "credential_present": credential_present,
        "ready": provider in {"anthropic", "openai", "deepseek"}
        and bool(model)
        and credential_present,
        "max_output_tokens": max_output_tokens,
        "campaign_provider_policy": "exact_provider_model_strict_no_fallback",
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_contract() -> dict[str, Any]:
    root = _repo_root()
    files: dict[str, str] = {}
    for relative in _SOURCE_CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            raise ClassicScanPlanError(f"scanner contract file is missing: {relative}")
        files[relative] = sha256(path.read_bytes()).hexdigest()
    payload = {
        "version": _SOURCE_CONTRACT_VERSION,
        "files": files,
    }
    payload["sha256"] = _sha256(payload)
    return payload


def _campaign_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else get_config().runs_dir / "classic_scan"


def _campaign_dir(campaign_id: str, output_root: str | Path | None = None) -> Path:
    return _campaign_root(output_root) / _validate_campaign_id(campaign_id)


def _state_path(campaign_id: str, output_root: str | Path | None = None) -> Path:
    return _campaign_dir(campaign_id, output_root) / "campaign_state.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    # The file fsync protects its content; syncing the containing directory
    # makes the atomic rename durable across a power loss as well.
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _resolve_cell_workers(value: int | None) -> int:
    raw: Any = os.getenv(_CELL_WORKERS_ENV, "1") if value is None else value
    if isinstance(raw, bool):
        raise ValueError("cell workers must be an integer from 1 through 4")
    try:
        workers = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell workers must be an integer from 1 through 4") from exc
    if str(raw).strip() != str(workers) or not (_MIN_CELL_WORKERS <= workers <= _MAX_CELL_WORKERS):
        raise ValueError("cell workers must be an integer from 1 through 4")
    return workers


def _read_json_object(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ClassicScanPlanError(f"JSON object required: {path}")
    return payload


def _raw_file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_new_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one new mode-0600 JSON artifact without replacing prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ClassicScanPlanError(
                f"refusing to replace an existing reconciliation artifact: {path}"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _expected_cell_ids() -> list[str]:
    return [f"{sector}:{band}" for band in V1_ATOMIC_BANDS for sector in CANONICAL_SWEEP_SECTORS]


def _manifest_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    expected = _expected_cell_ids()
    observed = [str(cell.get("cell_id") or "") for cell in report.get("cells") or []]
    if report.get("expected_cells") != len(expected):
        blockers.append("EXPECTED_CELL_COUNT_MISMATCH")
    if report.get("cells_resolved") != len(expected):
        blockers.append("RESOLVED_CELL_COUNT_MISMATCH")
    if len(observed) != len(set(observed)):
        blockers.append("DUPLICATE_CELL_ID")
    if set(observed) != set(expected):
        blockers.append("CELL_GRID_SET_MISMATCH")
    if report.get("loader_residual_tickers"):
        blockers.append("LOADER_RESIDUAL")
    if report.get("source_error_cells"):
        blockers.append("CANDIDATE_SOURCE_ERROR")
    if report.get("gate_error_cells"):
        blockers.append("STRUCTURAL_GATE_ERROR")
    if report.get("newly_synced_cross_check_violations"):
        blockers.append("SYNC_CROSS_CHECK_VIOLATION")
    if (report.get("membership") or {}).get("membership_unaccounted_tickers"):
        blockers.append("MEMBERSHIP_PARTITION_INCOMPLETE")
    return blockers


def _report_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "generated_at": report.get("generated_at"),
        "as_of": report.get("as_of"),
        "pipeline_version": report.get("pipeline_version"),
        "coverage_campaign_id": report.get("coverage_campaign_id"),
        "allow_carried_verdicts": report.get("allow_carried_verdicts"),
        "target_scope": report.get("target_scope"),
        "target_ticker_count": report.get("target_ticker_count"),
        "target_tickers_sha256": report.get("target_tickers_sha256"),
        "membership_fingerprint": (report.get("membership") or {}).get(
            "current_membership_fingerprint"
        ),
        "report_sha256": _sha256(report),
    }


def _manifest_blocker_inputs(report: dict[str, Any]) -> dict[str, Any]:
    blockers = set(_manifest_blockers(report))
    expected_cell_ids = _expected_cell_ids()
    observed_cell_ids = [str(cell.get("cell_id") or "") for cell in report.get("cells") or []]
    inputs: dict[str, Any] = {}
    if "EXPECTED_CELL_COUNT_MISMATCH" in blockers:
        inputs["EXPECTED_CELL_COUNT_MISMATCH"] = {
            "required_expected_cells": len(expected_cell_ids),
            "reported_expected_cells": report.get("expected_cells"),
        }
    if "RESOLVED_CELL_COUNT_MISMATCH" in blockers:
        inputs["RESOLVED_CELL_COUNT_MISMATCH"] = {
            "required_resolved_cells": len(expected_cell_ids),
            "reported_resolved_cells": report.get("cells_resolved"),
        }
    for blocker in ("DUPLICATE_CELL_ID", "CELL_GRID_SET_MISMATCH"):
        if blocker in blockers:
            inputs[blocker] = {
                "expected_cell_ids": expected_cell_ids,
                "expected_cell_ids_sha256": _sha256(expected_cell_ids),
                "observed_cell_ids": observed_cell_ids,
                "observed_cell_ids_sha256": _sha256(observed_cell_ids),
            }
    if "LOADER_RESIDUAL" in blockers:
        tickers = _normalized_tickers(report.get("loader_residual_tickers") or [])
        inputs["LOADER_RESIDUAL"] = {
            "ticker_count": len(tickers),
            "tickers": tickers,
            "tickers_sha256": _sha256(tickers),
        }
    for blocker, field in (
        ("CANDIDATE_SOURCE_ERROR", "source_error_cells"),
        ("STRUCTURAL_GATE_ERROR", "gate_error_cells"),
        ("SYNC_CROSS_CHECK_VIOLATION", "newly_synced_cross_check_violations"),
    ):
        if blocker in blockers:
            values = report.get(field) or []
            inputs[blocker] = {
                "items": values,
                "item_count": len(values),
                "items_sha256": _sha256(values),
            }
    if "MEMBERSHIP_PARTITION_INCOMPLETE" in blockers:
        tickers = _normalized_tickers(
            (report.get("membership") or {}).get("membership_unaccounted_tickers") or []
        )
        inputs["MEMBERSHIP_PARTITION_INCOMPLETE"] = {
            "ticker_count": len(tickers),
            "tickers": tickers,
            "tickers_sha256": _sha256(tickers),
        }
    return inputs


def _preflight_report_evidence(
    report: dict[str, Any],
    *,
    role: str,
    ignored_blockers: Sequence[str] = (),
    loader_residual_applied: bool = True,
) -> dict[str, Any]:
    observed = sorted(set(_manifest_blockers(report)))
    ignored = sorted(set(ignored_blockers) & set(observed))
    effective = sorted(set(observed) - set(ignored))
    raw_loader_residual = report.get("loader_residual_tickers")
    if not isinstance(raw_loader_residual, list):
        raise ClassicScanPlanError("preflight loader residual must be a ticker list")
    loader_residual = _normalized_tickers(raw_loader_residual)
    if raw_loader_residual != loader_residual:
        raise ClassicScanPlanError("preflight loader residual ticker list is not canonical")
    return {
        "role": role,
        "report_identity": _report_identity(report),
        "observed_blockers": observed,
        "effective_blockers": effective,
        "ignored_blockers": ignored,
        "loader_residual": {
            "applied_to_gate": loader_residual_applied,
            "ticker_count": len(loader_residual),
            "tickers": loader_residual,
            "tickers_sha256": _sha256(loader_residual),
        },
        "blocker_inputs": _manifest_blocker_inputs(report),
    }


def _preflight_evidence(
    *,
    lifetime_report: dict[str, Any],
    target_report: dict[str, Any],
    target_scoped_loader_residual: bool,
    provider_binding: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    lifetime = _preflight_report_evidence(
        lifetime_report,
        role="lifetime_full_universe",
        ignored_blockers=("LOADER_RESIDUAL",) if target_scoped_loader_residual else (),
        loader_residual_applied=not target_scoped_loader_residual,
    )
    target = _preflight_report_evidence(target_report, role="frozen_campaign_target")
    report_blockers = sorted(
        set(lifetime["effective_blockers"]) | set(target["effective_blockers"])
    )
    provider_blockers = [] if provider_binding.get("ready") is True else ["LLM_PROVIDER_NOT_READY"]
    blockers = sorted(set(report_blockers) | set(provider_blockers))
    return blockers, {
        "schema_version": "classic_scan_preflight_evidence_v1",
        "loader_residual_policy": (
            "frozen_missed_only_target_scope"
            if target_scoped_loader_residual
            else "full_universe_lifetime_and_target"
        ),
        "effective_blockers": blockers,
        "reports": [lifetime, target],
        "provider_binding_evaluation": {
            "effective_blockers": provider_blockers,
            "inputs": provider_binding,
        },
    }


def _review_work_units(report: dict[str, Any]) -> dict[str, int]:
    occurrences: dict[str, list[bool]] = {}
    active_cells = 0
    for cell in report.get("cells") or []:
        to_review = _normalized_tickers(cell.get("to_review_tickers") or [])
        if to_review:
            active_cells += 1
        unknown = set(_normalized_tickers(cell.get("unknown_cap_tickers") or []))
        for ticker in to_review:
            occurrences.setdefault(ticker, []).append(ticker in unknown)
    payable = avoided = 0
    for rows in occurrences.values():
        if len(rows) > 1 and all(rows):
            payable += 1
            avoided += len(rows) - 1
        else:
            payable += len(rows)
    return {
        "review_occurrences": sum(len(rows) for rows in occurrences.values()),
        "distinct_review_tickers": len(occurrences),
        "payable_review_units": payable,
        "unknown_cap_duplicate_reviews_avoided": avoided,
        "active_review_cells_before_projection": active_cells,
    }


def _artifact_cost_samples(
    *, provider: str, model: str, limit: int = 150
) -> list[tuple[float, int]]:
    artifacts = sorted(
        get_config().runs_dir.glob("autonomous_sector/*/autonomous_sector_run.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    samples: list[tuple[float, int]] = []
    for path in artifacts:
        if len(samples) >= limit:
            break
        try:
            artifact = _read_json_object(path)
        except Exception:
            continue
        if str(artifact.get("pipeline_version") or "v1").lower() != "v1":
            continue
        if str(artifact.get("status") or "").upper() != "COMPLETED":
            continue
        packets = artifact.get("company_packets") or []
        if not packets:
            continue
        attestation = provider_usage_attestation(artifact)
        bindings = {
            (str(row["provider"]), str(row["model"])) for row in attestation["provider_models"]
        }
        cost = float(attestation["cost_estimate_usd"])
        if not attestation["valid"]:
            continue
        if bindings == {(provider, model)} and cost > 0:
            samples.append((cost, len(packets)))
            continue
        if provider != "deepseek" or model != "deepseek-v4-pro":
            continue
        repriced_cost = 0.0
        seen: set[str] = set()

        # ``seen`` is bound as a default so each artifact's recursion closes over
        # its own dedup set rather than whichever set the loop last rebound.
        def visit(value: Any, seen: set[str] = seen) -> None:
            nonlocal repriced_cost
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "provider_usage" and isinstance(child, list):
                        for row in child:
                            if not isinstance(row, dict):
                                continue
                            fingerprint = _sha256(row)
                            if fingerprint in seen:
                                continue
                            seen.add(fingerprint)
                            input_tokens = row.get("input_tokens")
                            output_tokens = row.get("output_tokens")
                            reserved_tokens = row.get("reserved_output_tokens")
                            if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
                                continue
                            if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
                                output_tokens = 0
                            if isinstance(reserved_tokens, int) and not isinstance(reserved_tokens, bool):
                                output_tokens += max(0, reserved_tokens)
                            repriced_cost += _estimate_cost_usd(
                                model,
                                input_tokens,
                                output_tokens,
                                provider_name=provider,
                                # Treat all historical input as cache misses so
                                # the new-provider projection stays conservative.
                                cached_input_tokens=0,
                            )
                    else:
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(artifact)
        if repriced_cost > 0:
            samples.append((round(repriced_cost, 6), len(packets)))
    return samples


def _cost_estimate(*, provider: str, model: str, work: dict[str, int]) -> dict[str, Any]:
    samples = _artifact_cost_samples(provider=provider, model=model)
    if len(samples) >= 5:
        xs = [float(count) for _cost, count in samples]
        ys = [float(cost) for cost, _count in samples]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((value - mean_x) ** 2 for value in xs)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
            if denominator > 0
            else mean_y / max(1.0, mean_x)
        )
        per_review = max(0.005, slope)
        per_cell = max(
            0.005,
            median(max(0.0, y - per_review * x) for x, y in zip(xs, ys, strict=True)),
        )
        method = (
            "token_repriced_recursive_history_with_50pct_reserve"
            if provider == "deepseek" and model == "deepseek-v4-pro"
            else "recursive_physical_usage_history_with_50pct_reserve"
        )
    else:
        per_review = _FALLBACK_COST_PER_REVIEW_USD
        per_cell = _FALLBACK_BASE_COST_PER_CELL_USD
        method = "no_matching_history_conservative_fallback_with_50pct_reserve"
    raw = (
        per_review * work["payable_review_units"]
        + per_cell * work["active_review_cells_before_projection"]
    )
    return {
        "estimated_cost_usd": round(raw * _COST_SAFETY_MULTIPLIER, 2),
        "method": method,
        "matching_historical_runs": len(samples),
        "estimated_cost_per_review_usd": round(per_review, 6),
        "estimated_base_cost_per_active_cell_usd": round(per_cell, 6),
        "safety_multiplier": _COST_SAFETY_MULTIPLIER,
        **work,
    }


def _target_payload(
    *, campaign_id: str, mode: str, as_of: str, tickers: list[str]
) -> dict[str, Any]:
    normalized = _normalized_tickers(tickers)
    return {
        "schema_version": "classic_scan_target_v1",
        "campaign_id": campaign_id,
        "mode": mode,
        "effective_as_of": as_of,
        "ticker_count": len(normalized),
        "tickers_sha256": _sha256(normalized),
        "tickers": normalized,
    }


def _read_target_payload(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    tickers = _normalized_tickers(payload.get("tickers") or [])
    if payload.get("schema_version") != "classic_scan_target_v1":
        raise ClassicScanPlanError("unsupported classic coverage target schema")
    if payload.get("ticker_count") != len(tickers):
        raise ClassicScanPlanError("coverage target count is corrupt")
    if payload.get("tickers_sha256") != _sha256(tickers):
        raise ClassicScanPlanError("coverage target hash is corrupt")
    payload["tickers"] = tickers
    return payload


def _plan_contract(state: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "schema_version": state["schema_version"],
        "campaign_id": state["campaign_id"],
        "mode": state["mode"],
        "pipeline_version": state["pipeline_version"],
        "coverage_campaign_id": state["coverage_campaign_id"],
        "allow_carried_verdicts": state["allow_carried_verdicts"],
        "populate_watchlist": state["populate_watchlist"],
        "membership_fingerprint": state["membership_fingerprint"],
        "target_ticker_count": state["target_ticker_count"],
        "target_tickers_sha256": state["target_tickers_sha256"],
        "target_file_sha256": state["target_file_sha256"],
        "provider_binding": state["provider_binding"],
        "effective_as_of": state["effective_as_of"],
        "source_contract": state["source_contract"],
        "cell_contracts": [
            {
                "cell_id": cell["cell_id"],
                "loaded_tickers_sha256": cell["loaded_tickers_sha256"],
                "initial_uncovered_tickers_sha256": cell["initial_uncovered_tickers_sha256"],
                "initial_to_review_tickers_sha256": cell["initial_to_review_tickers_sha256"],
            }
            for cell in state["cells"]
        ],
        "cost_estimate": state["cost_estimate"],
    }
    if state["schema_version"] in {
        "classic_scan_campaign_v3",
        "classic_scan_campaign_v4",
    }:
        contract["resume_contract"] = state.get("resume_contract")
    if state["schema_version"] == "classic_scan_campaign_v4":
        contract["preflight_blockers"] = state.get("preflight_blockers")
        contract["preflight_evidence"] = state.get("preflight_evidence")
    return contract


def _canonical_blockers(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ClassicScanPlanError(f"{label} must be a blocker list")
    normalized = sorted({str(item).strip() for item in value if str(item).strip()})
    if value != normalized:
        raise ClassicScanPlanError(f"{label} is not canonical")
    return normalized


def _validate_counted_tickers(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ClassicScanPlanError(f"{label} must be an object")
    tickers = value.get("tickers")
    if not isinstance(tickers, list) or tickers != _normalized_tickers(tickers):
        raise ClassicScanPlanError(f"{label} tickers are not canonical")
    if value.get("ticker_count") != len(tickers):
        raise ClassicScanPlanError(f"{label} ticker count is corrupt")
    if value.get("tickers_sha256") != _sha256(tickers):
        raise ClassicScanPlanError(f"{label} ticker hash is corrupt")


def _validate_preflight_evidence(state: dict[str, Any]) -> None:
    evidence = state.get("preflight_evidence")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != (
        "classic_scan_preflight_evidence_v1"
    ):
        raise ClassicScanPlanError("campaign preflight evidence is missing or unsupported")
    state_blockers = _canonical_blockers(
        state.get("preflight_blockers"), label="campaign preflight blockers"
    )
    evidence_blockers = _canonical_blockers(
        evidence.get("effective_blockers"), label="preflight evidence blockers"
    )
    if evidence_blockers != state_blockers:
        raise ClassicScanPlanError("preflight evidence does not match campaign blockers")

    target_scoped = state.get("mode") == "missed-only"
    expected_policy = (
        "frozen_missed_only_target_scope" if target_scoped else "full_universe_lifetime_and_target"
    )
    if evidence.get("loader_residual_policy") != expected_policy:
        raise ClassicScanPlanError("campaign loader-residual policy is corrupt")
    reports = evidence.get("reports")
    if (
        not isinstance(reports, list)
        or not all(isinstance(row, dict) for row in reports)
        or [row.get("role") for row in reports]
        != [
            "lifetime_full_universe",
            "frozen_campaign_target",
        ]
    ):
        raise ClassicScanPlanError("campaign preflight report roles are corrupt")

    report_effective: set[str] = set()
    for row in reports:
        if not isinstance(row, dict):
            raise ClassicScanPlanError("campaign preflight report evidence is corrupt")
        observed = _canonical_blockers(
            row.get("observed_blockers"), label="observed report blockers"
        )
        effective = _canonical_blockers(
            row.get("effective_blockers"), label="effective report blockers"
        )
        ignored = _canonical_blockers(row.get("ignored_blockers"), label="ignored report blockers")
        if set(effective) & set(ignored) or sorted(set(effective) | set(ignored)) != observed:
            raise ClassicScanPlanError("campaign preflight blocker disposition is corrupt")
        if row["role"] == "lifetime_full_universe":
            expected_ignored = (
                ["LOADER_RESIDUAL"] if target_scoped and ("LOADER_RESIDUAL" in observed) else []
            )
            if ignored != expected_ignored:
                raise ClassicScanPlanError("lifetime loader-residual scope is corrupt")
        elif ignored:
            raise ClassicScanPlanError("target-scoped preflight blockers cannot be ignored")
        report_effective.update(effective)

        identity = row.get("report_identity")
        if not isinstance(identity, dict):
            raise ClassicScanPlanError("campaign preflight report identity is corrupt")
        if identity.get("schema_version") != "classic_full_universe_delta_v1":
            raise ClassicScanPlanError("campaign preflight report schema is corrupt")
        if identity.get("pipeline_version") != "v1":
            raise ClassicScanPlanError("campaign preflight report pipeline is corrupt")
        if identity.get("as_of") != state.get("effective_as_of"):
            raise ClassicScanPlanError("campaign preflight report date is corrupt")
        if not str(identity.get("generated_at") or "").strip():
            raise ClassicScanPlanError("campaign preflight report timestamp is missing")
        if not str(identity.get("target_scope") or "").strip():
            raise ClassicScanPlanError("campaign preflight target scope is missing")
        if identity.get("membership_fingerprint") != state.get("membership_fingerprint"):
            raise ClassicScanPlanError("campaign preflight membership identity is corrupt")
        if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("report_sha256") or "")):
            raise ClassicScanPlanError("campaign preflight report hash is corrupt")
        if row["role"] == "lifetime_full_universe":
            if (
                identity.get("coverage_campaign_id") is not None
                or (identity.get("allow_carried_verdicts") is not True)
                or identity.get("target_scope") != "all_eligible"
            ):
                raise ClassicScanPlanError("lifetime preflight report scope is corrupt")
        elif (
            identity.get("coverage_campaign_id") != state.get("coverage_campaign_id")
            or identity.get("allow_carried_verdicts") is not False
            or identity.get("target_scope") != "frozen_ticker_set"
            or identity.get("target_ticker_count") != state.get("target_ticker_count")
            or identity.get("target_tickers_sha256") != state.get("target_tickers_sha256")
        ):
            raise ClassicScanPlanError("target preflight report scope is corrupt")

        loader = row.get("loader_residual")
        _validate_counted_tickers(loader, label="preflight loader residual")
        expected_applied = not (row["role"] == "lifetime_full_universe" and target_scoped)
        if loader.get("applied_to_gate") is not expected_applied:
            raise ClassicScanPlanError("preflight loader-residual application is corrupt")
        blocker_inputs = row.get("blocker_inputs")
        if not isinstance(blocker_inputs, dict) or sorted(blocker_inputs) != observed:
            raise ClassicScanPlanError("campaign preflight blocker inputs are incomplete")
        if "LOADER_RESIDUAL" in blocker_inputs:
            loader_input = blocker_inputs["LOADER_RESIDUAL"]
            if loader_input != {
                "ticker_count": loader["ticker_count"],
                "tickers": loader["tickers"],
                "tickers_sha256": loader["tickers_sha256"],
            }:
                raise ClassicScanPlanError("loader-residual blocker inputs are corrupt")
        for blocker, value in blocker_inputs.items():
            if not isinstance(value, dict):
                raise ClassicScanPlanError("campaign preflight blocker input is corrupt")
            if blocker in {
                "LOADER_RESIDUAL",
                "EXPECTED_CELL_COUNT_MISMATCH",
                "RESOLVED_CELL_COUNT_MISMATCH",
            }:
                continue
            if blocker in {"DUPLICATE_CELL_ID", "CELL_GRID_SET_MISMATCH"}:
                if value.get("expected_cell_ids_sha256") != _sha256(
                    value.get("expected_cell_ids")
                ) or value.get("observed_cell_ids_sha256") != _sha256(
                    value.get("observed_cell_ids")
                ):
                    raise ClassicScanPlanError("cell-grid blocker inputs are corrupt")
            elif blocker == "MEMBERSHIP_PARTITION_INCOMPLETE":
                _validate_counted_tickers(value, label="membership blocker inputs")
            else:
                items = value.get("items")
                if (
                    not isinstance(items, list)
                    or value.get("item_count") != len(items)
                    or value.get("items_sha256") != _sha256(items)
                ):
                    raise ClassicScanPlanError("preflight blocker item inputs are corrupt")

    provider = evidence.get("provider_binding_evaluation")
    if not isinstance(provider, dict) or provider.get("inputs") != state.get("provider_binding"):
        raise ClassicScanPlanError("preflight provider evidence is corrupt")
    provider_blockers = _canonical_blockers(
        provider.get("effective_blockers"), label="preflight provider blockers"
    )
    expected_provider_blockers = (
        []
        if (state.get("provider_binding") or {}).get("ready") is True
        else ["LLM_PROVIDER_NOT_READY"]
    )
    if provider_blockers != expected_provider_blockers:
        raise ClassicScanPlanError("preflight provider blocker is corrupt")
    if sorted(report_effective | set(provider_blockers)) != evidence_blockers:
        raise ClassicScanPlanError("campaign effective preflight blockers are corrupt")


def _runtime_planned_contract(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_contract_sha256": state["source_contract"]["sha256"],
        "membership_fingerprint": state["membership_fingerprint"],
        "target_ticker_count": state["target_ticker_count"],
        "target_tickers_sha256": state["target_tickers_sha256"],
        "cell_contracts": [
            {
                "cell_id": cell["cell_id"],
                "loaded_tickers_sha256": cell["loaded_tickers_sha256"],
            }
            for cell in state["cells"]
        ],
    }


def _validate_runtime_preflight_failure(state: dict[str, Any]) -> None:
    record = state.get("latest_preflight_failure")
    if record is None:
        return
    if not isinstance(record, dict) or record.get("schema_version") != (
        "classic_scan_runtime_preflight_failure_v1"
    ):
        raise ClassicScanPlanError("runtime preflight failure evidence is unsupported")
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != _sha256(unsigned):
        raise ClassicScanPlanError("runtime preflight failure evidence hash is corrupt")
    if (
        state.get("status") != "STALE_PLAN"
        or not str(record.get("recorded_at") or "").strip()
        or record.get("error") != state.get("stop_reason")
    ):
        raise ClassicScanPlanError("runtime preflight failure disposition is corrupt")
    latest_report = state.get("latest_report")
    if not isinstance(latest_report, dict) or record.get("report") != (
        _preflight_report_evidence(
            latest_report,
            role="runtime_frozen_campaign_target",
        )
    ):
        raise ClassicScanPlanError("runtime preflight failure report is corrupt")
    if record.get("planned_contract") != _runtime_planned_contract(state):
        raise ClassicScanPlanError("runtime preflight planned contract is corrupt")
    current_source = record.get("current_source_contract")
    if not isinstance(current_source, dict) or set(current_source) not in (
        {"sha256"},
        {"error"},
    ):
        raise ClassicScanPlanError("runtime preflight source evidence is corrupt")
    if "sha256" in current_source and not re.fullmatch(
        r"[0-9a-f]{64}", str(current_source["sha256"])
    ):
        raise ClassicScanPlanError("runtime preflight source hash is corrupt")
    if "error" in current_source and not str(current_source["error"] or "").strip():
        raise ClassicScanPlanError("runtime preflight source error is corrupt")


def _validate_state_integrity(state: dict[str, Any]) -> None:
    if state.get("schema_version") not in _SUPPORTED_CLASSIC_SCAN_SCHEMA_VERSIONS:
        raise ClassicScanPlanError("unsupported classic scan campaign schema")
    campaign_id = _validate_campaign_id(str(state.get("campaign_id") or ""))
    if state.get("mode") not in CLASSIC_SCAN_MODES:
        raise ClassicScanPlanError("campaign mode is corrupt")
    if state.get("pipeline_version") != "v1":
        raise ClassicScanPlanError("classic campaign pipeline must remain v1")
    if state.get("coverage_campaign_id") != campaign_id:
        raise ClassicScanPlanError("campaign coverage scope is corrupt")
    if state.get("allow_carried_verdicts") is not False:
        raise ClassicScanPlanError("classic campaigns must require fresh target reviews")
    _effective_as_of(str(state.get("effective_as_of") or ""))
    cells = state.get("cells") or []
    cell_ids = [str(cell.get("cell_id") or "") for cell in cells]
    if cell_ids != _expected_cell_ids():
        raise ClassicScanPlanError("campaign cell grid is corrupt")
    valid_cell_ids = set(cell_ids)
    target_path = Path(str(state.get("target_file") or ""))
    target = _read_target_payload(target_path)
    if target.get("campaign_id") != campaign_id or target.get("mode") != state.get("mode"):
        raise ClassicScanPlanError("coverage target belongs to another campaign")
    if target.get("effective_as_of") != state.get("effective_as_of"):
        raise ClassicScanPlanError("coverage target date does not match campaign")
    if target.get("ticker_count") != state.get("target_ticker_count"):
        raise ClassicScanPlanError("coverage target count does not match campaign")
    if target.get("tickers_sha256") != state.get("target_tickers_sha256"):
        raise ClassicScanPlanError("coverage target ticker hash does not match campaign")
    if _sha256(target) != state.get("target_file_sha256"):
        raise ClassicScanPlanError("coverage target file hash does not match campaign")
    if state.get("schema_version") == "classic_scan_campaign_v4":
        _validate_preflight_evidence(state)
        _validate_runtime_preflight_failure(state)
    if _sha256(_plan_contract(state)) != state.get("plan_sha256"):
        raise ClassicScanPlanError("campaign plan hash is corrupt or stale")

    attempts = state.get("attempts") or []
    charged = 0.0
    unresolved = 0.0
    unresolved_cells: set[str] = set()
    unresolved_tickers: set[str] = set()
    for index, attempt in enumerate(attempts, start=1):
        if attempt.get("attempt_number") != index:
            raise ClassicScanPlanError("campaign attempt ordering is corrupt")
        attempt_cell_id = str(attempt.get("cell_id") or "")
        if attempt_cell_id not in valid_cell_ids:
            raise ClassicScanPlanError("campaign attempt references an invalid cell")
        if "cell_workers" in attempt:
            _resolve_cell_workers(attempt.get("cell_workers"))
        claims = _normalized_tickers(attempt.get("live_uncovered_before") or [])
        claimed_tickers_sha256 = attempt.get("claimed_tickers_sha256")
        if claimed_tickers_sha256 is None and "cell_workers" in attempt:
            raise ClassicScanPlanError("parallel campaign attempt is missing its ticker claim hash")
        if claimed_tickers_sha256 is not None and claimed_tickers_sha256 != _sha256(claims):
            raise ClassicScanPlanError("campaign attempt ticker claim hash is corrupt")
        reserve = float(attempt.get("reserved_cost_usd") or 0.0)
        if not math.isfinite(reserve) or reserve < 0:
            raise ClassicScanPlanError("campaign cost reservation is corrupt")
        if bool(attempt.get("cost_reconciled")):
            value = float(attempt.get("charged_cost_usd") or 0.0)
            if not math.isfinite(value) or value < 0:
                raise ClassicScanPlanError("campaign charged cost is corrupt")
            charged += value
        else:
            unresolved += reserve
            if attempt_cell_id in unresolved_cells:
                raise ClassicScanPlanError("unresolved attempts claim the same cell")
            overlap = unresolved_tickers & set(claims)
            if overlap:
                raise ClassicScanPlanError("unresolved attempts have overlapping ticker claims")
            unresolved_cells.add(attempt_cell_id)
            unresolved_tickers.update(claims)
    if abs(charged - float(state.get("cumulative_cost_usd") or 0.0)) > 1e-6:
        raise ClassicScanPlanError("campaign cumulative cost accounting is corrupt")
    if abs(unresolved - float(state.get("unresolved_cost_reservation_usd") or 0.0)) > 1e-6:
        raise ClassicScanPlanError("campaign unresolved reservation accounting is corrupt")


def _load_state(
    campaign_id: str, output_root: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
    path = _state_path(campaign_id, output_root)
    if not path.exists():
        raise FileNotFoundError(f"classic scan campaign state not found: {path}")
    payload = _read_json_object(path)
    if payload.get("campaign_id") != campaign_id:
        raise ClassicScanPlanError("campaign state id does not match its path")
    _validate_state_integrity(payload)
    return path, payload


def _completed_tickers_for_campaign(
    state: dict[str, Any], *, target_tickers: Sequence[Any]
) -> list[str]:
    """Return prior targets with no current residual cell in that campaign.

    A ticker can be terminal in one sector-band cell while remaining uncovered
    in another. Reuse the campaign-scoped report so latest same-cell state and
    coverage-receipt authorization stay authoritative; ticker-level ``ANY``
    completion would silently omit partial multi-cell work from a resume.
    """
    target = _normalized_tickers(target_tickers)
    report = _current_report(state, target_tickers=target)
    expected_cells = len(CANONICAL_SWEEP_SECTORS) * len(V1_ATOMIC_BANDS)
    if (
        report.get("schema_version") != "classic_full_universe_delta_v1"
        or report.get("coverage_campaign_id") != state["coverage_campaign_id"]
        or report.get("allow_carried_verdicts") is not False
        or report.get("target_ticker_count") != len(target)
        or report.get("target_tickers_sha256") != _sha256(target)
        or report.get("expected_cells") != expected_cells
        or report.get("cells_resolved") != expected_cells
    ):
        raise ClassicScanPlanError("resume source coverage report contract is invalid")
    residual_sets: list[set[str]] = []
    for field in ("coverage_residual_tickers", "loader_residual_tickers"):
        values = report.get(field)
        if not isinstance(values, list) or values != _normalized_tickers(values):
            raise ClassicScanPlanError(f"resume source {field} is invalid")
        normalized = set(values)
        if not normalized.issubset(target):
            raise ClassicScanPlanError(f"resume source {field} exceeds its target")
        residual_sets.append(normalized)
    if residual_sets[0] & residual_sets[1]:
        raise ClassicScanPlanError("resume source residual classes overlap")
    return sorted(set(target) - set().union(*residual_sets))


def validate_coverage_target_file(path: str | Path, *, campaign_id: str) -> list[str]:
    """Validate a child-run target against its registered campaign state."""
    normalized_id = _validate_campaign_id(campaign_id)
    target_path = Path(path).resolve()
    if target_path.name != "coverage_targets.json":
        raise ClassicScanPlanError("classic coverage target must be coverage_targets.json")
    state_path = target_path.parent / "campaign_state.json"
    if not state_path.is_file():
        raise ClassicScanPlanError("coverage target has no registered campaign state")
    state = _read_json_object(state_path)
    _validate_state_integrity(state)
    if state.get("campaign_id") != normalized_id:
        raise ClassicScanPlanError("coverage target campaign id mismatch")
    if Path(str(state.get("target_file") or "")).resolve() != target_path:
        raise ClassicScanPlanError("coverage target path is not campaign-authorized")
    return list(_read_target_payload(target_path)["tickers"])


def _campaign_summary(state: dict[str, Any]) -> dict[str, Any]:
    latest = state.get("latest_report") or {}
    attempts = state.get("attempts") or []
    last_attempt = attempts[-1] if attempts else None
    return {
        "schema_version": state["schema_version"],
        "campaign_id": state["campaign_id"],
        "mode": state["mode"],
        "status": state["status"],
        "stop_reason": state.get("stop_reason"),
        "plan_sha256": state["plan_sha256"],
        "source_contract_sha256": state["source_contract"]["sha256"],
        "membership_fingerprint": state["membership_fingerprint"],
        "effective_as_of": state["effective_as_of"],
        "coverage_campaign_id": state["coverage_campaign_id"],
        "allow_carried_verdicts": state["allow_carried_verdicts"],
        "populate_watchlist": state["populate_watchlist"],
        "target_ticker_count": state["target_ticker_count"],
        "target_tickers_sha256": state["target_tickers_sha256"],
        "target_file": state["target_file"],
        "provider_binding": state["provider_binding"],
        "resume_contract": state.get("resume_contract"),
        "preflight_blockers": state.get("preflight_blockers") or [],
        "preflight_evidence": state.get("preflight_evidence"),
        "latest_preflight_failure": state.get("latest_preflight_failure"),
        "cost_estimate": state["cost_estimate"],
        "authorized_max_cost_usd": state.get("authorized_max_cost_usd"),
        "cumulative_cost_usd": state.get("cumulative_cost_usd", 0.0),
        "unresolved_cost_reservation_usd": state.get("unresolved_cost_reservation_usd", 0.0),
        "attempt_count": len(attempts),
        "cell_workers": state.get("last_cell_workers", 1),
        "running_attempts": sum(1 for attempt in attempts if not attempt.get("cost_reconciled")),
        "known_physical_provider_attempts": sum(
            int(
                (attempt.get("provider_usage_attestation") or {}).get("physical_attempt_count") or 0
            )
            for attempt in attempts
        ),
        "last_attempt": last_attempt,
        "stalled_cells": state.get("stalled_cells") or [],
        "n_stalled_cells": len(state.get("stalled_cells") or []),
        "manifest": {
            "expected_cells": latest.get("expected_cells"),
            "cells_resolved": latest.get("cells_resolved"),
            "pending_cells": len(latest.get("pending_cells") or []),
            "eligible_common_equity": (latest.get("membership") or {}).get(
                "eligible_common_equity"
            ),
            "coverage_uncovered_distinct": (latest.get("distinct_ticker_totals") or {}).get(
                "coverage_uncovered"
            ),
            "fresh_review_queue_distinct": (latest.get("distinct_ticker_totals") or {}).get(
                "fresh_review_queue"
            ),
            "complete": latest.get("complete"),
        },
        "state_path": state["state_path"],
    }


def plan_campaign(
    *,
    mode: str,
    as_of: str | None = None,
    backfill_artifacts: bool = False,
    populate_watchlist: bool = True,
    resume_from_campaign_id: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create and persist one zero-LLM, exact-target campaign plan."""
    normalized_mode = _normalize_mode(mode)
    effective_as_of = _effective_as_of(as_of)
    resume_id = (
        _validate_campaign_id(resume_from_campaign_id)
        if resume_from_campaign_id is not None
        else None
    )
    if resume_id is not None and normalized_mode != "missed-only":
        raise ValueError("--resume-from-campaign-id requires --mode missed-only")
    stamp = _utc_now()
    short_mode = "missed" if normalized_mode == "missed-only" else "full"
    campaign_id = (
        f"classic_{short_mode}_{stamp.replace('-', '').replace(':', '')}_{uuid4().hex[:8]}"
    )
    binding = _configured_provider_binding()
    backfill_result = backfill_loaded_sets_from_artifacts() if backfill_artifacts else None
    lifetime_report = full_universe_delta_report(
        as_of=effective_as_of,
        coverage_campaign_id=None,
        allow_carried_verdicts=True,
    )
    eligible = set(
        _normalized_tickers(
            (lifetime_report.get("membership") or {}).get("eligible_common_equity_tickers", [])
        )
    )
    resume_contract: dict[str, Any] | None = None
    if resume_id is not None:
        _prior_state_path, prior_state = _load_state(resume_id, output_root)
        if prior_state["mode"] != "missed-only":
            raise ClassicScanPlanError(
                "resume source must be a missed-only classic scan campaign"
            )
        if prior_state["effective_as_of"] != effective_as_of:
            raise ClassicScanPlanError(
                "resume source effective_as_of does not match the requested plan"
            )
        if (
            prior_state.get("status") == "RUNNING"
            or float(prior_state.get("unresolved_cost_reservation_usd") or 0.0) > 0
            or any(
                not bool(attempt.get("cost_reconciled"))
                and float(attempt.get("reserved_cost_usd") or 0.0) > 0
                for attempt in prior_state.get("attempts") or []
            )
        ):
            raise ClassicScanPlanError(
                "resume source has active or unresolved paid work; reconcile it first"
            )
        prior_target_tickers = _target_tickers(prior_state)
        prior_target = set(prior_target_tickers)
        current_scope = eligible & prior_target
        dropped_tickers = sorted(prior_target - current_scope)
        completed_tickers = sorted(
            current_scope
            & set(
                _completed_tickers_for_campaign(
                    prior_state, target_tickers=sorted(current_scope)
                )
            )
        )
        target_tickers = sorted(current_scope - set(completed_tickers))
        resume_contract = {
            "campaign_id": resume_id,
            "plan_sha256": prior_state["plan_sha256"],
            "target_ticker_count": len(prior_target_tickers),
            "target_tickers_sha256": _sha256(prior_target_tickers),
            "current_scope_ticker_count": len(current_scope),
            "current_scope_tickers_sha256": _sha256(sorted(current_scope)),
            "dropped_current_ineligible_ticker_count": len(dropped_tickers),
            "dropped_current_ineligible_tickers_sha256": _sha256(dropped_tickers),
            "completed_ticker_count": len(completed_tickers),
            "completed_tickers_sha256": _sha256(completed_tickers),
        }
    elif normalized_mode == "full-rescan":
        target_tickers = sorted(eligible)
    else:
        target_tickers = sorted(
            eligible
            & (
                set(_normalized_tickers(lifetime_report.get("coverage_residual_tickers") or []))
                | set(_normalized_tickers(lifetime_report.get("loader_residual_tickers") or []))
            )
        )

    campaign_dir = _campaign_dir(campaign_id, output_root)
    target_path = campaign_dir / "coverage_targets.json"
    target_payload = _target_payload(
        campaign_id=campaign_id,
        mode=normalized_mode,
        as_of=effective_as_of,
        tickers=target_tickers,
    )
    _atomic_write_json(target_path, target_payload)
    report = full_universe_delta_report(
        as_of=effective_as_of,
        coverage_campaign_id=campaign_id,
        allow_carried_verdicts=False,
        target_tickers=target_tickers,
    )
    blockers, preflight_evidence = _preflight_evidence(
        lifetime_report=lifetime_report,
        target_report=report,
        target_scoped_loader_residual=normalized_mode == "missed-only",
        provider_binding=binding,
    )
    work = _review_work_units(report)
    cost_estimate = _cost_estimate(
        provider=str(binding["provider"]), model=str(binding["model"]), work=work
    )
    state_path = campaign_dir / "campaign_state.json"
    cells = [
        {
            "cell_id": str(cell["cell_id"]),
            "sector": str(cell["sector"]),
            "band": str(cell["band"]),
            "loaded": int(cell["loaded"]),
            "initial_uncovered": int(cell["coverage_uncovered"]),
            "initial_to_review": int(cell["to_review"]),
            "loaded_tickers": _normalized_tickers(cell["loaded_tickers"]),
            "initial_uncovered_tickers": _normalized_tickers(cell["coverage_uncovered_tickers"]),
            "initial_to_review_tickers": _normalized_tickers(cell["to_review_tickers"]),
            "loaded_tickers_sha256": _sha256(_normalized_tickers(cell["loaded_tickers"])),
            "initial_uncovered_tickers_sha256": _sha256(
                _normalized_tickers(cell["coverage_uncovered_tickers"])
            ),
            "initial_to_review_tickers_sha256": _sha256(
                _normalized_tickers(cell["to_review_tickers"])
            ),
        }
        for cell in report["cells"]
    ]
    status = "BLOCKED" if blockers else "COMPLETE" if report["complete"] else "PLANNED"
    state: dict[str, Any] = {
        "schema_version": CLASSIC_SCAN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "mode": normalized_mode,
        "pipeline_version": "v1",
        "created_at": stamp,
        "updated_at": stamp,
        "status": status,
        "stop_reason": ",".join(blockers) if blockers else None,
        "membership_fingerprint": (report.get("membership") or {})[
            "current_membership_fingerprint"
        ],
        "effective_as_of": effective_as_of,
        "provider_binding": binding,
        "resume_contract": resume_contract,
        "source_contract": _source_contract(),
        "populate_watchlist": bool(populate_watchlist),
        "coverage_campaign_id": campaign_id,
        "allow_carried_verdicts": False,
        "target_ticker_count": len(target_tickers),
        "target_tickers_sha256": _sha256(target_tickers),
        "target_file": str(target_path.resolve()),
        "target_file_sha256": _sha256(target_payload),
        "backfill_artifacts": backfill_result,
        "preflight_blockers": blockers,
        "preflight_evidence": preflight_evidence,
        "cost_estimate": cost_estimate,
        "authorized_max_cost_usd": None,
        "cumulative_cost_usd": 0.0,
        "unresolved_cost_reservation_usd": 0.0,
        "authorization_history": [],
        "reconciliation_history": [],
        "attempts": [],
        "cells": cells,
        "latest_report": report,
        "state_path": str(state_path.resolve()),
    }
    state["plan_sha256"] = _sha256(_plan_contract(state))
    _atomic_write_json(state_path, state)
    _validate_state_integrity(state)
    return _campaign_summary(state)


def _target_tickers(state: dict[str, Any]) -> list[str]:
    return list(_read_target_payload(Path(state["target_file"]))["tickers"])


def _current_report(
    state: dict[str, Any], *, target_tickers: Sequence[Any] | None = None
) -> dict[str, Any]:
    return full_universe_delta_report(
        as_of=state["effective_as_of"],
        coverage_campaign_id=state["coverage_campaign_id"],
        allow_carried_verdicts=False,
        target_tickers=(
            _target_tickers(state)
            if target_tickers is None
            else _normalized_tickers(target_tickers)
        ),
    )


def _quick_membership_fingerprint() -> str:
    marks = ",".join("?" for _ in CANONICAL_SWEEP_SECTORS)
    with get_db() as conn:
        rows = conn.execute(
            f"""
            WITH latest_non_null AS (
                SELECT ticker, MAX(as_of_date) AS as_of_date
                FROM sector_inference
                WHERE inferred_sector IN ({marks})
                GROUP BY ticker
            )
            SELECT DISTINCT UPPER(si.ticker) AS ticker,
                            LOWER(si.inferred_sector) AS sector
            FROM sector_inference si
            JOIN latest_non_null latest
              ON latest.ticker = si.ticker
             AND latest.as_of_date = si.as_of_date
            WHERE si.inferred_sector IN ({marks})
            ORDER BY ticker
            """,
            [*CANONICAL_SWEEP_SECTORS, *CANONICAL_SWEEP_SECTORS],
        ).fetchall()
    return sha256(
        json.dumps(
            [(str(row["ticker"]), str(row["sector"])) for row in rows],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _assert_source_and_membership(state: dict[str, Any]) -> None:
    if _source_contract()["sha256"] != state["source_contract"]["sha256"]:
        raise ClassicScanPlanError("scanner source contract changed after planning")
    if _quick_membership_fingerprint() != state["membership_fingerprint"]:
        raise ClassicScanPlanError("membership fingerprint changed after planning")


def _assert_initial_plan_fresh(state: dict[str, Any], report: dict[str, Any]) -> None:
    blockers = _manifest_blockers(report)
    if blockers:
        raise ClassicScanPlanError("current manifest is blocked: " + ",".join(blockers))
    if (report.get("membership") or {}).get("current_membership_fingerprint") != state.get(
        "membership_fingerprint"
    ):
        raise ClassicScanPlanError("membership fingerprint changed after planning")
    if report.get("target_tickers_sha256") != state.get("target_tickers_sha256"):
        raise ClassicScanPlanError("live target scope does not match frozen target")
    current_cells = {str(cell.get("cell_id") or ""): cell for cell in report.get("cells") or []}
    for planned in state["cells"]:
        current = current_cells.get(planned["cell_id"])
        if (
            current is None
            or _sha256(_normalized_tickers(current.get("loaded_tickers") or []))
            != planned["loaded_tickers_sha256"]
        ):
            raise ClassicScanPlanError(
                f"candidate membership changed for {planned['cell_id']} before spend"
            )
    _assert_source_and_membership(state)


def _runtime_preflight_failure(
    state: dict[str, Any], report: dict[str, Any], error: ClassicScanPlanError
) -> dict[str, Any]:
    try:
        current_source_contract: dict[str, Any] = {
            "sha256": _source_contract()["sha256"]
        }
    except Exception as exc:
        current_source_contract = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    record = {
        "schema_version": "classic_scan_runtime_preflight_failure_v1",
        "recorded_at": _utc_now(),
        "error": str(error),
        "report": _preflight_report_evidence(
            report,
            role="runtime_frozen_campaign_target",
        ),
        "planned_contract": _runtime_planned_contract(state),
        "current_source_contract": current_source_contract,
    }
    record["record_sha256"] = _sha256(record)
    return record


def _validate_authorization(
    state: dict[str, Any],
    *,
    authorized_max_cost_usd: float,
    expect_plan_sha256: str,
    expect_provider: str,
    expect_model: str,
    accept_estimate_shortfall: bool = False,
) -> float:
    try:
        ceiling = float(authorized_max_cost_usd)
    except (TypeError, ValueError) as exc:
        raise ClassicScanAuthorizationError(
            "authorized max cost must be a finite positive number"
        ) from exc
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise ClassicScanAuthorizationError("authorized max cost must be a finite positive number")
    estimate = float((state.get("cost_estimate") or {}).get("estimated_cost_usd") or 0)
    if ceiling < estimate and not accept_estimate_shortfall:
        raise ClassicScanAuthorizationError(
            f"authorized ceiling ${ceiling:.2f} is below the plan estimate "
            f"${estimate:.2f}; refusing to silently run a partial campaign "
            "without --accept-estimate-shortfall"
        )
    if str(expect_plan_sha256 or "") != state["plan_sha256"]:
        raise ClassicScanAuthorizationError("authorized plan SHA-256 does not match")
    binding = state["provider_binding"]
    if str(expect_provider or "").strip().lower() != binding["provider"]:
        raise ClassicScanAuthorizationError("authorized provider does not match plan")
    if str(expect_model or "").strip() != binding["model"]:
        raise ClassicScanAuthorizationError("authorized model does not match plan")
    current = _configured_provider_binding()
    if not current["ready"]:
        raise ClassicScanAuthorizationError("configured LLM provider is not ready")
    if current != binding:
        raise ClassicScanAuthorizationError(
            "configured provider/model/output contract changed after planning"
        )
    if float(state.get("cumulative_cost_usd") or 0.0) > ceiling:
        raise ClassicScanAuthorizationError(
            "recorded campaign spend already exceeds this authorization"
        )
    return ceiling


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ClassicScanLocked(f"classic scan lock is already held: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_existing_lock(path: Path) -> Iterator[None]:
    """Lock an existing campaign file without creating or rewriting owner data."""

    if not path.is_file():
        raise ClassicScanPlanError(f"required classic scan lock file is missing: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ClassicScanPlanError(f"classic scan lock file cannot be opened: {path}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ClassicScanLocked(f"classic scan lock is already held: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _child_argv(state: dict[str, Any], cell: dict[str, Any], reservation: float) -> list[str]:
    sector = str(cell.get("sector") or "")
    band = str(cell.get("band") or "")
    if sector not in CANONICAL_SWEEP_SECTORS or band not in V1_ATOMIC_BANDS:
        raise ClassicScanPlanError(f"invalid planned cell: {sector}:{band}")
    argv = [
        sys.executable,
        "-m",
        "app.cli",
        "autonomous-sector-run",
        "--sector",
        sector,
        "--market-cap-focus",
        band,
        "--only-unswept",
        "--pipeline-version",
        "v1",
        "--coverage-campaign-id",
        state["coverage_campaign_id"],
        "--coverage-target-file",
        state["target_file"],
        "--no-carry-prior-verdicts",
        "--as-of",
        state["effective_as_of"],
        "--max-cost-usd",
        f"{reservation:.6f}",
        "--strict-cost-cap",
    ]
    if not state["populate_watchlist"]:
        argv.append("--no-watchlist")
    return argv


def _default_command_runner(
    argv: Sequence[str], *, cwd: Path, timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
        shell=False,
    )


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[index + end :].strip():
            return value
    return None


def _normalize_attestation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClassicScanPlanError("child result has no provider usage attestation")
    try:
        cost = float(value.get("cost_estimate_usd"))
        attempts = int(value.get("physical_attempt_count"))
    except (TypeError, ValueError) as exc:
        raise ClassicScanPlanError("child provider usage attestation is malformed") from exc
    if value.get("valid") is not True or not math.isfinite(cost) or cost < 0 or attempts < 0:
        raise ClassicScanPlanError("child provider usage attestation is invalid")
    bindings = sorted(
        {
            (
                str(row.get("provider") or "").strip().lower(),
                str(row.get("model") or "").strip(),
            )
            for row in value.get("provider_models") or []
            if isinstance(row, dict)
        }
    )
    digest = str(value.get("usage_records_sha256") or "")
    if len(digest) != 64:
        raise ClassicScanPlanError("child provider usage digest is malformed")
    if (attempts == 0 and (cost > 0 or bindings)) or (attempts > 0 and not bindings):
        raise ClassicScanPlanError("child provider usage count, cost, and binding do not reconcile")
    return {
        "valid": True,
        "physical_attempt_count": attempts,
        "cost_estimate_usd": round(cost, 6),
        "provider_models": [{"provider": provider, "model": model} for provider, model in bindings],
        "usage_records_sha256": digest,
    }


def _attestations_for_child(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_attestation = _normalize_attestation(payload.get("provider_usage_attestation"))
    incremental_value = payload.get("provider_usage_incremental_attestation")
    incremental_attestation = (
        _normalize_attestation(incremental_value)
        if incremental_value is not None
        else total_attestation
    )
    if (
        incremental_attestation["physical_attempt_count"]
        > total_attestation["physical_attempt_count"]
        or incremental_attestation["cost_estimate_usd"]
        > total_attestation["cost_estimate_usd"] + 1e-6
    ):
        raise ClassicScanPlanError("child incremental usage exceeds total physical usage")
    artifact_value = str(payload.get("artifact_path") or "").strip()
    if artifact_value:
        artifact_path = Path(artifact_value)
        if not artifact_path.is_absolute():
            artifact_path = _repo_root() / artifact_path
        if not artifact_path.is_file():
            raise ClassicScanPlanError("child artifact is missing for usage attestation")
        artifact_payload = _read_json_object(artifact_path)
        artifact_total_attestation = _normalize_attestation(
            provider_usage_attestation(artifact_payload)
        )
        artifact_incremental_attestation = _normalize_attestation(
            provider_usage_attestation(
                artifact_payload,
                include_reused=False,
            )
        )
        if artifact_total_attestation != total_attestation:
            raise ClassicScanPlanError(
                "child stdout total usage attestation does not match persisted artifact"
            )
        if artifact_incremental_attestation != incremental_attestation:
            raise ClassicScanPlanError(
                "child stdout incremental usage attestation does not match persisted artifact"
            )
    return total_attestation, incremental_attestation


def _write_attempt_logs(
    campaign_dir: Path,
    *,
    attempt_number: int,
    cell_id: str,
    stdout: str,
    stderr: str,
) -> dict[str, str]:
    safe_cell = cell_id.replace(":", "__")
    attempt_dir = campaign_dir / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{attempt_number:04d}_{safe_cell}"
    stdout_path = attempt_dir / f"{prefix}.stdout.log"
    stderr_path = attempt_dir / f"{prefix}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {"stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}


def _live_uncovered_for_cell(state: dict[str, Any], cell: dict[str, Any]) -> list[str]:
    with get_db() as conn:
        split = split_loaded_coverage(
            conn,
            band=cell["band"],
            sector=cell["sector"],
            loaded_tickers=list(cell["loaded_tickers"]),
            pipeline_version="v1",
            coverage_campaign_id=state["coverage_campaign_id"],
        )
    return list(split["uncovered"])


def _cell_reservation(
    state: dict[str, Any], cell: dict[str, Any], live_uncovered: list[str]
) -> float:
    estimate = state["cost_estimate"]
    review_count = len(set(live_uncovered) & set(cell["initial_to_review_tickers"]))
    if review_count == 0:
        return 0.0
    raw = (
        float(estimate["estimated_base_cost_per_active_cell_usd"])
        + float(estimate["estimated_cost_per_review_usd"]) * review_count
    )
    return round(max(0.005, raw * float(estimate["safety_multiplier"])), 6)


def _refresh_cost_totals(state: dict[str, Any]) -> None:
    attempts = state.get("attempts") or []
    state["cumulative_cost_usd"] = round(
        sum(
            float(row.get("charged_cost_usd") or 0.0)
            for row in attempts
            if row.get("cost_reconciled")
        ),
        6,
    )
    state["unresolved_cost_reservation_usd"] = round(
        sum(
            float(row.get("reserved_cost_usd") or 0.0)
            for row in attempts
            if not row.get("cost_reconciled")
        ),
        6,
    )


def _resolve_attempt_cost(
    state: dict[str, Any], attempt: dict[str, Any], charged_cost: float
) -> None:
    if attempt.get("cost_reconciled"):
        raise ClassicScanPlanError("attempt cost was already reconciled")
    attempt["charged_cost_usd"] = round(float(charged_cost), 6)
    attempt["cost_reconciled"] = True
    _refresh_cost_totals(state)


_ATTEMPT_STOP_SEVERITY = {
    "SAFE_RETRY_REQUIRED": 10,
    "PROVIDER_MODEL_DRIFT": 30,
    "CELL_COST_RESERVATION_BREACH": 40,
    "AUTHORIZED_COST_BREACH": 50,
    "UNRESOLVED_ATTEMPT": 100,
    "UNRESOLVED_INTERRUPTED": 100,
}


def _emit_progress(
    state: dict[str, Any],
    progress: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    """Record best-effort telemetry failures without affecting paid work."""

    if progress is None:
        return
    try:
        progress(event)
    except Exception as exc:  # telemetry must never strand a paid reservation
        state.setdefault("telemetry_errors", []).append(
            {
                "at": _utc_now(),
                "event": str(event.get("event") or "UNKNOWN"),
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
        )


def _rate_limit_signal(result: Any, payload: dict[str, Any]) -> bool:
    coverage = payload.get("coverage_accounting") or {}
    failed = (
        int(getattr(result, "returncode", 0) or 0) != 0
        or str(payload.get("status") or "").strip().upper()
        in {"ERROR", "FAILED", "INCOMPLETE", "PARTIAL"}
        or bool(coverage.get("rerun_required"))
    )
    if not failed:
        return False

    fragments: list[str] = [str(getattr(result, "stderr", "") or "")]
    error_keys = {
        "error",
        "errors",
        "error_code",
        "error_type",
        "exception",
        "http_status",
        "reason",
        "response_status",
        "status_code",
    }
    for container in (payload, coverage):
        if not isinstance(container, dict):
            continue
        fragments.extend(str(container[key]) for key in error_keys if key in container)

    def collect_usage_errors(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = str(key).strip().lower()
                if normalized_key == "provider_usage":
                    collect_usage_errors(child)
                elif normalized_key in error_keys:
                    fragments.append(str(child))
                elif isinstance(child, (dict, list)):
                    collect_usage_errors(child)
        elif isinstance(value, list):
            for child in value:
                collect_usage_errors(child)

    collect_usage_errors(payload.get("provider_usage"))
    text = " ".join(fragments).lower()
    return any(
        marker in text
        for marker in ("rate limit", "rate_limit", "status=429", "status 429", "too many requests")
    )


def _attempt_stop(attempt: dict[str, Any]) -> tuple[int, str, str] | None:
    attempt_status = str(attempt.get("status") or "")
    severity = _ATTEMPT_STOP_SEVERITY.get(attempt_status)
    if severity is None:
        return None
    if attempt_status.startswith("UNRESOLVED_"):
        return (
            severity,
            "RECONCILIATION_REQUIRED",
            str(attempt.get("error") or f"UNRESOLVED:{attempt['cell_id']}"),
        )
    if attempt_status == "SAFE_RETRY_REQUIRED":
        return (
            severity,
            "INCOMPLETE_RETRY_REQUIRED",
            str(attempt.get("stop_reason") or f"CELL_RETRY_REQUIRED:{attempt['cell_id']}"),
        )
    return (
        severity,
        attempt_status,
        str(attempt.get("stop_reason") or f"{attempt_status}:{attempt['cell_id']}"),
    )


def _highest_attempt_stop(
    attempts: Sequence[dict[str, Any]],
) -> tuple[int, str, str] | None:
    candidates = [stop for attempt in attempts if (stop := _attempt_stop(attempt))]
    return max(candidates, key=lambda row: row[0]) if candidates else None


def _highest_blocking_attempt_stop(
    attempts: Sequence[dict[str, Any]],
) -> tuple[int, str, str] | None:
    blocked = {
        "PROVIDER_MODEL_DRIFT",
        "CELL_COST_RESERVATION_BREACH",
        "AUTHORIZED_COST_BREACH",
    }
    candidates = [
        stop
        for attempt in attempts
        if str(attempt.get("status") or "") in blocked
        if (stop := _attempt_stop(attempt))
    ]
    return max(candidates, key=lambda row: row[0]) if candidates else None


def _settle_child_outcome(
    *,
    state: dict[str, Any],
    attempt: dict[str, Any],
    outcome: Any,
    campaign_dir: Path,
    ceiling: float,
) -> BaseException | None:
    """Reconcile one launched child; never called from worker threads."""

    cell_id = str(attempt["cell_id"])
    if isinstance(outcome, BaseException):
        interrupted = isinstance(outcome, (KeyboardInterrupt, SystemExit))
        attempt["status"] = "UNRESOLVED_INTERRUPTED" if interrupted else "UNRESOLVED_ATTEMPT"
        attempt["completed_at"] = _utc_now()
        attempt["error"] = f"{type(outcome).__name__}: {outcome}"
        return outcome if interrupted else None

    try:
        logs = _write_attempt_logs(
            campaign_dir,
            attempt_number=int(attempt["attempt_number"]),
            cell_id=cell_id,
            stdout=str(outcome.stdout or ""),
            stderr=str(outcome.stderr or ""),
        )
        attempt.update(logs)
        attempt["returncode"] = int(outcome.returncode)
        payload = _last_json_object(str(outcome.stdout or ""))
        if payload is None:
            raise ClassicScanPlanError("child result did not end with a JSON object")
        total_attestation, incremental_attestation = _attestations_for_child(payload)
    except BaseException as exc:  # paid state is ambiguous; reserve stays unresolved
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        attempt["status"] = "UNRESOLVED_INTERRUPTED" if interrupted else "UNRESOLVED_ATTEMPT"
        attempt["completed_at"] = _utc_now()
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        return exc if interrupted else None

    cell_cost = float(incremental_attestation["cost_estimate_usd"])
    attempt["provider_usage_attestation"] = incremental_attestation
    attempt["provider_usage_total_attestation"] = total_attestation
    attempt["result_status"] = payload.get("status")
    attempt["artifact_path"] = payload.get("artifact_path")
    if attempt.get("cost_reconciled"):
        if abs(float(attempt.get("charged_cost_usd") or 0.0) - cell_cost) > 1e-6:
            raise ClassicScanPlanError("replayed child cost does not match reconciled attempt")
        # A coordinator interrupt can land after the reconciliation flag but
        # before aggregate totals or final status. Recompute, then finish the
        # deterministic classification without charging twice.
        _refresh_cost_totals(state)
    else:
        _resolve_attempt_cost(state, attempt, cell_cost)
    expected_binding = {
        (
            str(state["provider_binding"]["provider"]),
            str(state["provider_binding"]["model"]),
        )
    }
    actual_bindings = {
        (str(row["provider"]), str(row["model"]))
        for row in incremental_attestation["provider_models"]
    }
    total_bindings = {
        (str(row["provider"]), str(row["model"])) for row in total_attestation["provider_models"]
    }
    attempt["completed_at"] = _utc_now()
    if (
        int(total_attestation["physical_attempt_count"]) > 0 and total_bindings != expected_binding
    ) or (
        int(incremental_attestation["physical_attempt_count"]) > 0
        and actual_bindings != expected_binding
    ):
        attempt["status"] = "PROVIDER_MODEL_DRIFT"
        attempt["stop_reason"] = f"PROVIDER_MODEL_DRIFT:{cell_id}"
    elif cell_cost > float(attempt["reserved_cost_usd"]) + 1e-6:
        attempt["status"] = "CELL_COST_RESERVATION_BREACH"
        attempt["stop_reason"] = f"CELL_COST_RESERVATION_BREACH:{cell_id}"
    elif float(state["cumulative_cost_usd"]) > ceiling + 1e-6:
        attempt["status"] = "AUTHORIZED_COST_BREACH"
        attempt["stop_reason"] = f"AUTHORIZED_COST_BREACH:{cell_id}"
    else:
        rerun_required = (
            bool((payload.get("coverage_accounting") or {}).get("rerun_required"))
            or int((payload.get("delta_audit") or {}).get("remaining_uncovered") or 0) > 0
        )
        rate_limited = _rate_limit_signal(outcome, payload)
        if int(outcome.returncode) != 0 or rerun_required or rate_limited:
            attempt["status"] = "SAFE_RETRY_REQUIRED"
            attempt["stop_reason"] = (
                f"RATE_LIMIT_BACKPRESSURE:{cell_id}"
                if rate_limited
                else f"CELL_RETRY_REQUIRED:{cell_id}"
            )
        else:
            attempt["status"] = "OK"
    return None


def run_campaign(
    *,
    campaign_id: str,
    authorized_max_cost_usd: float,
    expect_plan_sha256: str,
    expect_provider: str,
    expect_model: str,
    accept_estimate_shortfall: bool = False,
    output_root: str | Path | None = None,
    cell_timeout_seconds: int = _DEFAULT_CELL_TIMEOUT_SECONDS,
    cell_workers: int | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run or resume frozen cells in bounded, coordinator-owned waves."""
    normalized_id = _validate_campaign_id(campaign_id)
    root = _campaign_root(output_root)
    campaign_dir = _campaign_dir(normalized_id, output_root)
    runner = command_runner or _default_command_runner
    workers = _resolve_cell_workers(cell_workers)
    timeout = int(cell_timeout_seconds)
    if timeout <= 0:
        raise ValueError("cell timeout must be positive")

    with (
        _exclusive_lock(root / ".paid_scan.lock"),
        _exclusive_lock(campaign_dir / ".campaign.lock"),
    ):
        state_path, state = _load_state(normalized_id, output_root)
        if float(state["unresolved_cost_reservation_usd"]) > 0:
            raise ClassicScanAuthorizationError(
                "campaign has an unresolved paid-attempt reservation; use "
                "classic-scan reconcile only after confirming the child stopped"
            )
        if state["status"] in _BLOCKED_RUN_STATUSES:
            raise ClassicScanPlanError(
                f"campaign is blocked: {state['status']}:{state.get('stop_reason')}"
            )
        if workers > 1 and bool(state["populate_watchlist"]):
            raise ClassicScanAuthorizationError(
                "parallel classic cells require a --no-watchlist campaign; "
                "Phase A does not authorize concurrent watchlist mutation"
            )
        ceiling = _validate_authorization(
            state,
            authorized_max_cost_usd=authorized_max_cost_usd,
            expect_plan_sha256=expect_plan_sha256,
            expect_provider=expect_provider,
            expect_model=expect_model,
            accept_estimate_shortfall=accept_estimate_shortfall,
        )

        # Recover any product artifact written immediately before a prior child
        # or parent crash, so a resume never purchases the same review again.
        state["last_run_artifact_backfill"] = backfill_loaded_sets_from_artifacts()
        report = _current_report(state)
        try:
            _assert_initial_plan_fresh(state, report)
        except ClassicScanPlanError as exc:
            state["latest_report"] = report
            state["latest_preflight_failure"] = _runtime_preflight_failure(
                state, report, exc
            )
            state["status"] = "STALE_PLAN"
            state["stop_reason"] = str(exc)
            state["updated_at"] = _utc_now()
            _atomic_write_json(state_path, state)
            _validate_state_integrity(state)
            raise
        state["latest_report"] = report
        state["authorized_max_cost_usd"] = ceiling
        state["last_cell_workers"] = workers
        state["authorization_history"].append(
            {
                "authorized_at": _utc_now(),
                "max_cost_usd": ceiling,
                "plan_sha256": expect_plan_sha256,
                "provider": str(expect_provider).lower(),
                "model": expect_model,
                "plan_estimate_usd": float(
                    (state.get("cost_estimate") or {}).get("estimated_cost_usd") or 0
                ),
                "estimate_shortfall_accepted": bool(accept_estimate_shortfall),
                "cell_workers": workers,
            }
        )
        if report.get("complete"):
            state["status"] = "COMPLETE"
            state["stop_reason"] = None
            state["updated_at"] = _utc_now()
            _atomic_write_json(state_path, state)
            return _campaign_summary(state)

        state["status"] = "RUNNING"
        state["stop_reason"] = None
        state["updated_at"] = _utc_now()
        _atomic_write_json(state_path, state)

        pending_ids = {str(cell.get("cell_id") or "") for cell in report.get("pending_cells") or []}
        pending_snapshot = [cell for cell in state["cells"] if str(cell["cell_id"]) in pending_ids]
        planned_by_id = {str(cell["cell_id"]): cell for cell in pending_snapshot}
        queue = list(pending_snapshot)
        queued_ids = {str(cell["cell_id"]) for cell in queue}
        last_uncovered_count: dict[str, int] = {}
        stalled_cells: list[dict[str, Any]] = []
        wave_number = 0
        stopped = False
        deferred_interrupt: BaseException | None = None

        while not stopped:
            try:
                _assert_source_and_membership(state)
            except ClassicScanPlanError as exc:
                state["status"] = "SOURCE_CONTRACT_DRIFT"
                state["stop_reason"] = str(exc)
                stopped = True
                break

            current_wave = wave_number + 1
            claimed_tickers: set[str] = set()
            wave: list[dict[str, Any]] = []
            budget_candidate: tuple[dict[str, Any], float, float] | None = None
            cells_to_consider = len(queue)
            while queue and cells_to_consider > 0 and len(wave) < workers:
                cell = queue.pop(0)
                cell_id = str(cell["cell_id"])
                queued_ids.discard(cell_id)
                cells_to_consider -= 1
                live_uncovered = _normalized_tickers(_live_uncovered_for_cell(state, cell))
                if not live_uncovered:
                    continue
                prior_count = last_uncovered_count.get(cell_id)
                if prior_count is not None and len(live_uncovered) >= prior_count:
                    # A retry is authorized only after strict live progress.
                    # This check runs before reserving the next wave, so a cell
                    # that made no progress is named and abandoned without a
                    # second charge.  Other queued cells remain eligible.
                    stalled_cells.append(
                        {
                            "cell_id": cell_id,
                            "sector": cell["sector"],
                            "band": cell["band"],
                            "uncovered_before": prior_count,
                            "uncovered_after": len(live_uncovered),
                            "residual_tickers": list(live_uncovered),
                        }
                    )
                    _emit_progress(
                        state,
                        progress,
                        {
                            "event": "CELL_STALLED",
                            "campaign_id": normalized_id,
                            "cell_id": cell_id,
                            "uncovered_before": prior_count,
                            "uncovered_after": len(live_uncovered),
                        },
                    )
                    continue
                if claimed_tickers & set(live_uncovered):
                    # An unknown-cap name can occur in five band cells.  Claim
                    # the whole live set so one wave can never buy it twice.
                    queue.append(cell)
                    queued_ids.add(cell_id)
                    continue
                reservation = _cell_reservation(state, cell, live_uncovered)
                available = round(
                    ceiling
                    - float(state["cumulative_cost_usd"])
                    - float(state["unresolved_cost_reservation_usd"]),
                    6,
                )
                if reservation > available + 1e-6:
                    if budget_candidate is None:
                        budget_candidate = (cell, reservation, available)
                    queue.append(cell)
                    queued_ids.add(cell_id)
                    continue
                last_uncovered_count[cell_id] = len(live_uncovered)
                attempt_number = len(state["attempts"]) + 1
                attempt: dict[str, Any] = {
                    "attempt_number": attempt_number,
                    "wave_number": current_wave,
                    "cell_workers": workers,
                    "cell_id": cell_id,
                    "sector": cell["sector"],
                    "band": cell["band"],
                    "started_at": _utc_now(),
                    "status": "RUNNING",
                    "live_uncovered_before": live_uncovered,
                    "claimed_tickers_sha256": _sha256(live_uncovered),
                    "reserved_cost_usd": reservation,
                    "cost_reconciled": False,
                    "argv": _child_argv(state, cell, reservation),
                }
                state["attempts"].append(attempt)
                state["unresolved_cost_reservation_usd"] = round(
                    float(state["unresolved_cost_reservation_usd"]) + reservation,
                    6,
                )
                attempt["remaining_authorized_cost_usd_after_reservation"] = round(
                    ceiling
                    - float(state["cumulative_cost_usd"])
                    - float(state["unresolved_cost_reservation_usd"]),
                    6,
                )
                wave.append(attempt)
                claimed_tickers.update(live_uncovered)

            if not wave:
                if budget_candidate is not None:
                    cell, reservation, available = budget_candidate
                    state["status"] = "BUDGET_EXHAUSTED"
                    state["stop_reason"] = (
                        f"NEXT_CELL_RESERVATION_{reservation:.6f}_EXCEEDS_"
                        f"REMAINING_{available:.6f}:{cell['cell_id']}"
                    )
                    stopped = True
                break

            wave_number = current_wave
            if (
                float(state["cumulative_cost_usd"])
                + float(state["unresolved_cost_reservation_usd"])
                > ceiling + 1e-6
            ):
                raise ClassicScanPlanError(
                    "aggregate in-flight reservations exceed owner authorization"
                )
            state["updated_at"] = _utc_now()
            state["status"] = "RUNNING"
            state["stop_reason"] = None
            # One durable coordinator write reserves the entire wave before
            # any worker can reach a paid child process.
            _atomic_write_json(state_path, state)
            _validate_state_integrity(state)
            try:
                for attempt in wave:
                    _emit_progress(
                        state,
                        progress,
                        {
                            "event": "CELL_START",
                            "campaign_id": normalized_id,
                            "cell_id": attempt["cell_id"],
                            "attempt_number": attempt["attempt_number"],
                            "wave_number": current_wave,
                            "cell_workers": workers,
                            "reserved_cost_usd": attempt["reserved_cost_usd"],
                            "remaining_authorized_cost_usd": attempt[
                                "remaining_authorized_cost_usd_after_reservation"
                            ],
                        },
                    )
                state["updated_at"] = _utc_now()
                _atomic_write_json(state_path, state)
            except (KeyboardInterrupt, SystemExit) as exc:
                # No child has been submitted yet, so every reservation can
                # be released exactly rather than conservatively charged.
                for attempt in wave:
                    _resolve_attempt_cost(state, attempt, 0.0)
                    attempt["status"] = "SAFE_RETRY_REQUIRED"
                    attempt["completed_at"] = _utc_now()
                    attempt["stop_reason"] = (
                        f"COORDINATOR_INTERRUPTED_BEFORE_DISPATCH:{attempt['cell_id']}"
                    )
                state["status"] = "INCOMPLETE_RETRY_REQUIRED"
                state["stop_reason"] = "COORDINATOR_INTERRUPTED_BEFORE_DISPATCH"
                state["updated_at"] = _utc_now()
                _atomic_write_json(state_path, state)
                _validate_state_integrity(state)
                raise exc

            future_attempts: dict[Future[Any], dict[str, Any]] = {}
            settled_futures: set[Future[Any]] = set()
            executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="classic-cell",
            )
            coordinator_interrupt: BaseException | None = None
            dispatching_attempt: dict[str, Any] | None = None

            def settle_future(
                future: Future[Any],
                future_map: dict[Future[Any], dict[str, Any]],
                settled: set[Future[Any]],
                wave_attempts: Sequence[dict[str, Any]],
                wave_id: int,
            ) -> None:
                nonlocal deferred_interrupt
                attempt = future_map[future]
                if future in settled:
                    return
                try:
                    outcome: Any = future.result()
                except BaseException as exc:  # noqa: BLE001
                    outcome = exc
                interrupt = _settle_child_outcome(
                    state=state,
                    attempt=attempt,
                    outcome=outcome,
                    campaign_dir=campaign_dir,
                    ceiling=ceiling,
                )
                if interrupt is not None and deferred_interrupt is None:
                    deferred_interrupt = interrupt
                interim_stop = _highest_attempt_stop(wave_attempts)
                if interim_stop is not None:
                    _severity, state["status"], state["stop_reason"] = interim_stop
                if attempt["status"] == "OK":
                    _emit_progress(
                        state,
                        progress,
                        {
                            "event": "CELL_COMPLETE",
                            "campaign_id": normalized_id,
                            "cell_id": attempt["cell_id"],
                            "wave_number": wave_id,
                            "cell_cost_usd": attempt["charged_cost_usd"],
                            "cumulative_cost_usd": state["cumulative_cost_usd"],
                        },
                    )
                state["updated_at"] = _utc_now()
                _atomic_write_json(state_path, state)
                _validate_state_integrity(state)
                settled.add(future)

            try:
                for attempt in wave:
                    dispatching_attempt = attempt
                    future = executor.submit(
                        runner,
                        attempt["argv"],
                        cwd=_repo_root(),
                        timeout_seconds=timeout,
                    )
                    future_attempts[future] = attempt
                    dispatching_attempt = None
                # Completion-order settlement checkpoints exact child usage as
                # soon as it is known; attempt numbering remains canonical.
                for future in as_completed(future_attempts):
                    settle_future(
                        future,
                        future_attempts,
                        settled_futures,
                        wave,
                        current_wave,
                    )
            except (KeyboardInterrupt, SystemExit) as exc:
                coordinator_interrupt = exc
                deferred_interrupt = exc
                submitted_attempt_ids = {id(attempt) for attempt in future_attempts.values()}
                for attempt in wave:
                    if id(attempt) in submitted_attempt_ids:
                        continue
                    if attempt is dispatching_attempt:
                        attempt["status"] = "UNRESOLVED_INTERRUPTED"
                        attempt["completed_at"] = _utc_now()
                        attempt["error"] = (
                            "CoordinatorInterrupt: child dispatch outcome is ambiguous; "
                            "paid outcome is unresolved"
                        )
                    else:
                        _resolve_attempt_cost(state, attempt, 0.0)
                        attempt["status"] = "SAFE_RETRY_REQUIRED"
                        attempt["completed_at"] = _utc_now()
                        attempt["stop_reason"] = (
                            f"COORDINATOR_INTERRUPTED_BEFORE_DISPATCH:{attempt['cell_id']}"
                        )
                for future, attempt in future_attempts.items():
                    if future in settled_futures:
                        continue
                    if attempt.get("status") != "RUNNING":
                        # The outcome mutated in memory before interruption;
                        # the aggregate write below checkpoints it exactly.
                        settled_futures.add(future)
                        continue
                    if future.done():
                        settle_future(
                            future,
                            future_attempts,
                            settled_futures,
                            wave,
                            current_wave,
                        )
                        continue
                    if future.cancel():
                        _resolve_attempt_cost(state, attempt, 0.0)
                        attempt["status"] = "SAFE_RETRY_REQUIRED"
                        attempt["completed_at"] = _utc_now()
                        attempt["stop_reason"] = (
                            f"COORDINATOR_INTERRUPTED_BEFORE_DISPATCH:{attempt['cell_id']}"
                        )
                    elif future.done():
                        settle_future(
                            future,
                            future_attempts,
                            settled_futures,
                            wave,
                            current_wave,
                        )
                    else:
                        attempt["status"] = "UNRESOLVED_INTERRUPTED"
                        attempt["completed_at"] = _utc_now()
                        attempt["error"] = (
                            "CoordinatorInterrupt: child may still be running; "
                            "paid outcome is unresolved"
                        )
                interruption_stop = _highest_attempt_stop(wave)
                if interruption_stop is None:
                    state["status"] = "INCOMPLETE_RETRY_REQUIRED"
                    state["stop_reason"] = "COORDINATOR_INTERRUPTED_AFTER_SETTLEMENT"
                else:
                    _severity, state["status"], state["stop_reason"] = interruption_stop
                state["updated_at"] = _utc_now()
                _atomic_write_json(state_path, state)
                _validate_state_integrity(state)
            finally:
                executor.shutdown(wait=coordinator_interrupt is None, cancel_futures=False)

            if coordinator_interrupt is not None:
                stopped = True

            requeue_attempts = [
                attempt
                for attempt in wave
                if str(attempt.get("status") or "") == "SAFE_RETRY_REQUIRED"
                and str(attempt.get("stop_reason") or "").startswith(
                    "CELL_RETRY_REQUIRED:"
                )
            ]
            requeue_attempt_ids = {id(attempt) for attempt in requeue_attempts}
            wave_stop = _highest_attempt_stop(
                [attempt for attempt in wave if id(attempt) not in requeue_attempt_ids]
            )
            if wave_stop is not None:
                _severity, state["status"], state["stop_reason"] = wave_stop
                stopped = True
            elif not stopped:
                # Source 851419c2 had no concurrent failure taxonomy.  Fail
                # closed here: only an ordinary first-pass remainder is
                # requeued.  Rate limits, ambiguous outcomes, hard contract
                # failures, and interrupts halt after their launched siblings
                # have drained and settled.
                for attempt in requeue_attempts:
                    cell_id = str(attempt["cell_id"])
                    cell = planned_by_id[cell_id]
                    if cell_id not in queued_ids:
                        queue.append(cell)
                        queued_ids.add(cell_id)
                    _emit_progress(
                        state,
                        progress,
                        {
                            "event": "CELL_REQUEUED",
                            "campaign_id": normalized_id,
                            "cell_id": cell_id,
                            "attempt_number": attempt["attempt_number"],
                            "wave_number": current_wave,
                            "uncovered_before": len(
                                attempt.get("live_uncovered_before") or []
                            ),
                            "cumulative_cost_usd": state["cumulative_cost_usd"],
                        },
                    )

        state["stalled_cells"] = stalled_cells
        if stopped:
            # Persist the safety decision before any fallible diagnostic
            # refresh so a reporting error can never weaken the stop.
            state["updated_at"] = _utc_now()
            _atomic_write_json(state_path, state)
            _validate_state_integrity(state)
        if deferred_interrupt is not None:
            raise deferred_interrupt

        try:
            state["last_run_artifact_backfill"] = backfill_loaded_sets_from_artifacts()
            final_report = _current_report(state)
            state["latest_report"] = final_report
            state.pop("diagnostic_refresh_error", None)
        except Exception as exc:
            final_report = state.get("latest_report") or {}
            state["diagnostic_refresh_error"] = {
                "at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
            if not stopped:
                state["status"] = "INCOMPLETE_RETRY_REQUIRED"
                state["stop_reason"] = "FINAL_DIAGNOSTIC_REFRESH_FAILED"
                stopped = True

        if not stopped and "diagnostic_refresh_error" not in state:
            try:
                _assert_source_and_membership(state)
            except ClassicScanPlanError as exc:
                state["status"] = "SOURCE_CONTRACT_DRIFT"
                state["stop_reason"] = str(exc)
                stopped = True
        if not stopped:
            if final_report.get("complete") and not _manifest_blockers(final_report):
                state["status"] = "COMPLETE"
                state["stop_reason"] = None
            elif stalled_cells:
                state["status"] = "INCOMPLETE_CELLS_STALLED"
                state["stop_reason"] = "CELLS_STALLED_NO_PROGRESS:" + ",".join(
                    str(entry["cell_id"]) for entry in stalled_cells
                )
            else:
                state["status"] = "INCOMPLETE_RETRY_REQUIRED"
                state["stop_reason"] = "FROZEN_TARGET_REMAINS_UNCOVERED"

        state["updated_at"] = _utc_now()
        _atomic_write_json(state_path, state)
        _validate_state_integrity(state)
        return _campaign_summary(state)


def reconcile_campaign(
    *,
    campaign_id: str,
    assume_reserved_spent: bool,
    confirm_child_stopped: bool,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Conservatively consume unresolved reservations after operator confirmation."""
    if not assume_reserved_spent or not confirm_child_stopped:
        raise ClassicScanAuthorizationError(
            "reconciliation requires --assume-reserved-spent and --confirm-child-stopped"
        )
    normalized_id = _validate_campaign_id(campaign_id)
    root = _campaign_root(output_root)
    campaign_dir = _campaign_dir(normalized_id, output_root)
    with (
        _exclusive_lock(root / ".paid_scan.lock"),
        _exclusive_lock(campaign_dir / ".campaign.lock"),
    ):
        state_path, state = _load_state(normalized_id, output_root)
        unresolved = [
            attempt for attempt in state["attempts"] if not attempt.get("cost_reconciled")
        ]
        if not unresolved:
            raise ClassicScanPlanError("campaign has no unresolved cost reservation")
        reconciled_total = 0.0
        for attempt in unresolved:
            reserve = float(attempt["reserved_cost_usd"])
            _resolve_attempt_cost(state, attempt, reserve)
            attempt["status"] = "RECONCILED_RESERVED_AS_SPENT"
            attempt["reconciled_at"] = _utc_now()
            reconciled_total += reserve
        state["reconciliation_history"].append(
            {
                "reconciled_at": _utc_now(),
                "assumption": "unresolved_reserved_cost_assumed_fully_spent",
                "amount_usd": round(reconciled_total, 6),
                "child_stopped_confirmed": True,
            }
        )
        ceiling = state.get("authorized_max_cost_usd")
        blocking_stop = _highest_blocking_attempt_stop(state["attempts"])
        if blocking_stop is not None:
            _severity, state["status"], state["stop_reason"] = blocking_stop
        else:
            state["status"] = (
                "BUDGET_EXHAUSTED"
                if ceiling is not None and float(state["cumulative_cost_usd"]) >= float(ceiling)
                else "INCOMPLETE_RETRY_REQUIRED"
            )
            state["stop_reason"] = "UNRESOLVED_RESERVATION_CONSERVATIVELY_CHARGED"
        state["updated_at"] = _utc_now()
        _atomic_write_json(state_path, state)
        _validate_state_integrity(state)
        return _campaign_summary(state)


def reconcile_terminal_data_gaps(
    *,
    campaign_id: str,
    expect_plan_sha256: str,
    expect_state_sha256: str,
    expect_target_tickers_sha256: str,
    artifact_path: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Lock and reconcile stopped zero-provider evidence without mutating it."""

    normalized_id = _validate_campaign_id(campaign_id)
    campaign_dir = _campaign_dir(normalized_id, output_root)
    with _exclusive_existing_lock(campaign_dir / ".campaign.lock"):
        return _reconcile_terminal_data_gaps_under_lock(
            campaign_id=normalized_id,
            expect_plan_sha256=expect_plan_sha256,
            expect_state_sha256=expect_state_sha256,
            expect_target_tickers_sha256=expect_target_tickers_sha256,
            artifact_path=artifact_path,
            output_root=output_root,
        )


def _reconcile_terminal_data_gaps_under_lock(
    *,
    campaign_id: str,
    expect_plan_sha256: str,
    expect_state_sha256: str,
    expect_target_tickers_sha256: str,
    artifact_path: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Reconcile stopped zero-provider cell evidence without awarding coverage.

    The result is execution accounting, not an investment-coverage receipt. It
    reconstructs exact direct and unknown-cap-projected terminal reasons from
    exact-byte-bound child logs, binds them to every frozen cell occurrence,
    and deliberately leaves every NEEDS_DATA ticker outside coverage.
    """

    normalized_id = _validate_campaign_id(campaign_id)
    state_path = _state_path(normalized_id, output_root)
    if not state_path.is_file():
        raise FileNotFoundError(f"no classic campaign state at {state_path}")
    state_sha256 = _raw_file_sha256(state_path)
    if state_sha256 != str(expect_state_sha256).strip().lower():
        raise ClassicScanPlanError("campaign state bytes do not match the expected SHA-256")
    loaded_path, state = _load_state(normalized_id, output_root)
    if loaded_path.resolve() != state_path.resolve():
        raise ClassicScanPlanError("campaign state resolved through an unexpected path")
    if _raw_file_sha256(state_path) != state_sha256:
        raise ClassicScanPlanError("campaign state changed while it was being loaded")
    if state["plan_sha256"] != str(expect_plan_sha256).strip().lower():
        raise ClassicScanPlanError("campaign plan does not match the expected SHA-256")
    if state["target_tickers_sha256"] != str(expect_target_tickers_sha256).strip().lower():
        raise ClassicScanPlanError("campaign target does not match the expected SHA-256")
    if state["status"] == "RUNNING":
        raise ClassicScanPlanError("running campaign evidence cannot be terminally reconciled")
    if float(state["unresolved_cost_reservation_usd"]) != 0:
        raise ClassicScanPlanError("unresolved paid work cannot be terminally reconciled")

    campaign_dir = _campaign_dir(normalized_id, output_root).resolve()
    target_path = Path(state["target_file"])
    target_sha256_before = _raw_file_sha256(target_path)
    target_payload = _read_target_payload(target_path)
    if _raw_file_sha256(target_path) != target_sha256_before:
        raise ClassicScanPlanError("frozen target changed while it was being loaded")
    if _sha256(target_payload) != state["target_file_sha256"]:
        raise ClassicScanPlanError("frozen target canonical payload does not match campaign")
    target_tickers = list(target_payload["tickers"])
    target_set = set(target_tickers)
    cells_by_id = {str(cell["cell_id"]): cell for cell in state["cells"]}
    loaded_by_cell = {
        cell_id: set(_normalized_tickers(cell.get("loaded_tickers") or []))
        for cell_id, cell in cells_by_id.items()
    }
    loaded_union = set().union(*loaded_by_cell.values()) if loaded_by_cell else set()
    if loaded_union != target_set:
        raise ClassicScanPlanError("frozen cell membership does not partition the target")

    evidence_by_occurrence: dict[tuple[str, str], dict[str, Any]] = {}
    ticker_dispositions: dict[str, str] = {}
    log_bindings: list[dict[str, Any]] = []

    def add_evidence(
        *,
        cell_id: str,
        ticker: str,
        disposition: str,
        attempt_number: int,
        source_kind: str,
        source_cell_id: str,
    ) -> None:
        normalized_ticker = str(ticker).strip().upper()
        if cell_id not in cells_by_id or normalized_ticker not in loaded_by_cell[cell_id]:
            raise ClassicScanPlanError(
                f"terminal evidence escapes frozen cell membership: {cell_id}:{normalized_ticker}"
            )
        existing_ticker = ticker_dispositions.get(normalized_ticker)
        if existing_ticker is not None and existing_ticker != disposition:
            raise ClassicScanPlanError(
                f"conflicting terminal reasons for frozen ticker {normalized_ticker}"
            )
        ticker_dispositions[normalized_ticker] = disposition
        key = (cell_id, normalized_ticker)
        existing = evidence_by_occurrence.get(key)
        if existing is not None and existing["disposition"] != disposition:
            raise ClassicScanPlanError(
                f"conflicting terminal reasons for frozen cell occurrence {cell_id}:{normalized_ticker}"
            )
        if existing is None:
            existing = {"disposition": disposition, "sources": []}
            evidence_by_occurrence[key] = existing
        source = {
            "attempt_number": attempt_number,
            "source_kind": source_kind,
            "source_cell_id": source_cell_id,
        }
        if source not in existing["sources"]:
            existing["sources"].append(source)

    def direct_dispositions(payload: dict[str, Any]) -> tuple[dict[str, str], set[str], set[str]]:
        selection = payload.get("candidate_selection")
        if not isinstance(selection, dict):
            raise ClassicScanPlanError("child evidence has no candidate selection")
        direct: dict[str, str] = {}

        def add_direct(ticker: Any, disposition: str) -> None:
            normalized_ticker = str(ticker).strip().upper()
            if not normalized_ticker:
                raise ClassicScanPlanError("child terminal evidence has an empty ticker")
            existing = direct.get(normalized_ticker)
            if existing is not None and existing != disposition:
                raise ClassicScanPlanError(
                    f"child evidence gives {normalized_ticker} conflicting terminal reasons"
                )
            direct[normalized_ticker] = disposition

        delta = selection.get("delta_audit")
        zero = (
            delta.get("zero_cost_terminal_coverage")
            if isinstance(delta, dict)
            else None
        )
        if not isinstance(zero, dict):
            raise ClassicScanPlanError("child evidence has no zero-cost terminal accounting")
        zero_tickers = set(_normalized_tickers(zero.get("tickers") or []))
        zero_dispositions = zero.get("candidate_dispositions")
        if not isinstance(zero_dispositions, dict):
            raise ClassicScanPlanError("zero-cost terminal dispositions are malformed")
        normalized_zero = {
            str(ticker).strip().upper(): str(value or "").strip().upper()
            for ticker, value in zero_dispositions.items()
            if str(ticker).strip()
        }
        if set(normalized_zero) != zero_tickers or any(
            value != "STRUCTURAL_SCREENED" for value in normalized_zero.values()
        ):
            raise ClassicScanPlanError("zero-cost terminal evidence is not an exact structural set")
        for ticker in sorted(normalized_zero):
            add_direct(ticker, "STRUCTURAL_SCREENED")

        history = selection.get("financial_history_filter")
        if not isinstance(history, dict):
            raise ClassicScanPlanError("child evidence has no financial-history filter")
        sparse_tickers = set(_normalized_tickers(history.get("excluded_tickers") or []))
        if sparse_tickers:
            minimum_rows = history.get("minimum_rows")
            year_counts = history.get("year_counts")
            if (
                history.get("status") != "FILTERED_SPARSE_FINANCIAL_HISTORY"
                or type(minimum_rows) is not int
                or minimum_rows != 3
                or not isinstance(year_counts, dict)
                or set(str(ticker).strip().upper() for ticker in year_counts) != sparse_tickers
                or any(
                    type(value) is not int or value < 0 or value >= minimum_rows
                    for value in year_counts.values()
                )
            ):
                raise ClassicScanPlanError("sparse-history terminal evidence is malformed")
        elif (
            history.get("status") != "NO_FILTER_ALL_CANDIDATES_REPORTABLE"
            or history.get("minimum_rows") != 3
            or history.get("year_counts") != {}
        ):
            raise ClassicScanPlanError("no-filter financial-history evidence is malformed")
        for ticker in sorted(sparse_tickers):
            add_direct(ticker, "NEEDS_DATA_SPARSE_HISTORY")

        valuation = selection.get("valuation_anchor_filter")
        if not isinstance(valuation, dict):
            raise ClassicScanPlanError("child evidence has no valuation-anchor filter")
        raw_dispositions = valuation.get("dispositions")
        if not isinstance(raw_dispositions, list):
            raise ClassicScanPlanError("valuation-anchor dispositions are malformed")
        missing_valuation_tickers: set[str] = set()
        for row in raw_dispositions:
            if not isinstance(row, dict):
                raise ClassicScanPlanError("valuation-anchor disposition is malformed")
            ticker = str(row.get("ticker") or "").strip().upper()
            if (
                not ticker
                or row.get("terminal_state") != "NEEDS_DATA"
                or row.get("scope_status") != "IN_SCOPE"
                or row.get("screen_status") != "INCOMPLETE"
                or row.get("reason_codes") != ["MISSING_VALUATION"]
                or row.get("last_completed_stage") != "VALUATION_ANCHOR_PREFLIGHT"
                or row.get("review_status") != "NOT_STARTED"
                or row.get("underwriting_verdict") is not None
                or row.get("underwriting_confidence") is not None
                or row.get("watchlist_eligible") is not False
            ):
                raise ClassicScanPlanError("valuation-anchor terminal evidence is not exact")
            missing_valuation_tickers.add(ticker)
            add_direct(ticker, "NEEDS_DATA_MISSING_VALUATION")
        if (
            valuation.get("status") != "NEEDS_DATA_ALL_CANDIDATES_MISSING_VALUATION"
            or valuation.get("reason_code") != "MISSING_VALUATION"
            or _normalized_tickers(valuation.get("ready_tickers") or [])
            or set(_normalized_tickers(valuation.get("input_tickers") or []))
            != missing_valuation_tickers
            or set(_normalized_tickers(valuation.get("needs_data_tickers") or []))
            != missing_valuation_tickers
        ):
            raise ClassicScanPlanError("valuation-anchor terminal set is malformed")
        return direct, zero_tickers, sparse_tickers | missing_valuation_tickers

    def projected_bands(
        value: Any,
        *,
        allowed_tickers: set[str],
        current_band: str,
    ) -> tuple[set[str], list[str]]:
        if not isinstance(value, dict):
            return set(), []
        tickers = set(_normalized_tickers(value.get("tickers") or []))
        bands = [str(band).strip().lower() for band in value.get("bands") or []]
        if (
            tickers - allowed_tickers
            or len(bands) != len(set(bands))
            or any(band not in V1_ATOMIC_BANDS or band == current_band for band in bands)
        ):
            raise ClassicScanPlanError("unknown-cap projection evidence is malformed")
        return tickers, bands

    total_physical_attempts = 0
    total_provider_cost = 0.0
    attempted_cells: set[str] = set()
    for expected_attempt_number, attempt in enumerate(state.get("attempts") or [], start=1):
        cell_id = str(attempt.get("cell_id") or "")
        if (
            attempt.get("attempt_number") != expected_attempt_number
            or attempt.get("status") != "SAFE_RETRY_REQUIRED"
            or attempt.get("cost_reconciled") is not True
            or float(attempt.get("charged_cost_usd") or 0.0) != 0
            or int(attempt.get("returncode") or 0) != 1
            or str(attempt.get("result_status") or "") != "FAILED"
        ):
            raise ClassicScanPlanError("campaign attempt is not a settled zero-cost retry result")
        if cell_id in attempted_cells:
            raise ClassicScanPlanError("campaign repeats a frozen cell attempt")
        attempt_argv = attempt.get("argv")
        expected_argv = _child_argv(
            state,
            cells_by_id[cell_id],
            float(attempt.get("reserved_cost_usd") or 0.0),
        )
        if (
            not isinstance(attempt_argv, list)
            or not attempt_argv
            or not str(attempt_argv[0] or "").strip()
            or attempt_argv[1:] != expected_argv[1:]
        ):
            raise ClassicScanPlanError("campaign attempt invocation contract is not exact")
        stdout_path = Path(str(attempt.get("stdout_path") or ""))
        try:
            resolved_stdout = stdout_path.resolve(strict=True)
        except OSError as exc:
            raise ClassicScanPlanError("campaign stdout evidence is missing") from exc
        if resolved_stdout.parent != campaign_dir / "attempts":
            raise ClassicScanPlanError("campaign stdout evidence escapes the campaign directory")
        stdout_bytes = resolved_stdout.read_bytes()
        stdout_sha256 = sha256(stdout_bytes).hexdigest()
        try:
            stdout_text = stdout_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ClassicScanPlanError("campaign stdout evidence is not UTF-8") from exc
        payload = _last_json_object(stdout_text)
        if payload is None:
            raise ClassicScanPlanError("campaign stdout does not end in a JSON object")
        total_attestation, incremental_attestation = _attestations_for_child(payload)
        attempt_incremental = _normalize_attestation(attempt.get("provider_usage_attestation"))
        attempt_total = _normalize_attestation(attempt.get("provider_usage_total_attestation"))
        if (
            incremental_attestation != attempt_incremental
            or total_attestation != attempt_total
            or incremental_attestation["physical_attempt_count"] != 0
            or incremental_attestation["cost_estimate_usd"] != 0
            or total_attestation["physical_attempt_count"] != 0
            or total_attestation["cost_estimate_usd"] != 0
        ):
            raise ClassicScanPlanError("campaign attempt is not exact zero-provider evidence")
        total_physical_attempts += int(incremental_attestation["physical_attempt_count"])
        total_provider_cost += float(incremental_attestation["cost_estimate_usd"])
        sector = str(payload.get("sector") or "").strip().lower()
        band = str(payload.get("market_cap_focus") or "").strip().lower()
        if cell_id != f"{sector}:{band}":
            raise ClassicScanPlanError("campaign stdout cell does not match its attempt")
        selection = payload.get("candidate_selection")
        if not isinstance(selection, dict) or (
            _normalized_tickers(selection.get("loaded_tickers") or [])
            != list(cells_by_id[cell_id]["loaded_tickers"])
            or selection.get("coverage_campaign_id") != normalized_id
            or selection.get("coverage_target_file") != state["target_file"]
            or selection.get("coverage_expected_provider")
            != state["provider_binding"]["provider"]
            or selection.get("coverage_expected_model") != state["provider_binding"]["model"]
        ):
            raise ClassicScanPlanError("campaign stdout is not bound to the frozen cell plan")
        coverage = payload.get("coverage_accounting")
        if (
            payload.get("status") != "FAILED"
            or payload.get("artifact_path") is not None
            or not isinstance(coverage, dict)
            or coverage.get("rerun_required") is not True
        ):
            raise ClassicScanPlanError("campaign stdout is not a failed pre-provider data gap")

        direct, structural_tickers, nonstructural_tickers = direct_dispositions(payload)
        prior_projected_current = {
            ticker
            for ticker in loaded_by_cell[cell_id]
            if any(
                source["source_kind"] == "EXPLICIT_UNKNOWN_CAP_CROSS_BAND_PROJECTION"
                for source in evidence_by_occurrence.get((cell_id, ticker), {}).get(
                    "sources", []
                )
            )
        }
        if set(direct) | prior_projected_current != loaded_by_cell[cell_id]:
            raise ClassicScanPlanError(
                "child terminal evidence and prior explicit projection do not exactly "
                "partition its frozen cell"
            )
        missing_valuation_count = sum(
            disposition == "NEEDS_DATA_MISSING_VALUATION"
            for disposition in direct.values()
        )
        if (
            payload.get("pipeline_version") != "v1"
            or payload.get("scan_family") != "normal"
            or payload.get("final_verdict") != "NO_SELECTION"
            or payload.get("no_selection_reason")
            != (
                "Deterministic valuation-anchor gate stopped before any sector "
                "provider work: NEEDS_DATA: MISSING_VALUATION across all admitted companies."
            )
            or payload.get("degraded_states") != ["NEEDS_DATA"]
            or any(
                type(payload.get(field)) is not int
                for field in (
                    "company_packets",
                    "missing_valuation_anchor_count",
                    "valuation_anchor_count",
                    "tool_calls",
                    "research_questions",
                    "expected_return_scenarios",
                )
            )
            or payload.get("company_packets") != missing_valuation_count
            or payload.get("missing_valuation_anchor_count") != missing_valuation_count
            or payload.get("valuation_anchor_count") != 0
            or payload.get("tool_calls") != 0
            or payload.get("research_questions") != 0
            or payload.get("expected_return_scenarios") != 0
            or not str(payload.get("run_id") or "").strip()
        ):
            raise ClassicScanPlanError("campaign stdout is not an exact pre-provider refusal")
        for ticker, disposition in sorted(direct.items()):
            add_evidence(
                cell_id=cell_id,
                ticker=ticker,
                disposition=disposition,
                attempt_number=expected_attempt_number,
                source_kind="DIRECT_ATTEMPT",
                source_cell_id=cell_id,
            )

        projected, bands = projected_bands(
            coverage.get("unknown_cap_cross_band_projection"),
            allowed_tickers=nonstructural_tickers | prior_projected_current,
            current_band=band,
        )
        for ticker in sorted(projected):
            disposition = direct.get(ticker)
            if disposition is None:
                disposition = evidence_by_occurrence[(cell_id, ticker)]["disposition"]
            for projected_band in bands:
                if ticker not in direct:
                    existing_projected = evidence_by_occurrence.get(
                        (f"{sector}:{projected_band}", ticker)
                    )
                    if (
                        existing_projected is None
                        or existing_projected["disposition"] != disposition
                    ):
                        raise ClassicScanPlanError(
                            "repeated unknown-cap projection is not already exactly proven"
                        )
                    continue
                add_evidence(
                    cell_id=f"{sector}:{projected_band}",
                    ticker=ticker,
                    disposition=disposition,
                    attempt_number=expected_attempt_number,
                    source_kind="EXPLICIT_UNKNOWN_CAP_CROSS_BAND_PROJECTION",
                    source_cell_id=cell_id,
                )

        zero = selection["delta_audit"]["zero_cost_terminal_coverage"]
        projected, bands = projected_bands(
            zero.get("unknown_cap_cross_band_projection"),
            allowed_tickers=structural_tickers,
            current_band=band,
        )
        for ticker in sorted(projected):
            for projected_band in bands:
                add_evidence(
                    cell_id=f"{sector}:{projected_band}",
                    ticker=ticker,
                    disposition=direct[ticker],
                    attempt_number=expected_attempt_number,
                    source_kind="EXPLICIT_UNKNOWN_CAP_CROSS_BAND_PROJECTION",
                    source_cell_id=cell_id,
                )
        attempted_cells.add(cell_id)
        stderr_path = Path(str(attempt.get("stderr_path") or ""))
        try:
            resolved_stderr = stderr_path.resolve(strict=True)
        except OSError as exc:
            raise ClassicScanPlanError("campaign stderr evidence is missing") from exc
        if resolved_stderr.parent != campaign_dir / "attempts":
            raise ClassicScanPlanError("campaign stderr evidence escapes the campaign directory")
        stderr_bytes = resolved_stderr.read_bytes()
        stderr_sha256 = sha256(stderr_bytes).hexdigest()
        log_bindings.append(
            {
                "attempt_number": expected_attempt_number,
                "cell_id": cell_id,
                "run_id": str(payload["run_id"]),
                "argv_sha256": _sha256(attempt_argv),
                "stdout_path": str(resolved_stdout),
                "stdout_sha256": stdout_sha256,
                "stderr_path": str(resolved_stderr),
                "stderr_sha256": stderr_sha256,
            }
        )

    if not log_bindings:
        raise ClassicScanPlanError(
            "campaign has no exact-byte-bound attempt evidence to reconcile"
        )
    if set(ticker_dispositions) != target_set:
        raise ClassicScanPlanError("terminal ticker evidence does not exactly reconcile the target")
    expected_occurrences = {
        (cell_id, ticker)
        for cell_id, tickers in loaded_by_cell.items()
        for ticker in tickers
    }
    if set(evidence_by_occurrence) != expected_occurrences:
        raise ClassicScanPlanError(
            "terminal evidence does not account for every frozen cell occurrence"
        )

    disposition_tickers = {
        disposition: sorted(
            ticker
            for ticker, observed_disposition in ticker_dispositions.items()
            if observed_disposition == disposition
        )
        for disposition in (
            "NEEDS_DATA_MISSING_VALUATION",
            "NEEDS_DATA_SPARSE_HISTORY",
            "STRUCTURAL_SCREENED",
        )
    }
    needs_data_tickers = sorted(
        set(disposition_tickers["NEEDS_DATA_MISSING_VALUATION"])
        | set(disposition_tickers["NEEDS_DATA_SPARSE_HISTORY"])
    )
    classifications = {
        disposition: {
            "ticker_count": len(tickers),
            "tickers_sha256": _sha256(tickers),
            "tickers_sha256_sorted_upper_newline": sha256(
                ("".join(f"{ticker}\n" for ticker in tickers)).encode("utf-8")
            ).hexdigest(),
            "tickers": tickers,
        }
        for disposition, tickers in disposition_tickers.items()
    }

    cell_accounting: list[dict[str, Any]] = []
    projected_only_cells = 0
    for cell_id in _expected_cell_ids():
        loaded = sorted(loaded_by_cell[cell_id])
        reasons = {
            disposition: sorted(
                ticker
                for ticker in loaded
                if evidence_by_occurrence[(cell_id, ticker)]["disposition"] == disposition
            )
            for disposition in classifications
        }
        sources = sorted(
            {
                (
                    int(source["attempt_number"]),
                    str(source["source_kind"]),
                    str(source["source_cell_id"]),
                )
                for ticker in loaded
                for source in evidence_by_occurrence[(cell_id, ticker)]["sources"]
            }
        )
        if not loaded:
            resolution_source = "NO_FROZEN_TARGET_MEMBERS"
        elif cell_id in attempted_cells:
            all_direct_in_cell = all(
                any(
                    source["source_kind"] == "DIRECT_ATTEMPT"
                    and source["source_cell_id"] == cell_id
                    for source in evidence_by_occurrence[(cell_id, ticker)]["sources"]
                )
                for ticker in loaded
            )
            resolution_source = (
                "DIRECT_ATTEMPT_WITH_BOUND_PROJECTIONS"
                if all_direct_in_cell
                else "ATTEMPT_WITH_PRIOR_EXPLICIT_PROJECTION"
            )
        else:
            resolution_source = "EXPLICIT_UNKNOWN_CAP_CROSS_BAND_PROJECTION"
            projected_only_cells += 1
        cell_accounting.append(
            {
                "cell_id": cell_id,
                "sector": cells_by_id[cell_id]["sector"],
                "band": cells_by_id[cell_id]["band"],
                "attempted_in_source_campaign": cell_id in attempted_cells,
                "terminally_accounted": True,
                "resolution_source": resolution_source,
                "loaded_ticker_count": len(loaded),
                "loaded_tickers_sha256": _sha256(loaded),
                "reason_counts": {
                    disposition: len(tickers) for disposition, tickers in reasons.items()
                },
                "reasons": {
                    disposition: {
                        "ticker_count": len(tickers),
                        "tickers_sha256": _sha256(tickers),
                        "tickers": tickers,
                    }
                    for disposition, tickers in reasons.items()
                    if tickers
                },
                "evidence_sources": [
                    {
                        "attempt_number": attempt_number,
                        "source_kind": source_kind,
                        "source_cell_id": source_cell_id,
                    }
                    for attempt_number, source_kind, source_cell_id in sources
                ],
            }
        )

    reconciliation_source_contract = _source_contract()
    report = {
        "schema_version": _TERMINAL_DATA_GAP_RECONCILIATION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "campaign": {
            "campaign_id": normalized_id,
            "campaign_state_status": state["status"],
            "campaign_stop_reason": state.get("stop_reason"),
            "plan_sha256": state["plan_sha256"],
            "state_path": str(state_path.resolve()),
            "state_file_sha256": state_sha256,
            "source_contract_sha256": state["source_contract"]["sha256"],
            "charged_cost_usd": round(float(state["cumulative_cost_usd"]), 6),
            "unresolved_cost_reservation_usd": round(
                float(state["unresolved_cost_reservation_usd"]), 6
            ),
            "current_source_contract_matches_plan": (
                reconciliation_source_contract["sha256"]
                == state["source_contract"]["sha256"]
            ),
            "effective_as_of": state["effective_as_of"],
        },
        "reconciliation_source_contract": reconciliation_source_contract,
        "frozen_target": {
            "target_path": str(target_path.resolve()),
            "ticker_count": len(target_tickers),
            "tickers_sha256": state["target_tickers_sha256"],
            "canonical_payload_sha256": state["target_file_sha256"],
            "raw_file_sha256": target_sha256_before,
        },
        "source_attempt_evidence": {
            "attempt_count": len(log_bindings),
            "attempted_cell_count": len(attempted_cells),
            "attempted_cell_ids_sha256": _sha256(sorted(attempted_cells)),
            "physical_provider_attempt_count": total_physical_attempts,
            "provider_cost_usd": round(total_provider_cost, 6),
            "log_bindings_sha256": _sha256(log_bindings),
            "logs": log_bindings,
        },
        "classification": {
            "ticker_count": len(ticker_dispositions),
            "tickers_sha256": _sha256(sorted(ticker_dispositions)),
            "bucket_count": len(classifications),
            "buckets": classifications,
            "source_reason_codes": {
                "NEEDS_DATA_MISSING_VALUATION": "MISSING_VALUATION",
                "NEEDS_DATA_SPARSE_HISTORY": "NEEDS_DATA_SPARSE_HISTORY",
                "STRUCTURAL_SCREENED": "STRUCTURAL_SCREENED",
            },
            "needs_data_ticker_count": len(needs_data_tickers),
            "needs_data_tickers_sha256": _sha256(needs_data_tickers),
            "needs_data_tickers_sha256_sorted_upper_newline": sha256(
                ("".join(f"{ticker}\n" for ticker in needs_data_tickers)).encode("utf-8")
            ).hexdigest(),
        },
        "terminal_execution_accounting": {
            "status": "COMPLETE",
            "complete": True,
            "semantics": (
                "Every frozen ticker occurrence has an exact stopped-run reason. "
                "This closes same-input execution accounting only."
            ),
            "expected_cell_count": len(_expected_cell_ids()),
            "accounted_cell_count": len(cell_accounting),
            "attempted_cell_count": len(attempted_cells),
            "projected_only_nonempty_cell_count": projected_only_cells,
            "ticker_occurrence_count": len(expected_occurrences),
            "accounted_ticker_occurrence_count": len(evidence_by_occurrence),
            "cells": cell_accounting,
        },
        "investment_coverage": {
            "status": "INCOMPLETE_NEEDS_DATA",
            "complete": False,
            "needs_data_ticker_count": len(needs_data_tickers),
            "needs_data_coverage_credit": 0,
            "decision_eligible_claims_created": 0,
            "semantics": (
                "NEEDS_DATA is a terminal execution explanation, not investment "
                "coverage or decision eligibility."
            ),
        },
        "operator_recommendation": {
            "resume_same_frozen_inputs": False,
            "paid_provider_work_recommended": False,
            "next_action": (
                "Repair or newly source the missing frozen-date evidence, then create "
                "a fresh reviewed plan. Do not resume this historical campaign."
            ),
        },
    }
    if _raw_file_sha256(state_path) != state_sha256:
        raise ClassicScanPlanError("campaign state changed during reconciliation")
    if _raw_file_sha256(target_path) != target_sha256_before:
        raise ClassicScanPlanError("frozen target changed during reconciliation")
    for binding in log_bindings:
        if _raw_file_sha256(Path(binding["stdout_path"])) != binding["stdout_sha256"]:
            raise ClassicScanPlanError("campaign stdout changed during reconciliation")
        if _raw_file_sha256(Path(binding["stderr_path"])) != binding["stderr_sha256"]:
            raise ClassicScanPlanError("campaign stderr changed during reconciliation")
    output_path = Path(artifact_path)
    protected_paths = {
        state_path.resolve(),
        target_path.resolve(),
        *(Path(row["stdout_path"]).resolve() for row in log_bindings),
        *(Path(row["stderr_path"]).resolve() for row in log_bindings),
    }
    if output_path.resolve() in protected_paths:
        raise ClassicScanPlanError("reconciliation output would overwrite source evidence")
    _write_new_private_json(output_path, report)
    if _raw_file_sha256(state_path) != state_sha256:
        raise ClassicScanPlanError("campaign state changed while publishing reconciliation")
    mode = output_path.stat().st_mode & 0o777
    if mode != 0o600:
        raise ClassicScanPlanError("reconciliation artifact is not mode 0600")
    return {
        "status": "TERMINAL_ACCOUNTING_COMPLETE_COVERAGE_INCOMPLETE",
        "terminal_execution_accounting_complete": True,
        "investment_coverage_complete": False,
        "needs_data_ticker_count": len(needs_data_tickers),
        "artifact_path": str(output_path.resolve()),
        "artifact_sha256": _raw_file_sha256(output_path),
        "artifact_mode": "0600",
        "campaign_state_sha256": state_sha256,
    }


def campaign_status(*, campaign_id: str, output_root: str | Path | None = None) -> dict[str, Any]:
    _path, state = _load_state(_validate_campaign_id(campaign_id), output_root)
    return _campaign_summary(state)


def verify_campaign(*, campaign_id: str, output_root: str | Path | None = None) -> dict[str, Any]:
    """Read-only closure check against the frozen campaign target."""
    _path, state = _load_state(_validate_campaign_id(campaign_id), output_root)
    report = _current_report(state)
    blockers = _manifest_blockers(report)
    membership_matches = (report.get("membership") or {}).get(
        "current_membership_fingerprint"
    ) == state["membership_fingerprint"]
    source_matches = _source_contract()["sha256"] == state["source_contract"]["sha256"]
    target_matches = report.get("target_tickers_sha256") == state["target_tickers_sha256"]
    complete = bool(
        state["status"] == "COMPLETE"
        and float(state["unresolved_cost_reservation_usd"]) == 0
        and report.get("complete")
        and not blockers
        and membership_matches
        and source_matches
        and target_matches
    )
    return {
        "campaign_id": state["campaign_id"],
        "mode": state["mode"],
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "complete": complete,
        "campaign_state_status": state["status"],
        "plan_sha256": state["plan_sha256"],
        "source_contract_sha256": state["source_contract"]["sha256"],
        "source_contract_matches_plan": source_matches,
        "membership_fingerprint": state["membership_fingerprint"],
        "membership_matches_plan": membership_matches,
        "target_ticker_count": state["target_ticker_count"],
        "target_matches_plan": target_matches,
        "preflight_blockers": blockers,
        "provider_binding": state["provider_binding"],
        "authorized_max_cost_usd": state.get("authorized_max_cost_usd"),
        "cumulative_cost_usd": state["cumulative_cost_usd"],
        "unresolved_cost_reservation_usd": state["unresolved_cost_reservation_usd"],
        "pending_cells": report.get("pending_cells") or [],
        "distinct_ticker_totals": report.get("distinct_ticker_totals") or {},
        "state_path": state["state_path"],
    }


__all__ = [
    "CLASSIC_SCAN_MODES",
    "CLASSIC_SCAN_SCHEMA_VERSION",
    "ClassicScanAuthorizationError",
    "ClassicScanError",
    "ClassicScanLocked",
    "ClassicScanPlanError",
    "campaign_status",
    "plan_campaign",
    "reconcile_campaign",
    "reconcile_terminal_data_gaps",
    "run_campaign",
    "validate_coverage_target_file",
    "verify_campaign",
]
