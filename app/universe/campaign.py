from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"

CAMPAIGN_RUNNING = "RUNNING"
CAMPAIGN_PARTIAL = "PARTIAL"
CAMPAIGN_DONE = "DONE"
CAMPAIGN_CANCELLED = "CANCELLED"
CAMPAIGN_FAILED = "FAILED"

ITEM_PENDING = "PENDING"
ITEM_RUNNING = "RUNNING"
ITEM_DONE = "DONE"
ITEM_PARTIAL = "PARTIAL"
ITEM_FAILED = "FAILED"
ITEM_CANCELLED = "CANCELLED"

STOP_COMPLETED = "COMPLETED"
STOP_CANCEL_REQUESTED = "CANCEL_REQUESTED"
STOP_MAX_ITEMS_REACHED = "MAX_ITEMS_REACHED"
STOP_SPEC_CHANGED = "SPEC_CHANGED"

_STATUS_RANK = {"PASS": 0, "WATCH": 1, "FAIL": 2}


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _slug(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return token or "item"


def _dedupe_refs(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _campaign_policy_from_plan(plan: list[dict[str, Any]]) -> str:
    policies = [
        str(((item.get("rollup_params") or {}) if isinstance(item.get("rollup_params"), dict) else {}).get("policy") or "value_first")
        for item in plan
        if isinstance(item, dict)
    ]
    if "value_first_ready" in policies:
        return "value_first_ready"
    if "value_first_durable_returns" in policies:
        return "value_first_durable_returns"
    if "value_first_revenue_resilience" in policies:
        return "value_first_revenue_resilience"
    if "value_first_owner_earnings_hardness" in policies:
        return "value_first_owner_earnings_hardness"
    if "value_first_asset_support_quality" in policies:
        return "value_first_asset_support_quality"
    if "value_first_residual_equity" in policies:
        return "value_first_residual_equity"
    if "value_first_balance_sheet" in policies:
        return "value_first_balance_sheet"
    if "value_first_cash_earnings" in policies:
        return "value_first_cash_earnings"
    if "value_first_reinvestment" in policies:
        return "value_first_reinvestment"
    if "value_first_confidence" in policies:
        return "value_first_confidence"
    if "value_first_trustworthy" in policies:
        return "value_first_trustworthy"
    if "value_first_typed" in policies:
        return "value_first_typed"
    if "value_first_discipline" in policies:
        return "value_first_discipline"
    if "value_first_intangible" in policies:
        return "value_first_intangible"
    if "value_first_quality" in policies:
        return "value_first_quality"
    if "value_first_memory" in policies:
        return "value_first_memory"
    return policies[0] if policies else "value_first"


def _campaign_paths(campaign_run_id: str) -> dict[str, Path]:
    cfg = get_config()
    root = cfg.campaigns_dir / campaign_run_id
    return {
        "root": root,
        "state_path": root / "campaign_state.json",
        "summary_path": root / "campaign_summary.json",
        "log_path": root / "campaign_log.jsonl",
        "master_shortlist_json_path": root / "master_shortlist.json",
        "master_shortlist_md_path": root / "master_shortlist.md",
        "master_watchlist_state_path": root / "master_watchlist_state.json",
        "promotion_state_path": root / "promotion_state.json",
        "promotion_candidates_path": root / "promotion_candidates.json",
        "priority_lanes_path": root / "priority_lanes.json",
        "escalation_plan_path": root / "escalation_plan.json",
        "escalation_queue_path": root / "escalation_queue.json",
        "escalation_summary_path": root / "escalation_summary.json",
    }


def _child_batch_run_id(universe_run_id: str) -> str:
    return f"{universe_run_id}_depth_batch"


def _child_expected_artifacts(universe_run_id: str) -> dict[str, str]:
    cfg = get_config()
    batch_run_id = _child_batch_run_id(universe_run_id)
    autopilot_dir = cfg.outputs_dir / "universe" / universe_run_id / "autopilot"
    batch_dir = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id
    return {
        "autopilot_state_path": str(autopilot_dir / "autopilot_state.json"),
        "autopilot_summary_path": str(autopilot_dir / "autopilot_summary.json"),
        "memo_pack_manifest_path": str(batch_dir / "memo_pack" / "memo_pack_manifest.json"),
        "watchlist_state_path": str(autopilot_dir / "watchlist_state.json"),
        "global_shortlist_path": str(batch_dir / "global_shortlist.json"),
        "global_rollup_path": str(batch_dir / "global_rollup.json"),
        "dossier_pack_manifest_path": str(batch_dir / "dossier_pack" / "dossier_pack_manifest.json"),
    }


def _child_scout_coverage_path(universe_run_id: str) -> Path:
    return get_config().sectors_dir / universe_run_id / "universe_coverage.json"


def _child_validation_summary_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "validation_summary.json"


def _child_filter_audit_summary_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "filter_audit_summary.json"


def _child_borderline_audit_summary_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "borderline_audit_summary.json"


def _child_second_look_summary_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "second_look_summary.json"


def _child_carry_forward_summary_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "carry_forward_summary.json"


def _child_review_plan_summary_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "review_planner" / "review_plan_summary.json"


def _child_review_intake_summary_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "review_intake" / "review_intake_summary.json"


def _child_review_outcomes_summary_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "review_intake" / "review_outcomes_summary.json"


def _child_filter_calibration_summary_path(validation_run_id: str) -> Path:
    return get_config().outputs_dir / "validation" / validation_run_id / "calibration" / "calibration_summary.json"


def _child_review_memory_ref_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "review_memory_ref.json"


def _child_review_cycle_ref_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "review_cycle_ref.json"


def _child_review_compare_ref_path(universe_run_id: str) -> Path:
    return get_config().outputs_dir / "universe" / universe_run_id / "review_compare_ref.json"


def _load_child_scout_facts_lookup(universe_run_id: str) -> dict[str, dict[str, Any]]:
    coverage = _safe_json(_child_scout_coverage_path(universe_run_id))
    lookup: dict[str, dict[str, Any]] = {}
    for row in (coverage.get("rows") or []):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        lookup[ticker] = row
    return lookup


def _overlay_scout_facts_fields(candidate: dict[str, Any], scout_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ticker = str(candidate.get("ticker") or "").strip().upper()
    scout_row = scout_lookup.get(ticker)
    if not ticker or not isinstance(scout_row, dict):
        return candidate

    merged = dict(candidate)
    merged.update(
        {
            "facts_blocker_class": str(scout_row.get("facts_blocker_class") or merged.get("facts_blocker_class") or "FACTS_OK"),
            "facts_blocker_retryable": bool(scout_row.get("facts_blocker_retryable", merged.get("facts_blocker_retryable", False))),
            "facts_blocker_terminal": bool(scout_row.get("facts_blocker_terminal", merged.get("facts_blocker_terminal", False))),
            "facts_blocker_partial_usable": bool(
                scout_row.get("facts_blocker_partial_usable", merged.get("facts_blocker_partial_usable", False))
            ),
            "facts_missing_key_inputs": [
                str(value)
                for value in (scout_row.get("facts_missing_key_inputs") or merged.get("facts_missing_key_inputs") or [])
                if str(value).strip()
            ],
            "facts_retry_recommended": bool(
                scout_row.get("facts_retry_recommended", merged.get("facts_retry_recommended", False))
            ),
            "facts_blocker_reason_codes": [
                str(code)
                for code in (scout_row.get("facts_blocker_reason_codes") or merged.get("facts_blocker_reason_codes") or [])
                if str(code).strip()
            ],
            "facts_recommended_action": str(
                scout_row.get("facts_recommended_action") or merged.get("facts_recommended_action") or "NONE"
            ),
            "fail_due_to_missing_evidence": bool(
                scout_row.get("fail_due_to_missing_evidence", merged.get("fail_due_to_missing_evidence", False))
            ),
            "fail_due_to_economic_weakness": bool(
                scout_row.get("fail_due_to_economic_weakness", merged.get("fail_due_to_economic_weakness", False))
            ),
            "primary_fail_domain": str(scout_row.get("primary_fail_domain") or merged.get("primary_fail_domain") or "NONE"),
        }
    )
    merged["derived_from"] = _dedupe_refs(
        list(candidate.get("derived_from") or [])
        + [f"universe_coverage.rows[{ticker}]"]
    )
    return merged


def _merge_params(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (override or {}).items():
        out[str(key)] = value
    return out


def load_campaign_spec(path_or_args: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_args, dict):
        payload = dict(path_or_args)
        source_path = ""
    else:
        path = Path(str(path_or_args))
        if not path.exists():
            raise ValueError(f"Campaign spec not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to parse campaign spec: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Campaign spec must be a JSON object.")
        source_path = str(path)

    items = [row for row in (payload.get("items") or []) if isinstance(row, dict)]
    if not items:
        raise ValueError("Campaign spec must include a non-empty items list.")

    normalized = dict(payload)
    normalized["items"] = items
    normalized["source_path"] = source_path
    normalized["as_of_date"] = str(normalized.get("as_of_date") or "").strip()
    return normalized


def build_campaign_plan(spec: dict[str, Any]) -> list[dict[str, Any]]:
    campaign_run_id = str(spec.get("campaign_run_id") or "").strip()
    default_as_of = str(spec.get("as_of_date") or "").strip()

    base_scout = spec.get("scout_params") if isinstance(spec.get("scout_params"), dict) else {}
    base_depth = spec.get("depth_batch_params") if isinstance(spec.get("depth_batch_params"), dict) else {}
    base_rollup = spec.get("rollup_params") if isinstance(spec.get("rollup_params"), dict) else {}
    base_dossier = spec.get("dossier_pack_params") if isinstance(spec.get("dossier_pack_params"), dict) else {}
    base_memo = spec.get("memo_pack_params") if isinstance(spec.get("memo_pack_params"), dict) else {}

    seen_labels: set[str] = set()
    plan: list[dict[str, Any]] = []
    for idx, raw in enumerate([row for row in (spec.get("items") or []) if isinstance(row, dict)], start=1):
        label = str(raw.get("label") or f"item_{idx:03d}").strip()
        label_slug = _slug(label)
        if label_slug in seen_labels:
            raise ValueError(f"Duplicate campaign item label after normalization: {label}")
        seen_labels.add(label_slug)

        universe_run_id = f"{campaign_run_id}__{label_slug}" if campaign_run_id else label_slug
        as_of_date = str(raw.get("as_of_date") or default_as_of).strip()
        if not as_of_date:
            raise ValueError(f"Campaign item {label} is missing as_of_date.")

        scout_params = _merge_params(base_scout, raw.get("scout_params") if isinstance(raw.get("scout_params"), dict) else {})
        depth_params = _merge_params(base_depth, raw.get("depth_batch_params") if isinstance(raw.get("depth_batch_params"), dict) else {})
        rollup_params = _merge_params(base_rollup, raw.get("rollup_params") if isinstance(raw.get("rollup_params"), dict) else {})
        dossier_params = _merge_params(base_dossier, raw.get("dossier_pack_params") if isinstance(raw.get("dossier_pack_params"), dict) else {})
        memo_params = _merge_params(base_memo, raw.get("memo_pack_params") if isinstance(raw.get("memo_pack_params"), dict) else {})

        if isinstance(raw.get("universe_file"), str) and str(raw.get("universe_file")).strip():
            scout_params["universe_csv"] = str(raw.get("universe_file"))
        if isinstance(raw.get("universe_source"), str) and str(raw.get("universe_source")).strip():
            scout_params["universe_source"] = str(raw.get("universe_source"))
        if _is_num(raw.get("universe_limit")):
            scout_params["universe_limit"] = int(raw.get("universe_limit"))
        if _is_num(raw.get("top_n")):
            top_n = max(1, int(raw.get("top_n")))
            rollup_params["top_n"] = top_n
            dossier_params["top_n"] = top_n
            memo_params["top_n"] = top_n
        policy = str(raw.get("policy") or rollup_params.get("policy") or "value_first")
        rollup_params["policy"] = policy
        dossier_params["policy"] = policy
        memo_params["policy"] = policy
        if _is_num(raw.get("max_runs")):
            depth_params["max_runs"] = int(raw.get("max_runs"))

        plan.append(
            {
                "item_id": f"{idx:03d}_{label_slug}",
                "plan_index": int(idx - 1),
                "label": label,
                "label_slug": label_slug,
                "as_of_date": as_of_date,
                "universe_file": str(raw.get("universe_file") or ""),
                "universe_source": str(raw.get("universe_source") or ""),
                "scout_params": scout_params,
                "depth_batch_params": depth_params,
                "rollup_params": rollup_params,
                "dossier_pack_params": dossier_params,
                "memo_pack_params": memo_params,
                "status": ITEM_PENDING,
                "child_universe_run_id": universe_run_id,
                "artifact_paths": _child_expected_artifacts(universe_run_id),
                "error": None,
            }
        )
    return plan


def _state_items_signature(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("item_id") or "") for item in items if isinstance(item, dict)]


def _default_state(*, campaign_run_id: str, spec: dict[str, Any], plan: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "campaign_run_id": campaign_run_id,
        "status": CAMPAIGN_RUNNING,
        "stop_reason_code": "",
        "stop_summary": "",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "source_campaign_file": str(spec.get("source_path") or ""),
        "as_of_date": str(spec.get("as_of_date") or ""),
        "policy": _campaign_policy_from_plan(plan),
        "spec_snapshot": {
            "as_of_date": str(spec.get("as_of_date") or ""),
            "item_count": len(plan),
            "item_ids": _state_items_signature(plan),
        },
        "artifact_paths": {},
        "items": plan,
        "cursor_next_idx": 0,
        "active_item_id": "",
    }


def _campaign_artifact_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "campaign_state_path": str(paths["state_path"]),
        "campaign_summary_path": str(paths["summary_path"]),
        "master_shortlist_json_path": str(paths["master_shortlist_json_path"]),
        "master_shortlist_md_path": str(paths["master_shortlist_md_path"]),
        "master_watchlist_state_path": str(paths["master_watchlist_state_path"]),
        "promotion_state_path": str(paths["promotion_state_path"]),
        "promotion_candidates_path": str(paths["promotion_candidates_path"]),
        "priority_lanes_path": str(paths["priority_lanes_path"]),
        "escalation_plan_path": str(paths["escalation_plan_path"]),
        "escalation_queue_path": str(paths["escalation_queue_path"]),
        "escalation_summary_path": str(paths["escalation_summary_path"]),
    }


def _load_director_metadata(paths: dict[str, Path]) -> dict[str, Any]:
    return _safe_json(paths["root"] / "director_metadata.json")


def _write_state_and_summary(paths: dict[str, Path], state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now_iso()
    state["artifact_paths"] = _campaign_artifact_paths(paths)
    _json_write(paths["state_path"], state)
    _json_write(paths["summary_path"], _build_summary(paths, state))


def _artifact_exists(path_value: str | None) -> bool:
    return bool(path_value) and Path(str(path_value)).exists()


def _item_artifacts_exist(item: dict[str, Any]) -> bool:
    artifacts = item.get("artifact_paths") if isinstance(item.get("artifact_paths"), dict) else {}
    required = [
        str(artifacts.get("autopilot_state_path") or ""),
        str(artifacts.get("autopilot_summary_path") or ""),
        str(artifacts.get("memo_pack_manifest_path") or ""),
        str(artifacts.get("watchlist_state_path") or ""),
    ]
    return all(_artifact_exists(path) for path in required)


def _status_rank(status: str) -> int:
    return _STATUS_RANK.get(str(status or "").upper(), 3)


def _desc_key(value: Any) -> tuple[int, float]:
    if _is_num(value):
        return (0, -float(value))
    return (1, 0.0)


def _int_or_zero(value: Any) -> int:
    return int(value) if _is_num(value) else 0


def _confidence_rank(value: Any) -> int:
    token = str(value or "CONFIDENCE_UNKNOWN").upper()
    if token == "HIGH_CONFIDENCE":
        return 0
    if token == "MEDIUM_CONFIDENCE":
        return 1
    if token == "LOW_CONFIDENCE":
        return 2
    return 3


def _value_type_rank(value: Any) -> int:
    token = str(value or "UNKNOWN_VALUE_TYPE").upper()
    if token == "ASSET_BACKED_VALUE":
        return 0
    if token == "EARNINGS_POWER_VALUE":
        return 1
    if token == "QUALITY_VALUE":
        return 2
    if token == "CYCLICAL_VALUE":
        return 3
    if token == "FRAGILE_VALUE":
        return 4
    return 5


def _integrity_rank(value: Any) -> int:
    token = str(value or "INTEGRITY_UNKNOWN").upper()
    if token == "INTEGRITY_OK":
        return 0
    if token == "INTEGRITY_WARNING":
        return 1
    if token == "INTEGRITY_SUSPECT":
        return 2
    return 3


def _readiness_rank(value: Any) -> int:
    token = str(value or "READINESS_UNKNOWN").upper()
    if token == "INVESTABLE_NOW":
        return 0
    if token == "RESEARCH_WORTHY_NOT_READY":
        return 1
    if token == "WATCH_ONLY":
        return 2
    if token == "NOT_INVESTABLE":
        return 3
    return 4


def _reinvestment_rank(value: Any) -> int:
    token = str(value or "REINVESTMENT_EFFICIENCY_UNKNOWN").upper()
    if token == "HIGH_REINVESTMENT_EFFICIENCY":
        return 0
    if token == "MODERATE_REINVESTMENT_EFFICIENCY":
        return 1
    if token == "REINVESTMENT_EFFICIENCY_UNKNOWN":
        return 2
    if token == "LOW_REINVESTMENT_EFFICIENCY":
        return 3
    return 4


def _returns_persistence_rank(value: Any) -> int:
    token = str(value or "RETURNS_PERSISTENCE_UNKNOWN").upper()
    if token == "HIGH_RETURNS_PERSISTENCE":
        return 0
    if token == "MODERATE_RETURNS_PERSISTENCE":
        return 1
    if token == "RETURNS_PERSISTENCE_UNKNOWN":
        return 2
    if token == "LOW_RETURNS_PERSISTENCE":
        return 3
    return 4


def _revenue_dependence_rank(value: Any) -> int:
    token = str(value or "REVENUE_DEPENDENCE_UNKNOWN").upper()
    if token == "LOW_REVENUE_DEPENDENCE_RISK":
        return 0
    if token == "MODERATE_REVENUE_DEPENDENCE_RISK":
        return 1
    if token == "REVENUE_DEPENDENCE_UNKNOWN":
        return 2
    if token == "HIGH_REVENUE_DEPENDENCE_RISK":
        return 3
    return 4


def _accounting_quality_rank(value: Any) -> int:
    token = str(value or "ACCOUNTING_QUALITY_UNKNOWN").upper()
    if token == "HIGH_ACCOUNTING_QUALITY":
        return 0
    if token == "MODERATE_ACCOUNTING_QUALITY":
        return 1
    if token == "ACCOUNTING_QUALITY_UNKNOWN":
        return 2
    if token == "LOW_ACCOUNTING_QUALITY":
        return 3
    return 4


def _maintenance_capex_credibility_rank(value: Any) -> int:
    token = str(value or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN").upper()
    if token == "HIGH_MAINTENANCE_CAPEX_CREDIBILITY":
        return 0
    if token == "MODERATE_MAINTENANCE_CAPEX_CREDIBILITY":
        return 1
    if token == "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN":
        return 2
    if token == "LOW_MAINTENANCE_CAPEX_CREDIBILITY":
        return 3
    return 4


def _asset_quality_rank(value: Any) -> int:
    token = str(value or "ASSET_QUALITY_UNKNOWN").upper()
    if token == "HIGH_ASSET_QUALITY":
        return 0
    if token == "MODERATE_ASSET_QUALITY":
        return 1
    if token == "ASSET_QUALITY_UNKNOWN":
        return 2
    if token == "LOW_ASSET_QUALITY":
        return 3
    return 4


def _downside_realization_rank(value: Any) -> int:
    token = str(value or "DOWNSIDE_REALIZATION_UNKNOWN").upper()
    if token == "HIGH_DOWNSIDE_REALIZATION_CREDIBILITY":
        return 0
    if token == "MODERATE_DOWNSIDE_REALIZATION_CREDIBILITY":
        return 1
    if token == "DOWNSIDE_REALIZATION_UNKNOWN":
        return 2
    if token == "LOW_DOWNSIDE_REALIZATION_CREDIBILITY":
        return 3
    return 4


def _obligation_burden_rank(value: Any) -> int:
    token = str(value or "OBLIGATION_BURDEN_UNKNOWN").upper()
    if token == "LOW_OBLIGATION_BURDEN":
        return 0
    if token == "MODERATE_OBLIGATION_BURDEN":
        return 1
    if token == "OBLIGATION_BURDEN_UNKNOWN":
        return 2
    if token == "HIGH_OBLIGATION_BURDEN":
        return 3
    return 4


def _claim_priority_pressure_rank(value: Any) -> int:
    token = str(value or "CLAIM_PRIORITY_PRESSURE_UNKNOWN").upper()
    if token == "LOW_CLAIM_PRIORITY_PRESSURE":
        return 0
    if token == "MODERATE_CLAIM_PRIORITY_PRESSURE":
        return 1
    if token == "CLAIM_PRIORITY_PRESSURE_UNKNOWN":
        return 2
    if token == "HIGH_CLAIM_PRIORITY_PRESSURE":
        return 3
    return 4


def _asset_intensity_rank(value: Any) -> int:
    token = str(value or "ASSET_INTENSITY_UNKNOWN").upper()
    if token == "LOW_ASSET_INTENSITY":
        return 0
    if token == "MODERATE_ASSET_INTENSITY":
        return 1
    if token == "ASSET_INTENSITY_UNKNOWN":
        return 2
    if token == "HIGH_ASSET_INTENSITY":
        return 3
    return 4


def _balance_sheet_stress_rank(value: Any) -> int:
    token = str(value or "BALANCE_SHEET_STRESS_UNKNOWN").upper()
    if token == "LOW_BALANCE_SHEET_STRESS":
        return 0
    if token == "MODERATE_BALANCE_SHEET_STRESS":
        return 1
    if token == "BALANCE_SHEET_STRESS_UNKNOWN":
        return 2
    if token == "HIGH_BALANCE_SHEET_STRESS":
        return 3
    return 4


def _refinancing_risk_rank(value: Any) -> int:
    token = str(value or "REFINANCING_RISK_UNKNOWN").upper()
    if token == "LOW_REFINANCING_RISK":
        return 0
    if token == "MODERATE_REFINANCING_RISK":
        return 1
    if token == "REFINANCING_RISK_UNKNOWN":
        return 2
    if token == "HIGH_REFINANCING_RISK":
        return 3
    return 4


def _capital_allocation_discipline_rank(value: Any) -> int:
    token = str(value or "CAPITAL_ALLOCATION_UNKNOWN").upper()
    if token == "OWNER_FRIENDLY_DISCIPLINED":
        return 0
    if token == "MIXED_CAPITAL_ALLOCATION":
        return 1
    if token == "CAPITAL_ALLOCATION_UNKNOWN":
        return 2
    if token == "OWNER_DILUTIVE_OR_DESTRUCTIVE":
        return 3
    return 4


def _normalization_credibility_rank(value: Any) -> int:
    token = str(value or "NORMALIZATION_CREDIBILITY_UNKNOWN").upper()
    if token == "HIGH_NORMALIZATION_CREDIBILITY":
        return 0
    if token == "MODERATE_NORMALIZATION_CREDIBILITY":
        return 1
    if token == "NORMALIZATION_CREDIBILITY_UNKNOWN":
        return 2
    if token == "LOW_NORMALIZATION_CREDIBILITY":
        return 3
    return 4


def _cyclical_risk_rank(value: Any) -> int:
    token = str(value or "CYCLE_RISK_UNKNOWN").upper()
    if token == "MID_CYCLE_REASONABLE":
        return 0
    if token == "TROUGH_EARNINGS_RISK":
        return 1
    if token == "CYCLE_RISK_UNKNOWN":
        return 2
    if token == "PEAK_EARNINGS_RISK":
        return 3
    return 2


def _memory_priority_lookup() -> dict[str, dict[str, Any]]:
    from app.universe.research_memory import load_research_memory

    memory = load_research_memory()
    return memory.get("tickers") if isinstance(memory.get("tickers"), dict) else {}


def _memory_priority_fields(ticker: str, memory_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entry = memory_lookup.get(str(ticker).upper()) if isinstance(memory_lookup, dict) else None
    if not isinstance(entry, dict):
        return {
            "memory_priority_total": 0,
            "memory_priority_reason_codes": [],
            "memory_priority_source_campaign_run_ids": [],
        }
    memory_priority = entry.get("memory_priority") if isinstance(entry.get("memory_priority"), dict) else {}
    return {
        "memory_priority_total": int(memory_priority.get("memory_priority_total") or entry.get("memory_priority_total") or 0),
        "memory_priority_reason_codes": [
            str(code)
            for code in (
                memory_priority.get("memory_priority_reason_codes")
                or entry.get("memory_priority_reason_codes")
                or []
            )
            if str(code).strip()
        ],
        "memory_priority_source_campaign_run_ids": [
            str(run_id)
            for run_id in (
                memory_priority.get("source_campaign_run_ids")
                or entry.get("memory_priority_source_campaign_run_ids")
                or []
            )
            if str(run_id).strip()
        ],
    }


def _effective_campaign_policy(state: dict[str, Any]) -> str:
    policy = str(state.get("policy") or "").strip()
    if policy:
        return policy
    items = [row for row in (state.get("items") or []) if isinstance(row, dict)]
    return _campaign_policy_from_plan(items)


def _master_sort_key(row: dict[str, Any], *, policy: str = "value_first") -> tuple[Any, ...]:
    base = (
        _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
        _desc_key(row.get("implied_return_base", UNKNOWN)),
    )
    policy_norm = str(policy or "value_first")
    if policy_norm == "value_first_ready":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_durable_returns":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _returns_persistence_rank(row.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")),
            _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
            _capital_allocation_discipline_rank(
                row.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN")
            ),
            _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _normalization_credibility_rank(
                row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
            ),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_revenue_resilience":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _returns_persistence_rank(row.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")),
            _revenue_dependence_rank(row.get("revenue_dependence_risk_class", "REVENUE_DEPENDENCE_UNKNOWN")),
            _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
            _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
            _capital_allocation_discipline_rank(
                row.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN")
            ),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _normalization_credibility_rank(
                row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
            ),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_owner_earnings_hardness":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _maintenance_capex_credibility_rank(
                row.get("maintenance_capex_credibility_class", "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN")
            ),
            _asset_intensity_rank(row.get("asset_intensity_class", "ASSET_INTENSITY_UNKNOWN")),
            _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
            _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
            _capital_allocation_discipline_rank(
                row.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN")
            ),
            _returns_persistence_rank(row.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _normalization_credibility_rank(
                row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
            ),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_asset_support_quality":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _downside_realization_rank(
                row.get(
                    "downside_realization_credibility_class",
                    "DOWNSIDE_REALIZATION_UNKNOWN",
                )
            ),
            _asset_quality_rank(row.get("asset_quality_class", "ASSET_QUALITY_UNKNOWN")),
            _balance_sheet_stress_rank(
                row.get("balance_sheet_stress_class", "BALANCE_SHEET_STRESS_UNKNOWN")
            ),
            _accounting_quality_rank(
                row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")
            ),
            _maintenance_capex_credibility_rank(
                row.get(
                    "maintenance_capex_credibility_class",
                    "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN",
                )
            ),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _normalization_credibility_rank(
                row.get(
                    "normalization_credibility_class",
                    "NORMALIZATION_CREDIBILITY_UNKNOWN",
                )
            ),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_residual_equity":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _claim_priority_pressure_rank(
                row.get("claim_priority_pressure_class", "CLAIM_PRIORITY_PRESSURE_UNKNOWN")
            ),
            _obligation_burden_rank(row.get("obligation_burden_class", "OBLIGATION_BURDEN_UNKNOWN")),
            _downside_realization_rank(
                row.get("downside_realization_credibility_class", "DOWNSIDE_REALIZATION_UNKNOWN")
            ),
            _balance_sheet_stress_rank(
                row.get("balance_sheet_stress_class", "BALANCE_SHEET_STRESS_UNKNOWN")
            ),
            _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
            _maintenance_capex_credibility_rank(
                row.get("maintenance_capex_credibility_class", "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN")
            ),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _normalization_credibility_rank(
                row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
            ),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_balance_sheet":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _balance_sheet_stress_rank(
                row.get("balance_sheet_stress_class", "BALANCE_SHEET_STRESS_UNKNOWN")
            ),
            _refinancing_risk_rank(row.get("refinancing_risk_class", "REFINANCING_RISK_UNKNOWN")),
            _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
            _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
            _capital_allocation_discipline_rank(
                row.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN")
            ),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _normalization_credibility_rank(
                row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
            ),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_cash_earnings":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
            _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
            _capital_allocation_discipline_rank(
                row.get("capital_allocation_discipline_class", "CAPITAL_ALLOCATION_UNKNOWN")
            ),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _normalization_credibility_rank(
                row.get("normalization_credibility_class", "NORMALIZATION_CREDIBILITY_UNKNOWN")
            ),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_reinvestment":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _reinvestment_rank(row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
            _desc_key(row.get("capital_allocation_score", UNKNOWN)),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_trustworthy":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _desc_key(row.get("implied_return_base", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_typed":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _desc_key(row.get("implied_return_base", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_confidence":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _desc_key(row.get("implied_return_base", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _desc_key(row.get("mos_epv", UNKNOWN)),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_discipline":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _desc_key(row.get("implied_return_base", UNKNOWN)),
            _desc_key(row.get("mos_epv", UNKNOWN)),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_memory":
        return (
            *base,
            -int(row.get("memory_priority_total") or 0),
            _desc_key(row.get("mos_epv", UNKNOWN)),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            _desc_key(row.get("composite_score_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_intangible":
        return (
            *base,
            _desc_key(row.get("mos_epv", UNKNOWN)),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            -int(row.get("memory_priority_total") or 0),
            _desc_key(row.get("composite_score_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_quality":
        return (
            *base,
            _desc_key(row.get("mos_epv", UNKNOWN)),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            _desc_key(row.get("intangible_economics_total", UNKNOWN)),
            -int(row.get("memory_priority_total") or 0),
            _desc_key(row.get("composite_score_total", UNKNOWN)),
            str(row.get("ticker") or ""),
        )
    if policy_norm == "value_first_cycle_aware":
        return (
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            _cyclical_risk_rank(row.get("cyclical_valuation_risk_class", "CYCLE_RISK_UNKNOWN")),
            _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
            _desc_key(row.get("mos_to_floor", UNKNOWN)),
            _desc_key(row.get("implied_return_base", UNKNOWN)),
            _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
            _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
            _desc_key(row.get("oe_quality_total", UNKNOWN)),
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        )
    return (
        *base,
        _desc_key(row.get("mos_epv", UNKNOWN)),
        _desc_key(row.get("owner_earnings_yield_ev_3y", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("composite_score_total", UNKNOWN)),
        -int(row.get("memory_priority_total") or 0),
        str(row.get("ticker") or ""),
    )


def _campaign_markdown(*, campaign_run_id: str, rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Campaign Master Shortlist")
    lines.append("")
    lines.append(f"- Campaign Run: `{campaign_run_id}`")
    lines.append(f"- Generated At: `{utc_now_iso()}`")
    lines.append("")
    lines.append("| Rank | Ticker | Gate | Ready | MOS Floor | Implied Return | Valuation Conf. | Integrity | BS Stress | Refi Risk | Accounting | Value Type | Fragility | MOS EPV | OE Yield EV 3Y | OE Quality | Intangible | Reinvestment | Downside Support | Memory | Blocker | Next Step | Memo |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |")
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("ticker") or ""),
                    str(row.get("value_gate_status") or UNKNOWN),
                    str(row.get("investment_readiness_class") or UNKNOWN),
                    str(row.get("mos_to_floor") if _is_num(row.get("mos_to_floor")) else UNKNOWN),
                    str(row.get("implied_return_base") if _is_num(row.get("implied_return_base")) else UNKNOWN),
                    str(row.get("valuation_confidence_class") or UNKNOWN),
                    str(row.get("valuation_integrity_class") or UNKNOWN),
                    str(row.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN"),
                    str(row.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"),
                    str(row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"),
                    str(row.get("value_type_primary") or UNKNOWN),
                    str(row.get("valuation_fragility_status") or UNKNOWN),
                    str(row.get("mos_epv") if _is_num(row.get("mos_epv")) else UNKNOWN),
                    str(row.get("owner_earnings_yield_ev_3y") if _is_num(row.get("owner_earnings_yield_ev_3y")) else UNKNOWN),
                    str(row.get("oe_quality_total") if _is_num(row.get("oe_quality_total")) else UNKNOWN),
                    str(row.get("intangible_economics_total") if _is_num(row.get("intangible_economics_total")) else UNKNOWN),
                    str(row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"),
                    str(row.get("downside_support_type") or UNKNOWN),
                    str(int(row.get("memory_priority_total") or 0)),
                    str(row.get("blocker_stack_primary") or row.get("primary_blocker") or UNKNOWN),
                    str(row.get("primary_next_step") or UNKNOWN),
                    str(row.get("memo_path") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _candidate_from_memo(
    *,
    item: dict[str, Any],
    memo_payload: dict[str, Any],
    memo_paths: dict[str, str],
    shortlist_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    header = memo_payload.get("header") if isinstance(memo_payload.get("header"), dict) else {}
    decision = memo_payload.get("decision_snapshot") if isinstance(memo_payload.get("decision_snapshot"), dict) else {}
    gd = memo_payload.get("graham_dodd") if isinstance(memo_payload.get("graham_dodd"), dict) else {}
    yields = (
        memo_payload.get("owner_earnings_and_yield")
        if isinstance(memo_payload.get("owner_earnings_and_yield"), dict)
        else {}
    )
    shortlist_row = shortlist_lookup.get(str(header.get("ticker") or "").upper(), {})
    source_runs = [
        {
            "campaign_item": str(item.get("label") or ""),
            "universe_run_id": str(item.get("child_universe_run_id") or ""),
            "batch_run_id": _child_batch_run_id(str(item.get("child_universe_run_id") or "")),
        }
    ]
    return {
        "ticker": str(header.get("ticker") or "").upper(),
        "campaign_item": str(item.get("label") or ""),
        "item_index": int(item.get("plan_index") or 0),
        "universe_run_id": str(item.get("child_universe_run_id") or ""),
        "batch_run_id": _child_batch_run_id(str(item.get("child_universe_run_id") or "")),
        "best_rank_seen": int(memo_payload.get("rank_global") or 0),
        "value_gate_status": str(decision.get("value_gate_status") or memo_payload.get("value_gate_status") or UNKNOWN).upper(),
        "implied_return_base": decision.get("implied_return_base", UNKNOWN),
        "mos_epv": gd.get("mos_epv", UNKNOWN),
        "mos_netnet": gd.get("mos_netnet", UNKNOWN),
        "owner_earnings_yield_ev_3y": yields.get("owner_earnings_yield_ev_3y", UNKNOWN),
        "yield_metric_used": str(yields.get("yield_metric_used") or UNKNOWN),
        "owner_earnings_stability_score": memo_payload.get("owner_earnings_quality", {}).get("owner_earnings_stability_score", UNKNOWN)
        if isinstance(memo_payload.get("owner_earnings_quality"), dict)
        else shortlist_row.get("owner_earnings_stability_score", UNKNOWN),
        "capital_allocation_score": memo_payload.get("owner_earnings_quality", {}).get("capital_allocation_score", UNKNOWN)
        if isinstance(memo_payload.get("owner_earnings_quality"), dict)
        else shortlist_row.get("capital_allocation_score", UNKNOWN),
        "cash_conversion_score": memo_payload.get("owner_earnings_quality", {}).get("cash_conversion_score", UNKNOWN)
        if isinstance(memo_payload.get("owner_earnings_quality"), dict)
        else shortlist_row.get("cash_conversion_score", UNKNOWN),
        "oe_quality_total": memo_payload.get("owner_earnings_quality", {}).get("oe_quality_total", UNKNOWN)
        if isinstance(memo_payload.get("owner_earnings_quality"), dict)
        else shortlist_row.get("oe_quality_total", UNKNOWN),
        "oe_quality_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("owner_earnings_quality", {}).get("oe_quality_reason_codes")
                    if isinstance(memo_payload.get("owner_earnings_quality"), dict)
                    else shortlist_row.get("oe_quality_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "gross_margin_durability_score": memo_payload.get("modern_intangible_economics", {}).get("gross_margin_durability_score", UNKNOWN)
        if isinstance(memo_payload.get("modern_intangible_economics"), dict)
        else shortlist_row.get("gross_margin_durability_score", UNKNOWN),
        "balance_sheet_optionality_score": memo_payload.get("modern_intangible_economics", {}).get("balance_sheet_optionality_score", UNKNOWN)
        if isinstance(memo_payload.get("modern_intangible_economics"), dict)
        else shortlist_row.get("balance_sheet_optionality_score", UNKNOWN),
        "cycle_resilience_score": memo_payload.get("modern_intangible_economics", {}).get("cycle_resilience_score", UNKNOWN)
        if isinstance(memo_payload.get("modern_intangible_economics"), dict)
        else shortlist_row.get("cycle_resilience_score", UNKNOWN),
        "rnd_productivity_score": memo_payload.get("modern_intangible_economics", {}).get("rnd_productivity_score", UNKNOWN)
        if isinstance(memo_payload.get("modern_intangible_economics"), dict)
        else shortlist_row.get("rnd_productivity_score", UNKNOWN),
        "sga_leverage_score": memo_payload.get("modern_intangible_economics", {}).get("sga_leverage_score", UNKNOWN)
        if isinstance(memo_payload.get("modern_intangible_economics"), dict)
        else shortlist_row.get("sga_leverage_score", UNKNOWN),
        "owner_value_capture_score": memo_payload.get("modern_intangible_economics", {}).get("owner_value_capture_score", UNKNOWN)
        if isinstance(memo_payload.get("modern_intangible_economics"), dict)
        else shortlist_row.get("owner_value_capture_score", UNKNOWN),
        "intangible_economics_total": memo_payload.get("modern_intangible_economics", {}).get("intangible_economics_total", UNKNOWN)
        if isinstance(memo_payload.get("modern_intangible_economics"), dict)
        else shortlist_row.get("intangible_economics_total", UNKNOWN),
        "rnd_productivity_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("modern_intangible_economics", {}).get("rnd_productivity_reason_codes")
                    if isinstance(memo_payload.get("modern_intangible_economics"), dict)
                    else shortlist_row.get("rnd_productivity_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "sga_leverage_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("modern_intangible_economics", {}).get("sga_leverage_reason_codes")
                    if isinstance(memo_payload.get("modern_intangible_economics"), dict)
                    else shortlist_row.get("sga_leverage_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "owner_value_capture_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("modern_intangible_economics", {}).get("owner_value_capture_reason_codes")
                    if isinstance(memo_payload.get("modern_intangible_economics"), dict)
                    else shortlist_row.get("owner_value_capture_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "reinvestment_efficiency_class": str(
            (
                memo_payload.get("incremental_reinvestment_efficiency", {}).get("reinvestment_efficiency_class")
                if isinstance(memo_payload.get("incremental_reinvestment_efficiency"), dict)
                else shortlist_row.get("reinvestment_efficiency_class")
            )
            or "REINVESTMENT_EFFICIENCY_UNKNOWN"
        ),
        "reinvestment_efficiency_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("incremental_reinvestment_efficiency", {}).get(
                        "reinvestment_efficiency_reason_codes"
                    )
                    if isinstance(memo_payload.get("incremental_reinvestment_efficiency"), dict)
                    else shortlist_row.get("reinvestment_efficiency_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "reinvestment_support_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("incremental_reinvestment_efficiency", {}).get(
                        "reinvestment_support_signals"
                    )
                    if isinstance(memo_payload.get("incremental_reinvestment_efficiency"), dict)
                    else shortlist_row.get("reinvestment_support_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "reinvestment_headwind_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("incremental_reinvestment_efficiency", {}).get(
                        "reinvestment_headwind_signals"
                    )
                    if isinstance(memo_payload.get("incremental_reinvestment_efficiency"), dict)
                    else shortlist_row.get("reinvestment_headwind_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "primary_reinvestment_caution": str(
            (
                memo_payload.get("incremental_reinvestment_efficiency", {}).get(
                    "primary_reinvestment_caution"
                )
                if isinstance(memo_payload.get("incremental_reinvestment_efficiency"), dict)
                else shortlist_row.get("primary_reinvestment_caution")
            )
            or "REINVESTMENT_UNCLEAR"
        ),
        "reinvestment_efficiency_summary": str(
            (
                memo_payload.get("incremental_reinvestment_efficiency", {}).get(
                    "reinvestment_efficiency_summary"
                )
                if isinstance(memo_payload.get("incremental_reinvestment_efficiency"), dict)
                else shortlist_row.get("reinvestment_efficiency_summary")
            )
            or ""
        ),
        "returns_persistence_class": str(
            (
                memo_payload.get("returns_on_capital_persistence_economic_durability", {}).get(
                    "returns_persistence_class"
                )
                if isinstance(memo_payload.get("returns_on_capital_persistence_economic_durability"), dict)
                else shortlist_row.get("returns_persistence_class")
            )
            or "RETURNS_PERSISTENCE_UNKNOWN"
        ),
        "returns_persistence_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("returns_on_capital_persistence_economic_durability", {}).get(
                        "returns_persistence_reason_codes"
                    )
                    if isinstance(memo_payload.get("returns_on_capital_persistence_economic_durability"), dict)
                    else shortlist_row.get("returns_persistence_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "returns_support_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("returns_on_capital_persistence_economic_durability", {}).get(
                        "returns_support_signals"
                    )
                    if isinstance(memo_payload.get("returns_on_capital_persistence_economic_durability"), dict)
                    else shortlist_row.get("returns_support_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "returns_headwind_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("returns_on_capital_persistence_economic_durability", {}).get(
                        "returns_headwind_signals"
                    )
                    if isinstance(memo_payload.get("returns_on_capital_persistence_economic_durability"), dict)
                    else shortlist_row.get("returns_headwind_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "primary_returns_caution": str(
            (
                memo_payload.get("returns_on_capital_persistence_economic_durability", {}).get(
                    "primary_returns_caution"
                )
                if isinstance(memo_payload.get("returns_on_capital_persistence_economic_durability"), dict)
                else shortlist_row.get("primary_returns_caution")
            )
            or "RETURNS_DURABILITY_UNCLEAR"
        ),
        "economic_durability_summary": str(
            (
                memo_payload.get("returns_on_capital_persistence_economic_durability", {}).get(
                    "economic_durability_summary"
                )
                if isinstance(memo_payload.get("returns_on_capital_persistence_economic_durability"), dict)
                else shortlist_row.get("economic_durability_summary")
            )
            or ""
        ),
        "revenue_dependence_risk_class": str(
            (
                memo_payload.get("customer_concentration_revenue_dependence_risk", {}).get(
                    "revenue_dependence_risk_class"
                )
                if isinstance(memo_payload.get("customer_concentration_revenue_dependence_risk"), dict)
                else shortlist_row.get("revenue_dependence_risk_class")
            )
            or "REVENUE_DEPENDENCE_UNKNOWN"
        ),
        "revenue_dependence_risk_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("customer_concentration_revenue_dependence_risk", {}).get(
                        "revenue_dependence_risk_reason_codes"
                    )
                    if isinstance(memo_payload.get("customer_concentration_revenue_dependence_risk"), dict)
                    else shortlist_row.get("revenue_dependence_risk_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "revenue_dependence_support_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("customer_concentration_revenue_dependence_risk", {}).get(
                        "revenue_dependence_support_signals"
                    )
                    if isinstance(memo_payload.get("customer_concentration_revenue_dependence_risk"), dict)
                    else shortlist_row.get("revenue_dependence_support_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "revenue_dependence_headwind_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("customer_concentration_revenue_dependence_risk", {}).get(
                        "revenue_dependence_headwind_signals"
                    )
                    if isinstance(memo_payload.get("customer_concentration_revenue_dependence_risk"), dict)
                    else shortlist_row.get("revenue_dependence_headwind_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "primary_revenue_dependence_caution": str(
            (
                memo_payload.get("customer_concentration_revenue_dependence_risk", {}).get(
                    "primary_revenue_dependence_caution"
                )
                if isinstance(memo_payload.get("customer_concentration_revenue_dependence_risk"), dict)
                else shortlist_row.get("primary_revenue_dependence_caution")
            )
            or "REVENUE_BASE_UNCLEAR"
        ),
        "revenue_fragility_summary": str(
            (
                memo_payload.get("customer_concentration_revenue_dependence_risk", {}).get(
                    "revenue_fragility_summary"
                )
                if isinstance(memo_payload.get("customer_concentration_revenue_dependence_risk"), dict)
                else shortlist_row.get("revenue_fragility_summary")
            )
            or ""
        ),
        "asset_intensity_class": str(
            (
                memo_payload.get("maintenance_capex_asset_intensity_discipline", {}).get(
                    "asset_intensity_class"
                )
                if isinstance(memo_payload.get("maintenance_capex_asset_intensity_discipline"), dict)
                else shortlist_row.get("asset_intensity_class")
            )
            or "ASSET_INTENSITY_UNKNOWN"
        ),
        "asset_intensity_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("maintenance_capex_asset_intensity_discipline", {}).get(
                        "asset_intensity_reason_codes"
                    )
                    if isinstance(memo_payload.get("maintenance_capex_asset_intensity_discipline"), dict)
                    else shortlist_row.get("asset_intensity_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "maintenance_capex_credibility_class": str(
            (
                memo_payload.get("maintenance_capex_asset_intensity_discipline", {}).get(
                    "maintenance_capex_credibility_class"
                )
                if isinstance(memo_payload.get("maintenance_capex_asset_intensity_discipline"), dict)
                else shortlist_row.get("maintenance_capex_credibility_class")
            )
            or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
        ),
        "maintenance_capex_credibility_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("maintenance_capex_asset_intensity_discipline", {}).get(
                        "maintenance_capex_credibility_reason_codes"
                    )
                    if isinstance(memo_payload.get("maintenance_capex_asset_intensity_discipline"), dict)
                    else shortlist_row.get("maintenance_capex_credibility_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "maintenance_capex_support_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("maintenance_capex_asset_intensity_discipline", {}).get(
                        "maintenance_capex_support_signals"
                    )
                    if isinstance(memo_payload.get("maintenance_capex_asset_intensity_discipline"), dict)
                    else shortlist_row.get("maintenance_capex_support_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "maintenance_capex_headwind_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("maintenance_capex_asset_intensity_discipline", {}).get(
                        "maintenance_capex_headwind_signals"
                    )
                    if isinstance(memo_payload.get("maintenance_capex_asset_intensity_discipline"), dict)
                    else shortlist_row.get("maintenance_capex_headwind_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "primary_maintenance_capex_caution": str(
            (
                memo_payload.get("maintenance_capex_asset_intensity_discipline", {}).get(
                    "primary_maintenance_capex_caution"
                )
                if isinstance(memo_payload.get("maintenance_capex_asset_intensity_discipline"), dict)
                else shortlist_row.get("primary_maintenance_capex_caution")
            )
            or "OWNER_EARNINGS_UNCLEAR"
        ),
        "maintenance_capex_discipline_summary": str(
            (
                memo_payload.get("maintenance_capex_asset_intensity_discipline", {}).get(
                    "maintenance_capex_discipline_summary"
                )
                if isinstance(memo_payload.get("maintenance_capex_asset_intensity_discipline"), dict)
                else shortlist_row.get("maintenance_capex_discipline_summary")
            )
            or ""
        ),
        "accounting_quality_class": str(
            (
                memo_payload.get("accounting_quality_cash_earnings_discipline", {}).get(
                    "accounting_quality_class"
                )
                if isinstance(memo_payload.get("accounting_quality_cash_earnings_discipline"), dict)
                else shortlist_row.get("accounting_quality_class")
            )
            or "ACCOUNTING_QUALITY_UNKNOWN"
        ),
        "accounting_quality_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("accounting_quality_cash_earnings_discipline", {}).get(
                        "accounting_quality_reason_codes"
                    )
                    if isinstance(memo_payload.get("accounting_quality_cash_earnings_discipline"), dict)
                    else shortlist_row.get("accounting_quality_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "cash_earnings_support_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("accounting_quality_cash_earnings_discipline", {}).get(
                        "cash_earnings_support_signals"
                    )
                    if isinstance(memo_payload.get("accounting_quality_cash_earnings_discipline"), dict)
                    else shortlist_row.get("cash_earnings_support_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "cash_earnings_headwind_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("accounting_quality_cash_earnings_discipline", {}).get(
                        "cash_earnings_headwind_signals"
                    )
                    if isinstance(memo_payload.get("accounting_quality_cash_earnings_discipline"), dict)
                    else shortlist_row.get("cash_earnings_headwind_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "primary_accounting_caution": str(
            (
                memo_payload.get("accounting_quality_cash_earnings_discipline", {}).get(
                    "primary_accounting_caution"
                )
                if isinstance(memo_payload.get("accounting_quality_cash_earnings_discipline"), dict)
                else shortlist_row.get("primary_accounting_caution")
            )
            or "ACCOUNTING_QUALITY_UNCLEAR"
        ),
        "cash_earnings_discipline_summary": str(
            (
                memo_payload.get("accounting_quality_cash_earnings_discipline", {}).get(
                    "cash_earnings_discipline_summary"
                )
                if isinstance(memo_payload.get("accounting_quality_cash_earnings_discipline"), dict)
                else shortlist_row.get("cash_earnings_discipline_summary")
            )
            or ""
        ),
        "balance_sheet_stress_class": str(
            (
                memo_payload.get("balance_sheet_stress_refinancing_risk", {}).get(
                    "balance_sheet_stress_class"
                )
                if isinstance(memo_payload.get("balance_sheet_stress_refinancing_risk"), dict)
                else shortlist_row.get("balance_sheet_stress_class")
            )
            or "BALANCE_SHEET_STRESS_UNKNOWN"
        ),
        "balance_sheet_stress_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("balance_sheet_stress_refinancing_risk", {}).get(
                        "balance_sheet_stress_reason_codes"
                    )
                    if isinstance(memo_payload.get("balance_sheet_stress_refinancing_risk"), dict)
                    else shortlist_row.get("balance_sheet_stress_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "refinancing_risk_class": str(
            (
                memo_payload.get("balance_sheet_stress_refinancing_risk", {}).get(
                    "refinancing_risk_class"
                )
                if isinstance(memo_payload.get("balance_sheet_stress_refinancing_risk"), dict)
                else shortlist_row.get("refinancing_risk_class")
            )
            or "REFINANCING_RISK_UNKNOWN"
        ),
        "refinancing_risk_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("balance_sheet_stress_refinancing_risk", {}).get(
                        "refinancing_risk_reason_codes"
                    )
                    if isinstance(memo_payload.get("balance_sheet_stress_refinancing_risk"), dict)
                    else shortlist_row.get("refinancing_risk_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "balance_sheet_support_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("balance_sheet_stress_refinancing_risk", {}).get(
                        "balance_sheet_support_signals"
                    )
                    if isinstance(memo_payload.get("balance_sheet_stress_refinancing_risk"), dict)
                    else shortlist_row.get("balance_sheet_support_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "balance_sheet_headwind_signals": [
            str(code)
            for code in (
                (
                    memo_payload.get("balance_sheet_stress_refinancing_risk", {}).get(
                        "balance_sheet_headwind_signals"
                    )
                    if isinstance(memo_payload.get("balance_sheet_stress_refinancing_risk"), dict)
                    else shortlist_row.get("balance_sheet_headwind_signals")
                )
                or []
            )
            if str(code).strip()
        ],
        "primary_balance_sheet_caution": str(
            (
                memo_payload.get("balance_sheet_stress_refinancing_risk", {}).get(
                    "primary_balance_sheet_caution"
                )
                if isinstance(memo_payload.get("balance_sheet_stress_refinancing_risk"), dict)
                else shortlist_row.get("primary_balance_sheet_caution")
            )
            or "BALANCE_SHEET_UNCLEAR"
        ),
        "balance_sheet_discipline_summary": str(
            (
                memo_payload.get("balance_sheet_stress_refinancing_risk", {}).get(
                    "balance_sheet_discipline_summary"
                )
                if isinstance(memo_payload.get("balance_sheet_stress_refinancing_risk"), dict)
                else shortlist_row.get("balance_sheet_discipline_summary")
            )
            or ""
        ),
        "normalized_earnings_power_value": memo_payload.get("intrinsic_value_discipline", {}).get("normalized_earnings_power_value", UNKNOWN)
        if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
        else shortlist_row.get("normalized_earnings_power_value", UNKNOWN),
        "normalized_earnings_power_method_used": str(
            (
                memo_payload.get("intrinsic_value_discipline", {}).get("normalized_earnings_power_method_used")
                if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                else shortlist_row.get("normalized_earnings_power_method_used")
            )
            or UNKNOWN
        ),
        "normalized_earnings_power_status": str(
            (
                memo_payload.get("intrinsic_value_discipline", {}).get("normalized_earnings_power_status")
                if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                else shortlist_row.get("normalized_earnings_power_status")
            )
            or UNKNOWN
        ),
        "normalized_earnings_power_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("intrinsic_value_discipline", {}).get("normalized_earnings_power_reason_codes")
                    if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                    else shortlist_row.get("normalized_earnings_power_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "intrinsic_floor": memo_payload.get("intrinsic_value_discipline", {}).get("intrinsic_floor", UNKNOWN)
        if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
        else shortlist_row.get("intrinsic_floor", UNKNOWN),
        "intrinsic_base": memo_payload.get("intrinsic_value_discipline", {}).get("intrinsic_base", UNKNOWN)
        if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
        else shortlist_row.get("intrinsic_base", UNKNOWN),
        "intrinsic_ceiling": memo_payload.get("intrinsic_value_discipline", {}).get("intrinsic_ceiling", UNKNOWN)
        if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
        else shortlist_row.get("intrinsic_ceiling", UNKNOWN),
        "mos_to_floor": memo_payload.get("intrinsic_value_discipline", {}).get("mos_to_floor", UNKNOWN)
        if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
        else shortlist_row.get("mos_to_floor", UNKNOWN),
        "mos_to_base": memo_payload.get("intrinsic_value_discipline", {}).get("mos_to_base", UNKNOWN)
        if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
        else shortlist_row.get("mos_to_base", UNKNOWN),
        "mos_classification": str(
            (
                memo_payload.get("intrinsic_value_discipline", {}).get("mos_classification")
                if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                else shortlist_row.get("mos_classification")
            )
            or UNKNOWN
        ),
        "downside_support_type": str(
            (
                memo_payload.get("intrinsic_value_discipline", {}).get("downside_support_type")
                if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                else shortlist_row.get("downside_support_type")
            )
            or UNKNOWN
        ),
        "evidence_sufficiency_class": str(
            (
                memo_payload.get("evidence_sufficiency_for_mos", {}).get("evidence_sufficiency_class")
                if isinstance(memo_payload.get("evidence_sufficiency_for_mos"), dict)
                else shortlist_row.get("evidence_sufficiency_class")
            )
            or UNKNOWN
        ),
        "mos_assessment_status": str(
            (
                memo_payload.get("evidence_sufficiency_for_mos", {}).get("mos_assessment_status")
                if isinstance(memo_payload.get("evidence_sufficiency_for_mos"), dict)
                else shortlist_row.get("mos_assessment_status")
            )
            or UNKNOWN
        ),
        "valuation_range_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("intrinsic_value_discipline", {}).get("valuation_range_reason_codes")
                    if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                    else shortlist_row.get("valuation_range_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "downside_support_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("intrinsic_value_discipline", {}).get("downside_support_reason_codes")
                    if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                    else shortlist_row.get("downside_support_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "valuation_support_count": _int_or_zero(
            memo_payload.get("valuation_confidence_fragility", {}).get("valuation_support_count", UNKNOWN)
            if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
            else shortlist_row.get("valuation_support_count", UNKNOWN)
        ),
        "valuation_support_types_present": [
            str(value)
            for value in (
                (
                    memo_payload.get("valuation_confidence_fragility", {}).get("valuation_support_types_present")
                    if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                    else shortlist_row.get("valuation_support_types_present")
                )
                or []
            )
            if str(value).strip()
        ],
        "valuation_support_count_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("valuation_confidence_fragility", {}).get("valuation_support_count_reason_codes")
                    if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                    else shortlist_row.get("valuation_support_count_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "valuation_convergence_status": str(
            (
                memo_payload.get("valuation_confidence_fragility", {}).get("valuation_convergence_status")
                if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                else shortlist_row.get("valuation_convergence_status")
            )
            or UNKNOWN
        ),
        "valuation_convergence_band_pct": (
            memo_payload.get("valuation_confidence_fragility", {}).get("valuation_convergence_band_pct", UNKNOWN)
            if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
            else shortlist_row.get("valuation_convergence_band_pct", UNKNOWN)
        ),
        "valuation_convergence_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("valuation_confidence_fragility", {}).get("valuation_convergence_reason_codes")
                    if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                    else shortlist_row.get("valuation_convergence_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "valuation_fragility_status": str(
            (
                memo_payload.get("valuation_confidence_fragility", {}).get("valuation_fragility_status")
                if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                else shortlist_row.get("valuation_fragility_status")
            )
            or UNKNOWN
        ),
        "valuation_fragility_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("valuation_confidence_fragility", {}).get("valuation_fragility_reason_codes")
                    if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                    else shortlist_row.get("valuation_fragility_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "valuation_confidence_class": str(
            (
                memo_payload.get("valuation_confidence_fragility", {}).get("valuation_confidence_class")
                if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                else shortlist_row.get("valuation_confidence_class")
            )
            or UNKNOWN
        ),
        "valuation_confidence_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("valuation_confidence_fragility", {}).get("valuation_confidence_reason_codes")
                    if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                    else shortlist_row.get("valuation_confidence_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "valuation_integrity_class": str(
            (
                memo_payload.get("valuation_integrity_audit", {}).get("valuation_integrity_class")
                if isinstance(memo_payload.get("valuation_integrity_audit"), dict)
                else shortlist_row.get("valuation_integrity_class")
            )
            or UNKNOWN
        ),
        "valuation_integrity_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("valuation_integrity_audit", {}).get("valuation_integrity_reason_codes")
                    if isinstance(memo_payload.get("valuation_integrity_audit"), dict)
                    else shortlist_row.get("valuation_integrity_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "evidence_sufficiency_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("evidence_sufficiency_for_mos", {}).get("evidence_sufficiency_reason_codes")
                    if isinstance(memo_payload.get("evidence_sufficiency_for_mos"), dict)
                    else shortlist_row.get("evidence_sufficiency_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "mos_guardrail_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("evidence_sufficiency_for_mos", {}).get("mos_guardrail_reason_codes")
                    if isinstance(memo_payload.get("evidence_sufficiency_for_mos"), dict)
                    else shortlist_row.get("mos_guardrail_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "investment_readiness_class": str(
            (
                memo_payload.get("investment_readiness_blocker_stack", {}).get("investment_readiness_class")
                if isinstance(memo_payload.get("investment_readiness_blocker_stack"), dict)
                else shortlist_row.get("investment_readiness_class")
            )
            or UNKNOWN
        ),
        "investment_readiness_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("investment_readiness_blocker_stack", {}).get("investment_readiness_reason_codes")
                    if isinstance(memo_payload.get("investment_readiness_blocker_stack"), dict)
                    else shortlist_row.get("investment_readiness_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "blocker_stack_primary": str(
            (
                memo_payload.get("investment_readiness_blocker_stack", {}).get("blocker_stack_primary")
                if isinstance(memo_payload.get("investment_readiness_blocker_stack"), dict)
                else shortlist_row.get("blocker_stack_primary")
            )
            or UNKNOWN
        ),
        "primary_next_step": str(
            (
                memo_payload.get("investment_readiness_blocker_stack", {}).get("primary_next_step")
                if isinstance(memo_payload.get("investment_readiness_blocker_stack"), dict)
                else shortlist_row.get("primary_next_step")
            )
            or UNKNOWN
        ),
        "value_type_primary": str(
            (
                memo_payload.get("value_type_classification", {}).get("value_type_primary")
                if isinstance(memo_payload.get("value_type_classification"), dict)
                else shortlist_row.get("value_type_primary")
            )
            or UNKNOWN
        ),
        "value_type_secondary": (
            str(
                (
                    memo_payload.get("value_type_classification", {}).get("value_type_secondary")
                    if isinstance(memo_payload.get("value_type_classification"), dict)
                    else shortlist_row.get("value_type_secondary")
                )
            )
            if str(
                (
                    memo_payload.get("value_type_classification", {}).get("value_type_secondary")
                    if isinstance(memo_payload.get("value_type_classification"), dict)
                    else shortlist_row.get("value_type_secondary")
                )
                or ""
            ).strip()
            else None
        ),
        "value_type_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("value_type_classification", {}).get("value_type_reason_codes")
                    if isinstance(memo_payload.get("value_type_classification"), dict)
                    else shortlist_row.get("value_type_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "value_type_support_summary": str(
            (
                memo_payload.get("value_type_classification", {}).get("value_type_support_summary")
                if isinstance(memo_payload.get("value_type_classification"), dict)
                else shortlist_row.get("value_type_support_summary")
            )
            or ""
        ),
        "facts_blocker_class": str(shortlist_row.get("facts_blocker_class") or "FACTS_OK"),
        "facts_blocker_retryable": bool(shortlist_row.get("facts_blocker_retryable", False)),
        "facts_blocker_terminal": bool(shortlist_row.get("facts_blocker_terminal", False)),
        "facts_blocker_partial_usable": bool(shortlist_row.get("facts_blocker_partial_usable", False)),
        "facts_missing_key_inputs": [str(value) for value in (shortlist_row.get("facts_missing_key_inputs") or []) if str(value).strip()],
        "facts_retry_recommended": bool(shortlist_row.get("facts_retry_recommended", False)),
        "facts_blocker_reason_codes": [str(code) for code in (shortlist_row.get("facts_blocker_reason_codes") or []) if str(code).strip()],
        "facts_recommended_action": str(shortlist_row.get("facts_recommended_action") or "NONE"),
        "fail_due_to_missing_evidence": bool(shortlist_row.get("fail_due_to_missing_evidence", False)),
        "fail_due_to_economic_weakness": bool(shortlist_row.get("fail_due_to_economic_weakness", False)),
        "primary_fail_domain": str(shortlist_row.get("primary_fail_domain") or "NONE"),
        "intangible_economics_reason_codes": [
            str(code)
            for code in (
                (
                    memo_payload.get("modern_intangible_economics", {}).get("intangible_economics_reason_codes")
                    if isinstance(memo_payload.get("modern_intangible_economics"), dict)
                    else shortlist_row.get("intangible_economics_reason_codes")
                )
                or []
            )
            if str(code).strip()
        ],
        "primary_blocker": str(memo_payload.get("primary_blocker") or UNKNOWN),
        "memo_path": str(memo_paths.get("memo_md_path") or memo_paths.get("memo_json_path") or ""),
        "composite_score_total": shortlist_row.get("composite_score_total", UNKNOWN),
        "source_runs": source_runs,
        "derived_from": _dedupe_refs(
            list(memo_payload.get("derived_from_index") or [])
            + list(shortlist_row.get("derived_from") or [])
        ),
    }


def _candidate_from_shortlist(*, item: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "campaign_item": str(item.get("label") or ""),
        "item_index": int(item.get("plan_index") or 0),
        "universe_run_id": str(item.get("child_universe_run_id") or ""),
        "batch_run_id": _child_batch_run_id(str(item.get("child_universe_run_id") or "")),
        "best_rank_seen": int(row.get("rank_global") or 0),
        "value_gate_status": str(row.get("value_gate_status") or UNKNOWN).upper(),
        "implied_return_base": row.get("implied_return_base", UNKNOWN),
        "mos_epv": row.get("mos_epv", UNKNOWN),
        "mos_netnet": row.get("mos_netnet", UNKNOWN),
        "owner_earnings_yield_ev_3y": row.get("owner_earnings_yield_ev_3y", UNKNOWN),
        "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
        "owner_earnings_stability_score": row.get("owner_earnings_stability_score", UNKNOWN),
        "capital_allocation_score": row.get("capital_allocation_score", UNKNOWN),
        "cash_conversion_score": row.get("cash_conversion_score", UNKNOWN),
        "oe_quality_total": row.get("oe_quality_total", UNKNOWN),
        "oe_quality_reason_codes": [
            str(code)
            for code in (row.get("oe_quality_reason_codes") or [])
            if str(code).strip()
        ],
        "gross_margin_durability_score": row.get("gross_margin_durability_score", UNKNOWN),
        "balance_sheet_optionality_score": row.get("balance_sheet_optionality_score", UNKNOWN),
        "cycle_resilience_score": row.get("cycle_resilience_score", UNKNOWN),
        "rnd_productivity_score": row.get("rnd_productivity_score", UNKNOWN),
        "sga_leverage_score": row.get("sga_leverage_score", UNKNOWN),
        "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
        "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
        "reinvestment_efficiency_class": str(
            row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
        ),
        "reinvestment_efficiency_reason_codes": [
            str(code)
            for code in (row.get("reinvestment_efficiency_reason_codes") or [])
            if str(code).strip()
        ],
        "reinvestment_support_signals": [
            str(code)
            for code in (row.get("reinvestment_support_signals") or [])
            if str(code).strip()
        ],
        "reinvestment_headwind_signals": [
            str(code)
            for code in (row.get("reinvestment_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_reinvestment_caution": str(
            row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
        ),
        "reinvestment_efficiency_summary": str(
            row.get("reinvestment_efficiency_summary") or ""
        ),
        "returns_persistence_class": str(
            row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
        ),
        "returns_persistence_reason_codes": [
            str(code)
            for code in (row.get("returns_persistence_reason_codes") or [])
            if str(code).strip()
        ],
        "returns_support_signals": [
            str(code)
            for code in (row.get("returns_support_signals") or [])
            if str(code).strip()
        ],
        "returns_headwind_signals": [
            str(code)
            for code in (row.get("returns_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_returns_caution": str(
            row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
        ),
        "economic_durability_summary": str(
            row.get("economic_durability_summary") or ""
        ),
        "revenue_dependence_risk_class": str(
            row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
        ),
        "revenue_dependence_risk_reason_codes": [
            str(code)
            for code in (row.get("revenue_dependence_risk_reason_codes") or [])
            if str(code).strip()
        ],
        "revenue_dependence_support_signals": [
            str(code)
            for code in (row.get("revenue_dependence_support_signals") or [])
            if str(code).strip()
        ],
        "revenue_dependence_headwind_signals": [
            str(code)
            for code in (row.get("revenue_dependence_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_revenue_dependence_caution": str(
            row.get("primary_revenue_dependence_caution") or "REVENUE_BASE_UNCLEAR"
        ),
        "revenue_fragility_summary": str(
            row.get("revenue_fragility_summary") or ""
        ),
        "asset_intensity_class": str(
            row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
        ),
        "asset_intensity_reason_codes": [
            str(code)
            for code in (row.get("asset_intensity_reason_codes") or [])
            if str(code).strip()
        ],
        "maintenance_capex_credibility_class": str(
            row.get("maintenance_capex_credibility_class") or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
        ),
        "maintenance_capex_credibility_reason_codes": [
            str(code)
            for code in (row.get("maintenance_capex_credibility_reason_codes") or [])
            if str(code).strip()
        ],
        "maintenance_capex_support_signals": [
            str(code)
            for code in (row.get("maintenance_capex_support_signals") or [])
            if str(code).strip()
        ],
        "maintenance_capex_headwind_signals": [
            str(code)
            for code in (row.get("maintenance_capex_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_maintenance_capex_caution": str(
            row.get("primary_maintenance_capex_caution") or "OWNER_EARNINGS_UNCLEAR"
        ),
        "maintenance_capex_discipline_summary": str(
            row.get("maintenance_capex_discipline_summary") or ""
        ),
        "rnd_productivity_reason_codes": [
            str(code)
            for code in (row.get("rnd_productivity_reason_codes") or [])
            if str(code).strip()
        ],
        "sga_leverage_reason_codes": [
            str(code)
            for code in (row.get("sga_leverage_reason_codes") or [])
            if str(code).strip()
        ],
        "owner_value_capture_reason_codes": [
            str(code)
            for code in (row.get("owner_value_capture_reason_codes") or [])
            if str(code).strip()
        ],
        "normalized_earnings_power_value": row.get("normalized_earnings_power_value", UNKNOWN),
        "normalized_earnings_power_method_used": str(row.get("normalized_earnings_power_method_used") or UNKNOWN),
        "normalized_earnings_power_status": str(row.get("normalized_earnings_power_status") or UNKNOWN),
        "normalized_earnings_power_reason_codes": [
            str(code)
            for code in (row.get("normalized_earnings_power_reason_codes") or [])
            if str(code).strip()
        ],
        "intrinsic_floor": row.get("intrinsic_floor", UNKNOWN),
        "intrinsic_base": row.get("intrinsic_base", UNKNOWN),
        "intrinsic_ceiling": row.get("intrinsic_ceiling", UNKNOWN),
        "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
        "mos_to_base": row.get("mos_to_base", UNKNOWN),
        "mos_classification": str(row.get("mos_classification") or UNKNOWN),
        "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
        "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or UNKNOWN),
        "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
        "valuation_range_reason_codes": [
            str(code)
            for code in (row.get("valuation_range_reason_codes") or [])
            if str(code).strip()
        ],
        "downside_support_reason_codes": [
            str(code)
            for code in (row.get("downside_support_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
        "valuation_support_types_present": [
            str(value)
            for value in (row.get("valuation_support_types_present") or [])
            if str(value).strip()
        ],
        "valuation_support_count_reason_codes": [
            str(code)
            for code in (row.get("valuation_support_count_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
        "valuation_convergence_band_pct": row.get("valuation_convergence_band_pct", UNKNOWN),
        "valuation_convergence_reason_codes": [
            str(code)
            for code in (row.get("valuation_convergence_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
        "valuation_fragility_reason_codes": [
            str(code)
            for code in (row.get("valuation_fragility_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
        "valuation_confidence_reason_codes": [
            str(code)
            for code in (row.get("valuation_confidence_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
        "valuation_integrity_reason_codes": [
            str(code)
            for code in (row.get("valuation_integrity_reason_codes") or [])
            if str(code).strip()
        ],
        "evidence_sufficiency_reason_codes": [
            str(code)
            for code in (row.get("evidence_sufficiency_reason_codes") or [])
            if str(code).strip()
        ],
        "mos_guardrail_reason_codes": [
            str(code)
            for code in (row.get("mos_guardrail_reason_codes") or [])
            if str(code).strip()
        ],
        "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
        "investment_readiness_reason_codes": [
            str(code)
            for code in (row.get("investment_readiness_reason_codes") or [])
            if str(code).strip()
        ],
        "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
        "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
        "value_type_primary": str(row.get("value_type_primary") or UNKNOWN),
        "value_type_secondary": (
            str(row.get("value_type_secondary"))
            if str(row.get("value_type_secondary") or "").strip()
            else None
        ),
        "value_type_reason_codes": [
            str(code)
            for code in (row.get("value_type_reason_codes") or [])
            if str(code).strip()
        ],
        "value_type_support_summary": str(row.get("value_type_support_summary") or ""),
        "facts_blocker_class": str(row.get("facts_blocker_class") or "FACTS_OK"),
        "facts_blocker_retryable": bool(row.get("facts_blocker_retryable", False)),
        "facts_blocker_terminal": bool(row.get("facts_blocker_terminal", False)),
        "facts_blocker_partial_usable": bool(row.get("facts_blocker_partial_usable", False)),
        "facts_missing_key_inputs": [str(value) for value in (row.get("facts_missing_key_inputs") or []) if str(value).strip()],
        "facts_retry_recommended": bool(row.get("facts_retry_recommended", False)),
        "facts_blocker_reason_codes": [str(code) for code in (row.get("facts_blocker_reason_codes") or []) if str(code).strip()],
        "facts_recommended_action": str(row.get("facts_recommended_action") or "NONE"),
        "fail_due_to_missing_evidence": bool(row.get("fail_due_to_missing_evidence", False)),
        "fail_due_to_economic_weakness": bool(row.get("fail_due_to_economic_weakness", False)),
        "primary_fail_domain": str(row.get("primary_fail_domain") or "NONE"),
        "intangible_economics_reason_codes": [
            str(code)
            for code in (row.get("intangible_economics_reason_codes") or [])
            if str(code).strip()
        ],
        "primary_blocker": str(row.get("primary_blocker") or UNKNOWN),
        "memo_path": "",
        "composite_score_total": row.get("composite_score_total", UNKNOWN),
        "source_runs": [
            {
                "campaign_item": str(item.get("label") or ""),
                "universe_run_id": str(item.get("child_universe_run_id") or ""),
                "batch_run_id": _child_batch_run_id(str(item.get("child_universe_run_id") or "")),
            }
        ],
        "derived_from": [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()],
    }


def _collect_item_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = item.get("artifact_paths") if isinstance(item.get("artifact_paths"), dict) else {}
    universe_run_id = str(item.get("child_universe_run_id") or "")
    memo_manifest = _safe_json(Path(str(artifacts.get("memo_pack_manifest_path") or "")))
    global_shortlist = _safe_json(Path(str(artifacts.get("global_shortlist_path") or "")))
    scout_lookup = _load_child_scout_facts_lookup(universe_run_id) if universe_run_id else {}
    shortlist_lookup = {
        str(row.get("ticker") or "").upper(): row
        for row in (global_shortlist.get("rows") or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }

    rows: list[dict[str, Any]] = []
    memos = [row for row in (memo_manifest.get("memos") or []) if isinstance(row, dict)]
    for memo_row in memos:
        memo_path = Path(str(memo_row.get("memo_json_path") or ""))
        memo_payload = _safe_json(memo_path)
        if not memo_payload:
            continue
        rows.append(
            _overlay_scout_facts_fields(
                _candidate_from_memo(
                    item=item,
                    memo_payload=memo_payload,
                    memo_paths={
                        "memo_json_path": str(memo_row.get("memo_json_path") or ""),
                        "memo_md_path": str(memo_row.get("memo_md_path") or ""),
                    },
                    shortlist_lookup=shortlist_lookup,
                ),
                scout_lookup,
            )
        )
    if rows:
        return rows
    for row in [row for row in (global_shortlist.get("rows") or []) if isinstance(row, dict)]:
        rows.append(_overlay_scout_facts_fields(_candidate_from_shortlist(item=item, row=row), scout_lookup))
    return rows


def _write_master_shortlist(paths: dict[str, Path], state: dict[str, Any]) -> dict[str, Any]:
    items = [row for row in (state.get("items") or []) if isinstance(row, dict)]
    policy = _effective_campaign_policy(state)
    memory_lookup = _memory_priority_lookup()
    item_rows: list[dict[str, Any]] = []
    for item in items:
        if str(item.get("status") or "").upper() not in {ITEM_DONE, ITEM_PARTIAL}:
            continue
        item_rows.extend(_collect_item_rows(item))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in item_rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        grouped.setdefault(ticker, []).append(row)

    master_rows: list[dict[str, Any]] = []
    for ticker in sorted(grouped.keys()):
        rows = grouped[ticker]
        representative_candidates = [
            {
                **row,
                **_memory_priority_fields(ticker, memory_lookup),
            }
            for row in rows
        ]
        representative = sorted(representative_candidates, key=lambda row: _master_sort_key(row, policy=policy))[0]
        latest = sorted(rows, key=lambda item: (int(item.get("item_index") or 0), int(item.get("best_rank_seen") or 10**9), str(item.get("campaign_item") or "")))[-1]
        source_runs: list[dict[str, Any]] = []
        seen_source: set[tuple[str, str, str]] = set()
        for row in sorted(rows, key=lambda item: (int(item.get("item_index") or 0), str(item.get("campaign_item") or ""), str(item.get("universe_run_id") or ""))):
            for source in [src for src in (row.get("source_runs") or []) if isinstance(src, dict)]:
                key = (
                    str(source.get("campaign_item") or ""),
                    str(source.get("universe_run_id") or ""),
                    str(source.get("batch_run_id") or ""),
                )
                if key in seen_source:
                    continue
                seen_source.add(key)
                source_runs.append(
                    {
                        "campaign_item": key[0],
                        "universe_run_id": key[1],
                        "batch_run_id": key[2],
                    }
                )
        merged = dict(representative)
        merged["ticker"] = ticker
        merged["source_runs"] = source_runs
        merged["best_rank_seen"] = min(int(row.get("best_rank_seen") or 10**9) for row in rows)
        merged["latest_value_gate_status"] = str(latest.get("value_gate_status") or UNKNOWN)
        merged["latest_primary_blocker"] = str(latest.get("primary_blocker") or UNKNOWN)
        merged["derived_from"] = _dedupe_refs([ref for row in rows for ref in list(row.get("derived_from") or [])])
        merged.update(_memory_priority_fields(ticker, memory_lookup))
        master_rows.append(merged)

    ranked = sorted(master_rows, key=lambda row: _master_sort_key(row, policy=policy))
    for idx, row in enumerate(ranked, start=1):
        row["rank_master"] = int(idx)

    payload = {
        "campaign_run_id": str(state.get("campaign_run_id") or ""),
        "generated_at": utc_now_iso(),
        "policy_effective": policy,
        "candidate_count_total": len(item_rows),
        "candidate_count_ranked": len(ranked),
        "rows": ranked,
    }
    _json_write(paths["master_shortlist_json_path"], payload)
    paths["master_shortlist_md_path"].write_text(
        _campaign_markdown(campaign_run_id=str(state.get("campaign_run_id") or ""), rows=ranked[:50]),
        encoding="utf-8",
    )
    return payload


def _write_master_watchlist_state(paths: dict[str, Path], state: dict[str, Any]) -> dict[str, Any]:
    items = [row for row in (state.get("items") or []) if isinstance(row, dict)]
    merged: dict[str, Any] = {}
    for item in sorted(items, key=lambda row: (int(row.get("plan_index") or 0), str(row.get("label") or ""))):
        if str(item.get("status") or "").upper() not in {ITEM_DONE, ITEM_PARTIAL}:
            continue
        watchlist_path = Path(str((item.get("artifact_paths") or {}).get("watchlist_state_path") or ""))
        child = _safe_json(watchlist_path)
        tickers = child.get("tickers") if isinstance(child.get("tickers"), dict) else {}
        for ticker in sorted(tickers.keys()):
            child_row = tickers.get(ticker) if isinstance(tickers.get(ticker), dict) else {}
            if ticker not in merged:
                merged[ticker] = {
                    "first_seen_campaign_run_id": str(state.get("campaign_run_id") or ""),
                    "last_seen_campaign_run_id": str(state.get("campaign_run_id") or ""),
                    "first_seen_universe_run_id": str(item.get("child_universe_run_id") or ""),
                    "last_seen_universe_run_id": str(item.get("child_universe_run_id") or ""),
                    "latest_value_gate_status": str(child_row.get("last_value_gate_status") or UNKNOWN),
                    "latest_implied_return_base": child_row.get("last_implied_return_base", UNKNOWN),
                    "latest_primary_blocker": str(child_row.get("last_primary_blocker_code") or UNKNOWN),
                    "appearances_count": 1,
                    "history": [],
                }
            else:
                merged[ticker]["last_seen_campaign_run_id"] = str(state.get("campaign_run_id") or "")
                merged[ticker]["last_seen_universe_run_id"] = str(item.get("child_universe_run_id") or "")
                merged[ticker]["latest_value_gate_status"] = str(child_row.get("last_value_gate_status") or UNKNOWN)
                merged[ticker]["latest_implied_return_base"] = child_row.get("last_implied_return_base", UNKNOWN)
                merged[ticker]["latest_primary_blocker"] = str(child_row.get("last_primary_blocker_code") or UNKNOWN)
                merged[ticker]["appearances_count"] = int(merged[ticker].get("appearances_count") or 0) + 1

            history = [row for row in (merged[ticker].get("history") or []) if isinstance(row, dict)]
            history.append(
                {
                    "campaign_item": str(item.get("label") or ""),
                    "universe_run_id": str(item.get("child_universe_run_id") or ""),
                    "value_gate_status": str(child_row.get("last_value_gate_status") or UNKNOWN),
                    "implied_return_base": child_row.get("last_implied_return_base", UNKNOWN),
                    "primary_blocker": str(child_row.get("last_primary_blocker_code") or UNKNOWN),
                    "last_rank": int(child_row.get("last_rank") or 0),
                }
            )
            merged[ticker]["history"] = history[-10:]

    gate_counts: dict[str, int] = {}
    for row in merged.values():
        gate = str(row.get("latest_value_gate_status") or UNKNOWN)
        gate_counts[gate] = gate_counts.get(gate, 0) + 1

    payload = {
        "campaign_run_id": str(state.get("campaign_run_id") or ""),
        "generated_at": utc_now_iso(),
        "ticker_count": len(merged),
        "gate_counts": dict(sorted(gate_counts.items(), key=lambda kv: str(kv[0]))),
        "tickers": dict(sorted(merged.items(), key=lambda kv: kv[0])),
    }
    _json_write(paths["master_watchlist_state_path"], payload)
    return payload


def _build_summary(paths: dict[str, Path], state: dict[str, Any]) -> dict[str, Any]:
    items = [row for row in (state.get("items") or []) if isinstance(row, dict)]
    counts = {
        "planned": len(items),
        "done": len([row for row in items if str(row.get("status") or "").upper() == ITEM_DONE]),
        "partial": len([row for row in items if str(row.get("status") or "").upper() == ITEM_PARTIAL]),
        "failed": len([row for row in items if str(row.get("status") or "").upper() == ITEM_FAILED]),
        "cancelled": len([row for row in items if str(row.get("status") or "").upper() == ITEM_CANCELLED]),
        "running": len([row for row in items if str(row.get("status") or "").upper() == ITEM_RUNNING]),
        "pending": len([row for row in items if str(row.get("status") or "").upper() == ITEM_PENDING]),
    }
    master_shortlist = _safe_json(paths["master_shortlist_json_path"])
    master_watchlist = _safe_json(paths["master_watchlist_state_path"])
    promotion_state = _safe_json(paths["promotion_state_path"])
    promotion_candidates = _safe_json(paths["promotion_candidates_path"])
    escalation_summary = _safe_json(paths["escalation_summary_path"])
    research_memory_dir = get_config().outputs_dir / "research_memory"
    research_memory_path = research_memory_dir / "research_memory.json"
    research_memory_summary_path = research_memory_dir / "research_memory_summary.json"
    research_memory_summary = _safe_json(research_memory_summary_path)
    top_10 = [
        {
            "ticker": str(row.get("ticker") or ""),
            "value_gate_status": str(row.get("value_gate_status") or UNKNOWN),
            "implied_return_base": row.get("implied_return_base", UNKNOWN),
            "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
            "mos_classification": str(row.get("mos_classification") or UNKNOWN),
            "mos_epv": row.get("mos_epv", UNKNOWN),
            "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
            "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
            "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
            "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
            "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
            "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or UNKNOWN),
            "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
            "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
            "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
            "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
            "value_type_primary": str(row.get("value_type_primary") or UNKNOWN),
            "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
            "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
            "oe_quality_total": row.get("oe_quality_total", UNKNOWN),
            "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
            "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
            "returns_persistence_class": str(
                row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
            ),
            "primary_returns_caution": str(
                row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
            ),
            "reinvestment_efficiency_class": str(
                row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
            ),
            "primary_reinvestment_caution": str(
                row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
            ),
            "asset_intensity_class": str(
                row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
            ),
            "maintenance_capex_credibility_class": str(
                row.get("maintenance_capex_credibility_class") or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
            ),
            "primary_maintenance_capex_caution": str(
                row.get("primary_maintenance_capex_caution") or "OWNER_EARNINGS_UNCLEAR"
            ),
            "asset_quality_class": str(
                row.get("asset_quality_class") or "ASSET_QUALITY_UNKNOWN"
            ),
            "downside_realization_credibility_class": str(
                row.get("downside_realization_credibility_class")
                or "DOWNSIDE_REALIZATION_UNKNOWN"
            ),
            "primary_asset_support_caution": str(
                row.get("primary_asset_support_caution") or "ASSET_SUPPORT_UNCLEAR"
            ),
            "obligation_burden_class": str(
                row.get("obligation_burden_class") or "OBLIGATION_BURDEN_UNKNOWN"
            ),
            "claim_priority_pressure_class": str(
                row.get("claim_priority_pressure_class") or "CLAIM_PRIORITY_PRESSURE_UNKNOWN"
            ),
            "primary_obligation_caution": str(
                row.get("primary_obligation_caution") or "RESIDUAL_EQUITY_UNCLEAR"
            ),
            "balance_sheet_stress_class": str(
                row.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN"
            ),
            "refinancing_risk_class": str(
                row.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
            ),
            "primary_balance_sheet_caution": str(
                row.get("primary_balance_sheet_caution") or "BALANCE_SHEET_UNCLEAR"
            ),
            "accounting_quality_class": str(
                row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
            ),
            "primary_accounting_caution": str(
                row.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
            ),
            "memory_priority_total": int(row.get("memory_priority_total") or 0),
        }
        for row in [row for row in (master_shortlist.get("rows") or []) if isinstance(row, dict)][:10]
    ]
    shortlist_top_memory_candidates = sorted(
        [row for row in (master_shortlist.get("rows") or []) if isinstance(row, dict)],
        key=lambda row: (
            -int(row.get("memory_priority_total") or 0),
            _status_rank(str(row.get("value_gate_status") or UNKNOWN)),
            str(row.get("ticker") or ""),
        ),
    )[:10]
    shortlist_tickers = {
        str(row.get("ticker") or "").upper()
        for row in (master_shortlist.get("rows") or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    research_top_memory_candidates = [
        row
        for row in (research_memory_summary.get("top_memory_priority_candidates") or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").upper() in shortlist_tickers
    ]
    top_memory_priority_candidates = research_top_memory_candidates[:10] if research_top_memory_candidates else shortlist_top_memory_candidates
    top_promotion = [
        {
            "ticker": str(row.get("ticker") or ""),
            "priority_lane": str(row.get("priority_lane") or UNKNOWN),
            "latest_value_gate_status": str(row.get("latest_value_gate_status") or UNKNOWN),
            "implied_return_base": row.get("implied_return_base", UNKNOWN),
            "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
            "mos_classification": str(row.get("mos_classification") or UNKNOWN),
            "mos_epv": row.get("mos_epv", UNKNOWN),
            "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
            "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
            "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
            "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
            "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
            "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or UNKNOWN),
            "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
            "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
            "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
            "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
            "value_type_primary": str(row.get("value_type_primary") or UNKNOWN),
            "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
            "oe_quality_total": row.get("oe_quality_total", UNKNOWN),
            "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
            "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
            "returns_persistence_class": str(
                row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
            ),
            "primary_returns_caution": str(
                row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
            ),
            "reinvestment_efficiency_class": str(
                row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
            ),
            "primary_reinvestment_caution": str(
                row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
            ),
            "asset_intensity_class": str(
                row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
            ),
            "maintenance_capex_credibility_class": str(
                row.get("maintenance_capex_credibility_class")
                or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
            ),
            "primary_maintenance_capex_caution": str(
                row.get("primary_maintenance_capex_caution") or "OWNER_EARNINGS_UNCLEAR"
            ),
            "asset_quality_class": str(
                row.get("asset_quality_class") or "ASSET_QUALITY_UNKNOWN"
            ),
            "downside_realization_credibility_class": str(
                row.get("downside_realization_credibility_class")
                or "DOWNSIDE_REALIZATION_UNKNOWN"
            ),
            "primary_asset_support_caution": str(
                row.get("primary_asset_support_caution") or "ASSET_SUPPORT_UNCLEAR"
            ),
            "obligation_burden_class": str(
                row.get("obligation_burden_class") or "OBLIGATION_BURDEN_UNKNOWN"
            ),
            "claim_priority_pressure_class": str(
                row.get("claim_priority_pressure_class") or "CLAIM_PRIORITY_PRESSURE_UNKNOWN"
            ),
            "primary_obligation_caution": str(
                row.get("primary_obligation_caution") or "RESIDUAL_EQUITY_UNCLEAR"
            ),
            "accounting_quality_class": str(
                row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
            ),
            "primary_accounting_caution": str(
                row.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
            ),
            "appearances_count": int(row.get("appearances_count") or 0),
            "memory_priority_total": int(row.get("memory_priority_total") or 0),
        }
        for row in [row for row in (promotion_candidates.get("rows") or []) if isinstance(row, dict)][:10]
    ]
    item_blockers: list[dict[str, Any]] = []
    retryable_facts_blocker_count = 0
    terminal_facts_blocker_count = 0
    partial_usable_facts_count = 0
    top_retryable_facts_blockers: list[dict[str, Any]] = []
    child_runs_with_facts_degradation: list[dict[str, Any]] = []
    child_validation_rows: list[dict[str, Any]] = []
    child_filter_rows: list[dict[str, Any]] = []
    child_borderline_rows: list[dict[str, Any]] = []
    child_second_look_rows: list[dict[str, Any]] = []
    child_carry_forward_rows: list[dict[str, Any]] = []
    child_review_plan_rows: list[dict[str, Any]] = []
    child_review_intake_rows: list[dict[str, Any]] = []
    child_review_memory_rows: list[dict[str, Any]] = []
    child_review_cycle_rows: list[dict[str, Any]] = []
    child_review_compare_rows: list[dict[str, Any]] = []
    child_calibration_rows: list[dict[str, Any]] = []
    evidence_fail_count = 0
    economic_fail_count = 0
    mixed_fail_count = 0
    other_fail_count = 0
    for item in items:
        item_status = str(item.get("status") or ITEM_PENDING).upper()
        if item_status == ITEM_DONE:
            continue
        artifact_paths = item.get("artifact_paths") if isinstance(item.get("artifact_paths"), dict) else {}
        autopilot_summary = _safe_json(Path(str(artifact_paths.get("autopilot_summary_path") or "")))
        item_blockers.append(
            {
                "item_id": str(item.get("item_id") or ""),
                "label": str(item.get("label") or ""),
                "status": item_status,
                "child_universe_run_id": str(item.get("child_universe_run_id") or ""),
                "stop_reason_code": str(
                    autopilot_summary.get("stop_reason_code")
                    or item.get("error")
                    or UNKNOWN
                ),
                "stop_summary": str(
                    autopilot_summary.get("stop_summary")
                    or item.get("error")
                    or ""
                ),
                "primary_scout_blocker": str(autopilot_summary.get("primary_scout_blocker") or ""),
                "scout_hydration_status": str(autopilot_summary.get("scout_hydration_status") or UNKNOWN),
                "scout_last_progress_phase": str(autopilot_summary.get("scout_last_progress_phase") or UNKNOWN),
                "scout_stalled_reason_code": str(autopilot_summary.get("scout_stalled_reason_code") or ""),
                "retryable_facts_blocker_count": int(autopilot_summary.get("retryable_facts_blocker_count") or 0),
                "terminal_facts_blocker_count": int(autopilot_summary.get("terminal_facts_blocker_count") or 0),
                "partial_usable_facts_count": int(autopilot_summary.get("partial_usable_facts_count") or 0),
            }
        )
    for item in items:
        artifact_paths = item.get("artifact_paths") if isinstance(item.get("artifact_paths"), dict) else {}
        autopilot_summary = _safe_json(Path(str(artifact_paths.get("autopilot_summary_path") or "")))
        child_universe_run_id = str(item.get("child_universe_run_id") or "")
        validation_summary = _safe_json(_child_validation_summary_path(child_universe_run_id))
        filter_summary = _safe_json(_child_filter_audit_summary_path(child_universe_run_id))
        borderline_summary = _safe_json(_child_borderline_audit_summary_path(child_universe_run_id))
        second_look_summary = _safe_json(_child_second_look_summary_path(child_universe_run_id))
        carry_forward_summary = _safe_json(_child_carry_forward_summary_path(child_universe_run_id))
        review_plan_summary = _safe_json(_child_review_plan_summary_path(child_universe_run_id))
        review_intake_summary = _safe_json(_child_review_intake_summary_path(child_universe_run_id))
        review_outcomes_summary = _safe_json(_child_review_outcomes_summary_path(child_universe_run_id))
        calibration_summary = {}
        if validation_summary:
            calibration_run_id = str(validation_summary.get("latest_validation_run_id") or "")
            if calibration_run_id:
                calibration_summary = _safe_json(_child_filter_calibration_summary_path(calibration_run_id))
        retryable_facts_blocker_count += int(autopilot_summary.get("retryable_facts_blocker_count") or 0)
        terminal_facts_blocker_count += int(autopilot_summary.get("terminal_facts_blocker_count") or 0)
        partial_usable_facts_count += int(autopilot_summary.get("partial_usable_facts_count") or 0)
        fail_counts = autopilot_summary.get("economic_fail_count_vs_evidence_fail_count")
        if isinstance(fail_counts, dict):
            evidence_fail_count += int(fail_counts.get("evidence_fail_count") or 0)
            economic_fail_count += int(fail_counts.get("economic_fail_count") or 0)
            mixed_fail_count += int(fail_counts.get("mixed_fail_count") or 0)
            other_fail_count += int(fail_counts.get("other_fail_count") or 0)
        facts_rows = [row for row in (autopilot_summary.get("top_retryable_facts_blockers") or []) if isinstance(row, dict)]
        for row in facts_rows:
            top_retryable_facts_blockers.append(
                {
                    "child_universe_run_id": str(item.get("child_universe_run_id") or ""),
                    "ticker": str(row.get("ticker") or ""),
                    "facts_blocker_class": str(row.get("facts_blocker_class") or UNKNOWN),
                    "facts_recommended_action": str(row.get("facts_recommended_action") or "NONE"),
                    "primary_fail_domain": str(row.get("primary_fail_domain") or "NONE"),
                }
            )
        if (
            int(autopilot_summary.get("retryable_facts_blocker_count") or 0) > 0
            or int(autopilot_summary.get("terminal_facts_blocker_count") or 0) > 0
            or int(autopilot_summary.get("partial_usable_facts_count") or 0) > 0
            or str(autopilot_summary.get("scout_hydration_status") or "").upper() == "DEGRADED"
        ):
            child_runs_with_facts_degradation.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": str(item.get("child_universe_run_id") or ""),
                    "retryable_facts_blocker_count": int(autopilot_summary.get("retryable_facts_blocker_count") or 0),
                    "terminal_facts_blocker_count": int(autopilot_summary.get("terminal_facts_blocker_count") or 0),
                    "partial_usable_facts_count": int(autopilot_summary.get("partial_usable_facts_count") or 0),
                    "scout_hydration_status": str(autopilot_summary.get("scout_hydration_status") or UNKNOWN),
                }
            )
        if validation_summary:
            child_validation_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_validation_run_id": str(validation_summary.get("latest_validation_run_id") or ""),
                    "benchmark_count": int(validation_summary.get("benchmark_count") or 0),
                    "benchmark_pass_rate": float(validation_summary.get("benchmark_pass_rate") or 0.0),
                    "benchmark_mixed_rate": float(validation_summary.get("benchmark_mixed_rate") or 0.0),
                    "top_validation_failure_modes": [
                        row
                        for row in (validation_summary.get("top_validation_failure_modes") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "updated_at": str(validation_summary.get("updated_at") or ""),
                }
            )
        if filter_summary:
            child_filter_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_filter_audit_run_id": str(filter_summary.get("run_id") or ""),
                    "total_ticker_count": int(filter_summary.get("total_ticker_count") or 0),
                    "evidence_stop_count": int(filter_summary.get("evidence_stop_count") or 0),
                    "economics_stop_count": int(filter_summary.get("economics_stop_count") or 0),
                    "queue_budget_stop_count": int(filter_summary.get("queue_budget_stop_count") or 0),
                    "memo_survivor_count": int(filter_summary.get("memo_survivor_count") or 0),
                    "top_terminal_reasons": [
                        row for row in (filter_summary.get("top_terminal_reasons") or []) if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(filter_summary.get("generated_at") or ""),
                }
            )
        if borderline_summary:
            child_borderline_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_borderline_audit_run_id": str(borderline_summary.get("run_id") or ""),
                    "total_stopped_names": int(borderline_summary.get("total_stopped_names") or 0),
                    "borderline_case_rate": float(borderline_summary.get("borderline_case_rate") or 0.0),
                    "rescue_queue_count": int(borderline_summary.get("rescue_queue_count") or 0),
                    "top_rescue_queue_names": [
                        row for row in (borderline_summary.get("top_rescue_queue_names") or []) if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(borderline_summary.get("generated_at") or ""),
                }
            )
        if second_look_summary:
            child_second_look_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_second_look_run_id": str(second_look_summary.get("run_id") or ""),
                    "total_reviewed": int(second_look_summary.get("total_reviewed") or 0),
                    "possible_false_negative_count": int(second_look_summary.get("possible_false_negative_count") or 0),
                    "evidence_review_recommended_count": int(
                        second_look_summary.get("evidence_review_recommended_count") or 0
                    ),
                    "queue_review_recommended_count": int(
                        second_look_summary.get("queue_review_recommended_count") or 0
                    ),
                    "top_names_for_reconsideration": [
                        row
                        for row in (second_look_summary.get("top_names_for_reconsideration") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(second_look_summary.get("generated_at") or ""),
                }
            )
        if carry_forward_summary:
            child_carry_forward_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_carry_forward_run_id": str(carry_forward_summary.get("run_id") or ""),
                    "total_carry_forward_eligible": int(carry_forward_summary.get("total_carry_forward_eligible") or 0),
                    "high_priority_count": int(
                        (carry_forward_summary.get("counts_by_carry_forward_priority") or {}).get(
                            "HIGH_RECURSIVE_REVIEW", 0
                        )
                    ),
                    "top_carry_forward_names": [
                        row
                        for row in (carry_forward_summary.get("top_carry_forward_names") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(carry_forward_summary.get("generated_at") or ""),
                }
            )
        if review_plan_summary:
            child_review_plan_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_review_plan_run_id": str(review_plan_summary.get("run_id") or ""),
                    "total_review_plan_eligible": int(review_plan_summary.get("total_review_plan_eligible") or 0),
                    "high_priority_count": int(
                        (review_plan_summary.get("counts_by_review_plan_priority") or {}).get(
                            "HIGH_NEXT_CYCLE_REVIEW", 0
                        )
                    ),
                    "bucket_counts": review_plan_summary.get("counts_by_next_cycle_review_bucket") or {},
                    "top_review_plan_names": [
                        row
                        for row in (review_plan_summary.get("top_review_plan_names") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(review_plan_summary.get("generated_at") or ""),
                }
            )
        review_memory_ref = _safe_json(_child_review_memory_ref_path(child_universe_run_id))
        review_memory_summary: dict[str, Any] = {}
        if review_memory_ref:
            mem_summary_path_str = str(review_memory_ref.get("review_memory_summary_path") or "")
            if mem_summary_path_str:
                from pathlib import Path as _Path
                review_memory_summary = _safe_json(_Path(mem_summary_path_str))
        if review_intake_summary:
            outcome_counts = {}
            if review_outcomes_summary:
                outcome_counts = review_outcomes_summary.get("counts_by_review_outcome_class") or {}
            child_review_intake_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_review_intake_run_id": str(review_intake_summary.get("run_id") or ""),
                    "total_review_intake_eligible": int(review_intake_summary.get("total_review_intake_eligible") or 0),
                    "high_priority_count": int(
                        (review_intake_summary.get("counts_by_review_intake_priority") or {}).get(
                            "HIGH_REVIEW_INTAKE", 0
                        )
                    ),
                    "review_outcome_counts": outcome_counts,
                    "top_review_reentry_names": [
                        row
                        for row in (review_intake_summary.get("top_review_reentry_names") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(review_intake_summary.get("generated_at") or ""),
                }
            )
        if review_memory_summary:
            child_review_memory_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_review_memory_run_id": str(review_memory_ref.get("memory_run_id") or ""),
                    "total_tracked_tickers": int(review_memory_summary.get("total_tracked_tickers") or 0),
                    "productivity_counts": review_memory_summary.get("counts_by_review_productivity_class") or {},
                    "delta_counts": review_memory_summary.get("counts_by_review_delta_class") or {},
                    "top_high_productivity_names": [
                        row
                        for row in (review_memory_summary.get("top_high_productivity_names") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "top_churn_risk_names": [
                        row
                        for row in (review_memory_summary.get("top_churn_risk_names") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(review_memory_summary.get("generated_at") or ""),
                }
            )
        review_cycle_ref = _safe_json(_child_review_cycle_ref_path(child_universe_run_id))
        review_cycle_summary: dict[str, Any] = {}
        if review_cycle_ref:
            cycle_summary_path_str = str(review_cycle_ref.get("cycle_summary_path") or "")
            if cycle_summary_path_str:
                from pathlib import Path as _Path
                review_cycle_summary = _safe_json(_Path(cycle_summary_path_str))
        if review_cycle_summary:
            child_review_cycle_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_review_cycle_run_id": str(review_cycle_ref.get("cycle_id") or ""),
                    "prior_run_id": str(review_cycle_summary.get("prior_run_id") or ""),
                    "total_selected": int(review_cycle_summary.get("total_selected") or 0),
                    "total_seed_candidates": int(review_cycle_summary.get("total_seed_candidates") or 0),
                    "generated_at": str(review_cycle_summary.get("generated_at") or ""),
                }
            )
        review_compare_ref = _safe_json(_child_review_compare_ref_path(child_universe_run_id))
        review_compare_summary: dict[str, Any] = {}
        if review_compare_ref:
            compare_summary_path_str = str(review_compare_ref.get("compare_summary_path") or "")
            if compare_summary_path_str:
                from pathlib import Path as _Path
                review_compare_summary = _safe_json(_Path(compare_summary_path_str))
        if review_compare_summary:
            child_review_compare_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_review_compare_run_id": str(review_compare_ref.get("cycle_id") or ""),
                    "total_compared": int(review_compare_summary.get("total_compared") or 0),
                    "delta_counts": review_compare_summary.get("counts_by_compare_delta_class") or {},
                    "top_gain_names": [
                        row
                        for row in (review_compare_summary.get("top_gain_names") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "top_negative_confirmation_names": [
                        row
                        for row in (review_compare_summary.get("top_negative_confirmation_names") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(review_compare_summary.get("generated_at") or ""),
                }
            )
        if calibration_summary:
            child_calibration_rows.append(
                {
                    "item_id": str(item.get("item_id") or ""),
                    "label": str(item.get("label") or ""),
                    "child_universe_run_id": child_universe_run_id,
                    "latest_filter_calibration_run_id": str(validation_summary.get("latest_validation_run_id") or ""),
                    "high_severity_count": int(
                        ((calibration_summary.get("recommendation_counts_by_severity") or {}) if isinstance(calibration_summary.get("recommendation_counts_by_severity"), dict) else {}).get("HIGH") or 0
                    ),
                    "top_recommendation_codes": [
                        row
                        for row in (calibration_summary.get("top_recommendation_codes") or [])
                        if isinstance(row, dict)
                    ][:5],
                    "generated_at": str(calibration_summary.get("generated_at") or ""),
                }
            )

    validation_failure_counter: dict[str, int] = {}
    validation_case_count = 0
    validation_pass_weight = 0.0
    validation_mixed_weight = 0.0
    latest_validation_run_id = ""
    latest_validation_updated_at = ""
    for row in child_validation_rows:
        benchmark_count = int(row.get("benchmark_count") or 0)
        validation_case_count += benchmark_count
        validation_pass_weight += benchmark_count * float(row.get("benchmark_pass_rate") or 0.0)
        validation_mixed_weight += benchmark_count * float(row.get("benchmark_mixed_rate") or 0.0)
        updated_at = str(row.get("updated_at") or "")
        if updated_at >= latest_validation_updated_at:
            latest_validation_updated_at = updated_at
            latest_validation_run_id = str(row.get("latest_validation_run_id") or "")
        for failure_row in row.get("top_validation_failure_modes") or []:
            reason_code = str(failure_row.get("reason_code") or "").strip()
            if not reason_code:
                continue
            validation_failure_counter[reason_code] = validation_failure_counter.get(reason_code, 0) + int(
                failure_row.get("count") or 0
            )
    top_validation_failure_modes = [
        {"reason_code": reason_code, "count": count}
        for reason_code, count in sorted(
            validation_failure_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    filter_terminal_reason_counter: dict[str, int] = {}
    filter_total_ticker_count = 0
    filter_evidence_stop_count = 0
    filter_economics_stop_count = 0
    filter_queue_budget_stop_count = 0
    filter_memo_survivor_count = 0
    latest_filter_audit_run_id = ""
    latest_filter_generated_at = ""
    for row in child_filter_rows:
        filter_total_ticker_count += int(row.get("total_ticker_count") or 0)
        filter_evidence_stop_count += int(row.get("evidence_stop_count") or 0)
        filter_economics_stop_count += int(row.get("economics_stop_count") or 0)
        filter_queue_budget_stop_count += int(row.get("queue_budget_stop_count") or 0)
        filter_memo_survivor_count += int(row.get("memo_survivor_count") or 0)
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_filter_generated_at:
            latest_filter_generated_at = generated_at
            latest_filter_audit_run_id = str(row.get("latest_filter_audit_run_id") or "")
        for reason_row in row.get("top_terminal_reasons") or []:
            reason_code = str(reason_row.get("reason_code") or "").strip()
            if not reason_code:
                continue
            filter_terminal_reason_counter[reason_code] = filter_terminal_reason_counter.get(reason_code, 0) + int(
                reason_row.get("count") or 0
            )
    top_filter_failure_modes = [
        {"reason_code": reason_code, "count": count}
        for reason_code, count in sorted(
            filter_terminal_reason_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    rescue_queue_total = 0
    borderline_total_stopped = 0
    borderline_weight = 0.0
    latest_borderline_audit_run_id = ""
    latest_borderline_generated_at = ""
    rescue_name_counter: dict[str, int] = {}
    for row in child_borderline_rows:
        total_stopped_names = int(row.get("total_stopped_names") or 0)
        rescue_queue_total += int(row.get("rescue_queue_count") or 0)
        borderline_total_stopped += total_stopped_names
        borderline_weight += total_stopped_names * float(row.get("borderline_case_rate") or 0.0)
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_borderline_generated_at:
            latest_borderline_generated_at = generated_at
            latest_borderline_audit_run_id = str(row.get("latest_borderline_audit_run_id") or "")
        for rescue_row in row.get("top_rescue_queue_names") or []:
            ticker = str(rescue_row.get("ticker") or "").strip().upper()
            if ticker:
                rescue_name_counter[ticker] = rescue_name_counter.get(ticker, 0) + 1
    top_rescue_queue_names = [
        ticker
        for ticker, _count in sorted(
            rescue_name_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    latest_second_look_run_id = ""
    latest_second_look_generated_at = ""
    second_look_review_count = 0
    second_look_possible_false_negative_count = 0
    second_look_evidence_review_count = 0
    second_look_queue_review_count = 0
    second_look_name_counter: dict[str, int] = {}
    for row in child_second_look_rows:
        second_look_review_count += int(row.get("total_reviewed") or 0)
        second_look_possible_false_negative_count += int(row.get("possible_false_negative_count") or 0)
        second_look_evidence_review_count += int(row.get("evidence_review_recommended_count") or 0)
        second_look_queue_review_count += int(row.get("queue_review_recommended_count") or 0)
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_second_look_generated_at:
            latest_second_look_generated_at = generated_at
            latest_second_look_run_id = str(row.get("latest_second_look_run_id") or "")
        for reconsider_row in row.get("top_names_for_reconsideration") or []:
            ticker = str(reconsider_row.get("ticker") or "").strip().upper()
            if ticker:
                second_look_name_counter[ticker] = second_look_name_counter.get(ticker, 0) + 1
    top_second_look_reconsideration_names = [
        ticker
        for ticker, _count in sorted(
            second_look_name_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    latest_filter_calibration_run_id = ""
    latest_calibration_generated_at = ""
    calibration_high_severity_count = 0
    calibration_recommendation_counter: dict[str, int] = {}
    for row in child_calibration_rows:
        calibration_high_severity_count += int(row.get("high_severity_count") or 0)
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_calibration_generated_at:
            latest_calibration_generated_at = generated_at
            latest_filter_calibration_run_id = str(row.get("latest_filter_calibration_run_id") or "")
        for rec_row in row.get("top_recommendation_codes") or []:
            recommendation_code = str(rec_row.get("recommendation_code") or "").strip()
            if not recommendation_code:
                continue
            calibration_recommendation_counter[recommendation_code] = calibration_recommendation_counter.get(recommendation_code, 0) + int(
                rec_row.get("count") or 0
            )
    top_filter_calibration_recommendations = [
        {"recommendation_code": recommendation_code, "count": count}
        for recommendation_code, count in sorted(
            calibration_recommendation_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    latest_carry_forward_run_id = ""
    latest_carry_forward_generated_at = ""
    carry_forward_count = 0
    carry_forward_high_priority_count = 0
    carry_forward_name_counter: dict[str, int] = {}
    for row in child_carry_forward_rows:
        carry_forward_count += int(row.get("total_carry_forward_eligible") or 0)
        carry_forward_high_priority_count += int(row.get("high_priority_count") or 0)
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_carry_forward_generated_at:
            latest_carry_forward_generated_at = generated_at
            latest_carry_forward_run_id = str(row.get("latest_carry_forward_run_id") or "")
        for cf_row in row.get("top_carry_forward_names") or []:
            ticker = str(cf_row.get("ticker") or "").strip().upper()
            if ticker:
                carry_forward_name_counter[ticker] = carry_forward_name_counter.get(ticker, 0) + 1
    top_carry_forward_names = [
        ticker
        for ticker, _count in sorted(
            carry_forward_name_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    latest_review_plan_run_id = ""
    latest_review_plan_generated_at = ""
    review_plan_count = 0
    review_plan_high_priority_count = 0
    review_plan_bucket_counts: dict[str, int] = {}
    review_plan_name_counter: dict[str, int] = {}
    for row in child_review_plan_rows:
        review_plan_count += int(row.get("total_review_plan_eligible") or 0)
        review_plan_high_priority_count += int(row.get("high_priority_count") or 0)
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_review_plan_generated_at:
            latest_review_plan_generated_at = generated_at
            latest_review_plan_run_id = str(row.get("latest_review_plan_run_id") or "")
        for bucket, count in (row.get("bucket_counts") or {}).items():
            review_plan_bucket_counts[str(bucket)] = review_plan_bucket_counts.get(str(bucket), 0) + int(count or 0)
        for rp_row in row.get("top_review_plan_names") or []:
            ticker = str(rp_row.get("ticker") or "").strip().upper()
            if ticker:
                review_plan_name_counter[ticker] = review_plan_name_counter.get(ticker, 0) + 1
    top_review_plan_names = [
        ticker
        for ticker, _count in sorted(
            review_plan_name_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    latest_review_intake_run_id = ""
    latest_review_intake_generated_at = ""
    review_intake_count = 0
    review_intake_high_priority_count = 0
    review_outcome_counts: dict[str, int] = {}
    review_intake_name_counter: dict[str, int] = {}
    for row in child_review_intake_rows:
        review_intake_count += int(row.get("total_review_intake_eligible") or 0)
        review_intake_high_priority_count += int(row.get("high_priority_count") or 0)
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_review_intake_generated_at:
            latest_review_intake_generated_at = generated_at
            latest_review_intake_run_id = str(row.get("latest_review_intake_run_id") or "")
        for outcome_cls, count in (row.get("review_outcome_counts") or {}).items():
            review_outcome_counts[str(outcome_cls)] = review_outcome_counts.get(str(outcome_cls), 0) + int(count or 0)
        for ri_row in row.get("top_review_reentry_names") or []:
            ticker = str(ri_row.get("ticker") or "").strip().upper()
            if ticker:
                review_intake_name_counter[ticker] = review_intake_name_counter.get(ticker, 0) + 1
    top_review_intake_names = [
        ticker
        for ticker, _count in sorted(
            review_intake_name_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    latest_review_memory_run_id = ""
    latest_review_memory_generated_at = ""
    review_productivity_counts: dict[str, int] = {}
    review_memory_high_productivity_name_counter: dict[str, int] = {}
    review_memory_churn_risk_name_counter: dict[str, int] = {}
    action_efficacy_snapshot: dict[str, str] = {}
    for row in child_review_memory_rows:
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_review_memory_generated_at:
            latest_review_memory_generated_at = generated_at
            latest_review_memory_run_id = str(row.get("latest_review_memory_run_id") or "")
        for cls, count in (row.get("productivity_counts") or {}).items():
            review_productivity_counts[str(cls)] = review_productivity_counts.get(str(cls), 0) + int(count or 0)
        for hp_row in row.get("top_high_productivity_names") or []:
            ticker = str(hp_row.get("ticker") or "").strip().upper()
            if ticker:
                review_memory_high_productivity_name_counter[ticker] = (
                    review_memory_high_productivity_name_counter.get(ticker, 0) + 1
                )
        for churn_row in row.get("top_churn_risk_names") or []:
            ticker = str(churn_row.get("ticker") or "").strip().upper()
            if ticker:
                review_memory_churn_risk_name_counter[ticker] = (
                    review_memory_churn_risk_name_counter.get(ticker, 0) + 1
                )
    top_high_productivity_review_names = [
        ticker
        for ticker, _count in sorted(
            review_memory_high_productivity_name_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    top_review_churn_risk_names = [
        ticker
        for ticker, _count in sorted(
            review_memory_churn_risk_name_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    latest_review_cycle_run_id = ""
    latest_review_cycle_generated_at = ""
    review_cycle_reentered_count = 0
    for row in child_review_cycle_rows:
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_review_cycle_generated_at:
            latest_review_cycle_generated_at = generated_at
            latest_review_cycle_run_id = str(row.get("latest_review_cycle_run_id") or "")
        review_cycle_reentered_count += int(row.get("total_selected") or 0)
    latest_review_compare_run_id = ""
    latest_review_compare_generated_at = ""
    review_compare_delta_counts: dict[str, int] = {}
    review_compare_gain_counter: dict[str, int] = {}
    review_compare_neg_conf_counter: dict[str, int] = {}
    for row in child_review_compare_rows:
        generated_at = str(row.get("generated_at") or "")
        if generated_at >= latest_review_compare_generated_at:
            latest_review_compare_generated_at = generated_at
            latest_review_compare_run_id = str(row.get("latest_review_compare_run_id") or "")
        for cls, count in (row.get("delta_counts") or {}).items():
            review_compare_delta_counts[str(cls)] = review_compare_delta_counts.get(str(cls), 0) + int(count or 0)
        for gain_row in row.get("top_gain_names") or []:
            ticker = str(gain_row.get("ticker") or "").strip().upper()
            if ticker:
                review_compare_gain_counter[ticker] = review_compare_gain_counter.get(ticker, 0) + 1
        for neg_row in row.get("top_negative_confirmation_names") or []:
            ticker = str(neg_row.get("ticker") or "").strip().upper()
            if ticker:
                review_compare_neg_conf_counter[ticker] = review_compare_neg_conf_counter.get(ticker, 0) + 1
    top_review_gain_names = [
        ticker
        for ticker, _count in sorted(
            review_compare_gain_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    top_review_negative_confirmation_names = [
        ticker
        for ticker, _count in sorted(
            review_compare_neg_conf_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:10]
    ]
    return {
        "campaign_run_id": str(state.get("campaign_run_id") or ""),
        "status": str(state.get("status") or CAMPAIGN_RUNNING),
        "policy": _effective_campaign_policy(state),
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "updated_at": utc_now_iso(),
        "latest_validation_run_id": latest_validation_run_id,
        "benchmark_pass_rate": round(validation_pass_weight / validation_case_count, 4) if validation_case_count else 0.0,
        "benchmark_mixed_rate": round(validation_mixed_weight / validation_case_count, 4) if validation_case_count else 0.0,
        "top_validation_failure_modes": top_validation_failure_modes,
        "latest_filter_audit_run_id": latest_filter_audit_run_id,
        "filter_evidence_stop_rate": round(filter_evidence_stop_count / filter_total_ticker_count, 4)
        if filter_total_ticker_count
        else 0.0,
        "filter_economics_stop_rate": round(filter_economics_stop_count / filter_total_ticker_count, 4)
        if filter_total_ticker_count
        else 0.0,
        "filter_queue_budget_stop_rate": round(filter_queue_budget_stop_count / filter_total_ticker_count, 4)
        if filter_total_ticker_count
        else 0.0,
        "filter_memo_survivor_rate": round(filter_memo_survivor_count / filter_total_ticker_count, 4)
        if filter_total_ticker_count
        else 0.0,
        "top_filter_failure_modes": top_filter_failure_modes,
        "latest_borderline_audit_run_id": latest_borderline_audit_run_id,
        "rescue_queue_count": int(rescue_queue_total),
        "borderline_case_rate": round(borderline_weight / borderline_total_stopped, 4)
        if borderline_total_stopped
        else 0.0,
        "top_rescue_queue_names": top_rescue_queue_names,
        "latest_second_look_run_id": latest_second_look_run_id,
        "second_look_review_count": int(second_look_review_count),
        "second_look_possible_false_negative_count": int(second_look_possible_false_negative_count),
        "second_look_evidence_review_count": int(second_look_evidence_review_count),
        "second_look_queue_review_count": int(second_look_queue_review_count),
        "top_second_look_reconsideration_names": top_second_look_reconsideration_names,
        "latest_carry_forward_run_id": latest_carry_forward_run_id,
        "carry_forward_count": int(carry_forward_count),
        "carry_forward_high_priority_count": int(carry_forward_high_priority_count),
        "top_carry_forward_names": top_carry_forward_names,
        "latest_review_plan_run_id": latest_review_plan_run_id,
        "review_plan_count": int(review_plan_count),
        "review_plan_high_priority_count": int(review_plan_high_priority_count),
        "review_plan_bucket_counts": review_plan_bucket_counts,
        "top_review_plan_names": top_review_plan_names,
        "latest_review_intake_run_id": latest_review_intake_run_id,
        "review_intake_count": int(review_intake_count),
        "review_intake_high_priority_count": int(review_intake_high_priority_count),
        "review_outcome_counts": review_outcome_counts,
        "top_review_intake_names": top_review_intake_names,
        "latest_review_memory_run_id": latest_review_memory_run_id,
        "review_productivity_counts": review_productivity_counts,
        "top_high_productivity_review_names": top_high_productivity_review_names,
        "top_review_churn_risk_names": top_review_churn_risk_names,
        "action_efficacy_snapshot": action_efficacy_snapshot,
        "latest_review_cycle_run_id": latest_review_cycle_run_id,
        "review_cycle_reentered_count": int(review_cycle_reentered_count),
        "latest_review_compare_run_id": latest_review_compare_run_id,
        "review_compare_delta_counts": review_compare_delta_counts,
        "top_review_gain_names": top_review_gain_names,
        "top_review_negative_confirmation_names": top_review_negative_confirmation_names,
        "latest_filter_calibration_run_id": latest_filter_calibration_run_id,
        "calibration_high_severity_count": int(calibration_high_severity_count),
        "top_filter_calibration_recommendations": top_filter_calibration_recommendations,
        "item_counts": counts,
        "top_10_master_shortlist": top_10,
        "top_memory_priority_candidates": [
            {
                "ticker": str(row.get("ticker") or ""),
                "value_gate_status": str(
                    row.get("value_gate_status")
                    or row.get("latest_value_gate_status")
                    or UNKNOWN
                ),
                "implied_return_base": row.get("implied_return_base", UNKNOWN),
                "oe_quality_total": row.get("oe_quality_total", UNKNOWN),
                "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
                "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
                "returns_persistence_class": str(
                    row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
                ),
                "primary_returns_caution": str(
                    row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
                ),
                "reinvestment_efficiency_class": str(
                    row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
                ),
                "primary_reinvestment_caution": str(
                    row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
                ),
                "asset_intensity_class": str(
                    row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
                ),
                "maintenance_capex_credibility_class": str(
                    row.get("maintenance_capex_credibility_class") or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
                ),
                "primary_maintenance_capex_caution": str(
                    row.get("primary_maintenance_capex_caution") or "OWNER_EARNINGS_UNCLEAR"
                ),
                "asset_quality_class": str(
                    row.get("asset_quality_class") or "ASSET_QUALITY_UNKNOWN"
                ),
                "downside_realization_credibility_class": str(
                    row.get("downside_realization_credibility_class")
                    or "DOWNSIDE_REALIZATION_UNKNOWN"
                ),
                "primary_asset_support_caution": str(
                    row.get("primary_asset_support_caution") or "ASSET_SUPPORT_UNCLEAR"
                ),
                "obligation_burden_class": str(
                    row.get("obligation_burden_class") or "OBLIGATION_BURDEN_UNKNOWN"
                ),
                "claim_priority_pressure_class": str(
                    row.get("claim_priority_pressure_class") or "CLAIM_PRIORITY_PRESSURE_UNKNOWN"
                ),
                "primary_obligation_caution": str(
                    row.get("primary_obligation_caution") or "RESIDUAL_EQUITY_UNCLEAR"
                ),
                "memory_priority_total": int(row.get("memory_priority_total") or 0),
                "memory_priority_reason_codes": [
                    str(code)
                    for code in (row.get("memory_priority_reason_codes") or [])
                    if str(code).strip()
                ],
            }
            for row in top_memory_priority_candidates
        ],
        "lane_counts": promotion_state.get("lane_counts") if isinstance(promotion_state.get("lane_counts"), dict) else {},
        "top_10_promotion_candidates": top_promotion,
        "top_blockers_preventing_lane_1": promotion_state.get("top_blockers_preventing_lane_1")
        if isinstance(promotion_state.get("top_blockers_preventing_lane_1"), list)
        else [],
        "retryable_facts_blocker_count": int(retryable_facts_blocker_count),
        "terminal_facts_blocker_count": int(terminal_facts_blocker_count),
        "partial_usable_facts_count": int(partial_usable_facts_count),
        "top_retryable_facts_blockers": sorted(
            top_retryable_facts_blockers,
            key=lambda row: (
                str(row.get("facts_blocker_class") or ""),
                str(row.get("ticker") or ""),
                str(row.get("child_universe_run_id") or ""),
            ),
        )[:10],
        "child_runs_with_facts_degradation": sorted(
            child_runs_with_facts_degradation,
            key=lambda row: (
                -int(row.get("retryable_facts_blocker_count") or 0),
                -int(row.get("terminal_facts_blocker_count") or 0),
                str(row.get("child_universe_run_id") or ""),
            ),
        )[:10],
        "economic_fail_count_vs_evidence_fail_count": {
            "evidence_fail_count": int(evidence_fail_count),
            "economic_fail_count": int(economic_fail_count),
            "mixed_fail_count": int(mixed_fail_count),
            "other_fail_count": int(other_fail_count),
        },
        "item_blockers": item_blockers[:10],
        "escalation_queue_count": int(escalation_summary.get("escalation_queue_count") or 0),
        "lane_to_action_counts": escalation_summary.get("lane_to_action_counts")
        if isinstance(escalation_summary.get("lane_to_action_counts"), dict)
        else {},
        "top_10_escalation_actions": escalation_summary.get("top_10_escalation_actions")
        if isinstance(escalation_summary.get("top_10_escalation_actions"), list)
        else [],
        "executed_queue_count": int(escalation_summary.get("executed_queue_count") or 0),
        "completed_action_counts": escalation_summary.get("completed_action_counts")
        if isinstance(escalation_summary.get("completed_action_counts"), dict)
        else {},
        "lane_change_counts": escalation_summary.get("lane_change_counts")
        if isinstance(escalation_summary.get("lane_change_counts"), dict)
        else {},
        "active_watchlist_count": int(escalation_summary.get("active_watchlist_count") or 0),
        "research_memory_path": str(research_memory_path),
        "research_memory_summary_path": str(research_memory_summary_path),
        "top_improvers": research_memory_summary.get("top_improvers")
        if isinstance(research_memory_summary.get("top_improvers"), list)
        else [],
        "top_deteriorations": research_memory_summary.get("top_deteriorations")
        if isinstance(research_memory_summary.get("top_deteriorations"), list)
        else [],
        "master_shortlist_json_path": str(paths["master_shortlist_json_path"]),
        "master_shortlist_md_path": str(paths["master_shortlist_md_path"]),
        "master_watchlist_state_path": str(paths["master_watchlist_state_path"]),
        "promotion_state_path": str(paths["promotion_state_path"]),
        "promotion_candidates_path": str(paths["promotion_candidates_path"]),
        "priority_lanes_path": str(paths["priority_lanes_path"]),
        "escalation_plan_path": str(paths["escalation_plan_path"]),
        "escalation_queue_path": str(paths["escalation_queue_path"]),
        "escalation_summary_path": str(paths["escalation_summary_path"]),
        "watchlist_ticker_count": int(master_watchlist.get("ticker_count") or 0),
        "validated_child_runs": child_validation_rows[:10],
        "filter_audited_child_runs": child_filter_rows[:10],
        "borderline_audited_child_runs": child_borderline_rows[:10],
        "second_look_child_runs": child_second_look_rows[:10],
        # Orchestrator visibility fields (optional, populated externally)
        "latest_director_run_id": str(state.get("latest_director_run_id") or ""),
        "director_attention_target_count": int(state.get("director_attention_target_count") or 0),
        "director_blocker_clear_count": int(state.get("director_blocker_clear_count") or 0),
        "director_leave_dead_count": int(state.get("director_leave_dead_count") or 0),
        "director_reconsider_queue_count": int(state.get("director_reconsider_queue_count") or 0),
        "top_director_recommendations": [
            row for row in (state.get("top_director_recommendations") or [])
            if isinstance(row, dict)
        ][:5],
        # Orchestrator evaluation visibility fields (optional, populated externally)
        "latest_director_evaluation_run_id": state.get("latest_director_evaluation_run_id"),
        "director_eval_result_counts": state.get("director_eval_result_counts", {}),
        "director_doctrine_violation_count": state.get("director_doctrine_violation_count", 0),
        "top_director_eval_failures": state.get("top_director_eval_failures", []),
        # Orchestrator alignment visibility fields (optional, populated externally)
        "latest_director_alignment_run_id": state.get("latest_director_alignment_run_id"),
        "director_alignment_counts": state.get("director_alignment_counts", {}),
        "useful_director_disagreement_count": state.get("useful_director_disagreement_count", 0),
        "noisy_director_disagreement_count": state.get("noisy_director_disagreement_count", 0),
        "doctrine_risk_director_disagreement_count": state.get("doctrine_risk_director_disagreement_count", 0),
        "top_director_useful_disagreements": state.get("top_director_useful_disagreements", []),
        # Orchestrator shadow visibility fields (optional, populated externally — advisory only)
        "latest_director_shadow_run_id": state.get("latest_director_shadow_run_id"),
        "latest_director_shadow_outcomes_run_id": state.get("latest_director_shadow_outcomes_run_id"),
        "director_shadow_counts": state.get("director_shadow_counts", {}),
        "director_shadow_outcome_counts": state.get("director_shadow_outcome_counts", {}),
        "top_validated_director_recommendations": [
            row for row in (state.get("top_validated_director_recommendations") or [])
            if isinstance(row, (str, dict))
        ][:5],
        "top_doctrine_risk_director_recommendations": [
            row for row in (state.get("top_doctrine_risk_director_recommendations") or [])
            if isinstance(row, (str, dict))
        ][:5],
        # Orchestrator shadow scenario benchmark visibility (optional, advisory only)
        "latest_director_shadow_scenario_run_id": state.get("latest_director_shadow_scenario_run_id"),
        "director_shadow_scenario_counts": state.get("director_shadow_scenario_counts", {}),
        "director_shadow_scenario_validated_count": int(
            state.get("director_shadow_scenario_validated_count") or 0
        ),
        "director_shadow_scenario_doctrine_risk_count": int(
            state.get("director_shadow_scenario_doctrine_risk_count") or 0
        ),
        # Orchestrator tuning governance visibility (optional, advisory only)
        "latest_director_governance_run_id": state.get("latest_director_governance_run_id"),
        "director_governance_verdict": str(
            state.get("director_governance_verdict") or ""
        ),
        "director_change_control_top_recommendations": list(
            state.get("director_change_control_top_recommendations") or []
        )[:5],
        "director_governance_target_counts": state.get("director_governance_target_counts", {}),
        "director_governance_doctrine_risk_snapshot": int(
            state.get("director_governance_doctrine_risk_snapshot") or 0
        ),
        # Orchestrator variant registry visibility (optional, advisory only)
        "latest_director_variant_registry_run_id": state.get("latest_director_variant_registry_run_id"),
        "director_variant_counts": state.get("director_variant_counts", {}),
        "director_variant_top_promising": list(
            state.get("director_variant_top_promising") or []
        )[:3],
        "director_variant_top_too_thin": list(
            state.get("director_variant_top_too_thin") or []
        )[:3],
        "director_variant_governance_snapshot": str(
            state.get("director_variant_governance_snapshot") or ""
        ),
        "director_variant_drift_snapshot": str(
            state.get("director_variant_drift_snapshot") or ""
        ),
        # Orchestrator source-run battery visibility (optional, advisory only)
        "latest_director_source_battery_run_id": state.get(
            "latest_director_source_battery_run_id"
        ),
        "director_source_battery_source_counts": state.get(
            "director_source_battery_source_counts", {}
        ),
        "director_source_battery_top_sources": list(
            state.get("director_source_battery_top_sources") or []
        )[:3],
        "director_source_battery_top_variants": list(
            state.get("director_source_battery_top_variants") or []
        )[:3],
        "director_source_battery_verdict": str(
            state.get("director_source_battery_verdict") or ""
        ),
        # Orchestrator source qualification + battery curation visibility (optional, advisory only)
        "latest_director_source_qualification_run_id": state.get(
            "latest_director_source_qualification_run_id"
        ),
        "director_source_qualification_counts": state.get(
            "director_source_qualification_counts", {}
        ),
        "director_battery_curation_counts": state.get(
            "director_battery_curation_counts", {}
        ),
        "director_top_curated_sources": list(
            state.get("director_top_curated_sources") or []
        )[:3],
        "director_top_excluded_sources": list(
            state.get("director_top_excluded_sources") or []
        )[:3],
        # Orchestrator battery harvest visibility (optional, advisory only)
        "latest_director_battery_harvest_run_id": state.get(
            "latest_director_battery_harvest_run_id"
        ),
        "director_battery_harvest_counts": state.get(
            "director_battery_harvest_counts", {}
        ),
        "director_battery_harvest_top_sources": list(
            state.get("director_battery_harvest_top_sources") or []
        )[:3],
        "director_battery_harvest_top_variants": list(
            state.get("director_battery_harvest_top_variants") or []
        )[:3],
        "director_battery_harvest_verdict": str(
            state.get("director_battery_harvest_verdict") or ""
        ),
        # Orchestrator source yield registry visibility (optional, advisory only)
        "latest_director_source_yield_run_id": state.get("latest_director_source_yield_run_id"),
        "director_source_yield_counts": state.get("director_source_yield_counts", {}),
        "director_top_high_yield_sources": list(
            state.get("director_top_high_yield_sources") or []
        )[:3],
        "director_top_too_thin_sources": list(
            state.get("director_top_too_thin_sources") or []
        )[:3],
        "director_source_yield_governance_snapshot": str(
            state.get("director_source_yield_governance_snapshot") or ""
        ),
        "director_source_yield_drift_snapshot": str(
            state.get("director_source_yield_drift_snapshot") or ""
        ),
        # Orchestrator harvest continuation / follow-through visibility (optional, advisory only)
        "latest_director_harvest_continuation_run_id": state.get(
            "latest_director_harvest_continuation_run_id"
        ),
        "latest_director_harvest_followthrough_run_id": state.get(
            "latest_director_harvest_followthrough_run_id"
        ),
        "director_harvest_continuation_counts": state.get(
            "director_harvest_continuation_counts", {}
        ),
        "director_harvest_followthrough_counts": state.get(
            "director_harvest_followthrough_counts", {}
        ),
        "top_director_promising_pairs": list(
            state.get("top_director_promising_pairs") or []
        )[:3],
        # Orchestrator diversity gap / next-harvest visibility (optional, advisory only)
        "latest_director_diversity_gap_run_id": state.get(
            "latest_director_diversity_gap_run_id"
        ),
        "director_diversity_gap_counts": state.get("director_diversity_gap_counts", {}),
        "director_next_harvest_pair_count": int(
            state.get("director_next_harvest_pair_count") or 0
        ),
        "director_top_diversity_gap_hotspots": list(
            state.get("director_top_diversity_gap_hotspots") or []
        )[:3],
        "director_top_next_harvest_pairs": list(
            state.get("director_top_next_harvest_pairs") or []
        )[:3],
        # Orchestrator next-harvest execution + diversity closure visibility (optional, advisory only)
        "latest_director_next_harvest_execution_run_id": state.get(
            "latest_director_next_harvest_execution_run_id"
        ),
        "director_next_harvest_execution_counts": state.get(
            "director_next_harvest_execution_counts", {}
        ),
        "director_diversity_closure_counts": state.get(
            "director_diversity_closure_counts", {}
        ),
        "director_top_diversity_improved_pairs": list(
            state.get("director_top_diversity_improved_pairs") or []
        )[:3],
        "director_diversity_closure_verdict": str(
            state.get("director_diversity_closure_verdict") or ""
        ),
        # Orchestrator evidence refresh + rebase visibility (optional, advisory only)
        "latest_director_evidence_refresh_run_id": state.get(
            "latest_director_evidence_refresh_run_id"
        ),
        "director_evidence_refresh_stage_counts": dict(
            state.get("director_evidence_refresh_stage_counts") or {}
        ),
        "director_rebase_snapshot": str(
            state.get("director_rebase_snapshot") or ""
        ),
        "director_top_rebase_improvements": list(
            state.get("director_top_rebase_improvements") or []
        )[:3],
        "director_top_still_thin_areas": list(
            state.get("director_top_still_thin_areas") or []
        )[:3],
        # Replay-backed downstream rebase + signal propagation visibility (optional, advisory only)
        "latest_director_replay_rebase_run_id": state.get(
            "latest_director_replay_rebase_run_id"
        ),
        "director_replay_rebase_verdict": str(
            state.get("director_replay_rebase_verdict") or ""
        ),
        "director_replay_backed_proportion": float(
            state.get("director_replay_backed_proportion") or 0.0
        ),
        "director_signal_propagation_result_class": str(
            state.get("director_signal_propagation_result_class") or ""
        ),
        "director_replay_dominant_blocker": str(
            state.get("director_replay_dominant_blocker") or ""
        ),
        # Orchestrator battery materialization + readiness visibility (optional, advisory only)
        "latest_director_battery_materialize_run_id": state.get(
            "latest_director_battery_materialize_run_id"
        ),
        "director_battery_materialize_counts": dict(
            state.get("director_battery_materialize_counts") or {}
        ),
        "director_battery_readiness_counts": dict(
            state.get("director_battery_readiness_counts") or {}
        ),
        "director_top_materialized_sources": list(
            state.get("director_top_materialized_sources") or []
        )[:3],
        "director_battery_refresh_readiness_verdict": str(
            state.get("director_battery_refresh_readiness_verdict") or ""
        ),
        # Orchestrator experiment matrix visibility (optional, advisory only)
        "latest_director_experiment_run_id": state.get("latest_director_experiment_run_id"),
        "director_experiment_variant_counts": state.get("director_experiment_variant_counts", {}),
        "director_experiment_top_variant": str(
            state.get("director_experiment_top_variant") or ""
        ),
        "director_experiment_pairwise_snapshot": str(
            state.get("director_experiment_pairwise_snapshot") or ""
        ),
        "director_experiment_evidence_present": bool(
            state.get("director_experiment_evidence_present")
        ),
        # Orchestrator recommendation mix calibration visibility (optional, advisory only)
        "latest_director_calibration_run_id": state.get("latest_director_calibration_run_id"),
        "director_recommendation_mix_counts": state.get("director_recommendation_mix_counts", {}),
        "dominant_director_recommendation_type": str(
            state.get("dominant_director_recommendation_type") or ""
        ),
        "dominant_director_recommendation_share": float(
            state.get("dominant_director_recommendation_share") or 0.0
        ),
        "top_director_calibration_recommendations": list(
            state.get("top_director_calibration_recommendations") or []
        )[:5],
        # Orchestrator shadow compare visibility (optional, advisory only)
        "latest_director_shadow_compare_run_id": state.get(
            "latest_director_shadow_compare_run_id"
        ),
        "director_shadow_compare_counts": state.get("director_shadow_compare_counts", {}),
        "director_shadow_calibration_verdict": str(
            state.get("director_shadow_calibration_verdict") or ""
        ),
        "top_improved_director_shadow_tickers": list(
            state.get("top_improved_director_shadow_tickers") or []
        )[:5],
        "top_worse_director_shadow_tickers": list(
            state.get("top_worse_director_shadow_tickers") or []
        )[:5],
        # Orchestrator shadow replay harness visibility (optional, advisory only)
        "latest_director_shadow_replay_director_run_id": state.get(
            "latest_director_shadow_replay_director_run_id"
        ),
        "latest_director_shadow_replay_new_run_id": state.get(
            "latest_director_shadow_replay_new_run_id"
        ),
        "director_shadow_replay_inject_count": int(
            state.get("director_shadow_replay_inject_count") or 0
        ),
        "director_shadow_replay_compare_verdict": str(
            state.get("director_shadow_replay_compare_verdict") or ""
        ),
        "director_shadow_replay_signal_gained_count": int(
            state.get("director_shadow_replay_signal_gained_count") or 0
        ),
        # Orchestrator shadow replay signal extraction compare visibility (optional, advisory only)
        "latest_director_shadow_replay_compare_run_id": state.get(
            "latest_director_shadow_replay_compare_run_id"
        ),
        "director_shadow_replay_result_counts": dict(
            state.get("director_shadow_replay_result_counts") or {}
        ),
        "director_shadow_replay_injected_signal_gain_rate": float(
            state.get("director_shadow_replay_injected_signal_gain_rate") or 0.0
        ),
        "director_shadow_replay_monitor_skip_confirmed_rate": float(
            state.get("director_shadow_replay_monitor_skip_confirmed_rate") or 0.0
        ),
        "director_shadow_replay_verdict": str(
            state.get("director_shadow_replay_verdict") or ""
        ),
        "top_director_replay_signal_names": list(
            state.get("top_director_replay_signal_names") or []
        )[:5],
        # Replay efficacy calibration visibility (optional, advisory only)
        "latest_director_replay_calibration_run_id": state.get(
            "latest_director_replay_calibration_run_id"
        ),
        "director_replay_family_efficacy_snapshot": str(
            state.get("director_replay_family_efficacy_snapshot") or ""
        ),
        "director_replay_priority_policy_result": str(
            state.get("director_replay_priority_policy_result") or ""
        ),
        "director_replay_injected_cohort_verdict": str(
            state.get("director_replay_injected_cohort_verdict") or ""
        ),
        "director_replay_monitor_cohort_verdict": str(
            state.get("director_replay_monitor_cohort_verdict") or ""
        ),
        "top_director_replay_calibration_findings": list(
            state.get("top_director_replay_calibration_findings") or []
        )[:5],
        # Replay ledger + longitudinal policy stability visibility (optional, advisory only)
        "latest_director_replay_ledger_run_id": state.get(
            "latest_director_replay_ledger_run_id"
        ),
        "director_policy_stability_class": str(
            state.get("director_policy_stability_class") or ""
        ),
        "director_policy_drift_class": str(
            state.get("director_policy_drift_class") or ""
        ),
        "director_replay_sample_size_class": str(
            state.get("director_replay_sample_size_class") or ""
        ),
        "top_director_policy_stability_findings": list(
            state.get("top_director_policy_stability_findings") or []
        )[:5],
    }


def _refresh_master_outputs(paths: dict[str, Path], state: dict[str, Any]) -> None:
    master_shortlist = _write_master_shortlist(paths, state)
    master_watchlist = _write_master_watchlist_state(paths, state)
    director_metadata = _load_director_metadata(paths)
    from app.universe.promotion import write_promotion_artifacts

    promotion_paths = write_promotion_artifacts(
        str(state.get("campaign_run_id") or ""),
        master_watchlist_state=master_watchlist,
        master_shortlist=master_shortlist,
    )
    from app.universe.escalation import write_escalation_artifacts

    write_escalation_artifacts(
        str(state.get("campaign_run_id") or ""),
        promotion_state=_safe_json(Path(str(promotion_paths["promotion_state_path"]))),
        priority_lanes=_safe_json(Path(str(promotion_paths["priority_lanes_path"]))),
        campaign_state=state,
    )
    from app.universe.research_memory import update_research_memory_from_campaign

    research_summary_seed = _build_summary(paths, state)
    research_summary_seed["campaign_summary_path"] = str(paths["summary_path"])
    update_research_memory_from_campaign(
        str(state.get("campaign_run_id") or ""),
        campaign_summary=research_summary_seed,
        master_shortlist=master_shortlist,
        master_watchlist_state=master_watchlist,
        promotion_state=_safe_json(Path(str(promotion_paths["promotion_state_path"]))),
        escalation_results={
            **_safe_json(paths["root"] / "escalation_results.json"),
            "escalation_results_path": str(paths["root"] / "escalation_results.json"),
        },
        director_metadata=director_metadata,
    )
    _write_state_and_summary(paths, state)


def _run_autopilot(
    *,
    universe_run_id: str,
    as_of_date: str,
    scout_params: dict[str, Any],
    depth_batch_params: dict[str, Any],
    rollup_params: dict[str, Any],
    dossier_pack_params: dict[str, Any],
    memo_pack_params: dict[str, Any],
    resume: bool,
    force: bool,
    depth: str = "fundamentals",
) -> dict[str, Any]:
    from app.universe.autopilot import run_universe_autopilot

    return run_universe_autopilot(
        universe_run_id=universe_run_id,
        as_of_date=as_of_date,
        scout_params=scout_params,
        depth_batch_params=depth_batch_params,
        rollup_params=rollup_params,
        dossier_pack_params=dossier_pack_params,
        memo_pack_params=memo_pack_params,
        resume=resume,
        depth=depth,
        force=force,
    )


def _open_autopilot(universe_run_id: str) -> dict[str, Any]:
    from app.universe.autopilot import open_universe_autopilot

    return open_universe_autopilot(universe_run_id)


def _cancel_autopilot(universe_run_id: str, reason: str) -> dict[str, Any]:
    from app.universe.autopilot import cancel_universe_autopilot

    return cancel_universe_autopilot(universe_run_id, reason)


def run_campaign(
    spec: Path | str | dict[str, Any],
    *,
    campaign_run_id: str,
    resume: bool = True,
    force: bool = False,
    max_items: int | None = None,
    depth: str = "fundamentals",
) -> dict[str, Any]:
    spec_payload = load_campaign_spec(spec)
    spec_payload["campaign_run_id"] = campaign_run_id
    plan = build_campaign_plan(spec_payload)
    paths = _campaign_paths(campaign_run_id)

    if force:
        state = _default_state(campaign_run_id=campaign_run_id, spec=spec_payload, plan=plan)
    else:
        state = _safe_json(paths["state_path"]) if resume else {}
        if state:
            existing_sig = [str(token) for token in (((state.get("spec_snapshot") or {}).get("item_ids")) or []) if str(token).strip()]
            new_sig = _state_items_signature(plan)
            if existing_sig and existing_sig != new_sig:
                raise ValueError("Existing campaign spec differs from requested spec; rerun with --force.")
            items_by_id = {
                str(item.get("item_id") or ""): item
                for item in [row for row in (state.get("items") or []) if isinstance(row, dict)]
            }
            merged_items: list[dict[str, Any]] = []
            for item in plan:
                existing = items_by_id.get(str(item.get("item_id") or ""))
                if existing:
                    merged = dict(item)
                    merged.update(
                        {
                            key: value
                            for key, value in existing.items()
                            if key in {"status", "started_at", "finished_at", "updated_at", "error", "artifact_paths"}
                        }
                    )
                    merged_items.append(merged)
                else:
                    merged_items.append(item)
            state["items"] = merged_items
            state["spec_snapshot"] = {
                "as_of_date": str(spec_payload.get("as_of_date") or ""),
                "item_count": len(plan),
                "item_ids": _state_items_signature(plan),
            }
            state["source_campaign_file"] = str(spec_payload.get("source_path") or "")
            state["policy"] = _campaign_policy_from_plan(plan)
        else:
            state = _default_state(campaign_run_id=campaign_run_id, spec=spec_payload, plan=plan)

    if str(state.get("status") or "").upper() == CAMPAIGN_CANCELLED:
        _write_state_and_summary(paths, state)
        return {
            "status": CAMPAIGN_CANCELLED,
            "campaign_run_id": campaign_run_id,
            "campaign_state_path": str(paths["state_path"]),
            "campaign_summary_path": str(paths["summary_path"]),
        }

    executed = 0
    state["status"] = CAMPAIGN_RUNNING
    state["stop_reason_code"] = ""
    state["stop_summary"] = ""
    _write_state_and_summary(paths, state)

    for idx, item in enumerate([row for row in (state.get("items") or []) if isinstance(row, dict)]):
        latest_state = _safe_json(paths["state_path"])
        if str(latest_state.get("status") or "").upper() == CAMPAIGN_CANCELLED:
            state = latest_state
            break

        if _is_num(max_items) and int(max_items) > 0 and executed >= int(max_items):
            state["status"] = CAMPAIGN_PARTIAL
            state["stop_reason_code"] = STOP_MAX_ITEMS_REACHED
            state["stop_summary"] = "Invocation item cap reached."
            break

        item_status = str(item.get("status") or ITEM_PENDING).upper()
        if resume and not force and item_status == ITEM_DONE and _item_artifacts_exist(item):
            continue

        state["active_item_id"] = str(item.get("item_id") or "")
        item["status"] = ITEM_RUNNING
        item["started_at"] = utc_now_iso()
        item["updated_at"] = utc_now_iso()
        item["error"] = None
        _write_state_and_summary(paths, state)
        _append_jsonl(
            paths["log_path"],
            {
                "ts": utc_now_iso(),
                "event": "campaign_item_started",
                "campaign_run_id": campaign_run_id,
                "item_id": item.get("item_id"),
                "label": item.get("label"),
                "child_universe_run_id": item.get("child_universe_run_id"),
            },
        )

        result = _run_autopilot(
            universe_run_id=str(item.get("child_universe_run_id") or ""),
            as_of_date=str(item.get("as_of_date") or ""),
            scout_params=item.get("scout_params") if isinstance(item.get("scout_params"), dict) else {},
            depth_batch_params=item.get("depth_batch_params") if isinstance(item.get("depth_batch_params"), dict) else {},
            rollup_params=item.get("rollup_params") if isinstance(item.get("rollup_params"), dict) else {},
            dossier_pack_params=item.get("dossier_pack_params") if isinstance(item.get("dossier_pack_params"), dict) else {},
            memo_pack_params=item.get("memo_pack_params") if isinstance(item.get("memo_pack_params"), dict) else {},
            resume=resume and not force,
            depth=depth,
            force=force,
        )
        child_open = _open_autopilot(str(item.get("child_universe_run_id") or ""))
        stages = child_open.get("stages") if isinstance(child_open.get("stages"), dict) else {}
        memo_stage = stages.get("MEMO_PACK") if isinstance(stages.get("MEMO_PACK"), dict) else {}
        rollup_stage = stages.get("ROLLUP") if isinstance(stages.get("ROLLUP"), dict) else {}
        dossier_stage = stages.get("DOSSIER_PACK") if isinstance(stages.get("DOSSIER_PACK"), dict) else {}

        item["artifact_paths"] = _child_expected_artifacts(str(item.get("child_universe_run_id") or ""))
        item["artifact_paths"].update(
            {
                "autopilot_state_path": str(child_open.get("autopilot_state_path") or item["artifact_paths"]["autopilot_state_path"]),
                "autopilot_summary_path": str(child_open.get("autopilot_summary_path") or item["artifact_paths"]["autopilot_summary_path"]),
                "memo_pack_manifest_path": str((memo_stage.get("artifact_paths") or {}).get("manifest_path") or item["artifact_paths"]["memo_pack_manifest_path"]),
                "watchlist_state_path": str((memo_stage.get("artifact_paths") or {}).get("watchlist_state_path") or item["artifact_paths"]["watchlist_state_path"]),
                "global_shortlist_path": str((rollup_stage.get("artifact_paths") or {}).get("global_shortlist_json_path") or item["artifact_paths"]["global_shortlist_path"]),
                "global_rollup_path": str((rollup_stage.get("artifact_paths") or {}).get("global_rollup_json_path") or item["artifact_paths"]["global_rollup_path"]),
                "dossier_pack_manifest_path": str((dossier_stage.get("artifact_paths") or {}).get("manifest_path") or item["artifact_paths"]["dossier_pack_manifest_path"]),
            }
        )

        child_status = str(result.get("status") or child_open.get("run_status") or UNKNOWN).upper()
        if child_status == CAMPAIGN_DONE:
            item["status"] = ITEM_DONE
        elif child_status == CAMPAIGN_CANCELLED or child_status == "CANCELLED":
            item["status"] = ITEM_CANCELLED
            state["status"] = CAMPAIGN_CANCELLED
            state["stop_reason_code"] = STOP_CANCEL_REQUESTED
            state["stop_summary"] = "Child autopilot cancelled."
        elif child_status == CAMPAIGN_FAILED or child_status == "FAILED":
            item["status"] = ITEM_FAILED
            item["error"] = str(result.get("error") or child_open.get("stop_summary") or "Child autopilot failed.")
            state["status"] = CAMPAIGN_PARTIAL
        else:
            item["status"] = ITEM_PARTIAL
            item["error"] = str(result.get("stop_summary") or child_open.get("stop_summary") or "")
            state["status"] = CAMPAIGN_PARTIAL

        item["finished_at"] = utc_now_iso()
        item["updated_at"] = utc_now_iso()
        state["cursor_next_idx"] = int(idx + 1)
        state["active_item_id"] = ""
        executed += 1
        _append_jsonl(
            paths["log_path"],
            {
                "ts": utc_now_iso(),
                "event": "campaign_item_finished",
                "campaign_run_id": campaign_run_id,
                "item_id": item.get("item_id"),
                "label": item.get("label"),
                "status": item.get("status"),
            },
        )
        _refresh_master_outputs(paths, state)

        if str(item.get("status") or "").upper() == ITEM_CANCELLED:
            break

    if str(state.get("status") or "").upper() == CAMPAIGN_RUNNING:
        items = [row for row in (state.get("items") or []) if isinstance(row, dict)]
        if all(str(row.get("status") or "").upper() == ITEM_DONE for row in items):
            state["status"] = CAMPAIGN_DONE
            state["stop_reason_code"] = STOP_COMPLETED
            state["stop_summary"] = "Campaign completed."
        elif any(str(row.get("status") or "").upper() == ITEM_FAILED for row in items):
            state["status"] = CAMPAIGN_PARTIAL
            state["stop_reason_code"] = STOP_COMPLETED
            state["stop_summary"] = "Campaign completed with failed items."
        else:
            state["status"] = CAMPAIGN_PARTIAL
            if not str(state.get("stop_reason_code") or "").strip():
                state["stop_reason_code"] = STOP_MAX_ITEMS_REACHED
                state["stop_summary"] = "Campaign paused before completion."

    _refresh_master_outputs(paths, state)
    final_summary = _safe_json(paths["summary_path"])
    return {
        "status": str(state.get("status") or CAMPAIGN_RUNNING),
        "campaign_run_id": campaign_run_id,
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "campaign_state_path": str(paths["state_path"]),
        "campaign_summary_path": str(paths["summary_path"]),
        "master_shortlist_json_path": str(paths["master_shortlist_json_path"]),
        "master_watchlist_state_path": str(paths["master_watchlist_state_path"]),
        "promotion_state_path": str(paths["promotion_state_path"]),
        "promotion_candidates_path": str(paths["promotion_candidates_path"]),
        "priority_lanes_path": str(paths["priority_lanes_path"]),
        "escalation_plan_path": str(paths["escalation_plan_path"]),
        "escalation_queue_path": str(paths["escalation_queue_path"]),
        "escalation_summary_path": str(paths["escalation_summary_path"]),
        "research_memory_path": str(final_summary.get("research_memory_path") or ""),
        "research_memory_summary_path": str(final_summary.get("research_memory_summary_path") or ""),
    }


def campaign_status(campaign_run_id: str) -> dict[str, Any]:
    paths = _campaign_paths(campaign_run_id)
    state = _safe_json(paths["state_path"])
    if not state:
        return {
            "status": "MISSING",
            "campaign_run_id": campaign_run_id,
            "campaign_state_path": str(paths["state_path"]),
            "campaign_summary_path": str(paths["summary_path"]),
        }
    summary = _safe_json(paths["summary_path"])
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "run_status": str(state.get("status") or CAMPAIGN_RUNNING),
        "policy": str(summary.get("policy") or _effective_campaign_policy(state)),
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "latest_validation_run_id": str(summary.get("latest_validation_run_id") or ""),
        "benchmark_pass_rate": float(summary.get("benchmark_pass_rate") or 0.0),
        "benchmark_mixed_rate": float(summary.get("benchmark_mixed_rate") or 0.0),
        "top_validation_failure_modes": summary.get("top_validation_failure_modes")
        if isinstance(summary.get("top_validation_failure_modes"), list)
        else [],
        "latest_filter_audit_run_id": str(summary.get("latest_filter_audit_run_id") or ""),
        "filter_evidence_stop_rate": float(summary.get("filter_evidence_stop_rate") or 0.0),
        "filter_economics_stop_rate": float(summary.get("filter_economics_stop_rate") or 0.0),
        "filter_queue_budget_stop_rate": float(summary.get("filter_queue_budget_stop_rate") or 0.0),
        "filter_memo_survivor_rate": float(summary.get("filter_memo_survivor_rate") or 0.0),
        "top_filter_failure_modes": summary.get("top_filter_failure_modes")
        if isinstance(summary.get("top_filter_failure_modes"), list)
        else [],
        "latest_borderline_audit_run_id": str(summary.get("latest_borderline_audit_run_id") or ""),
        "rescue_queue_count": int(summary.get("rescue_queue_count") or 0),
        "borderline_case_rate": float(summary.get("borderline_case_rate") or 0.0),
        "top_rescue_queue_names": summary.get("top_rescue_queue_names")
        if isinstance(summary.get("top_rescue_queue_names"), list)
        else [],
        "latest_second_look_run_id": str(summary.get("latest_second_look_run_id") or ""),
        "second_look_review_count": int(summary.get("second_look_review_count") or 0),
        "second_look_possible_false_negative_count": int(
            summary.get("second_look_possible_false_negative_count") or 0
        ),
        "second_look_evidence_review_count": int(summary.get("second_look_evidence_review_count") or 0),
        "second_look_queue_review_count": int(summary.get("second_look_queue_review_count") or 0),
        "top_second_look_reconsideration_names": summary.get("top_second_look_reconsideration_names")
        if isinstance(summary.get("top_second_look_reconsideration_names"), list)
        else [],
        "latest_filter_calibration_run_id": str(summary.get("latest_filter_calibration_run_id") or ""),
        "calibration_high_severity_count": int(summary.get("calibration_high_severity_count") or 0),
        "top_filter_calibration_recommendations": summary.get("top_filter_calibration_recommendations")
        if isinstance(summary.get("top_filter_calibration_recommendations"), list)
        else [],
        "item_counts": summary.get("item_counts") if isinstance(summary.get("item_counts"), dict) else {},
        "lane_counts": summary.get("lane_counts") if isinstance(summary.get("lane_counts"), dict) else {},
        "escalation_queue_count": int(summary.get("escalation_queue_count") or 0),
        "lane_to_action_counts": summary.get("lane_to_action_counts")
        if isinstance(summary.get("lane_to_action_counts"), dict)
        else {},
        "executed_queue_count": int(summary.get("executed_queue_count") or 0),
        "completed_action_counts": summary.get("completed_action_counts")
        if isinstance(summary.get("completed_action_counts"), dict)
        else {},
        "lane_change_counts": summary.get("lane_change_counts")
        if isinstance(summary.get("lane_change_counts"), dict)
        else {},
        "active_watchlist_count": int(summary.get("active_watchlist_count") or 0),
        "research_memory_path": str(summary.get("research_memory_path") or ""),
        "research_memory_summary_path": str(summary.get("research_memory_summary_path") or ""),
        "top_improvers": summary.get("top_improvers") if isinstance(summary.get("top_improvers"), list) else [],
        "top_deteriorations": summary.get("top_deteriorations")
        if isinstance(summary.get("top_deteriorations"), list)
        else [],
        "top_memory_priority_candidates": summary.get("top_memory_priority_candidates")
        if isinstance(summary.get("top_memory_priority_candidates"), list)
        else [],
        "retryable_facts_blocker_count": int(summary.get("retryable_facts_blocker_count") or 0),
        "terminal_facts_blocker_count": int(summary.get("terminal_facts_blocker_count") or 0),
        "partial_usable_facts_count": int(summary.get("partial_usable_facts_count") or 0),
        "top_retryable_facts_blockers": summary.get("top_retryable_facts_blockers")
        if isinstance(summary.get("top_retryable_facts_blockers"), list)
        else [],
        "child_runs_with_facts_degradation": summary.get("child_runs_with_facts_degradation")
        if isinstance(summary.get("child_runs_with_facts_degradation"), list)
        else [],
        "economic_fail_count_vs_evidence_fail_count": summary.get("economic_fail_count_vs_evidence_fail_count")
        if isinstance(summary.get("economic_fail_count_vs_evidence_fail_count"), dict)
        else {},
        "item_blockers": summary.get("item_blockers") if isinstance(summary.get("item_blockers"), list) else [],
        "cursor_next_idx": int(state.get("cursor_next_idx") or 0),
        "active_item_id": str(state.get("active_item_id") or ""),
        "campaign_state_path": str(paths["state_path"]),
        "campaign_summary_path": str(paths["summary_path"]),
        "master_shortlist_json_path": str(paths["master_shortlist_json_path"]),
        "master_watchlist_state_path": str(paths["master_watchlist_state_path"]),
        "promotion_state_path": str(paths["promotion_state_path"]),
        "promotion_candidates_path": str(paths["promotion_candidates_path"]),
        "priority_lanes_path": str(paths["priority_lanes_path"]),
        "escalation_plan_path": str(paths["escalation_plan_path"]),
        "escalation_queue_path": str(paths["escalation_queue_path"]),
        "escalation_summary_path": str(paths["escalation_summary_path"]),
    }


def open_campaign(campaign_run_id: str) -> dict[str, Any]:
    paths = _campaign_paths(campaign_run_id)
    state = _safe_json(paths["state_path"])
    if not state:
        return {
            "status": "MISSING",
            "campaign_run_id": campaign_run_id,
            "campaign_state_path": str(paths["state_path"]),
            "campaign_summary_path": str(paths["summary_path"]),
        }
    summary = _safe_json(paths["summary_path"])
    master_shortlist = _safe_json(paths["master_shortlist_json_path"])
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "run_status": str(state.get("status") or CAMPAIGN_RUNNING),
        "policy": str(summary.get("policy") or _effective_campaign_policy(state)),
        "stop_reason_code": str(state.get("stop_reason_code") or ""),
        "stop_summary": str(state.get("stop_summary") or ""),
        "latest_validation_run_id": str(summary.get("latest_validation_run_id") or ""),
        "benchmark_pass_rate": float(summary.get("benchmark_pass_rate") or 0.0),
        "benchmark_mixed_rate": float(summary.get("benchmark_mixed_rate") or 0.0),
        "top_validation_failure_modes": summary.get("top_validation_failure_modes")
        if isinstance(summary.get("top_validation_failure_modes"), list)
        else [],
        "latest_filter_audit_run_id": str(summary.get("latest_filter_audit_run_id") or ""),
        "filter_evidence_stop_rate": float(summary.get("filter_evidence_stop_rate") or 0.0),
        "filter_economics_stop_rate": float(summary.get("filter_economics_stop_rate") or 0.0),
        "filter_queue_budget_stop_rate": float(summary.get("filter_queue_budget_stop_rate") or 0.0),
        "filter_memo_survivor_rate": float(summary.get("filter_memo_survivor_rate") or 0.0),
        "top_filter_failure_modes": summary.get("top_filter_failure_modes")
        if isinstance(summary.get("top_filter_failure_modes"), list)
        else [],
        "latest_borderline_audit_run_id": str(summary.get("latest_borderline_audit_run_id") or ""),
        "rescue_queue_count": int(summary.get("rescue_queue_count") or 0),
        "borderline_case_rate": float(summary.get("borderline_case_rate") or 0.0),
        "top_rescue_queue_names": summary.get("top_rescue_queue_names")
        if isinstance(summary.get("top_rescue_queue_names"), list)
        else [],
        "latest_second_look_run_id": str(summary.get("latest_second_look_run_id") or ""),
        "second_look_review_count": int(summary.get("second_look_review_count") or 0),
        "second_look_possible_false_negative_count": int(
            summary.get("second_look_possible_false_negative_count") or 0
        ),
        "second_look_evidence_review_count": int(summary.get("second_look_evidence_review_count") or 0),
        "second_look_queue_review_count": int(summary.get("second_look_queue_review_count") or 0),
        "top_second_look_reconsideration_names": summary.get("top_second_look_reconsideration_names")
        if isinstance(summary.get("top_second_look_reconsideration_names"), list)
        else [],
        "latest_filter_calibration_run_id": str(summary.get("latest_filter_calibration_run_id") or ""),
        "calibration_high_severity_count": int(summary.get("calibration_high_severity_count") or 0),
        "top_filter_calibration_recommendations": summary.get("top_filter_calibration_recommendations")
        if isinstance(summary.get("top_filter_calibration_recommendations"), list)
        else [],
        "item_counts": summary.get("item_counts") if isinstance(summary.get("item_counts"), dict) else {},
        "lane_counts": summary.get("lane_counts") if isinstance(summary.get("lane_counts"), dict) else {},
        "escalation_queue_count": int(summary.get("escalation_queue_count") or 0),
        "lane_to_action_counts": summary.get("lane_to_action_counts")
        if isinstance(summary.get("lane_to_action_counts"), dict)
        else {},
        "executed_queue_count": int(summary.get("executed_queue_count") or 0),
        "completed_action_counts": summary.get("completed_action_counts")
        if isinstance(summary.get("completed_action_counts"), dict)
        else {},
        "lane_change_counts": summary.get("lane_change_counts")
        if isinstance(summary.get("lane_change_counts"), dict)
        else {},
        "active_watchlist_count": int(summary.get("active_watchlist_count") or 0),
        "research_memory_path": str(summary.get("research_memory_path") or ""),
        "research_memory_summary_path": str(summary.get("research_memory_summary_path") or ""),
        "top_improvers": summary.get("top_improvers") if isinstance(summary.get("top_improvers"), list) else [],
        "top_deteriorations": summary.get("top_deteriorations")
        if isinstance(summary.get("top_deteriorations"), list)
        else [],
        "top_memory_priority_candidates": summary.get("top_memory_priority_candidates")
        if isinstance(summary.get("top_memory_priority_candidates"), list)
        else [],
        "retryable_facts_blocker_count": int(summary.get("retryable_facts_blocker_count") or 0),
        "terminal_facts_blocker_count": int(summary.get("terminal_facts_blocker_count") or 0),
        "partial_usable_facts_count": int(summary.get("partial_usable_facts_count") or 0),
        "top_retryable_facts_blockers": summary.get("top_retryable_facts_blockers")
        if isinstance(summary.get("top_retryable_facts_blockers"), list)
        else [],
        "child_runs_with_facts_degradation": summary.get("child_runs_with_facts_degradation")
        if isinstance(summary.get("child_runs_with_facts_degradation"), list)
        else [],
        "economic_fail_count_vs_evidence_fail_count": summary.get("economic_fail_count_vs_evidence_fail_count")
        if isinstance(summary.get("economic_fail_count_vs_evidence_fail_count"), dict)
        else {},
        "item_blockers": summary.get("item_blockers") if isinstance(summary.get("item_blockers"), list) else [],
        "top_10_master_shortlist": [
            {
                "ticker": str(row.get("ticker") or ""),
                "value_gate_status": str(row.get("value_gate_status") or UNKNOWN),
                "implied_return_base": row.get("implied_return_base", UNKNOWN),
                "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
                "mos_classification": str(row.get("mos_classification") or UNKNOWN),
                "mos_epv": row.get("mos_epv", UNKNOWN),
                "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
                "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
                "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
                "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
                "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
                "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or UNKNOWN),
                "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
                "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
                "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
                "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
                "value_type_primary": str(row.get("value_type_primary") or UNKNOWN),
                "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
                "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
                "oe_quality_total": row.get("oe_quality_total", UNKNOWN),
                "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
                "facts_blocker_class": str(row.get("facts_blocker_class") or "FACTS_OK"),
                "primary_fail_domain": str(row.get("primary_fail_domain") or "NONE"),
                "memory_priority_total": int(row.get("memory_priority_total") or 0),
            }
            for row in [row for row in (master_shortlist.get("rows") or []) if isinstance(row, dict)][:10]
        ],
        "top_10_promotion_candidates": summary.get("top_10_promotion_candidates")
        if isinstance(summary.get("top_10_promotion_candidates"), list)
        else [],
        "top_10_escalation_actions": summary.get("top_10_escalation_actions")
        if isinstance(summary.get("top_10_escalation_actions"), list)
        else [],
        "campaign_state_path": str(paths["state_path"]),
        "campaign_summary_path": str(paths["summary_path"]),
        "master_shortlist_json_path": str(paths["master_shortlist_json_path"]),
        "master_shortlist_md_path": str(paths["master_shortlist_md_path"]),
        "master_watchlist_state_path": str(paths["master_watchlist_state_path"]),
        "promotion_state_path": str(paths["promotion_state_path"]),
        "promotion_candidates_path": str(paths["promotion_candidates_path"]),
        "priority_lanes_path": str(paths["priority_lanes_path"]),
        "escalation_plan_path": str(paths["escalation_plan_path"]),
        "escalation_queue_path": str(paths["escalation_queue_path"]),
        "escalation_summary_path": str(paths["escalation_summary_path"]),
    }


def cancel_campaign(campaign_run_id: str, reason: str) -> dict[str, Any]:
    paths = _campaign_paths(campaign_run_id)
    state = _safe_json(paths["state_path"])
    if not state:
        return {
            "status": "MISSING",
            "campaign_run_id": campaign_run_id,
            "campaign_state_path": str(paths["state_path"]),
        }
    state["status"] = CAMPAIGN_CANCELLED
    state["stop_reason_code"] = STOP_CANCEL_REQUESTED
    state["stop_summary"] = str(reason or "Cancelled by operator.")
    active_item_id = str(state.get("active_item_id") or "")
    for item in [row for row in (state.get("items") or []) if isinstance(row, dict)]:
        if str(item.get("item_id") or "") != active_item_id:
            continue
        if str(item.get("status") or "").upper() == ITEM_RUNNING:
            item["status"] = ITEM_CANCELLED
            item["updated_at"] = utc_now_iso()
            _cancel_autopilot(str(item.get("child_universe_run_id") or ""), reason)
    _write_state_and_summary(paths, state)
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "run_status": CAMPAIGN_CANCELLED,
        "stop_reason_code": STOP_CANCEL_REQUESTED,
        "stop_summary": str(state.get("stop_summary") or ""),
        "campaign_state_path": str(paths["state_path"]),
        "campaign_summary_path": str(paths["summary_path"]),
    }
