from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso
from app.universe.promotion import (
    LANE_1_HIGH_PRIORITY,
    LANE_2_RESEARCH_QUEUE,
    LANE_3_MONITOR,
    LANE_4_DEPRIORITIZED,
)


UNKNOWN = "UNKNOWN"

ACTION_CLEAR_BLOCKERS = "CLEAR_BLOCKERS"
ACTION_REBUILD_DEPTH_RUN = "REBUILD_DEPTH_RUN"
ACTION_BUILD_REFRESHED_MEMO = "BUILD_REFRESHED_MEMO"
ACTION_ADD_TO_ACTIVE_WATCHLIST = "ADD_TO_ACTIVE_WATCHLIST"
ACTION_RECHECK_PROMOTION = "RECHECK_PROMOTION"
ACTION_SCHEDULE_LIGHT_REFRESH = "SCHEDULE_LIGHT_REFRESH"
ACTION_RETRY_COMPANYFACTS_HYDRATION = "RETRY_COMPANYFACTS_HYDRATION"
ACTION_RECHECK_FACTS_CACHE = "RECHECK_FACTS_CACHE"
ACTION_DEFER_UNTIL_EVIDENCE_REFRESH = "DEFER_UNTIL_EVIDENCE_REFRESH"
ACTION_NO_ACTION = "NO_ACTION"
ACTION_DEEPEN_FILING_DIFF = "DEEPEN_FILING_DIFF"
ACTION_RUN_TARGETED_PATTERN_SCAN = "RUN_TARGETED_PATTERN_SCAN"
ACTION_TRACK_VARIANT_PERCEPTION = "TRACK_VARIANT_PERCEPTION"
ACTION_RESOLVE_SYNTHESIS_GAP = "RESOLVE_SYNTHESIS_GAP"

STATUS_PLANNED = "PLANNED"

_LANE_ORDER = {
    LANE_1_HIGH_PRIORITY: 0,
    LANE_2_RESEARCH_QUEUE: 1,
    LANE_3_MONITOR: 2,
    LANE_4_DEPRIORITIZED: 3,
}
_ACTION_ORDER = {
    ACTION_RETRY_COMPANYFACTS_HYDRATION: 0,
    ACTION_RECHECK_FACTS_CACHE: 1,
    ACTION_CLEAR_BLOCKERS: 2,
    ACTION_DEEPEN_FILING_DIFF: 3,
    ACTION_RUN_TARGETED_PATTERN_SCAN: 4,
    ACTION_REBUILD_DEPTH_RUN: 5,
    ACTION_TRACK_VARIANT_PERCEPTION: 6,
    ACTION_RESOLVE_SYNTHESIS_GAP: 7,
    ACTION_BUILD_REFRESHED_MEMO: 8,
    ACTION_ADD_TO_ACTIVE_WATCHLIST: 9,
    ACTION_RECHECK_PROMOTION: 10,
    ACTION_SCHEDULE_LIGHT_REFRESH: 11,
    ACTION_DEFER_UNTIL_EVIDENCE_REFRESH: 12,
    ACTION_NO_ACTION: 13,
}


def _escalation_paths(campaign_run_id: str) -> dict[str, Path]:
    root = get_config().campaigns_dir / campaign_run_id
    return {
        "root": root,
        "escalation_plan_path": root / "escalation_plan.json",
        "escalation_queue_path": root / "escalation_queue.json",
        "escalation_summary_path": root / "escalation_summary.json",
        "campaign_state_path": root / "campaign_state.json",
        "promotion_state_path": root / "promotion_state.json",
        "priority_lanes_path": root / "priority_lanes.json",
        "master_watchlist_state_path": root / "master_watchlist_state.json",
    }


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


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _int_or_zero(value: Any) -> int:
    return int(value) if _is_num(value) else 0


def _desc_key(value: Any) -> tuple[int, float]:
    if _is_num(value):
        return (0, -float(value))
    return (1, 0.0)


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


def _action_rank(action_type: str) -> int:
    return _ACTION_ORDER.get(str(action_type or ""), len(_ACTION_ORDER))


def _variant_confidence_rank(value: Any) -> int:
    token = str(value or "NONE").upper()
    if token == "HIGH":
        return 0
    if token == "MEDIUM":
        return 1
    if token == "LOW":
        return 2
    return 3


def _lane_rank(lane: str) -> int:
    return _LANE_ORDER.get(str(lane or ""), len(_LANE_ORDER))


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


def _load_memory_lookup() -> dict[str, dict[str, Any]]:
    from app.universe.research_memory import load_research_memory

    memory = load_research_memory()
    return memory.get("tickers") if isinstance(memory.get("tickers"), dict) else {}


def _memory_priority_explanation(row: dict[str, Any]) -> str:
    reason_codes = {str(code) for code in (row.get("memory_priority_reason_codes") or []) if str(code).strip()}
    total = int(row.get("memory_priority_total") or 0)
    if total > 0 and any(
        code in {
            "REPEATED_SURVIVOR_2_PLUS",
            "REPEATED_SURVIVOR_3_PLUS",
            "GATE_UPGRADE",
            "LANE_UPGRADE",
            "IMPLIED_RETURN_UNKNOWN_TO_KNOWN",
            "MOS_EPV_UNKNOWN_TO_KNOWN",
            "BLOCKER_TERMINAL_TO_HYDRABLE",
            "BLOCKER_CLEARED",
            "ESCALATION_IMPROVED",
        }
        for code in reason_codes
    ):
        return "RECURRENT_IMPROVER"
    if total < 0 or any(
        code in {
            "GATE_DOWNGRADE",
            "LANE_DOWNGRADE",
            "IMPLIED_RETURN_KNOWN_TO_UNKNOWN",
            "RECURRING_UNCHANGED_BLOCKER_3_PLUS",
            "RECURRING_TERMINAL_BLOCKER",
            "ESCALATION_DETERIORATED",
        }
        for code in reason_codes
    ):
        return "REPEATED_STALLED_NAME"
    return "NEUTRAL_MEMORY"


def _oe_quality_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    blocker = _blocker_code(row)
    oe_quality_total = row.get("oe_quality_total", UNKNOWN)
    if (
        lane == LANE_2_RESEARCH_QUEUE
        and blocker not in {"", "NONE", UNKNOWN}
        and "TERMINAL_BLOCKER" not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and _is_num(oe_quality_total)
        and float(oe_quality_total) >= 8.0
    ):
        return "HIGH_OE_QUALITY_SUPPORT"
    return ""


def _intangible_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    blocker = _blocker_code(row)
    intangible_total = row.get("intangible_economics_total", UNKNOWN)
    if (
        lane == LANE_2_RESEARCH_QUEUE
        and blocker not in {"", "NONE", UNKNOWN}
        and "TERMINAL_BLOCKER" not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and _is_num(intangible_total)
        and float(intangible_total) >= 7.0
    ):
        return "STRONG_INTANGIBLE_ECONOMICS_SUPPORT"
    return ""


def _owner_value_capture_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    blocker = _blocker_code(row)
    owner_value_capture_score = row.get("owner_value_capture_score", UNKNOWN)
    if (
        lane == LANE_2_RESEARCH_QUEUE
        and blocker not in {"", "NONE", UNKNOWN}
        and "TERMINAL_BLOCKER" not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and _is_num(owner_value_capture_score)
        and float(owner_value_capture_score) >= 4.0
    ):
        return "STRONG_OWNER_VALUE_CAPTURE_SUPPORT"
    return ""


def _owner_value_capture_headwind_code(row: dict[str, Any]) -> str:
    reason_codes = {
        str(code)
        for code in (
            list(row.get("owner_value_capture_reason_codes") or [])
            + list(row.get("oe_quality_reason_codes") or [])
        )
        if str(code).strip()
    }
    owner_value_capture_score = row.get("owner_value_capture_score", UNKNOWN)
    if (
        "EXCESS_DILUTION" in reason_codes
        or "WEAK_PER_SHARE_CAPTURE" in reason_codes
        or (_is_num(owner_value_capture_score) and float(owner_value_capture_score) <= 1.0)
    ):
        return "WEAK_OWNER_VALUE_CAPTURE_HEADWIND"
    return ""


def _reinvestment_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    blocker = _blocker_code(row)
    reinvestment_class = str(row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN").upper()
    if (
        lane == LANE_2_RESEARCH_QUEUE
        and blocker not in {"", "NONE", UNKNOWN}
        and "TERMINAL_BLOCKER" not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and reinvestment_class == "HIGH_REINVESTMENT_EFFICIENCY"
    ):
        return "PRODUCTIVE_REINVESTMENT_SUPPORT"
    return ""


def _reinvestment_headwind_code(row: dict[str, Any]) -> str:
    reinvestment_class = str(row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN").upper()
    reasons = {
        str(code)
        for code in (
            list(row.get("reinvestment_efficiency_reason_codes") or [])
            + list(row.get("reinvestment_headwind_signals") or [])
        )
        if str(code).strip()
    }
    if reinvestment_class == "LOW_REINVESTMENT_EFFICIENCY":
        if (
            "CAPITAL_HUNGRY_GROWTH_HEADWIND" in reasons
            or "HIGH_CAPEX_BURDEN" in reasons
            or "CAPITAL_HUNGRY_GROWTH" in reasons
        ):
            return "CAPITAL_HUNGRY_GROWTH_HEADWIND"
        return "LOW_REINVESTMENT_EFFICIENCY_HEADWIND"
    if reinvestment_class == "REINVESTMENT_EFFICIENCY_UNKNOWN":
        return "REINVESTMENT_EFFICIENCY_UNKNOWN"
    return ""


def _accounting_quality_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    accounting_class = str(row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN").upper()
    if lane == LANE_2_RESEARCH_QUEUE and accounting_class == "HIGH_ACCOUNTING_QUALITY":
        return "HIGH_ACCOUNTING_QUALITY_SUPPORT"
    return ""


def _accounting_quality_headwind_code(row: dict[str, Any]) -> str:
    accounting_class = str(row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN").upper()
    reason_codes = {
        str(code)
        for code in (
            list(row.get("accounting_quality_reason_codes") or [])
            + list(row.get("cash_earnings_headwind_signals") or [])
        )
        if str(code).strip()
    }
    if accounting_class == "LOW_ACCOUNTING_QUALITY":
        if "ACCRUAL_HEAVY_EARNINGS" in reason_codes or "ACCRUAL_HEAVY_EARNINGS_HEADWIND" in reason_codes:
            return "ACCRUAL_HEAVY_EARNINGS_HEADWIND"
        if "WEAK_FCF_TO_EARNINGS_CONVERSION" in reason_codes or "WEAK_CASH_CONVERSION_HEADWIND" in reason_codes:
            return "WEAK_CASH_CONVERSION_HEADWIND"
        return "LOW_ACCOUNTING_QUALITY_HEADWIND"
    if accounting_class == "ACCOUNTING_QUALITY_UNKNOWN":
        return "ACCOUNTING_QUALITY_UNKNOWN"
    return ""


def _balance_sheet_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    stress_class = str(row.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN").upper()
    if lane == LANE_2_RESEARCH_QUEUE and stress_class == "LOW_BALANCE_SHEET_STRESS":
        return "LOW_BALANCE_SHEET_STRESS_SUPPORT"
    return ""


def _balance_sheet_headwind_code(row: dict[str, Any]) -> str:
    stress_class = str(row.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN").upper()
    refinancing_class = str(row.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN").upper()
    reason_codes = {
        str(code)
        for code in (
            list(row.get("balance_sheet_stress_reason_codes") or [])
            + list(row.get("refinancing_risk_reason_codes") or [])
            + list(row.get("balance_sheet_headwind_signals") or [])
        )
        if str(code).strip()
    }
    if refinancing_class == "HIGH_REFINANCING_RISK":
        return "HIGH_REFINANCING_RISK_HEADWIND"
    if stress_class == "HIGH_BALANCE_SHEET_STRESS":
        if "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE" in reason_codes:
            return "CAPITAL_STRUCTURE_DOMINATES_DOWNSIDE"
        return "HIGH_BALANCE_SHEET_STRESS_HEADWIND"
    if stress_class == "BALANCE_SHEET_STRESS_UNKNOWN":
        return "BALANCE_SHEET_STRESS_UNKNOWN"
    return ""


def _returns_persistence_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    returns_class = str(row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN").upper()
    if lane == LANE_2_RESEARCH_QUEUE and returns_class == "HIGH_RETURNS_PERSISTENCE":
        return "HIGH_RETURNS_PERSISTENCE_SUPPORT"
    return ""


def _returns_persistence_headwind_code(row: dict[str, Any]) -> str:
    returns_class = str(row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN").upper()
    reason_codes = {
        str(code)
        for code in (
            list(row.get("returns_persistence_reason_codes") or [])
            + list(row.get("returns_headwind_signals") or [])
        )
        if str(code).strip()
    }
    if returns_class == "LOW_RETURNS_PERSISTENCE":
        if "INCREMENTAL_RETURNS_DETERIORATING" in reason_codes:
            return "INCREMENTAL_RETURNS_DETERIORATION"
        return "LOW_RETURNS_PERSISTENCE_HEADWIND"
    if returns_class == "RETURNS_PERSISTENCE_UNKNOWN":
        return "RETURNS_PERSISTENCE_UNKNOWN"
    return ""


def _revenue_dependence_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    revenue_class = str(row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN").upper()
    if lane == LANE_2_RESEARCH_QUEUE and revenue_class == "LOW_REVENUE_DEPENDENCE_RISK":
        return "LOW_REVENUE_DEPENDENCE_SUPPORT"
    return ""


def _revenue_dependence_headwind_code(row: dict[str, Any]) -> str:
    revenue_class = str(row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN").upper()
    reason_codes = {
        str(code)
        for code in (
            list(row.get("revenue_dependence_risk_reason_codes") or [])
            + list(row.get("revenue_dependence_headwind_signals") or [])
        )
        if str(code).strip()
    }
    if revenue_class == "HIGH_REVENUE_DEPENDENCE_RISK":
        if "SINGLE_CUSTOMER_CONCENTRATION" in reason_codes or "TOP_CUSTOMER_DOMINANCE" in reason_codes:
            return "CUSTOMER_CONCENTRATION_HEADWIND"
        if "NARROW_CHANNEL_DEPENDENCE" in reason_codes:
            return "CHANNEL_DEPENDENCE_HEADWIND"
        return "HIGH_REVENUE_DEPENDENCE_HEADWIND"
    if revenue_class == "REVENUE_DEPENDENCE_UNKNOWN":
        return "REVENUE_DEPENDENCE_UNKNOWN"
    return ""


def _maintenance_capex_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    credibility_class = str(
        row.get("maintenance_capex_credibility_class") or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
    ).upper()
    asset_intensity_class = str(row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN").upper()
    if (
        lane == LANE_2_RESEARCH_QUEUE
        and credibility_class == "HIGH_MAINTENANCE_CAPEX_CREDIBILITY"
        and asset_intensity_class == "LOW_ASSET_INTENSITY"
    ):
        return "LOW_ASSET_INTENSITY_SUPPORT"
    return ""


def _maintenance_capex_headwind_code(row: dict[str, Any]) -> str:
    credibility_class = str(
        row.get("maintenance_capex_credibility_class") or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
    ).upper()
    asset_intensity_class = str(row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN").upper()
    if credibility_class == "LOW_MAINTENANCE_CAPEX_CREDIBILITY":
        return "LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND"
    if asset_intensity_class == "HIGH_ASSET_INTENSITY":
        return "HIGH_ASSET_INTENSITY_HEADWIND"
    if credibility_class == "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN":
        return "MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN"
    return ""


def _intrinsic_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    blocker = _blocker_code(row)
    support_type = str(row.get("downside_support_type") or UNKNOWN)
    mos_to_floor = row.get("mos_to_floor", UNKNOWN)
    if (
        lane == LANE_2_RESEARCH_QUEUE
        and blocker not in {"", "NONE", UNKNOWN}
        and "TERMINAL_BLOCKER" not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and (
            support_type in {"ASSET_SUPPORT", "EARNINGS_POWER_SUPPORT", "BALANCE_SHEET_SUPPORT"}
            or (_is_num(mos_to_floor) and float(mos_to_floor) >= 0.25)
        )
    ):
        return "REAL_DOWNSIDE_SUPPORT_PRESENT"
    return ""


def _mos_to_floor_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    blocker = _blocker_code(row)
    mos_to_floor = row.get("mos_to_floor", UNKNOWN)
    if (
        lane == LANE_2_RESEARCH_QUEUE
        and blocker not in {"", "NONE", UNKNOWN}
        and "TERMINAL_BLOCKER" not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and _is_num(mos_to_floor)
        and float(mos_to_floor) >= 0.25
    ):
        return "MOS_TO_FLOOR_ATTRACTIVE"
    return ""


def _intrinsic_headwind_code(row: dict[str, Any]) -> str:
    support_type = str(row.get("downside_support_type") or UNKNOWN)
    mos_classification = str(row.get("mos_classification") or UNKNOWN)
    mos_status = str(row.get("mos_assessment_status") or UNKNOWN).upper()
    if mos_status == "MOS_UNASSESSABLE":
        return ""
    if support_type in {"LIMITED_SUPPORT", "UNKNOWN_SUPPORT"} or mos_classification in {
        "NO_MARGIN_OF_SAFETY",
        "MOS_UNKNOWN",
    }:
        return "LIMITED_DOWNSIDE_SUPPORT"
    return ""


def _valuation_confidence_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    support_count = _int_or_zero(row.get("valuation_support_count"))
    confidence_class = str(row.get("valuation_confidence_class") or "CONFIDENCE_UNKNOWN").upper()
    fragility_status = str(row.get("valuation_fragility_status") or "FRAGILITY_UNKNOWN").upper()
    if (
        lane == LANE_2_RESEARCH_QUEUE
        and support_count >= 2
        and confidence_class in {"HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE"}
        and fragility_status in {"LOW_FRAGILITY", "MODERATE_FRAGILITY"}
    ):
        return "MULTI_SUPPORT_VALUE_CASE"
    return ""


def _valuation_fragility_support_code(row: dict[str, Any]) -> str:
    lane = str(row.get("priority_lane") or "")
    fragility_status = str(row.get("valuation_fragility_status") or "FRAGILITY_UNKNOWN").upper()
    if lane == LANE_2_RESEARCH_QUEUE and fragility_status == "LOW_FRAGILITY":
        return "LOW_FRAGILITY_UNDERWRITING"
    return ""


def _valuation_headwind_code(row: dict[str, Any]) -> str:
    convergence_status = str(row.get("valuation_convergence_status") or UNKNOWN).upper()
    support_count = _int_or_zero(row.get("valuation_support_count"))
    fragility_status = str(row.get("valuation_fragility_status") or "FRAGILITY_UNKNOWN").upper()
    if convergence_status == "WEAK_CONVERGENCE":
        return "SUPPORT_CONFLICT_HEADWIND"
    if support_count <= 1 or fragility_status == "HIGH_FRAGILITY":
        return "SINGLE_SUPPORT_FRAGILE"
    return ""


def _value_type_support_code(row: dict[str, Any]) -> str:
    token = str(row.get("value_type_primary") or UNKNOWN).upper()
    if token == "ASSET_BACKED_VALUE":
        return "ASSET_SUPPORT_CASE"
    if token == "EARNINGS_POWER_VALUE":
        return "EARNINGS_POWER_CASE"
    if token == "QUALITY_VALUE":
        return "QUALITY_VALUE_CASE"
    if token == "CYCLICAL_VALUE":
        return "CYCLICAL_VALUE_CASE"
    return ""


def _value_type_headwind_code(row: dict[str, Any]) -> str:
    if str(row.get("value_type_primary") or UNKNOWN).upper() == "FRAGILE_VALUE":
        return "FRAGILE_VALUE_HEADWIND"
    return ""


def _valuation_integrity_headwind_code(row: dict[str, Any]) -> str:
    token = str(row.get("valuation_integrity_class") or "INTEGRITY_UNKNOWN").upper()
    if token == "INTEGRITY_SUSPECT":
        return "INTEGRITY_SUSPECT_HEADWIND"
    if token == "INTEGRITY_WARNING":
        return "INTEGRITY_WARNING_HEADWIND"
    return ""


def _cyclical_risk_headwind_code(row: dict[str, Any]) -> str:
    token = str(row.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN").upper()
    if token == "PEAK_EARNINGS_RISK":
        return "PEAK_EARNINGS_CYCLICAL_HEADWIND"
    return ""


def _impairment_headwind_code(row: dict[str, Any]) -> str:
    cls = str(row.get("impairment_class_primary") or "").upper()
    if cls in {"CLEAR_IMPAIRMENT", "PROBABLE_IMPAIRMENT"}:
        return "IMPAIRMENT_HEADWIND"
    if cls == "STRUCTURALLY_WEAK_NOT_IMPAIRED":
        return "STRUCTURALLY_WEAK_HEADWIND"
    return ""


def _impairment_support_code(row: dict[str, Any]) -> str:
    cls = str(row.get("impairment_class_primary") or "").upper()
    if cls == "TEMPORARY_WEAKNESS":
        return "TEMPORARY_WEAKNESS_SUPPORT"
    if cls == "EVIDENCE_DEGRADED_NOT_ASSESSABLE":
        return "EVIDENCE_GAP_NOT_IMPAIRMENT"
    return ""


def _normalization_credibility_support_code(row: dict[str, Any]) -> str:
    """Return normalization credibility support code, or empty string."""
    cls = str(row.get("normalization_credibility_class") or "").upper()
    if cls == "HIGH_NORMALIZATION_CREDIBILITY":
        return "HIGH_NORMALIZATION_CREDIBILITY_SUPPORT"
    if cls == "MODERATE_NORMALIZATION_CREDIBILITY":
        return "MODERATE_NORMALIZATION_CREDIBILITY_SUPPORT"
    return ""


def _normalization_credibility_headwind_code(row: dict[str, Any]) -> str:
    """Return normalization credibility headwind code, or empty string."""
    cls = str(row.get("normalization_credibility_class") or "").upper()
    caution = str(row.get("primary_normalization_caution") or "").upper()
    if cls == "LOW_NORMALIZATION_CREDIBILITY":
        return "LOW_NORMALIZATION_CREDIBILITY_HEADWIND"
    if caution == "NORMALIZATION_BLOCKED_BY_EVIDENCE":
        return "NORMALIZATION_BLOCKED_BY_EVIDENCE"
    if caution == "NORMALIZATION_TOO_THEORETICAL":
        return "NORMALIZATION_TOO_THEORETICAL"
    return ""


def _readiness_support_code(row: dict[str, Any]) -> str:
    token = str(row.get("investment_readiness_class") or "READINESS_UNKNOWN").upper()
    if token == "INVESTABLE_NOW":
        return "READY_WITH_REAL_SUPPORT"
    if token == "RESEARCH_WORTHY_NOT_READY":
        return "BLOCKED_BUT_RESEARCH_WORTHY"
    if token == "WATCH_ONLY":
        return "WATCH_WITH_THIN_EDGE"
    if token == "NOT_INVESTABLE":
        return "NOT_INVESTABLE_STRUCTURAL"
    if token == "READINESS_UNKNOWN":
        return "READINESS_UNKNOWN_EVIDENCE_GAP"
    return ""


def _mos_guardrail_support_code(row: dict[str, Any]) -> str:
    mos_status = str(row.get("mos_assessment_status") or UNKNOWN).upper()
    if mos_status == "MOS_UNASSESSABLE":
        return "MOS_UNASSESSABLE_EVIDENCE_GAP"
    return ""


def _mos_guardrail_headwind_code(row: dict[str, Any]) -> str:
    mos_status = str(row.get("mos_assessment_status") or UNKNOWN).upper()
    sufficiency = str(row.get("evidence_sufficiency_class") or UNKNOWN).upper()
    if mos_status == "MOS_CONFIRMED_ABSENT":
        return "NO_MOS_CONFIRMED"
    if mos_status == "MOS_UNASSESSABLE" and sufficiency in {
        "PARTIAL_FOR_MOS",
        "INSUFFICIENT_FOR_MOS",
    }:
        return "EVIDENCE_BLOCKED_VALUE_CASE"
    return ""


def _source_universe_run_ids(row: dict[str, Any]) -> list[str]:
    run_ids: list[str] = []
    seen: set[str] = set()
    for source in [value for value in (row.get("source_runs") or []) if isinstance(value, dict)]:
        token = str(source.get("universe_run_id") or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        run_ids.append(token)
    return run_ids


def _source_rollup_paths(run_ids: list[str]) -> list[str]:
    cfg = get_config()
    paths: list[str] = []
    for run_id in run_ids:
        batch_run_id = f"{run_id}_depth_batch"
        path = cfg.outputs_dir / "universe" / run_id / "depth_batches" / batch_run_id / "global_rollup.json"
        paths.append(str(path))
    return paths


def _first_source_universe_run_id(row: dict[str, Any]) -> str:
    run_ids = _source_universe_run_ids(row)
    return run_ids[0] if run_ids else ""


def _blocker_code(row: dict[str, Any]) -> str:
    for value in [row.get("latest_primary_blocker"), row.get("primary_blocker")]:
        token = str(value or "").strip().upper()
        if token:
            return token
    return UNKNOWN


def _effective_implied_return(row: dict[str, Any]) -> Any:
    for value in [row.get("latest_implied_return_base"), row.get("implied_return_base")]:
        if _is_num(value):
            return float(value)
    return UNKNOWN


def _latest_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "implied_return_base": _effective_implied_return(row),
        "mos_epv": row.get("mos_epv", UNKNOWN),
        "yield_metric_used": str(row.get("yield_metric_used") or UNKNOWN),
        "value_gate_status": str(row.get("latest_value_gate_status") or row.get("value_gate_status") or UNKNOWN),
    }


def _history_blocker_unchanged(row: dict[str, Any]) -> bool:
    history = [value for value in (row.get("history") or []) if isinstance(value, dict)]
    if len(history) < 2 or not str(row.get("memo_path") or "").strip():
        return False
    latest_blocker = _blocker_code(row)
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    last_two = history[-2:]
    return all(
        str(item.get("primary_blocker") or UNKNOWN).upper() == latest_blocker
        and str(item.get("value_gate_status") or UNKNOWN).upper() == latest_gate
        for item in last_two
    )


def _clear_blocker_reason(blocker_code: str) -> str:
    blocker = str(blocker_code or UNKNOWN).upper()
    if blocker == "PRICE_UNKNOWN":
        return "HYDRATE_PRICE_SNAPSHOT"
    if blocker in {"MISSING_EV", "MISSING_CASH", "MISSING_DEBT", "MISSING_BOTH", "MISSING_NET_DEBT", "NO_FACTS"}:
        return "HYDRATE_FINANCIAL_FACTS"
    if blocker in {"MISSING_SHARES", "MISSING_FCF"}:
        return "HYDRATE_SHARES_FCF"
    if blocker in {
        "MISSING_GD_INPUTS",
        "MISSING_CURRENT_ASSETS",
        "MISSING_TOTAL_LIABILITIES",
        "MISSING_EARNINGS_STREAM",
    }:
        return "HYDRATE_GRAHAM_DODD_INPUTS"
    return "HYDRATE_TARGETED_INPUTS"


def _facts_action_type(row: dict[str, Any]) -> str:
    blocker_class = str(row.get("facts_blocker_class") or "FACTS_OK")
    if not bool(row.get("facts_retry_recommended")) or blocker_class == "FACTS_OK":
        return ""
    if blocker_class == "FACTS_NO_CACHE_OFFLINE":
        return ACTION_RECHECK_FACTS_CACHE
    if bool(row.get("facts_blocker_partial_usable")):
        return ACTION_DEFER_UNTIL_EVIDENCE_REFRESH
    if bool(row.get("facts_blocker_retryable")):
        return ACTION_RETRY_COMPANYFACTS_HYDRATION
    return ""


def _facts_action_reason(row: dict[str, Any]) -> str:
    blocker_class = str(row.get("facts_blocker_class") or "FACTS_OK")
    if blocker_class == "FACTS_PARTIAL_COVERAGE":
        return "FACTS_PARTIAL_USABLE_RECHECK"
    if blocker_class in {"FACTS_RETRYABLE_TIMEOUT", "FACTS_RETRYABLE_NETWORK_FAILURE", "FACTS_RETRYABLE_RATE_LIMIT", "FACTS_NO_CACHE_OFFLINE"}:
        return "FACTS_BLOCKER_RETRY_PATH"
    if bool(row.get("facts_blocker_terminal")) and blocker_class != "FACTS_OK":
        return "TERMINAL_FACTS_HEADWIND"
    return ""


def _ensure_l4_run_context(run_id: str, context: dict[str, Any]) -> dict[str, Any]:
    run_id_norm = str(run_id or "").strip()
    run_cache = context.get("run_cache") if isinstance(context.get("run_cache"), dict) else {}
    if run_id_norm in run_cache and isinstance(run_cache[run_id_norm], dict):
        return run_cache[run_id_norm]

    cfg = context.get("cfg") if hasattr(context.get("cfg"), "outputs_dir") else get_config()
    universe_root = cfg.outputs_dir / "universe" / run_id_norm
    autopilot_state = _safe_json(universe_root / "autopilot" / "autopilot_state.json")
    as_of_date = str(autopilot_state.get("as_of_date") or context.get("as_of_date") or "").strip()
    run_ctx = {
        "run_id": run_id_norm,
        "as_of_date": as_of_date,
        "depth": str(autopilot_state.get("depth") or "").strip().lower(),
        "variant_dir": universe_root / "variant_perceptions",
        "variant_cache": {},
        "filing_diff_dir": universe_root / "filing_diffs",
        "filing_diff_cache": {},
        "pattern_report_path": universe_root / "pattern_scan" / "pattern_scan_report.json",
        "pattern_report_available": None,
        "synthesis_cache": {},
    }
    run_cache[run_id_norm] = run_ctx
    context["run_cache"] = run_cache
    return run_ctx


def _load_variant_payload_for_ticker(ticker: str, run_ctx: dict[str, Any], cfg: Any) -> dict[str, Any]:
    cache = run_ctx.get("variant_cache") if isinstance(run_ctx.get("variant_cache"), dict) else {}
    ticker_norm = str(ticker or "").strip().upper()
    if ticker_norm in cache:
        return cache[ticker_norm] if isinstance(cache[ticker_norm], dict) else {}

    candidates: list[Path] = []
    as_of_date = str(run_ctx.get("as_of_date") or "").strip()
    variant_dir = run_ctx.get("variant_dir") if isinstance(run_ctx.get("variant_dir"), Path) else None
    if isinstance(variant_dir, Path) and variant_dir.exists():
        if as_of_date:
            candidates.append(variant_dir / f"{ticker_norm}_{as_of_date}.json")
        candidates.extend(
            sorted(
                variant_dir.glob(f"{ticker_norm}_*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
    if as_of_date:
        candidates.append(cfg.outputs_dir / "variant_perceptions" / f"{ticker_norm}_{as_of_date}.json")

    payload: dict[str, Any] = {}
    for path in candidates:
        payload = _safe_json(path)
        if payload:
            break
    cache[ticker_norm] = payload
    run_ctx["variant_cache"] = cache
    return payload


def _load_filing_diff_payload_for_ticker(ticker: str, run_ctx: dict[str, Any], cfg: Any) -> dict[str, Any]:
    cache = run_ctx.get("filing_diff_cache") if isinstance(run_ctx.get("filing_diff_cache"), dict) else {}
    ticker_norm = str(ticker or "").strip().upper()
    if ticker_norm in cache:
        return cache[ticker_norm] if isinstance(cache[ticker_norm], dict) else {}

    candidates: list[Path] = []
    filing_diff_dir = run_ctx.get("filing_diff_dir") if isinstance(run_ctx.get("filing_diff_dir"), Path) else None
    if isinstance(filing_diff_dir, Path):
        candidates.append(filing_diff_dir / f"{ticker_norm}.json")
    candidates.append(cfg.outputs_dir / "diffs" / f"{ticker_norm}_{str(run_ctx.get('run_id') or '').strip()}_diff.json")

    payload: dict[str, Any] = {}
    for path in candidates:
        payload = _safe_json(path)
        if payload:
            break
    cache[ticker_norm] = payload
    run_ctx["filing_diff_cache"] = cache
    return payload


def _pattern_report_available_for_run(run_ctx: dict[str, Any], cfg: Any) -> bool:
    if isinstance(run_ctx.get("pattern_report_available"), bool):
        return bool(run_ctx.get("pattern_report_available"))

    report_path = run_ctx.get("pattern_report_path") if isinstance(run_ctx.get("pattern_report_path"), Path) else None
    payload = _safe_json(report_path) if isinstance(report_path, Path) and report_path.exists() else {}
    if not payload:
        payload = _safe_json(cfg.outputs_dir / "patterns" / f"{str(run_ctx.get('run_id') or '').strip()}_pattern_scan.json")
    available = bool(payload)
    run_ctx["pattern_report_available"] = available
    return available


def _load_synthesis_payload_for_ticker(ticker: str, run_ctx: dict[str, Any], cfg: Any) -> dict[str, Any]:
    cache = run_ctx.get("synthesis_cache") if isinstance(run_ctx.get("synthesis_cache"), dict) else {}
    ticker_norm = str(ticker or "").strip().upper()
    if ticker_norm in cache:
        return cache[ticker_norm] if isinstance(cache[ticker_norm], dict) else {}

    candidates: list[Path] = []
    as_of_date = str(run_ctx.get("as_of_date") or "").strip()
    run_id = str(run_ctx.get("run_id") or "").strip()
    if as_of_date:
        candidates.append(cfg.synthesis_dir / f"{ticker_norm}_{as_of_date}_run_{run_id}.json")
    candidates.extend(
        sorted(
            cfg.synthesis_dir.glob(f"{ticker_norm}_*_run_{run_id}.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )

    payload: dict[str, Any] = {}
    source_path = ""
    for path in candidates:
        payload = _safe_json(path)
        if payload:
            source_path = str(path)
            break
    if payload:
        payload = {**payload, "_source_path": source_path}
    cache[ticker_norm] = payload
    run_ctx["synthesis_cache"] = cache
    return payload


def _l4_signal_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_perception_count": int(row.get("variant_perception_count") or 0),
        "variant_perception_max_confidence": str(row.get("variant_perception_max_confidence") or "NONE").upper(),
        "variant_perception_direction": str(row.get("variant_perception_direction") or "NONE").upper(),
        "variant_signal_source_count": int(row.get("variant_signal_source_count") or 0),
        "tech_category": str(row.get("tech_category") or "TRADITIONAL_OPERATING").upper(),
        "tech_valuation_divergence": row.get("tech_valuation_divergence", UNKNOWN),
        "filing_diff_high_materiality_count": int(row.get("filing_diff_high_materiality_count") or 0),
        "pattern_hit_count": int(row.get("pattern_hit_count") or 0),
        "pattern_confirmed_count": int(row.get("pattern_confirmed_count") or 0),
    }


def _l4_priority_boost(row: dict[str, Any]) -> int:
    boost = 0
    confidence = str(row.get("variant_perception_max_confidence") or "NONE").upper()
    direction = str(row.get("variant_perception_direction") or "NONE").upper()
    strength_flags = {str(flag).upper() for flag in (row.get("strength_flags") or []) if str(flag).strip()}
    risk_flags = {str(flag).upper() for flag in (row.get("risk_flags") or []) if str(flag).strip()}

    if confidence == "HIGH" and direction == "UNDERVALUED":
        boost += 2
    elif confidence == "MEDIUM" and direction == "UNDERVALUED":
        boost += 1
    elif confidence == "HIGH" and direction == "OVERVALUED":
        boost -= 2
    if "CONVERGENT_SIGNAL" in strength_flags:
        boost += 1
    if "OVERVALUATION_BLOCK_HIGH" in risk_flags:
        boost = min(boost, -2)
    return boost


def _strong_valuation_signal(row: dict[str, Any]) -> bool:
    if str(row.get("variant_perception_direction") or "NONE").upper() == "UNDERVALUED":
        return True
    if str(row.get("priority_lane") or "") == LANE_1_HIGH_PRIORITY:
        return True

    implied_return = _effective_implied_return(row)
    if _is_num(implied_return) and float(implied_return) >= 0.20:
        return True

    mos_to_floor = row.get("mos_to_floor", UNKNOWN)
    if _is_num(mos_to_floor) and float(mos_to_floor) >= 0.25:
        return True

    support_codes = {str(code).upper() for code in (row.get("priority_support_codes") or []) if str(code).strip()}
    return bool(
        support_codes.intersection(
            {
                "MULTI_SUPPORT_VALUE_CASE",
                "REAL_DOWNSIDE_SUPPORT_PRESENT",
                "MOS_TO_FLOOR_ATTRACTIVE",
                "HIGH_OE_QUALITY_SUPPORT",
                "STRONG_INTANGIBLE_ECONOMICS_SUPPORT",
                "STRONG_OWNER_VALUE_CAPTURE_SUPPORT",
            }
        )
    )


def _fundamentals_meet_pattern_scan_threshold(row: dict[str, Any]) -> bool:
    lane = str(row.get("priority_lane") or "")
    if lane not in {LANE_1_HIGH_PRIORITY, LANE_2_RESEARCH_QUEUE, LANE_3_MONITOR}:
        return False
    if "TERMINAL_BLOCKER" in {str(flag).upper() for flag in (row.get("risk_flags") or []) if str(flag).strip()}:
        return False
    readiness = str(row.get("investment_readiness_class") or "READINESS_UNKNOWN").upper()
    if readiness in {"INVESTABLE_NOW", "RESEARCH_WORTHY_NOT_READY", "WATCH_ONLY"}:
        return True
    evidence = str(row.get("evidence_sufficiency_class") or UNKNOWN).upper()
    return evidence not in {"INSUFFICIENT_FOR_MOS"}


def _best_variant_payload_for_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    cfg = context.get("cfg") if hasattr(context.get("cfg"), "outputs_dir") else get_config()
    best_payload: dict[str, Any] = {}
    best_score = (-1, -(10**9))
    ticker = str(row.get("ticker") or "").strip().upper()
    for run_id in _source_universe_run_ids(row):
        run_ctx = _ensure_l4_run_context(run_id, context)
        payload = _load_variant_payload_for_ticker(ticker, run_ctx, cfg)
        if not payload:
            continue
        perceptions = [item for item in (payload.get("perceptions") or []) if isinstance(item, dict)]
        confidences = [_variant_confidence_rank(item.get("confidence")) for item in perceptions]
        score = (len(perceptions), -(min(confidences) if confidences else 10**9))
        if score > best_score:
            best_score = score
            best_payload = payload
    return best_payload


def _filing_diff_exists_for_row(row: dict[str, Any], context: dict[str, Any]) -> bool:
    cfg = context.get("cfg") if hasattr(context.get("cfg"), "outputs_dir") else get_config()
    ticker = str(row.get("ticker") or "").strip().upper()
    for run_id in _source_universe_run_ids(row):
        run_ctx = _ensure_l4_run_context(run_id, context)
        if _load_filing_diff_payload_for_ticker(ticker, run_ctx, cfg):
            return True
    return False


def _pattern_scan_exists_for_row(row: dict[str, Any], context: dict[str, Any]) -> bool:
    cfg = context.get("cfg") if hasattr(context.get("cfg"), "outputs_dir") else get_config()
    for run_id in _source_universe_run_ids(row):
        run_ctx = _ensure_l4_run_context(run_id, context)
        if _pattern_report_available_for_run(run_ctx, cfg):
            return True
    return False


def _is_structural_gap(gap_description: str) -> bool:
    token = str(gap_description or "").strip().lower()
    if not token:
        return True
    return any(
        phrase in token
        for phrase in (
            "private company",
            "company is private",
            "not public",
            "provider disabled",
            "llm provider disabled",
            "unsupported",
        )
    )


def _recommended_gap_resolution(payload: dict[str, Any], gap_description: str) -> str:
    next_actions = payload.get("next_actions")
    if isinstance(next_actions, list):
        for action in next_actions:
            if not isinstance(action, dict):
                continue
            hint = str(action.get("query_or_url_hint") or "").strip()
            why = str(action.get("why") or "").strip()
            if hint and why:
                return f"{hint} - {why}"
            if hint:
                return hint
            if why:
                return why
    recommended = payload.get("recommended_next_actions")
    if isinstance(recommended, list):
        for action in recommended:
            token = str(action or "").strip()
            if token:
                return token
    gap_lower = str(gap_description or "").lower()
    if "debt covenant" in gap_lower or "covenant" in gap_lower:
        return "Review debt footnotes and covenant disclosures, then rerun synthesis."
    if "customer concentration" in gap_lower:
        return "Inspect concentration disclosures and filing diffs, then rerun synthesis."
    if "pattern" in gap_lower:
        return "Run the peer pattern scan and rerun synthesis with the updated evidence."
    if "filing diff" in gap_lower or "risk factor" in gap_lower:
        return "Run a targeted filing diff and rerun synthesis with the new change set."
    return "Gather the missing evidence and rerun synthesis for the ticker."


def _addressable_synthesis_gap_for_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any] | None:
    cfg = context.get("cfg") if hasattr(context.get("cfg"), "outputs_dir") else get_config()
    ticker = str(row.get("ticker") or "").strip().upper()
    for run_id in _source_universe_run_ids(row):
        run_ctx = _ensure_l4_run_context(run_id, context)
        payload = _load_synthesis_payload_for_ticker(ticker, run_ctx, cfg)
        if not payload:
            continue
        gaps = [str(value).strip() for value in (payload.get("evidence_gaps") or []) if str(value).strip()]
        for gap in gaps:
            if _is_structural_gap(gap):
                continue
            return {
                "gap_description": gap,
                "recommended_action": _recommended_gap_resolution(payload, gap),
                "requires_human_review": True,
                "synthesis_path": str(payload.get("_source_path") or ""),
                "source_run_id": str(run_ctx.get("run_id") or ""),
            }
    return None


def _planned_l4_actions(row: dict[str, Any], *, context: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not _source_universe_run_ids(row):
        return actions

    ticker = str(row.get("ticker") or "").strip().upper()
    lane = str(row.get("priority_lane") or "")
    risk_flags = {str(flag).upper() for flag in (row.get("risk_flags") or []) if str(flag).strip()}
    terminal = "TERMINAL_BLOCKER" in risk_flags
    variant_payload = _best_variant_payload_for_row(row, context)
    perceptions = [item for item in (variant_payload.get("perceptions") or []) if isinstance(item, dict)]
    filing_diff_exists = _filing_diff_exists_for_row(row, context)
    pattern_scan_exists = _pattern_scan_exists_for_row(row, context)
    l4_enabled_runs = any(
        str(_ensure_l4_run_context(run_id, context).get("depth") or "").lower() in {"full", "alpha-only"}
        for run_id in _source_universe_run_ids(row)
    )

    variant_direction = str(row.get("variant_perception_direction") or "NONE").upper()
    missing_sources = {
        str(source).upper()
        for source in (
            ((variant_payload.get("data_quality") or {}).get("missing_sources") or [])
            if isinstance(variant_payload.get("data_quality"), dict)
            else []
        )
        if str(source).strip()
    }
    variant_missing_diff = (
        not variant_payload
        or not isinstance(variant_payload.get("data_quality"), dict)
        or "FILING_DIFF" in missing_sources
    )
    if (
        not terminal
        and lane != LANE_4_DEPRIORITIZED
        and bool(variant_payload)
        and (variant_direction == "UNDERVALUED" or _strong_valuation_signal(row))
        and int(row.get("filing_diff_high_materiality_count") or 0) == 0
        and not filing_diff_exists
        and variant_missing_diff
    ):
        thesis_direction = variant_direction.lower() if variant_direction in {"UNDERVALUED", "OVERVALUED"} else "undervalued"
        actions.append(
            {
                "action_type": ACTION_DEEPEN_FILING_DIFF,
                "action_reason": (
                    "Strong valuation signal but no filing diff data - filing-level changes may "
                    f"confirm or contradict the {thesis_direction} thesis"
                ),
            }
        )

    if (
        not terminal
        and l4_enabled_runs
        and _fundamentals_meet_pattern_scan_threshold(row)
        and int(row.get("pattern_hit_count") or 0) == 0
        and not pattern_scan_exists
    ):
        actions.append(
            {
                "action_type": ACTION_RUN_TARGETED_PATTERN_SCAN,
                "action_reason": (
                    "No pattern scan results - cross-sectional analysis against sector peers may reveal predictive patterns"
                ),
            }
        )

    if int(row.get("variant_perception_count") or 0) >= 1 and str(
        row.get("variant_perception_max_confidence") or "NONE"
    ).upper() in {"HIGH", "MEDIUM"}:
        ranked_perceptions = sorted(
            [
                item
                for item in perceptions
                if str(item.get("testable_prediction") or "").strip()
                and str(item.get("time_horizon") or "").strip()
            ],
            key=lambda item: (
                _variant_confidence_rank(item.get("confidence")),
                str(item.get("perception_id") or ""),
            ),
        )
        if ranked_perceptions:
            best = ranked_perceptions[0]
            confidence = str(best.get("confidence") or "LOW").upper()
            direction = str(best.get("direction") or "UNDERVALUED").lower()
            time_horizon = str(best.get("time_horizon") or "MEDIUM").lower()
            actions.append(
                {
                    "action_type": ACTION_TRACK_VARIANT_PERCEPTION,
                    "action_reason": (
                        f"{confidence} confidence {direction} perception with testable prediction - "
                        f"registering for outcome tracking over {time_horizon} horizon"
                    ),
                    "action_metadata": {
                        "perception_ids": [
                            str(item.get("perception_id") or "")
                            for item in ranked_perceptions
                            if str(item.get("perception_id") or "").strip()
                        ],
                    },
                }
            )

    synthesis_gap = _addressable_synthesis_gap_for_row(row, context)
    if synthesis_gap:
        actions.append(
            {
                "action_type": ACTION_RESOLVE_SYNTHESIS_GAP,
                "action_reason": (
                    f"Synthesis identified evidence gap: {synthesis_gap['gap_description']} - "
                    "resolving may strengthen or invalidate the thesis"
                ),
                "action_metadata": synthesis_gap,
            }
        )

    return actions


def _command_for_action(
    *,
    campaign_run_id: str,
    action_type: str,
    row: dict[str, Any],
    campaign_config: dict[str, Any],
) -> str:
    ticker = str(row.get("ticker") or "").lower()
    as_of_date = str(campaign_config.get("as_of_date") or "").strip()
    source_campaign_file = str(campaign_config.get("source_campaign_file") or "").strip()
    top_n = max(1, int(campaign_config.get("top_n") or 10))
    policy = str(campaign_config.get("policy") or "value_first")
    first_run_id = _first_source_universe_run_id(row)
    batch_run_id = f"{first_run_id}_depth_batch" if first_run_id else ""

    if action_type == ACTION_CLEAR_BLOCKERS:
        return (
            "python -m app.cli universe-autopilot "
            f"--run-id {campaign_run_id}__lane2__{ticker} "
            f"--as-of {as_of_date} --max-runs 2 --top-n {top_n} --policy {policy} --resume"
        )
    if action_type == ACTION_DEEPEN_FILING_DIFF and first_run_id:
        return (
            "python -m app.cli filing-diff "
            f"--ticker {str(row.get('ticker') or '').upper()} "
            f"--run-id {first_run_id} --years-back 5"
        )
    if action_type == ACTION_RUN_TARGETED_PATTERN_SCAN and first_run_id:
        return f"python -m app.cli pattern-scan --run-id {first_run_id}"
    if action_type == ACTION_RETRY_COMPANYFACTS_HYDRATION:
        return (
            "python -m app.cli companyfacts-fetch "
            f"--tickers {str(row.get('ticker') or '').upper()} "
            f"--as-of {as_of_date} "
            f"--run-id {campaign_run_id}__facts_retry__{ticker}"
        )
    if action_type == ACTION_RECHECK_FACTS_CACHE:
        return (
            "python -m app.cli companyfacts-fetch "
            f"--tickers {str(row.get('ticker') or '').upper()} "
            f"--as-of {as_of_date} "
            f"--run-id {campaign_run_id}__facts_cache__{ticker}"
        )
    if action_type == ACTION_REBUILD_DEPTH_RUN:
        return (
            "python -m app.cli universe-autopilot "
            f"--run-id {campaign_run_id}__lane1__{ticker} "
            f"--as-of {as_of_date} --max-runs 5 --top-n {top_n} --policy {policy} --resume --force"
        )
    if action_type == ACTION_BUILD_REFRESHED_MEMO and first_run_id:
        return (
            "python -m app.cli universe-memo-pack "
            f"--universe-run-id {first_run_id} --batch-run-id {batch_run_id} "
            f"--top-n {top_n} --policy {policy}"
        )
    if action_type == ACTION_TRACK_VARIANT_PERCEPTION and first_run_id:
        return (
            "python -m app.cli variant-perception "
            f"--run-id {first_run_id} --ticker {str(row.get('ticker') or '').upper()}"
        )
    if action_type == ACTION_RESOLVE_SYNTHESIS_GAP:
        return f"python -m app.cli universe-campaign-escalation-open --campaign-run-id {campaign_run_id}"
    if action_type == ACTION_ADD_TO_ACTIVE_WATCHLIST and first_run_id:
        return f"python -m app.cli universe-watchlist-state-open --run-id {first_run_id}"
    if action_type == ACTION_RECHECK_PROMOTION:
        return f"python -m app.cli universe-campaign-promotion-open --campaign-run-id {campaign_run_id}"
    if action_type == ACTION_SCHEDULE_LIGHT_REFRESH and source_campaign_file:
        return (
            "python -m app.cli universe-campaign "
            f"--campaign-file {source_campaign_file} "
            f"--campaign-run-id {campaign_run_id}__monitor_refresh --resume --max-items 1"
        )
    if action_type == ACTION_DEFER_UNTIL_EVIDENCE_REFRESH:
        return f"python -m app.cli universe-scout-open --run-id {first_run_id}" if first_run_id else ""
    if action_type == ACTION_NO_ACTION:
        return f"python -m app.cli universe-campaign-open --campaign-run-id {campaign_run_id}"
    return ""


def _queue_entry(
    *,
    queue_rank: int,
    campaign_run_id: str,
    row: dict[str, Any],
    action_type: str,
    action_reason: str,
    campaign_config: dict[str, Any],
    action_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_ids = _source_universe_run_ids(row)
    l4_summary = _l4_signal_summary(row)
    return {
        "queue_rank": int(queue_rank),
        "ticker": str(row.get("ticker") or ""),
        "priority_lane": str(row.get("priority_lane") or UNKNOWN),
        "action_type": action_type,
        "action_reason": action_reason,
        "action_metadata": dict(action_metadata or {}),
        "blocking_reason_code": _blocker_code(row),
        "facts_blocker_class": str(row.get("facts_blocker_class") or "FACTS_OK"),
        "facts_blocker_retryable": bool(row.get("facts_blocker_retryable", False)),
        "facts_blocker_terminal": bool(row.get("facts_blocker_terminal", False)),
        "facts_blocker_partial_usable": bool(row.get("facts_blocker_partial_usable", False)),
        "facts_missing_key_inputs": [
            str(value)
            for value in (row.get("facts_missing_key_inputs") or [])
            if str(value).strip()
        ],
        "facts_retry_recommended": bool(row.get("facts_retry_recommended", False)),
        "facts_blocker_reason_codes": [
            str(code)
            for code in (row.get("facts_blocker_reason_codes") or [])
            if str(code).strip()
        ],
        "facts_recommended_action": str(row.get("facts_recommended_action") or "NONE"),
        "primary_fail_domain": str(row.get("primary_fail_domain") or "NONE"),
        "recommended_command": _command_for_action(
            campaign_run_id=campaign_run_id,
            action_type=action_type,
            row=row,
            campaign_config=campaign_config,
        ),
        "l4_signal_summary": l4_summary,
        "l4_priority_boost": _l4_priority_boost(row),
        "source_campaign_run_id": campaign_run_id,
        "source_universe_run_ids": run_ids,
        "latest_metrics": _latest_metrics(row),
        "artifacts_to_read": {
            "memo_path": str(row.get("memo_path") or ""),
            "watchlist_state_path": str(_escalation_paths(campaign_run_id)["master_watchlist_state_path"]),
            "source_rollup_paths": _source_rollup_paths(run_ids),
        },
        "status": STATUS_PLANNED,
        "appearances_count": int(row.get("appearances_count") or 0),
        "memory_priority_total": int(row.get("memory_priority_total") or 0),
        "memory_priority_reason_codes": [
            str(code)
            for code in (row.get("memory_priority_reason_codes") or [])
            if str(code).strip()
        ],
        "memory_priority_source_campaign_run_ids": [
            str(run_id)
            for run_id in (row.get("memory_priority_source_campaign_run_ids") or [])
            if str(run_id).strip()
        ],
        "memory_priority_explanation": _memory_priority_explanation(row),
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
        "asset_intensity_class": str(
            row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
        ),
        "asset_intensity_reason_codes": [
            str(code)
            for code in (row.get("asset_intensity_reason_codes") or [])
            if str(code).strip()
        ],
        "maintenance_capex_credibility_class": str(
            row.get("maintenance_capex_credibility_class")
            or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
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
        "accounting_quality_class": str(
            row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
        ),
        "accounting_quality_reason_codes": [
            str(code)
            for code in (row.get("accounting_quality_reason_codes") or [])
            if str(code).strip()
        ],
        "cash_earnings_support_signals": [
            str(code)
            for code in (row.get("cash_earnings_support_signals") or [])
            if str(code).strip()
        ],
        "cash_earnings_headwind_signals": [
            str(code)
            for code in (row.get("cash_earnings_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_accounting_caution": str(
            row.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
        ),
        "cash_earnings_discipline_summary": str(
            row.get("cash_earnings_discipline_summary") or ""
        ),
        "balance_sheet_stress_class": str(
            row.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN"
        ),
        "balance_sheet_stress_reason_codes": [
            str(code)
            for code in (row.get("balance_sheet_stress_reason_codes") or [])
            if str(code).strip()
        ],
        "refinancing_risk_class": str(
            row.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
        ),
        "refinancing_risk_reason_codes": [
            str(code)
            for code in (row.get("refinancing_risk_reason_codes") or [])
            if str(code).strip()
        ],
        "balance_sheet_support_signals": [
            str(code)
            for code in (row.get("balance_sheet_support_signals") or [])
            if str(code).strip()
        ],
        "balance_sheet_headwind_signals": [
            str(code)
            for code in (row.get("balance_sheet_headwind_signals") or [])
            if str(code).strip()
        ],
        "primary_balance_sheet_caution": str(
            row.get("primary_balance_sheet_caution") or "BALANCE_SHEET_UNCLEAR"
        ),
        "balance_sheet_discipline_summary": str(
            row.get("balance_sheet_discipline_summary") or ""
        ),
        "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
        "normalized_earnings_power_value": row.get("normalized_earnings_power_value", UNKNOWN),
        "normalized_earnings_power_method_used": str(row.get("normalized_earnings_power_method_used") or UNKNOWN),
        "normalized_earnings_power_status": str(row.get("normalized_earnings_power_status") or UNKNOWN),
        "intrinsic_floor": row.get("intrinsic_floor", UNKNOWN),
        "intrinsic_base": row.get("intrinsic_base", UNKNOWN),
        "intrinsic_ceiling": row.get("intrinsic_ceiling", UNKNOWN),
        "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
        "mos_to_base": row.get("mos_to_base", UNKNOWN),
        "mos_classification": str(row.get("mos_classification") or UNKNOWN),
        "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
        "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
        "valuation_support_types_present": [
            str(value)
            for value in (row.get("valuation_support_types_present") or [])
            if str(value).strip()
        ],
        "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
        "valuation_convergence_band_pct": row.get("valuation_convergence_band_pct", UNKNOWN),
        "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
        "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
        "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
        "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
        "investment_readiness_reason_codes": [
            str(code)
            for code in (row.get("investment_readiness_reason_codes") or [])
            if str(code).strip()
        ],
        "normalization_credibility_class": str(row.get("normalization_credibility_class") or "NORMALIZATION_CREDIBILITY_UNKNOWN"),
        "primary_normalization_caution": str(row.get("primary_normalization_caution") or "NORMALIZATION_UNCLEAR"),
        "capital_allocation_discipline_class": str(row.get("capital_allocation_discipline_class") or "CAPITAL_ALLOCATION_UNKNOWN"),
        "primary_capital_allocation_caution": str(row.get("primary_capital_allocation_caution") or "CAPITAL_ALLOCATION_UNCLEAR"),
        "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or UNKNOWN),
        "evidence_sufficiency_reason_codes": [
            str(code)
            for code in (row.get("evidence_sufficiency_reason_codes") or [])
            if str(code).strip()
        ],
        "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
        "mos_guardrail_reason_codes": [
            str(code)
            for code in (row.get("mos_guardrail_reason_codes") or [])
            if str(code).strip()
        ],
        "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
        "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
        "valuation_support_count_reason_codes": [
            str(code)
            for code in (row.get("valuation_support_count_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_convergence_reason_codes": [
            str(code)
            for code in (row.get("valuation_convergence_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_fragility_reason_codes": [
            str(code)
            for code in (row.get("valuation_fragility_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_confidence_reason_codes": [
            str(code)
            for code in (row.get("valuation_confidence_reason_codes") or [])
            if str(code).strip()
        ],
        "valuation_integrity_reason_codes": [
            str(code)
            for code in (row.get("valuation_integrity_reason_codes") or [])
            if str(code).strip()
        ],
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
        "intangible_economics_reason_codes": [
            str(code)
            for code in (row.get("intangible_economics_reason_codes") or [])
            if str(code).strip()
        ],
        "priority_support_codes": [
            code
            for code in [
                _memory_priority_explanation(row),
                _oe_quality_support_code(row),
                _intangible_support_code(row),
                _owner_value_capture_support_code(row),
                _owner_value_capture_headwind_code(row),
                _balance_sheet_support_code(row),
                _accounting_quality_support_code(row),
                _maintenance_capex_support_code(row),
                _returns_persistence_support_code(row),
                _revenue_dependence_support_code(row),
                _reinvestment_support_code(row),
                _balance_sheet_headwind_code(row),
                _accounting_quality_headwind_code(row),
                _maintenance_capex_headwind_code(row),
                _returns_persistence_headwind_code(row),
                _revenue_dependence_headwind_code(row),
                _reinvestment_headwind_code(row),
                _intrinsic_support_code(row),
                _mos_to_floor_support_code(row),
                _intrinsic_headwind_code(row),
                _valuation_confidence_support_code(row),
                _valuation_fragility_support_code(row),
                _valuation_headwind_code(row),
                _valuation_integrity_headwind_code(row),
                _cyclical_risk_headwind_code(row),
                _impairment_headwind_code(row),
                _impairment_support_code(row),
                _normalization_credibility_support_code(row),
                _normalization_credibility_headwind_code(row),
                _readiness_support_code(row),
                _mos_guardrail_support_code(row),
                _mos_guardrail_headwind_code(row),
                _value_type_support_code(row),
                _value_type_headwind_code(row),
                _facts_action_reason(row),
                "ECONOMICS_KNOWN_WEAK" if str(row.get("primary_fail_domain") or "") == "ECONOMICS" else "",
            ]
            if str(code).strip() and str(code).strip() != "NEUTRAL_MEMORY"
        ],
    }


def _queue_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    metrics = entry.get("latest_metrics") if isinstance(entry.get("latest_metrics"), dict) else {}
    return (
        _lane_rank(str(entry.get("priority_lane") or "")),
        -int(entry.get("l4_priority_boost") or 0),
        _action_rank(str(entry.get("action_type") or "")),
        _readiness_rank(entry.get("investment_readiness_class", "READINESS_UNKNOWN")),
        -int(entry.get("memory_priority_total") or 0),
        _desc_key(entry.get("mos_to_floor", UNKNOWN)),
        _confidence_rank(entry.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(entry.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _balance_sheet_stress_rank(
            entry.get("balance_sheet_stress_class", "BALANCE_SHEET_STRESS_UNKNOWN")
        ),
        _refinancing_risk_rank(entry.get("refinancing_risk_class", "REFINANCING_RISK_UNKNOWN")),
        _returns_persistence_rank(entry.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")),
        _revenue_dependence_rank(entry.get("revenue_dependence_risk_class", "REVENUE_DEPENDENCE_UNKNOWN")),
        _maintenance_capex_credibility_rank(
            entry.get("maintenance_capex_credibility_class", "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN")
        ),
        _asset_intensity_rank(entry.get("asset_intensity_class", "ASSET_INTENSITY_UNKNOWN")),
        _accounting_quality_rank(entry.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
        _reinvestment_rank(entry.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")),
        _value_type_rank(entry.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _desc_key(entry.get("valuation_support_count", UNKNOWN)),
        _desc_key(entry.get("oe_quality_total", UNKNOWN)),
        _desc_key(entry.get("intangible_economics_total", UNKNOWN)),
        _desc_key(entry.get("owner_value_capture_score", UNKNOWN)),
        _desc_key(metrics.get("implied_return_base", UNKNOWN)),
        -int(entry.get("appearances_count") or 0),
        str(entry.get("ticker") or ""),
    )


def build_escalation_plan(
    campaign_run_id: str,
    promotion_state: dict[str, Any],
    priority_lanes: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    campaign_config = {
        "as_of_date": str((config or {}).get("as_of_date") or ""),
        "source_campaign_file": str((config or {}).get("source_campaign_file") or ""),
        "policy": str((config or {}).get("policy") or "value_first"),
        "top_n": int((config or {}).get("top_n") or 10),
    }
    rows = [value for value in (promotion_state.get("rows") or []) if isinstance(value, dict)]
    memory_lookup = _load_memory_lookup()
    l4_context: dict[str, Any] = {
        "cfg": get_config(),
        "as_of_date": str(campaign_config.get("as_of_date") or ""),
        "run_cache": {},
    }
    ticker_plans: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []

    for raw_row in rows:
        row = dict(raw_row)
        fallback_memory = _memory_priority_fields(str(row.get("ticker") or ""), memory_lookup)
        row["memory_priority_total"] = int(
            row.get("memory_priority_total")
            if row.get("memory_priority_total") is not None
            else fallback_memory.get("memory_priority_total")
            or 0
        )
        row["memory_priority_reason_codes"] = [
            str(code)
            for code in (
                row.get("memory_priority_reason_codes")
                if isinstance(row.get("memory_priority_reason_codes"), list)
                else fallback_memory.get("memory_priority_reason_codes")
                or []
            )
            if str(code).strip()
        ]
        row["memory_priority_source_campaign_run_ids"] = [
            str(run_id)
            for run_id in (
                row.get("memory_priority_source_campaign_run_ids")
                if isinstance(row.get("memory_priority_source_campaign_run_ids"), list)
                else fallback_memory.get("memory_priority_source_campaign_run_ids")
                or []
            )
            if str(run_id).strip()
        ]
        lane = str(row.get("priority_lane") or UNKNOWN)
        blocker = _blocker_code(row)
        actions: list[dict[str, Any]] = []
        terminal = "TERMINAL_BLOCKER" in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        facts_action_type = _facts_action_type(row)
        facts_action_reason = _facts_action_reason(row)

        if lane == LANE_1_HIGH_PRIORITY and not terminal:
            if not _history_blocker_unchanged(row):
                actions.append(
                    {
                        "action_type": ACTION_REBUILD_DEPTH_RUN,
                        "action_reason": "ESCALATE_DEEPER_RESEARCH",
                    }
                )
            actions.append(
                {
                    "action_type": ACTION_BUILD_REFRESHED_MEMO,
                    "action_reason": "REFRESH_MEMO_AFTER_PROMOTION",
                }
            )
            actions.append(
                {
                    "action_type": ACTION_ADD_TO_ACTIVE_WATCHLIST,
                    "action_reason": "TRACK_AS_ACTIVE_CONVICTION",
                }
            )
        elif lane == LANE_2_RESEARCH_QUEUE:
            if facts_action_type:
                actions.append(
                    {
                        "action_type": facts_action_type,
                        "action_reason": facts_action_reason or "FACTS_BLOCKER_RETRY_PATH",
                    }
                )
            else:
                actions.append(
                    {
                        "action_type": ACTION_CLEAR_BLOCKERS,
                        "action_reason": _clear_blocker_reason(blocker),
                    }
                )
            actions.append(
                {
                    "action_type": ACTION_RECHECK_PROMOTION,
                    "action_reason": "RECHECK_AFTER_BLOCKER_CLEAR",
                }
            )
        elif lane == LANE_3_MONITOR:
            if facts_action_type:
                actions.append(
                    {
                        "action_type": facts_action_type,
                        "action_reason": facts_action_reason or "FACTS_PARTIAL_USABLE_RECHECK",
                    }
                )
            else:
                actions.append(
                    {
                        "action_type": ACTION_SCHEDULE_LIGHT_REFRESH,
                        "action_reason": "MONITOR_AND_REFRESH_LIGHTLY",
                    }
                )
        else:
            actions.append(
                {
                    "action_type": ACTION_NO_ACTION,
                    "action_reason": "RECHECK_ON_NEW_EVIDENCE",
                }
            )

        actions.extend(_planned_l4_actions(row, context=l4_context))

        ticker_plan = {
            "ticker": str(row.get("ticker") or ""),
            "priority_lane": lane,
            "latest_value_gate_status": str(row.get("latest_value_gate_status") or UNKNOWN),
            "latest_primary_blocker": blocker,
            "latest_metrics": _latest_metrics(row),
            "source_universe_run_ids": _source_universe_run_ids(row),
            "source_runs": [value for value in (row.get("source_runs") or []) if isinstance(value, dict)],
            "memo_path": str(row.get("memo_path") or ""),
            "memory_priority_total": int(row.get("memory_priority_total") or 0),
            "memory_priority_reason_codes": [
                str(code)
                for code in (row.get("memory_priority_reason_codes") or [])
                if str(code).strip()
            ],
            "memory_priority_explanation": _memory_priority_explanation(row),
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
            "intangible_economics_reason_codes": [
                str(code)
                for code in (row.get("intangible_economics_reason_codes") or [])
                if str(code).strip()
            ],
            "normalized_earnings_power_value": row.get("normalized_earnings_power_value", UNKNOWN),
            "normalized_earnings_power_method_used": str(row.get("normalized_earnings_power_method_used") or UNKNOWN),
            "normalized_earnings_power_status": str(row.get("normalized_earnings_power_status") or UNKNOWN),
            "intrinsic_floor": row.get("intrinsic_floor", UNKNOWN),
            "intrinsic_base": row.get("intrinsic_base", UNKNOWN),
            "intrinsic_ceiling": row.get("intrinsic_ceiling", UNKNOWN),
            "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
            "mos_to_base": row.get("mos_to_base", UNKNOWN),
            "mos_classification": str(row.get("mos_classification") or UNKNOWN),
            "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
            "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
            "valuation_support_types_present": [
                str(value)
                for value in (row.get("valuation_support_types_present") or [])
                if str(value).strip()
            ],
            "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
            "valuation_convergence_band_pct": row.get("valuation_convergence_band_pct", UNKNOWN),
            "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
            "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
            "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
            "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
            "investment_readiness_reason_codes": [
                str(code)
                for code in (row.get("investment_readiness_reason_codes") or [])
                if str(code).strip()
            ],
            "normalization_credibility_class": str(row.get("normalization_credibility_class") or "NORMALIZATION_CREDIBILITY_UNKNOWN"),
            "primary_normalization_caution": str(row.get("primary_normalization_caution") or "NORMALIZATION_UNCLEAR"),
            "capital_allocation_discipline_class": str(row.get("capital_allocation_discipline_class") or "CAPITAL_ALLOCATION_UNKNOWN"),
            "primary_capital_allocation_caution": str(row.get("primary_capital_allocation_caution") or "CAPITAL_ALLOCATION_UNCLEAR"),
            "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or UNKNOWN),
            "evidence_sufficiency_reason_codes": [
                str(code)
                for code in (row.get("evidence_sufficiency_reason_codes") or [])
                if str(code).strip()
            ],
            "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
            "mos_guardrail_reason_codes": [
                str(code)
                for code in (row.get("mos_guardrail_reason_codes") or [])
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
            "valuation_support_count_reason_codes": [
                str(code)
                for code in (row.get("valuation_support_count_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_convergence_reason_codes": [
                str(code)
                for code in (row.get("valuation_convergence_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_fragility_reason_codes": [
                str(code)
                for code in (row.get("valuation_fragility_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_confidence_reason_codes": [
                str(code)
                for code in (row.get("valuation_confidence_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_integrity_reason_codes": [
                str(code)
                for code in (row.get("valuation_integrity_reason_codes") or [])
                if str(code).strip()
            ],
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
            "facts_missing_key_inputs": [
                str(value)
                for value in (row.get("facts_missing_key_inputs") or [])
                if str(value).strip()
            ],
            "facts_retry_recommended": bool(row.get("facts_retry_recommended", False)),
            "facts_blocker_reason_codes": [
                str(code)
                for code in (row.get("facts_blocker_reason_codes") or [])
                if str(code).strip()
            ],
            "facts_recommended_action": str(row.get("facts_recommended_action") or "NONE"),
            "primary_fail_domain": str(row.get("primary_fail_domain") or "NONE"),
            "l4_signal_summary": _l4_signal_summary(row),
            "l4_priority_boost": _l4_priority_boost(row),
            "priority_support_codes": [
                code
                for code in [
                    _memory_priority_explanation(row),
                    _oe_quality_support_code(row),
                    _intangible_support_code(row),
                    _owner_value_capture_support_code(row),
                    _owner_value_capture_headwind_code(row),
                    _reinvestment_support_code(row),
                    _reinvestment_headwind_code(row),
                    _intrinsic_support_code(row),
                    _mos_to_floor_support_code(row),
                    _intrinsic_headwind_code(row),
                    _valuation_confidence_support_code(row),
                    _valuation_fragility_support_code(row),
                    _valuation_headwind_code(row),
                    _valuation_integrity_headwind_code(row),
                    _impairment_headwind_code(row),
                    _impairment_support_code(row),
                    _normalization_credibility_support_code(row),
                    _normalization_credibility_headwind_code(row),
                    _readiness_support_code(row),
                    _mos_guardrail_support_code(row),
                    _mos_guardrail_headwind_code(row),
                    _value_type_support_code(row),
                    _value_type_headwind_code(row),
                    _facts_action_reason(row),
                ]
                if str(code).strip() and str(code).strip() != "NEUTRAL_MEMORY"
            ],
            "actions": actions,
        }
        ticker_plans.append(ticker_plan)

        for action in actions:
            queue.append(
                _queue_entry(
                    queue_rank=0,
                    campaign_run_id=campaign_run_id,
                    row=row,
                    action_type=str(action.get("action_type") or ""),
                    action_reason=str(action.get("action_reason") or ""),
                    campaign_config=campaign_config,
                    action_metadata=action.get("action_metadata") if isinstance(action.get("action_metadata"), dict) else None,
                )
            )

    ranked_queue = sorted(queue, key=_queue_sort_key)
    for idx, entry in enumerate(ranked_queue, start=1):
        entry["queue_rank"] = int(idx)

    lane_to_action_counts: dict[str, dict[str, int]] = {}
    action_type_counts: dict[str, int] = {}
    for entry in ranked_queue:
        lane = str(entry.get("priority_lane") or UNKNOWN)
        action_type = str(entry.get("action_type") or UNKNOWN)
        lane_to_action_counts.setdefault(lane, {})
        lane_to_action_counts[lane][action_type] = lane_to_action_counts[lane].get(action_type, 0) + 1
        action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1

    return {
        "campaign_run_id": campaign_run_id,
        "generated_at": utc_now_iso(),
        "config_effective": campaign_config,
        "lane_counts": promotion_state.get("lane_counts") if isinstance(promotion_state.get("lane_counts"), dict) else {},
        "priority_lane_keys": [key for key in priority_lanes.keys() if key.startswith("lane_")],
        "ticker_count": len(ticker_plans),
        "escalation_queue_count": len(ranked_queue),
        "lane_to_action_counts": {
            lane: dict(sorted(counts.items(), key=lambda item: item[0]))
            for lane, counts in sorted(lane_to_action_counts.items(), key=lambda item: item[0])
        },
        "action_type_counts": dict(sorted(action_type_counts.items(), key=lambda item: item[0])),
        "tickers": ticker_plans,
        "queue": ranked_queue,
    }


def write_escalation_artifacts(
    campaign_run_id: str,
    *,
    promotion_state: dict[str, Any] | None = None,
    priority_lanes: dict[str, Any] | None = None,
    campaign_state: dict[str, Any] | None = None,
) -> dict[str, str]:
    paths = _escalation_paths(campaign_run_id)
    promotion_payload = promotion_state or _safe_json(paths["promotion_state_path"])
    lanes_payload = priority_lanes or _safe_json(paths["priority_lanes_path"])
    campaign_state_payload = campaign_state or _safe_json(paths["campaign_state_path"])

    plan = build_escalation_plan(
        campaign_run_id,
        promotion_payload,
        lanes_payload,
        config={
            "as_of_date": str(campaign_state_payload.get("as_of_date") or ""),
            "source_campaign_file": str(campaign_state_payload.get("source_campaign_file") or ""),
            "policy": str(campaign_state_payload.get("policy") or "value_first"),
            "top_n": 10,
        },
    )
    queue_payload = {
        "campaign_run_id": campaign_run_id,
        "generated_at": plan["generated_at"],
        "queue_count": int(plan.get("escalation_queue_count") or 0),
        "rows": plan.get("queue") if isinstance(plan.get("queue"), list) else [],
    }
    summary_payload = {
        "campaign_run_id": campaign_run_id,
        "generated_at": plan["generated_at"],
        "escalation_queue_count": int(plan.get("escalation_queue_count") or 0),
        "lane_to_action_counts": plan.get("lane_to_action_counts") if isinstance(plan.get("lane_to_action_counts"), dict) else {},
        "top_10_escalation_actions": [
            {
                "queue_rank": int(entry.get("queue_rank") or 0),
                "ticker": str(entry.get("ticker") or ""),
                "priority_lane": str(entry.get("priority_lane") or UNKNOWN),
                "action_type": str(entry.get("action_type") or UNKNOWN),
                "action_reason": str(entry.get("action_reason") or UNKNOWN),
                "memory_priority_total": int(entry.get("memory_priority_total") or 0),
                "memory_priority_explanation": str(entry.get("memory_priority_explanation") or "NEUTRAL_MEMORY"),
                "oe_quality_total": entry.get("oe_quality_total", UNKNOWN),
                "intangible_economics_total": entry.get("intangible_economics_total", UNKNOWN),
                "owner_value_capture_score": entry.get("owner_value_capture_score", UNKNOWN),
                "reinvestment_efficiency_class": str(
                    entry.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
                ),
                "primary_reinvestment_caution": str(
                    entry.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
                ),
                "accounting_quality_class": str(
                    entry.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
                ),
                "primary_accounting_caution": str(
                    entry.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
                ),
                "balance_sheet_stress_class": str(
                    entry.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN"
                ),
                "refinancing_risk_class": str(
                    entry.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
                ),
                "primary_balance_sheet_caution": str(
                    entry.get("primary_balance_sheet_caution") or "BALANCE_SHEET_UNCLEAR"
                ),
                "mos_to_floor": entry.get("mos_to_floor", UNKNOWN),
                "mos_classification": str(entry.get("mos_classification") or UNKNOWN),
                "valuation_support_count": _int_or_zero(entry.get("valuation_support_count")),
                "valuation_convergence_status": str(entry.get("valuation_convergence_status") or UNKNOWN),
                "valuation_fragility_status": str(entry.get("valuation_fragility_status") or UNKNOWN),
                "valuation_confidence_class": str(entry.get("valuation_confidence_class") or UNKNOWN),
                "valuation_integrity_class": str(entry.get("valuation_integrity_class") or UNKNOWN),
                "investment_readiness_class": str(entry.get("investment_readiness_class") or UNKNOWN),
                "evidence_sufficiency_class": str(entry.get("evidence_sufficiency_class") or UNKNOWN),
                "mos_assessment_status": str(entry.get("mos_assessment_status") or UNKNOWN),
                "blocker_stack_primary": str(entry.get("blocker_stack_primary") or UNKNOWN),
                "primary_next_step": str(entry.get("primary_next_step") or UNKNOWN),
                "value_type_primary": str(entry.get("value_type_primary") or UNKNOWN),
                "downside_support_type": str(entry.get("downside_support_type") or UNKNOWN),
                "facts_blocker_class": str(entry.get("facts_blocker_class") or "FACTS_OK"),
                "facts_recommended_action": str(entry.get("facts_recommended_action") or "NONE"),
                "primary_fail_domain": str(entry.get("primary_fail_domain") or "NONE"),
                "l4_signal_summary": entry.get("l4_signal_summary") if isinstance(entry.get("l4_signal_summary"), dict) else {},
                "l4_priority_boost": int(entry.get("l4_priority_boost") or 0),
                "priority_support_codes": [
                    str(code)
                    for code in (entry.get("priority_support_codes") or [])
                    if str(code).strip()
                ],
                "recommended_command": str(entry.get("recommended_command") or ""),
            }
            for entry in [value for value in (plan.get("queue") or []) if isinstance(value, dict)][:10]
        ],
    }

    _json_write(paths["escalation_plan_path"], plan)
    _json_write(paths["escalation_queue_path"], queue_payload)
    _json_write(paths["escalation_summary_path"], summary_payload)
    return {
        "escalation_plan_path": str(paths["escalation_plan_path"]),
        "escalation_queue_path": str(paths["escalation_queue_path"]),
        "escalation_summary_path": str(paths["escalation_summary_path"]),
    }


def open_escalation_plan(campaign_run_id: str) -> dict[str, Any]:
    paths = _escalation_paths(campaign_run_id)
    summary = _safe_json(paths["escalation_summary_path"])
    queue = _safe_json(paths["escalation_queue_path"])
    if not summary:
        return {
            "status": "MISSING",
            "campaign_run_id": campaign_run_id,
            "escalation_plan_path": str(paths["escalation_plan_path"]),
            "escalation_queue_path": str(paths["escalation_queue_path"]),
            "escalation_summary_path": str(paths["escalation_summary_path"]),
        }
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "escalation_queue_count": int(summary.get("escalation_queue_count") or 0),
        "lane_to_action_counts": summary.get("lane_to_action_counts") if isinstance(summary.get("lane_to_action_counts"), dict) else {},
        "top_10_escalation_actions": summary.get("top_10_escalation_actions")
        if isinstance(summary.get("top_10_escalation_actions"), list)
        else [],
        "top_10_queue": [
            {
                "queue_rank": int(entry.get("queue_rank") or 0),
                "ticker": str(entry.get("ticker") or ""),
                "priority_lane": str(entry.get("priority_lane") or UNKNOWN),
                "action_type": str(entry.get("action_type") or UNKNOWN),
                "blocking_reason_code": str(entry.get("blocking_reason_code") or UNKNOWN),
                "memory_priority_total": int(entry.get("memory_priority_total") or 0),
                "memory_priority_explanation": str(entry.get("memory_priority_explanation") or "NEUTRAL_MEMORY"),
                "oe_quality_total": entry.get("oe_quality_total", UNKNOWN),
                "intangible_economics_total": entry.get("intangible_economics_total", UNKNOWN),
                "owner_value_capture_score": entry.get("owner_value_capture_score", UNKNOWN),
                "reinvestment_efficiency_class": str(
                    entry.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
                ),
                "primary_reinvestment_caution": str(
                    entry.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
                ),
                "mos_to_floor": entry.get("mos_to_floor", UNKNOWN),
                "mos_classification": str(entry.get("mos_classification") or UNKNOWN),
                "valuation_support_count": _int_or_zero(entry.get("valuation_support_count")),
                "valuation_convergence_status": str(entry.get("valuation_convergence_status") or UNKNOWN),
                "valuation_fragility_status": str(entry.get("valuation_fragility_status") or UNKNOWN),
                "valuation_confidence_class": str(entry.get("valuation_confidence_class") or UNKNOWN),
                "valuation_integrity_class": str(entry.get("valuation_integrity_class") or UNKNOWN),
                "investment_readiness_class": str(entry.get("investment_readiness_class") or UNKNOWN),
                "evidence_sufficiency_class": str(entry.get("evidence_sufficiency_class") or UNKNOWN),
                "mos_assessment_status": str(entry.get("mos_assessment_status") or UNKNOWN),
                "blocker_stack_primary": str(entry.get("blocker_stack_primary") or UNKNOWN),
                "primary_next_step": str(entry.get("primary_next_step") or UNKNOWN),
                "value_type_primary": str(entry.get("value_type_primary") or UNKNOWN),
                "downside_support_type": str(entry.get("downside_support_type") or UNKNOWN),
                "facts_blocker_class": str(entry.get("facts_blocker_class") or "FACTS_OK"),
                "facts_recommended_action": str(entry.get("facts_recommended_action") or "NONE"),
                "primary_fail_domain": str(entry.get("primary_fail_domain") or "NONE"),
                "l4_signal_summary": entry.get("l4_signal_summary") if isinstance(entry.get("l4_signal_summary"), dict) else {},
                "l4_priority_boost": int(entry.get("l4_priority_boost") or 0),
                "priority_support_codes": [
                    str(code)
                    for code in (entry.get("priority_support_codes") or [])
                    if str(code).strip()
                ],
                "recommended_command": str(entry.get("recommended_command") or ""),
            }
            for entry in [value for value in (queue.get("rows") or []) if isinstance(value, dict)][:10]
        ],
        "escalation_plan_path": str(paths["escalation_plan_path"]),
        "escalation_queue_path": str(paths["escalation_queue_path"]),
        "escalation_summary_path": str(paths["escalation_summary_path"]),
    }
