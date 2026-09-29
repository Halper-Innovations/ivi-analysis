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
NONE = "NONE"

GATE_UPGRADE = "UPGRADE"
GATE_DOWNGRADE = "DOWNGRADE"
GATE_UNCHANGED = "UNCHANGED"

LANE_UPGRADE = "UPGRADE"
LANE_DOWNGRADE = "DOWNGRADE"
LANE_UNCHANGED = "UNCHANGED"

ESCALATION_IMPROVED = "IMPROVED"
ESCALATION_NO_CHANGE = "NO_CHANGE"
ESCALATION_DETERIORATED = "DETERIORATED"
ESCALATION_NOT_APPLICABLE = "NOT_APPLICABLE"

_GATE_ORDER = {
    "FAIL": 0,
    "WATCH": 1,
    "PASS": 2,
}
_LANE_ORDER = {
    LANE_4_DEPRIORITIZED: 0,
    LANE_3_MONITOR: 1,
    LANE_2_RESEARCH_QUEUE: 2,
    LANE_1_HIGH_PRIORITY: 3,
}
_BLOCKER_HYDRABLE = {
    "PRICE_UNKNOWN",
    "MISSING_EV",
    "MISSING_CASH",
    "MISSING_DEBT",
    "MISSING_BOTH",
    "MISSING_NET_DEBT",
    "NO_FACTS",
    "MISSING_SHARES",
    "MISSING_FCF",
    "MISSING_GD_INPUTS",
    "MISSING_CURRENT_ASSETS",
    "MISSING_TOTAL_LIABILITIES",
    "MISSING_EARNINGS_STREAM",
}
_BLOCKER_TERMINAL = {
    "FAIL_CONFIRMED",
    "TERMINAL_BLOCKER",
    "DELISTED",
    "EXCLUDED",
}
_SUPPORTIVE_MEMORY_REASONS = {
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
_STALLED_MEMORY_REASONS = {
    "GATE_DOWNGRADE",
    "LANE_DOWNGRADE",
    "IMPLIED_RETURN_KNOWN_TO_UNKNOWN",
    "RECURRING_UNCHANGED_BLOCKER_3_PLUS",
    "RECURRING_TERMINAL_BLOCKER",
    "ESCALATION_DETERIORATED",
}


def _research_memory_paths() -> dict[str, Path]:
    root = get_config().outputs_dir / "research_memory"
    return {
        "root": root,
        "memory_path": root / "research_memory.json",
        "summary_path": root / "research_memory_summary.json",
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


def _to_numeric_or_unknown(value: Any) -> float | int | str:
    if _is_num(value):
        return value
    return UNKNOWN


def _token(value: Any) -> str:
    token = str(value or "").strip().upper()
    return token or UNKNOWN


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


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _default_memory() -> dict[str, Any]:
    return {
        "generated_at": "",
        "last_updated_campaign_run_id": "",
        "ticker_count": 0,
        "updated_tickers": [],
        "tickers": {},
    }


def _empty_memory_priority() -> dict[str, Any]:
    return {
        "repeated_survivor_score": 0,
        "improvement_score": 0,
        "blocker_resolution_score": 0,
        "deterioration_penalty": 0,
        "memory_priority_total": 0,
        "memory_priority_reason_codes": [],
        "source_campaign_run_ids": [],
    }


def _ordered_change(prev_value: Any, new_value: Any, ranks: dict[str, int], *, improvement_label: str, deterioration_label: str) -> str:
    prev_token = _token(prev_value)
    new_token = _token(new_value)
    if prev_token == new_token:
        return GATE_UNCHANGED
    if prev_token not in ranks or new_token not in ranks:
        return GATE_UNCHANGED
    if ranks[new_token] > ranks[prev_token]:
        return improvement_label
    if ranks[new_token] < ranks[prev_token]:
        return deterioration_label
    return GATE_UNCHANGED


def _numeric_change(prev_value: Any, new_value: Any, *, baseline_exists: bool) -> dict[str, Any]:
    before = _to_numeric_or_unknown(prev_value)
    after = _to_numeric_or_unknown(new_value)
    if not baseline_exists:
        return {
            "from": before,
            "to": after,
            "direction": GATE_UNCHANGED,
        }
    if before == UNKNOWN and after == UNKNOWN:
        direction = GATE_UNCHANGED
    elif before == UNKNOWN and after != UNKNOWN:
        direction = "UNKNOWN_TO_KNOWN"
    elif before != UNKNOWN and after == UNKNOWN:
        direction = "KNOWN_TO_UNKNOWN"
    elif float(after) > float(before):
        direction = "UP"
    elif float(after) < float(before):
        direction = "DOWN"
    else:
        direction = GATE_UNCHANGED
    return {
        "from": before,
        "to": after,
        "direction": direction,
    }


def _blocker_class(code: Any) -> str:
    token = _token(code)
    if token in {UNKNOWN, NONE}:
        return token
    if token in _BLOCKER_TERMINAL:
        return "TERMINAL"
    if token in _BLOCKER_HYDRABLE:
        return "HYDRABLE"
    return "OTHER"


def _blocker_change(prev_value: Any, new_value: Any, *, baseline_exists: bool) -> dict[str, Any]:
    before = _token(prev_value)
    after = _token(new_value)
    if not baseline_exists:
        return {
            "from": before,
            "to": after,
            "changed": False,
            "classification": GATE_UNCHANGED,
        }
    if before == after:
        return {
            "from": before,
            "to": after,
            "changed": False,
            "classification": GATE_UNCHANGED,
        }
    before_class = _blocker_class(before)
    after_class = _blocker_class(after)
    classification = "CHANGED"
    if before not in {NONE, UNKNOWN} and after == NONE:
        classification = ESCALATION_IMPROVED
    elif before == NONE and after not in {NONE, UNKNOWN}:
        classification = ESCALATION_DETERIORATED
    elif before_class == "TERMINAL" and after_class == "HYDRABLE":
        classification = ESCALATION_IMPROVED
    elif before_class == "HYDRABLE" and after_class == "TERMINAL":
        classification = ESCALATION_DETERIORATED
    return {
        "from": before,
        "to": after,
        "changed": True,
        "classification": classification,
    }


def _escalation_effect(new_entry: dict[str, Any], delta: dict[str, Any], *, baseline_exists: bool) -> str:
    escalation_rows = [row for row in (new_entry.get("escalation_rows") or []) if isinstance(row, dict)]
    if not escalation_rows:
        return ESCALATION_NOT_APPLICABLE
    if not baseline_exists:
        lane_before = _token(escalation_rows[-1].get("lane_before"))
        lane_after = _token(escalation_rows[-1].get("lane_after"))
        if lane_before in _LANE_ORDER and lane_after in _LANE_ORDER and _LANE_ORDER[lane_after] > _LANE_ORDER[lane_before]:
            return ESCALATION_IMPROVED
        if lane_before in _LANE_ORDER and lane_after in _LANE_ORDER and _LANE_ORDER[lane_after] < _LANE_ORDER[lane_before]:
            return ESCALATION_DETERIORATED
        return ESCALATION_NO_CHANGE
    improved = (
        delta.get("gate_change") == GATE_UPGRADE
        or delta.get("lane_change") == LANE_UPGRADE
        or (delta.get("implied_return_change") or {}).get("direction") in {"UP", "UNKNOWN_TO_KNOWN"}
    )
    deteriorated = (
        delta.get("gate_change") == GATE_DOWNGRADE
        or delta.get("lane_change") == LANE_DOWNGRADE
        or (delta.get("implied_return_change") or {}).get("direction") in {"DOWN", "KNOWN_TO_UNKNOWN"}
    )
    if improved:
        return ESCALATION_IMPROVED
    if deteriorated:
        return ESCALATION_DETERIORATED
    return ESCALATION_NO_CHANGE


def _history_entry(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_run_id": str(snapshot.get("campaign_run_id") or ""),
        "timestamp": str(snapshot.get("timestamp") or utc_now_iso()),
        "value_gate_status": _token(snapshot.get("value_gate_status")),
        "priority_lane": str(snapshot.get("priority_lane") or UNKNOWN),
        "implied_return_base": _to_numeric_or_unknown(snapshot.get("implied_return_base")),
        "mos_epv": _to_numeric_or_unknown(snapshot.get("mos_epv")),
        "yield_metric_used": str(snapshot.get("yield_metric_used") or UNKNOWN),
        "primary_blocker": _token(snapshot.get("primary_blocker")),
        "blocker_reason_code": _token(snapshot.get("blocker_reason_code")),
        "memo_path": str(snapshot.get("memo_path") or ""),
        "derived_from": _dedupe_refs(list(snapshot.get("derived_from") or [])),
        "source_paths": snapshot.get("source_paths") if isinstance(snapshot.get("source_paths"), dict) else {},
        "director_metadata": _sanitize_director_metadata(snapshot.get("director_metadata")),
    }


def _latest_entry(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_run_id": str(snapshot.get("campaign_run_id") or ""),
        "value_gate_status": _token(snapshot.get("value_gate_status")),
        "priority_lane": str(snapshot.get("priority_lane") or UNKNOWN),
        "implied_return_base": _to_numeric_or_unknown(snapshot.get("implied_return_base")),
        "mos_epv": _to_numeric_or_unknown(snapshot.get("mos_epv")),
        "yield_metric_used": str(snapshot.get("yield_metric_used") or UNKNOWN),
        "primary_blocker": _token(snapshot.get("primary_blocker")),
        "blocker_reason_code": _token(snapshot.get("blocker_reason_code")),
        "memo_path": str(snapshot.get("memo_path") or ""),
        "derived_from": _dedupe_refs(list(snapshot.get("derived_from") or [])),
        "source_paths": snapshot.get("source_paths") if isinstance(snapshot.get("source_paths"), dict) else {},
        "director_metadata": _sanitize_director_metadata(snapshot.get("director_metadata")),
    }


def _campaign_source_paths(campaign_summary: dict[str, Any], escalation_results: dict[str, Any] | None) -> dict[str, str]:
    return {
        "campaign_summary_path": str(campaign_summary.get("campaign_summary_path") or ""),
        "master_shortlist_json_path": str(campaign_summary.get("master_shortlist_json_path") or ""),
        "master_watchlist_state_path": str(campaign_summary.get("master_watchlist_state_path") or ""),
        "promotion_state_path": str(campaign_summary.get("promotion_state_path") or ""),
        "escalation_results_path": str((escalation_results or {}).get("escalation_results_path") or ""),
    }


def _sanitize_director_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    agenda_generated_at = str(payload.get("agenda_generated_at") or "").strip()
    if agenda_generated_at:
        out["agenda_generated_at"] = agenda_generated_at
    if _is_num(payload.get("sector_priority_rank")):
        out["sector_priority_rank"] = int(payload.get("sector_priority_rank") or 0)
    recommended_depth = str(payload.get("recommended_depth") or "").strip()
    if recommended_depth:
        out["recommended_depth"] = recommended_depth
    director_reason_narrative = str(payload.get("director_reason_narrative") or "").strip()
    if director_reason_narrative:
        out["director_reason_narrative"] = director_reason_narrative
    return out


def _source_universe_run_ids(*rows: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        primary = str(row.get("universe_run_id") or "").strip()
        if primary and primary not in seen:
            seen.add(primary)
            out.append(primary)
        for source in [item for item in (row.get("source_runs") or []) if isinstance(item, dict)]:
            token = str(source.get("universe_run_id") or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def _resolve_director_metadata(
    *,
    shortlist_row: dict[str, Any],
    promotion_row: dict[str, Any],
    watch_row: dict[str, Any],
    director_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    direct = _sanitize_director_metadata(director_metadata)
    if direct:
        return direct
    by_universe_run_id = (
        director_metadata.get("by_universe_run_id")
        if isinstance(director_metadata, dict) and isinstance(director_metadata.get("by_universe_run_id"), dict)
        else {}
    )
    for run_id in _source_universe_run_ids(shortlist_row, promotion_row, watch_row):
        metadata = _sanitize_director_metadata(by_universe_run_id.get(run_id))
        if metadata:
            return metadata
    default = director_metadata.get("default") if isinstance(director_metadata, dict) else {}
    return _sanitize_director_metadata(default if isinstance(default, dict) else {})


def _history_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in (entry.get("history") or []) if isinstance(row, dict)]


def _memory_source_campaign_run_ids(entry: dict[str, Any]) -> list[str]:
    history = _history_rows(entry)
    run_ids = [str(row.get("campaign_run_id") or "") for row in history]
    latest = entry.get("latest") if isinstance(entry.get("latest"), dict) else {}
    run_ids.append(str(latest.get("campaign_run_id") or entry.get("last_seen_run_id") or ""))
    return _dedupe_refs(run_ids)


def _repeated_blocker_code(entry: dict[str, Any], *, min_appearances: int = 3) -> str:
    history = _history_rows(entry)
    if len(history) < min_appearances:
        return ""
    blockers = [_token(row.get("primary_blocker")) for row in history[-min_appearances:]]
    latest_blocker = blockers[-1]
    if latest_blocker in {UNKNOWN, NONE}:
        return ""
    if all(blocker == latest_blocker for blocker in blockers):
        return latest_blocker
    return ""


def _ticker_snapshot(
    *,
    ticker: str,
    campaign_run_id: str,
    campaign_summary: dict[str, Any],
    shortlist_row: dict[str, Any],
    watch_row: dict[str, Any],
    promotion_row: dict[str, Any],
    escalation_rows: list[dict[str, Any]],
    director_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    derived_from = _dedupe_refs(
        list(promotion_row.get("derived_from") or [])
        + list(shortlist_row.get("derived_from") or [])
    )
    primary_blocker = (
        promotion_row.get("latest_primary_blocker")
        or promotion_row.get("primary_blocker")
        or shortlist_row.get("latest_primary_blocker")
        or shortlist_row.get("primary_blocker")
        or watch_row.get("latest_primary_blocker")
        or UNKNOWN
    )
    return {
        "ticker": ticker,
        "campaign_run_id": campaign_run_id,
        "timestamp": utc_now_iso(),
        "value_gate_status": (
            promotion_row.get("latest_value_gate_status")
            or promotion_row.get("value_gate_status")
            or shortlist_row.get("latest_value_gate_status")
            or shortlist_row.get("value_gate_status")
            or watch_row.get("latest_value_gate_status")
            or UNKNOWN
        ),
        "priority_lane": promotion_row.get("priority_lane") or UNKNOWN,
        "implied_return_base": (
            promotion_row.get("implied_return_base")
            if _is_num(promotion_row.get("implied_return_base"))
            else shortlist_row.get("implied_return_base")
            if _is_num(shortlist_row.get("implied_return_base"))
            else watch_row.get("latest_implied_return_base")
        ),
        "mos_epv": promotion_row.get("mos_epv", shortlist_row.get("mos_epv", UNKNOWN)),
        "yield_metric_used": promotion_row.get("yield_metric_used") or shortlist_row.get("yield_metric_used") or UNKNOWN,
        "primary_blocker": primary_blocker,
        "blocker_reason_code": primary_blocker,
        "memo_path": promotion_row.get("memo_path") or shortlist_row.get("memo_path") or "",
        "derived_from": derived_from,
        "source_paths": _campaign_source_paths(campaign_summary, {"escalation_results_path": str((campaign_summary.get("escalation_results_path") or "")) or str((escalation_rows[0].get("escalation_results_path") if escalation_rows else "") or "")}),
        "escalation_rows": escalation_rows,
        "director_metadata": _sanitize_director_metadata(director_metadata),
    }


def _empty_memory_summary() -> dict[str, Any]:
    paths = _research_memory_paths()
    return {
        "campaign_run_id": "",
        "generated_at": "",
        "ticker_count": 0,
        "newly_added_count": 0,
        "upgraded_gate_count": 0,
        "upgraded_lane_count": 0,
        "unknown_to_known_implied_return_count": 0,
        "blocker_changes_count": 0,
        "top_improvers": [],
        "top_deteriorations": [],
        "top_memory_priority_candidates": [],
        "top_stalled_names": [],
        "unknown_to_known_valuation_conversions": [],
        "recurring_blockers": {},
        "recurring_blocker_names": [],
        "research_memory_path": str(paths["memory_path"]),
        "research_memory_summary_path": str(paths["summary_path"]),
    }


def load_research_memory() -> dict[str, Any]:
    paths = _research_memory_paths()
    payload = _safe_json(paths["memory_path"])
    tickers = payload.get("tickers") if isinstance(payload.get("tickers"), dict) else {}
    memory = _default_memory()
    memory["generated_at"] = str(payload.get("generated_at") or "")
    memory["last_updated_campaign_run_id"] = str(payload.get("last_updated_campaign_run_id") or "")
    memory["updated_tickers"] = [str(token) for token in (payload.get("updated_tickers") or []) if str(token).strip()]
    memory["tickers"] = {
        str(ticker).upper(): entry
        for ticker, entry in tickers.items()
        if str(ticker).strip() and isinstance(entry, dict)
    }
    memory["ticker_count"] = len(memory["tickers"])
    return memory


def build_ticker_delta(prev_entry: dict[str, Any], new_entry: dict[str, Any]) -> dict[str, Any]:
    baseline_exists = bool(str(prev_entry.get("campaign_run_id") or "").strip())
    delta = {
        "gate_change": _ordered_change(
            prev_entry.get("value_gate_status"),
            new_entry.get("value_gate_status"),
            _GATE_ORDER,
            improvement_label=GATE_UPGRADE,
            deterioration_label=GATE_DOWNGRADE,
        ),
        "lane_change": _ordered_change(
            prev_entry.get("priority_lane"),
            new_entry.get("priority_lane"),
            _LANE_ORDER,
            improvement_label=LANE_UPGRADE,
            deterioration_label=LANE_DOWNGRADE,
        ),
        "implied_return_change": _numeric_change(
            prev_entry.get("implied_return_base"),
            new_entry.get("implied_return_base"),
            baseline_exists=baseline_exists,
        ),
        "mos_epv_change": _numeric_change(
            prev_entry.get("mos_epv"),
            new_entry.get("mos_epv"),
            baseline_exists=baseline_exists,
        ),
        "blocker_change": _blocker_change(
            prev_entry.get("primary_blocker"),
            new_entry.get("primary_blocker"),
            baseline_exists=baseline_exists,
        ),
        "escalation_effect": ESCALATION_NOT_APPLICABLE,
    }
    delta["escalation_effect"] = _escalation_effect(new_entry, delta, baseline_exists=baseline_exists)
    return delta


def compute_memory_priority(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return _empty_memory_priority()

    latest = entry.get("latest") if isinstance(entry.get("latest"), dict) else {}
    delta = entry.get("delta_latest") if isinstance(entry.get("delta_latest"), dict) else {}
    appearances = max(int(entry.get("appearances_count") or 0), len(_history_rows(entry)))
    latest_gate = _token(latest.get("value_gate_status"))
    blocker_change = delta.get("blocker_change") if isinstance(delta.get("blocker_change"), dict) else {}
    implied_direction = str((delta.get("implied_return_change") or {}).get("direction") or GATE_UNCHANGED)
    mos_direction = str((delta.get("mos_epv_change") or {}).get("direction") or GATE_UNCHANGED)

    repeated_survivor_score = 0
    improvement_score = 0
    blocker_resolution_score = 0
    deterioration_penalty = 0
    reason_codes: list[str] = []

    if latest_gate in {"PASS", "WATCH"}:
        if appearances >= 3:
            repeated_survivor_score = 2
            reason_codes.append("REPEATED_SURVIVOR_3_PLUS")
        elif appearances >= 2:
            repeated_survivor_score = 1
            reason_codes.append("REPEATED_SURVIVOR_2_PLUS")

    if delta.get("gate_change") == GATE_UPGRADE:
        improvement_score += 2
        reason_codes.append("GATE_UPGRADE")
    if delta.get("lane_change") == LANE_UPGRADE:
        improvement_score += 2
        reason_codes.append("LANE_UPGRADE")
    if implied_direction == "UNKNOWN_TO_KNOWN":
        improvement_score += 2
        reason_codes.append("IMPLIED_RETURN_UNKNOWN_TO_KNOWN")
    if mos_direction == "UNKNOWN_TO_KNOWN":
        improvement_score += 1
        reason_codes.append("MOS_EPV_UNKNOWN_TO_KNOWN")

    blocker_before = _token(blocker_change.get("from"))
    blocker_after = _token(blocker_change.get("to"))
    if blocker_before not in {UNKNOWN, NONE} and blocker_after == NONE:
        blocker_resolution_score += 3
        reason_codes.append("BLOCKER_CLEARED")
    elif _blocker_class(blocker_before) == "TERMINAL" and _blocker_class(blocker_after) == "HYDRABLE":
        blocker_resolution_score += 2
        reason_codes.append("BLOCKER_TERMINAL_TO_HYDRABLE")

    if delta.get("gate_change") == GATE_DOWNGRADE:
        deterioration_penalty -= 2
        reason_codes.append("GATE_DOWNGRADE")
    if delta.get("lane_change") == LANE_DOWNGRADE:
        deterioration_penalty -= 2
        reason_codes.append("LANE_DOWNGRADE")
    if implied_direction == "KNOWN_TO_UNKNOWN":
        deterioration_penalty -= 2
        reason_codes.append("IMPLIED_RETURN_KNOWN_TO_UNKNOWN")

    repeated_blocker = _repeated_blocker_code(entry)
    if repeated_blocker:
        deterioration_penalty -= 1
        if _blocker_class(repeated_blocker) == "TERMINAL":
            reason_codes.append("RECURRING_TERMINAL_BLOCKER")
        else:
            reason_codes.append("RECURRING_UNCHANGED_BLOCKER_3_PLUS")

    escalation_effect = str(delta.get("escalation_effect") or ESCALATION_NOT_APPLICABLE)
    if escalation_effect == ESCALATION_IMPROVED:
        reason_codes.append("ESCALATION_IMPROVED")
    elif escalation_effect == ESCALATION_DETERIORATED:
        reason_codes.append("ESCALATION_DETERIORATED")

    total = _clamp_int(
        repeated_survivor_score + improvement_score + blocker_resolution_score + deterioration_penalty,
        -10,
        10,
    )
    return {
        "repeated_survivor_score": repeated_survivor_score,
        "improvement_score": improvement_score,
        "blocker_resolution_score": blocker_resolution_score,
        "deterioration_penalty": deterioration_penalty,
        "memory_priority_total": total,
        "memory_priority_reason_codes": _dedupe_refs(reason_codes),
        "source_campaign_run_ids": _memory_source_campaign_run_ids(entry),
    }


def _improvement_score(entry: dict[str, Any]) -> int:
    delta = entry.get("delta_latest") if isinstance(entry.get("delta_latest"), dict) else {}
    score = 0
    if delta.get("gate_change") == GATE_UPGRADE:
        score += 4
    if delta.get("lane_change") == LANE_UPGRADE:
        score += 3
    if (delta.get("implied_return_change") or {}).get("direction") in {"UP", "UNKNOWN_TO_KNOWN"}:
        score += 2
    if (delta.get("mos_epv_change") or {}).get("direction") in {"UP", "UNKNOWN_TO_KNOWN"}:
        score += 1
    if (delta.get("blocker_change") or {}).get("classification") == ESCALATION_IMPROVED:
        score += 1
    if delta.get("escalation_effect") == ESCALATION_IMPROVED:
        score += 1
    return score


def _deterioration_score(entry: dict[str, Any]) -> int:
    delta = entry.get("delta_latest") if isinstance(entry.get("delta_latest"), dict) else {}
    score = 0
    if delta.get("gate_change") == GATE_DOWNGRADE:
        score += 4
    if delta.get("lane_change") == LANE_DOWNGRADE:
        score += 3
    if (delta.get("implied_return_change") or {}).get("direction") in {"DOWN", "KNOWN_TO_UNKNOWN"}:
        score += 2
    if (delta.get("mos_epv_change") or {}).get("direction") in {"DOWN", "KNOWN_TO_UNKNOWN"}:
        score += 1
    if (delta.get("blocker_change") or {}).get("classification") == ESCALATION_DETERIORATED:
        score += 1
    if delta.get("escalation_effect") == ESCALATION_DETERIORATED:
        score += 1
    return score


def _summary_row(entry: dict[str, Any]) -> dict[str, Any]:
    delta = entry.get("delta_latest") if isinstance(entry.get("delta_latest"), dict) else {}
    latest = entry.get("latest") if isinstance(entry.get("latest"), dict) else {}
    memory_priority = entry.get("memory_priority") if isinstance(entry.get("memory_priority"), dict) else compute_memory_priority(entry)
    return {
        "ticker": str(entry.get("ticker") or ""),
        "campaign_run_id": str(latest.get("campaign_run_id") or entry.get("last_seen_run_id") or ""),
        "gate_change": str(delta.get("gate_change") or GATE_UNCHANGED),
        "lane_change": str(delta.get("lane_change") or LANE_UNCHANGED),
        "implied_return_direction": str((delta.get("implied_return_change") or {}).get("direction") or GATE_UNCHANGED),
        "mos_epv_direction": str((delta.get("mos_epv_change") or {}).get("direction") or GATE_UNCHANGED),
        "blocker_change": str((delta.get("blocker_change") or {}).get("classification") or GATE_UNCHANGED),
        "escalation_effect": str(delta.get("escalation_effect") or ESCALATION_NOT_APPLICABLE),
        "priority_lane": str(latest.get("priority_lane") or UNKNOWN),
        "primary_blocker": str(latest.get("primary_blocker") or UNKNOWN),
        "latest_value_gate_status": str(latest.get("value_gate_status") or UNKNOWN),
        "implied_return_base": latest.get("implied_return_base", UNKNOWN),
        "appearances_count": int(entry.get("appearances_count") or 0),
        "memory_priority_total": int(memory_priority.get("memory_priority_total") or 0),
        "memory_priority_reason_codes": [
            str(code)
            for code in (memory_priority.get("memory_priority_reason_codes") or [])
            if str(code).strip()
        ],
        "source_campaign_run_ids": [
            str(run_id)
            for run_id in (memory_priority.get("source_campaign_run_ids") or [])
            if str(run_id).strip()
        ],
    }


def _build_research_memory_summary(memory: dict[str, Any]) -> dict[str, Any]:
    paths = _research_memory_paths()
    tickers = memory.get("tickers") if isinstance(memory.get("tickers"), dict) else {}
    current_run_id = str(memory.get("last_updated_campaign_run_id") or "")
    all_entries = [entry for entry in tickers.values() if isinstance(entry, dict)]
    updated_entries = [
        entry
        for entry in all_entries
        if isinstance(entry, dict) and str(entry.get("last_seen_run_id") or "") == current_run_id
    ]
    newly_added_count = len([entry for entry in updated_entries if int(entry.get("appearances_count") or 0) <= 1])
    upgraded_gate_count = len(
        [entry for entry in updated_entries if ((entry.get("delta_latest") or {}).get("gate_change") == GATE_UPGRADE)]
    )
    upgraded_lane_count = len(
        [entry for entry in updated_entries if ((entry.get("delta_latest") or {}).get("lane_change") == LANE_UPGRADE)]
    )
    unknown_to_known_implied_return_count = len(
        [
            entry
            for entry in updated_entries
            if ((entry.get("delta_latest") or {}).get("implied_return_change") or {}).get("direction") == "UNKNOWN_TO_KNOWN"
        ]
    )
    blocker_changes_count = len(
        [
            entry
            for entry in updated_entries
            if bool(((entry.get("delta_latest") or {}).get("blocker_change") or {}).get("changed"))
        ]
    )

    improvers = sorted(
        [_summary_row(entry) for entry in updated_entries if _improvement_score(entry) > 0],
        key=lambda row: (
            -(
                (4 if row["gate_change"] == GATE_UPGRADE else 0)
                + (3 if row["lane_change"] == LANE_UPGRADE else 0)
                + (2 if row["implied_return_direction"] in {"UP", "UNKNOWN_TO_KNOWN"} else 0)
                + (1 if row["mos_epv_direction"] in {"UP", "UNKNOWN_TO_KNOWN"} else 0)
                + (1 if row["blocker_change"] == ESCALATION_IMPROVED else 0)
                + (1 if row["escalation_effect"] == ESCALATION_IMPROVED else 0)
            ),
            row["ticker"],
        ),
    )[:20]
    deteriorations = sorted(
        [_summary_row(entry) for entry in updated_entries if _deterioration_score(entry) > 0],
        key=lambda row: (
            -(
                (4 if row["gate_change"] == GATE_DOWNGRADE else 0)
                + (3 if row["lane_change"] == LANE_DOWNGRADE else 0)
                + (2 if row["implied_return_direction"] in {"DOWN", "KNOWN_TO_UNKNOWN"} else 0)
                + (1 if row["mos_epv_direction"] in {"DOWN", "KNOWN_TO_UNKNOWN"} else 0)
                + (1 if row["blocker_change"] == ESCALATION_DETERIORATED else 0)
                + (1 if row["escalation_effect"] == ESCALATION_DETERIORATED else 0)
            ),
            row["ticker"],
        ),
    )[:20]

    all_rows = [_summary_row(entry) for entry in all_entries]
    top_memory_priority_candidates = sorted(
        [row for row in all_rows if int(row.get("memory_priority_total") or 0) > 0],
        key=lambda row: (
            -int(row.get("memory_priority_total") or 0),
            -int(row.get("appearances_count") or 0),
            str(row.get("ticker") or ""),
        ),
    )[:20]
    top_stalled_names = sorted(
        [
            row
            for row in all_rows
            if int(row.get("memory_priority_total") or 0) < 0
            or any(code in _STALLED_MEMORY_REASONS for code in (row.get("memory_priority_reason_codes") or []))
        ],
        key=lambda row: (
            int(row.get("memory_priority_total") or 0),
            -int(row.get("appearances_count") or 0),
            str(row.get("ticker") or ""),
        ),
    )[:20]
    unknown_to_known_valuation_conversions = sorted(
        [row for row in all_rows if str(row.get("implied_return_direction") or "") == "UNKNOWN_TO_KNOWN"],
        key=lambda row: (
            -int(row.get("memory_priority_total") or 0),
            str(row.get("ticker") or ""),
        ),
    )[:20]

    recurring_blockers: dict[str, int] = {}
    recurring_blocker_names = []
    for entry in all_entries:
        latest = entry.get("latest") if isinstance(entry.get("latest"), dict) else {}
        blocker = _token(latest.get("primary_blocker"))
        if blocker in {UNKNOWN, NONE}:
            continue
        recurring_blockers[blocker] = recurring_blockers.get(blocker, 0) + 1
        recurring_blocker_names.append(
            {
                "ticker": str(entry.get("ticker") or ""),
                "primary_blocker": blocker,
                "appearances_count": int(entry.get("appearances_count") or 0),
                "memory_priority_total": int(
                    (
                        entry.get("memory_priority")
                        if isinstance(entry.get("memory_priority"), dict)
                        else compute_memory_priority(entry)
                    ).get("memory_priority_total")
                    or 0
                ),
            }
        )

    return {
        "campaign_run_id": current_run_id,
        "generated_at": utc_now_iso(),
        "ticker_count": len(tickers),
        "newly_added_count": newly_added_count,
        "upgraded_gate_count": upgraded_gate_count,
        "upgraded_lane_count": upgraded_lane_count,
        "unknown_to_known_implied_return_count": unknown_to_known_implied_return_count,
        "blocker_changes_count": blocker_changes_count,
        "top_improvers": improvers,
        "top_deteriorations": deteriorations,
        "top_memory_priority_candidates": top_memory_priority_candidates,
        "top_stalled_names": top_stalled_names,
        "unknown_to_known_valuation_conversions": unknown_to_known_valuation_conversions,
        "recurring_blockers": dict(sorted(recurring_blockers.items(), key=lambda item: item[0])),
        "recurring_blocker_names": sorted(
            recurring_blocker_names,
            key=lambda row: (
                -int(row.get("appearances_count") or 0),
                str(row.get("primary_blocker") or ""),
                str(row.get("ticker") or ""),
            ),
        )[:20],
        "research_memory_path": str(paths["memory_path"]),
        "research_memory_summary_path": str(paths["summary_path"]),
    }


def write_research_memory(memory: dict[str, Any]) -> str:
    paths = _research_memory_paths()
    normalized_tickers: dict[str, dict[str, Any]] = {}
    for ticker, entry in (
        (memory.get("tickers") or {}) if isinstance(memory.get("tickers"), dict) else {}
    ).items():
        ticker_norm = str(ticker).upper()
        if not ticker_norm or not isinstance(entry, dict):
            continue
        normalized_entry = dict(entry)
        memory_priority = compute_memory_priority(normalized_entry)
        normalized_entry["memory_priority"] = memory_priority
        normalized_entry["memory_priority_total"] = int(memory_priority.get("memory_priority_total") or 0)
        normalized_entry["memory_priority_reason_codes"] = list(memory_priority.get("memory_priority_reason_codes") or [])
        normalized_entry["memory_priority_source_campaign_run_ids"] = list(memory_priority.get("source_campaign_run_ids") or [])
        normalized_tickers[ticker_norm] = normalized_entry
    payload = _default_memory()
    payload.update(
        {
            "generated_at": str(memory.get("generated_at") or utc_now_iso()),
            "last_updated_campaign_run_id": str(memory.get("last_updated_campaign_run_id") or ""),
            "updated_tickers": [str(token) for token in (memory.get("updated_tickers") or []) if str(token).strip()],
            "tickers": dict(
                sorted(
                    normalized_tickers.items(),
                    key=lambda item: item[0],
                )
            ),
        }
    )
    payload["ticker_count"] = len(payload["tickers"])
    _json_write(paths["memory_path"], payload)
    _json_write(paths["summary_path"], _build_research_memory_summary(payload))
    return str(paths["memory_path"])


def update_research_memory_from_campaign(
    campaign_run_id: str,
    campaign_summary: dict[str, Any],
    master_shortlist: dict[str, Any],
    master_watchlist_state: dict[str, Any],
    promotion_state: dict[str, Any],
    escalation_results: dict[str, Any] | None = None,
    director_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memory = load_research_memory()
    tickers = memory.get("tickers") if isinstance(memory.get("tickers"), dict) else {}
    shortlist_lookup = {
        str(row.get("ticker") or "").upper(): row
        for row in (master_shortlist.get("rows") or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    promotion_lookup = {
        str(row.get("ticker") or "").upper(): row
        for row in (promotion_state.get("rows") or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    watch_lookup = master_watchlist_state.get("tickers") if isinstance(master_watchlist_state.get("tickers"), dict) else {}
    escalation_rows = [row for row in ((escalation_results or {}).get("rows") or []) if isinstance(row, dict)]
    escalation_lookup: dict[str, list[dict[str, Any]]] = {}
    escalation_results_path = str((escalation_results or {}).get("escalation_results_path") or "")
    for row in escalation_rows:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        item = dict(row)
        if escalation_results_path:
            item["escalation_results_path"] = escalation_results_path
        escalation_lookup.setdefault(ticker, []).append(item)

    source_paths = _campaign_source_paths(campaign_summary, escalation_results or {})
    ticker_universe = sorted(set(shortlist_lookup) | set(promotion_lookup) | set(watch_lookup))
    updated_tickers: list[str] = []
    for ticker in ticker_universe:
        shortlist_row = shortlist_lookup.get(ticker, {})
        promotion_row = promotion_lookup.get(ticker, {})
        watch_row = watch_lookup.get(ticker) if isinstance(watch_lookup.get(ticker), dict) else {}
        snapshot = _ticker_snapshot(
            ticker=ticker,
            campaign_run_id=campaign_run_id,
            campaign_summary={**campaign_summary, "escalation_results_path": source_paths["escalation_results_path"]},
            shortlist_row=shortlist_row,
            watch_row=watch_row,
            promotion_row=promotion_row,
            escalation_rows=escalation_lookup.get(ticker, []),
            director_metadata=_resolve_director_metadata(
                shortlist_row=shortlist_row,
                promotion_row=promotion_row,
                watch_row=watch_row,
                director_metadata=director_metadata,
            ),
        )
        snapshot["source_paths"] = source_paths
        existing = tickers.get(ticker) if isinstance(tickers.get(ticker), dict) else {}
        history = [row for row in (existing.get("history") or []) if isinstance(row, dict)]
        history = [row for row in history if str(row.get("campaign_run_id") or "") != campaign_run_id]
        prev_entry = history[-1] if history else {}
        history_entry = _history_entry(snapshot)
        delta_latest = build_ticker_delta(prev_entry, {**history_entry, "escalation_rows": escalation_lookup.get(ticker, [])})
        history.append(history_entry)
        tickers[ticker] = {
            "ticker": ticker,
            "first_seen_run_id": str(existing.get("first_seen_run_id") or campaign_run_id),
            "last_seen_run_id": campaign_run_id,
            "appearances_count": len(history),
            "latest": _latest_entry(snapshot),
            "history": history,
            "delta_latest": delta_latest,
        }
        updated_tickers.append(ticker)

    memory["generated_at"] = utc_now_iso()
    memory["last_updated_campaign_run_id"] = campaign_run_id
    memory["updated_tickers"] = updated_tickers
    memory["tickers"] = tickers
    write_research_memory(memory)
    summary = _safe_json(_research_memory_paths()["summary_path"]) or _empty_memory_summary()
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "research_memory_path": str(_research_memory_paths()["memory_path"]),
        "research_memory_summary_path": str(_research_memory_paths()["summary_path"]),
        "ticker_count": int(summary.get("ticker_count") or 0),
        "newly_added_count": int(summary.get("newly_added_count") or 0),
        "upgraded_gate_count": int(summary.get("upgraded_gate_count") or 0),
        "upgraded_lane_count": int(summary.get("upgraded_lane_count") or 0),
        "unknown_to_known_implied_return_count": int(summary.get("unknown_to_known_implied_return_count") or 0),
        "blocker_changes_count": int(summary.get("blocker_changes_count") or 0),
        "top_improvers": summary.get("top_improvers") if isinstance(summary.get("top_improvers"), list) else [],
        "top_deteriorations": summary.get("top_deteriorations") if isinstance(summary.get("top_deteriorations"), list) else [],
        "top_memory_priority_candidates": summary.get("top_memory_priority_candidates")
        if isinstance(summary.get("top_memory_priority_candidates"), list)
        else [],
        "updated_tickers": updated_tickers,
    }


def open_research_memory(ticker: str | None = None) -> dict[str, Any]:
    paths = _research_memory_paths()
    memory = load_research_memory()
    summary = _safe_json(paths["summary_path"])
    if not memory.get("tickers"):
        payload = _empty_memory_summary()
        payload["status"] = "MISSING"
        return payload
    if ticker:
        ticker_norm = str(ticker).strip().upper()
        entry = memory["tickers"].get(ticker_norm) if isinstance(memory["tickers"], dict) else None
        if not isinstance(entry, dict):
            return {
                "status": "MISSING",
                "ticker": ticker_norm,
                "research_memory_path": str(paths["memory_path"]),
                "research_memory_summary_path": str(paths["summary_path"]),
            }
        return {
            "status": "OK",
            "ticker": ticker_norm,
            "entry": entry,
            "research_memory_path": str(paths["memory_path"]),
            "research_memory_summary_path": str(paths["summary_path"]),
        }
    payload = summary if summary else _build_research_memory_summary(memory)
    payload["status"] = "OK"
    payload["research_memory_path"] = str(paths["memory_path"])
    payload["research_memory_summary_path"] = str(paths["summary_path"])
    return payload


def open_research_memory_priority() -> dict[str, Any]:
    paths = _research_memory_paths()
    summary = _safe_json(paths["summary_path"]) or _build_research_memory_summary(load_research_memory())
    if not summary.get("ticker_count"):
        payload = _empty_memory_summary()
        payload["status"] = "MISSING"
        return payload
    return {
        "status": "OK",
        "ticker_count": int(summary.get("ticker_count") or 0),
        "top_memory_priority_candidates": summary.get("top_memory_priority_candidates")
        if isinstance(summary.get("top_memory_priority_candidates"), list)
        else [],
        "top_stalled_names": summary.get("top_stalled_names")
        if isinstance(summary.get("top_stalled_names"), list)
        else [],
        "recurring_blockers": summary.get("recurring_blockers")
        if isinstance(summary.get("recurring_blockers"), dict)
        else {},
        "recurring_blocker_names": summary.get("recurring_blocker_names")
        if isinstance(summary.get("recurring_blocker_names"), list)
        else [],
        "unknown_to_known_valuation_conversions": summary.get("unknown_to_known_valuation_conversions")
        if isinstance(summary.get("unknown_to_known_valuation_conversions"), list)
        else [],
        "research_memory_path": str(paths["memory_path"]),
        "research_memory_summary_path": str(paths["summary_path"]),
    }


def diff_research_memory(ticker: str) -> dict[str, Any]:
    paths = _research_memory_paths()
    memory = load_research_memory()
    ticker_norm = str(ticker).strip().upper()
    entry = memory.get("tickers", {}).get(ticker_norm) if isinstance(memory.get("tickers"), dict) else None
    if not isinstance(entry, dict):
        return {
            "status": "MISSING",
            "ticker": ticker_norm,
            "research_memory_path": str(paths["memory_path"]),
            "research_memory_summary_path": str(paths["summary_path"]),
        }
    history = [row for row in (entry.get("history") or []) if isinstance(row, dict)]
    latest = history[-1] if history else {}
    prior = history[-2] if len(history) >= 2 else {}
    source_paths = latest.get("source_paths") if isinstance(latest.get("source_paths"), dict) else {}
    return {
        "status": "OK",
        "ticker": ticker_norm,
        "latest_state": entry.get("latest") if isinstance(entry.get("latest"), dict) else latest,
        "prior_state": prior,
        "delta_latest": entry.get("delta_latest") if isinstance(entry.get("delta_latest"), dict) else {},
        "memory_priority": entry.get("memory_priority") if isinstance(entry.get("memory_priority"), dict) else compute_memory_priority(entry),
        "source_paths": source_paths,
        "research_memory_path": str(paths["memory_path"]),
        "research_memory_summary_path": str(paths["summary_path"]),
    }
