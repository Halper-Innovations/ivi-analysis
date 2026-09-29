from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.universe.facts_blockers import enrich_facts_blocker_fields
from app.valuation.lineage import latest_decision_eligible_valuation_row


UNKNOWN = "UNKNOWN"

LANE_1_HIGH_PRIORITY = "LANE_1_HIGH_PRIORITY"
LANE_2_RESEARCH_QUEUE = "LANE_2_RESEARCH_QUEUE"
LANE_3_MONITOR = "LANE_3_MONITOR"
LANE_4_DEPRIORITIZED = "LANE_4_DEPRIORITIZED"

_LANE_ORDER = {
    LANE_1_HIGH_PRIORITY: 0,
    LANE_2_RESEARCH_QUEUE: 1,
    LANE_3_MONITOR: 2,
    LANE_4_DEPRIORITIZED: 3,
}
_GATE_ORDER = {"PASS": 0, "WATCH": 1, "FAIL": 2}
_NONE_BLOCKERS = {"", "NONE", UNKNOWN}


def _promotion_paths(campaign_run_id: str) -> dict[str, Path]:
    root = get_config().campaigns_dir / campaign_run_id
    return {
        "root": root,
        "promotion_state_path": root / "promotion_state.json",
        "promotion_candidates_path": root / "promotion_candidates.json",
        "priority_lanes_path": root / "priority_lanes.json",
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


def _variant_confidence_rank(value: Any) -> int:
    token = str(value or "NONE").upper()
    if token == "HIGH":
        return 0
    if token == "MEDIUM":
        return 1
    if token == "LOW":
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


def _thresholds_effective() -> dict[str, Any]:
    cfg = get_config()
    return {
        "promotion_min_appearances_high_priority": int(cfg.promotion_min_appearances_high_priority),
        "promotion_min_implied_return": float(cfg.promotion_min_implied_return),
        "promotion_terminal_blocker_codes": [
            str(token).upper() for token in cfg.promotion_terminal_blocker_codes
        ],
        "promotion_lane2_min_score": float(cfg.promotion_lane2_min_score),
    }


def _status_rank(status: str) -> int:
    return _GATE_ORDER.get(str(status or "").upper(), 3)


def _lane_rank(lane: str) -> int:
    return _LANE_ORDER.get(str(lane or ""), len(_LANE_ORDER))


def _load_memory_lookup() -> dict[str, dict[str, Any]]:
    from app.universe.research_memory import load_research_memory

    memory = load_research_memory()
    return memory.get("tickers") if isinstance(memory.get("tickers"), dict) else {}


def _memory_priority_fields(
    ticker: str, memory_lookup: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    entry = memory_lookup.get(str(ticker).upper()) if isinstance(memory_lookup, dict) else None
    if not isinstance(entry, dict):
        return {
            "memory_priority_total": 0,
            "memory_priority_reason_codes": [],
            "memory_priority_source_campaign_run_ids": [],
        }
    memory_priority = (
        entry.get("memory_priority") if isinstance(entry.get("memory_priority"), dict) else {}
    )
    return {
        "memory_priority_total": int(
            memory_priority.get("memory_priority_total") or entry.get("memory_priority_total") or 0
        ),
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


def _default_l4_signal_fields() -> dict[str, Any]:
    return {
        "variant_perception_count": 0,
        "variant_perception_max_confidence": "NONE",
        "variant_perception_direction": "NONE",
        "variant_signal_source_count": 0,
        "tech_category": "TRADITIONAL_OPERATING",
        "tech_valuation_divergence": UNKNOWN,
        "filing_diff_high_materiality_count": 0,
        "pattern_hit_count": 0,
        "pattern_confirmed_count": 0,
        "_variant_supporting_sources": [],
    }


def _normalize_divergence_pct(value: Any) -> Any:
    if not _is_num(value):
        return UNKNOWN
    divergence = float(value)
    if abs(divergence) <= 1.5:
        divergence *= 100.0
    return divergence


def _ensure_l4_run_context(run_id: str, context: dict[str, Any]) -> dict[str, Any]:
    run_cache = context.get("run_cache") if isinstance(context.get("run_cache"), dict) else {}
    if run_id in run_cache and isinstance(run_cache[run_id], dict):
        return run_cache[run_id]

    cfg = context.get("cfg") if hasattr(context.get("cfg"), "outputs_dir") else get_config()
    universe_root = cfg.outputs_dir / "universe" / str(run_id).strip()
    autopilot_state = _safe_json(universe_root / "autopilot" / "autopilot_state.json")
    as_of_date = str(autopilot_state.get("as_of_date") or "").strip()
    run_ctx = {
        "run_id": str(run_id).strip(),
        "universe_root": universe_root,
        "as_of_date": as_of_date,
        "variant_dir": universe_root / "variant_perceptions",
        "variant_cache": {},
        "pattern_report_path": universe_root / "pattern_scan" / "pattern_scan_report.json",
        "pattern_lookup": None,
        "filing_diff_dir": universe_root / "filing_diffs",
        "filing_diff_cache": {},
        "tech_adjustment_cache": {},
    }
    run_cache[str(run_id).strip()] = run_ctx
    context["run_cache"] = run_cache
    return run_ctx


def _load_variant_payload_for_ticker(
    ticker: str, run_ctx: dict[str, Any], cfg: Any
) -> dict[str, Any]:
    cache = run_ctx.get("variant_cache") if isinstance(run_ctx.get("variant_cache"), dict) else {}
    ticker_norm = str(ticker or "").strip().upper()
    if ticker_norm in cache:
        return cache[ticker_norm] if isinstance(cache[ticker_norm], dict) else {}

    candidates: list[Path] = []
    as_of_date = str(run_ctx.get("as_of_date") or "").strip()
    variant_dir = (
        run_ctx.get("variant_dir") if isinstance(run_ctx.get("variant_dir"), Path) else None
    )
    if variant_dir is not None and variant_dir.exists():
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
        candidates.append(
            cfg.outputs_dir / "variant_perceptions" / f"{ticker_norm}_{as_of_date}.json"
        )

    payload: dict[str, Any] = {}
    for path in candidates:
        payload = _safe_json(path)
        if payload:
            break
    cache[ticker_norm] = payload
    run_ctx["variant_cache"] = cache
    return payload


def _pattern_lookup_for_run(run_ctx: dict[str, Any], cfg: Any) -> dict[str, dict[str, int]]:
    if isinstance(run_ctx.get("pattern_lookup"), dict):
        return run_ctx["pattern_lookup"]

    report_path = (
        run_ctx.get("pattern_report_path")
        if isinstance(run_ctx.get("pattern_report_path"), Path)
        else None
    )
    payload = (
        _safe_json(report_path) if isinstance(report_path, Path) and report_path.exists() else {}
    )
    if not payload:
        fallback = (
            cfg.outputs_dir
            / "patterns"
            / f"{str(run_ctx.get('run_id') or '').strip()}_pattern_scan.json"
        )
        payload = _safe_json(fallback)
    lookup: dict[str, dict[str, int]] = {}
    for result in [row for row in (payload.get("pattern_results") or []) if isinstance(row, dict)]:
        for hit in [row for row in (result.get("hits") or []) if isinstance(row, dict)]:
            ticker = str(hit.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            entry = lookup.setdefault(
                ticker, {"pattern_hit_count": 0, "pattern_confirmed_count": 0}
            )
            entry["pattern_hit_count"] += 1
            if hit.get("outcome_confirmed") is True:
                entry["pattern_confirmed_count"] += 1
    run_ctx["pattern_lookup"] = lookup
    return lookup


def _load_filing_diff_payload_for_ticker(
    ticker: str, run_ctx: dict[str, Any], cfg: Any
) -> dict[str, Any]:
    cache = (
        run_ctx.get("filing_diff_cache")
        if isinstance(run_ctx.get("filing_diff_cache"), dict)
        else {}
    )
    ticker_norm = str(ticker or "").strip().upper()
    if ticker_norm in cache:
        return cache[ticker_norm] if isinstance(cache[ticker_norm], dict) else {}

    candidates: list[Path] = []
    filing_diff_dir = (
        run_ctx.get("filing_diff_dir") if isinstance(run_ctx.get("filing_diff_dir"), Path) else None
    )
    if filing_diff_dir is not None:
        candidates.append(filing_diff_dir / f"{ticker_norm}.json")
    candidates.append(
        cfg.outputs_dir
        / "diffs"
        / f"{ticker_norm}_{str(run_ctx.get('run_id') or '').strip()}_diff.json"
    )

    payload: dict[str, Any] = {}
    for path in candidates:
        payload = _safe_json(path)
        if payload:
            break
    cache[ticker_norm] = payload
    run_ctx["filing_diff_cache"] = cache
    return payload


def _load_tech_adjustment_payload_for_ticker(
    ticker: str, run_ctx: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    cache = (
        run_ctx.get("tech_adjustment_cache")
        if isinstance(run_ctx.get("tech_adjustment_cache"), dict)
        else {}
    )
    ticker_norm = str(ticker or "").strip().upper()
    if ticker_norm in cache:
        return cache[ticker_norm] if isinstance(cache[ticker_norm], dict) else {}

    as_of_date = str(run_ctx.get("as_of_date") or "").strip()
    payload: dict[str, Any] = {}
    if as_of_date:
        valuation_cache = (
            context.get("valuation_cache")
            if isinstance(context.get("valuation_cache"), dict)
            else {}
        )
        cache_key = (ticker_norm, as_of_date)
        if cache_key not in valuation_cache:
            with get_db() as conn:
                row = latest_decision_eligible_valuation_row(
                    conn,
                    ticker=ticker_norm,
                    method="tech_adjustment",
                    as_of_date=as_of_date,
                    exact_as_of_date=True,
                )
            if row:
                try:
                    valuation_cache[cache_key] = json.loads(row["outputs_json"] or "{}")
                except Exception:
                    valuation_cache[cache_key] = {}
            else:
                valuation_cache[cache_key] = {}
            context["valuation_cache"] = valuation_cache
        payload = (
            valuation_cache.get(cache_key)
            if isinstance(valuation_cache.get(cache_key), dict)
            else {}
        )

    cache[ticker_norm] = payload
    run_ctx["tech_adjustment_cache"] = cache
    return payload


def _load_l4_signals_for_ticker(
    ticker: str,
    run_id: str,
    campaign_run_id: str,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del campaign_run_id
    signal_payload = _default_l4_signal_fields()
    ctx = (
        context
        if isinstance(context, dict)
        else {"cfg": get_config(), "run_cache": {}, "valuation_cache": {}}
    )
    run_ctx = _ensure_l4_run_context(run_id, ctx)
    cfg = ctx.get("cfg") if hasattr(ctx.get("cfg"), "outputs_dir") else get_config()
    ticker_norm = str(ticker or "").strip().upper()

    variant_payload = _load_variant_payload_for_ticker(ticker_norm, run_ctx, cfg)
    perceptions = [
        row for row in (variant_payload.get("perceptions") or []) if isinstance(row, dict)
    ]
    if perceptions:
        signal_payload["variant_perception_count"] = len(perceptions)
        confidences = [
            str(row.get("confidence") or "LOW").upper()
            for row in perceptions
            if str(row.get("confidence") or "").strip()
        ]
        best_conf = min(confidences, key=_variant_confidence_rank) if confidences else "NONE"
        signal_payload["variant_perception_max_confidence"] = best_conf
        directions = {
            str(row.get("direction") or "").upper()
            for row in perceptions
            if str(row.get("direction") or "").strip()
        }
        if len(directions) == 1:
            signal_payload["variant_perception_direction"] = next(iter(directions))
        elif directions:
            signal_payload["variant_perception_direction"] = "MIXED"
        signal_payload["variant_signal_source_count"] = max(
            (
                len(
                    {
                        str(signal.get("source") or "").upper()
                        for signal in (perception.get("supporting_signals") or [])
                        if isinstance(signal, dict) and str(signal.get("source") or "").strip()
                    }
                )
                for perception in perceptions
            ),
            default=0,
        )
        supporting_sources: set[str] = set()
        for perception in perceptions:
            if str(perception.get("confidence") or "").upper() != best_conf:
                continue
            for signal in [
                row for row in (perception.get("supporting_signals") or []) if isinstance(row, dict)
            ]:
                source = str(signal.get("source") or "").upper()
                if source:
                    supporting_sources.add(source)
        signal_payload["_variant_supporting_sources"] = sorted(supporting_sources)

    diff_payload = _load_filing_diff_payload_for_ticker(ticker_norm, run_ctx, cfg)
    if diff_payload:
        signal_payload["filing_diff_high_materiality_count"] = len(
            [
                change
                for change in (diff_payload.get("changes") or [])
                if isinstance(change, dict)
                and str(change.get("materiality") or "").upper() == "HIGH"
            ]
        )

    pattern_lookup = _pattern_lookup_for_run(run_ctx, cfg)
    pattern_payload = (
        pattern_lookup.get(ticker_norm) if isinstance(pattern_lookup.get(ticker_norm), dict) else {}
    )
    if pattern_payload:
        signal_payload["pattern_hit_count"] = int(pattern_payload.get("pattern_hit_count") or 0)
        signal_payload["pattern_confirmed_count"] = int(
            pattern_payload.get("pattern_confirmed_count") or 0
        )

    tech_adjustment = _load_tech_adjustment_payload_for_ticker(ticker_norm, run_ctx, ctx)
    if tech_adjustment:
        category_payload = (
            tech_adjustment.get("category_classification")
            if isinstance(tech_adjustment.get("category_classification"), dict)
            else {}
        )
        signal_payload["tech_category"] = str(
            category_payload.get("category") or "TRADITIONAL_OPERATING"
        ).upper()
        signal_payload["tech_valuation_divergence"] = _normalize_divergence_pct(
            tech_adjustment.get("tech_valuation_divergence")
        )

    return signal_payload


def _merge_l4_signal_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged = _default_l4_signal_fields()
    valid = [payload for payload in payloads if isinstance(payload, dict)]
    if not valid:
        return merged

    merged["variant_perception_count"] = max(
        int(payload.get("variant_perception_count") or 0) for payload in valid
    )
    merged["variant_signal_source_count"] = max(
        int(payload.get("variant_signal_source_count") or 0) for payload in valid
    )
    merged["filing_diff_high_materiality_count"] = max(
        int(payload.get("filing_diff_high_materiality_count") or 0) for payload in valid
    )
    merged["pattern_hit_count"] = max(
        int(payload.get("pattern_hit_count") or 0) for payload in valid
    )
    merged["pattern_confirmed_count"] = max(
        int(payload.get("pattern_confirmed_count") or 0) for payload in valid
    )

    divergence_values = [
        payload.get("tech_valuation_divergence")
        for payload in valid
        if _is_num(payload.get("tech_valuation_divergence"))
    ]
    if divergence_values:
        merged["tech_valuation_divergence"] = max(
            (float(value) for value in divergence_values), key=lambda value: abs(value)
        )

    categories = [
        str(payload.get("tech_category") or "").upper()
        for payload in valid
        if str(payload.get("tech_category") or "").strip()
    ]
    merged["tech_category"] = next(
        (token for token in categories if token not in {"", "TRADITIONAL_OPERATING"}),
        categories[0] if categories else "TRADITIONAL_OPERATING",
    )

    best_rank = min(
        (
            _variant_confidence_rank(payload.get("variant_perception_max_confidence"))
            for payload in valid
        ),
        default=3,
    )
    best_payloads = [
        payload
        for payload in valid
        if _variant_confidence_rank(payload.get("variant_perception_max_confidence")) == best_rank
    ]
    merged["variant_perception_max_confidence"] = (
        str(best_payloads[0].get("variant_perception_max_confidence") or "NONE").upper()
        if best_payloads
        else "NONE"
    )
    directions = {
        str(payload.get("variant_perception_direction") or "").upper()
        for payload in best_payloads
        if str(payload.get("variant_perception_direction") or "").strip()
        and str(payload.get("variant_perception_direction") or "").upper() != "NONE"
    }
    if len(directions) == 1:
        merged["variant_perception_direction"] = next(iter(directions))
    elif directions:
        merged["variant_perception_direction"] = "MIXED"
    supporting_sources: set[str] = set()
    for payload in best_payloads:
        for source in payload.get("_variant_supporting_sources") or []:
            token = str(source or "").upper()
            if token:
                supporting_sources.add(token)
    merged["_variant_supporting_sources"] = sorted(supporting_sources)
    return merged


def _memory_supports_lane(row: dict[str, Any]) -> bool:
    if int(row.get("memory_priority_total") or 0) <= 0:
        return False
    reason_codes = {
        str(code) for code in (row.get("memory_priority_reason_codes") or []) if str(code).strip()
    }
    return any(
        code
        in {
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
    )


def _oe_quality_supports_lane(row: dict[str, Any]) -> bool:
    value = row.get("oe_quality_total", UNKNOWN)
    return _is_num(value) and float(value) >= 8.0


def _intangible_supports_lane(row: dict[str, Any]) -> bool:
    value = row.get("intangible_economics_total", UNKNOWN)
    blocker = _effective_blocker(row)
    return (
        _is_num(value)
        and float(value) >= 7.0
        and blocker not in _NONE_BLOCKERS
        and "TERMINAL_BLOCKER"
        not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
    )


def _owner_value_capture_supports_lane(row: dict[str, Any]) -> bool:
    value = row.get("owner_value_capture_score", UNKNOWN)
    blocker = _effective_blocker(row)
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    return (
        latest_gate in {"PASS", "WATCH"}
        and _is_num(value)
        and float(value) >= 4.0
        and blocker not in _NONE_BLOCKERS
        and "TERMINAL_BLOCKER"
        not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
    )


def _owner_value_capture_headwind(row: dict[str, Any]) -> bool:
    reason_codes = {
        str(code)
        for code in (
            list(row.get("owner_value_capture_reason_codes") or [])
            + list(row.get("oe_quality_reason_codes") or [])
        )
        if str(code).strip()
    }
    value = row.get("owner_value_capture_score", UNKNOWN)
    return (
        "EXCESS_DILUTION" in reason_codes
        or "WEAK_PER_SHARE_CAPTURE" in reason_codes
        or (_is_num(value) and float(value) <= 1.0)
    )


def _reinvestment_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    blocker = _effective_blocker(row)
    reinvestment_class = str(
        row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
    ).upper()
    return (
        latest_gate in {"PASS", "WATCH"}
        and blocker not in _NONE_BLOCKERS
        and "TERMINAL_BLOCKER"
        not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and reinvestment_class == "HIGH_REINVESTMENT_EFFICIENCY"
    )


def _reinvestment_headwind_code(row: dict[str, Any]) -> str:
    reinvestment_class = str(
        row.get("reinvestment_efficiency_class") or "REINVESTMENT_EFFICIENCY_UNKNOWN"
    ).upper()
    if reinvestment_class == "LOW_REINVESTMENT_EFFICIENCY":
        reasons = {
            str(code)
            for code in (
                list(row.get("reinvestment_efficiency_reason_codes") or [])
                + list(row.get("reinvestment_headwind_signals") or [])
            )
            if str(code).strip()
        }
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


def _accounting_quality_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    accounting_class = str(
        row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
    ).upper()
    return latest_gate in {"PASS", "WATCH"} and accounting_class == "HIGH_ACCOUNTING_QUALITY"


def _accounting_quality_headwind_code(row: dict[str, Any]) -> str:
    accounting_class = str(
        row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
    ).upper()
    reason_codes = {
        str(code)
        for code in (
            list(row.get("accounting_quality_reason_codes") or [])
            + list(row.get("cash_earnings_headwind_signals") or [])
        )
        if str(code).strip()
    }
    if accounting_class == "LOW_ACCOUNTING_QUALITY":
        if (
            "ACCRUAL_HEAVY_EARNINGS" in reason_codes
            or "ACCRUAL_HEAVY_EARNINGS_HEADWIND" in reason_codes
        ):
            return "ACCRUAL_HEAVY_EARNINGS_HEADWIND"
        if (
            "WEAK_FCF_TO_EARNINGS_CONVERSION" in reason_codes
            or "WEAK_CASH_CONVERSION_HEADWIND" in reason_codes
        ):
            return "WEAK_CASH_CONVERSION_HEADWIND"
        return "LOW_ACCOUNTING_QUALITY_HEADWIND"
    if accounting_class == "ACCOUNTING_QUALITY_UNKNOWN":
        return "ACCOUNTING_QUALITY_UNKNOWN"
    return ""


def _balance_sheet_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    stress_class = str(
        row.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN"
    ).upper()
    return latest_gate in {"PASS", "WATCH"} and stress_class == "LOW_BALANCE_SHEET_STRESS"


def _balance_sheet_headwind_code(row: dict[str, Any]) -> str:
    stress_class = str(
        row.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN"
    ).upper()
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


def _returns_persistence_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    returns_class = str(
        row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
    ).upper()
    return latest_gate in {"PASS", "WATCH"} and returns_class == "HIGH_RETURNS_PERSISTENCE"


def _returns_persistence_headwind_code(row: dict[str, Any]) -> str:
    returns_class = str(
        row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
    ).upper()
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


def _revenue_dependence_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    revenue_class = str(
        row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
    ).upper()
    return latest_gate in {"PASS", "WATCH"} and revenue_class == "LOW_REVENUE_DEPENDENCE_RISK"


def _revenue_dependence_headwind_code(row: dict[str, Any]) -> str:
    revenue_class = str(
        row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
    ).upper()
    reason_codes = {
        str(code)
        for code in (
            list(row.get("revenue_dependence_risk_reason_codes") or [])
            + list(row.get("revenue_dependence_headwind_signals") or [])
        )
        if str(code).strip()
    }
    if revenue_class == "HIGH_REVENUE_DEPENDENCE_RISK":
        if (
            "SINGLE_CUSTOMER_CONCENTRATION" in reason_codes
            or "TOP_CUSTOMER_DOMINANCE" in reason_codes
        ):
            return "CUSTOMER_CONCENTRATION_HEADWIND"
        if "NARROW_CHANNEL_DEPENDENCE" in reason_codes:
            return "CHANNEL_DEPENDENCE_HEADWIND"
        return "HIGH_REVENUE_DEPENDENCE_HEADWIND"
    if revenue_class == "REVENUE_DEPENDENCE_UNKNOWN":
        return "REVENUE_DEPENDENCE_UNKNOWN"
    return ""


def _maintenance_capex_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    credibility_class = str(
        row.get("maintenance_capex_credibility_class") or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
    ).upper()
    asset_intensity_class = str(
        row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
    ).upper()
    return (
        latest_gate in {"PASS", "WATCH"}
        and credibility_class == "HIGH_MAINTENANCE_CAPEX_CREDIBILITY"
        and asset_intensity_class == "LOW_ASSET_INTENSITY"
    )


def _maintenance_capex_headwind_code(row: dict[str, Any]) -> str:
    credibility_class = str(
        row.get("maintenance_capex_credibility_class") or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
    ).upper()
    asset_intensity_class = str(
        row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
    ).upper()
    if credibility_class == "LOW_MAINTENANCE_CAPEX_CREDIBILITY":
        return "LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND"
    if asset_intensity_class == "HIGH_ASSET_INTENSITY":
        return "HIGH_ASSET_INTENSITY_HEADWIND"
    if credibility_class == "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN":
        return "MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN"
    return ""


def _intrinsic_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    blocker = _effective_blocker(row)
    mos_to_floor = row.get("mos_to_floor", UNKNOWN)
    support_type = str(row.get("downside_support_type") or UNKNOWN)
    return (
        latest_gate in {"PASS", "WATCH"}
        and blocker not in _NONE_BLOCKERS
        and "TERMINAL_BLOCKER"
        not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and (
            (_is_num(mos_to_floor) and float(mos_to_floor) >= 0.25)
            or support_type in {"ASSET_SUPPORT", "EARNINGS_POWER_SUPPORT", "BALANCE_SHEET_SUPPORT"}
        )
    )


def _valuation_confidence_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    support_count = _int_or_zero(row.get("valuation_support_count"))
    confidence_class = str(row.get("valuation_confidence_class") or "CONFIDENCE_UNKNOWN").upper()
    fragility_status = str(row.get("valuation_fragility_status") or "FRAGILITY_UNKNOWN").upper()
    return (
        latest_gate in {"PASS", "WATCH"}
        and support_count >= 2
        and confidence_class in {"HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE"}
        and fragility_status in {"LOW_FRAGILITY", "MODERATE_FRAGILITY"}
    )


def _valuation_confidence_headwind(row: dict[str, Any]) -> bool:
    support_count = _int_or_zero(row.get("valuation_support_count"))
    convergence_status = str(row.get("valuation_convergence_status") or UNKNOWN).upper()
    fragility_status = str(row.get("valuation_fragility_status") or "FRAGILITY_UNKNOWN").upper()
    return (
        support_count <= 1
        or convergence_status == "WEAK_CONVERGENCE"
        or fragility_status == "HIGH_FRAGILITY"
    )


def _valuation_integrity_headwind(row: dict[str, Any]) -> str:
    token = str(row.get("valuation_integrity_class") or "INTEGRITY_UNKNOWN").upper()
    if token == "INTEGRITY_SUSPECT":
        return "INTEGRITY_SUSPECT_HEADWIND"
    if token == "INTEGRITY_WARNING":
        return "INTEGRITY_WARNING_HEADWIND"
    return ""


def _cyclical_risk_headwind(row: dict[str, Any]) -> str:
    token = str(row.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN").upper()
    if token == "PEAK_EARNINGS_RISK":
        return "PEAK_EARNINGS_CYCLICAL_HEADWIND"
    return ""


def _impairment_headwind_code(row: dict[str, Any]) -> str:
    """Return an impairment-related headwind code for the row, or empty string."""
    cls = str(row.get("impairment_class_primary") or "").upper()
    if cls == "CLEAR_IMPAIRMENT":
        return "IMPAIRMENT_HEADWIND"
    if cls == "PROBABLE_IMPAIRMENT":
        return "IMPAIRMENT_HEADWIND"
    if cls == "STRUCTURALLY_WEAK_NOT_IMPAIRED":
        return "STRUCTURALLY_WEAK_HEADWIND"
    return ""


def _impairment_support_code(row: dict[str, Any]) -> str:
    """Return an impairment-related support code for the row, or empty string."""
    cls = str(row.get("impairment_class_primary") or "").upper()
    if cls == "TEMPORARY_WEAKNESS":
        return "TEMPORARY_WEAKNESS_SUPPORT"
    if cls == "EVIDENCE_DEGRADED_NOT_ASSESSABLE":
        return "EVIDENCE_GAP_NOT_IMPAIRMENT"
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


def _value_type_supports_lane(row: dict[str, Any]) -> bool:
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    blocker = _effective_blocker(row)
    return (
        latest_gate in {"PASS", "WATCH"}
        and blocker not in _NONE_BLOCKERS
        and "TERMINAL_BLOCKER"
        not in [str(flag or "").upper() for flag in (row.get("risk_flags") or [])]
        and str(row.get("value_type_primary") or UNKNOWN).upper()
        in {
            "ASSET_BACKED_VALUE",
            "EARNINGS_POWER_VALUE",
            "QUALITY_VALUE",
            "CYCLICAL_VALUE",
        }
    )


def _value_type_headwind(row: dict[str, Any]) -> bool:
    return str(row.get("value_type_primary") or UNKNOWN).upper() == "FRAGILE_VALUE"


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


def _capital_allocation_support_code(row: dict[str, Any]) -> str:
    """Return capital allocation discipline support code, or empty string."""
    cls = str(row.get("capital_allocation_discipline_class") or "").upper()
    if cls == "OWNER_FRIENDLY_DISCIPLINED":
        return "OWNER_FRIENDLY_CAPITAL_ALLOCATION_SUPPORT"
    return ""


def _capital_allocation_headwind_code(row: dict[str, Any]) -> str:
    """Return capital allocation discipline headwind code, or empty string."""
    cls = str(row.get("capital_allocation_discipline_class") or "").upper()
    if cls == "OWNER_DILUTIVE_OR_DESTRUCTIVE":
        return "DILUTIVE_CAPITAL_ALLOCATION_HEADWIND"
    if cls == "MIXED_CAPITAL_ALLOCATION":
        return "PER_SHARE_VALUE_CAPTURE_MIXED"
    if cls == "CAPITAL_ALLOCATION_UNKNOWN":
        return "CAPITAL_ALLOCATION_UNKNOWN"
    return ""


def _limited_downside_support(row: dict[str, Any]) -> bool:
    support_type = str(row.get("downside_support_type") or UNKNOWN)
    mos_classification = str(row.get("mos_classification") or UNKNOWN)
    mos_status = str(row.get("mos_assessment_status") or UNKNOWN).upper()
    if mos_status == "MOS_UNASSESSABLE":
        return False
    return support_type in {"LIMITED_SUPPORT", "UNKNOWN_SUPPORT"} or mos_classification in {
        "NO_MARGIN_OF_SAFETY",
        "MOS_UNKNOWN",
    }


def _has_convergent_l4_signal(row: dict[str, Any]) -> bool:
    direction = str(row.get("variant_perception_direction") or "NONE").upper()
    supporting_sources = {
        str(source or "").upper()
        for source in (row.get("_variant_supporting_sources") or [])
        if str(source or "").strip()
    }
    return (
        direction in {"UNDERVALUED", "OVERVALUED"}
        and int(row.get("pattern_confirmed_count") or 0) >= 2
        and int(row.get("filing_diff_high_materiality_count") or 0) >= 1
        and {"PATTERN", "FILING_DIFF"}.issubset(supporting_sources)
    )


def _promote_lane_one_level(lane: str) -> str:
    if lane == LANE_4_DEPRIORITIZED:
        return LANE_3_MONITOR
    if lane == LANE_3_MONITOR:
        return LANE_2_RESEARCH_QUEUE
    if lane == LANE_2_RESEARCH_QUEUE:
        return LANE_1_HIGH_PRIORITY
    return LANE_1_HIGH_PRIORITY


def _effective_blocker(row: dict[str, Any]) -> str:
    for candidate in [
        row.get("latest_primary_blocker"),
        row.get("primary_blocker"),
    ]:
        token = str(candidate or "").strip().upper()
        if token:
            return token
    return UNKNOWN


def _effective_implied_return(row: dict[str, Any]) -> Any:
    for candidate in [
        row.get("latest_implied_return_base"),
        row.get("implied_return_base"),
    ]:
        if _is_num(candidate):
            return float(candidate)
    return UNKNOWN


def _build_strength_flags(row: dict[str, Any]) -> list[str]:
    thresholds = (
        row.get("thresholds_effective")
        if isinstance(row.get("thresholds_effective"), dict)
        else _thresholds_effective()
    )
    appearances = int(row.get("appearances_count") or 0)
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    effective_implied = _effective_implied_return(row)
    flags: list[str] = []
    if appearances >= int(
        thresholds["promotion_min_appearances_high_priority"]
    ) and latest_gate in {"PASS", "WATCH"}:
        flags.append("REPEATED_SURVIVOR")
    if _is_num(effective_implied) and float(effective_implied) >= float(
        thresholds["promotion_min_implied_return"]
    ):
        flags.append("KNOWN_UPSIDE")
    if any(
        _is_num(row.get(field)) and float(row.get(field)) > 0 for field in ["mos_epv", "mos_netnet"]
    ):
        flags.append("GD_MOS_PRESENT")
    if (
        _is_num(row.get("owner_earnings_yield_ev_3y"))
        and float(row.get("owner_earnings_yield_ev_3y")) > 0
    ):
        flags.append("YIELD_SIGNAL_PRESENT")
    if _is_num(row.get("composite_score_total")) and float(
        row.get("composite_score_total")
    ) >= float(thresholds["promotion_lane2_min_score"]):
        flags.append("COMPOSITE_SIGNAL_PRESENT")
    if _oe_quality_supports_lane(row):
        flags.append("HIGH_OE_QUALITY")
    if _intangible_supports_lane(row):
        flags.append("HIGH_INTANGIBLE_ECONOMICS")
    if _owner_value_capture_supports_lane(row):
        flags.append("STRONG_OWNER_VALUE_CAPTURE")
    if _reinvestment_supports_lane(row):
        flags.append("PRODUCTIVE_REINVESTMENT_SUPPORT")
    if _returns_persistence_supports_lane(row):
        flags.append("HIGH_RETURNS_PERSISTENCE_SUPPORT")
    if _revenue_dependence_supports_lane(row):
        flags.append("LOW_REVENUE_DEPENDENCE_SUPPORT")
    if _maintenance_capex_supports_lane(row):
        flags.append("LOW_ASSET_INTENSITY_SUPPORT")
    if _accounting_quality_supports_lane(row):
        flags.append("HIGH_ACCOUNTING_QUALITY_SUPPORT")
    if _balance_sheet_supports_lane(row):
        flags.append("LOW_BALANCE_SHEET_STRESS_SUPPORT")
    if _intrinsic_supports_lane(row):
        flags.append("REAL_DOWNSIDE_SUPPORT_PRESENT")
        if _is_num(row.get("mos_to_floor")) and float(row.get("mos_to_floor")) >= 0.25:
            flags.append("MOS_TO_FLOOR_ATTRACTIVE")
    if _valuation_confidence_supports_lane(row):
        flags.append("MULTI_SUPPORT_VALUE_CASE")
        if str(row.get("valuation_fragility_status") or "").upper() == "LOW_FRAGILITY":
            flags.append("LOW_FRAGILITY_UNDERWRITING")
    value_type_code = _value_type_support_code(row)
    if value_type_code and _value_type_supports_lane(row):
        flags.append(value_type_code)
    readiness_code = _readiness_support_code(row)
    if readiness_code in {"READY_WITH_REAL_SUPPORT", "BLOCKED_BUT_RESEARCH_WORTHY"}:
        flags.append(readiness_code)
    mos_guardrail_code = _mos_guardrail_support_code(row)
    if mos_guardrail_code:
        flags.append(mos_guardrail_code)
    impairment_support = _impairment_support_code(row)
    if impairment_support:
        flags.append(impairment_support)
    norm_cred_support = _normalization_credibility_support_code(row)
    if norm_cred_support:
        flags.append(norm_cred_support)
    cap_alloc_support = _capital_allocation_support_code(row)
    if cap_alloc_support:
        flags.append(cap_alloc_support)
    if (
        str(row.get("variant_perception_max_confidence") or "NONE").upper() == "HIGH"
        and str(row.get("variant_perception_direction") or "NONE").upper() == "UNDERVALUED"
    ):
        flags.append("PERCEPTION_BOOST_HIGH")
    elif (
        str(row.get("variant_perception_max_confidence") or "NONE").upper() == "MEDIUM"
        and str(row.get("variant_perception_direction") or "NONE").upper() == "UNDERVALUED"
    ):
        flags.append("PERCEPTION_SUPPORT_MEDIUM")
    if _has_convergent_l4_signal(row):
        flags.append("CONVERGENT_SIGNAL")
    if (
        _is_num(row.get("tech_valuation_divergence"))
        and float(row.get("tech_valuation_divergence")) > 30.0
        and str(row.get("tech_category") or "TRADITIONAL_OPERATING").upper()
        != "TRADITIONAL_OPERATING"
    ):
        flags.append("TECH_ADJUSTED_VALUE")
    return _dedupe_refs(flags)


def _build_risk_flags(row: dict[str, Any]) -> list[str]:
    thresholds = (
        row.get("thresholds_effective")
        if isinstance(row.get("thresholds_effective"), dict)
        else _thresholds_effective()
    )
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    blocker = _effective_blocker(row)
    appearances = int(row.get("appearances_count") or 0)
    effective_implied = _effective_implied_return(row)

    flags: list[str] = []
    if blocker not in _NONE_BLOCKERS:
        flags.append(blocker)
    if blocker in set(
        str(token).upper() for token in thresholds["promotion_terminal_blocker_codes"]
    ):
        flags.append("TERMINAL_BLOCKER")
    if latest_gate == "FAIL" and appearances >= 2:
        flags.append("REPEATED_FAIL")
    elif latest_gate == "FAIL":
        flags.append("FAIL_CONFIRMED")
    if _is_num(effective_implied) and float(effective_implied) < 0:
        flags.append("NEGATIVE_UPSIDE")
    if bool(row.get("facts_blocker_retryable")):
        flags.append("RETRYABLE_FACTS_BLOCKER")
    elif bool(row.get("facts_blocker_partial_usable")):
        flags.append("PARTIAL_FACTS_USABLE")
    elif bool(row.get("facts_blocker_terminal")) and str(
        row.get("facts_blocker_class") or ""
    ) not in {"", "FACTS_OK"}:
        flags.append("TERMINAL_FACTS_BLOCKER")
    if bool(row.get("fail_due_to_economic_weakness")):
        flags.append("ECONOMICS_KNOWN_WEAK")
    if _owner_value_capture_headwind(row):
        flags.append("WEAK_OWNER_VALUE_CAPTURE")
    reinvestment_headwind = _reinvestment_headwind_code(row)
    if reinvestment_headwind:
        flags.append(reinvestment_headwind)
    accounting_headwind = _accounting_quality_headwind_code(row)
    if accounting_headwind:
        flags.append(accounting_headwind)
    balance_sheet_headwind = _balance_sheet_headwind_code(row)
    if balance_sheet_headwind:
        flags.append(balance_sheet_headwind)
    returns_headwind = _returns_persistence_headwind_code(row)
    if returns_headwind:
        flags.append(returns_headwind)
    revenue_headwind = _revenue_dependence_headwind_code(row)
    if revenue_headwind:
        flags.append(revenue_headwind)
    maintenance_headwind = _maintenance_capex_headwind_code(row)
    if maintenance_headwind:
        flags.append(maintenance_headwind)
    if _limited_downside_support(row):
        flags.append("LIMITED_DOWNSIDE_SUPPORT")
    if _valuation_confidence_headwind(row):
        if _int_or_zero(row.get("valuation_support_count")) <= 1:
            flags.append("SINGLE_SUPPORT_FRAGILE")
        if str(row.get("valuation_convergence_status") or "").upper() == "WEAK_CONVERGENCE":
            flags.append("SUPPORT_CONFLICT_HEADWIND")
    if _value_type_headwind(row):
        flags.append("FRAGILE_VALUE_HEADWIND")
    integrity_headwind = _valuation_integrity_headwind(row)
    if integrity_headwind:
        flags.append(integrity_headwind)
    cyclical_headwind = _cyclical_risk_headwind(row)
    if cyclical_headwind:
        flags.append(cyclical_headwind)
    impairment_headwind = _impairment_headwind_code(row)
    if impairment_headwind:
        flags.append(impairment_headwind)
    norm_cred_headwind = _normalization_credibility_headwind_code(row)
    if norm_cred_headwind:
        flags.append(norm_cred_headwind)
    cap_alloc_headwind = _capital_allocation_headwind_code(row)
    if cap_alloc_headwind:
        flags.append(cap_alloc_headwind)
    readiness_code = _readiness_support_code(row)
    if readiness_code in {
        "WATCH_WITH_THIN_EDGE",
        "NOT_INVESTABLE_STRUCTURAL",
        "READINESS_UNKNOWN_EVIDENCE_GAP",
    }:
        flags.append(readiness_code)
    mos_guardrail_code = _mos_guardrail_headwind_code(row)
    if mos_guardrail_code:
        flags.append(mos_guardrail_code)
    if (
        str(row.get("variant_perception_max_confidence") or "NONE").upper() == "HIGH"
        and str(row.get("variant_perception_direction") or "NONE").upper() == "OVERVALUED"
    ):
        flags.append("OVERVALUATION_BLOCK_HIGH")
    elif (
        str(row.get("variant_perception_max_confidence") or "NONE").upper() == "MEDIUM"
        and str(row.get("variant_perception_direction") or "NONE").upper() == "OVERVALUED"
    ):
        flags.append("OVERVALUATION_RISK")
    return _dedupe_refs(flags)


def _base_priority_lane(row: dict[str, Any]) -> str:
    thresholds = (
        row.get("thresholds_effective")
        if isinstance(row.get("thresholds_effective"), dict)
        else _thresholds_effective()
    )
    appearances = int(row.get("appearances_count") or 0)
    latest_gate = str(row.get("latest_value_gate_status") or UNKNOWN).upper()
    blocker = _effective_blocker(row)
    terminal_blockers = {
        str(token).upper() for token in thresholds["promotion_terminal_blocker_codes"]
    }
    terminal = blocker in terminal_blockers
    effective_implied = _effective_implied_return(row)
    positive_signal = any(
        flag
        in {"KNOWN_UPSIDE", "GD_MOS_PRESENT", "YIELD_SIGNAL_PRESENT", "COMPOSITE_SIGNAL_PRESENT"}
        for flag in row.get("strength_flags") or []
    )
    facts_terminal = bool(row.get("facts_blocker_terminal")) and str(
        row.get("facts_blocker_class") or ""
    ) not in {"", "FACTS_OK"}
    hydrable_blocker = blocker not in _NONE_BLOCKERS and not terminal and not facts_terminal
    memory_support = _memory_supports_lane(row)
    oe_quality_support = _oe_quality_supports_lane(row)
    intangible_support = _intangible_supports_lane(row)
    owner_value_capture_support = _owner_value_capture_supports_lane(row)
    reinvestment_support = _reinvestment_supports_lane(row)
    intrinsic_support = _intrinsic_supports_lane(row)
    valuation_confidence_support = _valuation_confidence_supports_lane(row)
    value_type_support = _value_type_supports_lane(row)
    readiness_class = str(row.get("investment_readiness_class") or "READINESS_UNKNOWN").upper()

    if (
        appearances >= int(thresholds["promotion_min_appearances_high_priority"])
        and latest_gate in {"PASS", "WATCH"}
        and _is_num(effective_implied)
        and float(effective_implied) >= float(thresholds["promotion_min_implied_return"])
        and not terminal
        and blocker in _NONE_BLOCKERS
    ):
        return LANE_1_HIGH_PRIORITY

    if terminal or (latest_gate == "FAIL" and appearances >= 2):
        return LANE_4_DEPRIORITIZED

    if latest_gate in {"PASS", "WATCH"} and (
        positive_signal
        or hydrable_blocker
        or memory_support
        or oe_quality_support
        or intangible_support
        or owner_value_capture_support
        or reinvestment_support
        or intrinsic_support
        or valuation_confidence_support
        or value_type_support
        or readiness_class == "RESEARCH_WORTHY_NOT_READY"
    ):
        return LANE_2_RESEARCH_QUEUE

    if latest_gate == "FAIL" and not positive_signal:
        return LANE_4_DEPRIORITIZED

    return LANE_3_MONITOR


def classify_priority_lane(row: dict[str, Any]) -> str:
    variant_confidence = str(row.get("variant_perception_max_confidence") or "NONE").upper()
    variant_direction = str(row.get("variant_perception_direction") or "NONE").upper()
    if variant_confidence == "HIGH" and variant_direction == "OVERVALUED":
        return LANE_4_DEPRIORITIZED

    lane = _base_priority_lane(row)
    if variant_confidence == "HIGH" and variant_direction == "UNDERVALUED":
        return _promote_lane_one_level(lane)
    return lane


def _promotion_reason_codes(row: dict[str, Any]) -> list[str]:
    lane = str(row.get("priority_lane") or "")
    reasons: list[str] = []
    memory_support = _memory_supports_lane(row)
    facts_blocker_class = str(row.get("facts_blocker_class") or "FACTS_OK")
    facts_retryable = bool(row.get("facts_blocker_retryable"))
    facts_terminal = bool(row.get("facts_blocker_terminal"))
    facts_partial = bool(row.get("facts_blocker_partial_usable"))
    economics_known_weak = bool(row.get("fail_due_to_economic_weakness"))
    impairment_headwind = _impairment_headwind_code(row)
    impairment_support = _impairment_support_code(row)
    if lane == LANE_1_HIGH_PRIORITY:
        reasons.append("PROMOTE_HIGH_PRIORITY")
        for code in ["REPEATED_SURVIVOR", "KNOWN_UPSIDE", "GD_MOS_PRESENT", "YIELD_SIGNAL_PRESENT"]:
            if code in row.get("strength_flags", []):
                reasons.append(code)
        if impairment_support:
            reasons.append(impairment_support)
        if impairment_headwind:
            reasons.append(impairment_headwind)
    elif lane == LANE_2_RESEARCH_QUEUE:
        reasons.append("QUEUE_RESEARCH")
        if _effective_blocker(row) not in _NONE_BLOCKERS and "TERMINAL_BLOCKER" not in row.get(
            "risk_flags", []
        ):
            reasons.append("HYDRABLE_BLOCKER")
        for code in [
            "KNOWN_UPSIDE",
            "GD_MOS_PRESENT",
            "YIELD_SIGNAL_PRESENT",
            "COMPOSITE_SIGNAL_PRESENT",
        ]:
            if code in row.get("strength_flags", []):
                reasons.append(code)
                break
        if memory_support:
            reasons.append("MEMORY_PRIORITY_SUPPORT")
        if _oe_quality_supports_lane(row):
            reasons.append("HIGH_OE_QUALITY_SUPPORT")
        if _intangible_supports_lane(row):
            reasons.append("STRONG_INTANGIBLE_ECONOMICS_SUPPORT")
        if _owner_value_capture_supports_lane(row):
            reasons.append("STRONG_OWNER_VALUE_CAPTURE_SUPPORT")
        if _reinvestment_supports_lane(row):
            reasons.append("PRODUCTIVE_REINVESTMENT_SUPPORT")
        if _intrinsic_supports_lane(row):
            reasons.append("REAL_DOWNSIDE_SUPPORT_PRESENT")
            if _is_num(row.get("mos_to_floor")) and float(row.get("mos_to_floor")) >= 0.25:
                reasons.append("MOS_TO_FLOOR_ATTRACTIVE")
        if _valuation_confidence_supports_lane(row):
            reasons.append("MULTI_SUPPORT_VALUE_CASE")
            if str(row.get("valuation_fragility_status") or "").upper() == "LOW_FRAGILITY":
                reasons.append("LOW_FRAGILITY_UNDERWRITING")
        value_type_code = _value_type_support_code(row)
        if value_type_code and _value_type_supports_lane(row):
            reasons.append(value_type_code)
        readiness_code = _readiness_support_code(row)
        if readiness_code in {"READY_WITH_REAL_SUPPORT", "BLOCKED_BUT_RESEARCH_WORTHY"}:
            reasons.append(readiness_code)
        mos_guardrail_support = _mos_guardrail_support_code(row)
        if mos_guardrail_support:
            reasons.append(mos_guardrail_support)
        if facts_retryable:
            reasons.append("RETRYABLE_FACTS_BLOCKER")
        elif facts_partial:
            reasons.append("PARTIAL_FACTS_USABLE")
        elif facts_terminal and facts_blocker_class != "FACTS_OK":
            reasons.append("TERMINAL_FACTS_BLOCKER")
        if economics_known_weak:
            reasons.append("ECONOMICS_KNOWN_WEAK")
        if _owner_value_capture_headwind(row):
            reasons.append("WEAK_OWNER_VALUE_CAPTURE_HEADWIND")
        reinvestment_headwind = _reinvestment_headwind_code(row)
        if reinvestment_headwind:
            reasons.append(reinvestment_headwind)
        returns_headwind = _returns_persistence_headwind_code(row)
        if returns_headwind:
            reasons.append(returns_headwind)
        revenue_headwind = _revenue_dependence_headwind_code(row)
        if revenue_headwind:
            reasons.append(revenue_headwind)
        if _limited_downside_support(row):
            reasons.append("LIMITED_DOWNSIDE_SUPPORT")
        if _int_or_zero(row.get("valuation_support_count")) <= 1:
            reasons.append("SINGLE_SUPPORT_FRAGILE")
        if str(row.get("valuation_convergence_status") or "").upper() == "WEAK_CONVERGENCE":
            reasons.append("SUPPORT_CONFLICT_HEADWIND")
        if _value_type_headwind(row):
            reasons.append("FRAGILE_VALUE_HEADWIND")
        mos_guardrail_headwind = _mos_guardrail_headwind_code(row)
        if mos_guardrail_headwind:
            reasons.append(mos_guardrail_headwind)
        if impairment_support:
            reasons.append(impairment_support)
        if impairment_headwind:
            reasons.append(impairment_headwind)
    elif lane == LANE_3_MONITOR:
        reasons.append("MONITOR_PENDING")
        if _effective_blocker(row) not in _NONE_BLOCKERS:
            reasons.append(_effective_blocker(row))
        elif "KNOWN_UPSIDE" not in row.get("strength_flags", []):
            reasons.append("MISSING_STRONG_SIGNAL")
        if facts_retryable:
            reasons.append("RETRYABLE_FACTS_BLOCKER")
        elif facts_partial:
            reasons.append("PARTIAL_FACTS_USABLE")
        elif facts_terminal and facts_blocker_class != "FACTS_OK":
            reasons.append("TERMINAL_FACTS_BLOCKER")
        if economics_known_weak:
            reasons.append("ECONOMICS_KNOWN_WEAK")
        if _owner_value_capture_headwind(row):
            reasons.append("WEAK_OWNER_VALUE_CAPTURE_HEADWIND")
        reinvestment_headwind = _reinvestment_headwind_code(row)
        if reinvestment_headwind:
            reasons.append(reinvestment_headwind)
        if _limited_downside_support(row):
            reasons.append("LIMITED_DOWNSIDE_SUPPORT")
        if _int_or_zero(row.get("valuation_support_count")) <= 1:
            reasons.append("SINGLE_SUPPORT_FRAGILE")
        if str(row.get("valuation_convergence_status") or "").upper() == "WEAK_CONVERGENCE":
            reasons.append("SUPPORT_CONFLICT_HEADWIND")
        value_type_code = _value_type_support_code(row)
        if value_type_code:
            reasons.append(value_type_code)
        if _value_type_headwind(row):
            reasons.append("FRAGILE_VALUE_HEADWIND")
        readiness_code = _readiness_support_code(row)
        if readiness_code in {"WATCH_WITH_THIN_EDGE", "READINESS_UNKNOWN_EVIDENCE_GAP"}:
            reasons.append(readiness_code)
        mos_guardrail_support = _mos_guardrail_support_code(row)
        if mos_guardrail_support:
            reasons.append(mos_guardrail_support)
        mos_guardrail_headwind = _mos_guardrail_headwind_code(row)
        if mos_guardrail_headwind:
            reasons.append(mos_guardrail_headwind)
        if impairment_support:
            reasons.append(impairment_support)
        if impairment_headwind:
            reasons.append(impairment_headwind)
    else:
        reasons.append("DEPRIORITIZE")
        for code in [
            "TERMINAL_BLOCKER",
            "REPEATED_FAIL",
            "FAIL_CONFIRMED",
            _effective_blocker(row),
        ]:
            token = str(code or "").strip().upper()
            if token and token not in reasons and token not in {"NONE", UNKNOWN}:
                reasons.append(token)
                break
        if facts_retryable:
            reasons.append("RETRYABLE_FACTS_BLOCKER")
        elif facts_partial:
            reasons.append("PARTIAL_FACTS_USABLE")
        elif facts_terminal and facts_blocker_class != "FACTS_OK":
            reasons.append("TERMINAL_FACTS_BLOCKER")
        if economics_known_weak:
            reasons.append("ECONOMICS_KNOWN_WEAK")
        if _owner_value_capture_headwind(row):
            reasons.append("WEAK_OWNER_VALUE_CAPTURE_HEADWIND")
        reinvestment_headwind = _reinvestment_headwind_code(row)
        if reinvestment_headwind:
            reasons.append(reinvestment_headwind)
        if _limited_downside_support(row):
            reasons.append("LIMITED_DOWNSIDE_SUPPORT")
        if _int_or_zero(row.get("valuation_support_count")) <= 1:
            reasons.append("SINGLE_SUPPORT_FRAGILE")
        if str(row.get("valuation_convergence_status") or "").upper() == "WEAK_CONVERGENCE":
            reasons.append("SUPPORT_CONFLICT_HEADWIND")
        if _value_type_headwind(row):
            reasons.append("FRAGILE_VALUE_HEADWIND")
        integrity_headwind = _valuation_integrity_headwind(row)
        if integrity_headwind:
            reasons.append(integrity_headwind)
        readiness_code = _readiness_support_code(row)
        if readiness_code in {"NOT_INVESTABLE_STRUCTURAL", "READINESS_UNKNOWN_EVIDENCE_GAP"}:
            reasons.append(readiness_code)
        mos_guardrail_headwind = _mos_guardrail_headwind_code(row)
        if mos_guardrail_headwind:
            reasons.append(mos_guardrail_headwind)
        if impairment_support:
            reasons.append(impairment_support)
        if impairment_headwind:
            reasons.append(impairment_headwind)
    for code in [
        "PERCEPTION_BOOST_HIGH",
        "PERCEPTION_SUPPORT_MEDIUM",
        "CONVERGENT_SIGNAL",
        "TECH_ADJUSTED_VALUE",
        "OVERVALUATION_BLOCK_HIGH",
        "OVERVALUATION_RISK",
    ]:
        if code in (row.get("strength_flags") or []) or code in (row.get("risk_flags") or []):
            reasons.append(code)
    return _dedupe_refs(reasons)


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _lane_rank(str(row.get("priority_lane") or "")),
        _status_rank(str(row.get("latest_value_gate_status") or UNKNOWN)),
        _desc_key(_effective_implied_return(row)),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _balance_sheet_stress_rank(
            row.get("balance_sheet_stress_class", "BALANCE_SHEET_STRESS_UNKNOWN")
        ),
        _refinancing_risk_rank(row.get("refinancing_risk_class", "REFINANCING_RISK_UNKNOWN")),
        _returns_persistence_rank(
            row.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")
        ),
        _revenue_dependence_rank(
            row.get("revenue_dependence_risk_class", "REVENUE_DEPENDENCE_UNKNOWN")
        ),
        _maintenance_capex_credibility_rank(
            row.get("maintenance_capex_credibility_class", "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN")
        ),
        _asset_intensity_rank(row.get("asset_intensity_class", "ASSET_INTENSITY_UNKNOWN")),
        _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _reinvestment_rank(
            row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")
        ),
        _desc_key(row.get("valuation_support_count", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("owner_value_capture_score", UNKNOWN)),
        -int(row.get("memory_priority_total") or 0),
        _desc_key(row.get("mos_epv", UNKNOWN)),
        -int(row.get("appearances_count") or 0),
        _variant_confidence_rank(row.get("variant_perception_max_confidence", "NONE")),
        -int(row.get("variant_signal_source_count") or 0),
        -int(row.get("pattern_confirmed_count") or 0),
        str(row.get("ticker") or ""),
    )


def _lane_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _status_rank(str(row.get("latest_value_gate_status") or UNKNOWN)),
        _desc_key(_effective_implied_return(row)),
        _desc_key(row.get("mos_to_floor", UNKNOWN)),
        _readiness_rank(row.get("investment_readiness_class", "READINESS_UNKNOWN")),
        _confidence_rank(row.get("valuation_confidence_class", "CONFIDENCE_UNKNOWN")),
        _integrity_rank(row.get("valuation_integrity_class", "INTEGRITY_UNKNOWN")),
        _balance_sheet_stress_rank(
            row.get("balance_sheet_stress_class", "BALANCE_SHEET_STRESS_UNKNOWN")
        ),
        _refinancing_risk_rank(row.get("refinancing_risk_class", "REFINANCING_RISK_UNKNOWN")),
        _returns_persistence_rank(
            row.get("returns_persistence_class", "RETURNS_PERSISTENCE_UNKNOWN")
        ),
        _revenue_dependence_rank(
            row.get("revenue_dependence_risk_class", "REVENUE_DEPENDENCE_UNKNOWN")
        ),
        _maintenance_capex_credibility_rank(
            row.get("maintenance_capex_credibility_class", "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN")
        ),
        _asset_intensity_rank(row.get("asset_intensity_class", "ASSET_INTENSITY_UNKNOWN")),
        _accounting_quality_rank(row.get("accounting_quality_class", "ACCOUNTING_QUALITY_UNKNOWN")),
        _value_type_rank(row.get("value_type_primary", "UNKNOWN_VALUE_TYPE")),
        _reinvestment_rank(
            row.get("reinvestment_efficiency_class", "REINVESTMENT_EFFICIENCY_UNKNOWN")
        ),
        _desc_key(row.get("valuation_support_count", UNKNOWN)),
        _desc_key(row.get("oe_quality_total", UNKNOWN)),
        _desc_key(row.get("intangible_economics_total", UNKNOWN)),
        _desc_key(row.get("owner_value_capture_score", UNKNOWN)),
        -int(row.get("memory_priority_total") or 0),
        _desc_key(row.get("mos_epv", UNKNOWN)),
        -int(row.get("appearances_count") or 0),
        _variant_confidence_rank(row.get("variant_perception_max_confidence", "NONE")),
        -int(row.get("variant_signal_source_count") or 0),
        -int(row.get("pattern_confirmed_count") or 0),
        str(row.get("ticker") or ""),
    )


def build_promotion_state(
    campaign_run_id: str,
    master_watchlist_state: dict[str, Any],
    master_shortlist: dict[str, Any],
) -> dict[str, Any]:
    thresholds = _thresholds_effective()
    shortlist_rows = {
        str(row.get("ticker") or "").upper(): row
        for row in (master_shortlist.get("rows") or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    watch_tickers = (
        master_watchlist_state.get("tickers")
        if isinstance(master_watchlist_state.get("tickers"), dict)
        else {}
    )
    memory_lookup = _load_memory_lookup()
    l4_context: dict[str, Any] = {"cfg": get_config(), "run_cache": {}, "valuation_cache": {}}

    candidate_run_ids: set[str] = set()
    for row in shortlist_rows.values():
        for source in [item for item in (row.get("source_runs") or []) if isinstance(item, dict)]:
            run_id = str(source.get("universe_run_id") or "").strip()
            if run_id:
                candidate_run_ids.add(run_id)
    for watch_row in [row for row in watch_tickers.values() if isinstance(row, dict)]:
        for source in [item for item in (watch_row.get("history") or []) if isinstance(item, dict)]:
            run_id = str(source.get("universe_run_id") or "").strip()
            if run_id:
                candidate_run_ids.add(run_id)
    for run_id in sorted(candidate_run_ids):
        _ensure_l4_run_context(run_id, l4_context)

    rows: list[dict[str, Any]] = []
    for ticker in sorted(
        set(shortlist_rows) | set(str(key).upper() for key in watch_tickers.keys())
    ):
        shortlist_row = shortlist_rows.get(ticker, {})
        watch_row = (
            watch_tickers.get(ticker, {}) if isinstance(watch_tickers.get(ticker), dict) else {}
        )
        source_runs = [
            row for row in (shortlist_row.get("source_runs") or []) if isinstance(row, dict)
        ]
        if not source_runs:
            history = [row for row in (watch_row.get("history") or []) if isinstance(row, dict)]
            source_runs = [
                {
                    "campaign_item": str(entry.get("campaign_item") or ""),
                    "universe_run_id": str(entry.get("universe_run_id") or ""),
                    "batch_run_id": f"{str(entry.get('universe_run_id') or '')}_depth_batch"
                    if str(entry.get("universe_run_id") or "").strip()
                    else "",
                }
                for entry in history
            ]
        merged = {
            "ticker": ticker,
            "campaign_run_id": campaign_run_id,
            "priority_lane": "",
            "promotion_reason_codes": [],
            "appearances_count": int(watch_row.get("appearances_count") or 0),
            "best_rank_seen": int(shortlist_row.get("best_rank_seen") or 0),
            "value_gate_status": str(
                shortlist_row.get("value_gate_status")
                or watch_row.get("latest_value_gate_status")
                or UNKNOWN
            ).upper(),
            "latest_value_gate_status": str(
                watch_row.get("latest_value_gate_status")
                or shortlist_row.get("latest_value_gate_status")
                or shortlist_row.get("value_gate_status")
                or UNKNOWN
            ).upper(),
            "implied_return_base": shortlist_row.get("implied_return_base", UNKNOWN),
            "latest_implied_return_base": watch_row.get(
                "latest_implied_return_base", shortlist_row.get("implied_return_base", UNKNOWN)
            ),
            "latest_primary_blocker": str(
                watch_row.get("latest_primary_blocker")
                or shortlist_row.get("latest_primary_blocker")
                or shortlist_row.get("primary_blocker")
                or UNKNOWN
            ).upper(),
            "primary_blocker": str(
                shortlist_row.get("primary_blocker")
                or watch_row.get("latest_primary_blocker")
                or UNKNOWN
            ).upper(),
            "mos_epv": shortlist_row.get("mos_epv", UNKNOWN),
            "mos_netnet": shortlist_row.get("mos_netnet", UNKNOWN),
            "owner_earnings_yield_ev_3y": shortlist_row.get("owner_earnings_yield_ev_3y", UNKNOWN),
            "yield_metric_used": str(shortlist_row.get("yield_metric_used") or UNKNOWN),
            "owner_earnings_stability_score": shortlist_row.get(
                "owner_earnings_stability_score", UNKNOWN
            ),
            "capital_allocation_score": shortlist_row.get("capital_allocation_score", UNKNOWN),
            "cash_conversion_score": shortlist_row.get("cash_conversion_score", UNKNOWN),
            "oe_quality_total": shortlist_row.get("oe_quality_total", UNKNOWN),
            "oe_quality_reason_codes": [
                str(code)
                for code in (shortlist_row.get("oe_quality_reason_codes") or [])
                if str(code).strip()
            ],
            "gross_margin_durability_score": shortlist_row.get(
                "gross_margin_durability_score", UNKNOWN
            ),
            "balance_sheet_optionality_score": shortlist_row.get(
                "balance_sheet_optionality_score", UNKNOWN
            ),
            "cycle_resilience_score": shortlist_row.get("cycle_resilience_score", UNKNOWN),
            "rnd_productivity_score": shortlist_row.get("rnd_productivity_score", UNKNOWN),
            "sga_leverage_score": shortlist_row.get("sga_leverage_score", UNKNOWN),
            "owner_value_capture_score": shortlist_row.get("owner_value_capture_score", UNKNOWN),
            "intangible_economics_total": shortlist_row.get("intangible_economics_total", UNKNOWN),
            "rnd_productivity_reason_codes": [
                str(code)
                for code in (shortlist_row.get("rnd_productivity_reason_codes") or [])
                if str(code).strip()
            ],
            "sga_leverage_reason_codes": [
                str(code)
                for code in (shortlist_row.get("sga_leverage_reason_codes") or [])
                if str(code).strip()
            ],
            "owner_value_capture_reason_codes": [
                str(code)
                for code in (shortlist_row.get("owner_value_capture_reason_codes") or [])
                if str(code).strip()
            ],
            "reinvestment_efficiency_class": str(
                shortlist_row.get("reinvestment_efficiency_class")
                or "REINVESTMENT_EFFICIENCY_UNKNOWN"
            ),
            "reinvestment_efficiency_reason_codes": [
                str(code)
                for code in (shortlist_row.get("reinvestment_efficiency_reason_codes") or [])
                if str(code).strip()
            ],
            "reinvestment_support_signals": [
                str(code)
                for code in (shortlist_row.get("reinvestment_support_signals") or [])
                if str(code).strip()
            ],
            "reinvestment_headwind_signals": [
                str(code)
                for code in (shortlist_row.get("reinvestment_headwind_signals") or [])
                if str(code).strip()
            ],
            "primary_reinvestment_caution": str(
                shortlist_row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
            ),
            "reinvestment_efficiency_summary": str(
                shortlist_row.get("reinvestment_efficiency_summary") or ""
            ),
            "asset_intensity_class": str(
                shortlist_row.get("asset_intensity_class") or "ASSET_INTENSITY_UNKNOWN"
            ),
            "asset_intensity_reason_codes": [
                str(code)
                for code in (shortlist_row.get("asset_intensity_reason_codes") or [])
                if str(code).strip()
            ],
            "maintenance_capex_credibility_class": str(
                shortlist_row.get("maintenance_capex_credibility_class")
                or "MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN"
            ),
            "maintenance_capex_credibility_reason_codes": [
                str(code)
                for code in (shortlist_row.get("maintenance_capex_credibility_reason_codes") or [])
                if str(code).strip()
            ],
            "maintenance_capex_support_signals": [
                str(code)
                for code in (shortlist_row.get("maintenance_capex_support_signals") or [])
                if str(code).strip()
            ],
            "maintenance_capex_headwind_signals": [
                str(code)
                for code in (shortlist_row.get("maintenance_capex_headwind_signals") or [])
                if str(code).strip()
            ],
            "primary_maintenance_capex_caution": str(
                shortlist_row.get("primary_maintenance_capex_caution") or "OWNER_EARNINGS_UNCLEAR"
            ),
            "maintenance_capex_discipline_summary": str(
                shortlist_row.get("maintenance_capex_discipline_summary") or ""
            ),
            "returns_persistence_class": str(
                shortlist_row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
            ),
            "returns_persistence_reason_codes": [
                str(code)
                for code in (shortlist_row.get("returns_persistence_reason_codes") or [])
                if str(code).strip()
            ],
            "returns_support_signals": [
                str(code)
                for code in (shortlist_row.get("returns_support_signals") or [])
                if str(code).strip()
            ],
            "returns_headwind_signals": [
                str(code)
                for code in (shortlist_row.get("returns_headwind_signals") or [])
                if str(code).strip()
            ],
            "primary_returns_caution": str(
                shortlist_row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
            ),
            "economic_durability_summary": str(
                shortlist_row.get("economic_durability_summary") or ""
            ),
            "revenue_dependence_risk_class": str(
                shortlist_row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
            ),
            "revenue_dependence_risk_reason_codes": [
                str(code)
                for code in (shortlist_row.get("revenue_dependence_risk_reason_codes") or [])
                if str(code).strip()
            ],
            "revenue_dependence_support_signals": [
                str(code)
                for code in (shortlist_row.get("revenue_dependence_support_signals") or [])
                if str(code).strip()
            ],
            "revenue_dependence_headwind_signals": [
                str(code)
                for code in (shortlist_row.get("revenue_dependence_headwind_signals") or [])
                if str(code).strip()
            ],
            "primary_revenue_dependence_caution": str(
                shortlist_row.get("primary_revenue_dependence_caution") or "REVENUE_BASE_UNCLEAR"
            ),
            "revenue_fragility_summary": str(shortlist_row.get("revenue_fragility_summary") or ""),
            "accounting_quality_class": str(
                shortlist_row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
            ),
            "accounting_quality_reason_codes": [
                str(code)
                for code in (shortlist_row.get("accounting_quality_reason_codes") or [])
                if str(code).strip()
            ],
            "cash_earnings_support_signals": [
                str(code)
                for code in (shortlist_row.get("cash_earnings_support_signals") or [])
                if str(code).strip()
            ],
            "cash_earnings_headwind_signals": [
                str(code)
                for code in (shortlist_row.get("cash_earnings_headwind_signals") or [])
                if str(code).strip()
            ],
            "primary_accounting_caution": str(
                shortlist_row.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
            ),
            "cash_earnings_discipline_summary": str(
                shortlist_row.get("cash_earnings_discipline_summary") or ""
            ),
            "balance_sheet_stress_class": str(
                shortlist_row.get("balance_sheet_stress_class") or "BALANCE_SHEET_STRESS_UNKNOWN"
            ),
            "balance_sheet_stress_reason_codes": [
                str(code)
                for code in (shortlist_row.get("balance_sheet_stress_reason_codes") or [])
                if str(code).strip()
            ],
            "refinancing_risk_class": str(
                shortlist_row.get("refinancing_risk_class") or "REFINANCING_RISK_UNKNOWN"
            ),
            "refinancing_risk_reason_codes": [
                str(code)
                for code in (shortlist_row.get("refinancing_risk_reason_codes") or [])
                if str(code).strip()
            ],
            "balance_sheet_support_signals": [
                str(code)
                for code in (shortlist_row.get("balance_sheet_support_signals") or [])
                if str(code).strip()
            ],
            "balance_sheet_headwind_signals": [
                str(code)
                for code in (shortlist_row.get("balance_sheet_headwind_signals") or [])
                if str(code).strip()
            ],
            "primary_balance_sheet_caution": str(
                shortlist_row.get("primary_balance_sheet_caution") or "BALANCE_SHEET_UNCLEAR"
            ),
            "balance_sheet_discipline_summary": str(
                shortlist_row.get("balance_sheet_discipline_summary") or ""
            ),
            "intangible_economics_reason_codes": [
                str(code)
                for code in (shortlist_row.get("intangible_economics_reason_codes") or [])
                if str(code).strip()
            ],
            "normalized_earnings_power_value": shortlist_row.get(
                "normalized_earnings_power_value", UNKNOWN
            ),
            "normalized_earnings_power_method_used": str(
                shortlist_row.get("normalized_earnings_power_method_used") or UNKNOWN
            ),
            "normalized_earnings_power_status": str(
                shortlist_row.get("normalized_earnings_power_status") or UNKNOWN
            ),
            "normalized_earnings_power_reason_codes": [
                str(code)
                for code in (shortlist_row.get("normalized_earnings_power_reason_codes") or [])
                if str(code).strip()
            ],
            "intrinsic_floor": shortlist_row.get("intrinsic_floor", UNKNOWN),
            "intrinsic_base": shortlist_row.get("intrinsic_base", UNKNOWN),
            "intrinsic_ceiling": shortlist_row.get("intrinsic_ceiling", UNKNOWN),
            "mos_to_floor": shortlist_row.get("mos_to_floor", UNKNOWN),
            "mos_to_base": shortlist_row.get("mos_to_base", UNKNOWN),
            "mos_classification": str(shortlist_row.get("mos_classification") or UNKNOWN),
            "downside_support_type": str(shortlist_row.get("downside_support_type") or UNKNOWN),
            "valuation_range_reason_codes": [
                str(code)
                for code in (shortlist_row.get("valuation_range_reason_codes") or [])
                if str(code).strip()
            ],
            "downside_support_reason_codes": [
                str(code)
                for code in (shortlist_row.get("downside_support_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_support_count": _int_or_zero(shortlist_row.get("valuation_support_count")),
            "valuation_support_types_present": [
                str(value)
                for value in (shortlist_row.get("valuation_support_types_present") or [])
                if str(value).strip()
            ],
            "valuation_support_count_reason_codes": [
                str(code)
                for code in (shortlist_row.get("valuation_support_count_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_convergence_status": str(
                shortlist_row.get("valuation_convergence_status") or UNKNOWN
            ),
            "valuation_convergence_band_pct": shortlist_row.get(
                "valuation_convergence_band_pct", UNKNOWN
            ),
            "valuation_convergence_reason_codes": [
                str(code)
                for code in (shortlist_row.get("valuation_convergence_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_fragility_status": str(
                shortlist_row.get("valuation_fragility_status") or UNKNOWN
            ),
            "valuation_fragility_reason_codes": [
                str(code)
                for code in (shortlist_row.get("valuation_fragility_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_confidence_class": str(
                shortlist_row.get("valuation_confidence_class") or UNKNOWN
            ),
            "valuation_confidence_reason_codes": [
                str(code)
                for code in (shortlist_row.get("valuation_confidence_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_integrity_class": str(
                shortlist_row.get("valuation_integrity_class") or UNKNOWN
            ),
            "valuation_integrity_reason_codes": [
                str(code)
                for code in (shortlist_row.get("valuation_integrity_reason_codes") or [])
                if str(code).strip()
            ],
            "investment_readiness_class": str(
                shortlist_row.get("investment_readiness_class") or UNKNOWN
            ),
            "investment_readiness_reason_codes": [
                str(code)
                for code in (shortlist_row.get("investment_readiness_reason_codes") or [])
                if str(code).strip()
            ],
            "evidence_sufficiency_class": str(
                shortlist_row.get("evidence_sufficiency_class") or UNKNOWN
            ),
            "evidence_sufficiency_reason_codes": [
                str(code)
                for code in (shortlist_row.get("evidence_sufficiency_reason_codes") or [])
                if str(code).strip()
            ],
            "mos_assessment_status": str(shortlist_row.get("mos_assessment_status") or UNKNOWN),
            "mos_guardrail_reason_codes": [
                str(code)
                for code in (shortlist_row.get("mos_guardrail_reason_codes") or [])
                if str(code).strip()
            ],
            "blocker_stack_primary": str(shortlist_row.get("blocker_stack_primary") or UNKNOWN),
            "primary_next_step": str(shortlist_row.get("primary_next_step") or UNKNOWN),
            "value_type_primary": str(shortlist_row.get("value_type_primary") or UNKNOWN),
            "value_type_secondary": (
                str(shortlist_row.get("value_type_secondary"))
                if str(shortlist_row.get("value_type_secondary") or "").strip()
                else None
            ),
            "value_type_reason_codes": [
                str(code)
                for code in (shortlist_row.get("value_type_reason_codes") or [])
                if str(code).strip()
            ],
            "value_type_support_summary": str(
                shortlist_row.get("value_type_support_summary") or ""
            ),
            "facts_blocker_class": str(shortlist_row.get("facts_blocker_class") or "FACTS_OK"),
            "facts_blocker_retryable": bool(shortlist_row.get("facts_blocker_retryable", False)),
            "facts_blocker_terminal": bool(shortlist_row.get("facts_blocker_terminal", False)),
            "facts_blocker_partial_usable": bool(
                shortlist_row.get("facts_blocker_partial_usable", False)
            ),
            "facts_missing_key_inputs": [
                str(value)
                for value in (shortlist_row.get("facts_missing_key_inputs") or [])
                if str(value).strip()
            ],
            "facts_retry_recommended": bool(shortlist_row.get("facts_retry_recommended", False)),
            "facts_blocker_reason_codes": [
                str(code)
                for code in (shortlist_row.get("facts_blocker_reason_codes") or [])
                if str(code).strip()
            ],
            "facts_recommended_action": str(
                shortlist_row.get("facts_recommended_action") or "NONE"
            ),
            "fail_due_to_missing_evidence": bool(
                shortlist_row.get("fail_due_to_missing_evidence", False)
            ),
            "fail_due_to_economic_weakness": bool(
                shortlist_row.get("fail_due_to_economic_weakness", False)
            ),
            "primary_fail_domain": str(shortlist_row.get("primary_fail_domain") or "NONE"),
            "composite_score_total": shortlist_row.get("composite_score_total", UNKNOWN),
            "memo_path": str(shortlist_row.get("memo_path") or ""),
            "source_runs": source_runs,
            "derived_from": _dedupe_refs(list(shortlist_row.get("derived_from") or [])),
            "history": [row for row in (watch_row.get("history") or []) if isinstance(row, dict)][
                -10:
            ],
            "thresholds_effective": thresholds,
        }
        merged.update(_memory_priority_fields(ticker, memory_lookup))
        l4_payloads = [
            _load_l4_signals_for_ticker(
                ticker,
                str(source.get("universe_run_id") or ""),
                campaign_run_id,
                context=l4_context,
            )
            for source in source_runs
            if str(source.get("universe_run_id") or "").strip()
        ]
        merged.update(_merge_l4_signal_payloads(l4_payloads))
        merged.update(enrich_facts_blocker_fields(merged))
        merged["strength_flags"] = _build_strength_flags(merged)
        merged["risk_flags"] = _build_risk_flags(merged)
        merged["priority_lane"] = classify_priority_lane(merged)
        merged["promotion_reason_codes"] = _promotion_reason_codes(merged)
        rows.append(merged)

    ranked = sorted(rows, key=_candidate_sort_key)
    lane_counts = {
        LANE_1_HIGH_PRIORITY: len(
            [row for row in ranked if row.get("priority_lane") == LANE_1_HIGH_PRIORITY]
        ),
        LANE_2_RESEARCH_QUEUE: len(
            [row for row in ranked if row.get("priority_lane") == LANE_2_RESEARCH_QUEUE]
        ),
        LANE_3_MONITOR: len([row for row in ranked if row.get("priority_lane") == LANE_3_MONITOR]),
        LANE_4_DEPRIORITIZED: len(
            [row for row in ranked if row.get("priority_lane") == LANE_4_DEPRIORITIZED]
        ),
    }

    blocker_counts: dict[str, int] = {}
    blockers_preventing_lane_1: dict[str, int] = {}
    for row in ranked:
        blocker = _effective_blocker(row)
        if blocker in _NONE_BLOCKERS:
            blocker = "NO_PRIMARY_BLOCKER"
        blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        if row.get("priority_lane") != LANE_1_HIGH_PRIORITY:
            blockers_preventing_lane_1[blocker] = blockers_preventing_lane_1.get(blocker, 0) + 1

    return {
        "campaign_run_id": campaign_run_id,
        "generated_at": utc_now_iso(),
        "thresholds_effective": thresholds,
        "ticker_count": len(ranked),
        "lane_counts": lane_counts,
        "blocker_counts": dict(
            sorted(blocker_counts.items(), key=lambda item: (-(item[1]), item[0]))
        ),
        "top_blockers_preventing_lane_1": [
            {"reason_code": key, "count": value}
            for key, value in sorted(
                blockers_preventing_lane_1.items(), key=lambda item: (-(item[1]), item[0])
            )[:10]
        ],
        "rows": ranked,
    }


def write_promotion_artifacts(
    campaign_run_id: str,
    *,
    master_watchlist_state: dict[str, Any] | None = None,
    master_shortlist: dict[str, Any] | None = None,
) -> dict[str, str]:
    paths = _promotion_paths(campaign_run_id)
    shortlist_payload = master_shortlist or _safe_json(paths["root"] / "master_shortlist.json")
    watchlist_payload = master_watchlist_state or _safe_json(
        paths["root"] / "master_watchlist_state.json"
    )
    promotion_state = build_promotion_state(campaign_run_id, watchlist_payload, shortlist_payload)

    ranked = [row for row in (promotion_state.get("rows") or []) if isinstance(row, dict)]
    lanes = {
        "lane_1_high_priority": sorted(
            [row for row in ranked if row.get("priority_lane") == LANE_1_HIGH_PRIORITY],
            key=_lane_sort_key,
        ),
        "lane_2_research_queue": sorted(
            [row for row in ranked if row.get("priority_lane") == LANE_2_RESEARCH_QUEUE],
            key=_lane_sort_key,
        ),
        "lane_3_monitor": sorted(
            [row for row in ranked if row.get("priority_lane") == LANE_3_MONITOR],
            key=_lane_sort_key,
        ),
        "lane_4_deprioritized": sorted(
            [row for row in ranked if row.get("priority_lane") == LANE_4_DEPRIORITIZED],
            key=_lane_sort_key,
        ),
    }
    candidates_payload = {
        "campaign_run_id": campaign_run_id,
        "generated_at": promotion_state["generated_at"],
        "thresholds_effective": promotion_state["thresholds_effective"],
        "candidate_count": len(ranked),
        "rows": ranked,
        "top_10": ranked[:10],
    }
    lanes_payload = {
        "campaign_run_id": campaign_run_id,
        "generated_at": promotion_state["generated_at"],
        "thresholds_effective": promotion_state["thresholds_effective"],
        **lanes,
    }

    _json_write(paths["promotion_state_path"], promotion_state)
    _json_write(paths["promotion_candidates_path"], candidates_payload)
    _json_write(paths["priority_lanes_path"], lanes_payload)
    return {
        "promotion_state_path": str(paths["promotion_state_path"]),
        "promotion_candidates_path": str(paths["promotion_candidates_path"]),
        "priority_lanes_path": str(paths["priority_lanes_path"]),
    }


def open_promotion_state(campaign_run_id: str) -> dict[str, Any]:
    paths = _promotion_paths(campaign_run_id)
    state = _safe_json(paths["promotion_state_path"])
    candidates = _safe_json(paths["promotion_candidates_path"])
    if not state:
        return {
            "status": "MISSING",
            "campaign_run_id": campaign_run_id,
            "promotion_state_path": str(paths["promotion_state_path"]),
            "promotion_candidates_path": str(paths["promotion_candidates_path"]),
            "priority_lanes_path": str(paths["priority_lanes_path"]),
        }
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "ticker_count": int(state.get("ticker_count") or 0),
        "lane_counts": state.get("lane_counts")
        if isinstance(state.get("lane_counts"), dict)
        else {},
        "top_10_promotion_candidates": [
            {
                "ticker": str(row.get("ticker") or ""),
                "priority_lane": str(row.get("priority_lane") or UNKNOWN),
                "latest_value_gate_status": str(row.get("latest_value_gate_status") or UNKNOWN),
                "implied_return_base": _effective_implied_return(row),
                "mos_epv": row.get("mos_epv", UNKNOWN),
                "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
                "mos_classification": str(row.get("mos_classification") or UNKNOWN),
                "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
                "valuation_convergence_status": str(
                    row.get("valuation_convergence_status") or UNKNOWN
                ),
                "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
                "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
                "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
                "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
                "evidence_sufficiency_class": str(row.get("evidence_sufficiency_class") or UNKNOWN),
                "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
                "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
                "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
                "value_type_primary": str(row.get("value_type_primary") or UNKNOWN),
                "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
                "appearances_count": int(row.get("appearances_count") or 0),
                "oe_quality_total": row.get("oe_quality_total", UNKNOWN),
                "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
                "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
                "returns_persistence_class": str(
                    row.get("returns_persistence_class") or "RETURNS_PERSISTENCE_UNKNOWN"
                ),
                "primary_returns_caution": str(
                    row.get("primary_returns_caution") or "RETURNS_DURABILITY_UNCLEAR"
                ),
                "revenue_dependence_risk_class": str(
                    row.get("revenue_dependence_risk_class") or "REVENUE_DEPENDENCE_UNKNOWN"
                ),
                "primary_revenue_dependence_caution": str(
                    row.get("primary_revenue_dependence_caution") or "REVENUE_BASE_UNCLEAR"
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
                "accounting_quality_class": str(
                    row.get("accounting_quality_class") or "ACCOUNTING_QUALITY_UNKNOWN"
                ),
                "primary_accounting_caution": str(
                    row.get("primary_accounting_caution") or "ACCOUNTING_QUALITY_UNCLEAR"
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
                "memory_priority_total": int(row.get("memory_priority_total") or 0),
                "variant_perception_count": int(row.get("variant_perception_count") or 0),
                "variant_perception_max_confidence": str(
                    row.get("variant_perception_max_confidence") or "NONE"
                ),
                "variant_perception_direction": str(
                    row.get("variant_perception_direction") or "NONE"
                ),
                "variant_signal_source_count": int(row.get("variant_signal_source_count") or 0),
                "tech_category": str(row.get("tech_category") or "TRADITIONAL_OPERATING"),
                "tech_valuation_divergence": row.get("tech_valuation_divergence", UNKNOWN),
                "filing_diff_high_materiality_count": int(
                    row.get("filing_diff_high_materiality_count") or 0
                ),
                "pattern_hit_count": int(row.get("pattern_hit_count") or 0),
                "pattern_confirmed_count": int(row.get("pattern_confirmed_count") or 0),
            }
            for row in [row for row in (candidates.get("rows") or []) if isinstance(row, dict)][:10]
        ],
        "top_blockers_preventing_lane_1": state.get("top_blockers_preventing_lane_1")
        if isinstance(state.get("top_blockers_preventing_lane_1"), list)
        else [],
        "promotion_state_path": str(paths["promotion_state_path"]),
        "promotion_candidates_path": str(paths["promotion_candidates_path"]),
        "priority_lanes_path": str(paths["priority_lanes_path"]),
    }


def open_priority_lanes(campaign_run_id: str) -> dict[str, Any]:
    paths = _promotion_paths(campaign_run_id)
    lanes = _safe_json(paths["priority_lanes_path"])
    state = _safe_json(paths["promotion_state_path"])
    if not lanes:
        return {
            "status": "MISSING",
            "campaign_run_id": campaign_run_id,
            "priority_lanes_path": str(paths["priority_lanes_path"]),
            "promotion_state_path": str(paths["promotion_state_path"]),
        }
    return {
        "status": "OK",
        "campaign_run_id": campaign_run_id,
        "lane_counts": state.get("lane_counts")
        if isinstance(state.get("lane_counts"), dict)
        else {},
        "top_by_lane": {
            lane: [
                {
                    "ticker": str(row.get("ticker") or ""),
                    "latest_value_gate_status": str(row.get("latest_value_gate_status") or UNKNOWN),
                    "implied_return_base": _effective_implied_return(row),
                    "mos_epv": row.get("mos_epv", UNKNOWN),
                    "mos_to_floor": row.get("mos_to_floor", UNKNOWN),
                    "mos_classification": str(row.get("mos_classification") or UNKNOWN),
                    "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
                    "valuation_convergence_status": str(
                        row.get("valuation_convergence_status") or UNKNOWN
                    ),
                    "valuation_fragility_status": str(
                        row.get("valuation_fragility_status") or UNKNOWN
                    ),
                    "valuation_confidence_class": str(
                        row.get("valuation_confidence_class") or UNKNOWN
                    ),
                    "valuation_integrity_class": str(
                        row.get("valuation_integrity_class") or UNKNOWN
                    ),
                    "investment_readiness_class": str(
                        row.get("investment_readiness_class") or UNKNOWN
                    ),
                    "evidence_sufficiency_class": str(
                        row.get("evidence_sufficiency_class") or UNKNOWN
                    ),
                    "mos_assessment_status": str(row.get("mos_assessment_status") or UNKNOWN),
                    "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
                    "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
                    "value_type_primary": str(row.get("value_type_primary") or UNKNOWN),
                    "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
                    "oe_quality_total": row.get("oe_quality_total", UNKNOWN),
                    "intangible_economics_total": row.get("intangible_economics_total", UNKNOWN),
                    "owner_value_capture_score": row.get("owner_value_capture_score", UNKNOWN),
                    "reinvestment_efficiency_class": str(
                        row.get("reinvestment_efficiency_class")
                        or "REINVESTMENT_EFFICIENCY_UNKNOWN"
                    ),
                    "primary_reinvestment_caution": str(
                        row.get("primary_reinvestment_caution") or "REINVESTMENT_UNCLEAR"
                    ),
                    "memory_priority_total": int(row.get("memory_priority_total") or 0),
                    "variant_perception_count": int(row.get("variant_perception_count") or 0),
                    "variant_perception_max_confidence": str(
                        row.get("variant_perception_max_confidence") or "NONE"
                    ),
                    "variant_perception_direction": str(
                        row.get("variant_perception_direction") or "NONE"
                    ),
                    "variant_signal_source_count": int(row.get("variant_signal_source_count") or 0),
                    "tech_category": str(row.get("tech_category") or "TRADITIONAL_OPERATING"),
                    "tech_valuation_divergence": row.get("tech_valuation_divergence", UNKNOWN),
                    "filing_diff_high_materiality_count": int(
                        row.get("filing_diff_high_materiality_count") or 0
                    ),
                    "pattern_hit_count": int(row.get("pattern_hit_count") or 0),
                    "pattern_confirmed_count": int(row.get("pattern_confirmed_count") or 0),
                }
                for row in [row for row in (lanes.get(lane) or []) if isinstance(row, dict)][:10]
            ]
            for lane in [
                "lane_1_high_priority",
                "lane_2_research_queue",
                "lane_3_monitor",
                "lane_4_deprioritized",
            ]
        },
        "priority_lanes_path": str(paths["priority_lanes_path"]),
        "promotion_state_path": str(paths["promotion_state_path"]),
        "promotion_candidates_path": str(paths["promotion_candidates_path"]),
    }
