from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso


UNKNOWN = "UNKNOWN"
PASS = "PASS"
WATCH = "WATCH"
FAIL = "FAIL"
_VALID_GATE_STATUSES = {PASS, WATCH, FAIL}

CATEGORY_INPUT_MISSING = "INPUT_MISSING"
CATEGORY_SAFETY_FAIL = "SAFETY_FAIL"
CATEGORY_VALUE_FAIL = "VALUE_FAIL"
CATEGORY_PASS_REASON = "PASS_REASON"
_REASON_CATEGORIES = [
    CATEGORY_INPUT_MISSING,
    CATEGORY_SAFETY_FAIL,
    CATEGORY_VALUE_FAIL,
    CATEGORY_PASS_REASON,
]

# Input reasons
REASON_PRICE_UNKNOWN = "PRICE_UNKNOWN"
REASON_MISSING_INPUT_SHARES = "MISSING_INPUT_SHARES"
REASON_MISSING_INPUT_FCF = "MISSING_INPUT_FCF"
REASON_MISSING_INPUT_VALUATION = "MISSING_INPUT_VALUATION"

# MOS reasons
REASON_MOS_PASS = "MOS_PASS"
REASON_MOS_WATCH = "MOS_WATCH"
REASON_MOS_FAIL = "MOS_FAIL"

# Cash earning reasons
REASON_FCF_PASS = "FCF_POSITIVE_TREND_OK"
REASON_FCF_WATCH_TREND_UNKNOWN = "FCF_POSITIVE_TREND_UNKNOWN"
REASON_FCF_FAIL_TREND = "FCF_MARGIN_TREND_STRONGLY_NEGATIVE"
REASON_FCF_FAIL_LATEST_NEG = "FCF_LATEST_NEGATIVE"
REASON_FCF_FAIL_MAJOR_NEG = "FCF_NEGATIVE_MAJORITY_3Y"

# Balance-sheet reasons
REASON_BALANCE_PASS = "BALANCE_PASS"
REASON_BALANCE_WATCH_UNKNOWN = "BALANCE_WATCH_UNKNOWN"
REASON_BALANCE_WATCH_MID = "BALANCE_WATCH_MID"
REASON_BALANCE_FAIL = "BALANCE_FAIL"

# Dilution reasons
REASON_DILUTION_PASS = "DILUTION_PASS"
REASON_DILUTION_WATCH_UNKNOWN = "DILUTION_WATCH_UNKNOWN"
REASON_DILUTION_WATCH_MID = "DILUTION_WATCH_MID"
REASON_DILUTION_FAIL = "DILUTION_FAIL"

ACTION_HYDRATE_PRICE = "HYDRATE_PRICE_SNAPSHOT"
ACTION_HYDRATE_FACTS = "HYDRATE_FINANCIAL_FACTS"

DEFAULT_GATE_THRESHOLDS = {
    "mos_min": 0.30,
    "valuation_gap_min": 0.10,
    "net_debt_to_cfo_max": 2.5,
    "dilution_max": 0.02,
}
_DEFAULT_FIXED_THRESHOLDS = {
    "net_debt_to_cfo_fail_max": 4.0,
    "dilution_fail_min": 0.06,
    "fcf_margin_trend_strongly_negative": -0.01,
}

_REASON_TO_CATEGORY = {
    REASON_PRICE_UNKNOWN: CATEGORY_INPUT_MISSING,
    REASON_MISSING_INPUT_SHARES: CATEGORY_INPUT_MISSING,
    REASON_MISSING_INPUT_FCF: CATEGORY_INPUT_MISSING,
    REASON_MISSING_INPUT_VALUATION: CATEGORY_INPUT_MISSING,
    REASON_FCF_WATCH_TREND_UNKNOWN: CATEGORY_INPUT_MISSING,
    REASON_BALANCE_WATCH_UNKNOWN: CATEGORY_INPUT_MISSING,
    REASON_DILUTION_WATCH_UNKNOWN: CATEGORY_INPUT_MISSING,
    REASON_MOS_FAIL: CATEGORY_VALUE_FAIL,
    REASON_MOS_WATCH: CATEGORY_VALUE_FAIL,
    REASON_FCF_FAIL_TREND: CATEGORY_SAFETY_FAIL,
    REASON_FCF_FAIL_LATEST_NEG: CATEGORY_SAFETY_FAIL,
    REASON_FCF_FAIL_MAJOR_NEG: CATEGORY_SAFETY_FAIL,
    REASON_BALANCE_FAIL: CATEGORY_SAFETY_FAIL,
    REASON_BALANCE_WATCH_MID: CATEGORY_SAFETY_FAIL,
    REASON_DILUTION_FAIL: CATEGORY_SAFETY_FAIL,
    REASON_DILUTION_WATCH_MID: CATEGORY_SAFETY_FAIL,
    REASON_MOS_PASS: CATEGORY_PASS_REASON,
    REASON_FCF_PASS: CATEGORY_PASS_REASON,
    REASON_BALANCE_PASS: CATEGORY_PASS_REASON,
    REASON_DILUTION_PASS: CATEGORY_PASS_REASON,
}

_PRIMARY_BLOCKER_ORDER = [
    REASON_PRICE_UNKNOWN,
    REASON_MISSING_INPUT_SHARES,
    REASON_MISSING_INPUT_FCF,
    REASON_MISSING_INPUT_VALUATION,
    REASON_FCF_WATCH_TREND_UNKNOWN,
    REASON_BALANCE_WATCH_UNKNOWN,
    REASON_DILUTION_WATCH_UNKNOWN,
    REASON_MOS_FAIL,
    REASON_MOS_WATCH,
    REASON_FCF_FAIL_MAJOR_NEG,
    REASON_FCF_FAIL_LATEST_NEG,
    REASON_FCF_FAIL_TREND,
    REASON_BALANCE_FAIL,
    REASON_BALANCE_WATCH_MID,
    REASON_DILUTION_FAIL,
    REASON_DILUTION_WATCH_MID,
    REASON_MOS_PASS,
    REASON_FCF_PASS,
    REASON_BALANCE_PASS,
    REASON_DILUTION_PASS,
]


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_num(value: Any) -> bool:
    # bool is a subclass of int and every ordered comparison against NaN is
    # False, so without these two exclusions True scores as one dollar and a
    # NaN slope walks past the cash-earning-power trend test into PASS while a
    # NaN intrinsic value produces a silent MOS_FAIL verdict rather than a
    # missing-input WATCH. Matches app/valuation/guards.py:_is_num.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _to_num(value: Any) -> float | str:
    return float(value) if _is_num(value) else UNKNOWN


def _sorted_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _ticker_list_from_scoreboard(scoreboard_payload: dict[str, Any]) -> list[str]:
    rows = [row for row in (scoreboard_payload.get("rows") or []) if isinstance(row, dict)]
    return [
        str(row.get("ticker") or "").upper()
        for row in rows
        if str(row.get("ticker") or "").strip()
    ]


def _latest_row(fundamentals: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    rows = sorted(rows, key=lambda row: int(row.get("year", 0)))
    return rows[-1] if rows else {}


def _signal_value(fundamentals: dict[str, Any], signal: str) -> float | str:
    bucket = fundamentals.get("derived_signals") if isinstance(fundamentals.get("derived_signals"), dict) else {}
    row = bucket.get(signal) if isinstance(bucket, dict) else None
    value = row.get("value") if isinstance(row, dict) else UNKNOWN
    return float(value) if _is_num(value) else UNKNOWN


def _signal_refs(fundamentals: dict[str, Any], signal: str) -> list[str]:
    bucket = fundamentals.get("derived_signals") if isinstance(fundamentals.get("derived_signals"), dict) else {}
    row = bucket.get(signal) if isinstance(bucket, dict) else None
    refs = [str(ref) for ref in ((row or {}).get("derived_from") or []) if str(ref).strip()]
    if refs:
        return refs
    return [f"fundamentals.derived_signals.{signal}"]


def _row_metric_refs(fundamentals: dict[str, Any], metric: str) -> list[str]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    rows = sorted(rows, key=lambda row: int(row.get("year", 0)))
    if not rows:
        return [f"fundamentals.rows[*].{metric}"]
    year = int(rows[-1].get("year", 0))
    trace_map = fundamentals.get("row_traces") if isinstance(fundamentals.get("row_traces"), dict) else {}
    year_bucket = trace_map.get(str(year)) if isinstance(trace_map, dict) else {}
    metric_bucket = year_bucket.get(metric) if isinstance(year_bucket, dict) else {}
    refs = [str(ref) for ref in ((metric_bucket or {}).get("derived_from") or []) if str(ref).strip()]
    if refs:
        return refs
    return [f"fundamentals.rows[{year}].{metric}"]


def _claim_refs(valuation: dict[str, Any], claim: str) -> list[str]:
    claims = valuation.get("claims") if isinstance(valuation.get("claims"), dict) else {}
    row = claims.get(claim) if isinstance(claims, dict) else {}
    refs = [str(ref) for ref in ((row or {}).get("derived_from") or []) if str(ref).strip()]
    if refs:
        return refs
    return [f"valuation.claims.{claim}.value"]


def _coverage_by_ticker(coverage_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in (coverage_payload.get("entries") or []):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            out[ticker] = row
    return out


def _normalized_status(value: Any) -> str:
    token = str(value or "").upper().strip()
    return "OK" if token == "OK" else "UNKNOWN"


def _price_status(*, coverage: dict[str, Any], valuation: dict[str, Any]) -> str:
    status = _normalized_status(coverage.get("price_status"))
    if status == "OK":
        return "OK"
    price = ((valuation.get("input_snapshot") or {}).get("current_price") if isinstance(valuation, dict) else UNKNOWN)
    return "OK" if _is_num(price) and float(price) > 0 else "UNKNOWN"


def _shares_status(*, coverage: dict[str, Any], valuation: dict[str, Any]) -> str:
    status = _normalized_status(coverage.get("shares_status"))
    if status == "OK":
        return "OK"
    shares = ((valuation.get("input_snapshot") or {}).get("shares_outstanding") if isinstance(valuation, dict) else UNKNOWN)
    return "OK" if _is_num(shares) and float(shares) > 0 else "UNKNOWN"


def _fcf_status(*, coverage: dict[str, Any], valuation: dict[str, Any]) -> str:
    status = _normalized_status(coverage.get("fcf_status"))
    if status == "OK":
        return "OK"
    fcf_value = ((valuation.get("input_snapshot") or {}).get("fcf_latest") if isinstance(valuation, dict) else UNKNOWN)
    return "OK" if _is_num(fcf_value) else "UNKNOWN"


def _category_counts(reasons: list[str]) -> dict[str, int]:
    counts = {category: 0 for category in _REASON_CATEGORIES}
    for reason in _sorted_unique(reasons):
        category = _REASON_TO_CATEGORY.get(reason)
        if category is None:
            continue
        counts[category] += 1
    return counts


def _primary_blocker(*, status: str, reasons: list[str]) -> str:
    if str(status).upper() == PASS:
        return "NONE"
    reason_set = set(_sorted_unique(reasons))
    for reason in _PRIMARY_BLOCKER_ORDER:
        if reason in reason_set and _REASON_TO_CATEGORY.get(reason) != CATEGORY_PASS_REASON:
            return reason
    for reason in _PRIMARY_BLOCKER_ORDER:
        if reason in reason_set:
            return reason
    return "UNKNOWN"


def normalize_gate_thresholds(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = get_config()
    thresholds = {
        "mos_min": float(getattr(cfg, "gate_mos_min", DEFAULT_GATE_THRESHOLDS["mos_min"])),
        "valuation_gap_min": float(
            getattr(cfg, "gate_valuation_gap_min", DEFAULT_GATE_THRESHOLDS["valuation_gap_min"])
        ),
        "net_debt_to_cfo_max": float(
            getattr(cfg, "gate_net_debt_to_cfo_max", DEFAULT_GATE_THRESHOLDS["net_debt_to_cfo_max"])
        ),
        "dilution_max": float(getattr(cfg, "gate_dilution_max", DEFAULT_GATE_THRESHOLDS["dilution_max"])),
    }
    for key, value in (overrides or {}).items():
        if key not in thresholds:
            continue
        if not _is_num(value):
            continue
        thresholds[key] = float(value)

    # Keep deterministic and coherent even when pass-threshold overrides are relaxed.
    thresholds["mos_min"] = max(-5.0, min(5.0, float(thresholds["mos_min"])))
    thresholds["valuation_gap_min"] = max(-5.0, min(5.0, float(thresholds["valuation_gap_min"])))
    thresholds["net_debt_to_cfo_max"] = max(0.0, min(100.0, float(thresholds["net_debt_to_cfo_max"])))
    thresholds["dilution_max"] = max(-1.0, min(1.0, float(thresholds["dilution_max"])))

    # Derived fail boundaries used for WATCH-vs-FAIL partitioning.
    thresholds["net_debt_to_cfo_fail_max"] = max(
        float(_DEFAULT_FIXED_THRESHOLDS["net_debt_to_cfo_fail_max"]),
        float(thresholds["net_debt_to_cfo_max"]) + 0.10,
    )
    thresholds["dilution_fail_min"] = max(
        float(_DEFAULT_FIXED_THRESHOLDS["dilution_fail_min"]),
        float(thresholds["dilution_max"]) + 0.001,
    )
    thresholds["fcf_margin_trend_strongly_negative"] = float(
        _DEFAULT_FIXED_THRESHOLDS["fcf_margin_trend_strongly_negative"]
    )
    return thresholds


def classify_value_gate_ticker(
    *,
    ticker: str,
    valuation: dict[str, Any] | None,
    fundamentals: dict[str, Any] | None,
    valuation_coverage_entry: dict[str, Any] | None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ticker_norm = str(ticker).upper().strip()
    valuation_payload = valuation if isinstance(valuation, dict) else {}
    fundamentals_payload = fundamentals if isinstance(fundamentals, dict) else {}
    coverage_entry = valuation_coverage_entry if isinstance(valuation_coverage_entry, dict) else {}
    gate_thresholds = normalize_gate_thresholds(thresholds)

    latest = _latest_row(fundamentals_payload)
    rows = [row for row in (fundamentals_payload.get("rows") or []) if isinstance(row, dict)]
    rows = sorted(rows, key=lambda row: int(row.get("year", 0)))
    last_three_rows = rows[-3:]

    input_snapshot = valuation_payload.get("input_snapshot") if isinstance(valuation_payload.get("input_snapshot"), dict) else {}
    current_price = input_snapshot.get("current_price", UNKNOWN)
    intrinsic_per_share = valuation_payload.get("intrinsic_per_share_base", UNKNOWN)
    fcf_latest = input_snapshot.get("fcf_latest", latest.get("fcf", UNKNOWN))
    fcf_margin_trend_slope = _signal_value(fundamentals_payload, "fcf_margin_trend_slope")
    dilution_rate = _signal_value(fundamentals_payload, "dilution_rate_shares_cagr")
    net_debt_proxy = input_snapshot.get("net_debt", latest.get("net_debt", UNKNOWN))
    cfo_proxy = input_snapshot.get("cfo_value", latest.get("cfo", UNKNOWN))

    status_price = _price_status(coverage=coverage_entry, valuation=valuation_payload)
    status_shares = _shares_status(coverage=coverage_entry, valuation=valuation_payload)
    status_fcf = _fcf_status(coverage=coverage_entry, valuation=valuation_payload)

    valuation_gap: float | str = UNKNOWN
    net_debt_to_cfo: float | str = UNKNOWN

    gate_detail: dict[str, dict[str, Any]] = {
        "mos": {"status": WATCH, "reason": REASON_MISSING_INPUT_VALUATION},
        "cash_earning_power": {"status": WATCH, "reason": REASON_MISSING_INPUT_FCF},
        "balance_sheet_safety": {"status": WATCH, "reason": REASON_BALANCE_WATCH_UNKNOWN},
        "dilution_discipline": {"status": WATCH, "reason": REASON_DILUTION_WATCH_UNKNOWN},
    }
    gate_reasons: list[str] = []
    hydration_actions: list[str] = []

    # MOS gate with strict price/shares/fcf preconditions.
    if status_price != "OK":
        gate_detail["mos"] = {"status": WATCH, "reason": REASON_PRICE_UNKNOWN}
        gate_reasons.append(REASON_PRICE_UNKNOWN)
        hydration_actions.append(ACTION_HYDRATE_PRICE)
    elif status_shares != "OK":
        gate_detail["mos"] = {"status": WATCH, "reason": REASON_MISSING_INPUT_SHARES}
        gate_reasons.append(REASON_MISSING_INPUT_SHARES)
        hydration_actions.append(ACTION_HYDRATE_FACTS)
    elif status_fcf != "OK":
        gate_detail["mos"] = {"status": WATCH, "reason": REASON_MISSING_INPUT_FCF}
        gate_reasons.append(REASON_MISSING_INPUT_FCF)
        hydration_actions.append(ACTION_HYDRATE_FACTS)
    elif _is_num(intrinsic_per_share) and _is_num(current_price) and float(current_price) > 0:
        valuation_gap = (float(intrinsic_per_share) / float(current_price)) - 1.0
        if valuation_gap >= float(gate_thresholds["mos_min"]):
            gate_detail["mos"] = {"status": PASS, "reason": REASON_MOS_PASS}
            gate_reasons.append(REASON_MOS_PASS)
        elif valuation_gap >= float(gate_thresholds["valuation_gap_min"]):
            gate_detail["mos"] = {"status": WATCH, "reason": REASON_MOS_WATCH}
            gate_reasons.append(REASON_MOS_WATCH)
        else:
            gate_detail["mos"] = {"status": FAIL, "reason": REASON_MOS_FAIL}
            gate_reasons.append(REASON_MOS_FAIL)
    else:
        gate_detail["mos"] = {"status": WATCH, "reason": REASON_MISSING_INPUT_VALUATION}
        gate_reasons.append(REASON_MISSING_INPUT_VALUATION)

    # Cash earning power gate.
    last_three_fcf = [row.get("fcf", UNKNOWN) for row in last_three_rows]
    last_three_numeric = [float(value) for value in last_three_fcf if _is_num(value)]
    negative_last_three = len([value for value in last_three_numeric if value < 0])
    if not _is_num(fcf_latest):
        gate_detail["cash_earning_power"] = {"status": WATCH, "reason": REASON_MISSING_INPUT_FCF}
        gate_reasons.append(REASON_MISSING_INPUT_FCF)
        hydration_actions.append(ACTION_HYDRATE_FACTS)
    elif float(fcf_latest) < 0:
        if len(last_three_numeric) >= 3 and negative_last_three >= 2:
            gate_detail["cash_earning_power"] = {"status": FAIL, "reason": REASON_FCF_FAIL_MAJOR_NEG}
            gate_reasons.append(REASON_FCF_FAIL_MAJOR_NEG)
        else:
            gate_detail["cash_earning_power"] = {"status": FAIL, "reason": REASON_FCF_FAIL_LATEST_NEG}
            gate_reasons.append(REASON_FCF_FAIL_LATEST_NEG)
    elif _is_num(fcf_margin_trend_slope):
        if float(fcf_margin_trend_slope) <= float(gate_thresholds["fcf_margin_trend_strongly_negative"]):
            gate_detail["cash_earning_power"] = {"status": FAIL, "reason": REASON_FCF_FAIL_TREND}
            gate_reasons.append(REASON_FCF_FAIL_TREND)
        else:
            gate_detail["cash_earning_power"] = {"status": PASS, "reason": REASON_FCF_PASS}
            gate_reasons.append(REASON_FCF_PASS)
    else:
        gate_detail["cash_earning_power"] = {"status": WATCH, "reason": REASON_FCF_WATCH_TREND_UNKNOWN}
        gate_reasons.append(REASON_FCF_WATCH_TREND_UNKNOWN)

    # Balance-sheet safety gate.
    if _is_num(net_debt_proxy) and _is_num(cfo_proxy) and float(cfo_proxy) > 0:
        net_debt_to_cfo = float(net_debt_proxy) / float(cfo_proxy)
        if net_debt_to_cfo <= float(gate_thresholds["net_debt_to_cfo_max"]):
            gate_detail["balance_sheet_safety"] = {"status": PASS, "reason": REASON_BALANCE_PASS}
            gate_reasons.append(REASON_BALANCE_PASS)
        elif net_debt_to_cfo > float(gate_thresholds["net_debt_to_cfo_fail_max"]):
            gate_detail["balance_sheet_safety"] = {"status": FAIL, "reason": REASON_BALANCE_FAIL}
            gate_reasons.append(REASON_BALANCE_FAIL)
        else:
            gate_detail["balance_sheet_safety"] = {"status": WATCH, "reason": REASON_BALANCE_WATCH_MID}
            gate_reasons.append(REASON_BALANCE_WATCH_MID)
    else:
        gate_detail["balance_sheet_safety"] = {"status": WATCH, "reason": REASON_BALANCE_WATCH_UNKNOWN}
        gate_reasons.append(REASON_BALANCE_WATCH_UNKNOWN)

    # Dilution discipline gate.
    if _is_num(dilution_rate):
        if float(dilution_rate) <= float(gate_thresholds["dilution_max"]):
            gate_detail["dilution_discipline"] = {"status": PASS, "reason": REASON_DILUTION_PASS}
            gate_reasons.append(REASON_DILUTION_PASS)
        elif float(dilution_rate) >= float(gate_thresholds["dilution_fail_min"]):
            gate_detail["dilution_discipline"] = {"status": FAIL, "reason": REASON_DILUTION_FAIL}
            gate_reasons.append(REASON_DILUTION_FAIL)
        else:
            gate_detail["dilution_discipline"] = {"status": WATCH, "reason": REASON_DILUTION_WATCH_MID}
            gate_reasons.append(REASON_DILUTION_WATCH_MID)
    else:
        gate_detail["dilution_discipline"] = {"status": WATCH, "reason": REASON_DILUTION_WATCH_UNKNOWN}
        gate_reasons.append(REASON_DILUTION_WATCH_UNKNOWN)

    core_statuses = [str((gate_detail.get(name) or {}).get("status") or WATCH).upper() for name in gate_detail]
    if any(status == FAIL for status in core_statuses):
        gate_status = FAIL
    elif all(status == PASS for status in core_statuses):
        gate_status = PASS
    else:
        gate_status = WATCH

    gate_reasons = _sorted_unique(gate_reasons)
    reason_category_counts = _category_counts(gate_reasons)
    primary_blocker = _primary_blocker(status=gate_status, reasons=gate_reasons)

    inputs_used = {
        "current_price": {
            "value": _to_num(current_price),
            "derived_from": [f"valuation_{ticker_norm}.json.input_snapshot.current_price"],
        },
        "intrinsic_per_share_base": {
            "value": _to_num(intrinsic_per_share),
            "derived_from": _claim_refs(valuation_payload, "intrinsic_per_share_base"),
        },
        "price_status": {
            "value": status_price,
            "derived_from": [f"valuation_coverage.entries[{ticker_norm}].price_status"],
        },
        "shares_status": {
            "value": status_shares,
            "derived_from": [f"valuation_coverage.entries[{ticker_norm}].shares_status"],
        },
        "fcf_status": {
            "value": status_fcf,
            "derived_from": [f"valuation_coverage.entries[{ticker_norm}].fcf_status"],
        },
        "fcf_latest": {
            "value": _to_num(fcf_latest),
            "derived_from": _row_metric_refs(fundamentals_payload, "fcf"),
        },
        "fcf_margin_trend_slope": {
            "value": _to_num(fcf_margin_trend_slope),
            "derived_from": _signal_refs(fundamentals_payload, "fcf_margin_trend_slope"),
        },
        "net_debt_proxy": {
            "value": _to_num(net_debt_proxy),
            "derived_from": _row_metric_refs(fundamentals_payload, "net_debt"),
        },
        "cfo_proxy": {
            "value": _to_num(cfo_proxy),
            "derived_from": _row_metric_refs(fundamentals_payload, "cfo"),
        },
        "dilution_rate_shares_cagr": {
            "value": _to_num(dilution_rate),
            "derived_from": _signal_refs(fundamentals_payload, "dilution_rate_shares_cagr"),
        },
    }

    fcf_signals = {
        "fcf_latest": _to_num(fcf_latest),
        "fcf_margin_trend_slope": _to_num(fcf_margin_trend_slope),
        "last_3y_negative_count": int(negative_last_three),
        "last_3y_observation_count": int(len(last_three_numeric)),
    }

    return {
        "ticker": ticker_norm,
        "gate_status": gate_status if gate_status in _VALID_GATE_STATUSES else WATCH,
        "gate_reasons": gate_reasons,
        "reason_category_counts": reason_category_counts,
        "primary_blocker": primary_blocker,
        "gates": gate_detail,
        "hydration_actions": sorted(set(_sorted_unique(hydration_actions))),
        "inputs_used": inputs_used,
        "mos": _to_num(valuation_gap),
        "valuation_gap": _to_num(valuation_gap),
        "net_debt_to_cfo": _to_num(net_debt_to_cfo),
        "dilution_rate": _to_num(dilution_rate),
        "fcf_signals": fcf_signals,
    }


def _value_sorted(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda row: (
            -float(row.get("valuation_gap")),
            str(row.get("ticker") or ""),
        )
        if _is_num(row.get("valuation_gap"))
        else (1e9, str(row.get("ticker") or "")),
    )


def _missing_input_breakdown(entries: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "price": 0,
        "shares": 0,
        "fcf": 0,
        "valuation": 0,
        "balance_inputs": 0,
        "dilution_inputs": 0,
        "fcf_trend": 0,
    }
    for row in entries:
        reasons = {str(reason) for reason in (row.get("gate_reasons") or [])}
        if REASON_PRICE_UNKNOWN in reasons:
            out["price"] += 1
        if REASON_MISSING_INPUT_SHARES in reasons:
            out["shares"] += 1
        if REASON_MISSING_INPUT_FCF in reasons:
            out["fcf"] += 1
        if REASON_MISSING_INPUT_VALUATION in reasons:
            out["valuation"] += 1
        if REASON_BALANCE_WATCH_UNKNOWN in reasons:
            out["balance_inputs"] += 1
        if REASON_DILUTION_WATCH_UNKNOWN in reasons:
            out["dilution_inputs"] += 1
        if REASON_FCF_WATCH_TREND_UNKNOWN in reasons:
            out["fcf_trend"] += 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _threshold_pressure(entries: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "mos_min": 0,
        "valuation_gap_min": 0,
        "net_debt_to_cfo_max": 0,
        "dilution_max": 0,
        "non_threshold": 0,
    }
    for row in entries:
        reasons = {str(reason) for reason in (row.get("gate_reasons") or [])}
        if REASON_MOS_FAIL in reasons:
            out["valuation_gap_min"] += 1
            continue
        if REASON_MOS_WATCH in reasons:
            out["mos_min"] += 1
            continue
        if REASON_BALANCE_FAIL in reasons or REASON_BALANCE_WATCH_MID in reasons:
            out["net_debt_to_cfo_max"] += 1
            continue
        if REASON_DILUTION_FAIL in reasons or REASON_DILUTION_WATCH_MID in reasons:
            out["dilution_max"] += 1
            continue
        out["non_threshold"] += 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _what_would_flip_row(row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    valuation_gap = row.get("valuation_gap", UNKNOWN)
    net_debt_to_cfo = row.get("net_debt_to_cfo", UNKNOWN)
    dilution_rate = row.get("dilution_rate", UNKNOWN)
    reasons = {str(reason) for reason in (row.get("gate_reasons") or [])}

    to_watch_missing: list[str] = []
    to_pass_missing: list[str] = []

    required_watch_gap = _to_num(valuation_gap)
    if not _is_num(valuation_gap):
        to_watch_missing.append("valuation_gap")

    required_pass_mos = _to_num(valuation_gap)
    if not _is_num(valuation_gap):
        to_pass_missing.append("mos")

    required_pass_net_debt = _to_num(net_debt_to_cfo)
    if not _is_num(net_debt_to_cfo):
        to_pass_missing.append("net_debt_to_cfo")

    required_pass_dilution = _to_num(dilution_rate)
    if not _is_num(dilution_rate):
        to_pass_missing.append("dilution_rate")

    delta_watch_gap = (
        round(float(thresholds["valuation_gap_min"]) - float(valuation_gap), 6)
        if _is_num(valuation_gap)
        else UNKNOWN
    )
    delta_pass_mos = (
        round(float(thresholds["mos_min"]) - float(valuation_gap), 6)
        if _is_num(valuation_gap)
        else UNKNOWN
    )
    delta_pass_net_debt = (
        round(float(net_debt_to_cfo) - float(thresholds["net_debt_to_cfo_max"]), 6)
        if _is_num(net_debt_to_cfo)
        else UNKNOWN
    )
    delta_pass_dilution = (
        round(float(dilution_rate) - float(thresholds["dilution_max"]), 6)
        if _is_num(dilution_rate)
        else UNKNOWN
    )

    return {
        "ticker": str(row.get("ticker") or ""),
        "gate_status": str(row.get("gate_status") or WATCH),
        "primary_blocker": str(row.get("primary_blocker") or "UNKNOWN"),
        "gate_reasons": sorted(reasons),
        "metrics": {
            "valuation_gap": _to_num(valuation_gap),
            "net_debt_to_cfo": _to_num(net_debt_to_cfo),
            "dilution_rate": _to_num(dilution_rate),
        },
        "to_watch": {
            "required_valuation_gap_min": required_watch_gap,
            "threshold_delta": delta_watch_gap,
            "missing": sorted(set(to_watch_missing)),
        },
        "to_pass": {
            "required_mos_min": required_pass_mos,
            "required_net_debt_to_cfo_max": required_pass_net_debt,
            "required_dilution_max": required_pass_dilution,
            "threshold_delta": {
                "mos_min": delta_pass_mos,
                "net_debt_to_cfo_max": delta_pass_net_debt,
                "dilution_max": delta_pass_dilution,
            },
            "missing": sorted(set(to_pass_missing)),
        },
    }


def _build_calibration_payload(
    *,
    run_id: str,
    entries: list[dict[str, Any]],
    thresholds: dict[str, float],
    fail_near_miss_limit: int = 5,
) -> dict[str, Any]:
    pass_entries = [row for row in entries if str(row.get("gate_status") or "").upper() == PASS]
    watch_entries = [row for row in entries if str(row.get("gate_status") or "").upper() == WATCH]
    fail_entries = [row for row in entries if str(row.get("gate_status") or "").upper() == FAIL]

    blocker_hist: dict[str, int] = {}
    for row in entries:
        blocker = str(row.get("primary_blocker") or "UNKNOWN")
        blocker_hist[blocker] = blocker_hist.get(blocker, 0) + 1
    blocker_hist = dict(sorted(blocker_hist.items(), key=lambda kv: (-kv[1], kv[0])))

    near_fail_candidates = _value_sorted(fail_entries)[: max(1, int(fail_near_miss_limit))]
    what_would_flip = [_what_would_flip_row(row, thresholds) for row in (watch_entries + near_fail_candidates)]

    pressure = _threshold_pressure(entries)
    dominant_threshold_blocker = next(iter(pressure.keys())) if pressure else "non_threshold"

    return {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "counts": {
            "PASS": len(pass_entries),
            "WATCH": len(watch_entries),
            "FAIL": len(fail_entries),
        },
        "blocker_histogram": blocker_hist,
        "missing_input_breakdown": _missing_input_breakdown(entries),
        "threshold_summary": {
            "mos_min": float(thresholds["mos_min"]),
            "valuation_gap_min": float(thresholds["valuation_gap_min"]),
            "net_debt_to_cfo_max": float(thresholds["net_debt_to_cfo_max"]),
            "dilution_max": float(thresholds["dilution_max"]),
            "net_debt_to_cfo_fail_max": float(thresholds["net_debt_to_cfo_fail_max"]),
            "dilution_fail_min": float(thresholds["dilution_fail_min"]),
            "fcf_margin_trend_strongly_negative": float(thresholds["fcf_margin_trend_strongly_negative"]),
        },
        "threshold_pressure": pressure,
        "dominant_threshold_blocker": dominant_threshold_blocker,
        "near_miss_fail_limit": int(fail_near_miss_limit),
        "what_would_flip": what_would_flip,
        "derived_from": [
            "value_gates.entries[*]",
            "valuation_*.json",
            "fundamentals_*.json",
            "valuation_coverage.json",
        ],
    }


def _top_pass_rows(entries: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    pass_rows = _value_sorted(
        [row for row in entries if str(row.get("gate_status") or "").upper() == PASS]
    )
    return [
        {
            "ticker": str(row.get("ticker") or ""),
            "valuation_gap": _to_num(row.get("valuation_gap")),
            "gate_reasons": list(row.get("gate_reasons") or []),
        }
        for row in pass_rows[: max(1, int(top_n))]
    ]


def write_value_gates_for_run(
    *,
    run_id: str,
    output_dir: Path | None = None,
    tickers: list[str] | None = None,
    threshold_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    run_dir = output_dir or (cfg.sectors_dir / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    thresholds = normalize_gate_thresholds(threshold_overrides)
    scoreboard_payload = _safe_json(run_dir / "peer_scoreboard.json")
    coverage_payload = _safe_json(run_dir / "valuation_coverage.json")
    coverage_by_ticker = _coverage_by_ticker(coverage_payload)

    scoreboard_tickers = _ticker_list_from_scoreboard(scoreboard_payload)
    if tickers:
        requested = [str(t).upper().strip() for t in tickers if str(t).strip()]
        target_tickers = _sorted_unique(requested)
    elif scoreboard_tickers:
        target_tickers = _sorted_unique(scoreboard_tickers)
    else:
        discovered = [
            str(path.stem.replace("valuation_", "")).upper()
            for path in sorted(run_dir.glob("valuation_*.json"))
            if str(path.stem).startswith("valuation_")
        ]
        target_tickers = _sorted_unique(discovered)

    entries: list[dict[str, Any]] = []
    for ticker in target_tickers:
        valuation_payload = _safe_json(run_dir / f"valuation_{ticker}.json")
        fundamentals_payload = _safe_json(run_dir / f"fundamentals_{ticker}.json")
        coverage_entry = coverage_by_ticker.get(ticker, {})
        entry = classify_value_gate_ticker(
            ticker=ticker,
            valuation=valuation_payload,
            fundamentals=fundamentals_payload,
            valuation_coverage_entry=coverage_entry,
            thresholds=thresholds,
        )
        entries.append(entry)

    entries = sorted(entries, key=lambda row: str(row.get("ticker") or ""))
    pass_count = len([row for row in entries if str(row.get("gate_status") or "").upper() == PASS])
    watch_count = len([row for row in entries if str(row.get("gate_status") or "").upper() == WATCH])
    fail_count = len([row for row in entries if str(row.get("gate_status") or "").upper() == FAIL])

    payload = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "ticker_count": len(entries),
        "threshold_summary": {
            "mos_min": float(thresholds["mos_min"]),
            "valuation_gap_min": float(thresholds["valuation_gap_min"]),
            "net_debt_to_cfo_max": float(thresholds["net_debt_to_cfo_max"]),
            "dilution_max": float(thresholds["dilution_max"]),
            "net_debt_to_cfo_fail_max": float(thresholds["net_debt_to_cfo_fail_max"]),
            "dilution_fail_min": float(thresholds["dilution_fail_min"]),
            "fcf_margin_trend_strongly_negative": float(thresholds["fcf_margin_trend_strongly_negative"]),
        },
        "summary": {
            "counts": {
                "PASS": int(pass_count),
                "WATCH": int(watch_count),
                "FAIL": int(fail_count),
            },
            "top_pass_tickers": _top_pass_rows(entries, top_n=10),
        },
        "entries": entries,
    }
    gates_path = run_dir / "value_gates.json"
    gates_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    calibration_payload = _build_calibration_payload(
        run_id=run_id,
        entries=entries,
        thresholds=thresholds,
        fail_near_miss_limit=5,
    )
    calibration_path = run_dir / "value_gates_calibration.json"
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    payload["value_gates_path"] = str(gates_path)
    payload["value_gates_calibration_path"] = str(calibration_path)
    return payload


def open_value_gates_for_run(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.sectors_dir / run_id / "value_gates.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "value_gates_path": str(path),
        }
    payload = _safe_json(path)
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    counts = payload.get("summary", {}).get("counts") if isinstance(payload.get("summary"), dict) else {}
    if not isinstance(counts, dict):
        counts = {}
    counts_out = {
        "PASS": int(counts.get("PASS", len([row for row in entries if str(row.get("gate_status") or "").upper() == PASS]))),
        "WATCH": int(counts.get("WATCH", len([row for row in entries if str(row.get("gate_status") or "").upper() == WATCH]))),
        "FAIL": int(counts.get("FAIL", len([row for row in entries if str(row.get("gate_status") or "").upper() == FAIL]))),
    }

    pass_rows = _top_pass_rows(entries, top_n=top_n)
    fail_rows = sorted(
        [row for row in entries if str(row.get("gate_status") or "").upper() == FAIL],
        key=lambda row: str(row.get("ticker") or ""),
    )
    top_fail = [
        {
            "ticker": str(row.get("ticker") or ""),
            "primary_blocker": str(row.get("primary_blocker") or "UNKNOWN"),
            "gate_reasons": list(row.get("gate_reasons") or []),
            "reason_category_counts": row.get("reason_category_counts") or {},
        }
        for row in fail_rows[: max(1, int(top_n))]
    ]

    watch_rows = [row for row in entries if str(row.get("gate_status") or "").upper() == WATCH]
    watch_price = [
        str(row.get("ticker") or "")
        for row in watch_rows
        if REASON_PRICE_UNKNOWN in {str(reason) for reason in (row.get("gate_reasons") or [])}
    ]
    watch_facts = [
        str(row.get("ticker") or "")
        for row in watch_rows
        if (
            {REASON_MISSING_INPUT_SHARES, REASON_MISSING_INPUT_FCF}
            & {str(reason) for reason in (row.get("gate_reasons") or [])}
        )
    ]
    suggestions: list[str] = []
    if watch_price:
        suggestions.append(f"hydrate price: {','.join(sorted(set(watch_price)))}")
    if watch_facts:
        suggestions.append(f"hydrate facts: {','.join(sorted(set(watch_facts)))}")
    if not suggestions and watch_rows:
        suggestions.append("hydrate price/facts for WATCH tickers")

    return {
        "run_id": run_id,
        "status": "OK",
        "value_gates_path": str(path),
        "counts": counts_out,
        "top_pass_by_valuation_gap": pass_rows,
        "top_fail_with_reasons": top_fail,
        "suggestions": suggestions,
    }


def open_value_gates_calibration_for_run(*, run_id: str, top_n: int = 10) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.sectors_dir / run_id / "value_gates_calibration.json"
    if not path.exists():
        return {
            "run_id": run_id,
            "status": "MISSING",
            "value_gates_calibration_path": str(path),
        }
    payload = _safe_json(path)
    blocker_hist = payload.get("blocker_histogram") if isinstance(payload.get("blocker_histogram"), dict) else {}
    missing_inputs = payload.get("missing_input_breakdown") if isinstance(payload.get("missing_input_breakdown"), dict) else {}
    what_would_flip = [row for row in (payload.get("what_would_flip") or []) if isinstance(row, dict)]

    blockers_top = [
        {"primary_blocker": str(key), "count": int(value)}
        for key, value in sorted(
            blocker_hist.items(),
            key=lambda kv: (-int(kv[1]), str(kv[0])),
        )[: max(1, int(top_n))]
    ]
    near_misses = []
    for row in what_would_flip[: max(1, int(top_n))]:
        near_misses.append(
            {
                "ticker": str(row.get("ticker") or ""),
                "gate_status": str(row.get("gate_status") or WATCH),
                "primary_blocker": str(row.get("primary_blocker") or "UNKNOWN"),
                "to_pass": row.get("to_pass") or {},
            }
        )

    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    suggestions: list[str] = []
    if int(counts.get("PASS", 0)) == 0 and int(counts.get("WATCH", 0)) == 0:
        suggestions.append(f"CALIBRATION_REQUIRED: no PASS/WATCH; inspect blockers and run `value-gates-calibration-open --run-id {run_id}`")
    if int(missing_inputs.get("price", 0)) > 0:
        suggestions.append("hydrate price for WATCH/FAIL input-missing tickers")
    if int(missing_inputs.get("shares", 0)) > 0 or int(missing_inputs.get("fcf", 0)) > 0:
        suggestions.append("hydrate financial facts (shares/fcf) before changing thresholds")

    return {
        "run_id": run_id,
        "status": "OK",
        "value_gates_calibration_path": str(path),
        "counts": {
            "PASS": int(counts.get("PASS", 0)),
            "WATCH": int(counts.get("WATCH", 0)),
            "FAIL": int(counts.get("FAIL", 0)),
        },
        "threshold_summary": payload.get("threshold_summary") if isinstance(payload.get("threshold_summary"), dict) else {},
        "dominant_threshold_blocker": payload.get("dominant_threshold_blocker"),
        "top_blockers": blockers_top,
        "missing_input_breakdown": missing_inputs,
        "top_near_misses": near_misses,
        "suggestions": suggestions,
    }
