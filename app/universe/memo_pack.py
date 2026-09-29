from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import utc_now_iso
from app.universe.depth_rollup import write_depth_batch_rollup
from app.util.json_io import atomic_write_json


UNKNOWN = "UNKNOWN"
OK = "OK"
_STATUS_RANK = {"FAIL": 0, "WATCH": 1, "PASS": 2}


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload, indent=2)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _to_num(value: Any) -> float | int | str:
    return value if _is_num(value) else UNKNOWN


def _int_or_zero(value: Any) -> int:
    return int(value) if _is_num(value) else 0


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


def _memo_pack_dir(universe_run_id: str, batch_run_id: str, out_dir: Path | None = None) -> Path:
    if out_dir is not None:
        return out_dir
    cfg = get_config()
    return cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "memo_pack"


def _autopilot_dir(universe_run_id: str) -> Path:
    cfg = get_config()
    return cfg.outputs_dir / "universe" / universe_run_id / "autopilot"


def _trace_refs(metric_traces: dict[str, Any], key: str) -> list[str]:
    bucket = metric_traces.get(key) if isinstance(metric_traces, dict) else {}
    if not isinstance(bucket, dict):
        return []
    return [str(ref) for ref in (bucket.get("derived_from") or []) if str(ref).strip()]


def _inputs_refs(inputs_used: dict[str, Any], key: str) -> list[str]:
    bucket = inputs_used.get(key) if isinstance(inputs_used, dict) else {}
    if not isinstance(bucket, dict):
        return []
    return [str(ref) for ref in (bucket.get("derived_from") or []) if str(ref).strip()]


def _numeric_claim(
    *,
    value: Any,
    refs: list[Any],
    reason_code: str,
    coverage_ref: str,
) -> dict[str, Any]:
    derived = _dedupe_refs([*refs, coverage_ref])
    if _is_num(value):
        return {
            "value": value,
            "status": OK,
            "reason_code": OK,
            "derived_from": derived,
            "coverage_ref": coverage_ref,
        }
    return {
        "value": UNKNOWN,
        "status": UNKNOWN,
        "reason_code": str(reason_code or UNKNOWN),
        "derived_from": derived,
        "coverage_ref": coverage_ref,
    }


def _status_snapshot(*, status: Any, reason_code: Any, coverage_ref: str) -> dict[str, Any]:
    return {
        "status": str(status or UNKNOWN).upper(),
        "reason_code": str(reason_code or UNKNOWN),
        "coverage_ref": coverage_ref,
    }


def _collect_all_refs(payload: Any) -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "derived_from" and isinstance(value, list):
                out.extend([str(ref) for ref in value if str(ref).strip()])
            else:
                out.extend(_collect_all_refs(value))
    elif isinstance(payload, list):
        for item in payload:
            out.extend(_collect_all_refs(item))
    return out


def _unknown_reason_from_blocker(blocker: str) -> str:
    token = str(blocker or "").strip().upper()
    return token if token else UNKNOWN


def _lookup_entry(payload: dict[str, Any], ticker: str) -> dict[str, Any]:
    entries = [row for row in (payload.get("entries") or []) if isinstance(row, dict)]
    ticker_norm = str(ticker or "").strip().upper()
    for row in entries:
        if str(row.get("ticker") or "").strip().upper() == ticker_norm:
            return row
    rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
    for row in rows:
        if str(row.get("ticker") or "").strip().upper() == ticker_norm:
            return row
    return {}


def load_global_shortlist(universe_run_id: str, batch_run_id: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.outputs_dir / "universe" / universe_run_id / "depth_batches" / batch_run_id / "global_shortlist.json"
    payload = _safe_json(path)
    if not payload:
        raise ValueError(
            f"Missing global_shortlist.json for universe_run_id={universe_run_id} batch_run_id={batch_run_id}"
        )
    return {
        "global_shortlist_path": str(path),
        "payload": payload,
    }


def _load_sources_for_row(*, row: dict[str, Any], ticker: str) -> dict[str, Any]:
    cfg = get_config()
    source_runs = [item for item in (row.get("source_depth_runs") or []) if isinstance(item, dict)]
    run_id = str(row.get("run_id") or "")
    if not run_id and source_runs:
        run_id = str(source_runs[0].get("run_id") or "")
    run_dir = cfg.sectors_dir / run_id if run_id else Path("")
    scoreboard_payload = _safe_json(run_dir / "peer_scoreboard.json") if run_id else {}
    gates_payload = _safe_json(run_dir / "value_gates.json") if run_id else {}
    valuation_payload = _safe_json(run_dir / "valuation_coverage.json") if run_id else {}
    shares_payload = _safe_json(run_dir / "shares_coverage.json") if run_id else {}
    fcf_payload = _safe_json(run_dir / "fcf_coverage.json") if run_id else {}
    facts_payload = _safe_json(run_dir / "facts_coverage.json") if run_id else {}

    score_row = _lookup_entry(scoreboard_payload, ticker)
    gate_row = _lookup_entry(gates_payload, ticker)
    valuation_row = _lookup_entry(valuation_payload, ticker)
    shares_row = _lookup_entry(shares_payload, ticker)
    fcf_row = _lookup_entry(fcf_payload, ticker)
    facts_row = _lookup_entry(facts_payload, ticker)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir) if run_id else "",
        "score_row": score_row,
        "gate_row": gate_row,
        "valuation_row": valuation_row,
        "shares_row": shares_row,
        "fcf_row": fcf_row,
        "facts_row": facts_row,
    }


def _fundamental_claim(
    *,
    metric_values: dict[str, Any],
    metric_traces: dict[str, Any],
    key: str,
    reason_code: str,
    coverage_ref: str,
    fallback_value: Any = UNKNOWN,
    fallback_refs: list[Any] | None = None,
) -> dict[str, Any]:
    value = metric_values.get(key, fallback_value)
    refs = _trace_refs(metric_traces, key)
    if not refs and isinstance(fallback_refs, list):
        refs = [str(ref) for ref in fallback_refs if str(ref).strip()]
    return _numeric_claim(value=value, refs=refs, reason_code=reason_code, coverage_ref=coverage_ref)


def _add_blocker(
    blockers: list[dict[str, Any]],
    *,
    field: str,
    reason_code: str,
    coverage_ref: str,
    value: Any = UNKNOWN,
) -> None:
    key = (str(field), str(reason_code), str(coverage_ref))
    existing = {(str(item.get("field")), str(item.get("reason_code")), str(item.get("coverage_ref"))) for item in blockers}
    if key in existing:
        return
    blockers.append(
        {
            "field": str(field),
            "value": _to_num(value),
            "reason_code": str(reason_code or UNKNOWN),
            "coverage_ref": str(coverage_ref),
        }
    )


def build_investment_memo(
    ticker: str,
    *,
    universe_run_id: str,
    batch_run_id: str,
    sources: dict[str, Any],
) -> dict[str, Any]:
    ticker_norm = str(ticker or "").strip().upper()
    row = sources.get("shortlist_row") if isinstance(sources.get("shortlist_row"), dict) else {}
    score_row = sources.get("score_row") if isinstance(sources.get("score_row"), dict) else {}
    gate_row = sources.get("gate_row") if isinstance(sources.get("gate_row"), dict) else {}
    valuation_row = sources.get("valuation_row") if isinstance(sources.get("valuation_row"), dict) else {}
    shares_row = sources.get("shares_row") if isinstance(sources.get("shares_row"), dict) else {}
    fcf_row = sources.get("fcf_row") if isinstance(sources.get("fcf_row"), dict) else {}
    facts_row = sources.get("facts_row") if isinstance(sources.get("facts_row"), dict) else {}

    metric_values = score_row.get("metric_values") if isinstance(score_row.get("metric_values"), dict) else {}
    metric_traces = score_row.get("metric_traces") if isinstance(score_row.get("metric_traces"), dict) else {}
    inputs_used = gate_row.get("inputs_used") if isinstance(gate_row.get("inputs_used"), dict) else {}
    gate_reasons = [str(reason) for reason in (gate_row.get("gate_reasons") or row.get("value_gate_reasons") or []) if str(reason).strip()]
    source_depth_runs = [item for item in (row.get("source_depth_runs") or []) if isinstance(item, dict)]

    sector = str(row.get("sector") or (source_depth_runs[0].get("sector") if source_depth_runs else "UNKNOWN_SECTOR"))
    as_of_date = str(
        row.get("as_of_date")
        or (source_depth_runs[0].get("as_of_date") if source_depth_runs else "")
        or UNKNOWN
    )

    valuation_reason = str(
        row.get("valuation_reason_code")
        or valuation_row.get("valuation_reason_code")
        or UNKNOWN
    )
    price_reason = str(
        row.get("price_reason_code")
        or valuation_row.get("price_reason_code")
        or UNKNOWN
    )
    shares_reason = str(
        row.get("shares_reason_code")
        or shares_row.get("shares_reason_code")
        or valuation_row.get("shares_reason_code")
        or UNKNOWN
    )
    fcf_reason = str(
        row.get("fcf_reason_code")
        or fcf_row.get("fcf_reason_code")
        or valuation_row.get("fcf_reason_code")
        or UNKNOWN
    )
    facts_reason = str(
        row.get("facts_reason_code")
        or facts_row.get("fetch_reason_code")
        or facts_row.get("status")
        or UNKNOWN
    )
    blocker_reason = _unknown_reason_from_blocker(
        row.get("primary_blocker") or gate_row.get("primary_blocker") or UNKNOWN
    )

    implied_value = row.get("implied_return_base", metric_values.get("implied_return_base", valuation_row.get("implied_return_base", UNKNOWN)))
    implied_refs = _dedupe_refs(
        list(row.get("implied_return_base_derived_from") or [])
        + _trace_refs(metric_traces, "implied_return_base")
        + [f"sectors/{sources.get('run_id')}/valuation_coverage.entries[{ticker_norm}].implied_return_base"]
    )
    implied_claim = _numeric_claim(
        value=implied_value,
        refs=implied_refs,
        reason_code=valuation_reason,
        coverage_ref=f"valuation_coverage.entries[{ticker_norm}]",
    )

    intrinsic_value = row.get(
        "intrinsic_per_share_base",
        metric_values.get("intrinsic_per_share_base", valuation_row.get("intrinsic_per_share_base", UNKNOWN)),
    )
    intrinsic_refs = _dedupe_refs(
        list(row.get("intrinsic_per_share_base_derived_from") or [])
        + _trace_refs(metric_traces, "intrinsic_per_share_base")
        + [f"sectors/{sources.get('run_id')}/valuation_coverage.entries[{ticker_norm}].intrinsic_per_share_base"]
    )
    intrinsic_claim = _numeric_claim(
        value=intrinsic_value,
        refs=intrinsic_refs,
        reason_code=valuation_reason,
        coverage_ref=f"valuation_coverage.entries[{ticker_norm}]",
    )

    price_value = (
        (inputs_used.get("current_price") or {}).get("value")
        if isinstance(inputs_used.get("current_price"), dict)
        else valuation_row.get("current_price", row.get("current_price", UNKNOWN))
    )
    price_refs = _dedupe_refs(
        _inputs_refs(inputs_used, "current_price")
        + list(valuation_row.get("derived_from") or [])
        + [f"sectors/{sources.get('run_id')}/valuation_coverage.entries[{ticker_norm}].current_price"]
    )
    price_claim = _numeric_claim(
        value=price_value,
        refs=price_refs,
        reason_code=price_reason,
        coverage_ref=f"valuation_coverage.entries[{ticker_norm}]",
    )

    mos_epv = row.get("mos_epv", metric_values.get("mos_epv", UNKNOWN))
    mos_netnet = row.get("mos_netnet", metric_values.get("mos_netnet", UNKNOWN))
    epv_per_share = row.get("epv_per_share", metric_values.get("epv_per_share", UNKNOWN))
    netnet_per_share = row.get("netnet_per_share", metric_values.get("netnet_per_share", UNKNOWN))
    gd_reason = str(row.get("gd_primary_reason_code") or blocker_reason)
    gd_status = str(
        row.get("gd_value_status")
        or ("OK" if (_is_num(mos_epv) or _is_num(mos_netnet)) else UNKNOWN)
    ).upper()

    gd_claims = {
        "epv_per_share": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="epv_per_share",
            reason_code=gd_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=epv_per_share,
            fallback_refs=list(row.get("mos_epv_derived_from") or []),
        ),
        "mos_epv": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="mos_epv",
            reason_code=gd_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=mos_epv,
            fallback_refs=list(row.get("mos_epv_derived_from") or []),
        ),
        "netnet_per_share": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="netnet_per_share",
            reason_code=gd_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=netnet_per_share,
            fallback_refs=list(row.get("mos_netnet_derived_from") or []),
        ),
        "mos_netnet": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="mos_netnet",
            reason_code=gd_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=mos_netnet,
            fallback_refs=list(row.get("mos_netnet_derived_from") or []),
        ),
        # Convention label (audit: dual-mos-convention-same-name): mos_epv /
        # mos_netnet here are UPSIDE RATIOS (intrinsic/price - 1) — NOT the
        # textbook (intrinsic-price)/intrinsic used by the scorecard's
        # margin_of_safety_* fields. Same position: upside 0.667 == textbook 0.40.
        "mos_convention": "UPSIDE_RATIO (intrinsic/price - 1); scorecard margin_of_safety_* fields use TEXTBOOK (intrinsic-price)/intrinsic",
    }

    yield_metric_used = str(row.get("yield_metric_used") or metric_values.get("yield_metric_used") or UNKNOWN)
    yield_denominator_used = str(
        row.get("yield_denominator_used")
        or metric_values.get("yield_denominator_used")
        or metric_values.get("denominator_used")
        or UNKNOWN
    )
    yield_reason = str(row.get("yield_reason_code") or blocker_reason)
    owner_yield_ev = row.get("owner_earnings_yield_ev_3y", metric_values.get("owner_earnings_yield_ev_3y", UNKNOWN))
    fcf_yield_ev = row.get("fcf_yield_ev_3y", metric_values.get("fcf_yield_ev_3y", UNKNOWN))
    owner_yield_mc = row.get("owner_earnings_yield_3y", metric_values.get("owner_earnings_yield_3y", UNKNOWN))
    fcf_yield_mc = row.get("fcf_yield_3y", metric_values.get("fcf_yield_3y", UNKNOWN))

    yield_claims = {
        "owner_earnings_yield_ev_3y": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="owner_earnings_yield_ev_3y",
            reason_code=yield_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=owner_yield_ev,
            fallback_refs=list(row.get("owner_earnings_yield_ev_3y_derived_from") or []),
        ),
        "fcf_yield_ev_3y": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="fcf_yield_ev_3y",
            reason_code=yield_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=fcf_yield_ev,
            fallback_refs=list(row.get("fcf_yield_ev_3y_derived_from") or []),
        ),
        "owner_earnings_yield_3y": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="owner_earnings_yield_3y",
            reason_code=yield_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=owner_yield_mc,
            fallback_refs=list(row.get("owner_earnings_yield_3y_derived_from") or []),
        ),
        "fcf_yield_3y": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="fcf_yield_3y",
            reason_code=yield_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=fcf_yield_mc,
            fallback_refs=list(row.get("fcf_yield_3y_derived_from") or []),
        ),
    }

    yield_status = OK if any(claim.get("status") == OK for claim in yield_claims.values()) else UNKNOWN

    net_debt_value = row.get("net_debt_proxy", metric_values.get("net_debt_proxy", (inputs_used.get("net_debt_proxy") or {}).get("value", UNKNOWN)))
    dilution_value = row.get("dilution_rate_shares_cagr", metric_values.get("dilution_rate_shares_cagr", (inputs_used.get("dilution_rate_shares_cagr") or {}).get("value", UNKNOWN)))

    fundamentals_claims = {
        "revenue_cagr_5y": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="revenue_cagr_5y",
            reason_code=facts_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
        ),
        "revenue_cagr_10y": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="revenue_cagr_10y",
            reason_code=facts_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
        ),
        "operating_margin_trend_slope": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="operating_margin_trend_slope",
            reason_code=facts_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
        ),
        "gross_margin_trend_slope": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="gross_margin_trend_slope",
            reason_code=facts_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
        ),
        "fcf_margin_trend_slope": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="fcf_margin_trend_slope",
            reason_code=fcf_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
        ),
        "roic_proxy": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="roic_proxy",
            reason_code=facts_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
        ),
        "dilution_rate_shares_cagr": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="dilution_rate_shares_cagr",
            reason_code=shares_reason,
            coverage_ref=f"value_gates.entries[{ticker_norm}]",
            fallback_value=dilution_value,
            fallback_refs=_inputs_refs(inputs_used, "dilution_rate_shares_cagr"),
        ),
        "net_debt_proxy": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="net_debt_proxy",
            reason_code=facts_reason,
            coverage_ref=f"value_gates.entries[{ticker_norm}]",
            fallback_value=net_debt_value,
            fallback_refs=_inputs_refs(inputs_used, "net_debt_proxy"),
        ),
    }

    risk_keyword_delta_claim = _fundamental_claim(
        metric_values=metric_values,
        metric_traces=metric_traces,
        key="risk_factor_keyword_delta",
        reason_code=UNKNOWN,
        coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
    )
    risk_penalty_claim = _fundamental_claim(
        metric_values=metric_values,
        metric_traces=metric_traces,
        key="risk_penalty",
        reason_code=UNKNOWN,
        coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
        fallback_value=row.get("risk_penalty", UNKNOWN),
    )
    quality_claim = _fundamental_claim(
        metric_values=metric_values,
        metric_traces=metric_traces,
        key="quality_score",
        reason_code=UNKNOWN,
        coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
        fallback_value=row.get("quality_score", UNKNOWN),
    )
    oe_quality_reason_codes = [
        str(code)
        for code in (row.get("oe_quality_reason_codes") or [])
        if str(code).strip()
    ]
    oe_reason = oe_quality_reason_codes[0] if oe_quality_reason_codes else UNKNOWN
    oe_quality_claims = {
        "owner_earnings_stability_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="owner_earnings_stability_score",
            reason_code=oe_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("owner_earnings_stability_score", UNKNOWN),
            fallback_refs=list(row.get("owner_earnings_stability_score_derived_from") or []),
        ),
        "capital_allocation_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="capital_allocation_score",
            reason_code=oe_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("capital_allocation_score", UNKNOWN),
            fallback_refs=list(row.get("capital_allocation_score_derived_from") or []),
        ),
        "cash_conversion_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="cash_conversion_score",
            reason_code=oe_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("cash_conversion_score", UNKNOWN),
            fallback_refs=list(row.get("cash_conversion_score_derived_from") or []),
        ),
        "oe_quality_total": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="oe_quality_total",
            reason_code=oe_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("oe_quality_total", UNKNOWN),
            fallback_refs=list(row.get("oe_quality_total_derived_from") or []),
        ),
    }
    intangible_reason_codes = [
        str(code)
        for code in (row.get("intangible_economics_reason_codes") or [])
        if str(code).strip()
    ]
    intangible_reason = intangible_reason_codes[0] if intangible_reason_codes else UNKNOWN
    intangible_claims = {
        "gross_margin_durability_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="gross_margin_durability_score",
            reason_code=intangible_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("gross_margin_durability_score", UNKNOWN),
            fallback_refs=list(row.get("gross_margin_durability_score_derived_from") or []),
        ),
        "balance_sheet_optionality_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="balance_sheet_optionality_score",
            reason_code=intangible_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("balance_sheet_optionality_score", UNKNOWN),
            fallback_refs=list(row.get("balance_sheet_optionality_score_derived_from") or []),
        ),
        "cycle_resilience_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="cycle_resilience_score",
            reason_code=intangible_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("cycle_resilience_score", UNKNOWN),
            fallback_refs=list(row.get("cycle_resilience_score_derived_from") or []),
        ),
        "rnd_productivity_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="rnd_productivity_score",
            reason_code=intangible_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("rnd_productivity_score", UNKNOWN),
            fallback_refs=list(row.get("rnd_productivity_score_derived_from") or []),
        ),
        "sga_leverage_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="sga_leverage_score",
            reason_code=intangible_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("sga_leverage_score", UNKNOWN),
            fallback_refs=list(row.get("sga_leverage_score_derived_from") or []),
        ),
        "owner_value_capture_score": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="owner_value_capture_score",
            reason_code=intangible_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("owner_value_capture_score", UNKNOWN),
            fallback_refs=list(row.get("owner_value_capture_score_derived_from") or []),
        ),
        "intangible_economics_total": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="intangible_economics_total",
            reason_code=intangible_reason,
            coverage_ref=f"peer_scoreboard.rows[{ticker_norm}]",
            fallback_value=row.get("intangible_economics_total", UNKNOWN),
            fallback_refs=list(row.get("intangible_economics_total_derived_from") or []),
        ),
    }
    intrinsic_reason_codes = [
        str(code)
        for code in (row.get("valuation_range_reason_codes") or [])
        if str(code).strip()
    ]
    intrinsic_reason = intrinsic_reason_codes[0] if intrinsic_reason_codes else UNKNOWN
    normalized_reason_codes = [
        str(code)
        for code in (row.get("normalized_earnings_power_reason_codes") or [])
        if str(code).strip()
    ]
    normalized_reason = normalized_reason_codes[0] if normalized_reason_codes else intrinsic_reason
    downside_reason_codes = [
        str(code)
        for code in (row.get("downside_support_reason_codes") or [])
        if str(code).strip()
    ]
    downside_reason = downside_reason_codes[0] if downside_reason_codes else intrinsic_reason
    intrinsic_claims = {
        "normalized_earnings_power_value": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="normalized_earnings_power_value",
            reason_code=normalized_reason,
            coverage_ref=f"global_shortlist.rows[{ticker_norm}]",
            fallback_value=row.get("normalized_earnings_power_value", UNKNOWN),
            fallback_refs=list(
                row.get("normalized_earnings_power_derived_from")
                or row.get("normalized_earnings_power_value_derived_from")
                or []
            ),
        ),
        "intrinsic_floor": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="intrinsic_floor",
            reason_code=intrinsic_reason,
            coverage_ref=f"global_shortlist.rows[{ticker_norm}]",
            fallback_value=row.get("intrinsic_floor", UNKNOWN),
            fallback_refs=list(row.get("intrinsic_floor_derived_from") or []),
        ),
        "intrinsic_base": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="intrinsic_base",
            reason_code=intrinsic_reason,
            coverage_ref=f"global_shortlist.rows[{ticker_norm}]",
            fallback_value=row.get("intrinsic_base", UNKNOWN),
            fallback_refs=list(row.get("intrinsic_base_derived_from") or []),
        ),
        "intrinsic_ceiling": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="intrinsic_ceiling",
            reason_code=intrinsic_reason,
            coverage_ref=f"global_shortlist.rows[{ticker_norm}]",
            fallback_value=row.get("intrinsic_ceiling", UNKNOWN),
            fallback_refs=list(row.get("intrinsic_ceiling_derived_from") or []),
        ),
        "mos_to_floor": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="mos_to_floor",
            reason_code=intrinsic_reason,
            coverage_ref=f"global_shortlist.rows[{ticker_norm}]",
            fallback_value=row.get("mos_to_floor", UNKNOWN),
            fallback_refs=list(row.get("mos_to_floor_derived_from") or []),
        ),
        "mos_to_base": _fundamental_claim(
            metric_values=metric_values,
            metric_traces=metric_traces,
            key="mos_to_base",
            reason_code=intrinsic_reason,
            coverage_ref=f"global_shortlist.rows[{ticker_norm}]",
            fallback_value=row.get("mos_to_base", UNKNOWN),
            fallback_refs=list(row.get("mos_to_base_derived_from") or []),
        ),
    }
    intrinsic_section_refs = _dedupe_refs(
        _collect_all_refs(intrinsic_claims)
        + list(
            row.get("normalized_earnings_power_derived_from")
            or row.get("normalized_earnings_power_value_derived_from")
            or []
        )
        + list(row.get("intrinsic_floor_derived_from") or [])
        + list(row.get("intrinsic_base_derived_from") or [])
        + list(row.get("intrinsic_ceiling_derived_from") or [])
        + list(row.get("mos_to_floor_derived_from") or [])
        + list(row.get("mos_to_base_derived_from") or [])
    )
    valuation_confidence_reason_codes = [
        str(code)
        for code in (row.get("valuation_confidence_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_confidence_reason = valuation_confidence_reason_codes[0] if valuation_confidence_reason_codes else UNKNOWN
    valuation_fragility_reason_codes = [
        str(code)
        for code in (row.get("valuation_fragility_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_fragility_reason = valuation_fragility_reason_codes[0] if valuation_fragility_reason_codes else valuation_confidence_reason
    valuation_convergence_reason_codes = [
        str(code)
        for code in (row.get("valuation_convergence_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_convergence_reason = valuation_convergence_reason_codes[0] if valuation_convergence_reason_codes else valuation_confidence_reason
    valuation_support_reason_codes = [
        str(code)
        for code in (row.get("valuation_support_count_reason_codes") or [])
        if str(code).strip()
    ]
    valuation_support_reason = valuation_support_reason_codes[0] if valuation_support_reason_codes else valuation_confidence_reason
    valuation_confidence_refs = _dedupe_refs(
        list(row.get("valuation_support_count_derived_from") or [])
        + list(row.get("valuation_convergence_band_pct_derived_from") or [])
        + list(row.get("valuation_fragility_status_derived_from") or [])
        + list(row.get("valuation_confidence_class_derived_from") or [])
        + list(row.get("derived_from") or [])
    )
    valuation_confidence_claims = {
        "valuation_support_count": _numeric_claim(
            value=row.get("valuation_support_count", UNKNOWN),
            refs=valuation_confidence_refs,
            reason_code=valuation_support_reason,
            coverage_ref=f"global_shortlist.rows[{ticker_norm}]",
        ),
        "valuation_convergence_band_pct": _numeric_claim(
            value=row.get("valuation_convergence_band_pct", UNKNOWN),
            refs=valuation_confidence_refs,
            reason_code=valuation_convergence_reason,
            coverage_ref=f"global_shortlist.rows[{ticker_norm}]",
        ),
        "valuation_fragility_status": {
            "value": str(row.get("valuation_fragility_status") or UNKNOWN),
            "status": OK if str(row.get("valuation_fragility_status") or UNKNOWN) != UNKNOWN else UNKNOWN,
            "reason_code": valuation_fragility_reason,
            "derived_from": valuation_confidence_refs,
            "coverage_ref": f"global_shortlist.rows[{ticker_norm}]",
        },
        "valuation_confidence_class": {
            "value": str(row.get("valuation_confidence_class") or UNKNOWN),
            "status": OK if str(row.get("valuation_confidence_class") or UNKNOWN) != UNKNOWN else UNKNOWN,
            "reason_code": valuation_confidence_reason,
            "derived_from": valuation_confidence_refs,
            "coverage_ref": f"global_shortlist.rows[{ticker_norm}]",
        },
    }

    risk_flags: list[dict[str, Any]] = []
    if _is_num(risk_keyword_delta_claim.get("value")) and float(risk_keyword_delta_claim["value"]) > 0:
        risk_flags.append({"code": "RISK_KEYWORD_DELTA", "value": risk_keyword_delta_claim["value"]})
    if _is_num(dilution_value) and float(dilution_value) >= 0.06:
        risk_flags.append({"code": "EXCESS_DILUTION", "value": dilution_value})
    net_debt_to_cfo = gate_row.get("net_debt_to_cfo", UNKNOWN)
    if _is_num(net_debt_to_cfo) and float(net_debt_to_cfo) > 2.5:
        risk_flags.append({"code": "HIGH_LEVERAGE", "value": net_debt_to_cfo})
    primary_blocker = str(row.get("primary_blocker") or gate_row.get("primary_blocker") or "NONE")
    if primary_blocker not in {"", "NONE", UNKNOWN}:
        risk_flags.append({"code": f"PRIMARY_BLOCKER_{primary_blocker}", "value": primary_blocker})

    blockers: list[dict[str, Any]] = []
    for reason in gate_reasons:
        _add_blocker(
            blockers,
            field="value_gate_reason",
            reason_code=reason,
            coverage_ref=f"value_gates.entries[{ticker_norm}]",
            value=reason,
        )

    for field, status_payload in [
        ("price_status", _status_snapshot(status=row.get("price_status") or valuation_row.get("price_status"), reason_code=price_reason, coverage_ref=f"valuation_coverage.entries[{ticker_norm}]")),
        ("valuation_status", _status_snapshot(status=row.get("valuation_status") or valuation_row.get("valuation_status"), reason_code=valuation_reason, coverage_ref=f"valuation_coverage.entries[{ticker_norm}]")),
        ("shares_status", _status_snapshot(status=row.get("shares_status") or shares_row.get("shares_status"), reason_code=shares_reason, coverage_ref=f"shares_coverage.entries[{ticker_norm}]")),
        ("fcf_status", _status_snapshot(status=row.get("fcf_status") or fcf_row.get("fcf_status"), reason_code=fcf_reason, coverage_ref=f"fcf_coverage.entries[{ticker_norm}]")),
        ("facts_status", _status_snapshot(status=row.get("facts_status") or facts_row.get("status"), reason_code=facts_reason, coverage_ref=f"facts_coverage.entries[{ticker_norm}]")),
    ]:
        if status_payload["status"] != OK:
            _add_blocker(
                blockers,
                field=field,
                reason_code=status_payload["reason_code"],
                coverage_ref=status_payload["coverage_ref"],
                value=status_payload["status"],
            )

    for key, claim in [
        ("implied_return_base", implied_claim),
        ("intrinsic_per_share_base", intrinsic_claim),
        ("current_price", price_claim),
        ("mos_epv", gd_claims["mos_epv"]),
        ("mos_netnet", gd_claims["mos_netnet"]),
        ("owner_earnings_yield_ev_3y", yield_claims["owner_earnings_yield_ev_3y"]),
        ("fcf_yield_ev_3y", yield_claims["fcf_yield_ev_3y"]),
    ]:
        if str(claim.get("status") or UNKNOWN) == UNKNOWN:
            _add_blocker(
                blockers,
                field=key,
                reason_code=str(claim.get("reason_code") or UNKNOWN),
                coverage_ref=str(claim.get("coverage_ref") or UNKNOWN),
                value=UNKNOWN,
            )

    memo = {
        "header": {
            "ticker": ticker_norm,
            "sector": sector,
            "as_of_date": as_of_date,
            "run_ids": {
                "universe_run_id": universe_run_id,
                "batch_run_id": batch_run_id,
                "depth_run_ids": [str(item.get("run_id") or "") for item in source_depth_runs if str(item.get("run_id") or "").strip()],
            },
            "generated_at": utc_now_iso(),
        },
        "decision_snapshot": {
            "value_gate_status": str(row.get("value_gate_status") or gate_row.get("gate_status") or UNKNOWN).upper(),
            "implied_return_base": _to_num(implied_claim.get("value")),
            "implied_return_reason": str(implied_claim.get("reason_code") or UNKNOWN),
            "implied_return_claim": implied_claim,
            "price_status": str(row.get("price_status") or valuation_row.get("price_status") or UNKNOWN).upper(),
            "price_reason": price_reason,
            "price_claim": price_claim,
            "valuation_status": str(row.get("valuation_status") or valuation_row.get("valuation_status") or UNKNOWN).upper(),
            "valuation_reason": valuation_reason,
            "intrinsic_per_share_base": _to_num(intrinsic_claim.get("value")),
            "intrinsic_per_share_claim": intrinsic_claim,
        },
        "graham_dodd": {
            "epv_per_share": _to_num(gd_claims["epv_per_share"].get("value")),
            "mos_epv": _to_num(gd_claims["mos_epv"].get("value")),
            "netnet_per_share": _to_num(gd_claims["netnet_per_share"].get("value")),
            "mos_netnet": _to_num(gd_claims["mos_netnet"].get("value")),
            "gd_value_status": gd_status,
            "gd_primary_reason_code": gd_reason,
            "claims": gd_claims,
        },
        "owner_earnings_and_yield": {
            "yield_metric_used": yield_metric_used,
            "yield_denominator_used": yield_denominator_used,
            "owner_earnings_yield_ev_3y": _to_num(yield_claims["owner_earnings_yield_ev_3y"].get("value")),
            "fcf_yield_ev_3y": _to_num(yield_claims["fcf_yield_ev_3y"].get("value")),
            "owner_earnings_yield_3y": _to_num(yield_claims["owner_earnings_yield_3y"].get("value")),
            "fcf_yield_3y": _to_num(yield_claims["fcf_yield_3y"].get("value")),
            "yield_status": yield_status,
            "yield_reason_code": yield_reason,
            "claims": yield_claims,
        },
        "owner_earnings_quality": {
            "owner_earnings_stability_score": _to_num(oe_quality_claims["owner_earnings_stability_score"].get("value")),
            "capital_allocation_score": _to_num(oe_quality_claims["capital_allocation_score"].get("value")),
            "cash_conversion_score": _to_num(oe_quality_claims["cash_conversion_score"].get("value")),
            "oe_quality_total": _to_num(oe_quality_claims["oe_quality_total"].get("value")),
            "oe_quality_reason_codes": oe_quality_reason_codes,
            "claims": oe_quality_claims,
        },
        "modern_intangible_economics": {
            "gross_margin_durability_score": _to_num(intangible_claims["gross_margin_durability_score"].get("value")),
            "balance_sheet_optionality_score": _to_num(intangible_claims["balance_sheet_optionality_score"].get("value")),
            "cycle_resilience_score": _to_num(intangible_claims["cycle_resilience_score"].get("value")),
            "rnd_productivity_score": _to_num(intangible_claims["rnd_productivity_score"].get("value")),
            "sga_leverage_score": _to_num(intangible_claims["sga_leverage_score"].get("value")),
            "owner_value_capture_score": _to_num(intangible_claims["owner_value_capture_score"].get("value")),
            "intangible_economics_total": _to_num(intangible_claims["intangible_economics_total"].get("value")),
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
            "intangible_economics_reason_codes": intangible_reason_codes,
            "claims": intangible_claims,
        },
        "intrinsic_value_discipline": {
            "normalized_earnings_power_value": _to_num(intrinsic_claims["normalized_earnings_power_value"].get("value")),
            "normalized_earnings_power_method_used": str(row.get("normalized_earnings_power_method_used") or UNKNOWN),
            "normalized_earnings_power_status": str(row.get("normalized_earnings_power_status") or UNKNOWN),
            "normalized_earnings_power_reason_codes": normalized_reason_codes,
            "intrinsic_floor": _to_num(intrinsic_claims["intrinsic_floor"].get("value")),
            "intrinsic_base": _to_num(intrinsic_claims["intrinsic_base"].get("value")),
            "intrinsic_ceiling": _to_num(intrinsic_claims["intrinsic_ceiling"].get("value")),
            "mos_to_floor": _to_num(intrinsic_claims["mos_to_floor"].get("value")),
            "mos_to_base": _to_num(intrinsic_claims["mos_to_base"].get("value")),
            "mos_classification": str(row.get("mos_classification") or UNKNOWN),
            "downside_support_type": str(row.get("downside_support_type") or UNKNOWN),
            "downside_support_status": str(row.get("downside_support_status") or UNKNOWN),
            "valuation_range_reason_codes": intrinsic_reason_codes,
            "downside_support_reason_codes": downside_reason_codes,
            "claims": intrinsic_claims,
            "derived_from": intrinsic_section_refs,
        },
        "evidence_sufficiency_for_mos": {
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
            "derived_from": _dedupe_refs(
                list(row.get("evidence_sufficiency_class_derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "valuation_confidence_fragility": {
            "valuation_support_count": _int_or_zero(row.get("valuation_support_count")),
            "valuation_support_types_present": [
                str(value)
                for value in (row.get("valuation_support_types_present") or [])
                if str(value).strip()
            ],
            "valuation_support_count_reason_codes": valuation_support_reason_codes,
            "valuation_convergence_status": str(row.get("valuation_convergence_status") or UNKNOWN),
            "valuation_convergence_band_pct": _to_num(row.get("valuation_convergence_band_pct", UNKNOWN)),
            "valuation_convergence_reason_codes": valuation_convergence_reason_codes,
            "valuation_fragility_status": str(row.get("valuation_fragility_status") or UNKNOWN),
            "valuation_fragility_reason_codes": valuation_fragility_reason_codes,
            "valuation_confidence_class": str(row.get("valuation_confidence_class") or UNKNOWN),
            "valuation_confidence_reason_codes": valuation_confidence_reason_codes,
            "claims": valuation_confidence_claims,
            "derived_from": valuation_confidence_refs,
        },
        "valuation_integrity_audit": {
            "valuation_integrity_class": str(row.get("valuation_integrity_class") or UNKNOWN),
            "valuation_integrity_reason_codes": [
                str(code)
                for code in (row.get("valuation_integrity_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_consistency_status": str(row.get("valuation_consistency_status") or UNKNOWN),
            "valuation_consistency_reason_codes": [
                str(code)
                for code in (row.get("valuation_consistency_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_uniformity_group_id": (
                str(row.get("valuation_uniformity_group_id"))
                if str(row.get("valuation_uniformity_group_id") or "").strip()
                else None
            ),
            "valuation_uniformity_reason_codes": [
                str(code)
                for code in (row.get("valuation_uniformity_reason_codes") or [])
                if str(code).strip()
            ],
            "valuation_input_fingerprint": str(row.get("valuation_input_fingerprint") or ""),
            "valuation_input_provenance_summary": (
                row.get("valuation_input_provenance_summary")
                if isinstance(row.get("valuation_input_provenance_summary"), dict)
                else {}
            ),
            "derived_from": _dedupe_refs(
                list(row.get("valuation_integrity_class_derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "investment_readiness_blocker_stack": {
            "investment_readiness_class": str(row.get("investment_readiness_class") or UNKNOWN),
            "investment_readiness_reason_codes": [
                str(code)
                for code in (row.get("investment_readiness_reason_codes") or [])
                if str(code).strip()
            ],
            "blocker_stack_primary": str(row.get("blocker_stack_primary") or UNKNOWN),
            "blocker_stack_secondary": str(row.get("blocker_stack_secondary") or UNKNOWN),
            "blocker_stack_all": [
                str(code)
                for code in (row.get("blocker_stack_all") or [])
                if str(code).strip()
            ],
            "blocker_stack_retryable": bool(row.get("blocker_stack_retryable")),
            "blocker_stack_structural": bool(row.get("blocker_stack_structural")),
            "readiness_support_present": [
                str(code)
                for code in (row.get("readiness_support_present") or [])
                if str(code).strip()
            ],
            "readiness_support_missing": [
                str(code)
                for code in (row.get("readiness_support_missing") or [])
                if str(code).strip()
            ],
            "readiness_support_headwinds": [
                str(code)
                for code in (row.get("readiness_support_headwinds") or [])
                if str(code).strip()
            ],
            "primary_next_step": str(row.get("primary_next_step") or UNKNOWN),
            "primary_next_step_reason": str(row.get("primary_next_step_reason") or UNKNOWN),
            "derived_from": _dedupe_refs(
                list(row.get("investment_readiness_class_derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "value_type_classification": {
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
            "derived_from": _dedupe_refs(
                list(row.get("value_type_derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "cyclical_normalization_discipline": {
            "cyclical_profile_class": str(row.get("cyclical_profile_class") or "CYCLICALITY_UNKNOWN"),
            "cycle_position_class": str(row.get("cycle_position_class") or "CYCLE_POSITION_UNKNOWN"),
            "cyclical_valuation_risk_class": str(row.get("cyclical_valuation_risk_class") or "CYCLE_RISK_UNKNOWN"),
            "conservative_cyclical_denominator": row.get("conservative_cyclical_denominator", UNKNOWN),
            "cycle_aware_value_support_summary": str(row.get("cycle_aware_value_support_summary") or ""),
            "intrinsic_cycle_awareness_status": str(row.get("intrinsic_cycle_awareness_status") or "CYCLICALITY_UNKNOWN/CYCLE_RISK_UNKNOWN"),
            "cyclical_normalization_reason_codes": [
                str(code)
                for code in (
                    (row.get("cyclical_normalization_detail") or {}).get("cyclical_normalization_reason_codes") or []
                )
                if str(code).strip()
            ],
            "derived_from": _dedupe_refs(
                list((row.get("cyclical_normalization_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "business_impairment_classification": {
            "impairment_class_primary": str(row.get("impairment_class_primary") or "IMPAIRMENT_UNKNOWN"),
            "primary_underwriting_caution": str(row.get("primary_underwriting_caution") or "CAUTION_UNKNOWN"),
            "impairment_class_reason_codes": [
                str(code)
                for code in (row.get("impairment_class_reason_codes") or [])
                if str(code).strip()
            ],
            "weakness_source_flags": (row.get("impairment_classification_detail") or {}).get("weakness_source_flags") or {},
            "support_signals": [
                str(s) for s in ((row.get("impairment_classification_detail") or {}).get("support_signals") or [])
                if str(s).strip()
            ],
            "rebuttal_signals": [
                str(s) for s in ((row.get("impairment_classification_detail") or {}).get("rebuttal_signals") or [])
                if str(s).strip()
            ],
        },
        "normalization_recovery_credibility": {
            "normalization_credibility_class": str(row.get("normalization_credibility_class") or "NORMALIZATION_CREDIBILITY_UNKNOWN"),
            "primary_normalization_caution": str(row.get("primary_normalization_caution") or "NORMALIZATION_UNCLEAR"),
            "normalization_credibility_reason_codes": [
                str(code)
                for code in (row.get("normalization_credibility_reason_codes") or [])
                if str(code).strip()
            ],
            "recovery_support_signals": [
                str(s)
                for s in ((row.get("normalization_credibility_detail") or {}).get("recovery_support_signals") or [])
                if str(s).strip()
            ],
            "recovery_headwind_signals": [
                str(s)
                for s in ((row.get("normalization_credibility_detail") or {}).get("recovery_headwind_signals") or [])
                if str(s).strip()
            ],
            "normalization_support_summary": str(
                (row.get("normalization_credibility_detail") or {}).get("normalization_support_summary") or ""
            ),
            "derived_from": _dedupe_refs(
                list((row.get("normalization_credibility_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "per_share_capital_allocation": {
            "capital_allocation_discipline_class": str(row.get("capital_allocation_discipline_class") or "CAPITAL_ALLOCATION_UNKNOWN"),
            "primary_capital_allocation_caution": str(row.get("primary_capital_allocation_caution") or "CAPITAL_ALLOCATION_UNCLEAR"),
            "capital_allocation_discipline_reason_codes": [
                str(code)
                for code in (row.get("capital_allocation_discipline_reason_codes") or [])
                if str(code).strip()
            ],
            "per_share_support_signals": [
                str(s)
                for s in ((row.get("capital_allocation_discipline_detail") or {}).get("per_share_support_signals") or [])
                if str(s).strip()
            ],
            "per_share_headwind_signals": [
                str(s)
                for s in ((row.get("capital_allocation_discipline_detail") or {}).get("per_share_headwind_signals") or [])
                if str(s).strip()
            ],
            "per_share_value_capture_summary": str(
                (row.get("capital_allocation_discipline_detail") or {}).get("per_share_value_capture_summary") or ""
            ),
            "derived_from": _dedupe_refs(
                list((row.get("capital_allocation_discipline_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "incremental_reinvestment_efficiency": {
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
            "derived_from": _dedupe_refs(
                list((row.get("reinvestment_efficiency_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "maintenance_capex_asset_intensity_discipline": {
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
            "derived_from": _dedupe_refs(
                list((row.get("maintenance_capex_discipline_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "returns_on_capital_persistence_economic_durability": {
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
            "derived_from": _dedupe_refs(
                list((row.get("returns_persistence_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "customer_concentration_revenue_dependence_risk": {
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
            "derived_from": _dedupe_refs(
                list((row.get("revenue_dependence_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "accounting_quality_cash_earnings_discipline": {
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
            "derived_from": _dedupe_refs(
                list((row.get("accounting_quality_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "balance_sheet_stress_refinancing_risk": {
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
            "derived_from": _dedupe_refs(
                list((row.get("balance_sheet_stress_detail") or {}).get("derived_from") or [])
                + list(row.get("derived_from") or [])
            ),
        },
        "fundamentals_highlights": {
            "revenue_cagr_5y": _to_num(fundamentals_claims["revenue_cagr_5y"].get("value")),
            "revenue_cagr_10y": _to_num(fundamentals_claims["revenue_cagr_10y"].get("value")),
            "operating_margin_trend_slope": _to_num(fundamentals_claims["operating_margin_trend_slope"].get("value")),
            "gross_margin_trend_slope": _to_num(fundamentals_claims["gross_margin_trend_slope"].get("value")),
            "fcf_margin_trend_slope": _to_num(fundamentals_claims["fcf_margin_trend_slope"].get("value")),
            "roic_proxy": _to_num(fundamentals_claims["roic_proxy"].get("value")),
            "dilution_rate_shares_cagr": _to_num(fundamentals_claims["dilution_rate_shares_cagr"].get("value")),
            "net_debt_proxy": _to_num(fundamentals_claims["net_debt_proxy"].get("value")),
            "claims": fundamentals_claims,
        },
        "risk_section": {
            "risk_factor_keyword_delta": _to_num(risk_keyword_delta_claim.get("value")),
            "quality_score": _to_num(quality_claim.get("value")),
            "risk_penalty": _to_num(risk_penalty_claim.get("value")),
            "risk_flags": sorted(risk_flags, key=lambda item: (str(item.get("code") or ""), str(item.get("value") or ""))),
            "claims": {
                "risk_factor_keyword_delta": risk_keyword_delta_claim,
                "quality_score": quality_claim,
                "risk_penalty": risk_penalty_claim,
            },
        },
        "unknowns_and_blockers": sorted(
            blockers,
            key=lambda item: (str(item.get("field") or ""), str(item.get("reason_code") or ""), str(item.get("coverage_ref") or "")),
        ),
    }
    memo["derived_from_index"] = _dedupe_refs(
        _collect_all_refs(memo)
        + [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()]
        + [f"global_shortlist.rows[{ticker_norm}]"]
    )
    return memo


def _watchlist_entry_for(ticker: str):
    """Look up the latest watchlist entry for ``ticker``.

    Lazy import keeps ``app.universe.memo_pack`` free of an ``app.watchlist``
    import cycle and lets pure-render callers run without a watchlist DB; any
    failure (no DB, ticker absent) yields ``None`` so the decision block still
    renders from the intrinsic fallback.
    """

    if not ticker:
        return None
    try:
        from app.watchlist import store as _watchlist_store

        return _watchlist_store.get_latest(ticker)
    except Exception:
        return None


def _decision_header_lines(header: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    """Render the decision-first ``## Decision`` block above Decision Snapshot.

    Verdict source: the watchlist price-trigger STATUS
    drives the action verb and the conviction GRADE rides alongside. When the
    ticker is not on the watchlist the buy target falls back to
    ``intrinsic_per_share_base * 0.75`` (the same anchor x 0.75 rule the price
    trigger uses), so the block always renders an action.
    """

    from app.decision.decision_block import (
        ACTION_REVIEW_AT_TARGET,
        DecisionBlock,
        action_from_status_and_grade,
        buy_now_imperative_line,
        pct_to_target as _pct_to_target,
        render_decision_block_markdown,
    )

    ticker = str(header.get("ticker") or "")
    entry = _watchlist_entry_for(ticker)

    price_claim = decision.get("price_claim") if isinstance(decision.get("price_claim"), dict) else {}
    current_price = price_claim.get("value")
    if not _is_num(current_price):
        current_price = None

    status = entry.status if entry is not None else None
    grade = entry.conviction_grade if entry is not None else None
    confidence = entry.confidence if entry is not None else None

    buy_price_target = entry.buy_price_target if entry is not None else None
    if not _is_num(buy_price_target):
        intrinsic = decision.get("intrinsic_per_share_base")
        buy_price_target = round(intrinsic * 0.75, 2) if _is_num(intrinsic) else None

    from app.watchlist.contract import is_price_trigger_eligible

    price_trigger_eligible = (
        is_price_trigger_eligible(entry) if entry is not None else True
    )
    if entry is not None and not price_trigger_eligible:
        action = "Research only"
        presented_buy_price_target = None
    else:
        action = action_from_status_and_grade(
            status,
            grade,
            buy_price_target,
            price_trigger_eligible=price_trigger_eligible,
        )
        presented_buy_price_target = buy_price_target

    note: str | None = None
    value_gate_status = str(decision.get("value_gate_status") or UNKNOWN).upper()
    if value_gate_status == "PASS" and grade is not None and grade != "ACTIONABLE":
        note = (
            f"value gate PASS but watchlist grade {grade}; "
            "action follows the price-trigger status, grade sizes conviction"
        )

    block = DecisionBlock(
        action=action,
        current_price=current_price,
        buy_price_target=presented_buy_price_target,
        pct_to_target=_pct_to_target(current_price, presented_buy_price_target),
        base_case_expected_return=None,
        conviction_grade=grade,
        confidence=confidence,
        price_trigger_status=status,
        time_horizon="3-5 yr",
        what_would_change_my_mind=[],
        verdict_reconciliation_note=note,
    )

    lines = render_decision_block_markdown(block).split("\n")
    if action == ACTION_REVIEW_AT_TARGET:
        lines.append(
            buy_now_imperative_line(
                ticker, current_price, presented_buy_price_target, None, grade
            )
        )
    lines.append("")
    return lines


def _memo_markdown(memo: dict[str, Any]) -> str:
    header = memo.get("header") if isinstance(memo.get("header"), dict) else {}
    decision = memo.get("decision_snapshot") if isinstance(memo.get("decision_snapshot"), dict) else {}
    gd = memo.get("graham_dodd") if isinstance(memo.get("graham_dodd"), dict) else {}
    yields = memo.get("owner_earnings_and_yield") if isinstance(memo.get("owner_earnings_and_yield"), dict) else {}
    oe_quality = memo.get("owner_earnings_quality") if isinstance(memo.get("owner_earnings_quality"), dict) else {}
    intangible = (
        memo.get("modern_intangible_economics")
        if isinstance(memo.get("modern_intangible_economics"), dict)
        else {}
    )
    intrinsic = (
        memo.get("intrinsic_value_discipline")
        if isinstance(memo.get("intrinsic_value_discipline"), dict)
        else {}
    )
    evidence_sufficiency = (
        memo.get("evidence_sufficiency_for_mos")
        if isinstance(memo.get("evidence_sufficiency_for_mos"), dict)
        else {}
    )
    valuation_confidence = (
        memo.get("valuation_confidence_fragility")
        if isinstance(memo.get("valuation_confidence_fragility"), dict)
        else {}
    )
    valuation_integrity = (
        memo.get("valuation_integrity_audit")
        if isinstance(memo.get("valuation_integrity_audit"), dict)
        else {}
    )
    readiness = (
        memo.get("investment_readiness_blocker_stack")
        if isinstance(memo.get("investment_readiness_blocker_stack"), dict)
        else {}
    )
    value_type = (
        memo.get("value_type_classification")
        if isinstance(memo.get("value_type_classification"), dict)
        else {}
    )
    fundamentals = memo.get("fundamentals_highlights") if isinstance(memo.get("fundamentals_highlights"), dict) else {}
    risk = memo.get("risk_section") if isinstance(memo.get("risk_section"), dict) else {}
    impairment_cls = (
        memo.get("business_impairment_classification")
        if isinstance(memo.get("business_impairment_classification"), dict)
        else {}
    )
    norm_cred = (
        memo.get("normalization_recovery_credibility")
        if isinstance(memo.get("normalization_recovery_credibility"), dict)
        else {}
    )
    cap_alloc = (
        memo.get("per_share_capital_allocation")
        if isinstance(memo.get("per_share_capital_allocation"), dict)
        else {}
    )
    reinvestment = (
        memo.get("incremental_reinvestment_efficiency")
        if isinstance(memo.get("incremental_reinvestment_efficiency"), dict)
        else {}
    )
    maintenance_capex = (
        memo.get("maintenance_capex_asset_intensity_discipline")
        if isinstance(memo.get("maintenance_capex_asset_intensity_discipline"), dict)
        else {}
    )
    returns_persistence = (
        memo.get("returns_on_capital_persistence_economic_durability")
        if isinstance(memo.get("returns_on_capital_persistence_economic_durability"), dict)
        else {}
    )
    revenue_dependence = (
        memo.get("customer_concentration_revenue_dependence_risk")
        if isinstance(memo.get("customer_concentration_revenue_dependence_risk"), dict)
        else {}
    )
    accounting_quality = (
        memo.get("accounting_quality_cash_earnings_discipline")
        if isinstance(memo.get("accounting_quality_cash_earnings_discipline"), dict)
        else {}
    )
    balance_sheet = (
        memo.get("balance_sheet_stress_refinancing_risk")
        if isinstance(memo.get("balance_sheet_stress_refinancing_risk"), dict)
        else {}
    )

    lines: list[str] = []
    lines.append(f"# Investment Memo: {header.get('ticker', '')}")
    lines.append("")
    lines.append("## Header")
    lines.append(f"- ticker: `{header.get('ticker', '')}`")
    lines.append(f"- sector: `{header.get('sector', UNKNOWN)}`")
    lines.append(f"- as_of_date: `{header.get('as_of_date', UNKNOWN)}`")
    run_ids = header.get("run_ids") if isinstance(header.get("run_ids"), dict) else {}
    lines.append(f"- universe_run_id: `{run_ids.get('universe_run_id', '')}`")
    lines.append(f"- batch_run_id: `{run_ids.get('batch_run_id', '')}`")
    lines.append(f"- depth_run_ids: `{', '.join([str(x) for x in (run_ids.get('depth_run_ids') or [])])}`")
    lines.append(f"- generated_at: `{header.get('generated_at', '')}`")
    lines.append("")

    lines.extend(_decision_header_lines(header, decision))

    lines.append("## Decision Snapshot")
    lines.append(f"- value_gate_status: `{decision.get('value_gate_status', UNKNOWN)}`")
    lines.append(f"- implied_return_base: `{decision.get('implied_return_base', UNKNOWN)}`")
    lines.append(f"- implied_return_reason: `{decision.get('implied_return_reason', UNKNOWN)}`")
    lines.append(f"- price_status: `{decision.get('price_status', UNKNOWN)}`")
    lines.append(f"- price_reason: `{decision.get('price_reason', UNKNOWN)}`")
    lines.append(f"- valuation_status: `{decision.get('valuation_status', UNKNOWN)}`")
    lines.append(f"- valuation_reason: `{decision.get('valuation_reason', UNKNOWN)}`")
    lines.append("")

    lines.append("## Graham/Dodd")
    lines.append(f"- epv_per_share: `{gd.get('epv_per_share', UNKNOWN)}`")
    lines.append(f"- mos_epv: `{gd.get('mos_epv', UNKNOWN)}`")
    lines.append(f"- netnet_per_share: `{gd.get('netnet_per_share', UNKNOWN)}`")
    lines.append(f"- mos_netnet: `{gd.get('mos_netnet', UNKNOWN)}`")
    lines.append(f"- gd_value_status: `{gd.get('gd_value_status', UNKNOWN)}`")
    lines.append(f"- gd_primary_reason_code: `{gd.get('gd_primary_reason_code', UNKNOWN)}`")
    lines.append("")

    lines.append("## Owner Earnings and Yield")
    lines.append(f"- yield_metric_used: `{yields.get('yield_metric_used', UNKNOWN)}`")
    lines.append(f"- yield_denominator_used: `{yields.get('yield_denominator_used', UNKNOWN)}`")
    lines.append(f"- owner_earnings_yield_ev_3y: `{yields.get('owner_earnings_yield_ev_3y', UNKNOWN)}`")
    lines.append(f"- fcf_yield_ev_3y: `{yields.get('fcf_yield_ev_3y', UNKNOWN)}`")
    lines.append(f"- owner_earnings_yield_3y: `{yields.get('owner_earnings_yield_3y', UNKNOWN)}`")
    lines.append(f"- fcf_yield_3y: `{yields.get('fcf_yield_3y', UNKNOWN)}`")
    lines.append(f"- yield_status: `{yields.get('yield_status', UNKNOWN)}`")
    lines.append(f"- yield_reason_code: `{yields.get('yield_reason_code', UNKNOWN)}`")
    lines.append("")

    lines.append("## Owner Earnings Quality / Capital Allocation")
    lines.append(f"- owner_earnings_stability_score: `{oe_quality.get('owner_earnings_stability_score', UNKNOWN)}`")
    lines.append(f"- capital_allocation_score: `{oe_quality.get('capital_allocation_score', UNKNOWN)}`")
    lines.append(f"- cash_conversion_score: `{oe_quality.get('cash_conversion_score', UNKNOWN)}`")
    lines.append(f"- oe_quality_total: `{oe_quality.get('oe_quality_total', UNKNOWN)}`")
    lines.append(
        f"- oe_quality_reason_codes: `{', '.join([str(code) for code in (oe_quality.get('oe_quality_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Modern Intangible Economics Overlay")
    lines.append(f"- gross_margin_durability_score: `{intangible.get('gross_margin_durability_score', UNKNOWN)}`")
    lines.append(f"- balance_sheet_optionality_score: `{intangible.get('balance_sheet_optionality_score', UNKNOWN)}`")
    lines.append(f"- cycle_resilience_score: `{intangible.get('cycle_resilience_score', UNKNOWN)}`")
    lines.append(f"- rnd_productivity_score: `{intangible.get('rnd_productivity_score', UNKNOWN)}`")
    lines.append(
        f"- rnd_productivity_reason_codes: `{', '.join([str(code) for code in (intangible.get('rnd_productivity_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(f"- sga_leverage_score: `{intangible.get('sga_leverage_score', UNKNOWN)}`")
    lines.append(
        f"- sga_leverage_reason_codes: `{', '.join([str(code) for code in (intangible.get('sga_leverage_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(f"- owner_value_capture_score: `{intangible.get('owner_value_capture_score', UNKNOWN)}`")
    lines.append(
        f"- owner_value_capture_reason_codes: `{', '.join([str(code) for code in (intangible.get('owner_value_capture_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(f"- intangible_economics_total: `{intangible.get('intangible_economics_total', UNKNOWN)}`")
    lines.append(
        f"- intangible_economics_reason_codes: `{', '.join([str(code) for code in (intangible.get('intangible_economics_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in _collect_all_refs(intangible) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Intrinsic Value Discipline")
    lines.append(
        f"- normalized_earnings_power_method_used: `{intrinsic.get('normalized_earnings_power_method_used', UNKNOWN)}`"
    )
    lines.append(
        f"- normalized_earnings_power_value: `{intrinsic.get('normalized_earnings_power_value', UNKNOWN)}`"
    )
    lines.append(
        f"- normalized_earnings_power_status: `{intrinsic.get('normalized_earnings_power_status', UNKNOWN)}`"
    )
    lines.append(
        f"- normalized_earnings_power_reason_codes: `{', '.join([str(code) for code in (intrinsic.get('normalized_earnings_power_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(f"- intrinsic_floor: `{intrinsic.get('intrinsic_floor', UNKNOWN)}`")
    lines.append(f"- intrinsic_base: `{intrinsic.get('intrinsic_base', UNKNOWN)}`")
    lines.append(f"- intrinsic_ceiling: `{intrinsic.get('intrinsic_ceiling', UNKNOWN)}`")
    lines.append(f"- mos_to_floor: `{intrinsic.get('mos_to_floor', UNKNOWN)}`")
    lines.append(f"- mos_to_base: `{intrinsic.get('mos_to_base', UNKNOWN)}`")
    lines.append(f"- mos_classification: `{intrinsic.get('mos_classification', UNKNOWN)}`")
    lines.append(f"- downside_support_type: `{intrinsic.get('downside_support_type', UNKNOWN)}`")
    lines.append(f"- downside_support_status: `{intrinsic.get('downside_support_status', UNKNOWN)}`")
    lines.append(
        f"- valuation_range_reason_codes: `{', '.join([str(code) for code in (intrinsic.get('valuation_range_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- downside_support_reason_codes: `{', '.join([str(code) for code in (intrinsic.get('downside_support_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (intrinsic.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Evidence Sufficiency for Margin of Safety")
    lines.append(
        f"- evidence_sufficiency_class: `{evidence_sufficiency.get('evidence_sufficiency_class', UNKNOWN)}`"
    )
    lines.append(
        f"- mos_assessment_status: `{evidence_sufficiency.get('mos_assessment_status', UNKNOWN)}`"
    )
    lines.append(
        f"- evidence_sufficiency_reason_codes: `{', '.join([str(code) for code in (evidence_sufficiency.get('evidence_sufficiency_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- mos_guardrail_reason_codes: `{', '.join([str(code) for code in (evidence_sufficiency.get('mos_guardrail_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (evidence_sufficiency.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Valuation Confidence / Fragility")
    lines.append(f"- valuation_support_count: `{valuation_confidence.get('valuation_support_count', UNKNOWN)}`")
    lines.append(
        f"- valuation_support_types_present: `{', '.join([str(value) for value in (valuation_confidence.get('valuation_support_types_present') or []) if str(value).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- valuation_support_count_reason_codes: `{', '.join([str(code) for code in (valuation_confidence.get('valuation_support_count_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- valuation_convergence_status: `{valuation_confidence.get('valuation_convergence_status', UNKNOWN)}`"
    )
    lines.append(
        f"- valuation_convergence_band_pct: `{valuation_confidence.get('valuation_convergence_band_pct', UNKNOWN)}`"
    )
    lines.append(
        f"- valuation_convergence_reason_codes: `{', '.join([str(code) for code in (valuation_confidence.get('valuation_convergence_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- valuation_fragility_status: `{valuation_confidence.get('valuation_fragility_status', UNKNOWN)}`"
    )
    lines.append(
        f"- valuation_fragility_reason_codes: `{', '.join([str(code) for code in (valuation_confidence.get('valuation_fragility_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- valuation_confidence_class: `{valuation_confidence.get('valuation_confidence_class', UNKNOWN)}`"
    )
    lines.append(
        f"- valuation_confidence_reason_codes: `{', '.join([str(code) for code in (valuation_confidence.get('valuation_confidence_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (valuation_confidence.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Valuation Integrity Audit")
    lines.append(
        f"- valuation_integrity_class: `{valuation_integrity.get('valuation_integrity_class', UNKNOWN)}`"
    )
    lines.append(
        f"- valuation_consistency_status: `{valuation_integrity.get('valuation_consistency_status', UNKNOWN)}`"
    )
    lines.append(
        f"- valuation_uniformity_group_id: `{valuation_integrity.get('valuation_uniformity_group_id', UNKNOWN) if valuation_integrity.get('valuation_uniformity_group_id') else UNKNOWN}`"
    )
    lines.append(
        f"- valuation_uniformity_reason_codes: `{', '.join([str(code) for code in (valuation_integrity.get('valuation_uniformity_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- valuation_integrity_reason_codes: `{', '.join([str(code) for code in (valuation_integrity.get('valuation_integrity_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- valuation_consistency_reason_codes: `{', '.join([str(code) for code in (valuation_integrity.get('valuation_consistency_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- valuation_input_fingerprint: `{valuation_integrity.get('valuation_input_fingerprint', UNKNOWN) or UNKNOWN}`"
    )
    lines.append(
        f"- valuation_input_provenance_summary: `{json.dumps(valuation_integrity.get('valuation_input_provenance_summary') or {}, sort_keys=True) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (valuation_integrity.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Investment Readiness / Blocker Stack")
    lines.append(
        f"- investment_readiness_class: `{readiness.get('investment_readiness_class', UNKNOWN)}`"
    )
    lines.append(
        f"- investment_readiness_reason_codes: `{', '.join([str(code) for code in (readiness.get('investment_readiness_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- blocker_stack_primary: `{readiness.get('blocker_stack_primary', UNKNOWN)}`"
    )
    lines.append(
        f"- blocker_stack_secondary: `{readiness.get('blocker_stack_secondary', UNKNOWN)}`"
    )
    lines.append(
        f"- blocker_stack_all: `{', '.join([str(code) for code in (readiness.get('blocker_stack_all') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- blocker_stack_retryable: `{readiness.get('blocker_stack_retryable', False)}`"
    )
    lines.append(
        f"- blocker_stack_structural: `{readiness.get('blocker_stack_structural', False)}`"
    )
    lines.append(
        f"- readiness_support_present: `{', '.join([str(code) for code in (readiness.get('readiness_support_present') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- readiness_support_missing: `{', '.join([str(code) for code in (readiness.get('readiness_support_missing') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- readiness_support_headwinds: `{', '.join([str(code) for code in (readiness.get('readiness_support_headwinds') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(f"- primary_next_step: `{readiness.get('primary_next_step', UNKNOWN)}`")
    lines.append(
        f"- primary_next_step_reason: `{readiness.get('primary_next_step_reason', UNKNOWN)}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (readiness.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Value Type Classification")
    lines.append(f"- value_type_primary: `{value_type.get('value_type_primary', UNKNOWN)}`")
    lines.append(
        f"- value_type_secondary: `{value_type.get('value_type_secondary', UNKNOWN) if value_type.get('value_type_secondary') else UNKNOWN}`"
    )
    lines.append(
        f"- value_type_support_summary: `{value_type.get('value_type_support_summary', UNKNOWN) or UNKNOWN}`"
    )
    lines.append(
        f"- value_type_reason_codes: `{', '.join([str(code) for code in (value_type.get('value_type_reason_codes') or []) if str(code).strip()]) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (value_type.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Business Impairment vs Temporary Weakness")
    lines.append(f"- impairment_class_primary: `{impairment_cls.get('impairment_class_primary', 'IMPAIRMENT_UNKNOWN')}`")
    lines.append(f"- primary_underwriting_caution: `{impairment_cls.get('primary_underwriting_caution', 'CAUTION_UNKNOWN')}`")
    lines.append(
        f"- impairment_class_reason_codes: `{', '.join([str(c) for c in (impairment_cls.get('impairment_class_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    weakness_flags = impairment_cls.get("weakness_source_flags") or {}
    if weakness_flags:
        lines.append(
            f"- weakness_source_flags: `{', '.join(k for k, v in weakness_flags.items() if v) or 'none'}`"
        )
    support_sigs = impairment_cls.get("support_signals") or []
    lines.append(f"- support_signals: `{', '.join(support_sigs) or 'none'}`")
    rebuttal_sigs = impairment_cls.get("rebuttal_signals") or []
    lines.append(f"- rebuttal_signals: `{', '.join(rebuttal_sigs) or 'none'}`")
    lines.append("")

    lines.append("## Normalization / Recovery Credibility")
    lines.append(f"- normalization_credibility_class: `{norm_cred.get('normalization_credibility_class', 'NORMALIZATION_CREDIBILITY_UNKNOWN')}`")
    lines.append(f"- primary_normalization_caution: `{norm_cred.get('primary_normalization_caution', 'NORMALIZATION_UNCLEAR')}`")
    lines.append(f"- normalization_support_summary: `{norm_cred.get('normalization_support_summary', UNKNOWN) or UNKNOWN}`")
    lines.append(
        f"- normalization_credibility_reason_codes: `{', '.join([str(c) for c in (norm_cred.get('normalization_credibility_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    recovery_support = norm_cred.get("recovery_support_signals") or []
    lines.append(f"- recovery_support_signals: `{', '.join(recovery_support) or 'none'}`")
    recovery_headwinds = norm_cred.get("recovery_headwind_signals") or []
    lines.append(f"- recovery_headwind_signals: `{', '.join(recovery_headwinds) or 'none'}`")
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (norm_cred.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Per-Share Capital Allocation Discipline")
    lines.append(f"- capital_allocation_discipline_class: `{cap_alloc.get('capital_allocation_discipline_class', 'CAPITAL_ALLOCATION_UNKNOWN')}`")
    lines.append(f"- primary_capital_allocation_caution: `{cap_alloc.get('primary_capital_allocation_caution', 'CAPITAL_ALLOCATION_UNCLEAR')}`")
    lines.append(f"- per_share_value_capture_summary: `{cap_alloc.get('per_share_value_capture_summary', UNKNOWN) or UNKNOWN}`")
    lines.append(
        f"- capital_allocation_discipline_reason_codes: `{', '.join([str(c) for c in (cap_alloc.get('capital_allocation_discipline_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    per_share_support = cap_alloc.get("per_share_support_signals") or []
    lines.append(f"- per_share_support_signals: `{', '.join(per_share_support) or 'none'}`")
    per_share_headwinds = cap_alloc.get("per_share_headwind_signals") or []
    lines.append(f"- per_share_headwind_signals: `{', '.join(per_share_headwinds) or 'none'}`")
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (cap_alloc.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Incremental Reinvestment Efficiency")
    lines.append(
        f"- reinvestment_efficiency_class: `{reinvestment.get('reinvestment_efficiency_class', 'REINVESTMENT_EFFICIENCY_UNKNOWN')}`"
    )
    lines.append(
        f"- primary_reinvestment_caution: `{reinvestment.get('primary_reinvestment_caution', 'REINVESTMENT_UNCLEAR')}`"
    )
    lines.append(
        f"- reinvestment_efficiency_reason_codes: `{', '.join([str(c) for c in (reinvestment.get('reinvestment_efficiency_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- reinvestment_support_signals: `{', '.join([str(c) for c in (reinvestment.get('reinvestment_support_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- reinvestment_headwind_signals: `{', '.join([str(c) for c in (reinvestment.get('reinvestment_headwind_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- reinvestment_efficiency_summary: `{reinvestment.get('reinvestment_efficiency_summary', UNKNOWN) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (reinvestment.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Maintenance Capex / Asset Intensity Discipline")
    lines.append(
        f"- asset_intensity_class: `{maintenance_capex.get('asset_intensity_class', 'ASSET_INTENSITY_UNKNOWN')}`"
    )
    lines.append(
        f"- maintenance_capex_credibility_class: `{maintenance_capex.get('maintenance_capex_credibility_class', 'MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN')}`"
    )
    lines.append(
        f"- primary_maintenance_capex_caution: `{maintenance_capex.get('primary_maintenance_capex_caution', 'OWNER_EARNINGS_UNCLEAR')}`"
    )
    lines.append(
        f"- asset_intensity_reason_codes: `{', '.join([str(c) for c in (maintenance_capex.get('asset_intensity_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- maintenance_capex_credibility_reason_codes: `{', '.join([str(c) for c in (maintenance_capex.get('maintenance_capex_credibility_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- maintenance_capex_support_signals: `{', '.join([str(c) for c in (maintenance_capex.get('maintenance_capex_support_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- maintenance_capex_headwind_signals: `{', '.join([str(c) for c in (maintenance_capex.get('maintenance_capex_headwind_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- maintenance_capex_discipline_summary: `{maintenance_capex.get('maintenance_capex_discipline_summary', UNKNOWN) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (maintenance_capex.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Returns on Capital Persistence / Economic Durability")
    lines.append(
        f"- returns_persistence_class: `{returns_persistence.get('returns_persistence_class', 'RETURNS_PERSISTENCE_UNKNOWN')}`"
    )
    lines.append(
        f"- primary_returns_caution: `{returns_persistence.get('primary_returns_caution', 'RETURNS_DURABILITY_UNCLEAR')}`"
    )
    lines.append(
        f"- returns_persistence_reason_codes: `{', '.join([str(c) for c in (returns_persistence.get('returns_persistence_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- returns_support_signals: `{', '.join([str(c) for c in (returns_persistence.get('returns_support_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- returns_headwind_signals: `{', '.join([str(c) for c in (returns_persistence.get('returns_headwind_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- economic_durability_summary: `{returns_persistence.get('economic_durability_summary', UNKNOWN) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (returns_persistence.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Customer Concentration / Revenue Dependence Risk")
    lines.append(
        f"- revenue_dependence_risk_class: `{revenue_dependence.get('revenue_dependence_risk_class', 'REVENUE_DEPENDENCE_UNKNOWN')}`"
    )
    lines.append(
        f"- primary_revenue_dependence_caution: `{revenue_dependence.get('primary_revenue_dependence_caution', 'REVENUE_BASE_UNCLEAR')}`"
    )
    lines.append(
        f"- revenue_dependence_risk_reason_codes: `{', '.join([str(c) for c in (revenue_dependence.get('revenue_dependence_risk_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- revenue_dependence_support_signals: `{', '.join([str(c) for c in (revenue_dependence.get('revenue_dependence_support_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- revenue_dependence_headwind_signals: `{', '.join([str(c) for c in (revenue_dependence.get('revenue_dependence_headwind_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- revenue_fragility_summary: `{revenue_dependence.get('revenue_fragility_summary', UNKNOWN) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (revenue_dependence.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Accounting Quality / Cash Earnings Discipline")
    lines.append(
        f"- accounting_quality_class: `{accounting_quality.get('accounting_quality_class', 'ACCOUNTING_QUALITY_UNKNOWN')}`"
    )
    lines.append(
        f"- primary_accounting_caution: `{accounting_quality.get('primary_accounting_caution', 'ACCOUNTING_QUALITY_UNCLEAR')}`"
    )
    lines.append(
        f"- accounting_quality_reason_codes: `{', '.join([str(c) for c in (accounting_quality.get('accounting_quality_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- cash_earnings_support_signals: `{', '.join([str(c) for c in (accounting_quality.get('cash_earnings_support_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- cash_earnings_headwind_signals: `{', '.join([str(c) for c in (accounting_quality.get('cash_earnings_headwind_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- cash_earnings_discipline_summary: `{accounting_quality.get('cash_earnings_discipline_summary', UNKNOWN) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (accounting_quality.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Balance Sheet Stress / Refinancing Risk")
    lines.append(
        f"- balance_sheet_stress_class: `{balance_sheet.get('balance_sheet_stress_class', 'BALANCE_SHEET_STRESS_UNKNOWN')}`"
    )
    lines.append(
        f"- refinancing_risk_class: `{balance_sheet.get('refinancing_risk_class', 'REFINANCING_RISK_UNKNOWN')}`"
    )
    lines.append(
        f"- primary_balance_sheet_caution: `{balance_sheet.get('primary_balance_sheet_caution', 'BALANCE_SHEET_UNCLEAR')}`"
    )
    lines.append(
        f"- balance_sheet_stress_reason_codes: `{', '.join([str(c) for c in (balance_sheet.get('balance_sheet_stress_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- refinancing_risk_reason_codes: `{', '.join([str(c) for c in (balance_sheet.get('refinancing_risk_reason_codes') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- balance_sheet_support_signals: `{', '.join([str(c) for c in (balance_sheet.get('balance_sheet_support_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- balance_sheet_headwind_signals: `{', '.join([str(c) for c in (balance_sheet.get('balance_sheet_headwind_signals') or []) if str(c).strip()]) or 'none'}`"
    )
    lines.append(
        f"- balance_sheet_discipline_summary: `{balance_sheet.get('balance_sheet_discipline_summary', UNKNOWN) or UNKNOWN}`"
    )
    lines.append(
        f"- derived_from: `{', '.join([str(ref) for ref in (balance_sheet.get('derived_from') or []) if str(ref).strip()]) or UNKNOWN}`"
    )
    lines.append("")

    lines.append("## Fundamentals Highlights")
    for key in [
        "revenue_cagr_5y",
        "revenue_cagr_10y",
        "operating_margin_trend_slope",
        "gross_margin_trend_slope",
        "fcf_margin_trend_slope",
        "roic_proxy",
        "dilution_rate_shares_cagr",
        "net_debt_proxy",
    ]:
        lines.append(f"- {key}: `{fundamentals.get(key, UNKNOWN)}`")
    lines.append("")

    lines.append("## Risk Section")
    lines.append(f"- risk_factor_keyword_delta: `{risk.get('risk_factor_keyword_delta', UNKNOWN)}`")
    lines.append(f"- quality_score: `{risk.get('quality_score', UNKNOWN)}`")
    lines.append(f"- risk_penalty: `{risk.get('risk_penalty', UNKNOWN)}`")
    lines.append("- risk_flags:")
    for item in (risk.get("risk_flags") or []):
        if not isinstance(item, dict):
            continue
        lines.append(f"  - `{item.get('code', UNKNOWN)}`: `{item.get('value', UNKNOWN)}`")
    if not (risk.get("risk_flags") or []):
        lines.append("  - none")
    lines.append("")

    lines.append("## Unknowns and Blockers")
    blockers = memo.get("unknowns_and_blockers") if isinstance(memo.get("unknowns_and_blockers"), list) else []
    if blockers:
        for item in blockers:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- field=`{item.get('field', '')}` reason=`{item.get('reason_code', UNKNOWN)}` "
                f"coverage=`{item.get('coverage_ref', UNKNOWN)}`"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Derived From Index")
    for ref in memo.get("derived_from_index") or []:
        lines.append(f"- `{str(ref)}`")
    return "\n".join(lines).rstrip() + "\n"


def update_watchlist_state(prev_state_path: Path | None, new_run_context: dict[str, Any]) -> dict[str, Any]:
    prev_payload = _safe_json(prev_state_path) if isinstance(prev_state_path, Path) else {}
    prev_entries = prev_payload.get("tickers") if isinstance(prev_payload.get("tickers"), dict) else {}

    universe_run_id = str(new_run_context.get("universe_run_id") or "")
    batch_run_id = str(new_run_context.get("batch_run_id") or "")
    candidates = [row for row in (new_run_context.get("candidates") or []) if isinstance(row, dict)]

    updated: dict[str, Any] = {}
    for row in sorted(candidates, key=lambda item: (int(item.get("rank_global") or 10**9), str(item.get("ticker") or ""))):
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        prev = prev_entries.get(ticker) if isinstance(prev_entries.get(ticker), dict) else {}
        prev_gate = str(prev.get("last_value_gate_status") or UNKNOWN).upper()
        new_gate = str(row.get("value_gate_status") or UNKNOWN).upper()
        prev_blocker = str(prev.get("last_primary_blocker_code") or UNKNOWN)
        new_blocker = str(row.get("primary_blocker") or UNKNOWN)

        prev_implied = prev.get("last_implied_return_base")
        new_implied = row.get("implied_return_base", UNKNOWN)
        implied_delta: float | str = UNKNOWN
        if _is_num(prev_implied) and _is_num(new_implied):
            implied_delta = round(float(new_implied) - float(prev_implied), 6)

        gate_change = f"{prev_gate}->{new_gate}" if prev_gate != UNKNOWN else f"NEW->{new_gate}"
        blocker_change = f"{prev_blocker}->{new_blocker}" if prev_blocker != UNKNOWN else f"NEW->{new_blocker}"

        history = [item for item in (prev.get("history") or []) if isinstance(item, dict)]
        snapshot = {
            "run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "rank": int(row.get("rank_global") or 0),
            "value_gate_status": new_gate,
            "implied_return_base": new_implied if _is_num(new_implied) else UNKNOWN,
            "primary_blocker": new_blocker,
            "updated_at": utc_now_iso(),
        }
        if not history or any(
            str(history[-1].get(key)) != str(snapshot.get(key))
            for key in ["rank", "value_gate_status", "implied_return_base", "primary_blocker"]
        ):
            history.append(snapshot)
        history = history[-10:]

        updated[ticker] = {
            "first_seen_run_id": str(prev.get("first_seen_run_id") or universe_run_id),
            "last_seen_run_id": universe_run_id,
            "last_value_gate_status": new_gate,
            "last_implied_return_base": new_implied if _is_num(new_implied) else UNKNOWN,
            "last_primary_blocker_code": new_blocker,
            "last_rank": int(row.get("rank_global") or 0),
            "deltas": {
                "implied_return_base_change": implied_delta,
                "gate_status_change": gate_change,
                "blocker_change": blocker_change,
            },
            "history": history,
        }

    gate_counts: dict[str, int] = {}
    for entry in updated.values():
        gate = str(entry.get("last_value_gate_status") or UNKNOWN)
        gate_counts[gate] = gate_counts.get(gate, 0) + 1

    return {
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "generated_at": utc_now_iso(),
        "ticker_count": len(updated),
        "gate_counts": dict(sorted(gate_counts.items(), key=lambda kv: (str(kv[0])))),
        "tickers": dict(sorted(updated.items(), key=lambda kv: kv[0])),
    }


def write_watchlist_state(
    universe_run_id: str,
    candidates: list[dict[str, Any]],
    memo_pack_manifest: dict[str, Any],
    out_path: Path | None,
    *,
    prev_state_path: Path | None = None,
) -> dict[str, Any]:
    target_path = out_path or (_autopilot_dir(universe_run_id) / "watchlist_state.json")
    prior = prev_state_path if isinstance(prev_state_path, Path) else (target_path if target_path.exists() else None)
    payload = update_watchlist_state(
        prior,
        {
            "universe_run_id": universe_run_id,
            "batch_run_id": str(memo_pack_manifest.get("batch_run_id") or ""),
            "candidates": candidates,
        },
    )
    # Retain a .bak of the prior (already-valid) watchlist_state before overwrite
    # so a kill mid-write still leaves the previous good state recoverable. The
    # write itself is atomic (os.replace), so the live file is never truncated.
    if target_path.exists():
        try:
            shutil.copy2(target_path, target_path.with_suffix(target_path.suffix + ".bak"))
        except OSError:
            pass
    _json_write(target_path, payload)
    return {
        "watchlist_state_path": str(target_path),
        "ticker_count": int(payload.get("ticker_count") or 0),
        "gate_counts": payload.get("gate_counts") if isinstance(payload.get("gate_counts"), dict) else {},
    }


def write_investment_memo_pack(
    universe_run_id: str,
    batch_run_id: str,
    top_n: int,
    policy: str,
    out_dir: Path | None = None,
    *,
    prev_state_path: Path | None = None,
) -> dict[str, Any]:
    write_depth_batch_rollup(
        universe_run_id=universe_run_id,
        batch_run_id=batch_run_id,
        top_n=max(1, int(top_n)),
        policy=policy,
    )
    shortlist_payload = load_global_shortlist(universe_run_id, batch_run_id)
    shortlist_rows = [row for row in (shortlist_payload["payload"].get("rows") or []) if isinstance(row, dict)]
    shortlisted = shortlist_rows[: max(1, int(top_n))]

    pack_dir = _memo_pack_dir(universe_run_id, batch_run_id, out_dir=out_dir)
    memos_dir = pack_dir / "memos"
    memos_dir.mkdir(parents=True, exist_ok=True)

    memo_index: list[dict[str, Any]] = []
    memo_rows: list[dict[str, Any]] = []
    unknown_counts = {
        "unknown_implied_return": 0,
        "unknown_price": 0,
        "unknown_valuation": 0,
        "unknown_shares": 0,
        "unknown_fcf": 0,
        "unknown_facts": 0,
    }

    for rank, row in enumerate(shortlisted, start=1):
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        source_bundle = _load_sources_for_row(row=row, ticker=ticker)
        source_bundle["shortlist_row"] = row
        memo = build_investment_memo(
            ticker,
            universe_run_id=universe_run_id,
            batch_run_id=batch_run_id,
            sources=source_bundle,
        )
        memo["rank_global"] = int(rank)
        memo["value_gate_status"] = str(
            memo.get("decision_snapshot", {}).get("value_gate_status")
            if isinstance(memo.get("decision_snapshot"), dict)
            else UNKNOWN
        )
        memo["implied_return_base"] = (
            memo.get("decision_snapshot", {}).get("implied_return_base")
            if isinstance(memo.get("decision_snapshot"), dict)
            else UNKNOWN
        )
        memo["primary_blocker"] = (
            memo.get("unknowns_and_blockers", [{}])[0].get("reason_code", UNKNOWN)
            if isinstance(memo.get("unknowns_and_blockers"), list) and memo.get("unknowns_and_blockers")
            else "NONE"
        )

        memo_json_path = memos_dir / f"{ticker}.json"
        memo_md_path = memos_dir / f"{ticker}.md"
        _json_write(memo_json_path, memo)
        memo_md_path.write_text(_memo_markdown(memo), encoding="utf-8")
        memo_index.append(
            {
                "ticker": ticker,
                "rank_global": int(rank),
                "memo_json_path": str(memo_json_path),
                "memo_md_path": str(memo_md_path),
            }
        )

        decision = memo.get("decision_snapshot") if isinstance(memo.get("decision_snapshot"), dict) else {}
        if not _is_num(decision.get("implied_return_base")):
            unknown_counts["unknown_implied_return"] += 1
        if str(decision.get("price_status") or UNKNOWN).upper() != OK:
            unknown_counts["unknown_price"] += 1
        if str(decision.get("valuation_status") or UNKNOWN).upper() != OK:
            unknown_counts["unknown_valuation"] += 1

        blockers = memo.get("unknowns_and_blockers") if isinstance(memo.get("unknowns_and_blockers"), list) else []
        blocker_codes = {str(item.get("reason_code") or "") for item in blockers if isinstance(item, dict)}
        if "MISSING_SHARES" in blocker_codes:
            unknown_counts["unknown_shares"] += 1
        if "MISSING_FCF" in blocker_codes:
            unknown_counts["unknown_fcf"] += 1
        if "NO_FACTS" in blocker_codes:
            unknown_counts["unknown_facts"] += 1

        memo_rows.append(
            {
                "ticker": ticker,
                "rank_global": int(rank),
                "value_gate_status": memo.get("value_gate_status", UNKNOWN),
                "implied_return_base": memo.get("implied_return_base", UNKNOWN),
                "primary_blocker": memo.get("primary_blocker", UNKNOWN),
            }
        )

    manifest_path = pack_dir / "memo_pack_manifest.json"
    manifest = {
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "policy": str(policy),
        "top_n": int(top_n),
        "memo_count": len(memo_index),
        "global_shortlist_path": str(shortlist_payload.get("global_shortlist_path") or ""),
        "memos": memo_index,
        "unknown_counts": unknown_counts,
        "generated_at": utc_now_iso(),
    }
    _json_write(manifest_path, manifest)

    watch_state = write_watchlist_state(
        universe_run_id=universe_run_id,
        candidates=memo_rows,
        memo_pack_manifest=manifest,
        out_path=None,
        prev_state_path=prev_state_path,
    )
    manifest["watchlist_state_path"] = watch_state["watchlist_state_path"]
    _json_write(manifest_path, manifest)

    return {
        "status": OK,
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "memo_pack_dir": str(pack_dir),
        "manifest_path": str(manifest_path),
        "watchlist_state_path": watch_state["watchlist_state_path"],
        "memo_count": len(memo_index),
        "unknown_counts": unknown_counts,
        "top_10": memo_rows[:10],
    }


def open_investment_memo_pack(universe_run_id: str, batch_run_id: str) -> dict[str, Any]:
    pack_dir = _memo_pack_dir(universe_run_id, batch_run_id)
    manifest_path = pack_dir / "memo_pack_manifest.json"
    manifest = _safe_json(manifest_path)
    if not manifest:
        return {
            "status": "MISSING",
            "universe_run_id": universe_run_id,
            "batch_run_id": batch_run_id,
            "memo_pack_dir": str(pack_dir),
            "manifest_path": str(manifest_path),
        }

    memos = [row for row in (manifest.get("memos") or []) if isinstance(row, dict)]
    preview: list[dict[str, Any]] = []
    for item in memos[:10]:
        memo_payload = _safe_json(Path(str(item.get("memo_json_path") or "")))
        decision = memo_payload.get("decision_snapshot") if isinstance(memo_payload.get("decision_snapshot"), dict) else {}
        yields = (
            memo_payload.get("owner_earnings_and_yield")
            if isinstance(memo_payload.get("owner_earnings_and_yield"), dict)
            else {}
        )
        preview.append(
            {
                "ticker": str(item.get("ticker") or ""),
                "rank_global": int(item.get("rank_global") or 0),
                "implied_return_base": decision.get("implied_return_base", UNKNOWN),
                "mos_epv": (memo_payload.get("graham_dodd") or {}).get("mos_epv", UNKNOWN)
                if isinstance(memo_payload.get("graham_dodd"), dict)
                else UNKNOWN,
                "mos_to_floor": (memo_payload.get("intrinsic_value_discipline") or {}).get("mos_to_floor", UNKNOWN)
                if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                else UNKNOWN,
                "valuation_confidence_class": (memo_payload.get("valuation_confidence_fragility") or {}).get("valuation_confidence_class", UNKNOWN)
                if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                else UNKNOWN,
                "valuation_integrity_class": (memo_payload.get("valuation_integrity_audit") or {}).get("valuation_integrity_class", UNKNOWN)
                if isinstance(memo_payload.get("valuation_integrity_audit"), dict)
                else UNKNOWN,
                "evidence_sufficiency_class": (memo_payload.get("evidence_sufficiency_for_mos") or {}).get("evidence_sufficiency_class", UNKNOWN)
                if isinstance(memo_payload.get("evidence_sufficiency_for_mos"), dict)
                else UNKNOWN,
                "mos_assessment_status": (memo_payload.get("evidence_sufficiency_for_mos") or {}).get("mos_assessment_status", UNKNOWN)
                if isinstance(memo_payload.get("evidence_sufficiency_for_mos"), dict)
                else UNKNOWN,
                "investment_readiness_class": (memo_payload.get("investment_readiness_blocker_stack") or {}).get("investment_readiness_class", UNKNOWN)
                if isinstance(memo_payload.get("investment_readiness_blocker_stack"), dict)
                else UNKNOWN,
                "blocker_stack_primary": (memo_payload.get("investment_readiness_blocker_stack") or {}).get("blocker_stack_primary", UNKNOWN)
                if isinstance(memo_payload.get("investment_readiness_blocker_stack"), dict)
                else UNKNOWN,
                "primary_next_step": (memo_payload.get("investment_readiness_blocker_stack") or {}).get("primary_next_step", UNKNOWN)
                if isinstance(memo_payload.get("investment_readiness_blocker_stack"), dict)
                else UNKNOWN,
                "valuation_fragility_status": (memo_payload.get("valuation_confidence_fragility") or {}).get("valuation_fragility_status", UNKNOWN)
                if isinstance(memo_payload.get("valuation_confidence_fragility"), dict)
                else UNKNOWN,
                "value_type_primary": (memo_payload.get("value_type_classification") or {}).get("value_type_primary", UNKNOWN)
                if isinstance(memo_payload.get("value_type_classification"), dict)
                else UNKNOWN,
                "reinvestment_efficiency_class": (
                    (memo_payload.get("incremental_reinvestment_efficiency") or {}).get(
                        "reinvestment_efficiency_class",
                        "REINVESTMENT_EFFICIENCY_UNKNOWN",
                    )
                    if isinstance(memo_payload.get("incremental_reinvestment_efficiency"), dict)
                    else "REINVESTMENT_EFFICIENCY_UNKNOWN"
                ),
                "primary_reinvestment_caution": (
                    (memo_payload.get("incremental_reinvestment_efficiency") or {}).get(
                        "primary_reinvestment_caution",
                        "REINVESTMENT_UNCLEAR",
                    )
                    if isinstance(memo_payload.get("incremental_reinvestment_efficiency"), dict)
                    else "REINVESTMENT_UNCLEAR"
                ),
                "downside_support_type": (memo_payload.get("intrinsic_value_discipline") or {}).get("downside_support_type", UNKNOWN)
                if isinstance(memo_payload.get("intrinsic_value_discipline"), dict)
                else UNKNOWN,
                "yield_metric_used": yields.get("yield_metric_used", UNKNOWN),
            }
        )

    return {
        "status": OK,
        "universe_run_id": universe_run_id,
        "batch_run_id": batch_run_id,
        "memo_pack_dir": str(pack_dir),
        "manifest_path": str(manifest_path),
        "watchlist_state_path": str(manifest.get("watchlist_state_path") or ""),
        "memo_count": int(manifest.get("memo_count") or len(memos)),
        "unknown_counts": manifest.get("unknown_counts") if isinstance(manifest.get("unknown_counts"), dict) else {},
        "top_10": preview,
    }


def _watchlist_state_path(run_id: str) -> Path:
    return _autopilot_dir(run_id) / "watchlist_state.json"


def open_watchlist_state(run_id: str) -> dict[str, Any]:
    path = _watchlist_state_path(run_id)
    payload = _safe_json(path)
    if not payload:
        return {
            "status": "MISSING",
            "run_id": run_id,
            "watchlist_state_path": str(path),
        }
    tickers = payload.get("tickers") if isinstance(payload.get("tickers"), dict) else {}
    top = sorted(
        [
            {
                "ticker": ticker,
                "last_rank": int((entry or {}).get("last_rank") or 0),
                "last_value_gate_status": str((entry or {}).get("last_value_gate_status") or UNKNOWN),
                "last_implied_return_base": (entry or {}).get("last_implied_return_base", UNKNOWN),
                "last_primary_blocker_code": str((entry or {}).get("last_primary_blocker_code") or UNKNOWN),
            }
            for ticker, entry in tickers.items()
            if isinstance(entry, dict)
        ],
        key=lambda item: (int(item.get("last_rank") or 10**9), str(item.get("ticker") or "")),
    )[:10]
    return {
        "status": OK,
        "run_id": run_id,
        "watchlist_state_path": str(path),
        "ticker_count": int(payload.get("ticker_count") or len(tickers)),
        "gate_counts": payload.get("gate_counts") if isinstance(payload.get("gate_counts"), dict) else {},
        "top_10": top,
    }


def diff_watchlist_states(prev_run_id: str, run_id: str) -> dict[str, Any]:
    prev_payload = _safe_json(_watchlist_state_path(prev_run_id))
    cur_payload = _safe_json(_watchlist_state_path(run_id))
    if not prev_payload:
        return {
            "status": "MISSING_PREV",
            "prev_run_id": prev_run_id,
            "run_id": run_id,
            "prev_watchlist_state_path": str(_watchlist_state_path(prev_run_id)),
            "watchlist_state_path": str(_watchlist_state_path(run_id)),
        }
    if not cur_payload:
        return {
            "status": "MISSING",
            "prev_run_id": prev_run_id,
            "run_id": run_id,
            "prev_watchlist_state_path": str(_watchlist_state_path(prev_run_id)),
            "watchlist_state_path": str(_watchlist_state_path(run_id)),
        }

    prev_map = prev_payload.get("tickers") if isinstance(prev_payload.get("tickers"), dict) else {}
    cur_map = cur_payload.get("tickers") if isinstance(cur_payload.get("tickers"), dict) else {}
    prev_set = set(prev_map.keys())
    cur_set = set(cur_map.keys())

    additions = sorted(cur_set - prev_set)
    removals = sorted(prev_set - cur_set)
    upgrades: list[dict[str, Any]] = []
    downgrades: list[dict[str, Any]] = []
    blocker_changes: list[dict[str, Any]] = []
    implied_changes: list[dict[str, Any]] = []

    for ticker in sorted(prev_set & cur_set):
        prev_row = prev_map.get(ticker) if isinstance(prev_map.get(ticker), dict) else {}
        cur_row = cur_map.get(ticker) if isinstance(cur_map.get(ticker), dict) else {}

        prev_gate = str(prev_row.get("last_value_gate_status") or UNKNOWN).upper()
        cur_gate = str(cur_row.get("last_value_gate_status") or UNKNOWN).upper()
        prev_rank = _STATUS_RANK.get(prev_gate, -1)
        cur_rank = _STATUS_RANK.get(cur_gate, -1)
        if cur_rank > prev_rank:
            upgrades.append({"ticker": ticker, "from": prev_gate, "to": cur_gate})
        elif cur_rank < prev_rank:
            downgrades.append({"ticker": ticker, "from": prev_gate, "to": cur_gate})

        prev_blocker = str(prev_row.get("last_primary_blocker_code") or UNKNOWN)
        cur_blocker = str(cur_row.get("last_primary_blocker_code") or UNKNOWN)
        if prev_blocker != cur_blocker:
            blocker_changes.append({"ticker": ticker, "from": prev_blocker, "to": cur_blocker})

        prev_implied = prev_row.get("last_implied_return_base")
        cur_implied = cur_row.get("last_implied_return_base")
        if _is_num(prev_implied) and _is_num(cur_implied):
            implied_changes.append(
                {
                    "ticker": ticker,
                    "from": prev_implied,
                    "to": cur_implied,
                    "delta": round(float(cur_implied) - float(prev_implied), 6),
                }
            )

    implied_changes.sort(key=lambda item: (-float(item.get("delta") or 0.0), str(item.get("ticker") or "")))
    return {
        "status": OK,
        "prev_run_id": prev_run_id,
        "run_id": run_id,
        "prev_watchlist_state_path": str(_watchlist_state_path(prev_run_id)),
        "watchlist_state_path": str(_watchlist_state_path(run_id)),
        "summary": {
            "addition_count": len(additions),
            "removal_count": len(removals),
            "upgrade_count": len(upgrades),
            "downgrade_count": len(downgrades),
            "blocker_change_count": len(blocker_changes),
        },
        "additions": additions,
        "removals": removals,
        "upgrades": upgrades,
        "downgrades": downgrades,
        "blocker_changes": blocker_changes,
        "implied_return_changes_top": implied_changes[:10],
    }
