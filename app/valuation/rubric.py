from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app.valuation.engine import build_ticker_valuation
from app.valuation.fundamentals import UNKNOWN, build_fundamentals_frame


RUBRIC_VERSION = "v2.0"
DEFAULT_RUBRIC_WEIGHTS = {
    "quality": 0.25,
    "growth": 0.25,
    "capital_discipline": 0.25,
    "valuation": 0.25,
}

LOWER_BETTER_METRICS = {"risk_penalty"}
ALLOWED_VALUATION_REASON_CODES = {
    "PRICE_UNKNOWN",
    "MISSING_FCF",
    "MISSING_SHARES",
    "MISSING_NET_DEBT",
    "INVALID_DENOMINATOR",
    "MODEL_PRECONDITION_FAILED",
    "ENGINE_EXCEPTION",
}


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _safe_div(a: Any, b: Any) -> float | str:
    if not _is_num(a) or not _is_num(b) or float(b) == 0.0:
        return UNKNOWN
    return float(a) / float(b)


def _metric_trace_from_fundamentals(
    fundamentals: dict[str, Any], metric: str, *, year: int | None = None
) -> list[str]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    if not rows:
        return [f"fundamentals.rows[*].{metric}"]
    target_year = year if isinstance(year, int) else int(rows[-1].get("year", 0))
    trace_map = fundamentals.get("row_traces") or {}
    trace_bucket = (trace_map.get(str(target_year), {}) if isinstance(trace_map, dict) else {}).get(
        metric, {}
    )
    refs = [str(x) for x in (trace_bucket.get("derived_from") or []) if str(x).strip()]
    if refs:
        return refs
    return [f"fundamentals.rows[{target_year}].{metric}"]


def _signal_lookup(fundamentals: dict[str, Any], signal: str) -> tuple[float | str, list[str]]:
    derived = fundamentals.get("derived_signals") or {}
    row = derived.get(signal)
    if not isinstance(row, dict):
        return UNKNOWN, [f"fundamentals.derived_signals.{signal}"]
    value = row.get("value")
    refs = [str(x) for x in (row.get("derived_from") or []) if str(x).strip()]
    return (float(value) if _is_num(value) else UNKNOWN), (
        refs or [f"fundamentals.derived_signals.{signal}"]
    )


def _cagr_from_rows(rows: list[dict[str, Any]], metric: str, years: int) -> float | str:
    series = [
        (int(row.get("year", 0)), float(row[metric]))
        for row in rows
        if _is_num(row.get(metric)) and float(row.get(metric)) > 0
    ]
    if len(series) < 2:
        return UNKNOWN
    end_year, end_value = series[-1]
    target_year = end_year - int(years)
    candidates = [(year, value) for year, value in series if year <= target_year and value > 0]
    if candidates:
        start_year, start_value = candidates[-1]
    else:
        start_year, start_value = series[0]
    if start_value <= 0:
        return UNKNOWN
    span = max(1, end_year - start_year)
    return (end_value / start_value) ** (1.0 / span) - 1.0


def sanitize_rubric_weights(weights: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(weights, dict) or not weights:
        return dict(DEFAULT_RUBRIC_WEIGHTS)
    out: dict[str, float] = {}
    for key in ("quality", "growth", "capital_discipline", "valuation"):
        value = weights.get(key, DEFAULT_RUBRIC_WEIGHTS[key])
        if not _is_num(value):
            out[key] = float(DEFAULT_RUBRIC_WEIGHTS[key])
            continue
        out[key] = _clamp(float(value), 0.05, 0.70)
    total = sum(out.values())
    if total <= 0:
        return dict(DEFAULT_RUBRIC_WEIGHTS)
    normalized = {key: round(float(value) / float(total), 6) for key, value in out.items()}
    return normalized


def compute_value_first_score(
    *,
    fundamentals: dict[str, Any],
    valuation: dict[str, Any],
    whale_payload: dict[str, Any] | None = None,
    rubric_weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [row for row in (fundamentals.get("rows") or []) if isinstance(row, dict)]
    rows = sorted(rows, key=lambda row: int(row.get("year", 0)))
    latest = rows[-1] if rows else {}
    weights = sanitize_rubric_weights(rubric_weights)

    quality_reasons: list[dict[str, Any]] = []
    growth_reasons: list[dict[str, Any]] = []
    capital_reasons: list[dict[str, Any]] = []
    valuation_reasons: list[dict[str, Any]] = []
    risk_reasons: list[dict[str, Any]] = []
    gaps: list[str] = []

    # Quality 0-25.
    q = 0.0
    op_margin = latest.get("op_margin", UNKNOWN)
    fcf_margin = latest.get("fcf_margin", UNKNOWN)
    gm_margin = latest.get("gross_margin", UNKNOWN)
    net_debt = latest.get("net_debt", UNKNOWN)
    revenue = latest.get("revenue", UNKNOWN)
    if _is_num(op_margin):
        q += 8.0 if float(op_margin) >= 0.15 else (4.0 if float(op_margin) > 0.0 else 1.0)
    else:
        gaps.append("QUALITY_MISSING_OP_MARGIN")
    if _is_num(fcf_margin):
        q += 8.0 if float(fcf_margin) >= 0.10 else (4.0 if float(fcf_margin) > 0.0 else 1.0)
    else:
        gaps.append("QUALITY_MISSING_FCF_MARGIN")
    if _is_num(gm_margin):
        q += 4.0 if float(gm_margin) >= 0.40 else 2.0
    else:
        gaps.append("QUALITY_MISSING_GROSS_MARGIN")
    if _is_num(net_debt):
        if float(net_debt) <= 0:
            q += 5.0
        elif _is_num(revenue) and math.isfinite(float(revenue)) and float(revenue) > 0.0:
            q += 3.0 if float(net_debt) < 0.5 * float(revenue) else 1.0
        elif _is_num(revenue):
            gaps.append("QUALITY_INVALID_REVENUE_FOR_LEVERAGE")
        else:
            gaps.append("QUALITY_MISSING_REVENUE_FOR_LEVERAGE")
    else:
        gaps.append("QUALITY_MISSING_NET_DEBT")
    quality_score = _clamp(q, 0.0, 25.0)
    quality_reasons.append(
        {
            "reason": "Profitability and balance-sheet profile.",
            "value": round(quality_score, 6),
            "derived_from": (
                _metric_trace_from_fundamentals(fundamentals, "op_margin")
                + _metric_trace_from_fundamentals(fundamentals, "fcf_margin")
                + _metric_trace_from_fundamentals(fundamentals, "gross_margin")
                + _metric_trace_from_fundamentals(fundamentals, "net_debt")
                + _metric_trace_from_fundamentals(fundamentals, "revenue")
            ),
        }
    )

    # Growth 0-25.
    revenue_5y, rev_5y_refs = _signal_lookup(fundamentals, "revenue_cagr_5y")
    revenue_10y, rev_10y_refs = _signal_lookup(fundamentals, "revenue_cagr_10y")
    op_slope, op_slope_refs = _signal_lookup(fundamentals, "operating_margin_trend_slope")
    g = 0.0
    if _is_num(revenue_5y):
        g += 10.0 if float(revenue_5y) >= 0.12 else (6.0 if float(revenue_5y) >= 0.06 else 2.0)
    else:
        gaps.append("GROWTH_MISSING_REVENUE_CAGR_5Y")
    if _is_num(revenue_10y):
        g += 10.0 if float(revenue_10y) >= 0.08 else (6.0 if float(revenue_10y) >= 0.04 else 2.0)
    else:
        gaps.append("GROWTH_MISSING_REVENUE_CAGR_10Y")
    if _is_num(op_slope):
        g += 5.0 if float(op_slope) > 0 else 1.0
    else:
        gaps.append("GROWTH_MISSING_OP_MARGIN_SLOPE")
    growth_score = _clamp(g, 0.0, 25.0)
    growth_reasons.append(
        {
            "reason": "Revenue persistence and scaling quality.",
            "value": round(growth_score, 6),
            "derived_from": rev_5y_refs + rev_10y_refs + op_slope_refs,
        }
    )

    # Capital Discipline 0-25.
    c = 0.0
    dilution, dilution_refs = _signal_lookup(fundamentals, "dilution_rate_shares_cagr")
    cfo_margin = latest.get("cfo_margin", UNKNOWN)
    cfo_5y = _cagr_from_rows(rows, "cfo", 5)
    if _is_num(dilution):
        c += 9.0 if float(dilution) <= 0.02 else (5.0 if float(dilution) <= 0.05 else 1.0)
    else:
        gaps.append("CAPITAL_MISSING_DILUTION")
    if _is_num(cfo_margin):
        c += 8.0 if float(cfo_margin) >= 0.15 else (4.0 if float(cfo_margin) > 0.05 else 1.0)
    else:
        gaps.append("CAPITAL_MISSING_CFO_MARGIN")
    if _is_num(cfo_5y):
        c += 8.0 if float(cfo_5y) >= 0.05 else (4.0 if float(cfo_5y) >= 0 else 1.0)
    else:
        gaps.append("CAPITAL_MISSING_CFO_CAGR")
    capital_score = _clamp(c, 0.0, 25.0)
    capital_reasons.append(
        {
            "reason": "Cash conversion, dilution, and reinvestment capacity.",
            "value": round(capital_score, 6),
            "derived_from": dilution_refs
            + _metric_trace_from_fundamentals(fundamentals, "cfo_margin")
            + ["fundamentals.rows[*].cfo"],
        }
    )

    # Valuation 0-25.
    v = 0.0
    implied_return = valuation.get("implied_return_base", UNKNOWN)
    implied_growth = valuation.get("implied_fcf_growth", UNKNOWN)
    if _is_num(implied_return):
        ir = float(implied_return)
        v += 20.0 if ir >= 0.30 else (16.0 if ir >= 0.15 else (10.0 if ir >= 0.0 else 4.0))
    else:
        gaps.append("VALUATION_MISSING_IMPLIED_RETURN")
        v += 6.0
    if _is_num(implied_growth):
        ig = float(implied_growth)
        growth_threshold = max(float(revenue_10y), 0.05) if _is_num(revenue_10y) else 0.12
        if ig <= growth_threshold:
            v += 5.0
        elif ig <= (growth_threshold + 0.05):
            v += 3.0
        else:
            v += 1.0
    else:
        gaps.append("VALUATION_MISSING_IMPLIED_GROWTH")
        v += 2.0
    valuation_score = _clamp(v, 0.0, 25.0)
    valuation_reasons.append(
        {
            "reason": "Discount to intrinsic and reasonableness of implied growth.",
            "value": round(valuation_score, 6),
            "derived_from": [
                "valuation.claims.implied_return_base.derived_from",
                "valuation.claims.implied_fcf_growth.derived_from",
            ],
        }
    )

    # Risk penalty overlay.
    risk_penalty = 0.0
    if rows:
        first = rows[0]
        if _is_num(first.get("net_debt")) and _is_num(net_debt):
            start_debt = float(first.get("net_debt"))
            end_debt = float(net_debt)
            if end_debt > (start_debt + abs(start_debt) * 0.25):
                risk_penalty += 5.0
                risk_reasons.append(
                    {
                        "reason": "Net debt deteriorated materially.",
                        "value": 5.0,
                        "derived_from": [
                            f"fundamentals.rows[{int(first.get('year', 0))}].net_debt",
                            f"fundamentals.rows[{int(latest.get('year', 0))}].net_debt",
                        ],
                    }
                )
    if _is_num(dilution) and float(dilution) > 0.08:
        risk_penalty += 4.0
        risk_reasons.append(
            {
                "reason": "Heavy dilution trend.",
                "value": 4.0,
                "derived_from": dilution_refs,
            }
        )
    negative_fcf_streak = 0
    for row in reversed(rows):
        fcf = row.get("fcf", UNKNOWN)
        if _is_num(fcf) and float(fcf) < 0:
            negative_fcf_streak += 1
        else:
            break
    if negative_fcf_streak >= 3:
        risk_penalty += 6.0
        risk_reasons.append(
            {
                "reason": "Three-year negative FCF streak.",
                "value": 6.0,
                "derived_from": ["fundamentals.rows[*].fcf"],
            }
        )
    unknown_penalty = min(15.0, float(len(sorted(set(gaps)))) * 1.5)
    if unknown_penalty > 0:
        risk_penalty += unknown_penalty
        risk_reasons.append(
            {
                "reason": "Unknown inputs reduce confidence.",
                "value": round(unknown_penalty, 6),
                "derived_from": ["valuation.gaps", "fundamentals.gaps"],
            }
        )
    risk_penalty = _clamp(risk_penalty, 0.0, 25.0)

    whale_score = (whale_payload or {}).get("whale_signature_score", UNKNOWN)
    whale_component = _clamp(float(whale_score) * 0.15, 0.0, 15.0) if _is_num(whale_score) else 0.0

    weighted_core = (
        (quality_score * float(weights["quality"]))
        + (growth_score * float(weights["growth"]))
        + (capital_score * float(weights["capital_discipline"]))
        + (valuation_score * float(weights["valuation"]))
    ) * 4.0
    total_score = _clamp((weighted_core * 0.85) + whale_component - risk_penalty, 0.0, 100.0)

    return {
        "rubric_version": RUBRIC_VERSION,
        "weights": weights,
        "quality_score": round(quality_score, 6),
        "growth_score": round(growth_score, 6),
        "capital_discipline_score": round(capital_score, 6),
        "valuation_score": round(valuation_score, 6),
        "risk_penalty": round(risk_penalty, 6),
        "whale_component": round(whale_component, 6),
        "score_total": round(total_score, 6),
        "gaps": sorted(set(gaps)),
        "reasons": {
            "quality": quality_reasons,
            "growth": growth_reasons,
            "capital_discipline": capital_reasons,
            "valuation": valuation_reasons,
            "risk": risk_reasons,
        },
        "derived_from": [
            "fundamentals.rows[*]",
            "fundamentals.derived_signals",
            "valuation.claims",
            "whale_signals.whale_signature_score",
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_whale_payload(run_id: str, ticker: str, *, dossier_dir: Path) -> dict[str, Any]:
    path = dossier_dir / f"whale_signals_{ticker}.json"
    payload = _read_json(path)
    return payload if payload else {}


def _load_dossier_payload(run_id: str, ticker: str, *, dossier_dir: Path) -> dict[str, Any]:
    path = dossier_dir / ticker / "dossier.json"
    payload = _read_json(path)
    return payload if payload else {}


def _valuation_coverage_entry(
    *,
    ticker: str,
    fundamentals: dict[str, Any],
    valuation: dict[str, Any],
    price_coverage_entry: dict[str, Any] | None,
    shares_coverage_entry: dict[str, Any] | None,
    fcf_coverage_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    price_cov = price_coverage_entry if isinstance(price_coverage_entry, dict) else {}
    shares_cov = shares_coverage_entry if isinstance(shares_coverage_entry, dict) else {}
    fcf_cov = fcf_coverage_entry if isinstance(fcf_coverage_entry, dict) else {}
    price_result = price_cov.get("result") if isinstance(price_cov.get("result"), dict) else {}
    price_output = (
        price_cov.get("output_fields") if isinstance(price_cov.get("output_fields"), dict) else {}
    )
    input_snapshot = (
        valuation.get("input_snapshot") if isinstance(valuation.get("input_snapshot"), dict) else {}
    )
    has_numeric_price = _is_num(input_snapshot.get("current_price"))
    price_status = str(price_result.get("status") or "").upper()
    if price_status not in {"OK", "UNKNOWN"}:
        price_status = "OK" if has_numeric_price else "UNKNOWN"
    if price_status == "UNKNOWN" and has_numeric_price:
        price_status = "OK"
    price_reason_code = str(
        price_result.get("reason_code")
        or (
            "CACHE_HIT"
            if _is_num((valuation.get("input_snapshot") or {}).get("current_price"))
            else "PRICE_UNKNOWN"
        )
    )

    valuation_inputs = valuation.get("valuation_inputs")
    if not isinstance(valuation_inputs, dict):
        valuation_inputs = {}

    valuation_status = str(valuation.get("valuation_status") or "UNKNOWN").upper()
    if valuation_status not in {"OK", "UNKNOWN"}:
        valuation_status = "UNKNOWN"
    reason_code = str(valuation.get("valuation_reason_code") or "").strip().upper() or None
    if valuation_status != "OK" and reason_code not in ALLOWED_VALUATION_REASON_CODES:
        reason_code = "MODEL_PRECONDITION_FAILED"
    if valuation_status == "OK":
        reason_code = None

    price_source_resolution = str(input_snapshot.get("price_source_resolution") or "").strip() or (
        "run_scoped_output"
        if str(input_snapshot.get("current_price_source") or "") == "run_scoped_output"
        else (
            "disk_cache"
            if str(input_snapshot.get("current_price_source") or "") == "disk_cache"
            else "provider_live_fetch"
        )
    )
    price_asof_used = input_snapshot.get("price_asof_used") or (
        price_output.get("price_asof_used") if isinstance(price_output, dict) else None
    )
    shares_status = str(shares_cov.get("shares_status") or "").upper()
    if shares_status not in {"OK", "UNKNOWN"}:
        shares_status = (
            "OK"
            if _is_num(input_snapshot.get("shares_outstanding"))
            and float(input_snapshot.get("shares_outstanding")) > 0
            else "UNKNOWN"
        )
    shares_reason_code = str(
        shares_cov.get("shares_reason_code")
        or ("OK" if shares_status == "OK" else "MISSING_SHARES")
    )
    shares_source_resolution = (
        str(
            input_snapshot.get("shares_source_resolution")
            or shares_cov.get("shares_source_resolution")
            or ""
        ).strip()
        or "unknown"
    )
    fcf_status = str(fcf_cov.get("fcf_status") or "").upper()
    if fcf_status not in {"OK", "UNKNOWN"}:
        fcf_status = "OK" if _is_num(input_snapshot.get("fcf_latest")) else "UNKNOWN"
    fcf_reason_code = str(
        fcf_cov.get("fcf_reason_code") or ("OK" if fcf_status == "OK" else "MISSING_FCF")
    )
    fcf_source_resolution = (
        str(
            input_snapshot.get("fcf_source_resolution")
            or fcf_cov.get("fcf_source_resolution")
            or ""
        ).strip()
        or "unknown"
    )

    claim_bucket = valuation.get("claims") if isinstance(valuation.get("claims"), dict) else {}
    intrinsic_claim = (
        claim_bucket.get("intrinsic_per_share_base") if isinstance(claim_bucket, dict) else {}
    )
    implied_claim = (
        claim_bucket.get("implied_return_base") if isinstance(claim_bucket, dict) else {}
    )
    intrinsic_refs = (
        [str(x) for x in (intrinsic_claim.get("derived_from") or []) if str(x).strip()]
        if isinstance(intrinsic_claim, dict)
        else []
    )
    implied_refs = (
        [str(x) for x in (implied_claim.get("derived_from") or []) if str(x).strip()]
        if isinstance(implied_claim, dict)
        else []
    )

    return {
        "ticker": ticker,
        "price_status": price_status,
        "price_reason_code": price_reason_code,
        "price_asof_used": price_asof_used,
        "price_source_resolution": price_source_resolution,
        "shares_status": shares_status,
        "shares_reason_code": shares_reason_code,
        "shares_source_resolution": shares_source_resolution,
        "fcf_status": fcf_status,
        "fcf_reason_code": fcf_reason_code,
        "fcf_source_resolution": fcf_source_resolution,
        "valuation_inputs": valuation_inputs,
        "valuation_status": valuation_status,
        "valuation_reason_code": reason_code,
        "intrinsic_per_share_base": valuation.get("intrinsic_per_share_base", UNKNOWN),
        "implied_return_base": valuation.get("implied_return_base", UNKNOWN),
        "derived_from": (
            intrinsic_refs
            + implied_refs
            + [
                f"valuation_{ticker}.json.valuation_reason_code",
                f"fundamentals_{ticker}.json.rows[-1].fcf",
                f"fundamentals_{ticker}.json.rows[-1].shares_outstanding",
                f"fundamentals_{ticker}.json.rows[-1].net_debt",
                f"sector.price_coverage.entries[{ticker}]",
                f"sector.shares_coverage.entries[{ticker}]",
                f"sector.fcf_coverage.entries[{ticker}]",
            ]
        ),
    }


def apply_value_first_overlay(
    *,
    run_id: str,
    as_of_date: str,
    sector_run_dir: Path,
    dossier_run_dir: Path,
    rubric_weights: dict[str, Any] | None = None,
    with_prices: bool = True,
) -> dict[str, Any]:
    scoreboard_path = sector_run_dir / "peer_scoreboard.json"
    rankings_path = sector_run_dir / "peer_rankings.json"
    scoreboard = _read_json(scoreboard_path)
    rankings = _read_json(rankings_path)
    rows = [row for row in (scoreboard.get("rows") or []) if isinstance(row, dict)]
    ranking_rows = [row for row in (rankings.get("rankings") or []) if isinstance(row, dict)]
    ranking_by_ticker = {str(row.get("ticker") or "").upper(): row for row in ranking_rows}
    price_coverage_path = sector_run_dir / "price_coverage.json"
    price_coverage_payload = _read_json(price_coverage_path)
    price_coverage_rows = [
        row for row in (price_coverage_payload.get("entries") or []) if isinstance(row, dict)
    ]
    price_coverage_by_ticker = {
        str(row.get("ticker") or "").upper(): row for row in price_coverage_rows
    }
    shares_coverage_path = sector_run_dir / "shares_coverage.json"
    shares_coverage_payload = _read_json(shares_coverage_path)
    shares_coverage_rows = [
        row for row in (shares_coverage_payload.get("entries") or []) if isinstance(row, dict)
    ]
    shares_coverage_by_ticker = {
        str(row.get("ticker") or "").upper(): row for row in shares_coverage_rows
    }
    fcf_coverage_path = sector_run_dir / "fcf_coverage.json"
    fcf_coverage_payload = _read_json(fcf_coverage_path)
    fcf_coverage_rows = [
        row for row in (fcf_coverage_payload.get("entries") or []) if isinstance(row, dict)
    ]
    fcf_coverage_by_ticker = {
        str(row.get("ticker") or "").upper(): row for row in fcf_coverage_rows
    }
    facts_coverage_path = sector_run_dir / "facts_coverage.json"

    updated_rows: list[dict[str, Any]] = []
    valuation_coverage_entries: list[dict[str, Any]] = []
    metrics_added = [
        "intrinsic_per_share_base",
        "intrinsic_per_share_conservative",
        "valuation_gap",
        "implied_return_base",
        "implied_return_conservative",
        "implied_fcf_growth",
        "quality_score",
        "growth_score",
        "capital_discipline_score",
        "valuation_score",
        "risk_penalty",
        "score_total",
    ]
    coverage_metrics = [
        "price_status",
        "shares_status",
        "shares_reason_code",
        "fcf_status",
        "fcf_reason_code",
        "valuation_status",
        "valuation_reason_code",
    ]
    trace_missing_count = 0

    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        fundamentals_path = sector_run_dir / f"fundamentals_{ticker}.json"
        valuation_path = sector_run_dir / f"valuation_{ticker}.json"

        fundamentals = _read_json(fundamentals_path)
        if not fundamentals:
            dossier = _load_dossier_payload(run_id, ticker, dossier_dir=dossier_run_dir)
            if dossier:
                fundamentals = build_fundamentals_frame(dossier)
                fundamentals_path.write_text(json.dumps(fundamentals, indent=2), encoding="utf-8")
        valuation = _read_json(valuation_path)
        if not valuation and fundamentals:
            valuation = build_ticker_valuation(
                fundamentals,
                run_id=run_id,
                as_of_date=as_of_date,
                with_prices=with_prices,
            )
            valuation_path.write_text(json.dumps(valuation, indent=2), encoding="utf-8")
        whale_payload = _load_whale_payload(run_id, ticker, dossier_dir=dossier_run_dir)
        score_payload = compute_value_first_score(
            fundamentals=fundamentals or {},
            valuation=valuation or {},
            whale_payload=whale_payload,
            rubric_weights=rubric_weights,
        )

        metric_values = dict(row.get("metric_values") or {})
        metric_traces = dict(row.get("metric_traces") or {})

        claim_map = valuation.get("claims") if isinstance(valuation, dict) else {}
        for field in metrics_added:
            if field in {
                "quality_score",
                "growth_score",
                "capital_discipline_score",
                "valuation_score",
                "risk_penalty",
                "score_total",
            }:
                value = score_payload.get(field, UNKNOWN)
                refs = [f"value_first.{field}"] + list(score_payload.get("derived_from") or [])
            else:
                value = (valuation or {}).get(field, UNKNOWN)
                claim = (claim_map or {}).get(field) if isinstance(claim_map, dict) else None
                refs = (
                    [str(x) for x in ((claim or {}).get("derived_from") or []) if str(x).strip()]
                    if isinstance(claim, dict)
                    else []
                )
                if not refs:
                    refs = [f"valuation.{field}"]
            metric_values[field] = float(value) if _is_num(value) else UNKNOWN
            metric_traces[field] = {"derived_from": refs}
            if _is_num(metric_values[field]) and not refs:
                trace_missing_count += 1

        ticker_price_cov = price_coverage_by_ticker.get(ticker)
        ticker_shares_cov = shares_coverage_by_ticker.get(ticker)
        ticker_fcf_cov = fcf_coverage_by_ticker.get(ticker)
        if not isinstance(ticker_shares_cov, dict):
            ticker_shares_cov = (
                valuation.get("shares_coverage_entry") if isinstance(valuation, dict) else None
            )
        if not isinstance(ticker_fcf_cov, dict):
            ticker_fcf_cov = (
                valuation.get("fcf_coverage_entry") if isinstance(valuation, dict) else None
            )
        coverage_entry = _valuation_coverage_entry(
            ticker=ticker,
            fundamentals=fundamentals or {},
            valuation=valuation or {},
            price_coverage_entry=ticker_price_cov,
            shares_coverage_entry=ticker_shares_cov,
            fcf_coverage_entry=ticker_fcf_cov,
        )
        valuation_coverage_entries.append(coverage_entry)
        metric_values["price_status"] = coverage_entry["price_status"]
        metric_values["shares_status"] = coverage_entry["shares_status"]
        metric_values["shares_reason_code"] = coverage_entry["shares_reason_code"]
        metric_values["fcf_status"] = coverage_entry["fcf_status"]
        metric_values["fcf_reason_code"] = coverage_entry["fcf_reason_code"]
        metric_values["valuation_status"] = coverage_entry["valuation_status"]
        metric_values["valuation_reason_code"] = (
            coverage_entry["valuation_reason_code"]
            if coverage_entry["valuation_reason_code"]
            else UNKNOWN
        )
        metric_traces["price_status"] = {
            "derived_from": [f"sector.price_coverage.entries[{ticker}]"]
        }
        metric_traces["shares_status"] = {
            "derived_from": [f"sector.shares_coverage.entries[{ticker}]"]
        }
        metric_traces["shares_reason_code"] = {
            "derived_from": [f"sector.shares_coverage.entries[{ticker}].shares_reason_code"]
        }
        metric_traces["fcf_status"] = {"derived_from": [f"sector.fcf_coverage.entries[{ticker}]"]}
        metric_traces["fcf_reason_code"] = {
            "derived_from": [f"sector.fcf_coverage.entries[{ticker}].fcf_reason_code"]
        }
        metric_traces["valuation_status"] = {
            "derived_from": [f"valuation_coverage.entries[{ticker}].valuation_status"]
        }
        metric_traces["valuation_reason_code"] = {
            "derived_from": [f"valuation_coverage.entries[{ticker}].valuation_reason_code"]
        }

        row["metric_values"] = metric_values
        row["metric_traces"] = metric_traces
        row["value_first"] = {
            "quality_score": score_payload["quality_score"],
            "growth_score": score_payload["growth_score"],
            "capital_discipline_score": score_payload["capital_discipline_score"],
            "valuation_score": score_payload["valuation_score"],
            "risk_penalty": score_payload["risk_penalty"],
            "score_total": score_payload["score_total"],
            "weights": score_payload["weights"],
            "gaps": score_payload["gaps"],
            "reasons": score_payload["reasons"],
            "derived_from": score_payload["derived_from"],
        }
        updated_rows.append(row)

        ranking_row = ranking_by_ticker.get(ticker)
        if isinstance(ranking_row, dict):
            ranking_row["value_first"] = {
                "quality_score": score_payload["quality_score"],
                "growth_score": score_payload["growth_score"],
                "capital_discipline_score": score_payload["capital_discipline_score"],
                "valuation_score": score_payload["valuation_score"],
                "risk_penalty": score_payload["risk_penalty"],
                "score_total": score_payload["score_total"],
                "weights": score_payload["weights"],
            }
            ranking_row["overall_score_v2"] = score_payload["score_total"]
            ranking_row.setdefault("metric_ranks", {})

    updated_rows.sort(
        key=lambda row: (
            -float((row.get("metric_values") or {}).get("score_total", -1.0))
            if _is_num((row.get("metric_values") or {}).get("score_total"))
            else 1e9,
            str(row.get("ticker") or ""),
        ),
    )
    value_rank = [str(row.get("ticker") or "").upper() for row in updated_rows]
    for idx, row in enumerate(updated_rows, start=1):
        row["value_first_rank"] = int(idx)
        ticker = str(row.get("ticker") or "").upper()
        ranking_row = ranking_by_ticker.get(ticker)
        if isinstance(ranking_row, dict):
            ranking_row.setdefault("metric_ranks", {})
            ranking_row["metric_ranks"]["value_first_rank"] = int(idx)
    ranking_rows.sort(
        key=lambda row: (
            int((row.get("metric_ranks") or {}).get("value_first_rank") or 10_000),
            str(row.get("ticker") or ""),
        )
    )

    scoreboard["rows"] = updated_rows
    metrics = sorted(
        set(
            [str(metric) for metric in (scoreboard.get("metrics") or [])]
            + metrics_added
            + coverage_metrics
        )
    )
    scoreboard["metrics"] = metrics
    scoreboard["ranking_mode"] = "value_first_v1.1"
    scoreboard["derived_from"] = scoreboard.get("derived_from") or {}
    if isinstance(scoreboard["derived_from"], dict):
        scoreboard["derived_from"]["value_first"] = [
            "fundamentals_*.json",
            "valuation_*.json",
            "whale_signals_*.json",
        ]
    scoreboard_path.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")

    rankings["rankings"] = ranking_rows
    rankings["value_first_rank"] = value_rank
    rankings["ranking_mode"] = "value_first_v1.1"
    rankings["rubric_weights"] = sanitize_rubric_weights(rubric_weights)
    rankings_path.write_text(json.dumps(rankings, indent=2), encoding="utf-8")

    valuation_coverage_entries = sorted(
        [row for row in valuation_coverage_entries if isinstance(row, dict)],
        key=lambda row: str(row.get("ticker") or ""),
    )
    valuation_reason_counts: dict[str, int] = {}
    for row in valuation_coverage_entries:
        code = str(row.get("valuation_reason_code") or "OK")
        valuation_reason_counts[code] = valuation_reason_counts.get(code, 0) + 1
    valuation_coverage_path = sector_run_dir / "valuation_coverage.json"
    valuation_coverage_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(valuation_coverage_entries),
        "reason_counts": dict(sorted(valuation_reason_counts.items(), key=lambda kv: kv[0])),
        "entries": valuation_coverage_entries,
    }
    valuation_coverage_path.write_text(
        json.dumps(valuation_coverage_payload, indent=2), encoding="utf-8"
    )

    return {
        "run_id": run_id,
        "ranking_mode": "value_first_v1.1",
        "ticker_count": len(updated_rows),
        "value_first_rank": value_rank,
        "metrics_added": metrics_added,
        "coverage_metrics_added": coverage_metrics,
        "trace_missing_count": int(trace_missing_count),
        "peer_scoreboard_path": str(scoreboard_path),
        "peer_rankings_path": str(rankings_path),
        "shares_coverage_path": str(shares_coverage_path),
        "fcf_coverage_path": str(fcf_coverage_path),
        "facts_coverage_path": str(facts_coverage_path),
        "valuation_coverage_path": str(valuation_coverage_path),
    }
