from __future__ import annotations

import json
from typing import Any

from app.config import get_config
from app.db import get_db
from app.dossier.whale_signals import build_whale_signals


UNKNOWN = "UNKNOWN"
FUTURE_WHALE_SIGNATURE_WEIGHT = 0.65
FUTURE_WHALE_LEGACY_COMPONENT_WEIGHT = 35.0


def _signal_lookup(dossier: dict[str, Any], signal: str) -> float | str:
    for row in (dossier.get("time_series") or {}).get("derived_signals", []):
        if row.get("signal") == signal:
            value = row.get("value")
            return float(value) if isinstance(value, (int, float)) else UNKNOWN
    return UNKNOWN


def _latest_std_value(dossier: dict[str, Any], field: str) -> float | str:
    rows = (dossier.get("time_series") or {}).get("standardized_rows") or []
    if not rows:
        return UNKNOWN
    value = rows[-1].get(field, UNKNOWN)
    return float(value) if isinstance(value, (int, float)) else UNKNOWN


def _safe_ratio(numerator: Any, denominator: Any) -> float | str:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return UNKNOWN
    if float(denominator) == 0:
        return UNKNOWN
    return float(numerator) / float(denominator)


def _rank_values(values: list[tuple[str, float | str]], *, higher_is_better: bool = True) -> dict[str, int]:
    known = [(t, float(v)) for t, v in values if isinstance(v, (int, float))]
    known.sort(key=lambda x: x[1], reverse=higher_is_better)
    ranks: dict[str, int] = {}
    for idx, (ticker, _) in enumerate(known, start=1):
        ranks[ticker] = idx
    for ticker, _ in values:
        if ticker not in ranks:
            ranks[ticker] = len(known) + 1
    return ranks


def _best_citation(dossier: dict[str, Any]) -> dict[str, Any]:
    for claim in dossier.get("claims", []):
        citations = claim.get("citations") or []
        if citations:
            return citations[0]
    for item in dossier.get("items", []):
        if item.get("source_url"):
            return {
                "source_url": item.get("source_url"),
                "snippet": item.get("snippet", ""),
                "section_label": item.get("section_label"),
            }
    return {"source_url": "", "snippet": "", "section_label": None}


def _latest_score_context(ticker: str, as_of_date: str) -> dict[str, Any]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT subscores_json, total_score
            FROM scores
            WHERE ticker = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    if not row:
        return {"valuation_gap": UNKNOWN, "total_score": UNKNOWN}
    try:
        subscores = json.loads(row["subscores_json"] or "{}")
    except Exception:
        subscores = {}
    valuation_gap = subscores.get("valuation_gap", UNKNOWN)
    total_score = row["total_score"]
    return {
        "valuation_gap": float(valuation_gap) if isinstance(valuation_gap, (int, float)) else UNKNOWN,
        "total_score": float(total_score) if isinstance(total_score, (int, float)) else UNKNOWN,
    }


def _latest_price_context(ticker: str, as_of_date: str) -> dict[str, Any]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT provider, status, price
            FROM price_quotes
            WHERE ticker = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, fetched_at DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    if not row:
        return {"has_price": False, "price": UNKNOWN, "provider": None}
    return {
        "has_price": bool(row["status"] == "OK" and isinstance(row["price"], (int, float))),
        "price": float(row["price"]) if isinstance(row["price"], (int, float)) else UNKNOWN,
        "provider": row["provider"],
    }


def _score_components(dossier: dict[str, Any], *, as_of_date: str) -> dict[str, Any]:
    ticker = str(dossier.get("ticker") or "").upper()
    revenue_cagr_5y = _signal_lookup(dossier, "revenue_cagr_5y")
    revenue_cagr_10y = _signal_lookup(dossier, "revenue_cagr_10y")
    gross_slope = _signal_lookup(dossier, "gross_margin_trend_slope")
    op_slope = _signal_lookup(dossier, "operating_margin_trend_slope")
    fcf_slope = _signal_lookup(dossier, "fcf_margin_trend_slope")
    dilution = _signal_lookup(dossier, "dilution_rate_shares_cagr")
    risk_delta = _signal_lookup(dossier, "risk_factor_keyword_delta")
    roic_proxy = _signal_lookup(dossier, "roic_proxy")
    latest_cfo = _latest_std_value(dossier, "cfo")
    latest_net_income = _latest_std_value(dossier, "net_income")
    cash_conversion_proxy = _safe_ratio(latest_cfo, latest_net_income)
    net_debt = _latest_std_value(dossier, "net_debt")
    score_context = _latest_score_context(ticker, as_of_date)
    price_context = _latest_price_context(ticker, as_of_date)
    return {
        "revenue_cagr_5y": revenue_cagr_5y,
        "revenue_cagr_10y": revenue_cagr_10y,
        "gross_margin_trend_slope": gross_slope,
        "operating_margin_trend_slope": op_slope,
        "fcf_margin_trend_slope": fcf_slope,
        "dilution_rate_shares_cagr": dilution,
        "risk_factor_keyword_delta": risk_delta,
        "net_debt_latest": net_debt,
        "net_debt_proxy": net_debt,
        "roic_proxy": roic_proxy,
        "cash_conversion_proxy": cash_conversion_proxy,
        "valuation_gap": score_context.get("valuation_gap", UNKNOWN),
        "score_total": score_context.get("total_score", UNKNOWN),
        "has_price": bool(price_context.get("has_price")),
    }


def _to_float(value: Any, fallback: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else fallback


def _top_list(rows: list[dict[str, Any]], key: str, *, reverse: bool = True, limit: int = 3) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: (-_to_float(r.get(key), -1e9) if reverse else _to_float(r.get(key), 1e9), r["ticker"]))
    return sorted_rows[:limit]


def _top_whale_signals(whale_payload: dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
    rows = whale_payload.get("signals") or whale_payload.get("whale_signals_json") or []
    if not isinstance(rows, list):
        return []
    ranked = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: (-_to_float(row.get("score_contribution"), 0.0), str(row.get("signal") or "")),
    )
    out: list[dict[str, Any]] = []
    for row in ranked[: max(1, int(limit))]:
        out.append(
            {
                "signal": row.get("signal"),
                "score_contribution": row.get("score_contribution"),
                "status": row.get("status"),
                "explanation": row.get("explanation"),
                "derived_from": row.get("derived_from") or [],
            }
        )
    return out


SCOREBOARD_METRICS = [
    "revenue_cagr_10y",
    "operating_margin_trend_slope",
    "fcf_margin_trend_slope",
    "dilution_rate_shares_cagr",
    "roic_proxy",
    "cash_conversion_proxy",
    "net_debt_proxy",
    "whale_signature_score",
]

LOWER_BETTER_METRICS = {"dilution_rate_shares_cagr", "net_debt_proxy"}


def _signal_trace_map(dossier: dict[str, Any]) -> dict[str, list[str]]:
    traces: dict[str, list[str]] = {}
    for row in (dossier.get("time_series") or {}).get("derived_signals", []):
        if not isinstance(row, dict):
            continue
        signal = str(row.get("signal") or "").strip()
        if not signal:
            continue
        traces[signal] = [str(ref) for ref in (row.get("derived_from") or []) if str(ref).strip()]
    return traces


def _metric_trace(dossier: dict[str, Any], metric: str) -> list[str]:
    signal_traces = _signal_trace_map(dossier)
    if metric in signal_traces:
        return signal_traces[metric]
    std_rows = (dossier.get("time_series") or {}).get("standardized_rows") or []
    latest_year = std_rows[-1].get("year") if std_rows else None
    if metric in {"net_debt_proxy", "net_debt_latest"} and latest_year is not None:
        return [f"dossier.time_series.standardized_rows[{latest_year}].net_debt"]
    if metric == "cash_conversion_proxy" and latest_year is not None:
        return [
            f"dossier.time_series.standardized_rows[{latest_year}].cfo",
            f"dossier.time_series.standardized_rows[{latest_year}].net_income",
        ]
    if metric == "whale_signature_score":
        return ["dossier.whale_signals.whale_signature_score"]
    return [f"dossier.metrics.{metric}"]


def _quantile(values: list[float], q: float) -> float | str:
    if not values:
        return UNKNOWN
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[idx])


def _build_metric_distributions(scoreboard_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    distributions: dict[str, dict[str, Any]] = {}
    for metric in SCOREBOARD_METRICS:
        known_values = [
            float((row.get("metric_values") or {}).get(metric))
            for row in scoreboard_rows
            if isinstance((row.get("metric_values") or {}).get(metric), (int, float))
        ]
        distributions[metric] = {
            "median": _quantile(known_values, 0.50),
            "p25": _quantile(known_values, 0.25),
            "p75": _quantile(known_values, 0.75),
            "sample_size": len(known_values),
            "derived_from": [f"peer_scoreboard.rows[*].metric_values.{metric}"],
        }
    return distributions


def _winner_vs_peer_deltas(
    *,
    rankings: list[dict[str, Any]],
    scoreboard_rows: list[dict[str, Any]],
    distributions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_ticker = {row.get("ticker"): row for row in scoreboard_rows}
    out: list[dict[str, Any]] = []
    for ranked in rankings[:3]:
        ticker = ranked.get("ticker")
        scoreboard_row = by_ticker.get(ticker)
        if not scoreboard_row:
            continue
        bullets: list[str] = []
        delta_rows: list[dict[str, Any]] = []
        metric_values = scoreboard_row.get("metric_values") or {}
        for metric in SCOREBOARD_METRICS:
            value = metric_values.get(metric, UNKNOWN)
            dist = distributions.get(metric, {})
            median = dist.get("median", UNKNOWN)
            p75 = dist.get("p75", UNKNOWN)
            if not isinstance(value, (int, float)) or not isinstance(median, (int, float)):
                continue
            higher_is_better = metric not in LOWER_BETTER_METRICS
            delta_vs_median = float(value) - float(median) if higher_is_better else float(median) - float(value)
            delta_vs_p75 = (
                float(value) - float(p75)
                if higher_is_better and isinstance(p75, (int, float))
                else (float(p75) - float(value) if isinstance(p75, (int, float)) else UNKNOWN)
            )
            if delta_vs_median > 0:
                bullets.append(f"{metric}: above peer median by {delta_vs_median:.4f}.")
            if isinstance(delta_vs_p75, (int, float)) and delta_vs_p75 > 0:
                bullets.append(f"{metric}: stronger than peer p75 by {delta_vs_p75:.4f}.")
            delta_rows.append(
                {
                    "metric": metric,
                    "value": value,
                    "median": median,
                    "p75": p75,
                    "delta_vs_median": round(float(delta_vs_median), 6),
                    "delta_vs_p75": round(float(delta_vs_p75), 6) if isinstance(delta_vs_p75, (int, float)) else UNKNOWN,
                    "derived_from": [f"peer_scoreboard.distributions.{metric}"] + (_metric_trace(scoreboard_row.get("dossier") or {}, metric)),
                }
            )
        out.append(
            {
                "ticker": ticker,
                "bullets": bullets[:6],
                "deltas": delta_rows,
                "derived_from": [f"peer_scoreboard.rows[{ticker}]"],
            }
        )
    return out


def _common_gap_counts(scoreboard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in scoreboard_rows:
        whale_summary = row.get("whale_summary") or {}
        for gap in whale_summary.get("gaps") or []:
            if not isinstance(gap, dict):
                continue
            for metric in gap.get("missing_metrics") or []:
                metric_name = str(metric).strip()
                if not metric_name:
                    continue
                counts[metric_name] = counts.get(metric_name, 0) + 1
    return [
        {"metric": metric, "count": counts[metric], "derived_from": [f"peer_scoreboard.rows[*].whale_summary.gaps.{metric}"]}
        for metric in sorted(counts.keys(), key=lambda name: (-counts[name], name))
    ]


def build_peer_report(
    *,
    run_id: str,
    as_of_date: str,
    dossiers: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = get_config()
    out_dir = cfg.dossiers_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dossier in dossiers:
        ticker = str(dossier.get("ticker") or "").upper()
        if not ticker:
            continue
        metrics = _score_components(dossier, as_of_date=as_of_date)
        whale_payload = build_whale_signals(dossier)
        whale_score = whale_payload.get("whale_signature_score", UNKNOWN)
        metrics["whale_signature_score"] = float(whale_score) if isinstance(whale_score, (int, float)) else UNKNOWN
        metrics["whale_signal_count"] = len(whale_payload.get("signals") or [])
        metrics["whale_gap_count"] = len(whale_payload.get("gaps") or [])
        citation = _best_citation(dossier)
        rows.append(
            {
                "ticker": ticker,
                "metrics": metrics,
                "whale": whale_payload,
                "citation": citation,
                "dossier": dossier,
            }
        )

    # Rank components.
    quality_values = []
    future_values = []
    whale_values = []
    risk_values = []
    valuation_values = []
    for row in rows:
        ticker = row["ticker"]
        m = row["metrics"]
        quality_score = (
            _to_float(m.get("revenue_cagr_5y"))
            + _to_float(m.get("operating_margin_trend_slope"))
            + _to_float(m.get("fcf_margin_trend_slope"))
            + _to_float(m.get("gross_margin_trend_slope"))
            - _to_float(m.get("dilution_rate_shares_cagr"))
        )
        legacy_future_score = (
            _to_float(m.get("revenue_cagr_10y"))
            + _to_float(m.get("operating_margin_trend_slope"))
            + _to_float(m.get("fcf_margin_trend_slope"))
            + (_to_float(m.get("valuation_gap")) / 50.0)
            - _to_float(m.get("risk_factor_keyword_delta")) / 10.0
        )
        whale_signature_score = m.get("whale_signature_score", UNKNOWN)
        future_score = (
            FUTURE_WHALE_SIGNATURE_WEIGHT * _to_float(whale_signature_score, 0.0)
            + FUTURE_WHALE_LEGACY_COMPONENT_WEIGHT * legacy_future_score
        )
        risk_score = (
            -_to_float(m.get("net_debt_latest")) / 1_000_000_000.0
            - _to_float(m.get("risk_factor_keyword_delta"))
            - _to_float(m.get("dilution_rate_shares_cagr"))
        )
        valuation_score = m.get("valuation_gap", UNKNOWN) if m.get("has_price") else UNKNOWN
        row["quality_score"] = round(quality_score, 6)
        row["legacy_future_component"] = round(legacy_future_score, 6)
        row["whale_signature_score"] = whale_signature_score if isinstance(whale_signature_score, (int, float)) else UNKNOWN
        row["future_whale_score"] = round(future_score, 6)
        row["risk_score"] = round(risk_score, 6)
        row["valuation_score"] = round(float(valuation_score), 6) if isinstance(valuation_score, (int, float)) else UNKNOWN
        quality_values.append((ticker, row["quality_score"]))
        future_values.append((ticker, row["future_whale_score"]))
        whale_values.append((ticker, row["whale_signature_score"]))
        risk_values.append((ticker, row["risk_score"]))
        valuation_values.append((ticker, row["valuation_score"]))

    quality_rank = _rank_values(quality_values, higher_is_better=True)
    future_rank = _rank_values(future_values, higher_is_better=True)
    whale_rank = _rank_values(whale_values, higher_is_better=True)
    risk_rank = _rank_values(risk_values, higher_is_better=True)
    valuation_rank = _rank_values(valuation_values, higher_is_better=True)

    rankings: list[dict[str, Any]] = []
    for row in rows:
        ticker = row["ticker"]
        rankings.append(
            {
                "ticker": ticker,
                "overall_score": round(
                    float(
                        (len(rows) + 1 - quality_rank[ticker])
                        + (len(rows) + 1 - future_rank[ticker])
                        + (len(rows) + 1 - whale_rank[ticker])
                        + (len(rows) + 1 - risk_rank[ticker])
                    ),
                    4,
                ),
                "metric_values": row["metrics"],
                "metric_traces": {
                    metric: {"derived_from": _metric_trace(row.get("dossier") or {}, metric)}
                    for metric in (row.get("metrics") or {}).keys()
                },
                "metric_ranks": {
                    "quality_rank": quality_rank[ticker],
                    "future_whale_rank": future_rank[ticker],
                    "whale_signature_rank": whale_rank[ticker],
                    "risk_rank": risk_rank[ticker],
                    "valuation_rank": valuation_rank[ticker],
                },
                "whale_signature": {
                    "score": row.get("whale_signature_score", UNKNOWN),
                    "signals": row.get("whale", {}).get("signals") or [],
                    "top_signals": _top_whale_signals(row.get("whale", {}), limit=3),
                    "gaps": row.get("whale", {}).get("gaps") or [],
                    "total_penalty": row.get("whale", {}).get("total_penalty"),
                },
                "top_differentiators": [
                    {
                        "metric": "future_whale_score",
                        "value": row["future_whale_score"],
                        "peer_rank": future_rank[ticker],
                        "citation": row["citation"],
                        "summary": f"Future-whale composite ranks #{future_rank[ticker]} in peer set.",
                    },
                    {
                        "metric": "quality_score",
                        "value": row["quality_score"],
                        "peer_rank": quality_rank[ticker],
                        "citation": row["citation"],
                        "summary": f"Quality composite ranks #{quality_rank[ticker]} in peer set.",
                    },
                    {
                        "metric": "whale_signature_score",
                        "value": row.get("whale_signature_score", UNKNOWN),
                        "peer_rank": whale_rank[ticker],
                        "citation": row["citation"],
                        "summary": f"Whale-signature checklist ranks #{whale_rank[ticker]} in peer set.",
                    },
                    {
                        "metric": "risk_score",
                        "value": row["risk_score"],
                        "peer_rank": risk_rank[ticker],
                        "citation": row["citation"],
                        "summary": f"Risk composite ranks #{risk_rank[ticker]} in peer set.",
                    },
                ],
            }
        )
    rankings.sort(key=lambda row: (-float(row["overall_score"]), row["ticker"]))

    # Section lists (top 3 each).
    _by_ticker = {row["ticker"]: row for row in rows}
    rank_lookup = {row["ticker"]: idx for idx, row in enumerate(rankings, start=1)}
    top_compounders = _top_list(rows, "future_whale_score", reverse=True, limit=3)
    top_capital_discipline = _top_list(rows, "quality_score", reverse=True, limit=3)
    whale_leaders = _top_list(rows, "whale_signature_score", reverse=True, limit=5)
    top_balance_sheet = _top_list(rows, "risk_score", reverse=True, limit=3)
    top_valuation = [r for r in _top_list(rows, "valuation_score", reverse=True, limit=3) if isinstance(r.get("valuation_score"), (int, float))]

    scoreboard_rows: list[dict[str, Any]] = []
    for ranked in rankings:
        ticker = ranked["ticker"]
        detail = _by_ticker.get(ticker) or {}
        dossier = detail.get("dossier") or {}
        whale_payload = detail.get("whale") or {}
        metric_traces = {
            metric: {"derived_from": _metric_trace(dossier, metric)}
            for metric in (ranked.get("metric_values") or {}).keys()
        }
        scoreboard_rows.append(
            {
                "ticker": ticker,
                "overall_rank": rank_lookup.get(ticker),
                "overall_score": ranked.get("overall_score"),
                "metric_values": ranked.get("metric_values") or {},
                "metric_ranks": ranked.get("metric_ranks") or {},
                "metric_traces": metric_traces,
                "standardized_rows": (dossier.get("time_series") or {}).get("standardized_rows") or [],
                "standardized_row_traces": (dossier.get("time_series") or {}).get("standardized_row_traces") or {},
                "derived_signals": (dossier.get("time_series") or {}).get("derived_signals") or [],
                "whale_summary": {
                    "score": whale_payload.get("whale_signature_score", UNKNOWN),
                    "top_signals": _top_whale_signals(whale_payload, limit=3),
                    "gaps": whale_payload.get("gaps") or [],
                    "critical_unknowns": whale_payload.get("critical_unknowns") or [],
                },
                "gaps": whale_payload.get("gaps") or [],
                "dossier_path": dossier.get("artifacts", {}).get("dossier_json_path"),
                "dossier": dossier,
            }
        )
    metric_distributions = _build_metric_distributions(scoreboard_rows)
    winner_vs_peer = _winner_vs_peer_deltas(
        rankings=rankings,
        scoreboard_rows=scoreboard_rows,
        distributions=metric_distributions,
    )
    common_gaps = _common_gap_counts(scoreboard_rows)

    report_md_lines = [
        f"# Peer Differentiation Report ({run_id})",
        "",
        f"- As Of Date: `{as_of_date}`",
        f"- Peer Count: `{len(rankings)}`",
        "",
        "| Ticker | Future Whale Rank | Whale Signature Rank | Quality Rank | Valuation Rank | Risk Rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rankings:
        ranks = row["metric_ranks"]
        report_md_lines.append(
            f"| {row['ticker']} | {ranks.get('future_whale_rank')} | {ranks.get('whale_signature_rank')} | {ranks.get('quality_rank')} | {ranks.get('valuation_rank')} | {ranks.get('risk_rank')} |"
        )

    report_md_lines.append("")
    report_md_lines.append("## Scoreboard (Top 25)")
    report_md_lines.append("| Rank | Ticker | Whale Score | Rev CAGR 10Y | Op Margin Slope | FCF Margin Slope | Dilution CAGR |")
    report_md_lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in scoreboard_rows[:25]:
        values = row.get("metric_values") or {}
        def _fmt(name: str) -> str:
            value = values.get(name, UNKNOWN)
            return f"{float(value):.4f}" if isinstance(value, (int, float)) else "UNKNOWN"
        rank_text = row.get("overall_rank") or "UNKNOWN"
        report_md_lines.append(
            f"| {rank_text} | {row['ticker']} | {_fmt('whale_signature_score')} | {_fmt('revenue_cagr_10y')} | {_fmt('operating_margin_trend_slope')} | {_fmt('fcf_margin_trend_slope')} | {_fmt('dilution_rate_shares_cagr')} |"
        )

    report_md_lines.append("")
    report_md_lines.append("## Peer Distributions")
    for metric in SCOREBOARD_METRICS:
        dist = metric_distributions.get(metric, {})
        median = dist.get("median", UNKNOWN)
        p25 = dist.get("p25", UNKNOWN)
        p75 = dist.get("p75", UNKNOWN)
        size = dist.get("sample_size", 0)
        median_text = f"{float(median):.4f}" if isinstance(median, (int, float)) else "UNKNOWN"
        p25_text = f"{float(p25):.4f}" if isinstance(p25, (int, float)) else "UNKNOWN"
        p75_text = f"{float(p75):.4f}" if isinstance(p75, (int, float)) else "UNKNOWN"
        report_md_lines.append(
            f"- **{metric}**: median `{median_text}`; p25 `{p25_text}`; p75 `{p75_text}`; n=`{size}`."
        )

    report_md_lines.append("")
    report_md_lines.append("## Winner vs Peer Deltas")
    if not winner_vs_peer:
        report_md_lines.append("- No ranked candidates with sufficient comparative data.")
    else:
        for row in winner_vs_peer:
            report_md_lines.append(f"- **{row['ticker']}**:")
            bullets = row.get("bullets") or []
            if not bullets:
                report_md_lines.append("  - No positive metric deltas versus peer median/p75 with current data.")
            for bullet in bullets[:6]:
                report_md_lines.append(f"  - {bullet}")

    def _section(title: str, entries: list[dict[str, Any]], reason: str) -> None:
        report_md_lines.append("")
        report_md_lines.append(f"## {title}")
        if not entries:
            report_md_lines.append("- No ranked entries with sufficient data.")
            return
        for entry in entries:
            ticker = entry["ticker"]
            citation = (_by_ticker.get(ticker) or {}).get("citation", {})
            src = citation.get("source_url", "")
            snippet = str(citation.get("snippet", "")).strip()
            snippet_short = snippet[:180] + ("..." if len(snippet) > 180 else "")
            report_md_lines.append(
                f"- **{ticker}**: {reason}. [source]({src}) {snippet_short}"
            )

    _section("Top Compounders", top_compounders, "Revenue + margin + cash conversion trend composite is strongest")
    _section("Capital Discipline", top_capital_discipline, "Lower dilution pressure with improving cash flow quality")
    report_md_lines.append("")
    report_md_lines.append("## Whale Signature Leaders")
    if not whale_leaders:
        report_md_lines.append("- No whale-signature entries with sufficient data.")
    else:
        for entry in whale_leaders:
            ticker = entry["ticker"]
            whale_payload = (_by_ticker.get(ticker) or {}).get("whale", {})
            top_signals = _top_whale_signals(whale_payload, limit=3)
            signals_text = ", ".join([str(row.get("signal")) for row in top_signals if row.get("signal")]) or "UNKNOWN"
            score = whale_payload.get("whale_signature_score", UNKNOWN)
            score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "UNKNOWN"
            report_md_lines.append(f"- **{ticker}**: score `{score_text}`; top signals: {signals_text}.")
    _section("Balance Sheet Risk", top_balance_sheet, "Lower net-debt and risk-signal pressure")
    _section("Valuation vs Quality", top_valuation, "Valuation gap supports quality profile when price data is available")
    report_md_lines.append("")
    report_md_lines.append("## Data Gaps That Limit Confidence")
    if not common_gaps:
        report_md_lines.append("- No recurring gap metrics found.")
    else:
        for gap in common_gaps[:10]:
            report_md_lines.append(f"- **{gap['metric']}** appears in `{gap['count']}` peer gap entries.")
    for row in winner_vs_peer:
        report_md_lines.append(f"- **{row['ticker']} candidate gaps**:")
        ticker_row = next((candidate for candidate in scoreboard_rows if candidate.get("ticker") == row["ticker"]), None)
        ticker_gaps = ticker_row.get("gaps") if isinstance(ticker_row, dict) else []
        if not ticker_gaps:
            report_md_lines.append("  - none")
            continue
        listed = 0
        for gap in ticker_gaps:
            if not isinstance(gap, dict):
                continue
            signal = str(gap.get("signal") or "UNKNOWN")
            missing = ", ".join([str(metric) for metric in (gap.get("missing_metrics") or []) if str(metric).strip()]) or "UNKNOWN"
            report_md_lines.append(f"  - {signal}: {missing}")
            listed += 1
            if listed >= 5:
                break

    peer_rankings = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "future_whale_formula": {
            "whale_signature_weight": FUTURE_WHALE_SIGNATURE_WEIGHT,
            "legacy_component_weight": FUTURE_WHALE_LEGACY_COMPONENT_WEIGHT,
            "legacy_component_definition": (
                "revenue_cagr_10y + operating_margin_trend_slope + fcf_margin_trend_slope + "
                "(valuation_gap/50) - (risk_factor_keyword_delta/10)"
            ),
        },
        "tickers": [row["ticker"] for row in rankings],
        "rankings": rankings,
        "future_whale_rank": [row["ticker"] for row in sorted(rankings, key=lambda r: r["metric_ranks"]["future_whale_rank"])],
        "whale_signature_rank": [row["ticker"] for row in sorted(rankings, key=lambda r: r["metric_ranks"]["whale_signature_rank"])],
        "quality_rank": [row["ticker"] for row in sorted(rankings, key=lambda r: r["metric_ranks"]["quality_rank"])],
        "valuation_rank": [row["ticker"] for row in sorted(rankings, key=lambda r: r["metric_ranks"]["valuation_rank"])],
        "risk_rank": [row["ticker"] for row in sorted(rankings, key=lambda r: r["metric_ranks"]["risk_rank"])],
        "sections": {
            "top_compounders": [row["ticker"] for row in top_compounders],
            "capital_discipline": [row["ticker"] for row in top_capital_discipline],
            "whale_signature_leaders": [row["ticker"] for row in whale_leaders],
            "balance_sheet_risk": [row["ticker"] for row in top_balance_sheet],
            "valuation_vs_quality": [row["ticker"] for row in top_valuation],
        },
    }
    scoreboard_rows_out: list[dict[str, Any]] = []
    for row in scoreboard_rows:
        cleaned = dict(row)
        cleaned.pop("dossier", None)
        scoreboard_rows_out.append(cleaned)
    peer_scoreboard = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "metrics": SCOREBOARD_METRICS,
        "rows": scoreboard_rows_out,
        "distributions": metric_distributions,
        "winner_vs_peer_deltas": winner_vs_peer,
        "common_gaps": common_gaps,
        "derived_from": {
            "rows": ["dossier.time_series.standardized_rows", "dossier.time_series.derived_signals", "dossier.whale_signals"],
            "distributions": ["peer_scoreboard.rows[*].metric_values"],
        },
    }
    json_path = out_dir / "peer_rankings.json"
    json_path.write_text(json.dumps(peer_rankings, indent=2), encoding="utf-8")
    scoreboard_path = out_dir / "peer_scoreboard.json"
    scoreboard_path.write_text(json.dumps(peer_scoreboard, indent=2), encoding="utf-8")
    md_path = out_dir / "peer_report.md"
    md_path.write_text("\n".join(report_md_lines) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "peer_rankings_path": str(json_path),
        "peer_scoreboard_path": str(scoreboard_path),
        "peer_report_path": str(md_path),
        "rankings": rankings,
    }


def build_peer_report_from_run(*, run_id: str, as_of_date: str | None = None) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.dossiers_dir / run_id
    if not run_dir.exists():
        raise ValueError(f"dossier run not found: {run_id}")

    dossiers: list[dict[str, Any]] = []
    inferred_as_of = as_of_date
    for ticker_dir in sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.isupper()], key=lambda p: p.name):
        dossier_path = ticker_dir / "dossier.json"
        if not dossier_path.exists():
            continue
        payload = json.loads(dossier_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if inferred_as_of is None and isinstance(payload.get("as_of_date"), str):
            inferred_as_of = payload["as_of_date"]
        dossiers.append(payload)
    if not dossiers:
        raise ValueError(f"no dossier payloads found for run_id={run_id}")
    if inferred_as_of is None:
        inferred_as_of = "UNKNOWN"
    return build_peer_report(run_id=run_id, as_of_date=inferred_as_of, dossiers=dossiers)
