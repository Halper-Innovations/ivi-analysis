from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.dossier.whale_signals import run_whale_signals_for_run
from app.sector.schemas import validate_sector_decision_pack


UNKNOWN = "UNKNOWN"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _to_float(value: Any, fallback: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else fallback


def _metric_value(scoreboard_row: dict[str, Any], key: str) -> float | None:
    metric_values = scoreboard_row.get("metric_values") if isinstance(scoreboard_row, dict) else {}
    value = (metric_values or {}).get(key, None)
    return float(value) if isinstance(value, (int, float)) else None


def _ranking_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    metric_ranks = row.get("metric_ranks") or {}
    value_rank = metric_ranks.get("value_first_rank")
    future_rank = metric_ranks.get("future_whale_rank")
    whale_rank = metric_ranks.get("whale_signature_rank")
    return (
        int(value_rank) if isinstance(value_rank, int) else 10_000,
        int(future_rank) if isinstance(future_rank, int) else 10_000,
        int(whale_rank) if isinstance(whale_rank, int) else 10_000,
        str(row.get("ticker") or ""),
    )


def _flatten_gap_metrics(whale_signature: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for gap in whale_signature.get("gaps") or []:
        if not isinstance(gap, dict):
            continue
        for metric in gap.get("missing_metrics") or []:
            metric_norm = str(metric).strip()
            if metric_norm:
                out.append(metric_norm)
    deduped: list[str] = []
    seen: set[str] = set()
    for metric in out:
        if metric in seen:
            continue
        seen.add(metric)
        deduped.append(metric)
    return deduped


def _candidate_rows(rankings: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rankings, key=_ranking_sort_key)
    candidate_cap = max(3, min(5, int(limit)))
    return ordered[:candidate_cap]


def build_sector_decision_pack(
    *,
    sector: str,
    as_of_date: str,
    run_id: str,
    top_n: int = 10,
) -> dict[str, Any]:
    cfg = get_config()
    sector_dir = cfg.sectors_dir / run_id
    sector_dir.mkdir(parents=True, exist_ok=True)
    dossier_dir = cfg.dossiers_dir / run_id

    peer_rankings_path = sector_dir / "peer_rankings.json"
    if not peer_rankings_path.exists():
        peer_rankings_path = dossier_dir / "peer_rankings.json"
    peer_rankings = _read_json(peer_rankings_path)
    if not peer_rankings:
        raise ValueError(f"peer_rankings.json not found for run_id={run_id}")
    rankings = [row for row in (peer_rankings.get("rankings") or []) if isinstance(row, dict)]
    if not rankings:
        raise ValueError("peer_rankings has no ranking rows")
    ranking_mode = str(peer_rankings.get("ranking_mode") or "future_whale")
    rankings = sorted(rankings, key=_ranking_sort_key)

    peer_scoreboard_path = sector_dir / "peer_scoreboard.json"
    if not peer_scoreboard_path.exists():
        peer_scoreboard_path = dossier_dir / "peer_scoreboard.json"
    peer_scoreboard = _read_json(peer_scoreboard_path) or {}
    scoreboard_rows = [row for row in (peer_scoreboard.get("rows") or []) if isinstance(row, dict)]
    scoreboard_by_ticker = {str(row.get("ticker") or "").upper(): row for row in scoreboard_rows}

    whale_summary_path = sector_dir / "whale_signals_summary.json"
    if not whale_summary_path.exists():
        whale_summary_path = dossier_dir / "whale_signals_summary.json"
    if not whale_summary_path.exists():
        run_whale_signals_for_run(run_id=run_id)
        whale_summary_path = dossier_dir / "whale_signals_summary.json"
    whale_summary = _read_json(whale_summary_path) or {}

    peer_report_path = sector_dir / "peer_report.md"
    if not peer_report_path.exists():
        peer_report_path = dossier_dir / "peer_report.md"
    sector_synthesis_path = sector_dir / "sector_synthesis.json"
    sector_synthesis = _read_json(sector_synthesis_path) if sector_synthesis_path.exists() else None

    top_peers: list[dict[str, Any]] = []
    for idx, row in enumerate(rankings[:25], start=1):
        ticker = str(row.get("ticker") or "").upper()
        scoreboard_row = scoreboard_by_ticker.get(ticker, {})
        metric_ranks = row.get("metric_ranks") or {}
        top_peers.append(
            {
                "rank": idx,
                "ticker": ticker,
                "overall_score": _to_float(row.get("overall_score")),
                "future_whale_rank": metric_ranks.get("future_whale_rank"),
                "whale_signature_rank": metric_ranks.get("whale_signature_rank"),
                "quality_rank": metric_ranks.get("quality_rank"),
                "valuation_rank": metric_ranks.get("valuation_rank"),
                "risk_rank": metric_ranks.get("risk_rank"),
                "value_first_rank": metric_ranks.get("value_first_rank"),
                "quality_score": _metric_value(scoreboard_row, "quality_score"),
                "growth_score": _metric_value(scoreboard_row, "growth_score"),
                "capital_discipline_score": _metric_value(scoreboard_row, "capital_discipline_score"),
                "valuation_score": _metric_value(scoreboard_row, "valuation_score"),
                "risk_penalty": _metric_value(scoreboard_row, "risk_penalty"),
                "score_total": _metric_value(scoreboard_row, "score_total"),
                "implied_return_base": _metric_value(scoreboard_row, "implied_return_base"),
                "derived_from": [
                    f"peer_rankings.rankings[{idx-1}].overall_score",
                    f"peer_rankings.rankings[{idx-1}].metric_ranks",
                    f"peer_scoreboard.rows[{ticker}].metric_values",
                ],
            }
        )

    whale_leader_rows = sorted(
        rankings,
        key=lambda row: (
            int((row.get("metric_ranks") or {}).get("whale_signature_rank") or 10_000),
            str(row.get("ticker") or ""),
        ),
    )[:5]
    whale_leaders: list[dict[str, Any]] = []
    for row in whale_leader_rows:
        ticker = str(row.get("ticker") or "")
        whale_signature = row.get("whale_signature") or {}
        leader_top_signals = []
        for signal in (whale_signature.get("top_signals") or [])[:3]:
            if not isinstance(signal, dict):
                continue
            leader_top_signals.append(
                {
                    "signal": str(signal.get("signal") or ""),
                    "score_contribution": _to_float(signal.get("score_contribution")),
                    "status": str(signal.get("status") or "UNKNOWN"),
                }
            )
        whale_leaders.append(
            {
                "ticker": ticker,
                "whale_signature_score": _to_float(whale_signature.get("score")),
                "top_signals": leader_top_signals,
                "derived_from": [
                    f"peer_rankings.rankings[{ticker}].whale_signature.score",
                    f"peer_rankings.rankings[{ticker}].whale_signature.top_signals",
                ],
            }
        )

    top_candidates_raw = _candidate_rows(rankings, limit=max(3, min(5, int(top_n))))
    fallback_checks = [
        "Recent revenue growth persistence breaks below whale thresholds.",
        "FCF trend weakens while CFO/NI quality deteriorates.",
        "Net debt worsens materially without matching cash generation.",
    ]
    synthesis_checks = [str(item) for item in ((sector_synthesis or {}).get("falsifiers") or []) if str(item).strip()]
    change_mind_checks = synthesis_checks[:6] if synthesis_checks else fallback_checks

    top_candidates: list[dict[str, Any]] = []
    candidate_limit = max(3, min(5, int(top_n)))
    top_candidates_raw = top_candidates_raw[:candidate_limit]
    for row in top_candidates_raw:
        ticker = str(row.get("ticker") or "")
        metric_ranks = row.get("metric_ranks") or {}
        whale_signature = row.get("whale_signature") or {}
        scoreboard_row = scoreboard_by_ticker.get(ticker, {})
        quality_score = _metric_value(scoreboard_row, "quality_score")
        valuation_score = _metric_value(scoreboard_row, "valuation_score")
        implied_return = _metric_value(scoreboard_row, "implied_return_base")
        risk_penalty = _metric_value(scoreboard_row, "risk_penalty")
        score_total = _metric_value(scoreboard_row, "score_total")
        top_signal_names = [
            str(signal.get("signal"))
            for signal in (whale_signature.get("top_signals") or [])
            if isinstance(signal, dict) and str(signal.get("signal") or "").strip()
        ][:3]
        reasons = [
            f"Future whale rank #{metric_ranks.get('future_whale_rank', 'UNKNOWN')}",
            f"Whale signature rank #{metric_ranks.get('whale_signature_rank', 'UNKNOWN')}",
            f"Value-first score leadership rank #{metric_ranks.get('value_first_rank', 'UNKNOWN')}",
            f"Top whale signals: {', '.join(top_signal_names) if top_signal_names else 'UNKNOWN'}",
            "Quality-growth-capital-valuation decomposition remains favorable versus peers.",
        ]
        risks: list[str] = []
        if isinstance(risk_penalty, (int, float)) and float(risk_penalty) > 8:
            risks.append("Risk penalty is elevated versus peers.")
        if not isinstance(implied_return, (int, float)):
            risks.append("Valuation confidence is limited by missing market inputs.")
        gaps = sorted(set(_flatten_gap_metrics(whale_signature) + (scoreboard_row.get("value_first", {}).get("gaps") or [])))
        if gaps:
            risks.append("Data gaps reduce confidence in some score components.")
        claim_ids = [
            f"score_total_{ticker}",
            f"quality_score_{ticker}",
            f"valuation_score_{ticker}",
            f"implied_return_base_{ticker}",
        ]
        top_candidates.append(
            {
                "ticker": ticker,
                "reasons": reasons[:5],
                "risks": risks[:5],
                "gaps": gaps,
                "claim_ids": claim_ids,
                "what_would_change_my_mind": change_mind_checks[:3],
                "derived_from": [
                    f"peer_rankings.rankings[{ticker}].metric_ranks",
                    f"peer_rankings.rankings[{ticker}].whale_signature",
                    f"peer_scoreboard.rows[{ticker}].metric_values",
                ],
            }
        )

    numeric_claims: list[dict[str, Any]] = []
    for row in top_peers[: max(1, int(top_n))]:
        ticker = str(row["ticker"])
        numeric_claims.append(
            {
                "id": f"overall_score_{ticker}",
                "label": f"{ticker} overall_score",
                "value": _to_float(row.get("overall_score")),
                "derived_from": [f"peer_rankings.rankings[{ticker}].overall_score"],
            }
        )
        for metric in [
            "score_total",
            "quality_score",
            "growth_score",
            "capital_discipline_score",
            "valuation_score",
            "risk_penalty",
            "implied_return_base",
        ]:
            value = row.get(metric, UNKNOWN)
            if not isinstance(value, (int, float)):
                continue
            numeric_claims.append(
                {
                    "id": f"{metric}_{ticker}",
                    "label": f"{ticker} {metric}",
                    "value": float(value),
                    "derived_from": [f"peer_scoreboard.rows[{ticker}].metric_values.{metric}"],
                }
            )
    for row in whale_leaders:
        ticker = str(row["ticker"])
        numeric_claims.append(
            {
                "id": f"whale_signature_score_{ticker}",
                "label": f"{ticker} whale_signature_score",
                "value": _to_float(row.get("whale_signature_score")),
                "derived_from": [f"peer_rankings.rankings[{ticker}].whale_signature.score"],
            }
        )

    peer_comparison: list[dict[str, Any]] = []
    if len(top_candidates_raw) >= 2:
        anchor = str(top_candidates_raw[0].get("ticker") or "")
        for challenger_row in top_candidates_raw[1:3]:
            challenger = str(challenger_row.get("ticker") or "")
            metrics = []
            for metric in [
                "score_total",
                "quality_score",
                "valuation_score",
                "implied_return_base",
                "risk_penalty",
                "whale_signature_score",
            ]:
                left = _metric_value(scoreboard_by_ticker.get(anchor, {}), metric)
                right = _metric_value(scoreboard_by_ticker.get(challenger, {}), metric)
                metrics.append(
                    {
                        "metric": metric,
                        "left_ticker": anchor,
                        "left_value": left,
                        "right_ticker": challenger,
                        "right_value": right,
                        "derived_from": [
                            f"peer_scoreboard.rows[{anchor}].metric_values.{metric}",
                            f"peer_scoreboard.rows[{challenger}].metric_values.{metric}",
                        ],
                    }
                )
            peer_comparison.append(
                {
                    "left_ticker": anchor,
                    "right_ticker": challenger,
                    "metrics": metrics,
                    "derived_from": [f"peer_scoreboard.rows[{anchor}]", f"peer_scoreboard.rows[{challenger}]"],
                }
            )

    value_quality_scatter_summary = {
        "high_quality_high_value": len(
            [
                row for row in top_peers
                if isinstance(row.get("quality_score"), (int, float))
                and isinstance(row.get("implied_return_base"), (int, float))
                and float(row.get("quality_score")) >= 15.0
                and float(row.get("implied_return_base")) >= 0.15
            ]
        ),
        "high_quality_low_value": len(
            [
                row for row in top_peers
                if isinstance(row.get("quality_score"), (int, float))
                and isinstance(row.get("implied_return_base"), (int, float))
                and float(row.get("quality_score")) >= 15.0
                and float(row.get("implied_return_base")) < 0.15
            ]
        ),
        "lower_quality_high_value": len(
            [
                row for row in top_peers
                if isinstance(row.get("quality_score"), (int, float))
                and isinstance(row.get("implied_return_base"), (int, float))
                and float(row.get("quality_score")) < 15.0
                and float(row.get("implied_return_base")) >= 0.15
            ]
        ),
        "unknown_bucket": len(
            [
                row for row in top_peers
                if not isinstance(row.get("quality_score"), (int, float))
                or not isinstance(row.get("implied_return_base"), (int, float))
            ]
        ),
        "derived_from": [
            "peer_scoreboard.rows[*].metric_values.quality_score",
            "peer_scoreboard.rows[*].metric_values.implied_return_base",
        ],
    }

    payload = {
        "run_id": run_id,
        "sector": sector,
        "as_of_date": as_of_date,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ranking_mode": ranking_mode,
        "top_peers": top_peers,
        "whale_signature_leaders": whale_leaders,
        "top_candidates_to_deepen": top_candidates[:candidate_limit],
        "value_quality_scatter_summary": value_quality_scatter_summary,
        "peer_comparison": peer_comparison,
        "what_would_change_my_mind": change_mind_checks,
        "numeric_claims": numeric_claims,
        "artifacts": {
            "peer_rankings_path": str(peer_rankings_path),
            "peer_scoreboard_path": str(peer_scoreboard_path),
            "whale_signals_summary_path": str(whale_summary_path),
            "peer_report_path": str(peer_report_path) if peer_report_path.exists() else None,
            "sector_synthesis_path": str(sector_synthesis_path) if sector_synthesis_path.exists() else None,
            "dossier_dir": str(dossier_dir),
        },
    }
    packet = validate_sector_decision_pack(payload)
    json_payload = packet.model_dump(mode="json")

    decision_json_path = sector_dir / "decision_pack.json"
    decision_json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# Sector Deep Decision Pack ({run_id})",
        "",
        f"- Sector: `{sector}`",
        f"- As Of Date: `{as_of_date}`",
        f"- Ranking Mode: `{ranking_mode}`",
        "",
        "## Top 25 peers (ranked)",
        "| Rank | Ticker | Overall Score | Value Total | Quality | Growth | Capital | Valuation | Risk Penalty | Implied Return Base |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in json_payload["top_peers"]:
        def _fmt_metric(name: str) -> str:
            value = row.get(name, UNKNOWN)
            return f"{float(value):.4f}" if isinstance(value, (int, float)) else "UNKNOWN"
        md_lines.append(
            f"| {row['rank']} | {row['ticker']} | {row['overall_score']:.4f} | {_fmt_metric('score_total')} | "
            f"{_fmt_metric('quality_score')} | {_fmt_metric('growth_score')} | {_fmt_metric('capital_discipline_score')} | "
            f"{_fmt_metric('valuation_score')} | {_fmt_metric('risk_penalty')} | {_fmt_metric('implied_return_base')} |"
        )

    md_lines.append("")
    md_lines.append("## Top 5 whale-signature leaders (with top 3 signals each)")
    for row in json_payload["whale_signature_leaders"]:
        top_signals = ", ".join([signal["signal"] for signal in row.get("top_signals") or [] if signal.get("signal")]) or "UNKNOWN"
        md_lines.append(
            f"- **{row['ticker']}**: whale signature score `{row['whale_signature_score']:.2f}`; top signals: {top_signals}."
        )

    md_lines.append("")
    md_lines.append("## Top 3-5 candidates to deepen next")
    for row in json_payload["top_candidates_to_deepen"]:
        reasons = "; ".join(row.get("reasons") or [])
        risks = "; ".join(row.get("risks") or []) or "none"
        gaps = ", ".join(row.get("gaps") or []) or "none"
        md_lines.append(f"- **{row['ticker']}**: reasons: {reasons}. Risks: {risks}. Gaps: {gaps}.")

    md_lines.append("")
    md_lines.append("## Value vs Quality Summary")
    scatter = json_payload.get("value_quality_scatter_summary") or {}
    md_lines.append(
        f"- High quality + high value: `{scatter.get('high_quality_high_value', 0)}`"
    )
    md_lines.append(
        f"- High quality + lower value: `{scatter.get('high_quality_low_value', 0)}`"
    )
    md_lines.append(
        f"- Lower quality + high value: `{scatter.get('lower_quality_high_value', 0)}`"
    )
    md_lines.append(f"- Unknown bucket: `{scatter.get('unknown_bucket', 0)}`")

    md_lines.append("")
    md_lines.append("## Peer Comparison (#1 vs #2/#3)")
    if not json_payload.get("peer_comparison"):
        md_lines.append("- Not enough ranked peers for comparison.")
    else:
        for comparison in json_payload["peer_comparison"]:
            left = comparison.get("left_ticker")
            right = comparison.get("right_ticker")
            md_lines.append(f"### {left} vs {right}")
            md_lines.append("| Metric | Left | Right |")
            md_lines.append("|---|---:|---:|")
            for metric_row in comparison.get("metrics") or []:
                left_value = metric_row.get("left_value", UNKNOWN)
                right_value = metric_row.get("right_value", UNKNOWN)
                left_text = f"{float(left_value):.4f}" if isinstance(left_value, (int, float)) else "UNKNOWN"
                right_text = f"{float(right_value):.4f}" if isinstance(right_value, (int, float)) else "UNKNOWN"
                md_lines.append(f"| {metric_row.get('metric')} | {left_text} | {right_text} |")

    md_lines.append("")
    md_lines.append("## What would change my mind")
    for item in json_payload["what_would_change_my_mind"]:
        md_lines.append(f"- {item}")

    decision_md_path = sector_dir / "decision_pack.md"
    decision_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "sector": sector,
        "as_of_date": as_of_date,
        "decision_pack_path": str(decision_json_path),
        "decision_pack_md_path": str(decision_md_path),
        "top_candidates": [row["ticker"] for row in json_payload["top_candidates_to_deepen"]],
        "source_artifacts": payload["artifacts"],
    }
