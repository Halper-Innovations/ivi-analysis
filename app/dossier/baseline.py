from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.config import get_config
from app.dossier.runner import run_dossier_for_peer_set
from app.dossier.time_series import build_time_series
from app.dossier.whale_signals import WHALE_SIGNAL_WEIGHTS, build_whale_signals
from app.sector.peer_set import select_sector_peers
from app.sector.taxonomy import load_sector_taxonomy


def _normalize_tickers(values: list[str] | None) -> list[str]:
    return sorted({str(value).strip().upper() for value in (values or []) if str(value).strip()})


def _suppressed_rows_from_peer_payload(peer_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in peer_payload.get("all_rows") or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        reasons = sorted({str(reason) for reason in (row.get("suppression_reasons") or []) if str(reason).strip()})
        if not ticker or not reasons:
            continue
        rows.append({"ticker": ticker, "reasons": reasons})
    rows.sort(key=lambda item: item["ticker"])
    return rows


def _infer_sector_for_controls(*, winners: list[str], sector: str | None) -> str | None:
    if sector and str(sector).strip():
        return str(sector).strip()
    taxonomy = load_sector_taxonomy()
    counts: dict[str, int] = {}
    for ticker in winners:
        mapped = taxonomy.get(ticker)
        if not mapped:
            continue
        counts[mapped] = counts.get(mapped, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _generate_controls(
    *,
    winners: list[str],
    controls: list[str] | None,
    controls_count: int,
    as_of_date: str,
    years_back: int,
    peer_mode: str,
    sector: str | None,
    min_peers: int,
    max_peers: int | None,
    sic_expand: bool,
    sic_family: bool,
    include_foreign: bool,
    include_otc: bool,
    min_annual_filings: int | None,
) -> tuple[list[str], dict[str, Any]]:
    winners_set = set(winners)
    controls_norm = [ticker for ticker in _normalize_tickers(controls) if ticker not in winners_set]
    if controls_norm:
        return controls_norm, {
            "method": "provided",
            "sector": sector,
            "peer_mode": peer_mode,
            "requested_count": int(controls_count),
            "selected_count": len(controls_norm),
            "source": None,
            "suppressed_tickers": [],
        }

    inferred_sector = _infer_sector_for_controls(winners=winners, sector=sector)
    if not inferred_sector:
        return [], {
            "method": "auto",
            "sector": None,
            "peer_mode": peer_mode,
            "requested_count": int(controls_count),
            "selected_count": 0,
            "source": None,
            "suppressed_tickers": [],
        }

    target_count = max(1, int(controls_count))
    max_peer_limit = int(max_peers) if max_peers is not None else max(target_count * 4, target_count + 5)
    peer_payload = select_sector_peers(
        sector=inferred_sector,
        as_of_date=as_of_date,
        years_back=years_back,
        limit=max_peer_limit,
        min_peers=max(1, int(min_peers)),
        max_peers=max_peer_limit,
        sic_expand=bool(sic_expand),
        sic_family=bool(sic_family),
        include_foreign=bool(include_foreign),
        include_otc=bool(include_otc),
        min_annual_filings=min_annual_filings,
        exclude_tickers=sorted(winners_set),
        mode=peer_mode,
    )
    selected = []
    for ticker in peer_payload.get("selected_tickers") or []:
        if ticker in winners_set:
            continue
        selected.append(str(ticker).upper())
        if len(selected) >= target_count:
            break
    return selected, {
        "method": "auto",
        "sector": inferred_sector,
        "peer_mode": peer_mode,
        "requested_count": int(controls_count),
        "selected_count": len(selected),
        "source": peer_payload.get("peer_selection_summary"),
        "peer_counts": peer_payload.get("counts"),
        "suppressed_tickers": _suppressed_rows_from_peer_payload(peer_payload),
    }


def _truncate_dossier_early_window(dossier: dict[str, Any], early_window_years: int) -> tuple[dict[str, Any], list[int]]:
    items = [row for row in (dossier.get("items") or []) if isinstance(row, dict) and isinstance(row.get("year"), int)]
    if not items:
        years = [int(row.get("year")) for row in (dossier.get("time_series", {}).get("standardized_rows") or []) if isinstance(row.get("year"), int)]
        years = sorted(set(years))
        years_used = years[: max(1, int(early_window_years))] if years else []
        return dict(dossier), years_used

    years = sorted({int(row["year"]) for row in items})
    years_used = years[: max(1, int(early_window_years))]
    filtered_items = [row for row in items if int(row.get("year")) in set(years_used)]
    rebuilt = dict(dossier)
    rebuilt["items"] = filtered_items
    rebuilt["time_series"] = build_time_series(filtered_items)
    return rebuilt, years_used


def _signal_row_map(whale_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    signal_rows = {str(row.get("signal") or ""): row for row in (whale_payload.get("signals") or []) if isinstance(row, dict)}
    gap_map: dict[str, list[str]] = {}
    for gap in (whale_payload.get("gaps") or []):
        if not isinstance(gap, dict):
            continue
        signal = str(gap.get("signal") or "")
        missing = [str(metric) for metric in (gap.get("missing_metrics") or []) if str(metric).strip()]
        if signal and missing:
            gap_map[signal] = missing
    return signal_rows, gap_map


def _rate(rows: list[dict[str, Any]], *, signal: str, statuses: set[str]) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        status = str(((row.get("signals") or {}).get(signal) or {}).get("status") or "").upper()
        if status in statuses:
            hits += 1
    return round(float(hits) / float(len(rows)), 4)


def _unknown_rate(rows: list[dict[str, Any]], *, signal: str) -> float:
    if not rows:
        return 0.0
    unknown = 0
    for row in rows:
        status = str(((row.get("signals") or {}).get(signal) or {}).get("status") or "").upper()
        if status == "GAP":
            unknown += 1
    return round(float(unknown) / float(len(rows)), 4)


def _adjust_weight(*, current: float, delta: float, unknown_rate: float) -> tuple[float, str]:
    # Deterministic heuristic: raise weights for discriminative + low-unknown signals, lower for noisy signals.
    if delta >= 0.25 and unknown_rate <= 0.35:
        return min(30.0, current + 2.0), "increase_strong_signal"
    if delta >= 0.12 and unknown_rate <= 0.45:
        return min(30.0, current + 1.0), "increase_moderate_signal"
    if delta <= -0.10 or unknown_rate >= 0.65:
        return max(4.0, current - 2.0), "decrease_unreliable_signal"
    if unknown_rate >= 0.50:
        return max(4.0, current - 1.0), "decrease_high_unknown_rate"
    return current, "keep"


def run_whale_baseline(
    *,
    tickers: list[str],
    as_of_date: str,
    years_back: int,
    run_id: str,
    early_window_years: int = 5,
    controls: list[str] | None = None,
    controls_count: int = 20,
    peer_mode: str = "hybrid",
    sector: str | None = None,
    min_peers: int = 25,
    max_peers: int | None = None,
    sic_expand: bool = True,
    sic_family: bool = True,
    include_foreign: bool = True,
    include_otc: bool = False,
    min_annual_filings: int | None = None,
    sec_budget: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    winners = _normalize_tickers(tickers)
    if not winners:
        raise ValueError("at least one ticker is required")
    controls_generated, controls_meta = _generate_controls(
        winners=winners,
        controls=controls,
        controls_count=int(controls_count),
        as_of_date=as_of_date,
        years_back=int(years_back),
        peer_mode=peer_mode,
        sector=sector,
        min_peers=min_peers,
        max_peers=max_peers,
        sic_expand=sic_expand,
        sic_family=sic_family,
        include_foreign=include_foreign,
        include_otc=include_otc,
        min_annual_filings=min_annual_filings,
    )
    baseline_scope = sorted(set(winners + controls_generated))
    cohorts = {
        "winners": winners,
        "controls": controls_generated,
    }

    dossier_summary = run_dossier_for_peer_set(
        tickers=baseline_scope,
        as_of_date=as_of_date,
        years_back=int(years_back),
        run_id=run_id,
        workers=workers if workers is not None else min(4, max(1, len(baseline_scope))),
        min_annual_filings=min_annual_filings,
        sec_budget=sec_budget,
    )
    run_dir = cfg.dossiers_dir / run_id
    rows: list[dict[str, Any]] = []
    for ticker in winners + controls_generated:
        dossier_path = run_dir / ticker / "dossier.json"
        if not dossier_path.exists():
            continue
        dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
        if not isinstance(dossier, dict):
            continue
        early_dossier, years_used = _truncate_dossier_early_window(dossier, early_window_years)
        whale_payload = build_whale_signals(early_dossier)
        signals, signal_gaps = _signal_row_map(whale_payload)
        top_signals = sorted(
            [entry for entry in (whale_payload.get("signals") or []) if isinstance(entry, dict)],
            key=lambda entry: (-float(entry.get("score_contribution") or 0.0), str(entry.get("signal") or "")),
        )
        cohort = "winner" if ticker in winners else "control"
        rows.append(
            {
                "ticker": ticker,
                "cohort": cohort,
                "years_used": years_used,
                "years_used_count": len(years_used),
                "whale_signals_version": whale_payload.get("whale_signals_version"),
                "whale_signature_score": whale_payload.get("whale_signature_score"),
                "critical_unknowns": whale_payload.get("critical_unknowns") or [],
                "signals": signals,
                "signal_gaps": signal_gaps,
                "top_signals": [str(entry.get("signal")) for entry in top_signals[:3] if str(entry.get("signal") or "").strip()],
                "path": str(dossier_path),
            }
        )
    rows.sort(key=lambda row: (0 if row["cohort"] == "winner" else 1, row["ticker"]))

    winner_rows = [row for row in rows if row["cohort"] == "winner"]
    control_rows = [row for row in rows if row["cohort"] == "control"]
    signal_stats: list[dict[str, Any]] = []
    for signal in WHALE_SIGNAL_WEIGHTS.keys():
        winner_hit = _rate(winner_rows, signal=signal, statuses={"PASS", "PARTIAL"})
        control_hit = _rate(control_rows, signal=signal, statuses={"PASS", "PARTIAL"})
        delta = round(winner_hit - control_hit, 4)
        unknown_rate = _unknown_rate(rows, signal=signal)
        unknown_causes: dict[str, int] = {}
        for row in rows:
            for metric in (row.get("signal_gaps") or {}).get(signal, []):
                unknown_causes[metric] = unknown_causes.get(metric, 0) + 1
        sorted_causes = [
            {"metric": metric, "count": unknown_causes[metric]}
            for metric in sorted(unknown_causes.keys(), key=lambda m: (-unknown_causes[m], m))
        ]
        signal_stats.append(
            {
                "signal": signal,
                "winner_hit_rate": winner_hit,
                "control_hit_rate": control_hit,
                "delta": delta,
                "unknown_rate": unknown_rate,
                "unknown_metric_causes": sorted_causes,
            }
        )
    discriminative = sorted(signal_stats, key=lambda row: (-abs(float(row["delta"])), row["signal"]))
    weight_adjustments: list[dict[str, Any]] = []
    for row in signal_stats:
        signal = str(row["signal"])
        current = float(WHALE_SIGNAL_WEIGHTS.get(signal, 0.0))
        recommended, rule = _adjust_weight(current=current, delta=float(row["delta"]), unknown_rate=float(row["unknown_rate"]))
        weight_adjustments.append(
            {
                "signal": signal,
                "current_weight": current,
                "recommended_weight": round(float(recommended), 4),
                "delta": row["delta"],
                "unknown_rate": row["unknown_rate"],
                "rule": rule,
            }
        )

    out_dir = cfg.outputs_dir / "baselines" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows_path = out_dir / "baseline_rows.json"
    baseline_summary_path = out_dir / "baseline_summary.json"
    baseline_report_path = out_dir / "baseline_report.md"

    baseline_rows_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "years_back": int(years_back),
        "early_window_years": int(early_window_years),
        "cohorts": cohorts,
        "rows": rows,
    }
    baseline_rows_path.write_text(json.dumps(baseline_rows_payload, indent=2), encoding="utf-8")

    baseline_summary_payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "years_back": int(years_back),
        "early_window_years": int(early_window_years),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {
            "winners": winners,
            "controls": controls_generated,
            "counts": {"winners": len(winners), "controls": len(controls_generated), "total": len(rows)},
        },
        "controls_count_requested": int(controls_count),
        "include_foreign": bool(include_foreign),
        "include_otc": bool(include_otc),
        "min_annual_filings": int(min_annual_filings) if min_annual_filings is not None else None,
        "sec_budget_requested": int(sec_budget) if sec_budget is not None else None,
        "sec_budget_effective": (dossier_summary.get("sec_budget") or {}).get("effective"),
        "controls_generation": controls_meta,
        "signal_hit_rates": signal_stats,
        "most_discriminative_signals": discriminative,
        "weight_adjustments": weight_adjustments,
        "paths": {
            "baseline_rows_path": str(baseline_rows_path),
            "baseline_report_path": str(baseline_report_path),
        },
    }
    baseline_summary_path.write_text(json.dumps(baseline_summary_payload, indent=2), encoding="utf-8")

    lines = [
        f"# Whale Baseline Report ({run_id})",
        "",
        f"- As Of Date: `{as_of_date}`",
        f"- Winners: `{', '.join(winners)}`",
        f"- Controls: `{', '.join(controls_generated) if controls_generated else 'NONE'}`",
        f"- Controls Count Requested: `{int(controls_count)}`",
        f"- Years Back: `{int(years_back)}`",
        f"- Early Window Years: `{int(early_window_years)}`",
        "",
        "## Signal hit rates per cohort (winners vs controls)",
        "| Signal | Winner Hit Rate | Control Hit Rate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in signal_stats:
        lines.append(
            f"| {row['signal']} | {row['winner_hit_rate']:.2f} | {row['control_hit_rate']:.2f} | {row['delta']:.2f} |"
        )

    lines.append("")
    lines.append("## Most discriminative signals")
    for row in discriminative[:6]:
        lines.append(f"- **{row['signal']}**: cohort delta `{row['delta']:.2f}` (winner - control).")

    lines.append("")
    lines.append("## Unknown rates and UNKNOWN causes")
    for row in signal_stats:
        causes = ", ".join([f"{item['metric']} ({item['count']})" for item in row["unknown_metric_causes"][:5]]) or "none"
        lines.append(f"- **{row['signal']}**: unknown rate `{row['unknown_rate']:.2f}`; causes: {causes}.")

    lines.append("")
    lines.append("## Recommended default Whale Signals weight adjustments")
    lines.append("| Signal | Current | Recommended | Rule |")
    lines.append("|---|---:|---:|---|")
    for row in weight_adjustments:
        lines.append(
            f"| {row['signal']} | {row['current_weight']:.2f} | {row['recommended_weight']:.2f} | {row['rule']} |"
        )

    lines.append("")
    if not controls_generated:
        lines.append("## Controls Empty")
        lines.append(
            f"- Requested controls: `{int(controls_count)}`; generated controls: `{len(controls_generated)}`."
        )
        lines.append(f"- Peer mode: `{peer_mode}`; inferred sector: `{controls_meta.get('sector') or 'UNKNOWN'}`.")
        lines.append("- Peer selection summary:")
        lines.append("```json")
        lines.append(json.dumps(controls_meta.get("source") or {}, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("")
    lines.append("## Suppressed tickers during control selection")
    suppressed_rows = controls_meta.get("suppressed_tickers") or []
    if not suppressed_rows:
        lines.append("- None.")
    else:
        lines.append("| Ticker | Suppressor Reasons |")
        lines.append("|---|---|")
        for row in suppressed_rows:
            lines.append(f"| {row['ticker']} | {', '.join(row['reasons'])} |")

    lines.append("")
    lines.append("## Per-ticker early-window rows")
    for row in rows:
        lines.append(
            f"- **{row['ticker']}** ({row['cohort']}): score `{row['whale_signature_score']}`; "
            f"years used `{row['years_used']}`; top signals: {', '.join(row['top_signals']) or 'UNKNOWN'}."
        )
    baseline_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "tickers": baseline_scope,
        "winners": winners,
        "controls": controls_generated,
        "controls_count_requested": int(controls_count),
        "include_foreign": bool(include_foreign),
        "include_otc": bool(include_otc),
        "min_annual_filings": int(min_annual_filings) if min_annual_filings is not None else None,
        "sec_budget_requested": int(sec_budget) if sec_budget is not None else None,
        "sec_budget_effective": (dossier_summary.get("sec_budget") or {}).get("effective"),
        "controls_generation": controls_meta,
        "early_window_years": int(early_window_years),
        "dossier": dossier_summary,
        "baseline_rows_path": str(baseline_rows_path),
        "baseline_summary_path": str(baseline_summary_path),
        "baseline_report_path": str(baseline_report_path),
        "output_dir": str(out_dir),
    }
