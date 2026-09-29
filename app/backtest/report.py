"""Aggregate CLOSED backtest outcomes into an edge scorecard.

Segments by signal cohort (DEPLOY_READY cheap vs NOT_CHEAP) and by expectations-
gap bucket (stored in the row's status field), per horizon. Headline metric is
the edge estimate = avg_excess(cheap) - avg_excess(not-cheap).
"""
from __future__ import annotations

from statistics import mean, median
from typing import Any

from app.db import get_db


def _agg(rows: list[Any]) -> dict[str, Any]:
    excess = [float(r["excess_return_pct"]) for r in rows if r["excess_return_pct"] is not None]
    realized = [float(r["realized_return_pct"]) for r in rows if r["realized_return_pct"] is not None]
    # hit_rate/avg_excess are over n_with_excess (non-null excess), which may be < n
    return {
        "n": len(rows),
        "n_with_excess": len(excess),
        "hit_rate": round(mean(1.0 if e > 0 else 0.0 for e in excess), 4) if excess else None,
        "avg_excess": round(mean(excess), 4) if excess else None,
        "avg_realized": round(mean(realized), 4) if realized else None,
        "median_realized": round(median(realized), 4) if realized else None,
    }


def backtest_report(*, run_id_prefix: str, horizon: int) -> dict[str, Any]:
    """Edge scorecard over CLOSED backtest rows for one horizon."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT grade, status, cap_category, excess_return_pct, realized_return_pct FROM ticker_outcomes "
            "WHERE outcome_status = 'CLOSED' "
            "AND entry_price_source = 'historical_backtest' "
            "AND run_id LIKE ? AND horizon_days = ?",
            (f"{run_id_prefix}%", int(horizon)),
        ).fetchall()

    by_signal: dict[str, list] = {}
    by_bucket: dict[str, list] = {}
    by_cap: dict[str, list] = {}
    for r in rows:
        by_signal.setdefault(str(r["grade"]), []).append(r)
        by_bucket.setdefault(str(r["status"]), []).append(r)
        cap_key = str(r["cap_category"]) if r["cap_category"] is not None else "UNKNOWN_CAP"
        by_cap.setdefault(cap_key, []).append(r)

    cheap = _agg(by_signal.get("DEPLOY_READY", []))
    not_cheap = _agg(by_signal.get("NOT_CHEAP", []))
    edge = None
    if cheap["avg_excess"] is not None and not_cheap["avg_excess"] is not None:
        edge = round(cheap["avg_excess"] - not_cheap["avg_excess"], 4)

    return {
        "run_id_prefix": run_id_prefix,
        "horizon_days": int(horizon),
        "n": len(rows),
        "by_signal": {k: _agg(v) for k, v in by_signal.items()},
        "by_expectations_gap_bucket": {k: _agg(v) for k, v in by_bucket.items()},
        "by_cap_category": {k: _agg(v) for k, v in by_cap.items()},
        "edge_estimate": edge,
    }


def render_backtest_report(report: dict[str, Any]) -> str:
    """Render the scorecard as markdown for the CLI / diagnostic doc."""
    lines = [
        f"# Edge Backtest Report ({report['run_id_prefix']}, {report['horizon_days']}d)",
        "",
        f"- n (closed): {report['n']}",
        f"- **Edge estimate (cheap minus not-cheap avg excess): {report['edge_estimate']}**",
        "",
        "## By signal cohort",
    ]
    for k, v in sorted(report["by_signal"].items()):
        lines.append(f"- {k}: n={v['n']} (excess n={v['n_with_excess']}) hit_rate={v['hit_rate']} avg_excess={v['avg_excess']}")
    lines.append("")
    lines.append("## By expectations-gap bucket")
    for k, v in sorted(report["by_expectations_gap_bucket"].items()):
        lines.append(f"- {k}: n={v['n']} (excess n={v['n_with_excess']}) hit_rate={v['hit_rate']} avg_excess={v['avg_excess']}")
    lines.append("")
    lines.append("## By market-cap cohort")
    for k, v in sorted(report.get("by_cap_category", {}).items()):
        lines.append(f"- {k}: n={v['n']} (excess n={v['n_with_excess']}) hit_rate={v['hit_rate']} avg_excess={v['avg_excess']} avg_realized={v['avg_realized']}")
    return "\n".join(lines)


def _scenario_mean(realized: list[float], n_unresolved: int, imputed: float) -> float | None:
    """Mean realized with `n_unresolved` names imputed at `imputed` percent."""
    values = list(realized) + [float(imputed)] * int(n_unresolved)
    return round(mean(values), 4) if values else None


def compute_survivorship_bound(*, run_id_prefix: str, horizon: int) -> dict[str, Any]:
    """Bound the edge under three treatments of UNRESOLVED (likely-delisted) names.

    Realized-based (not excess-based): UNRESOLVED names have no benchmark return,
    so the bound imputes a realized return of {excluded, -50%, -100%} for them and
    recomputes each cohort's mean realized + the cheap-minus-not-cheap edge. This
    is the honest envelope around the survivorship gap.
    """
    with get_db() as conn:
        closed = conn.execute(
            "SELECT grade, realized_return_pct FROM ticker_outcomes "
            "WHERE outcome_status = 'CLOSED' AND entry_price_source = 'historical_backtest' "
            "AND run_id LIKE ? AND horizon_days = ? AND realized_return_pct IS NOT NULL",
            (f"{run_id_prefix}%", int(horizon)),
        ).fetchall()
        unresolved = conn.execute(
            "SELECT grade, COUNT(*) AS n FROM ticker_outcomes "
            "WHERE outcome_status = 'UNRESOLVED' AND entry_price_source = 'historical_backtest' "
            "AND run_id LIKE ? AND horizon_days = ? GROUP BY grade",
            (f"{run_id_prefix}%", int(horizon)),
        ).fetchall()

    realized_by_grade: dict[str, list[float]] = {}
    for r in closed:
        realized_by_grade.setdefault(str(r["grade"]), []).append(float(r["realized_return_pct"]))
    unresolved_by_grade = {str(r["grade"]): int(r["n"]) for r in unresolved}

    by_signal: dict[str, dict[str, Any]] = {}
    for grade in set(realized_by_grade) | set(unresolved_by_grade):
        realized = realized_by_grade.get(grade, [])
        n_unres = unresolved_by_grade.get(grade, 0)
        by_signal[grade] = {
            "n_closed": len(realized),
            "n_unresolved": n_unres,
            "avg_realized_excluded": round(mean(realized), 4) if realized else None,
            "avg_realized_minus50": _scenario_mean(realized, n_unres, -50.0),
            "avg_realized_minus100": _scenario_mean(realized, n_unres, -100.0),
        }

    def _edge(key: str) -> float | None:
        cheap = by_signal.get("DEPLOY_READY", {}).get(key)
        not_cheap = by_signal.get("NOT_CHEAP", {}).get(key)
        if cheap is None or not_cheap is None:
            return None
        return round(cheap - not_cheap, 4)

    return {
        "run_id_prefix": run_id_prefix,
        "horizon_days": int(horizon),
        "by_signal": by_signal,
        "edge_excluded": _edge("avg_realized_excluded"),
        "edge_minus50": _edge("avg_realized_minus50"),
        "edge_minus100": _edge("avg_realized_minus100"),
    }
