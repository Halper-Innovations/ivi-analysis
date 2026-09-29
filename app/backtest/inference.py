"""Quarter-clustered block bootstrap for backtest cohort contrasts.

Outcomes from the same as-of date share a market regime, so naive SEs over
pooled rows overstate precision. This resamples whole as_of_date clusters
with replacement (block bootstrap) to get an honest-ish SE/CI for the
treat-minus-control contrast. Stdlib only — sqlite3, random, statistics.
"""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from app.db import get_db

_LOAD_QUERY = (
    "SELECT ticker, as_of_date, grade, status, excess_return_pct, realized_return_pct "
    "FROM ticker_outcomes "
    "WHERE run_id = ? AND outcome_status = 'CLOSED' AND excess_return_pct IS NOT NULL "
    "ORDER BY as_of_date, ticker"
)


def load_closed_outcomes(run_id: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """CLOSED rows with non-null excess for one run_id, as plain dicts."""
    if db_path is not None:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(_LOAD_QUERY, (run_id,)).fetchall()
        finally:
            conn.close()
    else:
        with get_db() as conn:
            rows = conn.execute(_LOAD_QUERY, (run_id,)).fetchall()
    return [dict(r) for r in rows]


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile over a pre-sorted list (numpy 'linear')."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _ci95(samples: list[float]) -> tuple[float | None, float | None]:
    if not samples:
        return (None, None)
    s = sorted(samples)
    return (round(_percentile(s, 0.025), 4), round(_percentile(s, 0.975), 4))


def cluster_bootstrap_contrast(
    rows: list[dict[str, Any]],
    *,
    group_field: str = "grade",
    treat: str = "DEPLOY_READY",
    control: str = "NOT_CHEAP",
    value_field: str = "excess_return_pct",
    n_boot: int = 10_000,
    seed: int = 1337,
) -> dict[str, Any]:
    """Block bootstrap: resample as_of_date clusters with replacement.

    Each draw samples n_clusters labels with replacement; a cluster sampled
    twice contributes its rows twice. Draws where either cohort ends up empty
    are skipped (counted in skipped_draws). Deterministic for a given seed.
    """
    treat_by_cluster: dict[str, list[float]] = {}
    control_by_cluster: dict[str, list[float]] = {}
    for r in rows:
        if r.get(value_field) is None:
            continue
        group = r.get(group_field)
        if group == treat:
            treat_by_cluster.setdefault(str(r["as_of_date"]), []).append(float(r[value_field]))
        elif group == control:
            control_by_cluster.setdefault(str(r["as_of_date"]), []).append(float(r[value_field]))

    clusters = sorted(set(treat_by_cluster) | set(control_by_cluster))
    for label in clusters:
        treat_by_cluster.setdefault(label, [])
        control_by_cluster.setdefault(label, [])
    treat_all = [v for label in clusters for v in treat_by_cluster[label]]
    control_all = [v for label in clusters for v in control_by_cluster[label]]
    if not treat_all:
        raise ValueError(f"no rows with non-null {value_field!r} for treat cohort {treat!r}")
    if not control_all:
        raise ValueError(f"no rows with non-null {value_field!r} for control cohort {control!r}")

    per_cluster: dict[str, float | None] = {}
    for label in clusters:
        t, c = treat_by_cluster[label], control_by_cluster[label]
        per_cluster[label] = round(mean(t) - mean(c), 4) if t and c else None

    leave_one_out: dict[str, float | None] = {}
    for excluded in clusters:
        t = [v for label in clusters if label != excluded for v in treat_by_cluster[label]]
        c = [v for label in clusters if label != excluded for v in control_by_cluster[label]]
        leave_one_out[excluded] = round(mean(t) - mean(c), 4) if t and c else None

    rng = random.Random(seed)
    n_clusters = len(clusters)
    boot_means: list[float] = []
    boot_medians: list[float] = []
    skipped = 0
    for _ in range(int(n_boot)):
        sampled = [rng.choice(clusters) for _ in range(n_clusters)]
        t_vals: list[float] = []
        c_vals: list[float] = []
        for label in sampled:
            t_vals.extend(treat_by_cluster[label])
            c_vals.extend(control_by_cluster[label])
        if not t_vals or not c_vals:
            skipped += 1
            continue
        boot_means.append(mean(t_vals) - mean(c_vals))
        boot_medians.append(median(t_vals) - median(c_vals))

    return {
        "observed_mean_contrast": round(mean(treat_all) - mean(control_all), 4),
        "observed_median_contrast": round(median(treat_all) - median(control_all), 4),
        "boot_se_mean": round(stdev(boot_means), 4) if len(boot_means) >= 2 else None,
        "boot_se_median": round(stdev(boot_medians), 4) if len(boot_medians) >= 2 else None,
        "ci95_mean": _ci95(boot_means),
        "ci95_median": _ci95(boot_medians),
        "n_treat": len(treat_all),
        "n_control": len(control_all),
        "n_clusters": n_clusters,
        "per_cluster_contrasts": per_cluster,
        "leave_one_out_contrasts": leave_one_out,
        "n_boot_effective": len(boot_means),
        "skipped_draws": skipped,
    }


FEW_CLUSTER_WARNING = (
    "WARNING: only {n} time clusters (as-of dates) — bootstrap SE/CI are "
    "unreliable with so few clusters; read them as a LOWER BOUND on uncertainty."
)


def format_bootstrap_report(
    result: dict[str, Any],
    run_id: str,
    *,
    group_field: str = "grade",
    treat: str = "DEPLOY_READY",
    control: str = "NOT_CHEAP",
    value_field: str = "excess_return_pct",
) -> str:
    """Render the bootstrap result as a compact markdown block."""
    ci_m = result["ci95_mean"]
    ci_md = result["ci95_median"]
    lines = [
        f"# Cluster Bootstrap Contrast ({run_id})",
        "",
        f"- contrast: {treat} minus {control} on {value_field} (grouped by {group_field})",
        f"- n_treat={result['n_treat']} n_control={result['n_control']} "
        f"n_clusters={result['n_clusters']} (clustered by as_of_date)",
        f"- observed mean contrast: {result['observed_mean_contrast']}",
        f"- observed median contrast: {result['observed_median_contrast']}",
        f"- bootstrap draws: {result['n_boot_effective']} effective "
        f"({result['skipped_draws']} skipped — cohort empty in draw)",
        f"- SE(mean)={result['boot_se_mean']}  95% CI(mean)=[{ci_m[0]}, {ci_m[1]}]",
        f"- SE(median)={result['boot_se_median']}  95% CI(median)=[{ci_md[0]}, {ci_md[1]}]",
    ]
    if result["n_clusters"] < 5:
        lines.append(f"- {FEW_CLUSTER_WARNING.format(n=result['n_clusters'])}")
    lines.append("")
    lines.append("## Per-cluster mean contrasts")
    for label, value in sorted(result["per_cluster_contrasts"].items()):
        lines.append(f"- {label}: {value}")
    lines.append("")
    lines.append("## Leave-one-out mean contrasts (excluded date -> contrast)")
    for label, value in sorted(result["leave_one_out_contrasts"].items()):
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)
