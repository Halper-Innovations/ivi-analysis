"""Sweep roll-ups: indexed runs grouped into band × ISO-week bands.

A "sweep" is not a first-class artifact — it is the emergent shape of many
sector runs sharing a cap band inside a week. The roll-up answers: how many
sectors did that sweep cover, what did it spend, which verdicts came back,
and how many watchlist rows it produced (``watchlist.source_run_id``).
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

from app.web.readmodel.runs_index import list_indexed_runs


def _week_of(stamp: str | None) -> tuple[str, str, str] | None:
    """(label, monday_iso, sunday_iso) for an ISO timestamp or date."""
    if not stamp:
        return None
    text = str(stamp).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text).date()
    except ValueError:
        try:
            parsed = date.fromisoformat(text[:10])
        except ValueError:
            return None
    iso = parsed.isocalendar()
    monday = date.fromisocalendar(iso.year, iso.week, 1)
    return (
        f"{iso.year}-W{iso.week:02d}",
        monday.isoformat(),
        (monday + timedelta(days=6)).isoformat(),
    )


def _watchlist_counts(engine_conn: sqlite3.Connection | None) -> dict[str, int]:
    if engine_conn is None:
        return {}
    try:
        rows = engine_conn.execute(
            """
            SELECT source_run_id, COUNT(*) AS n
            FROM watchlist
            WHERE source_run_id IS NOT NULL
            GROUP BY source_run_id
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(row["source_run_id"]): int(row["n"]) for row in rows}


def sweep_rollups(
    ui_conn: sqlite3.Connection,
    *,
    engine_conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    indexed_count = int(ui_conn.execute("SELECT COUNT(*) AS n FROM run_index").fetchone()["n"])
    runs = list_indexed_runs(
        ui_conn,
        decision_eligible=True,
        limit=max(1, indexed_count),
    )
    watchlist_counts = _watchlist_counts(engine_conn)

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        week = _week_of(run["created_at"]) or _week_of(run["as_of_date"])
        if week is None:
            continue
        band = str(run["market_cap_focus"] or "unspecified")
        label, week_start, week_end = week
        group = groups.setdefault(
            (label, band),
            {
                "week": label,
                "week_start": week_start,
                "week_end": week_end,
                "band": band,
                "runs": 0,
                "sectors": set(),
                "scan_families": set(),
                "pipelines": set(),
                "verdicts": {},
                "cost_microdollars": 0,
                "cost_known_runs": 0,
                "run_ids": set(),
            },
        )
        group["runs"] += 1
        if run["sector"]:
            group["sectors"].add(str(run["sector"]))
        if run["scan_family"]:
            group["scan_families"].add(str(run["scan_family"]))
        if run["pipeline_version"]:
            group["pipelines"].add(str(run["pipeline_version"]))
        verdict = str(run["final_verdict"] or "UNKNOWN")
        group["verdicts"][verdict] = group["verdicts"].get(verdict, 0) + 1
        if isinstance(run["cost_microdollars"], int):
            group["cost_microdollars"] += run["cost_microdollars"]
            group["cost_known_runs"] += 1
        if run["run_id"]:
            group["run_ids"].add(str(run["run_id"]))

    rollups = []
    for group in groups.values():
        # Distinct run_ids: duplicated v2 smoke artifacts share an embedded
        # run_id and must not double-count the watchlist rows it produced.
        watchlist_rows = sum(watchlist_counts.get(run_id, 0) for run_id in group["run_ids"])
        cost = group["cost_microdollars"] if group["cost_known_runs"] else None
        rollups.append(
            {
                "week": group["week"],
                "week_start": group["week_start"],
                "week_end": group["week_end"],
                "band": group["band"],
                "runs": group["runs"],
                "sectors": sorted(group["sectors"]),
                "scan_families": sorted(group["scan_families"]),
                "pipelines": sorted(group["pipelines"]),
                "verdicts": dict(sorted(group["verdicts"].items())),
                "cost_microdollars": cost,
                "cost_usd": round(cost / 1_000_000, 6) if cost is not None else None,
                "cost_known_runs": group["cost_known_runs"],
                "watchlist_rows": watchlist_rows,
            }
        )
    rollups.sort(key=lambda g: (g["week_start"], g["band"]), reverse=True)
    return rollups
