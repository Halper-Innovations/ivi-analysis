"""Coverage atlas: the ``sector_run_loaded_sets`` ledger as sector × band cells.

This is ``ivi sweep-delta-report`` made visual, so the swept-ness semantics
must be the delta report's, not a reinterpretation:

* probe runs (``source = 'explicit_tickers'``) never count as coverage;
* v1 rows count only when the ticker's latest state in the cell is an
  explicit coverage-complete disposition; legacy loader-only rows do not count;
* v2 rows count only when the ticker's **latest** state in the cell is
  ``coverage_complete`` — a newer NEEDS_DATA reopens the name.

For v2 cells the swept set comes from the literal production helper
:func:`app.autonomous.sweep_delta.swept_tickers_for_band` (sector-scoped),
so the atlas cannot drift from the delta report; the v1 predicate is the
same SQL the helper runs, restricted to the cell's sector.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.autonomous.artifact_financial_audit import (
    financial_integrity_manifest_is_usable,
    run_id_is_decision_eligible,
)
from app.autonomous.sweep_delta import CANONICAL_SWEEP_SECTORS, swept_tickers_for_band

# Canonical presentation order for cap bands; observed extras append after.
BAND_ORDER = ("micro_cap", "small_cap", "smid_cap", "mid_cap", "large_and_mega")


def _age_days(loaded_at: str | None, *, now: datetime | None = None) -> float | None:
    if not loaded_at:
        return None
    try:
        stamp = datetime.fromisoformat(str(loaded_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    effective_now = now or datetime.now(timezone.utc)
    return round((effective_now - stamp).total_seconds() / 86400, 1)


def _ordered_bands(observed: set[str]) -> list[str]:
    ordered = [band for band in BAND_ORDER if band in observed]
    ordered.extend(sorted(observed - set(BAND_ORDER)))
    return ordered


def _ordered_sectors(observed: set[str]) -> list[str]:
    ordered = list(CANONICAL_SWEEP_SECTORS)
    ordered.extend(sorted(observed - set(CANONICAL_SWEEP_SECTORS)))
    return ordered


def _pipeline_cell_meta(
    conn: sqlite3.Connection, *, pipeline_version: str
) -> list[sqlite3.Row]:
    pipeline_clause = (
        "LOWER(COALESCE(pipeline_version, 'v1')) != 'v2'"
        if pipeline_version == "v1"
        else "pipeline_version = 'v2'"
    )
    rows = conn.execute(
        f"""
        SELECT DISTINCT sector, market_cap_focus AS band
        FROM sector_run_loaded_sets
        WHERE source != 'explicit_tickers'
          AND {pipeline_clause}
        """
    ).fetchall()
    return rows


def _coverage_run_rows(
    conn: sqlite3.Connection,
    *,
    sector: str,
    band: str,
    v1_tickers: set[str],
    v2_tickers: set[str],
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[Any] = [sector, band]
    if v1_tickers:
        marks = ",".join("?" for _ in v1_tickers)
        clauses.append(
            "(LOWER(COALESCE(pipeline_version, 'v1')) != 'v2' "
            "AND coverage_complete = 1 AND candidate_disposition IS NOT NULL "
            f"AND ticker IN ({marks}))"
        )
        params.extend(sorted(v1_tickers))
    if v2_tickers:
        marks = ",".join("?" for _ in v2_tickers)
        clauses.append(
            "(pipeline_version = 'v2' AND coverage_complete = 1 "
            f"AND ticker IN ({marks}))"
        )
        params.extend(sorted(v2_tickers))
    if not clauses:
        return []
    rows = conn.execute(
        f"""
        SELECT run_id, source, pipeline_version,
               MAX(loaded_at) AS loaded_at,
               COUNT(DISTINCT ticker) AS tickers
        FROM sector_run_loaded_sets
        WHERE LOWER(sector) = ? AND market_cap_focus = ?
          AND source != 'explicit_tickers'
          AND ({' OR '.join(clauses)})
        GROUP BY run_id, source, pipeline_version
        ORDER BY loaded_at DESC
        """,
        params,
    ).fetchall()
    return [
        row
        for row in rows
        if run_id_is_decision_eligible(row["run_id"])
    ]


def coverage_atlas(
    conn: sqlite3.Connection, *, now: datetime | None = None
) -> dict[str, Any]:
    """The full grid: canonical sectors × observed bands, cells where swept."""
    if not financial_integrity_manifest_is_usable():
        return {"sectors": list(CANONICAL_SWEEP_SECTORS), "bands": [], "cells": []}
    try:
        v1_rows = _pipeline_cell_meta(conn, pipeline_version="v1")
        v2_rows = _pipeline_cell_meta(conn, pipeline_version="v2")
    except sqlite3.OperationalError:
        # A books-of-record DB without the ledger table: an empty atlas is
        # the honest answer (nothing has ever been swept into it).
        return {"sectors": list(CANONICAL_SWEEP_SECTORS), "bands": [], "cells": []}

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for row in v1_rows:
        key = (str(row["sector"]), str(row["band"]))
        swept = swept_tickers_for_band(conn, key[1], key[0], pipeline_version="v1")
        if not swept:
            continue
        run_rows = _coverage_run_rows(
            conn,
            sector=key[0],
            band=key[1],
            v1_tickers=swept,
            v2_tickers=set(),
        )
        cells[key] = {
            "sector": key[0],
            "band": key[1],
            "tickers": len(swept),
            "last_loaded_at": max(
                (str(item["loaded_at"]) for item in run_rows if item["loaded_at"]),
                default=None,
            ),
            "run_count": len(run_rows),
            "pipelines": ["v1"],
            "_v1_tickers": swept,
        }
    for row in v2_rows:
        key = (str(row["sector"]), str(row["band"]))
        swept = swept_tickers_for_band(conn, key[1], key[0], pipeline_version="v2")
        if not swept:
            continue
        cell = cells.get(key)
        if cell is None:
            run_rows = _coverage_run_rows(
                conn,
                sector=key[0],
                band=key[1],
                v1_tickers=set(),
                v2_tickers=swept,
            )
            cells[key] = {
                "sector": key[0],
                "band": key[1],
                "tickers": len(swept),
                "last_loaded_at": max(
                    (
                        str(item["loaded_at"])
                        for item in run_rows
                        if item["loaded_at"]
                    ),
                    default=None,
                ),
                "run_count": len(run_rows),
                "pipelines": ["v2"],
                "_v1_tickers": set(),
            }
            continue
        v1_tickers = set(cell.pop("_v1_tickers", set()))
        run_rows = _coverage_run_rows(
            conn,
            sector=key[0],
            band=key[1],
            v1_tickers=v1_tickers,
            v2_tickers=swept,
        )
        cell["tickers"] = len(v1_tickers | swept)
        cell["last_loaded_at"] = max(
            (str(item["loaded_at"]) for item in run_rows if item["loaded_at"]),
            default=None,
        )
        cell["run_count"] = len(run_rows)
        cell["pipelines"].append("v2")

    observed_bands = {band for _, band in cells}
    observed_sectors = {sector for sector, _ in cells}
    cell_list = []
    for cell in cells.values():
        cell.pop("_v1_tickers", None)
        cell["age_days"] = _age_days(cell["last_loaded_at"], now=now)
        cell_list.append(cell)
    cell_list.sort(key=lambda c: (c["sector"], c["band"]))
    return {
        "sectors": _ordered_sectors(observed_sectors),
        "bands": _ordered_bands(observed_bands),
        "cells": cell_list,
    }


def coverage_cell(
    conn: sqlite3.Connection,
    *,
    sector: str,
    band: str,
    ui_conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """One cell opened: swept tickers plus the runs that loaded them."""
    sector = str(sector).strip().lower()
    band = str(band).strip().lower()
    if not financial_integrity_manifest_is_usable():
        return {"sector": sector, "band": band, "tickers": [], "runs": []}
    try:
        v1_tickers = swept_tickers_for_band(
            conn, band, sector, pipeline_version="v1"
        )
        v2_tickers = swept_tickers_for_band(
            conn, band, sector, pipeline_version="v2"
        )
        run_rows = _coverage_run_rows(
            conn,
            sector=sector,
            band=band,
            v1_tickers=v1_tickers,
            v2_tickers=v2_tickers,
        )
    except sqlite3.OperationalError:
        run_rows = []
        v1_tickers = set()
        v2_tickers = set()
    swept = v1_tickers | v2_tickers

    slugs: dict[str, str] = {}
    run_ids = [str(row["run_id"]) for row in run_rows]
    if ui_conn is not None and run_ids:
        marks = ",".join("?" for _ in run_ids)
        try:
            for row in ui_conn.execute(
                f"SELECT run_id, slug, COUNT(*) AS n FROM run_index"
                f" WHERE run_id IN ({marks}) GROUP BY run_id",
                run_ids,
            ):
                if int(row["n"]) == 1 and row["slug"]:
                    slugs[str(row["run_id"])] = str(row["slug"])
        except sqlite3.OperationalError:
            slugs = {}

    return {
        "sector": sector,
        "band": band,
        "tickers": sorted(swept),
        "runs": [
            {
                "run_id": str(row["run_id"]),
                "slug": slugs.get(str(row["run_id"])),
                "source": str(row["source"]),
                "pipeline_version": row["pipeline_version"],
                "loaded_at": row["loaded_at"],
                "tickers": int(row["tickers"]),
            }
            for row in run_rows
        ],
    }
