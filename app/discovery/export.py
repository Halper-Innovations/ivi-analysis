from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db
from app.discovery.runner import export_discovery_run, load_discovery_candidates
from app.universe.universe import get_active_universe_id, list_universe_members, load_universe_to_db


def discovery_output_paths(run_id: str) -> dict[str, Path]:
    cfg = get_config()
    return {
        "candidates": cfg.discovery_dir / f"discovery_candidates_{run_id}.json",
        "shortlist": cfg.discovery_dir / f"discovery_shortlist_{run_id}.json",
        "patch": cfg.discovery_dir / f"discovery_universe_patch_{run_id}.csv",
        "report": cfg.discovery_dir / f"discovery_report_{run_id}.md",
        "stats": cfg.discovery_dir / f"discovery_stats_{run_id}.json",
    }


def export_discovery_artifacts(run_id: str, out_dir: Path) -> dict[str, str]:
    return export_discovery_run(run_id, out_dir)


def write_universe_patch(run_id: str) -> Path:
    cfg = get_config()
    candidates = load_discovery_candidates(run_id)
    path = cfg.discovery_dir / f"discovery_universe_patch_{run_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ticker",
        "cik",
        "name",
        "homepage_url",
        "ir_rss_url",
        "allowlist_domains",
        "notes",
        "discovery_score",
        "recommended_action",
        "suggested_next_pipeline",
        "run_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            if candidate.get("recommended_action") != "ADD_TO_UNIVERSE":
                continue
            writer.writerow(
                {
                    "ticker": candidate.get("ticker", ""),
                    "cik": candidate.get("cik", ""),
                    "name": candidate.get("company_name", ""),
                    "homepage_url": "",
                    "ir_rss_url": "",
                    "allowlist_domains": "",
                    "notes": f"discovery run {run_id}",
                    "discovery_score": candidate.get("discovery_score", 0),
                    "recommended_action": candidate.get("recommended_action"),
                    "suggested_next_pipeline": candidate.get("suggested_next_pipeline"),
                    "run_id": run_id,
                }
            )
    return path


def apply_discovery_run(run_id: str, *, set_active: bool = False) -> dict[str, Any]:
    patch_path = write_universe_patch(run_id)
    candidates = load_discovery_candidates(run_id)
    add_rows = [row for row in candidates if row.get("recommended_action") == "ADD_TO_UNIVERSE"]
    added_tickers = {str(row.get("ticker") or "").upper() for row in add_rows if str(row.get("ticker") or "").strip()}

    merged_snapshot: dict[str, Any] | None = None
    with get_db() as conn:
        active_id = get_active_universe_id(conn)
        base_rows: list[dict[str, str]] = []
        if active_id:
            base_rows = list_universe_members(conn, active_id)
        existing = {str(row.get("ticker") or "").upper() for row in base_rows}
        for row in add_rows:
            ticker = str(row.get("ticker") or "").upper()
            if not ticker or ticker in existing:
                continue
            base_rows.append(
                {
                    "ticker": ticker,
                    "cik": str(row.get("cik") or "").strip(),
                    "name": str(row.get("company_name") or "").strip(),
                    "ir_rss_url": "",
                    "homepage_url": "",
                    "allowlist_domains": "",
                    "notes": f"discovery run {run_id}",
                }
            )
        if base_rows and set_active:
            universe_id, snapshot_hash, ticker_count = load_universe_to_db(
                conn,
                base_rows,
                source_path=patch_path,
                set_active=True,
            )
            merged_snapshot = {
                "universe_id": universe_id,
                "snapshot_hash": snapshot_hash,
                "ticker_count": ticker_count,
            }

    return {
        "run_id": run_id,
        "patch_path": str(patch_path),
        "added_count": len(added_tickers),
        "added_tickers": sorted(added_tickers),
        "set_active": set_active,
        "merged_snapshot": merged_snapshot,
    }
