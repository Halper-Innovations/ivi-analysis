from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import get_config
from app.sector.taxonomy import load_sector_taxonomy
from app.valuation.lineage import latest_decision_eligible_valuation_row


def list_taxonomy_sectors(
    *,
    taxonomy_path: Path | None = None,
    overrides_path: Path | None = None,
) -> list[dict[str, Any]]:
    mapping = load_sector_taxonomy(taxonomy_path=taxonomy_path, overrides_path=overrides_path)
    counts: dict[str, int] = {}
    for _, sector in mapping.items():
        counts[sector] = counts.get(sector, 0) + 1
    return [{"sector": sector, "count": counts[sector]} for sector in sorted(counts.keys())]


def list_scannable_sectors(
    *,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    cfg = get_config()
    resolved_db_path = db_path or cfg.db_path
    try:
        conn = sqlite3.connect(str(resolved_db_path))
        conn.row_factory = sqlite3.Row
    except Exception:
        return []

    try:
        rows = conn.execute(
            """
            WITH latest_sector AS (
                SELECT
                    ticker,
                    inferred_sector,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker
                        ORDER BY as_of_date DESC, id DESC
                    ) AS row_num
                FROM sector_inference
            )
            SELECT
                ls.ticker,
                ls.inferred_sector AS sector
            FROM latest_sector ls
            WHERE ls.row_num = 1
              AND ls.inferred_sector IS NOT NULL
              AND TRIM(ls.inferred_sector) != ''
            ORDER BY ls.ticker
            """
        ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            scorecard = latest_decision_eligible_valuation_row(
                conn,
                ticker=str(row["ticker"]),
                method="scorecard",
            )
            if scorecard is None:
                continue
            sector = str(row["sector"])
            counts[sector] = counts.get(sector, 0) + 1
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    return [
        {"sector": sector, "count": count}
        for sector, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def list_sector_catalogs(
    *,
    db_path: str | Path | None = None,
    taxonomy_path: Path | None = None,
    overrides_path: Path | None = None,
) -> dict[str, Any]:
    scannable = list_scannable_sectors(db_path=db_path)
    taxonomy = list_taxonomy_sectors(taxonomy_path=taxonomy_path, overrides_path=overrides_path)
    return {
        "default_source": "scannable",
        "count": len(scannable),
        "sectors": scannable,
        "scannable": {
            "count": len(scannable),
            "sectors": scannable,
        },
        "taxonomy": {
            "count": len(taxonomy),
            "sectors": taxonomy,
        },
    }
