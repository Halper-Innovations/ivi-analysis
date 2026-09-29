from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from app.config import get_config


def _read_taxonomy_csv(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = str(row.get("ticker") or "").strip().upper()
            sector = str(row.get("sector") or "").strip()
            if not ticker or not sector:
                continue
            mapping[ticker] = sector
    return mapping


def load_sector_taxonomy(
    *,
    taxonomy_path: Path | None = None,
    overrides_path: Path | None = None,
) -> dict[str, str]:
    cfg = get_config()
    taxonomy = _read_taxonomy_csv(taxonomy_path or cfg.sector_taxonomy_path)
    override_file = overrides_path or cfg.sector_overrides_path
    overrides = _read_taxonomy_csv(override_file)
    taxonomy.update(overrides)
    return {ticker: taxonomy[ticker] for ticker in sorted(taxonomy.keys())}


def list_sectors(
    *,
    taxonomy_path: Path | None = None,
    overrides_path: Path | None = None,
) -> list[dict[str, Any]]:
    mapping = load_sector_taxonomy(taxonomy_path=taxonomy_path, overrides_path=overrides_path)
    counts: dict[str, int] = {}
    for _, sector in mapping.items():
        counts[sector] = counts.get(sector, 0) + 1
    rows = [{"sector": sector, "count": counts[sector]} for sector in sorted(counts.keys())]
    return rows

