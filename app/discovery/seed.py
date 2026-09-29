from __future__ import annotations

import csv
from pathlib import Path

from app.universe.universe import get_active_universe_id, list_universe_members
from app.util.hashing import sha256_text


def read_seed_csv(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ticker" not in {h.strip().lower() for h in reader.fieldnames}:
            raise ValueError("Discovery seed CSV must include a ticker column")
        tickers: list[str] = []
        for row in reader:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            tickers.append(ticker)
    return tickers


def validate_seed_csv(path: Path) -> dict[str, int]:
    rows = read_seed_csv(path)
    seen: set[str] = set()
    duplicates = 0
    for ticker in rows:
        if ticker in seen:
            duplicates += 1
        seen.add(ticker)
    return {
        "rows": len(rows),
        "unique_tickers": len(seen),
        "duplicate_rows": duplicates,
    }


def read_exclude_csv(path: Path | None) -> set[str]:
    if not path:
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ticker" not in {h.strip().lower() for h in reader.fieldnames}:
            raise ValueError("Exclude CSV must include a ticker column")
        return {str(row.get("ticker") or "").strip().upper() for row in reader if str(row.get("ticker") or "").strip()}


def merge_seed_with_active_universe(conn, seed_tickers: list[str], exclude_tickers: set[str] | None = None) -> list[str]:
    exclude = {t.upper() for t in (exclude_tickers or set())}
    merged: set[str] = {t.upper() for t in seed_tickers if t.strip()}

    universe_id = get_active_universe_id(conn)
    if universe_id:
        members = list_universe_members(conn, universe_id)
        merged.update({row["ticker"].upper() for row in members if row.get("ticker")})

    if exclude:
        merged.difference_update(exclude)
    return sorted(merged)


def seed_snapshot_hash(tickers: list[str], *, include_metadata: dict[str, str] | None = None) -> str:
    payload = {
        "tickers": sorted({t.upper() for t in tickers if t.strip()}),
        "meta": include_metadata or {},
    }
    return sha256_text(str(payload))

