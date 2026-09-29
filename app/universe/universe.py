from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.db import utc_now_iso
from app.util.hashing import sha256_text


REQUIRED_COLUMNS = {"ticker"}
REQUIRED_BASE_COLUMNS = {"ticker", "cik"}
OVERRIDES_COLUMNS = {"ticker"}


def read_universe_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Universe CSV missing columns: {sorted(missing)}")
        rows: list[dict[str, str]] = []
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            cik = (row.get("cik") or "").strip().lstrip("0")
            name = (row.get("name") or "").strip()
            ir_rss_url = (row.get("ir_rss_url") or "").strip()
            homepage_url = (row.get("homepage_url") or "").strip()
            allowlist_domains = (row.get("allowlist_domains") or "").strip()
            notes = (row.get("notes") or "").strip()
            if not ticker:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "name": name,
                    "ir_rss_url": ir_rss_url,
                    "homepage_url": homepage_url,
                    "allowlist_domains": allowlist_domains,
                    "notes": notes,
                }
            )
        return rows


def read_base_universe_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_BASE_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Base universe CSV missing columns: {sorted(missing)}")
        rows: list[dict[str, str]] = []
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            cik = (row.get("cik") or "").strip().lstrip("0")
            if not ticker or not cik:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "name": (row.get("name") or "").strip(),
                    "ir_rss_url": "",
                    "homepage_url": "",
                    "allowlist_domains": "",
                    "notes": "",
                }
            )
        return rows


def read_metadata_overrides_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = OVERRIDES_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata overrides CSV missing columns: {sorted(missing)}")
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            out[ticker] = {
                "name": (row.get("name") or "").strip(),
                "homepage_url": (row.get("homepage_url") or "").strip(),
                "ir_rss_url": (row.get("ir_rss_url") or "").strip(),
                "allowlist_domains": (row.get("allowlist_domains") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
        return out


def merge_universe_rows(base_rows: list[dict[str, str]], overrides: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for base in base_rows:
        ticker = base["ticker"]
        patch = overrides.get(ticker, {})
        merged.append(
            {
                "ticker": ticker,
                "cik": base.get("cik", ""),
                "name": patch.get("name") or base.get("name", ""),
                "ir_rss_url": patch.get("ir_rss_url") or base.get("ir_rss_url", ""),
                "homepage_url": patch.get("homepage_url") or base.get("homepage_url", ""),
                "allowlist_domains": patch.get("allowlist_domains") or base.get("allowlist_domains", ""),
                "notes": patch.get("notes") or base.get("notes", ""),
            }
        )
    return merged


def fill_missing_ciks(rows: list[dict[str, str]], ticker_map: dict[str, str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        cik = row.get("cik") or ticker_map.get(row["ticker"], "")
        if not cik:
            continue
        out.append(
            {
                "ticker": row["ticker"],
                "cik": cik,
                "name": row.get("name", ""),
                "ir_rss_url": row.get("ir_rss_url", ""),
                "homepage_url": row.get("homepage_url", ""),
                "allowlist_domains": row.get("allowlist_domains", ""),
                "notes": row.get("notes", ""),
            }
        )
    return out


def _canonical_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        ticker = row["ticker"].strip().upper()
        if not ticker:
            continue
        seen[ticker] = {
            "ticker": ticker,
            "cik": (row.get("cik") or "").strip().lstrip("0"),
            "name": (row.get("name") or "").strip(),
            "ir_rss_url": (row.get("ir_rss_url") or "").strip(),
            "homepage_url": (row.get("homepage_url") or "").strip(),
            "allowlist_domains": (row.get("allowlist_domains") or "").strip(),
            "notes": (row.get("notes") or "").strip(),
        }
    return list(seen.values())


def compute_universe_snapshot_hash(rows: list[dict[str, str]]) -> str:
    canonical = _canonical_rows(rows)
    return sha256_text(json.dumps(canonical, separators=(",", ":"), sort_keys=True))


def persist_universe_snapshot(
    conn,
    rows: list[dict[str, str]],
    source_path: Path,
    *,
    set_active: bool = True,
) -> tuple[str, str, int]:
    canonical = _canonical_rows(rows)
    snapshot_hash = compute_universe_snapshot_hash(canonical)
    universe_id = f"u_{snapshot_hash[:16]}"
    now = utc_now_iso()

    conn.execute(
        """
        INSERT INTO universe_snapshots(universe_id, snapshot_hash, source_path, ticker_count, snapshot_json, created_at)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(universe_id) DO UPDATE SET
            snapshot_hash=excluded.snapshot_hash,
            source_path=excluded.source_path,
            ticker_count=excluded.ticker_count,
            snapshot_json=excluded.snapshot_json
        """,
        (
            universe_id,
            snapshot_hash,
            str(source_path),
            len(canonical),
            json.dumps(canonical),
            now,
        ),
    )

    conn.execute("DELETE FROM universe_members WHERE universe_id = ?", (universe_id,))
    for row in canonical:
        conn.execute(
            """
            INSERT INTO universe_members(
                universe_id, ticker, cik, name, ir_rss_url, homepage_url, allowlist_domains, notes, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                universe_id,
                row["ticker"],
                row["cik"],
                row.get("name", ""),
                row.get("ir_rss_url", ""),
                row.get("homepage_url", ""),
                row.get("allowlist_domains", ""),
                row.get("notes", ""),
                now,
            ),
        )

    for row in canonical:
        conn.execute(
            """
            INSERT INTO companies(
                ticker, cik, name, ir_rss_url, homepage_url, allowlist_domains, notes, universe_id, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                cik=excluded.cik,
                name=excluded.name,
                ir_rss_url=excluded.ir_rss_url,
                homepage_url=excluded.homepage_url,
                allowlist_domains=excluded.allowlist_domains,
                notes=excluded.notes,
                universe_id=excluded.universe_id
            """,
            (
                row["ticker"],
                row["cik"],
                row.get("name", ""),
                row.get("ir_rss_url", ""),
                row.get("homepage_url", ""),
                row.get("allowlist_domains", ""),
                row.get("notes", ""),
                universe_id,
                now,
            ),
        )

    if set_active:
        conn.execute(
            """
            INSERT INTO state(key, value_json, updated_at)
            VALUES('active_universe', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            (json.dumps({"universe_id": universe_id, "snapshot_hash": snapshot_hash}), now),
        )

    return universe_id, snapshot_hash, len(canonical)


def load_universe_to_db(
    conn,
    rows: list[dict[str, str]],
    source_path: Path,
    *,
    set_active: bool = True,
) -> tuple[str, str, int]:
    if not rows:
        raise ValueError("No valid universe rows to load")
    return persist_universe_snapshot(conn, rows, source_path, set_active=set_active)


def list_universe_snapshots(conn, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT universe_id, snapshot_hash, source_path, ticker_count, created_at
        FROM universe_snapshots
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "universe_id": row["universe_id"],
            "snapshot_hash": row["snapshot_hash"],
            "source_path": row["source_path"],
            "ticker_count": row["ticker_count"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def list_universe_members(conn, universe_id: str) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT ticker, cik, name, ir_rss_url, homepage_url, allowlist_domains, notes
        FROM universe_members
        WHERE universe_id = ?
        ORDER BY id ASC
        """,
        (universe_id,),
    ).fetchall()
    return [
        {
            "ticker": row["ticker"],
            "cik": row["cik"],
            "name": row["name"] or "",
            "ir_rss_url": row["ir_rss_url"] or "",
            "homepage_url": row["homepage_url"] or "",
            "allowlist_domains": row["allowlist_domains"] or "",
            "notes": row["notes"] or "",
        }
        for row in rows
    ]


def export_universe_snapshot(conn, universe_id: str, out_path: Path) -> Path:
    rows = list_universe_members(conn, universe_id)
    if not rows:
        raise ValueError(f"No members found for universe_id={universe_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ticker", "cik", "name", "homepage_url", "ir_rss_url", "allowlist_domains", "notes"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out_path


def get_active_universe_id(conn) -> str | None:
    row = conn.execute("SELECT value_json FROM state WHERE key = 'active_universe'").fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row["value_json"])
    except Exception:
        return None
    return payload.get("universe_id")
