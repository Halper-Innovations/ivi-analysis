from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.util.credential_hygiene import sanitize_url_credentials
from app.util.hashing import sha256_file, sha256_text


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        return out
    except Exception:
        return "UNKNOWN"


def _config_hash() -> str:
    cfg = get_config()
    payload = cfg.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)
    return sha256_text(serialized)


def _active_universe(conn) -> dict[str, Any]:
    state_row = conn.execute("SELECT value_json FROM state WHERE key = 'active_universe'").fetchone()
    if not state_row:
        return {"universe_id": None, "snapshot_hash": None, "ticker_count": 0}
    payload = json.loads(state_row["value_json"])
    universe_id = payload.get("universe_id")
    if not universe_id:
        return {"universe_id": None, "snapshot_hash": None, "ticker_count": 0}
    row = conn.execute(
        "SELECT snapshot_hash, ticker_count FROM universe_snapshots WHERE universe_id = ?",
        (universe_id,),
    ).fetchone()
    if not row:
        return {"universe_id": universe_id, "snapshot_hash": payload.get("snapshot_hash"), "ticker_count": 0}
    return {
        "universe_id": universe_id,
        "snapshot_hash": row["snapshot_hash"],
        "ticker_count": int(row["ticker_count"]),
    }


def _filings_accessions(conn, as_of_date: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT accession
        FROM filings
        WHERE COALESCE(filing_date, '1900-01-01') <= ?
          AND status IN ('downloaded', 'parsed')
        ORDER BY accession ASC
        """,
        (as_of_date,),
    ).fetchall()
    return [row["accession"] for row in rows]


def _price_sources(conn, as_of_date: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ticker, provider, as_of_date, status, source_url, fetched_at, expires_at, price
        FROM price_quotes
        WHERE as_of_date = ?
        ORDER BY ticker, provider
        """,
        (as_of_date,),
    ).fetchall()
    return [
        {
            "ticker": row["ticker"],
            "provider": row["provider"],
            "as_of_date": row["as_of_date"],
            "status": row["status"],
            "price": row["price"],
            "source_url": sanitize_url_credentials(row["source_url"]),
            "fetched_at": row["fetched_at"],
            "expires_at": row["expires_at"],
        }
        for row in rows
    ]


def write_run_manifest(as_of_date: str | None = None, run_id: str | None = None) -> Path:
    cfg = get_config()
    as_of_date = as_of_date or datetime.now(timezone.utc).date().isoformat()
    run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"

    with get_db() as conn:
        universe = _active_universe(conn)
        manifest = {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "as_of_date": as_of_date,
            "universe": universe,
            "universe_id": universe.get("universe_id"),
            "snapshot_hash": universe.get("snapshot_hash"),
            "filings_accessions": _filings_accessions(conn, as_of_date),
            "config_hash": _config_hash(),
            "git_commit": _git_commit(),
            "price_sources": _price_sources(conn, as_of_date),
        }

    cfg.manifests_dir.mkdir(parents=True, exist_ok=True)
    dated_path = cfg.manifests_dir / f"run_manifest_{as_of_date}.json"
    dated_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cfg.run_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest_hash = sha256_file(dated_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO run_manifests(run_id, as_of_date, manifest_path, manifest_hash, created_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                as_of_date=excluded.as_of_date,
                manifest_path=excluded.manifest_path,
                manifest_hash=excluded.manifest_hash
            """,
            (run_id, as_of_date, str(dated_path), manifest_hash, utc_now_iso()),
        )
    return dated_path
