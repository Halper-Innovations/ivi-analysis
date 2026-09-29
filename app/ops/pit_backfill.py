"""PIT backfill: stamp filed_date/form/accession onto existing
companyfacts_facts rows from the cached companyfacts JSON already on the
volume, and seed the append-only vintage table.

No network: re-runs the (now provenance-carrying) normalizer over each
cached payload and stamps rows whose stored value matches the normalized
one — a mismatch means the cache and the row disagree (stale cache or a
later restatement) and is COUNTED, never guessed at. Idempotent: only
NULL/empty filed_date rows are updated, vintage inserts dedupe on
(key, filed_date, value). Resumable by construction (commit per payload).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_config
from app.logging import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cik_to_tickers(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """cik10 -> tickers present in companyfacts_facts (registry-mapped)."""
    from app.universe.ticker_cik_map import load_ticker_cik_map

    try:
        db_tickers = {
            str(row[0]).upper()
            for row in conn.execute("SELECT DISTINCT ticker FROM companyfacts_facts")
        }
    except sqlite3.Error:
        return {}
    try:
        mapping = load_ticker_cik_map(refresh_if_missing=False) or {}
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, list[str]] = {}
    for ticker, cik in mapping.items():
        ticker_norm = str(ticker).upper()
        if ticker_norm in db_tickers:
            out.setdefault(str(cik).zfill(10), []).append(ticker_norm)
    return out


def run_pit_backfill(
    *,
    limit: int | None = None,
    years_back: int = 12,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    from app.db import connect, init_db
    from app.ingest.companyfacts import (
        normalize_annual_facts_from_raw,
        normalize_quarterly_facts_from_raw,
    )

    cfg = get_config()
    cache_dir = Path(cfg.cache_dir) / "companyfacts"
    if not cache_dir.is_dir():
        raise RuntimeError(f"companyfacts cache unreachable at {cache_dir}")

    path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    conn = connect(path)
    try:
        try:
            conn.execute("SELECT filed_date FROM companyfacts_facts LIMIT 1")
        except sqlite3.Error:
            init_db(conn=conn)
        cik_tickers = _cik_to_tickers(conn)

        counts = {
            "payloads_seen": 0,
            "payloads_matched": 0,
            "facts_normalized": 0,
            "rows_stamped": 0,
            "value_mismatches": 0,
            "vintages_added": 0,
        }
        now = _utc_now_iso()
        files = sorted(cache_dir.glob("*.json"))
        if limit is not None:
            files = files[: int(limit)]

        for file_path in files:
            counts["payloads_seen"] += 1
            cik10 = file_path.stem.zfill(10)
            tickers = cik_tickers.get(cik10)
            if not tickers:
                continue
            counts["payloads_matched"] += 1
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            raw = (
                payload.get("companyfacts")
                if isinstance(payload.get("companyfacts"), dict)
                else payload
            )
            if not isinstance(raw, dict):
                continue
            try:
                facts = normalize_annual_facts_from_raw(
                    raw, cik=cik10, years_back=years_back
                ) + normalize_quarterly_facts_from_raw(
                    raw, cik=cik10, years_back=years_back
                )
            except Exception:  # noqa: BLE001
                continue

            for ticker in tickers:
                for fact in facts:
                    counts["facts_normalized"] += 1
                    filed = str(fact.get("filed_date") or "")
                    if filed:
                        cursor = conn.execute(
                            """
                            UPDATE companyfacts_facts
                            SET filed_date = ?, form = ?, accession = ?
                            WHERE ticker = ? AND fiscal_year = ? AND period_type = ?
                              AND line_item = ?
                              AND (filed_date IS NULL OR filed_date = '')
                              AND value IS NOT NULL
                              AND ABS(value - ?) < 1e-6
                            """,
                            (
                                filed,
                                fact.get("form"),
                                fact.get("accession"),
                                ticker,
                                fact["fiscal_year"],
                                fact.get("period_type", "FY"),
                                fact["line_item"],
                                float(fact["value"]),
                            ),
                        )
                        if cursor.rowcount:
                            counts["rows_stamped"] += cursor.rowcount
                        else:
                            row = conn.execute(
                                "SELECT value, filed_date FROM companyfacts_facts "
                                "WHERE ticker=? AND fiscal_year=? AND period_type=? AND line_item=?",
                                (
                                    ticker,
                                    fact["fiscal_year"],
                                    fact.get("period_type", "FY"),
                                    fact["line_item"],
                                ),
                            ).fetchone()
                            if (
                                row is not None
                                and not (row["filed_date"] or "")
                                and row["value"] is not None
                                and abs(float(row["value"]) - float(fact["value"])) >= 1e-6
                            ):
                                counts["value_mismatches"] += 1
                    cursor = conn.execute(
                        """INSERT OR IGNORE INTO companyfacts_vintages
                           (ticker, fiscal_year, period_type, period_end, line_item,
                            value, units, filed_date, form, accession, recorded_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ticker,
                            fact["fiscal_year"],
                            fact.get("period_type", "FY"),
                            fact.get("period_end"),
                            fact["line_item"],
                            fact.get("value"),
                            fact.get("units"),
                            filed,
                            fact.get("form"),
                            fact.get("accession"),
                            now,
                        ),
                    )
                    if cursor.rowcount:
                        counts["vintages_added"] += cursor.rowcount
            conn.commit()
            if counts["payloads_matched"] % 200 == 0:
                logger.info(
                    "pit_backfill progress: %(payloads_matched)d payloads, "
                    "%(rows_stamped)d rows stamped",
                    counts,
                )
        return counts
    finally:
        conn.close()


__all__ = ["run_pit_backfill"]
