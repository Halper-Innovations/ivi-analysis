"""Coverage validation: detected events vs a hand-curated ground-truth CSV.

The harness measures, it does not curate. A hit is a matching event of the
right type with detection_date <= knowable_date + lag_days; coverage is
computed against detection_date (point-in-time), never detected_at, so a
2026 backfill validates a 2024 detection lag honestly.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import date, timedelta
from pathlib import Path

GROUND_TRUTH_COLUMNS = ["event_type", "cik", "ticker", "company_name", "knowable_date", "notes"]


def load_ground_truth(csv_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("cik") and not row.get("ticker"):
                continue
            rows.append(
                {
                    "event_type": str(row.get("event_type") or "").strip(),
                    "cik": str(row.get("cik") or "").strip().zfill(10),
                    "ticker": str(row.get("ticker") or "").strip().upper(),
                    "company_name": str(row.get("company_name") or "").strip(),
                    "knowable_date": str(row.get("knowable_date") or "").strip(),
                    "notes": str(row.get("notes") or "").strip(),
                }
            )
    return rows


def coverage_report(conn: sqlite3.Connection, truth: list[dict], *, lag_days: int = 7) -> dict:
    events = conn.execute(
        "SELECT cik, ticker, event_type, detection_date FROM corporate_events"
    ).fetchall()
    by_cik: dict[str, list[sqlite3.Row]] = {}
    by_ticker: dict[str, list[sqlite3.Row]] = {}
    for event in events:
        by_cik.setdefault(str(event["cik"]), []).append(event)
        if event["ticker"]:
            by_ticker.setdefault(str(event["ticker"]).upper(), []).append(event)

    total = len(truth)
    hits = 0
    by_type: dict[str, dict[str, int]] = {}
    misses: list[dict] = []
    for row in truth:
        bucket = by_type.setdefault(row["event_type"], {"total": 0, "hits": 0})
        bucket["total"] += 1
        candidates = by_cik.get(row["cik"]) or by_ticker.get(row["ticker"]) or []
        same_type = [c for c in candidates if c["event_type"] == row["event_type"]]
        if not candidates:
            misses.append({"cik": row["cik"], "event_type": row["event_type"], "reason": "NOT_DETECTED"})
            continue
        if not same_type:
            misses.append({"cik": row["cik"], "event_type": row["event_type"], "reason": "WRONG_TYPE"})
            continue
        deadline = (date.fromisoformat(row["knowable_date"]) + timedelta(days=lag_days)).isoformat()
        if any(c["detection_date"] <= deadline for c in same_type):
            hits += 1
            bucket["hits"] += 1
        else:
            misses.append({"cik": row["cik"], "event_type": row["event_type"], "reason": "DETECTED_LATE"})
    misses.sort(key=lambda m: m["cik"])
    return {
        "total": total,
        "hits": hits,
        "coverage": round(hits / total, 4) if total else 0.0,
        "by_type": by_type,
        "misses": misses,
    }
