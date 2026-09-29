from __future__ import annotations

import json
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.util.hashing import sha256_file


def build_shortlist(
    *,
    run_id: str,
    top_n: int,
    min_score: float,
    require_recent_research: bool,
) -> dict[str, Any]:
    cfg = get_config()
    cfg.shortlists_dir.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        score_rows = conn.execute(
            """
            SELECT ticker, as_of_date, total_score, decision, subscores_json, reasons_json
            FROM scores
            WHERE run_id = ? AND total_score >= ?
            ORDER BY total_score DESC, ticker ASC
            """,
            (run_id, float(min_score)),
        ).fetchall()

        shortlist_rows: list[dict[str, Any]] = []
        for row in score_rows:
            ticker = row["ticker"]
            signal_row = conn.execute(
                """
                SELECT recency_days_min, item_count_30d, summary_json
                FROM research_signals
                WHERE ticker = ? AND run_id = ? AND as_of_date <= ?
                ORDER BY as_of_date DESC, created_at DESC
                LIMIT 1
                """,
                (ticker, run_id, row["as_of_date"]),
            ).fetchone()
            recency = signal_row["recency_days_min"] if signal_row else None
            if require_recent_research and (recency is None or int(recency) > 180):
                continue

            delta_row = conn.execute(
                """
                SELECT changed, delta_path
                FROM ticker_deltas
                WHERE ticker = ? AND run_id = ?
                LIMIT 1
                """,
                (ticker, run_id),
            ).fetchone()
            delta_changed = bool(delta_row["changed"]) if delta_row else False
            delta_path = delta_row["delta_path"] if delta_row else None

            shortlist_rows.append(
                {
                    "ticker": ticker,
                    "as_of_date": row["as_of_date"],
                    "total_score": float(row["total_score"]),
                    "decision": row["decision"],
                    "delta_changed": delta_changed,
                    "delta_path": delta_path,
                    "research_recency_days_min": recency,
                    "research_item_count_30d": int(signal_row["item_count_30d"] or 0) if signal_row else 0,
                    "subscores": json.loads(row["subscores_json"] or "{}"),
                    "reasons": json.loads(row["reasons_json"] or "[]"),
                }
            )

        shortlist_rows.sort(key=lambda r: (not r["delta_changed"], -float(r["total_score"]), r["ticker"]))
        shortlist_rows = shortlist_rows[: max(1, int(top_n))]

    payload = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "top_n": int(top_n),
        "min_score": float(min_score),
        "require_recent_research": bool(require_recent_research),
        "rows": shortlist_rows,
    }
    json_path = cfg.shortlists_dir / f"shortlist_{run_id}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [f"# Alpha Shortlist ({run_id})", ""]
    for row in shortlist_rows:
        lines.append(
            f"- {row['ticker']}: score={row['total_score']} decision={row['decision']} "
            f"changed={row['delta_changed']} research_recency_days={row['research_recency_days_min']}"
        )
    md_path = cfg.shortlists_dir / f"shortlist_{run_id}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    shortlist_hash = sha256_file(json_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO shortlists(run_id, shortlist_path, shortlist_hash, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                shortlist_path=excluded.shortlist_path,
                shortlist_hash=excluded.shortlist_hash,
                created_at=excluded.created_at
            """,
            (run_id, str(json_path), shortlist_hash, utc_now_iso()),
        )

    return {
        "run_id": run_id,
        "count": len(shortlist_rows),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
