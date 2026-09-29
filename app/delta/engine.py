from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.ops.runs import list_runs
from app.util.hashing import sha256_file


def _resolve_prev_run_id(run_id: str, explicit_prev_run_id: str | None) -> str | None:
    if explicit_prev_run_id:
        return explicit_prev_run_id
    runs = list_runs(limit=200)
    for idx, run in enumerate(runs):
        if run.get("run_id") != run_id:
            continue
        if idx + 1 < len(runs):
            return runs[idx + 1].get("run_id")
        return None
    return None


def _line_item_citations(packet: dict[str, Any], line_item: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in packet.get("financials", []):
        if row.get("line_item") != line_item:
            continue
        citation = row.get("citation", {})
        source_url = citation.get("source_url")
        if not source_url:
            continue
        out.append(
            {
                "source_url": source_url,
                "snippet": citation.get("snippet", ""),
                "section_label": citation.get("section_label"),
            }
        )
        if len(out) >= 3:
            break
    return out


def _load_packet_for_ticker_as_of(conn, ticker: str, as_of_date: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT packet_path
        FROM evidence_packets
        WHERE ticker = ? AND as_of_date = ?
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if not row:
        return {}
    path = Path(row["packet_path"])
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fundamentals_for_ticker(conn, ticker: str, as_of_date: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT metrics_json
        FROM fundamentals
        WHERE ticker = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if not row:
        return {}
    try:
        payload = json.loads(row["metrics_json"])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _research_signals_for_run(conn, ticker: str, run_id: str, as_of_date: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT recency_days_min, item_count_30d, sentiment_flags_json, key_topics_json, summary_json
        FROM research_signals
        WHERE ticker = ? AND run_id = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT 1
        """,
        (ticker, run_id, as_of_date),
    ).fetchone()
    if not row:
        return {}
    return {
        "recency_days_min": row["recency_days_min"],
        "item_count_30d": int(row["item_count_30d"] or 0),
        "sentiment_flags": json.loads(row["sentiment_flags_json"] or "[]"),
        "key_topics": json.loads(row["key_topics_json"] or "[]"),
        "summary": json.loads(row["summary_json"] or "{}"),
    }


def _coverage_for_run(conn, ticker: str, run_id: str, as_of_date: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT forms_included_json, accession_numbers_json
        FROM filing_coverage
        WHERE ticker = ? AND run_id = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT 1
        """,
        (ticker, run_id, as_of_date),
    ).fetchone()
    if not row:
        return {"forms": [], "accessions": []}
    return {
        "forms": json.loads(row["forms_included_json"] or "[]"),
        "accessions": json.loads(row["accession_numbers_json"] or "[]"),
    }


def build_deltas(run_id: str, prev_run_id: str | None = None) -> dict[str, Any]:
    cfg = get_config()
    cfg.deltas_dir.mkdir(parents=True, exist_ok=True)
    resolved_prev = _resolve_prev_run_id(run_id, prev_run_id)

    with get_db() as conn:
        score_rows = conn.execute(
            """
            SELECT ticker, as_of_date
            FROM scores
            WHERE run_id = ?
            ORDER BY total_score DESC, ticker ASC
            """,
            (run_id,),
        ).fetchall()

        written = 0
        changed_count = 0
        for row in score_rows:
            ticker = row["ticker"]
            as_of_date = row["as_of_date"]
            current_cov = _coverage_for_run(conn, ticker, run_id, as_of_date)
            prev_cov = _coverage_for_run(conn, ticker, resolved_prev, as_of_date) if resolved_prev else {"forms": [], "accessions": []}
            current_accessions = {str(x) for x in current_cov.get("accessions", [])}
            prev_accessions = {str(x) for x in prev_cov.get("accessions", [])}
            new_accessions = sorted(current_accessions.difference(prev_accessions))

            new_filings: list[dict[str, Any]] = []
            if new_accessions:
                placeholders = ",".join("?" for _ in new_accessions)
                rows = conn.execute(
                    f"""
                    SELECT accession, form_type, filing_date, primary_doc_url
                    FROM filings
                    WHERE ticker = ? AND accession IN ({placeholders})
                    ORDER BY filing_date DESC
                    """,
                    (ticker, *new_accessions),
                ).fetchall()
                new_filings = [
                    {
                        "accession": r["accession"],
                        "form_type": r["form_type"],
                        "filing_date": r["filing_date"],
                        "source_url": r["primary_doc_url"],
                    }
                    for r in rows
                ]

            current_signals = _research_signals_for_run(conn, ticker, run_id, as_of_date)
            prev_signals = _research_signals_for_run(conn, ticker, resolved_prev, as_of_date) if resolved_prev else {}
            signal_changes = {
                "recency_days_min": {
                    "from": prev_signals.get("recency_days_min"),
                    "to": current_signals.get("recency_days_min"),
                },
                "item_count_30d": {
                    "from": prev_signals.get("item_count_30d"),
                    "to": current_signals.get("item_count_30d"),
                },
                "added_flags": sorted(
                    set(current_signals.get("sentiment_flags", [])).difference(prev_signals.get("sentiment_flags", []))
                ),
                "removed_flags": sorted(
                    set(prev_signals.get("sentiment_flags", [])).difference(current_signals.get("sentiment_flags", []))
                ),
            }

            current_metrics = _fundamentals_for_ticker(conn, ticker, as_of_date)
            prev_as_of = None
            if resolved_prev:
                prev_score = conn.execute(
                    """
                    SELECT as_of_date
                    FROM scores
                    WHERE ticker = ? AND run_id = ?
                    ORDER BY as_of_date DESC, created_at DESC
                    LIMIT 1
                    """,
                    (ticker, resolved_prev),
                ).fetchone()
                if prev_score:
                    prev_as_of = prev_score["as_of_date"]
            prev_metrics = _fundamentals_for_ticker(conn, ticker, prev_as_of or as_of_date) if resolved_prev else {}
            packet = _load_packet_for_ticker_as_of(conn, ticker, as_of_date)
            metric_map = {
                "revenue": "revenue",
                "operating_margin": "operating_income",
                "fcf": "fcf",
                "net_debt": "total_debt",
            }
            metric_changes: list[dict[str, Any]] = []
            for metric, line_item in metric_map.items():
                old = prev_metrics.get(metric)
                new = current_metrics.get(metric)
                if old == new:
                    continue
                if old is None and new is None:
                    continue
                metric_changes.append(
                    {
                        "metric": metric,
                        "from": old,
                        "to": new,
                        "citations": _line_item_citations(packet, line_item),
                        "derived_from": [f"fundamentals.metrics_json.{metric}"],
                    }
                )

            baseline = resolved_prev is None
            changed = bool(new_filings or metric_changes or signal_changes["added_flags"] or signal_changes["removed_flags"])
            if baseline:
                changed = True
            if changed:
                changed_count += 1

            delta_payload = {
                "run_id": run_id,
                "prev_run_id": resolved_prev,
                "ticker": ticker,
                "as_of_date": as_of_date,
                "baseline": baseline,
                "changed": changed,
                "new_filings": new_filings,
                "research_signal_changes": signal_changes,
                "financial_metric_changes": metric_changes,
                "generated_at": utc_now_iso(),
            }
            delta_path = cfg.deltas_dir / f"{ticker}_{run_id}.json"
            delta_path.write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")
            delta_hash = sha256_file(delta_path)
            conn.execute(
                """
                INSERT INTO ticker_deltas(ticker, run_id, prev_run_id, as_of_date, changed, delta_path, delta_hash, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, run_id) DO UPDATE SET
                    prev_run_id=excluded.prev_run_id,
                    as_of_date=excluded.as_of_date,
                    changed=excluded.changed,
                    delta_path=excluded.delta_path,
                    delta_hash=excluded.delta_hash,
                    created_at=excluded.created_at
                """,
                (
                    ticker,
                    run_id,
                    resolved_prev,
                    as_of_date,
                    1 if changed else 0,
                    str(delta_path),
                    delta_hash,
                    utc_now_iso(),
                ),
            )
            written += 1

    return {
        "run_id": run_id,
        "prev_run_id": resolved_prev,
        "deltas_written": written,
        "changed_tickers": changed_count,
        "out_dir": str(cfg.deltas_dir),
    }
