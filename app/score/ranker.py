from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from app.analyst.ops_snapshot import load_ops_analysis_snapshot, research_quality_to_rubric_dict
from app.analyst.output_store import latest_eligible_analysis_output_bytes
from app.config import get_config
from app.db import get_db, utc_now_iso
from app.logging import get_logger
from app.research.signals import row_to_signals_dict
from app.report.memo_builder import strict_memo_publishable
from app.score.rubric import score_packet
from app.util.financial_data_access import (
    ANNUAL_CACHED_FILING_FORM_TYPES,
    QUARTERLY_CACHED_FILING_FORM_TYPES,
    VALID_CACHED_FILING_STATUSES,
)


logger = get_logger(__name__)


def _latest_packet_rows(conn) -> list[Any]:
    return conn.execute(
        """
        SELECT ep1.*
        FROM evidence_packets ep1
        INNER JOIN (
            SELECT ticker, MAX(as_of_date) AS max_date
            FROM evidence_packets
            GROUP BY ticker
        ) ep2 ON ep1.ticker = ep2.ticker AND ep1.as_of_date = ep2.max_date
        """
    ).fetchall()


def _latest_analyst_decision(
    conn, ticker: str, as_of_date: str | None = None
) -> dict[str, Any] | None:
    # Bounded to the scoring date when one is given: a score dated 2025-03-31 must not be
    # classified by a decision written after it.
    result = latest_eligible_analysis_output_bytes(conn, ticker, "decision", as_of_date=as_of_date)
    if result is None:
        return None
    _, decision_bytes = result
    try:
        payload = json.loads(decision_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload.get("decision") if isinstance(payload, dict) else None


def _latest_research_quality(
    conn, ticker: str, as_of_date: str, run_id: str | None = None
) -> dict[str, Any] | None:
    _ = conn
    snapshot = load_ops_analysis_snapshot(ticker, as_of_date=as_of_date, run_id=run_id)
    return (
        research_quality_to_rubric_dict(snapshot.research_quality) if snapshot is not None else None
    )


def _latest_research_signals(
    conn,
    ticker: str,
    as_of_date: str,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    if run_id:
        row = conn.execute(
            """
            SELECT *
            FROM research_signals
            WHERE ticker = ? AND as_of_date = ? AND run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, as_of_date, run_id),
        ).fetchone()
        payload = row_to_signals_dict(row)
        if payload:
            return payload
    row = conn.execute(
        """
        SELECT *
        FROM research_signals
        WHERE ticker = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC, created_at DESC, id DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    # No further fallback: a signals row dated after the scoring date is look-ahead,
    # so "nothing at or before the date" is None and the caller scores it as missing.
    return row_to_signals_dict(row)


def _latest_filing_coverage(
    conn, ticker: str, as_of_date: str, run_id: str | None
) -> dict[str, Any] | None:
    if run_id:
        row = conn.execute(
            """
            SELECT forms_included_json, accession_numbers_json, coverage_score, missing_required_json
            FROM filing_coverage
            WHERE ticker = ? AND run_id = ? AND as_of_date = ?
            LIMIT 1
            """,
            (ticker, run_id, as_of_date),
        ).fetchone()
        if row:
            return {
                "forms_included": json.loads(row["forms_included_json"] or "[]"),
                "accession_numbers": json.loads(row["accession_numbers_json"] or "[]"),
                "coverage_score": float(row["coverage_score"] or 0.0),
                "missing_required": json.loads(row["missing_required_json"] or "[]"),
            }
    row = conn.execute(
        """
        SELECT forms_included_json, accession_numbers_json, coverage_score, missing_required_json
        FROM filing_coverage
        WHERE ticker = ? AND as_of_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if not row:
        return None
    return {
        "forms_included": json.loads(row["forms_included_json"] or "[]"),
        "accession_numbers": json.loads(row["accession_numbers_json"] or "[]"),
        "coverage_score": float(row["coverage_score"] or 0.0),
        "missing_required": json.loads(row["missing_required_json"] or "[]"),
    }


def _previous_filing_coverage_accessions(
    conn, ticker: str, as_of_date: str, run_id: str | None
) -> set[str]:
    if not run_id:
        return set()
    row = conn.execute(
        """
        SELECT accession_numbers_json
        FROM filing_coverage
        WHERE ticker = ? AND run_id != ? AND as_of_date <= ?
        ORDER BY as_of_date DESC, created_at DESC, id DESC
        LIMIT 1
        """,
        (ticker, run_id, as_of_date),
    ).fetchone()
    if not row:
        return set()
    return {str(x) for x in json.loads(row["accession_numbers_json"] or "[]")}


def _latest_previous_signal_flags(
    conn, ticker: str, as_of_date: str, run_id: str | None
) -> set[str]:
    if not run_id:
        return set()
    row = conn.execute(
        """
        SELECT sentiment_flags_json
        FROM research_signals
        WHERE ticker = ? AND run_id != ? AND as_of_date <= ?
        ORDER BY as_of_date DESC, created_at DESC, id DESC
        LIMIT 1
        """,
        (ticker, run_id, as_of_date),
    ).fetchone()
    if not row:
        return set()
    return {str(x) for x in json.loads(row["sentiment_flags_json"] or "[]")}


def _change_context(
    conn,
    *,
    ticker: str,
    as_of_date: str,
    run_id: str | None,
    filing_coverage: dict[str, Any] | None,
    research_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    current_accessions = set(str(x) for x in (filing_coverage or {}).get("accession_numbers", []))
    prev_accessions = _previous_filing_coverage_accessions(conn, ticker, as_of_date, run_id)
    new_accessions = sorted(current_accessions.difference(prev_accessions))

    new_recent_count = 0
    if new_accessions:
        as_of = date.fromisoformat(as_of_date)
        placeholders = ",".join("?" for _ in new_accessions)
        rows = conn.execute(
            f"""
            SELECT filing_date
            FROM filings
            WHERE ticker = ? AND accession IN ({placeholders})
            """,
            (ticker, *new_accessions),
        ).fetchall()
        for row in rows:
            try:
                filed = date.fromisoformat(row["filing_date"])
            except Exception:
                continue
            if 0 <= (as_of - filed).days <= 30:
                new_recent_count += 1

    current_flags = set(str(x) for x in (research_signals or {}).get("sentiment_flags", []))
    prev_flags = _latest_previous_signal_flags(conn, ticker, as_of_date, run_id)
    new_negative_flags = sorted(current_flags.difference(prev_flags))
    recency = (research_signals or {}).get("recency_days_min")
    research_stale = (recency is None) or (isinstance(recency, (int, float)) and recency > 180)

    return {
        "new_filing_recent_count": new_recent_count,
        "new_accessions": new_accessions,
        "new_negative_flags": new_negative_flags,
        "research_stale": bool(research_stale),
    }


def _passes_data_completeness_gate(
    conn, *, ticker: str, as_of_date: str, run_id: str | None
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        as_of = date.fromisoformat(as_of_date)
    except Exception:
        return False, ["invalid_as_of_date"]
    recent_q = date.fromordinal(as_of.toordinal() - 210).isoformat()
    recent_k = date.fromordinal(as_of.toordinal() - 540).isoformat()
    annual_forms = tuple(
        form
        for form in ANNUAL_CACHED_FILING_FORM_TYPES
        if form in {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
    )
    quarterly_forms = QUARTERLY_CACHED_FILING_FORM_TYPES
    status_placeholders = ",".join("?" for _ in VALID_CACHED_FILING_STATUSES)
    annual_placeholders = ",".join("?" for _ in annual_forms)
    quarterly_placeholders = ",".join("?" for _ in quarterly_forms)

    filing_ok_row = conn.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM filings
        WHERE ticker = ?
          AND status IN ({status_placeholders})
          AND (
            (form_type IN ({quarterly_placeholders}) AND COALESCE(filing_date, '1900-01-01') >= ?)
            OR (form_type IN ({annual_placeholders}) AND COALESCE(filing_date, '1900-01-01') >= ?)
          )
        """,
        (
            ticker,
            *VALID_CACHED_FILING_STATUSES,
            *quarterly_forms,
            recent_q,
            *annual_forms,
            recent_k,
        ),
    ).fetchone()
    filing_ok = int(filing_ok_row["n"] or 0) > 0
    if not filing_ok:
        reasons.append("missing_recent_10k_or_10q")

    signals = _latest_research_signals(conn, ticker, as_of_date, run_id=run_id)
    signals_present = signals is not None
    explicit_no_non_edgar = bool((signals or {}).get("summary", {}).get("no_non_edgar_coverage"))
    if not signals_present and not explicit_no_non_edgar:
        reasons.append("missing_research_signals")

    return filing_ok and (signals_present or explicit_no_non_edgar), reasons


def score_ticker(ticker: str, *, run_id: str | None = None) -> bool:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT as_of_date, packet_path
            FROM evidence_packets
            WHERE ticker = ?
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
        if not row:
            return False

        path = Path(row["packet_path"])
        if not path.exists():
            return False
        packet = json.loads(path.read_text(encoding="utf-8"))
        decision = _latest_analyst_decision(conn, ticker, row["as_of_date"])
        research_quality = _latest_research_quality(conn, ticker, row["as_of_date"], run_id=run_id)
        research_signals = _latest_research_signals(conn, ticker, row["as_of_date"], run_id=run_id)
        filing_coverage = _latest_filing_coverage(conn, ticker, row["as_of_date"], run_id=run_id)
        change_context = _change_context(
            conn,
            ticker=ticker,
            as_of_date=row["as_of_date"],
            run_id=run_id,
            filing_coverage=filing_coverage,
            research_signals=research_signals,
        )
        subscores, total, classification, reasons = score_packet(
            packet,
            decision,
            research_quality=research_quality,
            research_signals=research_signals,
            filing_coverage=filing_coverage,
            change_context=change_context,
        )

        conn.execute(
            """
            INSERT INTO scores(
                ticker, as_of_date, run_id, subscores_json, total_score, decision,
                is_candidate, is_publishable, candidate_run_id, reasons_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, 0, 0, NULL, ?, ?)
            ON CONFLICT(ticker, as_of_date) DO UPDATE SET
                run_id=excluded.run_id,
                subscores_json=excluded.subscores_json,
                total_score=excluded.total_score,
                decision=excluded.decision,
                reasons_json=excluded.reasons_json,
                created_at=excluded.created_at,
                is_candidate=0,
                is_publishable=0,
                candidate_run_id=NULL
            """,
            (
                ticker,
                row["as_of_date"],
                run_id,
                json.dumps(subscores),
                total,
                classification,
                json.dumps(reasons),
                utc_now_iso(),
            ),
        )
    return True


def _latest_per_ticker(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _recency_key(row: dict[str, Any]) -> tuple[str, str, int]:
        return (
            str(row.get("as_of_date") or ""),
            str(row.get("created_at") or ""),
            int(row.get("id") or 0),
        )

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row["ticker"]).upper()
        existing = latest.get(ticker)
        if existing is None:
            latest[ticker] = row
            continue
        if _recency_key(row) >= _recency_key(existing):
            latest[ticker] = row
    return list(latest.values())


def _row_passes_scope(row: dict[str, Any], scope_set: set[str]) -> bool:
    if not scope_set:
        return True
    return str(row.get("ticker", "")).upper() in scope_set


def _rows_for_ranking(
    conn,
    *,
    run_id: str | None,
    as_of_date: str | None,
    candidate_scope: list[str] | None,
) -> tuple[list[dict[str, Any]], str]:
    scope_set = {t.upper() for t in (candidate_scope or []) if t.strip()}

    if run_id:
        sql = """
            SELECT id, ticker, as_of_date, run_id, total_score, decision, reasons_json, created_at
            FROM scores
            WHERE run_id = ?
        """
        # For run-scoped ranking, do not filter by run-as-of date. Pick latest effective row per ticker in this run.
        rows = [dict(row) for row in conn.execute(sql, (run_id,)).fetchall()]
        rows = [row for row in rows if _row_passes_scope(row, scope_set)]
        rows = _latest_per_ticker(rows)
        rows.sort(key=lambda r: (-float(r.get("total_score") or 0), str(r.get("ticker") or "")))
        return rows, "run_id"

    params = []
    sql = """
        SELECT id, ticker, as_of_date, run_id, total_score, decision, reasons_json, created_at
        FROM scores
    """
    if as_of_date:
        sql += " WHERE as_of_date = ?"
        params.append(as_of_date)

    rows = [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]
    rows = [row for row in rows if _row_passes_scope(row, scope_set)]
    rows = _latest_per_ticker(rows)
    rows.sort(key=lambda r: (-float(r.get("total_score") or 0), str(r.get("ticker") or "")))
    return rows, "legacy_latest_per_ticker"


def score_and_rank(
    *,
    top_n: int = 10,
    memo_mode: str = "strict",
    run_id: str | None = None,
    candidate_scope: list[str] | None = None,
    as_of_date: str | None = None,
    rescore: bool = True,
) -> dict[str, Any]:
    cfg = get_config()

    scored = 0
    if rescore:
        if candidate_scope:
            tickers = [t.upper() for t in candidate_scope if t.strip()]
        else:
            with get_db() as conn:
                rows = _latest_packet_rows(conn)
            tickers = [row["ticker"] for row in rows]
        for ticker in tickers:
            if score_ticker(ticker, run_id=run_id):
                scored += 1

    with get_db() as conn:
        dataset_rows, selection_mode = _rows_for_ranking(
            conn,
            run_id=run_id,
            as_of_date=as_of_date,
            candidate_scope=candidate_scope,
        )

        if run_id:
            conn.execute(
                """
                UPDATE scores
                SET is_candidate = 0,
                    is_publishable = 0,
                    candidate_run_id = NULL
                WHERE run_id = ? OR candidate_run_id = ?
                """,
                (run_id, run_id),
            )

        top_n = max(0, int(top_n))
        candidate_rows = dataset_rows[:top_n]
        candidate_ids = {int(row["id"]) for row in candidate_rows}
        final_candidate_ids: set[int] = set()
        final_publishable_ids: set[int] = set()

        publishable_count = 0
        effective_candidate_count = 0
        for row in dataset_rows:
            row_id = int(row["id"])
            is_candidate = row_id in candidate_ids
            is_publishable = False
            if is_candidate:
                gate_ok, gate_reasons = _passes_data_completeness_gate(
                    conn,
                    ticker=row["ticker"],
                    as_of_date=row.get("as_of_date") or "",
                    run_id=run_id,
                )
                if not gate_ok:
                    is_candidate = False
                    reasons = json.loads(row.get("reasons_json") or "[]")
                    for reason in gate_reasons:
                        msg = f"Data completeness gate: {reason}"
                        if msg not in reasons:
                            reasons.append(msg)
                    row["reasons_json"] = json.dumps(reasons)
                    conn.execute(
                        "UPDATE scores SET reasons_json = ? WHERE id = ?",
                        (row["reasons_json"], row_id),
                    )
            if is_candidate:
                effective_candidate_count += 1
                final_candidate_ids.add(row_id)
                is_publishable, _ = strict_memo_publishable(row["ticker"], row.get("as_of_date"))
                if is_publishable:
                    publishable_count += 1
                    final_publishable_ids.add(row_id)
            conn.execute(
                """
                UPDATE scores
                SET is_candidate = ?,
                    is_publishable = ?,
                    candidate_run_id = ?
                WHERE id = ?
                """,
                (
                    1 if is_candidate else 0,
                    1 if is_publishable else 0,
                    run_id if run_id else None,
                    row_id,
                ),
            )

    ranking_payload = {
        "generated_at": utc_now_iso(),
        "memo_mode": memo_mode,
        "top_n": top_n,
        "run_id": run_id,
        "as_of_date": as_of_date,
        "selection_mode": selection_mode,
        "score_rows_considered": len(dataset_rows),
        "tickers_ranked": len(dataset_rows),
        "candidate_count": effective_candidate_count,
        "publishable_count": publishable_count,
        "rankings": [
            {
                "ticker": row["ticker"],
                "as_of_date": row["as_of_date"],
                "total_score": row["total_score"],
                "decision": row["decision"],
                "is_candidate": int(row["id"]) in final_candidate_ids,
                "is_publishable": int(row["id"]) in final_publishable_ids,
                "run_id": row.get("run_id"),
                "reasons": json.loads(row["reasons_json"]),
            }
            for row in dataset_rows
        ],
    }
    out_path = cfg.rankings_dir / f"rankings_{utc_now_iso()[:10]}.json"
    out_path.write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")

    logger.info("score_completed", extra={"stage_name": "score", "stage_count": scored})
    return {
        "scored": scored,
        "score_rows_considered": len(dataset_rows),
        "tickers_ranked": len(dataset_rows),
        "candidate_count": effective_candidate_count,
        "publishable_count": publishable_count,
        "selection_mode": selection_mode,
        "run_id": run_id,
    }
