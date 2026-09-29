"""Company depth tabs: research, dossier, decisions.

Research unpacks the latest ``deep_research`` valuation row (thesis block
with original vs adjusted values, calibrated adjustments, analyst notes,
filing citations) plus the ticker's ``evidence_items``. The dossier tab
serves the newest on-disk ``dossier.md`` through the Reader's allowlisted
renderer so claims travel with it. Decisions list the ticker's
``dispositions`` history and its ``ticker_outcomes`` record.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.autonomous.artifact_financial_audit import (
    PASS,
    artifact_decision_eligibility,
    financial_integrity_manifest_is_usable,
)
from app.outcomes.lineage import outcome_row_is_decision_eligible
from app.valuation.lineage import valuation_row_is_decision_eligible
from app.watchlist.lineage import watchlist_row_is_decision_eligible
from app.web.readmodel import reader as reader_model
from app.web.readmodel.db import table_exists

MAX_CITATIONS = 40
MAX_EVIDENCE = 40
MAX_NOTE_ITEMS = 12
MAX_OUTCOME_ROWS = 20

# thesis scalars surfaced verbatim (original vs adjusted lives here)
_THESIS_FIELDS = (
    "original_dcf",
    "adjusted_dcf",
    "original_epv",
    "adjusted_epv",
    "original_graham",
    "adjusted_intrinsic_mid",
    "adjusted_margin_of_safety",
    "adjusted_value_floored",
    "current_price",
    "average_coverage",
    "high_priority_unresolved",
    "hypotheses_confirmed",
    "hypotheses_contradicted",
    "hypotheses_partially_confirmed",
    "hypotheses_inconclusive",
)

_NOTE_SECTIONS = ("positives", "risks", "surprises", "adjustment_triggers")

_AUDIT_UNAVAILABLE = "FINANCIAL_INTEGRITY_AUDIT_UNAVAILABLE"
_REPORT_BLOCKED = "FINANCIAL_INTEGRITY_REPORT_BLOCKED"


def _read_json_dict(raw: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _note_item(raw: dict[str, Any]) -> dict[str, Any]:
    citations = [
        {
            "section": str(c.get("section") or ""),
            "excerpt": str(c.get("excerpt") or ""),
        }
        for c in raw.get("citations") or []
        if isinstance(c, dict)
    ]
    return {
        "claim": str(raw.get("claim") or ""),
        "direction": raw.get("direction"),
        "severity": raw.get("severity"),
        "suggested_adjustment": raw.get("suggested_adjustment"),
        "validation_status": raw.get("validation_status"),
        "citations": citations,
    }


def _analyst_notes(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    sections = {
        key: [
            _note_item(item)
            for item in (raw.get(key) or [])[:MAX_NOTE_ITEMS]
            if isinstance(item, dict)
        ]
        for key in _NOTE_SECTIONS
    }
    overall = raw.get("overall_assessment")
    if not any(sections.values()) and not overall:
        return None
    return {
        **sections,
        "overall_assessment": str(overall) if overall else None,
        "filing_sections_read": [
            str(s) for s in raw.get("filing_sections_read") or [] if isinstance(s, str)
        ],
    }


def _adjustment(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "hypothesis_source": raw.get("hypothesis_source"),
        "hypothesis_claim": raw.get("hypothesis_claim"),
        "hypothesis_direction": raw.get("hypothesis_direction"),
        "hypothesis_status": raw.get("hypothesis_status"),
        "affected_method": raw.get("affected_method"),
        "adjustment_magnitude": (
            float(raw["adjustment_magnitude"])
            if isinstance(raw.get("adjustment_magnitude"), (int, float))
            else None
        ),
        "adjustment_confidence": raw.get("adjustment_confidence"),
        "calibration_detail": raw.get("calibration_detail"),
    }


def _unresolved(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": raw.get("description"),
        "importance": raw.get("importance"),
        "unresolved_reason": raw.get("unresolved_reason"),
        "hypothesis_priority": raw.get("hypothesis_priority"),
        "hypothesis_direction": raw.get("hypothesis_direction"),
    }


def _citation(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_id": raw.get("citation_id"),
        "section": raw.get("section"),
        "excerpt": str(raw.get("excerpt") or ""),
        "relevance": raw.get("relevance"),
        "hypothesis_source": raw.get("hypothesis_source"),
        "source_form_type": raw.get("source_form_type"),
        "source_filing_date": raw.get("source_filing_date"),
        "source_title": raw.get("source_title"),
        "source_url": raw.get("source_url"),
    }


def research(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    """The latest deep-research pass unpacked, or an honest absence."""
    if not financial_integrity_manifest_is_usable():
        return {"available": False, "reason": _AUDIT_UNAVAILABLE}
    row = conn.execute(
        """
        SELECT ticker, as_of_date, method, inputs_json, outputs_json,
               warnings_json, created_at, valuation_writer_version,
               quality_gate_verdict, confidence_class, gate_reason_codes,
               valuation_headwinds, valuation_supports, source_run_id,
               source_artifact_path, source_artifact_sha256,
               financial_integrity_fingerprint
        FROM valuations
        WHERE ticker = ? AND method = 'deep_research'
        ORDER BY as_of_date DESC, created_at DESC, id DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    if row is None:
        return {"available": False, "reason": "NEVER_RESEARCHED"}
    if not valuation_row_is_decision_eligible(
        row,
        require_exact_source_payload=True,
    ):
        # Select newest first, then authorize that exact DB row. Filtering for
        # an older eligible row would resurrect superseded research.
        return {"available": False, "reason": _REPORT_BLOCKED}
    outputs = _read_json_dict(row["outputs_json"])
    report_path = outputs.get("report_path")
    if not report_path or artifact_decision_eligibility(Path(str(report_path))) != PASS:
        # The latest DB row remains authoritative even when its report is not
        # authorized.  Do not search for an older audited report: that would
        # resurrect research superseded by the blocked row.
        return {"available": False, "reason": _REPORT_BLOCKED}
    thesis_raw = outputs.get("thesis") if isinstance(outputs.get("thesis"), dict) else {}
    thesis = {field: thesis_raw.get(field) for field in _THESIS_FIELDS}
    return {
        "available": True,
        "reason": None,
        "as_of_date": str(row["as_of_date"]),
        "created_at": row["created_at"],
        "status": outputs.get("status"),
        "conviction_class": outputs.get("conviction_class"),
        "conviction_score": outputs.get("conviction_score"),
        "gate_action": outputs.get("gate_action"),
        "tension_type": outputs.get("tension_type"),
        "methods_agree": outputs.get("methods_agree"),
        "consensus_strength": outputs.get("consensus_strength"),
        "method_count": outputs.get("method_count"),
        "hypotheses_generated": outputs.get("hypotheses_generated"),
        "thesis": thesis,
        "adjustments": [
            _adjustment(a) for a in (thesis_raw.get("adjustments") or []) if isinstance(a, dict)
        ],
        "unresolved": [
            _unresolved(u) for u in (thesis_raw.get("unresolved") or []) if isinstance(u, dict)
        ],
        "analyst_notes": _analyst_notes(outputs.get("analyst_notes")),
        "citations": [
            _citation(c)
            for c in (outputs.get("citations") or [])[:MAX_CITATIONS]
            if isinstance(c, dict)
        ],
        "report_path": str(report_path) if report_path else None,
        "report_available": Path(str(report_path)).is_file(),
        "evidence": evidence_items(conn, ticker),
    }


def evidence_items(conn: sqlite3.Connection, ticker: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT evidence_id, as_of_date, source_type, source_title, source_url,
               source_published_at, excerpt_text
        FROM evidence_items
        WHERE ticker = ?
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT ?
        """,
        (ticker, MAX_EVIDENCE),
    ).fetchall()
    return [
        {
            "evidence_id": row["evidence_id"],
            "as_of_date": row["as_of_date"],
            "source_type": row["source_type"],
            "source_title": row["source_title"],
            "source_url": row["source_url"],
            "source_published_at": row["source_published_at"],
            "excerpt": row["excerpt_text"],
        }
        for row in rows
    ]


def dossier(ticker: str) -> dict[str, Any]:
    """Newest on-disk dossier for the ticker, rendered with its claims."""
    base = reader_model.outputs_dir() / "dossiers"
    candidates = sorted(
        base.glob(f"*/{ticker}/dossier.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"available": False, "reason": "NO_DOSSIER"}
    path = candidates[0]
    try:
        artifact = reader_model.render_artifact(str(path))
    except reader_model.ArtifactRefused as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "reason": None,
        "run_label": path.parent.parent.name,
        "others": len(candidates) - 1,
        **artifact,
    }


def _journal_command(ticker: str, disposition_id: int) -> str:
    # Mirrors the Today deck's decision cards (today.open_decisions).
    return (
        f"ivi investor journal {ticker} --disposition-id {disposition_id} "
        '--action acted|passed|deferred --reason <CODE> --rationale "<why>"'
    )


def _empty_decisions() -> dict[str, Any]:
    return {
        "dispositions": [],
        "outcomes": [],
        "outcomes_total": 0,
        "outcomes_open": 0,
        "outcomes_closed": 0,
    }


def _decision_eligible_history(
    rows: list[sqlite3.Row],
    *,
    source_field: str,
    ticker: str,
) -> list[sqlite3.Row]:
    """Filter a preselected history without promoting an older decision.

    The first row is the endpoint's newest selection.  If it is not positively
    authorized, the entire lineage is suppressed because an older row would
    otherwise become the apparent current decision.  Once the newest row is
    authorized, individually unaudited historical rows remain excluded.
    """

    if not rows or not watchlist_row_is_decision_eligible(
        rows[0],
        source_run_field=source_field,
        ticker=ticker,
    ):
        return []
    return [
        row
        for row in rows
        if watchlist_row_is_decision_eligible(
            row,
            source_run_field=source_field,
            ticker=ticker,
        )
    ]


def decisions(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    """Disposition history plus the outcome record for one ticker."""
    if not financial_integrity_manifest_is_usable():
        return _empty_decisions()
    # Dispositions hang off watchlist rows; with no watchlist table there are none.
    disposition_rows = (
        conn.execute(
            """
            SELECT d.id, d.kind, d.status, d.opened_at, d.opened_by, d.decided_at,
                   d.operator, d.reason_code, d.rationale, d.intended_size,
                   d.sizing_rationale, d.pre_mortem, d.trigger_snapshot_json,
                   d.event_id, w.*
            FROM dispositions AS d
            LEFT JOIN watchlist AS w ON w.id = d.watchlist_id
            WHERE d.ticker = ?
            ORDER BY d.opened_at DESC, d.id DESC
            """,
            (ticker,),
        ).fetchall()
        if table_exists(conn, "watchlist")
        else []
    )
    disposition_rows = _decision_eligible_history(
        disposition_rows,
        source_field="source_run_id",
        ticker=ticker,
    )
    dispositions = [
        {
            "id": int(row["id"]),
            "kind": row["kind"],
            "status": row["status"],
            "opened_at": row["opened_at"],
            "opened_by": row["opened_by"],
            "decided_at": row["decided_at"],
            "operator": row["operator"],
            "reason_code": row["reason_code"],
            "rationale": row["rationale"],
            "intended_size": row["intended_size"],
            "sizing_rationale": row["sizing_rationale"],
            "pre_mortem": row["pre_mortem"],
            "trigger": _read_json_dict(row["trigger_snapshot_json"]),
            "event_id": row["event_id"],
            "journal_command": (
                _journal_command(ticker, int(row["id"])) if str(row["status"]) == "OPEN" else None
            ),
        }
        for row in disposition_rows
    ]

    outcome_rows = conn.execute(
        """
        SELECT *
        FROM ticker_outcomes
        WHERE ticker = ?
        ORDER BY as_of_date DESC, id DESC
        LIMIT ?
        """,
        (ticker, MAX_OUTCOME_ROWS),
    ).fetchall()
    if not outcome_rows or not outcome_row_is_decision_eligible(outcome_rows[0]):
        outcome_rows = []
    else:
        outcome_rows = [row for row in outcome_rows if outcome_row_is_decision_eligible(row)]
    outcomes = [
        {
            "as_of_date": row["as_of_date"],
            "decision": row["decision"],
            "conviction": row["conviction"],
            "grade": row["grade"],
            "outcome_status": row["outcome_status"],
            "close_date": row["close_date"],
            "horizon_days": row["horizon_days"],
            "entry_price": row["entry_price"],
            "realized_return_pct": row["realized_return_pct"],
            "benchmark_return_pct": row["benchmark_return_pct"],
            "excess_return_pct": row["excess_return_pct"],
            "reached_buy_target": (
                bool(row["reached_buy_target"]) if row["reached_buy_target"] is not None else None
            ),
        }
        for row in outcome_rows
    ]
    outcomes_open = sum(str(row["outcome_status"]) == "OPEN" for row in outcome_rows)
    outcomes_closed = sum(str(row["outcome_status"]) == "CLOSED" for row in outcome_rows)
    return {
        "dispositions": dispositions,
        "outcomes": outcomes,
        "outcomes_total": len(outcome_rows),
        "outcomes_open": outcomes_open,
        "outcomes_closed": outcomes_closed,
    }
