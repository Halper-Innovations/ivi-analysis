"""Outcomes & calibration read model.

Method scoreboard aggregates ``deep_research_method_outcomes`` exactly as
the resolver writes them (CORRECT / INCORRECT / INCONCLUSIVE per dcf / epv /
graham; hit rate counts only resolved rows). Realized returns aggregate
closed ``ticker_outcomes`` with an excess-return record; the journal is the
decided-``dispositions`` ledger with the standing live-decision goal.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    authorized_artifact_bytes,
)
from app.outcomes.lineage import (
    outcome_row_is_decision_eligible,
    serialized_outcome_binding_is_decision_eligible,
)
from app.discovery.lineage import discovery_candidate_bindings_are_current
from app.valuation.lineage import valuation_row_is_decision_eligible

# The strategic-review kill criterion: ≥6 journaled live decisions.
JOURNAL_GOAL = 6
JOURNAL_DEADLINE = "2026-09-09"

# Fixed histogram edges (pct); open-ended at both tails.
HISTOGRAM_EDGES = (-50.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 50.0)

MAX_JOURNAL_ROWS = 100
MAX_CALIBRATION_ROWS = 50


def method_scoreboard(conn: sqlite3.Connection) -> dict[str, Any]:
    """Per-method hit rates, resolved-only, plus a monthly resolution series."""
    rows = conn.execute(
        """
        WITH valuation_versions AS (
            SELECT
                id AS source_valuation_id,
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at, valuation_writer_version,
                quality_gate_verdict, confidence_class, gate_reason_codes,
                valuation_headwinds, valuation_supports, source_run_id,
                source_artifact_path, source_artifact_sha256,
                financial_integrity_fingerprint, 1 AS is_live
            FROM valuations
            UNION ALL
            SELECT
                source_id AS source_valuation_id,
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at, valuation_writer_version,
                quality_gate_verdict, confidence_class, gate_reason_codes,
                valuation_headwinds, valuation_supports, source_run_id,
                source_artifact_path, source_artifact_sha256,
                financial_integrity_fingerprint, 0 AS is_live
            FROM valuations_history
        )
        SELECT
            m.id AS method_outcome_id,
            m.method AS outcome_method,
            m.outcome,
            m.unadjusted_source,
            d.source_as_of_date,
            d.source_run_id AS outcome_source_run_id,
            d.source_artifact_path AS outcome_source_artifact_path,
            d.source_artifact_sha256 AS outcome_source_artifact_sha256,
            d.source_valuation_fingerprint AS outcome_source_fingerprint,
            v.ticker,
            v.as_of_date,
            v.method,
            v.inputs_json,
            v.outputs_json,
            v.warnings_json,
            v.created_at,
            v.valuation_writer_version,
            v.quality_gate_verdict,
            v.confidence_class,
            v.gate_reason_codes,
            v.valuation_headwinds,
            v.valuation_supports,
            v.source_run_id,
            v.source_artifact_path,
            v.source_artifact_sha256,
            v.financial_integrity_fingerprint
        FROM deep_research_method_outcomes m
        JOIN deep_research_outcomes d ON d.id = m.outcome_id
        LEFT JOIN valuation_versions v
          ON (
              d.source_valuation_fingerprint IS NOT NULL
              AND v.financial_integrity_fingerprint =
                  d.source_valuation_fingerprint
          )
          OR (
              d.source_valuation_fingerprint IS NULL
              AND v.is_live = 1
              AND v.source_valuation_id = d.source_valuation_id
          )
        ORDER BY m.method, d.source_as_of_date
        """
    ).fetchall()
    authorized_by_outcome: dict[int, sqlite3.Row] = {}
    for row in rows:
        if (
            row["outcome_source_fingerprint"] is None
            or row["outcome_source_fingerprint"] != row["financial_integrity_fingerprint"]
            or row["outcome_source_run_id"] != row["source_run_id"]
            or row["outcome_source_artifact_path"] != row["source_artifact_path"]
            or row["outcome_source_artifact_sha256"] != row["source_artifact_sha256"]
            or not valuation_row_is_decision_eligible(row)
        ):
            continue
        authorized_by_outcome.setdefault(int(row["method_outcome_id"]), row)
    authorized = list(authorized_by_outcome.values())

    totals: dict[str, dict[str, Any]] = {}
    monthly: dict[tuple[str, str], dict[str, Any]] = {}
    for row in authorized:
        method = str(row["outcome_method"])
        outcome = str(row["outcome"])
        total = totals.setdefault(
            method,
            {
                "method": method,
                "n": 0,
                "correct": 0,
                "incorrect": 0,
                "inconclusive": 0,
                "unadjusted_source": False,
            },
        )
        total["n"] += 1
        if outcome == "CORRECT":
            total["correct"] += 1
        elif outcome == "INCORRECT":
            total["incorrect"] += 1
        elif outcome == "INCONCLUSIVE":
            total["inconclusive"] += 1
        total["unadjusted_source"] = bool(total["unadjusted_source"] or row["unadjusted_source"])

        month = str(row["source_as_of_date"] or "")[:7]
        month_row = monthly.setdefault(
            (month, method),
            {
                "month": month,
                "method": method,
                "n": 0,
                "correct": 0,
                "incorrect": 0,
            },
        )
        month_row["n"] += 1
        if outcome == "CORRECT":
            month_row["correct"] += 1
        elif outcome == "INCORRECT":
            month_row["incorrect"] += 1

    methods = []
    for method in sorted(totals):
        total = totals[method]
        resolved = int(total["correct"]) + int(total["incorrect"])
        methods.append(
            {
                **total,
                "hit_rate": (round(int(total["correct"]) / resolved, 4) if resolved else None),
            }
        )
    series = [monthly[key] for key in sorted(monthly)]
    return {"methods": methods, "monthly": series}


def _bucket_stats(rows: list[tuple[float, float | None]]) -> dict[str, Any]:
    """(excess, realized) aggregate: n, averages, hit rate on excess>0."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "avg_excess": None,
            "median_excess": None,
            "hit_rate": None,
            "avg_realized": None,
        }
    excess = [r[0] for r in rows]
    realized = [r[1] for r in rows if r[1] is not None]
    return {
        "n": n,
        "avg_excess": round(sum(excess) / n, 2),
        "median_excess": round(median(excess), 2),
        "hit_rate": round(sum(1 for e in excess if e > 0) / n, 4),
        "avg_realized": round(sum(realized) / len(realized), 2) if realized else None,
    }


def realized_returns(conn: sqlite3.Connection) -> dict[str, Any]:
    """Closed outcomes with an excess-return record, sliced honestly.

    These are scan-decision outcomes measured against the benchmark — the
    platform's paper record, not live trades (the journal below is live).
    """
    rows = conn.execute(
        """
        WITH latest_lineage AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, run_id
                       ORDER BY updated_at DESC, id DESC
                   ) AS lineage_rank
            FROM ticker_outcomes
        )
        SELECT *
        FROM latest_lineage
        WHERE lineage_rank = 1
          AND outcome_status = 'CLOSED'
          AND excess_return_pct IS NOT NULL
        """
    ).fetchall()
    authorized_rows = [row for row in rows if outcome_row_is_decision_eligible(row)]
    all_pairs = [(float(r["excess_return_pct"]), r["realized_return_pct"]) for r in authorized_rows]

    edges = HISTOGRAM_EDGES
    counts = [0] * (len(edges) + 1)
    for excess, _ in all_pairs:
        index = sum(1 for edge in edges if excess >= edge)
        counts[index] += 1
    bins = []
    for i, count in enumerate(counts):
        low = edges[i - 1] if i > 0 else None
        high = edges[i] if i < len(edges) else None
        bins.append({"low": low, "high": high, "count": count})

    by_decision: dict[str, list[tuple[float, float | None]]] = {}
    by_conviction: dict[int, list[tuple[float, float | None]]] = {}
    for row in authorized_rows:
        pair = (float(row["excess_return_pct"]), row["realized_return_pct"])
        by_decision.setdefault(str(row["decision"]), []).append(pair)
        if row["conviction"] is not None:
            by_conviction.setdefault(int(row["conviction"]), []).append(pair)

    return {
        "overall": _bucket_stats(all_pairs),
        "histogram": bins,
        "by_decision": [
            {"decision": decision, **_bucket_stats(pairs)}
            for decision, pairs in sorted(by_decision.items())
        ],
        "by_conviction": [
            {"conviction": conviction, **_bucket_stats(pairs)}
            for conviction, pairs in sorted(by_conviction.items())
        ],
    }


def journal(conn: sqlite3.Connection) -> dict[str, Any]:
    """The live-decision ledger: decided dispositions plus goal progress."""
    decided_rows = conn.execute(
        """
        SELECT id, ticker, kind, status, opened_at, decided_at, operator,
               reason_code, rationale, intended_size, sizing_rationale
        FROM dispositions
        WHERE status != 'OPEN'
        ORDER BY decided_at DESC, id DESC
        LIMIT ?
        """,
        (MAX_JOURNAL_ROWS,),
    ).fetchall()
    open_count = int(
        conn.execute("SELECT COUNT(*) AS n FROM dispositions WHERE status = 'OPEN'").fetchone()["n"]
    )
    entries = [
        {
            "id": int(row["id"]),
            "ticker": row["ticker"],
            "kind": row["kind"],
            "status": row["status"],
            "opened_at": row["opened_at"],
            "decided_at": row["decided_at"],
            "operator": row["operator"],
            "reason_code": row["reason_code"],
            "rationale": row["rationale"],
            "intended_size": row["intended_size"],
            "sizing_rationale": row["sizing_rationale"],
        }
        for row in decided_rows
    ]
    return {
        "entries": entries,
        "decided_count": len(entries),
        "open_count": open_count,
        "goal": JOURNAL_GOAL,
        "goal_deadline": JOURNAL_DEADLINE,
    }


def calibration(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Stored calibration reports with their headline blocks unpacked."""
    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT
                run_id, as_of_date, report_path, report_json, created_at,
                CASE
                    WHEN run_id LIKE 'calibration_report_%'
                    THEN 'grade_status_calibration'
                    ELSE 'discovery_calibration'
                END AS expected_report_family,
                ROW_NUMBER() OVER (
                    PARTITION BY CASE
                        WHEN run_id LIKE 'calibration_report_%'
                        THEN 'grade_status_calibration'
                        ELSE 'discovery_calibration'
                    END
                    ORDER BY created_at DESC, id DESC
                ) AS family_rank
            FROM calibration_reports
        )
        SELECT *
        FROM ranked
        WHERE family_rank = 1
        ORDER BY created_at DESC, run_id DESC
        """,
    ).fetchall()
    reports = []
    for row in rows:
        try:
            report_path = Path(str(row["report_path"] or "")).expanduser()
            if not report_path.is_absolute():
                continue
            integrity_status, report_bytes = authorized_artifact_bytes(report_path)
            if integrity_status != FINANCIAL_INTEGRITY_PASS or report_bytes is None:
                continue
            payload = json.loads(report_bytes)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("run_id") or "") != str(row["run_id"]):
            continue
        report_family = str(payload.get("report_family") or "")
        if report_family != str(row["expected_report_family"]):
            continue
        source_run_ids = payload.get("source_run_ids")
        if not isinstance(source_run_ids, list) or any(
            not isinstance(run_id, str) or not run_id.strip() for run_id in source_run_ids
        ):
            continue
        source_outcomes = payload.get("source_outcomes")
        if not isinstance(source_outcomes, list) or any(
            not isinstance(binding, dict)
            or not serialized_outcome_binding_is_decision_eligible(binding)
            for binding in source_outcomes
        ):
            continue
        bound_source_run_ids = sorted(
            {
                str(binding["outcome_state"]["run_id"])
                for binding in source_outcomes
                if isinstance(binding.get("outcome_state"), dict)
            }
        )
        if bound_source_run_ids != source_run_ids:
            continue
        if report_family == "discovery_calibration":
            source_candidates = payload.get("source_candidates")
            if not discovery_candidate_bindings_are_current(conn, source_candidates):
                continue
        report = {
            "run_id": row["run_id"],
            "as_of_date": row["as_of_date"],
            "created_at": row["created_at"],
            "report_path": row["report_path"],
            "report_family": report_family,
        }
        if report_family == "grade_status_calibration":
            overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
            report.update(
                {
                    "headline_hit_metric": payload.get("headline_hit_metric"),
                    "overall": {
                        "n": overall.get("n"),
                        "hit_rate": overall.get("hit_rate"),
                        "excess_hit_rate": overall.get("excess_hit_rate"),
                        "avg_return": overall.get("avg_return"),
                        "avg_excess": overall.get("avg_excess"),
                        "median_return": overall.get("median_return"),
                        "target_hit_rate": overall.get("target_hit_rate"),
                    },
                    "by_grade": payload.get("by_grade")
                    if isinstance(payload.get("by_grade"), dict)
                    else {},
                    "by_status": payload.get("by_status")
                    if isinstance(payload.get("by_status"), dict)
                    else {},
                }
            )
        else:
            report.update(
                {
                    "candidate_count": payload.get("candidate_count"),
                    "outcomes_total": payload.get("outcomes_total"),
                    "matched_outcomes": payload.get("matched_outcomes"),
                    "closed_outcomes_total": payload.get("closed_outcomes_total"),
                    "average_closed_return_pct": payload.get("average_closed_return_pct"),
                    "hit_rate_by_stage": payload.get("hit_rate_by_stage"),
                    "hit_rate_by_whale_fit_bucket": payload.get("hit_rate_by_whale_fit_bucket"),
                    "hit_rate_by_evidence_strength_bucket": payload.get(
                        "hit_rate_by_evidence_strength_bucket"
                    ),
                    "threshold_suggestions": payload.get("threshold_suggestions"),
                }
            )
        reports.append(report)
    return reports


def outcomes_deck(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "scoreboard": method_scoreboard(conn),
        "returns": realized_returns(conn),
        "journal": journal(conn),
        "calibration": calibration(conn),
    }
