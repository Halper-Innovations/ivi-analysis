"""Segment CLOSED ticker_outcomes into a calibration report.

This is the headline of the calibration/outcome loop: once forward returns have
been resolved (see ``return_resolver``), this module aggregates every CLOSED
``ticker_outcomes`` row into hit-rate / average-return / median-return /
average-excess-vs-benchmark statistics, segmented by conviction grade
(ACTIONABLE / WATCHLIST_ONLY / AVOID / DATA_INCOMPLETE) and by price status
(ACTIVE / DEPLOY_READY / ...), plus an overall roll-up.

Hit metrics (all three are reported):

* ``excess_hit_rate`` — fraction of rows that beat the benchmark (excess > 0).
  This is the HEADLINE metric.
* ``hit_rate`` — fraction with a positive absolute return (realized > 0).
* ``target_hit_rate`` — fraction that reached the buy target (the
  wait-for-correction trigger fired).

AVOID semantics: for AVOID-grade / PASS-decision rows a "hit"
means the name UNDERPERFORMED, so the realized- and excess-sign tests are
inverted for those rows (a correct AVOID is one that fell / lagged). The
target-reached test is never inverted (a correction either happened or did not).
``avg_return`` / ``median_return`` / ``avg_excess`` are reported as raw margins
(never sign-flipped) so the magnitude stays interpretable.

Pure aggregation lives in ``_segment_stats`` so it is unit-testable with literal
expected values. The builder writes a JSON + Markdown artifact into
``cfg.calibration_dir`` and upserts a ``calibration_reports`` row keyed by
``calibration_report_<as_of_date>`` (re-running upserts in place, never appends).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    authorized_artifact_bytes,
    write_typed_financial_authorization,
)
from app.config import get_config
from app.db import get_db, utc_now_iso
from app.outcomes.lineage import (
    outcome_row_is_decision_eligible,
    serialize_outcome_binding,
    serialized_outcome_binding_is_decision_eligible,
)

# Run-id prefix for grade/status calibration reports, distinct from the
# discovery-calibration family ("calibration_<timestamp>") that shares the
# calibration_reports table. Readers filter on this prefix to disambiguate.
REPORT_RUN_ID_PREFIX = "calibration_report_"

# Grades whose "hit" is an underperformance (a correct AVOID/PASS is one that
# fell or lagged the benchmark). The realized/excess sign tests invert for these.
_INVERTED_GRADES = {"AVOID"}


def _segment_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of CLOSED outcome rows into segment statistics.

    Each row is a mapping carrying ``realized_return_pct``, ``excess_return_pct``
    (may be None when no benchmark resolved), ``reached_buy_target`` (1/0/None),
    and ``invert`` (True for AVOID-family rows whose hit is an underperformance).

    A single pass over ``rows`` builds the realized list, so ``n`` and every
    hit-rate share the same denominator; ``avg_excess`` / ``excess_hit_rate`` use
    only the rows whose excess is present, and ``target_hit_rate`` only the rows
    whose reached flag is present. All rates/returns are rounded to 4 dp.
    """
    realized: list[float] = []
    realized_hits = 0
    excess: list[float] = []
    excess_hits = 0
    target_total = 0
    target_hits = 0

    for r in rows:
        invert = bool(r.get("invert"))
        rv = r.get("realized_return_pct")
        if isinstance(rv, (int, float)):
            rv = float(rv)
            realized.append(rv)
            if (rv < 0) if invert else (rv > 0):
                realized_hits += 1
        ev = r.get("excess_return_pct")
        if isinstance(ev, (int, float)):
            ev = float(ev)
            excess.append(ev)
            if (ev < 0) if invert else (ev > 0):
                excess_hits += 1
        tv = r.get("reached_buy_target")
        if isinstance(tv, (int, float, bool)):
            target_total += 1
            if int(tv) == 1:
                target_hits += 1

    n = len(realized)
    if n == 0:
        return {
            "n": 0,
            "hit_rate": None,
            "excess_hit_rate": None,
            "target_hit_rate": None,
            "avg_return": None,
            "median_return": None,
            "avg_excess": None,
        }
    return {
        "n": n,
        "hit_rate": round(realized_hits / n, 4),
        "excess_hit_rate": round(excess_hits / len(excess), 4) if excess else None,
        "target_hit_rate": round(target_hits / target_total, 4) if target_total else None,
        "avg_return": round(sum(realized) / n, 4),
        "median_return": round(statistics.median(realized), 4),
        "avg_excess": round(sum(excess) / len(excess), 4) if excess else None,
    }


def recompute_grade_status_calibration_claims(
    source_outcomes: Any,
) -> dict[str, Any] | None:
    """Recompute every published aggregate from serialized exact outcome states."""

    if not isinstance(source_outcomes, list):
        return None
    by_grade_rows: dict[str, list[dict[str, Any]]] = {}
    by_status_rows: dict[str, list[dict[str, Any]]] = {}
    by_decision_basis_rows: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for binding in source_outcomes:
        if not isinstance(binding, Mapping):
            return None
        state = binding.get("outcome_state")
        if not isinstance(state, Mapping):
            return None
        if str(state.get("outcome_status") or "").upper() != "CLOSED":
            return None
        if str(state.get("entry_price_source") or "") in {
            "historical_backtest",
            "pearl_validation",
        }:
            return None
        grade = str(state.get("grade") or "UNKNOWN")
        status = str(state.get("status") or "UNKNOWN")
        decision_basis = str(state.get("decision_basis") or "LEGACY_UNSPECIFIED")
        record = {
            "realized_return_pct": state.get("realized_return_pct"),
            "excess_return_pct": state.get("excess_return_pct"),
            "reached_buy_target": state.get("reached_buy_target"),
            "invert": grade.upper() in _INVERTED_GRADES,
        }
        all_rows.append(record)
        by_grade_rows.setdefault(grade, []).append(record)
        by_status_rows.setdefault(status, []).append(record)
        by_decision_basis_rows.setdefault(decision_basis, []).append(record)
    return {
        "by_grade": {key: _segment_stats(rows) for key, rows in by_grade_rows.items()},
        "by_status": {key: _segment_stats(rows) for key, rows in by_status_rows.items()},
        "by_decision_basis": {
            key: _segment_stats(rows) for key, rows in by_decision_basis_rows.items()
        },
        "overall": _segment_stats(all_rows),
    }


def _to_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Calibration Report — {report.get('as_of_date')}",
        "",
        f"Generated: {report.get('generated_at')}",
        "Headline hit metric: excess return > 0 (beat benchmark)",
        "Also reported: absolute return > 0 (hit_rate); reached buy target (target_hit_rate).",
        "AVOID rows invert the realized/excess sign tests (a correct AVOID underperforms).",
        "",
        "## Overall",
        "",
        f"- n: {report['overall']['n']}",
        f"- excess_hit_rate (headline): {report['overall']['excess_hit_rate']}",
        f"- hit_rate (absolute): {report['overall']['hit_rate']}",
        f"- target_hit_rate: {report['overall']['target_hit_rate']}",
        f"- avg_return: {report['overall']['avg_return']}",
        f"- median_return: {report['overall']['median_return']}",
        f"- avg_excess: {report['overall']['avg_excess']}",
        "",
        "## By grade",
        "",
    ]
    for grade in sorted(report["by_grade"]):
        s = report["by_grade"][grade]
        lines.append(
            f"- {grade}: n={s['n']} excess_hit_rate={s['excess_hit_rate']} "
            f"hit_rate={s['hit_rate']} target_hit_rate={s['target_hit_rate']} "
            f"avg_return={s['avg_return']} median_return={s['median_return']} "
            f"avg_excess={s['avg_excess']}"
        )
    lines.extend(["", "## By status", ""])
    for status in sorted(report["by_status"]):
        s = report["by_status"][status]
        lines.append(
            f"- {status}: n={s['n']} excess_hit_rate={s['excess_hit_rate']} "
            f"hit_rate={s['hit_rate']} target_hit_rate={s['target_hit_rate']} "
            f"avg_return={s['avg_return']} median_return={s['median_return']} "
            f"avg_excess={s['avg_excess']}"
        )
    lines.extend(["", "## By decision basis", ""])
    for basis in sorted(report.get("by_decision_basis", {})):
        s = report["by_decision_basis"][basis]
        lines.append(
            f"- {basis}: n={s['n']} excess_hit_rate={s['excess_hit_rate']} "
            f"hit_rate={s['hit_rate']} target_hit_rate={s['target_hit_rate']} "
            f"avg_return={s['avg_return']} median_return={s['median_return']} "
            f"avg_excess={s['avg_excess']}"
        )
    return "\n".join(lines) + "\n"


def build_calibration_report(
    as_of_date: str,
    *,
    db_path: str | Path | None = None,  # accepted for API symmetry; writes go via configured db
) -> dict[str, Any]:
    """Build, persist, and return a grade/status-segmented calibration report.

    Reads every CLOSED ``ticker_outcomes`` row, segments by ``grade`` and
    ``status`` (plus an overall roll-up), writes a JSON + Markdown artifact into
    ``cfg.calibration_dir``, and upserts a ``calibration_reports`` row keyed by
    ``calibration_report_<as_of_date>``.

    Measurement rows are excluded by ``entry_price_source``: historical backtest
    rows and validation-panel rows (``pearl_validation``) are readings of
    the record, not emitted verdicts, so they must not shift grade calibration.
    """
    cfg = get_config()
    cfg.calibration_dir.mkdir(parents=True, exist_ok=True)

    by_grade_rows: dict[str, list[dict[str, Any]]] = {}
    by_status_rows: dict[str, list[dict[str, Any]]] = {}
    by_decision_basis_rows: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    source_run_ids: set[str] = set()
    source_outcomes: list[dict[str, Any]] = []

    with get_db() as conn:
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
              AND (entry_price_source IS NULL
                   OR entry_price_source NOT IN ('historical_backtest', 'pearl_validation'))
            """
        ).fetchall()

    for row in rows:
        run_id = str(row["run_id"] or "")
        if not outcome_row_is_decision_eligible(row):
            continue
        serialized_binding = serialize_outcome_binding(row)
        if serialized_binding is None:
            continue
        source_run_ids.add(run_id)
        source_outcomes.append(serialized_binding)
        grade = row["grade"] or "UNKNOWN"
        status = row["status"] or "UNKNOWN"
        decision_basis = row["decision_basis"] or "LEGACY_UNSPECIFIED"
        record = {
            "realized_return_pct": row["realized_return_pct"],
            "excess_return_pct": row["excess_return_pct"],
            "reached_buy_target": row["reached_buy_target"],
            "invert": grade.upper() in _INVERTED_GRADES,
        }
        all_rows.append(record)
        by_grade_rows.setdefault(grade, []).append(record)
        by_status_rows.setdefault(status, []).append(record)
        by_decision_basis_rows.setdefault(decision_basis, []).append(record)

    report: dict[str, Any] = {
        "run_id": f"{REPORT_RUN_ID_PREFIX}{as_of_date}",
        "report_family": "grade_status_calibration",
        "as_of_date": as_of_date,
        "generated_at": utc_now_iso(),
        "headline_hit_metric": "excess_return_pct>0",
        "hit_metrics": ["excess_return_pct>0", "realized_return_pct>0", "reached_buy_target"],
        "avoid_sign_inverted": True,
        "source_run_ids": sorted(source_run_ids),
        "source_outcomes": sorted(
            source_outcomes,
            key=lambda binding: (
                str(binding["outcome_state"]["run_id"]),
                str(binding["outcome_state"]["ticker"]),
                int(binding["outcome_state"]["id"]),
            ),
        ),
        "by_grade": {g: _segment_stats(r) for g, r in by_grade_rows.items()},
        "by_status": {s: _segment_stats(r) for s, r in by_status_rows.items()},
        "by_decision_basis": {
            basis: _segment_stats(records) for basis, records in by_decision_basis_rows.items()
        },
        "overall": _segment_stats(all_rows),
    }

    json_path = cfg.calibration_dir / f"{report['run_id']}.json"
    md_path = cfg.calibration_dir / f"{report['run_id']}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    write_typed_financial_authorization(
        json_path,
        md_path,
        artifact_type="grade_status_calibration",
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO calibration_reports(run_id, as_of_date, report_json, report_path, created_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                as_of_date=excluded.as_of_date,
                report_json=excluded.report_json,
                report_path=excluded.report_path,
                created_at=excluded.created_at
            """,
            (
                report["run_id"],
                as_of_date,
                json.dumps(report, sort_keys=True),
                str(json_path),
                report["generated_at"],
            ),
        )

    return report


def latest_grade_status_report() -> dict[str, Any] | None:
    """Return the most recent grade/status calibration report.

    Filters on the ``calibration_report_`` run-id prefix so the discovery-
    calibration family that shares ``calibration_reports`` is never surfaced
    here (and vice versa — see ``app.discovery.calibration``).
    """
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT run_id, report_json, report_path, created_at
            FROM calibration_reports
            WHERE run_id LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (f"{REPORT_RUN_ID_PREFIX}%",),
        ).fetchone()
    if not row:
        return None
    report_path = Path(str(row["report_path"] or "")).expanduser()
    if not report_path.is_absolute():
        return None
    integrity_status, report_bytes = authorized_artifact_bytes(report_path)
    if integrity_status != FINANCIAL_INTEGRITY_PASS or report_bytes is None:
        return None
    try:
        payload = json.loads(report_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or str(payload.get("run_id") or "") != str(row["run_id"]):
        return None
    source_run_ids = payload.get("source_run_ids")
    if not isinstance(source_run_ids, list) or any(
        not isinstance(run_id, str) or not run_id.strip() for run_id in source_run_ids
    ):
        return None
    source_outcomes = payload.get("source_outcomes")
    if not isinstance(source_outcomes, list) or any(
        not isinstance(binding, Mapping)
        or not serialized_outcome_binding_is_decision_eligible(binding)
        for binding in source_outcomes
    ):
        return None
    bound_source_run_ids = sorted(
        {
            str(binding["outcome_state"]["run_id"])
            for binding in source_outcomes
            if isinstance(binding.get("outcome_state"), Mapping)
        }
    )
    if bound_source_run_ids != source_run_ids:
        return None
    payload["report_path"] = str(report_path)
    payload["created_at"] = row["created_at"]
    payload["run_id"] = row["run_id"]
    return payload
