from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    authorized_artifact_bytes,
    write_typed_financial_authorization,
)
from app.config import get_config
from app.db import get_db, utc_now_iso
from app.discovery.lineage import (
    discovery_candidate_bindings_are_current,
    serialize_discovery_candidate_binding,
)
from app.outcomes.lineage import (
    outcome_row_is_decision_eligible,
    serialize_outcome_binding,
    serialized_outcome_binding_is_decision_eligible,
)


def _bucket(value: float | int | None, *, edges: list[float], labels: list[str]) -> str:
    if value is None:
        return "UNKNOWN"
    v = float(value)
    for idx, edge in enumerate(edges):
        if v < edge:
            return labels[idx]
    return labels[-1]


def _load_target_discovery_runs(run_id: str | None, last_n: int) -> list[str]:
    with get_db() as conn:
        if run_id:
            row = conn.execute(
                "SELECT run_id FROM discovery_runs WHERE run_id = ? LIMIT 1", (run_id,)
            ).fetchone()
            if row:
                return [run_id]
            rows = conn.execute(
                """
                SELECT DISTINCT discovery_run_id
                FROM ticker_outcomes
                WHERE run_id = ? OR deep_run_id = ?
                ORDER BY discovery_run_id ASC
                """,
                (run_id, run_id),
            ).fetchall()
            runs = [str(r["discovery_run_id"]) for r in rows if r["discovery_run_id"]]
            if runs:
                return sorted(set(runs))
        rows = conn.execute(
            """
            SELECT run_id
            FROM discovery_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, int(last_n)),),
        ).fetchall()
    return [str(row["run_id"]) for row in rows]


def _load_candidates(discovery_run_ids: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    if not discovery_run_ids:
        return {}
    placeholders = ",".join("?" for _ in discovery_run_ids)
    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, ticker, run_id, discovery_score, payload_json, created_at,
                   publication_receipt_path, publication_receipt_sha256
            FROM discovery_candidates
            WHERE run_id IN ({placeholders})
            """,
            tuple(discovery_run_ids),
        ).fetchall()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        binding = serialize_discovery_candidate_binding(row)
        if binding is None:
            continue
        state = binding["candidate_state"]
        key = (str(state["ticker"]), str(state["run_id"]))
        if key in out:
            raise RuntimeError(
                "refusing calibration with duplicate discovery candidate "
                f"{state['run_id']}:{state['ticker']}"
            )
        out[key] = {
            "payload": dict(state["payload"]),
            "binding": binding,
        }
    return out


def _load_outcomes(discovery_run_ids: list[str]) -> list[dict[str, Any]]:
    if not discovery_run_ids:
        return []
    placeholders = ",".join("?" for _ in discovery_run_ids)
    with get_db() as conn:
        rows = conn.execute(
            f"""
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
              AND discovery_run_id IN ({placeholders})
            ORDER BY updated_at DESC, id DESC
            """,
            tuple(discovery_run_ids),
        ).fetchall()
    return [dict(row) for row in rows]


def _hit_rate(counts: dict[str, int]) -> float | None:
    closed = int(counts.get("closed", 0))
    if closed <= 0:
        return None
    return round((int(counts.get("positive", 0)) / closed) * 100.0, 2)


def _summarize_counts(counter_map: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(counter_map.keys()):
        row = counter_map[key]
        out[key] = {
            "closed": int(row.get("closed", 0)),
            "positive": int(row.get("positive", 0)),
            "negative": int(row.get("negative", 0)),
            "hit_rate_pct": _hit_rate(row),
        }
    return out


def _threshold_suggestions(
    *,
    by_stage: dict[str, dict[str, Any]],
    by_whale_fit: dict[str, dict[str, Any]],
    by_evidence_strength: dict[str, dict[str, Any]],
) -> list[str]:
    suggestions: list[str] = []
    adv = by_stage.get("ADVANCE_TO_DEEP", {})
    watch = by_stage.get("WATCHLIST_ONLY", {})
    adv_hit = adv.get("hit_rate_pct")
    watch_hit = watch.get("hit_rate_pct")
    if (
        isinstance(adv_hit, (int, float))
        and isinstance(watch_hit, (int, float))
        and adv_hit < watch_hit
    ):
        suggestions.append(
            "ADVANCE_TO_DEEP underperforms WATCHLIST_ONLY; consider raising evidence_strength floor."
        )
    if (by_whale_fit.get("18+", {}) or {}).get("hit_rate_pct") is None:
        suggestions.append(
            "Insufficient closed outcomes in high whale_fit bucket; collect more outcomes before threshold changes."
        )
    if (by_evidence_strength.get("7+", {}) or {}).get("hit_rate_pct") is not None and (
        by_evidence_strength.get("0-3", {}) or {}
    ).get("hit_rate_pct") is not None:
        high = float(by_evidence_strength["7+"]["hit_rate_pct"] or 0.0)
        low = float(by_evidence_strength["0-3"]["hit_rate_pct"] or 0.0)
        if high < low:
            suggestions.append(
                "High evidence_strength does not outperform low bucket; audit explainability/reason quality."
            )
    if not suggestions:
        suggestions.append("No threshold change suggested; continue collecting CLOSED outcomes.")
    return suggestions


def recompute_discovery_calibration_claims(
    source_outcomes: Any,
    source_candidates: Any,
) -> dict[str, Any] | None:
    """Recompute every published calibration claim from exact serialized states."""

    if not isinstance(source_outcomes, list) or not isinstance(source_candidates, list):
        return None
    candidates: dict[tuple[str, str], Mapping[str, Any]] = {}
    for binding in source_candidates:
        if not isinstance(binding, Mapping):
            return None
        state = binding.get("candidate_state")
        if not isinstance(state, Mapping) or not isinstance(state.get("payload"), Mapping):
            return None
        key = (str(state.get("ticker") or "").upper(), str(state.get("run_id") or ""))
        if not all(key) or key in candidates:
            return None
        candidates[key] = state["payload"]

    outcomes: list[Mapping[str, Any]] = []
    for binding in source_outcomes:
        if not isinstance(binding, Mapping):
            return None
        state = binding.get("outcome_state")
        if not isinstance(state, Mapping):
            return None
        outcomes.append(state)

    by_stage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"closed": 0, "positive": 0, "negative": 0}
    )
    by_whale_fit: dict[str, dict[str, int]] = defaultdict(
        lambda: {"closed": 0, "positive": 0, "negative": 0}
    )
    by_evidence_strength: dict[str, dict[str, int]] = defaultdict(
        lambda: {"closed": 0, "positive": 0, "negative": 0}
    )
    positive_reason_counts: Counter[str] = Counter()
    negative_reason_counts: Counter[str] = Counter()
    positive_gap_counts: Counter[str] = Counter()
    negative_gap_counts: Counter[str] = Counter()
    closed_returns: list[float] = []
    matched = 0
    for row in outcomes:
        key = (
            str(row.get("ticker") or "").upper(),
            str(row.get("discovery_run_id") or ""),
        )
        candidate = candidates.get(key)
        if candidate is None:
            return None
        matched += 1
        if str(row.get("outcome_status") or "OPEN").upper() != "CLOSED":
            continue
        realized = row.get("realized_return_pct")
        positive = isinstance(realized, (int, float)) and float(realized) > 0
        negative = isinstance(realized, (int, float)) and float(realized) <= 0
        if isinstance(realized, (int, float)):
            closed_returns.append(float(realized))
        stage = str(candidate.get("stage") or "UNKNOWN")
        whale_fit_bucket = _bucket(
            candidate.get("whale_fit_score")
            if isinstance(candidate.get("whale_fit_score"), (int, float))
            else None,
            edges=[10.0, 18.0],
            labels=["0-9", "10-17", "18+"],
        )
        evidence_bucket = _bucket(
            candidate.get("evidence_strength_score")
            if isinstance(candidate.get("evidence_strength_score"), (int, float))
            else None,
            edges=[4.0, 7.0],
            labels=["0-3", "4-6", "7+"],
        )
        for bucket_map, bucket_key in (
            (by_stage, stage),
            (by_whale_fit, whale_fit_bucket),
            (by_evidence_strength, evidence_bucket),
        ):
            bucket_map[bucket_key]["closed"] += 1
            if positive:
                bucket_map[bucket_key]["positive"] += 1
            elif negative:
                bucket_map[bucket_key]["negative"] += 1
        reasons = [str(item) for item in (candidate.get("key_reasons") or [])][:8]
        gaps = [str(item) for item in (candidate.get("gaps") or [])][:8]
        if positive:
            positive_reason_counts.update(reasons)
            positive_gap_counts.update(gaps)
        elif negative:
            negative_reason_counts.update(reasons)
            negative_gap_counts.update(gaps)

    by_stage_summary = _summarize_counts(by_stage)
    by_whale_summary = _summarize_counts(by_whale_fit)
    by_evidence_summary = _summarize_counts(by_evidence_strength)
    return {
        "candidate_count": len(candidates),
        "outcomes_total": len(outcomes),
        "excluded_outcomes_total": 0,
        "matched_outcomes": matched,
        "closed_outcomes_total": len(closed_returns),
        "average_closed_return_pct": (
            round(sum(closed_returns) / len(closed_returns), 4) if closed_returns else None
        ),
        "hit_rate_by_stage": by_stage_summary,
        "hit_rate_by_whale_fit_bucket": by_whale_summary,
        "hit_rate_by_evidence_strength_bucket": by_evidence_summary,
        "top_reason_counts": {
            "positive": [
                list(item)
                for item in sorted(
                    positive_reason_counts.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
            "negative": [
                list(item)
                for item in sorted(
                    negative_reason_counts.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
        },
        "top_gap_counts": {
            "positive": [
                list(item)
                for item in sorted(
                    positive_gap_counts.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
            "negative": [
                list(item)
                for item in sorted(
                    negative_gap_counts.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
        },
        "threshold_suggestions": _threshold_suggestions(
            by_stage=by_stage_summary,
            by_whale_fit=by_whale_summary,
            by_evidence_strength=by_evidence_summary,
        ),
    }


def _candidate_bindings_are_current(bindings: Any) -> bool:
    with get_db() as conn:
        return discovery_candidate_bindings_are_current(conn, bindings)


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Calibration Report {report['run_id']}",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Discovery runs considered: {', '.join(report['discovery_run_ids']) if report['discovery_run_ids'] else 'NONE'}",
        f"- Outcomes total: {report['outcomes_total']} (closed: {report['closed_outcomes_total']})",
        "",
        "## Hit Rate By Stage",
        "",
        "| Stage | Closed | Positive | Negative | Hit Rate % |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in (report.get("hit_rate_by_stage") or {}).items():
        lines.append(
            f"| {key} | {row.get('closed', 0)} | {row.get('positive', 0)} | {row.get('negative', 0)} | {row.get('hit_rate_pct', 'N/A')} |"
        )
    lines.extend(
        [
            "",
            "## Suggestions",
            "",
        ]
    )
    for item in report.get("threshold_suggestions", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def run_calibration(*, run_id: str | None = None, last_n: int = 5) -> dict[str, Any]:
    cfg = get_config()
    cfg.calibration_dir.mkdir(parents=True, exist_ok=True)

    target_discovery_run_ids = _load_target_discovery_runs(run_id, last_n)
    candidates = _load_candidates(target_discovery_run_ids)
    selected_outcomes = _load_outcomes(target_discovery_run_ids)
    discovery_run_ids = sorted({candidate_run_id for _ticker, candidate_run_id in candidates})
    outcomes: list[dict[str, Any]] = []
    source_outcomes: list[dict[str, Any]] = []
    source_candidates: dict[tuple[str, str], dict[str, Any]] = {
        key: dict(record["binding"]) for key, record in candidates.items()
    }
    source_run_ids: set[str] = set()
    for row in selected_outcomes:
        if not outcome_row_is_decision_eligible(row):
            continue
        binding = serialize_outcome_binding(row)
        if binding is None:
            continue
        candidate_key = (
            str(row.get("ticker") or "").upper(),
            str(row.get("discovery_run_id") or ""),
        )
        if candidate_key not in candidates:
            continue
        outcomes.append(row)
        source_outcomes.append(binding)
        source_run_ids.add(str(row.get("run_id") or ""))

    by_stage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"closed": 0, "positive": 0, "negative": 0}
    )
    by_whale_fit: dict[str, dict[str, int]] = defaultdict(
        lambda: {"closed": 0, "positive": 0, "negative": 0}
    )
    by_evidence_strength: dict[str, dict[str, int]] = defaultdict(
        lambda: {"closed": 0, "positive": 0, "negative": 0}
    )
    positive_reason_counts: Counter[str] = Counter()
    negative_reason_counts: Counter[str] = Counter()
    positive_gap_counts: Counter[str] = Counter()
    negative_gap_counts: Counter[str] = Counter()

    closed_returns: list[float] = []
    matched = 0
    for row in outcomes:
        key = (str(row.get("ticker") or "").upper(), str(row.get("discovery_run_id") or ""))
        candidate_record = candidates.get(key)
        if candidate_record is None:
            raise RuntimeError(
                f"refusing calibration with missing exact discovery candidate {key[1]}:{key[0]}"
            )
        candidate = candidate_record["payload"]
        matched += 1
        status = str(row.get("outcome_status") or "OPEN").upper()
        if status != "CLOSED":
            continue

        realized = row.get("realized_return_pct")
        positive = isinstance(realized, (int, float)) and float(realized) > 0
        negative = isinstance(realized, (int, float)) and float(realized) <= 0
        if isinstance(realized, (int, float)):
            closed_returns.append(float(realized))

        stage = str(candidate.get("stage") or "UNKNOWN")
        whale_fit_bucket = _bucket(
            candidate.get("whale_fit_score")
            if isinstance(candidate.get("whale_fit_score"), (int, float))
            else None,
            edges=[10.0, 18.0],
            labels=["0-9", "10-17", "18+"],
        )
        evidence_bucket = _bucket(
            candidate.get("evidence_strength_score")
            if isinstance(candidate.get("evidence_strength_score"), (int, float))
            else None,
            edges=[4.0, 7.0],
            labels=["0-3", "4-6", "7+"],
        )

        for bucket_map, bucket_key in [
            (by_stage, stage),
            (by_whale_fit, whale_fit_bucket),
            (by_evidence_strength, evidence_bucket),
        ]:
            bucket_map[bucket_key]["closed"] += 1
            if positive:
                bucket_map[bucket_key]["positive"] += 1
            elif negative:
                bucket_map[bucket_key]["negative"] += 1

        reasons = [str(x) for x in (candidate.get("key_reasons") or [])][:8]
        gaps = [str(x) for x in (candidate.get("gaps") or [])][:8]
        if positive:
            positive_reason_counts.update(reasons)
            positive_gap_counts.update(gaps)
        elif negative:
            negative_reason_counts.update(reasons)
            negative_gap_counts.update(gaps)

    now = utc_now_iso()
    report_id = run_id or f"calibration_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    report: dict[str, Any] = {
        "run_id": report_id,
        "report_family": "discovery_calibration",
        "target_run_id": run_id,
        "generated_at": now,
        "discovery_run_ids": discovery_run_ids,
        "candidate_count": len(candidates),
        "outcomes_total": len(outcomes),
        # Invalid rows are excluded before report construction and cannot become
        # a source for a published numerical claim.
        "excluded_outcomes_total": 0,
        "matched_outcomes": matched,
        "closed_outcomes_total": len(closed_returns),
        "source_run_ids": sorted(source_run_ids),
        "source_outcomes": sorted(
            source_outcomes,
            key=lambda binding: (
                str(binding["outcome_state"]["run_id"]),
                str(binding["outcome_state"]["ticker"]),
                int(binding["outcome_state"]["id"]),
            ),
        ),
        "source_candidates": sorted(
            source_candidates.values(),
            key=lambda binding: (
                str(binding["candidate_state"]["run_id"]),
                str(binding["candidate_state"]["ticker"]),
                int(binding["candidate_state"]["id"]),
            ),
        ),
        "average_closed_return_pct": round(sum(closed_returns) / len(closed_returns), 4)
        if closed_returns
        else None,
        "hit_rate_by_stage": _summarize_counts(by_stage),
        "hit_rate_by_whale_fit_bucket": _summarize_counts(by_whale_fit),
        "hit_rate_by_evidence_strength_bucket": _summarize_counts(by_evidence_strength),
        "top_reason_counts": {
            "positive": sorted(positive_reason_counts.items(), key=lambda x: (-x[1], x[0]))[:10],
            "negative": sorted(negative_reason_counts.items(), key=lambda x: (-x[1], x[0]))[:10],
        },
        "top_gap_counts": {
            "positive": sorted(positive_gap_counts.items(), key=lambda x: (-x[1], x[0]))[:10],
            "negative": sorted(negative_gap_counts.items(), key=lambda x: (-x[1], x[0]))[:10],
        },
    }
    report["threshold_suggestions"] = _threshold_suggestions(
        by_stage=report["hit_rate_by_stage"],
        by_whale_fit=report["hit_rate_by_whale_fit_bucket"],
        by_evidence_strength=report["hit_rate_by_evidence_strength_bucket"],
    )

    json_path = cfg.calibration_dir / f"{report_id}.json"
    md_path = cfg.calibration_dir / f"{report_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    if not _candidate_bindings_are_current(report["source_candidates"]):
        raise RuntimeError("discovery candidate lineage changed before calibration authorization")
    authorization_path = write_typed_financial_authorization(
        json_path,
        md_path,
        artifact_type="discovery_calibration",
    )
    if not _candidate_bindings_are_current(report["source_candidates"]):
        authorization_path.unlink(missing_ok=True)
        raise RuntimeError("discovery candidate lineage changed during calibration authorization")

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
                report_id,
                None,
                json.dumps(report, sort_keys=True),
                str(json_path),
                now,
            ),
        )

    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    return report


def latest_calibration_report() -> dict[str, Any] | None:
    # The calibration_reports table is shared with the grade/status report
    # family (run_id prefix "calibration_report_"; see
    # app.calibration.calibration_report). Exclude that prefix so the discovery-
    # calibration reader never surfaces a grade/status report's by_grade/by_status shape.
    from app.calibration.calibration_report import REPORT_RUN_ID_PREFIX

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT run_id, report_json, report_path, created_at
            FROM calibration_reports
            WHERE run_id NOT LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (f"{REPORT_RUN_ID_PREFIX}%",),
        ).fetchone()
    if not row:
        return None
    payload = _authorized_report_payload(row)
    if payload is None:
        return None
    payload["report_path"] = row["report_path"]
    payload["created_at"] = row["created_at"]
    payload["run_id"] = row["run_id"]
    return payload


def load_calibration_report(run_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT report_json, report_path, created_at
            FROM calibration_reports
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if not row:
        return None
    payload = _authorized_report_payload(row, expected_run_id=run_id)
    if payload is None:
        return None
    payload["report_path"] = row["report_path"]
    payload["created_at"] = row["created_at"]
    payload["run_id"] = run_id
    return payload


def _authorized_report_payload(
    row: Mapping[str, Any] | Any,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any] | None:
    values = dict(row)
    run_id = str(values.get("run_id") or expected_run_id or "")
    if expected_run_id is not None and run_id != expected_run_id:
        return None
    report_path = Path(str(values.get("report_path") or "")).expanduser()
    if not report_path.is_absolute():
        return None
    integrity_status, report_bytes = authorized_artifact_bytes(report_path)
    if integrity_status != FINANCIAL_INTEGRITY_PASS or report_bytes is None:
        return None
    try:
        payload = json.loads(report_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("report_family") != "discovery_calibration"
        or str(payload.get("run_id") or "") != run_id
    ):
        return None
    source_run_ids = payload.get("source_run_ids")
    source_outcomes = payload.get("source_outcomes")
    if (
        not isinstance(source_run_ids, list)
        or any(
            not isinstance(source_run_id, str) or not source_run_id
            for source_run_id in source_run_ids
        )
        or not isinstance(source_outcomes, list)
        or any(
            not isinstance(binding, Mapping)
            or not serialized_outcome_binding_is_decision_eligible(binding)
            for binding in source_outcomes
        )
    ):
        return None
    bound_run_ids = sorted(
        {
            str(binding["outcome_state"]["run_id"])
            for binding in source_outcomes
            if isinstance(binding.get("outcome_state"), Mapping)
        }
    )
    if bound_run_ids != source_run_ids:
        return None
    source_candidates = payload.get("source_candidates")
    if not _candidate_bindings_are_current(source_candidates):
        return None
    discovery_run_ids = payload.get("discovery_run_ids")
    if not isinstance(discovery_run_ids, list):
        return None
    bound_discovery_run_ids = sorted(
        {str(binding["candidate_state"]["run_id"]) for binding in source_candidates}
    )
    if any(run_id not in discovery_run_ids for run_id in bound_discovery_run_ids):
        return None
    return payload
