"""Fail-closed backfill for legacy valuation rows missing artifact lineage.

The backfill does not create authority. It searches already-authorized
autonomous-sector artifacts for an exact valuation source record, verifies the
artifact bytes through the financial-integrity manifest, and only then binds
the matching mutable row. Rows without one unique verified match stay intact.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.autonomous.artifact_financial_audit import PASS, authorized_artifact_bytes
from app.config import AppConfig, get_config
from app.db import connect, utc_now_iso
from app.valuation.lineage import (
    valuation_integrity_fingerprint,
    valuation_row_is_decision_eligible,
    valuation_source_record,
)

_REQUIRED_VALUATION_COLUMNS = {
    "id",
    "ticker",
    "as_of_date",
    "method",
    "inputs_json",
    "outputs_json",
    "warnings_json",
    "created_at",
    "valuation_writer_version",
    "quality_gate_verdict",
    "confidence_class",
    "gate_reason_codes",
    "valuation_headwinds",
    "valuation_supports",
    "source_run_id",
    "source_artifact_path",
    "source_artifact_sha256",
    "financial_integrity_fingerprint",
}
_REQUIRED_HISTORY_COLUMNS = {
    "source_id",
    *(_REQUIRED_VALUATION_COLUMNS - {"id"}),
    "archived_at",
}
_SOURCE_RECORD_IDENTITY_FIELDS = ("ticker", "as_of_date", "method", "created_at")


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {str(key): row[key] for key in row.keys()}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _assert_schema(conn: sqlite3.Connection) -> None:
    valuation_columns = _table_columns(conn, "valuations")
    history_columns = _table_columns(conn, "valuations_history")
    missing_valuations = sorted(_REQUIRED_VALUATION_COLUMNS - valuation_columns)
    missing_history = sorted(_REQUIRED_HISTORY_COLUMNS - history_columns)
    if missing_valuations or missing_history:
        details: list[str] = []
        if missing_valuations:
            details.append("valuations=" + ",".join(missing_valuations))
        if missing_history:
            details.append("valuations_history=" + ",".join(missing_history))
        raise RuntimeError("valuation binding backfill schema mismatch: " + "; ".join(details))


def _artifact_paths(roots: Sequence[str | Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if root.is_file():
            paths.add(root.resolve())
        elif root.is_dir():
            paths.update(path.resolve() for path in root.rglob("autonomous_sector_run.json"))
    return sorted(paths)


def _verified_source_records(
    roots: Sequence[str | Path],
) -> tuple[
    dict[tuple[str, str, str, str], list[dict[str, Any]]],
    dict[str, int],
]:
    by_identity: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    artifact_counts: Counter[str] = Counter()
    for path in _artifact_paths(roots):
        artifact_counts["scanned"] += 1
        status, source_bytes = authorized_artifact_bytes(path)
        if status != PASS or source_bytes is None:
            artifact_counts["unverified"] += 1
            continue
        try:
            payload = json.loads(source_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            artifact_counts["unverified"] += 1
            continue
        if not isinstance(payload, dict):
            artifact_counts["unverified"] += 1
            continue
        run_id = _normalized_text(payload.get("run_id"))
        raw_records = payload.get("valuation_source_records")
        if run_id is None or not isinstance(raw_records, list):
            artifact_counts["unverified"] += 1
            continue
        artifact_sha256 = hashlib.sha256(source_bytes).hexdigest()
        accepted = 0
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping) or not isinstance(raw_record.get("row"), Mapping):
                continue
            canonical = valuation_source_record(raw_record["row"])
            if canonical is None or canonical != dict(raw_record):
                continue
            record_row = dict(canonical["row"])
            if _normalized_text(record_row.get("source_run_id")) != run_id:
                continue
            identity = tuple(
                str(record_row.get(field) or "") for field in _SOURCE_RECORD_IDENTITY_FIELDS
            )
            if not all(identity):
                continue
            by_identity[identity].append(
                {
                    "run_id": run_id,
                    "path": str(path),
                    "sha256": artifact_sha256,
                    "record_row": record_row,
                    "source_bytes": source_bytes,
                }
            )
            accepted += 1
        if accepted:
            artifact_counts["verified"] += 1
        else:
            artifact_counts["unverified"] += 1
    return by_identity, dict(sorted(artifact_counts.items()))


def _candidate_binding(
    row: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    matches: list[dict[str, Any]] = []
    existing_run_id = _normalized_text(row.get("source_run_id"))
    for candidate in candidates:
        if existing_run_id is not None and candidate["run_id"] != existing_run_id:
            continue
        record_row = candidate["record_row"]
        if any(
            row.get(field) != expected
            for field, expected in record_row.items()
            if field != "source_run_id"
        ):
            continue
        bound = dict(row)
        bound.update(
            {
                "source_run_id": candidate["run_id"],
                "source_artifact_path": candidate["path"],
                "source_artifact_sha256": candidate["sha256"],
            }
        )
        bound["financial_integrity_fingerprint"] = valuation_integrity_fingerprint(bound)
        if not valuation_row_is_decision_eligible(bound):
            continue
        matches.append({**candidate, "bound": bound})
    if not matches:
        return None, "NO_UNIQUE_VERIFIED_ARTIFACT_MATCH"
    unique = {(item["run_id"], item["path"], item["sha256"]) for item in matches}
    if len(unique) != 1:
        return None, "AMBIGUOUS_VERIFIED_ARTIFACT_MATCH"
    return matches[0], "DERIVABLE"


def _archive_row(conn: sqlite3.Connection, row: dict[str, Any], *, archived_at: str) -> None:
    conn.execute(
        """
        INSERT INTO valuations_history(
            source_id, ticker, as_of_date, method, inputs_json,
            outputs_json, warnings_json, created_at,
            valuation_writer_version, quality_gate_verdict,
            confidence_class, gate_reason_codes, valuation_headwinds,
            valuation_supports, source_run_id, source_artifact_path,
            source_artifact_sha256, financial_integrity_fingerprint,
            archived_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["id"],
            row["ticker"],
            row["as_of_date"],
            row["method"],
            row["inputs_json"],
            row["outputs_json"],
            row["warnings_json"],
            row["created_at"],
            row["valuation_writer_version"],
            row["quality_gate_verdict"],
            row["confidence_class"],
            row["gate_reason_codes"],
            row["valuation_headwinds"],
            row["valuation_supports"],
            row["source_run_id"],
            row["source_artifact_path"],
            row["source_artifact_sha256"],
            row["financial_integrity_fingerprint"],
            archived_at,
        ),
    )


def backfill_valuation_bindings(
    *,
    artifact_roots: Sequence[str | Path] | None = None,
    apply: bool = False,
    cfg: AppConfig | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Report or bind legacy unbound rows using exact authorized artifacts."""

    base_cfg = cfg or get_config()
    resolved_cfg = (
        base_cfg.model_copy(update={"db_path": Path(db_path)})
        if db_path is not None and Path(base_cfg.db_path) != Path(db_path)
        else base_cfg
    )
    roots = tuple(artifact_roots or (Path(resolved_cfg.runs_dir) / "autonomous_sector",))
    records_by_identity, artifact_counts = _verified_source_records(roots)
    conn = connect(resolved_cfg.db_path, cfg=resolved_cfg)
    conn.row_factory = sqlite3.Row
    reasons: Counter[str] = Counter()
    changed = 0
    derivable = 0
    rows: list[sqlite3.Row] = []
    try:
        _assert_schema(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM valuations
            WHERE source_run_id IS NULL
               OR source_artifact_path IS NULL
               OR source_artifact_sha256 IS NULL
               OR financial_integrity_fingerprint IS NULL
            ORDER BY ticker, as_of_date, method, created_at, id
            """
        ).fetchall()
        for raw_row in rows:
            row = _row_dict(raw_row)
            identity = tuple(str(row.get(field) or "") for field in _SOURCE_RECORD_IDENTITY_FIELDS)
            match, reason = _candidate_binding(row, records_by_identity.get(identity, ()))
            if match is None:
                reasons[reason] += 1
                continue
            derivable += 1
            reasons[reason] += 1
            if not apply:
                continue
            bound = match["bound"]
            _archive_row(conn, row, archived_at=utc_now_iso())
            cursor = conn.execute(
                """
                UPDATE valuations
                SET source_run_id = ?,
                    source_artifact_path = ?,
                    source_artifact_sha256 = ?,
                    financial_integrity_fingerprint = ?
                WHERE id = ?
                  AND source_run_id IS ?
                  AND source_artifact_path IS ?
                  AND source_artifact_sha256 IS ?
                  AND financial_integrity_fingerprint IS ?
                """,
                (
                    bound["source_run_id"],
                    bound["source_artifact_path"],
                    bound["source_artifact_sha256"],
                    bound["financial_integrity_fingerprint"],
                    row["id"],
                    row["source_run_id"],
                    row["source_artifact_path"],
                    row["source_artifact_sha256"],
                    row["financial_integrity_fingerprint"],
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"valuation row {row['id']} changed during binding backfill")
            updated = conn.execute("SELECT * FROM valuations WHERE id = ?", (row["id"],)).fetchone()
            if updated is None or not valuation_row_is_decision_eligible(updated):
                raise RuntimeError(f"valuation row {row['id']} failed post-binding eligibility")
            status, final_bytes = authorized_artifact_bytes(Path(match["path"]))
            if status != PASS or final_bytes != match["source_bytes"]:
                raise RuntimeError(f"valuation source changed during binding for row {row['id']}")
            changed += 1
        if apply:
            conn.commit()
        else:
            conn.rollback()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "mode": "apply" if apply else "dry-run",
        "scanned": len(rows),
        "derivable": derivable,
        "underivable": len(rows) - derivable,
        "changed": changed,
        "reasons": dict(sorted(reasons.items())),
        "artifacts": artifact_counts,
    }


__all__ = ["backfill_valuation_bindings"]
