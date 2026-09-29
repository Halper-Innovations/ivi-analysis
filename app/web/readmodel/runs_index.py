"""Filesystem indexer for sector-run artifacts.

Sector runs persist as JSON on disk (there is no ``sector_runs`` table):
``data/outputs/runs/autonomous_sector/<run_id>/autonomous_sector_run.json``
for the v1 pipeline, and nested ``<root>/<sector>/autonomous_sector_run.json``
trees for v2 replays. This module walks those trees into the UI-owned store
``data/outputs/webui/ui.db`` with an mtime+size cache so a request never
re-parses hundreds of artifacts, and it quarantines unparseable artifacts
visibly (``parse_error``) instead of skipping them silently.

``ui.db`` is the only database this package writes; the books of record stay
behind the read-only layer in :mod:`app.web.readmodel.db`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    authorized_artifact_bytes,
    financial_integrity_manifest_is_usable,
)
from app.config import AppConfig, get_config

RUN_ARTIFACT_FILENAME = "autonomous_sector_run.json"
REPORT_FILENAME = "autonomous_sector_report.md"

UI_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS run_index (
    path TEXT PRIMARY KEY,
    slug TEXT,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    indexed_at TEXT NOT NULL,
    parse_error TEXT,
    artifact_sha256 TEXT,
    integrity_status TEXT NOT NULL DEFAULT 'UNAUDITED',
    decision_eligible INTEGER NOT NULL DEFAULT 0,
    run_id TEXT,
    kind TEXT NOT NULL DEFAULT 'autonomous_sector',
    contract_version TEXT,
    pipeline_version TEXT,
    sector TEXT,
    market_cap_focus TEXT,
    scan_family TEXT,
    as_of_date TEXT,
    created_at TEXT,
    completed_at TEXT,
    status TEXT,
    execution_status TEXT,
    decision_status TEXT,
    final_verdict TEXT,
    selected_ticker TEXT,
    no_selection_reason TEXT,
    examined_count INTEGER,
    disposition_counts_json TEXT,
    cost_microdollars INTEGER,
    report_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_index_sector ON run_index(sector);
CREATE INDEX IF NOT EXISTS idx_run_index_created ON run_index(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_run_index_slug ON run_index(slug);
"""


def ui_db_path(cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    return Path(cfg.outputs_dir) / "webui" / "ui.db"


def open_ui_db(cfg: AppConfig | None = None) -> sqlite3.Connection:
    path = ui_db_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    # ui.db is a rebuildable cache: when the schema gains a column, drop the
    # stale table (before the schema script, whose indexes name new columns)
    # and let the next refresh re-walk the artifacts instead of migrating.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(run_index)")}
    required_columns = {
        "slug",
        "artifact_sha256",
        "integrity_status",
        "decision_eligible",
    }
    if columns and not required_columns.issubset(columns):
        conn.executescript("DROP TABLE run_index;")
    conn.executescript(UI_SCHEMA_SQL)
    return conn


def discover_artifacts(runs_dir: Path) -> list[Path]:
    """All sector-run artifacts two levels below ``runs_dir``.

    Covers both layouts observed on disk — ``autonomous_sector/<run_id>/``
    (v1) and ``all_sector_v2_*/<sector>/`` (v2) — while excluding
    ``_archive`` trees.
    """

    if not runs_dir.exists():
        return []
    found = [p for p in runs_dir.glob(f"*/*/{RUN_ARTIFACT_FILENAME}") if "_archive" not in p.parts]
    return sorted(found)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_slug(artifact: Path) -> str:
    """Stable, unique run reference derived from the artifact's directory.

    Embedded ``run_id`` values are NOT unique on disk (v2 smoke/replay trees
    re-emit the same run_id under different roots), so links key on the two
    path components the discovery glob guarantees: ``<root>/<leaf>``.
    """

    return f"{artifact.parent.parent.name}/{artifact.parent.name}"


def _extract_cost_microdollars(payload: dict[str, Any]) -> int | None:
    lane_usage = payload.get("lane_usage")
    if not isinstance(lane_usage, dict) or not lane_usage:
        return None
    aggregate = lane_usage.get("aggregate")
    if isinstance(aggregate, dict) and isinstance(aggregate.get("cost_microdollars"), int):
        return aggregate["cost_microdollars"]
    lanes = lane_usage.get("lanes")
    if isinstance(lanes, dict):
        totals = [
            lane.get("cost_microdollars")
            for lane in lanes.values()
            if isinstance(lane, dict) and isinstance(lane.get("cost_microdollars"), int)
        ]
        if totals:
            return sum(totals)
    return None


def _extract_examined_count(payload: dict[str, Any]) -> int | None:
    for key in ("candidate_dispositions", "company_packets", "relative_ranking"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return len(value)
    return None


def _extract_disposition_counts(payload: dict[str, Any]) -> str | None:
    dispositions = payload.get("candidate_dispositions")
    if not isinstance(dispositions, list) or not dispositions:
        return None
    counts: dict[str, int] = {}
    for item in dispositions:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("last_completed_stage") or "UNKNOWN")
        counts[stage] = counts.get(stage, 0) + 1
    return json.dumps(dict(sorted(counts.items())), sort_keys=True) if counts else None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_summary(
    payload: dict[str, Any],
    artifact_path: Path,
    *,
    integrity_status: str,
) -> dict[str, Any]:
    report = artifact_path.parent / REPORT_FILENAME
    return {
        "run_id": _str_or_none(payload.get("run_id")) or artifact_path.parent.name,
        "contract_version": _str_or_none(payload.get("contract_version")),
        "pipeline_version": _str_or_none(payload.get("pipeline_version")) or "v1",
        "sector": _str_or_none(payload.get("sector")),
        "market_cap_focus": _str_or_none(payload.get("market_cap_focus")),
        "scan_family": _str_or_none(payload.get("scan_family")) or "normal",
        "as_of_date": _str_or_none(payload.get("as_of_date")),
        "created_at": _str_or_none(payload.get("created_at")),
        "completed_at": _str_or_none(payload.get("completed_at")),
        "status": _str_or_none(payload.get("status")),
        "execution_status": _str_or_none(payload.get("execution_status")),
        "decision_status": _str_or_none(payload.get("decision_status")),
        "final_verdict": _str_or_none(payload.get("final_verdict")),
        "selected_ticker": _str_or_none(payload.get("selected_ticker")),
        "no_selection_reason": _str_or_none(payload.get("no_selection_reason")),
        "examined_count": _extract_examined_count(payload),
        "disposition_counts_json": _extract_disposition_counts(payload),
        "cost_microdollars": _extract_cost_microdollars(payload),
        "report_path": str(report) if report.exists() else None,
        "integrity_status": integrity_status,
        "decision_eligible": int(integrity_status == FINANCIAL_INTEGRITY_PASS),
    }


_SUMMARY_COLUMNS = (
    "run_id",
    "contract_version",
    "pipeline_version",
    "sector",
    "market_cap_focus",
    "scan_family",
    "as_of_date",
    "created_at",
    "completed_at",
    "status",
    "execution_status",
    "decision_status",
    "final_verdict",
    "selected_ticker",
    "no_selection_reason",
    "examined_count",
    "disposition_counts_json",
    "cost_microdollars",
    "report_path",
    "integrity_status",
    "decision_eligible",
)


def _summary_from_current_authorized_bytes(
    row: sqlite3.Row | dict[str, Any],
    *,
    manifest_usable: bool,
) -> dict[str, Any]:
    """Return cache metadata plus a summary re-derived from exact current bytes."""

    payload = dict(row)
    artifact_path = Path(str(payload.get("path") or ""))
    for column in _SUMMARY_COLUMNS:
        payload[column] = None
    payload["artifact_sha256"] = None
    payload["run_id"] = artifact_path.parent.name or None
    payload["decision_eligible"] = 0
    if not manifest_usable:
        payload["integrity_status"] = "UNAUDITED"
        payload["parse_error"] = "financial_integrity:UNAUDITED"
        return payload

    current_status, current_bytes = authorized_artifact_bytes(artifact_path)
    payload["integrity_status"] = current_status
    payload["parse_error"] = f"financial_integrity:{current_status}"
    if current_status == FINANCIAL_INTEGRITY_PASS and current_bytes is not None:
        try:
            current_payload = json.loads(current_bytes.decode("utf-8"))
            if not isinstance(current_payload, dict):
                raise ValueError(
                    f"artifact root is {type(current_payload).__name__}, expected object"
                )
            current_summary = _extract_summary(
                current_payload,
                artifact_path,
                integrity_status=current_status,
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            payload["parse_error"] = f"{type(exc).__name__}: {exc}"
        else:
            payload.update(current_summary)
            payload["artifact_sha256"] = hashlib.sha256(current_bytes).hexdigest()
            payload["parse_error"] = None
            payload["decision_eligible"] = 1
    return payload


def refresh_index(
    ui_conn: sqlite3.Connection, *, runs_dir: Path | None = None, cfg: AppConfig | None = None
) -> dict[str, int]:
    """Incrementally reconcile ``run_index`` with the artifacts on disk.

    Current bytes are re-authorized on every refresh; unchanged authorized
    content avoids reparsing. Vanished files drop out of the index and parse
    failures are recorded rather than skipped.
    """

    cfg = cfg or get_config()
    runs_dir = runs_dir if runs_dir is not None else Path(cfg.runs_dir)
    cached = {
        row["path"]: (row["artifact_sha256"], row["integrity_status"])
        for row in ui_conn.execute("SELECT path, artifact_sha256, integrity_status FROM run_index")
    }
    summary = {"discovered": 0, "parsed": 0, "unchanged": 0, "quarantined": 0, "removed": 0}
    seen: set[str] = set()
    for artifact in discover_artifacts(runs_dir):
        key = str(artifact)
        seen.add(key)
        summary["discovered"] += 1
        stat = artifact.stat()
        integrity_status, authorized_bytes = authorized_artifact_bytes(artifact)
        artifact_sha256 = (
            hashlib.sha256(authorized_bytes).hexdigest()
            if integrity_status == FINANCIAL_INTEGRITY_PASS and authorized_bytes is not None
            else None
        )
        if cached.get(key) == (artifact_sha256, integrity_status):
            summary["unchanged"] += 1
            continue
        parse_error: str | None = None
        extracted: dict[str, Any] = {column: None for column in _SUMMARY_COLUMNS}
        if integrity_status != FINANCIAL_INTEGRITY_PASS or authorized_bytes is None:
            parse_error = f"financial_integrity:{integrity_status}"
            extracted["run_id"] = artifact.parent.name
            extracted["integrity_status"] = integrity_status
            extracted["decision_eligible"] = 0
            summary["quarantined"] += 1
        else:
            try:
                payload = json.loads(authorized_bytes.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(f"artifact root is {type(payload).__name__}, expected object")
                extracted = _extract_summary(
                    payload,
                    artifact,
                    integrity_status=integrity_status,
                )
                summary["parsed"] += 1
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                extracted["run_id"] = artifact.parent.name
                extracted["integrity_status"] = integrity_status
                extracted["decision_eligible"] = 0
                summary["quarantined"] += 1
        ui_conn.execute(
            f"""
            INSERT OR REPLACE INTO run_index (
                path, slug, mtime_ns, size, indexed_at, parse_error,
                artifact_sha256, kind,
                {", ".join(_SUMMARY_COLUMNS)}
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'autonomous_sector',
                      {", ".join("?" for _ in _SUMMARY_COLUMNS)})
            """,
            (
                key,
                artifact_slug(artifact),
                stat.st_mtime_ns,
                stat.st_size,
                _utc_now_iso(),
                parse_error,
                artifact_sha256,
                *[extracted[column] for column in _SUMMARY_COLUMNS],
            ),
        )
    vanished = set(cached) - seen
    for key in vanished:
        ui_conn.execute("DELETE FROM run_index WHERE path = ?", (key,))
        summary["removed"] += 1
    ui_conn.commit()
    return summary


def list_indexed_runs(
    ui_conn: sqlite3.Connection,
    *,
    sector: str | None = None,
    market_cap_focus: str | None = None,
    final_verdict: str | None = None,
    scan_family: str | None = None,
    pipeline_version: str | None = None,
    decision_eligible: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    manifest_usable = financial_integrity_manifest_is_usable()
    if decision_eligible is True and not manifest_usable:
        return []
    requested_fields = {
        "sector": sector,
        "market_cap_focus": market_cap_focus,
        "final_verdict": final_verdict,
        "scan_family": scan_family,
        "pipeline_version": pipeline_version,
    }
    # The cache is only a path inventory. Filtering, sorting, and returned
    # decision fields must never use its mutable summary columns.
    rows = ui_conn.execute("SELECT * FROM run_index").fetchall()
    payloads: list[dict[str, Any]] = []
    for row in rows:
        payload = _summary_from_current_authorized_bytes(
            row,
            manifest_usable=manifest_usable,
        )
        if any(
            value is not None and payload.get(column) != value
            for column, value in requested_fields.items()
        ):
            continue
        if decision_eligible is not None and (
            bool(payload["decision_eligible"]) is not decision_eligible
        ):
            continue
        payloads.append(payload)
    payloads.sort(
        key=lambda payload: (
            payload.get("created_at") is not None,
            str(payload.get("created_at") or ""),
            str(payload.get("run_id") or ""),
        ),
        reverse=True,
    )
    return payloads[offset : offset + limit]
