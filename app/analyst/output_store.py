"""Persistence helpers for analyst-facing output artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.autonomous.artifact_financial_audit import (
    PASS as FINANCIAL_INTEGRITY_PASS,
    authorized_artifact_bytes,
    financial_integrity_manifest_is_usable,
)
from app.analyst.evidence_bundle import AnalysisEvidenceBundle
from app.analyst.report_renderer import render_report
from app.analyst.thesis_contract import AnalysisReport
from app.config import get_config
from app.db import get_db, utc_now_iso
from app.util.hashing import sha256_file


@dataclass
class AnalysisOutputPaths:
    bundle_json: Path
    report_json: Path
    report_markdown: Path


def _output_dir(ticker: str, as_of_date: str) -> Path:
    cfg = get_config()
    path = cfg.analyst_outputs_dir / f"{ticker}_{as_of_date}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _upsert_output_path(
    ticker: str,
    as_of_date: str,
    output_type: str,
    path: Path,
) -> None:
    output_hash = sha256_file(path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analyst_outputs(ticker, as_of_date, output_type, output_path, output_hash, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date, output_type) DO UPDATE SET
                output_path=excluded.output_path,
                output_hash=excluded.output_hash
            """,
            (ticker, as_of_date, output_type, str(path), output_hash, utc_now_iso()),
        )


def latest_analysis_output_path(
    ticker: str,
    output_type: str,
    as_of_date: str | None = None,
) -> Path | None:
    """Return the latest indexed analyst output path for a ticker."""
    if not financial_integrity_manifest_is_usable():
        return None
    with get_db() as conn:
        return latest_eligible_analysis_output_path(
            conn,
            ticker,
            output_type,
            as_of_date=as_of_date,
        )


def latest_eligible_analysis_output_path(
    conn,
    ticker: str,
    output_type: str,
    *,
    as_of_date: str | None = None,
    exact_as_of_date: bool = False,
) -> Path | None:
    """Return the latest analyst path only when that exact row is audited PASS.

    Selection is intentionally latest-first and fail-closed. If the newest
    matching row is missing, stale, invalid, or unaudited, callers receive no
    output; an older row is never resurrected.
    """

    result = latest_eligible_analysis_output_bytes(
        conn,
        ticker,
        output_type,
        as_of_date=as_of_date,
        exact_as_of_date=exact_as_of_date,
    )
    return result[0] if result is not None else None


def latest_eligible_analysis_output_bytes(
    conn,
    ticker: str,
    output_type: str,
    *,
    as_of_date: str | None = None,
    exact_as_of_date: bool = False,
) -> tuple[Path, bytes] | None:
    """Return the newest eligible output and the exact authorized bytes.

    Decision consumers must parse these bytes instead of reopening the path
    after authorization.
    """

    if not financial_integrity_manifest_is_usable():
        return None
    if as_of_date and exact_as_of_date:
        row = conn.execute(
            """
            SELECT output_path
            FROM analyst_outputs
            WHERE ticker = ? AND output_type = ? AND as_of_date = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker.upper(), output_type, as_of_date),
        ).fetchone()
    elif as_of_date:
        row = conn.execute(
            """
            SELECT output_path
            FROM analyst_outputs
            WHERE ticker = ? AND output_type = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker.upper(), output_type, as_of_date),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT output_path
            FROM analyst_outputs
            WHERE ticker = ? AND output_type = ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker.upper(), output_type),
        ).fetchone()
    if not row:
        return None
    path = Path(row["output_path"])
    integrity_status, output_bytes = authorized_artifact_bytes(path)
    if integrity_status != FINANCIAL_INTEGRITY_PASS or output_bytes is None:
        return None
    return path, output_bytes


def latest_analysis_report(
    ticker: str,
    as_of_date: str | None = None,
) -> AnalysisReport | None:
    """Load the latest persisted AnalysisReport for a ticker."""
    if not financial_integrity_manifest_is_usable():
        return None
    with get_db() as conn:
        result = latest_eligible_analysis_output_bytes(
            conn,
            ticker,
            "analysis_report",
            as_of_date=as_of_date,
        )
    if result is None:
        return None
    _, report_bytes = result
    try:
        payload = json.loads(report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return AnalysisReport.from_dict(payload)


def persist_analysis_outputs(
    bundle: AnalysisEvidenceBundle,
    report: AnalysisReport,
) -> AnalysisOutputPaths:
    """Write the analyst bundle/report artifacts and index them in analyst_outputs."""
    out_dir = _output_dir(report.ticker, report.as_of_date)

    bundle_json = out_dir / "analysis_evidence_bundle.json"
    bundle_json.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    _upsert_output_path(report.ticker, report.as_of_date, "analysis_evidence_bundle", bundle_json)

    report_json = out_dir / "analysis_report.json"
    report_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    _upsert_output_path(report.ticker, report.as_of_date, "analysis_report", report_json)

    report_markdown = out_dir / "analysis_report.md"
    report_markdown.write_text(render_report(report), encoding="utf-8")
    _upsert_output_path(
        report.ticker, report.as_of_date, "analysis_report_markdown", report_markdown
    )

    return AnalysisOutputPaths(
        bundle_json=bundle_json,
        report_json=report_json,
        report_markdown=report_markdown,
    )
