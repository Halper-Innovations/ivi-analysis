from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.queue import dead_letter_count
from app.config import get_config
from app.db import get_db
from app.ops.gating import build_gating_report


def generate_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{ts}"


def get_run_directory(run_id: str) -> Path:
    cfg = get_config()
    return cfg.runs_dir / run_id


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_ticker_artifacts(
    conn,
    *,
    run_dir: Path,
    ticker: str,
    as_of_date: str,
    run_id: str,
) -> dict[str, str]:
    cfg = get_config()
    artifact: dict[str, str] = {}

    score_row = conn.execute(
        """
        SELECT as_of_date
        FROM scores
        WHERE ticker = ? AND run_id = ? AND as_of_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (ticker, run_id, as_of_date),
    ).fetchone()
    if score_row is None:
        score_row = conn.execute(
            """
            SELECT as_of_date
            FROM scores
            WHERE ticker = ? AND run_id = ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker, run_id),
        ).fetchone()
    score_as_of = score_row["as_of_date"] if score_row else as_of_date

    packet_row = conn.execute(
        """
        SELECT packet_path
        FROM evidence_packets
        WHERE ticker = ? AND as_of_date = ?
        LIMIT 1
        """,
        (ticker, score_as_of),
    ).fetchone()
    if packet_row:
        src = Path(packet_row["packet_path"])
        if src.exists():
            dst = run_dir / "evidence_packets" / src.name
            _copy_file(src, dst)
            artifact["packet_path"] = str(dst.relative_to(run_dir))

    research_row = conn.execute(
        """
        SELECT packet_path
        FROM research_packets
        WHERE ticker = ? AND run_id = ? AND as_of_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (ticker, run_id, score_as_of),
    ).fetchone()
    if research_row:
        src = Path(research_row["packet_path"])
        if src.exists():
            dst = run_dir / "research" / src.name
            _copy_file(src, dst)
            artifact["research_path"] = str(dst.relative_to(run_dir))

    analysis_row = conn.execute(
        """
        SELECT output_type, output_path
        FROM analyst_outputs
        WHERE ticker = ? AND as_of_date <= ? AND output_type IN ('analysis_report', 'analysis_report_markdown')
        ORDER BY as_of_date DESC, created_at DESC
        """,
        (ticker, score_as_of),
    ).fetchall()
    latest_analysis_by_type: dict[str, str] = {}
    for row in analysis_row:
        output_type = str(row["output_type"])
        if output_type not in latest_analysis_by_type:
            latest_analysis_by_type[output_type] = str(row["output_path"])

    report_json_path = latest_analysis_by_type.get("analysis_report")
    if report_json_path:
        src = Path(report_json_path)
        if src.exists():
            dst = run_dir / "analysis" / src.name
            _copy_file(src, dst)
            artifact["analysis_report_path"] = str(dst.relative_to(run_dir))

    report_md_path = latest_analysis_by_type.get("analysis_report_markdown")
    if report_md_path:
        src = Path(report_md_path)
        if src.exists():
            dst = run_dir / "analysis" / src.name
            _copy_file(src, dst)
            artifact["analysis_report_md_path"] = str(dst.relative_to(run_dir))

    memo_row = conn.execute(
        """
        SELECT memo_path
        FROM memos
        WHERE ticker = ? AND run_id = ? AND as_of_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (ticker, run_id, score_as_of),
    ).fetchone()
    if memo_row:
        src_md = Path(memo_row["memo_path"])
        if src_md.exists():
            memo_dir = src_md.parent
            dst_dir = run_dir / "memos" / memo_dir.name
            dst_md = dst_dir / src_md.name
            _copy_file(src_md, dst_md)
            artifact["memo_path"] = str(dst_md.relative_to(run_dir))
            src_json = memo_dir / "memo.json"
            if src_json.exists():
                dst_json = dst_dir / "memo.json"
                _copy_file(src_json, dst_json)
                artifact["memo_json_path"] = str(dst_json.relative_to(run_dir))

    gaps_src = cfg.gaps_dir / f"{ticker}_{run_id}.json"
    if gaps_src.exists():
        gaps_dst = run_dir / "gaps" / gaps_src.name
        _copy_file(gaps_src, gaps_dst)
        artifact["gaps_path"] = str(gaps_dst.relative_to(run_dir))

    synthesis_row = conn.execute(
        """
        SELECT packet_path
        FROM synthesis_packets
        WHERE ticker = ? AND run_id = ? AND as_of_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (ticker, run_id, score_as_of),
    ).fetchone()
    if synthesis_row:
        src = Path(synthesis_row["packet_path"])
        if src.exists():
            dst = run_dir / "synthesis" / src.name
            _copy_file(src, dst)
            artifact["synthesis_path"] = str(dst.relative_to(run_dir))

    return artifact


def _write_scores_artifacts(conn, *, run_dir: Path, tickers: list[str], as_of_date: str, run_id: str) -> dict[str, str]:
    if not tickers:
        return {}

    placeholders = ",".join("?" for _ in tickers)
    rows = conn.execute(
        f"""
        SELECT ticker, as_of_date, run_id, total_score, decision, subscores_json, reasons_json, is_candidate, is_publishable
        FROM scores
        WHERE ticker IN ({placeholders}) AND run_id = ?
        ORDER BY as_of_date DESC, total_score DESC, ticker ASC
        """,
        (*tickers, run_id),
    ).fetchall()

    # Keep latest effective row per ticker for run-scoped snapshot artifacts.
    latest_rows: dict[str, Any] = {}
    for row in rows:
        ticker = str(row["ticker"]).upper()
        existing = latest_rows.get(ticker)
        if existing is None:
            latest_rows[ticker] = row
            continue
        existing_key = (str(existing["as_of_date"] or ""), float(existing["total_score"] or 0))
        candidate_key = (str(row["as_of_date"] or ""), float(row["total_score"] or 0))
        if candidate_key >= existing_key:
            latest_rows[ticker] = row
    rows = sorted(latest_rows.values(), key=lambda r: (-float(r["total_score"] or 0), str(r["ticker"])))

    scores_payload = {
        "as_of_date": as_of_date,
        "tickers": tickers,
        "scores": [
            {
                "ticker": row["ticker"],
                "as_of_date": row["as_of_date"],
                "run_id": row["run_id"],
                "total_score": row["total_score"],
                "decision": row["decision"],
                "is_candidate": bool(row["is_candidate"]),
                "is_publishable": bool(row["is_publishable"]),
                "subscores": json.loads(row["subscores_json"]),
                "reasons": json.loads(row["reasons_json"]),
            }
            for row in rows
        ],
    }

    rankings_payload = {
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rankings": [
            {
                "ticker": row["ticker"],
                "as_of_date": row["as_of_date"],
                "run_id": row["run_id"],
                "total_score": row["total_score"],
                "decision": row["decision"],
                "is_candidate": bool(row["is_candidate"]),
                "is_publishable": bool(row["is_publishable"]),
                "reasons": json.loads(row["reasons_json"]),
            }
            for row in rows
        ],
    }

    scores_path = run_dir / "rankings" / "scores.json"
    rankings_path = run_dir / "rankings" / "rankings.json"
    _write_json(scores_path, scores_payload)
    _write_json(rankings_path, rankings_payload)

    return {
        "scores_path": str(scores_path.relative_to(run_dir)),
        "rankings_path": str(rankings_path.relative_to(run_dir)),
    }


def _load_index() -> list[dict[str, Any]]:
    cfg = get_config()
    payload = _read_json(cfg.runs_index_path, [])
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return payload["runs"]
    return []


def _save_index(entries: list[dict[str, Any]]) -> None:
    cfg = get_config()
    _write_json(cfg.runs_index_path, entries)


def list_runs(limit: int = 10) -> list[dict[str, Any]]:
    rows = sorted(_load_index(), key=lambda r: r.get("generated_at", ""), reverse=True)
    if limit > 0:
        return rows[:limit]
    return rows


def get_run_report(run_id: str) -> dict[str, Any] | None:
    run_dir = get_run_directory(run_id)
    if not run_dir.exists():
        return None

    entries = _load_index()
    entry = next((row for row in entries if row.get("run_id") == run_id), None)

    manifest = _read_json(run_dir / "run_manifest.json", {})
    gating = _read_json(run_dir / "gating_report.json", {})

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "index_entry": entry,
        "manifest": manifest,
        "gating_report": gating,
        "gating_report_json_path": str(run_dir / "gating_report.json"),
        "gating_report_csv_path": str(run_dir / "gating_report.csv"),
    }


def _append_index_entry(entry: dict[str, Any]) -> None:
    rows = _load_index()
    rows = [row for row in rows if row.get("run_id") != entry.get("run_id")]
    rows.append(entry)
    _save_index(rows)


def finalize_run_outputs(
    *,
    run_id: str,
    as_of_date: str,
    tickers_targeted: list[str],
    with_research: bool,
    dead_letter_before: int,
    manifest_path: Path,
) -> dict[str, Any]:
    cfg = get_config()
    run_dir = get_run_directory(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_dst = run_dir / "run_manifest.json"
    if manifest_path.exists():
        _copy_file(manifest_path, manifest_dst)
    manifest = _read_json(manifest_dst, {})

    artifact_paths: dict[str, dict[str, str]] = {}
    with get_db() as conn:
        for ticker in tickers_targeted:
            artifact_paths[ticker] = _copy_ticker_artifacts(
                conn,
                run_dir=run_dir,
                ticker=ticker,
                as_of_date=as_of_date,
                run_id=run_id,
            )
            artifact_paths[ticker]["manifest_path"] = "run_manifest.json"

        rankings_paths = _write_scores_artifacts(
            conn,
            run_dir=run_dir,
            tickers=tickers_targeted,
            as_of_date=as_of_date,
            run_id=run_id,
        )

    gating_json, gating_csv, gating_rows = build_gating_report(
        run_id=run_id,
        as_of_date=as_of_date,
        tickers=tickers_targeted,
        run_dir=run_dir,
        with_research=with_research,
        artifact_paths=artifact_paths,
    )

    dead_letter_after = dead_letter_count()
    dead_letter_delta = dead_letter_after - dead_letter_before

    tickers_processed = [
        row["ticker"]
        for row in gating_rows
        if any(
            [
                row.get("ingested"),
                row.get("parsed"),
                row.get("fundamentals"),
                row.get("valuation"),
                row.get("research"),
                row.get("evidence_packet"),
                row.get("scored"),
                row.get("memo_built"),
            ]
        )
    ]
    memos_count = sum(1 for row in gating_rows if row.get("memo_built"))
    candidates_count = sum(
        1
        for row in gating_rows
        if row.get("scored") and str(row.get("decision", "")).upper() not in {"ABSTAIN", "UNKNOWN"}
    )

    universe = manifest.get("universe", {}) if isinstance(manifest, dict) else {}
    entry = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of_date,
        "universe_id": universe.get("universe_id"),
        "snapshot_hash": universe.get("snapshot_hash"),
        "tickers_targeted": tickers_targeted,
        "tickers_processed": tickers_processed,
        "memos_count": memos_count,
        "candidates_count": candidates_count,
        "dead_letter_delta": dead_letter_delta,
        "config_hash": manifest.get("config_hash") if isinstance(manifest, dict) else None,
        "git_commit": manifest.get("git_commit") if isinstance(manifest, dict) else None,
        "price_provider": cfg.price_provider,
        "safe_mode": cfg.safe_mode,
        "run_dir": str(run_dir),
        "gating_report_json": str(gating_json),
        "gating_report_csv": str(gating_csv),
        "scores_path": rankings_paths.get("scores_path"),
        "rankings_path": rankings_paths.get("rankings_path"),
    }
    _append_index_entry(entry)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "gating_report_json": str(gating_json),
        "gating_report_csv": str(gating_csv),
        "memos_count": memos_count,
        "candidates_count": candidates_count,
        "dead_letter_delta": dead_letter_delta,
        "tickers_targeted": tickers_targeted,
        "tickers_processed": tickers_processed,
    }
