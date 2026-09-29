from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import get_db, utc_now_iso
from app.report.claims import validate_claims_have_evidence_or_derivation
from app.util.hashing import sha256_file


def _load_delta(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metric_claims(delta: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for idx, item in enumerate(delta.get("financial_metric_changes", [])):
        new_value = item.get("to")
        old_value = item.get("from")
        if isinstance(new_value, (int, float)):
            claims.append(
                {
                    "claim_id": f"delta_metric_new_{idx}",
                    "label": f"{item.get('metric')}_new",
                    "value": float(new_value),
                    "citations": item.get("citations") or [],
                    "derived_from": item.get("derived_from") or [],
                }
            )
        if isinstance(old_value, (int, float)):
            claims.append(
                {
                    "claim_id": f"delta_metric_old_{idx}",
                    "label": f"{item.get('metric')}_old",
                    "value": float(old_value),
                    "citations": item.get("citations") or [],
                    "derived_from": item.get("derived_from") or [],
                }
            )
    return claims


def build_delta_memo_for_ticker(run_id: str, ticker: str, *, only_changed: bool = False) -> Path | None:
    cfg = get_config()
    delta_path = cfg.deltas_dir / f"{ticker}_{run_id}.json"
    if not delta_path.exists():
        return None
    delta = _load_delta(delta_path)
    if not delta:
        return None
    changed = bool(delta.get("changed"))
    if only_changed and not changed:
        return None

    claims = _metric_claims(delta)
    claim_failures = validate_claims_have_evidence_or_derivation(claims)
    failed_labels = {
        failure.split("numeric claim missing citation/derivation:", 1)[1].strip()
        for failure in claim_failures
        if "numeric claim missing citation/derivation:" in failure
    }

    memo_dir = cfg.delta_memos_dir / f"{ticker}_{run_id}"
    memo_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {ticker} Delta Memo ({run_id})",
        "",
        "## Baseline" if delta.get("baseline") else "## Run-over-Run Delta",
        "- Baseline run with no previous comparison." if delta.get("baseline") else f"- Previous run: {delta.get('prev_run_id')}",
        "",
        "## What's New",
    ]

    new_filings = delta.get("new_filings") or []
    if new_filings:
        for filing in new_filings:
            lines.append(
                f"- {filing.get('form_type')} filed {filing.get('filing_date')} ({filing.get('accession')})"
                f" [source]({filing.get('source_url')})"
            )
    else:
        lines.append("- No new filings detected.")

    signal_changes = delta.get("research_signal_changes") or {}
    added_flags = signal_changes.get("added_flags") or []
    removed_flags = signal_changes.get("removed_flags") or []
    if added_flags or removed_flags:
        lines.append(f"- Added risk flags: {', '.join(added_flags) if added_flags else 'NONE'}")
        lines.append(f"- Removed risk flags: {', '.join(removed_flags) if removed_flags else 'NONE'}")

    lines.extend(["", "## Why It Matters"])
    if new_filings:
        lines.append("- New filings can reset assumptions and expected catalyst timing.")
    if added_flags:
        lines.append("- New negative flags can reduce confidence and increase downside risk.")
    if not new_filings and not added_flags:
        lines.append("- No material change drivers were detected in this cycle.")

    lines.extend(["", "## Updated Risks"])
    if added_flags:
        for flag in added_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- No newly triggered risk flags.")

    lines.extend(["", "## Financial Metric Deltas"])
    for idx, item in enumerate(delta.get("financial_metric_changes", [])):
        metric = str(item.get("metric") or f"metric_{idx}")
        old = item.get("from")
        new = item.get("to")
        old_label = f"{metric}_old"
        new_label = f"{metric}_new"
        if old_label in failed_labels or new_label in failed_labels:
            lines.append(f"- {metric}: Metric unavailable due to missing evidence.")
            continue
        lines.append(f"- {metric}: {old} -> {new}")

    lines.extend(["", "## Next Checks"])
    lines.append("- Re-run valuation and watch for follow-up 8-K/10-Q updates.")
    lines.append("- Confirm whether new risk flags persist in the next filing cycle.")

    lines.extend(["", "## Claims Trace"])
    for claim in claims:
        if claim["label"] in failed_labels:
            continue
        lines.append(
            f"- {claim['label']}={claim['value']} "
            f"(citations={len(claim.get('citations') or [])}, derived_from={','.join(claim.get('derived_from') or [])})"
        )
    if failed_labels:
        lines.append("- Suppressed numeric claims due to missing citation/derivation trace.")

    md_path = memo_dir / "delta_memo.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "run_id": run_id,
        "ticker": ticker,
        "delta_path": str(delta_path),
        "changed": changed,
        "baseline": bool(delta.get("baseline")),
        "claims": [c for c in claims if c["label"] not in failed_labels],
        "suppressed_claims": sorted(failed_labels),
        "created_at": utc_now_iso(),
        "memo_md_path": str(md_path),
    }
    json_path = memo_dir / "delta_memo.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    memo_hash = sha256_file(md_path)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO delta_memos(ticker, run_id, changed, memo_path, memo_hash, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, run_id) DO UPDATE SET
                changed=excluded.changed,
                memo_path=excluded.memo_path,
                memo_hash=excluded.memo_hash,
                created_at=excluded.created_at
            """,
            (ticker, run_id, 1 if changed else 0, str(md_path), memo_hash, utc_now_iso()),
        )
    return md_path


def build_delta_memos(run_id: str, *, top_n: int = 20, only_changed: bool = False) -> dict[str, Any]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT s.ticker, s.total_score
            FROM scores s
            WHERE s.run_id = ?
            ORDER BY s.total_score DESC, s.ticker ASC
            LIMIT ?
            """,
            (run_id, max(1, int(top_n))),
        ).fetchall()

    built = 0
    out_paths: list[str] = []
    for row in rows:
        path = build_delta_memo_for_ticker(run_id, row["ticker"], only_changed=only_changed)
        if path:
            built += 1
            out_paths.append(str(path))

    return {
        "run_id": run_id,
        "built": built,
        "only_changed": only_changed,
        "paths": out_paths,
    }
