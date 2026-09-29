"""Render a completed discover sweep as a markdown report."""

from __future__ import annotations

import json
from pathlib import Path

from app.discover.persistence import (
    get_sweep,
    list_stage2_results,
    list_stage3_results,
    list_stage4_results,
)


def render_sweep_report(db_path: str | Path, sweep_id: str) -> str:
    sweep = get_sweep(db_path, sweep_id)
    if sweep is None:
        return f"# Discover Sweep Report\n\nSweep {sweep_id} not found."

    raw_s2 = list_stage2_results(db_path, sweep_id)
    raw_s3 = list_stage3_results(db_path, sweep_id)
    raw_s4 = list_stage4_results(db_path, sweep_id)
    financial_state_invalid = (
        str(sweep.get("status") or "").strip().lower() == "invalid_financial_input"
    )
    if financial_state_invalid:
        s2: list[dict] = []
        s3: list[dict] = []
        s4: list[dict] = []
    else:
        s2 = list_stage2_results(
            db_path,
            sweep_id,
            require_financial_scope=True,
        )
        s3 = list_stage3_results(
            db_path,
            sweep_id,
            require_financial_scope=True,
        )
        s4 = list_stage4_results(
            db_path,
            sweep_id,
            require_financial_scope=True,
        )
    suppressed_counts = {
        "stage2": len(raw_s2) - len(s2),
        "stage3": len(raw_s3) - len(s3),
        "stage4": len(raw_s4) - len(s4),
    }

    s2_keeps = [r for r in s2 if r["decision"] == "KEEP"]
    s2_drops = [r for r in s2 if r["decision"] == "DROP"]
    s2_errors = [r for r in s2 if r["decision"] == "ERROR"]

    s3_by_ticker = {r["ticker"]: r for r in s3}

    buy_candidates = [r for r in s4 if r["verdict"] == "BUY"]
    watch_list = [r for r in s4 if r["verdict"] == "WATCH"]
    pass_list = [r for r in s4 if r["verdict"] == "PASS"]

    parts: list[str] = []
    parts.append("# Discover Sweep Report")
    parts.append("")
    parts.append(f"**Sweep ID:** `{sweep_id}`")
    parts.append(f"**Started:** {sweep['started_at']}")
    parts.append(f"**Finished:** {sweep.get('finished_at') or '(in progress)'}")
    parts.append(f"**Status:** {sweep['status']}")
    parts.append(f"**Universe size:** {sweep['universe_size']}")
    parts.append(f"**Limit applied:** {sweep.get('limit_applied') or 'none'}")
    parts.append(f"**Total cost:** ${sweep['total_cost_usd']:.4f}")
    if float(sweep.get("unpublished_attempt_cost_usd") or 0.0) > 0.0:
        parts.append(
            "**Unpublished paid-attempt exposure:** "
            f"${float(sweep['unpublished_attempt_cost_usd']):.4f} "
            f"(reserved/ambiguous: ${float(sweep.get('reserved_attempt_cost_usd') or 0.0):.4f})"
        )
    parts.append("")
    if financial_state_invalid:
        parts.append(
            "**Financial integrity:** INVALID_FINANCIAL_INPUT — all decision content is suppressed."
        )
        parts.append("")
    elif any(suppressed_counts.values()):
        parts.append(
            "**Financial integrity:** legacy or unbound results suppressed "
            f"(Stage 2: {suppressed_counts['stage2']}, "
            f"Stage 3: {suppressed_counts['stage3']}, "
            f"Stage 4: {suppressed_counts['stage4']})."
        )
        parts.append("")
    parts.append("## Funnel shape")
    parts.append("")
    parts.append(
        f"- Stage 2 classified: **{len(s2)}** tickers "
        f"({len(s2_keeps)} KEEP / {len(s2_drops)} DROP / {len(s2_errors)} ERROR)"
    )
    parts.append(f"- Stage 3 researched: **{len(s3)}** tickers")
    parts.append(f"- Stage 4 deep loops: **{len(s4)}** tickers")
    parts.append("")

    parts.append("## BUY candidates")
    parts.append("")
    if not buy_candidates:
        parts.append("No BUY candidates surfaced in this sweep.")
        parts.append("")
    else:
        for r in buy_candidates:
            key_findings = json.loads(r["key_findings_json"] or "[]")
            falsifiers = json.loads(r["falsifiers_json"] or "[]")
            parts.append(f"### {r['ticker']} — BUY ({r['confidence']})")
            parts.append("")
            parts.append(f"**Thesis:** {r['thesis']}")
            parts.append("")
            if key_findings:
                parts.append("**Key findings:**")
                for k in key_findings:
                    parts.append(f"- {k}")
                parts.append("")
            if falsifiers:
                parts.append("**Falsifiers:**")
                for f in falsifiers:
                    parts.append(f"- {f}")
                parts.append("")
            parts.append(
                f"*Turns: {r['num_turns']}, cost: ${r['cost_usd']:.4f}, "
                f"termination: {r['termination_reason']}*"
            )
            parts.append("")

    parts.append("## WATCH list")
    parts.append("")
    if not watch_list:
        parts.append("No WATCH verdicts.")
        parts.append("")
    else:
        for r in watch_list:
            parts.append(f"- **{r['ticker']}** ({r['confidence']}): {r['thesis']}")
        parts.append("")

    parts.append("## Stage 4 PASS list")
    parts.append("")
    if not pass_list:
        parts.append("No Stage 4 results.")
        parts.append("")
    else:
        for r in pass_list:
            parts.append(f"- **{r['ticker']}** ({r['confidence']}): {r['thesis']}")
        parts.append("")

    parts.append("## Stage 2 classifier detail")
    parts.append("")
    if s2_keeps:
        parts.append("### KEEP decisions")
        parts.append("")
        for r in s2_keeps:
            s3r = s3_by_ticker.get(r["ticker"])
            s3_label = f" → S3: {s3r['verdict']} ({s3r['confidence']})" if s3r else ""
            parts.append(f"- **{r['ticker']}** ({r['confidence']}): {r['reason']}{s3_label}")
        parts.append("")
    if s2_drops:
        parts.append("### DROP decisions")
        parts.append("")
        for r in s2_drops:
            parts.append(f"- **{r['ticker']}** ({r['confidence']}): {r['reason']}")
        parts.append("")
    if s2_errors:
        parts.append("### ERROR decisions")
        parts.append("")
        for r in s2_errors:
            parts.append(f"- **{r['ticker']}**: {r.get('error') or r.get('reason') or 'unknown'}")
        parts.append("")

    return "\n".join(parts)
