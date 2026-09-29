from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.analyst.ops_snapshot import (
    OpsAnalysisSnapshot,
    load_ops_analysis_snapshot,
    snapshot_to_research_payload,
)
from app.analyst.output_store import (
    latest_eligible_analysis_output_bytes,
    latest_eligible_analysis_output_path,
)
from app.config import get_config
from app.db import get_db, utc_now_iso
from app.fundamentals.normalize import UNKNOWN
from app.logging import get_logger
from app.report.claims import build_numeric_claims, validate_claims_have_evidence_or_derivation
from app.report.gaps import write_gaps_artifact


logger = get_logger(__name__)

STATUS_OK = "OK"
STATUS_RESEARCH_INCOMPLETE = "RESEARCH_INCOMPLETE"
STATUS_FINANCIALS_INCOMPLETE = "FINANCIALS_INCOMPLETE"
STATUS_PRICE_INCOMPLETE = "PRICE_INCOMPLETE"
STATUS_PARSE_LOW_CONFIDENCE = "PARSE_LOW_CONFIDENCE"
STATUS_PACKET_MISSING = "PACKET_MISSING"


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _latest_packet_row(conn, ticker: str, as_of_date: str | None = None) -> Any | None:
    if as_of_date:
        row = conn.execute(
            """
            SELECT as_of_date, packet_path
            FROM evidence_packets
            WHERE ticker = ? AND as_of_date = ?
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
        if row:
            return row
        # A run asked "as of" a date is served the newest packet AT OR BEFORE
        # that date, never one published after it. Without the bound a
        # backdated run silently became a present-day run — no refusal, no
        # warning — and anything that walks backwards through time to test how
        # the system would have behaved was reading tomorrow's evidence.
        return conn.execute(
            """
            SELECT as_of_date, packet_path
            FROM evidence_packets
            WHERE ticker = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    return conn.execute(
        """
        SELECT as_of_date, packet_path
        FROM evidence_packets
        WHERE ticker = ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()


def _latest_score_row(
    conn,
    ticker: str,
    as_of_date: str | None = None,
    run_id: str | None = None,
) -> Any | None:
    if run_id and as_of_date:
        row = conn.execute(
            """
            SELECT as_of_date, total_score, decision
            FROM scores
            WHERE ticker = ? AND as_of_date = ? AND run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, as_of_date, run_id),
        ).fetchone()
        if row:
            return row
        return None
    if run_id:
        row = conn.execute(
            """
            SELECT as_of_date, total_score, decision
            FROM scores
            WHERE ticker = ? AND run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, run_id),
        ).fetchone()
        if row:
            return row
    if as_of_date:
        row = conn.execute(
            """
            SELECT as_of_date, total_score, decision
            FROM scores
            WHERE ticker = ? AND as_of_date = ?
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
        if row:
            return row
        # Same bound as the packet lookup above: a memo dated in the past may
        # not quote a score computed after its own date.
        return conn.execute(
            """
            SELECT as_of_date, total_score, decision
            FROM scores
            WHERE ticker = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    return conn.execute(
        """
        SELECT as_of_date, total_score, decision
        FROM scores
        WHERE ticker = ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()


def _latest_output_path(
    conn, ticker: str, output_type: str, as_of_date: str | None = None
) -> Path | None:
    return latest_eligible_analysis_output_path(
        conn,
        ticker,
        output_type,
        as_of_date=as_of_date,
        exact_as_of_date=as_of_date is not None,
    )


def _latest_output(
    conn,
    ticker: str,
    output_type: str,
    as_of_date: str | None = None,
) -> tuple[Path, bytes] | None:
    return latest_eligible_analysis_output_bytes(
        conn,
        ticker,
        output_type,
        as_of_date=as_of_date,
        exact_as_of_date=as_of_date is not None,
    )


def _load_json_bytes(data: bytes | None) -> dict[str, Any] | None:
    if data is None:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_research_path(
    conn,
    ticker: str,
    as_of_date: str | None = None,
    run_id: str | None = None,
) -> Path | None:
    if run_id and as_of_date:
        row = conn.execute(
            """
            SELECT packet_path
            FROM research_packets
            WHERE ticker = ? AND run_id = ? AND as_of_date = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, run_id, as_of_date),
        ).fetchone()
        if row:
            return Path(row["packet_path"])
        return None
    if run_id:
        row = conn.execute(
            """
            SELECT packet_path
            FROM research_packets
            WHERE ticker = ? AND run_id = ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker, run_id),
        ).fetchone()
        if row:
            return Path(row["packet_path"])
    if as_of_date:
        row = conn.execute(
            """
            SELECT packet_path
            FROM research_packets
            WHERE ticker = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker, as_of_date),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT packet_path
            FROM research_packets
            WHERE ticker = ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
    if not row:
        return None
    return Path(row["packet_path"])


def _latest_research_signals(
    conn, ticker: str, as_of_date: str, run_id: str | None = None
) -> dict[str, Any] | None:
    if run_id:
        row = conn.execute(
            """
            SELECT *
            FROM research_signals
            WHERE ticker = ? AND as_of_date = ? AND run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, as_of_date, run_id),
        ).fetchone()
        if row:
            return {
                "has_earnings_release": bool(row["has_earnings_release"]),
                "has_investor_presentation": bool(row["has_investor_presentation"]),
                "sentiment_flags": json.loads(row["sentiment_flags_json"] or "[]"),
                "key_topics": json.loads(row["key_topics_json"] or "[]"),
                "summary": json.loads(row["summary_json"] or "{}"),
            }
    row = conn.execute(
        """
        SELECT *
        FROM research_signals
        WHERE ticker = ? AND as_of_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    ).fetchone()
    if not row:
        return None
    return {
        "has_earnings_release": bool(row["has_earnings_release"]),
        "has_investor_presentation": bool(row["has_investor_presentation"]),
        "sentiment_flags": json.loads(row["sentiment_flags_json"] or "[]"),
        "key_topics": json.loads(row["key_topics_json"] or "[]"),
        "summary": json.loads(row["summary_json"] or "{}"),
    }


def _quality_gate(
    packet: dict[str, Any], hypotheses: dict[str, Any] | None, red_team: dict[str, Any] | None
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    if not packet.get("valuations"):
        failures.append("valuation outputs missing")
    if not hypotheses or not hypotheses.get("hypotheses"):
        failures.append("hypotheses output missing")
    if not red_team or not red_team.get("red_team"):
        failures.append("red-team output missing")

    cited_claims = True
    if hypotheses and hypotheses.get("hypotheses"):
        for hyp in hypotheses["hypotheses"]:
            if not hyp.get("citations"):
                cited_claims = False
                break
    if not cited_claims:
        failures.append("key claims missing citations")

    return len(failures) == 0, failures


def _latest_manifest_path(conn, as_of_date: str, run_id: str | None = None) -> Path | None:
    if run_id:
        row = conn.execute(
            """
            SELECT manifest_path
            FROM run_manifests
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if row:
            path = Path(row["manifest_path"])
            if path.exists():
                return path
    row = conn.execute(
        """
        SELECT manifest_path
        FROM run_manifests
        WHERE as_of_date <= ?
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT 1
        """,
        (as_of_date,),
    ).fetchone()
    if not row:
        return None
    path = Path(row["manifest_path"])
    if not path.exists():
        return None
    return path


def _claim_fail_labels(claim_failures: list[str]) -> set[str]:
    labels: set[str] = set()
    for failure in claim_failures:
        marker = "numeric claim missing citation/derivation:"
        if marker in failure:
            labels.add(failure.split(marker, 1)[1].strip())
    return labels


def _build_claims(
    packet: dict[str, Any],
    score_row: Any | None,
    analysis_snapshot: OpsAnalysisSnapshot | None,
) -> list[dict[str, Any]]:
    claims = build_numeric_claims(packet)
    if score_row and isinstance(score_row["total_score"], (int, float)):
        claims.append(
            {
                "claim_id": "score_total",
                "label": "total_score",
                "value": float(score_row["total_score"]),
                "unit": "score",
                "citations": [],
                "derived_from": ["scores.total_score", "score.rubric"],
            }
        )

    quality = analysis_snapshot.research_quality if analysis_snapshot is not None else None
    quality_values = {
        "coverage_score": quality.coverage_score if quality is not None else None,
        "freshness_score": quality.freshness_score if quality is not None else None,
        "gap_score": quality.gap_score if quality is not None else None,
        "overall_research_score": quality.overall_score if quality is not None else None,
    }
    for key, value in quality_values.items():
        if isinstance(value, (int, float)):
            claims.append(
                {
                    "claim_id": f"research_quality_{key}",
                    "label": f"research_quality_{key}",
                    "value": float(value),
                    "unit": "score",
                    "citations": [],
                    "derived_from": [f"analysis_report.research_quality.{key}"],
                }
            )

    return claims


def _claim_lookup(valid_claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for claim in valid_claims:
        label = claim.get("label")
        if not isinstance(label, str):
            continue
        out[label] = claim
    return out


def _value_text(label: str, claim_by_label: dict[str, dict[str, Any]]) -> str | None:
    claim = claim_by_label.get(label)
    if not claim:
        return None
    value = claim.get("value")
    if not isinstance(value, (int, float)):
        return None
    return str(value)


def _metric_line(
    title: str, claim_label: str, claim_by_label: dict[str, dict[str, Any]], gaps: list[str]
) -> str:
    value = _value_text(claim_label, claim_by_label)
    if value is not None:
        return f"- {title}: {value}"
    gaps.append(f"Metric unavailable due to missing evidence: {claim_label}")
    return f"- {title}: Metric unavailable due to missing evidence."


def _valuation_table(
    packet: dict[str, Any], claim_by_label: dict[str, dict[str, Any]], gaps: list[str]
) -> str:
    valuations = packet.get("valuations", {})
    lines = [
        "| Method | Low | Base | High | Confidence |",
        "|---|---|---|---|---|",
    ]
    table_rows = [
        ("dcf", "DCF"),
        ("epv", "EPV"),
        ("graham", "Graham"),
        ("ncav", "NCAV"),
        ("reverse_dcf", "Reverse DCF"),
    ]
    for method, label in table_rows:
        method_blob = valuations.get(method, {}) if isinstance(valuations, dict) else {}
        outputs = method_blob.get("outputs", {}) if isinstance(method_blob, dict) else {}
        if not isinstance(outputs, dict):
            outputs = method_blob if isinstance(method_blob, dict) else {}

        if method == "reverse_dcf":
            value = _value_text("reverse_dcf_implied_growth", claim_by_label)
            if value is None:
                gaps.append(
                    "Metric unavailable due to missing evidence: reverse_dcf_implied_growth"
                )
                low = base = high = "Metric unavailable due to missing evidence"
            else:
                low = base = high = value
        elif method in {"epv", "graham", "ncav"}:
            value = _value_text(f"{method}_per_share_value_per_share", claim_by_label)
            if value is None:
                gaps.append(
                    f"Metric unavailable due to missing evidence: {method}_per_share_value_per_share"
                )
                low = base = high = "Metric unavailable due to missing evidence"
            else:
                low = base = high = value
        else:
            low = _value_text(f"{method}_per_share_low", claim_by_label)
            base = _value_text(f"{method}_per_share_base", claim_by_label)
            high = _value_text(f"{method}_per_share_high", claim_by_label)
            if low is None:
                gaps.append(f"Metric unavailable due to missing evidence: {method}_per_share_low")
                low = "Metric unavailable due to missing evidence"
            if base is None:
                gaps.append(f"Metric unavailable due to missing evidence: {method}_per_share_base")
                base = "Metric unavailable due to missing evidence"
            if high is None:
                gaps.append(f"Metric unavailable due to missing evidence: {method}_per_share_high")
                high = "Metric unavailable due to missing evidence"

        conf = outputs.get("confidence") or outputs.get("status") or "LOW"
        lines.append(f"| {label} | {low} | {base} | {high} | {conf} |")
    return "\n".join(lines)


def _status_flags(
    packet: dict[str, Any],
    analysis_snapshot: OpsAnalysisSnapshot | None,
    *,
    packet_missing: bool,
) -> list[str]:
    flags: list[str] = []

    if packet_missing:
        flags.append(STATUS_PACKET_MISSING)

    quality = analysis_snapshot.research_quality if analysis_snapshot is not None else None
    if (analysis_snapshot is None) or (quality is not None and quality.incomplete):
        flags.append(STATUS_RESEARCH_INCOMPLETE)

    fundamentals = packet.get("fundamentals", {}) if isinstance(packet, dict) else {}
    critical = ["revenue", "operating_margin", "fcf", "net_debt"]
    if any(fundamentals.get(k, UNKNOWN) == UNKNOWN for k in critical):
        flags.append(STATUS_FINANCIALS_INCOMPLETE)

    reverse_inputs = packet.get("valuations", {}).get("reverse_dcf", {}).get("inputs", {})
    if reverse_inputs.get("market_price", UNKNOWN) == UNKNOWN:
        flags.append(STATUS_PRICE_INCOMPLETE)

    line_items = {
        row.get("line_item")
        for row in packet.get("financials", [])
        if isinstance(row, dict) and row.get("line_item")
    }
    required_sections = {"revenue", "operating_income", "cfo", "capex", "cash", "total_debt"}
    if not packet.get("filings_used") or required_sections.difference(line_items):
        flags.append(STATUS_PARSE_LOW_CONFIDENCE)

    if not flags:
        return [STATUS_OK]
    return sorted(set(flags))


def _append_findings_section(
    md_lines: list[str], title: str, items: list[Any], *, attr: str = "claim"
) -> None:
    if title:
        md_lines.append(title)
    if not items:
        md_lines.append("- None.")
        return
    for item in items[:6]:
        value = getattr(item, attr, "")
        if not isinstance(value, str) or not value.strip():
            continue
        md_lines.append(f"- {value}")


def _next_actions_from_snapshot(snapshot: OpsAnalysisSnapshot | None) -> list[str]:
    if snapshot is None:
        return []
    actions: list[str] = []
    for question in snapshot.open_questions:
        if question.next_step:
            actions.append(question.next_step)
    if snapshot.research_quality is not None:
        for gap in snapshot.research_quality.top_gaps:
            if gap.recommended_action:
                actions.append(gap.recommended_action)
    deduped: list[str] = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)
    return deduped[:8]


def strict_memo_publishable(ticker: str, as_of_date: str | None = None) -> tuple[bool, list[str]]:
    with get_db() as conn:
        packet_row = _latest_packet_row(conn, ticker, as_of_date=as_of_date)
        score_row = _latest_score_row(conn, ticker, as_of_date=as_of_date)
        if not packet_row:
            return False, ["evidence packet missing"]
        effective_as_of = packet_row["as_of_date"]
        hypotheses_output = _latest_output(
            conn,
            ticker,
            "hypotheses",
            effective_as_of,
        )
        red_team_output = _latest_output(
            conn,
            ticker,
            "red_team",
            effective_as_of,
        )
        hypotheses = _load_json_bytes(
            hypotheses_output[1] if hypotheses_output is not None else None
        )
        red_team = _load_json_bytes(red_team_output[1] if red_team_output is not None else None)

    packet = _load_json(Path(packet_row["packet_path"]))
    if not packet:
        return False, ["evidence packet missing"]

    analysis_snapshot = load_ops_analysis_snapshot(ticker, as_of_date=effective_as_of)
    ok, failures = _quality_gate(packet, hypotheses, red_team)
    claims = _build_claims(packet, score_row, analysis_snapshot)
    claim_failures = validate_claims_have_evidence_or_derivation(claims)
    all_failures = failures + claim_failures
    return ok and not claim_failures, all_failures


def build_memo_for_ticker(
    ticker: str,
    *,
    memo_mode: str = "strict",
    run_id: str | None = None,
    as_of_date: str | None = None,
) -> bool:
    cfg = get_config()
    memo_mode = memo_mode.strip().lower()
    if memo_mode not in {"strict", "triage"}:
        raise ValueError("memo_mode must be one of: strict, triage")

    with get_db() as conn:
        packet_row = _latest_packet_row(conn, ticker, as_of_date=as_of_date)
        score_row = _latest_score_row(conn, ticker, as_of_date=as_of_date, run_id=run_id)

        if packet_row:
            effective_as_of = packet_row["as_of_date"]
        elif score_row:
            effective_as_of = score_row["as_of_date"]
        else:
            effective_as_of = as_of_date or datetime.now(timezone.utc).date().isoformat()

        hypotheses_output = _latest_output(
            conn,
            ticker,
            "hypotheses",
            effective_as_of,
        )
        red_team_output = _latest_output(
            conn,
            ticker,
            "red_team",
            effective_as_of,
        )
        decision_output = _latest_output(
            conn,
            ticker,
            "decision",
            effective_as_of,
        )
        hypotheses_path = hypotheses_output[0] if hypotheses_output is not None else None
        red_team_path = red_team_output[0] if red_team_output is not None else None
        hypotheses = _load_json_bytes(
            hypotheses_output[1] if hypotheses_output is not None else None
        )
        red_team = _load_json_bytes(red_team_output[1] if red_team_output is not None else None)
        analyst_decision = _load_json_bytes(
            decision_output[1] if decision_output is not None else None
        )
        research_signals = _latest_research_signals(conn, ticker, effective_as_of, run_id=run_id)
        manifest_path = _latest_manifest_path(conn, effective_as_of, run_id=run_id)
        from app.events.cheapness import latest_cheapness_by_ticker
        from app.events.flags import open_flags_by_ticker

        cheapness_report = latest_cheapness_by_ticker(conn, [ticker]).get(ticker.upper())
        event_flags = open_flags_by_ticker(conn).get(ticker.upper(), [])

    packet_path = Path(packet_row["packet_path"]) if packet_row else None
    packet = _load_json(packet_path) if packet_path else None
    packet_missing = packet is None
    if not packet:
        if memo_mode == "strict":
            return False
        packet = {
            "ticker": ticker,
            "as_of_date": effective_as_of,
            "filings_used": [],
            "extracted_facts": [],
            "financials": [],
            "fundamentals": {},
            "valuations": {},
            "deltas_vs_prior_period": {},
        }

    analysis_snapshot = load_ops_analysis_snapshot(
        ticker, as_of_date=effective_as_of, run_id=run_id
    )
    research = snapshot_to_research_payload(analysis_snapshot)
    if research_signals is None and isinstance(research, dict):
        candidate = research.get("signals")
        if isinstance(candidate, dict):
            research_signals = candidate

    ok, quality_failures = _quality_gate(packet, hypotheses, red_team)
    claims = _build_claims(packet, score_row, analysis_snapshot)
    claim_failures = validate_claims_have_evidence_or_derivation(claims)
    failed_labels = _claim_fail_labels(claim_failures)

    if memo_mode == "strict":
        if not ok:
            logger.info(
                "memo_skipped_quality_gate",
                extra={
                    "stage_name": "report",
                    "stage_ticker": ticker,
                    "stage_failures": ",".join(quality_failures),
                },
            )
            return False
        if claim_failures:
            logger.info(
                "memo_skipped_claim_evidence_gate",
                extra={
                    "stage_name": "report",
                    "stage_ticker": ticker,
                    "stage_failures": ",".join(claim_failures),
                },
            )
            return False

    valid_claims = [claim for claim in claims if claim.get("label") not in failed_labels]
    claim_by_label = _claim_lookup(valid_claims)

    decision = score_row["decision"] if score_row else "Research Only"
    status_flags = _status_flags(packet, analysis_snapshot, packet_missing=packet_missing)
    if STATUS_RESEARCH_INCOMPLETE in status_flags and decision not in {"Abstain", "Research Only"}:
        decision = "RESEARCH_INCOMPLETE"

    memo_dir = cfg.memos_dir / f"{ticker}_{effective_as_of}"
    memo_dir.mkdir(parents=True, exist_ok=True)

    memo_gaps: list[str] = []
    score_text = _value_text("total_score", claim_by_label)
    score_display = (
        score_text if score_text is not None else "Metric unavailable due to missing evidence"
    )

    md_lines = [
        f"# {ticker} Opportunity Memo ({effective_as_of})",
        "",
        f"Mode: **{memo_mode.upper()}**",
        f"Decision: **{decision}**",
        f"Status Flags: **{', '.join(status_flags)}**",
        f"Score: **{score_display}**",
    ]
    if event_flags:
        md_lines.append(
            f"Events Pending: **{', '.join(event_flags)}** — open corporate event; not presentable at target until disposed."
        )
    md_lines.extend(["", "## Known Reasons It May Be Cheap"])
    from app.events.cheapness import render_cheapness_block

    md_lines.extend(render_cheapness_block(cheapness_report))
    md_lines.extend(
        [
            "",
            "## Filings Used",
        ]
    )

    filings_used = packet.get("filings_used", []) if isinstance(packet, dict) else []
    if filings_used:
        for filing in filings_used:
            md_lines.append(
                f"- {filing.get('form_type')} {filing.get('filing_date')} {filing.get('accession')} ([source]({filing.get('primary_doc_url')}))"
            )
    else:
        md_lines.append("- Filing coverage incomplete for this memo.")

    md_lines.extend(
        [
            "",
            "## Business Snapshot (Facts Only)",
            _metric_line("Revenue", "revenue", claim_by_label, memo_gaps),
            _metric_line("Operating Margin", "operating_margin", claim_by_label, memo_gaps),
            _metric_line("FCF", "fcf", claim_by_label, memo_gaps),
            _metric_line("Net Debt", "net_debt", claim_by_label, memo_gaps),
            "",
            "## Valuation",
            _valuation_table(packet, claim_by_label, memo_gaps),
            "",
            "## Thesis (Valuation-Led)",
        ]
    )

    if hypotheses and hypotheses.get("hypotheses"):
        for hyp in hypotheses.get("hypotheses", [])[:5]:
            md_lines.append(f"- {hyp.get('direction')}: {hyp.get('claim')}")
    else:
        md_lines.append("- Analyst hypothesis coverage incomplete; see gaps and next actions.")

    if analysis_snapshot is not None:
        md_lines.extend(["", "## Analyst View"])
        if analysis_snapshot.analysis_source == "analysis_report":
            confidence = analysis_snapshot.confidence_label
            if analysis_snapshot.confidence_score is not None:
                confidence = f"{confidence} ({analysis_snapshot.confidence_score}/100)"
            md_lines.append(f"- Verdict: {analysis_snapshot.verdict}")
            md_lines.append(f"- Confidence: {confidence}")
            md_lines.append(f"- Thesis Summary: {analysis_snapshot.thesis_summary}")
        else:
            md_lines.append(
                "- Analyst verdict unavailable; using normalized legacy research fallback."
            )

        quality = analysis_snapshot.research_quality
        if quality is not None:
            q_overall = _value_text("research_quality_overall_research_score", claim_by_label)
            q_cov = _value_text("research_quality_coverage_score", claim_by_label)
            q_fresh = _value_text("research_quality_freshness_score", claim_by_label)
            q_gap = _value_text("research_quality_gap_score", claim_by_label)
            if all(x is not None for x in [q_overall, q_cov, q_fresh, q_gap]):
                md_lines.append(
                    f"- Research Quality: overall={q_overall} (coverage={q_cov}, freshness={q_fresh}, gaps={q_gap})"
                )
            else:
                md_lines.append("- Research Quality: Metric unavailable due to missing evidence.")
                memo_gaps.append("Metric unavailable due to missing evidence: research_quality")

            if quality.incomplete:
                md_lines.append("- Research Status: **RESEARCH_INCOMPLETE**")
            if quality.top_gaps:
                md_lines.extend(["", "### Top Evidence Gaps"])
                for gap in quality.top_gaps[:5]:
                    md_lines.append(f"- {gap.severity}: {gap.summary}")

        md_lines.extend(["", "### Positives"])
        _append_findings_section(md_lines, "", analysis_snapshot.positives)
        md_lines.extend(["", "### Risks"])
        _append_findings_section(md_lines, "", analysis_snapshot.risks)
        md_lines.extend(["", "### Recent Event Impacts"])
        _append_findings_section(md_lines, "", analysis_snapshot.recent_event_impacts)

        md_lines.extend(["", "### Open Questions"])
        if analysis_snapshot.open_questions:
            for item in analysis_snapshot.open_questions[:6]:
                line = f"- {item.question} ({item.importance})"
                if item.next_step:
                    line += f" - Next step: {item.next_step}"
                md_lines.append(line)
        else:
            md_lines.append("- None.")

        md_lines.extend(["", "### Falsifiers"])
        if analysis_snapshot.falsifiers:
            for item in analysis_snapshot.falsifiers[:6]:
                line = f"- {item.description}"
                if item.monitoring_hint:
                    line += f" - Monitor: {item.monitoring_hint}"
                md_lines.append(line)
        else:
            md_lines.append("- None.")

        md_lines.extend(["", "### Next Actions"])
        next_actions = _next_actions_from_snapshot(analysis_snapshot)
        if next_actions:
            for action in next_actions:
                md_lines.append(f"- {action}")
        else:
            md_lines.append("- None.")

    if research_signals:
        summary = research_signals.get("summary", {}) if isinstance(research_signals, dict) else {}
        md_lines.extend(["", "## Research Signals"])
        md_lines.append(f"- Freshness Bucket: {summary.get('freshness_bucket', 'UNKNOWN')}")
        md_lines.append(f"- Information Flow Bucket: {summary.get('flow_bucket', 'UNKNOWN')}")
        md_lines.append(
            "- Earnings Release Signal: "
            + ("PRESENT" if bool(research_signals.get("has_earnings_release")) else "ABSENT")
        )
        md_lines.append(
            "- Investor Presentation Signal: "
            + ("PRESENT" if bool(research_signals.get("has_investor_presentation")) else "ABSENT")
        )
        flags = research_signals.get("sentiment_flags") or []
        md_lines.append(f"- Risk Flags: {', '.join(flags) if flags else 'NONE'}")
        topics = research_signals.get("key_topics") or []
        md_lines.append(f"- Key Topics: {', '.join(topics[:10]) if topics else 'NONE'}")

    md_lines.extend(["", "## Risks and Falsifiers"])
    if analysis_snapshot is not None and analysis_snapshot.falsifiers:
        for item in analysis_snapshot.falsifiers[:6]:
            md_lines.append(f"- {item.description}")
    elif analyst_decision:
        for text in analyst_decision.get("decision", {}).get("falsifiers", []):
            md_lines.append(f"- {text}")
    else:
        md_lines.append("- Falsifier coverage incomplete; prioritize next filing review.")

    md_lines.extend(
        [
            "",
            "## Next Dates to Watch",
            "- Next SEC filing date: UNKNOWN (filing-derived date not explicitly available in packet)",
        ]
    )

    md_lines.extend(["", "## Evidence Appendix"])
    for fact in packet.get("extracted_facts", [])[:20]:
        citation = fact.get("citation", {})
        md_lines.append(
            f"- {fact.get('fact_type')}: {citation.get('source_url')} :: {citation.get('snippet', '')[:220]}"
        )
    if analysis_snapshot is not None:
        for item in analysis_snapshot.citations[:12]:
            md_lines.append(
                f"- research/{item.source_type}: {item.source_url} :: {item.excerpt[:220]}"
            )

    if memo_gaps or failed_labels:
        md_lines.extend(["", "## Evidence Gaps"])
        for gap in sorted(set(memo_gaps))[:12]:
            md_lines.append(f"- {gap}")
        for label in sorted(failed_labels):
            md_lines.append(f"- Suppressed numeric claim without trace: {label}")

    md_lines.extend(["", "## Claims Trace"])
    for claim in valid_claims:
        derived = ", ".join(claim.get("derived_from", [])) or "NONE"
        citation_count = len(claim.get("citations") or [])
        md_lines.append(
            f"- {claim['label']}={claim['value']} (derived_from={derived}; citations={citation_count})"
        )

    memo_md = "\n".join(md_lines)
    md_path = memo_dir / "memo.md"
    md_path.write_text(memo_md, encoding="utf-8")

    gaps_path = None
    if memo_mode == "triage":
        gaps_run_id = run_id or f"triage_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        gaps_path = write_gaps_artifact(
            ticker=ticker,
            as_of_date=effective_as_of,
            run_id=gaps_run_id,
            packet=packet,
            research=research,
            status_flags=status_flags,
            suppressed_claims=sorted(failed_labels) + sorted(set(memo_gaps)),
        )

    memo_json = {
        "ticker": ticker,
        "as_of_date": effective_as_of,
        "run_id": run_id,
        "decision": decision,
        "status_flags": status_flags,
        "memo_mode": memo_mode,
        "score": score_row["total_score"] if score_row else UNKNOWN,
        "packet_path": str(packet_path) if packet_path else None,
        "hypotheses_path": str(hypotheses_path) if hypotheses_path else None,
        "red_team_path": str(red_team_path) if red_team_path else None,
        "analysis_report_path": (
            str(analysis_snapshot.analysis_report_path)
            if analysis_snapshot is not None and analysis_snapshot.analysis_report_path is not None
            else None
        ),
        "analysis_source": analysis_snapshot.analysis_source
        if analysis_snapshot is not None
        else "missing",
        "research_path": (
            str(analysis_snapshot.legacy_research_path)
            if analysis_snapshot is not None and analysis_snapshot.legacy_research_path is not None
            else None
        ),
        "research_signals": research_signals,
        "memo_md_path": str(md_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "gaps_path": str(gaps_path) if gaps_path else None,
        "quality_failures": quality_failures,
        "suppressed_numeric_claims": sorted(failed_labels),
        "claims": valid_claims,
    }
    json_path = memo_dir / "memo.json"
    json_path.write_text(json.dumps(memo_json, indent=2), encoding="utf-8")

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO memos(ticker, as_of_date, run_id, memo_path, manifest_path, summary_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, as_of_date) DO UPDATE SET
                run_id=excluded.run_id,
                memo_path=excluded.memo_path,
                manifest_path=excluded.manifest_path,
                summary_json=excluded.summary_json
            """,
            (
                ticker,
                effective_as_of,
                run_id,
                str(md_path),
                str(manifest_path) if manifest_path else None,
                json.dumps(memo_json),
                utc_now_iso(),
            ),
        )
    return True


def build_top_memos(top_n: int = 5, *, memo_mode: str = "strict", run_id: str | None = None) -> int:
    memo_mode = memo_mode.strip().lower()
    if memo_mode == "triage" and not run_id:
        raise ValueError("run_id is required for triage memo mode")
    with get_db() as conn:
        if memo_mode == "triage":
            rows = conn.execute(
                """
                SELECT ticker, as_of_date
                FROM scores
                WHERE is_candidate = 1 AND candidate_run_id = ?
                ORDER BY total_score DESC, ticker ASC
                LIMIT ?
                """,
                (run_id, top_n),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    """
                    SELECT ticker, as_of_date
                    FROM scores
                    WHERE run_id = ?
                    ORDER BY total_score DESC, ticker ASC
                    LIMIT ?
                    """,
                    (run_id, top_n),
                ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT ticker, as_of_date
                FROM scores
                ORDER BY total_score DESC, ticker ASC
                LIMIT ?
                """,
                (top_n,),
            ).fetchall()

    built = 0
    for row in rows:
        if build_memo_for_ticker(
            row["ticker"],
            memo_mode=memo_mode,
            run_id=run_id,
            as_of_date=row["as_of_date"],
        ):
            built += 1

    logger.info("memo_completed", extra={"stage_name": "report", "stage_count": built})
    return built


def build_weekly_digest() -> Path:
    cfg = get_config()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT ticker, as_of_date, total_score, decision
            FROM scores
            ORDER BY total_score DESC
            LIMIT 20
            """
        ).fetchall()

    payload = {
        "generated_at": utc_now_iso(),
        "top_opportunities": [
            {
                "ticker": row["ticker"],
                "as_of_date": row["as_of_date"],
                "total_score": row["total_score"],
                "decision": row["decision"],
            }
            for row in rows
        ],
    }

    out_json = cfg.rankings_dir / f"weekly_digest_{utc_now_iso()[:10]}.json"
    out_md = cfg.rankings_dir / f"weekly_digest_{utc_now_iso()[:10]}.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = ["# Weekly Top Opportunities", ""]
    for item in payload["top_opportunities"]:
        md.append(f"- {item['ticker']}: score {item['total_score']} ({item['decision']})")
    out_md.write_text("\n".join(md), encoding="utf-8")

    return out_json
