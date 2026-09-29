from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.analyst.ops_snapshot import (
    load_ops_analysis_snapshot,
    research_quality_to_rubric_dict,
    snapshot_to_research_payload,
)
from app.analyst.output_store import latest_eligible_analysis_output_bytes
from app.db import get_db, utc_now_iso
from app.report.claims import build_numeric_claims, validate_claims_have_evidence_or_derivation
from app.valuation.lineage import latest_decision_eligible_valuation_rows


REASON_OK = "OK"
REASON_NO_RECENT_FILING = "NO_RECENT_FILING"
REASON_PARSE_NOT_COMPLETED = "PARSE_NOT_COMPLETED"
REASON_FUNDAMENTALS_MISSING = "FUNDAMENTALS_MISSING"
REASON_VALUATION_MISSING = "VALUATION_MISSING"
REASON_EVIDENCE_PACKET_MISSING = "EVIDENCE_PACKET_MISSING"
REASON_RESEARCH_PACKET_MISSING = "RESEARCH_PACKET_MISSING"
REASON_RESEARCH_INCOMPLETE = "RESEARCH_INCOMPLETE"
REASON_INSUFFICIENT_EVIDENCE_ITEMS = "INSUFFICIENT_EVIDENCE_ITEMS"
REASON_ANALYST_HYPOTHESES_MISSING = "ANALYST_HYPOTHESES_MISSING"
REASON_RED_TEAM_MISSING = "RED_TEAM_MISSING"
REASON_KEY_CLAIMS_MISSING_CITATIONS = "KEY_CLAIMS_MISSING_CITATIONS"
REASON_NUMERIC_CLAIM_NO_TRACE = "NUMERIC_CLAIM_NO_TRACE"
REASON_SCORE_MISSING = "SCORE_MISSING"
REASON_MEMO_NOT_BUILT = "MEMO_NOT_BUILT"
REASON_MISSING_PRICE = "MISSING_PRICE"
REASON_PARSER_MISSING_SECTION = "PARSER_MISSING_SECTION"
REASON_SKIPPED = "SKIPPED"


@dataclass
class GateStatus:
    gate: str
    status: str
    reason_code: str
    message: str
    fix_hint: str

    def as_dict(self) -> dict[str, str]:
        return {
            "gate": self.gate,
            "status": self.status,
            "reason_code": self.reason_code,
            "message": self.message,
            "fix_hint": self.fix_hint,
        }


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _latest_row(conn, query: str, args: tuple[Any, ...]) -> Any | None:
    return conn.execute(query, args).fetchone()


def _latest_analyst_output(
    conn, ticker: str, as_of_date: str, output_type: str
) -> dict[str, Any] | None:
    result = latest_eligible_analysis_output_bytes(
        conn,
        ticker,
        output_type,
        as_of_date=as_of_date,
    )
    if result is None:
        return None
    _, output_bytes = result
    try:
        payload = json.loads(output_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_research_packet(
    conn, ticker: str, as_of_date: str, run_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    row = _latest_row(
        conn,
        """
        SELECT packet_path, run_id
        FROM research_packets
        WHERE ticker = ? AND run_id = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT 1
        """,
        (ticker, run_id, as_of_date),
    )
    if not row:
        return None, None
    return _load_json(row["packet_path"]), row["run_id"]


def _memo_quality_failures(
    packet: dict[str, Any] | None,
    hypotheses: dict[str, Any] | None,
    red_team: dict[str, Any] | None,
) -> list[GateStatus]:
    failures: list[GateStatus] = []

    if not packet or not packet.get("valuations"):
        failures.append(
            GateStatus(
                gate="memo_quality",
                status="FAIL",
                reason_code=REASON_VALUATION_MISSING,
                message="Memo quality gate failed: valuation outputs are missing.",
                fix_hint="Run valuation and rebuild evidence packet before memo generation.",
            )
        )

    if not hypotheses or not hypotheses.get("hypotheses"):
        failures.append(
            GateStatus(
                gate="memo_quality",
                status="FAIL",
                reason_code=REASON_ANALYST_HYPOTHESES_MISSING,
                message="Memo quality gate failed: analyst hypotheses output is missing.",
                fix_hint="Run analyst agent for this ticker and ensure hypotheses.json is written.",
            )
        )
    else:
        missing_citations = any(not h.get("citations") for h in hypotheses.get("hypotheses", []))
        if missing_citations:
            failures.append(
                GateStatus(
                    gate="memo_quality",
                    status="FAIL",
                    reason_code=REASON_KEY_CLAIMS_MISSING_CITATIONS,
                    message="Memo quality gate failed: one or more key hypotheses lacks citations.",
                    fix_hint="Ensure analyst hypotheses include packet citations for every key claim.",
                )
            )

    if not red_team or not red_team.get("red_team"):
        failures.append(
            GateStatus(
                gate="memo_quality",
                status="FAIL",
                reason_code=REASON_RED_TEAM_MISSING,
                message="Memo quality gate failed: red-team output is missing.",
                fix_hint="Run red-team generation and persist red_team.json for this ticker.",
            )
        )

    return failures


def _numeric_claim_statuses(
    packet: dict[str, Any] | None,
    score_row: Any | None,
    research_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[GateStatus]]:
    claims: list[dict[str, Any]] = []
    if packet:
        claims.extend(build_numeric_claims(packet))

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

    if research_payload and isinstance(research_payload.get("claims"), list):
        for claim in research_payload["claims"]:
            value = claim.get("value")
            if not isinstance(value, (int, float)):
                continue
            claims.append(
                {
                    "claim_id": claim.get("claim_id", "research_claim"),
                    "label": claim.get("label", "research_claim"),
                    "value": value,
                    "unit": claim.get("unit"),
                    "citations": claim.get("citations") or [],
                    "derived_from": claim.get("derived_from") or [],
                }
            )

    failures = validate_claims_have_evidence_or_derivation(claims)
    if failures:
        status = [
            GateStatus(
                gate="numeric_claims",
                status="FAIL",
                reason_code=REASON_NUMERIC_CLAIM_NO_TRACE,
                message="; ".join(failures[:3]),
                fix_hint="Add citations or deterministic derived_from traces for each numeric claim.",
            )
        ]
    else:
        status = [
            GateStatus(
                gate="numeric_claims",
                status="PASS",
                reason_code=REASON_OK,
                message="All numeric claims include citation or derivation trace.",
                fix_hint="",
            )
        ]
    return claims, status


def _first_fix_hint(gates: list[GateStatus]) -> str:
    for gate in gates:
        if gate.status == "FAIL" and gate.fix_hint:
            return gate.fix_hint
    return "No blocking gate identified."


def evidence_count_for_run(conn: Any, ticker: str, as_of_date: str, run_id: str) -> int:
    """Evidence items this run collected, including ones a later run re-collected."""
    # An item re-collected by a later run moves its evidence_items row to that
    # run; the link table keeps this run's claim to it.
    has_links = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evidence_item_runs'"
    ).fetchone() is not None
    links_sql = (
        """
            UNION
            SELECT evidence_id
            FROM evidence_item_runs
            WHERE ticker = ? AND as_of_date = ? AND run_id = ?
        """
        if has_links
        else ""
    )
    ev_row = conn.execute(
        f"""
        SELECT COUNT(*) AS n FROM (
            SELECT evidence_id
            FROM evidence_items
            WHERE ticker = ? AND as_of_date = ? AND run_id = ?
            {links_sql}
        )
        """,
        (ticker, as_of_date, run_id) * (2 if has_links else 1),
    ).fetchone()
    return int(ev_row["n"] or 0)


def _ticker_report(
    conn,
    ticker: str,
    *,
    as_of_date: str,
    run_id: str,
    with_research: bool,
    artifact_paths: dict[str, dict[str, str]],
) -> dict[str, Any]:
    filing_count_row = conn.execute(
        """
        SELECT
            COUNT(*) AS filings_count,
            SUM(CASE WHEN status = 'parsed' THEN 1 ELSE 0 END) AS parsed_count
        FROM filings
        WHERE ticker = ? AND COALESCE(filing_date, ingested_as_of, '1900-01-01') <= ?
        """,
        (ticker, as_of_date),
    ).fetchone()
    filings_count = int(filing_count_row["filings_count"] or 0)
    parsed_count = int(filing_count_row["parsed_count"] or 0)

    fundamentals_row = _latest_row(
        conn,
        """
        SELECT as_of_date, metrics_json
        FROM fundamentals
        WHERE ticker = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    )
    fundamentals = json.loads(fundamentals_row["metrics_json"]) if fundamentals_row else None
    valuation_as_of = fundamentals_row["as_of_date"] if fundamentals_row else as_of_date

    valuation_rows = latest_decision_eligible_valuation_rows(
        conn,
        ticker=ticker,
        as_of_date=valuation_as_of,
        exact_as_of_date=True,
    )
    valuation_methods = {row["method"] for row in valuation_rows}
    valuation_map = {
        row["method"]: {
            "inputs": json.loads(row["inputs_json"]),
            "outputs": json.loads(row["outputs_json"]),
        }
        for row in valuation_rows
    }

    packet_row = _latest_row(
        conn,
        """
        SELECT as_of_date, packet_path
        FROM evidence_packets
        WHERE ticker = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (ticker, as_of_date),
    )
    packet_payload = _load_json(packet_row["packet_path"]) if packet_row else None

    score_row = _latest_row(
        conn,
        """
        SELECT as_of_date, total_score, decision, reasons_json
        FROM scores
        WHERE ticker = ? AND run_id = ?
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT 1
        """,
        (ticker, run_id),
    )
    score_as_of = str(score_row["as_of_date"]) if score_row else as_of_date

    memo_row = _latest_row(
        conn,
        """
        SELECT as_of_date, memo_path
        FROM memos
        WHERE ticker = ? AND run_id = ? AND as_of_date = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (ticker, run_id, score_as_of),
    )
    memo_exists = bool(memo_row and memo_row["memo_path"] and Path(memo_row["memo_path"]).exists())

    hypotheses_payload = _latest_analyst_output(conn, ticker, score_as_of, "hypotheses")
    red_team_payload = _latest_analyst_output(conn, ticker, score_as_of, "red_team")
    analysis_snapshot = load_ops_analysis_snapshot(ticker, as_of_date=score_as_of, run_id=run_id)
    research_payload = snapshot_to_research_payload(analysis_snapshot)
    quality = research_quality_to_rubric_dict(
        analysis_snapshot.research_quality if analysis_snapshot is not None else None
    )
    coverage_score = quality.get("coverage_score") if isinstance(quality, dict) else None
    freshness_score = quality.get("freshness_score") if isinstance(quality, dict) else None
    gap_score = quality.get("gap_score") if isinstance(quality, dict) else None
    overall_research_score = (
        quality.get("overall_research_score") if isinstance(quality, dict) else None
    )

    research_items_count = (
        int(analysis_snapshot.research_quality.evidence_count)
        if analysis_snapshot is not None
        and analysis_snapshot.research_quality is not None
        and analysis_snapshot.research_quality.evidence_count is not None
        else 0
    )

    evidence_items_count = 0
    if run_id:
        evidence_items_count = evidence_count_for_run(conn, ticker, score_as_of, run_id)
    if evidence_items_count == 0:
        evidence_items_count = research_items_count

    claims, numeric_statuses = _numeric_claim_statuses(packet_payload, score_row, research_payload)
    numeric_claims_count = len(claims)

    ingested = filings_count > 0
    parsed = parsed_count > 0
    fundamentals_ready = fundamentals is not None
    valuation_ready = {"dcf", "reverse_dcf"}.issubset(valuation_methods) or {
        "multiples",
        "dcf_lite",
    }.issubset(valuation_methods)
    research_ready = analysis_snapshot is not None
    research_stage_completed = research_ready if with_research else False
    evidence_packet_ready = packet_payload is not None
    scored = score_row is not None

    gates: list[GateStatus] = []

    if ingested:
        gates.append(
            GateStatus(
                gate="ingest",
                status="PASS",
                reason_code=REASON_OK,
                message=f"{filings_count} filing(s) available for ticker.",
                fix_hint="",
            )
        )
    else:
        gates.append(
            GateStatus(
                gate="ingest",
                status="FAIL",
                reason_code=REASON_NO_RECENT_FILING,
                message="No filing metadata available at or before as_of_date.",
                fix_hint="Run ingest-sec/backfill for this ticker and date range.",
            )
        )

    if parsed:
        gates.append(
            GateStatus(
                gate="parse",
                status="PASS",
                reason_code=REASON_OK,
                message=f"{parsed_count} filing(s) parsed.",
                fix_hint="",
            )
        )
    else:
        gates.append(
            GateStatus(
                gate="parse",
                status="FAIL",
                reason_code=REASON_PARSE_NOT_COMPLETED,
                message="No parsed filings found.",
                fix_hint="Run parse-filings for this ticker.",
            )
        )

    if fundamentals_ready:
        gates.append(
            GateStatus(
                gate="fundamentals",
                status="PASS",
                reason_code=REASON_OK,
                message="Fundamentals exist for ticker.",
                fix_hint="",
            )
        )
    else:
        gates.append(
            GateStatus(
                gate="fundamentals",
                status="FAIL",
                reason_code=REASON_FUNDAMENTALS_MISSING,
                message="Fundamentals row missing.",
                fix_hint="Run compute-fundamentals for this ticker.",
            )
        )

    if valuation_ready:
        gates.append(
            GateStatus(
                gate="valuation",
                status="PASS",
                reason_code=REASON_OK,
                message="Required valuation methods present.",
                fix_hint="",
            )
        )
    else:
        canonical_missing = sorted({"dcf", "reverse_dcf"}.difference(valuation_methods))
        legacy_missing = sorted({"multiples", "dcf_lite"}.difference(valuation_methods))
        gates.append(
            GateStatus(
                gate="valuation",
                status="FAIL",
                reason_code=REASON_VALUATION_MISSING,
                message=(
                    f"Missing valuation outputs: canonical[{', '.join(canonical_missing) if canonical_missing else 'none'}]; "
                    f"legacy[{', '.join(legacy_missing) if legacy_missing else 'none'}]."
                ),
                fix_hint="Run valuation generation and ensure at least canonical dcf + reverse_dcf outputs are stored.",
            )
        )

    if evidence_packet_ready:
        gates.append(
            GateStatus(
                gate="evidence_packet",
                status="PASS",
                reason_code=REASON_OK,
                message="Evidence packet exists.",
                fix_hint="",
            )
        )
    else:
        gates.append(
            GateStatus(
                gate="evidence_packet",
                status="FAIL",
                reason_code=REASON_EVIDENCE_PACKET_MISSING,
                message="Evidence packet missing.",
                fix_hint="Run build-evidence-packets after fundamentals and valuation stages.",
            )
        )

    if with_research:
        if not research_ready:
            gates.append(
                GateStatus(
                    gate="research",
                    status="FAIL",
                    reason_code=REASON_RESEARCH_PACKET_MISSING,
                    message="Research packet missing for this run scope.",
                    fix_hint="Run research-run for this ticker with a run_id tied to current run.",
                )
            )
        else:
            incomplete = bool(isinstance(quality, dict) and quality.get("incomplete"))
            if incomplete:
                top_gap_action = ""
                top_gaps = quality.get("top_gaps") if isinstance(quality, dict) else None
                if isinstance(top_gaps, list) and top_gaps:
                    first_gap = top_gaps[0]
                    if isinstance(first_gap, dict):
                        top_gap_action = str(first_gap.get("recommended_action") or "").strip()
                gates.append(
                    GateStatus(
                        gate="research",
                        status="FAIL",
                        reason_code=REASON_RESEARCH_INCOMPLETE,
                        message="Research quality threshold not met.",
                        fix_hint=top_gap_action or "Address top evidence gaps and rerun research.",
                    )
                )
            elif research_items_count < 3:
                gates.append(
                    GateStatus(
                        gate="research",
                        status="FAIL",
                        reason_code=REASON_INSUFFICIENT_EVIDENCE_ITEMS,
                        message=f"Only {research_items_count} research evidence item(s) collected.",
                        fix_hint="Increase allowed evidence coverage (e.g., IR RSS metadata + allowlist) and rerun.",
                    )
                )
            else:
                gates.append(
                    GateStatus(
                        gate="research",
                        status="PASS",
                        reason_code=REASON_OK,
                        message="Research packet present and quality threshold met.",
                        fix_hint="",
                    )
                )
    else:
        gates.append(
            GateStatus(
                gate="research",
                status="SKIP",
                reason_code=REASON_SKIPPED,
                message="Research stage not requested for this run.",
                fix_hint="Run with --with-research to include external evidence collection.",
            )
        )

    if scored:
        gates.append(
            GateStatus(
                gate="scoring",
                status="PASS",
                reason_code=REASON_OK,
                message="Score row exists.",
                fix_hint="",
            )
        )
    else:
        gates.append(
            GateStatus(
                gate="scoring",
                status="FAIL",
                reason_code=REASON_SCORE_MISSING,
                message="Score row missing.",
                fix_hint="Run score stage after valuation/evidence packet.",
            )
        )

    if packet_payload and parsed:
        required_sections = {"revenue", "operating_income", "cfo", "capex", "cash", "total_debt"}
        line_items = {
            row.get("line_item")
            for row in packet_payload.get("financials", [])
            if row.get("line_item")
        }
        missing_sections = sorted(required_sections.difference(line_items))
        if missing_sections:
            gates.append(
                GateStatus(
                    gate="parser_sections",
                    status="WARN",
                    reason_code=REASON_PARSER_MISSING_SECTION,
                    message=f"Missing parsed financial sections: {', '.join(missing_sections)}.",
                    fix_hint="Improve statement extractor coverage for missing sections.",
                )
            )
        else:
            gates.append(
                GateStatus(
                    gate="parser_sections",
                    status="PASS",
                    reason_code=REASON_OK,
                    message="Required financial sections parsed.",
                    fix_hint="",
                )
            )

    price_status = "UNKNOWN"
    reverse_inputs = valuation_map.get("reverse_dcf", {}).get("inputs", {})
    if isinstance(reverse_inputs, dict):
        price_status = str(reverse_inputs.get("price_status") or "UNKNOWN")
    if price_status.upper() == "UNKNOWN":
        gates.append(
            GateStatus(
                gate="price",
                status="WARN",
                reason_code=REASON_MISSING_PRICE,
                message="Market price unavailable; reverse DCF feasibility is limited.",
                fix_hint="Enable an allowed price provider or continue as research-only.",
            )
        )
    else:
        gates.append(
            GateStatus(
                gate="price",
                status="PASS",
                reason_code=REASON_OK,
                message=f"Market price status={price_status}.",
                fix_hint="",
            )
        )

    gates.extend(_memo_quality_failures(packet_payload, hypotheses_payload, red_team_payload))
    gates.extend(numeric_statuses)

    if memo_exists:
        gates.append(
            GateStatus(
                gate="memo",
                status="PASS",
                reason_code=REASON_OK,
                message="Memo built successfully.",
                fix_hint="",
            )
        )
    else:
        gates.append(
            GateStatus(
                gate="memo",
                status="FAIL",
                reason_code=REASON_MEMO_NOT_BUILT,
                message="Memo artifact missing for ticker.",
                fix_hint="Resolve earlier failed gates and rerun build-report/run-all.",
            )
        )

    artifact = artifact_paths.get(ticker, {})
    score_decision = score_row["decision"] if score_row else "UNKNOWN"
    key_metrics = {
        "coverage_score": coverage_score,
        "freshness_score": freshness_score,
        "gap_score": gap_score,
        "overall_research_score": overall_research_score,
        "price_status": price_status,
        "filings_count": filings_count,
        "score_decision": score_decision,
    }

    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "ingested": ingested,
        "parsed": parsed,
        "fundamentals": fundamentals_ready,
        "valuation": valuation_ready,
        "research": research_stage_completed,
        "evidence_packet": evidence_packet_ready,
        "scored": scored,
        "memo_built": memo_exists,
        "decision": score_decision,
        "gate_statuses": [g.as_dict() for g in gates],
        "minimal_fix": _first_fix_hint(gates),
        "key_metrics": key_metrics,
        "counts": {
            "evidence_items_count": evidence_items_count,
            "research_items_count": research_items_count,
            "numeric_claims_count": numeric_claims_count,
        },
        "artifacts": {
            "memo_path": artifact.get("memo_path"),
            "packet_path": artifact.get("packet_path"),
            "analysis_report_path": artifact.get("analysis_report_path"),
            "analysis_report_md_path": artifact.get("analysis_report_md_path"),
            "research_path": artifact.get("research_path"),
            "gaps_path": artifact.get("gaps_path"),
            "manifest_path": artifact.get("manifest_path"),
        },
    }


def build_gating_report(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str],
    run_dir: Path,
    with_research: bool,
    artifact_paths: dict[str, dict[str, str]],
) -> tuple[Path, Path, list[dict[str, Any]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with get_db() as conn:
        for ticker in tickers:
            rows.append(
                _ticker_report(
                    conn,
                    ticker,
                    as_of_date=as_of_date,
                    run_id=run_id,
                    with_research=with_research,
                    artifact_paths=artifact_paths,
                )
            )

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "generated_at": utc_now_iso(),
        "tickers_targeted": tickers,
        "tickers_processed": [
            row["ticker"]
            for row in rows
            if any(
                [
                    row.get("ingested"),
                    row.get("parsed"),
                    row.get("fundamentals"),
                    row.get("valuation"),
                    row.get("evidence_packet"),
                    row.get("scored"),
                    row.get("memo_built"),
                ]
            )
        ],
        "rows": rows,
    }

    json_path = run_dir / "gating_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = run_dir / "gating_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "as_of_date",
                "ingested",
                "parsed",
                "fundamentals",
                "valuation",
                "research",
                "evidence_packet",
                "scored",
                "memo_built",
                "decision",
                "minimal_fix",
                "gate_statuses",
                "key_metrics",
                "evidence_items_count",
                "research_items_count",
                "numeric_claims_count",
                "memo_path",
                "packet_path",
                "analysis_report_path",
                "analysis_report_md_path",
                "research_path",
                "gaps_path",
                "manifest_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            counts = row.get("counts", {})
            artifacts = row.get("artifacts", {})
            writer.writerow(
                {
                    "ticker": row.get("ticker"),
                    "as_of_date": row.get("as_of_date"),
                    "ingested": row.get("ingested"),
                    "parsed": row.get("parsed"),
                    "fundamentals": row.get("fundamentals"),
                    "valuation": row.get("valuation"),
                    "research": row.get("research"),
                    "evidence_packet": row.get("evidence_packet"),
                    "scored": row.get("scored"),
                    "memo_built": row.get("memo_built"),
                    "decision": row.get("decision"),
                    "minimal_fix": row.get("minimal_fix"),
                    "gate_statuses": json.dumps(row.get("gate_statuses", [])),
                    "key_metrics": json.dumps(row.get("key_metrics", {})),
                    "evidence_items_count": counts.get("evidence_items_count", 0),
                    "research_items_count": counts.get("research_items_count", 0),
                    "numeric_claims_count": counts.get("numeric_claims_count", 0),
                    "memo_path": artifacts.get("memo_path"),
                    "packet_path": artifacts.get("packet_path"),
                    "analysis_report_path": artifacts.get("analysis_report_path"),
                    "analysis_report_md_path": artifacts.get("analysis_report_md_path"),
                    "research_path": artifacts.get("research_path"),
                    "gaps_path": artifacts.get("gaps_path"),
                    "manifest_path": artifacts.get("manifest_path"),
                }
            )

    return json_path, csv_path, rows
