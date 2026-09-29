from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "data" / "outputs" / "runs"

SINGLE_VALID_VERDICTS = {"ACTIONABLE", "WATCHLIST_ONLY", "AVOID", "NO_WINNER", "DATA_INCOMPLETE"}
# DATA_INCOMPLETE: the resolve-then-promote tier emits it as a sector-level
# final verdict when the selected candidate is held purely on fetchable
# data-availability gaps (sector_runtime guardrail normalization).
SECTOR_VALID_VERDICTS = {"SELECTED", "WATCHLIST", "NO_SELECTION", "DATA_INCOMPLETE"}
VALID_STATUSES = {"COMPLETED", "FAILED", "ERROR", "BLOCKED", "DEGRADED"}
SYNTHETIC_TICKER_PREFIXES = ("AAA", "BBB")


@dataclass(frozen=True)
class EvalCheckResult:
    check: str
    passed: bool
    message: str = ""


def discover_single_artifact_paths() -> list[Path]:
    return _without_archived_paths(RUNS_ROOT.glob("autonomous/**/autonomous_run.json"))


def discover_sector_artifact_paths() -> list[Path]:
    return _without_archived_paths(RUNS_ROOT.glob("autonomous_sector/**/autonomous_sector_run.json"))


def discover_artifact_paths() -> list[Path]:
    return discover_single_artifact_paths() + discover_sector_artifact_paths()


def discover_single_report_paths() -> list[Path]:
    return _without_archived_paths(RUNS_ROOT.glob("autonomous/**/autonomous_research_report.md"))


def discover_sector_report_paths() -> list[Path]:
    return _without_archived_paths(RUNS_ROOT.glob("autonomous_sector/**/autonomous_sector_report.md"))


def _without_archived_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(path for path in paths if "_archive" not in path.parts)


def artifact_kind(path: Path) -> str:
    return "sector" if path.name == "autonomous_sector_run.json" else "single"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _add(results: list[EvalCheckResult], check: str, passed: bool, message: str = "") -> None:
    results.append(EvalCheckResult(check=check, passed=passed, message=message if not passed else ""))


def _ids(items: Iterable[dict[str, Any]], key: str) -> set[str]:
    return {str(item.get(key)) for item in items if isinstance(item, dict) and item.get(key)}


def _tool_evidence_refs(tool_calls: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        refs.update(str(item) for item in call.get("evidence_ref_ids") or [] if str(item))
    return refs


def _walk_ticker_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in {"ticker", "selected_ticker"} and isinstance(item, str):
                yield item
            elif normalized.endswith("tickers") and isinstance(item, list):
                for ticker in item:
                    if isinstance(ticker, str):
                        yield ticker
            else:
                yield from _walk_ticker_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_ticker_values(item)


def _synthetic_tickers(data: dict[str, Any]) -> list[str]:
    tickers: list[str] = []
    for ticker in _walk_ticker_values(data):
        normalized = ticker.upper()
        if normalized == "TEST" or normalized.startswith(SYNTHETIC_TICKER_PREFIXES):
            tickers.append(ticker)
    return sorted(set(tickers))


def _candidate_selection_tickers(data: dict[str, Any]) -> set[str]:
    candidate_selection = data.get("candidate_selection") if isinstance(data.get("candidate_selection"), dict) else {}
    selected = candidate_selection.get("selected_tickers") if isinstance(candidate_selection, dict) else []
    relative = data.get("relative_ranking") if isinstance(data.get("relative_ranking"), list) else []
    tickers = {str(item).upper() for item in selected or [] if str(item)}
    tickers.update(str(row.get("ticker")).upper() for row in relative if isinstance(row, dict) and row.get("ticker"))
    return tickers


def _company_packet_tickers(data: dict[str, Any]) -> set[str]:
    packets = data.get("company_packets") if isinstance(data.get("company_packets"), list) else []
    return {str(packet.get("ticker")).upper() for packet in packets if isinstance(packet, dict) and packet.get("ticker")}


def _audit_contradictions(data: dict[str, Any]) -> list[str]:
    contradictions: list[str] = []
    selection_audit = data.get("selection_audit") if isinstance(data.get("selection_audit"), dict) else {}
    audit_status = selection_audit.get("audit_status") or selection_audit.get("status")
    audited_verdict = selection_audit.get("final_verdict_after_audit")
    actionable = selection_audit.get("actionable")
    if audit_status == "BLOCKED" and (audited_verdict in {"ACTIONABLE", "SELECTED"} or actionable is True):
        contradictions.append("selection_audit BLOCKED but actionable/selected")
    if audit_status == "PASS" and (audited_verdict not in {"ACTIONABLE", "SELECTED", None} or actionable is False):
        contradictions.append("selection_audit PASS but non-actionable verdict")

    for row in data.get("relative_ranking") or []:
        if not isinstance(row, dict):
            continue
        row_status = row.get("audit_status")
        row_verdict = row.get("company_autonomy_verdict") or row.get("verdict") or row.get("final_verdict")
        row_actionable = row.get("actionable")
        ticker = row.get("ticker", "UNKNOWN")
        if row_status == "BLOCKED" and (row_actionable is True or row_verdict in {"ACTIONABLE", "SELECTED"}):
            contradictions.append(f"{ticker}: BLOCKED but actionable/selected")
        if row_status == "PASS" and (row_actionable is False or row_verdict in {"AVOID", "NO_SELECTION"}):
            contradictions.append(f"{ticker}: PASS but avoid/non-actionable")
    return contradictions


def _collect_section_sources(value: Any) -> list[str]:
    sources: list[str] = []
    if isinstance(value, dict):
        source = value.get("source")
        if isinstance(source, str):
            sources.append(source)
        for item in value.values():
            sources.extend(_collect_section_sources(item))
    elif isinstance(value, list):
        for item in value:
            sources.extend(_collect_section_sources(item))
    return sources


def _memo_body_mismatches(data: dict[str, Any]) -> list[str]:
    memo = data.get("memo_body")
    if not isinstance(memo, dict) or not memo:
        return []
    degraded_states = memo.get("degraded_states")
    if not isinstance(degraded_states, list):
        degraded_states = data.get("memo_body_degraded_states") if isinstance(data.get("memo_body_degraded_states"), list) else []
    sources = _collect_section_sources(memo)
    fallback_sources = [
        source for source in sources
        if "fallback" in source.lower() or "deterministic" in source.lower() or source.lower() in {"error", "degraded"}
    ]
    mismatches: list[str] = []
    if fallback_sources and not degraded_states:
        mismatches.append(f"fallback section source(s) {sorted(set(fallback_sources))} without degraded_states")
    if degraded_states and not fallback_sources and str(memo.get("status", "")).upper() == "LLM_GENERATED":
        mismatches.append(f"degraded_states present despite all-LLM memo sources: {degraded_states}")
    return mismatches


def artifact_soundness_results(path: Path) -> list[EvalCheckResult]:
    data = load_json(path)
    kind = artifact_kind(path)
    results: list[EvalCheckResult] = []

    tool_calls = data.get("tool_calls") if isinstance(data.get("tool_calls"), list) else []
    evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
    verdict = data.get("final_verdict")
    selected_ticker = data.get("selected_ticker")
    confidence = data.get("confidence")
    status = data.get("status")

    _add(results, "tool_calls_total", len(tool_calls) > 0, f"{path}: tool_calls_total is 0")

    evidence_ids = _ids(evidence, "evidence_id")
    cited_refs = _tool_evidence_refs(tool_calls)
    missing_refs = sorted(cited_refs - evidence_ids)
    _add(
        results,
        "evidence_refs_exist",
        not missing_refs,
        f"{path}: tool_calls cite missing evidence ids {missing_refs}",
    )

    valid_verdicts = SECTOR_VALID_VERDICTS if kind == "sector" else SINGLE_VALID_VERDICTS
    _add(
        results,
        "final_verdict_valid",
        verdict in valid_verdicts,
        f"{path}: final_verdict {verdict!r} not in {sorted(valid_verdicts)}",
    )

    actionable_verdicts = {"ACTIONABLE", "SELECTED"}
    _add(
        results,
        "actionable_selection_complete",
        verdict not in actionable_verdicts or (bool(selected_ticker) and bool(confidence)),
        f"{path}: {verdict} requires selected_ticker and confidence",
    )

    no_winner_verdicts = {"NO_WINNER", "NO_SELECTION"}
    no_selection_reason = data.get("no_selection_reason") if kind == "sector" else data.get("no_winner_reason")
    _add(
        results,
        "no_selection_complete",
        verdict not in no_winner_verdicts or (selected_ticker is None and bool(no_selection_reason)),
        f"{path}: {verdict} requires selected_ticker=null and no-selection/no-winner reason",
    )

    _add(
        results,
        "status_enum_valid",
        status in VALID_STATUSES and status != "COMPLETED_WITH_SKIPS",
        f"{path}: status {status!r} is invalid or skipped-success",
    )

    synthetic = _synthetic_tickers(data)
    _add(
        results,
        "no_synthetic_tickers",
        not synthetic,
        f"{path}: synthetic ticker(s) appear in production artifact: {synthetic}",
    )

    if kind == "sector":
        finalist_tickers = _candidate_selection_tickers(data)
        packet_tickers = _company_packet_tickers(data)
        missing_packets = sorted(finalist_tickers - packet_tickers)
        _add(
            results,
            "sector_finalists_have_packets",
            not missing_packets,
            f"{path}: finalist ticker(s) missing company_packets entries: {missing_packets}",
        )

        contradictions = _audit_contradictions(data)
        _add(
            results,
            "sector_audit_verdict_consistency",
            not contradictions,
            f"{path}: audit/verdict contradiction(s): {contradictions}",
        )

    mismatches = _memo_body_mismatches(data)
    _add(
        results,
        "memo_body_state_consistency",
        not mismatches,
        f"{path}: memo_body state mismatch(es): {mismatches}",
    )

    return results


def _heading_level(line: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+", line)
    return len(match.group(1)) if match else None


def markdown_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    target = heading.strip()
    start = None
    level = target.count("#", 0, len(target) - len(target.lstrip("#")))
    for idx, line in enumerate(lines):
        if line.strip() == target:
            start = idx + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start, len(lines)):
        next_level = _heading_level(lines[idx])
        if next_level is not None and next_level <= level:
            end = idx
            break
    return "\n".join(lines[start:end]).strip()


def marker_section(text: str, marker: str) -> str:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == marker:
            start = idx + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("**") and stripped.endswith("**"):
            end = idx
            break
        if re.match(r"^#{2,6}\s+", stripped):
            end = idx
            break
    return "\n".join(lines[start:end]).strip()


def candidate_sections(text: str) -> dict[str, str]:
    body = markdown_section(text, "## Per-Candidate Sections")
    if not body:
        return {}
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^###\s+(.+)$", body))
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        ticker = title.split("—", 1)[0].strip().split()[0]
        sections[ticker] = body[start:end].strip()
    return sections


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w$%.-]+\b", text))


def _bullets(text: str) -> list[str]:
    return [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


def _table_data_rows(text: str) -> list[str]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", stripped):
            continue
        if re.search(r"\bFY\b", stripped) and re.search(r"Revenue|Op Margin", stripped):
            continue
        rows.append(stripped)
    return rows


def _risk_bullet_has_business_fact(text: str) -> bool:
    if re.fullmatch(r"[A-Z0-9_ ,;:/().%-]+", text):
        return False
    return bool(re.search(r"\d+(?:\.\d+)?%?", text) or re.search(r"\b[A-Za-z]{5,}\b", text))


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def sector_report_shape_results(path: Path) -> list[EvalCheckResult]:
    text = path.read_text()
    results: list[EvalCheckResult] = []

    _add(results, "sector_executive_conclusion", bool(markdown_section(text, "## Executive Conclusion")), f"{path}: missing Executive Conclusion")

    cohort = markdown_section(text, "## Cohort Comparison")
    cohort_ok = bool(cohort) and (_word_count(cohort) >= 100 or "DEGRADED_STATE" in cohort or "deterministic" in cohort.lower())
    _add(results, "sector_cohort_comparison", cohort_ok, f"{path}: Cohort Comparison missing or too thin")

    triage = markdown_section(text, "## Triage Surprises")
    triage_ok = bool(triage) and (bool(_bullets(triage)) or "no surprises" in triage.lower() or "DEGRADED_STATE" in triage)
    _add(results, "sector_triage_surprises", triage_ok, f"{path}: Triage Surprises missing bullets/fallback")

    sections = candidate_sections(text)
    _add(results, "sector_candidate_sections_present", bool(sections), f"{path}: no per-candidate sections found")

    table_failures: list[str] = []
    business_quality_failures: list[str] = []
    valuation_failures: list[str] = []
    risk_failures: list[str] = []
    thesis_failures: list[str] = []
    falsifier_failures: list[str] = []
    open_question_failures: list[str] = []
    conviction_failures: list[str] = []
    if not sections:
        missing = "__NO_CANDIDATE_SECTIONS__"
        table_failures.append(missing)
        business_quality_failures.append(missing)
        valuation_failures.append(missing)
        risk_failures.append(missing)
        thesis_failures.append(missing)
        falsifier_failures.append(missing)
        open_question_failures.append(missing)
        conviction_failures.append(missing)
    for ticker, block in sections.items():
        table_block = marker_section(block, "**5-Year Financial Table** (FY, cached SEC companyfacts; USD where reported)")
        if not table_block:
            table_block = block.split("**Business Quality**", 1)[0]
        if len(_table_data_rows(table_block)) < 3:
            table_failures.append(ticker)
        if "**Business Quality**" not in block:
            business_quality_failures.append(ticker)
        if "**Valuation And Expected-Return Cases**" not in block:
            valuation_failures.append(ticker)
        risks = _bullets(marker_section(block, "**Key Risks**"))
        if len(risks) < 3 or any(not _risk_bullet_has_business_fact(risk) for risk in risks):
            risk_failures.append(ticker)
        thesis = marker_section(block, "**Thesis**")
        if _sentence_count(thesis) < 3:
            thesis_failures.append(ticker)
        if not _bullets(marker_section(block, "**Falsifiers**")):
            falsifier_failures.append(ticker)
        if not _bullets(marker_section(block, "**Open Questions**")):
            open_question_failures.append(ticker)
        if "**Conviction Grade With Reasoning**" not in block:
            conviction_failures.append(ticker)

    _add(results, "sector_candidate_5y_table", not table_failures, f"{path}: candidates with <3 5Y table rows {table_failures}")
    _add(results, "sector_candidate_business_quality", not business_quality_failures, f"{path}: missing Business Quality for {business_quality_failures}")
    _add(results, "sector_candidate_valuation_cases", not valuation_failures, f"{path}: missing valuation cases for {valuation_failures}")
    _add(results, "sector_candidate_risks", not risk_failures, f"{path}: Key Risks missing/thin/enum-only for {risk_failures}")
    _add(results, "sector_candidate_thesis", not thesis_failures, f"{path}: Thesis has <3 sentences for {thesis_failures}")
    _add(results, "sector_candidate_falsifiers", not falsifier_failures, f"{path}: missing falsifiers for {falsifier_failures}")
    _add(results, "sector_candidate_open_questions", not open_question_failures, f"{path}: missing open questions for {open_question_failures}")
    _add(results, "sector_candidate_conviction_reasoning", not conviction_failures, f"{path}: missing conviction reasoning for {conviction_failures}")

    appendix = markdown_section(text, "## Audit Appendix")
    appendix_failures = [
        label for label in ["### Run Metadata", "### Selection Audit", "### Tool Calls", "### Evidence References", "### AI Working Notes"]
        if label not in appendix
    ]
    _add(results, "sector_audit_appendix", bool(appendix) and not appendix_failures, f"{path}: audit appendix missing {appendix_failures}")
    return results


def single_report_shape_results(path: Path) -> list[EvalCheckResult]:
    text = path.read_text()
    results: list[EvalCheckResult] = []
    required_sections = {
        "single_company_identity": "## Company Identity",
        "single_research_questions": "## Research Questions",
        "single_executed_tools": "## Executed Tools",
        "single_evidence": "## Evidence",
        "single_belief_updates": "## Belief Updates",
        "single_candidate_decision": "## Candidate Decision",
        "single_warnings_audit_notes": "## Warnings And Audit Notes",
    }
    for check, heading in required_sections.items():
        _add(results, check, bool(markdown_section(text, heading)), f"{path}: missing {heading}")
    return results


def report_shape_results(path: Path) -> list[EvalCheckResult]:
    if path.name == "autonomous_sector_report.md":
        return sector_report_shape_results(path)
    return single_report_shape_results(path)


def format_failures(path: Path, failures: list[EvalCheckResult]) -> str:
    lines = [f"{path}: {len(failures)} eval failure(s)"]
    for failure in failures:
        lines.append(f"- {failure.check}: {failure.message}")
    return "\n".join(lines)


def build_eval_summary() -> dict[str, Any]:
    artifact_paths = discover_artifact_paths()
    report_paths = discover_single_report_paths() + discover_sector_report_paths()
    check_counts: dict[str, dict[str, int]] = {}
    artifact_failures: dict[str, list[dict[str, str]]] = {}

    def record(path: Path, result: EvalCheckResult) -> None:
        bucket = check_counts.setdefault(result.check, {"passed": 0, "failed": 0})
        bucket["passed" if result.passed else "failed"] += 1
        if not result.passed:
            artifact_failures.setdefault(str(path), []).append({"check": result.check, "message": result.message})

    for path in artifact_paths:
        for result in artifact_soundness_results(path):
            record(path, result)
    for path in report_paths:
        for result in report_shape_results(path):
            record(path, result)

    return {
        "artifact_count": len(artifact_paths),
        "report_count": len(report_paths),
        "checks": check_counts,
        "artifacts_with_failures": artifact_failures,
    }
