from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.autonomous.v1_financial_context import (
    bind_v1_financial_scope,
    build_canonical_v1_financial_context,
    financial_input_scenario,
)
from app.llm.providers.retry_guard import llm_physical_attempt_guard
from app.config import get_config
from app.db import utc_now_iso
from app.dossier.sections import section_by_label, segment_10k_sections
from app.llm.providers import get_llm_provider
from app.llm.usage_capture import (
    attach_provider_usage_to_exception,
    provider_failed_attempt_capture,
    provider_usage_records,
    provider_usage_records_from_exception,
    provider_usage_request,
    record_provider_usage,
)
from app.logging import get_logger
from app.util.hashing import sha256_text
from app.diff.schemas import (
    FilingChange,
    FilingDiffReport,
    SectionDiagnostic,
    SectionSkip,
    filing_change_set_schema_for_prompt,
    validate_filing_change_set,
)


logger = get_logger(__name__)

SECTION_PRIORITY: tuple[str, ...] = ("risk_factors", "md_and_a")
MIN_SECTION_CHARS = 200
MAX_SECTION_CHARS = 32000
MAX_CHANGES_PER_SECTION_PAIR = 2
MAX_TOTAL_CHANGES = 10


@dataclass(frozen=True)
class FilingSnapshot:
    accession: str
    filing_date: str
    fiscal_year: int
    local_path: str


def build_filing_diff_for_ticker(*, ticker: str, run_id: str, years_back: int = 5) -> Path:
    report = build_filing_diff_report(ticker=ticker, run_id=run_id, years_back=years_back)
    cfg = get_config()
    out_dir = cfg.outputs_dir / "diffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker.upper()}_{run_id}_diff.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def build_filing_diff_report(*, ticker: str, run_id: str, years_back: int = 5) -> FilingDiffReport:
    ticker_norm = str(ticker or "").strip().upper()
    dossier = _load_dossier(run_id=run_id, ticker=ticker_norm)
    as_of_date = str(dossier.get("as_of_date") or "")
    snapshots = _load_snapshots(dossier, years_back=years_back)
    provider = get_llm_provider()
    llm_enabled = bool(provider.enabled())
    canonical_packet = None
    if llm_enabled:
        financial_context = build_canonical_v1_financial_context(
            tickers=[ticker_norm],
            as_of_date=as_of_date,
            db_path=get_config().db_path,
        )
        canonical_packet = financial_context.packets[ticker_norm]

    years_compared: list[tuple[int, int]] = []
    changes: list[FilingChange] = []
    sections_compared: set[str] = set()
    sections_skipped: list[SectionSkip] = []
    section_diagnostics: list[SectionDiagnostic] = []
    truncation_applied = False

    for newer, older in zip(snapshots, snapshots[1:], strict=False):
        years_compared.append((older.fiscal_year, newer.fiscal_year))
        newer_sections = _load_sections(newer)
        older_sections = _load_sections(older)
        for section_name in SECTION_PRIORITY:
            newer_text = newer_sections.get(section_name)
            older_text = older_sections.get(section_name)
            if not newer_text or not older_text:
                sections_skipped.append(
                    SectionSkip(
                        section=section_name,
                        fiscal_year_from=older.fiscal_year,
                        fiscal_year_to=newer.fiscal_year,
                        reason="SECTION_MISSING",
                    )
                )
                section_diagnostics.append(
                    SectionDiagnostic(
                        section=section_name,
                        fiscal_year_from=older.fiscal_year,
                        fiscal_year_to=newer.fiscal_year,
                        similarity_ratio=None,
                        compared_chars_from=len(older_text or ""),
                        compared_chars_to=len(newer_text or ""),
                        llm_used=False,
                        llm_status="skipped",
                    )
                )
                continue
            if len(newer_text) < MIN_SECTION_CHARS or len(older_text) < MIN_SECTION_CHARS:
                sections_skipped.append(
                    SectionSkip(
                        section=section_name,
                        fiscal_year_from=older.fiscal_year,
                        fiscal_year_to=newer.fiscal_year,
                        reason="SECTION_TOO_SHORT",
                    )
                )
                section_diagnostics.append(
                    SectionDiagnostic(
                        section=section_name,
                        fiscal_year_from=older.fiscal_year,
                        fiscal_year_to=newer.fiscal_year,
                        similarity_ratio=_similarity_ratio(older_text, newer_text),
                        compared_chars_from=len(older_text),
                        compared_chars_to=len(newer_text),
                        llm_used=False,
                        llm_status="skipped",
                    )
                )
                continue

            older_trunc, older_truncated = _truncate_section_text(older_text)
            newer_trunc, newer_truncated = _truncate_section_text(newer_text)
            truncation_applied = truncation_applied or older_truncated or newer_truncated
            similarity_ratio = _similarity_ratio(older_trunc, newer_trunc)
            sections_compared.add(section_name)

            if llm_enabled:
                try:
                    llm_changes = _classify_material_changes(
                        ticker=ticker_norm,
                        section=section_name,
                        fiscal_year_from=older.fiscal_year,
                        fiscal_year_to=newer.fiscal_year,
                        older_text=older_trunc,
                        newer_text=newer_trunc,
                        canonical_packet=canonical_packet,
                        run_as_of_date=as_of_date,
                    )
                    changes.extend(llm_changes)
                    section_diagnostics.append(
                        SectionDiagnostic(
                            section=section_name,
                            fiscal_year_from=older.fiscal_year,
                            fiscal_year_to=newer.fiscal_year,
                            similarity_ratio=similarity_ratio,
                            compared_chars_from=len(older_trunc),
                            compared_chars_to=len(newer_trunc),
                            truncated_from=older_truncated,
                            truncated_to=newer_truncated,
                            llm_used=True,
                            llm_status="ok",
                        )
                    )
                except InvalidFinancialInputError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "filing_diff_llm_error",
                        extra={
                            "stage_name": "filing_diff",
                            "stage_ticker": ticker_norm,
                            "stage_section": section_name,
                            "stage_fiscal_year_from": older.fiscal_year,
                            "stage_fiscal_year_to": newer.fiscal_year,
                            "stage_error": str(exc)[:240],
                        },
                    )
                    section_diagnostics.append(
                        SectionDiagnostic(
                            section=section_name,
                            fiscal_year_from=older.fiscal_year,
                            fiscal_year_to=newer.fiscal_year,
                            similarity_ratio=similarity_ratio,
                            compared_chars_from=len(older_trunc),
                            compared_chars_to=len(newer_trunc),
                            truncated_from=older_truncated,
                            truncated_to=newer_truncated,
                            llm_used=False,
                            llm_status="error",
                            llm_error=str(exc)[:400],
                        )
                    )
            else:
                section_diagnostics.append(
                    SectionDiagnostic(
                        section=section_name,
                        fiscal_year_from=older.fiscal_year,
                        fiscal_year_to=newer.fiscal_year,
                        similarity_ratio=similarity_ratio,
                        compared_chars_from=len(older_trunc),
                        compared_chars_to=len(newer_trunc),
                        truncated_from=older_truncated,
                        truncated_to=newer_truncated,
                        llm_used=False,
                        llm_status="disabled",
                    )
                )

    changes = _cap_total_changes(changes)

    return FilingDiffReport(
        ticker=ticker_norm,
        run_id=run_id,
        as_of_date=as_of_date,
        years_compared=years_compared,
        changes=changes,
        sections_compared=sorted(sections_compared),
        sections_skipped=sections_skipped,
        section_diagnostics=section_diagnostics,
        truncation_applied=truncation_applied,
        llm_enabled=llm_enabled,
        created_at=utc_now_iso(),
    )


def summarize_diff_report(report: FilingDiffReport) -> dict[str, Any]:
    section_counts = Counter(change.section for change in report.changes)
    materiality_counts = Counter(change.materiality for change in report.changes)
    return {
        "ticker": report.ticker,
        "run_id": report.run_id,
        "llm_enabled": report.llm_enabled,
        "changes_found": len(report.changes),
        "sections_compared": report.sections_compared,
        "sections_skipped": [row.model_dump(mode="json") for row in report.sections_skipped],
        "count_by_section": dict(sorted(section_counts.items())),
        "count_by_materiality": dict(sorted(materiality_counts.items())),
        "truncation_applied": report.truncation_applied,
    }


def list_dossier_run_tickers(*, run_id: str) -> list[str]:
    cfg = get_config()
    run_dir = cfg.dossiers_dir / run_id
    if not run_dir.exists():
        raise ValueError(f"dossier run not found: {run_id}")
    out: list[str] = []
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and (child / "dossier.json").exists():
            out.append(child.name.upper())
    return out


def _load_dossier(*, run_id: str, ticker: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.dossiers_dir / run_id / ticker / "dossier.json"
    if not path.exists():
        raise ValueError(f"dossier.json not found for ticker={ticker} run_id={run_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid dossier payload")
    return payload


def _load_snapshots(dossier: dict[str, Any], *, years_back: int) -> list[FilingSnapshot]:
    docket = dossier.get("docket") or []
    snapshots: list[FilingSnapshot] = []
    seen_years: set[int] = set()
    for row in docket:
        if not isinstance(row, dict):
            continue
        local_path = str(row.get("local_path") or "").strip()
        if not local_path:
            continue
        fiscal_year = _fiscal_year_for_row(row)
        if fiscal_year in seen_years:
            continue
        seen_years.add(fiscal_year)
        snapshots.append(
            FilingSnapshot(
                accession=str(row.get("accession") or ""),
                filing_date=str(row.get("filing_date") or ""),
                fiscal_year=fiscal_year,
                local_path=local_path,
            )
        )
    snapshots.sort(key=lambda item: item.fiscal_year, reverse=True)
    return snapshots[: max(2, int(years_back))]


def _fiscal_year_for_row(row: dict[str, Any]) -> int:
    period_end = str(row.get("period_end") or "").strip()
    filing_date = str(row.get("filing_date") or "").strip()
    value = period_end or filing_date
    return int(value[:4])


def _load_sections(snapshot: FilingSnapshot) -> dict[str, str]:
    path = Path(snapshot.local_path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    spans = segment_10k_sections(text)
    out: dict[str, str] = {}
    for name in SECTION_PRIORITY:
        span = section_by_label(spans, name)
        if span and span.text:
            out[name] = span.text
    return out


def _truncate_section_text(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_SECTION_CHARS:
        return text, False
    return text[:MAX_SECTION_CHARS], True


def _similarity_ratio(left: str, right: str) -> float:
    return round(float(SequenceMatcher(None, left, right).ratio()), 4)


def _classify_material_changes(
    *,
    ticker: str,
    section: str,
    fiscal_year_from: int,
    fiscal_year_to: int,
    older_text: str,
    newer_text: str,
    canonical_packet: Any,
    run_as_of_date: str,
) -> list[FilingChange]:
    provider = get_llm_provider()
    prompt = _build_diff_prompt(
        ticker=ticker,
        section=section,
        fiscal_year_from=fiscal_year_from,
        fiscal_year_to=fiscal_year_to,
        older_text=older_text,
        newer_text=newer_text,
    )
    schema = filing_change_set_schema_for_prompt()
    schema_name = "filing_change_set_v1"
    cfg = get_config()
    max_output_tokens = (
        int(cfg.openai_max_output_tokens)
        if getattr(provider, "provider_name", "") == "openai"
        else int(cfg.anthropic_max_output_tokens)
    )

    def current_financial_scenario() -> dict[str, Any]:
        return financial_input_scenario(
            canonical_packet,
            financial_inputs={
                "filing_diff": {
                    "ticker": ticker,
                    "section": section,
                    "fiscal_year_from": fiscal_year_from,
                    "fiscal_year_to": fiscal_year_to,
                },
                "provider_prompt": prompt,
                "provider_schema": schema,
                "provider_schema_name": schema_name,
                "provider_name": getattr(provider, "provider_name", ""),
                "provider_model": getattr(provider, "model", ""),
                "provider_max_output_tokens": max_output_tokens,
            },
        )

    financial_scope = bind_v1_financial_scope(
        context=(f"filing_diff:{ticker}:{section}:{fiscal_year_from}:{fiscal_year_to}"),
        run_as_of_date=run_as_of_date,
        packets=(canonical_packet,),
        scenarios=(current_financial_scenario(),),
    )

    def require_exact_scope(_attempt=None):
        financial_scope.require(scenarios=(current_financial_scenario(),))

    failed_attempts: list[dict[str, Any]] = []
    successful_attempts: list[dict[str, Any]] = []
    try:
        with provider_usage_request(
            provider=provider,
            prompt=prompt,
            schema=schema,
            schema_name=schema_name,
            max_output_tokens=max_output_tokens,
        ) as request_kwargs:
            try:
                with (
                    provider_failed_attempt_capture(
                        provider=provider,
                        prompt=prompt,
                        schema_name=schema_name,
                        estimated_output_tokens=max_output_tokens,
                    ) as failed_attempts,
                    llm_physical_attempt_guard(require_exact_scope),
                ):
                    require_exact_scope()
                    result = provider.synthesize_json(
                        prompt=prompt,
                        schema=schema,
                        schema_name=schema_name,
                        **request_kwargs,
                    )
            except BaseException as exc:
                successful_attempts = provider_usage_records_from_exception(
                    provider=provider,
                    error=exc,
                    prompt=prompt,
                    schema_name=schema_name,
                )
                for usage_record in successful_attempts:
                    record_provider_usage(usage_record)
                attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
                try:
                    require_exact_scope()
                except InvalidFinancialInputError as integrity_exc:
                    attach_provider_usage_to_exception(
                        integrity_exc,
                        [*failed_attempts, *successful_attempts],
                    )
                    raise integrity_exc from exc
                raise
            successful_attempts = provider_usage_records(
                provider=provider,
                result=result,
                prompt=prompt,
                schema_name=schema_name,
            )
            for usage_record in successful_attempts:
                record_provider_usage(usage_record)

        require_exact_scope()
        payload = json.loads(result.json_text)
        if not isinstance(payload, dict):
            raise RuntimeError("filing diff output was not a JSON object")
        validated = validate_filing_change_set(payload)
        changes = _postprocess_changes(
            validated.changes,
            section=section,
            fiscal_year_from=fiscal_year_from,
            fiscal_year_to=fiscal_year_to,
        )
        require_exact_scope()
        return changes
    except BaseException as exc:
        attach_provider_usage_to_exception(exc, [*failed_attempts, *successful_attempts])
        raise


def _build_diff_prompt(
    *,
    ticker: str,
    section: str,
    fiscal_year_from: int,
    fiscal_year_to: int,
    older_text: str,
    newer_text: str,
) -> str:
    prompt_payload = {
        "ticker": ticker,
        "section": section,
        "fiscal_year_from": fiscal_year_from,
        "fiscal_year_to": fiscal_year_to,
        "truncation_note": (
            f"Each section may be truncated to the first {MAX_SECTION_CHARS} characters for cost control."
        ),
        "comparison_rule": "Identify only material disclosure changes. Ignore cosmetic wording, formatting, and boilerplate refreshes.",
        "change_type_examples": {
            "NEW_DISCLOSURE": "Later filing adds a new litigation, concentration, cybersecurity, or acquisition disclosure.",
            "REMOVED_DISCLOSURE": "Later filing removes a prior risk or dependency disclosure.",
            "LANGUAGE_SHIFT": "Management shifts from investing/building language to harvesting/completed language.",
            "QUANTITATIVE_CHANGE": "A threshold, concentration percentage, or stated exposure meaningfully changes.",
            "COMPETITIVE_SIGNAL": "A new competitor, substitute, or competitive pressure is acknowledged.",
            "RISK_SIGNAL": "Risk factor language becomes materially more severe, specific, or expansive.",
            "STRATEGIC_SIGNAL": "Business or MD&A language signals a strategic repositioning or capital allocation shift.",
        },
        "instructions": [
            "Return ONLY JSON matching the schema.",
            "If no material changes exist, return an empty changes list.",
            "Do not fabricate changes.",
            "Return at most 3 changes for this section pair.",
            "Summaries should explain why the change might matter to an investor.",
            "Use short excerpts directly tied to the change. Keep excerpts concise.",
            "Do NOT report routine annual KPI refreshes, ordinary revenue/subscriber growth updates, or generic restatements of current-year performance unless they reflect an accounting change, segment reporting change, strategic shift, new dependency, new risk category, or other disclosure investors would likely miss.",
            "Prefer changes involving acquisitions, partnerships, AI strategy, segment reorganization, accounting estimate or policy changes, impairment risk, compliance burdens, cybersecurity, litigation, competitive threats, and explicit management framing shifts.",
        ],
        "earlier_section_text": older_text,
        "later_section_text": newer_text,
    }
    compact = json.dumps(prompt_payload, sort_keys=True)
    return (
        "You are the VOE Filing Diff Engine.\n"
        "Compare the same 10-K section across consecutive annual filings.\n"
        "Focus on substantive disclosure changes only.\n"
        f"PROMPT_HASH_HINT={sha256_text(compact)}\n"
        f"INPUT_PAYLOAD={compact}"
    )


def _postprocess_changes(
    changes: list[FilingChange],
    *,
    section: str,
    fiscal_year_from: int,
    fiscal_year_to: int,
) -> list[FilingChange]:
    filtered: list[FilingChange] = []
    for change in changes:
        if change.section != section:
            continue
        if change.fiscal_year_from != fiscal_year_from or change.fiscal_year_to != fiscal_year_to:
            continue
        if _is_routine_metric_refresh(change):
            continue
        filtered.append(change)

    ranked = sorted(filtered, key=_change_score, reverse=True)
    deduped: list[FilingChange] = []
    for change in ranked:
        if any(_summaries_overlap(change.summary, prior.summary) for prior in deduped):
            continue
        deduped.append(change)
        if len(deduped) >= MAX_CHANGES_PER_SECTION_PAIR:
            break
    return deduped


def _change_score(change: FilingChange) -> tuple[int, int]:
    materiality_score = {"HIGH": 30, "MEDIUM": 20, "LOW": 10}.get(change.materiality, 0)
    type_score = {
        "NEW_DISCLOSURE": 14,
        "REMOVED_DISCLOSURE": 12,
        "STRATEGIC_SIGNAL": 12,
        "RISK_SIGNAL": 10,
        "COMPETITIVE_SIGNAL": 9,
        "LANGUAGE_SHIFT": 7,
        "QUANTITATIVE_CHANGE": 3,
    }.get(change.change_type, 0)
    text = " ".join(
        part
        for part in [change.summary, change.from_excerpt or "", change.to_excerpt or ""]
        if part
    ).lower()
    keyword_bonus = 0
    for needle in [
        "openai",
        "activision",
        "nuance",
        "segment",
        "recast",
        "accounting",
        "useful life",
        "impairment",
        "goodwill",
        "intangible",
        "responsible ai",
        "cybersecurity",
        "compliance",
        "workforce",
        "acquisition",
        "partnership",
    ]:
        if needle in text:
            keyword_bonus += 3
    if change.section == "risk_factors":
        keyword_bonus += 2
    return materiality_score + type_score + keyword_bonus, len(change.summary)


def _is_routine_metric_refresh(change: FilingChange) -> bool:
    text = " ".join(
        part
        for part in [change.summary, change.from_excerpt or "", change.to_excerpt or ""]
        if part
    ).lower()
    strategic_needles = [
        "segment",
        "recast",
        "accounting",
        "useful life",
        "openai",
        "activision",
        "nuance",
        "acquisition",
        "partnership",
        "impairment",
        "goodwill",
        "intangible",
        "responsible ai",
        "cybersecurity",
        "workforce",
        "layoff",
        "compliance",
        "ftc",
        "regulatory",
        "legal",
        "charge",
    ]
    if any(needle in text for needle in strategic_needles):
        return False
    if change.change_type == "REMOVED_DISCLOSURE":
        return True
    routine_needles = [
        "revenue increased",
        "revenue growth",
        "subscriber",
        "subscribers",
        "microsoft cloud revenue",
        "windows revenue",
        "windows oem",
        "headline financial highlights",
        "foreign exchange",
        "did not have a material impact",
        "reported revenue and expenses",
        "formerly commercial cloud",
        "presentation and method of calculation",
        "presentation changed",
        "metrics presentation",
        "metric definitions",
        "rebrands the disclosure concept",
        "deemphasis",
        "editorial update",
    ]
    if change.change_type == "QUANTITATIVE_CHANGE" and any(
        needle in text for needle in routine_needles
    ):
        return True
    if change.section == "md_and_a" and change.change_type == "QUANTITATIVE_CHANGE":
        return True
    if change.change_type == "LANGUAGE_SHIFT" and any(needle in text for needle in routine_needles):
        return True
    if change.change_type == "RISK_SIGNAL" and "foreign exchange" in text:
        return True
    if "completed our acquisition" in text and "different stated price" in text:
        return True
    if (
        "metric and presentation updates" in text
        and "none of these changes had a material impact" in text
    ):
        return True
    return False


def _summaries_overlap(left: str, right: str) -> bool:
    left_norm = " ".join(left.lower().split())
    right_norm = " ".join(right.lower().split())
    if not left_norm or not right_norm:
        return False
    return SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.75


def _cap_total_changes(changes: list[FilingChange]) -> list[FilingChange]:
    ranked = sorted(changes, key=_change_score, reverse=True)
    kept: list[FilingChange] = []
    for change in ranked:
        if any(
            existing.section == change.section
            and existing.fiscal_year_from == change.fiscal_year_from
            and existing.fiscal_year_to == change.fiscal_year_to
            and _summaries_overlap(existing.summary, change.summary)
            for existing in kept
        ):
            continue
        kept.append(change)
        if len(kept) >= MAX_TOTAL_CHANGES:
            break
    chronology = {
        (change.fiscal_year_from, change.fiscal_year_to): idx for idx, change in enumerate(changes)
    }
    return sorted(
        kept,
        key=lambda change: (
            chronology.get((change.fiscal_year_from, change.fiscal_year_to), 999),
            change.section,
            -_change_score(change)[0],
        ),
    )
