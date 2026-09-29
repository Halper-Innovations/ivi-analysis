from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.dossier.collector import (
    DossierStage1Filing,
    NO_ANNUAL_FILING_ERROR,
    collect_10k_docket_stage1_with_debug,
    materialize_and_parse_docket_stage1,
    read_filing_text,
)
from app.dossier.dossier_writer import write_ticker_dossier
from app.dossier.extractors import extract_annual_items
from app.dossier.peer_report import build_peer_report
from app.dossier.sections import segment_10k_sections
from app.dossier.time_series import build_time_series
from app.evidence.packet_builder import build_packet_for_ticker
from app.ingest.facts_writer import ensure_all_facts
from app.llm.providers import get_llm_provider
from app.llm.synthesis_agent import append_synthesis_section, run_synthesis_for_ticker
from app.logging import get_logger
from app.valuation.valuation_writer import append_valuation_section, ensure_valuation
from app.util.http import DomainBudgetExceeded, temporary_sec_domain_budget


logger = get_logger(__name__)


def _default_run_id() -> str:
    return f"dossier_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def _sanitize_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ").strip()[:400]


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "dossier_summary.json"


def _read_summary(run_dir: Path) -> dict[str, Any] | None:
    path = _summary_path(run_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    summary["updated_at"] = datetime.now(timezone.utc).isoformat()
    _summary_path(run_dir).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _init_summary(
    *,
    run_id: str,
    run_dir: Path,
    as_of_date: str,
    years_back: int,
    min_annual_filings: int | None,
    workers: int,
    tickers: list[str],
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "years_back": int(years_back),
        "min_annual_filings": int(min_annual_filings) if min_annual_filings is not None else None,
        "workers": int(workers),
        "status": "RUNNING",
        "tickers_requested": list(tickers),
        "tickers_built": [],
        "tickers_failed": [],
        "tickers_skipped": [],
        "tickers_skipped_budget": [],
        "tickers_pending": list(tickers),
        "full_count": 0,
        "partial_count": 0,
        "failed_count": 0,
        "ticker_results": {
            ticker: {
                "status": "PENDING",
                "error": None,
                "dossier_json_path": None,
                "dossier_md_path": None,
                "dossier_quality": None,
                "filing_body_cached": None,
            }
            for ticker in tickers
        },
        "peer_report_path": None,
        "peer_rankings_path": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_summary(run_dir, summary)
    return summary


def _recompute_summary_counts(summary: dict[str, Any]) -> None:
    results = summary.get("ticker_results") or {}
    built: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    skipped_budget: list[str] = []
    pending: list[str] = []
    full_count = 0
    partial_count = 0
    for ticker in sorted(results.keys()):
        payload = results[ticker] or {}
        status = str(payload.get("status") or "PENDING").upper()
        if status == "OK":
            built.append(ticker)
            quality = str(payload.get("dossier_quality") or "FULL").upper()
            if quality == "PARTIAL":
                partial_count += 1
            else:
                full_count += 1
        elif status == "FAILED":
            failed.append(ticker)
        elif status == "SKIPPED":
            skipped.append(ticker)
        elif status == "SKIPPED_BUDGET":
            skipped_budget.append(ticker)
        else:
            pending.append(ticker)
    summary["tickers_built"] = built
    summary["tickers_failed"] = failed
    summary["tickers_skipped"] = skipped
    summary["tickers_skipped_budget"] = skipped_budget
    summary["tickers_pending"] = pending
    summary["tickers_synthesis_skipped"] = [
        ticker
        for ticker in sorted(results.keys())
        if (results[ticker] or {}).get("synthesis", {}).get("status") == "SKIPPED"
    ]
    summary["full_count"] = int(full_count)
    summary["partial_count"] = int(partial_count)
    summary["failed_count"] = len(failed)


def _update_ticker_status(
    summary: dict[str, Any],
    *,
    ticker: str,
    status: str,
    error: str | None = None,
    dossier_json_path: str | None = None,
    dossier_md_path: str | None = None,
    dossier_quality: str | None = None,
    filing_body_cached: bool | None = None,
    synthesis: dict[str, Any] | None = None,
) -> None:
    bucket = summary.setdefault("ticker_results", {})
    payload = bucket.setdefault(
        ticker,
        {
            "status": "PENDING",
            "error": None,
            "dossier_json_path": None,
            "dossier_md_path": None,
            "dossier_quality": None,
            "filing_body_cached": None,
        },
    )
    payload["status"] = status
    payload["error"] = error
    if dossier_json_path is not None:
        payload["dossier_json_path"] = dossier_json_path
    if dossier_md_path is not None:
        payload["dossier_md_path"] = dossier_md_path
    if dossier_quality is not None:
        payload["dossier_quality"] = str(dossier_quality).upper()
    if filing_body_cached is not None:
        payload["filing_body_cached"] = bool(filing_body_cached)
    if synthesis is not None:
        payload["synthesis"] = synthesis
    _recompute_summary_counts(summary)


def _serialize_docket(
    stage1: list[DossierStage1Filing], filing_id_map: dict[str, int]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in stage1:
        out.append(
            {
                "ticker": item.ticker,
                "cik": item.cik,
                "accession": item.filing.accession,
                "form_type": item.filing.form_type,
                "filing_date": item.filing.filing_date.isoformat(),
                "period_end": item.filing.period_end,
                "primary_doc_url": item.filing.primary_doc_url,
                "local_path": item.local_path,
                "filing_id": filing_id_map.get(item.filing.accession),
            }
        )
    return out


def _collect_stage1_for_ticker(
    *,
    ticker: str,
    as_of_date: str,
    years_back: int,
    min_annual_filings: int | None = None,
) -> tuple[list[DossierStage1Filing], dict[str, Any]]:
    return collect_10k_docket_stage1_with_debug(
        ticker=ticker,
        as_of_date=as_of_date,
        years_back=years_back,
        include_amendments=True,
        include_foreign=True,
        min_annual_filings=min_annual_filings,
    )


def _coerce_stage1_result(
    *,
    ticker: str,
    raw_result: Any,
    as_of_date: str,
    years_back: int,
) -> tuple[list[DossierStage1Filing], dict[str, Any]]:
    stage1: list[DossierStage1Filing] | Any = []
    debug: dict[str, Any] = {}
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        stage1, debug = raw_result
    else:
        stage1 = raw_result
    if not isinstance(stage1, list):
        stage1 = []
    if not isinstance(debug, dict):
        debug = {}
    if not debug:
        debug = {
            "ticker": ticker,
            "query": {
                "as_of_date": as_of_date,
                "years_back": int(years_back),
            },
            "selected_count": len(stage1),
            "selected_accessions": [],
            "missing_years": [],
            "note": "debug_payload_not_returned_by_collector",
        }
    return stage1, debug


def _write_ticker_debug_artifact(*, run_dir: Path, ticker: str, payload: dict[str, Any]) -> str:
    path = run_dir / f"dossier_debug_{ticker}.json"
    payload["debug_artifact_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _companyfacts_items_from_selected_filings(*, docket: list[Any]) -> list[dict[str, Any]]:
    """Build a partial dossier only from facts bound to selected filings."""

    items_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for filing in docket:
        for item in extract_annual_items(filing=filing, sections=[]):
            value = item.get("value")
            year = int(item.get("year") or 0)
            metric = str(item.get("metric") or "").strip()
            derived_from = [
                str(ref) for ref in (item.get("derived_from") or []) if str(ref).strip()
            ]
            if (
                year <= 0
                or not metric
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not any(
                    ref.startswith(("financials.", "extracted_facts.")) for ref in derived_from
                )
            ):
                continue
            items_by_key[(year, metric)] = item
    return [items_by_key[key] for key in sorted(items_by_key.keys())]


def _build_partial_ticker_payload_from_companyfacts(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    stage1: list[DossierStage1Filing],
    docket: list[Any],
    filing_id_map: dict[str, int],
) -> dict[str, Any] | None:
    items = _companyfacts_items_from_selected_filings(docket=docket)
    if not items:
        return None
    time_series = build_time_series(items)
    return write_ticker_dossier(
        run_id=run_id,
        ticker=ticker,
        as_of_date=as_of_date,
        docket=_serialize_docket(stage1, filing_id_map),
        section_spans={},
        items=items,
        time_series=time_series,
        dossier_quality="PARTIAL",
        filing_body_cached=False,
    )


def _run_layer3_synthesis(
    *, ticker: str, as_of_date: str, run_id: str, dossier_md: str
) -> dict[str, Any] | None:
    """Layer 3 synthesis; returns a report only when the step was skipped.

    The synthesis agent refuses prompt evidence that lacks point-in-time
    provenance (``InvalidFinancialInputError`` with ``PROMPT_*`` violations)
    before it looks at the provider. With a live provider that refusal must
    stop the run: unprovenanced numbers would reach a prompt. With the
    provider disabled nothing reaches any prompt and the only output is
    placeholder prose, so the refusal only costs that placeholder; the step is
    skipped, and the skip is reported (run summary and log), not silent. Any
    other integrity failure, or any failure with a live provider, still
    propagates. The gate itself is unchanged.
    """

    try:
        synth_path = run_synthesis_for_ticker(ticker, as_of_date=as_of_date, run_id=run_id)
    except InvalidFinancialInputError as exc:
        codes = sorted({v.code for v in exc.result.violations})
        prompt_evidence_only = bool(codes) and all(c.startswith("PROMPT_") for c in codes)
        if not prompt_evidence_only or _llm_provider_is_live():
            raise
        skipped = {
            "status": "SKIPPED",
            "reason": "LLM_DISABLED_PROMPT_EVIDENCE_REFUSED",
            "detail": _sanitize_error(exc),
            "violation_codes": codes,
        }
        logger.warning(
            "layer3 synthesis skipped for %s: LLM provider is disabled and the prompt evidence "
            "failed the financial-integrity gate (%s); the deterministic valuation is unaffected",
            ticker,
            ", ".join(codes),
        )
        return skipped
    if synth_path:
        append_synthesis_section(ticker=ticker, run_id=run_id, dossier_md_path=dossier_md)
    return None


def _llm_provider_is_live() -> bool:
    return getattr(get_llm_provider(), "provider_name", "disabled") != "disabled"


def _build_ticker_payload_from_stage1(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    stage1: list[DossierStage1Filing],
    years_back: int = 10,
) -> dict[str, Any] | None:
    docket = materialize_and_parse_docket_stage1(stage1=stage1, as_of_date=as_of_date)
    if not docket:
        return None

    # Ensure clean companyfacts data is available before extraction begins
    try:
        ensure_all_facts(ticker, years_back=years_back)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("companyfacts fetch failed for %s: %s", ticker, exc)

    section_spans: dict[str, list[dict[str, Any]]] = {}
    items: list[dict[str, Any]] = []
    filing_id_map: dict[str, int] = {}
    filing_body_presence: list[bool] = []
    for filing in docket:
        filing_id_map[filing.accession] = int(filing.filing_id)
        text = read_filing_text(filing)
        filing_body_presence.append(bool(text))
        if not text:
            continue
        spans = segment_10k_sections(text)
        section_spans[filing.accession] = [
            {
                "section_label": span.section_label,
                "start_offset": span.start_offset,
                "end_offset": span.end_offset,
            }
            for span in spans
        ]
        items.extend(extract_annual_items(filing=filing, sections=spans))

    if not items:
        return _build_partial_ticker_payload_from_companyfacts(
            ticker=ticker,
            as_of_date=as_of_date,
            run_id=run_id,
            stage1=stage1,
            docket=docket,
            filing_id_map=filing_id_map,
        )
    time_series = build_time_series(items)
    all_filing_bodies_cached = bool(filing_body_presence) and all(filing_body_presence)
    payload = write_ticker_dossier(
        run_id=run_id,
        ticker=ticker,
        as_of_date=as_of_date,
        docket=_serialize_docket(stage1, filing_id_map),
        section_spans=section_spans,
        items=items,
        time_series=time_series,
        dossier_quality="FULL" if all_filing_bodies_cached else "PARTIAL",
        filing_body_cached=all_filing_bodies_cached,
    )

    # Layer 2: append valuation section to dossier.md
    if payload:
        dossier_md = (payload.get("artifacts") or {}).get("dossier_md_path")
        try:
            ensure_valuation(
                ticker,
                as_of_date,
                run_id=run_id,
                require_filed_asof=True,
            )
            if dossier_md:
                append_valuation_section(ticker, as_of_date, dossier_md)
        except InvalidFinancialInputError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("valuation layer failed for %s: %s", ticker, exc)
        # Layer 2.5: filing diff (year-over-year MD&A and Risk Factor changes)
        try:
            from app.diff.engine import build_filing_diff_for_ticker

            build_filing_diff_for_ticker(ticker=ticker, run_id=run_id)
        except InvalidFinancialInputError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("filing diff failed for %s: %s", ticker, exc)

        try:
            build_packet_for_ticker(ticker, as_of_date, dossier_run_id=run_id)
            if dossier_md:
                skipped = _run_layer3_synthesis(
                    ticker=ticker, as_of_date=as_of_date, run_id=run_id, dossier_md=dossier_md
                )
                if skipped:
                    payload["synthesis"] = skipped
        except InvalidFinancialInputError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("layer3 synthesis failed for %s: %s", ticker, exc)

    return payload


def run_dossier_for_peer_set(
    *,
    tickers: list[str],
    as_of_date: str,
    years_back: int = 10,
    run_id: str | None = None,
    workers: int = 2,
    resume: bool = False,
    min_annual_filings: int | None = None,
    sec_budget: int | None = None,
) -> dict[str, Any]:
    run_id = run_id or _default_run_id()
    run_dir = get_config().dossiers_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with temporary_sec_domain_budget(sec_budget=sec_budget) as effective_sec_budget:
        tickers_norm = sorted({t.strip().upper() for t in tickers if t.strip()})
        existing = _read_summary(run_dir) if resume else None
        if existing:
            tickers_from_summary = [str(t).upper() for t in existing.get("tickers_requested") or []]
            if tickers_from_summary:
                tickers_norm = sorted({t for t in tickers_from_summary if t})
            summary = existing
            summary["status"] = "RUNNING"
            summary["workers"] = int(workers)
            summary["years_back"] = int(summary.get("years_back", years_back))
            summary["as_of_date"] = str(summary.get("as_of_date", as_of_date))
            if summary.get("min_annual_filings") is None and min_annual_filings is not None:
                summary["min_annual_filings"] = int(min_annual_filings)
        else:
            summary = _init_summary(
                run_id=run_id,
                run_dir=run_dir,
                as_of_date=as_of_date,
                years_back=years_back,
                min_annual_filings=min_annual_filings,
                workers=workers,
                tickers=tickers_norm,
            )
        summary["sec_budget"] = {
            "requested": int(sec_budget) if sec_budget is not None else None,
            "effective": effective_sec_budget,
        }
        _write_summary(run_dir, summary)

        if not tickers_norm:
            summary["status"] = "DONE"
            _write_summary(run_dir, summary)
            summary["summary_path"] = str(_summary_path(run_dir))
            return summary

        to_process = []
        for ticker in tickers_norm:
            state = str(
                (summary.get("ticker_results") or {}).get(ticker, {}).get("status") or "PENDING"
            ).upper()
            if state in {"OK"} and resume:
                continue
            to_process.append(ticker)

        max_workers = max(1, int(workers))
        if max_workers > 1:
            logger.info(
                "dossier_parallel_download_serial_parse",
                extra={"stage_name": "dossier", "stage_workers": max_workers},
            )

        stage1_by_ticker: dict[str, list[DossierStage1Filing]] = {}
        cancelled = False
        budget_exceeded = False
        budget_error: str | None = None
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _collect_stage1_for_ticker,
                        **{
                            "ticker": ticker,
                            "as_of_date": summary["as_of_date"],
                            "years_back": int(summary["years_back"]),
                            **(
                                {"min_annual_filings": summary.get("min_annual_filings")}
                                if summary.get("min_annual_filings") is not None
                                else {}
                            ),
                        },
                    ): ticker
                    for ticker in to_process
                }
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        raw_result = future.result()
                    except DomainBudgetExceeded as exc:
                        budget_exceeded = True
                        budget_error = _sanitize_error(exc)
                        _write_ticker_debug_artifact(
                            run_dir=run_dir,
                            ticker=ticker,
                            payload={
                                "ticker": ticker,
                                "query": {
                                    "as_of_date": summary["as_of_date"],
                                    "years_back": int(summary["years_back"]),
                                },
                                "selected_count": 0,
                                "selected_accessions": [],
                                "missing_years": [],
                                "error": budget_error,
                                "skip_reason": "BUDGET_EXCEEDED",
                            },
                        )
                        _update_ticker_status(
                            summary, ticker=ticker, status="SKIPPED_BUDGET", error=budget_error
                        )
                        for pending in futures.keys():
                            if pending is not future:
                                pending.cancel()
                        _write_summary(run_dir, summary)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        message = _sanitize_error(exc)
                        if "Domain budget exceeded" in message:
                            budget_exceeded = True
                            budget_error = message
                            _update_ticker_status(
                                summary, ticker=ticker, status="SKIPPED_BUDGET", error=message
                            )
                            for pending in futures.keys():
                                if pending is not future:
                                    pending.cancel()
                            _write_summary(run_dir, summary)
                            continue
                        _write_ticker_debug_artifact(
                            run_dir=run_dir,
                            ticker=ticker,
                            payload={
                                "ticker": ticker,
                                "query": {
                                    "as_of_date": summary["as_of_date"],
                                    "years_back": int(summary["years_back"]),
                                },
                                "selected_count": 0,
                                "selected_accessions": [],
                                "missing_years": [],
                                "error": message,
                            },
                        )
                        _update_ticker_status(
                            summary, ticker=ticker, status="FAILED", error=message
                        )
                        _write_summary(run_dir, summary)
                        continue

                    stage1, debug_payload = _coerce_stage1_result(
                        ticker=ticker,
                        raw_result=raw_result,
                        as_of_date=summary["as_of_date"],
                        years_back=int(summary["years_back"]),
                    )
                    debug_payload.setdefault("ticker", ticker)
                    debug_payload["run_id"] = run_id
                    debug_payload["debug_artifact_path"] = _write_ticker_debug_artifact(
                        run_dir=run_dir,
                        ticker=ticker,
                        payload=debug_payload,
                    )
                    if not stage1:
                        skip_reason = str(debug_payload.get("skip_reason") or "").upper()
                        if skip_reason in {
                            "NO_ANNUAL_FILING_IN_WINDOW",
                            "NO_ANNUAL_FILING_AFTER_YEAR_BUCKETING",
                            "INSUFFICIENT_ANNUAL_FILINGS_IN_WINDOW",
                        }:
                            error_message = NO_ANNUAL_FILING_ERROR
                        elif skip_reason == "CIK_MISSING":
                            error_message = "Missing CIK"
                        elif skip_reason == "COMPANY_NOT_FOUND":
                            error_message = "Company not found"
                        else:
                            error_message = NO_ANNUAL_FILING_ERROR
                        _update_ticker_status(
                            summary, ticker=ticker, status="SKIPPED", error=error_message
                        )
                        _write_summary(run_dir, summary)
                        continue
                    stage1_by_ticker[ticker] = stage1
        except KeyboardInterrupt:
            cancelled = True
            summary["status"] = "PARTIAL"
            _write_summary(run_dir, summary)

        if budget_exceeded:
            for ticker in to_process:
                state = str(
                    (summary.get("ticker_results") or {}).get(ticker, {}).get("status") or "PENDING"
                ).upper()
                if state == "PENDING":
                    _update_ticker_status(
                        summary,
                        ticker=ticker,
                        status="SKIPPED_BUDGET",
                        error=budget_error or "Domain budget exceeded",
                    )
            summary["budget_exceeded"] = {
                "error": budget_error,
                "remaining_marked_skipped_budget": True,
            }
            _write_summary(run_dir, summary)

        results: list[dict[str, Any]] = []
        staged_tickers = sorted(stage1_by_ticker.keys())
        for index, ticker in enumerate(staged_tickers):
            if cancelled:
                break
            try:
                payload = _build_ticker_payload_from_stage1(
                    ticker=ticker,
                    as_of_date=summary["as_of_date"],
                    run_id=run_id,
                    stage1=stage1_by_ticker[ticker],
                    years_back=int(summary["years_back"]),
                )
                if not payload:
                    _update_ticker_status(
                        summary,
                        ticker=ticker,
                        status="SKIPPED",
                        error="No extractable dossier items from selected filings",
                    )
                else:
                    results.append(payload)
                    artifacts = payload.get("artifacts") or {}
                    _update_ticker_status(
                        summary,
                        ticker=ticker,
                        status="OK",
                        error=None,
                        dossier_json_path=artifacts.get("dossier_json_path"),
                        dossier_md_path=artifacts.get("dossier_md_path"),
                        dossier_quality=payload.get("dossier_quality"),
                        filing_body_cached=payload.get("filing_body_cached"),
                        synthesis=payload.get("synthesis"),
                    )
            except DomainBudgetExceeded as exc:
                budget_exceeded = True
                budget_error = _sanitize_error(exc)
                _update_ticker_status(
                    summary, ticker=ticker, status="SKIPPED_BUDGET", error=budget_error
                )
                for remaining in staged_tickers[index + 1 :]:
                    state = str(
                        (summary.get("ticker_results") or {}).get(remaining, {}).get("status")
                        or "PENDING"
                    ).upper()
                    if state == "PENDING":
                        _update_ticker_status(
                            summary,
                            ticker=remaining,
                            status="SKIPPED_BUDGET",
                            error=budget_error,
                        )
                break
            except KeyboardInterrupt:
                cancelled = True
                _update_ticker_status(
                    summary, ticker=ticker, status="FAILED", error="Interrupted by user"
                )
            except InvalidFinancialInputError as exc:
                message = _sanitize_error(exc)
                _update_ticker_status(
                    summary,
                    ticker=ticker,
                    status="FAILED",
                    error=message,
                )
                summary["status"] = "FAILED"
                summary["stop_reason_code"] = exc.status
                summary["stop_summary"] = message
                summary["financial_integrity"] = exc.result.to_dict()
                _write_summary(run_dir, summary)
                raise
            except Exception as exc:  # noqa: BLE001
                message = _sanitize_error(exc)
                if "Domain budget exceeded" in message:
                    budget_exceeded = True
                    budget_error = message
                    _update_ticker_status(
                        summary, ticker=ticker, status="SKIPPED_BUDGET", error=message
                    )
                    for remaining in staged_tickers[index + 1 :]:
                        state = str(
                            (summary.get("ticker_results") or {}).get(remaining, {}).get("status")
                            or "PENDING"
                        ).upper()
                        if state == "PENDING":
                            _update_ticker_status(
                                summary,
                                ticker=remaining,
                                status="SKIPPED_BUDGET",
                                error=message,
                            )
                    break
                logger.exception("dossier build failed for %s", ticker)
                _update_ticker_status(summary, ticker=ticker, status="FAILED", error=message)
            _write_summary(run_dir, summary)

        results.sort(key=lambda row: row.get("ticker", ""))

        # Post-loop: cross-sectional pattern scan across all tickers in this run
        if results:
            try:
                from app.patterns.scanner import scan_peer_set

                built_tickers = [
                    r.get("ticker") for r in results if isinstance(r, dict) and r.get("ticker")
                ]
                if len(built_tickers) >= 3:
                    pattern_report = scan_peer_set(run_id=run_id, tickers=built_tickers)
                    summary["pattern_scan"] = {
                        "peer_set_size": pattern_report.peer_set_size,
                        "patterns_with_signal": pattern_report.patterns_with_signal,
                        "recency_weighted_hit_count": pattern_report.recency_weighted_hit_count,
                    }
                    logger.info(
                        "Pattern scan complete: %d tickers, %d patterns with signal",
                        pattern_report.peer_set_size,
                        len(pattern_report.patterns_with_signal),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("pattern scan failed for run %s: %s", run_id, exc)
                summary["pattern_scan"] = {"error": str(exc)}

        if results:
            try:
                peer = build_peer_report(
                    run_id=run_id,
                    as_of_date=summary["as_of_date"],
                    dossiers=results,
                )
                summary["peer_report_path"] = peer.get("peer_report_path")
                summary["peer_rankings_path"] = peer.get("peer_rankings_path")
            except Exception as exc:  # noqa: BLE001
                summary["peer_report_path"] = None
                summary["peer_rankings_path"] = None
                summary["peer_error"] = _sanitize_error(exc)

        _recompute_summary_counts(summary)
        if (
            cancelled
            or summary["tickers_failed"]
            or summary["tickers_pending"]
            or summary["tickers_skipped_budget"]
        ):
            summary["status"] = "PARTIAL"
        else:
            summary["status"] = "DONE"
        _write_summary(run_dir, summary)
        logger.info(
            "Dossier run complete for %s: written=%s full=%s partial=%s attempted=%s failed=%s skipped=%s skipped_budget=%s",
            run_id,
            len(summary.get("tickers_built") or []),
            int(summary.get("full_count") or 0),
            int(summary.get("partial_count") or 0),
            len(summary.get("tickers_requested") or []),
            len(summary.get("tickers_failed") or []),
            len(summary.get("tickers_skipped") or []),
            len(summary.get("tickers_skipped_budget") or []),
        )
        summary["summary_path"] = str(_summary_path(run_dir))
        return summary


def resume_dossier_run(*, run_id: str, workers: int = 2) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.dossiers_dir / run_id
    summary = _read_summary(run_dir)
    if not summary:
        raise ValueError(f"dossier summary not found for run_id={run_id}")
    return run_dossier_for_peer_set(
        tickers=[str(t).upper() for t in (summary.get("tickers_requested") or [])],
        as_of_date=str(summary.get("as_of_date") or datetime.now(timezone.utc).date().isoformat()),
        years_back=int(summary.get("years_back") or 10),
        run_id=run_id,
        workers=workers,
        resume=True,
        min_annual_filings=(
            int(summary["min_annual_filings"])
            if summary.get("min_annual_filings") is not None
            else None
        ),
    )


def open_dossier_run(run_id: str) -> dict[str, Any]:
    cfg = get_config()
    base = cfg.dossiers_dir / run_id
    if not base.exists():
        raise ValueError(f"dossier run not found: {run_id}")
    summary = _read_summary(base) or {}
    peer_report_path = base / "peer_report.md"
    peer_rankings_path = base / "peer_rankings.json"
    ticker_dirs = sorted([p.name for p in base.iterdir() if p.is_dir()])
    payload = {
        "run_id": run_id,
        "status": summary.get("status", "UNKNOWN"),
        "run_dir": str(base),
        "summary_path": str(_summary_path(base)) if _summary_path(base).exists() else None,
        "peer_report_path": str(peer_report_path) if peer_report_path.exists() else None,
        "peer_rankings_path": str(peer_rankings_path) if peer_rankings_path.exists() else None,
        "tickers": ticker_dirs,
        "tickers_requested": summary.get("tickers_requested", ticker_dirs),
        "ticker_results": summary.get("ticker_results", {}),
        "missing_artifacts": [],
    }
    if not peer_report_path.exists():
        payload["missing_artifacts"].append("peer_report.md")
    if not peer_rankings_path.exists():
        payload["missing_artifacts"].append("peer_rankings.json")
    return payload


def compare_dossier_metric(*, run_id: str, metric: str) -> dict[str, Any]:
    cfg = get_config()
    run_dir = cfg.dossiers_dir / run_id
    if not run_dir.exists():
        raise ValueError(f"dossier run not found: {run_id}")
    rankings_path = run_dir / "peer_rankings.json"
    if not rankings_path.exists():
        status = (_read_summary(run_dir) or {}).get("status", "UNKNOWN")
        raise ValueError(
            f"peer rankings not available for run_id={run_id} (status={status}). "
            "Run may be partial; resume with `dossier-resume`."
        )
    payload = json.loads(rankings_path.read_text(encoding="utf-8"))
    rows = payload.get("rankings") or []
    out_rows = []
    missing_metric_tickers: list[str] = []
    for row in rows:
        ticker = row.get("ticker")
        metric_values = row.get("metric_values") or {}
        metric_ranks = row.get("metric_ranks") or {}
        if metric not in metric_values:
            missing_metric_tickers.append(str(ticker))
            continue
        value = metric_values.get(metric, "UNKNOWN")
        rank = metric_ranks.get(metric, None)
        out_rows.append({"ticker": ticker, "value": value, "rank": rank})
    if missing_metric_tickers:
        missing_metric_tickers = sorted({t for t in missing_metric_tickers if t})
        raise ValueError(f"metric missing for ticker(s): {', '.join(missing_metric_tickers)}")
    out_rows.sort(key=lambda r: (r["rank"] if isinstance(r["rank"], int) else 10_000, r["ticker"]))
    return {"run_id": run_id, "metric": metric, "rows": out_rows}
