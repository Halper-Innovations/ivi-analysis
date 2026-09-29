from __future__ import annotations

import json
import secrets
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from app.ingest.cache_readiness import cache_readiness_by_sector
from app.autonomous.sector_candidates import MARKET_CAP_FOCUS_TIERS
from app.config import AppConfig, ensure_directories, get_config
from app.db import get_db, init_db
from app.ingest.cik_registry import resolve as resolve_cik
from app.ingest.facts_writer import ensure_all_facts
from app.ingest.filings import ingest_with_policy
from app.market.price_provider import write_prices_for_run
from app.research.filing_context import load_research_filing_context
from app.util.dates import parse_yyyy_mm_dd, utc_now
from app.util.financial_data_access import ANNUAL_COMPANYFACTS_PERIOD_TYPES, companyfacts_is_fresh


_COMPLETE_STEP_STATUSES = {"OK", "SKIPPED", "DISABLED", "NO_DATA", "MISSING", "NO_FILINGS", "NO_READABLE_FILINGS"}
_INCOMPLETE_STEP_STATUSES = {"", "PENDING", "FAILED", "ERROR"}
SECTOR_SELECTION_CACHE_ORDER = "cache_order"
SECTOR_SELECTION_AUTONOMOUS_CANDIDATES = "autonomous_candidates"
SECTOR_SELECTION_MODES = {SECTOR_SELECTION_CACHE_ORDER, SECTOR_SELECTION_AUTONOMOUS_CANDIDATES}


def _normalize_tickers(values: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if values is None:
        return []
    raw = values.split(",") if isinstance(values, str) else list(values)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        ticker = str(item or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        out.append(ticker)
        seen.add(ticker)
    return out


def _normalize_sectors(values: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if values is None:
        return []
    raw = values.split(",") if isinstance(values, str) else list(values)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        sector = str(item or "").strip()
        if not sector or sector in seen:
            continue
        out.append(sector)
        seen.add(sector)
    return out


def _generated_run_id(created_at: str) -> str:
    date_part = str(created_at)[:10].replace("-", "")
    return f"financial_cache_refresh_{date_part}_{secrets.token_hex(3)}"


def _run_dir(cfg: AppConfig, run_id: str) -> Path:
    return cfg.runs_dir / "financial_cache_refresh" / run_id


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _ticker_sources_add(sources: dict[str, list[str]], ticker: str, source: str) -> None:
    ticker_norm = str(ticker or "").strip().upper()
    if not ticker_norm:
        return
    bucket = sources.setdefault(ticker_norm, [])
    if source not in bucket:
        bucket.append(source)


def _cap_bounds(market_cap_focus: str) -> tuple[float | None, float | None, str | None]:
    key = str(market_cap_focus or "smid_cap").strip().lower()
    if key in MARKET_CAP_FOCUS_TIERS:
        cap_min, cap_max = MARKET_CAP_FOCUS_TIERS[key]
        return cap_min, cap_max, None
    return None, None, f"UNKNOWN_MARKET_CAP_FOCUS:{market_cap_focus}"


def _sector_refresh_tickers(
    *,
    sector: str,
    market_cap_focus: str,
    limit: int | None,
    db_path: str | Path | None,
) -> dict[str, Any]:
    """Load sector tickers without invoking signal packets or LLM-backed scans."""

    cap_min, cap_max, cap_warning = _cap_bounds(market_cap_focus)
    warnings = [cap_warning] if cap_warning else []
    try:
        from app.sector.scan import load_sector_tickers

        loaded_rows = load_sector_tickers(
            sector=sector,
            db_path=db_path,
            cap_min=cap_min,
            cap_max=cap_max,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "sector": sector,
            "market_cap_focus": market_cap_focus,
            "selected_tickers": [],
            "source": "sector_scan_db",
            "loaded_tickers": [],
            "excluded_tickers": [],
            "warnings": warnings + [f"SECTOR_TICKER_SOURCE_FAILED:{exc}"],
            "ranking_basis": "none",
        }
    loaded_tickers = _normalize_tickers([ticker for ticker, _, _ in loaded_rows])
    selected = loaded_tickers[:limit] if limit is not None else loaded_tickers
    excluded = [ticker for ticker in loaded_tickers if ticker not in set(selected)]
    if not selected:
        warnings.append("NO_SECTOR_TICKERS_FOUND")
    return {
        "sector": sector,
        "market_cap_focus": market_cap_focus,
        "selection_mode": SECTOR_SELECTION_CACHE_ORDER,
        "selected_tickers": selected,
        "source": "sector_scan_db",
        "loaded_tickers": loaded_tickers,
        "excluded_tickers": excluded,
        "warnings": warnings,
        "ranking_basis": "sector_scan_db_order_cache_only",
    }


def _autonomous_candidate_refresh_tickers(
    *,
    sector: str,
    market_cap_focus: str,
    limit: int | None,
    db_path: str | Path | None,
) -> dict[str, Any]:
    """Use the benchmark candidate resolver so cache refresh matches likely autonomous runs."""

    from app.autonomous.sector_candidates import resolve_sector_candidate_tickers

    selection = resolve_sector_candidate_tickers(
        sector=sector,
        explicit_tickers=[],
        market_cap_focus=market_cap_focus,
        max_candidates=limit,
        db_path=db_path,
        filing_risk_use_llm=False,
    )
    payload = selection.to_dict()
    payload["selection_mode"] = SECTOR_SELECTION_AUTONOMOUS_CANDIDATES
    return payload


def _all_known_tickers(*, limit: int | None = None) -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT ticker FROM companies
            UNION
            SELECT ticker FROM universe_members
            ORDER BY ticker
            """
        ).fetchall()
    tickers = _normalize_tickers([row["ticker"] for row in rows])
    return tickers[:limit] if limit is not None and limit > 0 else tickers


def resolve_financial_cache_refresh_scope(
    *,
    tickers: list[str] | tuple[str, ...] | str | None = None,
    sectors: list[str] | tuple[str, ...] | str | None = None,
    all_known: bool = False,
    market_cap_focus: str = "smid_cap",
    max_tickers: int | None = None,
    max_tickers_per_sector: int | None = None,
    sector_selection_mode: str = SECTOR_SELECTION_CACHE_ORDER,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the ticker scope for a cache refresh without fetching live data."""

    explicit_tickers = _normalize_tickers(tickers)
    sector_list = _normalize_sectors(sectors)
    global_limit = int(max_tickers) if max_tickers is not None and int(max_tickers) > 0 else None
    per_sector_limit = (
        int(max_tickers_per_sector)
        if max_tickers_per_sector is not None and int(max_tickers_per_sector) > 0
        else None
    )
    resolved: list[str] = []
    seen: set[str] = set()
    ticker_sources: dict[str, list[str]] = {}
    warnings: list[str] = []
    sector_selections: list[dict[str, Any]] = []
    selection_mode = str(sector_selection_mode or SECTOR_SELECTION_CACHE_ORDER).strip().lower()
    if selection_mode not in SECTOR_SELECTION_MODES:
        warnings.append(f"UNKNOWN_SECTOR_SELECTION_MODE:{sector_selection_mode}")
        selection_mode = SECTOR_SELECTION_CACHE_ORDER

    def add_ticker(ticker: str, source: str) -> None:
        ticker_norm = str(ticker or "").strip().upper()
        if not ticker_norm:
            return
        if ticker_norm not in seen:
            if global_limit is not None and len(resolved) >= global_limit:
                return
            resolved.append(ticker_norm)
            seen.add(ticker_norm)
        _ticker_sources_add(ticker_sources, ticker_norm, source)

    for ticker in explicit_tickers:
        add_ticker(ticker, "explicit_tickers")

    for sector in sector_list:
        if selection_mode == SECTOR_SELECTION_AUTONOMOUS_CANDIDATES:
            try:
                selection_payload = _autonomous_candidate_refresh_tickers(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    limit=per_sector_limit,
                    db_path=db_path,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"AUTONOMOUS_CANDIDATE_SELECTION_FAILED:{sector}:{exc}")
                selection_payload = _sector_refresh_tickers(
                    sector=sector,
                    market_cap_focus=market_cap_focus,
                    limit=per_sector_limit,
                    db_path=db_path,
                )
                selection_payload.setdefault("warnings", []).append(
                    f"AUTONOMOUS_CANDIDATE_SELECTION_FAILED:{sector}:{exc}"
                )
        else:
            selection_payload = _sector_refresh_tickers(
                sector=sector,
                market_cap_focus=market_cap_focus,
                limit=per_sector_limit,
                db_path=db_path,
            )
        sector_selections.append(selection_payload)
        warnings.extend(str(item) for item in selection_payload.get("warnings", []) if str(item).strip())
        for ticker in selection_payload.get("selected_tickers", []):
            add_ticker(ticker, f"sector:{sector}")

    if all_known:
        for ticker in _all_known_tickers(limit=None):
            add_ticker(ticker, "all_known")

    if not resolved:
        warnings.append("NO_REFRESH_TICKERS_RESOLVED")

    return {
        "tickers": resolved,
        "ticker_sources": {ticker: ticker_sources.get(ticker, []) for ticker in resolved},
        "explicit_tickers": explicit_tickers,
        "sectors": sector_list,
        "all_known": bool(all_known),
        "market_cap_focus": market_cap_focus,
        "max_tickers": global_limit,
        "max_tickers_per_sector": per_sector_limit,
        "sector_selection_mode": selection_mode,
        "sector_selections": sector_selections,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _resolve_cik_for_ticker(ticker: str) -> tuple[str | None, str]:
    upper = ticker.upper().strip()
    with get_db() as conn:
        row = conn.execute("SELECT cik FROM companies WHERE ticker = ? LIMIT 1", (upper,)).fetchone()
        if row and row["cik"]:
            return str(row["cik"]).zfill(10), "companies"
        row = conn.execute("SELECT cik FROM universe_members WHERE ticker = ? LIMIT 1", (upper,)).fetchone()
        if row and row["cik"]:
            return str(row["cik"]).zfill(10), "universe_members"
    try:
        return str(resolve_cik(upper)).zfill(10), "ticker_cik_map"
    except Exception:
        return None, "unresolved"


def _companyfacts_diagnostics(ticker: str, *, ttl_seconds: int = 24 * 3600) -> dict[str, Any]:
    upper = ticker.upper().strip()
    with get_db() as conn:
        annual_rows = conn.execute(
            """
            SELECT COUNT(*) AS n, COUNT(DISTINCT fiscal_year) AS years, MAX(period_end) AS latest_period_end
            FROM companyfacts_facts
            WHERE ticker = ? AND period_type IN ('FY')
            """,
            (upper,),
        ).fetchone()
        quarterly_rows = conn.execute(
            """
            SELECT COUNT(*) AS n, COUNT(DISTINCT fiscal_year || ':' || period_type) AS periods,
                   MAX(period_end) AS latest_period_end
            FROM companyfacts_facts
            WHERE ticker = ? AND period_type NOT IN ('FY')
            """,
            (upper,),
        ).fetchone()
        latest_fetched = conn.execute(
            """
            SELECT MAX(fetched_at) AS latest_fetched_at
            FROM companyfacts_facts
            WHERE ticker = ?
            """,
            (upper,),
        ).fetchone()
        annual_fresh = companyfacts_is_fresh(
            conn,
            upper,
            ttl_seconds=ttl_seconds,
            period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        )
        quarterly_fresh = companyfacts_is_fresh(
            conn,
            upper,
            ttl_seconds=ttl_seconds,
            exclude_period_types=ANNUAL_COMPANYFACTS_PERIOD_TYPES,
        )
    annual_count = int(annual_rows["n"] or 0)
    quarterly_count = int(quarterly_rows["n"] or 0)
    return {
        "annual_rows": annual_count,
        "annual_years": int(annual_rows["years"] or 0),
        "annual_latest_period_end": annual_rows["latest_period_end"],
        "quarterly_rows": quarterly_count,
        "quarterly_periods": int(quarterly_rows["periods"] or 0),
        "quarterly_latest_period_end": quarterly_rows["latest_period_end"],
        "latest_fetched_at": latest_fetched["latest_fetched_at"],
        "annual_fresh": bool(annual_fresh),
        "quarterly_fresh": bool(quarterly_fresh),
        "has_any_facts": bool(annual_count or quarterly_count),
    }


def _filing_diagnostics(ticker: str, *, as_of_date: str, run_id: str) -> dict[str, Any]:
    upper = ticker.upper().strip()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT form_type, accession, filing_date, status, local_path
            FROM filings
            WHERE ticker = ? AND filing_date <= ?
            ORDER BY COALESCE(filing_date, '1900-01-01') DESC, id DESC
            """,
            (upper, as_of_date),
        ).fetchall()
        coverage = conn.execute(
            """
            SELECT forms_included_json, accession_numbers_json, coverage_score, missing_required_json
            FROM filing_coverage
            WHERE ticker = ? AND run_id = ? AND as_of_date = ?
            LIMIT 1
            """,
            (upper, run_id, as_of_date),
        ).fetchone()
    forms = sorted({str(row["form_type"] or "").upper() for row in rows if row["form_type"]})
    readable = [
        row
        for row in rows
        if str(row["local_path"] or "").strip() and Path(str(row["local_path"])).exists()
    ]
    coverage_payload: dict[str, Any] = {}
    if coverage:
        for key in ("forms_included_json", "accession_numbers_json", "missing_required_json"):
            raw = coverage[key]
            try:
                coverage_payload[key.replace("_json", "")] = json.loads(raw or "[]")
            except Exception:
                coverage_payload[key.replace("_json", "")] = []
        coverage_payload["coverage_score"] = coverage["coverage_score"]
    return {
        "filing_rows": len(rows),
        "forms_cached": forms,
        "readable_file_count": len(readable),
        "latest_filing_date": str(rows[0]["filing_date"]) if rows else None,
        "latest_form_type": str(rows[0]["form_type"]) if rows else None,
        "coverage": coverage_payload,
    }


def _step_is_complete(previous: dict[str, Any], step_name: str) -> bool:
    step = previous.get("steps", {}).get(step_name, {}) if isinstance(previous.get("steps"), dict) else {}
    status = str(step.get("status") or "").upper()
    return status in _COMPLETE_STEP_STATUSES and status not in _INCOMPLETE_STEP_STATUSES


def _blank_ticker_result(ticker: str, *, sources: list[str], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    prior = dict(previous or {})
    steps = prior.get("steps") if isinstance(prior.get("steps"), dict) else {}
    return {
        "ticker": ticker,
        "sources": sources,
        "cik": prior.get("cik"),
        "cik_source": prior.get("cik_source"),
        "status": "PENDING",
        "steps": dict(steps),
        "warnings": list(prior.get("warnings") or []),
        "errors": list(prior.get("errors") or []),
    }


def _set_step(result: dict[str, Any], step_name: str, payload: dict[str, Any]) -> None:
    result.setdefault("steps", {})[step_name] = payload


def _refresh_facts_step(result: dict[str, Any], *, ticker: str, years: int) -> None:
    try:
        ensure_all_facts(ticker, years_back=years)
        diagnostics = _companyfacts_diagnostics(ticker)
        status = "OK" if diagnostics["has_any_facts"] else "NO_DATA"
        _set_step(
            result,
            "facts",
            {
                "status": status,
                "years_back": int(years),
                **diagnostics,
            },
        )
    except Exception as exc:  # noqa: BLE001
        result.setdefault("errors", []).append(f"facts:{type(exc).__name__}:{exc}")
        _set_step(result, "facts", {"status": "FAILED", "years_back": int(years), "error": str(exc)})


def _refresh_filings_step(result: dict[str, Any], *, ticker: str, run_id: str, as_of_date: str) -> None:
    try:
        ingest_summary = ingest_with_policy(as_of_date=as_of_date, run_id=run_id, tickers=[ticker])
        context = load_research_filing_context(
            ticker,
            as_of_date=as_of_date,
            quarters=1,
            include_material_events=True,
        )
        diagnostics = _filing_diagnostics(ticker, as_of_date=as_of_date, run_id=run_id)
        if context.documents:
            status = "OK"
        elif diagnostics["filing_rows"]:
            status = "NO_READABLE_FILINGS"
        else:
            status = "NO_FILINGS"
        _set_step(
            result,
            "filings",
            {
                "status": status,
                "ingest_summary": ingest_summary,
                "readable_document_count": len(context.documents),
                "recent_filing_status": context.recent_filing_status,
                "recovered_document_count": context.recovered_document_count,
                "warnings": list(context.warnings),
                **diagnostics,
            },
        )
    except Exception as exc:  # noqa: BLE001
        result.setdefault("errors", []).append(f"filings:{type(exc).__name__}:{exc}")
        _set_step(result, "filings", {"status": "FAILED", "error": str(exc)})


def _mark_disabled(result: dict[str, Any], step_name: str, reason: str) -> None:
    _set_step(result, step_name, {"status": "DISABLED", "reason": reason})


def _mark_skipped(result: dict[str, Any], step_name: str, previous: dict[str, Any]) -> None:
    prior_step = previous.get("steps", {}).get(step_name, {}) if isinstance(previous.get("steps"), dict) else {}
    payload = dict(prior_step)
    payload["status"] = "SKIPPED"
    payload["reason"] = "completed_in_previous_manifest"
    _set_step(result, step_name, payload)


def _apply_price_summary(
    *,
    results_by_ticker: dict[str, dict[str, Any]],
    price_summary: dict[str, Any],
    price_tickers: list[str],
) -> None:
    rows_by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in (price_summary.get("rows") or [])
        if isinstance(row, dict)
    }
    for ticker in price_tickers:
        row = rows_by_ticker.get(ticker.upper())
        result = results_by_ticker[ticker]
        if row is None:
            _set_step(result, "price", {"status": "FAILED", "error": "price_summary_missing_ticker"})
            result.setdefault("errors", []).append("price:price_summary_missing_ticker")
            continue
        _set_step(
            result,
            "price",
            {
                "status": str(row.get("status") or "MISSING").upper(),
                "as_of_used": row.get("as_of_used"),
                "price": row.get("price"),
                "source": row.get("source"),
                "reason_code": row.get("reason_code"),
                "path": row.get("path"),
                "summary_path": price_summary.get("summary_path"),
            },
        )


def _finalize_ticker_status(result: dict[str, Any]) -> None:
    steps = result.get("steps") if isinstance(result.get("steps"), dict) else {}
    statuses = [str(step.get("status") or "").upper() for step in steps.values() if isinstance(step, dict)]
    if any(status == "FAILED" for status in statuses):
        result["status"] = "PARTIAL"
    elif statuses and all(status in _COMPLETE_STEP_STATUSES for status in statuses):
        result["status"] = "OK"
    elif statuses:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "NO_STEPS"


def _build_summary(
    *,
    run_id: str,
    as_of_date: str,
    created_at: str,
    completed_at: str,
    scope: dict[str, Any],
    ticker_results: list[dict[str, Any]],
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    ticker_status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in ticker_results)
    step_status_counts: dict[str, dict[str, int]] = {}
    for row in ticker_results:
        steps = row.get("steps") if isinstance(row.get("steps"), dict) else {}
        for step_name, step in steps.items():
            status = str((step or {}).get("status") or "UNKNOWN").upper() if isinstance(step, dict) else "UNKNOWN"
            step_status_counts.setdefault(step_name, {})
            step_status_counts[step_name][status] = step_status_counts[step_name].get(status, 0) + 1
    errors = [error for row in ticker_results for error in (row.get("errors") or [])]
    status = "COMPLETED"
    if not ticker_results:
        status = "NO_TARGETS"
    elif errors:
        status = "PARTIAL"
    summary = {
        "run_id": run_id,
        "status": status,
        "as_of_date": as_of_date,
        "created_at": created_at,
        "completed_at": completed_at,
        "ticker_count": len(ticker_results),
        "ticker_status_counts": dict(sorted(ticker_status_counts.items())),
        "step_status_counts": {key: dict(sorted(value.items())) for key, value in sorted(step_status_counts.items())},
        "error_count": len(errors),
        "warnings": list(scope.get("warnings") or []),
        "scope": scope,
        "ticker_results": ticker_results,
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }
    sector_selections = scope.get("sector_selections") if isinstance(scope.get("sector_selections"), list) else []
    if sector_selections:
        summary["cache_readiness"] = cache_readiness_by_sector(
            summary,
            max_candidates=scope.get("max_tickers_per_sector"),
        )
    return summary


def render_financial_cache_refresh_report(summary: dict[str, Any]) -> str:
    scope = summary.get("scope") if isinstance(summary.get("scope"), dict) else {}
    lines: list[str] = [
        "# Financial Cache Refresh Report",
        "",
        f"**Run ID:** {summary.get('run_id')}",
        f"**Status:** {summary.get('status')}",
        f"**As of:** {summary.get('as_of_date')}",
        f"**Tickers:** {summary.get('ticker_count')}",
        f"**Sector selection mode:** {scope.get('sector_selection_mode') or 'cache_order'}",
        "",
        "## Step Status Counts",
        "",
    ]
    step_counts = summary.get("step_status_counts") if isinstance(summary.get("step_status_counts"), dict) else {}
    if step_counts:
        lines.append("| Step | Status Counts |")
        lines.append("|------|---------------|")
        for step_name, counts in step_counts.items():
            rendered = ", ".join(f"{key}: {value}" for key, value in (counts or {}).items())
            lines.append(f"| {step_name} | {rendered or '—'} |")
    else:
        lines.append("No refresh steps were run.")

    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend(["", "## Scope Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    sector_selections = scope.get("sector_selections") if isinstance(scope.get("sector_selections"), list) else []
    if sector_selections:
        lines.extend(
            [
                "",
                "## Sector Selection",
                "",
                "| Sector | Mode | Ranking Basis | Selected Tickers | Warnings |",
                "|--------|------|---------------|------------------|----------|",
            ]
        )
        for selection in sector_selections:
            if not isinstance(selection, dict):
                continue
            lines.append(
                "| {sector} | {mode} | {basis} | {tickers} | {warnings} |".format(
                    sector=str(selection.get("sector") or "").replace("|", "\\|"),
                    mode=str(selection.get("selection_mode") or scope.get("sector_selection_mode") or "").replace("|", "\\|"),
                    basis=str(selection.get("ranking_basis") or "").replace("|", "\\|"),
                    tickers=", ".join(str(ticker) for ticker in selection.get("selected_tickers") or []).replace("|", "\\|"),
                    warnings=", ".join(str(item) for item in selection.get("warnings") or []).replace("|", "\\|"),
                )
            )

    cache_readiness = summary.get("cache_readiness") if isinstance(summary.get("cache_readiness"), dict) else {}
    if cache_readiness:
        rollups = cache_readiness.get("rollups") if isinstance(cache_readiness.get("rollups"), dict) else {}
        lines.extend(
            [
                "",
                "## Cache Readiness",
                "",
                f"**Overall status:** `{cache_readiness.get('overall_status')}`",
                f"**Coverage status counts:** `{rollups.get('coverage_status_counts') or {}}`",
                f"**Candidate readiness counts:** `{rollups.get('candidate_readiness_counts') or {}}`",
                f"**Recommended max tickers per sector:** `{rollups.get('recommended_cache_max_tickers_per_sector') or ''}`",
                "",
                "| Sector | Coverage | Ready | Partial | Not Usable | Execution Pool | Excluded | Warnings |",
                "|--------|----------|-------|---------|------------|----------------|----------|----------|",
            ]
        )
        sectors = cache_readiness.get("sectors") if isinstance(cache_readiness.get("sectors"), dict) else {}
        for sector, row in sectors.items():
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {sector} | {coverage} | {ready} | {partial} | {not_usable} | {pool} | {excluded} | {warnings} |".format(
                    sector=str(sector).replace("|", "\\|"),
                    coverage=str(row.get("coverage_status") or "-").replace("|", "\\|"),
                    ready=", ".join(str(ticker) for ticker in row.get("ready_tickers") or []).replace("|", "\\|") or "-",
                    partial=", ".join(str(ticker) for ticker in row.get("partial_tickers") or []).replace("|", "\\|") or "-",
                    not_usable=", ".join(str(ticker) for ticker in row.get("not_usable_tickers") or []).replace("|", "\\|") or "-",
                    pool=", ".join(str(ticker) for ticker in row.get("final_candidate_pool") or []).replace("|", "\\|") or "-",
                    excluded=", ".join(str(ticker) for ticker in row.get("excluded_tickers") or []).replace("|", "\\|") or "-",
                    warnings=", ".join(str(item) for item in (row.get("cache_limited_reasons") or row.get("cache_readiness_warnings") or [])).replace("|", "\\|") or "-",
                )
            )

    lines.extend(
        [
            "",
            "## Ticker Coverage",
            "",
            "| Ticker | CIK | Facts | Annual Rows | Quarterly Rows | Filings | Readable Docs | Price | Errors |",
            "|--------|-----|-------|-------------|----------------|---------|---------------|-------|--------|",
        ]
    )
    for row in summary.get("ticker_results") or []:
        steps = row.get("steps") if isinstance(row.get("steps"), dict) else {}
        facts = steps.get("facts") if isinstance(steps.get("facts"), dict) else {}
        filings = steps.get("filings") if isinstance(steps.get("filings"), dict) else {}
        price = steps.get("price") if isinstance(steps.get("price"), dict) else {}
        errors = "; ".join(str(item) for item in row.get("errors") or [])
        lines.append(
            "| {ticker} | {cik} | {facts_status} | {annual} | {quarterly} | {filing_status} | {readable} | {price_status} | {errors} |".format(
                ticker=row.get("ticker") or "",
                cik=row.get("cik") or "—",
                facts_status=facts.get("status") or "—",
                annual=facts.get("annual_rows") if facts.get("annual_rows") is not None else "—",
                quarterly=facts.get("quarterly_rows") if facts.get("quarterly_rows") is not None else "—",
                filing_status=filings.get("status") or "—",
                readable=filings.get("readable_document_count") if filings.get("readable_document_count") is not None else "—",
                price_status=price.get("status") or "—",
                errors=errors or "—",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _manifest_payload(
    *,
    run_id: str,
    created_at: str,
    as_of_date: str,
    scope: dict[str, Any],
    ticker_results: list[dict[str, Any]],
    options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": utc_now().isoformat(),
        "as_of_date": as_of_date,
        "scope": scope,
        "options": options,
        "ticker_results": ticker_results,
    }


def run_financial_cache_refresh(
    *,
    tickers: list[str] | tuple[str, ...] | str | None = None,
    sectors: list[str] | tuple[str, ...] | str | None = None,
    all_known: bool = False,
    market_cap_focus: str = "smid_cap",
    max_tickers: int | None = None,
    max_tickers_per_sector: int | None = None,
    sector_selection_mode: str = SECTOR_SELECTION_CACHE_ORDER,
    years: int = 10,
    as_of_date: str | None = None,
    weekly: bool = False,
    force: bool = False,
    resume_run_id: str | None = None,
    with_prices: bool = True,
    with_filings: bool = True,
    fallback_days: int | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Prewarm local financial caches for sector and benchmark work."""

    cfg = cfg or get_config()
    ensure_directories(cfg)
    init_db(cfg)
    anchor_date = parse_yyyy_mm_dd(as_of_date).isoformat() if as_of_date else date.today().isoformat()
    created_at = utc_now().isoformat()
    run_id = str(resume_run_id or "").strip() or _generated_run_id(created_at)
    out_dir = _run_dir(cfg, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "refresh_manifest.json"
    summary_path = out_dir / "refresh_summary.json"
    report_path = out_dir / "refresh_report.md"

    previous_manifest = _read_json(manifest_path) if resume_run_id else {}
    previous_by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in (previous_manifest.get("ticker_results") or [])
        if isinstance(row, dict)
    }

    scope = resolve_financial_cache_refresh_scope(
        tickers=tickers,
        sectors=sectors,
        all_known=all_known,
        market_cap_focus=market_cap_focus,
        max_tickers=max_tickers,
        max_tickers_per_sector=max_tickers_per_sector,
        sector_selection_mode=sector_selection_mode,
        db_path=cfg.db_path,
    )
    options = {
        "years": int(years),
        "weekly": bool(weekly),
        "force": bool(force),
        "max_tickers_per_sector": scope.get("max_tickers_per_sector"),
        "sector_selection_mode": scope.get("sector_selection_mode"),
        "with_prices": bool(with_prices),
        "with_filings": bool(with_filings),
        "fallback_days": fallback_days,
        "resume_run_id": resume_run_id,
    }

    results_by_ticker: dict[str, dict[str, Any]] = {}
    price_tickers: list[str] = []

    for ticker in scope["tickers"]:
        previous = previous_by_ticker.get(ticker, {})
        result = _blank_ticker_result(ticker, sources=scope.get("ticker_sources", {}).get(ticker, []), previous=previous)
        cik, cik_source = _resolve_cik_for_ticker(ticker)
        result["cik"] = cik
        result["cik_source"] = cik_source

        if not force and _step_is_complete(previous, "facts"):
            _mark_skipped(result, "facts", previous)
        else:
            _refresh_facts_step(result, ticker=ticker, years=max(1, int(years)))

        if not with_filings:
            _mark_disabled(result, "filings", "with_filings_false")
        elif not force and _step_is_complete(previous, "filings"):
            _mark_skipped(result, "filings", previous)
        else:
            _refresh_filings_step(result, ticker=ticker, run_id=run_id, as_of_date=anchor_date)

        if not with_prices:
            _mark_disabled(result, "price", "with_prices_false")
        elif not force and _step_is_complete(previous, "price"):
            _mark_skipped(result, "price", previous)
        else:
            price_tickers.append(ticker)

        results_by_ticker[ticker] = result
        _finalize_ticker_status(result)
        _write_json(
            manifest_path,
            _manifest_payload(
                run_id=run_id,
                created_at=created_at,
                as_of_date=anchor_date,
                scope=scope,
                ticker_results=list(results_by_ticker.values()),
                options=options,
            ),
        )

    if price_tickers:
        try:
            price_summary = write_prices_for_run(
                tickers=price_tickers,
                as_of_date=anchor_date,
                run_id=run_id,
                fallback_days=fallback_days if fallback_days is not None else cfg.price_fallback_days,
                cfg=cfg,
            )
            _apply_price_summary(
                results_by_ticker=results_by_ticker,
                price_summary=price_summary,
                price_tickers=price_tickers,
            )
        except Exception as exc:  # noqa: BLE001
            for ticker in price_tickers:
                result = results_by_ticker[ticker]
                result.setdefault("errors", []).append(f"price:{type(exc).__name__}:{exc}")
                _set_step(result, "price", {"status": "FAILED", "error": str(exc)})

    ticker_results = [results_by_ticker[ticker] for ticker in scope["tickers"]]
    for result in ticker_results:
        _finalize_ticker_status(result)

    completed_at = utc_now().isoformat()
    manifest_payload = _manifest_payload(
        run_id=run_id,
        created_at=created_at,
        as_of_date=anchor_date,
        scope=scope,
        ticker_results=ticker_results,
        options=options,
    )
    _write_json(manifest_path, manifest_payload)
    summary = _build_summary(
        run_id=run_id,
        as_of_date=anchor_date,
        created_at=created_at,
        completed_at=completed_at,
        scope=scope,
        ticker_results=ticker_results,
        manifest_path=manifest_path,
        report_path=report_path,
    )
    summary["summary_path"] = str(summary_path)
    _write_json(summary_path, summary)
    report_path.write_text(render_financial_cache_refresh_report(summary), encoding="utf-8")
    return summary


def run_financial_cache_refresh_for_benchmark(
    *,
    sectors: list[str] | tuple[str, ...] | str,
    market_cap_focus: str = "smid_cap",
    max_tickers: int | None = None,
    max_tickers_per_sector: int | None = None,
    sector_selection_mode: str = SECTOR_SELECTION_AUTONOMOUS_CANDIDATES,
    years: int = 10,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Small helper for benchmark prep scripts to prewarm benchmark sector data."""

    return run_financial_cache_refresh(
        sectors=sectors,
        market_cap_focus=market_cap_focus,
        max_tickers=max_tickers,
        max_tickers_per_sector=max_tickers_per_sector,
        sector_selection_mode=sector_selection_mode,
        years=years,
        as_of_date=as_of_date,
    )


__all__ = [
    "render_financial_cache_refresh_report",
    "resolve_financial_cache_refresh_scope",
    "run_financial_cache_refresh",
    "run_financial_cache_refresh_for_benchmark",
    "SECTOR_SELECTION_AUTONOMOUS_CANDIDATES",
    "SECTOR_SELECTION_CACHE_ORDER",
]
