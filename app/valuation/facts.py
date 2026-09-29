from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable

from app.config import AppConfig, get_config
from app.db import get_db, utc_now_iso
from app.market.company_facts_extract import extract_company_facts_asof
from app.market.company_facts_provider import FACTS_REASON_CODES, fetch_company_facts, normalize_cik
from app.universe.facts_blockers import enrich_facts_blocker_fields, summarize_facts_blockers
from app.universe.ticker_cik_map import load_ticker_cik_map
from app.valuation.fundamentals import UNKNOWN


_UNIVERSE_CIK_CACHE: dict[str, dict[str, str]] = {}
_FACTS_ROW_CACHE: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
_USD_TO_MUSD = 1_000_000.0
_SHARES_TO_MSHARES = 1_000_000.0

FactsProgressHook = Callable[[str, dict[str, Any]], None]


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _load_universe_cik_map(cfg: AppConfig) -> dict[str, str]:
    cache_key = str(Path(cfg.universe_path).expanduser().resolve())
    if cache_key in _UNIVERSE_CIK_CACHE:
        return dict(_UNIVERSE_CIK_CACHE[cache_key])
    mapping: dict[str, str] = {}
    path = cfg.universe_path
    if path.exists():
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    ticker = str(row.get("ticker") or "").strip().upper()
                    cik = normalize_cik(row.get("cik"))
                    if ticker and cik:
                        mapping[ticker] = cik
        except Exception:
            mapping = {}
    _UNIVERSE_CIK_CACHE[cache_key] = dict(mapping)
    return mapping


def _copy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _dedupe_refs(refs: list[str]) -> list[str]:
    out: list[str] = []
    for ref in refs:
        token = str(ref or "").strip()
        if token and token not in out:
            out.append(token)
    return out


def resolve_cik_for_ticker(
    ticker: str,
    *,
    cfg: AppConfig | None = None,
    refresh_if_missing: bool | None = None,
) -> str | None:
    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    if not ticker_norm:
        return None

    try:
        with get_db(cfg=cfg) as conn:
            row = conn.execute(
                """
                SELECT cik
                FROM companies
                WHERE ticker = ?
                LIMIT 1
                """,
                (ticker_norm,),
            ).fetchone()
        if row and row["cik"]:
            cik = normalize_cik(row["cik"])
            if cik:
                return cik
    except Exception:
        pass

    universe_map = _load_universe_cik_map(cfg)
    universe_cik = normalize_cik(universe_map.get(ticker_norm))
    if universe_cik:
        return universe_cik

    # VOE_NET_PROVIDER is the only network switch (not the LLM provider).
    offline = str(os.getenv("VOE_NET_PROVIDER", "")).strip().lower() == "disabled"
    if refresh_if_missing is None:
        refresh_if_missing = not offline
    try:
        ticker_map = load_ticker_cik_map(refresh_if_missing=refresh_if_missing)
    except Exception:
        ticker_map = {}
    mapped = normalize_cik(ticker_map.get(ticker_norm))
    return mapped or None


def clear_facts_row_cache() -> None:
    _FACTS_ROW_CACHE.clear()
    _UNIVERSE_CIK_CACHE.clear()


def _field_status(value: Any, *, missing_reason: str) -> tuple[str, str]:
    if _is_num(value):
        return "OK", "OK"
    return "UNKNOWN", missing_reason


def _to_musd(value: Any, unit: Any) -> float | str:
    if not _is_num(value) or str(unit or "").strip() != "USD":
        return UNKNOWN
    return float(value) / _USD_TO_MUSD


def _to_mshares(value: Any, unit: Any) -> float | str:
    if not _is_num(value) or str(unit or "").strip() != "shares":
        return UNKNOWN
    return float(value) / _SHARES_TO_MSHARES


def _overall_status(
    *, shares_status: str, cfo_status: str, capex_status: str, fcf_status: str
) -> str:
    statuses = [shares_status, cfo_status, capex_status, fcf_status]
    ok_count = len([status for status in statuses if status == "OK"])
    if ok_count == len(statuses):
        return "OK"
    if ok_count > 0:
        return "PARTIAL"
    return "UNKNOWN"


def _source_resolution_from_fetch(fetch_result: dict[str, Any]) -> str:
    source_resolution = str(fetch_result.get("source_resolution") or "").strip()
    if source_resolution in {"companyfacts_cache", "companyfacts_fetch"}:
        return source_resolution
    reason_code = str(fetch_result.get("reason_code") or "").strip().upper()
    if reason_code == "CACHE_HIT":
        return "companyfacts_cache"
    if reason_code == "FETCH_OK":
        return "companyfacts_fetch"
    return "unknown"


def _emit_progress(
    progress_hook: FactsProgressHook | None, phase: str, payload: dict[str, Any]
) -> None:
    if progress_hook is None:
        return
    try:
        progress_hook(str(phase), dict(payload))
    except Exception:
        return


def resolve_financial_facts_asof(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str | None = None,
    refresh: bool = False,
    sec_budget: int | None = None,
    progress_hook: FactsProgressHook | None = None,
    cfg: AppConfig | None = None,
    cache_only: bool = False,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    ticker_norm = str(ticker or "").strip().upper()
    asof_norm = str(as_of_date or "").strip()
    # Facts are resolved from config-scoped DB, universe, and CompanyFacts
    # cache roots. Reusing a row after any of those roots changes can bind a
    # different issuer's or run's lineage into the current financial scope.
    cache_key = (
        str(Path(cfg.db_path).expanduser().resolve()),
        str(Path(cfg.universe_path).expanduser().resolve()),
        str(Path(cfg.cache_dir).expanduser().resolve()),
        ticker_norm,
        asof_norm,
        str(run_id or ""),
        "cache_only" if cache_only else "network_permitted",
    )
    if not refresh and cache_key in _FACTS_ROW_CACHE:
        return _copy_payload(_FACTS_ROW_CACHE[cache_key])

    row: dict[str, Any] = {
        "ticker": ticker_norm,
        "requested_as_of": asof_norm,
        "run_id": run_id,
        "cik": None,
        "status": "UNKNOWN",
        "shares_status": "UNKNOWN",
        "shares_reason": "COMPANYFACTS_MISS",
        "shares_value": UNKNOWN,
        "shares_raw_value": UNKNOWN,
        "shares_input_unit": None,
        "shares_output_unit": None,
        "shares_asof_used": None,
        "shares_filed_date": None,
        "cfo_status": "UNKNOWN",
        "cfo_reason": "COMPANYFACTS_MISS",
        "cfo_value": UNKNOWN,
        "cfo_asof_used": None,
        "capex_status": "UNKNOWN",
        "capex_reason": "COMPANYFACTS_MISS",
        "capex_value": UNKNOWN,
        "capex_asof_used": None,
        "fcf_status": "UNKNOWN",
        "fcf_reason": "COMPANYFACTS_MISS",
        "fcf_value": UNKNOWN,
        "fcf_asof_used": None,
        "source_resolution": "unknown",
        "fetch_reason_code": None,
        "fetch_reason_detail": None,
        "cache_path": None,
        "source_url": None,
        "http_status": None,
        "network_attempted": False,
        "fetch_attempts": 0,
        "fetch_retries_configured": 0,
        "fetch_backoff_seconds": None,
        "fetch_timeout_seconds": None,
        "fetch_last_exception_type": None,
        "fetch_last_exception_message": None,
        "derived_from": [],
        "generated_at": utc_now_iso(),
    }

    def _cache_and_return(current_row: dict[str, Any]) -> dict[str, Any]:
        enriched = enrich_facts_blocker_fields(current_row)
        _FACTS_ROW_CACHE[cache_key] = _copy_payload(enriched)
        return _copy_payload(enriched)

    if not ticker_norm or not asof_norm:
        row["shares_reason"] = "EXCEPTION"
        row["cfo_reason"] = "EXCEPTION"
        row["capex_reason"] = "EXCEPTION"
        row["fcf_reason"] = "EXCEPTION"
        return _cache_and_return(row)

    cik = resolve_cik_for_ticker(ticker_norm, cfg=cfg)
    if not cik:
        row["shares_reason"] = "CIK_MISSING"
        row["cfo_reason"] = "CIK_MISSING"
        row["capex_reason"] = "CIK_MISSING"
        row["fcf_reason"] = "CIK_MISSING"
        row["fetch_reason_code"] = "CIK_MISSING"
        row["fetch_reason_detail"] = "CIK mapping not available for ticker."
        return _cache_and_return(row)

    row["cik"] = cik
    _emit_progress(
        progress_hook,
        "COMPANYFACTS_ACQUISITION",
        {
            "ticker": ticker_norm,
            "run_id": run_id,
            "cik": cik,
            "requested_as_of": asof_norm,
            "cache_key": {
                "ticker": ticker_norm,
                "as_of_date": asof_norm,
                "run_id": str(run_id or ""),
            },
        },
    )
    fetch_result = fetch_company_facts(
        cik,
        user_agent=cfg.sec_user_agent,
        sec_budget=sec_budget,
        cfg=cfg,
        cache_only=cache_only,
    )
    row["source_resolution"] = _source_resolution_from_fetch(fetch_result)
    row["fetch_reason_code"] = str(fetch_result.get("reason_code") or "")
    row["fetch_reason_detail"] = str(fetch_result.get("reason_detail") or "")
    row["cache_path"] = fetch_result.get("cache_path")
    row["source_url"] = fetch_result.get("source_url")
    row["http_status"] = fetch_result.get("http_status")
    row["network_attempted"] = bool(fetch_result.get("network_attempted"))
    row["fetch_attempts"] = int(fetch_result.get("attempts_made") or 0)
    row["fetch_retries_configured"] = int(fetch_result.get("retries_configured") or 0)
    row["fetch_backoff_seconds"] = fetch_result.get("backoff_seconds")
    row["fetch_timeout_seconds"] = fetch_result.get("timeout_seconds")
    row["fetch_last_exception_type"] = fetch_result.get("last_exception_type")
    row["fetch_last_exception_message"] = fetch_result.get("last_exception_message")
    row["derived_from"] = _dedupe_refs(
        [str(fetch_result.get("cache_path") or "")]
        + [str(ref) for ref in (fetch_result.get("derived_from") or []) if str(ref).strip()]
    )

    if str(fetch_result.get("status") or "").upper() != "OK":
        miss_code = str(fetch_result.get("reason_code") or "COMPANYFACTS_MISS").upper()
        if miss_code not in FACTS_REASON_CODES and miss_code != "CIK_MISSING":
            miss_code = "COMPANYFACTS_MISS"
        row["shares_reason"] = miss_code
        row["cfo_reason"] = miss_code
        row["capex_reason"] = miss_code
        row["fcf_reason"] = miss_code
        return _cache_and_return(row)

    companyfacts = fetch_result.get("companyfacts")
    if not isinstance(companyfacts, dict):
        row["shares_reason"] = "COMPANYFACTS_MISS"
        row["cfo_reason"] = "COMPANYFACTS_MISS"
        row["capex_reason"] = "COMPANYFACTS_MISS"
        row["fcf_reason"] = "COMPANYFACTS_MISS"
        return _cache_and_return(row)

    _emit_progress(
        progress_hook,
        "FACTS_NORMALIZATION",
        {
            "ticker": ticker_norm,
            "run_id": run_id,
            "requested_as_of": asof_norm,
            "fetch_reason_code": row["fetch_reason_code"],
            "network_attempted": row["network_attempted"],
            "fetch_attempts": row["fetch_attempts"],
        },
    )
    extracted = extract_company_facts_asof(companyfacts, asof_norm)
    shares = extracted.get("shares_outstanding_asof") if isinstance(extracted, dict) else None
    shares_guard = (
        extracted.get("shares_outstanding_guard") if isinstance(extracted, dict) else None
    )
    cfo = extracted.get("cfo_asof") if isinstance(extracted, dict) else None
    capex = extracted.get("capex_asof") if isinstance(extracted, dict) else None
    fcf = extracted.get("fcf_asof") if isinstance(extracted, dict) else None

    if isinstance(shares, dict) and _is_num(shares.get("value")):
        normalized_shares = _to_mshares(shares["value"], shares.get("unit"))
        row["shares_status"] = "OK" if _is_num(normalized_shares) else "UNKNOWN"
        row["shares_reason"] = "OK" if _is_num(normalized_shares) else "UNIT_AMBIGUOUS"
        row["shares_value"] = normalized_shares
        row["shares_asof_used"] = shares.get("fact_end_date")
        row["shares_filed_date"] = shares.get("filed_date")
        row["shares_input_unit"] = shares.get("unit")
        row["shares_output_unit"] = "shares_millions"
        row["shares_raw_value"] = shares.get("value")
        # A CompanyFacts share observation proves a dated raw share count, not
        # that no split occurred between that observation and a later quote.
        # Basis/factor are established only when quote-time split evidence is
        # reconciled in the market-cap resolver.
        row["shares_basis"] = None
        row["shares_split_adjustment_factor"] = None
        row["derived_from"] = _dedupe_refs(
            row["derived_from"]
            + [str(ref) for ref in (shares.get("derived_from") or []) if str(ref).strip()]
        )
    else:
        row["shares_status"], row["shares_reason"] = _field_status(None, missing_reason="TAG_MISS")
        # A count the share guard refused is not a missing tag: say which rule refused it.
        if isinstance(shares_guard, dict) and shares_guard.get("reason_code"):
            row["shares_reason"] = str(shares_guard["reason_code"])
    if isinstance(shares_guard, dict):
        row["shares_guard"] = shares_guard

    if isinstance(cfo, dict) and _is_num(cfo.get("value")):
        normalized_cfo = _to_musd(cfo["value"], cfo.get("unit"))
        row["cfo_status"] = "OK" if _is_num(normalized_cfo) else "UNKNOWN"
        row["cfo_reason"] = "OK" if _is_num(normalized_cfo) else "UNIT_AMBIGUOUS"
        row["cfo_value"] = normalized_cfo
        row["cfo_asof_used"] = cfo.get("fact_end_date")
        row["cfo_filed_date"] = cfo.get("filed_date")
        row["cfo_input_unit"] = cfo.get("unit")
        row["cfo_output_unit"] = "USD_millions"
        row["cfo_raw_value"] = cfo.get("value")
        row["derived_from"] = _dedupe_refs(
            row["derived_from"]
            + [str(ref) for ref in (cfo.get("derived_from") or []) if str(ref).strip()]
        )
    else:
        row["cfo_status"], row["cfo_reason"] = _field_status(None, missing_reason="TAG_MISS")

    if isinstance(capex, dict) and _is_num(capex.get("value")):
        normalized_capex = _to_musd(capex["value"], capex.get("unit"))
        row["capex_status"] = "OK" if _is_num(normalized_capex) else "UNKNOWN"
        row["capex_reason"] = "OK" if _is_num(normalized_capex) else "UNIT_AMBIGUOUS"
        row["capex_value"] = normalized_capex
        row["capex_asof_used"] = capex.get("fact_end_date")
        row["capex_filed_date"] = capex.get("filed_date")
        row["capex_input_unit"] = capex.get("unit")
        row["capex_output_unit"] = "USD_millions"
        row["capex_raw_value"] = capex.get("value")
        row["derived_from"] = _dedupe_refs(
            row["derived_from"]
            + [str(ref) for ref in (capex.get("derived_from") or []) if str(ref).strip()]
        )
    else:
        row["capex_status"], row["capex_reason"] = _field_status(None, missing_reason="TAG_MISS")

    if isinstance(fcf, dict) and _is_num(fcf.get("value")):
        normalized_fcf = _to_musd(fcf["value"], fcf.get("unit"))
        row["fcf_status"] = "OK" if _is_num(normalized_fcf) else "UNKNOWN"
        row["fcf_reason"] = "OK" if _is_num(normalized_fcf) else "UNIT_AMBIGUOUS"
        row["fcf_value"] = normalized_fcf
        row["fcf_asof_used"] = fcf.get("fact_end_date")
        row["fcf_filed_date"] = fcf.get("filed_date")
        row["fcf_input_unit"] = fcf.get("unit")
        row["fcf_output_unit"] = "USD_millions"
        row["fcf_raw_value"] = fcf.get("value")
        row["derived_from"] = _dedupe_refs(
            row["derived_from"]
            + [str(ref) for ref in (fcf.get("derived_from") or []) if str(ref).strip()]
        )
    else:
        row["fcf_status"], row["fcf_reason"] = _field_status(None, missing_reason="TAG_MISS")

    tag_refs = [ref for ref in row["derived_from"] if ref.startswith("companyfacts.")]
    non_tag_refs = [ref for ref in row["derived_from"] if not ref.startswith("companyfacts.")]
    row["derived_from"] = _dedupe_refs(tag_refs + non_tag_refs)
    row["status"] = _overall_status(
        shares_status=row["shares_status"],
        cfo_status=row["cfo_status"],
        capex_status=row["capex_status"],
        fcf_status=row["fcf_status"],
    )
    _emit_progress(
        progress_hook,
        "FACTS_READY",
        {
            "ticker": ticker_norm,
            "run_id": run_id,
            "requested_as_of": asof_norm,
            "status": row["status"],
            "fetch_reason_code": row["fetch_reason_code"],
            "fetch_attempts": row["fetch_attempts"],
        },
    )
    return _cache_and_return(row)


def _normalize_status(value: Any) -> str:
    status = str(value or "UNKNOWN").upper()
    return status if status in {"OK", "UNKNOWN"} else "UNKNOWN"


def _source_resolution_from_entries(shares_row: dict[str, Any], fcf_row: dict[str, Any]) -> str:
    candidates = [
        str(shares_row.get("shares_source_resolution") or "").strip().lower(),
        str(fcf_row.get("fcf_source_resolution") or "").strip().lower(),
        str(shares_row.get("shares_source") or "").strip().lower(),
        str(fcf_row.get("fcf_source") or "").strip().lower(),
    ]
    if any("companyfacts_fetch" in token for token in candidates):
        return "companyfacts_fetch"
    if any("companyfacts_cache" in token for token in candidates):
        return "companyfacts_cache"
    if any("historical_dossier" in token for token in candidates):
        return "historical_dossier"
    if any(token in {"current_run_dossier", "current_run_fundamentals"} for token in candidates):
        return "dossier"
    return "unknown"


def _build_facts_row_from_entries(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    shares_row: dict[str, Any] | None,
    fcf_row: dict[str, Any] | None,
) -> dict[str, Any]:
    shares_row = shares_row if isinstance(shares_row, dict) else {}
    fcf_row = fcf_row if isinstance(fcf_row, dict) else {}
    shares_status = _normalize_status(shares_row.get("shares_status"))
    cfo_status = _normalize_status(fcf_row.get("cfo_status"))
    capex_status = _normalize_status(fcf_row.get("capex_status"))
    fcf_status = _normalize_status(fcf_row.get("fcf_status"))
    derived = _dedupe_refs(
        [str(ref) for ref in (shares_row.get("derived_from") or []) if str(ref).strip()]
        + [str(ref) for ref in (fcf_row.get("derived_from") or []) if str(ref).strip()]
    )
    return enrich_facts_blocker_fields(
        {
            "ticker": str(ticker).upper(),
            "requested_as_of": as_of_date,
            "run_id": run_id,
            "cik": shares_row.get("cik") or fcf_row.get("cik"),
            "status": _overall_status(
                shares_status=shares_status,
                cfo_status=cfo_status,
                capex_status=capex_status,
                fcf_status=fcf_status,
            ),
            "shares_status": shares_status,
            "shares_reason": str(
                shares_row.get("shares_reason_code")
                or ("OK" if shares_status == "OK" else "COMPANYFACTS_MISS")
            ),
            "shares_value": shares_row.get("shares_value", UNKNOWN),
            "shares_raw_value": shares_row.get("shares_raw_value", UNKNOWN),
            "shares_input_unit": shares_row.get("shares_input_unit"),
            "shares_output_unit": shares_row.get("shares_output_unit"),
            "shares_asof_used": shares_row.get("shares_asof_used"),
            "shares_filed_date": shares_row.get("shares_filed_date"),
            "cfo_status": cfo_status,
            "cfo_reason": str(
                fcf_row.get("cfo_reason_code")
                or ("OK" if cfo_status == "OK" else "COMPANYFACTS_MISS")
            ),
            "cfo_value": fcf_row.get("cfo_value", UNKNOWN),
            "cfo_asof_used": fcf_row.get("cfo_asof_used"),
            "cfo_filed_date": fcf_row.get("cfo_filed_date"),
            "capex_status": capex_status,
            "capex_reason": str(
                fcf_row.get("capex_reason_code")
                or ("OK" if capex_status == "OK" else "COMPANYFACTS_MISS")
            ),
            "capex_value": fcf_row.get("capex_value", UNKNOWN),
            "capex_asof_used": fcf_row.get("capex_asof_used"),
            "capex_filed_date": fcf_row.get("capex_filed_date"),
            "fcf_status": fcf_status,
            "fcf_reason": str(
                fcf_row.get("fcf_reason_code")
                or ("OK" if fcf_status == "OK" else "COMPANYFACTS_MISS")
            ),
            "fcf_value": fcf_row.get("fcf_value", UNKNOWN),
            "fcf_asof_used": fcf_row.get("fcf_asof_used"),
            "fcf_filed_date": fcf_row.get("fcf_filed_date"),
            "source_resolution": _source_resolution_from_entries(shares_row, fcf_row),
            "fetch_reason_code": fcf_row.get("fetch_reason_code")
            or shares_row.get("fetch_reason_code"),
            "fetch_reason_detail": fcf_row.get("fetch_reason_detail")
            or shares_row.get("fetch_reason_detail"),
            "cache_path": fcf_row.get("cache_path") or shares_row.get("cache_path"),
            "source_url": fcf_row.get("source_url") or shares_row.get("source_url"),
            "http_status": fcf_row.get("http_status") or shares_row.get("http_status"),
            "network_attempted": bool(fcf_row.get("network_attempted"))
            or bool(shares_row.get("network_attempted")),
            "derived_from": derived,
            "generated_at": utc_now_iso(),
        }
    )


def _reason_counts(rows: list[dict[str, Any]], *, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get(key) or "UNKNOWN")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def write_facts_coverage_for_run(
    *,
    run_id: str,
    as_of_date: str,
    tickers: list[str] | None = None,
    output_dir: Path | None = None,
    cfg: AppConfig | None = None,
    sec_budget: int | None = None,
    refresh: bool = False,
    shares_entries: list[dict[str, Any]] | None = None,
    fcf_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    run_dir = output_dir or (cfg.sectors_dir / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        candidates = sorted(
            {
                str(path.stem).replace("fundamentals_", "").upper()
                for path in run_dir.glob("fundamentals_*.json")
                if str(path.stem).replace("fundamentals_", "").strip()
            }
        )
        if not candidates:
            dossier_root = cfg.dossiers_dir / run_id
            if dossier_root.exists():
                candidates = sorted(
                    [path.name.upper() for path in dossier_root.iterdir() if path.is_dir()]
                )
    else:
        candidates = sorted(
            {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        )

    shares_by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in (shares_entries or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    fcf_by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in (fcf_entries or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }

    rows: list[dict[str, Any]] = []
    remaining_budget = int(sec_budget) if sec_budget is not None else None
    for ticker in candidates:
        if ticker in shares_by_ticker or ticker in fcf_by_ticker:
            row = _build_facts_row_from_entries(
                ticker=ticker,
                as_of_date=as_of_date,
                run_id=run_id,
                shares_row=shares_by_ticker.get(ticker),
                fcf_row=fcf_by_ticker.get(ticker),
            )
        else:
            row = resolve_financial_facts_asof(
                ticker=ticker,
                as_of_date=as_of_date,
                run_id=run_id,
                refresh=refresh,
                sec_budget=remaining_budget,
                cfg=cfg,
            )
            if (
                remaining_budget is not None
                and bool(row.get("network_attempted"))
                and remaining_budget > 0
            ):
                remaining_budget -= 1
        rows.append(row)

    rows = sorted(rows, key=lambda row: str(row.get("ticker") or ""))
    status_counts: dict[str, int] = {}
    shares_ok = 0
    cfo_ok = 0
    capex_ok = 0
    fcf_ok = 0
    for row in rows:
        status = str(row.get("status") or "UNKNOWN").upper()
        status_counts[status] = status_counts.get(status, 0) + 1
        if str(row.get("shares_status") or "").upper() == "OK":
            shares_ok += 1
        if str(row.get("cfo_status") or "").upper() == "OK":
            cfo_ok += 1
        if str(row.get("capex_status") or "").upper() == "OK":
            capex_ok += 1
        if str(row.get("fcf_status") or "").upper() == "OK":
            fcf_ok += 1

    facts_blocker_summary = summarize_facts_blockers(rows)

    payload = {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "ticker_count": len(rows),
        "status_counts": dict(sorted(status_counts.items(), key=lambda kv: kv[0])),
        "shares_ok_count": int(shares_ok),
        "cfo_ok_count": int(cfo_ok),
        "capex_ok_count": int(capex_ok),
        "fcf_ok_count": int(fcf_ok),
        "facts_blocker_histogram": facts_blocker_summary.get("facts_blocker_histogram")
        if isinstance(facts_blocker_summary.get("facts_blocker_histogram"), dict)
        else {},
        "retryable_facts_blocker_count": int(
            facts_blocker_summary.get("retryable_facts_blocker_count") or 0
        ),
        "terminal_facts_blocker_count": int(
            facts_blocker_summary.get("terminal_facts_blocker_count") or 0
        ),
        "partial_usable_facts_count": int(
            facts_blocker_summary.get("partial_usable_facts_count") or 0
        ),
        "top_retryable_facts_blockers": facts_blocker_summary.get("top_retryable_facts_blockers")
        if isinstance(facts_blocker_summary.get("top_retryable_facts_blockers"), list)
        else [],
        "top_terminal_facts_blockers": facts_blocker_summary.get("top_terminal_facts_blockers")
        if isinstance(facts_blocker_summary.get("top_terminal_facts_blockers"), list)
        else [],
        "top_partial_usable_facts": facts_blocker_summary.get("top_partial_usable_facts")
        if isinstance(facts_blocker_summary.get("top_partial_usable_facts"), list)
        else [],
        "recommended_next_action_counts": facts_blocker_summary.get(
            "recommended_next_action_counts"
        )
        if isinstance(facts_blocker_summary.get("recommended_next_action_counts"), dict)
        else {},
        "economic_fail_count_vs_evidence_fail_count": facts_blocker_summary.get(
            "economic_fail_count_vs_evidence_fail_count"
        )
        if isinstance(facts_blocker_summary.get("economic_fail_count_vs_evidence_fail_count"), dict)
        else {},
        "reason_counts": {
            "shares_reason": _reason_counts(rows, key="shares_reason"),
            "cfo_reason": _reason_counts(rows, key="cfo_reason"),
            "capex_reason": _reason_counts(rows, key="capex_reason"),
            "fcf_reason": _reason_counts(rows, key="fcf_reason"),
        },
        "entries": rows,
        "generated_at": utc_now_iso(),
    }
    out_path = run_dir / "facts_coverage.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["facts_coverage_path"] = str(out_path)
    return payload
