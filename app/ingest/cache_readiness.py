"""Provider-free cache readiness classification for benchmark candidates."""
from __future__ import annotations

from collections import Counter
from typing import Any


CACHE_READY = "CACHE_READY"
CACHE_PARTIAL = "CACHE_PARTIAL"
CACHE_NOT_USABLE = "CACHE_NOT_USABLE"
CACHE_LIMITED = "CACHE_LIMITED"
CACHE_NOT_READY = "CACHE_NOT_READY"
CACHE_THIN_CANDIDATE_POOL = "CACHE_THIN_CANDIDATE_POOL"
CACHE_NO_SECTOR_SCOPE = "NO_SECTOR_SCOPE"
CACHE_STEP_OK = {"OK", "SKIPPED"}
CACHE_BUFFER_CAP = 32


def counter_dict(values: list[str]) -> dict[str, int]:
    """Return deterministic sorted counts for artifact/report output."""

    counter = Counter(str(value) for value in values if str(value))
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def cache_prewarm_limit(max_candidates: int | None, explicit_limit: int | None = None) -> int | None:
    """Return the benchmark/cycle default buffered cache scope."""

    if explicit_limit is not None and int(explicit_limit) > 0:
        return int(explicit_limit)
    if max_candidates is None or int(max_candidates) <= 0:
        return None
    return min(int(max_candidates) * 2, CACHE_BUFFER_CAP)


def recommended_cache_refresh_limit(*, max_candidates: int | None, warmed_count: int) -> int | None:
    """Recommend a larger provider-free refresh scope when the warmed pool is thin."""

    base = cache_prewarm_limit(max_candidates)
    if base is None:
        return None
    if warmed_count <= 0:
        return base
    return min(max(base, warmed_count * 2), CACHE_BUFFER_CAP)


def step_status(row: dict[str, Any], step_name: str) -> str:
    steps = row.get("steps") if isinstance(row.get("steps"), dict) else {}
    step = steps.get(step_name) if isinstance(steps.get(step_name), dict) else {}
    return str(step.get("status") or "MISSING").upper()


def ticker_cache_readiness(row: dict[str, Any] | None, ticker: str) -> dict[str, Any]:
    """Classify whether one refreshed ticker has enough cache to enter benchmark execution."""

    ticker_norm = str(ticker or "").strip().upper()
    if row is None:
        return {
            "ticker": ticker_norm,
            "readiness": CACHE_NOT_USABLE,
            "reasons": ["CACHE_MISSING_TICKER"],
            "step_statuses": {"facts": "MISSING", "filings": "MISSING", "price": "MISSING"},
        }

    facts_status = step_status(row, "facts")
    filing_status = step_status(row, "filings")
    price_status = step_status(row, "price")
    reasons: list[str] = []
    if row.get("errors"):
        reasons.append("REFRESH_FAILURE")
    if facts_status not in CACHE_STEP_OK:
        reasons.append(f"FACTS_NOT_OK:{facts_status}")
    if price_status not in CACHE_STEP_OK:
        reasons.append(f"PRICE_NOT_OK:{price_status}")

    if reasons:
        readiness = CACHE_NOT_USABLE
    elif filing_status in CACHE_STEP_OK:
        readiness = CACHE_READY
    else:
        readiness = CACHE_PARTIAL
        reasons.append(f"FILINGS_NOT_OK:{filing_status}")

    return {
        "ticker": ticker_norm,
        "readiness": readiness,
        "reasons": reasons,
        "step_statuses": {
            "facts": facts_status,
            "filings": filing_status,
            "price": price_status,
        },
    }


def cache_readiness_for_tickers(
    summary: dict[str, Any] | None,
    tickers: list[str],
    *,
    max_candidates: int | None,
) -> dict[str, Any]:
    """Classify a sector's warmed candidate list into benchmark-ready execution pool metadata."""

    tickers_norm = [str(ticker or "").upper() for ticker in tickers if str(ticker or "").strip()]
    desired_count = min(int(max_candidates), len(tickers_norm)) if max_candidates is not None and int(max_candidates) > 0 else len(tickers_norm)
    recommended_limit = recommended_cache_refresh_limit(max_candidates=max_candidates, warmed_count=len(tickers_norm))
    if not isinstance(summary, dict) or not tickers_norm:
        reason = "CACHE_REFRESH_UNAVAILABLE" if not isinstance(summary, dict) else "CACHE_NO_SECTOR_TICKERS"
        return {
            "coverage_status": CACHE_NOT_READY,
            "candidate_count": 0,
            "warmed_candidate_count": len(tickers_norm),
            "desired_candidate_count": desired_count,
            "ready_count": 0,
            "partial_count": 0,
            "not_usable_count": len(tickers_norm),
            "readiness_counts": {CACHE_NOT_USABLE: len(tickers_norm)} if tickers_norm else {},
            "candidate_readiness": [ticker_cache_readiness(None, ticker) for ticker in tickers_norm],
            "ready_tickers": [],
            "partial_tickers": [],
            "not_usable_tickers": tickers_norm,
            "excluded_tickers": tickers_norm,
            "final_candidate_pool": [],
            "cache_limited_reasons": [reason],
            "cache_readiness_warnings": [reason],
            "recommended_cache_max_tickers_per_sector": recommended_limit,
        }

    rows_by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in (summary.get("ticker_results") or [])
        if isinstance(row, dict)
    }
    entries = [ticker_cache_readiness(rows_by_ticker.get(ticker), ticker) for ticker in tickers_norm]
    readiness_counts = counter_dict([str(entry.get("readiness") or "") for entry in entries])
    ready_tickers = [str(entry["ticker"]) for entry in entries if entry.get("readiness") == CACHE_READY]
    partial_tickers = [str(entry["ticker"]) for entry in entries if entry.get("readiness") == CACHE_PARTIAL]
    not_usable_tickers = [str(entry["ticker"]) for entry in entries if entry.get("readiness") == CACHE_NOT_USABLE]
    final_candidate_pool = ready_tickers[:desired_count] if desired_count > 0 else ready_tickers
    warnings: list[str] = []
    limited_reasons: list[str] = []
    if partial_tickers:
        warnings.append(f"CACHE_PARTIAL_CANDIDATES:{len(partial_tickers)}")
    if not_usable_tickers:
        warnings.append(f"CACHE_NOT_USABLE_CANDIDATES:{len(not_usable_tickers)}")
    if not ready_tickers:
        limited_reasons.append("NO_CACHE_READY_CANDIDATES")
        coverage_status = CACHE_NOT_READY
    elif len(final_candidate_pool) < desired_count:
        limited_reasons.append(f"{CACHE_THIN_CANDIDATE_POOL}:{len(final_candidate_pool)}/{desired_count}")
        coverage_status = CACHE_LIMITED
    else:
        coverage_status = CACHE_READY
    return {
        "coverage_status": coverage_status,
        "candidate_count": len(final_candidate_pool),
        "warmed_candidate_count": len(tickers_norm),
        "desired_candidate_count": desired_count,
        "ready_count": len(ready_tickers),
        "partial_count": len(partial_tickers),
        "not_usable_count": len(not_usable_tickers),
        "readiness_counts": readiness_counts,
        "candidate_readiness": entries,
        "ready_tickers": ready_tickers,
        "partial_tickers": partial_tickers,
        "not_usable_tickers": not_usable_tickers,
        "excluded_tickers": partial_tickers + not_usable_tickers,
        "final_candidate_pool": final_candidate_pool,
        "cache_limited_reasons": limited_reasons,
        "cache_readiness_warnings": warnings,
        "recommended_cache_max_tickers_per_sector": recommended_limit,
    }


def refreshed_tickers_by_sector(summary: dict[str, Any] | None) -> dict[str, list[str]]:
    """Return the refresh candidate ordering per sector from a refresh summary."""

    if not isinstance(summary, dict):
        return {}
    scope = summary.get("scope") if isinstance(summary.get("scope"), dict) else {}
    by_sector: dict[str, list[str]] = {}
    for selection in scope.get("sector_selections") or []:
        if not isinstance(selection, dict):
            continue
        sector = str(selection.get("sector") or "").strip()
        if not sector:
            continue
        tickers = [
            str(ticker or "").upper()
            for ticker in (selection.get("selected_tickers") or [])
            if str(ticker or "").strip()
        ]
        by_sector[sector] = list(dict.fromkeys(tickers))
    return by_sector


def cache_readiness_by_sector(
    summary: dict[str, Any] | None,
    *,
    max_candidates: int | None,
    sectors: list[str] | None = None,
    tickers_by_sector: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build provider-free cache readiness rollups for every sector in a refresh summary."""

    sector_tickers = tickers_by_sector if tickers_by_sector is not None else refreshed_tickers_by_sector(summary)
    sector_filter = {str(sector or "").strip() for sector in (sectors or []) if str(sector or "").strip()}
    if sector_filter:
        sector_tickers = {sector: tickers for sector, tickers in sector_tickers.items() if sector in sector_filter}
    if not sector_tickers:
        return {
            "overall_status": CACHE_NO_SECTOR_SCOPE,
            "sector_count": 0,
            "sectors": {},
            "rollups": {
                "coverage_status_counts": {},
                "candidate_readiness_counts": {},
                "cache_readiness_warning_counts": {},
                "cache_limited_sectors": [],
            },
        }

    sector_results = {
        sector: cache_readiness_for_tickers(summary, list(tickers), max_candidates=max_candidates)
        for sector, tickers in sector_tickers.items()
    }
    coverage_statuses = [str(row.get("coverage_status") or "UNKNOWN") for row in sector_results.values()]
    candidate_counts = Counter()
    warning_values: list[str] = []
    limited_sectors: list[dict[str, Any]] = []
    recommended_limits = [
        int(row.get("recommended_cache_max_tickers_per_sector"))
        for row in sector_results.values()
        if row.get("recommended_cache_max_tickers_per_sector") is not None
    ]
    for sector, row in sector_results.items():
        candidate_counts.update(row.get("readiness_counts") or {})
        warning_values.extend(str(item) for item in row.get("cache_readiness_warnings") or [])
        reasons = list(row.get("cache_limited_reasons") or [])
        if reasons:
            limited_sectors.append({"sector": sector, "reasons": reasons})
    if any(status == CACHE_NOT_READY for status in coverage_statuses):
        overall_status = CACHE_NOT_READY
    elif any(status == CACHE_LIMITED for status in coverage_statuses):
        overall_status = CACHE_LIMITED
    elif coverage_statuses and all(status == CACHE_READY for status in coverage_statuses):
        overall_status = CACHE_READY
    else:
        overall_status = CACHE_LIMITED
    return {
        "overall_status": overall_status,
        "sector_count": len(sector_results),
        "sectors": sector_results,
        "rollups": {
            "coverage_status_counts": counter_dict(coverage_statuses),
            "candidate_readiness_counts": dict(sorted(candidate_counts.items())),
            "cache_readiness_warning_counts": counter_dict(warning_values),
            "cache_limited_sectors": limited_sectors,
            "recommended_cache_max_tickers_per_sector": max(recommended_limits) if recommended_limits else None,
        },
    }


__all__ = [
    "CACHE_BUFFER_CAP",
    "CACHE_LIMITED",
    "CACHE_NO_SECTOR_SCOPE",
    "CACHE_NOT_READY",
    "CACHE_NOT_USABLE",
    "CACHE_PARTIAL",
    "CACHE_READY",
    "CACHE_THIN_CANDIDATE_POOL",
    "cache_prewarm_limit",
    "cache_readiness_by_sector",
    "cache_readiness_for_tickers",
    "counter_dict",
    "recommended_cache_refresh_limit",
    "refreshed_tickers_by_sector",
    "ticker_cache_readiness",
]
