from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.logging import get_logger
from app.market.company_facts_provider import _write_cached_companyfacts, normalize_cik
from app.sector.taxonomy import load_sector_taxonomy
from app.universe.sector_universe import discover_sector_universe
from app.universe.ticker_cik_map import refresh_ticker_cik_cache
from app.util.http import HttpClient


logger = get_logger(__name__)
_BOOTSTRAP_TTL_SECONDS = 7 * 24 * 3600
_SEC_DELAY_SECONDS = 0.15


def _select_target_tickers(*, sector: str, tickers: list[str] | None = None) -> list[str]:
    try:
        discovered = discover_sector_universe(sector)
        discovered_tickers = sorted(
            {
                str(row.get("ticker") or "").strip().upper()
                for row in discovered
                if str(row.get("ticker") or "").strip()
            }
        )
        explicit = sorted({str(ticker).strip().upper() for ticker in (tickers or []) if str(ticker).strip()})
        if explicit:
            return [ticker for ticker in explicit if not discovered_tickers or ticker in set(discovered_tickers)]
        if discovered_tickers:
            return discovered_tickers
    except Exception:
        pass
    mapping = load_sector_taxonomy()
    sector_norm = str(sector or "").strip().lower()
    sector_tickers = sorted(
        [ticker for ticker, mapped_sector in mapping.items() if str(mapped_sector).strip().lower() == sector_norm]
    )
    explicit = sorted({str(ticker).strip().upper() for ticker in (tickers or []) if str(ticker).strip()})
    if explicit:
        return [ticker for ticker in explicit if not sector_tickers or ticker in set(sector_tickers)]
    return sector_tickers


def submissions_cache_path(cik: str | int, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    cik_norm = normalize_cik(cik)
    path = cfg.cache_dir / "submissions" / f"{cik_norm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cache_is_fresh(path: Path, *, ttl_seconds: int = _BOOTSTRAP_TTL_SECONDS) -> bool:
    if not path.exists():
        return False
    try:
        age_seconds = time.time() - path.stat().st_mtime
    except Exception:
        return False
    return age_seconds <= max(1, int(ttl_seconds))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_http_cache_json(http: HttpClient, url: str, payload: dict[str, Any]) -> Path:
    path = http.cache_path(url)
    _write_json(path, payload)
    return path


def bootstrap_sector_cache(
    *,
    sector: str,
    tickers: list[str] | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    http = HttpClient(cfg)
    target_tickers = _select_target_tickers(sector=sector, tickers=tickers)
    if not target_tickers:
        return {
            "status": "NO_TARGETS",
            "sector": sector,
            "tickers": [],
            "companyfacts_cached": 0,
            "submissions_cached": 0,
            "skipped": 0,
            "failed": 0,
            "ticker_results": [],
        }

    mapping = refresh_ticker_cik_cache(http=http)
    companyfacts_cached = 0
    submissions_cached = 0
    skipped = 0
    failed = 0
    ticker_results: list[dict[str, Any]] = []

    for ticker in target_tickers:
        ticker_token = str(ticker or "").strip().upper()
        cik = normalize_cik(mapping.get(ticker_token))
        result: dict[str, Any] = {
            "ticker": ticker_token,
            "cik": cik,
            "companyfacts_cache_path": "",
            "submissions_cache_path": "",
            "status": "PENDING",
            "steps": [],
        }
        if not cik:
            failed += 1
            result["status"] = "FAILED"
            result["error"] = "CIK_NOT_FOUND"
            ticker_results.append(result)
            logger.warning("bootstrap sector cache failed for %s: missing CIK", ticker_token)
            continue

        companyfacts_path = cfg.cache_dir / "companyfacts" / f"{cik}.json"
        submissions_path = submissions_cache_path(cik, cfg=cfg)
        result["companyfacts_cache_path"] = str(companyfacts_path)
        result["submissions_cache_path"] = str(submissions_path)

        ticker_failed = False

        if _cache_is_fresh(companyfacts_path):
            skipped += 1
            result["steps"].append({"kind": "companyfacts", "status": "SKIPPED", "reason": "FRESH_CACHE"})
        else:
            companyfacts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            try:
                companyfacts_payload = http.get_json(companyfacts_url, use_cache=True, cache_ttl_seconds=_BOOTSTRAP_TTL_SECONDS)
                _write_cached_companyfacts(
                    path=companyfacts_path,
                    cik=cik,
                    source_url=companyfacts_url,
                    http_status=200,
                    companyfacts=companyfacts_payload,
                )
                companyfacts_cached += 1
                result["steps"].append({"kind": "companyfacts", "status": "CACHED"})
                logger.info("Cached companyfacts for %s (CIK %s)", ticker_token, str(int(cik)))
            except Exception as exc:  # noqa: BLE001
                ticker_failed = True
                failed += 1
                result["steps"].append({"kind": "companyfacts", "status": "FAILED", "error": str(exc)})
                logger.warning("bootstrap companyfacts failed for %s (%s): %s", ticker_token, cik, exc)
            time.sleep(_SEC_DELAY_SECONDS)

        if _cache_is_fresh(submissions_path):
            skipped += 1
            result["steps"].append({"kind": "submissions", "status": "SKIPPED", "reason": "FRESH_CACHE"})
        else:
            submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            try:
                submissions_payload = http.get_json(submissions_url, use_cache=True, cache_ttl_seconds=_BOOTSTRAP_TTL_SECONDS)
                _write_json(submissions_path, submissions_payload)
                _write_http_cache_json(http, submissions_url, submissions_payload)
                submissions_cached += 1
                result["steps"].append({"kind": "submissions", "status": "CACHED"})
                logger.info("Cached submissions for %s (CIK %s)", ticker_token, str(int(cik)))
            except Exception as exc:  # noqa: BLE001
                ticker_failed = True
                failed += 1
                result["steps"].append({"kind": "submissions", "status": "FAILED", "error": str(exc)})
                logger.warning("bootstrap submissions failed for %s (%s): %s", ticker_token, cik, exc)
            time.sleep(_SEC_DELAY_SECONDS)

        result["status"] = "FAILED" if ticker_failed else "OK"
        ticker_results.append(result)

    logger.info(
        "Bootstrap complete: %s cached, %s skipped, %s failed",
        companyfacts_cached + submissions_cached,
        skipped,
        failed,
    )
    return {
        "status": "OK",
        "sector": sector,
        "tickers": target_tickers,
        "companyfacts_cached": companyfacts_cached,
        "submissions_cached": submissions_cached,
        "skipped": skipped,
        "failed": failed,
        "ticker_results": ticker_results,
    }
