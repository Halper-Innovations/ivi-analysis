from __future__ import annotations

import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from app.config import AppConfig, get_config
from app.ingest.sec_client import FilingStub, SecClient
from app.logging import get_logger
from app.valuation.facts import resolve_cik_for_ticker


logger = get_logger(__name__)


def filing_cache_path(*, cik: str, accession: str, cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    return (
        cfg.cache_dir
        / "filings"
        / str(cik).strip()
        / str(accession).strip()
        / "primary_document.html"
    )


def find_cached_filing(
    *,
    filing: FilingStub,
    cfg: AppConfig | None = None,
) -> Path | None:
    """Return an existing cached primary document without creating or copying."""

    cfg = cfg or get_config()
    candidates = [
        filing_cache_path(cik=filing.cik, accession=filing.accession, cfg=cfg),
        filing_cache_path(cik=filing.cik, accession=filing.accession_nodash, cfg=cfg),
    ]
    return next((path for path in candidates if path.is_file()), None)


def warm_cached_filing_to_raw(
    *,
    filing: FilingStub,
    target_path: Path,
    cfg: AppConfig | None = None,
) -> bool:
    cfg = cfg or get_config()
    source_path = find_cached_filing(filing=filing, cfg=cfg)
    if source_path is None:
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return True


def _annual_forms_for_warmup() -> list[str]:
    return ["10-K", "10-K/A", "20-F", "20-F/A", "40-F"]


def _select_target_tickers(*, sector: str, tickers: list[str] | None = None) -> list[str]:
    from app.sector.taxonomy import load_sector_taxonomy

    mapping = load_sector_taxonomy()
    sector_norm = str(sector or "").strip().lower()
    sector_tickers = sorted(
        [
            ticker
            for ticker, mapped_sector in mapping.items()
            if str(mapped_sector).strip().lower() == sector_norm
        ]
    )
    explicit = sorted(
        {str(ticker).strip().upper() for ticker in (tickers or []) if str(ticker).strip()}
    )
    if explicit:
        return [
            ticker for ticker in explicit if not sector_tickers or ticker in set(sector_tickers)
        ]
    return sector_tickers


def warm_filing_cache(
    *,
    sector: str,
    tickers: list[str] | None = None,
    years: int = 3,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config()
    target_tickers = _select_target_tickers(sector=sector, tickers=tickers)
    if not target_tickers:
        return {
            "status": "NO_TARGETS",
            "sector": sector,
            "tickers": [],
            "years": max(1, int(years)),
            "requested_filings": 0,
            "cached": 0,
            "skipped": 0,
            "failed": 0,
        }

    client = SecClient()
    as_of = date.today()
    start_year = max(1995, as_of.year - max(2, int(years) + 1))
    try:
        start_date = as_of.replace(year=start_year)
    except ValueError:
        start_date = as_of.replace(year=start_year, day=28)
    forms = _annual_forms_for_warmup()

    selected_by_ticker: dict[str, list[FilingStub]] = {}
    for ticker in target_tickers:
        cik = str(resolve_cik_for_ticker(ticker, cfg=cfg, refresh_if_missing=True) or "").strip()
        if not cik:
            selected_by_ticker[ticker] = []
            continue
        try:
            filings = client.list_filings_window(
                cik, start_date=start_date, end_date=as_of, forms=forms
            )
        except Exception:
            filings = client.list_cached_filings_window(
                cik, start_date=start_date, end_date=as_of, forms=forms
            )
        annuals = sorted(
            filings,
            key=lambda item: (
                str(item.period_end or ""),
                item.filing_date.isoformat(),
                item.form_type,
                item.accession,
            ),
            reverse=True,
        )
        selected: list[FilingStub] = []
        seen_years: set[int] = set()
        for filing in annuals:
            fiscal_year = int(str(filing.period_end or filing.filing_date.isoformat())[:4])
            if fiscal_year in seen_years:
                continue
            seen_years.add(fiscal_year)
            selected.append(filing)
            if len(selected) >= max(1, int(years)):
                break
        selected_by_ticker[ticker] = selected

    total = sum(len(rows) for rows in selected_by_ticker.values())
    cached = 0
    skipped = 0
    failed = 0
    _progress_lock = threading.Lock()
    _progress = [0]  # mutable counter for threads
    _cached = [0]
    _skipped = [0]
    _failed = [0]

    # Build flat work list: (ticker, filing) pairs that need downloading
    work_items: list[tuple[str, FilingStub]] = []
    for ticker in target_tickers:
        for filing in selected_by_ticker.get(ticker) or []:
            path = filing_cache_path(cik=filing.cik, accession=filing.accession, cfg=cfg)
            if path.exists():
                with _progress_lock:
                    _skipped[0] += 1
                    _progress[0] += 1
                continue
            work_items.append((ticker, filing))

    def _download_one(item: tuple[str, FilingStub]) -> None:
        ticker, filing = item
        path = filing_cache_path(cik=filing.cik, accession=filing.accession, cfg=cfg)
        try:
            payload = client.download_bytes(filing.primary_doc_url, use_cache=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            with _progress_lock:
                _cached[0] += 1
                _progress[0] += 1
                logger.info(
                    "Cached filing %s for %s (%s of %s)",
                    filing.accession,
                    ticker,
                    _progress[0],
                    total,
                )
        except Exception as exc:  # noqa: BLE001
            with _progress_lock:
                _failed[0] += 1
                _progress[0] += 1
            logger.warning(
                "filing cache warm-up failed for %s %s: %s", ticker, filing.accession, exc
            )

    # Download in parallel — rate limiter is global and thread-safe
    n_workers = min(cfg.max_workers, max(1, len(work_items)))
    if work_items:
        logger.info(
            "Downloading %s filings with %s workers (%s already cached)",
            len(work_items),
            n_workers,
            _skipped[0],
        )
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_download_one, item) for item in work_items]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass  # already logged inside _download_one

    cached = _cached[0]
    skipped = _skipped[0]
    failed = _failed[0]
    logger.info("Warm-up complete: %s cached, %s skipped, %s failed", cached, skipped, failed)
    return {
        "status": "OK",
        "sector": sector,
        "tickers": target_tickers,
        "years": max(1, int(years)),
        "requested_filings": total,
        "cached": cached,
        "skipped": skipped,
        "failed": failed,
    }
