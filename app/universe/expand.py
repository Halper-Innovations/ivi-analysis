"""Universe expansion: ingest and score tickers from the discovery seed.

For each ticker in the seed that doesn't have a scorecard:
  1. Resolve CIK via SEC company_tickers.json
  2. Fetch companyfacts from EDGAR (cached)
  3. Run the full valuation pipeline (DCF, EPV, Graham, scorecard)

Progress is reported every N tickers. The job is resumable — tickers
with existing scorecards are skipped.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)


@dataclass
class ExpansionSummary:
    seed_total: int
    already_scored: int
    attempted: int
    succeeded: int
    failed: int
    elapsed_seconds: float
    failed_tickers: list[str]


RUSSELL_3000_DEFAULT_PATH = Path("data/external/russell_3000.csv")


def _normalize_ticker_symbol(ticker: str, *, cik_map: dict[str, str] | None = None) -> str:
    """Normalize external ticker syntax toward the SEC ticker registry."""
    upper = str(ticker or "").upper().strip().replace(".", "-")
    if not upper or cik_map is None or upper in cik_map:
        return upper
    if "-" not in upper and len(upper) > 1:
        class_share_candidate = f"{upper[:-1]}-{upper[-1]}"
        if class_share_candidate in cik_map:
            return class_share_candidate
    return upper


def _load_seed_tickers(seed_path: Path) -> list[str]:
    """Read tickers from the discovery seed CSV."""
    with open(seed_path) as f:
        return [r["ticker"].upper().strip() for r in csv.DictReader(f)]


def _load_russell_3000_tickers(path: Path | None = None) -> list[str]:
    """Read the staged Russell 3000 constituent CSV."""
    from app.universe.ticker_cik_map import load_ticker_cik_map

    source_path = path or RUSSELL_3000_DEFAULT_PATH
    cik_map = load_ticker_cik_map()
    tickers: list[str] = []
    seen: set[str] = set()
    with open(source_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = _normalize_ticker_symbol(str(row.get("ticker") or ""), cik_map=cik_map)
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def _load_sec_registry_tickers() -> list[str]:
    """Load real operating-company tickers from the SEC registry.

    Filters to companies that actually file 10-K or 20-F (annual reports),
    excludes warrants/units/rights by ticker suffix, and sorts by filing
    recency (most recent filers first = most likely to have good data).

    This ensures ivi expand-universe --source sec always targets real
    companies first, not shells/SPACs/warrants.
    """
    import json as _json
    from pathlib import Path as _Path

    from app.universe.ticker_cik_map import load_ticker_cik_map
    from app.ingest.cik_registry import resolve

    cik_map = load_ticker_cik_map()
    submissions_dir = _Path("data/cache/submissions")

    # Suffixes that indicate warrants, units, rights — not operating companies
    _JUNK_SUFFIXES = ("W", "WS", "WT", "WI", "U", "UN", "R", "RT")

    real_tickers: list[tuple[str, str]] = []  # (ticker, latest_filing_date)
    junk_skipped = 0

    for ticker in cik_map:
        # Skip warrants/units/rights
        if len(ticker) > 3 and any(ticker.endswith(s) for s in _JUNK_SUFFIXES):
            junk_skipped += 1
            continue

        # Check if we have submissions cached
        try:
            cik = resolve(ticker)
        except Exception:
            continue

        sub_path = submissions_dir / f"{str(cik).zfill(10)}.json"
        if not sub_path.exists():
            continue

        # Check for 10-K or 20-F filings
        try:
            with open(sub_path) as f:
                sub = _json.load(f)
            forms = sub.get("filings", {}).get("recent", {}).get("form", [])
            dates = sub.get("filings", {}).get("recent", {}).get("filingDate", [])

            has_annual = any(
                f in ("10-K", "10-K/A", "10-KSB", "10-KSB/A", "20-F", "20-F/A") for f in forms[:30]
            )
            if not has_annual:
                continue

            # Use most recent filing date for sort priority
            latest_date = dates[0] if dates else "1900-01-01"
            real_tickers.append((ticker, latest_date))
        except Exception:
            continue

    # Sort by most recent filing first — real active companies get processed first
    real_tickers.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        "SEC registry filter: %d total, %d junk suffixes skipped, %d real 10-K/20-F filers",
        len(cik_map),
        junk_skipped,
        len(real_tickers),
    )

    return [t for t, _ in real_tickers]


def _existing_scorecards(db_path: str | Path) -> set[str]:
    """Return tickers whose newest scorecard row is decision-eligible."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM valuations WHERE method = 'scorecard'"
        ).fetchall()
        return {
            str(row["ticker"]).upper()
            for row in rows
            if latest_decision_eligible_valuation_row(
                conn,
                ticker=str(row["ticker"]),
                method="scorecard",
            )
            is not None
        }
    finally:
        conn.close()


def expand_universe(
    *,
    source: str = "seed",
    seed_path: Path | None = None,
    limit: int | None = None,
    as_of_date: str | None = None,
    report_every: int = 25,
    dry_run: bool = False,
) -> ExpansionSummary:
    """Expand the universe by ingesting and scoring tickers.

    source: "seed" (discovery_seed.csv, ~2500), "russell3000"
    (staged IWV holdings), or "sec" (full SEC registry, ~10,000+)
    Resumable: skips tickers that already have scorecards.
    """
    from app.config import get_config
    from app.ingest.cik_registry import resolve, CIKNotFoundError
    from app.ingest.facts_writer import ensure_all_facts
    from app.valuation.valuation_writer import ensure_valuation

    cfg = get_config()
    effective_date = as_of_date or date.today().isoformat()

    normalized_source = str(source or "seed").strip().lower().replace("_", "").replace("-", "")
    if normalized_source == "sec":
        seed = _load_sec_registry_tickers()
    elif normalized_source in {"russell3000", "russell"}:
        seed = _load_russell_3000_tickers()
    else:
        if seed_path is None:
            seed_path = Path(cfg.discovery_seed_path)
        seed = _load_seed_tickers(seed_path)
    existing = _existing_scorecards(cfg.db_path)

    todo = [t for t in seed if t not in existing]
    if limit is not None:
        todo = todo[:limit]

    logger.info(
        "expand_universe: %d seed, %d existing, %d to process%s",
        len(seed),
        len(existing),
        len(todo),
        " (dry run)" if dry_run else "",
    )

    if dry_run:
        return ExpansionSummary(
            seed_total=len(seed),
            already_scored=len(existing),
            attempted=0,
            succeeded=0,
            failed=0,
            elapsed_seconds=0.0,
            failed_tickers=[],
        )

    t0 = time.perf_counter()
    succeeded = 0
    failed = 0
    failed_tickers: list[str] = []

    def _process_one(ticker: str) -> tuple[str, bool, str]:
        """Process one ticker. Returns (ticker, success, error_msg)."""
        try:
            try:
                resolve(ticker)
            except CIKNotFoundError:
                return (ticker, False, "no_cik")
            ensure_all_facts(ticker)
            ensure_valuation(
                ticker,
                effective_date,
                require_filed_asof=True,
            )
            return (ticker, True, "")
        except Exception as exc:
            return (ticker, False, str(exc))

    # Use thread pool for IO-bound EDGAR fetches + CPU valuation
    # SQLite handles concurrent reads; writes serialize via its internal lock
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = 2  # Low thread count to avoid SQLite lock contention
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in todo}
        for future in as_completed(futures):
            ticker, success, error_msg = future.result()
            completed += 1
            if success:
                succeeded += 1
            else:
                failed += 1
                if error_msg != "no_cik":
                    failed_tickers.append(ticker)

            if completed % report_every == 0:
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - completed) / rate if rate > 0 else 0
                logger.info(
                    "expand_universe: %d/%d done (%d ok, %d fail) — %.1f/min, ~%.0f min remaining",
                    completed,
                    len(todo),
                    succeeded,
                    failed,
                    rate * 60,
                    remaining / 60,
                )

    elapsed = time.perf_counter() - t0
    logger.info(
        "expand_universe complete: %d attempted, %d succeeded, %d failed in %.0fs",
        len(todo),
        succeeded,
        failed,
        elapsed,
    )

    return ExpansionSummary(
        seed_total=len(seed),
        already_scored=len(existing),
        attempted=len(todo),
        succeeded=succeeded,
        failed=failed,
        elapsed_seconds=elapsed,
        failed_tickers=sorted(failed_tickers),
    )
