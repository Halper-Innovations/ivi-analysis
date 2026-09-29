"""SEC registrant census: the true US-exchange operating-company universe.

The platform's classified universe (sector_inference) is Russell-3000-derived
and structurally excludes most sub-$200M registrants. This module builds the
universe from the SEC's own exchange registry instead:

  1. Pull company_tickers_exchange.json (cik, name, ticker, exchange).
  2. Keep US exchange listings (NYSE / Nasdaq; the SEC registry folds NYSE
     American into "NYSE"). OTC is explicitly deferred as a v2 owner decision.
  3. Dedupe share classes to one primary ticker per CIK.
  4. Filter to operating companies via EDGAR submissions: files 10-K/10-Q,
     not a fund/trust (SIC 6722/6726), not a blank-check shell (SIC 6770),
     not a warrant/right/unit-style symbol.
  5. Persist to sec_registrants and report new operating companies vs the
     currently classified universe — the Phase A report gate.

All EDGAR access goes through the shared rate-limited HttpClient; submissions
payloads reuse the same on-disk cache as the sector classifier, so the census
never refetches what classification already cached (and vice versa).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.db import connect as db_connect
from app.config import AppConfig, get_config

logger = logging.getLogger(__name__)

EXCHANGE_REGISTRY_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
EXCHANGE_REGISTRY_CACHE_NAME = "company_tickers_exchange.json"
_REGISTRY_TTL_SECONDS = 7 * 24 * 3600

# The SEC exchange registry labels NYSE American listings as "NYSE", so the
# NYSE/Nasdaq pair covers the NYSE/NASDAQ/AMEX scope. OTC is deferred (v2).
IN_SCOPE_EXCHANGES = frozenset({"NYSE", "NASDAQ"})
DEFERRED_EXCHANGES = frozenset({"OTC"})

# Investment companies: management investment offices (6722) and investment
# offices NEC / closed-end funds and unit trusts (6726). Blank-check shells
# (6770) follow the existing classifier convention.
FUND_TRUST_SIC_CODES = frozenset({6722, 6726})
BLANK_CHECK_SIC = 6770

# 10-K prefix covers 10-K/A, 10-KSB, 10-K405...; 10-Q covers 10-Q/A, 10-QSB.
_ANNUAL_FORM_PREFIX = "10-K"
_QUARTERLY_FORM_PREFIX = "10-Q"
_FOREIGN_ANNUAL_PREFIXES = ("20-F", "40-F")

OPERATING = "OPERATING"
EXCHANGE_SCOPE_IN = "IN_SCOPE"
EXCHANGE_SCOPE_OTC = "OTC_DEFERRED"
EXCHANGE_SCOPE_OTHER = "OTHER_EXCHANGE"
EXCHANGE_SCOPE_NONE = "OFF_EXCHANGE"


@dataclass
class RegistrantRecord:
    cik: str
    primary_ticker: str
    tickers: list[str]
    name: str
    exchange: str
    exchange_scope: str
    sic: int | None = None
    sic_description: str | None = None
    operating_status: str = "PENDING"
    in_scope: bool = False
    latest_operating_form_date: str | None = None
    known_to_sector_inference: bool = False
    derived: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_cache_path(cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    return Path(cfg.cache_dir) / EXCHANGE_REGISTRY_CACHE_NAME


def load_exchange_registry(
    *,
    http: Any = None,
    cfg: AppConfig | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Load the SEC exchange registry as [{cik, name, ticker, exchange}, ...].

    Cached on disk for 7 days alongside company_tickers.json. ``refresh``
    forces a refetch regardless of cache age.
    """
    cfg = cfg or get_config()
    cache_path = _registry_cache_path(cfg)
    payload: dict[str, Any] | None = None
    if not refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age <= _REGISTRY_TTL_SECONDS:
            try:
                payload = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                payload = None
    if payload is None:
        if http is None:
            from app.util.http import HttpClient

            http = HttpClient(cfg)
        payload = http.get_json(
            EXCHANGE_REGISTRY_URL, use_cache=True, cache_ttl_seconds=_REGISTRY_TTL_SECONDS
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
    fields = [str(item) for item in payload.get("fields") or []]
    rows = payload.get("data") or []
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(fields):
            continue
        record = dict(zip(fields, row, strict=True))
        records.append(
            {
                "cik": str(record.get("cik") or "").strip(),
                "name": str(record.get("name") or "").strip(),
                "ticker": str(record.get("ticker") or "").strip().upper(),
                "exchange": str(record.get("exchange") or "").strip(),
            }
        )
    return [r for r in records if r["cik"] and r["ticker"]]


def _exchange_scope(exchanges: list[str]) -> tuple[str, str]:
    """Classify a CIK's listings; returns (scope, representative_exchange)."""
    normalized = [e.upper() for e in exchanges if e and e.upper() != "NONE"]
    in_scope = [e for e in normalized if e in IN_SCOPE_EXCHANGES]
    if in_scope:
        return EXCHANGE_SCOPE_IN, in_scope[0]
    otc = [e for e in normalized if e in DEFERRED_EXCHANGES]
    if otc:
        return EXCHANGE_SCOPE_OTC, otc[0]
    if normalized:
        return EXCHANGE_SCOPE_OTHER, normalized[0]
    return EXCHANGE_SCOPE_NONE, ""


def select_primary_ticker(listings: list[tuple[str, str]]) -> str:
    """Pick one primary ticker for a CIK from (ticker, exchange) listings.

    Preference: in-scope exchange listings first, then tickers without a
    share-class/preferred suffix separator, then shortest, then registry order.
    """
    if not listings:
        return ""
    indexed = list(enumerate(listings))

    def sort_key(item: tuple[int, tuple[str, str]]) -> tuple[int, int, int, int]:
        order, (ticker, exchange) = item
        on_scope_exchange = 0 if exchange.upper() in IN_SCOPE_EXCHANGES else 1
        has_separator = 1 if ("-" in ticker or "." in ticker) else 0
        return (on_scope_exchange, has_separator, len(ticker), order)

    return sorted(indexed, key=sort_key)[0][1][0]


def _classify_operating_status(
    *,
    primary_ticker: str,
    name: str,
    sic: int | None,
    forms: list[str],
) -> tuple[str, str | None]:
    """Operating-company filter from SEC submissions form history + SIC.

    Returns (operating_status, latest_operating_form_date is resolved by the
    caller). Name-pattern exclusions (fund/ETF/SPAC wording) are deliberately
    NOT applied here — they remain the sector classifier's policy at
    classification time, where they are recorded per ticker instead of
    silently shrinking the census.
    """
    if sic in FUND_TRUST_SIC_CODES:
        return "FUND_OR_TRUST_SIC", f"sic:{sic}"
    if sic == BLANK_CHECK_SIC:
        return "BLANK_CHECK_SIC", f"sic:{sic}"

    from app.sector.classifier import _detect_non_operating_symbol

    symbol_reason = _detect_non_operating_symbol(ticker=primary_ticker)
    if symbol_reason is not None:
        return "NON_OPERATING_SYMBOL", symbol_reason

    has_operating_form = any(
        f.startswith(_ANNUAL_FORM_PREFIX) or f.startswith(_QUARTERLY_FORM_PREFIX)
        for f in forms
    )
    if has_operating_form:
        return OPERATING, None
    has_foreign_annual = any(
        f.startswith(prefix) for f in forms for prefix in _FOREIGN_ANNUAL_PREFIXES
    )
    if has_foreign_annual:
        return "FOREIGN_FILER_ONLY", "forms:20-F/40-F_without_10-K/10-Q"
    return "NO_OPERATING_FORMS", "forms:no_10-K_or_10-Q"


def _submissions_forms_and_dates(submissions: dict[str, Any]) -> tuple[list[str], list[str]]:
    filings = submissions.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    if not isinstance(recent, dict):
        return [], []
    forms = [str(f or "") for f in recent.get("form") or []]
    dates = [str(d or "") for d in recent.get("filingDate") or []]
    return forms, dates


def _latest_operating_form_date(forms: list[str], dates: list[str]) -> str | None:
    best: str | None = None
    for form, filed in zip(forms, dates, strict=False):
        if form.startswith(_ANNUAL_FORM_PREFIX) or form.startswith(_QUARTERLY_FORM_PREFIX):
            if filed and (best is None or filed > best):
                best = filed
    return best


def _default_submissions_loader(
    *,
    cfg: AppConfig,
    fetch_missing: bool,
    max_fetches: int | None,
) -> Callable[[str], dict[str, Any] | None]:
    """Cache-first submissions loader with a fetch budget.

    Returns None when the payload is unavailable (missing from cache and the
    budget/flag disallows fetching, or the fetch failed). Stale cached
    payloads are used as-is — the census needs SIC + form history, both of
    which only change on new filings, and the weekly sync refreshes deltas.
    """
    from app.universe.sector_universe import _submissions_cache_path, load_company_submissions

    state = {"fetches": 0}

    def load(cik: str) -> dict[str, Any] | None:
        cache_path = _submissions_cache_path(cik, cfg=cfg)
        cached_exists = cache_path.exists()
        if not cached_exists:
            if not fetch_missing:
                return None
            if max_fetches is not None and state["fetches"] >= max_fetches:
                return None
            state["fetches"] += 1
        try:
            payload = load_company_submissions(cik, cfg=cfg, refresh_if_missing=False)
        except Exception as exc:  # noqa: BLE001 - census records, never aborts
            logger.warning("census: submissions load failed for CIK %s: %s", cik, exc)
            return None
        return payload if isinstance(payload, dict) and payload else None

    return load


def _classified_tickers(db_path: str | Path) -> set[str]:
    """Tickers with a real inferred sector (NULL-sector attempt rows excluded)."""
    conn = db_connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM sector_inference WHERE inferred_sector IS NOT NULL"
        ).fetchall()
        return {str(r[0]).upper() for r in rows}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def _tickers_with_local_shares(db_path: str | Path) -> set[str]:
    conn = db_connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM companyfacts_facts WHERE line_item = 'shares_outstanding'"
        ).fetchall()
        return {str(r[0]).upper() for r in rows}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def build_registrant_records(
    registry_rows: list[dict[str, Any]],
    *,
    submissions_loader: Callable[[str], dict[str, Any] | None],
    classified_tickers: set[str],
    workers: int = 4,
    progress_every: int = 500,
) -> list[RegistrantRecord]:
    """Group registry rows by CIK and classify each registrant's status."""
    by_cik: dict[str, dict[str, Any]] = {}
    for row in registry_rows:
        cik10 = row["cik"].zfill(10)
        entry = by_cik.setdefault(cik10, {"name": row["name"], "listings": []})
        entry["listings"].append((row["ticker"], row["exchange"]))
        if not entry["name"]:
            entry["name"] = row["name"]

    records: list[RegistrantRecord] = []
    pending: list[RegistrantRecord] = []
    for cik10, entry in sorted(by_cik.items()):
        listings = entry["listings"]
        scope, exchange = _exchange_scope([ex for _, ex in listings])
        primary = select_primary_ticker(listings)
        record = RegistrantRecord(
            cik=cik10,
            primary_ticker=primary,
            tickers=[t for t, _ in listings],
            name=entry["name"],
            exchange=exchange,
            exchange_scope=scope,
        )
        record.known_to_sector_inference = any(
            t.upper() in classified_tickers for t in record.tickers
        )
        if scope == EXCHANGE_SCOPE_IN:
            pending.append(record)
        else:
            record.operating_status = "NOT_EVALUATED_OFF_SCOPE_EXCHANGE"
        records.append(record)

    def evaluate(record: RegistrantRecord) -> None:
        submissions = submissions_loader(record.cik)
        if submissions is None:
            record.operating_status = "SUBMISSIONS_UNAVAILABLE"
            record.derived.append("submissions:unavailable")
            return
        sic_token = str(submissions.get("sic") or "").strip()
        record.sic = int(sic_token) if sic_token.isdigit() else None
        record.sic_description = str(submissions.get("sicDescription") or "").strip() or None
        forms, dates = _submissions_forms_and_dates(submissions)
        status, reason = _classify_operating_status(
            primary_ticker=record.primary_ticker,
            name=record.name,
            sic=record.sic,
            forms=forms,
        )
        record.operating_status = status
        if reason:
            record.derived.append(reason)
        if status == OPERATING:
            record.in_scope = True
            record.latest_operating_form_date = _latest_operating_form_date(forms, dates)

    completed = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(evaluate, record): record for record in pending}
        for future in as_completed(futures):
            future.result()
            completed += 1
            if progress_every and completed % progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "census: %d/%d in-scope registrants evaluated (%.0f/min)",
                    completed,
                    len(pending),
                    rate * 60,
                )
    return records


class CensusRefusedError(RuntimeError):
    """Fail-loud: the census refused to write its update pass.

    Raised when the classification inputs look broken (unreachable
    submissions cache mass-downgrading registrants) or when the run would
    flip an implausible share of the registry in one pass. The DB is left
    untouched; the caller alerts instead of silently rewriting the universe.
    """


# Refuse the update pass when more than this fraction of existing rows would
# change operating_status/in_scope in one run (absolute floor keeps small
# registries from tripping on normal churn).
MASS_FLIP_FRACTION = 0.02
MASS_FLIP_FLOOR = 50
# Refuse when this fraction of evaluated in-scope registrants had no readable
# submissions payload — the signature of an unmounted cache volume.
MAX_UNAVAILABLE_FRACTION = 0.20


def _flip_guard(
    records: list[RegistrantRecord],
    existing: dict[str, dict[str, Any]],
    *,
    allow_mass_update: bool,
) -> None:
    if allow_mass_update or not existing:
        return
    flips = 0
    for record in records:
        prior = existing.get(record.cik)
        if prior is None:
            continue
        if str(prior.get("operating_status")) != record.operating_status or bool(
            prior.get("in_scope")
        ) != bool(record.in_scope):
            flips += 1
    threshold = max(MASS_FLIP_FLOOR, int(MASS_FLIP_FRACTION * len(existing)))
    if flips > threshold:
        raise CensusRefusedError(
            f"census update refused: {flips} of {len(existing)} registrants would "
            f"flip status/scope in one run (threshold {threshold}); inputs look "
            "broken (unmounted cache? registry regression). Re-run with "
            "allow_mass_update=True only after inspecting."
        )


def _upsert_registrants(
    records: list[RegistrantRecord],
    *,
    db_path: str | Path,
    run_kind: str,
    allow_mass_update: bool = False,
) -> dict[str, int]:
    """Idempotent upsert; logs ADDED rows for first-seen CIKs."""
    from app.db import init_db

    now = _utc_now_iso()
    conn = db_connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        init_db(conn=conn)
        existing = {
            str(r["cik"]): dict(r)
            for r in conn.execute(
                "SELECT cik, operating_status, in_scope, removed_at FROM sec_registrants"
            ).fetchall()
        }
        _flip_guard(records, existing, allow_mass_update=allow_mass_update)
        added = updated = reinstated = 0
        for record in records:
            prior = existing.get(record.cik)
            if prior is None:
                conn.execute(
                    """
                    INSERT INTO sec_registrants(
                        cik, primary_ticker, all_tickers, name, exchange, exchange_scope,
                        sic, sic_description, operating_status, in_scope,
                        latest_operating_form_date, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.cik,
                        record.primary_ticker,
                        json.dumps(record.tickers),
                        record.name,
                        record.exchange,
                        record.exchange_scope,
                        record.sic,
                        record.sic_description,
                        record.operating_status,
                        1 if record.in_scope else 0,
                        record.latest_operating_form_date,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    "INSERT INTO universe_sync_log(run_at, run_kind, action, cik, ticker, detail) "
                    "VALUES (?, ?, 'ADDED', ?, ?, ?)",
                    (now, run_kind, record.cik, record.primary_ticker, record.operating_status),
                )
                added += 1
            else:
                if prior.get("removed_at"):
                    conn.execute(
                        "INSERT INTO universe_sync_log(run_at, run_kind, action, cik, ticker, detail) "
                        "VALUES (?, ?, 'REINSTATED', ?, ?, ?)",
                        (now, run_kind, record.cik, record.primary_ticker, record.operating_status),
                    )
                    reinstated += 1
                if bool(prior.get("in_scope")) != bool(record.in_scope):
                    conn.execute(
                        "INSERT INTO universe_sync_log(run_at, run_kind, action, cik, ticker, detail) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            now,
                            run_kind,
                            "SCOPE_ON" if record.in_scope else "SCOPE_OFF",
                            record.cik,
                            record.primary_ticker,
                            f"{prior.get('operating_status')} -> {record.operating_status}",
                        ),
                    )
                conn.execute(
                    """
                    UPDATE sec_registrants SET
                        primary_ticker = ?, all_tickers = ?, name = ?, exchange = ?,
                        exchange_scope = ?, sic = ?, sic_description = ?,
                        operating_status = ?, in_scope = ?,
                        latest_operating_form_date = ?, last_seen_at = ?, removed_at = NULL
                    WHERE cik = ?
                    """,
                    (
                        record.primary_ticker,
                        json.dumps(record.tickers),
                        record.name,
                        record.exchange,
                        record.exchange_scope,
                        record.sic,
                        record.sic_description,
                        record.operating_status,
                        1 if record.in_scope else 0,
                        record.latest_operating_form_date,
                        now,
                        record.cik,
                    ),
                )
                updated += 1
        conn.commit()
        return {"added": added, "updated": updated, "reinstated": reinstated}
    finally:
        conn.close()


def _cap_band_split(
    tickers: list[str],
    *,
    as_of_date: str,
    db_path: str | Path,
) -> dict[str, Any]:
    """Cap-band split through the band-filter fallback chain.

    Names without local companyfacts shares are reported as
    ``pre_ingest_unknown`` without touching the network — the chain cannot
    resolve them until Phase C ingests their facts. Names with local shares
    run the real chain (stale-shares tier + provider price).
    """
    from app.autonomous.cap_resolver import classify_market_cap_for_band_filter

    with_shares = _tickers_with_local_shares(db_path)
    bands: dict[str, int] = {}
    sources: dict[str, int] = {}
    per_ticker: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        if ticker.upper() not in with_shares:
            bands["pre_ingest_unknown"] = bands.get("pre_ingest_unknown", 0) + 1
            per_ticker[ticker] = {"cap_band": None, "cap_source": "pre_ingest_unknown"}
            continue
        classification = classify_market_cap_for_band_filter(
            ticker,
            as_of_date=as_of_date,
            db_path=db_path,
        )
        band = classification.cap_band or "unknown"
        bands[band] = bands.get(band, 0) + 1
        sources[classification.cap_source] = sources.get(classification.cap_source, 0) + 1
        per_ticker[ticker] = {
            "cap_band": classification.cap_band,
            "cap_source": classification.cap_source,
            "market_cap_mm": classification.market_cap_mm,
        }
    return {"band_counts": bands, "cap_source_counts": sources, "per_ticker": per_ticker}


def run_registrant_census(
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    http: Any = None,
    submissions_loader: Callable[[str], dict[str, Any] | None] | None = None,
    fetch_missing: bool = True,
    max_fetches: int | None = None,
    refresh_registry: bool = False,
    cap_bands: bool = True,
    workers: int = 4,
    allow_mass_update: bool = False,
) -> dict[str, Any]:
    """Phase A census: build, persist, and report the true-universe numbers."""
    cfg = cfg or get_config()
    db_path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    asof = str(as_of_date or "").strip() or date.today().isoformat()

    # Fail-loud: never classify against an unreachable submissions cache —
    # with the volume unmounted every loader call returns None and the whole
    # registry mass-downgrades to SUBMISSIONS_UNAVAILABLE.
    if submissions_loader is None:
        cache_dir = Path(cfg.cache_dir)
        if not cache_dir.is_dir():
            raise CensusRefusedError(
                f"census refused: submissions cache unreachable at {cache_dir} "
                "(volume unmounted?)"
            )

    registry_rows = load_exchange_registry(http=http, cfg=cfg, refresh=refresh_registry)
    classified = _classified_tickers(db_path)
    loader = submissions_loader or _default_submissions_loader(
        cfg=cfg, fetch_missing=fetch_missing, max_fetches=max_fetches
    )
    records = build_registrant_records(
        registry_rows,
        submissions_loader=loader,
        classified_tickers=classified,
        workers=workers,
    )

    # Fail-loud: a high unavailable rate means the inputs are broken, not
    # the registrants — refuse the update pass rather than persist downgrades.
    evaluated = [r for r in records if r.exchange_scope == EXCHANGE_SCOPE_IN]
    unavailable = sum(
        1 for r in evaluated if r.operating_status == "SUBMISSIONS_UNAVAILABLE"
    )
    if (
        not allow_mass_update
        and len(evaluated) >= 100
        and unavailable > MAX_UNAVAILABLE_FRACTION * len(evaluated)
    ):
        raise CensusRefusedError(
            f"census update refused: {unavailable}/{len(evaluated)} in-scope "
            f"registrants had unreadable submissions (> {MAX_UNAVAILABLE_FRACTION:.0%}); "
            "submissions cache looks unreachable"
        )

    upsert_counts = _upsert_registrants(
        records, db_path=db_path, run_kind="census", allow_mass_update=allow_mass_update
    )

    exchange_listing_counts: dict[str, int] = {}
    for row in registry_rows:
        key = row["exchange"] or "(none)"
        exchange_listing_counts[key] = exchange_listing_counts.get(key, 0) + 1

    scope_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for record in records:
        scope_counts[record.exchange_scope] = scope_counts.get(record.exchange_scope, 0) + 1
        if record.exchange_scope == EXCHANGE_SCOPE_IN:
            status_counts[record.operating_status] = (
                status_counts.get(record.operating_status, 0) + 1
            )

    operating = [r for r in records if r.in_scope and r.operating_status == OPERATING]
    new_operating = [r for r in operating if not r.known_to_sector_inference]
    new_by_exchange: dict[str, int] = {}
    for record in new_operating:
        new_by_exchange[record.exchange] = new_by_exchange.get(record.exchange, 0) + 1

    report: dict[str, Any] = {
        "as_of_date": asof,
        "generated_at": _utc_now_iso(),
        "registry": {
            "url": EXCHANGE_REGISTRY_URL,
            "listing_rows": len(registry_rows),
            "distinct_ciks": len({r["cik"] for r in registry_rows}),
            "exchange_listing_counts": exchange_listing_counts,
        },
        "exchange_scope_counts": scope_counts,
        "in_scope_exchange_registrants": sum(
            1 for r in records if r.exchange_scope == EXCHANGE_SCOPE_IN
        ),
        "operating_status_counts": status_counts,
        "operating_companies": len(operating),
        "known_to_sector_inference": sum(
            1 for r in operating if r.known_to_sector_inference
        ),
        "new_operating_companies": len(new_operating),
        "new_by_exchange": new_by_exchange,
        "sector_inference_distinct_tickers": len(classified),
        "registrant_table_counts": upsert_counts,
        "new_operating_tickers": sorted(r.primary_ticker for r in new_operating),
    }
    if cap_bands:
        report["new_by_cap_band"] = _cap_band_split(
            [r.primary_ticker for r in new_operating],
            as_of_date=asof,
            db_path=db_path,
        )
    return report


def write_census_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write the census report as JSON + a readable markdown summary."""
    base = Path(output_dir) if output_dir is not None else Path("data/outputs/universe")
    base.mkdir(parents=True, exist_ok=True)
    stamp = str(report.get("as_of_date") or date.today().isoformat()).replace("-", "")
    json_path = base / f"registrant_census_{stamp}.json"
    md_path = base / f"registrant_census_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        f"# SEC Registrant Census — {report.get('as_of_date')}",
        "",
        f"Registry: {report['registry']['listing_rows']} listings / "
        f"{report['registry']['distinct_ciks']} distinct CIKs",
        "",
        "## Exchange listings",
    ]
    for exchange, count in sorted(
        report["registry"]["exchange_listing_counts"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"- {exchange}: {count}")
    lines += ["", "## Registrant scope (by CIK)"]
    for scope, count in sorted(report["exchange_scope_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {scope}: {count}")
    lines += ["", "## Operating filter (in-scope exchanges)"]
    for status, count in sorted(report["operating_status_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {status}: {count}")
    lines += [
        "",
        "## Headline",
        f"- Operating companies on in-scope exchanges: **{report['operating_companies']}**",
        f"- Already classified (sector_inference): {report['known_to_sector_inference']}",
        f"- **New operating companies: {report['new_operating_companies']}**",
        f"- sector_inference distinct tickers (all sources): {report['sector_inference_distinct_tickers']}",
        "",
        "## New operating companies by exchange",
    ]
    for exchange, count in sorted(report.get("new_by_exchange", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"- {exchange}: {count}")
    cap_split = report.get("new_by_cap_band") or {}
    if cap_split:
        lines += ["", "## New operating companies by cap band (fallback chain)"]
        for band, count in sorted(cap_split.get("band_counts", {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"- {band}: {count}")
        lines.append("")
        lines.append(
            "pre_ingest_unknown = no local companyfacts shares yet; "
            "resolves in the post-ingest census."
        )
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path), "md": str(md_path)}
