"""Phase B/C intake for census-discovered registrants.

Classification (Phase B) is deterministic: SIC from cached EDGAR submissions
mapped through data/universe/sector_sic_ranges.json via the existing sector
classifier — no LLM. Names whose SIC cannot be mapped land on an
UNCLASSIFIED_REVIEW list with counts instead of being silently dropped.

Ingest (Phase C) pulls companyfacts through the existing rate-limited
SecClient + TAG_MAP facts writer, runs the deterministic valuation pipeline
(a name is only sweepable once it has a scorecard row), and applies the
structural exclusion gate at intake so shells never reach review or LLM
spend. Failures are reported by class, never skipped silently. Both phases
are resumable and idempotent.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.db import connect as db_connect
from app.config import AppConfig, get_config
from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)

UNCLASSIFIED_STATUSES = frozenset({"no_sic", "no_sector_match", "no_cik", "fetch_error"})

# Ingest failure classes (reported, never silently skipped).
INGEST_OK = "INGESTED"
INGEST_NO_CIK = "NO_CIK"
INGEST_NO_COMPANYFACTS = "NO_COMPANYFACTS"
INGEST_EMPTY_FACTS = "EMPTY_FACTS"
INGEST_VALUATION_FAILED = "VALUATION_FAILED"
INGEST_FETCH_ERROR = "FETCH_ERROR"
QUARANTINE_PREFIX = "QUARANTINE_STRUCTURAL"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = db_connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _operating_registrants(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT cik, primary_ticker, all_tickers, name, exchange, sic, sic_description,
               sector, classification_status, intake_status
        FROM sec_registrants
        WHERE in_scope = 1 AND operating_status = 'OPERATING' AND removed_at IS NULL
        ORDER BY primary_ticker
        """
    ).fetchall()
    return [dict(r) for r in rows]


def _classified_tickers(conn: sqlite3.Connection) -> set[str]:
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM sector_inference WHERE inferred_sector IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    return {str(r[0]).upper() for r in rows}


# ---------------------------------------------------------------------------
# Phase B — deterministic classification
# ---------------------------------------------------------------------------


def classify_new_registrants(
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    submissions_loader: Callable[[str], dict[str, Any] | None] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Classify operating registrants that have no real sector yet.

    Reuses the existing SIC classifier end to end (sector_sic_ranges.json,
    specificity-on-collision, deterministic name fallbacks) and records the
    outcome per registrant. Returns the classification report including the
    UNCLASSIFIED_REVIEW list.
    """
    from app.sector.classifier import classify_all

    cfg = cfg or get_config()
    db_path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    asof = str(as_of_date or "").strip() or date.today().isoformat()

    conn = _connect(db_path)
    try:
        registrants = _operating_registrants(conn)
        classified = _classified_tickers(conn)
    finally:
        conn.close()

    todo: list[dict[str, Any]] = []
    for record in registrants:
        tickers = [str(t).upper() for t in json.loads(record["all_tickers"] or "[]")]
        tickers.append(str(record["primary_ticker"]).upper())
        if any(t in classified for t in tickers):
            continue
        todo.append(record)
    if limit is not None:
        todo = todo[: int(limit)]

    summary = classify_all(
        db_path=db_path,
        as_of_date=asof,
        tickers=[r["primary_ticker"] for r in todo],
        submissions_loader=submissions_loader,
        cfg=cfg,
    )

    # Pull the per-ticker outcomes back out of sector_inference for this date.
    conn = _connect(db_path)
    try:
        outcome_rows = conn.execute(
            "SELECT ticker, inferred_sector, derived_from FROM sector_inference WHERE as_of_date = ?",
            (asof,),
        ).fetchall()
        outcomes = {str(r["ticker"]).upper(): dict(r) for r in outcome_rows}

        now = _utc_now_iso()
        unclassified_review: list[dict[str, Any]] = []
        sector_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        for record in todo:
            ticker = str(record["primary_ticker"]).upper()
            outcome = outcomes.get(ticker)
            sector = outcome.get("inferred_sector") if outcome else None
            derived = str(outcome.get("derived_from") or "") if outcome else ""
            if sector:
                status = "classified"
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            elif "exclude_non_operating" in derived:
                status = "excluded_non_operating"
            elif "no_sector_match" in derived:
                status = "no_sector_match"
            elif "no_cik" in derived or outcome is None:
                status = "no_cik"
            elif "no_sic" in derived:
                status = "no_sic"
            else:
                status = "fetch_error"
            status_counts[status] = status_counts.get(status, 0) + 1
            conn.execute(
                "UPDATE sec_registrants SET sector = ?, classification_status = ? WHERE cik = ?",
                (sector, status, record["cik"]),
            )
            conn.execute(
                "INSERT INTO universe_sync_log(run_at, run_kind, action, cik, ticker, detail) "
                "VALUES (?, 'classify', 'CLASSIFIED', ?, ?, ?)",
                (now, record["cik"], ticker, f"{status}:{sector or ''}"),
            )
            if status in UNCLASSIFIED_STATUSES:
                unclassified_review.append(
                    {
                        "ticker": ticker,
                        "cik": record["cik"],
                        "name": record["name"],
                        "sic": record["sic"],
                        "sic_description": record["sic_description"],
                        "status": status,
                    }
                )
        conn.commit()
    finally:
        conn.close()

    unmapped_sic_counts: dict[str, int] = {}
    for item in unclassified_review:
        if item["status"] == "no_sector_match" and item["sic"] is not None:
            key = f"{item['sic']} {item['sic_description'] or ''}".strip()
            unmapped_sic_counts[key] = unmapped_sic_counts.get(key, 0) + 1

    return {
        "as_of_date": asof,
        "generated_at": _utc_now_iso(),
        "scope": len(todo),
        "already_classified_skipped": len(registrants) - len(todo),
        "status_counts": status_counts,
        "sector_counts": dict(sorted(sector_counts.items(), key=lambda kv: -kv[1])),
        "unclassified_review_count": len(unclassified_review),
        "unclassified_review": sorted(unclassified_review, key=lambda r: r["ticker"]),
        "unmapped_sic_counts": dict(sorted(unmapped_sic_counts.items(), key=lambda kv: -kv[1])),
        "classifier_summary": {
            "total": summary.total,
            "classified": summary.classified,
            "excluded_non_operating": summary.excluded_non_operating,
            "no_sic": summary.no_sic,
            "no_sector_match": summary.no_sector_match,
            "fetch_error": summary.fetch_error,
            "skipped_existing": summary.skipped_existing,
        },
    }


# ---------------------------------------------------------------------------
# Phase C — companyfacts ingest + valuation + structural gate at intake
# ---------------------------------------------------------------------------


def _ingest_scope(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Classified operating registrants that are not yet sweepable.

    Sweepable means a scorecard valuation row exists — the sector candidate
    loader joins on it, so facts without a scorecard never reach a sweep.
    """
    rows = conn.execute(
        """
        SELECT r.cik, r.primary_ticker, r.name, r.sector, r.intake_status
        FROM sec_registrants r
        WHERE r.in_scope = 1 AND r.operating_status = 'OPERATING' AND r.removed_at IS NULL
          AND EXISTS (
              SELECT 1 FROM sector_inference si
              WHERE si.ticker = r.primary_ticker AND si.inferred_sector IS NOT NULL
          )
        ORDER BY r.primary_ticker
        """
    ).fetchall()
    return [dict(row) for row in rows if not _has_scorecard(conn, str(row["primary_ticker"]))]


def _has_annual_facts(conn: sqlite3.Connection, ticker: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM companyfacts_facts WHERE ticker = ? AND period_type = 'FY' LIMIT 1",
        (ticker,),
    ).fetchone()
    return row is not None


def _has_scorecard(conn: sqlite3.Connection, ticker: str) -> bool:
    return (
        latest_decision_eligible_valuation_row(
            conn,
            ticker=ticker,
            method="scorecard",
        )
        is not None
    )


def _classify_ingest_exception(exc: Exception) -> str:
    from app.ingest.cik_registry import CIKNotFoundError
    from app.util.http import DomainBudgetExceeded

    if isinstance(exc, CIKNotFoundError):
        return INGEST_NO_CIK
    if isinstance(exc, DomainBudgetExceeded):
        return "BUDGET_EXHAUSTED"
    if "404" in str(exc):
        return INGEST_NO_COMPANYFACTS
    return INGEST_FETCH_ERROR


def ingest_new_registrants(
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    limit: int | None = None,
    run_gate: bool = True,
    run_valuation: bool = True,
    facts_fn: Callable[[str], None] | None = None,
    valuation_fn: Callable[[str, str], None] | None = None,
    gate_fn: Callable[..., Any] | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Phase C: facts + deterministic valuation + structural gate at intake.

    Resumable: scope is "classified but not yet sweepable", so completed
    names drop out on the next run. Failures are classified, recorded on
    sec_registrants.intake_status, and reported — never silently skipped.
    """
    cfg = cfg or get_config()
    db_path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    asof = str(as_of_date or "").strip() or date.today().isoformat()

    conn = _connect(db_path)
    try:
        scope = _ingest_scope(conn)
    finally:
        conn.close()
    if limit is not None:
        scope = scope[: int(limit)]

    if facts_fn is None:
        from app.ingest.facts_writer import ensure_all_facts

        facts_fn = ensure_all_facts
    if valuation_fn is None:
        from app.valuation.valuation_writer import ensure_valuation

        valuation_fn = ensure_valuation
    if gate_fn is None:
        from app.autonomous.structural_gate import evaluate_structural_gate

        gate_fn = evaluate_structural_gate

    price_lookup = None
    if run_gate:
        from app.autonomous.cap_resolver import default_price_lookup

        price_lookup = default_price_lookup(cfg)

    results: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    gate_code_counts: dict[str, int] = {}
    budget_exhausted = False
    started = time.perf_counter()

    for index, record in enumerate(scope, start=1):
        ticker = str(record["primary_ticker"]).upper()
        outcome: dict[str, Any] = {
            "ticker": ticker,
            "cik": record["cik"],
            "sector": record["sector"],
        }

        if budget_exhausted:
            outcome["intake_status"] = "PENDING:BUDGET_EXHAUSTED"
            results.append(outcome)
            continue

        # 1. companyfacts via the rate-limited SecClient path (cache-first).
        try:
            facts_fn(ticker)
            fact_error = None
        except Exception as exc:  # noqa: BLE001 - classified, reported, resumable
            fact_error = _classify_ingest_exception(exc)
            outcome["fact_error_detail"] = f"{type(exc).__name__}: {exc}"
            if fact_error == "BUDGET_EXHAUSTED":
                budget_exhausted = True

        conn = _connect(db_path)
        try:
            has_facts = _has_annual_facts(conn, ticker)
        finally:
            conn.close()

        if fact_error is not None and not has_facts:
            intake_status = f"INGEST_FAILED:{fact_error}"
        elif not has_facts:
            intake_status = f"INGEST_FAILED:{INGEST_EMPTY_FACTS}"
        else:
            intake_status = INGEST_OK

        # 2. deterministic valuation (sweepability requires a scorecard row).
        valued = False
        if intake_status == INGEST_OK and run_valuation:
            valuation_fn(ticker, asof)
            conn = _connect(db_path)
            try:
                valued = _has_scorecard(conn, ticker)
            finally:
                conn.close()
            if not valued:
                intake_status = f"INGEST_FAILED:{INGEST_VALUATION_FAILED}"
        outcome["valued"] = valued

        # 3. structural exclusion gate at intake: shells never reach review.
        if intake_status == INGEST_OK and run_gate:
            cap_price = None
            cap_mm = None
            try:
                from app.autonomous.cap_resolver import classify_market_cap_for_band_filter

                classification = classify_market_cap_for_band_filter(
                    ticker, as_of_date=asof, db_path=db_path, price_lookup=price_lookup, cfg=cfg
                )
                cap_price = classification.price_used
                cap_mm = classification.market_cap_mm
                outcome["cap_band"] = classification.cap_band
                outcome["cap_source"] = classification.cap_source
            except Exception:  # noqa: BLE001 - gate must run even without cap inputs
                pass
            gate_result = gate_fn(
                ticker,
                as_of_date=asof,
                price=cap_price,
                market_cap_mm=cap_mm,
                db_path=db_path,
                cfg=cfg,
            )
            if getattr(gate_result, "excluded_error", False):
                # Fail-closed: the gate could not examine this name (engine
                # DB missing/locked). Never intake it as clean; the failure
                # code lands in the report so the run reads degraded.
                intake_status = f"EXCLUDED_ERROR:{gate_result.degraded_codes[0]}"
                outcome["gate_degraded_codes"] = list(gate_result.degraded_codes)
                for code in gate_result.degraded_codes:
                    gate_code_counts[f"EXCLUDED_ERROR:{code}"] = (
                        gate_code_counts.get(f"EXCLUDED_ERROR:{code}", 0) + 1
                    )
            elif gate_result.triggered_codes:
                intake_status = f"{QUARANTINE_PREFIX}:{gate_result.triggered_codes[0]}"
                outcome["gate_codes"] = list(gate_result.triggered_codes)
                outcome["gate_details"] = dict(gate_result.details)
                for code in gate_result.triggered_codes:
                    gate_code_counts[code] = gate_code_counts.get(code, 0) + 1

        outcome["intake_status"] = intake_status
        status_counts[intake_status] = status_counts.get(intake_status, 0) + 1
        results.append(outcome)

        now = _utc_now_iso()
        action = (
            "QUARANTINED"
            if intake_status.startswith(QUARANTINE_PREFIX)
            else ("INGESTED" if intake_status == INGEST_OK else "FAILED")
        )
        conn = _connect(db_path)
        try:
            conn.execute(
                "UPDATE sec_registrants SET intake_status = ?, intake_detail = ? WHERE cik = ?",
                (
                    intake_status,
                    json.dumps({k: v for k, v in outcome.items() if k not in {"ticker", "cik"}}),
                    record["cik"],
                ),
            )
            conn.execute(
                "INSERT INTO universe_sync_log(run_at, run_kind, action, cik, ticker, detail) "
                "VALUES (?, 'ingest', ?, ?, ?, ?)",
                (now, action, record["cik"], ticker, intake_status),
            )
            conn.commit()
        finally:
            conn.close()

        if progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed > 0 else 0.0
            logger.info(
                "ingest: %d/%d (%.1f/min) — last %s -> %s",
                index,
                len(scope),
                rate * 60,
                ticker,
                intake_status,
            )

    ingested = status_counts.get(INGEST_OK, 0)
    quarantined = sum(
        count for status, count in status_counts.items() if status.startswith(QUARANTINE_PREFIX)
    )
    return {
        "as_of_date": asof,
        "generated_at": _utc_now_iso(),
        "scope": len(scope),
        "ingested": ingested,
        "quarantined": quarantined,
        "failed": len(scope) - ingested - quarantined,
        "budget_exhausted": budget_exhausted,
        "status_counts": dict(sorted(status_counts.items(), key=lambda kv: -kv[1])),
        "gate_code_counts": gate_code_counts,
        "failures": [
            {
                "ticker": r["ticker"],
                "intake_status": r.get("intake_status"),
                "detail": r.get("fact_error_detail"),
            }
            for r in results
            if str(r.get("intake_status") or "").startswith("INGEST_FAILED")
        ],
        "results": results,
    }


def write_ingest_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    base = Path(output_dir) if output_dir is not None else Path("data/outputs/universe")
    base.mkdir(parents=True, exist_ok=True)
    stamp = str(report.get("as_of_date") or date.today().isoformat()).replace("-", "")
    json_path = base / f"ingest_{stamp}.json"
    md_path = base / f"ingest_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        f"# New-Registrant Ingest — {report.get('as_of_date')}",
        "",
        f"Scope: {report['scope']} classified-but-unsweepable registrants",
        "",
        f"- INGESTED (facts + scorecard + gate pass): **{report['ingested']}**",
        f"- QUARANTINE_STRUCTURAL at intake: {report['quarantined']}",
        f"- failed: {report['failed']}",
        "",
        "## Outcomes by class",
    ]
    for status, count in report["status_counts"].items():
        lines.append(f"- {status}: {count}")
    if report["gate_code_counts"]:
        lines += ["", "## Structural gate codes"]
        for code, count in sorted(report["gate_code_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {code}: {count}")
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path), "md": str(md_path)}


# ---------------------------------------------------------------------------
# Phase E — weekly universe sync (census + classify + ingest on the delta)
# ---------------------------------------------------------------------------

# Removal sanity floor: refuse to mark removals when more than this
# fraction of active registrants would leave in one sync (absolute floor for
# small registries) — a truncated registry download must not mass-delist.
MASS_REMOVAL_FRACTION = 0.02
MASS_REMOVAL_FLOOR = 20


class UniverseSyncRefusedError(RuntimeError):
    """Raised when a sync would remove an implausible share of the registry."""


def _mark_removed_registrants(
    *,
    registry_ciks: set[str],
    db_path: str | Path,
    allow_mass_removal: bool = False,
) -> list[dict[str, str]]:
    """Mark registrants that left the SEC registry; never delete rows."""
    now = _utc_now_iso()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT cik, primary_ticker FROM sec_registrants WHERE removed_at IS NULL"
        ).fetchall()
        leaving = [row for row in rows if str(row["cik"]) not in registry_ciks]
        threshold = max(MASS_REMOVAL_FLOOR, int(MASS_REMOVAL_FRACTION * len(rows)))
        if leaving and not allow_mass_removal and len(leaving) > threshold:
            raise UniverseSyncRefusedError(
                f"universe sync refused: {len(leaving)} of {len(rows)} active "
                f"registrants would be marked removed in one run (threshold "
                f"{threshold}); registry input looks truncated. Re-run with "
                "allow_mass_removal only after inspecting."
            )
        removed: list[dict[str, str]] = []
        for row in rows:
            if str(row["cik"]) in registry_ciks:
                continue
            conn.execute(
                "UPDATE sec_registrants SET removed_at = ? WHERE cik = ?",
                (now, row["cik"]),
            )
            conn.execute(
                "INSERT INTO universe_sync_log(run_at, run_kind, action, cik, ticker, detail) "
                "VALUES (?, 'sync', 'REMOVED', ?, ?, 'left SEC exchange registry')",
                (now, row["cik"], row["primary_ticker"]),
            )
            removed.append({"cik": str(row["cik"]), "ticker": str(row["primary_ticker"])})
        conn.commit()
        return removed
    finally:
        conn.close()


def _propagate_removals(
    removed: list[dict[str, str]],
    *,
    db_path: str | Path,
) -> dict[str, int]:
    """A sync-marked registry removal reaches every live surface.

    - Watchlist: live rows transition to terminal REMOVED via the existing
      soft-remove path (mark_status writes watchlist_history, retires all of
      the ticker's live rows) — the name stops quoting triggers.
    - Filing watch: the name is retired from the filing-watch set so polls and
      reprices stop treating it as a permanent NO_PRICE failure.
    Best-effort per name; one bad row never blocks the rest of the sync.
    """
    counts = {"watchlist_removed": 0}
    if not removed:
        return counts

    from app.watchlist.store import get_latest, mark_status

    for item in removed:
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        reason = "UNIVERSE_EXIT:left SEC exchange registry"
        try:
            entry = get_latest(ticker, db_path=db_path)
            if entry is not None and str(entry.status).upper() != "REMOVED":
                mark_status(
                    ticker,
                    "REMOVED",
                    reason,
                    source="universe_sync",
                    db_path=db_path,
                )
                counts["watchlist_removed"] += 1
        except Exception:  # noqa: BLE001 - propagate to the next name
            logger.warning("removal propagation (watchlist) failed for %s", ticker, exc_info=True)

    return counts


def sync_universe(
    *,
    as_of_date: str | None = None,
    db_path: str | Path | None = None,
    cfg: AppConfig | None = None,
    refresh_registry: bool = True,
    max_ingest: int | None = None,
    allow_mass_removal: bool = False,
    submissions_loader: Callable[[str], dict[str, Any] | None] | None = None,
    census_kwargs: dict[str, Any] | None = None,
    classify_fn: Callable[..., dict[str, Any]] | None = None,
    ingest_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Weekly registrant sync: diff -> classify -> ingest -> gate the delta.

    Idempotent and cheap: the census reuses the 7-day registry cache unless
    asked to refresh, submissions are fetched only for never-seen CIKs, and
    classification/ingest scopes are deltas by construction. Additions,
    removals, and intake outcomes all land in universe_sync_log.
    """
    from app.universe.registrant_census import load_exchange_registry, run_registrant_census

    cfg = cfg or get_config()
    db_path = Path(db_path) if db_path is not None else Path(cfg.db_path)
    asof = str(as_of_date or "").strip() or date.today().isoformat()

    census_args: dict[str, Any] = {
        "as_of_date": asof,
        "db_path": db_path,
        "cfg": cfg,
        "refresh_registry": refresh_registry,
        "cap_bands": False,
    }
    if submissions_loader is not None:
        census_args["submissions_loader"] = submissions_loader
    census_args.update(census_kwargs or {})
    census = run_registrant_census(**census_args)
    effective_loader = census_args.get("submissions_loader")

    registry_rows = load_exchange_registry(cfg=cfg, refresh=False, http=census_args.get("http"))
    registry_ciks = {str(row["cik"]).zfill(10) for row in registry_rows}
    removed = _mark_removed_registrants(
        registry_ciks=registry_ciks, db_path=db_path, allow_mass_removal=allow_mass_removal
    )
    removal_propagation = _propagate_removals(removed, db_path=db_path)

    classify = (classify_fn or classify_new_registrants)(
        as_of_date=asof, db_path=db_path, cfg=cfg, submissions_loader=effective_loader
    )
    ingest = (ingest_fn or ingest_new_registrants)(
        as_of_date=asof, db_path=db_path, cfg=cfg, limit=max_ingest
    )

    return {
        "as_of_date": asof,
        "generated_at": _utc_now_iso(),
        "registrants_added": census["registrant_table_counts"]["added"],
        "registrants_reinstated": census["registrant_table_counts"]["reinstated"],
        "registrants_removed": len(removed),
        "removed": removed,
        "removal_propagation": removal_propagation,
        "new_operating_companies": census["new_operating_companies"],
        "classified": classify["status_counts"].get("classified", 0),
        "unclassified_review_count": classify["unclassified_review_count"],
        "ingested": ingest["ingested"],
        "quarantined": ingest["quarantined"],
        "ingest_failed": ingest["failed"],
        "census": {
            key: census[key]
            for key in (
                "operating_companies",
                "known_to_sector_inference",
                "new_operating_companies",
            )
        },
    }


def write_sync_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    base = Path(output_dir) if output_dir is not None else Path("data/outputs/universe")
    base.mkdir(parents=True, exist_ok=True)
    stamp = str(report.get("as_of_date") or date.today().isoformat()).replace("-", "")
    json_path = base / f"sync_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_path = base / f"sync_{stamp}.md"
    md_path.write_text(
        "\n".join(
            [
                f"# Universe Sync — {report.get('as_of_date')}",
                "",
                f"- registrants added: {report['registrants_added']} "
                f"(reinstated: {report['registrants_reinstated']})",
                f"- registrants removed: {report['registrants_removed']}",
                f"- new operating companies: {report['new_operating_companies']}",
                f"- classified: {report['classified']} "
                f"(review list: {report['unclassified_review_count']})",
                f"- ingested: {report['ingested']} / quarantined: {report['quarantined']} "
                f"/ failed: {report['ingest_failed']}",
            ]
        )
        + "\n"
    )
    return {"json": str(json_path), "md": str(md_path)}


def write_classification_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    base = Path(output_dir) if output_dir is not None else Path("data/outputs/universe")
    base.mkdir(parents=True, exist_ok=True)
    stamp = str(report.get("as_of_date") or date.today().isoformat()).replace("-", "")
    json_path = base / f"classification_{stamp}.json"
    md_path = base / f"classification_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        f"# New-Registrant Classification — {report.get('as_of_date')}",
        "",
        f"Scope: {report['scope']} unclassified operating registrants "
        f"({report['already_classified_skipped']} already classified, skipped)",
        "",
        "## Outcomes",
    ]
    for status, count in sorted(report["status_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {status}: {count}")
    lines += ["", "## Classified by sector"]
    for sector, count in report["sector_counts"].items():
        lines.append(f"- {sector}: {count}")
    lines += [
        "",
        f"## UNCLASSIFIED_REVIEW ({report['unclassified_review_count']})",
        "",
        "Unmappable-SIC counts (extend sector_sic_ranges.json or accept):",
    ]
    for sic, count in report["unmapped_sic_counts"].items():
        lines.append(f"- SIC {sic}: {count}")
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path), "md": str(md_path)}
