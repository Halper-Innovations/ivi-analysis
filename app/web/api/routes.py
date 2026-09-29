"""Read-only API routes.

Every endpoint opens ``engine.db`` through the read-only layer and reports a
missing precondition as a structured 503 (``OfflineDetail``) so the SPA can
render the offline state instead of a stack trace.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.web.api.models import (
    BackupsResponse,
    CompanyDecisionsResponse,
    CompanyDossierResponse,
    CompanyResearchResponse,
    CompanyResponse,
    ConsolesResponse,
    CostsResponse,
    CoverageCellDetail,
    CoverageResponse,
    EventsResponse,
    FundamentalsResponse,
    HeartbeatsResponse,
    LogsResponse,
    LogTailResponse,
    MetaResponse,
    OpsHealthResponse,
    OutcomesResponse,
    ReaderArtifactResponse,
    ReaderLibraryResponse,
    RunDetailResponse,
    RunReportResponse,
    RunsIndexStats,
    RunsResponse,
    RunSummary,
    SearchResponse,
    SearchResult,
    SweepsResponse,
    TodayResponse,
    WatchlistResponse,
    WatchlistRow,
)
from app.web.readmodel import company as company_model
from app.web.readmodel import company_depth
from app.web.readmodel import events as events_model
from app.web.readmodel import ops_deck
from app.web.readmodel import outcomes as outcomes_model
from app.web.readmodel import reader as reader_model
from app.web.readmodel import today as today_model
from app.web.readmodel.coverage import coverage_atlas, coverage_cell
from app.web.readmodel.db import OfflineError, readonly_db, resolve_engine_db_path
from app.web.readmodel.run_detail import load_run_detail, load_run_report
from app.web.readmodel.search import search_companies
from app.web.readmodel.runs_index import list_indexed_runs, open_ui_db, refresh_index
from app.web.readmodel.sweeps import sweep_rollups
from app.web.readmodel.watchlist import queue_rows

router = APIRouter(prefix="/api", tags=["ivi"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _offline(exc: OfflineError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"precondition": exc.precondition, "detail": exc.detail},
    )


def _validate_as_of(as_of: str | None) -> None:
    if as_of is None:
        return
    try:
        date.fromisoformat(as_of)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="as_of must be a valid calendar date (YYYY-MM-DD)",
        ) from exc


@router.get("/watchlist", response_model=WatchlistResponse)
def api_watchlist(
    limit: int = Query(default=500, ge=1, le=2000),
    sector: str | None = None,
    band: str | None = None,
    scan_family: str | None = None,
    include_price_suspect: bool = True,
) -> WatchlistResponse:
    try:
        with readonly_db() as conn:
            rows = queue_rows(
                conn,
                limit=limit,
                sector=sector,
                band=band,
                scan_family=scan_family,
                include_price_suspect=include_price_suspect,
            )
    except OfflineError as exc:
        raise _offline(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return WatchlistResponse(
        rows=[WatchlistRow(**row) for row in rows],
        total=len(rows),
        generated_at=_now_iso(),
    )


@router.get("/today", response_model=TodayResponse)
def api_today() -> TodayResponse:
    """The morning-truth deck: health, decisions due, waterline, digest.

    Health comes from :func:`compute_data_health` (full production checks on
    the canonical DB; cheap readability mode under a VOE_DB_PATH override) —
    the same computation the digest and dead-man surfaces trust.
    """
    from app.ops.data_health import compute_data_health

    try:
        with readonly_db() as conn:
            queue = queue_rows(conn, limit=2000)
            decisions = today_model.open_decisions(conn)
            waterline, waterline_deeper = today_model.waterline(conn, queue)
            gate_blocked = today_model.gate_blocked(conn, queue)
    except OfflineError as exc:
        raise _offline(exc) from exc
    health = today_model.serialize_health(compute_data_health())
    return TodayResponse(
        health=health,  # type: ignore[arg-type]
        decisions=decisions,  # type: ignore[arg-type]
        waterline=waterline,  # type: ignore[arg-type]
        waterline_deeper=waterline_deeper,  # type: ignore[arg-type]
        gate_blocked=gate_blocked,  # type: ignore[arg-type]
        digest=today_model.latest_digest(),  # type: ignore[arg-type]
        generated_at=_now_iso(),
    )


@router.get("/search", response_model=SearchResponse)
def api_search(
    q: str = Query(min_length=1, max_length=80),
    limit: int = Query(default=20, ge=1, le=50),
) -> SearchResponse:
    """Ticker/name search over the registrant census + watchlist for ⌘K."""

    try:
        with readonly_db() as conn:
            has_watchlist = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'watchlist'"
            ).fetchone()
            queue = queue_rows(conn, limit=2000) if has_watchlist is not None else []
            status_by_ticker = {
                str(row["ticker"]).upper(): row.get("presented_status") for row in queue
            }
            results = search_companies(
                conn,
                q,
                limit=limit,
                eligible_watchlist_tickers=set(status_by_ticker),
            )
    except OfflineError as exc:
        raise _offline(exc) from exc
    return SearchResponse(
        query=q,
        results=[
            SearchResult(**item, presented_status=status_by_ticker.get(item["ticker"].upper()))
            for item in results
        ],
        generated_at=_now_iso(),
    )


@router.get("/company/{ticker}", response_model=CompanyResponse)
def api_company(
    ticker: str,
    as_of: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> CompanyResponse:
    normalized = ticker.upper()
    _validate_as_of(as_of)
    try:
        with readonly_db() as conn:
            if not company_model.company_exists(conn, normalized):
                raise HTTPException(status_code=404, detail=f"Unknown ticker: {normalized}")
            queue = queue_rows(conn, limit=2000)
            queue_row = next(
                (row for row in queue if str(row["ticker"]).upper() == normalized), None
            )
            try:
                snapshot = company_model.company_snapshot(
                    conn, normalized, queue_row=queue_row, as_of_date=as_of
                )
            except OfflineError as exc:
                if exc.precondition != company_model.MANIFEST_UNUSABLE:
                    raise
                # No audit on this install: answer with the explicit
                # "not audited yet" page, not an offline error. Decision data
                # stays withheld; see unaudited_company_snapshot.
                snapshot = company_model.unaudited_company_snapshot(
                    conn, normalized, as_of_date=as_of
                )
    except OfflineError as exc:
        raise _offline(exc) from exc
    return CompanyResponse(**snapshot)


@router.get("/company/{ticker}/fundamentals", response_model=FundamentalsResponse)
def api_company_fundamentals(
    ticker: str,
    basis: str = Query(default="fy", pattern="^(fy|ttm)$"),
    as_of: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> FundamentalsResponse:
    normalized = ticker.upper()
    if as_of is not None and basis == "ttm":
        raise HTTPException(status_code=422, detail="as_of supports basis=fy only")
    _validate_as_of(as_of)
    try:
        with readonly_db() as conn:
            if not company_model.company_exists(conn, normalized):
                raise HTTPException(status_code=404, detail=f"Unknown ticker: {normalized}")
            payload = (
                company_model.fundamentals_fy(conn, normalized, as_of_date=as_of)
                if basis == "fy"
                else company_model.fundamentals_ttm(conn, normalized)
            )
    except OfflineError as exc:
        raise _offline(exc) from exc
    return FundamentalsResponse(**payload, as_of=as_of)


def _require_known_ticker(conn, normalized: str) -> None:
    if not company_model.company_exists(conn, normalized):
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {normalized}")


@router.get("/company/{ticker}/research", response_model=CompanyResearchResponse)
def api_company_research(ticker: str) -> CompanyResearchResponse:
    normalized = ticker.upper()
    try:
        with readonly_db() as conn:
            _require_known_ticker(conn, normalized)
            payload = company_depth.research(conn, normalized)
    except OfflineError as exc:
        raise _offline(exc) from exc
    return CompanyResearchResponse(**payload, generated_at=_now_iso())


@router.get("/company/{ticker}/dossier", response_model=CompanyDossierResponse)
def api_company_dossier(ticker: str) -> CompanyDossierResponse:
    normalized = ticker.upper()
    try:
        with readonly_db() as conn:
            _require_known_ticker(conn, normalized)
    except OfflineError as exc:
        raise _offline(exc) from exc
    # The dossier is a file artifact; the DB round-trip above only vouches
    # for the ticker so unknown paths stay honest 404s.
    return CompanyDossierResponse(**company_depth.dossier(normalized), generated_at=_now_iso())


@router.get("/company/{ticker}/decisions", response_model=CompanyDecisionsResponse)
def api_company_decisions(ticker: str) -> CompanyDecisionsResponse:
    normalized = ticker.upper()
    try:
        with readonly_db() as conn:
            _require_known_ticker(conn, normalized)
            payload = company_depth.decisions(conn, normalized)
    except OfflineError as exc:
        raise _offline(exc) from exc
    return CompanyDecisionsResponse(**payload, generated_at=_now_iso())


def _run_summary(row: dict[str, object]) -> RunSummary:
    safe_row = dict(row)
    if not bool(safe_row.get("decision_eligible")):
        for field in (
            "decision_status",
            "final_verdict",
            "selected_ticker",
            "no_selection_reason",
            "disposition_counts_json",
        ):
            safe_row[field] = None
    cost_microdollars = row.get("cost_microdollars")
    cost_usd = (
        round(int(cost_microdollars) / 1_000_000, 6) if isinstance(cost_microdollars, int) else None
    )
    return RunSummary(**{**safe_row, "cost_usd": cost_usd})  # type: ignore[arg-type]


@router.get("/runs", response_model=RunsResponse)
def api_runs(
    sector: str | None = None,
    market_cap_focus: str | None = None,
    final_verdict: str | None = None,
    scan_family: str | None = None,
    pipeline_version: str | None = None,
    include_history: bool = False,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> RunsResponse:
    ui_conn = open_ui_db()
    try:
        # Current authorization is never optional. Unknown query parameters
        # such as the retired ``refresh=false`` flag cannot bypass this pass.
        stats = refresh_index(ui_conn)
        rows = list_indexed_runs(
            ui_conn,
            sector=sector,
            market_cap_focus=market_cap_focus,
            final_verdict=final_verdict,
            scan_family=scan_family,
            pipeline_version=pipeline_version,
            decision_eligible=None if include_history else True,
            limit=limit,
            offset=offset,
        )
        total = ui_conn.execute("SELECT COUNT(*) AS n FROM run_index").fetchone()["n"]
    finally:
        ui_conn.close()
    return RunsResponse(
        runs=[_run_summary(row) for row in rows],
        total=int(total),
        index=RunsIndexStats(**stats),
        generated_at=_now_iso(),
    )


def _resolve_detail(ref: str, loader) -> dict | None:
    """Refresh the filesystem index, then resolve a detail from current metadata."""
    ui_conn = open_ui_db()
    try:
        refresh_index(ui_conn)
        return loader(ui_conn, ref)
    finally:
        ui_conn.close()


# Registered before the greedy {ref:path} detail route: "a/b/report" must
# reach the report handler with ref="a/b".
@router.get("/runs/{ref:path}/report", response_model=RunReportResponse)
def api_run_report(ref: str) -> RunReportResponse:
    report = _resolve_detail(ref, load_run_report)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No report for run: {ref}")
    return RunReportResponse(**report, generated_at=_now_iso())


@router.get("/runs/{ref:path}", response_model=RunDetailResponse)
def api_run_detail(ref: str) -> RunDetailResponse:
    def loader(ui_conn, reference):
        try:
            with readonly_db() as engine_conn:
                return load_run_detail(ui_conn, reference, engine_conn=engine_conn)
        except OfflineError:
            # The artifact is the book of record here; a missing engine.db
            # only costs the watchlist links, not the whole page.
            return load_run_detail(ui_conn, reference, engine_conn=None)

    detail = _resolve_detail(ref, loader)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {ref}")
    detail["summary"] = _run_summary(detail["summary"])
    return RunDetailResponse(**detail, generated_at=_now_iso())


@router.get("/sweeps", response_model=SweepsResponse)
def api_sweeps() -> SweepsResponse:
    ui_conn = open_ui_db()
    try:
        refresh_index(ui_conn)
        try:
            with readonly_db() as engine_conn:
                groups = sweep_rollups(ui_conn, engine_conn=engine_conn)
        except OfflineError:
            groups = sweep_rollups(ui_conn, engine_conn=None)
    finally:
        ui_conn.close()
    return SweepsResponse(groups=groups, generated_at=_now_iso())  # type: ignore[arg-type]


@router.get("/coverage", response_model=CoverageResponse)
def api_coverage() -> CoverageResponse:
    try:
        with readonly_db() as conn:
            atlas = coverage_atlas(conn)
    except OfflineError as exc:
        raise _offline(exc) from exc
    return CoverageResponse(**atlas, generated_at=_now_iso())


@router.get("/coverage/cell", response_model=CoverageCellDetail)
def api_coverage_cell(sector: str, band: str) -> CoverageCellDetail:
    ui_conn = open_ui_db()
    try:
        refresh_index(ui_conn)  # runs listed in the cell link out by slug
        try:
            with readonly_db() as conn:
                cell = coverage_cell(conn, sector=sector, band=band, ui_conn=ui_conn)
        except OfflineError as exc:
            raise _offline(exc) from exc
    finally:
        ui_conn.close()
    return CoverageCellDetail(**cell, generated_at=_now_iso())


@router.get("/events", response_model=EventsResponse)
def api_events(
    lane: str | None = Query(default=None, pattern="^(opportunity|queue_protection)$"),
    event_type: str | None = Query(default=None, alias="type"),
    q: str | None = None,
    per_column: int = Query(default=40, ge=1, le=200),
) -> EventsResponse:
    try:
        with readonly_db() as conn:
            deck = events_model.events_deck(
                conn, lane=lane, event_type=event_type, q=q, per_column=per_column
            )
            strip = events_model.scan_strip(conn)
    except OfflineError as exc:
        raise _offline(exc) from exc
    return EventsResponse(**deck, scan_strip=strip, generated_at=_now_iso())


@router.get("/outcomes", response_model=OutcomesResponse)
def api_outcomes() -> OutcomesResponse:
    try:
        with readonly_db() as conn:
            deck = outcomes_model.outcomes_deck(conn)
    except OfflineError as exc:
        raise _offline(exc) from exc
    return OutcomesResponse(**deck, generated_at=_now_iso())


@router.get("/ops/health", response_model=OpsHealthResponse)
def api_ops_health() -> OpsHealthResponse:
    """The deadman wall: compute_data_health serialized verbatim."""
    from app.ops.data_health import compute_data_health

    health = today_model.serialize_health(compute_data_health())
    return OpsHealthResponse(health=health, generated_at=_now_iso())  # type: ignore[arg-type]


@router.get("/ops/heartbeats", response_model=HeartbeatsResponse)
def api_ops_heartbeats(days: int = Query(default=14, ge=1, le=60)) -> HeartbeatsResponse:
    try:
        with readonly_db() as conn:
            ledger = ops_deck.heartbeat_ledger(conn, days=days)
    except OfflineError as exc:
        raise _offline(exc) from exc
    return HeartbeatsResponse(**ledger, generated_at=_now_iso())


@router.get("/ops/backups", response_model=BackupsResponse)
def api_ops_backups() -> BackupsResponse:
    return BackupsResponse(**ops_deck.backups(), generated_at=_now_iso())


@router.get("/ops/costs", response_model=CostsResponse)
def api_ops_costs() -> CostsResponse:
    ui_conn = open_ui_db()
    try:
        refresh_index(ui_conn)
        try:
            with readonly_db() as engine_conn:
                ledger = ops_deck.cost_ledger(ui_conn, engine_conn)
        except OfflineError:
            # Run-artifact costs are still real without the books of record.
            ledger = ops_deck.cost_ledger(ui_conn, None)
    finally:
        ui_conn.close()
    return CostsResponse(**ledger, generated_at=_now_iso())


@router.get("/ops/logs", response_model=LogsResponse)
def api_ops_logs() -> LogsResponse:
    return LogsResponse(files=ops_deck.list_cron_logs(), generated_at=_now_iso())  # type: ignore[arg-type]


@router.get("/ops/logs/{name}", response_model=LogTailResponse)
def api_ops_log_tail(
    name: str,
    tail: int = Query(default=ops_deck.DEFAULT_TAIL_LINES, ge=1, le=ops_deck.MAX_TAIL_LINES),
) -> LogTailResponse:
    try:
        payload = ops_deck.read_log_tail(name, lines=tail)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No such log: {name}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LogTailResponse(**payload, generated_at=_now_iso())


@router.get("/ops/consoles", response_model=ConsolesResponse)
def api_ops_consoles() -> ConsolesResponse:
    try:
        with readonly_db() as conn:
            payload = ops_deck.consoles(conn)
    except OfflineError as exc:
        raise _offline(exc) from exc
    return ConsolesResponse(**payload, generated_at=_now_iso())


@router.get("/reader/library", response_model=ReaderLibraryResponse)
def api_reader_library() -> ReaderLibraryResponse:
    ui_conn = open_ui_db()
    try:
        refresh_index(ui_conn)
        payload = reader_model.library(ui_conn)
    finally:
        ui_conn.close()
    return ReaderLibraryResponse(**payload, generated_at=_now_iso())


@router.get("/reader/artifact", response_model=ReaderArtifactResponse)
def api_reader_artifact(path: str) -> ReaderArtifactResponse:
    try:
        payload = reader_model.render_artifact(path)
    except reader_model.ArtifactRefused as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ReaderArtifactResponse(**payload, generated_at=_now_iso())


@router.get("/meta", response_model=MetaResponse)
def api_meta() -> MetaResponse:
    engine_path = resolve_engine_db_path()
    watchlist_rows: int | None = None
    engine_present = engine_path.exists()
    if engine_present:
        try:
            with readonly_db() as conn:
                watchlist_rows = int(
                    conn.execute(
                        "SELECT COUNT(*) AS n FROM watchlist WHERE status != 'REMOVED'"
                    ).fetchone()["n"]
                )
        except sqlite3.OperationalError:
            # An engine.db without a watchlist table (fresh init) is still
            # online; anything else should surface loudly.
            watchlist_rows = None
    ui_conn = open_ui_db()
    try:
        refresh_index(ui_conn)
        runs_indexed = int(ui_conn.execute("SELECT COUNT(*) AS n FROM run_index").fetchone()["n"])
    finally:
        ui_conn.close()
    return MetaResponse(
        app="ivi",
        engine_db_path=str(engine_path),
        engine_db_present=engine_present,
        runs_indexed=runs_indexed,
        watchlist_rows=watchlist_rows,
        generated_at=_now_iso(),
    )


