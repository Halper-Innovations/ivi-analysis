"""Four-stage sweep orchestrator for ivi discover.

Runs Stage 2 classifier -> Stage 3 researcher (on KEEPs) -> Stage 4 deep
loop (on Stage 3 WATCH / BUY_CANDIDATE), writing all results to the
session DB. Each stage is dependency-injected so the orchestrator can
be unit-tested without real API calls.

Fast-path performance note: by default the sweep wires a bundle_builder
that closes over the universe's already-loaded scorecards and calls
``build_analysis_evidence_bundle_from_cached_scorecard``. This skips the
slow ``ensure_valuation`` recomputation on every Stage 3 / Stage 4 call.
Tests and callers with their own scorecard lookup can override
``bundle_builder`` explicitly.
"""

from __future__ import annotations

import copy
import json as _json
import logging
import sqlite3 as _sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.autonomous.v1_financial_context import (
    build_canonical_v1_financial_context,
)
from app.discover.persistence import (
    create_sweep,
    ensure_schema,
    finalize_sweep,
    get_sweep,
    get_sweep_universe,
    insert_sweep_universe,
    list_stage2_keeps,
    list_stage2_tickers,
    list_stage3_survivors,
    update_sweep_budget,
    update_sweep_status,
)
from app.discover.stage2 import Stage2Config, run_stage2
from app.discover.stage3 import Stage3Config, run_stage3
from app.discover.stage4 import Stage4Config, run_stage4

logger = logging.getLogger(__name__)


def _reject_unauthorized_scorecard(*, ticker: str, as_of_date: str) -> None:
    from app.autonomous.financial_integrity import FinancialIntegrityScope
    from app.autonomous.financial_integrity import (
        require_financial_integrity_scope,
    )

    require_financial_integrity_scope(
        FinancialIntegrityScope(
            context=f"discover_unauthorized_scorecard:{ticker}:{as_of_date}",
            run_as_of_date=as_of_date,
        )
    )


def _select_authorized_scorecard_row(
    conn: _sqlite3.Connection,
    *,
    ticker: str,
    as_of_date: str | None = None,
    exact_as_of_date: bool = False,
) -> Any | None:
    from app.valuation.lineage import latest_decision_eligible_valuation_row

    try:
        return latest_decision_eligible_valuation_row(
            conn,
            ticker=ticker,
            method="scorecard",
            as_of_date=as_of_date,
            exact_as_of_date=exact_as_of_date,
        )
    except _sqlite3.DatabaseError:
        # Legacy schemas without the exact-lineage columns are unaudited.
        return None


def _authorized_scorecard_input(
    row: Any,
    *,
    engine_db_path: str | Path,
) -> tuple[str, str, dict[str, Any], Any]:
    from app.valuation.lineage import valuation_row_is_decision_eligible

    ticker = str(row["ticker"]).strip().upper()
    as_of_date = str(row["as_of_date"]).strip()[:10]
    if not valuation_row_is_decision_eligible(row):
        _reject_unauthorized_scorecard(
            ticker=ticker,
            as_of_date=as_of_date,
        )
    scorecard = _json.loads(row["outputs_json"] or "{}")
    context = build_canonical_v1_financial_context(
        tickers=[ticker],
        as_of_date=as_of_date,
        db_path=engine_db_path,
        scorecard_evidence={ticker: (as_of_date, scorecard)},
    )
    packet = context.packets[ticker]
    canonical_scorecard = copy.deepcopy(packet.raw_valuation)
    pricing = dict(canonical_scorecard.get("pricing_zone_detail") or {})
    pricing.update(
        {
            "current_price": packet.current_price,
            "dcf_base": packet.dcf_value,
            "epv_adjusted": packet.epv_value,
            "graham_value_per_share": packet.graham_value,
        }
    )
    canonical_scorecard["pricing_zone_detail"] = pricing
    return ticker, as_of_date, canonical_scorecard, packet


def load_authorized_discover_universe(
    engine_db_path: str | Path,
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    """Select newest scorecards first, then require exact-source authorization."""

    conn = _sqlite3.connect(str(engine_db_path))
    conn.row_factory = _sqlite3.Row
    try:
        candidates = conn.execute(
            """
            SELECT ticker, MAX(as_of_date) AS as_of_date
            FROM valuations
            WHERE method = 'scorecard'
            GROUP BY ticker
            ORDER BY ticker
            """
        ).fetchall()
        rows = []
        for candidate in candidates:
            ticker = str(candidate["ticker"]).strip().upper()
            as_of_date = str(candidate["as_of_date"]).strip()[:10]
            row = _select_authorized_scorecard_row(
                conn,
                ticker=ticker,
            )
            if row is None:
                _reject_unauthorized_scorecard(
                    ticker=ticker,
                    as_of_date=as_of_date,
                )
            rows.append(row)
    finally:
        conn.close()

    universe: list[tuple[str, str, dict[str, Any]]] = []
    packets: dict[str, Any] = {}
    for row in rows:
        ticker, as_of_date, scorecard, packet = _authorized_scorecard_input(
            row,
            engine_db_path=engine_db_path,
        )
        universe.append((ticker, as_of_date, scorecard))
        packets[ticker] = packet
    return universe, packets


def _new_sweep_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _make_fast_bundle_builder(
    scorecard_lookup: dict[str, tuple[str, dict[str, Any]]],
    financial_packets: dict[str, Any],
) -> Callable[..., Any]:
    """Build a bundle_builder closure that uses the fast cached-scorecard path.

    For tickers in ``scorecard_lookup`` it calls
    ``build_analysis_evidence_bundle_from_cached_scorecard`` (skipping the
    slow ``ensure_all_facts -> ensure_valuation -> _load_scorecard`` chain).
    For unknown tickers it falls back to the full
    ``build_analysis_evidence_bundle`` (slow but correct).

    The closure captures ``scorecard_lookup`` by reference so the caller's
    universe data is available to every downstream Stage 3 / Stage 4 call
    without plumbing it through every function signature.
    """
    from app.analyst.bundle_builder import (
        build_analysis_evidence_bundle,
        build_analysis_evidence_bundle_from_cached_scorecard,
    )

    def fast_bundle_builder(ticker: str, as_of_date: str | None = None):
        entry = scorecard_lookup.get(ticker.upper())
        if entry is None:
            logger.warning(
                "sweep: no cached scorecard for %s, falling back to slow path",
                ticker,
            )
            return build_analysis_evidence_bundle(
                ticker,
                as_of_date=as_of_date,
                financial_packet=financial_packets.get(ticker.upper()),
            )
        sc_as_of, sc = entry
        return build_analysis_evidence_bundle_from_cached_scorecard(
            ticker=ticker,
            scorecard=sc,
            scorecard_as_of_date=sc_as_of,
            as_of_date=as_of_date,
            financial_packet=financial_packets.get(ticker.upper()),
        )

    return fast_bundle_builder


def _wire_fast_path(
    universe: list[tuple[str, str, dict[str, Any]]],
    bundle_builder: Callable[..., Any] | None,
    stage4_context_builder: Callable[[str], dict[str, Any]] | None,
    stage4_tool_dispatcher: Callable[[str, dict[str, Any]], str] | None,
    financial_packets: dict[str, Any],
) -> tuple[
    Callable[..., Any], Callable[[str], dict[str, Any]], Callable[[str, dict[str, Any]], str]
]:
    """Derive fast-path callables from universe scorecards if not provided."""
    if bundle_builder is None or (stage4_context_builder is None or stage4_tool_dispatcher is None):
        scorecard_lookup: dict[str, tuple[str, dict[str, Any]]] = {
            t.upper(): (d, sc) for t, d, sc in universe
        }
        derived_bundle_builder = _make_fast_bundle_builder(
            scorecard_lookup,
            financial_packets,
        )
        if bundle_builder is None:
            bundle_builder = derived_bundle_builder

        if stage4_context_builder is None or stage4_tool_dispatcher is None:
            from app.discover.stage4_context import make_stage4_helpers

            derived_ctx, derived_tools = make_stage4_helpers(
                bundle_builder,
                financial_packets=financial_packets,
            )
            if stage4_context_builder is None:
                stage4_context_builder = derived_ctx
            if stage4_tool_dispatcher is None:
                stage4_tool_dispatcher = derived_tools

    assert bundle_builder is not None
    assert stage4_context_builder is not None
    assert stage4_tool_dispatcher is not None
    return bundle_builder, stage4_context_builder, stage4_tool_dispatcher


def _run_sweep_core(
    db_path: str | Path,
    sid: str,
    universe: list[tuple[str, str, dict[str, Any]]],
    client,
    bundle_builder: Callable[..., Any],
    stage4_context_builder: Callable[[str], dict[str, Any]],
    stage4_tool_dispatcher: Callable[[str, dict[str, Any]], str],
    financial_packets: dict[str, Any],
    stage2_config: Stage2Config | None = None,
    stage3_config: Stage3Config | None = None,
    stage4_config: Stage4Config | None = None,
) -> str:
    """Shared sweep body. Each stage filters out already-done tickers."""
    try:
        run_stage2(
            db_path=db_path,
            sweep_id=sid,
            tickers_with_scorecards=universe,
            client=client,
            config=stage2_config,
            financial_packets=financial_packets,
        )

        keeps = list_stage2_keeps(
            db_path,
            sid,
            require_financial_scope=True,
        )
        logger.info("sweep %s: stage2 produced %d KEEPs", sid, len(keeps))
        if keeps:
            run_stage3(
                db_path=db_path,
                sweep_id=sid,
                tickers=[k["ticker"] for k in keeps],
                client=client,
                config=stage3_config,
                bundle_builder=bundle_builder,
                financial_packets=financial_packets,
            )

            survivors = list_stage3_survivors(
                db_path,
                sid,
                require_financial_scope=True,
            )
            logger.info("sweep %s: stage3 produced %d survivors", sid, len(survivors))
            if survivors:
                # Build Stage 3 → Stage 4 handoff: thread Stage 3 verdicts,
                # theses, and open questions into the Stage 4 context so the
                # deep loop can resolve prior findings instead of starting blind.
                from app.discover.persistence import list_stage3_results

                s3_results = list_stage3_results(
                    db_path,
                    sid,
                    require_financial_scope=True,
                )
                s3_lookup = {r["ticker"]: r for r in s3_results}

                # Re-derive Stage 4 helpers with Stage 3 context
                from app.discover.stage4_context import make_stage4_helpers

                s4ctx_with_handoff, s4tools_with_handoff = make_stage4_helpers(
                    bundle_builder,
                    stage3_lookup=s3_lookup,
                    financial_packets=financial_packets,
                )

                run_stage4(
                    db_path=db_path,
                    sweep_id=sid,
                    tickers=[s["ticker"] for s in survivors],
                    client=client,
                    config=stage4_config,
                    context_builder=s4ctx_with_handoff,
                    tool_dispatcher=s4tools_with_handoff,
                    financial_packets=financial_packets,
                )
    except InvalidFinancialInputError:
        logger.exception(
            "sweep %s rejected stale or invalid financial context",
            sid,
        )
        finalize_sweep(
            db_path,
            sid,
            status="invalid_financial_input",
        )
        raise
    except Exception:
        logger.exception("sweep %s aborted", sid)
        finalize_sweep(db_path, sid, status="aborted")
        raise

    finalize_sweep(db_path, sid, status="completed")
    return sid


def run_sweep(
    db_path: str | Path,
    universe: list[tuple[str, str, dict[str, Any]]],
    limit: int | None,
    budget_usd: float | None,
    client,
    bundle_builder: Callable[..., Any] | None = None,
    stage4_context_builder: Callable[[str], dict[str, Any]] | None = None,
    stage4_tool_dispatcher: Callable[[str, dict[str, Any]], str] | None = None,
    stage2_config: Stage2Config | None = None,
    stage3_config: Stage3Config | None = None,
    stage4_config: Stage4Config | None = None,
    sweep_id: str | None = None,
    financial_packets: dict[str, Any] | None = None,
) -> str:
    """Run a fresh four-stage funnel. Returns the sweep_id."""
    ensure_schema(db_path)
    sid = sweep_id or _new_sweep_id()
    truncated = universe if limit is None else universe[:limit]
    packet_map = {
        ticker.upper(): packet
        for ticker, packet in (financial_packets or {}).items()
        if any(ticker.upper() == item[0].upper() for item in truncated)
    }

    create_sweep(
        db_path=db_path,
        sweep_id=sid,
        universe_size=len(universe),
        limit_applied=limit,
        budget_usd=budget_usd,
    )
    insert_sweep_universe(
        db_path,
        sid,
        [(t, d) for t, d, sc in truncated],
    )

    logger.info("sweep %s: universe=%d, limited=%d", sid, len(universe), len(truncated))

    bb, s4ctx, s4tools = _wire_fast_path(
        truncated,
        bundle_builder,
        stage4_context_builder,
        stage4_tool_dispatcher,
        packet_map,
    )

    return _run_sweep_core(
        db_path=db_path,
        sid=sid,
        universe=truncated,
        client=client,
        bundle_builder=bb,
        stage4_context_builder=s4ctx,
        stage4_tool_dispatcher=s4tools,
        financial_packets=packet_map,
        stage2_config=stage2_config,
        stage3_config=stage3_config,
        stage4_config=stage4_config,
    )


def _lazy_backfill_universe(
    db_path: str | Path,
    sweep_id: str,
    engine_db_path: str | Path,
) -> list[tuple[str, str]]:
    """Synthesize a universe snapshot for a pre-snapshot sweep.

    Reads tickers from stage2_results, looks up raw scorecard dates, inserts
    the scheduling snapshot, and logs a warning. These dates are metadata
    only: ``_load_scorecards_for_snapshot`` subsequently requires the exact
    selected row to pass source authorization before any decision work resumes.
    """
    s2_tickers = list_stage2_tickers(
        db_path,
        sweep_id,
        require_financial_scope=True,
    )
    if not s2_tickers:
        logger.warning(
            "sweep %s: no stage2_results to backfill universe from",
            sweep_id,
        )
        return []

    # Look up as_of_date from the scorecard cache
    conn = _sqlite3.connect(str(engine_db_path))
    conn.row_factory = _sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in s2_tickers)
        rows = conn.execute(
            f"""
            SELECT v1.ticker, v1.as_of_date
            FROM valuations v1
            WHERE v1.method = 'scorecard'
              AND v1.ticker IN ({placeholders})
              AND v1.as_of_date = (
                  SELECT MAX(v2.as_of_date)
                  FROM valuations v2
                  WHERE v2.method = 'scorecard' AND v2.ticker = v1.ticker
              )
            """,
            list(s2_tickers),
        ).fetchall()
    finally:
        conn.close()

    cache_lookup = {r["ticker"]: r["as_of_date"] for r in rows}
    snapshot: list[tuple[str, str]] = []
    for ticker in sorted(s2_tickers):
        as_of = cache_lookup.get(ticker, "unknown")
        snapshot.append((ticker, as_of))

    insert_sweep_universe(db_path, sweep_id, snapshot)
    logger.warning(
        "sweep %s: lazy-backfilled universe from %d stage2_results — "
        "may be incomplete if the original sweep was killed mid-Stage-2",
        sweep_id,
        len(snapshot),
    )
    return snapshot


def _load_scorecards_for_snapshot(
    engine_db_path: str | Path,
    snapshot: list[tuple[str, str]],
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    """Load scorecard dicts from engine.db for the given snapshot tickers.

    Returns (ticker, as_of_date, scorecard_dict) triples. Tickers missing
    from the cache are logged and skipped.
    """
    tickers = [t for t, _ in snapshot]
    if not tickers:
        return [], {}

    conn = _sqlite3.connect(str(engine_db_path))
    conn.row_factory = _sqlite3.Row
    try:
        rows = []
        for ticker, snapshot_as_of in snapshot:
            exact_snapshot = snapshot_as_of != "unknown"
            if exact_snapshot:
                candidate = conn.execute(
                    """
                    SELECT ticker, as_of_date
                    FROM valuations
                    WHERE ticker = ? AND as_of_date = ? AND method = 'scorecard'
                    LIMIT 1
                    """,
                    (ticker, snapshot_as_of),
                ).fetchone()
            else:
                candidate = conn.execute(
                    """
                    SELECT ticker, as_of_date
                    FROM valuations
                    WHERE ticker = ? AND method = 'scorecard'
                    ORDER BY as_of_date DESC
                    LIMIT 1
                    """,
                    (ticker,),
                ).fetchone()
            if candidate is None:
                continue
            row = _select_authorized_scorecard_row(
                conn,
                ticker=ticker,
                as_of_date=(snapshot_as_of if exact_snapshot else None),
                exact_as_of_date=exact_snapshot,
            )
            if row is None:
                _reject_unauthorized_scorecard(
                    ticker=str(candidate["ticker"]).strip().upper(),
                    as_of_date=str(candidate["as_of_date"]).strip()[:10],
                )
            rows.append(row)
    finally:
        conn.close()

    universe: list[tuple[str, str, dict[str, Any]]] = []
    packets: dict[str, Any] = {}
    loaded_tickers: set[str] = set()
    for r in rows:
        try:
            ticker, as_of_date, sc, packet = _authorized_scorecard_input(
                r,
                engine_db_path=engine_db_path,
            )
        except _json.JSONDecodeError:
            logger.warning("sweep resume: invalid scorecard JSON for %s, skipping", r["ticker"])
            continue
        universe.append((ticker, as_of_date, sc))
        packets[ticker] = packet
        loaded_tickers.add(ticker)

    missing = set(tickers) - loaded_tickers
    for t in sorted(missing):
        logger.warning(
            "sweep resume: ticker %s in snapshot but not in scorecard cache — skipping",
            t,
        )

    return universe, packets


def resume_sweep(
    db_path: str | Path,
    engine_db_path: str | Path,
    sweep_id: str,
    client,
    budget_override: float | None = None,
    bundle_builder: Callable[..., Any] | None = None,
    stage4_context_builder: Callable[[str], dict[str, Any]] | None = None,
    stage4_tool_dispatcher: Callable[[str, dict[str, Any]], str] | None = None,
    stage2_config: Stage2Config | None = None,
    stage3_config: Stage3Config | None = None,
    stage4_config: Stage4Config | None = None,
) -> str:
    """Resume an existing sweep from where it stopped. Returns the sweep_id."""
    ensure_schema(db_path)
    sweep = get_sweep(db_path, sweep_id)
    if sweep is None:
        raise ValueError(f"sweep {sweep_id} not found")
    if sweep["status"] == "running":
        raise RuntimeError(f"sweep {sweep_id} has status 'running' — kill the process first")
    if sweep["status"] == "completed":
        logger.warning("resuming completed sweep %s — may be a no-op", sweep_id)

    # Load or backfill the universe snapshot
    snapshot = get_sweep_universe(db_path, sweep_id)
    if not snapshot:
        snapshot = _lazy_backfill_universe(db_path, sweep_id, engine_db_path)

    if not snapshot:
        logger.warning("sweep %s: empty universe snapshot, nothing to resume", sweep_id)
        finalize_sweep(db_path, sweep_id, status="completed")
        return sweep_id

    # Load scorecards for snapshot tickers from engine.db
    universe, financial_packets = _load_scorecards_for_snapshot(
        engine_db_path,
        snapshot,
    )
    logger.info(
        "sweep %s resume: loaded %d/%d scorecards from cache",
        sweep_id,
        len(universe),
        len(snapshot),
    )

    if not universe:
        logger.warning(
            "sweep %s: no scorecards found for any snapshot ticker — nothing to process",
            sweep_id,
        )
        finalize_sweep(db_path, sweep_id, status="completed")
        return sweep_id

    # Do not mutate the sweep into a running state until every exact snapshot
    # row has passed source-lineage and financial-input authorization.
    update_sweep_status(db_path, sweep_id, "running")

    if budget_override is not None:
        new_budget = sweep["total_cost_usd"] + budget_override
        update_sweep_budget(db_path, sweep_id, new_budget)
        logger.info(
            "sweep %s resume: budget override $%.2f (new cap $%.2f)",
            sweep_id,
            budget_override,
            new_budget,
        )

    bb, s4ctx, s4tools = _wire_fast_path(
        universe,
        bundle_builder,
        stage4_context_builder,
        stage4_tool_dispatcher,
        financial_packets,
    )

    return _run_sweep_core(
        db_path=db_path,
        sid=sweep_id,
        universe=universe,
        client=client,
        bundle_builder=bb,
        stage4_context_builder=s4ctx,
        stage4_tool_dispatcher=s4tools,
        financial_packets=financial_packets,
        stage2_config=stage2_config,
        stage3_config=stage3_config,
        stage4_config=stage4_config,
    )
