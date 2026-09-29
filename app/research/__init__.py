from __future__ import annotations

import json
import logging
from pathlib import Path

from app.db import get_db
from app.valuation.lineage import latest_decision_eligible_valuation_row

logger = logging.getLogger(__name__)


def _artifact_path_from_report_path(report_path: str) -> str | None:
    for suffix in ("_diagnostic_report.md", "_report.md"):
        if report_path.endswith(suffix):
            return report_path[: -len(suffix)] + ".json"
    return None


def run_legacy_research_packet_for_ticker(*args, **kwargs):
    """Compatibility wrapper for callers that still require ResearchPacket JSON."""
    from app.research.engine import run_research_agent_for_ticker as _impl

    return _impl(*args, **kwargs)


def run_legacy_research_for_scope(*args, **kwargs):
    """Compatibility wrapper for legacy batch packet generation."""
    from app.research.engine import run_research_for_scope as _impl

    return _impl(*args, **kwargs)


def latest_legacy_research_packet_path(
    ticker: str,
    as_of_date: str | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Return the latest legacy ResearchPacket artifact path."""
    from app.research.engine import latest_research_packet_path as _impl

    with get_db() as conn:
        return _impl(conn, ticker=ticker, as_of_date=as_of_date)


def latest_research_artifact_path(
    ticker: str,
    as_of_date: str | None = None,
) -> Path | None:
    """Return the latest deep-research JSON artifact path."""
    with get_db() as conn:
        row = latest_decision_eligible_valuation_row(
            conn,
            ticker=ticker.upper(),
            method="deep_research",
            as_of_date=as_of_date,
            exact_as_of_date=as_of_date is not None,
        )

    if not row or not row["outputs_json"]:
        return None

    try:
        payload = json.loads(row["outputs_json"])
    except Exception:
        return None

    artifact_path = payload.get("artifact_path")
    if not artifact_path and isinstance(payload.get("report_path"), str):
        artifact_path = _artifact_path_from_report_path(str(payload["report_path"]))

    if not artifact_path:
        return None

    path = Path(str(artifact_path))
    return path if path.exists() else None


def run_research_agent_for_ticker(
    ticker: str,
    as_of_date: str | None = None,
    run_id: str | None = None,
    source_filters: set[str] | None = None,
    *,
    years: int = 5,
    quarters: int = 0,
    build_legacy_packet: bool = True,
) -> Path | None:
    """Run canonical deep research and optionally backfill the legacy packet."""
    from app.research.deep_research import run_deep_research

    report = run_deep_research(
        ticker=ticker,
        as_of_date=as_of_date,
        years=years,
        quarters=quarters,
    )

    if build_legacy_packet:
        try:
            run_legacy_research_packet_for_ticker(
                ticker=ticker,
                as_of_date=as_of_date,
                run_id=run_id,
                source_filters=source_filters,
            )
        except Exception as exc:
            logger.warning(
                "research_api: legacy compatibility packet failed for %s: %s",
                ticker,
                exc,
            )

    try:
        from app.analyst.materializer import materialize_analysis_outputs_from_research

        materialize_analysis_outputs_from_research(
            report,
            years=years,
            quarters=quarters,
        )
    except Exception as exc:
        logger.warning(
            "research_api: analyst contract materialization failed for %s: %s",
            ticker,
            exc,
        )

    if report.artifact_path:
        artifact_path = Path(report.artifact_path)
        if artifact_path.exists():
            return artifact_path

    return latest_research_artifact_path(ticker=ticker, as_of_date=report.as_of_date)


def _active_universe_tickers() -> list[str]:
    from app.db import get_state

    with get_db() as conn:
        state = get_state(conn, "active_universe") or {}
        universe_id = state.get("universe_id")
        if universe_id:
            rows = conn.execute(
                "SELECT ticker FROM universe_members WHERE universe_id = ? ORDER BY id ASC",
                (universe_id,),
            ).fetchall()
            if rows:
                return [str(row["ticker"]).upper() for row in rows]

        rows = conn.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()
        return [str(row["ticker"]).upper() for row in rows]


def _top_ranked_tickers(top_n: int, as_of_date: str | None) -> list[str]:
    with get_db() as conn:
        if as_of_date:
            rows = conn.execute(
                """
                SELECT ticker
                FROM scores
                WHERE as_of_date <= ?
                ORDER BY total_score DESC, ticker ASC
                LIMIT ?
                """,
                (as_of_date, top_n),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ticker FROM scores ORDER BY total_score DESC, ticker ASC LIMIT ?",
                (top_n,),
            ).fetchall()
    return [str(row["ticker"]).upper() for row in rows]


def run_research_for_scope(
    as_of_date: str,
    *,
    top_n: int,
    use_active_universe: bool,
    run_id: str | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
    source_filters: set[str] | None = None,
    years: int = 5,
    quarters: int = 0,
    build_legacy_packet: bool = True,
) -> int:
    """Batch deep-research entrypoint with optional legacy packet backfill."""
    scope = [str(t).upper() for t in (tickers or []) if str(t).strip()]
    if not scope:
        if use_active_universe:
            scope = _active_universe_tickers()
        else:
            scope = _top_ranked_tickers(top_n=top_n, as_of_date=as_of_date)
            if not scope:
                scope = _active_universe_tickers()

    if limit is not None and limit > 0:
        scope = scope[:limit]

    built = 0
    for ticker in scope:
        path = run_research_agent_for_ticker(
            ticker=ticker,
            as_of_date=as_of_date,
            run_id=run_id,
            source_filters=source_filters,
            years=years,
            quarters=quarters,
            build_legacy_packet=build_legacy_packet,
        )
        if path:
            built += 1
    return built


def latest_research_packet_path(
    ticker: str,
    as_of_date: str | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Backwards-compatible alias for the latest canonical research artifact."""
    _ = run_id
    return latest_research_artifact_path(ticker=ticker, as_of_date=as_of_date)


__all__ = [
    "latest_legacy_research_packet_path",
    "latest_research_artifact_path",
    "latest_research_packet_path",
    "run_legacy_research_for_scope",
    "run_legacy_research_packet_for_ticker",
    "run_research_agent_for_ticker",
    "run_research_for_scope",
]
