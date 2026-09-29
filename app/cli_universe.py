"""`ivi universe` commands: registrant census, classification, ingest, sync.

The universe expansion program rebuilt ingestion around the SEC's own
exchange registry so the platform's reviewable universe is no longer
Russell-3000-derived. Each command is resumable and idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from app.logging import get_logger

universe_app = typer.Typer(help="True-universe census, classification, ingest, and sync")
logger = get_logger(__name__)


def _census_input_paths(
    *,
    sec_registry: Path,
    nasdaq_listed: Path,
    other_listed: Path,
    companiesmarketcap_pages: list[Path],
    stockanalysis_html: Path | None,
    terminal_cap_evidence: Path | None,
    submissions_dir: Path | None,
    fixed_cohort_dir: Path | None,
):
    from app.universe.us_equity_census import CensusInputPaths

    return CensusInputPaths(
        sec_registry=sec_registry,
        nasdaq_listed=nasdaq_listed,
        other_listed=other_listed,
        companiesmarketcap_pages=tuple(companiesmarketcap_pages),
        stockanalysis_html=stockanalysis_html,
        terminal_cap_evidence=terminal_cap_evidence,
        submissions_dir=submissions_dir,
        fixed_cohort_dir=fixed_cohort_dir,
    )


@universe_app.command("census-fetch-submissions")
def universe_census_fetch_submissions_cmd(
    sec_registry: Path = typer.Option(..., exists=True, dir_okay=False),
    nasdaq_listed: Path = typer.Option(..., exists=True, dir_okay=False),
    other_listed: Path = typer.Option(..., exists=True, dir_okay=False),
    companiesmarketcap_page: list[Path] = typer.Option(
        [], "--companiesmarketcap-page", exists=True, dir_okay=False
    ),
    stockanalysis_html: Path | None = typer.Option(None, exists=True, dir_okay=False),
    fixed_cohort_dir: Path | None = typer.Option(None, exists=True, file_okay=False),
    output_dir: Path = typer.Option(..., file_okay=False),
    as_of: str = typer.Option(..., help="Fixed evidence date (YYYY-MM-DD)"),
    max_fetches: int | None = typer.Option(
        None, min=1, help="Optional bounded SEC request count; reruns resume from snapshots"
    ),
) -> None:
    """Fetch free SEC submissions for the >=$10B/boundary source union."""
    from app.universe.us_equity_census import fetch_large_cap_sec_submissions

    inputs = _census_input_paths(
        sec_registry=sec_registry,
        nasdaq_listed=nasdaq_listed,
        other_listed=other_listed,
        companiesmarketcap_pages=companiesmarketcap_page,
        stockanalysis_html=stockanalysis_html,
        terminal_cap_evidence=None,
        submissions_dir=None,
        fixed_cohort_dir=fixed_cohort_dir,
    )
    result = fetch_large_cap_sec_submissions(
        inputs,
        output_dir=output_dir,
        as_of_date=as_of,
        max_fetches=max_fetches,
    )
    typer.echo(
        json.dumps(
            {
                "as_of_date": result.as_of_date,
                "output_dir": str(result.output_dir),
                "candidate_cik_count": result.candidate_cik_count,
                "existing_snapshot_count": result.existing_snapshot_count,
                "fetched_snapshot_count": result.fetched_snapshot_count,
                "failed_snapshot_count": result.failed_snapshot_count,
                "deferred_snapshot_count": result.deferred_snapshot_count,
                "unresolved_candidate_count": result.unresolved_candidate_count,
                "fetch_summary": str(result.output_dir / "fetch_summary.json"),
                "actual_llm_calls": result.actual_llm_calls,
                "actual_llm_cost_usd": result.actual_llm_cost_usd,
            },
            indent=2,
            sort_keys=True,
        )
    )


@universe_app.command("census-run")
def universe_census_run_cmd(
    sec_registry: Path = typer.Option(..., exists=True, dir_okay=False),
    nasdaq_listed: Path = typer.Option(..., exists=True, dir_okay=False),
    other_listed: Path = typer.Option(..., exists=True, dir_okay=False),
    companiesmarketcap_page: list[Path] = typer.Option(
        [], "--companiesmarketcap-page", exists=True, dir_okay=False
    ),
    stockanalysis_html: Path | None = typer.Option(None, exists=True, dir_okay=False),
    terminal_cap_evidence: Path | None = typer.Option(None, exists=True, dir_okay=False),
    submissions_dir: Path | None = typer.Option(None, exists=True, file_okay=False),
    fixed_cohort_dir: Path | None = typer.Option(None, exists=True, file_okay=False),
    output_dir: Path = typer.Option(..., file_okay=False),
    as_of: str = typer.Option(..., help="Fixed evidence date (YYYY-MM-DD)"),
    run_id: str | None = typer.Option(None, help="Stable run ID; same inputs resume idempotently"),
    db_path: Path | None = typer.Option(None, dir_okay=False),
    promote_sector_population: bool = typer.Option(
        True,
        help="Persist admitted primary tickers into sector_inference for future scans",
    ),
) -> None:
    """Run/resume the registry-first census and export its acceptance artifacts."""
    from app.db import connect
    from app.universe.us_equity_census import run_us_equity_census

    inputs = _census_input_paths(
        sec_registry=sec_registry,
        nasdaq_listed=nasdaq_listed,
        other_listed=other_listed,
        companiesmarketcap_pages=companiesmarketcap_page,
        stockanalysis_html=stockanalysis_html,
        terminal_cap_evidence=terminal_cap_evidence,
        submissions_dir=submissions_dir,
        fixed_cohort_dir=fixed_cohort_dir,
    )
    conn = connect(db_path)
    try:
        result = run_us_equity_census(
            conn=conn,
            inputs=inputs,
            output_dir=output_dir,
            as_of_date=as_of,
            run_id=run_id,
            promote_sector_population=promote_sector_population,
        )
    finally:
        conn.close()
    typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))


@universe_app.command("census-status")
def universe_census_status_cmd(
    run_id: str = typer.Argument(...),
    db_path: Path | None = typer.Option(None, dir_okay=False),
) -> None:
    """Show the persisted run checkpoint and every stage attempt."""
    from app.db import connect
    from app.universe.us_equity_census import load_census_status

    conn = connect(db_path)
    try:
        status = load_census_status(conn, run_id=run_id)
    finally:
        conn.close()
    if status is None:
        raise typer.BadParameter(f"unknown census run_id: {run_id}")
    typer.echo(json.dumps(status, indent=2, sort_keys=True))


@universe_app.command("census")
def universe_census_cmd(
    as_of: str = typer.Option(None, help="Effective date (YYYY-MM-DD, default today)"),
    fetch_missing: bool = typer.Option(
        True, help="Fetch EDGAR submissions for registrants missing from the local cache"
    ),
    max_fetches: int = typer.Option(
        None, help="Cap on new submissions fetches this run (resumable)"
    ),
    refresh_registry: bool = typer.Option(
        False, help="Force-refresh the SEC exchange registry instead of using the 7-day cache"
    ),
    cap_bands: bool = typer.Option(
        True, help="Split new operating companies by cap band via the fallback chain"
    ),
    workers: int = typer.Option(4, help="Concurrent submissions fetch workers"),
) -> None:
    """Phase A: census of US-exchange operating companies vs the classified universe."""
    from app.universe.registrant_census import run_registrant_census, write_census_report

    report = run_registrant_census(
        as_of_date=as_of,
        fetch_missing=fetch_missing,
        max_fetches=max_fetches,
        refresh_registry=refresh_registry,
        cap_bands=cap_bands,
        workers=workers,
    )
    paths = write_census_report(report)
    summary = {
        key: report[key]
        for key in (
            "in_scope_exchange_registrants",
            "operating_companies",
            "known_to_sector_inference",
            "new_operating_companies",
            "new_by_exchange",
            "operating_status_counts",
        )
        if key in report
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(f"report: {paths['md']}")


@universe_app.command("ingest")
def universe_ingest_cmd(
    as_of: str = typer.Option(None, help="Effective date (YYYY-MM-DD, default today)"),
    limit: int = typer.Option(None, help="Ingest at most N registrants (resumable)"),
    run_gate: bool = typer.Option(True, help="Run the structural exclusion gate at intake"),
    run_valuation: bool = typer.Option(
        True, help="Run the deterministic valuation pipeline (required for sweepability)"
    ),
) -> None:
    """Phase C: companyfacts + valuation + structural gate for classified registrants."""
    from app.universe.registrant_intake import ingest_new_registrants, write_ingest_report

    report = ingest_new_registrants(
        as_of_date=as_of, limit=limit, run_gate=run_gate, run_valuation=run_valuation
    )
    paths = write_ingest_report(report)
    summary = {
        key: report[key]
        for key in (
            "scope",
            "ingested",
            "quarantined",
            "failed",
            "budget_exhausted",
            "status_counts",
            "gate_code_counts",
        )
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(f"report: {paths['md']}")


@universe_app.command("sync")
def universe_sync_cmd(
    as_of: str = typer.Option(None, help="Effective date (YYYY-MM-DD, default today)"),
    refresh_registry: bool = typer.Option(
        True, help="Refresh the SEC exchange registry (weekly sync wants fresh data)"
    ),
    max_ingest: int = typer.Option(None, help="Cap delta ingest at N names (resumable)"),
) -> None:
    """Phase E: weekly registrant diff -> classify -> ingest -> gate. Cron-able."""
    from app.universe.registrant_intake import sync_universe, write_sync_report

    report = sync_universe(
        as_of_date=as_of, refresh_registry=refresh_registry, max_ingest=max_ingest
    )
    paths = write_sync_report(report)
    summary = {
        key: report[key]
        for key in (
            "registrants_added",
            "registrants_removed",
            "new_operating_companies",
            "classified",
            "unclassified_review_count",
            "ingested",
            "quarantined",
            "ingest_failed",
        )
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(f"report: {paths['md']}")


@universe_app.command("resolve-evidence")
def universe_resolve_evidence_cmd(
    as_of: str = typer.Option(None, help="Effective date (YYYY-MM-DD, default today)"),
    sectors: str = typer.Option(
        None, help="Comma-separated sector filter (default: every sector with an artifact)"
    ),
    max_names: int = typer.Option(None, help="Re-audit at most N held names (resumable)"),
    repair: bool = typer.Option(True, help="Run deterministic repairs before re-auditing"),
) -> None:
    """Phase D: repair fetchable gaps for held candidates and re-audit them."""
    from app.autonomous.evidence_resolution import (
        resolve_and_reaudit_pool,
        write_evidence_resolution_report,
    )

    report = resolve_and_reaudit_pool(
        as_of_date=as_of,
        sectors=[s.strip() for s in sectors.split(",")] if sectors else None,
        max_names=max_names,
        repair=repair,
    )
    paths = write_evidence_resolution_report(report)
    summary = {
        key: report[key]
        for key in (
            "held_total",
            "held_fetchable_only",
            "reaudited",
            "transitions",
            "promoted_count",
        )
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(f"report: {paths['md']}")


@universe_app.command("classify")
def universe_classify_cmd(
    as_of: str = typer.Option(None, help="Effective date (YYYY-MM-DD, default today)"),
    limit: int = typer.Option(None, help="Classify at most N registrants (resumable)"),
) -> None:
    """Phase B: deterministic SIC classification of census-discovered registrants."""
    from app.universe.registrant_intake import (
        classify_new_registrants,
        write_classification_report,
    )

    report = classify_new_registrants(as_of_date=as_of, limit=limit)
    paths = write_classification_report(report)
    summary = {
        key: report[key]
        for key in (
            "scope",
            "already_classified_skipped",
            "status_counts",
            "unclassified_review_count",
            "unmapped_sic_counts",
        )
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(f"report: {paths['md']}")
