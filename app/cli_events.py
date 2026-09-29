"""Corporate-events feed CLI sub-app (``ivi events ...``).

poll / backfill / surface / coverage / list — the detection feed; dispose —
the analyst pass that closes a queue-protection event and unblocks
DEPLOY_READY presentation; cheapness — the why-is-it-cheap pass over queue
names.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import typer

events_app = typer.Typer(help="Corporate events feed: detection, flags, cheapness")


def _previous_business_day(today: date | None = None) -> str:
    day = (today or datetime.now(ZoneInfo("America/New_York")).date()) - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.isoformat()


@events_app.command("poll")
def events_poll_cmd(
    scan_date: str | None = typer.Option(
        None, "--date",
        help="Index date YYYY-MM-DD (default: previous business day; today+ refused)",
    ),
) -> None:
    """Poll one EDGAR daily index through the detectors."""
    from app.logging import configure_logging
    from app.events.poller import poll_day

    configure_logging()
    target = scan_date or _previous_business_day()
    try:
        summary = poll_day(target)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(asdict(summary), indent=2))


@events_app.command("backfill")
def events_backfill_cmd(
    start: str = typer.Option(..., "--start", help="Window start YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Window end YYYY-MM-DD"),
    force: bool = typer.Option(False, "--force", help="Re-run days that already have OK scans"),
) -> None:
    """Backfill from quarterly full indexes (resumable per day)."""
    from app.logging import configure_logging
    from app.events.poller import backfill

    configure_logging()
    summaries = backfill(start, end, force=force)
    typer.echo(json.dumps([asdict(s) for s in summaries], indent=2))


@events_app.command("scan-gaps")
def events_scan_gaps_cmd(
    window_days: int = typer.Option(
        30, "--window-days", help="Trailing business-day window to self-heal"
    ),
) -> None:
    """Detect missed scan days in the trailing window and backfill them.

    Exits nonzero when a business day remains unrecovered after the backfill
    attempt so the heartbeat harness alerts.
    """
    from app.logging import configure_logging
    from app.events.poller import run_scan_gap_backfill

    configure_logging()
    result = run_scan_gap_backfill(window_days=window_days)
    typer.echo(json.dumps(result, indent=2))
    if result["unrecovered"]:
        raise typer.Exit(code=1)


@events_app.command("surface")
def events_surface_cmd(
    as_of: str | None = typer.Option(None, "--as-of", help="Intake as-of date YYYY-MM-DD"),
) -> None:
    """Route QUALIFIED opportunity events into watchlist intake."""
    from app.logging import configure_logging
    from app.events.intake import surface_qualified_events

    configure_logging()
    effective = as_of or date.today().isoformat()
    typer.echo(json.dumps(surface_qualified_events(as_of=effective), indent=2))


@events_app.command("coverage")
def events_coverage_cmd(
    csv_path: str | None = typer.Option(None, "--csv", help="Ground-truth CSV path"),
    lag_days: int = typer.Option(7, "--lag-days", help="Detection lag allowed for a hit"),
) -> None:
    """Coverage report vs the hand-curated ground-truth CSV."""
    from pathlib import Path

    from app.db import get_db
    from app.logging import configure_logging
    from app.events.validation import coverage_report, load_ground_truth

    configure_logging()
    default_csv = Path(__file__).parent / "events" / "ground_truth_2024_2025.csv"
    truth = load_ground_truth(csv_path or default_csv)
    with get_db() as conn:
        report = coverage_report(conn, truth, lag_days=lag_days)
    typer.echo(json.dumps(report, indent=2))


@events_app.command("list")
def events_list_cmd(
    status: str | None = typer.Option(None, "--status", help="Filter by lifecycle status"),
    event_type: str | None = typer.Option(None, "--type", help="Filter by event type"),
    open_only: bool = typer.Option(False, "--open", help="Only non-terminal events"),
) -> None:
    """List corporate events."""
    from app.db import get_db
    from app.logging import configure_logging

    configure_logging()
    conditions, params = [], []
    if status:
        conditions.append("status = ?")
        params.append(status.upper())
    if event_type:
        conditions.append("event_type = ?")
        params.append(event_type.lower())
    if open_only:
        conditions.append("status NOT IN ('DECIDED', 'EXPIRED')")
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM corporate_events {where} ORDER BY detection_date DESC, id DESC",
            params,
        ).fetchall()
    if not rows:
        typer.echo("No corporate events found.")
        return
    lines = [
        "| ID | Type | Ticker | CIK | Status | Detected | Qualified | Company |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['event_type']} | {row['ticker'] or 'UNKNOWN'} | "
            f"{row['cik']} | {row['status']} | {row['detection_date']} | "
            f"{row['qualification_date'] or '—'} | {row['company_name']} |"
        )
    typer.echo("\n".join(lines))


@events_app.command("dispose")
def events_dispose_cmd(
    event_id: int = typer.Argument(..., help="corporate_events.id to dispose"),
    note: str = typer.Option(..., "--note", help="Analyst disposition note (required)"),
    reason_code: str = typer.Option(
        "EVENT_REVIEWED",
        "--reason-code",
        help="Typed reason: EVENT_REVIEWED | PASSED_EVENT_RISK | OTHER ...",
    ),
    operator: str = typer.Option(
        "owner", "--operator", help="Who disposed the event (ledger attribution)"
    ),
) -> None:
    """Analyst pass: close an open event and clear its EVENT_PENDING flag.

    The disposal also lands in the typed disposition ledger with a
    structured reason code and operator — free-text-only disposals are gone.
    """
    from app.db import get_db
    from app.logging import configure_logging
    from app.events import store
    from app.events.flags import sync_event_pending_flags

    configure_logging()
    with get_db() as conn:
        event_row = conn.execute(
            "SELECT ticker FROM corporate_events WHERE id = ?", (event_id,)
        ).fetchone()
        moved = store.mark_decided(conn, event_id=event_id, note=note)
        sync = sync_event_pending_flags(conn)
    if not moved:
        typer.echo(f"Event {event_id} not disposed (already terminal or unknown id).")
        raise typer.Exit(code=1)

    from app.watchlist.dispositions import record_event_disposal

    ledger_row = record_event_disposal(
        ticker=event_row["ticker"] if event_row else None,
        event_id=event_id,
        operator=operator,
        reason_code=reason_code,
        note=note,
    )
    typer.echo(
        json.dumps(
            {"disposed": event_id, "flag_sync": sync, "disposition_id": ledger_row["id"]},
            indent=2,
        )
    )


@events_app.command("sync-flags")
def events_sync_flags_cmd() -> None:
    """Reconcile watchlist EVENT_PENDING flags with open events."""
    from app.db import get_db
    from app.logging import configure_logging
    from app.events.flags import sync_event_pending_flags

    configure_logging()
    with get_db() as conn:
        result = sync_event_pending_flags(conn)
    typer.echo(json.dumps(result, indent=2))


@events_app.command("dockets")
def events_dockets_cmd(
    scan_date: str | None = typer.Option(None, "--date", help="As-of date YYYY-MM-DD (default: today)"),
    lookback_days: int | None = typer.Option(None, "--lookback-days", help="Docket filed_after window (default: config, 30)"),
    tickers: list[str] = typer.Argument(None, help="Restrict to these tickers (default: at-target + ACTIONABLE names)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print hits without writing events"),
) -> None:
    """Scan CourtListener/RECAP for recent suits against queue names."""
    from app.logging import configure_logging
    from app.events.courtlistener import run_docket_scan

    configure_logging()
    result = run_docket_scan(
        scan_date, lookback_days=lookback_days,
        tickers=[t.upper() for t in (tickers or [])] or None, dry_run=dry_run,
    )
    typer.echo(json.dumps(result, indent=2))
    if result.get("status") in {"DISABLED", "NO_TOKEN"}:
        raise typer.Exit(code=1)


@events_app.command("news-scan")
def events_news_scan_cmd(
    scan_date: str | None = typer.Option(None, "--date", help="As-of date YYYY-MM-DD (default: today)"),
    tickers: list[str] = typer.Argument(None, help="Restrict to these tickers (default: at-target names)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print adverse hits without writing events"),
) -> None:
    """Scan Alpha Vantage news for adverse headlines on at-target names."""
    from app.logging import configure_logging
    from app.events.adverse_news import run_news_scan

    configure_logging()
    result = run_news_scan(
        scan_date, tickers=[t.upper() for t in (tickers or [])] or None, dry_run=dry_run,
    )
    typer.echo(json.dumps(result, indent=2))
    if result.get("status") in {"DISABLED", "NO_KEY"}:
        raise typer.Exit(code=1)


@events_app.command("cheapness")
def events_cheapness_cmd(
    tickers: list[str] = typer.Argument(None, help="Tickers (default: queue + DEPLOY_READY names)"),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD"),
    force: bool = typer.Option(False, "--force", help="Re-run even when the filing set is unchanged"),
    budget: float = typer.Option(0.05, "--budget", help="LLM budget per name (USD)"),
    full: bool = typer.Option(False, "--full", help="Print the full block per name"),
) -> None:
    """Build/refresh the known-reasons-it-may-be-cheap block for queue names."""
    from app.db import get_db
    from app.logging import configure_logging
    from app.events.cheapness import (
        build_cheapness_report,
        cheapness_headline,
        render_cheapness_block,
    )
    from app.ingest.sec_client import SecClient
    from app.events.submissions_adapter import SecSubmissionsAdapter
    from app.financial_integrity import InvalidFinancialInputError

    configure_logging()
    effective = as_of or date.today().isoformat()
    targets = [t.upper() for t in (tickers or [])]
    if not targets:
        from app.watchlist.store import list_active, watchlist_queue

        queue_rows = watchlist_queue(limit=25)
        deploy_entries = [
            *list_active(status="DEPLOY_READY"),
            *list_active(status="BUY_CONFIRMED"),
        ]
        targets = sorted(
            {str(r["ticker"]).upper() for r in queue_rows}
            | {entry.ticker.upper() for entry in deploy_entries}
        )
    client = SecClient()
    adapter = SecSubmissionsAdapter(client)
    lines = []
    refusals: list[dict[str, object]] = []
    for ticker in targets:
        try:
            with get_db() as conn:
                report = build_cheapness_report(
                    ticker, as_of=effective, conn=conn, client=client, adapter=adapter,
                    force=force, llm_budget_usd=budget,
                )
        except InvalidFinancialInputError as exc:
            codes = list(dict.fromkeys(item.code for item in exc.result.violations))
            refusal = {
                "ticker": ticker,
                "status": exc.status,
                "refusal_codes": codes,
                "financial_integrity": exc.result.to_dict(),
            }
            refusals.append(refusal)
            lines.append(
                f"## {ticker}\n- status: {exc.status}\n"
                f"- refusal_codes: {','.join(codes) or 'UNKNOWN'}"
            )
            continue
        if report.get("status") != "OK":
            lines.append(f"## {ticker}\n- status: {report.get('status')}")
            continue
        cache_note = " (cached)" if report.get("from_cache") else ""
        lines.append(f"## {ticker}{cache_note} — {cheapness_headline(report)}")
        if full:
            lines.extend(render_cheapness_block(report))
    typer.echo("\n".join(lines) if lines else "No targets.")
    if refusals:
        typer.echo(
            json.dumps(
                {
                    "status": "FAILED_FINANCIAL_INTEGRITY",
                    "targets": len(targets),
                    "integrity_refusals": len(refusals),
                    "refusals": refusals,
                },
                sort_keys=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1)


@events_app.command("triage-pack")
def events_triage_pack_cmd(
    limit: int = typer.Option(
        8, "--limit", help="Max events to pack per run (unmemo'd events re-enter tomorrow)"
    ),
) -> None:
    """Build the 8-K triage pack for the subscription analyst pass.

    Deterministic and $0: selects open queue-protection events flagging
    watchlist names (gate-blocked names first), fetches their filing text
    from EDGAR, and writes a self-contained pack. Prints the pack path on
    its last line, or NO_EVENTS when nothing needs a memo — the wrapper
    script keys off that line.
    """
    from app.events.triage_pack import build_pack
    from app.logging import configure_logging

    configure_logging()
    result = build_pack(limit=limit)
    if result["events"] == 0:
        typer.echo("NO_EVENTS")
        return
    typer.echo(f"events={result['events']} skipped_existing={result['skipped_existing']}")
    typer.echo(result["pack_path"])


@events_app.command("triage-verify")
def events_triage_verify_cmd(
    pack_path: str = typer.Argument(..., help="Path to a triage pack.json"),
) -> None:
    """Verify every memo a triage pack requested was written (exit 1 if not)."""
    from app.events.triage_pack import verify_pack
    from app.logging import configure_logging

    configure_logging()
    result = verify_pack(pack_path)
    typer.echo(json.dumps(result))
    if result["missing"]:
        raise typer.Exit(code=1)
