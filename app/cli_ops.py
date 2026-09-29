"""Operational CLI: preflight, heartbeat ledger, dead-man.

``ivi ops preflight``          — hard-exit environment guard for heartbeats
``ivi ops heartbeat-record``   — append one step outcome to heartbeat_runs
``ivi ops heartbeat-status``   — completion state for a heartbeat/date
``ivi ops deadman``            — 16:30 assertions: yesterday's event scan,
                                 today's reprice/price snapshots, heartbeat
                                 completions, backup freshness
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import typer

from app.config import DataVolumeError, check_data_volume, get_config

ops_app = typer.Typer(help="Operational guards: preflight, heartbeat ledger, dead-man checks")

# Threshold constants (price-snapshot/backup ceilings) live in
# app.ops.data_health so the digest health block and this dead-man check
# can never drift apart.
DEFAULT_DEADMAN_HEARTBEATS = "events_daily"


@ops_app.command("preflight")
def preflight_cmd(
    require_engine: bool = typer.Option(True, "--engine/--no-engine"),
    require_cache_dirs: bool = typer.Option(True, "--cache/--no-cache"),
) -> None:
    """Run the operational preflight; nonzero exit on any failed check."""
    from app.ops.preflight import run_preflight

    result = run_preflight(
        require_engine=require_engine, require_cache_dirs=require_cache_dirs
    )
    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.ok:
        raise typer.Exit(code=1)


@ops_app.command("heartbeat-record")
def heartbeat_record_cmd(
    heartbeat: str = typer.Option(..., "--heartbeat"),
    step: str = typer.Option(..., "--step"),
    exit_code: int = typer.Option(..., "--exit-code"),
    run_date: str | None = typer.Option(None, "--date"),
    detail: str | None = typer.Option(None, "--detail"),
) -> None:
    """Append one heartbeat step outcome to the heartbeat_runs ledger."""
    from app.ops.heartbeat_ledger import record_step

    row = record_step(
        heartbeat=heartbeat,
        step=step,
        exit_code=exit_code,
        run_date=run_date,
        detail=detail,
    )
    typer.echo(json.dumps(row))


@ops_app.command("heartbeat-status")
def heartbeat_status_cmd(
    heartbeat: str = typer.Option(..., "--heartbeat"),
    run_date: str | None = typer.Option(None, "--date"),
) -> None:
    """Completion state for one heartbeat/date. Exit 0 COMPLETE, 2 otherwise."""
    from app.ops.heartbeat_ledger import heartbeat_status

    status = heartbeat_status(heartbeat=heartbeat, run_date=run_date)
    typer.echo(json.dumps(status, indent=2))
    if status["status"] != "COMPLETE":
        raise typer.Exit(code=2)


@ops_app.command("pit-backfill")
def pit_backfill_cmd(
    limit: int | None = typer.Option(None, "--limit", help="Max cached payloads to process"),
    years_back: int = typer.Option(12, "--years-back"),
) -> None:
    """Stamp filed_date/form/accession from cached companyfacts JSON and
    seed the vintage table. No network; idempotent; resumable."""
    from app.ops.pit_backfill import run_pit_backfill

    counts = run_pit_backfill(limit=limit, years_back=years_back)
    typer.echo(json.dumps(counts, indent=2))


@ops_app.command("deadman")
def deadman_cmd(
    heartbeats: str = typer.Option(
        DEFAULT_DEADMAN_HEARTBEATS,
        "--heartbeats",
        help="Comma-separated heartbeat names whose ledger completion is asserted",
    ),
    check_backup: bool = typer.Option(True, "--backup/--no-backup"),
) -> None:
    """Dead-man assertions for the 16:30 check; nonzero exit on any red.

    Every red line is printed — the shell wrapper alerts with the output.
    """
    from app.ops.data_health import (
        check_backup_fresh,
        check_event_scan_fresh,
        check_price_snapshot_coverage,
        check_price_snapshots_fresh,
    )
    from app.ops.heartbeat_ledger import heartbeat_status

    cfg = get_config()
    today = date.today()
    reds: list[str] = []
    lines: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        lines.append(f"{'OK ' if ok else 'RED'} {name}: {detail}")
        if not ok:
            reds.append(f"{name}: {detail}")

    ok, detail = check_data_volume(cfg)
    check("data_volume_mounted", ok, detail)

    engine_path = Path(cfg.db_path)
    ok, detail = check_event_scan_fresh(engine_path, today=today)
    check("event_scan", ok, detail)
    ok, detail = check_price_snapshots_fresh(engine_path)
    check("price_snapshots", ok, detail)
    ok, detail = check_price_snapshot_coverage(engine_path)
    check("price_snapshot_coverage", ok, detail)

    for name in [h.strip() for h in heartbeats.split(",") if h.strip()]:
        try:
            status = heartbeat_status(heartbeat=name, run_date=today.isoformat())
            check(
                f"heartbeat_{name}",
                status["status"] == "COMPLETE",
                status["status"],
            )
        except (sqlite3.Error, DataVolumeError) as exc:
            check(f"heartbeat_{name}", False, f"ledger unreadable: {exc}")

    if check_backup:
        ok, detail = check_backup_fresh(today=today)
        check("backup", ok, detail)

    typer.echo("\n".join(lines))
    if reds:
        raise typer.Exit(code=1)
