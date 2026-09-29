"""`ivi backtest` CLI: run the point-in-time edge backtest + report the scorecard."""
from __future__ import annotations

import json
from datetime import datetime

import typer

from app.backtest.h1 import run_h1

backtest_app = typer.Typer(help="Point-in-time edge backtest (reconstruct -> resolve -> report).")


@backtest_app.command("run")
def run_cmd(
    universe: str = typer.Option("watchlist", "--universe", help="'watchlist' or comma-separated tickers"),
    dates: str = typer.Option(..., "--dates", help="Comma-separated as-of dates YYYY-MM-DD, e.g. 2024-01-02,2024-07-01"),
    horizons: str = typer.Option("90,365", "--horizons", help="Comma-separated forward horizons (days)"),
    run_id: str = typer.Option("backtest_pilot", "--run-id", help="Backtest run id (rows tagged run_id_h<horizon>)"),
) -> None:
    """Reconstruct + snapshot the deterministic signal for each (ticker, date)."""
    from app.logging import configure_logging
    from app.backtest.runner import run_backtest

    configure_logging()
    if universe == "watchlist":
        from app.watchlist.store import watchlist_queue
        tickers = sorted({str(r["ticker"]) for r in watchlist_queue(limit=10_000)})
    else:
        tickers = [t.strip().upper() for t in universe.split(",") if t.strip()]
    date_list = []
    for d in dates.split(","):
        d = d.strip()
        if not d:
            continue
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            raise typer.BadParameter(f"Invalid --dates value (expected YYYY-MM-DD): {d!r}") from None
        date_list.append(d)
    if not date_list:
        raise typer.BadParameter("--dates requires at least one valid YYYY-MM-DD date.")

    horizon_list = []
    for token in horizons.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            horizon_list.append(int(token))
        except ValueError:
            raise typer.BadParameter(f"Invalid horizon (must be an integer day count): {token!r}") from None
    if not horizon_list:
        raise typer.BadParameter("--horizons requires at least one valid integer day count.")

    summary = run_backtest(universe=tickers, dates=date_list, horizons=horizon_list, run_id=run_id)
    typer.echo(json.dumps({
        "universe_size": summary.universe_size,
        "reconstructed": summary.reconstructed,
        "skipped": summary.skipped,
        "rows_written": summary.rows_written,
        "skipped_reasons": summary.skipped_reasons,
    }, indent=2))
    typer.echo(
        "Next: `ivi calibration-resolve --as-of <today>` to close matured rows, "
        "then `ivi backtest report`."
    )


@backtest_app.command("run-h1")
def run_h1_cmd(
    per_band: int = typer.Option(250, "--per-band", help="Names sampled per cap band per date"),
    seed: int = typer.Option(42, "--seed", help="Stratified-sample RNG seed"),
    dates: list[str] = typer.Option(None, "--dates", help="As-of dates (repeatable); default = built-in quarterly grid"),
    horizons: list[int] = typer.Option([90, 365], "--horizons", help="Forward horizons in days (repeatable)"),
    run_id: str = typer.Option("h1_sample", "--run-id"),
) -> None:
    """Run the scale backtest: per-date point-in-time stratified sample.

    Network-bound. Idempotent/resumable (re-run to recover transient price
    failures). After it finishes: `ivi calibration-resolve --as-of <today>`
    (or the delisting-aware resolve in code), then
    `ivi backtest report --run-id-prefix h1_sample`.
    """
    from app.logging import configure_logging

    configure_logging()
    summary = run_h1(
        per_band=per_band,
        seed=seed,
        dates=list(dates) if dates else None,
        horizons=list(horizons),
        run_id=run_id,
    )
    typer.echo(json.dumps({
        "universe_size": summary.universe_size,
        "reconstructed": summary.reconstructed,
        "skipped": summary.skipped,
        "rows_written": summary.rows_written,
    }, indent=2))
    typer.echo(
        "Next: `ivi calibration-resolve --as-of <today>` to close matured rows, "
        "then `ivi backtest report --run-id-prefix h1_sample`."
    )


@backtest_app.command("bootstrap")
def bootstrap_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Exact run id, e.g. backtest_pilot_h365"),
    group_field: str = typer.Option("grade", "--group-field", help="Cohort column: 'grade' or 'status'"),
    treat: str = typer.Option("DEPLOY_READY", "--treat", help="Treatment cohort value"),
    control: str = typer.Option("NOT_CHEAP", "--control", help="Control cohort value"),
    value_field: str = typer.Option("excess_return_pct", "--value-field", help="'excess_return_pct' or 'realized_return_pct'"),
    n_boot: int = typer.Option(10_000, "--n-boot", help="Bootstrap draws"),
    seed: int = typer.Option(1337, "--seed", help="Bootstrap RNG seed"),
    db_path: str = typer.Option(None, "--db-path", help="Override DB path (default: configured engine.db)"),
) -> None:
    """Quarter-clustered block bootstrap SE/CI for a cohort contrast."""
    from app.logging import configure_logging
    from app.backtest.inference import (
        cluster_bootstrap_contrast,
        format_bootstrap_report,
        load_closed_outcomes,
    )

    configure_logging()
    rows = load_closed_outcomes(run_id, db_path=db_path)
    if not rows:
        typer.echo(f"No CLOSED rows with non-null excess_return_pct for run_id={run_id!r}.", err=True)
        raise typer.Exit(code=1)
    result = cluster_bootstrap_contrast(
        rows,
        group_field=group_field,
        treat=treat,
        control=control,
        value_field=value_field,
        n_boot=n_boot,
        seed=seed,
    )
    typer.echo(
        format_bootstrap_report(
            result,
            run_id,
            group_field=group_field,
            treat=treat,
            control=control,
            value_field=value_field,
        )
    )


@backtest_app.command("report")
def report_cmd(
    run_id_prefix: str = typer.Option("backtest_pilot", "--run-id-prefix"),
    horizon: int = typer.Option(365, "--horizon"),
) -> None:
    """Print the edge scorecard over CLOSED backtest rows."""
    from app.logging import configure_logging
    from app.backtest.report import backtest_report, render_backtest_report

    configure_logging()
    report = backtest_report(run_id_prefix=run_id_prefix, horizon=horizon)
    typer.echo(render_backtest_report(report))
