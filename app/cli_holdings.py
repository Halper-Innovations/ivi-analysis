"""`ivi holdings` commands: the held book.

add / close / list — the PROVISIONAL holdings registry (schema shared with
the portfolio effort); held-book — the daily exit-signal pass; exits — the
open exit-signal queue. Classic records and signals; it never sizes and
never places orders.
"""

from __future__ import annotations

import json

import typer

holdings_app = typer.Typer(help="Held-book registry, exit signals, daily monitoring pass")


@holdings_app.command("add")
def holdings_add_cmd(
    ticker: str = typer.Argument(...),
    entry_date: str = typer.Option(..., "--entry-date", help="YYYY-MM-DD"),
    entry_price: float | None = typer.Option(None, "--entry-price"),
    size: str | None = typer.Option(None, "--size", help="Free-form (e.g. '1200 sh', '2% NAV')"),
    size_pct_nav: float | None = typer.Option(None, "--size-pct-nav"),
    drawdown_alert_pct: float | None = typer.Option(
        None, "--drawdown-alert-pct", help="Per-position tripwire (default 25)"
    ),
    notes: str | None = typer.Option(None, "--notes"),
) -> None:
    """Register a held position; auto-links thesis references and
    force-registers the name into the filing watch + events protection scope."""
    from app.logging import configure_logging
    from app.holdings import add_holding

    configure_logging()
    row = add_holding(
        ticker=ticker,
        entry_date=entry_date,
        entry_price=entry_price,
        size=size,
        size_pct_nav=size_pct_nav,
        drawdown_alert_pct=drawdown_alert_pct,
        notes=notes,
    )
    typer.echo(json.dumps(row, indent=2, default=str))


@holdings_app.command("close")
def holdings_close_cmd(
    holding_id: int = typer.Argument(..., help="holdings.id"),
    close_price: float | None = typer.Option(None, "--close-price"),
) -> None:
    """Close a holding (auto-populates max_drawdown_pct on it and the linked
    outcome row)."""
    from app.logging import configure_logging
    from app.holdings import close_holding

    configure_logging()
    row = close_holding(holding_id=holding_id, close_price=close_price)
    typer.echo(json.dumps(row, indent=2, default=str))


@holdings_app.command("list")
def holdings_list_cmd(
    all_rows: bool = typer.Option(False, "--all", help="Include CLOSED holdings"),
) -> None:
    """List holdings (OPEN by default)."""
    from app.logging import configure_logging
    from app.holdings import list_holdings

    configure_logging()
    rows = list_holdings(status=None if all_rows else "OPEN")
    if not rows:
        typer.echo("No holdings.")
        return
    for row in rows:
        typer.echo(
            f"- #{row['id']} {row['ticker']} {row['status']} entry {row['entry_date']} "
            f"@ {row['entry_price']} size={row['size'] or row['size_pct_nav'] or 'n/a'} "
            f"peak={row['peak_price']}"
        )


@holdings_app.command("exits")
def holdings_exits_cmd() -> None:
    """Open exit signals — the typed feed the portfolio layer reads."""
    from app.logging import configure_logging
    from app.holdings import open_exit_signals

    configure_logging()
    rows = open_exit_signals()
    if not rows:
        typer.echo("No open exit signals.")
        return
    typer.echo(json.dumps(rows, indent=2, default=str))


@holdings_app.command("held-book")
def holdings_held_book_cmd() -> None:
    """Daily held-book pass: emit typed exit signals for OPEN holdings.

    Critical signals (gate flip, universe exit, drawdown tripwire,
    monitoring failure, contradicted thesis) push-alert the operator directly —
    the heartbeat step itself stays green so signal != outage.
    """
    from app.logging import configure_logging
    from app.holdings import run_held_book_pass
    from app.ops.alerts import post_alert

    configure_logging()
    report = run_held_book_pass()
    typer.echo(json.dumps(report, indent=2, default=str))
    if report["critical"]:
        lines = [
            f"{signal['ticker']}: {signal['signal_type']}"
            for signal in report["critical"][:6]
        ]
        post_alert(
            "IVI held-book EXIT SIGNALS",
            f"{len(report['critical'])} critical signal(s) — " + "; ".join(lines),
        )
