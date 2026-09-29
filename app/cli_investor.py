"""Investor-facing decision views CLI sub-app.

A single Typer group (``ivi investor``) surfacing the decision-first output for
the human investor: ``buy-now`` / ``ideas`` / ``memo`` / ``watchlist``.

Verdict source: the price-trigger STATUS drives the
action verb (``DEPLOY_READY`` + non-AVOID grade => "Review at target") and the
conviction GRADE rides alongside as sizing/trust. The deploy flag routes
attention; it is not a buy signal and no line rendered here may phrase it as
one. ``PRICE_DATA_SUSPECT`` and ``QUARANTINE`` statuses are never trustworthy
enough to render the at-target action; those rows are not in the eligible set
surfaced here. This sub-app only reads and renders existing numbers; it never
mutates the watchlist and never touches any other command group.
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import typer

from app.decision.decision_block import (
    ACTION_REVIEW_AT_TARGET,
    DEFAULT_TIME_HORIZON,
    DecisionBlock,
    action_from_status_and_grade,
    buy_now_imperative_line,
    pct_to_target,
    render_decision_block_markdown,
)
from app.watchlist.contract import is_price_trigger_eligible


investor_app = typer.Typer(help="Investor-facing decision views")

# Flag an at-target line when the latest price snapshot is older
# than 7 trading days. Advisory only; never gates the line.
_STALE_SNAPSHOT_TRADING_DAYS = 7

# ACTIONABLE rows carry a full conviction memo; the rest are price triggers
# without one and must never outrank a memo-backed name however deep their
# discount runs.
_GRADE_ORDER = {"ACTIONABLE": 0, "WATCHLIST_ONLY": 1, "DATA_INCOMPLETE": 2}

_TODAY_BLOCKER_ORDER = (
    "DATA_HEALTH_RED",
    "OPERATIONS_WARNING",
    "EVENT_PENDING",
    "MISSING_PRICE",
    "STALE_PRICE",
    "MISSING_EVALUATION_DATE",
    "NOT_AT_TARGET",
    "NON_ACTIONABLE_GRADE",
    "UNTRUSTED_STATUS_OR_PROVENANCE",
    "PRICE_DATA_SUSPECT_OR_QUARANTINE",
    "CAPACITY_LIMITED_OR_UNKNOWN",
    "SUPERSEDED_DISPOSITION",
    "OPEN_CURRENT_DISPOSITION",
)

_TODAY_RESOLUTION_COMMANDS = {
    "DATA_HEALTH_RED": "`ivi ops preflight`",
    "OPERATIONS_WARNING": "`ivi ops preflight`",
    "EVENT_PENDING": "`ivi events list --open`",
    "MISSING_PRICE": "`ivi watchlist check-triggers <TICKER>`",
    "STALE_PRICE": "`ivi watchlist check-triggers <TICKER>`",
    "MISSING_EVALUATION_DATE": "`ivi watchlist refresh <TICKER>`",
    "UNTRUSTED_STATUS_OR_PROVENANCE": "`ivi investor memo <TICKER>`",
    "PRICE_DATA_SUSPECT_OR_QUARANTINE": (
        "`ivi watchlist check-triggers <TICKER> --force`"
    ),
    "CAPACITY_LIMITED_OR_UNKNOWN": (
        "`ivi watchlist backfill-adv --ticker <TICKER>`"
    ),
    "SUPERSEDED_DISPOSITION": (
        "`ivi investor journal <TICKER> --disposition-id <ID> "
        "--action passed|deferred ...`"
    ),
    "OPEN_CURRENT_DISPOSITION": (
        "`ivi investor journal <TICKER> --disposition-id <ID> ...`"
    ),
}


def _grade_rank(row: dict) -> int:
    return _GRADE_ORDER.get(str(row.get("conviction_grade") or "").upper(), 3)


def _v2_provenance_lines(entry: object) -> list[str]:
    """Literal v2 research state for investor-facing single-name views."""

    pipeline_version = str(getattr(entry, "pipeline_version", "") or "").lower()
    if pipeline_version != "v2":
        return []
    return [
        f"- PIPELINE: {pipeline_version}",
        "- CANDIDATE DISPOSITION: "
        f"{getattr(entry, 'candidate_disposition', None) or 'n/a'}",
        f"- DECISION BASIS: {getattr(entry, 'decision_basis', None) or 'n/a'}",
        "- SELECTION VALIDATION: "
        f"{getattr(entry, 'selection_validation_status', None) or 'n/a'}",
        f"- INVESTABLE: {'yes' if is_price_trigger_eligible(entry) else 'no'}",
    ]


def _parse_checked_at(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_snapshot_stale(checked_at: str | None, *, now: datetime | None = None) -> bool:
    """A snapshot older than 7 trading days (~9 calendar days) is stale."""
    parsed = _parse_checked_at(checked_at)
    if parsed is None:
        return False
    effective_now = now or datetime.now(timezone.utc)
    # 7 trading days span roughly 9 calendar days once a weekend is included.
    calendar_days = _STALE_SNAPSHOT_TRADING_DAYS + 2
    return (effective_now - parsed).days > calendar_days


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _today_money(value: object) -> str:
    return f"${float(value):,.2f}" if _finite_positive(value) else "n/a"


def _today_confidence_rank(value: object) -> int:
    return {"HIGH": 0, "MODERATE": 1, "LOW": 2}.get(
        str(value or "").upper(), 3
    )


def render_investor_today(
    *,
    queue_rows: list[dict[str, Any]],
    health: Any,
    open_rows: list[dict[str, Any]],
    now: datetime,
    price_max_age_days: int,
    adv_floor: float,
) -> str:
    """Pure Markdown renderer for the daily investor decision surface."""

    effective_now = now
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=timezone.utc)
    effective_now = effective_now.astimezone(timezone.utc)

    blockers: dict[str, list[str]] = {
        category: [] for category in _TODAY_BLOCKER_ORDER
    }

    def add_blocker(category: str, example: str) -> None:
        if example not in blockers[category]:
            blockers[category].append(example)

    health_blocking = bool(getattr(health, "blocking", False))
    health_has_failures = bool(getattr(health, "red", False))
    if health_has_failures:
        health_category = (
            "DATA_HEALTH_RED" if health_blocking else "OPERATIONS_WARNING"
        )
        for check in getattr(health, "checks", []) or []:
            if not bool(check.get("ok")):
                add_blocker(
                    health_category,
                    f"{check.get('name')}: {check.get('detail')}",
                )

    stale_open_tickers: set[str] = set()
    current_open_ids: set[int] = set()
    for disposition in open_rows:
        if str(disposition.get("kind")) != "AT_TARGET":
            continue
        ticker = str(disposition.get("ticker") or "UNKNOWN").upper()
        disposition_id = disposition.get("id")
        state = str(disposition.get("source_state") or "")
        if state == "SUPERSEDED_STALE_SOURCE":
            stale_open_tickers.add(ticker)
            add_blocker(
                "SUPERSEDED_DISPOSITION",
                f"{ticker} disposition #{disposition_id}",
            )
        elif state == "CURRENT":
            watchlist_id = disposition.get("watchlist_id")
            if isinstance(watchlist_id, int):
                current_open_ids.add(watchlist_id)
            add_blocker(
                "OPEN_CURRENT_DISPOSITION",
                f"{ticker} disposition #{disposition_id}",
            )

    candidates: list[dict[str, Any]] = []
    for row in queue_rows:
        ticker = str(row.get("ticker") or "UNKNOWN").upper()
        row_blockers: set[str] = set()

        def block(
            category: str,
            detail: str,
            *,
            _row_blockers: set[str] = row_blockers,
            _ticker: str = ticker,
        ) -> None:
            _row_blockers.add(category)
            add_blocker(category, f"{_ticker} ({detail})")

        status = str(row.get("status") or "").upper()
        grade = str(row.get("conviction_grade") or "").upper()
        price = row.get("latest_price")
        target = row.get("buy_price_target")
        checked_at = row.get("latest_price_checked_at")
        checked = _parse_checked_at(checked_at)

        if row.get("event_pending"):
            block("EVENT_PENDING", str(row["event_pending"]))
        if not _finite_positive(price) or checked_at is None:
            block("MISSING_PRICE", "fresh snapshot unavailable")
        elif checked is None:
            block("STALE_PRICE", f"unparseable timestamp {checked_at}")
        else:
            age_days = (effective_now - checked).total_seconds() / 86400.0
            if age_days > float(price_max_age_days):
                block("STALE_PRICE", f"{age_days:.1f}d old")
        if not str(row.get("last_evaluated_at") or "").strip():
            block("MISSING_EVALUATION_DATE", "last_evaluated_at absent")
        if _finite_positive(price) and _finite_positive(target):
            if float(price) > float(target):
                block("NOT_AT_TARGET", f"{_today_money(price)} > {_today_money(target)}")
        if grade != "ACTIONABLE":
            block("NON_ACTIONABLE_GRADE", grade or "missing grade")
        if (
            status not in {"DEPLOY_READY", "BUY_CONFIRMED"}
            or not is_price_trigger_eligible(row)
            or not _finite_positive(target)
        ):
            block(
                "UNTRUSTED_STATUS_OR_PROVENANCE",
                f"status={status or 'missing'}",
            )
        if status in {"PRICE_DATA_SUSPECT", "QUARANTINE"}:
            block("PRICE_DATA_SUSPECT_OR_QUARANTINE", status)
        adv20 = row.get("adv_dollar_20d")
        if not _finite_positive(adv20) or float(adv20) <= float(adv_floor):
            block(
                "CAPACITY_LIMITED_OR_UNKNOWN",
                "ADV unknown"
                if not _finite_positive(adv20)
                else f"ADV ${float(adv20):,.0f} <= floor ${float(adv_floor):,.0f}",
            )
        if ticker in stale_open_tickers:
            row_blockers.add("SUPERSEDED_DISPOSITION")
        if isinstance(row.get("id"), int) and row["id"] in current_open_ids:
            row_blockers.add("OPEN_CURRENT_DISPOSITION")

        if not row_blockers and not health_blocking:
            candidates.append(row)

    candidates.sort(
        key=lambda row: (
            _today_confidence_rank(row.get("confidence")),
            float(row.get("distance_from_buy_pct") or 0.0),
            str(row.get("ticker") or ""),
        )
    )
    candidates = candidates[:3]

    health_state = (
        "RED" if health_blocking else "AMBER" if health_has_failures else "GREEN"
    )
    lines = [
        "# IVI Investor Today",
        "",
        f"- Generated: {effective_now.isoformat()}",
        f"- Data health: {health_state}",
        "",
        "## Review at target",
    ]
    if not candidates:
        lines.append("No review-ready candidates today.")
    for row in candidates:
        ticker = str(row.get("ticker") or "UNKNOWN").upper()
        distance = float(row.get("distance_from_buy_pct") or 0.0)
        falsifier = next(
            (
                str(item).strip()
                for item in row.get("falsifiers", []) or []
                if str(item).strip()
            ),
            "not recorded",
        )
        pipeline = str(row.get("pipeline_version") or "legacy")
        provenance = (
            f"pipeline={pipeline}; disposition={row.get('candidate_disposition') or 'n/a'}; "
            f"basis={row.get('decision_basis') or 'n/a'}; "
            f"validation={row.get('selection_validation_status') or 'n/a'}"
        )
        lines.extend(
            [
                "",
                f"### {ticker} — REVIEW AT TARGET",
                f"- Current price: {_today_money(row.get('latest_price'))} "
                f"({row.get('latest_price_checked_at')})",
                f"- Target: {_today_money(row.get('buy_price_target'))}; "
                f"distance {distance:+.1f}%",
                f"- Conviction: {row.get('conviction_grade') or 'n/a'}; "
                f"confidence {row.get('confidence') or 'n/a'}",
                f"- Last evaluation: {row.get('last_evaluated_at')}",
                f"- Source: {row.get('source_sector') or 'n/a'}; "
                f"cap band {row.get('cap_band') or 'UNKNOWN_CAP'}",
                f"- Liquidity: ADV {_today_money(row.get('adv_dollar_20d'))}; "
                f"capacity {row.get('capacity_class') or 'ADV_UNKNOWN'}",
                f"- Event state: {row.get('event_pending') or 'CLEAR'}",
                f"- Provenance: {provenance}",
                f"- First useful falsifier: {falsifier}",
                "- review trigger, not a buy signal.",
            ]
        )

    lines.extend(["", "## Blockers"])
    for category in _TODAY_BLOCKER_ORDER:
        examples = blockers[category]
        line = f"- {category}: {len(examples)}"
        if examples:
            line += f" — {'; '.join(examples[:3])}"
        command = _TODAY_RESOLUTION_COMMANDS.get(category)
        if command and examples:
            line += f". Resolution: {command}"
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _buy_now_rows() -> tuple[list[dict], list[dict]]:
    """(presentable, event_blocked) DEPLOY_READY non-AVOID rows.

    Sorted conviction grade first (ACTIONABLE > WATCHLIST_ONLY >
    DATA_INCOMPLETE), then most-through-target within a grade — a deep
    discount on a memo-less name must not outrank a memo-backed one.

    A row with an open EVENT_PENDING flag is blocked from at-target
    presentation until the analyst disposes the event — it lands in the
    second list and is rendered as blocked, never as a review-at-target line.
    """
    from app.watchlist.store import watchlist_queue

    eligible = [
        row
        for row in watchlist_queue(limit=10_000)
        if str(row.get("status") or "") in {"DEPLOY_READY", "BUY_CONFIRMED"}
        and str(row.get("conviction_grade") or "").upper() != "AVOID"
        and is_price_trigger_eligible(row)
        and row.get("distance_from_buy_pct") is not None
    ]
    eligible.sort(
        key=lambda row: (
            _grade_rank(row),
            row["distance_from_buy_pct"],
            str(row.get("ticker") or ""),
        )
    )
    rows = [row for row in eligible if not row.get("event_pending")]
    blocked = [row for row in eligible if row.get("event_pending")]
    return rows, blocked


def _buy_now_row_lines(row: dict) -> list[str]:
    """Imperative line plus per-row caveats for one presentable at-target row."""
    # Omit the per-year expected-return token until the calibration loop
    # supplies a real number (do not fabricate).
    lines = [
        buy_now_imperative_line(
            str(row["ticker"]),
            row.get("latest_price"),
            row.get("buy_price_target"),
            None,
            row.get("conviction_grade"),
        )
    ]
    if not row.get("cap_band"):
        lines.append(
            "  (caveat: market cap UNKNOWN_CAP — band unverified; this "
            "name cannot be presented as in-band output)"
        )
    # Tradeability annotation — CAPACITY_LIMITED banner below the
    # dollar-ADV floor; explicit ADV_UNKNOWN when volume history is absent.
    from app.market.adv import adv_dollar_floor

    adv20 = row.get("adv_dollar_20d")
    if not isinstance(adv20, (int, float)):
        lines.append(
            "  (capacity: ADV_UNKNOWN — no volume history; tradeability unassessed)"
        )
    elif float(adv20) < adv_dollar_floor():
        lines.append(
            f"  (CAPACITY_LIMITED: 20d dollar-ADV ${float(adv20):,.0f} is below the "
            f"${adv_dollar_floor():,.0f} floor — real size cannot enter or exit "
            "at this trigger)"
        )
    else:
        lines.append(
            f"  (capacity: {row.get('capacity_class') or 'n/a'}, "
            f"20d dollar-ADV ${float(adv20):,.0f})"
        )
    cheapness = str(row.get("cheapness") or "n/a")
    if cheapness != "n/a":
        lines.append(f"  (why cheap: {cheapness})")
    checked_at = row.get("latest_price_checked_at")
    if checked_at is None:
        # No price snapshot exists, so the imperative is built off the
        # unverified addition price. Surface that rather than presenting an
        # old addition price as a live quote.
        lines.append(
            "  (caveat: no recent price snapshot — buy price uses the "
            "unverified addition price)"
        )
    elif _is_snapshot_stale(checked_at):
        lines.append(
            f"  (caveat: price snapshot {checked_at} "
            f"is older than {_STALE_SNAPSHOT_TRADING_DAYS} trading days)"
        )
    return lines


@investor_app.command("marks")
def investor_marks(
    refresh_benchmarks: bool = typer.Option(
        False, "--refresh-benchmarks", help="Fetch IWM/SPY anchors using the configured provider."
    ),
) -> None:
    """Mark every recorded buy pick using local quotes; never write to the database."""
    from contextlib import closing
    import sqlite3

    from app.calibration.picks_marks import (
        build_report, refresh_dates, render_markdown, write_report,
    )
    from app.config import get_config
    from app.db import connect

    cfg = get_config()
    today = date.today()
    refreshed = None
    note = "Network requests: 0 (local prices only)."
    try:
        with closing(connect(cfg=cfg, read_only=True)) as conn:
            conn.execute("BEGIN")
            if refresh_benchmarks:
                from app.calibration.picks_benchmarks import refresh_benchmarks as refresh

                refreshed = refresh(cfg, refresh_dates(conn, today))
                note = f"Benchmark refresh: {refreshed.request_count} {refreshed.request_count_unit}."
                if refreshed.failures:
                    note += "\n\nUnresolved benchmark anchors: " + "; ".join(refreshed.failures)
            report = build_report(
                conn, today=today, refreshed_prices=refreshed.prices if refreshed else None,
                refresh_note=note,
            )
        markdown, csv_path = write_report(report, cfg.outputs_dir)
    except (sqlite3.Error, OSError, ValueError) as exc:
        typer.echo(f"Cannot produce buy-pick marks: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(render_markdown(report))
    typer.echo(f"Saved {markdown} and {csv_path} (with summary and journal CSVs).")


@investor_app.command("today")
def today_cmd(
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write the exact Markdown report to this path and also print it",
    ),
) -> None:
    """Read-only daily decision truth: review candidates and blockers."""
    from app.market.adv import adv_dollar_floor
    from app.ops.data_health import compute_data_health
    from app.watchlist.dispositions import open_dispositions
    from app.watchlist.store import watchlist_queue
    from app.watchlist.triggers import _price_trigger_max_age_days

    now = datetime.now(timezone.utc).replace(microsecond=0)
    health = compute_data_health(now=now)
    markdown = render_investor_today(
        queue_rows=watchlist_queue(limit=10_000, include_price_suspect=True),
        health=health,
        open_rows=open_dispositions(),
        now=now,
        price_max_age_days=_price_trigger_max_age_days(),
        adv_floor=adv_dollar_floor(),
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    typer.echo(markdown, nl=False)


@investor_app.command("buy-now")
def buy_now_cmd() -> None:
    """Compatibility view; prefer `ivi investor today` for the daily decision pass."""
    from app.logging import configure_logging
    from app.ops.data_health import compute_data_health

    configure_logging()

    # Deterministic data-health preamble. Red renders first; an
    # engine-level failure blocks at-target rendering entirely — an empty or
    # shadow DB must never read as "nothing at target".
    health = compute_data_health()
    if health.red:
        typer.echo("\n".join(["# Data Health", *health.lines(), ""]))
    if health.blocking:
        typer.echo(
            "BLOCKED: engine DB failed basic integrity — refusing to render "
            "at-target candidates from a suspect DB."
        )
        raise typer.Exit(code=1)

    rows, blocked = _buy_now_rows()

    # The staleness caveat is ADVISORY by default. Setting
    # VOE_INVESTOR_STALENESS_BLOCKING=true promotes it to blocking — stale or
    # snapshot-less rows move to a blocked section instead of presenting as
    # actionable. Flipping the default is an explicit production-grade policy
    # decision for the operator, not a code default.
    stale_blocked: list[dict] = []
    if os.getenv("VOE_INVESTOR_STALENESS_BLOCKING", "").strip().lower() == "true":
        fresh_rows: list[dict] = []
        for row in rows:
            checked_at = row.get("latest_price_checked_at")
            if checked_at is None or _is_snapshot_stale(checked_at):
                stale_blocked.append(row)
            else:
                fresh_rows.append(row)
        rows = fresh_rows

    if not rows and not blocked and not stale_blocked:
        typer.echo("No at-target candidates: no DEPLOY_READY, non-AVOID rows.")
        return

    actionable = [row for row in rows if _grade_rank(row) == 0]
    secondary = [row for row in rows if _grade_rank(row) != 0]

    lines = ["# At Buy Target (Review)"]
    if not rows:
        lines.append("No presentable at-target candidates.")
    else:
        lines.extend(["", "## Actionable (conviction memo)"])
        if actionable:
            for row in actionable:
                lines.extend(_buy_now_row_lines(row))
        else:
            lines.append("No ACTIONABLE-grade names at target.")
        if secondary:
            lines.extend(["", "## No Conviction Memo (price triggers only)"])
            for row in secondary:
                lines.extend(_buy_now_row_lines(row))
    if blocked:
        lines.append("")
        lines.append("## Blocked Pending Event Review")
        for row in blocked:
            lines.append(
                f"- {row['ticker']}: {row['event_pending']} — at/below target but an "
                "open corporate event is undisposed; review the event "
                "(`ivi events list --open`) before any action."
            )
    if stale_blocked:
        lines.append("")
        lines.append("## Blocked: Stale Price Basis (VOE_INVESTOR_STALENESS_BLOCKING)")
        for row in stale_blocked:
            checked_at = row.get("latest_price_checked_at") or "no snapshot"
            lines.append(
                f"- {row['ticker']}: at/below target but the price basis is stale "
                f"({checked_at}) — refresh prices (`ivi watchlist check {row['ticker']}`) "
                "before any action."
            )

    # Open dispositions render until closed — every at-target surfacing
    # must terminate in a recorded ACTED/PASSED/DEFERRED decision.
    try:
        from app.watchlist.dispositions import open_dispositions

        open_rows = open_dispositions()
    except Exception:  # noqa: BLE001 - presentation survives a broken ledger
        open_rows = []
    if open_rows:
        lines.append("")
        lines.append("## Open Dispositions (decision required)")
        for row in open_rows:
            if row.get("source_state") == "SUPERSEDED_STALE_SOURCE":
                lines.append(
                    f"- {row['ticker']} ({row['kind']}, opened {str(row['opened_at'])[:10]}): "
                    "SUPERSEDED/STALE-SOURCE — explicit PASSED or DEFERRED required; "
                    f"close with `ivi investor journal {row['ticker']} "
                    f"--disposition-id {row['id']} --action passed|deferred "
                    "--reason <CODE> --rationale \"...\"`"
                )
            else:
                lines.append(
                    f"- {row['ticker']} ({row['kind']}, opened {str(row['opened_at'])[:10]}): "
                    f"close with `ivi investor journal {row['ticker']} "
                    f"--disposition-id {row['id']} --action acted|passed|deferred "
                    "--reason <CODE> --rationale \"...\"`"
                )
    typer.echo("\n".join(lines))


@investor_app.command("dispositions")
def dispositions_cmd() -> None:
    """List open dispositions — at-target surfacings awaiting a decision."""
    from app.logging import configure_logging
    from app.watchlist.dispositions import journal_entry_count, open_dispositions

    configure_logging()
    rows = open_dispositions()
    if not rows:
        typer.echo("No open dispositions.")
    for row in rows:
        state = str(row.get("source_state") or "n/a")
        suffix = (
            " — explicit PASSED or DEFERRED closure required"
            if state == "SUPERSEDED_STALE_SOURCE"
            else ""
        )
        typer.echo(
            f"- #{row['id']} {row['ticker']} ({row['kind']}, {state}) "
            f"opened {row['opened_at']}{suffix}"
        )
    typer.echo(
        f"\nJournal tally (run_id=journal_live): {journal_entry_count()} entries "
        "(kill criterion: >=6 journaled live decisions by 2026-09-09)."
    )


@investor_app.command("journal")
def journal_cmd(
    ticker: str = typer.Argument(..., help="Watchlist ticker to journal"),
    action: str = typer.Option(
        ..., "--action", help="acted | passed | deferred (maps to BUY/PASS/WATCH)"
    ),
    reason: str = typer.Option(
        ..., "--reason", help="Typed reason code (see ivi investor journal --help-reasons)"
    ),
    rationale: str = typer.Option(..., "--rationale", help="Why — the kill-switch record"),
    intended_size: str | None = typer.Option(None, "--size", help="Intended size (e.g. '2% NAV')"),
    sizing_rationale: str | None = typer.Option(None, "--sizing-rationale"),
    pre_mortem: str | None = typer.Option(None, "--pre-mortem"),
    disposition_id: int | None = typer.Option(
        None,
        "--disposition-id",
        help="Explicit OPEN disposition id (required to target a stale source)",
    ),
    operator: str = typer.Option("owner", "--operator", help="Who decided"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt"),
) -> None:
    """Journal a live decision — closes the ticker's open disposition and
    writes the ticker_outcomes journal row (run_id='journal_live').

    Pre-filled from the decision record but NEVER auto-emitted: the entry
    requires explicit confirmation (process theater is the named failure
    mode; an unconfirmed journal is worse than none).
    """
    from app.logging import configure_logging
    from app.watchlist.dispositions import REASON_CODES, close_disposition
    from app.watchlist.store import get_latest

    configure_logging()

    action_map = {"acted": "ACTED", "passed": "PASSED", "deferred": "DEFERRED"}
    status = action_map.get(action.strip().lower())
    if status is None:
        raise typer.BadParameter(f"--action must be one of {sorted(action_map)}")
    if reason.strip().upper() not in REASON_CODES:
        raise typer.BadParameter(f"--reason must be one of {REASON_CODES}")

    entry = get_latest(ticker)
    lines = [f"# Journal {ticker.upper()} — {status}"]
    if entry is not None:
        lines.append(
            f"- status: {entry.status} / grade: {entry.conviction_grade or 'n/a'} "
            f"({entry.confidence or 'n/a'})"
        )
        if is_price_trigger_eligible(entry):
            lines.append(f"- buy target: {entry.buy_price_target}")
        elif str(entry.pipeline_version or "").lower() == "v2":
            lines.append(
                "- price trigger: disabled until validated underwriting"
            )
        lines.append(
            f"- anchor: {entry.valuation_anchor_method or 'n/a'} "
            f"{entry.valuation_anchor_value or ''}"
        )
        lines.extend(_v2_provenance_lines(entry))
    else:
        lines.append("- (no live watchlist row — journaling as MANUAL)")
    lines.extend(
        [
            f"- reason: {reason.strip().upper()}",
            f"- rationale: {rationale}",
            f"- size: {intended_size or 'n/a'}",
            f"- operator: {operator}",
        ]
    )
    typer.echo("\n".join(lines))

    if not yes and not typer.confirm("Write this journal entry?"):
        typer.echo("Aborted — nothing written.")
        raise typer.Exit(code=1)

    result = close_disposition(
        ticker=ticker,
        status=status,
        operator=operator,
        reason_code=reason,
        rationale=rationale,
        intended_size=intended_size,
        sizing_rationale=sizing_rationale,
        pre_mortem=pre_mortem,
        disposition_id=disposition_id,
    )
    typer.echo(
        f"Disposition #{result['id']} closed as {result['status']} "
        f"({result['reason_code']}); journal row written (run_id=journal_live)."
    )


@investor_app.command("ideas")
def ideas_cmd(
    limit: int = typer.Option(25, "--limit", help="Maximum ideas to show"),
) -> None:
    """Top ACTIONABLE-grade names regardless of status, decision-first."""
    from app.logging import configure_logging
    from app.watchlist.store import watchlist_queue

    configure_logging()

    if limit <= 0:
        raise typer.BadParameter("--limit must be positive")

    rows = [
        row
        for row in watchlist_queue(limit=10_000)
        if str(row.get("conviction_grade") or "").upper() == "ACTIONABLE"
        and is_price_trigger_eligible(row)
    ]
    rows.sort(
        key=lambda row: (
            row["distance_from_buy_pct"]
            if row.get("distance_from_buy_pct") is not None
            else float("inf"),
            str(row.get("ticker") or ""),
        )
    )
    rows = rows[:limit]
    if not rows:
        typer.echo("No ACTIONABLE-grade ideas found.")
        return

    lines = ["# Ideas (ACTIONABLE conviction)"]
    for row in rows:
        status = str(row.get("status") or "")
        grade = row.get("conviction_grade")
        target = row.get("buy_price_target")
        current = row.get("latest_price")
        action = action_from_status_and_grade(
            status,
            grade,
            target,
            price_trigger_eligible=is_price_trigger_eligible(row),
        )
        block = DecisionBlock(
            action=action,
            current_price=current,
            buy_price_target=target,
            pct_to_target=pct_to_target(current, target),
            base_case_expected_return=None,
            conviction_grade=grade,
            confidence=row.get("confidence"),
            price_trigger_status=status,
            time_horizon=DEFAULT_TIME_HORIZON,
            what_would_change_my_mind=[],
            verdict_reconciliation_note=None,
        )
        lines.append("")
        lines.append(f"## {row['ticker']}")
        lines.append(render_decision_block_markdown(block))
    typer.echo("\n".join(lines))


@investor_app.command("memo")
def memo_cmd(
    ticker: str = typer.Argument(..., help="Ticker to render a decision block for"),
) -> None:
    """Print the decision-first block for one ticker from the watchlist."""
    from app.logging import configure_logging
    from app.watchlist.store import get_latest, get_latest_price

    configure_logging()

    entry = get_latest(ticker)
    if entry is None:
        raise typer.BadParameter(f"No watchlist entry found for {ticker.upper()}")

    current_price = entry.current_price_at_addition
    checked_at: str | None = None
    if entry.id is not None:
        snapshot = get_latest_price(entry.id)
        if snapshot is not None and isinstance(snapshot.get("price"), (int, float)):
            current_price = float(snapshot["price"])
            checked_at = snapshot.get("checked_at")

    price_eligible = is_price_trigger_eligible(entry)
    action = (
        action_from_status_and_grade(
            entry.status,
            entry.conviction_grade,
            entry.buy_price_target,
            price_trigger_eligible=True,
        )
        if price_eligible
        else "Research only"
    )
    presented_target = entry.buy_price_target if price_eligible else None
    block = DecisionBlock(
        action=action,
        current_price=current_price,
        buy_price_target=presented_target,
        pct_to_target=pct_to_target(current_price, presented_target),
        base_case_expected_return=None,
        conviction_grade=entry.conviction_grade,
        confidence=entry.confidence,
        price_trigger_status=entry.status,
        time_horizon=DEFAULT_TIME_HORIZON,
        what_would_change_my_mind=list(entry.falsifiers),
        verdict_reconciliation_note=None,
    )

    lines = render_decision_block_markdown(block).split("\n")
    lines.extend(_v2_provenance_lines(entry))
    if entry.event_pending:
        lines.append(
            f"- EVENT_PENDING: {entry.event_pending} — open corporate event "
            "undisposed; blocked from at-target presentation "
            "(`ivi events list --open`)"
        )
    if action == ACTION_REVIEW_AT_TARGET and not entry.event_pending:
        lines.append(
            buy_now_imperative_line(
                entry.ticker,
                current_price,
                entry.buy_price_target,
                None,
                entry.conviction_grade,
            )
        )
        if _is_snapshot_stale(checked_at):
            lines.append(
                f"- NOTE: price snapshot {checked_at} is older than "
                f"{_STALE_SNAPSHOT_TRADING_DAYS} trading days"
            )
    typer.echo("\n".join(lines))


@investor_app.command("watchlist")
def watchlist_cmd(
    limit: int = typer.Option(25, "--limit", help="Maximum rows to show"),
    sector: str | None = typer.Option(None, "--sector", help="Filter by source sector"),
    include_price_suspect: bool = typer.Option(
        False,
        "--include-price-suspect",
        help="Include rows currently flagged with suspect price data",
    ),
) -> None:
    """Thin wrapper over the watchlist review queue."""
    from app.logging import configure_logging
    from app.watchlist.store import watchlist_queue

    configure_logging()

    if limit <= 0:
        raise typer.BadParameter("--limit must be positive")

    rows = watchlist_queue(
        limit=limit, sector=sector, include_price_suspect=include_price_suspect
    )
    if not rows:
        typer.echo("No watchlist queue entries found.")
        return

    def _money(value: float | None) -> str:
        return "n/a" if value is None else f"${value:,.2f}"

    def _pct_points(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.1f}%"

    lines = [
        "| Ticker | Status | Conviction | Confidence | Latest Price | Buy Target | Distance From Buy | Pipeline | Disposition | Decision Basis | Validation | Band | Events | Source Sector |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        price_eligible = bool(row.get("price_trigger_eligible", True))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ticker"]),
                    str(row.get("presented_status") or row["status"] or "n/a"),
                    str(row["conviction_grade"] or "n/a"),
                    str(row["confidence"] or "n/a"),
                    _money(row["latest_price"]),
                    _money(row["buy_price_target"] if price_eligible else None),
                    _pct_points(row["distance_from_buy_pct"] if price_eligible else None),
                    str(row.get("pipeline_version") or "legacy"),
                    str(row.get("candidate_disposition") or "n/a"),
                    str(row.get("decision_basis") or "n/a"),
                    str(row.get("selection_validation_status") or "n/a"),
                    str(row.get("cap_band_label") or row.get("cap_band") or "UNKNOWN_CAP"),
                    str(row.get("event_pending") or "—"),
                    str(row["source_sector"] or "n/a"),
                ]
            )
            + " |"
        )
    typer.echo("\n".join(lines))
