"""Resolve realized + benchmark + excess forward returns for matured outcomes.

This closes the calibration/outcome loop: for every OPEN ``ticker_outcomes`` row
that has reached its ``entry_date + horizon_days`` mark, fetch the exit price (and
the cap-appropriate benchmark's entry/exit prices) from the date-aware price
provider, compute realized, benchmark, and excess returns, and mark the row CLOSED.

Rows whose maturity date is still in the future are left OPEN (not matured). Rows
that have matured but for which the provider cannot resolve an exit price are also
left OPEN (no price) so they can be retried on a later resolve pass.

The actual multi-year price backfill RUN is an owner-run manual step (network);
this module is tested with an injected fake provider and fixtures only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.db import get_db, utc_now_iso
from app.outcomes.lineage import (
    outcome_row_is_decision_eligible,
    refresh_outcome_integrity_fingerprint,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolveSummary:
    eligible: int = 0
    closed: int = 0
    skipped_not_matured: int = 0
    skipped_no_price: int = 0
    skipped_unauthorized: int = 0
    unresolved: int = 0
    # Audit coverage gap 4: live-quote entry prices (raw basis) against
    # adjusted-series exits mix dividend/split bases. The resolver QUANTIFIES
    # the divergence (count + log) without altering return math — backtest
    # rows are basis-consistent by construction (same provider series).
    dividend_basis_mismatches: int = 0


def _realized_pct(entry: float, exit_: float) -> float:
    """Forward return in percent: (exit - entry) / entry * 100."""
    return (exit_ - entry) / entry * 100.0


def _excess_pct(realized: float, benchmark: float) -> float:
    """Excess return over benchmark in percentage points: realized - benchmark."""
    return realized - benchmark


def _target_exit_date(entry_date: str, horizon_days: int) -> str:
    parsed = datetime.strptime(entry_date, "%Y-%m-%d").date()
    return (parsed + timedelta(days=int(horizon_days))).isoformat()


def resolve_open_outcomes(
    as_of_date: str,
    *,
    provider: Any | None = None,
    db_path: str | Path | None = None,  # accepted for API symmetry; writes go via configured db
    unresolved_grace_days: int | None = None,
) -> ResolveSummary:
    """Close every matured OPEN outcome with realized/benchmark/excess returns.

    For each OPEN row with a usable entry price and entry date, compute the
    target exit date (``entry_date + horizon_days``). Rows whose target exit is
    after ``as_of_date`` are left OPEN (not matured). Otherwise the provider is
    asked for the ticker's exit price and the benchmark's entry/exit prices; if
    the exit price is missing the row is left OPEN (no price), else the row is
    closed with the computed returns.

    When ``unresolved_grace_days`` is None (default) a matured row whose exit
    price cannot be fetched is left OPEN for a later retry pass — preserving the
    legacy behavior. When it is set, such a row is instead closed with
    ``outcome_status='UNRESOLVED'`` once ``target_exit + unresolved_grace_days``
    has passed ``as_of_date`` (grace=0 ⇒ mark on first matured pass). UNRESOLVED
    rows carry no realized return and are intentionally excluded from the CLOSED
    return/edge aggregation; the backtest survivorship bound counts them separately to
    quantify the delisting exposure.
    """
    if provider is None:
        from app.market.price_provider import get_default_provider

        provider = get_default_provider()

    summary = ResolveSummary()
    now = utc_now_iso()

    with get_db() as conn:
        rows = conn.execute(
            """
            WITH latest_lineage AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, run_id
                           ORDER BY updated_at DESC, id DESC
                       ) AS lineage_rank
                FROM ticker_outcomes
            )
            SELECT *
            FROM latest_lineage
            WHERE lineage_rank = 1
              AND outcome_status = 'OPEN'
              AND entry_price IS NOT NULL
              AND entry_price > 0
              AND entry_date IS NOT NULL
            """
        ).fetchall()

        for row in rows:
            # Selection happens in SQL first.  Only then may the exact source
            # run authorize resolution; filtering candidates earlier would
            # allow a superseded older decision to re-enter the loop.
            if not outcome_row_is_decision_eligible(row):
                summary.skipped_unauthorized += 1
                continue
            summary.eligible += 1
            entry_price = float(row["entry_price"])
            entry_date = str(row["entry_date"])
            horizon_days = int(row["horizon_days"])
            benchmark_symbol = row["benchmark_symbol"]
            buy_price_target = row["buy_price_target"]

            target_exit = _target_exit_date(entry_date, horizon_days)
            if target_exit > str(as_of_date):
                summary.skipped_not_matured += 1
                continue

            exit_snap = provider.get_price_asof(row["ticker"], target_exit)
            if exit_snap is None or not isinstance(exit_snap.price, (int, float)):
                summary.skipped_no_price += 1
                # Delisting-aware: a matured row the provider can't price is recorded
                # as UNRESOLVED (not silently left OPEN) so the survivorship bound can count it.
                if unresolved_grace_days is not None:
                    grace_cutoff = _target_exit_date(target_exit, int(unresolved_grace_days))
                    if grace_cutoff <= str(as_of_date):
                        from app.outcomes.store import archive_outcome_row

                        archive_outcome_row(conn, int(row["id"]))
                        conn.execute(
                            "UPDATE ticker_outcomes SET outcome_status = 'UNRESOLVED', "
                            "updated_at = ? WHERE id = ?",
                            (now, int(row["id"])),
                        )
                        if not refresh_outcome_integrity_fingerprint(conn, int(row["id"])):
                            raise RuntimeError(
                                "outcome source authorization changed while marking unresolved"
                            )
                        summary.unresolved += 1
                continue

            exit_price = float(exit_snap.price)
            realized = _realized_pct(entry_price, exit_price)

            # Dividend/split basis check for live-quote entries: the ledger
            # entry is a raw quote while exits come from the (dividend/split-)
            # adjusted series. A >2% gap vs the adjusted price at entry_date
            # means realized returns mix bases — counted for the diagnostic.
            entry_source = (
                str(row["entry_price_source"] or "") if "entry_price_source" in row.keys() else ""
            )
            if entry_source and entry_source != "historical_backtest":
                basis_snap = provider.get_price_asof(str(row["ticker"]), entry_date)
                if (
                    basis_snap is not None
                    and isinstance(basis_snap.price, (int, float))
                    and float(basis_snap.price) > 0
                    and abs(float(basis_snap.price) / entry_price - 1.0) > 0.02
                ):
                    summary.dividend_basis_mismatches += 1
                    logger.warning(
                        "DIVIDEND_BASIS_MISMATCH %s entry_date=%s ledger=%.4f adjusted=%.4f",
                        row["ticker"],
                        entry_date,
                        entry_price,
                        float(basis_snap.price),
                    )

            # Reached-buy-target hit metric. The
            # wait-for-correction trigger fires when the exit price has fallen to
            # or below the buy target. NULL when no target was recorded.
            reached_buy_target: int | None = None
            if isinstance(buy_price_target, (int, float)) and float(buy_price_target) > 0:
                reached_buy_target = 1 if exit_price <= float(buy_price_target) else 0

            benchmark_return: float | None = None
            excess: float | None = None
            if benchmark_symbol:
                bench_entry = provider.get_price_asof(benchmark_symbol, entry_date)
                bench_exit = provider.get_price_asof(benchmark_symbol, target_exit)
                if (
                    bench_entry is not None
                    and bench_exit is not None
                    and isinstance(bench_entry.price, (int, float))
                    and isinstance(bench_exit.price, (int, float))
                    and float(bench_entry.price) > 0
                ):
                    benchmark_return = _realized_pct(
                        float(bench_entry.price), float(bench_exit.price)
                    )
                    excess = _excess_pct(realized, benchmark_return)

            from app.outcomes.store import archive_outcome_row

            archive_outcome_row(conn, int(row["id"]))
            conn.execute(
                """
                UPDATE ticker_outcomes
                SET outcome_status = 'CLOSED',
                    close_date = ?,
                    realized_return_pct = ?,
                    benchmark_return_pct = ?,
                    excess_return_pct = ?,
                    reached_buy_target = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    exit_snap.as_of_date,
                    realized,
                    benchmark_return,
                    excess,
                    reached_buy_target,
                    now,
                    int(row["id"]),
                ),
            )
            if not refresh_outcome_integrity_fingerprint(conn, int(row["id"])):
                raise RuntimeError("outcome source authorization changed while resolving row")
            summary.closed += 1

    return summary
