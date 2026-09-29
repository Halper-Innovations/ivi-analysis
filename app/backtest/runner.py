"""Backtest orchestrator: reconstruct -> snapshot a ticker_outcomes row per horizon.

Backtest rows are tagged entry_price_source='historical_backtest' and a
backtest run_id (f"{run_id}_h{horizon}") so they never mix with live decisions
in any report. The existing resolver closes them forward once matured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.backtest.reconstruct import reconstruct_signal_asof
from app.calibration.decision_ledger import select_benchmark_symbol
from app.outcomes.store import add_outcome

logger = logging.getLogger(__name__)

ENTRY_PRICE_SOURCE = "historical_backtest"


@dataclass
class BacktestRunSummary:
    universe_size: int = 0
    reconstructed: int = 0
    skipped: int = 0
    rows_written: int = 0
    skipped_reasons: dict[str, str] = field(default_factory=dict)


def _decision_for_signal(deploy_ready: bool) -> str:
    # The deterministic cheapness signal: a "buy" when cheap vs anchor, else "watch".
    return "BUY" if deploy_ready else "WATCH"


def run_backtest(*, universe, dates, horizons, run_id, provider=None, filing_lag_days=90):
    """Reconstruct + snapshot the signal for every (ticker, date) x horizon.

    Idempotent per (ticker, as_of_date, run_id): re-running overwrites. Returns a
    summary with per-ticker skip reasons (no silent truncation).
    """
    universe = list(universe)
    summary = BacktestRunSummary(universe_size=len(universe))
    for ticker in universe:
        any_valid = False
        first_skip_reason: str | None = None
        for as_of_date in dates:
            result = reconstruct_signal_asof(
                ticker, as_of_date, provider=provider, filing_lag_days=filing_lag_days
            )
            if result.signal is None:
                if first_skip_reason is None:
                    first_skip_reason = result.skip_reason or "UNKNOWN"
                continue
            any_valid = True
            sig = result.signal
            benchmark = select_benchmark_symbol(market_cap_category=sig.cap_category)
            for horizon in horizons:
                add_outcome(
                    ticker=sig.ticker,
                    as_of_date=sig.as_of_date,
                    run_id=f"{run_id}_h{horizon}",
                    decision=_decision_for_signal(sig.deploy_ready),
                    conviction=2,
                    horizon_days=int(horizon),
                    entry_price=sig.price,
                    entry_price_source=ENTRY_PRICE_SOURCE,
                    entry_date=sig.as_of_date,
                    grade="DEPLOY_READY" if sig.deploy_ready else "NOT_CHEAP",
                    status=sig.expectations_gap_bucket,
                    benchmark_symbol=benchmark,
                    buy_price_target=sig.buy_price_target,
                    cap_category=sig.cap_category,
                )
                summary.rows_written += 1
        if any_valid:
            summary.reconstructed += 1
        else:
            summary.skipped += 1
            summary.skipped_reasons[ticker] = first_skip_reason or "UNKNOWN"
    logger.info(
        "backtest run %s: reconstructed=%d skipped=%d rows=%d",
        run_id, summary.reconstructed, summary.skipped, summary.rows_written,
    )
    return summary
