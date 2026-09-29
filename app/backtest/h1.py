"""Scale-phase orchestrator: per-date point-in-time stratified sample -> backtest.

For each as-of date it builds the survivorship-aware eligible universe, draws a
stratified per-band sample, and snapshots the reconstructed signal via the
existing run_backtest. run_id is constant across dates (rows are unique per
ticker+date+run_id), so the whole grid is one resumable, idempotent run.
"""

from __future__ import annotations

import logging

from app.backtest.runner import BacktestRunSummary, run_backtest
from app.backtest.universe import eligible_universe_asof, stratified_sample

logger = logging.getLogger(__name__)

# Quarterly entry grid 2022-Q1 .. 2025-Q2 (first business-ish day of each quarter).
# All entries mature for 365d before 2026-06-05; the provider's fallback_days
# resolves any non-trading day.
H1_DATE_GRID = [
    "2022-01-03",
    "2022-04-01",
    "2022-07-01",
    "2022-10-03",
    "2023-01-03",
    "2023-04-03",
    "2023-07-03",
    "2023-10-02",
    "2024-01-02",
    "2024-04-01",
    "2024-07-01",
    "2024-10-01",
    "2025-01-02",
    "2025-04-01",
]


def run_h1(
    *,
    per_band: int = 250,
    seed: int = 42,
    dates=None,
    horizons=(90, 365),
    run_id: str = "h1_sample",
    provider=None,
    filing_lag_days: int = 90,
    db_path=None,
) -> BacktestRunSummary:
    """Build + snapshot the stratified sample across the date grid. Idempotent."""
    if provider is None:
        from app.market.price_provider import get_default_provider

        provider = get_default_provider()

    dates = list(dates or H1_DATE_GRID)
    horizons = list(horizons)
    agg = BacktestRunSummary()

    for as_of_date in dates:
        eligible = eligible_universe_asof(
            as_of_date,
            filing_lag_days=filing_lag_days,
            provider=provider,
            db_path=db_path,
        )
        sample = stratified_sample(eligible, per_band=per_band, seed=seed)
        unknown = sum(1 for e in eligible if e.cap_category_asof == "UNKNOWN_CAP")
        logger.info(
            "h1 %s: eligible=%d unknown_cap=%d sampled=%d",
            as_of_date,
            len(eligible),
            unknown,
            len(sample),
        )
        s = run_backtest(
            universe=sample,
            dates=[as_of_date],
            horizons=horizons,
            run_id=run_id,
            provider=provider,
            filing_lag_days=filing_lag_days,
        )
        agg.universe_size += s.universe_size
        agg.reconstructed += s.reconstructed
        agg.skipped += s.skipped
        agg.rows_written += s.rows_written
        agg.skipped_reasons.update({f"{as_of_date}:{k}": v for k, v in s.skipped_reasons.items()})

    logger.info(
        "h1 run %s complete: reconstructed=%d skipped=%d rows=%d",
        run_id,
        agg.reconstructed,
        agg.skipped,
        agg.rows_written,
    )
    return agg
