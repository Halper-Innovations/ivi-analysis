"""Snapshot every emitted verdict into ticker_outcomes at decision time.

This is the entry point of the calibration/outcome loop: when the platform emits a
watchlist verdict, ``snapshot_decision`` records an OPEN ticker_outcomes row capturing
the entry price, grade, status, and benchmark so the return resolver can later close it
with realized + excess forward returns.

All emitted grades are recorded (ACTIONABLE/WATCHLIST_ONLY/AVOID/DATA_INCOMPLETE), not
only buys. The row is upserted on UNIQUE(ticker, as_of_date, run_id) so re-emitting the
same verdict in a run is idempotent and the latest grade wins.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.outcomes.store import add_outcome

if TYPE_CHECKING:
    # Import only for type-checking to avoid a runtime circular import:
    # app.watchlist.__init__ -> reevaluation -> store imports snapshot_decision
    # from this module. WatchlistEntry is used here solely as a type annotation
    # (the runtime body does attribute access only), so this stays under
    # TYPE_CHECKING and `from __future__ import annotations` keeps it a string.
    from app.watchlist.contract import WatchlistEntry


# Grade -> decision verdict for the ticker_outcomes ledger.
# ACTIONABLE is the only grade that maps to BUY; AVOID maps to PASS; everything
# else (WATCHLIST_ONLY, DATA_INCOMPLETE, unknown) is a WATCH.
_GRADE_TO_DECISION = {
    "ACTIONABLE": "BUY",
    "AVOID": "PASS",
}

# Confidence label -> conviction integer (clamped to 1-5).
_CONFIDENCE_TO_CONVICTION = {
    "HIGH": 4,
    "MODERATE": 3,
    "LOW": 2,
}

# Benchmark choice: SPY for large-cap names, IWM for
# small/mid (and micro/smid) names. The cap-appropriate symbol is stored
# per-outcome so the resolver/report stay benchmark-agnostic and the choice is
# auditable on every row.
BENCHMARK_LARGE_CAP = "SPY"
BENCHMARK_SMALL_MID_CAP = "IWM"

# market_cap_focus / market_cap_category tokens that mean "large cap" -> SPY.
# Anything else (small_cap, mid_cap, smid_cap, micro_cap, unknown/None) falls to
# the small/mid benchmark (IWM), which is the conservative default for a
# wait-for-correction tool that is overwhelmingly small/mid by mandate.
_LARGE_CAP_TOKENS = {
    "large_cap",
    "large",
    "large_cap_financials",
    "mega_cap",
    "mega",
    "large_and_mega",
}


def select_benchmark_symbol(
    market_cap_focus: str | None = None,
    market_cap_category: str | None = None,
) -> str:
    """Select the cap-appropriate benchmark symbol (SPY + IWM).

    Large-cap names benchmark against SPY; small/mid (and everything else,
    including unknown) benchmark against IWM. The per-name ``market_cap_category``
    wins when present; otherwise the sweep-level ``market_cap_focus`` decides.
    """
    for token in (market_cap_category, market_cap_focus):
        if token is None:
            continue
        normalized = str(token).strip().lower()
        if not normalized:
            continue
        return BENCHMARK_LARGE_CAP if normalized in _LARGE_CAP_TOKENS else BENCHMARK_SMALL_MID_CAP
    return BENCHMARK_SMALL_MID_CAP


def _decision_for_grade(grade: str | None) -> str:
    return _GRADE_TO_DECISION.get(str(grade or "").upper(), "WATCH")


def _conviction_for_confidence(confidence: str | None) -> int:
    conviction = _CONFIDENCE_TO_CONVICTION.get(str(confidence or "").upper(), 2)
    return max(1, min(5, conviction))


def snapshot_decision(
    entry: WatchlistEntry,
    *,
    run_id: str,
    as_of_date: str,
    grade: str | None = None,
    status: str | None = None,
    confidence: str | None = None,
    horizon_days: int = 365,
    benchmark_symbol: str | None = None,
    market_cap_focus: str | None = None,
    market_cap_category: str | None = None,
    db_path: Any | None = None,  # accepted for API symmetry; writes go via configured db
) -> dict[str, Any] | None:
    """Persist a single emitted verdict as an OPEN ticker_outcomes row.

    The benchmark is the cap-appropriate symbol (SPY for large-cap,
    IWM for small/mid) selected from ``market_cap_category`` / ``market_cap_focus``
    and stored per-outcome. An explicit ``benchmark_symbol`` overrides the
    cap-based selection (e.g. for tests or a forced benchmark).

    Returns the stored outcome row dict, or ``None`` when there is no usable entry
    price (``current_price_at_addition`` is None or <= 0) and nothing is written.
    """
    entry_price = entry.current_price_at_addition
    if entry_price is None or entry_price <= 0:
        return None

    resolved_grade = grade if grade is not None else entry.conviction_grade
    resolved_status = status if status is not None else entry.status
    resolved_confidence = confidence if confidence is not None else entry.confidence
    resolved_benchmark = (
        benchmark_symbol
        if benchmark_symbol is not None
        else select_benchmark_symbol(market_cap_focus, market_cap_category)
    )
    resolved_decision = _decision_for_grade(resolved_grade)
    if str(resolved_grade or "").upper() == "ACTIONABLE":
        from app.watchlist.contract import is_price_trigger_eligible

        if not is_price_trigger_eligible(entry):
            resolved_decision = "WATCH"

    return add_outcome(
        ticker=entry.ticker,
        as_of_date=as_of_date,
        run_id=run_id,
        decision=resolved_decision,
        conviction=_conviction_for_confidence(resolved_confidence),
        horizon_days=int(horizon_days),
        entry_price=float(entry_price),
        entry_price_source="watchlist_population",
        entry_date=as_of_date,
        grade=str(resolved_grade).upper() if resolved_grade else None,
        status=str(resolved_status).upper() if resolved_status else None,
        benchmark_symbol=resolved_benchmark,
        buy_price_target=entry.buy_price_target,
        pipeline_version=getattr(entry, "pipeline_version", None),
        candidate_disposition=getattr(entry, "candidate_disposition", None),
        decision_basis=getattr(entry, "decision_basis", None),
        selection_validation_status=getattr(
            entry, "selection_validation_status", None
        ),
        source_sector=getattr(entry, "source_sector", None),
    )
