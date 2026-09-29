"""Automated deep research outcome tracker.

Evaluates persisted deep-research theses against subsequent price reality.
Report-level and method-level outcome tracking.

Pure evaluation functions + thin orchestration layer.
No dependency on app/outcomes/ or the perception resolver.

Public API: scan_research_outcomes(scan_date, horizon_days) -> ScanResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.research.deep_research import ResearchReport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Report-level verdict thresholds (applied to adjusted_margin_of_safety)
MOS_UNDERVALUED_THRESHOLD = 0.03
MOS_OVERVALUED_THRESHOLD = -0.03

# Method-level neutral band (applied to (predicted - entry) / entry)
METHOD_NEUTRAL_BAND = 0.03

# Directional resolution thresholds (applied to price_change_pct)
CONFIRM_THRESHOLD_PCT = 10.0
DISCONFIRM_THRESHOLD_PCT = 15.0

# Supported valuation methods
SUPPORTED_METHODS: set[str] = {"dcf", "epv", "graham"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReportOutcome:
    thesis_verdict: str  # UNDERVALUED / OVERVALUED / FAIRLY_VALUED
    verdict_outcome: str  # CORRECT / INCORRECT / INCONCLUSIVE
    price_change_pct: float
    conviction_score: int | None
    conviction_class: str | None


@dataclass
class MethodOutcome:
    method: str  # dcf / epv / graham
    predicted_value: float
    direction: str  # UNDERVALUED / OVERVALUED / FAIRLY_VALUED
    outcome: str  # CORRECT / INCORRECT / INCONCLUSIVE
    unadjusted_source: bool  # True for graham


@dataclass
class TickerOutcomeSummary:
    ticker: str
    source_as_of_date: str
    target_exit_date: str
    exit_as_of_date: str  # actual price snapshot date (may differ from target)
    thesis_verdict: str
    verdict_outcome: str
    price_change_pct: float
    conviction_class: str | None
    methods_evaluated: int


@dataclass
class ScanResult:
    scan_date: str
    horizon_days: int
    total_eligible: int
    evaluated: int
    skipped_no_price: int
    outcomes: list[TickerOutcomeSummary]


# ---------------------------------------------------------------------------
# Pure Evaluators
# ---------------------------------------------------------------------------


def _derive_thesis_verdict(adjusted_mos: float) -> str:
    """Derive directional verdict from adjusted margin of safety."""
    if adjusted_mos > MOS_UNDERVALUED_THRESHOLD:
        return "UNDERVALUED"
    if adjusted_mos < MOS_OVERVALUED_THRESHOLD:
        return "OVERVALUED"
    return "FAIRLY_VALUED"


def _resolve_direction(direction: str, price_change_pct: float) -> str:
    """Shared directional resolution for report and method evaluators."""
    if direction == "FAIRLY_VALUED":
        return "INCONCLUSIVE"
    if direction == "UNDERVALUED":
        if price_change_pct > CONFIRM_THRESHOLD_PCT:
            return "CORRECT"
        if price_change_pct < -DISCONFIRM_THRESHOLD_PCT:
            return "INCORRECT"
        return "INCONCLUSIVE"
    if direction == "OVERVALUED":
        if price_change_pct < -CONFIRM_THRESHOLD_PCT:
            return "CORRECT"
        if price_change_pct > DISCONFIRM_THRESHOLD_PCT:
            return "INCORRECT"
        return "INCONCLUSIVE"
    return "INCONCLUSIVE"


def _method_direction(predicted_value: float, entry_price: float) -> str:
    """Derive method-level direction from predicted value vs entry price."""
    ratio = (predicted_value - entry_price) / entry_price
    if ratio > METHOD_NEUTRAL_BAND:
        return "UNDERVALUED"
    if ratio < -METHOD_NEUTRAL_BAND:
        return "OVERVALUED"
    return "FAIRLY_VALUED"


def evaluate_report_outcome(
    report: ResearchReport,
    entry_price: float,
    exit_price: float,
) -> ReportOutcome:
    """Pure report-level evaluation. No DB or I/O."""
    price_change_pct = ((exit_price - entry_price) / entry_price) * 100.0
    verdict = _derive_thesis_verdict(report.thesis.adjusted_margin_of_safety)
    outcome = _resolve_direction(verdict, price_change_pct)
    return ReportOutcome(
        thesis_verdict=verdict,
        verdict_outcome=outcome,
        price_change_pct=price_change_pct,
        conviction_score=report.conviction_score,
        conviction_class=report.conviction_class,
    )


def evaluate_method_outcome(
    method: str,
    predicted_value: float,
    entry_price: float,
    exit_price: float,
    *,
    unadjusted_source: bool = False,
) -> MethodOutcome:
    """Pure method-level evaluation. No DB or I/O."""
    price_change_pct = ((exit_price - entry_price) / entry_price) * 100.0
    direction = _method_direction(predicted_value, entry_price)
    outcome = _resolve_direction(direction, price_change_pct)
    return MethodOutcome(
        method=method,
        predicted_value=predicted_value,
        direction=direction,
        outcome=outcome,
        unadjusted_source=unadjusted_source,
    )


def evaluate_methods(
    report: ResearchReport,
    entry_price: float,
    exit_price: float,
) -> list[MethodOutcome]:
    """Evaluate all available methods from a report. Returns list of MethodOutcome."""
    results = []
    thesis = report.thesis
    method_values = [
        ("dcf", thesis.adjusted_dcf, False),
        ("epv", thesis.adjusted_epv, False),
        ("graham", thesis.original_graham, True),
    ]
    for method, value, unadjusted in method_values:
        if not isinstance(value, (int, float)):
            continue
        results.append(
            evaluate_method_outcome(
                method,
                float(value),
                entry_price,
                exit_price,
                unadjusted_source=unadjusted,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Orchestration (DB + price fetch)
# ---------------------------------------------------------------------------

import json
import logging
from datetime import date, timedelta

from app.db import get_db, utc_now_iso
from app.valuation.lineage import valuation_row_is_decision_eligible

logger = logging.getLogger(__name__)


def _compute_target_exit_date(source_as_of_date: str, horizon_days: int) -> str:
    """Return ISO date string: source_as_of_date + horizon_days."""
    return (date.fromisoformat(source_as_of_date) + timedelta(days=horizon_days)).isoformat()


def _load_eligible_reports(
    scan_date: str,
    horizon_days: int,
) -> list[tuple[ResearchReport, int, str]]:
    """Load eligible deep-research reports from DB.

    Returns list of (report, valuation_id, as_of_date) for reports that pass
    ALL eligibility checks: status OK, thesis present, usable entry price,
    usable adjusted_intrinsic_mid, target_exit_date <= scan_date.
    """
    from app.research.deep_research import ResearchReport as _RR

    cutoff = (date.fromisoformat(scan_date) - timedelta(days=horizon_days)).isoformat()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT *
               FROM valuations
               WHERE method = 'deep_research'
                 AND as_of_date <= ?
               ORDER BY as_of_date""",
            (cutoff,),
        ).fetchall()

    results = []
    for row in rows:
        if not valuation_row_is_decision_eligible(row):
            continue
        try:
            data = json.loads(row["outputs_json"])
            report = _RR.from_dict(data)
        except Exception:
            continue

        if report.status != "OK":
            continue
        if report.thesis is None:
            continue
        if (
            not isinstance(report.thesis.current_price, (int, float))
            or report.thesis.current_price <= 0
        ):
            continue
        if not isinstance(report.thesis.adjusted_intrinsic_mid, (int, float)):
            continue
        if not isinstance(report.thesis.adjusted_margin_of_safety, (int, float)):
            continue

        results.append((report, row["id"], row["as_of_date"]))

    return results


def _persist_outcome(
    conn,
    ticker: str,
    source_as_of_date: str,
    target_exit_date: str,
    scan_date: str,
    horizon_days: int,
    entry_price: float,
    exit_price: float,
    exit_as_of_date: str,
    exit_source: str,
    report_outcome: ReportOutcome,
    method_outcomes: list[MethodOutcome],
    *,
    source_valuation_id: int | None = None,
) -> None:
    """Persist parent + child outcome rows. Upserts on rerun."""
    now = utc_now_iso()
    source_run_id = None
    source_artifact_path = None
    source_artifact_sha256 = None
    source_valuation_fingerprint = None
    if source_valuation_id is not None:
        source_row = conn.execute(
            "SELECT * FROM valuations WHERE id = ?",
            (int(source_valuation_id),),
        ).fetchone()
        if source_row is not None and valuation_row_is_decision_eligible(source_row):
            source_run_id = source_row["source_run_id"]
            source_artifact_path = source_row["source_artifact_path"]
            source_artifact_sha256 = source_row["source_artifact_sha256"]
            source_valuation_fingerprint = source_row["financial_integrity_fingerprint"]
    conn.execute(
        """INSERT OR REPLACE INTO deep_research_outcomes
           (ticker, source_as_of_date, source_valuation_id, source_run_id,
            source_artifact_path, source_artifact_sha256,
            source_valuation_fingerprint,
            horizon_days, target_exit_date, scan_date,
            entry_price, exit_price,
            exit_as_of_date, exit_source,
            price_change_pct,
            thesis_verdict, verdict_outcome, conviction_score, conviction_class,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker,
            source_as_of_date,
            source_valuation_id,
            source_run_id,
            source_artifact_path,
            source_artifact_sha256,
            source_valuation_fingerprint,
            horizon_days,
            target_exit_date,
            scan_date,
            entry_price,
            exit_price,
            exit_as_of_date,
            exit_source,
            report_outcome.price_change_pct,
            report_outcome.thesis_verdict,
            report_outcome.verdict_outcome,
            report_outcome.conviction_score,
            report_outcome.conviction_class,
            now,
        ),
    )
    parent_id = conn.execute(
        "SELECT id FROM deep_research_outcomes WHERE ticker = ? AND source_as_of_date = ? AND horizon_days = ?",
        (ticker, source_as_of_date, horizon_days),
    ).fetchone()["id"]

    for mo in method_outcomes:
        conn.execute(
            """INSERT OR REPLACE INTO deep_research_method_outcomes
               (outcome_id, method, predicted_value, entry_price,
                direction, outcome, unadjusted_source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                parent_id,
                mo.method,
                mo.predicted_value,
                entry_price,
                mo.direction,
                mo.outcome,
                1 if mo.unadjusted_source else 0,
                now,
            ),
        )


def _get_exit_snapshot(ticker: str, target_exit_date: str):
    """Fetch exit price snapshot via market PriceProvider.

    Returns PriceSnapshot or None. The snapshot's as_of_date may differ from
    target_exit_date due to non-trading-day fallback.
    """
    try:
        from app.market.price_provider import get_default_provider

        snapshot = get_default_provider().get_price_asof(ticker, target_exit_date)
        if snapshot is not None and snapshot.price > 0:
            return snapshot
    except Exception as exc:
        logger.warning(
            "outcome_tracker: price fetch failed for %s @ %s: %s", ticker, target_exit_date, exc
        )
    return None


def scan_research_outcomes(
    scan_date: str,
    horizon_days: int = 90,
) -> ScanResult:
    """Scan eligible deep-research theses and evaluate against price reality.

    scan_date: controls eligibility cutoff -- theses whose target_exit_date
               is still in the future are skipped.
    horizon_days: fixed holding period. Exit price is looked up at
                  source_as_of_date + horizon_days, not at scan_date.

    Raises ValueError if horizon_days <= 0.
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    eligible = _load_eligible_reports(scan_date, horizon_days)
    outcomes: list[TickerOutcomeSummary] = []
    skipped = 0

    with get_db() as conn:
        for report, valuation_id, as_of_date in eligible:
            entry_price = float(report.thesis.current_price)
            target_exit = _compute_target_exit_date(as_of_date, horizon_days)
            snapshot = _get_exit_snapshot(report.ticker, target_exit)

            if snapshot is None:
                skipped += 1
                continue

            exit_price = snapshot.price
            report_outcome = evaluate_report_outcome(report, entry_price, exit_price)
            method_outcomes = evaluate_methods(report, entry_price, exit_price)

            _persist_outcome(
                conn,
                report.ticker,
                as_of_date,
                target_exit,
                scan_date,
                horizon_days,
                entry_price,
                exit_price,
                snapshot.as_of_date,
                snapshot.source,
                report_outcome,
                method_outcomes,
                source_valuation_id=valuation_id,
            )

            outcomes.append(
                TickerOutcomeSummary(
                    ticker=report.ticker,
                    source_as_of_date=as_of_date,
                    target_exit_date=target_exit,
                    exit_as_of_date=snapshot.as_of_date,
                    thesis_verdict=report_outcome.thesis_verdict,
                    verdict_outcome=report_outcome.verdict_outcome,
                    price_change_pct=report_outcome.price_change_pct,
                    conviction_class=report_outcome.conviction_class,
                    methods_evaluated=len(method_outcomes),
                )
            )

    return ScanResult(
        scan_date=scan_date,
        horizon_days=horizon_days,
        total_eligible=len(eligible),
        evaluated=len(outcomes),
        skipped_no_price=skipped,
        outcomes=outcomes,
    )
