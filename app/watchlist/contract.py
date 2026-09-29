from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


WATCHLIST_ACTIVE_STATUSES = {
    "ACTIVE",
    "DEPLOY_READY",
    "BUY_CONFIRMED",
    "UNCERTAIN",
    "PRICE_DATA_SUSPECT",
    "QUARANTINE",
    "CONTRADICTED",
    "RESOLVED",
}
WATCHLIST_STATUSES = WATCHLIST_ACTIVE_STATUSES | {"REMOVED"}
WATCHLIST_CONVICTION_GRADES = {"ACTIONABLE", "WATCHLIST_ONLY", "DATA_INCOMPLETE", "AVOID"}
WATCHLIST_CONFIDENCE_LABELS = {"HIGH", "MODERATE", "LOW"}
WATCHLIST_CONVICTION_SOURCES = {
    "sector_final_decision",
    "company_autonomy",
    "relative_ranking",
    "manual",
    "corporate_event",
    "pearl_scan",  # legacy deep-pass rows written before the review ceiling
    # Literal provenance: which deep-pass subsystem produced the verdict.
    "pearl_deterministic",
    "pearl_analyst_review",
    # Autonomous-sector v2 deterministic screen provenance. A screen-sourced
    # row is a research-queue entry, never an investable recommendation.
    "sector_screen",
}
WATCHLIST_SCAN_FAMILIES = {"normal", "pearl"}

AUTONOMOUS_SECTOR_PIPELINE_V2 = "v2"
WATCHLIST_CANDIDATE_DISPOSITIONS = {
    "OUT_OF_SCOPE",
    "SCREENED_OUT",
    "NEEDS_DATA",
    "READY_FOR_UNDERWRITING",
    "UNDERWRITTEN",
}
WATCHLIST_DECISION_BASES = {
    "SCREEN",
    "UNDERWRITING",
    "VALIDATED_UNDERWRITING",
}
V2_PRICE_TRIGGER_STATUSES = {"ACTIVE", "DEPLOY_READY", "BUY_CONFIRMED"}


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    # sqlite3.Row is key-addressable but does not register as a Mapping.
    try:
        if name in value.keys():
            return value[name]
    except (AttributeError, KeyError, TypeError):
        pass
    return getattr(value, name, None)


def has_validated_underwriting_provenance(value: Any) -> bool:
    """Whether a v2 row has the evidence provenance required for ACTIONABLE."""

    return (
        str(_field(value, "candidate_disposition") or "").strip().upper() == "UNDERWRITTEN"
        and str(_field(value, "decision_basis") or "").strip().upper() == "VALIDATED_UNDERWRITING"
        and str(_field(value, "selection_validation_status") or "").strip().upper() == "VALIDATED"
        and str(_field(value, "conviction_grade") or "").strip().upper() == "ACTIONABLE"
    )


def is_price_trigger_eligible(value: Any) -> bool:
    """Whether a row may enter an at-target/investable presentation.

    Historical and non-sector-v2 rows retain their existing semantics. Sector
    v2 fails closed: only a validated, underwritten ACTIONABLE row can arm a
    price trigger. Keeping this predicate pure lets storage, trigger, digest,
    disposition, and CLI paths share the exact same rule.
    """

    pipeline_version = str(_field(value, "pipeline_version") or "").strip().lower()
    if pipeline_version != AUTONOMOUS_SECTOR_PIPELINE_V2:
        return True
    return (
        has_validated_underwriting_provenance(value)
        and str(_field(value, "status") or "").strip().upper() in V2_PRICE_TRIGGER_STATUSES
    )


@dataclass
class WatchlistEntry:
    ticker: str
    status: str
    source_run_id: str
    added_at: str
    conviction_grade: str | None = None
    confidence: str | None = None
    conviction_source: str | None = None
    scan_family: str = "normal"
    valuation_anchor_method: str | None = None
    valuation_anchor_value: float | None = None
    buy_price_target: float | None = None
    current_price_at_addition: float | None = None
    thesis_text: str | None = None
    key_risks: list[str] = field(default_factory=list)
    falsifiers: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    source_sector: str | None = None
    last_evaluated_at: str | None = None
    current_event_watermark: dict[str, Any] | None = None
    status_reason: str | None = None
    market_cap_mm: float | None = None
    cap_source: str | None = None
    cap_band: str | None = None
    cap_asof: str | None = None
    pipeline_version: str | None = None
    candidate_disposition: str | None = None
    decision_basis: str | None = None
    selection_validation_status: str | None = None
    # Read-only on entries: comma-joined EVENT_PENDING:<TYPE> flags owned by
    # the events feed (app/events/flags.py); add_or_update never writes it.
    event_pending: str | None = None
    id: int | None = None

    def normalized(self) -> "WatchlistEntry":
        return WatchlistEntry(
            id=self.id,
            ticker=self.ticker.upper(),
            status=self.status.upper(),
            conviction_grade=self.conviction_grade.upper() if self.conviction_grade else None,
            confidence=self.confidence.upper() if self.confidence else None,
            conviction_source=self.conviction_source.lower() if self.conviction_source else None,
            scan_family=self.scan_family.lower() if self.scan_family else "normal",
            valuation_anchor_method=self.valuation_anchor_method,
            valuation_anchor_value=self.valuation_anchor_value,
            buy_price_target=self.buy_price_target,
            current_price_at_addition=self.current_price_at_addition,
            thesis_text=self.thesis_text,
            key_risks=[str(item) for item in self.key_risks],
            falsifiers=[str(item) for item in self.falsifiers],
            open_questions=[str(item) for item in self.open_questions],
            source_run_id=self.source_run_id,
            source_sector=self.source_sector,
            added_at=self.added_at,
            last_evaluated_at=self.last_evaluated_at,
            current_event_watermark=(
                dict(self.current_event_watermark)
                if self.current_event_watermark is not None
                else None
            ),
            status_reason=self.status_reason,
            market_cap_mm=self.market_cap_mm,
            cap_source=self.cap_source.lower() if self.cap_source else None,
            cap_band=self.cap_band.lower() if self.cap_band else None,
            cap_asof=self.cap_asof,
            pipeline_version=(self.pipeline_version.lower() if self.pipeline_version else None),
            candidate_disposition=(
                self.candidate_disposition.upper() if self.candidate_disposition else None
            ),
            decision_basis=self.decision_basis.upper() if self.decision_basis else None,
            selection_validation_status=(
                self.selection_validation_status.upper()
                if self.selection_validation_status
                else None
            ),
            event_pending=self.event_pending,
        )

    def to_dict(self) -> dict[str, Any]:
        entry = self.normalized()
        return {
            "id": entry.id,
            "ticker": entry.ticker,
            "status": entry.status,
            "conviction_grade": entry.conviction_grade,
            "confidence": entry.confidence,
            "conviction_source": entry.conviction_source,
            "scan_family": entry.scan_family,
            "valuation_anchor_method": entry.valuation_anchor_method,
            "valuation_anchor_value": entry.valuation_anchor_value,
            "buy_price_target": entry.buy_price_target,
            "current_price_at_addition": entry.current_price_at_addition,
            "thesis_text": entry.thesis_text,
            "key_risks": list(entry.key_risks),
            "falsifiers": list(entry.falsifiers),
            "open_questions": list(entry.open_questions),
            "source_run_id": entry.source_run_id,
            "source_sector": entry.source_sector,
            "added_at": entry.added_at,
            "last_evaluated_at": entry.last_evaluated_at,
            "current_event_watermark": entry.current_event_watermark,
            "status_reason": entry.status_reason,
            "market_cap_mm": entry.market_cap_mm,
            "cap_source": entry.cap_source,
            "cap_band": entry.cap_band,
            "cap_asof": entry.cap_asof,
            "pipeline_version": entry.pipeline_version,
            "candidate_disposition": entry.candidate_disposition,
            "decision_basis": entry.decision_basis,
            "selection_validation_status": entry.selection_validation_status,
            "event_pending": entry.event_pending,
        }
