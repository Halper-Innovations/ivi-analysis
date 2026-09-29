"""Persistent watchlist support for sector-run candidates."""

from app.watchlist.contract import WatchlistEntry
from app.watchlist.schema import ensure_watchlist_schema
from app.watchlist.reevaluation import (
    WatchlistEvidenceReference,
    WatchlistRefreshSummary,
    WatchlistReevaluationResult,
    detect_new_evidence,
    refresh_watchlist,
    reevaluate_entry,
)
from app.watchlist.store import (
    WatchlistPopulationResult,
    add_or_update,
    backfill_confidence_from_artifacts,
    get_history,
    get_latest,
    list_active,
    mark_status,
    populate_from_sector_artifact,
    record_reevaluation_result,
    remove,
    stats,
)

__all__ = [
    "WatchlistEntry",
    "WatchlistEvidenceReference",
    "WatchlistPopulationResult",
    "WatchlistRefreshSummary",
    "WatchlistReevaluationResult",
    "add_or_update",
    "backfill_confidence_from_artifacts",
    "detect_new_evidence",
    "ensure_watchlist_schema",
    "get_history",
    "get_latest",
    "list_active",
    "mark_status",
    "populate_from_sector_artifact",
    "record_reevaluation_result",
    "refresh_watchlist",
    "reevaluate_entry",
    "remove",
    "stats",
]
