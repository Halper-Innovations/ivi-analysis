from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import AppConfig, get_config
from app.research.schemas import EvidenceGap, ResearchPacket, ResearchQuality


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _coverage_score(packet: ResearchPacket, min_items_per_section: int = 1) -> float:
    sections = {
        "findings": packet.findings,
        "risks": packet.risks,
        "catalysts": packet.catalysts,
        "disconfirming_evidence": packet.disconfirming_evidence,
    }
    met = 0
    for entries in sections.values():
        if sum(1 for entry in entries if len(entry.evidence_item_ids) >= 1) >= min_items_per_section:
            met += 1
    return round((met / max(1, len(sections))) * 100.0, 2)


def _age_in_days(published_at: str | None, anchor: datetime) -> float | None:
    """Same rule as app/research/signals.py._age_in_days: an item carries an age
    only when it states a published date at or before the as-of date. An undated
    item carries no recency, and a future-dated one carries none either."""
    published = _parse_dt(published_at)
    if published is None:
        return None
    published = published.astimezone(timezone.utc)
    # The as-of date is a calendar DAY but the anchor is its midnight: an item published
    # later on the as-of day is 0 days old; only the next day or later is excluded.
    if published >= anchor + timedelta(days=1):
        return None
    return max(0.0, (anchor - published).total_seconds() / 86400.0)


def _freshness_score(packet: ResearchPacket) -> float:
    # Anchored to the packet's own as-of date, not the calendar day the job
    # runs: two identical packets scored a week apart used to get different
    # freshness scores, and the score is persisted and shown. A timestamp after
    # the as-of date carries no age (same rule as compute_research_signals).
    anchor = _parse_dt(packet.as_of_date) or datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    ages_days: list[float] = []
    for item in packet.evidence_items:
        days = _age_in_days(item.source_published_at, anchor)
        if days is not None:
            ages_days.append(days)

    if not ages_days:
        return 30.0
    avg = sum(ages_days) / len(ages_days)
    if avg <= 7:
        return 100.0
    if avg <= 30:
        return 85.0
    if avg <= 90:
        return 65.0
    if avg <= 180:
        return 45.0
    if avg <= 365:
        return 30.0
    return 15.0


def _gap_score(gaps: list[EvidenceGap]) -> float:
    weights = {"high": 18.0, "medium": 10.0, "low": 4.0}
    penalty = 0.0
    for gap in gaps:
        penalty += weights.get(gap.severity, 8.0)
    return round(max(0.0, 100.0 - penalty), 2)


def score_research_packet(packet: ResearchPacket, cfg: AppConfig | None = None) -> ResearchQuality:
    cfg = cfg or get_config()
    coverage = _coverage_score(packet)
    freshness = _freshness_score(packet)
    gap = _gap_score(packet.evidence_gaps)
    overall = round((0.45 * coverage) + (0.25 * freshness) + (0.30 * gap), 2)
    top_gaps = sorted(
        packet.evidence_gaps,
        key=lambda g: {"high": 0, "medium": 1, "low": 2}.get(g.severity, 3),
    )[:5]
    return ResearchQuality(
        coverage_score=coverage,
        freshness_score=round(freshness, 2),
        gap_score=gap,
        overall_research_score=overall,
        incomplete=overall < float(cfg.research_quality_threshold),
        top_gaps=top_gaps,
    )


def quality_summary_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        return None
    return quality
