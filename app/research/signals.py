from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.db import utc_now_iso
from app.research.schemas import EvidenceItem


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "quarter",
    "results",
    "announces",
    "report",
    "reports",
    "company",
    "inc",
    "corp",
    "ltd",
}

_SENTIMENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "guidance_raised": ("guidance raised", "raised guidance", "increased outlook"),
    "guidance_lowered": ("guidance lowered", "lowered guidance", "reduced outlook"),
    "restructuring": ("restructuring", "reorganization"),
    "investigation": ("investigation", "internal review"),
    "sec_subpoena": ("sec subpoena", "subpoena"),
    "bankruptcy": ("bankruptcy", "chapter 11", "chapter 7"),
    "going_concern": ("going concern",),
    "material_weakness": ("material weakness",),
}


class ResearchSignals(BaseModel):
    ticker: str
    as_of_date: str
    run_id: str
    recency_days_min: int | None = None
    item_count_30d: int = 0
    has_earnings_release: bool = False
    has_investor_presentation: bool = False
    sentiment_flags: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    evidence_item_ids: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    # A timestamp without an offset is UTC (how the pipeline stores them), not
    # the machine's local time: the same data must age the same everywhere.
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _known_by_as_of(item: EvidenceItem, as_of: datetime) -> bool:
    """Whether ``item`` existed by the end of the as-of day.

    Dated by its publication time when it has one. An undated item counts only
    if it was retrieved by then (a live run); in a backdated run it was fetched
    later and nothing shows it existed on the as-of day, so it is left out.
    """
    end_of_day = as_of + timedelta(days=1)
    published = _parse_ts(item.source_published_at)
    if published is not None:
        return published < end_of_day
    retrieved = _parse_ts(item.retrieved_at)
    return retrieved is not None and retrieved < end_of_day


def _as_of_dt(as_of_date: str) -> datetime:
    return datetime.combine(date.fromisoformat(as_of_date), datetime.min.time(), tzinfo=timezone.utc)


def _extract_topics(items: list[EvidenceItem]) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        text = (item.source_title or "").lower()
        for raw in _WORD_RE.findall(text):
            word = raw.lower()
            if word in _STOPWORDS:
                continue
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:10]]


def _extract_sentiment_flags(items: list[EvidenceItem]) -> list[str]:
    from app.parse.extractors.footnotes_signals import asserted_risk_excerpt

    flags: set[str] = set()
    for item in items:
        blob = f"{item.source_title or ''} {item.excerpt_text}".lower()
        for flag, patterns in _SENTIMENT_PATTERNS.items():
            if not any(pattern in blob for pattern in patterns):
                continue
            # A phrase hit is not a finding for these two: boilerplate such as
            # "risk that a material weakness exists" or "ongoing concern" must
            # not raise a risk flag (same rule as the filing extractor).
            if flag in {"going_concern", "material_weakness"} and (
                asserted_risk_excerpt(flag, blob) is None
            ):
                continue
            flags.add(flag)
    return sorted(flags)


def _age_in_days(published_at: str | None, retrieved_at: str | None, as_of: datetime) -> int | None:
    """Age of an evidence item at the as-of date, or None when it has none.

    Two items carry no age:

      * one PUBLISHED after the as-of calendar day (an item published at any time
        ON the as-of day is 0 days old) — it did not exist yet, and it does
        not fall back to the retrieval time either, because its own date is
        what is wrong; and
      * one with no published date at all — "an undated item carries no
        recency". Dating it by when we happened to fetch it is the wall clock,
        which in a backdated run is always later than the as-of date and used
        to clamp to "0 days old", making evidence of unknown age the freshest
        thing in the run.

    An earlier version of this fix kept a seven-day tolerance so that an
    undated item fetched the day after the as-of date still counted. That
    threshold appears in no specification and is not reinstated.
    """
    published = _parse_ts(published_at)
    # The as-of date is a calendar DAY, but ``as_of`` is its midnight: an item published
    # at noon on the as-of date existed by the end of that day, so it is not "future"
    # and it is 0 days old (never negative).
    if published is None or published >= as_of + timedelta(days=1):
        return None
    return max(0, int((as_of - published).total_seconds() // 86400))


def compute_research_signals(
    *,
    ticker: str,
    as_of_date: str,
    run_id: str,
    evidence_items: list[EvidenceItem],
) -> ResearchSignals:
    as_of = _as_of_dt(as_of_date)
    # Nothing published after the as-of day may move a signal (sentiment and
    # risk flags, topics, counts): it had not happened yet.
    evidence_items = [item for item in evidence_items if _known_by_as_of(item, as_of)]
    true_non_edgar = [item for item in evidence_items if item.source_type != "EDGAR"]
    non_edgar = true_non_edgar[:] if true_non_edgar else evidence_items

    ages: list[int] = []
    count_30d = 0
    has_earnings_release = False
    has_investor_presentation = False
    for item in non_edgar:
        days = _age_in_days(item.source_published_at, item.retrieved_at, as_of)
        if days is not None:
            ages.append(days)
            if days <= 30:
                count_30d += 1

        blob = f"{item.source_title or ''} {item.excerpt_text}".lower()
        if item.source_type == "sec_exhibit":
            if any(token in blob for token in ["earnings_release", "press release", "results of operations"]):
                has_earnings_release = True
            if any(token in blob for token in ["investor_presentation", "presentation", "slide"]):
                has_investor_presentation = True

    recency_days_min = min(ages) if ages else None
    sentiment_flags = _extract_sentiment_flags(non_edgar)
    key_topics = _extract_topics(non_edgar)
    evidence_ids = sorted({item.id for item in non_edgar})

    freshness_bucket = "UNKNOWN"
    if recency_days_min is not None:
        if recency_days_min <= 30:
            freshness_bucket = "RECENT"
        elif recency_days_min <= 180:
            freshness_bucket = "STALE"
        else:
            freshness_bucket = "VERY_STALE"

    flow_bucket = "UNKNOWN"
    if count_30d >= 3:
        flow_bucket = "ACTIVE"
    elif count_30d > 0:
        flow_bucket = "SPARSE"
    elif evidence_ids:
        flow_bucket = "QUIET"

    return ResearchSignals(
        ticker=ticker,
        as_of_date=as_of_date,
        run_id=run_id,
        recency_days_min=recency_days_min,
        item_count_30d=count_30d,
        has_earnings_release=has_earnings_release,
        has_investor_presentation=has_investor_presentation,
        sentiment_flags=sentiment_flags,
        key_topics=key_topics,
        evidence_item_ids=evidence_ids,
        summary={
            "freshness_bucket": freshness_bucket,
            "flow_bucket": flow_bucket,
            "no_non_edgar_coverage": len(true_non_edgar) == 0,
            "risk_flags_present": any(
                flag
                in {
                    "guidance_lowered",
                    "restructuring",
                    "investigation",
                    "sec_subpoena",
                    "bankruptcy",
                    "going_concern",
                    "material_weakness",
                }
                for flag in sentiment_flags
            ),
        },
    )


def persist_research_signals(conn, signals: ResearchSignals) -> None:
    conn.execute(
        """
        INSERT INTO research_signals(
            ticker, as_of_date, run_id, recency_days_min, item_count_30d,
            has_earnings_release, has_investor_presentation,
            sentiment_flags_json, key_topics_json, evidence_item_ids_json, summary_json, created_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, as_of_date, run_id) DO UPDATE SET
            recency_days_min=excluded.recency_days_min,
            item_count_30d=excluded.item_count_30d,
            has_earnings_release=excluded.has_earnings_release,
            has_investor_presentation=excluded.has_investor_presentation,
            sentiment_flags_json=excluded.sentiment_flags_json,
            key_topics_json=excluded.key_topics_json,
            evidence_item_ids_json=excluded.evidence_item_ids_json,
            summary_json=excluded.summary_json,
            created_at=excluded.created_at
        """,
        (
            signals.ticker,
            signals.as_of_date,
            signals.run_id,
            signals.recency_days_min,
            signals.item_count_30d,
            1 if signals.has_earnings_release else 0,
            1 if signals.has_investor_presentation else 0,
            json.dumps(signals.sentiment_flags),
            json.dumps(signals.key_topics),
            json.dumps(signals.evidence_item_ids),
            json.dumps(signals.summary),
            signals.created_at,
        ),
    )


def row_to_signals_dict(row: Any | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "ticker": row["ticker"],
        "as_of_date": row["as_of_date"],
        "run_id": row["run_id"],
        "recency_days_min": row["recency_days_min"],
        "item_count_30d": int(row["item_count_30d"] or 0),
        "has_earnings_release": bool(row["has_earnings_release"]),
        "has_investor_presentation": bool(row["has_investor_presentation"]),
        "sentiment_flags": json.loads(row["sentiment_flags_json"] or "[]"),
        "key_topics": json.loads(row["key_topics_json"] or "[]"),
        "evidence_item_ids": json.loads(row["evidence_item_ids_json"] or "[]"),
        "summary": json.loads(row["summary_json"] or "{}"),
        "created_at": row["created_at"],
    }
