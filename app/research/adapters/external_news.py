from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse

from app.research.adapters.base import AdapterContext, AdapterResult, ResearchAdapter
from app.research.schemas import CitationRef


_WS_RE = re.compile(r"\s+")
_MAX_SUMMARY_CHARS = 1600


def _clean_text(value: object) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip()


def _parse_alpha_vantage_time(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y%m%dT%H%M", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def _as_of_end(as_of_date: str | None) -> datetime | None:
    """The last minute of the as-of day (UTC), or None when it is not a date."""
    try:
        day = datetime.strptime(str(as_of_date or "")[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return day.replace(hour=23, minute=59, tzinfo=timezone.utc)


def _request_url(symbol: str, api_key: str, limit: int, as_of_date: str | None = None) -> str:
    params: dict[str, object] = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "sort": "LATEST",
        "limit": max(1, int(limit)),
    }
    end = _as_of_end(as_of_date)
    if end is not None:
        # Ask only for news up to the as-of day; rows are re-checked below.
        params["time_to"] = end.strftime("%Y%m%dT%H%M")
    params["apikey"] = api_key
    return "https://www.alphavantage.co/query?" + urlencode(params)


def _source_domain(url: str) -> str | None:
    host = (urlparse(url).hostname or "").strip().lower()
    return host or None


def _sentiment_text(row: dict) -> str:
    label = _clean_text(row.get("overall_sentiment_label"))
    score = _clean_text(row.get("overall_sentiment_score"))
    if label and score:
        return f"Overall sentiment: {label} ({score})."
    if label:
        return f"Overall sentiment: {label}."
    return ""


class ExternalNewsAdapter(ResearchAdapter):
    source_type = "external_news"

    def enabled(self) -> bool:
        return (
            not self.cfg.safe_mode
            and self.cfg.research_external_news_enabled
            and str(self.cfg.research_external_news_provider or "").lower() == "alpha_vantage"
            and bool(self.cfg.research_alpha_vantage_api_key)
        )

    def collect(self, ctx: AdapterContext) -> AdapterResult:
        out = AdapterResult()
        if not self.enabled():
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_EXTERNAL_NEWS_DISABLED",
                    severity="medium",
                    summary="External news adapter is disabled or missing an Alpha Vantage API key.",
                    source_type=self.source_type,
                    recommended_action="Set VOE_SAFE_MODE=false, VOE_RESEARCH_EXTERNAL_NEWS_ENABLED=true, VOE_RESEARCH_EXTERNAL_NEWS_PROVIDER=alpha_vantage, and ALPHA_VANTAGE_API_KEY.",
                )
            )
            return out

        try:
            raw = self.http_get_bytes(
                _request_url(
                    ctx.ticker,
                    str(self.cfg.research_alpha_vantage_api_key or ""),
                    self.cfg.research_external_news_max_items,
                    ctx.as_of_date,
                ),
                use_cache=True,
                cache_ttl_seconds=6 * 3600,
            ).decode("utf-8", errors="ignore")
            payload = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_EXTERNAL_NEWS_FETCH_FAILED",
                    severity="medium",
                    summary=f"Failed to fetch external news for {ctx.ticker}: {exc}",
                    source_type=self.source_type,
                    recommended_action="Verify Alpha Vantage news access and rerun research.",
                )
            )
            return out

        if not isinstance(payload, dict):
            return out
        provider_message = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
        if provider_message:
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_EXTERNAL_NEWS_PROVIDER_MESSAGE",
                    severity="medium",
                    summary=f"External news provider returned a message for {ctx.ticker}: {_clean_text(provider_message)[:240]}",
                    source_type=self.source_type,
                    recommended_action="Check Alpha Vantage entitlement, rate limit, ticker, and external-news provider settings.",
                )
            )
            return out

        feed = payload.get("feed") if isinstance(payload.get("feed"), list) else []
        as_of_end = _as_of_end(ctx.as_of_date)
        seen_urls: set[str] = set()
        for row in feed:
            if not isinstance(row, dict):
                continue
            published_at = _parse_alpha_vantage_time(row.get("time_published"))
            # Point in time: an undated story, or one published after the
            # as-of day, cannot inform a view as of that day.
            if published_at is None or (
                as_of_end is not None and datetime.fromisoformat(published_at) > as_of_end
            ):
                continue
            url = _clean_text(row.get("url"))
            title = _clean_text(row.get("title"))
            summary = _clean_text(row.get("summary"))
            if not url or not title or not summary:
                continue
            url_key = url.lower()
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            source = _clean_text(row.get("source"))
            domain = _clean_text(row.get("source_domain")) or (_source_domain(url) or "")
            sentiment = _sentiment_text(row)
            excerpt_parts = [
                f"Source: {source or domain or 'unknown'}",
                f"Domain: {domain}" if domain else "",
                sentiment,
                summary,
            ]
            excerpt = "\n".join(part for part in excerpt_parts if part)[:_MAX_SUMMARY_CHARS]
            citation_text = f"{title}. {summary}"[:600]
            out.evidence_items.append(
                self._make_item(
                    ctx=ctx,
                    source_type=self.source_type,
                    source_url=url,
                    source_title=title,
                    source_published_at=published_at,
                    excerpt_text=excerpt,
                    citations=[
                        CitationRef(
                            source_url=url,
                            snippet=citation_text,
                            section_label=self.source_type,
                        )
                    ],
                )
            )
            if len(out.evidence_items) >= max(1, int(self.cfg.research_external_news_max_items)):
                break

        if not out.evidence_items:
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_EXTERNAL_NEWS_NOT_FOUND",
                    severity="medium",
                    summary=f"No parseable external news was found for {ctx.ticker}.",
                    source_type=self.source_type,
                    recommended_action="Try a wider provider source or verify Alpha Vantage news coverage for the ticker.",
                )
            )
        return out


__all__ = [
    "ExternalNewsAdapter",
]
