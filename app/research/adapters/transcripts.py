from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from urllib.parse import urlencode

from app.research.adapters.base import AdapterContext, AdapterResult, ResearchAdapter
from app.research.schemas import CitationRef


_WS_RE = re.compile(r"\s+")
_TRANSCRIPT_MAX_CHARS = 4000


def _clean_text(value: object) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip()


def _quarter_end(year: int, quarter: int) -> date:
    month = quarter * 3
    day = 31 if month in {3, 12} else 30
    return date(year, month, day)


def _calendar_quarter(value: date) -> tuple[int, int]:
    return value.year, ((value.month - 1) // 3) + 1


def _previous_quarter(year: int, quarter: int) -> tuple[int, int]:
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1


def _recent_quarters(as_of_date: str, max_quarters: int) -> list[str]:
    try:
        anchor = date.fromisoformat(as_of_date)
    except ValueError:
        anchor = date.today()
    year, quarter = _calendar_quarter(anchor)
    out: list[str] = []
    for _ in range(max(1, int(max_quarters))):
        out.append(f"{year}Q{quarter}")
        year, quarter = _previous_quarter(year, quarter)
    return out


# With no call date from the provider, a transcript is dated this long after
# its fiscal quarter ends: calls are held some weeks after quarter end, so the
# quarter-end date would put the call's content before it was spoken.
_UNDATED_CALL_LAG_DAYS = 45
_CALL_DATE_FIELDS = ("call_date", "date", "published_at", "time_published", "reportedDate")


def _call_date(payload: dict) -> date | None:
    for field in _CALL_DATE_FIELDS:
        text = str(payload.get(field) or "").strip()
        if not text:
            continue
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
        try:
            return datetime.strptime(text[:8], "%Y%m%d").date()  # 20260416T1630
        except ValueError:
            continue
    return None


def _quarter_published_at(quarter: str) -> str | None:
    """The conservative date for a transcript with no call date: quarter end + 45 days."""
    match = re.fullmatch(r"(\d{4})Q([1-4])", str(quarter).strip())
    if not match:
        return None
    end = _quarter_end(int(match.group(1)), int(match.group(2)))
    return (end + timedelta(days=_UNDATED_CALL_LAG_DAYS)).isoformat()


def _source_url(symbol: str, quarter: str) -> str:
    return "https://www.alphavantage.co/query?" + urlencode(
        {
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": symbol,
            "quarter": quarter,
        }
    )


def _request_url(symbol: str, quarter: str, api_key: str) -> str:
    return "https://www.alphavantage.co/query?" + urlencode(
        {
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": symbol,
            "quarter": quarter,
            "apikey": api_key,
        }
    )


def _segment_text(segment: object) -> str:
    if isinstance(segment, str):
        return _clean_text(segment)
    if not isinstance(segment, dict):
        return ""
    speaker = _clean_text(segment.get("speaker") or segment.get("name"))
    title = _clean_text(segment.get("title") or segment.get("role"))
    content = _clean_text(
        segment.get("content")
        or segment.get("text")
        or segment.get("paragraph")
        or segment.get("transcript")
        or segment.get("speech")
    )
    sentiment = _clean_text(segment.get("sentiment"))
    if not content:
        return ""
    label_parts = [part for part in [speaker, title] if part]
    label = f"{' / '.join(label_parts)}: " if label_parts else ""
    suffix = f" [sentiment={sentiment}]" if sentiment else ""
    return f"{label}{content}{suffix}"


def _transcript_text(payload: dict) -> str:
    transcript = payload.get("transcript")
    if isinstance(transcript, list):
        parts = [_segment_text(item) for item in transcript]
        return "\n\n".join(part for part in parts if part).strip()
    if isinstance(transcript, str):
        return _clean_text(transcript)
    content = payload.get("content") or payload.get("text")
    if isinstance(content, str):
        return _clean_text(content)
    return ""


class TranscriptAdapter(ResearchAdapter):
    source_type = "TRANSCRIPT"

    def enabled(self) -> bool:
        return (
            not self.cfg.safe_mode
            and self.cfg.research_enable_transcripts
            and str(self.cfg.research_transcript_provider or "").lower() == "alpha_vantage"
            and bool(self.cfg.research_alpha_vantage_api_key)
        )

    def collect(self, ctx: AdapterContext) -> AdapterResult:
        out = AdapterResult()
        if not self.enabled():
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_TRANSCRIPTS_DISABLED",
                    severity="medium",
                    summary="Transcript adapter is disabled or missing an Alpha Vantage API key.",
                    source_type=self.source_type,
                    recommended_action="Set VOE_SAFE_MODE=false, VOE_RESEARCH_ENABLE_TRANSCRIPTS=true, VOE_RESEARCH_TRANSCRIPT_PROVIDER=alpha_vantage, and ALPHA_VANTAGE_API_KEY.",
                )
            )
            return out

        api_key = str(self.cfg.research_alpha_vantage_api_key or "")
        for quarter in _recent_quarters(ctx.as_of_date, self.cfg.research_transcript_max_quarters):
            public_url = _source_url(ctx.ticker, quarter)
            try:
                raw = self.http_get_bytes(
                    _request_url(ctx.ticker, quarter, api_key),
                    use_cache=True,
                    cache_ttl_seconds=24 * 3600,
                ).decode("utf-8", errors="ignore")
                payload = json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                out.evidence_gaps.append(
                    self._gap(
                        gap_id="GAP_TRANSCRIPT_FETCH_FAILED",
                        severity="medium",
                        summary=f"Failed to fetch transcript for {ctx.ticker} {quarter}: {exc}",
                        source_type=self.source_type,
                        recommended_action="Verify Alpha Vantage transcript access and rerun research.",
                    )
                )
                continue

            if not isinstance(payload, dict):
                continue
            provider_message = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
            if provider_message:
                out.evidence_gaps.append(
                    self._gap(
                        gap_id="GAP_TRANSCRIPT_PROVIDER_MESSAGE",
                        severity="medium",
                        summary=f"Transcript provider returned a message for {ctx.ticker} {quarter}: {_clean_text(provider_message)[:240]}",
                        source_type=self.source_type,
                        recommended_action="Check Alpha Vantage entitlement, rate limit, symbol, and quarter.",
                    )
                )
                continue

            text = _transcript_text(payload)
            if not text:
                continue
            title = _clean_text(payload.get("title")) or f"{ctx.ticker} earnings call transcript {quarter}"
            call_date = _call_date(payload)
            if call_date is not None:
                published_at = call_date.isoformat()
                date_note = ""
            else:
                published_at = _quarter_published_at(str(payload.get("quarter") or quarter))
                date_note = (
                    f"[Dated {published_at}: the provider gave no call date, so the transcript is "
                    f"dated {_UNDATED_CALL_LAG_DAYS} days after its quarter ends.]\n"
                )
            # Point in time: a call held after the as-of day had not happened yet.
            try:
                if published_at is None or published_at > date.fromisoformat(ctx.as_of_date).isoformat():
                    continue
            except ValueError:
                pass
            excerpt = (date_note + text)[:_TRANSCRIPT_MAX_CHARS]
            out.evidence_items.append(
                self._make_item(
                    ctx=ctx,
                    source_type=self.source_type,
                    source_url=public_url,
                    source_title=title,
                    source_published_at=published_at,
                    excerpt_text=excerpt,
                    citations=[
                        CitationRef(
                            source_url=public_url,
                            snippet=excerpt[:600],
                            section_label=self.source_type,
                        )
                    ],
                )
            )

        if not out.evidence_items and not out.evidence_gaps:
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_TRANSCRIPT_NOT_FOUND",
                    severity="medium",
                    summary=f"No parseable Alpha Vantage earnings-call transcript was found for {ctx.ticker}.",
                    source_type=self.source_type,
                    recommended_action="Try a wider quarter window or verify the ticker has Alpha Vantage transcript coverage.",
                )
            )
        return out


__all__ = [
    "TranscriptAdapter",
]
