from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any
import threading

from app.config import AppConfig, get_config
from app.research.schemas import CitationRef, EvidenceGap, EvidenceItem
from app.research.source_quality import classify_source_quality
from app.util.hashing import sha256_text
from app.util.http import HttpClient
from app.db import utc_now_iso

_RESEARCH_HTTP_SEMAPHORE = threading.BoundedSemaphore(max(1, int(get_config().research_max_http_concurrency)))


@dataclass
class AdapterContext:
    ticker: str
    as_of_date: str
    company_name: str | None
    packet: dict[str, Any]
    run_id: str | None = None
    ir_rss_url: str | None = None
    homepage_url: str | None = None
    allowlist_domains: tuple[str, ...] = ()


@dataclass
class AdapterResult:
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = field(default_factory=list)


class ResearchAdapter:
    source_type: str = "unknown"

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.http = HttpClient(self.cfg)

    def enabled(self) -> bool:
        return True

    def collect(self, ctx: AdapterContext) -> AdapterResult:
        raise NotImplementedError

    def http_get_bytes(self, *args, **kwargs):
        with _RESEARCH_HTTP_SEMAPHORE:
            return self.http.get_bytes(*args, **kwargs)

    def _make_item(
        self,
        *,
        ctx: AdapterContext,
        source_type: str,
        source_url: str,
        excerpt_text: str,
        citations: list[CitationRef],
        source_title: str | None = None,
        source_published_at: str | None = None,
        source_quality: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        source_title = (source_title or "").strip() or None
        source_quality = source_quality or classify_source_quality(
            source_type=source_type,
            source_url=source_url,
            published_at=source_published_at,
            as_of_date=ctx.as_of_date,
            source_reputation_path=self.cfg.research_source_reputation_path,
        )
        published_key = (source_published_at or "")[:10]
        title_hash = sha256_text(source_title or "")
        dedupe_key = sha256_text(f"{source_url}|{published_key}|{title_hash}")
        content_hash = sha256_text(excerpt_text)
        item_id = sha256_text(f"{ctx.ticker}|{dedupe_key}")[:24]
        payload = {
            "ticker": ctx.ticker,
            "as_of_date": ctx.as_of_date,
            "source_type": source_type,
            "source_url": source_url,
            "source_title": source_title,
            "source_published_at": source_published_at,
            "content_hash": content_hash,
            "dedupe_key": dedupe_key,
            "source_quality": source_quality,
        }
        return EvidenceItem(
            id=f"ev_{item_id}",
            ticker=ctx.ticker,
            as_of_date=ctx.as_of_date,
            source_type=source_type,  # type: ignore[arg-type]
            source_url=source_url,
            source_title=source_title,
            source_published_at=source_published_at,
            retrieved_at=utc_now_iso(),
            excerpt_text=excerpt_text,
            citations=citations,
            hash=sha256_text(json.dumps(payload, sort_keys=True)),
            content_hash=content_hash,
            dedupe_key=dedupe_key,
            adapter_run_id=ctx.run_id,
            source_quality=source_quality,
        )

    @staticmethod
    def _gap(gap_id: str, severity: str, summary: str, source_type: str, recommended_action: str) -> EvidenceGap:
        return EvidenceGap(
            gap_id=gap_id,
            severity=severity,  # type: ignore[arg-type]
            summary=summary,
            source_type=source_type,
            recommended_action=recommended_action,
        )
