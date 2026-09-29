from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Iterable

from app.config import get_config
from app.db import get_db
from app.research.adapters.base import AdapterContext
from app.research.adapters.company_news import CompanyNewsAdapter
from app.research.adapters.external_news import ExternalNewsAdapter
from app.research.adapters.ir_press import IRPressAdapter
from app.research.adapters.transcripts import TranscriptAdapter
from app.research.schemas import CitationRef, EvidenceItem


_CURRENT_EVENT_WINDOW_DAYS = 90
_CURRENT_EVENT_LIMIT = 20
_SUPPRESSED_EMPTY_GAPS = frozenset(
    {
        "GAP_IR_RSS_FILTERED_OUT",
        "GAP_COMPANY_NEWS_NOT_FOUND",
    }
)


@dataclass(frozen=True)
class CurrentEventDocument:
    ticker: str
    source_type: str
    published_at: str | None
    title: str
    source_url: str
    summary: str
    citations: list[CitationRef] = field(default_factory=list)
    source_role: str = "current_event"
    source_quality: dict[str, object] | None = None


@dataclass(frozen=True)
class CurrentEventContext:
    documents: list[CurrentEventDocument] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata_source: str = "unknown"
    homepage_url_present: bool = False
    ir_rss_url_present: bool = False
    allowlist_domains: list[str] = field(default_factory=list)
    allowlist_source: str = "none"

    @property
    def ordered_documents(self) -> list[CurrentEventDocument]:
        return sorted(
            self.documents,
            key=lambda doc: (_published_sort_key(doc.published_at), doc.source_type, doc.source_url),
            reverse=True,
        )

    @property
    def latest_document(self) -> CurrentEventDocument | None:
        ordered = self.ordered_documents
        return ordered[0] if ordered else None


def _parse_allowlist_domains(value: str | None) -> list[str]:
    domains: list[str] = []
    for token in re.split(r"[,;\s]+", str(value or "")):
        domain = token.lower().strip()
        if domain and domain not in domains:
            domains.append(domain)
    return domains


def _metadata_from_row(row, *, source: str) -> dict[str, object]:
    allowlist_domains = _parse_allowlist_domains(row["allowlist_domains"])
    return {
        "name": str(row["name"] or ""),
        "homepage_url": str(row["homepage_url"] or ""),
        "ir_rss_url": str(row["ir_rss_url"] or ""),
        "allowlist_domains": allowlist_domains,
        "metadata_source": source,
        "allowlist_source": source if allowlist_domains else "none",
    }


def _metadata_from_mapping(row: dict[str, str], *, source: str) -> dict[str, object]:
    allowlist_domains = _parse_allowlist_domains(row.get("allowlist_domains"))
    return {
        "name": str(row.get("name") or ""),
        "homepage_url": str(row.get("homepage_url") or ""),
        "ir_rss_url": str(row.get("ir_rss_url") or ""),
        "allowlist_domains": allowlist_domains,
        "metadata_source": source,
        "allowlist_source": source if allowlist_domains else "none",
    }


def _metadata_has_any_fields(metadata: dict[str, object]) -> bool:
    return bool(
        str(metadata.get("name") or "").strip()
        or str(metadata.get("homepage_url") or "").strip()
        or str(metadata.get("ir_rss_url") or "").strip()
        or list(metadata.get("allowlist_domains") or [])
    )


def _empty_metadata() -> dict[str, object]:
    return {
        "name": "",
        "homepage_url": "",
        "ir_rss_url": "",
        "allowlist_domains": [],
        "metadata_source": "missing",
        "allowlist_source": "none",
    }


def _resolve_existing_path(path: Path) -> Path | None:
    if path.exists():
        return path
    cfg = get_config()
    rooted = cfg.project_root / path
    return rooted if rooted.exists() else None


def _metadata_from_csv(path: Path, ticker: str, *, source: str) -> dict[str, object] | None:
    resolved = _resolve_existing_path(path)
    if resolved is None:
        return None
    try:
        with resolved.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("ticker") or "").strip().upper() != ticker.upper():
                    continue
                metadata = _metadata_from_mapping(row, source=source)
                return metadata if _metadata_has_any_fields(metadata) else None
    except OSError:
        return None
    return None


def _load_local_metadata(ticker: str) -> dict[str, object] | None:
    cfg = get_config()
    for path, source in (
        (cfg.universe_dir / "metadata_overrides.csv", "metadata_overrides_csv"),
        (cfg.universe_path, "universe_csv"),
    ):
        metadata = _metadata_from_csv(path, ticker, source=source)
        if metadata is not None:
            return metadata
    return None


def _load_company_metadata(ticker: str) -> dict[str, object]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT name, homepage_url, ir_rss_url, allowlist_domains
            FROM companies
            WHERE ticker = ?
            LIMIT 1
            """,
            (ticker.upper(),),
        ).fetchone()
        if row:
            metadata = _metadata_from_row(row, source="companies")
            if _metadata_has_any_fields(metadata):
                return metadata
        row = conn.execute(
            """
            SELECT name, homepage_url, ir_rss_url, allowlist_domains
            FROM universe_members
            WHERE ticker = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (ticker.upper(),),
        ).fetchone()
    if row:
        metadata = _metadata_from_row(row, source="universe_members")
        if _metadata_has_any_fields(metadata):
            return metadata
    return _load_local_metadata(ticker) or _empty_metadata()


def _canonical_adapter_cfg():
    cfg = get_config()
    if not cfg.safe_mode:
        return cfg
    return cfg.model_copy(update={"safe_mode": False})


def _build_adapter_context(ticker: str, as_of_date: str, metadata: dict[str, object]) -> AdapterContext:
    return AdapterContext(
        ticker=ticker.upper(),
        as_of_date=as_of_date,
        company_name=str(metadata.get("name") or "") or None,
        packet={},
        ir_rss_url=str(metadata.get("ir_rss_url") or "") or None,
        homepage_url=str(metadata.get("homepage_url") or "") or None,
        allowlist_domains=tuple(str(item) for item in (metadata.get("allowlist_domains") or [])),
    )


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _published_sort_key(value: str | None) -> str:
    dt = _parse_date(value)
    return dt.isoformat() if dt is not None else ""


def _within_window(item: EvidenceItem, *, anchor_date: date) -> bool:
    published = _parse_date(item.source_published_at)
    if published is None:
        return True
    published_date = published.date()
    if published_date > anchor_date:
        return False
    return (anchor_date - published_date).days <= _CURRENT_EVENT_WINDOW_DAYS


def _event_from_item(item: EvidenceItem) -> CurrentEventDocument:
    return CurrentEventDocument(
        ticker=item.ticker,
        source_type=item.source_type,
        published_at=item.source_published_at,
        title=item.source_title or item.source_url,
        source_url=item.source_url,
        summary=item.excerpt_text,
        citations=list(item.citations),
        source_quality=item.source_quality,
    )


def _warning_token(source_type: str, raw_gap_id: str) -> str:
    return f"current_event_gap:{source_type}:{raw_gap_id}"


def _suppress_gap(gap_id: str) -> bool:
    return gap_id in _SUPPRESSED_EMPTY_GAPS


def _collect_with_adapter(
    adapter,
    *,
    ticker: str,
    as_of_date: str,
    metadata: dict[str, object],
    enabled: bool,
) -> tuple[list[CurrentEventDocument], list[str]]:
    if not enabled:
        return [], [f"current_event_source_disabled:{adapter.source_type}"]

    ctx = _build_adapter_context(ticker, as_of_date, metadata)
    result = adapter.collect(ctx)

    try:
        anchor = date.fromisoformat(as_of_date)
    except ValueError:
        anchor = date.today()

    documents = [
        _event_from_item(item)
        for item in result.evidence_items
        if _within_window(item, anchor_date=anchor)
    ]
    warnings = [
        _warning_token(adapter.source_type, gap.gap_id)
        for gap in result.evidence_gaps
        if not _suppress_gap(gap.gap_id)
    ]
    return documents, warnings


def _dedupe_documents(documents: Iterable[CurrentEventDocument]) -> list[CurrentEventDocument]:
    seen_urls: set[str] = set()
    seen_title_dates: set[tuple[str, str]] = set()
    out: list[CurrentEventDocument] = []

    for document in sorted(
        documents,
        key=lambda doc: (_published_sort_key(doc.published_at), 1 if doc.source_type == "company_news" else 2, doc.title),
        reverse=True,
    ):
        url_key = document.source_url.strip().lower()
        title_key = " ".join(document.title.lower().split())
        date_key = (document.published_at or "")[:10]
        title_date_key = (title_key, date_key)
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_date_key in seen_title_dates:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_title_dates.add(title_date_key)
        out.append(document)
    return out


def load_current_event_context(ticker: str, *, as_of_date: str) -> CurrentEventContext:
    cfg = get_config()
    metadata = _load_company_metadata(ticker)
    adapter_cfg = _canonical_adapter_cfg()

    ir_adapter = IRPressAdapter(adapter_cfg)
    transcript_adapter = TranscriptAdapter(adapter_cfg)
    external_news_adapter = ExternalNewsAdapter(adapter_cfg)
    company_news_adapter = CompanyNewsAdapter(adapter_cfg)

    documents: list[CurrentEventDocument] = []
    warnings: list[str] = []

    ir_documents, ir_warnings = _collect_with_adapter(
        ir_adapter,
        ticker=ticker,
        as_of_date=as_of_date,
        metadata=metadata,
        enabled=cfg.research_ir_press_enabled,
    )
    company_news_documents, company_news_warnings = _collect_with_adapter(
        company_news_adapter,
        ticker=ticker,
        as_of_date=as_of_date,
        metadata=metadata,
        enabled=cfg.research_company_news_enabled,
    )
    if (
        not cfg.safe_mode
        and cfg.research_enable_transcripts
        and str(cfg.research_transcript_provider or "").lower() == "alpha_vantage"
        and bool(cfg.research_alpha_vantage_api_key)
    ):
        transcript_documents, transcript_warnings = _collect_with_adapter(
            transcript_adapter,
            ticker=ticker,
            as_of_date=as_of_date,
            metadata=metadata,
            enabled=True,
        )
        documents.extend(transcript_documents)
        warnings.extend(transcript_warnings)
    if (
        not cfg.safe_mode
        and cfg.research_external_news_enabled
        and str(cfg.research_external_news_provider or "").lower() == "alpha_vantage"
        and bool(cfg.research_alpha_vantage_api_key)
    ):
        external_news_documents, external_news_warnings = _collect_with_adapter(
            external_news_adapter,
            ticker=ticker,
            as_of_date=as_of_date,
            metadata=metadata,
            enabled=True,
        )
        documents.extend(external_news_documents)
        warnings.extend(external_news_warnings)
    documents.extend(ir_documents)
    documents.extend(company_news_documents)
    warnings.extend(ir_warnings)
    warnings.extend(company_news_warnings)

    deduped = _dedupe_documents(documents)
    capped = deduped[:_CURRENT_EVENT_LIMIT]
    if len(deduped) > len(capped):
        warnings.append(f"current_event_items_truncated:{len(capped)}/{len(deduped)}")

    return CurrentEventContext(
        documents=capped,
        warnings=list(dict.fromkeys(warnings)),
        metadata_source=str(metadata.get("metadata_source") or "unknown"),
        homepage_url_present=bool(str(metadata.get("homepage_url") or "").strip()),
        ir_rss_url_present=bool(str(metadata.get("ir_rss_url") or "").strip()),
        allowlist_domains=[str(item) for item in (metadata.get("allowlist_domains") or [])],
        allowlist_source=str(metadata.get("allowlist_source") or "none"),
    )
