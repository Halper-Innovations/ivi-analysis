from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse

from app.research.adapters.base import AdapterContext, AdapterResult, ResearchAdapter
from app.research.schemas import CitationRef


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_RSS_HINT_RE = re.compile(r"rss|atom|feed", re.IGNORECASE)
_KEYWORD_RE = re.compile(r"investor|news|press|media|relations", re.IGNORECASE)
_COMMON_SUBPATHS = ["/investors", "/investor-relations", "/news", "/press", "/press-releases"]


def _clean_text(value: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", value)).strip()


def _parse_pub_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue
    return None


def _rss_items(xml_text: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    ns_content = "{http://purl.org/rss/1.0/modules/content/}encoded"
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or item.findtext(ns_content) or "").strip()
        if title and link:
            items.append({"title": title, "link": link, "published": pub, "description": desc})

    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        pub = (entry.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
        desc = (
            entry.findtext("{http://www.w3.org/2005/Atom}summary")
            or entry.findtext("{http://www.w3.org/2005/Atom}content")
            or ""
        ).strip()
        link = ""
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        if link_el is not None:
            link = (link_el.attrib.get("href") or "").strip()
        if title and link:
            items.append({"title": title, "link": link, "published": pub, "description": desc})
    return items


class CompanyNewsAdapter(ResearchAdapter):
    source_type = "company_news"

    def enabled(self) -> bool:
        return (not self.cfg.safe_mode) and self.cfg.research_company_news_enabled

    def _allowlist_domains(self, ctx: AdapterContext | None = None) -> list[str]:
        domains: list[str] = []
        for raw in list(self.cfg.research_allowlist_domains) + list((ctx.allowlist_domains if ctx else ()) or ()):
            domain = str(raw or "").lower().strip()
            if domain and domain not in domains:
                domains.append(domain)
        return domains

    def _allowlisted(self, url: str, ctx: AdapterContext | None = None) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        for base in self._allowlist_domains(ctx):
            if not base:
                continue
            if base.startswith("*."):
                suffix = base[2:]
                if suffix and (host == suffix or host.endswith(f".{suffix}")):
                    return True
            elif host == base or host.endswith(f".{base}"):
                return True
        return False

    def _discover_links(self, base_url: str, html_text: str) -> list[str]:
        out: list[str] = []
        base_host = (urlparse(base_url).hostname or "").lower()
        for href in _HREF_RE.findall(html_text):
            href = href.strip()
            if not href:
                continue
            target = urljoin(base_url, href)
            parsed = urlparse(target)
            host = (parsed.hostname or "").lower()
            if host != base_host:
                continue
            if parsed.scheme not in {"http", "https"}:
                continue
            path = (parsed.path or "/").strip()
            if path == "/" and target not in out:
                out.append(target)
                continue
            segments = [seg for seg in path.split("/") if seg]
            if len(segments) > 1:
                continue
            if _KEYWORD_RE.search(path) and target not in out:
                out.append(target)
        return out

    def _discover_feeds(self, page_url: str, html_text: str, ctx: AdapterContext) -> list[str]:
        feeds: list[str] = []
        for href in _HREF_RE.findall(html_text):
            href = href.strip()
            if not href:
                continue
            if not _RSS_HINT_RE.search(href):
                continue
            target = urljoin(page_url, href)
            if self._allowlisted(target, ctx) and target not in feeds:
                feeds.append(target)
        return feeds

    def _listing_items(self, page_url: str, html_text: str, ctx: AdapterContext) -> list[dict[str, str]]:
        listing: list[dict[str, str]] = []
        for href in _HREF_RE.findall(html_text):
            target = urljoin(page_url, href.strip())
            if not self._allowlisted(target, ctx):
                continue
            path = (urlparse(target).path or "").lower()
            if not _KEYWORD_RE.search(path):
                continue
            title = (path.rsplit("/", 1)[-1] or "news_item").replace("-", " ").replace("_", " ").strip()
            if not title:
                continue
            listing.append({"title": title.title(), "link": target, "published": "", "description": _clean_text(title)})
            if len(listing) >= 12:
                break
        return listing

    def _fallback_pages(self, ctx: AdapterContext) -> list[str]:
        pages: list[str] = []
        tokens = [ctx.ticker.lower()]
        for token in (ctx.company_name or "").lower().split():
            if len(token) >= 4:
                tokens.append(token)
        candidate_domains = self._allowlist_domains(ctx)
        scored_domains = sorted(
            candidate_domains,
            key=lambda d: 0 if any(token in d.lower() for token in tokens) else 1,
        )
        for domain in scored_domains[:6]:
            for sub in _COMMON_SUBPATHS:
                pages.append(f"https://{domain.strip().lower()}{sub}")
        return pages

    def collect(self, ctx: AdapterContext) -> AdapterResult:
        out = AdapterResult()
        if not self.enabled():
            return out

        max_days = max(1, int(self.cfg.research_ir_max_days_back))
        max_items = max(1, int(self.cfg.research_ir_max_items_per_ticker))
        max_pages = max(1, int(self.cfg.research_company_news_max_pages))
        as_of_date_value = date.fromisoformat(ctx.as_of_date)
        min_ordinal = as_of_date_value.toordinal() - max_days

        pages: list[str] = []
        homepage_url = (ctx.homepage_url or "").strip()
        if homepage_url:
            if not self._allowlisted(homepage_url, ctx):
                out.evidence_gaps.append(
                    self._gap(
                        gap_id="GAP_COMPANY_HOMEPAGE_NOT_ALLOWLISTED",
                        severity="high",
                        summary=f"Homepage domain is not allowlisted: {homepage_url}",
                        source_type="company_news",
                        recommended_action="Add homepage domain to VOE_RESEARCH_ALLOWLIST_DOMAINS.",
                    )
                )
                return out
            pages.append(homepage_url)
            try:
                root_html = self.http_get_bytes(homepage_url, use_cache=True, cache_ttl_seconds=6 * 3600).decode(
                    "utf-8", errors="ignore"
                )
                pages.extend(self._discover_links(homepage_url, root_html))
            except Exception:
                pass
        else:
            pages = self._fallback_pages(ctx)
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_HOMEPAGE_URL_MISSING",
                    severity="medium",
                    summary="Homepage URL missing in universe metadata; using allowlist fallback path discovery.",
                    source_type="company_news",
                    recommended_action="Add `homepage_url` for this ticker to improve discovery precision.",
                )
            )

        seen_urls: set[str] = set()
        seen_items: set[str] = set()
        for page in pages[:max_pages]:
            if page in seen_urls or not self._allowlisted(page, ctx):
                continue
            seen_urls.add(page)

            try:
                html = self.http_get_bytes(page, use_cache=True, cache_ttl_seconds=6 * 3600).decode(
                    "utf-8", errors="ignore"
                )
            except Exception:
                continue

            raw_items: list[dict[str, str]] = []
            feeds = self._discover_feeds(page, html, ctx)
            for feed_url in feeds:
                try:
                    xml_text = self.http_get_bytes(feed_url, use_cache=True, cache_ttl_seconds=6 * 3600).decode(
                        "utf-8", errors="ignore"
                    )
                except Exception:
                    continue
                raw_items.extend(_rss_items(xml_text))
            if not raw_items:
                raw_items.extend(self._listing_items(page, html, ctx))

            for raw in raw_items:
                title = _clean_text(raw.get("title") or "")
                link = (raw.get("link") or "").strip()
                if not title or not link or not self._allowlisted(link, ctx):
                    continue

                published_iso = _parse_pub_date(raw.get("published"))
                if published_iso:
                    try:
                        published_date = datetime.fromisoformat(published_iso).date()
                        # Both ends of the window — see ir_press: an as-of run
                        # must not collect items published after its own date.
                        if (
                            published_date.toordinal() < min_ordinal
                            or published_date > as_of_date_value
                        ):
                            continue
                    except Exception:
                        pass

                excerpt = _clean_text(raw.get("description") or title)[:1000]
                dedupe_marker = f"{ctx.ticker}|{link}|{published_iso or ''}|{title.lower()}"
                if dedupe_marker in seen_items:
                    continue
                seen_items.add(dedupe_marker)

                citation_text = f"{title}. {excerpt}"[:600]
                out.evidence_items.append(
                    self._make_item(
                        ctx=ctx,
                        source_type="company_news",
                        source_url=link,
                        source_title=title,
                        source_published_at=published_iso,
                        excerpt_text=excerpt,
                        citations=[CitationRef(source_url=link, snippet=citation_text, section_label="company_news")],
                    )
                )
                if len(out.evidence_items) >= max_items:
                    break
            if len(out.evidence_items) >= max_items:
                break

        if not out.evidence_items:
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_COMPANY_NEWS_NOT_FOUND",
                    severity="medium",
                    summary="No parseable company news/IR items were discovered from homepage/fallback paths.",
                    source_type="company_news",
                    recommended_action="Add explicit homepage_url or ir_rss_url metadata and verify allowlist domains.",
                )
            )
        return out
