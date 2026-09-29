from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from app.research.adapters.base import AdapterContext, AdapterResult, ResearchAdapter
from app.research.schemas import CitationRef
from app.util.hashing import sha256_text


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


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


class IRPressAdapter(ResearchAdapter):
    source_type = "ir_press"

    def enabled(self) -> bool:
        return (not self.cfg.safe_mode) and self.cfg.research_ir_press_enabled

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

    @staticmethod
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

    def collect(self, ctx: AdapterContext) -> AdapterResult:
        out = AdapterResult()
        if not self.enabled():
            return out

        ir_rss_url = (ctx.ir_rss_url or "").strip()
        if not ir_rss_url:
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_IR_RSS_MISSING",
                    severity="high",
                    summary="No ir_rss_url configured in universe metadata for this ticker.",
                    source_type="ir_press",
                    recommended_action=(
                        "Add `ir_rss_url` to data/universe/universe.csv for this ticker, reload universe, and rerun research."
                    ),
                )
            )
            return out

        if not self._allowlisted(ir_rss_url, ctx):
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_IR_RSS_DOMAIN_NOT_ALLOWLISTED",
                    severity="high",
                    summary=f"IR RSS domain is not allowlisted: {ir_rss_url}",
                    source_type="ir_press",
                    recommended_action=(
                        "Add the IR domain to VOE_RESEARCH_ALLOWLIST_DOMAINS and rerun research."
                    ),
                )
            )
            return out

        try:
            xml_text = self.http_get_bytes(ir_rss_url, use_cache=True, cache_ttl_seconds=6 * 3600).decode(
                "utf-8", errors="ignore"
            )
        except Exception as exc:  # noqa: BLE001
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_IR_RSS_FETCH_FAILED",
                    severity="medium",
                    summary=f"Failed to fetch IR RSS feed: {exc}",
                    source_type="ir_press",
                    recommended_action=(
                        "Verify feed URL and allowlist domain, then rerun research."
                    ),
                )
            )
            return out

        items = self._rss_items(xml_text)
        if not items:
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_IR_RSS_EMPTY",
                    severity="medium",
                    summary="IR RSS feed returned no parseable items.",
                    source_type="ir_press",
                    recommended_action="Provide a valid RSS/Atom feed URL in universe metadata.",
                )
            )
            return out

        as_of = date.fromisoformat(ctx.as_of_date)
        max_days = max(1, int(self.cfg.research_ir_max_days_back))
        max_items = max(1, int(self.cfg.research_ir_max_items_per_ticker))
        min_date = as_of.toordinal() - max_days

        seen: set[tuple[str, str, str]] = set()
        for raw in items:
            title = _clean_text(raw.get("title") or "")
            link = (raw.get("link") or "").strip()
            if not title or not link:
                continue

            published_iso = _parse_pub_date(raw.get("published"))
            if published_iso:
                try:
                    pd = datetime.fromisoformat(published_iso).date()
                    # Both ends of the window. The lower bound was always here;
                    # without the upper one a run "as of" a past date collected
                    # releases published after it and cited them as evidence.
                    if pd.toordinal() < min_date or pd > as_of:
                        continue
                except Exception:
                    pass

            excerpt = _clean_text(raw.get("description") or "")
            if not excerpt:
                excerpt = title
            excerpt = excerpt[:1000]

            dedupe_key = (ctx.ticker, link, sha256_text(excerpt))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            citation_text = f"{title}. {excerpt}"[:600]
            out.evidence_items.append(
                self._make_item(
                    ctx=ctx,
                    source_type="ir_press",
                    source_url=link,
                    source_title=title,
                    source_published_at=published_iso,
                    excerpt_text=excerpt,
                    citations=[CitationRef(source_url=link, snippet=citation_text, section_label="ir_rss")],
                )
            )
            if len(out.evidence_items) >= max_items:
                break

        if not out.evidence_items:
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_IR_RSS_FILTERED_OUT",
                    severity="low",
                    summary="IR RSS items found, but none matched freshness/item limits.",
                    source_type="ir_press",
                    recommended_action=(
                        "Increase VOE_RESEARCH_IR_MAX_DAYS_BACK or verify publication cadence."
                    ),
                )
            )

        return out
