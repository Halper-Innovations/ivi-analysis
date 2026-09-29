from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path

from app.db import get_db
from app.research.adapters.base import AdapterContext, AdapterResult, ResearchAdapter
from app.research.schemas import CitationRef
from app.util.text import extract_snippet, normalize_whitespace


_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_EXHIBIT_RE = re.compile(r"exhibit\s*(99\.\d+)", re.IGNORECASE)
_ALT_EXHIBIT_RE = re.compile(r"ex[-\s]?99[.\-_]?(\d+)", re.IGNORECASE)
_ITEM_RE = re.compile(r"item\s*(2\.02|7\.01|8\.01|9\.01)", re.IGNORECASE)

_EARNINGS_HINTS = (
    "press release",
    "results of operations",
    "financial results",
    "quarterly results",
    "quarterly earnings",
    "earnings release",
    "earnings results",
    "guidance",
    "outlook",
)
_PRESENTATION_HINTS = (
    "investor presentation",
    "presentation",
    "slide deck",
    "slides",
    "supplemental presentation",
    "conference call",
)
_RECONCILIATION_HINTS = (
    "non-gaap",
    "reconciliation",
    "adjusted",
    "capital return",
    "share repurchase",
    "buyback",
    "dividend",
    "liquidity",
    "covenant",
    "maturity",
    "segment",
    "credit",
    "reserve",
    "charge-off",
    "charge off",
    "nonaccrual",
    "provision",
)


def _strip_tags(text: str) -> str:
    return normalize_whitespace(_TAG_RE.sub(" ", text))


def _headline(text: str) -> str:
    for pattern in (_TITLE_RE, _H1_RE):
        m = pattern.search(text)
        if m:
            title = _strip_tags(m.group(1))
            if title:
                return title[:220]
    lines = [line.strip() for line in _strip_tags(text).split(".") if line.strip()]
    return lines[0][:220] if lines else "SEC exhibit update"


def _marker_flags(text: str) -> tuple[bool, bool, list[str], list[str]]:
    lowered = text.lower()
    exhibit_codes = sorted(
        {
            code.lower()
            for code in _EXHIBIT_RE.findall(lowered)
        }
        | {
            f"99.{code}"
            for code in _ALT_EXHIBIT_RE.findall(lowered)
        }
    )
    item_codes = sorted({code for code in _ITEM_RE.findall(lowered)})
    has_earnings_terms = any(token in lowered for token in _EARNINGS_HINTS)
    has_presentation_terms = any(token in lowered for token in _PRESENTATION_HINTS)
    has_item_earnings = any(code in {"2.02", "8.01", "9.01"} for code in item_codes)
    has_item_presentation = any(code in {"7.01", "8.01", "9.01"} for code in item_codes)

    has_earnings = (has_earnings_terms and (bool(exhibit_codes) or has_item_earnings)) or (
        any(code in {"99.1", "99.2"} for code in exhibit_codes) and has_earnings_terms
    )
    has_presentation = (has_presentation_terms and (bool(exhibit_codes) or has_item_presentation)) or (
        any(code in {"99.1", "99.2", "99.3"} for code in exhibit_codes) and has_presentation_terms
    )
    return has_earnings, has_presentation, exhibit_codes, item_codes


def _evidence_focuses(text: str) -> list[tuple[str, str, str]]:
    lowered = text.lower()
    focuses: list[tuple[str, str, str]] = []
    if any(token in lowered for token in ("non-gaap", "reconciliation", "adjusted")):
        focuses.append(("non-gaap", "non_gaap", "non_gaap"))
    if any(token in lowered for token in ("share repurchase", "buyback", "dividend", "capital return")):
        focuses.append(("share repurchase", "capital_returns", "capital_returns"))
    if any(token in lowered for token in ("liquidity", "covenant", "maturity", "refinancing")):
        focuses.append(("liquidity", "liquidity_and_debt", "liquidity_and_debt"))
    if any(token in lowered for token in ("credit", "reserve", "charge-off", "charge off", "nonaccrual", "provision")):
        focuses.append(("credit", "credit_reserves", "credit_reserves"))
    if "segment" in lowered:
        focuses.append(("segment", "segment_reporting", "segment_reporting"))
    if any(token in lowered for token in ("financial results", "quarterly earnings", "earnings release")):
        focuses.append(("financial results", "earnings_release", "earnings_release"))
    if not focuses:
        focuses.append(("exhibit", "investor_update", "investor_update"))
    deduped: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for focus in focuses:
        if focus[2] in seen:
            continue
        seen.add(focus[2])
        deduped.append(focus)
    return deduped[:3]


def _published_iso(filing_date: str | None) -> str | None:
    if not filing_date:
        return None
    try:
        dt = datetime.fromisoformat(filing_date)
    except Exception:
        try:
            dt = datetime.combine(date.fromisoformat(filing_date), datetime.min.time())
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class SecExhibitsAdapter(ResearchAdapter):
    source_type = "sec_exhibit"

    def enabled(self) -> bool:
        return bool(self.cfg.research_exhibits_enabled)

    def _load_filing_text(self, local_path: str | None, source_url: str) -> str | None:
        if local_path:
            path = Path(local_path)
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8", errors="ignore")
        try:
            return self.http_get_bytes(source_url, use_cache=True, cache_ttl_seconds=24 * 3600).decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            return None

    def collect(self, ctx: AdapterContext) -> AdapterResult:
        out = AdapterResult()
        if not self.enabled():
            return out

        as_of = ctx.as_of_date
        scanned_rows = 0
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT accession, filing_date, primary_doc_url, local_path
                FROM filings
                WHERE ticker = ? AND form_type = '8-K' AND COALESCE(filing_date, '1900-01-01') <= ?
                ORDER BY filing_date DESC
                LIMIT 30
                """,
                (ctx.ticker, as_of),
            ).fetchall()

        for row in rows:
            scanned_rows += 1
            text = self._load_filing_text(row["local_path"], row["primary_doc_url"])
            if not text:
                continue
            lowered = text.lower()
            has_earnings, has_presentation, exhibit_codes, item_codes = _marker_flags(text)
            if not has_earnings and not has_presentation:
                continue

            title = _headline(text)
            marker = []
            if has_earnings:
                marker.append("earnings_release")
            if has_presentation:
                marker.append("investor_presentation")
            if item_codes:
                marker.extend(f"item_{code.replace('.', '')}" for code in item_codes[:2])
            marker_text = ",".join(marker)
            bullet_candidates = []
            for token in [
                "guidance",
                "outlook",
                "restructuring",
                "investigation",
                "sec subpoena",
                "bankruptcy",
                "going concern",
                "material weakness",
                "financial results",
                "quarterly earnings",
                "conference call",
                "non-gaap",
                "reconciliation",
                "share repurchase",
                "dividend",
                "liquidity",
                "covenant",
                "maturity",
                "credit",
                "reserve",
                "charge-off",
                "charge off",
                "nonaccrual",
                "provision",
                "segment",
            ]:
                snippet = extract_snippet(_strip_tags(text), token, width=260)
                if snippet and token.lower() in snippet.lower():
                    bullet_candidates.append(snippet)
            bullet_candidates = bullet_candidates[:3]
            fallback_token = "guidance" if "guidance" in lowered else ("financial results" if "financial results" in lowered else "exhibit")
            exhibit_note = f"codes={','.join(exhibit_codes)}. " if exhibit_codes else ""
            excerpt = f"{title}. {marker_text}. {exhibit_note}" + " ".join(
                bullet_candidates or [extract_snippet(_strip_tags(text), fallback_token, 260)]
            )
            excerpt = normalize_whitespace(excerpt)[:1200]

            source_url = row["primary_doc_url"]
            for focus_token, _focus_key, focus_label in _evidence_focuses(text):
                citation = CitationRef(
                    source_url=source_url,
                    snippet=extract_snippet(_strip_tags(text), focus_token if focus_token in lowered else fallback_token, width=500)[:600],
                    section_label=f"8k_{item_codes[0].replace('.', '')}_{focus_label}" if item_codes else f"8k_{focus_label}",
                )
                out.evidence_items.append(
                    self._make_item(
                        ctx=ctx,
                        source_type="sec_exhibit",
                        source_url=source_url,
                        source_title=f"{title} [{marker_text}|{focus_label}]",
                        source_published_at=_published_iso(row["filing_date"]),
                        excerpt_text=excerpt,
                        citations=[citation],
                    )
                )

        if not out.evidence_items:
            summary = "No 8-K exhibit evidence items (99.x earnings/presentation) were extracted."
            recommended_action = "Ingest/parse additional 8-K filings and confirm exhibit text availability."
            if scanned_rows > 0:
                summary = "Scanned available 8-K filings but did not find relevant earnings, presentation, or reconciliation exhibit evidence before the as-of date."
                recommended_action = "Confirm whether the issuer published relevant 8-K exhibits before the as-of date and ingest any linked presentation or reconciliation materials."
            out.evidence_gaps.append(
                self._gap(
                    gap_id="GAP_SEC_EXHIBITS_NOT_FOUND",
                    severity="low",
                    summary=summary,
                    source_type="sec_exhibit",
                    recommended_action=recommended_action,
                )
            )
        return out
