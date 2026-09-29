from __future__ import annotations

from app.research.adapters.base import AdapterContext, AdapterResult, ResearchAdapter
from app.research.schemas import CitationRef, EvidenceItem


class EdgarEvidenceAdapter(ResearchAdapter):
    source_type = "EDGAR"

    def collect(self, ctx: AdapterContext) -> AdapterResult:
        out = AdapterResult()

        for fact in ctx.packet.get("extracted_facts", [])[:80]:
            citation = fact.get("citation", {})
            source_url = citation.get("source_url") or ""
            snippet = (citation.get("snippet") or "").strip()
            if not source_url or not snippet:
                continue
            excerpt = f"{fact.get('fact_type')}: {snippet}"[:1200]
            out.evidence_items.append(
                self._make_item(
                    ctx=ctx,
                    source_type="EDGAR",
                    source_url=source_url,
                    excerpt_text=excerpt,
                    citations=[
                        CitationRef(
                            source_url=source_url,
                            snippet=snippet[:600],
                            section_label=citation.get("section_label"),
                        )
                    ],
                    source_title=fact.get("fact_type"),
                )
            )

        for row in ctx.packet.get("financials", [])[:80]:
            citation = row.get("citation", {})
            source_url = citation.get("source_url") or ""
            snippet = (citation.get("snippet") or "").strip()
            if not source_url or not snippet:
                continue
            excerpt = f"{row.get('line_item')}: {snippet}"[:1200]
            out.evidence_items.append(
                self._make_item(
                    ctx=ctx,
                    source_type="EDGAR",
                    source_url=source_url,
                    excerpt_text=excerpt,
                    citations=[CitationRef(source_url=source_url, snippet=snippet[:600], section_label="financials")],
                    source_title=row.get("line_item"),
                )
            )

        dedup: dict[str, EvidenceItem] = {}
        for item in out.evidence_items:
            dedup[item.id] = item
        out.evidence_items = list(dedup.values())
        return out
