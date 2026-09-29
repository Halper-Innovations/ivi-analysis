from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.ingest.sec_client import FilingStub


ANNUAL_FORMS = {"10-K", "20-F"}
QUARTERLY_FORMS = {"10-Q"}
EVENT_FORMS = {"8-K"}


@dataclass
class DiscoveryFilingSelection:
    annual: FilingStub | None
    quarters: list[FilingStub]
    event: FilingStub | None

    @property
    def all_filings(self) -> list[FilingStub]:
        rows: list[FilingStub] = []
        if self.annual:
            rows.append(self.annual)
        rows.extend(self.quarters)
        if self.event:
            rows.append(self.event)
        rows.sort(key=lambda f: f.filing_date, reverse=True)
        deduped: dict[str, FilingStub] = {}
        for row in rows:
            deduped[row.accession] = row
        return sorted(deduped.values(), key=lambda f: f.filing_date, reverse=True)


def _build_stub(
    *,
    cik: str,
    accession: str,
    form_type: str,
    filing_date: str,
    period_end: str | None,
    primary_document: str,
) -> FilingStub | None:
    if not accession or not primary_document:
        return None
    try:
        filed = date.fromisoformat(filing_date)
    except Exception:
        return None
    accession_nodash = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}"
    return FilingStub(
        cik=str(int(cik)),
        accession=str(accession),
        accession_nodash=accession_nodash,
        form_type=form_type.upper().strip(),
        filing_date=filed,
        period_end=period_end,
        primary_document=primary_document,
        primary_doc_url=f"{base}/{primary_document}",
        filing_index_url=f"{base}/index.json",
    )


def _is_financial_event_item(items_text: str) -> bool:
    lowered = (items_text or "").lower()
    return any(token in lowered for token in ["2.02", "9.01", "financial", "results", "earnings"])


def select_discovery_filings_from_submissions(
    submissions_payload: dict[str, Any],
    *,
    cik: str,
    as_of_date: date,
    max_quarters: int = 4,
    include_event_8k: bool = True,
) -> DiscoveryFilingSelection:
    recent = submissions_payload.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    period_ends = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])
    items = recent.get("items", [])

    rows: list[tuple[FilingStub, str]] = []
    for idx, accession in enumerate(accessions):
        form = str(forms[idx]).upper().strip() if idx < len(forms) else ""
        filing_date = str(filing_dates[idx]) if idx < len(filing_dates) else ""
        period_end = str(period_ends[idx]) if idx < len(period_ends) and period_ends[idx] else None
        primary_doc = str(primary_docs[idx]) if idx < len(primary_docs) else ""
        item_text = str(items[idx]) if idx < len(items) else ""
        stub = _build_stub(
            cik=cik,
            accession=str(accession),
            form_type=form,
            filing_date=filing_date,
            period_end=period_end,
            primary_document=primary_doc,
        )
        if stub is None:
            continue
        if stub.filing_date > as_of_date:
            continue
        rows.append((stub, item_text))

    rows.sort(key=lambda pair: pair[0].filing_date, reverse=True)

    annual: FilingStub | None = None
    quarters: list[FilingStub] = []
    event: FilingStub | None = None

    for stub, item_text in rows:
        form = stub.form_type
        if annual is None and form in ANNUAL_FORMS:
            annual = stub
            continue
        if form in QUARTERLY_FORMS and len(quarters) < max(1, max_quarters):
            quarters.append(stub)
            continue
        if include_event_8k and event is None and form in EVENT_FORMS and _is_financial_event_item(item_text):
            event = stub

    return DiscoveryFilingSelection(annual=annual, quarters=quarters, event=event)

