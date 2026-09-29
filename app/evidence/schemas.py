from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_url: str
    snippet: str
    section_label: str | None = None


class ExtractedFactRecord(BaseModel):
    fact_type: str
    value: dict[str, Any]
    citation: Citation


class FinancialRecord(BaseModel):
    statement_type: str
    line_item: str
    value: float | None
    units: str | None
    period: str | None
    citation: Citation


class FilingMetadata(BaseModel):
    accession: str
    form_type: str
    filing_date: str | None
    period_end: str | None
    primary_doc_url: str


class EvidencePacket(BaseModel):
    ticker: str
    as_of_date: str
    filings_used: list[FilingMetadata] = Field(default_factory=list)
    extracted_facts: list[ExtractedFactRecord] = Field(default_factory=list)
    financials: list[FinancialRecord] = Field(default_factory=list)
    fundamentals: dict[str, Any]
    valuations: dict[str, Any]
    deltas_vs_prior_period: dict[str, Any]
