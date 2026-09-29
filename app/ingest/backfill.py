from __future__ import annotations

from datetime import date

from app.ingest.filings import ingest_filings_between


def backfill_range(from_date: date, to_date: date, forms: list[str], as_of_date: str | None = None) -> int:
    if to_date < from_date:
        raise ValueError("to date cannot be before from date")
    return ingest_filings_between(from_date, to_date, forms, as_of_date=as_of_date)
