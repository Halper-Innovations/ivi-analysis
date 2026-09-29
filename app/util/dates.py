from __future__ import annotations

from datetime import date, datetime, timezone


def parse_yyyy_mm_dd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_date(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()
