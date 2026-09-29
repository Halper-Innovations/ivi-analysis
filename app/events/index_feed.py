"""EDGAR daily/full index URL builders + pure master.idx parser.

Pure parsing only — no I/O beyond URL string construction. Both the daily
index (undashed dates, `File Name` header) and the quarterly full index
(dashed dates, `Filename` header) parse through the same code path so a
backfill is a faithful day-by-day simulation of daily polling. The parser
anchors on the dashed separator line, never on header text, normalizes both
date formats to YYYY-MM-DD, and dedupes exact duplicate rows (daily files
carry them — verified on master.20250505.idx).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
_SEPARATOR_RE = re.compile(rb"^-{10,}\s*$")


@dataclass(frozen=True)
class IndexRow:
    cik: str            # zero-padded 10
    company_name: str
    form_type: str      # exact EDGAR string, e.g. "10-12B", "8-A12B", "25-NSE"
    date_filed: str     # "YYYY-MM-DD" — always dashed; parser normalizes daily "YYYYMMDD"
    file_name: str      # edgar/data/... path; accession derived from it

    @property
    def accession(self) -> str:
        match = _ACCESSION_RE.search(self.file_name)
        return match.group(1) if match else ""


CANDIDATE_FORMS = {
    "10-12B", "10-12B/A", "10-12G", "10-12G/A", "8-A12B", "8-A12B/A",
    "25", "25-NSE", "S-1", "S-1/A", "424B4", "8-K", "EFFECT", "CERT",
}


def daily_index_url(scan_date: str) -> str:
    year, month, day = scan_date.split("-")
    quarter = (int(month) - 1) // 3 + 1
    return (
        "https://www.sec.gov/Archives/edgar/daily-index/"
        f"{year}/QTR{quarter}/master.{year}{month}{day}.idx"
    )


def full_index_url(year: int, quarter: int) -> str:
    return f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"


def _normalize_date(raw: str) -> str | None:
    raw = raw.strip()
    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    return None


def parse_master_idx(raw: bytes) -> tuple[list[IndexRow], int]:
    """Parse master.idx bytes -> (rows, malformed_count).

    Total: malformed lines are counted, never raised on. Rows are the lines
    after the dashed separator with exactly 4 pipes; exact duplicate lines are
    deduped order-preservingly.
    """
    rows: list[IndexRow] = []
    malformed = 0
    seen_lines: set[bytes] = set()
    in_body = False
    for line in raw.splitlines():
        if not in_body:
            if _SEPARATOR_RE.match(line.strip()):
                in_body = True
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in seen_lines:
            continue
        seen_lines.add(stripped)
        text = stripped.decode("latin-1")
        parts = text.split("|")
        if len(parts) != 5:
            malformed += 1
            continue
        cik_raw, company_name, form_type, date_raw, file_name = (p.strip() for p in parts)
        date_filed = _normalize_date(date_raw)
        if not cik_raw.isdigit() or date_filed is None or not form_type:
            malformed += 1
            continue
        rows.append(
            IndexRow(
                cik=cik_raw.zfill(10),
                company_name=company_name,
                form_type=form_type,
                date_filed=date_filed,
                file_name=file_name,
            )
        )
    return rows, malformed


def candidate_rows(rows: list[IndexRow]) -> list[IndexRow]:
    return [r for r in rows if r.form_type in CANDIDATE_FORMS]
