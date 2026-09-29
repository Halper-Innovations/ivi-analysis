from __future__ import annotations

from app.events.index_feed import (
    CANDIDATE_FORMS,
    candidate_rows,
    daily_index_url,
    full_index_url,
    parse_master_idx,
)

DAILY_FIXTURE = b"""Description:           Daily Index of EDGAR Dissemination Feed
Last Data Received:     May 5, 2025
Comments:               webmaster@sec.gov
Anonymous FTP:          ftp://ftp.sec.gov/edgar/

CIK|Company Name|Form Type|Date Filed|File Name
--------------------------------------------------------------------------------
2041385|Ralliant Corp|10-12B|20250505|edgar/data/2041385/0001104659-25-044355.txt
2041385|Ralliant Corp|10-12B|20250505|edgar/data/2041385/0001104659-25-044355.txt
320193|Apple Inc.|8-K|20250505|edgar/data/320193/0000320193-25-000055.txt
BADROW-NO-PIPES
"""

QUARTERLY_FIXTURE = b"""Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:     June 30, 2025

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
2041385|Ralliant Corp|10-12B|2025-05-05|edgar/data/2041385/0001104659-25-044355.txt
"""


def test_parse_daily_normalizes_undashed_dates_and_dedupes():
    rows, malformed = parse_master_idx(DAILY_FIXTURE)
    assert malformed == 1
    assert len(rows) == 2  # exact duplicate row removed
    assert rows[0].cik == "0002041385"
    assert rows[0].company_name == "Ralliant Corp"
    assert rows[0].form_type == "10-12B"
    assert rows[0].date_filed == "2025-05-05"  # normalized from 20250505
    assert rows[0].accession == "0001104659-25-044355"
    assert rows[1].cik == "0000320193"
    assert rows[1].accession == "0000320193-25-000055"


def test_parse_quarterly_dashed_dates_and_filename_header():
    rows, malformed = parse_master_idx(QUARTERLY_FIXTURE)
    assert malformed == 0
    assert len(rows) == 1
    assert rows[0].date_filed == "2025-05-05"  # passed through unchanged


def test_url_builders_exact():
    assert daily_index_url("2026-06-08") == (
        "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR2/master.20260608.idx"
    )
    assert full_index_url(2024, 1) == (
        "https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/master.idx"
    )


def test_candidate_rows_filters_to_candidate_forms():
    rows, _ = parse_master_idx(DAILY_FIXTURE)
    assert [r.form_type for r in candidate_rows(rows)] == ["10-12B", "8-K"]


def test_candidate_forms_exact_set():
    assert CANDIDATE_FORMS == {
        "10-12B", "10-12B/A", "10-12G", "10-12G/A", "8-A12B", "8-A12B/A",
        "25", "25-NSE", "S-1", "S-1/A", "424B4", "8-K", "EFFECT", "CERT",
    }


def test_parse_handles_accession_with_subdir_filename():
    raw = (
        b"CIK|Company Name|Form Type|Date Filed|File Name\n"
        b"----------------------------------------\n"
        b"1692427|NCS Multistage Holdings, Inc.|425|20260601|"
        b"edgar/data/1692427/0001104659-26-061001.txt\n"
    )
    rows, malformed = parse_master_idx(raw)
    assert malformed == 0
    assert rows[0].form_type == "425"
    assert rows[0].date_filed == "2026-06-01"
    assert rows[0].accession == "0001104659-26-061001"
