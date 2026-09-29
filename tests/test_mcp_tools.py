"""The six MCP tool implementations, called directly (no MCP SDK involved)."""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta

import pytest

from app.ingest import companyfacts
from app.mcp_server import companies, filing_text, financials
from app.mcp_server.errors import SecToolError
from tests.sec_support import ACME_DOC_URL, FIXTURES, install_fake_sec, submissions_url

K23, K24, K25 = "0000999999-23-000010", "0000999999-24-000010", "0000999999-25-000010"
Q23, Q24, Q25 = "0000999999-23-000020", "0000999999-24-000020", "0000999999-25-000020"
ACME = {"cik": "0000999999", "ticker": "ACME", "name": "ACME WIDGETS INC"}


@pytest.fixture
def fake_sec(monkeypatch):
    yield from install_fake_sec(monkeypatch)


# ---------------------------------------------------------------------------
# lookup_company
# ---------------------------------------------------------------------------


def test_lookup_company_by_ticker_lists_exact_ticker_first(fake_sec):
    result = companies.lookup_company("acme")

    assert result == {
        "query": "acme",
        "match_count": 2,
        "matches": [
            {
                "ticker": "ACME",
                "cik": "0000999999",
                "name": "ACME WIDGETS INC",
                "exchange": "Nasdaq",
                "matched_on": "ticker",
            },
            {
                "ticker": "ACMR",
                "cik": "0000777777",
                "name": "ACME ROCKETS CORP",
                "exchange": None,
                "matched_on": "name",
            },
        ],
    }


def test_lookup_company_by_name_ranks_exact_name_before_prefix(fake_sec):
    result = companies.lookup_company("Coca-Cola")

    assert [(m["ticker"], m["matched_on"]) for m in result["matches"]] == [
        ("KO", "name"),
        ("COKE", "name"),
    ]


def test_lookup_company_ignores_apostrophes_like_sec_names_do(fake_sec):
    result = companies.lookup_company("Joe\u2019s Jeans")

    assert [(m["ticker"], m["name"]) for m in result["matches"]] == [("JOES", "JOES JEANS INC.")]
    assert companies.lookup_company("joe's")["matches"][0]["ticker"] == "JOES"


def test_lookup_company_by_cik_returns_every_share_class(fake_sec):
    result = companies.lookup_company("CIK0001067983")

    assert [(m["ticker"], m["cik"], m["exchange"]) for m in result["matches"]] == [
        ("BRK-B", "0001067983", "NYSE"),
        ("BRK-A", "0001067983", "NYSE"),
    ]


def test_lookup_company_by_cik_outside_ticker_list_reads_submissions(fake_sec):
    fake_sec.add_json(
        submissions_url("0000888888"),
        {"cik": "0000888888", "name": "OLD MILL TRUST", "tickers": [], "exchanges": []},
    )

    result = companies.lookup_company("888888")

    assert result["matches"] == [
        {
            "ticker": None,
            "cik": "0000888888",
            "name": "OLD MILL TRUST",
            "exchange": None,
            "matched_on": "cik",
        }
    ]


def test_lookup_company_not_found_returns_empty_matches_with_hint(fake_sec):
    result = companies.lookup_company("zzzz qqqq")

    assert result == {
        "query": "zzzz qqqq",
        "match_count": 0,
        "matches": [],
        "hint": (
            "No SEC registrant matched. Try a shorter name fragment, the exact ticker, "
            "or the CIK from EDGAR full-text search."
        ),
    }


def test_resolve_company_accepts_dotted_share_class(fake_sec):
    assert companies.resolve_company("brk.b") == companies.Company(
        cik="0001067983", ticker="BRK-B", name="BERKSHIRE HATHAWAY INC"
    )


# ---------------------------------------------------------------------------
# get_company_profile
# ---------------------------------------------------------------------------


def test_get_company_profile_happy_path(fake_sec):
    assert companies.get_company_profile("ACME") == {
        "cik": "0000999999",
        "name": "ACME WIDGETS INC",
        "tickers": ["ACME"],
        "exchanges": ["Nasdaq"],
        "sic": "3559",
        "sic_description": "Special Industry Machinery, NEC",
        "entity_type": "operating",
        "filer_category": "Non-accelerated filer",
        "fiscal_year_end": "12-31",
        "state_of_incorporation": "DE",
        "ein": "123456789",
        "business_address": "1 MAIN ST, SPRINGFIELD, IL 62701",
        "phone": "555-0100",
        "former_names": [{"name": "ACME GADGETS INC", "from": "2001-01-01", "to": "2015-06-30"}],
        "latest_annual_report": {
            "form": "10-K",
            "filed": "2025-02-14",
            "report_date": "2024-12-31",
            "accession": K25,
        },
        "latest_quarterly_report": {
            "form": "10-Q",
            "filed": "2025-05-01",
            "report_date": "2025-03-31",
            "accession": Q25,
        },
        "source": "https://data.sec.gov/submissions/CIK0000999999.json",
    }


def test_get_company_profile_unknown_ticker(fake_sec):
    with pytest.raises(SecToolError) as err:
        companies.get_company_profile("ZZZZ")
    assert str(err.value) == (
        "Unknown ticker 'ZZZZ'. Use lookup_company to search by name, or pass the CIK."
    )


def test_get_company_profile_unknown_cik(fake_sec):
    with pytest.raises(SecToolError) as err:
        companies.get_company_profile("42")
    assert str(err.value) == "No SEC filer has CIK 0000000042."


# ---------------------------------------------------------------------------
# list_filings
# ---------------------------------------------------------------------------


def test_list_filings_newest_first_without_touching_older_pages(fake_sec):
    result = companies.list_filings("ACME", forms=["10-K"], limit=2)

    assert result["company"] == ACME
    assert result["filters"] == {"forms": ["10-K"], "since": None, "until": None, "limit": 2}
    assert result["returned"] == 2
    assert result["more_available"] is True
    assert result["filings"] == [
        {
            "accession": K25,
            "form": "10-K",
            "filed": "2025-02-14",
            "report_date": "2024-12-31",
            "primary_document_url": ACME_DOC_URL,
            "index_url": (
                "https://www.sec.gov/Archives/edgar/data/999999/000099999925000010/"
                "0000999999-25-000010-index.htm"
            ),
            "description": "10-K",
        },
        {
            "accession": K24,
            "form": "10-K",
            "filed": "2024-02-15",
            "report_date": "2023-12-31",
            "primary_document_url": (
                "https://www.sec.gov/Archives/edgar/data/999999/000099999924000010/acme-20231231.htm"
            ),
            "index_url": (
                "https://www.sec.gov/Archives/edgar/data/999999/000099999924000010/"
                "0000999999-24-000010-index.htm"
            ),
            "description": "10-K",
        },
    ]
    assert not any("submissions-001" in url for url in fake_sec.calls)


def test_list_filings_reads_older_pages_when_needed(fake_sec):
    result = companies.list_filings("ACME", forms="10-K, 10-K/A", since="2010-01-01", limit=10)

    assert [(f["accession"], f["form"], f["filed"]) for f in result["filings"]] == [
        (K25, "10-K", "2025-02-14"),
        (K24, "10-K", "2024-02-15"),
        (K23, "10-K", "2023-02-15"),
        ("0000999999-19-000010", "10-K", "2019-02-20"),
        ("0000999999-12-000005", "10-K/A", "2012-06-01"),
    ]
    assert result["more_available"] is False
    assert "https://data.sec.gov/submissions/CIK0000999999-submissions-001.json" in fake_sec.calls


def test_list_filings_date_window_and_8k_items(fake_sec):
    result = companies.list_filings("999999", since="2025-04-01", until="2025-05-31")

    assert [(f["form"], f.get("items")) for f in result["filings"]] == [
        ("8-K", "2.02,9.01"),
        ("10-Q", None),
    ]


def test_list_filings_no_match_has_hint(fake_sec):
    result = companies.list_filings("ACME", forms=["S-1"])

    assert result["returned"] == 0
    assert result["filings"] == []
    assert result["hint"] == (
        "No filings matched. Form names are exact (amendments are separate, e.g. 10-K/A); "
        "widen the date range or drop the form filter."
    )


def test_list_filings_rejects_bad_dates(fake_sec):
    with pytest.raises(SecToolError) as err:
        companies.list_filings("ACME", since="2024/01/01")
    assert str(err.value) == "since must be a date in YYYY-MM-DD form, got '2024/01/01'."


# ---------------------------------------------------------------------------
# get_filing_text
# ---------------------------------------------------------------------------

ACME_TEXT_CHARS = 1308
ACME_SECTIONS = [
    {"heading": "PART I", "offset": 172},
    {"heading": "Item 1. Business", "offset": 180},
    {"heading": "Item 1A. Risk Factors", "offset": 532},
    {"heading": "PART II", "offset": 830},
    {
        "heading": (
            "Item 7. Management’s Discussion and Analysis of Financial Condition and "
            "Results of Operations"
        ),
        "offset": 839,
    },
    {"heading": "Item 8. Financial Statements and Supplementary Data", "offset": 1191},
]


def test_get_filing_text_cleans_inline_xbrl(fake_sec):
    result = filing_text.get_filing_text(ACME_DOC_URL)

    assert {k: v for k, v in result.items() if k != "text"} == {
        "url": ACME_DOC_URL,
        "accession": K25,
        "total_chars": ACME_TEXT_CHARS,
        "offset": 0,
        "returned_chars": ACME_TEXT_CHARS,
        "next_offset": None,
        "sections": ACME_SECTIONS,
    }
    text = result["text"]
    for noise in ("SCRIPT-NOISE", "HIDDEN-CONTEXT-NOISE", "COMMENT-NOISE", "display", "<", "&amp;"):
        assert noise not in text.replace("< 15%", "")
    assert text.startswith("UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\n\nFORM 10-K\n\n")
    assert "industrial widgets & gadgets to manufacturers in North America." in text
    assert "Revenues $ 120.0 $ 95.0\n\nNet income $ 12.0 $ 9.0" in text
    assert "Operating margin stayed < 15% because steel costs rose." in text
    assert text[839:].startswith("Item 7. Management’s Discussion")


def test_get_filing_text_pages_through_the_document(fake_sec):
    full = filing_text.get_filing_text(ACME_DOC_URL)["text"]
    pages = []
    offset = 0
    while offset is not None:
        page = filing_text.get_filing_text(ACME_DOC_URL, max_chars=500, offset=offset)
        pages.append(page)
        offset = page["next_offset"]

    assert [(p["offset"], p["returned_chars"], p["next_offset"]) for p in pages] == [
        (0, 500, 500),
        (500, 500, 1000),
        (1000, 308, None),
    ]
    assert "".join(p["text"] for p in pages) == full
    assert "sections" in pages[0] and "sections" not in pages[1]
    # One download serves every page.
    assert fake_sec.calls.count(ACME_DOC_URL) == 1


def test_get_filing_text_by_accession(fake_sec):
    result = filing_text.get_filing_text(K25, ticker_or_cik="ACME", max_chars=600)

    assert {k: result[k] for k in ("url", "accession", "form", "filed", "next_offset")} == {
        "url": ACME_DOC_URL,
        "accession": K25,
        "form": "10-K",
        "filed": "2025-02-14",
        "next_offset": 600,
    }


def test_get_filing_text_undashed_accession_in_older_page(fake_sec):
    fake_sec.add_bytes(
        "https://www.sec.gov/Archives/edgar/data/999999/000099999912000005/acme-10ka.htm",
        b"<html><body><p>Amendment No. 1</p></body></html>",
    )

    result = filing_text.get_filing_text("000099999912000005", ticker_or_cik="ACME")

    assert (result["form"], result["text"]) == ("10-K/A", "Amendment No. 1")


def test_get_filing_text_accession_needs_company(fake_sec):
    with pytest.raises(SecToolError) as err:
        filing_text.get_filing_text(K25)
    assert str(err.value) == (
        "An accession number needs ticker_or_cik (the filer), or pass the document URL "
        "from list_filings instead."
    )


def test_get_filing_text_unknown_accession(fake_sec):
    with pytest.raises(SecToolError) as err:
        filing_text.get_filing_text("0000999999-99-000001", ticker_or_cik="ACME")
    assert str(err.value) == (
        "Accession 0000999999-99-000001 is not among the filings of CIK 0000999999. "
        "Check the accession with list_filings, or pass the document URL instead."
    )


@pytest.mark.parametrize(
    ("url", "message"),
    [
        (
            "https://example.com/acme-20241231.htm",
            "Only sec.gov documents can be fetched; 'example.com' is not allowed.",
        ),
        (
            "https://www.sec.gov.evil.example/Archives/x.htm",
            "Only sec.gov documents can be fetched; 'www.sec.gov.evil.example' is not allowed.",
        ),
        (
            "https://notsec.gov/Archives/x.htm",
            "Only sec.gov documents can be fetched; 'notsec.gov' is not allowed.",
        ),
        (
            "ftp://www.sec.gov/Archives/x.htm",
            "Not a web URL: 'ftp://www.sec.gov/Archives/x.htm'. Pass an https://www.sec.gov/... "
            "URL or an accession.",
        ),
        (
            "file:///etc/passwd",
            "Not a web URL: 'file:///etc/passwd'. Pass an https://www.sec.gov/... URL or an "
            "accession.",
        ),
        (
            "https://user:secret@www.sec.gov/Archives/x.htm",
            "sec.gov URLs with credentials or custom ports are not allowed.",
        ),
    ],
)
def test_get_filing_text_rejects_non_sec_urls(fake_sec, url, message):
    with pytest.raises(SecToolError) as err:
        filing_text.get_filing_text(url)
    assert str(err.value) == message
    assert fake_sec.calls == []


def test_get_filing_text_unwraps_inline_viewer_links(fake_sec):
    viewer = (
        "https://www.sec.gov/ix?doc=/Archives/edgar/data/999999/000099999925000010/acme-20241231.htm"
    )
    assert filing_text.get_filing_text(viewer, max_chars=500)["url"] == ACME_DOC_URL


def test_get_filing_text_offset_past_end(fake_sec):
    with pytest.raises(SecToolError) as err:
        filing_text.get_filing_text(ACME_DOC_URL, offset=5000)
    assert str(err.value) == "offset 5000 is past the end of the document (total_chars=1308)."


def test_get_filing_text_refuses_pdf(fake_sec):
    url = "https://www.sec.gov/Archives/edgar/data/999999/000099999925000010/exhibit.pdf"
    fake_sec.add_bytes(url, b"%PDF-1.7 binary")
    with pytest.raises(SecToolError) as err:
        filing_text.get_filing_text(url)
    assert str(err.value) == "This document is a PDF; only HTML and text documents are supported."


def test_plain_text_documents_pass_through(fake_sec):
    url = "https://www.sec.gov/Archives/edgar/data/999999/000099999912000005/0000999999-12-000005.txt"
    fake_sec.add_bytes(url, b"ANNUAL REPORT\r\n\r\n\r\n\r\nItem 1.   Business\r\nWidgets.")
    result = filing_text.get_filing_text(url)
    assert result["text"] == "ANNUAL REPORT\n\nItem 1. Business\nWidgets."


# ---------------------------------------------------------------------------
# get_financials
# ---------------------------------------------------------------------------


def _values(result: dict, *items: str) -> list[tuple]:
    return [
        (p["fiscal_year"], p["fiscal_period"])
        + tuple(
            (p["values"][item]["value"], p["values"][item].get("accn"))
            if item in p["values"]
            else None
            for item in items
        )
        for p in result["periods"]
    ]


def test_get_financials_annual_latest_view_with_provenance(fake_sec):
    result = financials.get_financials(
        "ACME",
        years=30,
        line_items=["revenue", "net_income", "cash", "total_debt", "shares_outstanding"],
    )

    assert result["company"] == ACME
    assert result["period"] == "annual"
    assert result["as_of"] is None
    assert result["periods"] == [
        {
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "start": "2024-01-01",
            "end": "2024-12-31",
            "values": {
                "revenue": {"value": 120.0, "tag": "us-gaap:Revenues", "accn": K25},
                "net_income": {"value": 12.0, "tag": "us-gaap:NetIncomeLoss", "accn": K25},
                "cash": {
                    "value": 55.0,
                    "tag": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
                    "accn": K25,
                },
                "total_debt": {
                    "value": 25.0,
                    "sum_of": [
                        {"tag": "us-gaap:LongTermDebtNoncurrent", "value": 20.0, "accn": K25},
                        {"tag": "us-gaap:DebtCurrent", "value": 5.0, "accn": K25},
                    ],
                },
                "shares_outstanding": {
                    "value": 10.0,
                    "tag": "dei:EntityCommonStockSharesOutstanding",
                    "accn": K25,
                    "end": "2025-02-10",
                },
            },
        },
        {
            "fiscal_year": 2023,
            "fiscal_period": "FY",
            "start": "2023-01-01",
            "end": "2023-12-31",
            "values": {
                "revenue": {"value": 95.0, "tag": "us-gaap:Revenues", "accn": K25},
                "net_income": {"value": 9.0, "tag": "us-gaap:NetIncomeLoss", "accn": K25},
                "cash": {
                    "value": 45.0,
                    "tag": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
                    "accn": K25,
                },
                "shares_outstanding": {
                    "value": 10.5,
                    "tag": "dei:EntityCommonStockSharesOutstanding",
                    "accn": K24,
                    "end": "2024-02-10",
                },
            },
        },
        {
            "fiscal_year": 2022,
            "fiscal_period": "FY",
            "start": "2022-01-01",
            "end": "2022-12-31",
            "values": {
                "revenue": {"value": 80.0, "tag": "us-gaap:Revenues", "accn": K24},
                "cash": {
                    "value": 40.0,
                    "tag": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
                    "accn": K24,
                },
            },
        },
    ]
    assert result["filings"] == {
        K25: {"form": "10-K", "filed": "2025-02-14"},
        K24: {"form": "10-K", "filed": "2024-02-15"},
    }
    assert result["source"] == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000999999.json"
    assert result["notes"][-1] == (
        "Latest view: where a period was restated, the most recently filed value is shown."
    )


def test_get_financials_as_of_excludes_later_restatement(fake_sec):
    result = financials.get_financials(
        "ACME", as_of="2025-02-13", line_items=["revenue", "cash"]
    )

    assert _values(result, "revenue", "cash") == [
        (2023, "FY", (100.0, K24), (45.0, K24)),
        (2022, "FY", (80.0, K24), (40.0, K24)),
    ]
    assert result["as_of"] == "2025-02-13"
    assert result["filings"] == {K24: {"form": "10-K", "filed": "2024-02-15"}}
    assert result["notes"][-1] == (
        "Point-in-time: only facts filed on or before 2025-02-13 are used, so later "
        "restatements are excluded."
    )


def test_get_financials_as_of_includes_facts_filed_on_that_day(fake_sec):
    result = financials.get_financials("ACME", as_of="2025-02-14", line_items=["revenue"])

    assert _values(result, "revenue") == [
        (2024, "FY", (120.0, K25)),
        (2023, "FY", (95.0, K25)),
        (2022, "FY", (80.0, K24)),
    ]


def test_get_financials_as_of_before_first_annual_report(fake_sec):
    result = financials.get_financials("ACME", as_of="2024-02-14", line_items=["revenue", "cash"])

    assert _values(result, "revenue", "cash") == [(2022, "FY", (80.0, K23), (40.0, K23))]


def test_get_financials_never_returns_a_fact_filed_after_as_of(fake_sec):
    day = date(2022, 12, 1)
    while day <= date(2025, 6, 1):
        for period in ("annual", "quarterly"):
            result = financials.get_financials(
                "ACME", period=period, as_of=day.isoformat(), line_items=["all"]
            )
            filed = [entry["filed"] for entry in result["filings"].values()]
            assert all(f <= day.isoformat() for f in filed), (day, period, filed)
        day += timedelta(days=11)


def test_get_financials_quarterly_labels_comparatives_by_their_own_period(fake_sec):
    result = financials.get_financials("ACME", period="quarterly", line_items=["revenue", "cash"])

    assert [
        (p["fiscal_year"], p["fiscal_period"], p["start"], p["end"]) for p in result["periods"]
    ] == [
        (2025, "Q1", "2025-01-01", "2025-03-31"),
        (2024, "Q1", "2024-01-01", "2024-03-31"),
        (2023, "Q1", "2023-01-01", "2023-03-31"),
    ]
    assert _values(result, "revenue", "cash") == [
        (2025, "Q1", (33.0, Q25), (60.0, Q25)),
        (2024, "Q1", (29.0, Q25), (50.0, Q24)),
        (2023, "Q1", (25.0, Q24), (48.0, Q23)),
    ]


def test_get_financials_quarterly_as_of(fake_sec):
    result = financials.get_financials(
        "ACME", period="quarterly", as_of="2025-04-30", line_items=["revenue"]
    )

    assert _values(result, "revenue") == [
        (2024, "Q1", (30.0, Q24)),
        (2023, "Q1", (25.0, Q24)),
    ]


def test_get_financials_real_restatement_point_in_time(fake_sec):
    # Trimmed from SEC's real company-facts JSON: Coca-Cola recast 2017 and 2018
    # revenue in its 2019 10-K (filed 2020-02-24).
    latest = financials.get_financials("KO", years=30, line_items=["revenue"])
    as_filed = financials.get_financials("KO", as_of="2019-03-01", line_items=["revenue"])
    day_of = financials.get_financials("KO", as_of="2020-02-24", line_items=["revenue"])

    assert _values(latest, "revenue") == [
        (2018, "FY", (34300.0, "0000021344-21-000008")),
        (2017, "FY", (36212.0, "0000021344-20-000006")),
        (2016, "FY", (41863.0, "0000021344-19-000014")),
    ]
    assert _values(as_filed, "revenue") == [
        (2018, "FY", (31856.0, "0000021344-19-000014")),
        (2017, "FY", (35410.0, "0000021344-19-000014")),
        (2016, "FY", (41863.0, "0000021344-19-000014")),
    ]
    assert _values(day_of, "revenue")[0] == (2018, "FY", (34300.0, "0000021344-20-000006"))
    assert latest["company"] == {"cik": "0000021344", "ticker": "KO", "name": "COCA COLA CO"}


def test_get_financials_all_line_items_and_unknown_ones(fake_sec):
    result = financials.get_financials("ACME", years=1, line_items=["all"])
    assert result["line_items"] == list(financials.ALL_LINE_ITEMS)
    assert result["periods"][0]["values"]["inventory"] == {
        "value": 8.5,
        "tag": "us-gaap:InventoryNet",
        "accn": K25,
    }

    with pytest.raises(SecToolError) as err:
        financials.get_financials("ACME", line_items=["revenue", "ebitda"])
    assert str(err.value).startswith("Unknown line item(s): ebitda. Valid names: revenue, ")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"period": "monthly"}, "period must be 'annual' or 'quarterly'."),
        ({"years": 0}, "years must be between 1 and 30."),
        ({"as_of": "03/01/2019"}, "as_of must be a date in YYYY-MM-DD form, got '03/01/2019'."),
    ],
)
def test_get_financials_validates_arguments(fake_sec, kwargs, message):
    with pytest.raises(SecToolError) as err:
        financials.get_financials("ACME", **kwargs)
    assert str(err.value) == message


def test_get_financials_company_without_xbrl(fake_sec):
    with pytest.raises(SecToolError) as err:
        financials.get_financials("888888")
    assert str(err.value) == (
        "SEC has no XBRL financial data for CIK 0000888888. Company facts exist only for filers "
        "that tag their statements in XBRL (most operating companies since 2009-2011); funds, "
        "trusts and many older or foreign filers have none. list_filings and get_filing_text "
        "still work."
    )


# ---------------------------------------------------------------------------
# get_concept
# ---------------------------------------------------------------------------


def test_get_concept_annual_series_with_first_reported(fake_sec):
    result = financials.get_concept("ACME", "Revenues", period="annual")

    assert result["concept"] == "us-gaap:Revenues"
    assert result["unit"] == "USD"
    assert result["available_units"] == ["USD"]
    assert result["count"] == 3
    assert result["points"] == [
        {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "value": 120000000,
            "months": 12,
            "fy": 2024,
            "fp": "FY",
            "form": "10-K",
            "filed": "2025-02-14",
            "accn": K25,
        },
        {
            "start": "2023-01-01",
            "end": "2023-12-31",
            "value": 95000000,
            "months": 12,
            "fy": 2024,
            "fp": "FY",
            "form": "10-K",
            "filed": "2025-02-14",
            "accn": K25,
            "first_reported": {"value": 100000000, "filed": "2024-02-15", "accn": K24},
        },
        {
            "start": "2022-01-01",
            "end": "2022-12-31",
            "value": 80000000,
            "months": 12,
            "fy": 2023,
            "fp": "FY",
            "form": "10-K",
            "filed": "2024-02-15",
            "accn": K24,
        },
    ]


def test_get_concept_as_of_and_quarterly_filter(fake_sec):
    result = financials.get_concept("ACME", "revenues", period="quarterly", as_of="2025-04-30")

    assert result["concept"] == "us-gaap:Revenues"
    assert [(p["end"], p["value"], p["accn"]) for p in result["points"]] == [
        ("2024-03-31", 30000000, Q24),
        ("2023-03-31", 25000000, Q24),
    ]


def test_get_concept_instant_dei_series_and_limit(fake_sec):
    result = financials.get_concept(
        "ACME", "dei:EntityCommonStockSharesOutstanding", period="instant", limit=1
    )

    assert (result["concept"], result["unit"], result["count"], result["returned"]) == (
        "dei:EntityCommonStockSharesOutstanding",
        "shares",
        2,
        1,
    )
    assert result["truncated"] is True
    assert result["points"] == [
        {"end": "2025-02-10", "value": 10000000, "fy": 2024, "fp": "FY", "form": "10-K",
         "filed": "2025-02-14", "accn": K25}
    ]


def test_get_concept_unknown_concept_suggests_similar(fake_sec):
    with pytest.raises(SecToolError) as err:
        financials.get_concept("ACME", "Revenue")
    assert str(err.value) == (
        "ACME has no us-gaap:Revenue facts. Similar concepts this company reports: Revenues."
    )


def test_get_concept_unknown_unit(fake_sec):
    with pytest.raises(SecToolError) as err:
        financials.get_concept("ACME", "Revenues", unit="EUR")
    assert str(err.value) == "us-gaap:Revenues has no unit 'EUR'; available: USD."


# ---------------------------------------------------------------------------
# point-in-time and period-label primitives
# ---------------------------------------------------------------------------


def _acme_raw() -> dict:
    return json.loads((FIXTURES / "companyfacts_CIK0000999999.json").read_text(encoding="utf-8"))


# These primitives moved into the normalizer (app.ingest.companyfacts) when the
# server stopped pre-filtering and relabelling the payload itself; the tests follow.


def test_point_in_time_normalization_skips_undated_and_later_facts_without_mutating_input():
    raw = _acme_raw()
    raw["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
        {"end": "2021-12-31", "start": "2021-01-01", "val": 1, "accn": "x", "form": "10-K"}
    )  # undated: never visible
    before = copy.deepcopy(raw)

    rows = companyfacts.normalize_annual_facts_from_raw(
        raw, cik="0000999999", years_back=30, filed_as_of="2024-02-15"
    )
    quarterly = companyfacts.normalize_quarterly_facts_from_raw(
        raw, cik="0000999999", years_back=30, filed_as_of="2024-02-15"
    )

    assert raw == before
    revenue = sorted((r["fiscal_year"], r["value"], r["filed_date"]) for r in rows if r["line_item"] == "revenue")
    assert revenue == [(2022, 80.0, "2024-02-15"), (2023, 100.0, "2024-02-15")]
    assert all(r["filed_date"] <= "2024-02-15" for r in rows + quarterly)
    assert not [r for r in rows if r["line_item"] == "total_debt"]


def test_fiscal_calendar_maps_each_filing_period_end():
    assert companyfacts.fiscal_calendar(_acme_raw()) == {
        date(2022, 12, 31): (2022, "FY"),
        date(2023, 3, 31): (2023, "Q1"),
        date(2023, 12, 31): (2023, "FY"),
        date(2024, 3, 31): (2024, "Q1"),
        date(2024, 12, 31): (2024, "FY"),
        date(2025, 3, 31): (2025, "Q1"),
    }
    assert companyfacts.fiscal_calendar(_acme_raw(), filed_as_of="2024-02-15") == {
        date(2022, 12, 31): (2022, "FY"),
        date(2023, 3, 31): (2023, "Q1"),
        date(2023, 12, 31): (2023, "FY"),
    }


def test_quarterly_normalization_never_files_a_prior_year_end_balance_under_a_quarter():
    rows = companyfacts.normalize_quarterly_facts_from_raw(_acme_raw(), cik="0000999999", years_back=30)
    cash = {(r["fiscal_year"], r["period_type"]): (r["period_end"], r["value"]) for r in rows if r["line_item"] == "cash"}
    assert cash == {
        (2025, "Q1"): ("2025-03-31", 60.0),
        (2024, "Q1"): ("2024-03-31", 50.0),
        (2023, "Q1"): ("2023-03-31", 48.0),
    }
