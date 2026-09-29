"""The quarterly normalizer labels each fact by the period it measures, reads point-in-time
when asked, and names the XBRL concept behind every row.

SEC stamps every fact with the fiscal year/period of the FILING that carries it. A
first-quarter 10-Q carries last year's first quarter as a comparative and last year-end's
balance sheet, both stamped with this year's "Q1"; keyed on those stamps, fiscal 2026 Q1
revenue came out as fiscal 2025 Q1's figure and quarterly cash as the prior year-end.
Synthetic payloads; values in the normalizer's USD millions.
"""

from __future__ import annotations

from app.ingest.companyfacts import (
    normalize_annual_facts_from_raw,
    normalize_quarterly_facts_from_raw,
)

Q25 = "0000000001-25-000020"  # Q1 fiscal 2025 10-Q, filed 2025-05-01
K25 = "0000000001-26-000010"  # fiscal 2025 10-K, filed 2026-02-20
Q26 = "0000000001-26-000020"  # Q1 fiscal 2026 10-Q, filed 2026-05-01


def _fact(val, end, accn, filed, fy, fp, form, start=None):
    out = {"val": val, "end": end, "accn": accn, "filed": filed, "fy": fy, "fp": fp, "form": form}
    if start:
        out["start"] = start
    return out


def _payload() -> dict:
    revenue = [
        _fact(1_000e6, "2025-03-31", Q25, "2025-05-01", 2025, "Q1", "10-Q", "2025-01-01"),
        _fact(4_200e6, "2025-12-31", K25, "2026-02-20", 2025, "FY", "10-K", "2025-01-01"),
        # The fiscal 2026 Q1 10-Q: its own quarter, then last year's Q1 as a comparative
        # (restated to 990), both stamped fy 2026 / Q1 by SEC.
        _fact(1_100e6, "2026-03-31", Q26, "2026-05-01", 2026, "Q1", "10-Q", "2026-01-01"),
        _fact(990e6, "2025-03-31", Q26, "2026-05-01", 2026, "Q1", "10-Q", "2025-01-01"),
    ]
    cash = [
        _fact(300e6, "2025-03-31", Q25, "2025-05-01", 2025, "Q1", "10-Q"),
        _fact(350e6, "2025-12-31", K25, "2026-02-20", 2025, "FY", "10-K"),
        # The fiscal 2026 Q1 10-Q carries the prior year-end balance sheet, stamped Q1 2026.
        _fact(350e6, "2025-12-31", Q26, "2026-05-01", 2026, "Q1", "10-Q"),
        _fact(410e6, "2026-03-31", Q26, "2026-05-01", 2026, "Q1", "10-Q"),
    ]
    debt_current = [_fact(20e6, "2026-03-31", Q26, "2026-05-01", 2026, "Q1", "10-Q")]
    debt_noncurrent = [_fact(180e6, "2026-03-31", Q26, "2026-05-01", 2026, "Q1", "10-Q")]
    cover = [_fact(50_000_000, "2026-04-25", Q26, "2026-05-01", 2026, "Q1", "10-Q")]
    return {
        "facts": {
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": cover}}},
            "us-gaap": {
                "Revenues": {"units": {"USD": revenue}},
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": cash}},
                "DebtCurrent": {"units": {"USD": debt_current}},
                "LongTermDebtNoncurrent": {"units": {"USD": debt_noncurrent}},
            },
        }
    }


def _by_period(rows, line_item):
    return {
        (r["fiscal_year"], r["period_type"]): (r["value"], r["accession"])
        for r in rows
        if r["line_item"] == line_item
    }


def test_each_quarter_carries_its_own_figure_not_the_comparative():
    rows = normalize_quarterly_facts_from_raw(_payload(), cik="0000000001", years_back=5)
    assert _by_period(rows, "revenue") == {
        (2026, "Q1"): (1_100.0, Q26),
        (2025, "Q1"): (990.0, Q26),  # latest view: the comparative restates last year's Q1
    }
    assert _by_period(rows, "cash") == {
        (2026, "Q1"): (410.0, Q26),  # not the prior year-end's 350
        (2025, "Q1"): (300.0, Q25),
    }
    assert _by_period(rows, "shares_outstanding") == {(2026, "Q1"): (50.0, Q26)}


def test_point_in_time_quarterly_view_ignores_later_filings():
    rows = normalize_quarterly_facts_from_raw(
        _payload(), cik="0000000001", years_back=5, filed_as_of="2026-04-30"
    )
    assert _by_period(rows, "revenue") == {(2025, "Q1"): (1_000.0, Q25)}
    assert _by_period(rows, "cash") == {(2025, "Q1"): (300.0, Q25)}
    assert _by_period(rows, "total_debt") == {}


def test_rows_name_their_xbrl_concept_and_a_sum_names_its_components():
    rows = normalize_quarterly_facts_from_raw(_payload(), cik="0000000001", years_back=5)
    revenue = next(r for r in rows if r["line_item"] == "revenue" and r["fiscal_year"] == 2026)
    assert (revenue["taxonomy"], revenue["tag"], revenue["period_start"]) == (
        "us-gaap",
        "Revenues",
        "2026-01-01",
    )
    shares = next(r for r in rows if r["line_item"] == "shares_outstanding")
    assert (shares["taxonomy"], shares["tag"]) == ("dei", "EntityCommonStockSharesOutstanding")
    debt = next(r for r in rows if r["line_item"] == "total_debt")
    assert debt["value"] == 200.0
    assert debt["tag"] is None and debt["taxonomy"] is None
    assert debt["components"] == [
        {
            "taxonomy": "us-gaap",
            "tag": "LongTermDebtNoncurrent",
            "value": 180.0,
            "filed_date": "2026-05-01",
            "form": "10-Q",
            "accession": Q26,
        },
        {
            "taxonomy": "us-gaap",
            "tag": "DebtCurrent",
            "value": 20.0,
            "filed_date": "2026-05-01",
            "form": "10-Q",
            "accession": Q26,
        },
    ]


def test_annual_rows_carry_their_concept_too():
    rows = normalize_annual_facts_from_raw(_payload(), cik="0000000001", years_back=5)
    revenue = next(r for r in rows if r["line_item"] == "revenue")
    assert (revenue["fiscal_year"], revenue["value"], revenue["tag"], revenue["period_start"]) == (
        2025,
        4_200.0,
        "Revenues",
        "2025-01-01",
    )


# ── Every stored debt and share row keeps its filing date ────────────────────────
# The point-in-time path reads only rows whose filing was public by the as-of, so a row
# without a filing date is invisible there. Every resolver path must carry it.


def _dated_payload() -> dict:
    def fact(val, end, accn, filed, form="10-K", fy=2025, fp="FY"):
        return {"val": val, "end": end, "accn": accn, "filed": filed, "form": form, "fy": fy, "fp": fp}

    return {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [fact(40_000_000, "2026-02-10", "k25", "2026-02-20")]}
                }
            },
            "us-gaap": {
                # 2025: a direct complete total; 2024: current + noncurrent summed;
                # 2023: the instrument-family gap tier (revolver + notes).
                "DebtLongtermAndShorttermCombinedAmount": {
                    "units": {"USD": [fact(500e6, "2025-12-31", "k25", "2026-02-20")]}
                },
                "DebtCurrent": {"units": {"USD": [fact(50e6, "2024-12-31", "k24", "2025-02-20", fy=2024)]}},
                "LongTermDebtNoncurrent": {
                    "units": {"USD": [fact(400e6, "2024-12-31", "k24", "2025-02-20", fy=2024)]}
                },
                "LinesOfCreditCurrent": {
                    "units": {"USD": [fact(30e6, "2023-12-31", "k23", "2024-02-20", fy=2023)]}
                },
                "LongTermNotesPayable": {
                    "units": {"USD": [fact(300e6, "2023-12-31", "k23", "2024-02-20", fy=2023)]}
                },
                "CommonStockSharesOutstanding": {
                    "units": {"shares": [fact(39_000_000, "2024-12-31", "k24", "2025-02-20", fy=2024)]}
                },
            },
        }
    }


def test_every_debt_and_share_row_carries_its_filing():
    rows = normalize_annual_facts_from_raw(
        _dated_payload(), cik="0000000001", years_back=10, filed_as_of="2026-09-28"
    )
    got = {
        (r["line_item"], r["fiscal_year"]): (r["value"], r["filed_date"], r["accession"])
        for r in rows
        if r["line_item"] in ("total_debt", "shares_outstanding")
    }
    assert got == {
        ("total_debt", 2025): (500.0, "2026-02-20", "k25"),
        ("total_debt", 2024): (450.0, "2025-02-20", "k24"),
        ("total_debt", 2023): (330.0, "2024-02-20", "k23"),
        ("shares_outstanding", 2025): (40.0, "2026-02-20", "k25"),
        ("shares_outstanding", 2024): (39.0, "2025-02-20", "k24"),
    }
