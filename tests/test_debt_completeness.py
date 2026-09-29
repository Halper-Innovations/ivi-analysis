"""Total debt must be complete, on both the as-of path and the normalizer.

A candidate whose own definition is not every borrowing (long-term debt, a noncurrent line,
one instrument family) is completed from the same balance sheet's lines whose scopes are
disjoint by definition: long-term debt including current maturities, plus short-term
borrowings. Whatever the answer, a balance sheet that shows noncurrent long-term debt N and a
current debt line C carries at least N + max(C) of debt; a total below that floor (beyond a
2% allowance for issuance costs against principal) is refused: UNKNOWN on the as-of path, no
row in the normalizer. The floor only refuses; it is never returned as a total.
Values are synthetic except Coca-Cola's (a trimmed real payload); the normalizer works in USD
millions, the as-of path in USD.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ingest.companyfacts import normalize_annual_facts_from_raw
from app.market.company_facts_extract import (
    assemble_total_debt,
    debt_completeness_floor,
    extract_total_debt_asof,
    resolve_complete_total_debt,
)


def _raw(tags: dict[str, float], *, end: str = "2025-12-31", filed: str = "2026-02-20") -> dict:
    return {
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {
                                "val": value,
                                "end": end,
                                "filed": filed,
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "accn": "0000000001-26-000001",
                            }
                        ]
                    }
                }
                for tag, value in tags.items()
            }
        }
    }


def _annual_debt(raw: dict) -> list[float]:
    return [
        row["value"]
        for row in normalize_annual_facts_from_raw(raw, cik="0000000001", filed_as_of="2026-09-09")
        if row["line_item"] == "total_debt"
    ]


def test_the_floor_is_noncurrent_plus_the_largest_current_line_never_their_sum():
    assert debt_completeness_floor({}) is None
    assert debt_completeness_floor({"LongTermDebtNoncurrent": 900.0}) == 900.0
    assert (
        debt_completeness_floor(
            {"LongTermDebtNoncurrent": 900.0, "LongTermDebtCurrent": 100.0, "ShortTermBorrowings": 60.0}
        )
        == 1_000.0
    )
    assert debt_completeness_floor({"CommercialPaper": 250.0}) == 250.0


def test_a_consistent_long_term_total_is_kept_on_both_paths():
    # Principal due next year (101) sits just above the carrying current portion (100):
    # inside the 2% allowance, so the 1,000 long-term total stands.
    raw = _raw(
        {
            "LongTermDebt": 1_000_000_000,
            "LongTermDebtNoncurrent": 900_000_000,
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": 101_000_000,
        }
    )
    assert _annual_debt(raw) == [1_000.0]
    fact = extract_total_debt_asof(raw, "2026-09-09")
    assert fact["value"] == 1_000_000_000.0
    assert fact["tag"] == "LongTermDebt"


def test_long_term_debt_is_completed_with_its_short_term_borrowings_on_both_paths():
    # 100 of long-term debt and 50 of short-term borrowings: long-term debt alone is a
    # narrow amount; the balance sheet's own lines assemble the complete 150.
    raw = _raw(
        {
            "LongTermDebt": 100_000_000,
            "LongTermDebtNoncurrent": 100_000_000,
            "ShortTermBorrowings": 50_000_000,
        }
    )
    assert _annual_debt(raw) == [150.0]
    fact = extract_total_debt_asof(raw, "2026-09-09")
    assert fact["value"] == 150_000_000.0
    assert fact["tag"] == "LongTermDebt_plus_ShortTermBorrowings"


def test_noncurrent_debt_alone_is_completed_with_its_current_portion():
    raw = _raw({"LongTermDebtNoncurrent": 900_000_000, "LongTermDebtCurrent": 100_000_000})
    # The as-of path already summed the two aligned components; the normalizer used to
    # fall back to the 900 noncurrent amount alone.
    assert extract_total_debt_asof(raw, "2026-09-09")["value"] == 1_000_000_000.0
    assert _annual_debt(raw) == [1_000.0]


def test_noncurrent_debt_with_only_a_principal_schedule_is_unknown():
    # A current maturity is disclosed (principal due next year) but no carrying current
    # line: the complete total cannot be assembled, and 900 is not it.
    raw = _raw(
        {
            "LongTermDebtNoncurrent": 900_000_000,
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": 100_000_000,
        }
    )
    assert _annual_debt(raw) == []
    assert extract_total_debt_asof(raw, "2026-09-09") is None


def test_parts_that_add_up_to_less_than_the_candidate_do_not_replace_it():
    raw = _raw(
        {
            "NotesPayable": 1_000_000_000,
            "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities": 600_000_000,
        }
    )
    assert _annual_debt(raw) == [1_000.0]
    assert extract_total_debt_asof(raw, "2026-09-09")["value"] == 1_000_000_000.0


def test_principal_due_next_year_does_not_override_the_carrying_current_portion():
    # A 10-Q whose "due in the next twelve months" figure is for another window (PVH:
    # 1,168 against a carrying current portion of 12.9): the balance sheet's own current
    # line decides, and the long-term total stands.
    raw = _raw(
        {
            "LongTermDebt": 2_282_300_000,
            "LongTermDebtNoncurrent": 2_269_400_000,
            "LongTermDebtCurrent": 12_900_000,
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": 1_168_100_000,
            "ShortTermBorrowings": 0,
        }
    )
    assert _annual_debt(raw) == [2_282.3]
    fact = extract_total_debt_asof(raw, "2026-09-09")
    assert (fact["value"], fact["tag"]) == (2_282_300_000.0, "LongTermDebt")


def test_a_complete_tag_below_its_own_balance_sheet_is_replaced_by_the_assembled_total():
    # Walmart's DebtLongtermAndShorttermCombinedAmount (40,783) is its long-term debt
    # including current maturities; the same balance sheet carries 10,673 of short-term
    # borrowings on top.
    raw = _raw(
        {
            "DebtLongtermAndShorttermCombinedAmount": 40_783_000_000,
            "LongTermDebtNoncurrent": 36_887_000_000,
            "LongTermDebtCurrent": 3_896_000_000,
            "ShortTermBorrowings": 10_673_000_000,
        }
    )
    assert _annual_debt(raw) == [51_456.0]
    assert extract_total_debt_asof(raw, "2026-09-09")["value"] == 51_456_000_000.0


def test_a_noncurrent_line_is_the_whole_of_long_term_debt_only_without_any_current_portion():
    # Clorox: noncurrent debt 2,487 and commercial paper 1,591, no current portion at all.
    raw = _raw({"LongTermDebtNoncurrent": 2_487_000_000, "CommercialPaper": 1_591_000_000})
    assert _annual_debt(raw) == [4_078.0]
    assert extract_total_debt_asof(raw, "2026-09-09")["value"] == 4_078_000_000.0


def test_a_lease_inclusive_candidate_is_never_swapped_for_a_smaller_lease_free_total():
    # Noncurrent debt and finance leases 1,000; lease-free long-term debt 600. No
    # lease-inclusive current line, so nothing is assembled and the candidate stands.
    raw = _raw({"LongTermDebtAndCapitalLeaseObligations": 1_000_000_000, "LongTermDebt": 600_000_000})
    assert _annual_debt(raw) == [1_000.0]
    assert extract_total_debt_asof(raw, "2026-09-09")["value"] == 1_000_000_000.0


def test_a_complete_direct_total_passes_the_floor():
    raw = _raw(
        {
            "DebtLongtermAndShorttermCombinedAmount": 1_200_000_000,
            "LongTermDebtNoncurrent": 900_000_000,
            "LongTermDebtCurrent": 100_000_000,
            "ShortTermBorrowings": 200_000_000,
        }
    )
    assert _annual_debt(raw) == [1_200.0]
    assert extract_total_debt_asof(raw, "2026-09-09")["value"] == 1_200_000_000.0


def test_an_instrument_carrying_amount_alone_is_not_total_debt():
    raw = _raw({"DebtInstrumentCarryingAmount": 500_000_000})
    assert _annual_debt(raw) == []
    assert extract_total_debt_asof(raw, "2026-09-09") is None


def test_coca_cola_total_debt_includes_its_current_maturities_and_commercial_paper():
    """Coca-Cola's FY2025 10-K: noncurrent long-term debt and leases 42,119; current
    maturities 1,822; commercial paper 1,495; other short-term borrowings 56. The stored
    total_debt was the noncurrent 42,119 alone, understating net debt by 3.37 billion."""
    raw = json.loads(
        (Path(__file__).parent / "fixtures" / "companyfacts" / "KO_0000021344_debt.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        r
        for r in normalize_annual_facts_from_raw(
            raw, cik="0000021344", years_back=5, filed_as_of="2026-09-28"
        )
        if r["line_item"] == "total_debt"
    ]
    fy2025 = next(r for r in rows if r["fiscal_year"] == 2025)
    assert fy2025["value"] == 45_492.0
    assert [(c["tag"], c["value"]) for c in fy2025["components"]] == [
        ("LongTermDebtAndCapitalLeaseObligations", 42_119.0),
        ("LongTermDebtAndCapitalLeaseObligationsCurrent", 1_822.0),
        ("CommercialPaper", 1_495.0),
        ("OtherShortTermBorrowings", 56.0),
    ]
    assert fy2025["filed_date"] == "2026-02-20"
    fact = extract_total_debt_asof(raw, "2026-09-28")
    assert fact["value"] == 45_492_000_000.0
    assert fact["fact_end_date"] == "2025-12-31"


def test_long_term_measures_that_do_not_reconcile_establish_nothing():
    # Vivid Seats FY2022: debt-and-lease lines of 14.9 noncurrent + 14.9 current against
    # long-term debt of 264.9. A lease-inclusive total cannot sit below the lease-free one.
    raw = _raw(
        {
            "LongTermDebtAndCapitalLeaseObligations": 14_911_000,
            "LongTermDebtAndCapitalLeaseObligationsCurrent": 14_911_000,
            "LongTermDebt": 264_898_000,
        }
    )
    assert _annual_debt(raw) == []
    assert extract_total_debt_asof(raw, "2026-09-09") is None


def test_finance_leases_are_the_only_gap_allowed_between_the_two_families():
    # Alaska Air: debt and leases 5,783 + 452 = 6,235; debt alone 5,724 + 324 = 6,048.
    raw = _raw(
        {
            "LongTermDebtAndCapitalLeaseObligations": 5_783_000_000,
            "LongTermDebtAndCapitalLeaseObligationsCurrent": 452_000_000,
            "LongTermDebt": 6_048_000_000,
            "LongTermDebtNoncurrent": 5_724_000_000,
            "LongTermDebtCurrent": 324_000_000,
            "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": 780_000_000,
        }
    )
    assert _annual_debt(raw) == [6_235.0]
    assert extract_total_debt_asof(raw, "2026-09-09")["value"] == 6_235_000_000.0


def test_exxonmobil_noncurrent_debt_and_leases_is_completed_with_its_current_debt_line():
    """ExxonMobil Holdings' June 2026 10-Q: DebtCurrent 10,139 and the noncurrent-only
    LongTermDebtAndCapitalLeaseObligations 32,229. Both paths returned 32,229, the
    noncurrent side alone; the complete total is 42,368 (nothing short-term added on top of
    DebtCurrent, which already holds every current borrowing)."""
    raw = _raw(
        {"DebtCurrent": 10_139_000_000, "LongTermDebtAndCapitalLeaseObligations": 32_229_000_000},
        end="2026-06-30",
        filed="2026-08-04",
    )
    fact = extract_total_debt_asof(raw, "2026-09-01")
    assert (fact["value"], fact["tag"]) == (
        42_368_000_000.0,
        "LongTermDebtAndCapitalLeaseObligations_plus_DebtCurrent",
    )
    assert _annual_debt(raw) == [42_368.0]
    raw = _raw({"DebtCurrent": 100_000_000, "LongTermDebtNoncurrent": 900_000_000})
    assert extract_total_debt_asof(raw, "2026-09-09")["value"] == 1_000_000_000.0
    assert _annual_debt(raw) == [1_000.0]


def test_the_floor_counts_the_current_debt_line_and_the_noncurrent_debt_and_lease_line():
    assert (
        debt_completeness_floor(
            {"LongTermDebtAndCapitalLeaseObligations": 32_229.0, "DebtCurrent": 10_139.0}
        )
        == 42_368.0
    )
    # The lease-free noncurrent line wins when both are tagged.
    assert (
        debt_completeness_floor(
            {
                "LongTermDebtNoncurrent": 900.0,
                "LongTermDebtAndCapitalLeaseObligations": 950.0,
                "DebtCurrent": 100.0,
            }
        )
        == 1_000.0
    )


def test_a_noncurrent_only_candidate_is_unknown_when_nothing_completes_it():
    # Noncurrent debt and leases 900 beside a current portion of 100 that no shape pairs it
    # with (the lease-inclusive current line is missing): 900 is not the debt.
    raw = _raw(
        {"LongTermDebtAndCapitalLeaseObligations": 900_000_000, "LongTermDebtCurrent": 100_000_000}
    )
    assert extract_total_debt_asof(raw, "2026-09-09") is None
    assert _annual_debt(raw) == []


def test_the_current_debt_line_alone_is_not_total_debt():
    """A 10-K with DebtCurrent 100 and 900 of senior notes under SeniorLongTermNotes (a tag
    the chain does not read): the chain answered 100 and marked the period covered, so the
    gap tier never ran. DebtCurrent alone is now refused by the chain and leaves the period
    open; the gap tier sums the purely current line with the purely noncurrent notes.

    Policy call (2026-09-29): with nothing else reported, the gap tier's own rule applies --
    a purely current family stands alone there (short-term borrowings, commercial paper),
    so DebtCurrent does too, as before. A current line beside a family that has current
    AND noncurrent parts could overlap it, and stays UNKNOWN."""
    raw = _raw({"DebtCurrent": 100_000_000, "SeniorLongTermNotes": 900_000_000})
    assert _annual_debt(raw) == [1_000.0]
    assert extract_total_debt_asof(raw, "2026-09-09") is None
    assert _annual_debt(_raw({"DebtCurrent": 100_000_000})) == [100.0]
    overlapping = _raw(
        {
            "DebtCurrent": 100_000_000,
            "SeniorNotesCurrent": 50_000_000,
            "SeniorLongTermNotes": 900_000_000,
        }
    )
    assert _annual_debt(overlapping) == []


def test_short_term_borrowings_inside_the_current_portion_are_not_counted_twice():
    """Noncurrent 900, LongTermDebtCurrent 150 and ShortTermBorrowings 150. A filer that
    tags its whole current debt line as LongTermDebtCurrent owes 1,050; one whose current
    portion excludes the borrowings owes 1,200. Adding them gave 1,200 either way.

    Policy (2026-09-29, conservative): when the current portion is at least the short-term
    borrowings (so it could hold them), the balance sheet's own combined lines decide --
    DebtCurrent equal to the sum or LongTermDebt reconciling to noncurrent plus the current
    portion means disjoint; DebtCurrent equal to the current portion means overlap. With
    nothing to decide it the total is UNKNOWN, never the larger or smaller guess."""
    lines = {"LongTermDebtNoncurrent": 900.0, "LongTermDebtCurrent": 150.0, "ShortTermBorrowings": 150.0}
    assert resolve_complete_total_debt(900.0, tags=["LongTermDebtNoncurrent"], lines=lines) is None
    assert (
        resolve_complete_total_debt(
            1_050.0, tags=["LongTermDebtCurrent", "LongTermDebtNoncurrent"], lines=lines
        )
        is None
    )
    assert assemble_total_debt({**lines, "DebtCurrent": 150.0}, prefer="LongTermDebtNoncurrent")[
        0
    ] == 1_050.0
    assert assemble_total_debt({**lines, "LongTermDebt": 1_050.0})[0] == 1_200.0
    assert assemble_total_debt({**lines, "LongTermDebt": 900.0}, prefer="LongTermDebtNoncurrent") == (
        1_050.0,
        ["LongTermDebtNoncurrent", "LongTermDebtCurrent"],
    )
    # A current portion below the borrowings cannot hold them: summed as before.
    assert assemble_total_debt({**lines, "LongTermDebtCurrent": 100.0})[0] == 1_150.0
    raw = _raw(
        {
            "LongTermDebtNoncurrent": 900_000_000,
            "LongTermDebtCurrent": 150_000_000,
            "ShortTermBorrowings": 150_000_000,
        }
    )
    assert extract_total_debt_asof(raw, "2026-09-09") is None
    assert _annual_debt(raw) == []


def test_both_total_debt_paths_read_one_fallback_list():
    """LeMaitre as of 2026-09-01: its only debt is a convertible note tagged
    ConvertibleDebtNoncurrent. The normalizer read that tag and the as-of path did not, so
    net debt was UNKNOWN there."""
    from app.ingest.companyfacts import TAG_MAP
    from app.market.company_facts_extract import TOTAL_DEBT_FALLBACK_TAG_PRIORITY

    assert TAG_MAP["total_debt"] == [
        *(tag for _taxonomy, tag in TOTAL_DEBT_FALLBACK_TAG_PRIORITY),
        "DebtCurrent",
    ]
    raw = _raw({"ConvertibleDebtNoncurrent": 169_091_000}, end="2026-06-30", filed="2026-08-01")
    fact = extract_total_debt_asof(raw, "2026-09-01")
    assert (fact["value"], fact["tag"]) == (169_091_000.0, "ConvertibleDebtNoncurrent")
