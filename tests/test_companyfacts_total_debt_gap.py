"""total_debt for filers that tag borrowings only under instrument-level concepts.

About a third of cached filers with a current balance sheet resolved no
total_debt because their revolver, notes, loans or short-term borrowings sit
under concepts the main chain never read. The gap tier reads them, but only
for a period the main chain leaves empty, only from one filing, never adding a
family's total to its own parts, and only summing two families where they
cannot overlap. Every mapped concept is pinned below with exact literals.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from app.ingest.companyfacts import (
    _TOTAL_DEBT_GAP_DISQUALIFIERS,
    _TOTAL_DEBT_GAP_FAMILIES,
    normalize_annual_facts_from_raw,
    normalize_quarterly_facts_from_raw,
)

_END = "2024-12-31"
_FILED = "2025-02-20"
_ACCN = "0000000001-25-000001"


def _fact(
    val: float,
    *,
    end: str = _END,
    filed: str = _FILED,
    accn: str = _ACCN,
    form: str = "10-K",
    **extra: Any,
) -> dict[str, Any]:
    return {"end": end, "val": val, "form": form, "filed": filed, "accn": accn, **extra}


def _payload(tags: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"facts": {"us-gaap": {tag: {"units": {"USD": facts}} for tag, facts in tags.items()}}}


def _debt(raw: dict[str, Any], *, as_of: str = "2026-01-01") -> dict[int, tuple[float, str, str]]:
    rows = normalize_annual_facts_from_raw(raw, cik="0000000001", years_back=10, filed_as_of=as_of)
    return {
        int(r["fiscal_year"]): (r["value"], r["filed_date"], r["accession"])
        for r in rows
        if r["line_item"] == "total_debt"
    }


def _reversed_arrays(raw: dict[str, Any]) -> dict[str, Any]:
    flipped = copy.deepcopy(raw)
    for tags in flipped["facts"].values():
        for node in tags.values():
            for unit, facts in node["units"].items():
                node["units"][unit] = list(reversed(facts))
    return flipped


# ── every mapped concept, alone ───────────────────────────────────────────────

_EVERY_MAPPED_TAG = sorted(
    tag for members in _TOTAL_DEBT_GAP_FAMILIES.values() for tag in members if tag is not None
)


def test_the_mapped_concepts_are_exactly_these():
    assert _EVERY_MAPPED_TAG == [
        "CommercialPaper",
        "ConvertibleDebtCurrent",
        "ConvertibleNotesPayableCurrent",
        # Added 2026-09-29: reached only when the chain's answer was DebtCurrent alone.
        "DebtCurrent",
        "JuniorSubordinatedLongTermNotes",
        "JuniorSubordinatedNotes",
        "LinesOfCreditCurrent",
        "LoansPayable",
        "LoansPayableCurrent",
        "LoansPayableToBank",
        "LoansPayableToBankCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "LongTermDebtCurrent",
        "LongTermLineOfCredit",
        "LongTermLoansPayable",
        "LongTermNotesAndLoans",
        "LongTermNotesPayable",
        "NotesAndLoansPayable",
        "NotesAndLoansPayableCurrent",
        "NotesPayableCurrent",
        "NotesPayableToBankCurrent",
        "NotesPayableToBankNoncurrent",
        "OtherLoansPayable",
        "OtherLoansPayableCurrent",
        "OtherLongTermDebt",
        "OtherLongTermDebtCurrent",
        "OtherLongTermDebtNoncurrent",
        "OtherLongTermNotesPayable",
        "OtherNotesPayable",
        "OtherNotesPayableCurrent",
        "OtherShortTermBorrowings",
        "SecuredDebt",
        "SecuredDebtCurrent",
        "SecuredLongTermDebt",
        "SeniorLongTermNotes",
        "SeniorNotesCurrent",
        "ShortTermBankLoansAndNotesPayable",
        "ShortTermBorrowings",
        "ShortTermNonBankLoansAndNotesPayable",
        "SubordinatedDebt",
        "SubordinatedLongTermDebt",
        "UnsecuredDebt",
        "UnsecuredDebtCurrent",
        "UnsecuredLongTermDebt",
    ]


@pytest.mark.parametrize("tag", _EVERY_MAPPED_TAG)
def test_each_mapped_concept_alone_resolves_total_debt(tag: str):
    raw = _payload({tag: [_fact(123_450_000.0)]})
    assert _debt(raw) == {2024: (123.45, _FILED, _ACCN)}


# ── one family: a total is never added to its own parts ──────────────────────


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("loans_payable", 100.0),
        ("notes_and_loans_payable", 100.0),
        ("other_notes_payable", 100.0),
        ("other_long_term_debt", 100.0),
        ("secured_debt", 100.0),
        ("unsecured_debt", 100.0),
    ],
)
def test_family_total_wins_over_its_own_components(family: str, expected: float):
    total, current, noncurrent = _TOTAL_DEBT_GAP_FAMILIES[family]
    tags = {total: [_fact(100e6)], current: [_fact(30e6)], noncurrent: [_fact(70e6)]}
    assert _debt(_payload(tags)) == {2024: (expected, _FILED, _ACCN)}


@pytest.mark.parametrize(
    ("current", "noncurrent"),
    [
        ("LinesOfCreditCurrent", "LongTermLineOfCredit"),
        ("NotesPayableCurrent", "LongTermNotesPayable"),
        ("NotesPayableToBankCurrent", "NotesPayableToBankNoncurrent"),
        ("LoansPayableCurrent", "LongTermLoansPayable"),
        ("NotesAndLoansPayableCurrent", "LongTermNotesAndLoans"),
        ("OtherNotesPayableCurrent", "OtherLongTermNotesPayable"),
        ("OtherLongTermDebtCurrent", "OtherLongTermDebtNoncurrent"),
        ("SeniorNotesCurrent", "SeniorLongTermNotes"),
        ("SecuredDebtCurrent", "SecuredLongTermDebt"),
        ("UnsecuredDebtCurrent", "UnsecuredLongTermDebt"),
    ],
)
def test_family_current_plus_noncurrent_from_one_filing(current: str, noncurrent: str):
    raw = _payload({current: [_fact(30e6)], noncurrent: [_fact(70e6)]})
    assert _debt(raw) == {2024: (100.0, _FILED, _ACCN)}


def test_current_and_noncurrent_come_from_the_latest_filing_only():
    """The next year's 10-K re-reports only the noncurrent revolver balance for
    the prior balance-sheet date; the stale current part from the original
    filing must not be added to it."""
    raw = _payload(
        {
            "LinesOfCreditCurrent": [_fact(10e6)],
            "LongTermLineOfCredit": [
                _fact(50e6),
                _fact(55e6, filed="2026-02-20", accn="0000000001-26-000001"),
            ],
        }
    )
    assert _debt(raw, as_of="2026-03-01") == {2024: (55.0, "2026-02-20", "0000000001-26-000001")}
    assert _debt(raw, as_of="2026-01-01") == {2024: (60.0, _FILED, _ACCN)}


# ── two families: summed only where they cannot overlap ──────────────────────


def test_purely_current_family_plus_purely_noncurrent_family_are_summed():
    raw = _payload(
        {"ConvertibleDebtCurrent": [_fact(200e6)], "LongTermLineOfCredit": [_fact(385e6)]}
    )
    assert _debt(raw) == {2024: (585.0, _FILED, _ACCN)}


def test_secured_plus_unsecured_are_summed():
    raw = _payload({"SecuredDebt": [_fact(784e6)], "UnsecuredDebt": [_fact(6016e6)]})
    assert _debt(raw) == {2024: (6800.0, _FILED, _ACCN)}


@pytest.mark.parametrize(
    "tags",
    [
        # the same 4.1 loan tagged under an instrument family and a collateral family
        {"LoansPayable": 4.1e6, "UnsecuredLongTermDebt": 4.1e6},
        # two current-side families can overlap (short-term borrowings include bank loans)
        {"LongTermDebtCurrent": 1.6e6, "ShortTermBankLoansAndNotesPayable": 23.1e6},
        # two noncurrent-side families can overlap
        {"LongTermLineOfCredit": 164e6, "OtherLongTermDebtNoncurrent": 21.4e6},
        # a family with both sides next to another family is ambiguous
        {"LongTermLineOfCredit": 345e6, "NotesPayableCurrent": 4e6, "LongTermNotesPayable": 3.3e6},
        # three families
        {"LongTermDebtCurrent": 4e6, "LongTermLineOfCredit": 114e6, "SubordinatedDebt": 100e6},
    ],
)
def test_overlapping_families_stay_missing(tags: dict[str, float]):
    raw = _payload({tag: [_fact(val)] for tag, val in tags.items()})
    assert _debt(raw) == {}


# ── what the gap tier never does ─────────────────────────────────────────────


def test_a_period_the_main_chain_resolves_is_never_changed():
    # Migrated: the chain's long-term debt (500) is now completed with the same balance
    # sheet's short-term borrowings (25) -- long-term debt alone was a narrow amount. The
    # gap tier still adds nothing: its line-of-credit family (40) stays out.
    raw = _payload(
        {
            "LongTermDebt": [_fact(500e6)],
            "LinesOfCreditCurrent": [_fact(40e6)],
            "ShortTermBorrowings": [_fact(25e6)],
        }
    )
    assert _debt(raw) == {2024: (525.0, _FILED, _ACCN)}


def test_the_gap_tier_fills_only_the_uncovered_year():
    raw = _payload(
        {
            "LongTermDebt": [_fact(500e6, end="2023-12-31", filed="2024-02-20", accn="a23")],
            "LinesOfCreditCurrent": [
                _fact(40e6, end="2023-12-31", filed="2024-02-20", accn="a23"),
                _fact(45e6),
            ],
        }
    )
    assert _debt(raw) == {2023: (500.0, "2024-02-20", "a23"), 2024: (45.0, _FILED, _ACCN)}


@pytest.mark.parametrize("tag", _TOTAL_DEBT_GAP_DISQUALIFIERS)
def test_bank_and_insurer_funding_disqualifies_the_filing(tag: str):
    raw = _payload({"ShortTermBorrowings": [_fact(64_776e6)], tag: [_fact(1_000e6)]})
    assert _debt(raw) == {}


def test_a_zero_funding_balance_does_not_disqualify():
    raw = _payload({"LinesOfCreditCurrent": [_fact(23.6e6)], "OtherBorrowings": [_fact(0.0)]})
    assert _debt(raw) == {2024: (23.6, _FILED, _ACCN)}


def test_finance_leases_stay_out_of_total_debt():
    only_leases = _payload(
        {
            "FinanceLeaseLiability": [_fact(890e6)],
            "FinanceLeaseLiabilityCurrent": [_fact(136e6)],
            "FinanceLeaseLiabilityNoncurrent": [_fact(754e6)],
        }
    )
    assert _debt(only_leases) == {}
    with_revolver = _payload(
        {"FinanceLeaseLiability": [_fact(0.3e6)], "LongTermLineOfCredit": [_fact(4e6)]}
    )
    assert _debt(with_revolver) == {2024: (4.0, _FILED, _ACCN)}


def test_explicit_zero_balances_emit_no_row():
    """Turning a filed zero into total_debt = 0 is the evidenced-zero policy's
    job, not this tier's."""
    raw = _payload({"ShortTermBorrowings": [_fact(0.0)], "LinesOfCreditCurrent": [_fact(0.0)]})
    assert _debt(raw) == {}


# ── point in time and array order ────────────────────────────────────────────


def test_gap_filing_after_as_of_is_invisible_and_filed_on_as_of_is_visible():
    raw = _payload(
        {
            "NotesPayableCurrent": [
                _fact(10e6),
                _fact(12e6, form="10-K/A", filed="2025-06-30", accn="amend"),
            ]
        }
    )
    for payload in (raw, _reversed_arrays(raw)):
        assert _debt(payload, as_of="2025-02-19") == {}
        assert _debt(payload, as_of="2025-02-20") == {2024: (10.0, _FILED, _ACCN)}
        assert _debt(payload, as_of="2025-06-29") == {2024: (10.0, _FILED, _ACCN)}
        assert _debt(payload, as_of="2025-06-30") == {2024: (12.0, "2025-06-30", "amend")}


def test_gap_tier_is_array_order_independent():
    raw = _payload(
        {
            "LinesOfCreditCurrent": [
                _fact(10e6),
                _fact(11e6, filed="2026-02-20", accn="0000000001-26-000001"),
            ],
            "LongTermLineOfCredit": [
                _fact(50e6),
                _fact(52e6, filed="2026-02-20", accn="0000000001-26-000001"),
            ],
        }
    )
    expected = {2024: (63.0, "2026-02-20", "0000000001-26-000001")}
    assert (
        _debt(raw, as_of="2026-03-01")
        == _debt(_reversed_arrays(raw), as_of="2026-03-01")
        == expected
    )


def test_quarterly_gap_fill():
    raw = _payload(
        {
            "LinesOfCreditCurrent": [
                _fact(8e6, end="2025-06-30", form="10-Q", filed="2025-08-01", fy=2025, fp="Q2")
            ],
            "LongTermLineOfCredit": [
                _fact(42e6, end="2025-06-30", form="10-Q", filed="2025-08-01", fy=2025, fp="Q2")
            ],
        }
    )
    rows = normalize_quarterly_facts_from_raw(raw, cik="0000000001", years_back=10)
    assert [
        (r["fiscal_year"], r["period_type"], r["period_end"], r["value"], r["accession"])
        for r in rows
        if r["line_item"] == "total_debt"
    ] == [(2025, "Q2", "2025-06-30", 50.0, _ACCN)]
