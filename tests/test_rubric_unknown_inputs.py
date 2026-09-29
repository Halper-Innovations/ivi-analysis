"""Unknown inputs must never outscore known bad ones in the rubric: completeness counts only applicable data fields, NaN is never the
best score, and a missing filing-coverage row is not better than a known zero.
Hermetic, synthetic literals.
"""

from __future__ import annotations

import math

from app.fundamentals.normalize import UNKNOWN
from app.score.rubric import clamp, score_packet

TICKER = "TEST"



def _packet(fundamentals: dict) -> dict:
    return {
        "ticker": TICKER,
        "as_of_date": "2026-06-30",
        "fundamentals": fundamentals,
        "valuations": {},
        "filings": [],
        "extracted_facts": [],
        "deltas_vs_prior_period": {},
    }


# ── completeness counts only the fields that apply to the issuer ─────


OPERATING_FIELDS = {
    "revenue": 1000.0,
    "gross_margin": 0.4,
    "operating_margin": 0.1,
    "net_income": 80.0,
    "cfo": 120.0,
    "capex": 30.0,
    "fcf": 90.0,
    "fcf_margin": 0.09,
    "net_debt": 450.0,
    "total_assets": 2000.0,
    "debt_to_assets": 0.25,
}
BANK_ONLY_FIELDS = (
    "deposits",
    "loans",
    "investment_securities",
    "assets_under_management",
    "allowance_for_credit_losses",
    "provision_for_credit_losses",
    "net_charge_offs",
    "nonaccrual_loans",
    "deposits_to_assets",
    "loans_to_deposits",
    "allowance_to_loans",
    "provision_to_loans",
    "net_charge_offs_to_loans",
)


def test_a_fully_known_operating_company_scores_full_completeness():
    fundamentals = {
        **OPERATING_FIELDS,
        **{key: UNKNOWN for key in BANK_ONLY_FIELDS},
        "issuer_classification": "operating",
        "fcf_applicability": "standard",
    }
    assert score_packet(_packet(fundamentals))[0]["data_completeness"] == 20.0


def test_a_financial_issuer_is_still_measured_against_the_bank_fields():
    """11 operating fields known + 13 bank fields UNKNOWN = 11/24 of 20 = 9.17."""
    fundamentals = {
        **OPERATING_FIELDS,
        **{key: UNKNOWN for key in BANK_ONLY_FIELDS},
        "issuer_classification": "financial",
        "fcf_applicability": "sector_limited",
    }
    assert score_packet(_packet(fundamentals))[0]["data_completeness"] == 9.17


def test_an_unclassified_issuer_keeps_the_bank_fields_in_its_denominator():
    fundamentals = {**OPERATING_FIELDS, **{key: UNKNOWN for key in BANK_ONLY_FIELDS}}
    assert score_packet(_packet(fundamentals))[0]["data_completeness"] == 9.17


def test_a_none_value_is_not_a_known_metric():
    fundamentals = {"revenue": 1000.0, "cfo": None, "capex": UNKNOWN, "net_income": UNKNOWN}
    assert score_packet(_packet(fundamentals))[0]["data_completeness"] == 5.0


# ── NaN is never the best score ─────────────────────────────────────


def test_clamp_of_nan_is_the_low_bound_not_the_high_bound():
    assert clamp(float("nan"), 0, 20) == 0
    assert clamp(float("nan"), -5, 5) == -5
    assert clamp(7.0, 0, 20) == 7.0
    assert clamp(99.0, 0, 20) == 20
    assert clamp(-99.0, 0, 20) == 0


def test_nan_margins_are_unknown_and_score_the_neutral_ten():
    nan = float("nan")
    subscores, _, _, _ = score_packet(_packet({"operating_margin": nan, "fcf_margin": nan}))
    assert subscores["business_quality_durability"] == 10.0
    infinite = {"operating_margin": math.inf, "fcf_margin": math.inf}
    assert score_packet(_packet(infinite))[0]["business_quality_durability"] == 10.0


def test_nan_liquidity_is_unknown_and_scores_the_neutral_four():
    subscores, _, _, _ = score_packet(_packet({"liquidity_stress_score": float("nan")}))
    assert subscores["balance_sheet_dilution"] == 4.0


def test_nan_coverage_score_is_penalised_like_a_missing_row():
    nan_total = score_packet(_packet({}), filing_coverage={"coverage_score": float("nan")})[1]
    missing_total = score_packet(_packet({}), filing_coverage=None)[1]
    assert nan_total == missing_total


# ── a missing coverage row does not beat a known coverage of zero ────


def test_a_missing_filing_coverage_row_scores_the_same_as_a_known_zero():
    missing_sub, missing_total, _, missing_reasons = score_packet(_packet({}), filing_coverage=None)
    zero_sub, zero_total, _, _ = score_packet(_packet({}), filing_coverage={"coverage_score": 0.0})
    assert missing_sub["filing_coverage_penalty"] == -5.0
    assert zero_sub["filing_coverage_penalty"] == -5.0
    assert missing_total == zero_total
    assert "Filing coverage unknown; scored as zero coverage." in missing_reasons


def test_measured_full_coverage_is_not_penalised_and_beats_unknown():
    full_sub, full_total, _, _ = score_packet(_packet({}), filing_coverage={"coverage_score": 100.0})
    _, missing_total, _, _ = score_packet(_packet({}), filing_coverage=None)
    assert full_sub["filing_coverage_penalty"] == 0.0
    assert full_total == round(missing_total + 5.0, 2)
