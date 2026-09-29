"""Regression tests for app/valuation/accounting_quality.py and its callers.

Each test asserts the correct answer for a defect that was found in this module or
in the way its callers feed it.

Context: the module is sound when handed
a net-income series (see the controls in ``test_pre_valuation_gate``)
but two of its three callers never hand it one — the valuation gate passes
nothing (pinned in that file) and the scout passes rows without net income
(pinned here). Only ``app/universe/depth_rollup.py`` feeds it real rows, and
the in-module defects below are reachable from there.
"""

from __future__ import annotations

import json


from app.valuation.accounting_quality import UNKNOWN, compute_accounting_quality

OK_STATUS = {"facts_status": "OK", "shares_status": "OK", "fcf_status": "OK"}


# ── _recent_common_ratios: the "last 3 years" slide back to the last 3 profitable ones


def test_conversion_window_slides_back_to_the_last_profitable_years():
    """``cfo_to_net_income_median_3y`` must describe the last three years.

    Mechanism: the positive-denominator filter runs before ``[-window:]``, so
    the window is "the last three years with positive net income", however
    old. A company profitable 2019-2021 and loss-making 2022-2024 is scored
    as of 2025 on 2019-2021.

    Observed: cfo/NI median 1.0, fcf/NI 0.9, HIGH_ACCOUNTING_QUALITY with
    HIGH_ACCOUNTING_QUALITY_SUPPORT, derived_from citing rows[2019..2021].
    Correct: no positive-NI year in 2022-2024 -> ratios UNKNOWN -> the class
    is ACCOUNTING_QUALITY_UNKNOWN (nothing to call "HIGH" on).
    """
    rows = [
        *[{"year": y, "net_income": 100.0, "cfo": 100.0, "fcf": 90.0} for y in (2019, 2020, 2021)],
        *[
            {"year": y, "net_income": -500.0, "cfo": -50.0, "fcf": -60.0}
            for y in (2022, 2023, 2024)
        ],
    ]
    out = compute_accounting_quality("TEST", "2025-01-01", fundamentals={"rows": rows}, **OK_STATUS)
    assert out["cfo_to_net_income_median_3y"] == UNKNOWN
    assert out["accounting_quality_class"] == "ACCOUNTING_QUALITY_UNKNOWN"


# ── line 405: reinvestment class compared against an ACCOUNTING constant ────

DIVERGENT_ROWS = [
    {
        "year": year,
        "revenue": 1000.0 * 1.05**i,
        "accounts_receivable": 100.0 * 1.05**i,
        "net_income": 100.0,
        "cfo": 50.0,
        "fcf": 30.0,
        "sbc_total": 10.0 * 1.05**i,
    }
    for i, year in enumerate(range(2020, 2025))
]


def test_module_calls_the_divergent_fixture_low_without_the_allocator_payload():
    """Control: CFO/NI 0.5 and FCF/NI 0.3 is LOW_ACCOUNTING_QUALITY on its own."""
    out = compute_accounting_quality(
        "TEST",
        "2025-06-30",
        fundamentals={"rows": DIVERGENT_ROWS},
        reinvestment_efficiency_payload={
            "reinvestment_efficiency_class": "LOW_REINVESTMENT_EFFICIENCY"
        },
        **OK_STATUS,
    )
    assert out["accounting_quality_class"] == "LOW_ACCOUNTING_QUALITY"


def test_owner_earnings_support_granted_despite_low_reinvestment_efficiency():
    """The disciplined-allocator support point is meant to be withheld when
    reinvestment efficiency is LOW (the very next line adds the matching
    headwind for that case).

    Observed: OWNER_EARNINGS_SUPPORT_REPORTED_RESULTS *and*
    REPORTED_EARNINGS_NOT_OWNER_RELEVANT are both emitted; the phantom +1
    lifts support strength from 2 to 3 and the class from LOW to
    MODERATE_ACCOUNTING_QUALITY. Correct: LOW_ACCOUNTING_QUALITY, no support
    signal from the allocator.
    """
    out = compute_accounting_quality(
        "TEST",
        "2025-06-30",
        fundamentals={"rows": DIVERGENT_ROWS},
        capital_allocation_discipline_payload={
            "capital_allocation_discipline_class": "OWNER_FRIENDLY_DISCIPLINED"
        },
        reinvestment_efficiency_payload={
            "reinvestment_efficiency_class": "LOW_REINVESTMENT_EFFICIENCY"
        },
        **OK_STATUS,
    )
    assert "OWNER_EARNINGS_SUPPORT_REPORTED_RESULTS" not in out["cash_earnings_support_signals"]
    assert out["accounting_quality_class"] == "LOW_ACCOUNTING_QUALITY"


# ── app/universe/scout.py: the scout never hands the module a net-income series


def _companyfacts(values_by_tag: dict[str, float]) -> dict:
    def rows(value: float) -> list[dict]:
        return [
            {
                "val": value,
                "end": f"{year}-12-31",
                "filed": f"{year + 1}-02-15",
                "form": "10-K",
                "fp": "FY",
                "accn": f"a{year}",
            }
            for year in range(2020, 2025)
        ]

    return {
        "facts": {"us-gaap": {tag: {"units": {"USD": rows(v)}} for tag, v in values_by_tag.items()}}
    }


def test_scout_path_never_supplies_net_income_to_accounting_quality(tmp_path):
    """A filer whose cached facts carry NetIncomeLoss 100 against CFO 20 (and
    capex 15) is the module's textbook LOW; the scout must say so.

    Fixed: the scout used to hand the module rows with keys ['capex', 'cfo',
    'year'] only, so the record read ACCOUNTING_QUALITY_UNKNOWN with
    MISSING_ACCOUNTING_INPUTS; it now adds filed net income (and FCF and owner
    earnings) per year. Correct: LOW_ACCOUNTING_QUALITY. Downstream,
    the universe ranking key on this class is a constant and escalation's
    HIGH_ACCOUNTING_QUALITY_SUPPORT / LOW_ACCOUNTING_QUALITY_HEADWIND lanes
    are unreachable from the scout.
    """
    from app.universe import scout

    cache = tmp_path / "companyfacts.json"
    cache.write_text(
        json.dumps(
            {
                "companyfacts": _companyfacts(
                    {
                        "NetCashProvidedByUsedInOperatingActivities": 20.0,
                        "PaymentsToAcquirePropertyPlantAndEquipment": 15.0,
                        "NetIncomeLoss": 100.0,
                        "Revenues": 1000.0,
                    }
                )
            }
        ),
        encoding="utf-8",
    )
    facts_row = {
        "status": "OK",
        "shares_status": "OK",
        "cfo_status": "OK",
        "capex_status": "OK",
        "fcf_status": "OK",
        "shares_value": 10.0,
        "cfo_value": 20.0,
        "capex_value": 15.0,
        "fcf_value": 5.0,
        "cache_path": str(cache),
        "derived_from": [],
    }
    record, _, _ = scout._build_scout_record(
        ticker="TEST",
        as_of_date="2025-06-30",
        price_row={"status": "OK", "price": 10.0},
        facts_row=facts_row,
        net_debt_resolved={"net_debt_proxy": 50.0, "reason_code": "OK", "derived_from": []},
        thresholds=scout.normalize_scout_thresholds(),
        require_ev_yield=False,
    )
    assert record["accounting_quality_class"] == "LOW_ACCOUNTING_QUALITY"


def test_scout_accounting_quality_stays_unknown_when_no_net_income_was_filed(tmp_path):
    """Control for the fix above: net income is read from the filed facts, never
    invented. Without a NetIncomeLoss/ProfitLoss row the scout still reports UNKNOWN."""
    from app.universe import scout

    cache = tmp_path / "companyfacts.json"
    cache.write_text(
        json.dumps(
            {
                "companyfacts": _companyfacts(
                    {
                        "NetCashProvidedByUsedInOperatingActivities": 20.0,
                        "PaymentsToAcquirePropertyPlantAndEquipment": 15.0,
                        "Revenues": 1000.0,
                    }
                )
            }
        ),
        encoding="utf-8",
    )
    facts_row = {
        "status": "OK",
        "shares_status": "OK",
        "cfo_status": "OK",
        "capex_status": "OK",
        "fcf_status": "OK",
        "shares_value": 10.0,
        "cfo_value": 20.0,
        "capex_value": 15.0,
        "fcf_value": 5.0,
        "cache_path": str(cache),
        "derived_from": [],
    }
    record, _, _ = scout._build_scout_record(
        ticker="TEST",
        as_of_date="2025-06-30",
        price_row={"status": "OK", "price": 10.0},
        facts_row=facts_row,
        net_debt_resolved={"net_debt_proxy": 50.0, "reason_code": "OK", "derived_from": []},
        thresholds=scout.normalize_scout_thresholds(),
        require_ev_yield=False,
    )
    assert record["accounting_quality_class"] == "ACCOUNTING_QUALITY_UNKNOWN"
    assert "MISSING_ACCOUNTING_INPUTS" in record["accounting_quality_reason_codes"]
