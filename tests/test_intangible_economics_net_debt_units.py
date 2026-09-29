"""app/valuation/intangible_economics.py must not divide $millions of net debt by whole dollars.

On the universe-scout path the module is
handed ``net_debt_resolved`` from app/valuation/net_debt.py, which reports $millions
(net_debt.py:35, 52-68), while the cash-flow series it builds from the companyfacts
cache is in whole dollars (owner_earnings.py:35-37: "Companyfacts 'val' is used
unscaled"). ``_balance_sheet_optionality`` divides the two
(intangible_economics.py:795-799). Observed: net debt $3,000M against cash from
operations $500M (whole dollars) → ``net_debt_to_cfo_proxy`` 0.000006 and a
balance-sheet optionality score of 2.0 with no NET_DEBT_PRESSURE — a six-times-levered
issuer credited with a strong balance sheet. This is the same million-fold error that
was closed in balance_sheet_stress.py.

The builder flattens every claim to its bare value at the top level of the payload
(``net_debt_to_cfo_proxy``, ``balance_sheet_optionality_score``,
``balance_sheet_optionality_reason_codes``), which is what the assertions read.
"""

from __future__ import annotations

import math

from app.valuation.intangible_economics import _build_intangible_economics_payload


def _series_map() -> dict:
    return {
        "gross_margin": [],
        "revenue": [{"year": 2025, "value": 2_000_000_000.0, "derived_from": []}],
        "cfo": [{"year": 2025, "value": 500_000_000.0, "derived_from": []}],
        "fcf": [],
        "net_debt": [],
        "cash": [{"year": 2025, "value": 100_000_000.0, "derived_from": []}],
    }


def test_net_debt_in_millions_meets_companyfacts_cash_flow_in_one_unit():
    payload = _build_intangible_economics_payload(
        ticker="TST",
        as_of_date="2026-01-01",
        source_kind="COMPANYFACTS",
        series_map=_series_map(),
        owner_quality_payload={},
        net_debt_resolved={"net_debt_proxy": 3000.0, "derived_from": []},
    )
    assert math.isclose(payload["net_debt_to_cfo_proxy"], 6.0, rel_tol=1e-9)
    assert payload["balance_sheet_optionality_score"] == 0.0
    assert "NET_DEBT_PRESSURE" in payload["balance_sheet_optionality_reason_codes"]


def test_the_fundamentals_path_is_untouched():
    # Fundamentals rows are already $millions; no rescaling applies there.
    series = _series_map()
    series["cfo"] = [{"year": 2025, "value": 500.0, "derived_from": []}]
    series["revenue"] = [{"year": 2025, "value": 2000.0, "derived_from": []}]
    series["cash"] = [{"year": 2025, "value": 100.0, "derived_from": []}]
    payload = _build_intangible_economics_payload(
        ticker="TST",
        as_of_date="2026-01-01",
        source_kind="FUNDAMENTALS",
        series_map=series,
        owner_quality_payload={},
        net_debt_resolved={"net_debt_proxy": 3000.0, "derived_from": []},
    )
    assert math.isclose(payload["net_debt_to_cfo_proxy"], 6.0, rel_tol=1e-9)
    assert payload["balance_sheet_optionality_score"] == 0.0
    assert "NET_DEBT_PRESSURE" in payload["balance_sheet_optionality_reason_codes"]
