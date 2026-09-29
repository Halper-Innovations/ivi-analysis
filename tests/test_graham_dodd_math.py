"""Graham-Dodd valuation math: one test per defect found in a methodology audit.

Every assertion is an exact literal computed by hand in the test's own docstring,
so a future change to the module has to restate the arithmetic to pass.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.valuation.graham_dodd import UNKNOWN, compute_graham_dodd_overlay


SHARES = 10_000_000.0


def _flow(rows: list[tuple[str, str, float, str, str, str]]) -> dict:
    return {
        "units": {
            "USD": [
                {"end": end, "filed": filed, "val": val, "start": start, "form": form, "fp": fp}
                for end, filed, val, start, form, fp in rows
            ]
        }
    }


def _point(value: float, *, end: str = "2025-12-31", filed: str = "2026-01-31") -> dict:
    return {"units": {"USD": [{"end": end, "filed": filed, "val": value}]}}


def _write_facts(path: Path, *, us_gaap: dict, shares: float | None = SHARES) -> str:
    dei = {}
    if shares is not None:
        dei = {
            "EntityCommonStockSharesOutstanding": {
                "units": {"shares": [{"end": "2025-12-31", "filed": "2026-01-31", "val": shares}]}
            }
        }
    payload = {"companyfacts": {"facts": {"dei": dei, "us-gaap": us_gaap}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _overlay(
    cache_path: str,
    *,
    price_value: float | str = 20.0,
    price_status: str = "OK",
    discount_rate: float = 0.10,
    shares_value: float = SHARES,
    owner_normalized: float | str = UNKNOWN,
) -> dict:
    return compute_graham_dodd_overlay(
        ticker="AAA",
        as_of_date="2026-02-14",
        price_value=price_value,
        price_status=price_status,
        discount_rate=discount_rate,
        facts_row={
            "ticker": "AAA",
            "shares_status": "OK",
            # Facts rows carry shares in millions (app.valuation.facts).
            "shares_value": shares_value / 1_000_000.0,
            "cache_path": cache_path,
            "derived_from": ["facts.AAA"],
        },
        owner_payload={
            "summary": {
                "owner_earnings_normalized_3y": owner_normalized,
                "owner_earnings_normalized_method": "MEDIAN_3Y",
            },
            "derived_from": ["owner.AAA"],
        },
    )


def test_interim_ytd_row_is_not_taken_as_the_fiscal_year(tmp_path):
    """DEFECT 1 — a June-fiscal-year filer's 6-month 10-Q row was read as FY2024.

    Rows are bucketed by the CALENDAR year of `end`, and the old contest kept the
    latest `end` in each bucket. For a fiscal year ending 06-30, calendar 2024
    holds three rows: FY2024 (end 2024-06-30), Q1 (end 2024-09-30) and the
    six-month Q2 (end 2024-12-31). The Q2 row won.

    FCF = CFO - capex per fiscal year:
        FY2022  90,000,000 - 18,000,000 =  72,000,000
        FY2023  95,000,000 - 19,000,000 =  76,000,000
        FY2024 100,000,000 - 20,000,000 =  80,000,000
        Q2-2025 (6 months) 48,000,000 - 9,600,000 = 38,400,000

    Wrong: median(72,000,000, 76,000,000, 38,400,000) = 72,000,000
           EPV = 72,000,000 / 0.10 = 720,000,000 -> 720,000,000 / 10,000,000 = $72.00/sh
           mos_epv = 72.00 / 20.00 - 1 = 2.60
    Right: median(72,000,000, 76,000,000, 80,000,000) = 76,000,000
           EPV = 76,000,000 / 0.10 = 760,000,000 -> 760,000,000 / 10,000,000 = $76.00/sh
           mos_epv = 76.00 / 20.00 - 1 = 2.80
    """
    cache_path = _write_facts(
        tmp_path / "cf" / "d1.json",
        us_gaap={
            "NetCashProvidedByUsedInOperatingActivities": _flow(
                [
                    ("2022-06-30", "2022-08-15", 90_000_000.0, "2021-07-01", "10-K", "FY"),
                    ("2023-06-30", "2023-08-15", 95_000_000.0, "2022-07-01", "10-K", "FY"),
                    ("2024-06-30", "2024-08-15", 100_000_000.0, "2023-07-01", "10-K", "FY"),
                    ("2024-09-30", "2024-11-08", 25_000_000.0, "2024-07-01", "10-Q", "Q1"),
                    ("2024-12-31", "2025-02-07", 48_000_000.0, "2024-07-01", "10-Q", "Q2"),
                ]
            ),
            "PaymentsToAcquirePropertyPlantAndEquipment": _flow(
                [
                    ("2022-06-30", "2022-08-15", 18_000_000.0, "2021-07-01", "10-K", "FY"),
                    ("2023-06-30", "2023-08-15", 19_000_000.0, "2022-07-01", "10-K", "FY"),
                    ("2024-06-30", "2024-08-15", 20_000_000.0, "2023-07-01", "10-K", "FY"),
                    ("2024-09-30", "2024-11-08", 5_000_000.0, "2024-07-01", "10-Q", "Q1"),
                    ("2024-12-31", "2025-02-07", 9_600_000.0, "2024-07-01", "10-Q", "Q2"),
                ]
            ),
        },
    )

    payload = _overlay(cache_path)

    assert payload["inputs_used"]["normalized_cash_earnings"]["value"] == 76000000.0
    assert payload["inputs_used"]["normalized_cash_earnings"]["reason_code"] == "FCF_MEDIAN_3Y"
    assert payload["epv_value"] == 760000000.0
    assert payload["epv_per_share"] == 76.0
    assert payload["mos_epv"] == 2.8


def test_normalization_window_is_three_consecutive_years_not_three_values(tmp_path):
    """DEFECT 2 — three non-adjacent years were averaged and labelled MEDIAN_3Y.

    Capex is untagged for 2023 and 2024, so the CFO-capex intersection is
    {2021, 2022, 2025} and the old code took "the last three values".

        2021  60,000,000 - 10,000,000 = 50,000,000
        2022  55,000,000 - 10,000,000 = 45,000,000
        2025  15,000,000 -  5,000,000 = 10,000,000

    Wrong: median(50,000,000, 45,000,000, 10,000,000) = 45,000,000, sold as
           MEDIAN_3Y though the sample spans 2021-2025 and two thirds of it is
           three and four years stale.
           EPV = 45,000,000 / 0.10 = 450,000,000 -> $45.00/sh
           mos_epv = 45.00 / 20.00 - 1 = 1.25  (scout PASS at scout_mos_min 0.30)
    Right: the window is 2023-2025; only 2025 lands inside it.
           value = 10,000,000, reported as FCF_LATEST, not FCF_MEDIAN_3Y.
           EPV = 10,000,000 / 0.10 = 100,000,000 -> $10.00/sh
           mos_epv = 10.00 / 20.00 - 1 = -0.50  (scout FAIL)
    """
    cache_path = _write_facts(
        tmp_path / "cf" / "d2.json",
        us_gaap={
            "NetCashProvidedByUsedInOperatingActivities": _flow(
                [
                    ("2021-12-31", "2022-02-15", 60_000_000.0, "2021-01-01", "10-K", "FY"),
                    ("2022-12-31", "2023-02-15", 55_000_000.0, "2022-01-01", "10-K", "FY"),
                    ("2023-12-31", "2024-02-15", 30_000_000.0, "2023-01-01", "10-K", "FY"),
                    ("2024-12-31", "2025-02-15", 20_000_000.0, "2024-01-01", "10-K", "FY"),
                    ("2025-12-31", "2026-02-10", 15_000_000.0, "2025-01-01", "10-K", "FY"),
                ]
            ),
            "PaymentsToAcquirePropertyPlantAndEquipment": _flow(
                [
                    ("2021-12-31", "2022-02-15", 10_000_000.0, "2021-01-01", "10-K", "FY"),
                    ("2022-12-31", "2023-02-15", 10_000_000.0, "2022-01-01", "10-K", "FY"),
                    ("2025-12-31", "2026-02-10", 5_000_000.0, "2025-01-01", "10-K", "FY"),
                ]
            ),
        },
    )

    payload = _overlay(cache_path)

    assert payload["inputs_used"]["normalized_cash_earnings"]["value"] == 10000000.0
    assert payload["inputs_used"]["normalized_cash_earnings"]["reason_code"] == "FCF_LATEST"
    assert payload["epv_value"] == 100000000.0
    assert payload["epv_per_share"] == 10.0
    assert payload["mos_epv"] == -0.5


def test_untagged_preferred_stock_is_labelled_an_assumption_not_a_reading(tmp_path):
    """DEFECT 3 — a missing preferred-stock tag was published as reason_code OK.

    Zero is the right NCAV term when no preferred is outstanding, but the filer
    tagged nothing, `derived_from` is empty, and the payload still claimed OK.
    Wrong: {"value": 0.0, "derived_from": [], "reason_code": "OK"}
    Right: {"value": 0.0, "derived_from": [], "reason_code": "PREFERRED_STOCK_ASSUMED_ZERO"}

    NCAV is unchanged: 1,000,000,000 - 400,000,000 - 0 = 600,000,000 -> $60.00/sh.
    """
    cache_path = _write_facts(
        tmp_path / "cf" / "d3.json",
        us_gaap={
            "AssetsCurrent": _point(1_000_000_000.0),
            "Liabilities": _point(400_000_000.0),
        },
    )

    payload = _overlay(cache_path)

    assert payload["inputs_used"]["preferred_stock"]["value"] == 0.0
    assert payload["inputs_used"]["preferred_stock"]["derived_from"] == []
    assert (
        payload["inputs_used"]["preferred_stock"]["reason_code"] == "PREFERRED_STOCK_ASSUMED_ZERO"
    )
    assert payload["netnet_value"] == 600000000.0
    assert payload["netnet_per_share"] == 60.0

    tagged_path = _write_facts(
        tmp_path / "cf" / "d3_tagged.json",
        us_gaap={
            "AssetsCurrent": _point(1_000_000_000.0),
            "Liabilities": _point(400_000_000.0),
            "PreferredStockValue": _point(100_000_000.0),
        },
    )
    tagged = _overlay(tagged_path)
    assert tagged["inputs_used"]["preferred_stock"]["value"] == 100000000.0
    assert tagged["inputs_used"]["preferred_stock"]["reason_code"] == "OK"
    assert tagged["netnet_value"] == 500000000.0


def test_non_finite_fact_value_is_not_computable_rather_than_status_ok(tmp_path):
    """DEFECT 4 — a NaN in the cache produced epv_status OK with a NaN value.

    json.loads accepts the bare NaN token, and NaN passed every guard in the
    module: `NaN < 0` is False, `NaN > 0` is False. EPV then became NaN / 0.10.
    Wrong: epv_status "OK", epv_value NaN, mos_epv NaN, and a payload that
           json.dumps(..., allow_nan=False) refuses to serialize.
    Right: the fact is not a number, so the earnings stream is missing and the
           overlay says so.
    """
    path = tmp_path / "cf" / "d4.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"companyfacts": {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units":'
        ' {"shares": [{"end": "2025-12-31", "filed": "2026-01-31", "val": 10000000}]}}},'
        ' "us-gaap": {"FreeCashFlow": {"units": {"USD": [{"end": "2025-12-31", "filed":'
        ' "2026-02-10", "val": NaN, "form": "10-K", "fp": "FY"}]}}}}}}',
        encoding="utf-8",
    )

    payload = _overlay(str(path))

    assert payload["epv_status"] == "UNKNOWN"
    assert payload["epv_value"] == "UNKNOWN"
    assert payload["epv_per_share"] == "UNKNOWN"
    assert payload["epv_reason_code"] == "MISSING_EARNINGS_STREAM"
    assert payload["mos_epv"] == "UNKNOWN"
    assert json.dumps(payload, allow_nan=False).startswith("{")


def test_every_mos_return_path_uses_the_upside_ratio_convention(tmp_path):
    """ITEM 1 — mos_epv AND mos_netnet are upside ratios, never textbook MoS.

    See app/valuation/mos_conventions.py. For intrinsic 150 and price 100:
        upside   = 150 / 100 - 1       = 0.50   <- what this module returns
        textbook = (150 - 100) / 150   = 0.3333... <- what it must never return

    EPV path:    150,000,000 / 0.10 = 1,500,000,000 / 10,000,000 sh = $150.00
    NCAV path:   1,600,000,000 - 100,000,000 = 1,500,000,000 / 10,000,000 sh = $150.00
    """
    cache_path = _write_facts(
        tmp_path / "cf" / "conv.json",
        us_gaap={
            "AssetsCurrent": _point(1_600_000_000.0),
            "Liabilities": _point(100_000_000.0),
        },
    )

    payload = _overlay(cache_path, price_value=100.0, owner_normalized=150_000_000.0)

    assert payload["epv_per_share"] == 150.0
    assert payload["netnet_per_share"] == 150.0
    assert payload["mos_epv"] == 0.5
    assert payload["mos_netnet"] == 0.5
    assert payload["mos_epv"] != 0.3333333333333333
    assert payload["mos_netnet"] != 0.3333333333333333


def test_netnet_subtracts_total_liabilities_never_current_liabilities_alone(tmp_path):
    """ITEM 3 — NCAV = current assets - TOTAL liabilities - preferred.

    valuation_writer._ncav shipped the current-liabilities-only variant, so this
    pins the correct one here.

    Only LiabilitiesCurrent 100,000,000 and LiabilitiesNoncurrent 300,000,000 are
    tagged (no `Liabilities` roll-up), so the module must sum them to
    400,000,000:
        right: (1,000,000,000 - 400,000,000) / 10,000,000 = $60.00/sh
        wrong: (1,000,000,000 - 100,000,000) / 10,000,000 = $90.00/sh

    And when only current liabilities exist there is no total to subtract, so the
    answer is refused rather than understated.
    """
    cache_path = _write_facts(
        tmp_path / "cf" / "ncav.json",
        us_gaap={
            "AssetsCurrent": _point(1_000_000_000.0),
            "LiabilitiesCurrent": _point(100_000_000.0),
            "LiabilitiesNoncurrent": _point(300_000_000.0),
        },
    )

    payload = _overlay(cache_path)

    assert payload["inputs_used"]["total_liabilities"]["value"] == 400000000.0
    assert payload["netnet_value"] == 600000000.0
    assert payload["netnet_per_share"] == 60.0
    assert payload["mos_netnet"] == 2.0

    current_only_path = _write_facts(
        tmp_path / "cf" / "ncav_current_only.json",
        us_gaap={
            "AssetsCurrent": _point(1_000_000_000.0),
            "LiabilitiesCurrent": _point(100_000_000.0),
        },
    )
    current_only = _overlay(current_only_path)
    assert current_only["netnet_status"] == "UNKNOWN"
    assert current_only["netnet_value"] == "UNKNOWN"
    assert current_only["netnet_reason_code"] == "MISSING_TOTAL_LIABILITIES"


def test_zero_denominators_are_refused_with_a_reason_not_divided_by(tmp_path):
    """ITEM 6 — every division guards its denominator and names why it stopped.

    Three denominators can be zero from real filings: the discount rate, the
    share count, and the price. None raises; each returns UNKNOWN plus a code.
    """
    cache_path = _write_facts(
        tmp_path / "cf" / "zero.json",
        us_gaap={
            "AssetsCurrent": _point(1_000_000_000.0),
            "Liabilities": _point(400_000_000.0),
        },
    )

    zero_discount = _overlay(cache_path, discount_rate=0.0, owner_normalized=100_000_000.0)
    assert zero_discount["epv_value"] == "UNKNOWN"
    assert zero_discount["epv_reason_code"] == "INVALID_DISCOUNT_RATE"

    no_shares_path = _write_facts(
        tmp_path / "cf" / "zero_shares.json",
        us_gaap={
            "AssetsCurrent": _point(1_000_000_000.0),
            "Liabilities": _point(400_000_000.0),
        },
        shares=None,
    )
    zero_shares = _overlay(no_shares_path, shares_value=0.0, owner_normalized=100_000_000.0)
    assert zero_shares["epv_per_share"] == "UNKNOWN"
    assert zero_shares["epv_reason_code"] == "MISSING_SHARES"
    assert zero_shares["netnet_per_share"] == "UNKNOWN"
    assert zero_shares["netnet_reason_code"] == "MISSING_SHARES"

    # EPV = 100,000,000 / 0.10 = 1,000,000,000 / 10,000,000 sh = $100.00/sh; the
    # per-share value is computable, only the price-relative ratio is not.
    zero_price = _overlay(cache_path, price_value=0.0, owner_normalized=100_000_000.0)
    assert zero_price["epv_per_share"] == 100.0
    assert zero_price["mos_epv"] == "UNKNOWN"
    assert zero_price["mos_epv_reason_code"] == "PRICE_UNKNOWN"
    assert zero_price["mos_netnet"] == "UNKNOWN"
    assert zero_price["mos_netnet_reason_code"] == "PRICE_UNKNOWN"
