"""A dated owner summary cannot override newer operating cash-flow evidence."""

import pytest

from app.valuation import intrinsic_discipline as intrinsic


def _owner_payload(year_values, *, as_of_date="2022-03-01"):
    return {
        "as_of_date": as_of_date,
        "series": [{"year": year, "owner_earnings": value} for year, value in year_values],
        "summary": {
            "owner_earnings_normalized_3y": 40.0,
            "owner_earnings_normalized_method": "MEDIAN_3Y",
            "owner_earnings_points": 3,
        },
        "derived_from": ["owner-summary"],
    }


def _current_rows(value=-10.0):
    return [{"year": year, "cfo": value, "capex": 0.0, "fcf": value}
            for year in [2022, 2023, 2024]]


@pytest.mark.parametrize("as_of_date", ["2022-03-01", "2025-03-01"])
def test_intrinsic_old_positive_owner_summary_cannot_override_later_losses(as_of_date):
    # A newly stamped payload does not refresh the fiscal years behind median40.
    stale_owner = _owner_payload([(2019, 30.0), (2020, 40.0), (2021, 50.0)],
                                 as_of_date=as_of_date)
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", fundamentals={"rows": _current_rows()},
        owner_payload=stale_owner, price_value=10.0, shares_value=10.0, net_debt_value=0.0,
    )
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_status"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["NEGATIVE_NORMALIZED_EARNINGS"]
    assert result["intrinsic_base"] == "UNKNOWN"


def test_scout_fact_path_cannot_reintroduce_the_same_stale_owner_series(monkeypatch):
    # The fact path builds its owner series from the stale payload itself. Merely
    # rejecting its summary must not let that old series become the fallback.
    companyfacts = {"facts": {"us-gaap": {
        tag: {"units": {"USD": [
            {"start": f"{year}-01-01", "end": f"{year}-12-31",
             "filed": f"{year + 1}-02-01", "form": "10-K", "fp": "FY", "val": value}
            for year in [2022, 2023, 2024]
        ]}}
        for tag, value in [("NetCashProvidedByUsedInOperatingActivities", -10.0),
                           ("PaymentsToAcquirePropertyPlantAndEquipment", 0.0)]
    }}}
    monkeypatch.setattr(intrinsic, "_load_companyfacts_payload", lambda _: companyfacts)
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", facts_row={"status": "OK"},
        owner_payload=_owner_payload([(2019, 30.0), (2020, 40.0), (2021, 50.0)]),
        price_value=10.0, shares_value=10.0, net_debt_value=0.0,
    )
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["NEGATIVE_NORMALIZED_EARNINGS"]


def test_same_period_owner_summary_retains_its_existing_selection_contract():
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", fundamentals={"rows": _current_rows(100.0)},
        owner_payload=_owner_payload([(2022, 30.0), (2023, 40.0), (2024, 50.0)],
                                     as_of_date="2025-03-01"),
        price_value=10.0, shares_value=10.0, net_debt_value=0.0,
    )
    assert result["normalized_earnings_power_value"] == 40.0
    assert result["normalized_earnings_power_method_used"] == "OWNER_EARNINGS_SELECTED"
    assert "owner-summary" in result["normalized_earnings_power_derived_from"]


def test_old_dated_summary_without_a_series_cannot_override_current_losses():
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", fundamentals={"rows": _current_rows()},
        owner_payload=_owner_payload([], as_of_date="2022-03-01"),
    )
    assert result["normalized_earnings_power_value"] == "UNKNOWN"


def test_current_unknown_owner_year_does_not_refresh_an_old_numeric_summary():
    owner = _owner_payload([(2019, 30.0), (2020, 40.0), (2021, 50.0),
                            (2022, "UNKNOWN"), (2023, "UNKNOWN"), (2024, "UNKNOWN")],
                           as_of_date="2025-03-01")
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", fundamentals={"rows": _current_rows()}, owner_payload=owner,
    )
    assert result["normalized_earnings_power_value"] == "UNKNOWN"


@pytest.mark.parametrize("recent_capex_known", [False, True])
def test_raw_stale_owner_and_missing_recent_capex_cannot_readmit_old_profit(
    monkeypatch, recent_capex_known
):
    # Combines the independent raw-FCF regression with the old owner-summary source.
    cfo = [(2019, 30.0), (2020, 40.0), (2021, 50.0),
           (2022, -10.0), (2023, -10.0), (2024, -10.0)]
    capex_years = range(2019, 2025) if recent_capex_known else range(2019, 2022)
    companyfacts = {"facts": {"us-gaap": {
        tag: {"units": {"USD": [
            {"start": f"{year}-01-01", "end": f"{year}-12-31",
             "filed": f"{year + 1}-02-01", "form": "10-K", "fp": "FY", "val": value}
            for year, value in rows
        ]}}
        for tag, rows in [("NetCashProvidedByUsedInOperatingActivities", cfo),
                          ("PaymentsToAcquirePropertyPlantAndEquipment",
                           [(year, 0.0) for year in capex_years])]
    }}}
    monkeypatch.setattr(intrinsic, "_load_companyfacts_payload", lambda _: companyfacts)
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", facts_row={},
        owner_payload=_owner_payload([(2019, 30.0), (2020, 40.0), (2021, 50.0)]),
        price_value=10.0, shares_value=10.0, net_debt_value=0.0,
    )
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_status"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["NEGATIVE_NORMALIZED_EARNINGS"]
    assert result["intrinsic_base"] == "UNKNOWN"
