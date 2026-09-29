"""Capital spending reduces intrinsic cash earnings under either source sign."""

import pytest

from app.valuation import intrinsic_discipline as intrinsic


def _fundamentals(capex):
    return {"rows": [
        {"year": year, "cfo": 100.0, "capex": capex}
        for year in (2022, 2023, 2024)
    ]}


def _fundamentals_result(fundamentals):
    return intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", fundamentals=fundamentals, owner_payload={},
        owner_quality_payload={"owner_earnings_stability_score": 4.0},
        price_value=10.0, shares_value=10.0, net_debt_value=0.0,
    )


@pytest.mark.parametrize("capex", [20.0, -20.0])
def test_intrinsic_owner_earnings_subtracts_maintenance_spending_magnitude(capex):
    # CFO100 - 60% of spending20 = 88; 10x earnings / shares10 = base88.
    fundamentals = _fundamentals(capex)
    rows = intrinsic._build_series_from_fundamentals(fundamentals)["owner_earnings"]
    assert [row["value"] for row in rows] == [88.0, 88.0, 88.0]
    result = _fundamentals_result(fundamentals)
    assert result["normalized_earnings_power_value"] == 88.0
    assert result["normalized_earnings_power_method_used"] == "OWNER_EARNINGS_SELECTED"
    assert result["intrinsic_base"] == 88.0


@pytest.mark.parametrize("capex", [200.0, -200.0])
def test_intrinsic_owner_loss_does_not_become_positive_from_capex_sign(capex):
    # CFO100 - 60% of spending200 = -20. The existing CFO proxy is separate.
    fundamentals = _fundamentals(capex)
    rows = intrinsic._build_series_from_fundamentals(fundamentals)["owner_earnings"]
    assert [row["value"] for row in rows] == [-20.0, -20.0, -20.0]
    result = _fundamentals_result(fundamentals)
    assert result["normalized_earnings_power_value"] == 100.0
    assert result["normalized_earnings_power_method_used"] == "CFO_PROXY_SELECTED"
    assert result["normalized_earnings_power_status"] == "LOW_CONFIDENCE"


def _fact_series_and_result(monkeypatch, *, cfo, capex):
    companyfacts = {"facts": {"us-gaap": {
        tag: {"units": {"USD": [
            {"start": f"{year}-01-01", "end": f"{year}-12-31",
             "filed": f"{year + 1}-02-01", "form": "10-K", "fp": "FY", "val": value}
            for year in (2022, 2023, 2024)
        ]}}
        for tag, value in [("NetCashProvidedByUsedInOperatingActivities", cfo),
                           ("PaymentsToAcquirePropertyPlantAndEquipment", capex)]
    }}}
    monkeypatch.setattr(intrinsic, "_load_companyfacts_payload", lambda _: companyfacts)
    series = intrinsic._build_series_from_facts(
        as_of_date="2025-03-01", facts_row={}, owner_payload={},
        net_debt_value=0.0, net_debt_refs=[],
    )
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", facts_row={}, owner_payload={},
        # On the facts path an explicit share count is in millions (facts-row units).
        price_value=10.0, shares_value=0.00001, net_debt_value=0.0,
    )
    return series, result


@pytest.mark.parametrize("capex", [20.0, -20.0])
def test_intrinsic_raw_fcf_subtracts_full_spending_magnitude(monkeypatch, capex):
    # CFO100 - spending20 = FCF80; 10x earnings / shares10 = base80.
    series, result = _fact_series_and_result(monkeypatch, cfo=100.0, capex=capex)
    assert [row["value"] for row in series["fcf"]] == [80.0, 80.0, 80.0]
    assert result["normalized_earnings_power_value"] == 80.0
    assert result["normalized_earnings_power_method_used"] == "FCF_SELECTED"
    assert result["intrinsic_base"] == 80.0


@pytest.mark.parametrize("capex", [200.0, -200.0])
def test_intrinsic_raw_fcf_loss_preserves_existing_cfo_proxy_policy(monkeypatch, capex):
    # CFO100 - spending200 = FCF-100, while the positive CFO proxy still exists.
    series, result = _fact_series_and_result(monkeypatch, cfo=100.0, capex=capex)
    assert [row["value"] for row in series["fcf"]] == [-100.0, -100.0, -100.0]
    assert result["normalized_earnings_power_value"] == 100.0
    assert result["normalized_earnings_power_method_used"] == "CFO_PROXY_SELECTED"
    assert result["normalized_earnings_power_status"] == "LOW_CONFIDENCE"


@pytest.mark.parametrize("capex", [20.0, -20.0])
def test_intrinsic_negative_cfo_and_spending_cannot_create_positive_fcf(monkeypatch, capex):
    # CFO-10 - spending20 = FCF-30. There is no positive source or other anchor.
    series, result = _fact_series_and_result(monkeypatch, cfo=-10.0, capex=capex)
    assert [row["value"] for row in series["fcf"]] == [-30.0, -30.0, -30.0]
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_status"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["NEGATIVE_NORMALIZED_EARNINGS"]
    assert result["intrinsic_base"] == "UNKNOWN"


def test_facts_path_share_count_is_millions_and_net_debt_is_not_subtracted_twice(monkeypatch):
    """The scout hands a facts-row share count (millions) and a net debt in
    millions; the companyfacts cash flows are whole dollars. FCF of $100,000,000
    a year x 10 = $1,000,000,000 of equity value over 10,000,000 shares is $100.00.
    FCF is after interest (a levered stream), so net debt ($500M) is not
    subtracted again. Before: (1e9 - 500) / 10 = 99,999,950 a share."""
    companyfacts = {"facts": {"us-gaap": {
        tag: {"units": {"USD": [
            {"start": f"{year}-01-01", "end": f"{year}-12-31",
             "filed": f"{year + 1}-02-01", "form": "10-K", "fp": "FY", "val": value}
            for year in (2022, 2023, 2024)
        ]}}
        for tag, value in [("NetCashProvidedByUsedInOperatingActivities", 100_000_000.0),
                           ("PaymentsToAcquirePropertyPlantAndEquipment", 0.0)]
    }}}
    monkeypatch.setattr(intrinsic, "_load_companyfacts_payload", lambda _: companyfacts)
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", facts_row={}, owner_payload={},
        price_value=50.0, shares_value=10.0, net_debt_value=500.0,
    )
    assert result["normalized_earnings_power_value"] == 100_000_000.0
    assert result["intrinsic_base"] == 100.0
    assert result["intrinsic_floor"] == 80.0
    assert result["intrinsic_ceiling"] == 120.0
