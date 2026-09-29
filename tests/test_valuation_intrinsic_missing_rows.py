"""Recent missing fiscal rows must survive the production input adapters."""

import pytest

from app.valuation import intrinsic_discipline as intrinsic

def _cashflow_rows(year_values):
    return [
        {"year": year, "cfo": value, "capex": 0.0, "fcf": value}
        for year, value in year_values
    ]


def _power(rows):
    return intrinsic._normalized_earnings_power(
        owner_payload={},
        series_map=intrinsic._build_series_from_fundamentals({"rows": rows}),
        owner_quality_payload={},
        intangible_payload={},
        maintenance_capex_payload={},
    )


def test_intrinsic_missing_recent_rows_cannot_reuse_old_positive_earnings():
    rows = _cashflow_rows(
        [(2019, 30.0), (2020, 40.0), (2021, 50.0),
         (2022, "UNKNOWN"), (2023, "UNKNOWN"), (2024, "UNKNOWN")]
    )
    result = _power(rows)
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_status"] == "UNKNOWN"
    # The public consumer must not form a price from the old median40 either.
    public = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", fundamentals={"rows": rows}, owner_payload={},
        price_value=10.0, shares_value=10.0, net_debt_value=0.0,
    )
    assert public["normalized_earnings_power_value"] == "UNKNOWN"
    assert public["intrinsic_base"] == "UNKNOWN"


def test_owner_payload_adapter_preserves_recent_unknown_years():
    payload = {"series": [
        {"year": year, "owner_earnings": value, "derived_from": [f"owner:{year}"]}
        for year, value in [(2019, 30.0), (2020, 40.0), (2021, 50.0),
                            (2022, "UNKNOWN"), (2023, "UNKNOWN"), (2024, "UNKNOWN")]
    ]}
    series = intrinsic._owner_series_from_owner_payload(payload)
    assert series[-1] == {"year": 2024, "value": "UNKNOWN", "derived_from": ["owner:2024"]}
    assert intrinsic._median_positive(series) == (
        "UNKNOWN", ["owner:2022", "owner:2023", "owner:2024"], 0
    )


def test_recent_missing_capex_cannot_pull_in_old_owner_earnings():
    rows = _cashflow_rows([(2019, 30.0), (2020, 40.0), (2021, 50.0)])
    rows += [{"year": year, "cfo": -10.0, "capex": "UNKNOWN", "fcf": -10.0}
             for year in [2022, 2023, 2024]]
    assert _power(rows)["normalized_earnings_power_value"] == "UNKNOWN"


def test_known_zero_recent_years_remain_observed_nonpositive_values():
    rows = _cashflow_rows([(2021, 1000.0), (2022, 0.0), (2023, 0.0), (2024, 0.0)])
    series = intrinsic._build_series_from_fundamentals({"rows": rows})
    assert [row["value"] for row in series["owner_earnings"][-3:]] == [0.0, 0.0, 0.0]
    result = _power(rows)
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["INSUFFICIENT_NORMALIZED_INPUTS"]


def _raw_result(monkeypatch, *, cfo, capex, fcf=(), facts_row=None, owner_payload=None,
                revenue=()):
    companyfacts = {"facts": {"us-gaap": {
        tag: {"units": {"USD": [
            {"start": f"{year}-01-01", "end": f"{year}-12-31",
             "filed": f"{year + 1}-02-01", "form": "10-K", "fp": "FY", "val": value}
            for year, value in rows
        ]}}
        for tag, rows in [("NetCashProvidedByUsedInOperatingActivities", cfo),
                          ("PaymentsToAcquirePropertyPlantAndEquipment", capex),
                          ("FreeCashFlow", fcf), ("Revenues", revenue)]
    }}}
    monkeypatch.setattr(intrinsic, "_load_companyfacts_payload", lambda _: companyfacts)
    series = intrinsic._build_series_from_facts(
        as_of_date="2025-03-01", facts_row=facts_row or {}, owner_payload=owner_payload or {},
        net_debt_value=0.0, net_debt_refs=[],
    )
    result = intrinsic.compute_intrinsic_discipline(
        "FIXTURE", "2025-03-01", facts_row=facts_row or {}, owner_payload=owner_payload or {},
        price_value=10.0, shares_value=10.0, net_debt_value=0.0,
    )
    return series, result


def test_raw_recent_missing_capex_cannot_reintroduce_old_derived_fcf(monkeypatch):
    # Old30/40/50 profits cannot fill three later missing cash-spending inputs.
    series, result = _raw_result(
        monkeypatch,
        cfo=[(2019, 30.0), (2020, 40.0), (2021, 50.0),
             (2022, -10.0), (2023, -10.0), (2024, -10.0)],
        capex=[(2019, 0.0), (2020, 0.0), (2021, 0.0)],
    )
    assert [(row["year"], row["value"]) for row in series["fcf"][-3:]] == [
        (2022, "UNKNOWN"), (2023, "UNKNOWN"), (2024, "UNKNOWN")
    ]
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["intrinsic_base"] == "UNKNOWN"


def test_raw_recent_missing_cfo_cannot_reintroduce_old_cfo_proxy(monkeypatch):
    series, result = _raw_result(
        monkeypatch, cfo=[(2019, 30.0), (2020, 40.0), (2021, 50.0)],
        capex=[(year, 0.0) for year in range(2019, 2025)],
    )
    assert [(row["year"], row["value"]) for row in series["cfo"][-3:]] == [
        (2022, "UNKNOWN"), (2023, "UNKNOWN"), (2024, "UNKNOWN")
    ]
    assert [row["value"] for row in series["fcf"][-3:]] == ["UNKNOWN"] * 3
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["intrinsic_base"] == "UNKNOWN"


@pytest.mark.parametrize("newer_source", ["cfo", "capex"])
def test_raw_direct_fcf_respects_newer_declared_financial_years(monkeypatch, newer_source):
    cfo = [(2019, -10.0), (2020, -10.0), (2021, -10.0)]
    capex = [(2019, 0.0), (2020, 0.0), (2021, 0.0)]
    (cfo if newer_source == "cfo" else capex).extend(
        [(year, -10.0 if newer_source == "cfo" else 0.0) for year in (2022, 2023, 2024)]
    )
    series, result = _raw_result(
        monkeypatch, cfo=cfo, capex=capex,
        fcf=[(2019, 30.0), (2020, 40.0), (2021, 50.0)],
    )
    assert [(row["year"], row["value"]) for row in series["fcf"][-3:]] == [
        (2022, "UNKNOWN"), (2023, "UNKNOWN"), (2024, "UNKNOWN")
    ]
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["intrinsic_base"] == "UNKNOWN"


def test_raw_known_zero_capex_preserves_measured_current_losses(monkeypatch):
    series, result = _raw_result(
        monkeypatch,
        cfo=[(2019, 30.0), (2020, 40.0), (2021, 50.0),
             (2022, -10.0), (2023, -10.0), (2024, -10.0)],
        capex=[(year, 0.0) for year in range(2019, 2025)],
    )
    assert [row["value"] for row in series["fcf"][-3:]] == [-10.0] * 3
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["NEGATIVE_NORMALIZED_EARNINGS"]


def test_raw_current_direct_fcf_remains_available_without_cfo(monkeypatch):
    series, result = _raw_result(
        monkeypatch, cfo=[], capex=[], fcf=[(2022, 20.0), (2023, 20.0), (2024, 20.0)],
    )
    assert [row["value"] for row in series["fcf"]] == [20.0] * 3
    assert result["normalized_earnings_power_value"] == 20.0
    assert result["normalized_earnings_power_method_used"] == "FCF_SELECTED"


def test_raw_padding_does_not_invent_wholly_absent_calendar_years(monkeypatch):
    series, result = _raw_result(
        monkeypatch, cfo=[(2019, 30.0), (2021, 50.0)], capex=[(2019, 0.0), (2021, 0.0)],
    )
    assert [row["year"] for row in series["cfo"]] == [2019, 2021]
    assert [row["year"] for row in series["fcf"]] == [2019, 2021]
    assert result["normalized_earnings_power_value"] == 40.0


def test_raw_scalar_only_inputs_retain_existing_undated_fallback(monkeypatch):
    monkeypatch.setattr(intrinsic, "_load_companyfacts_payload", lambda _: {})
    series = intrinsic._build_series_from_facts(
        as_of_date="2025-03-01", facts_row={"cfo_value": 100.0, "fcf_value": 80.0},
        owner_payload={}, net_debt_value=0.0, net_debt_refs=[],
    )
    # Facts-row scalars are USD millions; the series is whole dollars.
    assert series["cfo"] == [{"year": 9999, "value": 100_000_000.0, "derived_from": []}]
    assert series["fcf"] == [{"year": 9999, "value": 80_000_000.0, "derived_from": []}]


def _unknown_current_owner():
    return {
        "as_of_date": "2025-03-01",
        "series": [{"year": year, "owner_earnings": "UNKNOWN"}
                   for year in (2022, 2023, 2024)],
        "summary": {},
    }


def _scalar_facts():
    return {"as_of_date": "2025-03-01", "cfo_status": "OK", "cfo_value": 100.0,
            "fcf_status": "OK", "fcf_value": 80.0}


@pytest.mark.parametrize("raw_source", ["revenue", "capex", "direct_fcf"])
def test_raw_padding_does_not_block_scalars_when_metric_history_never_existed(
    monkeypatch, raw_source
):
    # An UNKNOWN owner year cannot manufacture raw history and disable fallback.
    series, result = _raw_result(
        monkeypatch, cfo=[], capex=[(2024, 20.0)] if raw_source == "capex" else [],
        fcf=[(2024, 60.0)] if raw_source == "direct_fcf" else [],
        revenue=[(2024, 500.0)] if raw_source == "revenue" else [],
        owner_payload=_unknown_current_owner(), facts_row=_scalar_facts(),
    )
    # Facts-row scalars are USD millions; the series is whole dollars.
    assert series["cfo"] == [{"year": 9999, "value": 100_000_000.0, "derived_from": []}]
    if raw_source == "direct_fcf":
        assert [row["value"] for row in series["fcf"]] == ["UNKNOWN", "UNKNOWN", 60.0]
    else:
        assert series["fcf"] == [{"year": 9999, "value": 80_000_000.0, "derived_from": []}]
    # Migrated 2026-09-29: the scalar fallback survives into the series (what this
    # test guards), but one undated or one profitable year is not a normalized
    # level, so no earnings power or price is formed from it (was 80 / 60).
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["normalized_earnings_power_reason_codes"] == ["INSUFFICIENT_NORMALIZED_INPUTS"]
    assert result["intrinsic_base"] == "UNKNOWN"


@pytest.mark.parametrize("direct_fcf", [False, True])
def test_scalar_fallback_does_not_overwrite_old_observed_history_and_recent_gaps(
    monkeypatch, direct_fcf
):
    history = [(2019, 30.0), (2020, 40.0), (2021, 50.0)]
    series, result = _raw_result(
        monkeypatch, cfo=history, capex=[(year, 0.0) for year in range(2019, 2025)],
        fcf=history if direct_fcf else [], facts_row=_scalar_facts(),
    )
    for metric in ("cfo", "fcf"):
        assert [(row["year"], row["value"]) for row in series[metric][-3:]] == [
            (2022, "UNKNOWN"), (2023, "UNKNOWN"), (2024, "UNKNOWN")
        ]
    assert result["normalized_earnings_power_value"] == "UNKNOWN"
    assert result["intrinsic_base"] == "UNKNOWN"
