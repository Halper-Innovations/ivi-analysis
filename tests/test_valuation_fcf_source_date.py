"""Source-date proof for historical FCF admission; all sources are in-memory fixtures."""

from datetime import datetime
from pathlib import Path

import pytest

from app.valuation import fcf


@pytest.mark.parametrize("source_date", [None, "not-a-date"], ids=["undated", "malformed"])
def test_fcf_unknown_source_date_cannot_supply_asof_value(monkeypatch, source_date):
    """Neither absent nor invalid source date proves FCF availability by the cutoff."""
    payload = {"time_series": {"standardized_rows": [{"year": 2024, "fcf": 80.0}]}}
    if source_date is not None:
        payload["as_of_date"] = source_date
    monkeypatch.setattr(fcf, "_safe_json", lambda _: payload)
    monkeypatch.setattr(fcf, "_iter_historical_dossiers", lambda **_: [])
    monkeypatch.setattr(fcf, "resolve_financial_facts_asof", lambda **_: {})
    value, coverage = fcf.resolve_fcf_asof("FIXTURE", "2024-03-01", "unestablished-source")
    assert value is None
    assert coverage["fcf_status"] == "UNKNOWN"


def test_fcf_known_eligible_source_date_retains_exact_value(monkeypatch):
    payload = {
        "as_of_date": "2024-03-01",
        "time_series": {"standardized_rows": [{"year": 2023, "fcf": 80.0}]},
    }
    monkeypatch.setattr(fcf, "_safe_json", lambda _: payload)
    monkeypatch.setattr(fcf, "_iter_historical_dossiers", lambda **_: [])
    monkeypatch.setattr(fcf, "resolve_financial_facts_asof", lambda **_: {})
    value, coverage = fcf.resolve_fcf_asof("FIXTURE", "2024-03-01", "eligible-source")
    assert value == 80.0
    assert coverage["fcf_status"] == "OK"


@pytest.mark.parametrize("source_date", [None, "not-a-date"])
def test_undated_current_source_uses_the_eligible_historical_value(monkeypatch, source_date):
    current = {
        "as_of_date": source_date,
        "time_series": {"standardized_rows": [{"year": 2024, "fcf": 80.0}]},
    }
    historical = {
        "as_of_date": "2024-02-01",
        "time_series": {"standardized_rows": [{"year": 2023, "fcf": 20.0}]},
    }
    monkeypatch.setattr(fcf, "_safe_json", lambda _: current)
    monkeypatch.setattr(
        fcf,
        "_iter_historical_dossiers",
        lambda **_: [(datetime(2024, 2, 1), "earlier", Path("historical.json"), historical)],
    )
    value, coverage = fcf.resolve_fcf_asof("FIXTURE", "2024-03-01", "undated")
    assert value == 20.0
    assert coverage["fcf_source"] == "historical_dossier:earlier"
    assert coverage["fcf_asof_used"] == "2024-02-01"
