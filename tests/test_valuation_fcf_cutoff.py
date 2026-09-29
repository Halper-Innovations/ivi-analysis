"""Hand-derived, scratch-only regressions for one formula defect."""

from app.valuation import fcf


def test_fcf_current_run_future_dossier_cannot_supply_historical_value(monkeypatch):
    """A dossier dated 2025 cannot supply an FCF requested as of 2024; no older source exists."""
    monkeypatch.setattr(
        fcf,
        "_safe_json",
        lambda _: {
            "as_of_date": "2025-03-01",
            "time_series": {"standardized_rows": [{"year": 2024, "fcf": 80.0}]},
        },
    )
    monkeypatch.setattr(fcf, "_iter_historical_dossiers", lambda **_: [])
    monkeypatch.setattr(fcf, "resolve_financial_facts_asof", lambda **_: {})
    value, _ = fcf.resolve_fcf_asof("FIXTURE", "2024-03-01", "future-run")
    assert value is None


def test_current_run_dossier_at_requested_cutoff_remains_eligible(monkeypatch):
    monkeypatch.setattr(
        fcf,
        "_safe_json",
        lambda _: {
            "as_of_date": "2024-03-01",
            "time_series": {"standardized_rows": [{"year": 2023, "fcf": 80.0}]},
        },
    )
    value, coverage = fcf.resolve_fcf_asof("FIXTURE", "2024-03-01", "current")
    assert value == 80.0
    assert coverage["fcf_source"] == "current_run_dossier"
