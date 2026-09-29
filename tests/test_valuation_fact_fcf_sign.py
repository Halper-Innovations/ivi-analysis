"""Cash-flow deductions use spending magnitude; source amounts keep their signs."""

import pytest

from app.config import AppConfig
from app.market.company_facts_extract import extract_company_facts_asof
from app.valuation import facts, fcf


def _companyfacts(cfo: float, capex: float) -> dict:
    def fact(value: float) -> dict:
        return {
            "units": {
                "USD": [
                    {
                        "start": "2024-01-01",
                        "end": "2024-12-31",
                        "filed": "2025-02-01",
                        "form": "10-K",
                        "fp": "FY",
                        "fy": 2024,
                        "val": value,
                    }
                ]
            }
        }

    return {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": fact(cfo),
                "PaymentsToAcquirePropertyPlantAndEquipment": fact(capex),
            }
        }
    }


@pytest.mark.parametrize(
    ("cfo", "capex", "expected"),
    [(100.0, 20.0, 80.0), (100.0, -20.0, 80.0), (-10.0, -20.0, -30.0)],
)
def test_fact_fcf_deducts_spending_and_preserves_source_signs(cfo, capex, expected):
    extracted = extract_company_facts_asof(_companyfacts(cfo, capex), "2025-03-01")

    # 100 - 20 = 80 under either spending sign; -10 - 20 remains a loss of 30.
    assert extracted["fcf_asof"]["value"] == expected
    assert extracted["fcf_asof"]["computation"] == "CFO_MINUS_CAPEX"
    assert extracted["fcf_asof"]["bridge_context"]["formula"] == "FCF = CFO - abs(CapEx)"
    assert extracted["fcf_asof"]["bridge_context"]["cfo_value"] == cfo
    assert extracted["fcf_asof"]["bridge_context"]["capex_value"] == capex
    assert extracted["cfo_asof"]["value"] == cfo
    assert extracted["capex_asof"]["value"] == capex


@pytest.mark.parametrize(
    ("cfo_millions", "capex_millions", "expected_millions"),
    [(100.0, 20.0, 80.0), (100.0, -20.0, 80.0), (-10.0, -20.0, -30.0)],
)
def test_fact_fcf_sign_survives_facts_and_fcf_resolvers(
    monkeypatch, tmp_path, cfo_millions, capex_millions, expected_millions
):
    cfg = AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "engine.db",
        universe_path=tmp_path / "universe.csv",
        cache_dir=tmp_path / "cache",
    )
    payload = _companyfacts(cfo_millions * 1_000_000.0, capex_millions * 1_000_000.0)
    monkeypatch.setattr(fcf, "get_config", lambda: cfg)
    monkeypatch.setattr(fcf, "_iter_historical_dossiers", lambda **_kwargs: [])
    monkeypatch.setattr(facts, "resolve_cik_for_ticker", lambda *_args, **_kwargs: "0000000001")
    monkeypatch.setattr(
        facts,
        "fetch_company_facts",
        lambda *_args, **_kwargs: {
            "status": "OK",
            "reason_code": "CACHE_HIT",
            "source_resolution": "companyfacts_cache",
            "companyfacts": payload,
            "network_attempted": False,
        },
    )
    facts.clear_facts_row_cache()
    try:
        # Both production resolvers run. Only external acquisition is replaced.
        value, coverage = fcf.resolve_fcf_asof("FIXTURE", "2025-03-01", None)
        assert value == expected_millions
        assert coverage["fcf_value"] == expected_millions
        assert coverage["fcf_status"] == "OK"
        assert coverage["fcf_reason_code"] == "COMPANYFACTS_CFO_CAPEX_HIT"
        assert coverage["cfo_value"] == cfo_millions
        assert coverage["capex_value"] == capex_millions
        assert coverage["network_attempted"] is False
    finally:
        facts.clear_facts_row_cache()
