"""Capex is a magnitude wherever it is deducted.

A filer that reports capital expenditure as a negative element (58 fiscal-year
rows across 36 issuers in the live store when this was written) turns every
``cash flow - capex`` into ``cash flow + capex``: free cash flow and owner
earnings come out ABOVE cash from operations, and the spike caps that key on
``value > 2 x mean`` invert, keeping the spike year and capping the ordinary
ones. The valuation writer and the pre-valuation gate are pinned by
``tests/test_valuation_writer_regressions.py`` and
``tests/test_pre_valuation_gate_regressions.py``; these are the three remaining
producers that make the same deduction.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.valuation.fcf import _extract_fcf_from_dossier
from app.valuation.owner_earnings import compute_owner_earnings_series
from app.valuation.rnd_capitalization import _normalize_capex, _owner_earnings_by_year


def _write_companyfacts(
    path: Path, *, cfo_rows: list[tuple[str, float]], capex_rows: list[tuple[str, float]]
) -> None:
    payload = {
        "companyfacts": {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {"end": end_date, "filed": end_date, "val": value}
                                for end_date, value in cfo_rows
                            ]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {
                            "USD": [
                                {"end": end_date, "filed": end_date, "val": value}
                                for end_date, value in capex_rows
                            ]
                        }
                    },
                }
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_owner_earnings_series_deducts_negated_capex_as_a_magnitude(monkeypatch, tmp_path):
    """CFO 150 against capex -50 is owner earnings 120 at a 0.60 ratio, not 180."""
    cache_path = tmp_path / "companyfacts" / "0000000001.json"
    _write_companyfacts(
        cache_path,
        cfo_rows=[("2024-12-31", 120.0), ("2025-12-31", 150.0)],
        capex_rows=[("2024-12-31", -40.0), ("2025-12-31", -50.0)],
    )
    monkeypatch.setattr(
        "app.valuation.owner_earnings.resolve_financial_facts_asof",
        lambda **_kwargs: {
            "ticker": "AAA",
            "cache_path": str(cache_path),
            "derived_from": ["facts.AAA"],
        },
    )
    payload = compute_owner_earnings_series(
        "AAA",
        "2026-02-14",
        years_back=5,
        maintenance_capex_ratio=0.60,
    )
    row_2025 = {int(row["year"]): row for row in payload["series"]}[2025]
    assert row_2025["capex"] == -50.0
    assert row_2025["maintenance_capex_proxy"] == 30.0
    assert row_2025["owner_earnings"] == 120.0


def test_fcf_from_dossier_deducts_negated_capex_as_a_magnitude():
    """FCF = CFO - |capex|; the filed sign is still reported in capex_value."""
    payload = {
        "ticker": "AAA",
        "as_of_date": "2025-06-30",
        "time_series": {"standardized_rows": [{"year": 2024, "cfo": 100.0, "capex": -20.0}]},
    }
    out = _extract_fcf_from_dossier(payload, run_id="run_1", ticker="AAA")
    assert out["capex_value"] == -20.0
    assert out["fcf_value"] == 80.0


def test_rnd_capitalization_owner_earnings_deducts_negated_capex_as_a_magnitude():
    """The R&D module carries its own copy of both the spike cap and the deduction."""
    negated = [(2024, -100.0), (2023, -20.0), (2022, -20.0), (2021, -20.0), (2020, -20.0)]
    assert _normalize_capex(negated) == _normalize_capex([(year, -v) for year, v in negated])

    rows = _owner_earnings_by_year(
        cfo_rows=[{"year": 2024, "value": 100.0}],
        capex_rows=[{"year": 2024, "value": -20.0}],
        sbc_rows=[{"year": 2024, "value": 0.0}],
    )
    assert rows[2024]["normalized_capex"] == 20.0
    assert rows[2024]["value"] == 80.0
