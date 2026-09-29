from __future__ import annotations

import json
from pathlib import Path

from app.valuation.owner_earnings import _annual_series_for_priority, compute_owner_earnings_series


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


def test_owner_earnings_maintenance_capex_proxy_is_deterministic(monkeypatch, tmp_path):
    cache_path = tmp_path / "companyfacts" / "0000000001.json"
    _write_companyfacts(
        cache_path,
        cfo_rows=[("2024-12-31", 120.0), ("2025-12-31", 150.0)],
        capex_rows=[("2024-12-31", 40.0), ("2025-12-31", 50.0)],
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
    rows_by_year = {int(row["year"]): row for row in payload["series"]}
    row_2025 = rows_by_year[2025]
    assert row_2025["maintenance_capex_proxy"] == 30.0
    assert row_2025["owner_earnings"] == 120.0
    assert row_2025["proxy_flags"]["maintenance_capex_proxy_applied"] is True


def test_owner_earnings_normalization_prefers_median_then_avg(monkeypatch, tmp_path):
    cache_path = tmp_path / "companyfacts" / "0000000002.json"
    _write_companyfacts(
        cache_path,
        cfo_rows=[("2023-12-31", 140.0), ("2024-12-31", 150.0), ("2025-12-31", 160.0)],
        capex_rows=[("2023-12-31", 40.0), ("2024-12-31", 50.0), ("2025-12-31", 60.0)],
    )
    monkeypatch.setattr(
        "app.valuation.owner_earnings.resolve_financial_facts_asof",
        lambda **_kwargs: {
            "ticker": "BBB",
            "cache_path": str(cache_path),
            "derived_from": ["facts.BBB"],
        },
    )
    payload = compute_owner_earnings_series(
        "BBB", "2026-02-14", years_back=5, maintenance_capex_ratio=0.60
    )
    assert payload["summary"]["owner_earnings_normalized_method"] == "MEDIAN_3Y"
    assert payload["summary"]["owner_earnings_normalized_3y"] == 120.0

    payload_2y = compute_owner_earnings_series(
        "BBB", "2026-02-14", years_back=2, maintenance_capex_ratio=0.60
    )
    assert payload_2y["summary"]["owner_earnings_normalized_method"] == "AVG_2Y"
    assert payload_2y["summary"]["owner_earnings_normalized_3y"] == 122.0


def test_owner_earnings_reason_codes_are_stable(monkeypatch, tmp_path):
    cache_path = tmp_path / "companyfacts" / "0000000003.json"
    _write_companyfacts(
        cache_path,
        cfo_rows=[("2025-12-31", -10.0)],
        capex_rows=[("2025-12-31", 20.0)],
    )
    monkeypatch.setattr(
        "app.valuation.owner_earnings.resolve_financial_facts_asof",
        lambda **_kwargs: {
            "ticker": "CCC",
            "cache_path": str(cache_path),
            "derived_from": ["facts.CCC"],
        },
    )
    payload = compute_owner_earnings_series(
        "CCC", "2026-02-14", years_back=5, maintenance_capex_ratio=0.60
    )
    reason_codes = set(payload["reason_codes"])
    assert "NEGATIVE_CFO" in reason_codes
    assert "NEGATIVE_OWNER_EARNINGS" in reason_codes


def test_annual_series_prefers_fy_rows_over_interim_ytd_rows():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "ResearchAndDevelopmentExpense": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-05-31",
                                "start": "2024-06-01",
                                "filed": "2025-07-30",
                                "form": "10-K",
                                "fp": "FY",
                                "val": 9860.0,
                            },
                            {
                                "end": "2025-11-30",
                                "start": "2025-06-01",
                                "filed": "2025-12-20",
                                "form": "10-Q",
                                "fp": "Q2",
                                "val": 5051.0,
                            },
                            {
                                "end": "2026-02-28",
                                "start": "2025-06-01",
                                "filed": "2026-03-11",
                                "form": "10-Q",
                                "fp": "Q3",
                                "val": 7658.0,
                            },
                        ]
                    }
                }
            }
        }
    }

    series = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date="2026-03-23",
        priority=[("us-gaap", "ResearchAndDevelopmentExpense")],
        expected_unit_exact=("usd",),
    )

    latest = series[-1]
    assert latest["year"] == 2025
    assert latest["value"] == 9860.0
    assert latest["end_date"] == "2025-05-31"
    assert latest["form"] == "10-K"


def test_annual_series_rejects_missing_and_post_asof_filing_dates():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "ResearchAndDevelopmentExpense": {
                    "units": {
                        "USD": [
                            {
                                "end": "2023-12-31",
                                "filed": "2024-02-20",
                                "form": "10-K",
                                "fp": "FY",
                                "val": 70.0,
                            },
                            {
                                "end": "2024-12-31",
                                "filed": "2025-02-20",
                                "form": "10-K",
                                "fp": "FY",
                                "val": 80.0,
                            },
                            {
                                "end": "2025-12-31",
                                "form": "10-K",
                                "fp": "FY",
                                "val": 90.0,
                            },
                        ]
                    }
                }
            }
        }
    }

    series = _annual_series_for_priority(
        companyfacts=companyfacts,
        as_of_date="2025-01-15",
        priority=[("us-gaap", "ResearchAndDevelopmentExpense")],
        expected_unit_exact=("usd",),
    )

    assert [(row["year"], row["value"]) for row in series] == [(2023, 70.0)]
    assert series[0]["filed"] == "2024-02-20"
    assert "filed=2024-02-20" in series[0]["ref"]


# ── both sides of the subtraction must cover the same period ──────────────────
# CFO and capex were bucketed by calendar year alone, so
# a full-year cash flow could be paired with a nine-month capex and the
# difference published as a year of owner earnings.


def _write_mixed_period_companyfacts(path: Path) -> None:
    payload = {
        "companyfacts": {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-01-01",
                                    "end": "2025-12-31",
                                    "filed": "2026-02-01",
                                    "fp": "FY",
                                    "form": "10-K",
                                    "val": 150.0,
                                }
                            ]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {
                            "USD": [
                                {
                                    # Nine months, filed inside the same 10-K.
                                    "start": "2025-01-01",
                                    "end": "2025-09-30",
                                    "filed": "2026-02-01",
                                    "fp": "FY",
                                    "form": "10-K",
                                    "val": 30.0,
                                }
                            ]
                        }
                    },
                }
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_a_nine_month_capex_does_not_pair_with_a_full_year_cash_flow(monkeypatch, tmp_path):
    cache_path = tmp_path / "companyfacts" / "0000000002.json"
    _write_mixed_period_companyfacts(cache_path)
    monkeypatch.setattr(
        "app.valuation.owner_earnings.resolve_financial_facts_asof",
        lambda **_kwargs: {
            "ticker": "BBB",
            "cache_path": str(cache_path),
            "derived_from": ["facts.BBB"],
        },
    )
    payload = compute_owner_earnings_series("BBB", "2026-06-30", years_back=5)
    row = {int(r["year"]): r for r in payload["series"]}[2025]
    assert "PERIOD_MISMATCH" in row["reason_codes"]
    assert row["owner_earnings"] == "UNKNOWN"
    assert row["maintenance_capex_proxy"] == "UNKNOWN"
