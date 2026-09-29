from __future__ import annotations

import json
from pathlib import Path

from app.valuation.graham_dodd import UNKNOWN, _fact_rows, compute_graham_dodd_overlay


def _write_companyfacts(
    path: Path,
    *,
    shares: float = 10.0,
    current_assets: float | None = None,
    liabilities: float | None = None,
    preferred_stock: float | None = None,
    fcf_rows: list[tuple[str, float]] | None = None,
    cfo_rows: list[tuple[str, float]] | None = None,
    capex_rows: list[tuple[str, float]] | None = None,
) -> None:
    payload = {
        "companyfacts": {
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"end": "2025-12-31", "filed": "2026-01-31", "val": shares},
                            ]
                        }
                    }
                },
                "us-gaap": {
                    **(
                        {
                            "AssetsCurrent": {
                                "units": {
                                    "USD": [
                                        {
                                            "end": "2025-12-31",
                                            "filed": "2026-01-31",
                                            "val": current_assets,
                                        }
                                    ]
                                }
                            }
                        }
                        if isinstance(current_assets, (int, float))
                        else {}
                    ),
                    **(
                        {
                            "Liabilities": {
                                "units": {
                                    "USD": [
                                        {
                                            "end": "2025-12-31",
                                            "filed": "2026-01-31",
                                            "val": liabilities,
                                        }
                                    ]
                                }
                            }
                        }
                        if isinstance(liabilities, (int, float))
                        else {}
                    ),
                    **(
                        {
                            "PreferredStockValue": {
                                "units": {
                                    "USD": [
                                        {
                                            "end": "2025-12-31",
                                            "filed": "2026-01-31",
                                            "val": preferred_stock,
                                        }
                                    ]
                                }
                            }
                        }
                        if isinstance(preferred_stock, (int, float))
                        else {}
                    ),
                    **(
                        {
                            "FreeCashFlow": {
                                "units": {
                                    "USD": [
                                        {"end": end_date, "filed": end_date, "val": value}
                                        for end_date, value in fcf_rows
                                    ]
                                }
                            }
                        }
                        if fcf_rows
                        else {}
                    ),
                    **(
                        {
                            "NetCashProvidedByUsedInOperatingActivities": {
                                "units": {
                                    "USD": [
                                        {"end": end_date, "filed": end_date, "val": value}
                                        for end_date, value in cfo_rows
                                    ]
                                }
                            }
                        }
                        if cfo_rows
                        else {}
                    ),
                    **(
                        {
                            "PaymentsToAcquirePropertyPlantAndEquipment": {
                                "units": {
                                    "USD": [
                                        {"end": end_date, "filed": end_date, "val": value}
                                        for end_date, value in capex_rows
                                    ]
                                }
                            }
                        }
                        if capex_rows
                        else {}
                    ),
                },
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_epv_computation_from_owner_earnings_normalized(tmp_path):
    cache_path = tmp_path / "companyfacts" / "0000000001.json"
    _write_companyfacts(cache_path, shares=10.0)
    facts_row = {
        "ticker": "AAA",
        "shares_status": "OK",
        "shares_value": 0.00001,  # 10 shares; facts rows carry shares in millions
        "cache_path": str(cache_path),
        "derived_from": ["facts.AAA"],
    }
    owner_payload = {
        "summary": {
            "owner_earnings_normalized_3y": 100.0,
            "owner_earnings_normalized_method": "MEDIAN_3Y",
        },
        "derived_from": ["owner.AAA"],
    }
    payload = compute_graham_dodd_overlay(
        ticker="AAA",
        as_of_date="2026-02-14",
        price_value=20.0,
        price_status="OK",
        discount_rate=0.10,
        facts_row=facts_row,
        owner_payload=owner_payload,
    )
    assert payload["epv_status"] == "OK"
    assert payload["epv_value"] == 1000.0
    assert payload["epv_per_share"] == 100.0
    assert payload["mos_epv"] == 4.0


def test_netnet_computation_from_facts(tmp_path):
    cache_path = tmp_path / "companyfacts" / "0000000002.json"
    _write_companyfacts(
        cache_path,
        shares=10.0,
        current_assets=500.0,
        liabilities=300.0,
        preferred_stock=20.0,
    )
    facts_row = {
        "ticker": "BBB",
        "shares_status": "OK",
        "shares_value": 0.00001,  # 10 shares; facts rows carry shares in millions
        "cache_path": str(cache_path),
        "derived_from": ["facts.BBB"],
    }
    payload = compute_graham_dodd_overlay(
        ticker="BBB",
        as_of_date="2026-02-14",
        price_value=20.0,
        price_status="OK",
        discount_rate=0.10,
        facts_row=facts_row,
        owner_payload={"summary": {}, "derived_from": []},
    )
    assert payload["netnet_status"] == "OK"
    assert payload["netnet_value"] == 180.0
    assert payload["netnet_per_share"] == 18.0
    assert round(float(payload["mos_netnet"]), 6) == -0.1


def test_mos_requires_price(tmp_path):
    cache_path = tmp_path / "companyfacts" / "0000000003.json"
    _write_companyfacts(
        cache_path,
        shares=10.0,
        current_assets=500.0,
        liabilities=300.0,
        preferred_stock=0.0,
    )
    facts_row = {
        "ticker": "CCC",
        "shares_status": "OK",
        "shares_value": 0.00001,  # 10 shares; facts rows carry shares in millions
        "cache_path": str(cache_path),
        "derived_from": ["facts.CCC"],
    }
    owner_payload = {
        "summary": {
            "owner_earnings_normalized_3y": 100.0,
            "owner_earnings_normalized_method": "MEDIAN_3Y",
        },
        "derived_from": ["owner.CCC"],
    }
    payload = compute_graham_dodd_overlay(
        ticker="CCC",
        as_of_date="2026-02-14",
        price_value=UNKNOWN,
        price_status="UNKNOWN",
        discount_rate=0.10,
        facts_row=facts_row,
        owner_payload=owner_payload,
    )
    assert payload["epv_status"] == "OK"
    assert payload["mos_epv"] == UNKNOWN
    assert payload["mos_netnet"] == UNKNOWN
    assert payload["mos_epv_reason_code"] == "PRICE_UNKNOWN"
    assert payload["mos_netnet_reason_code"] == "PRICE_UNKNOWN"


def test_graham_mos_epv_uses_upside_ratio_convention(tmp_path):
    """FIX 6: mos_epv uses the UPSIDE-RATIO convention (epv_per_share/price - 1).

    For epv_per_share=150, price=100 the upside-ratio MoS is 0.50
    (NOT the textbook (intrinsic-price)/intrinsic = 0.333...).
    """
    cache_path = tmp_path / "companyfacts" / "0000000009.json"
    _write_companyfacts(cache_path, shares=10.0)
    facts_row = {
        "ticker": "UPS",
        "shares_status": "OK",
        "shares_value": 0.00001,  # 10 shares; facts rows carry shares in millions
        "cache_path": str(cache_path),
        "derived_from": ["facts.UPS"],
    }
    # EPV = owner_earnings_normalized / discount_rate = 150 / 0.10 = 1500 EV
    # epv_per_share = 1500 / 10 shares = 150
    owner_payload = {
        "summary": {
            "owner_earnings_normalized_3y": 150.0,
            "owner_earnings_normalized_method": "MEDIAN_3Y",
        },
        "derived_from": ["owner.UPS"],
    }
    payload = compute_graham_dodd_overlay(
        ticker="UPS",
        as_of_date="2026-02-14",
        price_value=100.0,
        price_status="OK",
        discount_rate=0.10,
        facts_row=facts_row,
        owner_payload=owner_payload,
    )
    assert payload["epv_per_share"] == 150.0
    assert payload["mos_epv"] == 0.50


def test_graham_fact_rows_reject_missing_and_post_asof_filing_dates():
    companyfacts = {
        "facts": {
            "us-gaap": {
                "AssetsCurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2023-12-31",
                                "filed": "2024-02-20",
                                "accn": "0001",
                                "val": 100.0,
                            },
                            {
                                "end": "2024-12-31",
                                "filed": "2025-02-20",
                                "val": 200.0,
                            },
                            {
                                "end": "2025-12-31",
                                "val": 300.0,
                            },
                        ]
                    }
                }
            }
        }
    }

    rows = _fact_rows(
        companyfacts=companyfacts,
        taxonomy="us-gaap",
        tag="AssetsCurrent",
        as_of_date="2025-01-15",
        expected_unit_exact=("usd",),
    )

    assert [(row["end_date"], row["value"]) for row in rows] == [("2023-12-31", 100.0)]
    assert rows[0]["filed"] == "2024-02-20"
    assert "filed=2024-02-20" in rows[0]["ref"]
    assert "accn=0001" in rows[0]["ref"]


def test_facts_row_shares_are_millions_and_per_share_values_use_whole_shares(tmp_path):
    """A facts row reports shares in millions (``shares_output_unit ==
    "shares_millions"``); owner earnings are whole dollars. Owner earnings of
    $50,000,000 at 10% over 100,000,000 shares is an EPV of $500,000,000 and
    $5.00 a share — not $5,000,000 a share, which dividing by the raw row value
    (100.0) gave. The facts-row path and the companyfacts fallback agree.
    """
    cache_path = tmp_path / "companyfacts" / "0000000009.json"
    _write_companyfacts(cache_path, shares=100_000_000.0)
    owner_payload = {
        "summary": {
            "owner_earnings_normalized_3y": 50_000_000.0,
            "owner_earnings_normalized_method": "MEDIAN_3Y",
        },
        "derived_from": ["owner.AAA"],
    }
    per_share = []
    for facts_row in (
        {"ticker": "AAA", "shares_status": "OK", "shares_value": 100.0,
         "shares_output_unit": "shares_millions", "cache_path": str(cache_path),
         "derived_from": ["facts.AAA"]},
        {"ticker": "AAA", "shares_status": "UNKNOWN", "cache_path": str(cache_path),
         "derived_from": ["facts.AAA"]},
    ):
        payload = compute_graham_dodd_overlay(
            ticker="AAA",
            as_of_date="2026-02-14",
            price_value=4.0,
            price_status="OK",
            discount_rate=0.10,
            facts_row=facts_row,
            owner_payload=owner_payload,
        )
        per_share.append(payload["epv_per_share"])
    assert per_share == [5.0, 5.0]
