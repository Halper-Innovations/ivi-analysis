"""Pure and scratch-only reproductions of valuation defects."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

scratch = Path(globals()["scratch"])  # supplied by the test fixture via runpy.init_globals
os.environ["VOE_DATA_DIR"] = str(scratch / "data")
os.environ["VOE_DB_PATH"] = str(scratch / "data" / "engine.db")
os.environ["VOE_LLM_PROVIDER"] = "disabled"
os.environ["VOE_PRICE_PROVIDER"] = "disabled"

import app
from app.config import get_config
from app.synthesis.variant_builder import _extract_valuation_signals
from app.valuation.accounting_quality import _recent_common_ratios
from app.valuation.valuation_writer import _epv
from app.web.readmodel.company import _drop_rows_superseded_by_a_block, _apply_durable_dcf
from app.valuation.owner_earnings import compute_owner_earnings_series
from app.valuation.net_debt import resolve_net_debt_proxy
from app.research.signals import _age_in_days

results = {"app_path": str(Path(app.__file__).resolve())}
pzd = {"dcf_base": 30.0, "dcf_raw_base": 50.0}
vals = {
    "scorecard": {"outputs": {"pricing_zone_detail": pzd}},
    "dcf": {"outputs": {"base": 50.0}},
    "reverse_dcf": {"inputs": {"price": 10.0}, "outputs": {}},
}
signals, _, _ = _extract_valuation_signals(ticker="TEST", valuations=vals)
results["phantom_adjusted_dcf"] = {
    "input_methods": list(vals),
    "signals": [s.model_dump() for s in signals],
    "expected_types": ["DCF_DISCOUNT"],
}
card = {
    "method": "dcf_adjusted",
    "fair_value": {"kind": "band", "low": 35.0, "base": 40.0, "high": 45.0},
}
_apply_durable_dcf(card, pzd)
results["adjusted_method_overwritten"] = card
rows = [
    {
        "method": "dcf",
        "as_of_date": "2026-08-15",
        "source_run_id": "old",
        "created_at": "2026-08-15T01:00:00Z",
        "quality_gate_verdict": None,
        "outputs_json": '{"status":"OK","base":50.0}',
    },
    {
        "method": "scorecard",
        "as_of_date": "2026-08-15",
        "source_run_id": "new",
        "created_at": "2026-08-15T12:00:00Z",
        "quality_gate_verdict": "BLOCK",
        "outputs_json": '{"signal":"VALUATION_BLOCKED"}',
    },
]
results["same_day_block"] = {
    "actual_methods": [r["method"] for r in _drop_rows_superseded_by_a_block(rows)],
    "expected_methods": ["scorecard"],
}
oi = [(2020, 10.0), (2021, 10.0), (2022, 10.0)]
revenue = [(2020, 100.0), (2021, 100.0), (2022, 100.0), (2023, 500.0)]
results["epv_stale_current_revenue"] = {
    "actual": _epv(oi, 0.0, 100.0, revenue_series=revenue, wacc=0.10),
    "expected_current_revenue": 500.0,
    "expected_per_share": 3.95,
}
rows = [
    *[{"year": y, "cfo": 100.0, "net_income": 100.0} for y in [2019, 2020, 2021]],
    *[{"year": y, "cfo": "UNKNOWN", "net_income": -100.0} for y in [2022, 2023, 2024]],
]
results["accounting_window_skips_missing_recent"] = {
    "actual": _recent_common_ratios(rows, "cfo", "net_income"),
    "expected": [[], []],
}


def fact(start, end, value, filed="2026-02-15"):
    return {"start": start, "end": end, "filed": filed, "form": "10-K", "fp": "FY", "val": value}


def owner_case(name, cfo, capex):
    payload = {
        "companyfacts": {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [cfo]}},
                    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [capex]}},
                }
            }
        }
    }
    path = scratch / f"{name}.json"
    path.write_text(json.dumps(payload))
    import app.valuation.owner_earnings as oe

    original = oe.resolve_financial_facts_asof
    oe.resolve_financial_facts_asof = lambda **kw: {
        "ticker": "TEST",
        "cache_path": str(path),
        "derived_from": [],
    }
    try:
        return compute_owner_earnings_series("TEST", "2026-06-30", maintenance_capex_ratio=0.6)
    finally:
        oe.resolve_financial_facts_asof = original


results["shifted_equal_annual_lengths"] = owner_case(
    "shifted", fact("2024-04-01", "2025-03-31", 150), fact("2025-01-01", "2025-12-31", 30)
)
results["matched_quarters_called_annual"] = owner_case(
    "quarter", fact("2025-01-01", "2025-03-31", 150), fact("2025-01-01", "2025-03-31", 30)
)


def instant(end, value, filed):
    return {"end": end, "filed": filed, "form": "10-Q", "val": value}


payload = {
    "companyfacts": {
        "cik": 900,
        "entityName": "TEST",
        "facts": {
            "us-gaap": {
                "DebtCurrent": {
                    "units": {"USD": [instant("2025-09-30", 200_000_000, "2025-11-15")]}
                },
                "LongTermDebtNoncurrent": {
                    "units": {"USD": [instant("2025-09-30", 800_000_000, "2025-11-15")]}
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {"USD": [instant("2025-09-30", 100_000_000, "2025-11-15")]}
                },
                "OperatingLeaseLiability": {
                    "units": {
                        "USD": [
                            instant("2025-09-30", 300_000_000, "2025-11-15"),
                            instant("2025-12-31", 500_000_000, "2026-01-31"),
                        ]
                    }
                },
            }
        },
    }
}
path = scratch / "lease.json"
path.write_text(json.dumps(payload))
get_config.cache_clear()
result = resolve_net_debt_proxy(
    "TEST",
    "2026-02-14",
    cfg=get_config(),
    facts_row={"ticker": "TEST", "cache_path": str(path), "derived_from": []},
)
results["lease_alignment_uses_wrong_filing_cutoff"] = {
    "actual": result,
    "expected_aligned_lease": 300.0,
    "expected_net_debt": 1200.0,
}
for token in ["1", "true"]:
    os.environ["VOE_ISSUER_CLASSIFICATION_BY_SIC"] = token
    get_config.cache_clear()
    results[f"sic_env_{token}"] = get_config().issuer_classification_by_sic
results["same_calendar_day_freshness"] = {
    "actual": _age_in_days(
        "2024-06-30T12:00:00Z", None, datetime(2024, 6, 30, tzinfo=timezone.utc)
    ),
    "expected": 0,
}


def chart_value_in_temp_db(monkeypatch, tmp_path):
    """Writer-shaped durable-scorecard/raw-method rows; no inherited test imports."""
    from app.db import get_db, init_db
    from app.web.readmodel import company

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    get_config.cache_clear()
    init_db(get_config())
    inputs = {
        "current_price": 28.0,
        "price_as_of_date": "2026-08-01",
        "shares": 100.0,
        "net_debt": 50.0,
    }
    scorecard = {
        "signal": "HOLD",
        "legacy_signal": "FAIRLY_VALUED",
        "type": "STANDARD",
        "pricing_zone": "GROWTH_DEPENDENT",
        "pricing_zone_detail": {
            "current_price": 28.0,
            "epv_adjusted": 20.0,
            "dcf_base": 30.0,
            "dcf_raw_base": 50.0,
            "dcf_durable_base": 30.0,
            "graham_value_per_share": 18.0,
            "gate_action": "PROCEED",
        },
        "quality_context": {
            "gate_action": "PROCEED",
            "dcf_durable": {"base": 30.0, "status": "OK"},
        },
    }
    with get_db() as conn:
        for method, outputs in [
            ("dcf", {"status": "OK", "low": 40.0, "base": 50.0, "high": 60.0, "flags": []}),
            ("scorecard", scorecard),
        ]:
            conn.execute(
                "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json, "
                "warnings_json, created_at, source_run_id, valuation_writer_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "TEST",
                    "2026-08-01",
                    method,
                    json.dumps(inputs),
                    json.dumps(outputs),
                    "[]",
                    "2026-08-01T12:00:00+00:00",
                    "fixture-run",
                    "fixture-writer",
                ),
            )
        chart = company.valuation_evolution(conn, "TEST")
        return next(row for row in chart if row["method"] == "dcf")["value"]
