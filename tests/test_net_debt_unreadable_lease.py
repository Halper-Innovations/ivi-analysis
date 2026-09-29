"""app/valuation/net_debt.py must not book an unreadable lease as a silent zero.

The lease-inclusive proxy is
``debt + lease − cash`` where ``lease_value = ... else 0.0`` when the operating-lease
liability cannot be read (net_debt.py:222). Observed on a cache carrying only a debt
and a cash fact: ``net_debt_proxy`` 500.0 (debt 800 − cash 300), lease UNKNOWN,
``net_debt_flags`` [] — nothing downstream can tell an assumed zero from a read one.
The lease-inclusive number is the leverage basis; an assumed term in it must be named.
(The flag is LEASE_LIABILITY_UNKNOWN, the name tests/test_net_debt_resolver.py
already expects for the same condition.) Whether such a proxy may keep HIGH confidence is a
contract question (tests/test_net_debt_resolver.py pins HIGH on debt and cash alone)
and is left open.
"""

from __future__ import annotations

import json

from app.valuation.net_debt import resolve_net_debt_proxy


def _cache(tmp_path, facts: dict) -> str:
    path = tmp_path / "companyfacts.json"
    path.write_text(json.dumps({"facts": {"us-gaap": facts}}), encoding="utf-8")
    return str(path)


def _fact(value: float) -> dict:
    return {
        "units": {
            "USD": [
                {
                    "val": value,
                    "end": "2025-12-31",
                    "filed": "2026-02-15",
                    "form": "10-K",
                    "fp": "FY",
                    "accn": "0000000000-26-000001",
                }
            ]
        }
    }


def test_an_unreadable_lease_is_named_in_the_flags(tmp_path):
    cache_path = _cache(
        tmp_path,
        {"Debt": _fact(800_000_000), "CashAndCashEquivalentsAtCarryingValue": _fact(300_000_000)},
    )
    out = resolve_net_debt_proxy(
        "TST", "2026-06-01", facts_row={"cache_path": cache_path, "derived_from": [], "status": "OK"}
    )
    assert out["status"] == "OK"
    assert out["net_debt_proxy"] == 500.0
    assert out["net_debt_proxy_lease_exclusive"] == 500.0
    assert "LEASE_LIABILITY_UNKNOWN" in out["net_debt_flags"]


def test_a_read_lease_is_lease_adjusted_not_unknown(tmp_path):
    cache_path = _cache(
        tmp_path,
        {
            "Debt": _fact(800_000_000),
            "CashAndCashEquivalentsAtCarryingValue": _fact(300_000_000),
            "OperatingLeaseLiability": _fact(50_000_000),
        },
    )
    out = resolve_net_debt_proxy(
        "TST", "2026-06-01", facts_row={"cache_path": cache_path, "derived_from": [], "status": "OK"}
    )
    assert out["net_debt_proxy"] == 550.0
    assert "LEASE_ADJUSTED" in out["net_debt_flags"]
    assert "LEASE_LIABILITY_UNKNOWN" not in out["net_debt_flags"]
    assert out["net_debt_confidence"] == "HIGH"
