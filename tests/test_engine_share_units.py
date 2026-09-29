"""The engine's share count is in millions; its label and its guard must say so.

Observed on a live store: ``resolve_shares_asof`` for a filer with about 123 million
shares returned ``123.265922`` with ``shares_unit == "shares_millions"``. That same value flows to engine.py:1426 unchanged, is stamped
``"unit": "shares"`` in the share evidence (engine.py:1647), and is what
``validate_denominators`` receives (engine.py:1255-1259). guards.py's absurd-count
ceiling of 10,000,000,000,000 (guards.py:39) is therefore compared with a number of
MILLIONS: it fires only above ten quadrillion shares and never in practice. The
ceiling was once kept in absolute shares on the premise that "the only caller passes
absolute shares"; the resolver's own unit stamp shows that premise was wrong. Same ceiling, right unit.
"""

from __future__ import annotations

import pytest

from app.valuation.engine import build_ticker_valuation
from app.valuation.guards import validate_denominators


def _payload() -> dict:
    return {
        "ticker": "TST",
        "rows": [
            {
                "year": 2025,
                "fcf": 50.0,
                "cfo": 55.0,
                "capex": 5.0,
                "shares_outstanding": 133.947444,
                "net_debt": 0.0,
                "revenue": 1000.0,
                "op_margin": 0.10,
                "fcf_margin": 0.05,
            }
        ],
        "derived_signals": {},
    }


def test_the_engine_labels_its_share_evidence_in_millions():
    out = build_ticker_valuation(_payload(), run_id=None, as_of_date=None, with_prices=False)
    evidence = out["shares_evidence"]
    assert evidence["shares_outstanding"] == 133.947444
    assert evidence["unit"] == "shares_millions"


@pytest.mark.parametrize(
    "shares_millions",
    [
        0.55,  # Berkshire A, the smallest real cap table
        145.461,  # ResMed FY2021
        15_000.0,  # the largest listed counts
    ],
)
def test_real_share_counts_in_millions_validate(shares_millions):
    ok, reason, details = validate_denominators({"shares_outstanding": shares_millions})
    assert ok is True
    assert reason is None
    assert details["shares_outstanding"] == shares_millions


def test_the_ten_trillion_share_ceiling_is_applied_in_the_callers_unit():
    # Ten trillion shares is 10,000,000 million shares.
    assert validate_denominators({"shares_outstanding": 10_000_000.0})[:2] == (True, None)
    assert validate_denominators({"shares_outstanding": 10_000_001.0})[:2] == (
        False,
        "ABSURD_SHARES",
    )
