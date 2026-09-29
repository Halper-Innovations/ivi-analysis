"""NaN and infinity are never numbers the system acts on.

A NaN fundamental must not count toward data completeness, and a hand-written
buy target of infinity must not become the watchlist or memo buy price.
"""

from __future__ import annotations

import math

import pytest

from app.autonomous.sector_contract import SectorCompanyFinancialPacket
from app.fundamentals.normalize import UNKNOWN
from app.score.rubric import score_packet


def _packet(**fundamentals):
    return {
        "fundamentals": fundamentals,
        "valuations": {"reverse_dcf": {"inputs": {"market_price": "UNKNOWN"}}},
        "deltas_vs_prior_period": {},
        "extracted_facts": [],
    }


def test_nan_metrics_are_not_known():
    with_nan = score_packet(_packet(revenue=1000.0, fcf=math.nan, net_debt=math.inf))[0]
    with_unknown = score_packet(_packet(revenue=1000.0, fcf=UNKNOWN, net_debt=UNKNOWN))[0]
    assert with_nan["data_completeness"] == with_unknown["data_completeness"]


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_non_finite_hand_written_buy_target_is_ignored(bad):
    from app.autonomous import sector_report
    from app.watchlist import store

    packet = SectorCompanyFinancialPacket(
        ticker="INF",
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        current_price=70.0,
        valuation={"buy_below_price": bad},
    )
    assert store._explicit_buy_price_target(packet) is None
    # No anchor either: nothing to buy at.
    assert store._buy_price_target(packet) is None
    assert sector_report._buy_price_target(packet) is None
