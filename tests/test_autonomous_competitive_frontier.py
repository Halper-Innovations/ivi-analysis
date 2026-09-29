from __future__ import annotations

import pytest

from app.autonomous.competitive_frontier import (
    CompetitiveFrontierState,
    build_competitive_frontier,
    validate_closed_competitive_frontier_payload,
)
from app.autonomous.sector_contract import (
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
)


def _packet(
    ticker: str,
    score: float | None,
    *,
    valuation: dict | None = None,
    returns_on_capital: dict | None = None,
) -> SectorCompanyFinancialPacket:
    score_components = {"deterministic_score": score} if score is not None else {}
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="READY",
        model_fit_status="SUPPORTED",
        data_quality_status="COMPLETE",
        valuation=dict(valuation or {}),
        returns_on_capital=dict(returns_on_capital or {}),
        score_components=score_components,
    )


def _base_scenario(
    ticker: str,
    annualized_return: float | None,
    *,
    unsupported_assumptions: list[str] | None = None,
    horizon_years: int = 5,
) -> SectorExpectedReturnScenario:
    return SectorExpectedReturnScenario(
        scenario_id=f"{ticker}-base-{horizon_years}",
        ticker=ticker,
        scenario_name="base",
        horizon_years=horizon_years,
        current_price=10.0,
        estimated_future_value_per_share=20.0,
        annualized_return=annualized_return,
        unsupported_assumptions=list(unsupported_assumptions or []),
    )


def _persisted_closed_payload(state: CompetitiveFrontierState) -> dict:
    payload = state.to_dict()
    payload.update(
        {
            "status": "CLOSED",
            "minimum_reviews_required": min(3, len(state.candidates)),
            "successful_review_count": len(state.reviewed_tickers),
            "attempted_tickers": list(state.reviewed_tickers),
            "failed_review_tickers": [],
        }
    )
    return payload


def test_closed_frontier_payload_rebuilds_and_reconciles_operational_counts() -> None:
    state = build_competitive_frontier(
        [_packet("AAA", 9.0), _packet("BBB", 8.0)],
        [_base_scenario("AAA", 0.12), _base_scenario("BBB", 0.08)],
        reviewed_tickers=["AAA", "BBB"],
    )
    payload = _persisted_closed_payload(state)

    rebuilt = validate_closed_competitive_frontier_payload(
        payload,
        expected_candidate_tickers=["BBB", "AAA"],
        expected_reviewed_tickers=["AAA", "BBB"],
    )

    assert rebuilt.closure_certificate.closed is True
    assert rebuilt.reviewed_tickers == ("AAA", "BBB")


def test_closed_frontier_payload_rejects_coherent_metrics_forged_away_from_sources() -> None:
    source_packets = [_packet("AAA", 9.0), _packet("BBB", 8.0)]
    source_scenarios = [_base_scenario("AAA", 0.12), _base_scenario("BBB", 0.08)]
    forged_state = build_competitive_frontier(
        [_packet("AAA", 1.0), _packet("BBB", 99.0)],
        [_base_scenario("AAA", 0.01), _base_scenario("BBB", 0.90)],
        reviewed_tickers=["AAA", "BBB"],
    )
    forged_payload = _persisted_closed_payload(forged_state)

    with pytest.raises(ValueError, match="source-backed state"):
        validate_closed_competitive_frontier_payload(
            forged_payload,
            expected_candidate_tickers=["AAA", "BBB"],
            expected_reviewed_tickers=["AAA", "BBB"],
            source_company_packets=source_packets,
            source_scenarios=source_scenarios,
        )


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("frontier_tickers", ["BBB"]),
        ("dominated_tickers", []),
        ("successful_review_count", 1),
        ("failed_review_tickers", ["BBB"]),
    ],
)
def test_closed_frontier_payload_rejects_forged_derived_or_count_fields(
    field: str,
    forged_value: object,
) -> None:
    state = build_competitive_frontier(
        [_packet("AAA", 9.0), _packet("BBB", 8.0)],
        [_base_scenario("AAA", 0.12), _base_scenario("BBB", 0.08)],
        reviewed_tickers=["AAA", "BBB"],
    )
    payload = _persisted_closed_payload(state)
    payload[field] = forged_value

    with pytest.raises(ValueError, match="competitive frontier"):
        validate_closed_competitive_frontier_payload(payload)


def test_closed_frontier_payload_rejects_open_or_incomplete_review_proof() -> None:
    state = build_competitive_frontier(
        [_packet("AAA", 9.0), _packet("BBB", 8.0)],
        [_base_scenario("AAA", 0.12), _base_scenario("BBB", 0.08)],
        reviewed_tickers=["AAA"],
    )
    payload = state.to_dict()
    payload.update(
        {
            "status": "OPEN",
            "minimum_reviews_required": 2,
            "successful_review_count": 1,
            "attempted_tickers": ["AAA"],
            "failed_review_tickers": [],
        }
    )

    with pytest.raises(ValueError, match="not closed"):
        validate_closed_competitive_frontier_payload(payload)


def test_shuffled_pool_over_25_has_stable_metric_rank_and_top_25() -> None:
    packets = [_packet(f"C{index:02d}", float(index)) for index in range(30)]
    scenarios = [_base_scenario(f"C{index:02d}", 0.05 + index / 1000) for index in range(30)]
    shuffled_packets = packets[::2] + packets[1::2]
    shuffled_scenarios = scenarios[1::2] + scenarios[::2]

    ordered = build_competitive_frontier(packets, scenarios)
    shuffled = build_competitive_frontier(shuffled_packets, shuffled_scenarios)

    expected_full_rank = [
        "C29",
        "C28",
        "C27",
        "C26",
        "C25",
        "C24",
        "C23",
        "C22",
        "C21",
        "C20",
        "C19",
        "C18",
        "C17",
        "C16",
        "C15",
        "C14",
        "C13",
        "C12",
        "C11",
        "C10",
        "C09",
        "C08",
        "C07",
        "C06",
        "C05",
        "C04",
        "C03",
        "C02",
        "C01",
        "C00",
    ]
    expected_top_25 = [
        "C29",
        "C28",
        "C27",
        "C26",
        "C25",
        "C24",
        "C23",
        "C22",
        "C21",
        "C20",
        "C19",
        "C18",
        "C17",
        "C16",
        "C15",
        "C14",
        "C13",
        "C12",
        "C11",
        "C10",
        "C09",
        "C08",
        "C07",
        "C06",
        "C05",
    ]
    assert [candidate.ticker for candidate in ordered.candidates] == expected_full_rank
    assert [candidate.ticker for candidate in shuffled.candidates] == expected_full_rank
    assert list(ordered.top_tickers) == expected_top_25
    assert list(shuffled.top_tickers) == expected_top_25
    assert ordered.frontier_tickers == ("C29",)
    assert ordered.dominated_tickers == (
        "C28",
        "C27",
        "C26",
        "C25",
        "C24",
        "C23",
        "C22",
        "C21",
        "C20",
        "C19",
        "C18",
        "C17",
        "C16",
        "C15",
        "C14",
        "C13",
        "C12",
        "C11",
        "C10",
        "C09",
        "C08",
        "C07",
        "C06",
        "C05",
        "C04",
        "C03",
        "C02",
        "C01",
        "C00",
    )


def test_pareto_frontier_separates_nondominated_and_dominated_candidates() -> None:
    packets = [
        _packet("AAA", 9.0),
        _packet("BBB", 8.0),
        _packet("CCC", 7.0),
        _packet("DDD", 6.0),
    ]
    scenarios = [
        _base_scenario("AAA", 0.10),
        _base_scenario("BBB", 0.09),
        _base_scenario("CCC", 0.15),
        _base_scenario("DDD", 0.08),
    ]

    state = build_competitive_frontier(packets, scenarios)

    assert [candidate.ticker for candidate in state.candidates] == [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
    ]
    assert state.frontier_tickers == ("AAA", "CCC")
    assert state.pending_tickers == ("AAA", "CCC")
    assert state.dominated_tickers == ("BBB", "DDD")
    assert state.dominators_by_ticker == {
        "BBB": ("AAA",),
        "DDD": ("AAA", "BBB", "CCC"),
    }
    assert state.next_batch == ("AAA", "CCC")
    assert state.closure_certificate.to_dict() == {
        "status": "OPEN",
        "closed": False,
        "frontier_tickers": ["AAA", "CCC"],
        "reviewed_tickers": [],
        "pending_tickers": ["AAA", "CCC"],
        "dominated_tickers": ["BBB", "DDD"],
        "unresolved_tickers": [],
        "reason_codes": ["UNREVIEWED_NONDOMINATED_CANDIDATES"],
    }


def test_missing_and_unreliable_return_remain_explicit_and_block_closure() -> None:
    packets = [
        _packet("AAA", 10.0),
        _packet("BBB", 9.0),
        _packet("CCC", 8.0),
    ]
    scenarios = [
        _base_scenario(
            "BBB",
            0.20,
            unsupported_assumptions=["Terminal multiple has no evidence."],
        ),
        _base_scenario("CCC", 0.12),
    ]

    state = build_competitive_frontier(packets, scenarios)

    assert [candidate.ticker for candidate in state.candidates] == ["CCC", "AAA", "BBB"]
    rows = {candidate.ticker: candidate for candidate in state.candidates}
    assert rows["AAA"].base_return_reliability == "MISSING"
    assert rows["AAA"].base_return_reason_codes == ("BASE_SCENARIO_MISSING",)
    assert rows["BBB"].base_annualized_return == 0.20
    assert rows["BBB"].base_return_reliability == "UNRELIABLE"
    assert rows["BBB"].base_return_reason_codes == ("UNSUPPORTED_BASE_RETURN_ASSUMPTIONS",)
    assert rows["CCC"].base_return_reliability == "RELIABLE"
    assert state.frontier_tickers == ("CCC",)
    assert state.unresolved_tickers == ("AAA", "BBB")
    assert state.next_batch == ("CCC",)

    reviewed = state.mark_reviewed(["CCC"])

    assert reviewed.needs_continuation is False
    assert reviewed.next_batch == ()
    assert reviewed.closure_certificate.to_dict() == {
        "status": "OPEN",
        "closed": False,
        "frontier_tickers": ["CCC"],
        "reviewed_tickers": ["CCC"],
        "pending_tickers": [],
        "dominated_tickers": [],
        "unresolved_tickers": ["AAA", "BBB"],
        "reason_codes": ["UNRESOLVED_FRONTIER_METRICS"],
    }


def test_frontier_continues_in_batches_of_three_until_all_nondominated_are_reviewed() -> None:
    packets = [
        _packet("AAA", 7.0),
        _packet("BBB", 6.0),
        _packet("CCC", 5.0),
        _packet("DDD", 4.0),
        _packet("EEE", 3.0),
        _packet("FFF", 2.0),
        _packet("GGG", 1.0),
    ]
    scenarios = [
        _base_scenario("AAA", 0.01),
        _base_scenario("BBB", 0.02),
        _base_scenario("CCC", 0.03),
        _base_scenario("DDD", 0.04),
        _base_scenario("EEE", 0.05),
        _base_scenario("FFF", 0.06),
        _base_scenario("GGG", 0.07),
    ]

    initial = build_competitive_frontier(packets, scenarios)
    assert initial.frontier_tickers == ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
    assert initial.next_batch == ("AAA", "BBB", "CCC")
    assert initial.needs_continuation is True

    after_first = initial.mark_reviewed(["CCC", "AAA", "BBB"])
    assert after_first.reviewed_tickers == ("AAA", "BBB", "CCC")
    assert after_first.pending_tickers == ("DDD", "EEE", "FFF", "GGG")
    assert after_first.next_batch == ("DDD", "EEE", "FFF")

    after_second = after_first.mark_reviewed(["DDD", "EEE", "FFF"])
    assert after_second.pending_tickers == ("GGG",)
    assert after_second.next_batch == ("GGG",)
    assert after_second.closure_certificate.closed is False

    completed = after_second.mark_reviewed(["GGG"])
    assert completed.reviewed_tickers == ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
    assert completed.pending_tickers == ()
    assert completed.next_batch == ()
    assert completed.needs_continuation is False
    assert completed.closure_certificate.to_dict() == {
        "status": "CLOSED",
        "closed": True,
        "frontier_tickers": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        "reviewed_tickers": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        "pending_tickers": [],
        "dominated_tickers": [],
        "unresolved_tickers": [],
        "reason_codes": [],
    }


def test_packet_factor_fallback_is_input_order_independent() -> None:
    packets = [
        _packet(
            "LOW",
            None,
            valuation={"discount_to_anchor": -0.10, "implied_growth": 0.10},
            returns_on_capital={"roic": 0.02},
        ),
        _packet(
            "HIGH",
            None,
            valuation={"discount_to_anchor": 0.50, "implied_growth": 0.02},
            returns_on_capital={"roic": 0.20},
        ),
        _packet(
            "MID",
            None,
            valuation={"discount_to_anchor": 0.20, "implied_growth": 0.05},
            returns_on_capital={"roic": 0.10},
        ),
    ]
    scenarios = [
        _base_scenario("LOW", 0.08),
        _base_scenario("HIGH", 0.14),
        _base_scenario("MID", 0.11),
    ]

    forward = build_competitive_frontier(packets, scenarios)
    reversed_state = build_competitive_frontier(list(reversed(packets)), scenarios)

    assert [candidate.ticker for candidate in forward.candidates] == ["HIGH", "MID", "LOW"]
    assert [candidate.ticker for candidate in reversed_state.candidates] == [
        "HIGH",
        "MID",
        "LOW",
    ]
    assert [candidate.deterministic_score_source for candidate in forward.candidates] == [
        "CROSS_SECTIONAL_FACTORS",
        "CROSS_SECTIONAL_FACTORS",
        "CROSS_SECTIONAL_FACTORS",
    ]
    assert forward.frontier_tickers == ("HIGH",)


def test_frontier_state_round_trip_preserves_cursor_and_derived_certificate() -> None:
    state = build_competitive_frontier(
        [_packet("AAA", 3.0), _packet("BBB", 2.0), _packet("CCC", 1.0)],
        [
            _base_scenario("AAA", 0.08),
            _base_scenario("BBB", 0.10),
            _base_scenario("CCC", 0.12),
        ],
        reviewed_tickers=["AAA"],
        top_n=2,
        batch_size=3,
    )

    payload = state.to_dict()
    restored = CompetitiveFrontierState.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.top_tickers == ("AAA", "BBB")
    assert restored.reviewed_tickers == ("AAA",)
    assert restored.next_batch == ("BBB", "CCC")
    assert restored.closure_certificate.closed is False
