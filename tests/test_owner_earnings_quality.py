from __future__ import annotations

from pathlib import Path

from app.db import init_db
from app.universe.depth_rollup import build_global_shortlist
from app.universe.escalation import ACTION_CLEAR_BLOCKERS, build_escalation_plan
from app.universe.memo_pack import _memo_markdown, build_investment_memo
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, build_promotion_state
from app.valuation.owner_earnings_quality import compute_owner_earnings_quality, open_owner_earnings_quality, write_owner_earnings_quality_for_run


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\nBBB,2,BBB\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _fundamentals_payload(ticker: str, rows: list[dict[str, float | int | str]]) -> dict:
    trace_fields = ["revenue", "cfo", "capex", "fcf", "shares_outstanding", "net_debt"]
    row_traces: dict[str, dict[str, dict[str, list[str]]]] = {}
    for row in rows:
        year = int(row["year"])
        row_traces[str(year)] = {
            field: {"derived_from": [f"fundamentals.{ticker}.{year}.{field}"]}
            for field in trace_fields
            if field in row
        }
    return {
        "ticker": ticker,
        "run_id": f"{ticker.lower()}_run",
        "as_of_date": "2026-02-14",
        "rows": rows,
        "row_traces": row_traces,
        "derived_signals": {},
    }


def _value_row(
    ticker: str,
    *,
    oe_quality_total,
    quality_score=12.0,
    gate="WATCH",
    implied_return=0.20,
    mos_epv=0.20,
    owner_yield=0.05,
) -> dict:
    return {
        "ticker": ticker,
        "run_id": f"run_{ticker.lower()}",
        "sector": "Software",
        "as_of_date": "2026-02-14",
        "source_depth_runs": [{"run_id": f"run_{ticker.lower()}", "sector": "Software", "as_of_date": "2026-02-14"}],
        "value_gate_status": gate,
        "value_gate_reasons": [],
        "primary_blocker": "MISSING_EV",
        "implied_return_base": implied_return,
        "implied_return_base_derived_from": [f"value.{ticker}.implied_return_base"],
        "intrinsic_per_share_base": 100.0,
        "intrinsic_per_share_base_derived_from": [f"value.{ticker}.intrinsic_per_share_base"],
        "mos_epv": mos_epv,
        "mos_epv_derived_from": [f"value.{ticker}.mos_epv"],
        "mos_netnet": "UNKNOWN",
        "mos_netnet_derived_from": [f"value.{ticker}.mos_netnet"],
        "owner_earnings_yield_ev_3y": owner_yield,
        "owner_earnings_yield_ev_3y_derived_from": [f"value.{ticker}.owner_earnings_yield_ev_3y"],
        "fcf_yield_ev_3y": 0.04,
        "fcf_yield_ev_3y_derived_from": [f"value.{ticker}.fcf_yield_ev_3y"],
        "yield_metric_used": "owner_earnings_yield_ev_3y",
        "yield_denominator_used": "EV",
        "owner_earnings_stability_score": 4.0 if oe_quality_total != "UNKNOWN" else "UNKNOWN",
        "owner_earnings_stability_score_derived_from": [f"oe.{ticker}.owner_earnings_stability_score"],
        "capital_allocation_score": 4.0 if oe_quality_total != "UNKNOWN" else "UNKNOWN",
        "capital_allocation_score_derived_from": [f"oe.{ticker}.capital_allocation_score"],
        "cash_conversion_score": 3.0 if oe_quality_total != "UNKNOWN" else "UNKNOWN",
        "cash_conversion_score_derived_from": [f"oe.{ticker}.cash_conversion_score"],
        "oe_quality_total": oe_quality_total,
        "oe_quality_total_derived_from": [f"oe.{ticker}.oe_quality_total"],
        "oe_quality_reason_codes": ["SHAREHOLDER_FRIENDLY"] if oe_quality_total != "UNKNOWN" else ["CASH_CONVERSION_UNKNOWN"],
        "quality_score": quality_score,
        "risk_penalty": -2.0,
        "price_status": "OK",
        "valuation_status": "OK",
        "shares_status": "OK",
        "fcf_status": "OK",
        "facts_status": "OK",
        "price_reason_code": "OK",
        "valuation_reason_code": "OK",
        "shares_reason_code": "OK",
        "fcf_reason_code": "OK",
        "facts_reason_code": "OK",
        "derived_from": [f"shortlist.{ticker}"],
    }


def test_stability_scoring_is_deterministic_from_fundamentals():
    payload = compute_owner_earnings_quality(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_payload(
            "AAA",
            [
                {"year": 2021, "revenue": 100.0, "cfo": 30.0, "capex": 6.0, "fcf": 24.0, "shares_outstanding": 100.0, "net_debt": 40.0},
                {"year": 2022, "revenue": 105.0, "cfo": 32.0, "capex": 6.0, "fcf": 26.0, "shares_outstanding": 99.0, "net_debt": 35.0},
                {"year": 2023, "revenue": 110.0, "cfo": 34.0, "capex": 7.0, "fcf": 27.0, "shares_outstanding": 98.0, "net_debt": 32.0},
                {"year": 2024, "revenue": 116.0, "cfo": 36.0, "capex": 7.0, "fcf": 29.0, "shares_outstanding": 97.0, "net_debt": 29.0},
                {"year": 2025, "revenue": 121.0, "cfo": 38.0, "capex": 7.0, "fcf": 31.0, "shares_outstanding": 96.0, "net_debt": 25.0},
            ],
        ),
    )

    assert payload["owner_earnings_positive_years_5y"] == 5.0
    assert payload["owner_earnings_stability_score"] == 5.0
    assert payload["owner_earnings_volatility_5y"] != "UNKNOWN"


def test_dilution_capex_and_debt_scoring_is_deterministic():
    payload = compute_owner_earnings_quality(
        "BBB",
        "2026-02-14",
        fundamentals=_fundamentals_payload(
            "BBB",
            [
                {"year": 2021, "revenue": 100.0, "cfo": 20.0, "capex": 12.0, "fcf": 8.0, "shares_outstanding": 100.0, "net_debt": 20.0},
                {"year": 2022, "revenue": 100.0, "cfo": 20.0, "capex": 13.0, "fcf": 7.0, "shares_outstanding": 110.0, "net_debt": 24.0},
                {"year": 2023, "revenue": 100.0, "cfo": 20.0, "capex": 14.0, "fcf": 6.0, "shares_outstanding": 121.0, "net_debt": 28.0},
                {"year": 2024, "revenue": 100.0, "cfo": 20.0, "capex": 15.0, "fcf": 5.0, "shares_outstanding": 133.1, "net_debt": 32.0},
                {"year": 2025, "revenue": 100.0, "cfo": 20.0, "capex": 16.0, "fcf": 4.0, "shares_outstanding": 146.41, "net_debt": 36.0},
            ],
        ),
    )

    assert payload["capital_allocation_score"] == 0.0
    assert "EXCESS_DILUTION" in payload["capital_allocation_reason_codes"]
    assert "HIGH_CAPEX_BURDEN" in payload["capital_allocation_reason_codes"]
    assert "DEBT_ACCUMULATION" in payload["capital_allocation_reason_codes"]


def test_cash_conversion_scoring_uses_real_latest_values():
    payload = compute_owner_earnings_quality(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_payload(
            "AAA",
            [
                {"year": 2024, "revenue": 100.0, "cfo": 24.0, "capex": 4.0, "fcf": 18.0, "shares_outstanding": 100.0, "net_debt": 20.0},
                {"year": 2025, "revenue": 120.0, "cfo": 30.0, "capex": 5.0, "fcf": 24.0, "shares_outstanding": 99.0, "net_debt": 18.0},
            ],
        ),
    )

    assert payload["cfo_margin_proxy"] == 0.25
    assert payload["fcf_conversion_proxy"] == 0.8
    assert payload["cash_conversion_score"] == 5.0


def test_unknown_handling_and_reason_codes_are_explicit():
    payload = compute_owner_earnings_quality(
        "AAA",
        "2026-02-14",
        fundamentals=_fundamentals_payload(
            "AAA",
            [
                {"year": 2024, "revenue": "UNKNOWN", "cfo": 12.0, "capex": 6.0, "fcf": "UNKNOWN", "shares_outstanding": 100.0, "net_debt": 20.0},
                {"year": 2025, "revenue": "UNKNOWN", "cfo": 10.0, "capex": 7.0, "fcf": "UNKNOWN", "shares_outstanding": 101.0, "net_debt": 21.0},
            ],
        ),
    )

    assert payload["owner_earnings_stability_score"] == "UNKNOWN"
    assert "INSUFFICIENT_OWNER_EARNINGS_HISTORY" in payload["owner_earnings_stability_reason_codes"]
    assert payload["cash_conversion_score"] == "UNKNOWN"
    assert "MISSING_REVENUE" in payload["cash_conversion_reason_codes"]
    assert "MISSING_FCF" in payload["cash_conversion_reason_codes"]


def test_value_first_quality_shortlist_prefers_higher_oe_quality():
    payload = build_global_shortlist(
        [
            _value_row("AAA", oe_quality_total=11.0, quality_score=8.0),
            _value_row("BBB", oe_quality_total=3.0, quality_score=18.0),
        ],
        top_n=2,
        policy="value_first_quality",
    )

    assert [row["ticker"] for row in payload["rows_top_n"]] == ["AAA", "BBB"]


def test_promotion_and_escalation_use_high_oe_quality_support(monkeypatch, tmp_path):
    _cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.promotion._load_memory_lookup", lambda: {})
    monkeypatch.setattr("app.universe.escalation._load_memory_lookup", lambda: {})

    master_watchlist = {
        "campaign_run_id": "campaign_oe",
        "tickers": {
            "AAA": {
                "latest_value_gate_status": "WATCH",
                "latest_implied_return_base": "UNKNOWN",
                "latest_primary_blocker": "MISSING_EV",
                "appearances_count": 2,
                "history": [],
            }
        },
    }
    master_shortlist = {
        "campaign_run_id": "campaign_oe",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "item_a", "universe_run_id": "run_a", "batch_run_id": "run_a_depth_batch"}],
                "best_rank_seen": 2,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": "UNKNOWN",
                "mos_epv": "UNKNOWN",
                "mos_netnet": "UNKNOWN",
                "owner_earnings_yield_ev_3y": "UNKNOWN",
                "yield_metric_used": "UNKNOWN",
                "primary_blocker": "MISSING_EV",
                "latest_primary_blocker": "MISSING_EV",
                "owner_earnings_stability_score": 4.0,
                "capital_allocation_score": 4.0,
                "cash_conversion_score": 3.0,
                "oe_quality_total": 11.0,
                "oe_quality_reason_codes": ["SHAREHOLDER_FRIENDLY"],
                "composite_score_total": 5.0,
                "memo_path": "",
                "derived_from": [],
            }
        ],
    }

    promotion_payload = build_promotion_state("campaign_oe", master_watchlist, master_shortlist)
    promotion_row = promotion_payload["rows"][0]
    assert promotion_row["priority_lane"] == LANE_2_RESEARCH_QUEUE
    assert "HIGH_OE_QUALITY_SUPPORT" in promotion_row["promotion_reason_codes"]

    queue_payload = build_escalation_plan(
        "campaign_oe",
        promotion_payload,
        {"campaign_run_id": "campaign_oe", "lane_2_research_queue": promotion_payload["rows"]},
        config={"as_of_date": "2026-02-14", "source_campaign_file": "data/universe/sample_campaign.json", "top_n": 5, "policy": "value_first"},
    )
    entry = [row for row in queue_payload["queue"] if row["action_type"] == ACTION_CLEAR_BLOCKERS][0]
    assert "HIGH_OE_QUALITY_SUPPORT" in entry["priority_support_codes"]


def test_memo_pack_includes_owner_earnings_quality_section():
    memo = build_investment_memo(
        "AAA",
        universe_run_id="u_oe",
        batch_run_id="b_oe",
        sources={
            "shortlist_row": _value_row("AAA", oe_quality_total=11.0),
            "score_row": {
                "ticker": "AAA",
                "metric_values": {
                    "implied_return_base": 0.20,
                    "intrinsic_per_share_base": 120.0,
                    "current_price": 100.0,
                    "mos_epv": 0.20,
                    "mos_netnet": "UNKNOWN",
                    "epv_per_share": 130.0,
                    "netnet_per_share": "UNKNOWN",
                    "owner_earnings_yield_ev_3y": 0.05,
                    "fcf_yield_ev_3y": 0.04,
                    "owner_earnings_yield_3y": 0.05,
                    "fcf_yield_3y": 0.04,
                    "revenue_cagr_5y": 0.10,
                    "revenue_cagr_10y": 0.08,
                    "operating_margin_trend_slope": 0.01,
                    "gross_margin_trend_slope": 0.01,
                    "fcf_margin_trend_slope": 0.01,
                    "roic_proxy": 0.15,
                    "dilution_rate_shares_cagr": 0.00,
                    "net_debt_proxy": 20.0,
                    "risk_factor_keyword_delta": 0.0,
                    "quality_score": 12.0,
                    "risk_penalty": -1.0,
                    "owner_earnings_stability_score": 4.0,
                    "capital_allocation_score": 4.0,
                    "cash_conversion_score": 3.0,
                    "oe_quality_total": 11.0,
                },
                "metric_traces": {
                    key: {"derived_from": [f"trace.AAA.{key}"]}
                    for key in [
                        "implied_return_base",
                        "intrinsic_per_share_base",
                        "current_price",
                        "mos_epv",
                        "mos_netnet",
                        "epv_per_share",
                        "netnet_per_share",
                        "owner_earnings_yield_ev_3y",
                        "fcf_yield_ev_3y",
                        "owner_earnings_yield_3y",
                        "fcf_yield_3y",
                        "revenue_cagr_5y",
                        "revenue_cagr_10y",
                        "operating_margin_trend_slope",
                        "gross_margin_trend_slope",
                        "fcf_margin_trend_slope",
                        "roic_proxy",
                        "dilution_rate_shares_cagr",
                        "net_debt_proxy",
                        "risk_factor_keyword_delta",
                        "quality_score",
                        "risk_penalty",
                        "owner_earnings_stability_score",
                        "capital_allocation_score",
                        "cash_conversion_score",
                        "oe_quality_total",
                    ]
                },
            },
            "gate_row": {
                "ticker": "AAA",
                "gate_status": "WATCH",
                "gate_reasons": [],
                "primary_blocker": "MISSING_EV",
                "inputs_used": {
                    "current_price": {"value": 100.0, "derived_from": ["price.AAA"]},
                    "net_debt_proxy": {"value": 20.0, "derived_from": ["netdebt.AAA"]},
                    "dilution_rate_shares_cagr": {"value": 0.0, "derived_from": ["dilution.AAA"]},
                },
                "net_debt_to_cfo": 1.0,
            },
            "valuation_row": {
                "ticker": "AAA",
                "price_status": "OK",
                "valuation_status": "OK",
                "price_reason_code": "OK",
                "valuation_reason_code": "OK",
                "current_price": 100.0,
                "implied_return_base": 0.20,
                "intrinsic_per_share_base": 120.0,
                "derived_from": ["valuation.AAA"],
            },
            "shares_row": {"ticker": "AAA", "shares_status": "OK", "shares_reason_code": "OK", "derived_from": ["shares.AAA"]},
            "fcf_row": {"ticker": "AAA", "fcf_status": "OK", "fcf_reason_code": "OK", "derived_from": ["fcf.AAA"]},
            "facts_row": {"ticker": "AAA", "status": "OK", "fetch_reason_code": "OK", "derived_from": ["facts.AAA"]},
        },
    )

    assert memo["owner_earnings_quality"]["oe_quality_total"] == 11.0
    markdown = _memo_markdown(memo)
    assert "## Owner Earnings Quality / Capital Allocation" in markdown
    assert "- oe_quality_total: `11.0`" in markdown


def test_write_and_open_owner_earnings_quality_artifact(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "oe_quality_run"
    output_path = cfg.outputs_dir / "universe" / run_id / "owner_earnings_quality.json"
    payload = write_owner_earnings_quality_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA"],
        output_path=output_path,
        scoreboard_rows=[
            {
                "ticker": "AAA",
                "owner_earnings_quality_detail": compute_owner_earnings_quality(
                    "AAA",
                    "2026-02-14",
                    fundamentals=_fundamentals_payload(
                        "AAA",
                        [
                            {"year": 2024, "revenue": 100.0, "cfo": 24.0, "capex": 4.0, "fcf": 18.0, "shares_outstanding": 100.0, "net_debt": 20.0},
                            {"year": 2025, "revenue": 120.0, "cfo": 30.0, "capex": 5.0, "fcf": 24.0, "shares_outstanding": 99.0, "net_debt": 18.0},
                        ],
                    ),
                ),
            }
        ],
    )

    assert Path(payload["owner_earnings_quality_path"]).exists()
    opened = open_owner_earnings_quality(run_id=run_id)
    assert opened["status"] == "OK"
    assert opened["known_count"] == 1


def _ko_shaped_rows(shares_by_year: dict[int, float]) -> list[dict[str, float | int | str]]:
    return [
        {
            "year": year,
            "revenue": 100.0,
            "cfo": 30.0,
            "capex": 6.0,
            "fcf": 24.0,
            "shares_outstanding": shares,
            "net_debt": 40.0,
        }
        for year, shares in sorted(shares_by_year.items())
    ]


# A 2-for-1 split in 2012, then a count that falls every year: 2,260 -> 4,520
# at the split, then down 10 a year to 4,400 in 2024. Measured first-to-last
# over the whole history that is (4,400 / 2,260) ** (1/13) - 1 = 5.26% a year
# of "dilution".
KO_SHAPED_SHARES = {2011: 2260.0, 2012: 4520.0, **{2012 + i: 4520.0 - 10.0 * i for i in range(1, 13)}}


def test_split_in_the_filed_history_is_not_read_as_dilution():
    """Fixed 2026-09-29. The KO shape: its 2012 2-for-1 split read as about 4% a
    year of dilution while its count fell. Dilution is now measured over the
    newest four fiscal years (the pre-valuation gate's window): 4,430 in 2021
    to 4,400 in 2024 is (4,400 / 4,430) ** (1/3) - 1 = -0.2262% a year, which
    earns the +1.0 a stable-or-shrinking count gets (between -1% and +2%).
    The allocation score is 2.0 base + 1.0 dilution + 2.0 capex burden
    (6 / 30 = 0.20 <= 0.25) + 0.0 net debt (flat) = 5.0; under the split-blind
    figure the dilution term was 0.0 and the score 4.0. The quality total
    carries the same +1.0.
    """
    payload = compute_owner_earnings_quality(
        "KOS", "2026-02-14", fundamentals=_fundamentals_payload("KOS", _ko_shaped_rows(KO_SHAPED_SHARES))
    )

    assert KO_SHAPED_SHARES[2021] == 4430.0 and KO_SHAPED_SHARES[2024] == 4400.0
    assert payload["dilution_rate_shares_cagr"] == (4400.0 / 4430.0) ** (1.0 / 3.0) - 1.0
    assert round(payload["dilution_rate_shares_cagr"], 6) == -0.002262
    assert payload["dilution_share_count_breaks"] == []
    assert payload["capital_allocation_score"] == 5.0
    assert payload["oe_quality_total"] == (
        payload["owner_earnings_stability_score"] + 5.0 + payload["cash_conversion_score"]
    )


def test_split_inside_the_recent_window_without_a_filed_split_is_unknown():
    """Migrated 2026-09-29 (review H5). A doubling between 2023 and 2024 used to be
    taken for a 2-for-1 split and the rate measured after it (4,400 -> 4,356 =
    -1.0%). Nothing filed says it was a split (the fundamentals path carries no
    split-ratio facts), and a doubling by issuance looks the same, so the rate is
    UNKNOWN with the break named, not a guess in either direction."""
    shares = {2022: 2220.0, 2023: 2210.0, 2024: 4400.0, 2025: 4356.0}
    payload = compute_owner_earnings_quality(
        "SPL", "2026-02-14", fundamentals=_fundamentals_payload("SPL", _ko_shaped_rows(shares))
    )

    assert payload["dilution_share_count_breaks"] == [2024]
    assert payload["dilution_share_count_splits"] == []
    assert payload["dilution_rate_shares_cagr"] == "UNKNOWN"
    assert payload["dilution_rate_reason_code"] == "SHARE_COUNT_BREAK_UNCORROBORATED"
    assert "EXCESS_DILUTION" not in payload["capital_allocation_reason_codes"]


def test_a_fifty_percent_raise_is_not_a_split():
    """Review H5, 2026-09-29. 100 -> 100 -> 150 -> 152 is a 50% equity raise: the
    1.5x move sits exactly on the 3-for-2 split factor. The old rule called it a
    split and measured only 150 -> 152 (+1.33% a year), which scored as a
    stable count (+1.0). With no filed split to corroborate it the rate is
    UNKNOWN, and the allocation score takes no dilution term: 2.0 base + 2.0
    capex burden = 4.0."""
    shares = {2022: 100.0, 2023: 100.0, 2024: 150.0, 2025: 152.0}
    payload = compute_owner_earnings_quality(
        "RAI", "2026-02-14", fundamentals=_fundamentals_payload("RAI", _ko_shaped_rows(shares))
    )

    assert payload["dilution_rate_shares_cagr"] == "UNKNOWN"
    assert payload["dilution_rate_reason_code"] == "SHARE_COUNT_BREAK_UNCORROBORATED"
    assert payload["dilution_share_count_breaks"] == [2024]
    assert payload["capital_allocation_score"] == 4.0


def test_recent_dilution_rate_split_adjusts_a_filed_split():
    """A filed 3-for-2 ratio in 2024 corroborates the 1.5x move: the earlier
    counts are restated on the new basis (100 -> 150) and the rate runs across
    the whole window, 150 -> 152 over three years."""
    from app.valuation.owner_earnings_quality import _recent_dilution_rate

    rows = [
        {"year": year, "value": value, "derived_from": [f"s.{year}"]}
        for year, value in ((2022, 100.0), (2023, 100.0), (2024, 150.0), (2025, 152.0))
    ]
    split = [{"year": 2024, "value": 1.5, "derived_from": ["split.2024"]}]

    rate, refs, breaks, splits = _recent_dilution_rate(rows, split)
    assert rate == (152.0 / 150.0) ** (1.0 / 3.0) - 1.0
    assert breaks == [] and splits == [2024]
    assert refs == ["s.2022", "s.2025", "split.2024"]

    # A ratio filed for a different split (2-for-1) does not corroborate 1.5x.
    rate, _refs, breaks, splits = _recent_dilution_rate(
        rows, [{"year": 2024, "value": 2.0, "derived_from": []}]
    )
    assert (rate, breaks, splits) == ("UNKNOWN", [2024], [])


def test_companyfacts_path_reads_the_filed_split_ratio(monkeypatch, tmp_path):
    """The companyfacts path hands the issuer's filed split ratio
    (StockholdersEquityNoteStockSplitConversionRatio1) to the dilution rate."""
    import json

    cfg = _init_cfg(monkeypatch, tmp_path)

    def fact(val, end, fy):
        return {"val": val, "end": end, "filed": f"{fy + 1}-02-15", "fy": fy, "fp": "FY",
                "form": "10-K", "accn": f"0000000000-{fy - 2000}-000001", "frame": f"CY{fy}Q4I"}

    counts = {2022: 100_000_000, 2023: 100_000_000, 2024: 150_000_000, 2025: 152_000_000}
    companyfacts = {
        "cik": 1,
        "facts": {
            "us-gaap": {
                "CommonStockSharesOutstanding": {
                    "units": {"shares": [fact(v, f"{y}-12-31", y) for y, v in counts.items()]}
                },
            },
        },
    }
    path = tmp_path / "cf.json"
    path.write_text(json.dumps(companyfacts), encoding="utf-8")
    owner = {"summary": {}, "series": [], "derived_from": []}

    unfiled = compute_owner_earnings_quality(
        "CFS", "2026-06-30", facts_row={"cache_path": str(path)}, owner_payload=owner, cfg=cfg
    )
    assert unfiled["dilution_rate_shares_cagr"] == "UNKNOWN"
    assert unfiled["dilution_share_count_breaks"] == [2024]

    companyfacts["facts"]["us-gaap"]["StockholdersEquityNoteStockSplitConversionRatio1"] = {
        "units": {"pure": [{**fact(1.5, "2024-12-31", 2024), "start": "2024-01-01"}]}
    }
    path.write_text(json.dumps(companyfacts), encoding="utf-8")
    filed = compute_owner_earnings_quality(
        "CFS", "2026-06-30", facts_row={"cache_path": str(path)}, owner_payload=owner, cfg=cfg
    )
    assert filed["dilution_share_count_splits"] == [2024]
    assert filed["dilution_share_count_breaks"] == []
    assert filed["dilution_rate_shares_cagr"] == (152.0 / 150.0) ** (1.0 / 3.0) - 1.0


def test_three_for_two_split_inside_the_gate_band_is_still_a_break():
    """A 3-for-2 split is a ratio of exactly 1.5, inside the [2/3, 3/2] band;
    it matches a clean split factor and is a break. With nothing after it in
    the window the rate is UNKNOWN, and the score takes no dilution term."""
    shares = {2022: 100.0, 2023: 100.0, 2024: 100.0, 2025: 150.0}
    payload = compute_owner_earnings_quality(
        "TFT", "2026-02-14", fundamentals=_fundamentals_payload("TFT", _ko_shaped_rows(shares))
    )

    assert payload["dilution_share_count_breaks"] == [2025]
    assert payload["dilution_rate_shares_cagr"] == "UNKNOWN"
    assert payload["dilution_rate_reason_code"] == "SHARE_COUNT_BREAK_UNCORROBORATED"
    assert payload["capital_allocation_score"] == 4.0  # 2.0 + capex burden 2.0, no dilution term


def test_real_dilution_below_a_split_factor_still_counts():
    """10% a year of issuance is not a split: 100 -> 110 -> 121 -> 133.1."""
    shares = {2022: 100.0, 2023: 110.0, 2024: 121.0, 2025: 133.1}
    payload = compute_owner_earnings_quality(
        "DIL", "2026-02-14", fundamentals=_fundamentals_payload("DIL", _ko_shaped_rows(shares))
    )

    assert payload["dilution_share_count_breaks"] == []
    assert round(payload["dilution_rate_shares_cagr"], 12) == 0.1
    assert "EXCESS_DILUTION" in payload["capital_allocation_reason_codes"]
