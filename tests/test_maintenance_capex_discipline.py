from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.escalation import build_escalation_plan
from app.universe.memo_pack import _memo_markdown
from app.universe.promotion import LANE_4_DEPRIORITIZED, build_promotion_state
from app.universe.ranking import ranking_sort_key
from app.valuation.intrinsic_discipline import compute_intrinsic_discipline
from app.valuation.maintenance_capex_discipline import (
    ASSET_INTENSITY_UNKNOWN,
    HIGH_ASSET_INTENSITY,
    HIGH_MAINTENANCE_CAPEX_CREDIBILITY,
    LOW_ASSET_INTENSITY,
    LOW_MAINTENANCE_CAPEX_CREDIBILITY,
    MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN,
    MODERATE_ASSET_INTENSITY,
    MODERATE_MAINTENANCE_CAPEX_CREDIBILITY,
    compute_maintenance_capex_discipline,
    write_maintenance_capex_discipline_for_run,
)
from app.valuation.owner_earnings_quality import compute_owner_earnings_quality
from app.valuation.valuation_confidence import compute_valuation_confidence


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "sample_universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker\nAAA\nBBB\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _fundamentals_rows(
    *,
    revenue: list[float] | None = None,
    cfo: list[float] | None = None,
    capex: list[float] | None = None,
    owner_earnings: list[float] | None = None,
    depreciation: list[float] | None = None,
    shares: float = 10.0,
    net_debt: float = 0.0,
) -> dict:
    years = [2021, 2022, 2023, 2024, 2025]
    revenue = revenue or [100.0, 105.0, 110.0, 116.0, 122.0]
    cfo = cfo or [20.0, 21.0, 22.0, 23.0, 24.0]
    capex = capex or [3.0, 3.0, 3.2, 3.2, 3.4]
    owner_earnings = owner_earnings or [17.0, 18.0, 18.5, 19.5, 20.0]
    depreciation = depreciation or [3.0, 3.0, 3.1, 3.2, 3.3]
    rows = []
    for idx, year in enumerate(years):
        rows.append(
            {
                "year": year,
                "revenue": revenue[idx],
                "cfo": cfo[idx],
                "capex": capex[idx],
                "fcf": cfo[idx] - capex[idx],
                "owner_earnings": owner_earnings[idx],
                "depreciation": depreciation[idx],
                "maintenance_capex_proxy": capex[idx] * 0.60,
                "shares_outstanding": shares,
                "net_debt": net_debt,
            }
        )
    return {"rows": rows, "derived_from": ["maintenance.fixture"]}


def _intrinsic_payload(*, mos_to_floor: float | str, mos_classification: str, owner_selected: bool = False) -> dict:
    reason_codes = ["OWNER_EARNINGS_SELECTED"] if owner_selected else ["FCF_SELECTED"]
    method_used = "OWNER_EARNINGS_SELECTED" if owner_selected else "FCF_SELECTED"
    return {
        "mos_to_floor": mos_to_floor,
        "mos_to_base": 0.30 if isinstance(mos_to_floor, (int, float)) else "UNKNOWN",
        "mos_classification": mos_classification,
        "downside_support_type": "EARNINGS_POWER_SUPPORT",
        "normalized_earnings_power_status": "OK",
        "normalized_earnings_power_method_used": method_used,
        "normalized_earnings_power_reason_codes": reason_codes,
        "derived_from": ["intrinsic.fixture"],
        "claims": {
            "mos_to_floor": {"value": mos_to_floor, "derived_from": ["intrinsic.fixture.floor"]},
            "normalized_earnings_power_value": {"value": 85.0, "derived_from": ["intrinsic.fixture.power"]},
        },
    }


def test_low_asset_intensity_and_high_maintenance_capex_credibility():
    payload = compute_maintenance_capex_discipline(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    assert payload["asset_intensity_class"] == LOW_ASSET_INTENSITY
    assert payload["maintenance_capex_credibility_class"] == HIGH_MAINTENANCE_CAPEX_CREDIBILITY
    assert payload["primary_maintenance_capex_caution"] == "OWNER_EARNINGS_SUPPORTIVE"


def test_moderate_maintenance_capex_profile_for_mixed_burden_case():
    payload = compute_maintenance_capex_discipline(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(
            revenue=[100, 102, 104, 106, 108],
            cfo=[20, 20, 21, 21, 22],
            capex=[7, 8, 7, 8, 8],
            owner_earnings=[13, 12, 14, 13, 14],
            depreciation=[5, 6, 5, 6, 6],
        ),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    assert payload["asset_intensity_class"] == MODERATE_ASSET_INTENSITY
    assert payload["maintenance_capex_credibility_class"] == MODERATE_MAINTENANCE_CAPEX_CREDIBILITY
    assert payload["primary_maintenance_capex_caution"] == "OWNER_EARNINGS_MIXED"


def test_high_asset_intensity_and_low_maintenance_capex_credibility():
    payload = compute_maintenance_capex_discipline(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(
            revenue=[100, 102, 104, 106, 108],
            cfo=[18, 18, 19, 19, 20],
            capex=[18, 20, 19, 21, 22],
            owner_earnings=[8, 7, 7, 6, 6],
            depreciation=[10, 11, 10, 11, 12],
        ),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    assert payload["asset_intensity_class"] == HIGH_ASSET_INTENSITY
    assert payload["maintenance_capex_credibility_class"] == LOW_MAINTENANCE_CAPEX_CREDIBILITY
    assert "LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND" in payload["maintenance_capex_credibility_reason_codes"]


def test_unknown_maintenance_capex_behavior_when_evidence_thin():
    payload = compute_maintenance_capex_discipline(
        "AAA",
        "2026-03-08",
        fundamentals={"rows": [{"year": 2025, "revenue": 100.0}], "derived_from": ["thin.fixture"]},
        facts_status="UNKNOWN",
        shares_status="UNKNOWN",
        price_status="UNKNOWN",
    )
    assert payload["asset_intensity_class"] == ASSET_INTENSITY_UNKNOWN
    assert payload["maintenance_capex_credibility_class"] == MAINTENANCE_CAPEX_CREDIBILITY_UNKNOWN
    assert "MAINTENANCE_CAPEX_DISCIPLINE_UNKNOWN" in payload["maintenance_capex_credibility_reason_codes"]


def test_owner_earnings_quality_and_valuation_confidence_apply_maintenance_headwind_honestly():
    maintenance_payload = {
        "asset_intensity_class": HIGH_ASSET_INTENSITY,
        "asset_intensity_reason_codes": ["HIGH_ASSET_INTENSITY_HEADWIND"],
        "maintenance_capex_credibility_class": LOW_MAINTENANCE_CAPEX_CREDIBILITY,
        "maintenance_capex_credibility_reason_codes": [
            "LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND",
            "OWNER_EARNINGS_DENOMINATOR_SENSITIVE",
        ],
        "maintenance_capex_headwind_signals": [
            "OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING",
            "HEAVY_ASSET_REPLACEMENT_BURDEN",
        ],
        "maintenance_capex_support_signals": [],
        "primary_maintenance_capex_caution": "OWNER_EARNINGS_HEADWIND",
        "maintenance_capex_discipline_summary": "capital burden makes owner earnings less trustworthy",
        "derived_from": ["maintenance.fixture"],
    }
    owner_quality = compute_owner_earnings_quality(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(),
        maintenance_capex_payload=maintenance_payload,
    )
    assert "LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND" in owner_quality["oe_quality_reason_codes"]
    assert "OWNER_EARNINGS_DENOMINATOR_SENSITIVE" in owner_quality["oe_quality_reason_codes"]

    confidence = compute_valuation_confidence(
        "AAA",
        "2026-03-08",
        intrinsic_payload=_intrinsic_payload(
            mos_to_floor=0.35,
            mos_classification="ADEQUATE_MARGIN_OF_SAFETY",
            owner_selected=True,
        ),
        maintenance_capex_payload=maintenance_payload,
        price_status="OK",
        shares_status="OK",
        fcf_status="OK",
        facts_status="OK",
        valuation_status="OK",
        epv_per_share=80.0,
        epv_refs=["epv.fixture"],
        netnet_per_share=60.0,
        netnet_refs=["netnet.fixture"],
        existing_intrinsic_base=90.0,
        existing_intrinsic_base_refs=["intrinsic.fixture.base"],
        existing_intrinsic_conservative=70.0,
        existing_intrinsic_conservative_refs=["intrinsic.fixture.conservative"],
    )
    assert confidence["valuation_confidence_class"] == "LOW_CONFIDENCE"
    assert "LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND" in confidence["valuation_confidence_reason_codes"]


def test_intrinsic_discipline_surfaces_owner_earnings_sensitivity():
    payload = compute_intrinsic_discipline(
        "AAA",
        "2026-03-08",
        fundamentals={
            "rows": [
                {"year": 2023, "shares_outstanding": 10.0, "net_debt": 0.0},
                {"year": 2024, "shares_outstanding": 10.0, "net_debt": 0.0},
                {"year": 2025, "shares_outstanding": 10.0, "net_debt": 0.0},
            ],
            "derived_from": ["fundamentals.owner_only"],
        },
        owner_payload={
            "summary": {
                "owner_earnings_normalized_3y": 100.0,
                "owner_earnings_normalized_method": "MEDIAN_POSITIVE_3Y",
                "owner_earnings_points": 3,
            },
            "series": [
                {"year": 2023, "owner_earnings": 95.0, "derived_from": ["owner.2023"]},
                {"year": 2024, "owner_earnings": 100.0, "derived_from": ["owner.2024"]},
                {"year": 2025, "owner_earnings": 105.0, "derived_from": ["owner.2025"]},
            ],
            "derived_from": ["owner.fixture"],
        },
        owner_quality_payload={"owner_earnings_stability_score": 4.0, "oe_quality_total": 8.0},
        intangible_payload={"cycle_resilience_score": 3.0},
        maintenance_capex_payload={
            "asset_intensity_class": HIGH_ASSET_INTENSITY,
            "maintenance_capex_credibility_class": LOW_MAINTENANCE_CAPEX_CREDIBILITY,
            "maintenance_capex_credibility_reason_codes": ["LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND"],
            "maintenance_capex_headwind_signals": ["OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING"],
            "derived_from": ["maintenance.fixture"],
        },
        price_value=50.0,
        price_refs=["price.AAA"],
    )
    assert payload["normalized_earnings_power_method_used"] == "OWNER_EARNINGS_SELECTED"
    assert payload["normalized_earnings_power_status"] == "LOW_CONFIDENCE"
    assert "OWNER_EARNINGS_DENOMINATOR_SENSITIVE" in payload["normalized_earnings_power_reason_codes"]


def test_memo_pack_includes_maintenance_capex_section():
    memo = _memo_markdown(
        {
            "header": {"ticker": "AAA", "as_of_date": "2026-03-08"},
            "maintenance_capex_asset_intensity_discipline": {
                "asset_intensity_class": LOW_ASSET_INTENSITY,
                "asset_intensity_reason_codes": ["LOW_ASSET_INTENSITY_SUPPORT"],
                "maintenance_capex_credibility_class": HIGH_MAINTENANCE_CAPEX_CREDIBILITY,
                "maintenance_capex_credibility_reason_codes": ["CAPEX_PROXY_APPEARS_REASONABLE"],
                "maintenance_capex_support_signals": ["CAPEX_LIGHT_SCALING_PRESENT"],
                "maintenance_capex_headwind_signals": [],
                "primary_maintenance_capex_caution": "OWNER_EARNINGS_SUPPORTIVE",
                "maintenance_capex_discipline_summary": "owner earnings appear reasonably supported",
                "derived_from": ["maintenance.fixture"],
            },
        }
    )
    assert "## Maintenance Capex / Asset Intensity Discipline" in memo
    assert HIGH_MAINTENANCE_CAPEX_CREDIBILITY in memo


def test_promotion_and_escalation_surface_maintenance_without_overriding_fail():
    master_shortlist = {
        "campaign_run_id": "camp_maintenance",
        "rows": [
            {
                "ticker": "AAA",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_maintenance__core"}],
                "best_rank_seen": 1,
                "value_gate_status": "WATCH",
                "latest_value_gate_status": "WATCH",
                "implied_return_base": 0.18,
                "primary_blocker": "MISSING_FCF",
                "latest_primary_blocker": "MISSING_FCF",
                "asset_intensity_class": LOW_ASSET_INTENSITY,
                "asset_intensity_reason_codes": ["LOW_ASSET_INTENSITY_SUPPORT"],
                "maintenance_capex_credibility_class": HIGH_MAINTENANCE_CAPEX_CREDIBILITY,
                "maintenance_capex_credibility_reason_codes": ["CAPEX_PROXY_APPEARS_REASONABLE"],
                "maintenance_capex_support_signals": ["CAPEX_LIGHT_SCALING_PRESENT"],
                "maintenance_capex_headwind_signals": [],
                "primary_maintenance_capex_caution": "OWNER_EARNINGS_SUPPORTIVE",
                "maintenance_capex_discipline_summary": "supportive",
                "memo_path": "memo/AAA.md",
            },
            {
                "ticker": "BBB",
                "source_runs": [{"campaign_item": "core", "universe_run_id": "camp_maintenance__core"}],
                "best_rank_seen": 2,
                "value_gate_status": "FAIL",
                "latest_value_gate_status": "FAIL",
                "implied_return_base": 0.01,
                "primary_blocker": "PRICE_UNKNOWN",
                "latest_primary_blocker": "PRICE_UNKNOWN",
                "asset_intensity_class": HIGH_ASSET_INTENSITY,
                "asset_intensity_reason_codes": ["HIGH_ASSET_INTENSITY_HEADWIND"],
                "maintenance_capex_credibility_class": LOW_MAINTENANCE_CAPEX_CREDIBILITY,
                "maintenance_capex_credibility_reason_codes": ["LOW_MAINTENANCE_CAPEX_CREDIBILITY_HEADWIND"],
                "maintenance_capex_support_signals": [],
                "maintenance_capex_headwind_signals": ["OWNER_EARNINGS_DENOMINATOR_TOO_FLATTERING"],
                "primary_maintenance_capex_caution": "OWNER_EARNINGS_HEADWIND",
                "maintenance_capex_discipline_summary": "headwind",
                "memo_path": "memo/BBB.md",
            },
        ],
    }
    master_watchlist_state = {
        "campaign_run_id": "camp_maintenance",
        "tickers": {
            "AAA": {"appearances_count": 2, "latest_value_gate_status": "WATCH", "latest_primary_blocker": "MISSING_FCF", "history": []},
            "BBB": {"appearances_count": 2, "latest_value_gate_status": "FAIL", "latest_primary_blocker": "PRICE_UNKNOWN", "history": []},
        },
    }
    promotion_state = build_promotion_state("camp_maintenance", master_watchlist_state, master_shortlist)
    row_bbb = next(row for row in promotion_state["rows"] if row["ticker"] == "BBB")
    assert row_bbb["maintenance_capex_credibility_class"] == LOW_MAINTENANCE_CAPEX_CREDIBILITY
    assert row_bbb["priority_lane"] == LANE_4_DEPRIORITIZED

    lanes = {
        "lane_1_high_priority": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_1_HIGH_PRIORITY"],
        "lane_2_research_queue": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_2_RESEARCH_QUEUE"],
        "lane_3_monitor": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_3_MONITOR"],
        "lane_4_deprioritized": [row for row in promotion_state["rows"] if row["priority_lane"] == "LANE_4_DEPRIORITIZED"],
    }
    escalation = build_escalation_plan("camp_maintenance", promotion_state, lanes)
    assert any("maintenance_capex_credibility_class" in entry for entry in escalation["queue"])


def test_maintenance_capex_cli_open_and_owner_earnings_hardness_sort(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    detail_low = compute_maintenance_capex_discipline(
        "AAA",
        "2026-03-08",
        fundamentals=_fundamentals_rows(),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    detail_high = compute_maintenance_capex_discipline(
        "BBB",
        "2026-03-08",
        fundamentals=_fundamentals_rows(
            revenue=[100, 102, 104, 106, 108],
            cfo=[18, 18, 19, 19, 20],
            capex=[18, 20, 19, 21, 22],
            owner_earnings=[8, 7, 7, 6, 6],
            depreciation=[10, 11, 10, 11, 12],
        ),
        facts_status="OK",
        shares_status="OK",
        price_status="OK",
    )
    run_id = "maintenance_capex_test_open"
    output_path = cfg.outputs_dir / "universe" / run_id / "maintenance_capex_discipline.json"
    write_maintenance_capex_discipline_for_run(
        run_id=run_id,
        as_of_date="2026-03-08",
        tickers=["AAA", "BBB"],
        output_path=output_path,
        scoreboard_rows=[
            {"ticker": "AAA", "maintenance_capex_discipline_detail": detail_low},
            {"ticker": "BBB", "maintenance_capex_discipline_detail": detail_high},
        ],
    )

    result = runner.invoke(app, ["universe-maintenance-capex-open", "--run-id", run_id])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "OK"
    assert payload["counts_by_maintenance_capex_credibility_class"][HIGH_MAINTENANCE_CAPEX_CREDIBILITY] == 1
    assert payload["counts_by_maintenance_capex_credibility_class"][LOW_MAINTENANCE_CAPEX_CREDIBILITY] == 1

    rows = [
        {
            "ticker": "AAA",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "maintenance_capex_credibility_class": HIGH_MAINTENANCE_CAPEX_CREDIBILITY,
            "asset_intensity_class": LOW_ASSET_INTENSITY,
            "accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY",
            "reinvestment_efficiency_class": "MODERATE_REINVESTMENT_EFFICIENCY",
            "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION",
            "returns_persistence_class": "MODERATE_RETURNS_PERSISTENCE",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 7.0,
            "intangible_economics_total": 6.0,
        },
        {
            "ticker": "BBB",
            "scout_status": "WATCH",
            "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
            "mos_to_floor": 0.20,
            "valuation_confidence_class": "MEDIUM_CONFIDENCE",
            "valuation_integrity_class": "INTEGRITY_OK",
            "maintenance_capex_credibility_class": LOW_MAINTENANCE_CAPEX_CREDIBILITY,
            "asset_intensity_class": HIGH_ASSET_INTENSITY,
            "accounting_quality_class": "MODERATE_ACCOUNTING_QUALITY",
            "reinvestment_efficiency_class": "MODERATE_REINVESTMENT_EFFICIENCY",
            "capital_allocation_discipline_class": "MIXED_CAPITAL_ALLOCATION",
            "returns_persistence_class": "MODERATE_RETURNS_PERSISTENCE",
            "value_type_primary": "EARNINGS_POWER_VALUE",
            "normalization_credibility_class": "MODERATE_NORMALIZATION_CREDIBILITY",
            "oe_quality_total": 7.0,
            "intangible_economics_total": 6.0,
        },
    ]
    ordered = sorted(rows, key=lambda row: ranking_sort_key(row, policy="value_first_owner_earnings_hardness"))
    assert [row["ticker"] for row in ordered] == ["AAA", "BBB"]
