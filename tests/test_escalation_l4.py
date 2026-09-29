from __future__ import annotations

import json
from pathlib import Path

from app.db import init_db
from app.universe.escalation import (
    ACTION_CLEAR_BLOCKERS,
    ACTION_DEEPEN_FILING_DIFF,
    ACTION_RECHECK_PROMOTION,
    ACTION_RESOLVE_SYNTHESIS_GAP,
    ACTION_RUN_TARGETED_PATTERN_SCAN,
    ACTION_SCHEDULE_LIGHT_REFRESH,
    ACTION_TRACK_VARIANT_PERCEPTION,
    build_escalation_plan,
)
from app.universe.escalation_runner import (
    RESULT_DONE,
    RESULT_PLANNED_ONLY,
    _execute_queue_item,
    _runner_paths,
)
from app.universe.promotion import LANE_2_RESEARCH_QUEUE, LANE_3_MONITOR


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "sample_universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker\nAAA\nBBB\nCCC\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _priority_lanes_stub() -> dict:
    return {
        "campaign_run_id": "campaign_l4",
        "generated_at": "2026-03-22T00:00:00+00:00",
        "lane_1_high_priority": [],
        "lane_2_research_queue": [],
        "lane_3_monitor": [],
        "lane_4_deprioritized": [],
    }


def _source_run(cfg, run_id: str, *, as_of_date: str = "2026-03-22", tickers: list[str] | None = None) -> None:
    root = cfg.outputs_dir / "universe" / run_id
    _write_json(
        root / "autopilot" / "autopilot_state.json",
        {"universe_run_id": run_id, "as_of_date": as_of_date, "depth": "full"},
    )
    _write_json(
        root / "depth_batches" / f"{run_id}_depth_batch" / "batch_state.json",
        {
            "status": "DONE",
            "completed_runs": [
                {"ticker": ticker, "run_id": f"{run_id}__{ticker.lower()}"}
                for ticker in (tickers or ["AAA", "BBB", "CCC"])
            ],
        },
    )


def _variant_report_payload(
    *,
    ticker: str,
    confidence: str = "HIGH",
    direction: str = "UNDERVALUED",
    missing_sources: list[str] | None = None,
) -> dict:
    return {
        "run_id": "run",
        "ticker": ticker,
        "as_of_date": "2026-03-22",
        "perceptions": [
            {
                "perception_id": f"{ticker}_2026-03-22_{direction.lower()}_1",
                "ticker": ticker,
                "as_of_date": "2026-03-22",
                "thesis": f"{ticker} thesis",
                "direction": direction,
                "confidence": confidence,
                "implied_vs_estimated": {
                    "market_implied_growth": 0.12,
                    "estimated_fair_growth": 0.06,
                    "gap_pct": -0.5,
                },
                "supporting_signals": [
                    {
                        "source": "VALUATION",
                        "signal_type": "VALUATION_DISCOUNT",
                        "direction": "SUPPORTS_UNDERVALUED" if direction == "UNDERVALUED" else "SUPPORTS_OVERVALUED",
                        "strength": "HIGH",
                        "summary": "Strong valuation discount",
                        "derived_from": ["valuation.scorecard"],
                    }
                ],
                "contradicting_signals": [],
                "testable_prediction": "Operating margin should expand by >2pp within the next 2 annual filings.",
                "time_horizon": "MEDIUM",
                "catalyst": "Next annual filing",
                "risk": "Margins fail to expand",
                "derived_from": ["valuation.scorecard"],
                "generated_at": "2026-03-22T00:00:00+00:00",
            }
        ],
        "signal_summary": {"by_source": {"VALUATION": 1}},
        "data_quality": {
            "missing_sources": missing_sources or [],
        },
    }


def _row(
    ticker: str,
    *,
    lane: str = LANE_2_RESEARCH_QUEUE,
    source_runs: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    row = {
        "ticker": ticker,
        "priority_lane": lane,
        "latest_value_gate_status": "WATCH",
        "latest_implied_return_base": 0.28,
        "latest_primary_blocker": "NONE",
        "primary_blocker": "NONE",
        "memo_path": f"memo/{ticker}.md",
        "appearances_count": 1,
        "risk_flags": [],
        "strength_flags": [],
        "history": [],
        "source_runs": source_runs or [],
        "investment_readiness_class": "RESEARCH_WORTHY_NOT_READY",
        "evidence_sufficiency_class": "SUFFICIENT",
        "priority_support_codes": ["MULTI_SUPPORT_VALUE_CASE"],
        "variant_perception_count": 0,
        "variant_perception_max_confidence": "NONE",
        "variant_perception_direction": "NONE",
        "variant_signal_source_count": 0,
        "tech_category": "TRADITIONAL_OPERATING",
        "tech_valuation_divergence": "UNKNOWN",
        "filing_diff_high_materiality_count": 0,
        "pattern_hit_count": 0,
        "pattern_confirmed_count": 0,
    }
    if extra:
        row.update(extra)
    return row


def _promotion_state(rows: list[dict]) -> dict:
    return {
        "campaign_run_id": "campaign_l4",
        "generated_at": "2026-03-22T00:00:00+00:00",
        "lane_counts": {},
        "rows": rows,
    }


def _queue_item(
    *,
    ticker: str,
    action_type: str,
    source_run_id: str = "sector_run",
    action_metadata: dict | None = None,
) -> dict:
    return {
        "queue_rank": 1,
        "ticker": ticker,
        "priority_lane": LANE_2_RESEARCH_QUEUE,
        "action_type": action_type,
        "action_reason": f"reason_{action_type.lower()}",
        "action_metadata": action_metadata or {},
        "blocking_reason_code": "NONE",
        "recommended_command": "python -m app.cli noop",
        "source_campaign_run_id": "campaign_l4",
        "source_universe_run_ids": [source_run_id],
        "latest_metrics": {"implied_return_base": 0.25},
        "artifacts_to_read": {"memo_path": "", "watchlist_state_path": "", "source_rollup_paths": []},
        "l4_signal_summary": {},
        "l4_priority_boost": 0,
        "status": "PLANNED",
        "appearances_count": 1,
    }


def test_deepen_filing_diff_planning(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _source_run(cfg, "sector_run")
    _write_json(
        cfg.outputs_dir / "universe" / "sector_run" / "variant_perceptions" / "AAA_2026-03-22.json",
        _variant_report_payload(ticker="AAA", confidence="HIGH", direction="UNDERVALUED", missing_sources=["FILING_DIFF"]),
    )
    monkeypatch.setattr("app.universe.escalation._load_memory_lookup", lambda: {})

    row = _row(
        "AAA",
        source_runs=[{"campaign_item": "software", "universe_run_id": "sector_run", "batch_run_id": "sector_run_depth_batch"}],
        extra={
            "variant_perception_count": 1,
            "variant_perception_max_confidence": "HIGH",
            "variant_perception_direction": "UNDERVALUED",
        },
    )
    payload = build_escalation_plan(
        "campaign_l4",
        _promotion_state([row]),
        _priority_lanes_stub(),
        config={"as_of_date": "2026-03-22", "source_campaign_file": "data/universe/sample.json"},
    )

    actions = [item["action_type"] for item in payload["queue"] if item["ticker"] == "AAA"]
    deepen = next(item for item in payload["queue"] if item["ticker"] == "AAA" and item["action_type"] == ACTION_DEEPEN_FILING_DIFF)

    assert ACTION_DEEPEN_FILING_DIFF in actions
    assert "no filing diff data" in deepen["action_reason"].lower()


def test_run_targeted_pattern_scan_planning(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _source_run(cfg, "sector_run")
    monkeypatch.setattr("app.universe.escalation._load_memory_lookup", lambda: {})

    row = _row(
        "BBB",
        lane=LANE_3_MONITOR,
        source_runs=[{"campaign_item": "software", "universe_run_id": "sector_run", "batch_run_id": "sector_run_depth_batch"}],
        extra={
            "priority_support_codes": [],
            "latest_implied_return_base": "UNKNOWN",
            "investment_readiness_class": "WATCH_ONLY",
        },
    )
    payload = build_escalation_plan(
        "campaign_l4",
        _promotion_state([row]),
        _priority_lanes_stub(),
        config={"as_of_date": "2026-03-22", "source_campaign_file": "data/universe/sample.json"},
    )

    actions = [item["action_type"] for item in payload["queue"] if item["ticker"] == "BBB"]
    assert ACTION_RUN_TARGETED_PATTERN_SCAN in actions


def test_track_variant_perception_planning(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _source_run(cfg, "sector_run")
    _write_json(
        cfg.outputs_dir / "universe" / "sector_run" / "variant_perceptions" / "CCC_2026-03-22.json",
        _variant_report_payload(ticker="CCC", confidence="HIGH", direction="UNDERVALUED"),
    )
    monkeypatch.setattr("app.universe.escalation._load_memory_lookup", lambda: {})

    row = _row(
        "CCC",
        source_runs=[{"campaign_item": "software", "universe_run_id": "sector_run", "batch_run_id": "sector_run_depth_batch"}],
        extra={
            "variant_perception_count": 1,
            "variant_perception_max_confidence": "HIGH",
            "variant_perception_direction": "UNDERVALUED",
        },
    )
    payload = build_escalation_plan(
        "campaign_l4",
        _promotion_state([row]),
        _priority_lanes_stub(),
        config={"as_of_date": "2026-03-22", "source_campaign_file": "data/universe/sample.json"},
    )

    actions = [item["action_type"] for item in payload["queue"] if item["ticker"] == "CCC"]
    assert ACTION_TRACK_VARIANT_PERCEPTION in actions


def test_priority_boost_for_high_perception(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.escalation._load_memory_lookup", lambda: {})

    rows = [
        _row(
            "AAA",
            lane=LANE_3_MONITOR,
            source_runs=[],
            extra={
                "variant_perception_count": 1,
                "variant_perception_max_confidence": "HIGH",
                "variant_perception_direction": "UNDERVALUED",
            },
        ),
        _row("BBB", lane=LANE_3_MONITOR, source_runs=[], extra={"latest_implied_return_base": "UNKNOWN", "priority_support_codes": []}),
    ]
    payload = build_escalation_plan(
        "campaign_l4",
        _promotion_state(rows),
        _priority_lanes_stub(),
        config={"as_of_date": "2026-03-22", "source_campaign_file": "data/universe/sample.json"},
    )

    monitor_rows = [item for item in payload["queue"] if item["action_type"] == ACTION_SCHEDULE_LIGHT_REFRESH]
    assert [item["ticker"] for item in monitor_rows[:2]] == ["AAA", "BBB"]


def test_priority_demotion_for_overvaluation(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.escalation._load_memory_lookup", lambda: {})

    rows = [
        _row("AAA", lane=LANE_3_MONITOR, source_runs=[], extra={"latest_implied_return_base": "UNKNOWN", "priority_support_codes": []}),
        _row(
            "BBB",
            lane=LANE_3_MONITOR,
            source_runs=[],
            extra={
                "variant_perception_count": 1,
                "variant_perception_max_confidence": "HIGH",
                "variant_perception_direction": "OVERVALUED",
                "risk_flags": ["OVERVALUATION_BLOCK_HIGH"],
                "latest_implied_return_base": "UNKNOWN",
                "priority_support_codes": [],
            },
        ),
    ]
    payload = build_escalation_plan(
        "campaign_l4",
        _promotion_state(rows),
        _priority_lanes_stub(),
        config={"as_of_date": "2026-03-22", "source_campaign_file": "data/universe/sample.json"},
    )

    monitor_rows = [item for item in payload["queue"] if item["action_type"] == ACTION_SCHEDULE_LIGHT_REFRESH]
    assert [item["ticker"] for item in monitor_rows[:2]] == ["AAA", "BBB"]


def test_backward_compatibility_without_l4_data(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.escalation._load_memory_lookup", lambda: {})

    row = _row(
        "AAA",
        source_runs=[],
        extra={
            "latest_primary_blocker": "PRICE_UNKNOWN",
            "primary_blocker": "PRICE_UNKNOWN",
            "risk_flags": ["PRICE_UNKNOWN"],
        },
    )
    for field in (
        "variant_perception_count",
        "variant_perception_max_confidence",
        "variant_perception_direction",
        "variant_signal_source_count",
        "tech_category",
        "tech_valuation_divergence",
        "filing_diff_high_materiality_count",
        "pattern_hit_count",
        "pattern_confirmed_count",
    ):
        row.pop(field, None)
    payload = build_escalation_plan(
        "campaign_l4",
        _promotion_state([row]),
        _priority_lanes_stub(),
        config={"as_of_date": "2026-03-22", "source_campaign_file": "data/universe/sample.json"},
    )

    actions = [item["action_type"] for item in payload["queue"] if item["ticker"] == "AAA"]
    assert actions == [ACTION_CLEAR_BLOCKERS, ACTION_RECHECK_PROMOTION]


def test_runner_dispatches_new_l4_actions(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_l4_exec"
    _write_json(
        cfg.campaigns_dir / campaign_run_id / "campaign_state.json",
        {"campaign_run_id": campaign_run_id, "as_of_date": "2026-03-22"},
    )
    _source_run(cfg, "sector_run")
    paths = _runner_paths(campaign_run_id)

    class _DiffReport:
        def model_dump(self, mode: str = "json") -> dict:
            return {
                "ticker": "AAA",
                "changes": [
                    {"materiality": "HIGH"},
                    {"materiality": "LOW"},
                ],
            }

    class _PatternReport:
        def model_dump(self, mode: str = "json") -> dict:
            return {
                "run_id": "sector_run",
                "peer_set_size": 3,
                "pattern_results": [{"hit_count": 2}],
                "patterns_with_signal": ["pattern_1"],
            }

    monkeypatch.setattr("app.diff.engine.build_filing_diff_report", lambda **_kwargs: _DiffReport())
    monkeypatch.setattr(
        "app.synthesis.variant_builder.build_variant_perceptions",
        lambda **_kwargs: _variant_report_payload(ticker="AAA"),
    )
    monkeypatch.setattr("app.patterns.scanner.scan_peer_set", lambda **_kwargs: _PatternReport())
    monkeypatch.setattr(
        "app.patterns.scanner.summarize_pattern_scan_for_ticker",
        lambda _report, _ticker: {"pattern_hit_count": 2},
    )

    deepen = _execute_queue_item(
        campaign_run_id=campaign_run_id,
        queue_item=_queue_item(ticker="AAA", action_type=ACTION_DEEPEN_FILING_DIFF),
        as_of_date="2026-03-22",
        results_rows=[],
        paths=paths,
    )
    pattern = _execute_queue_item(
        campaign_run_id=campaign_run_id,
        queue_item=_queue_item(ticker="AAA", action_type=ACTION_RUN_TARGETED_PATTERN_SCAN),
        as_of_date="2026-03-22",
        results_rows=[],
        paths=paths,
    )

    _write_json(
        cfg.outputs_dir / "universe" / "sector_run" / "variant_perceptions" / "AAA_2026-03-22.json",
        _variant_report_payload(ticker="AAA"),
    )
    track = _execute_queue_item(
        campaign_run_id=campaign_run_id,
        queue_item=_queue_item(ticker="AAA", action_type=ACTION_TRACK_VARIANT_PERCEPTION),
        as_of_date="2026-03-22",
        results_rows=[],
        paths=paths,
    )

    assert deepen["status"] == RESULT_DONE
    assert deepen["outcome_summary"]["changes_found"] == 2
    assert deepen["outcome_summary"]["high_materiality_count"] == 1

    assert pattern["status"] == RESULT_DONE
    assert pattern["outcome_summary"]["hits_for_target_ticker"] == 2

    assert track["status"] == RESULT_DONE
    assert track["outcome_summary"]["perceptions_registered"] == 1
    assert (cfg.outputs_dir / "perception_tracking").exists()


def test_resolve_synthesis_gap_is_planned_only(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    campaign_run_id = "campaign_l4_gap"
    _write_json(
        cfg.campaigns_dir / campaign_run_id / "campaign_state.json",
        {"campaign_run_id": campaign_run_id, "as_of_date": "2026-03-22"},
    )
    paths = _runner_paths(campaign_run_id)

    result = _execute_queue_item(
        campaign_run_id=campaign_run_id,
        queue_item=_queue_item(
            ticker="AAA",
            action_type=ACTION_RESOLVE_SYNTHESIS_GAP,
            action_metadata={
                "gap_description": "Customer concentration remains unclear from current filings.",
                "recommended_action": "Inspect segment disclosures and concentration footnotes, then rerun synthesis.",
            },
        ),
        as_of_date="2026-03-22",
        results_rows=[],
        paths=paths,
    )

    assert result["status"] == RESULT_PLANNED_ONLY
    assert result["outcome_summary"]["status"] == RESULT_PLANNED_ONLY
    assert "customer concentration" in result["outcome_summary"]["gap_description"].lower()
