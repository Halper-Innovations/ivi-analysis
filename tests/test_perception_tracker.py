from __future__ import annotations

import json

from app.calibration.perception_tracker import (
    list_pending_perceptions,
    list_resolvable_perceptions,
    register_perception,
    register_perceptions_from_report,
)
from app.db import init_db
from app.synthesis.schemas import SignalEvidence, VariantPerception, VariantPerceptionReport


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    taxonomy = data_dir / "universe" / "sector_taxonomy.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\nBBB,2,BBB\n", encoding="utf-8")
    taxonomy.write_text("ticker,sector\nAAA,Software\nBBB,Semis\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_PATH", str(taxonomy))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _perception(*, ticker: str = "AAA", perception_id: str = "p1", confidence: str = "MEDIUM") -> VariantPerception:
    return VariantPerception(
        perception_id=perception_id,
        ticker=ticker,
        as_of_date="2026-03-22",
        thesis="Test thesis",
        direction="UNDERVALUED",
        confidence=confidence,
        implied_vs_estimated={"market_implied_growth": 0.1, "estimated_fair_growth": 0.2, "gap_pct": 0.1},
        supporting_signals=[
            SignalEvidence(
                source="PATTERN",
                signal_type="deferred_revenue_leading_indicator",
                direction="SUPPORTS_UNDERVALUED",
                strength="HIGH",
                summary="Test",
                derived_from=["pattern.test"],
            ),
            SignalEvidence(
                source="FILING_DIFF",
                signal_type="STRATEGIC_SIGNAL",
                direction="SUPPORTS_UNDERVALUED",
                strength="MEDIUM",
                summary="Test",
                derived_from=["diff.test"],
            ),
        ],
        contradicting_signals=[],
        testable_prediction="Margins should expand.",
        time_horizon="SHORT",
        catalyst="Catalyst",
        risk="Margins fail to expand.",
        derived_from=["variant.test"],
        generated_at="2026-03-22T10:00:00+00:00",
    )


def test_register_perception_creates_tracking_file(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.perception_tracker.utc_now_iso", lambda: "2026-03-22T00:00:00+00:00")

    record = register_perception(_perception(), cfg=cfg)
    path = cfg.outputs_dir / "perception_tracking" / "AAA_p1.json"

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["perception_id"] == "p1"
    assert payload["status"] == "PENDING"
    assert payload["expected_resolution_date"] == "2027-03-22"
    assert payload["pattern_ids_involved"] == ["deferred_revenue_leading_indicator"]
    assert payload["diff_signal_types_involved"] == ["STRATEGIC_SIGNAL"]
    assert record.ticker == "AAA"


# FIX 4: expected_resolution_date must be anchored on the perception as_of_date
# (point-in-time), NOT on the wall-clock registered_at timestamp. With a SHORT
# (365-day) horizon and as_of_date 2026-03-22, the expected resolution is
# 2027-03-22 regardless of when registration physically happens.
def test_expected_resolution_anchored_on_as_of_date(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    # registered_at is much later than as_of_date; anchoring on registered_at
    # would yield 2027-08-09, anchoring on as_of_date yields 2027-03-22.
    monkeypatch.setattr("app.calibration.perception_tracker.utc_now_iso", lambda: "2026-08-10T00:00:00+00:00")

    record = register_perception(_perception(), cfg=cfg)

    assert record.expected_resolution_date == "2027-03-22"
    assert record.registered_at == "2026-08-10T00:00:00+00:00"


def test_register_from_report_creates_all_files(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.perception_tracker.utc_now_iso", lambda: "2026-03-22T00:00:00+00:00")
    report = VariantPerceptionReport(
        run_id="run1__software",
        ticker="AAA",
        as_of_date="2026-03-22",
        perceptions=[
            _perception(perception_id="p1"),
            _perception(perception_id="p2"),
            _perception(ticker="BBB", perception_id="p3"),
        ],
        signal_summary={},
        data_quality={},
    )

    count = register_perceptions_from_report(report, cfg=cfg)

    assert count == 3
    assert (cfg.outputs_dir / "perception_tracking" / "AAA_p1.json").exists()
    assert (cfg.outputs_dir / "perception_tracking" / "AAA_p2.json").exists()
    assert (cfg.outputs_dir / "perception_tracking" / "BBB_p3.json").exists()


def test_list_pending_and_resolvable(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.calibration.perception_tracker.utc_now_iso", lambda: "2026-03-22T00:00:00+00:00")
    register_perception(_perception(perception_id="p1"), cfg=cfg)
    register_perception(_perception(perception_id="p2"), cfg=cfg)

    pending = list_pending_perceptions(cfg=cfg)
    resolvable = list_resolvable_perceptions("2027-03-23", cfg=cfg)

    assert len(pending) == 2
    assert len(resolvable) == 2
