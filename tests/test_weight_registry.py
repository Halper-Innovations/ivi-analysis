from __future__ import annotations

import json

from app.calibration.weight_registry import (
    compute_calibration_weights,
    load_calibration_weights,
    write_calibration_weights,
)


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    taxonomy = data_dir / "universe" / "sector_taxonomy.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\nBBB,2,BBB\n", encoding="utf-8")
    taxonomy.write_text("ticker,sector\nAAA,Software\nBBB,Software\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_PATH", str(taxonomy))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_compute_weights_from_resolved_perceptions(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    tracking_dir = cfg.outputs_dir / "perception_tracking"
    for idx in range(10):
        pattern_id = "pattern_a" if idx < 6 else "pattern_b"
        status = "CONFIRMED" if idx in {0, 1, 2, 6, 7} else "DISCONFIRMED"
        _write_json(
            tracking_dir / f"AAA_p{idx}.json",
            {
                "perception_id": f"p{idx}",
                "ticker": "AAA" if idx < 5 else "BBB",
                "as_of_date": "2026-03-22",
                "thesis": "x",
                "direction": "UNDERVALUED",
                "confidence": "MEDIUM",
                "testable_prediction": "x",
                "falsification_trigger": "x",
                "time_horizon": "SHORT",
                "expected_resolution_date": "2027-03-22",
                "supporting_signal_sources": ["PATTERN", "FILING_DIFF"],
                "pattern_ids_involved": [pattern_id],
                "diff_signal_types_involved": ["STRATEGIC_SIGNAL"],
                "status": status,
                "registered_at": "2026-03-22T00:00:00+00:00",
                "resolved_at": "2027-03-22T00:00:00+00:00",
                "resolution": {
                    "outcome_status": status,
                    "price_change_pct": 12.0,
                    "resolution_method": "price_directional_v1",
                    "notes": "x",
                    "resolved_at": "2027-03-22T00:00:00+00:00",
                },
                "derived_from": [],
            },
        )

    weights = compute_calibration_weights(cfg=cfg)

    assert weights.pattern_weights["pattern_a"].hit_rate == 0.5
    assert weights.pattern_weights["pattern_b"].hit_rate == 0.5
    assert weights.diff_signal_weights["STRATEGIC_SIGNAL"].sample_size == 10
    assert weights.sector_accuracy["software"].sample_size == 10


# FIX 3: sample_size must be the decisive denominator (confirmed + disconfirmed),
# NOT including INCONCLUSIVE. 1 confirmed + 4 inconclusive for a pattern must report
# hit_rate=1.0 but sample_size=1, so a thin sample is visible to downstream gates.
def test_sample_size_excludes_inconclusive(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    tracking_dir = cfg.outputs_dir / "perception_tracking"
    for idx in range(5):
        status = "CONFIRMED" if idx == 0 else "INCONCLUSIVE"
        _write_json(
            tracking_dir / f"AAA_p{idx}.json",
            {
                "perception_id": f"p{idx}",
                "ticker": "AAA",
                "as_of_date": "2026-03-22",
                "thesis": "x",
                "direction": "UNDERVALUED",
                "confidence": "MEDIUM",
                "testable_prediction": "x",
                "falsification_trigger": "x",
                "time_horizon": "SHORT",
                "expected_resolution_date": "2027-03-22",
                "supporting_signal_sources": ["PATTERN", "FILING_DIFF"],
                "pattern_ids_involved": ["pattern_thin"],
                "diff_signal_types_involved": ["THIN_SIGNAL"],
                "status": status,
                "registered_at": "2026-03-22T00:00:00+00:00",
                "resolved_at": "2027-03-22T00:00:00+00:00",
                "resolution": {
                    "outcome_status": status,
                    "price_change_pct": 12.0,
                    "resolution_method": "price_directional_v1",
                    "notes": "x",
                    "resolved_at": "2027-03-22T00:00:00+00:00",
                },
                "derived_from": [],
            },
        )

    weights = compute_calibration_weights(cfg=cfg)

    assert weights.pattern_weights["pattern_thin"].hit_rate == 1.0
    assert weights.pattern_weights["pattern_thin"].sample_size == 1
    assert weights.diff_signal_weights["THIN_SIGNAL"].sample_size == 1
    assert weights.sector_accuracy["software"].sample_size == 1


def test_compute_weights_with_no_resolved_perceptions(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    weights = compute_calibration_weights(cfg=cfg)
    assert weights.total_resolved == 0
    assert weights.pattern_weights == {}


def test_write_and_load_weights_round_trip(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    weights = compute_calibration_weights(cfg=cfg)
    path = write_calibration_weights(weights, cfg=cfg)
    loaded = load_calibration_weights(cfg=cfg)
    assert path.exists()
    assert loaded["total_resolved"] == 0


def test_load_weights_defaults_when_missing(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    loaded = load_calibration_weights(cfg=cfg)
    assert loaded["pattern_weights"] == {}
    assert loaded["sector_accuracy"] == {}
