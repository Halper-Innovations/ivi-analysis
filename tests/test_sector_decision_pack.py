from __future__ import annotations

import json
from pathlib import Path

from app.db import init_db
from app.sector.decision_pack import build_sector_decision_pack


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_build_sector_decision_pack_outputs_artifacts(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_decision_pack_test"
    sector_dir = cfg.sectors_dir / run_id
    sector_dir.mkdir(parents=True, exist_ok=True)
    dossier_dir = cfg.dossiers_dir / run_id
    dossier_dir.mkdir(parents=True, exist_ok=True)

    peer_rankings = {
        "rankings": [
            {
                "ticker": "AAA",
                "overall_score": 10.0,
                "metric_ranks": {
                    "future_whale_rank": 1,
                    "whale_signature_rank": 1,
                    "quality_rank": 1,
                    "valuation_rank": 1,
                    "risk_rank": 1,
                },
                "whale_signature": {
                    "score": 80.0,
                    "top_signals": [
                        {"signal": "growth_persistence", "score_contribution": 20.0, "status": "PASS"},
                        {"signal": "margin_expansion", "score_contribution": 12.0, "status": "PASS"},
                        {"signal": "fcf_inflection", "score_contribution": 14.0, "status": "PASS"},
                    ],
                    "gaps": [],
                },
            },
            {
                "ticker": "BBB",
                "overall_score": 9.0,
                "metric_ranks": {
                    "future_whale_rank": 2,
                    "whale_signature_rank": 2,
                    "quality_rank": 2,
                    "valuation_rank": 2,
                    "risk_rank": 2,
                },
                "whale_signature": {
                    "score": 70.0,
                    "top_signals": [{"signal": "growth_persistence", "score_contribution": 15.0, "status": "PASS"}],
                    "gaps": [{"signal": "balance_sheet_resilience", "missing_metrics": ["net_debt"]}],
                },
            },
        ]
    }
    (sector_dir / "peer_rankings.json").write_text(json.dumps(peer_rankings), encoding="utf-8")
    (sector_dir / "peer_report.md").write_text("# peer report", encoding="utf-8")
    (sector_dir / "sector_synthesis.json").write_text(json.dumps({"falsifiers": ["disconfirming check"]}), encoding="utf-8")
    (dossier_dir / "whale_signals_summary.json").write_text(
        json.dumps({"rows": [{"ticker": "AAA"}, {"ticker": "BBB"}]}),
        encoding="utf-8",
    )

    summary = build_sector_decision_pack(
        sector="Software",
        as_of_date="2026-02-13",
        run_id=run_id,
        top_n=2,
    )
    decision_json = Path(summary["decision_pack_path"])
    decision_md = Path(summary["decision_pack_md_path"])
    assert decision_json.exists()
    assert decision_md.exists()
    payload = json.loads(decision_json.read_text(encoding="utf-8"))
    assert payload["top_peers"]
    assert payload["top_candidates_to_deepen"]
    assert all(claim["derived_from"] for claim in payload["numeric_claims"])
    assert "Top 25 peers (ranked)" in decision_md.read_text(encoding="utf-8")
