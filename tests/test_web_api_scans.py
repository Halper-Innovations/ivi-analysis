from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.db import init_db
from app.web.main import app

client = TestClient(app)


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


V2_PAYLOAD = {
    "run_id": "all_sector_v2_20260715_utilities",
    "contract_version": "autonomous_sector_financial_run_v2",
    "pipeline_version": "v2",
    "sector": "utilities",
    "market_cap_focus": "large_and_mega",
    "scan_family": "normal",
    "as_of_date": "2026-07-15",
    "created_at": "2026-07-17T09:31:34+00:00",
    "status": "COMPLETED",
    "final_verdict": "SELECTED",
    "selected_ticker": "AEE",
    "candidate_dispositions": [
        {
            "ticker": "AEE",
            "primary_ticker": "AEE",
            "terminal_state": "UNDERWRITTEN",
            "underwriting_verdict": "ACTIONABLE",
            "watchlist_eligible": True,
            "screen_result": {"status": "PASS", "reason_codes": [], "gate_evaluations": []},
        },
        {
            "ticker": "BADCO",
            "primary_ticker": "BADCO",
            "terminal_state": "SCREENED_OUT",
            "reason_codes": ["PENNY_FLOOR"],
            "screen_result": {
                "status": "FAIL",
                "reason_codes": ["PENNY_FLOOR"],
                "gate_evaluations": [
                    {
                        "rule_id": "PENNY_FLOOR",
                        "status": "FAIL",
                        "applicable": True,
                        "observed_value": 0.42,
                        "threshold": 1.0,
                        "reason_code": "PENNY_FLOOR",
                        "evidence_url": None,
                        "notes": [],
                    }
                ],
            },
        },
    ],
    "lane_usage": {},
    "lane_budget": {},
    "provider_usage": [],
}


def _seed_run(cfg) -> None:
    run_dir = Path(cfg.runs_dir) / "all_sector_v2_replay" / "utilities"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "autonomous_sector_run.json").write_text(json.dumps(V2_PAYLOAD), encoding="utf-8")
    (run_dir / "autonomous_sector_report.md").write_text(
        "# Utilities\n\nSelected AEE.\n", encoding="utf-8"
    )


def _seed_invalid_history_run(cfg) -> None:
    run_dir = Path(cfg.runs_dir) / "history_invalid" / "energy"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **V2_PAYLOAD,
        "run_id": "historical_invalid_energy",
        "sector": "energy",
        "final_verdict": "SELECTED",
        "selected_ticker": "BADCO",
        "no_selection_reason": "Decision content that must not surface.",
    }
    (run_dir / "autonomous_sector_run.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "autonomous_sector_report.md").write_text(
        "# Historical energy\n\nSelected BADCO.\n", encoding="utf-8"
    )


def _seed_ledger_and_watchlist(cfg) -> None:
    from app.autonomous.sweep_delta import record_loaded_set

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    record_loaded_set(
        conn,
        run_id="all_sector_v2_20260715_utilities",
        sector="utilities",
        market_cap_focus="large_and_mega",
        source="sector_scan_db",
        tickers=["AEE", "BADCO"],
        loaded_at="2026-07-15T10:00:00+00:00",
        pipeline_version="v2",
        candidate_dispositions={"AEE": "UNDERWRITTEN", "BADCO": "SCREENED_OUT"},
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            status TEXT,
            conviction_grade TEXT,
            source_run_id TEXT,
            added_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO watchlist(ticker, status, conviction_grade, source_run_id, added_at)"
        " VALUES('AEE', 'ACTIVE', 'ACTIONABLE',"
        " 'all_sector_v2_20260715_utilities', '2026-07-15T12:00:00+00:00')"
    )
    conn.commit()
    conn.close()


def test_api_run_detail_by_slug_and_run_id(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run(cfg)
    _seed_ledger_and_watchlist(cfg)

    by_slug = client.get("/api/runs/all_sector_v2_replay/utilities")
    assert by_slug.status_code == 200
    payload = by_slug.json()
    assert payload["summary"]["run_id"] == "all_sector_v2_20260715_utilities"
    assert payload["summary"]["slug"] == "all_sector_v2_replay/utilities"
    assert payload["quarantined"] is False
    assert [(s["key"], s["count"]) for s in payload["funnel"]] == [
        ("SCREENED_OUT", 1),
        ("UNDERWRITTEN", 1),
    ]
    badco = next(c for c in payload["candidates"] if c["ticker"] == "BADCO")
    assert badco["failed_gates"] == 1
    assert badco["gates"][0]["rule_id"] == "PENNY_FLOOR"
    assert payload["costs"]["available"] is False
    assert payload["watchlist_rows"] == [
        {
            "watchlist_id": 1,
            "ticker": "AEE",
            "status": "ACTIVE",
            "conviction_grade": "ACTIONABLE",
        }
    ]
    assert payload["report_available"] is True

    by_run_id = client.get("/api/runs/all_sector_v2_20260715_utilities")
    assert by_run_id.status_code == 200
    assert by_run_id.json()["summary"]["slug"] == "all_sector_v2_replay/utilities"


def test_api_run_detail_unknown_is_404(monkeypatch, tmp_path):
    _init_temp_env(monkeypatch, tmp_path)
    response = client.get("/api/runs/no_such_run")
    assert response.status_code == 404


def test_api_run_report_renders(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run(cfg)
    response = client.get("/api/runs/all_sector_v2_replay/utilities/report")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "all_sector_v2_20260715_utilities"
    assert payload["slug"] == "all_sector_v2_replay/utilities"
    assert "<h1>Utilities</h1>" in payload["html"]

    missing = client.get("/api/runs/no_such_run/report")
    assert missing.status_code == 404


def test_known_run_requests_refresh_replaced_artifact_and_report(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run(cfg)
    ref = "/api/runs/all_sector_v2_replay/utilities"

    first_detail = client.get(ref)
    first_report = client.get(f"{ref}/report")
    assert first_detail.json()["summary"]["selected_ticker"] == "AEE"
    assert "Selected AEE." in first_report.json()["html"]

    run_dir = Path(cfg.runs_dir) / "all_sector_v2_replay" / "utilities"
    changed_payload = {
        **V2_PAYLOAD,
        "selected_ticker": "DUK",
    }
    (run_dir / "autonomous_sector_run.json").write_text(
        json.dumps(changed_payload),
        encoding="utf-8",
    )
    (run_dir / "autonomous_sector_report.md").write_text(
        "# Utilities\n\nSelected DUK.\n",
        encoding="utf-8",
    )

    current_detail = client.get(ref)
    current_report = client.get(f"{ref}/report")
    assert current_detail.status_code == 200
    assert current_detail.json()["summary"]["selected_ticker"] == "DUK"
    assert current_detail.json()["decision"]["selected_ticker"] == "DUK"
    assert current_report.status_code == 200
    assert "Selected DUK." in current_report.json()["html"]
    assert "Selected AEE." not in current_report.json()["html"]


def test_api_sweeps_groups_runs(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run(cfg)
    _seed_ledger_and_watchlist(cfg)
    response = client.get("/api/sweeps")
    assert response.status_code == 200
    groups = response.json()["groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["week"] == "2026-W29"
    assert group["band"] == "large_and_mega"
    assert group["runs"] == 1
    assert group["sectors"] == ["utilities"]
    assert group["verdicts"] == {"SELECTED": 1}
    assert group["cost_usd"] is None
    assert group["watchlist_rows"] == 1


def test_api_runs_excludes_invalid_history_and_redacts_explicit_history(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run(cfg)
    _seed_invalid_history_run(cfg)

    def integrity_status(path):
        return "INVALID" if "history_invalid" in str(path) else "PASS"

    history_reads = 0

    def authorized_bytes(path):
        nonlocal history_reads
        data = Path(path).read_bytes()
        if "history_invalid" not in str(path):
            return "PASS", data
        # The refresh parses the cached historical identity; the subsequent
        # list authorization rejects its decision fields.
        history_reads += 1
        return ("PASS", data) if history_reads % 2 else ("INVALID", None)

    monkeypatch.setattr(
        "app.web.readmodel.runs_index.authorized_artifact_bytes",
        authorized_bytes,
    )
    monkeypatch.setattr(
        "app.web.readmodel.run_detail.authorized_artifact_bytes",
        lambda path: (
            integrity_status(path),
            Path(path).read_bytes() if integrity_status(path) == "PASS" else None,
        ),
    )

    current = client.get("/api/runs")
    assert current.status_code == 200
    assert [run["run_id"] for run in current.json()["runs"]] == ["all_sector_v2_20260715_utilities"]

    history = client.get("/api/runs", params={"include_history": "true"})
    assert history.status_code == 200
    by_run_id = {run["run_id"]: run for run in history.json()["runs"]}
    # Invalid bytes are not parsed for metadata; the only trusted identifier
    # is the path-derived run directory.
    invalid = by_run_id["energy"]
    assert invalid["integrity_status"] == "INVALID"
    assert invalid["decision_eligible"] is False
    assert invalid["final_verdict"] is None
    assert invalid["selected_ticker"] is None
    assert invalid["no_selection_reason"] is None

    detail = client.get("/api/runs/history_invalid/energy")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["summary"]["integrity_status"] == "INVALID"
    assert detail_payload["summary"]["decision_eligible"] is False
    assert detail_payload["summary"]["final_verdict"] is None
    assert detail_payload["funnel"] == []
    assert detail_payload["candidates"] == []
    assert detail_payload["ranking"] == []
    assert detail_payload["packets"] == []
    assert detail_payload["watchlist_rows"] == []

    report = client.get("/api/runs/history_invalid/energy/report")
    assert report.status_code == 200
    report_payload = report.json()
    assert report_payload["integrity_status"] == "INVALID"
    assert report_payload["decision_eligible"] is False
    assert "excluded from current decisions" in report_payload["html"]
    assert "Selected BADCO" not in report_payload["html"]

    sweeps = client.get("/api/sweeps")
    assert sweeps.status_code == 200
    assert sweeps.json()["groups"][0]["runs"] == 1
    assert sweeps.json()["groups"][0]["sectors"] == ["utilities"]


def test_api_coverage_atlas_and_cell(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run(cfg)
    _seed_ledger_and_watchlist(cfg)

    atlas = client.get("/api/coverage")
    assert atlas.status_code == 200
    payload = atlas.json()
    assert payload["bands"] == ["large_and_mega"]
    cell = next(c for c in payload["cells"] if c["sector"] == "utilities")
    # Both dispositions are coverage-complete states.
    assert cell["tickers"] == 2
    assert cell["pipelines"] == ["v2"]

    detail = client.get(
        "/api/coverage/cell", params={"sector": "utilities", "band": "large_and_mega"}
    )
    assert detail.status_code == 200
    cell_detail = detail.json()
    assert cell_detail["tickers"] == ["AEE", "BADCO"]
    assert cell_detail["runs"][0]["run_id"] == "all_sector_v2_20260715_utilities"
    assert cell_detail["runs"][0]["slug"] == "all_sector_v2_replay/utilities"


def test_api_coverage_missing_engine_db_is_structured_503(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    response = client.get("/api/coverage")
    assert response.status_code == 503
    assert response.json()["detail"]["precondition"] == "engine_db_missing"
