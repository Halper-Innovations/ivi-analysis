from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from app.db import init_db
from app.watchlist.contract import WatchlistEntry
from app.watchlist.schema import ensure_watchlist_schema
from app.watchlist.store import add_or_update, add_price_snapshot
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
    ensure_watchlist_schema(db_path)
    return cfg


def _entry(*, ticker: str, status: str = "ACTIVE") -> WatchlistEntry:
    return WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade="ACTIONABLE" if status == "DEPLOY_READY" else "WATCHLIST_ONLY",
        confidence="HIGH",
        conviction_source="company_autonomy",
        scan_family="normal",
        valuation_anchor_method="DCF",
        valuation_anchor_value=106.67,
        buy_price_target=80.0,
        current_price_at_addition=100.0,
        thesis_text="Durable candidate with a buy-price anchor.",
        key_risks=["Margin compression"],
        falsifiers=["Revenue decline persists"],
        open_questions=[],
        source_run_id="sector_run_1",
        source_sector="industrial_tech",
        added_at="2026-05-08T12:00:00+00:00",
    )


def _seed_watchlist(cfg) -> None:
    db_path = cfg.db_path
    active_id = add_or_update(_entry(ticker="AAA"), db_path=db_path)
    deploy_id = add_or_update(_entry(ticker="BBB", status="DEPLOY_READY"), db_path=db_path)
    add_price_snapshot(active_id, price=95.0, checked_at="2026-07-20T12:00:00+00:00", db_path=db_path)
    add_price_snapshot(deploy_id, price=78.0, checked_at="2026-07-20T12:00:00+00:00", db_path=db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE watchlist SET event_pending = 'merger' WHERE id = ?", (deploy_id,))
    conn.commit()
    conn.close()


def _seed_run_artifact(cfg) -> Path:
    run_dir = Path(cfg.runs_dir) / "autonomous_sector" / "autonomous_sector_biotech_20260101_abc123"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "autonomous_sector_biotech_20260101_abc123",
        "contract_version": "autonomous_sector_financial_run_v1",
        "sector": "biotech",
        "market_cap_focus": "mid_cap",
        "scan_family": "normal",
        "as_of_date": "2026-01-01",
        "created_at": "2026-01-01T10:00:00Z",
        "completed_at": "2026-01-01T11:00:00Z",
        "status": "COMPLETED",
        "final_verdict": "WATCHLIST",
        "company_packets": [{"ticker": "AAA"}, {"ticker": "BBB"}],
    }
    (run_dir / "autonomous_sector_run.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def test_api_watchlist_returns_queue_rows(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_watchlist(cfg)
    response = client.get("/api/watchlist")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    by_ticker = {row["ticker"]: row for row in payload["rows"]}
    assert by_ticker["BBB"]["status"] == "DEPLOY_READY"
    assert by_ticker["BBB"]["presented_status"] == "EVENT_PENDING"
    assert by_ticker["BBB"]["event_pending"] == "merger"
    assert by_ticker["BBB"]["latest_price"] == 78.0
    assert by_ticker["BBB"]["distance_from_buy_pct"] == -2.5
    assert by_ticker["AAA"]["presented_status"] == "ACTIVE"
    assert by_ticker["AAA"]["falsifiers"] == ["Revenue decline persists"]
    assert by_ticker["AAA"]["capacity_class"] == "ADV_UNKNOWN"


def test_api_watchlist_missing_engine_db_is_structured_503(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    response = client.get("/api/watchlist")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["precondition"] == "engine_db_missing"
    assert detail["detail"].endswith("engine.db")


def test_api_runs_indexes_and_lists_artifacts(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run_artifact(cfg)
    response = client.get("/api/runs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["index"] == {
        "discovered": 1,
        "parsed": 1,
        "unchanged": 0,
        "quarantined": 0,
        "removed": 0,
    }
    run = payload["runs"][0]
    assert run["run_id"] == "autonomous_sector_biotech_20260101_abc123"
    assert run["sector"] == "biotech"
    assert run["market_cap_focus"] == "mid_cap"
    assert run["final_verdict"] == "WATCHLIST"
    assert run["pipeline_version"] == "v1"
    assert run["examined_count"] == 2
    assert run["cost_usd"] is None
    assert run["parse_error"] is None

    second = client.get("/api/runs")
    assert second.json()["index"]["unchanged"] == 1
    # The retired flag is intentionally ignored: current-byte authorization
    # and index reconciliation cannot be bypassed by an unknown query param.
    no_refresh = client.get("/api/runs", params={"refresh": "false"})
    assert no_refresh.json()["index"] == {
        "discovered": 1,
        "parsed": 0,
        "unchanged": 1,
        "quarantined": 0,
        "removed": 0,
    }


def test_api_runs_filters(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_run_artifact(cfg)
    match = client.get("/api/runs", params={"sector": "biotech"})
    assert match.json()["runs"][0]["sector"] == "biotech"
    miss = client.get("/api/runs", params={"sector": "utilities"})
    assert miss.json()["runs"] == []
    assert miss.json()["total"] == 1


def test_api_meta_reports_shell_state(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_watchlist(cfg)
    _seed_run_artifact(cfg)
    response = client.get("/api/meta")
    assert response.status_code == 200
    payload = response.json()
    assert payload["app"] == "ivi"
    assert payload["engine_db_present"] is True
    assert payload["watchlist_rows"] == 2
    assert payload["runs_indexed"] == 1


def test_spa_root_serves_boot_page_without_dist(monkeypatch, tmp_path):
    _init_temp_env(monkeypatch, tmp_path)
    monkeypatch.setattr("app.web.main.WEBUI_DIST", tmp_path / "no_dist")
    response = client.get("/")
    assert response.status_code == 200
    assert "IVI is not built yet" in response.text


def test_spa_serves_built_files_and_falls_back_to_index(monkeypatch, tmp_path):
    _init_temp_env(monkeypatch, tmp_path)
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>observatory shell</html>", encoding="utf-8")
    (dist / "assets" / "index-abc.js").write_text("console.log('hi')", encoding="utf-8")
    monkeypatch.setattr("app.web.main.WEBUI_DIST", dist)
    assert client.get("/").text == "<html>observatory shell</html>"
    assert client.get("/assets/index-abc.js").text == "console.log('hi')"
    # Unknown SPA routes fall back to the shell for client-side routing.
    assert client.get("/gauge-lab").text == "<html>observatory shell</html>"
    # Reserved namespaces stay honest 404s; /ops belongs to the SPA now.
    assert client.get("/api/nope").status_code == 404
    assert client.get("/legacy/nope").status_code == 404
    assert client.get("/ops").text == "<html>observatory shell</html>"
    # Path traversal cannot escape the dist directory.
    assert client.get("/assets/../../secret").text == "<html>observatory shell</html>"


def test_retired_legacy_namespaces_are_honest_404s(monkeypatch, tmp_path):
    """The Jinja console was deleted in Phase 4; its URLs must 404, never
    silently serve the SPA shell."""
    _init_temp_env(monkeypatch, tmp_path)
    for path in ("/legacy", "/candidates", "/ticker/AAA", "/status", "/legacy/ops"):
        assert client.get(path).status_code == 404, path


def test_voe_web_refuses_non_local_host(monkeypatch):
    import uvicorn

    from app.cli import app as cli_app

    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append(k))
    runner = CliRunner()
    result = runner.invoke(cli_app, ["web", "--host", "0.0.0.0"])
    assert result.exit_code == 2
    assert calls == []


def test_voe_web_launches_uvicorn_on_localhost_defaults(monkeypatch):
    import uvicorn

    from app.cli import app as cli_app

    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda target, **k: calls.append({"target": target, **k}))
    runner = CliRunner()
    result = runner.invoke(cli_app, ["web"])
    assert result.exit_code == 0
    assert calls == [
        {"target": "app.web.main:app", "host": "127.0.0.1", "port": 8321, "reload": False}
    ]
