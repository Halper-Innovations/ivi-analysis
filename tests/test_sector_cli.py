from __future__ import annotations

import json
import os

from typer.testing import CliRunner

from app.cli import app
from app.db import get_db, init_db
from app.rlm.state import init_loop_state, save_loop_state, upsert_rlm_run_row


runner = CliRunner()


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    taxonomy = data_dir / "universe" / "sector_taxonomy.csv"
    taxonomy.parent.mkdir(parents=True, exist_ok=True)
    taxonomy.write_text("ticker,sector\nAAA,Software\nBBB,Software\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_PATH", str(taxonomy))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _insert_scorecard(conn, *, ticker: str, as_of_date: str = "2026-02-13") -> None:
    conn.execute(
        """
        INSERT INTO valuations(
            ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at
        )
        VALUES(?, ?, 'scorecard', '{}', '{}', '[]', '2026-02-13T00:00:00Z')
        """,
        (ticker, as_of_date),
    )


def _insert_sector_inference(conn, *, ticker: str, sector: str, as_of_date: str, row_id: int | None = None) -> None:
    if row_id is None:
        conn.execute(
            """
            INSERT INTO sector_inference(
                ticker, as_of_date, inferred_sector, score, derived_from, created_at
            )
            VALUES(?, ?, ?, 1.0, '[]', '2026-02-13T00:00:00Z')
            """,
            (ticker, as_of_date, sector),
        )
        return
    conn.execute(
        """
        INSERT INTO sector_inference(
            id, ticker, as_of_date, inferred_sector, score, derived_from, created_at
        )
        VALUES(?, ?, ?, ?, 1.0, '[]', '2026-02-13T00:00:00Z')
        """,
        (row_id, ticker, as_of_date, sector),
    )


def _seed_scannable_catalog() -> None:
    with get_db() as conn:
        _insert_scorecard(conn, ticker="AAA")
        _insert_scorecard(conn, ticker="BBB")
        _insert_scorecard(conn, ticker="CCC")
        _insert_scorecard(conn, ticker="EEE")

        _insert_sector_inference(conn, ticker="AAA", sector="legacy_sector", as_of_date="2026-01-01", row_id=1)
        _insert_sector_inference(conn, ticker="AAA", sector="semiconductors", as_of_date="2026-02-13", row_id=2)
        _insert_sector_inference(conn, ticker="BBB", sector="enterprise_software", as_of_date="2026-02-13", row_id=3)
        _insert_sector_inference(conn, ticker="CCC", sector="biotech", as_of_date="2026-02-13", row_id=4)
        _insert_sector_inference(conn, ticker="DDD", sector="semiconductors", as_of_date="2026-02-13", row_id=5)
        _insert_sector_inference(conn, ticker="EEE", sector="semiconductors", as_of_date="2026-02-13", row_id=6)
        conn.commit()


def test_sector_cli_commands(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_scannable_catalog()

    listed = runner.invoke(app, ["sector-list"])
    assert listed.exit_code == 0, listed.output
    listed_payload = json.loads(listed.output)
    assert listed_payload["default_source"] == "scannable"
    assert listed_payload["count"] == 3
    assert listed_payload["sectors"] == [
        {"sector": "semiconductors", "count": 2},
        {"sector": "biotech", "count": 1},
        {"sector": "enterprise_software", "count": 1},
    ]
    assert listed_payload["scannable"]["count"] == listed_payload["count"]
    assert listed_payload["taxonomy"]["count"] == 1
    assert listed_payload["taxonomy"]["sectors"] == [{"sector": "Software", "count": 2}]

    peer_kwargs: dict = {}

    def _fake_select_peers(**kwargs):
        peer_kwargs.update(kwargs)
        return {
            "sector": kwargs["sector"],
            "as_of_date": kwargs["as_of_date"],
            "selected_tickers": ["AAA", "BBB"],
            "rows": [],
            "counts": {"selected": 2},
        }

    monkeypatch.setattr("app.sector.peer_set.select_sector_peers", _fake_select_peers)
    peers = runner.invoke(
        app,
        [
            "sector-peers",
            "--sector",
            "Software",
            "--as-of",
            "2026-02-13",
            "--limit",
            "2",
            "--min-peers-dossierable",
            "2",
            "--max-peer-scan",
            "88",
        ],
    )
    assert peers.exit_code == 0, peers.output
    peers_payload = json.loads(peers.output)
    assert peers_payload["selected_tickers"] == ["AAA", "BBB"]
    assert peer_kwargs["min_peers"] == 2
    assert peer_kwargs["max_peer_scan"] == 88

    monkeypatch.setattr(
        "app.sector.synthesis.run_sector_synthesis",
        lambda **kwargs: {"run_id": kwargs["run_id"], "path": "data/outputs/sectors/x/sector_synthesis.json"},
    )
    synth = runner.invoke(
        app,
        ["sector-synth-run", "--sector", "Software", "--as-of", "2026-02-13", "--run-id", "sector_cli_test"],
    )
    assert synth.exit_code == 0, synth.output
    synth_payload = json.loads(synth.output)
    assert synth_payload["run_id"] == "sector_cli_test"

    cycle_kwargs: dict = {}

    def _fake_cycle(**kwargs):
        cycle_kwargs.update(kwargs)
        return {"run_id": kwargs.get("run_id") or "sector_auto", "status": "DONE"}

    monkeypatch.setattr("app.sector.cycle.run_sector_cycle", _fake_cycle)
    cycle = runner.invoke(
        app,
        [
            "sector-cycle",
            "--sector",
            "Software",
            "--as-of",
            "2026-02-13",
            "--peer-limit",
            "2",
            "--limit-dossiers",
            "2",
            "--years-back",
            "5",
            "--sec-budget",
            "1500",
            "--min-peers-dossierable",
            "2",
            "--max-peer-scan",
            "99",
            "--workers",
            "1",
        ],
    )
    assert cycle.exit_code == 0, cycle.output
    cycle_payload = json.loads(cycle.output)
    assert cycle_payload["status"] == "DONE"
    assert cycle_kwargs["sec_budget"] == 1500
    assert cycle_kwargs["min_peers_dossierable"] == 2
    assert cycle_kwargs["max_peer_scan"] == 99

    monkeypatch.setattr(
        "app.sector.cycle.open_sector_run",
        lambda **kwargs: {"run_id": kwargs["run_id"], "status": "DONE"},
    )
    opened = runner.invoke(app, ["sector-open", "--run-id", "sector_cli_test"])
    assert opened.exit_code == 0, opened.output
    opened_payload = json.loads(opened.output)
    assert opened_payload["run_id"] == "sector_cli_test"

    monkeypatch.setattr(
        "app.sector.cycle.sector_run_status",
        lambda **kwargs: {"run_id": kwargs["run_id"], "status": "DONE", "artifact_count_present": 3, "artifact_count_missing": 0},
    )
    status = runner.invoke(app, ["sector-run-status", "--run-id", "sector_cli_test"])
    assert status.exit_code == 0, status.output
    status_payload = json.loads(status.output)
    assert status_payload["run_id"] == "sector_cli_test"
    assert status_payload["status"] == "DONE"

    monkeypatch.setattr(
        "app.sector.cycle.sector_scoreboard_open",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "status": "OK",
            "peer_scoreboard_path": "data/outputs/sectors/sector_cli_test/peer_scoreboard.json",
            "ticker_count": 2,
            "metrics": ["revenue_cagr_10y"],
        },
    )
    scoreboard_open = runner.invoke(app, ["sector-scoreboard-open", "--run-id", "sector_cli_test"])
    assert scoreboard_open.exit_code == 0, scoreboard_open.output
    scoreboard_open_payload = json.loads(scoreboard_open.output)
    assert scoreboard_open_payload["status"] == "OK"

    monkeypatch.setattr(
        "app.sector.cycle.sector_scoreboard_compare",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "metric": kwargs["metric"],
            "rows": [{"ticker": "AAA", "value": 0.1, "rank": 1}],
        },
    )
    scoreboard_compare = runner.invoke(
        app,
        ["sector-scoreboard-compare", "--run-id", "sector_cli_test", "--metric", "revenue_cagr_10y"],
    )
    assert scoreboard_compare.exit_code == 0, scoreboard_compare.output
    scoreboard_compare_payload = json.loads(scoreboard_compare.output)
    assert scoreboard_compare_payload["metric"] == "revenue_cagr_10y"

    monkeypatch.setattr(
        "app.valuation.value_gates.open_value_gates_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "status": "OK",
            "counts": {"PASS": 1, "WATCH": 1, "FAIL": 0},
            "top_pass_by_valuation_gap": [{"ticker": "AAA", "valuation_gap": 0.35}],
            "top_fail_with_reasons": [],
            "suggestions": ["hydrate price/facts for WATCH tickers"],
        },
    )
    value_gates_open = runner.invoke(app, ["value-gates-open", "--run-id", "sector_cli_test"])
    assert value_gates_open.exit_code == 0, value_gates_open.output
    value_gates_payload = json.loads(value_gates_open.output)
    assert value_gates_payload["status"] == "OK"
    assert value_gates_payload["counts"]["PASS"] == 1

    monkeypatch.setattr(
        "app.valuation.value_gates.open_value_gates_calibration_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "status": "OK",
            "counts": {"PASS": 1, "WATCH": 1, "FAIL": 0},
            "threshold_summary": {"mos_min": 0.3},
            "dominant_threshold_blocker": "mos_min",
            "top_blockers": [{"primary_blocker": "MOS_WATCH", "count": 1}],
            "missing_input_breakdown": {"price": 1},
            "top_near_misses": [{"ticker": "AAA"}],
            "suggestions": [],
        },
    )
    calibration_open = runner.invoke(app, ["value-gates-calibration-open", "--run-id", "sector_cli_test"])
    assert calibration_open.exit_code == 0, calibration_open.output
    calibration_payload = json.loads(calibration_open.output)
    assert calibration_payload["status"] == "OK"
    assert calibration_payload["counts"]["PASS"] == 1

    from app.config import get_config

    artifact_dir = get_config().sectors_dir / "sector_cli_test"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "valuation_AAA.json"
    artifact_path.write_text('{"ticker":"AAA","ok":true}\n', encoding="utf-8")
    artifact_open = runner.invoke(
        app,
        [
            "sector-artifact-open",
            "--run-id",
            "sector_cli_test",
            "--kind",
            "valuation",
            "--ticker",
            "AAA",
            "--head",
            "1",
        ],
    )
    assert artifact_open.exit_code == 0, artifact_open.output
    artifact_payload = json.loads(artifact_open.output)
    assert artifact_payload["status"] == "OK"
    assert artifact_payload["path"].endswith("valuation_AAA.json")

    monkeypatch.setattr(
        "app.market.shares_provider.write_shares_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "requested_as_of_date": kwargs["as_of_date"],
            "ticker_count": len(kwargs["tickers"]),
            "ok_count": len(kwargs["tickers"]),
            "unknown_count": 0,
            "rows": [],
            "summary_path": "data/outputs/shares/x/shares_summary.json",
        },
    )
    shares_fetch = runner.invoke(
        app,
        ["shares-fetch", "--tickers", "AAA,BBB", "--as-of", "2026-02-13", "--run-id", "shares_cli_test"],
    )
    assert shares_fetch.exit_code == 0, shares_fetch.output
    shares_payload = json.loads(shares_fetch.output)
    assert shares_payload["run_id"] == "shares_cli_test"
    assert shares_payload["ticker_count"] == 2


def test_scan_invalid_sector_uses_scannable_sector_list(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_scannable_catalog()

    result = runner.invoke(app, ["scan", "unknown_sector", "--dry-run"])

    assert result.exit_code == 1
    assert "No tickers classified as 'unknown_sector'." in result.output
    assert "Available sectors:" in result.output
    assert "  semiconductors: 2 tickers" in result.output
    assert "  biotech: 1 tickers" in result.output
    assert "  enterprise_software: 1 tickers" in result.output
    assert "legacy_sector" not in result.output


def test_list_scannable_sectors_uses_latest_inference_and_scorecard_filter(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_scannable_catalog()

    from app.sector.catalog import list_scannable_sectors

    rows = list_scannable_sectors()

    assert rows == [
        {"sector": "semiconductors", "count": 2},
        {"sector": "biotech", "count": 1},
        {"sector": "enterprise_software", "count": 1},
    ]


def test_list_scannable_sectors_excludes_tickers_with_newer_null_sector_rows(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_scannable_catalog()

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sector_inference(
                ticker, as_of_date, inferred_sector, score, derived_from, created_at
            )
            VALUES('EEE', '2026-03-01', NULL, 0.0, '["method:exclude_non_operating"]', '2026-03-01T00:00:00Z')
            """
        )
        conn.commit()

    from app.sector.catalog import list_scannable_sectors

    rows = list_scannable_sectors()

    assert rows == [
        {"sector": "biotech", "count": 1},
        {"sector": "enterprise_software", "count": 1},
        {"sector": "semiconductors", "count": 1},
    ]

    listed = runner.invoke(app, ["sector-list"])
    assert listed.exit_code == 0, listed.output
    listed_payload = json.loads(listed.output)
    assert listed_payload["sectors"] == rows


def test_sector_rlm_cli_commands(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    loop_kwargs: dict = {}

    def _fake_run_sector_rlm_loop(**kwargs):
        loop_kwargs.update(kwargs)
        return {"run_id": kwargs["run_id"], "status": "DONE", "iteration": 1}

    monkeypatch.setattr("app.rlm.loop.run_sector_rlm_loop", _fake_run_sector_rlm_loop)
    run = runner.invoke(
        app,
        [
            "sector-rlm",
            "--sector",
            "Software",
            "--as-of",
            "2026-02-13",
            "--run-id",
            "sector_rlm_cli_test",
            "--iterations",
            "2",
            "--top-k",
            "5",
            "--budget-usd",
            "1.5",
            "--gate-mos-min",
            "0.25",
            "--gate-valuation-gap-min",
            "0.08",
            "--gate-net-debt-to-cfo-max",
            "3.0",
            "--gate-dilution-max",
            "0.03",
            "--timeout-per-stage",
            "12",
            "--force-restart",
        ],
    )
    assert run.exit_code == 0, run.output
    run_payload = json.loads(run.output)
    assert run_payload["status"] == "DONE"
    assert loop_kwargs["run_id"] == "sector_rlm_cli_test"
    assert loop_kwargs["budget_usd"] == 1.5
    assert loop_kwargs["force_restart"] is True
    assert loop_kwargs["with_prices"] is True
    assert loop_kwargs["gate_mos_min"] == 0.25
    assert loop_kwargs["gate_valuation_gap_min"] == 0.08
    assert loop_kwargs["gate_net_debt_to_cfo_max"] == 3.0
    assert loop_kwargs["gate_dilution_max"] == 0.03
    assert loop_kwargs["timeout_per_stage"] == 12.0

    run_no_prices = runner.invoke(
        app,
        [
            "sector-rlm",
            "--sector",
            "Software",
            "--as-of",
            "2026-02-13",
            "--run-id",
            "sector_rlm_cli_test_no_prices",
            "--iterations",
            "1",
            "--top-k",
            "2",
            "--no-with-prices",
        ],
    )
    assert run_no_prices.exit_code == 0, run_no_prices.output
    assert loop_kwargs["with_prices"] is False

    depth = runner.invoke(
        app,
        [
            "sector-rlm-depth",
            "--sector",
            "Software",
            "--as-of",
            "2026-02-13",
            "--run-id",
            "sector_rlm_cli_test_depth",
            "--iterations",
            "1",
            "--top-k",
            "3",
            "--gate-mos-min",
            "0.2",
            "--timeout-per-stage",
            "9",
        ],
    )
    assert depth.exit_code == 0, depth.output
    depth_payload = json.loads(depth.output)
    assert depth_payload["status"] == "DONE"
    assert loop_kwargs["mode"] == "depth"
    assert loop_kwargs["gate_mos_min"] == 0.2
    assert loop_kwargs["timeout_per_stage"] == 9.0

    monkeypatch.setattr(
        "app.rlm.loop.sector_rlm_status",
        lambda **kwargs: {"run_id": kwargs["run_id"], "status": "STOPPED", "iteration": 1},
    )
    status = runner.invoke(app, ["sector-rlm-status", "--run-id", "sector_rlm_cli_test"])
    assert status.exit_code == 0, status.output
    status_payload = json.loads(status.output)
    assert status_payload["run_id"] == "sector_rlm_cli_test"
    assert status_payload["status"] == "STOPPED"

    monkeypatch.setattr(
        "app.rlm.loop.sector_rlm_open",
        lambda **kwargs: {"run_id": kwargs["run_id"], "status": "DONE", "run_dir": "data/outputs/sectors/sector_rlm_cli_test"},
    )
    opened = runner.invoke(app, ["sector-rlm-open", "--run-id", "sector_rlm_cli_test"])
    assert opened.exit_code == 0, opened.output
    opened_payload = json.loads(opened.output)
    assert opened_payload["run_id"] == "sector_rlm_cli_test"

    resume_kwargs: dict = {}

    def _fake_resume_sector_rlm_loop(**kwargs):
        resume_kwargs.update(kwargs)
        return {"run_id": kwargs["run_id"], "status": "STOPPED", "iteration": 2}

    monkeypatch.setattr("app.rlm.loop.resume_sector_rlm_loop", _fake_resume_sector_rlm_loop)
    resumed = runner.invoke(
        app,
        [
            "sector-rlm-resume",
            "--run-id",
            "sector_rlm_cli_test",
            "--iterations",
            "4",
            "--top-k",
            "7",
            "--workers",
            "2",
            "--timeout-per-stage",
            "15",
            "--force-restart",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    resumed_payload = json.loads(resumed.output)
    assert resumed_payload["run_id"] == "sector_rlm_cli_test"
    assert resume_kwargs["iterations"] == 4
    assert resume_kwargs["top_k"] == 7
    assert resume_kwargs["workers"] == 2
    assert resume_kwargs["force_restart"] is True
    assert resume_kwargs["timeout_per_stage"] == 15.0

    monkeypatch.setattr(
        "app.rlm.loop.cancel_sector_rlm_run",
        lambda **kwargs: {"run_id": kwargs["run_id"], "status": "CANCELLED", "reason": kwargs["reason"]},
    )
    cancelled = runner.invoke(
        app,
        ["sector-rlm-cancel", "--run-id", "sector_rlm_cli_test", "--reason", "operator stop"],
    )
    assert cancelled.exit_code == 0, cancelled.output
    cancelled_payload = json.loads(cancelled.output)
    assert cancelled_payload["status"] == "CANCELLED"


def test_sector_rlm_tail_command(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_rlm_tail_test"
    state = init_loop_state(
        run_id=run_id,
        sector="Software",
        as_of_date="2026-02-13",
        max_iterations=2,
        llm_budget_usd=1.0,
        sec_budget_count=50,
    )
    state.status = "RUNNING"
    state.peer_set = ["AAA", "BBB"]
    state.top_k_current = ["AAA"]
    save_loop_state(state)
    upsert_rlm_run_row(state)

    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = run_dir / "rlm_heartbeat.json"
    heartbeat_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "status": "RUNNING",
                "phase": "executor",
                "iteration": 0,
                "current_ticker": "AAA",
                "last_action": "BUILD_DOSSIERS",
                "updated_at": "2026-02-13T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    log_path = run_dir / "rlm.log"
    log_path.write_text(
        "\n".join(
            [
                '{"ts":"2026-02-13T00:00:00Z","level":"INFO","event":"loop_start"}',
                '{"ts":"2026-02-13T00:00:01Z","level":"INFO","event":"iteration_start"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tail = runner.invoke(
        app,
        [
            "sector-rlm-tail",
            "--run-id",
            run_id,
            "--interval",
            "1",
            "--lines",
            "2",
            "--once",
        ],
    )
    assert tail.exit_code == 0, tail.output
    assert "status=" in tail.output
    assert "rlm.log" in tail.output


def test_sector_scoreboard_compare_implied_return_ranks(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_price_rank"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sector_summary.json").write_text(
        json.dumps({"run_id": run_id, "status": "DONE", "sector": "Software", "as_of_date": "2026-02-13"}),
        encoding="utf-8",
    )
    (run_dir / "peer_scoreboard.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "rows": [
                    {
                        "ticker": "AAA",
                        "metric_values": {"has_price": True, "implied_return_base": 0.22, "valuation_gap": 0.22},
                        "metric_traces": {"implied_return_base": {"derived_from": ["valuation.claims.implied_return_base"]}},
                    },
                    {
                        "ticker": "BBB",
                        "metric_values": {"has_price": True, "implied_return_base": 0.08, "valuation_gap": 0.08},
                        "metric_traces": {"implied_return_base": {"derived_from": ["valuation.claims.implied_return_base"]}},
                    },
                    {
                        "ticker": "CCC",
                        "metric_values": {"has_price": False, "implied_return_base": "UNKNOWN", "valuation_gap": "UNKNOWN"},
                        "metric_traces": {"implied_return_base": {"derived_from": []}},
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["sector-scoreboard-compare", "--run-id", run_id, "--metric", "implied_return_base"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["known_count"] == 2
    assert payload["rows"][0]["ticker"] == "AAA"
    assert payload["rows"][0]["rank"] == 1
    assert payload["rows"][1]["ticker"] == "BBB"
    assert payload["rows"][1]["rank"] == 2
    assert payload["rows"][2]["ticker"] == "CCC"
    assert payload["rows"][2]["rank"] is None


def test_price_coverage_open_and_scoreboard_reference(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_price_coverage_cli"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "sector_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "DONE",
                "sector": "Software",
                "as_of_date": "2026-02-14",
                "artifacts": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "peer_scoreboard.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "rows": [
                    {"ticker": "AAA", "metric_values": {"implied_return_base": "UNKNOWN"}},
                    {"ticker": "BBB", "metric_values": {"implied_return_base": 0.11}},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "price_coverage.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "as_of_date": "2026-02-14",
                "with_prices": True,
                "reason_counts": {"PROVIDER_OK": 1, "SYMBOL_NOT_FOUND": 1},
                "entries": [
                    {
                        "ticker": "AAA",
                        "requested_as_of": "2026-02-14",
                        "resolved_symbol": "aaa.us",
                        "provider_attempts": [{"provider": "stooq", "status": "SYMBOL_NOT_FOUND"}],
                        "cache": {"hit": False, "path": "x", "snapshot_found": False, "cached_as_of_used": None},
                        "market_day": {"requested_day_type": "UNKNOWN", "fallback_days_checked": 7, "asof_used": None},
                        "result": {"status": "UNKNOWN", "reason_code": "SYMBOL_NOT_FOUND", "reason_detail": "override needed"},
                        "output_fields": {"current_price": "UNKNOWN", "price_asof_used": None, "price_source": None, "confidence": None},
                    },
                    {
                        "ticker": "BBB",
                        "requested_as_of": "2026-02-14",
                        "resolved_symbol": "bbb.us",
                        "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                        "cache": {"hit": False, "path": "x", "snapshot_found": False, "cached_as_of_used": None},
                        "market_day": {"requested_day_type": "NON_TRADING", "fallback_days_checked": 2, "asof_used": "2026-02-13"},
                        "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                        "output_fields": {"current_price": 100.0, "price_asof_used": "2026-02-13", "price_source": "stooq", "confidence": "MEDIUM"},
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "shares_coverage.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "as_of_date": "2026-02-14",
                "reason_counts": {"NO_FILINGS": 1, "PROVIDER_OK": 1},
                "entries": [
                    {
                        "ticker": "AAA",
                        "requested_as_of": "2026-02-14",
                        "shares_status": "UNKNOWN",
                        "shares_reason_code": "NO_FILINGS",
                        "shares_reason_detail": "missing filings",
                        "shares_value": "UNKNOWN",
                        "shares_asof_used": None,
                        "shares_source": None,
                        "shares_source_resolution": "provider_lookup",
                        "confidence": None,
                        "resolved_via": "UNKNOWN",
                        "cache": {"hit": False, "path": "x", "snapshot_found": False, "cached_as_of_used": None},
                        "provider_attempts": [{"provider": "shares_filings", "status": "NO_FILINGS"}],
                        "derived_from": ["market.shares_provider[AAA]"],
                    },
                    {
                        "ticker": "BBB",
                        "requested_as_of": "2026-02-14",
                        "shares_status": "OK",
                        "shares_reason_code": "PROVIDER_OK",
                        "shares_reason_detail": "ok",
                        "shares_value": 1000.0,
                        "shares_asof_used": "2026-02-13",
                        "shares_source": "filings_cover_page",
                        "shares_source_resolution": "provider_lookup",
                        "confidence": "HIGH",
                        "resolved_via": "COVER_PAGE",
                        "cache": {"hit": False, "path": "x", "snapshot_found": False, "cached_as_of_used": None},
                        "provider_attempts": [{"provider": "shares_filings", "status": "PROVIDER_OK"}],
                        "derived_from": ["market.shares_provider[BBB]"],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "fcf_coverage.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "as_of_date": "2026-02-14",
                "reason_counts": {"DOSSIER_NO_FCF": 1, "OK": 1},
                "entries": [
                    {
                        "ticker": "AAA",
                        "requested_as_of": "2026-02-14",
                        "fcf_status": "UNKNOWN",
                        "fcf_reason_code": "DOSSIER_NO_FCF",
                        "fcf_reason_detail": "missing fcf",
                        "fcf_value": "UNKNOWN",
                        "fcf_asof_used": None,
                        "fcf_source": None,
                        "fcf_source_resolution": "unknown",
                        "derived_from": ["dossiers.run.AAA.time_series.standardized_rows[2025].fcf"],
                    },
                    {
                        "ticker": "BBB",
                        "requested_as_of": "2026-02-14",
                        "fcf_status": "OK",
                        "fcf_reason_code": "OK",
                        "fcf_reason_detail": "ok",
                        "fcf_value": 120.0,
                        "fcf_asof_used": "2026-02-13",
                        "fcf_source": "current_run_dossier",
                        "fcf_source_resolution": "current_run_dossier",
                        "derived_from": ["dossiers.run.BBB.time_series.standardized_rows[2025].fcf"],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "valuation_coverage.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "as_of_date": "2026-02-14",
                "reason_counts": {"MISSING_SHARES": 1, "OK": 1},
                "entries": [
                    {
                        "ticker": "AAA",
                        "price_status": "OK",
                        "price_reason_code": "CACHE_HIT",
                        "price_source_resolution": "disk_cache",
                        "shares_status": "UNKNOWN",
                        "shares_reason_code": "NO_FILINGS",
                        "shares_source_resolution": "provider_lookup",
                        "fcf_status": "UNKNOWN",
                        "fcf_reason_code": "DOSSIER_NO_FCF",
                        "fcf_source_resolution": "unknown",
                        "valuation_inputs": {
                            "fcf_latest": "OK",
                            "shares_latest": "UNKNOWN",
                            "net_debt_latest": "OK",
                        },
                        "valuation_status": "UNKNOWN",
                        "valuation_reason_code": "MISSING_SHARES",
                        "intrinsic_per_share_base": "UNKNOWN",
                        "implied_return_base": "UNKNOWN",
                        "derived_from": ["valuation_AAA.json.claims.implied_return_base.value"],
                    },
                    {
                        "ticker": "BBB",
                        "price_status": "OK",
                        "price_reason_code": "PROVIDER_OK",
                        "price_source_resolution": "provider_live_fetch",
                        "shares_status": "OK",
                        "shares_reason_code": "PROVIDER_OK",
                        "shares_source_resolution": "provider_lookup",
                        "fcf_status": "OK",
                        "fcf_reason_code": "OK",
                        "fcf_source_resolution": "current_run_dossier",
                        "valuation_inputs": {
                            "fcf_latest": "OK",
                            "shares_latest": "OK",
                            "net_debt_latest": "OK",
                        },
                        "valuation_status": "OK",
                        "valuation_reason_code": None,
                        "intrinsic_per_share_base": 122.5,
                        "implied_return_base": 0.11,
                        "derived_from": ["valuation_BBB.json.claims.implied_return_base.value"],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    coverage = runner.invoke(app, ["price-coverage-open", "--run-id", run_id])
    assert coverage.exit_code == 0, coverage.output
    coverage_payload = json.loads(coverage.output)
    assert coverage_payload["reason_counts"] == {"PROVIDER_OK": 1, "SYMBOL_NOT_FOUND": 1}
    assert coverage_payload["unknown_tickers"] == [
        {
            "ticker": "AAA",
            "reason_code": "SYMBOL_NOT_FOUND",
            "reason_detail": "override needed",
            "suggestions": ["Add symbol override entry to config/price_symbol_overrides.csv."],
        }
    ]
    assert any("symbol overrides" in line for line in coverage_payload["suggestions"])

    shares_coverage = runner.invoke(app, ["shares-coverage-open", "--run-id", run_id])
    assert shares_coverage.exit_code == 0, shares_coverage.output
    shares_payload = json.loads(shares_coverage.output)
    assert shares_payload["reason_counts"] == {"NO_FILINGS": 1, "PROVIDER_OK": 1}
    assert shares_payload["unknown_tickers"] == [
        {
            "ticker": "AAA",
            "shares_reason_code": "NO_FILINGS",
            "shares_reason_detail": "missing filings",
        }
    ]
    assert shares_payload["price_ok_but_shares_unknown"] == [
        {
            "ticker": "AAA",
            "price_reason_code": "CACHE_HIT",
            "shares_reason_code": "NO_FILINGS",
        }
    ]
    fcf_coverage = runner.invoke(app, ["fcf-coverage-open", "--run-id", run_id])
    assert fcf_coverage.exit_code == 0, fcf_coverage.output
    fcf_payload = json.loads(fcf_coverage.output)
    assert fcf_payload["reason_counts"] == {"DOSSIER_NO_FCF": 1, "OK": 1}
    assert fcf_payload["unknown_tickers"] == [
        {
            "ticker": "AAA",
            "fcf_reason_code": "DOSSIER_NO_FCF",
            "fcf_reason_detail": "missing fcf",
        }
    ]
    assert fcf_payload["price_ok_but_fcf_unknown"] == [
        {
            "ticker": "AAA",
            "price_reason_code": "CACHE_HIT",
            "fcf_reason_code": "DOSSIER_NO_FCF",
        }
    ]

    valuation_cov = runner.invoke(app, ["valuation-coverage-open", "--run-id", run_id])
    assert valuation_cov.exit_code == 0, valuation_cov.output
    valuation_payload = json.loads(valuation_cov.output)
    assert valuation_payload["reason_counts"] == {"MISSING_SHARES": 1, "OK": 1}
    assert valuation_payload["price_unknown_reason_counts"] == {}
    assert valuation_payload["price_ok_but_valuation_unknown"] == [
        {
            "ticker": "AAA",
            "valuation_reason_code": "MISSING_SHARES",
            "price_reason_code": "CACHE_HIT",
        }
    ]

    scoreboard_open = runner.invoke(app, ["sector-scoreboard-open", "--run-id", run_id])
    assert scoreboard_open.exit_code == 0, scoreboard_open.output
    scoreboard_payload = json.loads(scoreboard_open.output)
    assert scoreboard_payload["status"] == "OK"
    assert scoreboard_payload["price_coverage_path"].endswith("/price_coverage.json")
    assert scoreboard_payload["price_coverage_reason_counts"] == {"PROVIDER_OK": 1, "SYMBOL_NOT_FOUND": 1}
    assert scoreboard_payload["shares_coverage_path"].endswith("/shares_coverage.json")
    assert scoreboard_payload["shares_coverage_reason_counts"] == {"NO_FILINGS": 1, "PROVIDER_OK": 1}
    assert scoreboard_payload["fcf_coverage_path"].endswith("/fcf_coverage.json")
    assert scoreboard_payload["fcf_coverage_reason_counts"] == {"DOSSIER_NO_FCF": 1, "OK": 1}
    assert scoreboard_payload["valuation_coverage_path"].endswith("/valuation_coverage.json")
    assert scoreboard_payload["valuation_coverage_reason_counts"] == {"MISSING_SHARES": 1, "OK": 1}
    assert scoreboard_payload["implied_return_known_count"] == 1
    assert scoreboard_payload["implied_return_unknown_count"] == 1
    assert scoreboard_payload["valuation_unknown_reason_counts"] == {"MISSING_SHARES": 1}
    assert scoreboard_payload["price_known_count"] == 2
    assert scoreboard_payload["price_unknown_count"] == 0
    assert scoreboard_payload["price_unknown_reason_counts"] == {}


def test_sector_prewarm_prices_cli(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "sector_prewarm_cli"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sector_summary.json").write_text(
        json.dumps({"run_id": run_id, "sector": "Software", "as_of_date": "2026-02-14"}),
        encoding="utf-8",
    )
    (run_dir / "peer_scoreboard.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"ticker": "AAA"},
                    {"ticker": "BBB"},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "app.market.price_prewarm.write_prices_prewarm_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "as_of_date": kwargs["as_of_date"],
            "fallback_days": kwargs["fallback_days"],
            "tickers_requested": kwargs["tickers"],
            "tickers_ok": ["AAA"],
            "tickers_unknown": ["BBB"],
            "ok_count": 1,
            "unknown_count": 1,
            "reason_counts": {"PROVIDER_OK": 1, "PROVIDER_NO_DATA": 1},
            "prices_prewarm_path": str(run_dir / "prices_prewarm.json"),
        },
    )

    prewarm = runner.invoke(
        app,
        ["sector-prewarm-prices", "--run-id", run_id, "--fallback-days", "6"],
    )
    assert prewarm.exit_code == 0, prewarm.output
    payload = json.loads(prewarm.output)
    assert payload["run_id"] == run_id
    assert payload["fallback_days"] == 6
    assert payload["tickers_requested"] == ["AAA", "BBB"]

    prewarm_override = runner.invoke(
        app,
        ["sector-prewarm-prices", "--run-id", run_id, "--fallback-days", "4", "--tickers", "BBB"],
    )
    assert prewarm_override.exit_code == 0, prewarm_override.output
    override_payload = json.loads(prewarm_override.output)
    assert override_payload["fallback_days"] == 4
    assert override_payload["tickers_requested"] == ["BBB"]


def test_offline_seed_prices_cli(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.market.price_provider.write_prices_for_run",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "requested_as_of_date": kwargs["as_of_date"],
            "local_only": kwargs["local_only"],
            "ticker_count": len(kwargs["tickers"]),
            "ok_count": 1,
            "missing_count": 1,
            "reason_counts": {"CACHE_HIT": 1, "OFFLINE_NO_CACHE": 1},
            "rows": [],
            "summary_path": "data/outputs/prices/offline_seed/prices_summary.json",
        },
    )

    result = runner.invoke(
        app,
        [
            "offline-seed-prices",
            "--run-id",
            "offline_seed",
            "--tickers",
            "AAPL,NVDA",
            "--as-of",
            "2026-02-14",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == "offline_seed"
    assert payload["mode"] == "offline_local_seed"
    assert payload["local_only"] is True
    assert payload["ticker_count"] == 2
