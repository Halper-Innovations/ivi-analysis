from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import get_db, init_db, utc_now_iso


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA Corp\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_facts_coverage_open_reports_field_counts(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "facts_cov_open"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "facts_coverage.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "ticker_count": 2,
                "facts_blocker_histogram": {"FACTS_NO_CACHE_OFFLINE": 1, "FACTS_OK": 1},
                "retryable_facts_blocker_count": 1,
                "terminal_facts_blocker_count": 0,
                "partial_usable_facts_count": 0,
                "top_retryable_facts_blockers": [
                    {
                        "ticker": "BBB",
                        "facts_blocker_class": "FACTS_NO_CACHE_OFFLINE",
                        "facts_recommended_action": "RECHECK_FACTS_CACHE",
                    }
                ],
                "top_terminal_facts_blockers": [],
                "top_partial_usable_facts": [],
                "recommended_next_action_counts": {"RECHECK_FACTS_CACHE": 1},
                "economic_fail_count_vs_evidence_fail_count": {
                    "evidence_fail_count": 1,
                    "economic_fail_count": 0,
                    "mixed_fail_count": 0,
                    "other_fail_count": 0,
                },
                "reason_counts": {
                    "shares_reason": {"OK": 1, "CIK_MISSING": 1},
                    "cfo_reason": {"OK": 1, "OFFLINE_NO_CACHE": 1},
                    "capex_reason": {"OK": 1, "OFFLINE_NO_CACHE": 1},
                    "fcf_reason": {"OK": 1, "OFFLINE_NO_CACHE": 1},
                },
                "entries": [
                    {
                        "ticker": "AAA",
                        "status": "OK",
                        "shares_status": "OK",
                        "shares_reason": "OK",
                        "cfo_status": "OK",
                        "cfo_reason": "OK",
                        "capex_status": "OK",
                        "capex_reason": "OK",
                        "fcf_status": "OK",
                        "fcf_reason": "OK",
                    },
                    {
                        "ticker": "BBB",
                        "status": "UNKNOWN",
                        "shares_status": "UNKNOWN",
                        "shares_reason": "CIK_MISSING",
                        "cfo_status": "UNKNOWN",
                        "cfo_reason": "OFFLINE_NO_CACHE",
                        "capex_status": "UNKNOWN",
                        "capex_reason": "OFFLINE_NO_CACHE",
                        "fcf_status": "UNKNOWN",
                        "fcf_reason": "OFFLINE_NO_CACHE",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["facts-coverage-open", "--run-id", run_id])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ticker_count"] == 2
    assert payload["shares_ok_count"] == 1
    assert payload["cfo_ok_count"] == 1
    assert payload["capex_ok_count"] == 1
    assert payload["fcf_ok_count"] == 1
    assert payload["retryable_facts_blocker_count"] == 1
    assert payload["top_retryable_facts_blockers"][0]["facts_blocker_class"] == "FACTS_NO_CACHE_OFFLINE"
    assert payload["reason_counts"]["shares_reason"] == {"CIK_MISSING": 1, "OK": 1}
    assert payload["unknown_tickers"][0]["ticker"] == "BBB"
    assert any("companyfacts-fetch" in suggestion for suggestion in payload["suggestions"])


def test_companyfacts_fetch_cli_uses_cache_when_offline(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    run_id = "companyfacts_fetch_cli"
    cache_path = cfg.cache_dir / "companyfacts" / "0000000001.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cik": "0000000001",
                "retrieved_at": utc_now_iso(),
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                "http_status": 200,
                "size_bytes": 100,
                "companyfacts": {
                    "cik": "0000000001",
                    "facts": {"dei": {}, "us-gaap": {}},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Corp', ?)
            ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik, name=excluded.name
            """,
            (utc_now_iso(),),
        )

    result = runner.invoke(
        app,
        [
            "companyfacts-fetch",
            "--tickers",
            "AAA,BBB",
            "--as-of",
            "2026-02-14",
            "--run-id",
            run_id,
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ticker_count"] == 2
    assert payload["ok_count"] == 1
    assert payload["reason_counts"]["CACHE_HIT"] == 1
    assert payload["reason_counts"]["CIK_MISSING"] == 1
    assert (cfg.outputs_dir / "companyfacts" / run_id / "companyfacts_summary.json").exists()
    assert (cfg.outputs_dir / "companyfacts" / run_id / "AAA.json").exists()
