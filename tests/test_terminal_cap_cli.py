from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from app.autonomous.terminal_cap_search import (
    WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
    estimate_terminal_cap_search_worst_case_cost_usd,
    terminal_cap_search_ledger_fingerprint,
    whole_run_preflight_request_fingerprint,
)
from app.autonomous.run_contract import AutonomousRunBudget
from app.autonomous.sector_runtime import DEFAULT_SECTOR_OBJECTIVE
from app.cli import app


runner = CliRunner()


def _preflight(
    path: Path,
    *,
    max_cost_usd: float = 100.0,
    ledger_path: Path | None = None,
) -> Path:
    resolved_ledger = ledger_path or path.with_name(
        f"{path.stem}.terminal_cap_search_ledger.json"
    )
    reserve = estimate_terminal_cap_search_worst_case_cost_usd(
        max_attempts=2,
        max_tool_calls_per_attempt=2,
    )
    other_lanes = 80.0 - reserve
    path.write_text(
        json.dumps(
            {
                "artifact_type": WHOLE_RUN_PREFLIGHT_ARTIFACT_TYPE,
                "status": "AUTHORIZED",
                "run_id": "paid-all-sector-preflight",
                "authorized_at": "2026-07-16T12:00:00+00:00",
                "model": "gpt-5.5",
                "request_fingerprint": whole_run_preflight_request_fingerprint(
                    sectors=["energy"],
                    objective=DEFAULT_SECTOR_OBJECTIVE,
                    as_of_date="2026-07-16",
                    market_cap_focus="smid_cap",
                    pipeline_version="v2",
                    budget=AutonomousRunBudget(
                        max_tool_calls=16,
                        max_turns=6,
                        max_cost_usd=None,
                        timebox_seconds=None,
                        max_candidates=None,
                    ).to_dict(),
                    max_candidates=None,
                ),
                "max_cost_usd": max_cost_usd,
                "worst_case_cost_usd": 80.0,
                "terminal_cap_search_reserved_cost_usd": reserve,
                "lane_worst_case_costs_usd": {
                    "provider_preflight": 0.05,
                    "parent_research": 10.0,
                    "company_underwriting": other_lanes - 15.05,
                    "selected_company_validation": 5.0,
                    "repair_fallback": reserve,
                },
                "terminal_cap_search_max_attempts": 2,
                "terminal_cap_search_max_tool_calls_per_attempt": 2,
                "terminal_cap_search_ledger_fingerprint": (
                    terminal_cap_search_ledger_fingerprint(resolved_ledger)
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_benchmark_cli_constructs_terminal_search_only_from_persisted_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    import app.autonomous.sector_benchmark as benchmark

    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"run_id": "benchmark-no-spend", "sector_results": [], "rollups": {}}

    monkeypatch.setattr(benchmark, "run_autonomous_sector_benchmark", fake_run)
    monkeypatch.setattr(
        benchmark,
        "persist_autonomous_sector_benchmark",
        lambda artifact: SimpleNamespace(
            summary_json=tmp_path / "benchmark.json",
            report_md=tmp_path / "benchmark.md",
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_compact_summary",
        lambda artifact: {"run_id": artifact["run_id"]},
    )
    ledger_path = tmp_path / "terminal-cap-attempts.json"
    preflight_path = _preflight(
        tmp_path / "whole-run-preflight.json",
        ledger_path=ledger_path,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-benchmark",
            "--sectors",
            "energy",
            "--pipeline-version",
            "v2",
            "--as-of",
            "2026-07-16",
            "--terminal-cap-search-preflight",
            str(preflight_path),
            "--terminal-cap-search-ledger",
            str(ledger_path),
            "--no-provider-preflight",
        ],
    )

    assert result.exit_code == 0, result.output
    callback = captured["terminal_cap_search"]
    assert callback.authorization.run_id == "paid-all-sector-preflight"
    assert callback.authorization.max_cost_usd == 100.0
    assert callback.attempt_count == 0
    assert ledger_path.exists()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["attempt_count"] == 0
    assert ledger["authorization"]["whole_run_worst_case_cost_usd"] == 80.0


def test_benchmark_cli_stops_before_spend_when_preflight_ceiling_exceeds_100(
    monkeypatch, tmp_path: Path
) -> None:
    import app.autonomous.sector_benchmark as benchmark

    called = False

    def fake_run(**kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(benchmark, "run_autonomous_sector_benchmark", fake_run)
    preflight_path = _preflight(
        tmp_path / "over-budget-preflight.json",
        max_cost_usd=100.01,
    )

    result = runner.invoke(
        app,
        [
            "autonomous-sector-benchmark",
            "--sectors",
            "energy",
            "--pipeline-version",
            "v2",
            "--as-of",
            "2026-07-16",
            "--terminal-cap-search-preflight",
            str(preflight_path),
        ],
    )

    assert result.exit_code != 0
    assert "whole-run cost ceiling must be within $100.00" in result.output
    assert called is False
