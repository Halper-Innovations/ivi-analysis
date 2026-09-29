from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.autonomous import accepted_census_free_data_repair as repair


def _readiness(status: str) -> dict[str, Any]:
    return {
        "status": "COMPLETED",
        "readiness_status": status,
        "authority_validation_status": "PASSED",
        "authority_fingerprint": "authority-fingerprint",
        "validation_errors": [],
        "census_lineage": {"run_id": "accepted-run"},
        "counts": {
            "sector_count": 1,
            "membership_candidates": 2,
            "execution_candidates": 1,
            "deferred_by_bound": 1,
            "excluded_candidates": 0,
            "ready": 1 if status == "READY" else 0,
            "needs_data": 0 if status == "READY" else 1,
            "incomplete": 0,
        },
        "sector_bindings": {
            "energy": {
                "membership_tickers": ["AAA", "BBB"],
                "execution_tickers": ["AAA"],
                "deferred_by_bound_tickers": ["BBB"],
                "excluded_tickers": [],
                "execution_bound": 1,
            }
        },
        "sector_results": {
            "energy": {
                "candidate_rows": [
                    {
                        "ticker": "AAA",
                        "readiness": status,
                        "missing_inputs": [] if status == "READY" else ["FACTS"],
                    }
                ]
            }
        },
    }


def _payload() -> dict[str, dict[str, Any]]:
    return {
        "energy": {
            "sector": "energy",
            "execution_bound": 1,
            "execution_bound_frozen": True,
            "execution_tickers": ["AAA"],
        }
    }


def test_free_repair_uses_exact_frozen_scope_and_reruns_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readiness_results = iter([_readiness("NEEDS_DATA"), _readiness("READY")])
    monkeypatch.setattr(
        repair,
        "run_v2_readiness_preflight",
        lambda **_kwargs: next(readiness_results),
    )
    calls: list[dict[str, Any]] = []

    def fake_staged_repair(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "status": "COMPLETED",
            "execution_status": "COMPLETED",
            "completed_tickers": ["AAA"],
            "pending_tickers": [],
        }

    monkeypatch.setattr(repair, "pre_assembly_data_gap_repair", fake_staged_repair)
    metric_results = iter(
        [
            {"domain_count:data.sec.gov": 7, "domain_count:stooq.com": 3},
            {"domain_count:data.sec.gov": 9, "domain_count:stooq.com": 4},
        ]
    )
    monkeypatch.setattr(repair, "_http_metrics", lambda _cfg: next(metric_results))
    provider = object()
    monkeypatch.setattr(
        repair,
        "build_free_stooq_price_provider",
        lambda _cfg: provider,
    )
    cfg = SimpleNamespace(
        db_path=tmp_path / "engine.db",
        runs_dir=tmp_path / "runs",
    )

    result = repair.run_accepted_census_free_data_repair(
        benchmark_run_id="benchmark-1",
        sectors=["energy"],
        candidate_payloads=_payload(),
        as_of_date="2026-07-17",
        market_cap_focus="large_and_mega",
        execution_set_fingerprint="execution-fingerprint",
        request_fingerprint="request-fingerprint",
        cfg=cfg,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["tickers"] == ["AAA"]
    assert call["candidate_context"]["execution_tickers"] == ["AAA"]
    assert call["apply_repairs"] is True
    assert call["terminal_cap_search"] is None
    assert call["price_provider"] is provider
    assert call["filing_windows_days"] == {
        "10-K": 800,
        "10-K/A": 800,
        "20-F": 800,
        "20-F/A": 800,
        "40-F": 800,
        "40-F/A": 800,
    }
    assert result["status"] == "COMPLETED"
    assert result["scope_preserved"] is True
    assert result["readiness_transition_counts"] == {
        "NEEDS_DATA->READY": 1
    }
    assert result["actual_usage"] == {
        "model_calls": 0,
        "search_calls": 0,
        "paid_provider_calls": 0,
        "network_calls": 3,
        "free_network_calls": 3,
        "cost_microdollars": 0,
        "cost_usd": "0.000000",
    }
    assert result["network"]["calls_by_domain"] == {
        "data.sec.gov": 2,
        "stooq.com": 1,
    }
    assert result["production_sector_scan_exercised"] is False
    assert result["packet_materialized_count"] == 0


def test_free_repair_stops_before_mutation_when_authority_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failed = _readiness("INCOMPLETE")
    failed.update(
        {
            "status": "INCOMPLETE",
            "authority_validation_status": "FAILED",
            "validation_errors": ["ValueError:lineage drift"],
        }
    )
    monkeypatch.setattr(
        repair,
        "run_v2_readiness_preflight",
        lambda **_kwargs: failed,
    )
    monkeypatch.setattr(
        repair,
        "pre_assembly_data_gap_repair",
        lambda **_kwargs: pytest.fail("repair must not run after authority failure"),
    )
    monkeypatch.setattr(
        repair,
        "_http_metrics",
        lambda _cfg: pytest.fail("network metrics must not start before authority"),
    )
    monkeypatch.setattr(
        repair,
        "build_free_stooq_price_provider",
        lambda _cfg: pytest.fail("price provider must not build before authority"),
    )

    result = repair.run_accepted_census_free_data_repair(
        benchmark_run_id="benchmark-1",
        sectors=["energy"],
        candidate_payloads=_payload(),
        as_of_date="2026-07-17",
        market_cap_focus="large_and_mega",
        execution_set_fingerprint="execution-fingerprint",
        request_fingerprint="request-fingerprint",
        cfg=SimpleNamespace(
            db_path=tmp_path / "engine.db",
            runs_dir=tmp_path / "runs",
        ),
    )

    assert result["status"] == "INCOMPLETE"
    assert result["authority_validation_status"] == "FAILED"
    assert result["validation_errors"] == ["ValueError:lineage drift"]
    assert result["actual_usage"]["network_calls"] == 0


@pytest.mark.parametrize(
    ("sector_count", "bound", "execution_count", "message"),
    [
        (5, 1, 1, "between 1 and 4 sectors"),
        (1, 4, 1, "bound between 1 and 3"),
        (1, 1, 0, "at least one frozen execution name"),
    ],
)
def test_free_repair_scope_limits_fail_closed(
    sector_count: int,
    bound: int,
    execution_count: int,
    message: str,
) -> None:
    sectors = [f"sector_{index}" for index in range(sector_count)]
    payloads = {
        sector: {
            "execution_bound": bound,
            "execution_tickers": [
                f"T{sector_index}_{name_index}"
                for name_index in range(execution_count)
            ],
        }
        for sector_index, sector in enumerate(sectors)
    }
    with pytest.raises(ValueError, match=message):
        repair._validate_scope_limits(
            sectors=sectors,
            candidate_payloads=payloads,
            market_cap_focus="large_and_mega",
        )


def test_free_repair_total_execution_ceiling_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repair, "FREE_DATA_REPAIR_MAX_EXECUTION_NAMES", 11)
    sectors = ["sector_0", "sector_1", "sector_2", "sector_3"]
    payloads = {
        sector: {
            "execution_bound": 3,
            "execution_tickers": [
                f"T{sector_index}_0",
                f"T{sector_index}_1",
                f"T{sector_index}_2",
            ],
        }
        for sector_index, sector in enumerate(sectors)
    }

    with pytest.raises(ValueError, match="11-name ceiling"):
        repair._validate_scope_limits(
            sectors=sectors,
            candidate_payloads=payloads,
            market_cap_focus="large_and_mega",
        )


def test_free_price_provider_excludes_configured_keyed_sources(tmp_path: Path) -> None:
    from app.config import get_config
    from app.market.price_provider import StooqProvider, StooqSecondaryProvider

    cfg = get_config().model_copy(
        update={
            "cache_dir": tmp_path / "cache",
            "eodhd_apikey": "configured-paid-key",
            "stooq_apikey": "configured-key",
        }
    )
    provider = repair.build_free_stooq_price_provider(cfg)

    assert [type(item) for item in provider.providers] == [
        StooqProvider,
        StooqSecondaryProvider,
    ]
    assert [item.cfg.eodhd_apikey for item in provider.providers] == [None, None]
    assert [item.cfg.stooq_apikey for item in provider.providers] == [None, None]
