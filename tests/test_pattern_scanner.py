from __future__ import annotations

import json
from pathlib import Path

from app.patterns.scanner import scan_peer_set, summarize_pattern_scan_for_ticker


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def _write_dossier(cfg, *, run_id: str, ticker: str, as_of_date: str) -> None:
    dossier_dir = cfg.dossiers_dir / run_id / ticker
    dossier_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "run_id": run_id,
        "as_of_date": as_of_date,
        "docket": [],
        "items": [],
        "time_series": {"rows": [], "standardized_rows": []},
    }
    (dossier_dir / "dossier.json").write_text(json.dumps(payload), encoding="utf-8")


def _companyfacts_payload(
    *,
    revenue: dict[int, float],
    deferred_revenue: dict[int, float],
    gross_margin: float = 0.72,
    rnd_ratio: float = 0.18,
    capex_ratio: float = 0.04,
) -> dict:
    years = sorted(revenue)

    def _rows(values: dict[int, float]) -> list[dict]:
        return [
            {"end": f"{year}-12-31", "filed": f"{year + 1}-02-01", "val": value}
            for year, value in sorted(values.items())
        ]

    gross_profit = {year: revenue[year] * gross_margin for year in years}
    rnd = {year: revenue[year] * rnd_ratio for year in years}
    capex = {year: revenue[year] * capex_ratio for year in years}
    return {
        "entityName": "Example Software Co.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": _rows(revenue)}},
                "ContractWithCustomerLiability": {"units": {"USD": _rows(deferred_revenue)}},
                "GrossProfit": {"units": {"USD": _rows(gross_profit)}},
                "ResearchAndDevelopmentExpense": {"units": {"USD": _rows(rnd)}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": _rows(capex)}},
            }
        },
    }


def _write_companyfacts(tmp_path: Path, ticker: str, payload: dict) -> str:
    path = tmp_path / f"{ticker}.json"
    path.write_text(json.dumps({"companyfacts": payload}), encoding="utf-8")
    return str(path)


def test_scan_peer_set_counts_hits_and_skips_short_histories(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "pattern_run"
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for ticker in tickers:
        _write_dossier(cfg, run_id=run_id, ticker=ticker, as_of_date="2025-03-01")

    cache_paths = {
        "AAA": _write_companyfacts(
            tmp_path,
            "AAA",
            _companyfacts_payload(
                revenue={2020: 100.0, 2021: 110.0, 2022: 121.0, 2023: 155.0, 2024: 195.0},
                deferred_revenue={2020: 80.0, 2021: 112.0, 2022: 151.2, 2023: 190.0, 2024: 220.0},
            ),
        ),
        "BBB": _write_companyfacts(
            tmp_path,
            "BBB",
            _companyfacts_payload(
                revenue={2020: 100.0, 2021: 110.0, 2022: 121.0, 2023: 133.1, 2024: 146.4},
                deferred_revenue={2020: 90.0, 2021: 98.0, 2022: 108.0, 2023: 119.0, 2024: 131.0},
            ),
        ),
        "CCC": _write_companyfacts(
            tmp_path,
            "CCC",
            _companyfacts_payload(
                revenue={2020: 200.0, 2021: 220.0, 2022: 242.0, 2023: 310.0, 2024: 380.0},
                deferred_revenue={2020: 160.0, 2021: 224.0, 2022: 302.4, 2023: 360.0, 2024: 420.0},
            ),
        ),
        "DDD": _write_companyfacts(
            tmp_path,
            "DDD",
            _companyfacts_payload(
                revenue={2020: 100.0, 2021: 110.0, 2022: 121.0, 2023: 130.0},
                deferred_revenue={2020: 70.0, 2021: 77.0, 2022: 100.0, 2023: 140.0},
            ),
        ),
        "EEE": _write_companyfacts(
            tmp_path,
            "EEE",
            _companyfacts_payload(
                revenue={2023: 100.0, 2024: 110.0},
                deferred_revenue={2023: 80.0, 2024: 95.0},
            ),
        ),
    }

    monkeypatch.setattr("app.patterns.scanner.list_dossier_run_tickers", lambda run_id: tickers)
    monkeypatch.setattr(
        "app.patterns.scanner.resolve_financial_facts_asof",
        lambda **kwargs: {
            "ticker": kwargs["ticker"],
            "cache_path": cache_paths[kwargs["ticker"]],
            "derived_from": [f"facts.{kwargs['ticker']}"],
        },
    )

    report = scan_peer_set(run_id=run_id, patterns=["deferred_revenue_leading_indicator"], cfg=cfg)

    assert report.peer_set_size == 5
    assert report.peer_set_tickers == tickers
    assert len(report.pattern_results) == 1
    result = report.pattern_results[0]
    assert result.pattern_id == "deferred_revenue_leading_indicator"
    assert result.hit_count == 3
    assert result.confirmed_count == 2
    assert result.unconfirmed_count == 1
    assert result.sample_size == 2
    assert result.hit_rate == 1.0
    assert "deferred_revenue_leading_indicator" not in report.patterns_with_signal

    summary = summarize_pattern_scan_for_ticker(report, "AAA")
    assert summary["pattern_hit_count"] == 1
    assert "Deferred Revenue Leading Indicator" in summary["summary_text"]
