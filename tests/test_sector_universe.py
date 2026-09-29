from __future__ import annotations

import csv
import json

from app.config import AppConfig, get_config
from app.db import get_db, init_db
from app.valuation.facts import resolve_financial_facts_asof
from app.universe.sector_universe import (
    build_companyfacts_fundamentals_frame,
    deep_scan_sector,
    discover_sector_universe,
    load_sector_ticker_map,
    quick_screen_sector_universe,
)


def _cfg(tmp_path) -> AppConfig:
    data_dir = tmp_path / "data"
    cfg = AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "engine.db",
        cache_dir=data_dir / "cache",
        outputs_dir=data_dir / "outputs",
        dossiers_dir=data_dir / "outputs" / "dossiers",
        sectors_dir=data_dir / "outputs" / "sectors",
        universe_dir=data_dir / "universe",
        sector_sic_config_path=data_dir / "universe" / "sector_sic_ranges.json",
        sector_taxonomy_path=data_dir / "universe" / "sector_taxonomy.csv",
        sector_overrides_path=data_dir / "universe" / "sector_overrides.csv",
    )
    for path in [cfg.data_dir, cfg.cache_dir, cfg.outputs_dir, cfg.dossiers_dir, cfg.sectors_dir, cfg.universe_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return cfg


def test_discover_sector_universe_filters_by_sic_and_supplements(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.sector_sic_config_path.write_text(
        json.dumps({"enterprise_software": {"sic_ranges": ["7371-7379"]}, "large_cap_financials": {"sic_ranges": ["6020-6029"]}}),
        encoding="utf-8",
    )
    with cfg.sector_overrides_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "sector"])
        writer.writeheader()
        writer.writerow({"ticker": "BRK-B", "sector": "large_cap_financials"})

    monkeypatch.setattr(
        "app.universe.sector_universe.load_company_ticker_rows",
        lambda **kwargs: [
            {"ticker": "MSFT", "cik": "0000789019", "company_name": "Microsoft Corp"},
            {"ticker": "ORCL", "cik": "0001341439", "company_name": "Oracle Corp"},
            {"ticker": "JPM", "cik": "0000019617", "company_name": "JPMorgan Chase"},
            {"ticker": "BRK-B", "cik": "0001067983", "company_name": "Berkshire Hathaway"},
        ],
    )

    submissions = {
        "0000789019": {"sic": "7372"},
        "0001341439": {"sic": "7372"},
        "0000019617": {"sic": "6021"},
        "0001067983": {"sic": "6331"},
    }
    monkeypatch.setattr(
        "app.universe.sector_universe.load_company_submissions",
        lambda cik, **kwargs: submissions[str(cik)],
    )

    software = discover_sector_universe("enterprise_software", cfg=cfg, refresh_cache=True)
    financials = discover_sector_universe("large_cap_financials", cfg=cfg, refresh_cache=True)

    assert [row["ticker"] for row in software] == ["MSFT", "ORCL"]
    assert {row["ticker"] for row in financials} == {"JPM", "BRK-B"}
    assert next(row for row in financials if row["ticker"] == "BRK-B")["supplemental"] is True


def test_load_sector_ticker_map_prefers_dynamic_but_keeps_static_supplements(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.sector_sic_config_path.write_text(
        json.dumps({"enterprise_software": {"sic_ranges": ["7371-7379"]}}),
        encoding="utf-8",
    )
    with cfg.sector_taxonomy_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "sector"])
        writer.writeheader()
        writer.writerow({"ticker": "LEGACY", "sector": "enterprise_software"})

    monkeypatch.setattr(
        "app.universe.sector_universe.discover_sector_universe",
        lambda sector, **kwargs: [{"ticker": "MSFT"}, {"ticker": "ORCL"}],
    )

    mapping = load_sector_ticker_map(sectors=["enterprise_software"], cfg=cfg, discover_missing=True)

    assert mapping["enterprise_software"] == ["LEGACY", "MSFT", "ORCL"]


def test_quick_screen_and_companyfacts_fundamentals_frame(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)

    monkeypatch.setattr("app.universe.sector_universe.ensure_facts", lambda ticker, years_back=10: None)

    def _insert_year(ticker: str, year: int, revenue: float, net_income: float) -> None:
        rows = [
            ("revenue", revenue),
            ("gross_profit", revenue * 0.7),
            ("operating_income", revenue * 0.2),
            ("net_income", net_income),
            ("cfo", revenue * 0.18),
            ("capex", revenue * 0.03),
            ("cash", 20.0),
            ("total_debt", 10.0),
            ("shares_outstanding", 100.0),
            ("total_assets", revenue * 1.5),
        ]
        with get_db() as conn:
            for line_item, value in rows:
                conn.execute(
                    """
                    INSERT INTO companyfacts_facts(ticker, fiscal_year, period_end, line_item, value, units, source_url, fetched_at)
                    VALUES(?, ?, ?, ?, ?, 'USD_millions', 'test', '2026-03-24T00:00:00Z')
                    """,
                    (ticker, year, f"{year}-12-31", line_item, value),
                )

    for year, revenue, net_income in [(2022, 150.0, 12.0), (2023, 160.0, 14.0), (2024, 170.0, 16.0)]:
        _insert_year("AAA", year, revenue, net_income)
    for year, revenue, net_income in [(2022, 50.0, -2.0), (2023, 60.0, -1.0), (2024, 70.0, -3.0)]:
        _insert_year("BBB", year, revenue, net_income)

    rows = [
        {"ticker": "AAA", "cik": "1", "company_name": "AAA Corp", "sic": 7372},
        {"ticker": "BBB", "cik": "2", "company_name": "BBB Corp", "sic": 7372},
    ]
    screened, passed = quick_screen_sector_universe(rows, as_of_date="2026-03-24", cfg=cfg)
    fundamentals = build_companyfacts_fundamentals_frame("AAA", as_of_date="2026-03-24", run_id="test_run")

    assert len(screened) == 2
    assert [row["ticker"] for row in passed] == ["AAA"]
    assert len(fundamentals["rows"]) == 3
    assert fundamentals["rows"][-1]["fcf"] == 25.5
    assert fundamentals["derived_signals"]["revenue_cagr_3y"]["value"] != "UNKNOWN"


def test_quick_screen_filters_companyfacts_misses_instead_of_crashing(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)

    def _ensure_facts(ticker: str, years_back: int = 10) -> None:
        if ticker.upper() == "MISS":
            raise RuntimeError("404 companyfacts")

    monkeypatch.setattr("app.universe.sector_universe.ensure_facts", _ensure_facts)

    with get_db() as conn:
        for year, revenue, net_income in [(2022, 120.0, 10.0), (2023, 140.0, 12.0), (2024, 160.0, 14.0)]:
            for line_item, value in [
                ("revenue", revenue),
                ("gross_profit", revenue * 0.7),
                ("operating_income", revenue * 0.2),
                ("net_income", net_income),
                ("cfo", revenue * 0.15),
                ("capex", revenue * 0.03),
                ("cash", 20.0),
                ("total_debt", 5.0),
                ("shares_outstanding", 50.0),
                ("total_assets", revenue * 1.2),
            ]:
                conn.execute(
                    """
                    INSERT INTO companyfacts_facts(ticker, fiscal_year, period_end, line_item, value, units, source_url, fetched_at)
                    VALUES(?, ?, ?, ?, ?, 'USD_millions', 'test', '2026-03-24T00:00:00Z')
                    """,
                    ("GOOD", year, f"{year}-12-31", line_item, value),
                )

    screened, passed = quick_screen_sector_universe(
        [
            {"ticker": "GOOD", "cik": "1", "company_name": "Good Co", "sic": 7372},
            {"ticker": "MISS", "cik": "2", "company_name": "Missing Co", "sic": 7372},
        ],
        as_of_date="2026-03-24",
        cfg=cfg,
    )

    assert len(screened) == 2
    assert [row["ticker"] for row in passed] == ["GOOD"]
    miss = next(row for row in screened if row["ticker"] == "MISS")
    assert miss["pass_screen"] is False
    assert miss["screen_reason"] == "COMPANYFACTS_UNAVAILABLE"


def test_build_companyfacts_fundamentals_frame_uses_fy_rows_only(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)

    monkeypatch.setattr("app.universe.sector_universe.ensure_facts", lambda ticker, years_back=10: None)

    with get_db() as conn:
        rows = [
            ("FY", "2024-12-31", "revenue", 1000.0),
            ("FY", "2024-12-31", "gross_profit", 400.0),
            ("FY", "2024-12-31", "operating_income", 200.0),
            ("Q1", "2024-03-31", "revenue", 100.0),
            ("Q1", "2024-03-31", "gross_profit", 30.0),
            ("Q1", "2024-03-31", "operating_income", 10.0),
        ]
        for period_type, period_end, line_item, value in rows:
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
                )
                VALUES(?, ?, ?, ?, ?, ?, 'USD_millions', 'test', '2026-03-24T00:00:00Z')
                """,
                ("AAA", 2024, period_type, period_end, line_item, value),
            )

    fundamentals = build_companyfacts_fundamentals_frame("AAA", as_of_date="2026-03-24", run_id="fy_only")

    assert len(fundamentals["rows"]) == 1
    assert fundamentals["rows"][0]["revenue"] == 1000.0
    assert fundamentals["rows"][0]["gross_profit"] == 400.0
    assert fundamentals["rows"][0]["gross_margin"] == 0.4


def test_deep_scan_allows_zero_tier3_limit(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)

    monkeypatch.setattr(
        "app.universe.sector_universe.discover_sector_universe",
        lambda sector, cfg=None: [{"ticker": "AAA", "cik": "1", "company_name": "AAA Corp", "sic": 7372}],
    )
    monkeypatch.setattr(
        "app.universe.sector_universe.quick_screen_sector_universe",
        lambda rows, as_of_date, cfg=None: (rows, rows),
    )
    monkeypatch.setattr(
        "app.universe.sector_universe._build_tier2_artifacts",
        lambda sector, run_id, tickers, as_of_date, cfg=None: {
            "zone_distribution": {"GROWTH_DEPENDENT": 1},
            "value_gate_counts": {"FAIL": 1},
            "ranked_candidates": [{"ticker": "AAA", "pricing_zone": "GROWTH_DEPENDENT"}],
        },
    )
    monkeypatch.setattr(
        "app.universe.sector_universe._run_tier3",
        lambda run_id, tier3_tickers, as_of_date, cfg=None: {"tier3_tickers": tier3_tickers},
    )

    summary = deep_scan_sector(
        sector="enterprise_software",
        as_of_date="2026-03-24",
        tier1_limit=10,
        tier2_limit=5,
        tier3_limit=0,
        run_id="test_zero_tier3",
        cfg=cfg,
    )

    assert summary["tier3_tickers"] == []


def test_deep_scan_prioritizes_margin_of_safety_for_tier3_and_emits_alerts(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)

    monkeypatch.setattr(
        "app.universe.sector_universe.discover_sector_universe",
        lambda sector, cfg=None: [
            {"ticker": "DXC", "cik": "1", "company_name": "DXC Corp", "sic": 7372},
            {"ticker": "DDI", "cik": "2", "company_name": "DDI Corp", "sic": 7372},
            {"ticker": "AAA", "cik": "3", "company_name": "AAA Corp", "sic": 7372},
            {"ticker": "BBB", "cik": "4", "company_name": "BBB Corp", "sic": 7372},
        ],
    )
    monkeypatch.setattr(
        "app.universe.sector_universe.quick_screen_sector_universe",
        lambda rows, as_of_date, cfg=None: (rows, rows),
    )
    monkeypatch.setattr(
        "app.universe.sector_universe._build_tier2_artifacts",
        lambda sector, run_id, tickers, as_of_date, cfg=None: {
            "zone_distribution": {"MARGIN_OF_SAFETY": 2, "GROWTH_DEPENDENT": 1, "SPECULATIVE_PREMIUM": 1},
            "value_gate_counts": {"FAIL": 2, "PASS": 1, "WATCH": 1},
            "ranked_candidates": [
                {
                    "ticker": "AAA",
                    "value_gate_status": "PASS",
                    "primary_blocker": "NONE",
                    "pricing_zone": "GROWTH_DEPENDENT",
                    "pricing_zone_detail": {"current_price": 50.0, "epv_adjusted": 60.0},
                },
                {
                    "ticker": "DXC",
                    "value_gate_status": "FAIL",
                    "primary_blocker": "DILUTION_WATCH_MID",
                    "pricing_zone": "MARGIN_OF_SAFETY",
                    "pricing_zone_detail": {
                        "current_price": 11.62,
                        "epv_adjusted": 42.93,
                        "margin_of_safety_vs_epv_adjusted": 0.729,
                    },
                },
                {
                    "ticker": "BBB",
                    "value_gate_status": "WATCH",
                    "primary_blocker": "NONE",
                    "pricing_zone": "SPECULATIVE_PREMIUM",
                    "pricing_zone_detail": {"current_price": 40.0, "epv_adjusted": 20.0},
                },
                {
                    "ticker": "DDI",
                    "value_gate_status": "FAIL",
                    "primary_blocker": "FCF_MARGIN_TREND_STRONGLY_NEGATIVE",
                    "pricing_zone": "MARGIN_OF_SAFETY",
                    "pricing_zone_detail": {
                        "current_price": 8.37,
                        "epv_adjusted": 104.38,
                        "margin_of_safety_vs_epv_adjusted": 0.92,
                    },
                },
            ],
        },
    )
    monkeypatch.setattr(
        "app.universe.sector_universe._run_tier3",
        lambda run_id, tier3_tickers, as_of_date, cfg=None: {"tier3_tickers": tier3_tickers},
    )

    summary = deep_scan_sector(
        sector="enterprise_software",
        as_of_date="2026-03-24",
        tier1_limit=10,
        tier2_limit=4,
        tier3_limit=3,
        run_id="test_margin_of_safety_tier3",
        cfg=cfg,
    )

    assert summary["tier3_tickers"][:2] == ["DXC", "DDI"]
    assert summary["tier3_margin_of_safety"] == ["DXC", "DDI"]
    assert summary["tier3_tickers"][2] == "AAA"
    assert summary["alerts"] == [
        {
            "alert_type": "MARGIN_OF_SAFETY_DETECTED",
            "ticker": "DXC",
            "price": 11.62,
            "epv_adjusted": 42.93,
            "margin_of_safety_pct": 0.729,
            "margin_of_safety_convention": "TEXTBOOK (intrinsic-price)/intrinsic",
            "gate_status": "FAIL",
            "primary_blocker": "DILUTION_WATCH_MID",
            "epv_quality": None,
            "revenue_cagr_5y": None,
            "earnings_quality": None,
            "gate_action": None,
        },
        {
            "alert_type": "MARGIN_OF_SAFETY_DETECTED",
            "ticker": "DDI",
            "price": 8.37,
            "epv_adjusted": 104.38,
            "margin_of_safety_pct": 0.92,
            "margin_of_safety_convention": "TEXTBOOK (intrinsic-price)/intrinsic",
            "gate_status": "FAIL",
            "primary_blocker": "FCF_MARGIN_TREND_STRONGLY_NEGATIVE",
            "epv_quality": None,
            "revenue_cagr_5y": None,
            "earnings_quality": None,
            "gate_action": None,
        },
    ]


def test_companyfacts_fallback_normalizes_units_to_millions(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)

    monkeypatch.setattr("app.valuation.facts.resolve_cik_for_ticker", lambda *args, **kwargs: "0000000001")
    monkeypatch.setattr(
        "app.valuation.facts.fetch_company_facts",
        lambda *args, **kwargs: {
            "status": "OK",
            "reason_code": "FETCH_OK",
            "reason_detail": "",
            "source_resolution": "companyfacts_fetch",
            "cache_path": str(cfg.cache_dir / "companyfacts" / "0000000001.json"),
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
            "http_status": 200,
            "network_attempted": True,
            "attempts_made": 1,
            "retries_configured": 0,
            "companyfacts": {
                "facts": {
                    "dei": {
                        "EntityCommonStockSharesOutstanding": {
                            "units": {
                                "shares": [
                                    {"val": 479_000_000, "end": "2025-12-31", "filed": "2026-02-12", "form": "10-K"}
                                ]
                            }
                        }
                    },
                    "us-gaap": {
                        "NetCashProvidedByUsedInOperatingActivities": {
                            "units": {
                                "USD": [
                                    {"val": 2_883_000_000, "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-12", "form": "10-K"}
                                ]
                            }
                        },
                        "PaymentsToAcquirePropertyPlantAndEquipment": {
                            "units": {
                                "USD": [
                                    {"val": 288_000_000, "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-12", "form": "10-K"}
                                ]
                            }
                        },
                    },
                }
            },
        },
    )

    row = resolve_financial_facts_asof(ticker="CTSH", as_of_date="2026-03-24", cfg=cfg)

    assert row["shares_value"] == 479.0
    assert row["cfo_value"] == 2883.0
    assert row["capex_value"] == 288.0
    assert row["fcf_value"] == 2595.0


def test_tier2_ranked_candidates_uses_scorecard_zone_and_demotes_anomalies(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    init_db(cfg)
    run_dir = cfg.sectors_dir / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "value_gates.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"ticker": "AAA", "gate_status": "PASS", "primary_blocker": "NONE"},
                    {"ticker": "BBB", "gate_status": "PASS", "primary_blocker": "NONE"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "valuation_AAA.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "implied_return_base": 0.25,
                "pricing_zone": "MARGIN_OF_SAFETY",
                "pricing_zone_detail": {"epv_adjusted": 120.0, "dcf_base": 140.0, "current_price": 90.0},
                "valuation_reason_code": None,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "valuation_BBB.json").write_text(
        json.dumps(
            {
                "ticker": "BBB",
                "implied_return_base": 250.0,
                "pricing_zone": "MARGIN_OF_SAFETY",
                "pricing_zone_detail": {"epv_adjusted": 999.0, "dcf_base": 999.0, "current_price": 1.0},
                "valuation_reason_code": "VALUATION_ANOMALY",
            }
        ),
        encoding="utf-8",
    )

    from app.universe import sector_universe as sector_universe_module

    ranked = sector_universe_module._tier2_ranked_candidates(run_dir)

    assert ranked[0]["ticker"] == "AAA"
    assert ranked[0]["pricing_zone"] == "MARGIN_OF_SAFETY"
    assert ranked[1]["ticker"] == "BBB"
    assert ranked[1]["pricing_zone"] == "VALUATION_ANOMALY"
    assert ranked[1]["implied_return_base"] == "UNKNOWN"
