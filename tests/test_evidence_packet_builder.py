from __future__ import annotations

import json

from app.db import get_db, init_db, utc_now_iso
from app.evidence.packet_builder import (
    _companyfacts_frame,
    _latest_companyfacts_rows,
    build_packet_for_ticker,
)


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _write_dossier(
    cfg, *, run_id: str, ticker: str, as_of_date: str, revenue: float, fcf: float
) -> None:
    dossier_dir = cfg.dossiers_dir / run_id / ticker
    dossier_dir.mkdir(parents=True, exist_ok=True)
    dossier_payload = {
        "ticker": ticker,
        "run_id": run_id,
        "as_of_date": as_of_date,
        "docket": [
            {
                "accession": "0000000000-26-000001",
                "form_type": "10-K",
                "filing_date": as_of_date,
                "period_end": "2025-12-31",
                "primary_doc_url": "https://www.sec.gov/example",
            }
        ],
        "items": [
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "revenue",
                "value": revenue,
                "units": "USD_millions",
                "section_label": "financial_statements",
                "source_url": "https://www.sec.gov/example",
                "snippet": f"Revenue {revenue}",
                "derived_from": ["companyfacts_facts.revenue"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": f"Revenue {revenue}",
                        "section_label": "financial_statements",
                    }
                ],
            },
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "cfo",
                "value": 20.0,
                "units": "USD_millions",
                "section_label": "financial_statements",
                "source_url": "https://www.sec.gov/example",
                "snippet": "CFO 20",
                "derived_from": ["companyfacts_facts.cfo"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "CFO 20",
                        "section_label": "financial_statements",
                    }
                ],
            },
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "capex",
                "value": 5.0,
                "units": "USD_millions",
                "section_label": "financial_statements",
                "source_url": "https://www.sec.gov/example",
                "snippet": "Capex 5",
                "derived_from": ["companyfacts_facts.capex"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "Capex 5",
                        "section_label": "financial_statements",
                    }
                ],
            },
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "shares_outstanding",
                "value": 10.0,
                "units": "shares_millions",
                "section_label": "financial_statements",
                "source_url": "https://www.sec.gov/example",
                "snippet": "Shares 10",
                "derived_from": ["companyfacts_facts.shares_outstanding"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "Shares 10",
                        "section_label": "financial_statements",
                    }
                ],
            },
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "r_and_d_total",
                "value": 18.0,
                "units": "USD_millions",
                "section_label": "financial_statements",
                "source_url": "https://www.sec.gov/example",
                "snippet": "Research and Development 18",
                "derived_from": ["dossier.extractors.r_and_d_total"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "Research and Development 18",
                        "section_label": "financial_statements",
                    }
                ],
            },
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "deferred_revenue_amount",
                "value": 40.0,
                "section_label": "notes",
                "source_url": "https://www.sec.gov/example",
                "snippet": "Deferred revenue was 40",
                "derived_from": ["dossier.extractors.deferred_revenue_amount"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "Deferred revenue was 40",
                        "section_label": "notes",
                    }
                ],
            },
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "rpo_amount",
                "value": 120.0,
                "section_label": "notes",
                "source_url": "https://www.sec.gov/example",
                "snippet": "RPO was 120",
                "derived_from": ["dossier.extractors.rpo_amount"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "RPO was 120",
                        "section_label": "notes",
                    }
                ],
            },
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "customer_concentration_pct",
                "value": 0.12,
                "section_label": "customer_concentration",
                "source_url": "https://www.sec.gov/example",
                "snippet": "One major customer represented 12% of revenue",
                "derived_from": ["dossier.extractors.customer_concentration_pct"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "One major customer represented 12% of revenue",
                        "section_label": "notes",
                    }
                ],
            },
            {
                "ticker": ticker,
                "year": 2025,
                "metric": "segment_count",
                "value": 2.0,
                "section_label": "segment_info",
                "source_url": "https://www.sec.gov/example",
                "snippet": "Cloud; Subscription",
                "derived_from": ["extracted_facts.segments_signal"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "Cloud; Subscription",
                        "section_label": "segment_info",
                    }
                ],
            },
        ],
        "time_series": {
            "standardized_rows": [
                {
                    "year": 2025,
                    "revenue": revenue,
                    "gross_profit": "UNKNOWN",
                    "operating_income": "UNKNOWN",
                    "net_income": "UNKNOWN",
                    "cfo": 20.0,
                    "capex": 5.0,
                    "fcf": fcf,
                    "shares_outstanding": 10.0,
                    "net_debt": 3.0,
                    "r_and_d_total": 18.0,
                    "deferred_revenue_amount": 40.0,
                    "rpo_amount": 120.0,
                    "customer_concentration_pct": 0.12,
                    "segment_count": 2.0,
                }
            ],
            "rows": [
                {
                    "year": 2025,
                    "revenue": revenue,
                    "cfo": 20.0,
                    "capex": 5.0,
                    "fcf": fcf,
                    "shares_outstanding": 10.0,
                    "net_debt": 3.0,
                    "r_and_d_total": 18.0,
                    "deferred_revenue_amount": 40.0,
                    "rpo_amount": 120.0,
                    "customer_concentration_pct": 0.12,
                    "segment_count": 2.0,
                }
            ],
            "standardized_row_traces": {},
            "derived_signals": [
                {
                    "signal": "r_and_d_intensity_latest",
                    "value": 0.18,
                    "derived_from": [
                        "dossier.time_series.rows[2025].r_and_d_total",
                        "dossier.time_series.rows[2025].revenue",
                    ],
                },
                {
                    "signal": "deferred_revenue_to_revenue_latest",
                    "value": 0.4,
                    "derived_from": [
                        "dossier.time_series.rows[2025].deferred_revenue_amount",
                        "dossier.time_series.rows[2025].revenue",
                    ],
                },
                {
                    "signal": "rpo_to_revenue_latest",
                    "value": 1.2,
                    "derived_from": [
                        "dossier.time_series.rows[2025].rpo_amount",
                        "dossier.time_series.rows[2025].revenue",
                    ],
                },
            ],
        },
    }
    (dossier_dir / "dossier.json").write_text(json.dumps(dossier_payload), encoding="utf-8")


def test_build_packet_for_ticker_falls_back_to_dossier_when_fundamentals_missing(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "dossier_pkt_test"
    _write_dossier(
        cfg, run_id=run_id, ticker="AAA", as_of_date="2026-02-13", revenue=100.0, fcf=15.0
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at, valuation_writer_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAA",
                "2026-02-13",
                "dcf",
                json.dumps({"price": 10.0}),
                json.dumps({"status": "OK", "base": 12.0, "low": 10.0, "high": 14.0}),
                json.dumps([]),
                utc_now_iso(),
                "1.0.0",
            ),
        )

    path = build_packet_for_ticker("AAA", "2026-02-13", dossier_run_id=run_id)
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fundamentals"]["revenue"] == 100.0
    assert payload["fundamentals"]["fcf"] == 15.0
    assert payload["fundamentals"]["r_and_d_total"] == 18.0
    assert payload["fundamentals"]["deferred_revenue_amount"] == 40.0
    assert payload["fundamentals"]["rpo_amount"] == 120.0
    assert payload["fundamentals"]["segment_count"] == 2.0
    assert payload["fundamentals"]["deferred_revenue_to_revenue_latest"] == 0.4
    assert payload["financials"][0]["statement_type"] == "companyfacts"
    shares_row = next(
        row for row in payload["financials"] if row["line_item"] == "shares_outstanding"
    )
    assert shares_row["units"] == "shares_millions"
    assert payload["extracted_facts"][0]["value"]["derived_from"] == ["companyfacts_facts.revenue"]
    assert payload["valuations"]["dcf"]["outputs"]["base"] == 12.0


def test_build_packet_for_ticker_prefers_dossier_when_legacy_fundamentals_exist(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "dossier_preferred_test"
    _write_dossier(
        cfg, run_id=run_id, ticker="AAA", as_of_date="2026-02-13", revenue=100.0, fcf=15.0
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "AAA",
                "2026-02-13",
                json.dumps({"revenue": 999.0, "fcf": 777.0}),
                utc_now_iso(),
            ),
        )

    path = build_packet_for_ticker("AAA", "2026-02-13", dossier_run_id=run_id)
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fundamentals"]["revenue"] == 100.0
    assert payload["fundamentals"]["fcf"] == 15.0


def test_build_packet_for_ticker_attaches_cross_filing_pattern_summary(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "pattern_packet_test"
    _write_dossier(
        cfg, run_id=run_id, ticker="AAA", as_of_date="2026-02-13", revenue=100.0, fcf=15.0
    )
    patterns_dir = cfg.outputs_dir / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    (patterns_dir / f"{run_id}_pattern_scan.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "scan_date": utc_now_iso(),
                "peer_set_size": 4,
                "peer_set_tickers": ["AAA", "BBB", "CCC", "DDD"],
                "pattern_results": [
                    {
                        "pattern_id": "deferred_revenue_leading_indicator",
                        "hypothesis": "Deferred revenue leads revenue acceleration.",
                        "hit_count": 3,
                        "confirmed_count": 2,
                        "unconfirmed_count": 1,
                        "hit_rate": 0.67,
                        "sample_size": 3,
                        "hits": [
                            {
                                "ticker": "AAA",
                                "pattern_id": "deferred_revenue_leading_indicator",
                                "years_detected": [2022, 2023],
                                "detection_strength": 0.8,
                                "outcome_confirmed": True,
                                "outcome_value": 0.12,
                                "outcome_details": "Revenue accelerated.",
                                "derived_from": [
                                    "companyfacts.us-gaap.ContractWithCustomerLiability"
                                ],
                            }
                        ],
                    }
                ],
                "patterns_with_signal": ["deferred_revenue_leading_indicator"],
            }
        ),
        encoding="utf-8",
    )

    path = build_packet_for_ticker("AAA", "2026-02-13", dossier_run_id=run_id)
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cross_filing_patterns"]["pattern_hit_count"] == 1
    assert "Deferred Revenue Leading Indicator" in payload["cross_filing_patterns"]["summary_text"]


def test_build_packet_for_ticker_without_as_of_uses_latest_dossier_and_prior_deltas(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _write_dossier(
        cfg, run_id="run_old", ticker="AAA", as_of_date="2026-01-13", revenue=90.0, fcf=10.0
    )
    old_path = build_packet_for_ticker("AAA", "2026-01-13", dossier_run_id="run_old")
    assert old_path is not None

    _write_dossier(
        cfg, run_id="run_new", ticker="AAA", as_of_date="2026-02-13", revenue=100.0, fcf=15.0
    )
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at, valuation_writer_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAA",
                "2026-02-13",
                "scorecard",
                json.dumps({"price": 10.0}),
                json.dumps({"signal": "UNDERVALUED", "type": "EARNINGS_DRIVEN"}),
                json.dumps([]),
                utc_now_iso(),
                "1.0.0",
            ),
        )

    latest_path = build_packet_for_ticker("AAA")
    assert latest_path is not None
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["as_of_date"] == "2026-02-13"
    assert payload["deltas_vs_prior_period"]["revenue"] == 10.0
    assert payload["valuations"]["scorecard"]["outputs"]["signal"] == "UNDERVALUED"


def test_build_packet_backfills_missing_fundamentals_and_price_from_local_sources(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    run_id = "dossier_gap_fill_test"
    dossier_dir = cfg.dossiers_dir / run_id / "AAA"
    dossier_dir.mkdir(parents=True, exist_ok=True)
    dossier_payload = {
        "ticker": "AAA",
        "run_id": run_id,
        "as_of_date": "2026-02-13",
        "docket": [
            {
                "accession": "0000000000-26-000001",
                "form_type": "10-K",
                "filing_date": "2026-02-13",
                "period_end": "2025-12-31",
                "primary_doc_url": "https://www.sec.gov/example",
            }
        ],
        "items": [
            {
                "ticker": "AAA",
                "year": 2025,
                "metric": "revenue",
                "value": 100.0,
                "section_label": "financial_statements",
                "source_url": "https://www.sec.gov/example",
                "snippet": "Revenue 100",
                "derived_from": ["companyfacts_facts.revenue"],
                "citations": [
                    {
                        "source_url": "https://www.sec.gov/example",
                        "snippet": "Revenue 100",
                        "section_label": "financial_statements",
                    }
                ],
            }
        ],
        "time_series": {
            "standardized_rows": [
                {
                    "year": 2025,
                    "revenue": 100.0,
                    "gross_profit": "UNKNOWN",
                    "operating_income": "UNKNOWN",
                    "net_income": "UNKNOWN",
                    "cfo": "UNKNOWN",
                    "capex": "UNKNOWN",
                    "fcf": "UNKNOWN",
                    "shares_outstanding": "UNKNOWN",
                    "net_debt": "UNKNOWN",
                }
            ],
            "rows": [{"year": 2025, "revenue": 100.0}],
            "standardized_row_traces": {},
            "derived_signals": [],
        },
    }
    (dossier_dir / "dossier.json").write_text(json.dumps(dossier_payload), encoding="utf-8")

    monkeypatch.setattr(
        "app.evidence.packet_builder._resolve_local_facts_backfill",
        lambda **_kwargs: {
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/AAA.json",
            "derived_from": ["companyfacts.us-gaap.NetCashProvidedByUsedInOperatingActivities"],
            "cfo_value": 20.0,
            "capex_value": 5.0,
            "fcf_value": 15.0,
            "net_debt_proxy": 3.0,
            "net_debt_derived_from": ["companyfacts.derived.net_debt_proxy"],
        },
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at, valuation_writer_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAA",
                "2026-02-13",
                "reverse_dcf",
                json.dumps(
                    {
                        "market_price": "UNKNOWN",
                        "price_status": "UNKNOWN",
                        "price_source": "disabled",
                        "price_as_of_date": "2026-02-13",
                        "price_fetched_at": "stale",
                        "price_source_url": "",
                    }
                ),
                json.dumps({"status": "OK"}),
                json.dumps([]),
                utc_now_iso(),
                "1.0.0",
            ),
        )
        conn.execute(
            """
            INSERT INTO price_quotes(
                ticker, provider, as_of_date, price, currency, source_url, status, fetched_at, expires_at, raw_json, quote_hash
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAA",
                "stooq",
                "2026-02-13",
                55.0,
                "USD",
                "https://stooq.example/aaa.us.csv",
                "OK",
                utc_now_iso(),
                utc_now_iso(),
                "{}",
                "quote-hash",
            ),
        )

    path = build_packet_for_ticker("AAA", "2026-02-13", dossier_run_id=run_id)
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fundamentals"]["cfo"] == 20.0
    assert payload["fundamentals"]["capex"] == 5.0
    assert payload["fundamentals"]["fcf"] == 15.0
    assert payload["fundamentals"]["net_debt"] == 3.0
    assert payload["valuations"]["reverse_dcf"]["inputs"]["market_price"] == 55.0
    assert payload["valuations"]["reverse_dcf"]["inputs"]["price"] == 55.0
    assert payload["valuations"]["reverse_dcf"]["inputs"]["price_status"] == "OK"
    assert any(row["fact_type"] == "fcf" for row in payload["extracted_facts"])
    assert any(row["line_item"] == "net_debt" for row in payload["financials"])


def test_build_packet_uses_disk_cache_price_when_db_quote_missing(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    cache_path = cfg.cache_dir / "prices" / "AAA.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "entries": [
                    {
                        "requested_as_of_date": "2026-02-12",
                        "source": "stooq",
                        "snapshot": {
                            "ticker": "AAA",
                            "as_of_date": "2026-02-12",
                            "price": 44.5,
                            "currency": "USD",
                            "source": "stooq",
                            "retrieved_at": "2026-02-13T01:00:00+00:00",
                            "url": "https://stooq.example/aaa.us.csv",
                            "confidence": "HIGH",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "AAA",
                "2026-02-13",
                json.dumps(
                    {"revenue": 100.0, "cfo": 20.0, "capex": 5.0, "fcf": 15.0, "net_debt": 3.0}
                ),
                utc_now_iso(),
            ),
        )
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at, valuation_writer_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAA",
                "2026-02-13",
                "reverse_dcf",
                json.dumps(
                    {
                        "market_price": "UNKNOWN",
                        "price_status": "UNKNOWN",
                        "price_source": "disabled",
                    }
                ),
                json.dumps({"status": "OK"}),
                json.dumps([]),
                utc_now_iso(),
                "1.0.0",
            ),
        )

    path = build_packet_for_ticker("AAA", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    inputs = payload["valuations"]["reverse_dcf"]["inputs"]
    assert inputs["market_price"] == 44.5
    assert inputs["price"] == 44.5
    assert inputs["price_source"] == "stooq"
    assert inputs["price_as_of_date"] == "2026-02-12"
    assert inputs["price_source_url"] == "https://stooq.example/aaa.us.csv"
    assert inputs["price_fetched_at"] == "2026-02-13T01:00:00+00:00"


def test_build_packet_overlays_missing_net_debt_and_shares_into_valuation_inputs(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "AAA",
                "2026-02-13",
                json.dumps({"revenue": 100.0, "net_debt": 3.0, "shares_outstanding": 10.4}),
                utc_now_iso(),
            ),
        )
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at, valuation_writer_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "AAA",
                "2026-02-13",
                "reverse_dcf",
                json.dumps(
                    {
                        "market_price": "UNKNOWN",
                        "net_debt": "UNKNOWN",
                        "shares_outstanding": "UNKNOWN",
                    }
                ),
                json.dumps({"status": "OK"}),
                json.dumps([]),
                utc_now_iso(),
                "1.0.0",
            ),
        )

    path = build_packet_for_ticker("AAA", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    inputs = payload["valuations"]["reverse_dcf"]["inputs"]
    assert inputs["net_debt"] == 3.0
    assert inputs["net_debt_status"] == "OK"
    assert inputs["shares_outstanding"] == 10.4
    assert inputs["shares_status"] == "OK"


def test_build_packet_promotes_companyfacts_rows_into_sparse_packets(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000000.json"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "AAA",
                "2026-02-13",
                json.dumps(
                    {
                        "revenue": 100.0,
                        "net_income": 15.0,
                        "cfo": 20.0,
                        "capex": 5.0,
                        "fcf": 15.0,
                        "net_debt": 3.0,
                        "shares_outstanding": 10.0,
                    }
                ),
                utc_now_iso(),
            ),
        )
        for fiscal_year, period_end, line_item, value, units in (
            (2025, "2025-12-31", "revenue", 100.0, "USD_millions"),
            (2025, "2025-12-31", "net_income", 15.0, "USD_millions"),
            (2025, "2025-12-31", "cfo", 20.0, "USD_millions"),
            (2025, "2025-12-31", "capex", 5.0, "USD_millions"),
            (2025, "2025-12-31", "cash", 7.0, "USD_millions"),
            (2025, "2025-12-31", "total_debt", 10.0, "USD_millions"),
            (2025, "2025-12-31", "shares_outstanding", 10.0, "shares_millions"),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_end, line_item, value, units,
                    source_url, fetched_at, filed_date, form, accession
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, '2026-02-01', '10-K', '0000000000-26-000001')
                """,
                (
                    "AAA",
                    fiscal_year,
                    period_end,
                    line_item,
                    value,
                    units,
                    source_url,
                    utc_now_iso(),
                ),
            )

    path = build_packet_for_ticker("AAA", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))

    revenue_fact = next(row for row in payload["extracted_facts"] if row["fact_type"] == "revenue")
    assert revenue_fact["citation"]["source_url"] == source_url
    assert revenue_fact["value"]["derived_from"] == ["companyfacts_facts.revenue"]
    assert revenue_fact["provenance"]["filed_date"] == "2026-02-01"
    assert revenue_fact["provenance"]["period_end"] == "2025-12-31"
    assert revenue_fact["provenance"]["source"] == "SEC_companyfacts"
    assert revenue_fact["provenance"]["accession"] == "0000000000-26-000001"
    assert payload["fundamentals_provenance"]["revenue"]["value"] == 100.0

    fcf_fact = next(row for row in payload["extracted_facts"] if row["fact_type"] == "fcf")
    assert fcf_fact["citation"]["source_url"] == source_url
    assert fcf_fact["value"]["derived_from"] == [
        "companyfacts_facts.cfo",
        "companyfacts_facts.capex",
    ]
    assert fcf_fact["provenance"]["formula"] == "cfo - capex"
    assert fcf_fact["provenance"]["filed_date"] == "2026-02-01"

    net_debt_fact = next(
        row for row in payload["extracted_facts"] if row["fact_type"] == "net_debt"
    )
    assert net_debt_fact["value"]["derived_from"] == [
        "companyfacts_facts.total_debt",
        "companyfacts_facts.cash",
    ]

    shares_row = next(
        row for row in payload["financials"] if row["line_item"] == "shares_outstanding"
    )
    assert shares_row["citation"]["source_url"] == source_url
    assert shares_row["units"] == "shares_millions"


def test_build_packet_uses_cached_submission_metadata_when_filings_table_empty(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            ("AAA", "2026-02-13", json.dumps({"revenue": 100.0}), utc_now_iso()),
        )

    monkeypatch.setattr(
        "app.evidence.packet_builder._cached_filing_metadata_for_ticker",
        lambda **_kwargs: [
            {
                "accession": "0000000000-26-000001",
                "form_type": "10-K",
                "filing_date": "2026-02-13",
                "period_end": "2025-12-31",
                "primary_doc_url": "https://www.sec.gov/Archives/example-10k.htm",
            },
            {
                "accession": "0000000000-25-000099",
                "form_type": "10-Q",
                "filing_date": "2025-11-03",
                "period_end": "2025-09-30",
                "primary_doc_url": "https://www.sec.gov/Archives/example-10q.htm",
            },
        ],
    )

    path = build_packet_for_ticker("AAA", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["filings_used"]) == 2
    assert payload["filings_used"][0]["form_type"] == "10-K"
    assert (
        payload["filings_used"][0]["primary_doc_url"]
        == "https://www.sec.gov/Archives/example-10k.htm"
    )


def test_build_packet_adds_companyfacts_trend_frame_to_sparse_fundamentals(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000000.json"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "AAA",
                "2026-02-13",
                json.dumps(
                    {
                        "revenue": 160.0,
                        "cfo": 40.0,
                        "capex": 14.0,
                        "fcf": 26.0,
                        "shares_outstanding": 10.4,
                    }
                ),
                utc_now_iso(),
            ),
        )
        for (
            fiscal_year,
            period_end,
            revenue,
            cfo,
            capex,
            shares,
            gross_profit,
            operating_income,
        ) in (
            (2022, "2022-12-31", 100.0, 22.0, 10.0, 10.0, 60.0, 24.0),
            (2023, "2023-12-31", 120.0, 28.0, 11.0, 10.1, 74.0, 30.0),
            (2024, "2024-12-31", 145.0, 35.0, 12.0, 10.2, 90.0, 37.0),
            (2025, "2025-12-31", 160.0, 40.0, 14.0, 10.4, 101.0, 41.0),
        ):
            for line_item, value, units in (
                ("revenue", revenue, "USD_millions"),
                ("gross_profit", gross_profit, "USD_millions"),
                ("operating_income", operating_income, "USD_millions"),
                ("cfo", cfo, "USD_millions"),
                ("capex", capex, "USD_millions"),
                ("shares_outstanding", shares, "shares_millions"),
            ):
                conn.execute(
                    """
                    INSERT INTO companyfacts_facts(
                        ticker, fiscal_year, period_end, line_item, value, units,
                        source_url, fetched_at, filed_date, form, accession
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, '2026-02-01', '10-K', '0000000000-26-000001')
                    """,
                    (
                        "AAA",
                        fiscal_year,
                        period_end,
                        line_item,
                        value,
                        units,
                        source_url,
                        utc_now_iso(),
                    ),
                )

    path = build_packet_for_ticker("AAA", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    fundamentals = payload["fundamentals"]

    assert len(fundamentals["rows"]) == 4
    assert fundamentals["rows"][-1]["year"] == 2025
    assert fundamentals["rows"][-1]["fcf"] == 26.0
    assert fundamentals["rows"][-1]["gross_margin"] == 101.0 / 160.0
    assert "2025" in fundamentals["row_traces"]
    assert fundamentals["row_traces"]["2025"]["fcf"]["derived_from"] == [
        "companyfacts_facts.cfo",
        "companyfacts_facts.capex",
    ]
    assert isinstance(fundamentals["derived_signals"]["revenue_cagr_3y"]["value"], float)
    assert isinstance(fundamentals["derived_signals"]["dilution_rate_shares_cagr"]["value"], float)
    assert (
        fundamentals["revenue_cagr_3y"]
        == fundamentals["derived_signals"]["revenue_cagr_3y"]["value"]
    )
    assert (
        fundamentals["dilution_rate_shares_cagr"]
        == fundamentals["derived_signals"]["dilution_rate_shares_cagr"]["value"]
    )


def test_companyfacts_frame_ignores_quarterly_rows_when_building_annual_history(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000000.json"
    with get_db() as conn:
        for line_item, value, units, period_type, period_end in (
            ("revenue", 160.0, "USD_millions", "FY", "2025-12-31"),
            ("cfo", 40.0, "USD_millions", "FY", "2025-12-31"),
            ("capex", 14.0, "USD_millions", "FY", "2025-12-31"),
            ("cash", 7.0, "USD_millions", "FY", "2025-12-31"),
            ("shares_outstanding", 10.4, "shares_millions", "FY", "2025-12-31"),
            ("total_debt", 99.0, "USD_millions", "Q3", "2025-09-30"),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form, accession
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-02-01', '10-K', '0000000000-26-000001')
                """,
                (
                    "AAA",
                    2025,
                    period_type,
                    period_end,
                    line_item,
                    value,
                    units,
                    source_url,
                    utc_now_iso(),
                ),
            )

        frame = _companyfacts_frame(conn, ticker="AAA", as_of_date="2026-02-13")

    assert frame["rows"] == [
        {
            "year": 2025,
            "revenue": 160.0,
            "gross_profit": "UNKNOWN",
            "operating_income": "UNKNOWN",
            "net_income": "UNKNOWN",
            "cfo": 40.0,
            "capex": 14.0,
            "fcf": 26.0,
            "shares_outstanding": 10.4,
            "net_debt": "UNKNOWN",
            "r_and_d_total": "UNKNOWN",
            "share_repurchases_amount": "UNKNOWN",
            "dividends_paid_amount": "UNKNOWN",
            "deposits": "UNKNOWN",
            "loans": "UNKNOWN",
            "investment_securities": "UNKNOWN",
            "total_assets": "UNKNOWN",
            "assets_under_management": "UNKNOWN",
            "allowance_for_credit_losses": "UNKNOWN",
            "provision_for_credit_losses": "UNKNOWN",
            "net_charge_offs": "UNKNOWN",
            "nonaccrual_loans": "UNKNOWN",
            "deposits_to_assets": "UNKNOWN",
            "loans_to_deposits": "UNKNOWN",
            "allowance_to_loans": "UNKNOWN",
            "provision_to_loans": "UNKNOWN",
            "net_charge_offs_to_loans": "UNKNOWN",
            "debt_to_assets": "UNKNOWN",
            "gross_margin": "UNKNOWN",
            "op_margin": "UNKNOWN",
            "fcf_margin": 0.1625,
            "cfo_margin": 0.25,
        }
    ]


def test_companyfacts_blank_or_ambiguous_units_cannot_create_prompt_evidence(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000000.json"
    with get_db() as conn:
        for line_item, value, units in (
            ("cfo", 40.0, ""),
            ("capex", 14.0, "USD"),
            ("cash", 7.0, None),
            ("total_debt", 10.0, "shares_millions"),
            ("shares_outstanding", 10.4, "USD_millions"),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession
                ) VALUES(
                    'AAA', 2025, 'FY', '2025-12-31', ?, ?, ?, ?, ?,
                    '2026-02-01', '10-K', '0000000000-26-000001'
                )
                """,
                (line_item, value, units, source_url, utc_now_iso()),
            )

        frame = _companyfacts_frame(
            conn,
            ticker="AAA",
            as_of_date="2026-02-13",
        )
        latest = _latest_companyfacts_rows(
            conn,
            ticker="AAA",
            as_of_date="2026-02-13",
        )

    row = frame["rows"][-1]
    assert row["cfo"] == "UNKNOWN"
    assert row["capex"] == "UNKNOWN"
    assert row["fcf"] == "UNKNOWN"
    assert row["net_debt"] == "UNKNOWN"
    assert row["shares_outstanding"] == "UNKNOWN"
    assert frame["row_traces"]["2025"]["fcf"]["provenance"] is None
    assert frame["row_traces"]["2025"]["net_debt"]["provenance"] is None
    assert latest == {}


def test_latest_companyfacts_rows_uses_latest_annual_period_not_latest_quarter(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000000.json"
    with get_db() as conn:
        for fiscal_year, period_type, period_end, line_item, value in (
            (2025, "FY", "2025-12-31", "revenue", 160.0),
            (2025, "FY", "2025-12-31", "shares_outstanding", 10.4),
            (2026, "Q3", "2026-09-30", "revenue", 999.0),
            (2026, "Q3", "2026-09-30", "shares_outstanding", 11.1),
        ):
            units = "shares_millions" if line_item == "shares_outstanding" else "USD_millions"
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form, accession
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-11-01', '10-K', '0000000000-26-000001')
                """,
                (
                    "AAA",
                    fiscal_year,
                    period_type,
                    period_end,
                    line_item,
                    value,
                    units,
                    source_url,
                    utc_now_iso(),
                ),
            )

        latest_rows = _latest_companyfacts_rows(conn, ticker="AAA", as_of_date="2026-11-15")

    assert latest_rows["revenue"]["period_end"] == "2025-12-31"
    assert latest_rows["revenue"]["value"] == 160.0
    assert latest_rows["shares_outstanding"]["period_end"] == "2025-12-31"
    assert latest_rows["shares_outstanding"]["value"] == 10.4


def test_latest_companyfacts_rows_excludes_literal_post_as_of_filing(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_vintages(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, filed_date, form, accession, recorded_at,
                source_url
            ) VALUES(
                'AAA', 2025, 'FY', '2025-12-31', 'revenue',
                140.0, 'USD_millions', '2026-02-01', '10-K',
                '0000000000-26-000001', ?,
                'https://data.sec.gov/companyfacts.json'
            )
            """,
            (utc_now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_end, line_item, value, units,
                source_url, fetched_at, filed_date, form, accession
            ) VALUES(
                'AAA', 2025, '2025-12-31', 'revenue', 999.0,
                'USD_millions', 'https://data.sec.gov/companyfacts.json',
                ?, '2026-02-14', '10-K/A', '0000000000-26-000002'
            )
            """,
            (utc_now_iso(),),
        )

        latest_rows = _latest_companyfacts_rows(
            conn,
            ticker="AAA",
            as_of_date="2026-02-13",
        )

    assert latest_rows["revenue"]["value"] == 140.0
    assert latest_rows["revenue"]["filed_date"] == "2026-02-01"
    assert latest_rows["revenue"]["accession"] == "0000000000-26-000001"


def test_companyfacts_frame_uses_annual_cache_without_quarterly_db_blocking(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    cache_path = cfg.cache_dir / "companyfacts" / "0000000001.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "companyfacts": {
                    "facts": {
                        "us-gaap": {
                            "Revenues": {
                                "units": {
                                    "USD": [
                                        {
                                            "form": "10-K",
                                            "fy": 2025,
                                            "start": "2025-01-01",
                                            "end": "2025-12-31",
                                            "filed": "2026-02-01",
                                            "accn": "0000000000-26-000001",
                                            "val": 160_000_000.0,
                                        }
                                    ]
                                }
                            },
                            "NetCashProvidedByUsedInOperatingActivities": {
                                "units": {
                                    "USD": [
                                        {
                                            "form": "10-K",
                                            "fy": 2025,
                                            "start": "2025-01-01",
                                            "end": "2025-12-31",
                                            "filed": "2026-02-01",
                                            "accn": "0000000000-26-000001",
                                            "val": 40_000_000.0,
                                        }
                                    ]
                                }
                            },
                            "PaymentsToAcquirePropertyPlantAndEquipment": {
                                "units": {
                                    "USD": [
                                        {
                                            "form": "10-K",
                                            "fy": 2025,
                                            "start": "2025-01-01",
                                            "end": "2025-12-31",
                                            "filed": "2026-02-01",
                                            "accn": "0000000000-26-000001",
                                            "val": 14_000_000.0,
                                        }
                                    ]
                                }
                            },
                        },
                        "dei": {
                            "EntityCommonStockSharesOutstanding": {
                                "units": {
                                    "shares": [
                                        {
                                            "form": "10-K",
                                            "fy": 2025,
                                            "end": "2025-12-31",
                                            "filed": "2026-02-01",
                                            "accn": "0000000000-26-000001",
                                            "val": 10_400_000.0,
                                        }
                                    ]
                                }
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with get_db() as conn:
        for line_item, value, units in (
            ("revenue", 999.0, "USD_millions"),
            ("cfo", 90.0, "USD_millions"),
            ("capex", 10.0, "USD_millions"),
        ):
            conn.execute(
                """
                    INSERT INTO companyfacts_facts(
                        ticker, fiscal_year, period_type, period_end, line_item,
                        value, units, source_url, fetched_at, filed_date, form, accession
                    ) VALUES(?, ?, 'Q3', '2025-09-30', ?, ?, ?, 'https://data.sec.gov/q3', ?, '2025-11-01', '10-Q', '0000000000-25-000003')
                    """,
                ("AAA", 2025, line_item, value, units, utc_now_iso()),
            )

        monkeypatch.setattr(
            "app.evidence.packet_builder.resolve_cik_for_ticker",
            lambda *_args, **_kwargs: "0000000001",
        )
        frame = _companyfacts_frame(conn, ticker="AAA", as_of_date="2026-02-13")

    assert frame["rows"][-1]["year"] == 2025
    assert frame["rows"][-1]["revenue"] == 160.0
    assert frame["rows"][-1]["cfo"] == 40.0
    assert frame["rows"][-1]["capex"] == 14.0
    assert frame["rows"][-1]["shares_outstanding"] == 10.4
    assert frame["rows"][-1]["fcf"] == 26.0


def test_build_packet_surfaces_companyfacts_capital_allocation_and_rnd(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000000.json"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "AAA",
                "2026-02-13",
                json.dumps(
                    {
                        "revenue": 160.0,
                        "cfo": 40.0,
                        "capex": 14.0,
                        "fcf": 26.0,
                        "shares_outstanding": 10.4,
                    }
                ),
                utc_now_iso(),
            ),
        )
        for line_item, value, units in (
            ("revenue", 160.0, "USD_millions"),
            ("cfo", 40.0, "USD_millions"),
            ("capex", 14.0, "USD_millions"),
            ("shares_outstanding", 10.4, "shares_millions"),
            ("r_and_d_total", 12.0, "USD_millions"),
            ("share_repurchases_amount", 20.0, "USD_millions"),
            ("dividends_paid_amount", 9.0, "USD_millions"),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_end, line_item, value, units,
                    source_url, fetched_at, filed_date, form, accession
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, '2026-02-01', '10-K', '0000000000-26-000001')
                """,
                ("AAA", 2025, "2025-12-31", line_item, value, units, source_url, utc_now_iso()),
            )

    path = build_packet_for_ticker("AAA", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    fundamentals = payload["fundamentals"]

    assert fundamentals["r_and_d_total"] == 12.0
    assert fundamentals["share_repurchases_amount"] == 20.0
    assert fundamentals["dividends_paid_amount"] == 9.0
    assert fundamentals["r_and_d_intensity_latest"] == 12.0 / 160.0
    assert fundamentals["rows"][-1]["share_repurchases_amount"] == 20.0
    assert fundamentals["rows"][-1]["dividends_paid_amount"] == 9.0


def test_build_packet_surfaces_bank_like_companyfacts_fields(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "JPM",
                "2026-02-13",
                json.dumps(
                    {"revenue": 100.0, "issuer_classification": "financial", "fcf": "UNKNOWN"}
                ),
                utc_now_iso(),
            ),
        )
        for line_item, value, units in (
            ("revenue", 100.0, "USD_millions"),
            ("deposits", 2500.0, "USD_millions"),
            ("loans", 1400.0, "USD_millions"),
            ("investment_securities", 600.0, "USD_millions"),
            ("total_assets", 3900.0, "USD_millions"),
            ("allowance_for_credit_losses", 35.0, "USD_millions"),
            ("provision_for_credit_losses", 12.0, "USD_millions"),
            ("net_charge_offs", 4.0, "USD_millions"),
            ("nonaccrual_loans", 20.0, "USD_millions"),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_end, line_item, value, units,
                    source_url, fetched_at, filed_date, form, accession
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, '2026-02-01', '10-K', '0000019617-26-000001')
                """,
                ("JPM", 2025, "2025-12-31", line_item, value, units, source_url, utc_now_iso()),
            )

    path = build_packet_for_ticker("JPM", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    fundamentals = payload["fundamentals"]
    assert fundamentals["deposits"] == 2500.0
    assert fundamentals["loans"] == 1400.0
    assert fundamentals["investment_securities"] == 600.0
    assert fundamentals["total_assets"] == 3900.0
    assert fundamentals["allowance_for_credit_losses"] == 35.0
    assert fundamentals["provision_for_credit_losses"] == 12.0
    assert fundamentals["net_charge_offs"] == 4.0
    assert fundamentals["nonaccrual_loans"] == 20.0
    assert fundamentals["deposits_to_assets_latest"] == 2500.0 / 3900.0
    assert fundamentals["loans_to_deposits_latest"] == 1400.0 / 2500.0
    assert fundamentals["allowance_to_loans_latest"] == 35.0 / 1400.0
    assert fundamentals["provision_to_loans_latest"] == 12.0 / 1400.0
    assert fundamentals["net_charge_offs_to_loans_latest"] == 4.0 / 1400.0
    assert fundamentals["rows"][-1]["deposits"] == 2500.0
    assert fundamentals["rows"][-1]["loans"] == 1400.0
    assert fundamentals["rows"][-1]["deposits_to_assets"] == 2500.0 / 3900.0
    assert fundamentals["rows"][-1]["loans_to_deposits"] == 1400.0 / 2500.0
    assert fundamentals["rows"][-1]["provision_to_loans"] == 12.0 / 1400.0
    assert fundamentals["rows"][-1]["net_charge_offs_to_loans"] == 4.0 / 1400.0
    assert fundamentals["allowance_to_loans_history"] == [{"year": 2025, "value": 35.0 / 1400.0}]
    assert fundamentals["provision_to_loans_history"] == [{"year": 2025, "value": 12.0 / 1400.0}]
    assert fundamentals["net_charge_offs_to_loans_history"] == [
        {"year": 2025, "value": 4.0 / 1400.0}
    ]
    assert any(row["line_item"] == "deposits" for row in payload["financials"])
    assert any(row["line_item"] == "loans" for row in payload["financials"])
    assert any(row["line_item"] == "investment_securities" for row in payload["financials"])
    assert any(row["line_item"] == "provision_for_credit_losses" for row in payload["financials"])
    assert any(row["line_item"] == "net_charge_offs" for row in payload["financials"])


def test_build_packet_surfaces_bank_credit_ratio_history_across_years(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "JPM",
                "2026-02-13",
                json.dumps({"issuer_classification": "financial", "revenue": 100.0}),
                utc_now_iso(),
            ),
        )
        for fiscal_year, period_end, loans, allowance, provision, charge_offs in (
            (2023, "2023-12-31", 1200.0, 30.0, 10.0, 3.0),
            (2024, "2024-12-31", 1300.0, 32.5, 11.0, 3.5),
            (2025, "2025-12-31", 1400.0, 35.0, 12.0, 4.0),
        ):
            for line_item, value in (
                ("revenue", 100.0 + float(fiscal_year - 2023)),
                ("loans", loans),
                ("allowance_for_credit_losses", allowance),
                ("provision_for_credit_losses", provision),
                ("net_charge_offs", charge_offs),
            ):
                conn.execute(
                    """
                    INSERT INTO companyfacts_facts(
                        ticker, fiscal_year, period_end, line_item, value, units,
                        source_url, fetched_at, filed_date, form, accession
                    ) VALUES(?, ?, ?, ?, ?, 'USD_millions', ?, ?, '2026-02-01', '10-K', '0000019617-26-000001')
                    """,
                    ("JPM", fiscal_year, period_end, line_item, value, source_url, utc_now_iso()),
                )

    path = build_packet_for_ticker("JPM", "2026-02-13")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fundamentals = payload["fundamentals"]
    assert fundamentals["allowance_to_loans_history"] == [
        {"year": 2023, "value": 30.0 / 1200.0},
        {"year": 2024, "value": 32.5 / 1300.0},
        {"year": 2025, "value": 35.0 / 1400.0},
    ]
    assert fundamentals["provision_to_loans_history"] == [
        {"year": 2023, "value": 10.0 / 1200.0},
        {"year": 2024, "value": 11.0 / 1300.0},
        {"year": 2025, "value": 12.0 / 1400.0},
    ]
    assert fundamentals["net_charge_offs_to_loans_history"] == [
        {"year": 2023, "value": 3.0 / 1200.0},
        {"year": 2024, "value": 3.5 / 1300.0},
        {"year": 2025, "value": 4.0 / 1400.0},
    ]


def test_build_packet_uses_filing_financial_fallback_for_bank_credit_metrics(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "BAC",
                "2026-02-13",
                json.dumps(
                    {
                        "issuer_classification": "financial",
                        "loans": "UNKNOWN",
                        "allowance_for_credit_losses": "UNKNOWN",
                    }
                ),
                utc_now_iso(),
            ),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url, local_path, hash, ingested_as_of, status, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "0000070858",
                "BAC",
                "0000070858-26-000001",
                "10-K",
                "2026-02-10",
                "2025-12-31",
                "https://www.sec.gov/Archives/edgar/data/70858/test.htm",
                "",
                "h",
                "2026-02-13",
                "parsed",
                utc_now_iso(),
                utc_now_iso(),
            ),
        )
        filing_id = conn.execute("SELECT id FROM filings WHERE ticker = 'BAC'").fetchone()["id"]
        for line_item, value in (
            ("loans", 1500.0),
            ("allowance_for_credit_losses", 22.0),
            ("provision_for_credit_losses", 6.0),
            ("net_charge_offs", 2.0),
        ):
            conn.execute(
                """
                INSERT INTO financials(filing_id, statement_type, line_item, value, units, period, source_url, snippet, created_at)
                VALUES(?, 'balance_sheet', ?, ?, 'USD_millions', '2025-12-31', 'https://www.sec.gov/Archives/edgar/data/70858/test.htm', ?, ?)
                """,
                (filing_id, line_item, value, f"{line_item}: {value}", utc_now_iso()),
            )

    path = build_packet_for_ticker("BAC", "2026-02-13")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fundamentals = payload["fundamentals"]
    assert fundamentals["loans"] == 1500.0
    assert fundamentals["allowance_for_credit_losses"] == 22.0
    assert fundamentals["provision_for_credit_losses"] == 6.0
    assert fundamentals["net_charge_offs"] == 2.0
    assert fundamentals["allowance_to_loans_latest"] == 22.0 / 1500.0
    assert fundamentals["provision_to_loans_latest"] == 6.0 / 1500.0
    assert fundamentals["net_charge_offs_to_loans_latest"] == 2.0 / 1500.0


def test_build_packet_computes_fundamentals_from_companyfacts_when_missing(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000072971.json"
    with get_db() as conn:
        for line_item, value, units in (
            ("revenue", 90.0, "USD_millions"),
            ("deposits", 1800.0, "USD_millions"),
            ("loans", 1000.0, "USD_millions"),
            ("total_assets", 2500.0, "USD_millions"),
            ("allowance_for_credit_losses", 18.0, "USD_millions"),
            ("provision_for_credit_losses", 7.0, "USD_millions"),
            ("net_charge_offs", 2.5, "USD_millions"),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_end, line_item, value, units,
                    source_url, fetched_at, filed_date, form, accession
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, '2026-02-01', '10-K', '0000072971-26-000001')
                """,
                ("WFC", 2025, "2025-12-31", line_item, value, units, source_url, utc_now_iso()),
            )

    path = build_packet_for_ticker("WFC", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    fundamentals = payload["fundamentals"]
    assert fundamentals["issuer_classification"] == "financial"
    assert fundamentals["loans"] == 1000.0
    assert fundamentals["allowance_for_credit_losses"] == 18.0
    assert fundamentals["provision_for_credit_losses"] == 7.0
    assert fundamentals["net_charge_offs"] == 2.5
    assert fundamentals["allowance_to_loans"] == 18.0 / 1000.0


def test_build_packet_records_upstream_resolution_metadata_for_backfilled_fcf_and_bank_net_debt(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json, created_at)
            VALUES(?, ?, ?, '[]', ?)
            """,
            (
                "JPM",
                "2026-02-13",
                json.dumps(
                    {
                        "issuer_classification": "financial",
                        "revenue": 100.0,
                        "fcf": "UNKNOWN",
                        "net_debt": "UNKNOWN",
                    }
                ),
                utc_now_iso(),
            ),
        )

    monkeypatch.setattr(
        "app.evidence.packet_builder._resolve_local_facts_backfill",
        lambda **_kwargs: {
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
            "derived_from": ["companyfacts.cache.JPM"],
            "fcf_value": 15.0,
            "metric_support": {
                "fcf": {
                    "status": "OK",
                    "resolution": "DERIVED",
                    "reason_code": "COMPANYFACTS_CFO_CAPEX_HIT",
                    "bridge_context": {"formula": "FCF = CFO - CapEx"},
                },
                "net_debt": {
                    "status": "UNKNOWN",
                    "resolution": "UNAVAILABLE",
                    "reason_code": "BANK_SPECIFIC_HANDLING",
                    "issuer_classification": "financial",
                },
            },
        },
    )

    path = build_packet_for_ticker("JPM", "2026-02-13")
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    resolution = payload["fundamentals"]["upstream_evidence_resolution"]
    assert resolution["fcf"]["resolution"] == "DERIVED"
    assert resolution["fcf"]["reason_code"] == "COMPANYFACTS_CFO_CAPEX_HIT"
    assert resolution["net_debt"]["reason_code"] == "BANK_SPECIFIC_HANDLING"
    fcf_fact = next(row for row in payload["extracted_facts"] if row["fact_type"] == "fcf")
    assert fcf_fact["value"]["bridge_context"]["formula"] == "FCF = CFO - CapEx"


def test_ten_year_share_count_growth_is_split_adjusted_only_for_a_filed_split():
    """Migrated 2026-09-29 (review H5). A doubling in 2020 was taken for a 2-for-1
    split and only 2020-2023 measured. It is a split only when a filed split
    ratio corroborates it; then the earlier counts are restated (10.0 -> 20.0)
    and the whole window measured: 20.0 in 2015 to 19.0 in 2023 over 8 years.
    Unfiled, a doubling could as well be issuance, so the rate is UNKNOWN; so is
    a 50% raise (10 -> 15) that sits on the 3-for-2 factor."""
    from app.evidence.packet_builder import UNKNOWN, _share_count_cagr

    series = [(2015, 10.0), (2016, 10.0), (2017, 10.0), (2018, 10.0), (2019, 10.0),
              (2020, 20.0), (2021, 19.6), (2022, 19.3), (2023, 19.0)]
    split_2020 = [{"year": 2020, "value": 2.0, "derived_from": ["split.2020"]}]
    value = _share_count_cagr(series, 10, split_2020)
    assert abs(value - ((19.0 / 20.0) ** (1.0 / 8.0) - 1.0)) < 1e-12
    assert _share_count_cagr(series, 10) == UNKNOWN

    raise_window = [(2020, 10.0), (2021, 10.0), (2022, 15.0), (2023, 15.2)]
    assert _share_count_cagr(raise_window, 10) == UNKNOWN
    assert _share_count_cagr([(2023, 10.0)], 10) == UNKNOWN
    smooth = [(2020, 10.0), (2021, 10.0), (2022, 9.5)]
    assert abs(_share_count_cagr(smooth, 10) - ((9.5 / 10.0) ** 0.5 - 1.0)) < 1e-12
    # A break older than the measured window does not touch the rate.
    old_break = [(2010, 5.0), (2011, 10.0), (2021, 10.0), (2022, 9.9), (2023, 9.8)]
    assert abs(_share_count_cagr(old_break, 2) - ((9.8 / 10.0) ** 0.5 - 1.0)) < 1e-12
