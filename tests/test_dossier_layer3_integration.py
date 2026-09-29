from __future__ import annotations

import json
from datetime import date

from app.db import get_db, init_db, utc_now_iso
from app.dossier.runner import _build_ticker_payload_from_stage1
from tests.financial_integrity_helpers import materialized_no_split_proof


class _FakeStage1:
    def __init__(self, filing):
        self.ticker = "AAA"
        self.cik = "0000000001"
        self.filing = filing
        self.local_path = "/tmp/example"


class _FakeFiling:
    accession = "0000000001-26-000001"
    form_type = "10-K"
    filing_date = date(2026, 2, 13)
    period_end = "2025-12-31"
    primary_doc_url = "https://www.sec.gov/example"
    filing_id = 1
    ticker = "AAA"


def _init_temp_db(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache",
        lambda http=None: {},
    )
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_dossier_build_appends_layer3_synthesis_section(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    accession = "0000000001-26-000001"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '0000000001', 'AAA Fixture', ?)
            """,
            (utc_now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            ) VALUES
                ('AAA', 2025, 'FY', '2025-12-31', 'revenue', 100.0, 'USD_millions', ?, ?, '2026-02-13', '10-K', ?),
                ('AAA', 2025, 'FY', '2025-12-31', 'cfo', 20.0, 'USD_millions', ?, ?, '2026-02-13', '10-K', ?),
                ('AAA', 2025, 'FY', '2025-12-31', 'capex', 5.0, 'USD_millions', ?, ?, '2026-02-13', '10-K', ?),
                ('AAA', 2025, 'FY', '2025-12-31', 'shares_outstanding', 10.0, 'shares_millions', ?, ?, '2026-02-13', '10-K', ?),
                ('AAA', 2025, 'FY', '2025-12-31', 'cash', 8.0, 'USD_millions', ?, ?, '2026-02-13', '10-K', ?),
                ('AAA', 2025, 'FY', '2025-12-31', 'total_debt', 11.0, 'USD_millions', ?, ?, '2026-02-13', '10-K', ?),
                ('AAA', 2025, 'FY', '2025-12-31', 'equity', 40.0, 'USD_millions', ?, ?, '2026-02-13', '10-K', ?)
            """,
            tuple(value for _ in range(7) for value in (source_url, utc_now_iso(), accession)),
        )

    companyfacts = {
        "cik": "0000000001",
        "entityName": "AAA Fixture",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "val": 10_000_000.0,
                                "end": "2025-12-31",
                                "filed": "2026-02-13",
                                "form": "10-K",
                                "fy": 2025,
                                "accn": accession,
                            }
                        ]
                    }
                }
            }
        },
    }
    companyfacts_dir = cfg.cache_dir / "companyfacts"
    companyfacts_dir.mkdir(parents=True, exist_ok=True)
    (companyfacts_dir / "0000000001.json").write_text(
        json.dumps(
            {
                "cik": "0000000001",
                "retrieved_at": utc_now_iso(),
                "source_url": source_url,
                "http_status": 200,
                "size_bytes": len(json.dumps(companyfacts, sort_keys=True).encode("utf-8")),
                "companyfacts": companyfacts,
            }
        ),
        encoding="utf-8",
    )
    submissions_dir = cfg.cache_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    (submissions_dir / "0000000001.json").write_text(
        json.dumps(
            {
                "tickers": ["AAA"],
                "exchanges": ["NYSE"],
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "filingDate": ["2026-02-13"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    price_dir = cfg.outputs_dir / "prices" / "dossier_layer3_test"
    price_dir.mkdir(parents=True, exist_ok=True)
    (price_dir / "AAA.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "requested_as_of_date": "2026-02-13",
                "status": "OK",
                "snapshot": {
                    "ticker": "AAA",
                    "as_of_date": "2026-02-13",
                    "price": 10.0,
                    "currency": "USD",
                    "source": "fixture_quote",
                    "retrieved_at": "2026-02-13T12:00:00+00:00",
                    "url": "https://example.test/quotes/AAA",
                    "confidence": "HIGH",
                    "price_basis": "UNADJUSTED",
                    "raw_price": 10.0,
                    "split_adjustment_factor": 1.0,
                    "no_intervening_split_proof": materialized_no_split_proof(
                        ticker="AAA",
                        period_start="2025-12-31",
                        period_end="2026-02-13",
                    ),
                },
                "diagnostic": {
                    "result": {
                        "reason_code": "CACHE_HIT",
                        "reason_detail": "Literal run-scoped fixture quote.",
                    },
                    "output_fields": {
                        "current_price": 10.0,
                        "price_asof_used": "2026-02-13",
                        "price_source": "fixture_quote",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "app.dossier.runner.materialize_and_parse_docket_stage1",
        lambda **kwargs: [kwargs["stage1"][0].filing],
    )
    monkeypatch.setattr(
        "app.dossier.runner.read_filing_text", lambda filing: "business risk factors md&a"
    )
    monkeypatch.setattr("app.dossier.runner.segment_10k_sections", lambda text: [])
    monkeypatch.setattr(
        "app.dossier.runner.extract_annual_items",
        lambda **kwargs: [
            {
                "ticker": "AAA",
                "year": 2025,
                "metric": "revenue",
                "value": 100.0,
                "units": "USD_millions",
                "section_label": "financial_statements",
                "source_url": source_url,
                "snippet": "Revenue 100",
                "derived_from": ["financials.revenue"],
                "citations": [
                    {
                        "source_url": source_url,
                        "snippet": "Revenue 100",
                        "section_label": "financial_statements",
                    }
                ],
                "filing_accession": accession,
                "filing_date": "2026-02-13",
                "period_end": "2025-12-31",
            },
            {
                "ticker": "AAA",
                "year": 2025,
                "metric": "cfo",
                "value": 20.0,
                "units": "USD_millions",
                "section_label": "financial_statements",
                "source_url": source_url,
                "snippet": "CFO 20",
                "derived_from": ["financials.cfo"],
                "citations": [
                    {
                        "source_url": source_url,
                        "snippet": "CFO 20",
                        "section_label": "financial_statements",
                    }
                ],
                "filing_accession": accession,
                "filing_date": "2026-02-13",
                "period_end": "2025-12-31",
            },
            {
                "ticker": "AAA",
                "year": 2025,
                "metric": "capex",
                "value": 5.0,
                "units": "USD_millions",
                "section_label": "financial_statements",
                "source_url": source_url,
                "snippet": "Capex 5",
                "derived_from": ["financials.capex"],
                "citations": [
                    {
                        "source_url": source_url,
                        "snippet": "Capex 5",
                        "section_label": "financial_statements",
                    }
                ],
                "filing_accession": accession,
                "filing_date": "2026-02-13",
                "period_end": "2025-12-31",
            },
        ],
    )
    monkeypatch.setattr(
        "app.dossier.runner.build_time_series",
        lambda items: {
            "rows": [
                {
                    "year": 2025,
                    "revenue": 100.0,
                    "cfo": 20.0,
                    "capex": 5.0,
                    "fcf": 15.0,
                    "shares_outstanding": 10.0,
                    "net_debt": 3.0,
                }
            ],
            "standardized_rows": [
                {
                    "year": 2025,
                    "revenue": 100.0,
                    "gross_profit": "UNKNOWN",
                    "operating_income": "UNKNOWN",
                    "net_income": "UNKNOWN",
                    "cfo": 20.0,
                    "capex": 5.0,
                    "fcf": 15.0,
                    "shares_outstanding": 10.0,
                    "shares_yoy_change": "UNKNOWN",
                    "net_debt": 3.0,
                }
            ],
            "standardized_row_traces": {},
            "derived_signals": [],
        },
    )
    monkeypatch.setattr("app.dossier.runner.ensure_all_facts", lambda *args, **kwargs: None)

    payload = _build_ticker_payload_from_stage1(
        ticker="AAA",
        as_of_date="2026-02-13",
        run_id="dossier_layer3_test",
        stage1=[_FakeStage1(_FakeFiling())],
        years_back=10,
    )
    assert payload is not None
    md_path = cfg.dossiers_dir / "dossier_layer3_test" / "AAA" / "dossier.md"
    text = md_path.read_text(encoding="utf-8")
    assert "## Layer 3: Qualitative Synthesis" in text
    assert "### Verdict" in text
