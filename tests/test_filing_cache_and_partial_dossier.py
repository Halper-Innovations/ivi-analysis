from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.dossier.collector import DossierStage1Filing
from app.dossier.filing_cache import filing_cache_path, warm_filing_cache
from app.dossier.runner import run_dossier_for_peer_set
from app.db import get_db
from app.ingest.filings import _download_filing, _upsert_filing
from app.ingest.sec_client import FilingStub


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _get_config
    from app.db import init_db

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _companyfacts_cache(tmp_path: Path, ticker: str) -> str:
    payload = {
        "companyfacts": {
            "entityName": f"{ticker} Corp",
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 100.0}]
                        }
                    },
                    "GrossProfit": {
                        "units": {
                            "USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 70.0}]
                        }
                    },
                    "OperatingIncomeLoss": {
                        "units": {
                            "USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 20.0}]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 15.0}]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 30.0}]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 5.0}]}
                    },
                    "ResearchAndDevelopmentExpense": {
                        "units": {
                            "USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 18.0}]
                        }
                    },
                    "ContractWithCustomerLiability": {
                        "units": {
                            "USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 40.0}]
                        }
                    },
                },
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 10.0}]
                        }
                    }
                },
            },
        }
    }
    path = tmp_path / f"{ticker}_companyfacts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_partial_dossier_is_written_when_filing_body_missing(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    filing = FilingStub(
        cik="1",
        accession="0001-2025-000001",
        accession_nodash="00012025000001",
        form_type="10-K",
        filing_date=date.fromisoformat("2025-02-01"),
        period_end="2024-12-31",
        primary_document="doc.htm",
        primary_doc_url="https://www.sec.gov/doc.htm",
        filing_index_url="https://www.sec.gov/index.json",
    )
    monkeypatch.setattr(
        "app.dossier.runner._collect_stage1_for_ticker",
        lambda *, ticker, as_of_date, years_back, min_annual_filings=None: (
            [DossierStage1Filing(ticker=ticker, cik="1", filing=filing, local_path=None)],
            {
                "ticker": ticker,
                "query": {"as_of_date": as_of_date, "years_back": years_back},
                "selected_accessions": [],
            },
        ),
    )
    monkeypatch.setattr(
        "app.dossier.runner.materialize_and_parse_docket_stage1",
        lambda *, stage1, as_of_date: [
            type(
                "FilingRow",
                (),
                {
                    "ticker": stage1[0].ticker,
                    "cik": stage1[0].cik,
                    "accession": stage1[0].filing.accession,
                    "form_type": stage1[0].filing.form_type,
                    "filing_date": stage1[0].filing.filing_date.isoformat(),
                    "period_end": stage1[0].filing.period_end,
                    "primary_doc_url": stage1[0].filing.primary_doc_url,
                    "local_path": None,
                    "filing_id": 1,
                },
            )()
        ],
    )
    monkeypatch.setattr("app.dossier.runner.ensure_all_facts", lambda ticker, years_back=10: None)
    now = "2026-03-23T12:00:00+00:00"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                'AAA', 2024, 'FY', '2024-12-31', 'revenue', 999.0,
                'USD_millions', 'https://data.sec.gov/current', ?,
                '2025-03-01', '10-K/A', '0001-2025-000001-A'
            )
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_vintages(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, filed_date, form, accession, recorded_at,
                issuer_cik, source_url
            )
            VALUES(
                'AAA', 2024, 'FY', '2024-12-31', 'revenue', 100.0,
                'USD_millions', '2025-02-01', '10-K',
                '0001-2025-000001', ?, '1',
                'https://data.sec.gov/original'
            )
            """,
            (now,),
        )
    monkeypatch.setattr("app.dossier.runner.ensure_valuation", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.dossier.runner.build_packet_for_ticker", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.dossier.runner.run_synthesis_for_ticker", lambda *args, **kwargs: "")

    summary = run_dossier_for_peer_set(
        tickers=["AAA"],
        as_of_date="2026-03-23",
        run_id="partial_dossier_test",
        workers=1,
    )

    assert summary["partial_count"] == 1
    assert summary["full_count"] == 0
    payload = json.loads(
        (cfg.dossiers_dir / "partial_dossier_test" / "AAA" / "dossier.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["dossier_quality"] == "PARTIAL"
    assert payload["filing_body_cached"] is False
    revenue = next(item for item in payload["items"] if item["metric"] == "revenue")
    assert revenue["value"] == 100.0
    assert revenue["filing_accession"] == "0001-2025-000001"
    assert revenue["filing_date"] == "2025-02-01"
    assert revenue["period_end"] == "2024-12-31"


def test_warm_filing_cache_writes_primary_document(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    taxonomy_path = cfg.sector_taxonomy_path
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_path.write_text("ticker,sector\nAAA,Enterprise Software\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.dossier.filing_cache.resolve_cik_for_ticker",
        lambda ticker, cfg=None, refresh_if_missing=True: "1",
    )

    filing = FilingStub(
        cik="1",
        accession="0001-2025-000001",
        accession_nodash="00012025000001",
        form_type="10-K",
        filing_date=date.fromisoformat("2025-02-01"),
        period_end="2024-12-31",
        primary_document="doc.htm",
        primary_doc_url="https://www.sec.gov/doc.htm",
        filing_index_url="https://www.sec.gov/index.json",
    )

    class FakeSecClient:
        def list_filings_window(self, cik, *, start_date, end_date, forms):
            return [filing]

        def list_cached_filings_window(self, cik, *, start_date, end_date, forms):
            return [filing]

        def download_bytes(self, url, *, use_cache=True):
            return b"<html>cached filing</html>"

    monkeypatch.setattr("app.dossier.filing_cache.SecClient", FakeSecClient)

    summary = warm_filing_cache(sector="Enterprise Software", years=2, cfg=cfg)

    assert summary["cached"] == 1
    assert filing_cache_path(cik="1", accession="0001-2025-000001", cfg=cfg).exists()


def test_download_filing_offline_returns_none_without_network(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    filing = FilingStub(
        cik="1",
        accession="0001-2025-000001",
        accession_nodash="00012025000001",
        form_type="10-K",
        filing_date=date.fromisoformat("2025-02-01"),
        period_end="2024-12-31",
        primary_document="doc.htm",
        primary_doc_url="https://www.sec.gov/doc.htm",
        filing_index_url="https://www.sec.gov/index.json",
    )

    class FakeSecClient:
        def download_bytes(self, url, *, use_cache=True):
            raise AssertionError("network download should not be attempted in offline mode")

    with get_db() as conn:
        filing_id = _upsert_filing(conn, "AAA", filing, "2026-03-23")
        local_path = _download_filing(conn, FakeSecClient(), filing_id, filing)

    assert local_path is None
