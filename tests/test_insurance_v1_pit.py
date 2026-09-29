from __future__ import annotations

import json

from app.db import get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def test_real_v1_insurance_signal_excludes_filed_after_asof_facts(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    ticker = "IPIT"
    cik = "0000000042"
    accession = "0000000042-26-000001"
    now = utc_now_iso()
    filing_path = tmp_path / "ipit-10k.txt"
    filing_path.write_text(
        "We write property and casualty insurance and reinsurance.",
        encoding="utf-8",
    )
    submissions_path = cfg.cache_dir / "submissions" / f"{cik}.json"
    submissions_path.parent.mkdir(parents=True, exist_ok=True)
    submissions_path.write_text(
        json.dumps(
            {
                "name": "Example Property and Casualty Insurance Corp",
                "tickers": [ticker],
                "exchanges": ["NYSE"],
            }
        ),
        encoding="utf-8",
    )
    scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {
            "current_price": 8.0,
            "dcf_base": 12.0,
            "epv_adjusted": 11.0,
            "gate_action": "PROCEED",
        },
        "quality_context": {"gate_action": "PROCEED"},
    }
    with get_db() as conn:
        conn.execute(
            "INSERT INTO companies(ticker, cik, name, created_at) VALUES(?, ?, ?, ?)",
            (
                ticker,
                cik,
                "Example Property and Casualty Insurance Corp",
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO sector_inference(
                ticker, as_of_date, inferred_sector, score, derived_from,
                created_at
            )
            VALUES(?, '2026-03-15', 'insurance', 1.0, '[]', ?)
            """,
            (ticker, now),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            )
            VALUES(
                ?, ?, ?, '10-K', '2026-02-15', '2025-12-31',
                'https://www.sec.gov/Archives/edgar/data/42/filing.htm',
                ?, 'parsed', ?, ?
            )
            """,
            (cik, ticker, accession, str(filing_path), now, now),
        )
        scorecard_cursor = conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at
            )
            VALUES(?, '2026-03-15', 'scorecard', '{}', ?, '[]', ?)
            """,
            (ticker, json.dumps(scorecard), now),
        )
        facts = [
            (2025, "2025-12-31", "equity", 1000.0, "2026-02-15", accession),
            (
                2025,
                "2025-12-31",
                "shares_outstanding",
                100.0,
                "2026-02-15",
                accession,
            ),
            (2025, "2025-12-31", "net_income", 120.0, "2026-02-15", accession),
            (2024, "2024-12-31", "net_income", 100.0, "2025-02-15", "prior-2025"),
            (2023, "2023-12-31", "net_income", 80.0, "2024-02-15", "prior-2024"),
            (
                2026,
                "2026-02-28",
                "equity",
                9000.0,
                "2026-04-15",
                "future-amendment",
            ),
            (
                2026,
                "2026-02-28",
                "shares_outstanding",
                100.0,
                "2026-04-15",
                "future-amendment",
            ),
            (
                2026,
                "2026-02-28",
                "net_income",
                900.0,
                "2026-04-15",
                "future-amendment",
            ),
        ]
        for fiscal_year, period_end, line_item, value, filed_date, fact_accession in facts:
            units = "shares_millions" if line_item == "shares_outstanding" else "USD_millions"
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession
                )
                VALUES(
                    ?, ?, 'FY', ?, ?, ?, ?,
                    'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json',
                    ?, ?, '10-K', ?
                )
                """,
                (
                    ticker,
                    fiscal_year,
                    period_end,
                    line_item,
                    value,
                    units,
                    now,
                    filed_date,
                    fact_accession,
                ),
            )

    from tests.financial_integrity_helpers import authorize_valuation_rows

    authorize_valuation_rows(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        run_id="autonomous_sector_insurance_ipit_authorized",
        row_ids=[int(scorecard_cursor.lastrowid)],
        issuer_ciks={ticker: cik},
    )
    monkeypatch.setattr(
        "app.alpha.signal_assembler.scan_filing_risks",
        lambda *_args, **_kwargs: {
            "status": "NO_FILING",
            "competitive_disruption": "UNKNOWN",
            "secular_decline": "UNKNOWN",
            "regulatory_legal": "UNKNOWN",
            "customer_concentration": "UNKNOWN",
            "summary": "",
        },
    )
    from app.alpha.signal_assembler import assemble_signal_packet

    packet = assemble_signal_packet(
        ticker,
        as_of_date="2026-03-15",
        current_price_override=8.0,
        issuer_cik=cik,
        issuer_aliases=(ticker,),
        db_path=cfg.db_path,
        cfg=cfg,
    )

    assert packet.insurance_method == "insurance_common"
    assert packet.insurance_value == 10.0
    assert packet.insurance_valuation["average_net_income"] == 100.0
    assert packet.insurance_valuation["source_references"] == {
        "annual_companyfacts_rows": 5,
        "latest_equity_period": "2025-12-31",
        "latest_shares_period": "2025-12-31",
        "latest_equity_filed_date": "2026-02-15",
        "latest_shares_filed_date": "2026-02-15",
        "net_income_year_count": 3,
    }
