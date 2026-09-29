from __future__ import annotations

import json

from app.alpha.report_writer import generate_alpha_report
from app.alpha.schemas import SectorAlphaReport
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


def test_alpha_report_uses_canonical_packet_and_excludes_post_asof_facts(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at
            )
            VALUES('AAA', '2026-03-01', 'scorecard', '{}', ?, '[]', ?)
            """,
            (
                json.dumps(
                    {
                        "pricing_zone_detail": {
                            "current_price": 999.0,
                            "dcf_base": 1_999.0,
                            "gate_action": "PROCEED",
                        }
                    }
                ),
                utc_now_iso(),
            ),
        )
        for fiscal_year, period_end, filed_date, values in (
            (
                2024,
                "2024-12-31",
                "2025-02-01",
                {
                    "revenue": 100.0,
                    "operating_income": 20.0,
                    "net_income": 15.0,
                    "cfo": 30.0,
                    "capex": 5.0,
                },
            ),
            (
                2025,
                "2025-12-31",
                "2026-03-01",
                {
                    "revenue": 9_999.0,
                    "operating_income": 8_888.0,
                    "net_income": 7_777.0,
                    "cfo": 6_666.0,
                    "capex": 5_555.0,
                },
            ),
        ):
            conn.executemany(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession
                )
                VALUES(
                    'AAA', ?, 'FY', ?, ?, ?, 'USD_millions',
                    'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                    ?, ?, '10-K', ?
                )
                """,
                [
                    (
                        fiscal_year,
                        period_end,
                        line_item,
                        value,
                        utc_now_iso(),
                        filed_date,
                        ("0000000001-25-000001" if fiscal_year == 2024 else "0000000001-26-000001"),
                    )
                    for line_item, value in values.items()
                ],
            )

    report = SectorAlphaReport(
        sector="software",
        total_candidates=1,
        rounds=[],
        winner="AAA",
        winner_thesis="Canonical packet wins.",
        winner_conviction="HIGH",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Execution",
        falsification_trigger="Cash flow weakens",
        time_horizon="12 months",
        signal_packets={
            "AAA": {
                "ticker": "AAA",
                "current_price": 50.0,
                "current_price_as_of_date": "2026-02-13",
                "dcf_value": 75.0,
                "epv_value": 65.0,
                "gate_verdict": "PROCEED",
                "moat_classification": "NARROW",
                "moat_score": 3,
                "downside_risk_class": "MODERATE",
                "confidence_class": "MODERATE",
                "valuation_headwinds": [],
                "valuation_supports": ["CASH_GENERATION"],
            }
        },
    )

    path = generate_alpha_report(
        report,
        tmp_path / "alpha_report.md",
        as_of_date="2026-02-13",
    )
    text = path.read_text(encoding="utf-8")

    assert "**Financial evidence as of:** 2026-02-13" in text
    assert "| Current Price | $50.00 |" in text
    assert "| DCF Intrinsic | $75.00 |" in text
    assert "| 2024 | 100 | 20 | 20.0% | 15 | 30 | 25 |" in text
    assert "$999.00" not in text
    assert "$1,999.00" not in text
    assert "9,999" not in text
