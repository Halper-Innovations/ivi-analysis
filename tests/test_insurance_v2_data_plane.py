from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.alpha.schemas import SolvencyAssessment
from app.config import AppConfig
from app.db import connect, init_db, utc_now_iso


def _config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / "configured-global"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "poison-global.db",
        cache_dir=cache_dir,
        safe_mode=True,
    )


def _seed_injected_insurer(
    db_path: Path,
    *,
    cfg: AppConfig,
    filing_path: Path,
) -> None:
    conn = connect(db_path, cfg=cfg)
    try:
        init_db(conn=conn)
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('ORD', '0000000042',
                   'Example Property and Casualty Insurance Corporation Common Stock', ?)
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO sector_inference(
                ticker, as_of_date, inferred_sector, score, derived_from, created_at
            ) VALUES('ORD', '2026-02-01', 'insurance', 1.0, '[]', ?)
            """,
            (now,),
        )
        # The issuer filing is deliberately cached under an ADR alias. Known
        # CIK scope must find it without falling back to ticker-global reads.
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            ) VALUES(
                '0000000042', 'ADR', 'issuer-2025', '10-K', '2026-02-20',
                '2025-12-31', 'https://www.sec.gov/Archives/issuer-2025', ?,
                'OK', ?, ?
            )
            """,
            (str(filing_path), now, now),
        )
        source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json"
        # The current equity row is a future restatement and must not be
        # visible at the replay date. Its earlier vintage is the PIT value.
        for line_item, value, filed_date, accession in (
            ("equity", 5000.0, "2026-08-01", "future-amendment"),
            ("shares_outstanding", 100.0, "2026-03-01", "original"),
            ("net_income", 100.0, "2026-03-01", "original"),
        ):
            conn.execute(
                """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession
                ) VALUES('ADR', 2025, 'FY', '2025-12-31', ?, ?, 'USD', ?, ?, ?,
                         '10-K', ?)
                """,
                (line_item, value, source_url, now, filed_date, accession),
            )
        conn.execute(
            """
            INSERT INTO companyfacts_vintages(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, filed_date, form, accession, recorded_at,
                issuer_cik, source_url
            ) VALUES(
                'ADR', 2025, 'FY', '2025-12-31', 'equity', 1000.0, 'USD',
                '2026-03-01', '10-K', 'original', ?, '0000000042', ?
            )
            """,
            (now, source_url),
        )
        scorecard = {
            "pricing_zone": "MARGIN_OF_SAFETY",
            "pricing_zone_detail": {
                "current_price": 999.0,
                "dcf_base": 9999.0,
                "epv_adjusted": 8888.0,
            },
        }
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at
            ) VALUES('ORD', '2026-03-01', 'scorecard', '{}', ?, '[]', ?)
            """,
            (json.dumps(scorecard), now),
        )
        conn.commit()
    finally:
        conn.close()


def test_v2_insurance_packet_is_injected_issuer_bound_pit_and_read_only(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    db_path = tmp_path / "injected.db"
    filing_path = tmp_path / "issuer-10k.html"
    filing_path.write_text(
        """
        <html><body>
        We write property and casualty insurance and maintain loss reserves.
        Our combined ratio was 92.0% and our loss ratio was 61.0%.
        </body></html>
        """,
        encoding="utf-8",
    )
    submissions = cfg.cache_dir / "submissions"
    submissions.mkdir(parents=True)
    (submissions / "0000000042.json").write_text(
        json.dumps(
            {
                "name": "Example Property and Casualty Insurance Corporation",
                "tickers": ["ORD"],
                "exchanges": ["NYSE"],
            }
        ),
        encoding="utf-8",
    )
    _seed_injected_insurer(db_path, cfg=cfg, filing_path=filing_path)

    # Every v2 reader receives db_path. Any configured/global DB access or
    # persistence attempt is a regression.
    with ExitStack() as stack:
        for target in (
            "app.insurance.sources.get_db",
            "app.insurance.valuation.get_db",
            "app.insurance.packet.get_db",
        ):
            stack.enter_context(
                patch(target, side_effect=AssertionError("global DB access is forbidden"))
            )
        from app.insurance.packet import build_insurance_packet

        packet = build_insurance_packet(
            "ORD",
            as_of_date="2026-06-30",
            scorecard={
                "pricing_zone_detail": {"current_price": 999.0},
            },
            persist=True,
            pipeline_version="v2",
            current_price_override=7.0,
            issuer_cik="42",
            aliases=("ORD", "ADR"),
            db_path=db_path,
            cfg=cfg,
        )
        adr_packet = build_insurance_packet(
            "ADR",
            as_of_date="2026-06-30",
            scorecard={},
            persist=True,
            pipeline_version="v2",
            current_price_override=3.5,
            issuer_cik="42",
            aliases=("ADR", "ORD"),
            db_path=db_path,
            cfg=cfg,
        )

    assert packet["pipeline_version"] == "v2"
    assert packet["evidence_policy"] == "issuer_bound_fixed_asof_primary_source_v2"
    assert packet["routing"]["security_type"] == "common"
    assert packet["routing"]["issuer_type"] == "insurance_underwriter"
    assert packet["routing"]["filing_evidence"]["accession"] == "issuer-2025"
    assert packet["valuation"]["model_status"] == "OK"
    assert packet["valuation"]["adjusted_book_value_per_share"] == 10.0
    assert packet["valuation"]["valuation_anchor"] == 10.0
    assert packet["valuation"]["current_price"] == 7.0
    assert packet["operating_metrics"]["combined_ratio"] == 0.92
    assert adr_packet["model_status"] == "NEEDS_DATA"
    assert adr_packet["routing"]["security_type"] == "SECURITY_TYPE_UNKNOWN"
    assert "NON_PRIMARY_SECURITY_RATIO_UNRESOLVED" in adr_packet["model_blockers"]
    assert adr_packet["valuation"].get("valuation_anchor") is None

    conn = connect(db_path, cfg=cfg)
    try:
        persisted = conn.execute(
            """
            SELECT method FROM valuations
            WHERE method IN ('security_routing', 'insurance_common', 'insurance_packet')
            """
        ).fetchall()
    finally:
        conn.close()
    assert persisted == []
    assert not cfg.db_path.exists()

    conn = connect(db_path, cfg=cfg)
    try:
        conn.execute("DELETE FROM companyfacts_facts WHERE line_item = 'shares_outstanding'")
        conn.commit()
    finally:
        conn.close()
    with patch(
        "app.insurance.packet.get_db",
        side_effect=AssertionError("v2 NEEDS_DATA must remain read-only"),
    ):
        needs_data = build_insurance_packet(
            "ORD",
            as_of_date="2026-06-30",
            scorecard={},
            persist=True,
            pipeline_version="v2",
            current_price_override=7.0,
            issuer_cik="42",
            aliases=("ORD", "ADR"),
            db_path=db_path,
            cfg=cfg,
        )
    assert needs_data["model_status"] == "NEEDS_DATA"
    assert "MISSING_SHARES_OUTSTANDING" in needs_data["model_blockers"]


def test_v2_signal_assembler_threads_repaired_price_and_injected_scope(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    db_path = tmp_path / "signal-injected.db"
    filing_path = tmp_path / "issuer-10k.txt"
    filing_path.write_text("We write property and casualty insurance.", encoding="utf-8")
    _seed_injected_insurer(db_path, cfg=cfg, filing_path=filing_path)

    insurance_packet = {
        "routing": {"security_type": "common", "issuer_type": "insurance_underwriter"},
        "valuation": {"model_status": "OK", "method": "insurance_common", "valuation_anchor": 10.0},
        "model_status": "OK",
        "model_blockers": [],
        "model_fit_warnings": [],
        "generic_valuation_valid": False,
    }
    with (
        patch(
            "app.alpha.signal_assembler.build_insurance_packet",
            return_value=insurance_packet,
        ) as build,
        patch(
            "app.alpha.signal_assembler.scan_filing_risks",
            return_value={"status": "NO_FILING"},
        ),
        patch("app.alpha.signal_assembler.detect_anomalies", return_value=[]),
        patch(
            "app.alpha.signal_assembler.assess_solvency",
            return_value=SolvencyAssessment(solvency_risk="LOW"),
        ),
        patch("app.alpha.signal_assembler.analyze_method_tensions", return_value={}),
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        signal = assemble_signal_packet(
            "ORD",
            as_of_date="2026-06-30",
            pipeline_version="v2",
            current_price_override=7.0,
            issuer_cik="42",
            issuer_aliases=("ORD", "ADR"),
            db_path=db_path,
            cfg=cfg,
        )

    assert signal.current_price == 7.0
    assert signal.insurance_value == 10.0
    assert signal.dcf_value is None
    assert build.call_args.kwargs == {
        "as_of_date": "2026-06-30",
        "scorecard": signal.raw_valuation,
        "persist": False,
        "pipeline_version": "v2",
        "current_price_override": 7.0,
        "issuer_cik": "42",
        "aliases": ("ADR", "ORD"),
        "db_path": db_path,
        "cfg": cfg,
    }
