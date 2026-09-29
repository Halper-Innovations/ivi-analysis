"""Tests for app.alpha.signal_assembler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.alpha.schemas import SolvencyAssessment
from app.db import connect, get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache",
        lambda http=None: {},
    )
    monkeypatch.setattr(
        "app.autonomous.sector_financial_packets._historical_fiscal_year_prices",
        lambda ticker, period_ends, *, as_of_date: {},
    )
    from app.config import get_config as _gc

    _gc.cache_clear()
    cfg = _gc()
    init_db(cfg)
    return cfg


def _seed_valuation_data(ticker: str = "AAPL"):
    now = utc_now_iso()
    with get_db() as conn:
        # Seed companyfacts for quarterly test
        for item, val in [("revenue", 400000.0), ("operating_income", 120000.0)]:
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, 2024, "FY", "2024-09-28", item, val, "USD_millions", "", now),
            )
        # Seed a quarterly row
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, 2025, "Q1", "2024-12-28", "revenue", 95000.0, "USD_millions", "", now),
        )
        # Seed a scorecard valuation row
        # Note: the valuations table may have extra columns added by _ensure_column, but the base INSERT needs
        # ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker,
                "2024-09-28",
                "scorecard",
                "{}",
                json.dumps(
                    {
                        "legacy_signal": "OVERVALUED",
                        "pricing_zone": "SPECULATIVE_PREMIUM",
                        "pricing_zone_detail": {
                            "dcf_base": 180.0,
                            "epv_adjusted": 150.0,
                            "current_price": 250.0,
                            "gate_action": "PROCEED",
                        },
                        "quality_context": {
                            "gate_action": "PROCEED",
                            "earnings_quality": "HIGH",
                        },
                        "moat_strength": {
                            "moat_class": "MODERATE_MOAT",
                            "moat_score": 5,
                        },
                        "downside_scenario": {
                            "downside_risk_class": "LIMITED",
                        },
                    }
                ),
                "[]",
                now,
            ),
        )


def test_assemble_packet_basic(monkeypatch, tmp_path):
    """Signal packet should contain valuation data from DB."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("AAPL")
    from app.alpha.signal_assembler import assemble_signal_packet

    packet = assemble_signal_packet("AAPL")
    assert packet.ticker == "AAPL"
    assert packet.dcf_value == 180.0
    assert packet.current_price == 250.0
    assert packet.gate_verdict == "PROCEED"
    assert packet.moat_score == 5
    assert packet.downside_risk_class == "LIMITED"


def test_assemble_packet_missing_ticker(monkeypatch, tmp_path):
    """Missing ticker should return a packet with None/empty fields."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.alpha.signal_assembler import assemble_signal_packet

    packet = assemble_signal_packet("ZZZZ")
    assert packet.ticker == "ZZZZ"
    assert packet.dcf_value is None
    assert packet.gate_verdict is None


def test_summary_dict_excludes_none(monkeypatch, tmp_path):
    """to_summary_dict should only include populated fields."""
    _init_temp_db(monkeypatch, tmp_path)
    from app.alpha.signal_assembler import assemble_signal_packet

    packet = assemble_signal_packet("ZZZZ")
    summary = packet.to_summary_dict()
    assert "ticker" in summary
    assert "dcf_value" not in summary


def test_assemble_multiple_tickers(monkeypatch, tmp_path):
    """assemble_sector_packets should return packets for all tickers."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("AAPL")
    _seed_valuation_data("MSFT")
    from app.alpha.signal_assembler import assemble_sector_packets

    packets = assemble_sector_packets(["AAPL", "MSFT", "ZZZZ"])
    assert len(packets) == 3
    assert packets["AAPL"].dcf_value == 180.0
    assert packets["ZZZZ"].dcf_value is None


def test_explicit_database_scopes_real_filing_risk_assembly(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    copied_root = tmp_path / "copied"
    copied_root.mkdir()
    copied_db = copied_root / "engine.db"
    with connect(copied_db) as conn:
        init_db(conn=conn)

    filing_path = copied_root / "root-10k.html"
    filing_path.write_text(
        "<html><body><h1>Item 1A. Risk Factors</h1><p>"
        + ("regulatory litigation compliance " * 30)
        + "</p><h1>Item 1B. Unresolved Staff Comments</h1></body></html>",
        encoding="utf-8",
    )
    now = utc_now_iso()
    with connect(copied_db) as conn:
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "111",
                "ROOT",
                "0000000111-26-000001",
                "10-K",
                "2026-03-01",
                "2025-12-31",
                "https://www.sec.gov/Archives/root.htm",
                str(filing_path),
                "parsed",
                now,
                now,
            ),
        )

    disabled_provider = MagicMock()
    disabled_provider.provider_name = "disabled"
    with (
        patch(
            "app.alpha.filing_risk_scan.get_llm_provider",
            return_value=disabled_provider,
        ),
        patch("app.alpha.signal_assembler.detect_anomalies", return_value=[]),
        patch(
            "app.alpha.signal_assembler.assess_solvency",
            return_value=SolvencyAssessment(solvency_risk="LOW"),
        ),
        patch("app.alpha.signal_assembler.analyze_method_tensions", return_value={}),
    ):
        from app.alpha.filing_risk_scan import _RISK_CACHE
        from app.alpha.signal_assembler import assemble_signal_packet

        _RISK_CACHE.clear()
        packet = assemble_signal_packet(
            "ROOT",
            filing_risk_use_llm=False,
            as_of_date="2026-04-01",
            pipeline_version="v1",
            db_path=copied_db,
        )
        _RISK_CACHE.clear()

    assert packet.filing_risk_status == "KEYWORD_FALLBACK"
    assert packet.filing_risk_signals["regulatory_legal"] == "HIGH"
    assert packet.filing_risk_metadata["source_accession"] == ("0000000111-26-000001")


def test_quarterly_revenue_trend(monkeypatch, tmp_path):
    """Packet should include quarterly revenue data from companyfacts_facts."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("AAPL")
    from app.alpha.signal_assembler import assemble_signal_packet

    packet = assemble_signal_packet("AAPL")
    assert packet.latest_quarterly_revenue == 95000.0
    assert "Q1" in (packet.latest_quarterly_period or "")


def test_v2_packet_rejects_unbound_scorecard_and_never_uses_live_or_future_evidence(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    old_scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {"dcf_base": 100.0, "current_price": 10.0},
    }
    future_scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {"dcf_base": 900.0, "current_price": 999.0},
    }
    with get_db() as conn:
        for as_of, payload in (
            ("2024-12-31", old_scorecard),
            ("2026-12-31", future_scorecard),
        ):
            conn.execute(
                """INSERT INTO valuations
                   (ticker, as_of_date, method, inputs_json, outputs_json,
                    warnings_json, created_at)
                   VALUES (?, ?, 'scorecard', '{}', ?, '[]', ?)""",
                ("PITX", as_of, json.dumps(payload), now),
            )
        for as_of, value in (("2024-12-31", 80.0), ("2026-12-31", 880.0)):
            conn.execute(
                """INSERT INTO valuations
                   (ticker, as_of_date, method, inputs_json, outputs_json,
                    warnings_json, created_at)
                   VALUES (?, ?, 'dcf', '{}', ?, '[]', ?)""",
                ("PITX", as_of, json.dumps({"status": "OK", "base": value}), now),
            )
        for fiscal_year, period_end, period_type, line_item, value, filed_date in (
            (2024, "2024-12-31", "FY", "equity", 100.0, "2025-01-15"),
            (2026, "2026-12-31", "FY", "equity", -500.0, "2027-01-15"),
            (2024, "2024-09-30", "Q3", "revenue", 50.0, "2024-11-01"),
            (2025, "2024-12-31", "Q1", "revenue", 777.0, "2025-03-01"),
            (2026, "2026-09-30", "Q3", "revenue", 999.0, "2026-11-01"),
        ):
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession)
                   VALUES (?, ?, ?, ?, ?, ?, 'USD_millions', ?, ?, ?, ?, ?)""",
                (
                    "PITX",
                    fiscal_year,
                    period_type,
                    period_end,
                    line_item,
                    value,
                    "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000077.json",
                    now,
                    filed_date,
                    "10-K" if period_type == "FY" else "10-Q",
                    f"0000000077-{filed_date[2:4]}-{fiscal_year:06d}",
                ),
            )

    risk_calls: list[dict[str, object]] = []

    def fake_risk(ticker, **kwargs):
        risk_calls.append({"ticker": ticker, **kwargs})
        return {"status": "NO_FILING"}

    with (
        patch("app.alpha.signal_assembler.scan_filing_risks", side_effect=fake_risk),
        patch("app.alpha.signal_assembler._fetch_live_price", return_value=777.0) as live,
        patch("app.alpha.signal_assembler.build_insurance_packet", return_value={}),
        patch("app.alpha.signal_assembler.analyze_method_tensions", return_value={}) as tension,
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet(
            "PITX",
            as_of_date="2025-01-31",
            pipeline_version="v2",
        )

    assert packet.dcf_value is None
    assert packet.raw_valuation == {}
    assert packet.valuation_provenance_status == "INVALID"
    assert packet.valuation_provenance_blockers == [
        "VALUATION_PIPELINE_VERSION_MISMATCH",
        "VALUATION_FACTS_NOT_FILED_ASOF",
        "VALUATION_ISSUER_MISMATCH",
        "VALUATION_PRICE_MISMATCH",
        "VALUATION_PRICE_CURRENCY_MISMATCH",
        "VALUATION_PACKET_PRICE_CURRENCY_MISMATCH",
        "VALUATION_FACTS_REVISION_MISMATCH",
    ]
    assert packet.current_price is None
    assert packet.latest_quarterly_revenue == 50.0
    assert packet.solvency_risk == "LOW"
    assert packet.anomaly_count == 0
    assert risk_calls == [
        {
            "ticker": "PITX",
            "use_llm": False,
            "as_of_date": "2025-01-31",
            "issuer_cik": None,
            "aliases": ("PITX",),
            "issuer_aware": True,
            "allow_network_materialization": False,
        }
    ]
    tension.assert_not_called()
    live.assert_not_called()


def test_v2_wrong_issuer_scorecard_cannot_enter_financial_packet_or_prompt(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json"
    with get_db() as conn:
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date)
               VALUES ('BOUND', 2024, 'FY', '2024-12-31', 'revenue', 100,
                       'USD_millions', ?, ?, '2025-01-15')""",
            (source_url, now),
        )
        from app.valuation.valuation_writer import valuation_facts_fingerprint

        fingerprint = valuation_facts_fingerprint(
            "BOUND",
            conn,
            as_of_date="2025-01-31",
            issuer_cik="42",
            issuer_aliases=("BOUND",),
        )
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at)
               VALUES ('BOUND', '2025-01-31', 'scorecard', ?, ?, '[]', ?)""",
            (
                json.dumps(
                    {
                        "pipeline_version": "v2",
                        "require_filed_asof": True,
                        "issuer_cik": "43",
                        "market_price": 42.0,
                        "price_currency": "USD",
                        "price_as_of_date": "2025-01-30",
                        "facts_fingerprint": fingerprint,
                    }
                ),
                json.dumps(
                    {
                        "pricing_zone": "MARGIN_OF_SAFETY",
                        "pricing_zone_detail": {
                            "dcf_base": 999.0,
                            "epv_adjusted": 888.0,
                            "current_price": 42.0,
                        },
                        "moat_strength": {"moat_score": 10},
                    }
                ),
                now,
            ),
        )
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at)
               VALUES ('BOUND', '2025-01-31', 'reverse_dcf', '{}', ?, '[]', ?)""",
            (
                json.dumps(
                    {
                        "status": "OK",
                        "outputs": {"implied_growth": 999.0},
                        "expectations_gap": {
                            "gap": 998.0,
                            "bucket": "EXPENSIVE_VS_EXPECTATIONS",
                            "supportable_growth": 1.0,
                        },
                    }
                ),
                now,
            ),
        )

    with (
        patch("app.alpha.signal_assembler.scan_filing_risks", return_value={"status": "NO_FILING"}),
        patch("app.alpha.signal_assembler.build_insurance_packet", return_value={}),
        patch("app.alpha.signal_assembler.detect_anomalies", return_value=[]),
        patch(
            "app.alpha.signal_assembler.assess_solvency",
            return_value=SolvencyAssessment(solvency_risk="LOW"),
        ),
        patch("app.alpha.signal_assembler.analyze_method_tensions") as tension,
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet(
            "BOUND",
            as_of_date="2025-01-31",
            pipeline_version="v2",
            current_price_override=42.0,
            price_snapshot={
                "price": 42.0,
                "currency": "USD",
                "as_of_date": "2025-01-30",
            },
            issuer_cik="42",
            issuer_aliases=("BOUND",),
            db_path=cfg.db_path,
            cfg=cfg,
        )

    assert packet.current_price == 42.0
    assert packet.dcf_value is None
    assert packet.epv_value is None
    assert packet.moat_score is None
    assert packet.raw_valuation == {}
    assert packet.valuation_provenance_status == "INVALID"
    assert packet.valuation_provenance_blockers == ["VALUATION_ISSUER_MISMATCH"]
    tension.assert_not_called()

    from app.autonomous.sector_financial_packets import (
        build_sector_company_financial_packet,
    )

    company_packet = build_sector_company_financial_packet(
        packet,
        sector=None,
        as_of_date="2025-01-31",
        pipeline_version="v2",
        cap_classification={
            "current_price": 42.0,
            "current_price_currency": "USD",
            "current_price_as_of_date": "2025-01-30",
            "issuer_cik": "42",
        },
    )
    assert company_packet.valuation["valuation_anchor"] is None
    assert company_packet.valuation["implied_growth"] is None
    assert company_packet.valuation["expectations_gap"] is None
    assert company_packet.valuation["expectations_gap_bucket"] == ("EXPECTATIONS_GAP_UNRELIABLE")
    assert company_packet.valuation["expectations_gap_provenance_status"] == (
        "SUPPRESSED_INVALID_SCORECARD"
    )
    assert company_packet.valuation["expectations_gap_provenance_blockers"] == [
        "VALUATION_ISSUER_MISMATCH"
    ]
    assert company_packet.blockers == [
        "MISSING_VALUATION",
        "VALUATION_ISSUER_MISMATCH",
    ]
    assert company_packet.accounting_quality["valuation_provenance_status"] == "INVALID"
    assert company_packet.accounting_quality["valuation_provenance_blockers"] == [
        "VALUATION_ISSUER_MISMATCH"
    ]
    assert "999.0" not in json.dumps(company_packet.to_dict(), sort_keys=True)

    from app.autonomous.sector_runtime import (
        _compact_packet,
        _cross_sectional_factor_vectors,
        _selection_audit_for_ticker,
    )

    factor = _cross_sectional_factor_vectors([company_packet])[0]
    assert factor.gap is None
    assert "999.0" not in json.dumps(_compact_packet(company_packet), sort_keys=True)
    audit = _selection_audit_for_ticker(
        selected_ticker="BOUND",
        packets_by_ticker={"BOUND": company_packet},
        scenarios=[],
        tool_calls=[],
        evidence=[],
        degraded_states=[],
    )
    assert "EXPENSIVE_VS_EXPECTATIONS" not in audit["confidence_caps"]


def test_v2_validated_scorecard_supplies_bound_anchor_and_method_tension(
    monkeypatch,
    tmp_path,
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    source_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json"
    with get_db() as conn:
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date)
               VALUES ('VALID', 2024, 'FY', '2024-12-31', 'revenue', 100,
                       'USD_millions', ?, ?, '2025-01-15')""",
            (source_url, now),
        )
        from app.valuation.valuation_writer import valuation_facts_fingerprint

        fingerprint = valuation_facts_fingerprint(
            "VALID",
            conn,
            as_of_date="2025-01-31",
            issuer_cik="42",
            issuer_aliases=("VALID",),
        )
        inputs = {
            "pipeline_version": "v2",
            "require_filed_asof": True,
            "issuer_cik": "42",
            "market_price": 42.0,
            "price_currency": "USD",
            "price_as_of_date": "2025-01-30",
            "facts_fingerprint": fingerprint,
        }
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at)
               VALUES ('VALID', '2025-01-31', 'scorecard', ?, ?, '[]', ?)""",
            (
                json.dumps(inputs),
                json.dumps(
                    {
                        "pricing_zone": "MARGIN_OF_SAFETY",
                        "pricing_zone_detail": {
                            "dcf_base": 100.0,
                            "epv_adjusted": 80.0,
                            "current_price": 42.0,
                        },
                    }
                ),
                now,
            ),
        )
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at)
               VALUES ('VALID', '2025-01-31', 'dcf', '{}',
                       '{"status":"OK","base":999}', '[]', ?)""",
            (now,),
        )
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at)
               VALUES ('VALID', '2025-01-31', 'reverse_dcf', '{}', ?, '[]', ?)""",
            (
                json.dumps(
                    {
                        "status": "OK",
                        "outputs": {"implied_growth": 777.0},
                        "expectations_gap": {
                            "gap": 776.0,
                            "bucket": "EXPENSIVE_VS_EXPECTATIONS",
                            "supportable_growth": 1.0,
                        },
                    }
                ),
                now,
            ),
        )

    with (
        patch("app.alpha.signal_assembler.scan_filing_risks", return_value={"status": "NO_FILING"}),
        patch("app.alpha.signal_assembler.build_insurance_packet", return_value={}),
        patch("app.alpha.signal_assembler.detect_anomalies", return_value=[]),
        patch(
            "app.alpha.signal_assembler.assess_solvency",
            return_value=SolvencyAssessment(solvency_risk="LOW"),
        ),
        patch("app.alpha.signal_assembler.analyze_method_tensions", return_value={}) as tension,
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet(
            "VALID",
            as_of_date="2025-01-31",
            pipeline_version="v2",
            current_price_override=42.0,
            price_snapshot={
                "price": 42.0,
                "currency": "USD",
                "as_of_date": "2025-01-30",
            },
            issuer_cik="42",
            issuer_aliases=("VALID",),
            db_path=cfg.db_path,
            cfg=cfg,
        )

    assert packet.valuation_provenance_status == "VALIDATED"
    assert packet.valuation_provenance_blockers == []
    assert packet.dcf_value == 100.0
    assert packet.epv_value == 80.0
    assert packet.current_price == 42.0
    assert tension.call_args.kwargs["dcf_value"] == 100.0
    assert tension.call_args.kwargs["epv_value"] == 80.0
    assert tension.call_args.kwargs["current_price"] == 42.0

    from app.autonomous.sector_financial_packets import (
        build_sector_company_financial_packet,
    )

    company_packet = build_sector_company_financial_packet(
        packet,
        sector=None,
        as_of_date="2025-01-31",
        pipeline_version="v2",
        cap_classification={
            "current_price": 42.0,
            "current_price_currency": "USD",
            "current_price_as_of_date": "2025-01-30",
            "issuer_cik": "42",
            "issuer_aliases": ["VALID"],
        },
    )
    assert company_packet.valuation["implied_growth"] is None
    assert company_packet.valuation["expectations_gap"] is None
    assert company_packet.valuation["expectations_gap_bucket"] == ("EXPECTATIONS_GAP_UNRELIABLE")
    assert company_packet.valuation["expectations_gap_provenance_status"] == "INVALID"
    assert company_packet.valuation["expectations_gap_provenance_blockers"] == [
        "VALUATION_PIPELINE_VERSION_MISMATCH",
        "VALUATION_FACTS_NOT_FILED_ASOF",
        "VALUATION_ISSUER_MISMATCH",
        "VALUATION_PRICE_MISMATCH",
        "VALUATION_PRICE_CURRENCY_MISMATCH",
        "VALUATION_PRICE_ASOF_MISMATCH",
        "VALUATION_FACTS_REVISION_MISMATCH",
    ]
    assert "777.0" not in json.dumps(company_packet.to_dict(), sort_keys=True)


def test_v2_packet_uses_only_explicit_cap_stage_price(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("CAPX")
    with (
        patch("app.alpha.signal_assembler.scan_filing_risks", return_value={"status": "NO_FILING"}),
        patch("app.alpha.signal_assembler._fetch_live_price", return_value=777.0) as live,
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet(
            "CAPX",
            as_of_date="2025-01-31",
            pipeline_version="v2",
            current_price_override=42.0,
            issuer_cik="42",
            issuer_aliases=("CAPX",),
            db_path=cfg.db_path,
            cfg=cfg,
        )

    assert packet.current_price == 42.0
    live.assert_not_called()


def test_filing_risk_signals_attached(monkeypatch, tmp_path):
    """Signal packet should include filing risk signals when available."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("NICE")

    mock_risk = {
        "competitive_disruption": "HIGH",
        "secular_decline": "MODERATE",
        "regulatory_legal": "LOW",
        "customer_concentration": "LOW",
        "summary": "AI disruption threat to contact center business.",
        "status": "OK",
        "evidence_status": "READABLE_RISK_SECTION",
        "source_accession": "0001003935-26-000010",
        "source_form_type": "20-F",
        "source_filing_date": "2026-02-26",
        "source_filing_age_days": 59,
        "source_issuer_cik": "0001003935",
        "analysis_as_of_date": "2026-04-26",
        "risk_text_chars": 1234,
        "warnings": [],
    }
    with (
        patch("app.alpha.signal_assembler.scan_filing_risks", return_value=mock_risk),
        patch("app.alpha.signal_assembler.build_insurance_packet", return_value={}),
        patch(
            "app.alpha.signal_assembler.foreign_normalized_facts_gap_reason",
            return_value="IFRS_FACTS_UNSUPPORTED",
        ),
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet(
            "NICE",
            as_of_date="2026-04-26",
            pipeline_version="v2",
        )

    assert packet.filing_risk_signals.get("competitive_disruption") == "HIGH"
    assert packet.filing_risk_signals.get("foreign_facts_gap_reason") == "IFRS_FACTS_UNSUPPORTED"
    assert packet.filing_risk_status == "OK"
    assert packet.filing_risk_metadata == {
        "evidence_status": "READABLE_RISK_SECTION",
        "source_accession": "0001003935-26-000010",
        "source_form_type": "20-F",
        "source_filing_date": "2026-02-26",
        "source_filing_age_days": 59,
        "risk_text_chars": 1234,
        "warnings": [],
    }
    summary = packet.to_summary_dict()
    assert "filing_risk_signals" in summary
    assert "filing_risk_metadata" in summary


def test_v1_packet_does_not_run_v2_foreign_gap_classification(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("NICE")
    mock_risk = {
        "status": "OK",
        "source_form_type": "20-F",
        "source_issuer_cik": "0001003935",
    }
    with (
        patch("app.alpha.signal_assembler.scan_filing_risks", return_value=mock_risk),
        patch("app.alpha.signal_assembler.build_insurance_packet", return_value={}),
        patch("app.alpha.signal_assembler.foreign_normalized_facts_gap_reason") as foreign_gap,
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("NICE", pipeline_version="v1")

    foreign_gap.assert_not_called()
    assert "foreign_facts_gap_reason" not in packet.filing_risk_signals


def test_v2_sector_assembly_passes_cap_identity_to_company_packet(monkeypatch):
    captured: list[dict] = []

    def fake_assemble(ticker, **kwargs):
        captured.append({"ticker": ticker, **kwargs})
        return object()

    with patch(
        "app.alpha.signal_assembler.assemble_signal_packet",
        side_effect=fake_assemble,
    ):
        from app.alpha.signal_assembler import assemble_sector_packets

        packets = assemble_sector_packets(
            ["ADR"],
            as_of_date="2026-04-26",
            pipeline_version="v2",
            current_prices={"ADR": 42.0},
            issuer_contexts={
                "ADR": {
                    "issuer_cik": "42",
                    "issuer_primary_ticker": "ORD",
                    "issuer_listed_tickers": ["ORD", "ADR"],
                    "current_price_as_of_date": "2026-04-25",
                    "current_price_currency": "USD",
                    "current_price_source": "stooq",
                    "current_price_source_url": "https://stooq.example/adr",
                    "current_price_confidence": "MEDIUM",
                }
            },
        )

    assert list(packets) == ["ADR"]
    assert captured == [
        {
            "ticker": "ADR",
            "filing_risk_use_llm": False,
            "as_of_date": "2026-04-26",
            "pipeline_version": "v2",
            "current_price_override": 42.0,
            "price_snapshot": {
                "price": 42.0,
                "as_of_date": "2026-04-25",
                "currency": "USD",
                "source": "stooq",
                "url": "https://stooq.example/adr",
                "confidence": "MEDIUM",
            },
            "issuer_cik": "42",
            "issuer_aliases": ("ADR", "ORD"),
            "db_path": None,
            "allowed_filing_roots": (),
        }
    ]


def test_v1_sector_assembly_freezes_explicit_cap_quote_and_disables_filing_llm():
    captured: list[dict] = []

    def fake_assemble(ticker, **kwargs):
        captured.append({"ticker": ticker, **kwargs})
        return object()

    with patch(
        "app.alpha.signal_assembler.assemble_signal_packet",
        side_effect=fake_assemble,
    ):
        from app.alpha.signal_assembler import assemble_sector_packets

        packets = assemble_sector_packets(
            ["BASIS"],
            as_of_date="2026-07-22",
            pipeline_version="v1",
            current_prices={"BASIS": 125.0},
            issuer_contexts={
                "BASIS": {
                    "price_as_of_date": "2026-07-21",
                    "price_currency": "USD",
                    "price_source": "fixture",
                    "price_source_url": "https://example.test/basis",
                    "price_confidence": "HIGH",
                }
            },
        )

    assert list(packets) == ["BASIS"]
    assert captured[0]["filing_risk_use_llm"] is False
    assert captured[0]["as_of_date"] == "2026-07-22"
    assert captured[0]["pipeline_version"] == "v1"
    assert captured[0]["current_price_override"] == 125.0
    assert captured[0]["price_snapshot"] == {
        "price": 125.0,
        "as_of_date": "2026-07-21",
        "currency": "USD",
        "source": "fixture",
        "url": "https://example.test/basis",
        "confidence": "HIGH",
    }


def test_v2_quarterly_trend_binds_cik_filed_asof_and_explicit_db(tmp_path):
    db_path = tmp_path / "quarterly-context.db"
    conn = connect(db_path)
    try:
        init_db(conn=conn)
        now = utc_now_iso()
        rows = (
            ("ORD", 2025, "Q1", "2025-03-31", 100.0, "2025-04-20", 42),
            ("ORD", 2025, "Q2", "2025-06-30", 110.0, "2025-07-20", 42),
            ("ORD", 2025, "Q3", "2025-09-30", 999.0, "2025-11-20", 42),
            ("ADR", 2025, "Q2", "2025-06-30", 888.0, "2025-07-20", 43),
        )
        for ticker, year, period_type, period_end, value, filed_date, cik in rows:
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession)
                   VALUES (?, ?, ?, ?, 'revenue', ?, 'USD_millions', ?, ?, ?,
                           '10-Q', ?)""",
                (
                    ticker,
                    year,
                    period_type,
                    period_end,
                    value,
                    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                    now,
                    filed_date,
                    f"{cik:010d}-25-00000{period_type[-1]}",
                ),
            )
        conn.commit()
    finally:
        conn.close()

    from app.alpha.signal_assembler import _load_quarterly_revenue_trend

    with patch(
        "app.alpha.signal_assembler.get_db",
        side_effect=AssertionError("explicit db_path must not open the configured DB"),
    ):
        latest, period, trend = _load_quarterly_revenue_trend(
            "ADR",
            as_of_date="2025-08-31",
            require_filed_asof=True,
            issuer_cik="42",
            aliases=("ADR", "ORD"),
            db_path=db_path,
        )

    assert latest == 110.0
    assert period == "FY2025Q2"
    assert trend == "ACCELERATING"


def test_v1_signal_subreaders_keep_legacy_one_ticker_call_shape(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    with (
        patch(
            "app.alpha.signal_assembler.scan_filing_risks",
            return_value={"status": "NO_FILING"},
        ),
        patch(
            "app.alpha.signal_assembler._load_quarterly_revenue_trend",
            return_value=(None, None, "UNKNOWN"),
        ) as quarterly,
        patch("app.alpha.signal_assembler.detect_anomalies", return_value=[]) as anomalies,
        patch(
            "app.alpha.signal_assembler.assess_solvency",
            return_value=SolvencyAssessment(solvency_risk="UNKNOWN"),
        ) as solvency,
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        assemble_signal_packet("LEGACY", pipeline_version="v1")

    quarterly.assert_called_once_with("LEGACY")
    anomalies.assert_called_once_with("LEGACY")
    solvency.assert_called_once_with("LEGACY")


def test_v1_fixed_asof_subreaders_require_filing_visibility_and_issuer_identity(
    monkeypatch,
    tmp_path,
):
    from app.insurance.routing import SecurityRoutingResult

    cfg = _init_temp_db(monkeypatch, tmp_path)
    routing_result = SecurityRoutingResult(
        ticker="LEGACY",
        security_type="common",
        issuer_type="insurance_underwriter",
        insurance_subtype="pc_insurer",
        accounting_regime="US_GAAP",
        model_status="ROUTED",
    )
    with (
        patch(
            "app.alpha.signal_assembler._load_scorecard_record",
            return_value=(
                "2026-02-13",
                {
                    "pricing_zone": "INSUFFICIENT_DATA",
                    "pricing_zone_detail": {},
                },
            ),
        ),
        patch(
            "app.insurance.packet.route_security",
            return_value=routing_result,
        ) as insurance_routing,
        patch(
            "app.insurance.packet.calculate_insurance_common_valuation",
            return_value={
                "status": "OK",
                "model_status": "OK",
                "method": "insurance_common",
                "reason_codes": [],
            },
        ) as insurance_valuation,
        patch(
            "app.insurance.packet.build_insurance_operating_metrics",
            return_value={
                "status": "NOT_APPLICABLE",
                "reason_codes": [],
            },
        ) as insurance_metrics,
        patch(
            "app.alpha.signal_assembler.scan_filing_risks",
            return_value={"status": "NO_FILING"},
        ) as filing_risk,
        patch(
            "app.alpha.signal_assembler._load_quarterly_revenue_trend",
            return_value=(None, None, "UNKNOWN"),
        ) as quarterly,
        patch("app.alpha.signal_assembler.detect_anomalies", return_value=[]) as anomalies,
        patch(
            "app.alpha.signal_assembler.assess_solvency",
            return_value=SolvencyAssessment(solvency_risk="UNKNOWN"),
        ) as solvency,
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        assemble_signal_packet(
            "LEGACY",
            as_of_date="2026-02-13",
            pipeline_version="v1",
            issuer_cik="42",
            issuer_aliases=("LEGACY", "PRIMARY"),
            db_path=cfg.db_path,
            cfg=cfg,
        )

    quarterly.assert_called_once_with(
        "LEGACY",
        as_of_date="2026-02-13",
        require_filed_asof=True,
        issuer_cik="42",
        aliases=("LEGACY", "PRIMARY"),
        db_path=cfg.db_path,
        cfg=cfg,
    )
    anomalies.assert_called_once_with(
        "LEGACY",
        as_of_date="2026-02-13",
        require_filed_asof=True,
        issuer_cik="42",
        aliases=("LEGACY", "PRIMARY"),
        db_path=cfg.db_path,
    )
    solvency.assert_called_once_with(
        "LEGACY",
        as_of_date="2026-02-13",
        require_filed_asof=True,
        issuer_cik="42",
        aliases=("LEGACY", "PRIMARY"),
        db_path=cfg.db_path,
    )
    assert filing_risk.call_args.kwargs["issuer_cik"] == "42"
    assert filing_risk.call_args.kwargs["aliases"] == ("LEGACY", "PRIMARY")
    assert filing_risk.call_args.kwargs["issuer_aware"] is True
    assert filing_risk.call_args.kwargs["allow_network_materialization"] is False
    for call in (
        insurance_routing.call_args,
        insurance_valuation.call_args,
        insurance_metrics.call_args,
    ):
        assert call.kwargs["pipeline_version"] == "v2"
        assert call.kwargs["issuer_cik"] == "42"
        assert call.kwargs["aliases"] == ("LEGACY", "PRIMARY")
        assert call.kwargs["db_path"] == cfg.db_path
        assert call.kwargs["cfg"] == cfg


def test_v1_fixed_asof_quarterly_reader_rejects_postfiled_row(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.executemany(
            """
                INSERT INTO companyfacts_facts(
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession
                )
                VALUES(
                    'LEGACY', 2025, ?, ?, 'revenue', ?, 'USD_millions',
                    'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                    ?, ?, '10-Q', ?
                )
                """,
            [
                (
                    "Q1",
                    "2025-03-31",
                    100.0,
                    now,
                    "2025-04-20",
                    "0000000001-25-000001",
                ),
                (
                    "Q2",
                    "2025-06-30",
                    999.0,
                    now,
                    "2026-03-01",
                    "0000000001-26-000001",
                ),
            ],
        )

    from app.alpha.signal_assembler import _load_quarterly_revenue_trend

    latest, period, trend = _load_quarterly_revenue_trend(
        "LEGACY",
        as_of_date="2026-02-13",
        require_filed_asof=True,
    )

    assert latest == 100.0
    assert period == "FY2025Q1"
    assert trend == "UNKNOWN"


def test_signal_packet_preserves_serialized_going_concern_assertions(
    monkeypatch,
    tmp_path,
):
    from app.alpha.schemas import GoingConcernAssertion

    _init_temp_db(monkeypatch, tmp_path)
    assertion = GoingConcernAssertion(
        subject="CONSOLIDATED_SUBSIDIARY",
        subject_detail="wholly owned consolidated subsidiary",
        assertion_mode="AFFIRMATIVE_CURRENT",
        blockable=True,
        accession="0000000042-26-000009",
        form_type="10-K",
        filing_date="2026-02-20",
        section="FINANCIAL_STATEMENTS_NOTES",
        excerpt=(
            "Our wholly owned consolidated subsidiary has substantial doubt "
            "about its ability to continue as a going concern."
        ),
        corroborating_distress=("DEBT_DUE_WITHIN_12MO",),
        issuer_cik="42",
        source_url="https://www.sec.gov/Archives/edgar/data/42/subsidiary.htm",
        content_revision="sha256:subsidiary-v1",
    )
    assessment = SolvencyAssessment(
        solvency_risk="CRITICAL",
        going_concern_language=True,
        going_concern_assertions=[assertion],
        signals=["GOING_CONCERN_LANGUAGE"],
    )

    with (
        patch(
            "app.alpha.signal_assembler.scan_filing_risks",
            return_value={"status": "NO_FILING"},
        ),
        patch(
            "app.alpha.signal_assembler._load_quarterly_revenue_trend",
            return_value=(None, None, "UNKNOWN"),
        ),
        patch("app.alpha.signal_assembler.detect_anomalies", return_value=[]),
        patch(
            "app.alpha.signal_assembler.assess_solvency",
            return_value=assessment,
        ),
        patch("app.alpha.signal_assembler.analyze_method_tensions", return_value={}),
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("ATTR", pipeline_version="v1")

    expected = [
        {
            "subject": "CONSOLIDATED_SUBSIDIARY",
            "subject_detail": "wholly owned consolidated subsidiary",
            "assertion_mode": "AFFIRMATIVE_CURRENT",
            "blockable": True,
            "accession": "0000000042-26-000009",
            "form_type": "10-K",
            "filing_date": "2026-02-20",
            "section": "FINANCIAL_STATEMENTS_NOTES",
            "excerpt": (
                "Our wholly owned consolidated subsidiary has substantial doubt "
                "about its ability to continue as a going concern."
            ),
            "corroborating_distress": ["DEBT_DUE_WITHIN_12MO"],
            "issuer_cik": "42",
            "source_url": "https://www.sec.gov/Archives/edgar/data/42/subsidiary.htm",
            "content_revision": "sha256:subsidiary-v1",
        }
    ]
    assert packet.research_report["solvency"]["going_concern_assertions"] == expected
    assert (
        packet.to_summary_dict()["research_report"]["solvency"]["going_concern_assertions"]
        == expected
    )


def test_research_report_attached(monkeypatch, tmp_path):
    """Signal packet should include anomalies and solvency (no LLM investigation at this stage)."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("TEST")
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO companyfacts_facts
               (ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("TEST", 2025, "FY", "2025-12-31", "equity", -10.0, "USD_millions", "", now),
        )

    with patch(
        "app.alpha.signal_assembler.scan_filing_risks",
        return_value={
            "status": "NO_FILING",
            "competitive_disruption": "UNKNOWN",
            "secular_decline": "UNKNOWN",
            "regulatory_legal": "UNKNOWN",
            "customer_concentration": "UNKNOWN",
            "summary": "",
        },
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("TEST")

    assert packet.anomaly_count >= 1
    assert packet.solvency_risk in ("CRITICAL", "ELEVATED", "LOW", "UNKNOWN")
    assert packet.research_status is not None
    assert packet.research_report.get("investigations", []) == []


def test_method_tension_attached(monkeypatch, tmp_path):
    """Signal packet should include method tension analysis."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("TEST")
    # Seed individual method rows — DCF and EPV with divergent values to trigger tension
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                "TEST",
                "2024-09-28",
                "dcf",
                "{}",
                json.dumps({"status": "OK", "base": 150.0, "low": 120.0, "high": 180.0}),
                "[]",
                now,
            ),
        )
        conn.execute(
            "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                "TEST",
                "2024-09-28",
                "epv",
                "{}",
                json.dumps(
                    {"status": "OK", "value_per_share": 50.0, "avg_operating_income": 100.0}
                ),
                "[]",
                now,
            ),
        )

    with patch(
        "app.alpha.signal_assembler.scan_filing_risks",
        return_value={
            "status": "NO_FILING",
            "competitive_disruption": "UNKNOWN",
            "secular_decline": "UNKNOWN",
            "regulatory_legal": "UNKNOWN",
            "customer_concentration": "UNKNOWN",
            "summary": "",
        },
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("TEST")

    assert packet.method_tension_type is not None
    assert packet.growth_dependency_ratio is not None


def test_unmeasured_zones_contribute_no_anchor_methods(monkeypatch, tmp_path):
    """Review ANCHOR-2 / GZ-2 / ZONE-ALLOWLIST-LIVE-DIVERGENCE: the backtest
    measures only MARGIN_OF_SAFETY / GROWTH_DEPENDENT / SPECULATIVE_PREMIUM.
    Live packets anchored (and the watchlist could DEPLOY_READY) on
    INSUFFICIENT_DATA-zone names whose pzd still carries a positive dcf_base
    or graham/ncav per-share — a cohort the calibration never measures. Zones outside the
    surviving set (and zoneless pre-zone scorecards) must contribute no
    anchor methods; raw_valuation stays attached for LLM context."""
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "INSF",
                "2024-09-28",
                "scorecard",
                "{}",
                json.dumps(
                    {
                        "pricing_zone": "INSUFFICIENT_DATA",
                        "pricing_zone_detail": {
                            "reason": "Current price or adjusted EPV unavailable.",
                            "dcf_base": 80.0,
                            "epv_adjusted": None,
                            "graham_value_per_share": 25.0,
                            "ncav_value_per_share": 8.0,
                            "current_price": 30.0,
                        },
                    }
                ),
                "[]",
                now,
            ),
        )
    from app.alpha.signal_assembler import assemble_signal_packet

    packet = assemble_signal_packet("INSF")
    assert packet.pricing_zone == "INSUFFICIENT_DATA"
    assert packet.dcf_value is None
    assert packet.epv_value is None
    assert packet.graham_value is None
    assert packet.ncav_value is None
    assert packet.raw_valuation  # LLM context retained


def test_zoneless_legacy_scorecard_contributes_no_anchor_methods(monkeypatch, tmp_path):
    """A scorecard without pricing_zone predates the zone-aware writer; the
    backtest skips those rows, so live must not anchor them either."""
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "LGCY",
                "2024-09-28",
                "scorecard",
                "{}",
                json.dumps(
                    {
                        "pricing_zone_detail": {
                            "dcf_base": 180.0,
                            "epv_adjusted": 150.0,
                            "current_price": 250.0,
                        },
                    }
                ),
                "[]",
                now,
            ),
        )
    from app.alpha.signal_assembler import assemble_signal_packet

    packet = assemble_signal_packet("LGCY")
    assert packet.dcf_value is None
    assert packet.epv_value is None


def _seed_method_row(ticker, method, as_of, outputs, *, run_id=None, created_at="2026-09-01T00:00:00Z"):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json,"
            " warnings_json, created_at, source_run_id) VALUES (?,?,?,?,?,?,?,?)",
            (ticker, as_of, method, "{}", json.dumps(outputs), "[]", created_at, run_id),
        )


def test_legacy_method_values_come_from_one_run_not_the_newest_row_per_method(
    monkeypatch, tmp_path
):
    """Each method was read as 'newest row at or before the as-of date', so a
    tension compared a DCF from one run with an EPV from another. Only the newest run counts."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_method_row("MIX", "dcf", "2024-06-30", {"status": "OK", "base": 150.0}, run_id="old")
    _seed_method_row("MIX", "graham", "2024-06-30", {"status": "OK", "value_per_share": 90.0}, run_id="old")
    _seed_method_row(
        "MIX", "epv", "2024-09-28", {"status": "OK", "value_per_share": 50.0}, run_id="new"
    )
    _seed_method_row(
        "MIX", "ncav", "2024-09-28", {"status": "OK", "value_per_share": 20.0}, run_id="new"
    )

    from app.alpha.signal_assembler import _legacy_method_values

    with get_db() as conn:
        values = _legacy_method_values(
            conn, ticker="MIX", as_of_date=None, issuer_cik=None, issuer_aliases=()
        )
    assert values == {"epv": 50.0, "ncav": 20.0}


def test_legacy_method_values_share_a_run_when_the_same_day_has_two_runs(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_method_row(
        "SAME", "dcf", "2024-09-28", {"status": "OK", "base": 150.0}, run_id="a",
        created_at="2026-09-01T00:00:00Z",
    )
    _seed_method_row(
        "SAME", "epv", "2024-09-28", {"status": "OK", "value_per_share": 50.0}, run_id="b",
        created_at="2026-09-02T00:00:00Z",
    )

    from app.alpha.signal_assembler import _legacy_method_values

    with get_db() as conn:
        values = _legacy_method_values(
            conn, ticker="SAME", as_of_date=None, issuer_cik=None, issuer_aliases=()
        )
    assert values == {"epv": 50.0}


def test_legacy_method_values_exclude_negative_and_non_ok_values(monkeypatch, tmp_path):
    """An EPV_NEGATIVE row's negative per-share 'value' used to enter the tension analysis,
    dragging the intrinsic range below zero and the growth-dependency ratio above 1."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_method_row("NEG", "dcf", "2024-09-28", {"status": "OK", "base": 150.0}, run_id="r")
    _seed_method_row(
        "NEG", "epv", "2024-09-28", {"status": "EPV_NEGATIVE", "value_per_share": -20.0}, run_id="r"
    )
    _seed_method_row(
        "NEG", "graham", "2024-09-28", {"status": "OK", "value_per_share": 0.0}, run_id="r"
    )
    _seed_method_row(
        "NEG", "ncav", "2024-09-28", {"status": "NOT_APPLICABLE", "value_per_share": 5.0}, run_id="r"
    )

    from app.alpha.signal_assembler import _legacy_method_values

    with get_db() as conn:
        values = _legacy_method_values(
            conn, ticker="NEG", as_of_date=None, issuer_cik=None, issuer_aliases=()
        )
    assert values == {"dcf": 150.0}


def test_negative_epv_no_longer_stretches_the_packet_intrinsic_range(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_valuation_data("NEG")
    _seed_method_row("NEG", "dcf", "2024-09-28", {"status": "OK", "base": 150.0}, run_id="r")
    _seed_method_row(
        "NEG", "epv", "2024-09-28", {"status": "EPV_NEGATIVE", "value_per_share": -20.0}, run_id="r"
    )
    _seed_method_row("NEG", "graham", "2024-09-28", {"status": "OK", "value_per_share": 90.0}, run_id="r")

    with patch(
        "app.alpha.signal_assembler.scan_filing_risks",
        return_value={
            "status": "NO_FILING",
            "competitive_disruption": "UNKNOWN",
            "secular_decline": "UNKNOWN",
            "regulatory_legal": "UNKNOWN",
            "customer_concentration": "UNKNOWN",
            "summary": "",
        },
    ):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("NEG")

    assert packet.intrinsic_range_low == 90.0
    assert packet.intrinsic_range_high == 150.0
