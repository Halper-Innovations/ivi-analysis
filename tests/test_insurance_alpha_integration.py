from __future__ import annotations

import json
from unittest.mock import patch

import pytest

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


def _seed_scorecard(ticker: str, *, price: float = 20.0) -> int:
    now = utc_now_iso()
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
            VALUES(?, '2026-02-01', 'scorecard', '{}', ?, '[]', ?)
            """,
            (
                ticker,
                json.dumps(
                    {
                        "legacy_signal": "UNDERVALUED",
                        "pricing_zone_detail": {
                            "dcf_base": 100.0,
                            "epv_adjusted": 120.0,
                            "current_price": price,
                            "gate_action": "PROCEED",
                        },
                        "quality_context": {"gate_action": "PROCEED", "earnings_quality": "MEDIUM"},
                    }
                ),
                now,
            ),
        )
        return int(cursor.lastrowid)


def _authorize_scorecard(
    monkeypatch,
    tmp_path,
    *,
    cfg,
    ticker: str,
    row_id: int,
) -> None:
    from tests.financial_integrity_helpers import authorize_valuation_rows

    authorize_valuation_rows(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        run_id=f"autonomous_sector_insurance_{ticker.lower()}_authorized",
        row_ids=[row_id],
        issuer_ciks={ticker: "0000000099"},
    )


def _seed_companyfact(
    ticker: str,
    *,
    year: int,
    line_item: str,
    value: float,
    filed_date: str | None = None,
) -> None:
    # Ingestion stamps normalized units; the alpha report writer refuses any
    # other monetary unit, so fixtures must carry the production-true stamp.
    units = "shares_millions" if line_item == "shares_outstanding" else "USD_millions"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units,
                source_url, fetched_at, filed_date, form, accession
            )
            VALUES(
                ?, ?, 'FY', ?, ?, ?, ?,
                'https://example.test/companyfacts', ?, ?, '10-K', ?
            )
            """,
            (
                ticker,
                year,
                f"{year}-12-31",
                line_item,
                value,
                units,
                utc_now_iso(),
                filed_date,
                f"{ticker}-{year}-000001",
            ),
        )


def _seed_company_and_filing(
    ticker: str, *, name: str, text: str, tmp_path, sector: str = "insurance"
) -> None:
    now = utc_now_iso()
    path = tmp_path / f"{ticker.lower()}_10k.txt"
    path.write_text(text, encoding="utf-8")
    cik = "0000000099"
    submission_path = tmp_path / "data" / "cache" / "submissions" / f"{cik.zfill(10)}.json"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(
        json.dumps({"name": name, "tickers": [ticker], "exchanges": ["NYSE"]}),
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            "INSERT INTO companies(ticker, cik, name, created_at) VALUES(?, '0000000099', ?, ?)",
            (ticker, name, now),
        )
        conn.execute(
            """
            INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at)
            VALUES(?, '2026-02-01', ?, 1.0, '[]', ?)
            """,
            (ticker, sector, now),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            )
            VALUES('0000000099', ?, ?, '10-K', '2026-02-01', '2025-12-31', 'https://example.test/10k', ?, 'OK', ?, ?)
            """,
            (ticker, f"{ticker}-2026", str(path), now, now),
        )


def _no_filing_risks():
    return {
        "status": "NO_FILING",
        "competitive_disruption": "UNKNOWN",
        "secular_decline": "UNKNOWN",
        "regulatory_legal": "UNKNOWN",
        "customer_concentration": "UNKNOWN",
        "summary": "",
    }


def test_alpha_packet_suppresses_generic_dcf_for_preferred_and_uses_insurance_anchor(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    scorecard_id = _seed_scorecard("IPRF", price=20.0)
    _seed_company_and_filing(
        "IPRF",
        name="Example Insurer Depositary Shares 6.000% Non-Cumulative Preferred Series A",
        text=(
            "Each depositary share represents preferred stock with a $25 liquidation preference. "
            "The dividend rate is 6.000%. The preferred stock is non-cumulative."
        ),
        tmp_path=tmp_path,
    )
    _authorize_scorecard(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        ticker="IPRF",
        row_id=scorecard_id,
    )

    with patch("app.alpha.signal_assembler.scan_filing_risks", return_value=_no_filing_risks()):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("IPRF")

    assert packet.security_type == "depositary_preferred"
    assert packet.issuer_type == "insurance_underwriter"
    assert packet.model_status == "OK"
    assert packet.dcf_value is None
    assert packet.epv_value is None
    assert packet.insurance_method == "insurance_preferred"
    assert packet.insurance_value == 25.0
    assert "GENERIC_DCF_EPV_SUPPRESSED" in packet.valuation_headwinds

    from app.alpha.consensus_ranker import rank_by_consensus

    ranked = rank_by_consensus({"IPRF": packet}).ranked
    assert ranked[0].method_discounts == {"insurance": 0.2}
    assert ranked[0].adjustments[0] == "INSURANCE_MODEL_USED (insurance_preferred)"


def test_alpha_packet_blocks_generic_dcf_for_common_insurer_without_valid_model(
    monkeypatch, tmp_path
):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    scorecard_id = _seed_scorecard("IBLK", price=20.0)
    _seed_company_and_filing(
        "IBLK",
        name="Example Life Insurance Corp",
        text="The company writes life insurance and annuity products subject to LDTI.",
        tmp_path=tmp_path,
    )
    _authorize_scorecard(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        ticker="IBLK",
        row_id=scorecard_id,
    )

    with patch("app.alpha.signal_assembler.scan_filing_risks", return_value=_no_filing_risks()):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("IBLK")

    assert packet.security_type == "common"
    assert packet.issuer_type == "insurance_underwriter"
    assert packet.model_status == "NEEDS_DATA"
    assert packet.dcf_value is None
    assert packet.epv_value is None
    assert packet.insurance_value is None
    assert packet.model_blockers == [
        "MISSING_OR_NONPOSITIVE_BOOK_VALUE",
        "MISSING_SHARES_OUTSTANDING",
        "MISSING_NORMALIZED_ROE_INPUT",
    ]


def test_alpha_tool_fetches_insurance_evidence_packet(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    scorecard_id = _seed_scorecard("IPRF", price=20.0)
    _seed_company_and_filing(
        "IPRF",
        name="Example Insurer Depositary Shares 6.000% Non-Cumulative Preferred Series A",
        text="Depositary share preferred stock. The dividend rate is 6.000%. Liquidation preference of $25. Non-cumulative.",
        tmp_path=tmp_path,
    )
    _authorize_scorecard(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        ticker="IPRF",
        row_id=scorecard_id,
    )

    with patch("app.alpha.signal_assembler.scan_filing_risks", return_value=_no_filing_risks()):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("IPRF")

    from app.alpha.llm_tools import AlphaToolContext, dispatch_alpha_tool

    ctx = AlphaToolContext(
        sector="insurance",
        ticker="IPRF",
        packet=packet,
        as_of_date="2026-02-01",
    )
    result = dispatch_alpha_tool("fetch_insurance_evidence_packet", {}, ctx)

    assert result["status"] == "ok"
    assert result["insurance_packet"]["model_status"] == "OK"
    assert result["insurance_packet"]["valuation"]["valuation_anchor"] == 25.0


def test_alpha_report_suppresses_generic_dcf_for_insurance_security(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    scorecard_id = _seed_scorecard("IPRF", price=20.0)
    _seed_company_and_filing(
        "IPRF",
        name="Example Insurer Depositary Shares 6.000% Non-Cumulative Preferred Series A",
        text="Depositary share preferred stock. The dividend rate is 6.000%. Liquidation preference of $25. Non-cumulative.",
        tmp_path=tmp_path,
    )
    _authorize_scorecard(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        ticker="IPRF",
        row_id=scorecard_id,
    )

    with patch("app.alpha.signal_assembler.scan_filing_risks", return_value=_no_filing_risks()):
        from app.alpha.signal_assembler import assemble_signal_packet

        packet = assemble_signal_packet("IPRF")

    from app.alpha.report_writer import generate_alpha_report
    from app.alpha.schemas import SectorAlphaReport

    report = SectorAlphaReport(
        sector="insurance",
        total_candidates=1,
        rounds=[],
        winner="IPRF",
        winner_thesis="Preferred-security model is the canonical anchor.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Terms could be incomplete.",
        falsification_trigger="Terms differ from cached filing.",
        time_horizon="12 months",
        signal_packets={"IPRF": packet.to_summary_dict()},
        selection_basis="llm_decision",
    )
    path = generate_alpha_report(report, tmp_path / "alpha_report.md")
    text = path.read_text(encoding="utf-8")

    assert "Generic DCF/EPV anchors are suppressed" in text
    assert "| DCF Intrinsic |" not in text
    assert "| Liquidation Preference | $25.00 |" in text


def test_alpha_report_suppresses_generic_dcf_for_runner_up_and_finalists(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_scorecard("IPRF", price=20.0)
    _seed_scorecard("IPRG", price=18.0)

    insurance_packet = {
        "model_status": "OK",
        "generic_valuation_valid": False,
        "routing": {
            "security_type": "depositary_preferred",
            "security_identity_status": "EXPLICIT_NON_COMMON",
            "issuer_type": "insurance_underwriter",
            "insurance_subtype": "pc_insurer",
            "accounting_regime": "US_GAAP",
        },
        "valuation": {
            "method": "insurance_preferred",
            "valuation_anchor": 25.0,
            "current_price": 20.0,
            "preferred_terms": {"coupon_rate": 0.06, "cumulative": False},
            "current_yield": 0.075,
            "yield_to_worst": 0.075,
        },
        "model_blockers": [],
        "model_fit_warnings": ["GENERIC_DCF_EPV_SUPPRESSED"],
    }

    from app.alpha.report_writer import generate_alpha_report
    from app.alpha.schemas import ComparisonRound, SectorAlphaReport

    report = SectorAlphaReport(
        sector="insurance",
        total_candidates=2,
        rounds=[
            ComparisonRound(
                round_number=1,
                candidates_entering=["IPRF", "IPRG"],
                candidates_eliminated=[],
                candidates_remaining=["IPRF", "IPRG"],
                reasoning="Finalists.",
                elimination_criteria="Test.",
            )
        ],
        winner="IPRF",
        winner_thesis="Preferred-security model is canonical.",
        winner_conviction="LOW",
        runner_up="IPRG",
        runner_up_thesis="Runner-up also uses preferred-security model.",
        key_risk="Terms could be incomplete.",
        falsification_trigger="Terms differ from cached filing.",
        time_horizon="12 months",
        signal_packets={
            "IPRF": {"insurance_packet": insurance_packet},
            "IPRG": {"insurance_packet": {**insurance_packet, "ticker": "IPRG"}},
        },
        selection_basis="llm_decision",
    )
    path = generate_alpha_report(report, tmp_path / "alpha_report.md")
    text = path.read_text(encoding="utf-8")

    assert "| DCF Intrinsic |" not in text
    assert (
        "| Ticker | Price | Primary Anchor | Discount | Gate | Moat | Downside | Rev CAGR 5y |"
        in text
    )
    assert text.count("Generic DCF/EPV anchors are suppressed") >= 2
    assert "$100.00" not in text


def test_alpha_report_skips_sparse_years_and_renders_missing_values_as_dash(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_scorecard("IPRF", price=20.0)
    _seed_companyfact("IPRF", year=2026, line_item="shares_outstanding", value=10.0)
    _seed_companyfact(
        "IPRF",
        year=2025,
        line_item="revenue",
        value=100.0,
        filed_date="2026-02-01",
    )
    _seed_companyfact(
        "IPRF",
        year=2025,
        line_item="cfo",
        value=50.0,
        filed_date="2026-02-01",
    )

    from app.alpha.report_writer import generate_alpha_report
    from app.alpha.schemas import SectorAlphaReport

    report = SectorAlphaReport(
        sector="insurance",
        total_candidates=1,
        rounds=[],
        winner="IPRF",
        winner_thesis="Sparse rows should not look complete.",
        winner_conviction="LOW",
        runner_up=None,
        runner_up_thesis="",
        key_risk="Missing data.",
        falsification_trigger="Data fills in.",
        time_horizon="12 months",
        signal_packets={"IPRF": {}},
        selection_basis="llm_decision",
    )
    path = generate_alpha_report(
        report,
        tmp_path / "alpha_report.md",
        as_of_date="2026-02-13",
    )
    text = path.read_text(encoding="utf-8")

    assert "| 2026 " not in text
    assert "| 2025 | 100 | — | — | — | 50 | — |" in text


def test_alpha_report_financials_refuse_unnormalized_units():
    from app.alpha.report_writer import (
        AlphaFinancialUnitError,
        derive_alpha_report_financials,
    )

    row = {
        "ticker": "IPRF",
        "fiscal_year": 2025,
        "line_item": "revenue",
        "value": 100_000_000.0,
        "units": "USD",
    }
    with pytest.raises(AlphaFinancialUnitError) as excinfo:
        derive_alpha_report_financials([row], years=5)
    assert excinfo.value.row == row
    assert "units='USD'" in str(excinfo.value)
    assert "line_item='revenue'" in str(excinfo.value)
