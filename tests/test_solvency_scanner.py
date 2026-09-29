"""Tests for app.alpha.solvency_scanner."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.db import connect, get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _gc

    _gc.cache_clear()
    cfg = _gc()
    init_db(cfg)
    return cfg


def _seed_facts(ticker, facts_list):
    now = utc_now_iso()
    with get_db() as conn:
        for fy, pt, li, val in facts_list:
            conn.execute(
                """INSERT OR REPLACE INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item, value,
                    units, source_url, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, fy, pt, f"{fy}-12-31", li, val, "USD_millions", "", now),
            )


def _seed_filing(tmp_path, ticker, text_content, form_type="10-K"):
    filing_dir = tmp_path / "filings"
    filing_dir.mkdir(exist_ok=True)
    path = filing_dir / f"{ticker.lower()}-test.htm"
    path.write_text(f"<html><body>{text_content}</body></html>", encoding="utf-8")
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO filings
               (cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "0001234567",
                ticker,
                "0001234567-26-000001",
                form_type,
                "2026-02-26",
                "2025-12-31",
                "https://sec.gov/test",
                str(path),
                "OK",
                now,
                now,
            ),
        )
    return str(path)


_FROZEN_GOING_CONCERN_FALSE_POSITIVES = (
    (
        "IQV",
        "Any of the foregoing may have a material adverse impact on our ability to "
        "provide services to our clients or maintain our profitability. There is "
        "ongoing concern from privacy advocates, regulators and others regarding "
        "data protection and privacy issues.",
        None,
        None,
        None,
    ),
    (
        "ICE",
        "For example, starting as of September 30, 2023, Bakkt has disclosed that "
        "it is monitoring its ability to continue as a going concern, and such "
        "disclosures have continued in Bakkt's most recent SEC filings. The carrying "
        "value of our equity method investment in Bakkt was $9 million.",
        "THIRD_PARTY",
        "AFFIRMATIVE_CURRENT",
        "ANNUAL_FILING_OTHER",
    ),
    (
        "KO",
        "In addition, ongoing concern over climate change is expected to continue to "
        "result in additional legal or regulatory requirements designed to reduce "
        "or mitigate the effects of climate change.",
        None,
        None,
        None,
    ),
    (
        "APA",
        "The corresponding obligations of such parties may increase substantially, "
        "thereby causing a significant impact on the counterparties' solvency and "
        "ability to continue as a going concern.",
        "COUNTERPARTY",
        "HYPOTHETICAL",
        "ANNUAL_FILING_OTHER",
    ),
    (
        "PLTR",
        "One or more of our partners in such a relationship may independently suffer "
        "a bankruptcy or other economic hardship that negatively affects its ability "
        "to continue as a going concern or successfully perform on its obligation.",
        "PARTNER",
        "HYPOTHETICAL",
        "ANNUAL_FILING_OTHER",
    ),
    (
        "HPE",
        "Hewlett Packard Enterprise Company and subsidiaries. Notes to Consolidated "
        "Financial Statements. We consider market conditions, the ability to operate "
        "as a going concern, and other factors which indicate that the carrying "
        "amount of the investment might not be recoverable.",
        "INVESTEE",
        "ACCOUNTING_POLICY",
        "FINANCIAL_STATEMENTS_NOTES",
    ),
    (
        "BSX",
        "We consider a significant adverse change in the regulatory, economic or "
        "technological environment of an investee or a significant doubt about an "
        "investee's ability to continue as a going concern. If we identify an "
        "impairment indicator, we estimate the fair value of the investment.",
        "INVESTEE",
        "ACCOUNTING_POLICY",
        "ANNUAL_FILING_OTHER",
    ),
    (
        "AVGO",
        "We evaluate the earnings performance, credit rating, asset quality, business "
        "prospects of the investee, and financial indicators of the investee's ability "
        "to continue as a going concern. Business combinations are accounted for under "
        "the acquisition method.",
        "INVESTEE",
        "ACCOUNTING_POLICY",
        "ANNUAL_FILING_OTHER",
    ),
)


def test_critical_solvency_all_signals(monkeypatch, tmp_path):
    """Company with negative equity, low current ratio, and going concern language = CRITICAL."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST",
        [
            (2025, "FY", "equity", -16.0),
            (2025, "FY", "current_assets", 31.0),
            (2025, "FY", "current_liabilities", 50.0),
            (2025, "FY", "cash", 8.0),
            (2025, "FY", "cfo", -1.0),
            (2025, "FY", "total_debt", 15.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "TEST",
        "There is substantial doubt about our ability to continue as a going concern. "
        "No assurance can be given as to our ability to procure additional financing. "
        "We recorded a full valuation allowance against our deferred tax assets.",
    )
    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("TEST")
    assert result.solvency_risk == "CRITICAL"
    assert result.negative_equity is True
    assert result.going_concern_language is True
    assert result.no_assurance_financing is True
    assert result.valuation_allowance_full is True
    assert len(result.going_concern_assertions) == 1
    assertion = result.going_concern_assertions[0]
    assert assertion.subject == "REGISTRANT"
    assert assertion.assertion_mode == "AFFIRMATIVE_CURRENT"
    assert assertion.blockable is True
    assert assertion.corroborating_distress == (
        "NEGATIVE_EQUITY",
        "CRITICAL_LIQUIDITY_RATIO_BELOW_0_7",
        "NO_ASSURANCE_FINANCING",
    )


def test_low_solvency_healthy(monkeypatch, tmp_path):
    """Healthy company should get LOW solvency risk."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "GOOD",
        [
            (2025, "FY", "equity", 100.0),
            (2025, "FY", "current_assets", 50.0),
            (2025, "FY", "current_liabilities", 25.0),
            (2025, "FY", "cash", 30.0),
            (2025, "FY", "cfo", 20.0),
        ],
    )
    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("GOOD")
    assert result.solvency_risk == "LOW"
    assert result.negative_equity is False


def test_elevated_some_signals(monkeypatch, tmp_path):
    """Company with some but not all distress signals = ELEVATED."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "MED",
        [
            (2025, "FY", "equity", 5.0),
            (2025, "FY", "current_assets", 20.0),
            (2025, "FY", "current_liabilities", 25.0),  # ratio 0.8
            (2025, "FY", "cash", 3.0),
            (2025, "FY", "cfo", -2.0),
        ],
    )
    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("MED")
    assert result.solvency_risk == "ELEVATED"


def test_no_data_returns_unknown(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("ZZZZ")
    assert result.solvency_risk == "UNKNOWN"


@pytest.mark.parametrize(
    ("ticker", "filing_text", "expected_subject", "expected_mode", "expected_section"),
    _FROZEN_GOING_CONCERN_FALSE_POSITIVES,
    ids=("IQV", "ICE", "KO", "APA", "PLTR", "HPE", "BSX", "AVGO"),
)
def test_named_large_cap_going_concern_false_positives_never_block(
    monkeypatch,
    tmp_path,
    ticker,
    filing_text,
    expected_subject,
    expected_mode,
    expected_section,
):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        ticker,
        [
            (2025, "FY", "equity", 25.0),
            (2025, "FY", "current_assets", 40.0),
            (2025, "FY", "current_liabilities", 30.0),
            (2025, "FY", "cash", 12.0),
            (2025, "FY", "cfo", 8.0),
            (2025, "FY", "total_debt", 5.0),
        ],
    )
    _seed_filing(tmp_path, ticker, filing_text)

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency(ticker)

    assert result.going_concern_language is False
    assert result.solvency_risk == "LOW"
    assert "GOING_CONCERN_LANGUAGE" not in result.signals
    if expected_subject is None:
        assert result.going_concern_assertions == []
        return

    assert len(result.going_concern_assertions) == 1
    assertion = result.going_concern_assertions[0]
    assert assertion.subject == expected_subject
    assert assertion.assertion_mode == expected_mode
    assert assertion.blockable is False
    assert assertion.accession == "0001234567-26-000001"
    assert assertion.form_type == "10-K"
    assert assertion.filing_date == "2026-02-26"
    assert assertion.section == expected_section
    assert assertion.corroborating_distress == ()
    assert assertion.issuer_cik == "0001234567"
    assert assertion.source_url == "https://sec.gov/test"
    assert assertion.content_revision is not None
    assert "going concern" in assertion.excerpt.lower()


def test_convertible_notes_due(monkeypatch, tmp_path):
    """Filing mentioning debt due within 12 months should flag debt_due_within_12mo."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "TEST2",
        [
            (2025, "FY", "equity", 10.0),
            (2025, "FY", "total_debt", 15.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "TEST2",
        "The October 2025 Convertible Note and the November 2025 Convertible Note "
        "have repayment dates during the year ending December 31, 2026.",
    )
    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("TEST2")
    assert result.debt_due_within_12mo is True


def test_negated_going_concern_language_does_not_trigger(monkeypatch, tmp_path):
    """Negated audit language should not count as a real going-concern signal."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "NEGATE",
        [
            (2025, "FY", "equity", 25.0),
            (2025, "FY", "current_assets", 40.0),
            (2025, "FY", "current_liabilities", 30.0),
            (2025, "FY", "cash", 12.0),
            (2025, "FY", "cfo", 8.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "NEGATE",
        "Management believes substantial doubt about the Company's ability to continue "
        "as a going concern does not exist for the twelve months following issuance.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("NEGATE")
    assert result.going_concern_language is False
    assert result.solvency_risk == "LOW"
    assert len(result.going_concern_assertions) == 1
    assert result.going_concern_assertions[0].subject == "REGISTRANT"
    assert result.going_concern_assertions[0].assertion_mode == "NEGATED"
    assert result.going_concern_assertions[0].blockable is False


def test_plural_negated_going_concern_language_does_not_trigger(monkeypatch, tmp_path):
    """2026-07-20 audit GC-3: plural "do not raise" is a negation, not distress."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "NEGPLU",
        [
            (2025, "FY", "equity", 25.0),
            (2025, "FY", "current_assets", 40.0),
            (2025, "FY", "current_liabilities", 30.0),
            (2025, "FY", "cash", 12.0),
            (2025, "FY", "cfo", 8.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "NEGPLU",
        "The conditions and events described above do not raise substantial doubt "
        "about our ability to continue as a going concern.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("NEGPLU")
    assert result.going_concern_language is False
    assert result.solvency_risk == "LOW"
    assert len(result.going_concern_assertions) == 1
    assert result.going_concern_assertions[0].subject == "REGISTRANT"
    assert result.going_concern_assertions[0].assertion_mode == "NEGATED"
    assert result.going_concern_assertions[0].blockable is False


def test_statutory_doubt_exists_despite_plan_modal_triggers(monkeypatch, tmp_path):
    """2026-07-20 audit GC-1: "plans may not be implemented, and substantial doubt
    exists" asserts current doubt — the modal governs the plans, not the doubt."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "GCSTAT",
        [
            (2025, "FY", "equity", 25.0),
            (2025, "FY", "current_assets", 40.0),
            (2025, "FY", "current_liabilities", 30.0),
            (2025, "FY", "cash", 12.0),
            (2025, "FY", "cfo", 8.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "GCSTAT",
        "Management's plans may not be successfully implemented, and substantial "
        "doubt exists about the Company's ability to continue as a going concern.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("GCSTAT")
    assert result.going_concern_language is True
    assert result.solvency_risk == "ELEVATED"
    assert len(result.going_concern_assertions) == 1
    assertion = result.going_concern_assertions[0]
    assert assertion.subject == "REGISTRANT"
    assert assertion.subject_detail == "GCSTAT"
    assert assertion.assertion_mode == "AFFIRMATIVE_CURRENT"
    assert assertion.blockable is True


def test_hypothetical_going_concern_language_does_not_trigger(monkeypatch, tmp_path):
    """Risk-factor hypotheticals should not be treated as present going-concern distress."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "HYPOTH",
        [
            (2025, "FY", "equity", 25.0),
            (2025, "FY", "current_assets", 40.0),
            (2025, "FY", "current_liabilities", 30.0),
            (2025, "FY", "cash", 12.0),
            (2025, "FY", "cfo", 8.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "HYPOTH",
        "If we fail to satisfy future covenant tests, there could be substantial doubt "
        "about our ability to continue as a going concern.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("HYPOTH")
    assert result.going_concern_language is False
    assert result.solvency_risk == "LOW"
    assert len(result.going_concern_assertions) == 1
    assert result.going_concern_assertions[0].subject == "REGISTRANT"
    assert result.going_concern_assertions[0].assertion_mode == "HYPOTHETICAL"
    assert result.going_concern_assertions[0].blockable is False


def test_affirmative_going_concern_without_other_distress_is_elevated(monkeypatch, tmp_path):
    """An affirmative mention alone should elevate risk, not force CRITICAL."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "GCWARN",
        [
            (2025, "FY", "equity", 25.0),
            (2025, "FY", "current_assets", 40.0),
            (2025, "FY", "current_liabilities", 30.0),
            (2025, "FY", "cash", 12.0),
            (2025, "FY", "cfo", 8.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "GCWARN",
        "These conditions raise substantial doubt about our ability to continue as a going concern.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("GCWARN")
    assert result.going_concern_language is True
    assert result.solvency_risk == "ELEVATED"
    assert len(result.going_concern_assertions) == 1
    assertion = result.going_concern_assertions[0]
    assert assertion.subject == "REGISTRANT"
    assert assertion.subject_detail == "GCWARN"
    assert assertion.assertion_mode == "AFFIRMATIVE_CURRENT"
    assert assertion.blockable is True
    assert assertion.accession == "0001234567-26-000001"
    assert assertion.form_type == "10-K"
    assert assertion.filing_date == "2026-02-26"
    assert assertion.section == "ANNUAL_FILING_OTHER"
    assert assertion.excerpt == (
        "These conditions raise substantial doubt about our ability to continue as a going concern."
    )
    assert assertion.corroborating_distress == ()


def test_affirmative_consolidated_subsidiary_assertion_blocks_even_with_net_cash(
    monkeypatch,
    tmp_path,
):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "SUBGC",
        [
            (2025, "FY", "equity", 500.0),
            (2025, "FY", "current_assets", 1000.0),
            (2025, "FY", "current_liabilities", 100.0),
            (2025, "FY", "cash", 500.0),
            (2025, "FY", "cfo", 100.0),
            (2025, "FY", "total_debt", 0.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "SUBGC",
        "Management of the Company's wholly owned consolidated subsidiary concluded "
        "that substantial doubt exists about the subsidiary's ability to continue as "
        "a going concern.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("SUBGC")

    assert result.going_concern_language is True
    assert result.solvency_risk == "ELEVATED"
    assert "fortress" not in result.details.lower()
    assert len(result.going_concern_assertions) == 1
    assertion = result.going_concern_assertions[0]
    assert assertion.subject == "CONSOLIDATED_SUBSIDIARY"
    assert assertion.assertion_mode == "AFFIRMATIVE_CURRENT"
    assert assertion.blockable is True
    assert assertion.corroborating_distress == ()


def test_no_assurance_competition_language_does_not_trigger_financing_flag(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "COMPETE",
        [
            (2025, "FY", "equity", 1000.0),
            (2025, "FY", "current_assets", 500.0),
            (2025, "FY", "current_liabilities", 250.0),
            (2025, "FY", "cash", 200.0),
            (2025, "FY", "cfo", 100.0),
            (2025, "FY", "total_debt", 100.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "COMPETE",
        "There can be no assurance that we will continue to compete successfully in the future. "
        "We may decide to enter into additional debt arrangements as part of ordinary capital management.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("COMPETE")
    assert result.no_assurance_financing is False
    assert result.solvency_risk == "LOW"


def test_no_assurance_investment_language_does_not_trigger_financing_flag(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "INVEST",
        [
            (2025, "FY", "equity", 1000.0),
            (2025, "FY", "current_assets", 500.0),
            (2025, "FY", "current_liabilities", 250.0),
            (2025, "FY", "cash", 200.0),
            (2025, "FY", "cfo", 100.0),
            (2025, "FY", "total_debt", 100.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "INVEST",
        "There is no assurance that we will enter into an investment and partnership agreement "
        "or that a transaction will be completed. Our investment portfolio contains industry "
        "sector concentration risks and capital market volatility.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("INVEST")
    assert result.no_assurance_financing is False
    assert result.solvency_risk == "LOW"


def test_no_assurance_financing_language_still_triggers(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "FINRISK",
        [
            (2025, "FY", "equity", 20.0),
            (2025, "FY", "current_assets", 30.0),
            (2025, "FY", "current_liabilities", 20.0),
            (2025, "FY", "cash", 5.0),
            (2025, "FY", "cfo", 2.0),
            (2025, "FY", "total_debt", 25.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "FINRISK",
        "No assurance can be given as to our ability to procure additional financing "
        "on acceptable terms when required.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("FINRISK")
    assert result.no_assurance_financing is True
    assert result.solvency_risk == "ELEVATED"


# ---------------------------------------------------------------------------
# Net-cash fortress override tests
# (Fixes HRMY-style false positives where going-concern boilerplate in a 10-K
# flipped a $589M net-cash company to CRITICAL.)
# ---------------------------------------------------------------------------


def test_is_net_cash_fortress_relative():
    """Net cash >= 20% of market cap → fortress via the relative test."""
    from app.alpha.solvency_scanner import _is_net_cash_fortress

    # HRMY-like: $753M cash, $164M debt, $1.7B mcap → $589M net cash = 35% of mcap
    assert _is_net_cash_fortress(753, 164, 1700) is True

    # Neither test triggers: small absolute net cash AND small % of mcap
    # Net cash = $50M (below $100M absolute floor) and 5% of mcap (below 20%)
    assert _is_net_cash_fortress(50, 0, 1000) is False

    # Moderate absolute net cash but heavily leveraged — net_cash NOT >= 2x debt
    # $400M cash, $350M debt = $50M net cash, 10% of mcap → fails both
    assert _is_net_cash_fortress(400, 350, 500) is False


def test_is_net_cash_fortress_absolute_no_mcap():
    """When mcap is unknown, $100M+ net cash with low debt still triggers."""
    from app.alpha.solvency_scanner import _is_net_cash_fortress

    # $500M cash, $100M debt, no mcap known → $400M net cash, 4x debt → fortress
    assert _is_net_cash_fortress(500, 100, None) is True

    # $80M net cash is below absolute threshold
    assert _is_net_cash_fortress(80, 0, None) is False

    # $200M cash but $150M debt → net_cash=50M < threshold
    assert _is_net_cash_fortress(200, 150, None) is False


def test_is_net_cash_fortress_requires_positive_net_cash():
    """Heavily leveraged company is NOT a fortress even with large absolute cash."""
    from app.alpha.solvency_scanner import _is_net_cash_fortress

    # Boeing-like: $10B cash, $50B debt → net_cash is negative
    assert _is_net_cash_fortress(10_000, 50_000, 100_000) is False
    assert _is_net_cash_fortress(0, 100, 1000) is False
    assert _is_net_cash_fortress(None, 100, 1000) is False


def test_is_net_cash_fortress_requires_explicit_debt():
    """Missing debt is unavailable, not a semantic zero-debt fact."""
    from app.alpha.solvency_scanner import _is_net_cash_fortress

    assert _is_net_cash_fortress(500, None, 1000) is False
    assert _is_net_cash_fortress(500, 0, 1000) is True


def test_net_cash_override_beats_going_concern_boilerplate(monkeypatch, tmp_path):
    """HRMY regression: innocent 'going concern' boilerplate in a 10-K must NOT
    flip a net-cash fortress to CRITICAL."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "HEALTHY",
        [
            (2025, "FY", "equity", 2000.0),
            (2025, "FY", "current_assets", 1000.0),
            (2025, "FY", "current_liabilities", 500.0),
            (2025, "FY", "cash", 753.0),
            (2025, "FY", "cfo", 350.0),
            (2025, "FY", "total_debt", 164.0),
        ],
    )
    # Filing contains going-concern boilerplate (common in risk factors)
    _seed_filing(
        tmp_path,
        "HEALTHY",
        "If we fail to meet certain covenants, there could be substantial doubt "
        "about our ability to continue as a going concern in the future. "
        "No assurance can be given that we will secure future financing.",
    )

    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("HEALTHY")
    # Net-cash override fires: $589M net cash overwhelms all text-based signals
    assert result.solvency_risk == "LOW"
    assert "fortress" in result.details.lower() or "net cash" in result.details.lower()


def test_net_cash_override_respects_negative_equity(monkeypatch, tmp_path):
    """If equity is actually negative, net-cash override must NOT fire.
    Negative equity is a real red flag distinct from cash position (e.g.,
    companies that bought back so much stock they went negative).
    """
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "NEGEQUITY",
        [
            (2025, "FY", "equity", -50.0),  # negative
            (2025, "FY", "cash", 500.0),
            (2025, "FY", "total_debt", 100.0),
            (2025, "FY", "current_assets", 600.0),
            (2025, "FY", "current_liabilities", 200.0),
        ],
    )
    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("NEGEQUITY")
    # Override should NOT fire — negative equity signal preserved
    assert result.solvency_risk != "LOW"
    assert result.negative_equity is True


def test_critical_stays_critical_for_real_distress(monkeypatch, tmp_path):
    """Regression guard: a truly distressed company (low cash, high debt, going concern)
    must still be flagged CRITICAL after this fix."""
    _init_temp_db(monkeypatch, tmp_path)
    _seed_facts(
        "DISTRESS",
        [
            (2025, "FY", "equity", -16.0),
            (2025, "FY", "current_assets", 31.0),
            (2025, "FY", "current_liabilities", 50.0),
            (2025, "FY", "cash", 8.0),
            (2025, "FY", "cfo", -1.0),
            (2025, "FY", "total_debt", 50.0),
        ],
    )
    _seed_filing(
        tmp_path,
        "DISTRESS",
        "There is substantial doubt about our ability to continue as a going concern. "
        "No assurance can be given as to our ability to procure additional financing.",
    )
    from app.alpha.solvency_scanner import assess_solvency

    result = assess_solvency("DISTRESS")
    assert result.solvency_risk == "CRITICAL"


def test_v2_solvency_binds_cik_filed_asof_filing_and_explicit_db(tmp_path):
    db_path = tmp_path / "issuer-solvency.db"
    safe_filing = tmp_path / "safe.htm"
    safe_filing.write_text("<html><body>Ordinary annual risk disclosure.</body></html>")
    contaminated_filing = tmp_path / "contaminated.htm"
    contaminated_filing.write_text(
        "<html><body>There is substantial doubt about our ability to continue "
        "as a going concern.</body></html>"
    )
    conn = connect(db_path)
    try:
        init_db(conn=conn)
        now = utc_now_iso()
        facts = (
            ("ORD", 2024, "equity", 100.0, "2025-02-15", 42, "0000000042-25-000001"),
            (
                "ORD",
                2024,
                "current_assets",
                200.0,
                "2025-02-15",
                42,
                "0000000042-25-000001",
            ),
            (
                "ORD",
                2024,
                "current_liabilities",
                100.0,
                "2025-02-15",
                42,
                "0000000042-25-000001",
            ),
            ("ORD", 2024, "cash", 10.0, "2025-02-15", 42, "0000000042-25-000001"),
            ("ORD", 2024, "cfo", 20.0, "2025-02-15", 42, "0000000042-25-000001"),
            ("ORD", 2026, "equity", -800.0, "2027-02-15", 42, "0000000042-27-000001"),
            ("ADR", 2024, "equity", -900.0, "2025-02-15", 43, "0000000043-25-000001"),
        )
        for ticker, year, line_item, value, filed_date, cik, accession in facts:
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession)
                   VALUES (?, ?, 'FY', ?, ?, ?, 'USD_millions', ?, ?, ?,
                           '10-K', ?)""",
                (
                    ticker,
                    year,
                    f"{year}-12-31",
                    line_item,
                    value,
                    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                    now,
                    filed_date,
                    accession,
                ),
            )
        for cik, ticker, accession, filing_date, path in (
            ("42", "ORD", "0000000042-25-000001", "2025-02-20", safe_filing),
            ("43", "ADR", "0000000043-25-000001", "2025-03-20", contaminated_filing),
        ):
            conn.execute(
                """INSERT INTO filings
                   (cik, ticker, accession, form_type, filing_date, period_end,
                    primary_doc_url, local_path, status, created_at, updated_at)
                   VALUES (?, ?, ?, '10-K', ?, '2024-12-31', ?, ?, 'OK', ?, ?)""",
                (
                    cik,
                    ticker,
                    accession,
                    filing_date,
                    f"https://www.sec.gov/Archives/{accession}.htm",
                    str(path),
                    now,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    from app.alpha.solvency_scanner import assess_solvency

    with patch(
        "app.alpha.solvency_scanner.get_db",
        side_effect=AssertionError("explicit db_path must not open the configured DB"),
    ):
        result = assess_solvency(
            "ADR",
            as_of_date="2025-06-30",
            require_filed_asof=True,
            issuer_cik="42",
            aliases=("ADR", "ORD"),
            db_path=db_path,
        )

    assert result.solvency_risk == "LOW"
    assert result.negative_equity is False
    assert result.going_concern_language is False


def _seed_strict_historical_solvency_fixture(
    *,
    db_path,
    filing_path,
    filing_date: str,
    filing_accession: str,
) -> None:
    conn = connect(db_path)
    try:
        init_db(conn=conn)
        now = utc_now_iso()
        for line_item, value in (
            ("equity", 100.0),
            ("current_assets", 200.0),
            ("current_liabilities", 100.0),
            ("cash", 10.0),
            ("cfo", 20.0),
            ("total_debt", 5.0),
        ):
            conn.execute(
                """INSERT INTO companyfacts_facts
                   (ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, source_url, fetched_at, filed_date, form,
                    accession)
                   VALUES ('HIST', 2023, 'FY', '2023-12-31', ?, ?,
                           'USD_millions',
                           'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json',
                           ?, '2024-02-15', '10-K', '0000000042-24-000001')""",
                (line_item, value, now),
            )
        conn.execute(
            """INSERT INTO filings
               (cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at)
               VALUES ('42', 'HIST', ?, '10-K', ?, '2023-12-31', ?, ?,
                       'OK', ?, ?)""",
            (
                filing_accession,
                filing_date,
                f"https://www.sec.gov/Archives/{filing_accession}.htm",
                str(filing_path),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_strict_historical_solvency_never_reads_only_post_cutoff_annual(
    tmp_path,
):
    db_path = tmp_path / "future-only.db"
    future_filing = tmp_path / "future-annual.htm"
    future_filing.write_text(
        "<html><body>There is substantial doubt about our ability to continue "
        "as a going concern.</body></html>",
        encoding="utf-8",
    )
    _seed_strict_historical_solvency_fixture(
        db_path=db_path,
        filing_path=future_filing,
        filing_date="2025-02-20",
        filing_accession="0000000042-25-000001",
    )

    from app.alpha.solvency_scanner import assess_solvency

    with (
        patch(
            "app.alpha.filing_risk_scan._find_latest_annual_path",
            return_value=(str(future_filing), "10-K"),
        ) as legacy_fallback,
        patch(
            "app.research.filing_context._download_primary_document_to_raw",
            side_effect=AssertionError("strict historical solvency must remain local-only"),
        ) as network_materialization,
    ):
        result = assess_solvency(
            "HIST.A",
            as_of_date="2024-06-30",
            require_filed_asof=True,
            issuer_cik="42",
            aliases=("HIST", "HIST.A"),
            db_path=db_path,
        )

    legacy_fallback.assert_not_called()
    network_materialization.assert_not_called()
    assert result.solvency_risk == "LOW"
    assert result.going_concern_language is False
    assert result.going_concern_assertions == []
    assert "GOING_CONCERN_LANGUAGE" not in result.signals


def test_strict_historical_solvency_reads_pre_cutoff_issuer_bound_annual(
    tmp_path,
):
    db_path = tmp_path / "pre-cutoff.db"
    historical_filing = tmp_path / "historical-annual.htm"
    historical_filing.write_text(
        "<html><body>These conditions raise substantial doubt about our ability "
        "to continue as a going concern.</body></html>",
        encoding="utf-8",
    )
    _seed_strict_historical_solvency_fixture(
        db_path=db_path,
        filing_path=historical_filing,
        filing_date="2024-02-20",
        filing_accession="0000000042-24-000001",
    )

    from app.alpha.solvency_scanner import assess_solvency

    with (
        patch(
            "app.alpha.filing_risk_scan._find_latest_annual_path",
            side_effect=AssertionError("bounded evidence must not use legacy fallback"),
        ) as legacy_fallback,
        patch(
            "app.research.filing_context._download_primary_document_to_raw",
            side_effect=AssertionError("strict historical solvency must remain local-only"),
        ) as network_materialization,
    ):
        result = assess_solvency(
            "HIST.A",
            as_of_date="2024-06-30",
            require_filed_asof=True,
            issuer_cik="42",
            aliases=("HIST", "HIST.A"),
            db_path=db_path,
        )

    legacy_fallback.assert_not_called()
    network_materialization.assert_not_called()
    assert result.solvency_risk == "ELEVATED"
    assert result.going_concern_language is True
    assert len(result.going_concern_assertions) == 1
    assertion = result.going_concern_assertions[0]
    assert assertion.accession == "0000000042-24-000001"
    assert assertion.filing_date == "2024-02-20"
    assert assertion.issuer_cik == "42"
    assert assertion.blockable is True


# --- filed assertion, not a keyword hit (going-concern BLOCK provenance) ---------------
# Sentences that mention the phrase while saying the doubt is absent, or while describing
# the ASC 205-40 evaluation process, used to come back AFFIRMATIVE_CURRENT about the
# registrant and therefore blockable. The affirmative controls below must keep blocking.

_NOT_ASSERTIONS = (
    (
        "Management evaluates whether there are conditions and events, considered in the "
        "aggregate, that raise substantial doubt about the Company's ability to continue as "
        "a going concern within one year after the date the financial statements are issued.",
        "ACCOUNTING_POLICY",
    ),
    (
        "We evaluate whether substantial doubt about our ability to continue as a going "
        "concern exists.",
        "ACCOUNTING_POLICY",
    ),
    (
        "Our independent registered public accounting firm's report does not include an "
        "explanatory paragraph regarding our ability to continue as a going concern.",
        "NEGATED",
    ),
    (
        "We do not believe there is substantial doubt about our ability to continue as a "
        "going concern.",
        "NEGATED",
    ),
    (
        "Based on this evaluation, management concluded that no conditions or events raise "
        "substantial doubt about the Company's ability to continue as a going concern.",
        "NEGATED",
    ),
    (
        "We are not aware of any conditions that raise substantial doubt about our ability "
        "to continue as a going concern.",
        "NEGATED",
    ),
    (
        "We have not concluded that there is substantial doubt about our ability to "
        "continue as a going concern.",
        "NEGATED",
    ),
    (
        "In each of the past three years our auditor did not express substantial doubt "
        "about our ability to continue as a going concern.",
        "NEGATED",
    ),
)

_STILL_ASSERTIONS = (
    "The Company has suffered recurring losses from operations and has a net capital "
    "deficiency that raise substantial doubt about its ability to continue as a going concern.",
    "There is substantial doubt about our ability to continue as a going concern.",
    "Management has concluded that substantial doubt exists about the Company's ability to "
    "continue as a going concern.",
    "We have not identified sources of financing, and there is substantial doubt about our "
    "ability to continue as a going concern.",
    "Management evaluated whether there are conditions and events that raise substantial "
    "doubt and concluded that substantial doubt exists about the Company's ability to "
    "continue as a going concern.",
)


@pytest.mark.parametrize(("text", "mode"), _NOT_ASSERTIONS)
def test_negations_and_evaluation_language_are_not_blockable_assertions(text, mode):
    from app.alpha.solvency_scanner import detect_going_concern_assertions

    assertions = detect_going_concern_assertions(text, ticker="APP")
    assert [(a.assertion_mode, a.blockable) for a in assertions] == [(mode, False)]


@pytest.mark.parametrize("text", _STILL_ASSERTIONS)
def test_real_going_concern_assertions_still_block(text):
    from app.alpha.solvency_scanner import detect_going_concern_assertions

    assertions = detect_going_concern_assertions(text, ticker="XYZ")
    assert [(a.assertion_mode, a.subject, a.blockable) for a in assertions] == [
        ("AFFIRMATIVE_CURRENT", "REGISTRANT", True)
    ]


def test_going_concern_asserted_needs_the_flag_and_a_stored_blockable_excerpt():
    from app.alpha.schemas import GoingConcernAssertion, SolvencyAssessment
    from app.alpha.solvency_scanner import going_concern_asserted

    filed = GoingConcernAssertion(
        subject="REGISTRANT",
        assertion_mode="AFFIRMATIVE_CURRENT",
        accession="0000000001-26-000001",
        form_type="10-K",
        filing_date="2026-02-20",
        section="NOTES",
        excerpt="There is substantial doubt about our ability to continue as a going concern.",
        blockable=True,
    )
    assert going_concern_asserted(
        SolvencyAssessment(
            solvency_risk="ELEVATED", going_concern_language=True, going_concern_assertions=[filed]
        )
    )
    # The dict form the packets and tool payloads carry.
    assert going_concern_asserted(
        {"going_concern_language": True, "going_concern_assertions": [filed.to_dict()]}
    )
    # A bare flag, a missing/empty list, a non-blockable assertion, or an excerpt-less one is not.
    assert not going_concern_asserted({"going_concern_language": True})
    assert not going_concern_asserted(
        {"going_concern_language": True, "going_concern_assertions": []}
    )
    assert not going_concern_asserted(
        {
            "going_concern_language": True,
            "going_concern_assertions": [{**filed.to_dict(), "blockable": False}],
        }
    )
    assert not going_concern_asserted(
        {
            "going_concern_language": True,
            "going_concern_assertions": [{**filed.to_dict(), "excerpt": " "}],
        }
    )
    # An assertion with the flag off is not a block either.
    assert not going_concern_asserted(
        {"going_concern_language": False, "going_concern_assertions": [filed.to_dict()]}
    )
    assert not going_concern_asserted(None)
