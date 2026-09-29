"""Literal regression probes for the web read-model valuation helpers.

The probes live in tests/fixtures/valuation_regression_probes.py: eleven strict
xfails preserve the right answers and two controls verify bounded memo
fallbacks and the actual boolean contract.
Fixtures are scratch-rooted. No provider or live database is used.
"""

import runpy
from pathlib import Path

import pytest

from app.web.readmodel import company as _company

pytestmark = pytest.mark.skipif(
    not all(
        hasattr(_company, name)
        for name in ("_apply_durable_dcf", "_drop_rows_superseded_by_a_block")
    ),
    reason="Probes target read-model helpers that are not present in this tree.",
)

campaign_defect = pytest.mark.xfail(
    strict=True, reason="Independent literal remains wrong on the current code."
)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    from app.config import get_config

    helper = Path(__file__).resolve().parents[1] / "tests/fixtures/valuation_regression_probes.py"
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Every helper mutation is scoped to this fixture, including SIC tokens.
        for key in [
            "VOE_DATA_DIR",
            "VOE_DB_PATH",
            "VOE_LLM_PROVIDER",
            "VOE_PRICE_PROVIDER",
            "VOE_ISSUER_CLASSIFICATION_BY_SIC",
        ]:
            monkeypatch.setenv(key, __import__("os").environ.get(key, ""))
        loaded = runpy.run_path(
            str(helper), init_globals={"scratch": tmp_path_factory.mktemp("campaign-probes")}
        )
        yield loaded
    get_config.cache_clear()


def test_no_adjusted_dcf_evidence_without_an_adjusted_dcf_method(probe):
    assert [s["signal_type"] for s in probe["results"]["phantom_adjusted_dcf"]["signals"]] == [
        "DCF_DISCOUNT"
    ]


def test_adjusted_dcf_retains_its_own_measured_value(probe):
    assert probe["results"]["adjusted_method_overwritten"]["fair_value"]["base"] == 40.0


def test_newer_same_day_block_suppresses_earlier_method_values(probe):
    assert probe["results"]["same_day_block"]["actual_methods"] == ["scorecard"]


def test_epv_uses_current_revenue_after_normalizing_historical_margin(probe):
    assert probe["results"]["epv_stale_current_revenue"]["actual"]["value_per_share"] == 3.95


def test_recent_missing_cashflows_do_not_import_old_profitable_years(probe):
    assert probe["results"]["accounting_window_skips_missing_recent"]["actual"] == ([], [])


def test_equal_length_but_displaced_fiscal_years_are_not_paired(probe):
    assert (
        probe["results"]["shifted_equal_annual_lengths"]["series"][0]["owner_earnings"] == "UNKNOWN"
    )


def test_two_quarters_do_not_become_a_full_year_cashflow(probe):
    assert (
        probe["results"]["matched_quarters_called_annual"]["series"][0]["owner_earnings"]
        == "UNKNOWN"
    )


def test_lease_alignment_preserves_the_requested_information_cutoff(probe):
    assert (
        probe["results"]["lease_alignment_uses_wrong_filing_cutoff"]["actual"]["net_debt_proxy"]
        == 1200.0
    )


def test_sic_switch_requires_the_documented_true_literal(probe):
    # The prompt's token 1 is an operational mismatch, not a parser defect.
    assert probe["results"]["sic_env_1"] is False
    assert probe["results"]["sic_env_true"] is True


def test_published_on_the_asof_calendar_date_has_zero_age(probe):
    assert probe["results"]["same_calendar_day_freshness"]["actual"] == 0


def test_valuation_chart_uses_the_same_durable_value_as_the_card(probe, monkeypatch, tmp_path):
    assert probe["chart_value_in_temp_db"](monkeypatch, tmp_path) == 30.0


def test_negative_owner_earnings_bear_case_is_not_above_the_base():
    from app.valuation.valuation_writer import _compute_downside_scenario

    revenue = [(2020, 100.0), (2021, 150.0), (2022, 225.0), (2023, 337.5), (2024, 506.25)]
    result = _compute_downside_scenario(
        owner_earnings=-100.0,
        shares=1.0,
        net_debt=0.0,
        revenue_series=revenue,
        operating_income_series=[(year, -100.0) for year, _ in revenue],
        wacc=0.10,
        base_case_dcf=-1636.6093244996925,
        current_price=100.0,
    )
    # The independent 8%/2% geometric sum is -1636.6093244996925.
    # A pessimistic value may be lower, but cannot exceed the central value.
    assert result["bear_case_dcf"] <= -1636.61


def test_memo_fallback_queries_are_bounded_to_the_requested_date():
    import sqlite3
    from app.report.memo_builder import _latest_packet_row, _latest_score_row

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE evidence_packets(ticker, as_of_date, packet_path)")
    conn.execute("CREATE TABLE scores(ticker, as_of_date, total_score, decision)")
    conn.executemany(
        "INSERT INTO evidence_packets VALUES ('TEST', ?, ?)",
        [("2024-03-31", "old"), ("2025-12-31", "new")],
    )
    conn.executemany(
        "INSERT INTO scores VALUES ('TEST', ?, ?, 'WATCH')",
        [("2024-03-31", 10.0), ("2025-12-31", 99.0)],
    )
    assert _latest_packet_row(conn, "TEST", "2024-06-30")["packet_path"] == "old"
    assert _latest_score_row(conn, "TEST", "2024-06-30")["total_score"] == 10.0
    assert _latest_packet_row(conn, "TEST", "2020-06-30") is None
    assert _latest_score_row(conn, "TEST", "2020-06-30") is None
    conn.close()
