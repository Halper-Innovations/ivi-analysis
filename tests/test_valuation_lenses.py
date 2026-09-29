"""Tests for the complementary valuation lenses (goal task ENHANCE):
EV/EBIT multiple anchor, FCF-yield anchor, tangible-asset/liquidation floor.

Each lens is deterministic, point-in-time-safe (driven off the as-of facts
dict), persisted per (ticker, as_of, method), and NEVER gates — they feed the
scorecard and select_anchor() provenance only.
"""

from __future__ import annotations

import json

import pytest

from app.db import get_db, init_db, utc_now_iso


def _facts(**overrides):
    base = {
        "revenue": [(2023, 1000.0), (2022, 950.0), (2021, 900.0)],
        "operating_income": [(2023, 200.0), (2022, 190.0), (2021, 180.0)],
        "cfo": [(2023, 200.0), (2022, 185.0), (2021, 170.0)],
        "capex": [(2023, 35.0), (2022, 32.0), (2021, 30.0)],
        "equity": [(2023, 600.0), (2022, 550.0), (2021, 500.0)],
        "shares_outstanding": [(2023, 100.0), (2022, 100.0), (2021, 100.0)],
    }
    base.update(overrides)
    return base


# ── EV/EBIT multiple anchor ───────────────────────────────────────────────────


def test_ev_ebit_anchor_basic():
    from app.valuation.lenses import ev_ebit_anchor

    result = ev_ebit_anchor(
        _facts(),
        shares=100.0,
        bridge_deduction=100.0,
        category="TRADITIONAL_OPERATING",
        current_price=10.0,
    )
    assert result["status"] == "OK"
    # EBIT median(200, 190, 180) = 190; multiple 8x -> EV 1520; equity 1420
    assert result["value_per_share"] == pytest.approx(14.2)
    assert result["basis"]["ebit_normalized"] == 190.0
    assert result["basis"]["multiple"] == 8.0
    assert (
        result["basis"]["convention"]
        == "ENTERPRISE (EV/EBIT); equity = multiple x EBIT - net debt - senior claims"
    )


def test_ev_ebit_anchor_negative_earnings_power_not_meaningful():
    from app.valuation.lenses import ev_ebit_anchor

    facts = _facts(operating_income=[(2023, -50.0), (2022, -40.0), (2021, -30.0)])
    result = ev_ebit_anchor(
        facts, shares=100.0, bridge_deduction=0.0, category="TRADITIONAL_OPERATING"
    )
    assert result["status"] == "EV_EBIT_NOT_MEANINGFUL"
    assert result["value_per_share"] is None


def test_ev_ebit_anchor_flags_rich_current_multiple():
    from app.valuation.lenses import ev_ebit_anchor

    # price 30 -> market EV = 3000 + 100 = 3100 vs EBIT 190 -> ~16.3x >> 8x
    result = ev_ebit_anchor(
        _facts(),
        shares=100.0,
        bridge_deduction=100.0,
        category="TRADITIONAL_OPERATING",
        current_price=30.0,
    )
    assert "EV_EBIT_RICH_VS_ANCHOR_MULTIPLE" in result["flags"]
    assert result["basis"]["implied_current_multiple"] == pytest.approx(3100.0 / 190.0)


# ── FCF-yield anchor ──────────────────────────────────────────────────────────


def test_fcf_yield_anchor_basic():
    from app.valuation.lenses import fcf_yield_anchor

    result = fcf_yield_anchor(_facts(), shares=100.0, category="TRADITIONAL_OPERATING")
    assert result["status"] == "OK"
    # FCF by common year: 165, 153, 140 -> median 153; equity basis (FCFE-ish,
    # CFO is post-interest): 153 / 0.08 = 1912.5 -> 19.125/share, NO net-debt
    # deduction (that would double-count debt service).
    assert result["value_per_share"] == pytest.approx(19.125)
    assert result["basis"]["required_yield"] == 0.08
    assert "EQUITY" in result["basis"]["convention"]


def test_fcf_yield_anchor_negative_fcf_not_meaningful():
    from app.valuation.lenses import fcf_yield_anchor

    facts = _facts(capex=[(2023, 300.0), (2022, 300.0), (2021, 300.0)])
    result = fcf_yield_anchor(facts, shares=100.0, category="TRADITIONAL_OPERATING")
    assert result["status"] == "FCF_YIELD_NOT_MEANINGFUL"
    assert result["value_per_share"] is None


# ── tangible-asset / liquidation floor ────────────────────────────────────────


def test_tangible_floor_basic():
    from app.valuation.lenses import tangible_floor

    facts = _facts(
        goodwill=[(2023, 100.0)],
        intangible_assets=[(2023, 50.0)],
    )
    result = tangible_floor(facts, shares=100.0, ncav_value_per_share=1.0)
    assert result["status"] == "OK"
    # tangible equity = 600 - 100 - 50 = 450 -> tangible BVPS 4.50
    assert result["basis"]["tangible_book_per_share"] == pytest.approx(4.5)
    # floor = max(NCAV 1.0, 0.65 * 4.50 = 2.925)
    assert result["value_per_share"] == pytest.approx(2.925)
    assert result["basis"]["floor_rule"] == "max(corrected NCAV, 0.65 x tangible BVPS)"


def test_tangible_floor_negative_tangible_book():
    from app.valuation.lenses import tangible_floor

    facts = _facts(
        goodwill=[(2023, 500.0)],
        intangible_assets=[(2023, 200.0)],
    )
    result = tangible_floor(facts, shares=100.0, ncav_value_per_share=None)
    assert result["status"] == "NO_TANGIBLE_FLOOR"
    assert result["value_per_share"] is None
    assert result["basis"]["tangible_book_per_share"] == pytest.approx(-1.0)


def test_tangible_floor_deducts_senior_claims():
    from app.valuation.lenses import tangible_floor

    facts = _facts(
        goodwill=[(2023, 100.0)],
        intangible_assets=[(2023, 50.0)],
        preferred_equity=[(2023, 50.0)],
        noncontrolling_interest=[(2023, 100.0)],
    )
    result = tangible_floor(facts, shares=100.0, ncav_value_per_share=None)
    # tangible common equity = 600 - 100 - 50 - 50 - 100 = 300 -> 3.0/share
    assert result["basis"]["tangible_book_per_share"] == pytest.approx(3.0)


# ── persistence + provenance integration ─────────────────────────────────────


def test_lenses_persisted_and_in_provenance(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())

    fields = {
        "revenue": [900.0, 950.0, 1000.0],
        "operating_income": [180.0, 190.0, 200.0],
        "net_income": [120.0, 130.0, 140.0],
        "equity": [500.0, 550.0, 600.0],
        "cfo": [170.0, 185.0, 200.0],
        "capex": [30.0, 32.0, 35.0],
        "shares_outstanding": [100.0, 100.0, 100.0],
        "total_debt": [200.0, 200.0, 200.0],
        "cash": [80.0, 90.0, 100.0],
        "preferred_equity": [0.0, 0.0, 0.0],
        "noncontrolling_interest": [0.0, 0.0, 0.0],
        "goodwill": [90.0, 95.0, 100.0],
        "intangible_assets": [40.0, 45.0, 50.0],
    }
    with get_db() as conn:
        for line_item, values in fields.items():
            for year, value in zip([2021, 2022, 2023], values, strict=True):
                conn.execute(
                    "INSERT INTO companyfacts_facts "
                    "(ticker, fiscal_year, period_type, period_end, line_item, "
                    "value, units, source_url, fetched_at, filed_date, accession) "
                    "VALUES ('LENSX', ?, 'FY', ?, ?, ?, 'USD', ?, ?, ?, ?)",
                    (
                        year,
                        f"{year}-12-31",
                        line_item,
                        float(value),
                        "https://example.test/companyfacts/LENSX",
                        utc_now_iso(),
                        f"{year + 1}-02-15",
                        f"LENSX-{year}",
                    ),
                )
        conn.commit()

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("LENSX", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        methods = {
            r["method"]: json.loads(r["outputs_json"] or "{}")
            for r in conn.execute(
                "SELECT method, outputs_json FROM valuations WHERE ticker='LENSX'"
            ).fetchall()
        }

    assert methods["ev_ebit"]["status"] == "OK"
    assert methods["fcf_yield"]["status"] == "OK"
    assert methods["tangible_floor"]["status"] == "OK"

    pzd = methods["scorecard"].get("pricing_zone_detail") or {}
    assert isinstance(pzd.get("ev_ebit_value_per_share"), (int, float))
    assert isinstance(pzd.get("fcf_yield_value_per_share"), (int, float))
    assert isinstance(pzd.get("tangible_floor_per_share"), (int, float))


def test_select_anchor_extra_candidates_are_provenance_only():
    """Lenses appear in the candidates dict but can never win the anchor."""
    from app.valuation.anchor_policy import select_anchor

    sel = select_anchor(
        dcf=50.0,
        epv=40.0,
        extra_candidates={"ev_ebit": 500.0, "fcf_yield": 400.0, "tangible_floor": 300.0},
    )
    assert (sel.method, sel.value) == ("dcf", 50.0)
    assert sel.candidates["ev_ebit"] == 500.0
    assert sel.candidates["tangible_floor"] == 300.0
