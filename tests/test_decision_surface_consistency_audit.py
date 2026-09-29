"""Decision-surface consistency fixes from a valuation
methodology audit.

Finding families covered here:
  - graham-discount-inversion (5 sites -> one shared helper)
  - backtest-live-anchor-divergence / anchor-rule-backtest-vs-live-divergence
    (ONE shared select_anchor())
  - dual-mos-convention-same-name (naming split)
  - ncav-current-liabilities-only
  - deploy-inside-growth-dependent-zone (zone/scorecard consistency)
  - reverse-DCF saturation family
"""
from __future__ import annotations

import json

import pytest

from app.db import get_db, init_db, utc_now_iso


def _init_cfg(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache",
        lambda http=None: {},
    )
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _backtest_writer_record(method, outputs):
    from app.valuation.lineage import valuation_source_record

    record = valuation_source_record(
        {
            "ticker": "AAA",
            "as_of_date": "2023-10-04",
            "method": method,
            "inputs_json": "{}",
            "outputs_json": json.dumps(outputs, sort_keys=True),
            "warnings_json": "[]",
            "created_at": "2024-01-02T00:00:00+00:00",
            "valuation_writer_version": "test",
            "quality_gate_verdict": None,
            "confidence_class": None,
            "gate_reason_codes": None,
            "valuation_headwinds": None,
            "valuation_supports": None,
            "source_run_id": "backtest_recon_2024-01-02",
        }
    )
    assert record is not None
    return record


# ── graham-discount-inversion ─────────────────────────────────────────────────

def test_graham_inversion_roundtrip():
    """iv=100: stored textbook d=(iv-price)/iv must invert back to 100 at any
    price (audit fixture: price 60 -> d 0.4; price 90 -> d 0.1; price 150 ->
    d -0.5). The old price/(1+d) produced 42.86 / 81.82 / 300.0."""
    from app.valuation.mos_conventions import graham_value_from_textbook_discount

    for price in (60.0, 90.0, 150.0):
        d = (100.0 - price) / 100.0
        assert graham_value_from_textbook_discount(price, d) == pytest.approx(100.0)


def test_graham_inversion_guards():
    from app.valuation.mos_conventions import graham_value_from_textbook_discount

    assert graham_value_from_textbook_discount(60.0, -1.0) is None  # sentinel
    assert graham_value_from_textbook_discount(60.0, 1.0) is None   # div-by-zero
    assert graham_value_from_textbook_discount(60.0, 1.5) is None   # impossible
    assert graham_value_from_textbook_discount(None, 0.4) is None
    assert graham_value_from_textbook_discount(60.0, None) is None
    assert graham_value_from_textbook_discount(0.0, 0.4) is None


def _scorecard_with_graham_discount(price=60.0, graham_disc=0.4):
    return {
        "discounts": {"dcf": 0.2, "epv": 0.1, "graham": graham_disc},
        # price 60 < epv 70 -> MARGIN_OF_SAFETY; a zone is required since the
        # assembler only populates anchor methods for the measured zones
        # (review ANCHOR-2/GZ-2).
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {
            "dcf_base": 80.0,
            "epv_adjusted": 70.0,
            "current_price": price,
            "gate_action": "PROCEED",
            "terminal_growth_used": 0.02,
        },
        "quality_context": {"gate_action": "PROCEED"},
        "wacc_detail": {"adjusted_wacc": 0.10},
        "legacy_signal": "UNDERVALUED",
    }


def test_bundle_builder_graham_uses_correct_inversion():
    from app.analyst.bundle_builder import _build_valuation_snapshot

    snap = _build_valuation_snapshot(
        _scorecard_with_graham_discount(), {}, None, None,
    )
    assert snap.graham_value == pytest.approx(100.0)


def test_deep_research_tensions_graham_uses_correct_inversion():
    """The intrinsic range fed to method tension must top out at the correctly
    inverted Graham value (100), not the sign-inverted 42.86."""
    from app.research.deep_research import _compute_tensions_from_scorecard

    sc = _scorecard_with_graham_discount()
    tensions = _compute_tensions_from_scorecard(sc, {})
    rng = tensions.get("intrinsic_range") or {}
    assert rng.get("high") == pytest.approx(100.0)


def test_thesis_updater_graham_uses_correct_inversion():
    from app.research.thesis_updater import _extract_valuation_inputs

    inputs = _extract_valuation_inputs(_scorecard_with_graham_discount())
    assert inputs.graham == pytest.approx(100.0)


# ── shared select_anchor() ────────────────────────────────────────────────────

def test_select_anchor_max_of_positive_dcf_epv():
    """The audit's worked divergence: dcf=80, epv=100 — live first-of picked
    dcf (80); the canonical rule takes max(positive) = epv 100."""
    from app.valuation.anchor_policy import select_anchor

    sel = select_anchor(dcf=80.0, epv=100.0, graham=90.0, ncav=5.0)
    assert (sel.method, sel.value) == ("epv", 100.0)
    assert sel.reason == "MAX_POSITIVE_DCF_EPV"


def test_select_anchor_negative_dcf_does_not_block_positive_epv():
    """Audit fixture: dcf=-3, epv=60 — live anchored on -3 and the store
    dropped the entry; canonical rule falls through to epv."""
    from app.valuation.anchor_policy import select_anchor

    sel = select_anchor(dcf=-3.0, epv=60.0)
    assert (sel.method, sel.value) == ("epv", 60.0)


def test_select_anchor_graham_then_ncav_fallback():
    from app.valuation.anchor_policy import select_anchor

    sel = select_anchor(dcf=-5.0, epv=None, graham=40.0, ncav=30.0)
    assert (sel.method, sel.value) == ("graham", 40.0)
    assert sel.reason == "FALLBACK_GRAHAM"

    sel = select_anchor(dcf=None, epv=-2.0, graham=-1.0, ncav=30.0)
    assert (sel.method, sel.value) == ("ncav", 30.0)
    assert sel.reason == "FALLBACK_NCAV"


def test_select_anchor_no_positive_candidates():
    from app.valuation.anchor_policy import select_anchor

    sel = select_anchor(dcf=-5.0, epv=-2.0, graham=None, ncav=-1.0)
    assert sel.method is None and sel.value is None
    assert sel.reason == "NO_POSITIVE_ANCHOR"


def test_select_anchor_sector_specific_wins_only_when_positive():
    from app.valuation.anchor_policy import select_anchor

    sel = select_anchor(dcf=80.0, epv=100.0, sector_specific=("insurance", 120.0))
    assert (sel.method, sel.value) == ("insurance", 120.0)
    assert sel.reason == "SECTOR_SPECIFIC"

    sel = select_anchor(dcf=80.0, epv=100.0, sector_specific=("insurance", -5.0))
    assert (sel.method, sel.value) == ("epv", 100.0)


def test_packet_anchor_uses_canonical_rule():
    """sector_financial_packets must route through select_anchor: max of
    positive (dcf, epv), not first-numeric."""
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.sector_financial_packets import _generic_valuation_anchor

    packet = TickerSignalPacket(ticker="DIVRG")
    packet.dcf_value = 20.0
    packet.epv_value = 80.0
    packet.graham_value = 70.0
    packet.ncav_value = 10.0
    assert _generic_valuation_anchor(packet) == ("epv", 80.0)

    packet.dcf_value = -3.0
    packet.epv_value = 60.0
    assert _generic_valuation_anchor(packet) == ("epv", 60.0)


def test_reconstruct_anchor_prefers_higher_epv(monkeypatch, tmp_path):
    """Backtest and live now share the rule: with epv_adjusted > dcf_base the
    anchor is the EPV leg (already max-based pre-fix; pins the shared path)."""
    _init_cfg(monkeypatch, tmp_path)
    from app.backtest import reconstruct

    records = [
        _backtest_writer_record(
            "scorecard",
            {
                "pricing_zone": "MARGIN_OF_SAFETY",
                "pricing_zone_detail": {"dcf_base": 80.0, "epv_adjusted": 100.0},
            },
        ),
        _backtest_writer_record(
            "reverse_dcf",
            {"expectations_gap": {"bucket": "CHEAP_VS_EXPECTATIONS"}},
        ),
    ]
    ensure_call = {}

    def _fake_ensure(*args, **kwargs):
        ensure_call["args"] = args
        ensure_call["kwargs"] = kwargs
        return records

    monkeypatch.setattr(reconstruct, "ensure_valuation", _fake_ensure)
    monkeypatch.setattr(reconstruct, "_live_sector_model_routed", lambda *args: False)
    monkeypatch.setattr(reconstruct, "cap_category_asof", lambda *a: "small_cap")

    def _forbid_db_read():
        raise AssertionError("reconstruction must not reread mutable valuation rows")

    monkeypatch.setattr("app.db.get_db", _forbid_db_read)

    class _P:
        def get_price_asof(self, t, d):
            from dataclasses import dataclass

            @dataclass
            class S:
                ticker: str
                as_of_date: str
                price: float
                raw_price: float

            return S(t, d, 50.0, 50.0)

    provider = _P()
    result = reconstruct.reconstruct_signal_asof("AAA", "2024-01-02", provider=provider)
    assert result.signal is not None
    assert result.signal.anchor == 100.0
    assert result.signal.buy_price_target == 75.0
    assert result.signal.expectations_gap_bucket == "CHEAP_VS_EXPECTATIONS"
    assert ensure_call["args"] == ("AAA", "2023-10-04")
    assert ensure_call["kwargs"] == {
        "provider": provider,
        "run_id": "backtest_recon_2024-01-02",
        "price_override": 50.0,
        "force_refresh": True,
        "require_filed_asof": True,
        "raise_on_error": True,
    }


# ── deploy-inside-growth-dependent-zone (zone/scorecard consistency) ──────────

def test_zone_buy_label_requires_minimum_textbook_mos():
    """A 0.01% margin must not render as BUY: the MARGIN_OF_SAFETY zone maps
    to BUY only at >=25% textbook MoS (the deploy discount), else HOLD
    (audit: deploy-inside-growth-dependent-zone)."""
    from app.valuation.valuation_writer import _margin_of_safety_scorecard

    methods = {
        "dcf": {"status": "OK", "base": 120.0},
        "epv": {"status": "OK", "value_per_share": 100.0},
        "graham": {"status": "OK", "value_per_share": 90.0},
        "ncav": {"status": "OK", "value_per_share": -5.0, "signal": "NCAV_NO_ASSET_FLOOR"},
    }
    thin = _margin_of_safety_scorecard(methods, price=99.99, shares=10.0, net_debt=5.0)
    assert thin["pricing_zone"] == "MARGIN_OF_SAFETY"
    assert thin["signal"] == "HOLD"

    deep = _margin_of_safety_scorecard(methods, price=60.0, shares=10.0, net_debt=5.0)
    assert deep["pricing_zone"] == "MARGIN_OF_SAFETY"
    assert deep["signal"] == "BUY"


def test_variant_builder_overvalued_reads_legacy_signal():
    """scorecard['signal'] is the zone action (BUY/HOLD/PASS...); OVERVALUED
    lives in legacy_signal — the old check could never fire, silently
    dropping the SCORECARD_OVERVALUED evidence path."""
    from app.synthesis.variant_builder import _extract_valuation_signals

    valuations = {
        "scorecard": {
            "outputs": {
                "signal": "PASS",
                "legacy_signal": "OVERVALUED",
                "discounts": {},
                "track_comparison": {},
            }
        }
    }
    signals, _, _ = _extract_valuation_signals(ticker="OVRV", valuations=valuations)
    assert any(s.signal_type == "SCORECARD_OVERVALUED" for s in signals)


# ── reverse-DCF saturation family ─────────────────────────────────────────────

def test_reverse_dcf_negative_margin_interior_root():
    """Audit fixture (bisection-monotonicity-negative-margin): a cash-rich
    money-loser with a well-defined root at g=0 — the old increasing-only
    update slid to +0.60 saturated, wrong by 60pp."""
    from app.valuation.reverse_dcf import implied_growth_from_price

    outputs, _ = implied_growth_from_price(
        market_price=2.2926,
        shares_outstanding=100.0,
        net_debt=-800.0,
        base_revenue=1000.0,
        margin=-0.10,
    )
    assert outputs["implied_growth_saturated"] is False
    assert abs(outputs["implied_growth"]) < 1e-3


def test_reverse_dcf_saturated_low_reports_direction():
    """Deep-cheap (EV/Rev 0.40 at 30% margin): saturates LOW — unambiguously
    max-cheap, must be distinguishable from the unsolvable HIGH tail
    (audit: sat-conflates-deep-cheap-with-unsolvable)."""
    from app.valuation.reverse_dcf import implied_growth_from_price

    outputs, _ = implied_growth_from_price(
        market_price=4.0,
        shares_outstanding=100.0,
        net_debt=0.0,
        base_revenue=1000.0,
        margin=0.30,
    )
    assert outputs["implied_growth_saturated"] is True
    assert outputs["implied_growth_saturated_bound"] == "LOW"
    assert outputs["margin_sign"] == "POSITIVE"
    assert outputs["implied_growth"] == -0.25


def test_reverse_dcf_money_loser_saturates_high():
    from app.valuation.reverse_dcf import implied_growth_from_price

    outputs, _ = implied_growth_from_price(
        market_price=1.0,
        shares_outstanding=100.0,
        net_debt=0.0,
        base_revenue=1000.0,
        margin=-0.10,
    )
    assert outputs["implied_growth_saturated"] is True
    assert outputs["implied_growth_saturated_bound"] == "LOW"
    assert outputs["margin_sign"] == "NEGATIVE"
    # For a money-loser dcf_ev is DECREASING in growth, so the maximum EV sits
    # at the LOW growth endpoint and the returned value is that endpoint. The
    # label names the endpoint the value actually sits at. This test formerly
    # asserted "HIGH" alongside -0.25 and documented the decoupling as
    # intentional; that contradiction was filed as a defect, so the label now follows the value. The
    # price-side meaning is still recoverable from (bound, margin_sign).
    assert outputs["implied_growth"] == -0.25


def test_supportable_growth_negative_history_not_floored():
    """Audit fixture (supportable-growth-clamp-sign-bias): a -10%/yr decliner
    priced for -8% is paying for MORE growth than history supports — the 0.0
    floor flipped it to CHEAP_VS_EXPECTATIONS."""
    from app.valuation.expectations_gap import (
        compute_expectations_gap,
        estimate_supportable_growth,
    )

    supportable, basis = estimate_supportable_growth(-0.10, None, [])
    assert supportable == -0.10
    gap = compute_expectations_gap(-0.08, supportable, False)
    assert gap["gap"] == pytest.approx(0.02)
    assert gap["bucket"] == "FAIRLY_PRICED_EXPECTATIONS"


def test_supportable_growth_haircut_only_on_positive():
    """Haircutting a NEGATIVE supportable by 0.5 would RAISE it."""
    from app.valuation.expectations_gap import estimate_supportable_growth

    supportable, _ = estimate_supportable_growth(
        -0.10, None, ["EARNINGS_QUALITY_HEADWIND"]
    )
    assert supportable == -0.10

    supportable, basis = estimate_supportable_growth(
        0.10, None, ["EARNINGS_QUALITY_HEADWIND"]
    )
    assert supportable == pytest.approx(0.05)
    assert basis == "CAGR_QUALITY_HAIRCUT"


def test_supportable_basis_label_reflects_owner_earnings_fallback():
    from app.valuation.expectations_gap import estimate_supportable_growth

    _, basis = estimate_supportable_growth(None, 0.08, [])
    assert basis == "OWNER_EARNINGS_CAGR_5Y"


def test_expectations_gap_saturated_low_positive_margin_is_cheap_floor():
    """Saturated-LOW + positive margin + known supportable = the deepest-cheap
    cohort (the pilot's discarded +38pp bucket) — emit CHEAP with the gap as
    an upper bound (true gap <= -0.35) instead of discarding. It was labelled a
    floor, which states the opposite inequality."""
    from app.valuation.expectations_gap import compute_expectations_gap

    gap = compute_expectations_gap(
        -0.25, 0.10, True, saturated_bound="LOW", margin_sign="POSITIVE",
    )
    assert gap["bucket"] == "CHEAP_VS_EXPECTATIONS"
    assert gap["gap"] == pytest.approx(-0.35)
    assert gap.get("gap_is_upper_bound") is True
    assert "gap_is_floor" not in gap


def test_expectations_gap_saturated_high_or_negative_margin_unreliable():
    from app.valuation.expectations_gap import compute_expectations_gap

    high = compute_expectations_gap(
        0.60, 0.10, True, saturated_bound="HIGH", margin_sign="POSITIVE",
    )
    assert high["bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"

    neg = compute_expectations_gap(
        -0.25, 0.10, True, saturated_bound="LOW", margin_sign="NEGATIVE",
    )
    assert neg["bucket"] == "EXPECTATIONS_GAP_UNRELIABLE"


def test_writer_feasibility_unknown_when_saturated(monkeypatch):
    """Saturated solves must not be graded REASONABLE/IMPLAUSIBLE off a
    clipped bound (audit: saturated-bound-leaks-to-flag-ignoring-consumers)."""
    import app.valuation.valuation_writer as vw

    monkeypatch.setattr(
        vw, "implied_growth_from_price",
        lambda **kwargs: (
            {"implied_growth": 0.60, "implied_growth_saturated": True,
             "implied_growth_saturated_bound": "HIGH", "margin_sign": "POSITIVE",
             "feasibility_gap_score": None, "target_ev": 100.0},
            ["Implied growth solve saturated at bound"],
        ),
    )
    result = vw._run_reverse_dcf(
        10.0, 100.0, 0.0, 1000.0, 100.0, revenue_cagr_5y=0.04,
    )
    assert result["feasibility"] == "UNKNOWN"


def test_enrichment_nulls_saturated_implied_growth():
    from app.evidence.enrichment import _valuation_spread_analysis

    packet = {
        "valuations": {
            "reverse_dcf": {
                "outputs": {
                    "outputs": {
                        "implied_growth": -0.25,
                        "implied_growth_saturated": True,
                    },
                },
            },
        },
    }
    ctx = _valuation_spread_analysis(packet)
    assert ctx.get("implied_growth_rate") is None
    assert ctx.get("implied_growth_feasibility") is None


# ── ncav-current-liabilities-only ─────────────────────────────────────────────

def test_ncav_subtracts_total_liabilities_not_current_only():
    """Textbook Graham NCAV = current assets - TOTAL liabilities. The audit
    fixture: a levered company (CL 20, TL 100) trading at 3 with ZERO real
    NCAV was labeled NCAV_NET_NET because only current liabilities were
    subtracted (audit: ncav-current-liabilities-only)."""
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        50.0, 200.0, 100.0, 60.0, 10.0, 3.0,
        accounts_receivable=20.0,
        inventory=30.0,
        current_assets=100.0,
        current_liabilities=20.0,
    )
    # liquid assets = 50 + 0.75*20 + 0.50*30 = 80; minus TOTAL liabilities 100
    assert result["value_per_share"] == pytest.approx(-2.0)
    assert result["signal"] == "NCAV_NO_ASSET_FLOOR"


def test_ncav_current_liabilities_only_is_flagged_proxy():
    """When total liabilities are missing, current liabilities are an
    UNDERSTATING proxy and must be flagged as such."""
    from app.valuation.valuation_writer import _ncav

    result = _ncav(
        50.0, 200.0, None, 60.0, 10.0, 3.0,
        accounts_receivable=20.0,
        inventory=30.0,
        current_assets=100.0,
        current_liabilities=20.0,
    )
    assert result["value_per_share"] == pytest.approx(6.0)
    assert "LIABILITIES_CURRENT_ONLY_PROXY" in result["flags"]


def test_signal_assembler_graham_uses_correct_inversion(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)
               VALUES (?, ?, 'scorecard', '{}', ?, '[]', ?)""",
            ("GRHM", "2024-09-28", json.dumps(_scorecard_with_graham_discount()), now),
        )
        conn.commit()

    from app.alpha.signal_assembler import assemble_signal_packet

    packet = assemble_signal_packet("GRHM", filing_risk_use_llm=False)
    assert packet.graham_value == pytest.approx(100.0)
