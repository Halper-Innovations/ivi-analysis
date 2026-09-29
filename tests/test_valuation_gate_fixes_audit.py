"""Gate fixes from a valuation methodology audit.

  - zero-oe-gate-full-capex-vs-maintenance (ZERO_OWNER_EARNINGS aligned to
    the canonical maintenance-capex convention)
  - secular-decline-block-single-spike-peak (spike-aware SEVERE_SECULAR_DECLINE)
  - liquidity-crisis-cross-year-ratio (common-year LIQUIDITY_CRISIS)
"""
from __future__ import annotations

from unittest.mock import patch


def _facts(**overrides):
    base = {
        "revenue": [(2025, 100.0), (2024, 98.0), (2023, 95.0), (2022, 92.0), (2021, 90.0)],
        "cfo": [(2025, 20.0), (2024, 19.0), (2023, 18.0), (2022, 17.0), (2021, 16.0)],
        "operating_income": [(2025, 15.0), (2024, 14.0), (2023, 13.0), (2022, 12.0), (2021, 11.0)],
        "net_income": [(2025, 10.0), (2024, 9.5), (2023, 9.0), (2022, 8.5), (2021, 8.0)],
        "total_debt": [(2025, 30.0)],
        "cash": [(2025, 15.0)],
        "equity": [(2025, 50.0)],
        "shares_outstanding": [(2025, 10.0)],
    }
    base.update(overrides)
    return base


def _ctx(facts, category="TRADITIONAL_OPERATING"):
    from app.valuation.pre_valuation_gate import compute_quality_context

    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        return compute_quality_context("TEST", "2026-03-25", facts=facts, category=category)


# ── zero-oe-gate-full-capex-vs-maintenance ────────────────────────────────────

def test_zero_oe_gate_uses_maintenance_capex_for_growth_software():
    """Audit fixture: CFO 100/yr, capex 120/110/105, SBC 5 — the canonical
    owner earnings at ENTERPRISE_SOFTWARE ratio 0.25 are strongly positive
    (+67), so the gate must NOT block under a 'zero owner earnings' label
    its own valuation stack contradicts."""
    facts = _facts(
        # growing >8%/yr so the growth-aware ratio keeps the category 0.25
        revenue=[(2025, 150.0), (2024, 130.0), (2023, 110.0), (2022, 95.0), (2021, 80.0)],
        cfo=[(2025, 100.0), (2024, 100.0), (2023, 100.0)],
        capex=[(2025, 120.0), (2024, 110.0), (2023, 105.0)],
        sbc=[(2025, 5.0), (2024, 5.0), (2023, 5.0)],
    )
    ctx = _ctx(facts, category="ENTERPRISE_SOFTWARE")
    assert "ZERO_OWNER_EARNINGS" not in (ctx.get("gate_reason_codes") or [])
    assert ctx["gate_action"] != "BLOCK"


def test_zero_oe_gate_still_blocks_flat_revenue_cash_burner():
    """A flat-revenue firm gets maintenance ratio 1.0 (its capex IS
    maintenance): CFO persistently below full capex + SBC stays blocked."""
    facts = _facts(
        revenue=[(2025, 100.0), (2024, 100.0), (2023, 100.0), (2022, 100.0), (2021, 100.0)],
        cfo=[(2025, 100.0), (2024, 100.0), (2023, 100.0)],
        capex=[(2025, 120.0), (2024, 110.0), (2023, 105.0)],
        sbc=[(2025, 5.0), (2024, 5.0), (2023, 5.0)],
    )
    ctx = _ctx(facts, category="TRADITIONAL_OPERATING")
    assert ctx["gate_action"] == "BLOCK"
    assert "ZERO_OWNER_EARNINGS" in (ctx.get("gate_reason_codes") or [])


def test_zero_oe_gate_one_bad_year_insufficient():
    facts = _facts(
        cfo=[(2025, -10.0), (2024, 50.0), (2023, 50.0)],
        capex=[(2025, 5.0), (2024, 5.0), (2023, 5.0)],
    )
    ctx = _ctx(facts)
    assert "ZERO_OWNER_EARNINGS" not in (ctx.get("gate_reason_codes") or [])


# ── secular-decline-block-single-spike-peak ───────────────────────────────────

def test_secular_decline_spike_peak_does_not_block():
    """Audit fixture: 50,55,60,140(spike),65,70 — a steadily growing company
    must not be blocked for 'secular decline' measured off a one-year
    licensing spike."""
    facts = _facts(
        revenue=[(2024, 70.0), (2023, 65.0), (2022, 140.0), (2021, 60.0), (2020, 55.0), (2019, 50.0)],
    )
    ctx = _ctx(facts)
    assert "SEVERE_SECULAR_DECLINE" not in (ctx.get("gate_reason_codes") or [])
    assert ctx["gate_action"] != "BLOCK"


def test_secular_decline_genuine_still_blocks():
    """A genuine >30% multi-year decline still blocks."""
    facts = _facts(
        revenue=[(2024, 55.0), (2023, 70.0), (2022, 85.0), (2021, 100.0), (2020, 95.0)],
    )
    ctx = _ctx(facts)
    assert "SEVERE_SECULAR_DECLINE" in (ctx.get("gate_reason_codes") or [])
    assert ctx["gate_action"] == "BLOCK"


def test_secular_decline_spike_at_window_edge_does_not_block():
    """Review SPIKE-EDGE-WINDOW-REBLOCK: three years later the same spike
    slides to the first slot of the 6-year window (140,65,70,75,80,85) and
    loses one visible neighbor — the two-neighbor test stopped detecting it
    and the same growing company re-blocked for one year. A one-sided 2.0x
    test against the single visible neighbor must keep it unblocked."""
    facts = _facts(
        revenue=[(2027, 85.0), (2026, 80.0), (2025, 75.0), (2024, 70.0), (2023, 65.0), (2022, 140.0)],
    )
    ctx = _ctx(facts)
    assert "SEVERE_SECULAR_DECLINE" not in (ctx.get("gate_reason_codes") or [])
    assert ctx["gate_action"] != "BLOCK"


def test_secular_decline_genuine_peak_at_window_edge_still_blocks():
    """A genuine decline whose peak sits at the window edge is NOT a spike
    (peak < 2x its neighbor) and must still block."""
    facts = _facts(
        revenue=[(2024, 40.0), (2023, 55.0), (2022, 70.0), (2021, 85.0), (2020, 100.0)],
    )
    ctx = _ctx(facts)
    assert "SEVERE_SECULAR_DECLINE" in (ctx.get("gate_reason_codes") or [])
    assert ctx["gate_action"] == "BLOCK"


# ── liquidity-crisis-cross-year-ratio ─────────────────────────────────────────

def test_liquidity_crisis_uses_common_year():
    """Audit fixture: CA only through FY2022 (400), CL through FY2024 (900)
    and FY2022 (380). Mixing years gave 400/900 = 0.44 -> spurious BLOCK;
    the aligned year 2022 gives 400/380 = 1.05 — healthy."""
    facts = _facts(
        current_assets=[(2022, 400.0)],
        current_liabilities=[(2024, 900.0), (2022, 380.0)],
    )
    ctx = _ctx(facts)
    assert "LIQUIDITY_CRISIS" not in (ctx.get("gate_reason_codes") or [])


def test_liquidity_crisis_fires_on_aligned_distress():
    facts = _facts(
        current_assets=[(2025, 100.0)],
        current_liabilities=[(2025, 300.0)],
    )
    ctx = _ctx(facts)
    assert "LIQUIDITY_CRISIS" in (ctx.get("gate_reason_codes") or [])
    assert ctx["gate_action"] == "BLOCK"


def test_liquidity_crisis_skipped_when_no_common_year():
    facts = _facts(
        current_assets=[(2023, 100.0)],
        current_liabilities=[(2024, 900.0)],
    )
    ctx = _ctx(facts)
    assert "LIQUIDITY_CRISIS" not in (ctx.get("gate_reason_codes") or [])


# ── verified-LOW batch (cheap fixes in already-touched files) ─────────────────

def test_dcf_negative_oe_scenarios_keep_low_le_base_le_high():
    """dcf-negative-oe-scenario-inversion: for negative OE the published
    low/base/high were inverted (low was the HIGHEST value)."""
    from app.valuation.valuation_writer import _discounted_owner_earnings

    rev = [(2024, 1100.0), (2023, 1050.0), (2022, 1000.0)]
    result = _discounted_owner_earnings(-100.0, 10.0, 0.0, rev)
    assert result["status"] == "OK"
    assert result["low"] <= result["base"] <= result["high"]


def test_dcf_invalid_terminal_assumptions_fail_loudly():
    """dcf-terminal-guard-fabricates-value: wacc <= terminal growth must not
    silently fabricate a ~1000x terminal multiple."""
    from app.valuation.valuation_writer import _discounted_owner_earnings

    rev = [(2024, 1100.0), (2023, 1050.0), (2022, 1000.0)]
    result = _discounted_owner_earnings(100.0, 10.0, 0.0, rev, wacc=0.02, terminal_growth=0.02)
    assert result["status"] == "METHOD_INSUFFICIENT_DATA"
    assert "INVALID_DISCOUNT_ASSUMPTIONS" in result["flags"]
    assert result["base"] is None


def test_epv_no_tax_shield_on_losses():
    """negative-epv-tax-shield-on-losses: taxing a loss at 21% credited an
    immediate full tax shield, shrinking negative EPVs."""
    from app.valuation.valuation_writer import _epv

    oi_series = [(2025, -50.0), (2024, -50.0), (2023, -50.0)]
    # Flat revenue: the normalized margin on current revenue is the same -50.0
    # the levels average gave, so this case keeps its own arithmetic.
    revenue_series = [(year, 1000.0) for year, _ in oi_series]
    result = _epv(
        oi_series, net_debt=-100.0, shares=10.0, revenue_series=revenue_series, wacc=0.10
    )
    # EV = -50/0.10 = -500; equity = -500 + 100 = -400 -> -40.0/share
    # (NOT -50*0.79/0.10 = -395 -> -29.5). Since 2026-09-29 a result with no
    # earnings power publishes no value; the negative reading carries it.
    assert result["value_per_share"] is None
    assert result["status"] == "EPV_NEGATIVE"
    assert result["negative_value_per_share"] == -40.0
    assert "EPV_NO_TAX_SHIELD_ON_LOSSES" in result["flags"]


def test_ic_weak_audit_trail_describes_band():
    """wacc-ic-weak-audit-trail-band: the rule fires on 1.5 <= IC < 3.0 but
    recorded threshold=3.0/'<' — the audit trail misdescribed it for IC<1.5."""
    from app.valuation.valuation_writer import _compute_quality_wacc

    facts = {
        "operating_income": [(2024, 10.0)],
        "interest_expense": [(2024, 10.0)],  # IC = 1.0 -> CRITICAL only
        "revenue": [(2024, 100.0), (2023, 95.0)],
    }
    detail = _compute_quality_wacc(facts)
    weak = next(r for r in detail["rule_evaluations"] if r["code"] == "INTEREST_COVERAGE_WEAK")
    assert weak["fired"] is False
    assert weak["comparison"] == "1.5 <= ic < 3.0"
    assert weak["threshold"] == {"low": 1.5, "high": 3.0}


def test_shares_sub_million_counts_not_rescued_to_thousands():
    """share-scale-1000x-sub-million-counts: a genuine 820k-share count
    (post heavy reverse split) was stored as 820 shares_millions."""
    from app.ingest.companyfacts import _to_millions

    assert _to_millions(820_000.0, "shares_outstanding") == 0.82
    assert _to_millions(800_000.0, "shares_outstanding") == 0.8


def test_cash_tag_priority_puts_restricted_inclusive_last():
    """cash-tag-order-restricted-cash-divergence: ingest ranked the
    restricted-cash-inclusive tag above plain cash tags; the as-of extractor
    ranks it last — the two net-debt surfaces could resolve different cash."""
    from app.ingest.companyfacts import TAG_MAP

    cash_tags = TAG_MAP["cash"]
    assert cash_tags.index("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents") > cash_tags.index("CashAndCashEquivalents")
    assert cash_tags.index("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents") > cash_tags.index("CashEquivalentsAtCarryingValue")


def test_dei_shares_use_fiscal_year_metadata_not_cover_date_year():
    """dei-shares-fiscal-year-label-shift: dei cover-page instants are dated
    months after FY end, so a Dec-FYE company's share count was labeled
    FY+1, misaligning the shares series with the statement years."""
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = {"facts": {"dei": {
        "EntityCommonStockSharesOutstanding": {
            "units": {"shares": [
                {"end": "2025-02-15", "val": 50_000_000.0, "form": "10-K", "filed": "2025-02-20", "fy": 2024, "fp": "FY", "accn": "a-1"},
            ]}
        },
    }}}
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    rows = [f for f in facts if f["line_item"] == "shares_outstanding"]
    assert len(rows) == 1
    assert rows[0]["fiscal_year"] == 2024
    assert rows[0]["value"] == 50.0


# ── coverage gaps ─────────────────────────────────────────────────────────────

def test_ev_bridge_deducts_preferred_and_nci(monkeypatch, tmp_path):
    """Coverage gap 2: preferred stock and noncontrolling interest are senior
    to common — the EV->equity bridge must deduct them like debt."""
    import json as _json

    from app.db import get_db, init_db, utc_now_iso

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
        "preferred_equity": [50.0, 50.0, 50.0],
        "noncontrolling_interest": [30.0, 30.0, 30.0],
    }
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('SENIOR', '0000000456', 'Senior Claims Test Co', ?)
            """,
            (utc_now_iso(),),
        )
        for line_item, values in fields.items():
            for year, value in zip([2021, 2022, 2023], values, strict=True):
                conn.execute(
                    "INSERT INTO companyfacts_facts "
                    "(ticker, fiscal_year, period_type, period_end, line_item, value, units, "
                    "source_url, fetched_at, filed_date, form, accession) "
                    "VALUES ('SENIOR', ?, 'FY', ?, ?, ?, ?, "
                    "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000456.json', "
                    "?, ?, '10-K', ?)",
                    (
                        year,
                        f"{year}-12-31",
                        line_item,
                        float(value),
                        (
                            "shares_millions"
                            if line_item == "shares_outstanding"
                            else "USD_millions"
                        ),
                        utc_now_iso(),
                        f"{year + 1}-02-15",
                        f"0000000456-{str(year + 1)[-2:]}-000001",
                    ),
                )
        conn.commit()

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("SENIOR", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        row = conn.execute(
            "SELECT inputs_json, outputs_json FROM valuations WHERE ticker='SENIOR' AND method='scorecard'"
        ).fetchone()
    inputs = _json.loads(row["inputs_json"])
    outputs = _json.loads(row["outputs_json"])
    # debt 200 - cash 100 + preferred 50 + NCI 30 = 180
    assert inputs["net_debt"] == 180.0
    flags = (outputs.get("quality_context") or {}).get("net_debt_flags") or []
    assert "SENIOR_CLAIMS_DEDUCTED" in flags


def test_ingest_has_preferred_and_nci_tags():
    from app.ingest.companyfacts import PLAUSIBILITY, TAG_MAP

    assert "PreferredStockValue" in TAG_MAP["preferred_equity"]
    assert "MinorityInterest" in TAG_MAP["noncontrolling_interest"]
    assert "preferred_equity" in PLAUSIBILITY
    assert "noncontrolling_interest" in PLAUSIBILITY


def test_split_basis_profile_flags_capital_event(monkeypatch, tmp_path):
    """Coverage gap 3: adjusted backtest prices vs as-of share counts diverge
    by the split factor for names with splits after T — the profile helper
    quantifies the affected names for the diagnostic."""
    from app.backtest.split_basis import share_count_basis_profile
    from app.db import get_db, init_db, utc_now_iso

    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())
    with get_db() as conn:
        for year, val, pe in [(2021, 100.0, "2021-12-31"), (2023, 100.0, "2023-12-31"), (2025, 10.0, "2025-12-31")]:
            conn.execute(
                "INSERT INTO companyfacts_facts "
                "(ticker, fiscal_year, period_type, period_end, line_item, value, units, fetched_at) "
                "VALUES ('SPLITX', ?, 'FY', ?, 'shares_outstanding', ?, 'shares_millions', ?)",
                (year, pe, val, utc_now_iso()),
            )
        conn.commit()

    profile = share_count_basis_profile("SPLITX", "2024-01-01")
    # as-of count 100M, latest count 10M -> 10x reverse split after T
    assert profile["shares_asof"] == 100.0
    assert profile["shares_latest"] == 10.0
    assert profile["classification"] == "CAPITAL_EVENT_SUSPECT"

    stable = share_count_basis_profile("NOPE", "2024-01-01")
    assert stable["classification"] == "NO_SHARES_DATA"


def test_resolver_counts_dividend_basis_mismatches(monkeypatch, tmp_path):
    """Coverage gap 4: a live-quote entry price against an adjusted-series
    exit mixes bases; the resolver quantifies (not alters) the divergence."""
    from app.calibration.return_resolver import resolve_open_outcomes
    from app.db import init_db
    from app.outcomes.store import add_outcome

    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    init_db(get_config())

    add_outcome(
        ticker="DIVX",
        as_of_date="2024-01-02",
        run_id="live_test",
        decision="BUY",
        conviction=2,
        horizon_days=30,
        entry_price=100.0,  # live quote (raw basis)
        entry_price_source="watchlist_population",
        entry_date="2024-01-02",
        grade="ACTIONABLE",
        benchmark_symbol=None,
    )

    class _P:
        def get_price_asof(self, ticker, as_of_date):
            from dataclasses import dataclass

            @dataclass
            class S:
                ticker: str
                as_of_date: str
                price: float

            # Adjusted series: 90 at entry (>2% below the raw 100), 95 at exit
            return S(ticker, as_of_date, 90.0 if as_of_date == "2024-01-02" else 95.0)

    summary = resolve_open_outcomes("2024-06-01", provider=_P())
    assert summary.closed == 1
    assert summary.dividend_basis_mismatches == 1


def test_insurance_model_blocks_invalid_coe_growth(monkeypatch):
    """Coverage gap 1: env-overridable cost-of-equity/growth could drive the
    perpetual-spread denominator to a 1000x multiplier."""
    monkeypatch.setenv("VOE_INSURANCE_LONG_RUN_GROWTH", "0.20")
    from app.insurance.valuation import calculate_insurance_common_valuation

    result = calculate_insurance_common_valuation(
        "NOINS",
        as_of_date="2024-01-01",
        routing={"security_type": "COMMON", "issuer_type": "INSURANCE_UNDERWRITER"},
    )
    # With growth >= cost of equity the model must refuse, not extrapolate.
    if result.get("model_status") == "OK":
        raise AssertionError("model must not produce an anchor with growth >= cost of equity")
    assert "INVALID_COE_GROWTH_ASSUMPTIONS" in (result.get("reason_codes") or []) or result.get(
        "model_status"
    ) in ("MODEL_BLOCKED", "NOT_APPLICABLE")


def test_insurance_residual_value_capped_to_book_multiple():
    from app.insurance.valuation import _residual_value_guarded

    # spread 0.30 clamped to 0.20; denominator 0.05 -> 1 + 0.20/0.05 = 5x
    # book, capped at 4x.
    value, guards = _residual_value_guarded(10.0, 0.40, 0.10, 0.05)
    assert value == 40.0
    assert "ROE_SPREAD_CLAMPED" in guards
    assert "RESIDUAL_CAPPED_AT_BOOK_MULTIPLE" in guards

    # Deep value destruction floors at zero, never a negative anchor.
    value, guards = _residual_value_guarded(10.0, -0.40, 0.10, 0.05)
    assert value == 0.0
    assert "RESIDUAL_FLOORED_AT_ZERO" in guards


def test_rnd_incomplete_vintage_years_not_merged():
    """rnd-vintage-boundary-overstates-early-years: years whose vintage stack
    predates the available history get no amortization charge, overstating
    the adjustment — they are flagged and excluded from the writer merge."""
    from app.valuation.rnd_capitalization import compute_rnd_adjusted_earnings
    from app.valuation.tech_category import ENTERPRISE_SOFTWARE

    payload = {
        "entityName": "T",
        "facts": {"us-gaap": {
            tag: {"units": {"USD": [
                {"end": f"{y}-12-31", "filed": f"{y + 1}-02-01", "val": v}
                for y, v in rows
            ]}}
            for tag, rows in {
                "RevenueFromContractWithCustomerExcludingAssessedTax": [(2021, 900e6), (2022, 1000e6), (2023, 1100e6), (2024, 1200e6)],
                "GrossProfit": [(2021, 630e6), (2022, 700e6), (2023, 770e6), (2024, 840e6)],
                "ResearchAndDevelopmentExpense": [(2021, 100e6), (2022, 120e6), (2023, 150e6), (2024, 180e6)],
                "OperatingIncomeLoss": [(2021, 180e6), (2022, 190e6), (2023, 195e6), (2024, 200e6)],
                "NetCashProvidedByUsedInOperatingActivities": [(2021, 210e6), (2022, 220e6), (2023, 230e6), (2024, 240e6)],
                "PaymentsToAcquirePropertyPlantAndEquipment": [(2021, 30e6), (2022, 32e6), (2023, 34e6), (2024, 36e6)],
            }.items()
        }},
    }
    result = compute_rnd_adjusted_earnings(
        "VINT", "2025-03-01", category=ENTERPRISE_SOFTWARE, companyfacts=payload,
    )
    assert result["status"] == "OK"
    by_year = {row["year"]: row for row in result["time_series"]}
    # life=3, history starts 2021: only 2024 has a complete vintage stack.
    assert by_year[2024]["vintage_complete"] is True
    assert by_year[2022]["vintage_complete"] is False
    assert by_year[2023]["vintage_complete"] is False
