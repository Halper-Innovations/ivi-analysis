"""Independent-review fixes in app/valuation (2026-09-29).

Each test pins the corrected answer with exact literals worked by hand in its
docstring; every one of them failed on the code before its fix.
"""

from __future__ import annotations


def _series(values, latest=2026):
    """(fiscal_year, value) pairs, latest year first."""
    return [(latest - i, v) for i, v in enumerate(values)]


def test_strong_grower_latest_cfo_is_paired_with_latest_uncapped_capex():
    """A >8% grower keeps its latest CFO unsmoothed (the strong-grower exemption).
    Its capex must be the same year's, uncapped. MSFT-shaped figures ($M):
    CFO 182,935 (5-yr median 118,548 -> a 'spike', exempt); capex 5-yr mean
    55,393.8, so 115,948 is capped to the mean and the normalized capex was
    43,282.96, giving owner earnings 139,652.04 against FCF 66,987. Paired with
    the latest capex: 182,935 - 115,948 = 66,987."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = {
        "cfo": _series([182935.0, 136162.0, 118548.0, 87582.0, 89035.0]),
        "capex": _series([115948.0, 64551.0, 44477.0, 28107.0, 23886.0]),
        "sbc": _series([0.0, 0.0, 0.0, 0.0, 0.0]),
        "revenue": _series([331839.0, 281724.0, 245122.0, 211915.0, 198270.0]),
    }
    result = _compute_owner_earnings(facts)
    assert result["cfo_used"] == 182935.0
    assert result["normalized_capex"] == 115948.0
    assert result["owner_earnings_latest"] == 66987.0
    assert "CAPEX_LATEST_PAIRED_WITH_GROWER_CFO" in result["flags"]
    assert result["cfo_normalization"] == "GROWER_LATEST_KEPT"


def test_non_grower_spike_keeps_both_sides_normalized():
    """No exemption: CFO smoothed to its median 100, capex normalized (flat 40)."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = {
        "cfo": _series([150.0, 100.0, 95.0, 105.0, 100.0]),
        "capex": _series([80.0, 40.0, 40.0, 40.0, 40.0]),
        "sbc": _series([0.0] * 5),
        "revenue": _series([1000.0] * 5),
    }
    result = _compute_owner_earnings(facts)
    # capex mean 48 -> no year above 96 -> normalized 48; 100 - 48 = 52
    assert result["owner_earnings_latest"] == 52.0
    assert "CAPEX_LATEST_PAIRED_WITH_GROWER_CFO" not in result["flags"]


def test_depreciation_only_da_is_not_reduced_by_intangible_amortization_again():
    """A filer whose D&A row came from the ``Depreciation`` concept (MSFT FY2024:
    Depreciation 15,200, AmortizationOfIntangibleAssets 4,800) already excludes
    intangible amortization. Physical D&A is (15,200 + 11,000 + 12,600) / 3 =
    12,933.33, not (10,400 + 8,500 + 10,600) / 3 = 9,833.33."""
    import pytest

    from app.valuation.valuation_writer import _epv_da_and_maintenance_capex

    facts = {
        "depreciation_amortization": [(2024, 15200.0), (2023, 11000.0), (2022, 12600.0)],
        "intangible_amortization": [(2024, 4800.0), (2023, 2500.0), (2022, 2000.0)],
        "capex": [(2024, 44477.0), (2023, 28107.0), (2022, 23886.0)],
    }
    concepts = {2024: "Depreciation", 2023: "Depreciation", 2022: "Depreciation"}
    out = _epv_da_and_maintenance_capex(facts, 1.0, da_source_concepts=concepts)
    assert out["depreciation_amortization"] == pytest.approx(38800.0 / 3.0, abs=1e-9)
    assert out["intangible_amortization_excluded"] is False
    assert out["depreciation_only_years"] == [2022, 2023, 2024]
    # A combined D&A concept (or an unknown one) still has amortization taken out.
    mixed = _epv_da_and_maintenance_capex(
        facts, 1.0, da_source_concepts={2024: "DepreciationDepletionAndAmortization"}
    )
    assert mixed["depreciation_amortization"] == pytest.approx(29500.0 / 3.0, abs=1e-9)


def test_ingest_persists_the_da_concept_and_the_writer_reads_it(monkeypatch, tmp_path):
    """The D&A row carries the concept it was read from (``source_tags``), and
    the writer's lookup returns it by fiscal year."""
    from app.config import get_config
    from app.db import get_db, init_db, utc_now_iso
    from app.ingest.companyfacts import normalize_annual_facts_from_raw
    from app.valuation.valuation_writer import _da_source_concepts

    payload = {"facts": {"us-gaap": {"Depreciation": {"units": {"USD": [
        {"start": "2023-07-01", "end": "2024-06-30", "val": 15_200_000_000.0,
         "form": "10-K", "fp": "FY", "fy": 2024, "filed": "2024-07-30", "accn": "a-1"},
    ]}}}}}
    rows = [
        r for r in normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
        if r["line_item"] == "depreciation_amortization"
    ]
    assert [(r["value"], r["source_tags"]) for r in rows] == [(15200.0, "Depreciation")]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    with get_db(cfg=cfg) as conn:
        conn.execute(
            "INSERT INTO companyfacts_facts (ticker, fiscal_year, period_type, period_end, "
            "line_item, value, units, source_url, fetched_at, filed_date, form, accession, "
            "source_tags) VALUES ('ZZZ', 2024, 'FY', '2024-06-30', 'depreciation_amortization', "
            "15200.0, 'USD_millions', 'https://example.test/cf', ?, '2024-07-30', '10-K', "
            "'a-1', 'Depreciation')",
            (utc_now_iso(),),
        )
        concepts = _da_source_concepts(
            conn, "ZZZ", as_of_date="2025-01-01", issuer_cik=None, issuer_aliases=()
        )
    assert concepts == {2024: "Depreciation"}


# ── writer integration helpers ────────────────────────────────────────────────


def _init_writer_db(monkeypatch, tmp_path):
    from app.config import get_config
    from app.db import init_db

    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache", lambda http=None: {}
    )
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _seed(ticker, fields, years):
    from app.db import get_db, utc_now_iso

    with get_db() as conn:
        for line_item, values in fields.items():
            for year, value in zip(years, values, strict=True):
                conn.execute(
                    "INSERT INTO companyfacts_facts (ticker, fiscal_year, period_type, "
                    "period_end, line_item, value, units, source_url, fetched_at, filed_date, "
                    "accession) VALUES (?, ?, 'FY', ?, ?, ?, 'USD', ?, ?, ?, ?)",
                    (ticker, year, f"{year}-12-31", line_item, float(value),
                     f"https://example.test/companyfacts/{ticker}", utc_now_iso(),
                     f"{year + 1}-02-15", f"{ticker}-{year}"),
                )
        conn.commit()


def _outputs(ticker, method):
    import json

    from app.db import get_db

    with get_db() as conn:
        row = conn.execute(
            "SELECT outputs_json FROM valuations WHERE ticker = ? AND method = ?",
            (ticker, method),
        ).fetchone()
    return json.loads(row["outputs_json"]) if row else None


_YEARS5 = [2019, 2020, 2021, 2022, 2023]


def _flat_company(revenue):
    return {
        "revenue": revenue,
        "operating_income": [20.0] * 5,
        "net_income": [14.0] * 5,
        "equity": [300.0] * 5,
        "cfo": [22.0] * 5,
        "capex": [5.0] * 5,
        "shares_outstanding": [10.0] * 5,
        "total_debt": [0.0] * 5,
        "cash": [0.0] * 5,
        "preferred_equity": [0.0] * 5,
        "noncontrolling_interest": [0.0] * 5,
    }


def test_epv_capitalizes_the_spike_corrected_current_revenue(monkeypatch, tmp_path):
    """Revenue 100, 100, 100, 100 then a one-off 300 (a detected non-recurring
    spike, durable base 100) with operating income flat at 20. The EPV applies
    its margin to the DURABLE current revenue: margin 80 / 400 over the four
    ordinary years plus the spike year re-based to 100 -> 100 / 500 = 20%, x 100
    = normalized EBIT 20. Fed the raw series it took 100 / 700 x 300 = 42.86."""
    import pytest

    from app.valuation.valuation_writer import ensure_valuation

    _init_writer_db(monkeypatch, tmp_path)
    _seed("SPKE", _flat_company([100.0, 100.0, 100.0, 100.0, 300.0]), _YEARS5)
    ensure_valuation("SPKE", "2024-04-02", provider=None, force_refresh=True)
    epv = _outputs("SPKE", "epv")
    assert epv["current_revenue"] == 100.0
    assert epv["normalized_margin"] == pytest.approx(0.2, abs=1e-12)
    assert epv["normalized_ebit"] == pytest.approx(20.0, abs=1e-9)


def _downside(**overrides):
    from app.valuation.valuation_writer import _compute_downside_scenario

    kwargs = dict(
        owner_earnings=100.0,
        shares=10.0,
        net_debt=0.0,
        revenue_series=[(2024, 1000.0), (2023, 500.0), (2022, 250.0)],
        operating_income_series=[(2024, 200.0), (2023, 100.0), (2022, 50.0)],
        wacc=0.10,
        base_case_dcf=500.0,
    )
    kwargs.update(overrides)
    return _compute_downside_scenario(**kwargs)


def test_bear_epv_is_the_low_margin_on_current_revenue():
    """Margin is 20% every year; current revenue 1,000. Bear EBIT = 20% x 1,000 =
    200; NOPAT at the 21% statutory default 158; / 10% = 1,580; / 10 shares =
    158.00. The lowest operating-income LEVEL (50, from a year when revenue was
    250) gave 39.50 — an old, small year, not a stressed margin."""
    result = _downside()
    assert result["bear_case_epv"] == 158.0
    assert result["assumptions"]["operating_margin_low"] == 0.2
    assert result["assumptions"]["current_revenue"] == 1000.0


def test_bear_epv_gives_no_value_from_net_cash_beside_a_loss():
    """Low year: operating loss -10 on revenue 1,000 (-1% margin x 1,000 = -10),
    net cash 1,000. Before: (-100 + 1,000) / 10 = +90 a share of 'earnings power'
    made of cash. No earnings power -> the bear EPV is at most zero."""
    result = _downside(
        net_debt=-1000.0,
        revenue_series=[(2024, 1000.0), (2023, 1000.0), (2022, 1000.0)],
        operating_income_series=[(2024, 50.0), (2023, -10.0), (2022, 30.0)],
    )
    assert result["bear_case_epv"] == 0.0
    assert "BEAR_EPV_NO_EARNINGS_POWER" in result["flags"]
    assert result["downside_risk_class"] == "SEVERE"


def test_bear_epv_is_not_computed_for_a_reit():
    result = _downside(is_reit=True)
    assert result["bear_case_epv"] is None
    assert "BEAR_EPV_NOT_APPLICABLE_REIT" in result["flags"]


def test_downside_refuses_a_nan_share_count():
    """NaN passed ``shares <= 0`` and produced NaN bear values."""
    result = _downside(shares=float("nan"))
    assert result["bear_case_dcf"] is None
    assert result["bear_case_epv"] is None
    assert result["downside_risk_class"] == "UNKNOWN"


def test_unknown_reit_status_is_flagged_not_silently_ordinary(monkeypatch, tmp_path):
    """With no SIC on file the writer cannot tell a REIT from an ordinary
    company; EPV and EV/EBIT (the methods a REIT invalidates) carry
    REIT_STATUS_UNKNOWN. A known non-REIT SIC carries no such flag."""
    from app.valuation import valuation_writer

    _init_writer_db(monkeypatch, tmp_path)
    _seed("NOSIC", _flat_company([100.0, 105.0, 110.0, 115.0, 120.0]), _YEARS5)
    _seed("HASSIC", _flat_company([100.0, 105.0, 110.0, 115.0, 120.0]), _YEARS5)
    answers = {"NOSIC": (False, "NO_REGISTRANT_ROW"), "HASSIC": (False, "OK")}
    monkeypatch.setattr(
        valuation_writer, "lookup_is_reit", lambda **kw: answers[str(kw["ticker"])]
    )
    for ticker in answers:
        valuation_writer.ensure_valuation(ticker, "2024-04-02", provider=None, force_refresh=True)
    assert "REIT_STATUS_UNKNOWN" in _outputs("NOSIC", "epv")["flags"]
    assert "REIT_STATUS_UNKNOWN" in _outputs("NOSIC", "ev_ebit")["flags"]
    assert _outputs("NOSIC", "epv")["reit_status_reason"] == "NO_REGISTRANT_ROW"
    assert "REIT_STATUS_UNKNOWN" not in _outputs("HASSIC", "epv")["flags"]
    assert "REIT_STATUS_UNKNOWN" not in _outputs("HASSIC", "ev_ebit")["flags"]


def _dip_facts(*, cfo, revenue, operating_income):
    return {
        "cfo": _series(cfo, latest=2024),
        "capex": _series([40.0] * 5, latest=2024),
        "sbc": _series([5.0] * 5, latest=2024),
        "revenue": _series(revenue, latest=2024),
        "operating_income": _series(operating_income, latest=2024),
    }


def test_steady_decliner_cfo_dip_is_a_real_decline_not_lifted():
    """Revenue and operating income fall 10% a year (1,000 -> 656.1; 200 ->
    131.22). Neither latest year sits 1.25x below its own 5-year median (810 /
    1.25 = 648 < 656.1; 162 / 1.25 = 129.6 < 131.22), so the old rule lifted
    the CFO dip (60 against a median of 100) to 100: 100 - 40 - 5 = 55.
    Negative 5-year growth makes it a real decline: 60 - 40 - 5 = 15."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    result = _compute_owner_earnings(_dip_facts(
        cfo=[60.0, 100.0, 98.0, 102.0, 100.0],
        revenue=[656.1, 729.0, 810.0, 900.0, 1000.0],
        operating_income=[131.22, 145.8, 162.0, 180.0, 200.0],
    ))
    assert result["owner_earnings_latest"] == 15.0
    assert result["cfo_normalization"] == "DIP_KEPT_REAL_DECLINE"


def test_a_cash_burning_year_is_never_lifted_to_the_median():
    """CFO -20 against a median of 100 with a steady business: the old rule
    lifted it to 100 (55 of owner earnings). A non-positive CFO year stands:
    -20 - 40 - 5 = -65."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    result = _compute_owner_earnings(_dip_facts(
        cfo=[-20.0, 100.0, 98.0, 102.0, 100.0],
        revenue=[1010.0, 1000.0, 990.0, 1000.0, 1005.0],
        operating_income=[201.0, 200.0, 199.0, 201.0, 200.0],
    ))
    assert result["cfo_used"] == -20.0
    assert result["owner_earnings_latest"] == -65.0
    assert result["cfo_normalization"] == "DIP_KEPT_NONPOSITIVE"
    assert "CFO_DIP_NONPOSITIVE_KEPT" in result["flags"]


def test_recent_window_is_adjacent_fiscal_years_not_newest_rows():
    """Filed years 2016, 2021-2024: a five-year window ending 2024 is 2020-2024,
    so it holds four rows; 2016 is not spliced in as the fifth."""
    from app.valuation.owner_earnings_quality import _recent_common_ratios
    from app.valuation.valuation_writer import _n_years

    facts = {"revenue": [(2024, 5.0), (2023, 4.0), (2022, 3.0), (2021, 2.0), (2016, 1.0)]}
    assert _n_years(facts, "revenue", n=5) == [(2024, 5.0), (2023, 4.0), (2022, 3.0), (2021, 2.0)]

    num = [{"year": y, "value": v} for y, v in [(2019, 9.0), (2023, 4.0), (2024, 6.0)]]
    den = [{"year": y, "value": 2.0} for y in (2019, 2023, 2024)]
    assert [value for value, _refs in _recent_common_ratios(num, den, window=3)] == [2.0, 3.0]


def test_pricing_zone_keeps_an_adjusted_epv_of_exactly_zero(monkeypatch):
    """``epv_adjusted or epv`` replaced an adjusted EPV of 0.0 with the
    unadjusted EPV (50.0). Zero is a reading; the zone receives 0.0."""
    from app.valuation import valuation_writer

    seen = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        return {"pricing_zone": "INSUFFICIENT_DATA"}

    monkeypatch.setattr(valuation_writer, "_compute_pricing_zone", _spy)
    valuation_writer._margin_of_safety_scorecard(
        {
            "dcf": {"status": "OK", "base": 80.0},
            "epv": {"status": "OK", "value_per_share": 50.0},
            "epv_adjusted": {"status": "OK", "value_per_share": 0.0},
            "graham": {"status": "OK", "value_per_share": 40.0},
        },
        60.0,
        shares=10.0,
        net_debt=0.0,
    )
    assert seen["epv_adjusted"] == 0.0


def test_writer_hands_the_gates_filed_splits_to_sbc_trajectory(monkeypatch, tmp_path):
    """Shares 10, 10 then 20, 20, 20 with a filed 2-for-1 split in 2021. The
    writer passes the gate's split ratios to the SBC trajectory, so 2021 is a
    split (no uncorroborated break, and not +100% dilution)."""
    from app.valuation import pre_valuation_gate, valuation_writer

    monkeypatch.setattr(
        pre_valuation_gate,
        "split_ratio_rows_from_companyfacts",
        lambda *_a, **_k: [{"year": 2021, "value": 2.0, "derived_from": ["split"]}],
    )
    _init_writer_db(monkeypatch, tmp_path)
    fields = _flat_company([100.0, 105.0, 110.0, 115.0, 120.0])
    fields["shares_outstanding"] = [10.0, 10.0, 20.0, 20.0, 20.0]
    fields["sbc"] = [1.0] * 5
    _seed("SPLT", fields, _YEARS5)
    valuation_writer.ensure_valuation("SPLT", "2024-04-02", provider=None, force_refresh=True)
    sbc = _outputs("SPLT", "scorecard")["quality_context"]["sbc_trajectory"]
    assert sbc["shares_count_breaks"] == []
    assert "SHARE_COUNT_BREAK_UNCORROBORATED" not in sbc["sbc_flags"]
    assert [row["change_pct"] for row in sbc["shares_yoy_changes"]] == [0.0, 0.0, 0.0, 0.0]


def test_quick_value_prints_the_published_durable_dcf():
    """After a revenue spike the writer publishes the durable DCF (30.00) and
    keeps the raw 50.00 only for audit; ``ivi value`` printed the raw 50.00.
    It now prints 30.00 with the durable range, and the margin of safety at a
    price of 24 is (30 - 24) / 30 = 20%."""
    from app.valuation.quick_value import _method_row

    scorecard = {
        "pricing_zone_detail": {"dcf_base": 30.0, "dcf_raw_base": 50.0},
        "quality_context": {"dcf_durable": {"low": 25.0, "base": 30.0, "high": 36.0}},
    }
    row = _method_row("dcf", "DCF (base)", {"base": 50.0, "low": 40.0, "high": 60.0}, 24.0, scorecard)
    assert (row["value_per_share"], row["low"], row["high"]) == (30.0, 25.0, 36.0)
    assert row["margin_of_safety"] == 0.2
    assert "DCF_DURABLE_BASE_PUBLISHED" in row["flags"]
    plain = _method_row("dcf", "DCF (base)", {"base": 50.0, "low": 40.0, "high": 60.0}, 24.0, {})
    assert (plain["value_per_share"], plain["low"], plain["high"]) == (50.0, 40.0, 60.0)


def test_engine_fcf_fallback_subtracts_the_capex_magnitude():
    """CFO 100 with capex tagged negated (-30): FCF is 100 - 30 = 70, not
    100 - (-30) = 130."""
    from app.valuation.engine import build_ticker_valuation

    payload = {
        "ticker": "TST",
        "rows": [{"year": 2025, "cfo": 100.0, "capex": -30.0, "shares_outstanding": 10.0,
                  "net_debt": 0.0, "revenue": 1000.0}],
        "derived_signals": {},
    }
    out = build_ticker_valuation(payload, with_prices=False)
    assert out["fcf_coverage_entry"]["fcf_value"] == 70.0


def test_engine_is_num_rejects_bool_nan_and_infinity():
    from app.valuation.engine import _is_num

    assert [_is_num(v) for v in (1, 2.5, True, False, float("nan"), float("inf"), "1")] == [
        True, True, False, False, False, False, False
    ]


def test_gate_stress_net_debt_counts_short_term_investments_as_cash():
    """Debt 100, cash 30, short-term investments 50: net debt 20, as the
    valuation's own bridge counts it (the gate said 70)."""
    from app.valuation.pre_valuation_gate import _quality_module_rows

    rows = _quality_module_rows({
        "total_debt": [(2024, 100.0)],
        "cash": [(2024, 30.0)],
        "short_term_investments": [(2024, 50.0)],
    })
    assert rows[0]["net_debt"] == 20.0


def test_cash_coverage_uses_the_same_cash_as_net_debt_to_ebitda():
    """Cash 30 + short-term investments 50 over debt 100 = 0.8 (was 30 / 100)."""
    from app.valuation.valuation_writer import _capital_structure_health

    result = _capital_structure_health({
        "total_debt": [(2024, 100.0)],
        "cash": [(2024, 30.0)],
        "short_term_investments": [(2024, 50.0)],
        "equity": [(2024, 400.0)],
        "operating_income": [(2024, 40.0)],
    })
    assert result["cash_coverage"] == 0.8
    assert result["cash_coverage_basis"] == "CASH_PLUS_SHORT_TERM_INVESTMENTS"
    assert result["net_debt_to_ebitda"] == 0.5


def test_cash_adjusted_epv_takes_the_da_adjustment_with_a_tagged_addback(monkeypatch, tmp_path):
    """Operating income 20 on revenue 100 each year, with 10 of directly tagged
    intangible amortization added back (30, a 30% margin, on current revenue 100).
    With the add-back tagged, the base EPV's (physical D&A - maintenance capex)
    adjustment applies too: EBIT 30 + (D&A - maintenance capex), taxed at the
    21% default, over the WACC, over 10 shares."""
    import pytest

    from app.valuation.valuation_writer import ensure_valuation

    _init_writer_db(monkeypatch, tmp_path)
    fields = _flat_company([100.0] * 5)
    fields["depreciation_amortization"] = [15.0] * 5
    fields["intangible_amortization"] = [10.0] * 5
    fields["intangible_assets"] = [200.0] * 5
    fields["capex"] = [2.0] * 5
    _seed("AMRT", fields, _YEARS5)
    ensure_valuation("AMRT", "2024-04-02", provider=None, force_refresh=True)
    base = _outputs("AMRT", "epv")
    adjustment = base["depreciation_amortization"] - base["maintenance_capex"]
    expected = (30.0 + adjustment) * (1 - 0.21) / base["wacc"] / 10.0
    pzd = _outputs("AMRT", "scorecard")["pricing_zone_detail"]
    assert adjustment != 0.0
    assert pzd["epv_cash_adjusted"] == pytest.approx(expected, rel=1e-12)
