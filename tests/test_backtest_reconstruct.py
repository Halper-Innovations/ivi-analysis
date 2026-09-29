from __future__ import annotations

import json

import pytest

from app.db import get_db, init_db, utc_now_iso


class _FixedProvider:
    provider_name = "fixed"

    def __init__(self, price):
        self._price = price

    def get_price_asof(self, ticker, as_of_date):
        from app.market.price_provider import PriceSnapshot

        return PriceSnapshot(
            ticker=ticker,
            as_of_date=as_of_date,
            price=self._price,
            source="fixed",
            raw_price=self._price,
        )

    def get_last_diagnostic(self, ticker, as_of_date):
        return None


def _init_cfg(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    # These reconstruction tests exercise valuation semantics, not SEC ticker
    # registry refresh. A missing temp cache must resolve locally and promptly.
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache",
        lambda http=None: {},
    )
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


_TEST_CIK = "0000000123"
_TEST_SOURCE_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/"
    f"CIK{_TEST_CIK}.json"
)


def _seed_fact(conn, ticker, line_item, fiscal_year, value, *, period_end):
    filed_date = f"{fiscal_year + 1}-02-15"
    conn.execute(
        """
        INSERT INTO companies(ticker, cik, name, created_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik
        """,
        (ticker.upper(), _TEST_CIK, f"{ticker.upper()} Test Co", utc_now_iso()),
    )
    conn.execute(
        "INSERT INTO companyfacts_facts "
        "(ticker, fiscal_year, period_type, period_end, line_item, value, units, "
        "source_url, fetched_at, filed_date, form, accession) "
        "VALUES (?, ?, 'FY', ?, ?, ?, ?, ?, ?, ?, '10-K', ?)",
        (
            ticker.upper(),
            fiscal_year,
            period_end,
            line_item,
            float(value),
            "shares_millions" if line_item == "shares_outstanding" else "USD_millions",
            _TEST_SOURCE_URL,
            utc_now_iso(),
            filed_date,
            f"{_TEST_CIK}-{str(fiscal_year + 1)[-2:]}-000001",
        ),
    )


# Minimal companyfacts line-item set discovered empirically (see report):
# revenue, operating_income, net_income, equity, cfo, capex, shares_outstanding,
# total_debt, cash for 3 consecutive FYs is sufficient for a positive EPV/DCF anchor.
_SEED_FIELDS = {
    "revenue": [900.0, 950.0, 1000.0],
    "operating_income": [180.0, 190.0, 200.0],
    "net_income": [120.0, 130.0, 140.0],
    "equity": [500.0, 550.0, 600.0],
    "cfo": [170.0, 185.0, 200.0],
    "capex": [30.0, 32.0, 35.0],
    "shares_outstanding": [100.0, 100.0, 100.0],
    "total_debt": [50.0, 50.0, 50.0],
    "cash": [80.0, 90.0, 100.0],
    "preferred_equity": [0.0, 0.0, 0.0],
    "noncontrolling_interest": [0.0, 0.0, 0.0],
}
_SEED_YEARS = [2021, 2022, 2023]


def _seed_company(conn, ticker):
    for line_item, values in _SEED_FIELDS.items():
        for year, value in zip(_SEED_YEARS, values, strict=True):
            _seed_fact(
                conn,
                ticker,
                line_item,
                year,
                value,
                period_end=f"{year}-12-31",
            )
    conn.commit()


def _writer_record(
    method,
    outputs=None,
    *,
    ticker="AAA",
    as_of_date="2023-10-04",
    run_id="backtest_recon_2024-01-02",
    outputs_json=None,
):
    return {
        "schema_version": "valuation_source_record_v1",
        "row": {
            "ticker": ticker,
            "as_of_date": as_of_date,
            "method": method,
            "inputs_json": "{}",
            "outputs_json": (
                outputs_json
                if outputs_json is not None
                else json.dumps({} if outputs is None else outputs, sort_keys=True)
            ),
            "warnings_json": "[]",
            "created_at": "2024-01-02T00:00:00+00:00",
            "valuation_writer_version": "test",
            "quality_gate_verdict": None,
            "confidence_class": None,
            "gate_reason_codes": None,
            "valuation_headwinds": None,
            "valuation_supports": None,
            "source_run_id": run_id,
        },
    }


def test_writer_output_payloads_accepts_only_exact_current_records():
    from app.backtest.reconstruct import _writer_output_payloads

    scorecard_payload = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {"dcf_base": 100.0, "epv_adjusted": 90.0},
    }
    reverse_payload = {"expectations_gap": {"bucket": "CHEAP_VS_EXPECTATIONS"}}
    scorecard = _writer_record("scorecard", scorecard_payload)
    reverse_dcf = _writer_record("reverse_dcf", reverse_payload)

    assert _writer_output_payloads(
        [scorecard, reverse_dcf],
        ticker="AAA",
        as_of_date="2023-10-04",
        run_id="backtest_recon_2024-01-02",
    ) == {
        "scorecard": scorecard_payload,
        "reverse_dcf": reverse_payload,
    }

    noncanonical = dict(scorecard)
    noncanonical["unexpected"] = True
    missing_row_field = {
        "schema_version": scorecard["schema_version"],
        "row": dict(scorecard["row"]),
    }
    missing_row_field["row"].pop("source_run_id")
    extra_row_field = {
        "schema_version": scorecard["schema_version"],
        "row": {**scorecard["row"], "unexpected": True},
    }
    invalid_record_sets = [
        [],
        [noncanonical],
        [missing_row_field],
        [extra_row_field],
        [_writer_record("scorecard", scorecard_payload, ticker="aaa")],
        [_writer_record("scorecard", scorecard_payload, as_of_date="2023-10-03")],
        [_writer_record("scorecard", scorecard_payload, run_id="another_run")],
        [_writer_record(" scorecard ", scorecard_payload)],
        [scorecard, scorecard],
        [_writer_record("scorecard", outputs_json="{")],
        [_writer_record("scorecard", [])],
    ]
    for records in invalid_record_sets:
        with pytest.raises(RuntimeError):
            _writer_output_payloads(
                records,
                ticker="AAA",
                as_of_date="2023-10-04",
                run_id="backtest_recon_2024-01-02",
            )


def test_reconstruct_missing_returned_methods_preserves_fail_closed_semantics(monkeypatch):
    from app.backtest import reconstruct

    monkeypatch.setattr(reconstruct, "_live_sector_model_routed", lambda *args: False)
    monkeypatch.setattr(reconstruct, "cap_category_asof", lambda *args: "small_cap")
    monkeypatch.setattr(
        reconstruct,
        "ensure_valuation",
        lambda *args, **kwargs: [
            _writer_record(
                "scorecard",
                {
                    "pricing_zone": "MARGIN_OF_SAFETY",
                    "pricing_zone_detail": {"dcf_base": 100.0, "epv_adjusted": 90.0},
                },
            )
        ],
    )

    missing_reverse = reconstruct.reconstruct_signal_asof(
        "AAA", "2024-01-02", provider=_FixedProvider(50.0)
    )
    assert missing_reverse.signal is not None
    assert missing_reverse.signal.expectations_gap_bucket == "EXPECTATIONS_GAP_UNRELIABLE"

    monkeypatch.setattr(
        reconstruct,
        "ensure_valuation",
        lambda *args, **kwargs: [
            _writer_record(
                "reverse_dcf",
                {"expectations_gap": {"bucket": "CHEAP_VS_EXPECTATIONS"}},
            )
        ],
    )
    missing_scorecard = reconstruct.reconstruct_signal_asof(
        "AAA", "2024-01-02", provider=_FixedProvider(50.0)
    )
    assert missing_scorecard.signal is None
    assert missing_scorecard.skip_reason == "NO_VALUATION_ANCHOR"


def test_reconstruct_returns_none_with_reason_when_no_fundamentals(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    from app.backtest.reconstruct import reconstruct_signal_asof

    result = reconstruct_signal_asof("NODATA", "2024-01-02", provider=_FixedProvider(10.0))
    assert result.signal is None
    # Granular zone reason (post-audit): a scorecard row exists but is zoned
    # INSUFFICIENT_DATA — distinguishable from "no scorecard at all" so the
    # diagnostic can quantify what the zone gate removed.
    assert result.skip_reason == "ZONE_INSUFFICIENT_DATA"


def test_reconstruct_returns_none_when_no_price(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "HASDATA")
    from app.backtest.reconstruct import reconstruct_signal_asof

    # Provider with no price -> NO_PRICE_AT_AS_OF, even with fundamentals seeded.
    result = reconstruct_signal_asof("HASDATA", "2024-07-01", provider=_FixedProvider(0.0))
    assert result.signal is None
    assert result.skip_reason == "NO_PRICE_AT_AS_OF"


def test_reconstruct_positive_path_deploy_ready(monkeypatch, tmp_path):
    """Seed 3 FYs ending <= the filing-lag cutoff, inject a deep-discount price,
    and assert the reconstructed signal exactly. The anchor (22.785048288155618)
    and gap bucket are read empirically from the persisted scorecard / reverse_dcf
    rows -- NOT recomputed here. as_of=2024-07-01 -> cutoff 2024-04-02, so the
    FY2023 row (period_end 2023-12-31) is the latest visible fiscal year.
    """
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "HASDATA")
    from app.backtest.reconstruct import reconstruct_signal_asof

    price = 12.0  # well below anchor * 0.75 == 17.088786216116713
    result = reconstruct_signal_asof("HASDATA", "2024-07-01", provider=_FixedProvider(price))

    assert result.skip_reason is None
    assert result.signal is not None
    signal = result.signal
    assert signal.ticker == "HASDATA"
    assert signal.as_of_date == "2024-07-01"
    # Anchor = max(dcf_base, epv_adjusted), read from the persisted scorecard
    # pricing_zone_detail (exact-literal; full-capex FCFF OE post-audit).
    assert signal.anchor == 22.785048288155618
    assert signal.price == 12.0
    # buy_price_target = round(22.785048288155618 * 0.75, 4)
    assert signal.buy_price_target == 17.0888
    # mos = round((22.785048288155618 - 12.0) / 22.785048288155618, 6)
    assert signal.mos == 0.473339
    assert signal.deploy_ready is True  # price 12.0 <= buy_price_target 17.0888
    # reverse_dcf row persists a real expectations-gap bucket at this discount.
    assert signal.expectations_gap_bucket == "CHEAP_VS_EXPECTATIONS"
    assert signal.cap_category is None


def test_refresh_failure_does_not_reuse_stale_measurement_rows(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "STALE")
    import app.backtest.reconstruct as reconstruct

    monkeypatch.setattr(reconstruct, "_live_sector_model_routed", lambda *args: False)
    provider = _FixedProvider(12.0)
    initial = reconstruct.reconstruct_signal_asof(
        "STALE",
        "2024-07-01",
        provider=provider,
    )
    assert initial.signal is not None
    with get_db() as conn:
        stale_rows = conn.execute(
            "SELECT COUNT(*) AS count FROM valuations_measurement "
            "WHERE ticker = 'STALE' AND as_of_date = '2024-04-02'"
        ).fetchone()["count"]
    assert stale_rows > 0

    import app.valuation.valuation_writer as valuation_writer

    inner_call = {}

    def _failed_inner(*args, **kwargs):
        inner_call["args"] = args
        inner_call["kwargs"] = kwargs
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(valuation_writer, "_ensure_valuation_inner", _failed_inner)
    result = reconstruct.reconstruct_signal_asof(
        "STALE",
        "2024-07-01",
        provider=provider,
    )

    assert result.signal is None
    assert result.skip_reason == "VALUATION_ERROR:RuntimeError"
    assert inner_call["args"] == ("STALE", "2024-04-02", provider)
    assert inner_call["kwargs"]["run_id"] == "backtest_recon_2024-07-01"
    assert inner_call["kwargs"]["price_override"] == 12.0
    assert inner_call["kwargs"]["force_refresh"] is True
    assert inner_call["kwargs"]["require_filed_asof"] is True


def test_reconstruct_ignores_future_fiscal_years(monkeypatch, tmp_path):
    """Look-ahead pin: a giant FY2024 row (period_end 2024-12-31, AFTER the
    2024-04-02 cutoff) must not move the anchor -- proving _load_facts'
    period_end <= as_of_date filter governs the as-of valuation.
    """
    _init_cfg(monkeypatch, tmp_path)
    future = {
        "revenue": 5000.0,
        "operating_income": 2000.0,
        "net_income": 1500.0,
        "equity": 3000.0,
        "cfo": 2000.0,
        "capex": 50.0,
        "shares_outstanding": 100.0,
        "total_debt": 50.0,
        "cash": 100.0,
    }
    with get_db() as conn:
        _seed_company(conn, "HASDATA")
        for line_item, value in future.items():
            _seed_fact(conn, "HASDATA", line_item, 2024, value, period_end="2024-12-31")
        conn.commit()

    from app.backtest.asof import assert_no_future_rows, effective_asof_cutoff
    from app.backtest.reconstruct import reconstruct_signal_asof

    cutoff = effective_asof_cutoff("2024-07-01")
    assert cutoff == "2024-04-02"

    result = reconstruct_signal_asof("HASDATA", "2024-07-01", provider=_FixedProvider(12.0))
    assert result.signal is not None
    # Unchanged from the no-future-FY case -> the FY2024 row was not consumed.
    assert result.signal.anchor == 22.785048288155618

    # Pin that every FY row feeding the anchor (those <= cutoff) is in-window.
    with get_db() as conn:
        rows = conn.execute(
            "SELECT period_end FROM companyfacts_facts "
            "WHERE ticker = 'HASDATA' AND period_type = 'FY' AND period_end <= ?",
            (cutoff,),
        ).fetchall()
    assert_no_future_rows(rows, as_of_date=cutoff, key="period_end")
    # And confirm the future row is genuinely present in the table (so the test
    # is exercising the filter, not an empty seed).
    with get_db() as conn:
        future_count = conn.execute(
            "SELECT COUNT(*) AS c FROM companyfacts_facts "
            "WHERE ticker = 'HASDATA' AND period_end = '2024-12-31'",
        ).fetchone()["c"]
    assert future_count == len(future)


def test_reconstruct_excludes_insurance_underwriter_names(monkeypatch, tmp_path):
    """Review ANCHOR-5: live anchors insurers on the residual-income insurance
    model (and can suppress ALL generic methods); the backtest has no
    insurance leg and would silently measure those names under generic
    dcf/epv anchors production never uses. Names the live surface routes
    away from generic anchors are excluded from measurement with a granular
    skip reason."""
    from types import SimpleNamespace

    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "INSURE")
    import app.insurance.routing as routing_mod
    import app.backtest.reconstruct as recon

    monkeypatch.setattr(
        routing_mod,
        "route_security",
        lambda ticker, as_of_date=None: SimpleNamespace(
            security_type="common", issuer_type="insurance_underwriter"
        ),
    )
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    result = recon.reconstruct_signal_asof("INSURE", "2024-07-01", provider=_FixedProvider(12.0))
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.signal is None
    assert result.skip_reason == "FINANCIAL_ISSUER_SECTOR_MODEL"


def test_reconstruct_excludes_non_common_securities(monkeypatch, tmp_path):
    """Preferred/depositary listings route to the preferred valuation live
    (generic_valuation_valid=False) — same exclusion."""
    from types import SimpleNamespace

    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "PREFD")
    import app.insurance.routing as routing_mod
    import app.backtest.reconstruct as recon

    monkeypatch.setattr(
        routing_mod,
        "route_security",
        lambda ticker, as_of_date=None: SimpleNamespace(
            security_type="preferred", issuer_type="non_insurer"
        ),
    )
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    result = recon.reconstruct_signal_asof("PREFD", "2024-07-01", provider=_FixedProvider(12.0))
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.signal is None
    assert result.skip_reason == "FINANCIAL_ISSUER_SECTOR_MODEL"


def test_reconstruct_keeps_unknown_and_non_insurer_names(monkeypatch, tmp_path):
    """UNKNOWN routing must NOT exclude (mirrors the live generic_invalid
    predicate: unknown security types keep generic anchors)."""
    from types import SimpleNamespace

    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "PLAIN")
    import app.insurance.routing as routing_mod
    import app.backtest.reconstruct as recon

    monkeypatch.setattr(
        routing_mod,
        "route_security",
        lambda ticker, as_of_date=None: SimpleNamespace(
            security_type="SECURITY_TYPE_UNKNOWN", issuer_type="ISSUER_TYPE_UNKNOWN"
        ),
    )
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    result = recon.reconstruct_signal_asof("PLAIN", "2024-07-01", provider=_FixedProvider(12.0))
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.skip_reason is None
    assert result.signal is not None


class _DualBasisProvider(_FixedProvider):
    """Adjusted price (returns basis) differs from raw close at T
    (classification basis) — the split/dividend-adjustment shape."""

    def __init__(self, adjusted, raw):
        super().__init__(adjusted)
        self._raw = raw

    def get_price_asof(self, ticker, as_of_date):
        from app.market.price_provider import PriceSnapshot

        return PriceSnapshot(
            ticker=ticker,
            as_of_date=as_of_date,
            price=self._price,
            source="fixed",
            raw_price=self._raw,
        )


def test_reconstruct_classifies_on_raw_close_when_available(monkeypatch, tmp_path):
    """Split-basis re-basing decision: deploy_ready/mos compare the as-of
    per-share anchor against the RAW close at T (the price live actually saw),
    while signal.price keeps the adjusted value so ledger entry/exit return
    legs stay on one adjusted series. Here the adjusted price (12.0) is deep
    in deploy territory but the raw close (30.0) is not: a reverse split
    between T and the download date inflated apparent cheapness."""
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "RAWBASIS")
    import app.backtest.reconstruct as recon

    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    result = recon.reconstruct_signal_asof(
        "RAWBASIS", "2024-07-01", provider=_DualBasisProvider(12.0, 30.0)
    )
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.signal is not None
    signal = result.signal
    # buy target = anchor 22.785048288155618 * 0.75 = 17.088786216116713
    assert signal.deploy_ready is False  # raw 30.0 > 17.09; adjusted 12 would have deployed
    assert signal.price == 12.0  # ledger/returns leg unchanged (adjusted)
    assert signal.classification_price == 30.0
    assert signal.price_basis == "raw_close"
    assert signal.mos == round((signal.anchor - 30.0) / signal.anchor, 6)
    # The raw quote must govern the writer too, not only the final MoS check:
    # adjusted 12.0 would be MARGIN_OF_SAFETY / cheap, while raw 30.0 is
    # SPECULATIVE_PREMIUM / expensive on these literal fixture fundamentals.
    assert signal.pricing_zone == "SPECULATIVE_PREMIUM"
    assert signal.expectations_gap_bucket == "EXPENSIVE_VS_EXPECTATIONS"


def test_reconstruct_rejects_adjusted_close_without_raw_basis(monkeypatch, tmp_path):
    """An adjusted close without its raw as-of quote cannot be combined with
    filed as-of shares, even when stale measurement rows already exist."""
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "ADJONLY")
    import app.backtest.reconstruct as recon

    class _AdjustedOnlyProvider:
        def get_price_asof(self, ticker, as_of_date):
            from app.market.price_provider import PriceSnapshot

            return PriceSnapshot(
                ticker=ticker,
                as_of_date=as_of_date,
                price=12.0,
                source="legacy-adjusted-only",
            )

    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    result = recon.reconstruct_signal_asof(
        "ADJONLY", "2024-07-01", provider=_AdjustedOnlyProvider()
    )
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.signal is None
    assert result.skip_reason == "NEEDS_DATA:RAW_PRICE_BASIS"


def test_reconstruct_quarantines_anchor_above_sanity_band(monkeypatch, tmp_path):
    """Backtest mirror of the live anchor-sanity quarantine: a $0.009
    junk OTC quote against a ~$23 anchor (ratio ~2,500x) must be excluded from
    measurement, not admitted as a deep-value deploy — one such row resolved
    +11,184pp excess and single-handedly inflated a headline mean."""
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "JUNKQ")
    import app.backtest.reconstruct as recon

    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    result = recon.reconstruct_signal_asof("JUNKQ", "2024-07-01", provider=_FixedProvider(0.009))
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.signal is None
    assert result.skip_reason == "QUARANTINE_ANCHOR_ABOVE_BAND"


def test_reconstruct_quarantines_anchor_below_sanity_band(monkeypatch, tmp_path):
    """Anchor below 0.2x the quote is the same magnitude/units-artifact class
    in the other direction (live quarantines both)."""
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "UNITSQ")
    import app.backtest.reconstruct as recon

    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    # anchor ~22.79; price 200 -> ratio ~0.114 < 0.2
    result = recon.reconstruct_signal_asof("UNITSQ", "2024-07-01", provider=_FixedProvider(200.0))
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.signal is None
    assert result.skip_reason == "QUARANTINE_ANCHOR_BELOW_BAND"


def test_reconstruct_sanity_band_uses_classification_price(monkeypatch, tmp_path):
    """The band evaluates against the RAW close (classification basis): an
    adjusted price inside the band must not mask a raw quote far outside it."""
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "RAWQ")
    import app.backtest.reconstruct as recon

    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    # adjusted 12.0 (ratio ~1.9, in band) but raw 0.009 (ratio ~2,532x)
    result = recon.reconstruct_signal_asof("RAWQ", "2024-07-01", provider=_DualBasisProvider(12.0, 0.009))
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.signal is None
    assert result.skip_reason == "QUARANTINE_ANCHOR_ABOVE_BAND"


def test_reconstruct_decline_class_capped_at_no_growth_anchor(monkeypatch, tmp_path):
    """Part-0c wired: a DECLINING-revenue company (CAGR ~ -3.3%, no gate
    block) must anchor at the no-growth EPV basis, not the DCF, end-to-end
    through ensure_valuation -> persisted scorecard -> select_anchor."""
    import json as _json

    _init_cfg(monkeypatch, tmp_path)
    fields = dict(_SEED_FIELDS)
    fields["revenue"] = [1000.0, 970.0, 935.0]
    with get_db() as conn:
        for line_item, values in fields.items():
            for year, value in zip(_SEED_YEARS, values, strict=True):
                _seed_fact(conn, "DECLN", line_item, year, value, period_end=f"{year}-12-31")
        conn.commit()
    import app.backtest.reconstruct as recon

    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    result = recon.reconstruct_signal_asof("DECLN", "2024-07-01", provider=_FixedProvider(12.0))
    recon._SECTOR_MODEL_ROUTING_CACHE.clear()
    assert result.signal is not None

    with get_db() as conn:
        row = conn.execute(
            "SELECT outputs_json FROM valuations_measurement WHERE ticker='DECLN' AND method='scorecard'"
        ).fetchone()
    out = _json.loads(row["outputs_json"])
    qc = out.get("quality_context") or {}
    pzd = out.get("pricing_zone_detail") or {}
    assert qc.get("revenue_trend_class") == "DECLINING"
    dcf_base = pzd.get("dcf_base")
    epv_adjusted = pzd.get("epv_adjusted")
    assert isinstance(dcf_base, (int, float)) and isinstance(epv_adjusted, (int, float))
    assert dcf_base > epv_adjusted  # the cap must have had something to do
    assert result.signal.anchor == epv_adjusted
