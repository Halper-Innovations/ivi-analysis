"""Anchor-correctness fixes from a valuation methodology audit.

Each test class maps to one confirmed finding family:
  - lease-double-count-vs-postlease-fcf (+ net_debt proxy lease-exclusive variant)
  - lease-current-portion-cross-surface-divergence (ingest component summation)
  - dcf-growth-on-maintenance-capex-oe / maint-capex-60pct-universal-haircut
  - owner-earnings-single-year-cfo-no-normalization
  - dcf-levered-cfo-discounted-at-wacc (FCFF conversion)
  - use-normalized-feeds-cfo-median-as-operating-income / epv-adjusted-drops-quality-normalization
  - stable-shares-stale-on-capital-events
  - dcf-anchor-ignores-durable-spike-correction
  - rnd-per-series-unit-scaling-mix
  - anomaly-zone-still-anchors-in-backtest
"""

from __future__ import annotations

import json

from app.db import get_db, init_db, utc_now_iso


def _flat_revenue(oi_series: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """A flat revenue line for `_epv`.

    With revenue flat, the normalized MARGIN on current revenue equals the mean
    of the operating-income LEVELS, so every case written against the pre-
    2026-09-02 method keeps its own arithmetic while satisfying the contract
    that the method must be handed a revenue series.
    """
    return [(year, 1000.0) for year, _ in oi_series]


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


def _seed_fact(conn, ticker, line_item, fiscal_year, value, *, period_end):
    conn.execute(
        "INSERT INTO companyfacts_facts "
        "(ticker, fiscal_year, period_type, period_end, line_item, value, units, "
        "source_url, fetched_at, filed_date, accession) "
        "VALUES (?, ?, 'FY', ?, ?, ?, 'USD', ?, ?, ?, ?)",
        (
            ticker.upper(),
            fiscal_year,
            period_end,
            line_item,
            float(value),
            f"https://example.test/companyfacts/{ticker.upper()}",
            utc_now_iso(),
            f"{fiscal_year + 1}-02-15",
            f"{ticker.upper()}-{fiscal_year}",
        ),
    )


_BASE_FIELDS = {
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
}
_SEED_YEARS = [2021, 2022, 2023]


def _seed_company(conn, ticker, fields=None, years=None):
    fields = fields or _BASE_FIELDS
    years = years or _SEED_YEARS
    for line_item, values in fields.items():
        for year, value in zip(years, values, strict=True):
            _seed_fact(conn, ticker, line_item, year, value, period_end=f"{year}-12-31")
    conn.commit()


def _method_row(conn, ticker, method):
    row = conn.execute(
        "SELECT inputs_json, outputs_json FROM valuations WHERE ticker = ? AND method = ?",
        (ticker.upper(), method),
    ).fetchone()
    assert row is not None, f"no {method} row persisted for {ticker}"
    return json.loads(row["inputs_json"] or "{}"), json.loads(row["outputs_json"] or "{}")


# ── lease-double-count-vs-postlease-fcf ───────────────────────────────────────


def test_valuation_bridge_uses_lease_exclusive_net_debt(monkeypatch, tmp_path):
    """ASC 842: CFO/OI flow bases are rent-burdened, so the DCF/EPV equity
    bridge must NOT also subtract the operating-lease liability."""
    _init_cfg(monkeypatch, tmp_path)
    fields = dict(_BASE_FIELDS)
    fields["operating_lease_liability"] = [400.0, 400.0, 400.0]
    with get_db() as conn:
        _seed_company(conn, "LEASEX", fields)

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("LEASEX", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        inputs, _ = _method_row(conn, "LEASEX", "dcf")
        _, scorecard = _method_row(conn, "LEASEX", "scorecard")

    # total_debt 200 - cash 100 = 100; the 400 lease liability is EXCLUDED.
    assert inputs["net_debt"] == 100.0
    flags = (scorecard.get("quality_context") or {}).get("net_debt_flags") or []
    assert "LEASE_EXCLUDED_POSTLEASE_FLOWS" in flags
    assert "LEASE_ADJUSTED" not in flags


def test_net_debt_proxy_exposes_lease_exclusive_variant(monkeypatch, tmp_path):
    """resolve_net_debt_proxy keeps the lease-INCLUSIVE proxy for leverage
    diagnostics but must expose a lease-EXCLUSIVE variant for valuation."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = {
        "entityName": "LEASE SPLIT INC",
        "facts": {
            "us-gaap": {
                "DebtCurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": 80_000_000.0,
                                "filed": "2026-01-20",
                                "form": "10-K",
                                "accn": "a-1",
                            }
                        ]
                    }
                },
                "LongTermDebtNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": 220_000_000.0,
                                "filed": "2026-01-20",
                                "form": "10-K",
                                "accn": "a-1",
                            }
                        ]
                    }
                },
                "OperatingLeaseLiability": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": 100_000_000.0,
                                "filed": "2026-01-20",
                                "form": "10-K",
                                "accn": "a-1",
                            }
                        ]
                    }
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": 60_000_000.0,
                                "filed": "2026-01-20",
                                "form": "10-K",
                                "accn": "a-1",
                            }
                        ]
                    }
                },
            }
        },
    }
    cache_path = cfg.cache_dir / "companyfacts" / "0000000900.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cik": "0000000900",
                "retrieved_at": utc_now_iso(),
                "http_status": 200,
                "companyfacts": payload,
            }
        ),
        encoding="utf-8",
    )

    from app.valuation.net_debt import resolve_net_debt_proxy

    result = resolve_net_debt_proxy(
        "LSX",
        "2026-02-14",
        run_id="nd_lease_split",
        facts_row={"ticker": "LSX", "cache_path": str(cache_path), "derived_from": ["facts.LSX"]},
        cfg=cfg,
    )
    assert result["status"] == "OK"
    # Lease-inclusive proxy unchanged: 300 debt + 100 lease - 60 cash = 340.
    assert result["net_debt_proxy"] == 340.0
    # Lease-exclusive variant for the valuation bridge: 300 - 60 = 240.
    assert result["net_debt_proxy_lease_exclusive"] == 240.0


def test_valuation_proxy_fallback_uses_lease_exclusive(monkeypatch, tmp_path):
    """When the inline facts path is unavailable and the as-of proxy supplies
    net debt, the valuation must consume the lease-exclusive variant."""
    _init_cfg(monkeypatch, tmp_path)
    # Seed everything EXCEPT total_debt/cash so the inline net-debt path fails
    # and ensure_valuation falls back to resolve_net_debt_proxy.
    fields = {k: v for k, v in _BASE_FIELDS.items() if k not in ("total_debt", "cash")}
    with get_db() as conn:
        _seed_company(conn, "PROXYL", fields)

    import app.valuation.valuation_writer as vw

    def _fake_proxy(ticker, as_of_date, **kwargs):
        return {
            "status": "OK",
            "net_debt_proxy": 500.0,  # lease-inclusive (leverage basis)
            "net_debt_proxy_lease_exclusive": 100.0,  # valuation basis
        }

    monkeypatch.setattr(vw, "resolve_net_debt_proxy", _fake_proxy)
    vw.ensure_valuation("PROXYL", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        inputs, _ = _method_row(conn, "PROXYL", "dcf")
    assert inputs["net_debt"] == 100.0


# ── lease-current-portion-cross-surface-divergence (ingest) ───────────────────


def _lease_fact(tag, val, *, end="2024-12-31", filed="2025-02-20", form="10-K"):
    return {
        tag: {
            "units": {
                "USD": [{"end": end, "val": val, "form": form, "filed": filed, "accn": "a-1"}]
            }
        }
    }


def test_ingest_lease_sums_current_plus_noncurrent_when_no_total_tag():
    """Filers tagging only the Current/Noncurrent components must not lose the
    current portion: per period_end, sum the components."""
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = {
        "facts": {
            "us-gaap": {
                **_lease_fact("OperatingLeaseLiabilityCurrent", 60_000_000.0),
                **_lease_fact("OperatingLeaseLiabilityNoncurrent", 440_000_000.0),
            }
        }
    }
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    rows = [f for f in facts if f["line_item"] == "operating_lease_liability"]
    assert len(rows) == 1
    assert rows[0]["value"] == 500.0


def test_ingest_lease_prefers_direct_total_tag():
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = {
        "facts": {
            "us-gaap": {
                **_lease_fact("OperatingLeaseLiability", 480_000_000.0),
                **_lease_fact("OperatingLeaseLiabilityCurrent", 60_000_000.0),
                **_lease_fact("OperatingLeaseLiabilityNoncurrent", 440_000_000.0),
            }
        }
    }
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    rows = [f for f in facts if f["line_item"] == "operating_lease_liability"]
    assert len(rows) == 1
    assert rows[0]["value"] == 480.0


def test_ingest_lease_noncurrent_only_still_ingested():
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = {
        "facts": {
            "us-gaap": {
                **_lease_fact("OperatingLeaseLiabilityNoncurrent", 440_000_000.0),
            }
        }
    }
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    rows = [f for f in facts if f["line_item"] == "operating_lease_liability"]
    assert len(rows) == 1
    assert rows[0]["value"] == 440.0


def test_ingest_restatement_cannot_cross_clobber_higher_priority_tag():
    """A later-filed fact from a LOWER-priority tag (e.g. a restricted-cash-
    inclusive comparative) must not replace the higher-priority tag's fact for
    the same fiscal year. Restatement replacement is same-tag only."""
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "val": 100_000_000.0,
                                "form": "10-K",
                                "filed": "2025-02-20",
                                "accn": "a-1",
                            },
                        ]
                    }
                },
                # Lower-priority, restricted-cash-inclusive tag filed LATER (next-year
                # comparative). Old behavior: clobbers the 100M fact. Fixed: ignored.
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "val": 150_000_000.0,
                                "form": "10-K",
                                "filed": "2026-02-20",
                                "accn": "a-2",
                            },
                        ]
                    }
                },
            }
        }
    }
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    rows = [f for f in facts if f["line_item"] == "cash"]
    assert len(rows) == 1
    assert rows[0]["value"] == 100.0


def test_ingest_restatement_same_tag_still_replaces():
    """Genuine restatements (same tag, later filed) must still win."""
    from app.ingest.companyfacts import normalize_annual_facts_from_raw

    payload = {
        "facts": {
            "us-gaap": {
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-12-31",
                                "val": 100_000_000.0,
                                "form": "10-K",
                                "filed": "2025-02-20",
                                "accn": "a-1",
                            },
                            {
                                "end": "2024-12-31",
                                "val": 120_000_000.0,
                                "form": "10-K/A",
                                "filed": "2025-06-01",
                                "accn": "a-2",
                            },
                        ]
                    }
                },
            }
        }
    }
    facts = normalize_annual_facts_from_raw(payload, cik="0000000001", years_back=10)
    rows = [f for f in facts if f["line_item"] == "cash"]
    assert len(rows) == 1
    assert rows[0]["value"] == 120.0


# ── maintenance-capex / CFO-normalization / FCFF owner-earnings family ────────


def _oe_facts(cfo, capex, sbc=None, revenue=None, interest_expense=None, cash=None):
    """Build a facts dict of (fiscal_year, value) series, latest year first."""

    def _series(vals):
        return [(2024 - i, v) for i, v in enumerate(vals)]

    facts = {"cfo": _series(cfo), "capex": _series(capex)}
    if sbc is not None:
        facts["sbc"] = _series(sbc)
    if revenue is not None:
        facts["revenue"] = _series(revenue)
    if interest_expense is not None:
        facts["interest_expense"] = _series(interest_expense)
    if cash is not None:
        facts["cash"] = _series(cash)
    return facts


def test_owner_earnings_normalizes_transient_cfo_peak():
    """Latest CFO > 1.25x the aligned median is a working-capital/cycle spike:
    substitute the median (audit: owner-earnings-single-year-cfo-no-normalization)."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[150.0, 100.0, 95.0, 105.0, 100.0],
        capex=[40.0, 40.0, 40.0, 40.0, 40.0],
        sbc=[5.0, 5.0, 5.0, 5.0, 5.0],
        revenue=[1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    )
    result = _compute_owner_earnings(facts)
    assert result["status"] == "OK"
    # median CFO 100 - capex 40 - sbc 5 = 55 (NOT 150 - 40 - 5 = 105)
    assert result["owner_earnings_latest"] == 55.0
    assert result["cfo_used"] == 100.0
    assert result["cfo_latest_raw"] == 150.0
    assert "CFO_PEAK_NORMALIZED" in result["flags"]
    assert result["confidence"] == "LOWER"


def test_owner_earnings_cfo_peak_exempt_for_strong_growers():
    """Strong secular growers (revenue CAGR > 8%) keep the latest CFO."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[150.0, 100.0, 95.0, 105.0, 100.0],
        capex=[40.0, 40.0, 40.0, 40.0, 40.0],
        sbc=[5.0, 5.0, 5.0, 5.0, 5.0],
        revenue=[1500.0, 1280.0, 1100.0, 950.0, 800.0],  # ~17%/yr
    )
    result = _compute_owner_earnings(facts)
    assert result["owner_earnings_latest"] == 105.0
    assert "CFO_PEAK_NORMALIZED" not in result["flags"]


def test_owner_earnings_stable_cfo_unchanged():
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[110.0, 100.0, 95.0, 105.0, 100.0],
        capex=[40.0, 40.0, 40.0, 40.0, 40.0],
        sbc=[5.0, 5.0, 5.0, 5.0, 5.0],
    )
    result = _compute_owner_earnings(facts)
    assert result["owner_earnings_latest"] == 65.0
    assert "CFO_PEAK_NORMALIZED" not in result["flags"]


def test_owner_earnings_fcff_interest_addback():
    """US-GAAP CFO is post-interest; the DCF discounts at WACC and subtracts
    net debt, so after-tax interest expense is added back (FCFF convention)."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[100.0, 100.0, 100.0],
        capex=[40.0, 40.0, 40.0],
        sbc=[5.0, 5.0, 5.0],
        interest_expense=[20.0, 20.0, 20.0],
    )
    result = _compute_owner_earnings(facts)
    # 100 - 40 - 5 + 20*(1-0.21) = 55 + 15.8 = 70.8
    assert abs(result["owner_earnings_latest"] - 70.8) < 1e-9
    assert result["interest_addback"] == 15.8
    assert "FCFF_INTEREST_ADDBACK" in result["flags"]


def test_owner_earnings_flags_unadjusted_interest_income_on_cash_rich():
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[100.0, 100.0, 100.0],
        capex=[40.0, 40.0, 40.0],
        sbc=[5.0, 5.0, 5.0],
        revenue=[200.0, 200.0, 200.0],
        cash=[500.0, 480.0, 460.0],  # cash >> revenue: interest income material
    )
    result = _compute_owner_earnings(facts)
    assert "INTEREST_INCOME_NOT_ADJUSTED" in result["flags"]


def test_dcf_owner_earnings_charges_full_capex(monkeypatch, tmp_path):
    """The DCF projects growth, so its OE base must charge FULL capex — the
    category maintenance haircut would credit growth at zero reinvestment cost
    (audit: dcf-growth-on-maintenance-capex-oe, maint-capex-60pct-universal-haircut)."""
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "FULLCPX")

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("FULLCPX", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        _, oe = _method_row(conn, "FULLCPX", "owner_earnings")

    # CFO 200 (stable: 200 <= 1.25 * median 185 = 231.25), full norm capex
    # mean(35, 32, 30) = 32.3333..., no SBC ingested -> 0.
    assert abs(oe["normalized_capex"] - (35.0 + 32.0 + 30.0) / 3.0) < 1e-9
    assert abs(oe["owner_earnings_latest"] - (200.0 - (35.0 + 32.0 + 30.0) / 3.0)) < 1e-9
    assert not any(str(f).startswith("MAINT_CAPEX_RATIO_") for f in oe.get("flags", []))


# ── anomaly-zone-still-anchors-in-backtest + shares exclusion ─────────────────


class _FixedProvider:
    provider_name = "fixed"

    def __init__(self, price):
        self._price = price

    def get_price_asof(self, ticker, as_of_date):
        from app.market.price_provider import PriceSnapshot

        # Reconstruct refuses adjusted-only quotes (NEEDS_DATA:RAW_PRICE_BASIS);
        # the raw close on the same basis keeps these fixtures classifiable.
        return PriceSnapshot(
            ticker=ticker,
            as_of_date=as_of_date,
            price=self._price,
            source="fixed",
            raw_price=self._price,
        )

    def get_last_diagnostic(self, ticker, as_of_date):
        return None


def test_reconstruct_skips_valuation_anomaly_zone(monkeypatch, tmp_path):
    """Negative-EPV names are zoned VALUATION_ANOMALY and live signals are
    suppressed — the backtest must not lift the surviving positive method as a
    full-confidence anchor (audit: anomaly-zone-still-anchors-in-backtest)."""
    _init_cfg(monkeypatch, tmp_path)
    fields = dict(_BASE_FIELDS)
    # Loss-making at the operating line (negative 5y avg OI -> negative EPV)
    # but CFO-positive, so the DCF leg alone would look cheap.
    fields["operating_income"] = [-60.0, -55.0, -50.0]
    fields["net_income"] = [-40.0, -38.0, -35.0]
    with get_db() as conn:
        _seed_company(conn, "ANOMX", fields)

    from app.backtest.reconstruct import reconstruct_signal_asof

    result = reconstruct_signal_asof("ANOMX", "2024-07-01", provider=_FixedProvider(5.0))
    assert result.signal is None
    assert result.skip_reason == "VALUATION_ANOMALY"


def test_reconstruct_skips_unstable_share_counts(monkeypatch, tmp_path):
    """SHARES_LATEST_FY_OUTLIER (uncorroborated >50% share jump) rows are
    excluded from deploy_ready classification — per-share anchors may be wrong
    by the full event factor."""
    _init_cfg(monkeypatch, tmp_path)
    fields = dict(_BASE_FIELDS)
    fields["shares_outstanding"] = [100.0, 100.0, 220.0]
    with get_db() as conn:
        _seed_company(conn, "UNSTBL", fields)

    from app.backtest.reconstruct import reconstruct_signal_asof

    result = reconstruct_signal_asof("UNSTBL", "2024-07-01", provider=_FixedProvider(12.0))
    assert result.signal is None
    assert result.skip_reason == "SHARES_UNSTABLE"


def test_reconstruct_signal_carries_pricing_zone(monkeypatch, tmp_path):
    """The signal exposes its pricing zone so the backtest diagnostic can stratify
    deploy_ready by zone."""
    _init_cfg(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_company(conn, "ZONEX")

    from app.backtest.reconstruct import reconstruct_signal_asof

    result = reconstruct_signal_asof("ZONEX", "2024-07-01", provider=_FixedProvider(12.0))
    assert result.signal is not None
    assert result.signal.pricing_zone in (
        "MARGIN_OF_SAFETY",
        "GROWTH_DEPENDENT",
        "SPECULATIVE_PREMIUM",
    )


# ── rnd-per-series-unit-scaling-mix ───────────────────────────────────────────


def _rnd_companyfacts(tags):
    return {
        "entityName": "Nano Tech Co.",
        "facts": {
            "us-gaap": {
                tag: {
                    "units": {
                        "USD": [
                            {"end": f"{year}-12-31", "filed": f"{year + 1}-02-01", "val": value}
                            for year, value in rows
                        ]
                    }
                }
                for tag, rows in tags.items()
            }
        },
    }


def test_rnd_scaling_is_global_across_all_series():
    """A microcap with sub-$1M R&D next to $M-scale revenue must get ONE unit
    decision: all series scaled to millions together. Per-series scaling left
    R&D in raw USD, saturating the ±40%-of-OI cap with mixed-unit arithmetic
    (audit: rnd-per-series-unit-scaling-mix)."""
    from app.valuation.rnd_capitalization import compute_rnd_adjusted_earnings
    from app.valuation.tech_category import ENTERPRISE_SOFTWARE

    payload = _rnd_companyfacts(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (2021, 20_000_000.0),
                (2022, 20_000_000.0),
                (2023, 20_000_000.0),
                (2024, 20_000_000.0),
            ],
            "GrossProfit": [
                (2021, 14_000_000.0),
                (2022, 14_000_000.0),
                (2023, 14_000_000.0),
                (2024, 14_000_000.0),
            ],
            "ResearchAndDevelopmentExpense": [
                (2021, 600_000.0),
                (2022, 700_000.0),
                (2023, 800_000.0),
                (2024, 900_000.0),
            ],
            "OperatingIncomeLoss": [
                (2021, 2_000_000.0),
                (2022, 2_000_000.0),
                (2023, 2_000_000.0),
                (2024, 2_000_000.0),
            ],
            "NetCashProvidedByUsedInOperatingActivities": [
                (2021, 3_200_000.0),
                (2022, 3_200_000.0),
                (2023, 3_200_000.0),
                (2024, 3_200_000.0),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                (2021, 450_000.0),
                (2022, 450_000.0),
                (2023, 450_000.0),
                (2024, 450_000.0),
            ],
            "ShareBasedCompensation": [
                (2021, 350_000.0),
                (2022, 350_000.0),
                (2023, 350_000.0),
                (2024, 350_000.0),
            ],
        }
    )
    result = compute_rnd_adjusted_earnings(
        "NANO",
        "2025-03-01",
        category=ENTERPRISE_SOFTWARE,
        companyfacts=payload,
    )
    assert result["status"] == "OK"
    # Consistent units (all /1e6): 2024 amortization = (0.6+0.7+0.8)/3 = 0.7;
    # adjustment = 0.9 - 0.7 = 0.2 ($0.2M) — NOT a saturated ±0.8 (40% of OI).
    assert abs(result["rnd_adjustment"] - 0.2) < 1e-6
    assert "RND_ADJUSTMENT_CAPPED" not in result["flags"]
    assert abs(result["adjusted_operating_income"] - 2.2) < 1e-6


def test_rnd_low_intensity_skip_gate_works_for_sub_million_rnd():
    """True intensity 2.5% must hit the <3% skip-gate; mixed units made the
    ratio rawUSD/millions (~25,000) and bypassed it."""
    from app.valuation.rnd_capitalization import compute_rnd_adjusted_earnings
    from app.valuation.tech_category import ENTERPRISE_SOFTWARE

    payload = _rnd_companyfacts(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                (2021, 20_000_000.0),
                (2022, 20_000_000.0),
                (2023, 20_000_000.0),
                (2024, 20_000_000.0),
            ],
            "GrossProfit": [
                (2021, 14_000_000.0),
                (2022, 14_000_000.0),
                (2023, 14_000_000.0),
                (2024, 14_000_000.0),
            ],
            "ResearchAndDevelopmentExpense": [
                (2021, 500_000.0),
                (2022, 500_000.0),
                (2023, 500_000.0),
                (2024, 500_000.0),
            ],
            "OperatingIncomeLoss": [
                (2021, 2_000_000.0),
                (2022, 2_000_000.0),
                (2023, 2_000_000.0),
                (2024, 2_000_000.0),
            ],
            "NetCashProvidedByUsedInOperatingActivities": [
                (2021, 3_200_000.0),
                (2022, 3_200_000.0),
                (2023, 3_200_000.0),
                (2024, 3_200_000.0),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                (2021, 450_000.0),
                (2022, 450_000.0),
                (2023, 450_000.0),
                (2024, 450_000.0),
            ],
        }
    )
    result = compute_rnd_adjusted_earnings(
        "NANO",
        "2025-03-01",
        category=ENTERPRISE_SOFTWARE,
        companyfacts=payload,
    )
    assert result["status"] == "SKIPPED_LOW_RND_INTENSITY"


# ── stable-shares-stale-on-capital-events ─────────────────────────────────────


def test_stable_shares_keeps_latest_on_corroborated_dilution():
    """A >50% share jump corroborated by an independent quarterly count is a
    genuine capital event (ATM/secondary) — keep the latest count, do NOT
    halve the divisor (audit: stable-shares-stale-on-capital-events)."""
    from app.valuation.share_count_stability import select_stable_shares

    series = [(2025, 200.0), (2024, 100.0), (2023, 98.0), (2022, 97.0)]
    shares, flag = select_stable_shares(series, corroborating_count=198.0)
    assert shares == 200.0
    assert flag == "SHARES_CAPITAL_EVENT"


def test_stable_shares_keeps_latest_on_corroborated_reverse_split():
    from app.valuation.share_count_stability import select_stable_shares

    series = [(2025, 10.0), (2024, 100.0), (2023, 99.0), (2022, 98.0)]
    shares, flag = select_stable_shares(series, corroborating_count=10.2)
    assert shares == 10.0
    assert flag == "SHARES_CAPITAL_EVENT"


def test_stable_shares_substitutes_median_when_corroboration_disagrees():
    """Cross-check disagreement = suspected corruption: keep the median."""
    from app.valuation.share_count_stability import select_stable_shares

    series = [(2025, 10.0), (2024, 100.0), (2023, 99.0), (2022, 98.0)]
    shares, flag = select_stable_shares(series, corroborating_count=99.0)
    assert shares == 99.0
    assert flag == "SHARES_LATEST_FY_OUTLIER"


def test_stable_shares_substitutes_median_without_corroboration():
    """No independent count available -> conservative median behavior holds."""
    from app.valuation.share_count_stability import select_stable_shares

    series = [(2025, 200.0), (2024, 100.0), (2023, 98.0), (2022, 97.0)]
    shares, flag = select_stable_shares(series)
    assert shares == 100.0
    assert flag == "SHARES_LATEST_FY_OUTLIER"


def test_valuation_uses_quarterly_corroboration_for_capital_event(monkeypatch, tmp_path):
    """End-to-end: a doubled latest-FY count corroborated by an as-of-visible
    quarterly share row is kept; pzd carries the shares flag for downstream
    consumers (backtest exclusion / output surfaces)."""
    _init_cfg(monkeypatch, tmp_path)
    fields = dict(_BASE_FIELDS)
    fields["shares_outstanding"] = [100.0, 100.0, 220.0]  # 2023 jump (2.2x)
    with get_db() as conn:
        _seed_company(conn, "ATMX", fields)
        # Quarterly cover-page count agreeing with the new level, visible as-of
        conn.execute(
            "INSERT INTO companyfacts_facts "
            "(ticker, fiscal_year, period_type, period_end, line_item, value, units, "
            "source_url, fetched_at, filed_date, accession) "
            "VALUES ('ATMX', 2024, 'Q1', '2024-03-31', 'shares_outstanding', "
            "222.0, 'shares_millions', ?, ?, '2024-05-01', 'ATMX-2024-Q1')",
            ("https://example.test/companyfacts/ATMX", utc_now_iso()),
        )
        conn.commit()

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("ATMX", "2024-06-30", provider=None, force_refresh=True)

    with get_db() as conn:
        inputs, scorecard = _method_row(conn, "ATMX", "scorecard")
    assert inputs["shares"] == 220.0
    pzd = scorecard.get("pricing_zone_detail") or {}
    assert pzd.get("shares_flag") == "SHARES_CAPITAL_EVENT"


def test_valuation_pzd_carries_outlier_flag_when_uncorroborated(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    fields = dict(_BASE_FIELDS)
    fields["shares_outstanding"] = [100.0, 100.0, 220.0]
    with get_db() as conn:
        _seed_company(conn, "CORRX", fields)

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("CORRX", "2024-06-30", provider=None, force_refresh=True)

    with get_db() as conn:
        inputs, scorecard = _method_row(conn, "CORRX", "scorecard")
    assert inputs["shares"] == 100.0  # median substitution preserved
    pzd = scorecard.get("pricing_zone_detail") or {}
    assert pzd.get("shares_flag") == "SHARES_LATEST_FY_OUTLIER"


# ── dcf-anchor-ignores-durable-spike-correction ───────────────────────────────


def test_durable_dcf_base_drives_pricing_zone_and_anchor(monkeypatch, tmp_path):
    """When a non-recurring revenue spike is detected, the corrected (durable)
    DCF must become pzd['dcf_base'] — the value the pricing zone, live packets
    and the backtest anchor all read — with the raw base preserved as
    pzd['dcf_raw_base'] (audit: dcf-anchor-ignores-durable-spike-correction)."""
    _init_cfg(monkeypatch, tmp_path)
    fields = dict(_BASE_FIELDS)
    # Latest-year revenue doubles (licensing/milestone spike): 900, 950 -> 2000
    fields["revenue"] = [900.0, 950.0, 2000.0]
    with get_db() as conn:
        _seed_company(conn, "SPIKEX", fields)

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("SPIKEX", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        _, dcf = _method_row(conn, "SPIKEX", "dcf")
        _, scorecard = _method_row(conn, "SPIKEX", "scorecard")

    pzd = scorecard.get("pricing_zone_detail") or {}
    qc = scorecard.get("quality_context") or {}
    durable = (qc.get("dcf_durable") or {}).get("base")
    raw_base = dcf.get("base")

    assert isinstance(durable, (int, float))
    assert isinstance(raw_base, (int, float))
    assert durable < raw_base  # the spike inflated the raw growth scenarios
    # The zone/anchor field carries the DURABLE value; raw is kept for audit.
    assert pzd.get("dcf_base") == durable
    assert pzd.get("dcf_raw_base") == raw_base


# ── EPV normalization pair ────────────────────────────────────────────────────


def test_epv_normalization_never_raises_anchor():
    """A 'conservative' USE_NORMALIZED adjustment must never RAISE the EPV
    (audit: use-normalized-feeds-cfo-median-as-operating-income, where the
    CFO-basis denominator raised EPV +59% vs doing nothing)."""
    from app.valuation.valuation_writer import _epv

    oi_series = [(2025, 30.0), (2024, 28.0), (2023, 25.0), (2022, 20.0), (2021, 18.0)]  # avg 24.2
    raw = _epv(oi_series, revenue_series=_flat_revenue(oi_series), net_debt=10.0, shares=5.0)
    result = _epv(
        oi_series,
        revenue_series=_flat_revenue(oi_series),
        net_debt=10.0,
        shares=5.0,
        epv_adjustment="USE_NORMALIZED",
        normalized_earnings=40.0,  # ABOVE avg
    )
    assert result["avg_operating_income"] == 24.2
    assert result["value_per_share"] == raw["value_per_share"]
    assert "EPV_CYCLICALLY_NORMALIZED" not in result["flags"]
    assert "EPV_NORMALIZATION_NOT_BINDING" in result["flags"]


def test_gate_normalized_earnings_from_oi_median(monkeypatch, tmp_path):
    """normalized_earnings must be on the SAME basis as the EPV input:
    median of last-5 operating income, negatives included — NOT a positive-only
    CFO median (which is after-tax + D&A-inclusive and gets re-taxed in _epv)."""
    _init_cfg(monkeypatch, tmp_path)

    from unittest.mock import patch

    from app.valuation.pre_valuation_gate import compute_quality_context

    facts = {
        # 20% off peak -> MODERATE_DECLINE -> epv_adjustment USE_NORMALIZED
        "revenue": [(2025, 80.0), (2024, 90.0), (2023, 100.0), (2022, 95.0), (2021, 92.0)],
        # negatives INCLUDED in the median: median(30, -10, 12, 14, 11) = 12
        "operating_income": [(2025, 30.0), (2024, -10.0), (2023, 12.0), (2022, 14.0), (2021, 11.0)],
        # CFO deliberately huge: the old code's positive-CFO median (~200)
        # would inflate the denominator
        "cfo": [(2025, 220.0), (2024, 200.0), (2023, 190.0), (2022, 180.0), (2021, 170.0)],
        "net_income": [(2025, 10.0), (2024, 9.0), (2023, 8.0), (2022, 7.0), (2021, 6.0)],
        "total_debt": [(2025, 30.0)],
        "cash": [(2025, 15.0)],
        "equity": [(2025, 50.0)],
        "shares_outstanding": [(2025, 10.0)],
    }
    with patch("app.valuation.pre_valuation_gate._safe_call", return_value={}):
        ctx = compute_quality_context("TEST", "2026-03-25", facts=facts)

    assert ctx["epv_adjustment"] == "USE_NORMALIZED"
    assert ctx["normalized_earnings"] == 12.0


def test_epv_adjusted_inherits_quality_normalization(monkeypatch, tmp_path):
    """The R&D-adjusted EPV (the pricing-zone/backtest anchor leg for tech
    categories) must apply the gate's normalization compositionally:
    normalized = gate OI-median + latest R&D delta (audit:
    epv-adjusted-drops-quality-normalization)."""
    _init_cfg(monkeypatch, tmp_path)
    fields = {
        # decline 20% from peak -> USE_NORMALIZED (and ADJUST, not BLOCK)
        "revenue": [100.0, 90.0, 80.0],
        # chronological [30, 10, 14]: avg 18, median 14
        "operating_income": [30.0, 10.0, 14.0],
        "net_income": [20.0, 6.0, 9.0],
        "equity": [500.0, 550.0, 600.0],
        "cfo": [25.0, 12.0, 16.0],
        "capex": [5.0, 5.0, 5.0],
        "shares_outstanding": [10.0, 10.0, 10.0],
        "total_debt": [20.0, 20.0, 20.0],
        "cash": [10.0, 10.0, 10.0],
        "preferred_equity": [0.0, 0.0, 0.0],
        "noncontrolling_interest": [0.0, 0.0, 0.0],
    }
    with get_db() as conn:
        _seed_company(conn, "RNDNRM", fields)

    import app.valuation.valuation_writer as vw

    monkeypatch.setattr(
        vw,
        "classify_company_category",
        lambda *a, **k: {
            "category": "ENTERPRISE_SOFTWARE",
            "confidence": "HIGH",
            "metrics_used": {},
        },
    )
    monkeypatch.setattr(
        vw,
        "compute_rnd_adjusted_earnings",
        lambda *a, **k: {
            "status": "OK",
            "amortization_life": 3,
            "rnd_adjustment": 5.0,
            "adjusted_owner_earnings": None,
            "flags": [],
            "guardrails": {
                "input_unit": "USD",
                "output_unit": "USD_millions",
                "input_unit_scale": 1_000_000.0,
            },
            "time_series": [
                {"year": 2021, "rnd_adjustment": 5.0},
                {"year": 2022, "rnd_adjustment": 5.0},
                {"year": 2023, "rnd_adjustment": 5.0},
            ],
        },
    )
    vw.ensure_valuation("RNDNRM", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        _, epv_adj = _method_row(conn, "RNDNRM", "epv_adjusted")

    # adjusted series avg = 18 + 5 = 23; normalized = gate median 14 + 5 = 19;
    # min(23, 19) -> 19, flagged.
    assert epv_adj["avg_operating_income"] == 19.0
    assert "EPV_CYCLICALLY_NORMALIZED" in epv_adj["flags"]


def test_gate_maintenance_ratio_growth_aware():
    """No-growth firms' capex is all maintenance: ratio 1.0 at CAGR <= 2%,
    category haircut only at demonstrated growth >= 8%, linear between."""
    from app.valuation.pre_valuation_gate import _growth_aware_maintenance_ratio

    assert _growth_aware_maintenance_ratio(0.60, None) == 0.60
    assert _growth_aware_maintenance_ratio(0.60, -0.05) == 1.0
    assert _growth_aware_maintenance_ratio(0.60, 0.02) == 1.0
    assert _growth_aware_maintenance_ratio(0.60, 0.08) == 0.60
    assert _growth_aware_maintenance_ratio(0.60, 0.12) == 0.60
    # Midpoint: 1.0 + (0.60 - 1.0) * 0.5 = 0.80
    assert abs(_growth_aware_maintenance_ratio(0.60, 0.05) - 0.80) < 1e-9


def test_owner_earnings_cfo_peak_exemption_spike_robust_unit():
    """Review OE-1: a latest-year revenue spike (KROS/PTCT licensing shape)
    inflates the endpoint CAGR past 8% and defeated the strong-grower
    exemption on exactly the co-spiking names the normalization targets.
    With the durable (spike-corrected) revenue series passed in, the
    exemption CAGR is flat -> normalization applies."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[150.0, 100.0, 95.0, 105.0, 100.0],
        capex=[40.0, 40.0, 40.0, 40.0, 40.0],
        sbc=[5.0, 5.0, 5.0, 5.0, 5.0],
        # latest-year revenue spike: endpoint CAGR (2000/1000)^(1/4)-1 = 18.9%
        revenue=[2000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    )
    durable = [(2024, 1000.0), (2023, 1000.0), (2022, 1000.0), (2021, 1000.0), (2020, 1000.0)]
    result = _compute_owner_earnings(facts, durable_revenue_series=durable)
    assert result["status"] == "OK"
    # median CFO 100 - capex 40 - sbc 5 = 55 (NOT raw 150 - 40 - 5 = 105)
    assert result["owner_earnings_latest"] == 55.0
    assert result["cfo_used"] == 100.0
    assert "CFO_PEAK_NORMALIZED" in result["flags"]


def test_owner_earnings_cfo_peak_exemption_genuine_grower_still_exempt_unit():
    """A genuine smooth grower stays exempt when the durable series IS the
    raw series (no spike detected -> caller passes None)."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[150.0, 100.0, 95.0, 105.0, 100.0],
        capex=[40.0, 40.0, 40.0, 40.0, 40.0],
        sbc=[5.0, 5.0, 5.0, 5.0, 5.0],
        revenue=[1500.0, 1280.0, 1100.0, 950.0, 800.0],  # ~17%/yr smooth
    )
    result = _compute_owner_earnings(facts, durable_revenue_series=None)
    assert result["owner_earnings_latest"] == 105.0
    assert "CFO_PEAK_NORMALIZED" not in result["flags"]


def test_owner_earnings_spike_exemption_wired_through_ensure_valuation(monkeypatch, tmp_path):
    """End-to-end (review OE-1): co-spiking revenue + CFO in the latest year.
    The writer must hoist the non-recurring-revenue detection ahead of owner
    earnings and pass the durable series, so the persisted owner_earnings row
    carries the normalized CFO, not the spike."""
    _init_cfg(monkeypatch, tmp_path)
    fields = {
        "revenue": [1000.0, 1000.0, 1000.0, 1000.0, 2000.0],
        "operating_income": [180.0, 185.0, 190.0, 195.0, 200.0],
        "net_income": [120.0, 125.0, 130.0, 135.0, 140.0],
        "equity": [500.0, 525.0, 550.0, 575.0, 600.0],
        "cfo": [100.0, 105.0, 95.0, 100.0, 150.0],
        "capex": [40.0, 40.0, 40.0, 40.0, 40.0],
        "sbc": [5.0, 5.0, 5.0, 5.0, 5.0],
        "shares_outstanding": [100.0, 100.0, 100.0, 100.0, 100.0],
        "total_debt": [200.0, 200.0, 200.0, 200.0, 200.0],
        "cash": [80.0, 85.0, 90.0, 95.0, 100.0],
    }
    with get_db() as conn:
        _seed_company(conn, "COSPIKE", fields, years=[2019, 2020, 2021, 2022, 2023])

    from app.valuation.valuation_writer import ensure_valuation

    ensure_valuation("COSPIKE", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        _, oe = _method_row(conn, "COSPIKE", "owner_earnings")

    # aligned median CFO 100 substituted for the 150 spike year:
    # 100 - 40 (flat capex, full ratio) - 5 (sbc) = 55
    assert oe["cfo_used"] == 100.0
    assert oe["owner_earnings_latest"] == 55.0
    assert "CFO_PEAK_NORMALIZED" in oe["flags"]


def test_proxy_net_debt_branch_flags_senior_claims_unknown(monkeypatch, tmp_path):
    """Review EVB-2: the inline nd_year branch deducts preferred/NCI, but the
    as-of proxy fallback (resolve_net_debt_proxy) has no senior-claims tag
    families — the same company gets a senior-claims-inclusive anchor on one
    branch and a senior-claims-free anchor on the other. Until the proxy
    extractor gains preferred/NCI families, proxy-path valuations must carry
    SENIOR_CLAIMS_UNKNOWN so the asymmetry is visible in pzd/coverage."""
    _init_cfg(monkeypatch, tmp_path)
    fields = {k: v for k, v in _BASE_FIELDS.items() if k not in ("total_debt", "cash")}
    # No (total_debt, cash) common year -> nd_year None -> proxy branch.
    with get_db() as conn:
        _seed_company(conn, "PROXYSC", fields)

    import app.valuation.valuation_writer as vw

    monkeypatch.setattr(
        vw,
        "resolve_net_debt_proxy",
        lambda *a, **k: {
            "status": "OK",
            "net_debt_proxy": 120.0,
            "net_debt_proxy_lease_exclusive": 100.0,
        },
    )
    vw.ensure_valuation("PROXYSC", "2024-04-02", provider=None, force_refresh=True)

    with get_db() as conn:
        _, scorecard = _method_row(conn, "PROXYSC", "scorecard")
    flags = (scorecard.get("quality_context") or {}).get("net_debt_flags") or []
    assert "ASOF_NET_DEBT_PROXY" in flags
    assert "SENIOR_CLAIMS_UNKNOWN" in flags


def test_owner_earnings_negative_median_cfo_spike_flagged_not_silent():
    """Review OE-3: a loss-history name (median CFO <= 0) whose latest CFO is
    a one-year positive spike keeps the spike — visibly flagged."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[80.0, -20.0, -30.0, -25.0, -15.0],
        capex=[10.0, 10.0, 10.0, 10.0, 10.0],
        sbc=[5.0, 5.0, 5.0, 5.0, 5.0],
    )
    result = _compute_owner_earnings(facts)
    assert result["cfo_used"] == 80.0
    assert "CFO_PEAK_CHECK_NEGATIVE_MEDIAN" in result["flags"]
    assert "CFO_PEAK_NORMALIZED" not in result["flags"]


def test_owner_earnings_short_aligned_history_flagged_not_silent():
    """Review OE-3: 5y CFO but only 2y of capex coverage -> aligned basis < 3
    -> the peak check can't run; the raw CFO stands but is flagged."""
    from app.valuation.valuation_writer import _compute_owner_earnings

    facts = _oe_facts(
        cfo=[150.0, 100.0, 95.0, 105.0, 100.0],
        capex=[40.0, 40.0],
        sbc=[5.0, 5.0],
    )
    result = _compute_owner_earnings(facts)
    assert result["cfo_used"] == 150.0
    assert "CFO_PEAK_CHECK_INSUFFICIENT_HISTORY" in result["flags"]
