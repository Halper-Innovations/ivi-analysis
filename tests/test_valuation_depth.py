from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from app.db import get_db, init_db, utc_now_iso
from app.market.price_provider import PriceSnapshot
from app.market.price_prewarm import write_prices_prewarm_for_run
from app.rlm.executor import execute_actions
from app.rlm.schemas import PlannerOutput
from app.rlm.state import LoopBudgets, LoopState
from app.sector.cycle import sector_scoreboard_compare
from app.valuation.engine import build_ticker_valuation, write_valuations_for_run
from app.valuation.fundamentals import UNKNOWN
from app.valuation.rubric import apply_value_first_overlay, compute_value_first_score

ALLOWED_VALUATION_REASON_CODES = {
    "PRICE_UNKNOWN",
    "MISSING_FCF",
    "MISSING_SHARES",
    "MISSING_NET_DEBT",
    "INVALID_DENOMINATOR",
    "MODEL_PRECONDITION_FAILED",
    "ENGINE_EXCEPTION",
}


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _fundamentals_payload(*, ticker: str, shares: float | str = 100.0) -> dict:
    rows = [
        {"year": 2019, "revenue": 80.0, "gross_profit": 48.0, "operating_income": 14.0, "net_income": 10.0, "cfo": 13.0, "capex": 3.0, "fcf": 10.0, "shares_outstanding": 100.0, "net_debt": 50.0, "gross_margin": 0.60, "op_margin": 0.175, "fcf_margin": 0.125, "cfo_margin": 0.1625},
        {"year": 2020, "revenue": 90.0, "gross_profit": 54.0, "operating_income": 18.0, "net_income": 13.0, "cfo": 15.0, "capex": 3.0, "fcf": 12.0, "shares_outstanding": 101.0, "net_debt": 45.0, "gross_margin": 0.60, "op_margin": 0.20, "fcf_margin": 0.1333, "cfo_margin": 0.1667},
        {"year": 2021, "revenue": 102.0, "gross_profit": 63.0, "operating_income": 22.0, "net_income": 16.0, "cfo": 18.0, "capex": 4.0, "fcf": 14.0, "shares_outstanding": shares, "net_debt": 40.0, "gross_margin": 0.6176, "op_margin": 0.2157, "fcf_margin": 0.1372, "cfo_margin": 0.1764},
    ]
    traces = {
        str(row["year"]): {
            metric: {"derived_from": [f"dossier.time_series.standardized_rows[{row['year']}].{metric}"], "citations": []}
            for metric in ["revenue", "gross_profit", "operating_income", "net_income", "cfo", "capex", "fcf", "shares_outstanding", "net_debt"]
        }
        for row in rows
    }
    return {
        "fundamentals_version": "v1.1",
        "ticker": ticker,
        "run_id": "valuation_depth_test",
        "as_of_date": "2026-02-13",
        "rows": rows,
        "row_traces": traces,
        "derived_signals": {
            "revenue_cagr_5y": {"value": 0.12, "derived_from": ["fundamentals.rows[*].revenue"]},
            "revenue_cagr_10y": {"value": 0.10, "derived_from": ["fundamentals.rows[*].revenue"]},
            "operating_margin_trend_slope": {"value": 0.01, "derived_from": ["fundamentals.rows[*].op_margin"]},
            "dilution_rate_shares_cagr": {"value": 0.01, "derived_from": ["fundamentals.rows[*].shares_outstanding"]},
        },
        "gaps": [],
        "generated_at": utc_now_iso(),
    }


def _insert_price_quote(*, ticker: str, as_of_date: str, price: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO price_quotes(
                ticker, provider, as_of_date, price, currency, source_url, status,
                fetched_at, expires_at, raw_json, quote_hash
            )
            VALUES(?, ?, ?, ?, 'USD', ?, 'OK', ?, ?, ?, ?)
            """,
            (
                ticker,
                "test_provider",
                as_of_date,
                float(price),
                f"https://example.com/{ticker}",
                now,
                now,
                json.dumps({"price": price}),
                f"hash-{ticker}-{as_of_date}",
            ),
        )


def _seed_companyfacts_cache(*, cfg, cik: str, fixture_name: str) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "companyfacts" / f"{fixture_name}.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    # The raw SEC units are literal shares and dollars. Scale the compact
    # fixture magnitudes into realistic raw observations so normalization to
    # shares/USD millions does not manufacture a micro-cap anomaly.
    for taxonomy in (payload.get("facts") or {}).values():
        if not isinstance(taxonomy, dict):
            continue
        for concept in taxonomy.values():
            units = concept.get("units") if isinstance(concept, dict) else None
            if not isinstance(units, dict):
                continue
            for unit_name in ("shares", "USD"):
                for observation in units.get(unit_name) or []:
                    if isinstance(observation, dict) and isinstance(observation.get("val"), (int, float)):
                        observation["val"] = float(observation["val"]) * 1_000_000.0
    cache_path = cfg.cache_dir / "companyfacts" / f"{str(cik).zfill(10)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cik": str(cik).zfill(10),
                "retrieved_at": utc_now_iso(),
                "source_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json",
                "http_status": 200,
                "size_bytes": len(json.dumps(payload)),
                "companyfacts": payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _attach_filed_shares_provenance(
    *,
    fundamentals: dict,
    ticker: str,
    cik: str,
    shares_mm: float,
    run_id: str,
    period_end: str = "2025-12-31",
    filed_date: str = "2026-02-01",
) -> None:
    normalized_cik = str(cik).zfill(10)
    source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized_cik}.json"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik
            """,
            (ticker, normalized_cik, f"{ticker} Test Co", utc_now_iso()),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            )
            VALUES(?, 2025, 'FY', ?, 'shares_outstanding', ?,
                   'shares_millions', ?, ?, ?, '10-K', ?)
            """,
            (
                ticker,
                period_end,
                float(shares_mm),
                source_url,
                utc_now_iso(),
                filed_date,
                f"{normalized_cik}-26-000001",
            ),
        )
    fundamentals.update(
        {
            "run_id": run_id,
            "shares_outstanding_latest": float(shares_mm),
            "raw_shares_source_value": float(shares_mm),
            "raw_shares_source_unit": "shares_millions",
            "raw_shares_outstanding_mm": float(shares_mm),
            "shares_outstanding_mm": float(shares_mm),
            "shares_unit": "shares_millions",
            "shares_basis": "UNADJUSTED",
            "split_adjustment_factor": 1.0,
            "split_effective_date": None,
            "shares_period_end": period_end,
            "shares_filed_date": filed_date,
            "shares_source": "SEC_COMPANYFACTS",
            "shares_source_reference": f"{source_url}#shares-outstanding",
            "shares_trace": {
                "value": float(shares_mm),
                "unit": "shares_millions",
                "period_end": period_end,
                "filed_date": filed_date,
                "source": "SEC_COMPANYFACTS",
                "source_reference": f"{source_url}#shares-outstanding",
            },
        }
    )


def _dossier_payload(*, ticker: str, base_revenue: float, growth: float, margin: float, shares_growth: float, net_debt_start: float, net_debt_step: float) -> dict:
    rows = []
    traces = {}
    revenue = float(base_revenue)
    shares = 100.0
    net_debt = float(net_debt_start)
    for year in range(2017, 2027):
        if year > 2017:
            revenue *= (1.0 + float(growth))
            shares *= (1.0 + float(shares_growth))
            net_debt += float(net_debt_step)
        cfo = revenue * (margin + 0.02)
        capex = revenue * 0.05
        row = {
            "year": year,
            "revenue": round(revenue, 6),
            "gross_profit": round(revenue * (margin + 0.25), 6),
            "operating_income": round(revenue * margin, 6),
            "net_income": round(revenue * (margin - 0.05), 6),
            "cfo": round(cfo, 6),
            "capex": round(capex, 6),
            "fcf": round(cfo - capex, 6),
            "shares_outstanding": round(shares, 6),
            "net_debt": round(net_debt, 6),
        }
        rows.append(row)
        traces[str(year)] = {
            metric: {"derived_from": [f"dossier.time_series.standardized_rows[{year}].{metric}"], "citations": []}
            for metric in ["revenue", "gross_profit", "operating_income", "net_income", "cfo", "capex", "fcf", "shares_outstanding", "net_debt"]
        }
    return {
        "run_id": "depth_overlay_test",
        "ticker": ticker,
        "as_of_date": "2026-02-13",
        "time_series": {
            "standardized_rows": rows,
            "standardized_row_traces": traces,
            "derived_signals": [],
        },
        "claims": [],
        "items": [],
        "artifacts": {},
    }


def test_value_first_rubric_is_deterministic(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    fundamentals = _fundamentals_payload(ticker="AAA")
    valuation = {
        "implied_return_base": 0.18,
        "implied_fcf_growth": 0.08,
        "claims": {
            "implied_return_base": {"value": 0.18, "derived_from": ["x"]},
            "implied_fcf_growth": {"value": 0.08, "derived_from": ["y"]},
        },
    }
    whale = {"whale_signature_score": 72.0}
    first = compute_value_first_score(fundamentals=fundamentals, valuation=valuation, whale_payload=whale)
    second = compute_value_first_score(fundamentals=fundamentals, valuation=valuation, whale_payload=whale)
    assert first == second
    assert isinstance(first["score_total"], float)


def test_valuation_numeric_claims_include_derived_from(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _insert_price_quote(ticker="AAA", as_of_date="2026-02-13", price=25.0)
    fundamentals = _fundamentals_payload(ticker="AAA")
    valuation = build_ticker_valuation(fundamentals, as_of_date="2026-02-13")
    assert valuation["valuation_status"] in {"OK", "UNKNOWN"}
    for claim in valuation["claims"].values():
        if isinstance(claim.get("value"), (int, float)):
            assert claim.get("derived_from")


def test_mock_price_provider_enables_numeric_implied_return(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)

    class _Provider:
        provider_name = "stooq"

        def get_price_asof(self, ticker: str, as_of_date: str):
            return SimpleNamespace(
                ticker=ticker,
                as_of_date=as_of_date,
                price=40.0,
                currency="USD",
                source="stooq",
                url=f"https://example.com/{ticker}",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "resolved_symbol": f"{ticker.lower()}.us",
                "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "TRADING", "fallback_days_checked": 1, "asof_used": as_of_date},
                "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                "output_fields": {
                    "current_price": 40.0,
                    "price_asof_used": as_of_date,
                    "price_source": "stooq",
                    "confidence": "HIGH",
                },
            }

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _Provider())

    fundamentals = _fundamentals_payload(ticker="AAA")
    valuation = build_ticker_valuation(fundamentals, as_of_date="2026-02-13")
    assert isinstance(valuation["implied_return_base"], float)
    assert isinstance(valuation["valuation_gap"], float)
    assert valuation["valuation_reason_code"] is None
    assert valuation["valuation_inputs"]["price_current"] == "OK"
    assert valuation["input_snapshot"]["current_price_source"] == "stooq"
    assert valuation["input_snapshot"]["price_source"] == "stooq"
    assert valuation["input_snapshot"]["price_asof_used"] == "2026-02-13"
    assert valuation["price_gap"] is None
    derived = valuation["claims"]["implied_return_base"]["derived_from"]
    assert any(str(ref).startswith("valuation.price_evidence.") for ref in derived)

    run_id = "price_provider_numeric"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fundamentals_AAA.json").write_text(json.dumps(fundamentals, indent=2), encoding="utf-8")
    summary = write_valuations_for_run(
        run_id=run_id,
        tickers=["AAA"],
        as_of_date="2026-02-13",
        output_dir=run_dir,
    )
    assert summary["prices_ok"] == 1
    assert summary["prices_unknown"] == 0
    assert summary["price_provider_effective"] == "stooq"
    coverage_path = run_dir / "price_coverage.json"
    assert coverage_path.exists()
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage["ticker_count"] == 1
    assert coverage["entries"][0]["result"]["reason_code"] in {"PROVIDER_OK", "CACHE_HIT"}


def test_valuation_uses_shares_snapshot_and_reduces_missing_shares(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "shares_resolver_numeric"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    class _PriceProvider:
        provider_name = "stooq"

        def get_price_asof(self, ticker: str, as_of_date: str):
            return SimpleNamespace(
                ticker=ticker,
                as_of_date=as_of_date,
                price=33.0,
                currency="USD",
                source="stooq",
                url=f"https://example.com/{ticker}",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "resolved_symbol": f"{ticker.lower()}.us",
                "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "TRADING", "fallback_days_checked": 1, "asof_used": as_of_date},
                "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                "output_fields": {"current_price": 33.0, "price_asof_used": as_of_date, "price_source": "stooq", "confidence": "HIGH"},
            }

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _PriceProvider())

    fundamentals = _fundamentals_payload(ticker="AAA", shares=UNKNOWN)
    for row in fundamentals["rows"]:
        row["shares_outstanding"] = UNKNOWN
    _attach_filed_shares_provenance(
        fundamentals=fundamentals,
        ticker="AAA",
        cik="1",
        shares_mm=125.0,
        run_id=run_id,
    )
    (run_dir / "fundamentals_AAA.json").write_text(json.dumps(fundamentals, indent=2), encoding="utf-8")
    valuation = build_ticker_valuation(fundamentals, run_id=run_id, as_of_date="2026-02-13")
    assert valuation["valuation_status"] == "OK"
    assert valuation["valuation_reason_code"] is None
    assert isinstance(valuation["intrinsic_per_share_base"], float)
    assert isinstance(valuation["implied_return_base"], float)
    assert valuation["shares_coverage_entry"]["shares_status"] == "OK"
    assert valuation["input_snapshot"]["shares_source_resolution"] == "current_run_fundamentals"
    assert valuation["input_snapshot"]["shares_status"] == "OK"
    assert valuation["shares_evidence"] is not None
    derived_refs = [str(ref) for ref in valuation["claims"]["implied_return_base"]["derived_from"]]
    assert any("shares" in ref for ref in derived_refs)


def test_companyfacts_cache_fallback_makes_multiple_implied_returns_numeric(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    run_id = "companyfacts_numeric_depth"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    class _PriceProvider:
        provider_name = "stooq"

        def get_price_asof(self, ticker: str, as_of_date: str):
            return SimpleNamespace(
                ticker=ticker,
                as_of_date=as_of_date,
                price=40.0,
                currency="USD",
                source="stooq",
                url=f"https://example.com/{ticker}",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "resolved_symbol": f"{ticker.lower()}.us",
                "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "TRADING", "fallback_days_checked": 1, "asof_used": as_of_date},
                "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                "output_fields": {"current_price": 40.0, "price_asof_used": as_of_date, "price_source": "stooq", "confidence": "HIGH"},
            }

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _PriceProvider())

    for ticker, cik, fixture in [("AAA", "1", "AAA"), ("BBB", "2", "BBB")]:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO companies(ticker, cik, name, created_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik, name=excluded.name
                """,
                (ticker, cik, f"{ticker} Inc", utc_now_iso()),
            )
        _seed_companyfacts_cache(cfg=cfg, cik=cik, fixture_name=fixture)
        fundamentals = _fundamentals_payload(ticker=ticker, shares=UNKNOWN)
        fundamentals["as_of_date"] = "2026-02-14"
        fundamentals["shares_outstanding_latest"] = UNKNOWN
        for row in fundamentals["rows"]:
            row["shares_outstanding"] = UNKNOWN
            row["fcf"] = UNKNOWN
            row["cfo"] = UNKNOWN
            row["capex"] = UNKNOWN
        (run_dir / f"fundamentals_{ticker}.json").write_text(json.dumps(fundamentals, indent=2), encoding="utf-8")

    summary = write_valuations_for_run(
        run_id=run_id,
        tickers=["AAA", "BBB"],
        as_of_date="2026-02-14",
        output_dir=run_dir,
    )
    assert summary["ok_count"] >= 2
    assert summary["facts_coverage_path"].endswith("facts_coverage.json")

    known_count = 0
    for ticker in ["AAA", "BBB"]:
        valuation = json.loads((run_dir / f"valuation_{ticker}.json").read_text(encoding="utf-8"))
        if isinstance(valuation["implied_return_base"], float):
            known_count += 1
        assert valuation["shares_coverage_entry"]["shares_reason_code"] == "COMPANYFACTS_HIT"
        assert valuation["fcf_coverage_entry"]["fcf_reason_code"] in {"COMPANYFACTS_CFO_CAPEX_HIT", "COMPANYFACTS_FCF_HIT"}
        derived = [str(ref) for ref in valuation["claims"]["implied_return_base"]["derived_from"]]
        assert any("companyfacts." in ref for ref in derived)
    assert known_count >= 2


def test_valuation_unknown_inputs_penalize_score(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    
    class _MissingProvider:
        provider_name = "fallback"

        def get_price_asof(self, ticker: str, as_of_date: str):
            _ = (ticker, as_of_date)
            return None

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "resolved_symbol": f"{ticker.lower()}.us",
                "provider_attempts": [{"provider": "fallback", "status": "PROVIDER_NO_DATA"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "UNKNOWN", "fallback_days_checked": 0, "asof_used": None},
                "result": {"status": "UNKNOWN", "reason_code": "PROVIDER_NO_DATA", "reason_detail": "missing"},
                "output_fields": {
                    "current_price": UNKNOWN,
                    "price_asof_used": None,
                    "price_source": "fallback",
                    "confidence": None,
                },
            }

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _MissingProvider())
    fundamentals = _fundamentals_payload(ticker="BBB", shares=UNKNOWN)
    valuation = build_ticker_valuation(fundamentals, as_of_date="2026-02-13")
    assert valuation["valuation_status"] == "UNKNOWN"
    assert valuation["implied_return_base"] == UNKNOWN
    assert valuation["valuation_gap"] == UNKNOWN
    assert valuation["valuation_reason_code"] == "PRICE_UNKNOWN"
    assert valuation["price_gap"]["reason_code"] == "PROVIDER_NO_DATA"
    assert valuation["price_gap"]["cache_hit"] is False
    assert valuation["price_gap"]["derived_from"] == ["sector.price_coverage.entries[BBB]"]
    assert valuation["price_coverage_entry"]["result"]["reason_code"] == "PROVIDER_NO_DATA"
    score = compute_value_first_score(
        fundamentals=fundamentals,
        valuation=valuation,
        whale_payload={"whale_signature_score": 60.0},
    )
    assert score["valuation_score"] <= 10.0
    assert score["risk_penalty"] >= 1.0


def test_price_ok_but_unknown_implied_has_reason_code(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    class _Provider:
        provider_name = "stooq"

        def get_price_asof(self, ticker: str, as_of_date: str):
            return SimpleNamespace(
                ticker=ticker,
                as_of_date=as_of_date,
                price=50.0,
                currency="USD",
                source="stooq",
                url=f"https://example.com/{ticker}",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "resolved_symbol": f"{ticker.lower()}.us",
                "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "TRADING", "fallback_days_checked": 1, "asof_used": as_of_date},
                "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                "output_fields": {"current_price": 50.0, "price_asof_used": as_of_date, "price_source": "stooq", "confidence": "HIGH"},
            }

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _Provider())
    fundamentals = _fundamentals_payload(ticker="ZZZ", shares=UNKNOWN)
    for row in fundamentals["rows"]:
        row["shares_outstanding"] = UNKNOWN
    valuation = build_ticker_valuation(fundamentals, as_of_date="2026-02-13")
    assert valuation["price_coverage_entry"]["result"]["status"] == "OK"
    assert isinstance(valuation["input_snapshot"]["current_price"], float)
    assert valuation["implied_return_base"] == UNKNOWN
    assert valuation["valuation_reason_code"] in ALLOWED_VALUATION_REASON_CODES


def test_price_ok_entries_have_numeric_implied_or_reason(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    class _Provider:
        provider_name = "stooq"

        def get_price_asof(self, ticker: str, as_of_date: str):
            return SimpleNamespace(
                ticker=ticker,
                as_of_date=as_of_date,
                price=35.0,
                currency="USD",
                source="stooq",
                url=f"https://example.com/{ticker}",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "resolved_symbol": f"{ticker.lower()}.us",
                "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "TRADING", "fallback_days_checked": 1, "asof_used": as_of_date},
                "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                "output_fields": {"current_price": 35.0, "price_asof_used": as_of_date, "price_source": "stooq", "confidence": "HIGH"},
            }

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _Provider())
    payloads = [
        _fundamentals_payload(ticker="AAA", shares=100.0),
        _fundamentals_payload(ticker="BBB", shares=UNKNOWN),
    ]
    for payload in payloads:
        valuation = build_ticker_valuation(payload, as_of_date="2026-02-13")
        assert valuation["price_coverage_entry"]["result"]["status"] == "OK"
        if isinstance(valuation["implied_return_base"], float):
            assert valuation["valuation_reason_code"] is None
        else:
            assert valuation["valuation_reason_code"] in ALLOWED_VALUATION_REASON_CODES


def test_run_scoped_price_cache_precedence(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "price_scope_precedence"
    out_dir = cfg.outputs_dir / "prices" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "AAA.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "requested_as_of_date": "2026-02-13",
                "status": "OK",
                "snapshot": {
                    "ticker": "AAA",
                    "as_of_date": "2026-02-13",
                    "price": 77.0,
                    "currency": "USD",
                    "source": "seeded",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "url": "https://example.com/seeded",
                    "confidence": "HIGH",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fundamentals = _fundamentals_payload(ticker="AAA")
    valuation = build_ticker_valuation(
        fundamentals,
        run_id=run_id,
        as_of_date="2026-02-13",
    )
    assert valuation["input_snapshot"]["current_price"] == 77.0
    assert valuation["input_snapshot"]["current_price_source"] == "run_scoped_output"
    assert valuation["input_snapshot"]["price_source_resolution"] == "run_scoped_output"


def test_price_coverage_reason_codes_stable(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    run_id = "price_reason_codes_stable"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fundamentals_aaa = _fundamentals_payload(ticker="AAA")
    fundamentals_bbb = _fundamentals_payload(ticker="BBB")
    (run_dir / "fundamentals_AAA.json").write_text(json.dumps(fundamentals_aaa, indent=2), encoding="utf-8")
    (run_dir / "fundamentals_BBB.json").write_text(json.dumps(fundamentals_bbb, indent=2), encoding="utf-8")

    price_out_dir = cfg.outputs_dir / "prices" / run_id
    price_out_dir.mkdir(parents=True, exist_ok=True)
    (price_out_dir / "AAA.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "requested_as_of_date": "2026-02-13",
                "status": "OK",
                "snapshot": {
                    "ticker": "AAA",
                    "as_of_date": "2026-02-13",
                    "price": 51.0,
                    "currency": "USD",
                    "source": "seeded",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "url": "https://example.com/AAA",
                    "confidence": "HIGH",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_valuations_for_run(
        run_id=run_id,
        tickers=["AAA", "BBB"],
        as_of_date="2026-02-13",
        output_dir=run_dir,
    )
    price_cov = json.loads((run_dir / "price_coverage.json").read_text(encoding="utf-8"))
    assert price_cov["reason_counts"]["CACHE_HIT"] == 1
    assert price_cov["reason_counts"]["OFFLINE_NO_CACHE"] == 1
    entries = {str(row["ticker"]): row for row in price_cov["entries"]}
    assert entries["AAA"]["source_resolution"] == "run_scoped_output"
    assert entries["BBB"]["result"]["reason_code"] == "OFFLINE_NO_CACHE"
    assert any("online environment once to seed caches" in msg for msg in entries["BBB"]["suggestions"])


def test_net_debt_resolver_fallback_unblocks_valuation(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    _insert_price_quote(ticker="AAA", as_of_date="2026-02-13", price=25.0)
    fundamentals = _fundamentals_payload(ticker="AAA")
    fundamentals["rows"][-1]["net_debt"] = UNKNOWN

    monkeypatch.setattr(
        "app.valuation.engine.resolve_net_debt_proxy",
        lambda **_kwargs: {
            "status": "OK",
            "reason_code": "OK",
            "net_debt_proxy": 38.0,
            "derived_from": ["companyfacts.LongTermDebt|end_date=2025-12-31", "companyfacts.CashAndCashEquivalentsAtCarryingValue|end_date=2025-12-31"],
            "total_debt": {"value": 90.0, "tag": "LongTermDebt", "date": "2025-12-31"},
            "cash_equivalents": {"value": 52.0, "tag": "CashAndCashEquivalentsAtCarryingValue", "date": "2025-12-31"},
        },
    )

    valuation = build_ticker_valuation(fundamentals, as_of_date="2026-02-13")
    assert valuation["valuation_reason_code"] is None
    assert valuation["input_snapshot"]["net_debt"] == 38.0
    assert valuation["input_snapshot"]["net_debt_source_resolution"] == "companyfacts_cache"
    assert valuation["net_debt_gap"] is None
    assert valuation["net_debt_coverage_entry"]["net_debt_status"] == "OK"
    assert valuation["claims"]["intrinsic_per_share_base"]["derived_from"]


def test_write_valuations_emits_net_debt_coverage_artifact(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "net_debt_cov_artifact"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fundamentals = _fundamentals_payload(ticker="AAA")
    fundamentals["rows"][-1]["net_debt"] = UNKNOWN
    (run_dir / "fundamentals_AAA.json").write_text(json.dumps(fundamentals, indent=2), encoding="utf-8")

    class _PriceProvider:
        provider_name = "stooq"

        def get_price_asof(self, ticker: str, as_of_date: str):
            return PriceSnapshot(
                ticker=ticker,
                as_of_date=as_of_date,
                price=44.0,
                currency="USD",
                source="stooq",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                url=f"https://example.com/{ticker}",
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "TRADING", "fallback_days_checked": 1, "asof_used": as_of_date},
                "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                "output_fields": {"current_price": 44.0, "price_asof_used": as_of_date, "price_source": "stooq", "confidence": "HIGH"},
            }

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _PriceProvider())
    monkeypatch.setattr(
        "app.valuation.engine.resolve_net_debt_proxy",
        lambda **_kwargs: {
            "status": "OK",
            "reason_code": "OK",
            "net_debt_proxy": 41.0,
            "derived_from": ["facts_coverage.rows[AAA]", "companyfacts.LongTermDebt|end_date=2025-12-31"],
            "total_debt": {"value": 95.0, "tag": "LongTermDebt", "date": "2025-12-31"},
            "cash_equivalents": {"value": 54.0, "tag": "CashAndCashEquivalentsAtCarryingValue", "date": "2025-12-31"},
        },
    )

    summary = write_valuations_for_run(
        run_id=run_id,
        tickers=["AAA"],
        as_of_date="2026-02-13",
        output_dir=run_dir,
    )
    assert summary["net_debt_ok"] == 1
    assert summary["net_debt_unknown"] == 0
    coverage = json.loads((run_dir / "net_debt_coverage.json").read_text(encoding="utf-8"))
    assert coverage["reason_counts"]["OK"] == 1
    assert coverage["entries"][0]["net_debt_value"] == 41.0


def test_rlm_price_hydration_action_executes_and_reduces_price_unknown(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    run_id = "rlm_price_hydration_depth"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for ticker in ["AAA", "BBB"]:
        fundamentals = _fundamentals_payload(ticker=ticker, shares=100.0)
        (run_dir / f"fundamentals_{ticker}.json").write_text(json.dumps(fundamentals, indent=2), encoding="utf-8")

    before = write_valuations_for_run(
        run_id=run_id,
        tickers=["AAA", "BBB"],
        as_of_date="2026-02-13",
        output_dir=run_dir,
    )
    assert before["prices_unknown"] == 2
    known_before = 0
    for ticker in ["AAA", "BBB"]:
        valuation_payload = json.loads((run_dir / f"valuation_{ticker}.json").read_text(encoding="utf-8"))
        if isinstance(valuation_payload.get("implied_return_base"), float):
            known_before += 1

    def _fake_prewarm(**kwargs):
        output_dir = cfg.outputs_dir / "prices" / kwargs["run_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "AAA.json").write_text(
            json.dumps(
                {
                    "ticker": "AAA",
                    "requested_as_of_date": kwargs["as_of_date"],
                    "status": "OK",
                    "snapshot": {
                        "ticker": "AAA",
                        "as_of_date": kwargs["as_of_date"],
                        "price": 44.0,
                        "currency": "USD",
                        "source": "seeded_hydration",
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "url": "https://example.com/AAA",
                        "confidence": "HIGH",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        prewarm_path = run_dir / "prices_prewarm.json"
        prewarm_path.write_text("{}", encoding="utf-8")
        summary_path = output_dir / "prices_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "as_of_date": kwargs["as_of_date"],
            "tickers_requested": kwargs["tickers"],
            "tickers_ok": ["AAA"],
            "tickers_unknown": ["BBB"],
            "ok_count": 1,
            "unknown_count": 1,
            "reason_counts": {"CACHE_HIT": 1, "OFFLINE_NO_CACHE": 1},
            "prices_prewarm_path": str(prewarm_path),
            "prices_summary_path": str(summary_path),
        }

    monkeypatch.setattr("app.rlm.executor.write_prices_prewarm_for_run", _fake_prewarm)

    state = LoopState(
        run_id=run_id,
        sector="Software",
        as_of_date="2026-02-13",
        peer_set=["AAA", "BBB"],
        top_k_current=["AAA", "BBB"],
        artifacts={},
        budgets_remaining=LoopBudgets(sec_budget_remaining=100, llm_budget_remaining=2.0, max_iterations=3),
    )
    planner = PlannerOutput.model_validate(
        {
            "iteration": 1,
            "objective": "hydrate prices and recompute valuation",
            "actions": [
                {"action_type": "HYDRATE_PRICE_SNAPSHOT", "tickers": ["AAA", "BBB"], "fallback_days": 5},
                {"action_type": "RECOMPUTE_VALUATION", "tickers": ["AAA", "BBB"], "limit": 2},
            ],
        }
    )

    result = execute_actions(
        state=state,
        planner_output=planner,
        top_k=2,
        years_back_default=10,
        workers=1,
        with_research=False,
        with_synthesis=False,
        mode="depth",
    )
    assert result["executed_count"] == 2
    assert state.artifacts["prices_prewarm_path"].endswith("prices_prewarm.json")
    assert state.artifacts["price_coverage_path"].endswith("price_coverage.json")

    after_price_cov = json.loads((run_dir / "price_coverage.json").read_text(encoding="utf-8"))
    unknown_after = len(
        [
            row
            for row in (after_price_cov.get("entries") or [])
            if str(((row.get("result") or {}).get("status") or "UNKNOWN")).upper() != "OK"
        ]
    )
    assert unknown_after == 1

    known_after = 0
    for ticker in ["AAA", "BBB"]:
        valuation_payload = json.loads((run_dir / f"valuation_{ticker}.json").read_text(encoding="utf-8"))
        if isinstance(valuation_payload.get("implied_return_base"), float):
            known_after += 1
    assert known_after > known_before


def test_shares_coverage_artifact_shape_and_reason_codes(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "shares_coverage_shape"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    class _PriceProvider:
        provider_name = "stooq"

        def get_price_asof(self, ticker: str, as_of_date: str):
            return SimpleNamespace(
                ticker=ticker,
                as_of_date=as_of_date,
                price=44.0,
                currency="USD",
                source="stooq",
                url=f"https://example.com/{ticker}",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "TRADING", "fallback_days_checked": 1, "asof_used": as_of_date},
                "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                "output_fields": {"current_price": 44.0, "price_asof_used": as_of_date, "price_source": "stooq", "confidence": "HIGH"},
            }

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _PriceProvider())

    fundamentals_aaa = _fundamentals_payload(ticker="AAA", shares=UNKNOWN)
    fundamentals_bbb = _fundamentals_payload(ticker="BBB", shares=UNKNOWN)
    for row in fundamentals_aaa["rows"]:
        row["shares_outstanding"] = UNKNOWN
    for row in fundamentals_bbb["rows"]:
        row["shares_outstanding"] = UNKNOWN
    _attach_filed_shares_provenance(
        fundamentals=fundamentals_aaa,
        ticker="AAA",
        cik="1",
        shares_mm=101.0,
        run_id=run_id,
    )
    (run_dir / "fundamentals_AAA.json").write_text(json.dumps(fundamentals_aaa, indent=2), encoding="utf-8")
    (run_dir / "fundamentals_BBB.json").write_text(json.dumps(fundamentals_bbb, indent=2), encoding="utf-8")

    summary = write_valuations_for_run(
        run_id=run_id,
        tickers=["AAA", "BBB"],
        as_of_date="2026-02-13",
        output_dir=run_dir,
    )
    assert summary["shares_ok"] == 1
    assert summary["shares_unknown"] == 1

    shares_cov = json.loads((run_dir / "shares_coverage.json").read_text(encoding="utf-8"))
    assert shares_cov["ticker_count"] == 2
    assert shares_cov["reason_counts"]["NO_HISTORICAL_DOSSIER"] == 1
    assert shares_cov["reason_counts"]["OK"] == 1
    entries = {row["ticker"]: row for row in shares_cov["entries"]}
    assert entries["AAA"]["shares_status"] == "OK"
    assert entries["BBB"]["shares_status"] == "UNKNOWN"
    for key in {"shares_reason_code", "shares_value", "derived_from"}:
        assert key in entries["AAA"]
        assert key in entries["BBB"]


def test_prewarm_prices_enables_numeric_implied_return(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "prewarm_numeric_implied"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    class _PriceProvider:
        provider_name = "stooq"

        def get_price_asof(self, ticker: str, as_of_date: str):
            return PriceSnapshot(
                ticker=ticker,
                as_of_date=as_of_date,
                price=55.0,
                currency="USD",
                source="stooq",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                url=f"https://example.com/{ticker}",
                confidence="HIGH",
            )

        def get_last_diagnostic(self, ticker: str, as_of_date: str):
            return {
                "ticker": ticker,
                "requested_as_of": as_of_date,
                "resolved_symbol": f"{ticker.lower()}.us",
                "attempted_symbols": [f"{ticker.lower()}.us"],
                "asof_final_used": as_of_date,
                "provider_attempts": [{"provider": "stooq", "status": "PROVIDER_OK"}],
                "cache": {"hit": False, "path": "", "snapshot_found": False, "cached_as_of_used": None},
                "market_day": {"requested_day_type": "TRADING", "fallback_days_checked": 1, "asof_used": as_of_date},
                "result": {"status": "OK", "reason_code": "PROVIDER_OK", "reason_detail": "ok"},
                "output_fields": {
                    "current_price": 55.0,
                    "price_asof_used": as_of_date,
                    "asof_final_used": as_of_date,
                    "price_source": "stooq",
                    "confidence": "HIGH",
                },
            }

    monkeypatch.setattr("app.market.price_provider.build_price_provider", lambda **_kwargs: _PriceProvider())
    prewarm = write_prices_prewarm_for_run(
        run_id=run_id,
        as_of_date="2026-02-13",
        tickers=["AAA"],
        fallback_days=5,
        cfg=cfg,
    )
    assert prewarm["ok_count"] == 1

    fundamentals = _fundamentals_payload(ticker="AAA")
    (run_dir / "fundamentals_AAA.json").write_text(json.dumps(fundamentals, indent=2), encoding="utf-8")

    summary = write_valuations_for_run(
        run_id=run_id,
        tickers=["AAA"],
        as_of_date="2026-02-13",
        output_dir=run_dir,
    )
    assert summary["prices_ok"] == 1
    valuation = json.loads((run_dir / "valuation_AAA.json").read_text(encoding="utf-8"))
    assert isinstance(valuation["implied_return_base"], float)
    assert valuation["input_snapshot"]["current_price"] == 55.0
    assert valuation["input_snapshot"]["price_source_resolution"] == "run_scoped_output"


def test_value_overlay_writes_metrics_and_compare_orders(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "depth_overlay_test"
    sector_dir = cfg.sectors_dir / run_id
    dossier_dir = cfg.dossiers_dir / run_id
    sector_dir.mkdir(parents=True, exist_ok=True)
    dossier_dir.mkdir(parents=True, exist_ok=True)
    (dossier_dir / "AAA").mkdir(parents=True, exist_ok=True)
    (dossier_dir / "BBB").mkdir(parents=True, exist_ok=True)

    (dossier_dir / "AAA" / "dossier.json").write_text(
        json.dumps(_dossier_payload(ticker="AAA", base_revenue=100.0, growth=0.15, margin=0.22, shares_growth=0.01, net_debt_start=20.0, net_debt_step=-1.0), indent=2),
        encoding="utf-8",
    )
    (dossier_dir / "BBB" / "dossier.json").write_text(
        json.dumps(_dossier_payload(ticker="BBB", base_revenue=120.0, growth=0.04, margin=0.11, shares_growth=0.05, net_debt_start=120.0, net_debt_step=8.0), indent=2),
        encoding="utf-8",
    )
    (dossier_dir / "whale_signals_AAA.json").write_text(json.dumps({"whale_signature_score": 82.0}), encoding="utf-8")
    (dossier_dir / "whale_signals_BBB.json").write_text(json.dumps({"whale_signature_score": 51.0}), encoding="utf-8")

    (sector_dir / "peer_scoreboard.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "rows": [
                    {"ticker": "AAA", "metric_values": {"whale_signature_score": 82.0}, "metric_traces": {}, "whale_summary": {"gaps": [], "top_signals": []}},
                    {"ticker": "BBB", "metric_values": {"whale_signature_score": 51.0}, "metric_traces": {}, "whale_summary": {"gaps": [], "top_signals": []}},
                ],
                "metrics": ["whale_signature_score"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (sector_dir / "peer_rankings.json").write_text(
        json.dumps(
            {
                "rankings": [
                    {"ticker": "AAA", "overall_score": 10.0, "metric_ranks": {"future_whale_rank": 1, "whale_signature_rank": 1}},
                    {"ticker": "BBB", "overall_score": 9.0, "metric_ranks": {"future_whale_rank": 2, "whale_signature_rank": 2}},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    class _Provider:
        provider_name = "stooq"

        def get_quote(self, ticker: str, as_of_date: str):
            price = 30.0 if ticker == "AAA" else 220.0
            return SimpleNamespace(
                ticker=ticker,
                as_of_date=as_of_date,
                price=price,
                currency="USD",
                provider="stooq",
                status="OK",
                source_url=f"https://example.com/{ticker}",
                fetched_at=datetime.now(timezone.utc).isoformat(),
                expires_at=datetime.now(timezone.utc).isoformat(),
                provenance={"mocked": True},
            )

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _Provider())

    overlay = apply_value_first_overlay(
        run_id=run_id,
        as_of_date="2026-02-13",
        sector_run_dir=sector_dir,
        dossier_run_dir=dossier_dir,
    )
    assert overlay["ticker_count"] == 2

    scoreboard_payload = json.loads((sector_dir / "peer_scoreboard.json").read_text(encoding="utf-8"))
    keys = set((scoreboard_payload["rows"][0].get("metric_values") or {}).keys())
    for key in {"quality_score", "growth_score", "capital_discipline_score", "valuation_score", "risk_penalty", "score_total", "implied_return_base", "valuation_gap"}:
        assert key in keys
    for key in {"price_status", "valuation_status", "valuation_reason_code"}:
        assert key in keys

    valuation_cov_path = sector_dir / "valuation_coverage.json"
    assert valuation_cov_path.exists()
    valuation_cov = json.loads(valuation_cov_path.read_text(encoding="utf-8"))
    assert valuation_cov["ticker_count"] == 2
    assert len(valuation_cov["entries"]) == 2

    compared = sector_scoreboard_compare(run_id=run_id, metric="implied_return_base")
    assert compared["rows"][0]["ticker"] == "AAA"
    assert compared["rows"][0]["rank"] == 1


def test_price_none_keeps_implied_return_unknown_and_compare_all_unknown(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "depth_overlay_price_unknown"
    sector_dir = cfg.sectors_dir / run_id
    dossier_dir = cfg.dossiers_dir / run_id
    sector_dir.mkdir(parents=True, exist_ok=True)
    dossier_dir.mkdir(parents=True, exist_ok=True)
    (dossier_dir / "AAA").mkdir(parents=True, exist_ok=True)
    (dossier_dir / "BBB").mkdir(parents=True, exist_ok=True)

    class _Provider:
        provider_name = "disabled"

        def get_price_asof(self, ticker: str, as_of_date: str):
            _ = (ticker, as_of_date)
            return None

    monkeypatch.setattr("app.valuation.engine.get_default_provider", lambda *_args, **_kwargs: _Provider())

    (dossier_dir / "AAA" / "dossier.json").write_text(
        json.dumps(_dossier_payload(ticker="AAA", base_revenue=100.0, growth=0.10, margin=0.20, shares_growth=0.01, net_debt_start=20.0, net_debt_step=0.5), indent=2),
        encoding="utf-8",
    )
    (dossier_dir / "BBB" / "dossier.json").write_text(
        json.dumps(_dossier_payload(ticker="BBB", base_revenue=120.0, growth=0.06, margin=0.12, shares_growth=0.03, net_debt_start=40.0, net_debt_step=2.0), indent=2),
        encoding="utf-8",
    )
    (dossier_dir / "whale_signals_AAA.json").write_text(json.dumps({"whale_signature_score": 70.0}), encoding="utf-8")
    (dossier_dir / "whale_signals_BBB.json").write_text(json.dumps({"whale_signature_score": 65.0}), encoding="utf-8")

    (sector_dir / "peer_scoreboard.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "rows": [
                    {"ticker": "AAA", "metric_values": {"whale_signature_score": 70.0}, "metric_traces": {}, "whale_summary": {"gaps": [], "top_signals": []}},
                    {"ticker": "BBB", "metric_values": {"whale_signature_score": 65.0}, "metric_traces": {}, "whale_summary": {"gaps": [], "top_signals": []}},
                ],
                "metrics": ["whale_signature_score"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (sector_dir / "peer_rankings.json").write_text(
        json.dumps(
            {
                "rankings": [
                    {"ticker": "AAA", "overall_score": 10.0, "metric_ranks": {"future_whale_rank": 1, "whale_signature_rank": 1}},
                    {"ticker": "BBB", "overall_score": 9.0, "metric_ranks": {"future_whale_rank": 2, "whale_signature_rank": 2}},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    apply_value_first_overlay(
        run_id=run_id,
        as_of_date="2026-02-13",
        sector_run_dir=sector_dir,
        dossier_run_dir=dossier_dir,
    )
    scoreboard = json.loads((sector_dir / "peer_scoreboard.json").read_text(encoding="utf-8"))
    for row in scoreboard.get("rows") or []:
        assert row.get("metric_values", {}).get("implied_return_base") == UNKNOWN

    compared = sector_scoreboard_compare(run_id=run_id, metric="implied_return_base")
    assert compared["status"] == "ALL_UNKNOWN"
    assert compared["hint"] == "Missing current_price; run with --with-prices and ensure price snapshots exist"


def test_saturated_reverse_dcf_implied_growth_unknown_to_consumers(monkeypatch, tmp_path):
    """Review RDCF-1: an unsolvable (saturated) reverse-DCF must surface
    implied growth as UNKNOWN to flag-ignoring consumers (value-first rubric
    via valuation['implied_fcf_growth'], persisted claims) instead of the
    clipped bound — post-fix the bound is -0.25 for positive-EV money-losers,
    which the rubric scored as top-bucket 'reasonable growth' (+5)."""
    _init_cfg(monkeypatch, tmp_path)
    _insert_price_quote(ticker="SATR", as_of_date="2026-02-13", price=50.0)
    fundamentals = _fundamentals_payload(ticker="SATR")
    # Money-loser with large net cash: margin -10%, net debt -800, revenue
    # 1000, price 50 x 100 shares — the review fixture that saturates at the
    # HIGH bound with NEGATIVE margin (implied_growth bound -0.25).
    for row in fundamentals["rows"]:
        row["revenue"] = 1000.0
        row["fcf_margin"] = -0.10
        row["op_margin"] = -0.10
        row["net_debt"] = -800.0
        row["fcf"] = -100.0
        row["operating_income"] = -100.0
        row["net_income"] = -120.0
        row["cfo"] = -80.0
    valuation = build_ticker_valuation(fundamentals, as_of_date="2026-02-13")
    assert valuation["implied_fcf_growth"] == UNKNOWN
    assert valuation["claims"]["implied_fcf_growth"]["value"] == UNKNOWN
    # The rubric must take the missing-growth neutral branch, not score the bound.
    score = compute_value_first_score(fundamentals=fundamentals, valuation=valuation)
    assert "VALUATION_MISSING_IMPLIED_GROWTH" in score["gaps"]


def test_net_debt_snapshot_coverage_reports_lease_exclusive_value(monkeypatch, tmp_path):
    """Review EVB-5: coverage['net_debt_value'] persisted the lease-INCLUSIVE
    proxy while reason_detail said '(lease-exclusive)' and the bridge used
    the exclusive value — self-contradicting metadata."""
    _init_cfg(monkeypatch, tmp_path)
    from app.valuation import engine as engine_mod

    monkeypatch.setattr(
        engine_mod,
        "resolve_net_debt_proxy",
        lambda **kwargs: {
            "status": "OK",
            "reason_code": "OK",
            "net_debt_proxy": 120.0,
            "net_debt_proxy_lease_exclusive": 100.0,
            "total_debt": {"value": 150.0, "tag": "LongTermDebt", "date": "2025-12-31"},
            "cash_equivalents": {"value": 50.0, "tag": "Cash", "date": "2025-12-31"},
            "derived_from": ["companyfacts.TEST"],
        },
    )
    value, coverage = engine_mod._select_net_debt_snapshot(
        ticker="TEST",
        as_of_date="2026-02-13",
        run_id=None,
        fundamentals_payload={},
        latest_row={},
    )
    assert value == 100.0
    assert coverage["net_debt_value"] == 100.0
    assert "lease-exclusive" in coverage["net_debt_reason_detail"]
