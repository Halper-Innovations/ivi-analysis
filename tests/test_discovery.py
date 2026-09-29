from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.cli import _run_all_impl
from app.db import get_db, init_db
from app.discovery.market_cap import compute_market_cap, persist_market_cap
from app.discovery.metrics import DiscoveryMetricsResult
from app.discovery.policy import DiscoveryFilingSelection
from app.discovery.rubric import determine_discovery_stage, score_discovery_candidate
from app.discovery.schemas import DiscoveryCandidate
from app.discovery.runner import (
    _candidate_from_cached_payload,
    _load_cached_candidate_payload,
    _persist_candidate,
    _write_report_md,
    run_discovery,
)
from app.discovery.seed import read_seed_csv, seed_snapshot_hash
from app.ingest.sec_client import FilingStub
from app.valuation.price_provider import PriceQuote


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _insert_discovery_run(
    conn,
    *,
    run_id: str,
    run_as_of_date: str,
    tickers: list[str],
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO discovery_runs(
            run_id, run_as_of_date, seed_hash, config_hash, seed_path,
            status, processed_count, phase, tickers_targeted_json,
            processed_effective_dates_json, stats_json, created_at, updated_at
        ) VALUES (?, ?, 'seed-fixture', 'config-fixture', 'seed.csv', 'COMPLETED',
                  1, 'full', ?, '{}', '{}', ?, ?)
        """,
        (
            run_id,
            run_as_of_date,
            json.dumps(tickers),
            created_at,
            created_at,
        ),
    )


def test_seed_hash_is_deterministic(tmp_path):
    seed = tmp_path / "seed.csv"
    seed.write_text("ticker\nMSFT\nAAPL\nMSFT\nGOOGL\n", encoding="utf-8")
    rows_a = read_seed_csv(seed)
    rows_b = read_seed_csv(seed)
    assert rows_a == rows_b

    hash_a = seed_snapshot_hash(rows_a, include_metadata={"seed_path": str(seed)})
    hash_b = seed_snapshot_hash(rows_b, include_metadata={"seed_path": str(seed)})
    assert hash_a == hash_b


def test_candidate_writer_publishes_exact_receipt_and_direct_rows_are_unproven(
    monkeypatch,
    tmp_path,
):
    from app.discovery.lineage import (
        discovery_candidate_bindings_are_current,
        serialize_discovery_candidate_binding,
    )

    cfg = _init_temp_db(monkeypatch, tmp_path)
    candidate = DiscoveryCandidate(
        ticker="PROVEN",
        cik="0000000001",
        run_id="discovery_receipt_test",
        run_as_of_date="2026-07-24",
        effective_as_of_date="2026-07-23",
        market_cap=1_000_000_000.0,
        discovery_score=75.0,
        stage="ADVANCE_TO_DEEP",
        whale_fit_score=19.0,
        evidence_strength_score=7.0,
        key_reasons=["fixture"],
        recommended_action="ADD_TO_UNIVERSE",
        suggested_next_pipeline="FULL_RESEARCH",
    )
    with get_db() as conn:
        _insert_discovery_run(
            conn,
            run_id=candidate.run_id,
            run_as_of_date="2026-07-24",
            tickers=["PROVEN", "FORGED"],
            created_at="2026-07-24T11:00:00Z",
        )
        _persist_candidate(conn, candidate)
        proven = conn.execute(
            "SELECT * FROM discovery_candidates WHERE ticker = 'PROVEN'"
        ).fetchone()
        binding = serialize_discovery_candidate_binding(proven)
        assert binding is not None
        receipt_path = Path(binding["publication_receipt"]["path"])
        assert receipt_path.is_relative_to(cfg.discovery_dir / "candidate_publications")
        assert receipt_path.is_file()
        assert receipt_path.stat().st_nlink == 1
        assert discovery_candidate_bindings_are_current(conn, [binding]) is True
        receipt_alias = tmp_path / "receipt_alias.json"
        os.link(receipt_path, receipt_alias)
        assert serialize_discovery_candidate_binding(proven) is None
        receipt_alias.unlink()
        assert serialize_discovery_candidate_binding(proven) == binding

        forged_payload = {
            **candidate.model_dump(mode="json"),
            "ticker": "FORGED",
            "discovery_score": 999.0,
        }
        conn.execute(
            """
            INSERT INTO discovery_candidates(
                ticker, run_id, discovery_score, payload_json, created_at
            ) VALUES ('FORGED', 'discovery_receipt_test', 999.0, ?, ?)
            """,
            (json.dumps(forged_payload), "2026-07-24T12:00:00Z"),
        )
        forged = conn.execute(
            "SELECT * FROM discovery_candidates WHERE ticker = 'FORGED'"
        ).fetchone()
        assert serialize_discovery_candidate_binding(forged) is None


def test_cached_candidate_requires_current_receipt_and_chains_exact_source(
    monkeypatch,
    tmp_path,
):
    from app.discovery.lineage import (
        serialize_discovery_candidate_binding,
        serialized_discovery_candidate_binding_is_current,
    )

    _init_temp_db(monkeypatch, tmp_path)
    source_run_id = "discovery_cache_source"
    copied_run_id = "discovery_cache_copy"
    run_as_of_date = "2026-07-24"
    accession = "0000000000-26-000001"
    source_candidate = DiscoveryCandidate(
        ticker="CACHE",
        cik="0000000001",
        run_id=source_run_id,
        run_as_of_date=run_as_of_date,
        effective_as_of_date="2026-07-23",
        market_cap=1_000_000_000.0,
        discovery_score=75.0,
        stage="ADVANCE_TO_DEEP",
        whale_fit_score=19.0,
        evidence_strength_score=7.0,
        subscores_json={
            "whale_fit": 19.0,
            "cap_band": 10.0,
            "scoring_version": "v1.2",
        },
        key_reasons=["fixture"],
        recommended_action="ADD_TO_UNIVERSE",
        suggested_next_pipeline="FULL_RESEARCH",
        artifacts={"filing_accessions_used": [accession]},
    )
    source_payload = source_candidate.model_dump(mode="json")
    with get_db() as conn:
        for run_id, created_at in (
            (source_run_id, "2026-07-24T12:00:00+00:00"),
            (copied_run_id, "2026-07-24T13:00:00+00:00"),
        ):
            conn.execute(
                """
                INSERT INTO discovery_runs(
                    run_id, run_as_of_date, seed_hash, config_hash, seed_path,
                    status, processed_count, phase, tickers_targeted_json,
                    processed_effective_dates_json, stats_json, created_at, updated_at
                ) VALUES (?, ?, 'seed', 'config', 'seed.csv', 'COMPLETED', 1, 'full',
                          '["CACHE"]', '{}', '{}', ?, ?)
                """,
                (run_id, run_as_of_date, created_at, created_at),
            )
        conn.execute(
            """
            INSERT INTO discovery_candidates(
                ticker, run_id, discovery_score, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_candidate.ticker,
                source_run_id,
                source_candidate.discovery_score,
                json.dumps(source_payload),
                "2026-07-24T12:00:00+00:00",
            ),
        )

        assert (
            _load_cached_candidate_payload(
                conn,
                ticker=source_candidate.ticker,
                run_as_of_date=run_as_of_date,
                run_id=copied_run_id,
                selected_accessions=[accession],
            )
            is None
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) AS count FROM discovery_candidates WHERE run_id = ?",
                (copied_run_id,),
            ).fetchone()["count"]
            == 0
        )

        _persist_candidate(conn, source_candidate)
        source_row = conn.execute(
            "SELECT * FROM discovery_candidates WHERE ticker = ? AND run_id = ?",
            (source_candidate.ticker, source_run_id),
        ).fetchone()
        source_binding = serialize_discovery_candidate_binding(source_row)
        assert source_binding is not None
        assert serialized_discovery_candidate_binding_is_current(conn, source_binding) is True
        current_run_cached = _load_cached_candidate_payload(
            conn,
            ticker=source_candidate.ticker,
            run_as_of_date=run_as_of_date,
            run_id=source_run_id,
            selected_accessions=[accession],
        )
        assert current_run_cached is not None
        assert current_run_cached.is_current_run is True
        assert current_run_cached.source_binding == source_binding

        cached = _load_cached_candidate_payload(
            conn,
            ticker=source_candidate.ticker,
            run_as_of_date=run_as_of_date,
            run_id=copied_run_id,
            selected_accessions=[accession],
        )
        assert cached is not None
        assert cached.is_current_run is False
        assert cached.source_binding == source_binding
        copied_candidate = _candidate_from_cached_payload(
            cached.payload,
            run_id=copied_run_id,
            run_as_of_date=run_as_of_date,
            newly_surfaced=False,
            repeat_surfaced=True,
        )
        _persist_candidate(
            conn,
            copied_candidate,
            source_candidate_binding=cached.source_binding,
        )

        copied_row = conn.execute(
            "SELECT * FROM discovery_candidates WHERE ticker = ? AND run_id = ?",
            (source_candidate.ticker, copied_run_id),
        ).fetchone()
        copied_binding = serialize_discovery_candidate_binding(copied_row)
        assert copied_binding is not None
        assert serialized_discovery_candidate_binding_is_current(conn, copied_binding) is True
        receipt_payload = json.loads(
            Path(copied_binding["publication_receipt"]["path"]).read_text(encoding="utf-8")
        )
        assert receipt_payload["source_candidate_binding"] == source_binding

        conn.execute(
            """
            UPDATE discovery_candidates
            SET created_at = ?
            WHERE ticker = ? AND run_id = ?
            """,
            (
                "2026-07-24T14:00:00+00:00",
                source_candidate.ticker,
                source_run_id,
            ),
        )
        assert serialized_discovery_candidate_binding_is_current(conn, copied_binding) is False
        assert (
            _load_cached_candidate_payload(
                conn,
                ticker=source_candidate.ticker,
                run_as_of_date=run_as_of_date,
                run_id="discovery_cache_next",
                selected_accessions=[accession],
            )
            is None
        )


def test_cached_candidate_rejects_future_dates_and_binds_exact_source_run(
    monkeypatch,
    tmp_path,
):
    from app.discovery.lineage import (
        serialize_discovery_candidate_binding,
        serialized_discovery_candidate_binding_is_current,
    )

    _init_temp_db(monkeypatch, tmp_path)
    source_run_id = "discovery_temporal_source"
    copied_run_id = "discovery_temporal_copy"
    next_run_id = "discovery_temporal_next"
    accession = "0000000000-26-000099"
    with get_db() as conn:
        _insert_discovery_run(
            conn,
            run_id=source_run_id,
            run_as_of_date="2026-07-24",
            tickers=["FUTR"],
            created_at="2026-07-24T12:00:00+00:00",
        )
        _insert_discovery_run(
            conn,
            run_id=copied_run_id,
            run_as_of_date="2026-07-24",
            tickers=["FUTR"],
            created_at="2026-07-24T13:00:00+00:00",
        )
        _insert_discovery_run(
            conn,
            run_id=next_run_id,
            run_as_of_date="2026-07-24",
            tickers=["FUTR"],
            created_at="2026-07-24T14:00:00+00:00",
        )

        future_candidate = DiscoveryCandidate(
            ticker="FUTR",
            cik="0000000001",
            run_id=source_run_id,
            run_as_of_date="2026-07-25",
            effective_as_of_date="2026-07-25",
            market_cap=1_000_000_000.0,
            discovery_score=75.0,
            stage="ADVANCE_TO_DEEP",
            whale_fit_score=19.0,
            evidence_strength_score=7.0,
            subscores_json={
                "whale_fit": 19.0,
                "cap_band": 10.0,
                "scoring_version": "v1.2",
            },
            key_reasons=["future evidence fixture"],
            recommended_action="ADD_TO_UNIVERSE",
            suggested_next_pipeline="FULL_RESEARCH",
            artifacts={"filing_accessions_used": [accession]},
        )
        with pytest.raises(RuntimeError, match="exact source run metadata"):
            _persist_candidate(conn, future_candidate)

        future_effective_candidate = future_candidate.model_copy(
            update={"run_as_of_date": "2026-07-24"}
        )
        with pytest.raises(RuntimeError, match="exact source run metadata"):
            _persist_candidate(conn, future_effective_candidate)
        with pytest.raises(ValueError, match="future-dated evidence"):
            _candidate_from_cached_payload(
                future_effective_candidate.model_dump(mode="json"),
                run_id=copied_run_id,
                run_as_of_date="2026-07-24",
                newly_surfaced=False,
                repeat_surfaced=True,
            )

        source_candidate = future_candidate.model_copy(
            update={
                "run_as_of_date": "2026-07-24",
                "effective_as_of_date": "2026-07-23",
            }
        )
        _persist_candidate(conn, source_candidate)
        source_row = conn.execute(
            "SELECT * FROM discovery_candidates WHERE ticker = ? AND run_id = ?",
            ("FUTR", source_run_id),
        ).fetchone()
        source_binding = serialize_discovery_candidate_binding(source_row)
        assert source_binding is not None
        assert source_binding["candidate_run_state"] == {
            "run_id": source_run_id,
            "run_as_of_date": "2026-07-24",
            "seed_hash": "seed-fixture",
            "config_hash": "config-fixture",
            "seed_path": "seed.csv",
            "tickers_targeted_json": '["FUTR"]',
            "created_at": "2026-07-24T12:00:00+00:00",
        }

        cached = _load_cached_candidate_payload(
            conn,
            ticker="FUTR",
            run_as_of_date="2026-07-24",
            run_id=copied_run_id,
            selected_accessions=[accession],
        )
        assert cached is not None
        copied_candidate = _candidate_from_cached_payload(
            cached.payload,
            run_id=copied_run_id,
            run_as_of_date="2026-07-24",
            newly_surfaced=False,
            repeat_surfaced=True,
        )
        laundered_candidate = copied_candidate.model_copy(
            update={
                "discovery_score": 99.0,
                "key_reasons": ["payload changed after source binding"],
            }
        )
        with pytest.raises(RuntimeError, match="exactly match its source payload"):
            _persist_candidate(
                conn,
                laundered_candidate,
                source_candidate_binding=cached.source_binding,
            )

        _persist_candidate(
            conn,
            copied_candidate,
            source_candidate_binding=cached.source_binding,
        )
        copied_row = conn.execute(
            "SELECT * FROM discovery_candidates WHERE ticker = ? AND run_id = ?",
            ("FUTR", copied_run_id),
        ).fetchone()
        copied_binding = serialize_discovery_candidate_binding(copied_row)
        assert copied_binding is not None
        assert serialized_discovery_candidate_binding_is_current(conn, copied_binding)

        conn.execute(
            "UPDATE discovery_runs SET config_hash = ? WHERE run_id = ?",
            ("mutated-config", source_run_id),
        )
        assert serialized_discovery_candidate_binding_is_current(conn, copied_binding) is False
        assert (
            _load_cached_candidate_payload(
                conn,
                ticker="FUTR",
                run_as_of_date="2026-07-24",
                run_id=next_run_id,
                selected_accessions=[accession],
            )
            is None
        )


def test_market_cap_computation_with_price_and_shares():
    class _FakeProvider:
        def get_quote(self, ticker: str, as_of_date: str) -> PriceQuote:
            return PriceQuote(
                ticker=ticker,
                as_of_date=as_of_date,
                price=40.0,
                currency="USD",
                provider="stooq",
                status="OK",
                source_url="https://stooq.com/q/l/",
                fetched_at="2026-02-13T00:00:00+00:00",
                expires_at="2026-02-13T01:00:00+00:00",
                provenance={"provider": "stooq"},
            )

    result = compute_market_cap(
        ticker="AAPL",
        run_id="run_test",
        run_as_of_date="2026-02-13",
        effective_as_of_date="2026-01-30",
        shares_outstanding=200_000_000.0,
        price_provider=_FakeProvider(),
        cap_min=5_000_000_000.0,
        cap_max=50_000_000_000.0,
    )
    assert result.market_cap == 8_000_000_000.0
    assert result.market_cap_unit == "USD"
    assert result.market_cap_in_band is True
    assert result.market_cap_status == "OK"
    assert result.price_currency == "USD"
    assert result.price_basis == "UNADJUSTED"
    assert result.shares_unit == "shares"
    assert result.shares_basis == "raw"
    assert len(result.quote_snapshot_id) == 64


def test_market_cap_persistence_carries_explicit_units_and_snapshot(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)

    class _FakeProvider:
        def get_quote(self, ticker: str, as_of_date: str) -> PriceQuote:
            return PriceQuote(
                ticker=ticker,
                as_of_date=as_of_date,
                price=250.0,
                currency="USD",
                provider="fixture",
                status="OK",
                source_url="https://example.test/quote",
                fetched_at="2026-02-13T00:00:00+00:00",
                expires_at="2026-02-13T01:00:00+00:00",
                provenance={"fixture": True},
            )

    result = compute_market_cap(
        ticker="MEGA",
        run_id="run_units",
        run_as_of_date="2026-02-13",
        effective_as_of_date="2026-02-13",
        shares_outstanding=8_000_000_000.0,
        price_provider=_FakeProvider(),
        cap_min=1_000_000_000.0,
        cap_max=3_000_000_000_000.0,
    )
    with get_db(cfg) as conn:
        persist_market_cap(conn, result)
        row = conn.execute(
            """
            SELECT market_cap, market_cap_unit, price_currency, price_basis,
                   quote_snapshot_id, shares_unit, shares_basis,
                   split_adjustment_factor, split_effective_date
            FROM market_caps
            WHERE ticker = 'MEGA' AND run_id = 'run_units'
            """
        ).fetchone()

    assert row["market_cap"] == 2_000_000_000_000.0
    assert row["market_cap_unit"] == "USD"
    assert row["price_currency"] == "USD"
    assert row["price_basis"] == "UNADJUSTED"
    assert row["quote_snapshot_id"] == result.quote_snapshot_id
    assert row["shares_unit"] == "shares"
    assert row["shares_basis"] == "raw"
    assert row["split_adjustment_factor"] == 1.0
    assert row["split_effective_date"] is None


def test_discovery_score_is_deterministic():
    metrics = {
        "ttm_revenue": 9_000_000_000.0,
        "gross_margin": 0.44,
        "operating_margin": 0.14,
        "fcf": 700_000_000.0,
        "shares_outstanding": 300_000_000.0,
        "revenue_acceleration": 0.05,
        "gross_margin_change_qoq": 0.015,
        "operating_margin_change_qoq": 0.012,
        "fcf_change_qoq": 80_000_000.0,
        "liquidity_stress_score": 2,
        "net_debt": 500_000_000.0,
    }
    a = score_discovery_candidate(
        metrics=metrics,
        market_cap=12_000_000_000.0,
        market_cap_in_band=True,
        price=40.0,
        shares_outstanding=300_000_000.0,
    )
    b = score_discovery_candidate(
        metrics=metrics,
        market_cap=12_000_000_000.0,
        market_cap_in_band=True,
        price=40.0,
        shares_outstanding=300_000_000.0,
    )
    assert a.total_score == b.total_score
    assert a.subscores == b.subscores
    assert a.flags == b.flags


def test_discovery_valuation_does_not_treat_missing_net_debt_as_zero():
    base_metrics = {
        "ttm_revenue": 1_000.0,
        "gross_margin": 0.4,
        "operating_margin": 0.1,
        "fcf": 100.0,
        "shares_outstanding": 10.0,
        "liquidity_stress_score": 2.0,
    }
    missing = score_discovery_candidate(
        metrics=base_metrics,
        market_cap=1_000.0,
        market_cap_in_band=True,
        price=None,
        shares_outstanding=10.0,
    )
    explicit_zero = score_discovery_candidate(
        metrics={**base_metrics, "net_debt": 0.0},
        market_cap=1_000.0,
        market_cap_in_band=True,
        price=None,
        shares_outstanding=10.0,
    )

    assert missing.subscores["valuation_plausibility"] == 0.0
    assert "VALUATION_PLAUSIBILITY_LOW_CONFIDENCE" in missing.flags
    assert explicit_zero.subscores["valuation_plausibility"] == 10.0
    assert "VALUATION_PLAUSIBILITY_LOW_CONFIDENCE" not in explicit_zero.flags


def test_market_cap_band_penalty_is_deterministic():
    metrics = {
        "ttm_revenue": 8_000_000_000.0,
        "gross_margin": 0.42,
        "operating_margin": 0.13,
        "fcf": 500_000_000.0,
        "shares_outstanding": 100_000_000.0,
        "revenue_acceleration": 0.04,
        "gross_margin_change_qoq": 0.01,
        "operating_margin_change_qoq": 0.01,
        "fcf_change_qoq": 50_000_000.0,
        "liquidity_stress_score": 2,
        "net_debt": -100_000_000.0,
        "cfo_margin": 0.15,
        "fcf_margin": 0.08,
        "cfo_to_net_income": 1.1,
        "shares_change_4q": 0.01,
    }
    in_band = score_discovery_candidate(
        metrics=metrics,
        market_cap=20_000_000_000.0,
        market_cap_in_band=True,
        price=40.0,
        shares_outstanding=100_000_000.0,
    )
    out_band = score_discovery_candidate(
        metrics=metrics,
        market_cap=80_000_000_000.0,
        market_cap_in_band=False,
        price=40.0,
        shares_outstanding=100_000_000.0,
    )
    assert in_band.subscores["cap_band"] > out_band.subscores["cap_band"]
    assert in_band.total_score > out_band.total_score
    assert out_band.subscores["cap_band"] <= -10.0


def test_stage_gate_advances_only_when_conditions_met():
    stage, _ = determine_discovery_stage(
        total_score=72.0,
        whale_fit_score=19.0,
        evidence_strength_score=6.0,
        market_cap=12_000_000_000.0,
        market_cap_in_band=True,
        critical_unknown_count=0,
        suppressed=False,
    )
    assert stage == "ADVANCE_TO_DEEP"

    stage_unknown_low, _ = determine_discovery_stage(
        total_score=72.0,
        whale_fit_score=14.0,
        evidence_strength_score=6.0,
        market_cap="UNKNOWN",
        market_cap_in_band=False,
        critical_unknown_count=0,
        suppressed=False,
    )
    assert stage_unknown_low != "ADVANCE_TO_DEEP"


def test_discovery_candidate_schema_enforces_numeric_claim_trace():
    base_payload = {
        "ticker": "AAPL",
        "cik": "320193",
        "company_name": "Apple Inc.",
        "run_id": "run_test",
        "run_as_of_date": "2026-02-13",
        "effective_as_of_date": "2026-01-30",
        "market_cap": 20_000_000_000.0,
        "discovery_score": 75.0,
        "subscores_json": {},
        "key_reasons": ["Positive gross margin"],
        "flags": [],
        "recommended_action": "ADD_TO_UNIVERSE",
        "suggested_next_pipeline": "FULL_RESEARCH",
        "artifacts": {"evidence_packet_path": None, "filing_accessions_used": []},
    }
    with pytest.raises(ValidationError, match="numeric claim requires"):
        DiscoveryCandidate(
            **base_payload,
            numeric_claims=[
                {
                    "claim_id": "c1",
                    "label": "ttm_revenue",
                    "value": 1.0,
                    "unit": "USD",
                    "citations": [],
                    "derived_from": [],
                }
            ],
        )

    valid = DiscoveryCandidate(
        **base_payload,
        numeric_claims=[
            {
                "claim_id": "c1",
                "label": "ttm_revenue",
                "value": 1.0,
                "unit": "USD",
                "citations": [],
                "derived_from": ["financials.revenue"],
            }
        ],
    )
    assert valid.numeric_claims[0].label == "ttm_revenue"


def test_discovery_report_includes_tickers(tmp_path):
    candidate = DiscoveryCandidate(
        ticker="AAPL",
        cik="320193",
        company_name="Apple Inc.",
        run_id="discovery_test",
        run_as_of_date="2026-02-13",
        effective_as_of_date="2026-01-30",
        market_cap=10_000_000_000.0,
        discovery_score=70.0,
        stage="ADVANCE_TO_DEEP",
        whale_fit_score=19.0,
        evidence_strength_score=6.0,
        explainability_score=4.0,
        subscores_json={},
        key_reasons=["Positive gross margin"],
        gaps=["MARKET_CAP_UNKNOWN"],
        reason_evidence=[],
        numeric_claims=[
            {
                "claim_id": "c1",
                "label": "ttm_revenue",
                "value": 1.0,
                "citations": [],
                "derived_from": ["discovery.metrics.ttm_revenue"],
            }
        ],
        flags=[],
        newly_surfaced=True,
        repeat_surfaced=False,
        recommended_action="ADD_TO_UNIVERSE",
        suggested_next_pipeline="FULL_RESEARCH",
        artifacts={"evidence_packet_path": None, "filing_accessions_used": []},
    )
    report_path = _write_report_md(
        tmp_path,
        "discovery_test",
        "2026-02-13",
        [candidate],
        {
            "tickers_processed": 1,
            "missing_cik_count": 0,
            "filtered_market_cap_count": 0,
            "unknown_market_cap_count": 0,
            "shortlisted_count": 1,
            "suppressed_counts": {},
            "seed_hash": "abc",
        },
    )
    text = report_path.read_text(encoding="utf-8")
    assert "AAPL" in text
    assert (
        "| Ticker | MktCap | Score | Stage | WhaleFit | Evidence | Key reasons (3 bullets) | Gaps |"
        in text
    )
    assert "## Top Advances" in text


def test_discovery_reason_trace_rules():
    candidate = DiscoveryCandidate(
        ticker="MSFT",
        cik="789019",
        company_name="Microsoft",
        run_id="discovery_test",
        run_as_of_date="2026-02-13",
        effective_as_of_date="2026-01-28",
        market_cap=12_000_000_000.0,
        discovery_score=72.0,
        stage="ADVANCE_TO_DEEP",
        whale_fit_score=20.0,
        evidence_strength_score=7.0,
        explainability_score=4.0,
        subscores_json={},
        key_reasons=["Revenue growth acceleration"],
        reason_evidence=[],
        numeric_claims=[
            {
                "claim_id": "m1",
                "label": "ttm_revenue",
                "value": 100.0,
                "citations": [],
                "derived_from": ["discovery.metrics.ttm_revenue"],
            },
            {
                "claim_id": "m2",
                "label": "market_cap",
                "value": 12_000_000_000.0,
                "citations": [],
                "derived_from": ["market_cap.price_provider.stooq"],
            },
        ],
        flags=[],
        newly_surfaced=True,
        repeat_surfaced=False,
        recommended_action="ADD_TO_UNIVERSE",
        suggested_next_pipeline="FULL_RESEARCH",
        artifacts={"evidence_packet_path": None, "filing_accessions_used": []},
    )
    for claim in candidate.numeric_claims:
        if isinstance(claim.value, (int, float)):
            assert claim.citations or claim.derived_from


def test_reason_objects_include_derived_from_for_numeric_drivers():
    metrics = {
        "ttm_revenue": 7_000_000_000.0,
        "gross_margin": 0.45,
        "operating_margin": 0.16,
        "fcf": 650_000_000.0,
        "shares_outstanding": 190_000_000.0,
        "revenue_acceleration": 0.05,
        "gross_margin_change_qoq": 0.02,
        "operating_margin_change_qoq": 0.015,
        "fcf_change_qoq": 55_000_000.0,
        "liquidity_stress_score": 2,
        "net_debt": -300_000_000.0,
        "cfo_margin": 0.17,
        "fcf_margin": 0.09,
        "cfo_to_net_income": 1.2,
        "shares_change_4q": 0.01,
    }
    result = score_discovery_candidate(
        metrics=metrics,
        market_cap=15_000_000_000.0,
        market_cap_in_band=True,
        price=55.0,
        shares_outstanding=190_000_000.0,
    )
    assert result.reason_objects
    for reason in result.reason_objects:
        assert reason["derived_from"]


def _fake_selection() -> DiscoveryFilingSelection:
    stub = FilingStub(
        cik="1000",
        accession="0001000-26-000001",
        accession_nodash="000100026000001",
        form_type="10-Q",
        filing_date=date.fromisoformat("2026-01-30"),
        period_end="2025-12-31",
        primary_document="doc.htm",
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/1000/000100026000001/doc.htm",
        filing_index_url="https://www.sec.gov/Archives/edgar/data/1000/000100026000001/index.json",
    )
    return DiscoveryFilingSelection(annual=None, quarters=[stub], event=None)


def _patch_discovery_for_unit(monkeypatch, *, bank_revenue: float = 1_000_000_000.0):
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda refresh_if_missing=True: {"BANK": "1000", "TECH": "2000"},
    )
    monkeypatch.setattr(
        "app.discovery.runner.select_discovery_filings_from_submissions",
        lambda *args, **kwargs: _fake_selection(),
    )
    monkeypatch.setattr(
        "app.discovery.runner._fetch_and_parse_minimal_filings",
        lambda conn, **kwargs: (["0001000-26-000001"], [1]),
    )
    monkeypatch.setattr("app.discovery.runner.parse_filing_by_id", lambda filing_id: True)
    monkeypatch.setattr(
        "app.discovery.runner._load_filing_text",
        lambda conn, ticker, accessions: (
            "bank holding company commercial banking deposits underwriting insurance insurer asset management commercial lending"
            if ticker == "BANK"
            else "software enterprise platform"
        ),
    )

    def _fake_metrics(conn, *, ticker: str, selected_accessions: list[str]):
        revenue = bank_revenue if ticker == "BANK" else 2_000_000_000.0
        return DiscoveryMetricsResult(
            effective_as_of_date="2026-01-30",
            metrics={
                "ttm_revenue": revenue,
                "gross_margin": 0.4,
                "operating_margin": 0.12,
                "cfo": 100.0,
                "capex": 20.0,
                "fcf": 80.0,
                "shares_outstanding": 200_000_000.0,
                "revenue_growth_recent": 0.1,
                "revenue_acceleration": 0.05,
                "gross_margin_change_qoq": 0.02,
                "operating_margin_change_qoq": 0.02,
                "fcf_change_qoq": 10.0,
                "liquidity_stress_score": 2,
                "net_debt": 100.0,
                "cfo_margin": 0.11,
                "fcf_margin": 0.08,
                "cfo_to_net_income": 1.05,
                "revenue_growth_yoy": 0.09,
                "revenue_scale_log": 9.0,
                "shares_change_4q": 0.01,
            },
            claims=[
                {
                    "claim_id": "ttm_revenue",
                    "label": "ttm_revenue",
                    "value": revenue,
                    "citations": [
                        {
                            "source_url": "https://www.sec.gov/doc",
                            "snippet": "revenue",
                            "section_label": None,
                        }
                    ],
                    "derived_from": ["discovery.metrics.ttm_revenue"],
                },
                {
                    "claim_id": "gross_margin",
                    "label": "gross_margin",
                    "value": 0.4,
                    "citations": [
                        {
                            "source_url": "https://www.sec.gov/doc",
                            "snippet": "gross profit",
                            "section_label": None,
                        }
                    ],
                    "derived_from": ["discovery.metrics.gross_margin"],
                },
                {
                    "claim_id": "operating_margin",
                    "label": "operating_margin",
                    "value": 0.12,
                    "citations": [
                        {
                            "source_url": "https://www.sec.gov/doc",
                            "snippet": "operating income",
                            "section_label": None,
                        }
                    ],
                    "derived_from": ["discovery.metrics.operating_margin"],
                },
                {
                    "claim_id": "fcf",
                    "label": "fcf",
                    "value": 80.0,
                    "citations": [
                        {
                            "source_url": "https://www.sec.gov/doc",
                            "snippet": "cash flow",
                            "section_label": None,
                        }
                    ],
                    "derived_from": ["discovery.metrics.fcf"],
                },
            ],
            flags=[],
            filing_accessions_used=selected_accessions,
        )

    monkeypatch.setattr("app.discovery.runner.compute_discovery_metrics", _fake_metrics)

    class _FakeSecClient:
        def __init__(self):
            self.http = type(
                "Http", (), {"metrics": staticmethod(lambda: {"throttled_count": 0})}
            )()

        def submissions(self, cik: str):
            return {"name": "Bank Corp" if cik == "1000" else "Tech Corp"}

    monkeypatch.setattr("app.discovery.runner.SecClient", _FakeSecClient)


def test_discovery_suppression_flags(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_DISCOVERY_SUPPRESS_FINANCIALS", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    _patch_discovery_for_unit(monkeypatch)

    seed = tmp_path / "seed.csv"
    seed.write_text("ticker\nBANK\nTECH\n", encoding="utf-8")
    summary = run_discovery(
        as_of_date="2026-02-13",
        tickers=["BANK", "TECH"],
        seed_path=seed,
        limit=2,
        top_k=2,
    )
    shortlist = json.loads(Path(summary["discovery_shortlist_path"]).read_text(encoding="utf-8"))
    assert shortlist["candidate_count"] == 1
    assert shortlist["tickers"] == ["TECH"]

    stats = json.loads(Path(summary["discovery_stats_path"]).read_text(encoding="utf-8"))
    assert stats["suppressed_counts"]["SKIPPED_SUPPRESSION:FINANCIALS"] >= 1


def test_discovery_lifecycle_updates(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _patch_discovery_for_unit(monkeypatch)
    seed = tmp_path / "seed.csv"
    seed.write_text("ticker\nTECH\n", encoding="utf-8")

    summary_1 = run_discovery(
        as_of_date="2026-02-13",
        tickers=["TECH"],
        seed_path=seed,
        limit=1,
        top_k=1,
    )
    summary_2 = run_discovery(
        as_of_date="2026-02-13",
        tickers=["TECH"],
        seed_path=seed,
        limit=1,
        top_k=1,
    )

    c1 = json.loads(Path(summary_1["discovery_candidates_path"]).read_text(encoding="utf-8"))[
        "candidates"
    ][0]
    c2 = json.loads(Path(summary_2["discovery_candidates_path"]).read_text(encoding="utf-8"))[
        "candidates"
    ][0]
    assert c1["newly_surfaced"] is True
    assert c1["repeat_surfaced"] is False
    assert c2["newly_surfaced"] is False
    assert c2["repeat_surfaced"] is True

    with get_db() as conn:
        row = conn.execute(
            "SELECT times_shortlisted, last_action FROM discovery_lifecycle WHERE ticker = 'TECH'"
        ).fetchone()
    assert int(row["times_shortlisted"]) == 2
    assert row["last_action"] in {"ADD_TO_UNIVERSE", "WATCHLIST_ONLY", "SKIP"}


def test_run_all_with_discovery_handoff_uses_top_k(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    manifest = cfg.outputs_dir / "manifests" / "run_manifest_test.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"run_id": "run_test"}), encoding="utf-8")

    processed: list[str] = []
    finalize_seen: dict[str, object] = {}

    monkeypatch.setattr(
        "app.discovery.runner.run_discovery",
        lambda **kwargs: {
            "run_id": "discovery_test",
            "shortlist_tickers": ["AAPL", "MSFT", "GOOGL"],
            "discovery_candidates_path": str(cfg.discovery_dir / "dummy.json"),
        },
    )
    monkeypatch.setattr("app.ingest.filings.ingest_with_policy", lambda **kwargs: {"ok": True})
    monkeypatch.setattr("app.parse.filing_parser.parse_pending_filings", lambda **kwargs: 0)
    monkeypatch.setattr(
        "app.fundamentals.metrics.compute_fundamentals_for_ticker", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        "app.valuation.sanity_checks.run_valuation_for_ticker", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        "app.evidence.packet_builder.build_packet_for_ticker", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.agent.analyst_agent.run_analyst_agent_for_ticker", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.agent.analyst_agent.run_red_team_for_ticker", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.score.ranker.score_ticker",
        lambda ticker, run_id=None: processed.append(ticker) or True,
    )
    monkeypatch.setattr(
        "app.score.ranker.score_and_rank",
        lambda **kwargs: {
            "score_rows_considered": len(processed),
            "tickers_ranked": len(processed),
            "candidate_count": len(processed),
            "publishable_count": 0,
        },
    )
    monkeypatch.setattr("app.report.memo_builder.build_top_memos", lambda **kwargs: 0)
    monkeypatch.setattr(
        "app.report.memo_builder.build_memo_for_ticker", lambda *args, **kwargs: True
    )
    monkeypatch.setattr("app.agent.scheduler.write_agent_status", lambda: None)
    monkeypatch.setattr("app.report.run_manifest.write_run_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr("app.agent.queue.dead_letter_count", lambda: 0)

    def _fake_finalize_run_outputs(**kwargs):
        finalize_seen["tickers_targeted"] = kwargs["tickers_targeted"]
        return {
            "run_id": kwargs["run_id"],
            "run_dir": str(tmp_path / "run"),
            "gating_report_json": "x",
            "gating_report_csv": "y",
            "memos_count": 0,
            "candidates_count": len(kwargs["tickers_targeted"]),
            "dead_letter_delta": 0,
            "tickers_targeted": kwargs["tickers_targeted"],
            "tickers_processed": kwargs["tickers_targeted"],
        }

    monkeypatch.setattr("app.ops.runs.finalize_run_outputs", _fake_finalize_run_outputs)

    summary = _run_all_impl(
        as_of="2026-02-13",
        with_research=False,
        with_synthesis=False,
        with_discovery=True,
        dossier_top=None,
        limit=None,
        tickers="BASE1,BASE2",
        discovery_top=2,
        discovery_seed=None,
        phase=["ingest", "valuation"],
        forms="10-K,10-Q,8-K",
        memo_mode="strict",
        top=5,
    )

    assert processed == ["AAPL", "MSFT"]
    assert finalize_seen["tickers_targeted"] == ["AAPL", "MSFT"]
    assert summary["discovery_shortlist_count"] == 3


def test_run_all_with_discovery_triggers_dossier_top(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    manifest = cfg.outputs_dir / "manifests" / "run_manifest_test.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"run_id": "run_test"}), encoding="utf-8")

    monkeypatch.setattr(
        "app.discovery.runner.run_discovery",
        lambda **kwargs: {
            "run_id": "discovery_test",
            "shortlist_tickers": ["AAPL", "MSFT", "GOOGL"],
            "discovery_candidates_path": str(cfg.discovery_dir / "dummy.json"),
        },
    )
    monkeypatch.setattr("app.ingest.filings.ingest_with_policy", lambda **kwargs: {"ok": True})
    monkeypatch.setattr("app.parse.filing_parser.parse_pending_filings", lambda **kwargs: 0)
    monkeypatch.setattr(
        "app.fundamentals.metrics.compute_fundamentals_for_ticker", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        "app.valuation.sanity_checks.run_valuation_for_ticker", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        "app.evidence.packet_builder.build_packet_for_ticker", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.agent.analyst_agent.run_analyst_agent_for_ticker", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.agent.analyst_agent.run_red_team_for_ticker", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("app.score.ranker.score_ticker", lambda ticker, run_id=None: True)
    monkeypatch.setattr(
        "app.score.ranker.score_and_rank",
        lambda **kwargs: {
            "score_rows_considered": 2,
            "tickers_ranked": 2,
            "candidate_count": 2,
            "publishable_count": 0,
        },
    )
    monkeypatch.setattr("app.report.memo_builder.build_top_memos", lambda **kwargs: 0)
    monkeypatch.setattr(
        "app.report.memo_builder.build_memo_for_ticker", lambda *args, **kwargs: True
    )
    monkeypatch.setattr("app.agent.scheduler.write_agent_status", lambda: None)
    monkeypatch.setattr("app.report.run_manifest.write_run_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr("app.agent.queue.dead_letter_count", lambda: 0)
    monkeypatch.setattr(
        "app.ops.runs.finalize_run_outputs",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "run_dir": str(tmp_path / "run"),
            "gating_report_json": "x",
            "gating_report_csv": "y",
            "memos_count": 0,
            "candidates_count": 2,
            "dead_letter_delta": 0,
            "tickers_targeted": kwargs["tickers_targeted"],
            "tickers_processed": kwargs["tickers_targeted"],
        },
    )
    seen: dict[str, Any] = {}

    def _fake_dossier(**kwargs):
        seen.update(kwargs)
        return {"run_id": kwargs["run_id"], "tickers_built": kwargs["tickers"][:]}

    monkeypatch.setattr("app.dossier.runner.run_dossier_for_peer_set", _fake_dossier)

    summary = _run_all_impl(
        as_of="2026-02-13",
        with_research=False,
        with_synthesis=False,
        with_discovery=True,
        dossier_top=2,
        limit=None,
        tickers="AAPL,MSFT",
        discovery_top=2,
        discovery_seed=None,
        phase=["ingest", "valuation"],
        forms="10-K,10-Q,8-K",
        memo_mode="strict",
        top=5,
    )
    assert summary["dossier"]["tickers_built"] == ["AAPL", "MSFT"]
    assert seen["tickers"] == ["AAPL", "MSFT"]
