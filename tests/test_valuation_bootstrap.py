from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.autonomous.sector_financial_packets import (
    build_sector_company_financial_packets_from_signal_packets,
)
from app.autonomous.v1_financial_context import build_canonical_v1_financial_context
from app.config import get_config
from app.db import get_db, init_db
from app.market.split_evidence import produce_split_lineage_evidence
from app.valuation.bootstrap import (
    VALUATION_BOOTSTRAP_RUN_PREFIX,
    bootstrap_relight_valuations,
)
from app.valuation.lineage import latest_decision_eligible_valuation_row
from app.watchlist.relight import RELIGHT_SCHEMA_VERSION, state_path
from tests.test_classic_postwrite_authorization import _baseline_manifest
from tests.test_split_evidence import _fixture_fetcher, _seed_primary_security
from tests.test_valuation_writer import _seed_companyfacts


def test_relight_valuation_bootstrap_publishes_and_binds_without_provider(
    monkeypatch,
    tmp_path,
):
    _baseline_manifest(monkeypatch, tmp_path)
    init_db()
    _seed_primary_security(monkeypatch, tmp_path / "data")
    cfg = get_config()
    with get_db() as conn:
        _seed_companyfacts(conn, ticker="LINE")
        # The shared fixture carries one fiscal-year count of 10.0 while the split
        # fixture below files a cover-page count of 100.0: a lone fiscal-year count
        # contradicted ten-fold by an independent count is exactly the case
        # select_stable_shares now refuses (an old stored valuation was once
        # divided by a seven-year-old count the same way). The
        # two counts agree here, and the anchor below is the same arithmetic on
        # 100.0 shares instead of 10.0.
        conn.execute(
            """
            UPDATE companyfacts_facts
            SET value = 100.0
            WHERE ticker = 'LINE' AND line_item = 'shares_outstanding'
            """
        )
        conn.execute(
            """
            UPDATE companyfacts_facts
            SET period_type = COALESCE(period_type, 'FY'),
                source_url =
                    'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                form = COALESCE(form, '10-K')
            WHERE ticker = 'LINE'
            """
        )
        conn.execute(
            """
            DELETE FROM companyfacts_facts
            WHERE ticker = 'LINE' AND line_item = 'preferred_equity'
            """
        )
    companyfacts_cache = cfg.cache_dir / "companyfacts" / "0000000001.json"
    companyfacts_cache.parent.mkdir(parents=True, exist_ok=True)
    companyfacts_cache.write_text(
        json.dumps(
            {
                "cik": "0000000001",
                "retrieved_at": "2026-07-28T12:00:00Z",
                "source_url": ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"),
                "companyfacts": {
                    "cik": 1,
                    "facts": {
                        "us-gaap": {
                            "StockholdersEquity": {
                                "units": {
                                    "USD": [
                                        {
                                            "end": "2024-12-31",
                                            "val": 200_000_000.0,
                                            "accn": "LINE-2024",
                                            "fy": 2024,
                                            "fp": "FY",
                                            "form": "10-K",
                                            "filed": "2025-02-15",
                                        }
                                    ]
                                }
                            }
                        }
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fetcher, calls = _fixture_fetcher(split_payload=[])
    assert (
        produce_split_lineage_evidence(
            ticker="LINE",
            as_of_date="2026-07-28",
            db_path=cfg.db_path,
            cfg=cfg,
            fetch_bytes=fetcher,
            retrieved_on="2026-07-28",
        )["status"]
        == "READY"
    )
    assert len(calls) == 2
    monkeypatch.setattr(
        "app.valuation.valuation_writer.get_market_price_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bootstrap must not construct a live price provider")
        ),
    )

    relight_id = "relight_20260728T120000Z_1234abcd"
    checkpoint = state_path(relight_id)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": RELIGHT_SCHEMA_VERSION,
                "relight_id": relight_id,
                "effective_as_of": "2026-07-28",
                "routed_tickers": ["LINE"],
                "cells": [
                    {
                        "cell_id": "industrials:mid_cap",
                        "sector": "industrials",
                        "band": "mid_cap",
                        "tickers": ["LINE"],
                        "status": "PENDING",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checkpoint_before = checkpoint.read_bytes()

    first = bootstrap_relight_valuations(relight_id=relight_id)
    assert checkpoint.read_bytes() == checkpoint_before
    assert first["requested"] == 1
    assert first["split_ready"] == 1
    assert first["split_unknown"] == 0
    assert first["PUBLISHED"] == 1, first["results"]
    assert first["ALREADY_AUTHORIZED"] == 0
    assert first["NEEDS_DATA"] == 0
    assert first["FAILED"] == 0
    assert first["provider_calls"] == 0
    assert first["llm_cost_usd"] == 0.0
    published = first["results"][0]
    assert published["ticker"] == "LINE"
    assert published["status"] == "PUBLISHED"
    assert published["run_id"].startswith(f"{VALUATION_BOOTSTRAP_RUN_PREFIX}_industrials_line_")
    assert published["bound_methods"] == [
        "capital_structure",
        "dcf",
        "epv",
        "ev_ebit",
        "fcf_yield",
        "graham",
        "ncav",
        "owner_earnings",
        "reverse_dcf",
        "roic",
        "scorecard",
        "tangible_floor",
    ]

    with get_db() as conn:
        assert {
            str(row[1]) for row in conn.execute("PRAGMA table_info(valuations)").fetchall()
        } >= {
            "source_run_id",
            "source_artifact_path",
            "source_artifact_sha256",
        }
        scorecard = latest_decision_eligible_valuation_row(
            conn,
            ticker="LINE",
            method="scorecard",
            as_of_date="2026-07-28",
            exact_as_of_date=True,
        )
    assert scorecard is not None
    assert scorecard["source_run_id"] == published["run_id"]
    assert scorecard["source_artifact_path"] == published["artifact_path"]
    assert scorecard["source_artifact_sha256"] == hashlib.sha256(
        Path(published["artifact_path"]).read_bytes()
    ).hexdigest()
    scorecard_inputs = json.loads(scorecard["inputs_json"])
    assert scorecard_inputs["evidenced_zero_facts"][0]["line_item"] == "preferred_equity"
    assert (
        scorecard_inputs["evidenced_zero_facts"][0]["derivation"]
        == "EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE"
    )

    context = build_canonical_v1_financial_context(
        tickers=["LINE"],
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
    )
    packet = build_sector_company_financial_packets_from_signal_packets(
        context.packets,
        sector="industrials",
        as_of_date="2026-07-28",
        cap_classifications=context.issuer_contexts,
        pipeline_version="v1",
    )[0]
    assert packet.valuation["available_methods"] == ["dcf", "epv", "graham"]
    assert packet.valuation["anchor_method"] == "dcf"
    # 174.94093041889468 on 10.0 shares; the same equity value over 100.0 shares.
    assert packet.valuation["valuation_anchor"] == 17.494093041889467

    run_dirs_before = sorted(cfg.runs_dir.glob("autonomous_sector/*"))
    second = bootstrap_relight_valuations(relight_id=relight_id)
    assert checkpoint.read_bytes() == checkpoint_before
    assert second["PUBLISHED"] == 0
    assert second["ALREADY_AUTHORIZED"] == 1
    assert second["NEEDS_DATA"] == 0
    assert second["FAILED"] == 0
    assert second["provider_calls"] == 0
    assert second["results"] == [
        {
            "ticker": "LINE",
            "status": "ALREADY_AUTHORIZED",
            "anchor_method": "dcf",
            "valuation_anchor": 17.494093041889467,
        }
    ]
    assert sorted(cfg.runs_dir.glob("autonomous_sector/*")) == run_dirs_before
