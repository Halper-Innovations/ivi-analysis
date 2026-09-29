from __future__ import annotations

import json

import pytest

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.db import init_db
from app.rlm.executor import _apply_depth_value_gate_policy_to_action
from app.rlm.planner import generate_plan
from app.rlm.schemas import Action
from app.rlm.state import init_loop_state
from app.universe.scout import run_universe_scout


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAA,1,AAA\nBBB,2,BBB\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_scout_price_status_ok_requires_current_price(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    facts_row = {
        "ticker": "AAA",
        "status": "OK",
        "shares_status": "OK",
        "shares_reason": "OK",
        "shares_value": 10.0,
        "cfo_status": "OK",
        "cfo_reason": "OK",
        "cfo_value": 30.0,
        "capex_status": "OK",
        "capex_reason": "OK",
        "capex_value": 10.0,
        "fcf_status": "OK",
        "fcf_reason": "OK",
        "fcf_value": 20.0,
        "fetch_reason_code": "CACHE_HIT",
        "cache_path": None,
        "derived_from": ["facts.AAA"],
    }
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(
                cfg.outputs_dir / "prices" / "scout_price_integrity" / "prices_summary.json"
            ),
            "rows": [{"ticker": "AAA", "status": "OK", "price": 12.0, "reason_code": "CACHE_HIT"}],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row)
    )
    monkeypatch.setattr(
        "app.valuation.owner_earnings.resolve_financial_facts_asof",
        lambda **_kwargs: dict(facts_row),
    )

    run_universe_scout(
        run_id="scout_price_integrity", as_of_date="2026-02-14", top_n=5, tickers=["AAA"]
    )

    coverage = json.loads(
        (cfg.sectors_dir / "scout_price_integrity" / "universe_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    row = coverage["rows"][0]
    assert row["price_status"] == "OK"
    assert row["current_price"] == 12.0
    assert row["shares_outstanding"] == 10.0
    assert row["market_cap"] == 120.0


def test_scout_invalid_ok_price_is_marked_partial(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(
                cfg.outputs_dir / "prices" / "scout_price_partial" / "prices_summary.json"
            ),
            "rows": [{"ticker": "AAA", "status": "OK", "price": None, "reason_code": "CACHE_HIT"}],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.resolve_financial_facts_asof",
        lambda **_kwargs: {
            "ticker": "AAA",
            "status": "UNKNOWN",
            "shares_status": "UNKNOWN",
            "shares_reason": "CIK_MISSING",
            "shares_value": "UNKNOWN",
            "cfo_status": "UNKNOWN",
            "cfo_reason": "CIK_MISSING",
            "cfo_value": "UNKNOWN",
            "capex_status": "UNKNOWN",
            "capex_reason": "CIK_MISSING",
            "capex_value": "UNKNOWN",
            "fcf_status": "UNKNOWN",
            "fcf_reason": "CIK_MISSING",
            "fcf_value": "UNKNOWN",
            "fetch_reason_code": "CIK_MISSING",
            "cache_path": None,
            "derived_from": ["facts.AAA"],
        },
    )

    run_universe_scout(
        run_id="scout_price_partial", as_of_date="2026-02-14", top_n=5, tickers=["AAA"]
    )

    coverage = json.loads(
        (cfg.sectors_dir / "scout_price_partial" / "universe_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    row = coverage["rows"][0]
    assert row["price_status"] == "PARTIAL"
    assert row["current_price"] == "UNKNOWN"
    assert row["price_reason_code"] == "INVALID_PRICE_VALUE"


def test_offline_depth_bootstrap_requires_financial_scope_before_fallback(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    state = init_loop_state(
        run_id="rlm_bootstrap_dossiers",
        sector="Software",
        as_of_date="2026-02-14",
        max_iterations=2,
        llm_budget_usd=1.0,
        sec_budget_count=100,
    )
    state.mode = "depth"
    state.rlm_version = "v1.1"
    state.iteration = 0
    state.peer_set = ["AAA", "BBB"]
    state.top_k_current = ["AAA", "BBB"]

    run_dir = cfg.sectors_dir / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    value_gates_path = run_dir / "value_gates.json"
    value_gates_path.write_text(
        json.dumps(
            {
                "entries": [
                    {"ticker": "AAA", "gate_status": "FAIL", "gate_reasons": ["PRICE_UNKNOWN"]},
                    {"ticker": "BBB", "gate_status": "FAIL", "gate_reasons": ["PRICE_UNKNOWN"]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state.artifacts["value_gates_path"] = str(value_gates_path)

    with pytest.raises(InvalidFinancialInputError):
        generate_plan(state=state, top_k=2, mode="depth")

    action = Action(action_type="BUILD_DOSSIERS", tickers=["AAA", "BBB"], limit=2, years_back=10)
    filtered, note = _apply_depth_value_gate_policy_to_action(
        state=state,
        action=action,
        effective_action="BUILD_DOSSIERS",
        top_k=2,
    )
    assert filtered is not None
    assert filtered.tickers == ["AAA", "BBB"]
    assert note == "bootstrap_dossier_build"
