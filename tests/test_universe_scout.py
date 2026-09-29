from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.universe.scout import (
    ScoutPhaseTimeoutError,
    open_universe_scout_status,
    open_net_debt_coverage,
    open_universe_scout,
    open_universe_scout_calibration,
    open_universe_yield_coverage,
    run_universe_scout_resume,
    run_universe_scout,
    run_universe_scout_to_depth,
)


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    universe = data_dir / "universe" / "universe.csv"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text("ticker,cik,name\nAAPL,1,AAPL\nMSFT,2,MSFT\nNVDA,3,NVDA\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_UNIVERSE_PATH", str(universe))
    monkeypatch.setenv("VOE_LLM_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _write_companyfacts_cache(
    path: Path,
    *,
    debt: float,
    cash: float,
    shares_values: list[tuple[str, float]],
    fcf_values: list[tuple[str, float]],
    cfo_values: list[tuple[str, float]] | None = None,
    capex_values: list[tuple[str, float]] | None = None,
    current_assets: float | None = None,
    total_liabilities: float | None = None,
    preferred_stock: float | None = None,
    scale: float = 1.0,
) -> None:
    """``scale`` multiplies the share counts and the cash flows. Companyfacts holds
    whole shares and whole dollars while the facts row a test pairs with it is in
    millions; ``scale=1_000_000`` writes the same company in both."""
    cfo_values = cfo_values or [("2025-12-31", 120.0)]
    capex_values = capex_values or [("2025-12-31", 40.0)]
    shares_values = [(end, value * scale) for end, value in shares_values]
    fcf_values = [(end, value * scale) for end, value in fcf_values]
    cfo_values = [(end, value * scale) for end, value in cfo_values]
    capex_values = [(end, value * scale) for end, value in capex_values]
    payload = {
        "companyfacts": {
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"end": end_date, "filed": f"{end_date}", "val": value}
                                for end_date, value in shares_values
                            ]
                        }
                    }
                },
                "us-gaap": {
                    "LongTermDebt": {"units": {"USD": [{"end": "2025-12-31", "filed": "2026-01-31", "val": debt}]}},
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {"USD": [{"end": "2025-12-31", "filed": "2026-01-31", "val": cash}]}
                    },
                    "FreeCashFlow": {
                        "units": {
                            "USD": [
                                {"end": end_date, "filed": f"{end_date}", "val": value}
                                for end_date, value in fcf_values
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {"end": end_date, "filed": f"{end_date}", "val": value}
                                for end_date, value in cfo_values
                            ]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {
                            "USD": [
                                {"end": end_date, "filed": f"{end_date}", "val": value}
                                for end_date, value in capex_values
                            ]
                        }
                    },
                    **(
                        {"AssetsCurrent": {"units": {"USD": [{"end": "2025-12-31", "filed": "2026-01-31", "val": current_assets}]}}}
                        if isinstance(current_assets, (int, float))
                        else {}
                    ),
                    **(
                        {"Liabilities": {"units": {"USD": [{"end": "2025-12-31", "filed": "2026-01-31", "val": total_liabilities}]}}}
                        if isinstance(total_liabilities, (int, float))
                        else {}
                    ),
                    **(
                        {"PreferredStockValue": {"units": {"USD": [{"end": "2025-12-31", "filed": "2026-01-31", "val": preferred_stock}]}}}
                        if isinstance(preferred_stock, (int, float))
                        else {}
                    ),
                },
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_universe_scout_deterministic_order_and_no_dossier(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    call_state = {"dossier_called": 0}

    def _forbid_dossier(*_args, **_kwargs):
        call_state["dossier_called"] += 1
        raise AssertionError("dossier stage must not run during universe scout")

    monkeypatch.setattr("app.dossier.runner.run_dossier_for_peer_set", _forbid_dossier)

    cache_dir = cfg.cache_dir / "companyfacts"
    aapl_cache = cache_dir / "0000000001.json"
    msft_cache = cache_dir / "0000000002.json"
    nvda_cache = cache_dir / "0000000003.json"
    _write_companyfacts_cache(
        aapl_cache,
        debt=100.0,
        cash=20.0,
        shares_values=[("2023-12-31", 9.0), ("2025-12-31", 10.0)],
        fcf_values=[("2023-12-31", 60.0), ("2024-12-31", 70.0), ("2025-12-31", 80.0)],
        scale=1_000_000.0,
    )
    _write_companyfacts_cache(
        msft_cache,
        debt=200.0,
        cash=20.0,
        shares_values=[("2025-12-31", 20.0)],
        fcf_values=[("2025-12-31", 10.0)],
        scale=1_000_000.0,
    )
    _write_companyfacts_cache(
        nvda_cache,
        debt=50.0,
        cash=5.0,
        shares_values=[("2024-12-31", 5.0), ("2025-12-31", 6.0)],
        fcf_values=[("2024-12-31", -2.0), ("2025-12-31", -3.0)],
        scale=1_000_000.0,
    )

    def _fake_write_prices_for_run(**_kwargs):
        return {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_det" / "prices_summary.json"),
            "rows": [
                {"ticker": "MSFT", "status": "OK", "price": 100.0, "reason_code": "CACHE_HIT"},
                {"ticker": "AAPL", "status": "OK", "price": 50.0, "reason_code": "CACHE_HIT"},
                {"ticker": "NVDA", "status": "OK", "price": 150.0, "reason_code": "CACHE_HIT"},
            ],
        }

    facts_map = {
        "AAPL": {
            "ticker": "AAPL",
            "status": "OK",
            "shares_status": "OK",
            "shares_reason": "OK",
            "shares_value": 10.0,
            "cfo_status": "OK",
            "cfo_reason": "OK",
            "cfo_value": 120.0,
            "capex_status": "OK",
            "capex_reason": "OK",
            "capex_value": 40.0,
            "fcf_status": "OK",
            "fcf_reason": "OK",
            "fcf_value": 80.0,
            "fetch_reason_code": "CACHE_HIT",
            "cache_path": str(aapl_cache),
            "derived_from": ["facts.AAPL"],
        },
        "MSFT": {
            "ticker": "MSFT",
            "status": "PARTIAL",
            "shares_status": "OK",
            "shares_reason": "OK",
            "shares_value": 20.0,
            "cfo_status": "OK",
            "cfo_reason": "OK",
            "cfo_value": 30.0,
            "capex_status": "UNKNOWN",
            "capex_reason": "TAG_MISS",
            "capex_value": "UNKNOWN",
            "fcf_status": "UNKNOWN",
            "fcf_reason": "TAG_MISS",
            "fcf_value": "UNKNOWN",
            "fetch_reason_code": "CACHE_HIT",
            "cache_path": str(msft_cache),
            "derived_from": ["facts.MSFT"],
        },
        "NVDA": {
            "ticker": "NVDA",
            "status": "OK",
            "shares_status": "OK",
            "shares_reason": "OK",
            "shares_value": 6.0,
            "cfo_status": "OK",
            "cfo_reason": "OK",
            "cfo_value": -10.0,
            "capex_status": "OK",
            "capex_reason": "OK",
            "capex_value": 1.0,
            "fcf_status": "OK",
            "fcf_reason": "OK",
            "fcf_value": -3.0,
            "fetch_reason_code": "CACHE_HIT",
            "cache_path": str(nvda_cache),
            "derived_from": ["facts.NVDA"],
        },
    }

    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _fake_write_prices_for_run)
    monkeypatch.setattr(
        "app.universe.scout.resolve_financial_facts_asof",
        lambda **kwargs: facts_map[str(kwargs["ticker"]).upper()],
    )

    summary = run_universe_scout(
        run_id="scout_det",
        as_of_date="2026-02-14",
        top_n=2,
        tickers=["MSFT", "AAPL", "NVDA", "AAPL"],
    )
    assert summary["status"] == "OK"
    scoreboard = json.loads((cfg.sectors_dir / "scout_det" / "universe_scoreboard.json").read_text(encoding="utf-8"))
    tickers = [str(row.get("ticker") or "") for row in scoreboard["rows"]]
    assert tickers == sorted(tickers, key=lambda ticker: (-next(r["score_total"] for r in scoreboard["rows"] if r["ticker"] == ticker), ticker))
    shortlist = json.loads((cfg.sectors_dir / "scout_det" / "universe_shortlist.json").read_text(encoding="utf-8"))
    assert shortlist["counts"]["PASS"] >= 1
    assert shortlist["top_candidates"][0]["ticker"] == "AAPL"
    calibration = json.loads(
        (cfg.sectors_dir / "scout_det" / "universe_scout_calibration.json").read_text(encoding="utf-8")
    )
    assert calibration["counts"]["PASS"] >= 1
    assert "thresholds_effective" in calibration
    assert "rows" in calibration
    aapl_row = next(row for row in scoreboard["rows"] if row["ticker"] == "AAPL")
    assert aapl_row["scout_status"] == "PASS"
    assert aapl_row["primary_blocker_category"] == "NONE"
    assert call_state["dossier_called"] == 0


def test_universe_scout_coverage_reason_code_stability(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_cov" / "prices_summary.json"),
            "rows": [{"ticker": "AAA", "status": "MISSING", "price": None, "reason_code": "OFFLINE_NO_CACHE"}],
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
    run_universe_scout(run_id="scout_cov", as_of_date="2026-02-14", top_n=5, tickers=["AAA"])
    coverage = json.loads((cfg.sectors_dir / "scout_cov" / "universe_coverage.json").read_text(encoding="utf-8"))
    assert coverage["rows"][0]["price_reason_code"] == "OFFLINE_NO_CACHE"
    assert coverage["rows"][0]["shares_reason_code"] == "CIK_MISSING"
    assert coverage["rows"][0]["primary_blocker_category"] == "MISSING_PRICE"
    assert coverage["unknown_reason_counts"]["OFFLINE_NO_CACHE"] >= 1
    assert coverage["unknown_reason_counts"]["CIK_MISSING"] >= 1


def test_universe_scout_to_depth_wiring_and_linkage(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    scout_run_id = "scout_link"
    depth_run_id = "depth_from_scout"
    scout_dir = cfg.sectors_dir / scout_run_id
    scout_dir.mkdir(parents=True, exist_ok=True)
    shortlist_path = scout_dir / "universe_shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            {
                "run_id": scout_run_id,
                "as_of_date": "2026-02-14",
                "top_candidates": [
                    {"ticker": "MSFT", "scout_status": "PASS", "score_total": 88.0, "derived_from": ["a"]},
                    {"ticker": "AAPL", "scout_status": "WATCH", "score_total": 70.0, "derived_from": ["b"]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_run_sector_rlm_loop(**kwargs):
        captured.update(kwargs)
        depth_dir = cfg.sectors_dir / kwargs["run_id"]
        depth_dir.mkdir(parents=True, exist_ok=True)
        return {"run_id": kwargs["run_id"], "status": "DONE", "artifacts": {}}

    monkeypatch.setattr("app.rlm.loop.run_sector_rlm_loop", _fake_run_sector_rlm_loop)
    payload = run_universe_scout_to_depth(
        scout_run_id=scout_run_id,
        depth_run_id=depth_run_id,
        sector="Software",
        iterations=2,
        top_k=5,
    )
    assert payload["status"] == "OK"
    assert captured["seed_tickers"] == ["MSFT", "AAPL"]
    assert captured["parent_run_id"] == scout_run_id
    assert str(captured["shortlist_source"]).endswith("universe_shortlist.json")
    linkage = json.loads((cfg.sectors_dir / depth_run_id / "universe_scout_linkage.json").read_text(encoding="utf-8"))
    assert linkage["scout_run_id"] == scout_run_id
    assert linkage["selected_tickers"] == ["MSFT", "AAPL"]
    assert any(row["derived_from"] for row in linkage["selected_candidates"])


def test_scout_primary_blocker_category_deterministic_with_mixed_missing(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_blockers" / "prices_summary.json"),
            "rows": [{"ticker": "AAA", "status": "MISSING", "price": None, "reason_code": "OFFLINE_NO_CACHE"}],
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
    run_universe_scout(run_id="scout_blockers", as_of_date="2026-02-14", top_n=5, tickers=["AAA"])
    calibration = json.loads(
        (cfg.sectors_dir / "scout_blockers" / "universe_scout_calibration.json").read_text(encoding="utf-8")
    )
    row = calibration["rows"][0]
    assert row["primary_blocker_category"] == "MISSING_PRICE"
    assert calibration["calibration_required"]["note_code"] == "CALIBRATION_REQUIRED"


def test_scout_near_miss_and_threshold_override_persistence(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_dir = cfg.cache_dir / "companyfacts"
    xyz_cache = cache_dir / "0000000010.json"
    _write_companyfacts_cache(
        xyz_cache,
        debt=50.0,
        cash=0.0,
        shares_values=[("2023-12-31", 9.8), ("2025-12-31", 10.0)],
        fcf_values=[("2023-12-31", 50.0), ("2024-12-31", 55.0), ("2025-12-31", 60.0)],
        scale=1_000_000.0,
    )
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_near" / "prices_summary.json"),
            "rows": [{"ticker": "XYZ", "status": "OK", "price": 60.0, "reason_code": "CACHE_HIT"}],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.resolve_financial_facts_asof",
        lambda **_kwargs: {
            "ticker": "XYZ",
            "status": "OK",
            "shares_status": "OK",
            "shares_reason": "OK",
            "shares_value": 10.0,
            "cfo_status": "OK",
            "cfo_reason": "OK",
            "cfo_value": 120.0,
            "capex_status": "OK",
            "capex_reason": "OK",
            "capex_value": 40.0,
            "fcf_status": "OK",
            "fcf_reason": "OK",
            "fcf_value": 60.0,
            "fetch_reason_code": "CACHE_HIT",
            "cache_path": str(xyz_cache),
            "derived_from": ["facts.XYZ"],
        },
    )
    base = run_universe_scout(
        run_id="scout_near_base",
        as_of_date="2026-02-14",
        top_n=5,
        tickers=["XYZ"],
        threshold_overrides={"scout_use_graham_dodd": False},
    )
    base_scoreboard = json.loads(
        (cfg.sectors_dir / "scout_near_base" / "universe_scoreboard.json").read_text(encoding="utf-8")
    )
    base_row = base_scoreboard["rows"][0]
    assert base_row["scout_status"] == "WATCH"
    assert base["thresholds_effective"]["scout_mos_min"] == 0.30
    assert base["calibration_required"]["required"] is True
    assert base["calibration_required"]["note_code"] == "CALIBRATION_REQUIRED"
    assert any(field["field"] == "scout_mos_min" for field in base_row["near_miss_fields"])
    opened_cal = open_universe_scout_calibration(run_id="scout_near_base")
    assert opened_cal["status"] == "OK"
    assert opened_cal["calibration_required"]["note_code"] == "CALIBRATION_REQUIRED"

    override = run_universe_scout(
        run_id="scout_near_override",
        as_of_date="2026-02-14",
        top_n=5,
        tickers=["XYZ"],
        threshold_overrides={"scout_mos_min": 0.10, "scout_use_graham_dodd": False},
    )
    override_scoreboard = json.loads(
        (cfg.sectors_dir / "scout_near_override" / "universe_scoreboard.json").read_text(encoding="utf-8")
    )
    override_row = override_scoreboard["rows"][0]
    assert override["thresholds_effective"]["scout_mos_min"] == 0.10
    assert override_row["scout_status"] == "PASS"


def test_scout_integration_changes_blocker_categories_deterministically(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_dir = cfg.cache_dir / "companyfacts"
    gd_cache = cache_dir / "0000000041.json"
    _write_companyfacts_cache(
        gd_cache,
        debt=10.0,
        cash=5.0,
        shares_values=[("2023-12-31", 10.0), ("2025-12-31", 10.0)],
        fcf_values=[("2023-12-31", 40.0), ("2024-12-31", 40.0), ("2025-12-31", 40.0)],
        cfo_values=[("2023-12-31", 80.0), ("2024-12-31", 80.0), ("2025-12-31", 80.0)],
        capex_values=[("2023-12-31", 40.0), ("2024-12-31", 40.0), ("2025-12-31", 40.0)],
        current_assets=300.0,
        total_liabilities=250.0,
        preferred_stock=0.0,
    )
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_gd_blockers" / "prices_summary.json"),
            "rows": [{"ticker": "GDB", "status": "OK", "price": 100.0, "reason_code": "CACHE_HIT"}],
        },
    )
    facts_row = {
        "ticker": "GDB",
        "status": "OK",
        "shares_status": "OK",
        "shares_reason": "OK",
        "shares_value": 10.0,
        "cfo_status": "OK",
        "cfo_reason": "OK",
        "cfo_value": 80.0,
        "capex_status": "OK",
        "capex_reason": "OK",
        "capex_value": 40.0,
        "fcf_status": "OK",
        "fcf_reason": "OK",
        "fcf_value": 40.0,
        "fetch_reason_code": "CACHE_HIT",
        "cache_path": str(gd_cache),
        "derived_from": ["facts.GDB"],
    }
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))
    monkeypatch.setattr("app.valuation.owner_earnings.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))

    run_universe_scout(run_id="scout_gd_blockers", as_of_date="2026-02-14", top_n=5, tickers=["GDB"])
    scoreboard = json.loads(
        (cfg.sectors_dir / "scout_gd_blockers" / "universe_scoreboard.json").read_text(encoding="utf-8")
    )
    row = scoreboard["rows"][0]
    assert row["scout_status"] == "FAIL"
    assert row["primary_blocker_category"] in {"INSUFFICIENT_MOS_EPV", "INSUFFICIENT_MOS_NETNET"}
    assert "INSUFFICIENT_MOS_EPV" in set(row["blocker_categories"])
    assert row["gd_value_status"] == "OK"
    assert row["epv_per_share"] != "UNKNOWN"
    assert row["mos_epv"] != "UNKNOWN"


def test_scout_owner_earnings_normalized_yield_can_upgrade_outcome(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_dir = cfg.cache_dir / "companyfacts"
    oey_cache = cache_dir / "0000000042.json"
    _write_companyfacts_cache(
        oey_cache,
        # Raw-USD companyfacts values: $0 debt, $2000M cash. net_debt._to_musd divides
        # by 1e6 -> debt 0.0, cash 2000.0 ($M), net_debt -2000 -> negative EV, so the
        # scout falls back to the market-cap owner-earnings yield (OWNER_EARNINGS_YIELD_3Y).
        debt=0.0,
        cash=2_000_000_000.0,
        shares_values=[("2023-12-31", 10.0), ("2024-12-31", 10.0), ("2025-12-31", 10.0)],
        fcf_values=[("2023-12-31", 10.0), ("2024-12-31", 5.0), ("2025-12-31", 4.0)],
        cfo_values=[("2023-12-31", 200.0), ("2024-12-31", 210.0), ("2025-12-31", 220.0)],
        capex_values=[("2023-12-31", 60.0), ("2024-12-31", 60.0), ("2025-12-31", 60.0)],
        scale=1_000_000.0,
    )
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_owner_yield" / "prices_summary.json"),
            "rows": [{"ticker": "OEY", "status": "OK", "price": 100.0, "reason_code": "CACHE_HIT"}],
        },
    )
    facts_row = {
        "ticker": "OEY",
        "status": "OK",
        "shares_status": "OK",
        "shares_reason": "OK",
        "shares_value": 10.0,
        "cfo_status": "OK",
        "cfo_reason": "OK",
        "cfo_value": 220.0,
        "capex_status": "OK",
        "capex_reason": "OK",
        "capex_value": 60.0,
        "fcf_status": "OK",
        "fcf_reason": "OK",
        "fcf_value": 4.0,
        "fetch_reason_code": "CACHE_HIT",
        "cache_path": str(oey_cache),
        "derived_from": ["facts.OEY"],
    }
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))
    monkeypatch.setattr("app.valuation.owner_earnings.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))

    run_universe_scout(run_id="scout_owner_yield", as_of_date="2026-02-14", top_n=5, tickers=["OEY"])
    scoreboard = json.loads(
        (cfg.sectors_dir / "scout_owner_yield" / "universe_scoreboard.json").read_text(encoding="utf-8")
    )
    row = scoreboard["rows"][0]
    assert row["scout_status"] == "PASS"
    assert str(row["yield_metric_used"]).startswith("OWNER_EARNINGS")
    assert row["yield_reason_code"] == "OWNER_EARNINGS_YIELD_3Y"
    assert row["metric_values"]["owner_earnings_yield_3y"] > 0.03
    assert row["metric_values"]["fcf_yield"] < 0.03
    yield_cov = open_universe_yield_coverage(run_id="scout_owner_yield")
    assert yield_cov["status"] == "OK"
    assert yield_cov["known_count"] == 1


def test_scout_ev_computation_and_ev_yield_precedence(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_dir = cfg.cache_dir / "companyfacts"
    ev_cache = cache_dir / "0000000077.json"
    _write_companyfacts_cache(
        ev_cache,
        # Raw-USD companyfacts values: $30M debt, $10M cash. net_debt._to_musd divides
        # by 1e6 -> debt 30.0, cash 10.0 ($M), net_debt 20 -> EV = market_cap 50 + 20 = 70.
        debt=30_000_000.0,
        cash=10_000_000.0,
        shares_values=[("2023-12-31", 5.0), ("2025-12-31", 5.0)],
        fcf_values=[("2023-12-31", 10.0), ("2024-12-31", 10.0), ("2025-12-31", 10.0)],
        cfo_values=[("2023-12-31", 25.0), ("2024-12-31", 25.0), ("2025-12-31", 25.0)],
        capex_values=[("2023-12-31", 20.0), ("2024-12-31", 20.0), ("2025-12-31", 20.0)],
    )
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_ev_pref" / "prices_summary.json"),
            "rows": [{"ticker": "EVP", "status": "OK", "price": 10.0, "reason_code": "CACHE_HIT"}],
        },
    )
    facts_row = {
        "ticker": "EVP",
        "status": "OK",
        "shares_status": "OK",
        "shares_reason": "OK",
        "shares_value": 5.0,
        "cfo_status": "OK",
        "cfo_reason": "OK",
        "cfo_value": 25.0,
        "capex_status": "OK",
        "capex_reason": "OK",
        "capex_value": 20.0,
        "fcf_status": "OK",
        "fcf_reason": "OK",
        "fcf_value": 10.0,
        "fetch_reason_code": "CACHE_HIT",
        "cache_path": str(ev_cache),
        "derived_from": ["facts.EVP"],
    }
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))
    monkeypatch.setattr("app.valuation.owner_earnings.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))

    run_universe_scout(run_id="scout_ev_pref", as_of_date="2026-02-14", top_n=5, tickers=["EVP"])
    scoreboard = json.loads((cfg.sectors_dir / "scout_ev_pref" / "universe_scoreboard.json").read_text(encoding="utf-8"))
    row = scoreboard["rows"][0]
    assert row["metric_values"]["market_cap"] == 50.0
    assert row["metric_values"]["ev"] == 70.0
    assert row["yield_denominator_used"] == "EV"
    assert str(row["yield_metric_used"]).startswith("OWNER_EARNINGS_EV_")
    yield_cov = open_universe_yield_coverage(run_id="scout_ev_pref")
    assert yield_cov["ev_known_count"] == 1
    assert yield_cov["yield_denominator_counts"]["EV"] == 1
    net_debt_cov = open_net_debt_coverage(run_id="scout_ev_pref")
    assert net_debt_cov["status"] == "OK"
    assert net_debt_cov["status_counts"]["OK"] == 1


def test_scout_require_ev_yield_forces_missing_ev_when_net_debt_missing(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_dir = cfg.cache_dir / "companyfacts"
    missing_ev_cache = cache_dir / "0000000088.json"
    payload = {
        "companyfacts": {
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {"shares": [{"end": "2025-12-31", "filed": "2025-12-31", "val": 10.0}]}
                    }
                },
                "us-gaap": {
                    "FreeCashFlow": {"units": {"USD": [{"end": "2025-12-31", "filed": "2025-12-31", "val": 40.0}]}},
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"end": "2025-12-31", "filed": "2025-12-31", "val": 100.0}]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [{"end": "2025-12-31", "filed": "2025-12-31", "val": 20.0}]}
                    },
                },
            }
        }
    }
    missing_ev_cache.parent.mkdir(parents=True, exist_ok=True)
    missing_ev_cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_ev_req" / "prices_summary.json"),
            "rows": [{"ticker": "EVM", "status": "OK", "price": 20.0, "reason_code": "CACHE_HIT"}],
        },
    )
    facts_row = {
        "ticker": "EVM",
        "status": "OK",
        "shares_status": "OK",
        "shares_reason": "OK",
        "shares_value": 10.0,
        "cfo_status": "OK",
        "cfo_reason": "OK",
        "cfo_value": 100.0,
        "capex_status": "OK",
        "capex_reason": "OK",
        "capex_value": 20.0,
        "fcf_status": "OK",
        "fcf_reason": "OK",
        "fcf_value": 40.0,
        "fetch_reason_code": "CACHE_HIT",
        "cache_path": str(missing_ev_cache),
        "derived_from": ["facts.EVM"],
    }
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))
    monkeypatch.setattr("app.valuation.owner_earnings.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))

    run_universe_scout(
        run_id="scout_ev_req",
        as_of_date="2026-02-14",
        top_n=5,
        tickers=["EVM"],
        threshold_overrides={"scout_require_ev_yield": True},
    )
    scoreboard = json.loads((cfg.sectors_dir / "scout_ev_req" / "universe_scoreboard.json").read_text(encoding="utf-8"))
    row = scoreboard["rows"][0]
    # Missing debt and cash evidence is not an estimated zero. Requiring an
    # EV-denominated yield must fail closed until net debt is sourced.
    assert row["scout_status"] == "FAIL"
    assert row["primary_blocker"] == "MISSING_EV"
    assert row["ev_status"] == "UNKNOWN"
    assert row["ev_reason_code"] == "MISSING_NET_DEBT"
    assert row["yield_blocker_subreason"] == "MISSING_NET_DEBT"


def test_scout_calibration_ev_yield_delta_deterministic(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_dir = cfg.cache_dir / "companyfacts"
    delta_cache = cache_dir / "0000000099.json"
    _write_companyfacts_cache(
        delta_cache,
        debt=20.0,
        cash=0.0,
        shares_values=[("2023-12-31", 10.0), ("2025-12-31", 10.0)],
        fcf_values=[("2023-12-31", 80.0), ("2024-12-31", 80.0), ("2025-12-31", 80.0)],
        cfo_values=[("2023-12-31", 20.0), ("2024-12-31", 20.0), ("2025-12-31", 20.0)],
        capex_values=[("2023-12-31", 30.0), ("2024-12-31", 30.0), ("2025-12-31", 30.0)],
    )
    monkeypatch.setattr(
        "app.universe.scout.write_prices_for_run",
        lambda **_kwargs: {
            "summary_path": str(cfg.outputs_dir / "prices" / "scout_ev_delta" / "prices_summary.json"),
            "rows": [{"ticker": "EVD", "status": "OK", "price": 10.0, "reason_code": "CACHE_HIT"}],
        },
    )
    facts_row = {
        "ticker": "EVD",
        "status": "OK",
        "shares_status": "OK",
        "shares_reason": "OK",
        "shares_value": 10.0,
        "cfo_status": "OK",
        "cfo_reason": "OK",
        "cfo_value": 20.0,
        "capex_status": "OK",
        "capex_reason": "OK",
        "capex_value": 30.0,
        "fcf_status": "OK",
        "fcf_reason": "OK",
        "fcf_value": 80.0,
        "fetch_reason_code": "CACHE_HIT",
        "cache_path": str(delta_cache),
        "derived_from": ["facts.EVD"],
    }
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))
    monkeypatch.setattr("app.valuation.owner_earnings.resolve_financial_facts_asof", lambda **_kwargs: dict(facts_row))

    run_universe_scout(
        run_id="scout_ev_delta",
        as_of_date="2026-02-14",
        top_n=5,
        tickers=["EVD"],
        threshold_overrides={"scout_use_graham_dodd": False},
    )
    calibration = open_universe_scout_calibration(run_id="scout_ev_delta")
    assert calibration["status"] == "OK"
    near = calibration["top_near_misses"][0]
    assert near["ticker"] == "EVD"
    assert near["yield_denominator_used"] == "EV"
    assert near["yield_delta_to_pass"] > 0


def test_universe_scout_to_depth_prioritizes_near_miss_watch_when_pass_zero(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    scout_run_id = "scout_watch_priority"
    depth_run_id = "depth_watch_priority"
    scout_dir = cfg.sectors_dir / scout_run_id
    scout_dir.mkdir(parents=True, exist_ok=True)
    (scout_dir / "universe_shortlist.json").write_text(
        json.dumps(
            {
                "run_id": scout_run_id,
                "as_of_date": "2026-02-14",
                "counts": {"PASS": 0, "WATCH": 3, "FAIL": 0},
                "top_candidates": [
                    {"ticker": "AAPL", "scout_status": "WATCH", "score_total": 90.0, "derived_from": ["a"]},
                    {"ticker": "MSFT", "scout_status": "WATCH", "score_total": 85.0, "derived_from": ["b"]},
                    {"ticker": "NVDA", "scout_status": "WATCH", "score_total": 80.0, "derived_from": ["c"]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (scout_dir / "universe_scout_calibration.json").write_text(
        json.dumps(
            {
                "run_id": scout_run_id,
                "top_near_misses": [
                    {"ticker": "NVDA", "delta_to_pass": 0.01},
                    {"ticker": "AAPL", "delta_to_pass": 0.02},
                    {"ticker": "MSFT", "delta_to_pass": 0.03},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_run_sector_rlm_loop(**kwargs):
        captured.update(kwargs)
        depth_dir = cfg.sectors_dir / kwargs["run_id"]
        depth_dir.mkdir(parents=True, exist_ok=True)
        return {"run_id": kwargs["run_id"], "status": "DONE", "artifacts": {}}

    monkeypatch.setattr("app.rlm.loop.run_sector_rlm_loop", _fake_run_sector_rlm_loop)
    payload = run_universe_scout_to_depth(
        scout_run_id=scout_run_id,
        depth_run_id=depth_run_id,
        sector="Software",
        iterations=2,
        top_k=5,
        shortlist_limit=2,
    )
    assert payload["status"] == "OK"
    assert payload["selection_rationale"] == "PASS=0_NEAR_MISS_WATCH_PRIORITY"
    assert captured["seed_tickers"] == ["NVDA", "AAPL"]
    linkage = json.loads((cfg.sectors_dir / depth_run_id / "universe_scout_linkage.json").read_text(encoding="utf-8"))
    assert linkage["selection_rationale"] == "PASS=0_NEAR_MISS_WATCH_PRIORITY"
    assert linkage["selected_tickers"] == ["NVDA", "AAPL"]


def _scaling_price_stub(**kwargs):
    rows = []
    for ticker in kwargs.get("tickers", []) or []:
        rows.append({"ticker": str(ticker).upper(), "status": "OK", "price": 10.0, "reason_code": "CACHE_HIT"})
    return {"summary_path": "stub://prices", "rows": rows}


def _scaling_facts_stub(network_attempted: bool = False):
    def _impl(**kwargs):
        ticker = str(kwargs.get("ticker") or "").upper()
        return {
            "ticker": ticker,
            "status": "OK",
            "shares_status": "OK",
            "shares_reason": "OK",
            "shares_value": 100.0,
            "cfo_status": "OK",
            "cfo_reason": "OK",
            "cfo_value": 50.0,
            "capex_status": "OK",
            "capex_reason": "OK",
            "capex_value": 10.0,
            "fcf_status": "OK",
            "fcf_reason": "OK",
            "fcf_value": 40.0,
            "fetch_reason_code": "CACHE_HIT",
            "cache_path": None,
            "derived_from": [f"facts.{ticker}"],
            "network_attempted": bool(network_attempted),
        }

    return _impl


def _scaling_net_debt_stub(**kwargs):
    ticker = str(kwargs.get("ticker") or "").upper()
    return {
        "ticker": ticker,
        "as_of_date": kwargs.get("as_of_date"),
        "status": "OK",
        "reason_code": "OK",
        "total_debt": {"value": 120.0, "tag": "Debt", "date": "2025-12-31", "derived_from": [f"netdebt.{ticker}.debt"]},
        "cash_equivalents": {"value": 20.0, "tag": "Cash", "date": "2025-12-31", "derived_from": [f"netdebt.{ticker}.cash"]},
        "net_debt_proxy": 100.0,
        "derived_from": [f"netdebt.{ticker}"],
    }


def _scaling_build_record_stub(*, ticker: str, **_kwargs):
    ticker_norm = str(ticker).upper()
    selector = sum(ord(ch) for ch in ticker_norm) % 3
    scout_status = "PASS" if selector == 0 else ("WATCH" if selector == 1 else "FAIL")
    score = float(sum(ord(ch) for ch in ticker_norm) % 100)
    score_row = {
        "ticker": ticker_norm,
        "scout_status": scout_status,
        "score_total": score,
        "score_components": {"valuation_mos": 10.0},
        "primary_blocker": "NONE" if scout_status == "PASS" else "INSUFFICIENT_MOS",
        "primary_blocker_category": "NONE" if scout_status == "PASS" else "INSUFFICIENT_MOS",
        "blocker_categories": [] if scout_status == "PASS" else ["INSUFFICIENT_MOS"],
        "delta_to_pass": 0.0 if scout_status == "PASS" else 0.1,
        "near_miss_fields": [] if scout_status == "PASS" else [{"field": "scout_mos_min", "value": 0.2, "threshold": 0.3, "delta": 0.1}],
        "recommendation": "ok" if scout_status == "PASS" else "tune threshold",
        "yield_metric_used": "OWNER_EARNINGS_EV_MEDIAN_3Y",
        "yield_gate_value_used": 0.05,
        "yield_status": "OK",
        "yield_reason_code": "OWNER_EARNINGS_YIELD_EV_3Y",
        "yield_blocker_subreason": "YIELD_ABOVE_THRESHOLD",
        "yield_denominator_used": "EV",
        "yield_calibration_note": "",
        "yield_delta_to_pass": 0.0,
        "ev_status": "OK",
        "ev_reason_code": "OK",
        "reasons": ["SCALING_TEST"],
        "metric_values": {
            "valuation_gap": 0.2,
            "fcf_yield": 0.04,
            "fcf_yield_3y": 0.04,
            "owner_earnings_yield_3y": 0.05,
            "owner_earnings_yield_ev_3y": 0.05,
            "market_cap": 1000.0,
            "ev": 1100.0,
            "net_debt_proxy": 100.0,
            "net_debt_to_cfo": 2.0,
            "dilution_rate": 0.01,
        },
        "inputs_used": {},
        "derived_from": [f"scaling.{ticker_norm}"],
    }
    coverage_row = {
        "ticker": ticker_norm,
        "scout_status": scout_status,
        "primary_blocker": score_row["primary_blocker"],
        "primary_blocker_category": score_row["primary_blocker_category"],
        "blocker_categories": score_row["blocker_categories"],
        "near_miss_fields": score_row["near_miss_fields"],
        "recommendation": score_row["recommendation"],
        "yield_metric_used": score_row["yield_metric_used"],
        "yield_status": "OK",
        "yield_reason_code": score_row["yield_reason_code"],
        "yield_blocker_subreason": score_row["yield_blocker_subreason"],
        "yield_denominator_used": "EV",
        "yield_calibration_note": "",
        "yield_delta_to_pass": score_row["yield_delta_to_pass"],
        "price_status": "OK",
        "price_reason_code": "CACHE_HIT",
        "facts_status": "OK",
        "fetch_reason_code": "CACHE_HIT",
        "shares_status": "OK",
        "shares_reason_code": "OK",
        "cfo_status": "OK",
        "cfo_reason_code": "OK",
        "capex_status": "OK",
        "capex_reason_code": "OK",
        "fcf_status": "OK",
        "fcf_reason_code": "OK",
        "net_debt_status": "OK",
        "net_debt_reason_code": "OK",
        "market_cap_status": "OK",
        "market_cap_reason_code": "PRICE_X_SHARES",
        "ev_status": "OK",
        "ev_reason_code": "OK",
        "require_ev_yield": False,
        "dilution_status": "OK",
        "dilution_reason_code": "OK",
        "fcf_stability_status": "OK",
        "fcf_stability_reason_code": "OK",
        "valuation_gap_status": "OK",
        "valuation_gap_reason_code": "OK",
        "unknown_reasons": [],
        "fcf_history_values": [1.0, 2.0, 3.0],
        "derived_from": [f"scaling.{ticker_norm}"],
    }
    yield_row = {
        "ticker": ticker_norm,
        "scout_status": scout_status,
        "yield_status": "OK",
        "yield_reason_code": score_row["yield_reason_code"],
        "yield_blocker_subreason": score_row["yield_blocker_subreason"],
        "yield_metric_used": score_row["yield_metric_used"],
        "yield_metric_type": "OWNER_EARNINGS",
        "yield_denominator_used": "EV",
        "yield_calibration_note": "",
        "yield_gate_value_used": 0.05,
        "yield_delta_to_pass": 0.0,
        "owner_earnings_yield_3y": 0.05,
        "fcf_yield_3y": 0.04,
        "owner_earnings_yield_ev_3y": 0.05,
        "fcf_yield_ev_3y": 0.04,
        "ev": 1100.0,
        "ev_value": 1100.0,
        "ev_status": "OK",
        "ev_reason_code": "OK",
        "ev_used": True,
        "net_debt_proxy_used": 100.0,
        "net_debt_proxy_reason_code": "OK",
        "denominator_used": "EV",
        "primary_blocker_category": score_row["primary_blocker_category"],
        "owner_earnings_summary": {},
        "maintenance_capex_ratio": 0.6,
        "proxy_flags": {"maintenance_capex_proxy": True, "owner_earnings_proxy": True},
        "inputs_used": {},
        "derived_from": [f"scaling.{ticker_norm}"],
    }
    return score_row, coverage_row, yield_row


def test_universe_scout_batches_are_deterministic(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _scaling_price_stub)
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", _scaling_facts_stub(network_attempted=False))
    monkeypatch.setattr("app.universe.scout.resolve_net_debt_proxy", _scaling_net_debt_stub)
    monkeypatch.setattr("app.universe.scout._build_scout_record", _scaling_build_record_stub)

    tickers = ["msft", "AAPL", "nvda", "AAPL", "meta", "googl"]
    run_universe_scout(run_id="scale_det_1", as_of_date="2026-02-14", tickers=tickers, batch_size=2, top_n=5)
    run_universe_scout(run_id="scale_det_2", as_of_date="2026-02-14", tickers=tickers, batch_size=2, top_n=5)

    def _batch_layout(run_id: str):
        from app.config import get_config

        cfg = get_config()
        out = []
        for path in sorted((cfg.outputs_dir / "universe" / run_id / "batches").glob("batch_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            out.append(payload["tickers"])
        return out

    assert _batch_layout("scale_det_1") == _batch_layout("scale_det_2")
    score_1 = open_universe_scout(run_id="scale_det_1")
    score_2 = open_universe_scout(run_id="scale_det_2")
    tickers_1 = [row["ticker"] for row in json.loads(Path(score_1["universe_scoreboard_path"]).read_text(encoding="utf-8"))["rows"]]
    tickers_2 = [row["ticker"] for row in json.loads(Path(score_2["universe_scoreboard_path"]).read_text(encoding="utf-8"))["rows"]]
    assert tickers_1 == tickers_2


def test_universe_scout_resume_skips_done_batches(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    call_count = {"count": 0}

    def _counting_build_record(**kwargs):
        call_count["count"] += 1
        return _scaling_build_record_stub(**kwargs)

    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _scaling_price_stub)
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", _scaling_facts_stub(network_attempted=False))
    monkeypatch.setattr("app.universe.scout.resolve_net_debt_proxy", _scaling_net_debt_stub)
    monkeypatch.setattr("app.universe.scout._build_scout_record", _counting_build_record)

    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    first = run_universe_scout(
        run_id="scale_resume",
        as_of_date="2026-02-14",
        tickers=tickers,
        batch_size=2,
        max_batches=1,
        top_n=5,
    )
    assert first["run_status"] == "PARTIAL"
    assert call_count["count"] == 2
    status_one = open_universe_scout_status(run_id="scale_resume")
    assert status_one["batches_done"] == 1

    resumed = run_universe_scout_resume(run_id="scale_resume", max_batches=10)
    assert resumed["run_status"] == "DONE"
    assert call_count["count"] == 5
    status_two = open_universe_scout_status(run_id="scale_resume")
    assert status_two["run_status"] == "DONE"
    assert status_two["batches_done"] == 3


def test_universe_scout_budget_exhaustion_marks_partial(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _scaling_price_stub)
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", _scaling_facts_stub(network_attempted=True))
    monkeypatch.setattr("app.universe.scout.resolve_net_debt_proxy", _scaling_net_debt_stub)
    monkeypatch.setattr("app.universe.scout._build_scout_record", _scaling_build_record_stub)

    payload = run_universe_scout(
        run_id="scale_budget",
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB", "CCC", "DDD"],
        batch_size=4,
        scout_sec_budget=1,
        top_n=4,
    )
    assert payload["run_status"] == "PARTIAL"
    assert payload["stop_reason_code"] == "BUDGET_EXHAUSTED"
    status = open_universe_scout_status(run_id="scale_budget")
    assert status["run_status"] == "PARTIAL"
    assert status["stop_reason_code"] == "BUDGET_EXHAUSTED"
    coverage = json.loads(Path(payload["universe_coverage_path"]).read_text(encoding="utf-8"))
    skipped = [
        row for row in coverage["rows"] if "SKIPPED_BUDGET" in [str(reason) for reason in (row.get("unknown_reasons") or [])]
    ]
    assert len(skipped) >= 1


def test_universe_scout_progress_fields_advance_deterministically(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _scaling_price_stub)
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", _scaling_facts_stub(network_attempted=False))
    monkeypatch.setattr("app.universe.scout.resolve_net_debt_proxy", _scaling_net_debt_stub)
    monkeypatch.setattr("app.universe.scout._build_scout_record", _scaling_build_record_stub)

    payload = run_universe_scout(
        run_id="scout_progress",
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        batch_size=2,
        top_n=2,
    )
    assert payload["run_status"] == "DONE"

    status = open_universe_scout_status(run_id="scout_progress")
    assert status["status"] == "OK"
    assert status["hydration_status"] == "OK"
    assert status["last_progress_phase"] == "DONE"
    assert status["price_stage_completed"] is True
    assert status["facts_stage_completed"] is True
    assert status["scoring_stage_completed"] is True
    assert status["last_completed_ticker"] == "BBB"
    assert status["tickers_completed"] == 2

    opened = open_universe_scout(run_id="scout_progress")
    assert opened["hydration_status"] == "OK"
    assert opened["hydration_progress"]["companyfacts_fetch_attempts"] == 0


def test_universe_scout_facts_timeout_degrades_to_reason_coded_unknowns(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _scaling_price_stub)

    def _timeout_facts(*, phase, ticker_label, timeout_seconds, fn, **kwargs):
        if phase == "COMPANYFACTS_ACQUISITION":
            raise ScoutPhaseTimeoutError(
                phase=phase,
                ticker=ticker_label,
                timeout_seconds=float(timeout_seconds or 1.0),
            )
        return fn(**kwargs)

    monkeypatch.setattr("app.universe.scout._call_with_timeout", _timeout_facts)

    payload = run_universe_scout(
        run_id="scout_timeout_facts",
        as_of_date="2026-02-14",
        tickers=["AAA"],
        top_n=1,
    )
    assert payload["run_status"] == "DONE"

    coverage = json.loads(Path(payload["universe_coverage_path"]).read_text(encoding="utf-8"))
    row = coverage["rows"][0]
    assert row["fetch_reason_code"] == "SCOUT_FACTS_TIMEOUT"
    assert row["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"
    assert row["facts_blocker_retryable"] is True
    assert row["facts_recommended_action"] == "RETRY_COMPANYFACTS_HYDRATION"
    assert row["primary_fail_domain"] == "EVIDENCE"
    assert "SCOUT_FACTS_TIMEOUT" in coverage["unknown_reason_counts"]
    assert coverage["retryable_facts_blocker_count"] == 1

    status = open_universe_scout_status(run_id="scout_timeout_facts")
    assert status["hydration_status"] == "DEGRADED"
    assert status["companyfacts_timeouts"] == 1
    assert status["primary_scout_blocker"] == "SCOUT_FACTS_TIMEOUT"
    assert status["retryable_facts_blocker_count"] == 1

    opened = open_universe_scout(run_id="scout_timeout_facts")
    assert opened["hydration_status"] == "DEGRADED"
    assert opened["primary_scout_blocker"] == "SCOUT_FACTS_TIMEOUT"
    assert opened["facts_blockers"]["retryable_facts_blocker_count"] == 1
    assert opened["facts_blockers"]["top_retryable_facts_blockers"][0]["facts_blocker_class"] == "FACTS_RETRYABLE_TIMEOUT"


def test_universe_scout_separates_evidence_fail_from_economic_fail(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _scaling_price_stub)
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", _scaling_facts_stub(network_attempted=False))
    monkeypatch.setattr("app.universe.scout.resolve_net_debt_proxy", _scaling_net_debt_stub)

    def _evidence_vs_economics_stub(*, ticker: str, **_kwargs):
        ticker_norm = str(ticker).upper()
        if ticker_norm == "AAA":
            score_row = {
                "ticker": "AAA",
                "scout_status": "FAIL",
                "score_total": 10.0,
                "score_components": {},
                "primary_blocker": "MISSING_FACTS",
                "primary_blocker_category": "MISSING_FACTS",
                "blocker_categories": ["MISSING_FACTS"],
                "delta_to_pass": "UNKNOWN",
                "near_miss_fields": [],
                "recommendation": "hydrate facts",
                "yield_metric_used": "UNKNOWN",
                "yield_gate_value_used": "UNKNOWN",
                "yield_status": "UNKNOWN",
                "yield_reason_code": "MISSING_FCF",
                "yield_blocker_subreason": "MISSING_FCF",
                "yield_denominator_used": "UNKNOWN",
                "yield_calibration_note": "",
                "yield_delta_to_pass": "UNKNOWN",
                "ev_status": "UNKNOWN",
                "ev_reason_code": "MISSING_EV",
                "reasons": ["FACTS_UNKNOWN"],
                "metric_values": {},
                "inputs_used": {},
                "derived_from": ["evidence.AAA"],
                "facts_blocker_class": "FACTS_RETRYABLE_TIMEOUT",
                "facts_blocker_retryable": True,
                "facts_blocker_terminal": False,
                "facts_blocker_partial_usable": False,
                "facts_missing_key_inputs": ["SHARES", "CFO", "CAPEX", "FCF"],
                "facts_retry_recommended": True,
                "facts_blocker_reason_codes": ["SCOUT_FACTS_TIMEOUT"],
                "facts_recommended_action": "RETRY_COMPANYFACTS_HYDRATION",
                "fail_due_to_missing_evidence": True,
                "fail_due_to_economic_weakness": False,
                "primary_fail_domain": "EVIDENCE",
            }
            coverage_row = {
                "ticker": "AAA",
                "scout_status": "FAIL",
                "primary_blocker": "MISSING_FACTS",
                "primary_blocker_category": "MISSING_FACTS",
                "blocker_categories": ["MISSING_FACTS"],
                "near_miss_fields": [],
                "recommendation": "hydrate facts",
                "yield_metric_used": "UNKNOWN",
                "yield_status": "UNKNOWN",
                "yield_reason_code": "MISSING_FCF",
                "yield_blocker_subreason": "MISSING_FCF",
                "yield_denominator_used": "UNKNOWN",
                "yield_calibration_note": "",
                "yield_delta_to_pass": "UNKNOWN",
                "price_status": "OK",
                "price_reason_code": "CACHE_HIT",
                "facts_status": "UNKNOWN",
                "fetch_reason_code": "SCOUT_FACTS_TIMEOUT",
                "shares_status": "UNKNOWN",
                "shares_reason_code": "SCOUT_FACTS_TIMEOUT",
                "cfo_status": "UNKNOWN",
                "cfo_reason_code": "SCOUT_FACTS_TIMEOUT",
                "capex_status": "UNKNOWN",
                "capex_reason_code": "SCOUT_FACTS_TIMEOUT",
                "fcf_status": "UNKNOWN",
                "fcf_reason_code": "SCOUT_FACTS_TIMEOUT",
                "net_debt_status": "UNKNOWN",
                "net_debt_reason_code": "NO_FACTS",
                "market_cap_status": "UNKNOWN",
                "market_cap_reason_code": "MISSING_SHARES",
                "ev_status": "UNKNOWN",
                "ev_reason_code": "MISSING_EV",
                "require_ev_yield": False,
                "dilution_status": "UNKNOWN",
                "dilution_reason_code": "UNKNOWN",
                "fcf_stability_status": "UNKNOWN",
                "fcf_stability_reason_code": "UNKNOWN",
                "valuation_gap_status": "UNKNOWN",
                "valuation_gap_reason_code": "MISSING_INTRINSIC_OR_PRICE",
                "unknown_reasons": ["SCOUT_FACTS_TIMEOUT"],
                "fcf_history_values": [],
                "derived_from": ["evidence.AAA"],
            }
        else:
            score_row = {
                "ticker": "BBB",
                "scout_status": "FAIL",
                "score_total": 20.0,
                "score_components": {},
                "primary_blocker": "NEGATIVE_CFO",
                "primary_blocker_category": "NEGATIVE_CFO",
                "blocker_categories": ["NEGATIVE_CFO"],
                "delta_to_pass": 0.2,
                "near_miss_fields": [],
                "recommendation": "keep fail",
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "yield_gate_value_used": 0.01,
                "yield_status": "LOW",
                "yield_reason_code": "LOW_YIELD_OWNER_EARNINGS_EV",
                "yield_blocker_subreason": "OWNER_EARNINGS",
                "yield_denominator_used": "EV",
                "yield_calibration_note": "",
                "yield_delta_to_pass": 0.03,
                "ev_status": "OK",
                "ev_reason_code": "OK",
                "reasons": ["FAIL_NEGATIVE_CFO"],
                "metric_values": {},
                "inputs_used": {},
                "derived_from": ["economics.BBB"],
                "facts_blocker_class": "FACTS_OK",
                "facts_blocker_retryable": False,
                "facts_blocker_terminal": False,
                "facts_blocker_partial_usable": False,
                "facts_missing_key_inputs": [],
                "facts_retry_recommended": False,
                "facts_blocker_reason_codes": [],
                "facts_recommended_action": "NONE",
                "fail_due_to_missing_evidence": False,
                "fail_due_to_economic_weakness": True,
                "primary_fail_domain": "ECONOMICS",
            }
            coverage_row = {
                "ticker": "BBB",
                "scout_status": "FAIL",
                "primary_blocker": "NEGATIVE_CFO",
                "primary_blocker_category": "NEGATIVE_CFO",
                "blocker_categories": ["NEGATIVE_CFO"],
                "near_miss_fields": [],
                "recommendation": "keep fail",
                "yield_metric_used": "owner_earnings_yield_ev_3y",
                "yield_status": "LOW",
                "yield_reason_code": "LOW_YIELD_OWNER_EARNINGS_EV",
                "yield_blocker_subreason": "OWNER_EARNINGS",
                "yield_denominator_used": "EV",
                "yield_calibration_note": "",
                "yield_delta_to_pass": 0.03,
                "price_status": "OK",
                "price_reason_code": "CACHE_HIT",
                "facts_status": "OK",
                "fetch_reason_code": "CACHE_HIT",
                "shares_status": "OK",
                "shares_reason_code": "OK",
                "cfo_status": "OK",
                "cfo_reason_code": "OK",
                "capex_status": "OK",
                "capex_reason_code": "OK",
                "fcf_status": "OK",
                "fcf_reason_code": "OK",
                "net_debt_status": "OK",
                "net_debt_reason_code": "OK",
                "market_cap_status": "OK",
                "market_cap_reason_code": "PRICE_X_SHARES",
                "ev_status": "OK",
                "ev_reason_code": "OK",
                "require_ev_yield": False,
                "dilution_status": "OK",
                "dilution_reason_code": "OK",
                "fcf_stability_status": "OK",
                "fcf_stability_reason_code": "OK",
                "valuation_gap_status": "OK",
                "valuation_gap_reason_code": "OK",
                "unknown_reasons": [],
                "fcf_history_values": [1.0],
                "derived_from": ["economics.BBB"],
            }
        yield_row = {
            "ticker": ticker_norm,
            "scout_status": "FAIL",
            "yield_status": coverage_row["yield_status"],
            "yield_reason_code": coverage_row["yield_reason_code"],
            "yield_blocker_subreason": coverage_row["yield_blocker_subreason"],
            "yield_metric_used": coverage_row["yield_metric_used"],
            "yield_metric_type": "OWNER_EARNINGS",
            "yield_denominator_used": coverage_row["yield_denominator_used"],
            "yield_calibration_note": coverage_row["yield_calibration_note"],
            "yield_gate_value_used": score_row["yield_gate_value_used"],
            "yield_delta_to_pass": score_row["yield_delta_to_pass"],
            "owner_earnings_yield_3y": "UNKNOWN",
            "fcf_yield_3y": "UNKNOWN",
            "owner_earnings_yield_ev_3y": "UNKNOWN",
            "fcf_yield_ev_3y": "UNKNOWN",
            "ev": "UNKNOWN",
            "ev_value": "UNKNOWN",
            "ev_status": coverage_row["ev_status"],
            "ev_reason_code": coverage_row["ev_reason_code"],
            "ev_used": False,
            "net_debt_proxy_used": "UNKNOWN",
            "net_debt_proxy_reason_code": coverage_row["net_debt_reason_code"],
            "denominator_used": coverage_row["yield_denominator_used"],
            "primary_blocker_category": score_row["primary_blocker_category"],
            "owner_earnings_summary": {},
            "maintenance_capex_ratio": 0.6,
            "proxy_flags": {"maintenance_capex_proxy": True, "owner_earnings_proxy": True},
            "inputs_used": {},
            "derived_from": [f"yield.{ticker_norm}"],
        }
        return score_row, coverage_row, yield_row

    monkeypatch.setattr("app.universe.scout._build_scout_record", _evidence_vs_economics_stub)

    payload = run_universe_scout(
        run_id="scout_fail_domains",
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        top_n=2,
    )
    assert payload["run_status"] == "DONE"

    coverage = json.loads(Path(payload["universe_coverage_path"]).read_text(encoding="utf-8"))
    rows = {row["ticker"]: row for row in coverage["rows"]}
    assert rows["AAA"]["fail_due_to_missing_evidence"] is True
    assert rows["AAA"]["primary_fail_domain"] == "EVIDENCE"
    assert rows["BBB"]["fail_due_to_economic_weakness"] is True
    assert rows["BBB"]["primary_fail_domain"] == "ECONOMICS"

    summary = json.loads(Path(payload["universe_summary_path"]).read_text(encoding="utf-8"))
    assert summary["facts_blockers"]["economic_fail_count_vs_evidence_fail_count"]["evidence_fail_count"] == 1
    assert summary["facts_blockers"]["economic_fail_count_vs_evidence_fail_count"]["economic_fail_count"] == 1


def test_universe_scout_scoring_timeout_fails_safely_with_stall_reason(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _scaling_price_stub)
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", _scaling_facts_stub(network_attempted=False))
    monkeypatch.setattr("app.universe.scout.resolve_net_debt_proxy", _scaling_net_debt_stub)
    monkeypatch.setattr("app.universe.scout._build_scout_record", _scaling_build_record_stub)

    def _timeout_scoring(*, phase, ticker_label, timeout_seconds, fn, **kwargs):
        if phase == "SCOUT_SCORING":
            raise ScoutPhaseTimeoutError(
                phase=phase,
                ticker=ticker_label,
                timeout_seconds=float(timeout_seconds or 1.0),
            )
        return fn(**kwargs)

    monkeypatch.setattr("app.universe.scout._call_with_timeout", _timeout_scoring)

    payload = run_universe_scout(
        run_id="scout_timeout_scoring",
        as_of_date="2026-02-14",
        tickers=["AAA"],
        top_n=1,
    )
    assert payload["run_status"] == "PARTIAL"
    assert payload["stop_reason_code"] == "STALE_SCOUT"

    status = open_universe_scout_status(run_id="scout_timeout_scoring")
    assert status["run_status"] == "PARTIAL"
    assert status["hydration_status"] == "STALE"
    assert status["hydration_phase"] == "SCOUT_SCORING"
    assert status["primary_scout_blocker"] == "SCOUT_SCORING_TIMEOUT"
    assert status["stalled_reason_code"] == "SCOUT_SCORING_TIMEOUT"

    opened = open_universe_scout(run_id="scout_timeout_scoring")
    assert opened["status"] == "OK"
    assert opened["primary_scout_blocker"] == "SCOUT_SCORING_TIMEOUT"
    assert opened["last_progress_phase"] == "SCOUT_SCORING"


def test_universe_scout_resume_after_stale_partial_is_deterministic(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    monkeypatch.setattr("app.universe.scout.write_prices_for_run", _scaling_price_stub)
    monkeypatch.setattr("app.universe.scout.resolve_financial_facts_asof", _scaling_facts_stub(network_attempted=False))
    monkeypatch.setattr("app.universe.scout.resolve_net_debt_proxy", _scaling_net_debt_stub)
    monkeypatch.setattr("app.universe.scout._build_scout_record", _scaling_build_record_stub)

    timeout_once = {"pending": True}

    def _timeout_once(*, phase, ticker_label, timeout_seconds, fn, **kwargs):
        if phase == "SCOUT_SCORING" and timeout_once["pending"]:
            timeout_once["pending"] = False
            raise ScoutPhaseTimeoutError(
                phase=phase,
                ticker=ticker_label,
                timeout_seconds=float(timeout_seconds or 1.0),
            )
        return fn(**kwargs)

    monkeypatch.setattr("app.universe.scout._call_with_timeout", _timeout_once)

    first = run_universe_scout(
        run_id="scout_resume_timeout",
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB"],
        batch_size=2,
        top_n=2,
    )
    assert first["run_status"] == "PARTIAL"
    assert first["stop_reason_code"] == "STALE_SCOUT"

    resumed = run_universe_scout_resume(run_id="scout_resume_timeout", max_batches=10)
    assert resumed["run_status"] == "DONE"

    scoreboard = json.loads(Path(resumed["universe_scoreboard_path"]).read_text(encoding="utf-8"))
    assert [row["ticker"] for row in scoreboard["rows"]] == ["BBB", "AAA"]
    status = open_universe_scout_status(run_id="scout_resume_timeout")
    assert status["run_status"] == "DONE"
    assert status["batches_done"] == 1


def test_universe_scout_cli_commands(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    def _fake_run_universe_scout(**kwargs):
        captured.update(kwargs)
        return {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "counts": {"PASS": 1, "WATCH": 1, "FAIL": 0},
            "thresholds_effective": {"scout_mos_min": 0.15},
        }

    monkeypatch.setattr("app.universe.scout.run_universe_scout", _fake_run_universe_scout)
    monkeypatch.setattr(
        "app.universe.scout.open_universe_scout",
        lambda **kwargs: {"status": "OK", "run_id": kwargs["run_id"], "counts": {"PASS": 1, "WATCH": 0, "FAIL": 0}},
    )
    monkeypatch.setattr(
        "app.universe.scout.open_universe_scout_calibration",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "counts": {"PASS": 0, "WATCH": 2, "FAIL": 0},
            "blocker_counts": {"INSUFFICIENT_MOS": 2},
            "top_near_misses": [{"ticker": "AAA", "delta_to_pass": 0.03}],
            "suggested_threshold_adjustments": ["If you reduce scout_mos_min from 0.3000 to 0.2500, PASS could increase by 1"],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.open_universe_yield_coverage",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "known_count": 1,
            "unknown_count": 0,
            "yield_blocker_breakdown": {},
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.open_graham_dodd",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "gd_known_count": 1,
            "gd_unknown_count": 0,
            "reason_counts": {"OK": 1},
            "top_mos_epv": [{"ticker": "AAPL", "mos_epv": 0.5}],
            "top_mos_netnet": [],
            "top_netnet_situations": [],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.open_universe_rankings",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "ticker_count": 2,
            "top_overall": [{"ticker": "AAPL", "composite_score_total": 75.0}],
            "top_pass": [{"ticker": "AAPL"}],
            "top_watch": [{"ticker": "MSFT"}],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.open_universe_depth_queue",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "queue_count": 1,
            "entries": [{"run_id_suggested": "depth_from_scout_cli_001", "tickers": ["AAPL", "MSFT"]}],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.universe_depth_queue_to_runs",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "selected_count": 1,
            "commands": [
                ".venv/bin/python -m app.cli sector-rlm --mode depth --sector Software --run-id depth_from_scout_cli_001 --iterations 2 --peer-limit 2 --min-peers-dossierable 2 --limit-dossiers 2 --top-k 2 --tickers AAPL,MSFT"
            ],
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.open_net_debt_coverage",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "ticker_count": 2,
            "status_counts": {"OK": 1, "UNKNOWN": 1},
            "reason_counts": {"OK": 1, "MISSING_DEBT": 1},
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.run_universe_scout_to_depth",
        lambda **kwargs: {"status": "OK", "scout_run_id": kwargs["scout_run_id"], "depth_run_id": kwargs["depth_run_id"]},
    )

    cmd = runner.invoke(
        app,
        [
            "universe-scout",
            "--as-of",
            "2026-02-14",
            "--tickers",
            "AAPL,MSFT",
            "--top-n",
            "5",
            "--scout-mos-min",
            "0.15",
            "--gd-discount-rate",
            "0.12",
            "--no-scout-use-graham-dodd",
            "--scout-require-ev-yield",
            "--run-id",
            "scout_cli",
        ],
    )
    assert cmd.exit_code == 0, cmd.output
    scout_payload = json.loads(cmd.output)
    assert scout_payload["status"] == "OK"
    assert captured["threshold_overrides"]["scout_mos_min"] == 0.15
    assert captured["threshold_overrides"]["gd_discount_rate"] == 0.12
    assert captured["threshold_overrides"]["scout_use_graham_dodd"] is False
    assert captured["threshold_overrides"]["scout_require_ev_yield"] is True

    open_cmd = runner.invoke(app, ["universe-scout-open", "--run-id", "scout_cli"])
    assert open_cmd.exit_code == 0, open_cmd.output
    open_payload = json.loads(open_cmd.output)
    assert open_payload["status"] == "OK"

    calibration_cmd = runner.invoke(app, ["universe-scout-calibration-open", "--run-id", "scout_cli"])
    assert calibration_cmd.exit_code == 0, calibration_cmd.output
    calibration_payload = json.loads(calibration_cmd.output)
    assert calibration_payload["status"] == "OK"
    assert calibration_payload["counts"]["WATCH"] == 2

    yield_cmd = runner.invoke(app, ["universe-yield-coverage-open", "--run-id", "scout_cli"])
    assert yield_cmd.exit_code == 0, yield_cmd.output
    yield_payload = json.loads(yield_cmd.output)
    assert yield_payload["status"] == "OK"
    assert yield_payload["known_count"] == 1

    gd_cmd = runner.invoke(app, ["graham-dodd-open", "--run-id", "scout_cli"])
    assert gd_cmd.exit_code == 0, gd_cmd.output
    gd_payload = json.loads(gd_cmd.output)
    assert gd_payload["status"] == "OK"
    assert gd_payload["gd_known_count"] == 1

    rankings_cmd = runner.invoke(app, ["universe-rankings-open", "--run-id", "scout_cli"])
    assert rankings_cmd.exit_code == 0, rankings_cmd.output
    rankings_payload = json.loads(rankings_cmd.output)
    assert rankings_payload["status"] == "OK"
    assert rankings_payload["ticker_count"] == 2

    depth_queue_cmd = runner.invoke(app, ["universe-depth-queue-open", "--run-id", "scout_cli"])
    assert depth_queue_cmd.exit_code == 0, depth_queue_cmd.output
    depth_queue_payload = json.loads(depth_queue_cmd.output)
    assert depth_queue_payload["status"] == "OK"
    assert depth_queue_payload["queue_count"] == 1

    depth_queue_runs_cmd = runner.invoke(
        app,
        ["universe-depth-queue-to-runs", "--run-id", "scout_cli", "--max-runs", "1"],
    )
    assert depth_queue_runs_cmd.exit_code == 0, depth_queue_runs_cmd.output
    depth_queue_runs_payload = json.loads(depth_queue_runs_cmd.output)
    assert depth_queue_runs_payload["status"] == "OK"
    assert depth_queue_runs_payload["selected_count"] == 1
    assert depth_queue_runs_payload["commands"][0].startswith(".venv/bin/python -m app.cli sector-rlm --mode depth")

    net_debt_cmd = runner.invoke(app, ["net-debt-coverage-open", "--run-id", "scout_cli"])
    assert net_debt_cmd.exit_code == 0, net_debt_cmd.output
    net_debt_payload = json.loads(net_debt_cmd.output)
    assert net_debt_payload["status"] == "OK"
    assert net_debt_payload["ticker_count"] == 2

    to_depth = runner.invoke(
        app,
        [
            "universe-scout-to-depth",
            "--run-id",
            "scout_cli",
            "--sector",
            "Software",
            "--iterations",
            "2",
            "--top-k",
            "5",
            "--depth-run-id",
            "depth_cli",
        ],
    )
    assert to_depth.exit_code == 0, to_depth.output
    handoff_payload = json.loads(to_depth.output)
    assert handoff_payload["status"] == "OK"


def test_universe_scout_cli_status_resume_cancel_commands(monkeypatch, tmp_path):
    _init_cfg(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.universe.scout.open_universe_scout_status",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "run_status": "PARTIAL",
            "batches_done": 1,
            "total_batches": 4,
            "last_completed_batch": 0,
            "stop_reason_code": "MAX_BATCHES_REACHED",
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.run_universe_scout_resume",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "run_status": "DONE",
            "stop_reason_code": "NONE",
            "batch_progress": {"batches_done": 4, "total_batches": 4},
        },
    )
    monkeypatch.setattr(
        "app.universe.scout.cancel_universe_scout_run",
        lambda **kwargs: {
            "status": "OK",
            "run_id": kwargs["run_id"],
            "run_status": "CANCELLED",
            "stop_reason_code": "CANCELLED",
            "stop_summary": kwargs["reason"],
        },
    )

    status_cmd = runner.invoke(app, ["universe-scout-status", "--run-id", "scout_cli_status"])
    assert status_cmd.exit_code == 0, status_cmd.output
    status_payload = json.loads(status_cmd.output)
    assert status_payload["status"] == "OK"
    assert status_payload["run_status"] == "PARTIAL"

    resume_cmd = runner.invoke(
        app,
        [
            "universe-scout-resume",
            "--run-id",
            "scout_cli_status",
            "--max-batches",
            "10",
            "--scout-sec-budget",
            "5",
            "--scout-net-budget",
            "3",
            "--scout-max-seconds",
            "60",
        ],
    )
    assert resume_cmd.exit_code == 0, resume_cmd.output
    resume_payload = json.loads(resume_cmd.output)
    assert resume_payload["status"] == "OK"
    assert resume_payload["run_status"] == "DONE"

    cancel_cmd = runner.invoke(
        app,
        [
            "universe-scout-cancel",
            "--run-id",
            "scout_cli_status",
            "--reason",
            "operator requested stop",
        ],
    )
    assert cancel_cmd.exit_code == 0, cancel_cmd.output
    cancel_payload = json.loads(cancel_cmd.output)
    assert cancel_payload["status"] == "OK"
    assert cancel_payload["run_status"] == "CANCELLED"


def test_corrupt_scout_state_raises_instead_of_silent_restart(monkeypatch, tmp_path):
    import pytest

    from app.universe.scout import _run_paths
    from app.util.json_io import JsonCorruptError

    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "u_corrupt"
    paths = _run_paths(cfg=cfg, run_id=run_id)
    paths["universe_dir"].mkdir(parents=True, exist_ok=True)
    # Simulate a kill mid-write: a truncated, unparseable scout_state.json that
    # previously had a RUNNING/PARTIAL status. A silent {} read would make the
    # resume gate restart the whole sweep from batch 0; instead this must raise.
    paths["state_path"].write_text('{"status": "RUNN', encoding="utf-8")

    with pytest.raises(JsonCorruptError):
        run_universe_scout(
            run_id=run_id,
            as_of_date="2025-12-31",
            tickers=["AAPL", "MSFT"],
            with_prices=False,
        )

    # The corrupt file must be left untouched (NOT silently overwritten / restarted).
    assert paths["state_path"].read_text(encoding="utf-8") == '{"status": "RUNN'


def test_force_restart_clears_corrupt_scout_state(monkeypatch, tmp_path):
    from app.universe.scout import _run_paths

    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "u_corrupt_force"
    paths = _run_paths(cfg=cfg, run_id=run_id)
    paths["universe_dir"].mkdir(parents=True, exist_ok=True)
    paths["state_path"].write_text('{"status": "RUNN', encoding="utf-8")

    # force_restart wipes the universe_dir before the resume-gate read, so a
    # corrupt prior state must not block a forced fresh run.
    result = run_universe_scout(
        run_id=run_id,
        as_of_date="2025-12-31",
        tickers=["AAPL", "MSFT"],
        with_prices=False,
        force_restart=True,
    )
    assert result["run_id"] == run_id


def test_scout_yields_put_whole_dollar_earnings_over_a_millions_market_cap():
    """Owner earnings and the companyfacts FCF series are whole dollars; the scout's
    market cap is price x facts-row shares (millions), so $millions. Owner earnings of
    $50,000,000 over a $1,000M market cap is a 5% yield, not 50,000 (5,000,000%)."""
    from app.universe.scout import _select_yield_metric

    profile = _select_yield_metric(
        price_status="OK",
        shares_status="OK",
        market_cap=1000.0,
        ev="UNKNOWN",
        ev_status="UNKNOWN",
        ev_reason_code="MISSING_EV",
        require_ev_yield=False,
        owner_payload={"summary": {
            "owner_earnings_normalized_3y": 50_000_000.0,
            "owner_earnings_latest": 60_000_000.0,
            "owner_earnings_normalized_method": "MEDIAN_3Y",
        }},
        fcf_series=[{"year": 2024, "value": 20_000_000.0}, {"year": 2025, "value": 40_000_000.0}],
        fcf_latest=40.0,
        thresholds={"scout_fcf_yield_min": 0.03},
    )
    assert profile["owner_earnings_yield_3y"] == 0.05
    assert profile["owner_earnings_yield_latest"] == 0.06
    assert profile["fcf_yield_3y"] == 0.03
