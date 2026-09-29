from __future__ import annotations

import json
from pathlib import Path

from app.valuation.value_gates import (
    FAIL,
    PASS,
    WATCH,
    classify_value_gate_ticker,
    open_value_gates_calibration_for_run,
    open_value_gates_for_run,
    write_value_gates_for_run,
)


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def _fundamentals_payload(
    *,
    fcf_latest: float,
    cfo_latest: float,
    net_debt_latest: float,
    fcf_slope: float | str,
    dilution: float | str,
) -> dict[str, object]:
    return {
        "rows": [
            {"year": 2023, "fcf": 10.0, "cfo": 20.0, "net_debt": 80.0},
            {"year": 2024, "fcf": 12.0, "cfo": 25.0, "net_debt": 70.0},
            {"year": 2025, "fcf": fcf_latest, "cfo": cfo_latest, "net_debt": net_debt_latest},
        ],
        "row_traces": {
            "2025": {
                "fcf": {"derived_from": ["fundamentals.rows[2025].fcf"]},
                "cfo": {"derived_from": ["fundamentals.rows[2025].cfo"]},
                "net_debt": {"derived_from": ["fundamentals.rows[2025].net_debt"]},
            }
        },
        "derived_signals": {
            "fcf_margin_trend_slope": {
                "value": fcf_slope,
                "derived_from": ["fundamentals.derived_signals.fcf_margin_trend_slope"],
            },
            "dilution_rate_shares_cagr": {
                "value": dilution,
                "derived_from": ["fundamentals.derived_signals.dilution_rate_shares_cagr"],
            },
        },
    }


def _seed_run_single_ticker(
    *,
    run_dir: Path,
    ticker: str,
    intrinsic_per_share: float | str,
    current_price: float | str,
    fcf_latest: float | str,
    shares_outstanding: float | str,
    net_debt: float | str,
    cfo_value: float | str,
    fcf_slope: float | str,
    dilution: float | str,
    price_status: str = "OK",
    shares_status: str = "OK",
    fcf_status: str = "OK",
) -> None:
    ticker = ticker.upper()
    (run_dir / "peer_scoreboard.json").write_text(
        json.dumps({"rows": [{"ticker": ticker}]}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "valuation_coverage.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "ticker": ticker,
                        "price_status": price_status,
                        "shares_status": shares_status,
                        "fcf_status": fcf_status,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / f"valuation_{ticker}.json").write_text(
        json.dumps(
            {
                "intrinsic_per_share_base": intrinsic_per_share,
                "input_snapshot": {
                    "current_price": current_price,
                    "fcf_latest": fcf_latest,
                    "shares_outstanding": shares_outstanding,
                    "net_debt": net_debt,
                    "cfo_value": cfo_value,
                },
                "claims": {
                    "intrinsic_per_share_base": {
                        "derived_from": ["valuation.claims.intrinsic_per_share_base.value"]
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / f"fundamentals_{ticker}.json").write_text(
        json.dumps(
            _fundamentals_payload(
                fcf_latest=float(fcf_latest) if isinstance(fcf_latest, (int, float)) else 0.0,
                cfo_latest=float(cfo_value) if isinstance(cfo_value, (int, float)) else 0.0,
                net_debt_latest=float(net_debt) if isinstance(net_debt, (int, float)) else 0.0,
                fcf_slope=fcf_slope,
                dilution=dilution,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def test_classify_value_gate_ticker_pass() -> None:
    entry = classify_value_gate_ticker(
        ticker="AAA",
        valuation={
            "intrinsic_per_share_base": 150.0,
            "input_snapshot": {
                "current_price": 100.0,
                "fcf_latest": 20.0,
                "shares_outstanding": 100.0,
                "net_debt": 60.0,
                "cfo_value": 40.0,
            },
            "claims": {
                "intrinsic_per_share_base": {"derived_from": ["valuation.claims.intrinsic_per_share_base.value"]},
            },
        },
        fundamentals=_fundamentals_payload(
            fcf_latest=20.0,
            cfo_latest=40.0,
            net_debt_latest=60.0,
            fcf_slope=0.01,
            dilution=0.01,
        ),
        valuation_coverage_entry={
            "ticker": "AAA",
            "price_status": "OK",
            "shares_status": "OK",
            "fcf_status": "OK",
        },
    )
    assert entry["gate_status"] == PASS
    assert entry["primary_blocker"] == "NONE"
    assert entry["valuation_gap"] == 0.5
    assert entry["reason_category_counts"]["PASS_REASON"] >= 1
    assert entry["hydration_actions"] == []


def test_primary_blocker_is_deterministic_and_prefers_input_missing() -> None:
    entry = classify_value_gate_ticker(
        ticker="BBB",
        valuation={
            "intrinsic_per_share_base": 120.0,
            "input_snapshot": {
                "current_price": "UNKNOWN",
                "fcf_latest": -4.0,
                "shares_outstanding": 100.0,
                "net_debt": 150.0,
                "cfo_value": 20.0,
            },
        },
        fundamentals=_fundamentals_payload(
            fcf_latest=-4.0,
            cfo_latest=20.0,
            net_debt_latest=150.0,
            fcf_slope=-0.03,
            dilution=0.08,
        ),
        valuation_coverage_entry={
            "ticker": "BBB",
            "price_status": "UNKNOWN",
            "shares_status": "OK",
            "fcf_status": "OK",
        },
    )
    assert entry["gate_status"] == FAIL
    assert "PRICE_UNKNOWN" in set(entry["gate_reasons"])
    assert any(reason.startswith("FCF_") for reason in entry["gate_reasons"])
    assert entry["primary_blocker"] == "PRICE_UNKNOWN"
    assert entry["reason_category_counts"]["INPUT_MISSING"] >= 1
    assert entry["reason_category_counts"]["SAFETY_FAIL"] >= 1


def test_write_open_and_calibration_artifact(monkeypatch, tmp_path) -> None:
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "value_gates_calibration_test"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "peer_scoreboard.json").write_text(
        json.dumps({"rows": [{"ticker": "AAA"}, {"ticker": "BBB"}]}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "valuation_coverage.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"ticker": "AAA", "price_status": "OK", "shares_status": "OK", "fcf_status": "OK"},
                    {"ticker": "BBB", "price_status": "UNKNOWN", "shares_status": "OK", "fcf_status": "OK"},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "valuation_AAA.json").write_text(
        json.dumps(
            {
                "intrinsic_per_share_base": 160.0,
                "input_snapshot": {
                    "current_price": 100.0,
                    "fcf_latest": 25.0,
                    "shares_outstanding": 100.0,
                    "net_debt": 50.0,
                    "cfo_value": 40.0,
                },
                "claims": {
                    "intrinsic_per_share_base": {
                        "derived_from": ["valuation.claims.intrinsic_per_share_base.value"]
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "valuation_BBB.json").write_text(
        json.dumps(
            {
                "intrinsic_per_share_base": 120.0,
                "input_snapshot": {
                    "current_price": "UNKNOWN",
                    "fcf_latest": 12.0,
                    "shares_outstanding": 100.0,
                    "net_debt": 80.0,
                    "cfo_value": 20.0,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "fundamentals_AAA.json").write_text(
        json.dumps(
            _fundamentals_payload(
                fcf_latest=25.0,
                cfo_latest=40.0,
                net_debt_latest=50.0,
                fcf_slope=0.01,
                dilution=0.01,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "fundamentals_BBB.json").write_text(
        json.dumps(
            _fundamentals_payload(
                fcf_latest=12.0,
                cfo_latest=20.0,
                net_debt_latest=80.0,
                fcf_slope="UNKNOWN",
                dilution="UNKNOWN",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = write_value_gates_for_run(run_id=run_id, output_dir=run_dir)
    assert summary["summary"]["counts"]["PASS"] == 1
    assert summary["summary"]["counts"]["WATCH"] == 1
    assert summary["summary"]["counts"]["FAIL"] == 0
    assert (run_dir / "value_gates.json").exists()
    assert (run_dir / "value_gates_calibration.json").exists()

    calibration_payload = json.loads((run_dir / "value_gates_calibration.json").read_text(encoding="utf-8"))
    assert set(calibration_payload.keys()) >= {
        "counts",
        "blocker_histogram",
        "missing_input_breakdown",
        "threshold_summary",
        "what_would_flip",
    }
    assert set((calibration_payload.get("counts") or {}).keys()) == {"PASS", "WATCH", "FAIL"}
    assert isinstance(calibration_payload.get("what_would_flip"), list)

    opened = open_value_gates_for_run(run_id=run_id, top_n=10)
    assert opened["status"] == "OK"
    assert opened["counts"]["PASS"] == 1
    calibration_opened = open_value_gates_calibration_for_run(run_id=run_id, top_n=10)
    assert calibration_opened["status"] == "OK"
    assert calibration_opened["counts"]["PASS"] == 1
    assert "threshold_summary" in calibration_opened
    assert isinstance(calibration_opened["top_blockers"], list)


def test_threshold_override_changes_status_deterministically(monkeypatch, tmp_path) -> None:
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "value_gates_override_test"
    run_dir = cfg.sectors_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _seed_run_single_ticker(
        run_dir=run_dir,
        ticker="AAA",
        intrinsic_per_share=120.0,  # valuation_gap = 0.20
        current_price=100.0,
        fcf_latest=20.0,
        shares_outstanding=100.0,
        net_debt=50.0,
        cfo_value=40.0,
        fcf_slope=0.01,
        dilution=0.01,
    )

    baseline = write_value_gates_for_run(run_id=run_id, output_dir=run_dir)
    baseline_entry = baseline["entries"][0]
    assert baseline_entry["gate_status"] == WATCH
    assert baseline["threshold_summary"]["mos_min"] == 0.30

    overridden = write_value_gates_for_run(
        run_id=run_id,
        output_dir=run_dir,
        threshold_overrides={"mos_min": 0.15},
    )
    overridden_entry = overridden["entries"][0]
    assert overridden_entry["gate_status"] == PASS
    assert overridden["threshold_summary"]["mos_min"] == 0.15
    assert overridden["summary"]["counts"]["PASS"] == 1


# ── type holes: NaN and bool are not numbers ──────────────────────────────────
# ``_is_num`` was a bare isinstance, so NaN, inf
# and True all validated as gate inputs. NaN passes no ordered comparison, so a
# NaN slope fell through the trend test into PASS and a NaN intrinsic value fell
# through both margin-of-safety bands into a FAIL verdict — a silent verdict on
# an unusable number rather than a missing input. Same shape as the fix landed
# in app/valuation/guards.py.


def _gate_entry(*, intrinsic: object, fcf_slope: object) -> dict:
    return classify_value_gate_ticker(
        ticker="AAA",
        valuation={
            "intrinsic_per_share_base": intrinsic,
            "input_snapshot": {
                "current_price": 100.0,
                "fcf_latest": 20.0,
                "shares_outstanding": 100.0,
                "net_debt": 60.0,
                "cfo_value": 40.0,
            },
        },
        fundamentals=_fundamentals_payload(
            fcf_latest=20.0,
            cfo_latest=40.0,
            net_debt_latest=60.0,
            fcf_slope=fcf_slope,
            dilution=0.01,
        ),
        valuation_coverage_entry={
            "ticker": "AAA",
            "price_status": "OK",
            "shares_status": "OK",
            "fcf_status": "OK",
        },
    )


def test_nan_intrinsic_value_is_a_missing_input_not_a_margin_of_safety_failure() -> None:
    entry = _gate_entry(intrinsic=float("nan"), fcf_slope=0.01)
    assert entry["gates"]["mos"]["status"] == WATCH
    assert entry["gates"]["mos"]["reason"] == "MISSING_INPUT_VALUATION"
    assert entry["valuation_gap"] == "UNKNOWN"


def test_nan_trend_slope_does_not_pass_the_cash_earning_power_gate() -> None:
    entry = _gate_entry(intrinsic=150.0, fcf_slope=float("nan"))
    assert entry["gates"]["cash_earning_power"]["status"] == WATCH
    assert entry["gates"]["cash_earning_power"]["reason"] == "FCF_POSITIVE_TREND_UNKNOWN"


def test_boolean_inputs_are_not_gate_numbers() -> None:
    entry = _gate_entry(intrinsic=True, fcf_slope=True)
    assert entry["gates"]["mos"]["status"] == WATCH
    assert entry["gates"]["mos"]["reason"] == "MISSING_INPUT_VALUATION"
    assert entry["gates"]["cash_earning_power"]["status"] == WATCH
