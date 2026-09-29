from __future__ import annotations

import json

from typer.testing import CliRunner

from app.cli import app
from app.db import init_db
from app.valuation.fundamental_regression_analytics import (
    FCF_SOURCE_BRIDGE,
    FCF_SOURCE_DIRECT,
    SHARES_SOURCE_DEI,
    SHARES_SOURCE_US_GAAP,
    TOTAL_DEBT_SOURCE_COMPONENT_SUM,
    TOTAL_DEBT_SOURCE_DIRECT,
    open_fundamental_regression_analytics,
    write_fundamental_regression_analytics_for_run,
)


runner = CliRunner()


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _score_row(
    *,
    ticker: str,
    shares_status: str,
    fcf_status: str,
    facts_status: str,
    price: float | str,
    shares: float | str,
    market_cap: float | str,
    cfo: float | str,
    capex: float | str,
    fcf: float | str,
    net_debt: float | str,
    ev: float | str,
    shares_refs: list[str],
    fcf_refs: list[str],
) -> dict:
    return {
        "ticker": ticker,
        "shares_status": shares_status,
        "fcf_status": fcf_status,
        "facts_status": facts_status,
        "derived_from": list(dict.fromkeys(shares_refs + fcf_refs)),
        "inputs_used": {
            "current_price": {"value": price, "derived_from": [f"prices.{ticker}"]},
            "shares_outstanding": {"value": shares, "derived_from": shares_refs},
            "market_cap": {"value": market_cap, "derived_from": [f"market_cap.{ticker}"]},
            "cfo_value": {"value": cfo, "derived_from": [ref for ref in fcf_refs if "OperatingActivities" in ref]},
            "capex_value": {
                "value": capex,
                "derived_from": [ref for ref in fcf_refs if "PaymentsToAcquirePropertyPlantAndEquipment" in ref],
            },
            "fcf_value": {"value": fcf, "derived_from": fcf_refs},
            "net_debt_proxy": {"value": net_debt, "derived_from": [f"net_debt.{ticker}"]},
            "enterprise_value": {"value": ev, "derived_from": [f"ev.{ticker}"]},
        },
    }


def _net_debt_payload(
    *,
    total_debt_value: float | str,
    total_debt_tag: str | None,
    total_debt_date: str | None,
    cash_value: float | str,
    cash_tag: str | None,
    cash_date: str | None,
    net_debt_proxy: float | str,
) -> dict:
    return {
        "ticker": "IGNORED",
        "total_debt": {
            "value": total_debt_value,
            "tag": total_debt_tag,
            "date": total_debt_date,
            "derived_from": [],
        },
        "cash_equivalents": {
            "value": cash_value,
            "tag": cash_tag,
            "date": cash_date,
            "derived_from": [],
        },
        "net_debt_proxy": net_debt_proxy,
        "derived_from": ["net_debt.coverage"],
    }


def test_write_fundamental_regression_analytics_for_run(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "fundamental_regression_test"
    rows = [
        _score_row(
            ticker="AAA",
            shares_status="OK",
            fcf_status="OK",
            facts_status="OK",
            price=10.0,
            shares=20.0,
            market_cap=200.0,
            cfo=40.0,
            capex=10.0,
            fcf=30.0,
            net_debt=50.0,
            ev=250.0,
            shares_refs=["companyfacts.dei.EntityCommonStockSharesOutstanding[end_date=2025-12-31,unit=shares]"],
            fcf_refs=["companyfacts.us-gaap.FreeCashFlow[end_date=2025-12-31,unit=USD]"],
        ),
        _score_row(
            ticker="BBB",
            shares_status="OK",
            fcf_status="OK",
            facts_status="OK",
            price=5.0,
            shares=10.0,
            market_cap=40.0,
            cfo=20.0,
            capex=4.0,
            fcf=10.0,
            net_debt=25.0,
            ev=70.0,
            shares_refs=["companyfacts.us-gaap.CommonStockSharesOutstanding[end_date=2025-09-30,unit=shares]"],
            fcf_refs=[
                "companyfacts.us-gaap.NetCashProvidedByUsedInOperatingActivities[end_date=2025-09-30,unit=USD]",
                "companyfacts.us-gaap.PaymentsToAcquirePropertyPlantAndEquipment[end_date=2025-09-30,unit=USD]",
            ],
        ),
        _score_row(
            ticker="CCC",
            shares_status="UNKNOWN",
            fcf_status="UNKNOWN",
            facts_status="UNKNOWN",
            price=8.0,
            shares="UNKNOWN",
            market_cap="UNKNOWN",
            cfo="UNKNOWN",
            capex="UNKNOWN",
            fcf="UNKNOWN",
            net_debt="UNKNOWN",
            ev="UNKNOWN",
            shares_refs=[],
            fcf_refs=[],
        ),
    ]
    net_debt_map = {
        "AAA": _net_debt_payload(
            total_debt_value=80.0,
            total_debt_tag="DebtLongtermAndShorttermCombinedAmount",
            total_debt_date="2025-12-31",
            cash_value=30.0,
            cash_tag="CashAndCashEquivalentsAtCarryingValue",
            cash_date="2025-12-31",
            net_debt_proxy=50.0,
        ),
        "BBB": _net_debt_payload(
            total_debt_value=40.0,
            total_debt_tag="DebtCurrent_plus_LongTermDebtNoncurrent",
            total_debt_date="2025-09-30",
            cash_value=10.0,
            cash_tag="CashAndCashEquivalentsAtCarryingValue",
            cash_date="2024-12-31",
            net_debt_proxy=25.0,
        ),
        "CCC": _net_debt_payload(
            total_debt_value="UNKNOWN",
            total_debt_tag=None,
            total_debt_date=None,
            cash_value="UNKNOWN",
            cash_tag=None,
            cash_date=None,
            net_debt_proxy="UNKNOWN",
        ),
    }

    payload = write_fundamental_regression_analytics_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA", "BBB", "CCC"],
        output_path=cfg.outputs_dir / "universe" / run_id / "fundamental_regression_analytics.json",
        markdown_path=cfg.outputs_dir / "universe" / run_id / "fundamental_regression_analytics.md",
        scoreboard_rows=rows,
        net_debt_resolved_by_ticker=net_debt_map,
    )

    assert payload["counts_by_shares_source_class"][SHARES_SOURCE_DEI] == 1
    assert payload["counts_by_shares_source_class"][SHARES_SOURCE_US_GAAP] == 1
    assert payload["counts_by_fcf_source_class"][FCF_SOURCE_DIRECT] == 1
    assert payload["counts_by_fcf_source_class"][FCF_SOURCE_BRIDGE] == 1
    assert payload["counts_by_total_debt_source_class"][TOTAL_DEBT_SOURCE_DIRECT] == 1
    assert payload["counts_by_total_debt_source_class"][TOTAL_DEBT_SOURCE_COMPONENT_SUM] == 1
    assert payload["unknown_input_counts"]["shares_unknown_count"] == 1
    assert payload["formula_mismatch_counts"]["MARKET_CAP_FORMULA_MISMATCH"] == 1
    assert payload["formula_mismatch_counts"]["ENTERPRISE_VALUE_FORMULA_MISMATCH"] == 1
    assert payload["formula_mismatch_counts"]["FCF_BRIDGE_MISMATCH"] == 1
    assert payload["formula_mismatch_counts"]["NET_DEBT_BRIDGE_MISMATCH"] == 1
    assert payload["debt_cash_date_alignment_counts"]["DIFFERENT_DATES"] == 1
    assert any(breach["metric"] == "shares_unknown_rate" for breach in payload["threshold_breaches"])
    assert any(row["ticker"] == "BBB" for row in payload["top_10_flagged_tickers"])
    assert (cfg.outputs_dir / "universe" / run_id / "fundamental_regression_analytics.md").exists()


def test_open_fundamental_regression_analytics(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "fundamental_regression_open"
    write_fundamental_regression_analytics_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA"],
        output_path=cfg.outputs_dir / "universe" / run_id / "fundamental_regression_analytics.json",
        scoreboard_rows=[
            _score_row(
                ticker="AAA",
                shares_status="OK",
                fcf_status="OK",
                facts_status="OK",
                price=10.0,
                shares=20.0,
                market_cap=200.0,
                cfo=40.0,
                capex=10.0,
                fcf=30.0,
                net_debt=50.0,
                ev=250.0,
                shares_refs=["companyfacts.dei.EntityCommonStockSharesOutstanding[end_date=2025-12-31,unit=shares]"],
                fcf_refs=["companyfacts.us-gaap.FreeCashFlow[end_date=2025-12-31,unit=USD]"],
            )
        ],
        net_debt_resolved_by_ticker={
            "AAA": _net_debt_payload(
                total_debt_value=80.0,
                total_debt_tag="DebtLongtermAndShorttermCombinedAmount",
                total_debt_date="2025-12-31",
                cash_value=30.0,
                cash_tag="CashAndCashEquivalentsAtCarryingValue",
                cash_date="2025-12-31",
                net_debt_proxy=50.0,
            )
        },
    )

    payload = open_fundamental_regression_analytics(run_id=run_id, top_n=5)
    assert payload["status"] == "OK"
    assert payload["ticker_count"] == 1
    assert payload["counts_by_shares_source_class"][SHARES_SOURCE_DEI] == 1


def test_fundamental_regression_cli_open(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    run_id = "fundamental_regression_cli"
    write_fundamental_regression_analytics_for_run(
        run_id=run_id,
        as_of_date="2026-02-14",
        tickers=["AAA"],
        output_path=cfg.outputs_dir / "universe" / run_id / "fundamental_regression_analytics.json",
        scoreboard_rows=[
            _score_row(
                ticker="AAA",
                shares_status="OK",
                fcf_status="OK",
                facts_status="OK",
                price=10.0,
                shares=20.0,
                market_cap=200.0,
                cfo=40.0,
                capex=10.0,
                fcf=30.0,
                net_debt=50.0,
                ev=250.0,
                shares_refs=["companyfacts.dei.EntityCommonStockSharesOutstanding[end_date=2025-12-31,unit=shares]"],
                fcf_refs=["companyfacts.us-gaap.FreeCashFlow[end_date=2025-12-31,unit=USD]"],
            )
        ],
        net_debt_resolved_by_ticker={
            "AAA": _net_debt_payload(
                total_debt_value=80.0,
                total_debt_tag="DebtLongtermAndShorttermCombinedAmount",
                total_debt_date="2025-12-31",
                cash_value=30.0,
                cash_tag="CashAndCashEquivalentsAtCarryingValue",
                cash_date="2025-12-31",
                net_debt_proxy=50.0,
            )
        },
    )

    result = runner.invoke(app, ["universe-fundamental-regression-open", "--run-id", run_id])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "OK"
    assert payload["counts_by_shares_source_class"][SHARES_SOURCE_DEI] == 1
