"""Date alignment of debt, cash and cash flow.

Debt, cash and cash flow must come from one balance-sheet period. Policy (conservative): a debt/cash date mismatch is flagged and capped below HIGH, and refused
(UNKNOWN, DATELINE_MISMATCH) when more than one quarter (95 days) apart; a cash
flow more than one fiscal year older than the balance sheet gives an UNKNOWN ratio.
"""

from __future__ import annotations

import json

from app.db import init_db
from app.valuation.balance_sheet_stress import compute_balance_sheet_stress
from app.valuation.net_debt import resolve_net_debt_proxy


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _resolve_with_facts(cfg, monkeypatch, tmp_path, payload: dict) -> dict:
    path = tmp_path / "cf.json"
    path.write_text(json.dumps({"companyfacts": payload}), encoding="utf-8")
    return resolve_net_debt_proxy(
        "DAT",
        "2026-02-14",
        facts_row={"ticker": "DAT", "cache_path": str(path), "derived_from": []},
        cfg=cfg,
    )


def _instant(end: str, value: float, form: str = "10-Q") -> dict:
    return {"units": {"USD": [{"end": end, "filed": "2026-01-31", "form": form, "val": value}]}}


def test_debt_and_cash_from_different_balance_sheets_are_not_high_confidence(monkeypatch, tmp_path):
    """Debt 200 + 800 at 2025-09-30; the only cash fact is 100 at 2024-12-31.

    Observed on main: net debt 900 at HIGH confidence, no flag. 1,000 of debt nine
    months newer than the 100 of cash is not one balance sheet. Correct: the
    mismatch is named (a flag containing DATELINE_MISMATCH for the cash operand)
    and the result is not HIGH confidence.
    """
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = {
        "cik": 901,
        "entityName": "Dateline Cash Co",
        "facts": {
            "us-gaap": {
                "DebtCurrent": _instant("2025-09-30", 200_000_000.0),
                "LongTermDebtNoncurrent": _instant("2025-09-30", 800_000_000.0),
                "CashAndCashEquivalentsAtCarryingValue": _instant(
                    "2024-12-31", 100_000_000.0, "10-K"
                ),
            }
        },
    }
    out = _resolve_with_facts(cfg, monkeypatch, tmp_path, payload)
    assert out["net_debt_confidence"] != "HIGH"
    assert any(
        "DATELINE_MISMATCH" in flag and not flag.startswith("LEASE_")
        for flag in out["net_debt_flags"]
    )


def test_an_unreadable_lease_does_not_keep_high_confidence(monkeypatch, tmp_path):
    """Debt 1,000 and cash 100 on one date, no operating-lease fact at all.

    net_debt_proxy is the lease-INCLUSIVE leverage basis; with the lease unknown
    it is really 900 lease-exclusive (flagged LEASE_LIABILITY_UNKNOWN). Observed:
    confidence HIGH. Correct: a basis it did not measure is not HIGH confidence.
    """
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = {
        "cik": 902,
        "entityName": "No Lease Co",
        "facts": {
            "us-gaap": {
                "DebtCurrent": _instant("2025-09-30", 200_000_000.0),
                "LongTermDebtNoncurrent": _instant("2025-09-30", 800_000_000.0),
                "CashAndCashEquivalentsAtCarryingValue": _instant("2025-09-30", 100_000_000.0),
            }
        },
    }
    out = _resolve_with_facts(cfg, monkeypatch, tmp_path, payload)
    assert out["net_debt_proxy"] == 900.0
    assert "LEASE_LIABILITY_UNKNOWN" in out["net_debt_flags"]
    assert out["net_debt_confidence"] != "HIGH"


def test_current_net_debt_is_not_divided_by_a_six_year_old_cash_flow():
    """Net debt 2,000 on the 2025-12-31 balance sheet; the only CFO row is FY2019 (500).

    Observed: net_debt_to_cfo 4.0 and HIGH_NET_DEBT_TO_CFO. The 2019 cash flow says
    nothing about servicing 2025 debt. Policy (conservative): a CFO more than one
    fiscal year older than the net-debt balance sheet cannot form the ratio, so it
    is UNKNOWN and the high-leverage signal is not fired from it.
    """
    payload = compute_balance_sheet_stress(
        "AAA",
        "2026-03-01",
        fundamentals={"rows": [{"year": 2019, "cfo": 500.0}]},
        net_debt_payload={
            "net_debt_proxy": 2000.0,
            "total_debt": {"value": 2200.0, "period_end": "2025-12-31"},
            "cash_equivalents": {"value": 200.0, "period_end": "2025-12-31"},
        },
        facts_status="OK",
    )
    assert payload["net_debt_to_cfo"] == "UNKNOWN"
    assert "HIGH_NET_DEBT_TO_CFO" not in payload["balance_sheet_headwind_signals"]


def test_debt_and_cash_more_than_a_fiscal_year_apart_are_refused(monkeypatch, tmp_path):
    """Debt at 2025-09-30, cash at 2023-12-31 (639 days): UNKNOWN, never netted."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = {
        "cik": 903,
        "entityName": "Stale Cash Co",
        "facts": {
            "us-gaap": {
                "DebtCurrent": _instant("2025-09-30", 200_000_000.0),
                "LongTermDebtNoncurrent": _instant("2025-09-30", 800_000_000.0),
                "CashAndCashEquivalentsAtCarryingValue": _instant(
                    "2023-12-31", 100_000_000.0, "10-K"
                ),
            }
        },
    }
    out = _resolve_with_facts(cfg, monkeypatch, tmp_path, payload)
    assert out["status"] == "UNKNOWN"
    assert out["reason_code"] == "DATELINE_MISMATCH"
    assert out["net_debt_proxy"] == "UNKNOWN"
    assert out["net_debt_confidence"] == "LOW"
    assert "DATELINE_MISMATCH:cash=2023-12-31:debt=2025-09-30" in out["net_debt_flags"]


def _dated_payload(*, debt_end: str, cash_end: str) -> dict:
    return {
        "cik": 904,
        "entityName": "Gap Co",
        "facts": {
            "us-gaap": {
                "DebtCurrent": _instant(debt_end, 200_000_000.0),
                "LongTermDebtNoncurrent": _instant(debt_end, 800_000_000.0),
                "CashAndCashEquivalentsAtCarryingValue": _instant(cash_end, 100_000_000.0),
            }
        },
    }


def test_debt_and_cash_are_netted_only_within_one_quarter(monkeypatch, tmp_path):
    """The allowed gap was one fiscal year (366 days): debt at 2025-09-30 was netted
    against cash from 2025-03-31, half a year apart, at MEDIUM confidence. It is one quarter
    now: 92 days apart is netted (named, MEDIUM); 183 days apart is refused."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    near = _resolve_with_facts(
        cfg, monkeypatch, tmp_path, _dated_payload(debt_end="2025-09-30", cash_end="2025-06-30")
    )
    assert (near["status"], near["net_debt_proxy"], near["net_debt_confidence"]) == (
        "OK",
        900.0,
        "MEDIUM",
    )
    assert "DATELINE_MISMATCH:cash=2025-06-30:debt=2025-09-30" in near["net_debt_flags"]
    far = _resolve_with_facts(
        cfg, monkeypatch, tmp_path, _dated_payload(debt_end="2025-09-30", cash_end="2025-03-31")
    )
    assert (far["status"], far["reason_code"], far["net_debt_proxy"]) == (
        "UNKNOWN",
        "DATELINE_MISMATCH",
        "UNKNOWN",
    )


def test_a_stale_debt_fact_is_dropped_not_shown_as_the_total_debt(monkeypatch, tmp_path):
    """JPMorgan's freshest undimensioned total debt is from 2014 beside 2026 cash, and the
    payload showed 368 billion of 2014 debt as its total debt. A debt fact older than the
    cash's balance sheet by more than the allowed gap is dropped: total debt UNKNOWN,
    reason DATELINE_MISMATCH, flagged STALE_DEBT_DROPPED -- and never an evidenced zero."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    out = _resolve_with_facts(
        cfg, monkeypatch, tmp_path, _dated_payload(debt_end="2014-06-30", cash_end="2025-09-30")
    )
    assert out["total_debt"]["value"] == "UNKNOWN"
    assert (out["status"], out["reason_code"], out["net_debt_confidence"]) == (
        "UNKNOWN",
        "DATELINE_MISMATCH",
        "LOW",
    )
    assert out["net_debt_flags"] == ["STALE_DEBT_DROPPED:debt=2014-06-30:cash=2025-09-30"]
    assert "total_debt_evidence" not in out
