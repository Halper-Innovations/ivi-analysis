from __future__ import annotations

import json
from pathlib import Path

from app.db import init_db, utc_now_iso
from app.valuation.net_debt import (
    REASON_BANK_SPECIFIC_HANDLING,
    REASON_MISSING_BOTH,
    REASON_MISSING_CASH,
    REASON_MISSING_DEBT,
    REASON_OK,
    resolve_net_debt_proxy,
)


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


def _fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / "companyfacts" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_companyfacts_cache(cfg, *, cik: str, fixture_name: str) -> Path:
    payload = _fixture(fixture_name)
    path = cfg.cache_dir / "companyfacts" / f"{str(cik).zfill(10)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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
    return path


def test_net_debt_resolver_ok_computes_with_tag_traces(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_path = _write_companyfacts_cache(cfg, cik="100", fixture_name="NET_DEBT_OK")
    facts_row = {
        "ticker": "AAA",
        "cache_path": str(cache_path),
        "derived_from": ["facts.AAA"],
    }
    result = resolve_net_debt_proxy(
        "AAA", "2026-02-14", run_id="nd_ok", facts_row=facts_row, cfg=cfg
    )
    assert result["status"] == "OK"
    assert result["reason_code"] == REASON_OK
    assert result["net_debt_proxy"] == 240.0
    refs = [str(ref) for ref in (result.get("derived_from") or [])]
    assert any("DebtCurrent" in ref or "LongTermDebtNoncurrent" in ref for ref in refs)
    assert any("end_date=2025-12-31" in ref for ref in refs)


def test_net_debt_resolver_missing_component_reason_codes(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    debt_only_path = _write_companyfacts_cache(cfg, cik="101", fixture_name="NET_DEBT_DEBT_ONLY")
    cash_only_path = _write_companyfacts_cache(cfg, cik="102", fixture_name="NET_DEBT_CASH_ONLY")
    both_missing_path = _write_companyfacts_cache(cfg, cik="103", fixture_name="BBB")

    debt_only = resolve_net_debt_proxy(
        "DDD",
        "2026-02-14",
        run_id="nd_missing",
        facts_row={
            "ticker": "DDD",
            "cache_path": str(debt_only_path),
            "derived_from": ["facts.DDD"],
        },
        cfg=cfg,
    )
    cash_only = resolve_net_debt_proxy(
        "CCC",
        "2026-02-14",
        run_id="nd_missing",
        facts_row={
            "ticker": "CCC",
            "cache_path": str(cash_only_path),
            "derived_from": ["facts.CCC"],
        },
        cfg=cfg,
    )
    both_missing = resolve_net_debt_proxy(
        "MMM",
        "2026-02-14",
        run_id="nd_missing",
        facts_row={
            "ticker": "MMM",
            "cache_path": str(both_missing_path),
            "derived_from": ["facts.MMM"],
        },
        cfg=cfg,
    )
    assert debt_only["status"] == "UNKNOWN"
    assert debt_only["reason_code"] == REASON_MISSING_CASH
    assert debt_only["net_debt_proxy"] == "UNKNOWN"
    assert cash_only["status"] == "UNKNOWN"
    assert cash_only["reason_code"] == REASON_MISSING_DEBT
    assert cash_only["net_debt_proxy"] == "UNKNOWN"
    assert cash_only["net_debt_confidence"] == "LOW"
    assert both_missing["status"] == "UNKNOWN"
    assert both_missing["reason_code"] == REASON_MISSING_BOTH
    assert both_missing["net_debt_proxy"] == "UNKNOWN"
    assert both_missing["net_debt_confidence"] == "LOW"


def test_net_debt_resolver_flags_bank_specific_handling(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_path = _write_companyfacts_cache(cfg, cik="19617", fixture_name="BANK_FINANCIAL")
    result = resolve_net_debt_proxy(
        "JPM",
        "2026-02-13",
        run_id="nd_bank",
        facts_row={"ticker": "JPM", "cache_path": str(cache_path), "derived_from": ["facts.JPM"]},
        cfg=cfg,
    )
    assert result["status"] == "UNKNOWN"
    assert result["reason_code"] == REASON_BANK_SPECIFIC_HANDLING
    assert result["issuer_classification"] == "financial"
    assert result["net_debt_proxy"] == "UNKNOWN"


def test_net_debt_resolver_missing_debt_tags_fail_closed(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_path = _write_companyfacts_cache(cfg, cik="104", fixture_name="NET_DEBT_CASH_ONLY")
    result = resolve_net_debt_proxy(
        "EEE",
        "2026-02-14",
        run_id="nd_estimated_zero",
        facts_row={"ticker": "EEE", "cache_path": str(cache_path), "derived_from": ["facts.EEE"]},
        cfg=cfg,
    )
    assert result["status"] == "UNKNOWN"
    assert result["reason_code"] == REASON_MISSING_DEBT
    assert result["net_debt_proxy"] == "UNKNOWN"
    assert result["net_debt_confidence"] == "LOW"
    assert result["total_debt"]["value"] == "UNKNOWN"
    assert result["total_debt"]["estimated_zero"] is False
    assert result["net_debt_resolution"] == "UNAVAILABLE"


def test_net_debt_resolver_missing_debt_and_cash_stay_missing(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    cache_path = cfg.cache_dir / "companyfacts" / "0000000105.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cik": "0000000105",
                "retrieved_at": utc_now_iso(),
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000105.json",
                "http_status": 200,
                "size_bytes": 2,
                "companyfacts": {"entityName": "No Debt Tags Co", "facts": {"us-gaap": {}}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = resolve_net_debt_proxy(
        "FFF",
        "2026-02-14",
        run_id="nd_estimated_zero_both_missing",
        facts_row={"ticker": "FFF", "cache_path": str(cache_path), "derived_from": ["facts.FFF"]},
        cfg=cfg,
    )
    assert result["status"] == "UNKNOWN"
    assert result["reason_code"] == REASON_MISSING_BOTH
    assert result["net_debt_proxy"] == "UNKNOWN"
    assert result["net_debt_confidence"] == "LOW"
    assert result["total_debt"]["value"] == "UNKNOWN"
    assert result["total_debt"]["estimated_zero"] is False
    assert result["cash_equivalents"]["value"] == "UNKNOWN"
    assert result["net_debt_resolution"] == "UNAVAILABLE"


def _write_raw_companyfacts(cfg, *, cik: str, payload: dict) -> Path:
    from app.db import utc_now_iso as _now

    path = cfg.cache_dir / "companyfacts" / f"{str(cik).zfill(10)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cik": str(cik).zfill(10),
                "retrieved_at": _now(),
                "source_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json",
                "http_status": 200,
                "size_bytes": 2,
                "companyfacts": payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_net_debt_resolver_accepts_only_reported_zero_operands(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    debt_zero_path = _write_raw_companyfacts(
        cfg,
        cik="106",
        payload={
            "entityName": "REPORTED ZERO DEBT INC",
            "facts": {
                "us-gaap": {
                    "LongTermDebt": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 0.0,
                                    "filed": "2026-01-20",
                                    "form": "10-K",
                                    "accn": "zero-debt-1",
                                }
                            ]
                        }
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 95_000_000.0,
                                    "filed": "2026-01-20",
                                    "form": "10-K",
                                    "accn": "zero-debt-1",
                                }
                            ]
                        }
                    },
                }
            },
        },
    )
    cash_zero_path = _write_raw_companyfacts(
        cfg,
        cik="107",
        payload={
            "entityName": "REPORTED ZERO CASH INC",
            "facts": {
                "us-gaap": {
                    "LongTermDebt": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 90_000_000.0,
                                    "filed": "2026-01-20",
                                    "form": "10-K",
                                    "accn": "zero-cash-1",
                                }
                            ]
                        }
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 0.0,
                                    "filed": "2026-01-20",
                                    "form": "10-K",
                                    "accn": "zero-cash-1",
                                }
                            ]
                        }
                    },
                }
            },
        },
    )

    debt_zero = resolve_net_debt_proxy(
        "ZD",
        "2026-02-14",
        facts_row={
            "ticker": "ZD",
            "cache_path": str(debt_zero_path),
            "derived_from": ["facts.ZD"],
        },
        cfg=cfg,
    )
    cash_zero = resolve_net_debt_proxy(
        "ZC",
        "2026-02-14",
        facts_row={
            "ticker": "ZC",
            "cache_path": str(cash_zero_path),
            "derived_from": ["facts.ZC"],
        },
        cfg=cfg,
    )

    assert debt_zero["status"] == "OK"
    assert debt_zero["total_debt"]["value"] == 0.0
    assert debt_zero["cash_equivalents"]["value"] == 95.0
    assert debt_zero["net_debt_proxy"] == -95.0
    assert debt_zero["net_debt_confidence"] == "MEDIUM"
    assert debt_zero["total_debt"]["filed_date"] == "2026-01-20"
    assert cash_zero["status"] == "OK"
    assert cash_zero["total_debt"]["value"] == 90.0
    assert cash_zero["cash_equivalents"]["value"] == 0.0
    assert cash_zero["net_debt_proxy"] == 90.0
    assert cash_zero["net_debt_confidence"] == "MEDIUM"


def test_net_debt_resolver_adds_operating_lease_liability_lease_adjusted(monkeypatch, tmp_path):
    """FIX 1: net-debt proxy includes operating lease liability and flags LEASE_ADJUSTED,
    matching the inline valuation_writer path."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    # Raw USD values (whole dollars): debt 300M, lease 100M, cash 60M -> net 340M.
    payload = {
        "entityName": "LEASE ADJ INC",
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
    cache_path = _write_raw_companyfacts(cfg, cik="900", payload=payload)
    result = resolve_net_debt_proxy(
        "LSE",
        "2026-02-14",
        run_id="nd_lease",
        facts_row={"ticker": "LSE", "cache_path": str(cache_path), "derived_from": ["facts.LSE"]},
        cfg=cfg,
    )
    assert result["status"] == "OK"
    assert result["reason_code"] == REASON_OK
    # 300M debt + 100M lease - 60M cash = 340M
    assert result["net_debt_proxy"] == 340.0
    assert "LEASE_ADJUSTED" in (result.get("net_debt_flags") or [])
    assert result["operating_lease_liability"]["value"] == 100.0


def test_net_debt_resolver_micro_cap_sub_100k_balance(monkeypatch, tmp_path):
    """FIX 2: a genuine sub-$100k raw-USD balance is scaled to $millions (÷1e6),
    not passed through as if already in millions."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = {
        "entityName": "MICRO CAP INC",
        "facts": {
            "us-gaap": {
                "LongTermDebt": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": 90_000.0,
                                "filed": "2026-01-20",
                                "form": "10-K",
                                "accn": "m-1",
                            }
                        ]
                    }
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": 50_000.0,
                                "filed": "2026-01-20",
                                "form": "10-K",
                                "accn": "m-1",
                            }
                        ]
                    }
                },
            }
        },
    }
    cache_path = _write_raw_companyfacts(cfg, cik="901", payload=payload)
    result = resolve_net_debt_proxy(
        "MIC",
        "2026-02-14",
        run_id="nd_micro",
        facts_row={"ticker": "MIC", "cache_path": str(cache_path), "derived_from": ["facts.MIC"]},
        cfg=cfg,
    )
    # $90,000 debt / 1e6 = 0.09 ; $50,000 cash / 1e6 = 0.05 ; net = 0.04 ($millions)
    assert result["total_debt"]["value"] == 0.09
    assert result["cash_equivalents"]["value"] == 0.05
    assert abs(result["net_debt_proxy"] - 0.04) < 1e-9


# ── the lease operand shares the debt's balance-sheet date, or says it does not
# An unreadable operating-lease liability was booked as a
# real zero, and the three operands were each resolved as "the freshest fact at
# or before the as-of date" independently, so the lease could come off a
# different balance sheet than the debt and cash it is added to.


def _lease_dateline_facts(*, lease_end: str) -> dict:
    def instant(tag: str, end: str, value: float) -> dict:
        return {
            "units": {
                "USD": [{"end": end, "filed": "2026-01-31", "form": "10-Q", "val": value}]
            }
        }

    return {
        "cik": 900,
        "entityName": "Dateline Co",
        "facts": {
            "us-gaap": {
                "DebtCurrent": instant("DebtCurrent", "2025-09-30", 200_000_000.0),
                "LongTermDebtNoncurrent": instant(
                    "LongTermDebtNoncurrent", "2025-09-30", 800_000_000.0
                ),
                "CashAndCashEquivalentsAtCarryingValue": instant(
                    "CashAndCashEquivalentsAtCarryingValue", "2025-09-30", 100_000_000.0
                ),
                "OperatingLeaseLiability": instant(
                    "OperatingLeaseLiability", lease_end, 500_000_000.0
                ),
            }
        },
    }


def _resolve_with_facts(cfg, monkeypatch, tmp_path, payload: dict) -> dict:
    path = tmp_path / "cf.json"
    path.write_text(json.dumps({"companyfacts": payload}), encoding="utf-8")
    return resolve_net_debt_proxy(
        "DAT",
        "2026-02-14",
        facts_row={"ticker": "DAT", "cache_path": str(path), "derived_from": []},
        cfg=cfg,
    )


def test_lease_from_the_debts_own_balance_sheet_is_included(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    out = _resolve_with_facts(
        cfg, monkeypatch, tmp_path, _lease_dateline_facts(lease_end="2025-09-30")
    )
    assert out["net_debt_proxy"] == 1400.0
    assert "LEASE_ADJUSTED" in out["net_debt_flags"]
    assert not [f for f in out["net_debt_flags"] if f.startswith("LEASE_DATELINE_MISMATCH")]


def test_a_lease_from_another_balance_sheet_date_is_named(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    out = _resolve_with_facts(
        cfg, monkeypatch, tmp_path, _lease_dateline_facts(lease_end="2024-12-31")
    )
    assert out["net_debt_proxy"] == 1400.0
    assert any(
        f.startswith("LEASE_DATELINE_MISMATCH:lease=2024-12-31:debt=2025-09-30")
        for f in out["net_debt_flags"]
    )


def test_an_unreadable_lease_is_not_a_filed_zero(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = _lease_dateline_facts(lease_end="2025-09-30")
    del payload["facts"]["us-gaap"]["OperatingLeaseLiability"]
    out = _resolve_with_facts(cfg, monkeypatch, tmp_path, payload)
    assert out["net_debt_proxy"] == 900.0
    assert "LEASE_LIABILITY_UNKNOWN" in out["net_debt_flags"]
    assert "LEASE_ADJUSTED" not in out["net_debt_flags"]


# ── issuer classification: SEC SIC code, on by default ────────────────────────
# The substring classifier reads XBRL tag NAMES and calls 71 of the top 200 liquid
# US companies financial, refusing their net-debt bridge. The SIC classifier is on
# by default; VOE_ISSUER_CLASSIFICATION_BY_SIC=false restores the substring rule.


def _operating_company_with_bank_shaped_tags() -> dict:
    def instant(end: str, value: float) -> dict:
        return {"units": {"USD": [{"end": end, "filed": "2026-01-31", "form": "10-K", "val": value}]}}

    return {
        "cik": 1045810,
        "entityName": "NVIDIA CORP",
        "facts": {
            "us-gaap": {
                "DebtCurrent": instant("2025-12-31", 100_000_000.0),
                "LongTermDebtNoncurrent": instant("2025-12-31", 900_000_000.0),
                "CashAndCashEquivalentsAtCarryingValue": instant("2025-12-31", 400_000_000.0),
                # The tags that trip the substring classifier on an operating company.
                "Deposits": instant("2025-12-31", 5_000_000.0),
                "InvestmentSecurities": instant("2025-12-31", 7_000_000.0),
            }
        },
    }


def _seed_registrant(cfg, *, cik: str, sic: int, ticker: str) -> None:
    from app.db import get_db

    with get_db(cfg) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sec_registrants(cik, primary_ticker, all_tickers, name, sic,"
            " exchange, exchange_scope, operating_status, in_scope, first_seen_at, last_seen_at)"
            " VALUES(?, ?, ?, ?, ?, 'NASDAQ', 'IN_SCOPE', 'OPERATING', 1, ?, ?)",
            (cik, ticker, json.dumps([ticker]), ticker, sic, utc_now_iso(), utc_now_iso()),
        )
        conn.commit()


def _resolve(cfg, tmp_path, payload: dict, ticker: str = "NVDA") -> dict:
    path = tmp_path / f"cf_{ticker}.json"
    path.write_text(json.dumps({"companyfacts": payload}), encoding="utf-8")
    return resolve_net_debt_proxy(
        ticker,
        "2026-02-14",
        facts_row={"ticker": ticker, "cache_path": str(path), "derived_from": []},
        cfg=cfg,
    )


def test_by_default_the_sic_classifier_decides(monkeypatch, tmp_path):
    monkeypatch.delenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", raising=False)
    cfg = _init_cfg(monkeypatch, tmp_path)
    assert cfg.issuer_classification_by_sic is True
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    out = _resolve(cfg, tmp_path, _operating_company_with_bank_shaped_tags())
    assert out["issuer_classification"] == "operating"
    assert out["issuer_classification_source"] == "sic"
    assert out["net_debt_proxy"] == 600.0


def test_with_the_sic_override_switched_off_the_substring_classifier_decides(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "false")
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    out = _resolve(cfg, tmp_path, _operating_company_with_bank_shaped_tags())
    assert out["issuer_classification"] == "financial"
    assert out["net_debt_proxy"] == "UNKNOWN"


def test_with_the_sic_override_on_a_semiconductor_maker_is_operating(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "true")
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    out = _resolve(cfg, tmp_path, _operating_company_with_bank_shaped_tags())
    assert out["issuer_classification"] == "operating"
    assert out["net_debt_proxy"] == 600.0


def test_with_the_sic_override_on_a_bank_is_still_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "true")
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0000070858", sic=6021, ticker="BAC")
    payload = _operating_company_with_bank_shaped_tags()
    payload["cik"] = 70858
    payload["entityName"] = "BANK OF AMERICA CORP"
    out = _resolve(cfg, tmp_path, payload, ticker="BAC")
    assert out["issuer_classification"] == "financial"
    assert out["net_debt_proxy"] == "UNKNOWN"


def test_a_reit_is_an_operating_issuer_under_the_sic_rule():
    from app.util.issuer_classification import classify_issuer_by_sic

    assert classify_issuer_by_sic(6798) == "operating"
    assert classify_issuer_by_sic(6500) == "operating"
    assert classify_issuer_by_sic(6021) == "financial"
    assert classify_issuer_by_sic(6712) == "financial"
    assert classify_issuer_by_sic(None) is None
    assert classify_issuer_by_sic("") is None


def test_the_override_reports_which_rule_answered(monkeypatch, tmp_path):
    """The SIC override must say when it did NOT apply.

    The lookup answered None for five different situations
    and the classifier silently fell back to the substring rule, so a run could
    have the override switched on and still be classified the old way with
    nothing in the payload saying so.
    """
    monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "true")
    cfg = _init_cfg(monkeypatch, tmp_path)
    payload = _operating_company_with_bank_shaped_tags()

    # No registrant row at all: the substring rule answers, and says why.
    out = _resolve(cfg, tmp_path, payload)
    assert out["issuer_classification"] == "financial"
    assert out["issuer_classification_source"] == "substring:NO_REGISTRANT_ROW"

    # A registrant row with no SIC on file: same shape, different reason.
    _seed_registrant(cfg, cik="0001045810", sic=None, ticker="NVDA")
    out = _resolve(cfg, tmp_path, payload)
    assert out["issuer_classification_source"] == "substring:NO_SIC_ON_FILE"

    # With a usable SIC the override answers and says so.
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    out = _resolve(cfg, tmp_path, payload)
    assert out["issuer_classification"] == "operating"
    assert out["issuer_classification_source"] == "sic"


def test_with_the_override_off_the_source_is_the_substring_rule(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "false")
    cfg = _init_cfg(monkeypatch, tmp_path)
    _seed_registrant(cfg, cik="0001045810", sic=3674, ticker="NVDA")
    out = _resolve(cfg, tmp_path, _operating_company_with_bank_shaped_tags())
    assert out["issuer_classification_source"] == "substring"
    assert out["issuer_classification"] == "financial"


# ── A coverage row with no usable facts never reads HIGH confidence ────


def test_coverage_row_without_usable_facts_is_low_confidence(monkeypatch, tmp_path):
    """The coverage writer used to default a missing confidence to HIGH, so a
    NO_FACTS row (and any resolved payload carrying no confidence of its own)
    was published as a HIGH-confidence net-debt row."""
    from app.valuation.net_debt import write_net_debt_coverage_for_run

    cfg = _init_cfg(monkeypatch, tmp_path)
    no_facts = resolve_net_debt_proxy(
        "NOF",
        "2026-02-14",
        run_id="nd_no_facts",
        facts_row={"ticker": "NOF", "derived_from": []},
        cfg=cfg,
    )
    assert no_facts["reason_code"] == "NO_FACTS"
    assert no_facts["net_debt_confidence"] == "LOW"

    payload = write_net_debt_coverage_for_run(
        run_id="nd_cov",
        as_of_date="2026-02-14",
        tickers=["NOF", "BARE"],
        output_path=tmp_path / "net_debt_coverage.json",
        resolved_by_ticker={
            "NOF": no_facts,
            "BARE": {"ticker": "BARE", "status": "UNKNOWN", "reason_code": "NO_FACTS"},
        },
        cfg=cfg,
    )
    confidence = {row["ticker"]: row["net_debt_confidence"] for row in payload["entries"]}
    assert confidence == {"BARE": "LOW", "NOF": "LOW"}


# ── The debt-only evidenced-zero policy on the as-of path ────────────────


def _debt_free_balance_sheet(**extra_lines: float) -> dict:
    """A 10-K whose liabilities (80,000,000) are fully accounted for by named
    non-debt lines, with cash of 30,000,000 and no debt concept."""

    def instant(value: float) -> dict:
        return {
            "units": {
                "USD": [
                    {
                        "end": "2025-12-31",
                        "val": value,
                        "filed": "2026-02-01",
                        "form": "10-K",
                        "fy": 2025,
                        "accn": "0000000900-26-000001",
                    }
                ]
            }
        }

    lines = {
        "Liabilities": 80_000_000.0,
        "AccountsPayableCurrent": 50_000_000.0,
        "AccruedLiabilitiesCurrent": 20_000_000.0,
        "OperatingLeaseLiabilityNoncurrent": 10_000_000.0,
        "CashAndCashEquivalentsAtCarryingValue": 30_000_000.0,
        **extra_lines,
    }
    return {
        "cik": 900,
        "entityName": "Debt Free Retail Co",
        "facts": {"us-gaap": {tag: instant(value) for tag, value in lines.items()}},
    }


def test_no_debt_concept_and_complete_liabilities_evidence_zero_debt(monkeypatch, tmp_path):
    """A filer with no debt fact was MISSING_DEBT even when its own balance
    sheet accounts for every liability without borrowing. Debt is now an
    evidenced zero there, and net debt is minus the cash."""
    monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "false")
    cfg = _init_cfg(monkeypatch, tmp_path)
    out = _resolve(cfg, tmp_path, _debt_free_balance_sheet(), ticker="DFREE")
    assert out["status"] == "OK"
    assert out["reason_code"] == REASON_OK
    assert out["total_debt"]["value"] == 0.0
    assert out["total_debt"]["tag"] == "EVIDENCED_ZERO_DEBT"
    assert out["net_debt_proxy_lease_exclusive"] == -30.0
    assert "DEBT_EVIDENCED_ZERO" in out["net_debt_flags"]
    assert out["total_debt_evidence"]["liabilities_components"] == [
        {"concept": "AccountsPayableCurrent", "value": 50_000_000.0},
        {"concept": "AccruedLiabilitiesCurrent", "value": 20_000_000.0},
        {"concept": "OperatingLeaseLiabilityNoncurrent", "value": 10_000_000.0},
    ]


def test_evidenced_zero_debt_is_refused_on_any_doubt(monkeypatch, tmp_path):
    """A revolver reported only under the line-of-credit family,
    liabilities the named lines do not reach, and missing cash all stay
    unknown."""
    monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "false")
    cfg = _init_cfg(monkeypatch, tmp_path)
    revolver = _debt_free_balance_sheet(LongTermLineOfCredit=5_000_000.0)
    assert _resolve(cfg, tmp_path, revolver, ticker="REVL")["reason_code"] == REASON_MISSING_DEBT

    unfooted = _debt_free_balance_sheet(Liabilities=95_000_000.0)
    assert _resolve(cfg, tmp_path, unfooted, ticker="UNFT")["reason_code"] == REASON_MISSING_DEBT

    no_cash = _debt_free_balance_sheet()
    del no_cash["facts"]["us-gaap"]["CashAndCashEquivalentsAtCarryingValue"]
    out = _resolve(cfg, tmp_path, no_cash, ticker="NOCSH")
    assert out["reason_code"] == REASON_MISSING_BOTH
    assert out["net_debt_proxy"] == "UNKNOWN"
