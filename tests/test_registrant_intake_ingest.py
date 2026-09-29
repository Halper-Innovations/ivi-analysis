"""Tests for app.universe.registrant_intake — Phase C ingest + gate at intake."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.universe.registrant_intake import ingest_new_registrants


@dataclass
class _GateResult:
    ticker: str
    as_of_date: str
    triggered_codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)


def _seed(conn: sqlite3.Connection, *, cik: str, ticker: str, sector: str | None) -> None:
    conn.execute(
        """
        INSERT INTO sec_registrants(
            cik, primary_ticker, all_tickers, name, exchange, exchange_scope,
            operating_status, in_scope, first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, 'NASDAQ', 'IN_SCOPE', 'OPERATING', 1, 'x', 'x')
        """,
        (cik, ticker, json.dumps([ticker]), f"{ticker} Co"),
    )
    if sector:
        conn.execute(
            "INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at) "
            "VALUES (?, '2026-06-11', ?, 1.0, '[]', 'x')",
            (ticker, sector),
        )


@pytest.fixture()
def ingest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    from app.db import init_db

    init_db(cfg)
    conn = sqlite3.connect(str(cfg.db_path))
    _seed(conn, cik="0000000020", ticker="GOOD", sector="industrial_tech")
    _seed(conn, cik="0000000021", ticker="NOCF", sector="telecom")
    _seed(conn, cik="0000000022", ticker="SHEL", sector="energy")
    _seed(conn, cik="0000000023", ticker="NOVA", sector="biotech")
    _seed(conn, cik="0000000024", ticker="UNCL", sector=None)  # unclassified: out of scope
    conn.commit()
    conn.close()
    yield cfg
    get_config.cache_clear()


def _fact_writer(cfg, tickers_with_facts: set[str]):
    def write(ticker: str) -> None:
        if ticker in {"NOCF"}:
            raise RuntimeError("SEC companyfacts returned HTTP 404")
        if ticker in tickers_with_facts:
            conn = sqlite3.connect(str(cfg.db_path))
            conn.execute(
                "INSERT OR IGNORE INTO companyfacts_facts"
                "(ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at) "
                "VALUES (?, 2025, 'FY', '2025-12-31', 'revenue', 100.0, 'USD', 'u', 'x')",
                (ticker,),
            )
            conn.commit()
            conn.close()
        # NOVA: call succeeds but writes nothing -> EMPTY_FACTS

    return write


def _valuation_writer(cfg, valued: set[str]):
    def write(ticker: str, as_of_date: str) -> None:
        if ticker in valued:
            conn = sqlite3.connect(str(cfg.db_path))
            conn.execute(
                "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
                "VALUES (?, ?, 'scorecard', '{}', '{}', '[]', 'x')",
                (ticker, as_of_date),
            )
            conn.commit()
            conn.close()

    return write


def _gate(quarantine: dict[str, list[str]]):
    def evaluate(ticker: str, **kwargs) -> _GateResult:
        codes = quarantine.get(ticker, [])
        return _GateResult(
            ticker=ticker,
            as_of_date=str(kwargs.get("as_of_date")),
            triggered_codes=list(codes),
            reasons=[f"QUARANTINE_STRUCTURAL:{c}" for c in codes],
            details={c: "detail" for c in codes},
        )

    return evaluate


def test_ingest_classifies_every_outcome(ingest_env) -> None:
    cfg = ingest_env
    report = ingest_new_registrants(
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        facts_fn=_fact_writer(cfg, {"GOOD", "SHEL"}),
        valuation_fn=_valuation_writer(cfg, {"GOOD", "SHEL"}),
        gate_fn=_gate({"SHEL": ["PENNY_FLOOR"]}),
        run_gate=True,
    )

    assert report["scope"] == 4  # UNCL has no sector -> out of scope
    assert report["ingested"] == 1
    assert report["quarantined"] == 1
    assert report["failed"] == 2
    assert report["status_counts"]["INGESTED"] == 1
    assert report["status_counts"]["QUARANTINE_STRUCTURAL:PENNY_FLOOR"] == 1
    assert report["status_counts"]["INGEST_FAILED:NO_COMPANYFACTS"] == 1
    assert report["status_counts"]["INGEST_FAILED:EMPTY_FACTS"] == 1
    assert report["gate_code_counts"] == {"PENNY_FLOOR": 1}
    failures = {f["ticker"]: f["intake_status"] for f in report["failures"]}
    assert failures == {
        "NOCF": "INGEST_FAILED:NO_COMPANYFACTS",
        "NOVA": "INGEST_FAILED:EMPTY_FACTS",
    }

    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    reg = {
        r["primary_ticker"]: dict(r)
        for r in conn.execute("SELECT primary_ticker, intake_status FROM sec_registrants")
    }
    log_actions = [
        tuple(r)
        for r in conn.execute(
            "SELECT action, ticker FROM universe_sync_log WHERE run_kind='ingest' ORDER BY ticker"
        )
    ]
    conn.close()
    assert reg["GOOD"]["intake_status"] == "INGESTED"
    assert reg["SHEL"]["intake_status"] == "QUARANTINE_STRUCTURAL:PENNY_FLOOR"
    assert reg["NOCF"]["intake_status"] == "INGEST_FAILED:NO_COMPANYFACTS"
    assert reg["UNCL"]["intake_status"] is None
    assert log_actions == [
        ("INGESTED", "GOOD"),
        ("FAILED", "NOCF"),
        ("FAILED", "NOVA"),
        ("QUARANTINED", "SHEL"),
    ]


def test_ingest_is_resumable_and_delta_aware(ingest_env) -> None:
    cfg = ingest_env
    kwargs = dict(
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        facts_fn=_fact_writer(cfg, {"GOOD", "SHEL"}),
        valuation_fn=_valuation_writer(cfg, {"GOOD", "SHEL"}),
        gate_fn=_gate({}),
        run_gate=True,
    )
    first = ingest_new_registrants(**kwargs)
    assert first["scope"] == 4
    assert first["ingested"] == 2
    second = ingest_new_registrants(**kwargs)
    # GOOD and SHEL became sweepable (scorecard rows) and drop out of scope;
    # the two failures remain in scope for retry.
    assert second["scope"] == 2
    assert {f["ticker"] for f in second["failures"]} == {"NOCF", "NOVA"}
