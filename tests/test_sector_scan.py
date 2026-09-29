"""Tests for app.sector.scan — sector scan pipeline."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tests.financial_integrity_helpers import canonicalize_financial_packet


# ---------------------------------------------------------------------------
# Fake Anthropic client for triage / deep review tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int = 25000
    output_tokens: int = 3000


@dataclass
class _FakeToolUseBlock:
    type: str = "tool_use"
    name: str = ""
    id: str = "tu"
    input: dict = field(default_factory=dict)


@dataclass
class _FakeMessage:
    content: list
    usage: _FakeUsage
    stop_reason: str = "tool_use"


class _ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls.append(kwargs)
            return self._outer._responses.pop(0)

    @property
    def messages(self):
        return _ScriptedClient._Messages(self)


def _triage_response(survivors: list[dict]) -> _FakeMessage:
    return _FakeMessage(
        content=[
            _FakeToolUseBlock(
                name="sector_triage",
                input={"survivors": survivors, "surprises": []},
            )
        ],
        usage=_FakeUsage(25000, 3000),
    )


def _deep_review_response(verdict="WATCH", thesis="Strong capital allocation") -> _FakeMessage:
    """Stage 4 finalize_analysis response (agentic deep review)."""
    return _FakeMessage(
        content=[
            _FakeToolUseBlock(
                name="finalize_analysis",
                input={
                    "verdict": verdict,
                    "confidence": "MODERATE",
                    "thesis": thesis,
                    "key_findings": ["FCF yield 8%"],
                    "open_questions": ["Capex sustainability?"],
                    "falsifiers": ["Cyclical downturn"],
                    "reasoning_trace": "Compared to peers, best FCF conversion",
                },
            )
        ],
        usage=_FakeUsage(8000, 2000),
    )


def _sample_bundle(ticker: str):
    from app.analyst.evidence_bundle import AnalysisEvidenceBundle, ValuationSnapshot

    return AnalysisEvidenceBundle(
        ticker=ticker,
        as_of_date="2026-04-12",
        built_at="2026-04-10T12:00:00+00:00",
        analysis_years=5,
        analysis_quarters=0,
        freshness_window_days=90,
        valuation=ValuationSnapshot(
            current_price=50.0,
            market_cap=500.0,
            dcf_base=110.0,
            epv_adjusted=100.0,
            graham_value=None,
            methods_agree=True,
            tension_type=None,
            gate_action="PROCEED",
            solvency_status="LOW",
            filing_risk_status="OK",
        ),
    )


def _integrity_scope(*tickers: str):
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        stable_quote_hash,
    )

    packets = []
    for ticker in tickers:
        snapshot_id = stable_quote_hash(
            ticker=ticker,
            price=50.0,
            as_of_date="2026-04-12",
            currency="USD",
            source="fixture_quote",
            price_basis="UNADJUSTED",
            raw_price=50.0,
            split_adjustment_factor=1.0,
        )
        packet = TickerSignalPacket(
            ticker=ticker,
            dcf_value=110.0,
            epv_value=100.0,
            current_price=50.0,
            current_price_unit="USD_per_share",
            current_price_as_of_date="2026-04-12",
            current_price_currency="USD",
            current_price_source="fixture_quote",
            quote_snapshot_id=snapshot_id,
            price_basis="UNADJUSTED",
            raw_price=50.0,
            split_adjustment_factor=1.0,
            split_lineage_proof={
                "status": "PASS",
                "period_start": "2026-04-12",
                "period_end": "2026-04-12",
                "verified_as_of": "2026-04-12",
                "source": "fixture_corporate_actions",
                "source_reference": "https://example.test/actions",
            },
            market_cap_mm=500.0,
            market_cap_unit="USD_millions",
            market_cap_source="fixture_market_cap",
            market_cap_effective_as_of_date="2026-04-12",
            market_cap_method="price_times_shares",
            shares_outstanding_mm=10.0,
            raw_shares_outstanding_mm=10.0,
            shares_unit="shares_millions",
            shares_basis="UNADJUSTED",
            shares_as_of_date="2026-04-12",
            shares_source="fixture_filing",
            issuer_quote_ratio=1.0,
            cap_stage_price=50.0,
            cap_stage_price_as_of_date="2026-04-12",
            cap_stage_price_currency="USD",
            cap_stage_price_source="fixture_quote",
            cap_stage_quote_snapshot_id=snapshot_id,
        )
        packets.append(
            canonicalize_financial_packet(
                packet,
                as_of_date="2026-04-12",
                shares_mm=10.0,
            )
        )
    return FinancialIntegrityScope(
        context="sector_scan_test",
        run_as_of_date="2026-04-12",
        packets=tuple(packets),
    )


@pytest.fixture
def scan_db(tmp_path, monkeypatch) -> Path:
    """Create a minimal engine.db with sector_inference + valuations tables."""
    db = tmp_path / "engine.db"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db))
    from app.autonomous.artifact_financial_audit import (
        AUDIT_SCHEMA_VERSION,
        AUDIT_SCOPE_ID,
        CANONICAL_AUDIT_ROOT_IDS,
    )
    from app.config import get_config
    from app.valuation.lineage import (
        valuation_integrity_fingerprint,
        valuation_source_record,
    )

    get_config.cache_clear()
    cfg = get_config()
    roots = {
        "autonomous_sector": (Path(cfg.runs_dir) / "autonomous_sector").resolve(),
        "analyst_output": Path(cfg.analyst_outputs_dir).resolve(),
        "scan": (Path(cfg.outputs_dir) / "scans").resolve(),
        "research_output": Path(cfg.research_dir).resolve(),
        "watchlist_report": (Path(cfg.outputs_dir) / "digests").resolve(),
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    run_id = "sector_scan_scorecard_fixture"
    issuer_ciks = {
        "NVDA": "1001",
        "AMD": "1002",
        "INTC": "1003",
        "MSFT": "1004",
    }
    scorecard_rows = []
    valuation_source_records = []
    for ticker, price, dcf, epv in [
        ("NVDA", 120.0, 95.0, 80.0),  # overvalued
        ("AMD", 80.0, 110.0, 100.0),  # undervalued
        ("INTC", 25.0, 40.0, 35.0),  # undervalued
        ("MSFT", 400.0, 350.0, 300.0),  # different sector
    ]:
        scorecard = {
            "pricing_zone": "MOS" if dcf > price else "ABOVE_IV",
            "pricing_zone_detail": {
                "current_price": price,
                "current_price_as_of_date": "2026-04-10",
                "current_price_currency": "USD",
                "current_price_source": "fixture_quote",
                "current_price_basis": "UNADJUSTED",
                "dcf_base": dcf,
                "epv_adjusted": epv,
                "gate_action": "PROCEED",
            },
            "quality_context": {
                "gate_action": "PROCEED",
                "confidence_class": "MODERATE",
            },
        }
        scorecard_json = json.dumps(scorecard)
        scorecard_rows.append((ticker, scorecard_json))
        source_record = valuation_source_record(
            {
                "ticker": ticker,
                "as_of_date": "2026-04-10",
                "method": "scorecard",
                "inputs_json": "{}",
                "outputs_json": scorecard_json,
                "warnings_json": "[]",
                "created_at": "2026-04-12T00:00:00",
                "valuation_writer_version": "fixture",
                "quality_gate_verdict": "PROCEED",
                "confidence_class": "MODERATE",
                "gate_reason_codes": "[]",
                "valuation_headwinds": "[]",
                "valuation_supports": "[]",
                "source_run_id": run_id,
            }
        )
        assert source_record is not None
        valuation_source_records.append(source_record)
    valuation_source_records.sort(
        key=lambda record: (
            record["row"]["ticker"],
            record["row"]["as_of_date"],
            record["row"]["method"],
            record["row"]["created_at"],
        )
    )
    source_path = roots["autonomous_sector"] / run_id / "autonomous_sector_run.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "company_packets": [
                    {
                        "ticker": ticker,
                        "issuer_cik": issuer_ciks[ticker],
                        "financial_integrity_status": "PASS",
                    }
                    for ticker in ("NVDA", "AMD", "INTC", "MSFT")
                ],
                "valuation_source_records": valuation_source_records,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "financial_integrity_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "audit_scope_id": AUDIT_SCOPE_ID,
                "generated_at": "2026-04-12T12:00:00Z",
                "complete": True,
                "source_roots": [
                    {
                        "family": family,
                        "root_id": CANONICAL_AUDIT_ROOT_IDS[family],
                        "path": str(root),
                    }
                    for family, root in roots.items()
                ],
                "summary": {
                    "artifacts_scanned": 1,
                    "tickers_scanned": 4,
                    "violations": 0,
                    "violations_by_invariant": {},
                    "affected_run_ids": 0,
                    "affected_tickers": 0,
                    "affected_run_id_values": [],
                    "affected_ticker_values": [],
                    "earliest_date": None,
                    "latest_date": None,
                    "llm_consumed_violation_count": 0,
                    "source_artifacts_rewritten": 0,
                },
                "invalid_run_ids": [],
                "artifacts": [
                    {
                        "path": str(source_path.resolve()),
                        "family": "autonomous_sector",
                        "sha256": source_sha256,
                        "integrity_status": "PASS",
                        "decision_eligible": True,
                        "run_id": run_id,
                    }
                ],
                "violations": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(manifest_path),
    )

    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE sector_inference (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            inferred_sector TEXT,
            score REAL NOT NULL DEFAULT 0,
            derived_from TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date)
        );
        CREATE TABLE companies (
            ticker TEXT PRIMARY KEY,
            cik TEXT NOT NULL,
            name TEXT,
            notes TEXT
        );
        CREATE TABLE valuations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            method TEXT NOT NULL,
            inputs_json TEXT NOT NULL DEFAULT '{}',
            outputs_json TEXT NOT NULL DEFAULT '{}',
            warnings_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            valuation_writer_version TEXT,
            quality_gate_verdict TEXT,
            confidence_class TEXT,
            gate_reason_codes TEXT,
            valuation_headwinds TEXT,
            valuation_supports TEXT,
            source_run_id TEXT,
            source_artifact_path TEXT,
            source_artifact_sha256 TEXT,
            financial_integrity_fingerprint TEXT,
            UNIQUE(ticker, as_of_date, method)
        );
    """)

    for ticker, sector in [
        ("NVDA", "semiconductors"),
        ("AMD", "semiconductors"),
        ("INTC", "semiconductors"),
        ("MSFT", "enterprise_software"),
    ]:
        conn.execute(
            "INSERT INTO companies(ticker, cik) VALUES (?, ?)",
            (ticker, issuer_ciks[ticker]),
        )
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, score, derived_from, created_at) "
            "VALUES (?, '2026-04-12', ?, 1.0, '[]', '2026-04-12T00:00:00')",
            (ticker, sector),
        )

    for ticker, scorecard_json in scorecard_rows:
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at, valuation_writer_version,
                quality_gate_verdict, confidence_class, gate_reason_codes,
                valuation_headwinds, valuation_supports, source_run_id,
                source_artifact_path, source_artifact_sha256
            )
            VALUES(
                ?, '2026-04-10', 'scorecard', '{}', ?, '[]',
                '2026-04-12T00:00:00', 'fixture', 'PROCEED', 'MODERATE',
                '[]', '[]', '[]', ?, ?, ?
            )
            """,
            (
                ticker,
                scorecard_json,
                run_id,
                str(source_path.resolve()),
                source_sha256,
            ),
        )
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT *
            FROM valuations
            WHERE ticker = ? AND as_of_date = '2026-04-10'
              AND method = 'scorecard'
            """,
            (ticker,),
        ).fetchone()
        conn.execute(
            """
            UPDATE valuations
            SET financial_integrity_fingerprint = ?
            WHERE id = ?
            """,
            (valuation_integrity_fingerprint(row), row["id"]),
        )

    conn.commit()
    conn.close()
    return db


def _seed_financials_block_db(tmp_path: Path, facts: dict[str, float]) -> Path:
    db_path = tmp_path / "financials.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE companyfacts_facts (
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                period_type TEXT NOT NULL,
                line_item TEXT NOT NULL,
                value REAL,
                UNIQUE(ticker, fiscal_year, period_type, line_item)
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, line_item, value
            )
            VALUES('AAA', 2025, 'FY', ?, ?)
            """,
            list(facts.items()),
        )
    return db_path


@pytest.mark.parametrize(
    ("facts", "missing_component"),
    [
        ({"cash": 25.0, "cfo": 20.0}, "total_debt"),
        ({"total_debt": 40.0, "cfo": 20.0}, "cash"),
    ],
)
def test_financials_block_does_not_zero_fill_missing_net_debt_components(
    tmp_path,
    facts,
    missing_component,
):
    from app.sector.scan import _build_financials_block

    db_path = _seed_financials_block_db(tmp_path, facts)
    block = _build_financials_block("AAA", db_path=db_path, market_cap_m=100.0)

    assert f"net_debt: NEEDS_DATA (explicit finite {missing_component} required)" in block
    assert (
        "enterprise_value (EV): NEEDS_DATA (requires explicit finite total_debt and cash)" in block
    )
    assert "ev_to_cfo:" not in block
    assert "no debt reported" not in block


def test_financials_block_accepts_explicit_zero_debt_and_cash(tmp_path):
    from app.sector.scan import _build_financials_block

    db_path = _seed_financials_block_db(
        tmp_path,
        {
            "cash": 0.0,
            "investment_securities": 0.0,
            "total_debt": 0.0,
            "cfo": 20.0,
        },
    )
    block = _build_financials_block("AAA", db_path=db_path, market_cap_m=100.0)

    assert "net_debt: 0M (debt 0 - cash+sec 0)" in block
    assert "enterprise_value (EV): 100M" in block
    assert "ev_to_cfo: 5.0x" in block
    assert "net_cash_pct_market_cap: 0.0%" in block


def test_financials_block_preserves_legitimate_negative_enterprise_value(tmp_path):
    from app.sector.scan import _build_financials_block

    db_path = _seed_financials_block_db(
        tmp_path,
        {
            "cash": 150.0,
            "investment_securities": 0.0,
            "total_debt": 0.0,
            "cfo": 20.0,
        },
    )
    block = _build_financials_block("AAA", db_path=db_path, market_cap_m=100.0)

    assert "net_debt: -150M (debt 0 - cash+sec 150)" in block
    assert "enterprise_value (EV): -50M  (mcap + net_debt)" in block
    assert "ev_to_cfo: -2.5x" in block
    assert "net_cash_pct_market_cap: 150.0%" in block


def test_financials_block_excludes_facts_filed_after_fixed_asof(tmp_path):
    from app.sector.scan import _build_financials_block

    db_path = tmp_path / "pit-financials.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE companyfacts_facts (
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                period_type TEXT NOT NULL,
                period_end TEXT,
                filed_date TEXT,
                accession TEXT,
                source_url TEXT,
                line_item TEXT NOT NULL,
                value REAL,
                UNIQUE(ticker, fiscal_year, period_type, line_item)
            )
            """
        )
        rows = [
            ("AAA", 2024, "FY", "2024-12-31", "2025-02-01", "visible", "cash", 10.0),
            ("AAA", 2024, "FY", "2024-12-31", "2025-02-01", "visible", "total_debt", 40.0),
            ("AAA", 2024, "FY", "2024-12-31", "2025-02-01", "visible", "cfo", 20.0),
            ("AAA", 2025, "FY", "2025-12-31", "2026-03-01", "future", "cash", 1_000.0),
            ("AAA", 2025, "FY", "2025-12-31", "2026-03-01", "future", "total_debt", 0.0),
            ("AAA", 2025, "FY", "2025-12-31", "2026-03-01", "future", "cfo", 999.0),
        ]
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, filed_date,
                accession, source_url, line_item, value
            )
            VALUES(?, ?, ?, ?, ?, ?,
                   'https://data.sec.gov/api/xbrl/companyfacts/test.json',
                   ?, ?)
            """,
            rows,
        )

    block = _build_financials_block(
        "AAA",
        db_path=db_path,
        market_cap_m=100.0,
        as_of_date="2026-02-13",
    )

    assert "FY2024: 10" in block
    assert "FY2025" not in block
    assert "net_debt: 30M (debt 40 - cash+sec 10)" in block
    assert "enterprise_value (EV): 130M  (mcap + net_debt)" in block


def test_report_snapshot_excludes_post_asof_and_undated_facts(tmp_path):
    from app.sector.scan import (
        SectorDeepResult,
        SectorTriageResult,
        render_scan_report,
    )

    db_path = tmp_path / "pit-report.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE companyfacts_facts (
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                period_type TEXT NOT NULL,
                period_end TEXT,
                filed_date TEXT,
                accession TEXT,
                source_url TEXT,
                line_item TEXT NOT NULL,
                value REAL,
                UNIQUE(ticker, fiscal_year, period_type, line_item)
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, filed_date,
                accession, source_url, line_item, value
            )
            VALUES(
                'AAA', ?, 'FY', ?, ?, ?,
                'https://data.sec.gov/api/xbrl/companyfacts/test.json',
                'revenue', ?
            )
            """,
            [
                (2023, "2023-12-31", "2024-02-01", "visible", 100.0),
                (2024, "2024-12-31", "2026-03-01", "future", 999.0),
                (2025, "2025-12-31", None, "undated", 888.0),
            ],
        )
    triage = SectorTriageResult(
        sector="test",
        candidates_count=1,
        survivors=[],
        cost_usd=0.0,
    )
    deep_results = [
        SectorDeepResult(
            ticker="AAA",
            triage_rank=1,
            triage_reasoning="literal PIT fixture",
            verdict="WATCH",
            confidence="HIGH",
            thesis_summary="Visible filing only.",
            cost_usd=0.0,
        )
    ]

    rendered = render_scan_report(
        sector="test",
        sector_size=1,
        pre_ranked=1,
        triage_result=triage,
        deep_results=deep_results,
        top_n=1,
        total_cost=0.0,
        scorecards={
            "AAA": {
                "pricing_zone_detail": {},
                "quality_context": {},
            }
        },
        as_of_date="2026-02-13",
        db_path=db_path,
    )

    assert "| 2023 | 100 |" in rendered
    assert "| 2024 |" not in rendered
    assert "| 2025 |" not in rendered


# --- Loader tests ---


def test_load_sector_tickers(scan_db):
    from app.sector.scan import load_sector_tickers

    result = load_sector_tickers(sector="semiconductors", db_path=scan_db)
    assert len(result) == 3
    tickers = {t for t, _, _ in result}
    assert tickers == {"NVDA", "AMD", "INTC"}
    for _, _, sc in result:
        assert isinstance(sc, dict)
        assert "pricing_zone_detail" in sc


def test_v1_loader_respects_run_as_of_for_membership_and_scorecard(scan_db):
    future_scorecard = {
        "pricing_zone": "ABOVE_IV",
        "pricing_zone_detail": {
            "current_price": 999.0,
            "dcf_base": 1.0,
            "epv_adjusted": 1.0,
        },
    }
    with sqlite3.connect(str(scan_db)) as conn:
        conn.execute(
            """INSERT INTO sector_inference
               (ticker, as_of_date, inferred_sector, score, derived_from, created_at)
               VALUES ('NVDA', '2026-07-30', 'semiconductors', 1.0, '[]',
                       '2026-07-30T00:00:00')"""
        )
        conn.execute(
            """INSERT INTO valuations
               (ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at)
               VALUES ('NVDA', '2026-07-30', 'scorecard', '{}', ?, '[]',
                       '2026-07-30T00:00:00')""",
            (json.dumps(future_scorecard),),
        )

    from app.sector.scan import load_sector_tickers_classified

    rows, _ = load_sector_tickers_classified(
        sector="semiconductors",
        db_path=scan_db,
        as_of_date="2026-04-12",
        pipeline_version="v1",
        allow_live_market_data=False,
    )

    nvda = next(row for row in rows if row[0] == "NVDA")
    assert nvda[1] == "2026-04-10"
    assert nvda[2]["pricing_zone_detail"]["current_price"] == 120.0


@pytest.mark.parametrize("pipeline_version", ["v1", "v2"])
def test_newest_unauthorized_scorecard_suppresses_older_authorized_evidence(
    scan_db,
    pipeline_version,
):
    invalid_newest = {
        "pricing_zone": "ABOVE_IV",
        "pricing_zone_detail": {
            "current_price": 999.0,
            "dcf_base": 1.0,
            "epv_adjusted": 1.0,
        },
    }
    with sqlite3.connect(str(scan_db)) as conn:
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at
            )
            VALUES(
                'NVDA', '2026-04-11', 'scorecard', '{}', ?, '[]',
                '2026-04-12T01:00:00'
            )
            """,
            (json.dumps(invalid_newest),),
        )

    from app.sector.scan import (
        _latest_scorecard_evidence,
        load_sector_tickers_classified,
    )

    assert _latest_scorecard_evidence(
        ["NVDA"],
        as_of_date="2026-04-12",
        db_path=scan_db,
    ) == {"NVDA": (None, {})}

    rows, _ = load_sector_tickers_classified(
        sector="semiconductors",
        db_path=scan_db,
        as_of_date="2026-04-12",
        pipeline_version=pipeline_version,
        allow_live_market_data=False,
    )
    by_ticker = {ticker: (row_as_of, scorecard) for ticker, row_as_of, scorecard in rows}

    assert by_ticker["NVDA"] == ("2026-04-12", {})
    assert by_ticker["AMD"][1]["pricing_zone_detail"]["current_price"] == 80.0


def test_load_sector_not_found(scan_db):
    from app.sector.scan import load_sector_tickers

    result = load_sector_tickers(sector="nonexistent", db_path=scan_db)
    assert result == []


def test_load_sector_tickers_cap_filter_uses_asof_market_cap(scan_db, monkeypatch):
    from app.autonomous.cap_resolver import CapClassification
    from app.sector.scan import load_sector_tickers

    seen: dict[str, object] = {}
    caps = {"AMD": 5_000.0, "INTC": 250.0, "NVDA": 20_000.0}

    def _fake_classify_tickers_for_market_cap(**kwargs):
        seen.update(kwargs)
        return {
            ticker: CapClassification(
                ticker=ticker,
                as_of_date="2026-04-12",
                market_cap_mm=market_cap,
                cap_source="fixture",
                cap_band="mid",
            )
            for ticker, market_cap in caps.items()
        }

    monkeypatch.setattr(
        "app.sector.scan.classify_tickers_for_market_cap",
        _fake_classify_tickers_for_market_cap,
    )

    result = load_sector_tickers(
        sector="semiconductors",
        db_path=scan_db,
        cap_min=300.0,
        cap_max=10_000.0,
        as_of_date="2026-04-12",
    )

    assert [ticker for ticker, _, _ in result] == ["AMD"]
    assert seen["tickers"] == ["AMD", "INTC", "NVDA"]
    assert seen["as_of_date"] == "2026-04-12"
    assert seen["db_path"] == scan_db
    assert seen["pipeline_version"] == "v1"


def test_load_sector_tickers_cap_filter_keeps_unknown_cap(scan_db, monkeypatch):
    from app.autonomous.cap_resolver import CapClassification
    from app.sector.scan import load_sector_tickers

    caps = {"AMD": None, "INTC": 250.0, "NVDA": 20_000.0}

    def _fake_classify_tickers_for_market_cap(**_kwargs):
        return {
            ticker: CapClassification(
                ticker=ticker,
                as_of_date="2026-04-12",
                market_cap_mm=market_cap,
                cap_source="unknown" if market_cap is None else "fixture",
                cap_band=None if market_cap is None else "mid",
            )
            for ticker, market_cap in caps.items()
        }

    monkeypatch.setattr(
        "app.sector.scan.classify_tickers_for_market_cap",
        _fake_classify_tickers_for_market_cap,
    )

    result = load_sector_tickers(
        sector="semiconductors",
        db_path=scan_db,
        cap_min=300.0,
        cap_max=10_000.0,
        as_of_date="2026-04-12",
    )

    assert [ticker for ticker, _, _ in result] == ["AMD"]


def test_pre_rank_sector(scan_db, monkeypatch):
    from app.sector.scan import load_sector_tickers, pre_rank_sector
    from app.alpha.schemas import TickerSignalPacket

    tickers_data = load_sector_tickers(sector="semiconductors", db_path=scan_db)
    ticker_list = [t for t, _, _ in tickers_data]

    scorecard_lookup = {t: sc for t, _, sc in tickers_data}

    def _fake_financial_context(*, tickers, **_kwargs):
        packets = {}
        for ticker in tickers:
            pzd = scorecard_lookup[ticker]["pricing_zone_detail"]
            packets[ticker] = TickerSignalPacket(
                ticker=ticker,
                current_price=pzd["current_price"],
                dcf_value=pzd["dcf_base"],
                epv_value=pzd["epv_adjusted"],
                gate_verdict="PROCEED",
            )
        return SimpleNamespace(packets=packets)

    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        _fake_financial_context,
    )

    ranked = pre_rank_sector(
        tickers=ticker_list,
        as_of_date="2026-04-12",
        db_path=scan_db,
        limit=50,
        filing_risk_use_llm=False,
    )
    assert len(ranked) == 3
    # INTC and AMD are undervalued (dcf > price), NVDA is overvalued
    assert ranked[-1].ticker == "NVDA"
    assert ranked[0].consensus_score > ranked[-1].consensus_score


def test_pre_rank_limit(scan_db, monkeypatch):
    from app.sector.scan import load_sector_tickers, pre_rank_sector
    from app.alpha.schemas import TickerSignalPacket

    tickers_data = load_sector_tickers(sector="semiconductors", db_path=scan_db)
    ticker_list = [t for t, _, _ in tickers_data]

    scorecard_lookup = {t: sc for t, _, sc in tickers_data}

    def _fake_financial_context(*, tickers, **_kwargs):
        packets = {}
        for ticker in tickers:
            pzd = scorecard_lookup[ticker]["pricing_zone_detail"]
            packets[ticker] = TickerSignalPacket(
                ticker=ticker,
                current_price=pzd["current_price"],
                dcf_value=pzd["dcf_base"],
                epv_value=pzd["epv_adjusted"],
                gate_verdict="PROCEED",
            )
        return SimpleNamespace(packets=packets)

    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        _fake_financial_context,
    )

    ranked = pre_rank_sector(
        tickers=ticker_list,
        as_of_date="2026-04-12",
        db_path=scan_db,
        limit=2,
        filing_risk_use_llm=False,
    )
    assert len(ranked) == 2


def test_pre_rank_sector_can_include_blocked_for_sector_specific_gate(monkeypatch):
    from app.sector.scan import pre_rank_sector
    from app.alpha.schemas import TickerSignalPacket

    def _fake_financial_context(*, tickers, **_kwargs):
        assert tickers == ["BANK", "ADR"]
        return SimpleNamespace(
            packets={
                "BANK": TickerSignalPacket(
                    ticker="BANK",
                    current_price=40.0,
                    gate_verdict="BLOCK",
                ),
                "ADR": TickerSignalPacket(
                    ticker="ADR",
                    current_price=16.0,
                    gate_verdict="PROCEED",
                ),
            }
        )

    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        _fake_financial_context,
    )

    default_ranked = pre_rank_sector(
        tickers=["BANK", "ADR"],
        as_of_date="2026-04-12",
        limit=0,
        filing_risk_use_llm=False,
    )
    included_ranked = pre_rank_sector(
        tickers=["BANK", "ADR"],
        as_of_date="2026-04-12",
        limit=0,
        filing_risk_use_llm=False,
        include_blocked=True,
    )

    assert [entry.ticker for entry in default_ranked] == ["ADR"]
    assert [entry.ticker for entry in included_ranked] == ["ADR", "BANK"]
    assert included_ranked[1].adjustments == ["NO_DISCOUNT_DATA", "GATE_BLOCKED"]


def test_pre_rank_sector_never_enables_filing_risk_provider(monkeypatch):
    from app.alpha.schemas import TickerSignalPacket
    from app.sector.scan import pre_rank_sector

    observed: dict[str, object] = {}

    def _financial_context(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            packets={
                "AAA": TickerSignalPacket(
                    ticker="AAA",
                    current_price=50.0,
                    dcf_value=75.0,
                    epv_value=65.0,
                    gate_verdict="PROCEED",
                )
            }
        )

    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        _financial_context,
    )

    ranked = pre_rank_sector(
        tickers=["AAA"],
        as_of_date="2026-04-12",
        db_path="/tmp/fixed.db",
        filing_risk_use_llm=True,
    )

    assert [entry.ticker for entry in ranked] == ["AAA"]
    assert observed == {
        "tickers": ["AAA"],
        "as_of_date": "2026-04-12",
        "db_path": "/tmp/fixed.db",
        "cfg": None,
    }


# --- Triage tests ---


def test_triage_happy_path():
    from app.sector.scan import run_sector_triage, ScanConfig

    survivors_payload = [
        {"ticker": "AMD", "rank": 1, "reasoning": "Best value in semis"},
        {"ticker": "INTC", "rank": 2, "reasoning": "Turnaround play"},
    ]
    client = _ScriptedClient([_triage_response(survivors_payload)])

    scorecards = {
        "AMD": {
            "pricing_zone": "MOS",
            "pricing_zone_detail": {"current_price": 80.0, "dcf_base": 110.0},
        },
        "INTC": {
            "pricing_zone": "MOS",
            "pricing_zone_detail": {"current_price": 25.0, "dcf_base": 40.0},
        },
        "NVDA": {
            "pricing_zone": "ABOVE_IV",
            "pricing_zone_detail": {"current_price": 120.0, "dcf_base": 95.0},
        },
    }

    result = run_sector_triage(
        client=client,
        sector="semiconductors",
        ranked_tickers=["AMD", "INTC", "NVDA"],
        scorecards=scorecards,
        config=ScanConfig(),
        integrity_scope=_integrity_scope("AMD", "INTC", "NVDA"),
    )

    assert len(result.survivors) == 2
    assert result.survivors[0]["ticker"] == "AMD"
    assert result.cost_usd > 0
    assert len(client.calls) == 1


def test_triage_prompt_uses_canonical_packets_not_raw_scorecards():
    from app.sector.scan import ScanConfig, run_sector_triage

    client = _ScriptedClient(
        [_triage_response([{"ticker": "AAA", "rank": 1, "reasoning": "Review"}])]
    )
    result = run_sector_triage(
        client=client,
        sector="software",
        ranked_tickers=["AAA"],
        scorecards={
            "AAA": {
                "pricing_zone_detail": {
                    "current_price": 987_654_321.0,
                    "market_cap": 987_654_321.0,
                }
            }
        },
        config=ScanConfig(),
        integrity_scope=_integrity_scope("AAA"),
    )

    prompt = client.calls[0]["messages"][0]["content"]
    assert result.survivors[0]["ticker"] == "AAA"
    assert "987654321" not in prompt
    assert '"current_price":50.0' in prompt
    assert '"market_cap_mm":500.0' in prompt
    assert "financial_integrity_scope_fingerprint:" in prompt


def test_triage_small_sector():
    from app.sector.scan import run_sector_triage, ScanConfig

    client = _ScriptedClient(
        [_triage_response([{"ticker": "AAPL", "rank": 1, "reasoning": "Only company"}])]
    )

    result = run_sector_triage(
        client=client,
        sector="test",
        ranked_tickers=["AAPL"],
        scorecards={"AAPL": {"pricing_zone": "MOS", "pricing_zone_detail": {}}},
        config=ScanConfig(max_deep_reviews=20),
        integrity_scope=_integrity_scope("AAPL"),
    )

    assert len(result.survivors) == 1


def test_triage_invalid_scope_suppresses_physical_provider_call():
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.financial_integrity import (
        FinancialIntegrityScope,
        InvalidFinancialInputError,
    )
    from app.sector.scan import ScanConfig, run_sector_triage

    client = _ScriptedClient(
        [_triage_response([{"ticker": "AAA", "rank": 1, "reasoning": "unused"}])]
    )
    invalid_scope = FinancialIntegrityScope(
        context="sector_triage_invalid_scope",
        run_as_of_date="2026-04-12",
        packets=(TickerSignalPacket(ticker="AAA", current_price=50.0),),
    )

    with pytest.raises(InvalidFinancialInputError):
        run_sector_triage(
            client=client,
            sector="software",
            ranked_tickers=["AAA"],
            scorecards={"AAA": {"pricing_zone": "MOS", "pricing_zone_detail": {}}},
            config=ScanConfig(),
            integrity_scope=invalid_scope,
        )

    assert client.calls == []


def test_triage_zero_budget_suppresses_physical_provider_call():
    from app.llm.providers.retry_guard import LLMCostBudgetExceeded
    from app.llm.usage_capture import provider_usage_budget
    from app.sector.scan import ScanConfig, run_sector_triage

    client = _ScriptedClient(
        [_triage_response([{"ticker": "AAA", "rank": 1, "reasoning": "unused"}])]
    )
    with (
        provider_usage_budget(0.0),
        pytest.raises(LLMCostBudgetExceeded),
    ):
        run_sector_triage(
            client=client,
            sector="software",
            ranked_tickers=["AAA"],
            scorecards={"AAA": {}},
            config=ScanConfig(),
            integrity_scope=_integrity_scope("AAA"),
        )

    assert client.calls == []


def test_triage_tiny_positive_budget_prevents_overshoot():
    from app.llm.providers.retry_guard import LLMCostBudgetExceeded
    from app.llm.usage_capture import provider_usage_budget
    from app.sector.scan import ScanConfig, run_sector_triage

    client = _ScriptedClient(
        [_triage_response([{"ticker": "AAA", "rank": 1, "reasoning": "unused"}])]
    )
    with (
        provider_usage_budget(0.0000005),
        pytest.raises(LLMCostBudgetExceeded),
    ):
        run_sector_triage(
            client=client,
            sector="software",
            ranked_tickers=["AAA"],
            scorecards={"AAA": {}},
            config=ScanConfig(),
            integrity_scope=_integrity_scope("AAA"),
        )

    assert client.calls == []


def test_triage_rejects_unbound_commodity_block_before_provider_call():
    from app.autonomous.financial_integrity import InvalidFinancialInputError
    from app.sector.scan import ScanConfig, run_sector_triage

    client = _ScriptedClient(
        [_triage_response([{"ticker": "AAA", "rank": 1, "reasoning": "unused"}])]
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_triage(
            client=client,
            sector="energy",
            ranked_tickers=["AAA"],
            scorecards={"AAA": {"pricing_zone": "MOS", "pricing_zone_detail": {}}},
            config=ScanConfig(),
            integrity_scope=_integrity_scope("AAA"),
            commodity_block="UNBOUND COMMODITY CONTEXT",
        )

    assert exc_info.value.violations[0].code == "UNBOUND_COMMODITY_CONTEXT"
    assert client.calls == []


# --- Deep review tests ---


def test_deep_review_one_ticker():
    from app.sector.scan import run_sector_deep_review, ScanConfig

    client = _ScriptedClient([_deep_review_response()])
    survivors = [{"ticker": "AMD", "rank": 1, "reasoning": "Best value"}]

    results, prompts = run_sector_deep_review(
        client=client,
        sector="semiconductors",
        triage_survivors=survivors,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        config=ScanConfig(),
        integrity_scopes={"AMD": _integrity_scope("AMD")},
    )

    assert len(results) == 1
    assert results[0].ticker == "AMD"
    assert results[0].verdict == "WATCH"
    assert results[0].thesis_summary == "Strong capital allocation"
    assert "AMD" in prompts  # prompt was captured


def test_deep_review_uses_only_integrity_bound_packet_financials(monkeypatch):
    from app.sector.scan import ScanConfig, run_sector_deep_review

    monkeypatch.setattr(
        "app.sector.scan._build_financials_block",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("deep review must not reread CompanyFacts")
        ),
    )
    client = _ScriptedClient([_deep_review_response()])

    results, prompts = run_sector_deep_review(
        client=client,
        sector="software",
        triage_survivors=[{"ticker": "AAA", "rank": 1, "reasoning": "Review"}],
        bundle_builder=lambda ticker, as_of_date=None: _sample_bundle(ticker),
        config=ScanConfig(),
        integrity_scopes={"AAA": _integrity_scope("AAA")},
    )

    assert results[0].error is None
    assert "CANONICAL FINANCIAL PACKET (financial-integrity bound)" in prompts["AAA"]
    assert '"current_price":50.0' in prompts["AAA"]
    assert '"quote_snapshot_id":' in prompts["AAA"]


def test_deep_review_bundle_mismatch_has_zero_provider_calls_and_uses_run_date():
    from dataclasses import replace

    from app.sector.scan import ScanConfig, run_sector_deep_review

    client = _ScriptedClient([_deep_review_response()])
    observed_as_of_dates: list[str | None] = []

    def _contradictory_bundle(ticker, as_of_date=None):
        observed_as_of_dates.append(as_of_date)
        bundle = _sample_bundle(ticker)
        return replace(
            bundle,
            valuation=replace(bundle.valuation, current_price=987_654_321.0),
        )

    results, prompts = run_sector_deep_review(
        client=client,
        sector="software",
        triage_survivors=[{"ticker": "AAA", "rank": 1, "reasoning": "Review"}],
        bundle_builder=_contradictory_bundle,
        config=ScanConfig(),
        integrity_scopes={"AAA": _integrity_scope("AAA")},
    )

    assert client.calls == []
    assert observed_as_of_dates == ["2026-04-12"]
    assert results[0].error is not None
    assert "does not match the canonical financial packet" in results[0].error
    assert prompts == {}


def test_deep_review_bundle_integrity_failure_aborts_before_provider_call():
    from app.autonomous.financial_integrity import (
        FinancialIntegrityGateResult,
        InvalidFinancialInputError,
    )
    from app.sector.scan import ScanConfig, run_sector_deep_review

    client = _ScriptedClient([_deep_review_response()])
    error = InvalidFinancialInputError(
        FinancialIntegrityGateResult(
            context="analyst_bundle",
            run_as_of_date="2026-04-12",
            status="NEEDS_DATA",
        )
    )

    def _invalid_bundle(ticker, as_of_date=None):
        raise error

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_deep_review(
            client=client,
            sector="software",
            triage_survivors=[{"ticker": "AAA", "rank": 1, "reasoning": "Review"}],
            bundle_builder=_invalid_bundle,
            config=ScanConfig(),
            integrity_scopes={"AAA": _integrity_scope("AAA")},
        )

    assert exc_info.value is error
    assert client.calls == []


def test_deep_review_rejects_unbound_commodity_block_before_provider_call():
    from app.autonomous.financial_integrity import InvalidFinancialInputError
    from app.sector.scan import ScanConfig, run_sector_deep_review

    client = _ScriptedClient([_deep_review_response()])

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_sector_deep_review(
            client=client,
            sector="energy",
            triage_survivors=[{"ticker": "AAA", "rank": 1, "reasoning": "unused"}],
            bundle_builder=lambda ticker, as_of_date=None: _sample_bundle(ticker),
            config=ScanConfig(),
            integrity_scopes={"AAA": _integrity_scope("AAA")},
            commodity_block="UNBOUND COMMODITY CONTEXT",
        )

    assert exc_info.value.violations[0].code == "UNBOUND_COMMODITY_CONTEXT"
    assert client.calls == []


def test_deep_review_zero_shared_budget_suppresses_stage4_provider_call():
    from app.discover.persistence import DiscoverCostBudgetExceeded
    from app.llm.usage_capture import provider_usage_budget
    from app.sector.scan import ScanConfig, run_sector_deep_review

    client = _ScriptedClient([_deep_review_response()])
    with (
        provider_usage_budget(0.0),
        pytest.raises(DiscoverCostBudgetExceeded),
    ):
        run_sector_deep_review(
            client=client,
            sector="software",
            triage_survivors=[{"ticker": "AAA", "rank": 1, "reasoning": "Review"}],
            bundle_builder=lambda ticker, as_of_date=None: _sample_bundle(ticker),
            config=ScanConfig(),
            integrity_scopes={"AAA": _integrity_scope("AAA")},
        )

    assert client.calls == []


def test_paid_sector_stages_include_nested_filing_risk_usage():
    from app.autonomous.financial_integrity import require_financial_integrity_scope
    from app.llm.usage_capture import (
        provider_usage_lane,
        provider_usage_records,
        provider_usage_request,
        record_provider_usage,
    )
    from app.sector.scan import ScanConfig, _run_paid_sector_stages

    class _NestedProvider:
        provider_name = "openai"
        model = "gpt-5-mini"

    scope = _integrity_scope("AAA")
    client = _ScriptedClient(
        [
            _triage_response([{"ticker": "AAA", "rank": 1, "reasoning": "Review"}]),
            _deep_review_response(),
        ]
    )

    def _bundle_with_nested_filing_usage(ticker, as_of_date=None):
        provider = _NestedProvider()
        prompt = "nested filing risk request"
        schema = {"type": "object"}
        with (
            provider_usage_lane(f"filing_risk:{ticker}"),
            provider_usage_request(
                provider=provider,
                prompt=prompt,
                schema=schema,
                schema_name="filing_risk_scan_v1",
                max_output_tokens=100,
            ),
        ):
            response = SimpleNamespace(
                json_text="{}",
                usage_input_tokens=100,
                usage_output_tokens=10,
                model=provider.model,
            )
            for usage_record in provider_usage_records(
                provider=provider,
                result=response,
                prompt=prompt,
                schema_name="filing_risk_scan_v1",
            ):
                record_provider_usage(usage_record)
        return _sample_bundle(ticker)

    triage, deep_results, _prompts, paid_usage = _run_paid_sector_stages(
        sector="software",
        budget_usd=1.0,
        client=client,
        ranked_tickers=["AAA"],
        scorecards={"AAA": {}},
        config=ScanConfig(),
        integrity_scope=scope,
        integrity_scope_fingerprint=require_financial_integrity_scope(scope).scope_fingerprint,
        financial_packets={"AAA": scope.packets[0]},
        bundle_builder=_bundle_with_nested_filing_usage,
        db_path=None,
    )

    assert len(client.calls) == 2
    assert triage.cost_usd == 0.12
    assert deep_results[0].cost_usd == 0.054
    assert paid_usage["physical_calls"] == 3
    assert paid_usage["cost_usd"] == 0.17413
    assert paid_usage["budget_remaining_usd"] == 0.82587
    assert {item["lane"] for item in paid_usage["provider_usage"]} == {
        "sector_triage:software",
        "filing_risk:AAA",
        "sector_deep_review:AAA",
    }


def test_deep_review_skips_on_error():
    from app.sector.scan import run_sector_deep_review, ScanConfig

    class _ErrorThenSuccess:
        def __init__(self):
            self.calls = []
            self._count = 0

        class _Messages:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.calls.append(kwargs)
                self._outer._count += 1
                if self._outer._count == 1:
                    raise RuntimeError("API timeout")
                return _deep_review_response()

        @property
        def messages(self):
            return _ErrorThenSuccess._Messages(self)

    client = _ErrorThenSuccess()
    survivors = [
        {"ticker": "AMD", "rank": 1, "reasoning": "Best value"},
        {"ticker": "INTC", "rank": 2, "reasoning": "Turnaround"},
    ]

    results, prompts = run_sector_deep_review(
        client=client,
        sector="semiconductors",
        triage_survivors=survivors,
        bundle_builder=lambda t, as_of_date=None: _sample_bundle(t),
        config=ScanConfig(),
        integrity_scopes={
            "AMD": _integrity_scope("AMD"),
            "INTC": _integrity_scope("INTC"),
        },
    )

    assert len(results) == 2
    assert results[0].ticker == "AMD"
    assert results[0].error is not None
    assert results[0].cost_usd > 0.0
    assert results[0]._provider_usage[0]["status"] == "ERROR"
    assert results[1].ticker == "INTC"
    assert results[1].verdict == "WATCH"


# --- Report tests ---


def test_render_scan_report():
    from app.sector.scan import render_scan_report, SectorTriageResult, SectorDeepResult

    triage = SectorTriageResult(
        sector="semiconductors",
        candidates_count=40,
        survivors=[{"ticker": "AMD", "rank": 1, "reasoning": "Best value"}],
        surprises=["NVDA surprisingly excluded"],
        cost_usd=0.12,
    )
    deep_results = [
        SectorDeepResult(
            ticker="AMD",
            triage_rank=1,
            triage_reasoning="Best value",
            verdict="WATCH",
            confidence="MODERATE",
            thesis_summary="Strong FCF in semis",
            key_numbers=["FCF yield 8%"],
            positives=["Share gains"],
            risks=["Cyclical"],
            open_questions=["Capex?"],
            cost_usd=0.80,
        ),
    ]

    md = render_scan_report(
        sector="semiconductors",
        sector_size=40,
        pre_ranked=40,
        triage_result=triage,
        deep_results=deep_results,
        top_n=10,
        total_cost=0.92,
    )

    assert "# Sector Scan: semiconductors" in md
    assert "AMD" in md
    assert "Strong FCF in semis" in md
    assert "WATCH" in md
    assert "**Scan family:** normal" in md


def test_run_scan_dry_run_summary_is_normal_family(scan_db):
    from app.sector.scan import run_scan

    summary = run_scan(sector="semiconductors", dry_run=True, db_path=scan_db)

    assert summary["scan_family"] == "normal"
    assert summary["dry_run"] is True


def test_run_scan_zero_budget_suppresses_all_provider_calls(monkeypatch, tmp_path):
    from app.llm.providers.retry_guard import LLMCostBudgetExceeded
    from app.sector.scan import run_scan

    scope = _integrity_scope("AAA")
    financial_context = SimpleNamespace(
        packets={"AAA": scope.packets[0]},
        scope=lambda **_kwargs: scope,
    )
    client = _ScriptedClient(
        [_triage_response([{"ticker": "AAA", "rank": 1, "reasoning": "unused"}])]
    )
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers",
        lambda **_kwargs: [("AAA", "2026-04-12", {})],
    )
    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        lambda **_kwargs: financial_context,
    )
    monkeypatch.setattr(
        "app.sector.scan.rank_by_consensus",
        lambda _packets: SimpleNamespace(
            ranked=[SimpleNamespace(ticker="AAA", consensus_score=1.0)],
            ranked_insufficient=[],
        ),
    )

    with pytest.raises(LLMCostBudgetExceeded):
        run_scan(
            sector="software",
            budget_usd=0.0,
            output_dir=tmp_path / "outputs",
            client=client,
            bundle_builder=lambda ticker, as_of_date=None: _sample_bundle(ticker),
            skip_preflight=True,
        )

    assert client.calls == []
    assert not (tmp_path / "outputs").exists()


def test_run_scan_default_bundle_builder_threads_canonical_parent_packet(
    monkeypatch,
):
    from app.sector.scan import SectorTriageResult, run_scan

    class _StopAfterBundle(RuntimeError):
        pass

    scope = _integrity_scope("AAA")
    packet = scope.packets[0]
    financial_context = SimpleNamespace(
        packets={"AAA": packet},
        scope=lambda **_kwargs: scope,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers",
        lambda **_kwargs: [("AAA", "2026-04-12", {})],
    )
    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        lambda **_kwargs: financial_context,
    )
    monkeypatch.setattr(
        "app.sector.scan.rank_by_consensus",
        lambda _packets: SimpleNamespace(
            ranked=[SimpleNamespace(ticker="AAA", consensus_score=1.0)],
            ranked_insufficient=[],
        ),
    )
    monkeypatch.setattr(
        "app.sector.scan.run_sector_triage",
        lambda **_kwargs: SectorTriageResult(
            sector="software",
            candidates_count=1,
            survivors=[{"ticker": "AAA", "rank": 1, "reasoning": "fixture"}],
        ),
    )

    def fake_cached_builder(**kwargs):
        captured.update(kwargs)
        return _sample_bundle("AAA")

    monkeypatch.setattr(
        "app.analyst.bundle_builder.build_analysis_evidence_bundle_from_cached_scorecard",
        fake_cached_builder,
    )

    def stop_after_bundle(**kwargs):
        kwargs["bundle_builder"]("AAA", as_of_date="2026-04-12")
        raise _StopAfterBundle

    monkeypatch.setattr(
        "app.sector.scan.run_sector_deep_review",
        stop_after_bundle,
    )

    with pytest.raises(_StopAfterBundle):
        run_scan(
            sector="software",
            client=object(),
            skip_preflight=True,
        )

    assert captured["ticker"] == "AAA"
    assert captured["financial_packet"] is packet


def test_run_scan_rejects_valid_but_drifted_scope_before_publication(
    monkeypatch,
    tmp_path,
):
    from app.autonomous.financial_integrity import InvalidFinancialInputError
    from app.sector.scan import run_scan

    scope = _integrity_scope("AAA")
    packet = scope.packets[0]
    financial_context = SimpleNamespace(
        packets={"AAA": packet},
        scope=lambda **_kwargs: scope,
    )
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers",
        lambda **_kwargs: [("AAA", "2026-04-12", {})],
    )
    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        lambda **_kwargs: financial_context,
    )
    monkeypatch.setattr(
        "app.sector.scan.rank_by_consensus",
        lambda _packets: SimpleNamespace(
            ranked=[SimpleNamespace(ticker="AAA", consensus_score=1.0)],
            ranked_insufficient=[],
        ),
    )
    monkeypatch.setattr(
        "app.sector.scan.run_sector_triage",
        lambda **_kwargs: SimpleNamespace(survivors=[], cost_usd=0.0),
    )

    def _deep_review(**_kwargs):
        # A changed DCF remains individually valid; only comparison with the
        # first authorization fingerprint detects the publication drift.
        packet.dcf_value = 111.0
        return [], {}

    monkeypatch.setattr("app.sector.scan.run_sector_deep_review", _deep_review)

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_scan(
            sector="software",
            output_dir=tmp_path / "outputs",
            client=object(),
            skip_preflight=True,
        )

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_INPUT_MUTATED"
    assert not (tmp_path / "outputs").exists()


def test_scan_artifact_publication_rejects_unbound_commodity_snapshots(tmp_path):
    from app.autonomous.financial_integrity import (
        InvalidFinancialInputError,
        require_financial_integrity_scope,
    )
    from app.market.commodity_context import CommoditySnapshot
    from app.sector.scan import (
        SectorTriageResult,
        _save_scan_artifacts,
    )

    scope = _integrity_scope("AAA")
    output_dir = tmp_path / "outputs"
    snapshot = CommoditySnapshot(
        symbol="CL=F",
        display_name="WTI Crude Oil",
        unit="USD/barrel",
        as_of="2026-04-12",
        current_price=80.0,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        _save_scan_artifacts(
            output_dir=output_dir,
            sector="energy",
            as_of_date="2026-04-12",
            triage_prompt="",
            triage_result=SectorTriageResult(sector="energy", candidates_count=1),
            deep_prompts={},
            deep_results=[],
            financial_packets={"AAA": scope.packets[0]},
            financial_integrity_result=require_financial_integrity_scope(scope),
            report_path=output_dir / "energy_2026-04-12.md",
            commodity_snapshots=[snapshot],
        )

    assert exc_info.value.violations[0].code == "UNBOUND_COMMODITY_CONTEXT"
    assert not output_dir.exists()


def test_run_scan_persists_auditable_exact_financial_contract(
    monkeypatch,
    tmp_path,
    scan_db,
):
    from app.autonomous.artifact_financial_audit import (
        audit_artifact_tree,
        audit_payload,
    )
    from app.autonomous.financial_integrity import require_financial_integrity_scope
    from app.config import get_config
    from app.sector.scan import SectorTriageResult, run_scan

    class _FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 4, 12)

    scope = _integrity_scope("AAA")
    packet = scope.packets[0]
    financial_context = SimpleNamespace(
        packets={"AAA": packet},
        scope=lambda **_kwargs: scope,
    )
    monkeypatch.setattr("app.sector.scan.date", _FixedDate)
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers",
        lambda **_kwargs: [("AAA", "2026-04-12", {})],
    )
    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        lambda **_kwargs: financial_context,
    )
    monkeypatch.setattr(
        "app.sector.scan.rank_by_consensus",
        lambda _packets: SimpleNamespace(
            ranked=[SimpleNamespace(ticker="AAA", consensus_score=1.0)],
            ranked_insufficient=[],
        ),
    )
    triage = SectorTriageResult(
        sector="software",
        candidates_count=1,
        survivors=[],
        surprises=[],
        cost_usd=0.0,
    )
    triage._prompt = "AAA current_price: 50.0 dcf_base: 110.0"
    monkeypatch.setattr("app.sector.scan.run_sector_triage", lambda **_kwargs: triage)
    monkeypatch.setattr(
        "app.sector.scan.run_sector_deep_review",
        lambda **_kwargs: ([], {}),
    )

    cfg = get_config()
    scans_root = (Path(cfg.outputs_dir) / "scans").resolve()
    summary = run_scan(
        sector="software",
        output_dir=scans_root,
        client=object(),
        db_path=scan_db,
        skip_preflight=True,
    )

    artifact_path = Path(summary["artifact_path"])
    report_path = Path(summary["report_path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    gate = require_financial_integrity_scope(scope)
    assert payload["as_of_date"] == "2026-04-12"
    assert payload["financial_context"] == {
        "scope_fingerprint": gate.scope_fingerprint,
        "tickers": [
            {
                "ticker": "AAA",
                "current_price": 50.0,
                "quote_snapshot_id": packet.quote_snapshot_id,
            }
        ],
    }
    assert payload["financial_integrity"]["status"] == "PASS"
    assert payload["financial_integrity"]["scope_fingerprint"] == gate.scope_fingerprint
    assert payload["financial_integrity"]["quote_snapshots"] == [
        {
            "ticker": "AAA",
            "price": 50.0,
            "currency": "USD",
            "as_of_date": "2026-04-12",
            "source": "fixture_quote",
            "source_url": "https://example.test/quotes/AAA",
            "price_unit": "USD_per_share",
            "price_basis": "UNADJUSTED",
            "raw_price": 50.0,
            "split_adjustment_factor": 1.0,
            "split_effective_date": None,
            "quote_snapshot_id": packet.quote_snapshot_id,
        }
    ]
    assert audit_payload(payload) == []

    reports = audit_artifact_tree(
        runs_root=Path(cfg.runs_dir) / "autonomous_sector",
        analyst_outputs_root=Path(cfg.analyst_outputs_dir),
        scans_root=scans_root,
        research_outputs_root=Path(cfg.research_dir),
        watchlist_report_roots=[Path(cfg.outputs_dir) / "digests"],
        analysis_dir=tmp_path / "scan_audit",
        generated_at=datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc),
    )
    manifest = json.loads(reports.manifest_json.read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["artifacts"]}
    assert records[str(artifact_path.resolve())]["integrity_status"] == "PASS"
    assert records[str(report_path.resolve())]["integrity_status"] == "PASS"
    assert not any(
        item["invariant"] == "SCAN_FINANCIAL_PROVENANCE_MISSING"
        and item["artifact_path"] in {str(artifact_path), str(report_path)}
        for item in manifest["violations"]
    )


def test_energy_run_scan_excludes_unbound_commodity_context(
    monkeypatch,
    scan_db,
):
    from app.config import get_config
    from app.sector.scan import run_scan

    class _FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 4, 12)

    scope = _integrity_scope("AAA")
    financial_context = SimpleNamespace(
        packets={"AAA": scope.packets[0]},
        scope=lambda **_kwargs: scope,
    )
    commodity_fetch_attempts: list[str] = []

    def _unexpected_commodity_fetch(sector, **_kwargs):
        commodity_fetch_attempts.append(sector)
        raise AssertionError("paid V1 scan must not enter the commodity network path")

    monkeypatch.setattr("app.sector.scan.date", _FixedDate)
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers",
        lambda **_kwargs: [("AAA", "2026-04-12", {})],
    )
    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        lambda **_kwargs: financial_context,
    )
    monkeypatch.setattr(
        "app.sector.scan.rank_by_consensus",
        lambda _packets: SimpleNamespace(
            ranked=[SimpleNamespace(ticker="AAA", consensus_score=1.0)],
            ranked_insufficient=[],
        ),
    )
    monkeypatch.setattr(
        "app.market.commodity_context.get_sector_commodity_context",
        _unexpected_commodity_fetch,
    )

    client = _ScriptedClient(
        [
            _triage_response([{"ticker": "AAA", "rank": 1, "reasoning": "Review"}]),
            _deep_review_response(),
        ]
    )
    scans_root = (Path(get_config().outputs_dir) / "scans").resolve()
    summary = run_scan(
        sector="energy",
        output_dir=scans_root,
        client=client,
        bundle_builder=lambda ticker, as_of_date=None: _sample_bundle(ticker),
        db_path=scan_db,
        skip_preflight=True,
    )

    artifact_path = Path(summary["artifact_path"])
    report_path = Path(summary["report_path"])
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    provider_calls = json.dumps(client.calls, sort_keys=True)

    assert commodity_fetch_attempts == []
    assert len(client.calls) == 2
    assert "COMMODITY CONTEXT" not in provider_calls
    assert "commodity_context" not in artifact_payload
    assert "COMMODITY CONTEXT" not in artifact_path.read_text(encoding="utf-8")
    assert "Commodity Context" not in report_path.read_text(encoding="utf-8")


def test_run_scan_interstage_drift_blocks_all_deep_provider_work(
    monkeypatch,
    tmp_path,
):
    from app.autonomous.financial_integrity import InvalidFinancialInputError
    from app.sector.scan import run_scan

    scope = _integrity_scope("AAA")
    packet = scope.packets[0]
    financial_context = SimpleNamespace(
        packets={"AAA": packet},
        scope=lambda **_kwargs: scope,
    )
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers",
        lambda **_kwargs: [("AAA", "2026-04-12", {})],
    )
    monkeypatch.setattr(
        "app.sector.scan.build_canonical_v1_financial_context",
        lambda **_kwargs: financial_context,
    )
    monkeypatch.setattr(
        "app.sector.scan.rank_by_consensus",
        lambda _packets: SimpleNamespace(
            ranked=[SimpleNamespace(ticker="AAA", consensus_score=1.0)],
            ranked_insufficient=[],
        ),
    )

    def _triage(**_kwargs):
        packet.dcf_value = 111.0
        return SimpleNamespace(
            survivors=[{"ticker": "AAA", "rank": 1, "reasoning": "test"}],
            cost_usd=0.0,
        )

    deep_calls: list[str] = []
    monkeypatch.setattr("app.sector.scan.run_sector_triage", _triage)
    monkeypatch.setattr(
        "app.sector.scan.run_sector_deep_review",
        lambda **_kwargs: deep_calls.append("deep") or ([], {}),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_scan(
            sector="software",
            output_dir=tmp_path / "outputs",
            client=object(),
            skip_preflight=True,
        )

    assert exc_info.value.violations[0].code == "BOUND_FINANCIAL_INPUT_MUTATED"
    assert deep_calls == []
    assert not (tmp_path / "outputs").exists()


def test_scan_cli_dry_run_labels_normal_family(scan_db, monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(scan_db))
    from app.config import get_config

    get_config.cache_clear()
    from app.cli import app

    result = CliRunner().invoke(app, ["scan", "semiconductors", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Scan family: normal" in result.output
    get_config.cache_clear()


def test_render_report_top_n():
    from app.sector.scan import render_scan_report, SectorTriageResult, SectorDeepResult

    triage = SectorTriageResult(sector="test", candidates_count=5, survivors=[], cost_usd=0.10)
    deep_results = [
        SectorDeepResult(
            ticker="A",
            triage_rank=1,
            triage_reasoning="ok",
            verdict="WATCH",
            confidence="HIGH",
            thesis_summary="Good",
            cost_usd=0.50,
        ),
        SectorDeepResult(
            ticker="B",
            triage_rank=2,
            triage_reasoning="ok",
            verdict="PASS",
            confidence="LOW",
            thesis_summary="Weak",
            cost_usd=0.50,
        ),
    ]

    md = render_scan_report(
        sector="test",
        sector_size=5,
        pre_ranked=5,
        triage_result=triage,
        deep_results=deep_results,
        top_n=1,
        total_cost=1.10,
    )

    assert "### #1:" in md
    assert "### #2:" not in md


def test_render_report_cap_filter_in_title():
    """Cap filter label should appear in the title and as a metadata line."""
    from app.sector.scan import render_scan_report, SectorTriageResult, SectorDeepResult

    triage = SectorTriageResult(
        sector="healthcare_pharma", candidates_count=5, survivors=[], cost_usd=0.10
    )
    deep_results = [
        SectorDeepResult(
            ticker="HRMY",
            triage_rank=1,
            triage_reasoning="ok",
            verdict="BUY",
            confidence="HIGH",
            thesis_summary="cheap",
            cost_usd=0.50,
        ),
    ]

    md = render_scan_report(
        sector="healthcare_pharma",
        sector_size=291,
        pre_ranked=148,
        triage_result=triage,
        deep_results=deep_results,
        top_n=5,
        total_cost=0.60,
        cap_filter_label="small/mid cap",
        cap_min=300.0,
        cap_max=10000.0,
    )

    # Title extended with filter
    assert "# Sector Scan: healthcare_pharma — small/mid cap" in md
    # Metadata line shows numeric range
    assert "**Cap filter:** small/mid cap" in md
    assert "$300M" in md
    assert "$10,000M" in md


def test_render_report_no_cap_filter_title_unchanged():
    """When no cap filter is applied, title and header must stay clean (regression guard)."""
    from app.sector.scan import render_scan_report, SectorTriageResult, SectorDeepResult

    triage = SectorTriageResult(sector="energy", candidates_count=5, survivors=[], cost_usd=0.10)
    deep_results = [
        SectorDeepResult(
            ticker="XOM",
            triage_rank=1,
            triage_reasoning="ok",
            verdict="WATCH",
            confidence="MODERATE",
            thesis_summary="pass",
            cost_usd=0.50,
        ),
    ]

    md = render_scan_report(
        sector="energy",
        sector_size=49,
        pre_ranked=22,
        triage_result=triage,
        deep_results=deep_results,
        top_n=5,
        total_cost=0.60,
    )

    assert "# Sector Scan: energy\n" in md
    assert "— " not in md.split("\n")[0]  # no em-dash suffix on title
    assert "**Cap filter:**" not in md


def test_render_report_price_under_ticker():
    """current_price should render as 'Price: $X.XX' under the ticker header."""
    from app.sector.scan import render_scan_report, SectorTriageResult, SectorDeepResult

    triage = SectorTriageResult(
        sector="semiconductors", candidates_count=5, survivors=[], cost_usd=0.10
    )
    deep_results = [
        SectorDeepResult(
            ticker="FSLR",
            triage_rank=1,
            triage_reasoning="ok",
            verdict="WATCH",
            confidence="MODERATE",
            thesis_summary="solar thesis",
            cost_usd=0.50,
            current_price=195.20,
        ),
        SectorDeepResult(
            ticker="ABC",
            triage_rank=2,
            triage_reasoning="ok",
            verdict="PASS",
            confidence="HIGH",
            thesis_summary="no price",
            cost_usd=0.50,
        ),  # current_price=None by default
    ]

    md = render_scan_report(
        sector="semiconductors",
        sector_size=50,
        pre_ranked=25,
        triage_result=triage,
        deep_results=deep_results,
        top_n=5,
        total_cost=1.00,
    )

    # FSLR has a price → shows
    assert "### #1: FSLR — WATCH (MODERATE)" in md
    assert "Price: $195.20" in md
    # ABC has no price → no price line for it
    abc_idx = md.find("### #2: ABC")
    abc_section = md[abc_idx : abc_idx + 200]
    assert "Price: $" not in abc_section


def test_render_report_price_between_header_and_thesis():
    """Price line must sit BETWEEN the '###' header and the '**Thesis:**' line."""
    from app.sector.scan import render_scan_report, SectorTriageResult, SectorDeepResult

    triage = SectorTriageResult(sector="energy", candidates_count=1, survivors=[], cost_usd=0.10)
    deep_results = [
        SectorDeepResult(
            ticker="DVN",
            triage_rank=1,
            triage_reasoning="ok",
            verdict="BUY",
            confidence="MODERATE",
            thesis_summary="cheap E&P",
            cost_usd=0.50,
            current_price=45.23,
        ),
    ]

    md = render_scan_report(
        sector="energy",
        sector_size=49,
        pre_ranked=22,
        triage_result=triage,
        deep_results=deep_results,
        top_n=5,
        total_cost=0.60,
    )

    lines = md.split("\n")
    header_line = next(i for i, l in enumerate(lines) if l.startswith("### #1: DVN"))
    price_line = next(i for i, l in enumerate(lines) if l == "Price: $45.23")
    thesis_line = next(i for i, l in enumerate(lines) if l.startswith("**Thesis:**"))

    assert header_line < price_line < thesis_line


def test_tearsheet_renders_valuation_and_flags_boxes_when_scorecard_present():
    """When a scorecard is passed in, the tearsheet must render the
    valuation box and the flags box with explanations."""
    from app.sector.scan import render_scan_report, SectorTriageResult, SectorDeepResult

    triage = SectorTriageResult(
        sector="healthcare_pharma", candidates_count=1, survivors=[], cost_usd=0.10
    )
    deep_results = [
        SectorDeepResult(
            ticker="ANIP",
            triage_rank=1,
            triage_reasoning="serial acquirer",
            verdict="BUY",
            confidence="MODERATE",
            thesis_summary="ANI Pharmaceuticals — cash-EPV materially above GAAP EPV.",
            key_numbers=["FCF $171M", "20% FCF yield"],
            cost_usd=0.50,
            current_price=78.07,
            buy_below_price=70.0,
        ),
    ]
    scorecards = {
        "ANIP": {
            "pricing_zone_detail": {
                "current_price": 78.07,
                "market_cap": 1700.0,
                "dcf_base": 104.0,
                "epv_adjusted": 5.67,
                "epv_cash_adjusted": 26.85,
                "graham_value": 30.5,
            },
            "quality_context": {
                "valuation_headwinds": [
                    "EPV_INTANGIBLE_AMORT_DISTORTION",
                    "NONRECURRING_ITEMS_HEADWIND",
                ],
            },
        },
    }

    md = render_scan_report(
        sector="healthcare_pharma",
        sector_size=261,
        pre_ranked=109,
        triage_result=triage,
        deep_results=deep_results,
        top_n=5,
        total_cost=2.50,
        cap_filter_label="small/mid cap",
        cap_min=300.0,
        cap_max=10000.0,
        scorecards=scorecards,
    )

    # Valuation snapshot table is rendered
    assert "**Valuation snapshot**" in md
    assert "| DCF base | $104.00" in md
    assert "| EPV adjusted (GAAP) | $5.67" in md
    assert "| EPV cash-adjusted | $26.85" in md
    assert "| **Buy below** | $70.00" in md
    # Flags box renders headwinds with explanations
    assert "**Flags box**" in md
    assert "EPV_INTANGIBLE_AMORT_DISTORTION" in md
    assert "GAAP EPV suppressed by purchase-price amortization" in md
    # Key findings now bullet list — NOT comma-joined paragraph
    assert "**Key findings:**" in md
    assert "- FCF $171M" in md
    assert "- 20% FCF yield" in md
    # The reviewer's specific anti-pattern: don't comma-join
    assert "FCF $171M, 20% FCF yield" not in md


def test_tearsheet_falls_back_when_no_scorecard():
    """When scorecards arg is None or missing the ticker, tearsheet still
    renders header + price + thesis + bullets (just skips boxes)."""
    from app.sector.scan import render_scan_report, SectorTriageResult, SectorDeepResult

    triage = SectorTriageResult(sector="energy", candidates_count=1, survivors=[], cost_usd=0.10)
    deep_results = [
        SectorDeepResult(
            ticker="DVN",
            triage_rank=1,
            triage_reasoning="ok",
            verdict="BUY",
            confidence="HIGH",
            thesis_summary="cheap E&P",
            cost_usd=0.50,
            current_price=45.23,
            key_numbers=["a", "b"],
        ),
    ]

    md = render_scan_report(
        sector="energy",
        sector_size=49,
        pre_ranked=22,
        triage_result=triage,
        deep_results=deep_results,
        top_n=5,
        total_cost=0.60,
        # No scorecards arg
    )

    assert "### #1: DVN" in md
    assert "Price: $45.23" in md
    assert "**Thesis:** cheap E&P" in md
    assert "- a" in md and "- b" in md
    # No valuation box without a scorecard
    assert "**Valuation snapshot**" not in md
    assert "**Flags box**" not in md
