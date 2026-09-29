from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.autonomous.artifact_financial_audit import (
    AUDIT_SCHEMA_VERSION,
    AUDIT_SCOPE_ID,
    CANONICAL_AUDIT_ROOT_IDS,
)
from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.db import init_db
from app.valuation.lineage import valuation_integrity_fingerprint
from app.watchlist.contract import WatchlistEntry
from app.watchlist.reevaluation import (
    detect_new_evidence,
    reevaluate_entry,
    refresh_watchlist,
)
from app.watchlist.store import add_or_update, get_history, get_latest, mark_status
from tests.financial_integrity_helpers import materialized_no_split_proof


_FIXTURE_SOURCE_RUN_ID = "watchlist_reevaluation_fixture"


def _provider_payload() -> dict[str, object]:
    return {
        "evaluation": "CONFIRMED",
        "summary": "The filing confirms the watchlist thesis.",
        "evidence_references": ["0000000000-26-000011"],
        "thesis_components": [
            {
                "component": "margin stabilization",
                "verdict": "still_holds",
                "evidence": "The filing reports stable margins.",
            }
        ],
    }


class _CountingProvider:
    provider_name = "test"

    def __init__(self) -> None:
        self.calls = 0

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            json_text=json.dumps(_provider_payload()),
            model="test-model",
            usage_input_tokens=100,
            usage_output_tokens=25,
            raw={},
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_audit_roots(cfg) -> dict[str, Path]:
    roots = {
        "autonomous_sector": (Path(cfg.runs_dir) / "autonomous_sector").resolve(),
        "analyst_output": Path(cfg.analyst_outputs_dir).resolve(),
        "scan": (Path(cfg.outputs_dir) / "scans").resolve(),
        "research_output": Path(cfg.research_dir).resolve(),
        "watchlist_report": (Path(cfg.outputs_dir) / "digests").resolve(),
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    return roots


def _install_exact_source_fixture(monkeypatch, tmp_path: Path, cfg) -> Path:
    roots = _canonical_audit_roots(cfg)
    source_dir = roots["autonomous_sector"] / _FIXTURE_SOURCE_RUN_ID
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = (source_dir / "autonomous_sector_run.json").resolve()
    fixture_tickers = ("AAA", "BBB")
    source_path.write_text(
        json.dumps(
            {
                "run_id": _FIXTURE_SOURCE_RUN_ID,
                "sector": "industrial_tech",
                "market_cap_focus": "mid_cap",
                "objective": "Exercise watchlist reevaluation against exact source lineage.",
                "as_of_date": date.today().isoformat(),
                "created_at": f"{date.today().isoformat()}T00:00:00Z",
                "status": "COMPLETED",
                "final_verdict": "NO_SELECTION",
                "scan_family": "normal",
                "company_packets": [
                    {
                        "ticker": ticker,
                        "financial_status": "READY",
                        "model_fit_status": "SUPPORTED",
                        "data_quality_status": "COMPLETE",
                        "current_price": 90.0,
                        "valuation": {
                            "valuation_anchor": 100.0,
                            "anchor_method": "DCF",
                            "buy_price_target": 75.0,
                        },
                        "financial_integrity_status": "PASS",
                    }
                    for ticker in fixture_tickers
                ],
                "relative_ranking": [
                    {
                        "ticker": ticker,
                        "company_autonomy_verdict": "WATCHLIST_ONLY",
                        "company_autonomy_confidence": "MODERATE",
                        "positioning_summary": (
                            f"{ticker} is a good business if margins stabilize."
                        ),
                    }
                    for ticker in fixture_tickers
                ],
                "memo_body": {
                    "candidates": {
                        ticker: {
                            "thesis": f"{ticker} is a good business if margins stabilize.",
                            "key_risks": ["Gross margin compression"],
                            "falsifiers": ["Revenue decline accelerates"],
                            "open_questions": ["Can working capital normalize?"],
                        }
                        for ticker in fixture_tickers
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_scope_id": AUDIT_SCOPE_ID,
        "generated_at": "2026-07-23T20:00:00Z",
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
            "tickers_scanned": len(fixture_tickers),
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
                "path": str(source_path),
                "family": "autonomous_sector",
                "sha256": _sha256(source_path),
                "integrity_status": "PASS",
                "decision_eligible": True,
                "run_id": _FIXTURE_SOURCE_RUN_ID,
            }
        ],
        "violations": [],
    }
    manifest_path = tmp_path / "financial_integrity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "VOE_FINANCIAL_INTEGRITY_MANIFEST",
        str(manifest_path),
    )
    return source_path


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    init_db()
    cfg = get_config()
    _install_exact_source_fixture(monkeypatch, tmp_path, cfg)
    get_config.cache_clear()
    return db_path


def _fixture_cik(ticker: str) -> str:
    return str(sum(ord(char) for char in ticker.upper())).zfill(10)


def _seed_canonical_financial_context(
    db_path: Path,
    *,
    ticker: str,
    as_of_date: str,
) -> None:
    from app.config import get_config
    from app.autonomous.artifact_financial_audit import (
        active_financial_integrity_manifest_path,
    )
    from app.valuation.lineage import valuation_source_record

    cik = _fixture_cik(ticker)
    created_at = f"{as_of_date}T12:00:00+00:00"
    split_proof = materialized_no_split_proof(
        ticker=ticker,
        period_start=as_of_date,
        period_end=as_of_date,
        verified_as_of=as_of_date,
        issuer_cik=cik,
    )
    source_path = (
        Path(get_config().runs_dir)
        / "autonomous_sector"
        / _FIXTURE_SOURCE_RUN_ID
        / "autonomous_sector_run.json"
    ).resolve()
    source_sha256 = _sha256(source_path)
    scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {
            "current_price": 10.0,
            "current_price_as_of_date": as_of_date,
            "current_price_currency": "USD",
            "current_price_source": "fixture_quote",
            "current_price_source_url": (f"https://example.test/quotes/{ticker.upper()}"),
            "current_price_basis": "UNADJUSTED",
            "current_raw_price": 10.0,
            "split_adjustment_factor": 1.0,
            "no_intervening_split_proof": split_proof,
            "dcf_base": 12.0,
            "epv_adjusted": 11.0,
            "gate_action": "PROCEED",
        },
        "quality_context": {
            "gate_action": "PROCEED",
            "confidence_class": "MODERATE",
        },
    }
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET cik=excluded.cik
            """,
            (
                ticker.upper(),
                cik,
                f"{ticker.upper()} Fixture",
                created_at,
            ),
        )
        conn.execute(
            """
            INSERT INTO valuations(
                ticker, as_of_date, method, inputs_json, outputs_json,
                warnings_json, created_at, valuation_writer_version,
                quality_gate_verdict, confidence_class, gate_reason_codes,
                valuation_headwinds, valuation_supports, source_run_id,
                source_artifact_path, source_artifact_sha256,
                financial_integrity_fingerprint
            )
            VALUES(
                ?, ?, 'scorecard', '{}', ?, '[]', ?, 'fixture',
                'PROCEED', 'MODERATE', '[]', '[]', '[]', ?, ?, ?, NULL
            )
            ON CONFLICT(ticker, as_of_date, method) DO UPDATE SET
                inputs_json=excluded.inputs_json,
                outputs_json=excluded.outputs_json,
                warnings_json=excluded.warnings_json,
                created_at=excluded.created_at,
                valuation_writer_version=excluded.valuation_writer_version,
                quality_gate_verdict=excluded.quality_gate_verdict,
                confidence_class=excluded.confidence_class,
                gate_reason_codes=excluded.gate_reason_codes,
                valuation_headwinds=excluded.valuation_headwinds,
                valuation_supports=excluded.valuation_supports,
                source_run_id=excluded.source_run_id,
                source_artifact_path=excluded.source_artifact_path,
                source_artifact_sha256=excluded.source_artifact_sha256,
                financial_integrity_fingerprint=NULL
            """,
            (
                ticker.upper(),
                as_of_date,
                json.dumps(scorecard, sort_keys=True),
                created_at,
                _FIXTURE_SOURCE_RUN_ID,
                str(source_path),
                source_sha256,
            ),
        )
        valuation_row = conn.execute(
            """
            SELECT *
            FROM valuations
            WHERE ticker = ? AND as_of_date = ? AND method = 'scorecard'
            """,
            (ticker.upper(), as_of_date),
        ).fetchone()
        assert valuation_row is not None
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                ?, ?, 'FY', ?, 'shares_outstanding', 10.0,
                'shares_millions', ?, ?, ?, '10-K', ?
            )
            ON CONFLICT(ticker, fiscal_year, period_type, line_item) DO UPDATE SET
                value=excluded.value,
                period_end=excluded.period_end,
                filed_date=excluded.filed_date,
                source_url=excluded.source_url,
                fetched_at=excluded.fetched_at,
                form=excluded.form,
                accession=excluded.accession
            """,
            (
                ticker.upper(),
                int(as_of_date[:4]),
                as_of_date,
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                created_at,
                as_of_date,
                f"{cik}-{as_of_date[2:4]}-000001",
            ),
        )
        source_artifact = json.loads(source_path.read_text(encoding="utf-8"))
        packets_by_ticker = {
            str(packet.get("ticker") or "").strip().upper(): dict(packet)
            for packet in source_artifact.get("company_packets", [])
            if isinstance(packet, dict) and str(packet.get("ticker") or "").strip()
        }
        packets_by_ticker[ticker.upper()] = {
            **packets_by_ticker.get(ticker.upper(), {}),
            "ticker": ticker.upper(),
            "issuer_cik": cik,
            "financial_integrity_status": "PASS",
        }
        source_rows = conn.execute(
            """
            SELECT *
            FROM valuations
            WHERE source_run_id = ?
            ORDER BY ticker, as_of_date, method, created_at
            """,
            (_FIXTURE_SOURCE_RUN_ID,),
        ).fetchall()
        source_records = [valuation_source_record(row) for row in source_rows]
        assert all(record is not None for record in source_records)
        source_artifact["company_packets"] = [
            packets_by_ticker[key] for key in sorted(packets_by_ticker)
        ]
        source_artifact["valuation_source_records"] = source_records
        source_path.write_text(
            json.dumps(source_artifact, sort_keys=True),
            encoding="utf-8",
        )
        final_source_sha256 = _sha256(source_path)
        conn.execute(
            """
            UPDATE valuations
            SET source_artifact_sha256 = ?
            WHERE source_run_id = ?
            """,
            (final_source_sha256, _FIXTURE_SOURCE_RUN_ID),
        )
        refreshed_rows = conn.execute(
            """
            SELECT *
            FROM valuations
            WHERE source_run_id = ?
            ORDER BY id
            """,
            (_FIXTURE_SOURCE_RUN_ID,),
        ).fetchall()
        for refreshed_row in refreshed_rows:
            conn.execute(
                """
                UPDATE valuations
                SET financial_integrity_fingerprint = ?
                WHERE id = ?
                """,
                (
                    valuation_integrity_fingerprint(refreshed_row),
                    refreshed_row["id"],
                ),
            )
        conn.commit()
    finally:
        conn.close()
    manifest_path = active_financial_integrity_manifest_path()
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"]["tickers_scanned"] = len(source_artifact["company_packets"])
    manifest["artifacts"][0]["sha256"] = final_source_sha256
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    submissions_dir = get_config().cache_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    (submissions_dir / f"{cik}.json").write_text(
        json.dumps(
            {
                "tickers": [ticker.upper()],
                "exchanges": ["NYSE"],
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "filingDate": [as_of_date],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _seed_entry(
    db_path: Path,
    *,
    ticker: str = "AAA",
    status: str = "ACTIVE",
    seed_financial_context: bool = True,
) -> WatchlistEntry:
    entry = WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade="WATCHLIST_ONLY",
        confidence="MODERATE",
        conviction_source="company_autonomy",
        valuation_anchor_method="DCF",
        valuation_anchor_value=100.0,
        buy_price_target=75.0,
        current_price_at_addition=90.0,
        thesis_text=f"{ticker.upper()} is a good business if margins stabilize.",
        key_risks=["Gross margin compression"],
        falsifiers=["Revenue decline accelerates"],
        open_questions=["Can working capital normalize?"],
        source_run_id=_FIXTURE_SOURCE_RUN_ID,
        source_sector="industrial_tech",
        added_at="2026-05-01T12:00:00+00:00",
    )
    add_or_update(entry, db_path=db_path)
    if seed_financial_context:
        _seed_canonical_financial_context(
            db_path,
            ticker=ticker,
            as_of_date=date.today().isoformat(),
        )
    latest = get_latest(ticker, db_path=db_path)
    assert latest is not None
    return latest


def _insert_filing(
    db_path: Path,
    *,
    ticker: str = "AAA",
    accession: str = "0000000000-26-000001",
    form_type: str = "10-Q",
    filing_date: str = "2026-05-03",
    local_path: str | None = None,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _fixture_cik(ticker),
                ticker,
                accession,
                form_type,
                filing_date,
                "2026-03-31",
                f"https://example.com/{accession}.htm",
                local_path,
                "parsed",
                "2026-05-03T12:00:00Z",
                "2026-05-03T12:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_detect_new_evidence_returns_10q_and_8k_after_since(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    filing_path = tmp_path / "aaa-10q.htm"
    filing_path.write_text(
        "<html><body>Revenue improved and gross margin stabilized.</body></html>", encoding="utf-8"
    )
    _insert_filing(
        db_path,
        accession="0000000000-26-000001",
        form_type="10-Q",
        filing_date="2026-05-03",
        local_path=str(filing_path),
    )
    _insert_filing(
        db_path, accession="0000000000-26-000002", form_type="8-K", filing_date="2026-05-04"
    )
    _insert_filing(
        db_path, accession="0000000000-26-000003", form_type="10-K", filing_date="2026-05-05"
    )
    _insert_filing(
        db_path, accession="0000000000-26-000004", form_type="10-Q", filing_date="2026-05-01"
    )

    evidence = detect_new_evidence(
        "AAA", since="2026-05-01", db_path=db_path, include_current_events=False
    )

    assert [item.accession for item in evidence] == ["0000000000-26-000001", "0000000000-26-000002"]
    assert [item.form_type for item in evidence] == ["10-Q", "8-K"]
    assert evidence[0].excerpt == "Revenue improved and gross margin stabilized."


def test_reevaluate_no_new_evidence_does_not_call_llm_and_updates_last_evaluated(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)

    def fail_llm(prompt):
        raise AssertionError("LLM should not be called when no new evidence exists")

    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", fail_llm)

    result = reevaluate_entry(entry, db_path=db_path, include_current_events=False)
    latest = get_latest("AAA", db_path=db_path)
    history = get_history("AAA", db_path=db_path)

    assert result.evaluation == "NO_NEW_EVIDENCE"
    assert result.llm_called is False
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert latest.last_evaluated_at is not None
    assert (
        latest.status_reason
        == "NO_MATERIAL_CHANGE: no new 10-Q, 8-K, or dated current-event evidence since last refresh."
    )
    assert history[-1]["field_name"] == "reevaluation"
    assert history[-1]["new_value"] == "NO_NEW_EVIDENCE"


def test_no_evidence_cas_cannot_overwrite_concurrent_manual_removal(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    import app.watchlist.reevaluation as reevaluation_module
    from app.watchlist.store import record_reevaluation_result as real_record

    def race_record(*args, **kwargs):
        mark_status(
            "AAA",
            "REMOVED",
            "Owner removed the entry before the no-evidence write.",
            "manual",
            db_path=db_path,
        )
        return real_record(*args, **kwargs)

    monkeypatch.setattr(
        reevaluation_module,
        "record_reevaluation_result",
        race_record,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        reevaluate_entry(
            entry,
            db_path=db_path,
            include_current_events=False,
        )

    assert "WATCHLIST_REEVALUATION_STATE_MUTATED" in {
        violation.code for violation in exc_info.value.violations
    }
    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None
    assert latest.status == "REMOVED"
    assert latest.status_reason == "Owner removed the entry before the no-evidence write."
    assert all(row["source"] != "reevaluation" for row in get_history("AAA", db_path=db_path))
    assert (
        list((get_config().runs_dir / "watchlist_reevaluation").glob("*/reevaluation.json")) == []
    )


def test_no_evidence_cas_rejects_filing_arriving_before_watermark_write(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    before_entry = get_latest("AAA", db_path=db_path)
    before_history = get_history("AAA", db_path=db_path)
    import app.watchlist.reevaluation as reevaluation_module
    from app.watchlist.store import record_reevaluation_result as real_record

    def race_record(*args, **kwargs):
        _insert_filing(
            db_path,
            accession="0000000000-26-000120",
            form_type="10-Q",
            filing_date="2026-05-03",
        )
        return real_record(*args, **kwargs)

    monkeypatch.setattr(
        reevaluation_module,
        "record_reevaluation_result",
        race_record,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        reevaluate_entry(
            entry,
            db_path=db_path,
            include_current_events=False,
        )

    assert "WATCHLIST_REEVALUATION_EVIDENCE_MUTATED" in {
        violation.code for violation in exc_info.value.violations
    }
    after_entry = get_latest("AAA", db_path=db_path)
    assert before_entry is not None
    assert after_entry is not None
    assert after_entry.to_dict() == before_entry.to_dict()
    assert get_history("AAA", db_path=db_path) == before_history
    assert (
        list((get_config().runs_dir / "watchlist_reevaluation").glob("*/reevaluation.json")) == []
    )


def test_no_evidence_cas_binds_proposed_current_event_watermark(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    before_entry = get_latest("AAA", db_path=db_path)
    import app.watchlist.reevaluation as reevaluation_module
    from app.watchlist.store import record_reevaluation_result as real_record

    event_visible = False

    def current_event_context(_ticker, *, as_of_date):
        assert as_of_date
        documents = []
        if event_visible:
            documents.append(
                SimpleNamespace(
                    source_type="company_news",
                    published_at="2026-04-01T08:30:00+00:00",
                    title="Old event became visible during publication",
                    source_url="https://example.com/aaa/old-event",
                    summary="Older than the explicit evidence boundary.",
                )
            )
        return SimpleNamespace(ordered_documents=documents)

    def expose_watermark_only_event(*args, **kwargs):
        nonlocal event_visible
        event_visible = True
        return real_record(*args, **kwargs)

    monkeypatch.setattr(
        reevaluation_module,
        "load_current_event_context",
        current_event_context,
    )
    monkeypatch.setattr(
        reevaluation_module,
        "record_reevaluation_result",
        expose_watermark_only_event,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        reevaluate_entry(
            entry,
            since="2026-05-01",
            db_path=db_path,
        )

    assert "WATCHLIST_REEVALUATION_EVIDENCE_MUTATED" in {
        violation.code for violation in exc_info.value.violations
    }
    after_entry = get_latest("AAA", db_path=db_path)
    assert before_entry is not None
    assert after_entry is not None
    assert after_entry.to_dict() == before_entry.to_dict()
    assert (
        list((get_config().runs_dir / "watchlist_reevaluation").glob("*/reevaluation.json")) == []
    )


def test_artifact_write_failure_leaves_watchlist_state_unchanged(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    before_entry = get_latest("AAA", db_path=db_path)
    before_history = get_history("AAA", db_path=db_path)

    def fail_artifact_write(**_kwargs):
        raise OSError("simulated artifact write failure")

    monkeypatch.setattr(
        "app.watchlist.reevaluation._persist_reevaluation_artifact",
        fail_artifact_write,
    )

    with pytest.raises(OSError, match="simulated artifact write failure"):
        reevaluate_entry(
            entry,
            db_path=db_path,
            include_current_events=False,
        )

    after_entry = get_latest("AAA", db_path=db_path)
    assert before_entry is not None
    assert after_entry is not None
    assert after_entry.to_dict() == before_entry.to_dict()
    assert get_history("AAA", db_path=db_path) == before_history
    conn = sqlite3.connect(str(db_path))
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM watchlist_reevaluation_publications").fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_current_event_arriving_after_commit_remains_eligible_with_default_since(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    import app.watchlist.reevaluation as reevaluation_module
    import app.watchlist.store as store_module

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 7, 24)

    event_visible = False
    adapter_calls = 0

    def current_event_context(ticker, *, as_of_date):
        nonlocal adapter_calls
        adapter_calls += 1
        assert ticker == "AAA"
        assert as_of_date == "2026-07-24"
        documents = []
        if event_visible:
            documents.append(
                SimpleNamespace(
                    source_type="company_news",
                    published_at="2026-07-24T08:30:00+00:00",
                    title="AAA announces a same-day operating update",
                    source_url="https://example.com/aaa/same-day-update",
                    summary="The update arrived after the prior reevaluation committed.",
                )
            )
        return SimpleNamespace(ordered_documents=documents)

    monkeypatch.setattr(reevaluation_module, "date", FixedDate)
    monkeypatch.setattr(
        reevaluation_module,
        "load_current_event_context",
        current_event_context,
    )
    monkeypatch.setattr(
        store_module,
        "_utc_now_iso",
        lambda: "2026-07-24T12:00:00+00:00",
    )
    real_record = store_module.record_reevaluation_result

    def publish_then_arrive(*args, **kwargs):
        nonlocal event_visible
        result = real_record(*args, **kwargs)
        event_visible = True
        return result

    monkeypatch.setattr(
        reevaluation_module,
        "record_reevaluation_result",
        publish_then_arrive,
    )

    first_result = reevaluate_entry(entry, db_path=db_path)
    committed_entry = get_latest("AAA", db_path=db_path)

    assert first_result.evaluation == "NO_NEW_EVIDENCE"
    assert committed_entry is not None
    assert committed_entry.last_evaluated_at == "2026-07-24T12:00:00+00:00"
    assert committed_entry.current_event_watermark == {
        "as_of_date": "2026-07-24",
        "events": [],
        "schema": "watchlist_current_event_watermark_v1",
    }

    second_result = reevaluate_entry(
        committed_entry,
        dry_run=True,
        db_path=db_path,
    )

    assert second_result.evaluation == "DRY_RUN_NEW_EVIDENCE"
    assert second_result.evidence_count == 1
    assert second_result.evidence_references == ["https://example.com/aaa/same-day-update"]
    assert adapter_calls == 3


def test_undated_current_event_is_not_watermarked_before_it_becomes_dated(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    import app.watchlist.reevaluation as reevaluation_module

    published_at: str | None = None

    def current_event_context(ticker, *, as_of_date):
        assert ticker == "AAA"
        assert as_of_date
        return SimpleNamespace(
            ordered_documents=[
                SimpleNamespace(
                    source_type="company_news",
                    published_at=published_at,
                    title="AAA operating update",
                    source_url="https://example.com/aaa/operating-update",
                    summary="The same source later receives a usable timestamp.",
                )
            ]
        )

    monkeypatch.setattr(
        reevaluation_module,
        "load_current_event_context",
        current_event_context,
    )

    first_result = reevaluate_entry(entry, db_path=db_path)
    committed_entry = get_latest("AAA", db_path=db_path)

    assert first_result.evaluation == "NO_NEW_EVIDENCE"
    assert committed_entry is not None
    assert committed_entry.current_event_watermark == {
        "as_of_date": date.today().isoformat(),
        "events": [],
        "schema": "watchlist_current_event_watermark_v1",
    }

    published_at = f"{date.today().isoformat()}T08:30:00+00:00"
    second_result = reevaluate_entry(
        committed_entry,
        dry_run=True,
        db_path=db_path,
    )

    assert second_result.evaluation == "DRY_RUN_NEW_EVIDENCE"
    assert second_result.evidence_count == 1
    assert second_result.evidence_references == ["https://example.com/aaa/operating-update"]


def test_reevaluate_confirmed_preserves_status_and_persists_artifact(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path, accession="0000000000-26-000005", form_type="10-Q", filing_date="2026-05-03"
    )

    def fake_llm(prompt, **_kwargs):
        return (
            {
                "evaluation": "CONFIRMED",
                "summary": "The new filing confirms the margin-stabilization thesis.",
                "evidence_references": ["0000000000-26-000005"],
                "thesis_components": [
                    {
                        "component": "margin stabilization",
                        "verdict": "strengthened",
                        "evidence": "Gross margin commentary improved.",
                    }
                ],
            },
            {"cost_estimate_usd": 0.18, "provider": "anthropic", "model": "claude-opus-4-6"},
        )

    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", fake_llm)

    result = reevaluate_entry(
        entry, since="2026-05-01", db_path=db_path, include_current_events=False
    )
    latest = get_latest("AAA", db_path=db_path)
    artifact = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        publication = conn.execute(
            "SELECT * FROM watchlist_reevaluation_publications WHERE run_id = ?",
            (result.run_id,),
        ).fetchone()
    finally:
        conn.close()

    assert result.evaluation == "CONFIRMED"
    assert result.new_status == "ACTIVE"
    assert result.llm_called is True
    assert result.cost_estimate_usd == 0.18
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert latest.status_reason == "The new filing confirms the margin-stabilization thesis."
    assert artifact["llm_response"]["evaluation"] == "CONFIRMED"
    assert artifact["evidence"][0]["accession"] == "0000000000-26-000005"
    assert publication is not None
    assert publication["state_applied"] == 1
    assert json.loads(publication["artifact_json"]) == artifact
    assert (
        publication["artifact_sha256"]
        == hashlib.sha256(
            json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )


def test_reevaluation_cannot_overwrite_manual_removal_during_provider_call(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path,
        accession="0000000000-26-000099",
        form_type="8-K",
        filing_date="2026-05-03",
    )

    def mutating_llm(prompt, **_kwargs):
        mark_status(
            "AAA",
            "REMOVED",
            "Owner removed the entry while reevaluation was in flight.",
            "manual",
            db_path=db_path,
        )
        return (
            {
                "evaluation": "CONFIRMED",
                "summary": "This stale response confirms the prior thesis.",
                "evidence_references": ["0000000000-26-000099"],
                "thesis_components": [],
            },
            {"cost_estimate_usd": 0.18},
        )

    monkeypatch.setattr(
        "app.watchlist.reevaluation._call_reevaluation_llm",
        mutating_llm,
    )

    with pytest.raises(InvalidFinancialInputError):
        reevaluate_entry(
            entry,
            since="2026-05-01",
            db_path=db_path,
            include_current_events=False,
        )

    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None
    assert latest.status == "REMOVED"
    assert latest.status_reason == ("Owner removed the entry while reevaluation was in flight.")
    assert all(row["source"] != "reevaluation" for row in get_history("AAA", db_path=db_path))
    assert (
        list((get_config().runs_dir / "watchlist_reevaluation").glob("*/reevaluation.json")) == []
    )


def test_reevaluation_transactional_cas_blocks_post_rebind_state_change(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path,
        accession="0000000000-26-000098",
        form_type="8-K",
        filing_date="2026-05-03",
    )

    monkeypatch.setattr(
        "app.watchlist.reevaluation._call_reevaluation_llm",
        lambda prompt, **_kwargs: (
            {
                "evaluation": "CONFIRMED",
                "summary": "The evidence confirms the prior thesis.",
                "evidence_references": ["0000000000-26-000098"],
                "thesis_components": [],
            },
            {"cost_estimate_usd": 0.18},
        ),
    )
    import app.watchlist.reevaluation as reevaluation_module
    from app.watchlist.store import record_reevaluation_result as real_record

    def race_record(*args, **kwargs):
        mark_status(
            "AAA",
            "REMOVED",
            "Owner removed after post-response authorization.",
            "manual",
            db_path=db_path,
        )
        return real_record(*args, **kwargs)

    monkeypatch.setattr(
        reevaluation_module,
        "record_reevaluation_result",
        race_record,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        reevaluate_entry(
            entry,
            since="2026-05-01",
            db_path=db_path,
            include_current_events=False,
        )

    assert "WATCHLIST_REEVALUATION_STATE_MUTATED" in {
        violation.code for violation in exc_info.value.violations
    }
    latest = get_latest("AAA", db_path=db_path)
    assert latest is not None
    assert latest.status == "REMOVED"
    assert all(row["source"] != "reevaluation" for row in get_history("AAA", db_path=db_path))
    assert (
        list((get_config().runs_dir / "watchlist_reevaluation").glob("*/reevaluation.json")) == []
    )


def test_reevaluation_evidence_cas_blocks_post_rebind_filing_with_one_provider_call(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path,
        accession="0000000000-26-000121",
        form_type="10-Q",
        filing_date="2026-05-03",
    )
    before_entry = get_latest("AAA", db_path=db_path)
    before_history = get_history("AAA", db_path=db_path)
    provider = _CountingProvider()
    import app.watchlist.reevaluation as reevaluation_module
    from app.watchlist.store import record_reevaluation_result as real_record

    monkeypatch.setattr(
        reevaluation_module,
        "get_alpha_llm_provider",
        lambda: provider,
    )

    def race_record(*args, **kwargs):
        _insert_filing(
            db_path,
            accession="0000000000-26-000122",
            form_type="8-K",
            filing_date="2026-05-04",
        )
        return real_record(*args, **kwargs)

    monkeypatch.setattr(
        reevaluation_module,
        "record_reevaluation_result",
        race_record,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        reevaluate_entry(
            entry,
            since="2026-05-01",
            db_path=db_path,
            include_current_events=False,
        )

    assert "WATCHLIST_REEVALUATION_EVIDENCE_MUTATED" in {
        violation.code for violation in exc_info.value.violations
    }
    assert provider.calls == 1
    after_entry = get_latest("AAA", db_path=db_path)
    assert before_entry is not None
    assert after_entry is not None
    assert after_entry.to_dict() == before_entry.to_dict()
    assert get_history("AAA", db_path=db_path) == before_history
    assert (
        list((get_config().runs_dir / "watchlist_reevaluation").glob("*/reevaluation.json")) == []
    )


def test_reevaluate_contradicted_updates_status_with_history(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path, accession="0000000000-26-000006", form_type="8-K", filing_date="2026-05-03"
    )

    def fake_llm(prompt, **_kwargs):
        return (
            {
                "evaluation": "CONTRADICTED",
                "summary": "The 8-K contradicts the thesis because liquidity deteriorated.",
                "evidence_references": ["0000000000-26-000006"],
                "thesis_components": [
                    {
                        "component": "balance-sheet risk",
                        "verdict": "broken",
                        "evidence": "The company disclosed a covenant breach.",
                    }
                ],
            },
            {"cost_estimate_usd": 0.22},
        )

    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", fake_llm)

    result = reevaluate_entry(
        entry, since="2026-05-01", db_path=db_path, include_current_events=False
    )
    latest = get_latest("AAA", db_path=db_path)
    history = get_history("AAA", db_path=db_path)

    assert result.evaluation == "CONTRADICTED"
    assert result.new_status == "CONTRADICTED"
    assert latest is not None
    assert latest.status == "CONTRADICTED"
    assert [row["field_name"] for row in history][-3:] == [
        "status",
        "status_reason",
        "reevaluation",
    ]
    assert history[-3]["old_value"] == "ACTIVE"
    assert history[-3]["new_value"] == "CONTRADICTED"


def test_reevaluate_uncertain_updates_status(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path, accession="0000000000-26-000007", form_type="10-Q", filing_date="2026-05-03"
    )

    def fake_llm(prompt, **_kwargs):
        return (
            {
                "evaluation": "UNCERTAIN",
                "summary": "The filing clouds the thesis but does not break it.",
                "evidence_references": ["0000000000-26-000007"],
                "thesis_components": [
                    {
                        "component": "working capital",
                        "verdict": "weakened",
                        "evidence": "Inventory days increased.",
                    }
                ],
            },
            {"cost_estimate_usd": 0.19},
        )

    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", fake_llm)

    result = reevaluate_entry(
        entry, since="2026-05-01", db_path=db_path, include_current_events=False
    )
    latest = get_latest("AAA", db_path=db_path)

    assert result.evaluation == "UNCERTAIN"
    assert result.new_status == "UNCERTAIN"
    assert latest is not None
    assert latest.status == "UNCERTAIN"


def test_provider_failure_keeps_entry_and_evidence_watermarks_unchanged(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path, accession="0000000000-26-000008", form_type="10-Q", filing_date="2026-05-03"
    )

    def fail_llm(prompt, **_kwargs):
        raise RuntimeError("provider timed out")

    monkeypatch.setattr("app.watchlist.reevaluation._call_reevaluation_llm", fail_llm)

    result = reevaluate_entry(
        entry, since="2026-05-01", db_path=db_path, include_current_events=False
    )
    latest = get_latest("AAA", db_path=db_path)

    assert result.evaluation == "PROVIDER_FAILED"
    assert result.new_status == "ACTIVE"
    assert result.degraded_states == ["LLM_PROVIDER_TIMEOUT"]
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert latest.status_reason is None
    assert latest.last_evaluated_at is None
    assert latest.current_event_watermark is None
    assert (
        len(
            detect_new_evidence(
                "AAA",
                since="2026-05-01",
                db_path=db_path,
                include_current_events=False,
            )
        )
        == 1
    )
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        publication = conn.execute(
            """
            SELECT state_applied, evaluation, artifact_json
            FROM watchlist_reevaluation_publications
            WHERE run_id = ?
            """,
            (result.run_id,),
        ).fetchone()
    finally:
        conn.close()
    assert publication is not None
    assert publication["state_applied"] == 0
    assert publication["evaluation"] == "PROVIDER_FAILED:LLM_PROVIDER_TIMEOUT"
    assert json.loads(publication["artifact_json"])["result"]["evaluation"] == "PROVIDER_FAILED"


def test_openai_watchlist_call_disables_output_expansion_and_uses_one_physical_call(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path,
        accession="0000000000-26-000009",
        form_type="10-Q",
        filing_date="2026-05-03",
    )

    class OpenAIStrictProvider(_CountingProvider):
        provider_name = "openai"

        def __init__(self):
            super().__init__()
            self.allow_output_token_retry: list[bool | None] = []

        def synthesize_json(self, **kwargs):
            self.allow_output_token_retry.append(kwargs.get("allow_output_token_retry"))
            return super().synthesize_json(**kwargs)

    provider = OpenAIStrictProvider()
    monkeypatch.setattr(
        "app.watchlist.reevaluation.get_alpha_llm_provider",
        lambda: provider,
    )

    result = reevaluate_entry(
        entry,
        since="2026-05-01",
        db_path=db_path,
        include_current_events=False,
    )

    assert result.evaluation == "CONFIRMED"
    assert provider.calls == 1
    assert provider.allow_output_token_retry == [False]


def test_failed_attempt_consumes_refresh_budget_and_strict_mode_suppresses_retry(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_entry(db_path, ticker="AAA")
    _seed_entry(db_path, ticker="BBB")
    _insert_filing(
        db_path,
        ticker="AAA",
        accession="0000000195-26-000011",
        form_type="10-Q",
        filing_date="2026-05-03",
    )
    _insert_filing(
        db_path,
        ticker="BBB",
        accession="0000000198-26-000011",
        form_type="10-Q",
        filing_date="2026-05-03",
    )

    class RetryableFailureProvider(_CountingProvider):
        def synthesize_json(self, **_kwargs):
            self.calls += 1
            raise RuntimeError("temporarily unavailable")

    provider = RetryableFailureProvider()
    monkeypatch.setattr(
        "app.watchlist.reevaluation.get_alpha_llm_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        "app.watchlist.reevaluation.estimate_preflight_cost_usd",
        lambda *_args, **_kwargs: 0.01,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._estimated_llm_call_cost_usd",
        lambda _provider, _kwargs: 0.04,
    )

    summary = refresh_watchlist(
        since="2026-05-01",
        max_cost_usd=0.05,
        db_path=db_path,
        include_current_events=False,
    )

    assert provider.calls == 1
    assert len(summary.results) == 2
    assert summary.results[0].evaluation == "PROVIDER_FAILED"
    assert summary.results[0].llm_called is True
    assert summary.results[0].cost_estimate_usd == 0.04
    assert summary.results[1].evaluation == "PROVIDER_FAILED"
    assert summary.results[1].llm_called is False
    assert summary.results[1].cost_estimate_usd == 0.0
    assert summary.results[1].degraded_states == ["LLM_COST_BUDGET_EXCEEDED"]
    assert summary.total_cost_estimate_usd == 0.04
    assert get_latest("AAA", db_path=db_path).last_evaluated_at is None
    assert get_latest("BBB", db_path=db_path).last_evaluated_at is None

    first_artifact = json.loads(Path(summary.results[0].artifact_path).read_text(encoding="utf-8"))
    assert first_artifact["provider_meta"]["physical_call_count"] == 1
    assert first_artifact["provider_meta"]["cost_context"]["cumulative_cost_usd"] == 0.04


def test_refresh_watchlist_respects_max_cost_before_llm(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_entry(db_path)
    _insert_filing(
        db_path, accession="0000000000-26-000009", form_type="10-Q", filing_date="2026-05-03"
    )
    provider = _CountingProvider()
    provider_factory_calls = 0

    def provider_factory():
        nonlocal provider_factory_calls
        provider_factory_calls += 1
        return provider

    monkeypatch.setattr(
        "app.watchlist.reevaluation.estimate_preflight_cost_usd", lambda *args, **kwargs: 0.50
    )
    monkeypatch.setattr(
        "app.watchlist.reevaluation.get_alpha_llm_provider",
        provider_factory,
    )

    summary = refresh_watchlist(
        since="2026-05-01",
        max_cost_usd=0.10,
        db_path=db_path,
        include_current_events=False,
    )
    latest = get_latest("AAA", db_path=db_path)

    assert len(summary.results) == 1
    assert summary.results[0].evaluation == "BUDGET_SKIPPED"
    assert summary.results[0].llm_called is False
    assert provider_factory_calls == 1
    assert provider.calls == 0
    assert latest is not None
    assert latest.status == "ACTIVE"
    assert latest.last_evaluated_at is None


def test_refresh_missing_canonical_financial_scope_has_zero_calls_or_mutation(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path, seed_financial_context=False)
    _insert_filing(
        db_path,
        accession="0000000000-26-000010",
        form_type="10-Q",
        filing_date="2026-05-03",
    )
    before_entry = get_latest("AAA", db_path=db_path)
    before_history = get_history("AAA", db_path=db_path)
    provider = _CountingProvider()
    provider_factory_calls = 0

    def provider_factory():
        nonlocal provider_factory_calls
        provider_factory_calls += 1
        return provider

    monkeypatch.setattr(
        "app.watchlist.reevaluation.get_alpha_llm_provider",
        provider_factory,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        refresh_watchlist(
            ticker="AAA",
            since="2026-05-01",
            db_path=db_path,
            include_current_events=False,
        )

    assert exc_info.value.status == "NEEDS_DATA"
    assert provider_factory_calls == 0
    assert provider.calls == 0
    after_entry = get_latest("AAA", db_path=db_path)
    assert before_entry is not None
    assert after_entry is not None
    assert after_entry.to_dict() == before_entry.to_dict()
    assert get_history("AAA", db_path=db_path) == before_history
    from app.config import get_config

    artifact_root = get_config().runs_dir / "watchlist_reevaluation"
    assert not artifact_root.exists()


def test_refresh_real_provider_path_binds_exact_numeric_prompt_inputs(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_entry(db_path)
    _insert_filing(
        db_path,
        accession="0000000000-26-000011",
        form_type="10-Q",
        filing_date="2026-05-03",
    )
    provider = _CountingProvider()
    monkeypatch.setattr(
        "app.watchlist.reevaluation.get_alpha_llm_provider",
        lambda: provider,
    )
    import app.watchlist.reevaluation as reevaluation_module

    real_bind = reevaluation_module.bind_v1_financial_scope
    captured: dict[str, object] = {}

    def capture_bind(**kwargs):
        bound = real_bind(**kwargs)
        captured["bound"] = bound
        return bound

    monkeypatch.setattr(
        reevaluation_module,
        "bind_v1_financial_scope",
        capture_bind,
    )

    summary = refresh_watchlist(
        ticker="AAA",
        since="2026-05-01",
        db_path=db_path,
        include_current_events=False,
    )

    assert provider.calls == 1
    assert summary.results[0].evaluation == "CONFIRMED"
    bound = captured["bound"]
    packet = bound.packets[0]
    assert packet["current_price"] == 10.0
    assert packet["current_price_unit"] == "USD_per_share"
    assert packet["price_basis"] == "UNADJUSTED"
    assert packet["quote_snapshot_id"]
    assert packet["market_cap_mm"] == 100.0
    assert packet["market_cap_unit"] == "USD_millions"
    assert packet["market_cap_method"] == "price_times_shares_divided_by_issuer_quote_ratio"
    assert packet["shares_outstanding_mm"] == 10.0
    assert packet["raw_shares_outstanding_mm"] == 10.0
    assert packet["shares_unit"] == "shares_millions"
    assert packet["shares_basis"] == "UNADJUSTED"
    assert packet["split_lineage_proof"]["status"] == "PASS"
    scenario = bound.scenarios[0]
    financial_inputs = scenario["financial_inputs"]
    assert financial_inputs["numeric_decision_fields"] == {
        "valuation_anchor_value": 100.0,
        "buy_price_target": 75.0,
        "current_price_at_addition": 90.0,
    }
    provider_request = financial_inputs["provider_request"]
    assert provider_request["schema"] == reevaluation_module.REEVALUATION_SCHEMA
    assert provider_request["schema_name"] == "watchlist_reevaluation"
    assert provider_request["max_output_tokens"] == 1000
    assert '"valuation_anchor_value": 100.0' in provider_request["prompt"]
    assert '"buy_price_target": 75.0' in provider_request["prompt"]
    assert '"current_price_at_addition": 90.0' in provider_request["prompt"]


def test_refresh_legacy_provider_signature_is_adapted_before_one_call(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_entry(db_path)
    _insert_filing(
        db_path,
        accession="0000000000-26-000011",
        form_type="10-Q",
        filing_date="2026-05-03",
    )

    class LegacyProvider(_CountingProvider):
        def synthesize_json(self, *, prompt, schema, schema_name=None):
            self.calls += 1
            return SimpleNamespace(
                json_text=json.dumps(_provider_payload()),
                model="legacy-model",
                usage_input_tokens=100,
                usage_output_tokens=25,
                raw={},
            )

    provider = LegacyProvider()
    monkeypatch.setattr(
        "app.watchlist.reevaluation.get_alpha_llm_provider",
        lambda: provider,
    )
    import app.autonomous.sector_runtime as sector_runtime

    real_require_unchanged = sector_runtime.require_unchanged_financial_integrity_scope
    revalidation_calls = 0

    def counted_require_unchanged(*args, **kwargs):
        nonlocal revalidation_calls
        revalidation_calls += 1
        return real_require_unchanged(*args, **kwargs)

    monkeypatch.setattr(
        sector_runtime,
        "require_unchanged_financial_integrity_scope",
        counted_require_unchanged,
    )

    result = reevaluate_entry(
        get_latest("AAA", db_path=db_path),
        since="2026-05-01",
        db_path=db_path,
        include_current_events=False,
    )

    assert result.evaluation == "CONFIRMED"
    assert provider.calls == 1
    # The logical scope, provider wrapper, pre-reservation prompt binding,
    # and one physical attempt each revalidate the same immutable
    # fingerprint. Signature adaptation must not remove any of those guards
    # or create a second paid attempt.
    assert revalidation_calls == 4


def test_refresh_retry_revalidates_scope_and_propagates_integrity_failure(
    monkeypatch,
    tmp_path,
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    entry = _seed_entry(db_path)
    _insert_filing(
        db_path,
        accession="0000000000-26-000011",
        form_type="10-Q",
        filing_date="2026-05-03",
    )
    before_entry = get_latest("AAA", db_path=db_path)
    before_history = get_history("AAA", db_path=db_path)
    captured: dict[str, object] = {}
    import app.watchlist.reevaluation as reevaluation_module

    real_bind = reevaluation_module.bind_v1_financial_scope

    def capture_bind(**kwargs):
        bound = real_bind(**kwargs)
        captured["bound"] = bound
        return bound

    monkeypatch.setattr(
        reevaluation_module,
        "bind_v1_financial_scope",
        capture_bind,
    )

    class MutatingRetryProvider(_CountingProvider):
        def synthesize_json(self, **_kwargs):
            self.calls += 1
            bound = captured["bound"]
            bound.scenarios[0]["financial_inputs"]["numeric_decision_fields"][
                "buy_price_target"
            ] = 74.0
            raise RuntimeError("temporarily unavailable")

    provider = MutatingRetryProvider()
    monkeypatch.setattr(
        reevaluation_module,
        "get_alpha_llm_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        "app.llm.providers.retry_guard.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        reevaluate_entry(
            entry,
            since="2026-05-01",
            db_path=db_path,
            include_current_events=False,
        )

    assert "BOUND_FINANCIAL_INPUT_MUTATED" in {
        violation.code for violation in exc_info.value.violations
    }
    assert provider.calls == 1
    after_entry = get_latest("AAA", db_path=db_path)
    assert before_entry is not None
    assert after_entry is not None
    assert after_entry.to_dict() == before_entry.to_dict()
    assert get_history("AAA", db_path=db_path) == before_history


def test_buy_confirmed_is_refreshable():
    # BUY_CONFIRMED is the highest-conviction price state (catalyst-confirmed
    # below target). It must stay in the re-evaluation loop so a contradicting
    # 8-K/10-Q can still downgrade the very name you are most likely to act on;
    # otherwise it only re-evaluates after a price move knocks it back down.
    from app.watchlist.reevaluation import REFRESHABLE_STATUSES

    assert "BUY_CONFIRMED" in REFRESHABLE_STATUSES


def _insert_corporate_event(
    db_path: Path,
    *,
    ticker: str = "AAA",
    event_type: str = "unreviewed_8k",
    status: str = "DETECTED",
    detection_date: str = "2026-06-30",
    accession: str | None = "0000000000-26-000101",
    form_type: str = "8-K",
    filing_date: str = "2026-06-30",
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            INSERT INTO corporate_events(
                cik, event_type, anchor_accession, company_name, ticker,
                ticker_state, status, detection_date, detail_json, source_mode,
                detected_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1234567890",
                event_type,
                accession or "0000000000-26-000101",
                "AAA Corp",
                ticker,
                "RESOLVED",
                status,
                detection_date,
                json.dumps({"summary": "Fresh 8-K on an at-target name."}),
                "daily",
                f"{detection_date}T12:00:00+00:00",
                f"{detection_date}T12:00:00+00:00",
            ),
        )
        event_id = int(cursor.lastrowid)
        if accession:
            conn.execute(
                """
                INSERT INTO corporate_event_filings(
                    event_id, cik, accession, form_type, filing_date, role,
                    detail_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    "1234567890",
                    accession,
                    form_type,
                    filing_date,
                    "FRESH_8K",
                    "{}",
                    f"{filing_date}T12:00:00+00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def test_detect_new_evidence_sees_events_feed_8k_not_in_filings_table(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _insert_corporate_event(db_path, filing_date="2026-06-30", detection_date="2026-06-30")

    evidence = detect_new_evidence(
        "AAA", since="2026-06-01", db_path=db_path, include_current_events=False
    )

    assert len(evidence) == 1
    assert evidence[0].source_type == "corporate_event"
    assert evidence[0].accession == "0000000000-26-000101"
    assert evidence[0].form_type == "8-K"
    assert evidence[0].title == "8-K filed 2026-06-30 (UNREVIEWED_8K event)"


def test_detect_new_evidence_open_protection_event_ignores_watermark(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    # 8-K filed BEFORE the since watermark, but the event is still open
    # (undisposed backlog) — it must still surface as evidence.
    _insert_corporate_event(db_path, filing_date="2026-06-30", detection_date="2026-06-30")

    evidence = detect_new_evidence(
        "AAA", since="2026-07-04", db_path=db_path, include_current_events=False
    )

    assert len(evidence) == 1
    assert evidence[0].source_type == "corporate_event"


def test_detect_new_evidence_decided_event_respects_watermark(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _insert_corporate_event(
        db_path, status="DECIDED", filing_date="2026-06-30", detection_date="2026-06-30"
    )

    evidence = detect_new_evidence(
        "AAA", since="2026-07-04", db_path=db_path, include_current_events=False
    )

    assert evidence == []


def test_detect_new_evidence_open_non_protection_event_respects_watermark(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    # busted_ipo is not a queue-protection event type: an open detection older
    # than the watermark must not re-surface on every refresh.
    _insert_corporate_event(
        db_path,
        event_type="busted_ipo",
        accession=None,
        detection_date="2026-06-30",
    )

    evidence = detect_new_evidence(
        "AAA", since="2026-07-04", db_path=db_path, include_current_events=False
    )

    assert evidence == []


def test_detect_new_evidence_dedupes_event_filing_already_in_filings_table(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _insert_filing(
        db_path, accession="0000000000-26-000101", form_type="8-K", filing_date="2026-06-30"
    )
    _insert_corporate_event(db_path, filing_date="2026-06-30", detection_date="2026-06-30")

    evidence = detect_new_evidence(
        "AAA", since="2026-06-01", db_path=db_path, include_current_events=False
    )

    assert [item.accession for item in evidence] == ["0000000000-26-000101"]
    assert evidence[0].source_type == "filing"
