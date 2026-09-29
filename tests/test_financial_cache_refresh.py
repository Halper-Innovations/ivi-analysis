from __future__ import annotations

from types import SimpleNamespace

from app.autonomous.sector_candidates import SectorCandidateSelection
from app.db import get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _seed_company(ticker: str, cik: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO companies(ticker, cik, name, created_at) VALUES(?, ?, ?, ?)",
            (ticker, cik, f"{ticker} Inc.", utc_now_iso()),
        )


def _insert_companyfacts_rows(ticker: str) -> None:
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES(?, 2025, 'FY', '2025-12-31', 'revenue', 100.0, 'USD_millions', 'test', ?)
            ON CONFLICT(ticker, fiscal_year, period_type, line_item)
            DO UPDATE SET value=excluded.value, fetched_at=excluded.fetched_at
            """,
            (ticker, now),
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value, units, source_url, fetched_at
            ) VALUES(?, 2026, 'Q1', '2026-03-31', 'revenue', 25.0, 'USD_millions', 'test', ?)
            ON CONFLICT(ticker, fiscal_year, period_type, line_item)
            DO UPDATE SET value=excluded.value, fetched_at=excluded.fetched_at
            """,
            (ticker, now),
        )


def _insert_filing_row(ticker: str, cik: str) -> None:
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end, primary_doc_url,
                local_path, status, created_at, updated_at
            ) VALUES(?, ?, ?, '10-K', '2026-02-15', '2025-12-31', 'https://sec.test/doc.htm',
                NULL, 'OK', ?, ?)
            ON CONFLICT(cik, accession) DO UPDATE SET ticker=excluded.ticker, updated_at=excluded.updated_at
            """,
            (cik, ticker, f"{cik}-26-000001", now, now),
        )


def test_resolve_financial_cache_refresh_scope_explicit_and_all_known(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company("CCC", "0000000003")
    _seed_company("BBB", "0000000002")

    from app.ingest.financial_cache_refresh import resolve_financial_cache_refresh_scope

    scope = resolve_financial_cache_refresh_scope(
        tickers=["AAA", "BBB", "AAA"],
        all_known=True,
        max_tickers=3,
    )

    assert scope["tickers"] == ["AAA", "BBB", "CCC"]
    assert scope["ticker_sources"]["AAA"] == ["explicit_tickers"]
    assert scope["ticker_sources"]["BBB"] == ["explicit_tickers", "all_known"]
    assert scope["ticker_sources"]["CCC"] == ["all_known"]
    assert scope["max_tickers"] == 3


def test_resolve_financial_cache_refresh_scope_uses_cache_only_sector_loader(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_load_sector_tickers(**kwargs):
        captured.update(kwargs)
        return [
            ("AAA", "2026-05-01", {}),
            ("BBB", "2026-05-01", {}),
            ("CCC", "2026-05-01", {}),
        ]

    monkeypatch.setattr("app.sector.scan.load_sector_tickers", fake_load_sector_tickers)
    from app.ingest.financial_cache_refresh import resolve_financial_cache_refresh_scope

    scope = resolve_financial_cache_refresh_scope(
        sectors=["enterprise_software"],
        market_cap_focus="smid_cap",
        max_tickers=2,
    )

    assert captured["sector"] == "enterprise_software"
    assert captured["cap_min"] == 500.0  # smid_cap lower bound (2026-06-11 band redefinition)
    assert captured["cap_max"] == 10000.0
    assert scope["tickers"] == ["AAA", "BBB"]
    assert scope["ticker_sources"]["AAA"] == ["sector:enterprise_software"]
    assert scope["sector_selection_mode"] == "cache_order"
    assert scope["sector_selections"][0]["selection_mode"] == "cache_order"
    assert scope["sector_selections"][0]["ranking_basis"] == "sector_scan_db_order_cache_only"


def test_resolve_financial_cache_refresh_scope_can_use_autonomous_candidates(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    captured: dict = {}

    def fake_resolver(**kwargs):
        captured.update(kwargs)
        return SectorCandidateSelection(
            sector=kwargs["sector"],
            market_cap_focus=kwargs["market_cap_focus"],
            selected_tickers=["PAYC", "DBX"],
            source="sector_scan_db",
            loaded_tickers=["PAYC", "DBX", "BOX"],
            excluded_tickers=["BOX"],
            warnings=["FILTERED_UNUSABLE_CANDIDATES:1"],
            ranking_basis="consensus_pre_rank_selectable",
        )

    monkeypatch.setattr("app.autonomous.sector_candidates.resolve_sector_candidate_tickers", fake_resolver)
    from app.ingest.financial_cache_refresh import resolve_financial_cache_refresh_scope

    scope = resolve_financial_cache_refresh_scope(
        sectors=["enterprise_software"],
        market_cap_focus="smid_cap",
        max_tickers_per_sector=2,
        sector_selection_mode="autonomous_candidates",
    )

    assert captured["sector"] == "enterprise_software"
    assert captured["market_cap_focus"] == "smid_cap"
    assert captured["max_candidates"] == 2
    assert captured["filing_risk_use_llm"] is False
    assert scope["sector_selection_mode"] == "autonomous_candidates"
    assert scope["tickers"] == ["PAYC", "DBX"]
    assert scope["ticker_sources"]["PAYC"] == ["sector:enterprise_software"]
    assert scope["sector_selections"][0]["selection_mode"] == "autonomous_candidates"
    assert scope["sector_selections"][0]["selected_tickers"] == ["PAYC", "DBX"]
    assert scope["sector_selections"][0]["ranking_basis"] == "consensus_pre_rank_selectable"
    assert scope["warnings"] == ["FILTERED_UNUSABLE_CANDIDATES:1"]


def test_resolve_financial_cache_refresh_scope_falls_back_when_autonomous_selection_fails(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    def failing_resolver(**kwargs):
        raise RuntimeError("ranker unavailable")

    def fake_load_sector_tickers(**kwargs):
        return [
            ("AAA", "2026-05-01", {}),
            ("BBB", "2026-05-01", {}),
            ("CCC", "2026-05-01", {}),
        ]

    monkeypatch.setattr("app.autonomous.sector_candidates.resolve_sector_candidate_tickers", failing_resolver)
    monkeypatch.setattr("app.sector.scan.load_sector_tickers", fake_load_sector_tickers)
    from app.ingest.financial_cache_refresh import resolve_financial_cache_refresh_scope

    scope = resolve_financial_cache_refresh_scope(
        sectors=["enterprise_software"],
        max_tickers_per_sector=2,
        sector_selection_mode="autonomous_candidates",
    )

    assert scope["sector_selection_mode"] == "autonomous_candidates"
    assert scope["tickers"] == ["AAA", "BBB"]
    assert scope["sector_selections"][0]["selection_mode"] == "cache_order"
    assert scope["sector_selections"][0]["ranking_basis"] == "sector_scan_db_order_cache_only"
    assert scope["warnings"] == ["AUTONOMOUS_CANDIDATE_SELECTION_FAILED:enterprise_software:ranker unavailable"]
    assert scope["sector_selections"][0]["warnings"] == [
        "AUTONOMOUS_CANDIDATE_SELECTION_FAILED:enterprise_software:ranker unavailable"
    ]


def test_resolve_financial_cache_refresh_scope_caps_each_sector_before_global_limit(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    def fake_load_sector_tickers(**kwargs):
        if kwargs["sector"] == "sector_a":
            return [
                ("AAA", "2026-05-01", {}),
                ("AAB", "2026-05-01", {}),
                ("AAC", "2026-05-01", {}),
            ]
        return [
            ("BBA", "2026-05-01", {}),
            ("BBB", "2026-05-01", {}),
            ("BBC", "2026-05-01", {}),
        ]

    monkeypatch.setattr("app.sector.scan.load_sector_tickers", fake_load_sector_tickers)
    from app.ingest.financial_cache_refresh import resolve_financial_cache_refresh_scope

    scope = resolve_financial_cache_refresh_scope(
        sectors=["sector_a", "sector_b"],
        max_tickers_per_sector=2,
        max_tickers=3,
    )

    assert scope["sector_selections"][0]["selected_tickers"] == ["AAA", "AAB"]
    assert scope["sector_selections"][1]["selected_tickers"] == ["BBA", "BBB"]
    assert scope["tickers"] == ["AAA", "AAB", "BBA"]
    assert scope["max_tickers_per_sector"] == 2
    assert scope["max_tickers"] == 3


def test_run_financial_cache_refresh_orchestrates_steps_and_writes_outputs(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_company("AAA", "0000000001")
    _seed_company("BBB", "0000000002")
    calls: list[str] = []

    def fake_ensure(ticker, years_back):
        calls.append(f"facts:{ticker}:{years_back}")
        _insert_companyfacts_rows(ticker)

    def fake_ingest(**kwargs):
        ticker = kwargs["tickers"][0]
        calls.append(f"filings:{ticker}:{kwargs['as_of_date']}")
        _insert_filing_row(ticker, "0000000001" if ticker == "AAA" else "0000000002")
        return {"tickers": 1, "filings_considered": 1, "filings_upserted": 1}

    def fake_context(ticker, **kwargs):
        return SimpleNamespace(
            documents=[object()],
            warnings=[],
            recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE",
            recovered_document_count=0,
        )

    def fake_prices(**kwargs):
        calls.append("prices:" + ",".join(kwargs["tickers"]))
        return {
            "summary_path": str(cfg.outputs_dir / "prices" / kwargs["run_id"] / "prices_summary.json"),
            "rows": [
                {"ticker": "AAA", "status": "OK", "as_of_used": "2026-05-01", "price": 10.0, "source": "test", "reason_code": "PROVIDER_OK", "path": "aaa.json"},
                {"ticker": "BBB", "status": "MISSING", "as_of_used": None, "price": None, "source": None, "reason_code": "PROVIDER_NO_DATA", "path": "bbb.json"},
            ],
        }

    monkeypatch.setattr("app.ingest.financial_cache_refresh.ensure_all_facts", fake_ensure)
    monkeypatch.setattr("app.ingest.financial_cache_refresh.ingest_with_policy", fake_ingest)
    monkeypatch.setattr("app.ingest.financial_cache_refresh.load_research_filing_context", fake_context)
    monkeypatch.setattr("app.ingest.financial_cache_refresh.write_prices_for_run", fake_prices)

    from app.ingest.financial_cache_refresh import run_financial_cache_refresh

    summary = run_financial_cache_refresh(
        tickers=["AAA", "BBB"],
        years=10,
        as_of_date="2026-05-01",
        cfg=cfg,
    )

    assert calls == [
        "facts:AAA:10",
        "filings:AAA:2026-05-01",
        "facts:BBB:10",
        "filings:BBB:2026-05-01",
        "prices:AAA,BBB",
    ]
    assert summary["status"] == "COMPLETED"
    assert summary["ticker_count"] == 2
    assert summary["step_status_counts"]["facts"] == {"OK": 2}
    assert summary["step_status_counts"]["filings"] == {"OK": 2}
    assert summary["step_status_counts"]["price"] == {"MISSING": 1, "OK": 1}
    assert summary["ticker_results"][0]["steps"]["facts"]["annual_rows"] == 1
    assert summary["ticker_results"][0]["steps"]["facts"]["quarterly_rows"] == 1
    assert summary["summary_path"].endswith("refresh_summary.json")
    assert summary["manifest_path"].endswith("refresh_manifest.json")
    assert summary["report_path"].endswith("refresh_report.md")


def test_run_financial_cache_refresh_adds_sector_cache_readiness(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_company("AAA", "0000000001")
    _seed_company("BBB", "0000000002")

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers",
        lambda **kwargs: [
            ("AAA", "2026-05-01", {}),
            ("BBB", "2026-05-01", {}),
        ],
    )

    def fake_ensure(ticker, years_back):
        _insert_companyfacts_rows(ticker)

    def fake_ingest(**kwargs):
        ticker = kwargs["tickers"][0]
        if ticker == "AAA":
            _insert_filing_row(ticker, "0000000001")
        return {"tickers": 1}

    def fake_context(ticker, **kwargs):
        return SimpleNamespace(
            documents=[object()] if ticker == "AAA" else [],
            warnings=[],
            recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE" if ticker == "AAA" else "NO_RECENT_FILINGS_CACHED",
            recovered_document_count=0,
        )

    monkeypatch.setattr("app.ingest.financial_cache_refresh.ensure_all_facts", fake_ensure)
    monkeypatch.setattr("app.ingest.financial_cache_refresh.ingest_with_policy", fake_ingest)
    monkeypatch.setattr("app.ingest.financial_cache_refresh.load_research_filing_context", fake_context)
    monkeypatch.setattr(
        "app.ingest.financial_cache_refresh.write_prices_for_run",
        lambda **kwargs: {
            "summary_path": "prices.json",
            "rows": [
                {"ticker": "AAA", "status": "OK", "as_of_used": "2026-05-01", "price": 10.0, "source": "test", "reason_code": "PROVIDER_OK", "path": "aaa.json"},
                {"ticker": "BBB", "status": "OK", "as_of_used": "2026-05-01", "price": 20.0, "source": "test", "reason_code": "PROVIDER_OK", "path": "bbb.json"},
            ],
        },
    )

    from app.ingest.financial_cache_refresh import render_financial_cache_refresh_report, run_financial_cache_refresh

    summary = run_financial_cache_refresh(
        sectors=["sector_a"],
        market_cap_focus="smid_cap",
        max_tickers_per_sector=2,
        as_of_date="2026-05-01",
        cfg=cfg,
    )
    report = render_financial_cache_refresh_report(summary)

    assert summary["cache_readiness"]["overall_status"] == "CACHE_LIMITED"
    assert summary["cache_readiness"]["sectors"]["sector_a"]["ready_tickers"] == ["AAA"]
    assert summary["cache_readiness"]["sectors"]["sector_a"]["partial_tickers"] == ["BBB"]
    assert summary["cache_readiness"]["sectors"]["sector_a"]["recommended_cache_max_tickers_per_sector"] == 4
    assert "**Overall status:** `CACHE_LIMITED`" in report
    assert "| sector_a | CACHE_LIMITED | AAA | BBB | - | AAA | BBB | CACHE_THIN_CANDIDATE_POOL:1/2 |" in report


def test_run_financial_cache_refresh_records_failures_without_aborting(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_company("AAA", "0000000001")
    _seed_company("BBB", "0000000002")

    def fake_ensure(ticker, years_back):
        if ticker == "BBB":
            raise RuntimeError("boom")
        _insert_companyfacts_rows(ticker)

    monkeypatch.setattr("app.ingest.financial_cache_refresh.ensure_all_facts", fake_ensure)
    monkeypatch.setattr("app.ingest.financial_cache_refresh.ingest_with_policy", lambda **kwargs: {"tickers": 1})
    monkeypatch.setattr(
        "app.ingest.financial_cache_refresh.load_research_filing_context",
        lambda *args, **kwargs: SimpleNamespace(documents=[], warnings=[], recent_filing_status="NO_RECENT_FILINGS_CACHED", recovered_document_count=0),
    )
    monkeypatch.setattr("app.ingest.financial_cache_refresh.write_prices_for_run", lambda **kwargs: {"summary_path": "prices.json", "rows": []})

    from app.ingest.financial_cache_refresh import run_financial_cache_refresh

    summary = run_financial_cache_refresh(tickers=["AAA", "BBB"], with_prices=False, as_of_date="2026-05-01", cfg=cfg)

    assert summary["status"] == "PARTIAL"
    assert summary["error_count"] == 1
    assert summary["ticker_results"][1]["steps"]["facts"]["status"] == "FAILED"
    assert summary["ticker_results"][0]["status"] == "OK"


def test_run_financial_cache_refresh_resume_skips_completed_steps_and_force_reruns(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)
    _seed_company("AAA", "0000000001")
    calls: list[str] = []

    def fake_ensure(ticker, years_back):
        calls.append(f"facts:{ticker}")
        _insert_companyfacts_rows(ticker)

    monkeypatch.setattr("app.ingest.financial_cache_refresh.ensure_all_facts", fake_ensure)
    monkeypatch.setattr("app.ingest.financial_cache_refresh.ingest_with_policy", lambda **kwargs: {"tickers": 1})
    monkeypatch.setattr(
        "app.ingest.financial_cache_refresh.load_research_filing_context",
        lambda *args, **kwargs: SimpleNamespace(documents=[object()], warnings=[], recent_filing_status="RECENT_FILING_CONTEXT_AVAILABLE", recovered_document_count=0),
    )
    monkeypatch.setattr(
        "app.ingest.financial_cache_refresh.write_prices_for_run",
        lambda **kwargs: {"summary_path": "prices.json", "rows": [{"ticker": "AAA", "status": "OK", "as_of_used": "2026-05-01", "price": 10.0, "source": "test", "reason_code": "PROVIDER_OK", "path": "aaa.json"}]},
    )

    from app.ingest.financial_cache_refresh import run_financial_cache_refresh

    first = run_financial_cache_refresh(tickers=["AAA"], as_of_date="2026-05-01", cfg=cfg)
    second = run_financial_cache_refresh(tickers=["AAA"], as_of_date="2026-05-01", resume_run_id=first["run_id"], cfg=cfg)
    third = run_financial_cache_refresh(tickers=["AAA"], as_of_date="2026-05-01", resume_run_id=first["run_id"], force=True, cfg=cfg)

    assert calls == ["facts:AAA", "facts:AAA"]
    assert second["step_status_counts"]["facts"] == {"SKIPPED": 1}
    assert second["step_status_counts"]["price"] == {"SKIPPED": 1}
    assert third["step_status_counts"]["facts"] == {"OK": 1}
