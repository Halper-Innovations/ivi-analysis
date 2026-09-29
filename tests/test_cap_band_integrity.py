from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _legacy_packet_fixture_bypasses_new_integrity_gate(monkeypatch):
    monkeypatch.setattr(
        "app.watchlist.store.artifact_decision_eligibility", lambda payload: "PASS"
    )

from app.autonomous.cap_resolver import CapClassification, SecurityIdentity
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
)
from tests.financial_integrity_helpers import materialized_no_split_proof


def _init_temp_db(monkeypatch, tmp_path) -> Path:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache",
        lambda http=None: {},
    )
    monkeypatch.setattr(
        "app.autonomous.cap_resolver.default_price_lookup",
        lambda *_args, **_kwargs: (lambda ticker, as_of_date: None),
    )
    from app.config import get_config

    get_config.cache_clear()
    return db_path


def _seed_sector_scan_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sector_inference (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            inferred_sector TEXT,
            score REAL NOT NULL DEFAULT 0,
            derived_from TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            UNIQUE(ticker, as_of_date)
        );
        CREATE TABLE IF NOT EXISTS valuations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            method TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
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
        CREATE TABLE IF NOT EXISTS companyfacts_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            period_type TEXT NOT NULL DEFAULT 'FY',
            period_end TEXT NOT NULL,
            line_item TEXT NOT NULL,
            value REAL,
            units TEXT,
            source_url TEXT,
            filed_date TEXT,
            accession TEXT,
            fetched_at TEXT NOT NULL,
            UNIQUE(ticker, fiscal_year, period_type, line_item)
        );
        """
    )
    for ticker in ("INBD", "VNOM", "UNKN"):
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES (?, '2026-06-01', 'energy', '2026-06-01T00:00:00+00:00')",
            (ticker,),
        )
        conn.execute(
            "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)"
            " VALUES (?, '2026-06-01', 'scorecard', '{}', ?, '[]', '2026-06-01T00:00:00+00:00')",
            (
                ticker,
                json.dumps(
                    {
                        "pricing_zone_detail": {
                            "current_price": 155.0,
                            "current_price_as_of_date": "2026-06-01",
                            "current_price_currency": "USD",
                            "current_price_source": "fixture_scorecard_quote",
                            "current_price_source_url": "https://example.test/quote",
                            "current_price_basis": "UNADJUSTED",
                            "split_adjustment_factor": 1.0,
                            "no_intervening_split_proof": materialized_no_split_proof(
                                ticker=ticker,
                                period_start="2026-03-31",
                                period_end="2026-06-01",
                                issuer_cik={
                                    "INBD": "0000000002",
                                    "UNKN": "0000000003",
                                    "VNOM": "0000000001",
                                }[ticker],
                            ),
                        }
                    }
                ),
            ),
        )
    # VNOM profile: strict as-of cap misses but a last-known share count exists.
    conn.execute(
        "INSERT INTO companyfacts_facts "
        "(ticker, fiscal_year, period_type, period_end, line_item, value, units, "
        "source_url, filed_date, accession, fetched_at)"
        " VALUES ('VNOM', 2026, 'Q1', '2026-03-31', 'shares_outstanding', "
        "58.0, 'shares_millions', "
        "'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json', "
        "'2026-05-01', '0000000001-26-000001', "
        "'2026-06-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()


def test_load_sector_tickers_classified_band_integrity(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_sector_scan_db(db_path)
    monkeypatch.setattr(
        "app.autonomous.cap_resolver.resolve_security_identity",
        lambda ticker, **_kwargs: SecurityIdentity(
            ticker=ticker,
            issuer_cik={
                "INBD": "0000000002",
                "UNKN": "0000000003",
                "VNOM": "0000000001",
            }[ticker],
            issuer_primary_ticker=ticker,
            issuer_listed_tickers=(ticker,),
            security_role="PRIMARY",
            is_secondary_class=False,
            is_adr=False,
            identity_source="fixture_primary_security",
            identity_source_url="https://www.sec.gov/submissions/fixture.json",
            identity_as_of_date="2026-06-01",
            identity_confidence="HIGH",
        ),
    )

    def fake_strict(**kwargs):
        if kwargs["ticker"] == "INBD":
            return 250.0, {
                "shares_outstanding": 1.6129,
                "shares_asof_used": "2026-06-01",
                "shares_source_resolution": "companyfacts",
            }
        return None, {"market_cap_reason_code": "SHARES_UNKNOWN"}

    monkeypatch.setattr("app.valuation.shares.resolve_market_cap_from_price_asof", fake_strict)

    from app.sector.scan import load_sector_tickers_classified

    rows, classifications = load_sector_tickers_classified(
        sector="energy",
        db_path=db_path,
        cap_min=0.0,
        cap_max=500.0,
    )
    loaded = sorted(ticker for ticker, _, _ in rows)
    # INBD is in-band strict; UNKN stays sweepable as unknown; VNOM resolved
    # OUT of the micro band via stale shares (58.0mm x $155.00 = $8,990mm).
    assert loaded == ["INBD", "UNKN"]
    assert classifications["VNOM"].cap_source == "stale_shares"
    assert classifications["VNOM"].market_cap_mm == 8990.0
    assert classifications["VNOM"].cap_band == "mid"
    assert classifications["VNOM"].in_band(0.0, 500.0) is False
    assert classifications["INBD"].cap_source == "asof_companyfacts"
    assert classifications["INBD"].cap_band == "micro"
    assert classifications["UNKN"].cap_source == "unknown"
    assert classifications["UNKN"].band_label == "UNKNOWN_CAP"


def test_v2_sector_cap_stage_uses_run_asof_and_does_not_relabel_stale_quote(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_sector_scan_db(db_path)
    captured: list[dict[str, object]] = []
    terminal_lookup = lambda ticker, as_of_date, identity: None

    def fake_classify(ticker: str, **kwargs):
        captured.append({"ticker": ticker, **kwargs})
        return CapClassification(
            ticker=ticker,
            as_of_date=str(kwargs["as_of_date"]),
            market_cap_mm=12_500.0,
            cap_source="terminal_exchange",
            cap_band="large_cap",
            price_used=100.0,
            price_as_of_date="2026-06-11",
        )

    monkeypatch.setattr(
        "app.autonomous.cap_resolver.classify_market_cap_for_band_filter",
        fake_classify,
    )
    monkeypatch.setattr(
        "app.autonomous.cap_resolver.default_price_lookup",
        lambda *_args, **_kwargs: (lambda ticker, as_of_date: None),
    )

    from app.sector.scan import load_sector_tickers_classified

    rows, _classifications = load_sector_tickers_classified(
        sector="energy",
        db_path=db_path,
        cap_min=10_000.0,
        cap_max=None,
        as_of_date="2026-06-11",
        pipeline_version="v2",
        terminal_cap_lookup=terminal_lookup,
    )

    assert sorted(ticker for ticker, _, _ in rows) == ["INBD", "UNKN", "VNOM"]
    assert len(captured) == 3
    assert {row["as_of_date"] for row in captured} == {"2026-06-11"}
    assert {row["asof_price"] for row in captured} == {155.0}
    assert {
        row["asof_price_provenance"]["as_of_date"] for row in captured
    } == {"2026-06-01"}
    assert {row["pipeline_version"] for row in captured} == {"v2"}
    assert all(row["terminal_cap_lookup"] is terminal_lookup for row in captured)


def test_v2_sector_cap_stage_can_freeze_without_live_market_data(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_sector_scan_db(db_path)
    monkeypatch.setenv("VOE_CAP_EODHD_FUNDAMENTALS_ENABLED", "true")
    from app.config import get_config

    get_config.cache_clear()
    crossings: list[str] = []

    def forbidden_price_lookup(*args, **kwargs):
        crossings.append("price_lookup")
        raise AssertionError("live price lookup forbidden during candidate freeze")

    def forbidden_fundamentals(*args, **kwargs):
        crossings.append("fundamentals")
        raise AssertionError("live fundamentals lookup forbidden during candidate freeze")

    def forbidden_terminal(*args, **kwargs):
        crossings.append("terminal")
        raise AssertionError("terminal lookup forbidden during candidate freeze")

    def forbidden_network(*args, **kwargs):
        crossings.append("network")
        raise AssertionError("network request forbidden during candidate freeze")

    monkeypatch.setattr(
        "app.autonomous.cap_resolver.default_price_lookup",
        forbidden_price_lookup,
    )
    monkeypatch.setattr(
        "app.autonomous.cap_resolver._eodhd_fundamentals_market_cap_mm",
        forbidden_fundamentals,
    )
    monkeypatch.setattr("requests.sessions.Session.request", forbidden_network)

    from app.sector.scan import load_sector_tickers_classified

    rows, classifications = load_sector_tickers_classified(
        sector="energy",
        db_path=db_path,
        cap_min=10_000.0,
        cap_max=None,
        as_of_date="2026-06-11",
        pipeline_version="v2",
        terminal_cap_lookup=forbidden_terminal,
        allow_live_market_data=False,
    )

    assert crossings == []
    assert sorted(ticker for ticker, _, _ in rows) == ["INBD", "UNKN", "VNOM"]
    assert all(row.cap_source == "unknown" for row in classifications.values())
    assert all(row.in_band(10_000.0, None) is None for row in classifications.values())


def test_v2_sector_membership_keeps_missing_and_malformed_scorecards(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_sector_scan_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES ('NOSC', '2026-06-02', 'energy', '2026-06-02T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES ('BADJ', '2026-06-03', 'energy', '2026-06-03T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)"
            " VALUES ('BADJ', '2026-06-03', 'scorecard', '{}', '{bad json', '[]', '2026-06-03T00:00:00+00:00')"
        )

    captured: list[tuple[str, object]] = []

    def fake_classify(ticker: str, **kwargs):
        captured.append((ticker, kwargs["asof_price"]))
        return CapClassification(
            ticker=ticker,
            as_of_date=str(kwargs["as_of_date"]),
            market_cap_mm=12_500.0,
            cap_source="terminal_exchange",
            cap_band="large_cap",
        )

    monkeypatch.setattr(
        "app.autonomous.cap_resolver.classify_market_cap_for_band_filter",
        fake_classify,
    )
    monkeypatch.setattr(
        "app.autonomous.cap_resolver.default_price_lookup",
        lambda *_args, **_kwargs: (lambda ticker, as_of_date: None),
    )

    from app.sector.scan import load_sector_tickers_classified

    rows, classifications = load_sector_tickers_classified(
        sector="energy",
        db_path=db_path,
        cap_min=10_000.0,
        cap_max=None,
        as_of_date="2026-06-11",
        pipeline_version="v2",
    )

    by_ticker = {ticker: (row_as_of, scorecard) for ticker, row_as_of, scorecard in rows}
    assert by_ticker["NOSC"] == ("2026-06-02", {})
    assert by_ticker["BADJ"] == ("2026-06-03", {})
    assert set(classifications) == {"INBD", "VNOM", "UNKN", "NOSC", "BADJ"}
    assert ("NOSC", None) in captured
    assert ("BADJ", None) in captured


def test_v1_sector_membership_keeps_missing_and_malformed_scorecards(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_sector_scan_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES ('NOSC', '2026-06-02', 'energy', '2026-06-02T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES ('BADJ', '2026-06-03', 'energy', '2026-06-03T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at)"
            " VALUES ('BADJ', '2026-06-03', 'scorecard', '{}', '{bad json', '[]', '2026-06-03T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES ('OLDNULL', '2026-06-01', 'energy', '2026-06-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES ('OLDNULL', '2026-06-04', NULL, '2026-06-04T00:00:00+00:00')"
        )
        conn.execute(
            "CREATE TABLE sec_registrants (primary_ticker TEXT, removed_at TEXT)"
        )
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES ('REMOVED', '2026-06-01', 'energy', '2026-06-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO sector_inference (ticker, as_of_date, inferred_sector, created_at)"
            " VALUES ('REMOVED', '2026-06-04', NULL, '2026-06-04T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO sec_registrants VALUES ('REMOVED', '2026-06-04T00:00:00Z')"
        )

    from app.sector.scan import load_sector_tickers_classified

    rows, _classifications = load_sector_tickers_classified(
        sector="energy",
        db_path=db_path,
        pipeline_version="v1",
    )

    by_ticker = {ticker: (row_as_of, scorecard) for ticker, row_as_of, scorecard in rows}
    assert by_ticker["NOSC"] == ("2026-06-02", {})
    assert by_ticker["BADJ"] == ("2026-06-03", {})
    assert by_ticker["OLDNULL"] == ("2026-06-01", {})
    assert "REMOVED" not in by_ticker


def test_v2_membership_reconciles_registry_removal_point_in_time(
    monkeypatch, tmp_path
):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    _seed_sector_scan_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE sec_registrants (primary_ticker TEXT, removed_at TEXT)"
        )
        for ticker in ("OLDX", "LATER"):
            conn.execute(
                "INSERT INTO sector_inference "
                "(ticker, as_of_date, inferred_sector, created_at) "
                "VALUES (?, '2026-05-01', 'energy', '2026-05-01T00:00:00+00:00')",
                (ticker,),
            )
        conn.execute(
            "INSERT INTO sec_registrants VALUES ('OLDX', '2026-05-15T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO sec_registrants VALUES ('LATER', '2026-07-01T00:00:00Z')"
        )

    monkeypatch.setattr(
        "app.autonomous.cap_resolver.classify_market_cap_for_band_filter",
        lambda ticker, **kwargs: CapClassification(
            ticker=ticker,
            as_of_date=str(kwargs["as_of_date"]),
            market_cap_mm=12_500.0,
            cap_source="terminal_exchange",
            cap_band="large_cap",
        ),
    )
    monkeypatch.setattr(
        "app.autonomous.cap_resolver.default_price_lookup",
        lambda *_args, **_kwargs: (lambda ticker, as_of_date: None),
    )

    from app.sector.scan import load_sector_tickers_classified

    rows, classifications = load_sector_tickers_classified(
        sector="energy",
        db_path=db_path,
        cap_min=10_000.0,
        cap_max=None,
        as_of_date="2026-06-11",
        pipeline_version="v2",
    )

    loaded = {ticker for ticker, _, _ in rows}
    assert "OLDX" not in loaded
    assert "LATER" in loaded
    assert set(classifications) >= {"OLDX", "LATER"}
    assert classifications["OLDX"].scope_status == "OUT_OF_SCOPE"
    assert classifications["OLDX"].scope_reason == (
        "SEC_REGISTRANT_REMOVED_AS_OF_SCAN"
    )
    assert classifications["LATER"].scope_status == "IN_SCOPE"


def test_resolve_sector_candidates_records_cap_audit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    loader_kwargs: dict[str, object] = {}
    classifications = {
        "INBD": CapClassification(
            ticker="INBD",
            as_of_date="2026-06-01",
            market_cap_mm=250.0,
            cap_source="asof_companyfacts",
            cap_band="micro",
        ),
        "VNOM": CapClassification(
            ticker="VNOM",
            as_of_date="2026-06-01",
            market_cap_mm=8990.0,
            cap_source="stale_shares",
            cap_band="mid",
        ),
        "UNKN": CapClassification(
            ticker="UNKN",
            as_of_date="2026-06-01",
            market_cap_mm=None,
            cap_source="unknown",
            cap_band=None,
        ),
    }
    rows = [("INBD", "2026-06-01", {}), ("UNKN", "2026-06-01", {})]
    def fake_load_sector_tickers_classified(**kwargs):
        loader_kwargs.update(kwargs)
        return rows, classifications

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        fake_load_sector_tickers_classified,
    )
    monkeypatch.setattr("app.sector.scan.pre_rank_sector", lambda **kwargs: [])
    monkeypatch.setattr(
        "app.autonomous.sector_candidates._filter_common_equity_tickers",
        lambda tickers: (tickers, [], []),
    )
    monkeypatch.setattr(
        "app.autonomous.sector_candidates._apply_structural_gate",
        lambda tickers, **kwargs: (tickers, [], {}),
    )

    from app.autonomous.sector_candidates import resolve_sector_candidate_tickers

    selection = resolve_sector_candidate_tickers(
        sector="energy",
        market_cap_focus="micro_cap",
        filing_risk_use_llm=False,
        allow_live_market_data=False,
        coverage_only=True,
    )
    assert loader_kwargs["pipeline_version"] == "v1"
    assert loader_kwargs["allow_live_market_data"] is False
    assert selection.selected_tickers == ["INBD", "UNKN"]
    assert "CAP_RESOLVED_OUT_OF_BAND:1:VNOM" in selection.warnings
    assert "UNKNOWN_CAP_INCLUDED:1" in selection.warnings
    assert "VNOM" in selection.excluded_tickers
    assert selection.cap_classifications["VNOM"]["cap_source"] == "stale_shares"
    assert selection.cap_classifications["UNKN"]["band_label"] == "UNKNOWN_CAP"
    payload = selection.to_dict()
    assert payload["cap_classifications"]["VNOM"]["market_cap_mm"] == 8990.0


def _packet(
    ticker: str,
    *,
    current_price: float,
    market_cap_mm: float | None = None,
    market_cap_source: str | None = None,
    market_cap_category: str | None = None,
) -> SectorCompanyFinancialPacket:
    return SectorCompanyFinancialPacket(
        ticker=ticker,
        financial_status="OK",
        model_fit_status="OK",
        data_quality_status="OK",
        market_cap_category=market_cap_category,
        market_cap_mm=market_cap_mm,
        market_cap_source=market_cap_source,
        current_price=current_price,
        valuation={"anchor_method": "DCF", "valuation_anchor": 106.67, "buy_price_target": 80.0},
    )


def _artifact() -> AutonomousSectorFinancialRunArtifact:
    return AutonomousSectorFinancialRunArtifact(
        run_id="autonomous_sector_energy_capband_test",
        sector="energy",
        market_cap_focus="micro_cap",
        objective="Cap-band intake test.",
        as_of_date="2026-06-01",
        created_at="2026-06-01T12:00:00Z",
        completed_at="2026-06-01T12:01:00Z",
        status="COMPLETED",
        final_verdict="WATCHLIST",
        selected_ticker=None,
        confidence="MODERATE",
        candidate_selection={"selected_tickers": ["AAA", "BBB"]},
        company_packets=[
            _packet(
                "AAA",
                current_price=70.0,
                market_cap_mm=150.0,
                market_cap_source="stale_shares",
                market_cap_category="micro",
            ),
            _packet("BBB", current_price=95.0),
        ],
        relative_ranking=[
            {
                "ticker": "AAA",
                "company_autonomy_verdict": "ACTIONABLE",
                "company_autonomy_confidence": "HIGH",
            },
            {
                "ticker": "BBB",
                "company_autonomy_verdict": "WATCHLIST_ONLY",
                "company_autonomy_confidence": "MODERATE",
            },
        ],
    )


def test_populate_carries_cap_provenance_onto_rows(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    from app.watchlist.store import get_latest, populate_from_sector_artifact

    result = populate_from_sector_artifact(_artifact(), db_path=db_path)
    assert result.added_or_updated == 2

    aaa = get_latest("AAA", db_path=db_path)
    assert aaa is not None
    assert aaa.market_cap_mm == 150.0
    assert aaa.cap_source == "stale_shares"
    assert aaa.cap_band == "micro"
    assert aaa.cap_asof == "2026-06-01"

    # Legacy-shaped packet without cap provenance falls back to the light
    # offline chain: no companyfacts table in this DB -> explicit unknown.
    bbb = get_latest("BBB", db_path=db_path)
    assert bbb is not None
    assert bbb.cap_source == "unknown"
    assert bbb.cap_band is None
    assert bbb.market_cap_mm is None


def test_watchlist_queue_band_scope_excludes_out_of_band_and_unknown(monkeypatch, tmp_path):
    db_path = _init_temp_db(monkeypatch, tmp_path)
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.store import add_or_update, watchlist_queue

    def _entry(ticker: str, market_cap_mm: float | None, cap_band: str | None, cap_source: str) -> WatchlistEntry:
        return WatchlistEntry(
            ticker=ticker,
            status="ACTIVE",
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="company_autonomy",
            buy_price_target=80.0,
            current_price_at_addition=100.0,
            source_run_id="run_capband",
            source_sector="energy",
            added_at="2026-06-01T12:00:00+00:00",
            market_cap_mm=market_cap_mm,
            cap_source=cap_source,
            cap_band=cap_band,
            cap_asof="2026-06-01",
        )

    add_or_update(_entry("MICR", 150.0, "micro", "asof_companyfacts"), db_path=db_path)
    add_or_update(_entry("VNOM", 8990.0, "mid", "stale_shares"), db_path=db_path)
    add_or_update(_entry("UNKN", None, None, "unknown"), db_path=db_path)

    micro_rows = watchlist_queue(limit=25, band="micro_cap", db_path=db_path)
    assert [row["ticker"] for row in micro_rows] == ["MICR"]
    assert micro_rows[0]["cap_band_label"] == "micro"

    all_rows = watchlist_queue(limit=25, db_path=db_path)
    labels = {row["ticker"]: row["cap_band_label"] for row in all_rows}
    assert labels == {"MICR": "micro", "VNOM": "mid", "UNKN": "UNKNOWN_CAP"}

    with pytest.raises(ValueError):
        watchlist_queue(limit=25, band="bogus_band", db_path=db_path)


def test_packet_builder_propagates_cap_classification(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.sector_financial_packets import (
        build_sector_company_financial_packets_from_signal_packets,
    )

    signal = TickerSignalPacket(ticker="AAA", current_price=70.0, dcf_value=150.0)
    packets = build_sector_company_financial_packets_from_signal_packets(
        {"AAA": signal},
        sector="energy",
        as_of_date="2026-06-01",
        cap_classifications={
            "AAA": {
                "market_cap_mm": 150.0,
                "cap_source": "stale_shares",
                "cap_band": "micro",
                "band_label": "micro",
                "cap_effective_as_of_date": "2026-05-29",
                "cap_source_kind": "SEC",
                "cap_source_name": "sec_companyfacts_stale_shares_derived",
                "cap_source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                "cap_confidence": "MEDIUM",
                "price_used": 70.0,
                "price_as_of_date": "2026-05-29",
                "price_source": "exchange_close",
                "price_source_url": "https://exchange.example.com/AAA/close",
                "price_confidence": "HIGH",
                "issuer_cik": "0000000001",
                "issuer_primary_ticker": "AAA",
                "issuer_listed_tickers": ["AAA", "AAA.B"],
                "security_role": "PRIMARY",
                "is_secondary_class": False,
                "is_adr": False,
                "identity_source": "sec_submissions_cache",
                "identity_source_url": "https://data.sec.gov/submissions/CIK0000000001.json",
                "identity_confidence": "HIGH",
            }
        },
    )
    assert len(packets) == 1
    assert packets[0].market_cap_mm == 150.0
    assert packets[0].market_cap_source == "stale_shares"
    assert packets[0].market_cap_category == "micro"
    assert packets[0].market_cap_effective_as_of_date == "2026-05-29"
    assert packets[0].market_cap_source_kind == "SEC"
    assert packets[0].market_cap_source_name == "sec_companyfacts_stale_shares_derived"
    assert packets[0].market_cap_source_url == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"
    )
    assert packets[0].market_cap_confidence == "MEDIUM"
    assert packets[0].cap_stage_price == 70.0
    assert packets[0].cap_stage_price_as_of_date == "2026-05-29"
    assert packets[0].cap_stage_price_source == "exchange_close"
    assert packets[0].cap_stage_price_source_url == (
        "https://exchange.example.com/AAA/close"
    )
    assert packets[0].cap_stage_price_confidence == "HIGH"
    assert packets[0].issuer_cik == "0000000001"
    assert packets[0].issuer_primary_ticker == "AAA"
    assert packets[0].issuer_listed_tickers == ["AAA", "AAA.B"]
    assert packets[0].security_role == "PRIMARY"
    assert packets[0].is_secondary_class is False
    assert packets[0].is_adr is False
    assert packets[0].identity_source == "sec_submissions_cache"
    assert packets[0].identity_source_url == (
        "https://data.sec.gov/submissions/CIK0000000001.json"
    )
    assert packets[0].identity_confidence == "HIGH"


def test_v2_packet_uses_cap_stage_price_when_signal_price_was_lost(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    from app.alpha.schemas import TickerSignalPacket
    from app.autonomous.sector_financial_packets import (
        build_sector_company_financial_packets_from_signal_packets,
    )

    signal = TickerSignalPacket(ticker="ZXAA", current_price=None, dcf_value=150.0)
    packets = build_sector_company_financial_packets_from_signal_packets(
        {"ZXAA": signal},
        sector="energy",
        as_of_date="2026-07-15",
        pipeline_version="v2",
        cap_classifications={
            "ZXAA": {
                "market_cap_mm": 12_500.0,
                "cap_source": "terminal_exchange",
                "cap_band": "large_cap",
                "price_used": 100.0,
                "price_currency": "USD",
                "price_as_of_date": "2026-07-15",
                "price_source": "exchange_close",
                "price_source_url": "https://exchange.example/ZXAA/close",
                "price_confidence": "HIGH",
            }
        },
    )

    assert signal.current_price is None
    assert packets[0].current_price == 100.0
    assert packets[0].cap_stage_price == 100.0
    assert "MISSING_PRICE" not in packets[0].blockers
    assert packets[0].valuation["discount_to_anchor"] == pytest.approx(1 / 3)
