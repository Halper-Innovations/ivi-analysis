"""Tests for app.sector.classifier — SIC-based sector auto-classification."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import AppConfig
from app.sector.classifier import (
    build_sic_to_sector_index,
    classify_all,
    classify_ticker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeCfg:
    db_path: str = ""
    sector_sic_config_path: str = ""


_SECTOR_CONFIG_TWO_SECTORS = {
    "enterprise_software": {
        "label": "Enterprise Software",
        "sic_ranges": [(7370, 7374)],
    },
    "semiconductors": {
        "label": "Semiconductors",
        "sic_ranges": [(3674, 3674)],
    },
}

_SECTOR_CONFIG_OVERLAP = {
    "broad_tech": {
        "label": "Broad Tech",
        "sic_ranges": [(7370, 7380)],  # 11 codes
    },
    "enterprise_software": {
        "label": "Enterprise Software",
        "sic_ranges": [(7372, 7374)],  # 3 codes — more specific
    },
}


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
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


def _insert_scorecard(conn: sqlite3.Connection, ticker: str) -> None:
    conn.execute(
        "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
        "VALUES(?, '2026-04-11', 'scorecard', '{}', '{}', '[]', ?)",
        (ticker, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 1. build_sic_to_sector_index — basic non-overlapping
# ---------------------------------------------------------------------------


def test_build_sic_to_sector_index_basic(monkeypatch):
    monkeypatch.setattr(
        "app.sector.classifier.build_sic_to_sector_index.__wrapped__",
        None,  # placeholder — we patch the dependency instead
    ) if False else None  # noqa — just a comment anchor

    monkeypatch.setattr(
        "app.universe.sector_universe.load_sector_sic_config",
        lambda **kw: _SECTOR_CONFIG_TWO_SECTORS,
    )

    index = build_sic_to_sector_index(cfg=_FakeCfg())

    # enterprise_software covers 7370-7374 (5 codes)
    assert index[7370] == "enterprise_software"
    assert index[7372] == "enterprise_software"
    assert index[7374] == "enterprise_software"
    # semiconductors covers 3674 only
    assert index[3674] == "semiconductors"
    # something outside both ranges is absent
    assert 9999 not in index


# ---------------------------------------------------------------------------
# 2. build_sic_to_sector_index — overlap prefers specific
# ---------------------------------------------------------------------------


def test_build_sic_to_sector_index_overlap_prefers_specific(monkeypatch):
    monkeypatch.setattr(
        "app.universe.sector_universe.load_sector_sic_config",
        lambda **kw: _SECTOR_CONFIG_OVERLAP,
    )

    index = build_sic_to_sector_index(cfg=_FakeCfg())

    # 7372-7374 overlap: enterprise_software (3 codes) beats broad_tech (11 codes)
    assert index[7372] == "enterprise_software"
    assert index[7373] == "enterprise_software"
    assert index[7374] == "enterprise_software"
    # 7370-7371 only in broad_tech
    assert index[7370] == "broad_tech"
    assert index[7371] == "broad_tech"
    # 7375+ only in broad_tech
    assert index[7375] == "broad_tech"
    assert index[7380] == "broad_tech"


# ---------------------------------------------------------------------------
# 3. classify_ticker — success
# ---------------------------------------------------------------------------


def test_classify_ticker_success():
    cik_map = {"AAPL": "320193"}
    sic_index = {7372: "enterprise_software", 3674: "semiconductors"}

    def loader(cik):
        return {"sic": "7372", "name": "Apple Inc"}

    result = classify_ticker(
        "AAPL",
        cik_map=cik_map,
        sic_index=sic_index,
        submissions_loader=loader,
    )
    assert result.status == "classified"
    assert result.sector == "enterprise_software"
    assert result.sic == 7372
    assert result.cik == "320193"
    assert result.score == 1.0
    assert "method:sic_range_match" in result.derived_from


# ---------------------------------------------------------------------------
# 4. classify_ticker — no CIK
# ---------------------------------------------------------------------------


def test_classify_ticker_no_cik():
    result = classify_ticker(
        "ZZZZ",
        cik_map={},
        sic_index={7372: "enterprise_software"},
        submissions_loader=lambda cik: {},
    )
    assert result.status == "no_cik"
    assert result.cik is None
    assert result.sector is None
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# 5. classify_ticker — no SIC in submissions
# ---------------------------------------------------------------------------


def test_classify_ticker_no_sic():
    cik_map = {"FOO": "12345"}

    def loader(cik):
        return {"name": "Foo Corp"}  # no sic field

    result = classify_ticker(
        "FOO",
        cik_map=cik_map,
        sic_index={7372: "enterprise_software"},
        submissions_loader=loader,
    )
    assert result.status == "no_sic"
    assert result.cik == "12345"
    assert result.sic is None
    assert result.sector is None
    assert result.derived_from == ["cik:12345", "name:Foo Corp", "method:no_sic"]


def test_classify_ticker_no_sic_name_fallback_to_capital_markets():
    result = classify_ticker(
        "ARCC",
        cik_map={"ARCC": "1287750"},
        sic_index={},
        submissions_loader=lambda cik: {"name": "ARES CAPITAL CORP"},
    )

    assert result.status == "classified"
    assert result.cik == "1287750"
    assert result.sic is None
    assert result.sector == "capital_markets"
    assert result.score == 1.0
    assert result.derived_from == [
        "cik:1287750",
        "name:ARES CAPITAL CORP",
        "sector:capital_markets",
        "name_pattern:capital_corp",
        "method:no_sic_name_fallback",
    ]


def test_classify_ticker_no_sic_whitehorse_finance_fallback():
    result = classify_ticker(
        "WHF",
        cik_map={"WHF": "1552198"},
        sic_index={},
        submissions_loader=lambda cik: {"name": "WhiteHorse Finance, Inc."},
    )

    assert result.status == "classified"
    assert result.cik == "1552198"
    assert result.sic is None
    assert result.sector == "capital_markets"
    assert result.score == 1.0
    assert result.derived_from == [
        "cik:1552198",
        "name:WhiteHorse Finance, Inc.",
        "sector:capital_markets",
        "name_pattern:finance_entity",
        "method:no_sic_name_fallback",
    ]


def test_classify_ticker_no_sic_allied_gold_fallback():
    result = classify_ticker(
        "AAUC",
        cik_map={"AAUC": "1993344"},
        sic_index={},
        submissions_loader=lambda cik: {"name": "Allied Gold Corp"},
    )

    assert result.status == "classified"
    assert result.cik == "1993344"
    assert result.sic is None
    assert result.sector == "metals_mining"
    assert result.score == 1.0
    assert result.derived_from == [
        "cik:1993344",
        "name:Allied Gold Corp",
        "sector:metals_mining",
        "name_pattern:mining_or_royalty",
        "method:no_sic_name_fallback",
    ]


def test_classify_ticker_no_sic_total_return_name_is_excluded():
    result = classify_ticker(
        "EQS",
        cik_map={"EQS": "878932"},
        sic_index={},
        submissions_loader=lambda cik: {"name": "EQUUS TOTAL RETURN, INC."},
    )

    assert result.status == "excluded_non_operating"
    assert result.sector is None
    assert result.sic is None
    assert "name:total_return_vehicle" in result.derived_from
    assert "method:exclude_non_operating" in result.derived_from


# ---------------------------------------------------------------------------
# 6. classify_ticker — SIC not in any sector range
# ---------------------------------------------------------------------------


def test_classify_ticker_sic_no_sector_match():
    cik_map = {"BAR": "99999"}

    def loader(cik):
        return {"sic": "9999"}

    result = classify_ticker(
        "BAR",
        cik_map=cik_map,
        sic_index={7372: "enterprise_software"},
        submissions_loader=loader,
    )
    assert result.status == "no_sector_match"
    assert result.sic == 9999
    assert result.sector is None
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# 6b. classify_ticker — excluded non-operating symbols
# ---------------------------------------------------------------------------


def test_classify_ticker_excludes_spac_name():
    cik_map = {"AACI": "2092897"}

    def loader(cik):
        return {"sic": "6770", "name": "Armada Acquisition Corp. I"}

    result = classify_ticker(
        "AACI",
        cik_map=cik_map,
        sic_index={6770: "capital_markets"},
        submissions_loader=loader,
    )
    assert result.status == "excluded_non_operating"
    assert result.sector is None
    assert result.sic == 6770
    assert "method:exclude_non_operating" in result.derived_from


def test_classify_ticker_excludes_special_share_class_ticker():
    result = classify_ticker(
        "APO-PA",
        cik_map={"APO-PA": "123456"},
        sic_index={6282: "capital_markets"},
        submissions_loader=lambda cik: {"sic": "6282", "name": "Apollo Global Management"},
    )
    assert result.status == "excluded_non_operating"
    assert result.sector is None
    assert "ticker:special_share_class" in result.derived_from


def test_classify_ticker_excludes_exchange_traded_product_name():
    result = classify_ticker(
        "UVXY",
        cik_map={"UVXY": "1415311"},
        sic_index={6221: "capital_markets"},
        submissions_loader=lambda cik: {
            "sic": "6221",
            "name": "ProShares Ultra VIX Short-Term Futures ETF",
        },
    )
    assert result.status == "excluded_non_operating"
    assert result.sector is None
    assert "method:exclude_non_operating" in result.derived_from


def test_classify_ticker_no_sector_booking_name_fallback():
    result = classify_ticker(
        "BKNG",
        cik_map={"BKNG": "1075531"},
        sic_index={},
        submissions_loader=lambda cik: {"sic": "4700", "name": "Booking Holdings Inc."},
    )

    assert result.status == "classified"
    assert result.cik == "1075531"
    assert result.sic == 4700
    assert result.sector == "internet_services"
    assert result.score == 1.0
    assert result.derived_from == [
        "cik:1075531",
        "sic:4700",
        "name:Booking Holdings Inc.",
        "sector:internet_services",
        "name_pattern:booking",
        "method:no_sector_name_fallback",
    ]


def test_classify_ticker_no_sector_packaging_name_fallback():
    result = classify_ticker(
        "GPK",
        cik_map={"GPK": "1652533"},
        sic_index={},
        submissions_loader=lambda cik: {"sic": "2650", "name": "GRAPHIC PACKAGING HOLDING CO"},
    )

    assert result.status == "classified"
    assert result.cik == "1652533"
    assert result.sic == 2650
    assert result.sector == "consumer_staples"
    assert result.score == 1.0
    assert result.derived_from == [
        "cik:1652533",
        "sic:2650",
        "name:GRAPHIC PACKAGING HOLDING CO",
        "sector:consumer_staples",
        "name_pattern:paper_packaging",
        "method:no_sector_name_fallback",
    ]


# ---------------------------------------------------------------------------
# 7. classify_all — writes to DB
# ---------------------------------------------------------------------------


def test_classify_all_writes_to_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    _create_tables(conn)
    _insert_scorecard(conn, "MSFT")
    _insert_scorecard(conn, "NVDA")
    conn.close()

    fixed_index = {7372: "enterprise_software", 3674: "semiconductors"}
    fixed_cik_map = {"MSFT": "789019", "NVDA": "1045810"}

    monkeypatch.setattr(
        "app.sector.classifier.build_sic_to_sector_index",
        lambda **kw: fixed_index,
    )
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda **kw: fixed_cik_map,
    )

    def fake_submissions(cik):
        return {"789019": {"sic": "7372"}, "1045810": {"sic": "3674"}}.get(cik, {})

    summary = classify_all(
        db_path=db_file,
        submissions_loader=fake_submissions,
        cfg=_FakeCfg(db_path=str(db_file)),
        as_of_date="2026-04-11",
    )

    assert summary.classified == 2
    assert summary.excluded_non_operating == 0
    assert summary.no_cik == 0
    assert summary.total == 2

    # Verify rows actually written
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM sector_inference ORDER BY ticker").fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0]["ticker"] == "MSFT"
    assert rows[0]["inferred_sector"] == "enterprise_software"
    assert rows[1]["ticker"] == "NVDA"
    assert rows[1]["inferred_sector"] == "semiconductors"


# ---------------------------------------------------------------------------
# 8. classify_all — dry run writes zero rows
# ---------------------------------------------------------------------------


def test_classify_all_dry_run(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    _create_tables(conn)
    _insert_scorecard(conn, "MSFT")
    conn.close()

    monkeypatch.setattr(
        "app.sector.classifier.build_sic_to_sector_index",
        lambda **kw: {7372: "enterprise_software"},
    )
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda **kw: {"MSFT": "789019"},
    )

    summary = classify_all(
        db_path=db_file,
        dry_run=True,
        submissions_loader=lambda cik: {"sic": "7372"},
        cfg=_FakeCfg(db_path=str(db_file)),
        as_of_date="2026-04-11",
    )

    assert summary.classified == 1
    assert summary.excluded_non_operating == 0

    # But no rows written
    conn = sqlite3.connect(str(db_file))
    count = conn.execute("SELECT COUNT(*) FROM sector_inference").fetchone()[0]
    conn.close()
    assert count == 0


def test_classify_all_accepts_explicit_ticker_scope_without_scorecards(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    _create_tables(conn)
    conn.close()

    monkeypatch.setattr(
        "app.sector.classifier.build_sic_to_sector_index",
        lambda **kw: {7372: "enterprise_software"},
    )
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda **kw: {"MSFT": "789019"},
    )

    summary = classify_all(
        db_path=db_file,
        tickers=["msft", "MSFT"],
        submissions_loader=lambda cik: {"sic": "7372"},
        cfg=_FakeCfg(db_path=str(db_file)),
        as_of_date="2026-04-11",
    )

    assert summary.total == 1
    assert summary.classified == 1
    assert summary.sector_counts == {"enterprise_software": 1}

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT ticker, inferred_sector FROM sector_inference").fetchone()
    conn.close()
    assert dict(row) == {"ticker": "MSFT", "inferred_sector": "enterprise_software"}


# ---------------------------------------------------------------------------
# 9. classify_all — skip existing
# ---------------------------------------------------------------------------


def test_classify_all_skip_existing(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    _create_tables(conn)
    _insert_scorecard(conn, "MSFT")
    _insert_scorecard(conn, "NVDA")
    # Pre-insert a classification for MSFT
    conn.execute(
        "INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at) "
        "VALUES('MSFT', '2026-04-11', 'enterprise_software', 1.0, '[]', ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "app.sector.classifier.build_sic_to_sector_index",
        lambda **kw: {3674: "semiconductors", 7372: "enterprise_software"},
    )
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda **kw: {"MSFT": "789019", "NVDA": "1045810"},
    )

    summary = classify_all(
        db_path=db_file,
        submissions_loader=lambda cik: {"sic": "3674"},
        cfg=_FakeCfg(db_path=str(db_file)),
        as_of_date="2026-04-11",
    )

    assert summary.skipped_existing == 1
    assert summary.classified == 1  # only NVDA classified
    assert summary.excluded_non_operating == 0

    # Verify MSFT row untouched, NVDA row added
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM sector_inference ORDER BY ticker").fetchall()
    conn.close()
    assert len(rows) == 2
    assert rows[0]["ticker"] == "MSFT"
    assert rows[0]["inferred_sector"] == "enterprise_software"  # original
    assert rows[1]["ticker"] == "NVDA"
    assert rows[1]["inferred_sector"] == "semiconductors"


# ---------------------------------------------------------------------------
# 10. classify_all — force overwrite
# ---------------------------------------------------------------------------


def test_classify_all_force_overwrite(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    _create_tables(conn)
    _insert_scorecard(conn, "MSFT")
    # Pre-insert with old sector
    conn.execute(
        "INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at) "
        "VALUES('MSFT', '2026-04-11', 'old_sector', 0.5, '[]', ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "app.sector.classifier.build_sic_to_sector_index",
        lambda **kw: {7372: "enterprise_software"},
    )
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda **kw: {"MSFT": "789019"},
    )

    summary = classify_all(
        db_path=db_file,
        force=True,
        submissions_loader=lambda cik: {"sic": "7372"},
        cfg=_FakeCfg(db_path=str(db_file)),
        as_of_date="2026-04-11",
    )

    assert summary.skipped_existing == 0
    assert summary.classified == 1
    assert summary.excluded_non_operating == 0

    # Verify row was updated
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM sector_inference WHERE ticker = 'MSFT'").fetchone()
    conn.close()
    assert row["inferred_sector"] == "enterprise_software"
    assert row["score"] == 1.0


# ---------------------------------------------------------------------------
# 11. classify_all — excluded non-operating names are counted and persisted
# ---------------------------------------------------------------------------


def test_classify_all_counts_excluded_non_operating(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    _create_tables(conn)
    _insert_scorecard(conn, "AACI")
    conn.close()

    monkeypatch.setattr(
        "app.sector.classifier.build_sic_to_sector_index",
        lambda **kw: {6770: "capital_markets"},
    )
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.load_ticker_cik_map",
        lambda **kw: {"AACI": "2092897"},
    )

    summary = classify_all(
        db_path=db_file,
        submissions_loader=lambda cik: {"sic": "6770", "name": "Armada Acquisition Corp. I"},
        cfg=_FakeCfg(db_path=str(db_file)),
        as_of_date="2026-04-11",
    )

    assert summary.total == 1
    assert summary.classified == 0
    assert summary.excluded_non_operating == 1
    assert summary.no_sector_match == 0

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM sector_inference WHERE ticker = 'AACI'").fetchone()
    conn.close()

    assert row["inferred_sector"] is None
    assert "method:exclude_non_operating" in row["derived_from"]


# ---------------------------------------------------------------------------
# 12. shipped sector map — expanded operating coverage
# ---------------------------------------------------------------------------


def test_default_sector_map_covers_new_operating_sics():
    config_path = Path(__file__).resolve().parent / "fixtures" / "sector_sic_ranges.json"

    cfg = AppConfig(sector_sic_config_path=config_path)

    index = build_sic_to_sector_index(cfg=cfg)

    assert index[1220] == "energy"
    assert index[1221] == "energy"
    assert index[2111] == "consumer_staples"
    assert index[2211] == "consumer_discretionary"
    assert index[2221] == "consumer_discretionary"
    assert index[2300] == "consumer_discretionary"
    assert index[2430] == "building_products"
    assert index[3411] == "diversified_industrials"
    assert index[3490] == "diversified_industrials"
    assert index[3021] == "consumer_discretionary"
    assert index[3140] == "consumer_discretionary"
    assert index[3231] == "building_products"
    assert index[3241] == "building_products"
    assert index[3272] == "building_products"
    assert index[3949] == "consumer_discretionary"
    assert index[4581] == "transportation_logistics"
    assert index[4955] == "business_services"
    assert index[5030] == "building_products"
    assert index[5031] == "building_products"
    assert index[5045] == "industrial_tech"
    assert index[5065] == "industrial_tech"
    assert index[5160] == "chemicals"
    assert index[5172] == "energy"
    assert index[5810] == "restaurants_food_service"
    assert index[3672] == "semiconductors"
    assert index[3678] == "industrial_tech"
    assert index[4610] == "energy"
    assert index[6035] == "large_cap_financials"
    assert index[6162] == "capital_markets"
    assert index[6163] == "capital_markets"
    assert index[6221] == "capital_markets"
    assert index[7000] == "hospitality_gaming"
    assert index[7200] == "consumer_services"
    assert index[7320] == "business_services"
    assert index[7359] == "business_services"
    assert index[7380] == "business_services"
    assert index[8351] == "education_services"
    assert index[2851] == "chemicals"
    assert index[3677] == "industrial_tech"
    assert index[3751] == "consumer_discretionary"
    assert index[3910] == "consumer_discretionary"
    assert index[3720] == "aerospace_defense"
    assert index[7830] == "media_entertainment"
    assert index[8071] == "healthcare_services"
    assert index[8200] == "education_services"
    assert index[7011] == "hospitality_gaming"
    assert index[1700] == "construction_services"
    assert index[7311] == "media_entertainment"
    assert index[8711] == "business_services"
    assert index[8742] == "business_services"
    assert 6770 not in index
