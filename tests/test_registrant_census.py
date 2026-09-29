"""Tests for app.universe.registrant_census — the Phase A true-universe census."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.universe.registrant_census import (
    EXCHANGE_SCOPE_IN,
    EXCHANGE_SCOPE_NONE,
    EXCHANGE_SCOPE_OTC,
    EXCHANGE_SCOPE_OTHER,
    OPERATING,
    _classify_operating_status,
    _exchange_scope,
    run_registrant_census,
    select_primary_ticker,
)


# ---------------------------------------------------------------------------
# Primary ticker selection / share-class dedupe
# ---------------------------------------------------------------------------

def test_primary_ticker_prefers_short_unsuffixed_common() -> None:
    listings = [
        ("JPM", "NYSE"),
        ("JPM-PC", "NYSE"),
        ("AMJB", "NYSE"),
        ("JPM-PD", "NYSE"),
        ("VYLD", "NYSE"),
    ]
    assert select_primary_ticker(listings) == "JPM"


def test_primary_ticker_dashed_classes_fall_back_to_registry_order() -> None:
    assert select_primary_ticker([("BRK-B", "NYSE"), ("BRK-A", "NYSE")]) == "BRK-B"


def test_primary_ticker_prefers_in_scope_exchange_listing() -> None:
    assert select_primary_ticker([("ASMLF", "OTC"), ("ASML", "Nasdaq")]) == "ASML"


def test_primary_ticker_empty() -> None:
    assert select_primary_ticker([]) == ""


# ---------------------------------------------------------------------------
# Exchange scope
# ---------------------------------------------------------------------------

def test_exchange_scope_classification() -> None:
    assert _exchange_scope(["Nasdaq"]) == (EXCHANGE_SCOPE_IN, "NASDAQ")
    assert _exchange_scope(["NYSE", "OTC"]) == (EXCHANGE_SCOPE_IN, "NYSE")
    assert _exchange_scope(["OTC"]) == (EXCHANGE_SCOPE_OTC, "OTC")
    assert _exchange_scope(["CBOE"]) == (EXCHANGE_SCOPE_OTHER, "CBOE")
    assert _exchange_scope(["None", ""]) == (EXCHANGE_SCOPE_NONE, "")


# ---------------------------------------------------------------------------
# Operating-company filter
# ---------------------------------------------------------------------------

def test_operating_status_fund_trust_sic_excluded() -> None:
    status, reason = _classify_operating_status(
        primary_ticker="XFND", name="Some Closed End Fund", sic=6726, forms=["10-K"]
    )
    assert status == "FUND_OR_TRUST_SIC"
    assert reason == "sic:6726"
    status, _ = _classify_operating_status(
        primary_ticker="XMGT", name="Mgmt Investment Office", sic=6722, forms=["10-K"]
    )
    assert status == "FUND_OR_TRUST_SIC"


def test_operating_status_blank_check_sic_excluded() -> None:
    status, reason = _classify_operating_status(
        primary_ticker="SPAC", name="Acquisition Corp I", sic=6770, forms=["10-K"]
    )
    assert status == "BLANK_CHECK_SIC"
    assert reason == "sic:6770"


def test_operating_status_warrant_symbol_excluded() -> None:
    status, reason = _classify_operating_status(
        primary_ticker="ABCD-WS", name="ABCD Inc Warrants", sic=3674, forms=["10-K"]
    )
    assert status == "NON_OPERATING_SYMBOL"
    assert reason == "ticker:special_share_class"


def test_operating_status_10k_and_10ksb_filers_pass() -> None:
    status, _ = _classify_operating_status(
        primary_ticker="OPCO", name="Operating Co", sic=3674, forms=["8-K", "10-K", "4"]
    )
    assert status == OPERATING
    status, _ = _classify_operating_status(
        primary_ticker="SMCO", name="Small Co", sic=3674, forms=["10-KSB"]
    )
    assert status == OPERATING
    status, _ = _classify_operating_status(
        primary_ticker="QRCO", name="Quarterly Co", sic=3674, forms=["10-Q/A"]
    )
    assert status == OPERATING


def test_operating_status_foreign_filer_only() -> None:
    status, reason = _classify_operating_status(
        primary_ticker="ADRC", name="Foreign Issuer PLC", sic=3674, forms=["20-F", "6-K"]
    )
    assert status == "FOREIGN_FILER_ONLY"
    assert reason == "forms:20-F/40-F_without_10-K/10-Q"


def test_operating_status_no_operating_forms() -> None:
    status, reason = _classify_operating_status(
        primary_ticker="NOFO", name="Registrant", sic=3674, forms=["8-K", "S-1"]
    )
    assert status == "NO_OPERATING_FORMS"
    assert reason == "forms:no_10-K_or_10-Q"


# ---------------------------------------------------------------------------
# End-to-end census on fixture data (no network)
# ---------------------------------------------------------------------------

_SUBMISSIONS_FIXTURES: dict[str, dict] = {
    # Operating micro-cap on Nasdaq, unknown to sector_inference (the RSSS class).
    "0000000001": {
        "sic": "7372",
        "sicDescription": "Prepackaged Software",
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "8-K"],
                "filingDate": ["2026-03-30", "2026-05-10", "2026-05-20"],
            }
        },
    },
    # Operating, already classified (the known universe).
    "0000000002": {
        "sic": "3821",
        "sicDescription": "Lab Instruments",
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q"],
                "filingDate": ["2026-02-27", "2026-05-01"],
            }
        },
    },
    # Closed-end fund: excluded by SIC even though it files 10-K-like forms.
    "0000000003": {
        "sic": "6726",
        "sicDescription": "Investment Offices NEC",
        "filings": {"recent": {"form": ["N-CEN", "10-K"], "filingDate": ["2026-01-15", "2026-02-15"]}},
    },
    # Foreign private issuer on NYSE: 20-F only.
    "0000000004": {
        "sic": "2834",
        "sicDescription": "Pharma",
        "filings": {"recent": {"form": ["20-F", "6-K"], "filingDate": ["2026-04-30", "2026-05-30"]}},
    },
}

_REGISTRY_FIXTURE = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [1, "NewMicro Software Inc", "NMSW", "Nasdaq"],
        [2, "Known Instruments Corp", "KNWN", "Nasdaq"],
        [2, "Known Instruments Corp", "KNWN-PA", "Nasdaq"],
        [3, "Yield Fund Trust", "YFT", "NYSE"],
        [4, "Foreign Pharma PLC", "FPHA", "NYSE"],
        [5, "OTC Shell Co", "OTCS", "OTC"],
        [6, "Unlisted Registrant", "UNLR", None],
    ],
}


class _StubHttp:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def get_json(self, url: str, **kwargs) -> dict:
        self.calls += 1
        return self.payload


def _fixture_loader(cik: str) -> dict | None:
    return _SUBMISSIONS_FIXTURES.get(cik)


@pytest.fixture()
def census_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    from app.db import init_db

    init_db(cfg)
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute(
        "INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at) "
        "VALUES ('KNWN', '2026-04-19', 'industrial_tech', 1.0, '[]', '2026-04-19T00:00:00Z')"
    )
    conn.commit()
    conn.close()
    yield cfg
    get_config.cache_clear()


def test_run_registrant_census_end_to_end(census_env) -> None:
    cfg = census_env
    report = run_registrant_census(
        cfg=cfg,
        db_path=cfg.db_path,
        http=_StubHttp(_REGISTRY_FIXTURE),
        submissions_loader=_fixture_loader,
        cap_bands=True,
        workers=1,
    )

    assert report["registry"]["listing_rows"] == 7
    assert report["registry"]["distinct_ciks"] == 6
    assert report["exchange_scope_counts"][EXCHANGE_SCOPE_IN] == 4
    assert report["exchange_scope_counts"][EXCHANGE_SCOPE_OTC] == 1
    assert report["exchange_scope_counts"][EXCHANGE_SCOPE_NONE] == 1

    assert report["operating_status_counts"]["OPERATING"] == 2
    assert report["operating_status_counts"]["FUND_OR_TRUST_SIC"] == 1
    assert report["operating_status_counts"]["FOREIGN_FILER_ONLY"] == 1

    assert report["operating_companies"] == 2
    assert report["known_to_sector_inference"] == 1
    assert report["new_operating_companies"] == 1
    assert report["new_operating_tickers"] == ["NMSW"]
    assert report["new_by_exchange"] == {"NASDAQ": 1}
    # No companyfacts shares ingested yet: the new name is pre-ingest unknown.
    assert report["new_by_cap_band"]["band_counts"] == {"pre_ingest_unknown": 1}

    # Registrant table persisted with correct scope/status.
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    rows = {r["cik"]: dict(r) for r in conn.execute("SELECT * FROM sec_registrants")}
    conn.close()
    assert len(rows) == 6
    nmsw = rows["0000000001"]
    assert nmsw["primary_ticker"] == "NMSW"
    assert nmsw["in_scope"] == 1
    assert nmsw["operating_status"] == "OPERATING"
    assert nmsw["sic"] == 7372
    assert nmsw["latest_operating_form_date"] == "2026-05-10"
    knwn = rows["0000000002"]
    assert knwn["primary_ticker"] == "KNWN"
    assert json.loads(knwn["all_tickers"]) == ["KNWN", "KNWN-PA"]
    assert rows["0000000003"]["operating_status"] == "FUND_OR_TRUST_SIC"
    assert rows["0000000005"]["exchange_scope"] == EXCHANGE_SCOPE_OTC
    assert rows["0000000005"]["operating_status"] == "NOT_EVALUATED_OFF_SCOPE_EXCHANGE"


def test_census_rerun_is_idempotent(census_env) -> None:
    cfg = census_env
    kwargs = dict(
        cfg=cfg,
        db_path=cfg.db_path,
        http=_StubHttp(_REGISTRY_FIXTURE),
        submissions_loader=_fixture_loader,
        cap_bands=False,
        workers=1,
    )
    first = run_registrant_census(**kwargs)
    second = run_registrant_census(**kwargs)
    assert first["new_operating_companies"] == second["new_operating_companies"] == 1
    assert second["registrant_table_counts"] == {"added": 0, "updated": 6, "reinstated": 0}

    conn = sqlite3.connect(str(cfg.db_path))
    added_log = conn.execute(
        "SELECT COUNT(*) FROM universe_sync_log WHERE action = 'ADDED'"
    ).fetchone()[0]
    registrants = conn.execute("SELECT COUNT(*) FROM sec_registrants").fetchone()[0]
    conn.close()
    assert registrants == 6
    assert added_log == 6  # only the first run logged additions


def test_scope_transitions_append_exact_on_off_history(tmp_path: Path) -> None:
    from app.universe.registrant_census import RegistrantRecord, _upsert_registrants

    db_path = tmp_path / "engine.db"
    base = dict(
        cik="0000000042",
        primary_ticker="HIST",
        tickers=["HIST"],
        name="HISTORY ISSUER",
        exchange="NASDAQ",
        exchange_scope=EXCHANGE_SCOPE_IN,
    )
    _upsert_registrants(
        [RegistrantRecord(**base, operating_status="NO_OPERATING_FORMS", in_scope=False)],
        db_path=db_path,
        run_kind="fixture",
    )
    _upsert_registrants(
        [RegistrantRecord(**base, operating_status="OPERATING", in_scope=True)],
        db_path=db_path,
        run_kind="fixture",
    )
    _upsert_registrants(
        [RegistrantRecord(**base, operating_status="FOREIGN_FILER_ONLY", in_scope=False)],
        db_path=db_path,
        run_kind="fixture",
    )
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT action, run_kind, cik, ticker, detail FROM universe_sync_log "
        "WHERE action LIKE 'SCOPE_%' ORDER BY rowid"
    ).fetchall()
    conn.close()
    assert rows == [
        (
            "SCOPE_ON",
            "fixture",
            "0000000042",
            "HIST",
            "NO_OPERATING_FORMS -> OPERATING",
        ),
        (
            "SCOPE_OFF",
            "fixture",
            "0000000042",
            "HIST",
            "OPERATING -> FOREIGN_FILER_ONLY",
        ),
    ]


def test_census_unavailable_submissions_marked_not_dropped(census_env) -> None:
    cfg = census_env

    def loader(cik: str) -> dict | None:
        return None

    report = run_registrant_census(
        cfg=cfg,
        db_path=cfg.db_path,
        http=_StubHttp(_REGISTRY_FIXTURE),
        submissions_loader=loader,
        cap_bands=False,
        workers=1,
    )
    assert report["operating_status_counts"]["SUBMISSIONS_UNAVAILABLE"] == 4
    assert report["operating_companies"] == 0
