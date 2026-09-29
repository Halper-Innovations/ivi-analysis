"""Point-in-time company-page contracts: filings, valuations, and route edges."""

from __future__ import annotations

import json
import sqlite3

from fastapi.testclient import TestClient

from app.db import init_db
from app.watchlist.schema import ensure_watchlist_schema
from app.web.main import app

client = TestClient(app)


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    ensure_watchlist_schema(db_path)
    return cfg


def _seed_fundamental_versions(cfg) -> None:
    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO companyfacts_facts (
            ticker, fiscal_year, period_type, period_end, line_item, value,
            units, source_url, fetched_at, filed_date, form, accession
        ) VALUES (
            'TMF', 2024, 'FY', '2024-12-31', 'revenue', 100.0,
            'USD_millions', 'https://example.test/TMF',
            '2025-02-15T12:00:00+00:00', '2025-02-15', '10-K', 'tmf-original'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO companyfacts_vintages (
            ticker, fiscal_year, period_type, period_end, line_item, value,
            units, filed_date, form, accession, recorded_at, source_url
        ) VALUES (
            'TMF', 2024, 'FY', '2024-12-31', 'revenue', 90.0,
            'USD_millions', '2025-06-01', '10-K/A', 'tmf-restated',
            '2025-06-01T12:00:00+00:00', 'https://example.test/TMF'
        )
        """
    )
    conn.commit()
    conn.close()


def _seed_valuation_versions(cfg) -> None:
    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        """
        INSERT INTO valuations_history (
            source_id, ticker, as_of_date, method, inputs_json, outputs_json,
            warnings_json, created_at, quality_gate_verdict, confidence_class,
            gate_reason_codes, valuation_headwinds, valuation_supports, archived_at
        ) VALUES (
            17, 'TMV', '2026-05-01', 'dcf', '{}', ?, '[]',
            '2026-05-01T09:00:00+00:00', 'PROCEED', 'HIGH', '[]', '[]', '[]',
            '2026-06-01T09:00:00+00:00'
        )
        """,
        (json.dumps({"status": "OK", "low": 80.0, "base": 100.0, "high": 120.0}),),
    )
    conn.execute(
        """
        INSERT INTO valuations (
            ticker, as_of_date, method, inputs_json, outputs_json, warnings_json,
            created_at, quality_gate_verdict, confidence_class, gate_reason_codes,
            valuation_headwinds, valuation_supports
        ) VALUES (
            'TMV', '2026-06-01', 'dcf', '{}', ?, '[]',
            '2026-06-01T09:00:00+00:00', 'PROCEED', 'HIGH', '[]', '[]', '[]'
        )
        """,
        (json.dumps({"status": "OK", "low": 160.0, "base": 200.0, "high": 240.0}),),
    )
    conn.commit()
    conn.close()


def test_fundamentals_restatement_visibility(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_fundamental_versions(cfg)

    original = client.get(
        "/api/company/TMF/fundamentals",
        params={"basis": "fy", "as_of": "2025-03-01"},
    )
    assert original.status_code == 200
    assert original.json()["as_of"] == "2025-03-01"
    assert original.json()["fiscal_years"] == [2024]
    assert original.json()["series"]["revenue"] == [100.0]

    restated = client.get(
        "/api/company/TMF/fundamentals",
        params={"basis": "fy", "as_of": "2025-07-01"},
    )
    assert restated.status_code == 200
    assert restated.json()["as_of"] == "2025-07-01"
    assert restated.json()["fiscal_years"] == [2024]
    assert restated.json()["series"]["revenue"] == [90.0]

    before_filing = client.get(
        "/api/company/TMF/fundamentals",
        params={"basis": "fy", "as_of": "2024-12-31"},
    )
    assert before_filing.status_code == 200
    assert before_filing.json()["fiscal_years"] == []
    assert before_filing.json()["series"]["revenue"] == []

    live = client.get("/api/company/TMF/fundamentals", params={"basis": "fy"})
    assert live.status_code == 200
    assert live.json()["as_of"] is None
    assert live.json()["fiscal_years"] == [2024]
    assert live.json()["series"]["revenue"] == [100.0]


def test_valuation_record_visibility(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_valuation_versions(cfg)

    archived = client.get("/api/company/TMV", params={"as_of": "2026-05-15"})
    assert archived.status_code == 200
    assert archived.json()["as_of"] == "2026-05-15"
    assert archived.json()["valuations"][0]["method"] == "dcf"
    assert archived.json()["valuations"][0]["fair_value"] == {
        "kind": "band",
        "value": None,
        "low": 80.0,
        "base": 100.0,
        "high": 120.0,
    }

    live = client.get("/api/company/TMV", params={"as_of": "2026-06-15"})
    assert live.status_code == 200
    assert live.json()["valuations"][0]["method"] == "dcf"
    assert live.json()["valuations"][0]["fair_value"] == {
        "kind": "band",
        "value": None,
        "low": 160.0,
        "base": 200.0,
        "high": 240.0,
    }

    before_creation = client.get("/api/company/TMV", params={"as_of": "2026-04-01"})
    assert before_creation.status_code == 200
    assert before_creation.json()["valuations"] == []
    assert before_creation.json()["shelves"] == []


def test_evolution_clips_future_points(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_valuation_versions(cfg)

    response = client.get("/api/company/TMV", params={"as_of": "2026-05-15"})
    assert response.status_code == 200
    assert response.json()["evolution"] == [
        {
            "method": "dcf",
            "as_of_date": "2026-05-01",
            "value": 100.0,
            "archived": True,
            "pre_hardening": True,
        }
    ]


def test_time_machine_contract_edges(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_valuation_versions(cfg)

    malformed = client.get("/api/company/TMV", params={"as_of": "2026-5-1"})
    assert malformed.status_code == 422

    ttm = client.get(
        "/api/company/TMV/fundamentals",
        params={"basis": "ttm", "as_of": "2026-05-15"},
    )
    assert ttm.status_code == 422
    assert ttm.json()["detail"] == "as_of supports basis=fy only"

    unknown = client.get("/api/company/ZZZZ", params={"as_of": "2026-05-15"})
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "Unknown ticker: ZZZZ"

    historical = client.get("/api/company/TMV", params={"as_of": "2026-05-15"})
    assert historical.status_code == 200
    assert historical.json()["as_of"] == "2026-05-15"

    present = client.get("/api/company/TMV")
    assert present.status_code == 200
    assert present.json()["as_of"] is None


def test_time_machine_rejects_impossible_calendar_dates(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_valuation_versions(cfg)

    february_overflow = client.get(
        "/api/company/TDW", params={"as_of": "2026-02-31"}
    )
    assert february_overflow.status_code == 422
    assert february_overflow.json()["detail"] == (
        "as_of must be a valid calendar date (YYYY-MM-DD)"
    )

    month_overflow = client.get(
        "/api/company/TDW", params={"as_of": "2026-13-45"}
    )
    assert month_overflow.status_code == 422
    assert month_overflow.json()["detail"] == (
        "as_of must be a valid calendar date (YYYY-MM-DD)"
    )

    fundamentals_overflow = client.get(
        "/api/company/TDW/fundamentals",
        params={"basis": "fy", "as_of": "2026-02-31"},
    )
    assert fundamentals_overflow.status_code == 422
    assert fundamentals_overflow.json()["detail"] == (
        "as_of must be a valid calendar date (YYYY-MM-DD)"
    )

    valid = client.get("/api/company/TMV", params={"as_of": "2026-05-15"})
    assert valid.status_code == 200
    assert valid.json()["as_of"] == "2026-05-15"
