from __future__ import annotations

import json
from datetime import date

from app.db import init_db
from app.ingest.sec_client import SecClient
from app.util.http import HttpClient


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def test_http_client_read_cached_json_uses_public_cache_path(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    client = HttpClient(cfg)
    url = "https://data.sec.gov/submissions/CIK0000000001.json"
    cache_path = client.cache_path(url)
    cache_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    assert client.read_cached_json(url) == {"ok": True}


def test_sec_client_lists_cached_filings_window(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    http = HttpClient(cfg)

    root_url = "https://data.sec.gov/submissions/CIK0000000001.json"
    root_payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000000001-26-000001"],
                "form": ["10-K"],
                "filingDate": ["2026-02-13"],
                "reportDate": ["2025-12-31"],
                "primaryDocument": ["annual.htm"],
            },
            "files": [{"name": "CIK0000000001-submissions-001.json"}],
        }
    }
    child_url = "https://data.sec.gov/submissions/CIK0000000001-submissions-001.json"
    child_payload = {
        "accessionNumber": ["0000000001-25-000099"],
        "form": ["10-Q"],
        "filingDate": ["2025-11-03"],
        "reportDate": ["2025-09-30"],
        "primaryDocument": ["quarterly.htm"],
    }

    http.cache_path(root_url).write_text(json.dumps(root_payload), encoding="utf-8")
    http.cache_path(child_url).write_text(json.dumps(child_payload), encoding="utf-8")

    client = SecClient()
    stubs = client.list_cached_filings_window(
        "0000000001",
        start_date=date(2025, 1, 1),
        end_date=date(2026, 2, 13),
        forms=["10-K", "10-Q"],
    )

    assert [stub.form_type for stub in stubs] == ["10-K", "10-Q"]
    assert stubs[0].primary_doc_url.endswith("/annual.htm")
    assert stubs[1].primary_doc_url.endswith("/quarterly.htm")
