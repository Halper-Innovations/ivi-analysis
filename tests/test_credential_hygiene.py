from __future__ import annotations

import json
import sqlite3
import stat
from hashlib import sha256
from pathlib import Path

import pytest

from app.config import AppConfig
from app.market.company_facts_provider import fetch_company_facts
from app.util.credential_hygiene import (
    InsecureEnvPermissionsError,
    InvalidSecUserAgentError,
    contains_credential_material,
    require_private_env_file,
    sanitize_json_value,
    sanitize_url_credentials,
    validate_sec_user_agent,
)
from app.util.credential_scrub import run_credential_scrub
from app.util.http import HttpClient


_SECRET = "OLD-EODHD-KEY-123"
_SECRET_URL = (
    "https://eodhd.com/api/eod/AAA.US?api_token=OLD-EODHD-KEY-123"
    "&from=2026-06-01&to=2026-06-08&fmt=json"
)
_PUBLIC_URL = (
    "https://eodhd.com/api/eod/AAA.US?from=2026-06-01&to=2026-06-08&fmt=json"
)


def _price_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE price_quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            provider TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            price REAL,
            currency TEXT,
            source_url TEXT,
            status TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            quote_hash TEXT NOT NULL,
            UNIQUE(ticker, provider, as_of_date)
        )
        """
    )
    for table in ("valuations", "valuations_history"):
        conn.execute(
            f"""
            CREATE TABLE {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inputs_json TEXT NOT NULL,
                outputs_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL
            )
            """
        )
    raw_json = json.dumps(
        {
            "ticker": "AAA",
            "source_url": _SECRET_URL,
            "request_params": {"api_token": _SECRET, "fmt": "json"},
        },
        sort_keys=True,
    )
    conn.execute(
        """
        INSERT INTO price_quotes(
            ticker, provider, as_of_date, price, currency, source_url,
            status, fetched_at, expires_at, raw_json, quote_hash
        ) VALUES('AAA', 'eodhd', '2026-06-08', 42.0, 'USD', ?, 'OK',
                 '2026-06-08T12:00:00+00:00', '2026-06-09T12:00:00+00:00', ?, ?)
        """,
        (_SECRET_URL, raw_json, sha256(raw_json.encode("utf-8")).hexdigest()),
    )
    for table in ("valuations", "valuations_history"):
        conn.execute(
            f"INSERT INTO {table}(inputs_json, outputs_json, warnings_json) "
            "VALUES(?, '{}', '[]')",
            (json.dumps({"price_context": {"price_source_url": _SECRET_URL}}),),
        )
    conn.commit()
    conn.close()


def test_url_and_nested_json_sanitization_remove_credential_material() -> None:
    assert sanitize_url_credentials(_SECRET_URL) == _PUBLIC_URL
    payload = sanitize_json_value(
        {
            "url": _SECRET_URL,
            "params": {"api_token": _SECRET, "fmt": "json"},
            "error": f"request failed for {_SECRET_URL}",
        }
    )
    assert payload == {
        "url": _PUBLIC_URL,
        "params": {"api_token": "REDACTED", "fmt": "json"},
        "error": (
            "request failed for https://eodhd.com/api/eod/AAA.US?api_token=REDACTED"
            "&from=2026-06-01&to=2026-06-08&fmt=json"
        ),
    }
    assert contains_credential_material(payload) is False
    assert _SECRET not in json.dumps(payload, sort_keys=True)


def test_post_rotation_scrub_is_dry_run_first_and_covers_db_json_and_env(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "engine.db"
    _price_db(db_path)
    json_root = tmp_path / "cache" / "prices"
    json_root.mkdir(parents=True)
    cache_path = json_root / "AAA.json"
    cache_path.write_text(
        json.dumps(
            {
                "snapshot": {"url": _SECRET_URL},
                "diagnostic": {"request": {"api_token": _SECRET}},
            }
        ),
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("VOE_EODHD_APIKEY=replacement-key\n", encoding="utf-8")
    env_path.chmod(0o644)

    dry_run = run_credential_scrub(
        database_paths=[db_path],
        json_roots=[json_root],
        env_path=env_path,
    )
    assert dry_run.to_dict() == {
        "apply": False,
        "key_rotation_confirmed": False,
        "databases": (
            {
                "path": str(db_path),
                "rows_scanned": 3,
                "rows_changed": 3,
                "source_urls_changed": 1,
                "raw_json_changed": 1,
                "additional_json_values_changed": 2,
                "tables_changed": (
                    ("price_quotes", 1),
                    ("valuations", 1),
                    ("valuations_history", 1),
                ),
                "applied": False,
            },
        ),
        "json_roots": (
            {
                "path": str(json_root),
                "files_scanned": 1,
                "files_changed": 1,
                "applied": False,
            },
        ),
        "env": {
            "path": str(env_path),
            "exists": True,
            "mode_before": "0644",
            "mode_after": "0644",
            "change_required": True,
            "applied": False,
        },
    }
    assert _SECRET in cache_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o644

    with pytest.raises(RuntimeError, match="KEY_ROTATION_CONFIRMATION_REQUIRED"):
        run_credential_scrub(
            database_paths=[db_path],
            json_roots=[json_root],
            env_path=env_path,
            apply=True,
        )

    applied = run_credential_scrub(
        database_paths=[db_path],
        json_roots=[json_root],
        env_path=env_path,
        apply=True,
        key_rotation_confirmed=True,
    )
    assert applied.databases[0].rows_changed == 3
    assert applied.json_roots[0].files_changed == 1
    assert applied.env is not None
    assert applied.env.mode_after == "0600"

    conn = sqlite3.connect(db_path)
    source_url, raw_json, quote_hash = conn.execute(
        "SELECT source_url, raw_json, quote_hash FROM price_quotes"
    ).fetchone()
    conn.close()
    assert source_url == _PUBLIC_URL
    assert _SECRET not in raw_json
    assert json.loads(raw_json)["request_params"]["api_token"] == "REDACTED"
    assert quote_hash == sha256(raw_json.encode("utf-8")).hexdigest()
    conn = sqlite3.connect(db_path)
    for table in ("valuations", "valuations_history"):
        inputs_json = conn.execute(f"SELECT inputs_json FROM {table}").fetchone()[0]
        assert json.loads(inputs_json)["price_context"]["price_source_url"] == _PUBLIC_URL
        assert _SECRET not in inputs_json
    conn.close()
    assert _SECRET not in cache_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_sec_user_agent_and_env_permissions_fail_closed_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(InvalidSecUserAgentError, match="SEC_USER_AGENT_INVALID"):
        validate_sec_user_agent("ValuationEngine/0.1 (contact: your_email@example.com)")
    with pytest.raises(InvalidSecUserAgentError, match="placeholder contact domain"):
        validate_sec_user_agent("IVI/1.0 (contact: alerts@research.example.org)")
    assert (
        validate_sec_user_agent("IVI/1.0 (contact: operations@ivi-research.com)")
        == "IVI/1.0 (contact: operations@ivi-research.com)"
    )

    env_path = tmp_path / ".env"
    env_path.write_text("VOE_EODHD_APIKEY=replacement-key\n", encoding="utf-8")
    env_path.chmod(0o644)
    with pytest.raises(InsecureEnvPermissionsError, match="mode is 0644"):
        require_private_env_file(env_path)

    cfg = AppConfig(
        cache_dir=tmp_path / "cache",
        sec_user_agent="ValuationEngine/0.1 (contact: your_email@example.com)",
    )
    client = HttpClient(cfg)
    first_cache_path = client.cache_path(
        "https://eodhd.com/api/eod/AAA.US",
        {"api_token": "FIRST-KEY", "fmt": "json"},
    )
    second_cache_path = client.cache_path(
        "https://eodhd.com/api/eod/AAA.US",
        {"api_token": "SECOND-KEY", "fmt": "json"},
    )
    assert first_cache_path == second_cache_path
    network_calls: list[str] = []

    def _unexpected_get(url, **kwargs):
        del kwargs
        network_calls.append(str(url))
        raise AssertionError("network call should not occur")

    monkeypatch.setattr(client.session, "get", _unexpected_get)
    with pytest.raises(InvalidSecUserAgentError, match="SEC_USER_AGENT_INVALID"):
        client.get_bytes(
            "https://data.sec.gov/submissions/CIK0000000001.json",
            use_cache=False,
        )
    assert network_calls == []

    monkeypatch.delenv("VOE_NET_PROVIDER", raising=False)
    monkeypatch.delenv("VOE_LLM_PROVIDER", raising=False)
    result = fetch_company_facts("1", sec_budget=1, cfg=cfg)
    assert result["status"] == "MISSING"
    assert result["reason_code"] == "SEC_USER_AGENT_INVALID"
    assert result["network_attempted"] is False
    assert result["attempts_made"] == 0
