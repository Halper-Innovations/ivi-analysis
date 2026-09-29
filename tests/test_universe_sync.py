"""Tests for app.universe.registrant_intake.sync_universe — Phase E weekly sync."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.universe.registrant_intake import sync_universe


_SUBMISSIONS = {
    "0000000030": {
        "sic": "3674",
        "sicDescription": "Semiconductors",
        "filings": {"recent": {"form": ["10-K", "10-Q"], "filingDate": ["2026-03-01", "2026-05-01"]}},
    },
    "0000000031": {
        "sic": "3674",
        "sicDescription": "Semiconductors",
        "filings": {"recent": {"form": ["10-K"], "filingDate": ["2026-02-01"]}},
    },
}


def _loader(cik: str):
    return _SUBMISSIONS.get(str(cik).strip().zfill(10))


class _StubHttp:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def get_json(self, url: str, **kwargs) -> dict:
        return self.payload


_REGISTRY_V1 = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [30, "Alpha Semi Inc", "ASMI", "Nasdaq"],
        [31, "Beta Semi Corp", "BSMC", "Nasdaq"],
    ],
}

_REGISTRY_V2 = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [
        [30, "Alpha Semi Inc", "ASMI", "Nasdaq"],
        # BSMC (cik 31) left the registry; nothing new arrived.
    ],
}


@pytest.fixture()
def sync_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv(
        "VOE_SECTOR_SIC_CONFIG_PATH",
        str(Path(__file__).resolve().parent / "fixtures" / "sector_sic_ranges.json"),
    )
    get_config.cache_clear()
    cfg = get_config()
    from app.db import init_db

    init_db(cfg)
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "company_tickers.json").write_text(
        json.dumps(
            {
                "0": {"cik_str": 30, "ticker": "ASMI", "title": "Alpha Semi Inc"},
                "1": {"cik_str": 31, "ticker": "BSMC", "title": "Beta Semi Corp"},
            }
        )
    )
    yield cfg
    get_config.cache_clear()


def _registry_cache_path(cfg) -> Path:
    return Path(cfg.cache_dir) / "company_tickers_exchange.json"


def _fake_ingest_factory(cfg, calls: list[str]):
    def fake_ingest(**kwargs):
        # Record the delta scope and mark it sweepable like the real ingest.
        conn = sqlite3.connect(str(cfg.db_path))
        rows = conn.execute(
            """
            SELECT r.primary_ticker FROM sec_registrants r
            WHERE r.in_scope = 1 AND r.operating_status = 'OPERATING' AND r.removed_at IS NULL
              AND EXISTS (SELECT 1 FROM sector_inference si
                          WHERE si.ticker = r.primary_ticker AND si.inferred_sector IS NOT NULL)
              AND NOT EXISTS (SELECT 1 FROM valuations v
                              WHERE v.ticker = r.primary_ticker AND v.method = 'scorecard')
            """
        ).fetchall()
        for (ticker,) in rows:
            calls.append(ticker)
            conn.execute(
                "INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json, warnings_json, created_at) "
                "VALUES (?, ?, 'scorecard', '{}', '{}', '[]', 'x')",
                (ticker, kwargs.get("as_of_date") or "2026-06-11"),
            )
        conn.commit()
        conn.close()
        return {"ingested": len(rows), "quarantined": 0, "failed": 0, "status_counts": {}}

    return fake_ingest


def test_sync_bootstrap_then_delta_and_removal(sync_env) -> None:
    cfg = sync_env
    ingested: list[str] = []

    # Bootstrap sync: both registrants arrive, classify, ingest.
    first = sync_universe(
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        refresh_registry=False,
        census_kwargs={"http": _StubHttp(_REGISTRY_V1), "submissions_loader": _loader},
        ingest_fn=_fake_ingest_factory(cfg, ingested),
    )
    assert first["registrants_added"] == 2
    assert first["registrants_removed"] == 0
    assert first["new_operating_companies"] == 2
    assert first["classified"] == 2
    assert first["ingested"] == 2
    assert sorted(ingested) == ["ASMI", "BSMC"]

    # Weekly delta: BSMC left the registry. Force the registry cache to v2.
    _registry_cache_path(cfg).write_text(json.dumps(_REGISTRY_V2))
    second = sync_universe(
        as_of_date="2026-06-18",
        db_path=cfg.db_path,
        cfg=cfg,
        refresh_registry=False,
        census_kwargs={"http": _StubHttp(_REGISTRY_V2), "submissions_loader": _loader},
        ingest_fn=_fake_ingest_factory(cfg, ingested),
    )
    assert second["registrants_added"] == 0
    assert second["registrants_removed"] == 1
    assert second["removed"] == [{"cik": "0000000031", "ticker": "BSMC"}]
    assert second["classified"] == 0
    assert second["ingested"] == 0  # ASMI already sweepable; BSMC removed

    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    bsmc = conn.execute(
        "SELECT removed_at FROM sec_registrants WHERE primary_ticker='BSMC'"
    ).fetchone()
    log = conn.execute(
        "SELECT action, COUNT(*) n FROM universe_sync_log GROUP BY action ORDER BY action"
    ).fetchall()
    conn.close()
    assert bsmc["removed_at"] is not None
    actions = {row["action"]: row["n"] for row in log}
    assert actions["ADDED"] == 2
    assert actions["REMOVED"] == 1

    # Idempotency: re-running the same week changes nothing.
    third = sync_universe(
        as_of_date="2026-06-18",
        db_path=cfg.db_path,
        cfg=cfg,
        refresh_registry=False,
        census_kwargs={"http": _StubHttp(_REGISTRY_V2), "submissions_loader": _loader},
        ingest_fn=_fake_ingest_factory(cfg, ingested),
    )
    assert third["registrants_added"] == 0
    assert third["registrants_removed"] == 0
    assert third["ingested"] == 0
