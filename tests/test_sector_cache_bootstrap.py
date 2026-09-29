from __future__ import annotations

import json


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    taxonomy = data_dir / "universe" / "sector_taxonomy.csv"
    overrides = data_dir / "universe" / "sector_overrides.csv"
    taxonomy.parent.mkdir(parents=True, exist_ok=True)
    taxonomy.write_text("ticker,sector\nHON,Industrials\nITW,Industrials\n", encoding="utf-8")
    overrides.write_text("ticker,sector\nHON,industrial_tech\nITW,industrial_tech\n", encoding="utf-8")
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_SECTOR_TAXONOMY_PATH", str(taxonomy))
    monkeypatch.setenv("VOE_SECTOR_OVERRIDES_PATH", str(overrides))
    monkeypatch.setenv("VOE_SAFE_MODE", "false")
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def test_bootstrap_sector_cache_writes_companyfacts_and_submissions(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    from app.universe.bootstrap_sector_cache import bootstrap_sector_cache
    from app.util.http import HttpClient

    mapping = {"HON": "773840", "ITW": "49826"}
    calls: list[str] = []

    def _fake_refresh_ticker_cik_cache(http=None):
        return dict(mapping)

    def _fake_get_json(self, url, *, params=None, use_cache=True, cache_ttl_seconds=None):
        calls.append(url)
        if "company_tickers.json" in url:
            return {}
        if "companyfacts" in url:
            cik = url.split("CIK", 1)[1].split(".json", 1)[0]
            return {"cik": cik, "facts": {"us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": {}}}}
        if "submissions" in url:
            cik = url.split("CIK", 1)[1].split(".json", 1)[0]
            return {"cik": cik, "filings": {"recent": {"accessionNumber": [], "form": [], "filingDate": [], "reportDate": [], "primaryDocument": []}}}
        raise AssertionError(url)

    monkeypatch.setattr("app.universe.bootstrap_sector_cache.refresh_ticker_cik_cache", _fake_refresh_ticker_cik_cache)
    monkeypatch.setattr(HttpClient, "get_json", _fake_get_json)
    monkeypatch.setattr("app.universe.bootstrap_sector_cache.time.sleep", lambda _: None)

    payload = bootstrap_sector_cache(sector="industrial_tech", cfg=cfg)

    assert payload["status"] == "OK"
    assert payload["companyfacts_cached"] == 2
    assert payload["submissions_cached"] == 2
    assert payload["failed"] == 0

    hon_cik = "0000773840"
    companyfacts_path = cfg.cache_dir / "companyfacts" / f"{hon_cik}.json"
    submissions_path = cfg.cache_dir / "submissions" / f"{hon_cik}.json"
    assert companyfacts_path.exists()
    assert submissions_path.exists()
    companyfacts_payload = json.loads(companyfacts_path.read_text(encoding="utf-8"))
    submissions_payload = json.loads(submissions_path.read_text(encoding="utf-8"))
    assert companyfacts_payload["cik"] == hon_cik
    assert isinstance(companyfacts_payload.get("companyfacts"), dict)
    assert submissions_payload["cik"] == hon_cik

    http_cache_path = HttpClient(cfg).cache_path(f"https://data.sec.gov/submissions/CIK{hon_cik}.json")
    assert http_cache_path.exists()
    assert any("companyfacts" in url for url in calls)
    assert any("submissions" in url for url in calls)


def test_bootstrap_sector_cache_skips_fresh_cache(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    from app.universe.bootstrap_sector_cache import bootstrap_sector_cache, submissions_cache_path

    hon_cik = "0000773840"
    companyfacts_path = cfg.cache_dir / "companyfacts" / f"{hon_cik}.json"
    companyfacts_path.parent.mkdir(parents=True, exist_ok=True)
    companyfacts_path.write_text(json.dumps({"companyfacts": {"facts": {}}, "cik": hon_cik}), encoding="utf-8")
    sub_path = submissions_cache_path(hon_cik, cfg=cfg)
    sub_path.write_text(json.dumps({"cik": hon_cik, "filings": {"recent": {}}}), encoding="utf-8")

    monkeypatch.setattr("app.universe.bootstrap_sector_cache.refresh_ticker_cik_cache", lambda http=None: {"HON": "773840", "ITW": "49826"})
    monkeypatch.setattr("app.universe.bootstrap_sector_cache.time.sleep", lambda _: None)

    calls: list[str] = []

    def _fake_get_json(self, url, *, params=None, use_cache=True, cache_ttl_seconds=None):
        calls.append(url)
        return {"noop": True}

    monkeypatch.setattr("app.universe.bootstrap_sector_cache.HttpClient.get_json", _fake_get_json)

    payload = bootstrap_sector_cache(sector="industrial_tech", tickers=["HON"], cfg=cfg)

    assert payload["skipped"] == 2
    assert payload["failed"] == 0
    assert calls == []
