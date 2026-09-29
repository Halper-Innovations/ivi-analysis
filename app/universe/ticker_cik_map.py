from __future__ import annotations

import json
from pathlib import Path

from app.config import get_config
from app.util.http import HttpClient, NetworkDisabledError


SEC_TICKER_JSON_URL = "https://www.sec.gov/files/company_tickers.json"


def cached_mapping_path() -> Path:
    cfg = get_config()
    path = cfg.cache_dir / "company_tickers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def refresh_ticker_cik_cache(http: HttpClient | None = None) -> dict[str, str]:
    http = http or HttpClient(get_config())
    payload = http.get_json(SEC_TICKER_JSON_URL, use_cache=True, cache_ttl_seconds=7 * 24 * 3600)
    path = cached_mapping_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return parse_mapping(payload)


def parse_mapping(payload: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for _, row in payload.items():
        ticker = str(row.get("ticker", "")).upper().strip()
        cik_int = row.get("cik_str")
        if ticker and cik_int is not None:
            mapping[ticker] = str(cik_int)
    return mapping


def _refresh_or_empty(http: HttpClient | None) -> dict[str, str]:
    # Offline (VOE_NET_PROVIDER=disabled) with nothing cached: an empty map,
    # never a download. The HTTP client refuses the request itself.
    try:
        return refresh_ticker_cik_cache(http=http)
    except NetworkDisabledError:
        return {}


def load_ticker_cik_map(http: HttpClient | None = None, refresh_if_missing: bool = True) -> dict[str, str]:
    path = cached_mapping_path()
    if not path.exists():
        if not refresh_if_missing:
            return {}
        return _refresh_or_empty(http)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return parse_mapping(payload)
    except Exception:
        if refresh_if_missing:
            return _refresh_or_empty(http)
        return {}
