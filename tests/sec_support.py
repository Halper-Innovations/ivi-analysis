"""Shared fixtures for the MCP server tests: a fake SEC EDGAR behind the real HttpClient.

Only ``requests.Session.get`` is replaced, so the repository's HTTP layer (the
sec.gov allowlist, User-Agent check, rate limiter, disk cache and retry policy)
still runs. Unknown URLs answer 404, like EDGAR.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import requests

FIXTURES = Path(__file__).parent / "fixtures" / "sec"
TEST_USER_AGENT = "IVI Analysis test suite tests@ivi-analysis.org"

ACME_CIK = "0000999999"
ACME_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/999999/000099999925000010/acme-20241231.htm"
)
EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def submissions_url(cik10: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik10}.json"


def companyfacts_url(cik10: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


class FakeSec:
    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, bytes]] = {}
        self.calls: list[str] = []

    def add_file(self, url: str, name: str) -> None:
        self.routes[url] = (200, (FIXTURES / name).read_bytes())

    def add_json(self, url: str, payload: Any) -> None:
        self.routes[url] = (200, json.dumps(payload).encode("utf-8"))

    def add_bytes(self, url: str, body: bytes, status: int = 200) -> None:
        self.routes[url] = (status, body)

    def get(self, url: str, params: Any = None, timeout: Any = None, **_: Any) -> requests.Response:
        del params, timeout
        self.calls.append(url)
        status, body = self.routes.get(url, (404, b"<html>Not Found</html>"))
        response = requests.Response()
        response.status_code = status
        response._content = body
        response.url = url
        response.reason = "OK" if status < 400 else "Error"
        return response


def reset_mcp_caches() -> None:
    """Clear the MCP server's in-process caches (a no-op when it isn't installed)."""
    try:
        from app.mcp_server import companies, filing_text, financials
    except ImportError:
        return

    with companies._lock:
        companies._ticker_rows_cache.clear()
        companies._exchange_cache.clear()
    financials.clear_memory_cache()
    filing_text.clear_text_cache()


def install_fake_sec(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSec]:
    """Generator behind each test module's ``fake_sec`` fixture.

    A fake EDGAR with the synthetic ACME filer and a trimmed real KO payload.
    """

    from app.config import get_config
    from app.universe import ticker_cik_map

    monkeypatch.setenv("VOE_SEC_USER_AGENT", TEST_USER_AGENT)
    monkeypatch.setenv("VOE_SEC_RPS", "1000")
    # A fake network: the transport below is mocked, so the switch is on.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    monkeypatch.setenv("VOE_SEC_BACKOFF", "0")
    get_config.cache_clear()
    reset_mcp_caches()

    sec = FakeSec()
    sec.add_file(EXCHANGE_URL, "company_tickers_exchange.json")
    sec.add_file(submissions_url(ACME_CIK), "submissions_CIK0000999999.json")
    sec.add_file(
        "https://data.sec.gov/submissions/CIK0000999999-submissions-001.json",
        "submissions_CIK0000999999-submissions-001.json",
    )
    sec.add_file(companyfacts_url(ACME_CIK), "companyfacts_CIK0000999999.json")
    sec.add_file(companyfacts_url("0000021344"), "companyfacts_CIK0000021344_trimmed.json")
    sec.add_file(ACME_DOC_URL, "acme-20241231.htm")
    monkeypatch.setattr(
        requests.Session, "get", lambda self, url, **kwargs: sec.get(url, **kwargs)
    )

    # The SEC ticker file as the repository's ticker map caches it.
    mapping_path = ticker_cik_map.cached_mapping_path()
    mapping_path.write_bytes((FIXTURES / "company_tickers.json").read_bytes())

    yield sec
    reset_mcp_caches()
    get_config.cache_clear()
