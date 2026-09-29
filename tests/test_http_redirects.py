"""HttpClient redirects and size cap, through the real ``requests.Session``.

Only the transport adapter is faked (``HTTPAdapter.send``), so requests' own
redirect machinery would run if the client let it. Redirects are followed by
hand with the allowlist re-checked on each hop; a response that lands on
another host is never cached under the original URL's key.
"""

from __future__ import annotations

import pytest
import requests
from requests.adapters import HTTPAdapter
from requests.models import Response

from app.config import get_config
from app.util.http import (
    AllowlistError,
    HttpClient,
    ResponseTooLarge,
    TooManyRedirects,
    reset_domain_request_counts,
)
from tests.sec_support import TEST_USER_AGENT

SRC = "https://www.sec.gov/Archives/edgar/data/1/000000000000000001/r.htm"


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    monkeypatch.setenv("VOE_SEC_USER_AGENT", TEST_USER_AGENT)
    monkeypatch.setenv("VOE_SEC_RPS", "1000")
    monkeypatch.setenv("VOE_SEC_BACKOFF", "0")
    monkeypatch.setenv("VOE_SEC_MAX_RETRIES", "0")
    get_config.cache_clear()
    reset_domain_request_counts()
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
    sent: list[tuple[str, str | None]] = []

    def _send(self, request, **kwargs):
        sent.append((request.url, request.headers.get("User-Agent")))
        status, headers, body = routes.get(request.url, (404, {}, b"missing"))
        response = Response()
        response.status_code = status
        response.headers.update(headers)
        response._content = body
        response.url = request.url
        response.request = request
        response.connection = self
        return response

    monkeypatch.setattr(HTTPAdapter, "send", _send)
    yield routes, sent
    reset_domain_request_counts()
    get_config.cache_clear()


def test_redirect_to_unlisted_host_is_refused_before_it_is_sent(transport):
    routes, sent = transport
    routes[SRC] = (302, {"Location": "https://evil.example.org/steal"}, b"")
    with pytest.raises(AllowlistError, match="evil.example.org"):
        HttpClient(get_config()).get_bytes(SRC, use_cache=False)
    assert [url for url, _ in sent] == [SRC]


def test_same_host_redirect_is_followed_and_cached(transport):
    routes, sent = transport
    moved = "https://www.sec.gov/Archives/edgar/data/1/000000000000000001/moved.htm"
    routes[SRC] = (301, {"Location": "/Archives/edgar/data/1/000000000000000001/moved.htm"}, b"")
    routes[moved] = (200, {}, b"<p>moved</p>")
    client = HttpClient(get_config())
    assert client.get_bytes(SRC) == b"<p>moved</p>"
    assert [url for url, _ in sent] == [SRC, moved]
    assert client.cache_path(SRC).read_bytes() == b"<p>moved</p>"


def test_cross_host_redirect_is_returned_but_not_cached(transport):
    routes, sent = transport
    other = "https://data.sec.gov/other.json"
    routes[SRC] = (302, {"Location": other}, b"")
    routes[other] = (200, {}, b"{}")
    client = HttpClient(get_config())
    assert client.get_bytes(SRC) == b"{}"
    assert not client.cache_path(SRC).exists()


def test_redirect_loop_stops_after_five_hops(transport):
    routes, sent = transport
    for i in range(10):
        routes[f"https://www.sec.gov/r{i}"] = (302, {"Location": f"/r{i + 1}"}, b"")
    with pytest.raises(TooManyRedirects, match="more than 5 redirects"):
        HttpClient(get_config()).get_bytes("https://www.sec.gov/r0", use_cache=False)
    assert len(sent) == 6


def test_oversized_response_is_refused_and_not_cached(transport):
    routes, _ = transport
    routes[SRC] = (200, {}, b"x" * 2048)
    client = HttpClient(get_config())
    with pytest.raises(ResponseTooLarge):
        client.get_bytes(SRC, max_bytes=1024)
    assert not client.cache_path(SRC).exists()
    assert isinstance(ResponseTooLarge("x"), requests.RequestException)
