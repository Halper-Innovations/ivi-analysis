"""HttpClient: client errors fail at once, server errors are retried, budgets can be reset."""

from __future__ import annotations

import pytest
import requests

from app.config import get_config
from app.util.http import DomainBudgetExceeded, HttpClient, reset_domain_request_counts
from tests.sec_support import TEST_USER_AGENT, FakeSec

URL = "https://data.sec.gov/submissions/CIK0000999999.json"


@pytest.fixture
def sec(monkeypatch):
    monkeypatch.setenv("VOE_SEC_USER_AGENT", TEST_USER_AGENT)
    monkeypatch.setenv("VOE_SEC_RPS", "1000")
    monkeypatch.setenv("VOE_SEC_BACKOFF", "0")
    # A fake network: the transport below is mocked, so the switch is on.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    monkeypatch.setenv("VOE_SEC_MAX_RETRIES", "3")
    get_config.cache_clear()
    reset_domain_request_counts()
    fake = FakeSec()
    monkeypatch.setattr(requests.Session, "get", lambda self, url, **kw: fake.get(url, **kw))
    yield fake
    reset_domain_request_counts()
    get_config.cache_clear()


def test_client_error_is_raised_after_one_attempt(sec):
    sec.add_bytes(URL, b"missing", status=404)

    with pytest.raises(requests.HTTPError) as err:
        HttpClient(get_config()).get_bytes(URL, use_cache=False)

    assert err.value.response.status_code == 404
    assert sec.calls == [URL]


def test_server_error_is_retried(sec):
    responses = iter([(503, b"busy"), (502, b"busy"), (200, b'{"ok": true}')])

    def flaky(url, **kwargs):
        status, body = next(responses)
        sec.add_bytes(URL, body, status=status)
        return FakeSec.get(sec, url, **kwargs)

    sec.get = flaky  # type: ignore[method-assign]

    assert HttpClient(get_config()).get_json(URL, use_cache=False) == {"ok": True}
    assert sec.calls == [URL, URL, URL]


def test_reset_domain_request_counts_restores_the_budget(sec, monkeypatch):
    monkeypatch.setenv("VOE_MAX_REQUESTS_DATA_SEC_DOMAIN", "1")
    get_config.cache_clear()
    sec.add_bytes(URL, b"{}")
    client = HttpClient(get_config())

    client.get_bytes(URL, use_cache=False)
    with pytest.raises(DomainBudgetExceeded):
        client.get_bytes(URL, use_cache=False)
    reset_domain_request_counts()
    assert client.get_bytes(URL, use_cache=False) == b"{}"
