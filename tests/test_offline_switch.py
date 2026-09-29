"""VOE_NET_PROVIDER=disabled means no outbound request, from any path.

The switch is enforced once in ``HttpClient.get_bytes`` (a cache hit still
answers; a cache miss raises ``NetworkDisabledError``, a
``requests.ConnectionError``), at the LLM transports, and for push alerts.
The SEC ticker list is not downloaded offline: the map is empty instead.
"""

from __future__ import annotations

import pytest
import requests
from typer.testing import CliRunner

from app.config import get_config
from app.util.http import HttpClient, NetworkDisabledError
from tests.sec_support import TEST_USER_AGENT

URL = "https://data.sec.gov/submissions/CIK0000999999.json"


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_SEC_USER_AGENT", TEST_USER_AGENT)
    get_config.cache_clear()
    attempts: list[str] = []

    def _send(self, request, **kwargs):
        attempts.append(request.url)
        raise AssertionError(f"network attempted while offline: {request.url}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _send)
    yield attempts
    get_config.cache_clear()


def test_cache_miss_raises_offline_error_without_a_request(offline):
    with pytest.raises(NetworkDisabledError) as err:
        HttpClient(get_config()).get_bytes(URL)
    assert str(err.value) == (
        "offline: VOE_NET_PROVIDER=disabled and "
        "https://data.sec.gov/submissions/CIK0000999999.json is not in the local cache"
    )
    assert isinstance(err.value, requests.ConnectionError)
    assert offline == []


def test_cache_hit_still_answers_offline(offline):
    client = HttpClient(get_config())
    client.cache_path(URL).write_bytes(b'{"cached": true}')
    assert client.get_json(URL) == {"cached": True}
    assert offline == []


def test_ticker_map_is_not_downloaded_offline(offline):
    from app.universe import ticker_cik_map

    assert not ticker_cik_map.cached_mapping_path().exists()
    assert ticker_cik_map.load_ticker_cik_map() == {}
    assert offline == []


def test_cik_resolution_offline_says_offline(offline):
    from app.ingest.cik_registry import CIKNotFoundError, resolve

    with pytest.raises(CIKNotFoundError) as err:
        resolve("KO")
    assert str(err.value) == (
        "No CIK found for ticker 'KO': offline (VOE_NET_PROVIDER=disabled) "
        "and the SEC ticker list is not cached."
    )




def test_openai_transport_refuses_offline(offline, monkeypatch):
    from app.llm.providers.openai_provider import OpenAIProvider

    OpenAIProvider._reset_circuit_breaker()
    monkeypatch.setenv("VOE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VOE_OPENAI_API_KEY", "sk-test-key")
    get_config.cache_clear()
    posts: list[str] = []
    monkeypatch.setattr(
        "app.llm.providers.openai_provider.requests.post",
        lambda url, **kw: posts.append(url),
    )
    with pytest.raises(NetworkDisabledError):
        OpenAIProvider(get_config()).synthesize_json(
            prompt="x", schema={"type": "object", "properties": {}}, schema_name="probe"
        )
    assert posts == []


def test_push_alerts_are_local_only_offline(offline, monkeypatch, tmp_path):
    from app.ops import alerts

    monkeypatch.setenv("VOE_ALERT_NTFY_TOPIC", "probe-topic")
    opened: list[object] = []
    monkeypatch.setattr(alerts.urllib.request, "urlopen", lambda *a, **k: opened.append(a))
    alerts.post_alert("title", "body")
    assert opened == []
    log = get_config().outputs_dir / "cron" / "alerts.log"
    assert log.read_text(encoding="utf-8").endswith("\ttitle\tbody\n")


@pytest.mark.financial_integrity_contract
def test_analyze_offline_makes_zero_network_attempts(offline, monkeypatch, tmp_path):
    """The suite no longer stubs the ticker-list download for every test, so
    this runs the real refresh path. Any socket use would also be recorded by
    the hermeticity guard and fail the test."""

    from app.cli import app
    from app.db import init_db

    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    init_db(get_config())

    result = CliRunner().invoke(app, ["analyze", "KO"])

    assert offline == []
    assert "HTTPSConnectionPool" not in result.output
    assert "Status: NO_SCORECARD" in result.output
