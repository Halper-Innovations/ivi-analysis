"""The companyfacts download never follows a redirect to another host."""

from __future__ import annotations

import requests

from app.config import get_config


class _Redirect:
    status_code = 302
    headers = {"Location": "https://evil.example.org/facts.json"}

    def json(self):  # pragma: no cover - must never be read
        raise AssertionError("a redirect body must not be parsed as companyfacts")


def test_companyfacts_fetch_refuses_to_follow_redirects(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    monkeypatch.setenv("VOE_SEC_USER_AGENT", "IVI tests qa@ivi-analysis.org")
    get_config.cache_clear()
    calls: list[dict] = []

    def fake_get(self, url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _Redirect()

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    from app.market.company_facts_provider import fetch_company_facts

    result = fetch_company_facts("0000021344", cfg=get_config())

    assert calls, "the fetch must be attempted"
    assert all(call.get("allow_redirects") is False for call in calls)
    assert all(call["url"].startswith("https://data.sec.gov/") for call in calls)
    assert result["status"] != "OK"
    assert "companyfacts" not in result or not result.get("companyfacts")
