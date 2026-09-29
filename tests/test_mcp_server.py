"""The MCP server as a client sees it: startup checks, tool registration, calls over a session."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp.server.mcpserver")

from mcp import Client  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

from app.mcp_server import runtime, server  # noqa: E402
from tests.sec_support import ACME_DOC_URL, TEST_USER_AGENT, install_fake_sec  # noqa: E402

TICKER_OR_CIK = {
    "description": "Ticker (e.g. 'KO', 'BRK.B') or SEC CIK (e.g. '21344' or 'CIK0000021344').",
    "title": "Ticker Or Cik",
    "type": "string",
}


@pytest.fixture
def fake_sec(monkeypatch):
    yield from install_fake_sec(monkeypatch)


def _session_call(coro_fn):
    """Run ``coro_fn(client)`` inside an initialized in-memory MCP session (JSON-RPC framed)."""

    result: dict = {}

    async def main() -> None:
        async with Client(server.build_server(), mode="legacy") as client:
            result["value"] = await coro_fn(client)

    anyio.run(main)
    return result["value"]


def _call(name: str, arguments: dict):
    return _session_call(lambda client: client.call_tool(name, arguments))


def _json(result) -> dict:
    assert result.is_error is False, result.content[0].text
    assert len(result.content) == 1
    return json.loads(result.content[0].text)


def _error_text(result) -> str:
    assert result.is_error is True
    assert len(result.content) == 1
    return result.content[0].text


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


@pytest.fixture
def no_serving(monkeypatch):
    runs: list[str] = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport="stdio", **kw: runs.append(transport))
    return runs


def test_main_refuses_to_start_without_user_agent(monkeypatch, capsys, no_serving):
    monkeypatch.delenv("VOE_SEC_USER_AGENT", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        server.main()

    assert exit_info.value.code == 2
    assert no_serving == []
    assert capsys.readouterr().err == (
        "ivi-mcp: refusing to start. SEC_USER_AGENT_INVALID: set VOE_SEC_USER_AGENT to an "
        "application identity containing a monitored contact email, e.g. "
        'VOE_SEC_USER_AGENT="Jane Doe jane.doe@yourdomain.com". The SEC requires every '
        "automated client to identify itself: https://www.sec.gov/os/accessing-edgar-data\n"
    )


def test_main_refuses_placeholder_user_agent(monkeypatch, capsys, no_serving):
    monkeypatch.setenv("VOE_SEC_USER_AGENT", "Jane Doe jane.doe@example.com")

    with pytest.raises(SystemExit) as exit_info:
        server.main()

    assert exit_info.value.code == 2
    assert no_serving == []
    assert capsys.readouterr().err == (
        "ivi-mcp: refusing to start. SEC_USER_AGENT_INVALID: VOE_SEC_USER_AGENT contains a "
        "placeholder contact domain.\n"
    )


def test_main_serves_stdio_with_a_valid_user_agent(monkeypatch, capsys, no_serving):
    monkeypatch.setenv("VOE_SEC_USER_AGENT", TEST_USER_AGENT)

    server.main()

    assert no_serving == ["stdio"]
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout belongs to the JSON-RPC stream


def test_main_reports_an_unsafe_env_file_instead_of_crashing(monkeypatch, capsys, no_serving):
    from app.util.credential_hygiene import InsecureEnvPermissionsError

    def unsafe_env():
        raise InsecureEnvPermissionsError(
            "ENV_PERMISSIONS_INSECURE: /work/.env mode is 0644; run chmod 600 /work/.env."
        )

    monkeypatch.setattr(server, "prepare_runtime", unsafe_env)

    with pytest.raises(SystemExit) as exit_info:
        server.main()

    assert exit_info.value.code == 2
    assert no_serving == []
    assert capsys.readouterr().err == (
        "ivi-mcp: refusing to start. ENV_PERMISSIONS_INSECURE: /work/.env mode is 0644; run "
        "chmod 600 /work/.env.\n"
    )


@pytest.mark.subprocess
def test_importing_the_entry_point_does_not_load_config():
    # The console script imports this module before main() runs; if that import
    # reached app.config, a bad .env would crash with a traceback instead of
    # main()'s one-line refusal.
    import subprocess
    import sys

    probe = (
        "import sys, app.mcp_server.server; "
        "print(sorted(m for m in ('app.config', 'app.util.http') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert out == "[]\n"


def test_prepare_runtime_defaults_data_dir_to_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("VOE_DATA_DIR", raising=False)
    monkeypatch.delenv("VOE_DB_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setenv("VOE_SEC_USER_AGENT", TEST_USER_AGENT)

    cache_dir = runtime.prepare_runtime()

    expected = tmp_path / ".cache" / "ivi-analysis"
    assert cache_dir == expected / "cache"
    assert cache_dir.is_dir()
    assert Path(runtime.os.environ["VOE_DATA_DIR"]) == expected
    assert Path(runtime.os.environ["VOE_DB_PATH"]) == expected / "engine.db"


def test_apply_data_dir_default_keeps_an_explicit_setting(tmp_path):
    env = {"VOE_DATA_DIR": str(tmp_path / "mine")}

    assert runtime.apply_data_dir_default(env) == tmp_path / "mine"
    assert env == {
        "VOE_DATA_DIR": str(tmp_path / "mine"),
        "VOE_DB_PATH": str(tmp_path / "mine" / "engine.db"),
    }


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("darwin", {}, "/home/u/Library/Caches/ivi-analysis"),
        ("linux", {}, "/home/u/.cache/ivi-analysis"),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, "/xdg/ivi-analysis"),
        ("linux", {"XDG_CACHE_HOME": "relative"}, "/home/u/.cache/ivi-analysis"),
        ("win32", {"LOCALAPPDATA": "/appdata"}, "/appdata/ivi-analysis"),
    ],
)
def test_default_data_dir_per_platform(platform, env, expected):
    assert runtime.default_data_dir(env, platform=platform, home=Path("/home/u")) == Path(expected)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_list_tools_returns_exactly_the_six_tools_with_their_schemas():
    tools = _session_call(lambda client: client.list_tools()).tools

    assert [tool.name for tool in tools] == list(server.TOOL_NAMES)
    schemas = {tool.name: tool.input_schema for tool in tools}
    assert {name: (list(s["properties"]), s["required"]) for name, s in schemas.items()} == {
        "lookup_company": (["query", "limit"], ["query"]),
        "get_company_profile": (["ticker_or_cik"], ["ticker_or_cik"]),
        "list_filings": (["ticker_or_cik", "forms", "since", "until", "limit"], ["ticker_or_cik"]),
        "get_filing_text": (
            ["url_or_accession", "ticker_or_cik", "max_chars", "offset"],
            ["url_or_accession"],
        ),
        "get_financials": (
            ["ticker_or_cik", "period", "years", "as_of", "line_items"],
            ["ticker_or_cik"],
        ),
        "get_concept": (
            ["ticker_or_cik", "concept", "taxonomy", "unit", "as_of", "period", "limit"],
            ["ticker_or_cik", "concept"],
        ),
    }
    assert schemas["get_company_profile"]["properties"]["ticker_or_cik"] == TICKER_OR_CIK
    assert schemas["get_financials"]["properties"]["period"] == {
        "default": "annual",
        "description": "Fiscal years or fiscal quarters.",
        "enum": ["annual", "quarterly"],
        "title": "Period",
        "type": "string",
    }
    assert schemas["get_concept"]["properties"]["period"]["enum"] == [
        "all",
        "annual",
        "quarterly",
        "instant",
    ]
    assert schemas["list_filings"]["properties"]["forms"]["anyOf"] == [
        {"items": {"type": "string"}, "type": "array"},
        {"type": "string"},
        {"type": "null"},
    ]
    for tool in tools:
        assert tool.description == server.tool_descriptions()[tool.name]
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.idempotent_hint is True
        assert tool.annotations.open_world_hint is True


def test_financials_description_lists_every_line_item():
    from app.mcp_server.financials import ALL_LINE_ITEMS

    description = server.tool_descriptions()["get_financials"]
    assert all(item in description for item in ALL_LINE_ITEMS)


# ---------------------------------------------------------------------------
# calls over the session
# ---------------------------------------------------------------------------


def test_lookup_company_over_session_returns_compact_json(fake_sec):
    result = _call("lookup_company", {"query": "ACME", "limit": 1})

    assert result.content[0].text.startswith('{"query":"ACME","match_count":2,"matches":[{')
    assert _json(result) == {
        "query": "ACME",
        "match_count": 2,
        "matches": [
            {
                "ticker": "ACME",
                "cik": "0000999999",
                "name": "ACME WIDGETS INC",
                "exchange": "Nasdaq",
                "matched_on": "ticker",
            }
        ],
        "truncated": True,
    }


def test_get_company_profile_over_session(fake_sec):
    payload = _json(_call("get_company_profile", {"ticker_or_cik": "CIK0000999999"}))
    assert (payload["name"], payload["sic"], payload["fiscal_year_end"]) == (
        "ACME WIDGETS INC",
        "3559",
        "12-31",
    )


def test_list_filings_over_session_accepts_a_single_form_string(fake_sec):
    payload = _json(_call("list_filings", {"ticker_or_cik": "ACME", "forms": "10-Q", "limit": 5}))
    assert [f["accession"] for f in payload["filings"]] == [
        "0000999999-25-000020",
        "0000999999-24-000020",
        "0000999999-23-000020",
    ]


def test_get_filing_text_over_session(fake_sec):
    payload = _json(_call("get_filing_text", {"url_or_accession": ACME_DOC_URL, "max_chars": 500}))
    assert (payload["total_chars"], payload["next_offset"], len(payload["text"])) == (1308, 500, 500)


def test_get_financials_over_session_point_in_time(fake_sec):
    payload = _json(
        _call(
            "get_financials",
            {"ticker_or_cik": "ACME", "as_of": "2025-02-13", "line_items": ["revenue"]},
        )
    )
    assert [(p["fiscal_year"], p["values"]["revenue"]) for p in payload["periods"]] == [
        (2023, {"value": 100.0, "tag": "us-gaap:Revenues", "accn": "0000999999-24-000010"}),
        (2022, {"value": 80.0, "tag": "us-gaap:Revenues", "accn": "0000999999-24-000010"}),
    ]


def test_get_concept_over_session(fake_sec):
    payload = _json(
        _call("get_concept", {"ticker_or_cik": "ACME", "concept": "InventoryNet", "limit": 1})
    )
    assert payload["points"] == [
        {
            "end": "2024-12-31",
            "value": 8500000,
            "fy": 2024,
            "fp": "FY",
            "form": "10-K",
            "filed": "2025-02-14",
            "accn": "0000999999-25-000010",
        }
    ]


def test_unknown_ticker_is_a_clean_tool_error(fake_sec):
    text = _error_text(_call("get_company_profile", {"ticker_or_cik": "ZZZZ"}))
    assert text == (
        "Error executing tool get_company_profile: Unknown ticker 'ZZZZ'. Use lookup_company to "
        "search by name, or pass the CIK."
    )


def test_missing_document_is_a_clean_tool_error(fake_sec):
    url = "https://www.sec.gov/Archives/edgar/data/999999/000099999925000010/missing.htm"
    text = _error_text(_call("get_filing_text", {"url_or_accession": url}))
    assert text == (
        f"Error executing tool get_filing_text: SEC EDGAR has nothing at {url} (HTTP 404)."
    )
    assert fake_sec.calls == [url]  # a 404 is an answer, not retried


def test_non_sec_url_is_rejected_before_any_request(fake_sec):
    text = _error_text(_call("get_filing_text", {"url_or_accession": "https://example.com/a.htm"}))
    assert text == (
        "Error executing tool get_filing_text: Only sec.gov documents can be fetched; "
        "'example.com' is not allowed."
    )
    assert fake_sec.calls == []


def test_unexpected_failures_never_leak_a_traceback(fake_sec, monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.mcp_server.companies.get_company_profile", boom)
    text = _error_text(_call("get_company_profile", {"ticker_or_cik": "ACME"}))
    assert text == (
        "Error executing tool get_company_profile: Unexpected internal error (RuntimeError): boom"
    )
    assert "Traceback" not in text


def test_invalid_arguments_are_reported_not_raised(fake_sec):
    text = _error_text(_call("get_financials", {"ticker_or_cik": "ACME", "period": "monthly"}))
    assert text.startswith("Error executing tool get_financials: ")
    assert "period" in text


def test_domain_budget_caps_one_call_not_the_server_lifetime(fake_sec, monkeypatch):
    from app.config import get_config

    monkeypatch.setenv("VOE_MAX_REQUESTS_DATA_SEC_DOMAIN", "1")
    get_config.cache_clear()

    async def two_calls(client):
        first = await client.call_tool("get_company_profile", {"ticker_or_cik": "ACME"})
        second = await client.call_tool("get_concept", {"ticker_or_cik": "ACME", "concept": "Revenues"})
        return first, second

    first, second = _session_call(two_calls)
    assert _json(first)["name"] == "ACME WIDGETS INC"
    assert _json(second)["concept"] == "us-gaap:Revenues"
