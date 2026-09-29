"""MCP-server hardening checks that live beside the shared HTTP and credential tests."""

from __future__ import annotations

import pytest

from app.util.http import NetworkDisabledError, ResponseTooLarge
from tests.test_offline_switch import offline  # noqa: F401  (pytest fixture)
from tests.test_http_redirects import SRC, transport  # noqa: F401  (transport is a pytest fixture)


def test_mcp_tool_reports_offline(offline):  # noqa: F811 (imported pytest fixture)
    from app.mcp_server import filing_text
    from app.mcp_server.errors import describe_exception

    filing_text.clear_text_cache()
    with pytest.raises(NetworkDisabledError) as err:
        filing_text.get_filing_text("https://www.sec.gov/Archives/edgar/data/1/x.htm")
    assert describe_exception(err.value) == (
        "Offline: VOE_NET_PROVIDER=disabled and this SEC data is not in the local cache. "
        "Unset VOE_NET_PROVIDER (or set it to enabled) to fetch it."
    )
    assert offline == []


def test_declared_length_over_cap_is_refused(transport):  # noqa: F811 (imported pytest fixture)
    routes, _ = transport
    routes[SRC] = (200, {"Content-Length": str(30 * 1024 * 1024)}, b"")
    from app.mcp_server.errors import describe_exception
    from app.mcp_server.filing_text import MAX_FILING_BYTES
    from app.ingest.sec_client import SecClient

    with pytest.raises(ResponseTooLarge) as err:
        SecClient().download_bytes(SRC, max_bytes=MAX_FILING_BYTES)
    assert describe_exception(err.value) == (
        f"response from {SRC} is larger than 25 MB; refusing to download it."
    )


def test_mcp_generic_error_is_one_line():
    from app.mcp_server.errors import describe_exception

    message = describe_exception(ValueError("header value 'Probe\r\nX-Evil: 1' api_token=S"))
    assert message == (
        "Unexpected internal error (ValueError): header value 'Probe  X-Evil: 1' api_token=REDACTED"
    )
