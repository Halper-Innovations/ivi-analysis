"""Placeholder SEC identities are refused; credentials are redacted in free text.

The documented placeholder (``you@yourdomain.com``, and the validator's own
``jane.doe@yourdomain.com`` example) must not pass validation: a copy-paste of
the docs would otherwise send the SEC a fake contact. A user agent containing
a line break is refused without echoing it.
"""

from __future__ import annotations

import pytest

from app.util.credential_hygiene import (
    InvalidSecUserAgentError,
    redact_credential_text,
    validate_sec_user_agent,
)


@pytest.mark.parametrize(
    "ua",
    [
        "Your Name you@yourdomain.com",
        "Jane Doe jane.doe@yourdomain.com",
        "Jane Doe jane@mail.yourdomain.com",
    ],
)
def test_placeholder_domain_is_rejected(ua):
    with pytest.raises(InvalidSecUserAgentError, match="placeholder contact domain"):
        validate_sec_user_agent(ua)


def test_validator_example_itself_fails_validation():
    with pytest.raises(InvalidSecUserAgentError) as err:
        validate_sec_user_agent("")
    example = str(err.value).split("VOE_SEC_USER_AGENT=")[1].split('"')[1]
    assert example == "Jane Doe jane.doe@yourdomain.com"
    with pytest.raises(InvalidSecUserAgentError):
        validate_sec_user_agent(example)


def test_user_agent_with_newline_is_refused_without_echo():
    with pytest.raises(InvalidSecUserAgentError) as err:
        validate_sec_user_agent("Probe Person\nprobe@realmail-probe.org")
    assert "probe@realmail-probe.org" not in str(err.value)
    assert str(err.value).startswith("SEC_USER_AGENT_INVALID: VOE_SEC_USER_AGENT contains a line break")


def test_real_identity_still_passes():
    assert validate_sec_user_agent("Jane Doe jane@realmail-probe.org") == "Jane Doe jane@realmail-probe.org"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GET failed: api_token=SEKRET1 for url", "GET failed: api_token=REDACTED for url"),
        ("{'apikey': 'SEKRET3'}", "{'apikey': 'REDACTED'}"),
        ('{"api_key": "SEKRET12"}', '{"api_key": "REDACTED"}'),
        ("headers={'Authorization': 'Token SEKRET4'}", "headers={'Authorization': 'REDACTED'}"),
        ("Authorization: Bearer SEKRET11", "Authorization: REDACTED"),
        ("https://x.org/?api%5Ftoken=SEKRET9", "https://x.org/?api%5Ftoken=REDACTED"),
        ("https://x.org/;api_token=SEKRET10", "https://x.org/;api_token=REDACTED"),
        ("https://x.org/?password=SEKRET7", "https://x.org/?password=REDACTED"),
        ("User-Agent: Probe probe@realmail.org", "User-Agent: Probe probe@realmail.org"),
    ],
)
def test_free_text_credentials_are_redacted(text, expected):
    assert redact_credential_text(text) == expected
