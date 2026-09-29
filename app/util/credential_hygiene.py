from __future__ import annotations

import json
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REDACTED_CREDENTIAL = "REDACTED"

_DEFAULT_SENSITIVE_QUERY_NAMES = frozenset(
    {
        "access_token",
        "api-key",
        "api_key",
        "api_token",
        "apikey",
        "auth_token",
        "authorization",
        "client_secret",
        "password",
        "secret",
        "token",
        "x-api-key",
    }
)
_REDACTED_VALUES = frozenset(
    {
        "",
        "<redacted>",
        "[redacted]",
        "redacted",
    }
)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE)
_RESERVED_CONTACT_DOMAINS = frozenset(
    {
        "example.com",
        "example.net",
        "example.org",
        # The placeholder in the README, .env.example and the error below.
        "yourdomain.com",
    }
)


class InvalidSecUserAgentError(RuntimeError):
    """Raised before an SEC request can use a missing or placeholder identity."""


class InsecureEnvPermissionsError(RuntimeError):
    """Raised before a credential-bearing .env file can be loaded too broadly."""


def _sensitive_names(additional_sensitive_names: Sequence[str] = ()) -> frozenset[str]:
    return _DEFAULT_SENSITIVE_QUERY_NAMES | {
        str(name).strip().lower()
        for name in additional_sensitive_names
        if str(name).strip()
    }


def _name_alternation(additional_sensitive_names: Sequence[str] = ()) -> str:
    names = sorted(_sensitive_names(additional_sensitive_names), key=len, reverse=True)
    # Also match a percent-encoded underscore (api%5Ftoken).
    return "|".join(re.escape(name).replace("_", "(?:_|%5[Ff])") for name in names)


def _credential_pattern(additional_sensitive_names: Sequence[str] = ()) -> re.Pattern[str]:
    alternation = _name_alternation(additional_sensitive_names)
    # A query field (?name= / &name= / ;name=) or a bare name=value in free text.
    return re.compile(
        rf"(?P<prefix>^|[?&;\s,(\[{{])(?P<name>{alternation})=(?P<value>[^&#\s\"'<>]*)",
        re.IGNORECASE,
    )


def _quoted_credential_pattern(additional_sensitive_names: Sequence[str] = ()) -> re.Pattern[str]:
    # A dict/JSON repr: 'apikey': 'VALUE' or "Authorization": "Token VALUE".
    alternation = _name_alternation(additional_sensitive_names)
    return re.compile(
        rf"(?P<key>(?P<q>['\"])(?:{alternation})(?P=q)\s*:\s*)(?P<vq>['\"])(?P<value>.*?)(?P=vq)",
        re.IGNORECASE,
    )


_HEADER_CREDENTIAL_RE = re.compile(
    r"(?P<name>\b(?:proxy-)?authorization|\bx-api-key)(?P<sep>\s*:\s*)(?P<value>[^\r\n,'\"}]+)",
    re.IGNORECASE,
)


def _text_may_contain_credentials(
    value: str,
    *,
    additional_sensitive_names: Sequence[str] = (),
) -> bool:
    text = str(value)
    if _credential_pattern(additional_sensitive_names).search(text):
        return True
    userinfo = re.search(
        r"https?://(?P<value>[^/@\s]+)@",
        text,
        flags=re.IGNORECASE,
    )
    if (
        userinfo is not None
        and userinfo.group("value").strip().lower() not in _REDACTED_VALUES
    ):
        return True
    names = sorted(_sensitive_names(additional_sensitive_names), key=len, reverse=True)
    alternation = "|".join(re.escape(name) for name in names)
    return bool(
        re.search(
            rf"[\"'](?:{alternation})[\"']\s*:",
            text,
            flags=re.IGNORECASE,
        )
    )


def redact_credential_text(
    value: str | None,
    *,
    additional_sensitive_names: Sequence[str] = (),
) -> str | None:
    """Redact query credentials and URL userinfo from arbitrary diagnostic text."""

    if value is None:
        return None
    text = str(value)
    pattern = _credential_pattern(additional_sensitive_names)
    text = pattern.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('name')}={REDACTED_CREDENTIAL}"
        ),
        text,
    )
    text = _quoted_credential_pattern(additional_sensitive_names).sub(
        lambda match: f"{match.group('key')}{match.group('vq')}{REDACTED_CREDENTIAL}{match.group('vq')}",
        text,
    )
    text = _HEADER_CREDENTIAL_RE.sub(
        lambda match: f"{match.group('name')}{match.group('sep')}{REDACTED_CREDENTIAL}",
        text,
    )
    return re.sub(
        r"(?P<scheme>https?://)[^/@\s]+@",
        rf"\g<scheme>{REDACTED_CREDENTIAL}@",
        text,
        flags=re.IGNORECASE,
    )


def sanitize_url_credentials(
    value: str | None,
    *,
    additional_sensitive_names: Sequence[str] = (),
) -> str | None:
    """Return a canonical URL with credential query fields and userinfo removed."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return text
    sensitive_names = _sensitive_names(additional_sensitive_names)
    try:
        parsed = urlsplit(text)
    except ValueError:
        return redact_credential_text(
            text,
            additional_sensitive_names=additional_sensitive_names,
        )
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return redact_credential_text(
            text,
            additional_sensitive_names=additional_sensitive_names,
        )
    # A URL in a diagnostic may have surrounding text. In that case avoid
    # treating the entire message as one URL and redact in place instead.
    if any(character.isspace() for character in text):
        return redact_credential_text(
            text,
            additional_sensitive_names=additional_sensitive_names,
        )
    query = [
        (name, item)
        for name, item in parse_qsl(parsed.query, keep_blank_values=True)
        if name.strip().lower() not in sensitive_names
    ]
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            urlencode(query, doseq=True),
            parsed.fragment,
        )
    )


def sanitize_json_value(
    value: Any,
    *,
    additional_sensitive_names: Sequence[str] = (),
) -> Any:
    """Recursively remove credential material while retaining useful provenance."""

    sensitive_names = _sensitive_names(additional_sensitive_names)
    if isinstance(value, Mapping):
        sanitized: dict[Any, Any] = {}
        for key, item in value.items():
            if str(key).strip().lower() in sensitive_names:
                sanitized[key] = REDACTED_CREDENTIAL
            else:
                sanitized[key] = sanitize_json_value(
                    item,
                    additional_sensitive_names=additional_sensitive_names,
                )
        return sanitized
    if isinstance(value, list):
        return [
            sanitize_json_value(item, additional_sensitive_names=additional_sensitive_names)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            sanitize_json_value(item, additional_sensitive_names=additional_sensitive_names)
            for item in value
        )
    if isinstance(value, str):
        return sanitize_url_credentials(
            value,
            additional_sensitive_names=additional_sensitive_names,
        )
    return value


def contains_credential_material(
    value: Any,
    *,
    additional_sensitive_names: Sequence[str] = (),
) -> bool:
    """Return True only when a value still contains non-redacted credentials."""

    sensitive_names = _sensitive_names(additional_sensitive_names)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().lower() in sensitive_names:
                if str(item or "").strip().lower() not in _REDACTED_VALUES:
                    return True
            elif contains_credential_material(
                item,
                additional_sensitive_names=additional_sensitive_names,
            ):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(
            contains_credential_material(
                item,
                additional_sensitive_names=additional_sensitive_names,
            )
            for item in value
        )
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if re.search(r"https?://[^/@\s]+@", text, flags=re.IGNORECASE):
        return True
    pattern = _credential_pattern(additional_sensitive_names)
    return any(
        match.group("value").strip().lower() not in _REDACTED_VALUES
        for match in pattern.finditer(text)
    )


def sanitize_json_text(
    value: str,
    *,
    additional_sensitive_names: Sequence[str] = (),
) -> tuple[str, bool]:
    """Scrub a JSON document, falling back to safe text redaction if malformed."""

    text = str(value)
    if not _text_may_contain_credentials(
        text,
        additional_sensitive_names=additional_sensitive_names,
    ):
        return text, False
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        sanitized = redact_credential_text(
            text,
            additional_sensitive_names=additional_sensitive_names,
        ) or ""
        return sanitized, sanitized != text
    if not contains_credential_material(
        payload,
        additional_sensitive_names=additional_sensitive_names,
    ):
        return text, False
    sanitized_payload = sanitize_json_value(
        payload,
        additional_sensitive_names=additional_sensitive_names,
    )
    return json.dumps(sanitized_payload, sort_keys=True), True


def validate_sec_user_agent(value: str | None) -> str:
    """Validate the contact-bearing identity required before any SEC request."""

    user_agent = str(value or "").strip()
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in user_agent):
        # Never echo it: a header value with a line break can smuggle headers.
        raise InvalidSecUserAgentError(
            "SEC_USER_AGENT_INVALID: VOE_SEC_USER_AGENT contains a line break or other "
            "control character; set it to one line: a name and a contact email."
        )
    match = _EMAIL_RE.search(user_agent)
    if not user_agent or match is None:
        raise InvalidSecUserAgentError(
            "SEC_USER_AGENT_INVALID: set VOE_SEC_USER_AGENT to an application identity "
            "containing a monitored contact email, e.g. "
            "VOE_SEC_USER_AGENT=\"Jane Doe jane.doe@yourdomain.com\". The SEC requires "
            "every automated client to identify itself: "
            "https://www.sec.gov/os/accessing-edgar-data"
        )
    domain = match.group(1).lower().rstrip(".")
    if any(
        domain == reserved or domain.endswith(f".{reserved}")
        for reserved in _RESERVED_CONTACT_DOMAINS
    ) or domain.endswith(".invalid"):
        raise InvalidSecUserAgentError(
            "SEC_USER_AGENT_INVALID: VOE_SEC_USER_AGENT contains a placeholder contact domain."
        )
    if "your_email" in user_agent.lower():
        raise InvalidSecUserAgentError(
            "SEC_USER_AGENT_INVALID: VOE_SEC_USER_AGENT contains the repository placeholder."
        )
    return user_agent


def require_private_env_file(path: str | Path) -> int | None:
    """Refuse to load an existing .env file readable or writable by group/other."""

    env_path = Path(path)
    if not env_path.exists():
        return None
    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode & 0o077:
        raise InsecureEnvPermissionsError(
            f"ENV_PERMISSIONS_INSECURE: {env_path} mode is {mode:04o}; run chmod 600 {env_path}."
        )
    return mode
