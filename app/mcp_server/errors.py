"""Turn failures into short, model-readable messages. Never a stack trace."""

from __future__ import annotations

import json
import logging

import requests

from app.util.credential_hygiene import InvalidSecUserAgentError, redact_credential_text
from app.util.http import (
    AllowlistError,
    DomainBudgetExceeded,
    NetworkDisabledError,
    ResponseTooLarge,
    TooManyRedirects,
)

logger = logging.getLogger("app.mcp_server")


class SecToolError(Exception):
    """An anticipated failure whose message is safe and useful to show the caller."""


def describe_exception(exc: BaseException) -> str:
    """One-line explanation of ``exc`` for a tool result. Logs unexpected ones to stderr."""

    if isinstance(exc, SecToolError):
        return str(exc)
    if isinstance(exc, InvalidSecUserAgentError):
        return str(exc)
    if isinstance(exc, AllowlistError):
        return f"Only sec.gov URLs can be fetched ({exc})."
    if isinstance(exc, DomainBudgetExceeded):
        return f"Request budget exhausted for this call: {exc}."
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        status = response.status_code if response is not None else None
        url = redact_credential_text(response.url) if response is not None else "unknown URL"
        if status == 404:
            return f"SEC EDGAR has nothing at {url} (HTTP 404)."
        if status == 403:
            return (
                "SEC EDGAR refused the request (HTTP 403). This usually means the "
                "VOE_SEC_USER_AGENT identity was rejected or the SEC rate limit was hit; "
                "wait a minute and retry."
            )
        return f"SEC EDGAR returned HTTP {status} for {url}."
    if isinstance(exc, (ResponseTooLarge, TooManyRedirects)):
        return f"{redact_credential_text(str(exc))}."
    if isinstance(exc, NetworkDisabledError):
        return (
            "Offline: VOE_NET_PROVIDER=disabled and this SEC data is not in the local cache. "
            "Unset VOE_NET_PROVIDER (or set it to enabled) to fetch it."
        )
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return (
            "Could not reach SEC EDGAR "
            f"({type(exc).__name__}). Check the network connection and retry."
        )
    if isinstance(exc, json.JSONDecodeError):
        return "SEC EDGAR returned a response that is not valid JSON; retry later."
    logger.error("Unexpected tool failure", exc_info=exc)
    detail = _one_line(redact_credential_text(str(exc)) or "")
    return f"Unexpected internal error ({type(exc).__name__}): {detail}"


def _one_line(text: str, limit: int = 300) -> str:
    """Control characters (a newline in a header value, say) become spaces; long text is cut."""

    flat = "".join(" " if ord(ch) < 32 or ord(ch) == 127 else ch for ch in text)
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."
