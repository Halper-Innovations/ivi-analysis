"""Strip HTML tags and normalize whitespace from filing text.

SEC EDGAR 10-K filings are served as HTML. The raw text contains
<span>, <div>, <table>, style attributes, and other markup that wastes
tokens and confuses LLM analysis. This module converts HTML to clean
prose while preserving paragraph structure.
"""

from __future__ import annotations

import re

# Pre-compiled patterns for performance (applied to millions of chars)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&nbsp;": " ",
    "&quot;": '"',
    "&#8217;": "'",
    "&#8220;": "\u201c",
    "&#8221;": "\u201d",
    "&#8212;": "\u2014",
    "&#8211;": "\u2013",
    "&#160;": " ",
    "&#xa0;": " ",
}
_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _replace_entity(match: re.Match) -> str:
    entity = match.group(0)
    return _ENTITY_MAP.get(entity, " ")


def strip_html(text: str) -> str:
    """Convert HTML to clean text.

    - Removes all HTML tags
    - Decodes common HTML entities
    - Collapses excessive whitespace while preserving paragraph breaks
    - Returns plain text suitable for LLM consumption
    """
    if not text:
        return text

    # Replace block-level tags with newlines to preserve paragraph structure
    text = re.sub(r"<(?:br|p|div|tr|li|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|tr|li|h[1-6]|table)>", "\n", text, flags=re.IGNORECASE)

    # Remove all remaining tags
    text = _TAG_RE.sub("", text)

    # Decode HTML entities
    text = _ENTITY_RE.sub(_replace_entity, text)

    # Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove empty lines at start/end
    return text.strip()
