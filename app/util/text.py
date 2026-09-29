from __future__ import annotations

import re


WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_snippet(text: str, token: str, width: int = 300) -> str:
    text = normalize_whitespace(text)
    idx = text.lower().find(token.lower())
    if idx < 0:
        return text[: min(len(text), width)]
    start = max(0, idx - width // 2)
    end = min(len(text), idx + width // 2)
    return text[start:end]


def try_parse_number(raw: str) -> float | None:
    cleaned = raw.strip().replace(",", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = f"-{cleaned[1:-1]}"
    try:
        return float(cleaned)
    except ValueError:
        return None
