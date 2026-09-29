"""Plain text of an EDGAR document, paged, with a section index for 10-K/10-Q style filings."""

from __future__ import annotations

import html
import re
import threading
from collections import OrderedDict
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

from app.ingest.sec_client import SecClient
from app.mcp_server.companies import find_filing, parse_accession, resolve_company
from app.mcp_server.errors import SecToolError
from app.util.html_strip import strip_html

# A filing document larger than this is refused rather than read into memory.
MAX_FILING_BYTES = 25 * 1024 * 1024

MIN_PAGE_CHARS = 500
MAX_PAGE_CHARS = 100_000
MAX_SECTIONS = 60
_TEXT_CACHE_SLOTS = 4

_DROP_BLOCKS = [
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"<head\b[^>]*>.*?</head\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<script\b[^>]*>.*?</script\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<style\b[^>]*>.*?</style\s*>", re.DOTALL | re.IGNORECASE),
    # Inline XBRL keeps thousands of hidden contexts and facts in its header.
    re.compile(r"<ix:header\b[^>]*>.*?</ix:header\s*>", re.DOTALL | re.IGNORECASE),
]
# One table cell: not self-closing (iXBRL writes thousands of empty <td/>), and
# never spanning into another cell, row or table.
_CELL = re.compile(
    r"(<t[dh]\b[^>]*(?<!/)>)((?:(?!</?(?:t[dhr]|table)\b).)*?)(</t[dh]\s*>)",
    re.DOTALL | re.IGNORECASE,
)
_BLOCK_TAG = re.compile(r"</?(?:p|div|br|li|h[1-6])\b[^>]*>", re.IGNORECASE)
_ENTITY = re.compile(r"&(?:#\d+|#[xX][0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);")
_MARKUP_ENTITIES = {"<", ">", "&"}
_ODD_SPACES = re.compile("[\u00a0\u2002\u2003\u2009\u202f]")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
_MANY_SPACES = re.compile(r"[ \t]{2,}")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_PRE_BLOCK = re.compile(r"(<pre\b[^>]*>.*?</pre\s*>)", re.DOTALL | re.IGNORECASE)
_SOURCE_WHITESPACE = re.compile(r"\s+")
_ARCHIVE_PATH = re.compile(r"/Archives/edgar/data/\d+/(\d{10})(\d{2})(\d{6})/")
_HEADING = re.compile(r"^(PART\s+(?:I{1,3}|IV)\b|ITEM\s+(\d{1,2}[A-D]?)\b)", re.IGNORECASE)

_cache_lock = threading.Lock()
_text_cache: OrderedDict[str, tuple[str, list[dict[str, Any]]]] = OrderedDict()


def _decode_entity(match: re.Match[str]) -> str:
    entity = match.group(0)
    decoded = html.unescape(entity)
    # Leave markup-significant entities for strip_html so decoding them cannot
    # fabricate tags out of literal text.
    return entity if decoded in _MARKUP_ENTITIES or decoded == entity else decoded


def _flatten_cell(match: re.Match[str]) -> str:
    # Keep a table row on one line: block tags inside a cell become spaces,
    # and the cell boundary itself becomes a space.
    return f"{match.group(1)}{_BLOCK_TAG.sub(' ', match.group(2))} {match.group(3)}"


def html_to_text(markup: str) -> str:
    """Readable text from an EDGAR HTML/iXBRL document (uses app.util.html_strip)."""

    text = markup
    for pattern in _DROP_BLOCKS:
        text = pattern.sub(" ", text)
    # Source line breaks are just whitespace in HTML (line structure comes from
    # block tags, which strip_html turns into newlines), except inside <pre>.
    parts = _PRE_BLOCK.split(text)
    text = "".join(
        part if idx % 2 else _SOURCE_WHITESPACE.sub(" ", part) for idx, part in enumerate(parts)
    )
    text = _CELL.sub(_flatten_cell, text)
    text = _ENTITY.sub(_decode_entity, text)
    text = _ODD_SPACES.sub(" ", text)
    text = _ZERO_WIDTH.sub("", text)
    text = strip_html(text)
    # strip_html collapses blank runs before it trims each line, so lines that
    # held only whitespace leave long runs of empty lines behind.
    return _MANY_NEWLINES.sub("\n\n", text)


def plain_to_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _ODD_SPACES.sub(" ", _ZERO_WIDTH.sub("", text))
    text = _MANY_SPACES.sub(" ", text)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def document_to_text(content: bytes) -> str:
    head = content[:1024]
    if head.lstrip().startswith(b"%PDF"):
        raise SecToolError("This document is a PDF; only HTML and text documents are supported.")
    if b"\x00" in head:
        raise SecToolError("This document is binary (an image or archive), not text.")
    try:
        raw = content.decode("utf-8")
    except UnicodeDecodeError:
        raw = content.decode("cp1252", errors="replace")
    if re.search(r"<(html|body|div|p|table|document|xbrl|ix:|font|span)\b", raw[:20000], re.IGNORECASE):
        return html_to_text(raw)
    return plain_to_text(raw)


def section_index(text: str) -> list[dict[str, Any]]:
    """Offsets of PART/Item headings. The last occurrence wins, skipping the table of contents."""

    found: dict[tuple[str, str], dict[str, Any]] = {}
    part = ""
    offset = 0
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        match = _HEADING.match(stripped)
        if match and len(stripped) <= 150:
            heading = stripped
            if len(heading) < 16:
                # "Item 7." alone on a line: its title is on the next line.
                following = next((ln.strip() for ln in lines[idx + 1 : idx + 3] if ln.strip()), "")
                if following and len(following) <= 120 and not _HEADING.match(following):
                    heading = f"{heading} {following}"
            if match.group(2):
                key = (part, "ITEM " + match.group(2).upper())
            else:
                part = " ".join(match.group(1).upper().split())
                key = (part, "")
            found[key] = {"heading": heading[:160], "offset": offset}
        offset += len(line) + 1
    # A table of contents that precedes the first PART line leaves part-less
    # duplicates of items that reappear under a PART; drop those.
    placed = {item for (part_key, item) in found if part_key and item}
    kept = [entry for (part_key, item), entry in found.items() if part_key or item not in placed]
    return sorted(kept, key=lambda s: s["offset"])[:MAX_SECTIONS]


def normalize_sec_url(url: str) -> str:
    """Validate a sec.gov URL (https, no credentials); unwrap inline-XBRL viewer links."""

    text = str(url or "").strip()
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() not in {"http", "https"} or not host:
        raise SecToolError(f"Not a web URL: {text!r}. Pass an https://www.sec.gov/... URL or an accession.")
    if not (host == "sec.gov" or host.endswith(".sec.gov")):
        raise SecToolError(f"Only sec.gov documents can be fetched; {host!r} is not allowed.")
    if parsed.username or parsed.password or parsed.port not in (None, 443, 80):
        raise SecToolError("sec.gov URLs with credentials or custom ports are not allowed.")
    if parsed.path.rstrip("/") in {"/ix", "/cgi-bin/viewer"}:
        doc = (parse_qs(parsed.query).get("doc") or [""])[0]
        if doc.startswith("/"):
            return f"https://www.sec.gov{doc}"
    return urlunparse(("https", host, parsed.path or "/", "", parsed.query, ""))


def _cleaned(url: str) -> tuple[str, list[dict[str, Any]]]:
    with _cache_lock:
        if url in _text_cache:
            _text_cache.move_to_end(url)
            return _text_cache[url]
    text = document_to_text(SecClient().download_bytes(url, max_bytes=MAX_FILING_BYTES))
    entry = (text, section_index(text))
    with _cache_lock:
        _text_cache[url] = entry
        while len(_text_cache) > _TEXT_CACHE_SLOTS:
            _text_cache.popitem(last=False)
    return entry


def clear_text_cache() -> None:
    with _cache_lock:
        _text_cache.clear()


def get_filing_text(
    url_or_accession: str,
    ticker_or_cik: str | None = None,
    max_chars: int = 20_000,
    offset: int = 0,
) -> dict[str, Any]:
    target = str(url_or_accession or "").strip()
    if not target:
        raise SecToolError("url_or_accession is required.")
    max_chars = max(MIN_PAGE_CHARS, min(int(max_chars), MAX_PAGE_CHARS))
    offset = int(offset)
    if offset < 0:
        raise SecToolError("offset must be 0 or greater.")

    meta: dict[str, Any] = {}
    accession = parse_accession(target)
    if accession:
        if not ticker_or_cik:
            raise SecToolError(
                "An accession number needs ticker_or_cik (the filer), or pass the document URL "
                "from list_filings instead."
            )
        company = resolve_company(ticker_or_cik)
        filing = find_filing(company.cik, accession)
        if not filing.get("primary_document_url"):
            raise SecToolError(
                f"Filing {accession} has no primary document; browse its index: {filing['index_url']}"
            )
        url = filing["primary_document_url"]
        meta = {"accession": accession, "form": filing["form"], "filed": filing["filed"]}
    else:
        url = normalize_sec_url(target)
        archived = _ARCHIVE_PATH.search(urlparse(url).path)
        if archived:
            meta = {"accession": "-".join(archived.groups())}

    text, sections = _cleaned(url)
    total = len(text)
    if total == 0:
        raise SecToolError(f"The document at {url} has no readable text.")
    if offset >= total:
        raise SecToolError(f"offset {offset} is past the end of the document (total_chars={total}).")
    page = text[offset : offset + max_chars]
    end = offset + len(page)
    result: dict[str, Any] = {"url": url, **meta}
    result.update(
        {
            "total_chars": total,
            "offset": offset,
            "returned_chars": len(page),
            "next_offset": end if end < total else None,
        }
    )
    if offset == 0 and sections:
        result["sections"] = sections
    result["text"] = page
    return result
