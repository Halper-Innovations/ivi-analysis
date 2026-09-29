"""One markdown renderer for every artifact surface.

Digests, run reports, and Reader content all render through the same
commonmark+tables pipeline with raw HTML disabled — artifact HTML is never
trusted, and the Reader stays visually consistent because there is exactly
one renderer to style against. The Reader variant additionally stamps
heading ids and returns the table of contents.
"""

from __future__ import annotations

import re
from typing import Any

from markdown_it import MarkdownIt

_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")

_SLUG_STRIP = re.compile(r"[^a-z0-9 _-]+")
_SLUG_SPACES = re.compile(r"[\s_]+")


def render_markdown(text: str) -> str:
    return _markdown.render(text)


def _slugify(title: str) -> str:
    slug = _SLUG_SPACES.sub("-", _SLUG_STRIP.sub("", title.strip().lower())).strip("-")
    return slug or "section"


def render_markdown_with_toc(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Render with heading ids stamped; returns (html, toc).

    toc entries: {"level": int, "text": str, "anchor": str}. Anchors are
    slugified heading text, deduplicated with ``-2``, ``-3``… suffixes so
    repeated section names stay addressable.
    """
    tokens = _markdown.parse(text)
    toc: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1] if index + 1 < len(tokens) else None
        title = inline.content if inline is not None and inline.type == "inline" else ""
        anchor = _slugify(title)
        count = seen.get(anchor, 0)
        seen[anchor] = count + 1
        if count:
            anchor = f"{anchor}-{count + 1}"
        token.attrSet("id", anchor)
        toc.append({"level": int(token.tag[1]), "text": title, "anchor": anchor})
    html = _markdown.renderer.render(tokens, _markdown.options, {})
    return html, toc
