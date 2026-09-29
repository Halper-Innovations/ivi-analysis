"""Atomic JSON I/O helpers.

These helpers exist to prevent state-file corruption when a process is killed
mid-write (e.g. ``scout_state.json`` is rewritten many times per batch). A
non-atomic ``path.write_text(...)`` can leave a half-written, unparseable file
behind; on resume a silent ``return {}`` then masks the corruption and the
caller restarts work from scratch.

``atomic_write_json`` writes to a sibling temp file, fsyncs, and ``os.replace``s
it into place (atomic on POSIX). ``read_json_safe`` distinguishes a genuinely
absent file (return ``default``) from a corrupt one (raise loudly by default, or
fall back to ``default`` only when explicitly asked).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonCorruptError(ValueError):
    """Raised when a JSON file exists but cannot be parsed."""


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write ``payload`` as JSON to ``path``.

    Writes to a unique sibling ``.tmp`` file, flushes + fsyncs the file,
    ``os.replace``s it into place, then fsyncs the parent directory so a
    reader never sees a partial file and a kill mid-write cannot corrupt the
    destination.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def read_json_safe(
    path: Path,
    *,
    default: Any = None,
    on_corrupt: str = "raise",
) -> Any:
    """Read JSON from ``path``, distinguishing absent from corrupt.

    - Missing file (``FileNotFoundError``): return ``default``.
    - Parse error on an existing file: with ``on_corrupt='raise'`` (default),
      log loudly and raise :class:`JsonCorruptError`; with
      ``on_corrupt='default'``, log loudly and return ``default``.
    """
    if on_corrupt not in {"raise", "default"}:
        raise ValueError(f"on_corrupt must be 'raise' or 'default', got {on_corrupt!r}")
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Corrupt JSON at %s: %s", path, exc)
        if on_corrupt == "raise":
            raise JsonCorruptError(f"Corrupt JSON at {path}: {exc}") from exc
        return default
