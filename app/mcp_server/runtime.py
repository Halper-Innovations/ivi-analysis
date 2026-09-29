"""Process setup for the MCP server: where data lives, and the SEC identity check.

An MCP client launches the server from whatever working directory it likes
(often ``/`` or a read-only app bundle), so the repository's relative ``data/``
default is wrong here. Unless ``VOE_DATA_DIR`` is set, the server keeps its
HTTP cache in the per-user cache directory instead.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from app.util.credential_hygiene import validate_sec_user_agent

APP_DIR_NAME = "ivi-analysis"


def default_data_dir(
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """The per-user cache directory the server uses when ``VOE_DATA_DIR`` is unset.

    macOS: ``~/Library/Caches/ivi-analysis``; Windows: ``%LOCALAPPDATA%\\ivi-analysis``;
    elsewhere ``$XDG_CACHE_HOME/ivi-analysis`` or ``~/.cache/ivi-analysis``.
    """

    env = os.environ if environ is None else environ
    plat = sys.platform if platform is None else platform
    base_home = Path.home() if home is None else home
    if plat == "darwin":
        return base_home / "Library" / "Caches" / APP_DIR_NAME
    if plat.startswith("win"):
        local = env.get("LOCALAPPDATA", "").strip()
        root = Path(local) if local else base_home / "AppData" / "Local"
        return root / APP_DIR_NAME
    xdg = env.get("XDG_CACHE_HOME", "").strip()
    if xdg and Path(xdg).is_absolute():
        return Path(xdg) / APP_DIR_NAME
    return base_home / ".cache" / APP_DIR_NAME


def apply_data_dir_default(environ: dict[str, str] | None = None) -> Path:
    """Point ``VOE_DATA_DIR`` (and ``VOE_DB_PATH``) at the user cache dir unless already set.

    Must run before ``get_config()`` is first called. Returns the effective data dir.
    """

    env = os.environ if environ is None else environ
    configured = env.get("VOE_DATA_DIR", "").strip()
    if configured:
        data_dir = Path(configured).expanduser()
    else:
        data_dir = default_data_dir(env)
        env["VOE_DATA_DIR"] = str(data_dir)
    if not env.get("VOE_DB_PATH", "").strip():
        # Nothing in the server opens the database, but a relative default would
        # anchor inside the installed package if anything ever did.
        env["VOE_DB_PATH"] = str(data_dir / "engine.db")
    return data_dir


def prepare_runtime() -> Path:
    """Set the data-dir default, load config, and validate the SEC identity.

    Raises ``InvalidSecUserAgentError`` (with the operator-facing message) when
    ``VOE_SEC_USER_AGENT`` is missing or a placeholder. Returns the cache dir.
    """

    from app.config import get_config

    apply_data_dir_default()
    get_config.cache_clear()
    cfg = get_config()
    validate_sec_user_agent(cfg.sec_user_agent)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir
