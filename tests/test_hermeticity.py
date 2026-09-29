from __future__ import annotations

import socket
import sqlite3
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


OWNER_DATA = Path(__file__).resolve().parents[1] / "data"


def _live_conftest(request: pytest.FixtureRequest) -> ModuleType:
    """The root ``tests/conftest.py`` instance pytest registered as a plugin.

    ``tests`` has no ``__init__.py``, so ``import tests.conftest`` would build a
    second, dead copy with an empty ledger. The plugin manager holds the one
    live instance whose audit hook actually records violations — reach it there
    rather than guessing from ``sys.modules``.
    """

    for plugin in request.config.pluginmanager.get_plugins():
        if not isinstance(plugin, ModuleType):
            continue
        path = (getattr(plugin, "__file__", "") or "").replace("\\", "/")
        if path.endswith("/tests/conftest.py") and hasattr(plugin, "_VIOLATIONS"):
            return plugin
    raise AssertionError("live tests/conftest plugin not found")


def test_network_dns_is_blocked_before_resolution(expect_hermeticity_violation) -> None:
    with pytest.raises(RuntimeError, match="socket.getaddrinfo"):
        socket.getaddrinfo("example.invalid", 443)


def test_owner_file_open_is_blocked(expect_hermeticity_violation) -> None:
    with pytest.raises(RuntimeError, match="live_data"):
        (OWNER_DATA / "engine.db").open("rb")


def test_owner_path_stat_is_blocked(expect_hermeticity_violation) -> None:
    with pytest.raises(RuntimeError, match="os.stat"):
        (OWNER_DATA / "engine.db").exists()


def test_owner_sqlite_connection_is_blocked(expect_hermeticity_violation) -> None:
    with pytest.raises(RuntimeError, match="SQLite owner DB access"):
        sqlite3.connect(str(OWNER_DATA / "engine.db"))


def test_subprocess_is_blocked_before_spawn(expect_hermeticity_violation) -> None:
    with pytest.raises(RuntimeError, match="subprocess.Popen"):
        subprocess.run(["/usr/bin/true"], check=True)


def test_swallowed_violation_is_recorded_for_teardown(
    request: pytest.FixtureRequest,
) -> None:
    """A caught HermeticityViolation must still be queued so teardown surfaces it.

    Production code sometimes wraps I/O in a broad ``except``; the embargo must
    not be defeated by swallowing the exception. Swallow one here and assert it
    was recorded against this node (which is what the teardown check and the
    session-final backstop both consume), then undo the deliberate hit so this
    self-test leaves no residue in the violation ledger or counters.
    """

    conftest = _live_conftest(request)

    nodeid = request.node.nodeid
    blocked_before = conftest._COUNTS.get("network_blocked", 0)
    try:
        socket.getaddrinfo("swallowed.invalid", 443)
    except RuntimeError:
        pass  # swallow exactly as buggy production code might
    assert conftest._VIOLATIONS.get(nodeid), "swallowed violation was not recorded"
    conftest._VIOLATIONS.pop(nodeid, None)
    conftest._COUNTS["network_blocked"] = blocked_before


def test_temp_paths_remain_readable_and_writable(tmp_path: Path) -> None:
    target = tmp_path / "allowed.txt"
    target.write_text("ok", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "ok"
