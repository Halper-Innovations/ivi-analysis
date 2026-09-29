"""Smoke test: every registered CLI command's lazily-imported module must import.

Many CLI commands defer their heavy imports to call time (``from app.X import Y``
inside the command body). That means a deleted/renamed module does NOT break
``import app.cli`` — the breakage only surfaces as a ``ModuleNotFoundError`` when
the command is actually invoked. This test walks every registered command,
extracts the modules it lazily imports, and confirms each one imports cleanly,
so a future missing-module regression fails here instead of in production.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap

import pytest

import app.cli as cli


def _lazy_imported_modules(func) -> list[str]:
    """Return the ``app.*`` modules a command body imports (lazy or otherwise)."""
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return []
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("app."):
                modules.append(module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app."):
                    modules.append(alias.name)
    return modules


def _command_import_cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for command in cli.app.registered_commands:
        callback = command.callback
        if callback is None:
            continue
        command_name = command.name or getattr(callback, "__name__", "<unknown>")
        for module in _lazy_imported_modules(callback):
            key = (command_name, module)
            if key in seen:
                continue
            seen.add(key)
            cases.append(key)
    return cases


_CASES = _command_import_cases()

# Module prefixes whose command modules may be legitimately absent (none in this edition).
_KNOWN_MISSING_PREFIXES: tuple[str, ...] = ()


def _is_known_missing(module: str) -> bool:
    if not module.startswith(_KNOWN_MISSING_PREFIXES):
        return False
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        return True
    return False


def test_cli_app_imports() -> None:
    # Importing the CLI module must never raise (imports are lazy by design).
    importlib.reload(cli)


def test_command_import_cases_are_discovered() -> None:
    # Guard the harness itself: if discovery returns nothing the test below is a no-op.
    assert len(_CASES) > 0


def test_no_validation_command_imports_missing_module() -> None:
    """The validation-module cleanup target: every app.validation.* command resolves."""
    broken: list[str] = []
    for command_name, module in _CASES:
        if not module.startswith("app.validation."):
            continue
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            broken.append(f"{command_name} -> {module} ({exc})")
    assert broken == [], (
        "These CLI commands lazily import missing app.validation modules: " + "; ".join(broken)
    )


@pytest.mark.parametrize(
    ("command_name", "module"),
    _CASES,
    ids=[f"{name}->{module}" for name, module in _CASES],
)
def test_command_lazy_import_resolves(command_name: str, module: str, request) -> None:
    if _is_known_missing(module):
        request.node.add_marker(
            pytest.mark.xfail(
                reason=f"{module} is an absent module (out of scope)",
                strict=False,
            )
        )
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"CLI command {command_name!r} lazily imports {module!r}, "
            f"but the module is missing: {exc}"
        )
