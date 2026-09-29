"""``--as-of`` is validated up front, and ``ivi value`` never prints a traceback.

A malformed date is a usage error (exit 2) naming the option, for ``value``,
``analyze`` and ``deep-research``. An unexpected failure inside ``ivi value``
becomes one line on stderr and exit 1.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "argv",
    [
        ["value", "KO", "--as-of", "2024-13-45"],
        ["analyze", "KO", "--as-of", "yesterday"],
        ["deep-research", "--ticker", "KO", "--as-of", "2024/06/30"],
    ],
)
def test_bad_as_of_is_a_usage_error(argv, monkeypatch):
    called: list[object] = []
    monkeypatch.setattr("app.research.deep_research.run_deep_research", lambda *a, **k: called.append(a))
    monkeypatch.setattr("app.valuation.quick_value.run_value", lambda *a, **k: called.append(a))
    result = runner.invoke(app, argv, env={"COLUMNS": "240"})
    assert result.exit_code == 2, result.output
    flat = " ".join(re.sub("[\u2500-\u257f]", " ", result.output).split())
    assert "--as-of" in flat
    assert "is not a date; use YYYY-MM-DD, e.g. 2024-06-30." in flat
    assert called == []
    assert "Traceback" not in result.output


def test_value_unexpected_error_is_one_line(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("disk\nfull api_token=SEKRET")

    monkeypatch.setattr("app.valuation.quick_value.run_value", _boom)
    result = runner.invoke(app, ["value", "KO", "--as-of", "2024-06-30"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.output.strip() == "ivi value KO failed: RuntimeError: disk full api_token=REDACTED"
