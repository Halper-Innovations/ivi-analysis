from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.util.json_io import JsonCorruptError, atomic_write_json, read_json_safe


def test_atomic_write_json_writes_payload(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"status": "RUNNING", "batch_index": 3})
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "RUNNING", "batch_index": 3}


def test_atomic_write_json_leaves_no_tmp_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"a": 1})
    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == ["state.json"]


def test_atomic_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "state.json"
    atomic_write_json(target, {"k": "v"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"k": "v"}


def test_atomic_write_json_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text("{\"old\": true}", encoding="utf-8")
    atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}


def test_atomic_write_json_indent(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"a": 1}, indent=2)
    assert target.read_text(encoding="utf-8") == '{\n  "a": 1\n}'


def test_read_json_safe_missing_returns_default(tmp_path: Path) -> None:
    target = tmp_path / "absent.json"
    assert read_json_safe(target, default={}) == {}
    assert read_json_safe(target, default=None) is None
    assert read_json_safe(target, default=[1, 2]) == [1, 2]


def test_read_json_safe_reads_valid(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"status": "PARTIAL"}', encoding="utf-8")
    assert read_json_safe(target, default={}) == {"status": "PARTIAL"}


def test_read_json_safe_corrupt_raises_by_default(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"status": "RUNN', encoding="utf-8")  # truncated mid-write
    with pytest.raises(JsonCorruptError):
        read_json_safe(target, default={})


def test_read_json_safe_corrupt_default_mode_returns_default(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text('not json at all', encoding="utf-8")
    assert read_json_safe(target, default={"fallback": 1}, on_corrupt="default") == {"fallback": 1}


def test_read_json_safe_corrupt_logs_loudly(tmp_path: Path, caplog) -> None:
    target = tmp_path / "state.json"
    target.write_text('{broken', encoding="utf-8")
    with caplog.at_level("ERROR"):
        with pytest.raises(JsonCorruptError):
            read_json_safe(target, default={})
    assert any("Corrupt JSON" in rec.message for rec in caplog.records)


def test_read_json_safe_invalid_on_corrupt_arg(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json_safe(target, default={}, on_corrupt="bogus")
