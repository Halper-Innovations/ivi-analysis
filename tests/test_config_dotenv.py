"""How .env files are read, and where the database defaults to.

Policy (2026-09-29, conservative call): the working directory's ``.env`` is
read only with ``VOE_DOTENV_CWD=1``; any ``.env`` contributes ``VOE_*`` names
only; a group/other-accessible file is skipped with a stderr warning rather
than crashing every command. The Anthropic SDK is always given an explicit
base URL from ``VOE_*`` config, so an ambient ``ANTHROPIC_BASE_URL`` cannot
redirect the API key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import app.config as config_mod
from app.config import get_config, load_env_files


def _write_env(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)
    return path


def test_cwd_env_is_not_read_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VOE_DOTENV_CWD", raising=False)
    monkeypatch.chdir(tmp_path)
    candidates = config_mod._dotenv_candidates()
    assert tmp_path / ".env" not in candidates
    assert candidates == [Path(config_mod.__file__).resolve().parents[1] / ".env"]


def test_cwd_env_is_read_only_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("VOE_DOTENV_CWD", "1")
    monkeypatch.chdir(tmp_path)
    candidates = config_mod._dotenv_candidates()
    assert candidates[0] == tmp_path.resolve() / ".env" or candidates[0] == tmp_path / ".env"


def test_deleted_cwd_does_not_crash(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VOE_DOTENV_CWD", "1")

    def _gone() -> Path:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(config_mod.Path, "cwd", staticmethod(_gone))
    candidates = config_mod._dotenv_candidates()
    assert len(candidates) == 1
    assert "working directory no longer exists" in capsys.readouterr().err


def test_only_voe_names_are_imported(tmp_path, monkeypatch):
    env = _write_env(
        tmp_path / ".env",
        "VOE_PROBE_SETTING=from-file\n"
        "ANTHROPIC_BASE_URL=https://attacker.test\n"
        "VOE_LLM_PROBE=anthropic\n",
    )
    for name in ("VOE_PROBE_SETTING", "ANTHROPIC_BASE_URL", "VOE_LLM_PROBE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VOE_LLM_PROBE", "kept")
    assert load_env_files([env]) == [env]
    assert os.environ["VOE_PROBE_SETTING"] == "from-file"
    assert "ANTHROPIC_BASE_URL" not in os.environ
    # Never overwrites what the shell already set.
    assert os.environ["VOE_LLM_PROBE"] == "kept"
    monkeypatch.delenv("VOE_PROBE_SETTING")


def test_world_readable_env_is_skipped_with_warning(tmp_path, monkeypatch, capsys):
    env = _write_env(tmp_path / ".env", "VOE_PROBE_OPEN=1\n", mode=0o644)
    monkeypatch.delenv("VOE_PROBE_OPEN", raising=False)
    assert load_env_files([env]) == []
    assert "VOE_PROBE_OPEN" not in os.environ
    err = capsys.readouterr().err
    assert f"warning: skipping {env}: ENV_PERMISSIONS_INSECURE" in err


def test_db_path_follows_data_dir_when_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.delenv("VOE_DB_PATH", raising=False)
    get_config.cache_clear()
    try:
        assert get_config().db_path == tmp_path / "d" / "engine.db"
    finally:
        get_config.cache_clear()


def test_explicit_db_path_still_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "other.db"))
    get_config.cache_clear()
    try:
        assert get_config().db_path == tmp_path / "other.db"
    finally:
        get_config.cache_clear()


def test_anthropic_client_gets_explicit_base_url(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://attacker.test")
    monkeypatch.delenv("VOE_ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("VOE_ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    get_config.cache_clear()
    captured: dict[str, object] = {}

    import app.llm.providers.anthropic_provider as ap

    class _Stop(Exception):
        pass

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise _Stop()

    monkeypatch.setattr(ap._anthropic_sdk, "Anthropic", _FakeAnthropic)
    try:
        assert get_config().anthropic_base_url == "https://api.anthropic.com"
        provider = ap.AnthropicProvider(get_config())
        with pytest.raises(_Stop):
            provider.synthesize_json(prompt="x", schema={"type": "object"})
    finally:
        get_config.cache_clear()
    assert captured.get("base_url") == "https://api.anthropic.com"
