"""Mount guard: the external data volume must be the volume we expect.

macOS remounts a drive whose mount point is occupied as "/Volumes/<name> 1".
Anything that wrote through the original path while the drive was away has
written a second copy of the world onto the internal disk, and `mount` cannot
tell the two apart. The sentinel is a file ON the volume carrying that
volume's own UUID, so it is absent exactly when the real volume is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    DataVolumeError,
    assert_data_volume,
    check_data_volume,
    ensure_directories,
    get_config,
)

UUID = "A9DB09EF-2885-4967-8AC1-F75E9E0F10D8"
OTHER = "00000000-0000-0000-0000-000000000000"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_config_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def _cfg(monkeypatch, tmp_path: Path, *, uuid: str | None, sentinel: Path | None):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    if uuid is None:
        monkeypatch.delenv("VOE_DATA_VOLUME_UUID", raising=False)
    else:
        monkeypatch.setenv("VOE_DATA_VOLUME_UUID", uuid)
    if sentinel is None:
        monkeypatch.delenv("VOE_DATA_VOLUME_SENTINEL", raising=False)
    else:
        monkeypatch.setenv("VOE_DATA_VOLUME_SENTINEL", str(sentinel))
    get_config.cache_clear()
    return get_config()


# ---------------------------------------------------------------------------
# the check itself
# ---------------------------------------------------------------------------


def test_unset_uuid_disables_the_check(monkeypatch, tmp_path):
    """A developer checkout with no drive must not be stopped."""
    cfg = _cfg(monkeypatch, tmp_path, uuid=None, sentinel=tmp_path / "absent")
    ok, detail = check_data_volume(cfg)
    assert ok is True
    assert detail == f"disabled (no volume sentinel at {tmp_path / 'absent'})"


def test_sentinel_default_follows_the_data_dir(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=None)
    assert cfg.data_volume_sentinel == tmp_path / "data" / ".ivi-volume"


def test_matching_sentinel_passes(monkeypatch, tmp_path):
    sentinel = tmp_path / ".ivi-volume"
    sentinel.write_text(f"{UUID}\n", encoding="utf-8")
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    assert check_data_volume(cfg) == (True, f"{sentinel} -> {UUID}")


def test_sentinel_case_is_not_significant(monkeypatch, tmp_path):
    sentinel = tmp_path / ".ivi-volume"
    sentinel.write_text(UUID.lower(), encoding="utf-8")
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    ok, detail = check_data_volume(cfg)
    assert ok is True
    assert detail == f"{sentinel} -> {UUID.lower()}"


def test_missing_sentinel_is_red(monkeypatch, tmp_path):
    sentinel = tmp_path / ".ivi-volume"
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    assert check_data_volume(cfg) == (
        False,
        f"volume not mounted: sentinel missing {sentinel}",
    )


def test_dangling_symlink_is_red_and_names_its_target(monkeypatch, tmp_path):
    """The unplugged-drive state: the link survives, the target does not."""
    target = tmp_path / "Volumes" / "ExternalDrive" / "ivi-data" / ".ivi-volume"
    sentinel = tmp_path / ".ivi-volume"
    sentinel.symlink_to(target)
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    assert check_data_volume(cfg) == (
        False,
        f"volume not mounted: dangling sentinel {sentinel} -> {target}",
    )


def test_wrong_uuid_is_red(monkeypatch, tmp_path):
    """A different drive mounted at the same path is not our drive."""
    sentinel = tmp_path / ".ivi-volume"
    sentinel.write_text(OTHER, encoding="utf-8")
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    assert check_data_volume(cfg) == (
        False,
        f"wrong volume at {sentinel}: found {OTHER}, expected {UUID}",
    )


def test_empty_sentinel_is_red(monkeypatch, tmp_path):
    sentinel = tmp_path / ".ivi-volume"
    sentinel.write_text("", encoding="utf-8")
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    assert check_data_volume(cfg) == (
        False,
        f"wrong volume at {sentinel}: found (empty), expected {UUID}",
    )


def test_assert_data_volume_raises_with_the_detail(monkeypatch, tmp_path):
    sentinel = tmp_path / ".ivi-volume"
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    with pytest.raises(DataVolumeError) as excinfo:
        assert_data_volume(cfg)
    assert str(excinfo.value) == f"volume not mounted: sentinel missing {sentinel}"


# ---------------------------------------------------------------------------
# the writers
# ---------------------------------------------------------------------------


def test_ensure_directories_creates_nothing_when_the_volume_is_absent(monkeypatch, tmp_path):
    """The split-brain guard: never mkdir -p through an unmounted volume."""
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=tmp_path / ".ivi-volume")
    with pytest.raises(DataVolumeError):
        ensure_directories(cfg)
    assert not (tmp_path / "data").exists()


def test_ensure_directories_works_when_the_volume_is_present(monkeypatch, tmp_path):
    sentinel = tmp_path / ".ivi-volume"
    sentinel.write_text(UUID, encoding="utf-8")
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    ensure_directories(cfg)
    assert (tmp_path / "data" / "outputs").is_dir()


def test_get_db_refuses_to_open_when_the_volume_is_absent(monkeypatch, tmp_path):
    from app.db import get_db

    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=tmp_path / ".ivi-volume")
    with pytest.raises(DataVolumeError):
        with get_db(cfg):
            pass
    assert not (tmp_path / "data" / "engine.db").exists()


def test_connect_refuses_and_creates_no_shadow_database(monkeypatch, tmp_path):
    from app.db import connect

    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=tmp_path / ".ivi-volume")
    with pytest.raises(DataVolumeError):
        connect(tmp_path / "shadow.db", cfg=cfg)
    assert not (tmp_path / "shadow.db").exists()


# ---------------------------------------------------------------------------
# preflight and deadman surfaces
# ---------------------------------------------------------------------------


def test_preflight_reports_the_volume_check_first(monkeypatch, tmp_path):
    from app.ops.preflight import run_preflight

    sentinel = tmp_path / ".ivi-volume"
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    result = run_preflight(cfg, require_engine=False, require_cache_dirs=False)
    assert result.checks[0].name == "data_volume_mounted"
    assert result.checks[0].ok is False
    assert result.checks[0].detail == f"volume not mounted: sentinel missing {sentinel}"
    assert result.ok is False


def test_preflight_green_when_the_sentinel_matches(monkeypatch, tmp_path):
    from app.ops.preflight import run_preflight

    sentinel = tmp_path / ".ivi-volume"
    sentinel.write_text(UUID, encoding="utf-8")
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    result = run_preflight(cfg, require_engine=False, require_cache_dirs=False)
    assert result.checks[0].to_dict() == {
        "name": "data_volume_mounted",
        "ok": True,
        "detail": f"{sentinel} -> {UUID}",
    }
    assert result.ok is True


# ---------------------------------------------------------------------------
# the cron shell guard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the leftover-mount-point case: a sentinel that reads back perfectly while
# every write lands on the internal disk. Only the device number sees it.
# ---------------------------------------------------------------------------


def test_symlink_resolving_onto_the_repo_device_is_red(monkeypatch, tmp_path):
    """macOS left an ordinary directory where the drive used to mount."""
    stale = tmp_path / "Volumes" / "ExternalDrive" / "ivi-data"
    stale.mkdir(parents=True)
    (stale / ".ivi-volume").write_text(UUID, encoding="utf-8")
    sentinel = tmp_path / ".ivi-volume"
    sentinel.symlink_to(stale / ".ivi-volume")
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    ok, detail = check_data_volume(cfg)
    assert ok is False
    assert detail.startswith(
        f"volume not mounted: {sentinel} resolves onto the repo's own device ("
    )
    assert detail.endswith(") -- a leftover mount-point directory, not the drive")


def test_a_symlinked_sentinel_arms_the_check_without_any_uuid(monkeypatch, tmp_path):
    """Provisioning is the switch: no VOE_DATA_VOLUME_UUID needed to be armed."""
    sentinel = tmp_path / ".ivi-volume"
    sentinel.symlink_to(tmp_path / "Volumes" / "ExternalDrive" / "ivi-data" / ".ivi-volume")
    cfg = _cfg(monkeypatch, tmp_path, uuid=None, sentinel=sentinel)
    ok, detail = check_data_volume(cfg)
    assert ok is False
    assert detail.startswith("volume not mounted: dangling sentinel")


def test_non_utf8_sentinel_is_red_not_a_traceback(monkeypatch, tmp_path):
    sentinel = tmp_path / ".ivi-volume"
    sentinel.write_bytes(b"\xff\xfe\x00")
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    ok, detail = check_data_volume(cfg)
    assert ok is False
    assert detail.startswith(f"sentinel unreadable {sentinel}: UnicodeDecodeError:")


def test_directory_sentinel_says_so(monkeypatch, tmp_path):
    sentinel = tmp_path / ".ivi-volume"
    sentinel.mkdir()
    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=sentinel)
    assert check_data_volume(cfg) == (
        False,
        f"sentinel is a directory, not a file: {sentinel}",
    )


# ---------------------------------------------------------------------------
# read-only opens stay available: engine.db is on the internal disk, and the
# checks that report a missing drive must not be blinded by it
# ---------------------------------------------------------------------------


def test_read_only_connect_still_works_with_the_volume_absent(monkeypatch, tmp_path):
    import sqlite3

    from app.db import connect

    db = tmp_path / "cold.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("INSERT INTO t(id) VALUES (7)")
    conn.commit()
    conn.close()

    cfg = _cfg(monkeypatch, tmp_path, uuid=UUID, sentinel=tmp_path / ".ivi-volume")
    assert check_data_volume(cfg)[0] is False  # the guard really is armed and red

    ro = connect(db, cfg=cfg, read_only=True)
    try:
        assert ro.execute("SELECT id FROM t").fetchone()[0] == 7
    finally:
        ro.close()


# ---------------------------------------------------------------------------
# the two implementations must agree, state for state
# ---------------------------------------------------------------------------


