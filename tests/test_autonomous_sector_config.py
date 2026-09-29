from __future__ import annotations

import pytest

from app.config import (
    AppConfig,
    canonical_market_cap_focus,
    get_config,
    resolve_autonomous_sector_pipeline_version,
)


def test_v2_cap_band_allowlist_defaults_empty(monkeypatch):
    monkeypatch.delenv("VOE_AUTONOMOUS_SECTOR_V2_CAP_BANDS", raising=False)
    get_config.cache_clear()

    cfg = get_config()

    assert cfg.autonomous_sector_v2_cap_bands == []
    assert resolve_autonomous_sector_pipeline_version("large_and_mega", cfg=cfg) == "v1"
    get_config.cache_clear()


def test_v2_cap_band_allowlist_parses_normalizes_and_deduplicates(monkeypatch):
    monkeypatch.setenv(
        "VOE_AUTONOMOUS_SECTOR_V2_CAP_BANDS",
        " large_and_mega,MEGA_CAP,large_and_mega, small-cap ",
    )
    get_config.cache_clear()

    cfg = get_config()

    assert cfg.autonomous_sector_v2_cap_bands == [
        "large_and_mega",
        "mega_cap",
        "small-cap",
    ]
    assert resolve_autonomous_sector_pipeline_version("large_and_mega", cfg=cfg) == "v2"
    assert resolve_autonomous_sector_pipeline_version("small cap", cfg=cfg) == "v2"
    assert resolve_autonomous_sector_pipeline_version("mid_cap", cfg=cfg) == "v1"
    get_config.cache_clear()


def test_explicit_pipeline_version_overrides_rollout_allowlist():
    cfg = AppConfig(autonomous_sector_v2_cap_bands=["large_and_mega"])

    assert (
        resolve_autonomous_sector_pipeline_version(
            "large_and_mega",
            requested="V1",
            cfg=cfg,
        )
        == "v1"
    )
    assert resolve_autonomous_sector_pipeline_version("micro", requested="v2", cfg=cfg) == "v2"
    with pytest.raises(ValueError, match="must be 'v1' or 'v2'"):
        resolve_autonomous_sector_pipeline_version("large_and_mega", requested="v3", cfg=cfg)


@pytest.mark.parametrize(
    ("allowlisted", "requested_band"),
    [
        ("micro", "micro_cap"),
        ("micro_cap", "micro"),
        ("small", "small_cap"),
        ("small_cap", "small"),
        ("mid", "mid_cap"),
        ("mid_cap", "mid"),
        ("large", "large_cap"),
        ("mega", "mega_cap"),
    ],
)
def test_rollout_allowlist_canonicalizes_cap_band_aliases(
    allowlisted, requested_band
):
    cfg = AppConfig(autonomous_sector_v2_cap_bands=[allowlisted])

    assert (
        resolve_autonomous_sector_pipeline_version(requested_band, cfg=cfg) == "v2"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("small cap", "small_cap"),
        ("large-cap", "large_cap"),
        ("large and mega", "large_and_mega"),
        ("mega", "mega_cap"),
        ("smid", "smid_cap"),
    ],
)
def test_market_cap_focus_canonicalizer_is_shared_across_alias_forms(value, expected):
    assert canonical_market_cap_focus(value) == expected
