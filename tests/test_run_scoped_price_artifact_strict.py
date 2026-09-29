"""A run-scoped price artifact is evidence only for the exact date it names.

An artifact naming no ``requested_as_of_date`` used to be accepted for any date, and a
boolean or non-positive/non-finite snapshot price was accepted as a price. Mirrors the
valuation writer's ``_load_run_scoped_price_artifact``. Conservative call: the snapshot's
own price is required (no fallback to diagnostic fields in these three readers).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.autonomous.price_repair import _run_candidates
from app.config import AppConfig
from app.market.price_provider import _load_run_scoped_output
from app.valuation import engine


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "x.db",
        outputs_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
        sectors_dir=tmp_path / "sectors",
    )


def _write(tmp_path: Path, *, requested, price) -> Path:
    path = tmp_path / "outputs" / "prices" / "run1" / "AAA.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "OK",
        "snapshot": {
            "ticker": "AAA",
            "as_of_date": "2026-06-01",
            "price": price,
            "currency": "USD",
            "source": "p",
        },
    }
    if requested is not None:
        payload["requested_as_of_date"] = requested
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _engine(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "get_config", lambda: _cfg(tmp_path))
    return engine._load_run_scoped_price_snapshot("run1", "AAA", "2026-06-01")[0]


def _provider(tmp_path):
    return _load_run_scoped_output(
        tmp_path / "outputs" / "prices" / "run1" / "AAA.json",
        requested_as_of_date="2026-06-01",
    )[0]


def _repair(tmp_path):
    cands, _err = _run_candidates(
        cfg=_cfg(tmp_path), run_id="run1", ticker="AAA", requested_as_of_date="2026-06-01"
    )
    return cands or None


READERS = {
    "engine": None,
    "provider": _provider,
    "repair": _repair,
}


def _read(name, tmp_path, monkeypatch):
    if name == "engine":
        return _engine(tmp_path, monkeypatch)
    return READERS[name](tmp_path)


@pytest.mark.parametrize("reader", ["engine", "provider", "repair"])
def test_exact_date_and_good_price_accepted(reader, tmp_path, monkeypatch):
    _write(tmp_path, requested="2026-06-01", price=31.0)
    assert _read(reader, tmp_path, monkeypatch) is not None


@pytest.mark.parametrize("reader", ["engine", "provider", "repair"])
def test_undated_artifact_rejected(reader, tmp_path, monkeypatch):
    _write(tmp_path, requested=None, price=31.0)
    assert _read(reader, tmp_path, monkeypatch) is None


@pytest.mark.parametrize("reader", ["engine", "provider", "repair"])
def test_other_date_rejected(reader, tmp_path, monkeypatch):
    _write(tmp_path, requested="2026-05-29", price=31.0)
    assert _read(reader, tmp_path, monkeypatch) is None


@pytest.mark.parametrize("reader", ["engine", "provider", "repair"])
@pytest.mark.parametrize("bad", [True, 0, -5.0, "31", None, float("nan"), float("inf")])
def test_bad_price_rejected(reader, bad, tmp_path, monkeypatch):
    _write(tmp_path, requested="2026-06-01", price=bad)
    assert _read(reader, tmp_path, monkeypatch) is None
