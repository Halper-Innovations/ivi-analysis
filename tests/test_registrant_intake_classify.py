"""Tests for app.universe.registrant_intake — Phase B deterministic classification."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.universe.registrant_intake import classify_new_registrants


_SUBMISSIONS: dict[str, dict] = {
    # Maps via SIC range (3821 -> industrial_tech).
    "0000000010": {"sic": "3821", "sicDescription": "Lab Instruments", "name": "Maps Co"},
    # Unmappable SIC -> UNCLASSIFIED_REVIEW (no_sector_match).
    "0000000011": {"sic": "9995", "sicDescription": "Non-Classifiable Establishments", "name": "Farm Services Co"},
    # No SIC at all and a name with no fallback pattern -> no_sic.
    "0000000012": {"sicDescription": "", "name": "Mystery Holdings Inc"},
}


def _loader(cik: str) -> dict | None:
    # Production code passes CIKs in mixed padded/unpadded forms and
    # normalizes downstream; the fixture accepts both.
    return _SUBMISSIONS.get(str(cik).strip().zfill(10))


def _seed_registrant(
    conn: sqlite3.Connection,
    *,
    cik: str,
    ticker: str,
    name: str,
    sic: int | None,
    sic_description: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sec_registrants(
            cik, primary_ticker, all_tickers, name, exchange, exchange_scope,
            sic, sic_description, operating_status, in_scope,
            first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, 'NASDAQ', 'IN_SCOPE', ?, ?, 'OPERATING', 1,
                  '2026-06-11T00:00:00Z', '2026-06-11T00:00:00Z')
        """,
        (cik, ticker, json.dumps([ticker]), name, sic, sic_description),
    )


@pytest.fixture()
def intake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import get_config

    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    monkeypatch.setenv(
        "VOE_SECTOR_SIC_CONFIG_PATH",
        str(Path(__file__).resolve().parent / "fixtures" / "sector_sic_ranges.json"),
    )
    get_config.cache_clear()
    cfg = get_config()
    from app.db import init_db

    init_db(cfg)

    # The classifier resolves tickers through the SEC company_tickers cache.
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "company_tickers.json").write_text(
        json.dumps(
            {
                "0": {"cik_str": 10, "ticker": "MAPS", "title": "Maps Co"},
                "1": {"cik_str": 11, "ticker": "FARM", "title": "Farm Services Co"},
                "2": {"cik_str": 12, "ticker": "MYST", "title": "Mystery Holdings Inc"},
                "3": {"cik_str": 13, "ticker": "KNWN", "title": "Known Co"},
            }
        )
    )

    conn = sqlite3.connect(str(cfg.db_path))
    _seed_registrant(conn, cik="0000000010", ticker="MAPS", name="Maps Co", sic=3821)
    _seed_registrant(
        conn, cik="0000000011", ticker="FARM", name="Farm Services Co", sic=9995,
        sic_description="Non-Classifiable Establishments",
    )
    _seed_registrant(conn, cik="0000000012", ticker="MYST", name="Mystery Holdings Inc", sic=None)
    _seed_registrant(conn, cik="0000000013", ticker="KNWN", name="Known Co", sic=3821)
    conn.execute(
        "INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at) "
        "VALUES ('KNWN', '2026-04-19', 'industrial_tech', 1.0, '[]', '2026-04-19T00:00:00Z')"
    )
    conn.commit()
    conn.close()
    yield cfg
    get_config.cache_clear()


def test_classify_new_registrants_end_to_end(intake_env) -> None:
    cfg = intake_env
    report = classify_new_registrants(
        as_of_date="2026-06-11",
        db_path=cfg.db_path,
        cfg=cfg,
        submissions_loader=_loader,
    )

    assert report["scope"] == 3  # KNWN already classified, skipped
    assert report["already_classified_skipped"] == 1
    assert report["status_counts"]["classified"] == 1
    assert report["status_counts"]["no_sector_match"] == 1
    assert report["status_counts"]["no_sic"] == 1
    assert report["sector_counts"] == {"industrial_tech": 1}
    assert report["unclassified_review_count"] == 2
    review_tickers = {item["ticker"]: item["status"] for item in report["unclassified_review"]}
    assert review_tickers == {"FARM": "no_sector_match", "MYST": "no_sic"}
    assert report["unmapped_sic_counts"] == {"9995 Non-Classifiable Establishments": 1}

    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    si = conn.execute(
        "SELECT inferred_sector FROM sector_inference WHERE ticker='MAPS' AND as_of_date='2026-06-11'"
    ).fetchone()
    assert si["inferred_sector"] == "industrial_tech"
    reg = {
        r["primary_ticker"]: dict(r)
        for r in conn.execute(
            "SELECT primary_ticker, sector, classification_status FROM sec_registrants"
        )
    }
    conn.close()
    assert reg["MAPS"]["sector"] == "industrial_tech"
    assert reg["MAPS"]["classification_status"] == "classified"
    assert reg["FARM"]["sector"] is None
    assert reg["FARM"]["classification_status"] == "no_sector_match"
    assert reg["MYST"]["classification_status"] == "no_sic"
    assert reg["KNWN"]["classification_status"] is None  # untouched, already classified


def test_classify_rerun_skips_attempted_names(intake_env) -> None:
    cfg = intake_env
    first = classify_new_registrants(
        as_of_date="2026-06-11", db_path=cfg.db_path, cfg=cfg, submissions_loader=_loader
    )
    second = classify_new_registrants(
        as_of_date="2026-06-11", db_path=cfg.db_path, cfg=cfg, submissions_loader=_loader
    )
    assert first["scope"] == 3
    # MAPS got a real sector; FARM/MYST recorded NULL-sector attempt rows and
    # stay in scope for retry (e.g. after a sic-ranges extension) but skip via
    # the classifier's same-day dedupe instead of refetching.
    assert second["scope"] == 2
    assert second["classifier_summary"]["skipped_existing"] == 2
    assert second["status_counts"] == {"no_sector_match": 1, "no_sic": 1}
