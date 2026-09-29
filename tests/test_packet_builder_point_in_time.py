"""Legacy evidence-packet edge cases: the companyfacts overlay of
UNKNOWN filing-route values, and filings bounded to the packet date.
Hermetic: temp DB via ``init_db``, no network.
"""

from __future__ import annotations

import json

import app.evidence.packet_builder as pb
from app.db import get_db, init_db, utc_now_iso
from app.fundamentals.normalize import UNKNOWN

TICKER = "TEST"


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


# ── Edge cases in the packet builder ─────────────────────────


def test_overlay_fills_unknown_flat_keys_but_not_sector_limited_free_cash_flow(
    monkeypatch, tmp_path
):
    _init_temp_db(monkeypatch, tmp_path)
    frame = {
        "rows": [
            {
                "year": 2025,
                "revenue": 1000.0,
                "op_margin": 0.3,
                "fcf": 250.0,
                "fcf_margin": 0.25,
                "gross_margin": 0.5,
            }
        ],
        "row_traces": {},
        "derived_signals": {},
        "gaps": [],
    }
    monkeypatch.setattr(pb, "_companyfacts_frame", lambda conn, *, ticker, as_of_date: dict(frame))
    fundamentals = {
        "revenue": UNKNOWN,
        "operating_margin": UNKNOWN,
        "gross_margin": 0.44,  # a known filing value is never replaced
        "fcf": UNKNOWN,
        "fcf_margin": UNKNOWN,
        "fcf_applicability": "sector_limited",
    }
    with get_db() as conn:
        pb._overlay_companyfacts_trends(
            conn, ticker=TICKER, as_of_date="2026-06-30", fundamentals=fundamentals
        )
    assert fundamentals["revenue"] == 1000.0
    assert fundamentals["operating_margin"] == 0.3
    assert fundamentals["gross_margin"] == 0.44
    assert fundamentals["fcf"] == UNKNOWN
    assert fundamentals["fcf_margin"] == UNKNOWN


def test_legacy_packet_leaves_out_filings_with_no_filing_date(monkeypatch, tmp_path):
    """An undated filing cannot be shown to pre-date the packet, so it is not used."""
    _init_temp_db(monkeypatch, tmp_path)
    with get_db() as conn:
        now = utc_now_iso()
        conn.execute(
            "INSERT INTO fundamentals(ticker, as_of_date, metrics_json, quality_flags_json,"
            " created_at) VALUES(?, '2025-03-31', ?, '{}', ?)",
            (TICKER, json.dumps({"revenue": UNKNOWN}), now),
        )
        for accession, filed in (("0-25-1", "2025-02-15"), ("0-25-2", "2025-03-31"), ("0-25-3", None)):
            conn.execute(
                "INSERT INTO filings(cik, ticker, accession, form_type, filing_date,"
                " primary_doc_url, status, created_at, updated_at)"
                " VALUES('0', ?, ?, '10-K', ?, 'https://www.sec.gov/x', 'parsed', ?, ?)",
                (TICKER, accession, filed, now, now),
            )
        payload = pb._build_packet_payload(conn, TICKER, "2025-03-31")
    assert payload is not None
    assert [f["accession"] for f in payload["filings_used"]] == ["0-25-2", "0-25-1"]
