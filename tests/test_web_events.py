"""Events read model + /api/events: lanes, lifecycle columns, scan strip."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.db import init_db
from app.events import store
from app.web.main import app
from app.web.readmodel.events import events_deck, scan_strip

client = TestClient(app)

NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)  # a Wednesday


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _conn(cfg) -> sqlite3.Connection:
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_events(cfg) -> dict[str, int]:
    conn = _conn(cfg)
    ids: dict[str, int] = {}
    ids["spinoff"] = store.upsert_event(
        conn,
        cik="0000000001",
        event_type="spinoff",
        anchor_accession="0000000001-26-000001",
        company_name="SpinCo Parent",
        detection_date="2026-07-15",
        source_mode="daily",
        detail={"form": "10-12B"},
    )
    ids["busted_ipo"] = store.upsert_event(
        conn,
        cik="0000000002",
        event_type="busted_ipo",
        anchor_accession="0000000002-26-000001",
        company_name="Fresh Float Inc",
        detection_date="2026-07-01",
        source_mode="daily",
    )
    store.set_ticker(conn, event_id=ids["busted_ipo"], ticker="FFI")
    store.mark_qualified(conn, event_id=ids["busted_ipo"], qualification_date="2026-07-10")
    ids["merger"] = store.upsert_event(
        conn,
        cik="0000000003",
        event_type="merger",
        anchor_accession="0000000003-26-000001",
        company_name="Target Corp Industries",
        detection_date="2026-07-20",
        source_mode="daily",
    )
    store.set_ticker(conn, event_id=ids["merger"], ticker="TGT1")
    store.attach_filing(
        conn,
        event_id=ids["merger"],
        cik="0000000003",
        accession="0000000003-26-000001",
        form_type="425",
        filing_date="2026-07-20",
        role="MERGER_FILING",
    )
    ids["decided_8k"] = store.upsert_event(
        conn,
        cik="0000000004",
        event_type="unreviewed_8k",
        anchor_accession="0000000004-26-000001",
        company_name="Reviewed Co",
        detection_date="2026-07-01",
        source_mode="daily",
    )
    store.mark_decided(conn, event_id=ids["decided_8k"], note="reviewed, benign")
    ids["expired_ipo"] = store.upsert_event(
        conn,
        cik="0000000005",
        event_type="busted_ipo",
        anchor_accession="0000000005-26-000001",
        company_name="Missed Window Inc",
        detection_date="2026-06-01",
        source_mode="backfill",
    )
    store.mark_expired(conn, event_id=ids["expired_ipo"], expiry_reason="DECLINE_MISS")
    conn.commit()
    conn.close()
    return ids


def test_events_deck_lanes_columns_and_cards(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    ids = _seed_events(cfg)
    conn = _conn(cfg)
    deck = events_deck(conn, now=NOW)
    conn.close()

    opportunity, queue_protection = deck["lanes"]
    assert opportunity["lane"] == "opportunity"
    assert opportunity["total"] == 3
    assert opportunity["open_total"] == 2
    assert opportunity["type_counts"] == {"busted_ipo": 2, "spinoff": 1}
    by_status = {col["status"]: col for col in opportunity["columns"]}
    assert [col["status"] for col in opportunity["columns"]] == [
        "DETECTED",
        "QUALIFIED",
        "SURFACED",
        "DECIDED",
        "EXPIRED",
    ]
    assert by_status["DETECTED"]["total"] == 1
    assert by_status["QUALIFIED"]["total"] == 1
    assert by_status["EXPIRED"]["total"] == 1

    spin = by_status["DETECTED"]["cards"][0]
    assert spin["id"] == ids["spinoff"]
    assert spin["ticker"] is None
    assert spin["ticker_state"] == "UNKNOWN_TICKER"
    assert spin["age_days"] == 7
    assert spin["detail"] == {"form": "10-12B"}
    assert spin["dispose_command"] == (
        f'ivi events dispose {ids["spinoff"]} --note "<why>" '
        "--reason-code EVENT_REVIEWED|PASSED_EVENT_RISK|OTHER"
    )

    expired = by_status["EXPIRED"]["cards"][0]
    assert expired["expiry_reason"] == "DECLINE_MISS"
    assert expired["dispose_command"] is None

    assert queue_protection["lane"] == "queue_protection"
    assert queue_protection["total"] == 2
    assert queue_protection["open_total"] == 1
    merger = {c["status"]: c for c in queue_protection["columns"]}["DETECTED"]["cards"][0]
    assert merger["ticker"] == "TGT1"
    assert merger["age_days"] == 2
    assert merger["filings"] == [
        {
            "accession": "0000000003-26-000001",
            "form_type": "425",
            "filing_date": "2026-07-20",
            "role": "MERGER_FILING",
        }
    ]

    # spinoff (UNKNOWN) + expired busted IPO (never resolved) count as unknown.
    assert deck["unknown_ticker_total"] == 3


def test_events_deck_filters_and_column_cap(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_events(cfg)
    conn = _conn(cfg)

    only_opportunity = events_deck(conn, lane="opportunity", now=NOW)
    assert only_opportunity["lanes"][0]["total"] == 3
    assert only_opportunity["lanes"][1]["total"] == 0

    only_mergers = events_deck(conn, event_type="merger", now=NOW)
    assert only_mergers["lanes"][1]["type_counts"] == {"merger": 1}
    assert only_mergers["lanes"][0]["total"] == 0

    by_query = events_deck(conn, q="target corp", now=NOW)
    assert by_query["lanes"][1]["total"] == 1
    assert by_query["lanes"][0]["total"] == 0

    for i in range(3):
        store.upsert_event(
            conn,
            cik=f"000000010{i}",
            event_type="dilution",
            anchor_accession=f"000000010{i}-26-000001",
            company_name=f"Dilution Co {i}",
            detection_date="2026-07-18",
            source_mode="daily",
        )
    conn.commit()
    capped = events_deck(conn, event_type="dilution", per_column=2, now=NOW)
    detected = {c["status"]: c for c in capped["lanes"][1]["columns"]}["DETECTED"]
    assert detected["total"] == 3
    assert len(detected["cards"]) == 2
    conn.close()


def test_scan_strip_marks_business_day_gaps(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    conn = _conn(cfg)
    store.record_scan(
        conn,
        scan_date="2026-07-21",
        mode="daily",
        status="OK",
        counters={"index_rows": 3000, "candidate_rows": 200, "events_created": 5, "events_updated": 1},
    )
    store.record_scan(conn, scan_date="2026-07-20", mode="daily", status="FAILED", counters={})
    store.record_scan(
        conn,
        scan_date="2026-07-17",
        mode="daily",
        status="OK",
        counters={"index_rows": 2800, "candidate_rows": 190, "events_created": 0, "events_updated": 0},
    )
    conn.commit()

    strip = scan_strip(conn, business_days=5, now=NOW)
    conn.close()
    assert [(d["date"], d["status"]) for d in strip["days"]] == [
        ("2026-07-21", "OK"),
        ("2026-07-20", "FAILED"),
        ("2026-07-17", "OK"),
        ("2026-07-16", "MISSING"),
        ("2026-07-15", "MISSING"),
    ]
    assert strip["gap_count"] == 3
    assert strip["window_business_days"] == 5
    assert strip["days"][0]["index_rows"] == 3000
    assert strip["days"][0]["events_created"] == 5
    assert strip["days"][3]["index_rows"] is None


def test_api_events_endpoint(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_events(cfg)
    response = client.get("/api/events", params={"per_column": 5, "lane": "opportunity"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["lanes"][0]["total"] == 3
    assert payload["lanes"][1]["total"] == 0
    assert payload["scan_strip"]["window_business_days"] == 30
    assert payload["unknown_ticker_total"] == 2

    bad_lane = client.get("/api/events", params={"lane": "nope"})
    assert bad_lane.status_code == 422
