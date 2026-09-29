from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.validation import coverage_report, load_ground_truth


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def test_load_ground_truth(tmp_path):
    csv_path = tmp_path / "truth.csv"
    csv_path.write_text(
        "event_type,cik,ticker,company_name,knowable_date,notes\n"
        "spinoff,2041385,RAL,Ralliant Corp,2025-05-05,fortive spinoff\n"
    )
    rows = load_ground_truth(csv_path)
    assert rows == [{
        "event_type": "spinoff",
        "cik": "0002041385",
        "ticker": "RAL",
        "company_name": "Ralliant Corp",
        "knowable_date": "2025-05-05",
        "notes": "fortive spinoff",
    }]


def test_coverage_report_boundaries(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    truth = [
        {"event_type": "spinoff", "cik": "0000000001", "ticker": "AAA",
         "company_name": "A", "knowable_date": "2024-03-01", "notes": ""},
        {"event_type": "spinoff", "cik": "0000000002", "ticker": "BBB",
         "company_name": "B", "knowable_date": "2024-03-01", "notes": ""},
        {"event_type": "spinoff", "cik": "0000000003", "ticker": "CCC",
         "company_name": "C", "knowable_date": "2024-03-01", "notes": ""},
        {"event_type": "ch11_emergence", "cik": "0000000004", "ticker": "DDD",
         "company_name": "D", "knowable_date": "2024-03-01", "notes": ""},
    ]
    with get_db() as conn:
        store.upsert_event(  # exact-day hit
            conn, cik="0000000001", event_type="spinoff",
            anchor_accession="0000000001-24-000001", company_name="A",
            detection_date="2024-03-01", source_mode="backfill",
        )
        store.upsert_event(  # +7-day boundary hit
            conn, cik="0000000002", event_type="spinoff",
            anchor_accession="0000000002-24-000001", company_name="B",
            detection_date="2024-03-08", source_mode="backfill",
        )
        store.upsert_event(  # late -> miss
            conn, cik="0000000003", event_type="spinoff",
            anchor_accession="0000000003-24-000001", company_name="C",
            detection_date="2024-03-09", source_mode="backfill",
        )
        # cik 0000000004 absent -> NOT_DETECTED
        report = coverage_report(conn, truth)
    assert report["total"] == 4
    assert report["hits"] == 2
    assert report["coverage"] == 0.5
    assert report["by_type"] == {
        "spinoff": {"total": 3, "hits": 2},
        "ch11_emergence": {"total": 1, "hits": 0},
    }
    assert report["misses"] == [
        {"cik": "0000000003", "event_type": "spinoff", "reason": "DETECTED_LATE"},
        {"cik": "0000000004", "event_type": "ch11_emergence", "reason": "NOT_DETECTED"},
    ]


def test_coverage_wrong_type(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    truth = [{"event_type": "spinoff", "cik": "0000000005", "ticker": "EEE",
              "company_name": "E", "knowable_date": "2024-03-01", "notes": ""}]
    with get_db() as conn:
        store.upsert_event(
            conn, cik="0000000005", event_type="busted_ipo",
            anchor_accession="0000000005-24-000001", company_name="E",
            detection_date="2024-03-01", source_mode="backfill",
        )
        report = coverage_report(conn, truth)
    assert report["misses"] == [
        {"cik": "0000000005", "event_type": "spinoff", "reason": "WRONG_TYPE"},
    ]
