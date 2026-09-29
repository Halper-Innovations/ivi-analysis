from __future__ import annotations

import pytest
import requests

from app.config import get_config
from app.db import get_db, init_db
from app.events.poller import apply_expiry, backfill, poll_day
from app.events import store


DAILY_FIXTURE = b"""Description:           Daily Index of EDGAR Dissemination Feed
Last Data Received:     May 5, 2025

CIK|Company Name|Form Type|Date Filed|File Name
--------------------------------------------------------------------------------
2041385|Ralliant Corp|10-12B|20250505|edgar/data/2041385/0001104659-25-044355.txt
2041385|Ralliant Corp|10-12B|20250505|edgar/data/2041385/0001104659-25-044355.txt
320193|Apple Inc.|8-K|20250505|edgar/data/320193/0000320193-25-000055.txt
"""

QUARTERLY_FIXTURE = b"""Description:           Master Index of EDGAR Dissemination Feed

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
2041385|Ralliant Corp|10-12B|2025-05-05|edgar/data/2041385/0001104659-25-044355.txt
320193|Apple Inc.|8-K|2025-05-06|edgar/data/320193/0000320193-25-000056.txt
"""


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


class FakeClient:
    def __init__(self, payload=DAILY_FIXTURE, error=None):
        self.payload = payload
        self.error = error
        self.calls: list[str] = []

    def download_bytes(self, url, *, use_cache=True):
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return self.payload


class FakeReader:
    def items_for(self, cik, accession):
        return "2.02"

    def prior_item103(self, cik, *, before):
        return None

    def prior_registration(self, cik, *, before):
        return None

    def has_prior_periodic_filings(self, cik, *, before):
        return True


def _poll(client, scan_date="2025-05-05", **kwargs):
    reader = FakeReader()
    defaults = dict(
        client=client, submissions=reader, history=reader, registrations=reader,
        mapping={}, watchlist_ciks={}, today_et="2025-05-06",
    )
    defaults.update(kwargs)
    return poll_day(scan_date, **defaults)


def test_poll_day_ok(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    client = FakeClient()
    summary = _poll(client)
    assert summary.status == "OK"
    assert summary.index_rows == 2  # duplicate row deduped by the parser
    assert summary.candidate_rows == 2
    assert summary.events_created == 1
    with get_db() as conn:
        scan = conn.execute("SELECT * FROM corporate_event_scans").fetchone()
        assert scan["status"] == "OK"
        assert scan["index_rows"] == 2
        assert scan["candidate_rows"] == 2


def test_poll_day_idempotent(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    client = FakeClient()
    _poll(client)
    summary = _poll(client)
    assert summary.events_created == 0
    with get_db() as conn:
        n = conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"]
        assert n == 1


def test_current_day_guard_before_any_fetch(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    client = FakeClient()
    with pytest.raises(ValueError):
        _poll(client, scan_date="2025-05-06")
    with pytest.raises(ValueError):
        _poll(client, scan_date="2025-05-07")
    assert client.calls == []
    with get_db() as conn:
        n = conn.execute("SELECT COUNT(*) c FROM corporate_event_scans").fetchone()["c"]
        assert n == 0


def _http_error(status_code):
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(response=response)


def test_poll_day_404_records_no_index(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    summary = _poll(FakeClient(error=_http_error(404)))
    assert summary.status == "NO_INDEX"
    with get_db() as conn:
        scan = conn.execute("SELECT * FROM corporate_event_scans").fetchone()
        assert scan["status"] == "NO_INDEX"


def test_poll_day_500_records_error(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    summary = _poll(FakeClient(error=_http_error(500)))
    assert summary.status == "ERROR"
    with get_db() as conn:
        scan = conn.execute("SELECT * FROM corporate_event_scans").fetchone()
        assert scan["status"] == "ERROR"


def test_backfill_resumes_off_scan_rows(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        store.record_scan(conn, scan_date="2025-05-05", mode="backfill", status="OK", counters={})
    reader = FakeReader()
    client = FakeClient(payload=QUARTERLY_FIXTURE)
    summaries = backfill(
        "2025-05-01", "2025-05-31", client=client, today_et="2025-06-01",
        watchlist_ciks={}, mapping={},
    )
    assert [s.scan_date for s in summaries] == ["2025-05-06"]
    summaries_forced = backfill(
        "2025-05-01", "2025-05-31", client=client, force=True, today_et="2025-06-01",
        watchlist_ciks={}, mapping={},
    )
    assert [s.scan_date for s in summaries_forced] == ["2025-05-05", "2025-05-06"]


def test_apply_expiry_literal_boundaries(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        store.upsert_event(
            conn, cik="0000123456", event_type="spinoff",
            anchor_accession="0001234567-24-000001", company_name="SPINCO",
            detection_date="2024-01-02", source_mode="backfill",
        )
        assert apply_expiry(conn, scan_date="2024-09-27") == 0
        assert apply_expiry(conn, scan_date="2024-09-29") == 1
        row = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert row["status"] == "EXPIRED"
        assert row["expiry_reason"] == "NO_EFFECTIVENESS_270D"


def test_queue_protection_events_never_expire(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        store.upsert_event(
            conn, cik="0001692427", event_type="merger",
            anchor_accession="0001104659-26-061001", company_name="NCS MULTISTAGE",
            detection_date="2020-01-02", source_mode="daily",
        )
        assert apply_expiry(conn, scan_date="2026-06-10") == 0
        row = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert row["status"] == "DETECTED"
