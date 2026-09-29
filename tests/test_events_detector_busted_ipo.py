from __future__ import annotations

import json
from types import SimpleNamespace

from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.busted_ipo_qualifier import qualify_busted_ipos
from app.events.detectors import detect_busted_ipos
from app.events.index_feed import IndexRow


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def _row(cik, form_type, date_filed, accession, name="NewCo, Inc."):
    return IndexRow(
        cik=cik, company_name=name, form_type=form_type, date_filed=date_filed,
        file_name=f"edgar/data/{int(cik)}/{accession}.txt",
    )


class FakeHistory:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def has_prior_periodic_filings(self, cik, *, before):
        return self.mapping.get(cik, False)


def test_fresh_s1_creates_event(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        counters = detect_busted_ipos(
            conn, [_row("0000444444", "S-1", "2024-02-20", "0000444444-24-000001")],
            scan_date="2024-02-20", source_mode="daily", history=FakeHistory(),
        )
        assert counters.events_created == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["event_type"] == "busted_ipo"
        assert event["status"] == "DETECTED"


def test_reporting_company_s1_skipped(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        detect_busted_ipos(
            conn, [_row("0000444444", "S-1", "2024-02-20", "0000444444-24-000001")],
            scan_date="2024-02-20", source_mode="daily",
            history=FakeHistory({"0000444444": True}),
        )
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 0
        skip = conn.execute("SELECT * FROM corporate_event_skips").fetchone()
        assert skip["reason_code"] == "PRIOR_REPORTING_S1"


def test_s1_history_fetch_failure_skips(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    class FailingHistory:
        def has_prior_periodic_filings(self, cik, *, before):
            return None

    with get_db() as conn:
        detect_busted_ipos(
            conn, [_row("0000444444", "S-1", "2024-02-20", "0000444444-24-000001")],
            scan_date="2024-02-20", source_mode="daily", history=FailingHistory(),
        )
        skip = conn.execute("SELECT * FROM corporate_event_skips").fetchone()
        assert skip["reason_code"] == "SUBMISSIONS_FETCH_FAILED"


def test_s1_then_424b4_sets_priced_date(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    history = FakeHistory()
    with get_db() as conn:
        detect_busted_ipos(
            conn, [_row("0000444444", "S-1", "2024-02-20", "0000444444-24-000001")],
            scan_date="2024-02-20", source_mode="daily", history=history,
        )
        detect_busted_ipos(
            conn, [_row("0000444444", "424B4", "2024-03-15", "0000444444-24-000002")],
            scan_date="2024-03-15", source_mode="daily", history=history,
        )
        events = conn.execute("SELECT * FROM corporate_events").fetchall()
        assert len(events) == 1
        assert events[0]["status"] == "DETECTED"
        assert json.loads(events[0]["detail_json"])["priced_date"] == "2024-03-15"


def test_orphan_424b4_without_history_creates_event(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        detect_busted_ipos(
            conn, [_row("0000555555", "424B4", "2024-03-15", "0000555555-24-000009")],
            scan_date="2024-03-15", source_mode="backfill", history=FakeHistory(),
        )
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["anchor_accession"] == "0000555555-24-000009"
        assert json.loads(event["detail_json"])["priced_date"] == "2024-03-15"


def test_orphan_424b4_with_history_is_followon(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        detect_busted_ipos(
            conn, [_row("0000555555", "424B4", "2024-03-15", "0000555555-24-000009")],
            scan_date="2024-03-15", source_mode="daily",
            history=FakeHistory({"0000555555": True}),
        )
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 0
        skip = conn.execute("SELECT * FROM corporate_event_skips").fetchone()
        assert skip["reason_code"] == "FOLLOWON_424B4"


class FakeProvider:
    """get_price_asof keyed by exact as-of date string; None when absent."""

    def __init__(self, closes: dict[str, float]):
        self.closes = closes
        self.calls: list[tuple[str, str]] = []

    def get_price_asof(self, ticker, as_of_date):
        self.calls.append((ticker, as_of_date))
        price = self.closes.get(as_of_date)
        if price is None:
            return None
        return SimpleNamespace(price=price)


def _seed_priced_event(conn, *, ticker: str | None = "NEWC") -> int:
    history = FakeHistory()
    detect_busted_ipos(
        conn, [
            _row("0000444444", "S-1", "2024-02-20", "0000444444-24-000001"),
            _row("0000444444", "424B4", "2024-03-15", "0000444444-24-000002"),
        ],
        scan_date="2024-03-15", source_mode="daily", history=history,
    )
    event_id = int(conn.execute("SELECT id FROM corporate_events").fetchone()["id"])
    if ticker:
        store.set_ticker(conn, event_id=event_id, ticker=ticker)
    return event_id


def test_qualifier_qualifies_on_price_decline(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_priced_event(conn)
        # Priced 2024-03-15 -> seasoned 2024-06-13 (90d). $10 -> $4.50 = -55%.
        provider = FakeProvider({"2024-03-15": 10.0, "2024-06-13": 4.5})

        result = qualify_busted_ipos(conn, as_of="2024-09-01", provider=provider)

        assert result == {"candidates": 1, "qualified": 1, "expired_not_met": 0, "skipped": 0}
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["status"] == "QUALIFIED"
        assert event["qualification_date"] == "2024-09-01"
        detail = json.loads(event["detail_json"])
        assert detail["baseline_price"] == 10.0
        assert detail["baseline_date"] == "2024-03-15"
        assert detail["seasoned_price"] == 4.5
        assert detail["decline_pct"] == 0.55

        # Re-run: the event is no longer DETECTED, nothing to do.
        rerun = qualify_busted_ipos(conn, as_of="2024-09-02", provider=provider)
        assert rerun == {"candidates": 0, "qualified": 0, "expired_not_met": 0, "skipped": 0}


def test_qualifier_expires_terminally_when_threshold_not_met(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_priced_event(conn)
        # $10 -> $6 = -40% < 50% threshold: terminal, never re-polled.
        provider = FakeProvider({"2024-03-15": 10.0, "2024-06-13": 6.0})

        result = qualify_busted_ipos(conn, as_of="2024-09-01", provider=provider)

        assert result == {"candidates": 1, "qualified": 0, "expired_not_met": 1, "skipped": 0}
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["status"] == "EXPIRED"
        assert event["expiry_reason"] == "DECLINE_THRESHOLD_NOT_MET"
        assert json.loads(event["detail_json"])["decline_pct"] == 0.4


def test_qualifier_baseline_falls_forward_to_first_close(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_priced_event(conn)
        # No close on pricing day; debut print 5 days later. $9 -> $4 = -55.6%.
        provider = FakeProvider({"2024-03-20": 9.0, "2024-06-13": 4.0})

        result = qualify_busted_ipos(conn, as_of="2024-09-01", provider=provider)

        assert result["qualified"] == 1
        detail = json.loads(
            conn.execute("SELECT detail_json FROM corporate_events").fetchone()["detail_json"]
        )
        assert detail["baseline_date"] == "2024-03-20"
        assert detail["baseline_price"] == 9.0


def test_qualifier_transient_skips_keep_event_detected(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_priced_event(conn, ticker=None)
        # Ticker unresolved: skipped before any provider call.
        result = qualify_busted_ipos(conn, as_of="2024-09-01", provider=FakeProvider({}))
        assert result == {"candidates": 1, "qualified": 0, "expired_not_met": 0, "skipped": 1}
        skip = conn.execute(
            "SELECT * FROM corporate_event_skips WHERE detector='busted_ipo_qualifier'"
        ).fetchone()
        assert skip["reason_code"] == "TICKER_UNRESOLVED"

        # Resolve the ticker but query before seasoning: still DETECTED.
        event_id = int(conn.execute("SELECT id FROM corporate_events").fetchone()["id"])
        store.set_ticker(conn, event_id=event_id, ticker="NEWC")
        unseasoned = FakeProvider({})
        result2 = qualify_busted_ipos(conn, as_of="2024-04-01", provider=unseasoned)
        assert result2["skipped"] == 1
        assert unseasoned.calls == []
        assert conn.execute("SELECT status FROM corporate_events").fetchone()["status"] == "DETECTED"

        # Seasoned but no price history: transient skip, still DETECTED.
        result3 = qualify_busted_ipos(conn, as_of="2024-09-01", provider=FakeProvider({}))
        assert result3["skipped"] == 1
        assert conn.execute("SELECT status FROM corporate_events").fetchone()["status"] == "DETECTED"


def test_rerun_zero_deltas(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    history = FakeHistory()
    rows = [
        _row("0000444444", "S-1", "2024-02-20", "0000444444-24-000001"),
        _row("0000444444", "424B4", "2024-03-15", "0000444444-24-000002"),
    ]
    with get_db() as conn:
        detect_busted_ipos(conn, rows, scan_date="2024-03-15", source_mode="daily", history=history)
        counters = detect_busted_ipos(
            conn, rows, scan_date="2024-03-15", source_mode="daily", history=history,
        )
        assert counters.events_created == 0
        assert counters.events_updated == 0
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 1
