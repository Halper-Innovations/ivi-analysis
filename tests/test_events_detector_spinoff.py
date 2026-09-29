from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db
from app.events.detectors import apply_spinoff_auto_effect, detect_spinoffs
from app.events.index_feed import IndexRow


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def _row(cik, form_type, date_filed, accession, name="Ralliant Corp"):
    return IndexRow(
        cik=cik, company_name=name, form_type=form_type, date_filed=date_filed,
        file_name=f"edgar/data/{int(cik)}/{accession}.txt",
    )


class FakeRegistrations:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = []

    def prior_registration(self, cik, *, before):
        self.calls.append((cik, before))
        return self.mapping.get(cik)


def test_initial_10_12b_creates_detected_event(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [_row("0002041385", "10-12B", "2025-05-05", "0001104659-25-044355")]
    with get_db() as conn:
        counters = detect_spinoffs(
            conn, rows, scan_date="2025-05-05", source_mode="daily",
            registrations=FakeRegistrations(),
        )
        assert counters.events_created == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["status"] == "DETECTED"
        assert event["detection_date"] == "2025-05-05"
        assert event["anchor_accession"] == "0001104659-25-044355"

        # Idempotent re-run.
        counters2 = detect_spinoffs(
            conn, rows, scan_date="2025-05-05", source_mode="daily",
            registrations=FakeRegistrations(),
        )
        assert counters2.events_created == 0
        assert counters2.events_updated == 0
        n = conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"]
        assert n == 1


def test_registration_amendment_cert_chain(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    regs = FakeRegistrations()
    with get_db() as conn:
        detect_spinoffs(
            conn, [_row("0002041385", "10-12B", "2025-05-05", "0001104659-25-044355")],
            scan_date="2025-05-05", source_mode="daily", registrations=regs,
        )
        detect_spinoffs(
            conn, [_row("0002041385", "10-12B/A", "2025-05-20", "0001104659-25-050000")],
            scan_date="2025-05-20", source_mode="daily", registrations=regs,
        )
        detect_spinoffs(
            conn, [_row("0002041385", "CERT", "2025-06-25", "0001104659-25-060000")],
            scan_date="2025-06-25", source_mode="daily", registrations=regs,
        )
        events = conn.execute("SELECT * FROM corporate_events").fetchall()
        assert len(events) == 1
        assert events[0]["status"] == "QUALIFIED"
        assert events[0]["qualification_date"] == "2025-06-25"
        roles = [
            r["role"] for r in conn.execute(
                "SELECT role FROM corporate_event_filings ORDER BY id"
            ).fetchall()
        ]
        assert roles == ["REGISTRATION", "AMENDMENT", "CERTIFICATION"]


def test_same_day_out_of_order_pairing(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [
        _row("0002041385", "CERT", "2025-06-02", "0001104659-25-060001"),
        _row("0002041385", "10-12B", "2025-06-02", "0001104659-25-060002"),
    ]
    with get_db() as conn:
        detect_spinoffs(
            conn, rows, scan_date="2025-06-02", source_mode="daily",
            registrations=FakeRegistrations(),
        )
        events = conn.execute("SELECT * FROM corporate_events").fetchall()
        assert len(events) == 1
        assert events[0]["status"] == "QUALIFIED"


def test_cert_backstop_hit(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    regs = FakeRegistrations({"0002041385": ("0001104659-23-099999", "2023-11-20")})
    with get_db() as conn:
        detect_spinoffs(
            conn, [_row("0002041385", "CERT", "2024-02-01", "0001104659-24-010000")],
            scan_date="2024-02-01", source_mode="backfill", registrations=regs,
        )
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["anchor_accession"] == "0001104659-23-099999"
        assert event["detection_date"] == "2024-02-01"
        assert event["status"] == "QUALIFIED"
        assert event["qualification_date"] == "2024-02-01"


def test_cert_backstop_miss_counts_not_skips(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        counters = detect_spinoffs(
            conn, [_row("0002041385", "CERT", "2024-02-01", "0001104659-24-010000")],
            scan_date="2024-02-01", source_mode="backfill",
            registrations=FakeRegistrations(),
        )
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM corporate_event_skips").fetchone()["c"] == 0
        assert counters.counts.get("n_cert_backstop_miss") == 1


def test_unmatched_effect_ignored(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        detect_spinoffs(
            conn, [_row("0002041385", "EFFECT", "2024-02-01", "9999999995-24-000111")],
            scan_date="2024-02-01", source_mode="daily",
            registrations=FakeRegistrations(),
        )
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM corporate_event_skips").fetchone()["c"] == 0


def test_10_12g_auto_effectiveness_at_60_days(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        detect_spinoffs(
            conn, [_row("0002041385", "10-12G", "2025-05-05", "0001104659-25-044356")],
            scan_date="2025-05-05", source_mode="daily",
            registrations=FakeRegistrations(),
        )
        # Too early: +59 days.
        assert apply_spinoff_auto_effect(conn, scan_date="2025-07-03") == 0
        n = apply_spinoff_auto_effect(conn, scan_date="2025-07-05")
        assert n == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["status"] == "QUALIFIED"
        assert event["qualification_date"] == "2025-07-04"  # exactly +60 days
