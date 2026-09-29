from __future__ import annotations

import json

from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.detectors import detect_ch11, normalize_company_name
from app.events.index_feed import IndexRow


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()


def _row(cik, form_type, date_filed, accession, name="Acme Holdings, Inc."):
    return IndexRow(
        cik=cik, company_name=name, form_type=form_type, date_filed=date_filed,
        file_name=f"edgar/data/{int(cik)}/{accession}.txt",
    )


class FakeSubmissions:
    def __init__(self, items=None, prior103=None):
        self.items = items or {}
        self.prior103 = prior103 or {}

    def items_for(self, cik, accession):
        return self.items.get((cik, accession))

    def prior_item103(self, cik, *, before):
        return self.prior103.get(cik)


def test_normalize_company_name():
    assert normalize_company_name("Acme Holdings, Inc.") == "ACME HOLDINGS"
    assert normalize_company_name("ACME HOLDINGS CORP") == "ACME HOLDINGS"
    assert normalize_company_name("Acme Holdings Corp New") == "ACME HOLDINGS"


def test_8k_item_103_creates_event_others_counted(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={
        ("0000111111", "0000111111-24-000010"): "1.03,9.01",
        ("0000222222", "0000222222-24-000011"): "2.02",
    })
    rows = [
        _row("0000111111", "8-K", "2024-01-10", "0000111111-24-000010"),
        _row("0000222222", "8-K", "2024-01-10", "0000222222-24-000011"),
    ]
    with get_db() as conn:
        counters = detect_ch11(
            conn, rows, scan_date="2024-01-10", source_mode="daily", submissions=subs,
        )
        assert counters.events_created == 1
        assert counters.counts["n_8k_scanned"] == 2
        assert counters.counts["n_8k_item103"] == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["event_type"] == "ch11_emergence"
        assert event["detection_date"] == "2024-01-10"
        assert conn.execute("SELECT COUNT(*) c FROM corporate_event_skips").fetchone()["c"] == 0


def test_8k_fetch_failure_records_skip(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [_row("0000111111", "8-K", "2024-01-10", "0000111111-24-000010")]
    with get_db() as conn:
        detect_ch11(
            conn, rows, scan_date="2024-01-10", source_mode="daily",
            submissions=FakeSubmissions(),
        )
        skip = conn.execute("SELECT * FROM corporate_event_skips").fetchone()
        assert skip["reason_code"] == "SUBMISSIONS_FETCH_FAILED"


def test_25nse_unlinked_is_counter_not_skip(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    rows = [_row("0000333333", "25-NSE", "2024-02-01", "0001143313-24-000023")]
    with get_db() as conn:
        counters = detect_ch11(
            conn, rows, scan_date="2024-02-01", source_mode="daily",
            submissions=FakeSubmissions(),
        )
        assert counters.counts == {"n_25nse_unlinked": 1}
        assert conn.execute("SELECT COUNT(*) c FROM corporate_event_skips").fetchone()["c"] == 0


def test_25nse_attaches_when_event_open(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={("0000111111", "0000111111-24-000010"): "1.03"})
    with get_db() as conn:
        detect_ch11(
            conn, [_row("0000111111", "8-K", "2024-01-10", "0000111111-24-000010")],
            scan_date="2024-01-10", source_mode="daily", submissions=subs,
        )
        counters = detect_ch11(
            conn, [_row("0000111111", "25-NSE", "2024-02-01", "0001143313-24-000023")],
            scan_date="2024-02-01", source_mode="daily", submissions=subs,
        )
        assert counters.counts.get("n_25nse_unlinked") is None
        role = conn.execute(
            "SELECT role FROM corporate_event_filings ORDER BY id DESC LIMIT 1"
        ).fetchone()["role"]
        assert role == "DELISTING"


def test_same_cik_8a12b_qualifies(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={("0000111111", "0000111111-24-000010"): "1.03,9.01"})
    with get_db() as conn:
        detect_ch11(
            conn, [_row("0000111111", "8-K", "2024-01-10", "0000111111-24-000010")],
            scan_date="2024-01-10", source_mode="daily", submissions=subs,
        )
        detect_ch11(
            conn, [_row("0000111111", "8-A12B", "2024-05-02", "0000111111-24-000050")],
            scan_date="2024-05-02", source_mode="daily", submissions=subs,
        )
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["status"] == "QUALIFIED"
        assert event["qualification_date"] == "2024-05-02"
        roles = [
            r["role"] for r in conn.execute(
                "SELECT role FROM corporate_event_filings ORDER BY id"
            ).fetchall()
        ]
        assert roles == ["BANKRUPTCY_8K", "RELISTING"]


def test_same_day_out_of_order_8a12b_then_8k(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={("0000111111", "0000111111-24-000010"): "1.03"})
    rows = [
        _row("0000111111", "8-A12B", "2024-01-10", "0000111111-24-000011"),
        _row("0000111111", "8-K", "2024-01-10", "0000111111-24-000010"),
    ]
    with get_db() as conn:
        detect_ch11(conn, rows, scan_date="2024-01-10", source_mode="daily", submissions=subs)
        events = conn.execute("SELECT * FROM corporate_events").fetchall()
        assert len(events) == 1
        assert events[0]["status"] == "QUALIFIED"


def test_new_cik_relisting_name_match(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={("0000111111", "0000111111-24-000010"): "1.03"})
    with get_db() as conn:
        detect_ch11(
            conn,
            [_row("0000111111", "8-K", "2024-01-10", "0000111111-24-000010",
                  name="Acme Holdings, Inc.")],
            scan_date="2024-01-10", source_mode="daily", submissions=subs,
        )
        event_id = conn.execute("SELECT id FROM corporate_events").fetchone()["id"]
        store.set_ticker(conn, event_id=event_id, ticker="OLDT")

        detect_ch11(
            conn,
            [_row("0000999999", "8-A12B", "2024-06-03", "0000999999-24-000001",
                  name="ACME HOLDINGS CORP")],
            scan_date="2024-06-03", source_mode="daily", submissions=subs,
        )
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["status"] == "QUALIFIED"
        detail = json.loads(event["detail_json"])
        assert detail["relisting_cik"] == "0000999999"
        assert event["ticker"] is None
        assert event["ticker_state"] == "UNKNOWN_TICKER"
        filing = conn.execute(
            "SELECT * FROM corporate_event_filings WHERE role='RELISTING'"
        ).fetchone()
        assert json.loads(filing["detail_json"])["cross_cik_link"] == "0000111111"


def test_1_03_backstop_hit(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(prior103={"0000333333": ("0000333333-23-000050", "2023-02-14")})
    with get_db() as conn:
        detect_ch11(
            conn,
            [_row("0000333333", "8-A12B", "2024-04-02", "0000333333-24-000001")],
            scan_date="2024-04-02", source_mode="backfill", submissions=subs,
        )
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["anchor_accession"] == "0000333333-23-000050"
        assert event["detection_date"] == "2024-04-02"
        assert event["status"] == "QUALIFIED"
        roles = [
            r["role"] for r in conn.execute(
                "SELECT role FROM corporate_event_filings ORDER BY id"
            ).fetchall()
        ]
        assert roles == ["BANKRUPTCY_8K", "RELISTING"]


def test_8a12b_amendment_never_creates(monkeypatch, tmp_path):
    """The GNK class: a rights-plan 8-A12B/A from a long-emerged company must
    not anchor a fresh emergence on its decade-old bankruptcy 8-K."""
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(prior103={"0001326200": ("0001140361-14-027824", "2014-07-07")})
    with get_db() as conn:
        counters = detect_ch11(
            conn,
            [_row("0001326200", "8-A12B/A", "2026-06-02", "0001140361-26-023628",
                  name="GENCO SHIPPING & TRADING LTD")],
            scan_date="2026-06-02", source_mode="daily", submissions=subs,
        )
        assert counters.events_created == 0
        assert counters.counts == {"n_8a12b_amendment_unlinked": 1}
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM corporate_event_skips").fetchone()["c"] == 0


def test_1_03_backstop_age_bound(monkeypatch, tmp_path):
    """An initial 8-A12B whose only 1.03 anchor is older than 3 years skips."""
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(prior103={"0000333333": ("0000333333-14-000050", "2014-07-07")})
    with get_db() as conn:
        detect_ch11(
            conn,
            [_row("0000333333", "8-A12B", "2026-06-02", "0000333333-26-000001")],
            scan_date="2026-06-02", source_mode="daily", submissions=subs,
        )
        assert conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"] == 0
        skip = conn.execute("SELECT * FROM corporate_event_skips").fetchone()
        assert skip["reason_code"] == "NO_CH11_LINK_8A"


def test_1_03_backstop_miss_skips(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        detect_ch11(
            conn,
            [_row("0000333333", "8-A12B", "2024-04-02", "0000333333-24-000001")],
            scan_date="2024-04-02", source_mode="daily", submissions=FakeSubmissions(),
        )
        skip = conn.execute("SELECT * FROM corporate_event_skips").fetchone()
        assert skip["reason_code"] == "NO_CH11_LINK_8A"


def test_rerun_zero_deltas(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    subs = FakeSubmissions(items={("0000111111", "0000111111-24-000010"): "1.03"})
    rows = [
        _row("0000111111", "8-K", "2024-01-10", "0000111111-24-000010"),
        _row("0000111111", "8-A12B", "2024-01-10", "0000111111-24-000011"),
    ]
    with get_db() as conn:
        detect_ch11(conn, rows, scan_date="2024-01-10", source_mode="daily", submissions=subs)
        before = {
            "events": conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"],
            "filings": conn.execute("SELECT COUNT(*) c FROM corporate_event_filings").fetchone()["c"],
            "skips": conn.execute("SELECT COUNT(*) c FROM corporate_event_skips").fetchone()["c"],
        }
        counters = detect_ch11(
            conn, rows, scan_date="2024-01-10", source_mode="daily", submissions=subs,
        )
        assert counters.events_created == 0
        assert counters.events_updated == 0
        after = {
            "events": conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"],
            "filings": conn.execute("SELECT COUNT(*) c FROM corporate_event_filings").fetchone()["c"],
            "skips": conn.execute("SELECT COUNT(*) c FROM corporate_event_skips").fetchone()["c"],
        }
        assert before == after
