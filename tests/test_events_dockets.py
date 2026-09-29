"""Litigation-docket gate: recent CourtListener suits against queue names open
litigation_docket queue-protection events with synthetic CL-<docket_id> anchors."""

from __future__ import annotations

from app.config import get_config
from app.db import get_db, init_db
from app.events.courtlistener import (
    DocketHit,
    case_name_matches,
    litigation_targets,
    nos_code_from_text,
)
from app.events.detectors import detect_litigation_dockets
from app.events.flags import sync_event_pending_flags
from app.watchlist.schema import ensure_watchlist_schema


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


class FakeSearcher:
    def __init__(self, by_name=None, fail_names=()):
        self.by_name = by_name or {}
        self.fail_names = set(fail_names)
        self.queries = []

    def search_recent_dockets(self, company_name, *, filed_after):
        self.queries.append((company_name, filed_after))
        if company_name in self.fail_names:
            return None
        return self.by_name.get(company_name, [])


def _hit(docket_id="4567890", case_name="Smith v. NCS Multistage Holdings, Inc.",
         nature_of_suit="850 Securities/Commodities/Exchange", date_filed="2026-06-20"):
    return DocketHit(
        docket_id=docket_id, case_name=case_name, court="txsd",
        date_filed=date_filed, nature_of_suit=nature_of_suit,
        nos_code=nos_code_from_text(nature_of_suit),
        url=f"https://www.courtlistener.com/docket/{docket_id}/",
    )


TARGETS = [{
    "cik": "0001692427", "ticker": "NCSM",
    "company_name": "NCS Multistage Holdings, Inc.",
}]
NOS = {850, 370, 410}


def test_securities_suit_opens_event_and_blocks(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    searcher = FakeSearcher(by_name={TARGETS[0]["company_name"]: [_hit()]})
    with get_db() as conn:
        conn.execute(
            "INSERT INTO watchlist(ticker, status, source_run_id, added_at) "
            "VALUES('NCSM', 'DEPLOY_READY', 'run_NCSM', '2026-06-01T00:00:00Z')"
        )
        counters = detect_litigation_dockets(
            conn, TARGETS, scan_date="2026-07-01", lookback_days=30,
            nos_codes=NOS, source_mode="daily", searcher=searcher,
        )
        assert counters.events_created == 1
        event = conn.execute("SELECT * FROM corporate_events").fetchone()
        assert event["event_type"] == "litigation_docket"
        assert event["anchor_accession"] == "CL-4567890"
        assert event["ticker"] == "NCSM"
        assert event["detection_date"] == "2026-06-20"
        sync_event_pending_flags(conn)
        row = conn.execute("SELECT event_pending FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["event_pending"] == "EVENT_PENDING:LITIGATION_DOCKET"


def test_nos_outside_whitelist_filtered(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    slip_and_fall = _hit(nature_of_suit="360 P.I.: Other")
    searcher = FakeSearcher(by_name={TARGETS[0]["company_name"]: [slip_and_fall]})
    with get_db() as conn:
        counters = detect_litigation_dockets(
            conn, TARGETS, scan_date="2026-07-01", lookback_days=30,
            nos_codes=NOS, source_mode="daily", searcher=searcher,
        )
        assert counters.events_created == 0
        assert counters.counts == {"n_docket_filtered_nos": 1}


def test_caption_name_mismatch_filtered(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    wrong_company = _hit(case_name="Jones v. NCS Semiconductor Corp")
    searcher = FakeSearcher(by_name={TARGETS[0]["company_name"]: [wrong_company]})
    with get_db() as conn:
        counters = detect_litigation_dockets(
            conn, TARGETS, scan_date="2026-07-01", lookback_days=30,
            nos_codes=NOS, source_mode="daily", searcher=searcher,
        )
        assert counters.events_created == 0
        assert counters.counts == {"n_docket_filtered_name": 1}


def test_generic_short_name_never_searched(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    searcher = FakeSearcher()
    targets = [{"cik": "0000012345", "ticker": "IO", "company_name": "IO Inc"}]
    with get_db() as conn:
        counters = detect_litigation_dockets(
            conn, targets, scan_date="2026-07-01", lookback_days=30,
            nos_codes=NOS, source_mode="daily", searcher=searcher,
        )
        assert counters.events_created == 0
        assert counters.counts == {"n_docket_name_too_generic": 1}
        assert searcher.queries == []


def test_rerun_is_idempotent(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    searcher = FakeSearcher(by_name={TARGETS[0]["company_name"]: [_hit()]})
    with get_db() as conn:
        for _ in range(2):
            counters = detect_litigation_dockets(
                conn, TARGETS, scan_date="2026-07-01", lookback_days=30,
                nos_codes=NOS, source_mode="daily", searcher=searcher,
            )
        assert counters.events_created == 0
        n = conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"]
        assert n == 1


def test_fetch_failure_counted_not_fatal(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    searcher = FakeSearcher(fail_names={TARGETS[0]["company_name"]})
    with get_db() as conn:
        counters = detect_litigation_dockets(
            conn, TARGETS, scan_date="2026-07-01", lookback_days=30,
            nos_codes=NOS, source_mode="daily", searcher=searcher,
        )
        assert counters.events_created == 0
        assert counters.counts == {"n_docket_fetch_failed": 1}


def test_case_name_matching_rules():
    assert case_name_matches("ACME Semiconductor Corp", "Doe v. Acme Semiconductor Corp.")
    assert not case_name_matches("ACME Semiconductor Corp", "Doe v. Acme Inc")
    # Suffix-stripped full-token match: "Holdings" belongs to the core name.
    assert case_name_matches(
        "NCS Multistage Holdings, Inc.", "Smith v. NCS MULTISTAGE HOLDINGS INC"
    )
    assert not case_name_matches("IO Inc", "Smith v. IO Inc")  # too short/generic


def test_nos_code_from_text():
    assert nos_code_from_text("850 Securities/Commodities/Exchange") == 850
    assert nos_code_from_text("370 Other Fraud") == 370
    assert nos_code_from_text("Securities") is None
    assert nos_code_from_text(None) is None
    assert nos_code_from_text("") is None


def test_litigation_targets_scope(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        for ticker, status, grade in (
            ("NCSM", "DEPLOY_READY", "ACTIONABLE"),
            ("BOOM", "ACTIVE", "ACTIONABLE"),
            ("QUIE", "ACTIVE", "WATCHLIST_ONLY"),
            ("AVDT", "DEPLOY_READY", "AVOID"),
        ):
            conn.execute(
                "INSERT INTO watchlist(ticker, status, conviction_grade, source_run_id, added_at) "
                "VALUES(?, ?, ?, ?, '2026-06-01T00:00:00Z')",
                (ticker, status, grade, f"run_{ticker}"),
            )
        conn.execute(
            "INSERT INTO sec_registrants(cik, primary_ticker, name, exchange_scope, operating_status, first_seen_at, last_seen_at) "
            "VALUES('0001692427', 'NCSM', 'NCS Multistage Holdings, Inc.', 'ALL', 'OPERATING', "
            "'2026-06-01T00:00:00Z', '2026-06-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO sec_registrants(cik, primary_ticker, name, exchange_scope, operating_status, first_seen_at, last_seen_at) "
            "VALUES('0000034067', 'BOOM', 'DMC Global Inc.', 'ALL', 'OPERATING', "
            "'2026-06-01T00:00:00Z', '2026-06-01T00:00:00Z')"
        )
        targets = litigation_targets(conn)
        assert [t["ticker"] for t in targets] == ["BOOM", "NCSM"]
        assert targets[1] == {
            "cik": "0001692427",
            "ticker": "NCSM",
            "company_name": "NCS Multistage Holdings, Inc.",
        }
