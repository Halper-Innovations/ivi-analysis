"""Web read-model fixes: rebased evolution chart, header price fallback, waterline price
basis, duplicate-row price path, and the negative-EPV gauge card.

Judgment calls (documented, conservative):
- A row written before a recorded split is divided by the recorded factor; when the
  factor is contradictory for the split date the point is dropped and counted.
- A watchlist row whose only price is the price at addition is left off the waterline
  strip rather than drawn as live.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.config import get_config
from app.db import init_db
from app.web.readmodel import company, today


@pytest.fixture()
def conn(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db(get_config())
    monkeypatch.setattr(company, "valuation_row_is_decision_eligible", lambda row, **k: True)
    c = sqlite3.connect(tmp_path / "data" / "engine.db")
    c.row_factory = sqlite3.Row
    yield c
    c.close()
    get_config.cache_clear()


def _val(conn, ticker, day, method, outputs, inputs=None, created=None):
    conn.execute(
        "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json, "
        "warnings_json, created_at) VALUES (?,?,?,?,?,?,?)",
        (ticker, day, method, json.dumps(inputs or {}), json.dumps(outputs), "[]",
         created or f"{day}T12:00:00+00:00"),
    )


def _split_quote(conn, ticker, effective, factor, day="2026-09-01", provider="p"):
    conn.execute(
        "INSERT INTO price_quotes (ticker, provider, as_of_date, price, status, fetched_at, "
        "expires_at, raw_json, quote_hash, split_adjustment_factor, split_effective_date) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (ticker, provider, day, 10.0, "OK", day, day, "{}", f"{ticker}{provider}{day}", factor,
         effective),
    )


def test_evolution_rebases_pre_split_values_to_the_current_basis(conn):
    _val(conn, "SPL", "2026-03-01", "epv", {"status": "OK", "value_per_share": 100.0})
    _val(conn, "SPL", "2026-08-01", "epv", {"status": "OK", "value_per_share": 55.0})
    _split_quote(conn, "SPL", "2026-06-01", 2.0)
    conn.commit()
    points, stats = company.valuation_evolution_with_drops(conn, "SPL")
    assert [(p["as_of_date"], p["value"]) for p in points] == [
        ("2026-03-01", 50.0),
        ("2026-08-01", 55.0),
    ]
    assert stats == {"dropped_pre_split": 0, "rebased": 1}


def test_evolution_drops_pre_split_points_when_the_factor_is_contradictory(conn):
    _val(conn, "SPL", "2026-03-01", "epv", {"status": "OK", "value_per_share": 100.0})
    _val(conn, "SPL", "2026-08-01", "epv", {"status": "OK", "value_per_share": 55.0})
    _split_quote(conn, "SPL", "2026-06-01", 2.0, provider="a")
    _split_quote(conn, "SPL", "2026-06-01", 3.0, provider="b")
    conn.commit()
    points, stats = company.valuation_evolution_with_drops(conn, "SPL")
    assert [p["value"] for p in points] == [55.0]
    assert stats == {"dropped_pre_split": 1, "rebased": 0}


def test_evolution_does_not_plot_a_negative_epv(conn):
    _val(conn, "NEG", "2026-08-01", "epv", {"status": "OK", "value_per_share": -4.0})
    conn.commit()
    assert company.valuation_evolution(conn, "NEG") == []


def test_negative_epv_card_is_not_a_price_and_has_no_shelf(conn):
    _val(conn, "NEG", "2026-08-01", "epv", {"status": "EPV_NEGATIVE", "value_per_share": -4.0})
    conn.commit()
    cards = company.latest_valuations(conn, "NEG")
    assert cards[0]["fair_value"] == {"kind": "not_a_price", "value": -4.0}
    assert cards[0]["status"] == "EPV_NEGATIVE"
    assert company.gauge_shelves(cards, anchor_method=None, anchor_value=None) == []


def test_header_price_for_a_non_watchlist_ticker_comes_from_valuation_inputs(conn):
    _val(conn, "NWL", "2026-08-01", "epv", {"status": "OK", "value_per_share": 9.0},
         inputs={"current_price": 12.5, "price_as_of_date": "2026-07-31"})
    conn.commit()
    prices = company._price_from_valuation_inputs(conn, "NWL")
    assert prices["latest"] == 12.5
    assert prices["checked_at"] == "2026-07-31"
    assert prices["source"] == "valuation_inputs"


def _row(**over):
    base = {
        "id": 1, "ticker": "AAA", "presented_status": "ACTIVE", "conviction_grade": None,
        "confidence": None, "latest_price": 90.0, "latest_price_checked_at": None,
        "latest_price_basis": "SNAPSHOT", "buy_price_target": 100.0,
        "distance_from_buy_pct": -10.0, "status": "ACTIVE", "valuation_anchor_method": None,
        "valuation_anchor_value": None, "source_sector": None,
    }
    base.update(over)
    return base


def test_waterline_drops_rows_that_only_have_the_price_at_addition(conn):
    queue = [
        _row(id=1, ticker="LIVE"),
        _row(id=2, ticker="OLD", latest_price_basis="PRICE_AT_ADDITION", latest_price=100.0,
             distance_from_buy_pct=0.0),
    ]
    items, deeper = today.waterline(conn, queue)
    assert [i["ticker"] for i in items] == ["LIVE"]
    assert deeper == {"count": 0, "names": []}


def test_waterline_uses_the_newest_snapshot_across_duplicate_rows(conn):
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(watchlist)")]
    assert "ticker" in cols
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.schema import ensure_watchlist_schema
    from app.watchlist.store import add_or_update, add_price_snapshot
    import dataclasses

    db_path = get_config().db_path
    ensure_watchlist_schema(db_path)
    entry = WatchlistEntry(
        ticker="DUP", status="ACTIVE", conviction_grade="WATCHLIST_ONLY", confidence="HIGH",
        conviction_source="company_autonomy", scan_family="normal",
        valuation_anchor_method="DCF", valuation_anchor_value=106.67, buy_price_target=80.0,
        current_price_at_addition=100.0, thesis_text="t", key_risks=[], falsifiers=[],
        open_questions=[], source_run_id="r1", source_sector="s",
        added_at="2026-05-08T12:00:00+00:00",
    )
    old_id = add_or_update(entry, db_path=db_path)
    add_price_snapshot(old_id, price=82.0, checked_at="2026-09-01T00:00:00+00:00", db_path=db_path)
    new_id = add_or_update(dataclasses.replace(entry, source_run_id="r2"), db_path=db_path)
    assert new_id != old_id
    # The presented row (new_id) has no snapshot: it would show the price at addition.
    queue = [
        _row(id=new_id, ticker="DUP", latest_price=100.0,
             latest_price_basis="PRICE_AT_ADDITION", distance_from_buy_pct=25.0,
             buy_price_target=80.0)
    ]
    items, _ = today.waterline(conn, queue)
    assert [(i["ticker"], i["latest_price"], i["distance_from_buy_pct"]) for i in items] == [
        ("DUP", 82.0, 2.5)
    ]


def test_gate_blocked_uses_the_newest_snapshot_and_never_shows_price_at_addition(conn):
    from app.watchlist.contract import WatchlistEntry
    from app.watchlist.schema import ensure_watchlist_schema
    from app.watchlist.store import add_or_update, add_price_snapshot

    db_path = get_config().db_path
    ensure_watchlist_schema(db_path)

    def _entry(ticker):
        return WatchlistEntry(
            ticker=ticker, status="ACTIVE", conviction_grade="WATCHLIST_ONLY",
            confidence="HIGH", conviction_source="company_autonomy", scan_family="normal",
            valuation_anchor_method="DCF", valuation_anchor_value=106.67,
            buy_price_target=80.0, current_price_at_addition=100.0, thesis_text="t",
            key_risks=[], falsifiers=[], open_questions=[], source_run_id="r1",
            source_sector="s", added_at="2026-05-08T12:00:00+00:00",
        )

    snap_id = add_or_update(_entry("SNP"), db_path=db_path)
    add_price_snapshot(snap_id, price=82.0, checked_at="2026-09-01T00:00:00+00:00",
                       db_path=db_path)
    queue = [
        _row(id=snap_id, ticker="SNP", presented_status="EVENT_PENDING", event_pending="m",
             latest_price=100.0, latest_price_basis="PRICE_AT_ADDITION",
             distance_from_buy_pct=25.0, buy_price_target=80.0),
        _row(id=99, ticker="OLD", presented_status="EVENT_PENDING", event_pending="m",
             latest_price=100.0, latest_price_basis="PRICE_AT_ADDITION",
             distance_from_buy_pct=25.0),
        _row(id=98, ticker="LIVE", presented_status="EVENT_PENDING", event_pending="m",
             latest_price=90.0, distance_from_buy_pct=-10.0, buy_price_target=100.0),
    ]
    out = {i["ticker"]: i for i in today.gate_blocked(conn, queue)}
    assert (out["SNP"]["latest_price"], out["SNP"]["distance_from_buy_pct"]) == (82.0, 2.5)
    assert (out["OLD"]["latest_price"], out["OLD"]["distance_from_buy_pct"]) == (None, None)
    assert (out["LIVE"]["latest_price"], out["LIVE"]["distance_from_buy_pct"]) == (90.0, -10.0)


# --- historical (as-of) views are bounded by the as-of date -----------------


def test_historical_evolution_ignores_splits_after_the_as_of_date(conn):
    """Viewed as of 2026-05-01, a split on 2026-06-01 had not happened: the
    2026-03-01 value stays on its own basis (100), not the later one (50)."""
    _val(conn, "SPL", "2026-03-01", "epv", {"status": "OK", "value_per_share": 100.0})
    _split_quote(conn, "SPL", "2026-06-01", 2.0)
    conn.commit()
    points, stats = company.valuation_evolution_with_drops(conn, "SPL", as_of_date="2026-05-01")
    assert [(p["as_of_date"], p["value"]) for p in points] == [("2026-03-01", 100.0)]
    assert stats == {"dropped_pre_split": 0, "rebased": 0}


def test_historical_header_price_is_not_a_later_quote(conn):
    _val(conn, "NWL", "2026-07-01", "epv", {"status": "OK", "value_per_share": 9.0},
         inputs={"current_price": 20.0, "price_as_of_date": "2026-09-01"})
    _val(conn, "NWL", "2026-06-01", "epv", {"status": "OK", "value_per_share": 9.0},
         inputs={"current_price": 11.0, "price_as_of_date": "2026-06-01"}, created="2026-06-01T12:00:00+00:00")
    conn.commit()
    prices = company._price_from_valuation_inputs(conn, "NWL", as_of_date="2026-07-15")
    assert prices["latest"] == 11.0
    assert prices["checked_at"] == "2026-06-01"


def test_historical_view_honors_a_block_known_by_then(conn):
    _val(conn, "BLK", "2026-05-01", "epv", {"status": "OK", "value_per_share": 9.0})
    conn.execute(
        "INSERT INTO valuations (ticker, as_of_date, method, inputs_json, outputs_json, "
        "warnings_json, created_at, quality_gate_verdict) VALUES (?,?,?,?,?,?,?,?)",
        ("BLK", "2026-06-01", "scorecard", "{}", json.dumps({"signal": "VALUATION_BLOCKED"}), "[]",
         "2026-06-01T12:00:00+00:00", "BLOCK"),
    )
    conn.commit()
    methods = [
        str(r["method"]) for r in company._latest_valuation_rows(conn, "BLK", as_of_date="2026-07-01")
    ]
    assert methods == ["scorecard"]
    # Before the block, the EPV row was current.
    earlier = [
        str(r["method"]) for r in company._latest_valuation_rows(conn, "BLK", as_of_date="2026-05-15")
    ]
    assert earlier == ["epv"]


def test_durable_dcf_card_is_a_line_at_the_base():
    """A durable (spike-corrected) DCF has no low/high; as a band without ends
    the gauge drew it at $0. It is a line at the durable base."""
    card = {"method": "dcf", "fair_value": {"kind": "band", "low": 40.0, "base": 50.0, "high": 60.0}}
    company._apply_durable_dcf(card, {"dcf_base": 30.0, "dcf_raw_base": 50.0})
    assert card["fair_value"] == {"kind": "line", "value": 30.0}
    shelves = company.gauge_shelves([card], anchor_method=None, anchor_value=None)
    assert shelves == [{"method": "dcf", "label": "DCF", "value": 30.0, "emphasized": False}]
