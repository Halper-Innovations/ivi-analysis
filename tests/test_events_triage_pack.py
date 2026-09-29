from __future__ import annotations

import json

from typer.testing import CliRunner

from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.flags import sync_event_pending_flags
from app.events.triage_pack import (
    build_pack,
    dispose_command,
    memo_path,
    select_events,
    strip_html,
    verify_pack,
)
from app.watchlist.schema import ensure_watchlist_schema

runner = CliRunner()


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


def _add_watch_row(conn, ticker, status="DEPLOY_READY", **extra):
    columns = {
        "ticker": ticker,
        "status": status,
        "source_run_id": f"run_{ticker}",
        "added_at": "2026-06-01T00:00:00Z",
        "conviction_grade": "ACTIONABLE",
        "confidence": "HIGH",
        "buy_price_target": 80.0,
        "thesis_text": "Durable niche operator trading below replacement value.",
        "falsifiers_json": json.dumps(["Loss of the flagship contract"]),
        "source_sector": "industrials",
        "cap_band": "small_cap",
        **extra,
    }
    keys = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    conn.execute(f"INSERT INTO watchlist({keys}) VALUES({marks})", list(columns.values()))


def _add_event(
    conn,
    *,
    cik,
    ticker,
    event_type="merger",
    detection_date="2026-07-01",
    accession="0001104659-26-061001",
):
    eid = store.upsert_event(
        conn,
        cik=cik,
        event_type=event_type,
        anchor_accession=accession,
        company_name=f"{ticker} CO",
        detection_date=detection_date,
        source_mode="daily",
        detail={},
    )
    store.set_ticker(conn, event_id=eid, ticker=ticker)
    return eid


def test_select_orders_gate_blocked_first_and_skips_unflagged(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "BLOK", status="DEPLOY_READY")
        _add_watch_row(conn, "WAIT", status="ACTIVE")
        _add_watch_row(conn, "CLEAN", status="DEPLOY_READY")  # no event
        waiting = _add_event(conn, cik="0000000001", ticker="WAIT", detection_date="2026-07-20")
        blocked = _add_event(conn, cik="0000000002", ticker="BLOK", detection_date="2026-07-01")
        sync_event_pending_flags(conn)

        selected, skipped = select_events(conn, limit=8)

    assert skipped == 0
    assert [item["event"]["id"] for item in selected] == [blocked, waiting]
    assert selected[0]["watch"]["ticker"] == "BLOK"


def test_select_excludes_decided_events_and_respects_limit(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "AAA")
        newer = _add_event(
            conn,
            cik="0000000003",
            ticker="AAA",
            detection_date="2026-07-20",
            accession="0000000003-26-000001",
        )
        older = _add_event(
            conn,
            cik="0000000003",
            ticker="AAA",
            event_type="dilution",
            detection_date="2026-07-10",
            accession="0000000003-26-000002",
        )
        done = _add_event(
            conn,
            cik="0000000003",
            ticker="AAA",
            event_type="restatement",
            detection_date="2026-07-21",
            accession="0000000003-26-000003",
        )
        store.mark_decided(conn, event_id=done, note="reviewed")
        sync_event_pending_flags(conn)

        selected, _ = select_events(conn, limit=1)
        assert [item["event"]["id"] for item in selected] == [newer]

        selected_all, _ = select_events(conn, limit=8)
        assert [item["event"]["id"] for item in selected_all] == [newer, older]


def test_select_skips_events_with_existing_memo(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "MEMO")
        eid = _add_event(conn, cik="0000000004", ticker="MEMO")
        sync_event_pending_flags(conn)

        target = memo_path(eid, "MEMO")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("already reviewed", encoding="utf-8")

        selected, skipped = select_events(conn, limit=8)
    assert selected == []
    assert skipped == 1


def test_build_pack_writes_docs_manifest_and_context(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "TDW", buy_price_target=81.04)
        eid = _add_event(conn, cik="0001692427", ticker="TDW", accession="0001104659-26-061001")
        conn.execute(
            "INSERT INTO watchlist_price_snapshots(watchlist_id, price, checked_at, source) "
            "SELECT id, 75.68, '2026-07-20T12:00:00Z', 'yahoo' FROM watchlist WHERE ticker='TDW'",
        )
        sync_event_pending_flags(conn)

    def fetch_json(url):
        assert url == (
            "https://www.sec.gov/Archives/edgar/data/1692427/000110465926061001/index.json"
        )
        return {
            "directory": {
                "item": [
                    {"name": "tm2600001-1_8k.htm", "size": 40000},
                    {"name": "tm2600001-1_ex99-1.htm", "size": 20000},
                    {"name": "0001104659-26-061001-index.htm", "size": 4000},
                ]
            }
        }

    def fetch_bytes(url):
        return b"<html><body><p>Merger  Agreement</p><script>x=1</script></body></html>"

    result = build_pack(limit=8, fetch_bytes=fetch_bytes, fetch_json=fetch_json)
    assert result["events"] == 1
    assert result["skipped_existing"] == 0

    pack = json.loads(open(result["pack_path"], encoding="utf-8").read())
    entry = pack["events"][0]
    assert entry["event_id"] == eid
    assert entry["ticker"] == "TDW"
    assert entry["flag"] == "EVENT_PENDING:MERGER"
    assert entry["dispose_command"] == (
        f'ivi events dispose {eid} --reason-code EVENT_REVIEWED --note "<one-line factual note>"'
    )
    assert entry["watchlist"]["latest_price"] == 75.68
    assert entry["watchlist"]["buy_price_target"] == 81.04
    assert entry["watchlist"]["distance_from_buy_pct"] == -6.6
    assert entry["watchlist"]["falsifiers"] == ["Loss of the flagship contract"]
    assert entry["watchlist"]["event_pending"] == "EVENT_PENDING:MERGER"

    filings = entry["filings"]
    assert len(filings) == 1
    assert filings[0]["accession"] == "0001104659-26-061001"
    assert filings[0]["fetch_error"] is None
    assert len(filings[0]["docs"]) == 2
    doc_text = open(filings[0]["docs"][0], encoding="utf-8").read()
    assert doc_text == "Merger Agreement"
    # Primary document leads; the exhibit follows; the index page is excluded.
    assert "_8k.htm" in filings[0]["docs"][0]
    assert "_ex99-1.htm" in filings[0]["docs"][1]


def test_build_pack_records_fetch_error_but_still_packs(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "ERR")
        _add_event(conn, cik="0000000005", ticker="ERR")
        sync_event_pending_flags(conn)

    def failing_json(url):
        raise RuntimeError("edgar down")

    def failing_bytes(url):
        raise RuntimeError("edgar down")

    result = build_pack(limit=8, fetch_bytes=failing_bytes, fetch_json=failing_json)
    assert result["events"] == 1
    pack_path = result["pack_path"]
    entry = json.loads(open(pack_path, encoding="utf-8").read())["events"][0]
    assert entry["filings"][0]["docs"] == []
    assert entry["filings"][0]["fetch_error"] == "no_documents_fetched"


def test_build_pack_no_events(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    result = build_pack(limit=8)
    assert result == {"pack_path": None, "events": 0, "skipped_existing": 0}


def test_verify_pack_reports_missing_then_written(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "VRFY")
        eid = _add_event(conn, cik="0000000006", ticker="VRFY")
        sync_event_pending_flags(conn)

    def fetch_json(url):
        return {"directory": {"item": [{"name": "a_8k.htm", "size": 100}]}}

    result = build_pack(limit=8, fetch_bytes=lambda url: b"<p>ok</p>", fetch_json=fetch_json)
    pack_path = result["pack_path"]

    before = verify_pack(pack_path)
    assert before == {
        "expected": 1,
        "written": 0,
        "missing": [{"event_id": eid, "ticker": "VRFY"}],
    }

    target = memo_path(eid, "VRFY")
    target.write_text(
        "# 8-K triage — VRFY\n" + ("- verdict: THESIS_INTACT\n" * 20), encoding="utf-8"
    )
    after = verify_pack(pack_path)
    assert after == {"expected": 1, "written": 1, "missing": []}


def test_strip_html_drops_scripts_and_collapses_whitespace():
    raw = b"<html><style>p{}</style><body><h1>Item  1.01</h1><script>x</script><p>Entry into a\nMaterial Agreement</p></body></html>"
    assert strip_html(raw) == "Item 1.01 Entry into a\nMaterial Agreement"


def test_cli_triage_pack_no_events(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    from app.cli import app

    result = runner.invoke(app, ["events", "triage-pack", "--limit", "3"])
    assert result.exit_code == 0, result.output
    assert "NO_EVENTS" in result.output


def test_cli_triage_verify_exits_nonzero_on_missing(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _add_watch_row(conn, "CLIV")
        _add_event(conn, cik="0000000007", ticker="CLIV")
        sync_event_pending_flags(conn)
    result = build_pack(
        limit=8,
        fetch_bytes=lambda url: b"<p>ok</p>",
        fetch_json=lambda url: {"directory": {"item": [{"name": "a_8k.htm", "size": 9}]}},
    )
    from app.cli import app

    verify = runner.invoke(app, ["events", "triage-verify", result["pack_path"]])
    assert verify.exit_code == 1
    assert '"written": 0' in verify.output


def test_dispose_command_literal():
    assert dispose_command(123) == (
        'ivi events dispose 123 --reason-code EVENT_REVIEWED --note "<one-line factual note>"'
    )
