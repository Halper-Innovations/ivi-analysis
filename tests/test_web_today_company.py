"""Phase 1 API surface: /api/today and /api/company/*.

Hermetic tmp-DB fixtures; exact-literal assertions. The decisions section
must agree with the dispositions store's own derivation (shared code path),
and the waterline must agree with ``watchlist_queue`` presentation.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.db import init_db
from app.watchlist.contract import WatchlistEntry
from app.watchlist.dispositions import sync_at_target_dispositions
from app.watchlist.schema import ensure_watchlist_schema
from app.watchlist.store import add_or_update, add_price_snapshot
from app.web.main import app
from app.web.readmodel import today

client = TestClient(app)


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    ensure_watchlist_schema(db_path)
    return cfg


def _entry(*, ticker: str, status: str = "ACTIVE") -> WatchlistEntry:
    return WatchlistEntry(
        ticker=ticker,
        status=status,
        conviction_grade="ACTIONABLE" if status == "DEPLOY_READY" else "WATCHLIST_ONLY",
        confidence="HIGH",
        conviction_source="company_autonomy",
        scan_family="normal",
        valuation_anchor_method="DCF",
        valuation_anchor_value=106.67,
        buy_price_target=80.0,
        current_price_at_addition=100.0,
        thesis_text="Durable candidate with a buy-price anchor.",
        key_risks=["Margin compression"],
        falsifiers=["Revenue decline persists"],
        open_questions=["Customer concentration?"],
        source_run_id="sector_run_1",
        source_sector="industrial_tech",
        added_at="2026-05-08T12:00:00+00:00",
    )


def _fresh_iso(hours_ago: float = 1.0) -> str:
    return (
        (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).replace(microsecond=0).isoformat()
    )


def _seed_floor(cfg) -> dict[str, int]:
    """AAA waiting, CCC at target (opens a disposition), BBB event-blocked,
    QQQ quarantined, SSS price-suspect."""
    db_path = cfg.db_path
    ids = {
        "AAA": add_or_update(_entry(ticker="AAA"), db_path=db_path),
        "CCC": add_or_update(_entry(ticker="CCC", status="DEPLOY_READY"), db_path=db_path),
        "BBB": add_or_update(_entry(ticker="BBB", status="DEPLOY_READY"), db_path=db_path),
        "QQQ": add_or_update(_entry(ticker="QQQ", status="QUARANTINE"), db_path=db_path),
        "SSS": add_or_update(_entry(ticker="SSS", status="PRICE_DATA_SUSPECT"), db_path=db_path),
    }
    checked_at = _fresh_iso()
    add_price_snapshot(ids["AAA"], price=95.0, checked_at=checked_at, db_path=db_path)
    add_price_snapshot(ids["CCC"], price=78.0, checked_at=checked_at, db_path=db_path)
    add_price_snapshot(ids["BBB"], price=78.0, checked_at=checked_at, db_path=db_path)
    add_price_snapshot(ids["QQQ"], price=10.0, checked_at=checked_at, db_path=db_path)
    add_price_snapshot(ids["SSS"], price=10.0, checked_at=checked_at, db_path=db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE watchlist SET event_pending = 'merger' WHERE id = ?", (ids["BBB"],))
    conn.commit()
    conn.close()
    sync_at_target_dispositions(db_path)
    return ids


def _seed_digests(cfg) -> None:
    digests = cfg.outputs_dir / "digests"
    digests.mkdir(parents=True, exist_ok=True)
    (digests / "digest_2026-07-20.md").write_text("# Old digest\n", encoding="utf-8")
    (digests / "digest_2026-07-21.md").write_text(
        "# IVI Watchlist Daily Digest\n\n## Newly Deploy-Ready\n- none\n",
        encoding="utf-8",
    )


def test_latest_digest_renders_authorized_bytes_not_swapped_path(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_digests(cfg)
    latest_path = cfg.outputs_dir / "digests" / "digest_2026-07-21.md"
    authorized_bytes = latest_path.read_bytes()

    def authorize_then_swap(path):
        assert Path(path) == latest_path
        latest_path.write_text("# FORGED DIGEST\n", encoding="utf-8")
        return "PASS", authorized_bytes

    monkeypatch.setattr(today, "authorized_artifact_bytes", authorize_then_swap)
    digest = today.latest_digest()
    assert digest is not None
    assert "IVI Watchlist Daily Digest" in digest["html"]
    assert "FORGED DIGEST" not in digest["html"]


def test_api_today_full_deck(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    ids = _seed_floor(cfg)
    _seed_digests(cfg)

    response = client.get("/api/today")
    assert response.status_code == 200
    payload = response.json()

    # Health: VOE_DB_PATH override → cheap mode, single readability check.
    assert payload["health"]["state"] == "GREEN"
    assert payload["health"]["blocking"] is False
    assert payload["health"]["checks"] == [
        {"name": "db_readable", "ok": True, "detail": str(cfg.db_path)}
    ]

    # Decisions: exactly the at-target CCC disposition, currently actionable.
    assert len(payload["decisions"]) == 1
    decision = payload["decisions"][0]
    assert decision["ticker"] == "CCC"
    assert decision["kind"] == "AT_TARGET"
    assert decision["watchlist_status"] == "DEPLOY_READY"
    assert decision["conviction_grade"] == "ACTIONABLE"
    assert decision["source_state"] == "CURRENT"
    assert decision["is_current_actionable"] is True
    assert decision["latest_price"] == 78.0
    assert decision["buy_price_target"] == 80.0
    assert decision["distance_from_buy_pct"] == -2.5
    assert decision["falsifiers"] == ["Revenue decline persists"]
    assert decision["journal_command"] == (
        f"ivi investor journal CCC --disposition-id {decision['id']} "
        '--action acted|passed|deferred --reason <CODE> --rationale "<why>"'
    )
    assert decision["trigger"]["watchlist_status"] == "DEPLOY_READY"

    # Waterline: sorted by |distance| (nearest the line on either side, ties
    # keep queue order); QUARANTINE and PRICE_DATA_SUSPECT stay off the
    # strip; BBB floats as EVENT_PENDING. Everything fits under the limit,
    # so the deeper block is empty.
    tickers = [item["ticker"] for item in payload["waterline"]]
    assert tickers == ["CCC", "BBB", "AAA"]
    assert payload["waterline_deeper"] == {"count": 0, "names": []}
    top = payload["waterline"][0]
    assert top["distance_from_buy_pct"] == -2.5
    assert top["presented_status"] == "DEPLOY_READY"
    assert top["wake"] == [78.0]
    assert top["valuation_anchor_method"] == "DCF"
    assert top["valuation_anchor_value"] == 106.67
    assert top["price_suspect"] is False
    assert top["price_age_hours"] is not None
    assert abs(top["price_age_hours"] - 1.0) < 0.1
    assert payload["waterline"][1]["presented_status"] == "EVENT_PENDING"
    assert payload["waterline"][2]["distance_from_buy_pct"] == 18.75

    # Gate-blocked: BBB with its open event flag.
    assert payload["gate_blocked"] == [
        {
            "watchlist_id": ids["BBB"],
            "ticker": "BBB",
            "event_pending": "merger",
            "conviction_grade": "ACTIONABLE",
            "distance_from_buy_pct": -2.5,
            "latest_price": 78.0,
            "buy_price_target": 80.0,
        }
    ]

    # Digest: newest rendered, previous date carried.
    assert payload["digest"]["date"] == "2026-07-21"
    assert payload["digest"]["previous_date"] == "2026-07-20"
    assert "<h1>IVI Watchlist Daily Digest</h1>" in payload["digest"]["html"]
    assert "Newly Deploy-Ready" in payload["digest"]["html"]


def test_waterline_selection_is_abs_distance_with_deeper_summary(monkeypatch, tmp_path):
    """The strip picks names nearest the line on EITHER side; names cut off
    below the line land in the deeper summary, far-above names never do."""
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_floor(cfg)

    from app.web.api.routes import queue_rows, readonly_db
    from app.web.readmodel import today as today_model

    with readonly_db() as conn:
        queue = queue_rows(conn, limit=2000)

        # Synthetic ordering check, independent of the seeded floor: |+2| < |+5| < |-30|.
        synthetic = [
            {
                "id": 900,
                "ticker": "FAR",
                "presented_status": "ACTIVE",
                "conviction_grade": None,
                "confidence": None,
                "latest_price": 70.0,
                "latest_price_checked_at": None,
                "buy_price_target": 100.0,
                "distance_from_buy_pct": -30.0,
                "status": "ACTIVE",
                "valuation_anchor_method": None,
                "valuation_anchor_value": None,
                "source_sector": None,
            },
            {
                "id": 901,
                "ticker": "MID",
                "presented_status": "ACTIVE",
                "conviction_grade": None,
                "confidence": None,
                "latest_price": 105.0,
                "latest_price_checked_at": None,
                "buy_price_target": 100.0,
                "distance_from_buy_pct": 5.0,
                "status": "ACTIVE",
                "valuation_anchor_method": None,
                "valuation_anchor_value": None,
                "source_sector": None,
            },
            {
                "id": 902,
                "ticker": "NEAR",
                "presented_status": "ACTIVE",
                "conviction_grade": None,
                "confidence": None,
                "latest_price": 102.0,
                "latest_price_checked_at": None,
                "buy_price_target": 100.0,
                "distance_from_buy_pct": 2.0,
                "status": "ACTIVE",
                "valuation_anchor_method": None,
                "valuation_anchor_value": None,
                "source_sector": None,
            },
        ]
        items, deeper = today_model.waterline(conn, synthetic)
        assert [item["ticker"] for item in items] == ["NEAR", "MID", "FAR"]
        assert deeper == {"count": 0, "names": []}

        # Seeded floor with limit=1: CCC (|-2.5|) survives the cut; BBB
        # (-2.5, below the line) lands in deeper; AAA (+18.75, above the
        # line) is cut but is NOT "deeper".
        items, deeper = today_model.waterline(conn, queue, limit=1)
        assert [item["ticker"] for item in items] == ["CCC"]
        assert deeper == {
            "count": 1,
            "names": [{"ticker": "BBB", "distance_from_buy_pct": -2.5}],
        }


def test_api_today_missing_engine_db_is_structured_503(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    from app.config import get_config

    get_config.cache_clear()
    response = client.get("/api/today")
    assert response.status_code == 503
    assert response.json()["detail"]["precondition"] == "engine_db_missing"


def test_api_today_no_digests_renders_null(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_floor(cfg)
    response = client.get("/api/today")
    assert response.status_code == 200
    assert response.json()["digest"] is None


def _seed_valuations(cfg) -> None:
    conn = sqlite3.connect(cfg.db_path)
    rows = [
        # Pre-hardening run (2026-06-10) for dcf/epv/graham/fcf_yield/scorecard.
        (
            "CCC",
            "2026-06-10",
            "dcf",
            json.dumps(
                {
                    "status": "OK",
                    "low": 74.0,
                    "base": 102.0,
                    "high": 128.0,
                    "flags": [],
                    "wacc_detail": {
                        "baseline_wacc": 0.1,
                        "adjusted_wacc": 0.095,
                        "adjustments": [
                            {
                                "code": "STRONG_CASH_CONVERSION",
                                "delta": -0.005,
                                "reason": "CFO/NI above 1.5x",
                            }
                        ],
                    },
                }
            ),
            "2026-06-10T10:00:00+00:00",
            "PROCEED",
            "HIGH",
            json.dumps(["OK"]),
            json.dumps(["SECULAR_DECLINE"]),
            json.dumps(["NET_CASH"]),
        ),
        (
            "CCC",
            "2026-06-10",
            "epv",
            json.dumps({"status": "OK", "value_per_share": 88.0, "flags": []}),
            "2026-06-10T10:00:00+00:00",
            "PROCEED",
            "HIGH",
            "[]",
            "[]",
            "[]",
        ),
        (
            "CCC",
            "2026-06-10",
            "graham",
            json.dumps(
                {
                    "status": "OK",
                    "value_per_share": 96.5,
                    "buy_price": 64.6,
                    "normalized_eps": 5.2,
                    "bvps": 41.0,
                    "flags": [],
                }
            ),
            "2026-06-10T10:00:00+00:00",
            "PROCEED",
            "HIGH",
            "[]",
            "[]",
            "[]",
        ),
        (
            "CCC",
            "2026-06-10",
            "fcf_yield",
            json.dumps(
                {
                    "status": "METHOD_INSUFFICIENT_DATA",
                    "value_per_share": None,
                    "flags": ["INSUFFICIENT_INPUTS"],
                }
            ),
            "2026-06-10T10:00:00+00:00",
            None,
            None,
            "[]",
            "[]",
            "[]",
        ),
        # Post-hardening graham re-run: becomes the latest card and the second
        # evolution point.
        (
            "CCC",
            "2026-07-18",
            "graham",
            json.dumps(
                {
                    "status": "OK",
                    "value_per_share": 97.5,
                    "buy_price": 65.3,
                    "normalized_eps": 5.3,
                    "bvps": 41.5,
                    "flags": [],
                }
            ),
            "2026-07-18T10:00:00+00:00",
            "PROCEED",
            "HIGH",
            "[]",
            "[]",
            "[]",
        ),
    ]
    conn.executemany(
        """
        INSERT INTO valuations (
            ticker, as_of_date, method, inputs_json, outputs_json, warnings_json,
            created_at, quality_gate_verdict, confidence_class, gate_reason_codes,
            valuation_headwinds, valuation_supports
        ) VALUES (?, ?, ?, '{}', ?, '[]', ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_api_company_masthead_gauge_and_cards(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_floor(cfg)
    _seed_valuations(cfg)

    response = client.get("/api/company/ccc")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "CCC"

    # The queue row rides along (shared watchlist_queue derivation).
    assert payload["watchlist"]["presented_status"] == "DEPLOY_READY"
    assert payload["watchlist"]["valuation_anchor_method"] == "DCF"
    assert payload["watchlist"]["valuation_anchor_value"] == 106.67

    # Profile narrative fields.
    assert payload["profile"]["thesis_text"] == "Durable candidate with a buy-price anchor."
    assert payload["profile"]["key_risks"] == ["Margin compression"]
    assert payload["profile"]["open_questions"] == ["Customer concentration?"]

    # Gauge shelves: the DCF band, EPV and the latest Graham line, plus the
    # watchlist ANCHOR on its own emphasized line. The anchor here is 106.67
    # while the DCF card's base is 102.00 — they come from different runs — and
    # since 2026-09-02 the gauge draws the number the buy target actually hangs
    # from instead of emphasizing a card that is not it.
    assert payload["shelves"] == [
        {
            "method": "dcf",
            "label": "DCF",
            "value": None,
            "low": 74.0,
            "base": 102.0,
            "high": 128.0,
            "emphasized": False,
        },
        {
            "method": "epv",
            "label": "EPV",
            "value": 88.0,
            "low": None,
            "base": None,
            "high": None,
            "emphasized": False,
        },
        {
            "method": "graham",
            "label": "Graham",
            "value": 97.5,
            "low": None,
            "base": None,
            "high": None,
            "emphasized": False,
        },
        {
            "method": "dcf",
            "label": "DCF (anchor)",
            "value": 106.67,
            "low": None,
            "base": None,
            "high": None,
            "emphasized": True,
        },
    ]

    # Cards: latest per method; the insufficient method stays honest.
    by_method = {card["method"]: card for card in payload["valuations"]}
    assert by_method["graham"]["as_of_date"] == "2026-07-18"
    assert by_method["graham"]["fair_value"]["value"] == 97.5
    assert by_method["graham"]["extras"]["buy_price"] == 65.3
    assert by_method["dcf"]["wacc"]["adjusted_wacc"] == 0.095
    assert by_method["dcf"]["wacc"]["adjustments"][0]["code"] == "STRONG_CASH_CONVERSION"
    assert by_method["dcf"]["headwinds"] == ["SECULAR_DECLINE"]
    assert by_method["dcf"]["supports"] == ["NET_CASH"]
    assert by_method["fcf_yield"]["status"] == "METHOD_INSUFFICIENT_DATA"
    assert by_method["fcf_yield"]["fair_value"] == {
        "kind": "none",
        "value": None,
        "low": None,
        "base": None,
        "high": None,
    }

    # Evolution: graham twice, straddling the hardening cutoff.
    graham_points = [p for p in payload["evolution"] if p["method"] == "graham"]
    assert graham_points == [
        {
            "method": "graham",
            "as_of_date": "2026-06-10",
            "value": 96.5,
            "archived": False,
            "pre_hardening": True,
        },
        {
            "method": "graham",
            "as_of_date": "2026-07-18",
            "value": 97.5,
            "archived": False,
            "pre_hardening": False,
        },
    ]

    # Prices: wake and latest from snapshots.
    assert payload["prices"]["latest"] == 78.0
    assert payload["prices"]["wake"] == [78.0]


def test_api_company_prices_join_through_ticker_across_dup_rows(monkeypatch, tmp_path):
    """Dup-ticker rows exist in the books; snapshots on a superseded row must
    still feed the company page (a company's price history is one series)."""
    cfg = _init_temp_env(monkeypatch, tmp_path)
    db_path = cfg.db_path
    old_entry = _entry(ticker="DUP")
    old_id = add_or_update(old_entry, db_path=db_path)
    add_price_snapshot(old_id, price=41.5, checked_at=_fresh_iso(2.0), db_path=db_path)
    new_entry = dataclasses.replace(
        _entry(ticker="DUP", status="DEPLOY_READY"), source_run_id="sector_run_2"
    )
    new_id = add_or_update(new_entry, db_path=db_path)
    assert new_id != old_id  # genuinely two rows

    response = client.get("/api/company/DUP")
    assert response.status_code == 200
    payload = response.json()
    assert payload["prices"]["latest"] == 41.5
    assert payload["prices"]["wake"] == [41.5]


def test_api_company_unknown_ticker_404(monkeypatch, tmp_path):
    _init_temp_env(monkeypatch, tmp_path)
    response = client.get("/api/company/ZZZZ")
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown ticker: ZZZZ"


def _seed_companyfacts(cfg) -> None:
    conn = sqlite3.connect(cfg.db_path)
    facts = {
        2024: {
            "revenue": 100.0,
            "gross_profit": 40.0,
            "operating_income": 12.0,
            "net_income": 8.0,
            "cfo": 20.0,
            "capex": 5.0,
            "shares_outstanding": 10.0,
            "total_debt": 50.0,
            "cash": 10.0,
            "equity": 40.0,
            "total_assets": 120.0,
        },
        2025: {
            "revenue": 110.0,
            "gross_profit": 45.0,
            "operating_income": 14.0,
            "net_income": 9.0,
            "cfo": 22.0,
            "capex": 6.0,
            "shares_outstanding": 10.5,
            "total_debt": 48.0,
            "cash": 12.0,
            "equity": 44.0,
            "total_assets": 130.0,
        },
    }
    for year, items in facts.items():
        for line_item, value in items.items():
            conn.execute(
                """
                INSERT INTO companyfacts_facts (
                    ticker, fiscal_year, period_type, period_end, line_item,
                    value, units, fetched_at
                ) VALUES ('CCC', ?, 'FY', ?, ?, ?, 'USD_millions', '2026-07-01T00:00:00+00:00')
                """,
                (year, f"{year}-12-31", line_item, value),
            )
    conn.commit()
    conn.close()


def test_api_company_fundamentals_fy(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_floor(cfg)
    _seed_companyfacts(cfg)

    response = client.get("/api/company/CCC/fundamentals", params={"basis": "fy"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["basis"] == "fy"
    assert payload["units"] == "USD_millions"
    assert payload["fiscal_years"] == [2024, 2025]
    assert payload["series"]["revenue"] == [100.0, 110.0]
    assert payload["series"]["sbc"] == [None, None]
    assert payload["derived"]["fcf"] == [15.0, 16.0]
    assert payload["derived"]["net_debt"] == [40.0, 36.0]
    assert payload["derived"]["gross_margin"] == [0.4, 0.4091]
    assert payload["derived"]["operating_margin"] == [0.12, 0.1273]
    assert payload["derived"]["net_margin"] == [0.08, 0.0818]
    assert payload["derived"]["roe"] == [0.2, 0.2045]
    assert payload["derived"]["shares_dilution_pct"] == [None, 5.0]


def test_api_company_fundamentals_ttm_not_available(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_floor(cfg)
    response = client.get("/api/company/CCC/fundamentals", params={"basis": "ttm"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["reason"] == "TTM_NOT_AVAILABLE"
