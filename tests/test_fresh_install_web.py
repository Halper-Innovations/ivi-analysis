"""A fresh install must not 500.

A new user runs ``ivi init-db`` (or ``ivi value KO``, which does the same) and
then opens the web UI. Three contracts hold from that first minute:

- ``init_db`` creates every table the web read model and the read-only CLI
  paths query (they used to be created on first write, so a fresh database
  answered ``no such table: watchlist``);
- every GET route answers without a server error, including on a database that
  predates that change and still lacks those tables;
- the company page for a name valued with ``ivi value`` says plainly that it is
  not audited, keeps every decision field empty, and shows the research
  valuation in its own labelled block.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from app.config import get_config
from app.db import get_db, init_db
from app.web.main import app
from app.web.readmodel import company as company_model
from app.web.readmodel.db import OfflineError

client = TestClient(app, raise_server_exceptions=False)

# Tables that used to be created lazily on first write, not by init_db.
LATE_TABLES = (
    "watchlist",
    "watchlist_history",
    "watchlist_price_snapshots",
    "watchlist_reevaluation_publications",
    "holdings",
    "exit_signals",
)

MANIFEST_UNUSABLE = "financial_integrity_manifest_unusable"


def _tables(db_path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()


def _fresh(isolated_data_root, monkeypatch):
    """The state a new user is in: config pointed at an empty data dir, no audit."""
    monkeypatch.delenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", raising=False)
    cfg = get_config()
    init_db(cfg)
    return cfg


def _audited(monkeypatch, tmp_path):
    """An install that *has* run the financial-integrity audit (a real, usable manifest)."""
    from tests.test_classic_postwrite_authorization import _baseline_manifest

    _baseline_manifest(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    cfg = get_config()
    init_db(cfg)
    return cfg


def _drop_late_tables(db_path) -> None:
    """Recreate the shape of a database initialized before init_db owned them."""
    conn = sqlite3.connect(db_path)
    for table in (
        "watchlist_reevaluation_publications",
        "watchlist_price_snapshots",
        "watchlist_history",
        "watchlist",
        "holdings",
        "exit_signals",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()


def _insert_valuation(
    conn, ticker: str, method: str, outputs: dict, *, as_of: str = "2026-09-25", run_id=None
) -> None:
    conn.execute(
        """
        INSERT INTO valuations(ticker, as_of_date, method, inputs_json, outputs_json,
                               warnings_json, created_at, source_run_id)
        VALUES (?, ?, ?, '{}', ?, '[]', ?, ?)
        """,
        (
            ticker,
            as_of,
            method,
            json.dumps(outputs),
            datetime.now(timezone.utc).isoformat(),
            run_id,
        ),
    )


def _all_get_urls() -> list[tuple[str, str]]:
    """(template, concrete url) for every GET route, from the app's own OpenAPI schema."""
    urls: list[tuple[str, str]] = []
    for template, operations in app.openapi()["paths"].items():
        operation = operations.get("get")
        if operation is None:
            continue
        url = re.sub(r"\{[^}]+\}", "KO", template)
        required = [
            f"{p['name']}=KO"
            for p in operation.get("parameters", [])
            if p["in"] == "query" and p.get("required")
        ]
        urls.append((template, url + ("?" + "&".join(required) if required else "")))
    return urls


def _sweep() -> dict[str, int]:
    return {template: client.get(url).status_code for template, url in _all_get_urls()}


# --- init_db owns every table the read paths query ---------------------------------


def test_init_db_creates_the_tables_the_read_paths_query(isolated_data_root):
    cfg = get_config()
    assert not (set(LATE_TABLES) & (_tables(cfg.db_path) if cfg.db_path.exists() else set()))

    init_db(cfg)

    assert set(LATE_TABLES) <= _tables(cfg.db_path)
    conn = sqlite3.connect(cfg.db_path)
    watchlist_columns = {row[1] for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()}
    conn.close()
    # The full column set (base + every migration), not just the base table.
    assert {"ticker", "status", "source_run_id", "adv_dollar_20d", "cap_band"} <= watchlist_columns
    assert "current_event_watermark_json" in watchlist_columns

    init_db(cfg)  # idempotent
    assert set(LATE_TABLES) <= _tables(cfg.db_path)


def test_init_db_on_a_caller_connection_creates_them_too():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row  # what init_db(conn=...) callers already provide
    init_db(conn=conn)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert set(LATE_TABLES) <= names


# --- no GET route 500s --------------------------------------------------------------


def test_route_enumeration_covers_the_company_and_search_routes():
    templates = {template for template, _ in _all_get_urls()}
    assert {
        "/api/watchlist",
        "/api/today",
        "/api/search",
        "/api/company/{ticker}",
        "/api/company/{ticker}/fundamentals",
        "/api/company/{ticker}/research",
        "/api/company/{ticker}/dossier",
        "/api/company/{ticker}/decisions",
        "/api/runs",
        "/api/meta",
    } <= templates
    assert len(templates) >= 20


@pytest.mark.financial_integrity_contract
@pytest.mark.parametrize("valued", [False, True], ids=["init_db_only", "after_ivi_value"])
def test_no_get_route_errors_on_a_fresh_install(isolated_data_root, monkeypatch, valued):
    # Unpatched integrity gates: a fresh install has no audit manifest.
    cfg = _fresh(isolated_data_root, monkeypatch)
    if valued:
        with get_db(cfg=cfg) as conn:
            _insert_valuation(
                conn, "KO", "dcf", {"status": "OK", "low": 40, "base": 50, "high": 60}
            )

    statuses = _sweep()

    assert {t: s for t, s in statuses.items() if s >= 500} == {}
    # The routes a new user hits first answer with data, not an error.
    assert statuses["/api/watchlist"] == 200
    assert statuses["/api/today"] == 200
    assert statuses["/api/search"] == 200
    assert statuses["/api/company/{ticker}"] == (200 if valued else 404)
    assert statuses["/api/company/{ticker}/fundamentals"] == (200 if valued else 404)


@pytest.mark.financial_integrity_contract
def test_fresh_install_today_health_is_not_red_for_a_missing_table(isolated_data_root, monkeypatch):
    _fresh(isolated_data_root, monkeypatch)

    health = client.get("/api/today").json()["health"]

    db_check = next(c for c in health["checks"] if c["name"] == "db_readable")
    assert db_check["ok"] is True, db_check


@pytest.mark.financial_integrity_contract
@pytest.mark.parametrize("valued", [False, True], ids=["init_db_only", "after_ivi_value"])
def test_no_get_route_errors_on_a_database_that_predates_init_db_owning_the_tables(
    monkeypatch, tmp_path, valued
):
    # An audited install on a database created before the fix: the watchlist
    # family is absent, so every reader must answer "not populated" rather
    # than "no such table". (With a real manifest the queue really queries.)
    cfg = _audited(monkeypatch, tmp_path)
    _drop_late_tables(cfg.db_path)
    assert not (set(LATE_TABLES) & _tables(cfg.db_path))
    if valued:
        with get_db(cfg=cfg) as conn:
            _insert_valuation(
                conn, "KO", "dcf", {"status": "OK", "low": 40, "base": 50, "high": 60}
            )

    statuses = _sweep()

    assert {t: s for t, s in statuses.items() if s >= 500} == {}
    assert statuses["/api/watchlist"] == 200
    assert statuses["/api/today"] == 200
    assert statuses["/api/search"] == 200
    assert statuses["/api/company/{ticker}"] == (200 if valued else 404)
    # Reading did not create the tables: the web layer never writes engine.db.
    assert not (set(LATE_TABLES) & _tables(cfg.db_path))
    watchlist = client.get("/api/watchlist").json()
    assert (watchlist["rows"], watchlist["total"]) == ([], 0)
    if valued:
        assert client.get("/api/company/KO/decisions").json()["dispositions"] == []
        assert [r["ticker"] for r in client.get("/api/search?q=KO").json()["results"]] == []


def test_a_missing_table_the_read_model_cannot_live_without_is_a_named_503(
    isolated_data_root, monkeypatch
):
    cfg = _fresh(isolated_data_root, monkeypatch)
    with get_db(cfg=cfg) as conn:
        _insert_valuation(conn, "KO", "dcf", {"status": "OK", "low": 40, "base": 50, "high": 60})
        conn.execute("DROP TABLE ticker_outcomes")

    response = client.get("/api/company/KO/decisions")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["precondition"] == "engine_db_schema_incomplete"
    assert "no such table: ticker_outcomes" in detail["detail"]
    assert "ivi init-db" in detail["detail"]


# --- the company page after `ivi value` on a fresh install ---------------------------


def _run_ivi_value(monkeypatch, ticker: str = "KO", as_of: str = "2026-09-25"):
    """`ivi value` composed offline: the fetch and the writer are stubbed, the DB is real."""
    monkeypatch.setattr(
        "app.valuation.facts.resolve_cik_for_ticker", lambda t, cfg=None: "0000021344"
    )
    monkeypatch.setattr(
        "app.market.company_facts_provider.fetch_company_facts",
        lambda cik, cfg=None: {"status": "OK", "reason_code": "FETCH_OK", "size_bytes": 5_000_000},
    )
    def fake_facts(t, years_back=10):
        # `ivi value` refuses a registrant with no annual report; one FY row stands in.
        with get_db() as conn:
            conn.execute(
                "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
                "line_item, value, units, source_url, fetched_at, filed_date, form, accession) "
                "VALUES (?, 2025, 'FY', '2025-12-31', 'revenue', 1.0, 'USD_millions', '', "
                "'2026-09-25T00:00:00+00:00', '2026-02-20', '10-K', 'x')",
                (t.upper(),),
            )

    monkeypatch.setattr("app.ingest.facts_writer.ensure_all_facts", fake_facts)

    def fake_writer(t, as_of_date, provider, **kwargs):
        with get_db() as conn:
            _insert_valuation(
                conn,
                t,
                "dcf",
                {"status": "OK", "low": 40.0, "base": 50.0, "high": 60.0},
                as_of=as_of_date,
            )
            _insert_valuation(
                conn, t, "graham", {"status": "OK", "value_per_share": 20.0}, as_of=as_of_date
            )
            _insert_valuation(
                conn, t, "scorecard", {"signal": "PASS", "pricing_zone": "FAIR"}, as_of=as_of_date
            )
        return []

    monkeypatch.setattr("app.valuation.valuation_writer.ensure_valuation", fake_writer)
    import app.cli

    return CliRunner().invoke(app.cli.app, ["value", ticker, "--as-of", as_of])


@pytest.mark.financial_integrity_contract
def test_company_page_after_ivi_value_says_not_audited_and_shows_the_research_rows(
    isolated_data_root, monkeypatch
):
    monkeypatch.delenv("VOE_FINANCIAL_INTEGRITY_MANIFEST", raising=False)
    result = _run_ivi_value(monkeypatch)
    assert result.exit_code == 2, result.output  # values, no price offline

    response = client.get("/api/company/KO")

    assert response.status_code == 200
    payload = response.json()
    assert payload["audit"]["state"] == "NOT_AUDITED"
    assert payload["audit"]["reason"] == MANIFEST_UNUSABLE
    assert payload["audit"]["message"].startswith("Not audited yet.")
    assert "`ivi value`" in payload["audit"]["message"]
    # Nothing decision-bearing is presented.
    assert payload["watchlist"] is None
    assert payload["profile"] is None
    assert payload["valuations"] == []
    assert payload["shelves"] == []
    assert payload["evolution"] == []
    assert payload["prices"]["latest"] is None
    # The valuation the user just computed is there, in its own labelled block.
    research = {card["method"]: card for card in payload["research_valuations"]}
    assert sorted(research) == ["dcf", "graham", "scorecard"]
    assert research["dcf"]["fair_value"] == {
        "kind": "band",
        "value": None,
        "low": 40.0,
        "base": 50.0,
        "high": 60.0,
    }
    assert research["graham"]["fair_value"]["value"] == 20.0
    assert research["dcf"]["as_of_date"] == "2026-09-25"


@pytest.mark.financial_integrity_contract
def test_the_snapshot_itself_still_refuses_without_an_audit(isolated_data_root, monkeypatch):
    # The gate is untouched: only the API route turns the refusal into the
    # explicit "not audited yet" page, and that page carries no decision data.
    cfg = _fresh(isolated_data_root, monkeypatch)
    with get_db(cfg=cfg) as conn:
        _insert_valuation(conn, "KO", "dcf", {"status": "OK", "low": 40, "base": 50, "high": 60})
        with pytest.raises(OfflineError) as excinfo:
            company_model.company_snapshot(conn, "KO", queue_row=None)
    assert excinfo.value.precondition == MANIFEST_UNUSABLE


@pytest.mark.financial_integrity_contract
def test_a_known_name_with_no_valuation_is_told_what_to_run(isolated_data_root, monkeypatch):
    cfg = _fresh(isolated_data_root, monkeypatch)
    with get_db(cfg=cfg) as conn:
        conn.execute(
            "INSERT INTO companyfacts_facts(ticker, fiscal_year, period_type, period_end, "
            "line_item, value, fetched_at) "
            "VALUES ('KO', 2025, 'FY', '2025-12-31', 'revenue', 47000.0, '2026-09-25T00:00:00+00:00')"
        )

    payload = client.get("/api/company/KO").json()

    assert payload["audit"]["state"] == "NO_VALUATION"
    assert payload["audit"]["command"] == "ivi value KO"
    assert payload["audit"]["message"].startswith("No valuation on file for KO. Run `ivi value KO`")
    assert payload["research_valuations"] == []
    assert payload["valuations"] == []


@pytest.mark.financial_integrity_contract
def test_research_rows_on_an_audited_install_are_labelled_not_promoted(monkeypatch, tmp_path):
    # Even with a usable audit manifest, a lineage-free row is not
    # decision-eligible, so it lands in the research block only.
    cfg = _audited(monkeypatch, tmp_path)
    with get_db(cfg=cfg) as conn:
        _insert_valuation(conn, "KO", "dcf", {"status": "OK", "low": 40, "base": 50, "high": 60})

    payload = client.get("/api/company/KO").json()

    assert payload["audit"]["state"] == "NOT_AUDITED"
    assert payload["audit"]["reason"] == "NO_AUDITED_VALUATION"
    assert payload["audit"]["message"].startswith("Research output only.")
    assert payload["valuations"] == []
    assert payload["shelves"] == []
    assert [card["method"] for card in payload["research_valuations"]] == ["dcf"]


@pytest.mark.financial_integrity_contract
def test_a_row_that_names_a_run_but_is_not_authorized_stays_suppressed(monkeypatch, tmp_path):
    # A claimed decision row that fails verification is not research output:
    # it must not surface anywhere on the page.
    cfg = _audited(monkeypatch, tmp_path)
    with get_db(cfg=cfg) as conn:
        _insert_valuation(
            conn,
            "KO",
            "dcf",
            {"status": "OK", "low": 40, "base": 50, "high": 60},
            run_id="some_run",
        )

    payload = client.get("/api/company/KO").json()

    assert payload["valuations"] == []
    assert payload["research_valuations"] == []
    # Rows exist, so the page must not claim there is no valuation on file.
    assert payload["audit"]["state"] == "NOT_AUDITED"
    assert payload["audit"]["reason"] == "AUDIT_NOT_HOLDING"
    assert payload["audit"]["message"].startswith("Valuation rows exist for KO but are withheld")
    assert payload["audit"]["command"] is None


@pytest.mark.financial_integrity_contract
def test_run_claimed_rows_without_any_audit_are_withheld_and_the_page_says_so(
    isolated_data_root, monkeypatch
):
    cfg = _fresh(isolated_data_root, monkeypatch)
    with get_db(cfg=cfg) as conn:
        _insert_valuation(
            conn,
            "KO",
            "dcf",
            {"status": "OK", "low": 40, "base": 50, "high": 60},
            run_id="some_run",
        )

    payload = client.get("/api/company/KO").json()

    assert payload["valuations"] == []
    assert payload["research_valuations"] == []
    assert payload["audit"]["state"] == "NOT_AUDITED"
    assert payload["audit"]["reason"] == MANIFEST_UNUSABLE
    assert payload["audit"]["message"].startswith("Valuation rows exist for KO but are withheld")


def test_ivi_value_rows_carry_no_run_lineage():
    # The research block keys on "no source_run_id". `ivi value` calls the
    # writer without a run id, and that is what the writer stamps.
    from app.valuation.lineage import valuation_source_lineage

    assert valuation_source_lineage(None) == {
        "source_run_id": None,
        "source_artifact_path": None,
        "source_artifact_sha256": None,
    }
