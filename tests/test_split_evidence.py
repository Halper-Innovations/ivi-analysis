"""Production split-lineage evidence and offline cap-resolution contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.autonomous.financial_integrity import authoritative_split_proof_reference
from app.config import get_config
from app.db import get_db, init_db
from app.market import split_evidence
from app.market.split_evidence import (
    _load_issuer_and_shares,
    load_persisted_split_lineage_quote,
    prepare_relight_split_lineage_evidence,
    produce_split_lineage_evidence,
)
from app.sector.scan import classify_tickers_for_market_cap


def _seed_primary_security(monkeypatch, isolated_data_root: Path) -> None:
    del monkeypatch
    init_db()
    cfg = get_config()
    submissions = cfg.cache_dir / "submissions"
    submissions.mkdir(parents=True, exist_ok=True)
    submissions.joinpath("0000000001.json").write_text(
        json.dumps(
            {
                "tickers": ["LINE"],
                "exchanges": ["NYSE"],
                "filings": {
                    "recent": {
                        "form": ["10-Q"],
                        "filingDate": ["2026-05-15"],
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with get_db() as conn:
        assert {str(row[1]) for row in conn.execute("PRAGMA table_info(companies)").fetchall()} >= {
            "ticker",
            "cik",
        }
        assert {
            str(row[1]) for row in conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
        } >= {
            "ticker",
            "period_end",
            "line_item",
            "value",
            "units",
            "source_url",
            "filed_date",
        }
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('LINE', '0000000001', 'Lineage Fixture', '2026-05-15T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item,
                value, units, source_url, fetched_at, filed_date, form,
                accession
            )
            VALUES(
                'LINE', 2026, 'Q1', '2026-05-01', 'shares_outstanding',
                100.0, 'shares_millions',
                'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                '2026-05-15T00:00:00Z', '2026-05-15', '10-Q',
                '0000000001-26-000001'
            )
            """
        )


def _fixture_fetcher(
    *,
    split_payload: list[dict[str, object]],
    close: float = 40.0,
    adjusted_close: float = 40.0,
):
    calls: list[tuple[str, dict[str, object]]] = []

    def fetch(url: str, params: dict[str, object]) -> bytes:
        calls.append((url, dict(params)))
        if "/api/eod/" in url:
            return json.dumps(
                [
                    {
                        "date": "2026-07-28",
                        "close": close,
                        "adjusted_close": adjusted_close,
                    }
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        if "/api/splits/" in url:
            return json.dumps(split_payload, separators=(",", ":")).encode("utf-8")
        raise AssertionError(f"unexpected fixture URL: {url}")

    return fetch, calls


def _insert_share_row(
    conn,
    *,
    ticker: str,
    cik: str,
    source_url: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO companyfacts_facts(
            ticker, fiscal_year, period_type, period_end, line_item,
            value, units, source_url, fetched_at, filed_date, form,
            accession
        )
        VALUES(
            ?, 2026, 'Q1', '2026-05-01', 'shares_outstanding',
            100.0, 'shares_millions', ?,
            '2026-05-15T00:00:00Z', '2026-05-15', '10-Q', ?
        )
        """,
        (
            ticker,
            source_url
            if source_url is not None
            else f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            f"{cik}-26-000001",
        ),
    )


def _insert_registrant(
    conn,
    *,
    cik: str,
    primary_ticker: str,
    all_tickers: list[str],
) -> None:
    conn.execute(
        """
        INSERT INTO sec_registrants(
            cik, primary_ticker, all_tickers, exchange_scope,
            operating_status, first_seen_at, last_seen_at
        )
        VALUES(?, ?, ?, 'NYSE', 'active', '2026-05-15T00:00:00Z',
               '2026-07-28T00:00:00Z')
        """,
        (cik, primary_ticker, json.dumps(all_tickers)),
    )


def test_issuer_cik_fallbacks_are_source_specific_and_aliases_are_exact(
    isolated_data_root,
):
    init_db()
    cfg = get_config()
    with get_db() as conn:
        assert {str(row[1]) for row in conn.execute("PRAGMA table_info(companies)").fetchall()} >= {
            "ticker",
            "cik",
        }
        assert {
            str(row[1]) for row in conn.execute("PRAGMA table_info(companyfacts_facts)").fetchall()
        } >= {"ticker", "source_url"}
        assert {
            str(row[1]) for row in conn.execute("PRAGMA table_info(sec_registrants)").fetchall()
        } >= {"cik", "primary_ticker", "all_tickers"}
        _insert_share_row(conn, ticker="SOURCE", cik="0000000001")
        _insert_share_row(
            conn,
            ticker="COMPANY",
            cik="0000000002",
            source_url="https://example.test/company-shares",
        )
        _insert_share_row(
            conn,
            ticker="PRIMARY",
            cik="0000000003",
            source_url="https://example.test/primary-shares",
        )
        _insert_share_row(
            conn,
            ticker="ALIAS",
            cik="0000000004",
            source_url="https://example.test/alias-shares",
        )
        _insert_share_row(
            conn,
            ticker="FUZZY",
            cik="0000000005",
            source_url="https://example.test/fuzzy-shares",
        )
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('COMPANY', '0000000002', 'Company Fixture',
                   '2026-05-15T00:00:00Z')
            """
        )
        _insert_registrant(
            conn,
            cik="0000000003",
            primary_ticker="PRIMARY",
            all_tickers=["PRIMARY"],
        )
        _insert_registrant(
            conn,
            cik="0000000004",
            primary_ticker="ROOT",
            all_tickers=["ROOT", "ALIAS"],
        )
        _insert_registrant(
            conn,
            cik="0000000005",
            primary_ticker="OTHER",
            all_tickers=["PREFIXFUZZYSUFFIX"],
        )

    expected = {
        "SOURCE": ("0000000001", "companyfacts_source_url"),
        "COMPANY": ("0000000002", "companies"),
        "PRIMARY": ("0000000003", "sec_registrants_primary_ticker"),
        "ALIAS": ("0000000004", "sec_registrants_all_tickers_exact"),
    }
    for ticker, (cik, source) in expected.items():
        loaded = _load_issuer_and_shares(
            ticker=ticker,
            as_of_date="2026-07-28",
            db_path=cfg.db_path,
            cfg=cfg,
        )
        assert loaded is not None
        assert loaded["issuer_cik"] == cik
        assert loaded["issuer_cik_source"] == source

    assert (
        _load_issuer_and_shares(
            ticker="FUZZY",
            as_of_date="2026-07-28",
            db_path=cfg.db_path,
            cfg=cfg,
        )
        is None
    )


def test_parent_fetcher_scopes_eodhd_without_mutating_safe_mode(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    get_config.cache_clear()
    cfg = get_config().model_copy(update={"safe_mode": True, "eodhd_apikey": "fixture-token"})
    clients = []

    class FixtureHttpClient:
        def __init__(self, client_cfg):
            clients.append(client_cfg)

        def get_bytes(self, url, **_kwargs):
            if "/api/eod/" in url:
                return b'[{"date":"2026-07-28","close":40.0,"adjusted_close":40.0}]'
            if "/api/splits/" in url:
                return b"[]"
            raise AssertionError(f"unexpected fixture URL: {url}")

    monkeypatch.setattr(split_evidence, "HttpClient", FixtureHttpClient)
    produced = produce_split_lineage_evidence(
        ticker="LINE",
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        retrieved_on="2026-07-28",
    )

    assert produced["status"] == "READY"
    assert produced["issuer_cik_source"] == "companyfacts_source_url"
    assert cfg.safe_mode is True
    assert len(clients) == 1
    assert clients[0].safe_mode is False
    assert clients[0].eodhd_apikey == "fixture-token"
    assert os.environ["VOE_SAFE_MODE"] == "true"


def test_relight_fixture_corpus_preserves_three_name_fail_closed_remainder(
    isolated_data_root,
):
    init_db()
    cfg = get_config()
    ready_tickers = ["DAN", "LKQ", "PHIN", *[f"R{index:03d}" for index in range(1, 80)]]
    missing_tickers = ["APWC", "BILI", "VNOM"]
    with get_db() as conn:
        for index, ticker in enumerate(ready_tickers, start=1):
            _insert_share_row(conn, ticker=ticker, cik=f"{index:010d}")
        conn.execute(
            """
            INSERT INTO price_quotes(
                ticker, provider, as_of_date, price, currency, price_basis,
                split_adjustment_factor, split_effective_date, source_url,
                status, fetched_at, expires_at, raw_json, quote_hash
            )
            VALUES(
                'DAN', 'eodhd_split_lineage', '2026-07-28', 1.0, 'USD',
                'UNADJUSTED', 1.0, NULL, 'https://eodhd.com/api/eod/DAN.US',
                'OK', '2026-07-28T00:00:00Z', '2026-07-29T00:00:00Z',
                '{}', 'preexisting-fixture-row'
            )
            """
        )

    fetch, calls = _fixture_fetcher(split_payload=[])
    prepared = prepare_relight_split_lineage_evidence(
        tickers=[*ready_tickers, *missing_tickers],
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=fetch,
        retrieved_on="2026-07-28",
    )

    assert len(ready_tickers) == 82
    assert prepared["requested"] == 85
    assert prepared["ready"] == 82
    assert prepared["unknown"] == 3
    assert [
        result["ticker"] for result in prepared["results"] if result["status"] == "UNKNOWN"
    ] == ["APWC", "BILI", "VNOM"]
    assert {
        result["reason"] for result in prepared["results"] if result["status"] == "UNKNOWN"
    } == {"ISSUER_OR_SHARES_MISSING"}
    assert len(calls) == 164
    assert all(missing not in url for missing in missing_tickers for url, _params in calls)
    with get_db() as conn:
        assert {
            str(row[1]) for row in conn.execute("PRAGMA table_info(price_quotes)").fetchall()
        } >= {"ticker", "provider", "as_of_date", "price", "raw_json"}
        dan = conn.execute(
            """
            SELECT price, raw_json
            FROM price_quotes
            WHERE ticker = 'DAN'
              AND provider = 'eodhd_split_lineage'
              AND as_of_date = '2026-07-28'
            """
        ).fetchone()
    assert dan is not None
    assert dan["price"] == 40.0
    assert json.loads(dan["raw_json"])["record_type"] == "proof_carrying_price_quote"


def test_materialized_no_split_evidence_resolves_offline_with_full_lineage(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    cfg = get_config()

    before = classify_tickers_for_market_cap(
        tickers=["LINE"],
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        pipeline_version="v1",
        allow_live_market_data=False,
        cfg=cfg,
    )["LINE"].to_dict()
    assert before["cap_source"] == "unknown"
    assert before["market_cap_mm"] is None
    assert before["detail"] == "shares_known_price_missing"

    fetch, calls = _fixture_fetcher(split_payload=[])
    produced = produce_split_lineage_evidence(
        ticker="LINE",
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=fetch,
        retrieved_on="2026-07-28",
    )
    assert produced["status"] == "READY"
    assert produced["issuer_cik"] == "0000000001"
    assert produced["shares_period_end"] == "2026-05-01"
    assert produced["quote_as_of_date"] == "2026-07-28"
    assert produced["price_basis"] == "UNADJUSTED"
    assert produced["proof_kind"] == "NO_INTERVENING_SPLIT"
    assert calls == [
        (
            "https://eodhd.com/api/eod/LINE.US",
            {
                "api_token": "",
                "from": "2026-07-21",
                "to": "2026-07-28",
                "fmt": "json",
            },
        ),
        (
            "https://eodhd.com/api/splits/LINE",
            {
                "api_token": "",
                "from": "2026-05-01",
                "to": "2026-07-28",
                "fmt": "json",
            },
        ),
    ]

    persisted = load_persisted_split_lineage_quote(
        "LINE",
        "2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
    )
    assert persisted is not None
    proof = persisted["no_intervening_split_proof"]
    assert proof["status"] == "PASS"
    assert proof["period_start"] == "2026-05-01"
    assert proof["period_end"] == "2026-07-28"
    assert proof["verified_as_of"] == "2026-07-28"
    assert proof["retrieved_at"] == "2026-07-28"
    assert proof["source"] == "eodhd_splits"
    assert proof["source_reference"] == "https://eodhd.com/api/splits/LINE"
    assert proof["source_provider"] == "eodhd"
    assert proof["provider_symbol"] == "LINE"
    assert proof["provider_response_sha256"] == (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )
    assert authoritative_split_proof_reference(
        proof,
        expected_ticker="LINE",
        expected_issuer_cik="0000000001",
        expected_as_of_date="2026-07-28",
    )

    after = classify_tickers_for_market_cap(
        tickers=["LINE"],
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        pipeline_version="v1",
        allow_live_market_data=False,
        cfg=cfg,
    )["LINE"].to_dict()
    assert after["cap_source"] == "stale_shares"
    assert after["cap_band"] == "mid"
    assert after["market_cap_mm"] == 4_000.0
    assert after["price_used"] == 40.0
    assert after["price_as_of_date"] == "2026-07-28"
    assert after["price_source"] == "eodhd_split_lineage"
    assert after["price_source_url"] == "https://eodhd.com/api/eod/LINE.US"
    assert after["price_currency"] == "USD"
    assert after["price_basis"] == "UNADJUSTED"
    assert after["raw_price"] == 40.0
    assert after["shares_mm"] == 100.0
    assert after["shares_period_end"] == "2026-05-01"
    assert after["shares_basis"] == "UNADJUSTED"
    assert after["raw_shares_outstanding_mm"] == 100.0
    assert after["split_adjustment_factor"] == 1.0
    assert after["split_effective_date"] is None
    assert after["split_lineage_proof"]["materialized_sha256"] == (proof["materialized_sha256"])


def test_intervening_split_without_filed_lineage_stays_unknown(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    cfg = get_config()
    fetch, _calls = _fixture_fetcher(
        split_payload=[{"date": "2026-06-01", "split": "2/1"}],
        close=40.0,
        adjusted_close=20.0,
    )

    produced = produce_split_lineage_evidence(
        ticker="LINE",
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=fetch,
        retrieved_on="2026-07-28",
    )
    assert produced == {
        "ticker": "LINE",
        "status": "UNKNOWN",
        "reason": "INTERVENING_SPLIT_PROOF_INCOMPLETE",
    }
    assert (
        load_persisted_split_lineage_quote(
            "LINE",
            "2026-07-28",
            db_path=cfg.db_path,
            cfg=cfg,
        )
        is None
    )


def test_materialized_split_event_normalizes_shares_offline(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    cfg = get_config()
    fetch, _calls = _fixture_fetcher(
        split_payload=[
            {
                "date": "2026-06-01",
                "factor": 2.0,
                "filed_date": "2026-05-31",
            }
        ],
        close=40.0,
        adjusted_close=20.0,
    )

    produced = produce_split_lineage_evidence(
        ticker="LINE",
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=fetch,
        retrieved_on="2026-07-28",
    )
    assert produced["status"] == "READY"
    assert produced["price_basis"] == "SPLIT_ADJUSTED"
    assert produced["proof_kind"] == "SPLIT_EVENT"

    classification = classify_tickers_for_market_cap(
        tickers=["LINE"],
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        pipeline_version="v1",
        allow_live_market_data=False,
        cfg=cfg,
    )["LINE"].to_dict()
    assert classification["cap_source"] == "stale_shares"
    assert classification["cap_band"] == "mid"
    assert classification["market_cap_mm"] == 4_000.0
    assert classification["price_used"] == 20.0
    assert classification["raw_price"] == 40.0
    assert classification["price_basis"] == "SPLIT_ADJUSTED"
    assert classification["shares_mm"] == 200.0
    assert classification["raw_shares_outstanding_mm"] == 100.0
    assert classification["shares_basis"] == "SPLIT_ADJUSTED"
    assert classification["split_adjustment_factor"] == 2.0
    assert classification["split_effective_date"] == "2026-06-01"
    assert classification["split_lineage_proof"]["filed_date"] == "2026-05-31"
    assert classification["split_lineage_proof"]["proof_kind"] == "SPLIT_EVENT"


def test_valid_persisted_proof_reuses_before_late_retrieval_gate(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    cfg = get_config()
    fetch, calls = _fixture_fetcher(split_payload=[])
    first = produce_split_lineage_evidence(
        ticker="LINE",
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=fetch,
        retrieved_on="2026-07-28",
    )
    assert calls == [
        (
            "https://eodhd.com/api/eod/LINE.US",
            {
                "api_token": "",
                "from": "2026-07-21",
                "to": "2026-07-28",
                "fmt": "json",
            },
        ),
        (
            "https://eodhd.com/api/splits/LINE",
            {
                "api_token": "",
                "from": "2026-05-01",
                "to": "2026-07-28",
                "fmt": "json",
            },
        ),
    ]
    forbidden_calls: list[str] = []

    def forbidden_fetch(url, _params):
        forbidden_calls.append(url)
        raise AssertionError("valid persisted proof must bypass the fetcher")

    reused = produce_split_lineage_evidence(
        ticker="LINE",
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=forbidden_fetch,
        retrieved_on="2026-07-29",
    )

    assert reused == first
    assert reused["ticker"] == "LINE"
    assert reused["status"] == "READY"
    assert reused["issuer_cik"] == "0000000001"
    assert reused["issuer_cik_source"] == "companyfacts_source_url"
    assert reused["shares_period_end"] == "2026-05-01"
    assert reused["quote_as_of_date"] == "2026-07-28"
    assert reused["price_basis"] == "UNADJUSTED"
    assert reused["proof_kind"] == "NO_INTERVENING_SPLIT"
    assert forbidden_calls == []
    persisted = load_persisted_split_lineage_quote(
        "LINE",
        "2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
    )
    assert persisted is not None
    proof = persisted["no_intervening_split_proof"]
    assert proof["retrieved_at"] == "2026-07-28"
    assert proof["verified_as_of"] == "2026-07-28"


def test_no_persisted_proof_retrieved_after_run_asof_halts_before_network(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    cfg = get_config()

    def forbidden_fetch(_url, _params):
        raise AssertionError("historical proof must halt before network")

    assert (
        load_persisted_split_lineage_quote(
            "LINE",
            "2026-07-28",
            db_path=cfg.db_path,
            cfg=cfg,
        )
        is None
    )
    assert produce_split_lineage_evidence(
        ticker="LINE",
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=forbidden_fetch,
        retrieved_on="2026-07-29",
    ) == {
        "ticker": "LINE",
        "status": "UNKNOWN",
        "reason": "PROOF_RETRIEVED_AFTER_RUN_AS_OF",
    }


def test_invalid_persisted_proof_never_bypasses_late_retrieval_gate(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    cfg = get_config()
    fetch, _calls = _fixture_fetcher(split_payload=[])
    assert (
        produce_split_lineage_evidence(
            ticker="LINE",
            as_of_date="2026-07-28",
            db_path=cfg.db_path,
            cfg=cfg,
            fetch_bytes=fetch,
            retrieved_on="2026-07-28",
        )["status"]
        == "READY"
    )
    persisted = load_persisted_split_lineage_quote(
        "LINE",
        "2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
    )
    assert persisted is not None
    proof = persisted["no_intervening_split_proof"]
    response_path = (
        cfg.cache_dir / "corporate_actions" / "v1" / proof["provider_response_relative_path"]
    )
    response_path.write_bytes(b"[ ]")
    forbidden_calls: list[str] = []

    def forbidden_fetch(url, _params):
        forbidden_calls.append(url)
        raise AssertionError("late invalid persisted proof must halt before network")

    assert produce_split_lineage_evidence(
        ticker="LINE",
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=forbidden_fetch,
        retrieved_on="2026-07-29",
    ) == {
        "ticker": "LINE",
        "status": "UNKNOWN",
        "reason": "PROOF_RETRIEVED_AFTER_RUN_AS_OF",
    }
    assert forbidden_calls == []


def test_prepare_relight_split_lineage_evidence_is_idempotent(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    cfg = get_config()
    fetch, calls = _fixture_fetcher(split_payload=[])
    first = prepare_relight_split_lineage_evidence(
        tickers=["LINE"],
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=fetch,
        retrieved_on="2026-07-28",
    )
    assert first["as_of_date"] == "2026-07-28"
    assert first["requested"] == 1
    assert first["ready"] == 1
    assert first["unknown"] == 0
    assert first["results"][0]["ticker"] == "LINE"
    assert first["results"][0]["status"] == "READY"
    assert len(calls) == 2
    second_fetch_calls: list[str] = []

    def forbidden_fetch(url, _params):
        second_fetch_calls.append(url)
        raise AssertionError("idempotent second pass must not fetch")

    second = prepare_relight_split_lineage_evidence(
        tickers=["LINE"],
        as_of_date="2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
        fetch_bytes=forbidden_fetch,
        retrieved_on="2026-07-29",
    )

    assert second == first
    assert second_fetch_calls == []


def test_offline_reader_rejects_tampered_exact_provider_response(
    monkeypatch,
    isolated_data_root,
):
    _seed_primary_security(monkeypatch, isolated_data_root)
    cfg = get_config()
    fetch, _calls = _fixture_fetcher(split_payload=[])
    assert (
        produce_split_lineage_evidence(
            ticker="LINE",
            as_of_date="2026-07-28",
            db_path=cfg.db_path,
            cfg=cfg,
            fetch_bytes=fetch,
            retrieved_on="2026-07-28",
        )["status"]
        == "READY"
    )
    persisted = load_persisted_split_lineage_quote(
        "LINE",
        "2026-07-28",
        db_path=cfg.db_path,
        cfg=cfg,
    )
    assert persisted is not None
    proof = persisted["no_intervening_split_proof"]
    response_path = (
        cfg.cache_dir / "corporate_actions" / "v1" / proof["provider_response_relative_path"]
    )
    response_path.write_bytes(b"[ ]")

    assert (
        load_persisted_split_lineage_quote(
            "LINE",
            "2026-07-28",
            db_path=cfg.db_path,
            cfg=cfg,
        )
        is None
    )
