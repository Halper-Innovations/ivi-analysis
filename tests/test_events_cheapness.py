from __future__ import annotations

import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.db import get_db, init_db
from app.events import store
from app.events.cheapness import (
    FLAG_MEZZANINE,
    KNOWN_EVENTS,
    LLM_UNAVAILABLE,
    NO_KNOWN_EVENT,
    build_cheapness_report,
    cheapness_headline,
    extract_risk_sections,
    filings_fingerprint,
    latest_cheapness_by_ticker,
    mezzanine_check,
    render_cheapness_block,
)
from app.watchlist.schema import ensure_watchlist_schema


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """Cheapness behavior tests use synthetic cap inputs."""

    class _Scope:
        def __init__(self, *, context, run_as_of_date, packets, scenarios):
            self.context = context
            self.run_as_of_date = run_as_of_date
            self.packets = tuple(packets)
            self.scenarios = tuple(scenarios)
            payload = json.dumps(
                list(scenarios),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            self.expected_scope_fingerprint = hashlib.sha256(payload).hexdigest()

        def require(self, **_kwargs):
            return SimpleNamespace(
                passed=True,
                scope_fingerprint=self.expected_scope_fingerprint,
            )

    def _context(*, tickers, **_kwargs):
        return SimpleNamespace(
            packets={
                ticker: {
                    "ticker": ticker,
                    "market_cap_mm": 141.03,
                    "market_cap_unit": "USD_millions",
                }
                for ticker in tickers
            }
        )

    monkeypatch.setattr(
        "app.events.cheapness.build_canonical_v1_financial_context",
        _context,
    )
    monkeypatch.setattr(
        "app.events.cheapness.bind_v1_financial_scope",
        lambda **kwargs: _Scope(
            context=kwargs["context"],
            run_as_of_date=kwargs["run_as_of_date"],
            packets=kwargs["packets"],
            scenarios=kwargs["scenarios"],
        ),
    )

    def _sector_gate(scope, **_kwargs):
        return SimpleNamespace(
            passed=True,
            status="PASS",
            violations=(),
            scope_fingerprint=scope.expected_scope_fingerprint,
        )

    # The strict cost wrapper reuses the production sector provider adapter,
    # whose integrity hook sees the same synthetic scope. Keep this fixture's
    # deliberate test-only authorization isolated from provider parsing too.
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.require_financial_integrity_scope",
        _sector_gate,
    )
    monkeypatch.setattr(
        "app.autonomous.sector_runtime.require_unchanged_financial_integrity_scope",
        lambda scope, **kwargs: _sector_gate(scope, **kwargs),
    )


def _init(monkeypatch, tmp_path):
    monkeypatch.setenv("VOE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOE_DB_PATH", str(tmp_path / "data" / "engine.db"))
    get_config.cache_clear()
    init_db()
    ensure_watchlist_schema()


# A BOOM-shaped raw companyfacts excerpt: redeemable NCI carrying amount of
# $187.0M against a ~$141M market cap (the Arcadia put class). Tests that
# pass as_of="2025-06-11" treat these as live; the staleness tests use their
# own fixtures.
BOOM_LIKE_FACTS = {
    "cik": 34067,
    "facts": {
        "us-gaap": {
            "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                "units": {
                    "USD": [
                        {
                            "end": "2024-12-31",
                            "val": 184_000_000,
                            "accn": "0000034067-25-000010",
                            "filed": "2025-02-24",
                        },
                        {
                            "end": "2025-03-31",
                            "val": 187_000_000,
                            "accn": "0000034067-25-000030",
                            "filed": "2025-05-01",
                        },
                    ]
                }
            }
        }
    },
}


def test_mezzanine_check_fires_above_threshold():
    result = mezzanine_check(BOOM_LIKE_FACTS, 141.03, as_of="2025-06-11")
    assert result["flag"] == FLAG_MEZZANINE
    assert result["mezzanine_total_mm"] == 187.0
    assert result["ratio_to_cap"] == round(187.0 / 141.03, 4)
    assert result["components"][0]["concept"] == (
        "RedeemableNoncontrollingInterestEquityCarryingAmount"
    )
    assert result["components"][0]["as_of"] == "2025-03-31"  # latest instant wins
    assert result["components"][0]["filed_date"] == "2025-05-01"
    assert result["components"][0]["reported_unit"] == "USD"
    assert result["components"][0]["unit"] == "USD_millions"
    assert result["components"][0]["accession"] == "0000034067-25-000030"
    assert result["components"][0]["source_url"] == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000034067.json"
    )
    assert result["source_references"] == [
        (
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000034067.json"
            "#us-gaap:RedeemableNoncontrollingInterestEquityCarryingAmount:"
            "accession=0000034067-25-000030"
        )
    ]


def test_mezzanine_check_clean_record():
    result = mezzanine_check({"facts": {"us-gaap": {}}}, 100.0, as_of="2025-06-11")
    assert result["flag"] is None
    assert result["status"] == "NO_MEZZANINE_CONCEPTS"


def test_mezzanine_check_below_threshold_no_flag():
    facts = {
        "cik": 444444,
        "facts": {
            "us-gaap": {
                "TemporaryEquityCarryingAmountAttributableToParent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-03-31",
                                "val": 10_000_000,
                                "accn": "x",
                                "filed": "2025-05-01",
                            }
                        ]
                    }
                }
            }
        },
    }
    result = mezzanine_check(facts, 100.0, as_of="2025-06-11")
    assert result["flag"] is None
    assert result["ratio_to_cap"] == 0.1


def test_mezzanine_umbrella_concept_prevents_double_count():
    facts = {
        "cik": 444444,
        "facts": {
            "us-gaap": {
                "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-03-31",
                                "val": 50_000_000,
                                "accn": "a",
                                "filed": "2025-05-01",
                            }
                        ]
                    }
                },
                "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-03-31",
                                "val": 40_000_000,
                                "accn": "b",
                                "filed": "2025-05-01",
                            }
                        ]
                    }
                },
            }
        },
    }
    result = mezzanine_check(facts, 100.0, as_of="2025-06-11")
    assert result["mezzanine_total_mm"] == 50.0  # umbrella only, not 90


def test_mezzanine_no_market_cap():
    result = mezzanine_check(BOOM_LIKE_FACTS, None, as_of="2025-06-11")
    assert result["status"] == "NO_MARKET_CAP"
    assert result["flag"] is None


def test_mezzanine_selects_latest_fact_filed_by_as_of():
    facts = {
        "cik": 34067,
        "facts": {
            "us-gaap": {
                "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                    "units": {
                        "USD": [
                            {
                                "end": "2023-12-31",
                                "val": 20_000_000,
                                "accn": "0000034067-24-000010",
                                "filed": "2024-02-15",
                            },
                            {
                                "end": "2024-03-31",
                                "val": 90_000_000,
                                "accn": "0000034067-24-000030",
                                "filed": "2024-05-01",
                            },
                        ]
                    }
                }
            }
        },
    }

    result = mezzanine_check(facts, 100.0, as_of="2024-04-15")

    assert result["status"] == "OK"
    assert result["mezzanine_total_mm"] == 20.0
    assert result["ratio_to_cap"] == 0.2
    assert result["components"][0]["period_end"] == "2023-12-31"
    assert result["components"][0]["filed_date"] == "2024-02-15"
    assert result["components"][0]["accession"] == "0000034067-24-000010"


@pytest.mark.parametrize(
    ("payload", "reason_code"),
    [
        (
            {
                "cik": 34067,
                "facts": {
                    "us-gaap": {
                        "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                            "units": {
                                "EUR": [
                                    {
                                        "end": "2024-03-31",
                                        "val": 999_000_000,
                                        "accn": "0000034067-24-000030",
                                        "filed": "2024-05-01",
                                    }
                                ]
                            }
                        }
                    }
                },
            },
            "MISSING_LITERAL_USD_UNIT",
        ),
        (
            {
                "cik": 34067,
                "facts": {
                    "us-gaap": {
                        "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                            "units": {
                                "USD": [
                                    {
                                        "end": "2024-03-31",
                                        "val": 999_000_000,
                                        "accn": "",
                                        "filed": "2024-05-01",
                                    }
                                ]
                            }
                        }
                    }
                },
            },
            "MISSING_ACCESSION",
        ),
        (
            {
                "facts": {
                    "us-gaap": {
                        "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                            "units": {
                                "USD": [
                                    {
                                        "end": "2024-03-31",
                                        "val": 999_000_000,
                                        "accn": "0000034067-24-000030",
                                        "filed": "2024-05-01",
                                    }
                                ]
                            }
                        }
                    }
                },
            },
            "MISSING_SOURCE_URL",
        ),
    ],
)
def test_mezzanine_missing_exact_provenance_fails_closed(payload, reason_code):
    result = mezzanine_check(payload, 100.0, as_of="2024-06-01")

    assert result["status"] == "NEEDS_DATA"
    assert result["reason_codes"] == [reason_code]
    assert result["flag"] is None
    assert result["mezzanine_total_mm"] is None
    assert result["components"] == []
    assert result["source_references"] == []
    assert "999000000" not in json.dumps(result)


def test_mezzanine_stale_concepts_never_count():
    """The ACEL/BTM class: extinguished de-SPAC-era temporary equity keeps its
    last-ever value in companyfacts forever — it must not sum with live
    concepts or fire the flag."""
    facts = {
        "cik": 1698991,
        "facts": {
            "us-gaap": {
                # Dead since 2018 (pre-de-SPAC temporary equity).
                "TemporaryEquityCarryingAmountAttributableToParent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2018-06-30",
                                "val": 432_361_000,
                                "accn": "0001564590-18-019168",
                                "filed": "2018-08-09",
                            }
                        ]
                    }
                },
                # Live, small.
                "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                    "units": {
                        "USD": [
                            {
                                "end": "2026-03-31",
                                "val": 4_070_000,
                                "accn": "0001698991-26-000037",
                                "filed": "2026-05-01",
                            }
                        ]
                    }
                },
            }
        },
    }
    result = mezzanine_check(facts, 1071.13, as_of="2026-06-12")
    assert result["flag"] is None
    assert result["mezzanine_total_mm"] == 4.07
    assert result["ratio_to_cap"] == round(4.07 / 1071.13, 4)
    assert [c["concept"] for c in result["stale_components"]] == [
        "TemporaryEquityCarryingAmountAttributableToParent",
    ]


def test_mezzanine_only_stale_concepts_no_flag():
    facts = {
        "cik": 1193125,
        "facts": {
            "us-gaap": {
                "TemporaryEquityCarryingAmountAttributableToParent": {
                    "units": {
                        "USD": [
                            {
                                "end": "2023-03-31",
                                "val": 326_647_000,
                                "accn": "0001193125-23-136197",
                                "filed": "2023-05-15",
                            }
                        ]
                    }
                },
            }
        },
    }
    result = mezzanine_check(facts, 28.16, as_of="2026-06-12")
    assert result["flag"] is None
    assert result["status"] == "NO_MEZZANINE_CONCEPTS"
    assert len(result["stale_components"]) == 1


# Live-dated variant for build_cheapness_report tests run as_of 2026-06-11.
LIVE_MEZZ_FACTS = {
    "cik": 34067,
    "facts": {
        "us-gaap": {
            "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                "units": {
                    "USD": [
                        {
                            "end": "2026-03-31",
                            "val": 187_000_000,
                            "accn": "0000034067-26-000036",
                            "filed": "2026-05-01",
                        },
                    ]
                }
            }
        }
    },
}


class FakeAdapter:
    def __init__(self, filings):
        self.filings = filings
        self.calls = 0

    def filings_window(self, cik, *, start, end):
        self.calls += 1
        return [f for f in self.filings if start <= f["filing_date"] <= end]


class FakeClient:
    def __init__(self, docs=None):
        self.docs = docs or {}

    def download_bytes(self, url, *, use_cache=True):
        for fragment, payload in self.docs.items():
            if fragment in url:
                return payload
        raise RuntimeError(f"no fixture for {url}")


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.prompts = []

    def synthesize_json(self, *, prompt, schema, schema_name=None):
        self.calls += 1
        self.prompts.append(prompt)
        from app.llm.providers.disabled_provider import LLMResult

        return LLMResult(
            json_text=json.dumps(self.payload),
            model="fake-model",
            usage_input_tokens=10,
            usage_output_tokens=10,
            raw={},
        )


FILINGS = [
    {
        "accession": "0000444444-26-000030",
        "form": "10-Q",
        "filing_date": "2026-05-01",
        "items": "",
        "primary_document": "tenq.htm",
    },
    {
        "accession": "0000444444-26-000010",
        "form": "8-K",
        "filing_date": "2026-02-10",
        "items": "2.02,9.01",
        "primary_document": "eightk.htm",
    },
]


def _seed_registrant(conn, ticker="TEST", cik="0000444444"):
    conn.execute(
        """
        INSERT INTO sec_registrants(
            cik, primary_ticker, exchange_scope, operating_status,
            first_seen_at, last_seen_at, name)
        VALUES(?, ?, 'listed', 'operating', 'x', 'x', 'TEST CO')
        """,
        (cik, ticker),
    )


def test_strict_llm_reservation_uses_utf8_byte_upper_bound(monkeypatch):
    import app.autonomous.sector_runtime as sector_runtime

    captured: dict[str, object] = {}

    def capture_cost(**kwargs):
        captured.update(kwargs)
        return 0.012345

    monkeypatch.setattr(sector_runtime, "_estimate_llm_cost_usd", capture_cost)
    provider = SimpleNamespace(provider_name="test")

    estimated = sector_runtime._estimated_llm_call_cost_usd(
        provider,
        {"prompt": "é", "max_output_tokens": 7},
    )

    assert estimated == 0.012345
    assert captured["input_tokens"] == 1026
    assert captured["output_tokens"] == 7


def test_cheapness_cost_overrun_is_durably_rejected_and_cannot_be_reused(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    import app.events.cheapness as cheapness_module

    attempt_request = {
        "attempt_id": "a" * 64,
        "request_sha256": "b" * 64,
        "provider": "test",
        "model": "literal-model",
        "estimated_cost_usd": 0.01,
    }
    llm = {
        "verdict": NO_KNOWN_EVENT,
        "bullets": [],
        "model": "literal-model",
        "usage": {"call_count": 1},
        "cost_usd": 0.02,
        "budget_usd": 0.05,
    }

    with get_db() as conn:
        _seed_registrant(conn)
        reusable, status = cheapness_module._reserve_cheapness_paid_attempt(
            conn,
            attempt_request=attempt_request,
            ticker="TEST",
            cik="0000444444",
            as_of="2026-06-11",
            filings_fingerprint_value="literal-filings",
            scope_fingerprint="c" * 64,
        )
        assert reusable is None
        assert status == "RESERVED"

        with pytest.raises(cheapness_module.CheapnessCostIntegrityError, match="actual"):
            cheapness_module._complete_cheapness_paid_attempt(
                conn,
                attempt_id="a" * 64,
                llm=llm,
            )
        row = conn.execute("SELECT * FROM cheapness_paid_attempts").fetchone()
        assert row["status"] == "REJECTED"
        assert row["accounted_cost_usd"] == 0.02

        with pytest.raises(
            cheapness_module.CheapnessPaidAttemptAmbiguousError,
            match="rejected paid response",
        ):
            cheapness_module._reserve_cheapness_paid_attempt(
                conn,
                attempt_request=attempt_request,
                ticker="TEST",
                cik="0000444444",
                as_of="2026-06-11",
                filings_fingerprint_value="literal-filings",
                scope_fingerprint="c" * 64,
            )


def test_build_report_caches_by_fingerprint(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    adapter = FakeAdapter(FILINGS)
    provider = FakeProvider(
        {
            "verdict": KNOWN_EVENTS,
            "bullets": [
                {
                    "text": "Guidance cut disclosed in results 8-K.",
                    "accessions": ["0000444444-26-000010"],
                },
            ],
        }
    )
    client = FakeClient({"tenq.htm": b"<html>no sections here</html>"})
    with get_db() as conn:
        _seed_registrant(conn)
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=client,
            adapter=adapter,
            provider=provider,
            market_cap_mm=100.0,
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )
        assert report["status"] == "OK"
        assert report["from_cache"] is False
        assert report["llm_verdict"] == KNOWN_EVENTS
        assert report["llm_bullets"] == [
            {
                "text": "Guidance cut disclosed in results 8-K.",
                "accessions": ["0000444444-26-000010"],
            },
        ]
        assert provider.calls == 1

        # Same filing set -> served from cache, no second LLM call.
        report2 = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=client,
            adapter=adapter,
            provider=provider,
            market_cap_mm=100.0,
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )
        assert report2["from_cache"] is True
        assert provider.calls == 1

        # A new filing changes the fingerprint -> re-run.
        adapter.filings.insert(
            0,
            {
                "accession": "0000444444-26-000050",
                "form": "8-K",
                "filing_date": "2026-06-10",
                "items": "1.01",
                "primary_document": "k2.htm",
            },
        )
        report3 = build_cheapness_report(
            "TEST",
            as_of="2026-06-12",
            conn=conn,
            client=client,
            adapter=adapter,
            provider=provider,
            market_cap_mm=100.0,
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )
        assert report3["from_cache"] is False
        assert provider.calls == 2


def test_cache_miss_when_canonical_financial_scope_changes(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    adapter = FakeAdapter(FILINGS)
    provider = FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []})
    with get_db() as conn:
        _seed_registrant(conn)
        first = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=adapter,
            provider=provider,
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )
        first_row = dict(
            conn.execute("SELECT * FROM cheapness_reports WHERE ticker = 'TEST'").fetchone()
        )

        monkeypatch.setattr(
            "app.events.cheapness.build_canonical_v1_financial_context",
            lambda *, tickers, **_kwargs: SimpleNamespace(
                packets={
                    ticker: {
                        "ticker": ticker,
                        "market_cap_mm": 200.0,
                        "market_cap_unit": "USD_millions",
                    }
                    for ticker in tickers
                }
            ),
        )
        second = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=adapter,
            provider=provider,
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )
        conn.execute(
            """
            UPDATE cheapness_reports
            SET financial_scope_fingerprint = ?,
                financial_scope_publication_fingerprint = ?,
                publication_source_sha256 = ?,
                publication_row_sha256 = ?,
                as_of = ?,
                flags_json = ?,
                deterministic_json = ?,
                llm_verdict = ?,
                llm_bullets_json = ?,
                llm_model = ?,
                updated_at = ?
            WHERE ticker = 'TEST'
            """,
            (
                first_row["financial_scope_fingerprint"],
                first_row["financial_scope_publication_fingerprint"],
                first_row["publication_source_sha256"],
                first_row["publication_row_sha256"],
                first_row["as_of"],
                first_row["flags_json"],
                first_row["deterministic_json"],
                first_row["llm_verdict"],
                first_row["llm_bullets_json"],
                first_row["llm_model"],
                first_row["updated_at"],
            ),
        )
        assert latest_cheapness_by_ticker(conn, ["TEST"]) == {}

    assert first["from_cache"] is False
    assert second["from_cache"] is False
    assert first["financial_scope_fingerprint"] != second["financial_scope_fingerprint"]
    assert provider.calls == 2


def test_latest_cheapness_suppresses_legacy_unbound_row(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO cheapness_reports(
                ticker, cik, filings_fingerprint, financial_scope_fingerprint,
                as_of, flags_json, deterministic_json, llm_verdict,
                llm_bullets_json, llm_model, created_at, updated_at
            )
            VALUES(
                'TEST', '0000444444', 'legacy', NULL, '2026-06-11',
                '[]', '{}', 'KNOWN_EVENTS', '[]', 'legacy-model', 'x', 'x'
            )
            """
        )

        assert latest_cheapness_by_ticker(conn, ["TEST"]) == {}


def test_latest_cheapness_rejects_valid_self_attested_hashes_without_ledger(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    import app.events.cheapness as cheapness_module

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO cheapness_reports(
                ticker, cik, filings_fingerprint, financial_scope_fingerprint,
                financial_scope_publication_fingerprint,
                publication_source_sha256, publication_row_sha256,
                as_of, flags_json, deterministic_json, llm_verdict,
                llm_bullets_json, llm_model, created_at, updated_at
            )
            VALUES(
                'TEST', '0000444444', 'fake-filings', ?, ?, ?, ?,
                '2026-06-11', '["MEZZANINE_OBLIGATION"]', '{}',
                'KNOWN_EVENTS',
                '[{"text":"self-attested decision","accessions":["fake"]}]',
                'fake-model', 'x', 'x'
            )
            """,
            ("a" * 64, "a" * 64, "b" * 64, None),
        )
        self_attested_row_sha256 = (
            "5bbddc3003ecd27aaa1fbb7d2a60005ab72edb16a5731f7eb8a40a6e75858571"
        )
        conn.execute(
            """
            UPDATE cheapness_reports
            SET publication_row_sha256 = ?
            WHERE ticker = 'TEST'
            """,
            (self_attested_row_sha256,),
        )
        row = conn.execute("SELECT * FROM cheapness_reports WHERE ticker = 'TEST'").fetchone()
        assert (
            cheapness_module._cheapness_publication_row_sha256(row)
            == "5bbddc3003ecd27aaa1fbb7d2a60005ab72edb16a5731f7eb8a40a6e75858571"
        )

        assert latest_cheapness_by_ticker(conn, ["TEST"]) == {}


def test_cheapness_authorization_ledger_is_append_only(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_registrant(conn)
        build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=FakeAdapter(FILINGS),
            provider=FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []}),
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )

    conn = sqlite3.connect(get_config().db_path)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM cheapness_publication_authorizations").fetchone()[0]
            == 1
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                """
                UPDATE cheapness_publication_authorizations
                SET authorized_at = 'tampered'
                """
            )
        conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM cheapness_publication_authorizations")
        conn.rollback()
        assert (
            conn.execute("SELECT COUNT(*) FROM cheapness_publication_authorizations").fetchone()[0]
            == 1
        )
    finally:
        conn.close()


def test_latest_invalid_cheapness_row_suppresses_older_authorized_row(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        _seed_registrant(conn)
        build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=FakeAdapter(FILINGS),
            provider=FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []}),
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )
        assert "TEST" in latest_cheapness_by_ticker(conn, ["TEST"])
        conn.execute(
            """
            INSERT INTO cheapness_reports(
                ticker, cik, filings_fingerprint, financial_scope_fingerprint,
                financial_scope_publication_fingerprint,
                publication_source_sha256, publication_row_sha256,
                as_of, flags_json, deterministic_json, llm_verdict,
                llm_bullets_json, llm_model, created_at, updated_at
            )
            VALUES(
                'TEST', '0000444444', 'newer-invalid', ?, ?, ?, ?,
                '2026-06-12', '[]', '{}', 'KNOWN_EVENTS', '[]',
                'fake-model', 'y', 'y'
            )
            """,
            ("a" * 64, "a" * 64, "a" * 64, "a" * 64),
        )

        assert latest_cheapness_by_ticker(conn, ["TEST"]) == {}


def test_cheapness_event_mutation_after_provider_response_prevents_persistence(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._estimated_llm_call_cost_usd",
        lambda _provider, _kwargs: 0.01,
    )
    with get_db() as conn:
        _seed_registrant(conn)

        class MutatingProvider(FakeProvider):
            _handles_retry_guard = True
            provider_name = "fakeprovider"

            def synthesize_json(self, **kwargs):
                self.calls += 1
                self.prompts.append(kwargs["prompt"])
                event_id = store.upsert_event(
                    conn,
                    cik="0000444444",
                    event_type="merger",
                    anchor_accession="0000444444-26-000099",
                    company_name="TEST CO",
                    detection_date="2026-06-10",
                    source_mode="daily",
                )
                store.set_ticker(conn, event_id=event_id, ticker="TEST")
                from app.llm.providers.disabled_provider import LLMResult

                return LLMResult(
                    json_text=json.dumps(self.payload),
                    model="fake-model",
                    usage_input_tokens=10,
                    usage_output_tokens=10,
                    raw={},
                )

        provider = MutatingProvider({"verdict": NO_KNOWN_EVENT, "bullets": []})
        with pytest.raises(InvalidFinancialInputError) as exc_info:
            build_cheapness_report(
                "TEST",
                as_of="2026-06-11",
                conn=conn,
                client=FakeClient(),
                adapter=FakeAdapter(FILINGS),
                provider=provider,
                raw_companyfacts={"facts": {"us-gaap": {}}},
            )

        assert provider.calls == 1, str(exc_info.value)
        assert conn.execute("SELECT COUNT(*) FROM cheapness_reports").fetchone()[0] == 0
        attempt = conn.execute("SELECT * FROM cheapness_paid_attempts").fetchone()
        assert attempt is not None
        assert attempt["status"] == "REJECTED"
        assert attempt["accounted_cost_usd"] >= 0.0


def test_cheapness_crash_before_publication_reuses_returned_attempt_without_second_call(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    import app.events.cheapness as cheapness_module

    provider = FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []})
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._estimated_llm_call_cost_usd",
        lambda _provider, _kwargs: 0.04,
    )
    real_publish = cheapness_module._publish_cheapness_report

    def crash_before_publication(**_kwargs):
        raise RuntimeError("simulated crash before publication")

    with get_db() as conn:
        _seed_registrant(conn)
        monkeypatch.setattr(
            cheapness_module,
            "_publish_cheapness_report",
            crash_before_publication,
        )
        with pytest.raises(RuntimeError, match="simulated crash before publication"):
            build_cheapness_report(
                "TEST",
                as_of="2026-06-11",
                conn=conn,
                client=FakeClient(),
                adapter=FakeAdapter(FILINGS),
                provider=provider,
                raw_companyfacts={"facts": {"us-gaap": {}}},
                llm_budget_usd=0.05,
            )
        first_attempt = conn.execute("SELECT * FROM cheapness_paid_attempts").fetchone()
        assert first_attempt is not None
        assert first_attempt["status"] == "RETURNED"
        assert conn.execute("SELECT COUNT(*) FROM cheapness_reports").fetchone()[0] == 0

        monkeypatch.setattr(
            cheapness_module,
            "_publish_cheapness_report",
            real_publish,
        )
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=FakeAdapter(FILINGS),
            provider=provider,
            raw_companyfacts={"facts": {"us-gaap": {}}},
            llm_budget_usd=0.05,
        )
        published_attempt = conn.execute("SELECT * FROM cheapness_paid_attempts").fetchone()

    assert provider.calls == 1
    assert report["paid_attempt_id"] == first_attempt["attempt_id"]
    assert published_attempt["status"] == "PUBLISHED"
    assert published_attempt["publication_row_sha256"] == report["publication_row_sha256"]


def test_cheapness_openai_disables_output_expansion_and_uses_one_physical_call(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)

    class OpenAIStrictProvider:
        provider_name = "openai"

        def __init__(self):
            self.calls = 0
            self.allow_output_token_retry: list[bool | None] = []

        def synthesize_json(self, **kwargs):
            self.calls += 1
            self.allow_output_token_retry.append(kwargs.get("allow_output_token_retry"))
            from app.llm.providers.disabled_provider import LLMResult

            return LLMResult(
                json_text=json.dumps({"verdict": NO_KNOWN_EVENT, "bullets": []}),
                model="fake-openai-model",
                usage_input_tokens=10,
                usage_output_tokens=10,
                raw={},
            )

    provider = OpenAIStrictProvider()
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._estimated_llm_call_cost_usd",
        lambda _provider, _kwargs: 0.04,
    )

    with get_db() as conn:
        _seed_registrant(conn)
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=FakeAdapter(FILINGS),
            provider=provider,
            raw_companyfacts={"facts": {"us-gaap": {}}},
            llm_budget_usd=0.05,
        )
        attempt = conn.execute("SELECT * FROM cheapness_paid_attempts").fetchone()

    assert provider.calls == 1
    assert provider.allow_output_token_retry == [False]
    assert attempt is not None
    assert attempt["status"] == "PUBLISHED"
    assert report["paid_attempt_id"] == attempt["attempt_id"]


def test_cheapness_publication_boundary_blocks_concurrent_event_write(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    import app.events.cheapness as cheapness_module

    real_source_sha256 = cheapness_module._cheapness_publication_source_sha256
    race_outcomes: list[str] = []

    with get_db() as conn:
        _seed_registrant(conn)
        conn.commit()

        def source_sha256_with_concurrent_event(**kwargs):
            race_conn = sqlite3.connect(get_config().db_path, timeout=0)
            try:
                event_id = store.upsert_event(
                    race_conn,
                    cik="0000444444",
                    event_type="merger",
                    anchor_accession="0000444444-26-000097",
                    company_name="TEST CO",
                    detection_date="2026-06-10",
                    source_mode="daily",
                )
                store.set_ticker(race_conn, event_id=event_id, ticker="TEST")
                race_conn.commit()
            except sqlite3.OperationalError as exc:
                race_outcomes.append(str(exc))
                race_conn.rollback()
            else:
                race_outcomes.append("committed")
            finally:
                race_conn.close()
            return real_source_sha256(**kwargs)

        monkeypatch.setattr(
            cheapness_module,
            "_cheapness_publication_source_sha256",
            source_sha256_with_concurrent_event,
        )
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=FakeAdapter(FILINGS),
            provider=FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []}),
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )

        assert report["status"] == "OK"
        assert race_outcomes == ["database is locked"]
        assert store.open_queue_protection_events(conn, ciks=["0000444444"]) == {}


def test_uncited_bullets_dropped(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    adapter = FakeAdapter(FILINGS)
    provider = FakeProvider(
        {
            "verdict": KNOWN_EVENTS,
            "bullets": [
                {"text": "Cited claim.", "accessions": ["0000444444-26-000010"]},
                {"text": "Fabricated claim.", "accessions": ["9999999999-26-999999"]},
                {"text": "No citation at all.", "accessions": []},
            ],
        }
    )
    with get_db() as conn:
        _seed_registrant(conn)
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=adapter,
            provider=provider,
            market_cap_mm=100.0,
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )
        assert len(report["llm_bullets"]) == 1
        assert report["llm_bullets"][0]["text"] == "Cited claim."
        assert report["deterministic"]["llm_dropped_uncited"] == 2


def test_deterministic_flags_force_known_events(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    adapter = FakeAdapter(FILINGS)
    provider = FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []})
    with get_db() as conn:
        _seed_registrant(conn, cik="0000034067")
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=adapter,
            provider=provider,
            market_cap_mm=141.03,
            raw_companyfacts=LIVE_MEZZ_FACTS,
        )
        assert FLAG_MEZZANINE in report["flags"]
        assert report["llm_verdict"] == KNOWN_EVENTS  # overridden
        component = report["deterministic"]["mezzanine"]["components"][0]
        assert component["filed_date"] == "2026-05-01"
        assert component["source_reference"] == (
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000034067.json"
            "#us-gaap:RedeemableNoncontrollingInterestEquityCarryingAmount:"
            "accession=0000034067-26-000036"
        )
        assert component["reported_unit"] == "USD"
        assert component["unit"] == "USD_millions"
        assert component["source_reference"] in provider.prompts[0]


def test_invalid_mezzanine_fact_cannot_enter_prompt_as_overhang(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    provider = FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []})
    invalid_facts = {
        "cik": 34067,
        "facts": {
            "us-gaap": {
                "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                    "units": {
                        "USD": [
                            {
                                "end": "2026-03-31",
                                "val": 999_000_000,
                                "accn": "",
                                "filed": "2026-05-01",
                            }
                        ]
                    }
                }
            }
        },
    }
    with get_db() as conn:
        _seed_registrant(conn, cik="0000034067")
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=FakeAdapter(FILINGS),
            provider=provider,
            raw_companyfacts=invalid_facts,
        )

    mezzanine = report["deterministic"]["mezzanine"]
    assert mezzanine["status"] == "NEEDS_DATA"
    assert mezzanine["reason_codes"] == ["MISSING_ACCESSION"]
    assert report["flags"] == []
    assert report["llm_verdict"] == NO_KNOWN_EVENT
    assert '"reported_value"' not in provider.prompts[0]
    assert "999000000" not in provider.prompts[0]
    assert "RedeemableNoncontrollingInterestEquityCarryingAmount" not in provider.prompts[0]
    assert '"status": "NEEDS_DATA"' in provider.prompts[0]


def test_open_event_appears_in_flags_and_block(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    adapter = FakeAdapter(FILINGS)
    provider = FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []})
    with get_db() as conn:
        _seed_registrant(conn)
        eid = store.upsert_event(
            conn,
            cik="0000444444",
            event_type="merger",
            anchor_accession="0000444444-26-000040",
            company_name="TEST CO",
            detection_date="2026-06-01",
            source_mode="daily",
        )
        store.set_ticker(conn, event_id=eid, ticker="TEST")
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=adapter,
            provider=provider,
            market_cap_mm=100.0,
            raw_companyfacts={"facts": {"us-gaap": {}}},
        )
        assert "EVENT_PENDING:MERGER" in report["flags"]
        block = "\n".join(render_cheapness_block(report))
        assert "merger since 2026-06-01" in block


def test_llm_unavailable_keeps_deterministic_block(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)

    class ExplodingProvider:
        def synthesize_json(self, **kwargs):
            raise RuntimeError("LLM provider disabled")

    adapter = FakeAdapter(FILINGS)
    with get_db() as conn:
        _seed_registrant(conn, cik="0000034067")
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=adapter,
            provider=ExplodingProvider(),
            market_cap_mm=141.03,
            raw_companyfacts=LIVE_MEZZ_FACTS,
        )
        assert report["llm_verdict"] == LLM_UNAVAILABLE
        assert FLAG_MEZZANINE in report["flags"]
        headline = cheapness_headline(report)
        assert "MEZZANINE_OBLIGATION" in headline
        assert "LLM_UNAVAILABLE" in headline


def test_cheapness_failed_attempt_is_charged_once_and_persisted(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)

    class RetryableFailureProvider:
        provider_name = "test"

        def __init__(self):
            self.calls = 0

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            raise RuntimeError("temporarily unavailable")

    provider = RetryableFailureProvider()
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._estimated_llm_call_cost_usd",
        lambda _provider, _kwargs: 0.04,
    )

    with get_db() as conn:
        _seed_registrant(conn)
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=FakeAdapter(FILINGS),
            provider=provider,
            raw_companyfacts={"facts": {"us-gaap": {}}},
            llm_budget_usd=0.05,
        )
        row = conn.execute("SELECT * FROM cheapness_reports WHERE ticker = 'TEST'").fetchone()
        attempt = conn.execute("SELECT * FROM cheapness_paid_attempts").fetchone()

    assert provider.calls == 1
    assert report["llm_verdict"] == LLM_UNAVAILABLE
    assert report["llm_cost_usd"] == 0.04
    assert report["llm_budget_usd"] == 0.05
    assert report["llm_usage"]["call_count"] == 1
    assert report["llm_usage"]["cumulative_cost_usd"] == 0.04
    assert report["llm_usage"]["reserved_cost_usd"] == 0.0
    assert row is not None
    assert row["llm_cost_usd"] == 0.04
    assert row["llm_budget_usd"] == 0.05
    assert json.loads(row["llm_usage_json"])["call_count"] == 1
    assert attempt is not None
    assert attempt["status"] == "PUBLISHED"
    assert attempt["outcome"] == "ERROR"
    assert json.loads(attempt["result_json"])["verdict"] == LLM_UNAVAILABLE
    assert json.loads(attempt["result_json"])["error"] == (
        "test:cheapness_summary_v1 exceeded max retries (0) after 1 attempts: "
        "temporarily unavailable"
    )
    assert "$0.0400 charged / $0.0500 hard cap" in "\n".join(render_cheapness_block(report))


def test_cheapness_hard_budget_blocks_provider_before_physical_attempt(
    monkeypatch,
    tmp_path,
):
    _init(monkeypatch, tmp_path)
    provider = FakeProvider({"verdict": NO_KNOWN_EVENT, "bullets": []})
    monkeypatch.setattr(
        "app.autonomous.sector_runtime._estimated_llm_call_cost_usd",
        lambda _provider, _kwargs: 0.06,
    )

    with get_db() as conn:
        _seed_registrant(conn)
        report = build_cheapness_report(
            "TEST",
            as_of="2026-06-11",
            conn=conn,
            client=FakeClient(),
            adapter=FakeAdapter(FILINGS),
            provider=provider,
            raw_companyfacts={"facts": {"us-gaap": {}}},
            llm_budget_usd=0.05,
        )

    assert provider.calls == 0
    assert report["llm_verdict"] == LLM_UNAVAILABLE
    assert report["llm_cost_usd"] == 0.0
    assert report["llm_budget_usd"] == 0.05
    assert report["llm_usage"]["call_count"] == 0
    assert report["llm_usage"]["events"] == []
    assert "LLM cost budget exceeded" in report["deterministic"]["llm_error"]


def test_extract_risk_sections_from_10k_html(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    html = (
        b"<html><body>"
        b"<p>Item 1. Business</p><p>We make widgets.</p>"
        b"<p>Item 3. Legal Proceedings</p>"
        b"<p>The company is defending a patent suit seeking $40 million.</p>"
        b"<p>Item 7. Management's Discussion and Analysis</p><p>MD&A text.</p>"
        b"<p>COMMITMENTS AND CONTINGENCIES</p>"
        b"<p>An outstanding put option requires the company to purchase the minority stake.</p>"
        b"</body></html>"
    )
    client = FakeClient({"tenk.htm": html})
    filings = [
        {
            "accession": "0000444444-26-000001",
            "form": "10-K",
            "filing_date": "2026-03-01",
            "items": "",
            "primary_document": "tenk.htm",
        }
    ]
    sections = extract_risk_sections(client, "0000444444", filings)
    assert "legal_proceedings" in sections
    assert "patent suit" in sections["legal_proceedings"]["excerpt"]
    assert sections["legal_proceedings"]["accession"] == "0000444444-26-000001"
    assert "commitments_contingencies" in sections
    assert "put option" in sections["commitments_contingencies"]["excerpt"]


def test_headline_no_known_event():
    report = {"status": "OK", "flags": [], "llm_verdict": NO_KNOWN_EVENT, "llm_bullets": []}
    assert cheapness_headline(report) == NO_KNOWN_EVENT
    assert cheapness_headline(None) == "n/a"


def test_fingerprint_changes_on_new_filing():
    base = filings_fingerprint(FILINGS)
    assert base == filings_fingerprint(list(reversed(FILINGS)))  # order-insensitive
    extended = filings_fingerprint(
        FILINGS
        + [
            {
                "accession": "0000444444-26-000099",
                "form": "8-K",
                "filing_date": "2026-06-10",
                "items": "",
                "primary_document": "x.htm",
            }
        ]
    )
    assert extended != base
