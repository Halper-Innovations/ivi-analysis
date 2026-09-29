"""Named acceptance tests for the events feed + cheapness pass.

Three live failures define this spec, and each test below carries the REAL
EDGAR data (fetched 2026-06-11) as literals:

- NCSM surfaced DEPLOY_READY ten days into a definitive merger — eight Form
  425s hit EDGAR 2026-06-01/02 under CIK 0001692427 plus a merger 8-K with
  item 1.01 (accession 0001193125-26-252096).
- BOOM surfaced with a redeemable-NCI put obligation larger than its market
  cap — RedeemableNoncontrollingInterestEquityCarryingAmount $187,080,000 as
  of 2026-03-31 (10-Q accession 0000034067-26-000036) vs $141.03M cap.
- KEQU is the clean control: no mezzanine concepts in companyfacts, no
  protected-form filings — the pass must say NO_KNOWN_EVENT, not cry wolf.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from app.config import get_config
from app.db import get_db, init_db
from app.events.cheapness import (
    FLAG_MEZZANINE,
    NO_KNOWN_EVENT,
    build_cheapness_report,
    cheapness_headline,
    mezzanine_check,
    render_cheapness_block,
)
from app.events.detectors import detect_queue_protection
from app.events.flags import sync_event_pending_flags
from app.events.index_feed import IndexRow
from app.watchlist.contract import WatchlistEntry
from app.watchlist.digest import render_digest
from app.watchlist.store import add_or_update, add_price_snapshot, get_latest, watchlist_queue
from tests.financial_integrity_helpers import canonicalize_financial_packet


def _canonical_packet_for(ticker: str, as_of_date: str) -> dict:
    from app.autonomous.financial_integrity import stable_quote_hash
    from tests.test_financial_integrity import _valid_packet

    packet = copy.deepcopy(_valid_packet())

    def _replace_fixture_identity(value):
        if isinstance(value, dict):
            return {key: _replace_fixture_identity(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_replace_fixture_identity(item) for item in value]
        if isinstance(value, str):
            return (
                value.replace("2026-07-21", as_of_date)
                .replace("MEGA", ticker)
                .replace("/mega", f"/{ticker.lower()}")
            )
        return value

    packet = _replace_fixture_identity(packet)
    quote = {
        "ticker": packet["ticker"],
        "price": packet["current_price"],
        "as_of_date": packet["current_price_as_of_date"],
        "currency": packet["current_price_currency"],
        "source": packet["current_price_source"],
        "source_url": packet["current_price_source_url"],
        "price_basis": packet["price_basis"],
        "raw_price": packet["raw_price"],
        "split_adjustment_factor": packet["split_adjustment_factor"],
        "split_effective_date": packet["split_effective_date"],
        "split_lineage_proof": packet["split_lineage_proof"],
    }
    snapshot_id = stable_quote_hash(quote)
    packet["quote_snapshot_id"] = snapshot_id
    packet["cap_stage_quote_snapshot_id"] = snapshot_id
    for trace in packet["metric_traces"].values():
        trace["quote_snapshot_id"] = snapshot_id
    market_cap_mm = {
        "BOOM": 141.03,
        "KEQU": 102.50,
    }.get(ticker, 2_000.0)
    shares_outstanding_mm = market_cap_mm / 100.0
    packet["market_cap_mm"] = market_cap_mm
    packet["shares_outstanding_mm"] = shares_outstanding_mm
    packet["raw_shares_outstanding_mm"] = shares_outstanding_mm
    canonicalize_financial_packet(
        packet,
        as_of_date=as_of_date,
        shares_mm=shares_outstanding_mm,
    )
    snapshot_id = packet["quote_snapshot_id"]
    market_cap_trace = packet["metric_traces"]["market_cap_mm"]
    market_cap_trace["quote_snapshot_id"] = snapshot_id
    market_cap_trace["output"] = market_cap_mm
    market_cap_trace["recomputed_output"] = market_cap_mm
    market_cap_trace["inputs"]["shares_outstanding_mm"] = shares_outstanding_mm
    market_cap_trace["input_provenance"]["shares_outstanding_mm"].update(
        {
            "value": shares_outstanding_mm,
            "source": "SEC_COMPANYFACTS",
            "period_end": as_of_date,
            "filed_date": as_of_date,
            "source_reference": packet["shares_source_url"],
            "raw_source_value": shares_outstanding_mm * 1_000_000.0,
            "raw_source_unit": "shares",
            "normalized_value": shares_outstanding_mm,
            "normalized_unit": "shares_millions",
            "split_adjustment_factor": 1.0,
        }
    )
    packet["valuation"] = {}
    packet["metric_traces"] = {"market_cap_mm": market_cap_trace}
    return packet


@pytest.fixture(autouse=True)
def _canonical_financial_context(monkeypatch):
    monkeypatch.setattr(
        "app.events.cheapness.build_canonical_v1_financial_context",
        lambda *, tickers, as_of_date, **_kwargs: SimpleNamespace(
            packets={ticker: _canonical_packet_for(ticker, as_of_date) for ticker in tickers}
        ),
    )


def _init(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    init_db()


def _add_watch_row(ticker, *, buy_target, price, status="DEPLOY_READY"):
    add_or_update(
        WatchlistEntry(
            ticker=ticker,
            status=status,
            conviction_grade="WATCHLIST_ONLY",
            confidence="MODERATE",
            conviction_source="sector_final_decision",
            valuation_anchor_method="DCF",
            valuation_anchor_value=buy_target / 0.55,
            buy_price_target=buy_target,
            current_price_at_addition=price,
            thesis_text=f"{ticker} thesis",
            source_run_id=f"run_{ticker}",
            source_sector="energy_services",
            added_at="2026-05-20T00:00:00Z",
        )
    )
    entry = get_latest(ticker)
    add_price_snapshot(entry.id, price=price, checked_at="2026-06-10T00:00:00Z", source="test")


# ---------------------------------------------------------------------------
# NCSM — EVENT_PENDING:MERGER from the June 1 425s; DEPLOY_READY blocked
# ---------------------------------------------------------------------------

NCSM_CIK = "0001692427"

# Real daily-index rows: Form 425s filed under NCSM's CIK on 2026-06-01,
# and the merger 8-K (items 1.01,5.07,8.01,9.01) on 2026-06-02.
NCSM_425_ROWS_JUNE1 = [
    IndexRow(
        cik=NCSM_CIK,
        company_name="NCS Multistage Holdings, Inc.",
        form_type="425",
        date_filed="2026-06-01",
        file_name=f"edgar/data/1692427/{accession}.txt",
    )
    for accession in (
        "0001603923-26-000054",
        "0001603923-26-000055",
        "0001603923-26-000059",
        "0001193125-26-249808",
        "0001193125-26-251591",
        "0001193125-26-251608",
    )
]
NCSM_8K_ROW_JUNE2 = IndexRow(
    cik=NCSM_CIK,
    company_name="NCS Multistage Holdings, Inc.",
    form_type="8-K",
    date_filed="2026-06-02",
    file_name="edgar/data/1692427/0001193125-26-252096.txt",
)


class NCSMSubmissions:
    def items_for(self, cik, accession):
        if (cik, accession) == (NCSM_CIK, "0001193125-26-252096"):
            return "1.01,5.07,8.01,9.01"
        return ""

    def prior_item103(self, cik, *, before):
        return None


def test_acceptance_ncsm_merger_flagged_and_blocked(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_watch_row("NCSM", buy_target=57.18, price=24.95)

    watchlist_ciks = {NCSM_CIK: "NCSM"}
    with get_db() as conn:
        counters = detect_queue_protection(
            conn,
            NCSM_425_ROWS_JUNE1,
            scan_date="2026-06-01",
            source_mode="backfill",
            watchlist_ciks=watchlist_ciks,
            submissions=NCSMSubmissions(),
        )
        assert counters.events_created == 1  # six 425s -> ONE merger event
        detect_queue_protection(
            conn,
            [NCSM_8K_ROW_JUNE2],
            scan_date="2026-06-02",
            source_mode="backfill",
            watchlist_ciks=watchlist_ciks,
            submissions=NCSMSubmissions(),
        )
        sync_event_pending_flags(conn)

        merger = conn.execute("SELECT * FROM corporate_events WHERE event_type='merger'").fetchone()
        assert merger["status"] == "DETECTED"
        assert merger["detection_date"] == "2026-06-01"
        assert merger["ticker"] == "NCSM"
        assert merger["anchor_accession"] == "0001603923-26-000054"
        n_filings = conn.execute(
            "SELECT COUNT(*) c FROM corporate_event_filings WHERE event_id=?",
            (merger["id"],),
        ).fetchone()["c"]
        assert n_filings == 6

        # Watchlist row relabeled, with an audit-history row.
        row = conn.execute("SELECT * FROM watchlist WHERE ticker='NCSM'").fetchone()
        assert row["status"] == "DEPLOY_READY"  # stored status untouched
        assert "EVENT_PENDING:MERGER" in row["event_pending"]
        history = conn.execute(
            "SELECT * FROM watchlist_history WHERE field_name='event_pending'"
        ).fetchone()
        assert "EVENT_PENDING:MERGER" in history["new_value"]

    # Blocked from DEPLOY_READY presentation on every surface.
    queue_row = {r["ticker"]: r for r in watchlist_queue(limit=25)}["NCSM"]
    assert queue_row["presented_status"] == "EVENT_PENDING"
    digest = render_digest(days_back=30)
    at_target = digest.split("## At Buy Target (Review)")[1].split("##")[0]
    assert "NCSM" not in at_target
    blocked = digest.split("## Blocked Pending Event Review")[1].split("##")[0]
    assert "NCSM" in blocked and "EVENT_PENDING:MERGER" in blocked

    # The flag also carries the 8-K item 1.01 event.
    with get_db() as conn:
        types = sorted(
            r["event_type"]
            for r in conn.execute(
                "SELECT event_type FROM corporate_events WHERE cik=?", (NCSM_CIK,)
            ).fetchall()
        )
    assert types == ["material_agreement", "merger"]


def test_acceptance_ncsm_idempotent_re_poll(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_watch_row("NCSM", buy_target=57.18, price=24.95)
    watchlist_ciks = {NCSM_CIK: "NCSM"}
    with get_db() as conn:
        detect_queue_protection(
            conn,
            NCSM_425_ROWS_JUNE1,
            scan_date="2026-06-01",
            source_mode="backfill",
            watchlist_ciks=watchlist_ciks,
            submissions=NCSMSubmissions(),
        )
        counters = detect_queue_protection(
            conn,
            NCSM_425_ROWS_JUNE1,
            scan_date="2026-06-01",
            source_mode="backfill",
            watchlist_ciks=watchlist_ciks,
            submissions=NCSMSubmissions(),
        )
        assert counters.events_created == 0
        assert counters.events_updated == 0
        n = conn.execute("SELECT COUNT(*) c FROM corporate_events").fetchone()["c"]
        assert n == 1


# ---------------------------------------------------------------------------
# BOOM — MEZZANINE_OBLIGATION from the real redeemable-NCI carrying amount
# ---------------------------------------------------------------------------

BOOM_CIK = "0000034067"
BOOM_MARKET_CAP_MM = 141.03  # live watchlist row, cap_source=asof_companyfacts

# Real us-gaap concepts from companyfacts CIK0000034067 (fetched 2026-06-11):
# the Arcadia put sits in temporary equity at $187.08M as of 2026-03-31.
BOOM_RAW_COMPANYFACTS = {
    "cik": 34067,
    "facts": {
        "us-gaap": {
            "RedeemableNoncontrollingInterestEquityCarryingAmount": {
                "units": {
                    "USD": [
                        {
                            "end": "2026-03-31",
                            "val": 187_080_000,
                            "accn": "0000034067-26-000036",
                            "form": "10-Q",
                            "filed": "2026-05-01",
                        },
                    ]
                }
            },
            "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests": {
                "units": {
                    "USD": [
                        {
                            "end": "2026-03-31",
                            "val": 187_080_000,
                            "accn": "0000034067-26-000036",
                            "form": "10-Q",
                            "filed": "2026-05-01",
                        },
                    ]
                }
            },
        }
    },
}

BOOM_FILINGS = [
    {
        "accession": "0000034067-26-000036",
        "form": "10-Q",
        "filing_date": "2026-05-01",
        "items": "",
        "primary_document": "boom-20260331.htm",
    },
    {
        "accession": "0000034067-26-000010",
        "form": "10-K",
        "filing_date": "2026-02-24",
        "items": "",
        "primary_document": "boom-20251231.htm",
    },
]


class BOOMAdapter:
    def filings_window(self, cik, *, start, end):
        return [f for f in BOOM_FILINGS if start <= f["filing_date"] <= end]


class BOOMProvider:
    """Canned analyst summary citing the put by its real accession."""

    def synthesize_json(self, *, prompt, schema, schema_name=None):
        from app.llm.providers.disabled_provider import LLMResult

        payload = {
            "verdict": "KNOWN_EVENTS",
            "bullets": [
                {
                    "text": (
                        "Arcadia redeemable noncontrolling-interest put obligation "
                        "carried at $187.1M in temporary equity — larger than the "
                        "entire market cap."
                    ),
                    "accessions": ["0000034067-26-000036"],
                }
            ],
        }
        return LLMResult(
            json_text=json.dumps(payload),
            model="test-model",
            usage_input_tokens=0,
            usage_output_tokens=0,
            raw={},
        )


class _NoDocsClient:
    def download_bytes(self, url, *, use_cache=True):
        raise RuntimeError("no document fetches in this test")


def test_acceptance_boom_mezzanine_obligation_fires():
    result = mezzanine_check(BOOM_RAW_COMPANYFACTS, BOOM_MARKET_CAP_MM, as_of="2026-06-11")
    assert result["flag"] == FLAG_MEZZANINE
    assert result["mezzanine_total_mm"] == 187.08
    assert result["ratio_to_cap"] == round(187.08 / 141.03, 4)  # 1.3265 — >100% of cap
    # Umbrella concept used once; no double count with the component concept.
    assert result["components"] == [
        {
            "concept": (
                "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests"
            ),
            "reported_value": 187_080_000.0,
            "reported_unit": "USD",
            "value_mm": 187.08,
            "unit": "USD_millions",
            "as_of": "2026-03-31",
            "period_end": "2026-03-31",
            "filed_date": "2026-05-01",
            "accession": "0000034067-26-000036",
            "source_url": ("https://data.sec.gov/api/xbrl/companyfacts/CIK0000034067.json"),
            "source_reference": (
                "https://data.sec.gov/api/xbrl/companyfacts/CIK0000034067.json"
                "#us-gaap:"
                "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests:"
                "accession=0000034067-26-000036"
            ),
        }
    ]


def test_acceptance_boom_explanation_block_cites_the_put(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, exchange_scope, operating_status,
                first_seen_at, last_seen_at, name)
            VALUES(?, 'BOOM', 'listed', 'operating', 'x', 'x', 'DMC Global Inc.')
            """,
            (BOOM_CIK,),
        )
        report = build_cheapness_report(
            "BOOM",
            as_of="2026-06-11",
            conn=conn,
            client=_NoDocsClient(),
            adapter=BOOMAdapter(),
            provider=BOOMProvider(),
            market_cap_mm=BOOM_MARKET_CAP_MM,
            raw_companyfacts=BOOM_RAW_COMPANYFACTS,
        )
    assert report["status"] == "OK"
    assert FLAG_MEZZANINE in report["flags"]
    assert report["llm_verdict"] == "KNOWN_EVENTS"
    block = "\n".join(render_cheapness_block(report))
    assert "Arcadia redeemable noncontrolling-interest put" in block
    assert "0000034067-26-000036" in block  # the filing behind the claim
    assert "$187.1M" in block or "187.1" in block
    headline = cheapness_headline(report)
    assert "MEZZANINE_OBLIGATION" in headline


# ---------------------------------------------------------------------------
# KEQU — clean record renders NO_KNOWN_EVENT (the pass does not cry wolf)
# ---------------------------------------------------------------------------

KEQU_CIK = "0000055529"

# Real trailing filings (subset; fetched 2026-06-11). Routine record:
# 10-K, 10-Qs, results 8-Ks. No merger paper, no SC 13D, no S-1/424B.
KEQU_FILINGS = [
    {
        "accession": "0000055529-26-000014",
        "form": "8-K",
        "filing_date": "2026-06-10",
        "items": "8.01,9.01",
        "primary_document": "kequ-20260610.htm",
    },
    {
        "accession": "0000055529-26-000009",
        "form": "10-Q",
        "filing_date": "2026-03-13",
        "items": "",
        "primary_document": "kequ-20260131.htm",
    },
    {
        "accession": "0000055529-26-000006",
        "form": "8-K",
        "filing_date": "2026-03-11",
        "items": "2.02,9.01",
        "primary_document": "kequ-20260311.htm",
    },
    {
        "accession": "0000055529-25-000054",
        "form": "10-Q",
        "filing_date": "2025-12-12",
        "items": "",
        "primary_document": "kequ-20251031.htm",
    },
    {
        "accession": "0000055529-25-000026",
        "form": "10-K",
        "filing_date": "2025-07-02",
        "items": "",
        "primary_document": "kequ-20250430.htm",
    },
]

# Real companyfacts check (2026-06-11): KEQU carries NO redeemable/temporary
# equity concepts at all.
KEQU_RAW_COMPANYFACTS = {"facts": {"us-gaap": {}}}


class KEQUAdapter:
    def filings_window(self, cik, *, start, end):
        return [f for f in KEQU_FILINGS if start <= f["filing_date"] <= end]


class KEQUProvider:
    def synthesize_json(self, *, prompt, schema, schema_name=None):
        from app.llm.providers.disabled_provider import LLMResult

        return LLMResult(
            json_text=json.dumps({"verdict": NO_KNOWN_EVENT, "bullets": []}),
            model="test-model",
            usage_input_tokens=0,
            usage_output_tokens=0,
            raw={},
        )


def test_acceptance_kequ_renders_no_known_event(monkeypatch, tmp_path):
    _init(monkeypatch, tmp_path)
    _add_watch_row("KEQU", buy_target=26.51, price=30.00, status="ACTIVE")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sec_registrants(
                cik, primary_ticker, exchange_scope, operating_status,
                first_seen_at, last_seen_at, name)
            VALUES(?, 'KEQU', 'listed', 'operating', 'x', 'x', 'KEWAUNEE SCIENTIFIC CORP /DE/')
            """,
            (KEQU_CIK,),
        )
        # The queue-protection detector sees KEQU's real June rows and stays quiet.
        kequ_rows = [
            IndexRow(
                cik=KEQU_CIK,
                company_name="KEWAUNEE SCIENTIFIC CORP /DE/",
                form_type=f["form"],
                date_filed=f["filing_date"],
                file_name=f"edgar/data/55529/{f['accession']}.txt",
            )
            for f in KEQU_FILINGS
            if f["filing_date"] >= "2026-06-01"
        ]

        class KEQUSubmissions:
            def items_for(self, cik, accession):
                return "8.01,9.01" if accession == "0000055529-26-000014" else ""

            def prior_item103(self, cik, *, before):
                return None

        counters = detect_queue_protection(
            conn,
            kequ_rows,
            scan_date="2026-06-10",
            source_mode="backfill",
            watchlist_ciks={KEQU_CIK: "KEQU"},
            submissions=KEQUSubmissions(),
        )
        assert counters.events_created == 0  # 8.01 results 8-K is not a protected item
        sync_event_pending_flags(conn)
        row = conn.execute("SELECT event_pending FROM watchlist WHERE ticker='KEQU'").fetchone()
        assert row["event_pending"] is None

        report = build_cheapness_report(
            "KEQU",
            as_of="2026-06-11",
            conn=conn,
            client=_NoDocsClient(),
            adapter=KEQUAdapter(),
            provider=KEQUProvider(),
            market_cap_mm=102.50,
            raw_companyfacts=KEQU_RAW_COMPANYFACTS,
        )
    assert report["flags"] == []
    assert report["llm_verdict"] == NO_KNOWN_EVENT
    assert report["deterministic"]["mezzanine"]["status"] == "NO_MEZZANINE_CONCEPTS"
    assert cheapness_headline(report) == NO_KNOWN_EVENT
    block = "\n".join(render_cheapness_block(report))
    assert "Filing record clean over the trailing 12 months." in block

    # Queue renders the clean verdict, and KEQU is not blocked anywhere.
    queue_row = {r["ticker"]: r for r in watchlist_queue(limit=25)}["KEQU"]
    assert queue_row["cheapness"] == NO_KNOWN_EVENT
    assert queue_row["presented_status"] == "ACTIVE"
