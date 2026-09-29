from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.db import get_db, init_db, utc_now_iso
from app.util.http import HttpClient


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _init_temp_env(monkeypatch, tmp_path, *, net_provider: str = "enabled"):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    # Network-enabled paths are exercised with mocked transports; the conftest
    # default is VOE_NET_PROVIDER=disabled, so opt in explicitly.
    monkeypatch.setenv("VOE_NET_PROVIDER", net_provider)
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _seed_submissions_cache(cfg):
    http = HttpClient(cfg)
    url = "https://data.sec.gov/submissions/CIK0001205922.json"
    payload = json.loads((FIXTURE_DIR / "form4_submissions_CIK0001205922.json").read_text(encoding="utf-8"))
    http.cache_path(url).write_text(json.dumps(payload), encoding="utf-8")


def _seed_filing_cik(cfg, ticker: str, cik: str) -> None:
    now = utc_now_iso()
    with get_db(cfg) as conn:
        conn.execute(
            """
            INSERT INTO filings
                (cik, ticker, accession, form_type, filing_date, primary_doc_url, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (cik, ticker, "0001205922-24-000001", "10-K", "2024-03-01", "https://example/10k.htm", "OK", now, now),
        )
        conn.commit()


def test_list_form4_stubs_returns_only_in_window_form4(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_submissions_cache(cfg)

    from app.catalyst.form4 import list_form4_stubs

    stubs = list_form4_stubs(cik="1205922", since=date(2024, 9, 2))

    assert len(stubs) == 3
    assert all(stub.form_type == "4" for stub in stubs)


def test_list_form4_stubs_accession_order(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_submissions_cache(cfg)

    from app.catalyst.form4 import list_form4_stubs

    stubs = list_form4_stubs(cik="1205922", since=date(2024, 9, 2))

    assert [stub.accession for stub in stubs] == [
        "0001062993-24-019266",
        "0001062993-24-019265",
        "0001062993-24-016842",
    ]


def test_list_form4_stubs_excludes_non_form4(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_submissions_cache(cfg)

    from app.catalyst.form4 import list_form4_stubs

    stubs = list_form4_stubs(cik="1205922", since=date(2024, 9, 2))

    non_form4 = [stub for stub in stubs if stub.form_type != "4"]
    assert len(non_form4) == 0


def test_read_form4_stubs_primary_doc_url(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_submissions_cache(cfg)
    _seed_filing_cik(cfg, ticker="X", cik="1205922")

    from app.catalyst.form4 import read_form4_stubs

    with get_db(cfg) as conn:
        events = read_form4_stubs(ticker="X", conn=conn, as_of_date="2024-12-01", lookback_days=90)

    assert len(events) == 3
    assert events[0].primary_doc_url == (
        "https://www.sec.gov/Archives/edgar/data/1205922/000106299324019266/form4.xml"
    )
    assert events[0].transaction_code is None
    assert events[0].is_purchase is None


# --- Form 4 XML transaction-code parse + net-disabled degradation ---


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes()


def test_parse_form4_xml_purchase():
    from app.catalyst.form4 import parse_form4_xml

    parsed = parse_form4_xml(_fixture_bytes("form4_purchase.xml"))

    assert parsed["transaction_code"] == "P"
    assert parsed["is_open_market_purchase"] is True


def test_parse_form4_xml_sale():
    from app.catalyst.form4 import parse_form4_xml

    parsed = parse_form4_xml(_fixture_bytes("form4_sale.xml"))

    assert parsed["transaction_code"] == "S"
    assert parsed["is_open_market_purchase"] is False


def test_parse_form4_xml_grant_excluded():
    from app.catalyst.form4 import parse_form4_xml

    parsed = parse_form4_xml(_fixture_bytes("form4_grant.xml"))

    assert parsed["transaction_code"] == "A"
    assert parsed["is_open_market_purchase"] is False


def test_parse_form4_detects_purchase_when_not_first_transaction():
    # A Form-4 that bundles a grant 'A' before an open-market purchase 'P'
    # must still be flagged as an open-market purchase. The old parser read only
    # the FIRST transaction code ('A') and missed the P.
    from app.catalyst.form4 import parse_form4_xml

    parsed = parse_form4_xml(_fixture_bytes("form4_grant_then_purchase.xml"))

    assert parsed["is_open_market_purchase"] is True


def test_enrich_events_net_enabled_sets_is_purchase(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    assert cfg.net_provider == "enabled"

    from app.catalyst.form4 import Form4Event, enrich_events

    event = Form4Event(
        filing_date=date(2024, 11, 18),
        accession="0001062993-24-019266",
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/1205922/000106299324019266/form4.xml",
    )

    purchase_xml = _fixture_bytes("form4_purchase.xml")
    enriched = enrich_events([event], fetch=lambda url: purchase_xml)

    assert enriched[0].transaction_code == "P"
    assert enriched[0].is_purchase is True


def test_enrich_events_net_disabled_leaves_is_purchase_none(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path, net_provider="disabled")
    assert cfg.net_provider == "disabled"

    from app.catalyst.form4 import Form4Event, enrich_events

    event = Form4Event(
        filing_date=date(2024, 11, 18),
        accession="0001062993-24-019266",
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/1205922/000106299324019266/form4.xml",
    )

    def _boom(url):  # pragma: no cover - must never be called when disabled
        raise AssertionError("fetch must not run when net is disabled")

    enriched = enrich_events([event], fetch=_boom)

    assert enriched[0].is_purchase is None
    assert enriched[0].transaction_code is None


# --- catalyst context builder wiring form4 + buyback ---


def _seed_submissions_cache_for(cfg, cik_padded: str, fixture_name: str) -> None:
    http = HttpClient(cfg)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    payload = json.loads((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))
    http.cache_path(url).write_text(json.dumps(payload), encoding="utf-8")


def _seed_filing_cik_with(cfg, ticker: str, cik: str) -> None:
    now = utc_now_iso()
    with get_db(cfg) as conn:
        conn.execute(
            """
            INSERT INTO filings
                (cik, ticker, accession, form_type, filing_date, primary_doc_url, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (cik, ticker, f"{cik}-24-999001", "10-K", "2024-03-01", "https://example/10k.htm", "OK", now, now),
        )
        conn.commit()


def test_catalyst_context_two_distinct_owners_confirmed(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    assert cfg.net_provider == "enabled"
    _seed_submissions_cache(cfg)
    _seed_filing_cik(cfg, ticker="X", cik="1205922")

    from app.catalyst.context import catalyst_context_for_ticker
    from app.catalyst.overlay import compute_catalyst_signal

    owner_a = _fixture_bytes("form4_purchase_owner_a.xml")
    owner_b = _fixture_bytes("form4_purchase_owner_b.xml")

    # Two distinct reporting owners (A and B) within the window, all P-coded;
    # the third in-window Form-4 repeats owner A so distinct_buyers stays 2.
    by_url = {
        "https://www.sec.gov/Archives/edgar/data/1205922/000106299324019266/form4.xml": owner_a,
        "https://www.sec.gov/Archives/edgar/data/1205922/000106299324019265/form4.xml": owner_b,
        "https://www.sec.gov/Archives/edgar/data/1205922/000106299324016842/form4.xml": owner_a,
    }

    with get_db(cfg) as conn:
        ctx = catalyst_context_for_ticker(
            conn,
            ticker="X",
            as_of_date="2024-12-01",
            lookback_days=90,
            fetch=lambda url: by_url[url],
        )

    assert ctx["insider_distinct_buyers"] == 2
    assert ctx["insider_open_market_purchase_count"] == 3
    assert ctx["codes_unverified"] is False
    assert compute_catalyst_signal(ctx).label == "CONFIRMED"


def test_catalyst_context_same_owner_repeats_weak(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    assert cfg.net_provider == "enabled"
    _seed_submissions_cache(cfg)
    _seed_filing_cik(cfg, ticker="X", cik="1205922")

    from app.catalyst.context import catalyst_context_for_ticker
    from app.catalyst.overlay import compute_catalyst_signal

    owner_a = _fixture_bytes("form4_purchase_owner_a.xml")

    # Three Form-4s all from the SAME reporting owner, all P-coded.
    with get_db(cfg) as conn:
        ctx = catalyst_context_for_ticker(
            conn,
            ticker="X",
            as_of_date="2024-12-01",
            lookback_days=90,
            fetch=lambda url: owner_a,
        )

    assert ctx["insider_distinct_buyers"] == 1
    assert ctx["insider_open_market_purchase_count"] == 3
    assert ctx["codes_unverified"] is False
    assert compute_catalyst_signal(ctx).label == "WEAK"


def test_catalyst_context_owner_less_purchases_do_not_form_cluster(monkeypatch, tmp_path):
    # Purchases whose reporting-owner identity fails to parse (malformed /
    # partial XML, e.g. no <reportingOwner> block) must NOT each count as a
    # distinct buyer — that would manufacture a false CONFIRMED >=2-buyer cluster
    # out of filings we cannot attribute. They count toward open_market_count only.
    cfg = _init_temp_env(monkeypatch, tmp_path)
    assert cfg.net_provider == "enabled"
    _seed_submissions_cache(cfg)
    _seed_filing_cik(cfg, ticker="X", cik="1205922")

    from app.catalyst.context import catalyst_context_for_ticker
    from app.catalyst.overlay import compute_catalyst_signal

    # All in-window Form-4s are P-coded but owner-less (form4_purchase.xml has no
    # <reportingOwner>), so _reporting_owner_identity returns None for each.
    owner_less = _fixture_bytes("form4_purchase.xml")

    with get_db(cfg) as conn:
        ctx = catalyst_context_for_ticker(
            conn,
            ticker="X",
            as_of_date="2024-12-01",
            lookback_days=90,
            fetch=lambda url: owner_less,
        )

    assert ctx["insider_distinct_buyers"] == 0
    assert ctx["insider_open_market_purchase_count"] == 3
    assert compute_catalyst_signal(ctx).label != "CONFIRMED"


def test_catalyst_context_net_disabled_codes_unverified_weak(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path, net_provider="disabled")
    assert cfg.net_provider == "disabled"
    _seed_submissions_cache_for(cfg, "0002205922", "form4_submissions_CIK0002205922.json")
    _seed_filing_cik_with(cfg, ticker="Y", cik="2205922")

    from app.catalyst.context import catalyst_context_for_ticker
    from app.catalyst.overlay import compute_catalyst_signal

    def _boom(url):  # pragma: no cover - must never be called when disabled
        raise AssertionError("fetch must not run when net is disabled")

    with get_db(cfg) as conn:
        ctx = catalyst_context_for_ticker(
            conn,
            ticker="Y",
            as_of_date="2024-12-01",
            lookback_days=90,
            fetch=_boom,
        )

    # Two in-window Form-4 filings, codes unknown (net disabled): distinct_buyers
    # counts filers but no open-market purchase can be confirmed, so WEAK only.
    assert ctx["insider_filing_count"] == 2
    assert ctx["insider_open_market_purchase_count"] == 0
    assert ctx["codes_unverified"] is True
    assert compute_catalyst_signal(ctx).label == "WEAK"


def test_catalyst_context_persists_to_catalyst_events(monkeypatch, tmp_path):
    # The context builder memoizes the parsed Form-4 cluster + the
    # buyback signal to the catalyst_events cache so trigger runs do not re-fetch
    # / re-parse form4.xml and the parsed events stay auditable.
    cfg = _init_temp_env(monkeypatch, tmp_path)
    assert cfg.net_provider == "enabled"
    _seed_submissions_cache(cfg)
    _seed_filing_cik(cfg, ticker="X", cik="1205922")

    from app.catalyst.context import catalyst_context_for_ticker
    from app.db import get_catalyst_event

    owner_a = _fixture_bytes("form4_purchase_owner_a.xml")
    owner_b = _fixture_bytes("form4_purchase_owner_b.xml")
    by_url = {
        "https://www.sec.gov/Archives/edgar/data/1205922/000106299324019266/form4.xml": owner_a,
        "https://www.sec.gov/Archives/edgar/data/1205922/000106299324019265/form4.xml": owner_b,
        "https://www.sec.gov/Archives/edgar/data/1205922/000106299324016842/form4.xml": owner_a,
    }

    with get_db(cfg) as conn:
        catalyst_context_for_ticker(
            conn,
            ticker="X",
            as_of_date="2024-12-01",
            lookback_days=90,
            fetch=lambda url: by_url[url],
        )
        conn.commit()
        cached = get_catalyst_event(
            conn,
            ticker="X",
            as_of_date="2024-12-01",
            catalyst_type="INSIDER_BUY_CLUSTER",
        )

    assert cached is not None
    assert cached["signal_label"] == "CONFIRMED"
    assert cached["score"] == 4.0
    assert cached["detail"]["insider_distinct_buyers"] == 2
    assert cached["detail"]["insider_filing_count"] == 3


def test_catalyst_context_persist_false_skips_cache(monkeypatch, tmp_path):
    # persist=False (e.g. a pure point-in-time backtest read) must not write rows.
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_submissions_cache(cfg)
    _seed_filing_cik(cfg, ticker="X", cik="1205922")

    from app.catalyst.context import catalyst_context_for_ticker
    from app.db import get_catalyst_event

    owner_a = _fixture_bytes("form4_purchase_owner_a.xml")

    with get_db(cfg) as conn:
        catalyst_context_for_ticker(
            conn,
            ticker="X",
            as_of_date="2024-12-01",
            lookback_days=90,
            fetch=lambda url: owner_a,
            persist=False,
        )
        conn.commit()
        cached = get_catalyst_event(
            conn,
            ticker="X",
            as_of_date="2024-12-01",
            catalyst_type="INSIDER_BUY_CLUSTER",
        )

    assert cached is None
