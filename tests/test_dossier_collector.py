from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.db import get_db, init_db, utc_now_iso
from app.dossier.collector import (
    NO_ANNUAL_FILING_ERROR,
    _select_annual_filings,
    _select_quarterly_filings,
    collect_10k_docket_stage1_with_debug,
    collect_quarterly_docket,
)
from app.dossier.runner import run_dossier_for_peer_set
from app.ingest.sec_client import FilingStub


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    # Network-enabled paths are exercised with mocked transports; the conftest
    # default is VOE_NET_PROVIDER=disabled, so opt in explicitly.
    monkeypatch.setenv("VOE_NET_PROVIDER", "enabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _stub(*, cik: str, accession: str, form_type: str, filing_date: str, period_end: str) -> FilingStub:
    accession_nodash = accession.replace("-", "")
    return FilingStub(
        cik=cik,
        accession=accession,
        accession_nodash=accession_nodash,
        form_type=form_type,
        filing_date=date.fromisoformat(filing_date),
        period_end=period_end,
        primary_document="doc.htm",
        primary_doc_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/doc.htm",
        filing_index_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/index.json",
    )


def test_select_annual_filings_prefers_latest_filing_date_and_accepts_10ka():
    filings = [
        _stub(cik="1", accession="0001-2015-000001", form_type="10-K", filing_date="2015-02-10", period_end="2014-12-31"),
        _stub(cik="1", accession="0001-2015-000002", form_type="10-K/A", filing_date="2015-03-05", period_end="2014-12-31"),
        _stub(cik="1", accession="0001-2014-000003", form_type="10-K", filing_date="2014-02-12", period_end="2013-12-31"),
    ]
    selected = _select_annual_filings(
        filings,
        years_back=2,
        as_of_date=date(2015, 4, 1),
        include_amendments=True,
    )
    assert [filing.accession for filing in selected] == [
        "0001-2015-000002",
        "0001-2014-000003",
    ]


def test_select_annual_filings_accepts_20f_and_40f():
    filings = [
        _stub(cik="1", accession="0001-2015-000004", form_type="20-F", filing_date="2015-02-25", period_end="2014-12-31"),
        _stub(cik="1", accession="0001-2015-000005", form_type="20-F/A", filing_date="2015-03-20", period_end="2014-12-31"),
        _stub(cik="1", accession="0001-2014-000006", form_type="40-F", filing_date="2014-03-10", period_end="2013-12-31"),
    ]
    selected = _select_annual_filings(
        filings,
        years_back=2,
        as_of_date=date(2015, 4, 1),
        include_amendments=True,
    )
    assert [filing.accession for filing in selected] == [
        "0001-2015-000005",
        "0001-2014-000006",
    ]


def test_select_quarterly_filings_keeps_original_and_amendment_and_respects_as_of():
    """New contract (F-FS-5, 2026-08-05): a reading docket emits BOTH the
    original 10-Q and its 10-Q/A — supersession is for parsed numbers, not
    reading surfaces. Period cap still counts periods, not filings."""
    filings = [
        _stub(cik="1", accession="0001-2015-000010", form_type="10-Q", filing_date="2015-05-08", period_end="2015-03-31"),
        _stub(cik="1", accession="0001-2015-000011", form_type="10-Q/A", filing_date="2015-06-02", period_end="2015-03-31"),
        _stub(cik="1", accession="0001-2014-000012", form_type="10-Q", filing_date="2014-11-06", period_end="2014-09-30"),
        _stub(cik="1", accession="0001-2014-000013", form_type="10-Q", filing_date="2014-08-07", period_end="2014-06-30"),
        # Filed after as_of: must never appear.
        _stub(cik="1", accession="0001-2015-000014", form_type="10-Q", filing_date="2015-08-06", period_end="2015-06-30"),
        # Annual form: not a quarterly candidate.
        _stub(cik="1", accession="0001-2015-000015", form_type="10-K", filing_date="2015-02-20", period_end="2014-12-31"),
    ]
    selected = _select_quarterly_filings(
        filings,
        quarters_back=2,
        as_of_date=date(2015, 7, 1),
        include_amendments=True,
    )
    # Two newest periods survive (2015-03-31, 2014-09-30); both filings of
    # the amended period are emitted, newest first.
    assert [filing.accession for filing in selected] == [
        "0001-2015-000011",
        "0001-2015-000010",
        "0001-2014-000012",
    ]


def test_select_quarterly_filings_same_day_amendment_does_not_hide_original():
    """F-FS-5 regression pin: SHEN's Q2-2026 10-Q and 10-Q/A were filed the
    SAME day (2026-07-29); the accession tie-break kept only the 39KB
    partial amendment and hid the 1.2MB substantive original from the seat."""
    filings = [
        _stub(cik="1", accession="0000354963-26-000209", form_type="10-Q", filing_date="2026-07-29", period_end="2026-06-30"),
        _stub(cik="1", accession="0000354963-26-000214", form_type="10-Q/A", filing_date="2026-07-29", period_end="2026-06-30"),
    ]
    selected = _select_quarterly_filings(
        filings,
        quarters_back=4,
        as_of_date=date(2026, 8, 5),
        include_amendments=True,
    )
    assert [filing.accession for filing in selected] == [
        "0000354963-26-000214",
        "0000354963-26-000209",
    ]


def test_select_quarterly_filings_without_amendments_keeps_original():
    filings = [
        _stub(cik="1", accession="0001-2015-000010", form_type="10-Q", filing_date="2015-05-08", period_end="2015-03-31"),
        _stub(cik="1", accession="0001-2015-000011", form_type="10-Q/A", filing_date="2015-06-02", period_end="2015-03-31"),
    ]
    selected = _select_quarterly_filings(
        filings,
        quarters_back=4,
        as_of_date=date(2015, 7, 1),
        include_amendments=False,
    )
    assert [filing.accession for filing in selected] == ["0001-2015-000010"]


def test_collect_quarterly_docket_lists_quarterly_forms_and_orders_desc(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Corp', ?)
            """,
            (now,),
        )

    filings = [
        _stub(cik="1", accession="0001-2015-000010", form_type="10-Q", filing_date="2015-05-08", period_end="2015-03-31"),
        _stub(cik="1", accession="0001-2014-000012", form_type="10-Q", filing_date="2014-11-06", period_end="2014-09-30"),
        _stub(cik="1", accession="0001-2015-000014", form_type="10-Q", filing_date="2015-08-06", period_end="2015-06-30"),
    ]
    calls: dict[str, object] = {}

    class FakeSecClient:
        def list_filings_window(self, cik: str, *, start_date: date, end_date: date, forms: list[str]):  # noqa: ANN001
            calls["cik"] = cik
            calls["start_date"] = start_date
            calls["end_date"] = end_date
            calls["forms"] = list(forms)
            return list(filings)

    monkeypatch.setattr("app.dossier.collector.SecClient", FakeSecClient)
    monkeypatch.setattr(
        "app.dossier.collector._download_primary_doc_no_db",
        lambda client, filing: f"/tmp/{filing.accession_nodash}.htm",
    )

    docket = collect_quarterly_docket(
        ticker="AAA",
        as_of_date="2015-07-01",
        quarters_back=8,
        parse_downloaded=False,
    )

    assert calls["cik"] == "1"
    assert calls["forms"] == ["10-Q", "10-Q/A"]
    assert calls["end_date"] == date(2015, 7, 1)
    assert calls["start_date"] == date(2012, 7, 1)
    assert [f.accession for f in docket] == [
        "0001-2015-000010",
        "0001-2014-000012",
    ]
    assert all(f.form_type == "10-Q" for f in docket)
    assert all(f.filing_id == -1 for f in docket)
    assert docket[0].local_path == "/tmp/00012015000010.htm"


def test_collect_quarterly_docket_without_cik_returns_empty(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.dossier.collector.resolve_cik_for_ticker", lambda *a, **k: None
    )
    assert (
        collect_quarterly_docket(ticker="ZZZZ", as_of_date="2015-07-01", parse_downloaded=False)
        == []
    )


def test_collect_stage1_uses_window_query_and_builds_debug_missing_years(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Corp', ?)
            """,
            (now,),
        )

    filings = [
        _stub(cik="1", accession="0001-2015-000001", form_type="10-K", filing_date="2015-02-10", period_end="2014-12-31"),
        _stub(cik="1", accession="0001-2015-000002", form_type="10-K/A", filing_date="2015-03-05", period_end="2014-12-31"),
        _stub(cik="1", accession="0001-2014-000003", form_type="10-K", filing_date="2014-02-12", period_end="2013-12-31"),
    ]
    calls: dict[str, object] = {}

    class FakeSecClient:
        def list_filings_window(self, cik: str, *, start_date: date, end_date: date, forms: list[str]):  # noqa: ANN001
            calls["cik"] = cik
            calls["start_date"] = start_date
            calls["end_date"] = end_date
            calls["forms"] = list(forms)
            return list(filings)

    monkeypatch.setattr("app.dossier.collector.SecClient", FakeSecClient)
    monkeypatch.setattr(
        "app.dossier.collector._download_primary_doc_no_db",
        lambda client, filing: f"/tmp/{filing.accession_nodash}.htm",
    )

    stage1, debug = collect_10k_docket_stage1_with_debug(
        ticker="AAA",
        as_of_date="2015-04-01",
        years_back=3,
        include_amendments=True,
    )

    assert calls["cik"] == "1"
    assert calls["forms"] == ["10-K", "10-K/A", "20-F", "20-F/A", "40-F"]
    assert calls["start_date"] == date(2012, 4, 1)
    assert calls["end_date"] == date(2015, 4, 1)
    assert [row.filing.accession for row in stage1] == [
        "0001-2015-000002",
        "0001-2014-000003",
    ]
    assert debug["counts_by_form_type"] == {"10-K": 2, "10-K/A": 1}
    assert [row["accession"] for row in debug["selected_accessions"]] == [
        "0001-2015-000002",
        "0001-2014-000003",
    ]
    missing = {row["fiscal_year"]: row["reason"] for row in debug["missing_years"]}
    assert missing[2012] == "NO_ANNUAL_FILING_FOR_FISCAL_YEAR"
    assert debug["eligible"] is True
    assert debug["preflight"]["annual_filing_count_window"] == 3


def test_preflight_ineligible_skips_download_calls(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Corp', ?)
            """,
            (now,),
        )

    filings = [
        _stub(cik="1", accession="0001-2015-000001", form_type="10-K", filing_date="2015-02-10", period_end="2014-12-31"),
    ]
    calls = {"download": 0}

    class FakeSecClient:
        def list_filings_window(self, cik: str, *, start_date: date, end_date: date, forms: list[str]):  # noqa: ANN001
            return list(filings)

    def _fake_download(client, filing):  # noqa: ANN001
        calls["download"] += 1
        return f"/tmp/{filing.accession_nodash}.htm"

    monkeypatch.setattr("app.dossier.collector.SecClient", FakeSecClient)
    monkeypatch.setattr("app.dossier.collector._download_primary_doc_no_db", _fake_download)

    stage1, debug = collect_10k_docket_stage1_with_debug(
        ticker="AAA",
        as_of_date="2015-04-01",
        years_back=10,
        include_amendments=True,
        min_annual_filings=3,
    )

    assert stage1 == []
    assert calls["download"] == 0
    assert debug["eligible"] is False
    assert debug["skip_reason"] == "INSUFFICIENT_ANNUAL_FILINGS_IN_WINDOW"


def test_dossier_runner_writes_per_ticker_debug_artifact(monkeypatch, tmp_path):
    cfg = _init_temp_db(monkeypatch, tmp_path)

    monkeypatch.setattr(
        "app.dossier.runner._collect_stage1_for_ticker",
        lambda *, ticker, as_of_date, years_back: (
            [],
            {
                "ticker": ticker,
                "query": {"as_of_date": as_of_date, "years_back": years_back},
                "counts_by_form_type": {},
                "selected_accessions": [],
                "missing_years": [{"fiscal_year": 2024, "reason": "NO_ANNUAL_FILING_FOR_FISCAL_YEAR"}],
            },
        ),
    )

    summary = run_dossier_for_peer_set(
        tickers=["AAA"],
        as_of_date="2026-02-13",
        years_back=10,
        run_id="dossier_debug_artifact_test",
        workers=1,
    )
    assert summary["ticker_results"]["AAA"]["status"] == "SKIPPED"
    assert summary["ticker_results"]["AAA"]["error"] == NO_ANNUAL_FILING_ERROR

    debug_path = cfg.dossiers_dir / "dossier_debug_artifact_test" / "dossier_debug_AAA.json"
    assert debug_path.exists()
    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "AAA"
    assert payload["missing_years"][0]["reason"] == "NO_ANNUAL_FILING_FOR_FISCAL_YEAR"
    assert Path(payload["debug_artifact_path"]).resolve() == debug_path.resolve()


def test_collect_stage1_offline_uses_cached_submissions_first(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES('AAA', '1', 'AAA Corp', ?)
            """,
            (now,),
        )

    filings = [
        _stub(cik="1", accession="0001-2015-000001", form_type="10-K", filing_date="2015-02-10", period_end="2014-12-31"),
    ]
    calls = {"window": 0, "cached": 0}

    class FakeSecClient:
        def list_filings_window(self, cik: str, *, start_date: date, end_date: date, forms: list[str]):  # noqa: ANN001
            calls["window"] += 1
            return []

        def list_cached_filings_window(self, cik: str, *, start_date: date, end_date: date, forms: list[str]):  # noqa: ANN001
            calls["cached"] += 1
            return list(filings)

    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    monkeypatch.setattr("app.dossier.collector.SecClient", FakeSecClient)
    calls["download"] = 0
    monkeypatch.setattr(
        "app.dossier.collector._download_primary_doc_no_db",
        lambda client, filing: calls.__setitem__("download", calls["download"] + 1) or None,
    )

    stage1, _debug = collect_10k_docket_stage1_with_debug(
        ticker="AAA",
        as_of_date="2015-04-01",
        years_back=3,
        include_amendments=True,
    )

    assert calls["window"] == 0
    assert calls["cached"] == 1
    assert calls["download"] == 1
    assert len(stage1) == 1


def test_download_primary_doc_no_db_offline_skips_network(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    filing = _stub(
        cik="1",
        accession="0001-2015-000001",
        form_type="10-K",
        filing_date="2015-02-10",
        period_end="2014-12-31",
    )

    class FakeSecClient:
        def download_bytes(self, url, *, use_cache=True):
            raise AssertionError("network should not be attempted in offline mode")

    from app.dossier.collector import _download_primary_doc_no_db

    assert _download_primary_doc_no_db(FakeSecClient(), filing) is None
