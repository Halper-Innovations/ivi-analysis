"""A fresh install has no registrant table; the SIC is fetched once from the SEC.

The REIT rule (SIC 6798) and the bank/insurer routing both read the SIC from the
shared lookup. With an empty ``sec_registrants`` they used to see no SIC at all.
``ensure_registrant_sic`` fetches the submissions JSON once, stores the SIC, and the
shared lookup finds it afterwards. Offline or on a failed fetch nothing changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from app.config import get_config
from app.db import get_db, init_db
from app.util import http as http_module
from app.util.issuer_classification import (
    ISSUER_CLASS_FINANCIAL,
    ensure_registrant_sic,
    lookup_is_reit,
    lookup_registrant_sic,
    resolve_issuer_classification,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sec_submissions_reit.json"
URL = "https://data.sec.gov/submissions/CIK0001234567.json"


class _Response:
    def __init__(self, payload: dict | None, status: int = 200):
        self.status_code = status
        self.content = json.dumps(payload or {}).encode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


class _Session:
    def __init__(self, calls: list[str], payload: dict | None, status: int = 200):
        self.calls = calls
        self.headers: dict[str, str] = {}
        self._payload = payload
        self._status = status

    def get(self, url, params=None, timeout=None, **kwargs):
        # The client follows redirects by hand (allow_redirects=False, stream=True).
        self.calls.append(url)
        return _Response(self._payload, self._status)


def _fresh_install(monkeypatch, *, net: str = "enabled"):
    monkeypatch.setenv("VOE_NET_PROVIDER", net)
    monkeypatch.setenv("VOE_SEC_USER_AGENT", "IVI tests qa@ivi.test")
    monkeypatch.setenv("VOE_ISSUER_CLASSIFICATION_BY_SIC", "true")
    monkeypatch.setenv("VOE_SEC_MAX_RETRIES", "0")
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _install_session(monkeypatch, payload, status: int = 200) -> list[str]:
    calls: list[str] = []
    original = http_module.HttpClient.__init__

    def init(self, cfg=None):
        original(self, cfg)
        headers = self.session.headers
        self.session = _Session(calls, payload, status)
        self.session.headers = headers

    monkeypatch.setattr(http_module.HttpClient, "__init__", init)
    return calls


def test_missing_sic_is_fetched_once_and_stored(isolated_data_root, monkeypatch):
    cfg = _fresh_install(monkeypatch)
    calls = _install_session(monkeypatch, json.loads(FIXTURE.read_text()))

    assert lookup_registrant_sic(cik="1234567", cfg=cfg) == (None, "NO_REGISTRANT_ROW")

    assert ensure_registrant_sic(cik="1234567", cfg=cfg) == ("6798", "OK")
    assert calls == [URL]
    with get_db(cfg) as conn:
        row = conn.execute("SELECT cik, sic, sic_description FROM sec_sic_cache").fetchone()
    assert (row["cik"], row["sic"], row["sic_description"]) == (
        "0001234567",
        "6798",
        "Real Estate Investment Trusts",
    )

    # The shared lookup now answers, and a second ensure makes no request.
    assert lookup_registrant_sic(cik="1234567", cfg=cfg) == ("6798", "OK")
    assert lookup_is_reit(cik="1234567", cfg=cfg) == (True, "OK")
    assert ensure_registrant_sic(cik="1234567", cfg=cfg) == ("6798", "OK")
    assert calls == [URL]


def test_fetched_bank_sic_routes_the_issuer_as_financial(isolated_data_root, monkeypatch):
    """Migrated 2026-09-29: the shared resolver now fetches a missing SIC itself, so the
    first call already answers from the SIC (it used to need ``ensure_registrant_sic``
    first and fell back to the tag-name rule before it)."""
    cfg = _fresh_install(monkeypatch)
    payload = {**json.loads(FIXTURE.read_text()), "sic": "6022"}
    calls = _install_session(monkeypatch, payload)

    # Nothing in the text says "bank", so only the SIC can route it.
    assert resolve_issuer_classification(cik="1234567", texts=["widgets"], cfg=cfg) == (
        ISSUER_CLASS_FINANCIAL,
        "sic",
    )
    assert calls == [URL]


def test_offline_makes_no_request_and_leaves_the_gap_explicit(isolated_data_root, monkeypatch):
    cfg = _fresh_install(monkeypatch, net="disabled")
    calls = _install_session(monkeypatch, json.loads(FIXTURE.read_text()))

    assert ensure_registrant_sic(cik="1234567", cfg=cfg) == (None, "NO_REGISTRANT_ROW")
    assert calls == []
    assert lookup_is_reit(cik="1234567", cfg=cfg) == (False, "NO_REGISTRANT_ROW")


@pytest.mark.parametrize(
    "payload,status",
    [
        (None, 404),  # unknown CIK
        ({"cik": "1234567", "name": "No SIC Co"}, 200),  # payload without a sic field
        ({"cik": "1234567", "sic": ""}, 200),  # empty sic
    ],
)
def test_failed_or_empty_fetch_stores_nothing(isolated_data_root, monkeypatch, payload, status):
    cfg = _fresh_install(monkeypatch)
    _install_session(monkeypatch, payload, status)

    assert ensure_registrant_sic(cik="1234567", cfg=cfg) == (None, "NO_REGISTRANT_ROW")
    with get_db(cfg) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sec_sic_cache").fetchone()[0] == 0


def test_a_registrant_row_with_a_sic_never_triggers_a_fetch(isolated_data_root, monkeypatch):
    cfg = _fresh_install(monkeypatch)
    calls = _install_session(monkeypatch, json.loads(FIXTURE.read_text()))
    with get_db(cfg) as conn:
        conn.execute(
            "INSERT INTO sec_registrants (cik, primary_ticker, exchange_scope, sic, "
            "operating_status, first_seen_at, last_seen_at) "
            "VALUES ('0001234567', 'EXRT', 'NYSE', 6022, 'OPERATING', '2026-01-01', '2026-01-01')"
        )
    assert ensure_registrant_sic(cik="1234567", cfg=cfg) == ("6022", "OK")
    assert calls == []


def test_every_classification_path_fetches_a_missing_sic_once(isolated_data_root, monkeypatch):
    """Fresh install: only ``ivi value`` called ``ensure_registrant_sic``, so every other
    path read no SIC and fell back to the tag-name rule, which calls Coca-Cola, Procter &
    Gamble, Costco and Realty Income financial (their net debt BANK_SPECIFIC_HANDLING).
    The shared lookup now fetches the SIC once itself -- by CIK, or by the ``companies``
    table's CIK for a ticker-only caller -- and the fetched code decides."""
    from app.valuation.net_debt import _infer_companyfacts_issuer_classification

    cfg = _fresh_install(monkeypatch)
    payload = {**json.loads(FIXTURE.read_text()), "sic": "2080"}  # beverages
    calls = _install_session(monkeypatch, payload)
    with get_db(cfg) as conn:
        conn.execute("INSERT INTO companies (ticker, cik, created_at) VALUES ('BEVCO', '0001234567', '2026-01-01')")
        conn.commit()
    bank_like = {
        "cik": 1234567,
        "entityName": "Beverage Co",
        "facts": {"us-gaap": {"Deposits": {}, "LoansAndLeasesReceivableNetReportedAmount": {}}},
    }
    # The tag-name rule alone would call it financial.
    assert resolve_issuer_classification(
        texts=["deposits", "loans"], line_items=["deposits", "loans"], cfg=cfg
    ) == (ISSUER_CLASS_FINANCIAL, "substring:NO_IDENTIFIER")

    assert _infer_companyfacts_issuer_classification(bank_like, cfg=cfg) == ("operating", "sic")
    assert resolve_issuer_classification(
        ticker="bevco", line_items=["deposits", "loans"], cfg=cfg
    ) == ("operating", "sic")
    assert lookup_is_reit(cik="1234567", cfg=cfg) == (False, "OK")
    assert calls == [URL]


def test_the_shared_lookup_offline_or_after_a_failed_fetch_is_unchanged(
    isolated_data_root, monkeypatch
):
    cfg = _fresh_install(monkeypatch, net="disabled")
    calls = _install_session(monkeypatch, json.loads(FIXTURE.read_text()))
    assert resolve_issuer_classification(cik="1234567", texts=["widgets"], cfg=cfg) == (
        "operating",
        "substring:NO_REGISTRANT_ROW",
    )
    assert calls == []

    cfg = _fresh_install(monkeypatch)
    calls = _install_session(monkeypatch, None, 404)
    for _ in range(2):
        assert resolve_issuer_classification(cik="7654321", texts=["widgets"], cfg=cfg) == (
            "operating",
            "substring:NO_REGISTRANT_ROW",
        )
    # A failed fetch is not retried on every classification call.
    assert calls == ["https://data.sec.gov/submissions/CIK0007654321.json"]
