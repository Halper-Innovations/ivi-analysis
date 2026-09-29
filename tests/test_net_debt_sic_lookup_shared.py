"""net_debt no longer keeps a private SIC query; it uses the shared registrant lookup."""
from __future__ import annotations

import sqlite3

from app.config import AppConfig
from app.valuation import net_debt


def _cfg(tmp_path):
    return AppConfig(data_dir=tmp_path / "data", db_path=tmp_path / "t.db")


def test_sic_lookup_goes_through_shared_resolver(monkeypatch, tmp_path):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return "3571", "OK"

    monkeypatch.setattr(net_debt, "registrant_sic", fake)
    assert net_debt._sic_for_cik("320193", cfg=_cfg(tmp_path)) == ("3571", "OK")
    assert calls and calls[0]["cik"] == "320193"


def test_empty_cik_keeps_its_reason(tmp_path):
    assert net_debt._sic_for_cik("", cfg=_cfg(tmp_path)) == (None, "NO_CIK")


def test_real_lookup_unchanged_behavior(tmp_path):
    cfg = _cfg(tmp_path)
    con = sqlite3.connect(cfg.db_path)
    con.execute("CREATE TABLE sec_registrants (cik TEXT, sic TEXT, primary_ticker TEXT, all_tickers TEXT)")
    con.execute("INSERT INTO sec_registrants VALUES ('0000320193','3571','AAA','[]')")
    con.execute("INSERT INTO sec_registrants VALUES ('0000000002','','BBB','[]')")
    con.commit()
    con.close()
    assert net_debt._sic_for_cik("320193", cfg=cfg) == ("3571", "OK")
    assert net_debt._sic_for_cik("2", cfg=cfg) == (None, "NO_SIC_ON_FILE")
    assert net_debt._sic_for_cik("999", cfg=cfg) == (None, "NO_REGISTRANT_ROW")
