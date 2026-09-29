from __future__ import annotations

from app.config import get_config
from app.db import init_db
from app.dossier.runner import resume_dossier_run, run_dossier_for_peer_set
from app.util.http import DomainBudgetExceeded


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    cfg = _get_config()
    init_db(cfg)
    return cfg


def _skip_payload(*, ticker: str, as_of_date: str, years_back: int) -> tuple[list, dict]:
    return (
        [],
        {
            "ticker": ticker,
            "query": {"as_of_date": as_of_date, "years_back": years_back},
            "counts_by_form_type": {},
            "selected_accessions": [],
            "missing_years": [],
            "eligible": False,
            "skip_reason": "INSUFFICIENT_ANNUAL_FILINGS_IN_WINDOW",
        },
    )


def test_sec_budget_override_reflected_and_avoids_immediate_budget_failure(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)

    def _budget_sensitive_collect(**kwargs):
        if int(get_config().max_requests_data_sec_domain) <= 1200:
            raise DomainBudgetExceeded("Domain budget exceeded for data.sec.gov (1200 requests)")
        return _skip_payload(
            ticker=str(kwargs["ticker"]),
            as_of_date=str(kwargs["as_of_date"]),
            years_back=int(kwargs["years_back"]),
        )

    monkeypatch.setattr("app.dossier.runner._collect_stage1_for_ticker", _budget_sensitive_collect)

    without_override = run_dossier_for_peer_set(
        tickers=["AAA"],
        as_of_date="2026-02-13",
        years_back=10,
        run_id="dossier_budget_no_override",
        workers=1,
    )
    assert without_override["ticker_results"]["AAA"]["status"] == "SKIPPED_BUDGET"

    with_override = run_dossier_for_peer_set(
        tickers=["AAA"],
        as_of_date="2026-02-13",
        years_back=10,
        run_id="dossier_budget_with_override",
        workers=1,
        sec_budget=1500,
    )
    assert with_override["ticker_results"]["AAA"]["status"] == "SKIPPED"
    assert with_override["sec_budget"]["requested"] == 1500
    assert with_override["sec_budget"]["effective"]["data.sec.gov"] >= 1500
    assert with_override["sec_budget"]["effective"]["www.sec.gov"] >= 1500


def test_resume_retries_skipped_budget_tickers(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    run_id = "dossier_budget_resume_test"

    def _raise_budget_once(**kwargs):
        raise DomainBudgetExceeded("Domain budget exceeded for data.sec.gov (1200 requests)")

    monkeypatch.setattr("app.dossier.runner._collect_stage1_for_ticker", _raise_budget_once)
    first = run_dossier_for_peer_set(
        tickers=["AAA", "BBB"],
        as_of_date="2026-02-13",
        years_back=10,
        run_id=run_id,
        workers=1,
    )
    assert first["ticker_results"]["AAA"]["status"] == "SKIPPED_BUDGET"
    assert first["ticker_results"]["BBB"]["status"] == "SKIPPED_BUDGET"

    monkeypatch.setattr(
        "app.dossier.runner._collect_stage1_for_ticker",
        lambda **kwargs: _skip_payload(
            ticker=str(kwargs["ticker"]),
            as_of_date=str(kwargs["as_of_date"]),
            years_back=int(kwargs["years_back"]),
        ),
    )
    resumed = resume_dossier_run(run_id=run_id, workers=1)
    assert resumed["ticker_results"]["AAA"]["status"] == "SKIPPED"
    assert resumed["ticker_results"]["BBB"]["status"] == "SKIPPED"
