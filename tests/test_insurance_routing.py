from __future__ import annotations

import json

from app.db import get_db, init_db, utc_now_iso


def _init_temp_db(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    monkeypatch.setenv("VOE_SAFE_MODE", "true")
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _write_submission(tmp_path, *, cik: str, name: str, tickers: list[str]) -> None:
    path = tmp_path / "data" / "cache" / "submissions" / f"{cik.zfill(10)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"name": name, "tickers": tickers, "exchanges": ["NYSE"] * len(tickers)}),
        encoding="utf-8",
    )


def _seed_company(
    ticker: str,
    *,
    name: str,
    sector: str | None = "insurance",
    filing_text: str | None = None,
    tmp_path=None,
    submission_tickers: list[str] | None = None,
) -> None:
    now = utc_now_iso()
    local_path = None
    cik = f"000{len(ticker):07d}"
    if filing_text is not None and tmp_path is not None:
        local_path = tmp_path / f"{ticker.lower()}_10k.txt"
        local_path.write_text(filing_text, encoding="utf-8")
    if tmp_path is not None and submission_tickers is not None:
        _write_submission(tmp_path, cik=cik, name=name, tickers=submission_tickers)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO companies(ticker, cik, name, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (ticker, cik, name, now),
        )
        if sector is not None:
            conn.execute(
                """
                INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at)
                VALUES(?, ?, ?, 1.0, '[]', ?)
                """,
                (ticker, "2026-01-01", sector, now),
            )
        if local_path is not None:
            conn.execute(
                """
                INSERT INTO filings(
                    cik, ticker, accession, form_type, filing_date, period_end,
                    primary_doc_url, local_path, status, created_at, updated_at
                )
                VALUES(?, ?, ?, '10-K', '2026-02-01', '2025-12-31', ?, ?, 'OK', ?, ?)
                """,
                (cik, ticker, f"{ticker}-2026", "https://example.test/10k", str(local_path), now, now),
            )


def _seed_filing_only(ticker: str, *, cik: str, filing_text: str, tmp_path) -> None:
    now = utc_now_iso()
    local_path = tmp_path / f"{ticker.lower()}_10k.txt"
    local_path.write_text(filing_text, encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at)
            VALUES(?, ?, 'insurance', 1.0, '[]', ?)
            """,
            (ticker, "2026-01-01", now),
        )
        conn.execute(
            """
            INSERT INTO filings(
                cik, ticker, accession, form_type, filing_date, period_end,
                primary_doc_url, local_path, status, created_at, updated_at
            )
            VALUES(?, ?, ?, '10-K', '2026-02-01', '2025-12-31', ?, ?, 'OK', ?, ?)
            """,
            (cik, ticker, f"{ticker}-2026", "https://example.test/10k", str(local_path), now, now),
        )


def test_common_insurer_routes_to_insurance_common(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "INSR",
        name="Example Mutual Insurance Corp",
        filing_text="The company writes life insurance and annuity products. LDTI market risk benefits apply.",
        tmp_path=tmp_path,
        submission_tickers=["INSR"],
    )

    from app.insurance.routing import route_security

    route = route_security("INSR", as_of_date="2026-02-01")

    assert route.security_type == "common"
    assert route.security_identity_status == "VERIFIED_COMMON_PRIMARY_TICKER"
    assert route.issuer_type == "insurance_underwriter"
    assert route.insurance_subtype == "life_annuity"
    assert route.accounting_regime == "US_GAAP_LDTI"
    assert route.model_status == "ROUTED"


def test_depositary_preferred_routes_to_preferred_model(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "INSP",
        name="Example Insurance Depositary Shares 6.000% Non-Cumulative Preferred Series A",
        filing_text="Depositary shares represent preferred stock with a $25 liquidation preference.",
        tmp_path=tmp_path,
        submission_tickers=["INSP"],
    )

    from app.insurance.routing import route_security

    route = route_security("INSP", as_of_date="2026-02-01")

    assert route.security_type == "depositary_preferred"
    assert route.issuer_type == "insurance_underwriter"
    assert route.model_status == "ROUTED"
    assert "SECURITY_IS_DEPOSITARY_PREFERRED" in route.reason_codes


def test_missing_security_evidence_blocks_unknown_security(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache",
        lambda http=None: {},
    )

    from app.insurance.routing import route_security

    route = route_security("VOID", as_of_date="2026-02-01")

    assert route.security_type == "SECURITY_TYPE_UNKNOWN"
    assert route.model_status == "MODEL_BLOCKED"
    assert route.reason_codes == ["SECURITY_TYPE_UNKNOWN", "ISSUER_TYPE_UNKNOWN"]


def test_broker_services_do_not_route_to_underwriting_model(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "BRKR",
        name="Example Insurance Brokerage Inc",
        filing_text="Revenue is primarily brokerage commissions and risk consulting fees.",
        tmp_path=tmp_path,
        submission_tickers=["BRKR"],
    )

    from app.insurance.routing import route_security

    route = route_security("BRKR", as_of_date="2026-02-01")

    assert route.security_type == "common"
    assert route.issuer_type == "insurance_services"
    assert route.insurance_subtype == "broker_services"
    assert route.model_status == "NOT_APPLICABLE"


def test_non_insurance_sector_incidental_insurance_words_do_not_route_to_underwriter(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "SOFT",
        name="Example Cloud Workflow Inc",
        sector="enterprise_software",
        filing_text=(
            "We maintain corporate insurance policies and may submit claims after losses. "
            "Some customers are insurance companies. Revenue comes from cloud software subscriptions "
            "and professional services."
        ),
        tmp_path=tmp_path,
        submission_tickers=["SOFT"],
    )

    from app.insurance.routing import route_security

    route = route_security("SOFT", as_of_date="2026-02-01")

    assert route.security_type == "common"
    assert route.issuer_type == "non_insurer"
    assert route.insurance_subtype is None
    assert route.model_status == "NOT_APPLICABLE"
    assert "text:insurance_underwriting_evidence" not in route.derived_from


def test_multi_security_cik_non_primary_ticker_blocks_unknown_security(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "ACGLN",
        name="Arch Capital Group Ltd.",
        filing_text="The company writes property and casualty insurance and reinsurance.",
        tmp_path=tmp_path,
        submission_tickers=["ACGL", "ACGLO", "ACGLN"],
    )

    from app.insurance.routing import route_security

    route = route_security("ACGLN", as_of_date="2026-02-01")

    assert route.security_type == "SECURITY_TYPE_UNKNOWN"
    assert route.security_identity_status == "UNVERIFIED_MULTI_SECURITY_NON_PRIMARY"
    assert route.issuer_primary_ticker == "ACGL"
    assert route.issuer_listed_tickers == ["ACGL", "ACGLO", "ACGLN"]
    assert route.model_status == "MODEL_BLOCKED"
    assert "MULTI_SECURITY_CIK_NON_PRIMARY_TICKER" in route.reason_codes
    assert "SECURITY_IDENTITY_UNVERIFIED" in route.reason_codes


def test_primary_ticker_from_submissions_routes_common_without_company_row(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _write_submission(tmp_path, cik="1234567", name="Example P&C Group Ltd.", tickers=["EPCL"])
    _seed_filing_only(
        "EPCL",
        cik="1234567",
        filing_text="The company writes property and casualty insurance with claims and loss reserves.",
        tmp_path=tmp_path,
    )

    from app.insurance.routing import route_security

    route = route_security("EPCL", as_of_date="2026-02-01")

    assert route.company_name == "Example P&C Group Ltd."
    assert route.security_type == "common"
    assert route.security_identity_status == "VERIFIED_COMMON_PRIMARY_TICKER"
    assert route.issuer_type == "insurance_underwriter"
    assert route.model_status == "ROUTED"


def test_primary_common_not_reclassified_by_issuer_preferred_boilerplate(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "ICOM",
        name="Example Insurance Common Corp",
        filing_text=(
            "The company writes insurance. The issuer may have preferred stock and depositary shares "
            "outstanding with liquidation preferences, but this annual report is for the issuer."
        ),
        tmp_path=tmp_path,
        submission_tickers=["ICOM"],
    )

    from app.insurance.routing import route_security

    route = route_security("ICOM", as_of_date="2026-02-01")

    assert route.security_type == "common"
    assert route.security_identity_status == "VERIFIED_COMMON_PRIMARY_TICKER"
    assert route.model_status == "ROUTED"


def test_mortgage_insurer_subtype_beats_captive_reinsurance_language(monkeypatch, tmp_path):
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "MINS",
        name="Example Mortgage Insurance Holdings Inc",
        filing_text=(
            "The company writes primary mortgage insurance. It uses captive reinsurance and ceded premiums "
            "as part of its capital management program."
        ),
        tmp_path=tmp_path,
        submission_tickers=["MINS"],
    )

    from app.insurance.routing import route_security

    route = route_security("MINS", as_of_date="2026-02-01")

    assert route.security_type == "common"
    assert route.insurance_subtype == "title_mortgage_specialty"


def test_suffix_unit_hint_overridden_for_primary_ticker(monkeypatch, tmp_path):
    # KEQU-class false positive: an operating common whose ticker merely ends
    # in "U". The issuer's submissions profile lists it as the only ticker, so
    # the suffix hint must yield to the registry evidence.
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "XQRU",
        name="Xqru Scientific Corporation",
        sector="industrial_tech",
        filing_text="The company manufactures laboratory furniture and fume hoods.",
        tmp_path=tmp_path,
        submission_tickers=["XQRU"],
    )

    from app.insurance.routing import route_security

    route = route_security("XQRU", as_of_date="2026-02-01")

    assert route.security_type == "common"
    assert route.security_identity_status == "VERIFIED_COMMON_PRIMARY_TICKER_SUFFIX_OVERRIDE"
    assert route.model_status == "NOT_APPLICABLE"
    assert "SECURITY_IS_UNIT" not in route.reason_codes
    assert "submission_cache:primary_ticker_overrides_suffix_hint" in route.derived_from


def test_suffix_unit_hint_stands_for_real_spac_unit(monkeypatch, tmp_path):
    # A genuine SPAC unit trades alongside the issuer's base ticker; the
    # suffix hint must NOT be overridden when the ticker is non-primary.
    _init_temp_db(monkeypatch, tmp_path)
    _seed_company(
        "XYZU",
        name="Xyz Holdings Inc",
        sector=None,
        filing_text="",
        tmp_path=tmp_path,
        submission_tickers=["XYZ", "XYZU", "XYZW"],
    )

    from app.insurance.routing import route_security

    route = route_security("XYZU", as_of_date="2026-02-01")

    assert route.security_type == "unit"
    assert route.security_identity_status == "EXPLICIT_NON_COMMON"
    assert "SECURITY_IS_UNIT" in route.reason_codes


def test_cik_registry_fallback_resolves_submissions_profile(monkeypatch, tmp_path):
    # Census-discovered name: no companies row, no cached filing. The CIK
    # registry fallback must still surface the submissions identity profile
    # so the suffix override can fire.
    cfg = _init_temp_db(monkeypatch, tmp_path)
    from app.db import get_db as _get_db

    with _get_db() as conn:
        conn.execute(
            "INSERT INTO sector_inference(ticker, as_of_date, inferred_sector, score, derived_from, created_at) "
            "VALUES('NWCU', '2026-01-01', 'industrial_tech', 1.0, '[]', 'x')"
        )
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "company_tickers.json").write_text(
        json.dumps({"0": {"cik_str": 77, "ticker": "NWCU", "title": "Newco Industries"}}),
        encoding="utf-8",
    )
    _write_submission(tmp_path, cik="77", name="Newco Industries Inc", tickers=["NWCU"])

    from app.insurance.routing import route_security

    route = route_security("NWCU", as_of_date="2026-02-01")

    assert route.security_type == "common"
    assert route.security_identity_status == "VERIFIED_COMMON_PRIMARY_TICKER_SUFFIX_OVERRIDE"
    assert route.issuer_primary_ticker == "NWCU"
