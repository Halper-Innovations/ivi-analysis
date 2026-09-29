from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.config import get_config
from app.db import init_db
from app.valuation.valuation_writer import (
    _compute_pricing_zone,
    _load_valuation_facts,
    ensure_valuation,
    valuation_facts_fingerprint,
)
from tests.test_valuation_writer import _make_conn, _seed_companyfacts

_AS_OF_DATE = "2026-03-19"
_CIK = "0000000042"
_SOURCE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json"


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _raw_entry(
    value: float,
    *,
    fiscal_year: int = 2024,
    filed_date: str = "2025-02-15",
) -> dict[str, object]:
    return {
        "end": f"{fiscal_year}-12-31",
        "val": value,
        "accn": f"0000000042-{str(fiscal_year)[2:]}-000001",
        "fy": fiscal_year,
        "fp": "FY",
        "form": "10-K",
        "filed": filed_date,
    }


def _concept(value: float, *, fiscal_year: int = 2024) -> dict[str, object]:
    return {"units": {"USD": [_raw_entry(value, fiscal_year=fiscal_year)]}}


def _write_companyfacts(
    cfg,
    *,
    include_equity: bool = True,
    include_liabilities: bool = True,
    extra_concepts: dict[str, dict[str, object]] | None = None,
) -> None:
    concepts: dict[str, dict[str, object]] = {}
    if include_equity:
        concepts["StockholdersEquity"] = _concept(6_000_000_000.0)
    if include_liabilities:
        concepts["Liabilities"] = _concept(8_000_000_000.0)
    concepts.update(extra_concepts or {})
    companyfacts = {
        "cik": "42",
        "entityName": "Fixture Issuer",
        "facts": {"us-gaap": concepts},
    }
    cache_path = cfg.cache_dir / "companyfacts" / f"{_CIK}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cik": _CIK,
                "retrieved_at": "2026-03-19T12:00:00+00:00",
                "source_url": _SOURCE_URL,
                "http_status": 200,
                "companyfacts": companyfacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# Named non-debt lines of the same filing that add up to the 8,000,000,000
# liabilities total (the complete-liabilities test).
_COMPLETE_LIABILITY_LINES = {
    "AccountsPayableCurrent": _concept(5_000_000_000.0),
    "AccruedLiabilitiesCurrent": _concept(2_000_000_000.0),
    "OperatingLeaseLiabilityNoncurrent": _concept(1_000_000_000.0),
}


def _seed_strict_fixture(conn, *, ticker: str) -> None:
    _seed_companyfacts(conn, ticker=ticker)
    conn.execute(
        """
        UPDATE companyfacts_facts
        SET source_url = ?,
            form = '10-K',
            filed_date = '2025-02-15'
        WHERE ticker = ?
        """,
        (_SOURCE_URL, ticker),
    )
    conn.commit()


def _load(conn, cfg, *, ticker: str):
    return _load_valuation_facts(
        ticker,
        conn,
        as_of_date=_AS_OF_DATE,
        issuer_cik=_CIK,
        issuer_aliases=(ticker,),
        cfg=cfg,
    )


def _run_writer(conn, cfg, *, ticker: str) -> tuple[dict, dict]:
    with patch("app.valuation.valuation_writer.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda _self: conn
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        records = ensure_valuation(
            ticker,
            _AS_OF_DATE,
            provider=None,
            price_override=50.0,
            force_refresh=True,
            cfg=cfg,
            issuer_cik=_CIK,
            issuer_aliases=(ticker,),
            require_filed_asof=True,
            raise_on_error=True,
        )
    assert records
    row = conn.execute(
        """
        SELECT inputs_json, outputs_json
        FROM valuations
        WHERE ticker = ? AND method = 'scorecard'
        """,
        (ticker,),
    ).fetchone()
    assert row is not None
    return json.loads(row["inputs_json"]), json.loads(row["outputs_json"])


def test_lkq_shaped_missing_preferred_is_evidenced_zero_and_numeric(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="LKQX")
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = 'LKQX' AND line_item = 'preferred_equity'"
    )
    conn.execute(
        """
        UPDATE companyfacts_facts
        SET value = CASE line_item
            WHEN 'total_debt' THEN 3695.0
            WHEN 'cash' THEN 319.0
            WHEN 'noncontrolling_interest' THEN 24.0
            WHEN 'equity' THEN 6000.0
            WHEN 'total_liabilities' THEN 8000.0
            WHEN 'shares_outstanding' THEN 300.0
            WHEN 'cfo' THEN value * 20.0
            WHEN 'capex' THEN value * 20.0
            WHEN 'operating_income' THEN value * 30.0
            WHEN 'net_income' THEN value * 30.0
            WHEN 'revenue' THEN value * 30.0
            ELSE value
        END
        WHERE ticker = 'LKQX'
        """
    )
    conn.commit()
    _write_companyfacts(
        cfg,
        extra_concepts={
            "LongTermDebt": _concept(3_695_000_000.0),
            "MinorityInterest": _concept(24_000_000.0),
        },
    )

    inputs, outputs = _run_writer(conn, cfg, ticker="LKQX")
    assert inputs["net_debt"] == 3400.0
    # MARGIN_OF_SAFETY since 2026-09-02: earnings power applies the
    # through-cycle MARGIN to CURRENT revenue instead of averaging past
    # operating-income levels, and this fixture's revenue rises, so its EPV
    # rises above the price and the name moves out of the growth-dependent
    # zone. The subject of this test — the evidenced-zero preferred claim and
    # the net-debt bridge below — is unchanged.
    assert outputs["pricing_zone"] == "MARGIN_OF_SAFETY"
    assert outputs["signal"] == "HOLD"
    assert outputs["quality_context"]["net_debt_flags"] == ["SENIOR_CLAIMS_DEDUCTED"]
    proof = inputs["evidenced_zero_facts"]
    assert len(proof) == 1
    assert proof[0]["line_item"] == "preferred_equity"
    assert proof[0]["fiscal_year"] == 2024
    assert proof[0]["value"] == 0.0
    assert proof[0]["derivation"] == "EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE"
    assert proof[0]["basis_row"]["line_item"] == "equity"
    assert proof[0]["basis_row"]["value"] == 6000.0
    assert proof[0]["absent_concepts"] == [
        "PreferredStockValue",
        "PreferredStockValueOutstanding",
        "TemporaryEquityCarryingAmountAttributableToParent",
    ]
    assert len(proof[0]["materialized_companyfacts"]["sha256"]) == 64
    stored_fingerprint = inputs["facts_fingerprint"]
    assert stored_fingerprint == valuation_facts_fingerprint(
        "LKQX",
        conn,
        as_of_date=_AS_OF_DATE,
        issuer_cik=_CIK,
        issuer_aliases=("LKQX",),
    )
    materialized_path = Path(proof[0]["materialized_companyfacts"]["path"])
    materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    materialized["retrieved_at"] = "2026-03-19T12:00:01+00:00"
    materialized_path.write_text(json.dumps(materialized, indent=2), encoding="utf-8")
    assert (
        valuation_facts_fingerprint(
            "LKQX",
            conn,
            as_of_date=_AS_OF_DATE,
            issuer_cik=_CIK,
            issuer_aliases=("LKQX",),
        )
        != stored_fingerprint
    )


def test_prior_positive_preferred_stays_senior_claims_unknown(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="PRIORX")
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = 'PRIORX' AND line_item = 'preferred_equity'"
    )
    conn.execute(
        """
        INSERT INTO companyfacts_facts(
            ticker, fiscal_year, period_type, period_end, line_item, value,
            units, source_url, fetched_at, filed_date, form, accession
        ) VALUES(
            'PRIORX', 2023, 'FY', '2023-12-31', 'preferred_equity', 17.0,
            'USD_millions', ?, '2025-02-15T00:00:00+00:00',
            '2024-02-15', '10-K', '0000000042-23-000001'
        )
        """,
        (_SOURCE_URL,),
    )
    conn.commit()
    _write_companyfacts(
        cfg,
        extra_concepts={"PreferredStockValue": _concept(17_000_000.0, fiscal_year=2023)},
    )

    facts, proofs = _load(conn, cfg, ticker="PRIORX")
    assert facts["preferred_equity"] == [(2023, 17.0)]
    assert not any(proof["line_item"] == "preferred_equity" for proof in proofs)
    inputs, outputs = _run_writer(conn, cfg, ticker="PRIORX")
    assert inputs["net_debt"] == "UNKNOWN"
    assert outputs["pricing_zone"] == "INSUFFICIENT_DATA"
    assert outputs["quality_context"]["net_debt_flags"] == [
        "SENIOR_CLAIMS_UNKNOWN",
        "NET_DEBT_UNKNOWN",
    ]


def test_negative_nci_stays_blocked_by_frozen_nonnegative_contract(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="BYDX")
    conn.execute(
        "UPDATE companyfacts_facts SET value = -0.617 "
        "WHERE ticker = 'BYDX' AND line_item = 'noncontrolling_interest'"
    )
    conn.commit()
    _write_companyfacts(
        cfg,
        extra_concepts={"MinorityInterest": _concept(-617_000.0)},
    )

    facts, proofs = _load(conn, cfg, ticker="BYDX")
    assert facts["noncontrolling_interest"] == [(2024, -0.617)]
    assert proofs == []
    inputs, outputs = _run_writer(conn, cfg, ticker="BYDX")
    assert inputs["net_debt"] == "UNKNOWN"
    assert outputs["pricing_zone"] == "INSUFFICIENT_DATA"
    assert outputs["quality_context"]["net_debt_flags"] == [
        "SENIOR_CLAIMS_UNKNOWN",
        "NET_DEBT_UNKNOWN",
    ]


def test_missing_equity_aggregate_never_qualifies_senior_claim_zero(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="NOEQX")
    conn.execute(
        "DELETE FROM companyfacts_facts "
        "WHERE ticker = 'NOEQX' AND line_item IN "
        "('equity', 'preferred_equity', 'noncontrolling_interest')"
    )
    conn.commit()
    _write_companyfacts(cfg, include_equity=False)

    facts, proofs = _load(conn, cfg, ticker="NOEQX")
    assert "preferred_equity" not in facts
    assert "noncontrolling_interest" not in facts
    assert proofs == []
    inputs, outputs = _run_writer(conn, cfg, ticker="NOEQX")
    assert inputs["net_debt"] == "UNKNOWN"
    assert outputs["pricing_zone"] == "INSUFFICIENT_DATA"


def test_null_normalized_senior_claim_row_is_not_concept_absence(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="NULLPREF")
    conn.execute(
        "UPDATE companyfacts_facts SET value = NULL "
        "WHERE ticker = 'NULLPREF' AND line_item = 'preferred_equity'"
    )
    conn.commit()
    _write_companyfacts(cfg)

    facts, proofs = _load(conn, cfg, ticker="NULLPREF")
    assert "preferred_equity" not in facts
    assert not any(proof["line_item"] == "preferred_equity" for proof in proofs)


def test_debt_absence_with_complete_liabilities_mints_provenance_zero(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="DEBTFREE")
    # Migrated for the complete-liabilities test: the
    # filing now also reports the named non-debt lines that add up to its
    # liabilities total, and the absence scan covers the facts spine's debt
    # families after the tag list.
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = 'DEBTFREE' AND line_item = 'total_debt'"
    )
    conn.commit()
    _write_companyfacts(cfg, extra_concepts=_COMPLETE_LIABILITY_LINES)

    facts, proofs = _load(conn, cfg, ticker="DEBTFREE")
    assert facts["total_debt"] == [(2024, 0.0)]
    debt_proof = next(proof for proof in proofs if proof["line_item"] == "total_debt")
    assert debt_proof["derivation"] == "EVIDENCED_ZERO_DEBT_INSTRUMENT_ABSENCE"
    assert debt_proof["basis_row"]["line_item"] == "total_liabilities"
    assert debt_proof["basis_row"]["value"] == 300.0
    assert debt_proof["liabilities_components"] == [
        {"concept": "AccountsPayableCurrent", "value": 5_000_000_000.0},
        {"concept": "AccruedLiabilitiesCurrent", "value": 2_000_000_000.0},
        {"concept": "OperatingLeaseLiabilityNoncurrent", "value": 1_000_000_000.0},
    ]
    for family_member in (
        "LinesOfCreditCurrent",
        "LongTermLineOfCredit",
        "LoansPayable",
        "LongTermLoansPayable",
        "SecuredDebt",
        "UnsecuredDebt",
        "SubordinatedDebt",
        "JuniorSubordinatedNotes",
        "Deposits",
    ):
        assert family_member in debt_proof["absent_concepts"]
    assert debt_proof["absent_concepts"][:35] == [
        "Debt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtCurrent",
        "DebtAndCapitalLeaseObligations",
        "DebtCurrent",
        "DebtLongtermAndShorttermCombinedAmount",
        "DebtInstrumentCarryingAmount",
        "Borrowings",
        "ShortTermBorrowings",
        "ShortTermDebt",
        "CommercialPaper",
        "NotesPayable",
        "NotesPayableCurrent",
        "NotesAndLoansPayableLongtermPortion",
        "SeniorNotes",
        "SeniorNotesNoncurrent",
        "ConvertibleLongTermNotesPayable",
        "ConvertibleNotesPayable",
        "ConvertibleDebtNoncurrent",
        "ConvertibleDebt",
        "ConvertibleSubordinatedDebtNoncurrent",
        "LongTermLoansFromBank",
        "LineOfCredit",
        "FinanceLeaseLiability",
        "FinanceLeaseLiabilityCurrent",
        "FinanceLeaseLiabilityNoncurrent",
        "CapitalLeaseObligations",
        "CapitalLeaseObligationsCurrent",
        "CapitalLeaseObligationsNoncurrent",
    ]
    assert debt_proof["basis_raw_records"][0]["concept"] == "Liabilities"
    assert len(debt_proof["materialized_companyfacts"]["sha256"]) == 64


def test_debt_concept_or_missing_liabilities_keeps_debt_absent(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="DEBTBLOCK")
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = 'DEBTBLOCK' AND line_item = 'total_debt'"
    )
    conn.commit()
    _write_companyfacts(
        cfg,
        extra_concepts={"FinanceLeaseLiability": _concept(9_000_000.0)},
    )

    facts, proofs = _load(conn, cfg, ticker="DEBTBLOCK")
    assert "total_debt" not in facts
    assert not any(proof["line_item"] == "total_debt" for proof in proofs)

    conn.execute(
        "DELETE FROM companyfacts_facts "
        "WHERE ticker = 'DEBTBLOCK' AND line_item = 'total_liabilities'"
    )
    conn.commit()
    _write_companyfacts(cfg, include_liabilities=False)
    facts, proofs = _load(conn, cfg, ticker="DEBTBLOCK")
    assert "total_debt" not in facts
    assert not any(proof["line_item"] == "total_debt" for proof in proofs)


def test_debt_zero_needs_liabilities_the_named_lines_fully_account_for(
    monkeypatch,
    tmp_path,
):
    """The complete-liabilities test: no debt tag is not enough.
    Named non-debt lines of 7,000,000,000 against a total of 8,000,000,000 leave
    1,000,000,000 unexplained, which could be debt under a line nothing here
    reads, so no zero is minted."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="UNFOOTED")
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = 'UNFOOTED' AND line_item = 'total_debt'"
    )
    conn.commit()
    _write_companyfacts(
        cfg,
        extra_concepts={
            "AccountsPayableCurrent": _concept(5_000_000_000.0),
            "AccruedLiabilitiesCurrent": _concept(2_000_000_000.0),
        },
    )

    facts, proofs = _load(conn, cfg, ticker="UNFOOTED")
    assert "total_debt" not in facts
    assert not any(proof["line_item"] == "total_debt" for proof in proofs)


def test_debt_reported_only_under_a_revolver_or_loan_family_is_not_debt_free(
    monkeypatch,
    tmp_path,
):
    """The tag list missed the facts spine's gap families. A
    filer reporting its debt only as a long-term line of credit, a loan
    payable, secured or subordinated debt passed as debt-free."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    for index, concept in enumerate(
        ("LongTermLineOfCredit", "LinesOfCreditCurrent", "LoansPayable", "SecuredDebt",
         "SubordinatedDebt")
    ):
        ticker = f"FAM{index}"
        conn = _make_conn()
        _seed_strict_fixture(conn, ticker=ticker)
        conn.execute(
            "DELETE FROM companyfacts_facts WHERE ticker = ? AND line_item = 'total_debt'",
            (ticker,),
        )
        conn.commit()
        _write_companyfacts(
            cfg,
            extra_concepts={**_COMPLETE_LIABILITY_LINES, concept: _concept(400_000_000.0)},
        )

        facts, proofs = _load(conn, cfg, ticker=ticker)
        assert "total_debt" not in facts, concept
        assert not any(proof["line_item"] == "total_debt" for proof in proofs), concept


def _seed_prior_nci(conn, *, ticker: str, fiscal_year: int) -> None:
    conn.execute(
        "DELETE FROM companyfacts_facts "
        "WHERE ticker = ? AND line_item = 'noncontrolling_interest'",
        (ticker,),
    )
    conn.execute(
        """
        INSERT INTO companyfacts_facts(
            ticker, fiscal_year, period_type, period_end, line_item, value,
            units, source_url, fetched_at, filed_date, form, accession
        ) VALUES(
            ?, ?, 'FY', ?, 'noncontrolling_interest', 5.0,
            'USD_millions', ?, '2025-02-15T00:00:00+00:00',
            ?, '10-K', ?
        )
        """,
        (
            ticker,
            fiscal_year,
            f"{fiscal_year}-12-31",
            _SOURCE_URL,
            f"{fiscal_year + 1}-02-15",
            f"0000000042-{str(fiscal_year)[2:]}-000001",
        ),
    )
    conn.commit()


def test_a_stale_minority_interest_does_not_block_net_debt(
    monkeypatch,
    tmp_path,
):
    """A minority interest last reported in 2012 blocked the 2024 net debt
    (the Lululemon shape): the prior-positive rule had no horizon. Every filing
    since presented equity without it, so for 2024 it is absent, not blocking;
    the same-year conditions (equity reported, no claim row, no claim
    concept in the year) still apply."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="STALENCI")
    _seed_prior_nci(conn, ticker="STALENCI", fiscal_year=2012)
    _write_companyfacts(
        cfg,
        extra_concepts={
            "LongTermDebt": _concept(50_000_000.0),
            "MinorityInterest": _concept(5_000_000.0, fiscal_year=2012),
        },
    )

    facts, proofs = _load(conn, cfg, ticker="STALENCI")
    assert (2024, 0.0) in facts["noncontrolling_interest"]
    nci_proof = next(proof for proof in proofs if proof["line_item"] == "noncontrolling_interest")
    assert nci_proof["derivation"] == "EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE"
    assert nci_proof["fiscal_year"] == 2024
    inputs, outputs = _run_writer(conn, cfg, ticker="STALENCI")
    assert inputs["net_debt"] == 20.0
    assert "NET_DEBT_UNKNOWN" not in outputs["quality_context"]["net_debt_flags"]


def test_a_recent_minority_interest_still_blocks_and_the_horizon_is_three_years(
    monkeypatch,
    tmp_path,
):
    """Control for the stale rule: a claim reported one or two fiscal years
    before the net-debt year still blocks; three years before is stale."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    for prior_year, blocks in ((2023, True), (2022, True), (2021, False)):
        ticker = f"NCI{prior_year}"
        conn = _make_conn()
        _seed_strict_fixture(conn, ticker=ticker)
        _seed_prior_nci(conn, ticker=ticker, fiscal_year=prior_year)
        _write_companyfacts(
            cfg,
            extra_concepts={"MinorityInterest": _concept(5_000_000.0, fiscal_year=prior_year)},
        )
        facts, proofs = _load(conn, cfg, ticker=ticker)
        minted = any(proof["line_item"] == "noncontrolling_interest" for proof in proofs)
        assert minted is (not blocks), prior_year
        assert ((2024, 0.0) in facts["noncontrolling_interest"]) is (not blocks), prior_year


def test_debt_zero_is_not_minted_for_an_irrelevant_older_gap(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="OLDGAP")
    for line_item, value in (("cash", 25.0), ("total_liabilities", 290.0)):
        conn.execute(
            """
            INSERT INTO companyfacts_facts(
                ticker, fiscal_year, period_type, period_end, line_item, value,
                units, source_url, fetched_at, filed_date, form, accession
            ) VALUES(
                'OLDGAP', 2023, 'FY', '2023-12-31', ?, ?,
                'USD_millions', ?, '2024-02-15T00:00:00+00:00',
                '2024-02-15', '10-K', '0000000042-23-000001'
            )
            """,
            (line_item, value, _SOURCE_URL),
        )
    conn.commit()
    _write_companyfacts(cfg)

    facts, proofs = _load(conn, cfg, ticker="OLDGAP")
    assert facts["total_debt"] == [(2024, 50.0)]
    assert not any(proof["line_item"] == "total_debt" for proof in proofs)


def test_missing_cash_never_becomes_evidenced_zero(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="NOCASH")
    conn.execute("DELETE FROM companyfacts_facts WHERE ticker = 'NOCASH' AND line_item = 'cash'")
    conn.commit()
    _write_companyfacts(
        cfg,
        extra_concepts={"LongTermDebt": _concept(50_000_000.0)},
    )

    facts, proofs = _load(conn, cfg, ticker="NOCASH")
    assert "cash" not in facts
    assert not any(proof["line_item"] == "cash" for proof in proofs)
    inputs, outputs = _run_writer(conn, cfg, ticker="NOCASH")
    assert inputs["net_debt"] == "UNKNOWN"
    assert outputs["pricing_zone"] == "INSUFFICIENT_DATA"
    assert outputs["quality_context"]["net_debt_flags"] == ["NET_DEBT_UNKNOWN"]


def test_valuation_blocked_gate_remains_dominant_with_qualified_zero(
    monkeypatch,
    tmp_path,
):
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="BLOCKX")
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = 'BLOCKX' AND line_item = 'preferred_equity'"
    )
    conn.commit()
    _write_companyfacts(cfg)
    blocked_context = {
        "gate_action": "BLOCK",
        "gate_reason": "FIXTURE_BLOCK",
        "gate_reason_codes": ["FIXTURE_BLOCK"],
        "confidence_class": "INSUFFICIENT",
        "valuation_headwinds": [],
        "valuation_supports": [],
    }
    with patch(
        "app.valuation.pre_valuation_gate.compute_quality_context",
        return_value=blocked_context,
    ):
        inputs, outputs = _run_writer(conn, cfg, ticker="BLOCKX")
    assert inputs["evidenced_zero_facts"][0]["line_item"] == "preferred_equity"
    assert outputs["signal"] == "VALUATION_BLOCKED"
    assert outputs["pricing_zone"] == "VALUATION_BLOCKED"
    assert outputs["pricing_zone_detail"]["gate_reason"] == "FIXTURE_BLOCK"


def test_pricing_zone_thresholds_and_suppression_are_unchanged():
    kwargs = {
        "shares": 10.0,
        "adjusted_wacc": 0.10,
        "adjusted_avg_operating_income": 20.0,
        "revenue_latest": 1000.0,
        "net_debt_to_ebitda_proxy": 0.0,
    }
    assert (
        _compute_pricing_zone(
            epv_adjusted=100.0,
            dcf_base=120.0,
            current_price=99.0,
            net_debt=0.0,
            **kwargs,
        )["zone"]
        == "MARGIN_OF_SAFETY"
    )
    assert (
        _compute_pricing_zone(
            epv_adjusted=100.0,
            dcf_base=120.0,
            current_price=100.0,
            net_debt=0.0,
            **kwargs,
        )["zone"]
        == "GROWTH_DEPENDENT"
    )
    assert (
        _compute_pricing_zone(
            epv_adjusted=100.0,
            dcf_base=120.0,
            current_price=120.0,
            net_debt=0.0,
            **kwargs,
        )["zone"]
        == "GROWTH_DEPENDENT"
    )
    assert (
        _compute_pricing_zone(
            epv_adjusted=100.0,
            dcf_base=120.0,
            current_price=121.0,
            net_debt=0.0,
            **kwargs,
        )["zone"]
        == "SPECULATIVE_PREMIUM"
    )
    insufficient = _compute_pricing_zone(
        epv_adjusted=100.0,
        dcf_base=120.0,
        current_price=99.0,
        net_debt=None,
        **kwargs,
    )
    assert insufficient["zone"] == "INSUFFICIENT_DATA"
    assert insufficient["detail"]["reason"] == "Net debt unavailable."


def test_a_positive_claim_reported_at_the_balance_sheet_date_still_refuses(
    monkeypatch,
    tmp_path,
):
    """Control for the at-date rule (2026-09-29 basket): a later 10-Q's
    comparative that reports a POSITIVE claim at the net-debt balance-sheet date
    is never replaced by a zero, even though the normalized annual facts carry
    no row for it; only reported zeros qualify."""
    cfg = _init_cfg(monkeypatch, tmp_path)
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="ATDATE")
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = 'ATDATE' AND line_item = 'preferred_equity'"
    )
    conn.commit()
    comparative = {
        "end": "2024-12-31",
        "val": 17_000_000.0,
        "accn": "0000000042-25-000009",
        "fy": 2025,
        "fp": "Q1",
        "form": "10-Q",
        "filed": "2025-05-01",
    }
    _write_companyfacts(
        cfg,
        extra_concepts={"PreferredStockValue": {"units": {"USD": [comparative]}}},
    )

    facts, proofs = _load(conn, cfg, ticker="ATDATE")
    assert "preferred_equity" not in facts
    assert not any(proof["line_item"] == "preferred_equity" for proof in proofs)

    comparative["val"] = 0
    _write_companyfacts(
        cfg,
        extra_concepts={"PreferredStockValue": {"units": {"USD": [comparative]}}},
    )
    facts, proofs = _load(conn, cfg, ticker="ATDATE")
    assert (2024, 0.0) in facts["preferred_equity"]
    proof = next(proof for proof in proofs if proof["line_item"] == "preferred_equity")
    assert proof["derivation"] == "EVIDENCED_ZERO_SENIOR_CLAIM_REPORTED"


def test_filings_made_after_the_as_of_date_never_decide_an_evidenced_zero(
    monkeypatch,
    tmp_path,
):
    """Point in time: three checks read every filing in the payload, including ones filed
    after the as-of date (2026-03-19). A later 10-Q's positive claim AT the balance-sheet
    date, a later-filed positive claim in the stale window, and a later-filed debt concept
    for the year each refused a zero the filings public at the as-of date support. Only
    as-of-visible records count now."""
    later = "2026-05-01"
    cfg = _init_cfg(monkeypatch, tmp_path)

    # 1. A positive claim at the balance-sheet date, first reported after the as-of date.
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="LATEATDATE")
    conn.execute(
        "DELETE FROM companyfacts_facts "
        "WHERE ticker = 'LATEATDATE' AND line_item = 'preferred_equity'"
    )
    conn.commit()
    late_comparative = {
        "end": "2024-12-31",
        "val": 17_000_000.0,
        "accn": "0000000042-26-000009",
        "fy": 2026,
        "fp": "Q1",
        "form": "10-Q",
        "filed": later,
    }
    _write_companyfacts(
        cfg,
        extra_concepts={"PreferredStockValue": {"units": {"USD": [late_comparative]}}},
    )
    facts, proofs = _load(conn, cfg, ticker="LATEATDATE")
    assert (2024, 0.0) in facts["preferred_equity"]
    proof = next(proof for proof in proofs if proof["line_item"] == "preferred_equity")
    assert proof["derivation"] == "EVIDENCED_ZERO_SENIOR_CLAIM_ABSENCE"

    # 2. A recent positive minority interest (fiscal 2023) filed only after the as-of date.
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="LATENCI")
    late_nci = {**_raw_entry(5_000_000.0, fiscal_year=2023, filed_date=later)}
    _write_companyfacts(
        cfg, extra_concepts={"MinorityInterest": {"units": {"USD": [late_nci]}}}
    )
    facts, proofs = _load(conn, cfg, ticker="LATENCI")
    assert (2024, 0.0) in facts["noncontrolling_interest"]

    # 3. A debt concept for fiscal 2024 filed only after the as-of date.
    conn = _make_conn()
    _seed_strict_fixture(conn, ticker="LATEDEBT")
    conn.execute(
        "DELETE FROM companyfacts_facts WHERE ticker = 'LATEDEBT' AND line_item = 'total_debt'"
    )
    conn.commit()
    late_debt = _raw_entry(9_000_000.0, filed_date=later)
    _write_companyfacts(
        cfg,
        extra_concepts={
            **_COMPLETE_LIABILITY_LINES,
            "LongTermDebt": {"units": {"USD": [late_debt]}},
        },
    )
    facts, proofs = _load(conn, cfg, ticker="LATEDEBT")
    assert facts["total_debt"] == [(2024, 0.0)]
