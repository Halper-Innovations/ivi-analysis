"""The share-count guard, end to end through the as-of resolver.

The defect this pins: ResMed's FY2021 cover-page share count was filed about 1,000x too
small, and the extractor prefers the freshest period end -- which a cover-page instant
always is -- so the spine picked the broken count over the correct balance-sheet one. Every
per-share number divides by that count, so a 1,000x-too-small denominator produces a
1,000x-too-high buy target that looks entirely plausible.

These cases moved from an earlier power-of-ten guard (which compared the freshest cover
count with the freshest balance-sheet count from any filing and always took the balance
sheet) to the share-count guard in app/market/shares_guard.py: same-filing
references, the company's own history, an explicit refusal. Each case below keeps its
scenario; the expectations are the new contract's. The pure guard's own cases are in
tests/test_shares_guard.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db import init_db
from app.market.company_facts_provider import companyfacts_cache_path
from app.valuation.facts import clear_facts_row_cache
from app.valuation.shares import resolve_market_cap_from_price_asof, resolve_shares_asof

RMD_CIK = "0000943819"
RMD_SOURCE_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{RMD_CIK}.json"
FIXTURES = Path(__file__).parent / "fixtures" / "companyfacts_shares_guard"
ACCN = "0000943819-21-000064"


def _init_cfg(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_SEC_USER_AGENT", "IVI tests qa@ivi.test")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    clear_facts_row_cache()
    cfg = _get_config()
    init_db(cfg)
    monkeypatch.setattr(
        "app.valuation.facts.resolve_cik_for_ticker",
        lambda *_args, **_kwargs: RMD_CIK,
    )
    return cfg


def _share_row(value: float, *, end: str, filed: str, accn: str = ACCN) -> dict:
    return {"val": value, "end": end, "filed": filed, "form": "10-K", "accn": accn}


def _write_payload(cfg, companyfacts: dict) -> None:
    path = companyfacts_cache_path(RMD_CIK, cfg=cfg)
    path.write_text(
        json.dumps({"source_url": RMD_SOURCE_URL, "companyfacts": companyfacts}), encoding="utf-8"
    )


def _write_companyfacts_cache(cfg, *, cover_page: dict | None, balance_sheet: dict | None) -> None:
    """Seed the local CompanyFacts cache with a cover-page and/or balance-sheet count."""

    facts: dict = {}
    if cover_page is not None:
        facts["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [cover_page]}}}
    if balance_sheet is not None:
        facts["us-gaap"] = {"CommonStockSharesOutstanding": {"units": {"shares": [balance_sheet]}}}
    _write_payload(cfg, {"facts": facts})


def _slipped_resmed_filing(cfg) -> None:
    # Cover page instant, dated a month after fiscal year end, so it wins on freshness --
    # and is filed 1,000x too small. The balance sheet in the SAME filing is right.
    _write_companyfacts_cache(
        cfg,
        cover_page=_share_row(145_461, end="2021-07-30", filed="2021-08-05"),
        balance_sheet=_share_row(145_461_000, end="2021-06-30", filed="2021-08-05"),
    )


def _bypass_guard(monkeypatch) -> None:
    """The unguarded chooser, as the spine ran before the guard: pass its pick through."""

    monkeypatch.setattr(
        "app.market.company_facts_extract.guard_shares_fact",
        lambda _cf, _asof, *, unguarded, priority: (unguarded, {"outcome": "BYPASSED"}),
    )
    clear_facts_row_cache()


# --- The recorded defect -----------------------------------------------------


def test_resmed_fy2021_cover_page_scale_defect_loses_to_the_balance_sheet(monkeypatch, tmp_path):
    """Cover page filed 145,461 shares where the same filing's balance sheet says 145,461,000."""

    cfg = _init_cfg(monkeypatch, tmp_path)
    _slipped_resmed_filing(cfg)

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 145.461
    assert coverage["shares_value"] == 145.461
    assert coverage["shares_status"] == "OK"
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT_GUARDED"
    guard = coverage["shares_guard"]
    assert guard["outcome"] == "FELL_THROUGH"
    assert [(r["reason"], r["value"]) for r in guard["rejected"]] == [
        ("SHARES_CONTRADICTED", 145_461.0)
    ]
    assert guard["rejected"][0]["references"][0]["basis"] == "balance_sheet"
    assert guard["rejected"][0]["references"][0]["verdict"] == "contradicts"
    assert guard["accepted"]["by"] == "unchecked", "one filing, no history, no second reference"
    # Provenance follows the value: the record points at the balance-sheet fact.
    assert coverage["shares_asof_used"] == "2021-06-30"
    assert coverage["shares_filed_date"] == "2021-08-05"
    assert coverage["raw_shares_source_value"] == 145_461_000.0
    assert coverage["raw_shares_source_unit"] == "shares"
    assert coverage["raw_shares_outstanding_mm"] == 145.461
    assert coverage["shares_source_url"] == RMD_SOURCE_URL
    balance_ref = (
        "companyfacts.us-gaap.CommonStockSharesOutstanding[end_date=2021-06-30,"
        f"unit=shares,filed=2021-08-05,accn={ACCN}]"
    )
    assert balance_ref in coverage["derived_from"]
    assert not any("EntityCommonStockSharesOutstanding" in ref for ref in coverage["derived_from"])
    assert coverage["shares_reason_detail"] == (
        "Resolved shares_outstanding from SEC companyfacts after the share-count guard: "
        "The guard refused 1 fresher share-count candidate(s) (SHARES_CONTRADICTED) and "
        f"accepted {balance_ref}."
    )


def test_before_the_guard_the_same_filing_resolved_a_thousandfold_small(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _slipped_resmed_filing(cfg)
    _bypass_guard(monkeypatch)

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 0.145461
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT"


def test_guard_decision_is_warned_not_silent(monkeypatch, tmp_path, caplog):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _slipped_resmed_filing(cfg)

    with caplog.at_level("WARNING", logger="app.valuation.shares"):
        value, _coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 145.461
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].getMessage() == (
        "shares: RMD share-count guard FELL_THROUGH -- The guard refused 1 fresher share-count "
        "candidate(s) (SHARES_CONTRADICTED) and accepted "
        "companyfacts.us-gaap.CommonStockSharesOutstanding[end_date=2021-06-30,unit=shares,"
        f"filed=2021-08-05,accn={ACCN}]."
    )


def test_fired_guard_keeps_the_downstream_market_cap_chain_consistent(monkeypatch, tmp_path):
    """The swap rewrites raw value, unit and dates together, so market cap still reconciles."""

    cfg = _init_cfg(monkeypatch, tmp_path)
    _slipped_resmed_filing(cfg)

    value, coverage = resolve_market_cap_from_price_asof(
        ticker="RMD",
        as_of_date="2021-08-31",
        price=25.0,
    )

    assert coverage["market_cap_status"] == "OK"
    assert coverage["market_cap_reason_code"] == "OK"
    assert coverage["shares_outstanding"] == 145.461
    assert coverage["raw_shares_outstanding_mm"] == 145.461
    assert coverage["raw_shares_source_value"] == 145_461_000.0
    assert coverage["raw_shares_source_unit"] == "shares"
    assert coverage["shares_asof_used"] == "2021-06-30"
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT_GUARDED"
    assert coverage["shares_guard"]["outcome"] == "FELL_THROUGH"
    assert value == 25.0 * 145.461


# --- The real ResMed payload -------------------------------------------------


def test_resmed_real_payload_resolves_to_145_6_million_not_145_681(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_payload(cfg, json.loads((FIXTURES / "RMD_0000943819.json").read_text(encoding="utf-8")))

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-09-15", run_id=None)

    assert value == 145.648358
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT_GUARDED"
    assert coverage["shares_asof_used"] == "2021-06-30"
    assert coverage["shares_filed_date"] == "2021-08-17"
    assert coverage["raw_shares_source_value"] == 145_648_358.0
    assert [r["reason"] for r in coverage["shares_guard"]["rejected"]] == ["SHARES_DISCONTINUITY"]

    market_cap, cap_coverage = resolve_market_cap_from_price_asof(
        ticker="RMD", as_of_date="2021-09-15", price=290.0
    )
    assert cap_coverage["shares_outstanding"] == 145.648358
    assert market_cap == pytest.approx(290.0 * 145.648358)


def test_resmed_real_payload_before_the_guard(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_payload(cfg, json.loads((FIXTURES / "RMD_0000943819.json").read_text(encoding="utf-8")))
    _bypass_guard(monkeypatch)

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-09-15", run_id=None)

    assert value == 0.145681
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT"


@pytest.mark.parametrize(
    ("fixture", "as_of"),
    [
        ("RMD_0000943819.json", "2026-09-28"),
        ("RMD_0000943819.json", "2021-06-01"),
        ("CPK_0000019745.json", "2026-09-28"),
        ("PKG_0000075677.json", "2025-12-01"),
    ],
)
def test_an_untouched_company_is_byte_identical_to_before(monkeypatch, tmp_path, fixture, as_of):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_payload(cfg, json.loads((FIXTURES / fixture).read_text(encoding="utf-8")))

    guarded_value, guarded = resolve_shares_asof(ticker="RMD", as_of_date=as_of, run_id=None)
    assert guarded["shares_guard"]["outcome"] == "PASS"
    _bypass_guard(monkeypatch)
    before_value, before = resolve_shares_asof(ticker="RMD", as_of_date=as_of, run_id=None)

    assert guarded_value == before_value
    guarded.pop("shares_guard")
    before.pop("shares_guard")
    assert json.dumps(guarded, sort_keys=True) == json.dumps(before, sort_keys=True)


# --- What the guard must NOT reject: ordinary corporate actions ---------------


def test_guard_does_not_touch_an_ordinary_five_percent_difference(monkeypatch, tmp_path):
    """5% of issuance between the balance-sheet date and the cover page is normal."""

    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_companyfacts_cache(
        cfg,
        cover_page=_share_row(105_000_000, end="2021-07-30", filed="2021-08-05"),
        balance_sheet=_share_row(100_000_000, end="2021-06-30", filed="2021-08-05"),
    )

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 105.0
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT"
    assert coverage["shares_guard"]["outcome"] == "PASS"
    assert coverage["shares_guard"]["accepted"]["by"] == "reference"
    assert coverage["shares_asof_used"] == "2021-07-30"


def test_guard_does_not_touch_a_three_for_one_split(monkeypatch, tmp_path):
    """3x sits between the corroboration and contradiction bands: history decides, and
    with no prior filing the count is accepted unchecked -- with both notes on record."""

    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_companyfacts_cache(
        cfg,
        cover_page=_share_row(300_000_000, end="2021-07-30", filed="2021-08-05"),
        balance_sheet=_share_row(100_000_000, end="2021-06-30", filed="2021-08-05"),
    )

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 300.0
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT"
    assert coverage["shares_guard"]["outcome"] == "PASS"
    assert coverage["shares_guard"]["bases_disagreed"] is True
    assert coverage["shares_guard"]["unchecked"] is True


def test_guard_does_not_reject_a_fifty_for_one_split(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    covers = [
        _share_row(10_000_000, end=f"{y}-02-20", filed=f"{y}-02-25", accn=f"a-{y}")
        for y in (2021, 2022, 2023, 2024)
    ] + [_share_row(500_000_000, end="2025-02-20", filed="2025-02-25", accn="a-2025")]
    _write_payload(
        cfg, {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": covers}}}}}
    )

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2025-03-01", run_id=None)

    assert value == 500.0
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT"
    assert coverage["shares_guard"]["rejected"] == []
    assert coverage["shares_guard"]["accepted"]["by"] == "history"


# --- What the guard must catch -------------------------------------------------


def test_guard_rejects_a_cover_page_a_thousand_times_too_small(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_companyfacts_cache(
        cfg,
        cover_page=_share_row(100_000, end="2021-07-30", filed="2021-08-05"),
        balance_sheet=_share_row(100_000_000, end="2021-06-30", filed="2021-08-05"),
    )

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 100.0
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT_GUARDED"
    assert coverage["shares_guard"]["rejected"][0]["reason"] == "SHARES_CONTRADICTED"
    assert coverage["shares_guard"]["rejected"][0]["value"] == 100_000.0
    assert coverage["shares_asof_used"] == "2021-06-30"


def test_guard_rejects_a_cover_page_a_thousand_times_too_large(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_companyfacts_cache(
        cfg,
        cover_page=_share_row(100_000_000_000, end="2021-07-30", filed="2021-08-05"),
        balance_sheet=_share_row(100_000_000, end="2021-06-30", filed="2021-08-05"),
    )

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 100.0
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT_GUARDED"
    assert coverage["shares_guard"]["rejected"][0]["value"] == 100_000_000_000.0


def test_a_refused_count_is_unknown_with_the_reason_never_a_substitute(monkeypatch, tmp_path):
    """Packaging Corp's 2026 10-K cover (89.2 billion) against its own diluted count."""

    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_payload(cfg, json.loads((FIXTURES / "PKG_0000075677.json").read_text(encoding="utf-8")))
    monkeypatch.setenv("VOE_SHARES_ALLOW_MKTCAP_DERIVE", "true")
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    derive_calls: list[str] = []
    monkeypatch.setattr(
        "app.valuation.shares._load_market_cap_snapshot",
        lambda **kwargs: derive_calls.append("cap") or (10_000.0, ["snapshot"]),
    )

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2026-03-15", run_id=None)

    assert value is None
    assert coverage["shares_status"] == "UNKNOWN"
    assert coverage["shares_value"] == "UNKNOWN"
    assert coverage["shares_reason_code"] == "SHARES_CONTRADICTED"
    assert coverage["shares_reason_detail"] == (
        "No share-count candidate survived the guard (SHARES_CONTRADICTED). "
        "No substitute count was used."
    )
    assert coverage["shares_guard"]["outcome"] == "REFUSED"
    assert derive_calls == [], "the market-cap/price derivation is not reached"

    _cap, cap_coverage = resolve_market_cap_from_price_asof(
        ticker="RMD", as_of_date="2026-03-15", price=200.0
    )
    assert cap_coverage["market_cap_reason_code"] == "SHARES_UNKNOWN"
    assert cap_coverage["shares_reason_code"] == "SHARES_CONTRADICTED"

    # The facts row the scout and the engine read says the same, not TAG_MISS.
    from app.valuation.facts import resolve_financial_facts_asof

    facts_row = resolve_financial_facts_asof(ticker="RMD", as_of_date="2026-03-15", cfg=cfg)
    assert facts_row["shares_status"] == "UNKNOWN"
    assert facts_row["shares_reason"] == "SHARES_CONTRADICTED"
    assert facts_row["shares_guard"]["outcome"] == "REFUSED"


# --- One source only -----------------------------------------------------------


def test_single_source_single_filing_is_accepted_unchecked_and_says_so(monkeypatch, tmp_path):
    """Nothing to check against -- no second basis in the filing, no prior filing -- so the
    count stands, flagged unchecked. This is the residual: a lone slipped cover with no
    history is not detectable from the filings alone."""

    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_companyfacts_cache(
        cfg,
        cover_page=_share_row(145_461, end="2021-07-30", filed="2021-08-05"),
        balance_sheet=None,
    )

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 0.145461
    assert coverage["shares_status"] == "OK"
    assert coverage["shares_reason_code"] == "COMPANYFACTS_HIT"
    assert coverage["shares_guard"]["outcome"] == "PASS"
    assert coverage["shares_guard"]["unchecked"] is True
    assert coverage["shares_guard"]["accepted"]["note"] == (
        "No prior filing on this tag and no same-filing reference decided; accepted unchecked."
    )


def test_single_source_balance_sheet_only(monkeypatch, tmp_path):
    cfg = _init_cfg(monkeypatch, tmp_path)
    _write_companyfacts_cache(
        cfg,
        cover_page=None,
        balance_sheet=_share_row(145_461_000, end="2021-06-30", filed="2021-08-05"),
    )

    value, coverage = resolve_shares_asof(ticker="RMD", as_of_date="2021-08-31", run_id=None)

    assert value == 145.461
    assert coverage["shares_status"] == "OK"
    assert coverage["shares_guard"]["outcome"] == "PASS"
    assert coverage["shares_guard"]["unchecked"] is True


# --- Consumers of the resolver do not re-admit a refused count ---------------------


def test_engine_does_not_fall_back_to_its_rows_after_a_refusal(monkeypatch, tmp_path):
    from app.valuation import engine

    refused = {
        "shares_status": "UNKNOWN",
        "shares_reason_code": "SHARES_CONTRADICTED",
        "shares_value": "UNKNOWN",
        "shares_guard": {"outcome": "REFUSED", "reason_code": "SHARES_CONTRADICTED"},
    }
    monkeypatch.setattr(engine, "resolve_shares_asof", lambda **_kw: (None, dict(refused)))
    monkeypatch.setattr(engine, "get_default_provider", lambda *_a, **_k: None, raising=False)
    payload = {
        "ticker": "RMD",
        "as_of_date": "2021-09-15",
        "rows": [{"year": 2021, "shares_outstanding": 0.145681, "fcf": 1.0, "net_debt": 0.0}],
    }
    valuation = engine.build_ticker_valuation(payload, as_of_date="2021-09-15", with_prices=False)
    entry = valuation["shares_coverage_entry"]
    assert entry["shares_reason_code"] == "SHARES_CONTRADICTED"
    assert entry["shares_status"] == "UNKNOWN"
    assert valuation["input_snapshot"]["shares_outstanding"] == "UNKNOWN"

    # Control: an ordinary miss still falls back to the rows, as before.
    miss = {"shares_status": "UNKNOWN", "shares_reason_code": "COMPANYFACTS_MISS", "shares_guard": None}
    monkeypatch.setattr(engine, "resolve_shares_asof", lambda **_kw: (None, dict(miss)))
    valuation = engine.build_ticker_valuation(payload, as_of_date="2021-09-15", with_prices=False)
    assert valuation["shares_coverage_entry"]["shares_reason_code"] == "OK"
    assert valuation["input_snapshot"]["shares_outstanding"] == 0.145681


def test_graham_dodd_does_not_re_pick_a_refused_count():
    from app.valuation.graham_dodd import _resolve_shares

    pkg = json.loads((FIXTURES / "PKG_0000075677.json").read_text(encoding="utf-8"))
    refused_row = {
        "shares_status": "UNKNOWN",
        "shares_value": "UNKNOWN",
        "derived_from": ["cache"],
        "shares_guard": {"outcome": "REFUSED", "reason_code": "SHARES_CONTRADICTED"},
    }
    assert _resolve_shares(facts_row=refused_row, companyfacts=pkg, as_of_date="2026-03-15") == (
        "UNKNOWN",
        ["cache"],
        "SHARES_CONTRADICTED",
    )
    # With no facts row to consult, its own fallback is the same guarded chooser.
    assert _resolve_shares(facts_row={}, companyfacts=pkg, as_of_date="2026-03-15") == (
        "UNKNOWN",
        [],
        "SHARES_CONTRADICTED",
    )
    value, refs, reason = _resolve_shares(facts_row={}, companyfacts=pkg, as_of_date="2026-05-10")
    assert (value, reason) == (89_098_647.0, "OK")
    assert refs == [
        "companyfacts.dei.EntityCommonStockSharesOutstanding[end_date=2026-05-01,unit=shares,"
        "filed=2026-05-08,accn=0001193125-26-213675]"
    ]
