from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from app.autonomous import sector_candidates as candidate_module
from app.autonomous.sector_candidates import SECURITY_TYPE_NON_COMMON_EQUITY
from app.autonomous.sector_candidates import resolve_sector_candidate_tickers


def _isolate_candidate_boundaries(monkeypatch, *, preserve_security_filter=False):
    """Keep ranking tests independent of registry and structural-gate state."""

    if not preserve_security_filter:
        monkeypatch.setattr(
            candidate_module,
            "_filter_common_equity_tickers",
            lambda tickers: (tickers, [], []),
        )
    monkeypatch.setattr(
        candidate_module,
        "_apply_structural_gate",
        lambda tickers, **kwargs: (tickers, [], {}),
    )


def _stub_v1_explicit_cap_loader(monkeypatch):
    """Give explicit-selection contract tests deterministic in-band evidence."""
    from app.autonomous.cap_resolver import CapClassification

    def fake_load(*, explicit_tickers, as_of_date, **_kwargs):
        classifications = {
            ticker: CapClassification(
                ticker=ticker,
                as_of_date=as_of_date,
                market_cap_mm=1_500.0,
                cap_source="asof_companyfacts",
                cap_band="small",
            )
            for ticker in explicit_tickers
        }
        return (
            [(ticker, as_of_date, {}) for ticker in explicit_tickers],
            classifications,
        )

    monkeypatch.setattr("app.sector.scan.load_sector_tickers_classified", fake_load)


def test_v1_explicit_path_carries_full_cap_quote_lineage_without_filtering(monkeypatch, tmp_path):
    from app.autonomous.cap_resolver import CapClassification

    snapshot_id = "a" * 64
    classifications = {
        "INBD": CapClassification(
            ticker="INBD",
            as_of_date="2026-07-28",
            market_cap_mm=1_500.0,
            market_cap_unit="USD_millions",
            cap_source="stale_shares",
            cap_band="small",
            price_used=12.0,
            price_as_of_date="2026-07-28",
            price_source="fixture_quote",
            price_source_url="https://example.test/quotes/INBD",
            price_currency="USD",
            price_confidence="HIGH",
            quote_snapshot_id=snapshot_id,
            price_basis="UNADJUSTED",
            raw_price=12.0,
            shares_mm=125.0,
            raw_shares_outstanding_mm=125.0,
            raw_shares_source_value=125_000_000.0,
            raw_shares_source_unit="shares",
            shares_unit="shares_millions",
            shares_basis="UNADJUSTED",
            shares_source="SEC_COMPANYFACTS",
            shares_source_url="https://data.sec.gov/fixture/INBD.json",
            issuer_quote_ratio=1.0,
            split_adjustment_factor=1.0,
            split_lineage_proof={"status": "NO_SPLIT_REQUIRED"},
            shares_period_end="2026-06-30",
            shares_filed_date="2026-07-25",
            cap_effective_as_of_date="2026-07-28",
            cap_source_kind="SEC",
            cap_source_name="fixture_cap_chain",
            cap_source_url="https://data.sec.gov/fixture/INBD.json",
            cap_confidence="HIGH",
            cap_method="price_times_shares_divided_by_issuer_quote_ratio",
        ),
        "OUTB": CapClassification(
            ticker="OUTB",
            as_of_date="2026-07-28",
            market_cap_mm=3_500.0,
            market_cap_unit="USD_millions",
            cap_source="asof_companyfacts",
            cap_band="mid",
        ),
        "UNKN": CapClassification(
            ticker="UNKN",
            as_of_date="2026-07-28",
            market_cap_mm=None,
            cap_source="unknown",
            cap_band=None,
        ),
    }
    classify_call: dict[str, object] = {}
    gate_call: dict[str, object] = {}

    def fake_classify(**kwargs):
        classify_call.update(kwargs)
        return classifications

    def fake_gate(tickers, **kwargs):
        gate_call.update(kwargs)
        return tickers, [], {}

    monkeypatch.setattr("app.sector.scan.classify_tickers_for_market_cap", fake_classify)
    monkeypatch.setattr(
        candidate_module,
        "_filter_common_equity_tickers",
        lambda tickers: (tickers, [], []),
    )
    monkeypatch.setattr(candidate_module, "_apply_structural_gate", fake_gate)
    db_path = tmp_path / "fixture.db"
    sqlite3.connect(str(db_path)).close()

    selection = resolve_sector_candidate_tickers(
        sector="specialty_manufacturing",
        explicit_tickers=["INBD", "OUTB", "UNKN"],
        market_cap_focus="small_cap",
        db_path=db_path,
        as_of_date="2026-07-28",
        pipeline_version="v1",
        allow_live_market_data=False,
    )

    assert classify_call["tickers"] == ["INBD", "OUTB", "UNKN"]
    assert classify_call["as_of_date"] == "2026-07-28"
    assert classify_call["db_path"] == db_path
    assert classify_call["pipeline_version"] == "v1"
    assert classify_call["allow_live_market_data"] is False
    assert selection.selected_tickers == ["INBD", "OUTB", "UNKN"]
    assert selection.loaded_tickers == ["INBD", "OUTB", "UNKN"]
    assert selection.excluded_tickers == []
    assert selection.warnings == [
        "CAP_RESOLVED_OUT_OF_BAND:1:OUTB",
        "UNKNOWN_CAP_INCLUDED:1",
    ]
    assert gate_call["exclude"] is False
    assert gate_call["cap_classifications"] == selection.cap_classifications
    in_band = selection.cap_classifications["INBD"]
    assert {
        "market_cap_mm": in_band["market_cap_mm"],
        "market_cap_unit": in_band["market_cap_unit"],
        "cap_source": in_band["cap_source"],
        "cap_method": in_band["cap_method"],
        "cap_effective_as_of_date": in_band["cap_effective_as_of_date"],
        "price_used": in_band["price_used"],
        "price_as_of_date": in_band["price_as_of_date"],
        "price_source": in_band["price_source"],
        "price_source_url": in_band["price_source_url"],
        "price_currency": in_band["price_currency"],
        "price_confidence": in_band["price_confidence"],
        "quote_snapshot_id": in_band["quote_snapshot_id"],
        "price_basis": in_band["price_basis"],
        "raw_price": in_band["raw_price"],
        "shares_mm": in_band["shares_mm"],
        "raw_shares_outstanding_mm": in_band["raw_shares_outstanding_mm"],
        "raw_shares_source_value": in_band["raw_shares_source_value"],
        "raw_shares_source_unit": in_band["raw_shares_source_unit"],
        "shares_unit": in_band["shares_unit"],
        "shares_basis": in_band["shares_basis"],
        "shares_source": in_band["shares_source"],
        "shares_source_url": in_band["shares_source_url"],
        "shares_period_end": in_band["shares_period_end"],
        "shares_filed_date": in_band["shares_filed_date"],
        "issuer_quote_ratio": in_band["issuer_quote_ratio"],
        "split_adjustment_factor": in_band["split_adjustment_factor"],
        "split_lineage_proof": in_band["split_lineage_proof"],
    } == {
        "market_cap_mm": 1_500.0,
        "market_cap_unit": "USD_millions",
        "cap_source": "stale_shares",
        "cap_method": "price_times_shares_divided_by_issuer_quote_ratio",
        "cap_effective_as_of_date": "2026-07-28",
        "price_used": 12.0,
        "price_as_of_date": "2026-07-28",
        "price_source": "fixture_quote",
        "price_source_url": "https://example.test/quotes/INBD",
        "price_currency": "USD",
        "price_confidence": "HIGH",
        "quote_snapshot_id": "a" * 64,
        "price_basis": "UNADJUSTED",
        "raw_price": 12.0,
        "shares_mm": 125.0,
        "raw_shares_outstanding_mm": 125.0,
        "raw_shares_source_value": 125_000_000.0,
        "raw_shares_source_unit": "shares",
        "shares_unit": "shares_millions",
        "shares_basis": "UNADJUSTED",
        "shares_source": "SEC_COMPANYFACTS",
        "shares_source_url": "https://data.sec.gov/fixture/INBD.json",
        "shares_period_end": "2026-06-30",
        "shares_filed_date": "2026-07-25",
        "issuer_quote_ratio": 1.0,
        "split_adjustment_factor": 1.0,
        "split_lineage_proof": {"status": "NO_SPLIT_REQUIRED"},
    }


def test_v2_explicit_path_classifies_every_security_and_applies_cap_band(monkeypatch):
    from app.autonomous.cap_resolver import CapClassification

    classifications = {
        "INBD": CapClassification(
            ticker="INBD",
            as_of_date="2026-07-15",
            market_cap_mm=12_500.0,
            cap_source="terminal_exchange",
            cap_band="large_cap",
        ),
        "OUTB": CapClassification(
            ticker="OUTB",
            as_of_date="2026-07-15",
            market_cap_mm=8_000.0,
            cap_source="terminal_exchange",
            cap_band="mid",
        ),
        "UNKN": CapClassification(
            ticker="UNKN",
            as_of_date="2026-07-15",
            market_cap_mm=None,
            cap_source="unknown",
            cap_band=None,
        ),
    }
    seen: dict[str, object] = {}
    terminal_lookup = lambda ticker, as_of_date, identity: None

    def fake_classify(**kwargs):
        seen.update(kwargs)
        return classifications

    monkeypatch.setattr(
        "app.sector.scan.classify_tickers_for_market_cap",
        fake_classify,
    )
    monkeypatch.setattr(
        candidate_module,
        "_apply_structural_gate",
        lambda tickers, **kwargs: (tickers, [], {}),
    )

    selection = resolve_sector_candidate_tickers(
        sector="energy",
        explicit_tickers=["INBD", "OUTB", "UNKN"],
        market_cap_focus="large_cap",
        as_of_date="2026-07-15",
        pipeline_version="v2",
        terminal_cap_lookup=terminal_lookup,
        allow_live_market_data=False,
    )

    assert seen["tickers"] == ["INBD", "OUTB", "UNKN"]
    assert seen["as_of_date"] == "2026-07-15"
    assert seen["pipeline_version"] == "v2"
    assert seen["terminal_cap_lookup"] is terminal_lookup
    assert seen["allow_live_market_data"] is False
    assert selection.selected_tickers == ["INBD", "UNKN"]
    assert selection.membership_tickers == ["INBD", "UNKN"]
    assert selection.execution_tickers == ["INBD", "UNKN"]
    assert selection.deferred_by_bound_tickers == []
    assert selection.loaded_tickers == ["INBD", "UNKN"]
    assert selection.excluded_tickers == ["OUTB"]
    assert set(selection.cap_classifications) == {"INBD", "OUTB", "UNKN"}
    assert selection.warnings == [
        "CAP_RESOLVED_OUT_OF_BAND:1:OUTB",
        "UNKNOWN_CAP_INCLUDED:1",
        "V2_PRE_RANK_AND_GATES_DEFERRED_UNTIL_POST_REPAIR",
    ]


def test_v2_candidate_pool_classifies_before_ranking_and_preserves_exclusions(
    monkeypatch,
):
    from app.autonomous.cap_resolver import CapClassification

    classifications = {
        "INBD": CapClassification(
            ticker="INBD",
            as_of_date="2026-07-15",
            market_cap_mm=20_000.0,
            cap_source="terminal_provider",
            cap_band="large_cap",
        ),
        "OUTB": CapClassification(
            ticker="OUTB",
            as_of_date="2026-07-15",
            market_cap_mm=3_000.0,
            cap_source="asof_companyfacts",
            cap_band="mid",
        ),
        "UNKN": CapClassification(
            ticker="UNKN",
            as_of_date="2026-07-15",
            market_cap_mm=None,
            cap_source="unknown",
            cap_band=None,
        ),
    }
    rank_calls: list[list[str]] = []
    monkeypatch.setattr(
        "app.sector.scan.classify_tickers_for_market_cap",
        lambda **kwargs: classifications,
    )

    def fake_rank(*, tickers, **kwargs):
        rank_calls.append(tickers)
        return [SimpleNamespace(ticker="UNKN"), SimpleNamespace(ticker="INBD")]

    monkeypatch.setattr("app.sector.scan.pre_rank_sector", fake_rank)
    monkeypatch.setattr(candidate_module, "_entry_is_selectable", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        candidate_module,
        "_apply_structural_gate",
        lambda tickers, **kwargs: (tickers, [], {}),
    )

    selection = resolve_sector_candidate_tickers(
        sector="energy",
        candidate_pool_tickers=["INBD", "OUTB", "UNKN"],
        market_cap_focus="large_cap",
        max_candidates=1,
        as_of_date="2026-07-15",
        pipeline_version="v2",
    )

    assert rank_calls == []
    assert selection.selected_tickers == ["INBD", "UNKN"]
    assert selection.membership_tickers == ["INBD", "UNKN"]
    assert selection.execution_tickers == ["INBD"]
    assert selection.deferred_by_bound_tickers == ["UNKN"]
    assert selection.execution_bound == 1
    assert selection.membership_fingerprint == (
        "7c627b0a66ef107e2ab963c6a97baab6afc0ea0f096cdccdad5b67df880aa667"
    )
    assert selection.execution_fingerprint == (
        "1486cc9019200a5198234d008ddfd5966cbf8c651ae7863b75276f9ee48b8563"
    )
    assert selection.loaded_tickers == ["INBD", "UNKN"]
    assert selection.excluded_tickers == ["OUTB"]
    assert selection.cap_classifications["OUTB"]["cap_band"] == "mid"
    assert selection.warnings == [
        "CAP_RESOLVED_OUT_OF_BAND:1:OUTB",
        "UNKNOWN_CAP_INCLUDED:1",
        "V2_PRE_RANK_AND_GATES_DEFERRED_UNTIL_POST_REPAIR",
    ]
    assert selection.ranking_basis == "v2_discovery_input_order_pending_repaired_rank"
    assert selection.structural_gate_results == {}


def test_v2_sector_source_defers_legacy_rank_gate_and_candidate_limit(monkeypatch):
    from app.autonomous.cap_resolver import CapClassification

    seen: dict[str, object] = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        return (
            [("AAA", "2026-07-15", {}), ("BBB", "2026-07-15", {})],
            {
                ticker: CapClassification(
                    ticker=ticker,
                    as_of_date="2026-07-15",
                    market_cap_mm=15_000.0,
                    cap_source="terminal_exchange",
                    cap_band="large_cap",
                )
                for ticker in ("AAA", "BBB")
            },
        )

    monkeypatch.setattr("app.sector.scan.load_sector_tickers_classified", fake_load)
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy rank called")),
    )
    monkeypatch.setattr(
        candidate_module,
        "_apply_structural_gate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy gate called")),
    )

    selection = resolve_sector_candidate_tickers(
        sector="energy",
        market_cap_focus="large_cap",
        max_candidates=1,
        as_of_date="2026-07-15",
        pipeline_version="v2",
        filing_risk_use_llm=True,
        allow_live_market_data=False,
    )

    assert seen["pipeline_version"] == "v2"
    assert seen["as_of_date"] == "2026-07-15"
    assert seen["allow_live_market_data"] is False
    assert selection.selected_tickers == ["AAA", "BBB"]
    assert selection.membership_tickers == ["AAA", "BBB"]
    assert selection.execution_tickers == ["AAA"]
    assert selection.deferred_by_bound_tickers == ["BBB"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "v2_discovery_membership_order_pending_repaired_rank"
    assert selection.warnings == [
        "V2_PRE_RANK_AND_GATES_DEFERRED_UNTIL_POST_REPAIR",
    ]


def test_explicit_tickers_are_normalized_deduped_and_capped(monkeypatch):
    _isolate_candidate_boundaries(monkeypatch)
    _stub_v1_explicit_cap_loader(monkeypatch)
    selection = resolve_sector_candidate_tickers(
        sector="specialty_manufacturing",
        explicit_tickers=[" aaa ", "BBB", "aaa", "CCC"],
        market_cap_focus="small_cap",
        max_candidates=2,
    )

    assert selection.selected_tickers == ["AAA", "BBB"]
    assert selection.source == "explicit_tickers"
    assert selection.requested_tickers == ["AAA", "BBB", "CCC"]
    assert selection.loaded_tickers == ["AAA", "BBB", "CCC"]
    assert selection.excluded_tickers == ["CCC"]
    assert selection.ranking_basis == "input_order"
    assert selection.warnings == []


def test_security_type_filter_excludes_debt_preferred_and_depositary_series(monkeypatch):
    profiles = {
        "0000936340": {
            "issuer_primary_ticker": "DTE",
            "issuer_listed_tickers": ["DTE", "DTW", "DTB", "DTG", "DTK"],
        },
        "0000811156": {
            "issuer_primary_ticker": "CMS",
            "issuer_listed_tickers": ["CMS", "CMSA", "CMSC", "CMSD", "CMS-PC"],
        },
        "0000008063": {
            "issuer_primary_ticker": "ATRO",
            "issuer_listed_tickers": ["ATRO", "ATROB"],
        },
    }
    cik_by_ticker = {
        "DTE": "0000936340",
        "DTW": "0000936340",
        "DTB": "0000936340",
        "DTG": "0000936340",
        "CMS": "0000811156",
        "CMSA": "0000811156",
        "CMSC": "0000811156",
        "CMSD": "0000811156",
        "ATKR": "0000012345",
        "BAM": "0000012346",
        "CALM": "0000012347",
        "GPK": "0000012348",
        "ATROB": "0000008063",
    }

    monkeypatch.setattr(
        candidate_module,
        "_ticker_cik_for_security_filter",
        lambda ticker: cik_by_ticker.get(ticker),
    )
    monkeypatch.setattr(
        candidate_module,
        "_submission_profile_for_security_filter",
        lambda cik: profiles.get(str(cik), {}),
    )

    assert candidate_module._security_filter_for_ticker("DTW").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("DTB").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("DTG").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("CMSA").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("CMSC").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("CMSD").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("SOCGP").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("SOCGM").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("WELPP").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("WELPM").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("ACGLN").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("HPP-A").is_common_equity is False
    assert candidate_module._security_filter_for_ticker("CMS").is_common_equity is True
    assert candidate_module._security_filter_for_ticker("BAM").is_common_equity is True
    assert candidate_module._security_filter_for_ticker("ATKR").is_common_equity is True
    assert candidate_module._security_filter_for_ticker("GPK").is_common_equity is True
    assert candidate_module._security_filter_for_ticker("CALM").is_common_equity is True
    assert candidate_module._security_filter_for_ticker("ATROB").is_common_equity is True


def test_explicit_tickers_default_to_uncapped_full_review(monkeypatch):
    _stub_v1_explicit_cap_loader(monkeypatch)
    selection = resolve_sector_candidate_tickers(
        sector="specialty_manufacturing",
        explicit_tickers=["AAA", "BBB", "CCC"],
        market_cap_focus="small_cap",
    )

    assert selection.selected_tickers == ["AAA", "BBB", "CCC"]
    assert selection.source == "explicit_tickers"
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "input_order"
    assert set(selection.cap_classifications) == {"AAA", "BBB", "CCC"}


def test_sector_source_uses_market_cap_focus_and_consensus_pre_rank(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    def fake_load_sector_tickers(
        *,
        sector,
        db_path=None,
        cap_min=None,
        cap_max=None,
        pipeline_version="v1",
        allow_live_market_data=True,
        as_of_date=None,
    ):
        calls["load"] = {
            "sector": sector,
            "db_path": db_path,
            "cap_min": cap_min,
            "cap_max": cap_max,
            "pipeline_version": pipeline_version,
            "allow_live_market_data": allow_live_market_data,
            "as_of_date": as_of_date,
        }
        return [
            ("AAA", "2026-04-26", {}),
            ("BBB", "2026-04-26", {}),
            ("CCC", "2026-04-26", {}),
        ], {}

    def fake_pre_rank_sector(
        *,
        tickers,
        limit,
        as_of_date,
        db_path,
        include_blocked=False,
    ):
        calls["rank"] = {
            "tickers": tickers,
            "limit": limit,
            "as_of_date": as_of_date,
            "db_path": db_path,
            "include_blocked": include_blocked,
        }
        return [SimpleNamespace(ticker="BBB"), SimpleNamespace(ticker="AAA")]

    monkeypatch.setattr("app.sector.scan.load_sector_tickers_classified", fake_load_sector_tickers)
    monkeypatch.setattr("app.sector.scan.pre_rank_sector", fake_pre_rank_sector)

    # Fail-closed: the structural gate excludes every name when the engine
    # DB is missing, so the test DB must exist (empty = minimal = benign).
    test_db = tmp_path / "test.db"
    sqlite3.connect(str(test_db)).close()

    selection = resolve_sector_candidate_tickers(
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        max_candidates=2,
        db_path=str(test_db),
        as_of_date="2026-04-26",
    )

    assert selection.selected_tickers == ["BBB", "AAA"]
    assert selection.source == "sector_scan_db"
    assert selection.loaded_tickers == ["AAA", "BBB", "CCC"]
    assert selection.excluded_tickers == ["CCC"]
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert selection.warnings == []
    assert calls["load"] == {
        "sector": "specialty_manufacturing",
        "db_path": str(test_db),
        "cap_min": 500.0,
        "cap_max": 2500.0,
        "pipeline_version": "v1",
        "allow_live_market_data": False,
        "as_of_date": "2026-04-26",
    }
    assert calls["rank"] == {
        "tickers": ["AAA", "BBB", "CCC"],
        "limit": 0,
        "as_of_date": "2026-04-26",
        "db_path": str(test_db),
        "include_blocked": False,
    }


def test_sector_db_source_persists_structural_gate_evidence(monkeypatch):
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: (
            [
                ("AAA", "2026-07-16", {}),
                ("BBB", "2026-07-16", {}),
            ],
            {},
        ),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda **kwargs: [SimpleNamespace(ticker="AAA"), SimpleNamespace(ticker="BBB")],
    )
    monkeypatch.setattr(candidate_module, "_entry_is_selectable", lambda *args, **kwargs: True)
    gate_evidence = {
        "BBB": {
            "ticker": "BBB",
            "as_of_date": "2026-07-16",
            "quarantined": True,
            "excluded_error": False,
            "triggered_codes": ["DELISTING_NOTICE"],
            "reasons": ["QUARANTINE_STRUCTURAL:DELISTING_NOTICE"],
        }
    }
    monkeypatch.setattr(
        candidate_module,
        "_apply_structural_gate",
        lambda tickers, **kwargs: (["AAA"], ["BBB"], gate_evidence),
    )

    selection = resolve_sector_candidate_tickers(
        sector="energy",
        market_cap_focus="large_and_mega",
    )

    assert selection.source == "sector_scan_db"
    assert selection.selected_tickers == ["AAA"]
    assert selection.excluded_tickers == ["BBB"]
    assert selection.structural_gate_results == gate_evidence


def test_sector_source_filters_non_common_equity_before_pre_rank(monkeypatch):
    _isolate_candidate_boundaries(monkeypatch, preserve_security_filter=True)
    calls: dict[str, object] = {}

    def fake_security_filter(ticker):
        if ticker in {"DTW", "DTB", "DTG"}:
            return candidate_module.SecurityTypeFilterResult(
                ticker=ticker,
                is_common_equity=False,
                reason=SECURITY_TYPE_NON_COMMON_EQUITY,
                detail="known_non_common_equity_series",
            )
        return candidate_module.SecurityTypeFilterResult(ticker=ticker, is_common_equity=True)

    monkeypatch.setattr(candidate_module, "_security_filter_for_ticker", fake_security_filter)
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: (
            [
                ("DTE", "2026-04-26", {}),
                ("DTW", "2026-04-26", {}),
                ("DTB", "2026-04-26", {}),
                ("DTG", "2026-04-26", {}),
                ("GPK", "2026-04-26", {}),
            ],
            {},
        ),
    )

    def fake_pre_rank_sector(
        *,
        tickers,
        limit,
        as_of_date,
        db_path,
        include_blocked=False,
    ):
        calls["rank"] = {
            "tickers": tickers,
            "limit": limit,
            "as_of_date": as_of_date,
            "db_path": db_path,
            "include_blocked": include_blocked,
        }
        return [SimpleNamespace(ticker="GPK"), SimpleNamespace(ticker="DTE")]

    monkeypatch.setattr("app.sector.scan.pre_rank_sector", fake_pre_rank_sector)

    selection = resolve_sector_candidate_tickers(
        sector="utilities",
        market_cap_focus="mid_cap",
        max_candidates=8,
        as_of_date="2026-04-26",
    )

    assert selection.selected_tickers == ["GPK", "DTE"]
    assert selection.loaded_tickers == ["DTE", "GPK"]
    assert selection.excluded_tickers == ["DTW", "DTB", "DTG"]
    assert selection.warnings == [
        "SECURITY_TYPE_NON_COMMON_EQUITY:3:DTW:known_non_common_equity_series,DTB:known_non_common_equity_series,DTG:known_non_common_equity_series",
        "SELECTABLE_CANDIDATES_BELOW_LIMIT:2/8",
    ]
    assert calls["rank"] == {
        "tickers": ["DTE", "GPK"],
        "limit": 0,
        "as_of_date": "2026-04-26",
        "db_path": None,
        "include_blocked": False,
    }


def test_sector_source_default_reviews_all_loaded_ranked_candidates(monkeypatch):
    _isolate_candidate_boundaries(monkeypatch)
    packet_unusable = SimpleNamespace(
        model_status="OK",
        model_blockers=[],
        gate_verdict="BLOCK",
        current_price=10.0,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )
    packet_selectable = SimpleNamespace(
        model_status="OK",
        model_blockers=[],
        gate_verdict="PROCEED",
        current_price=10.0,
        dcf_value=20.0,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: (
            [("AAA", "2026-04-26", {}), ("BBB", "2026-04-26", {}), ("CCC", "2026-04-26", {})],
            {},
        ),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="BBB", methods_with_value=1, packet=packet_selectable),
            SimpleNamespace(ticker="AAA", methods_with_value=0, packet=packet_unusable),
            SimpleNamespace(ticker="CCC", methods_with_value=1, packet=packet_selectable),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
    )

    assert selection.selected_tickers == ["BBB", "AAA", "CCC"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_all_loaded"
    assert selection.warnings == ["UNUSABLE_CANDIDATES_INCLUDED_FOR_FULL_REVIEW:1"]


def test_sector_source_default_appends_loaded_tickers_omitted_by_pre_rank(monkeypatch):
    _isolate_candidate_boundaries(monkeypatch)
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: (
            [("AAA", "2026-04-26", {}), ("BBB", "2026-04-26", {}), ("CCC", "2026-04-26", {})],
            {},
        ),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="BBB", methods_with_value=1, packet=None),
            SimpleNamespace(ticker="AAA", methods_with_value=1, packet=None),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
    )

    assert selection.selected_tickers == ["BBB", "AAA", "CCC"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_all_loaded"
    assert selection.warnings == ["PRE_RANK_OMITTED_LOADED_CANDIDATES:1"]


def test_max_candidates_zero_is_rejected():
    try:
        resolve_sector_candidate_tickers(
            sector="specialty_manufacturing",
            explicit_tickers=["AAA"],
            market_cap_focus="small_cap",
            max_candidates=0,
        )
    except ValueError as exc:
        assert str(exc) == "max_candidates must be a positive integer when provided"
    else:
        raise AssertionError("expected max_candidates=0 to be rejected")


def test_sector_source_filters_unusable_ranked_candidates_before_capping(monkeypatch):
    _isolate_candidate_boundaries(monkeypatch)
    packet_blocked = SimpleNamespace(
        model_status="MODEL_BLOCKED",
        model_blockers=["SECURITY_IDENTITY_UNVERIFIED"],
        gate_verdict="PROCEED",
        current_price=20.0,
        dcf_value=30.0,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )
    packet_selectable = SimpleNamespace(
        model_status="OK",
        model_blockers=[],
        gate_verdict="PROCEED",
        current_price=20.0,
        dcf_value=30.0,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: (
            [
                ("BAD", "2026-04-26", {}),
                ("AAA", "2026-04-26", {}),
                ("BBB", "2026-04-26", {}),
                ("BLK", "2026-04-26", {}),
            ],
            {},
        ),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="BAD", methods_with_value=0, packet=None),
            SimpleNamespace(ticker="AAA", methods_with_value=1, packet=packet_selectable),
            SimpleNamespace(ticker="BBB", methods_with_value=1, packet=packet_selectable),
            SimpleNamespace(ticker="BLK", methods_with_value=1, packet=packet_blocked),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="insurance",
        market_cap_focus="small_cap",
        max_candidates=3,
    )

    assert selection.selected_tickers == ["AAA", "BBB"]
    assert selection.excluded_tickers == ["BAD", "BLK"]
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert selection.warnings == [
        "FILTERED_UNUSABLE_CANDIDATES:2",
        "SELECTABLE_CANDIDATES_BELOW_LIMIT:2/3",
    ]


def _write_book_value_rows(db_path, ticker: str, years: list[int]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS companyfacts_facts (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                period_type TEXT NOT NULL DEFAULT 'FY',
                period_end TEXT NOT NULL,
                filed_date TEXT,
                line_item TEXT NOT NULL,
                value REAL,
                units TEXT,
                source_url TEXT,
                fetched_at TEXT NOT NULL,
                form TEXT,
                accession TEXT
            )
            """
        )
        for year in years:
            conn.execute(
                """
                INSERT INTO companyfacts_facts
                    (ticker, fiscal_year, period_type, period_end, filed_date,
                     line_item, value, units, source_url, fetched_at, form,
                     accession)
                VALUES (
                    ?, ?, 'FY', ?, '2026-05-11', 'equity', 1200.0,
                    'USD_millions',
                    'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                    '2026-05-11T00:00:00Z', '10-K', ?
                )
                """,
                (ticker, year, f"{year}-12-31", f"0000000001-{str(year)[2:]}-000001"),
            )
            conn.execute(
                """
                INSERT INTO companyfacts_facts
                    (ticker, fiscal_year, period_type, period_end, filed_date,
                     line_item, value, units, source_url, fetched_at, form,
                     accession)
                VALUES (
                    ?, ?, 'FY', ?, '2026-05-11', 'shares_outstanding',
                    100.0, 'shares_millions',
                    'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                    '2026-05-11T00:00:00Z', '10-K', ?
                )
                """,
                (ticker, year, f"{year}-12-31", f"0000000001-{str(year)[2:]}-000001"),
            )


def _write_capital_markets_operating_rows(db_path, ticker: str, years: list[int]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS companyfacts_facts (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                period_type TEXT NOT NULL DEFAULT 'FY',
                period_end TEXT NOT NULL,
                filed_date TEXT,
                line_item TEXT NOT NULL,
                value REAL,
                units TEXT,
                source_url TEXT,
                fetched_at TEXT NOT NULL,
                form TEXT,
                accession TEXT
            )
            """
        )
        for year in years:
            for line_item, value in [
                ("revenue", 900.0),
                ("net_income", 120.0),
                ("total_assets", 1500.0),
            ]:
                conn.execute(
                    """
                    INSERT INTO companyfacts_facts
                        (ticker, fiscal_year, period_type, period_end, filed_date,
                         line_item, value, units, source_url, fetched_at, form,
                         accession)
                    VALUES (
                        ?, ?, 'FY', ?, '2026-05-16', ?, ?, 'USD_millions',
                        'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                        '2026-05-16T00:00:00Z', '10-K', ?
                    )
                    """,
                    (
                        ticker,
                        year,
                        f"{year}-12-31",
                        line_item,
                        value,
                        f"0000000001-{str(year)[2:]}-000001",
                    ),
                )


def test_financial_services_shape_excludes_post_asof_and_undated_fy_rows(
    tmp_path,
):
    db_path = tmp_path / "engine.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE companyfacts_facts(
                ticker TEXT,
                fiscal_year INTEGER,
                period_type TEXT,
                period_end TEXT,
                filed_date TEXT,
                line_item TEXT,
                value REAL,
                units TEXT,
                source_url TEXT,
                form TEXT,
                accession TEXT
            )
            """
        )
        for fiscal_year, period_end, filed_date in (
            (2023, "2023-12-31", "2024-02-01"),
            (2024, "2024-12-31", "2026-03-01"),
            (2025, "2025-12-31", None),
        ):
            for line_item, value in (
                ("equity", 1200.0),
                ("shares_outstanding", 100.0),
            ):
                conn.execute(
                    """
                    INSERT INTO companyfacts_facts(
                        ticker, fiscal_year, period_type, period_end, filed_date,
                        line_item, value, units, source_url, form, accession
                    )
                    VALUES(
                        'BANK', ?, 'FY', ?, ?, ?, ?, ?,
                        'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json',
                        '10-K', ?
                    )
                    """,
                    (
                        fiscal_year,
                        period_end,
                        filed_date,
                        line_item,
                        value,
                        (
                            "shares_millions"
                            if line_item == "shares_outstanding"
                            else "USD_millions"
                        ),
                        f"0000000001-{str(fiscal_year)[2:]}-000001",
                    ),
                )

    assert (
        candidate_module._cached_book_value_years(
            "BANK",
            as_of_date="2026-02-13",
            db_path=db_path,
        )
        == 1
    )


def test_financial_services_book_value_shape_keeps_reasonable_candidate(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    _write_book_value_rows(db_path, "BANK", [2025, 2024, 2023])
    packet_financial = SimpleNamespace(
        ticker="BANK",
        model_status="OK",
        model_blockers=[],
        gate_verdict="BLOCK",
        current_price=42.0,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )
    packet_incomplete = SimpleNamespace(
        ticker="THIN",
        model_status="OK",
        model_blockers=[],
        gate_verdict="BLOCK",
        current_price=18.0,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("BANK", "2026-05-11", {}), ("THIN", "2026-05-11", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="BANK", methods_with_value=0, packet=packet_financial),
            SimpleNamespace(ticker="THIN", methods_with_value=0, packet=packet_incomplete),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="large_cap_financials",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
        as_of_date="2026-05-11",
    )

    assert selection.selected_tickers == ["BANK"]
    assert selection.excluded_tickers == ["THIN"]
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert selection.warnings == [
        "FILTERED_UNUSABLE_CANDIDATES:1",
        "SELECTABLE_CANDIDATES_BELOW_LIMIT:1/8",
    ]


def test_insurance_book_value_shape_does_not_require_current_price(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    _write_book_value_rows(db_path, "INSR", [2025, 2024, 2023])
    packet_financial = SimpleNamespace(
        ticker="INSR",
        model_status="OK",
        model_blockers=[],
        gate_verdict="BLOCK",
        current_price=None,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )
    packet_incomplete = SimpleNamespace(
        ticker="THIN",
        model_status="OK",
        model_blockers=[],
        gate_verdict="BLOCK",
        current_price=None,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("INSR", "2026-05-16", {}), ("THIN", "2026-05-16", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="INSR", methods_with_value=0, packet=packet_financial),
            SimpleNamespace(ticker="THIN", methods_with_value=0, packet=packet_incomplete),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="insurance",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
    )

    assert selection.selected_tickers == ["INSR"]
    assert selection.excluded_tickers == ["THIN"]
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert selection.warnings == [
        "FILTERED_UNUSABLE_CANDIDATES:1",
        "SELECTABLE_CANDIDATES_BELOW_LIMIT:1/8",
    ]


def test_capital_markets_book_value_shape_does_not_require_current_price(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    _write_book_value_rows(db_path, "CAPM", [2025, 2024, 2023])
    packet_financial = SimpleNamespace(
        ticker="CAPM",
        model_status="OK",
        model_blockers=[],
        gate_verdict="BLOCK",
        current_price=None,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("CAPM", "2026-05-16", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="CAPM", methods_with_value=0, packet=packet_financial),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="capital_markets",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
    )

    assert selection.selected_tickers == ["CAPM"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert selection.warnings == ["SELECTABLE_CANDIDATES_BELOW_LIMIT:1/8"]


def test_capital_markets_operating_asset_shape_keeps_candidate_without_book_value(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "engine.db"
    _write_capital_markets_operating_rows(db_path, "CAPM", [2025, 2024, 2023])
    packet_financial = SimpleNamespace(
        ticker="CAPM",
        model_status="OK",
        model_blockers=[],
        gate_verdict="PROCEED",
        current_price=55.0,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("CAPM", "2026-05-16", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="CAPM", methods_with_value=0, packet=packet_financial),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="capital_markets",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
    )

    assert selection.selected_tickers == ["CAPM"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert selection.warnings == ["SELECTABLE_CANDIDATES_BELOW_LIMIT:1/8"]


def test_insurance_book_value_shape_allows_routing_only_model_blockers(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    _write_book_value_rows(db_path, "AGO", [2025, 2024, 2023])
    packet_financial = SimpleNamespace(
        ticker="AGO",
        model_status="MODEL_BLOCKED",
        model_blockers=[
            "SECURITY_TYPE_UNKNOWN",
            "INSURANCE_SUBTYPE_UNCLEAR",
            "NOT_INSURANCE_VALUATION_TARGET",
        ],
        gate_verdict="PROCEED",
        current_price=82.0,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("AGO", "2026-05-16", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="AGO", methods_with_value=0, packet=packet_financial),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="insurance",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
    )

    assert selection.selected_tickers == ["AGO"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert selection.warnings == ["SELECTABLE_CANDIDATES_BELOW_LIMIT:1/8"]


def test_financial_services_book_value_shape_rejects_binding_model_blockers(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    _write_book_value_rows(db_path, "PREF", [2025, 2024, 2023])
    packet_financial = SimpleNamespace(
        ticker="PREF",
        model_status="MODEL_BLOCKED",
        model_blockers=["NOT_INSURANCE_COMMON_EQUITY"],
        gate_verdict="PROCEED",
        current_price=25.0,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("PREF", "2026-05-16", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="PREF", methods_with_value=0, packet=packet_financial),
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="insurance",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
    )

    assert selection.selected_tickers == ["PREF"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_unusable_fallback"
    assert selection.warnings == [
        "FILTERED_UNUSABLE_CANDIDATES:1",
        "NO_SELECTABLE_CANDIDATES_FOUND",
    ]


def test_financial_services_pre_rank_includes_blocked_for_book_value_gate(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    _write_book_value_rows(db_path, "BANK", [2025, 2024, 2023])
    calls: dict[str, object] = {}
    packet_financial = SimpleNamespace(
        ticker="BANK",
        model_status="OK",
        model_blockers=[],
        gate_verdict="BLOCK",
        current_price=42.0,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("BANK", "2026-05-11", {}), ("ADR", "2026-05-11", {})], {}),
    )

    def fake_pre_rank_sector(
        *,
        tickers,
        limit,
        as_of_date,
        db_path,
        include_blocked=False,
    ):
        calls["rank"] = {
            "tickers": tickers,
            "limit": limit,
            "as_of_date": as_of_date,
            "db_path": db_path,
            "include_blocked": include_blocked,
        }
        return [
            SimpleNamespace(ticker="ADR", methods_with_value=0, packet=None),
            SimpleNamespace(ticker="BANK", methods_with_value=0, packet=packet_financial),
        ]

    monkeypatch.setattr("app.sector.scan.pre_rank_sector", fake_pre_rank_sector)

    selection = resolve_sector_candidate_tickers(
        sector="large_cap_financials",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
        as_of_date="2026-05-11",
    )

    assert selection.selected_tickers == ["BANK"]
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert calls["rank"] == {
        "tickers": ["BANK", "ADR"],
        "limit": 0,
        "as_of_date": "2026-05-11",
        "db_path": db_path,
        "include_blocked": True,
    }


def test_industrial_candidate_with_generic_valuation_still_passes(monkeypatch):
    _isolate_candidate_boundaries(monkeypatch)
    packet_industrial = SimpleNamespace(
        model_status="OK",
        model_blockers=[],
        gate_verdict="PROCEED",
        current_price=20.0,
        dcf_value=35.0,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("IND", "2026-05-11", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="IND", methods_with_value=1, packet=packet_industrial)
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        max_candidates=8,
    )

    assert selection.selected_tickers == ["IND"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_selectable"
    assert selection.warnings == ["SELECTABLE_CANDIDATES_BELOW_LIMIT:1/8"]


def test_industrial_book_value_shape_without_generic_valuation_still_fails(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    _write_book_value_rows(db_path, "IND", [2025, 2024, 2023])
    packet_industrial = SimpleNamespace(
        ticker="IND",
        model_status="OK",
        model_blockers=[],
        gate_verdict="BLOCK",
        current_price=None,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("IND", "2026-05-16", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="IND", methods_with_value=0, packet=packet_industrial)
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
    )

    assert selection.selected_tickers == ["IND"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_unusable_fallback"
    assert selection.warnings == [
        "FILTERED_UNUSABLE_CANDIDATES:1",
        "NO_SELECTABLE_CANDIDATES_FOUND",
    ]


def test_industrial_capital_markets_operating_shape_still_fails(monkeypatch, tmp_path):
    db_path = tmp_path / "engine.db"
    _write_capital_markets_operating_rows(db_path, "IND", [2025, 2024, 2023])
    packet_industrial = SimpleNamespace(
        ticker="IND",
        model_status="OK",
        model_blockers=[],
        gate_verdict="PROCEED",
        current_price=55.0,
        dcf_value=None,
        epv_value=None,
        graham_value=None,
        ncav_value=None,
        insurance_value=None,
    )

    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("IND", "2026-05-16", {})], {}),
    )
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="IND", methods_with_value=0, packet=packet_industrial)
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="industrial_tech",
        market_cap_focus="mid_cap",
        max_candidates=8,
        db_path=db_path,
    )

    assert selection.selected_tickers == ["IND"]
    assert selection.excluded_tickers == []
    assert selection.ranking_basis == "consensus_pre_rank_unusable_fallback"
    assert selection.warnings == [
        "FILTERED_UNUSABLE_CANDIDATES:1",
        "NO_SELECTABLE_CANDIDATES_FOUND",
    ]


def test_sector_source_falls_back_to_loaded_order_when_pre_rank_fails(monkeypatch):
    _isolate_candidate_boundaries(monkeypatch)
    monkeypatch.setattr(
        "app.sector.scan.load_sector_tickers_classified",
        lambda **kwargs: ([("AAA", "2026-04-26", {}), ("BBB", "2026-04-26", {})], {}),
    )

    def failing_pre_rank_sector(
        *,
        tickers,
        limit,
        as_of_date,
        db_path,
        include_blocked=False,
    ):
        raise RuntimeError("ranker unavailable")

    monkeypatch.setattr("app.sector.scan.pre_rank_sector", failing_pre_rank_sector)

    selection = resolve_sector_candidate_tickers(
        sector="specialty_manufacturing",
        market_cap_focus="small_cap",
        max_candidates=1,
    )

    assert selection.selected_tickers == ["AAA"]
    assert selection.excluded_tickers == ["BBB"]
    assert selection.ranking_basis == "loaded_order"
    assert selection.warnings == ["PRE_RANK_FAILED:ranker unavailable"]


def test_unknown_market_cap_focus_warns_and_uses_uncapped_sector_load(monkeypatch):
    _isolate_candidate_boundaries(monkeypatch)
    captured: dict[str, object] = {}

    def fake_load_sector_tickers(
        *,
        sector,
        db_path=None,
        cap_min=None,
        cap_max=None,
        pipeline_version="v1",
        allow_live_market_data=True,
        as_of_date=None,
    ):
        captured["cap_min"] = cap_min
        captured["cap_max"] = cap_max
        captured["pipeline_version"] = pipeline_version
        captured["allow_live_market_data"] = allow_live_market_data
        captured["as_of_date"] = as_of_date
        return [("AAA", "2026-04-26", {})], {}

    monkeypatch.setattr("app.sector.scan.load_sector_tickers_classified", fake_load_sector_tickers)
    monkeypatch.setattr(
        "app.sector.scan.pre_rank_sector",
        lambda *, tickers, limit, as_of_date, db_path, include_blocked=False: [
            SimpleNamespace(ticker="AAA")
        ],
    )

    selection = resolve_sector_candidate_tickers(
        sector="specialty_manufacturing",
        market_cap_focus="nano_cap",
        max_candidates=8,
        as_of_date="2026-04-26",
    )

    assert selection.selected_tickers == ["AAA"]
    assert selection.warnings == [
        "UNKNOWN_MARKET_CAP_FOCUS:nano_cap",
        "SELECTABLE_CANDIDATES_BELOW_LIMIT:1/8",
    ]
    assert captured == {
        "cap_min": None,
        "cap_max": None,
        "pipeline_version": "v1",
        "allow_live_market_data": False,
        "as_of_date": "2026-04-26",
    }


def test_sector_source_returns_empty_selection_when_no_tickers_found(monkeypatch):
    monkeypatch.setattr("app.sector.scan.load_sector_tickers_classified", lambda **kwargs: ([], {}))

    selection = resolve_sector_candidate_tickers(
        sector="missing_sector",
        market_cap_focus="small_cap",
        max_candidates=8,
    )

    assert selection.selected_tickers == []
    assert selection.source == "sector_scan_db"
    assert selection.loaded_tickers == []
    assert selection.ranking_basis == "none"
    assert selection.warnings == ["NO_SECTOR_TICKERS_FOUND"]


def test_market_cap_focus_tiers_2026_06_redefinition():
    """Band redefinition (2026-06-11): exact literals, one canonical place."""
    from app.autonomous.sector_candidates import MARKET_CAP_FOCUS_TIERS

    assert MARKET_CAP_FOCUS_TIERS["micro_cap"] == (0.0, 500.0)
    assert MARKET_CAP_FOCUS_TIERS["micro"] == (0.0, 500.0)
    assert MARKET_CAP_FOCUS_TIERS["small_cap"] == (500.0, 2_500.0)
    assert MARKET_CAP_FOCUS_TIERS["small"] == (500.0, 2_500.0)
    assert MARKET_CAP_FOCUS_TIERS["mid_cap"] == (2_500.0, 10_000.0)
    assert MARKET_CAP_FOCUS_TIERS["mid"] == (2_500.0, 10_000.0)
    assert MARKET_CAP_FOCUS_TIERS["smid_cap"] == (500.0, 10_000.0)
    assert MARKET_CAP_FOCUS_TIERS["large_cap"] == (10_000.0, 200_000.0)
    assert MARKET_CAP_FOCUS_TIERS["mega_cap"] == (200_000.0, None)
    assert MARKET_CAP_FOCUS_TIERS["large_and_mega"] == (10_000.0, None)
    assert MARKET_CAP_FOCUS_TIERS["all"] == (None, None)
    assert MARKET_CAP_FOCUS_TIERS["any"] == (None, None)


def test_market_cap_focus_tiers_contiguous_non_overlapping():
    """Canonical atomic bands tile the positive cap range without gaps."""
    from app.autonomous.sector_candidates import MARKET_CAP_FOCUS_TIERS

    micro = MARKET_CAP_FOCUS_TIERS["micro_cap"]
    small = MARKET_CAP_FOCUS_TIERS["small_cap"]
    mid = MARKET_CAP_FOCUS_TIERS["mid_cap"]
    smid = MARKET_CAP_FOCUS_TIERS["smid_cap"]
    large = MARKET_CAP_FOCUS_TIERS["large_cap"]
    mega = MARKET_CAP_FOCUS_TIERS["mega_cap"]
    large_and_mega = MARKET_CAP_FOCUS_TIERS["large_and_mega"]
    assert micro[0] == 0.0
    assert micro[1] == small[0]
    assert small[1] == mid[0]
    assert mid[1] == large[0]
    assert large[1] == mega[0]
    assert mega[1] is None
    assert smid == (small[0], mid[1])
    assert large_and_mega == (large[0], None)


@pytest.mark.parametrize(
    ("market_cap_focus", "expected"),
    [
        ("small cap", (500.0, 2_500.0, None)),
        ("large", (10_000.0, 200_000.0, None)),
        ("mega-cap", (200_000.0, None, None)),
        ("large and mega", (10_000.0, None, None)),
        ("smid", (500.0, 10_000.0, None)),
    ],
)
def test_cap_bounds_use_canonical_market_cap_focus_aliases(market_cap_focus, expected):
    from app.autonomous.sector_candidates import _cap_bounds

    assert _cap_bounds(market_cap_focus) == expected
