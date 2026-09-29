"""``ivi valuation-run`` with the LLM disabled (the default) must finish.

The dossier stage ends with an LLM summary (Layer 3). The synthesis agent
refuses evidence that lacks point-in-time provenance before it even looks at
the provider, and on a free-data install that is the usual case. That refusal
used to abort the whole run with ``PROMPT_FINANCIAL_PROVENANCE_MISSING`` even
though nothing was going to a provider. Now, with the provider disabled, the
step is skipped and reported and the deterministic valuation completes. With a
live provider the refusal still stops the run, and the gate itself is unchanged.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from typer.testing import CliRunner

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.config import get_config
from app.db import get_db, init_db, utc_now_iso
from app.dossier.runner import _run_layer3_synthesis
from app.llm.synthesis_agent import run_synthesis_for_ticker

AS_OF = "2026-09-25"
CODE = "PROMPT_FINANCIAL_PROVENANCE_MISSING"


def _init(monkeypatch, tmp_path, *, provider: str = "disabled"):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(data_dir / "engine.db"))
    monkeypatch.setenv("VOE_LLM_PROVIDER", provider)
    if provider == "openai":
        # A configured provider; the gate refuses before any call is made.
        monkeypatch.setenv("VOE_OPENAI_API_KEY", "sk-test-not-used")
    monkeypatch.setattr(
        "app.universe.ticker_cik_map.refresh_ticker_cik_cache", lambda http=None: {}
    )
    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _seed_unprovenanced_packet(cfg, ticker: str = "KO") -> None:
    """An evidence packet with numeric fundamentals and no provenance record for them.

    This is the shape the packet builder produces from free-data companyfacts.
    """
    path = cfg.outputs_dir / "evidence" / f"{ticker}_{AS_OF}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ticker": ticker, "fundamentals": {"revenue": 47000.0}, "filings_used": []}),
        encoding="utf-8",
    )
    with get_db() as conn:
        conn.execute(
            "INSERT INTO evidence_packets(ticker, as_of_date, packet_path, packet_hash, created_at) "
            "VALUES (?, ?, ?, 'fixture', ?)",
            (ticker, AS_OF, str(path), utc_now_iso()),
        )


def test_the_synthesis_gate_still_refuses_unprovenanced_evidence(monkeypatch, tmp_path):
    # The precondition of the bug, and proof the gate is untouched: even with the
    # provider disabled the agent itself still raises.
    cfg = _init(monkeypatch, tmp_path)
    _seed_unprovenanced_packet(cfg)

    with pytest.raises(InvalidFinancialInputError) as excinfo:
        run_synthesis_for_ticker("KO", as_of_date=AS_OF, run_id="t1")

    assert CODE in {v.code for v in excinfo.value.result.violations}


def test_layer3_is_skipped_and_reported_when_the_llm_is_disabled(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    _seed_unprovenanced_packet(cfg)
    dossier_md = tmp_path / "dossier.md"
    dossier_md.write_text("# KO\n", encoding="utf-8")

    skipped = _run_layer3_synthesis(
        ticker="KO", as_of_date=AS_OF, run_id="t1", dossier_md=str(dossier_md)
    )

    assert skipped is not None
    assert skipped["status"] == "SKIPPED"
    assert skipped["reason"] == "LLM_DISABLED_PROMPT_EVIDENCE_REFUSED"
    assert CODE in skipped["violation_codes"]
    assert CODE in skipped["detail"]
    assert dossier_md.read_text(encoding="utf-8") == "# KO\n"  # no placeholder section appended


def test_layer3_other_integrity_failures_still_stop_the_run_even_when_disabled(
    monkeypatch, tmp_path
):
    # Only the prompt-evidence refusal is skippable. A failure of the canonical
    # financial scope (quote, cap, share lineage) is not about any prompt.
    from tests.test_provider_financial_gate_followup import _raise_invalid_financial_scope

    _init(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "app.dossier.runner.run_synthesis_for_ticker",
        lambda *a, **k: _raise_invalid_financial_scope("synthesis_agent:KO:t1"),
    )

    with pytest.raises(InvalidFinancialInputError) as excinfo:
        _run_layer3_synthesis(
            ticker="KO", as_of_date=AS_OF, run_id="t1", dossier_md=str(tmp_path / "dossier.md")
        )

    assert not any(v.code.startswith("PROMPT_") for v in excinfo.value.result.violations)


def test_layer3_still_stops_the_run_when_a_live_provider_is_configured(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path, provider="openai")
    _seed_unprovenanced_packet(cfg)

    with pytest.raises(InvalidFinancialInputError) as excinfo:
        _run_layer3_synthesis(
            ticker="KO", as_of_date=AS_OF, run_id="t1", dossier_md=str(tmp_path / "dossier.md")
        )

    assert CODE in {v.code for v in excinfo.value.result.violations}


# --- the whole command ------------------------------------------------------------------


class _Filing:
    accession = "0000021344-26-000001"
    form_type = "10-K"
    filing_date = date(2026, 2, 13)
    period_end = "2025-12-31"
    primary_doc_url = "https://www.sec.gov/example"
    filing_id = 1
    ticker = "KO"


class _Stage1:
    ticker = "KO"
    cik = "0000021344"
    filing = _Filing()
    local_path = "/tmp/example"


def _stub_dossier_inputs(monkeypatch, cfg) -> None:
    """Everything before Layer 3 is stubbed; the packet, the gate and the writers are real."""
    dossier_dir = cfg.dossiers_dir / "t1" / "KO"
    dossier_dir.mkdir(parents=True, exist_ok=True)
    dossier_md = dossier_dir / "dossier.md"
    dossier_md.write_text("# KO dossier\n", encoding="utf-8")

    monkeypatch.setattr(
        "app.dossier.runner._collect_stage1_for_ticker", lambda **kwargs: ([_Stage1()], {})
    )
    monkeypatch.setattr(
        "app.dossier.runner.materialize_and_parse_docket_stage1", lambda **kwargs: [_Filing()]
    )
    monkeypatch.setattr("app.dossier.runner.read_filing_text", lambda filing: "business risk")
    monkeypatch.setattr("app.dossier.runner.segment_10k_sections", lambda text: [])
    monkeypatch.setattr(
        "app.dossier.runner.extract_annual_items",
        lambda **kwargs: [{"ticker": "KO", "year": 2025, "metric": "revenue", "value": 47000.0}],
    )
    monkeypatch.setattr("app.dossier.runner.build_time_series", lambda items: {"rows": []})
    monkeypatch.setattr("app.dossier.runner.ensure_all_facts", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.dossier.runner.write_ticker_dossier",
        lambda **kwargs: {
            "artifacts": {"dossier_md_path": str(dossier_md), "dossier_json_path": None},
            "dossier_quality": "PARTIAL",
            "filing_body_cached": False,
        },
    )
    monkeypatch.setattr("app.dossier.runner.ensure_valuation", lambda *a, **k: [])
    monkeypatch.setattr("app.dossier.runner.append_valuation_section", lambda *a, **k: None)
    monkeypatch.setattr("app.diff.engine.build_filing_diff_for_ticker", lambda **kwargs: None)
    monkeypatch.setattr(
        "app.dossier.runner.build_packet_for_ticker",
        lambda ticker, as_of_date, dossier_run_id=None: _seed_unprovenanced_packet(cfg, ticker),
    )


def _valuation_run():
    import app.cli

    return CliRunner().invoke(
        app.cli.app,
        ["valuation-run", "--tickers", "KO", "--as-of", AS_OF, "--run-id", "t1"],
    )


def test_valuation_run_completes_with_the_llm_disabled_and_says_it_skipped_the_summary(
    monkeypatch, tmp_path
):
    cfg = _init(monkeypatch, tmp_path)
    _stub_dossier_inputs(monkeypatch, cfg)

    result = _valuation_run()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    dossier = payload["dossier_summary"]
    assert dossier["status"] == "DONE"
    assert dossier["tickers_built"] == ["KO"]
    assert dossier["tickers_synthesis_skipped"] == ["KO"]
    synthesis = dossier["ticker_results"]["KO"]["synthesis"]
    assert synthesis["status"] == "SKIPPED"
    assert synthesis["reason"] == "LLM_DISABLED_PROMPT_EVIDENCE_REFUSED"
    assert CODE in synthesis["violation_codes"]
    # The deterministic stages ran after the skipped step.
    assert payload["fundamentals_summary"]["ticker_count"] == 1
    assert payload["valuation_summary"]["ticker_count"] == 1
    assert payload["synthesis_skipped"]["tickers"] == ["KO"]


def test_valuation_run_still_fails_when_a_live_provider_would_see_unprovenanced_evidence(
    monkeypatch, tmp_path
):
    cfg = _init(monkeypatch, tmp_path, provider="openai")
    _stub_dossier_inputs(monkeypatch, cfg)

    result = _valuation_run()

    assert result.exit_code != 0
    assert isinstance(result.exception, InvalidFinancialInputError)
    summary = json.loads((cfg.dossiers_dir / "t1" / "dossier_summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["stop_reason_code"] in {"NEEDS_DATA", "INVALID_FINANCIAL_INPUT"}
