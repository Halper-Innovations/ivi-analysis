from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.autonomous.financial_integrity import InvalidFinancialInputError
from app.autonomous.v1_financial_context import (
    bind_v1_financial_scope as real_bind_v1_financial_scope,
)
from app.cli import app
from app.diff.engine import (
    MAX_SECTION_CHARS,
    _classify_material_changes,
    build_filing_diff_for_ticker,
    build_filing_diff_report,
)
from app.diff.schemas import FilingDiffReport, validate_filing_diff_report
from app.llm.providers.disabled_provider import LLMResult
from app.llm.usage_capture import (
    attached_provider_usage_records,
    provider_usage_capture,
)
from tests.test_financial_integrity import _valid_packet


runner = CliRunner()


@pytest.fixture(autouse=True)
def _stub_financial_authorization(monkeypatch):
    """Diff rendering tests use synthetic dossiers without quote lineage."""

    class _Scope:
        def require(self, **_kwargs):
            return None

    monkeypatch.setattr(
        "app.diff.engine.bind_v1_financial_scope",
        lambda **_kwargs: _Scope(),
    )


def _init_temp_data(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config as _get_config

    _get_config.cache_clear()
    return _get_config()


def _write_filing(path: Path, *, risk_text: str | None, mdna_text: str | None) -> None:
    business = "Item 1. Business Our business sells software and cloud services."
    parts = [business]
    if risk_text is not None:
        parts.append(f"Item 1A. Risk Factors {risk_text}")
    if mdna_text is not None:
        parts.append(f"Item 7. Management's Discussion and Analysis {mdna_text}")
    parts.append("Item 8. Financial Statements and Supplementary Data audited statements here.")
    path.write_text("\n\n".join(parts), encoding="utf-8")


def _seed_dossier_run(
    cfg, *, run_id: str, ticker: str = "MSFT", long_risk: bool = False, missing_mdna: bool = False
):
    run_dir = cfg.dossiers_dir / run_id / ticker
    run_dir.mkdir(parents=True, exist_ok=True)
    filing_2024 = run_dir / "2024_10k.txt"
    filing_2025 = run_dir / "2025_10k.txt"
    filing_2026 = run_dir / "2026_10k.txt"

    base_risk = "We depend on enterprise demand and channel execution. " * 8
    changed_risk = (
        (
            "We depend on enterprise demand, channel execution, and AI infrastructure capacity. "
            * 600
        )
        if long_risk
        else "We depend on enterprise demand, channel execution, and AI infrastructure capacity. "
        * 8
    )
    older_mdna = (
        "We are investing in cloud capacity, datacenter buildout, and platform expansion. " * 8
    )
    newer_mdna = (
        "We have completed major capacity investments and are focusing on harvesting utilization and margin leverage. "
        * 8
    )

    _write_filing(filing_2024, risk_text=base_risk, mdna_text=older_mdna)
    _write_filing(
        filing_2025,
        risk_text=changed_risk,
        mdna_text=None if missing_mdna else newer_mdna,
    )
    _write_filing(
        filing_2026,
        risk_text="We face evolving competitive and cybersecurity threats. " * 8,
        mdna_text="We continue optimizing cloud utilization and integrating acquired assets. " * 8,
    )

    dossier = {
        "ticker": ticker,
        "run_id": run_id,
        "as_of_date": "2026-03-19",
        "docket": [
            {
                "ticker": ticker,
                "accession": "0001-2026",
                "form_type": "10-K",
                "filing_date": "2026-07-30",
                "period_end": "2026-06-30",
                "primary_doc_url": "https://www.sec.gov/example-2026",
                "local_path": str(filing_2026),
            },
            {
                "ticker": ticker,
                "accession": "0001-2025",
                "form_type": "10-K",
                "filing_date": "2025-07-30",
                "period_end": "2025-06-30",
                "primary_doc_url": "https://www.sec.gov/example-2025",
                "local_path": str(filing_2025),
            },
            {
                "ticker": ticker,
                "accession": "0001-2024",
                "form_type": "10-K",
                "filing_date": "2024-07-30",
                "period_end": "2024-06-30",
                "primary_doc_url": "https://www.sec.gov/example-2024",
                "local_path": str(filing_2024),
            },
        ],
        "section_spans": {},
        "items": [],
        "time_series": {},
        "artifacts": {},
    }
    (run_dir / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
    return run_dir


class _DisabledProvider:
    provider_name = "disabled"

    def enabled(self) -> bool:
        return False


class _FakeDiffProvider:
    provider_name = "openai"

    def __init__(self):
        self.calls = 0

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
        _ = (prompt, schema, schema_name)
        self.calls += 1
        payload = {
            "changes": [
                {
                    "section": "risk_factors",
                    "fiscal_year_from": 2024,
                    "fiscal_year_to": 2025,
                    "change_type": "RISK_SIGNAL",
                    "materiality": "HIGH",
                    "summary": "The company added AI infrastructure capacity as a named risk, signaling a new operational dependency.",
                    "from_excerpt": "We depend on enterprise demand and channel execution.",
                    "to_excerpt": "We depend on enterprise demand, channel execution, and AI infrastructure capacity.",
                }
            ]
        }
        return LLMResult(
            json_text=json.dumps(payload),
            model="gpt-5-mini",
            usage_input_tokens=800,
            usage_output_tokens=200,
            raw={},
        )


class _NoisyDiffProvider:
    provider_name = "openai"

    def enabled(self) -> bool:
        return True

    def synthesize_json(self, *, prompt: str, schema: dict, schema_name: str | None = None):
        _ = (prompt, schema, schema_name)
        payload = {
            "changes": [
                {
                    "section": "md_and_a",
                    "fiscal_year_from": 2024,
                    "fiscal_year_to": 2025,
                    "change_type": "QUANTITATIVE_CHANGE",
                    "materiality": "MEDIUM",
                    "summary": "Microsoft Cloud revenue increased year over year.",
                    "from_excerpt": "Microsoft Cloud revenue increased 23% to $137.4 billion.",
                    "to_excerpt": "Microsoft Cloud revenue increased 23% to $168.9 billion.",
                },
                {
                    "section": "md_and_a",
                    "fiscal_year_from": 2024,
                    "fiscal_year_to": 2025,
                    "change_type": "QUANTITATIVE_CHANGE",
                    "materiality": "MEDIUM",
                    "summary": "Subscriber counts increased year over year.",
                    "from_excerpt": "Subscribers increased to 82.5 million.",
                    "to_excerpt": "Subscribers increased to 95.0 million.",
                },
                {
                    "section": "md_and_a",
                    "fiscal_year_from": 2024,
                    "fiscal_year_to": 2025,
                    "change_type": "STRATEGIC_SIGNAL",
                    "materiality": "HIGH",
                    "summary": "New disclosure describes the OpenAI strategic partnership and Azure exclusivity.",
                    "from_excerpt": None,
                    "to_excerpt": "Microsoft and OpenAI maintain a long-term strategic partnership...",
                },
                {
                    "section": "md_and_a",
                    "fiscal_year_from": 2024,
                    "fiscal_year_to": 2025,
                    "change_type": "LANGUAGE_SHIFT",
                    "materiality": "HIGH",
                    "summary": "Company reorganized reportable segments and recast prior periods.",
                    "from_excerpt": "We report our financial performance based on the following segments...",
                    "to_excerpt": "In August 2024, we announced changes to the composition of our segments...",
                },
                {
                    "section": "md_and_a",
                    "fiscal_year_from": 2024,
                    "fiscal_year_to": 2025,
                    "change_type": "LANGUAGE_SHIFT",
                    "materiality": "MEDIUM",
                    "summary": "Company reorganized reportable segments and recast prior periods for comparability.",
                    "from_excerpt": "We report our financial performance based on the following segments...",
                    "to_excerpt": "Prior period segment information has been recast...",
                },
            ]
        }
        return LLMResult(
            json_text=json.dumps(payload),
            model="gpt-5-mini",
            usage_input_tokens=900,
            usage_output_tokens=250,
            raw={},
        )


def test_filing_diff_schema_round_trip(monkeypatch, tmp_path):
    cfg = _init_temp_data(monkeypatch, tmp_path)
    _seed_dossier_run(cfg, run_id="diff_schema")
    monkeypatch.setattr("app.diff.engine.get_llm_provider", lambda: _DisabledProvider())

    report = build_filing_diff_report(ticker="MSFT", run_id="diff_schema", years_back=3)
    payload = report.model_dump(mode="json")
    validated = validate_filing_diff_report(payload)
    assert isinstance(validated, FilingDiffReport)
    assert validated.ticker == "MSFT"


def test_filing_diff_uses_mock_llm_and_persists(monkeypatch, tmp_path):
    cfg = _init_temp_data(monkeypatch, tmp_path)
    _seed_dossier_run(cfg, run_id="diff_llm")
    fake = _FakeDiffProvider()
    monkeypatch.setattr("app.diff.engine.get_llm_provider", lambda: fake)

    path = build_filing_diff_for_ticker(ticker="MSFT", run_id="diff_llm", years_back=3)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.exists()
    assert payload["ticker"] == "MSFT"
    assert payload["llm_enabled"] is True
    assert len(payload["changes"]) >= 1
    assert fake.calls == 4


def test_filing_diff_rebuilds_bound_scenario_and_carries_billed_usage(
    monkeypatch,
) -> None:
    packet = _valid_packet()

    class _MutatingProvider:
        provider_name = "openai"

        def synthesize_json(self, **_kwargs):
            packet["current_price"] = 101.0
            return LLMResult(
                json_text='{"changes":[]}',
                model="gpt-5-mini",
                usage_input_tokens=100,
                usage_output_tokens=20,
                raw={},
            )

    monkeypatch.setattr(
        "app.diff.engine.bind_v1_financial_scope",
        real_bind_v1_financial_scope,
    )
    monkeypatch.setattr(
        "app.diff.engine.get_llm_provider",
        lambda: _MutatingProvider(),
    )

    with (
        provider_usage_capture("rlm_executor") as captured,
        pytest.raises(InvalidFinancialInputError) as exc_info,
    ):
        _classify_material_changes(
            ticker="MEGA",
            section="risk_factors",
            fiscal_year_from=2024,
            fiscal_year_to=2025,
            older_text="Legacy risk language. " * 20,
            newer_text="Updated risk language. " * 20,
            canonical_packet=packet,
            run_as_of_date="2026-07-21",
        )

    assert len(captured) == 1
    assert captured[0]["cost_estimate_usd"] > 0.0
    attached = attached_provider_usage_records(exc_info.value)
    assert len(attached) == 1
    assert attached[0]["cost_estimate_usd"] == captured[0]["cost_estimate_usd"]


def test_filing_diff_failed_call_cannot_hide_scope_mutation(monkeypatch) -> None:
    packet = _valid_packet()

    class _FailingMutatingProvider:
        provider_name = "openai"

        def __init__(self) -> None:
            self.calls = 0

        def synthesize_json(self, **_kwargs):
            self.calls += 1
            packet["current_price"] = 101.0
            raise RuntimeError("provider failed after mutating bound input")

    provider = _FailingMutatingProvider()
    monkeypatch.setattr(
        "app.diff.engine.bind_v1_financial_scope",
        real_bind_v1_financial_scope,
    )
    monkeypatch.setattr(
        "app.diff.engine.get_llm_provider",
        lambda: provider,
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        _classify_material_changes(
            ticker="MEGA",
            section="risk_factors",
            fiscal_year_from=2024,
            fiscal_year_to=2025,
            older_text="Legacy risk language. " * 20,
            newer_text="Updated risk language. " * 20,
            canonical_packet=packet,
            run_as_of_date="2026-07-21",
        )

    assert provider.calls == 1
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert {violation.code for violation in exc_info.value.violations} == {
        "SCENARIO_QUOTE_VALUE_MISMATCH"
    }


def test_filing_diff_disabled_provider_fallback_records_similarity(monkeypatch, tmp_path):
    cfg = _init_temp_data(monkeypatch, tmp_path)
    _seed_dossier_run(cfg, run_id="diff_disabled")
    monkeypatch.setattr("app.diff.engine.get_llm_provider", lambda: _DisabledProvider())

    report = build_filing_diff_report(ticker="MSFT", run_id="diff_disabled", years_back=3)

    assert report.llm_enabled is False
    assert report.changes == []
    assert report.section_diagnostics
    assert all(
        diag.similarity_ratio is not None
        for diag in report.section_diagnostics
        if diag.llm_status == "disabled"
    )


def test_filing_diff_truncation_sets_flag(monkeypatch, tmp_path):
    cfg = _init_temp_data(monkeypatch, tmp_path)
    _seed_dossier_run(cfg, run_id="diff_trunc", long_risk=True)
    monkeypatch.setattr("app.diff.engine.get_llm_provider", lambda: _DisabledProvider())

    report = build_filing_diff_report(ticker="MSFT", run_id="diff_trunc", years_back=3)

    assert report.truncation_applied is True
    assert any(
        diag.truncated_from or diag.truncated_to
        for diag in report.section_diagnostics
        if diag.compared_chars_from <= MAX_SECTION_CHARS
        or diag.compared_chars_to <= MAX_SECTION_CHARS
    )


def test_filing_diff_missing_section_records_structured_skip(monkeypatch, tmp_path):
    cfg = _init_temp_data(monkeypatch, tmp_path)
    _seed_dossier_run(cfg, run_id="diff_missing", missing_mdna=True)
    monkeypatch.setattr("app.diff.engine.get_llm_provider", lambda: _DisabledProvider())

    report = build_filing_diff_report(ticker="MSFT", run_id="diff_missing", years_back=3)

    assert any(
        skip.section == "md_and_a" and skip.reason == "SECTION_MISSING"
        for skip in report.sections_skipped
    )


def test_filing_diff_cli_commands(monkeypatch, tmp_path):
    cfg = _init_temp_data(monkeypatch, tmp_path)
    _seed_dossier_run(cfg, run_id="diff_cli", ticker="MSFT")
    _seed_dossier_run(cfg, run_id="diff_cli", ticker="AAPL")
    monkeypatch.setattr("app.diff.engine.get_llm_provider", lambda: _DisabledProvider())

    single = runner.invoke(
        app,
        ["filing-diff", "--ticker", "MSFT", "--run-id", "diff_cli", "--years-back", "3"],
    )
    assert single.exit_code == 0, single.output
    assert "MSFT_diff_cli_diff.json" in single.output

    batch = runner.invoke(
        app,
        ["filing-diff-all", "--run-id", "diff_cli", "--years-back", "3", "--limit", "1"],
    )
    assert batch.exit_code == 0, batch.output
    payload = json.loads(batch.output)
    assert payload["run_id"] == "diff_cli"
    assert len(payload["tickers_processed"]) == 1


def test_filing_diff_filters_routine_mdna_refreshes_and_dedupes(monkeypatch, tmp_path):
    cfg = _init_temp_data(monkeypatch, tmp_path)
    _seed_dossier_run(cfg, run_id="diff_filter")
    monkeypatch.setattr("app.diff.engine.get_llm_provider", lambda: _NoisyDiffProvider())

    report = build_filing_diff_report(ticker="MSFT", run_id="diff_filter", years_back=3)
    kept = [
        change
        for change in report.changes
        if change.section == "md_and_a"
        and change.fiscal_year_from == 2024
        and change.fiscal_year_to == 2025
    ]

    assert len(kept) == 2
    assert any(
        "OpenAI" in change.summary or "OpenAI" in (change.to_excerpt or "") for change in kept
    )
    assert any("segments" in change.summary for change in kept)


def test_filing_diff_caps_total_changes(monkeypatch, tmp_path):
    cfg = _init_temp_data(monkeypatch, tmp_path)
    _seed_dossier_run(cfg, run_id="diff_cap")
    monkeypatch.setattr("app.diff.engine.get_llm_provider", lambda: _NoisyDiffProvider())

    report = build_filing_diff_report(ticker="MSFT", run_id="diff_cap", years_back=3)

    assert len(report.changes) <= 10
