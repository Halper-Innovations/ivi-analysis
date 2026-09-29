"""Adversarial financial-integrity coverage for active discover entrypoints."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from app.autonomous.financial_integrity import InvalidFinancialInputError


class _Messages:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _Client:
    def __init__(self, responses=()):
        self.messages = _Messages(responses)


class _MutatingMessages(_Messages):
    def __init__(self, response, mutation):
        super().__init__([response])
        self.mutation = mutation

    def create(self, **kwargs):
        self.mutation()
        return super().create(**kwargs)


class _FailingMutatingMessages(_Messages):
    def __init__(self, mutation):
        super().__init__()
        self.mutation = mutation

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.mutation()
        raise RuntimeError("synthetic provider failure after input mutation")


@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _Tool:
    name: str
    input: dict
    id: str = "tool-1"
    type: str = "tool_use"


@dataclass
class _Response:
    content: list
    usage: _Usage = field(default_factory=_Usage)
    stop_reason: str = "tool_use"


def _ineligible_engine_db(path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE valuations (
                ticker TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                method TEXT NOT NULL,
                outputs_json TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO valuations
                (ticker, as_of_date, method, outputs_json)
            VALUES (?, ?, 'scorecard', ?)
            """,
            (
                "BAD",
                "2026-07-22",
                json.dumps({"pricing_zone_detail": {"current_price": 1.0}}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_discover_stage2_and_stage3_missing_packet_make_zero_calls() -> None:
    from app.discover.stage2 import Stage2Config, classify_ticker
    from app.discover.stage3 import Stage3Config, research_ticker

    client = _Client()
    with pytest.raises(InvalidFinancialInputError):
        classify_ticker(
            client,
            "BAD",
            "2026-07-22",
            {"pricing_zone_detail": {"current_price": 1.0}},
            Stage2Config(),
        )

    bundle = SimpleNamespace(as_of_date="2026-07-22")
    with pytest.raises(InvalidFinancialInputError):
        research_ticker(
            client,
            "BAD",
            Stage3Config(),
            lambda *_args, **_kwargs: bundle,
        )

    assert client.messages.calls == []


def test_discover_stage4_requires_packet_and_gates_every_turn() -> None:
    from app.discover.stage4 import Stage4Config, deep_research_ticker
    from tests.test_financial_integrity import _valid_packet

    missing_client = _Client()
    with pytest.raises(InvalidFinancialInputError):
        deep_research_ticker(
            missing_client,
            "MEGA",
            Stage4Config(),
            lambda ticker: {
                "ticker": ticker,
                "as_of_date": "2026-07-22",
                "user_message": ticker,
            },
            lambda _name, _input: "",
        )
    assert missing_client.messages.calls == []

    client = _Client(
        [
            _Response([_Tool("fetch_companyfacts", {"ticker": "MEGA", "line_items": ["revenue"]})]),
            _Response(
                [
                    _Tool(
                        "finalize_analysis",
                        {
                            "verdict": "WATCH",
                            "confidence": "MODERATE",
                            "thesis": "Bound",
                            "key_findings": [],
                            "open_questions": [],
                            "falsifiers": [],
                            "reasoning_trace": "Bound",
                        },
                    )
                ]
            ),
        ]
    )
    result = deep_research_ticker(
        client,
        "MEGA",
        Stage4Config(),
        lambda ticker: {
            "ticker": ticker,
            "as_of_date": "2026-07-22",
            "user_message": ticker,
        },
        lambda _name, _input: "revenue: 100 USD",
        financial_packet=_valid_packet(),
    )

    assert result.verdict == "WATCH"
    assert len(client.messages.calls) == 2


def test_discover_stage2_rejects_scorecard_mutation_during_paid_call(
    tmp_path,
) -> None:
    from app.discover.persistence import (
        create_sweep,
        ensure_schema,
        list_stage2_results,
    )
    from app.discover.stage2 import Stage2Config, run_stage2
    from tests.test_financial_integrity import _valid_packet

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(session_db, "stage2-mutation", 1, None, 1.0)
    scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "signal": "BUY_CANDIDATE",
        "pricing_zone_detail": {
            "current_price": 100.0,
            "dcf_base": 125.0,
            "epv_adjusted": 110.0,
        },
    }
    response = _Response(
        [
            _Tool(
                "classify_ticker",
                {
                    "decision": "KEEP",
                    "confidence": "HIGH",
                    "reason": "Old price",
                },
            )
        ]
    )
    messages = _MutatingMessages(
        response,
        lambda: scorecard["pricing_zone_detail"].__setitem__(
            "current_price",
            101.0,
        ),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_stage2(
            session_db,
            "stage2-mutation",
            [("MEGA", "2026-07-22", scorecard)],
            SimpleNamespace(messages=messages),
            config=Stage2Config(),
            financial_packets={"MEGA": _valid_packet()},
        )

    assert {item.code for item in exc_info.value.violations} == {"DISCOVER_FINANCIAL_SCOPE_MUTATED"}
    assert len(messages.calls) == 1
    assert list_stage2_results(session_db, "stage2-mutation") == []


def test_discover_stage3_rejects_bundle_mutation_during_paid_call(
    tmp_path,
) -> None:
    from app.discover.persistence import (
        create_sweep,
        ensure_schema,
        list_stage3_results,
    )
    from app.discover.stage3 import Stage3Config, run_stage3
    from tests.test_discover_stage3 import _sample_bundle
    from tests.test_financial_integrity import _valid_packet

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(session_db, "stage3-mutation", 1, None, 1.0)
    bundle = _sample_bundle("MEGA")
    bundle.as_of_date = "2026-07-22"
    response = _Response(
        [
            _Tool(
                "stage3_analysis",
                {
                    "verdict": "BUY_CANDIDATE",
                    "confidence": "HIGH",
                    "thesis_summary": "Old DCF",
                    "key_numbers": [],
                    "positives": [],
                    "risks": [],
                    "open_questions": [],
                    "reasoning_trace": "Old DCF",
                },
            )
        ]
    )
    messages = _MutatingMessages(
        response,
        lambda: setattr(bundle.valuation, "dcf_base", 999.0),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_stage3(
            session_db,
            "stage3-mutation",
            ["MEGA"],
            SimpleNamespace(messages=messages),
            config=Stage3Config(),
            bundle_builder=lambda *_args, **_kwargs: bundle,
            financial_packets={"MEGA": _valid_packet()},
        )

    assert {item.code for item in exc_info.value.violations} == {"DISCOVER_FINANCIAL_SCOPE_MUTATED"}
    assert len(messages.calls) == 1
    assert list_stage3_results(session_db, "stage3-mutation") == []


def test_discover_failed_calls_cannot_hide_stage2_or_stage3_input_mutation() -> None:
    from app.discover.stage2 import Stage2Config, classify_ticker
    from app.discover.stage3 import Stage3Config, research_ticker
    from tests.test_discover_stage3 import _sample_bundle
    from tests.test_financial_integrity import _valid_packet

    scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {
            "current_price": 100.0,
            "dcf_base": 125.0,
            "epv_adjusted": 110.0,
        },
    }
    stage2_messages = _FailingMutatingMessages(
        lambda: scorecard["pricing_zone_detail"].__setitem__("current_price", 101.0)
    )
    with pytest.raises(InvalidFinancialInputError) as stage2_error:
        classify_ticker(
            SimpleNamespace(messages=stage2_messages),
            "MEGA",
            "2026-07-22",
            scorecard,
            Stage2Config(),
            financial_packet=_valid_packet(),
        )

    bundle = _sample_bundle("MEGA")
    bundle.as_of_date = "2026-07-22"
    stage3_messages = _FailingMutatingMessages(lambda: setattr(bundle.valuation, "dcf_base", 999.0))
    with pytest.raises(InvalidFinancialInputError) as stage3_error:
        research_ticker(
            SimpleNamespace(messages=stage3_messages),
            "MEGA",
            Stage3Config(),
            lambda *_args, **_kwargs: bundle,
            financial_packet=_valid_packet(),
        )

    assert {item.code for item in stage2_error.value.violations} == {
        "DISCOVER_FINANCIAL_SCOPE_MUTATED"
    }
    assert {item.code for item in stage3_error.value.violations} == {
        "DISCOVER_FINANCIAL_SCOPE_MUTATED"
    }
    assert len(stage2_messages.calls) == 1
    assert len(stage3_messages.calls) == 1


def test_stage2_and_stage3_runners_write_separate_scope_authorizations(
    tmp_path,
) -> None:
    from app.discover.persistence import (
        create_sweep,
        ensure_schema,
        list_stage2_results,
        list_stage3_results,
    )
    from app.discover.stage2 import Stage2Config, run_stage2
    from app.discover.stage3 import Stage3Config, run_stage3
    from tests.test_discover_stage3 import _sample_bundle
    from tests.test_financial_integrity import _valid_packet

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(session_db, "stage23-authorization", 1, None, 1.0)
    packet = _valid_packet()
    scorecard = {
        "pricing_zone": "MARGIN_OF_SAFETY",
        "pricing_zone_detail": {
            "current_price": 100.0,
            "dcf_base": 125.0,
            "epv_adjusted": 110.0,
        },
    }
    run_stage2(
        session_db,
        "stage23-authorization",
        [("MEGA", "2026-07-22", scorecard)],
        _Client(
            [
                _Response(
                    [
                        _Tool(
                            "classify_ticker",
                            {
                                "decision": "KEEP",
                                "confidence": "HIGH",
                                "reason": "Bound Stage 2",
                            },
                        )
                    ]
                )
            ]
        ),
        config=Stage2Config(),
        financial_packets={"MEGA": packet},
    )
    bundle = _sample_bundle("MEGA")
    bundle.as_of_date = "2026-07-22"
    run_stage3(
        session_db,
        "stage23-authorization",
        ["MEGA"],
        _Client(
            [
                _Response(
                    [
                        _Tool(
                            "stage3_analysis",
                            {
                                "verdict": "WATCH",
                                "confidence": "MODERATE",
                                "thesis_summary": "Bound Stage 3",
                                "key_numbers": [],
                                "positives": [],
                                "risks": [],
                                "open_questions": [],
                                "reasoning_trace": "Bound",
                            },
                        )
                    ]
                )
            ]
        ),
        config=Stage3Config(),
        bundle_builder=lambda *_args, **_kwargs: bundle,
        financial_packets={"MEGA": packet},
    )

    assert [
        row["decision"]
        for row in list_stage2_results(
            session_db,
            "stage23-authorization",
            require_financial_scope=True,
        )
    ] == ["KEEP"]
    assert [
        row["verdict"]
        for row in list_stage3_results(
            session_db,
            "stage23-authorization",
            require_financial_scope=True,
        )
    ] == ["WATCH"]
    conn = sqlite3.connect(session_db)
    try:
        assert (
            conn.execute(
                """
                SELECT COUNT(*)
                FROM discover_publication_authorizations
                WHERE sweep_id = 'stage23-authorization'
                """
            ).fetchone()[0]
            == 2
        )
    finally:
        conn.close()


def test_discover_authorization_failure_rolls_back_result_and_ledger_atomically(
    tmp_path,
    monkeypatch,
) -> None:
    from app.discover import persistence as persistence_module
    import app.discover.stage2 as stage2_module
    from app.discover.persistence import create_sweep, ensure_schema, list_stage2_results
    from tests.test_financial_integrity import _valid_packet

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(session_db, "post-insert-race", 1, None, 1.0)
    real_authorize = persistence_module._authorize_discover_result_in_transaction

    def mutate_and_rehash(conn, *args, **kwargs):
        conn.execute(
            """
            UPDATE stage2_results
            SET reason = 'tampered after insert'
            WHERE sweep_id = 'post-insert-race' AND ticker = 'MEGA'
            """
        )
        row = conn.execute(
            """
            SELECT *
            FROM stage2_results
            WHERE sweep_id = 'post-insert-race' AND ticker = 'MEGA'
            """
        ).fetchone()
        tampered_sha256 = persistence_module._publication_row_sha256(2, row)
        conn.execute(
            """
            UPDATE stage2_results
            SET publication_row_sha256 = ?
            WHERE sweep_id = 'post-insert-race' AND ticker = 'MEGA'
            """,
            (tampered_sha256,),
        )
        return real_authorize(conn, *args, **kwargs)

    monkeypatch.setattr(
        persistence_module,
        "_authorize_discover_result_in_transaction",
        mutate_and_rehash,
    )
    response = _Response(
        [
            _Tool(
                "classify_ticker",
                {
                    "decision": "KEEP",
                    "confidence": "HIGH",
                    "reason": "original provider decision",
                },
            )
        ]
    )

    with pytest.raises(ValueError, match="changed before authorization"):
        stage2_module.run_stage2(
            session_db,
            "post-insert-race",
            [
                (
                    "MEGA",
                    "2026-07-22",
                    {
                        "pricing_zone": "MARGIN_OF_SAFETY",
                        "pricing_zone_detail": {
                            "current_price": 100.0,
                            "dcf_base": 125.0,
                            "epv_adjusted": 110.0,
                        },
                    },
                )
            ],
            _Client([response]),
            config=stage2_module.Stage2Config(),
            financial_packets={"MEGA": _valid_packet()},
        )

    assert (
        list_stage2_results(
            session_db,
            "post-insert-race",
            require_financial_scope=True,
        )
        == []
    )
    conn = sqlite3.connect(session_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM stage2_results").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM discover_publication_authorizations").fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_discover_stage4_rejects_context_mutation_during_paid_call(
    tmp_path,
) -> None:
    from app.discover.persistence import (
        create_sweep,
        ensure_schema,
        list_stage4_results,
    )
    from app.discover.stage4 import Stage4Config, run_stage4
    from tests.test_financial_integrity import _valid_packet

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(session_db, "stage4-mutation", 1, None, 1.0)
    context = {
        "ticker": "MEGA",
        "as_of_date": "2026-07-22",
        "user_message": "price=100",
    }
    response = _Response(
        [
            _Tool(
                "finalize_analysis",
                {
                    "verdict": "BUY",
                    "confidence": "HIGH",
                    "thesis": "Old context",
                    "key_findings": [],
                    "open_questions": [],
                    "falsifiers": [],
                    "reasoning_trace": "Old context",
                },
            )
        ]
    )
    messages = _MutatingMessages(
        response,
        lambda: context.__setitem__("user_message", "price=1"),
    )

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_stage4(
            session_db,
            "stage4-mutation",
            ["MEGA"],
            SimpleNamespace(messages=messages),
            config=Stage4Config(max_turns=1),
            context_builder=lambda _ticker: context,
            tool_dispatcher=lambda _name, _input: "",
            financial_packets={"MEGA": _valid_packet()},
        )

    assert {item.code for item in exc_info.value.violations} == {"DISCOVER_FINANCIAL_SCOPE_MUTATED"}
    assert len(messages.calls) == 1
    assert list_stage4_results(session_db, "stage4-mutation") == []


def test_discover_failed_call_cannot_hide_stage4_context_mutation() -> None:
    from app.discover.stage4 import Stage4Config, deep_research_ticker
    from tests.test_financial_integrity import _valid_packet

    context = {
        "ticker": "MEGA",
        "as_of_date": "2026-07-22",
        "user_message": "price=100",
    }
    messages = _FailingMutatingMessages(lambda: context.__setitem__("user_message", "price=1"))

    with pytest.raises(InvalidFinancialInputError) as caught:
        deep_research_ticker(
            SimpleNamespace(messages=messages),
            "MEGA",
            Stage4Config(max_turns=1),
            lambda _ticker: context,
            lambda _name, _input: "",
            financial_packet=_valid_packet(),
        )

    assert {item.code for item in caught.value.violations} == {"DISCOVER_FINANCIAL_SCOPE_MUTATED"}
    assert len(messages.calls) == 1


def test_discover_stage4_rejects_changed_tool_evidence_before_persistence(
    tmp_path,
) -> None:
    from app.discover.persistence import (
        create_sweep,
        ensure_schema,
        list_stage4_results,
    )
    from app.discover.stage4 import Stage4Config, run_stage4
    from tests.test_financial_integrity import _valid_packet

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(session_db, "stage4-tool-mutation", 1, None, 1.0)
    client = _Client(
        [
            _Response(
                [
                    _Tool(
                        "fetch_companyfacts",
                        {"ticker": "MEGA", "line_items": ["revenue"]},
                    )
                ]
            ),
            _Response(
                [
                    _Tool(
                        "finalize_analysis",
                        {
                            "verdict": "BUY",
                            "confidence": "HIGH",
                            "thesis": "Old tool evidence",
                            "key_findings": [],
                            "open_questions": [],
                            "falsifiers": [],
                            "reasoning_trace": "Old tool evidence",
                        },
                    )
                ]
            ),
        ]
    )
    tool_outputs = iter(("revenue: 100 USD", "revenue: 101 USD"))

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_stage4(
            session_db,
            "stage4-tool-mutation",
            ["MEGA"],
            client,
            config=Stage4Config(),
            context_builder=lambda ticker: {
                "ticker": ticker,
                "as_of_date": "2026-07-22",
                "user_message": f"Analyze {ticker}",
            },
            tool_dispatcher=lambda _name, _input: next(tool_outputs),
            financial_packets={"MEGA": _valid_packet()},
        )

    assert {item.code for item in exc_info.value.violations} == {"DISCOVER_FINANCIAL_SCOPE_MUTATED"}
    assert len(client.messages.calls) == 2
    assert list_stage4_results(session_db, "stage4-tool-mutation") == []


def test_discover_stage4_cache_binds_dynamic_tool_evidence(
    tmp_path,
) -> None:
    from app.discover.persistence import create_sweep, ensure_schema
    from app.discover.stage4 import Stage4Config, run_stage4
    from tests.test_financial_integrity import _valid_packet

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(
        session_db,
        "stage4-tool-lineage",
        universe_size=1,
        limit_applied=None,
        budget_usd=1.0,
    )
    context_builder = lambda ticker: {
        "ticker": ticker,
        "as_of_date": "2026-07-22",
        "user_message": f"Analyze {ticker}",
    }
    first_client = _Client(
        [
            _Response(
                [
                    _Tool(
                        "fetch_companyfacts",
                        {"ticker": "MEGA", "line_items": ["revenue"]},
                    )
                ]
            ),
            _Response(
                [
                    _Tool(
                        "finalize_analysis",
                        {
                            "verdict": "WATCH",
                            "confidence": "MODERATE",
                            "thesis": "Bound tool evidence",
                            "key_findings": [],
                            "open_questions": [],
                            "falsifiers": [],
                            "reasoning_trace": "Bound",
                        },
                    )
                ]
            ),
        ]
    )
    run_stage4(
        session_db,
        "stage4-tool-lineage",
        ["MEGA"],
        first_client,
        config=Stage4Config(),
        context_builder=context_builder,
        tool_dispatcher=lambda _name, _input: "revenue: 100 USD",
        financial_packets={"MEGA": _valid_packet()},
    )
    assert len(first_client.messages.calls) == 2
    conn = sqlite3.connect(session_db)
    conn.row_factory = sqlite3.Row
    try:
        stage_row = conn.execute(
            """
            SELECT publication_row_sha256, financial_scope_fingerprint,
                   publication_evidence_json
            FROM stage4_results
            WHERE sweep_id = 'stage4-tool-lineage' AND ticker = 'MEGA'
            """
        ).fetchone()
        authorization = conn.execute(
            """
            SELECT publication_row_sha256, financial_scope_fingerprint,
                   publication_evidence_sha256
            FROM discover_publication_authorizations
            WHERE stage = 4
              AND sweep_id = 'stage4-tool-lineage'
              AND ticker = 'MEGA'
            """
        ).fetchall()
    finally:
        conn.close()
    assert len(authorization) == 1
    assert authorization[0]["publication_row_sha256"] == stage_row["publication_row_sha256"]
    assert (
        authorization[0]["financial_scope_fingerprint"] == stage_row["financial_scope_fingerprint"]
    )
    assert len(authorization[0]["publication_evidence_sha256"]) == 64

    matching_client = _Client()
    assert (
        run_stage4(
            session_db,
            "stage4-tool-lineage",
            ["MEGA"],
            matching_client,
            config=Stage4Config(),
            context_builder=context_builder,
            tool_dispatcher=lambda _name, _input: "revenue: 100 USD",
            financial_packets={"MEGA": _valid_packet()},
        )
        == []
    )
    assert matching_client.messages.calls == []

    changed_client = _Client()
    with pytest.raises(InvalidFinancialInputError) as exc_info:
        run_stage4(
            session_db,
            "stage4-tool-lineage",
            ["MEGA"],
            changed_client,
            config=Stage4Config(),
            context_builder=context_builder,
            tool_dispatcher=lambda _name, _input: "revenue: 101 USD",
            financial_packets={"MEGA": _valid_packet()},
        )

    assert "DISCOVER_RESULT_FINANCIAL_SCOPE_MISMATCH" in {
        violation.code for violation in exc_info.value.violations
    }
    assert changed_client.messages.calls == []


def test_fresh_discover_rejects_ineligible_newest_row_before_client(
    tmp_path,
    monkeypatch,
) -> None:
    from typer.testing import CliRunner

    from app.cli import app

    engine_db = tmp_path / "engine.db"
    _ineligible_engine_db(engine_db)
    monkeypatch.setattr(
        "app.config.get_config",
        lambda: SimpleNamespace(
            db_path=engine_db,
            anthropic_api_key="must-not-be-used",
        ),
    )
    constructed: list[str] = []
    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda **_kwargs: constructed.append("constructed"),
    )

    result = CliRunner().invoke(app, ["discover"])

    assert result.exit_code != 0
    assert isinstance(result.exception, InvalidFinancialInputError)
    assert constructed == []


def test_resume_rejects_ineligible_snapshot_before_calls_or_state_change(
    tmp_path,
) -> None:
    from app.discover.persistence import (
        create_sweep,
        ensure_schema,
        get_sweep,
        insert_sweep_universe,
        update_sweep_status,
    )
    from app.discover.sweep import resume_sweep

    session_db = tmp_path / "discover_sessions.db"
    engine_db = tmp_path / "engine.db"
    _ineligible_engine_db(engine_db)
    ensure_schema(session_db)
    create_sweep(
        session_db,
        "resume-invalid",
        universe_size=1,
        limit_applied=None,
        budget_usd=1.0,
    )
    insert_sweep_universe(
        session_db,
        "resume-invalid",
        [("BAD", "2026-07-22")],
    )
    update_sweep_status(session_db, "resume-invalid", "aborted")
    client = _Client()

    with pytest.raises(InvalidFinancialInputError):
        resume_sweep(
            session_db,
            engine_db,
            "resume-invalid",
            client,
        )

    assert client.messages.calls == []
    assert get_sweep(session_db, "resume-invalid")["status"] == "aborted"


@pytest.mark.parametrize(
    ("stored_fingerprint", "expected_code"),
    [
        (None, "DISCOVER_RESULT_FINANCIAL_SCOPE_MISSING"),
        ("0" * 64, "DISCOVER_RESULT_FINANCIAL_SCOPE_MISSING"),
    ],
)
def test_resume_rejects_unbound_or_stale_results_without_paid_rerun(
    tmp_path,
    monkeypatch,
    stored_fingerprint,
    expected_code,
) -> None:
    from app.discover.persistence import (
        build_discover_publication_evidence,
        create_sweep,
        ensure_schema,
        finalize_sweep,
        get_sweep,
        insert_stage2_result,
        insert_stage3_result,
        insert_stage4_result,
        insert_sweep_universe,
    )
    from app.discover.report import render_sweep_report
    from app.discover.sweep import resume_sweep
    from tests.test_financial_integrity import _valid_packet

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(
        session_db,
        "stale-resume",
        universe_size=1,
        limit_applied=None,
        budget_usd=1.0,
    )
    insert_sweep_universe(
        session_db,
        "stale-resume",
        [("MEGA", "2026-07-22")],
    )
    evidence = (
        {
            stage: build_discover_publication_evidence(
                stage=stage,
                ticker="MEGA",
                scope_fingerprint=stored_fingerprint,
                primary_evidence={"fixture": f"stale-stage-{stage}"},
                stage4_tool_manifest=[] if stage == 4 else None,
            )
            for stage in (2, 3, 4)
        }
        if stored_fingerprint is not None
        else {}
    )
    insert_stage2_result(
        session_db,
        sweep_id="stale-resume",
        ticker="MEGA",
        decision="KEEP",
        confidence="HIGH",
        reason="STALE_STAGE2_REASON",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        wall_ms=1,
        financial_scope_fingerprint=stored_fingerprint,
        financial_scope_publication_fingerprint=stored_fingerprint,
        publication_evidence=evidence.get(2),
    )
    insert_stage3_result(
        session_db,
        sweep_id="stale-resume",
        ticker="MEGA",
        verdict="WATCH",
        confidence="HIGH",
        thesis_summary="STALE_STAGE3_THESIS",
        key_numbers=[],
        positives=[],
        risks=[],
        open_questions=[],
        reasoning_trace="stale",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        wall_ms=1,
        financial_scope_fingerprint=stored_fingerprint,
        financial_scope_publication_fingerprint=stored_fingerprint,
        publication_evidence=evidence.get(3),
    )
    insert_stage4_result(
        session_db,
        sweep_id="stale-resume",
        ticker="MEGA",
        verdict="BUY",
        confidence="HIGH",
        thesis="STALE_STAGE4_BUY_THESIS",
        key_findings=[],
        open_questions=[],
        falsifiers=[],
        reasoning_trace="stale",
        num_turns=1,
        tool_call_counts={},
        termination_reason="finalize_analysis",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        wall_seconds=0.01,
        financial_scope_fingerprint=stored_fingerprint,
        financial_scope_publication_fingerprint=stored_fingerprint,
        financial_scope_manifest=[] if stored_fingerprint is not None else None,
        publication_evidence=evidence.get(4),
    )
    finalize_sweep(session_db, "stale-resume", "aborted")

    packet = _valid_packet()
    monkeypatch.setattr(
        "app.discover.sweep._load_scorecards_for_snapshot",
        lambda *_args, **_kwargs: (
            [
                (
                    "MEGA",
                    "2026-07-22",
                    {
                        "pricing_zone": "MARGIN_OF_SAFETY",
                        "pricing_zone_detail": {
                            "current_price": 100.0,
                            "dcf_base": 125.0,
                            "epv_adjusted": 110.0,
                        },
                    },
                )
            ],
            {"MEGA": packet},
        ),
    )
    client = _Client()

    with pytest.raises(InvalidFinancialInputError) as exc_info:
        resume_sweep(
            session_db,
            tmp_path / "unused-engine.db",
            "stale-resume",
            client,
            bundle_builder=lambda *_args, **_kwargs: None,
            stage4_context_builder=lambda ticker: {
                "ticker": ticker,
                "as_of_date": "2026-07-22",
                "user_message": ticker,
            },
            stage4_tool_dispatcher=lambda _name, _input: "",
        )

    assert expected_code in {violation.code for violation in exc_info.value.violations}
    assert client.messages.calls == []
    assert get_sweep(session_db, "stale-resume")["status"] == "invalid_financial_input"
    report = render_sweep_report(session_db, "stale-resume")
    assert "INVALID_FINANCIAL_INPUT" in report
    assert "STALE_STAGE2_REASON" not in report
    assert "STALE_STAGE3_THESIS" not in report
    assert "STALE_STAGE4_BUY_THESIS" not in report


def test_completed_report_suppresses_legacy_unbound_decisions(
    tmp_path,
) -> None:
    from app.discover.persistence import (
        create_sweep,
        ensure_schema,
        finalize_sweep,
        insert_stage4_result,
    )
    from app.discover.report import render_sweep_report

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(
        session_db,
        "legacy-report",
        universe_size=1,
        limit_applied=None,
        budget_usd=1.0,
    )
    insert_stage4_result(
        session_db,
        sweep_id="legacy-report",
        ticker="MEGA",
        verdict="BUY",
        confidence="HIGH",
        thesis="LEGACY_STALE_BUY_THESIS",
        key_findings=[],
        open_questions=[],
        falsifiers=[],
        reasoning_trace="legacy",
        num_turns=1,
        tool_call_counts={},
        termination_reason="finalize_analysis",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        wall_seconds=0.01,
        financial_scope_fingerprint="a" * 64,
    )
    insert_stage4_result(
        session_db,
        sweep_id="legacy-report",
        ticker="DRIFT",
        verdict="BUY",
        confidence="HIGH",
        thesis="MISMATCHED_PUBLICATION_BUY_THESIS",
        key_findings=[],
        open_questions=[],
        falsifiers=[],
        reasoning_trace="mismatched",
        num_turns=1,
        tool_call_counts={},
        termination_reason="finalize_analysis",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        wall_seconds=0.01,
        financial_scope_fingerprint="a" * 64,
        financial_scope_publication_fingerprint="b" * 64,
        financial_scope_manifest=[],
    )
    finalize_sweep(session_db, "legacy-report", "completed")

    report = render_sweep_report(session_db, "legacy-report")

    assert "legacy or unbound results suppressed" in report
    assert "LEGACY_STALE_BUY_THESIS" not in report
    assert "MISMATCHED_PUBLICATION_BUY_THESIS" not in report


def test_public_insert_helper_cannot_self_authorize_fabricated_stage4_buy(
    tmp_path,
) -> None:
    from app.discover.persistence import (
        build_discover_publication_evidence,
        create_sweep,
        ensure_schema,
        finalize_sweep,
        insert_stage4_result,
        list_stage4_results,
    )
    from app.discover.report import render_sweep_report

    session_db = tmp_path / "discover_sessions.db"
    ensure_schema(session_db)
    create_sweep(
        session_db,
        "fabricated-buy",
        universe_size=1,
        limit_applied=None,
        budget_usd=1.0,
    )
    invented_scope = "a" * 64
    invented_evidence = build_discover_publication_evidence(
        stage=4,
        ticker="FAKE",
        scope_fingerprint=invented_scope,
        primary_evidence={"invented": "provider context"},
        stage4_tool_manifest=[],
    )
    insert_stage4_result(
        session_db,
        sweep_id="fabricated-buy",
        ticker="FAKE",
        verdict="BUY",
        confidence="HIGH",
        thesis="FABRICATED_SELF_AUTHORIZED_BUY",
        key_findings=["invented finding"],
        open_questions=[],
        falsifiers=[],
        reasoning_trace="invented",
        num_turns=1,
        tool_call_counts={},
        termination_reason="finalize_analysis",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        wall_seconds=0.01,
        financial_scope_fingerprint=invented_scope,
        financial_scope_publication_fingerprint=invented_scope,
        financial_scope_manifest=[],
        publication_evidence=invented_evidence,
    )
    finalize_sweep(session_db, "fabricated-buy", "completed")

    assert (
        list_stage4_results(
            session_db,
            "fabricated-buy",
            require_financial_scope=True,
        )
        == []
    )
    report = render_sweep_report(session_db, "fabricated-buy")
    assert "FABRICATED_SELF_AUTHORIZED_BUY" not in report
    assert "Stage 4: 1" in report

    conn = sqlite3.connect(session_db)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM discover_publication_authorizations").fetchone()[0]
            == 0
        )
    finally:
        conn.close()
