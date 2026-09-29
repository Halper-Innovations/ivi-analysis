from __future__ import annotations

import contextvars
import hashlib
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import pytest

# CI runners (GITHUB_ACTIONS, FORCE_COLOR) make Typer force a colour terminal, which
# splits the CLI text tests assert on with ANSI codes. Tests read plain output.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ.pop("FORCE_COLOR", None)


class HermeticityViolation(RuntimeError):
    """Raised before a test can cross a protected I/O boundary."""


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROTECTED_DATA_ROOT = os.path.normcase(os.path.abspath(_REPO_ROOT / "data"))
# Extra roots a test must never touch: the resolved target of data/ when it is a
# symlink (e.g. onto an external drive), plus any os.pathsep-separated paths in
# IVI_TEST_PROTECTED_ROOTS.
_EXTRA_PROTECTED_ROOTS = tuple(
    sorted(
        {
            os.path.normcase(os.path.abspath(root))
            for root in (
                *os.environ.get("IVI_TEST_PROTECTED_ROOTS", "").split(os.pathsep),
                str((_REPO_ROOT / "data").resolve()),
            )
            if root
        }
        - {_PROTECTED_DATA_ROOT}
    )
)
_CURRENT_ITEM: contextvars.ContextVar[pytest.Item | None] = contextvars.ContextVar(
    "ivi_current_pytest_item",
    default=None,
)
_EXPECTED_VIOLATION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ivi_expected_hermeticity_violation",
    default=False,
)
_SESSION_ACTIVE = False
# Recorded (non-expected) violations, keyed by the nodeid that triggered them.
# A test's teardown pops and surfaces its own key; anything still present at
# session end was blocked but never attributed to a failing test — a swallowed
# embargo hit (e.g. a HermeticityViolation caught by a broad ``except`` in
# production code, or a violation raised during collection with no owning test).
# ``pytest_sessionfinish`` fails the run on any such residue so a block can
# never pass silently.
_VIOLATIONS: dict[str, list[str]] = defaultdict(list)
_COUNTS: Counter[str] = Counter()

_ORIGINAL_OS_STAT = os.stat
_ORIGINAL_OS_LSTAT = os.lstat
_ORIGINAL_OS_ACCESS = os.access
_ORIGINAL_OS_READLINK = os.readlink

_PAID_PROVIDER_HOST_SUFFIXES = (
    "api.anthropic.com",
    "api.openai.com",
    "api.tavily.com",
    "eodhd.com",
)
_PAID_CREDENTIAL_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "VOE_OPENAI_API_KEY",
    "VOE_EODHD_APIKEY",
    "EODHD_API_KEY",
    "FRED_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "TAVILY_API_KEY",
    "PERPLEXITY_API_KEY",
)
_SAFE_GIT_SUBCOMMANDS = {"diff", "rev-parse", "status"}


def _nodeid() -> str:
    item = _CURRENT_ITEM.get()
    return item.nodeid if item is not None else "<collection>"


def _has_marker(name: str) -> bool:
    item = _CURRENT_ITEM.get()
    return item is not None and item.get_closest_marker(name) is not None


def _violate(kind: str, detail: str) -> None:
    message = f"Hermeticity embargo blocked {kind} in {_nodeid()}: {detail}"
    if os.environ.get("VOE_TEST_HERMETIC_TRACE") == "1":
        message += "\n" + "".join(traceback.format_stack(limit=24))
    if _EXPECTED_VIOLATION.get():
        raise HermeticityViolation(message)
    _COUNTS[f"{kind}_blocked"] += 1
    _VIOLATIONS[_nodeid()].append(message)
    raise HermeticityViolation(message)


def _lexical_path(value: object) -> str | None:
    if isinstance(value, int):
        return None
    try:
        raw = os.fsdecode(os.fspath(value))
    except TypeError:
        return None
    if raw.startswith("file:"):
        raw = raw[5:].split("?", 1)[0]
    if not raw:
        raw = os.getcwd()
    return os.path.normcase(os.path.abspath(raw))


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _protected_kind(value: object) -> tuple[str, str] | None:
    path = _lexical_path(value)
    if path is None:
        return None
    if any(_within(path, root) for root in _EXTRA_PROTECTED_ROOTS):
        return "live_data", path
    if _within(path, _PROTECTED_DATA_ROOT):
        outputs_root = os.path.join(_PROTECTED_DATA_ROOT, "outputs")
        return ("live_artifact" if _within(path, outputs_root) else "live_data"), path
    return None


def _write_open(mode: object, flags: object) -> bool:
    if isinstance(mode, str) and any(token in mode for token in ("w", "a", "x", "+")):
        return True
    if isinstance(flags, int):
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
        return bool(flags & write_flags)
    return False


def _guard_path(value: object, *, operation: str, write: bool = False) -> None:
    if not _SESSION_ACTIVE:
        return
    protected = _protected_kind(value)
    if protected is None:
        return
    kind, path = protected
    # Pytest probes ``<candidate>/pyvenv.cfg`` while deciding whether a
    # directory is a virtual environment.  This metadata-only collection
    # probe is internal to pytest and cannot expose owner data.
    if _CURRENT_ITEM.get() is None and operation == "os.stat":
        collection_probe = os.path.relpath(path, _PROTECTED_DATA_ROOT)
        if path == _PROTECTED_DATA_ROOT or collection_probe in {
            "pyvenv.cfg",
            os.path.join("conda-meta", "history"),
        }:
            return
    if write:
        _violate(kind, f"owner-path mutation is forbidden ({operation}: {path})")
    if kind == "live_artifact" and _has_marker("live_artifact"):
        _COUNTS["live_artifact_allowed"] += 1
        return
    if kind == "live_data" and _has_marker("live_data"):
        _COUNTS["live_data_allowed"] += 1
        return
    _violate(kind, f"{operation} requires an explicit @{kind} marker ({path})")


def _host_from_socket_event(event: str, args: tuple[object, ...]) -> str | None:
    if event == "socket.getaddrinfo" and args:
        return str(args[0]).strip().lower()
    if event in {"socket.gethostbyname", "socket.gethostbyname_ex"} and args:
        return str(args[0]).strip().lower()
    if event in {"socket.connect", "socket.connect_ex", "socket.sendto"} and len(args) > 1:
        address = args[1]
        if isinstance(address, tuple) and address:
            return str(address[0]).strip().lower()
    return None


def _guard_network(event: str, args: tuple[object, ...]) -> None:
    host = _host_from_socket_event(event, args)
    if host and any(
        host == suffix or host.endswith(f".{suffix}") for suffix in _PAID_PROVIDER_HOST_SUFFIXES
    ):
        _violate("network", f"paid-provider host is forbidden in every test lane ({host})")
    if _has_marker("network") or _has_marker("sec_live"):
        _COUNTS["network_allowed"] += 1
        return
    destination = host or repr(args[1] if len(args) > 1 else args)
    _violate("network", f"{event} to {destination} requires @network or @sec_live")


def _audit_hook(event: str, args: tuple[object, ...]) -> None:
    if not _SESSION_ACTIVE:
        return
    if event in {
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.connect",
        "socket.connect_ex",
        "socket.sendto",
    }:
        _guard_network(event, args)
        return
    if event == "open" and args:
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        _guard_path(args[0], operation="open", write=_write_open(mode, flags))
        return
    if event in {"os.listdir", "os.scandir", "os.chdir"} and args:
        _guard_path(args[0], operation=event)
        return
    if (
        event
        in {
            "os.remove",
            "os.rmdir",
            "os.mkdir",
            "os.rename",
            "os.replace",
            "os.chmod",
            "os.chown",
            "os.truncate",
            "os.utime",
            "os.symlink",
        }
        and args
    ):
        _guard_path(args[0], operation=event, write=True)
        if event in {"os.rename", "os.replace", "os.symlink"} and len(args) > 1:
            _guard_path(args[1], operation=event, write=True)
        return
    if event == "sqlite3.connect" and args:
        protected = _protected_kind(args[0])
        if protected is None:
            return
        raw = os.fsdecode(os.fspath(args[0]))
        readonly_uri = raw.startswith("file:") and "mode=ro" in raw.partition("?")[2].split("&")
        if protected[0] == "live_data" and _has_marker("live_data") and readonly_uri:
            _COUNTS["live_data_allowed"] += 1
            return
        _violate(
            "live_data",
            f"SQLite owner DB access requires @live_data and a file: URI with mode=ro ({protected[1]})",
        )
    if event in {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.posix_spawnp",
        "pty.spawn",
    }:
        if event == "subprocess.Popen" and _safe_git_subprocess(args):
            _COUNTS["safe_git_subprocess_allowed"] += 1
            return
        if _has_marker("subprocess"):
            _COUNTS["subprocess_allowed"] += 1
            return
        executable = args[0] if args else "unknown"
        _violate("subprocess", f"{event} requires an explicit @subprocess marker ({executable})")


def _safe_git_subprocess(args: tuple[object, ...]) -> bool:
    """Allow only the arena's reviewed, read-only git metadata probes."""

    if not args or os.path.basename(str(args[0])) != "git" or len(args) < 2:
        return False
    argv = args[1]
    if not isinstance(argv, (list, tuple)):
        return False
    words = [os.fsdecode(os.fspath(value)) for value in argv]
    if not words or os.path.basename(words[0]) != "git":
        return False
    subcommand = next((word for word in words[1:] if not word.startswith("-")), None)
    return subcommand in _SAFE_GIT_SUBCOMMANDS


sys.addaudithook(_audit_hook)


def _guarded_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
    _guard_path(path, operation="os.stat")
    return _ORIGINAL_OS_STAT(path, *args, **kwargs)


def _guarded_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
    _guard_path(path, operation="os.lstat")
    return _ORIGINAL_OS_LSTAT(path, *args, **kwargs)


def _guarded_access(path: object, mode: int, *args: object, **kwargs: object) -> bool:
    _guard_path(path, operation="os.access", write=bool(mode & os.W_OK))
    return _ORIGINAL_OS_ACCESS(path, mode, *args, **kwargs)


def _guarded_readlink(path: object, *args: object, **kwargs: object) -> str | bytes:
    _guard_path(path, operation="os.readlink")
    return _ORIGINAL_OS_READLINK(path, *args, **kwargs)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--order-seed",
        action="store",
        default=None,
        help="Deterministically hash-order collected tests with the supplied seed.",
    )


def pytest_configure(config: pytest.Config) -> None:
    del config
    global _SESSION_ACTIVE
    for name in _PAID_CREDENTIAL_NAMES:
        os.environ[name] = ""
    os.stat = _guarded_stat  # type: ignore[assignment]
    os.lstat = _guarded_lstat  # type: ignore[assignment]
    os.access = _guarded_access  # type: ignore[assignment]
    os.readlink = _guarded_readlink  # type: ignore[assignment]
    _SESSION_ACTIVE = True


def pytest_unconfigure(config: pytest.Config) -> None:
    del config
    global _SESSION_ACTIVE
    _SESSION_ACTIVE = False
    os.stat = _ORIGINAL_OS_STAT  # type: ignore[assignment]
    os.lstat = _ORIGINAL_OS_LSTAT  # type: ignore[assignment]
    os.access = _ORIGINAL_OS_ACCESS  # type: ignore[assignment]
    os.readlink = _ORIGINAL_OS_READLINK  # type: ignore[assignment]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    seed = config.getoption("--order-seed")
    if seed is None:
        return
    seed_text = str(seed)
    items.sort(key=lambda item: hashlib.sha256(f"{seed_text}\0{item.nodeid}".encode()).digest())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    del nextitem
    token = _CURRENT_ITEM.set(item)
    try:
        yield
    finally:
        _CURRENT_ITEM.reset(token)


@pytest.fixture(autouse=True)
def _isolate_test_runtime(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Route every unmarked test through its own writable runtime root.

    Relative production defaults such as ``data/cache`` resolve below the
    per-test directory, while absolute attempts to reach owner paths remain
    subject to the embargo. Explicit live-audit tests retain owner defaults.
    """

    from app.config import get_config

    get_config.cache_clear()
    live_audit = (
        request.node.get_closest_marker("live_data") is not None
        or request.node.get_closest_marker("live_artifact") is not None
    )
    if not live_audit:
        runtime_root = tmp_path.with_name(f"{tmp_path.name}-runtime")
        data_root = runtime_root / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        outputs_root = data_root / "outputs"
        integrity_roots = {
            "autonomous_sector": outputs_root / "runs" / "autonomous_sector",
            "analyst_output": outputs_root / "analyst_outputs",
            "scan": outputs_root / "scans",
            "research_output": outputs_root / "research",
            "watchlist_report": outputs_root / "digests",
        }
        for integrity_root in integrity_roots.values():
            integrity_root.mkdir(parents=True)
        integrity_sentinel = integrity_roots["autonomous_sector"] / "audited_test_fixture.json"
        integrity_sentinel.write_text("{}\n", encoding="utf-8")
        integrity_sentinel_bytes = integrity_sentinel.read_bytes()
        integrity_manifest = runtime_root / "financial_integrity_manifest.json"
        integrity_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "financial_integrity_audit_v1",
                    "audit_scope_id": "ivi_current_decision_artifacts_v1",
                    "generated_at": "2026-07-22T00:00:00Z",
                    "complete": True,
                    "source_roots": [
                        {
                            "family": family,
                            "root_id": {
                                "autonomous_sector": "autonomous_sector_runs",
                                "analyst_output": "analyst_outputs",
                                "scan": "scan_outputs",
                                "research_output": "research_outputs",
                                "watchlist_report": "watchlist_reports",
                            }[family],
                            "path": str(root.resolve()),
                        }
                        for family, root in integrity_roots.items()
                    ],
                    "summary": {
                        "artifacts_scanned": 1,
                        "tickers_scanned": 0,
                        "violations": 0,
                        "violations_by_invariant": {},
                        "affected_run_ids": 0,
                        "affected_tickers": 0,
                        "affected_run_id_values": [],
                        "affected_ticker_values": [],
                        "earliest_date": None,
                        "latest_date": None,
                        "llm_consumed_violation_count": 0,
                        "source_artifacts_rewritten": 0,
                    },
                    "invalid_run_ids": [],
                    "artifacts": [
                        {
                            "path": str(integrity_sentinel.resolve()),
                            "family": "autonomous_sector",
                            "sha256": hashlib.sha256(integrity_sentinel_bytes).hexdigest(),
                            "integrity_status": "PASS",
                            "decision_eligible": True,
                            "run_id": None,
                        }
                    ],
                    "violations": [],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("VOE_DATA_DIR", str(data_root))
        monkeypatch.setenv("VOE_DB_PATH", str(data_root / "engine.db"))
        # The shipped default price source is a live free provider; tests opt in explicitly.
        monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
        # VOE_NET_PROVIDER is the only network switch (the LLM provider no
        # longer implies offline). Tests that exercise network-enabled paths
        # set it to "enabled" explicitly and mock the transport.
        monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
        # Legacy unit tests explicitly model an already-audited decision book.
        # Missing/malformed-manifest behavior has a dedicated unpatched suite.
        monkeypatch.setenv(
            "VOE_FINANCIAL_INTEGRITY_MANIFEST",
            str(integrity_manifest),
        )
        needs_repo_cwd = request.node.get_closest_marker("subprocess") is not None
        if not needs_repo_cwd:
            monkeypatch.chdir(runtime_root)
        integrity_contract_tests = {
            "test_artifact_financial_audit.py",
            "test_classic_postwrite_authorization.py",
            "test_financial_integrity_coverage.py",
            "test_financial_integrity_manifest_fail_closed.py",
            "test_scorecard_exact_authorization.py",
            "test_sector_scan.py",
            "test_valuation_consumer_authorization.py",
            "test_watchlist_reevaluation.py",
            "test_web_financial_lineage.py",
        }
        is_integrity_contract = (
            Path(str(request.node.path)).name in integrity_contract_tests
            or request.node.get_closest_marker("financial_integrity_contract") is not None
        )
        if not is_integrity_contract:
            # The legacy suite's synthetic artifacts model an audited PASS
            # book. Contract-specific suites above exercise real manifest,
            # hash, stale-audit, and invalid-artifact behavior unpatched.
            for target in ("app.web.readmodel.reader.artifact_decision_eligibility",):
                monkeypatch.setattr(target, lambda _value: "PASS")
            # Many legacy helpers re-point VOE_DATA_DIR after this autouse
            # fixture constructs its coherent placeholder census.  Those
            # tests model an already-audited book, so keep the availability
            # predicate aligned with the exact PASS aliases below.  The
            # contract suites are excluded above and exercise real root
            # binding, census validation, stale hashes, and fail-closed state.
            #
            # Patch consumer aliases before their canonical definitions.
            # Several consumers import these functions by value.  If a
            # consumer is first imported while the canonical definition is
            # already patched, monkeypatch records the PASS shim as the
            # consumer's original value and leaks it after teardown.
            for target in (
                "app.analyst.output_store.financial_integrity_manifest_is_usable",
                "app.autonomous.sweep_delta.financial_integrity_manifest_is_usable",
                "app.watchlist.digest.financial_integrity_manifest_is_usable",
                "app.watchlist.store.financial_integrity_manifest_is_usable",
                "app.web.readmodel.company.financial_integrity_manifest_is_usable",
                "app.web.readmodel.company_depth.financial_integrity_manifest_is_usable",
                "app.web.readmodel.coverage.financial_integrity_manifest_is_usable",
                "app.web.readmodel.runs_index.financial_integrity_manifest_is_usable",
                "app.web.readmodel.today.financial_integrity_manifest_is_usable",
                "app.autonomous.artifact_financial_audit.financial_integrity_manifest_is_usable",
            ):
                monkeypatch.setattr(target, lambda *_args, **_kwargs: True)
            for target in (
                "app.web.readmodel.coverage.run_id_is_decision_eligible",
                "app.autonomous.artifact_financial_audit.run_id_is_decision_eligible",
            ):
                monkeypatch.setattr(target, lambda *_args, **_kwargs: True)
            for target in (
                "app.watchlist.dispositions.watchlist_row_is_decision_eligible",
                "app.watchlist.store.watchlist_row_is_decision_eligible",
                "app.web.readmodel.company.watchlist_row_is_decision_eligible",
                "app.web.readmodel.company_depth.watchlist_row_is_decision_eligible",
                "app.web.readmodel.run_detail.watchlist_row_is_decision_eligible",
                "app.watchlist.digest.watchlist_row_is_decision_eligible",
            ):
                monkeypatch.setattr(target, lambda *_args, **_kwargs: True)

            def legacy_test_terminal_dispositions(
                run_id,
                *,
                pipeline_version,
                conn=None,
            ):
                del pipeline_version
                if conn is None:
                    return None
                rows = conn.execute(
                    """
                    SELECT ticker, candidate_disposition
                    FROM sector_run_loaded_sets
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchall()
                return {
                    str(row["ticker"]).strip().upper(): str(row["candidate_disposition"] or "")
                    .strip()
                    .upper()
                    for row in rows
                    if str(row["ticker"] or "").strip()
                    and str(row["candidate_disposition"] or "").strip()
                }

            monkeypatch.setattr(
                "app.autonomous.sweep_delta._authorized_terminal_dispositions",
                legacy_test_terminal_dispositions,
            )
            for target in (
                "app.autonomous.sweep_delta.outcome_row_is_decision_eligible",
                "app.calibration.calibration_report.outcome_row_is_decision_eligible",
                "app.calibration.return_resolver.outcome_row_is_decision_eligible",
                "app.discovery.calibration.outcome_row_is_decision_eligible",
                "app.outcomes.store.outcome_row_is_decision_eligible",
                "app.web.readmodel.company_depth.outcome_row_is_decision_eligible",
                "app.web.readmodel.outcomes.outcome_row_is_decision_eligible",
            ):
                monkeypatch.setattr(target, lambda *_args, **_kwargs: True)
            for target in (
                "app.calibration.calibration_report.serialized_outcome_binding_is_decision_eligible",
                "app.discovery.calibration.serialized_outcome_binding_is_decision_eligible",
                "app.web.readmodel.outcomes.serialized_outcome_binding_is_decision_eligible",
            ):
                monkeypatch.setattr(target, lambda *_args, **_kwargs: True)

            def legacy_test_outcome_binding(row):
                return {
                    "schema_version": "ticker_outcome_report_binding_v1",
                    "outcome_state": {
                        "id": int(row["id"]),
                        "run_id": str(row["run_id"]),
                        "ticker": str(row["ticker"]),
                    },
                    "financial_integrity_fingerprint": "legacy-test-binding",
                }

            for target in (
                "app.calibration.calibration_report.serialize_outcome_binding",
                "app.discovery.calibration.serialize_outcome_binding",
            ):
                monkeypatch.setattr(target, legacy_test_outcome_binding)
            monkeypatch.setattr(
                "app.outcomes.store.refresh_outcome_integrity_fingerprint",
                lambda *_args, **_kwargs: True,
            )
            monkeypatch.setattr(
                "app.calibration.return_resolver.refresh_outcome_integrity_fingerprint",
                lambda *_args, **_kwargs: True,
            )
            for target in (
                "app.calibration.outcome_tracker.valuation_row_is_decision_eligible",
                "app.web.readmodel.company.valuation_row_is_decision_eligible",
                "app.web.readmodel.company_depth.valuation_row_is_decision_eligible",
                "app.web.readmodel.outcomes.valuation_row_is_decision_eligible",
                "app.valuation.lineage.valuation_row_is_decision_eligible",
            ):
                monkeypatch.setattr(
                    target,
                    lambda *_args, **_kwargs: True,
                )
            monkeypatch.setattr(
                "app.web.readmodel.reader.authorized_artifact_bytes",
                lambda path: ("PASS", Path(path).read_bytes()),
            )
            monkeypatch.setattr(
                "app.calibration.calibration_report.authorized_artifact_bytes",
                lambda path: ("PASS", Path(path).read_bytes()),
            )
            monkeypatch.setattr(
                "app.discovery.calibration.authorized_artifact_bytes",
                lambda path: ("PASS", Path(path).read_bytes()),
            )
            monkeypatch.setattr(
                "app.web.readmodel.outcomes.authorized_artifact_bytes",
                lambda path: ("PASS", Path(path).read_bytes()),
            )
            monkeypatch.setattr(
                "app.calibration.calibration_report.write_typed_financial_authorization",
                lambda *_args, **_kwargs: None,
            )
            monkeypatch.setattr(
                "app.discovery.calibration.write_typed_financial_authorization",
                lambda *_args, **_kwargs: None,
            )
            monkeypatch.setattr(
                "app.web.readmodel.runs_index.authorized_artifact_bytes",
                lambda path: ("PASS", Path(path).read_bytes()),
            )
            for target in (
                "app.analyst.output_store.authorized_artifact_bytes",
                "app.web.readmodel.run_detail.authorized_artifact_bytes",
                "app.web.readmodel.today.authorized_artifact_bytes",
            ):
                monkeypatch.setattr(
                    target,
                    lambda path, *_args: ("PASS", Path(path).read_bytes()),
                )
            monkeypatch.setattr(
                "app.watchlist.digest.authorized_watchlist_decision_binding",
                lambda row, *_args, **_kwargs: {
                    "source_run_id": str(dict(row).get("source_run_id") or "legacy_test"),
                    "source_artifact_path": str(integrity_sentinel.resolve()),
                    "source_artifact_sha256": hashlib.sha256(integrity_sentinel_bytes).hexdigest(),
                    "source_decision_fingerprint": "a" * 64,
                },
            )
        get_config.cache_clear()
    yield
    get_config.cache_clear()
    violations = _VIOLATIONS.pop(request.node.nodeid, [])
    if violations:
        pytest.fail("\n".join(dict.fromkeys(violations)), pytrace=False)


@pytest.fixture
def expect_hermeticity_violation() -> Iterator[None]:
    """Allow embargo self-tests to assert the immediate exception without tainting teardown."""

    token = _EXPECTED_VIOLATION.set(True)
    try:
        yield
    finally:
        _EXPECTED_VIOLATION.reset(token)


@pytest.fixture
def isolated_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Give a test a complete temp-rooted data/config surface without initializing a schema."""

    data_root = tmp_path / "isolated_data"
    data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOE_DATA_DIR", str(data_root))
    monkeypatch.setenv("VOE_DB_PATH", str(data_root / "engine.db"))
    monkeypatch.setenv("VOE_PRICE_PROVIDER", "disabled")
    monkeypatch.setenv("VOE_NET_PROVIDER", "disabled")
    from app.config import get_config

    get_config.cache_clear()
    yield data_root
    get_config.cache_clear()


def _residual_violations() -> dict[str, list[str]]:
    """Recorded violations that no test teardown ever surfaced."""

    return {nodeid: messages for nodeid, messages in _VIOLATIONS.items() if messages}


def pytest_terminal_summary(terminalreporter: Any) -> None:
    terminalreporter.write_sep(
        "-",
        "hermeticity " + " ".join(f"{key}={_COUNTS[key]}" for key in sorted(_COUNTS)),
    )
    residual = _residual_violations()
    if residual:
        terminalreporter.write_sep("-", "swallowed hermeticity violations", red=True)
        for nodeid, messages in residual.items():
            for message in dict.fromkeys(messages):
                terminalreporter.write_line(f"{nodeid}: {message}")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the session if any embargo violation was blocked but never surfaced.

    Runs first and lifts the embargo: every test has finished, and pytest's own
    temp-directory cleanup walks old ``pytest-N`` trees with dir_fd-relative
    opens (bare names such as ``data``) that would otherwise read as the repo's
    live ``data/`` directory.

    A per-test teardown only fails the run when the violation is attributed to
    that test's nodeid. A violation caught by production code (swallowed) or
    raised during collection would otherwise leave a residue here that no
    teardown pops, letting a real embargo hit pass silently. Convert any such
    residue into a hard non-zero exit.
    """

    global _SESSION_ACTIVE
    _SESSION_ACTIVE = False
    if _residual_violations() and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
