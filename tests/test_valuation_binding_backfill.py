from __future__ import annotations

import hashlib
import json
import sqlite3

from app.valuation.binding_backfill import backfill_valuation_bindings
from app.valuation.lineage import (
    valuation_row_is_decision_eligible,
    valuation_source_record,
)
from tests.test_web_financial_lineage import (
    _canonical_roots,
    _init,
    _insert_valuation,
    _install_manifest,
    _source_artifact,
)


def _record_for_legacy_row(row: sqlite3.Row, *, run_id: str) -> dict:
    values = {str(key): row[key] for key in row.keys()}
    values["source_run_id"] = run_id
    record = valuation_source_record(values)
    assert record is not None
    return record


def test_binding_backfill_only_binds_exact_sha_verified_artifact(monkeypatch, tmp_path):
    cfg = _init(monkeypatch, tmp_path)
    roots = _canonical_roots(tmp_path, autonomous_root=cfg.runs_dir)
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    good_id = _insert_valuation(
        conn,
        ticker="GOOD",
        as_of_date="2026-07-28",
        method="scorecard",
        value=80.0,
        source_path=None,
        source_run_id=None,
    )
    bad_id = _insert_valuation(
        conn,
        ticker="BAD",
        as_of_date="2026-07-28",
        method="scorecard",
        value=70.0,
        source_path=None,
        source_run_id=None,
    )
    conn.commit()
    good_before = conn.execute("SELECT * FROM valuations WHERE id = ?", (good_id,)).fetchone()
    bad_before = conn.execute("SELECT * FROM valuations WHERE id = ?", (bad_id,)).fetchone()
    assert good_before is not None
    assert bad_before is not None
    assert valuation_row_is_decision_eligible(good_before) is False

    good_path = _source_artifact(
        roots["autonomous_sector"],
        "run_good",
        "GOOD",
        valuation_source_records=[_record_for_legacy_row(good_before, run_id="run_good")],
    )
    bad_path = _source_artifact(
        roots["autonomous_sector"],
        "run_bad",
        "BAD",
        valuation_source_records=[_record_for_legacy_row(bad_before, run_id="run_bad")],
    )
    _install_manifest(
        monkeypatch,
        tmp_path,
        roots=roots,
        artifacts=[
            ("autonomous_sector", good_path, "run_good"),
            ("autonomous_sector", bad_path, "run_bad"),
        ],
    )
    bad_payload = json.loads(bad_path.read_text(encoding="utf-8"))
    bad_payload["post_audit_mutation"] = True
    bad_path.write_text(json.dumps(bad_payload, sort_keys=True), encoding="utf-8")
    bad_artifact_bytes = bad_path.read_bytes()
    bad_row_bytes = tuple(bad_before)

    dry_run = backfill_valuation_bindings(
        artifact_roots=[roots["autonomous_sector"]],
        apply=False,
        cfg=cfg,
    )
    assert dry_run == {
        "mode": "dry-run",
        "scanned": 2,
        "derivable": 1,
        "underivable": 1,
        "changed": 0,
        "reasons": {
            "DERIVABLE": 1,
            "NO_UNIQUE_VERIFIED_ARTIFACT_MATCH": 1,
        },
        "artifacts": {"scanned": 2, "unverified": 1, "verified": 1},
    }

    applied = backfill_valuation_bindings(
        artifact_roots=[roots["autonomous_sector"]],
        apply=True,
        cfg=cfg,
    )
    assert applied["changed"] == 1
    good_after = conn.execute("SELECT * FROM valuations WHERE id = ?", (good_id,)).fetchone()
    bad_after = conn.execute("SELECT * FROM valuations WHERE id = ?", (bad_id,)).fetchone()
    assert good_after is not None
    assert bad_after is not None
    assert good_after["source_run_id"] == "run_good"
    assert good_after["source_artifact_path"] == str(good_path.resolve())
    assert good_after["source_artifact_sha256"] == hashlib.sha256(good_path.read_bytes()).hexdigest()
    assert valuation_row_is_decision_eligible(good_after) is True
    assert tuple(bad_after) == bad_row_bytes
    assert bad_path.read_bytes() == bad_artifact_bytes
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM valuations_history WHERE source_id = ?", (good_id,)
        ).fetchone()[0]
        == 1
    )

    second_apply = backfill_valuation_bindings(
        artifact_roots=[roots["autonomous_sector"]],
        apply=True,
        cfg=cfg,
    )
    assert second_apply["scanned"] == 1
    assert second_apply["derivable"] == 0
    assert second_apply["underivable"] == 1
    assert second_apply["changed"] == 0
    assert tuple(conn.execute("SELECT * FROM valuations WHERE id = ?", (bad_id,)).fetchone()) == (
        bad_row_bytes
    )
    conn.close()
