from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.autonomous.sweep_delta import split_unswept, swept_tickers_for_band
from app.web.readmodel.coverage import coverage_atlas, coverage_cell
from app.web.readmodel.runs_index import UI_SCHEMA_SQL
from app.web.readmodel.sweeps import sweep_rollups

# The acceptance contract: the atlas must agree with `ivi sweep-delta-report`
# on a fixture ledger. The ledger is built through the production writer
# (record_loaded_set) so schema and normalization cannot drift, and parity is
# asserted against the exact helpers the delta report runs.


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _seed_ledger(conn: sqlite3.Connection) -> None:
    from app.autonomous.sweep_delta import record_loaded_set

    # v1 sweep coverage in two sectors of one band.
    record_loaded_set(
        conn,
        run_id="autonomous_sector_biotech_20260101_aaa111",
        sector="biotech",
        market_cap_focus="mid_cap",
        source="sector_scan_db",
        tickers=["AAA", "BBB"],
        loaded_at="2026-01-01T10:00:00+00:00",
        pipeline_version="v1",
        candidate_dispositions={
            "AAA": "LLM_CANDIDATE_REVIEW_COMPLETED",
            "BBB": "LLM_CANDIDATE_REVIEW_COMPLETED",
        },
    )
    record_loaded_set(
        conn,
        run_id="autonomous_sector_utilities_20260102_bbb222",
        sector="utilities",
        market_cap_focus="mid_cap",
        source="sector_scan_db",
        tickers=["UUU"],
        loaded_at="2026-01-02T10:00:00+00:00",
        pipeline_version="v1",
        candidate_dispositions={"UUU": "LLM_CANDIDATE_REVIEW_COMPLETED"},
    )
    # A probe run: never coverage, in any view.
    record_loaded_set(
        conn,
        run_id="autonomous_probe_20260103_ccc333",
        sector="biotech",
        market_cap_focus="mid_cap",
        source="explicit_tickers",
        tickers=["PROBE"],
        loaded_at="2026-01-03T10:00:00+00:00",
        pipeline_version="v1",
        candidate_dispositions={"PROBE": "LLM_CANDIDATE_REVIEW_COMPLETED"},
    )
    # v2 coverage: CMP completes, REOPEN completes then reopens (NEEDS_DATA
    # in a later run) — the delta report treats REOPEN as unswept again.
    record_loaded_set(
        conn,
        run_id="all_sector_v2_20260110_utilities",
        sector="utilities",
        market_cap_focus="large_and_mega",
        source="sector_scan_db",
        tickers=["CMP", "REOPEN"],
        loaded_at="2026-01-10T10:00:00+00:00",
        pipeline_version="v2",
        candidate_dispositions={"CMP": "UNDERWRITTEN", "REOPEN": "SCREENED_OUT"},
    )
    record_loaded_set(
        conn,
        run_id="all_sector_v2_20260111_utilities",
        sector="utilities",
        market_cap_focus="large_and_mega",
        source="sector_scan_db",
        tickers=["REOPEN"],
        loaded_at="2026-01-11T10:00:00+00:00",
        pipeline_version="v2",
        candidate_dispositions={"REOPEN": "NEEDS_DATA"},
    )
    # Mixed-generation cell: v1 and v2 rows for the same (sector, band).
    record_loaded_set(
        conn,
        run_id="autonomous_sector_energy_20260104_ddd444",
        sector="energy",
        market_cap_focus="small_cap",
        source="sector_scan_db",
        tickers=["EEE", "SHARED"],
        loaded_at="2026-01-04T10:00:00+00:00",
        pipeline_version="v1",
        candidate_dispositions={
            "EEE": "LLM_CANDIDATE_REVIEW_COMPLETED",
            "SHARED": "LLM_CANDIDATE_REVIEW_COMPLETED",
        },
    )
    record_loaded_set(
        conn,
        run_id="all_sector_v2_20260112_energy",
        sector="energy",
        market_cap_focus="small_cap",
        source="sector_scan_db",
        tickers=["SHARED", "FFF"],
        loaded_at="2026-01-12T10:00:00+00:00",
        pipeline_version="v2",
        candidate_dispositions={"SHARED": "UNDERWRITTEN", "FFF": "OUT_OF_SCOPE"},
    )
    # Legacy loader-only rows stay auditable but never inflate the atlas.
    record_loaded_set(
        conn,
        run_id="autonomous_sector_biotech_legacy_loader_only",
        sector="biotech",
        market_cap_focus="mid_cap",
        source="sector_scan_db",
        tickers=["LEGACY"],
        loaded_at="2026-01-13T10:00:00+00:00",
        pipeline_version="v1",
    )


def _cells_by_key(atlas: dict) -> dict[tuple[str, str], dict]:
    return {(cell["sector"], cell["band"]): cell for cell in atlas["cells"]}


def test_atlas_v1_cells_match_delta_report_band_coverage():
    conn = _conn()
    _seed_ledger(conn)
    atlas = coverage_atlas(conn)
    cells = _cells_by_key(atlas)

    assert cells[("biotech", "mid_cap")]["tickers"] == 2
    assert cells[("utilities", "mid_cap")]["tickers"] == 1
    # The union of the band's cells is exactly the set the delta report
    # subtracts as "swept" for that band.
    assert swept_tickers_for_band(conn, "mid_cap", pipeline_version="v1") == {
        "AAA",
        "BBB",
        "UUU",
    }
    # The probe never surfaces: no cell counts it, no extra cell exists.
    assert cells[("biotech", "mid_cap")]["run_count"] == 1
    assert ("biotech", "mid_cap") in cells
    band_cells = [c for c in atlas["cells"] if c["band"] == "mid_cap"]
    assert sum(c["tickers"] for c in band_cells) == 3


def test_atlas_v2_cell_matches_sector_scoped_delta_semantics():
    conn = _conn()
    _seed_ledger(conn)
    atlas = coverage_atlas(conn)
    cells = _cells_by_key(atlas)

    expected = swept_tickers_for_band(
        conn, "large_and_mega", "utilities", pipeline_version="v2"
    )
    assert expected == {"CMP"}  # REOPEN's latest state is NEEDS_DATA
    cell = cells[("utilities", "large_and_mega")]
    assert cell["tickers"] == 1
    assert cell["pipelines"] == ["v2"]
    # The later NEEDS_DATA row reopens REOPEN and is not a covering run.
    assert cell["run_count"] == 1
    assert cell["last_loaded_at"] == "2026-01-10T10:00:00+00:00"

    # And split_unswept (the delta report's core) agrees end to end.
    split = split_unswept(
        conn,
        band="large_and_mega",
        sector="utilities",
        selected_tickers=["CMP", "REOPEN", "NEWCO"],
        pipeline_version="v2",
    )
    assert split["swept"] == ["CMP"]
    assert split["unswept"] == ["REOPEN", "NEWCO"]


def test_atlas_mixed_generation_cell_counts_distinct_union():
    conn = _conn()
    _seed_ledger(conn)
    cells = _cells_by_key(coverage_atlas(conn))
    cell = cells[("energy", "small_cap")]
    # v1 {EEE, SHARED} ∪ v2-complete {SHARED} — FFF's latest state is
    # OUT_OF_SCOPE which IS coverage-complete, so {EEE, SHARED, FFF}.
    assert cell["tickers"] == 3
    assert cell["pipelines"] == ["v1", "v2"]
    assert cell["run_count"] == 2
    assert cell["last_loaded_at"] == "2026-01-12T10:00:00+00:00"


def test_atlas_axes_are_canonical_plus_observed():
    conn = _conn()
    _seed_ledger(conn)
    atlas = coverage_atlas(conn)
    assert atlas["bands"] == ["small_cap", "mid_cap", "large_and_mega"]
    # Canonical sectors lead in canonical order; all three observed are canonical.
    assert atlas["sectors"][:3] == ["aerospace_defense", "automotive", "biotech"]
    assert "energy" in atlas["sectors"]
    assert len(atlas["sectors"]) == 34


def test_atlas_without_ledger_table_is_empty_not_error():
    conn = _conn()
    atlas = coverage_atlas(conn)
    assert atlas["cells"] == []
    assert atlas["bands"] == []
    assert len(atlas["sectors"]) == 34


def test_coverage_cell_lists_swept_tickers_and_runs():
    conn = _conn()
    _seed_ledger(conn)
    ui_conn = sqlite3.connect(":memory:")
    ui_conn.row_factory = sqlite3.Row
    ui_conn.executescript(UI_SCHEMA_SQL)
    ui_conn.execute(
        "INSERT INTO run_index (path, slug, mtime_ns, size, indexed_at, run_id)"
        " VALUES ('p1', 'autonomous_sector/autonomous_sector_energy_20260104_ddd444',"
        " 0, 0, 'now', 'autonomous_sector_energy_20260104_ddd444')"
    )
    cell = coverage_cell(conn, sector="energy", band="small_cap", ui_conn=ui_conn)
    assert cell["tickers"] == ["EEE", "FFF", "SHARED"]
    assert [run["run_id"] for run in cell["runs"]] == [
        "all_sector_v2_20260112_energy",
        "autonomous_sector_energy_20260104_ddd444",
    ]
    v1_run = cell["runs"][1]
    assert v1_run["slug"] == "autonomous_sector/autonomous_sector_energy_20260104_ddd444"
    assert v1_run["tickers"] == 2
    assert cell["runs"][0]["slug"] is None


def test_coverage_cell_excludes_probe_runs():
    conn = _conn()
    _seed_ledger(conn)
    cell = coverage_cell(conn, sector="biotech", band="mid_cap")
    assert cell["tickers"] == ["AAA", "BBB"]
    assert [run["run_id"] for run in cell["runs"]] == [
        "autonomous_sector_biotech_20260101_aaa111"
    ]


# ------------------------------------------------------------- sweep roll-ups


def _ui_conn_with_runs(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(UI_SCHEMA_SQL)
    rows = [
        # Two mid_cap runs in ISO week 2026-W23 (June 1 = Monday).
        ("p1", "s1", "r_mid_a", "biotech", "mid_cap", "normal", "v1",
         "NO_SELECTION", 500000, "2026-06-01T10:00:00+00:00", None, 1, None),
        ("p2", "s2", "r_mid_b", "utilities", "mid_cap", "normal", "v1",
         "WATCHLIST", None, "2026-06-03T10:00:00+00:00", None, 1, None),
        # micro_cap the following week, dated only by as_of_date.
        ("p3", "s3", "r_micro", "energy", "micro_cap", "normal", "v1",
         "SELECTED", 250000, None, "2026-06-08", 1, None),
        # Duplicate embedded run_id across two artifacts, same group.
        ("p4", "s4", "r_dup", "hospitality_gaming", "micro_cap", "normal", "v2",
         "NO_SELECTION", None, None, "2026-06-08", 1, None),
        ("p5", "s5", "r_dup", "hospitality_gaming", "micro_cap", "normal", "v2",
         "NO_SELECTION", None, None, "2026-06-08", 1, None),
        # Integrity-invalid history: never contributes to current sweep totals.
        ("p_invalid", "s_invalid", "r_invalid", "energy", "micro_cap", "normal", "v2",
         "SELECTED", 900000, None, "2026-06-08", 0, None),
        # Quarantined artifact: excluded from roll-ups entirely.
        ("p6", "s6", "r_broken", None, None, None, None,
         None, None, None, None, 0, "JSONDecodeError: boom"),
    ]
    stored_rows = []
    for (
        path_name,
        slug,
        run_id,
        sector,
        band,
        family,
        pipeline,
        verdict,
        cost,
        created,
        as_of,
        eligible,
        error,
    ) in rows:
        artifact_path = tmp_path / path_name
        if eligible:
            payload = {
                "run_id": run_id,
                "sector": sector,
                "market_cap_focus": band,
                "scan_family": family,
                "pipeline_version": pipeline,
                "final_verdict": verdict,
                "created_at": created,
                "as_of_date": as_of,
                "status": "COMPLETED",
            }
            if cost is not None:
                payload["lane_usage"] = {"aggregate": {"cost_microdollars": cost}}
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            # Exact-current authorization reads the current artifact bytes;
            # malformed current bytes must stay excluded regardless of the
            # stale summary columns cached below.
            artifact_path.write_text("{not-json", encoding="utf-8")
        stored_rows.append(
            (
                str(artifact_path),
                slug,
                run_id,
                sector,
                band,
                family,
                pipeline,
                verdict,
                cost,
                created,
                as_of,
                eligible,
                error,
            )
        )
    conn.executemany(
        """
        INSERT INTO run_index (
            path, slug, mtime_ns, size, indexed_at, run_id, sector,
            market_cap_focus, scan_family, pipeline_version, final_verdict,
            cost_microdollars, created_at, as_of_date, decision_eligible, parse_error
        ) VALUES (?, ?, 0, 0, 'now', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (path, slug, run_id, sector, band, family, pipeline, verdict,
             cost, created, as_of, eligible, error)
            for (path, slug, run_id, sector, band, family, pipeline, verdict,
                 cost, created, as_of, eligible, error) in stored_rows
        ],
    )
    return conn


def _engine_with_watchlist() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE watchlist (id INTEGER PRIMARY KEY, ticker TEXT, source_run_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO watchlist(ticker, source_run_id) VALUES(?, ?)",
        [
            ("AAA", "r_mid_b"),
            ("BBB", "r_mid_b"),
            ("CCC", "r_dup"),
        ],
    )
    return conn


def test_sweep_rollups_group_by_band_and_week(tmp_path):
    rollups = sweep_rollups(
        _ui_conn_with_runs(tmp_path),
        engine_conn=_engine_with_watchlist(),
    )
    assert [(g["week"], g["band"]) for g in rollups] == [
        ("2026-W24", "micro_cap"),
        ("2026-W23", "mid_cap"),
    ]
    micro = rollups[0]
    assert micro["week_start"] == "2026-06-08"
    assert micro["week_end"] == "2026-06-14"
    assert micro["runs"] == 3
    assert micro["sectors"] == ["energy", "hospitality_gaming"]
    assert micro["pipelines"] == ["v1", "v2"]
    assert micro["verdicts"] == {"NO_SELECTION": 2, "SELECTED": 1}
    assert micro["cost_microdollars"] == 250000
    assert micro["cost_usd"] == 0.25
    assert micro["cost_known_runs"] == 1
    # r_dup appears twice but its watchlist rows count once.
    assert micro["watchlist_rows"] == 1

    mid = rollups[1]
    assert mid["runs"] == 2
    assert mid["sectors"] == ["biotech", "utilities"]
    assert mid["verdicts"] == {"NO_SELECTION": 1, "WATCHLIST": 1}
    assert mid["cost_microdollars"] == 500000
    assert mid["watchlist_rows"] == 2


def test_sweep_rollups_without_engine_still_group(tmp_path):
    rollups = sweep_rollups(_ui_conn_with_runs(tmp_path), engine_conn=None)
    assert [g["watchlist_rows"] for g in rollups] == [0, 0]
