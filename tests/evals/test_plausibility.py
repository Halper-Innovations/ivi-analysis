from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.config import get_config
from tests.evals.eval_utils import RUNS_ROOT, load_json


pytestmark = pytest.mark.eval_gate


ACTIONABLE_VERDICTS = {"ACTIONABLE", "SELECTED"}
PLAUSIBILITY_CAP = "SUSPICIOUS_MAGNITUDE_DCF"
PLAUSIBILITY_RATIO_LIMIT = 2.5


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_verdict(value: Any) -> str | None:
    if not value:
        return None
    normalized = str(value).upper().replace("-", "_").replace(" ", "_")
    if normalized in {"ACTIONABLE", "SELECTED", "WATCHLIST_ONLY", "AVOID"}:
        return normalized
    return None


def _caps_contain_suspicious(caps: Any) -> bool:
    if isinstance(caps, str):
        return PLAUSIBILITY_CAP in caps
    if isinstance(caps, list):
        return any(str(item) == PLAUSIBILITY_CAP for item in caps)
    return False


def _packet_anchor_and_price(packet: dict[str, Any]) -> tuple[float | None, float | None]:
    valuation = packet.get("valuation") if isinstance(packet.get("valuation"), dict) else {}
    anchor = _as_float(
        valuation.get("generic_anchor_value")
        or valuation.get("valuation_anchor_value")
        or valuation.get("anchor_value")
        or valuation.get("dcf_value")
    )
    price = _as_float(packet.get("current_price") or valuation.get("current_price"))
    return anchor, price


def _ranking_by_ticker(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["ticker"]).upper(): row
        for row in data.get("relative_ranking") or []
        if isinstance(row, dict) and row.get("ticker")
    }


def _packet_by_ticker(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(packet["ticker"]).upper(): packet
        for packet in data.get("company_packets") or []
        if isinstance(packet, dict) and packet.get("ticker")
    }


def _artifact_has_plausibility_guard(data: dict[str, Any], ticker: str) -> bool:
    normalized_ticker = ticker.upper()
    ranking = _ranking_by_ticker(data).get(normalized_ticker, {})
    packet = _packet_by_ticker(data).get(normalized_ticker, {})
    business_quality = packet.get("business_quality") if isinstance(packet.get("business_quality"), dict) else {}
    selection_audit = data.get("selection_audit") if isinstance(data.get("selection_audit"), dict) else {}
    selected_ticker = str(data.get("selected_ticker") or "").upper()

    has_cap = (
        _caps_contain_suspicious(ranking.get("confidence_caps"))
        or _caps_contain_suspicious(ranking.get("confidence_cap_reasons"))
        or _caps_contain_suspicious(packet.get("confidence_caps"))
        or _caps_contain_suspicious(packet.get("confidence_cap_reasons"))
    )
    low_confidence = (
        ranking.get("company_autonomy_confidence") == "LOW"
        or ranking.get("confidence_ceiling") == "LOW"
        or business_quality.get("confidence_class") == "LOW"
    )
    if normalized_ticker == selected_ticker:
        has_cap = has_cap or _caps_contain_suspicious(selection_audit.get("confidence_caps"))
        low_confidence = low_confidence or selection_audit.get("confidence_ceiling") == "LOW"
    return has_cap or low_confidence


def plausibility_failures_for_artifact(
    data: dict[str, Any],
    *,
    path: Path = Path("synthetic_artifact.json"),
) -> list[str]:
    ranking = _ranking_by_ticker(data)
    selected_ticker = str(data.get("selected_ticker") or "").upper()
    final_verdict = _normalize_verdict(data.get("final_verdict"))
    failures: list[str] = []

    for ticker, packet in sorted(_packet_by_ticker(data).items()):
        row = ranking.get(ticker, {})
        verdict = _normalize_verdict(
            row.get("conviction_grade")
            or row.get("company_autonomy_verdict")
            or row.get("verdict")
            or row.get("final_verdict")
        )
        if ticker == selected_ticker and not verdict and final_verdict in ACTIONABLE_VERDICTS:
            verdict = final_verdict
        if verdict not in ACTIONABLE_VERDICTS:
            continue

        anchor, price = _packet_anchor_and_price(packet)
        if anchor is None or price is None or price <= 0:
            continue
        ratio = anchor / price
        if ratio <= PLAUSIBILITY_RATIO_LIMIT:
            continue
        if not _artifact_has_plausibility_guard(data, ticker):
            failures.append(
                f"{path}: {ticker} has {ratio:.2f}x valuation_anchor/current_price and {verdict} "
                f"without {PLAUSIBILITY_CAP} or LOW confidence"
            )
    return failures


def _load_source_artifact(
    source_run_id: str,
    *,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Any] | None:
    path = runs_root / "autonomous_sector" / source_run_id / "autonomous_sector_run.json"
    if not path.exists():
        return None
    return load_json(path)


def plausibility_failures_for_watchlist(
    *,
    db_path: Path | str | None = None,
    runs_root: Path = RUNS_ROOT,
) -> list[str]:
    conn = sqlite3.connect(str(db_path or get_config().db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM watchlist
            WHERE status != 'REMOVED'
              AND conviction_grade IN ('ACTIONABLE', 'SELECTED')
              AND valuation_anchor_value IS NOT NULL
              AND current_price_at_addition IS NOT NULL
              AND current_price_at_addition > 0
            """
        ).fetchall()
    finally:
        conn.close()

    failures: list[str] = []
    for row in rows:
        ratio = float(row["valuation_anchor_value"]) / float(row["current_price_at_addition"])
        if ratio <= PLAUSIBILITY_RATIO_LIMIT:
            continue
        reason = str(row["status_reason"] or "")
        has_db_guard = PLAUSIBILITY_CAP in reason or "confidence=LOW" in reason
        artifact = _load_source_artifact(
            str(row["source_run_id"]),
            runs_root=runs_root,
        )
        has_artifact_guard = bool(artifact and _artifact_has_plausibility_guard(artifact, str(row["ticker"])))
        if not (has_db_guard or has_artifact_guard):
            failures.append(
                f"watchlist:{row['ticker']} id={row['id']} has {ratio:.2f}x valuation_anchor/current_price_at_addition "
                f"and {row['conviction_grade']} without {PLAUSIBILITY_CAP} or LOW confidence"
            )
    return failures


def test_watchlist_plausibility_magnitude(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    get_config.cache_clear()

    from app.watchlist.schema import ensure_watchlist_schema

    ensure_watchlist_schema(db_path)

    # Seed one ACTIONABLE row with ratio <= PLAUSIBILITY_RATIO_LIMIT (100/80 = 1.25x).
    # This exercises the query path and confirms a clean row produces no failures.
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO watchlist (
            ticker, status, conviction_grade, valuation_anchor_value,
            current_price_at_addition, status_reason, source_run_id, added_at
        ) VALUES ('TSTX', 'ACTIVE', 'ACTIONABLE', 100.0, 80.0, NULL, 'ci_seed_run', '2026-01-01T00:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()

    failures = plausibility_failures_for_watchlist(
        db_path=db_path,
        runs_root=tmp_path / "runs",
    )

    assert not failures, "\n".join(failures)


def test_plausibility_accepts_large_anchor_when_low_confidence() -> None:
    data = {
        "relative_ranking": [
            {
                "ticker": "GPOR",
                "company_autonomy_verdict": "ACTIONABLE",
                "company_autonomy_confidence": "LOW",
            }
        ],
        "company_packets": [
            {
                "ticker": "GPOR",
                "current_price": 100.0,
                "valuation": {"generic_anchor_value": 300.0, "current_price": 100.0},
                "confidence_caps": [],
            }
        ],
    }

    assert plausibility_failures_for_artifact(data) == []


def test_plausibility_flags_actionable_large_anchor_without_cap() -> None:
    data = {
        "relative_ranking": [{"ticker": "GPOR", "company_autonomy_verdict": "ACTIONABLE"}],
        "company_packets": [
            {
                "ticker": "GPOR",
                "current_price": 100.0,
                "valuation": {"generic_anchor_value": 300.0, "current_price": 100.0},
                "confidence_caps": [],
            }
        ],
    }

    failures = plausibility_failures_for_artifact(data)

    assert len(failures) == 1
    assert "3.00x" in failures[0]
    assert PLAUSIBILITY_CAP in failures[0]


def test_plausibility_flags_watchlist_row_without_guard(tmp_path: Path) -> None:
    db_path = tmp_path / "watchlist.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE watchlist (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            status TEXT,
            conviction_grade TEXT,
            valuation_anchor_value REAL,
            current_price_at_addition REAL,
            status_reason TEXT,
            source_run_id TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO watchlist (
            id, ticker, status, conviction_grade, valuation_anchor_value,
            current_price_at_addition, status_reason, source_run_id
        ) VALUES (1, 'GPOR', 'ACTIVE', 'ACTIONABLE', 300.0, 100.0, NULL, 'synthetic')
        """
    )
    conn.commit()
    conn.close()

    failures = plausibility_failures_for_watchlist(
        db_path=db_path,
        runs_root=tmp_path / "runs",
    )

    assert len(failures) == 1
    assert "watchlist:GPOR" in failures[0]
