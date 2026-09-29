"""Reader read model + /api/reader/*: library discovery, allowlisted render, TOC."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import init_db
from app.web.main import app
from app.web.readmodel import reader
from app.web.readmodel.md import render_markdown_with_toc

client = TestClient(app)


def _init_temp_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "engine.db"
    monkeypatch.setenv("VOE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VOE_DB_PATH", str(db_path))
    from app.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    init_db(cfg)
    return cfg


def _seed_library(cfg) -> dict[str, Path]:
    outputs = Path(cfg.outputs_dir)
    digests = outputs / "digests"
    digests.mkdir(parents=True, exist_ok=True)
    (digests / "digest_2026-07-20.md").write_text("# Digest\nolder", encoding="utf-8")
    (digests / "digest_2026-07-21.md").write_text("# Digest\nnewer", encoding="utf-8")
    (digests / "notes.md").write_text("not a digest", encoding="utf-8")

    dossier_dir = outputs / "dossiers" / "deep_scan_x" / "AAA"
    dossier_dir.mkdir(parents=True, exist_ok=True)
    dossier_md = dossier_dir / "dossier.md"
    dossier_md.write_text(
        "# AAA 10-K Dossier\n\n## Risks\none\n\n## Risks\ntwo\n", encoding="utf-8"
    )
    (dossier_dir / "dossier.json").write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "2025_revenue",
                        "label": "revenue::2025",
                        "value": 123.0,
                        "unit": "USD_millions",
                        "citations": [
                            {
                                "source_url": "https://example.test/facts",
                                "snippet": "revenue was 123",
                                "section_label": "financial_statements",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (outputs / "dossiers" / "deep_scan_x" / "peer_report.md").write_text(
        "# Peer Report\n", encoding="utf-8"
    )

    memo_dir = outputs / "memos" / "AAA_2026-05-01"
    memo_dir.mkdir(parents=True, exist_ok=True)
    (memo_dir / "memo.md").write_text("# AAA Memo\n", encoding="utf-8")

    return {"dossier": dossier_md, "digest": digests / "digest_2026-07-21.md"}


def test_render_markdown_with_toc_stamps_and_dedupes_anchors():
    html, toc = render_markdown_with_toc(
        "# Title\n\n## Risks\nfirst\n\n## Risks\nsecond\n\n### Fine Print!\n"
    )
    assert toc == [
        {"level": 1, "text": "Title", "anchor": "title"},
        {"level": 2, "text": "Risks", "anchor": "risks"},
        {"level": 2, "text": "Risks", "anchor": "risks-2"},
        {"level": 3, "text": "Fine Print!", "anchor": "fine-print"},
    ]
    assert '<h1 id="title">Title</h1>' in html
    assert '<h2 id="risks-2">Risks</h2>' in html


def test_library_families_totals_and_meta(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    _seed_library(cfg)
    payload = reader.library(None)
    families = {f["family"]: f for f in payload["families"]}
    assert list(families) == ["digest", "dossier", "peer_report", "memo"]
    assert families["digest"]["total"] == 2
    assert families["digest"]["label"] == "Daily digests"
    assert [i["meta"]["date"] for i in families["digest"]["items"]] == [
        "2026-07-21",
        "2026-07-20",
    ]
    dossier_item = families["dossier"]["items"][0]
    assert dossier_item["title"] == "AAA 10-K dossier"
    assert dossier_item["meta"] == {
        "ticker": "AAA",
        "run_label": "deep_scan_x",
        "has_claims": True,
    }
    assert families["memo"]["items"][0]["title"] == "AAA memo · 2026-05-01"


def test_render_artifact_returns_toc_claims_and_title(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    seeded = _seed_library(cfg)
    payload = reader.render_artifact(str(seeded["dossier"]))
    assert payload["title"] == "AAA 10-K Dossier"
    assert payload["toc"][0] == {
        "level": 1,
        "text": "AAA 10-K Dossier",
        "anchor": "aaa-10-k-dossier",
    }
    assert payload["claims"] == [
        {
            "claim_id": "2025_revenue",
            "label": "revenue::2025",
            "value": 123.0,
            "unit": "USD_millions",
            "citations": [
                {
                    "source_url": "https://example.test/facts",
                    "snippet": "revenue was 123",
                    "section_label": "financial_statements",
                }
            ],
        }
    ]
    digest = reader.render_artifact(str(seeded["digest"]))
    assert digest["claims"] is None
    assert "<h1" in digest["html"]


def test_render_artifact_suppresses_invalid_decision_content(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    seeded = _seed_library(cfg)
    monkeypatch.setattr(
        reader,
        "authorized_artifact_bytes",
        lambda _path: ("INVALID", None),
    )

    payload = reader.render_artifact(str(seeded["dossier"]))

    assert payload["integrity_status"] == "INVALID"
    assert payload["decision_eligible"] is False
    assert payload["claims"] is None
    assert "excluded from current decisions" in payload["html"]
    assert "original artifact body and claims are suppressed" in payload["html"]
    assert "AAA 10-K Dossier" not in payload["html"]
    assert "revenue::2025" not in payload["html"]


def test_render_artifact_refusals(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    seeded = _seed_library(cfg)
    outputs = Path(cfg.outputs_dir)

    with pytest.raises(reader.ArtifactRefused):
        reader.render_artifact("/etc/passwd")
    with pytest.raises(reader.ArtifactRefused):
        reader.render_artifact(str(outputs / ".." / "engine.db"))
    with pytest.raises(reader.ArtifactRefused):
        reader.render_artifact(str(outputs / "digests" / "missing.md"))
    with pytest.raises(reader.ArtifactRefused):
        reader.render_artifact(str(outputs / "digests"))
    with pytest.raises(reader.ArtifactRefused):
        reader.render_artifact("")
    not_md = outputs / "digests" / "raw.txt"
    not_md.write_text("text", encoding="utf-8")
    with pytest.raises(reader.ArtifactRefused):
        reader.render_artifact(str(not_md))
    # A markdown path that escapes outputs via traversal inside the string.
    with pytest.raises(reader.ArtifactRefused):
        reader.render_artifact(str(outputs / "digests" / ".." / ".." / ".." / "x.md"))
    assert reader.render_artifact(str(seeded["digest"]))["title"] == "Digest"


def test_api_reader_endpoints(monkeypatch, tmp_path):
    cfg = _init_temp_env(monkeypatch, tmp_path)
    seeded = _seed_library(cfg)

    library = client.get("/api/reader/library")
    assert library.status_code == 200
    families = {f["family"]: f["total"] for f in library.json()["families"]}
    assert families == {"digest": 2, "dossier": 1, "peer_report": 1, "memo": 1}

    artifact = client.get("/api/reader/artifact", params={"path": str(seeded["dossier"])})
    assert artifact.status_code == 200
    assert artifact.json()["title"] == "AAA 10-K Dossier"
    assert len(artifact.json()["claims"]) == 1

    refused = client.get("/api/reader/artifact", params={"path": "/etc/passwd"})
    assert refused.status_code == 404
