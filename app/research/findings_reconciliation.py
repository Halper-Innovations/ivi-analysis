"""Findings reconciliation — Stage C of the parallel hybrid pipeline.

Projects deterministic pipeline outputs into a normalized shape, matches
against LLM analyst notes, produces agreement metrics.

Public API: reconcile_findings(analyst_notes, hypotheses, evidence_results)
            — owns projection internally; callers pass raw pipeline outputs.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.research.analyst_notes import AnalystNote, AnalystNotes
from app.research.hypothesis_generator import Hypothesis
from app.research.evidence_searcher import EvidenceResult


# ---------------------------------------------------------------------------
# Content Word Extraction
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the of in for by on to and or is are was were be with from at as "
    "this that it its not no has have had been will would could should may "
    "can do does did than more also into over such".split()
)

_SECTION_LABELS = frozenset({"mda", "risk_factors", "fin_notes", "business", "other"})


def extract_content_words(text: str) -> set[str]:
    """Extract content words: lowercase, alphanumeric, length >= 4, no stopwords, no section labels."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {
        t for t in tokens
        if len(t) >= 4 and t not in _STOPWORDS and t not in _SECTION_LABELS
    }


# ---------------------------------------------------------------------------
# Pipeline Finding (normalized shape)
# ---------------------------------------------------------------------------

@dataclass
class PipelineFinding:
    """Normalized finding from the deterministic pipeline."""
    claim: str
    direction: str         # BULLISH / BEARISH / NEUTRAL
    section: str | None    # most frequent section from CONFIRMS citations
    content_words: set[str]
    source_hypothesis: str
    evidence_status: str   # CONFIRMED / PARTIALLY_CONFIRMED


_PROJECTABLE_STATUSES = frozenset({"CONFIRMED", "PARTIALLY_CONFIRMED"})


def _collapse_section(evidence_result: EvidenceResult) -> str | None:
    """Derive section from confirmed citations.

    Rule: most frequent section among CONFIRMS-status items' citations.
    Tie-break by first citation order. Item-level statuses are
    CONFIRMS / CONTRADICTS / INCONCLUSIVE / NOT_FOUND / UNCLASSIFIED.
    """
    section_counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    idx = 0
    for item in evidence_result.evidence_item_results:
        # PARTIALLY_CONFIRMED is a hypothesis-level status, not item-level.
        # Item-level confirming status is only "CONFIRMS".
        if item.status != "CONFIRMS":
            continue
        for citation in item.citations:
            section = citation.section
            section_counts[section] += 1
            if section not in first_seen:
                first_seen[section] = idx
            idx += 1

    if not section_counts:
        return None

    return min(
        section_counts,
        key=lambda s: (-section_counts[s], first_seen.get(s, 0)),
    )


def project_pipeline_findings(
    hypotheses: list[Hypothesis],
    evidence_results: list[EvidenceResult],
) -> list[PipelineFinding]:
    """Project deterministic pipeline outputs into normalized PipelineFindings.

    Only includes hypotheses with CONFIRMED or PARTIALLY_CONFIRMED status
    and at least one citation-backed section.
    """
    findings: list[PipelineFinding] = []

    for er in evidence_results:
        if er.hypothesis_status not in _PROJECTABLE_STATUSES:
            continue

        section = _collapse_section(er)
        if section is None:
            continue

        hyp = er.hypothesis
        findings.append(PipelineFinding(
            claim=hyp.claim,
            direction=hyp.direction,
            section=section,
            content_words=extract_content_words(hyp.claim),
            source_hypothesis=hyp.source,
            evidence_status=er.hypothesis_status,
        ))

    return findings


# ---------------------------------------------------------------------------
# Matched Finding
# ---------------------------------------------------------------------------

@dataclass
class MatchedFinding:
    """A finding confirmed by both paths."""
    llm_note: AnalystNote
    pipeline_finding: PipelineFinding
    shared_words: list[str]


@dataclass
class MergedFindings:
    """Output of Stage C reconciliation."""
    both_paths: list[MatchedFinding]
    llm_only: list[AnalystNote]
    pipeline_only: list[PipelineFinding]
    agreement_score: float  # 0-1, informational only


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_MIN_SHARED_WORDS = 2


def _match_score(llm_note: AnalystNote, pipeline: PipelineFinding) -> tuple[int, list[str]]:
    """Compute match score between an LLM note and a pipeline finding.

    Returns (shared_word_count, shared_words). Returns (0, []) if not a match.
    Match requires: same section, same direction, >= 2 shared content words.
    """
    if not llm_note.direction or not pipeline.direction:
        return 0, []
    if llm_note.direction != pipeline.direction:
        return 0, []

    # Section match — use most frequent section across all LLM citations
    if not llm_note.citations or pipeline.section is None:
        return 0, []
    section_counts: Counter[str] = Counter(c.section for c in llm_note.citations)
    llm_section = section_counts.most_common(1)[0][0]
    if llm_section != pipeline.section:
        return 0, []

    # Content word overlap
    llm_words = extract_content_words(llm_note.claim)
    shared = sorted(llm_words & pipeline.content_words)
    if len(shared) < _MIN_SHARED_WORDS:
        return 0, []

    return len(shared), shared


def _collect_all_notes(analyst_notes: AnalystNotes) -> list[AnalystNote]:
    """Flatten all notes from AnalystNotes into a single list."""
    return (
        list(analyst_notes.positives)
        + list(analyst_notes.risks)
        + list(analyst_notes.surprises)
        + list(analyst_notes.adjustment_triggers)
    )


def reconcile_findings(
    analyst_notes: AnalystNotes | None,
    hypotheses: list[Hypothesis],
    evidence_results: list[EvidenceResult],
) -> MergedFindings | None:
    """Compare LLM analyst notes against deterministic pipeline findings.

    Owns projection: calls project_pipeline_findings internally.
    Uses greedy one-to-one matching ranked by shared content word count.
    Returns None if analyst_notes is None.
    """
    if analyst_notes is None:
        return None

    pipeline_findings = project_pipeline_findings(hypotheses, evidence_results)
    all_llm_notes = _collect_all_notes(analyst_notes)

    candidates: list[tuple[int, int, int, list[str]]] = []
    for li, note in enumerate(all_llm_notes):
        for pi, pf in enumerate(pipeline_findings):
            score, words = _match_score(note, pf)
            if score >= _MIN_SHARED_WORDS:
                candidates.append((score, li, pi, words))

    candidates.sort(key=lambda c: -c[0])

    matched_llm: set[int] = set()
    matched_pipe: set[int] = set()
    both_paths: list[MatchedFinding] = []

    for score, li, pi, words in candidates:
        if li in matched_llm or pi in matched_pipe:
            continue
        both_paths.append(MatchedFinding(
            llm_note=all_llm_notes[li],
            pipeline_finding=pipeline_findings[pi],
            shared_words=words,
        ))
        matched_llm.add(li)
        matched_pipe.add(pi)

    llm_only = [n for i, n in enumerate(all_llm_notes) if i not in matched_llm]
    pipe_only = [f for i, f in enumerate(pipeline_findings) if i not in matched_pipe]

    total = len(both_paths) + len(llm_only) + len(pipe_only)
    agreement_score = len(both_paths) / total if total > 0 else 0.0

    return MergedFindings(
        both_paths=both_paths,
        llm_only=llm_only,
        pipeline_only=pipe_only,
        agreement_score=agreement_score,
    )
