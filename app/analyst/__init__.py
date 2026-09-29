"""Analyst contract layer.

Pure dataclass contracts for what an analyst can see (AnalysisEvidenceBundle)
and what an analyst concludes (AnalysisReport). Introduced as Redirect Task 1;
additive and parallel to the existing app.research path until later tasks
migrate the live pipeline onto these contracts.

**Import-safety boundary:** this module imports ONLY the pure contract
dataclasses. It does not import `app.analyst.bundle_builder`, because the
builder transitively imports `app.research.adapters.base`, which calls
`get_config()` at module scope. `from app.analyst import AnalysisEvidenceBundle`
must remain a pure contract import with no DB / network / file / config
side effects. Callers that need the builder import it directly:

    from app.analyst.bundle_builder import build_analysis_evidence_bundle

No I/O, no network, no DB, no config lookups, no LLM calls at import time.
"""

from app.analyst.evidence_bundle import (
    AnalysisEvidenceBundle,
    BundleEvent,
    BundleFiling,
    PriorThesisSnapshot,
    ValuationSnapshot,
)
from app.analyst.thesis_contract import (
    AnalysisCitation,
    AnalysisFinding,
    AnalysisReport,
    ExpectationsGap,
    Falsifier,
    OpenQuestion,
    ValuationConclusion,
)

__all__ = [
    "AnalysisCitation",
    "AnalysisEvidenceBundle",
    "AnalysisFinding",
    "AnalysisReport",
    "BundleEvent",
    "BundleFiling",
    "ExpectationsGap",
    "Falsifier",
    "OpenQuestion",
    "PriorThesisSnapshot",
    "ValuationConclusion",
    "ValuationSnapshot",
]
