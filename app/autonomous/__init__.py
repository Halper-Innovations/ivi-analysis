"""Pure contracts for bounded autonomous analyst runs.

This package intentionally exposes only dataclass contracts. It must stay safe
to import without touching config, DB, network, filesystem, or LLM providers.
Runtime orchestration should live in a separate module once the contract is
stable.
"""
from __future__ import annotations

from app.autonomous.run_contract import (
    AutonomousRunArtifact,
    AutonomousRunBudget,
    AutonomousRunRequest,
    BeliefUpdate,
    CandidateDecision,
    EvidenceReference,
    ResearchQuestion,
    ToolCallRecord,
)
from app.autonomous.sector_contract import (
    AutonomousSectorFinancialRunArtifact,
    SectorCompanyFinancialPacket,
    SectorExpectedReturnScenario,
    SectorFinalDecision,
    SectorFinancialFramework,
    SectorResearchQuestion,
)

__all__ = [
    "AutonomousRunArtifact",
    "AutonomousRunBudget",
    "AutonomousRunRequest",
    "AutonomousSectorFinancialRunArtifact",
    "BeliefUpdate",
    "CandidateDecision",
    "EvidenceReference",
    "ResearchQuestion",
    "SectorCompanyFinancialPacket",
    "SectorExpectedReturnScenario",
    "SectorFinalDecision",
    "SectorFinancialFramework",
    "SectorResearchQuestion",
    "ToolCallRecord",
]
