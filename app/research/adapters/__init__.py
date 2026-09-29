from __future__ import annotations

from app.config import AppConfig, get_config
from app.research.adapters.base import AdapterContext, AdapterResult, ResearchAdapter
from app.research.adapters.company_news import CompanyNewsAdapter
from app.research.adapters.edgar import EdgarEvidenceAdapter
from app.research.adapters.external_news import ExternalNewsAdapter
from app.research.adapters.ir_press import IRPressAdapter
from app.research.adapters.sec_exhibits import SecExhibitsAdapter
from app.research.adapters.transcripts import TranscriptAdapter


def build_adapter_chain(cfg: AppConfig | None = None) -> list[ResearchAdapter]:
    cfg = cfg or get_config()
    return [
        EdgarEvidenceAdapter(cfg),
        SecExhibitsAdapter(cfg),
        TranscriptAdapter(cfg),
        ExternalNewsAdapter(cfg),
        IRPressAdapter(cfg),
        CompanyNewsAdapter(cfg),
    ]


__all__ = [
    "AdapterContext",
    "AdapterResult",
    "ResearchAdapter",
    "CompanyNewsAdapter",
    "EdgarEvidenceAdapter",
    "ExternalNewsAdapter",
    "IRPressAdapter",
    "SecExhibitsAdapter",
    "TranscriptAdapter",
    "build_adapter_chain",
]
