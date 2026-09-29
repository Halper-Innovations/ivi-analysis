from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ChangeType = Literal[
    "NEW_DISCLOSURE",
    "REMOVED_DISCLOSURE",
    "LANGUAGE_SHIFT",
    "QUANTITATIVE_CHANGE",
    "COMPETITIVE_SIGNAL",
    "RISK_SIGNAL",
    "STRATEGIC_SIGNAL",
]

Materiality = Literal["HIGH", "MEDIUM", "LOW"]
SectionName = Literal["risk_factors", "md_and_a"]


class FilingChange(BaseModel):
    section: SectionName
    fiscal_year_from: int
    fiscal_year_to: int
    change_type: ChangeType
    materiality: Materiality
    summary: str
    from_excerpt: str | None = None
    to_excerpt: str | None = None


class FilingChangeSet(BaseModel):
    changes: list[FilingChange] = Field(default_factory=list)


class SectionSkip(BaseModel):
    section: str
    fiscal_year_from: int | None = None
    fiscal_year_to: int | None = None
    reason: str


class SectionDiagnostic(BaseModel):
    section: str
    fiscal_year_from: int
    fiscal_year_to: int
    similarity_ratio: float | None = None
    compared_chars_from: int
    compared_chars_to: int
    truncated_from: bool = False
    truncated_to: bool = False
    llm_used: bool = False
    llm_status: Literal["ok", "disabled", "error", "skipped"] = "skipped"
    llm_error: str | None = None


class FilingDiffReport(BaseModel):
    ticker: str
    run_id: str
    as_of_date: str
    years_compared: list[tuple[int, int]] = Field(default_factory=list)
    changes: list[FilingChange] = Field(default_factory=list)
    sections_compared: list[str] = Field(default_factory=list)
    sections_skipped: list[SectionSkip] = Field(default_factory=list)
    section_diagnostics: list[SectionDiagnostic] = Field(default_factory=list)
    truncation_applied: bool = False
    llm_enabled: bool
    created_at: str

    @model_validator(mode="after")
    def _validate_created_at(self) -> "FilingDiffReport":
        datetime.fromisoformat(self.created_at)
        return self


def filing_change_set_schema_for_prompt() -> dict[str, Any]:
    schema = FilingChangeSet.model_json_schema()
    _enforce_additional_properties_false(schema)
    return schema


def validate_filing_change_set(payload: dict[str, Any]) -> FilingChangeSet:
    return FilingChangeSet.model_validate(payload)


def validate_filing_diff_report(payload: dict[str, Any]) -> FilingDiffReport:
    return FilingDiffReport.model_validate(payload)


def _enforce_additional_properties_false(node: object) -> None:
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type == "object":
            if "additionalProperties" not in node:
                node["additionalProperties"] = False
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
        for value in node.values():
            _enforce_additional_properties_false(value)
    elif isinstance(node, list):
        for item in node:
            _enforce_additional_properties_false(item)
