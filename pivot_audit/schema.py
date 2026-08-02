"""Domain models and annotation schema for pivot auditing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalEvidence:
    ecosystem: str
    package: str
    source_dataset: str
    source_id: str
    source_reference: str
    affected_versions: tuple[str, ...]
    version_specific: bool
    version_constraint: str
    evidence_type: str
    evidence_value: str
    expected_behavior: str
    trigger: str
    trigger_evidence: str
    trigger_source_reference: str
    trigger_extraction: str
    trigger_confidence: str
    evidence_confidence: str
    extraction: str
    attribution_scope: str = "unknown"
