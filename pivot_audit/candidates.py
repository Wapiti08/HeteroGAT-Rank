"""Dataset-independent candidate identifiers and review metadata."""

import hashlib

from .schema import ExternalEvidence


def candidate_id(record_id: str, evidence: ExternalEvidence) -> str:
    raw = "\x1f".join((record_id, evidence.source_dataset, evidence.source_id, evidence.evidence_type, evidence.evidence_value))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def review_priority(evidence: ExternalEvidence, compatibility: str) -> str:
    if compatibility == "exact" and evidence.extraction in {"structured", "derived_structured_url"} and evidence.evidence_type in {"ip", "domain", "url", "endpoint"}:
        return "high"
    if compatibility in {"exact", "range_unresolved"} and evidence.evidence_type in {"ip", "domain", "url", "string", "endpoint", "payload_file"}:
        return "medium"
    return "low"
