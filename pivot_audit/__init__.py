"""Pivot ground-truth construction and observability auditing."""

from .candidates import candidate_id
from .matching import match_evidence
from .normalization import normalize_ecosystem, normalize_package, version_compatibility
from .osptrack import observed_trigger_evidence, telemetry_by_phase_from_row, telemetry_from_row
from .osptrack_builder import build_pivot_gt_candidates
from .review_queue import build_review_queue
from .schema import ExternalEvidence
from .sources import informative_behavior, load_backstabbers, load_malicious_packages_info
from .triggers import compare_trigger, extract_reported_trigger
from .validation import ALLOWED_ANNOTATIONS, load_annotations, validate_annotations

__all__ = [
    "ALLOWED_ANNOTATIONS",
    "ExternalEvidence",
    "build_pivot_gt_candidates",
    "build_review_queue",
    "candidate_id",
    "compare_trigger",
    "extract_reported_trigger",
    "informative_behavior",
    "load_annotations",
    "load_backstabbers",
    "load_malicious_packages_info",
    "match_evidence",
    "normalize_ecosystem",
    "normalize_package",
    "observed_trigger_evidence",
    "telemetry_by_phase_from_row",
    "telemetry_from_row",
    "validate_annotations",
    "version_compatibility",
]
