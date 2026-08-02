"""Compatibility facade for the refactored pivot_audit package.

New code should import from ``pivot_audit`` or the focused modules directly.
"""

from .candidates import candidate_id, review_priority
from .matching import match_evidence, match_osptrack_evidence
from .normalization import *  # noqa: F401,F403
from .osptrack import (
    behavior_activity, observed_trigger_evidence, telemetry_by_phase_from_row,
    telemetry_from_row, telemetry_preview,
)
from .osptrack_builder import audit, build_pivot_gt_candidates
from .schema import ExternalEvidence
from .sources import *  # noqa: F401,F403
from .triggers import compare_trigger, extract_reported_trigger, normalize_trigger

__all__ = [name for name in globals() if not name.startswith("_")]
