"""Pure normalization and version-alignment helpers."""

from __future__ import annotations

import ast
import ipaddress
import math
from typing import Any, Iterable

from .schema import ExternalEvidence


ECOSYSTEM_ALIASES = {
    "crates-io": "crates.io", "crates.io": "crates.io",
    "rubygem": "rubygems", "rubygems": "rubygems",
    "pypi": "pypi", "npm": "npm", "mavencentral": "maven", "maven": "maven",
}


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_ecosystem(value: Any) -> str:
    text = clean(value).lower()
    return ECOSYSTEM_ALIASES.get(text, text)


def normalize_package(value: Any) -> str:
    return clean(value).lower()


def normalize_domain(value: str) -> str:
    return value.strip().lower().rstrip(".")


def valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def literal_list(value: Any) -> list[str]:
    text = clean(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, (list, tuple, set)):
        return [clean(item) for item in parsed if clean(item)]
    return [clean(parsed)] if clean(parsed) else []


def extract_versions(affected: Iterable[dict[str, Any]]) -> tuple[tuple[str, ...], bool, str]:
    versions: set[str] = set()
    has_constraint = has_ranges = has_explicit = False
    for item in affected:
        explicit = item.get("versions") or []
        if explicit:
            has_constraint = has_explicit = True
            versions.update(clean(v) for v in explicit if clean(v))
        for range_item in item.get("ranges") or []:
            has_ranges = True
            events = range_item.get("events") or []
            has_constraint = has_constraint or bool(events)
            for event in events:
                introduced = clean(event.get("introduced"))
                if introduced and introduced != "0":
                    versions.add(introduced)
    constraint = "mixed" if has_ranges and has_explicit else "range" if has_ranges else "exact_versions" if has_explicit else "none"
    return tuple(sorted(versions)), has_constraint, constraint


def version_compatibility(version: str, evidence: ExternalEvidence) -> str:
    if version in evidence.affected_versions:
        return "exact"
    if not evidence.version_specific:
        return "unspecified"
    if evidence.version_constraint in {"range", "mixed"}:
        return "range_unresolved"
    return "mismatch_or_unresolved"
