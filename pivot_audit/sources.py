"""Adapters for independent malicious-package evidence sources."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import pandas as pd

from .normalization import (
    clean, extract_versions, literal_list, normalize_domain, normalize_ecosystem,
    normalize_package, valid_ip,
)
from .schema import ExternalEvidence
from .behaviors import extract_reported_behaviors
from .triggers import extract_reported_trigger, structured_trigger


URL_RE = re.compile(r"https?://[^\s\]>)\"']+", re.IGNORECASE)
IP_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
DOMAIN_RE = re.compile(r"(?<![@\w.-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}(?![\w.-])", re.IGNORECASE)


def iter_ioc_values(iocs: Any) -> Iterator[tuple[str, str]]:
    if not isinstance(iocs, dict):
        return
    mapping = {"urls": "url", "domains": "domain", "ips": "ip", "strings": "string"}
    for key, evidence_type in mapping.items():
        values = iocs.get(key) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            if clean(value):
                cleaned = clean(value)
                yield evidence_type, cleaned
                if evidence_type == "url":
                    try:
                        parsed = urlparse(cleaned)
                        if parsed.hostname and parsed.port:
                            yield "endpoint", f"{normalize_domain(parsed.hostname)}:{parsed.port}"
                        basename = Path(parsed.path).name
                        if Path(basename).suffix.lower() in {".exe", ".dll", ".ps1", ".bat", ".cmd", ".sh", ".py", ".js"}:
                            yield "payload_file", basename
                    except ValueError:
                        pass


def extract_text_iocs(text: str) -> Iterator[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for url in URL_RE.findall(text):
        item = ("url", url.rstrip(".,;"))
        if item not in seen:
            seen.add(item); yield item
    remaining = URL_RE.sub(" ", text)
    for ip in IP_RE.findall(remaining):
        item = ("ip", ip)
        if valid_ip(ip) and item not in seen:
            seen.add(item); yield item
    for domain in DOMAIN_RE.findall(remaining):
        item = ("domain", normalize_domain(domain))
        if item not in seen:
            seen.add(item); yield item


def informative_behavior(text: str) -> bool:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---" or "Per source details" in stripped or stripped.startswith("## Source:"):
            continue
        lines.append(stripped)
    return len(re.findall(r"[A-Za-z0-9]+", " ".join(lines))) >= 6


def load_malicious_packages_info(root: Path) -> list[ExternalEvidence]:
    output: list[ExternalEvidence] = []
    for path in sorted(root.glob("*/*/*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        affected = [a for a in record.get("affected") or [] if isinstance(a, dict)]
        versions, version_specific, version_constraint = extract_versions(affected)
        package_info = affected[0].get("package", {}) if affected else {}
        details = clean(record.get("details"))
        source_reference = ";".join(
            clean(r.get("url")) for r in record.get("references") or [] if clean(r.get("url"))
        )
        trigger = extract_reported_trigger(details, source_reference)
        common = dict(
            ecosystem=normalize_ecosystem(package_info.get("ecosystem") or path.parts[-3]),
            package=normalize_package(package_info.get("name") or path.parts[-2]),
            source_dataset="malicious-packages-info", source_id=clean(record.get("id") or path.stem),
            source_reference=source_reference,
            affected_versions=versions, version_specific=version_specific,
            version_constraint=version_constraint, expected_behavior=details,
            trigger=trigger.trigger, trigger_evidence=trigger.evidence,
            trigger_source_reference=trigger.source_reference,
            trigger_extraction=trigger.extraction, trigger_confidence=trigger.confidence,
        )
        for evidence_type, value in iter_ioc_values((record.get("database_specific") or {}).get("iocs")):
            output.append(ExternalEvidence(
                **common, evidence_type=evidence_type, evidence_value=value,
                evidence_confidence="high",
                extraction="derived_structured_url" if evidence_type in {"endpoint", "payload_file"} else "structured",
                attribution_scope="record_attached",
            ))
        for behavior in extract_reported_behaviors(
            details, source_reference, package=common["package"],
        ):
            output.append(ExternalEvidence(**{
                **common, "evidence_type": "behavior_event", "evidence_value": behavior.behavior_type,
                "expected_behavior": behavior.evidence, "evidence_confidence": behavior.confidence,
                "extraction": behavior.extraction, "attribution_scope": "referenced_report",
            }))
        output.append(ExternalEvidence(
            **common, evidence_type="package_report", evidence_value="",
            evidence_confidence="unknown", extraction="advisory_text",
            attribution_scope="package_record",
        ))
    return output


def load_backstabbers(index_path: Path) -> list[ExternalEvidence]:
    frame = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    output: list[ExternalEvidence] = []
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        versions = tuple(sorted(set(literal_list(row.get("Affected Version")))))
        details = clean(row.get("Details"))
        objectives = literal_list(row.get("Objective"))
        behavior = "; ".join(p for p in (", ".join(objectives), details) if p)
        ecosystem, package = normalize_ecosystem(row.get("Type")), normalize_package(row.get("Package Name"))
        source_reference = clean(row.get("Source"))
        trigger = structured_trigger(clean(row.get("Trigger")), source_reference)
        common = dict(
            ecosystem=ecosystem, package=package, source_dataset="backstabbers",
            source_id=f"backstabbers:{ecosystem}:{package}:row-{row_number}", source_reference=source_reference,
            affected_versions=versions, version_specific=bool(versions),
            version_constraint="exact_versions" if versions else "none",
            expected_behavior=behavior, trigger=trigger.trigger,
            trigger_evidence=trigger.evidence,
            trigger_source_reference=trigger.source_reference,
            trigger_extraction=trigger.extraction,
            trigger_confidence=trigger.confidence,
        )
        for evidence_type, value in extract_text_iocs(details):
            output.append(ExternalEvidence(
                **common, evidence_type=evidence_type, evidence_value=value,
                evidence_confidence="low", extraction="details_regex",
                attribution_scope="package_row",
            ))
        location = clean(row.get("Location of malicious snippet"))
        if location:
            output.append(ExternalEvidence(
                **common, evidence_type="file", evidence_value=location,
                evidence_confidence="high", extraction="structured_location",
                attribution_scope="package_row",
            ))
        for reported in extract_reported_behaviors(
            behavior, source_reference, package=package, objectives=objectives,
        ):
            output.append(ExternalEvidence(**{
                **common, "evidence_type": "behavior_event", "evidence_value": reported.behavior_type,
                "expected_behavior": reported.evidence, "evidence_confidence": reported.confidence,
                "extraction": reported.extraction, "attribution_scope": "package_row_or_reference",
            }))
        output.append(ExternalEvidence(
            **common, evidence_type="package_report", evidence_value="",
            evidence_confidence="unknown", extraction="index_text",
            attribution_scope="package_row",
        ))
    return output


def load_external_evidence(malicious_info: Path, backstabbers_index: Path) -> list[ExternalEvidence]:
    evidence = load_malicious_packages_info(malicious_info)
    evidence.extend(load_backstabbers(backstabbers_index))
    return evidence
