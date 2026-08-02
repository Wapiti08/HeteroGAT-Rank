"""Reported-trigger provenance, runtime-phase evidence, and comparison policy."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


KNOWN_TRIGGERS = {"install", "import", "runtime", "function_call", "test"}
SOURCE_SECTION_RE = re.compile(
    r"^## Source:\s*(?P<source>[^\s(]+)(?:\s*\((?P<source_id>[^)]+)\))?\s*$",
    re.MULTILINE,
)
SENTENCE_RE = re.compile(r"[^\n.!?]+(?:[.!?]|$)")


@dataclass(frozen=True)
class TriggerEvidence:
    trigger: str
    evidence: str
    source_reference: str
    extraction: str
    confidence: str


def normalize_trigger(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "installation": "install",
        "preinstall": "install",
        "pre_install": "install",
        "postinstall": "install",
        "post_install": "install",
        "execution": "runtime",
        "run": "runtime",
        "call": "function_call",
        "function": "function_call",
        "testing": "test",
    }
    return aliases.get(text, text) if aliases.get(text, text) in KNOWN_TRIGGERS else "unknown"


def split_source_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SOURCE_SECTION_RE.finditer(text or ""))
    if not matches:
        return [("advisory_text", text or "")]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group("source"), text[match.end() : end].strip()))
    return sections


def _sentences(text: str) -> Iterable[str]:
    for match in SENTENCE_RE.finditer(text.replace("\r", " ").replace("\n", " ")):
        sentence = " ".join(match.group(0).split())
        if sentence:
            yield sentence


def _trigger_from_sentence(sentence: str) -> str:
    lower = sentence.lower()
    # Conditional execution is more specific than mentions of phases it evades.
    if re.search(r"(?:only\s+)?trigger(?:s|ed)?\s+when .{0,80}(?:function|method).{0,40}(?:call|invoke)", lower):
        return "function_call"
    if re.search(r"(?:when|once|upon|during|on)\s+(?:the\s+)?(?:package\s+)?install(?:ed|ation|ing)?\b", lower):
        return "install"
    if re.search(r"\b(?:pre[_ -]?install|post[_ -]?install|install scripts?|setup\.py)\b", lower) and re.search(
        r"\b(?:execute|executed|execution|run|runs|running|trigger|payload|malicious)\b", lower
    ):
        return "install"
    if re.search(r"(?:when|once|upon|during|at|on)\s+(?:the\s+)?(?:package\s+)?(?:is\s+)?import(?:ed|ing)?\b", lower):
        return "import"
    if re.search(r"\b(?:require\(|importing the package|at import time)\b", lower):
        return "import"
    if re.search(r"(?:when|once|upon|during|at)\s+(?:the\s+)?(?:package\s+)?(?:is\s+)?(?:run|executed|used)|\bat runtime\b", lower):
        return "runtime"
    if re.search(r"\b(?:test script|during tests?|when tests? (?:are )?run)\b", lower):
        return "test"
    return "unknown"


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


@lru_cache(maxsize=1)
def load_reference_catalog() -> dict[str, dict[str, Any]]:
    path = Path(__file__).with_name("data") / "trigger_reference_evidence.json"
    records = json.loads(path.read_text())
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        for reference in record["references"]:
            output[_normalize_url(reference)] = record
    return output


def reference_catalog_record(source_reference: str) -> tuple[str, dict] | None:
    catalog = load_reference_catalog()
    for reference in (item for item in source_reference.split(";") if item):
        record = catalog.get(_normalize_url(reference))
        if record:
            return reference, record
    return None


def extract_reported_trigger(details: str, source_reference: str) -> TriggerEvidence:
    """Extract only source-supported triggers; never infer them from package identity."""
    catalog_match = reference_catalog_record(source_reference)
    if catalog_match:
        reference, record = catalog_match
        location = record.get("evidence_location", "")
        evidence = record["evidence"] + (f" Location: {location}." if location else "")
        return TriggerEvidence(
            trigger=normalize_trigger(record["trigger"]),
            evidence=evidence,
            source_reference=reference,
            extraction="curated_reference",
            confidence=record.get("confidence", "medium"),
        )

    for source_name, section in split_source_sections(details):
        for sentence in _sentences(section):
            trigger = _trigger_from_sentence(sentence)
            if trigger != "unknown":
                return TriggerEvidence(
                    trigger=trigger,
                    evidence=sentence[:500],
                    source_reference=source_reference,
                    extraction=f"advisory_text_regex:{source_name}",
                    confidence="medium",
                )

    return TriggerEvidence(
        trigger="unknown",
        evidence="No explicit trigger mechanism was found in the locally available advisory text or curated referenced reports.",
        source_reference=source_reference,
        extraction="unavailable",
        confidence="unknown",
    )


def structured_trigger(value: str, source_reference: str) -> TriggerEvidence:
    trigger = normalize_trigger(value)
    if trigger == "unknown":
        return TriggerEvidence(
            trigger="unknown",
            evidence="The structured source did not provide a recognized trigger value.",
            source_reference=source_reference,
            extraction="structured_missing",
            confidence="unknown",
        )
    return TriggerEvidence(
        trigger=trigger,
        evidence=f"The structured source labels the package trigger as {trigger}.",
        source_reference=source_reference,
        extraction="structured_trigger",
        confidence="high",
    )


def compare_trigger(reported_trigger: str, observed_triggers: Iterable[str]) -> str:
    reported = normalize_trigger(reported_trigger)
    observed = {normalize_trigger(item) for item in observed_triggers}
    observed.discard("unknown")
    if reported == "unknown":
        return "reported_trigger_unknown"
    if reported in observed:
        return "match"
    if reported in {"runtime", "function_call", "test"}:
        return "not_determinable_from_phase_telemetry"
    return "reported_trigger_not_observed"
