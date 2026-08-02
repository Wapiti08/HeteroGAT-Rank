"""Report-backed behavior extraction and conservative telemetry matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .triggers import reference_catalog_record, split_source_sections


BEHAVIOR_TYPES = {
    "network_request", "payload_download", "process_execution", "sensitive_data_access",
    "archive_collection", "exfiltration", "persistence", "reverse_shell", "backdoor",
    "denial_of_service", "propagation",
}


@dataclass(frozen=True)
class ReportedBehavior:
    behavior_type: str
    evidence: str
    source_reference: str
    extraction: str
    confidence: str


OBJECTIVE_MAP = {
    "data exfiltration": ("exfiltration",),
    "backdoor": ("backdoor",),
    "dropper": ("payload_download", "process_execution"),
    "reverse shell": ("reverse_shell",),
    "denial of service": ("denial_of_service",),
    "worm": ("propagation",),
}


TEXT_RULES = {
    "reverse_shell": re.compile(r"\breverse shell\b", re.I),
    "payload_download": re.compile(r"\b(?:download|fetch|dropper|retrieve).{0,100}(?:payload|executable|script|file|code)|\bdownloads?\b", re.I),
    "process_execution": re.compile(r"\b(?:execute|executes|executed|launch|spawn|run).{0,100}(?:payload|executable|script|command|code)|(?:payload|executable|script).{0,60}\b(?:execute|launch|run)", re.I),
    "archive_collection": re.compile(r"\b(?:archive|compress|zip).{0,100}(?:file|director|data|source)", re.I),
    "persistence": re.compile(r"\b(?:persistence|autorun|startup folder|crontab|systemd)\b", re.I),
    "sensitive_data_access": re.compile(r"\b(?:steal|collect|scan|extract|access).{0,120}(?:credential|secret|password|cookie|token|ssh|source code|configuration|environment variable|sensitive)", re.I),
    "exfiltration": re.compile(r"\b(?:exfiltrat|upload|send).{0,120}(?:data|file|credential|secret|information|archive|webhook|server)", re.I),
    "network_request": re.compile(r"\b(?:http get|webhook|connects? to|network request)\b", re.I),
    "backdoor": re.compile(r"\bbackdoor\b", re.I),
    "denial_of_service": re.compile(r"\bdenial of service\b|\bdos attack\b", re.I),
    "propagation": re.compile(r"\b(?:worm|self-propagat|spread to)\b", re.I),
}


def extract_reported_behaviors(
    details: str, source_reference: str, *, package: str, objectives: Iterable[str] = (),
) -> list[ReportedBehavior]:
    output: dict[str, ReportedBehavior] = {}
    catalog_match = reference_catalog_record(source_reference)
    if catalog_match:
        reference, record = catalog_match
        for behavior in record.get("behaviors", []):
            packages = {str(value).lower() for value in behavior.get("packages", [])}
            if packages and package.lower() not in packages:
                continue
            kind = behavior.get("type", "")
            if kind in BEHAVIOR_TYPES:
                output[kind] = ReportedBehavior(
                    kind, behavior.get("evidence", ""), reference,
                    "curated_reference_behavior", behavior.get("confidence", "medium"),
                )

    for objective in objectives:
        for kind in OBJECTIVE_MAP.get(str(objective).strip().lower(), ()):
            output.setdefault(kind, ReportedBehavior(
                kind, f"The structured source objective is {objective}.", source_reference,
                "structured_objective", "high",
            ))

    for source_name, section in split_source_sections(details):
        normalized = " ".join(section.split())
        for kind, pattern in TEXT_RULES.items():
            match = pattern.search(normalized)
            if match and kind not in output:
                start, end = max(0, match.start() - 100), min(len(normalized), match.end() + 140)
                output[kind] = ReportedBehavior(
                    kind, normalized[start:end][:500], source_reference,
                    f"advisory_behavior_regex:{source_name}", "medium",
                )
    return list(output.values())


def _contains_any(values: Iterable[str], patterns: Iterable[str]) -> bool:
    return any(pattern in value.lower() for value in values for pattern in patterns)


def match_behavior(behavior_type: str, telemetry: dict[str, Any]) -> tuple[str, str]:
    """Return conservative behavioral support, never an IOC exact match."""
    commands = telemetry.get("non_harness_commands", telemetry.get("commands", set()))
    processes = telemetry.get("non_harness_processes", telemetry.get("processes", set()))
    reads = telemetry.get("file_reads", set())
    writes = telemetry.get("file_writes", set())
    files = telemetry.get("files", set())
    benign_domains = {
        "registry.npmjs.org", "npmjs.org", "pypi.org", "files.pythonhosted.org",
        "rubygems.org", "crates.io", "index.crates.io",
    }
    domains = {str(value).lower() for value in telemetry.get("domains", set())}
    suspicious_domains = {
        domain for domain in domains
        if domain not in benign_domains and not any(domain.endswith("." + known) for known in benign_domains)
    }
    nonstandard_ports = {
        str(port) for port in telemetry.get("ports", set()) if str(port) not in {"53", "80", "443"}
    }
    command_network = _contains_any(commands, ("http://", "https://", "ftp://", "curl ", "wget "))
    network = bool(suspicious_domains or nonstandard_ports or command_network)
    command_or_process = bool(commands or processes)

    if behavior_type == "payload_download":
        explicit = _contains_any(commands, ("curl ", "wget ", "invoke-webrequest", "urlretrieve", "downloadfile"))
        if explicit:
            return "behavior_observed", "download command observed"
        if network and writes:
            return "behavior_partially_observed", "network activity and file writes observed"
    elif behavior_type == "process_execution":
        if command_or_process:
            return "behavior_partially_observed", "non-harness command or process observed without a report-specific process indicator"
    elif behavior_type == "sensitive_data_access":
        sensitive = ("/.ssh/", ".npmrc", "credentials", "passwd", "shadow", "cookie", "login data", "wallet", ".env")
        if _contains_any(reads or files, sensitive):
            return "behavior_partially_observed", "sensitive-path access observed without confirmed data-flow attribution"
    elif behavior_type == "archive_collection":
        if _contains_any(commands, ("tar ", "zip ", "gzip ", "7z ")) or _contains_any(writes or files, (".zip", ".tgz", ".tar", ".gz")):
            return "behavior_observed", "archive command or archive artifact observed"
    elif behavior_type == "exfiltration":
        if network and (reads or _contains_any(commands, ("upload", "ftp ", "scp ", "webhook"))):
            return "behavior_partially_observed", "network activity plus collection/upload context observed"
    elif behavior_type == "persistence":
        persistence = ("startup", "crontab", "/cron", "systemd", "currentversion\\run", "launchagents")
        if _contains_any(commands, persistence) or _contains_any(writes or files, persistence):
            return "behavior_observed", "persistence command or path observed"
    elif behavior_type == "reverse_shell":
        reverse = ("/dev/tcp/", "nc -e", "ncat ", "bash -i", "powershell -nop", "socket.connect")
        if _contains_any(commands, reverse):
            return "behavior_observed", "reverse-shell command pattern observed"
        if network and _contains_any(commands, ("bash", "sh ", "powershell", "cmd.exe")):
            return "behavior_partially_observed", "shell execution and network activity observed"
    elif behavior_type == "network_request":
        if network:
            return "behavior_partially_observed", "network activity observed without semantic flow confirmation"
    elif behavior_type in {"backdoor", "denial_of_service", "propagation"}:
        return "behavior_unobservable", "processed telemetry cannot establish this semantic behavior"
    return "behavior_not_observed", "required supporting telemetry was not observed"
