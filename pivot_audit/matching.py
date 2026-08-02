"""Evidence-to-telemetry matching policies."""

from urllib.parse import urlparse

from .normalization import normalize_domain
from .schema import ExternalEvidence


def match_osptrack_evidence(
    evidence: ExternalEvidence, telemetry: dict[str, set[str]]
) -> tuple[str, str, str]:
    value = evidence.evidence_value.strip()
    if evidence.evidence_type == "ip":
        return ("exact_ioc", value, "socket") if value in telemetry["ips"] else ("unmatched", "", "")
    if evidence.evidence_type == "domain":
        domain = normalize_domain(value)
        return ("exact_ioc", domain, "network") if domain in telemetry["domains"] else ("unmatched", "", "")
    if evidence.evidence_type == "endpoint":
        endpoint = value.lower()
        return ("exact_ioc", endpoint, "socket_endpoint") if endpoint in telemetry.get("endpoints", set()) else ("unmatched", "", "")
    if evidence.evidence_type == "url":
        for command in telemetry["commands"]:
            if value.lower() in command.lower():
                return "exact_ioc", command, "command"
        host = normalize_domain(urlparse(value).hostname or "")
        return ("partial_ioc", host, "network_host_component") if host and host in telemetry["domains"] else ("unmatched", "", "")
    if evidence.evidence_type == "string":
        for command in telemetry["commands"]:
            if value and value.lower() in command.lower():
                return "exact_ioc", command, "command"
        return "unmatched", "", ""
    if evidence.evidence_type == "file":
        needle = value.lower().lstrip("./")
        if "/" not in needle and "\\" not in needle:
            return "unmatched", "", ""
        for path in telemetry["files"]:
            if path.lower().lstrip("./").endswith(needle):
                return "partial_ioc", path, "file_path"
        return "unmatched", "", ""
    if evidence.evidence_type == "payload_file":
        needle = value.lower()
        for command in telemetry["commands"]:
            if needle in command.lower():
                return "exact_ioc", command, "command_payload"
        for process in telemetry.get("processes", set()):
            if process.lower() == needle:
                return "exact_ioc", process, "process_name"
        for path in telemetry["files"]:
            if path.replace("\\", "/").rsplit("/", 1)[-1].lower() == needle:
                return "exact_ioc", path, "file_name"
        return "unmatched", "", ""
    return "not_applicable", "", ""


# Stable compatibility name.
match_evidence = match_osptrack_evidence
