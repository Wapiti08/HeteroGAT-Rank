"""OSPTrack telemetry decoding; independent of evidence sources and matching."""

from __future__ import annotations

import json
import os
import shlex
from collections import defaultdict
from typing import Any

from utils.osptrack_decode import as_list, get_dict, iter_dicts, normalize_command

from .normalization import clean, normalize_domain, valid_ip


PHASES = ("import", "install")
TELEMETRY_KEYS = (
    "ips", "domains", "ports", "endpoints", "commands", "non_harness_commands",
    "processes", "non_harness_processes", "files", "file_reads", "file_writes", "file_deletes",
)
HARNESS_COMMAND_MARKERS = (
    "/usr/bin/npm init", "/usr/bin/npm install", "npm init", "npm install",
    "analyze-node.js", "sleep 30m", "pip install", "python -m pip",
    "gem install", "cargo install",
)


def is_harness_command(command: str) -> bool:
    lower = command.lower()
    return any(marker in lower for marker in HARNESS_COMMAND_MARKERS)


def command_process(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return os.path.basename(parts[0]) if parts else ""


def telemetry_by_phase_from_row(row: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    """Decode telemetry without discarding OSPTrack's execution phase."""
    by_phase: dict[str, dict[str, set[str]]] = {}
    for phase in PHASES:
        telemetry: dict[str, set[str]] = defaultdict(set)
        by_phase[phase] = telemetry
        for entry in iter_dicts(row.get(f"{phase}_Sockets")):
            address = normalize_domain(clean(get_dict(entry, "Address")))
            port = clean(get_dict(entry, "Port"))
            if address:
                telemetry["ips" if valid_ip(address) else "domains"].add(address)
                if port and port != "0":
                    telemetry["ports"].add(port)
                    telemetry["endpoints"].add(f"{address}:{port}")
            for hostname in as_list(get_dict(entry, "Hostnames", [])):
                if clean(hostname):
                    telemetry["domains"].add(normalize_domain(clean(hostname)))
        for entry in iter_dicts(row.get(f"{phase}_DNS")):
            for query in as_list(get_dict(entry, "Queries", [])):
                if isinstance(query, dict) and clean(query.get("Hostname")):
                    telemetry["domains"].add(normalize_domain(clean(query.get("Hostname"))))
        for entry in iter_dicts(row.get(f"{phase}_Commands")):
            command = normalize_command(get_dict(entry, "Command", []))
            if command:
                telemetry["commands"].add(command)
                process = command_process(command)
                if process:
                    telemetry["processes"].add(process)
                if not is_harness_command(command):
                    telemetry["non_harness_commands"].add(command)
                    if process:
                        telemetry["non_harness_processes"].add(process)
        for entry in iter_dicts(row.get(f"{phase}_Files")):
            path = clean(get_dict(entry, "Path"))
            if path:
                telemetry["files"].add(path)
                if entry.get("Read") is True:
                    telemetry["file_reads"].add(path)
                if entry.get("Write") is True:
                    telemetry["file_writes"].add(path)
                if entry.get("Delete") is True:
                    telemetry["file_deletes"].add(path)
    return by_phase


def merge_phase_telemetry(
    by_phase: dict[str, dict[str, set[str]]],
) -> dict[str, set[str]]:
    telemetry: dict[str, set[str]] = defaultdict(set)
    for phase in PHASES:
        for key in TELEMETRY_KEYS:
            telemetry[key].update(by_phase.get(phase, {}).get(key, set()))
    return telemetry


def telemetry_from_row(row: dict[str, Any]) -> dict[str, set[str]]:
    return merge_phase_telemetry(telemetry_by_phase_from_row(row))


def observed_trigger_evidence(
    by_phase: dict[str, dict[str, set[str]]], *, command_limit: int = 5,
) -> tuple[tuple[str, ...], str]:
    observed: list[str] = []
    evidence: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        telemetry = by_phase.get(phase, {})
        families = [key for key in TELEMETRY_KEYS if telemetry.get(key)]
        if not families:
            continue
        observed.append(phase)
        evidence[phase] = {
            "activity_families": families,
            "commands": sorted(telemetry.get("commands", set()))[:command_limit],
        }
    return tuple(observed), json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def behavior_activity(telemetry: dict[str, set[str]]) -> str:
    families = []
    if telemetry["domains"] or telemetry["ips"]:
        families.append("network")
    if telemetry["commands"]:
        families.append("command")
    if telemetry["files"]:
        families.append("file")
    return ",".join(families)


def telemetry_preview(telemetry: dict[str, set[str]], key: str, limit: int) -> str:
    return json.dumps(sorted(telemetry[key])[:limit], ensure_ascii=False)
