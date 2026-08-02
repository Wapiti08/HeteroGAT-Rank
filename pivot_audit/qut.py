"""QUT-DV25 adapter for runtime-pivot ground-truth candidate construction."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from utils.qut_decode import parse_listish

from .candidates import candidate_id, review_priority
from .behaviors import match_behavior
from .normalization import clean, normalize_domain, version_compatibility
from .reporting import write_review_file
from .review_queue import build_review_queue
from .schema import ExternalEvidence
from .sources import load_external_evidence
from .triggers import compare_trigger


QUT_FILES = {
    "install": "QUT-DV25_Install_Traces/QUT-DV25_Install_Traces.csv",
    "syscall": "QUT-DV25_SysCall_Traces/QUT-DV25_SysCall_Traces.csv",
    "filetop": "QUT-DV25_Filetop_Traces/QUT-DV25_Filetop_Traces.csv",
    "opensnoop": "QUT-DV25_Opensnoop_Traces/QUT-DV25_Opensnoop_Traces.csv",
    "tcp": "QUT-DV25_TCP_Traces/QUT-DV25_TCP_Traces.csv",
    "pattern": "QUT-DV25_Pattern_Traces/QUT-DV25_Pattern_Traces.csv",
}

ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".whl", ".zip", ".tgz")
IP_PORT_RE = re.compile(r"^(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s*->\s*(?P<port>\d{1,5})$")
PORT_STATE_RE = re.compile(r"^(?P<port>\d{1,5})\s*->\s*(?P<state>[A-Z_]+)$")


def strip_archive_suffix(filename: str) -> str:
    lower = filename.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def align_qut_identity(filename: str, known_packages: set[str]) -> tuple[str, str, str]:
    """Return package, version, and alignment status using longest-name match."""
    stem = strip_archive_suffix(filename).lower()
    matches = [name for name in known_packages if stem == name or stem.startswith(name + "-")]
    if not matches:
        return stem, "", "unmatched_filename"
    package = max(matches, key=len)
    remainder = stem[len(package) :].lstrip("-")
    # Wheel filenames append compatibility tags after the version.
    version = remainder.split("-", 1)[0] if filename.lower().endswith(".whl") else remainder
    return package, version, "longest_external_name"


def load_qut_tables(base: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for kind, relative in QUT_FILES.items():
        path = base / relative
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        tables[kind] = frame.set_index("Package_Name", drop=False)
    return tables


def _row(tables: dict[str, pd.DataFrame], kind: str, package_file: str) -> dict[str, Any]:
    if package_file not in tables[kind].index:
        return {}
    value = tables[kind].loc[package_file]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[0]
    return value.to_dict()


def qut_telemetry(tables: dict[str, pd.DataFrame], package_file: str) -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "ips": set(), "ports": set(), "endpoints": set(), "states": set(), "processes": set(),
        "non_harness_processes": set(), "syscalls": set(), "patterns": set(), "file_buckets": set(),
    }
    tcp = _row(tables, "tcp", package_file)
    try:
        transitions = ast.literal_eval(clean(tcp.get("State_Transition")))
    except (SyntaxError, ValueError):
        transitions = {}
    if isinstance(transitions, dict):
        for raw_key in transitions:
            key = clean(raw_key)
            match = IP_PORT_RE.match(key)
            if match:
                telemetry["ips"].add(match.group("ip"))
                telemetry["ports"].add(match.group("port"))
                telemetry["endpoints"].add(f"{match.group('ip')}:{match.group('port')}")
                continue
            match = PORT_STATE_RE.match(key)
            if match:
                telemetry["ports"].add(match.group("port"))
                telemetry["states"].add(match.group("state"))

    filetop = _row(tables, "filetop", package_file)
    for column in ("Read_Processes", "Write_Processes", "File_Access_Processes"):
        telemetry["processes"].update(clean(v) for v in parse_listish(filetop.get(column)) if clean(v))
    harness_processes = {
        "pip", "pip3", "python", "python3", "sh", "bash", "dash", "tar", "gzip",
        "gcc", "cc1", "ld", "make", "chmod", "cp", "mv", "rm",
    }
    telemetry["non_harness_processes"].update(
        process for process in telemetry["processes"] if process.lower() not in harness_processes
    )

    syscall = _row(tables, "syscall", package_file)
    telemetry["syscalls"].update(
        clean(v) for v in parse_listish(syscall.get("Unique_System_Calls_List")) if clean(v)
    )

    pattern = _row(tables, "pattern", package_file)
    for index in range(1, 11):
        value = clean(pattern.get(f"Pattern_{index}"))
        if value:
            telemetry["patterns"].add(value)

    opensnoop = _row(tables, "opensnoop", package_file)
    buckets = {
        "Root_DIR_Installation": "ROOT_DIR",
        "Temporary_DIR_Installation": "TMP_DIR",
        "Home_DIR_Installation": "HOME_DIR",
        "Etc_DIR_Installation": "ETC_DIR",
        "Other_DIR_Installation": "OTHER_DIR",
    }
    for column, bucket in buckets.items():
        try:
            present = float(clean(opensnoop.get(column)) or 0) > 0
        except ValueError:
            present = False
        if present:
            telemetry["file_buckets"].add(bucket)
    return telemetry


def qut_capability(evidence: ExternalEvidence) -> str:
    if evidence.evidence_type in {"ip", "endpoint"}:
        return "observable"
    if evidence.evidence_type in {"file", "string", "payload_file"}:
        return "partial"
    if evidence.evidence_type in {"domain", "url"}:
        return "unobservable"
    return "unknown"


def match_qut_evidence(evidence: ExternalEvidence, telemetry: dict[str, Any]) -> tuple[str, str, str]:
    value = evidence.evidence_value.strip()
    capability = qut_capability(evidence)
    if evidence.evidence_type == "ip":
        return ("exact_ioc", value, "tcp_state_transition") if value in telemetry["ips"] else ("unmatched", "", "")
    if evidence.evidence_type == "endpoint":
        return ("exact_ioc", value, "tcp_state_transition") if value in telemetry["endpoints"] else ("unmatched", "", "")
    if evidence.evidence_type == "payload_file":
        matches = [process for process in telemetry["processes"] if process.lower() == value.lower()]
        return ("exact_ioc", matches[0], "process_name") if matches else ("ttp_annotation_required", "", "aggregated_process_names_only")
    if evidence.evidence_type == "url":
        host = normalize_domain(urlparse(value).hostname or "")
        # Processed QUT has no domains; never infer a domain from unrelated IPs.
        return "unobservable", host, "domain_not_retained"
    if evidence.evidence_type == "domain":
        return "unobservable", normalize_domain(value), "domain_not_retained"
    if capability == "partial":
        return "ttp_annotation_required", "", "aggregated_telemetry_only"
    return "not_applicable", "", ""


def preview(telemetry: dict[str, Any], key: str, limit: int = 50) -> str:
    return json.dumps(sorted(telemetry[key])[:limit], ensure_ascii=False)


def build_qut_pivot_gt_candidates(
    tables: dict[str, pd.DataFrame], evidence: list[ExternalEvidence]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    by_package: dict[str, list[ExternalEvidence]] = defaultdict(list)
    for item in evidence:
        if item.ecosystem == "pypi":
            by_package[item.package].append(item)
    known_packages = set(by_package)

    package_rows: list[dict[str, Any]] = []
    pivot_rows: list[dict[str, Any]] = []
    ttp_rows: list[dict[str, Any]] = []
    install = tables["install"]
    malicious = install[install["Level"].astype(str).isin({"1", "1.0"})]
    for package_file in malicious["Package_Name"].tolist():
        package, version, identity_status = align_qut_identity(package_file, known_packages)
        candidates = by_package.get(package, [])
        applicable = [item for item in candidates if version_compatibility(version, item) != "mismatch_or_unresolved"]
        telemetry = qut_telemetry(tables, package_file)
        observed_triggers = ("install",)
        observed_trigger_detail = json.dumps(
            {"install": {"source": "QUT-DV25_Install_Traces", "package_file": package_file}},
            ensure_ascii=False,
            sort_keys=True,
        )
        exact = 0
        reported = 0
        observable_unmatched = 0
        has_behavior = any(item.evidence_type == "behavior_event" for item in applicable)
        seen_ttp: set[tuple[str, str, str]] = set()
        for item in applicable:
            compatibility = version_compatibility(version, item)
            if item.evidence_type in {
                "ip", "domain", "url", "string", "file", "endpoint", "payload_file",
            }:
                reported += 1
                level, observed_value, observed_field = match_qut_evidence(item, telemetry)
                exact += int(level == "exact_ioc")
                observable_unmatched += int(qut_capability(item) == "observable" and level == "unmatched")
                pivot_rows.append({
                    "candidate_id": candidate_id(f"qut:{package_file}", item),
                    "dataset": "qut",
                    "qut_package_file": package_file,
                    "ecosystem": "pypi",
                    "package": package,
                    "version": version,
                    "identity_alignment": identity_status,
                    "package_label": "malicious",
                    "package_label_status": "trusted_qut_ground_truth",
                    **asdict(item),
                    "affected_versions": ";".join(item.affected_versions),
                    "version_compatibility": compatibility,
                    "telemetry_capability_auto": qut_capability(item),
                    "review_priority": review_priority(item, compatibility),
                    "match_level": level,
                    "observed_value": observed_value,
                    "observed_field": observed_field,
                    "observed_trigger": "install",
                    "observed_trigger_evidence": observed_trigger_detail,
                    "trigger_match_auto": compare_trigger(item.trigger, observed_triggers),
                    "trace_ips_preview": preview(telemetry, "ips"),
                    "trace_ports_preview": preview(telemetry, "ports"),
                    "trace_endpoints_preview": preview(telemetry, "endpoints"),
                    "trace_processes_preview": preview(telemetry, "processes"),
                    "trace_non_harness_processes": preview(telemetry, "non_harness_processes"),
                    "trace_syscalls_preview": preview(telemetry, "syscalls"),
                    "trace_patterns_preview": preview(telemetry, "patterns", 20),
                    "trace_file_buckets": preview(telemetry, "file_buckets"),
                })
            elif item.evidence_type == "behavior_event" and (
                item.source_dataset, item.source_id, item.evidence_value
            ) not in seen_ttp:
                seen_ttp.add((item.source_dataset, item.source_id, item.evidence_value))
                behavior_level, behavior_observation = match_behavior(item.evidence_value, telemetry)
                ttp_rows.append({
                    "candidate_id": candidate_id(f"qut:{package_file}", item),
                    "dataset": "qut",
                    "qut_package_file": package_file,
                    "ecosystem": "pypi",
                    "package": package,
                    "version": version,
                    "identity_alignment": identity_status,
                    "package_label": "malicious",
                    "package_label_status": "trusted_qut_ground_truth",
                    "source_dataset": item.source_dataset,
                    "source_id": item.source_id,
                    "source_reference": item.source_reference,
                    "affected_versions": ";".join(item.affected_versions),
                    "version_compatibility": compatibility,
                    "reported_behavior_type": item.evidence_value,
                    "reported_behavior": item.expected_behavior,
                    "behavior_extraction": item.extraction,
                    "behavior_confidence": item.evidence_confidence,
                    "attribution_scope": item.attribution_scope,
                    "behavior_match_auto": behavior_level,
                    "observed_behavior_evidence": behavior_observation,
                    "reported_trigger": item.trigger,
                    "trigger_evidence": item.trigger_evidence,
                    "trigger_source_reference": item.trigger_source_reference,
                    "trigger_extraction": item.trigger_extraction,
                    "trigger_confidence": item.trigger_confidence,
                    "observed_trigger": "install",
                    "observed_trigger_evidence": observed_trigger_detail,
                    "trigger_match_auto": compare_trigger(item.trigger, observed_triggers),
                    "trace_ips_preview": preview(telemetry, "ips"),
                    "trace_ports_preview": preview(telemetry, "ports"),
                    "trace_endpoints_preview": preview(telemetry, "endpoints"),
                    "trace_processes_preview": preview(telemetry, "processes"),
                    "trace_non_harness_processes": preview(telemetry, "non_harness_processes"),
                    "trace_syscalls_preview": preview(telemetry, "syscalls"),
                    "trace_patterns_preview": preview(telemetry, "patterns", 20),
                    "trace_file_buckets": preview(telemetry, "file_buckets"),
                })

        if exact:
            status = "exact_reported_pivot_auto_matched"
        elif observable_unmatched:
            status = "reported_observable_pivot_not_auto_matched"
        elif reported:
            status = "reported_pivot_requires_coarse_review"
        elif has_behavior:
            status = "ttp_annotation_required"
        else:
            status = "pivot_gt_unavailable"
        reported_triggers = sorted({item.trigger for item in applicable if item.trigger != "unknown"})
        trigger_matches = sorted({compare_trigger(item.trigger, observed_triggers) for item in applicable})
        package_rows.append({
            "dataset": "qut",
            "qut_package_file": package_file,
            "ecosystem": "pypi",
            "package": package,
            "version": version,
            "identity_alignment": identity_status,
            "package_label": "malicious",
            "package_label_status": "trusted_qut_ground_truth",
            "external_candidate_records": len(candidates),
            "applicable_records": len(applicable),
            "reported_triggers": ";".join(reported_triggers),
            "reported_trigger_status": "known" if reported_triggers else "unknown" if applicable else "no_external_evidence",
            "observed_triggers": "install",
            "observed_trigger_evidence": observed_trigger_detail,
            "trigger_match_auto": ";".join(trigger_matches),
            "reported_pivot_candidates": reported,
            "exact_ioc_matches": exact,
            "observable_unmatched_candidates": observable_unmatched,
            "observed_ips": len(telemetry["ips"]),
            "observed_ports": len(telemetry["ports"]),
            "observed_processes": len(telemetry["processes"]),
            "observed_syscalls": len(telemetry["syscalls"]),
            "observed_patterns": len(telemetry["patterns"]),
            "observed_file_buckets": len(telemetry["file_buckets"]),
            "pivot_gt_status": status,
            "verification_status": "pending",
            "annotator": "",
            "annotator_note": "",
        })

    packages = pd.DataFrame(package_rows)
    pivots = pd.DataFrame(pivot_rows)
    ttps = pd.DataFrame(ttp_rows)
    summary = {
        "schema_version": 3,
        "dataset": "qut",
        "package_label_assumption": "QUT Level=1 labels are trusted ground truth",
        "warning": "QUT processed telemetry retains exact IPs but not domains, URLs, commands, or file paths",
        "qut_malicious_records": len(packages),
        "records_by_pivot_gt_status": packages.pivot_gt_status.value_counts().sort_index().to_dict(),
        "records_with_external_identity_match": int((packages.external_candidate_records > 0).sum()),
        "records_with_reported_trigger": int((packages.reported_trigger_status == "known").sum()),
        "records_with_observed_trigger": int((packages.observed_triggers != "").sum()),
        "records_with_auto_trigger_match": int(packages.trigger_match_auto.str.split(";").apply(lambda x: "match" in x).sum()),
        "records_with_reported_pivot_candidates": int((packages.reported_pivot_candidates > 0).sum()),
        "records_with_exact_ioc_match": int((packages.exact_ioc_matches > 0).sum()),
        "records_with_observed_ips": int((packages.observed_ips > 0).sum()),
        "records_with_observed_ports": int((packages.observed_ports > 0).sum()),
        "records_with_processes": int((packages.observed_processes > 0).sum()),
        "records_with_syscalls": int((packages.observed_syscalls > 0).sum()),
        "records_with_patterns": int((packages.observed_patterns > 0).sum()),
        "records_with_file_buckets": int((packages.observed_file_buckets > 0).sum()),
        "reported_pivot_candidate_rows": len(pivots),
        "ttp_candidate_rows": len(ttps),
        "pivot_candidates_by_capability": pivots.telemetry_capability_auto.value_counts().sort_index().to_dict(),
        "pivot_candidates_by_type": pivots.evidence_type.value_counts().sort_index().to_dict(),
    }
    return packages, pivots, ttps, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qut-base", type=Path,
        default=Path("data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets"),
    )
    parser.add_argument("--malicious-info", type=Path, default=Path("data/malicious-packages-info"))
    parser.add_argument(
        "--backstabbers-index", type=Path,
        default=Path("data/Backstabbers-Knife-Collection/package_index.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("ground_truth/qut_pivot_gt"))
    args = parser.parse_args()

    evidence = load_external_evidence(args.malicious_info, args.backstabbers_index)
    outputs = build_qut_pivot_gt_candidates(load_qut_tables(args.qut_base), evidence)
    packages, pivots, ttps, summary = outputs
    review = build_review_queue(pivots, ttps, dataset="qut")
    path = write_review_file(args.out, review)
    print(json.dumps({**summary, "review_tasks": len(review), "review_file": str(path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
