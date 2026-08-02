"""Orchestrates OSPTrack pivot candidate construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .candidates import candidate_id, review_priority
from .behaviors import match_behavior
from .matching import match_osptrack_evidence
from .normalization import clean, normalize_ecosystem, normalize_package, version_compatibility
from .osptrack import (
    behavior_activity, merge_phase_telemetry, observed_trigger_evidence,
    telemetry_by_phase_from_row, telemetry_preview,
)
from .schema import ExternalEvidence
from .triggers import compare_trigger


def build_pivot_gt_candidates(
    osp_csv: Path, evidence: list[ExternalEvidence], *, chunksize: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    by_package: dict[tuple[str, str], list[ExternalEvidence]] = defaultdict(list)
    for item in evidence:
        by_package[(item.ecosystem, item.package)].append(item)
    columns = [
        "Unnamed: 0", "Ecosystem", "Name", "Version", "Label", "Sub_Label",
        "import_Files", "import_Sockets", "import_Commands", "import_DNS",
        "install_Files", "install_Sockets", "install_Commands", "install_DNS",
    ]
    packages, pivot_candidates, ttp_candidates = [], [], []
    for chunk in pd.read_csv(osp_csv, usecols=columns, dtype=str, chunksize=chunksize, keep_default_na=False):
        for row in chunk.to_dict("records"):
            if clean(row.get("Label")) not in {"1", "1.0", "true", "True"}:
                continue
            ecosystem, package = normalize_ecosystem(row.get("Ecosystem")), normalize_package(row.get("Name"))
            version, record_id = clean(row.get("Version")), clean(row.get("Unnamed: 0"))
            phase_telemetry = telemetry_by_phase_from_row(row)
            telemetry = merge_phase_telemetry(phase_telemetry)
            observed_triggers, observed_trigger_detail = observed_trigger_evidence(phase_telemetry)
            observed_trigger_text = ";".join(observed_triggers)
            candidates = by_package.get((ecosystem, package), [])
            applicable = [e for e in candidates if version_compatibility(version, e) != "mismatch_or_unresolved"]
            exact = partial = reported = unmatched = 0
            has_behavior = any(item.evidence_type == "behavior_event" for item in applicable)
            sources, seen_ttp = set(), set()
            for item in applicable:
                compatibility = version_compatibility(version, item)
                level, observed_value, observed_field = match_osptrack_evidence(item, telemetry)
                is_pivot = item.evidence_type in {
                    "ip", "domain", "url", "string", "file", "endpoint", "payload_file",
                }
                reported += int(is_pivot); exact += int(level == "exact_ioc"); partial += int(level == "partial_ioc")
                unmatched += int(is_pivot and level == "unmatched")
                sources.add(item.source_dataset)
                if is_pivot:
                    pivot_candidates.append({
                        "candidate_id": candidate_id(record_id, item), "ecosystem": ecosystem,
                        "package": package, "version": version, "osp_record_id": record_id,
                        "package_label": "malicious", "package_label_status": "trusted_osptrack_ground_truth",
                        **asdict(item), "affected_versions": ";".join(item.affected_versions),
                        "version_compatibility": compatibility, "review_priority": review_priority(item, compatibility),
                        "match_level": level, "observed_value": observed_value, "observed_field": observed_field,
                        "observed_trigger": observed_trigger_text,
                        "observed_trigger_evidence": observed_trigger_detail,
                        "trigger_match_auto": compare_trigger(item.trigger, observed_triggers),
                        "trace_ips_preview": telemetry_preview(telemetry, "ips", 50),
                        "trace_domains_preview": telemetry_preview(telemetry, "domains", 50),
                        "trace_commands_preview": telemetry_preview(telemetry, "commands", 20),
                        "trace_non_harness_commands": telemetry_preview(telemetry, "non_harness_commands", 20),
                        "trace_processes_preview": telemetry_preview(telemetry, "processes", 20),
                        "trace_ports_preview": telemetry_preview(telemetry, "ports", 20),
                        "trace_endpoints_preview": telemetry_preview(telemetry, "endpoints", 20),
                        "trace_files_preview": telemetry_preview(telemetry, "files", 20),
                        "trace_file_reads_preview": telemetry_preview(telemetry, "file_reads", 20),
                        "trace_file_writes_preview": telemetry_preview(telemetry, "file_writes", 20),
                    })
                elif item.evidence_type == "behavior_event" and (
                    item.source_dataset, item.source_id, item.evidence_value
                ) not in seen_ttp:
                    seen_ttp.add((item.source_dataset, item.source_id, item.evidence_value))
                    behavior_level, behavior_observation = match_behavior(item.evidence_value, telemetry)
                    ttp_candidates.append({
                        "candidate_id": candidate_id(record_id, item), "osp_record_id": record_id,
                        "ecosystem": ecosystem, "package": package, "version": version,
                        "package_label": "malicious", "package_label_status": "trusted_osptrack_ground_truth",
                        "source_dataset": item.source_dataset, "source_id": item.source_id,
                        "source_reference": item.source_reference, "affected_versions": ";".join(item.affected_versions),
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
                        "observed_trigger": observed_trigger_text,
                        "observed_trigger_evidence": observed_trigger_detail,
                        "trigger_match_auto": compare_trigger(item.trigger, observed_triggers),
                        "observed_activity_families": behavior_activity(telemetry),
                        "trace_ips_preview": telemetry_preview(telemetry, "ips", 50),
                        "trace_domains_preview": telemetry_preview(telemetry, "domains", 50),
                        "trace_commands_preview": telemetry_preview(telemetry, "commands", 20),
                        "trace_non_harness_commands": telemetry_preview(telemetry, "non_harness_commands", 20),
                        "trace_processes_preview": telemetry_preview(telemetry, "processes", 20),
                        "trace_ports_preview": telemetry_preview(telemetry, "ports", 20),
                        "trace_endpoints_preview": telemetry_preview(telemetry, "endpoints", 20),
                        "trace_files_preview": telemetry_preview(telemetry, "files", 20),
                        "trace_file_reads_preview": telemetry_preview(telemetry, "file_reads", 20),
                        "trace_file_writes_preview": telemetry_preview(telemetry, "file_writes", 20),
                    })
            if exact:
                status = "exact_reported_pivot_auto_matched"
            elif partial:
                status = "partial_reported_pivot_auto_matched"
            elif reported:
                status = "reported_pivot_not_auto_matched"
            elif has_behavior:
                status = "ttp_annotation_required"
            elif candidates and not applicable:
                status = "pivot_version_unresolved"
            else:
                status = "pivot_gt_unavailable"
            reported_triggers = sorted({item.trigger for item in applicable if item.trigger != "unknown"})
            trigger_matches = sorted({compare_trigger(item.trigger, observed_triggers) for item in applicable})
            packages.append({
                "ecosystem": ecosystem, "package": package, "version": version, "osp_record_id": record_id,
                "package_label": "malicious", "package_label_status": "trusted_osptrack_ground_truth",
                "sub_label": clean(row.get("Sub_Label")), "external_sources": ";".join(sorted(sources)),
                "candidate_records": len(candidates), "applicable_records": len(applicable),
                "reported_triggers": ";".join(reported_triggers),
                "reported_trigger_status": "known" if reported_triggers else "unknown" if applicable else "no_external_evidence",
                "observed_triggers": observed_trigger_text,
                "observed_trigger_evidence": observed_trigger_detail,
                "trigger_match_auto": ";".join(trigger_matches),
                "reported_pivot_candidates": reported, "exact_ioc_matches": exact,
                "partial_ioc_matches": partial, "unmatched_iocs": unmatched,
                "observed_activity": behavior_activity(telemetry), "observed_ips": len(telemetry["ips"]),
                "observed_domains": len(telemetry["domains"]), "observed_commands": len(telemetry["commands"]),
                "observed_files": len(telemetry["files"]), "pivot_gt_status": status,
                "verification_status": "pending", "annotator": "", "annotator_note": "",
            })
    package_frame, pivot_frame, ttp_frame = map(pd.DataFrame, (packages, pivot_candidates, ttp_candidates))
    summary = {
        "schema_version": 3,
        "package_label_assumption": "OSPTrack malicious labels are trusted ground truth",
        "warning": "reported_pivot_not_auto_matched is not a confirmed telemetry gap; manual verification is required",
        "osp_malicious_records": len(package_frame),
        "packages_by_ecosystem": package_frame.ecosystem.value_counts().sort_index().to_dict(),
        "records_by_pivot_gt_status": package_frame.pivot_gt_status.value_counts().sort_index().to_dict(),
        "packages_with_external_name_match": int((package_frame.candidate_records > 0).sum()),
        "packages_with_version_applicable_evidence": int((package_frame.applicable_records > 0).sum()),
        "records_with_reported_trigger": int((package_frame.reported_trigger_status == "known").sum()),
        "records_with_observed_trigger": int((package_frame.observed_triggers != "").sum()),
        "records_with_auto_trigger_match": int(package_frame.trigger_match_auto.str.split(";").apply(lambda x: "match" in x).sum()),
        "records_with_reported_pivot_candidates": int((package_frame.reported_pivot_candidates > 0).sum()),
        "packages_with_exact_ioc_match": int((package_frame.exact_ioc_matches > 0).sum()),
        "packages_with_partial_ioc_match": int((package_frame.partial_ioc_matches > 0).sum()),
        "external_evidence_records": len(evidence), "reported_pivot_candidate_rows": len(pivot_frame),
        "ttp_candidate_rows": len(ttp_frame),
        "pivot_candidates_by_type": pivot_frame.evidence_type.value_counts().sort_index().to_dict(),
        "pivot_candidates_by_review_priority": pivot_frame.review_priority.value_counts().sort_index().to_dict(),
    }
    return package_frame, pivot_frame, ttp_frame, summary


audit = build_pivot_gt_candidates
