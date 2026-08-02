"""Compose one compact, evidence-backed review queue per runtime dataset."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


REVIEW_COLUMNS = {
    "review_decision": "",
    "review_matching_event": "",
    "reviewer": "",
    "review_note": "",
    "review_status": "pending",
}


def _review_id(dataset: str, record_id: str, source_dataset: str, source_id: str) -> str:
    raw = "\0".join((dataset, record_id, source_dataset, source_id))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _osp_capability(evidence_type: str) -> str:
    if evidence_type in {"ip", "domain", "endpoint"}:
        return "observable"
    if evidence_type in {"url", "file", "string", "payload_file"}:
        return "partial"
    return "unknown"


def _assessment(levels: set[str]) -> str:
    if levels == {"exact_ioc"}:
        return "reported_ioc_observed"
    if "exact_ioc" in levels or "partial_ioc" in levels:
        return "reported_ioc_partially_observed"
    if levels == {"unmatched"}:
        return "reported_ioc_not_observed"
    return "inconclusive"


def _compact_values(values: pd.Series) -> str:
    return ";".join(sorted({str(value) for value in values if str(value)}))


def build_review_queue(
    pivots: pd.DataFrame, behaviors: pd.DataFrame, *, dataset: str,
) -> pd.DataFrame:
    """Group high-value IOC and behavior evidence by execution and source report."""
    if pivots.empty and behaviors.empty:
        return pd.DataFrame()
    record_column = "osp_record_id" if dataset == "osptrack" else "qut_package_file"
    if pivots.empty:
        candidates = pivots.copy()
    else:
        pivot_eligible = pivots.review_priority.eq("high")
        if dataset == "qut":
            pivot_eligible &= pivots.telemetry_capability_auto.eq("observable")
        selected_keys = {
            tuple(row) for row in pivots.loc[pivot_eligible, [record_column, "source_dataset", "source_id"]].to_numpy()
        }
        candidates = pivots[
            pivots[[record_column, "source_dataset", "source_id"]].apply(tuple, axis=1).isin(selected_keys)
        ].copy()
    if behaviors.empty:
        behavior_candidates = behaviors.copy()
    else:
        behavior_eligible = (
            behaviors.behavior_confidence.eq("high")
            & behaviors.version_compatibility.eq("exact")
            & behaviors.trigger_match_auto.eq("match")
        )
        behavior_candidates = behaviors[behavior_eligible].copy()
    rows: list[dict[str, Any]] = []
    keys = [record_column, "source_dataset", "source_id"]
    pivot_groups = (
        {key: group for key, group in candidates.groupby(keys, sort=True)} if not candidates.empty else {}
    )
    behavior_groups = (
        {key: group for key, group in behavior_candidates.groupby(keys, sort=True)}
        if not behavior_candidates.empty else {}
    )
    for record_id, source_dataset, source_id in sorted(set(pivot_groups) | set(behavior_groups)):
        key = (record_id, source_dataset, source_id)
        group = pivot_groups.get(key, pd.DataFrame())
        behavior_group = behavior_groups.get(key, pd.DataFrame())
        first = group.iloc[0] if not group.empty else behavior_group.iloc[0]
        items = []
        capabilities = []
        for item in (
            group.sort_values(["evidence_type", "evidence_value"]).to_dict("records")
            if not group.empty else []
        ):
            capability = (
                item.get("telemetry_capability_auto", "")
                if dataset == "qut"
                else _osp_capability(item["evidence_type"])
            )
            capabilities.append(capability)
            items.append({
                "candidate_id": item["candidate_id"],
                "type": item["evidence_type"],
                "value": item["evidence_value"],
                "attribution_scope": item.get("attribution_scope", "unknown"),
                "telemetry_capability": capability,
                "match_level": item["match_level"],
                "observed_value": item.get("observed_value", ""),
                "observed_field": item.get("observed_field", ""),
            })
        behavior_items = []
        for item in (
            behavior_group.sort_values("reported_behavior_type").to_dict("records")
            if not behavior_group.empty else []
        ):
            behavior_items.append({
                "type": item["reported_behavior_type"],
                "attribution_scope": item.get("attribution_scope", "unknown"),
                "report_evidence": item["reported_behavior"],
                "match_level": item["behavior_match_auto"],
                "observed_evidence": item["observed_behavior_evidence"],
            })

        levels = set(group.match_level) if not group.empty else set()
        ioc_assessment = _assessment(levels) if levels else "no_concrete_ioc"
        behavior_levels = set(behavior_group.behavior_match_auto) if not behavior_group.empty else set()
        behavior_assessment = ";".join(sorted(behavior_levels)) if behavior_levels else "no_reported_behavior"
        reasons = []
        combined = pd.concat([group, behavior_group], ignore_index=True, sort=False)
        if not combined.version_compatibility.eq("exact").all():
            reasons.append("version applicability is not an exact external-source match")
        if not combined.trigger_match_auto.eq("match").all():
            reasons.append("reported and observed triggers are not conclusively matched")
        if not combined.trigger_confidence.eq("high").all():
            reasons.append("reported-trigger provenance is not high confidence")
        if not group.empty and not group.extraction.isin({"structured", "derived_structured_url"}).all():
            reasons.append("one or more IOCs were not structurally extracted")
        if (
            not group.empty
            and "attribution_scope" in group
            and group.attribution_scope.eq("record_attached").any()
            and ioc_assessment != "reported_ioc_observed"
        ):
            reasons.append("record-attached IOC applicability to this exact execution is not independently proven")
        if capabilities and not all(capability == "observable" for capability in capabilities):
            reasons.append("one or more IOC types are only partially observable in this dataset")
        if behavior_levels and behavior_levels != {"behavior_observed"}:
            reasons.append("reported behavior is only partially observed, absent, or semantically unobservable")
        confidence = "low" if reasons else "high"
        ioc_complete = ioc_assessment in {"no_concrete_ioc", "reported_ioc_observed"}
        behavior_complete = behavior_assessment in {"no_reported_behavior", "behavior_observed"}
        manual_required = not (ioc_complete and behavior_complete and confidence == "high")
        if ioc_assessment == "reported_ioc_not_observed":
            review_reason = "Confirm the missing reported IOC and the associated report-backed behavior evidence."
        elif ioc_assessment == "reported_ioc_partially_observed":
            review_reason = "Confirm the partial IOC match and its associated behavior sequence."
        elif behavior_levels and behavior_levels != {"behavior_observed"}:
            review_reason = "Confirm whether the reported command/process/file/network behavior occurred."
        elif ioc_assessment == "reported_ioc_observed" or behavior_assessment == "behavior_observed":
            review_reason = "Automatic evidence is strong; optional spot-check only." if not manual_required else "Confirm the observed runtime pivot."
        else:
            review_reason = "Resolve conflicting or incomplete automatic evidence."
        basis = (
            f"External IOC extraction={_compact_values(group.extraction) if not group.empty else 'none'}; "
            f"IOC attribution={_compact_values(group.attribution_scope) if not group.empty and 'attribution_scope' in group else 'unknown'}; "
            f"behavior extraction={_compact_values(behavior_group.behavior_extraction) if not behavior_group.empty else 'none'}; "
            f"behavior attribution={_compact_values(behavior_group.attribution_scope) if not behavior_group.empty and 'attribution_scope' in behavior_group else 'unknown'}; "
            f"version={_compact_values(combined.version_compatibility)}; "
            f"reported trigger={getattr(first, 'trigger', getattr(first, 'reported_trigger', 'unknown'))} ({first.trigger_confidence}); "
            f"observed trigger={first.observed_trigger or 'none'}; "
            f"trigger comparison={_compact_values(combined.trigger_match_auto)}; "
            f"IOC result={ioc_assessment}; behavior result={behavior_assessment}."
        )
        if reasons:
            basis += " Low-confidence reasons: " + "; ".join(reasons) + "."

        trace_columns = [column for column in combined.columns if column.startswith("trace_")]
        reported_trigger = getattr(first, "trigger", getattr(first, "reported_trigger", "unknown"))
        scope = "both" if items and behavior_items else "artifact_ioc" if items else "behavior_event"
        row = {
            "review_id": _review_id(dataset, str(record_id), source_dataset, source_id),
            "dataset": dataset,
            "record_id": record_id,
            "ecosystem": first.ecosystem,
            "package": first.package,
            "version": first.version,
            "source_dataset": source_dataset,
            "source_id": source_id,
            "source_reference": first.source_reference,
            "affected_versions": first.affected_versions,
            "version_compatibility": _compact_values(combined.version_compatibility),
            "evidence_scope": scope,
            "reported_trigger": reported_trigger,
            "trigger_evidence": first.trigger_evidence,
            "trigger_source_reference": first.trigger_source_reference,
            "trigger_confidence": first.trigger_confidence,
            "observed_trigger": first.observed_trigger,
            "observed_trigger_evidence": first.observed_trigger_evidence,
            "trigger_match_auto": _compact_values(combined.trigger_match_auto),
            "reported_ioc_count": len(items),
            "reported_iocs": json.dumps(items, ensure_ascii=False, sort_keys=True),
            "reported_behavior_count": len(behavior_items),
            "reported_behaviors": json.dumps(behavior_items, ensure_ascii=False, sort_keys=True),
            "auto_ioc_assessment": ioc_assessment,
            "auto_behavior_assessment": behavior_assessment,
            "confidence": confidence,
            "manual_review_required": "yes" if manual_required else "no",
            "review_reason": review_reason,
            "determination_basis": basis,
            **{column: first.get(column, "") for column in trace_columns},
            **REVIEW_COLUMNS,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    confidence_order = pd.Categorical(frame.confidence, categories=["low", "high"], ordered=True)
    return frame.assign(_confidence_order=confidence_order).sort_values(
        ["manual_review_required", "_confidence_order", "package", "version", "record_id"],
        ascending=[False, True, True, True, True],
        ignore_index=True,
    ).drop(columns="_confidence_order")
