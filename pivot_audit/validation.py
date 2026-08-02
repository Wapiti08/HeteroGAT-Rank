#!/usr/bin/env python3
"""Validate the compact dataset review queue and optionally export verified rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ALLOWED_ANNOTATIONS = {
    "confidence": {"high", "low"},
    "manual_review_required": {"yes", "no"},
    "review_decision": {
        "confirmed", "partially_confirmed", "not_observed", "not_applicable", "inconclusive",
    },
    "review_status": {"pending", "verified", "needs_second_review"},
}


def load_annotations(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    if "review_id" not in frame:
        raise ValueError("Missing required column: review_id")
    duplicates = frame[frame.review_id.duplicated(keep=False)]
    if not duplicates.empty:
        ids = ", ".join(sorted(duplicates.review_id.unique())[:10])
        raise ValueError(f"Duplicate review_id values across inputs: {ids}")
    return frame


def validate_annotations(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column, allowed in ALLOWED_ANNOTATIONS.items():
        if column not in frame:
            errors.append(f"missing column {column}")
            continue
        invalid = frame.loc[(frame[column] != "") & ~frame[column].isin(allowed), ["review_id", column]]
        for row in invalid.head(20).itertuples(index=False):
            errors.append(f"{row.review_id}: invalid {column}={getattr(row, column)!r}")
    if errors:
        return errors

    for row in frame[frame.review_status == "verified"].itertuples(index=False):
        if not row.review_decision.strip():
            errors.append(f"{row.review_id}: verified row is missing review_decision")
        if not row.reviewer.strip():
            errors.append(f"{row.review_id}: verified row is missing reviewer")
        if row.review_decision in {"confirmed", "partially_confirmed", "not_observed"} and not row.review_matching_event.strip():
            errors.append(f"{row.review_id}: verified decision is missing review_matching_event")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Annotated review.tsv files")
    parser.add_argument("--out", type=Path, help="Optional verified-row export; no file is created by default")
    args = parser.parse_args()
    frame = load_annotations(args.inputs)
    errors = validate_annotations(frame)
    if errors:
        raise SystemExit("Annotation validation failed:\n- " + "\n- ".join(errors))

    verified = frame[frame.review_status == "verified"].copy()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        verified.to_csv(args.out, sep="\t", index=False)
    summary = {
        "input_rows": len(frame),
        "verified_rows": len(verified),
        "verified_decisions": verified.review_decision.value_counts().sort_index().to_dict(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
