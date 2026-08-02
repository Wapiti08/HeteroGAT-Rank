"""Command-line composition root for OSPTrack pivot auditing."""

import argparse
import json
from pathlib import Path

from .osptrack_builder import build_pivot_gt_candidates
from .reporting import write_review_file
from .review_queue import build_review_queue
from .sources import load_external_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osp-csv", type=Path, default=Path("data/OSPTrack/label_data.csv"))
    parser.add_argument("--malicious-info", type=Path, default=Path("data/malicious-packages-info"))
    parser.add_argument("--backstabbers-index", type=Path, default=Path("data/Backstabbers-Knife-Collection/package_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("ground_truth/osptrack_pivot_gt"))
    parser.add_argument("--chunksize", type=int, default=1000)
    args = parser.parse_args()
    evidence = load_external_evidence(args.malicious_info, args.backstabbers_index)
    outputs = build_pivot_gt_candidates(args.osp_csv, evidence, chunksize=args.chunksize)
    packages, pivots, ttps, summary = outputs
    review = build_review_queue(pivots, ttps, dataset="osptrack")
    path = write_review_file(args.out, review)
    print(json.dumps({**summary, "review_tasks": len(review), "review_file": str(path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
