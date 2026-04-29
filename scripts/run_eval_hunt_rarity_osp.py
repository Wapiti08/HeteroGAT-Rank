from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--val-list", type=str, default="splits_full/osp_train.txt")
    ap.add_argument("--test-list", type=str, default="splits_full/osp_test.txt")
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--explainer-ckpt", type=str, required=True)
    ap.add_argument("--rarity-stats", type=str, required=True)
    ap.add_argument("--rarity-normalize", type=str, default="osp")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--op-fprs", type=str, default="0.01,0.05,0.10")
    ap.add_argument("--timeout-s", type=int, default=0)

    # A small but useful grid for OSP: focus on NET types and compare to ALL.
    ap.add_argument("--etypes", type=str, default="NET_ONLY;CONNECT_ONLY;ALL")
    ap.add_argument("--lambdas", type=str, default="0.05,0.1,0.3")
    ap.add_argument("--idf-caps", type=str, default="1,2,3")
    ap.add_argument("--dry-run", action="store_true", default=False)

    args = ap.parse_args()

    # Default filters tuned for OSP noise.
    filter_args = (
        "--filter-net --filter-net-ip "
        "--filter-system-files --filter-tmp-tempfile --filter-cmd-noise "
        "--dedup-dst --max-per-etype 8"
    )

    cmd = [
        sys.executable,
        "scripts/run_eval_hunt_rarity_grid.py",
        "--name",
        "osp_hunt_rarity",
        "--out-dir",
        str(args.out_dir),
        "--val-list",
        str(args.val_list),
        "--test-list",
        str(args.test_list),
        "--backbone-ckpt",
        str(args.backbone_ckpt),
        "--explainer-ckpt",
        str(args.explainer_ckpt),
        "--rarity-stats",
        str(args.rarity_stats),
        "--rarity-normalize",
        str(args.rarity_normalize),
        "--device",
        str(args.device),
        "--k",
        str(int(args.k)),
        "--op-fprs",
        str(args.op_fprs),
        "--filter-args",
        filter_args,
        "--etypes",
        str(args.etypes),
        "--lambdas",
        str(args.lambdas),
        "--idf-caps",
        str(args.idf_caps),
        "--timeout-s",
        str(int(args.timeout_s)),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    raise SystemExit(__import__("subprocess").run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()

