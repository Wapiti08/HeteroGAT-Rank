#!/usr/bin/env python3
"""Benchmark QUT-DV25 GNN backbones: train time, inference ms/graph, params, AUROC/F1."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

import torch

from comp.gnn_baselines.bench import (
    BACKBONE_DEFAULT_EPOCHS,
    BackboneBenchResult,
    run_backbone_benchmark,
)

TSV_COLUMNS = [
    "backbone",
    "epochs",
    "train_time_s",
    "train_time_min",
    "infer_mean_ms",
    "infer_p50_ms",
    "infer_p90_ms",
    "params",
    "trainable_params",
    "auroc",
    "f1",
    "test_n",
    "train_n",
    "device",
]


def _parse_backbones(value: str) -> list[str]:
    v = value.strip().lower()
    if v in ("all", "*"):
        return list(BACKBONE_DEFAULT_EPOCHS.keys())
    return [b.strip().lower() for b in v.split(",") if b.strip()]


def _epochs_for_backbone(name: str, args: argparse.Namespace) -> int:
    if args.epochs is not None:
        return int(args.epochs)
    per = getattr(args, f"{name}_epochs", None)
    if per is not None:
        return int(per)
    return int(BACKBONE_DEFAULT_EPOCHS[name])


def _row_from_result(r: BackboneBenchResult) -> dict:
    row = r.to_row()
    row["train_time_min"] = r.train_time_s / 60.0 if r.train_time_s == r.train_time_s else float("nan")
    return row


def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(TSV_COLUMNS)]
    for row in rows:
        cells = []
        for col in TSV_COLUMNS:
            val = row.get(col, "")
            if isinstance(val, float):
                if val != val:
                    cells.append("nan")
                elif col in ("auroc", "f1"):
                    cells.append(f"{val:.4f}")
                elif col == "train_time_min":
                    cells.append(f"{val:.2f}")
                elif col == "train_time_s":
                    cells.append(f"{val:.1f}")
                elif col.startswith("infer_"):
                    cells.append(f"{val:.2f}")
                else:
                    cells.append(f"{val:.4g}")
            else:
                cells.append(str(val))
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n")


def _print_latex_row(row: dict) -> None:
    name = {"RGCN": "R-GCN", "GAT": "GAT", "HAN": "HAN", "HGT": "HGT"}.get(row["backbone"], row["backbone"])
    tmin = row.get("train_time_min")
    train_s = "..." if tmin != tmin else f"{tmin:.1f}"
    infer_s = "..." if row.get("infer_mean_ms") != row.get("infer_mean_ms") else f"{row['infer_mean_ms']:.1f}"
    params_m = row.get("params", 0) / 1e6
    auroc = row.get("auroc")
    f1 = row.get("f1")
    auroc_s = "..." if auroc is None else f"{auroc:.4f}"
    f1_s = "..." if f1 is None else f"{f1:.4f}"
    print(f"{name} & {train_s} & {infer_s} & {params_m:.2f}M & {auroc_s} & {f1_s} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark QUT GNN backbones for efficiency table.")
    ap.add_argument("--graphs", type=str, nargs="+", default=["artifacts/qut_all"])
    ap.add_argument("--train-list", type=str, default="splits_full/qut_train.txt")
    ap.add_argument("--test-list", type=str, default="splits_full/qut_test.txt")
    ap.add_argument("--backbone", type=str, default="all", help="all or comma-separated: rgcn,gat,han,hgt")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs for all backbones")
    ap.add_argument("--rgcn-epochs", type=int, default=None)
    ap.add_argument("--gat-epochs", type=int, default=None)
    ap.add_argument("--han-epochs", type=int, default=None)
    ap.add_argument("--hgt-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reweight", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-node-features", action="store_true", default=False)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--edge-dim", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=50, help="Warmup graphs before timing inference")
    ap.add_argument(
        "--infer-include-transfer",
        action="store_true",
        help="Include host-to-device transfer in inference timing (default: forward only)",
    )
    ap.add_argument("--skip-train", action="store_true", help="Load checkpoint(s) and only measure infer/metrics")
    ap.add_argument("--rgcn-ckpt", type=str, default="")
    ap.add_argument("--gat-ckpt", type=str, default="")
    ap.add_argument("--han-ckpt", type=str, default="")
    ap.add_argument("--hgt-ckpt", type=str, default="")
    ap.add_argument("--out", type=str, default="ablation/qut_backbone_efficiency.tsv")
    ap.add_argument("--meta-out", type=str, default="", help="Optional JSON metadata path")
    ap.add_argument("--latex", action="store_true", help="Print LaTeX table rows to stdout")
    args = ap.parse_args()

    backbones = _parse_backbones(args.backbone)
    ckpt_map = {
        "rgcn": args.rgcn_ckpt,
        "gat": args.gat_ckpt,
        "han": args.han_ckpt,
        "hgt": args.hgt_ckpt,
    }
    use_node_features = not bool(args.no_node_features)

    rows: list[dict] = []
    for name in backbones:
        epochs = _epochs_for_backbone(name, args)
        ckpt_s = ckpt_map.get(name, "")
        ckpt_path = Path(ckpt_s) if ckpt_s else None
        if args.skip_train and (ckpt_path is None or not ckpt_path.exists()):
            raise SystemExit(f"--skip-train requires existing --{name}-ckpt")

        print(f"\n== backbone={name.upper()} epochs={epochs} ==")
        result = run_backbone_benchmark(
            name,
            graphs=list(args.graphs),
            train_list=args.train_list,
            test_list=args.test_list,
            epochs=epochs,
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            reweight=bool(args.reweight),
            use_node_features=use_node_features,
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            heads=int(args.heads),
            edge_dim=int(args.edge_dim),
            warmup=int(args.warmup),
            include_transfer=bool(args.infer_include_transfer),
            ckpt_path=ckpt_path,
            skip_train=bool(args.skip_train),
        )
        row = _row_from_result(result)
        rows.append(row)
        print(
            f"train_time={row['train_time_s']:.1f}s ({row['train_time_min']:.2f} min) "
            f"infer_mean={row['infer_mean_ms']:.2f}ms params={row['params']} "
            f"auroc={row.get('auroc')} f1={row.get('f1')}"
        )

    out_path = Path(args.out)
    _write_tsv(out_path, rows)
    print(f"\nwrote_tsv={out_path.as_posix()}")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "graphs": args.graphs,
        "train_list": args.train_list,
        "test_list": args.test_list,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "use_node_features": use_node_features,
        "infer_include_transfer": bool(args.infer_include_transfer),
        "warmup": args.warmup,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows": rows,
    }
    meta_path = Path(args.meta_out) if args.meta_out else out_path.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote_meta={meta_path.as_posix()}")

    if args.latex:
        print("\n% LaTeX rows (train min / infer ms / params M / AUROC / F1):")
        for row in rows:
            _print_latex_row(row)


if __name__ == "__main__":
    main()
