from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Ensure repo root importable when running as script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from model.rgcn import RGCNGraphClassifier  # noqa: E402
from ranking_explain.pgexplainer import PGExplainer  # noqa: E402
from ranking_explain.hunt import topk_edges  # noqa: E402


def load_one_graph(path: Path) -> HeteroData:
    obj = torch.load(path, map_location="cpu")
    return HeteroData.from_dict(obj["data_dict"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=str, required=True, help="Path to *.graph.pt")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.graph)
    data = load_one_graph(p)
    data = data.to("cpu")

    backbone = RGCNGraphClassifier(hidden_dim=64, num_layers=2, dropout=0.2, num_classes=2)
    explainer = PGExplainer(hidden_dim=64)

    ranked = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=args.k)
    print(f"graph={p.name} topk={len(ranked)}")
    for r in ranked[: args.k]:
        print(f"{r.score:.4f}\t{r.etype}\t{r.src_type}:{r.src_key} -> {r.dst_type}:{r.dst_key}")


if __name__ == "__main__":
    main()

