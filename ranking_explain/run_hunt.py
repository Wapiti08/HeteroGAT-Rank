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

def _load_backbone(path: str, *, device: torch.device) -> RGCNGraphClassifier:
    ckpt = torch.load(path, map_location=device)
    kwargs = ckpt.get("model_kwargs", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2, "num_classes": 2})
    model = RGCNGraphClassifier(**kwargs).to(device)
    schema = ckpt.get("schema", {})
    nnt = int(schema.get("num_node_types", 0))
    nr = int(schema.get("num_relations", 0))
    if nnt > 0 and nr > 0:
        model.materialize(num_node_types=nnt, num_relations=nr, device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _load_explainer(path: str, *, device: torch.device) -> PGExplainer:
    ckpt = torch.load(path, map_location=device)
    kwargs = ckpt.get("explainer_kwargs", {"hidden_dim": 64})
    explainer = PGExplainer(**kwargs).to(device)
    explainer.load_state_dict(ckpt["state_dict"])
    explainer.eval()
    return explainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=str, required=True, help="Path to *.graph.pt")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--backbone-ckpt", type=str, default="", help="Backbone checkpoint (.pt)")
    ap.add_argument("--explainer-ckpt", type=str, default="", help="Explainer checkpoint (.pt)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    args = ap.parse_args()

    p = Path(args.graph)
    data = load_one_graph(p)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = data.to(device)

    backbone = (
        _load_backbone(args.backbone_ckpt, device=device)
        if args.backbone_ckpt
        else RGCNGraphClassifier(hidden_dim=64, num_layers=2, dropout=0.2, num_classes=2).to(device)
    )
    explainer = _load_explainer(args.explainer_ckpt, device=device) if args.explainer_ckpt else PGExplainer(hidden_dim=64).to(device)

    ranked = topk_edges(backbone=backbone, explainer=explainer, hetero_batch=data, k=args.k)
    print(f"graph={p.name} topk={len(ranked)}")
    for r in ranked[: args.k]:
        print(f"{r.score:.4f}\t{r.etype}\t{r.src_type}:{r.src_key} -> {r.dst_type}:{r.dst_key}")


if __name__ == "__main__":
    main()

