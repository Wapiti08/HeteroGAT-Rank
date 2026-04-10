from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

# Ensure repo root importable when running as script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from comp.gnn_baselines.dataset import CanonicalGraphDataset  # noqa: E402
from model.rgcn import RGCNGraphClassifier  # noqa: E402
from ranking_explain.pgexplainer import PGExplainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", default=["artifacts/qut_small", "artifacts/osp_small"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--sparsity", type=float, default=0.05, help="Target keep ratio (e.g., 0.05 keeps ~5% edges)")
    ap.add_argument("--sparsity-coef", type=float, default=1.0, help="Penalty weight for matching sparsity target")
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--backbone-ckpt", type=str, default="", help="Optional: load backbone checkpoint (.pt)")
    ap.add_argument("--save-ckpt", type=str, default="", help="Optional: save explainer checkpoint (.pt)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CanonicalGraphDataset(args.graphs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    backbone = RGCNGraphClassifier(hidden_dim=64, num_layers=2, dropout=0.2, num_classes=2).to(device)
    if args.backbone_ckpt:
        ckpt = torch.load(args.backbone_ckpt, map_location=device)
        schema = ckpt.get("schema", {})
        nnt = int(schema.get("num_node_types", 0))
        nr = int(schema.get("num_relations", 0))
        if nnt > 0 and nr > 0:
            backbone.materialize(num_node_types=nnt, num_relations=nr, device=device)
        backbone.load_state_dict(ckpt["state_dict"])
        backbone.eval()

    explainer = PGExplainer(
        hidden_dim=64,
        temperature=args.temp,
        sparsity_target=args.sparsity,
        sparsity_coef=args.sparsity_coef,
        entropy_coef=args.entropy,
    ).to(device)
    opt = torch.optim.Adam(explainer.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        explainer.train()
        tot = 0.0
        for batch in loader:
            batch = batch.to(device)
            out = explainer(backbone=backbone, hetero_batch=batch)
            loss = out["loss"].total
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        print(f"epoch {epoch} explainer_loss={tot/max(1,len(loader)):.4f}")

    if args.save_ckpt:
        p = Path(args.save_ckpt)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "pgexplainer",
                "explainer_kwargs": {
                    "hidden_dim": 64,
                    "temperature": args.temp,
                    "sparsity_target": args.sparsity,
                    "sparsity_coef": args.sparsity_coef,
                    "entropy_coef": args.entropy,
                },
                "state_dict": explainer.state_dict(),
            },
            p,
        )
        print(f"saved_ckpt={p.as_posix()}")


if __name__ == "__main__":
    main()

