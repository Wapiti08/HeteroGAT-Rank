from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# Ensure repo root importable when running as script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from comp.gnn_baselines.dataset import CanonicalGraphDataset
from comp.gnn_baselines.train_common import (
    class_weight_from_dataset,
    evaluate_classifier,
    parse_ks,
    read_list,
)
from model.rgcn import RGCNGraphClassifier

def _save_rgcn_checkpoint(*, path: Path, model: RGCNGraphClassifier) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_node_types = int(model.node_type_emb.num_embeddings) if model.node_type_emb is not None else 0
    num_relations = int(getattr(model, "_num_relations", 0))
    torch.save(
        {
            "kind": "rgcn_graph_classifier",
            "model_kwargs": {
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
                "num_classes": model.num_classes,
            },
            "schema": {"num_node_types": num_node_types, "num_relations": num_relations},
            "state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", default=["artifacts/qut_small", "artifacts/osp_small"])
    ap.add_argument("--train-list", type=str, default="", help="Optional: path to train *.graph.pt list")
    ap.add_argument("--test-list", type=str, default="", help="Optional: path to test *.graph.pt list")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--reweight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weights in CE (default: on)",
    )
    ap.add_argument(
        "--no-node-features",
        action="store_true",
        default=False,
        help="Disable per-node canonical features (type-only inputs)",
    )
    ap.add_argument(
        "--ks",
        type=str,
        default="5,10,20",
        help="Comma-separated list of K for Precision@K / Recall@K on test set",
    )
    ap.add_argument("--save-ckpt", type=str, default="", help="Optional: path to save model checkpoint (.pt)")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_node_features = not bool(args.no_node_features)
    train_files = read_list(args.train_list)
    test_files = read_list(args.test_list)
    if train_files and test_files:
        train_ds = CanonicalGraphDataset(graph_files=train_files, use_node_features=use_node_features)
        test_ds = CanonicalGraphDataset(graph_files=test_files, use_node_features=use_node_features)
    else:
        ds = CanonicalGraphDataset(args.graphs, use_node_features=use_node_features)
        n = len(ds)
        n_train = max(1, int(0.8 * n))
        n_test = n - n_train
        train_ds, test_ds = random_split(ds, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    model = RGCNGraphClassifier(hidden_dim=64, num_layers=2, dropout=0.2, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_weight = class_weight_from_dataset(train_ds, device=device) if args.reweight else None
    if class_weight is not None:
        print(f"class_weight={class_weight.tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            y = batch["y"].view(-1)  # [B]
            logits = model(batch)  # [B,2]
            # Skip batches with no labeled samples.
            if (y != -1).sum().item() == 0:
                continue
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch} train_loss={total/max(1,len(train_loader)):.4f}")

    if args.save_ckpt:
        _save_rgcn_checkpoint(path=Path(args.save_ckpt), model=model)
        print(f"saved_ckpt={args.save_ckpt}")

    evaluate_classifier(model, test_loader, device=device, ks=parse_ks(args.ks))


if __name__ == "__main__":
    main()

