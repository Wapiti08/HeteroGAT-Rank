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
from model.rgcn import RGCNGraphClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", default=["artifacts/qut_small", "artifacts/osp_small"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CanonicalGraphDataset(args.graphs)
    n = len(ds)
    n_train = max(1, int(0.8 * n))
    n_test = n - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = RGCNGraphClassifier(hidden_dim=64, num_layers=2, dropout=0.2, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            y = batch["y"].view(-1)  # [B]
            logits = model(batch)  # [B,2]
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch} train_loss={total/max(1,len(train_loader)):.4f}")

    # quick eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            y = batch["y"].view(-1)
            pred = model(batch).argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    acc = correct / max(1, total)
    print(f"test_acc={acc:.4f} (n={total})")


if __name__ == "__main__":
    main()

