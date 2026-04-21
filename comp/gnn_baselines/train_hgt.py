from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score

# Ensure repo root importable when running as script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from comp.gnn_baselines.dataset import CanonicalGraphDataset  # noqa: E402
from model.hgt import HGTGraphClassifier  # noqa: E402


def _read_list(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p.as_posix())
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _save_ckpt(*, path: Path, model: HGTGraphClassifier) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "hgt_graph_classifier",
            "model_kwargs": {
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "heads": model.heads,
                "dropout": model.dropout,
                "num_classes": model.num_classes,
            },
            "state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=str, nargs="+", default=["artifacts/qut_small", "artifacts/osp_small"])
    ap.add_argument("--train-list", type=str, default="", help="Optional: path to train *.graph.pt list")
    ap.add_argument("--test-list", type=str, default="", help="Optional: path to test *.graph.pt list")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reweight", action="store_true", default=False)
    ap.add_argument("--ks", type=str, default="5,10,20")
    ap.add_argument("--save-ckpt", type=str, default="")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = _read_list(args.train_list)
    test_files = _read_list(args.test_list)
    if train_files and test_files:
        train_ds = CanonicalGraphDataset(graph_files=train_files)
        test_ds = CanonicalGraphDataset(graph_files=test_files)
    else:
        ds = CanonicalGraphDataset(args.graphs)
        n = len(ds)
        n_train = max(1, int(0.8 * n))
        n_test = n - n_train
        train_ds, test_ds = random_split(ds, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = HGTGraphClassifier(
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        heads=int(args.heads),
        dropout=0.2,
        num_classes=2,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_weight = None
    if args.reweight:
        ys = []
        for i in range(len(train_ds)):
            y = int(train_ds[i]["y"].item())
            if y != -1:
                ys.append(y)
        if ys:
            n0 = sum(1 for y in ys if y == 0)
            n1 = sum(1 for y in ys if y == 1)
            w0 = (n0 + n1) / max(1, n0)
            w1 = (n0 + n1) / max(1, n1)
            class_weight = torch.tensor([w0, w1], dtype=torch.float, device=device)
            print(f"class_weight={class_weight.tolist()} (n0={n0}, n1={n1})")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            y = batch["y"].view(-1)
            logits = model(batch)
            if (y != -1).sum().item() == 0:
                continue
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch} train_loss={total/max(1,len(train_loader)):.4f}")

    if args.save_ckpt:
        _save_ckpt(path=Path(args.save_ckpt), model=model)
        print(f"saved_ckpt={args.save_ckpt}")

    model.eval()
    all_y, all_pred, all_prob1 = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            y = batch["y"].view(-1)
            logits = model(batch)
            prob1 = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(dim=-1)
            mask = (y != -1)
            if mask.sum().item() == 0:
                continue
            all_y.append(y[mask].detach().cpu())
            all_pred.append(pred[mask].detach().cpu())
            all_prob1.append(prob1[mask].detach().cpu())
    if not all_y:
        print("test: no labeled samples (all y == -1)")
        return
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_prob1 = torch.cat(all_prob1).numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    auroc = None
    auprc = None
    try:
        auroc = float(roc_auc_score(y_true, y_prob1))
    except Exception:
        pass
    try:
        auprc = float(average_precision_score(y_true, y_prob1))
    except Exception:
        pass
    print(f"test_n={len(y_true)} acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f} auroc={auroc} auprc={auprc}")
    print(f"confusion_matrix [[tn, fp],[fn, tp]] = [[{tn}, {fp}], [{fn}, {tp}]]")

    ks: list[int] = []
    for part in str(args.ks).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ks.append(int(part))
        except Exception:
            pass
    ks = sorted(set(k for k in ks if k > 0))
    pos_total = int((y_true == 1).sum())
    if pos_total == 0:
        print("ranking: no positive samples in test; skip Precision@K/Recall@K")
        return
    order = y_prob1.argsort()[::-1]
    y_sorted = y_true[order]
    for k in ks:
        kk = min(k, len(y_sorted))
        if kk <= 0:
            continue
        tp_at_k = int((y_sorted[:kk] == 1).sum())
        print(f"P@{k}={tp_at_k/kk:.4f} R@{k}={tp_at_k/pos_total:.4f} (tp@{k}={tp_at_k}, pos_total={pos_total})")


if __name__ == "__main__":
    main()

