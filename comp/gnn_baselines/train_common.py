from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


def read_list(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p.as_posix())
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def iter_labeled_graphs(ds: Dataset):
    """Yield `HeteroData` examples from a dataset or `random_split` subset."""
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        for idx in ds.indices:
            yield base[idx]
        return
    for i in range(len(ds)):
        yield ds[i]


def collect_hetero_metadata(*datasets: Dataset) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Union node/edge types across datasets for HAN/HGT materialization."""
    node_types: set[str] = set()
    edge_types: set[tuple[str, str, str]] = set()
    for ds in datasets:
        for data in iter_labeled_graphs(ds):
            if not isinstance(data, HeteroData):
                continue
            node_types.update(data.node_types)
            edge_types.update(data.edge_types)
    return sorted(node_types), sorted(edge_types)


def class_weight_from_dataset(train_ds: Dataset, *, device: torch.device) -> Optional[torch.Tensor]:
    ys: list[int] = []
    for i in range(len(train_ds)):
        y = int(train_ds[i]["y"].item())
        if y != -1:
            ys.append(y)
    if not ys:
        return None
    n0 = sum(1 for y in ys if y == 0)
    n1 = sum(1 for y in ys if y == 1)
    w0 = (n0 + n1) / max(1, n0)
    w1 = (n0 + n1) / max(1, n1)
    return torch.tensor([w0, w1], dtype=torch.float, device=device)


def parse_ks(ks: str) -> list[int]:
    out: list[int] = []
    for part in str(ks).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return sorted({k for k in out if k > 0})


@torch.no_grad()
def evaluate_classifier_metrics(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
) -> dict:
    """Return classification metrics on labeled test samples."""
    model.eval()
    all_y, all_pred, all_prob1 = [], [], []
    for batch in loader:
        batch = batch.to(device)
        y = batch["y"].view(-1)
        logits = model(batch)
        prob1 = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)
        mask = y != -1
        if mask.sum().item() == 0:
            continue
        all_y.append(y[mask].detach().cpu())
        all_pred.append(pred[mask].detach().cpu())
        all_prob1.append(prob1[mask].detach().cpu())

    if not all_y:
        return {"test_n": 0}

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_prob1 = torch.cat(all_prob1).numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
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

    return {
        "test_n": int(len(y_true)),
        "acc": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auroc": auroc,
        "auprc": auprc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob1": y_prob1,
    }


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    ks: Sequence[int],
) -> None:
    metrics = evaluate_classifier_metrics(model, loader, device=device)
    if metrics.get("test_n", 0) == 0:
        print("test: no labeled samples (all y == -1)")
        return

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    y_prob1 = metrics["y_prob1"]
    print(
        f"test_n={metrics['test_n']} acc={metrics['acc']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} auroc={metrics['auroc']} "
        f"auprc={metrics['auprc']}"
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"confusion_matrix [[tn, fp],[fn, tp]] = [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"pred_pos={int((y_pred == 1).sum())} pred_neg={int((y_pred == 0).sum())}")

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
        print(f"P@{k}={tp_at_k / kk:.4f} R@{k}={tp_at_k / pos_total:.4f} (tp@{k}={tp_at_k}, pos_total={pos_total})")
