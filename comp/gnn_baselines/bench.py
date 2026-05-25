from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader

from comp.gnn_baselines.dataset import CanonicalGraphDataset
from comp.gnn_baselines.train_common import (
    class_weight_from_dataset,
    collect_hetero_metadata,
    evaluate_classifier_metrics,
    read_list,
)
from graph.canonical_features import CANONICAL_NODE_FEAT_DIM
from model.gat import GATHomGraphClassifier
from model.han import HANGraphClassifier
from model.hgt import HGTGraphClassifier, load_hgt_graph_classifier
from model.node_encoder import CanonicalNodeEncoder
from model.rgcn import RGCNGraphClassifier, load_rgcn_graph_classifier


BACKBONE_DEFAULT_EPOCHS: dict[str, int] = {
    "rgcn": 20,
    "gat": 20,
    "han": 5,
    "hgt": 5,
}


@dataclass
class BackboneBenchResult:
    backbone: str
    epochs: int
    train_time_s: float
    infer_mean_ms: float
    infer_p50_ms: float
    infer_p90_ms: float
    params: int
    trainable_params: int
    auroc: Optional[float]
    f1: Optional[float]
    test_n: int
    train_n: int
    device: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return total, trainable


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_datasets(
    *,
    graphs: list[str],
    train_list: str,
    test_list: str,
    use_node_features: bool,
    seed: int,
) -> tuple[Dataset, Dataset]:
    train_files = read_list(train_list)
    test_files = read_list(test_list)
    if train_files and test_files:
        train_ds = CanonicalGraphDataset(graph_files=train_files, use_node_features=use_node_features)
        test_ds = CanonicalGraphDataset(graph_files=test_files, use_node_features=use_node_features)
        return train_ds, test_ds

    ds = CanonicalGraphDataset(graphs, use_node_features=use_node_features)
    n = len(ds)
    n_train = max(1, int(0.8 * n))
    n_test = n - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(ds, [n_train, n_test], generator=gen)
    return train_ds, test_ds


def _build_model(
    backbone: str,
    *,
    device: torch.device,
    train_ds: Dataset,
    test_ds: Dataset,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    edge_dim: int,
) -> nn.Module:
    name = backbone.lower()
    if name == "rgcn":
        return RGCNGraphClassifier(hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.2, num_classes=2).to(
            device
        )
    if name == "gat":
        return GATHomGraphClassifier(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=edge_dim,
            dropout=0.2,
            num_classes=2,
        ).to(device)
    if name == "han":
        model = HANGraphClassifier(hidden_dim=hidden_dim, heads=heads, dropout=0.2, num_classes=2).to(device)
        metadata = collect_hetero_metadata(train_ds, test_ds)
        model.materialize(metadata=metadata, device=device)
        return model
    if name == "hgt":
        model = HGTGraphClassifier(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=0.2,
            num_classes=2,
        ).to(device)
        metadata = collect_hetero_metadata(train_ds, test_ds)
        model.materialize(metadata=metadata, device=device)
        return model
    raise ValueError(f"Unknown backbone: {backbone!r}")


def _load_gat_checkpoint(path: Path, *, device: torch.device) -> GATHomGraphClassifier:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kwargs = ckpt.get(
        "model_kwargs",
        {"hidden_dim": 64, "num_layers": 2, "heads": 4, "dropout": 0.2, "num_classes": 2, "edge_dim": 16},
    )
    model = GATHomGraphClassifier(**kwargs).to(device)
    schema = ckpt.get("schema", {})
    nnt = int(schema.get("num_node_types", 0))
    nr = int(schema.get("num_relations", 0))
    if nnt > 0 and nr > 0:
        model.materialize(num_node_types=nnt, num_relations=nr, device=device)
    state = ckpt["state_dict"]
    if any(k.startswith("node_encoder.") for k in state):
        model.node_encoder = CanonicalNodeEncoder(model.hidden_dim).to(device)
        model.node_encoder._ensure(
            num_node_types=max(nnt, 1),
            in_dim=CANONICAL_NODE_FEAT_DIM,
            device=device,
        )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _load_han_checkpoint(
    path: Path,
    *,
    device: torch.device,
    metadata: tuple[list[str], list[tuple[str, str, str]]],
) -> HANGraphClassifier:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kwargs = ckpt.get("model_kwargs", {"hidden_dim": 64, "heads": 4, "dropout": 0.2, "num_classes": 2})
    model = HANGraphClassifier(**kwargs).to(device)
    model.materialize(metadata=metadata, device=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model


def load_backbone_checkpoint(
    backbone: str,
    path: Path,
    *,
    device: torch.device,
    metadata: Optional[tuple[list[str], list[tuple[str, str, str]]]] = None,
) -> nn.Module:
    name = backbone.lower()
    if name == "rgcn":
        return load_rgcn_graph_classifier(path, device=device)
    if name == "gat":
        return _load_gat_checkpoint(path, device=device)
    if name == "han":
        if metadata is None:
            raise ValueError("HAN checkpoint load requires hetero metadata from train/test data")
        return _load_han_checkpoint(path, device=device, metadata=metadata)
    if name == "hgt":
        return load_hgt_graph_classifier(path, device=device)
    raise ValueError(f"Unknown backbone: {backbone!r}")


def train_backbone(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    reweight: bool,
    train_ds: Dataset,
) -> float:
    class_weight = class_weight_from_dataset(train_ds, device=device) if reweight else None
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.perf_counter()
    for _epoch in range(1, epochs + 1):
        model.train()
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
    _cuda_sync(device)
    return float(time.perf_counter() - t0)


@torch.no_grad()
def bench_infer_ms_per_graph(
    model: nn.Module,
    test_ds: Dataset,
    *,
    device: torch.device,
    warmup: int = 50,
    include_transfer: bool = False,
) -> tuple[float, float, float, int]:
    """Mean / p50 / p90 forward latency (ms) with batch_size=1 on the test set."""
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    latencies: list[float] = []
    n_total = len(loader)
    n_warm = min(max(0, warmup), max(0, n_total - 1))

    def _forward_ms(batch) -> float:
        if include_transfer:
            batch = batch.to(device)
            _cuda_sync(device)
            t0 = time.perf_counter()
            _ = model(batch)
            _cuda_sync(device)
            return (time.perf_counter() - t0) * 1000.0
        batch = batch.to(device)
        _ = model(batch)
        _cuda_sync(device)
        t0 = time.perf_counter()
        _ = model(batch)
        _cuda_sync(device)
        return (time.perf_counter() - t0) * 1000.0

    for i, batch in enumerate(loader):
        ms = _forward_ms(batch)
        if i >= n_warm:
            latencies.append(ms)

    if not latencies:
        return float("nan"), float("nan"), float("nan"), 0

    arr = np.asarray(latencies, dtype=float)
    return float(arr.mean()), float(np.quantile(arr, 0.5)), float(np.quantile(arr, 0.9)), int(len(arr))


def run_backbone_benchmark(
    backbone: str,
    *,
    graphs: list[str],
    train_list: str,
    test_list: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    reweight: bool,
    use_node_features: bool,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    edge_dim: int,
    warmup: int,
    include_transfer: bool,
    ckpt_path: Optional[Path] = None,
    skip_train: bool = False,
) -> BackboneBenchResult:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = _build_datasets(
        graphs=graphs,
        train_list=train_list,
        test_list=test_list,
        use_node_features=use_node_features,
        seed=seed,
    )
    metadata = collect_hetero_metadata(train_ds, test_ds) if backbone.lower() in ("han", "hgt") else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if ckpt_path is not None and skip_train:
        model = load_backbone_checkpoint(
            backbone,
            ckpt_path,
            device=device,
            metadata=metadata,
        )
        train_time_s = float("nan")
    else:
        model = _build_model(
            backbone,
            device=device,
            train_ds=train_ds,
            test_ds=test_ds,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=edge_dim,
        )
        train_time_s = float("nan")
        if not skip_train:
            train_time_s = train_backbone(
                model,
                train_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                reweight=reweight,
                train_ds=train_ds,
            )

    params, trainable_params = count_parameters(model)
    infer_mean, infer_p50, infer_p90, _infer_n = bench_infer_ms_per_graph(
        model,
        test_ds,
        device=device,
        warmup=warmup,
        include_transfer=include_transfer,
    )
    metrics = evaluate_classifier_metrics(model, test_loader, device=device)

    return BackboneBenchResult(
        backbone=backbone.upper(),
        epochs=int(epochs),
        train_time_s=float(train_time_s),
        infer_mean_ms=float(infer_mean),
        infer_p50_ms=float(infer_p50),
        infer_p90_ms=float(infer_p90),
        params=int(params),
        trainable_params=int(trainable_params),
        auroc=metrics.get("auroc"),
        f1=metrics.get("f1") if metrics.get("test_n", 0) else None,
        test_n=int(metrics.get("test_n", 0)),
        train_n=int(len(train_ds)),
        device=str(device),
    )
