from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn

from model.hgt import HGTGraphClassifier, load_hgt_graph_classifier
from model.rgcn import RGCNGraphClassifier, load_rgcn_graph_classifier


def load_graph_classifier(
    path: Union[str, Path],
    *,
    device: torch.device,
    weights_only: bool = False,
) -> nn.Module:
    """Load a graph classifier checkpoint written by ``comp/gnn_baselines/train_*.py``."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=weights_only)
    kind = str(ckpt.get("kind", "")).strip().lower()
    if not kind:
        kwargs = ckpt.get("model_kwargs", {})
        if "heads" in kwargs:
            kind = "hgt_graph_classifier"
        else:
            kind = "rgcn_graph_classifier"

    if kind in ("rgcn_graph_classifier", "rgcn"):
        return load_rgcn_graph_classifier(path, device=device, weights_only=weights_only)
    if kind in ("hgt_graph_classifier", "hgt"):
        return load_hgt_graph_classifier(path, device=device, weights_only=weights_only)

    raise ValueError(f"Unsupported backbone checkpoint kind={kind!r} ({path})")
