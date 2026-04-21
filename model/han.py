from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, global_mean_pool


class HANGraphClassifier(nn.Module):
    """HAN baseline on canonical `HeteroData`.

    Uses learnable per-node-type embeddings as initial features.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.num_classes = num_classes

        self.node_type_emb: Optional[nn.Embedding] = None
        self.conv: Optional[HANConv] = None
        self._metadata = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _ensure_modules(self, *, metadata, device: torch.device) -> None:
        if self.node_type_emb is None:
            self.node_type_emb = nn.Embedding(len(metadata[0]), self.hidden_dim).to(device)
        if self.conv is None:
            # HANConv expects (in_channels, out_channels) and uses meta-path attention.
            self.conv = HANConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                metadata=metadata,
                heads=self.heads,
                dropout=self.dropout,
            ).to(device)
            self._metadata = metadata

    def materialize(self, *, metadata, device: torch.device | None = None) -> None:
        if device is None:
            device = next(self.parameters(), torch.empty(0)).device  # type: ignore[arg-type]
        self._ensure_modules(metadata=metadata, device=device)
        self.to(device)

    def forward(self, hetero_batch) -> torch.Tensor:
        metadata = hetero_batch.metadata()
        dev = hetero_batch.edge_index_dict[next(iter(hetero_batch.edge_index_dict))].device if hetero_batch.edge_types else next(self.parameters()).device
        self._ensure_modules(metadata=metadata, device=dev)
        assert self.node_type_emb is not None and self.conv is not None

        x_dict: Dict[str, torch.Tensor] = {}
        for i, nt in enumerate(metadata[0]):
            n = int(hetero_batch[nt].num_nodes or 0)
            if n <= 0:
                continue
            x0 = self.node_type_emb.weight[i].unsqueeze(0).expand(n, -1)
            x_dict[nt] = x0

        x_dict = self.conv(x_dict, hetero_batch.edge_index_dict)
        # HANConv may return None for node types that have no valid meta-path output
        # in the current batch. Filter them out (their pooled contribution becomes 0).
        x_dict = {k: v for k, v in x_dict.items() if v is not None}
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        pooled = []
        bsz = int(hetero_batch["y"].view(-1).size(0)) if "y" in hetero_batch else None
        for nt, x in x_dict.items():
            batch = getattr(hetero_batch[nt], "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pooled.append(global_mean_pool(x, batch, size=bsz))
        if not pooled:
            g = torch.zeros((int(bsz or 1), self.hidden_dim), device=dev)
        else:
            g = torch.stack(pooled, dim=0).mean(dim=0)
        return self.classifier(g)

