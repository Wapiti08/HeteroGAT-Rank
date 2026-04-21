from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, global_mean_pool


class HGTGraphClassifier(nn.Module):
    """HGT baseline on canonical `HeteroData`.

    Canonical graphs may not provide node features. We use a learnable
    per-node-type embedding as initial features.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.num_classes = num_classes

        # Metadata-dependent modules (constructed lazily after seeing first batch).
        self.node_type_emb: Optional[nn.Embedding] = None
        self.convs: nn.ModuleList = nn.ModuleList()
        self._node_types: list[str] = []
        self._edge_types: list[tuple[str, str, str]] = []

        # Pool per type then aggregate.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _ensure_modules(self, *, metadata, device: torch.device) -> None:
        node_types, edge_types = metadata
        # Stable ordering.
        node_types = list(node_types)
        edge_types = list(edge_types)

        if self.node_type_emb is None:
            self.node_type_emb = nn.Embedding(len(node_types), self.hidden_dim).to(device)
            self._node_types = node_types
            self._edge_types = edge_types

        if len(self.convs) == 0:
            # HGTConv uses per-type linear projections internally.
            self.convs.append(HGTConv(self.hidden_dim, self.hidden_dim, metadata=metadata, heads=self.heads).to(device))
            for _ in range(self.num_layers - 1):
                self.convs.append(HGTConv(self.hidden_dim, self.hidden_dim, metadata=metadata, heads=self.heads).to(device))

    def materialize(self, *, metadata, device: torch.device | None = None) -> None:
        if device is None:
            device = next(self.parameters(), torch.empty(0)).device  # type: ignore[arg-type]
        self._ensure_modules(metadata=metadata, device=device)
        self.to(device)

    def forward(self, hetero_batch) -> torch.Tensor:
        metadata = hetero_batch.metadata()
        dev = hetero_batch.edge_index_dict[next(iter(hetero_batch.edge_index_dict))].device if hetero_batch.edge_types else next(self.parameters()).device
        self._ensure_modules(metadata=metadata, device=dev)
        assert self.node_type_emb is not None

        # Build x_dict from node-type embeddings.
        x_dict: Dict[str, torch.Tensor] = {}
        for i, nt in enumerate(metadata[0]):
            n = int(hetero_batch[nt].num_nodes or 0)
            if n <= 0:
                continue
            x0 = self.node_type_emb.weight[i].unsqueeze(0).expand(n, -1)
            x_dict[nt] = x0

        for conv in self.convs:
            x_dict = conv(x_dict, hetero_batch.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        # Pool each node type separately and mean-aggregate to a graph embedding.
        pooled = []
        bsz = int(hetero_batch["y"].view(-1).size(0)) if "y" in hetero_batch else None
        for nt, x in x_dict.items():
            batch = getattr(hetero_batch[nt], "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pooled.append(global_mean_pool(x, batch, size=bsz))
        if not pooled:
            # Fallback: empty graph (should not happen in practice).
            g = torch.zeros((int(bsz or 1), self.hidden_dim), device=dev)
        else:
            g = torch.stack(pooled, dim=0).mean(dim=0)
        return self.classifier(g)

