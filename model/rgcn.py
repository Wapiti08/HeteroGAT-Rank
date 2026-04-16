from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool


class RGCNGraphClassifier(nn.Module):
    """Simple, strong baseline: R-GCN on homogeneous view of canonical graphs.

    Designed to be *clean* and comparable:
    - No accelerate/DDP logic
    - No explanation/ranking state
    - Works even when node features are missing by using node-type embeddings
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # We embed node_type ids (produced by `to_homogeneous(add_node_type=True)`).
        # Size is set lazily once we see max node_type in the first forward.
        self.node_type_emb: Optional[nn.Embedding] = None

        # RGCN layers are also created lazily because num_relations depends on edge types present.
        self.convs: nn.ModuleList = nn.ModuleList()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._num_relations: int = 0

    def materialize(self, *, num_node_types: int, num_relations: int, device: torch.device) -> None:
        """Create dynamic modules with explicit sizes (for checkpoint loading)."""
        self.node_type_emb = nn.Embedding(num_node_types, self.hidden_dim).to(device)
        self._num_relations = int(num_relations)
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=self._num_relations).to(device))
        for _ in range(self.num_layers - 1):
            self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=self._num_relations).to(device))

    def _ensure_modules(self, num_node_types: int, num_relations: int, in_dim: int) -> None:
        if self.node_type_emb is None:
            self.node_type_emb = nn.Embedding(num_node_types, self.hidden_dim)
        elif self.node_type_emb.num_embeddings < num_node_types:
            # Expand embedding table to accommodate new node types across batches.
            old = self.node_type_emb
            new = nn.Embedding(num_node_types, self.hidden_dim).to(old.weight.device)
            with torch.no_grad():
                new.weight[: old.num_embeddings].copy_(old.weight)
            self.node_type_emb = new

        if len(self.convs) == 0:
            # First layer maps from in_dim to hidden_dim. In our case in_dim==hidden_dim.
            self.convs.append(RGCNConv(in_dim, self.hidden_dim, num_relations=num_relations))
            for _ in range(self.num_layers - 1):
                self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=num_relations))

    def materialize(self, *, num_node_types: int, num_relations: int, device: torch.device | None = None) -> None:
        """Materialize lazy modules for checkpoint loading.

        Some scripts load checkpoints before any forward pass. This helper ensures
        embeddings / conv layers exist with the expected shapes.
        """
        if device is None:
            device = next(self.parameters(), torch.empty(0)).device  # type: ignore[arg-type]
        self._ensure_modules(int(num_node_types), int(num_relations), in_dim=self.hidden_dim)
        self._num_relations = int(num_relations)
        self.to(device)

    def encode_homogeneous(self, data, edge_index_override=None, edge_type_override=None):
        """Encode a homogeneous PyG `Data` produced by `to_homogeneous`.

        Returns:
            x: node embeddings [N, hidden_dim]
            data: same data object (for access to `batch`, `edge_type`, etc.)
        """
        edge_index = edge_index_override if edge_index_override is not None else data.edge_index

        # Build features: if `x` missing, use node-type embedding only.
        node_type = data.node_type
        num_node_types = int(node_type.max().item()) + 1 if node_type.numel() else 1
        edge_type = edge_type_override if edge_type_override is not None else data.edge_type
        num_relations = int(edge_type.max().item()) + 1 if edge_type.numel() else 1

        self._ensure_modules(num_node_types, num_relations, in_dim=self.hidden_dim)
        assert self.node_type_emb is not None

        # Ensure modules live on the same device as the incoming batch.
        dev = node_type.device
        if self.node_type_emb.weight.device != dev:
            self.node_type_emb = self.node_type_emb.to(dev)
        for i, conv in enumerate(self.convs):
            self.convs[i] = conv.to(dev)

        x = self.node_type_emb(node_type)

        # If relations grow across batches, RGCNConv needs to be resized.
        # For baseline stability, rebuild conv stack when num_relations increases.
        if getattr(self, "_num_relations", 0) < num_relations:
            self._num_relations = num_relations
            self.convs = nn.ModuleList()
            self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=num_relations).to(dev))
            for _ in range(self.num_layers - 1):
                self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=num_relations).to(dev))

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x, data

    def forward(self, hetero_batch, *, edge_index_override=None, edge_type_override=None, return_node_emb=False):
        # Convert to homogeneous graph (keeps node_type and edge_type tensors).
        data = hetero_batch.to_homogeneous(add_node_type=True, add_edge_type=True)
        x, data = self.encode_homogeneous(
            data,
            edge_index_override=edge_index_override,
            edge_type_override=edge_type_override,
        )

        # graph pooling
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = global_mean_pool(x, batch)
        logits = self.classifier(g)
        if return_node_emb:
            return logits, x, data
        return logits

