from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATHomGraphClassifier(nn.Module):
    """Homogeneous GAT baseline on canonical graphs.

    We convert `HeteroData` to a homogeneous view and use:
    - node features: learnable embedding of `node_type`
    - edge features: learnable embedding of `edge_type` (passed as edge_attr)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 2,
        edge_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.edge_dim = edge_dim

        self.node_type_emb: Optional[nn.Embedding] = None
        self.edge_type_emb: Optional[nn.Embedding] = None
        self.convs: nn.ModuleList = nn.ModuleList()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._num_node_types: int = 0
        self._num_relations: int = 0

    def _ensure_tables(self, *, num_node_types: int, num_relations: int, device: torch.device) -> None:
        if self.node_type_emb is None:
            self.node_type_emb = nn.Embedding(num_node_types, self.hidden_dim).to(device)
        elif self.node_type_emb.num_embeddings < num_node_types:
            old = self.node_type_emb
            new = nn.Embedding(num_node_types, self.hidden_dim).to(device)
            with torch.no_grad():
                new.weight[: old.num_embeddings].copy_(old.weight)
            self.node_type_emb = new

        if self.edge_type_emb is None:
            self.edge_type_emb = nn.Embedding(num_relations, self.edge_dim).to(device)
        elif self.edge_type_emb.num_embeddings < num_relations:
            old = self.edge_type_emb
            new = nn.Embedding(num_relations, self.edge_dim).to(device)
            with torch.no_grad():
                new.weight[: old.num_embeddings].copy_(old.weight)
            self.edge_type_emb = new

        if len(self.convs) == 0:
            # Keep output dim == hidden_dim (use concat=False so heads are averaged).
            self.convs.append(
                GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=self.heads,
                    concat=False,
                    dropout=self.dropout,
                    add_self_loops=False,
                    edge_dim=self.edge_dim,
                ).to(device)
            )
            for _ in range(self.num_layers - 1):
                self.convs.append(
                    GATConv(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        heads=self.heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=self.edge_dim,
                    ).to(device)
                )

        self._num_node_types = max(self._num_node_types, int(num_node_types))
        self._num_relations = max(self._num_relations, int(num_relations))

    def materialize(self, *, num_node_types: int, num_relations: int, device: torch.device | None = None) -> None:
        if device is None:
            device = next(self.parameters(), torch.empty(0)).device  # type: ignore[arg-type]
        self._ensure_tables(num_node_types=int(num_node_types), num_relations=int(num_relations), device=device)
        self.to(device)

    def forward(self, hetero_batch) -> torch.Tensor:
        data = hetero_batch.to_homogeneous(add_node_type=True, add_edge_type=True)
        dev = data.edge_index.device

        node_type = data.node_type
        edge_type = data.edge_type
        num_node_types = int(node_type.max().item()) + 1 if node_type.numel() else 1
        num_relations = int(edge_type.max().item()) + 1 if edge_type.numel() else 1

        self._ensure_tables(num_node_types=num_node_types, num_relations=num_relations, device=dev)
        assert self.node_type_emb is not None and self.edge_type_emb is not None

        x = self.node_type_emb(node_type)
        edge_attr = self.edge_type_emb(edge_type)

        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = global_mean_pool(x, batch)
        return self.classifier(g)

