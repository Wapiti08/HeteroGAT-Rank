from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from graph.canonical_features import CANONICAL_NODE_FEAT_DIM


class CanonicalNodeEncoder(nn.Module):
    """Project deterministic canonical node features + type id to hidden dim."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_type_emb: Optional[nn.Embedding] = None
        self.proj: Optional[nn.Linear] = None

    def _ensure(self, *, num_node_types: int, in_dim: int, device: torch.device) -> None:
        if self.node_type_emb is None:
            self.node_type_emb = nn.Embedding(num_node_types, self.hidden_dim).to(device)
        elif self.node_type_emb.num_embeddings < num_node_types:
            old = self.node_type_emb
            new = nn.Embedding(num_node_types, self.hidden_dim).to(device)
            with torch.no_grad():
                new.weight[: old.num_embeddings].copy_(old.weight)
            self.node_type_emb = new

        if self.proj is None or self.proj.in_features != in_dim:
            self.proj = nn.Linear(in_dim, self.hidden_dim).to(device)

    def forward(self, x_raw: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        dev = x_raw.device
        num_node_types = int(node_type.max().item()) + 1 if node_type.numel() else 1
        in_dim = int(x_raw.size(-1))
        self._ensure(num_node_types=num_node_types, in_dim=in_dim, device=dev)
        assert self.node_type_emb is not None and self.proj is not None
        return self.proj(x_raw) + self.node_type_emb(node_type)
