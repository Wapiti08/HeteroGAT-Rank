from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, global_mean_pool

from graph.canonical_features import CANONICAL_NODE_FEAT_DIM, hetero_num_nodes, to_homogeneous_safe
from model.node_encoder import CanonicalNodeEncoder


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
        self.node_encoder: Optional[CanonicalNodeEncoder] = None
        self.convs: nn.ModuleList = nn.ModuleList()
        self._node_types: list[str] = []
        self._edge_types: list[tuple[str, str, str]] = []
        self._metadata = None
        self._pending_state: Optional[dict[str, torch.Tensor]] = None

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
            self._metadata = metadata

    def materialize(self, *, metadata, device: torch.device | None = None) -> None:
        if device is None:
            device = next(self.parameters(), torch.empty(0)).device  # type: ignore[arg-type]
        self._ensure_modules(metadata=metadata, device=device)
        self.to(device)

    def _ensure_node_encoder_from_state(
        self,
        state: dict[str, torch.Tensor],
        *,
        device: torch.device,
        num_node_types: int,
    ) -> None:
        if not any(k.startswith("node_encoder.") for k in state):
            return
        if self.node_encoder is None:
            self.node_encoder = CanonicalNodeEncoder(self.hidden_dim).to(device)
        self.node_encoder._ensure(
            num_node_types=max(int(num_node_types), 1),
            in_dim=CANONICAL_NODE_FEAT_DIM,
            device=device,
        )

    def _load_state_checked(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in HGT checkpoint: {unexpected}")
        allowed_missing = {"node_encoder.node_type_emb.weight", "node_encoder.proj.weight", "node_encoder.proj.bias"}
        extra_missing = [k for k in missing if k not in allowed_missing]
        if extra_missing:
            raise RuntimeError(f"Missing keys in HGT checkpoint: {extra_missing}")

    def _maybe_load_pending(self, hetero_batch) -> None:
        pending = getattr(self, "_pending_state", None)
        if pending is None:
            return
        metadata = hetero_batch.metadata()
        dev = (
            hetero_batch.edge_index_dict[next(iter(hetero_batch.edge_index_dict))].device
            if hetero_batch.edge_types
            else next(self.parameters()).device
        )
        self.materialize(metadata=metadata, device=dev)
        self._ensure_node_encoder_from_state(pending, device=dev, num_node_types=len(metadata[0]))
        self._load_state_checked(pending)
        self._pending_state = None

    @staticmethod
    def _hetero_edge_index_dict_from_hom_override(
        hetero_batch,
        *,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """Map PGExplainer's (possibly masked) homogeneous edges back to hetero stores."""
        hom = to_homogeneous_safe(hetero_batch, add_node_type=True, add_edge_type=True)
        n_hom = int(hom.edge_index.size(1))
        dev = hom.edge_index.device
        keep = torch.zeros(n_hom, dtype=torch.bool, device=dev)
        if n_hom > 0 and edge_index.numel() > 0:
            hom_src = hom.edge_index[0]
            hom_dst = hom.edge_index[1]
            hom_et = hom.edge_type
            for i in range(int(edge_index.size(1))):
                match = (
                    (hom_src == edge_index[0, i])
                    & (hom_dst == edge_index[1, i])
                    & (hom_et == edge_type[i])
                )
                keep |= match
        out: dict[tuple[str, str, str], torch.Tensor] = {}
        offset = 0
        for et in hetero_batch.edge_types:
            ei = hetero_batch[et].edge_index
            n_e = int(ei.size(1))
            if n_e <= 0:
                out[et] = ei
                continue
            out[et] = ei[:, keep[offset : offset + n_e]]
            offset += n_e
        if offset != n_hom:
            raise ValueError(
                f"homogeneous/hetero edge count mismatch while masking: hom={n_hom} hetero_sum={offset}"
            )
        return out

    @staticmethod
    def _x_dict_to_homogeneous(
        x_dict: Dict[str, torch.Tensor],
        *,
        node_types: list[str],
        hetero_batch,
        hidden_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for nt in node_types:
            if nt not in hetero_batch.node_types:
                continue
            n = hetero_num_nodes(hetero_batch, nt)
            if n <= 0:
                continue
            if nt in x_dict:
                parts.append(x_dict[nt])
            else:
                parts.append(torch.zeros(n, hidden_dim, device=device))
        if not parts:
            return torch.zeros(0, hidden_dim, device=device)
        return torch.cat(parts, dim=0)

    def forward(
        self,
        hetero_batch,
        *,
        edge_index_override=None,
        edge_type_override=None,
        return_node_emb: bool = False,
    ):
        self._maybe_load_pending(hetero_batch)
        metadata = self._metadata if getattr(self, "_metadata", None) is not None else hetero_batch.metadata()
        dev = (
            hetero_batch.edge_index_dict[next(iter(hetero_batch.edge_index_dict))].device
            if hetero_batch.edge_types
            else next(self.parameters()).device
        )
        self._ensure_modules(metadata=metadata, device=dev)
        assert self.node_type_emb is not None

        hom_data = None
        if return_node_emb:
            hom_data = to_homogeneous_safe(hetero_batch, add_node_type=True, add_edge_type=True)

        edge_index_dict = hetero_batch.edge_index_dict
        if edge_index_override is not None:
            if edge_type_override is None:
                raise ValueError("edge_type_override is required when edge_index_override is set")
            edge_index_dict = self._hetero_edge_index_dict_from_hom_override(
                hetero_batch,
                edge_index=edge_index_override,
                edge_type=edge_type_override,
            )

        # Build x_dict from node-type embeddings.
        x_dict: Dict[str, torch.Tensor] = {}
        for i, nt in enumerate(metadata[0]):
            if nt not in hetero_batch.node_types:
                continue
            n = hetero_num_nodes(hetero_batch, nt)
            if n <= 0:
                continue
            store = hetero_batch[nt]
            raw_x = getattr(store, "x", None)
            if raw_x is not None and raw_x.numel() > 0:
                if self.node_encoder is None:
                    self.node_encoder = CanonicalNodeEncoder(self.hidden_dim).to(dev)
                elif next(self.node_encoder.parameters()).device != dev:
                    self.node_encoder = self.node_encoder.to(dev)
                type_id = torch.full((n,), i, dtype=torch.long, device=dev)
                x_dict[nt] = self.node_encoder(raw_x, type_id)
            else:
                x0 = self.node_type_emb.weight[i].unsqueeze(0).expand(n, -1)
                x_dict[nt] = x0

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
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
            g = torch.zeros((int(bsz or 1), self.hidden_dim), device=dev)
        else:
            g = torch.stack(pooled, dim=0).mean(dim=0)
        logits = self.classifier(g)

        if return_node_emb:
            assert hom_data is not None
            if edge_index_override is not None:
                hom_data.edge_index = edge_index_override
                hom_data.edge_type = edge_type_override
            # Node order must match `hom_data` / `to_homogeneous` (graph store order),
            # not training `metadata[0]` (sorted union of types across the dataset).
            hom_node_types = list(getattr(hom_data, "_node_type_names", hetero_batch.node_types))
            x_hom = self._x_dict_to_homogeneous(
                x_dict,
                node_types=hom_node_types,
                hetero_batch=hetero_batch,
                hidden_dim=self.hidden_dim,
                device=dev,
            )
            if int(x_hom.size(0)) != int(hom_data.num_nodes):
                raise RuntimeError(
                    f"HGT node embedding count ({x_hom.size(0)}) != homogeneous nodes ({hom_data.num_nodes})"
                )
            return logits, x_hom, hom_data
        return logits


def load_hgt_graph_classifier(
    path: Union[str, Path],
    *,
    device: torch.device,
    weights_only: bool = False,
) -> HGTGraphClassifier:
    """Load an HGT checkpoint saved by ``train_hgt.py``."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=weights_only)
    kwargs = ckpt.get(
        "model_kwargs",
        {"hidden_dim": 64, "num_layers": 2, "heads": 4, "dropout": 0.2, "num_classes": 2},
    )
    model = HGTGraphClassifier(**kwargs).to(device)

    metadata = ckpt.get("metadata")
    state = ckpt["state_dict"]
    if metadata is not None:
        model.materialize(metadata=tuple(metadata), device=device)
        model._ensure_node_encoder_from_state(state, device=device, num_node_types=len(metadata[0]))
        model._load_state_checked(state)
    else:
        model._pending_state = ckpt["state_dict"]

    model.eval()
    return model

