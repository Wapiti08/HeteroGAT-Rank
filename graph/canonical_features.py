"""Deterministic per-node features for canonical `HeteroData` graphs.

Graphs built by `build_heterodata_from_events` store `node_key` per node type but
no `x`. GNN baselines that only embed node *type* give every node of the same
type identical inputs, which makes GAT/HAN/HGT prone to collapse. This module
attaches a fixed-size feature vector per node at load time (no graph rebuild).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch_geometric.data import HeteroData

# Keep in sync with `model.node_encoder.CANONICAL_NODE_FEAT_DIM`.
CANONICAL_NODE_FEAT_DIM = 32
_HASH_DIM = 16
_REL_BUCKETS = 12


def _stable_hash_vec(key: str, dim: int = _HASH_DIM) -> torch.Tensor:
    """Map a node key string to `dim` values in [-1, 1] (deterministic)."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=dim * 2).digest()
    out = torch.empty(dim, dtype=torch.float32)
    for i in range(dim):
        # two bytes -> value in [-1, 1]
        b0, b1 = digest[2 * i], digest[2 * i + 1]
        u = (b0 << 8) | b1
        out[i] = (u / 65535.0) * 2.0 - 1.0
    return out


def _rel_bucket(src_type: str, etype: str) -> int:
    name = f"{src_type}|{etype}"
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    return h % _REL_BUCKETS


def attach_canonical_node_features(data: HeteroData) -> HeteroData:
    """Attach `data[ntype].x` with shape `[num_nodes, CANONICAL_NODE_FEAT_DIM]`."""
    node_types: List[str] = list(data.node_types)
    if not node_types:
        return data

    type_to_id = {nt: i for i, nt in enumerate(node_types)}
    ntypes = len(node_types)

    # Per-node in/out degree (aggregated over all incident edge types).
    in_deg: Dict[str, torch.Tensor] = {}
    out_deg: Dict[str, torch.Tensor] = {}
    rel_out: Dict[str, torch.Tensor] = {}

    for nt in node_types:
        n = int(data[nt].num_nodes or 0)
        in_deg[nt] = torch.zeros(n, dtype=torch.float32)
        out_deg[nt] = torch.zeros(n, dtype=torch.float32)
        rel_out[nt] = torch.zeros(n, _REL_BUCKETS, dtype=torch.float32)

    for edge_type in data.edge_types:
        src_t, etype, dst_t = edge_type
        store = data[edge_type]
        if not hasattr(store, "edge_index") or store.edge_index.numel() == 0:
            continue
        ei = store.edge_index
        src = ei[0].long()
        dst = ei[1].long()
        bucket = _rel_bucket(str(src_t), str(etype))
        if src.numel():
            out_deg[src_t].index_add_(0, src, torch.ones(src.size(0)))
            rel_out[src_t].index_put_(
                (src, torch.full_like(src, bucket)),
                torch.ones(src.size(0)),
                accumulate=True,
            )
        if dst.numel():
            in_deg[dst_t].index_add_(0, dst, torch.ones(dst.size(0)))

    total_nodes = sum(int(data[nt].num_nodes or 0) for nt in node_types)
    total_nodes = max(1, total_nodes)

    for nt in node_types:
        n = int(data[nt].num_nodes or 0)
        if n <= 0:
            continue
        keys = getattr(data[nt], "node_key", None)
        if keys is None:
            keys = [f"{nt}::{i}" for i in range(n)]
        elif not isinstance(keys, list):
            keys = list(keys)

        rows: List[torch.Tensor] = []
        tin = in_deg[nt]
        tout = out_deg[nt]
        trel = rel_out[nt]
        type_norm = float(type_to_id[nt]) / max(1, ntypes - 1) if ntypes > 1 else 0.0
        for i in range(n):
            feat = torch.zeros(CANONICAL_NODE_FEAT_DIM, dtype=torch.float32)
            feat[0] = math.log1p(float(tin[i]))
            feat[1] = math.log1p(float(tout[i]))
            feat[2] = math.log1p(float(tin[i] + tout[i]))
            feat[3] = type_norm
            feat[4] = float(n) / float(total_nodes)
            feat[5 : 5 + _HASH_DIM] = _stable_hash_vec(str(keys[i]), dim=_HASH_DIM)
            # outgoing relation histogram (12 dims at the end)
            feat[CANONICAL_NODE_FEAT_DIM - _REL_BUCKETS :] = trel[i]
            rows.append(feat)
        data[nt].x = torch.stack(rows, dim=0)
    return data


def hetero_num_nodes(data: HeteroData, ntype: str) -> int:
    """Return node count for ``data[ntype]``, inferring from keys/features when needed."""
    if ntype not in data.node_types:
        return 0
    store = data[ntype]
    n = getattr(store, "num_nodes", None)
    if n is not None:
        return int(n)
    keys = getattr(store, "node_key", None)
    if keys is not None:
        return len(keys)
    x = getattr(store, "x", None)
    if x is not None and x.numel() > 0:
        return int(x.size(0))
    return 0


def active_node_types(data: HeteroData) -> List[str]:
    """Node types with at least one node in this graph."""
    return [nt for nt in data.node_types if hetero_num_nodes(data, nt) > 0]


def to_homogeneous_safe(
    data: HeteroData,
    *,
    add_node_type: bool = True,
    add_edge_type: bool = True,
):
    """``to_homogeneous`` on populated stores only (avoids empty placeholder node types)."""
    active = set(active_node_types(data))
    if not active:
        raise ValueError("HeteroData has no nodes")
    if active == set(data.node_types):
        return data.to_homogeneous(add_node_type=add_node_type, add_edge_type=add_edge_type)

    from torch_geometric.data import HeteroData as HData

    slim = HData()
    if "y" in data:
        slim["y"] = data["y"]
    for nt in data.node_types:
        if nt not in active:
            continue
        slim[nt] = data[nt]
    for edge_type in data.edge_types:
        src_t, _rel, dst_t = edge_type
        if src_t in active and dst_t in active:
            slim[edge_type] = data[edge_type]
    return slim.to_homogeneous(add_node_type=add_node_type, add_edge_type=add_edge_type)


def load_canonical_graph(
    path: Union[str, Path],
    *,
    use_node_features: bool = True,
    map_location: str = "cpu",
) -> HeteroData:
    """Load a saved ``*.graph.pt`` and optionally attach per-node features."""
    obj = torch.load(Path(path), map_location=map_location, weights_only=False)
    data = HeteroData.from_dict(obj["data_dict"])
    for nt in list(data.node_types):
        n = hetero_num_nodes(data, nt)
        if n > 0:
            data[nt].num_nodes = n
    if use_node_features:
        attach_canonical_node_features(data)
    return data
