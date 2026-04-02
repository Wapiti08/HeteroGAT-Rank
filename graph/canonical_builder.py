"""
Build PyG `HeteroData` from canonical `EdgeEvent`s.

This builder is intentionally minimal: it focuses on *structure* (node/edge ids,
edge_index) and preserves raw attrs for later encoding. That keeps it reusable
across datasets (OSPTrack, QUT-DV25, etc.) and across baselines.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple

from config.osptrack_canonical import EdgeEvent, NodeRef


def _node_key(n: NodeRef) -> Tuple[str, str]:
    return (n.type, n.key)


def build_heterodata_from_events(
    events: Iterable[EdgeEvent],
    *,
    y: int | None = None,
    keep_raw_attrs: bool = True,
) -> "HeteroData":
    """Convert canonical events to a `HeteroData`.

    - Node types come from `NodeRef.type`
    - Edge relation names come from `EdgeEvent.etype`
    - Edge direction is kept as-is (src -> dst)
    """
    # Lazy import so the rest of the repo (and lightweight tests) can run
    # even if torch/pyg aren't installed in the current interpreter.
    try:
        import torch  # type: ignore
        from torch_geometric.data import HeteroData  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency for graph building. Install `torch` and `torch-geometric` "
            "in the current Python environment to use `build_heterodata_from_events`."
        ) from e

    events = list(events)

    # 1) Assign per-node-type local indices (PyG expects per-type indexing).
    node_maps: Dict[str, Dict[str, int]] = defaultdict(dict)
    node_lists: DefaultDict[str, List[NodeRef]] = defaultdict(list)

    def get_node_idx(n: NodeRef) -> int:
        m = node_maps[n.type]
        if n.key not in m:
            m[n.key] = len(m)
            node_lists[n.type].append(n)
        return m[n.key]

    # 2) Build per-edge-type index lists.
    edge_pairs: DefaultDict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    edge_attrs: DefaultDict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for e in events:
        si = get_node_idx(e.src)
        ti = get_node_idx(e.dst)
        etype = (e.src.type, e.etype, e.dst.type)
        edge_pairs[etype].append((si, ti))
        if keep_raw_attrs:
            edge_attrs[etype].append(dict(e.attrs))

    data = HeteroData()
    if y is not None:
        data["y"] = torch.tensor([y], dtype=torch.long)

    # 3) Materialize node stores.
    for ntype, nodes in node_lists.items():
        data[ntype].num_nodes = len(nodes)
        if keep_raw_attrs:
            data[ntype].node_key = [n.key for n in nodes]
            data[ntype].raw = [dict(n.attrs) for n in nodes]

    # 4) Materialize edge stores.
    for etype, pairs in edge_pairs.items():
        if pairs:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data[etype].edge_index = edge_index
        if keep_raw_attrs:
            data[etype].raw = edge_attrs[etype]

    return data

