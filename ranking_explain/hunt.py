from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


@dataclass(frozen=True)
class RankedEdge:
    graph_id: int
    score: float
    etype: Tuple[str, str, str]  # (src_type, rel, dst_type)
    src_type: str
    src_key: str
    dst_type: str
    dst_key: str


def _node_offsets(hetero_batch, node_type_names: List[str]) -> Dict[str, int]:
    """Compute global id offsets used by `to_homogeneous` concatenation."""
    off: Dict[str, int] = {}
    cur = 0
    for nt in node_type_names:
        off[nt] = cur
        cur += int(hetero_batch[nt].num_nodes)
    return off


def _lookup_node_key(hetero_batch, node_type: str, local_id: int) -> str:
    store = hetero_batch[node_type]
    keys = getattr(store, "node_key", None)
    if keys is None:
        return f"{node_type}#{local_id}"
    if local_id < 0 or local_id >= len(keys):
        return f"{node_type}#{local_id}"
    return str(keys[local_id])


def topk_edges(
    *,
    backbone,
    explainer,
    hetero_batch,
    k: int = 10,
) -> List[RankedEdge]:
    """Return top-k edges per graph in the batch, mapped back to hetero node keys."""
    out = explainer.score_edges(backbone=backbone, hetero_batch=hetero_batch)
    data = out["data"]  # homogeneous Data
    scores = out["edge_score"]  # [E]
    return ranked_edges_from_scores(hetero_batch=hetero_batch, data=data, scores=scores, k=k)


def ranked_edges_from_scores(
    *,
    hetero_batch,
    data,
    scores: torch.Tensor,
    k: int = 10,
) -> List[RankedEdge]:
    """Map homogeneous edge scores back to hetero node keys and return top-k edges."""

    node_type_names: List[str] = list(getattr(data, "_node_type_names", []))
    edge_type_names: List[Tuple[str, str, str]] = list(getattr(data, "_edge_type_names", []))
    if not node_type_names or not edge_type_names:
        return []

    offsets = _node_offsets(hetero_batch, node_type_names)

    # Determine which graph an edge belongs to by source node batch id.
    node_batch = getattr(data, "batch", None)
    if node_batch is None:
        node_batch = torch.zeros(int(data.num_nodes), dtype=torch.long, device=scores.device)

    src = data.edge_index[0]
    dst = data.edge_index[1]
    edge_graph = node_batch[src].detach().cpu()

    # Group by graph id and take top-k.
    ranked: List[RankedEdge] = []
    for gid in edge_graph.unique().tolist():
        mask = (edge_graph == gid)
        idx = torch.nonzero(mask, as_tuple=False).view(-1).to(scores.device)
        if idx.numel() == 0:
            continue
        top = idx[torch.topk(scores[idx], k=min(k, idx.numel())).indices]

        for ei in top.tolist():
            et_id = int(data.edge_type[ei].item())
            etype = edge_type_names[et_id]
            src_gid = int(src[ei].item())
            dst_gid = int(dst[ei].item())

            src_type = node_type_names[int(data.node_type[src_gid].item())]
            dst_type = node_type_names[int(data.node_type[dst_gid].item())]

            src_local = src_gid - offsets[src_type]
            dst_local = dst_gid - offsets[dst_type]

            ranked.append(
                RankedEdge(
                    graph_id=int(gid),
                    score=float(scores[ei].item()),
                    etype=etype,
                    src_type=src_type,
                    src_key=_lookup_node_key(hetero_batch, src_type, int(src_local)),
                    dst_type=dst_type,
                    dst_key=_lookup_node_key(hetero_batch, dst_type, int(dst_local)),
                )
            )

    # Sort globally (within batch) by score desc
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked

