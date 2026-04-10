from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ExplainerLoss:
    fidelity: torch.Tensor
    sparsity: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor


class PGExplainer(nn.Module):
    """Backbone-agnostic edge-mask explainer (PGExplainer-style).

    Learns an edge mask m_e in (0,1) from node embeddings, then uses a
    straight-through hard mask to build a masked edge_index for a second forward.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_type_emb_dim: int = 16,
        temperature: float = 1.0,
        sparsity_target: float = 0.05,
        sparsity_coef: float = 1.0,
        entropy_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.sparsity_target = sparsity_target
        self.sparsity_coef = sparsity_coef
        self.entropy_coef = entropy_coef

        # Edge type embedding (uses homogeneous `edge_type` id).
        self.edge_type_emb = nn.Embedding(512, edge_type_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_type_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def edge_logits(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h = torch.cat([x[src], x[dst], self.edge_type_emb(edge_type)], dim=-1)
        return self.mlp(h).view(-1)  # [E]

    @staticmethod
    def _st_gumbel_sigmoid(logits: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
        u = torch.rand_like(logits)
        g = torch.log(u + 1e-8) - torch.log(1.0 - u + 1e-8)
        y = torch.sigmoid((logits + g) / max(temperature, 1e-6))
        y_hard = (y > 0.5).float()
        y_st = (y_hard - y).detach() + y
        return y, y_st

    @staticmethod
    def _mask_edges(
        edge_index: torch.Tensor, edge_type: torch.Tensor, hard_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keep = hard_mask > 0.5
        if keep.sum() == 0:
            keep = torch.zeros_like(keep)
            keep[0] = True
        return edge_index[:, keep], edge_type[keep]

    def forward(
        self,
        *,
        backbone,
        hetero_batch,
        target: Optional[torch.Tensor] = None,
    ):
        backbone.eval()
        with torch.no_grad():
            logits_ref, x_ref, data_ref = backbone(hetero_batch, return_node_emb=True)
            prob_ref = F.softmax(logits_ref, dim=-1)

        edge_index = data_ref.edge_index
        edge_type = data_ref.edge_type
        logits_e = self.edge_logits(x_ref, edge_index, edge_type)

        soft_mask, hard_mask_st = self._st_gumbel_sigmoid(logits_e, self.temperature)
        masked_edge_index, masked_edge_type = self._mask_edges(edge_index, edge_type, hard_mask_st)

        logits_masked = backbone(
            hetero_batch,
            edge_index_override=masked_edge_index,
            edge_type_override=masked_edge_type,
        )
        prob_masked = F.softmax(logits_masked, dim=-1)

        fidelity = F.kl_div(prob_masked.log(), prob_ref, reduction="batchmean")
        # Encourage the mask density to match a *target keep ratio* rather than
        # collapsing to all-zeros (which is a trivial optimum if we only penalize mean(mask)).
        sparsity = (soft_mask.mean() - float(self.sparsity_target)) ** 2
        ent = -(soft_mask * torch.log(soft_mask + 1e-8) + (1 - soft_mask) * torch.log(1 - soft_mask + 1e-8))
        entropy = ent.mean()

        total = fidelity + self.sparsity_coef * sparsity + self.entropy_coef * entropy

        return {
            "edge_logits": logits_e.detach(),
            "edge_mask": soft_mask.detach(),
            "loss": ExplainerLoss(fidelity=fidelity, sparsity=sparsity, entropy=entropy, total=total),
        }

    @torch.no_grad()
    def score_edges(self, *, backbone, hetero_batch) -> dict:
        """Deterministic edge scoring for hunting.

        Returns per-edge sigmoid scores on the homogeneous edge list produced by
        `backbone(..., return_node_emb=True)`.
        """
        backbone.eval()
        self.eval()
        logits, x, data = backbone(hetero_batch, return_node_emb=True)
        edge_index = data.edge_index
        edge_type = data.edge_type
        edge_logits = self.edge_logits(x, edge_index, edge_type)
        edge_score = torch.sigmoid(edge_logits)
        return {
            "logits": logits,
            "data": data,
            "edge_score": edge_score,
        }

