'''
 # @ Create Time: 2025-06-05 11:55:58
 # @ Modified time: 2025-06-05 12:01:02
 # @ Description: function to increase the explainability and the ensemble loss function to maximize the difference of positive and negative loss function
 '''
import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from utils import sparsepad


class MultiTypeAttentionPooling(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int=64, dropout: float=0.1):
        ''' multi-type attention pooling for heterogenous graph

        args:
            in_dim: input dimension of node features
            hidden_dim: hidden dimension for attention score computation
            dropout: dropout rate
        
        '''
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
        x_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
        ) -> torch.Tensor:
        ''' apply attention pooling over all node types
        
        Args:
            x_dict: dict of node_type -> [N, F]
            batch_dict: dict of node_type -> [N], indicates each node's graph in the batch

        returns:
            pooled_embedding: Tensor of shape [B, F]
        '''
        
        pooled_outputs = []
        batch_ids = None

        for node_type, x in x_dict.items():
            batch = batch_dict.get(node_type, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            
            score = self.attn_mlp(x).squeeze(-1)  # [N]
            attn = torch.zeros_like(score, device=x.device)  # [N]

            for b in torch.unique(batch):
                mask = (batch == b)
                mask = mask.to(score.device)
                score_b = score[mask]  # scores for nodes in the same graph
                attn_b = F.softmax(score_b, dim=0)  # softmax over the scores
                attn[mask] = attn_b  # assign back to the full attention vector
            
            attn = self.dropout(attn)  # apply dropout
            messages = attn.unsqueeze(-1) * x

            B = batch.max().item() + 1
            pooled = torch.zeros(B, x.size(1), device=x.device)  # [B, F]

            for b in torch.unique(batch):
                mask = (batch == b)
                mask = mask.to(attn.device)
                pooled[b] += torch.sum(messages[mask], dim=0)

            pooled_outputs.append(pooled)

            # capture batch ids for final alignment check
            if batch_ids is None:
                batch_ids = batch.unique()
            else:
                batch_ids = torch.unique(torch.cat([batch_ids, batch.unique()]))

        # Align all pooled outputs by batch size
        B = batch_ids.max().item() + 1
        for i in range(len(pooled_outputs)):
            if pooled_outputs[i].size(0) < B:
                pad = torch.zeros(B - pooled_outputs[i].size(0), pooled_outputs[i].size(1), device=pooled_outputs[i].device)
                pooled_outputs[i] = torch.cat([pooled_outputs[i], pad], dim=0)


        return torch.stack(pooled_outputs, dim=0).mean(dim=0)


class MultiTypeEdgePooling(nn.Module):
    def __init__(self, edge_attr_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        '''
        args:
        '''
        super().__init__()
        self.edge_mlps = nn.ModuleDict()
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
        batch_edge_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            edge_attr_dict: dict of edge_type -> [E, F] edge features
            batch_edge_dict: dict of edge_type -> [E] indicating graph membership

        Returns:
            pooled_edge_embedding: tensor of shape [B, F], aggregated per graph
        """
        pooled_edges = []
        max_B = 0
        for (src_node_type, edge_type, tgt_node_type), edge_attr in edge_attr_dict.items():
            # Get corresponding batch vector for this edge type
            batch_e = batch_edge_dict.get((src_node_type, edge_type, tgt_node_type), None)

            # Skip if no batch info or empty edge features
            if batch_e is None or edge_attr.size(0) == 0:
                print(f"[EdgePool] Skipping {edge_type} — batch_e is None or edge_attr.size(0)==0")
                continue

            # Convert sparse tensor to dense values if needed
            if edge_attr.is_sparse:
                edge_attr = edge_attr.to_dense()  # [E, F]
                assert edge_attr.size(0) == batch_e.size(0), \
                    f"[ERROR] edge_attr.size(0) = {edge_attr.size(0)} ≠ batch_e.size(0) = {batch_e.size(0)}"

            elif edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)  # if [E]，convert to [E, 1]

            elif edge_attr.dim() != 2:
                raise ValueError(f"[ERROR] Unexpected edge_attr shape: {edge_attr.shape}")
            
            # Initialize attention MLP for this edge_type if not already created
            if edge_type not in self.edge_mlps:
                self.edge_mlps[edge_type] = nn.Sequential(
                    nn.Linear(self.edge_attr_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 1)
                )

            edge_attr = edge_attr.to(next(self.edge_mlps[edge_type].parameters()).device) 
            batch_e = batch_e.to(edge_attr.device)
            # Compute attention score for each edge: shape [E]
            score = self.edge_mlps[edge_type](edge_attr).squeeze(-1)  # [E]
            # Compute attention weights via softmax per graph in batch
            attn = torch.zeros_like(score)
            for b in torch.unique(batch_e):
                mask = (batch_e == b)
                mask = mask.to(score.device)
                score_b = score[mask]
                attn_b = F.softmax(score_b, dim=0)
                attn[mask] = attn_b

            attn = self.dropout(attn)
            messages = attn.unsqueeze(-1) * edge_attr  # [E, F]

            # aggregate messages per graph
            B = batch_e.max().item() + 1
            max_B = max(max_B, B)
            pooled = torch.zeros(B, edge_attr.size(1), device=edge_attr.device)

            for b in torch.unique(batch_e):
                mask = (batch_e == b)
                mask = mask.to(attn.device)
                pooled[b] += torch.sum(messages[mask], dim=0)

            pooled_edges.append(pooled)

        # Pad pooled tensors to max_B so they can be stacked
        for i, pooled in enumerate(pooled_edges):
            if pooled.size(0) < max_B:
                pad = torch.zeros(max_B - pooled.size(0), pooled.size(1), device=pooled.device)
                pooled_edges[i] = torch.cat([pooled, pad], dim=0)

        # If no edge types were processed, return fallback zero tensor
        if not pooled_edges:    
            print("[EdgePool] Warning: No edge types were pooled. Returning zeros.")
            return torch.zeros(1, self.edge_attr_dim, device=next(self.parameters()).device)  # [1, F]

        return torch.stack(pooled_edges, dim=0).mean(dim=0)  # [B, F]











