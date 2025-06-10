'''
 # @ Create Time: 2025-06-05 11:55:58
 # @ Modified time: 2025-06-05 12:01:02
 # @ Description: function to increase the explainability and the ensemble loss function to maximize the difference of positive and negative loss function
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

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
    
    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        ''' apply attention pooling over all node types
        
        args:
            x_dict: dict of node type -> {node_type: [B, N, F]} or [N, F] is no batch
        
        returns:
            pooled_embedding: tensor of shape [F], aggregated from all node types
        '''
        pooled_outputs = []
        for node_type, x in x_dict.items():
            if x.dim() == 2:  # no batch dimension
                x = x.unsqueeze(0) # [N, F] -> [1, N, F]

            score = self.attn_mlp(x).squeeze(-1) # [B, N]
            # sum to 1
            attn = F.softmax(score, dim=1) # [B, N]
            attn = self.dropout(attn)
            pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1) # [B, F]
            pooled_outputs.append(pooled)

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

    def forward(self, edge_attr_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            edge_attr_dict: dict of edge_type -> [num_edges, F] edge features

        Returns:
            pooled_edge_embedding: tensor of shape [F], aggregated from all edge types
        """
        pooled_edges = []
        for (src_node_type, edge_type, tgt_node_type), edge_attr in edge_attr_dict.items():
            if edge_type not in self.edge_mlps:
                self.edge_mlps[edge_type] = nn.Sequential(
                    nn.Linear(self.edge_attr_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 1)
                )
            
            # check whether it is sparse tensor
            if edge_attr.is_sparse:
                edge_attr = edge_attr.to_dense()

            # add batch dimension
            if edge_attr.dim() == 2: # [E, F]
                edge_attr = edge_attr.unsqueeze(0) # [1, E, F]
            
            edge_attr = edge_attr.to(next(self.edge_mlps[edge_type].parameters()).device) 
            score = self.edge_mlps[edge_type](edge_attr) # [B, E, 1]
            # dimeansion 0 is different
            attn = F.softmax(score, dim=0)
            attn = self.dropout(attn)

            pooled = torch.sum(attn * edge_attr, dim=1)  # [B, F]
            pooled_edges.append(pooled)

        return torch.stack(pooled_edges, dim=0).mean(dim=0)











