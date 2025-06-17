'''
 # @ Create Time: 2025-06-05 16:00:49
 # @ Modified time: 2025-06-05 16:01:33
 # @ Description: implement enhanced classification loss with contrastive loss
 '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from collections import defaultdict
import random

def get_contrastive_pairs(graph_embeds: torch.Tensor, labels: torch.Tensor, max_pairs=64):
    """
    Return multiple positive and negative pairs as (B, 2, D)
    """
    pos_pairs, neg_pairs = [], []
    B = graph_embeds.size(0)
    for i in range(B):
        for j in range(i + 1, B):
            pair = torch.stack([graph_embeds[i], graph_embeds[j]], dim=0)  # shape [2, D]
            if labels[i] == labels[j]:
                pos_pairs.append(pair)
            else:
                neg_pairs.append(pair)

    if not pos_pairs or not neg_pairs:
        return None, None

    pair_count = min(len(pos_pairs), len(neg_pairs), max_pairs)
    pos_batch = torch.stack(random.sample(pos_pairs, pair_count))  # shape [N, 2, D]
    neg_batch = torch.stack(random.sample(neg_pairs, pair_count))  # shape [N, 2, D]
    return pos_batch, neg_batch

class CompositeLoss(nn.Module):
    def __init__(self, lambda_contrastive=0.1, lambda_sparsity=0.01, lambda_entropy=0.01):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_sparsity = lambda_sparsity
        self.lambda_entropy = lambda_entropy
    
    def forward(self,
                cls_loss: torch.Tensor,
                attn_weights: Dict[str, torch.Tensor] = None,
                graph_embeds: torch.Tensor = None,
                labels: torch.Tensor = None,
                return_details: bool = False) -> torch.Tensor | Dict[str, torch.Tensor]:
        '''
        composite loss for subgraph classification with enhanced discrimination

        when using pos and neg embeds, the input has to be a batch not individual subgraph
        '''
        loss = cls_loss
        losses = {'classification': cls_loss}

        # Dynamically sample pos/neg pairs for contrastive loss
        pos_embed, neg_embed = None, None
        if graph_embeds is not None and labels is not None:
            pos_embed, neg_embed = get_contrastive_pairs(graph_embeds, labels)

        if pos_embed is not None and neg_embed is not None:
            sim_pos = F.cosine_similarity(pos_embed[:, 0, :], pos_embed[:, 1, :], dim=1)  # (B,)
            sim_neg = F.cosine_similarity(neg_embed[:, 0, :], neg_embed[:, 1, :], dim=1)  # (B,)
            contrastive_loss = F.relu(1.0 - sim_pos + sim_neg).mean()
            loss = loss + self.lambda_contrastive * contrastive_loss
            losses['contrastive'] = contrastive_loss * self.lambda_contrastive
        
        if attn_weights:
            entropy_loss = 0
            sparsity_loss = 0
            for w in attn_weights.values():
                a = F.softmax(w.mean(dim=1), dim=0)
                entropy_loss += -(a * torch.log(a+1e-6)).sum()
                sparsity_loss += w.abs().sum()
            
            entropy_loss = self.lambda_entropy * entropy_loss
            sparsity_loss = self.lambda_sparsity * sparsity_loss

            loss += entropy_loss + sparsity_loss
            losses['entropy'] = entropy_loss
            losses['sparsity'] = sparsity_loss
    
        if return_details:
            losses['total'] = loss
            return losses
        else:
            return loss




