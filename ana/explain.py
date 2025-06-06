'''
 # @ Create Time: 2025-06-05 11:59:25
 # @ Modified time: 2025-06-05 15:23:18
 # @ Description: 
 '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from collections import defaultdict

def compute_edge_import(attn_weights: Dict[str, torch.Tensor],
                        edge_index_map: Dict[str, List[Tuple[str, str]]],
                        logits: torch.Tensor,
                        target_class: int) -> Dict[str, List[Tuple[Tuple[str, str], float]]]:
    """
    compute edge importance via Grad-CAM-like method: attention * gradient
    """
    edge_score_map = {}

    if logits.dim() > 1:
        logits = logits[:, target_class].sum()
    else:
        logits = logits[target_class]
    
    logits.backward(retain_graph=True)

    for edge_type, weights in attn_weights.items():
        if weights.grad is None:
            print(f"[Warning] No gradient for edge_type={edge_type}")
            continue
        
        grad = weights.grad
        mean_weights = weights.mean(dim=1)
        mean_grad = grad.mean(dim=1)

        importance = (mean_weights * mean_grad).abs()
        edge_list = edge_index_map.get(edge_type, [])

        edge_scores = []
        for i, score in enumerate(importance):
            if i<len(edge_list):
                edge_scores.append((edge_list[i], round(score.item(), 6)))
        
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        edge_score_map[edge_type] = edge_scores
    
    return edge_score_map

def compute_node_import(edge_score_map: Dict[str, List[Tuple[Tuple[str, str], float]]]) -> Dict[str, float]:
    """ aggregate edge importance scores to node-level importance

    args:
        edge_score_map: dict of {edge_type: list of ((src, tgt), score)}
    
    returns:
        node_score_map: dict {node_value: aggregated importance score}
    """
    node_score_map = defaultdict(float)
    for edge_type, edge_scores in edge_score_map.items():
        for (src, tgt), score in edge_scores:
            node_score_map[src] += score
            node_score_map[tgt] += score
    return dict(node_score_map)



        

