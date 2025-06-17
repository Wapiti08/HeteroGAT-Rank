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
    grads_storage = {}

    # register hook to store gradients
    for edge_type, attn in attn_weights.items():
        if attn.requires_grad:
            attn.register_hook(lambda grad, et=edge_type: grads_storage.setdefault(et, grad))
        else:
            attn.retain_grad()
            attn.register_hook(lambda grad, et=edge_type: grads_storage.setdefault(et, grad))

    # ensure logits is scalar
    if logits.numel() == 1:
        loss = logits.squeeze()
    elif logits.dim() == 2:
        loss = logits[:, target_class].sum()
    elif logits.dim() == 1:
        loss = logits[target_class]
    else:
        raise ValueError("Unexpected logits shape")
    
    # backprop to populate gradients
    loss.backward(retain_graph=True)

    for edge_type, alpha in attn_weights.items():
        grad = grads_storage.get(edge_type, None)
        if grad is None:
            print(f"[Warning] No gradient for edge_type={edge_type}")
            continue
    
        mean_weights = alpha.mean(dim=1)
        mean_grad = grad.mean(dim=1)
        importance = (mean_weights * mean_grad).abs()

        edge_list = edge_index_map.get(edge_type, [])
        edge_scores = [
            (edge_list[i], round(score.item(), 6))
            for i, score in enumerate(importance)
            if i < len(edge_list)
        ]
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


def explain_with_gradcam(model, dataloader, device, target_class=1, max_batches=None):
    """
    Optimized Grad-CAM style explanation over batches to rank edge and node importance.

    Args:
        model: trained GNN model with attention weights stored as model.latest_attn_weights
        dataloader: data loader for explanation (use small batch size)
        device: computation device
        target_class: class index to backprop for Grad-CAM
        max_batches: limit batches for speed/memory

    Returns:
        edge_score_map: {(edge_type, (src, tgt)): importance score}
        node_score_map: {node: importance score}
    """
    model.eval()

    edge_score_map_total = defaultdict(float)
    node_score_map_total = defaultdict(float)

    batch_count = 0

    with torch.enable_grad():
        for batch in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break

            batch = batch.to(device)

            # Forward pass: model must internally call retain_grad on attention weights
            logits, *_  = model(batch)
            attn_weights = model.latest_attn_weights
            edge_index_map = model.latest_edge_index_map  # assume you also save this in forward

            # Choose scalar class logit
            if logits.numel() == 1:
                loss = logits.squeeze()
            elif logits.dim() == 2:
                loss = logits[:, target_class].sum()
            elif logits.dim() == 1:
                loss = logits[target_class]
            else:
                raise ValueError("Unexpected logits shape")

            # Backward to populate .grad on attention weights
            model.zero_grad()
            loss.backward()

            # Compute edge importance
            for edge_type, alpha in attn_weights.items():
                grad = alpha.grad
                if grad is None:
                    print(f"[Warning] No gradient for edge_type={edge_type}")
                    continue

                mean_weights = alpha.mean(dim=1)
                mean_grad = grad.mean(dim=1)
                importance = (mean_weights * mean_grad).abs()

                edge_list = edge_index_map.get(edge_type, [])
                for i, score in enumerate(importance):
                    if i < len(edge_list):
                        edge_score_map_total[(edge_type, tuple(edge_list[i]))] += score.item()

            batch_count += 1

    # Aggregate node importance
    for (etype, (src, tgt)), score in edge_score_map_total.items():
        node_score_map_total[src] += score
        node_score_map_total[tgt] += score

    return dict(edge_score_map_total), dict(node_score_map_total)