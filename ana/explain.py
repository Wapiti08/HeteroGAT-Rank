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


def explain_with_gradcam(model, dataloader, device, target_class=1, max_batches=None):
    """
    Grad-CAM style explanation to rank edge and node importance.
    Also maps node scores back to original node values using model.reverse_node_id_map.
    
    Returns:
        edge_score_map: {(edge_type, (src_node_value, tgt_node_value)): importance}
        node_score_map: {original_node_value: importance}
    """
    model = getattr(model, "module", model)  # unwrap DDP
    model.eval()

    edge_score_map_total = defaultdict(float)
    node_score_map_total = defaultdict(float)

    reverse_node_id_map = getattr(model, "reverse_node_id_map", {})

    batch_count = 0

    with torch.enable_grad():
        for batch in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break

            batch = batch.to(device)
            logits, *_ = model(batch)
            attn_weights = model.latest_attn_weights
            edge_index_map = model.latest_edge_index_map

            if logits.numel() == 1:
                loss = logits.squeeze()
            elif logits.dim() == 2:
                loss = logits[:, target_class].sum()
            elif logits.dim() == 1:
                loss = logits[target_class]
            else:
                raise ValueError("Unexpected logits shape")

            model.zero_grad()
            loss.backward()

            for edge_type, alpha in attn_weights.items():
                grad = alpha.grad
                if grad is None:
                    print(f"[Warning] No gradient for edge_type={edge_type}")
                    continue

                importance = (alpha.mean(dim=1) * grad.mean(dim=1)).abs()
                edge_list = edge_index_map.get(edge_type, [])

                for i, score in enumerate(importance):
                    if i < len(edge_list):
                        src_id, tgt_id = edge_list[i]
                        if src_id == tgt_id:
                            continue
                        src_val = reverse_node_id_map.get(src_id, f"Node_{src_id}")
                        tgt_val = reverse_node_id_map.get(tgt_id, f"Node_{tgt_id}")
                        edge_score_map_total[(edge_type, (src_val, tgt_val))] += score.item()

            batch_count += 1

    for (_, (src_val, tgt_val)), score in edge_score_map_total.items():
        node_score_map_total[src_val] += score
        node_score_map_total[tgt_val] += score

    return dict(edge_score_map_total), dict(node_score_map_total)
