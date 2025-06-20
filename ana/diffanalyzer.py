'''
 # @ Create Time: 2025-03-12 11:41:27
 # @ Modified time: 2025-03-12 11:41:43
 # @ Description: 
 
    - extract subgraphs for different ecosystems
    - compare feature importance using the learned masks
    - rank important nodes/edges for intra-vs. inter-ecosystem connections

 '''

import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from model import mgan
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import pandas as pd


def rank_edges(atten_weights, edge_index_map, k:int, noise_factor:float):
    ''' rank top-k edges for each edge_type based on averaged attention weights

    params:
        atten_weights: dict of {edge_type: Tensor[num_edges, num_heads]}
        edge_index_map: dict of {edge_type: List[(src_node_value, tgt_node_value)]}
        k: number of top edges to select per edge type
        noise_factor: noise magnitude to beak ties
        
    return:
        top_k_edges: dict of {edge_type: List[Tuple[(src, tgt), score]]}
    '''
    top_k_edges = {}

    for edge_type, weights in atten_weights.items():
        if edge_type not in edge_index_map:
            print(f"Warning: {edge_type} not found in edge_index_map. Skipping.")
            top_k_edges[edge_type] = []
            continue

        # Compute the mean attention weight for each edge across all heads
        mean_weights = weights.mean(dim=1).cpu()  # shape: [num_edges]

        # add random noise to break ties in case of identical attention weights
        if noise_factor:
            noise = torch.randn(mean_weights.shape) * noise_factor
            mean_weights = mean_weights + noise

        # sort the edges by their mean attention weight
        sorted_indices = torch.argsort(mean_weights, descending=True)
        # select the top k edges 
        top_k_edge_indices = sorted_indices[:k]

        edge_list = edge_index_map[edge_type]
        selected = []

        for i in top_k_edge_indices:
            i = i.item()
            if i < len(edge_list):
                # tuple: ((src, tgt), score)
                selected.append((edge_list[i], round(mean_weights[i].item(), 4)))  

        top_k_edges[edge_type] = selected

    return top_k_edges


def rank_nodes_by_eco_system(edge_atten_map, node_json:list, k):
    ''' Rank target nodes by aggregated attention per eco-system (based on source node's eco)

    params:
        edge_attn_map: dict {(src_node_value, tgt_node_value): attention_score }
        node_json: list of node dict
        k: num of top tgt nodes to keep per eco-system
    
    returns:
        dict {eco_system: List[Tuple[tgt_node_value, score]]}

    '''
    # Create a dictionary to store attention scores by eco_system and tgt_node_value
    value_to_eco = {
        node["value"]: node["eco"]
        for node in node_json
        if "value" in node and "eco" in node
    }

    eco_system_rank_map = defaultdict(lambda: defaultdict(float))

    df = pd.DataFrame(node_json)

    # aggregate attention weights for each node based on its adjacent edges
    for (src_node_value, tgt_node_value), attn_weight in edge_atten_map.items():
        # Find the eco_system for this node type in the DataFrame
        # src_node_eco = df[df['value'] == src_node_value]['eco'].values
        src_node_eco = value_to_eco.get(src_node_value)

        if src_node_eco is None:
            continue
        eco_system_rank_map[src_node_eco][tgt_node_value] += attn_weight

    # Sort and get top k nodes per eco_system
    top_k_by_eco_system = {}
    for eco_system, tgt_nodes in eco_system_rank_map.items():
        sorted_tgt_nodes = sorted(tgt_nodes.items(), key=lambda x: x[1], reverse=True)
        top_k_by_eco_system[eco_system] = [(node, round(score, 4)) for node, score in sorted_tgt_nodes[:k]]

    return top_k_by_eco_system


def rank_nodes_global(edge_atten_map, k):
    ''' Rank target nodes globally by their total attention score.

    params:
        edge_atten_map: dict {(src_node_value, tgt_node_value): attention_score}
        k: number of top target nodes to return
    
    reutrns:
        list of (tgt_node_value, score)
    '''

    # Create a dictionary to store aggregated attention scores for each target node
    tgt_node_rank_map = defaultdict(float)

    # Iterate over edge_attention_map to aggregate attention weights for each tgt_node_value
    for (src_node_value, tgt_node_value), attn_weight in edge_atten_map.items():
        tgt_node_rank_map[tgt_node_value] += attn_weight
    
    # Sort the target nodes by their aggregated attention scores
    sorted_tgt_nodes = sorted(tgt_node_rank_map.items(), key=lambda x: x[1], reverse=True)
    
    return [(node, round(score, 4)) for node, score in sorted_tgt_nodes[:k]]

def final_sample(top_k_edges, top_k_nodes_by_eco, top_k_global_nodes):
    '''
    Final intersection sampling of nodes and edges, combining eco-system and global rankings.
    Params:
        top_k_edges: dict {edge_type: List[((src, tgt), score)]}
        top_k_nodes_by_eco: dict {eco: List[(tgt, score)]}
        top_k_global_nodes: List[(tgt, score)]

    Returns:
        dict with:
            - top_k_edges: same as input
            - top_k_nodes_by_eco: same as input
            - top_k_global_nodes: same as input
            - final_selected_nodes: List of nodes
            - final_selected_edges: dict {edge_type: List[((src, tgt), score)]}    
    '''

    selected_nodes = set([node for node, _ in top_k_global_nodes])
    for node_list in top_k_nodes_by_eco.values():
        selected_nodes.update([node for node, _ in node_list])

    final_selected_edges = {}
    for edge_type, edge_list in top_k_edges.items():
        filtered = [
            ((src, tgt), score)
            for ((src, tgt), score) in edge_list
            if src in selected_nodes or tgt in selected_nodes
        ]
        final_selected_edges[edge_type] = filtered

    return {
        'top_k_edges': top_k_edges,
        'top_k_nodes_by_eco': top_k_nodes_by_eco,
        'top_k_global_nodes': top_k_global_nodes,
        'final_selected_nodes': list(selected_nodes),
        'final_selected_edges': final_selected_edges
    }
