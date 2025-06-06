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

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

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

    top_k_edges_indices = {edge_type: [] for edge_type in atten_weights}
    for edge_type, weights in atten_weights.items():
        if edge_type not in edge_index_map:
            print(f"Warning: {edge_type} not found in edge_index_map. Skipping.")
            continue
        # weights is a tensor of shape (num_edges, num_heads)
        # Compute the mean attention weight for each edge across all heads
        mean_weights = weights.mean(dim=1).to(device)  # shape: [num_edges]

        # add random noise to break ties in case of identical attention weights
        if noise_factor:
            noise = torch.randn(mean_weights.shape) * noise_factor
            noise = noise.to(device)
            mean_weights = mean_weights + noise

        # sort the edges by their mean attention weight
        sorted_indices = torch.argsort(mean_weights, descending=True)
        # select the top k edges 
        top_k_edge_indices = sorted_indices[:k]

        # get the corresponding edge
        top_k_edge_for_type = top_k_edge_indices.tolist()

        # add the selected top k edges for the current edge type
        top_k_edges_indices[edge_type] = top_k_edge_for_type

    top_k_edges = {}

    for edge_type in atten_weights.keys():
        if edge_type not in top_k_edges_indices:
            # If edge_type was skipped earlier, assign an empty list
            top_k_edges[edge_type] = []
            continue

        edge_list = edge_index_map.get(edge_type, [])
        selected_edges = [edge_list[i] for i in top_k_edges_indices[edge_type] if i < len(edge_list)]
        top_k_edges[edge_type] = selected_edges
    
    return top_k_edges

def rank_nodes_by_eco_system(edge_atten_map, node_json:list, k):
    ''' Rank target nodes (tgt_node_value) by eco_system consideration
    
    '''
    # Create a dictionary to store attention scores by eco_system and tgt_node_value
    eco_system_rank_map = defaultdict(lambda: defaultdict(float))

    df = pd.DataFrame(node_json)

    # aggregate attention weights for each node based on its adjacent edges
    for (src_node_value, tgt_node_value), attn_weight in edge_atten_map.items():
        # Find the eco_system for this node type in the DataFrame
        src_node_eco = df[df['value'] == src_node_value]['eco'].values
        if len(src_node_eco) == 0:
            continue  # Skip if eco_system for source node is not found
        src_node_eco = src_node_eco[0]  # Assuming the eco_system is unique for each node

        # Aggregate attention score by eco_system for tgt_node_value
        eco_system_rank_map[src_node_eco][tgt_node_value] += attn_weight

    # Sort and get top k nodes per eco_system
    top_k_by_eco_system = {}
    for eco_system, tgt_nodes in eco_system_rank_map.items():
        sorted_tgt_nodes = sorted(tgt_nodes.items(), key=lambda x: x[1], reverse=True)
        top_k_by_eco_system[eco_system] = [node for node, score in sorted_tgt_nodes[:k]]

    return top_k_by_eco_system

def rank_nodes_global(edge_atten_map, k):
    ''' Rank target nodes (tgt_node_value) without considering eco_system '''

    # Create a dictionary to store aggregated attention scores for each target node
    tgt_node_rank_map = defaultdict(float)

    # Iterate over edge_attention_map to aggregate attention weights for each tgt_node_value
    for (src_node_value, tgt_node_value), attn_weight in edge_atten_map.items():
        tgt_node_rank_map[tgt_node_value] += attn_weight
    
    # Sort the target nodes by their aggregated attention scores
    sorted_tgt_nodes = sorted(tgt_node_rank_map.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top k target nodes globally
    top_k_global_nodes = [node for node, score in sorted_tgt_nodes[:k]]

    return top_k_global_nodes

def final_sample(top_k_edges, top_k_nodes_by_eco, top_k_global_nodes):
    '''
    Final sampling that returns actual node and edge values
    Params:
        atten_weights: dict[edge_type -> Tensor[num_edges, num_heads]]
        edge_atten_map: dict[(src_node_value, tgt_node_value) -> attn_score]
        edge_index_map: dict[edge_type -> List[(src_node_value, tgt_node_value)]]
        node_json: list of node dicts with keys including 'value' and 'eco'
        k: top-k value
        noise_factor: float for tie-breaking
    Returns:
        dict with top-k edges/nodes in original value form
    '''

    # step4: final sample - intersect results
    final_selected_nodes = set(top_k_global_nodes)

    for eco_nodes in top_k_nodes_by_eco.values():
        final_selected_nodes.update(eco_nodes)
    
    final_select_edges = {}
    for edge_type, edge_list in top_k_edges.items():
        # filter edges that connect to selected target nodes
        filtered = [
            (src, tgt)
            for (src, tgt) in edge_list
            if tgt in final_selected_nodes or src in final_selected_nodes
        ]
        final_select_edges[edge_type] = filtered

    return {
    'top_k_edges': top_k_edges,
    'top_k_nodes_by_eco': top_k_nodes_by_eco,
    'top_k_global_nodes': top_k_global_nodes,
    'final_selected_nodes': list(final_selected_nodes),
    'final_selected_edges': final_select_edges
    }

