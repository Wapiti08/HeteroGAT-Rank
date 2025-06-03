
import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.data import HeteroData
import os
import pickle


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def batch_dict(batch: HeteroData):
    ''' function to extract x_dict and edge_index_dict from custom batch data
    
    '''
    # extract node features into a dict
    x_dict = {node_type: batch[node_type].x for node_type in batch.node_types if 'x' in batch[node_type]}
    
    # extract edge_index into a dict
    edge_index_dict = {
        edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types if 'edge_index' in batch[edge_type]
    }

    # extract edge_attr into a dict
    edge_attr_dict = {
        edge_type: batch[edge_type].edge_attr for edge_type in batch.edge_types if 'edge_attr' in batch[edge_type]

    }

    return x_dict, edge_index_dict, edge_attr_dict


def miss_check(x_dict, node_types, hidden_dim):
    for node_type in node_types:
        if node_type not in x_dict:
            # Dummy tensor to avoid breaking GATConv
            x_dict[node_type] = torch.zeros((1, hidden_dim), device=device)

    return x_dict


def load_global_node_id_map(map_file_path):
    ''' load the global node ID mapping from a file '''
    if Path(map_file_path).exists():
        with open(map_file_path, 'rb') as f:
            global_node_id_map = pickle.load(f)
        return global_node_id_map['global_node_id_map']
    else:
        return None

def get_ori_node_value(global_node_id, reverse_node_id_map):
    '''
    args:
        reverse_node_id_map: from id to value
    '''
    return reverse_node_id_map.get(global_node_id, None)


def global_to_local_map(x_dict, edge_index_dict):
    ''' create a mapping of global node indices to local node indices
    
    args:
        x_dict (dict): dictionary containing node features for each node type
    
    returns:
        global_to_local_map: a mapping from global node indices to local indices
    '''
    global_to_local_map = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, tgt_type = edge_type
        num_src_nodes = x_dict[src_type].size(0)  # Local node count for source type
        num_tgt_nodes = x_dict[tgt_type].size(0)  # Local node count for target type
        
        # Create a mapping of global indices to local indices for the current subgraph (edge_type)
        src_local_indices = torch.arange(num_src_nodes) 
        tgt_local_indices = torch.arange(num_tgt_nodes)

        # mapping global to local indices
        global_to_local_map[src_type] = {i: idx for i, idx in enumerate(src_local_indices)}
        global_to_local_map[tgt_type] = {i: idx for i, idx in enumerate(tgt_local_indices)}
    
    return global_to_local_map


def remap_edge_indices(edge_index, global_to_local_mapping, src_type, tgt_type):
    """
    Remap the global node indices in edge_index to local indices using the local_to_global_mapping.
    
    Args:
        edge_index (tensor): The edge indices for a given edge type.
        global_to_local_mapping (dict): A mapping from global node indices to local node indices.
        
    Returns:
        remapped_edge_index (tensor): Edge indices remapped to local indices.
    """
    src = edge_index[0]
    tgt = edge_index[1]

    src_map = global_to_local_mapping[src_type]
    tgt_map = global_to_local_mapping[tgt_type]

    # Initialize lists to store the local indices
    src_local = []
    tgt_local = []
    # avoid miss matched indices
    for s, t in zip(src, tgt):
            s_item = s.item()
            t_item = t.item()
            
            if s_item in src_map and t_item in tgt_map:
                src_local.append(src_map[s_item])
                tgt_local.append(tgt_map[t_item])
            # else: skip the edge completely

    return torch.tensor([src_local, tgt_local], dtype=torch.long, device=edge_index.device)

# check the index and size to avoid cuda error
def sani_edge_index(edge_index, num_src_nodes, num_tgt_nodes, global_to_local_mapping):
    '''Fix the problem when index of node is not matched with node size (for subgraphs)'''

    # Remap global indices to local indices using the local_to_global_mapping
    remapped_edge_index = remap_edge_indices(edge_index, global_to_local_mapping)
    
    # Check if remapped edges are valid based on the subgraph size
    src_local = remapped_edge_index[0]
    tgt_local = remapped_edge_index[1]

    valid_mask = (src_local < num_src_nodes) & (tgt_local < num_tgt_nodes)

    # Count valid edges and filter out invalid ones
    edge_index = remapped_edge_index[:, valid_mask]
    
    return edge_index