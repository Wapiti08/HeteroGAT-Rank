
import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.nn import global_mean_pool
from model import DiffPool
from torch import nn
from datetime import datetime
from torch_geometric.data import HeteroData
import os

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


# check the index and size to avoid cuda error
def sani_edge_index(edge_index, num_src_nodes, num_tgt_nodes):
    ''' fix the problem when index of node is not matched with node size
    
    '''
    src = edge_index[0]
    tgt = edge_index[1]
    valid_mask = (src < num_src_nodes) & (tgt < num_tgt_nodes)
    return edge_index[:, valid_mask]