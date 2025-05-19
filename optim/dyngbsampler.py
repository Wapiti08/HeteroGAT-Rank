'''
 # @ Create Time: 2025-05-14 14:04:31
 # @ Modified time: 2025-05-14 14:04:35
 # @ Description: dynamic neighbor sampling for heterogeneous graph training
 '''
import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv
from optim import diffpool
import torch.nn.functional as F
import os
from utils import evals
from utils.pregraph import *
# import neighborsampler designed for heterogenous graphs
from torch_geometric.sampler.neighbor_sampler import NeighborSampler


def dynamic_sampling(edge_index, attn_weight, k):
    ''' dynamically sample k neighbors for each node based on attention weights
    
    '''
    if isinstance(edge_index, torch.Tensor):
        # get the index of non-zero
        edge_index = edge_index.coalesce().indices()

    _, sorted_indices = torch.topl(attn_weight, k, descending=True)
    
    

