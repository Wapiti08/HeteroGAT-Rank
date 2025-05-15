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


class DynamicNeighborSampler(NeighborSampler):
    def __init__(self, edge_index, num_nodes, num_neighbors, edge_weights=None, node_weights=None, **kwargs):
        super(DynamicNeighborSampler, self).__init__(edge_index, num_nodes, num_neighbors, **kwargs)
        # edge and node weights
        self.node_weights = node_weights
        self.edge_weights = edge_weights

    def _get_node_probs(self, atten_weights, node_idx):
        ''' calculate node probabilities based on attention weights of nodes
        
        here, the attention weights of tgt nodes is default equal to edge attention

        args:
            attn_weights: tensor, attention weights of edges: (src_type, edge_type, tgt_type)
            node_idx: tensor, indices of nodes to sample

        '''

        # initialize a tensor
        node_probs = torch.zeros(len(node_idx))

        # iterate over the atten_weights, extract the score of tgt_node
        

    
    def dynamic_nodes_sample(self, out_dict, atten_weights, inputs, neg_sampling=None):
        ''' use attention based sampling to dynamically weighting sample neighbors
        
        args:
            out_dict: dict, output of the attention layer
            atten_weights: tensor, attention weights of nodes
        '''



    

