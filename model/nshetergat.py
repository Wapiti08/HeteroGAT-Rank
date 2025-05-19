'''
 # @ Create Time: 2025-05-11 14:58:00
 # @ Modified time: 2025-05-11 14:58:09
 # @ Description: implement neighbor sampling based efficient computation for heterogeneous graph training
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
from torch_geometric.sampler.neighbor_sampler import NeighborSampler

# predefined node types
node_types = ["Path", "DNS Host", "Package_Name", "IP", "Command", "Port"]
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class NSHeteroGAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_clusters, fanout=10):
        ''' 
        args:
            fanout: number of neighbors to sample for each node --- start with small number 10

        '''
        super(NSHeteroGAT, self).__init__()

        self.edge_types = [
            ('Package_Name', 'Action', 'Path'),
            ('Package_Name', 'DNS', 'DNS Host'),
            ('Package_Name', 'CMD', 'Command'),
            ('Package_Name', 'Socket', 'IP'),
            ('Package_Name', 'Socket', 'Port'),
            ('Package_Name', 'Socket', 'Hostnames'),
        ]
    
        # GAT layers
        self.conv1 = HeteroConv(
            {et: GATv2Conv((-1, -1), hidden_channels, heads=num_heads,
                         add_self_loops=False)
             for et in self.edge_types},
              # aggregation method for multi-head attention
            aggr='mean',
        )

        # define second layer
        self.conv2 = HeteroConv(
            {et: GATv2Conv((-1, -1), hidden_channels, heads=num_heads,
                         add_self_loops=False)
             for et in self.edge_types},
            aggr='mean',
        )

        # diffpool layer for hierarchical feature aggregation
        self.hetero_gnn = diffpool.HeteroGNN(
            node_types, in_channels, hidden_channels, out_channels
            )

        # define fixed number for edge selection
        self.fanout = fanout

        # Classifier for binary classification (output of size 1), consider extra input for edge info
        self.classifier = torch.nn.Linear(out_channels * num_heads, 1)
        

    def neighbor_sample(self, data):
        ''' sample neighbors for each edge type
        args:
            data: Heterogeneous graph data object
        '''
        sampler = NeighborSampler(data, num_neighbors=self.fanout * len(self.edge_types),\
                        replace=False, directed=False)
        
        sampler = sampler.to(device)
        
        edge_sampler = sampler.

    

if __name__ == "__main__":
    # perform an ablation study
    fanout_list = [10, 20, 50, 100, 200, 500, 1000]

