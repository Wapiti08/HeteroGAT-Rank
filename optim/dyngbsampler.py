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
    

