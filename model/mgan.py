'''
 # @ Create Time: 2025-01-03 16:05:44
 # @ Modified time: 2025-01-03 16:06:18
 # @ Description: create attention-based Graph Neural Networks to learn feature importance
 '''

import torch
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric import nn

class MaskedHeteroGAT(torch.nn.Module):
    def __init__(self,):
        super(MaskedHeteroGAT, self).__init__()
        self.conv1 = 






class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_heads):
        super(HeteroConv, self).__init__()

        # define first layer
        self.conv1 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATConv((-1，1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，1), hidden_channels, heads=num_heads),

            },
            aggr='mean',
        )

        # define second layer
        self.conv2 = HeteroConv(
            {
                ()
            },
            aggr='mean',
        )
        # add final projection layer to output logits for binary classification
        self.classifier = nn.Linear(out_channels, 1)


    def forward(self, x_dict, edge_index_dict):
        '''
        :param x_dict:
        :param edge_index_dict: 
        '''
