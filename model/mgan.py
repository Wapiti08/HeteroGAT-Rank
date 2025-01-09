'''
 # @ Create Time: 2025-01-03 16:05:44
 # @ Modified time: 2025-01-03 16:06:18
 # @ Description: create attention-based Graph Neural Networks to learn feature importance
 '''

import torch
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric import nn

class MaskedHeteroGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_heads, num_edges, num_nodes):
        super(MaskedHeteroGAT, self).__init__()
    
        # learnable masks
        self. = 
        self.
    







class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_heads):
        super(HeteroConv, self).__init__()

        # define first layer
        self.conv1 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'CMD', 'Command'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'IP'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Port'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Hostnames'): GATConv((-1，-1), hidden_channels, heads=num_heads),
            },
            aggr='mean',
        )

        # define second layer
        self.conv2 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'CMD', 'Command'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'IP'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Port'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Hostnames'): GATConv((-1，-1), hidden_channels, heads=num_heads),
            },
            aggr='mean',
        )
        # add final projection layer to output logits for binary classification
        self.classifier = nn.Linear(out_channels, 1)


    def forward(self, x_dict, edge_index_dict):
        '''
        :param x_dict: a dict holding node feature informaiton for each individual node type
        :param edge_index_dict: a dict holding graph connectivity info for each individual edge type
        '''
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        # project to logits for classification
        logit_dict = {
            key: self.classifier(x).squeeze(-1) for key, x in x_dict.items()
        }

        return logit_dict