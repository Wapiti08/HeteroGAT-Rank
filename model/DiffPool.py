'''
 # @ Create Time: 2025-03-17 14:12:31
 # @ Modified time: 2025-05-07 10:01:22
 # @ Description: implement differential pooling for heterogeneous graph
 '''



from math import ceil
import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()
        # 
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class HeteroDiffPool(torch.nn.Module):
    ''' use two GNNs to pool the heterogenous graph
    
    '''
    def __init__(self, node_types, edge_types, num_features, num_classes, max_nodes=150):
        super(HeteroDiffPool, self).__init__()

        # initialize the pooling and embedding networks for per node type
        self.gnns = torch.nn.ModuleDict()

        for node_type in node_types:
            for edge_type in edge_types:
                self.gnns[f'{node_type}_{edge_type}'] = torch.nn.ModuleDict(
                    {
                    "gnn_pool": GNN(num_features, 64, max_nodes),
                    "gnn_embed": GNN(num_features, 64, 64, lin=False)
                })

        # for binary classification, output layer is 2
        self.lin1 = torch.nn.Linear(3 * 64 , 64)
        # num_classes is 2 for binary classification
        self.lin2 = torch.nn.Linear(64, num_classes) 


    def forward(self, x_dict, edge_index_dict, mask_dict=None):
        '''
        args:
            x_dict: dict of node types and their features
            edge_index_dict: dict of edge types and their indices
            mask_dict: dict where keys are node types, values are masks for nodes
        
        return:
            predicted probabilities for binary classification 
        '''

        # score dict and embed dict
        s_dict, x_dict_embed = {}, {}

        for node_type in x_dict.keys():
            for (src_node_type, edge_type, tgt_node_type) in edge_index_dict.keys():
                # Make sure we're processing edges of the correct node type
                if node_type == src_node_type:
                    # get the GNN for the current node and edge type
                    gnn_pool = self.gnns[f'{node_type}_{edge_type}']["gnn_pool"]
                    gnn_embed = self.gnns[f'{node_type}_{edge_type}']["gnn_embed"]

                    # apply the pooling GNN
                    s_dict[node_type] = gnn_pool(x_dict[node_type], edge_index_dict[edge_type], mask_dict[node_type])
                    
                    # apply the embedding GNN
                    x_dict_embed[node_type] = gnn_embed(x_dict[node_type], edge_index_dict[edge_type], mask_dict[node_type])

        # concatenate the scores and embeddings
        x_all = []
        adj_all = []

        for node_type in x_dict.keys():
            for (src_node_type, edge_type, tgt_node_type) in edge_index_dict.keys():
                # get the scores and embeddings for the current node and edge type
                edge_index = edge_index_dict.get((src_node_type, edge_type, tgt_node_type))
                # Check if the edge_index contains only zeros
                if torch.all(edge_index == 0):
                    print(f"Skipping zero-value edge_index for {src_node_type} -> {tgt_node_type}")
                    continue  # Skip processing this edge type

                print(edge_index)
                # Perform dense pooling only for non-zero edge_index tensors
                x, adj, l1, e1 = dense_diff_pool(x_dict_embed[node_type], edge_index,
                                                s_dict[node_type], mask_dict.get(node_type))
                x_all.append(x)
                adj_all.append(adj)
    
        # concatenate the pooled features and adjacency matrices
        x_all = torch.cat(x_all, dim=-1)
        # average over the subgraph
        x_all = x_all.mean(dim=1)

        # classification
        x = self.lin1(x_all).relu()
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)
