'''
 # @ Create Time: 2025-03-17 14:12:31
 # @ Modified time: 2025-05-07 10:01:22
 # @ Description: implement differential pooling for heterogeneous graph
 '''

import torch
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

node_types = ["Path", "DNS Host", "Package_Name", "IP", "Hostnames", "Command", "Port"]


def hetero_diff_pool(
    x_dict: Dict[str, Tensor],
    edge_index_dict: Dict[str, Tensor],
    s_dict: Dict[str, Tensor],
    mask_dict: Optional[Dict[str, Tensor]] = None,
    normalize: bool = True,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor]:
    '''
    Differentiable pooling for heterogeneous graphs.
    
    Args:
        x_dict (dict): A dictionary where keys are node types and values are 
                       the node feature tensors for each node type.
        edge_index_dict (dict): A dictionary where keys are edge types (or pairs 
                                of node types) and values are the edge index tensors.
        s_dict (dict): A dictionary where keys are node types and values are 
                       the assignment tensors (softmax assignments).
        mask_dict (dict, optional): A dictionary where keys are node types and 
                                    values are the mask tensors for valid nodes.
        normalize (bool): Whether to normalize the link prediction loss.
        
    Returns:
        Tuple of (pooled node feature dictionaries, coarsened adjacency dictionaries, 
                  link prediction loss, entropy loss).
    ''' 
    pooled_x_dict = {}
    pooled_adj_dict = {}
    link_loss = 0
    ent_loss = 0

    # process each node type and corresponding edges and assignments
    for node_type, x in x_dict.items():
        # get the assignment tensor for this node type
        s = s_dict[node_type]
        s = torch.softmax(s, dim=-1)

        # apply mask if porovided
        if mask_dict is not None and node_type in mask_dict:
            mask = mask_dict[node_type].view(x.size(0), x.size(1), 1).to(x.dtype)
            x = x * mask
            s = s * mask

        # pooled node features --- default two clusters
        pooled_x = torch.matmul(s.transpose(1, 2), x)
        pooled_x_dict[node_type] = pooled_x

        # for each edge type, handle the adjacency matrix and pooling
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type.startswith(node_type):
                # compute adj matrix for this edge type
                adj = torch.zeros(x.size(0), x.size(1), x.size(1), device=x.device)
                for i in range(x.size(0)):
                    # assume directed edges
                    adj[i, edge_index[0], edge_index[1]] = 1

                # apply pooling to adj matrix
                pooled_adj = torch.matmul(torch.matmul(s.transpose(1, 2),adj), s)
                pooled_adj_dict[edge_type] = pooled_adj

                # link prediction loss
                loss_loss += torch.norm(adj - torch.matmul(s, s.transpose(1,2)), p=2)
        
        # entropy loss
        ent_loss += torch.sum(s * torch.log(s + 1e-6), dim=1).mean()
    
    # normalize the link prediction loss
    if normalize:
        # numel is the number of edges in the graph
        link_loss = link_loss / sum([adj.numel() for adj in pooled_adj_dict.values()])

    return pooled_x_dict, pooled_adj_dict, link_loss, ent_loss

class HeteroGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        '''
        
        '''
        super(HeteroGNN, self).__init__()
        
        # define a DenseSAGEConv layer for each node type
        self.node_type_encoders = nn.ModuleDict()

        for node_type in node_types:
            # define the SAGEConv layer for each node type --- accept sparse matrix
            self.node_type_encoders[node_type] = SAGEConv(
                in_channels, hidden_channels
            )

        # define the pooling layer for each node type
        self.pooling_layers = nn.ModuleDict()

        for node_type in node_types:
            # learnable transformation of node embeddings
            self.pooling_layers[node_type] = nn.Linear(out_channels, out_channels)

    def forward(self, x_dict: dict, edge_index_dict:dict) -> dict:
        '''
        args:
            x_dict: {node_type: node_features}
            edge_index_dict: {edge_type (three-tuple): edge_indices}
        '''
        s_dict = {}

        # process each node type and its corresponding edges
        for node_type, x in x_dict.items():
            # Get the edge indices for each edge type involving this node type
            s = []
            for (src_node_type, edge_type, tgt_node_type), edge_index in edge_index_dict.items():
                if src_node_type == node_type or tgt_node_type == node_type:
                    # apply DenseSAGEConv for the current node type
                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    x = x.float()  # Ensures x is float32
                    edge_index = edge_index.float()  # Ensures edge_index is int64
                    print(x.shape, edge_index.shape)
                    node_features = self.node_type_encoders[node_type](x, edge_index)

                    # compute the assignment matrix for each node type (using softmax)
                    pooled_s = self.pooling_layers[node_type](node_features)
                    pooled_s = F.softmax(pooled_s, dim=-1)
                    s.append(pooled_s)
            
            # aggregate the assignment matrices for all edge types
            s.torch.stack(s, dim=0).mean(dim=0)
            s_dict[node_type] = s
        
        return s_dict



# class HeteroDiffPool(torch.nn.Module):
#     ''' use two GNNs to pool the heterogenous graph
    
#     '''
#     def __init__(self, node_types, edge_types, num_features, num_classes, max_nodes=150):
#         super(HeteroDiffPool, self).__init__()

#         # initialize the pooling and embedding networks for per node type
#         self.gnns = torch.nn.ModuleDict()

#         for node_type in node_types:
#             for (src_node_type, edge_type, tgt_node_type) in edge_types:
#                 if node_type == src_node_type:
#                     self.gnns[f'{node_type}_{edge_type}'] = torch.nn.ModuleDict(
#                         {
#                         "gnn_pool": HeteroGNN(num_features, 64, max_nodes),
#                         "gnn_embed": HeteroGNN(num_features, 64, 64)
#                     })

#         # for binary classification, output layer is 2
#         self.lin1 = torch.nn.Linear(3 * 64 , 64)
#         # num_classes is 2 for binary classification
#         self.lin2 = torch.nn.Linear(64, num_classes) 


#     def forward(self, x_dict, edge_index_dict, mask_dict=None):
#         '''
#         args:
#             x_dict: dict of node types and their features
#             edge_index_dict: dict of edge types and their indices
#             mask_dict: dict where keys are node types, values are masks for nodes
        
#         return:
#             predicted probabilities for binary classification 
#         '''


#         # add blank node feature for Package_Name
#         if "Package_Name" not in x_dict:
#             x_dict["Package_Name"] = torch.zeros((1, x_dict["Path"].size(1)), device=x_dict["Path"].device)

#         # score dict and embed dict
#         s_dict, x_dict_embed = {}, {}
#         # Initialize embed dict for all node types
#         for node_type in x_dict.keys():
#             x_dict_embed[node_type] = []

#         for node_type in x_dict.keys():
#             for edge_index in edge_index_dict.keys():
#                 (src_node_type, edge_type, tgt_node_type) = edge_index
#                 # Make sure we're processing edges of the correct node type
#                 if node_type == src_node_type:
#                     # get the GNN for the current node and edge type
#                     gnn_pool = self.gnns[f'{node_type}_{edge_type}']["gnn_pool"]
#                     gnn_embed = self.gnns[f'{node_type}_{edge_type}']["gnn_embed"]

#                     if mask_dict is None:
#                         # apply the pooling GNN
#                         print(edge_index_dict[edge_index])
#                         if edge_index_dict[edge_index].size(0) == 0:
#                             print(f"Skipping zero-value edge_index for {node_type} -> {tgt_node_type}")
#                             continue

#                         s_dict[node_type] = gnn_pool(x_dict[node_type], edge_index_dict[edge_index])
#                         # apply the embedding GNN
#                         x_dict_embed[node_type] = gnn_embed(x_dict[node_type], edge_index_dict[edge_index])

#                     else:
#                         if edge_index_dict[edge_index].size(0) == 0:
#                             print(f"Skipping zero-value edge_index for {node_type} -> {tgt_node_type}")
#                             continue
#                         # apply the pooling GNN
#                         s_dict[node_type] = gnn_pool(x_dict[node_type], edge_index_dict[edge_index], mask_dict[node_type])
#                         # apply the embedding GNN
#                         x_dict_embed[node_type] = gnn_embed(x_dict[node_type], edge_index_dict[edge_index], mask_dict[node_type])

#         # concatenate the scores and embeddings
#         x_all = []
#         adj_all = []

#         for node_type in x_dict.keys():
#             for edge_index in edge_index_dict.keys():
#                 (src_node_type, edge_type, tgt_node_type) = edge_index
#                 if node_type == src_node_type:
#                     # get the scores and embeddings for the current node and edge type
#                     edge_in = edge_index_dict.get(edge_index)
#                     # Check if the edge_index contains only zeros
#                     if torch.all(edge_in == 0):
#                         print(f"Skipping zero-value edge_index for {src_node_type} -> {tgt_node_type}")
#                         continue  # Skip processing this edge type
#                     if mask_dict is None:
#                         # Perform dense pooling only for non-zero edge_index tensors
#                         x, adj, l1, e1 = hetero_diff_pool(x_dict_embed[node_type], edge_in, s_dict[node_type])
#                     else:
#                         # Perform dense pooling only for non-zero edge_index tensors
#                         x, adj, l1, e1 = hetero_diff_pool(x_dict_embed[node_type], edge_in,
#                                                         s_dict[node_type], mask_dict.get(node_type))
                    
#                     x_all.append(x)
#                     adj_all.append(adj)

#         # concatenate the pooled features and adjacency matrices
#         x_all = torch.cat(x_all, dim=-1)
#         # average over the subgraph
#         x_all = x_all.mean(dim=1)

#         # classification
#         x = self.lin1(x_all).relu()
#         x = self.lin2(x)

#         return F.log_softmax(x, dim=-1)
