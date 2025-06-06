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
import torch_sparse

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
        if node_type not in s_dict:
            continue  # e.g. skip Package_Name

        s = torch.softmax(s, dim=-1)

        # apply mask if porovided
        if mask_dict is not None and node_type in mask_dict:
            mask = mask_dict[node_type].view(x.size(0), x.size(1), 1).to(x.dtype)
            x = x * mask
            s = s * mask

        # pooled node features --- default two clusters
        pooled_x = torch.matmul(s.T, x)
        pooled_x_dict[node_type] = pooled_x

        # entropy loss
        ent_loss += torch.sum(s * torch.log(s + 1e-6), dim=1).mean()

        # adjancy pooling for target node types
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, tgt_type = edge_type
            if tgt_type not in s_dict:
                continue  # only pool if target node type is clustered             
            
            s = torch.softmax(s_dict[tgt_type], dim=-1)
            num_nodes = s.size(0)

            # create sparse matrix
            edge_weight = torch.ones(edge_index.size(1), device=s.device)
            adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))

            try:
                pooled_adj = torch.sparse.mm(s.T, torch.sparse.mm(adj, s))  # (K, K)
                pooled_adj_dict[edge_type] = pooled_adj

                # link prediction loss
                link_loss += torch.norm(torch.sparse.mm(adj, s) - s, p=2)

            except RuntimeError as e:
                print(f"[Warning] Skipped edge_type {edge_type} due to sparse.mm error: {e}")
    
    if normalize and len(pooled_adj_dict) > 0:
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
            # if node_type != "Package_Name":
            # define the SAGEConv layer for each node type --- accept sparse matrix
            self.node_type_encoders[node_type] = SAGEConv(
                in_channels, hidden_channels
            )

        # define the pooling layer for each node type
        self.pooling_layers = nn.ModuleDict()

        for node_type in node_types:
            # if node_type != "Package_Name":
            # learnable transformation of node embeddings
            self.pooling_layers[node_type] = nn.Linear(out_channels, out_channels)


    def forward(self, x_dict: dict, edge_index_dict:dict) -> dict:
        '''
        args:
            x_dict: {node_type: node_features}
            edge_index_dict: {edge_type (three-tuple): edge_indices}
        '''
        s_dict = {}

        print(f"[Debug] edge_index_dict keys: {list(edge_index_dict.keys())}")
        print(f"[Debug] x_dict keys: {list(x_dict.keys())}")

        # process each node type and its corresponding edges
        for node_type, x in x_dict.items():
            if node_type == "Package_Name":
                continue
            # Get the edge indices for each edge type involving this node type
            s = []
            for (src_node_type, edge_type, tgt_node_type), edge_index in edge_index_dict.items():
                # process only tgt_node_type
                if tgt_node_type != node_type:
                    continue
            
                x_input = x_dict[tgt_node_type].to(device)
                edge_index = edge_index.to(device)

                # for security check
                if edge_index.max().item() >= x_input.size(0):
                    print(f"[Index Error] node_type={node_type}, x_input.size(0)={x_input.size(0)}, edge_index.max={edge_index.max().item()}")
                    print(f"edge_index: {edge_index}")
                    raise ValueError(f"Edge index out of bounds for node type {node_type}")
                
                # gnn computation
                node_features = self.node_type_encoders[node_type](x_input, edge_index)
                # compute the assignment matrix for each node type (using softmax)
                pooled_s = self.pooling_layers[node_type](node_features)
                pooled_s = F.softmax(pooled_s, dim=-1)
                s.append(pooled_s)
            
            if len(s)==0:
                print(f"[Warning] Skipping node_type={node_type}, no valid incoming edges.")
                continue
        
            # aggregate the assignment matrices for all edge types
            s = torch.stack(s, dim=0).mean(dim=0)
            s_dict[node_type] = s
        
        return s_dict
