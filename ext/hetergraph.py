'''
 # @ Create Time: 2025-04-25 15:10:02
 # @ Modified time: 2025-04-25 15:10:07
 # @ Description: create heterogenous data with formal method

all the edge types:

    Package_Name - Action - Path
    Package_Name - DNS - DNS Host
    Package_Name - CMD - Command
    Package_Name - Socket - IP
    Package_Name - Socket - Port
    Package_Name - Socket - Hostnames

 '''

import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
from torch_geometric.utils import subgraph
from torch_geometric.data import HeteroData
import torch
from utils import prostring
from ext import fea_encoder


str_node_list = ['Package_Name','Path','DNS Host', 'Command', "IP", "Hostnames", "Sockets"]
long_str_node_list = ['Path', "Command"]
cate_node_list = ['Action', "DNS"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define global diverse encoders
seq_encoder = fea_encoder.SeqEncoder(device=device)
iden_encoder = fea_encoder.IdenEncoder(dtype=torch.float, \
                                                output_dim=seq_encoder.embedding_dim)

# default embedding_size for category-like value
embedding_dim = 16
cate_encoder = fea_encoder.CateEncoder(embedding_dim)


def get_or_add_node(node, global_node_id_map, global_node_counter):
    """ Ensure that the node is in the global map and return its index. If not, add it to the map. """
    # Ensure the node is in the global map
    node_idx = global_node_id_map.get(node)
    
    if node_idx is None:
        # If not in global map, add it
        node_idx = global_node_counter
        global_node_id_map[node] = global_node_counter
        global_node_counter += 1
    
    return node_idx, global_node_id_map, global_node_counter


def hetero_graph_build(subgraph, global_node_id_map, global_node_counter):

    data = HeteroData()

    nodes = subgraph['nodes']
    edges = subgraph['edges']

    # process nodes
    node_indices = []
    for idx, node in enumerate(nodes):
        node_value = node['value']
        node_type = node['type']
        node_eco = node['eco']
        # check node type process long-length path/command to shorter version 
        if node_type in long_str_node_list:
            node_value = prostring.process_string(node_value)
        
        # check the global unique id
        if node_value not in global_node_id_map:
            global_node_id_map[node_value] = global_node_counter
            global_node_counter += 1

        global_node_id = global_node_id_map[node_value]

        # encode node value based on matched types
        if node_type in str_node_list:
            node_features = seq_encoder(node_value)
        elif node_type in cate_node_list:
            node_features = seq_encoder(node_value)
        # default use categorical encoder for eco
        eco_feature = cate_encoder(node_eco)
        
        # concatenate encoded node value with encode eco value 
        combined_features = torch.cat((node_features, eco_feature), dim=1)

        if node_type not in data:
            data[node_type] = HeteroData()  # Ensure that the node type is initialized

        # add node features for respective type and set num_nodes for each node type explicitly

        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            data[node_type].x = torch.cat((data[node_type].x, combined_features), dim=0)
        else:
            data[node_type].x = combined_features

        node_indices.append(global_node_id)


    # process edges (aligning edge indices with global node IDs)
    edge_indices = []
    for edge in edges:
        # Use get_or_add_node to handle source and target nodes
        source_idx, global_node_id_map, global_node_counter = \
            get_or_add_node(edge['source'], global_node_id_map, global_node_counter)
        target_idx, global_node_id_map, global_node_counter = \
            get_or_add_node(edge['target'], global_node_id_map, global_node_counter)

        edge_index = torch.tensor([[source_idx],[target_idx]], dtype=torch.long, device=device)

        # add edge attribute (one-hot encoding for edge "value")
        edge_value = edge['value']
        edge_attr = cate_encoder(edge_value)

        edge_indices.append(edge_index)

        edge_type = edge['type']

        # add edges to the corresponding relationship in HeteroData
        if edge_type == "Action":
            if 'Action' not in data.edge_types:
                data["Package_Name", "Action", "Path"].edge_index = edge_index
                data["Package_Name", "Action", "Path"].edge_attr = edge_attr
            else:
                data['Package_Name', 'Action', 'Path'].edge_index = \
                        torch.cat((data['Package_Name', 'Action', 'Path'].edge_index, edge_index), dim=1)
                data['Package_Name', 'Action', 'Path'].edge_attr = \
                        torch.cat((data['Package_Name', 'Action', 'Path'].edge_attr, edge_attr), dim=0)
        
        elif edge_type == 'DNS':
            if 'DNS' not in data.edge_types:
                data['Package_Name', 'DNS', 'DNS Host'].edge_index = edge_index
                data['Package_Name', 'DNS', 'DNS Host'].edge_attr = edge_attr
            else:
                data['Package_Name', 'DNS', 'DNS Host'].edge_index = \
                    torch.cat((data['Package_Name', 'DNS', 'DNS Host'].edge_index, edge_index), dim=1)
                data['Package_Name', 'DNS', 'DNS Host'].edge_attr = \
                    torch.cat((data['Package_Name', 'DNS', 'DNS Host'].edge_attr, edge_attr), dim=0)

        elif edge_type == 'CMD':
            if 'CMD' not in data.edge_types:
                data['Package_Name', 'CMD', 'Command'].edge_index = edge_index
                data['Package_Name', 'CMD', 'Command'].edge_attr = edge_attr
            else:
                data['Package_Name', 'CMD', 'Command'].edge_index = \
                    torch.cat((data['Package_Name', 'CMD', 'Command'].edge_index, edge_index), dim=1)
                data['Package_Name', 'CMD', 'Command'].edge_attr = \
                    torch.cat((data['Package_Name', 'CMD', 'Command'].edge_attr, edge_attr), dim=0)

        elif edge_type == 'Socket':
            # Handle multiple Socket edge types (Socket - IP, Socket - Port, Socket - Hostnames)
            if edge['target'] in ['IP']:
                if 'Socket' not in data.edge_types:
                    data['Package_Name', 'Socket', 'IP'].edge_index = edge_index
                    data['Package_Name', 'Socket', 'IP'].edge_attr = edge_attr
                else:
                    data['Package_Name', 'Socket', 'IP'].edge_index = \
                        torch.cat((data['Package_Name', 'Socket', 'IP'].edge_index, edge_index), dim=1)
                    data['Package_Name', 'Socket', 'IP'].edge_attr = \
                        torch.cat((data['Package_Name', 'Socket', 'IP'].edge_attr, edge_attr), dim=0)
                    
            elif edge['target'] in ['Port']:
                data['Package_Name', 'Socket', 'Port'].edge_index = edge_index
                data['Package_Name', 'Socket', 'Port'].edge_attr = edge_attr

            elif edge['target'] in ['Hostnames']:
                data['Package_Name', 'Socket', 'Hostnames'].edge_index = edge_index
                data['Package_Name', 'Socket', 'Hostnames'].edge_attr = edge_attr

    return data, global_node_id_map, global_node_counter


# def mask_subgraph(data: HeteroData):

#     # Subgraph extraction and node/edge masking
#     # Create a mask for nodes in the subgraph (e.g., for nodes 0 and 1)
#     node_mask = torch.tensor([True, True, False])  # Example: mask out node 2
#     edge_mask = torch.tensor([True, False])  # Example: mask out second edge

#     # Subgraph extraction using the node mask
#     subgraph_data = subgraph(node_mask, data.edge_index, relabel_nodes=True)

#     # Apply edge mask to subgraph
#     subgraph_data.edge_index = subgraph_data.edge_index[:, edge_mask]

if __name__ == "__main__":
    pass