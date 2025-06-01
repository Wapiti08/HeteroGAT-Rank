'''
 # @ Create Time: 2025-04-25 15:10:02
 # @ Modified time: 2025-04-25 15:10:07
 # @ Description: create heterogenous data with formal method

all the edge types:

    ('Package_Name', 'Action', 'Path'),
    ('Package_Name', 'DNS', 'DNS Host'),
    ('Package_Name', 'CMD', 'Command'),
    ('Package_Name', 'socket_ip', 'IP'),
    ('Package_Name', 'socket_port', 'Port'),
    ('Package_Name', 'socket_host', 'Hostnames'),



 '''

import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
from torch_geometric.utils import subgraph
from torch_geometric.data import HeteroData
import torch
from utils import prostring
from ext import fea_encoder
import pickle


str_node_list = ['Package_Name', 'Path', 'DNS Host', "Hostnames", 'Command', "IP", "Port"]
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


def nodes_process(nodes:list, global_node_id_map, global_node_counter, data: HeteroData):
    
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

        # Add node features
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            data[node_type].x = torch.cat((data[node_type].x, combined_features), dim=0)
        else:
            data[node_type].x = combined_features

        node_indices.append(global_node_id)

    # Set num_nodes explicitly
    for node_type in data.node_types:
        data[node_type].num_nodes = data[node_type].x.size(0)

    return data, global_node_id_map, global_node_counter


def edges_process(edges:list, global_node_id_map, global_node_counter, data: HeteroData):
    # initialize the edge_index for individual edge_type
    edge_type_dict = {
    ('Package_Name', 'DNS', 'DNS Host'): [],
    ('Package_Name', 'Action', 'Path'): [],
    ('Package_Name', 'CMD', 'Command'): [],
    ('Package_Name', 'socket_ip', 'IP'): [],
    ('Package_Name', 'socket_port', 'Port'): [],
    ('Package_Name', 'socket_host', 'Hostnames'):[],
    }

    for edge in edges:
        source_idx, global_node_id_map, global_node_counter = \
            get_or_add_node(edge['source'], global_node_id_map, global_node_counter)
        target_idx, global_node_id_map, global_node_counter = \
            get_or_add_node(edge['target'], global_node_id_map, global_node_counter)

        edge_type = edge['type']

        if edge_type == "Action":
            edge_type_dict[('Package_Name', 'Action', 'Path')].append((source_idx, target_idx))
        elif edge_type == 'DNS':
            edge_type_dict[('Package_Name', 'DNS', 'DNS Host')].append((source_idx, target_idx))
        elif edge_type == 'CMD':
            edge_type_dict[('Package_Name', 'CMD', 'Command')].append((source_idx, target_idx))
        elif edge_type == 'socket_ip' and edge['target'] == 'IP':
            edge_type_dict[('Package_Name', 'socket_ip', 'IP')].append((source_idx, target_idx))
        elif edge_type == 'socket_port' and edge['target'] == 'Port':
            edge_type_dict[('Package_Name', 'socket_port', 'Port')].append((source_idx, target_idx))
        elif edge_type == 'socket_host' and edge['target'] == 'Hostnames':
            edge_type_dict[('Package_Name', 'socket_host', 'Hostnames')].append((source_idx, target_idx))

    for edge_type, edge_tuples in edge_type_dict.items():
        if edge_tuples:
            edge_index = torch.tensor(edge_tuples, dtype=torch.long, device=device).t().contiguous()
            
            # get attribute of edge
            edge_value = edge['value']
            edge_attr = cate_encoder(edge_value)

            # process according to edge_type
            if edge_type == ('Package_Name', 'Action', 'Path'):
                if ('Package_Name', 'Action', 'Path') not in data.edge_types:
                    data["Package_Name", "Action", "Path"].edge_index = edge_index
                    data["Package_Name", "Action", "Path"].edge_attr = edge_attr
                else:
                    data['Package_Name', 'Action', 'Path'].edge_index = \
                        torch.cat((data['Package_Name', 'Action', 'Path'].edge_index, edge_index), dim=1)
                    data['Package_Name', 'Action', 'Path'].edge_attr = \
                        torch.cat((data['Package_Name', 'Action', 'Path'].edge_attr, edge_attr), dim=0)

            elif edge_type == ('Package_Name', 'DNS', 'DNS Host'):
                if ('Package_Name', 'DNS', 'DNS Host') not in data.edge_types:
                    data['Package_Name', 'DNS', 'DNS Host'].edge_index = edge_index
                    data['Package_Name', 'DNS', 'DNS Host'].edge_attr = edge_attr
                else:
                    data['Package_Name', 'DNS', 'DNS Host'].edge_index = \
                        torch.cat((data['Package_Name', 'DNS', 'DNS Host'].edge_index, edge_index), dim=1)
                    data['Package_Name', 'DNS', 'DNS Host'].edge_attr = \
                        torch.cat((data['Package_Name', 'DNS', 'DNS Host'].edge_attr, edge_attr), dim=0)

            elif edge_type == ('Package_Name', 'CMD', 'Command'):
                if ('Package_Name', 'CMD', 'Command') not in data.edge_types:
                    data['Package_Name', 'CMD', 'Command'].edge_index = edge_index
                    data['Package_Name', 'CMD', 'Command'].edge_attr = edge_attr
                else:
                    data['Package_Name', 'CMD', 'Command'].edge_index = \
                        torch.cat((data['Package_Name', 'CMD', 'Command'].edge_index, edge_index), dim=1)
                    data['Package_Name', 'CMD', 'Command'].edge_attr = \
                        torch.cat((data['Package_Name', 'CMD', 'Command'].edge_attr, edge_attr), dim=0)

            elif edge_type == ('Package_Name', 'socket_ip', 'IP'):
                if ('Package_Name', 'socket_ip', 'IP') not in data.edge_types:
                    data['Package_Name', 'socket_ip', 'IP'].edge_index = edge_index
                    data['Package_Name', 'socket_ip', 'IP'].edge_attr = edge_attr
                else:
                    data['Package_Name', 'socket_ip', 'IP'].edge_index = \
                        torch.cat((data['Package_Name', 'socket', 'IP'].edge_index, edge_index), dim=1)
                    data['Package_Name', 'socket_ip', 'IP'].edge_attr = \
                        torch.cat((data['Package_Name', 'socket_ip', 'IP'].edge_attr, edge_attr), dim=0)

            elif edge_type == ('Package_Name', 'socket_port', 'Port'):
                if ('Package_Name','socket_port', 'Port') not in data.edge_types:
                    data['Package_Name', 'socket_port', 'Port'].edge_index = edge_index
                    data['Package_Name', 'socket_port', 'Port'].edge_attr = edge_attr
                else:
                    data['Package_Name', 'socket_port', 'Port'].edge_index = \
                        torch.cat((data['Package_Name', 'socket_port', 'Port'].edge_index, edge_index), dim=1)
                    data['Package_Name', 'socket_port', 'Port'].edge_attr = \
                        torch.cat((data['Package_Name', 'socket_port', 'Port'].edge_attr, edge_attr), dim=0)
            
            elif edge_type == ('Package_Name', 'socket_host', 'Hostnames'):
                if ('Package_Name',  'socket_host', 'Hostnames') not in data.edge_types:
                    data['Package_Name',  'socket_host', 'Hostnames'].edge_index = edge_index
                    data['Package_Name',  'socket_host', 'Hostnames'].edge_attr = edge_attr
                else:
                    data['Package_Name',  'socket_host', 'Hostnames'].edge_index = \
                        torch.cat((data['Package_Name', 'socket_host', 'Hostnames'].edge_index, edge_index), dim=1)
                    data['Package_Name', 'socket_host', 'Hostnames'].edge_attr = \
                        torch.cat((data['Package_Name', 'socket_host', 'Hostnames'].edge_attr, edge_attr), dim=0)

    return data, global_node_id_map, global_node_counter


def hetero_graph_build(subgraph, global_node_id_map, global_node_counter):

    data = HeteroData()

    nodes = subgraph['nodes']
    edges = subgraph['edges']
    label = subgraph['label']
    
    # Add the label as a graph-level attribute
    data['label'] = torch.tensor([label], dtype=torch.long)

    data, global_node_id_map, global_node_counter = nodes_process(nodes, global_node_id_map, global_node_counter, data)

    # process edges (aligning edge indices with global node IDs)
    data, global_node_id_map, global_node_counter = edges_process(edges, global_node_id_map, global_node_counter, data)
    
    return data, global_node_id_map, global_node_counter

if __name__ == "__main__":
    pass