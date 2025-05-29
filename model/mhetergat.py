import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, global_mean_pool
from optim import diffpool
import torch.nn.functional as F
import os
from utils import evals
from utils.pregraph import *
from collections import defaultdict
from torch.nn import LazyLinear
import pandas as pd

# predefined node types
node_types = ["Path", "DNS Host", "Package_Name", "IP", "Command", "Port"]
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


class MaskedHeteroGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, processed_dir):
        '''
        args:
            in_channels: the dimensionality of the input node features
            hidden_channels: the dimensionality of the hidden node embeddings
            out_channels: the output dimension of the GAT layer before passing to the cls head
            num_heads: the number of attention heads in the GATv2Conv layers
            num_edges: the number of edges in a batch
            num_nodes: the number of nodes in a batch
        '''
        super(MaskedHeteroGAT, self).__init__()

        self.edge_types = [
            ('Package_Name', 'Action', 'Path'),
            ('Package_Name', 'DNS', 'DNS Host'),
            ('Package_Name', 'CMD', 'Command'),
            ('Package_Name', 'socket', 'IP'),
            ('Package_Name', 'socket', 'Port'),
        ]
    
        # GAT layers
        self.conv1 = HeteroConv(
            {et: 
            #  GATv2Conv((-1, -1), hidden_channels, heads=num_heads,
            #              add_self_loops=False)
             GATv2Conv(
                in_channels=(400, 400),  
                out_channels=64,  
                heads=num_heads,  
                concat=True,  
                negative_slope=0.2,
                add_self_loops=False
            )
             for et in self.edge_types},
              # aggregation method for multi-head attention
            aggr='mean',
        )

        # define second layer
        self.conv2 = HeteroConv(
            {et: 
             GATv2Conv(
                in_channels=(400, 400),  
                out_channels=64,  
                heads=num_heads,  
                concat=True,  
                negative_slope=0.2,
                add_self_loops=False
            )
             for et in self.edge_types},
            aggr='mean',
        )

        # diffpool layer for hierarchical feature aggregation
        self.hetero_gnn = diffpool.HeteroGNN(
            in_channels, hidden_channels, out_channels
            )

        node_id_map_file = Path(processed_dir).parent.joinpath('process_state.pkl')
        
        global_node_id_map = load_global_node_id_map(node_id_map_file)
        self.reverse_node_id_map = {v: k for k, v in global_node_id_map.items()}

        # for ranking purpose
        self.global_edge_atten_map = defaultdict(float)
        self.global_node_rank_map = defaultdict(float)
        self.global_node_eco_system_map = defaultdict(lambda: defaultdict(float))

        # Classifier for binary classification (output of size 1), consider extra input for edge info
        self.classifier = LazyLinear(1)

    def cal_attn_weight(self, conv_module, x_dict, edge_index_dict):
        ''' Custom version of HeteroGonv that returns attention weights
        
        returns:
            - updated x_dict
            - attn_weights: Dict[edge_type] = attention tensor
        '''
        attn_weights = {}
        out_dict = {}

        ori_x_dict = x_dict.copy()

        # Create a dictionary to store the mapping between nodes and attention scores
        edge_atten_map = {}

        # create a dict to save edge_list with original node value
        edge_index_map = {}

        # generate projection from global to local
        g2l_map = global_to_local_map(x_dict, edge_index_dict)

        for edge_type, conv in conv_module.convs.items():
            src_type, _, tgt_type = edge_type

            # ensure src_type is "Package_Name"
            if src_type == 'Package_Name':
                if src_type not in x_dict:
                    print(f"Warning: {src_type} not in x_dict, skipping this edge type.")
                    continue  # If Package_Name is missing, skip this edge type
                x_src = x_dict[src_type]
        
            # Ensure that tgt_type exists in x_dict
            if tgt_type not in x_dict:
                print(f"Warning: {tgt_type} not in x_dict, skipping this edge type.")
                continue  # Skip if the target type is missing

            x_tgt = x_dict[tgt_type]
            x_src = x_dict[src_type]
            edge_index = edge_index_dict[edge_type]

            if edge_index.is_sparse:
                edge_index = edge_index.coalesce().indices()

            local_edge_index = remap_edge_indices(edge_index, g2l_map, src_type, tgt_type)


            # shape of alpha is [num_edges, num_heads]
            out, (_, alpha) = conv((x_src, x_tgt), local_edge_index, return_attention_weights=True)

            assert local_edge_index.shape[1] == alpha.shape[0], \
            f"Edge count mismatch: edge_index has {local_edge_index.shape[1]}, but alpha has {alpha.shape[0]}"

            # collect edges in terms of actual node values
            edge_list = []

            # reverse lookup of global node indices to original node values
            for i in range(local_edge_index.shape[1]):
                src_node_idx = edge_index[0, i].item()
                tgt_node_idx = edge_index[1, i].item()

                # retrieve the original node values using reverse lookup
                src_node_value = get_ori_node_value(src_node_idx, self.reverse_node_id_map)
                tgt_node_value = get_ori_node_value(tgt_node_idx, self.reverse_node_id_map)

                # Store the edge, original node values, and the attention score in the map
                edge_atten_map[(src_node_value, tgt_node_value)] = alpha[i, 0].item()

                # collect edge in node-value form
                edge_list.append((src_node_value, tgt_node_value))

            attn_weights[edge_type] = alpha
            out_dict[tgt_type] = out_dict.get(tgt_type, 0) + out
            edge_index_map[edge_type] = edge_list
        
        # After the first pass, merge the updated features with the original ones for all node types
        for node_type in ori_x_dict:
            if node_type not in out_dict:
                out_dict[node_type] = ori_x_dict[node_type]

        return out_dict, attn_weights, edge_atten_map

    def forward(self, batch, **kwargs):
        '''
        args:
            batch: HeteroData type with node_types -> x and edge_types -> edge_index and edge_attr

        '''
        x_dict, edge_index_dict, edge_attr_dict = batch_dict(batch)
        # create the mapping from global indices to local indices
        # ---- first conv with attention
        x_dict_1, attn_weights_1, edge_atten_map_1, edge_index_map_1 = self.cal_attn_weight(self.conv1, x_dict, edge_index_dict)
        
        x_dict = {key: F.relu(x) for key, x in x_dict_1.items()}

        # ---- diffpool per node type, excluding "Package_Name"
        s_dict = self.hetero_gnn(x_dict, edge_index_dict)

        pooled_x_dict, pooled_adj_dict, link_loss, ent_loss = diffpool.hetero_diff_pool(
                    x_dict, edge_index_dict, s_dict
                )
        
        # last attention weight calculation after pooling
        x_dict_pooled, attn_weights_pooled, edge_atten_map_2, edge_index_map_2 = self.cal_attn_weight(self.conv2, pooled_x_dict, pooled_adj_dict)

        # --- continue with binary classification task at subgraph-level

        # Aggregate edge features
        # agg_edge = self.agg_edge_features(edge_attr_dict)

        # ---- pooling per node type, excluding "Package_Name"
        pooled_outputs = []
        # aggregate node features
        for node_type, x in x_dict_pooled.items():
            # global_mean_pool does not support sparse tensors
            x = x.to_dense()
            pooled = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(x.device))
            pooled_outputs.append(pooled)
    
        # ---- final classification
        graph_embed = torch.cat(pooled_outputs, dim=-1)

        # if agg_edge is not None:
        #     combined_features = torch.cat([all_pooled_nodes, agg_edge.view(-1,1)], dim=-1)
        # else:
        #     combined_features = all_pooled_nodes
        
        logits = self.classifier(graph_embed).squeeze(-1)

        return logits, link_loss + ent_loss, attn_weights_pooled, edge_atten_map_2, edge_index_map_2
    
    
    def evaluate(self, logits, batch, threshold=0.5):
        metrics = evals.evaluate(logits, batch, threshold)
        print("evaluation result: \n", metrics )
        return metrics

    def plot_metrics(self, y_true, y_prob, metrics):
        ''' plot roc and calculated metrics
        
        '''
        # define the default save name
        roc_save_path = "roc_curve_maskheterogat.png"
        metric_save_path = "metrics_bar_maskheterogat.png"

        evals.plot_roc(y_true, y_prob, roc_save_path)
        evals.plot_metrics_bar(metrics, metric_save_path)


    def agg_edge_features(self, edge_attr_dict):
        ''' aggregate edge attribute globally

        '''
        # Aggregate the edge attributes by computing the mean for each edge type
        agg_edges = [
            edge_attr.mean(dim=0) for edge_attr in edge_attr_dict.values()
        ]

        # If edge_pooled is available, use it instead
        return torch.cat(agg_edges, dim=-1) if len(agg_edges) > 0 else None


    def rank_edges(self, atten_weights, edge_index_map, k:int, noise_factor:float):
        ''' rank the edges based on the attention weights
        '''

        top_k_edges_indices = {}
        for edge_type, weights in atten_weights.items():
            if edge_type not in edge_index_map:
                print(f"Warning: {edge_type} not found in edge_index_map. Skipping.")
                continue
            # weights is a tensor of shape (num_edges, num_heads)
            # Compute the mean attention weight for each edge across all heads
            mean_weights = weights.mean(dim=1).to(device)  # shape: [num_edges]

            # add random noise to break ties in case of identical attention weights
            if noise_factor:
                noise = torch.randn(mean_weights.shape) * noise_factor
                noise = noise.to(device)
                mean_weights = mean_weights + noise
            # sort the edges by their mean attention weight
            sorted_indices = torch.argsort(mean_weights, descending=True)

            # select the top k edges 
            top_k_edge_indices = sorted_indices[:k]

            # get the corresponding edge
            top_k_edge_for_type = top_k_edge_indices.tolist()

            # add the selected top k edges for the current edge type
            top_k_edges_indices[edge_type] = top_k_edge_for_type

        top_k_edges = {}

        for edge_type in atten_weights.keys():
            if edge_type not in top_k_edges_indices:
                # If edge_type was skipped earlier, assign an empty list
                top_k_edges[edge_type] = []
                continue

            edge_list = edge_index_map.get(edge_type, [])
            selected_edges = [edge_list[i] for i in top_k_edges_indices[edge_type] if i < len(edge_list)]
            top_k_edges[edge_type] = selected_edges
    
        return top_k_edges
    
    def rank_nodes_by_eco_system(self, edge_atten_map, node_json:list, k):
        ''' Rank target nodes (tgt_node_value) by eco_system consideration
        
        '''
        # Create a dictionary to store attention scores by eco_system and tgt_node_value
        eco_system_rank_map = defaultdict(lambda: defaultdict(float))

        df = pd.DataFrame(node_json)

        # aggregate attention weights for each node based on its adjacent edges
        for (src_node_value, tgt_node_value), attn_weight in edge_atten_map.items():
            # Find the eco_system for this node type in the DataFrame
            src_node_eco = df[df['value'] == src_node_value]['eco'].values
            if len(src_node_eco) == 0:
                continue  # Skip if eco_system for source node is not found
            src_node_eco = src_node_eco[0]  # Assuming the eco_system is unique for each node

            # Aggregate attention score by eco_system for tgt_node_value
            eco_system_rank_map[src_node_eco][tgt_node_value] += attn_weight

        # Sort and get top k nodes per eco_system
        top_k_by_eco_system = {}
        for eco_system, tgt_nodes in eco_system_rank_map.items():
            sorted_tgt_nodes = sorted(tgt_nodes.items(), key=lambda x: x[1], reverse=True)
            top_k_by_eco_system[eco_system] = [node for node, score in sorted_tgt_nodes[:k]]

        return top_k_by_eco_system

    def rank_nodes_global(self, edge_atten_map, k):
        ''' Rank target nodes (tgt_node_value) without considering eco_system '''

        # Create a dictionary to store aggregated attention scores for each target node
        tgt_node_rank_map = defaultdict(float)

        # Iterate over edge_attention_map to aggregate attention weights for each tgt_node_value
        for (src_node_value, tgt_node_value), attn_weight in edge_atten_map.items():
            tgt_node_rank_map[tgt_node_value] += attn_weight
        
        # Sort the target nodes by their aggregated attention scores
        sorted_tgt_nodes = sorted(tgt_node_rank_map.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top k target nodes globally
        top_k_global_nodes = [node for node, score in sorted_tgt_nodes[:k]]

        return top_k_global_nodes

    def final_sample(self, top_k_edges, top_k_nodes_by_eco, top_k_global_nodes):
        '''
        Final sampling that returns actual node and edge values
        Params:
            atten_weights: dict[edge_type -> Tensor[num_edges, num_heads]]
            edge_atten_map: dict[(src_node_value, tgt_node_value) -> attn_score]
            edge_index_map: dict[edge_type -> List[(src_node_value, tgt_node_value)]]
            node_json: list of node dicts with keys including 'value' and 'eco'
            k: top-k value
            noise_factor: float for tie-breaking
        Returns:
            dict with top-k edges/nodes in original value form
        '''

        # step4: final sample - intersect results
        final_selected_nodes = set(top_k_global_nodes)

        for eco_nodes in top_k_nodes_by_eco.values():
            final_selected_nodes.update(eco_nodes)
        
        final_select_edges = {}
        for edge_type, edge_list in top_k_edges.items():
            # filter edges that connect to selected target nodes
            filtered = [
                (src, tgt)
                for (src, tgt) in edge_list
                if tgt in final_selected_nodes or src in final_selected_nodes
            ]
            final_select_edges[edge_type] = filtered

        return {
        'top_k_edges': top_k_edges,
        'top_k_nodes_by_eco': top_k_nodes_by_eco,
        'top_k_global_nodes': top_k_global_nodes,
        'final_selected_nodes': list(final_selected_nodes),
        'final_selected_edges': final_select_edges
        }

#     def compute_loss(self, probs, labels):
#         ''' compute the total loss, including BCE loss and regularization

#         :param probs: model output probabilities (after sigmoid activation)
#         :param labels: ground truth binary labels
#         :return: total loss (BCE loss + regularization)

#         '''
#         bce_loss = F.binary_cross_entropy(probs, labels.float())
#         reg_loss = mask_regularization(self.edge_mask, self.node_mask)

#         return bce_loss + reg_loss

# # regularization term to encourage sparsity
# def mask_regularization(edge_mask, node_mask, lambda_reg=0.01):
#     return lambda_reg * (torch.sum(torch.abs(edge_mask)) + torch.sum(torch.abs(node_mask)))

