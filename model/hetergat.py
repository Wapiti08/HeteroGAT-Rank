
import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from utils import sparsepad
from torch_geometric.nn import HeteroConv, GATConv,global_mean_pool
from torch import nn
import torch.nn.functional as F
import os
from utils import evals
from utils.pregraph import *
from collections import defaultdict
from torch.nn import LazyLinear, LayerNorm
import pandas as pd

# predefined node types
node_types = ["Path", "DNS Host", "Package_Name", "Hostnames", "IP", "Command", "Port"]

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class HeterGAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, edge_attr_dim, processed_dir):
        super(HeterGAT, self).__init__()

        self.edge_types = [
            ('Package_Name', 'Action', 'Path'),
            ('Package_Name', 'DNS', 'DNS Host'),
            ('Package_Name', 'CMD', 'Command'),
            ('Package_Name', 'socket', 'IP'),
            ('Package_Name', 'socket', 'Port'),
            ('Package_Name', 'socket', 'Hostnames'),
        ]

        self.conv1 = HeteroConv(
            {et: GATConv((-1, -1), hidden_channels, heads=num_heads,
                         add_self_loops=False, edge_dim = edge_attr_dim)
             for et in self.edge_types},
            aggr='mean',
        )

        self.conv2 = HeteroConv(
            {et: GATConv((-1, -1), hidden_channels, heads=num_heads,
                            add_self_loops=False, edge_dim = edge_attr_dim)
             for et in self.edge_types},
            aggr='mean',
        )

        # layernorm for each node type after conv1 and conv2
        self.ln1 = torch.nn.ModuleDict()
        self.ln2 = torch.nn.ModuleDict()

        for node_type in ["Package_Name", 'Path', 'DNS Host', 'Command', 'IP', 'Port']:
            self.ln1[node_type] = LayerNorm(hidden_channels * num_heads)
            self.ln2[node_type] = LayerNorm(hidden_channels * num_heads)

        # Binary cross-entropy loss with optional class weighting
        self.loss_fn = nn.BCEWithLogitsLoss()

        node_id_map_file = Path(processed_dir).parent.joinpath('process_state.pkl')

        global_node_id_map = load_global_node_id_map(node_id_map_file)
        self.reverse_node_id_map = {v: k for k, v in global_node_id_map.items()}

        # for ranking purpose
        self.global_edge_atten_map = defaultdict(float)
        self.global_node_rank_map = defaultdict(float)
        self.global_node_eco_system_map = defaultdict(lambda: defaultdict(float))

        # dynamic track the number of in_channels
        self.classifier = LazyLinear(1)

    def cal_attn_weight(self, conv_module, x_dict, edge_index_dict):
        ''' custom version of HeteroGonv that returns attention weights for bipartite graph
        
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

        return out_dict, attn_weights, edge_atten_map, edge_index_map


    def forward(self, batch, **kwargs):
        '''
        args:
            batch: HeteroData type with node_types -> x and edge_types -> edge_index and edge_attr
        '''
        x_dict, edge_index_dict, edge_attr_dict = batch_dict(batch)

        # ---- first conv with attention
        x_dict_1, attn_weights_1, edge_atten_map_1, edge_index_map_1 = self.cal_attn_weight(self.conv1, x_dict, edge_index_dict)
        
        # for debug
        # for node_type, x in x_dict_1.items():
        #     print(f"[x_dict_1] {node_type}: shape={x.shape}")

        x_dict = {}
        # bypass the process for Package_Name
        for node_type, x in x_dict_1.items():
            if node_type in self.ln1 and x.shape[-1] == 256:
                x_dict[node_type] = F.relu(self.ln1[node_type](x))
            else:
                x_dict[node_type] = x
        
        # ---- second conv with attention
        x_dict_2, attn_weights_2, edge_atten_map_2, edge_index_map_2 = self.cal_attn_weight(self.conv2, x_dict, edge_index_dict)

        # apply layernorm and relu again
        x_dict = {}
        # bypass the process for Package_Name
        for node_type, x in x_dict_2.items():
            if node_type in self.ln1 and x.shape[-1] == 256:
                x_dict[node_type] = F.relu(self.ln1[node_type](x))
            else:
                x_dict[node_type] = x

        # ---- pooling per node type, excluding "Package_Name"
        pooled_outputs = []

        for node_type, x in x_dict.items():
            # global_mean_pool does not support sparse tensors
            x = x.to_dense()
            pooled = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(x.device))
            pooled_outputs.append(pooled)

        # ---- final classification
        graph_embed = torch.cat(pooled_outputs, dim=-1)

        # in_channels = graph_embed.shape[1]
        # print("the input channels are:", in_channels)
        # dynamically infer classifier input size
        logits = self.classifier(graph_embed).squeeze(-1)

        return logits, attn_weights_2, edge_atten_map_2, edge_index_map_2
    
    
    def compute_loss(self, logits, batch):
        ''' computer the binary classification loss

        :param logits: the pooled total loss for subgraph-level classification
        :param x_dictL the dictionary of ground truth labels for each node type
        
        return: total loss across all node types
        '''
        y_true = batch['label'].float()
        assert logits.shape == y_true.shape, f"Shape mismatch: {logits.shape} vs {y_true.shape}"
        
        return self.loss_fn(logits, y_true)

    def evaluate(self, logits, batch, threshold=0.5):
        metrics = evals.evaluate(logits, batch, threshold)
        print("evaluation result: \n", metrics )
        return metrics

    def plot_metrics(self, y_true, y_prob, metrics):
        ''' plot roc and calculated metrics
        
        '''
        # define the default save name
        roc_save_path = "roc_curve_heterogat.png"
        metric_save_path = "metrics_bar_heterogat.png"

        evals.plot_roc(y_true, y_prob, roc_save_path)
        evals.plot_metrics_bar(metrics, metric_save_path)

    def rank_edges(self, atten_weights, edge_index_map, k:int, noise_factor:float):
        ''' rank the edges based on the attention weights
        '''

        top_k_edges_indices = {edge_type: [] for edge_type in atten_weights}
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


