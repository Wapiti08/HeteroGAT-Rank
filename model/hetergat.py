
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
            ('Package_Name', 'socket_ip', 'IP'),
            ('Package_Name', 'socket_port', 'Port'),
            ('Package_Name', 'socket_host', 'Hostnames'),
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

        for node_type in node_types:
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



