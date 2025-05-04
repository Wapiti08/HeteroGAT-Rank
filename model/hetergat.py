
import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.nn import HeteroConv, GATConv, GATv2Conv, global_mean_pool
from model import DiffPool
from torch import nn
import torch.nn.functional as F
from ext.iter_loader import IterSubGraphs
from torch_geometric.loader import DataLoader
from datetime import datetime
from torch_geometric.data import HeteroData
import os
from sklearn.model_selection import train_test_split
from utils import evals
from utils.pregraph import *

# predefined node types
node_types = ["Path", "DNS Host", "Package_Name", "IP", "Hostnames", "Command", "Port"]


class HeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads):
        super(HeteroGAT, self).__init__()

        self.edge_types = [
            ('Package_Name', 'Action', 'Path'),
            ('Package_Name', 'DNS', 'DNS Host'),
            ('Package_Name', 'CMD', 'Command'),
            ('Package_Name', 'Socket', 'IP'),
            ('Package_Name', 'Socket', 'Port'),
            ('Package_Name', 'Socket', 'Hostnames'),
        ]

        self.conv1 = HeteroConv(
            {et: GATConv((-1, -1), hidden_channels, heads=num_heads,
                         add_self_loops=False)
             for et in self.edge_types},
            aggr='mean',
        )

        self.conv2 = HeteroConv(
            {et: GATConv((-1, -1), hidden_channels, heads=num_heads,
                         add_self_loops=False)
             for et in self.edge_types},
            aggr='mean',
        )

        # Binary cross-entropy loss with optional class weighting
        self.loss_fn = nn.BCEWithLogitsLoss()

        # assumes pooling from all non-package node types - Package_Name is always the src node
        # self.classifier = nn.Linear(len(set(t for _, _, t in self.edge_types)) * hidden_channels * num_heads, 1)


    def miss_edge_index(self, edge_index_dict):
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, \
                                    device=next(self.parameters()).device)
        
        return edge_index_dict

    def cal_attn_weight(self, conv_module, x_dict, edge_index_dict):
        ''' custom version of HeteroGonv that returns attention weights
        
        returns:
            - updated x_dict
            - attn_weights: Dict[edge_type] = attention tensor
        '''
        attn_weights = {}
        out_dict = {}

        for edge_type, conv in conv_module.convs.items():
            src_type, _, tgt_type = edge_type
            x_src = x_dict[src_type]
            x_tgt = x_dict[tgt_type]
            edge_index = edge_index_dict[edge_type]

            out, (_, alpha) = conv((x_src, x_tgt), edge_index, return_attention_weights=True)
            attn_weights[edge_type] = alpha

            out_dict[tgt_type] = out_dict.get(tgt_type, 0) + out
        
        return out_dict, attn_weights


    def forward(self, batch, **kwargs):
        '''
        args:
            batch: HeteroData type with node_types -> x and edge_types -> edge_index and edge_attr
        '''
        x_dict, edge_index_dict, edge_attr_dict = batch_dict(batch)

        # sanitize edge indices
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, tgt_type = edge_type

            num_src = x_dict[src_type].size(0)
            num_tgt = x_dict[tgt_type].size(0)
            # fix unmatched size and index for node
            edge_index_dict[edge_type] = sani_edge_index(edge_index, num_src, num_tgt)

        # ---- check for missing node/edge types before conv1
        hidden_dim = next(x.shape[1] for x in x_dict.values() if x is not None and x.dim() == 2)
        # check potential miss node types
        x_dict = miss_check(x_dict, node_types, hidden_dim)
        edge_index_dict = self.miss_edge_index(edge_index_dict)

        # ---- first conv with attention
        x_dict_1, attn_weights_1 = self.cal_attn_weight(self.conv1, x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict_1.items()}

        # ---- check for missing node/edge types before conv2
        hidden_dim = next(x.shape[1] for x in x_dict.values() if x is not None and x.dim() == 2)
        # check potential miss node types
        x_dict = miss_check(x_dict, node_types, hidden_dim)
        edge_index_dict = self.miss_edge_index(edge_index_dict)
        
        # ---- second conv with attention
        x_dict, attn_weights_2 = self.cal_attn_weight(self.conv2,x_dict, edge_index_dict)

        # ---- pooling per node type, excluding "Package_Name"
        pooled_outputs = []
        for node_type, x in x_dict.items():
            if node_type == "Package_Name":
                continue
            if hasattr(batch[node_type], "batch"):
                pooled = global_mean_pool(x, batch[node_type].batch)
                pooled_outputs.append(pooled)

        # ---- final classification
        graph_embed = torch.cat(pooled_outputs, dim=-1)

        in_channels = graph_embed.shape[1]
        # dynamically infer classifier input size
        logits = nn.Linear(in_channels, 1).to(graph_embed.device)(graph_embed).squeeze(-1)

        return logits, {"conv1": attn_weights_1, "conv2": attn_weights_2}
    
    
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
        
