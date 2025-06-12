'''
 # @ Create Time: 2025-05-11 14:58:00
 # @ Modified time: 2025-05-11 14:58:09
 # @ Description: implement version with constractive loss
 '''
import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, global_mean_pool
from model.comploss import CompositeLoss
from optim import attenpool
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


def debug_shapes(x_dict, batch_dict, edge_attr_dict, edge_batch_dict, label):
    print("==== Node Feature Shapes ====")
    for k, v in x_dict.items():
        print(f"{k:>15}: x = {v.shape}, batch = {batch_dict.get(k, 'N/A').shape}")

    print("==== Edge Feature Shapes ====")
    for k, v in edge_attr_dict.items():
        batch_e = edge_batch_dict.get(k, None)
        print(f"{str(k):>40}: edge_attr = {v.shape}, batch = {batch_e.shape if batch_e is not None else 'N/A'}")

    print(f"==== Label Shape: {label.shape}")

class PNHeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, edge_attr_dim, num_heads, processed_dir):
        '''
        args:
            in_channels: the dimensionality of the input node features
            hidden_channels: the dimensionality of the hidden node embeddings
            out_channels: the output dimension of the GAT layer before passing to the cls head
            num_heads: the number of attention heads in the GATv2Conv layers
            num_edges: the number of edges in a batch
            num_nodes: the number of nodes in a batch
        '''
        super(PNHeteroGAT, self).__init__()

        self.edge_types = [
            ('Package_Name', 'Action', 'Path'),
            ('Package_Name', 'DNS', 'DNS Host'),
            ('Package_Name', 'CMD', 'Command'),
            ('Package_Name', 'socket_ip', 'IP'),
            ('Package_Name', 'socket_port', 'Port'),
            ('Package_Name', 'socket_host', 'Hostnames'),
        ]

        # GAT layers
        self.conv1 = HeteroConv(
            {et: 
            #  GATv2Conv((-1, -1), hidden_channels, heads=num_heads,
            #              add_self_loops=False)
             GATv2Conv(
                in_channels=(-1, -1),  
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
                in_channels=(-1, -1),
                out_channels=64,  
                heads=num_heads,  
                concat=True,  
                negative_slope=0.2,
                add_self_loops=False
            )
             for et in self.edge_types},
            aggr='mean',
        )

        # layernorm for each node type after conv1 and conv2
        self.ln1 = torch.nn.ModuleDict()
        self.ln2 = torch.nn.ModuleDict()

        for node_type in node_types:
            self.ln1[node_type] = LayerNorm(hidden_channels * num_heads)
            self.ln2[node_type] = LayerNorm(hidden_channels * num_heads)

        self.pkgname_projector = torch.nn.Linear(400, 256)
        # define attention pool for node and edge
        self.node_pool = attenpool.MultiTypeAttentionPooling(in_dim=256)
        self.edge_pool = attenpool.MultiTypeEdgePooling(edge_attr_dim=edge_attr_dim)

        node_id_map_file = Path(processed_dir).parent.joinpath('process_state.pkl')
        
        global_node_id_map = load_global_node_id_map(node_id_map_file)
        self.reverse_node_id_map = {v: k for k, v in global_node_id_map.items()}

        # for ranking purpose
        self.global_edge_atten_map = defaultdict(float)
        self.global_node_rank_map = defaultdict(float)
        self.global_node_eco_system_map = defaultdict(lambda: defaultdict(float))

        self.loss_fn = CompositeLoss(lambda_contrastive=0.0, lambda_sparsity=0.01, lambda_entropy=0.01)

        # Classifier for binary classification (output of size 1), consider extra input for edge info
        self.classifier = LazyLinear(1)


    @staticmethod
    def to_local_edge_indices(x_dict, edge_index_dict):
        g2l_map = global_to_local_map(x_dict, edge_index_dict)
        local_edge_index_dict = {
            etype: remap_edge_indices(edge_index, g2l_map, etype[0], etype[2])
            for etype, edge_index in edge_index_dict.items()
        }

        return local_edge_index_dict

    def cal_attn_weight(self, conv_module, x_dict, local_edge_index_dict):
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
            edge_index = local_edge_index_dict[edge_type]

            if edge_index.is_sparse:
                edge_index = edge_index.coalesce().indices()

            # shape of alpha is [num_edges, num_heads]
            out, (_, alpha) = conv((x_src, x_tgt), edge_index, return_attention_weights=True)

            assert edge_index.shape[1] == alpha.shape[0], \
            f"Edge count mismatch: edge_index has {edge_index.shape[1]}, but alpha has {alpha.shape[0]}"

            # collect edges in terms of actual node values
            edge_list = []

            # reverse lookup of global node indices to original node values
            for i in range(edge_index.shape[1]):
                src_node_idx = edge_index[0, i].item()
                tgt_node_idx = edge_index[1, i].item()

                # retrieve the original node values using reverse lookup
                src_node_value = get_ori_node_value(src_node_idx, self.reverse_node_id_map)
                tgt_node_value = get_ori_node_value(tgt_node_idx, self.reverse_node_id_map)

                # Store the edge, original node values, and the attention score in the map
                edge_atten_map[(src_node_value, tgt_node_value)] = alpha[i, 0].item()
                # collect edge in node-value form
                edge_index_map.setdefault(edge_type, []).append((src_node_value, tgt_node_value))

            attn_weights[edge_type] = alpha
            out_dict[tgt_type] = out_dict.get(tgt_type, 0) + out
        
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
        assert batch.label.size(0) > 1, "Only one graph in batch — check your DataLoader or dataset batching!"
        print(f"[DEBUG] batch['Path'].batch.unique() = {batch['Path'].batch.unique()}")

        x_dict, edge_index_dict, batch_dict, edge_attr_dict, edge_batch_dict = parse_batch_dict(batch)

        for k, v in batch_dict.items():
            print(f"[DEBUG] batch_dict[{k}]: shape = {v.shape}, unique = {torch.unique(v)}")

        for k, v in edge_batch_dict.items():
            print(f"[DEBUG] edge_batch_dict[{k}]: shape = {v.shape}, unique = {torch.unique(v)}")

        # convert global indices to local indices
        local_edge_index_dict_1 = self.to_local_edge_indices(x_dict, edge_index_dict)

        # GAT layer 1
        x_dict_1, attn_weights_1, edge_atten_map_1, edge_index_map_1 =\
            self.cal_attn_weight(self.conv1, x_dict, local_edge_index_dict_1)

        # apply layernorm, exclude Package_Name
        x_dict = {}
        for node_type, x in x_dict_1.items():
            if node_type == "Package_Name":
                if x.shape[-1] == 400:
                    x = self.pkgname_projector(x)
                x_dict[node_type] = x

            if node_type in self.ln1 and x.shape[-1] == 256:
                x_dict[node_type] = F.relu(self.ln1[node_type](x))
            else:
                x_dict[node_type] = x

        # GAT layer 2
        # Second conv still on original graph (no pooling)
        local_edge_index_dict_2 = self.to_local_edge_indices(x_dict, edge_index_dict)

        x_dict_2, attn_weights_2, edge_atten_map_2, edge_index_map_2 = \
            self.cal_attn_weight(self.conv2, x_dict, local_edge_index_dict_2)

        x_dict = {}
        for node_type, x in x_dict_2.items():
            if node_type == "Package_Name":
                if x.shape[-1] == 400:
                    x = self.pkgname_projector(x)
                x_dict[node_type] = x
            elif node_type in self.ln2 and x.shape[-1] == 256:
                x_dict[node_type] = F.relu(self.ln2[node_type](x))
            else:
                x_dict[node_type] = x


        # ---- Final node pooling (excluding Package_Name)
        x_dict_target = {ntype: x for ntype, x in x_dict.items() if ntype != "Package_Name"}
        batch_dict_target = {k: v for k, v in batch_dict.items() if k != 'Package_Name'}
        
        node_pool = self.node_pool(x_dict_target, batch_dict_target)

        # ---- Final edge pooling using attention weights
        # edge_avg_attr = {k: v.mean(dim=1) for k, v in attn_weights_1.items()}
        edge_pool = self.edge_pool(edge_attr_dict, edge_batch_dict)
        
        node_pool = node_pool.to(device)
        edge_pool = edge_pool.to(device)
        
        print(f"[Pooling] node_pool: {node_pool.shape}, edge_pool: {edge_pool.shape}, expected batch size: {batch.label.size(0)}")
        # last attention weight calculation after pooling
        graph_embed = torch.cat([node_pool, edge_pool], dim=-1)  # shape [2F]

        logits = self.classifier(graph_embed).squeeze(-1)
        print(logits)
        print(f"logits shape: {logits.shape}")
        # Make sure logits and label have the same shape
        label = batch.label.view(-1)  # shape: [B]
        print(f"logits shape: {logits.shape}, label shape: {label.shape}")

        # align with the shape
        # if logits.dim() == 0:
        #     logits = logits.unsqueeze(0)
        # if batch.label.dim() == 0:
        #     label = batch.label.unsqueeze(0)
        # else:
        #     label = batch.label
        
        # compute composite loss --- loss is a dict type
        loss_dict = self.loss_fn(
            cls_loss=F.binary_cross_entropy_with_logits(logits, label.float()),
            attn_weights=attn_weights_2,
            graph_embeds=graph_embed,
            labels = batch.label.float(),
            return_details=True  
        )

        return logits, loss_dict['total'], attn_weights_2, edge_atten_map_2, edge_index_map_2
    
    
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
