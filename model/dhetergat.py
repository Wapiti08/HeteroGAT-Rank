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
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

accelerator = Accelerator(device_placement=True)

# important to track unused parameters to avoid errors
if torch.cuda.device_count() > 1:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
else:
    accelerator = Accelerator()

device = accelerator.device

# predefined node types
node_types = ["Path", "DNS Host", "Package_Name", "Hostnames", "IP", "Command", "Port"]

class GraphPoolingModule(torch.nn.Module):
    def __init__(self, node_pool, edge_pool):
        super(GraphPoolingModule, self).__init__()
        self.node_pool = node_pool
        self.edge_pool = edge_pool

    def forward(self, x_dict, batch_dict, edge_attr_dict, edge_batch_dict):
        """
        concat operations on node_pool and edge_pool, output concatenated graph embedding。
        """
        # exclude Package_Name
        x_dict_target = {k: v for k, v in x_dict.items() if k != "Package_Name"}
        batch_dict_target = {k: v for k, v in batch_dict.items() if k != "Package_Name"}

        node_embed = self.node_pool(x_dict_target, batch_dict_target)
        edge_embed = self.edge_pool(edge_attr_dict, edge_batch_dict)

        graph_embed = torch.cat([node_embed, edge_embed], dim=-1)
        return graph_embed

class DiffHeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, edge_attr_dim, num_heads, processed_dir, enable_debug=False):
        '''
        args:
            in_channels: the dimensionality of the input node features
            hidden_channels: the dimensionality of the hidden node embeddings
            out_channels: the output dimension of the GAT layer before passing to the cls head
            num_heads: the number of attention heads in the GATv2Conv layers
            num_edges: the number of edges in a batch
            num_nodes: the number of nodes in a batch
        '''
        super(DiffHeteroGAT, self).__init__()

        self.edge_types = [
            ('Package_Name', 'Action', 'Path'),
            ('Package_Name', 'DNS', 'DNS Host'),
            ('Package_Name', 'CMD', 'Command'),
            ('Package_Name', 'socket_ip', 'IP'),
            ('Package_Name', 'socket_port', 'Port'),
            ('Package_Name', 'socket_host', 'Hostnames'),
        ]

        self.conv1 = self._build_hetero_gat(num_heads)
        self.conv2 = self._build_hetero_gat(num_heads)

        # layernorm for each node type after conv1 and conv2
        self.ln1 = torch.nn.ModuleDict({nt: LayerNorm(hidden_channels * num_heads) for nt in node_types})
        self.ln2 = torch.nn.ModuleDict({nt: LayerNorm(hidden_channels * num_heads) for nt in node_types})

        self.pkgname_projector = torch.nn.Linear(400, 256)
        # define attention pool for node and edge

        self.graph_pool = GraphPoolingModule(
            node_pool=attenpool.MultiTypeAttentionPooling(in_dim=256),
            edge_pool=attenpool.MultiTypeEdgePooling(edge_attr_dim=edge_attr_dim)
        )

        self.loss_fn = CompositeLoss(lambda_contrastive=0.0, lambda_sparsity=0.001, lambda_entropy=0.001)
        # Classifier for binary classification (output of size 1), consider extra input for edge info
        self.classifier = LazyLinear(1)

        self.enable_debug = enable_debug
        self.latest_attn_weights = {}
        self.latest_edge_index_map = {}
        self.reverse_node_id_map = {}
        self.reverse_node_id_vec = []

    def activate_debug(self, processed_dir: Path):
        """
        Lazy-load debug information like reverse node id mapping.
        Can be called during training to enable explanation-related features.
        """
        self.enable_debug = True
        node_id_map_file = Path(processed_dir).parent.joinpath('process_state.pkl')
        global_node_id_map = load_global_node_id_map(node_id_map_file)

        self.reverse_node_id_map = {v: k for k, v in global_node_id_map.items()}
        required_size = max(self.reverse_node_id_map.keys(), default=-1) + 1

        if len(self.reverse_node_id_vec) < required_size:
            reverse_list = [""] * required_size
            for idx, name in self.reverse_node_id_map.items():
                reverse_list[idx] = name
            self.reverse_node_id_vec = reverse_list


    def _build_hetero_gat(self, num_heads):
        return HeteroConv({
            et: GATv2Conv(
                in_channels=(-1, -1),
                out_channels=64,
                heads=num_heads,
                concat=True,
                negative_slope=0.2,
                add_self_loops=False
            ) for et in self.edge_types
        }, aggr='mean')

    def to_local_edge_indices(self, x_dict, edge_index_dict, batch=None):
        ''' convert global node indices to local node indices for each edge type
        cache pre-batch to avoid redundant computation
        
        '''
        # if batch has remapped results, directly return
        if batch is not None:
            local_edge_index_dict = batch.__dict__.get('_local_edge_index_dict', None)
            if local_edge_index_dict is not None:
                return local_edge_index_dict

        g2l_map = global_to_local_map(x_dict, edge_index_dict)
        local_edge_index_dict = {}

        for etype, edge_index in edge_index_dict.items():
            local_edge_index_dict[etype] = remap_edge_indices_vect(
                edge_index, g2l_map, etype[0], etype[2]
            )

        # cache remapped to batch
        if batch is not None:
            batch.__dict__['_local_edge_index_dict'] = local_edge_index_dict
        return local_edge_index_dict

    def cal_attn_weight(self, conv_module, x_dict, local_edge_index_dict):
        ''' Custom version of HeteroGonv that returns attention weights
        
        returns:
            - updated x_dict
            - attn_weights: Dict[edge_type] = attention tensor
        '''
        attn_weights = {}
        out_dict = {}

        # Only compute explanation info when enabled
        edge_atten_map = {} if self.enable_debug else None
        edge_index_map = {} if self.enable_debug else None

        for edge_type, conv in conv_module.convs.items():
            src_type, _, tgt_type = edge_type

            if src_type not in x_dict or tgt_type not in x_dict:
                if self.enable_debug:
                    print(f"Warning: {src_type if src_type not in x_dict else tgt_type} not in x_dict.")
                continue

            x_src, x_tgt = x_dict[src_type], x_dict[tgt_type]
            edge_index = local_edge_index_dict[edge_type]
            edge_index = edge_index.coalesce().indices() if edge_index.is_sparse else edge_index

            out, (_, alpha) = conv((x_src, x_tgt), edge_index, return_attention_weights=True)

            # keep the gradient of alpha
            if alpha.requires_grad:
                alpha.retain_grad()

            # Detach alpha from autograd + move to CPU
            if self.enable_debug and self.reverse_node_id_vec:
                with torch.no_grad():
                    alpha_cpu = alpha.detach().cpu()
                    edge_list = edge_index.T.tolist()
                    edges = [(self.reverse_node_id_vec[s], self.reverse_node_id_vec[t]) for s, t in edge_list]
                    scores = alpha_cpu[:, 0].tolist()
                    edge_atten_map.update(dict(zip(edges, scores)))
                    edge_index_map.setdefault(edge_type, []).extend(edges)

            attn_weights[edge_type] = alpha

            if tgt_type in out_dict:
                out_dict[tgt_type] = torch.add(out_dict[tgt_type], out)
            else:
                out_dict[tgt_type] = out
            
            if self.enable_debug:
                self.latest_attn_weights = attn_weights
                self.latest_edge_index_map = edge_index_map
        
        # After the first pass, merge the updated features with the original ones for all node types
        for node_type, x in x_dict.items():
            out_dict.setdefault(node_type, x)

        return out_dict, attn_weights, edge_atten_map, edge_index_map

    def _process_x_dict(self, x_dict_input, norm_layer):
        out_dict = {}
        for node_type, x in x_dict_input.items():
            if node_type == "Package_Name" and x.shape[-1] == 400:
                x = self.pkgname_projector(x)
            if node_type in norm_layer:
                x = norm_layer[node_type](x)
                F.relu_(x)  # inplace relu
            out_dict[node_type] = x
        return out_dict

    def forward(self, batch, **kwargs):
        '''
        args:
            batch: HeteroData type with node_types -> x and edge_types -> edge_index and edge_attr

        '''
        x_dict, edge_index_dict, batch_dict, edge_attr_dict, edge_batch_dict = parse_batch_dict(batch)

        # --- First conv ---
        local_edge_index_dict = self.to_local_edge_indices(x_dict, edge_index_dict, batch=batch)
        x_dict_1, attn_weights_1, edge_atten_map_1, edge_index_map_1 =\
            self.cal_attn_weight(self.conv1, x_dict, local_edge_index_dict)
        # apply layernorm, exclude Package_Name
        x_dict = self._process_x_dict(x_dict_1, self.ln1)

        # --- Second conv ---
        x_dict_2, attn_weights_2, edge_atten_map_2, edge_index_map_2 = \
            self.cal_attn_weight(self.conv2, x_dict, local_edge_index_dict)
        x_dict = self._process_x_dict(x_dict_2, self.ln2)

        # ---- Final node pooling (excluding Package_Name)
        graph_embed = self.graph_pool(
            x_dict=x_dict,
            batch_dict=batch_dict,
            edge_attr_dict=edge_attr_dict,
            edge_batch_dict=edge_batch_dict
        )

        logits = self.classifier(graph_embed).squeeze(-1)

        # align with the shape
        label = batch.label if batch.label.dim() > 0 else batch.label.unsqueeze(0)
        logits = logits if logits.dim() > 0 else logits.unsqueeze(0)

        if logits.shape != label.shape:
            print(f"Shape mismatch! logits: {logits.shape}, label: {label.shape}")
            print("Logits:", logits)
            print("Label :", label)

        # compute composite loss
        loss = self.loss_fn(
            cls_loss=F.binary_cross_entropy_with_logits(logits, label.float()),
            attn_weights=attn_weights_2
        )

        return logits, loss, attn_weights_2, edge_atten_map_2, edge_index_map_2
    
    
    def evaluate(self, logits, batch, threshold=0.5):
        metrics = evals.evaluate(logits, batch, threshold)
        print("evaluation result: \n", metrics )
        return metrics

    def plot_metrics(self, y_true, y_prob, metrics):
        ''' plot roc and calculated metrics
        
        '''
        # define the default save name
        roc_save_path = "roc_curve_dheterogat.png"
        metric_save_path = "metrics_bar_dheterogat.png"

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
