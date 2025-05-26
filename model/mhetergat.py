import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv
from optim import diffpool
import torch.nn.functional as F
import os
from utils import evals
from utils.pregraph import *
from collections import defaultdict

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

        self.global_node_id_map = load_global_node_id_map(node_id_map_file)

        # for ranking purpose
        self.global_edge_atten_map = defaultdict(float)
        self.global_node_rank_map = defaultdict(float)
        self.global_node_eco_system_map = defaultdict(lambda: defaultdict(float))

        # Classifier for binary classification (output of size 1), consider extra input for edge info
        self.classifier = torch.nn.Linear(out_channels * num_heads, 1)


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

            # reverse lookup of global node indices to original node values
            for i in range(local_edge_index.shape[1]):
                src_node_idx = edge_index[0, i].item()
                tgt_node_idx = edge_index[1, i].item()

                # retrieve the original node values using reverse lookup
                src_node_value = get_ori_node_value(src_node_idx, self.global_node_id_map)
                tgt_node_value = get_ori_node_value(tgt_node_idx, self.global_node_id_map)

                # Store the edge, original node values, and the attention score in the map
                edge_atten_map[(src_node_value, tgt_node_value)] = alpha[i, 0].item()

            attn_weights[edge_type] = alpha
            out_dict[tgt_type] = out_dict.get(tgt_type, 0) + out
        
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
        x_dict_1, attn_weights_1, edge_atten_map_1 = self.cal_attn_weight(self.conv1, x_dict, edge_index_dict)
        
        x_dict = {key: F.relu(x) for key, x in x_dict_1.items()}

        # ---- diffpool per node type, excluding "Package_Name"
        s_dict = self.hetero_gnn(x_dict, edge_index_dict)

        pooled_x_dict, pooled_adj_dict, link_loss, ent_loss = diffpool.hetero_diff_pool(
                    x_dict, edge_index_dict, s_dict
                )
        
        # last attention weight calculation after pooling
        x_dict_pooled, attn_weights_pooled, edge_atten_map_2 = self.cal_attn_weight(self.conv2, pooled_x_dict, pooled_adj_dict)

        # --- continue with binary classification task at subgraph-level
        ## do not choose x_dict_pooled because of it is not the final embeddings
        final_embed = pooled_x_dict
        # # Aggregate edge features
        agg_edge = self.agg_edge_features(edge_attr_dict)

        # combine pooled node features and aggregated edge features for classification
        all_pooled_nodes = torch.cat(list(final_embed.values()), dim=0)

        if agg_edge is not None:
            combined_features = torch.cat([all_pooled_nodes, agg_edge.view(-1,1)], dim=-1)
        else:
            combined_features = all_pooled_nodes
        
        logits = self.classifier(combined_features)
        probs = torch.sigmoid(logits)

        return probs, link_loss + ent_loss, attn_weights_pooled
    
    
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


    def ext_node_att(self, ):
        ''' extract the importance of nodes based on the learned mask
        
        '''
        # normalize node mask values
        node_att = self.node_mask.detach().cpu()
        norm_node_att = node_att / node_att.sum()

        return norm_node_att
    
    def rank_att(self, att_values):
        ''' rank importance of nodes / edges 

        '''
        ranked_indicies = torch.argsort(att_values, descending=True)
        return att_values[ranked_indicies]

    def compute_loss(self, probs, labels):
        ''' compute the total loss, including BCE loss and regularization

        :param probs: model output probabilities (after sigmoid activation)
        :param labels: ground truth binary labels
        :return: total loss (BCE loss + regularization)

        '''
        bce_loss = F.binary_cross_entropy(probs, labels.float())
        reg_loss = mask_regularization(self.edge_mask, self.node_mask)

        return bce_loss + reg_loss

# regularization term to encourage sparsity
def mask_regularization(edge_mask, node_mask, lambda_reg=0.01):
    return lambda_reg * (torch.sum(torch.abs(edge_mask)) + torch.sum(torch.abs(node_mask)))

