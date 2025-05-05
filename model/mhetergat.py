import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv,GATConv,global_mean_pool
from model import diffpool
import torch.nn.functional as F
import os
from utils import evals
from utils.pregraph import *

# predefined node types
node_types = ["Path", "DNS Host", "Package_Name", "IP", "Hostnames", "Command", "Port"]
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class MaskedHeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_clusters, num_edges, num_nodes):
        '''
        args:
            hidden_channels: the dimensionality of the hidden node embeddings
            out_channels: the output dimension of the GAT layer before passing to the cls head
            num_heads: the number of attention heads in the GATv2Conv layers
            num_clusters: the number of clusters used in DiffPool layer for hierarchical feature aggregation
            num_edges: the number of edges in a batch
            num_nodes: the number of nodes in a batch
        '''
        super(MaskedHeteroGAT, self).__init__()

        self.edge_types = [
            ('Package_Name', 'Action', 'Path'),
            ('Package_Name', 'DNS', 'DNS Host'),
            ('Package_Name', 'CMD', 'Command'),
            ('Package_Name', 'Socket', 'IP'),
            ('Package_Name', 'Socket', 'Port'),
            ('Package_Name', 'Socket', 'Hostnames'),
        ]

        # learnable masks ---- initialized with 1
        self.edge_mask = torch.nn.Parameter(torch.ones(num_edges), requires_grad=True) 
        self.node_mask = torch.nn.Parameter(torch.ones(num_nodes), requires_grad=True)
    
        # GAT layers
        self.conv1 = HeteroConv(
            {et: GATv2Conv((-1, -1), hidden_channels, heads=num_heads,
                         add_self_loops=False)
             for et in self.edge_types},
              # aggregation method for multi-head attention
            aggr='mean',
        )

        # define second layer
        self.conv2 = HeteroConv(
            {et: GATv2Conv((-1, -1), hidden_channels, heads=num_heads,
                         add_self_loops=False)
             for et in self.edge_types},
            aggr='mean',
        )

        # diffpool layer for hierarchical feature aggregation
        self.dplayer = diffpool.DiffPool(
                num_features = out_channels * num_heads,
                num_classes = num_clusters,
            )

        # Classifier for binary classification (output of size 1), consider extra input for edge info
        self.classifier = torch.nn.Linear(out_channels * num_heads, 1)
        
        # # Separate processing for 'package_name' if necessary
        # self.package_name_classifier = torch.nn.Linear(1, 1)

    def mask_miss_edge(self, edge_index_dict):
        ''' masking function to handle cases where the number of edges is zero
        or the shape of self.edge_mask doesn't match the number of edges
        
        '''
        masked_edge_index_dict = {}
        for key in self.conv2.convs.keys():
            if key in edge_index_dict:
                edge_index = edge_index_dict[key]
                num_edges = edge_index.size(1)
                if num_edges > 0:
                    # ensure the mask size matches the number of edges in this batch
                    mask = torch.bernoulli(self.edge_mask[:num_edges]).bool()

                    if mask.sum() > 0:
                        masked_edge_index_dict[key] = edge_index[:, mask]
                    else:
                        # fallback: keep original edge_index to avoid None
                        masked_edge_index_dict[key] = edge_index
                else:
                    masked_edge_index_dict[key] = edge_index
            else:
                # Handle the case where the edge type is missing in edge_index_dict
                print(f"Warning: Missing edge type '{key}' in edge_index_dict.")
                # You can choose to either skip it or handle it in another way
                masked_edge_index_dict[key] = None 

        return masked_edge_index_dict

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
            print(type(edge_index))
            print(edge_index)

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
        # create the mapping from global indices to local indices
        global_to_local_mapping = global_to_local_map(x_dict, edge_index_dict)

        # Process each edge type and sanitize edge indices
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, tgt_type = edge_type
            num_src_nodes = x_dict[src_type].size(0)  # Local node count for source type
            num_tgt_nodes = x_dict[tgt_type].size(0)  # Local node count for target type

            # Debug: Print the number of nodes for this edge type
            print(f"Sanitizing edge type {edge_type}: num_src_nodes = {num_src_nodes}, num_tgt_nodes = {num_tgt_nodes}")

            # Fix unmatched size and index for the current edge type
            edge_index_dict[edge_type] = sani_edge_index(edge_index, num_src_nodes, num_tgt_nodes, global_to_local_mapping)
    
        # apply edge masks (stochastic masking)
        masked_edge_index_dict = self.mask_miss_edge(edge_index_dict)

        # ---- check for missing node/edge types
        hidden_dim = next(x.shape[1] for x in x_dict.values() if x is not None and x.dim() == 2)
        x_dict = miss_check(x_dict, node_types, hidden_dim)

        # apply node masks
        masked_x_dict = {
            # element-wise multiplication with node features
            key: x * self.node_mask[:x.size(0)].reshape(-1,1)
            if self.node_mask.size(0) >= x.size(0) else \
                self.node_mask[:x.size(0)].expand(x.size(0), -1).reshape(-1,1)
            for key, x in x_dict.items()
        }

        node_type_outputs = []

        # process other node types (excluding "Package_Name")
        for node_type in node_types:
            if node_type != "Package_Name":
                # perform GAT layer
                x_dict_1, attn_weights_1 = self.cal_attn_weight(self.conv1, masked_x_dict, masked_edge_index_dict)
                x_dict = {key: F.relu(x) for key, x in x_dict_1.items()}
                out, (edge_index, alpha) = self.cal_attn_weight(self.conv2, x_dict, masked_edge_index_dict)

                # apply diffpool to hierarchical node embeddings
                x_pooled, edge_pooled, batch_pooled = self.dplayer(
                    x_dict[node_type], masked_edge_index_dict[node_type], batch)

                node_type_outputs.append(x_pooled)

        # use the last pooled node features for classificaton
        final_embed = node_type_outputs[-1]

        # aggregate edge features 
        agg_edge = self.agg_edge_features(edge_attr_dict)

        # pass through classification process for graph-level input
        probs = self.subgraph_cls(final_embed, agg_edge)

        return probs
    
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
        

    def subgraph_cls(self, final_embed, agg_edge):
        ''' process node feature at graph-level for binary classification

        :param x_pooled: pooled node features from DiffPool
        :param edge_dict: aggregated edge features
        '''
        # Concatenate aggregated edge info with pooled node features
        # x = torch.cat([x_pooled, edge_dict.view(-1, 1)], dim=-1)

        # Concatenate pooled node features with aggregated edge features
        x = torch.cat([final_embed, agg_edge.view(-1, 1)], dim=-1)  # Ensure edge features are 2D

        # Pass the concatenated features through the classifier
        logits = self.classifier(x)

        # Apply sigmoid for binary classification
        probs = torch.sigmoid(logits)
        return probs


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


    def ext_edge_att(self,):
        ''' extract the importance of edges based on learned mask

        '''
        # normalize edge mask values
        edge_att = self.edge_mask.detach().cpu()
        norm_edge_att = edge_att / edge_att.sum()

        return norm_edge_att

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

