'''
 # @ Create Time: 2025-01-03 16:05:44
 # @ Modified time: 2025-01-03 16:06:18
 # @ Description: create attention-based Graph Neural Networks to learn feature importance
 '''

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


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# predefined node types
node_types = ["Path", "DNS Host", "Package_Name", "IP", "Hostnames", "Command", "Port"]


def batch_dict(batch: HeteroData):
    ''' function to extract x_dict and edge_index_dict from custom batch data
    
    '''
    # extract node features into a dict
    x_dict = {node_type: batch[node_type].x for node_type in batch.node_types if 'x' in batch[node_type]}
    
    # extract edge_index into a dict
    edge_index_dict = {
        edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types if 'edge_index' in batch[edge_type]
    }

    # extract edge_attr into a dict
    edge_attr_dict = {
        edge_type: batch[edge_type].edge_attr for edge_type in batch.edge_types if 'edge_attr' in batch[edge_type]

    }

    return x_dict, edge_index_dict, edge_attr_dict


def miss_check(x_dict, node_types, hidden_dim):
    for node_type in node_types:
        if node_type not in x_dict:
            # Dummy tensor to avoid breaking GATConv
            x_dict[node_type] = torch.zeros((1, hidden_dim), device=device)

    return x_dict


# check the index and size to avoid cuda error
def sani_edge_index(edge_index, num_src_nodes, num_tgt_nodes):
    ''' fix the problem when index of node is not matched with node size
    
    '''
    src = edge_index[0]
    tgt = edge_index[1]
    valid_mask = (src < num_src_nodes) & (tgt < num_tgt_nodes)
    return edge_index[:, valid_mask]


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
        self.diffpool = DiffPool(
                in_channels= out_channels * num_heads,
                cluster_size = num_clusters,
            )

        # Classifier for binary classification (output of size 1), consider extra input for edge info
        self.classifier = torch.nn.Linear(out_channels * num_heads, 1)
        
        # # Separate processing for 'package_name' if necessary
        # self.package_name_classifier = torch.nn.Linear(1, 1)

    def mask_miss_edge(self, edge_index_dict):
        masked_edge_index_dict = {}
        for key in self.conv2.convs.keys():
            if key in edge_index_dict:
                edge_index = edge_index_dict[key]
                num_edges = edge_index.size(1)
                if num_edges > 0:
                    mask = torch.bernoulli(self.edge_mask.expand(num_edges)).bool()
                    if mask.sum() > 0:
                        masked_edge_index_dict[key] = edge_index[:, mask]
                    else:
                        # fallback: keep original edge_index to avoid None
                        masked_edge_index_dict[key] = edge_index
                else:
                    masked_edge_index_dict[key] = edge_index
            else:
                raise ValueError(f"[-] edge_index missing for expected edge type: {key}")
        

    def forward(self, batch, **kwargs):
        '''
        :param x_dict: a dict holding node feature informaiton for each individual node type
        :param edge_index_dict: a dict holding graph connectivity info for each individual edge type

        '''
        x_dict, edge_index_dict, edge_attr_dict = batch_dict(batch)
        hidden_dim = next(x.shape[1] for x in x_dict.values() if x is not None and x.dim() == 2)
        x_dict = miss_check(x_dict, node_types, hidden_dim)

        # apply edge masks (stochastic masking)
        masked_edge_index_dict = {
            # stochastic masking
            key: edge_index[:, torch.bernoulli(self.edge_mask).bool()]
            for key, edge_index in edge_index_dict.items()
        }

        # apply node masks
        masked_x_dict = {
            # element-wise multiplication with node features
            key: x * self.node_mask[:x.size(0)].view(-1,1)
            for key, x in x_dict.items()
        }

        node_type_outputs = []

        # process other node types (excluding "Package_Name")
        for node_type in node_types:
            if node_type != "Package_Name":

                # perform GAT layer
                out, (edge_index, alpha) = self.conv1(masked_x_dict, masked_edge_index_dict, return_attention_weights=True)
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                out, (edge_index, alpha) = self.conv2(x_dict, masked_edge_index_dict, return_attention_weights=True)

                # apply diffpool to hierarchical node embeddings
                x_pooled, edge_pooled, batch_pooled = self.diffpool(
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
        


if __name__ == "__main__":

    data_path = Path.cwd().parent.joinpath("ext", "output", "processed")
    print("Creating iterative dataset")
    dataset = IterSubGraphs(root=data_path, batch_size = 10)
    # load one .pt file at a time
    print("Creating subgraph dataloader")
    num_epochs = 10

    # split into train/test
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=32)

    train_loader = DataLoader(
        train_data,
        batch_size=10,
        shuffle=True,
        pin_memory=False,
        prefetch_factor=None
    )

    test_loader = DataLoader(
        test_data,
        batch_size=10,
        shuffle=True,
        pin_memory=False,
        prefetch_factor=None
    )


    model2 = HeteroGAT(
        hidden_channels=64,
        out_channels=256,
        num_heads=4
    ).to(device)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)

    print("Training HeteroGAT ...")

    conv_weight_dict_2 = {}
    # define the starting time
    start_time = datetime.now()
    for epoch in range(num_epochs):
        model2.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(next(model2.parameters()).device)  # Move batch to the same device as model
            optimizer2.zero_grad()

            # forward pass
            logits, conv_weight_dict_2 = model2.forward(batch)
            # compute loss
            loss = model2.compute_loss(logits, batch)
            # backward pass and optimization
            loss.backward()
            optimizer2.step()

            total_loss += loss.item()
        
        avg_loss = total_loss/len(train_loader)

        # ----- EVALUATION -----
        model2.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(next(model2.parameters()).device)
                logits, _ = model2(batch)
                all_logits.append(logits)
                all_labels.append(batch['label'])

        # Concatenate
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        metrics = model2.evaluate(all_logits, all_labels)

        model2.plot_metrics(
            all_labels,
            torch.sigmoid(logits).cpu().numpy(),
            metrics)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    


    print(f"Time spent for HeteroGAT is: {start_time - datetime.now()}")

    torch.save(model2, "heterogat_model.pth")


    print("Training MaskedHeteroGAT ...")
    batch = next(iter(train_loader))
    # Initialize model with required parameters
    model1 = MaskedHeteroGAT(
        hidden_channels=64, 
        out_channels=64, 
        num_heads=4, 
        num_clusters=20, 
        num_edges = batch.num_edges, 
        num_nodes= batch.num_nodes   
    ).to(device)

    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-4)

    # define the starting time
    start_time = datetime.now()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch=batch.to(device)
            
            optimizer.zero_grad()
            probs = model1(batch)
            
            labels = batch.y.to(device)
            loss = model1(probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"For MaskedHeteroGAT Time: Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
    
        # --- Evaluation -----
        model1.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(next(model1.parameters()).device)
                logits, _ = model1(batch)
                all_logits.append(logits)
                all_labels.append(batch['label'])
        
        # Concatenate
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        metrics = model1.evaluate(all_logits, all_labels)

        # Compute metrics
        metrics = model1.evaluate(all_logits, all_labels)

        model1.plot_metrics(
            all_labels,
            torch.sigmoid(logits).cpu().numpy(),
            metrics)


        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


    print(f"Time spent for MaskedHeteroGAT is: {start_time - datetime.now()}")
    # save the model after training
    torch.save(model1, "masked_heterogat_model.pth")



    
