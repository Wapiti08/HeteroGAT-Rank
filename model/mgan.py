'''
 # @ Create Time: 2025-01-03 16:05:44
 # @ Modified time: 2025-01-03 16:06:18
 # @ Description: create attention-based Graph Neural Networks to learn feature importance
 '''

import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from torch_geometric.nn import HeteroConv, GATConv, GATv2Conv
from model import DiffPool
from torch import nn
import torch.nn.functional as F
from ext.data_create import LabeledSubGraphs
from ext.iter_loader import IterSubGraphs
from torch_geometric.loader import DataLoader
from datetime import datetime

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
    
        # learnable masks ---- initialized with 1
        self.edge_mask = torch.nn.Parameter(torch.ones(num_edges), requires_grad=True) 
        self.node_mask = torch.nn.Parameter(torch.ones(num_nodes), requires_grad=True)
    
        # GAT layers
        self.conv1 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATv2Conv((-1, -1), hidden_channels, heads=num_heads,add_self_loops=False),
                ('Package_Name', 'DNS', 'Hostname'): GATv2Conv((-1, -1), hidden_channels, heads=num_heads,add_self_loops=False),
                ('Package_Name', 'CMD', 'Command'): GATv2Conv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'IP'): GATv2Conv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'Port'): GATv2Conv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'Hostnames'): GATv2Conv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            },
            aggr='mean',  # aggregation method for multi-head attention
        )

        # define second layer
        self.conv2 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATv2Conv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'DNS', 'Hostname'): GATv2Conv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'CMD', 'Command'): GATv2Conv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'IP'): GATv2Conv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'Port'): GATv2Conv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'Hostnames'): GATv2Conv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
            },
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch, node_types, **kwargs):
        '''
        :param x_dict: a dict holding node feature informaiton for each individual node type
        :param edge_index_dict: a dict holding graph connectivity info for each individual edge type

        '''
        # apply edge masks
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
                x_dict = self.conv1(masked_x_dict, masked_edge_index_dict)
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = self.conv2(x_dict, masked_edge_index_dict)

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

        # define first layer
        self.conv1 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATConv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'CMD', 'Command'): GATConv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'IP'): GATConv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'Port'): GATConv((-1,-1), hidden_channels, heads=num_heads, add_self_loops=False),
                ('Package_Name', 'Socket', 'Hostnames'): GATConv((-1,-1), hidden_channels, heads=num_heads,add_self_loops=False),
            },
            aggr='mean',
        )

        # define second layer
        self.conv2 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATConv((-1,-1), hidden_channels, heads=num_heads,add_self_loops=False),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1,-1), hidden_channels, heads=num_heads,add_self_loops=False),
                ('Package_Name', 'CMD', 'Command'): GATConv((-1,-1), hidden_channels, heads=num_heads,add_self_loops=False),
                ('Package_Name', 'Socket', 'IP'): GATConv((-1,-1), hidden_channels, heads=num_heads,add_self_loops=False),
                ('Package_Name', 'Socket', 'Port'): GATConv((-1,-1), hidden_channels, heads=num_heads,add_self_loops=False),
                ('Package_Name', 'Socket', 'Hostnames'): GATConv((-1,-1), hidden_channels, heads=num_heads,add_self_loops=False),
            },
            aggr='mean',
        )

        # add final projection layer to output logits for binary classification
        self.classifier = nn.Linear(out_channels, 1)
        # Binary cross-entropy loss with optional class weighting
        self.loss_fn = nn.BCEWithLogitsLoss()


    def forward(self, batch, **kwargs):
        '''
        :param x_dict: a dict holding node feature informaiton for each individual node type
        :param edge_index_dict: a dict holding graph connectivity info for each individual edge type
        '''
        print(batch)
        print(batch.batch)
        x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        # project to logits for classification
        logit_dict = {
            key: self.classifier(x).squeeze(-1) for key, x in x_dict.items()
        }

        return logit_dict
    
    def compute_loss(self, logit_dict, batch):
        ''' computer the binary classification loss

        :param logit_dict: dictionary of logits for each node type
        :param x_dictL the dictionary of ground truth labels for each node type
        
        return: total loss across all node types
        '''
        loss = 0
        for node_type in logit_dict:
            if hasattr(batch[node_type], 'y'):
                y_pred = logit_dict[node_type]
                y_true = batch[node_type].y.float()
                loss += self.loss_fn(y_pred, y_true)

        return loss


if __name__ == "__main__":

    data_path = Path.cwd().parent.joinpath("ext", "test-small", "processed")
    print("Creating iterative dataset")
    dataset = IterSubGraphs(root=data_path, batch_size = 1)
    # load one .pt file at a time
    print("Creating subgraph dataloader")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=4,  # Reduce to test performance; 20 may be overkill for file I/O
        # persistent_workers=True,
        pin_memory=False,
        prefetch_factor=None
    )
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    batch=next(iter(dataloader))
    

    model2 = HeteroGAT(
        hidden_channels=64,
        out_channels=64,
        num_heads=4
    ).to(device)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 10

    print("Training HeteroGAT ...")
    # define the starting time
    start_time = datetime.now()
    for epoch in range(num_epochs):
        model2.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(next(model2.parameters()).device)  # Move batch to the same device as model
            optimizer2.zero_grad()

            # forward pass
            logic_dict = model2.forward(batch)
            # compute loss
            loss = model2.compute_loss(logic_dict, batch)

            # backward pass and optimization
            loss.backward()
            optimizer2.step()

            total_loss += loss.item()
        
        avg_loss = total_loss/len(dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print(f"Time spent for HeteroGAT is: {start_time - datetime.now()}")
    torch.save(model2, "heterogat_model.pth")

    print("Training MaskedHeteroGAT ...")
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

    # predefined node types
    node_types = ["Path", "Hostname", "Package_Name", "IP", "Hostnames", "Command", "Port"]

    # define the starting time
    start_time = datetime.now()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch=batch.to(device)

            # extract necessary input from the batch
            x_dict = batch.x_dict
            edge_index_dict = batch.edge_index_dict
            edge_attr_dict = batch.edge_attr_dict
            batch_indices = batch.batch_dict
            
            optimizer.zero_grad()
            probs = model1(x_dict, edge_index_dict, edge_attr_dict, batch_indices, node_types)
            
            labels = batch.y.to(device)
            loss = model1(probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"For MaskedHeteroGAT Time: Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
    print(f"Time spent for MaskedHeteroGAT is: {start_time - datetime.now()}")
    # save the model after training
    torch.save(model1, "masked_heterogat_model.pth")



    
