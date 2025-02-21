'''
 # @ Create Time: 2025-01-03 16:05:44
 # @ Modified time: 2025-01-03 16:06:18
 # @ Description: create attention-based Graph Neural Networks to learn feature importance
 '''

import torch
from torch_geometric.nn import HeteroConv, GATConv, GATv2Conv, DiffPool
from torch_geometric import nn
import torch.nn.functional as F

class MaskedHeteroGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_heads, num_clusters, num_edges, num_nodes):
        super(MaskedHeteroGAT, self).__init__()
    
        # learnable masks ---- initialized with 1
        self.edge_mask = torch.nn.Parameter(torch.ones(num_edges), requires_grad=True) 
        self.node_mask = torch.nn.Parameter(torch.ones(num_nodes), requires_grad=True)
    
        # GAT layers
        self.conv1 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'CMD', 'Command'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'IP'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Port'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Hostnames'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
            },
            aggr='mean',
        )

        # define second layer
        self.conv2 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'CMD', 'Command'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'IP'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Port'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Hostnames'): GATv2Conv((-1，-1), hidden_channels, heads=num_heads),
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


    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch, node_type):
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

        # perform GAT layer
        x_dict = self.conv1(masked_x_dict, masked_edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, masked_edge_index_dict)

        # apply diffpool to hierarchical node embeddings
        x_pooled, edge_pooled, batch_pooled = self.diffpool(
            x_dict[node_type], masked_edge_index_dict[node_type], batch)

        # aggregate edge features
        agg_edge = self.agg_edge_features(edge_att_dict)

        # pass through classification process for graph-level input
        probs = self.subgraph_cls(x_pooled, agg_edge, batch_pooled)

        return probs
        

    def subgraph_cls(self, x_pooled, edge_dict, batch_pooled:
        ''' process node feature at graph-level for binary classification

        :param x_pooled: pooled node features from DiffPool
        :param edge_dict: aggregated edge features
        :param batch_pooled: updated batch indices after DiffPool
        '''
        # Concatenate aggregated edge info with pooled node features
        x = torch.cat([x_pooled, edge_dict.view(-1, 1)], dim=-1)

        # Pass concatenated features through the classifier
        logits = self.classifier(x)

        # Apply sigmoid for binary classification
        probs = torch.sigmoid(logits)
        return probs  


    def agg_edge_features(self, edge_attr_dict, edge_pooled):
        ''' aggregate edge attribute globally

        '''
        agg_edges = [
            edge_attr.mean() for key, edge_attr in edge_attr_dict.items()
        ]

        return edge_pooled if edge_pooled is not None else torch.tensor(agg_edges)


    def ext_node_att(self, ):
        ''' extract the importance of nodes based on the learned mask
        
        '''
        # normalize node mask values
        node_att = self.node_mask.detach().cpu()
        norm_node_att = node_att / mode_att.sum()

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
        return att_values[rank_indicies]

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
    def __init__(self, metadata, hidden_channels, out_channels, num_heads):
        super(HeteroConv, self).__init__()

        # define first layer
        self.conv1 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'CMD', 'Command'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'IP'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Port'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Hostnames'): GATConv((-1，-1), hidden_channels, heads=num_heads),
            },
            aggr='mean',
        )

        # define second layer
        self.conv2 = HeteroConv(
            {
                ('Package_Name', 'Action', 'Path'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'DNS', 'Hostname'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'CMD', 'Command'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'IP'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Port'): GATConv((-1，-1), hidden_channels, heads=num_heads),
                ('Package_Name', 'Socket', 'Hostnames'): GATConv((-1，-1), hidden_channels, heads=num_heads),
            },
            aggr='mean',
        )

        # add final projection layer to output logits for binary classification
        self.classifier = nn.Linear(out_channels, 1)
        # Binary cross-entropy loss with optional class weighting
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight]) if class_weight else None)


    def forward(self, x_dict, edge_index_dict):
        '''
        :param x_dict: a dict holding node feature informaiton for each individual node type
        :param edge_index_dict: a dict holding graph connectivity info for each individual edge type
        '''
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        # project to logits for classification
        logit_dict = {
            key: self.classifier(x).squeeze(-1) for key, x in x_dict.items()
        }

        return logit_dict
    
    def compute_loss(self, logic_dict, y_dict):
        ''' computer the binary classification loss

        :param logic_dict: dictionary of logits for each node type
        :param x_dictL the dictionary of ground truth labels for each node type
        
        return: total loss across all node types
        '''
        loss = 0
        for node_type in logic_dict:
            if node_type in y_dict:
                y_pred = logic_dict[node_type]
                y_true = y_dict[node_type].float()
                loss += self.loss_fn(y_pred, y_true)

        return loss

    def train_model(self, x_dict, edge_index_dict, y_dict, optimizer, num_epochs=100):
        ''' train model and print performance
        Args:
            x_dict: Node features.
            edge_index_dict: Edge indices for each edge type.
            y_dict: Ground truth labels.
            optimizer: Optimizer for the model.
            num_epochs: Number of training epochs.

        '''
        # loop in epochs
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            # forward pass
            logic_dict = self.forward(x_dict, edge_index_dict)

            # compute loss
            loss = self.compute_loss(logic_dict, y_dict)

            # backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")