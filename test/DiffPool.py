import torch
from torch_geometric.nn import dense_diff_pool
from torch_geometric.data import HeteroData

# Assume you have a HeteroData object named 'data'
# 'data' should contain node features and adjacency matrices for each node type
# For example: data['movie'].x, data['movie'].edge_index, data['director'].x, ...

# Example: Accessing node features and adjacency matrices
#  for different node types. Replace 'movie', 'director' with your actual node types.

# Get node features for movie type
x_movie = data['movie'].x

# Get adjacency matrix for movie type
adj_movie = data['movie'].adj_matrix

# Get assignment matrix for movie type (assumes you have learned this)
s_movie = learned_assignment_matrix_movie  # Your learned assignment matrix

# Apply dense_diff_pool to movie type
x_pooled_movie, adj_pooled_movie, loss_lp_movie, loss_e_movie = \
    dense_diff_pool(x_movie, adj_movie, s_movie)

# Repeat for other node types (e.g., 'director')
x_director = data['director'].x
adj_director = data['director'].adj_matrix
s_director = learned_assignment_matrix_director

x_pooled_director, adj_pooled_director, loss_lp_director, loss_e_director = \
    dense_diff_pool(x_director, adj_director, s_director)

# Now you have pooled features and adjacency matrices for each node type
# You can aggregate these to create a representation of the entire heterogeneous graph
# Example aggregation:
# pooled_features = torch.cat([x_pooled_movie, x_pooled_director], dim=0)
# pooled_adj = torch.cat([adj_pooled_movie, adj_pooled_director], dim=0)

#  loss = loss_lp_movie + loss_lp_director + loss_e_movie + loss_e_director
#  # Or other aggregation/combination as per your task.