'''
 # @ Create Time: 2025-05-14 19:05:03
 # @ Modified time: 2025-05-14 19:05:27
 # @ Description: define functions to create sparse padding for heterogeneous graphs
 '''

import torch

def sparse_padding_with_values(data, max_num, target_feature_dim):
    ''' pad an existing sparse matrix (with actual values) to the desired size, maintaining sparsity.

    
    '''
    exist_size = data.size(0)

    # If data is a sparse tensor, we can get the indices and values directly
    if data.is_sparse:
        row_indices, col_indices = data._indices()  # Get row and column indices
        values = data._values()  # Get values of the sparse matrix
    else:
        # If data is dense, we need to manually create row and col indices
        row_indices, col_indices = torch.nonzero(data, as_tuple=True)  # Get indices of non-zero elements
        values = data[row_indices, col_indices]  # Get corresponding values for the non-zero indices

    # create sparse tensor for existing data
    sparse_data = torch.sparse.FloatTensor(
        torch.stack([row_indices, col_indices]),
        values,
        torch.Size([max_num, target_feature_dim])
    )

    # Step 2: If padding is required, create sparse padding (filled with zeros)
    if exist_size < max_num:
        padding_size = max_num - exist_size
        padding_indices = torch.zeros(2, padding_size, dtype=torch.long)  # Create padding indices
        padding_values = torch.zeros(padding_size, dtype=torch.float32)  # Padding values (zeros)
        
        # Create sparse tensor for padding
        sparse_padding = torch.sparse.FloatTensor(
            padding_indices, 
            padding_values, 
            torch.Size([max_num, target_feature_dim])
        )

        # Add the padding to the existing sparse data
        sparse_data = sparse_data + sparse_padding

    return sparse_data


def sparse_padding_with_values_edges(data, max_num):
    ''' pad an existing sparse matrix (with actual values) to the desired size, maintaining sparsity.

    
    '''
    # for edge_index, it is [2, num_edges]
    exist_size = data.size(1)

    # If data is a sparse tensor, we can get the indices and values directly
    if data.is_sparse:
        row_indices, col_indices = data._indices()  # Get row and column indices
        values = data._values()  # Get values of the sparse matrix
    else:
        # If data is dense, we need to manually create row and col indices
        row_indices, col_indices = torch.nonzero(data, as_tuple=True)  # Get indices of non-zero elements
        values = data[row_indices, col_indices]  # Get corresponding values for the non-zero indices

    # create sparse tensor for existing data
    sparse_data = torch.sparse.FloatTensor(
        torch.stack([row_indices, col_indices]),
        values,
        torch.Size([2, max_num])
    )

    # Step 2: If padding is required, create sparse padding (filled with zeros)
    if exist_size < max_num:
        padding_size = max_num - exist_size
        padding_indices = torch.zeros(2, padding_size, dtype=torch.long)  # Create padding indices
        padding_values = torch.zeros(padding_size, dtype=torch.float32)  # Padding values (zeros)
        
        # Create sparse tensor for padding
        sparse_padding = torch.sparse.FloatTensor(
            padding_indices, 
            padding_values, 
            torch.Size([2, max_num])
        )

        # Add the padding to the existing sparse data
        sparse_data = sparse_data + sparse_padding

    return sparse_data


def sparse_zeros(max_num, target_feature_dim):
    ''' create a sparse tensor of zeros with the desired size
    '''
    # Create a sparse zero tensor for missing nodes (entire tensor will be zero)
    indices = torch.zeros(2, max_num * target_feature_dim, dtype=torch.long)  # Row and column indices (flattened)
    values = torch.zeros(max_num * target_feature_dim, dtype=torch.float32)  # All zero values

    sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size([max_num, target_feature_dim]))
    return sparse_tensor


def sparse_zeros_edges(max_num):
    indices = torch.zeros(2, max_num, dtype=torch.long)  # Two rows, `max_num` columns (for edge_index)
    values = torch.zeros(max_num, dtype=torch.float32)  # Zero values (edges)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size([2, max_num]))  # Shape: [2, max_num]

    return sparse_tensor

def sparse_zeros_edges_attrs(max_num, feature_dim=16):
    indices = torch.zeros(2, 0, dtype=torch.long)  # Initially empty indices
    values = torch.zeros(0, dtype=torch.float32)  # Initially empty values

    # Create the sparse tensor with shape [max_num, feature_dim]
    sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size([max_num, feature_dim]))

    return sparse_tensor