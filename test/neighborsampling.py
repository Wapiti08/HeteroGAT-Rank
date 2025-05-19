import torch

def dynamic_sampling_with_edge_type(edge_index, attn_weights, edge_types, k):
    """
    动态采样k个边，基于边的attn_weight，同时考虑边的类型edge_type（三元组形式）。
    支持稀疏张量处理。

    参数:
    - edge_index (Tensor or SparseTensor): 边的索引，可以是稀疏张量或密集张量，形状为[2, num_edges]。
    - attn_weights (dict): 边类型对应的注意力权重字典，形如：
        {('Package_Name', 'Action', 'Path'): torch.tensor([0.1, 0.3, 0.2]), ...}
    - edge_types (list): 边类型三元组的列表，例如：
        [('Package_Name', 'Action', 'Path'), ('Package_Name', 'DNS', 'DNS Host'), ...]
    - k (int): 需要选择的边的数量。

    返回:
    - sampled_edge_index (Tensor): 采样后的边索引，形状为[2, k]。
    - sampled_attn_weight (Tensor): 采样后的边注意力权重，形状为[k]。
    - sampled_edge_type (Tensor): 采样后的边类型，形状为[k]。
    """
    
    if isinstance(edge_index, torch.sparse.Tensor):
        # 处理稀疏张量情况
        edge_index = edge_index.coalesce().indices()  # 获取非零元素的索引
    
    sampled_edge_index = []
    sampled_attn_weight = []
    sampled_edge_type = []
    
    # 1. 对每种边类型分别进行采样
    for et in edge_types:
        # 获取当前类型的边的索引
        type_mask = (edge_types == et)  # 边类型匹配
        type_edge_index = edge_index[:, type_mask]
        type_attn_weight = attn_weights.get(et, torch.tensor([]))  # 获取该类型的注意力权重

        if type_attn_weight.numel() == 0:
            continue  # 如果该类型的边没有权重，跳过

        # 对当前类型的边进行排序并选择前k个边
        _, sorted_indices = torch.topk(type_attn_weight, k, largest=True)

        # 选择前k个边及其注意力权重
        sampled_edge_index.append(type_edge_index[:, sorted_indices])
        sampled_attn_weight.append(type_attn_weight[sorted_indices])
        sampled_edge_type.append(torch.full((k,), et, dtype=torch.long))  # 对应的边类型
        
    # 将采样结果拼接起来
    sampled_edge_index = torch.cat(sampled_edge_index, dim=1)
    sampled_attn_weight = torch.cat(sampled_attn_weight)
    sampled_edge_type = torch.cat(sampled_edge_type)

    return sampled_edge_index, sampled_attn_weight, sampled_edge_type


# 示例
# 假设edge_index是一个稀疏张量，形状为[2, num_edges]
edge_index_sparse = torch.sparse_coo_tensor(
    indices=torch.tensor([[0, 1, 2, 3, 4, 5],
                          [5, 6, 7, 8, 9, 10]]),
    values=torch.tensor([0.1, 0.8, 0.5, 0.2, 0.9, 0.7]),
    size=(6, 11)
)

# 假设attn_weights是每条边类型的注意力权重字典
attn_weights = {
    ('Package_Name', 'Action', 'Path'): torch.tensor([0.1, 0.3, 0.2]),  
    ('Package_Name', 'DNS', 'DNS Host'): torch.tensor([0.4, 0.5, 0.6]),
    ('Package_Name', 'CMD', 'Command'): torch.tensor([0.7, 0.8]),
    ('Package_Name', 'Socket', 'IP'): torch.tensor([0.9, 0.2]),
}

# 假设edge_types表示每条边的类型（三元组）
edge_types = [
    ('Package_Name', 'Action', 'Path'),
    ('Package_Name', 'DNS', 'DNS Host'),
    ('Package_Name', 'CMD', 'Command'),
    ('Package_Name', 'Socket', 'IP'),
    ('Package_Name', 'Socket', 'Port'),
    ('Package_Name', 'Socket', 'Hostnames'),
]

# 设定k为2，选择top 2的边
k = 2

sampled_edge_index, sampled_attn_weight, sampled_edge_type = dynamic_sampling_with_edge_type(edge_index_sparse, attn_weights, edge_types, k)

print("采样后的边索引：")
print(sampled_edge_index)
print("采样后的注意力权重：")
print(sampled_attn_weight)
print("采样后的边类型：")
print(sampled_edge_type)
