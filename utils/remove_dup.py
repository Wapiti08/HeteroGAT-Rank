'''
 # @ Create Time: 2025-06-08 11:12:01
 # @ Modified time: 2025-06-08 11:12:04
 # @ Description: remove potentially duplicated subgraphs from generation
 '''


import os
import torch
import hashlib
from pathlib import Path
import io
import os
import io
import torch
import hashlib
from torch_geometric.data import HeteroData
from pathlib import Path
from collections import defaultdict
from typing import Tuple
from multiprocessing import Pool, cpu_count


def tensor_to_numpy(tensor):
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


def process_file_for_hash(filepath: str) -> Tuple[str, str]:
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        h = hash_heterodata(data)
        return (filepath, h)
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return (filepath, None)

def hash_heterodata(data: HeteroData) -> str:
    m = hashlib.sha256()

    # hash label
    if "label" in data:
        m.update(str(int(data["label"].item())).encode())

    # hash node features
    for node_type in sorted(data.node_types):
        if "x" in data[node_type]:
            tensor = data[node_type]["x"]
            m.update(node_type.encode())
            m.update(str(tensor.shape).encode())
            # ✅ convert sparse tensors to dense before hashing
            if tensor.is_sparse:
                tensor = tensor.to_dense()
            m.update(tensor_to_numpy(tensor).tobytes())

    # hash edge types
    for edge_type in sorted(data.edge_types):
        edge = data[edge_type]
        if "edge_index" in edge:
            m.update(str(edge_type).encode())
            tensor = edge["edge_index"]
            if tensor.is_sparse:
                tensor = tensor.to_dense()
            m.update(str(tensor.shape).encode())
            m.update(tensor_to_numpy(tensor).tobytes())
        if "edge_attr" in edge:
            tensor = edge["edge_attr"]
            if tensor.is_sparse:
                tensor = tensor.to_dense()
            m.update(str(tensor.shape).encode())
            m.update(tensor_to_numpy(tensor).tobytes())

    return m.hexdigest()


def find_duplicates_by_content_parallel(directory: str, num_workers: int = None):
    if num_workers is None:
        num_workers = min(cpu_count(), 32)  # 避免线程爆炸

    # 构造所有待处理的文件路径
    file_list = [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.startswith("subgraph") and fname.endswith(".pt")
    ]

    seen = {}
    duplicates = []

    # 多进程并行计算 hash
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file_for_hash, file_list)

    for filepath, h in results:
        if h is None:
            continue
        fname = os.path.basename(filepath)
        if h in seen:
            duplicates.append((fname, os.path.basename(seen[h])))
        else:
            seen[h] = filepath

    print(f"\n🔁 Found {len(duplicates)} duplicates.")
    return duplicates


def remove_duplicates_by_content_parallel(directory: str):
    duplicates = find_duplicates_by_content_parallel(directory)
    for dup, original in duplicates:
        print(f"🗑 Removing duplicate {dup} (duplicate of {original})")
        os.remove(os.path.join(directory, dup))

# Example usage:
if __name__ == "__main__":

    import multiprocessing as mp
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # only for this script

    mp.set_start_method("spawn", force=True) 

    data_path = Path.cwd().parent.joinpath("ext", "new", "processed").as_posix()
    remove_duplicates_by_content_parallel(data_path)
