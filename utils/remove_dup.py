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

def tensor_to_numpy(tensor):
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()

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

def find_duplicates_by_content(directory):
    seen = {}
    duplicates = []

    for filename in sorted(os.listdir(directory)):
        if filename.startswith("subgraph") and filename.endswith(".pt"):
            path = os.path.join(directory, filename)
            try:
                data = torch.load(path, map_location="cpu",weights_only=False)
                h = hash_heterodata(data)
                if h in seen:
                    duplicates.append((filename, seen[h]))
                else:
                    seen[h] = filename
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    
    print(f"\n🔁 Found {len(duplicates)} duplicates.")
    return duplicates

def remove_duplicates_by_content(directory):
    duplicates = find_duplicates_by_content(directory)
    for dup, original in duplicates:
        print(f"🗑 Removing duplicate {dup} (duplicate of {original})")
        os.remove(os.path.join(directory, dup))

# Example usage:
if __name__ == "__main__":
    data_path = Path.cwd().parent.joinpath("ext", "new", "processed").as_posix()
    remove_duplicates_by_content(data_path)
