import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import os
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch_geometric.loader import DataLoader
import numpy as np
import time
from torch_geometric.data import HeteroData
from torch_geometric.data import Batch

class IterSubGraphs(Dataset):
    
    def __init__(self, root, batch_size=1, transform=None):
        self.batch_size = batch_size
        self.data_path = Path(root)
        self.file_list = sorted(self.data_path.glob("subgraph_*.pt"))
        self.transform = transform
        print(f"[Init] Found {len(self.file_list)} files in {self.data_path}")

    def __getitem__(self, index):
        file_path = self.file_list[index]  # Retrieve the file path based on the index
        
        print(f"[GetItem] Loading {file_path.name}")
        try:
            # set up weights_only to avoid loading error
            batch = torch.load(file_path, map_location="cpu", weights_only=False)

            if batch is None:
                print(f"[GetItem] subgraph in {file_path.name} is None.")
                return None
            print(f"[GetItem] Loaded {file_path.name}")
            
        except Exception as e:
            print(f"[GetItem] Failed to load {file_path.name}: {e}")
            return None
        
        # Apply transformations if any
        if self.transform:
            batch = self.transform(batch)
            print("[GetItem] Transform applied")
        
        return batch

    def __iter__(self):

        worker_info = get_worker_info()
        if worker_info is None:
            file_list = self.file_list
            print("[Iter] Running in main process")
        else:
            total = len(self.file_list)
            per_worker = int(np.ceil(total / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, total)
            file_list = self.file_list[start:end]
            print(f"[Worker {worker_info.id}] Processing files {start} to {end}")

        for file_path in file_list:
            print(f"[Iter] Loading {file_path.name}")
            try:
                batch = torch.load(file_path, map_location="cpu")

                if batch is None:
                    print(f"[Iter] subgraph in {file_path.name} is None.")
                    continue
                print(f"[Iter] Loaded {file_path.name}")
                print(f"[Iter] Loaded {file_path.name}")

            except Exception as e:
                print(f"[Iter] Failed to load {file_path.name}: {e}")
                continue

            # calculate and set num_nodes
            for node_type in batch.keys():  # Now iterating over node types like 'Package_Name', 'Path', etc.
                # Skip edge-related keys like 'edge_index' and 'edge_attr'
                if node_type in ['edge_index', 'edge_attr']:
                    continue
                if hasattr(batch[node_type], 'x'):
                    node_data = batch[node_type].x 
                    batch[node_type].num_nodes = node_data.size(0)  # Assuming node_data has the shape [num_nodes, features]
            
            if self.transform:
                batch = self.transform(batch)
                print("[Iter] Transform applied")
            yield batch
            print("[Iter] Yielded one batch")

    def __len__(self):
        return len(self.file_list)

def to_dense_safe(data, device):
    # Loop through all node types
    for key in data.node_types:
        if hasattr(data[key], 'x') and torch.is_tensor(data[key].x):
            if data[key].x.is_sparse:
                data[key].x = data[key].x.to_dense()  # Convert sparse to dense

    # Loop through all edge types
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_index') and torch.is_tensor(data[edge_type].edge_index):
            if data[edge_type].edge_index.is_sparse:
                data[edge_type].edge_index = data[edge_type].edge_index.to_dense().long()  # Convert sparse to dense
        
        if hasattr(data[edge_type], 'edge_attr') and torch.is_tensor(data[edge_type].edge_attr):
            if data[edge_type].edge_attr.is_sparse:
                data[edge_type].edge_attr = data[edge_type].edge_attr.to_dense()  # Convert sparse to dense

    # Return the entire data object moved to the target device after processing
    return data.to(device)  # Move the entire object to the device after processing

# Custom collate function to handle HeteroData objects
def process_data(data, device):
    return [to_dense_safe(d, device) for d in data]

def collate_hetero_data(batch, device):
    """Custom collate function to handle batching of HeteroData objects."""
    # Apply to_dense_safe before batching
    batch = Batch.from_data_list(batch)
    return batch.to(device)

if __name__ == "__main__":

    data_path = Path.cwd().joinpath("test-small", "processed")
    
    dataset = IterSubGraphs(root=data_path, batch_size=10)

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Start with batch size of 1 for testing
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid any worker-related issues
        pin_memory=False,
        prefetch_factor=None,
        collate_fn=None 
    )

    print("[Main] Starting dataloader iteration...")

    try:
        for idx, batch in enumerate(dataloader):
            print(f"[Main] --- BATCH {idx} RECEIVED ---")
            # Check if batch is None
            if batch is None:
                print(f"[Main] subgraph is None, skipping.")
                continue

            # Check if the saved file exists
            batch_filename = f"batch_{idx}.pt"
            saved_filepath = data_path.joinpath(batch_filename)
            if saved_filepath.exists():
                print(f"[Main] File {saved_filepath} saved successfully.")
            else:
                print(f"[Main] File {saved_filepath} not found.")

    except Exception as e:
        print(f"[ERROR] DataLoader iteration failed: {e}")