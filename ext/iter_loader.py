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
from dask import delayed

def to_sparse_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a dense tensor to a sparse tensor.
    If it's already sparse, returns it unchanged.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.is_sparse:
        return tensor
    # Convert to sparse COO format
    return tensor.to_sparse()

class IterSubGraphs(Dataset):
    def __init__(self, root, batch_size=1, transform=None):
        self.batch_size = batch_size
        self.data_path = Path(root)
        self.file_list = sorted(self.data_path.glob("subgraph_*.pt"))
        self.transform = transform
        print(f"[Init] Found {len(self.file_list)} files in {self.data_path}")

    def _make_sparse(self, batch: dict) -> dict:
        """
        Recursively convert all tensor components in the batch to sparse tensors.
        """
        # convert node features
        for node_type in batch.node_types:
            if hasattr(batch[node_type], 'x'):
                batch[node_type].x = to_sparse_tensor(batch[node_type].x)

        # convert edge features
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], 'edge_index'):
                batch[edge_type].edge_index = to_sparse_tensor(batch[edge_type].edge_index)
            if hasattr(batch[edge_type], 'edge_attr'):
                batch[edge_type].edge_attr = to_sparse_tensor(batch[edge_type].edge_attr)

        return batch

    def __getitem__(self, index):
        file_path = self.file_list[index]
        print(f"[GetItem] Loading {file_path.name}")
        try:
            batch = torch.load(file_path, map_location="cpu", weights_only=False)
            if batch is None:
                print(f"[GetItem] subgraph in {file_path.name} is None.")
                return None
            print(f"[GetItem] Loaded {file_path.name}")
        except Exception as e:
            print(f"[GetItem] Failed to load {file_path.name}: {e}")
            return None

        # Convert all tensor components to sparse
        batch = self._make_sparse(batch)
        print("[GetItem] Converted to sparse")

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
            except Exception as e:
                print(f"[Iter] Failed to load {file_path.name}: {e}")
                continue

            # calculate and set num_nodes
            for node_type, data in batch.items():
                if node_type in ['edge_index', 'edge_attr']:
                    continue
                if hasattr(data, 'x'):
                    data.num_nodes = data.x.size(0)

            # Convert to sparse
            batch = self._make_sparse(batch)
            print("[Iter] Converted to sparse")

            if self.transform:
                batch = self.transform(batch)
                print("[Iter] Transform applied")
            yield batch
            print("[Iter] Yielded one batch")

    def __len__(self):
        return len(self.file_list)


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