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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IterSubGraphs(IterableDataset):
    
    def __init__(self, root, batch_size=10, transform=None):
        self.batch_size = batch_size
        self.data_path = Path(root)
        self.file_list = sorted(self.data_path.glob("batch_*.pt"))
        self.transform = transform
        print(f"[Init] Found {len(self.file_list)} files in {self.data_path}")

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
                    print(f"[Iter] Batch in {file_path.name} is None.")
                    continue
                print(f"[Iter] Loaded {file_path.name}")
                print(f"[Iter] Loaded {file_path.name}")

            except Exception as e:
                print(f"[Iter] Failed to load {file_path.name}: {e}")
                continue

            # Optional: check type and size
            print(f"[Iter] batch type: {type(batch)}")
            print(f"[Iter] batch keys: {batch.keys() if hasattr(batch, 'keys') else 'N/A'}")

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


# Custom collate function to handle HeteroData objects
def collate_hetero_data(batch):
    """Custom collate function to handle batching of HeteroData objects."""
    return Batch.from_data_list(batch)

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
                print(f"[Main] Batch is None, skipping.")
                continue

            # Check if the saved file exists
            batch_filename = f"batch_{idx}.pt"
            saved_filepath = data_path.joinpath(batch_filename)
            if saved_filepath.exists():
                print(f"[Main] File {saved_filepath} saved successfully.")
            else:
                print(f"[Main] File {saved_filepath} not found.")

            time.sleep(0.5)  # Simulate processing time

            # Optionally, remove break to continue iterating
            break  # Only process the first batch for testing, remove this line to process all

    except Exception as e:
        print(f"[ERROR] DataLoader iteration failed: {e}")