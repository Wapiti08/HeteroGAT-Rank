import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import os
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.loader import DataLoader
import numpy as np
import time

try:
    import ray
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    use_ray = True
    ray_ObjectRef = ray.ObjectRef  # Save for isinstance checks
except ImportError:
    print("Ray not installed, continuing without Ray.")
    use_ray = False
    ray_ObjectRef = type(None)  # Dummy fallback type

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
                print(f"[Iter] Loaded {file_path.name}")
            except Exception as e:
                print(f"[Iter] Failed to load {file_path.name}: {e}")
                continue

            if use_ray and isinstance(batch, ray.ObjectRef):
                batch = ray.get(batch)
                print(f"[Worker {worker_info.id if worker_info else 0}] Retrieved from ray.ObjectRef")

            # Optional: check type and size
            print(f"[Iter] batch type: {type(batch)}")
            print(f"[Iter] batch keys: {batch.keys() if hasattr(batch, 'keys') else 'N/A'}")

            if self.transform:
                batch = self.transform(batch)
                print("[Iter] Transform applied")

            yield batch
            print("[Iter] Yielded one batch")

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    data_path = Path.cwd().joinpath("test-small", "processed")
    
    dataset = IterSubGraphs(root=data_path, batch_size=10)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=4,  # Reduce to test performance; 20 may be overkill for file I/O
        # persistent_workers=True,
        pin_memory=False,
        prefetch_factor=None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Main] Starting dataloader iteration...")

    try:
        for idx, batch in enumerate(dataloader):
            print(f"[Main] --- BATCH {idx} RECEIVED ---")
            
            if isinstance(batch, list) or isinstance(batch, tuple):
                print(f"[Main] batch is a {type(batch)} of len={len(batch)}")
                print(f"[Main] batch[0] type: {type(batch[0])}")
                print(f"[Main] batch[0] keys: {batch[0].keys() if hasattr(batch[0], 'keys') else 'N/A'}")
            else:
                print(f"[Main] batch type: {type(batch)}")
                print(f"[Main] batch keys: {batch.keys() if hasattr(batch, 'keys') else 'N/A'}")

            time.sleep(0.5)  # simulate some processing time
            break

    except Exception as e:
        print(f"[ERROR] DataLoader iteration failed: {e}")