'''
 # @ Create Time: 2024-12-18 09:27:11
 # @ Modified time: 2024-12-18 09:27:14
 # @ Description: create graph dataset suitable for GNN model training in pytorch in multiple steps (reduce one-time memory cost)
 
node: {
    'value': str content / numeric,
    "type": Package_Name | Path | IP | Hostname | Hostnames | Command | Port --- str content
    "eco": cate
 }

 edge: {
    "source": str content,
    "target": str content,
    "value": str content,
    "type": action (Path)| DNS(Hostname) | CMD (command) | Socket (IP, Port, Hostnames)
 }
 
 '''

import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import ray
import os
import os.path as osp
import torch
from torch_geometric.data import HeteroData, Dataset, Batch
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
import pickle
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
from torch_geometric.transforms import Pad
from tqdm import tqdm
from torch.utils.data import get_worker_info


# Initialize Ray (Make sure this is done once at the start of your script)
ray.init(ignore_reinit_error=True)

class IterSubGraphs(IterableDataset):
    
    def __init__(self, root, batch_size=10, transform=None):
        '''
        :param root: root directory where the dataset is stored
        :param batch_size: number of subgraphs to store in a single file
        :param transform: a function/transform applied to data objects
        '''
        self.batch_size = batch_size
        self.data_path = root
        self.file_list = sorted(self.data_path.glob("batch_*.pt"))
        self.transform = transform


    def __iter__(self):
        ''' yields batches one at a time to minimize moemory usage
        split file list per workder to avoid looping on the same list for all workers
        '''
        worker_info = get_worker_info()
        # single worker
        if worker_info is None:
            file_list = self.file_list
        
        else:
            total = len(self.file_list)
            per_worker = int(np.ceil(total / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, total)
            file_list = self.file_list[start:end]

        for file_path in file_list:
            batch = torch.load(file_path, map_location="cpu")
            
            # Ensure batch is resolved if stored as Ray ObjectRefs
            if isinstance(batch, ray.ObjectRef):
                batch = ray.get(batch)

            if self.transform:
                batch = self.transform(batch)
            yield batch


    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    # load pickle format of graph dataset with graph representations
    data_path = Path.cwd().joinpath("test-small", "processed")

    # create an instance of the dataset
    dataset = IterSubGraphs(root=data_path, batch_size=10)

    dataloader = DataLoader(
            dataset, 
            batch_size = 1, 
            shuffle=False, 
            num_workers=20,
            persistent_workers=True
            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch in dataloader:
        batch = batch[0].to(device)
        print(batch)
        break

    
    

