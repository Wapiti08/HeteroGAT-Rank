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
    "type": Action (Path)| DNS(Hostname) | CMD (command) | Socket (IP, Port, Hostnames)
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
import pickle
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
from torch_geometric.transforms import Pad
from tqdm import tqdm
import psutil
from utils import prostring
from ext import fea_encoder
from ext import hetergraph



def load_in_chunks(file_path, chunk_size):
    with open(file_path, 'rb') as fr:
        data = pickle.load(fr)[:50]
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]


@ray.remote
def process_subgraphs(subgraph, global_node_id_map, global_node_counter):
    '''  Processes a batch of subgraphs: encodes features, constructs heterogeneous graphs.

    args:

    return:
        torch_geometric.data.Batch: A batch of padded heterogeneous graph data objects.
    '''
    return hetergraph.hetero_graph_build(subgraph, global_node_id_map, global_node_counter)

class LabeledSubGraphs(Dataset):
    
    def __init__(self, root, batch_size=10, transform=None, pre_transform=None, pre_filter=None):
        '''
        :param root: root directory where the dataset is stored
        :param batch_size: number of subgraphs to store in a single file
        :param transform: a function/transform applied to data objects
        :param pre_transform: a function applied before saving to disk
        :param pre_filter: a function to filter data objects
        '''
        self.batch_size = self.get_adaptive_batch_size(batch_size)
        self.data_path = root
        super().__init__(root, transform, pre_transform, pre_filter)

    @staticmethod
    def get_adaptive_batch_size(default_bs):
        total_mem = psutil.virtual_memory().total/(1024 * 3)
        if total_mem < 16:
            return max(1, default_bs/2)
        return default_bs

    @property
    def raw_file_names(self):
        return ['test_top_100_subgraphs.pkl']
    
    @property
    def processed_file_names(self):
        # generate a list of filenames for processed files
        if not osp.exists(self.processed_dir):
            return []
        return [f'batch_{i}.pt' for i in range(len(os.listdir(self.processed_dir)))]


    def download(self):
        ''' if the raw data needs to be downloaded, implement it here
        
        '''
        pass

    @staticmethod
    def get_max_parallel_tasks(task_cpus = 4, utilization_ratio=0.9):
        # there are 32 total available cpus
        available = ray.available_resources().get("CPU", 32)  
        usable = int(available * utilization_ratio)
        return max(1, usable // task_cpus)

    def process(self,):
        # load the raw data --- non-packaged directory to avoid large size package to Ray
        raw_path = osp.join(self.data_path.parent.parent.joinpath("data").as_posix(),\
                             "test_top_100_subgraphs.pkl")
        

        ray.shutdown()
        # Set OOM mitigation variables before initializing Ray
        os.environ["RAY_memory_usage_threshold"] = "0.9"  # Adjust based on node capacity
        os.environ["RAY_memory_monitor_refresh_ms"] = "0" 

        # initialize ray
        ray.init(runtime_env={"working_dir": Path.cwd().parent.as_posix(), \
                        "excludes": ["logs/", "*.pt", "*.json", "*.csv", "*.pkl"]})
        # process batches in parallel
        max_parallel = self.get_max_parallel_tasks()

        tasks = []
        chunk_id = 0
        # Global ID mapping
        global_node_id_map = {}
        global_node_counter = 0

        for batch in load_in_chunks(raw_path, self.batch_size):
            for subgraph in batch:
                task = process_subgraphs.remote(subgraph, global_node_id_map, global_node_counter)  # Submit the task
                tasks.append(task)
        
        # collect the results
        results = ray.get(tasks)

        # merge the results from all the tasks
        for result in results:
            processed_data, local_node_map, local_counter = result
            # update global node ID and counter
            for key, value in local_node_map.items():
                if key not in global_node_id_map:
                    global_node_id_map[key] = global_node_counter
                    global_node_counter += 1

            # Save the processed data
            for i, data in enumerate(processed_data):
                torch.save(data, osp.join(self.processed_dir, f'batch_{i}.pt'))

        # Shutdown Ray after processing
        ray.shutdown()


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        batch = torch.load(osp.join(self.processed_dir, f'batch_{idx}.pt'))
        return batch


if __name__ == "__main__":
    # load pickle format of graph dataset with graph representations
    data_path = Path.cwd().joinpath("test-small")

    # create an instance of the dataset
    dataset = LabeledSubGraphs(root=data_path, batch_size=10)

    # access the length of dataset
    print(f"Dataset length: {len(dataset)}")

    # get a processed batch
    batch_idx = 0
    if batch_idx < len(dataset):
        batch_data = dataset.get(batch_idx)
        if isinstance(batch_data, ray.ObjectRef):
            print(f"loaded batch {batch_idx} with {len(ray.get(batch_data))} subgraphs")
        else:
            print(f"loaded batch {batch_idx} with {len(batch_data)} subgraphs")
    else:
        print("Batch index out of range")

    
    

