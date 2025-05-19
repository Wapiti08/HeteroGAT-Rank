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
import psutil
from ext import hetergraph
import gc
from utils import sparsepad

node_types = ["Path", "DNS Host", "Package_Name", "IP", "Command", "Port"]

edge_types = [
            ('Package_Name', 'Action', 'Path'),
            ('Package_Name', 'DNS', 'DNS Host'),
            ('Package_Name', 'CMD', 'Command'),
            ('Package_Name', 'Socket', 'IP'),
            ('Package_Name', 'Socket', 'Port'),
            # ('Package_Name', 'Socket', 'Hostnames'),
        ]


def load_in_chunks(file_path, chunk_size):
    with open(file_path, 'rb') as fr:
        data = pickle.load(fr)
        # second round of generation
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]

@ray.remote
def process_subgraphs(subgraph, global_node_id_map, global_node_counter):
    '''  Processes a batch of subgraphs: encodes features, constructs heterogeneous graphs.

    args:

    return:
        torch_geometric.data.Batch: A batch of padded heterogeneous graph data objects.
    '''
    # Check if the data is valid
    data, global_node_id_map, global_node_counter = hetergraph.hetero_graph_build(subgraph, global_node_id_map, global_node_counter)

    if data is None:
        print(f"Error: Processed data is None for subgraph {subgraph}")
        return None, global_node_id_map, global_node_counter

    return data, global_node_id_map, global_node_counter


def get_max_size(file_path):
    # Step 1: Load the data from the pickle file
    with open(file_path, 'rb') as fr:
        data = pickle.load(fr)

    # Step 2: Initialize the variables to track max nodes and max edges
    max_nodes_per_type = {}
    max_edges_per_type = {}

    # Step 3: Traverse through each subgraph to find the max nodes and edges
    for subgraph in data:
        # traverse through nodes to find the max number of nodes for each type
        nodes_df = pd.DataFrame(subgraph['nodes'])

        if nodes_df.empty:
            print("Warning: Empty nodes DataFrame in subgraph.")
            continue

        # calculate max nodes per type in the current subgraph
        for node_type, node_data in nodes_df.groupby("type"):
            num_nodes = len(node_data)
            if node_type not in max_nodes_per_type:
                max_nodes_per_type[node_type] = num_nodes
            else:
                max_nodes_per_type[node_type] = max(max_nodes_per_type[node_type], num_nodes)

        # convert edges to DataFrame
        edges_df = pd.DataFrame(subgraph['edges'])
        
        if edges_df.empty:
            print("Warning: Empty edges DataFrame in subgraph.")
            continue

        # calculate max edges per type in the current subgraph
        for edge_type, edge_data in edges_df.groupby("type"):
            num_edges = len(edge_data)
            if edge_type not in max_edges_per_type:
                max_edges_per_type[edge_type] = num_edges
            else:
                max_edges_per_type[edge_type] = max(max_edges_per_type[edge_type], num_edges)

    # Step 4: Return the max nodes and edges per type
    print("Max nodes per type:", max_nodes_per_type)
    print("Max edges per type:", max_edges_per_type)

    return max_nodes_per_type, max_edges_per_type


class LabeledSubGraphs(Dataset):
    
    def __init__(self, root, batch_size, transform, pre_transform, pre_filter):
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
        available_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # in GB
        if available_mem < 16:
            return max(1, default_bs // 2)
        elif available_mem < 64:
            return max(1, default_bs // 1.5)
        return default_bs

    @property
    def raw_file_names(self):
        return ['subgraphs.pkl']
    
    @property
    def processed_file_names(self):
        # generate a list of filenames for processed files
        if not osp.exists(self.processed_dir):
            return []
        return [f'batch_{i}.pt' for i in range(len(os.listdir(self.processed_dir)))]

    @staticmethod
    def get_max_parallel_tasks(task_cpus = 2, utilization_ratio=0.98):
        # there are 48 total available cpus
        available = ray.available_resources().get("CPU", 32)  
        usable = int(available * utilization_ratio)
        return max(1, usable // task_cpus)


    @staticmethod
    def pad_subgraph(subgraph, max_nodes_per_type, max_edges_per_type, target_feature_dim=400):
        """
        Pads the subgraph's node features and edge index to match the max_nodes
        :param subgraph: The subgraph (HeteroData) to pad
        :param max_num: the max number among max_nodes and max_edges
        :return: The padded subgraph
        """
        for node_type, max_num in max_nodes_per_type.items():
            # get node features
            for node_type in node_types:
                node_features = subgraph[node_type].x if node_type in  \
                    subgraph.node_types else sparsepad.sparse_zeros(max_num, target_feature_dim)
                
                if node_features.size(0) < max_num:
                    subgraph[node_type].x = sparsepad.sparse_padding_with_values(node_features, max_num, target_feature_dim)


        for edge_type, max_num in max_edges_per_type.items():
            for edge_type in edge_types:
                # Check if edge_type exists in the subgraph
                if edge_type in subgraph.edge_types:
                    # If edge_type exists, use the existing edge_index
                    edge_index = subgraph[edge_type].edge_index
                else:
                    # If edge_type does not exist, apply padding
                    edge_index = sparsepad.sparse_zeros_edges(max_num)
                    # Update the subgraph with the padded edge_index for the missing edge_type
                    subgraph[edge_type].edge_index = edge_index

                num_edges = edge_index.size(1)

                # cal necessary padding size
                padding_size = max_num - num_edges
                if padding_size > 0:
                    # edge_index is [2, max_num]
                    subgraph[edge_type].edge_index = sparsepad.sparse_padding_with_values_edges(edge_index, max_num)
                
                if hasattr(subgraph[edge_type], 'edge_attr'):
                    edge_attr = subgraph[edge_type].edge_attr
                    if edge_attr.size(0) < max_num:
                        # the default edge attr feature dim is 16
                        subgraph[edge_type].edge_attr = sparsepad.sparse_padding_with_values(edge_attr, max_num, 16)
                else:
                    # If edge_attr does not exist, create a new edge_attr filled with zeros
                    subgraph[edge_type].edge_attr = sparsepad.sparse_zeros_edges(max_num)
                    print(f"Created new edge_attr for {edge_type} with zeros.")
            
        return subgraph
        

    def process(self,):
        # load the raw data --- non-packaged directory to avoid large size package to Ray
        raw_path = osp.join(self.data_path.parent.parent.joinpath("data").as_posix(),\
                             "subgraphs.pkl")
        # get global max size in max_nodes and max_edges
        max_nodes_per_type, max_edges_per_type = get_max_size(raw_path)
        
        ray.shutdown()

        # Set OOM mitigation variables before initializing Ray
        os.environ["RAY_memory_usage_threshold"] = "0.95"  # Adjust based on node capacity
        os.environ["RAY_memory_monitor_refresh_ms"] = "0" 

        # initialize ray
        ray.init(ignore_reinit_error=True, runtime_env={"working_dir": Path.cwd().parent.as_posix(), \
                        "excludes": ["logs/", "*.pt", "*.json", "*.csv", "*.pkl"]})
        
        # second round of generation
        chunk_id = 0
        
        # Global ID mapping
        global_node_id_map = {}
        global_node_counter = 0

        # process in chunks, like every 100 subgraphs every time in 9k
        for batch in load_in_chunks(raw_path, self.batch_size):
            batch_tasks = []

            for subgraph in batch:
                task = process_subgraphs.remote(subgraph, global_node_id_map, global_node_counter)  # Submit the task
                batch_tasks.append(task)

            # Use ray.wait to wait for the tasks to finish, process them as they complete
            while batch_tasks:
                # Wait for any task to finish (returns finished task and remaining tasks)
                ready, remaining = ray.wait(batch_tasks, num_returns=1, timeout=5)
                
                for task in ready:

                    processed_data, local_node_map, local_counter = ray.get(task)
                    
                    if processed_data is not None:
                        # update global node ID and counter
                        global_node_id_map.update(local_node_map)
                        global_node_counter = max(global_node_counter, local_counter)

                        # fill missing nodes and saved processed data
                        processed_data = self.pad_subgraph(processed_data, max_nodes_per_type, max_edges_per_type)

                        # Save the entire HeteroData object as a .pt file
                        torch.save(processed_data, osp.join(self.processed_dir, f'subgraph_{chunk_id}.pt'))
                
                # Remove the tasks that have been processed
                batch_tasks = remaining
            
            # Increment chunk_id for the next batch
            chunk_id += 1
            # force garbage collection
            gc.collect() 
            
        # Shutdown Ray after processing
        ray.shutdown()

    def save_global_node_id_map(self, global_node_id_map):
        # Save the global node ID mapping to a file
        with Path(self.processed_dir).parent.joinpath('global_node_id_map.pkl', 'wb') as f:
            pickle.dump(global_node_id_map, f)
        print(f"Global node ID mapping saved to {Path(self.processed_dir).parent.joinpath('global_node_id_map.pkl')}")


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        batch = torch.load(osp.join(self.processed_dir, f'batch_{idx}.pt'))
        return batch


if __name__ == "__main__":
    # load pickle format of graph dataset with graph representations
    data_path = Path.cwd().joinpath("output_map")

    # create an instance of the dataset
    dataset = LabeledSubGraphs(data_path, 5, None, None, None)

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

    
    

