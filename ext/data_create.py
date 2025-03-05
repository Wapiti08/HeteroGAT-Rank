'''
 # @ Create Time: 2024-12-18 09:27:11
 # @ Modified time: 2024-12-18 09:27:14
 # @ Description: create graph dataset suitable for GNN model training in pytorch
 
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
from ext import encoder
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


@ray.remote
def process_subgraphs(subgraph_batch, max_nodes_per_type, max_edges_per_type):
    """
    Process a batch of subgraphs and return a batched data object.
    """
    seq_encoder = encoder.SequenceEncoder()
    iden_encoder = encoder.IdentityEncoder(dtype=torch.float, output_dim=seq_encoder.embedding_dim)

    # Initialize the Pad transform with the calculated max values
    pad_transform = Pad(max_num_nodes=max_nodes_per_type, max_num_edges=max_edges_per_type)

    data_list = []
    all_node_types = set()
    feature_dim = None
    
    for subgraph in subgraph_batch:
        nodes = subgraph['nodes']
        edges = subgraph['edges']
        label = subgraph['label']

        node_df = pd.DataFrame(nodes)
        edge_df = pd.DataFrame(edges)

        all_node_types.update(node_df['type'].unique())

        hetero_data = HeteroData()
        # Process nodes
        for node_type, type_nodes in node_df.groupby('type'):
            if node_type == 'Port':
                type_nodes['value'] = pd.to_numeric(type_nodes['value'], errors='coerce').fillna(0)
                features = iden_encoder(type_nodes[['value']])
            else:
                type_nodes['value'] = type_nodes['value'].fillna("")
                features = seq_encoder(type_nodes[['value']])

            # Set feature_dim based on the first processed node type
            if feature_dim is None:
                feature_dim = features.size(1)

            hetero_data[node_type].x = features
            hetero_data[node_type].num_nodes = len(type_nodes)

        # Process edges
        for edge_type, type_edges in edge_df.groupby('type'):
            type_edges['source'] = type_edges['source'].fillna("")
            type_edges['target'] = type_edges['target'].fillna("")

            sources = seq_encoder(type_edges[['source']].astype(str)).view(1, -1)
            targets = seq_encoder(type_edges[['target']].astype(str)).view(1, -1)

            hetero_data[edge_type].edge_index = torch.cat([sources, targets], dim=0)
            hetero_data[edge_type].edge_attr = seq_encoder(type_edges[['value']])

        # Add label
        hetero_data['label'] = torch.tensor(label, dtype=torch.long)

        # check whether there is missing key
        for node_type in all_node_types:
            if node_type not in hetero_data:
                hetero_data[node_type].x = torch.zeros((max_nodes_per_type[node_type], feature_dim))
                hetero_data[node_type].num_nodes = max_nodes_per_type[node_type]

        try:
            # apply padding transform
            hetero_data = pad_transform(hetero_data)
        except Exception as e:
            print(e)
            print(hetero_data)
            exit()

        data_list.append(hetero_data)

    # Create a batched data object
    batch = Batch.from_data_list(data_list)

    return batch


class LabeledSubGraphs(Dataset):
    
    def __init__(self, root, batch_size=10, transform=None, pre_transform=None, pre_filter=None):
        '''
        :param root: root directory where the dataset is stored
        :param batch_size: number of subgraphs to store in a single file
        :param transform: a function/transform applied to data objects
        :param pre_transform: a function applied before saving to disk
        :param pre_filter: a function to filter data objects
        '''
        self.batch_size = batch_size
        self.data_path = root
        super().__init__(root, transform, pre_transform, pre_filter)

        # initialize encoders for node/edge attributes
        # self.seq_encoder = encoder.SequenceEncoder()
        # self.iden_encoder = encoder.IdentityEncoder(dtype=torch.float)


    @property
    def raw_file_names(self):
        return ['subgraphs.pkl']
    
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

    def pad_size(self, subgraphs):
        # initialize dictionaries to track maximum counts
        max_nodes_per_type = {}
        max_edges_per_type = {}

        # First pass to determine max nodes and edges per type
        for subgraph in subgraphs:
            node_df = pd.DataFrame(subgraph['nodes'])
            edge_df = pd.DataFrame(subgraph['edges'])
            if "type" in node_df.columns:
                # Update max nodes per type
                for node_type, type_nodes in node_df.groupby('type'):
                    max_nodes_per_type[node_type] = max(max_nodes_per_type.get(node_type, 0), len(type_nodes))
            if "type" in edge_df.columns:
                # Update max edges per type
                for edge_type, type_edges in edge_df.groupby('type'):
                    max_edges_per_type[edge_type] = max(max_edges_per_type.get(edge_type, 0), len(type_edges))

        return max_nodes_per_type, max_edges_per_type

    def process(self):
        # load the raw data --- non-packaged directory to avoid large size package to Ray
        raw_path = osp.join(self.data_path.parent.parent.joinpath("data").as_posix(),\
                             "subgraphs.pkl")
        
        with open(raw_path, 'rb') as fr:
            subgraphs = pickle.load(fr)

        # initialize dictionaries to track maximum counts
        max_nodes_per_type = {}
        max_edges_per_type = {}

        max_nodes_per_type, max_edges_per_type = self.pad_size(subgraphs)
        
        # convert to pandas framework for easier processing
        subgraphs_df = pd.DataFrame(subgraphs)

        # split into batches
        num_batches = len(subgraphs_df) // self.batch_size + 1
        subgraph_batches = [
            subgraphs_df.iloc[i * self.batch_size: (i+1) * self.batch_size].to_dict(orient='records')
            for i in range(num_batches)
        ]

        # process batches in parallel
        ray.init(runtime_env={"working_dir": Path.cwd().parent.as_posix(), \
                              "excludes": ["logs/", "*.pt", "*.json", "*.csv", "*.pkl"]})

        try:
            tasks = [
                # process_subgraphs.remote(batch, self.seq_encoder, self.iden_encoder)
                process_subgraphs.remote(batch,max_nodes_per_type, max_edges_per_type )
                for batch in subgraph_batches
            ]
            results = ray.get(tasks)    
        finally:
            ray.shutdown()

        # save the processed batches
        for i, batch in enumerate(results):
            torch.save(batch, osp.join(self.processed_dir, f'batch_{i}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        batch = torch.load(osp.join(self.processed_dir, f'batch_{idx}.pt'))
        return batch


if __name__ == "__main__":
    # load pickle format of graph dataset with graph representations
    data_path = Path.cwd().joinpath("output")

    # create an instance of the dataset
    dataset = LabeledSubGraphs(root=data_path, batch_size=10)

    # access the length of dataset
    print(f"Dataset length: {len(dataset)}")

    # get a processed batch
    batch_idx = 0
    if batch_idx < len(dataset):
        batch_data = dataset.get(batch_idx)
        print(f"loaded batch {batch_idx} with {len(batch_data)} subgraphs")
    else:
        print("Batch index out of range")

    
    

