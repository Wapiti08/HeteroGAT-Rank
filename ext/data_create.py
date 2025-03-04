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


@ray.remote
def process_subgraphs(subgraph_batch):
    '''
    process a batch of subgraphs and return a batched data object
    '''

    seq_encoder = encoder.SequenceEncoder()
    iden_encoder = encoder.IdentityEncoder(dtype=torch.float, output_dim=seq_encoder.embedding_dim)

    max_edges = max([len(subgraph['edges']) for subgraph in subgraph_batch])

    data_list = []
    for subgraph in subgraph_batch[:10]:
        # encode nodes and edges
        nodes = subgraph['nodes']
        edges = subgraph['edges']
        label = subgraph['label']

        node_df = pd.DataFrame(nodes)
        edge_df = pd.DataFrame(edges)

        # initialize heterodata
        hetero_data = HeteroData()

        # process nodes
        for node_type in node_df['type'].unique():
            type_nodes = node_df[node_df['type']==node_type]
            if type_nodes.empty:
                continue

            if node_type == 'Port':
                # ensure the value is numeric
                type_nodes['value'] = pd.to_numeric(type_nodes['value'], errors='coerce').fillna(0)
                features = iden_encoder(type_nodes[['value']])

            else:
                # ensure all missing values are handled properly
                type_nodes['value'] = type_nodes['value'].fillna("")
                features = seq_encoder(type_nodes[['value']])

            hetero_data[node_type].x = features
            # comment it for second-stage exploration
            # hetero_data['node'].eco = cate_encoder(type_nodes[['eco']])

        # process edges
        for edge_type in edge_df['type'].unique():
            type_edges = edge_df[edge_df['type'] == edge_type]
            
            # handle missing part
            type_edges["source"] = type_edges['source'].fillna("")
            type_edges["target"] = type_edges["target"].fillna("")

            sources = seq_encoder(type_edges[['source']].astype(str))
            targets = seq_encoder(type_edges[['target']].astype(str))

            sources = torch.as_tensor(sources, dtype=torch.long).view(1, -1)
            targets = torch.as_tensor(targets, dtype=torch.long).view(1, -1)
            
            # ensure all edge indices are padded to max_edges
            pad_size = max(0, max_edges - sources.shape[1])             
            
            if pad_size > 0:
                # use -1 for padding
                pad_tensor = torch.full((1, pad_size), -1, dtype=torch.long)
                sources = torch.cat([sources, pad_tensor], dim = 1)
                targets = torch.cat([targets, pad_tensor], dim= 1)


            # ensure the shape is (2,max_edges)
            hetero_data[edge_type].edge_index = torch.cat([sources, targets], dim=0)

            edge_values = seq_encoder(type_edges[['value']])
            # pad attributes
            edge_values = torch.cat([edge_values, torch.zeros(pad_size, edge_values.shape[1])], dim=0)

            hetero_data[edge_type].edge_attr = edge_values

        # add labels
        hetero_data['label'] = torch.tensor(label, dtype=torch.long)

        data_list.append(hetero_data)
    
    print(data_list)
    
    # create a batched data object
    batch = Batch.from_data_list(data_list)

    # generate batch masks
    for edge_type in batch.edge_index_dict:
        mask = batch.edge_index_dict[edge_type] != -1
        batch.edge_mask_dict[edge_type] = mask

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


    def process(self):
        # load the raw data --- non-packaged directory to avoid large size package to Ray
        raw_path = osp.join(self.data_path.parent.parent.joinpath("data").as_posix(),\
                             "subgraphs.pkl")
        with open(raw_path, 'rb') as fr:
            subgraphs = pickle.load(fr)

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
                process_subgraphs.remote(batch)
                for batch in subgraph_batches[:10]
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

    
    

