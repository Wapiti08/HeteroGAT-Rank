'''
 # @ Create Time: 2024-12-18 09:27:11
 # @ Modified time: 2024-12-18 09:27:14
 # @ Description: create graph dataset suitable for GNN model training in pytorch
 
  node: {
    'value': str content / numeric,
    "type": Path | Package_Name | IP | Hostname | Command | Port --- str content
    "eco": cate
 }

 edge: {
    "source": str content,
    "target": str content,
    "value": str content,
    "type": action | DNS types
 }
 
 
 
 '''

import ray
from ext import encoder
import os
import os.path as osp
import torch
from torch_geometric.data import Data, Dataset, Batch
import pickle
import pandas as pd


@ray.remote
def process_subgraphs(subgraph_batch, seq_encoder, cate_encoder, iden_encoder):
    '''
    process a batch of subgraphs and return a batched data object
    '''
    data_list = []
    for subgraph in subgraph_batch:
        # encode nodes and edges
        nodes = subgraph['nodes']
        edges = subgraph['edges']
        label = subgraph['label']

        node_df = pd.DataFrame(nodes)
        edge_df = pd.DataFrame(edges)

        # apply encoders to node attributes
        node_features = []

        for idx, row in node_df.iterrows():
            if row['type'] == 'Port':
                # numeric encoding for Ports
                encoded_value = iden_encoder(pd.DataFrame([row['value']]))
            else:
                # sequence encoding for non-numeric values
                encoded_value = seq_encoder(pd.DataFrame([row['value']]))

            node_features.append(encoded_value)
        # concat like a batch
        node_features = torch.cat(node_features, dim=0)
        node_types = cate_encoder(node_df['type'])
        node_eco = cate_encoder(node_df['eco'])

        # apply encoders to edge attributes
        edge_sources = seq_encoder(edge_df['source'])
        edge_targets = seq_encoder(edge_df['target'])
        edge_values = seq_encoder(edge_df['value'])
        edge_types = cate_encoder(edge_df['type'])

        # construct a data object
        data = Data(
            x = torch.cat([node_features, node_types, node_eco], dim=1),
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long),
            edge_attr = torch.cat([edge_values, edge_types], dim=1),
            y = torch.tensor(label, dtype=torch.long)
        )

        data_list.append(data)
    
    # create a batched data object
    return Batch.from_data_list(data_list)



class LabeledSubGraphs(Dataset):
    
    def __init__(self, root, batch_size=100, transform=None, pre_transform=None, pre_filter=None):
        '''
        :param root: root directory where the dataset is stored
        :param batch_size: number of subgraphs to store in a single file
        :param transform: a function/transform applied to data objects
        :param pre_transform: a function applied before saving to disk
        :param pre_filter: a function to filter data objects
        '''
        self.batch_size = batch_size
        super().__init__(root, transform, pre_transform, pre_filter)

        # initialize encoders for node/edge attributes
        self.StrEncoder = encoder.SequenceEncoder()
        self.CateEncoder = encoder.CateEncoder()
        self.NumEncoder = encoder.IdentityEncoder(dtype=torch.float)


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
        ''' if the raw data needs to be downloaded, impplement it here
        
        '''
        raise NotImplementedError("Please provide the raw data manually.")


    def process(self):
        # load the raw data
        raw_path = osp.join(self.raw_dir, "subgraphs.pkl")
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
        ray.init()

        try:
            tasks = [
                process_subgraphs.remote(batch, self.seq_encoder, self.cate_encoder)
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

