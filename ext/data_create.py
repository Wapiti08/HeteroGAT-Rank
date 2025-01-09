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

import ray
from ext import encoder
import os
import os.path as osp
import torch
from torch_geometric.data import HeteroData, Dataset, Batch
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

        # initialize heterodata
        hetero_data = HeteroData()

        # process nodes
        for node_type in node_df['type'].unique():
            type_nodes = node_df[node_df['type']==node_type]
            if node_type == 'Port':
                features = iden_encoder(type_nodes[['value']])
            else:
                features = seq_encoder(type_nodes[['value']])
            hetero_data['node'].x = features
            # comment it for second-stage exploration
            # hetero_data['node'].eco = cate_encoder(type_nodes[['eco']])

        # process edges
        for edge_type in edge_df['type'].unique():
            type_edges = edge_df[edge_df['type'] == edge_type]
            sources = seq_encoder(type_edges[['source']])
            targets = seq_encoder(type_edges[['target']])
            edge_values = seq_encoder(type_edges[['value']])
            hetero_data[edge_type].edge_index = torch.tensor(
                [sources, targets], dtype=torch.long
            )
            hetero_data[edge_type].edge_attr = edge_values

        # add labels
        hetero_data['label'] = torch.tensor(label, dtype=torch.long)

        data_list.append(hetero_data)
    
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

