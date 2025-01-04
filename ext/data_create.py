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


    def download(self):
        ''' if the raw data needs to be downloaded, impplement it here
        
        '''
        raise NotImplementedError("Please provide the raw data manually.")


    def process(self):
        # load the raw data


        # process every subgraph within loop

            # construct node features

                # apply different encoding method 
        

                # concatenate node features

            # construct edge index and attributes
        

            # create the data object
        
            # apply filtering and transforms
        
            # add to batch
        
            # save the batch when full
        
        # save any remaining data
    

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        batch = torch.load(osp.join(self.processed_dir, f'batch_{idx}.pt'))
        return batch

