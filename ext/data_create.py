'''
 # @ Create Time: 2024-12-18 09:27:11
 # @ Modified time: 2024-12-18 09:27:14
 # @ Description: create graph dataset suitable for GNN model training in pytorch
 
  node: {
    'value': str content,
    "type": Path | Package_Name | IP | Hostname | Command | --- str content
            Port --- numeric
    "eco": cate
 }

 edge: {
    "source": str content,
    "target": str content,
    "value": str | list value,
    "type": action | DNS types
 }
 
 
 
 '''

import ray
from ext import encoder
import os
import os.path as osp
import torch
from torch_geometric.data import Data, Dataset
import pickle

class LabeledSubGraphs(Dataset):
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        '''
        :param root: root directory where the dataset is stored
        :param transform: a function/transform applied to data objects
        :param pre_transform: a function applied before saving to disk
        :param pre_filter: a function to filter data objects
        '''
        super().__init__(root, transform, pre_transform, pre_filter)

        # initialize encoders for node/edge attributes

        # fit encoders with the possible categories (set before processing)

    @property
    def raw_file_names(self):

        return ['subgraphs.pkl']
    
    @property
    def processed_file_names(self):
        # generate a list of filenames for processed files
    
    def download(self):
        ''' if the raw data needs to be downloaded, impplement it here
        
        '''

    
    def process(self):
        # load the raw data


        # process every subgraph within loop

            # construct node features

                # 

            # construct edge index and attributes
        

