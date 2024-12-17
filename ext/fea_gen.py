'''
 # @ Create Time: 2024-12-16 18:27:17
 # @ Modified time: 2024-12-16 18:27:30
 # @ Description: create graph representation from labeled dataset
 
 node: {
    "id": n,
    'value': str value,
    "type": Path | Package_Name | IP | Hostname | Port | 
        Command | User Home Directory | Container | TERM |
    "eco": 
 }

 edge: {
    "source": x,
    "target": y,
    "value": str | list value,
    "type": action | DNS types
 }
 
 
 '''
import pandas as pd
from pathlib import Path
import ray
import numpy as np
from collections import defaultdict


ray.init()

class FeatureExtractor:
    
    def __init__(self, data_path: Path):
        self.df = pd.read_pickle(data_path)

    def _split_chunks(self,):
        # Approximate memory per row (1 KB per row)
        rows_per_chunk = 500 * 1024 * 1024 // 1024

        # Split into chunks based on rows and cores
        num_chunks = min(64, len(self.df) // rows_per_chunk)
        chunks = np.array_split(self.df, num_chunks)

        return chunks

    def _node_build(self, chunk:pd.DataFrame):
        ''' extract node with node attributes from features
        
        ''' 


    def _root_node(self, row: dict, id: int):
        '''
        :param row: the iterate row from dataframe
        :param id: pre-assign node id
        '''
        return defaultdict{
            "id": id,
            'value': row['Name'] + '_' + row['Version'],
            "type": 'Package_Name', 
            "eco": row['Ecosystem'],
        }

    def _file_node(self, row: dict):
        



    def _socket_node(self,):


    def _cmd_node(self,):

    
    def _file_edge(self,):


    def _edge_build(self, ):
        ''' extract edge from features
        
        '''
        
    
    def _parall_process(self,):
