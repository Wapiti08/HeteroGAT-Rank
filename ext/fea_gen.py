'''
 # @ Create Time: 2024-12-16 18:27:17
 # @ Modified time: 2024-12-16 18:27:30
 # @ Description: create graph representation from labeled dataset
 
 node: {
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
import re
from tqdm import tqdm


ray.init()

@ray.remote
def _process_chunk(chunk: pd.DataFrame):
    """Helper function to process a single chunk of the DataFrame."""
    # define default data_path location
    data_path = Path.cwd().parent.joinpath('data', 'label_data.pkl')
    extractor = FeatureExtractor(data_path )  
    return extractor._node_edge_build(chunk)

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


    def _root_node(self, row: dict):
        '''
        :param row: the iterate row from dataframe
        :param id: pre-assign node id
        '''
        return {
            'value': row['Name'] + '_' + row['Version'],
            "type": 'Package_Name', 
            "eco": row['Ecosystem'],
        }

    def _file_nodes_edges(self, row: dict):
        ''' extract path and action entity from import/install_Files feature
        
        :return: list of nodes
        '''
        path_nodes, file_edges = [], []
        for feature in ['import_Files', 'install_Files']:
            if len(row[feature]) != 0:
                for entity_dict in row[feature]:
                    path_nodes.append(
                        {
                        'value': entity_dict['Path'],
                        'type': 'Path',
                        }
                    )

                    file_edges.append(
                        {
                        'source': row['Name'] + '_' + row['Version'] ,
                        'target': entity_dict['Path'],
                        'value': "_".join(key for key, value in entity_dict.items() if value is True),
                        'type': 'action',
                        
                        }
                    )

        return path_nodes, file_edges

    def _socket_nodes_edges(self, row: dict):
        ''' extract IP/Hostname/Port from import/install_Sockets
        
        '''
        nodes, edges = [], []
        # collect all values and filter out repeated values
        ipaddr_list, hostnames_list, port_list = [], [], []
        for feature in ['import_Sockets', 'install_Sockets']:
            if len(row[feature]) != 0:
                for entity_dict in row[feature]:
                    ipaddr_list.append(entity_dict['Address'])
                    if len(entity_dict['Hostnames'])!=0:
                        hostnames_list.extend(entity_dict['Hostnames'])
                    port_list.append(entity_dict['Port'])

        # filter out port with 0, address with '::1', blank hostname
        ipaddr_list = list(set(ipaddr_list))
        ipaddr_list.remove("::1")

        hostnames_list = list(set(hostnames_list))

        port_list = list(set(port_list))
        port_list.remove(0)

        
        # create nodes list spanning IP, hostname, port
        # create edge list from root node to three types of leaf nodes

        nodes.extend(
                [
                    {
                        'value': IP,
                        'type': 'Path',
                    } for IP in ipaddr_list
                ]
            )
    
        edges.extend(
            [
                {
                    'source': row['Name'] + '_' + row['Version'] ,
                    'target': IP,
                    'value': "",
                    'type': '',
                } for IP in ipaddr_list
            ]
        )

        nodes.extend(
            [
                {
                    'value': hostname,
                    'type': 'Hostnames'

                } for hostname in hostnames_list
            ]
        )

        edges.extend(
            [
                {
                    'source': row['Name'] + '_' + row['Version'] ,
                    'target': hostname,
                    'value': "",
                    'type': '',
                } for hostname in hostnames_list
            ]
        )

        nodes.extend(
            [
                {
                    'value': Port,
                    'type': 'Port'

                } for Port in port_list
            ]
        )

        edges.extend(
            [
                {
                    'source': row['Name'] + '_' + row['Version'] ,
                    'target': Port,
                    'value': "",
                    'type': '',
                } for Port in port_list
            ]
        )

        
        return nodes, edges


    def _cmd_node_edge(self, row: dict):
        ''' extract command (concatenated), user home directory, 
        contrainer, and term from import/install_Commands 
        '''
        nodes, edges = [], []
        for feature in ['import_Commands', 'install_Commands']:
            if len(row[feature]) != 0:
                for entity_dict in row[feature]:
                    # add command
                    concat_cmd = " ".join(cmd_string for cmd_string in entity_dict['Command'])
                    nodes.append(
                        {
                            'value': concat_cmd,
                            'type': "CMD",
                        }
                    )

                    edges.append(
                        {
                            'source': row['Name'] + '_' + row['Version'] ,
                            'target': concat_cmd,
                            'value': "",
                            'type': '',
                        }
                    )

        return nodes, edges

    def _node_edge_build(self, chunk: pd.DataFrame ):
        ''' extract nodes and  edges from different edges
        
        '''
        nodes = []
        edges = []
        for index, row in tqdm(chunk.iterrows(), desc='create edges from chunk', total=len(chunk)):
            nodes.append(self._root_node(row))
            file_nodes, file_edges = self._file_nodes_edges(row)
            nodes.extend(file_nodes)
            edges.extend(file_edges)

            socket_nodes, socket_edges = self._socket_nodes_edges(row)
            nodes.extend(socket_nodes)
            edges.extend(socket_edges)

            cmd_nodes, cmd_edges = self._cmd_node_edge(row)
            nodes.extend(cmd_nodes)
            edges.extend(cmd_edges)

        # remove repeated nodes and edges --- frozenset creates a hashable representation of dict
        nodes = list({frozenset(node.items()): node for node in nodes}.values())
        edges = list({frozenset(edge.items()): edge for edge in edges}.values())

        return nodes, edges
    
    def _parall_process(self):
        """
        Build knowledge graph in parallel using Ray.
        """
        # Split the dataframe into chunks
        chunks = self._split_chunks()

        # Create references to process each chunk in parallel
        futures = [_process_chunk.remote(chunk, self) for chunk in chunks]

        # Collect results from all chunks
        results = ray.get(futures)

        # Aggregate nodes and edges across all chunks
        all_nodes = []
        all_edges = []

        for nodes, edges in results:
            all_nodes.extend(nodes)
            all_edges.extend(edges)

        # Remove duplicate nodes and edges
        all_nodes = list({frozenset(node.items()): node for node in all_nodes}.values())
        all_edges = list({frozenset(edge.items()): edge for edge in all_edges}.values())

        return all_nodes, all_edges


if __name__ == "__main__":
    