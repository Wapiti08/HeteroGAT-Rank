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
from typing import List, Dict, Tuple
import pickle

class FeatureExtractor:
    
    def __init__(self, data_path: Path):
        self.df = pd.read_pickle(data_path)
        

    def _split_chunks(self, default_split = 5) -> List[pd.DataFrame]:
        # Approximate memory per row (1 KB per row)
        rows_per_chunk = 500 * 1024 * 1024 // 1024

        # Calculate the number of chunks based on rows and cores
        num_chunks = min(64, max(default_split, len(self.df) // rows_per_chunk))
        # Ensure at least one chunk
        print(f"Number of chunks: {num_chunks}")

        # Split the DataFrame into chunks
        chunks = np.array_split(self.df, num_chunks)
        print(f"Number of rows in each chunk: {[len(chunk) for chunk in chunks]}")
        return chunks

    @staticmethod
    def _create_node(value: str, node_type: str, eco: str) -> Dict:
        '''  Helper function to create a node '''
        return {"value": value, "type": node_type, "eco": eco}

    @staticmethod
    def _create_edge(source: str, target: str, edge_type:str = "", value:str="") -> Dict:
        ''' Helper function to create a node '''
        return {"source": source, "target": target, "type": edge_type, "value":value}


    def _root_node(self, row: dict) -> Dict:
        '''
        :param row: the iterate row from dataframe
        :param id: pre-assign node id
        '''
        return self._create_node(value=f"{row['Name']}_{row['Version']}",
                                 node_type="Package_Name",
                                 eco=f"{row['Ecosystem']}")

    def _file_nodes_edges(self, row: dict) -> Tuple[List[Dict], List[Dict]]:
        ''' extract path and action entity from import/install_Files feature
        
        :return: list of nodes
        '''
        path_nodes, file_edges = [], []
        for feature in ['import_Files', 'install_Files']:
            entities = row.get(feature)
            if isinstance(entities, (list, np.ndarray)) and len(entities) > 0:
                for entity in entities:
                    path_node = self._create_node(value=entity["Path"], node_type="Path", eco=row["Ecosystem"])
                    file_edge = self._create_edge(
                        source=f"{row['Name']}_{row['Version']}",
                        target=entity["Path"],
                        edge_type="action",
                        value="_".join(k for k, v in entity.items() if v is True),
                    )

                    path_nodes.append(path_node)
                    file_edges.append(file_edge)

        return path_nodes, file_edges

    def _socket_nodes_edges(self, row: dict) -> Tuple[List[Dict], List[Dict]]:
        ''' extract IP/Hostname/Port from import/install_Sockets
        
        '''
        nodes, edges = [], []
        # collect all values and filter out repeated values
        ip_list, host_list, port_list = set(), set(), set()
        for feature in ['import_Sockets', 'install_Sockets']:
            entities = row.get(feature)
            if isinstance(entities, (list, np.ndarray)) and len(entities) > 0:
                for entity in entities:
                    ip_list.add(entity.get("Address"))
                    host_list.update(entity.get("Hostnames", []))
                    port_list.add(entity.get("Port"))

        # filter out port with 0, address with '::1', blank hostname
        ip_list.discard("::1")
        port_list.discard(0)
        
        nodes.extend([self._create_node(ip, "IP", row["Ecosystem"]) for ip in ip_list])
        edges.extend([self._create_edge(f"{row['Name']}_{row['Version']}", ip) for ip in ip_list])

        nodes.extend([self._create_node(host, "Hostname", row["Ecosystem"]) for host in host_list])
        edges.extend([self._create_edge(f"{row['Name']}_{row['Version']}", host) for host in host_list])

        nodes.extend([self._create_node(str(port), "Port", row["Ecosystem"]) for port in port_list])
        edges.extend([self._create_edge(f"{row['Name']}_{row['Version']}", str(port)) for port in port_list])
        
        return nodes, edges


    def _cmd_node_edge(self, row: dict) -> Tuple[List[Dict], List[Dict]]:
        ''' extract command (concatenated), user home directory, 
        contrainer, and term from import/install_Commands 
        '''
        nodes, edges = [], []
        for feature in ['import_Commands', 'install_Commands']:
            entities = row.get(feature)
            if isinstance(entities, (list, np.ndarray)) and len(entities) > 0:
                for entity in entities:
                    cmd = " ".join(entity.get("Command", []))
                    nodes.append(self._create_node(cmd, "CMD", row["Ecosystem"]))
                    edges.append(self._create_edge(f"{row['Name']}_{row['Version']}", cmd))
        return nodes, edges

    @staticmethod
    def _remove_duplicates(items: List[Dict]) -> List[Dict]:
        """Remove duplicate dictionaries from a list."""
        return list({frozenset(item.items()): item for item in items}.values())

    def _construct_subgraph(self, label:str, rows: pd.DataFrame ) -> Dict:
        ''' extract nodes and  edges from different edges
        
        '''
        nodes = []
        edges = []
        for index, row in tqdm(rows.iterrows(), desc='create edges from chunk', total=len(rows)):
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

        nodes, edges = self._remove_duplicates(nodes), self._remove_duplicates(edges)

        return {'label': label, "nodes": nodes, "edges": edges}

    def _group_by_label(self) -> Dict[str, pd.DataFrame]:
        return {label: group for label, group in self.df.groupby("Label")}

    def _parall_process(self) -> List[Dict]:
        """
        Build knowledge graph in parallel using Ray.
        """
        label_groups = self._group_by_label()

        # Create references to process each chunk in parallel
        futures = [_process_label.remote(label, rows) for label, rows in label_groups.items()]

        # Collect results from all chunks
        results = ray.get(futures)

        return results

@ray.remote
def _process_label(label: str, rows: pd.DataFrame) -> Dict:
    """Helper function to process a single chunk of the DataFrame."""
    # define default data_path location
    data_path = Path.cwd().parent.joinpath('data', 'label_data.pkl')
    extractor = FeatureExtractor(data_path)  
    return extractor._construct_subgraph(label, rows)


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Define the path to the data file
    data_path = Path.cwd().parent.joinpath('data', 'label_data.pkl')

    # Check if the data file exists
    if not data_path.exists():
        print(f"Data file not found at: {data_path}")
        return

    # Create an instance of FeatureExtractor
    feature_extractor = FeatureExtractor(data_path)

    # Test the parallel processing method
    print("Starting parallel labeled subgraph construction...")
    subgraphs = feature_extractor._parall_process()

    # Display a summary of the results
    print(f"Total subgraphs constructed: {len(subgraphs)}")
    for subgraph in subgraphs:
        print(f"Label: {subgraph['label']}, Nodes: {len(subgraph['nodes'])}, Edges: {len(subgraph['edges'])}")


    # Optional: Save the results to files for inspection
    output_dir = Path.cwd().joinpath("output")
    output_dir.mkdir(exist_ok=True)

    # Save all subgraphs to a single pickle file
    pickle_file_path = output_dir.joinpath("subgraphs.pkl")

    with open(pickle_file_path, "wb") as f:
        pickle.dump(subgraphs, f)

    print(f"All subgraphs saved to: {pickle_file_path}")

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()

