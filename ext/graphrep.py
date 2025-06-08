'''
 # @ Create Time: 2024-12-16 18:27:17
 # @ Modified time: 2024-12-16 18:27:30
 # @ Description: create graph representation from labeled dataset (csv -> pkl)
 
 node: {
    'value': str value / numeric
    "type": Package_Name | Path | IP | DNS Host | Hostnames | Command | Port --- str content
    "eco": cate
     }

 edge: {
    "source": x,
    "target": y,
    "value": str,
    "type": Action (Path)| DNS (DNS Host) | CMD (Command) | Socket (IP, Port, Hostnames)

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


def process_string(input_string, max_len=500):
    # step1: split the string by spaces
    parts = input_string.split()

    # step2: remove repeated values -- keeps the order while removing duplicates
    unique_parts = list(dict.fromkeys(parts))

    # step3: keep only the last two layers of each path and remove hash-like parts
    processed_parts = []
    for part in unique_parts:
        # remove hash-like parts
        if re.search(r'[a-fA-F0-9\-]{5,}', part):
            continue

        # if the part looks like a path
        if "/" in part:
            path_parts = part.split("/")
            # keep the last two layers
            processed_part = '/'.join(path_parts[-2:])
            processed_parts.append(processed_part)
        else:
            processed_parts.append(part)
    
    # step4: tjoin the processed parts into a string
    result_string = ''.join(processed_parts)

    # step5: trim the string to the max length if necessary
    if len(result_string) > max_len:
        result_string = result_string[:max_len]

    return result_string

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

    def _dns_nodes_edges(self, row: dict) -> Tuple[List[Dict], List[Dict]]:
        ''' extract hostname as nodes and concatenated types as edges
        
        '''
        hostname_nodes, type_edges = [], []
        for feature in ['import_DNS', 'install_DNS']:
            entities = row.get(feature)
            # check whether entities are not blank
            if isinstance(entities, (list, np.ndarray)) and len(entities) > 0:
                for entity in entities:
                    queries = entity.get("Queries", [])

                    # ensure queries is iterable and contains valid data
                    if isinstance(queries, (list, np.ndarray)) and len(queries) >0:
                        for query in queries:
                            hostname = query.get("Hostname")
                            types = query.get("Types", [])
                        
                            if hostname:
                                # Create and add unique node
                                if hostname not in [node["value"] for node in hostname_nodes]:
                                    hostname_nodes.append(self._create_node(hostname, "DNS Host", row["Ecosystem"]))
                                
                                # Create and add edge with concatenated DNS types
                                if isinstance(types, (list, np.ndarray)) and len(types) > 0:
                                    concatenated_types = "_".join(types)
                                    type_edges.append(self._create_edge(
                                        source=f"{row['Name']}_{row['Version']}",
                                        target=hostname, 
                                        edge_type="DNS",
                                        value=concatenated_types
                                    ))

        return hostname_nodes, type_edges

    def _file_nodes_edges(self, row: dict) -> Tuple[List[Dict], List[Dict]]:
        ''' extract path and action entity from import/install_Files feature
        
        :return: list of nodes
        '''
        path_nodes, file_edges = [], []
        seen_paths = set()

        for feature in ['import_Files', 'install_Files']:
            entities = row.get(feature)
            if isinstance(entities, (list, np.ndarray)) and len(entities) > 0:
                for entity in entities:
                    raw_path = entity["Path"]
                    path_value = process_string(raw_path)
                    if path_value not in seen_paths:
                        seen_paths.add(path_value)
                        path_node = self._create_node(value=path_value, node_type="Path", eco=row["Ecosystem"])
                        path_nodes.append(path_node)

                    file_edge = self._create_edge(
                        source=f"{row['Name']}_{row['Version']}",
                        target=path_value,
                        edge_type="Action",
                        value="_".join(k for k, v in entity.items() if v is True),
                    )

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

        default_edge_type = "socket"
        default_edge_value = "access"

        # filter out port with 0, address with '::1', blank hostname
        ip_list.discard("::1")
        port_list.discard(0)
        
        nodes.extend([self._create_node(ip, "IP", row["Ecosystem"]) for ip in ip_list])
        edges.extend([self._create_edge(f"{row['Name']}_{row['Version']}", ip, \
                                        default_edge_type+"_ip", default_edge_value) for ip in ip_list])

        nodes.extend([self._create_node(host, "Hostnames", row["Ecosystem"]) for host in host_list])
        edges.extend([self._create_edge(f"{row['Name']}_{row['Version']}", host, \
                                        default_edge_type+"_host", default_edge_value) for host in host_list])

        nodes.extend([self._create_node(str(port), "Port", row["Ecosystem"]) for port in port_list])
        edges.extend([self._create_edge(f"{row['Name']}_{row['Version']}", port, \
                                        default_edge_type+"_port", default_edge_value) for port in port_list])
        
        return nodes, edges


    def _cmd_node_edge(self, row: dict) -> Tuple[List[Dict], List[Dict]]:
        ''' extract command (concatenated), user home directory, 
        contrainer, and term from import/install_Commands 
        '''
        default_edge_type = "CMD"
        default_edge_value = "execute"

        nodes, edges = [], []
        for feature in ['import_Commands', 'install_Commands']:
            entities = row.get(feature)
            if isinstance(entities, (list, np.ndarray)) and len(entities) > 0:
                for entity in entities:
                    cmd = " ".join(entity.get("Command", []))
                    # node type is Command not CMD
                    nodes.append(self._create_node(cmd, "Command", row["Ecosystem"]))
                    edges.append(self._create_edge(f"{row['Name']}_{row['Version']}", cmd, \
                                                   default_edge_type, default_edge_value))
        return nodes, edges


    @staticmethod
    def _remove_duplicates(items: List[Dict]) -> List[Dict]:
        """Remove duplicate dictionaries from a list."""
        return list({frozenset(item.items()): item for item in items}.values())


    def _construct_subgraph(self, row: dict ) -> Dict:
        ''' extract nodes and  edges from different edges
        
        '''
        nodes = []
        edges = []

        nodes.append(self._root_node(row))

        dns_nodes, dns_edges = self._dns_nodes_edges(row)
        nodes.extend(dns_nodes)
        edges.extend(dns_edges)

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

        return {'label': row.get("Label"), "nodes": nodes, "edges": edges}


    def _parall_process_rows(self) -> List[Dict]:
        """
        Build knowledge graph in parallel using Ray.
        """
        chunks = self._split_chunks()
        # Create references to process each chunk in parallel
        futures = [_process_chunk.remote(chunk) for chunk in chunks]

        # Collect results from all chunks
        chunk_results = ray.get(futures)
        # Flatten the list of lists into a single list
        results = [subgraph for chunk in chunk_results for subgraph in chunk]

        return results

@ray.remote
def _process_chunk(chunk: pd.DataFrame) -> List[Dict]:
    """Helper function to process a single chunk of the DataFrame."""
    # define default data_path location
    data_path = Path.cwd().parent.joinpath('data', 'label_data.pkl')
    extractor = FeatureExtractor(data_path)  
    return [extractor._construct_subgraph(row) for _, row in chunk.iterrows()]


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
    subgraphs = feature_extractor._parall_process_rows()

    # Display a summary of the results
    print(f"Total subgraphs constructed: {len(subgraphs)}")
    for i, subgraph in enumerate(subgraphs[:5]):  # Show first 5 for brevity
        print(f"Subgraph {i}: Label: {subgraph['label']}, Nodes: {len(subgraph['nodes'])}, Edges: {len(subgraph['edges'])}")


    # Optional: Save the results to files for inspection
    output_dir = Path.cwd().joinpath("output","raw")
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

