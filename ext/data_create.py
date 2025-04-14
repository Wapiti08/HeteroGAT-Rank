'''
 # @ Create Time: 2024-12-18 09:27:11
 # @ Modified time: 2024-12-18 09:27:14
 # @ Description: create graph dataset suitable for GNN model training in pytorch in multiple steps (reduce one-time memory cost)
 
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
import hashlib

def generate_node_id(row):
    '''
    generate a unique ID for each node based on its attributes
    '''
    '''
    Generate a unique ID for each node based on its attributes
    '''
    # Ensure all values are strings and non-empty
    type_val = str(row.get('type', '')).strip()
    value_val = str(row.get('value', '')).strip()
    eco_val = str(row.get('eco', '')).strip()

    # If all attributes are empty, raise a warning or use a fallback ID
    if not type_val and not value_val and not eco_val:
        raise ValueError(f"Invalid node attributes: {row}")

    unique_str = f"{type_val}_{value_val}_{eco_val}"
    return hash(unique_str)


def prepare_encode_jobs(subgraph_batch):
    ''' generate encoding jobs for parallel prcoessing using Ray

    args:
        subgraph_batch (list): a list of subgraph dictionaries
            each dict contains a 'nodes' key with node metadata
    
    returns:
        tuple:
            - encode_jobs (list[tuple[pd.DataFrame, str]]): a list of encoding jobs
                str can be 'seq' or 'iden'
            - node_type_map (list[tuple[int, str, int]]): metadata mapping node DataFrame ID
                to node_type and the number of nodes of that type
            - node_dataframes (dict): a dict mapping node dataframe ID to the dataframe itself
    
    '''
    encode_jobs = []
    node_type_map = []
    # cache of node DataFrames keyed by their id()
    node_dataframes = {}

    for subgraph in subgraph_batch:
        # convert list of node dicts into a DataFrame, drop row with all NaNs
        nodes = pd.DataFrame(subgraph['nodes']).dropna()
        nodes['id'] = nodes.apply(generate_node_id, axis=1)
        nodes.set_index('id', inplace=True)
        node_dataframes[id(nodes)] = nodes

        # group by node type to handle different encoding strategies
        for node_type, group in nodes.groupby('type'):
            df = pd.DataFrame(group['value'].fillna(""), columns=['value'])
            job_type = 'iden' if node_type == 'Port' else 'seq'
            if job_type == 'iden':
                df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
            encode_jobs.append((df, job_type))
            # Track the mapping of nodes for later reconstruction or alignment
            node_type_map.append((id(nodes), node_type, len(group)))

    return encode_jobs, node_type_map, node_dataframes


def decode_encoded_features(encoded_batches, node_type_map):
    features = {}
    enc_idx = 0
    for graph_id, node_type, count in node_type_map:
        features.setdefault(graph_id, {})[node_type] = encoded_batches[enc_idx]
        enc_idx += 1
    return features

def construct_hetero_data(subgraph, nodes, edges, node_features, feature_dim):
    ''' construct a torch_geometric.data.HeteroData object from a subgraph
    
    args:
        subgraph (dict): Dictionary containing nodes, edges, and label for one subgraph.
        nodes (pd.DataFrame): DataFrame of nodes, indexed by unique ID.
        edges (pd.DataFrame): DataFrame of edges, containing source, target, and type columns.
        node_features (dict): Dictionary mapping node types to their encoded features (Tensors).
        feature_dim (int): The dimensionality of the encoded node features.

    return:
        HeteroData: a torch_geometric HeteroData object representing the graph
    '''
    hetero_data = HeteroData()

    # add node features for each node type
    for node_type, feats in node_features.items():
        # convert to half precision to reduce memory
        feats = feats.half()
        hetero_data[node_type].x = feats
        hetero_data[node_type].num_nodes = feats.size(0)
    
    # filter out edges with missing nodes
    valid_edges = edges['sources'].isin(nodes.index) & edges['target'].isin(nodes.index)
    edges = edges[valid_edges]

    # group edges by edge type
    for edge_type, type_edges in edges.groupby('type'):
        # map original source/target node IDs to their positions in the node index
        src_ids = type_edges['source'].apply(lambda x: nodes.index.get_loc(x)).to_numpy(dtype=np.int32)
        tgt_ids = type_edges['target'].apply(lambda x: nodes.index.get_loc(x)).to_numpy(dtype=np.int32)

        # create pytorch edge index tensor: shape [2, num_edges]
        edge_index = torch.tensor([src_ids, tgt_ids], dtype=torch.long)

        # initialize edge attributes as zero vector
        edge_attr = torch.zeros((len(type_edges), feature_dim), dtype=torch.half)

        # store edge info under the corresponding edge type
        hetero_data[edge_type].edge_index = edge_index
        hetero_data[edge_type].edge_attr = edge_attr
    
    # add graph-level label
    hetero_data['label'] = torch.tensor(subgraph['label'], dtype=torch.long)

    return hetero_data


def apply_padding(hetero_data, all_node_types, feature_dim, max_nodes_per_type, pad_transform):
    for node_type in all_node_types:
        if node_type not in hetero_data:
            hetero_data[node_type].x = torch.zeros((max_nodes_per_type[node_type], feature_dim), dtype=torch.half)
            hetero_data[node_type].num_nodes = max_nodes_per_type[node_type]
    return pad_transform(hetero_data)


@ray.remote
class EncoderActor:
    def __init__(self,):
        # Import here to avoid issues with Ray serialization
        from ext import fea_encoder
        self.seq_encoder = fea_encoder.SequenceEncoder()
        self.iden_encoder = fea_encoder.IdentityEncoder(dtype=torch.float, \
                                                        output_dim=self.seq_encoder.embedding_dim)

    def encode_batch(self, df_list):
        results = []
        for df, encoder_type in df_list:
            if encoder_type == "seq":
                results.append(self.seq_encoder(df))
            else:
                results.append(self.iden_encoder(df))
        
        return results

@ray.remote
def process_subgraphs(subgraph_batch: list, max_nodes_per_type: dict, max_edges_per_type: dict, \
                      encoder_actor: EncoderActor):
    '''  Processes a batch of subgraphs: encodes features, constructs heterogeneous graphs, and applies padding.

    args:
        subgraph_batch (list): list of subgraph dicts, each with "nodes" and "edges"
        max_nodes_per_type (dict): Maximum number of nodes per node type (used for padding).
        max_edges_per_type (dict): Maximum number of edges per edge type (used for padding).
        encoder_actor (EncoderActor): Ray actor that handles parallel feature encoding.

    return:
        torch_geometric.data.Batch: A batch of padded heterogeneous graph data objects.
    '''
    # Initialize the Pad transform with the calculated max values
    pad_transform = Pad(max_num_nodes=max_nodes_per_type, max_num_edges=max_edges_per_type)

    data_list = []
    all_node_types = set()
    feature_dim = None
    # prepare feature encoding jobs for parallel processing
    encode_jobs, node_type_map, node_dataframes = prepare_encode_jobs(subgraph_batch)
    # run encoding jobs using Ray and collect encoded results
    encoded_batches = ray.get(encoder_actor.encode_batch.remote(encode_jobs))
    # decode results into a dict
    node_features_dict = decode_encoded_features(encoded_batches, node_type_map)

    for i, subgraph in enumerate(subgraph_batch):
        nodes = pd.DataFrame(subgraph['nodes']).dropna()
        nodes['id'] = nodes.apply(generate_node_id, axis=1)
        nodes.set_index('id', inplace=True)
        edges = pd.DataFrame(subgraph['edges']).dropna(subset=['source', 'target'])

        # Use ID of nodes DataFrame to fetch corresponding encoded features
        graph_id = id(nodes)
        node_features = node_features_dict.get(graph_id, {})
        # Determine feature dimension from the first non-empty feature dict
        if node_features and feature_dim is None:
            feature_dim = next(iter(node_features.values())).size(1)

        try:
            # Build a PyG HeteroData object from raw data + encoded features
            hetero_data = construct_hetero_data(subgraph, nodes, edges, node_features, feature_dim)
            all_node_types.update(nodes['type'].unique())
            # Pad the HeteroData object to max size per type
            hetero_data = apply_padding(hetero_data, all_node_types, feature_dim, max_nodes_per_type, pad_transform)
            data_list.append(hetero_data)
        except Exception as e:
            print(f"Error processing subgraph {i}: {e}")
            continue

    return Batch.from_data_list(data_list)


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
                for node_type, type_nodes in node_df.groupby('type'):
                    # Update max nodes per type
                    max_nodes_per_type[node_type] = max(max_nodes_per_type.get(node_type, 0), len(type_nodes))

            if "type" in edge_df.columns:
                # Update max edges per type
                for edge_type, type_edges in edge_df.groupby('type'):
                    for src, tgt in zip(type_edges['source'], type_edges['target']):
                        edge_tuple = (type_edges['source'].iloc[0], edge_type, type_edges['target'].iloc[0])
                        max_edges_per_type[edge_tuple] = max(max_edges_per_type.get(edge_tuple, 0), len(type_edges))

        return max_nodes_per_type, max_edges_per_type


    def process(self,):
        # load the raw data --- non-packaged directory to avoid large size package to Ray
        raw_path = osp.join(self.data_path.parent.parent.joinpath("data").as_posix(),\
                             "subgraphs.pkl")
        
        with open(raw_path, 'rb') as fr:
            subgraphs = pickle.load(fr)

        max_nodes_per_type, max_edges_per_type = self.pad_size(subgraphs)

        # convert to pandas framework for easier processing
        subgraphs_df = pd.DataFrame(subgraphs)

        # split into batches
        num_batches = len(subgraphs_df) // self.batch_size + 1
        subgraph_batches = (subgraphs[i * self.batch_size: (i+1) * self.batch_size] for i in range(num_batches))

        # Set OOM mitigation variables before initializing Ray
        os.environ["RAY_memory_usage_threshold"] = "0.9"  # Adjust based on node capacity
        os.environ["RAY_memory_monitor_refresh_ms"] = "0" 

        # process batches in parallel
        ray.init(runtime_env={"working_dir": Path.cwd().parent.as_posix(), \
                              "excludes": ["logs/", "*.pt", "*.json", "*.csv", "*.pkl"]})

        try:
            # create a single encoder actor
            encoder_actor = EncoderActor.remote()

            # check available resources
            available_resources = ray.available_resources()
            # use 80% of the cpus   
            max_parallel = int(available_resources.get("CPU", 4) * 0.8)

            @ray.remote(num_cpus=2)  # Adjust based on workload
            def process_subgraphs_wrapper(batch, max_nodes_per_type, max_edges_per_type, actor):
                return process_subgraphs.remote(batch, max_nodes_per_type, max_edges_per_type, actor)

            # process in rounds using ray.wait()
            tasks = []
            results = []

            for batch in subgraph_batches:
                if len(tasks) >= max_parallel:
                    # wait for at least one task to complete before submitting a new one
                    done, tasks = ray.wait(tasks, num_returns = 1)
                    results.append(ray.get(done[0]))
                
                # submit a new task
                tasks.append(process_subgraphs_wrapper.remote(batch, max_nodes_per_type, \
                                                              max_edges_per_type, encoder_actor))

            # collect remaining results
            for task in tasks:
                results.append(ray.get(task))

        finally:
            # free memory after execution
            del subgraph_batches
            ray.shutdown()

        # save the processed batches
        for i, batch in enumerate(results):
            # make sure there are no objectRefs in batch
            for node_type in batch.node_types:
                assert isinstance(batch[node_type].x, torch.Tensor), f"{node_type} has ObjectRef!"
            for edge_type in batch.edge_types:
                assert isinstance(batch[edge_type].edge_index, torch.Tensor)
                if hasattr(batch[edge_type], 'edge_attr'):
                    assert isinstance(batch[edge_type].edge_attr, torch.Tensor)
            # save processed batches in float16 instead of float32
            torch.save(batch, osp.join(self.processed_dir, f'batch_{i}.pt'))
        
        del results

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
        print(f"loaded batch {batch_idx} with {len(ray.get(batch_data))} subgraphs")
    else:
        print("Batch index out of range")

    
    

