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
from tqdm import tqdm
import psutil

# # Ray is properly initialized before any remote functions are called
if ray.is_initialized():
    ray.shutdown()

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

    for i, subgraph in enumerate(subgraph_batch):
        # convert list of node dicts into a DataFrame, drop row with all NaNs
        nodes = pd.DataFrame(subgraph['nodes']).dropna()
        nodes['id'] = nodes.apply(generate_node_id, axis=1)
        nodes.set_index('id', inplace=True)
        node_dataframes[i] = nodes

        # group by node type to handle different encoding strategies
        for node_type, group in nodes.groupby('type'):
            df = pd.DataFrame(group['value'].fillna(""), columns=['value'])
            job_type = 'iden' if node_type == 'Port' else 'seq'
            if job_type == 'iden':
                df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
            encode_jobs.append((df, job_type))
            # Track the mapping of nodes for later reconstruction or alignment
            node_type_map.append((i, node_type, len(group)))

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
    valid_edges = edges['source'].isin(nodes.index) & edges['target'].isin(nodes.index)
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

def load_in_chunks(file_path, chunk_size):
    with open(file_path, 'rb') as fr:
        data = pickle.load(fr)
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]



def pad_size(subgraphs):
    """Compute max nodes and edges per type in a batch of subgraphs."""
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

def compute_global_pad_size(file_path, chunk_size):
    global_max_nodes = {}
    global_max_edges = {}

    for chunk in load_in_chunks(file_path, chunk_size):
        local_nodes, local_edges = pad_size(chunk)

        for k, v in local_nodes.items():
            global_max_nodes[k] = max(global_max_nodes.get(k, 0), v)

        for k, v in local_edges.items():
            global_max_edges[k] = max(global_max_edges.get(k, 0), v)

    return global_max_nodes, global_max_edges


@ray.remote
class EncoderActor:
    def __init__(self,):
        # Import here to avoid issues with Ray serialization
        from ext import fea_encoder
        self.seq_encoder = fea_encoder.SequenceEncoder()
        self.iden_encoder = fea_encoder.IdentityEncoder(dtype=torch.float, \
                                                        output_dim=self.seq_encoder.embedding_dim)
        self._is_shutdown = False

    def encode_batch(self, df_list):
        try:
            print("=== Entered encode_batch ===")
            results = []
            for idx, (df, encoder_type) in enumerate(df_list):
                print(f"  [Job {idx}] Type: {encoder_type}, Shape: {df.shape}")
                if encoder_type == "seq":
                    encoded = self.seq_encoder(df)
                else:
                    encoded = self.iden_encoder(df)
                print(f"  [Job {idx}] Encoded shape: {encoded.shape}")
                results.append(encoded)
            return results
        except Exception as e:
            print("!!! Exception inside encode_batch:", e)
            import traceback
            traceback.print_exc()
            return []

    def shutdown(self):
        # Prevent shutdown if it's already done
        if not self._is_shutdown:
            print("Shutting down encoder actor...")
            self._is_shutdown = True
            ray.actor.exit_actor()
        else:
            print("Encoder actor is already shut down.")

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
    print(f"Inside process_subgraphs with batch size: {len(subgraph_batch)}")

    # Initialize the Pad transform with the calculated max values
    pad_transform = Pad(max_num_nodes=max_nodes_per_type, max_num_edges=max_edges_per_type)

    data_list = []
    all_node_types = set()
    feature_dim = None
    # prepare feature encoding jobs for parallel processing
    encode_jobs, node_type_map, node_dataframes = prepare_encode_jobs(subgraph_batch)
    # run encoding jobs using Ray and collect encoded results
    encoded_batches = ray.get(encoder_actor.encode_batch.remote(encode_jobs))
    #!! make sure actor is shutdown properly after calling them
    # ray.get(encoder_actor.shutdown.remote())
    # decode results into a dict
    node_features_dict = decode_encoded_features(encoded_batches, node_type_map)

    for i, subgraph in enumerate(subgraph_batch):
        nodes = pd.DataFrame(subgraph['nodes']).dropna()
        nodes['id'] = nodes.apply(generate_node_id, axis=1)
        nodes.set_index('id', inplace=True)
        edges = pd.DataFrame(subgraph['edges']).dropna(subset=['source', 'target'])

        # Use ID of nodes DataFrame to fetch corresponding encoded features
        node_features = node_features_dict.get(i, {})

        # Determine feature dimension from the first non-empty feature dict
        if not node_features:
            print(f"[Subgraph {i}] Skipping: no features found.")
            continue  # Skip this subgraph
        
        if feature_dim is None:
            first_tensor = next(iter(node_features.values()), None)
            if first_tensor is not None:
                feature_dim = first_tensor.size(1)
                print(f"[Subgraph {i}] feature_dim set to: {feature_dim}")
            else:
                print(f"[Subgraph {i}] No valid feature tensor found. Skipping.")
                continue

        try:
            # Build a PyG HeteroData object from raw data + encoded features
            hetero_data = construct_hetero_data(subgraph, nodes, edges, node_features, feature_dim)
            all_node_types.update(nodes['type'].unique())
            # Pad the HeteroData object to max size per type
            hetero_data = apply_padding(hetero_data, all_node_types, feature_dim, max_nodes_per_type, pad_transform)
            data_list.append(hetero_data)
        except Exception as e:
            print(f"[Subgraph {i}] Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
    if not data_list:
        raise ValueError("No valid subgraphs were processed — `data_list` is empty.")
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
        self.batch_size = self.get_adaptive_batch_size(batch_size)
        self.data_path = root
        super().__init__(root, transform, pre_transform, pre_filter)

    @staticmethod
    def get_adaptive_batch_size(default_bs):
        total_mem = psutil.virtual_memory().total/(1024 * 3)
        if total_mem < 16:
            return max(1, default_bs/2)
        return default_bs

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

    @staticmethod
    def get_max_parallel_tasks(task_cpus = 4, utilization_ratio=0.8):
        # there are 32 total available cpus
        available = ray.available_resources().get("CPU", 32)  
        usable = int(available * utilization_ratio)
        return max(1, usable // task_cpus)

    def process(self,):
        # load the raw data --- non-packaged directory to avoid large size package to Ray
        raw_path = osp.join(self.data_path.parent.parent.joinpath("data").as_posix(),\
                             "subgraphs.pkl")
        
        max_nodes_per_type, max_edges_per_type = compute_global_pad_size(raw_path, self.batch_size)

        ray.shutdown()
        # Set OOM mitigation variables before initializing Ray
        os.environ["RAY_memory_usage_threshold"] = "0.9"  # Adjust based on node capacity
        os.environ["RAY_memory_monitor_refresh_ms"] = "0" 
        # process batches in parallel
        ray.init(runtime_env={"working_dir": Path.cwd().parent.as_posix(), \
                              "excludes": ["logs/", "*.pt", "*.json", "*.csv", "*.pkl"]})

        # create a single encoder actor
        encoder_actor = EncoderActor.remote()

        max_parallel = self.get_max_parallel_tasks()

        @ray.remote(num_cpus=4)  # Adjust based on workload
        def process_subgraphs_wrapper(batch, max_nodes_per_type, max_edges_per_type, actor):
            return process_subgraphs.remote(batch, max_nodes_per_type, max_edges_per_type, actor)
        
        tasks = []
        results = []
        chunk_id = 0

        for batch in load_in_chunks(raw_path, self.batch_size):
            # Always submit the task
            obj_ref = process_subgraphs_wrapper.remote(batch, max_nodes_per_type, max_edges_per_type, encoder_actor)
            tasks.append(obj_ref)
            # If tasks reach max parallelism, wait for one to finish
            if len(tasks) >= max_parallel:
                done, tasks = ray.wait(tasks, num_returns=1)
                first_obj_ref = ray.get(done[0])     # This gives an ObjectRef returned by wrapper
                final_result = ray.get(first_obj_ref)
                results.append((chunk_id, final_result))
                chunk_id += 1

        # Process remaining tasks
        while tasks:
            done, tasks = ray.wait(tasks, num_returns=1)
            first_obj_ref = ray.get(done[0])     # This gives an ObjectRef returned by wrapper
            final_result = ray.get(first_obj_ref)
            results.append((chunk_id, final_result))
            chunk_id += 1 

        ray.shutdown()

        for i, batch in results:
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
        if isinstance(batch_data, ray.ObjectRef):
            print(f"loaded batch {batch_idx} with {len(ray.get(batch_data))} subgraphs")
        else:
            print(f"loaded batch {batch_idx} with {len(batch_data)} subgraphs")
    else:
        print("Batch index out of range")

    
    

