'''
 # @ Create Time: 2025-01-09 11:26:56
 # @ Modified time: 2025-01-12 19:22:15
 # @ Description: implement distributed training process with multi-cpus
 '''


# predefined node types
node_types = ["Path", "Hostname", "Package_Name", "IP", "Hostnames", "Command", "Port"]

import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from model import mgan
from ext.iter_loader import IterSubGraphs
from torch_geometric.loader import DataLoader
from mgan import *
from datetime import datetime

class DistributedTrainer:
    def __init__(self, model_class, model_kwargs: dict, x_dict: dict, edge_index_dict: dict, \
                 y_dict:dict, num_epochs: int, world_size: int):
        ''' initialize the distributed training setup

        Args:
            model_class (type): the model class
            model_kwargs (dict): custom parameters for model's initialization
            x_dict: node features
            edge_index_dict: graph structure (edges)
            y_dict: labels
            world_size: the number of processes

        '''
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.x_dict = x_dict
        self.edge_index_dict = edge_index_dict
        self.y_dict = y_dict
        self.num_epochs = num_epochs
        self.world_size = world_size

    @staticmethod
    def setup(rank, world_size):
        # rank --- define the rank of current process (0, world_size - 1)
        # world_size --- number of processes participating in the job
        dist.init_process_group('gloo', rank=rank, world_size=world_size)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)
    
    @staticmethod
    def prepare_batch_inputs(model, batch, device, node_types = None):
        batch = batch.to(device)
        
        if isinstance(model, MaskedHeteroGAT):

            return {
                # assign data to device
                "x_dict": batch.x_dict,
                "edge_index_dict": batch.edge_index_dict,
                "edge_attr_dict": batch.edge_attr_dict,
                "batch_indices": batch.batch_dict,
                "node_types": node_types
            }

        elif isinstance(model, HeteroGAT):
            return {"batch": batch}
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    def train(self, rank):
        # setup
        self.setup(rank, self.world_size)

        # create model instance and move it to approprite device
        model = self.model_class(
            **self.model_kwargs
        ).to(rank)

        model = DDP(model, device_ids=[rank])
        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # wrap data in distributedsampler --- define an example
        sampler = DistributedSampler(list(self.x_dict.keys()), shuffle=True)

        dataloader = DataLoader(
                list(self.x_dict.keys()),
                batch_size = 1,
                sampler = sampler)
        
        for epoch in range(self.num_epochs):
            # ensure different shuffling each epoch
            sampler.set_epoch(epoch)

            for batch in dataloader:
                batch = batch[0]
                batch_inputs = self.prepare_batch_inputs(model, batch, device, node_types)

                optimizer.zero_grad()

                if isinstance(model, MaskedHeteroGAT):
                    # forward pass
                    logit_dict = model(**batch_inputs)
                    # compute loss
                    loss = model(probs, batch.y.to(device))
                elif isinstance(model, HeteroGAT):
                    logit_dict = model(batch_inputs["batch"])
                    loss = model.compute_loss(logit_dict, batch_inputs["batch"])                    
                else:
                    raise ValueError("Unknown model type")

                # backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # main process
                if rank == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}")

            # Optional: print final batch loss at end of epoch
            if rank == 0:
                print(f"[Epoch {epoch + 1}] Final Batch Loss: {loss.item()}")

        # save model only on rank 0
        if rank == 0 and isinstance(model, MaskedHeteroGAT):
            save_path = Path("dis_trained_maskhetermodel.pth")
   

        elif rank == 0 and isinstance(model, HeteroGAT):
            save_path = Path("dis_trained_hetermodel.pth")
        
        # model.module refers to the original model
        self.save_model(model.module, save_path)
        print(f"Model saved to {save_path.resolve()}")

        # cleanup
        self.cleanup()
        
        return model

    def dist_train(self):
        ''' spawan multiple processes for distributed training 
        
        '''
        mp.spawn(
            self.train,
            args=(),
            nprocs=self.world_size,
            join=True
        )
    
if __name__ == "__main__":

    # load data parameters
    ## load data batch to get the number of edges and nodes
    data_path = Path.cwd().parent.joinpath("ext", "test-small", "processed")
    
    # for test, change batch size from 10 to 1
    print("Creating iterative dataset")
    dataset = IterSubGraphs(root=data_path, batch_size = 1)
    
    ## load one .pt file at a time
    print("Creating subgraph dataloader")
    dataloader = DataLoader(
            dataset, 
            batch_size = 1, 
            shuffle=False, 
            num_workers=20,
            persistent_workers=True
            )
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    batch=next(iter(dataloader))

    print("Extracting parameters to configure graph data")
    ## create other input parameters
    x_dict = batch.x_dict
    edge_index_dict = batch.edge_index_dict
    y_dict = batch.y_dict

    ## create kwargs dict
    model1_kwargs = {
            "hidden_channels": 64,
            "out_channels": 64,
            "num_heads": 4,
            "num_clusters": 20,
            "num_edges": batch.num_edges,
            "num_nodes": batch.num_nodes
    }

    model2_kwargs = {
            "hidden_channels": 64,
            "out_channels": 64,
            "num_heads": 4,
    }


    # load model class
    model_class_1 = MaskedHeteroGAT
    model_class_2 = HeteroGAT

    num_epochs = 10
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    print("Training MaskedHeteroGAT ...")
    # create distributedtrainer instance
    trainer1 = DistributedTrainer(
        model_class=model_class_1,
        model_kwargs=model1_kwargs,
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        y_dict=y_dict,
        num_epochs=num_epochs,
        world_size=world_size
    )
    # define the starting time
    start_time = datetime.now()
    trainer1.dist_train()
    print(f"Time spent for MaskedHeteroGAT in distributed setting up is: {start_time - datetime.now()}")

    print("Training HeteroGAT ...")
    # create distributedtrainer instance
    trainer2 = DistributedTrainer(
        model_class=model_class_2,
        model_kwargs=model2_kwargs,
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        y_dict=y_dict,
        num_epochs=num_epochs,
        world_size=world_size
    )
    # define the starting time
    start_time = datetime.now()
    trainer2.dist_train()
    print(f"Time spent for HeteroGAT in distributed setting up is: {start_time - datetime.now()}")

    