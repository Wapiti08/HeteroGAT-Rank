'''
 # @ Create Time: 2025-01-09 11:26:56
 # @ Modified time: 2025-01-12 19:22:15
 # @ Description: implement distributed training process with multi-cpus
 '''

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataloader, DistributedSampler


class DistributedTrainer:
    def __init__(self, model_class, model_kwargs: dict, x_dict: dict, edge_index_dict: dict, y_dict:dict, 
                 num_epochs: int, world_size: int):
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

        # wrap data in distributedsampler
        sampler = DistributedSampler(list(self.x_dict.keys()), shuffle=True)
        dataloader = Dataloader(
                list(self.x_dict.keys()),
                batch_size = 1,
                sampler = sampler)
        
        for epoch in range(self.num_epochs):
            model.train()
            # ensure different shuffling each epoch
            sampler.set_epoch(epoch)

            for batch in dataloader:
                # assign data to device
                batch_x_dict = {key: self.x_dict[key].to(rank) for key in batch}
                batch_y_dict = {key: self.y_dict[key].to(rank) for key in batch}
                batch_edge_index_dict = {key: self.edge_index_dict[key].to(rank) for key in batch}

                optimizer.zero_grad()

                # forward pass
                logit_dict = model(batch_x_dict, batch_edge_index_dict)

                # compute loss
                loss = model.compute_loss(logit_dict, batch_y_dict)

                # backward pass and optimization
                loss.backward()
                optimizer.step()
            
            if rank == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}")

        # cleanup
        self.cleanup()

    def dist_train(self):
        ''' spawan multiple processes for distributed training 
        
        '''
        mp.spawn(
            self.train,
            args=(),
            nprocs=self.world_size,
            join=True
        )
