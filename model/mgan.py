'''
 # @ Create Time: 2025-01-03 16:05:44
 # @ Modified time: 2025-01-03 16:06:18
 # @ Description: create attention-based Graph Neural Networks to learn feature importance
 '''

import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from ext.iter_loader import IterSubGraphs
from torch_geometric.loader import DataLoader
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from utils import evals
from model.hetergat import HeterGAT
from model.mhetergat import MaskedHeteroGAT
from torch_geometric.utils import to_dense_batch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# predefined node types
node_types = ["Path", "DNS Host", "Package_Name", "IP", "Hostnames", "Command", "Port"]

if __name__ == "__main__":

    data_path = Path.cwd().parent.joinpath("ext", "test-small", "processed")
    print("Creating iterative dataset")
    # return a batch of 10 subgraphs based on saved format
    dataset = IterSubGraphs(root=data_path, batch_size = 1)

    # load one .pt file at a time
    print("Creating subgraph dataloader")
    num_epochs = 10

    # split into train/test
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=32)

    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        pin_memory=False,
        prefetch_factor=None
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=True,
        pin_memory=False,
        prefetch_factor=None
    )


    model2 = HeterGAT(
        hidden_channels=64,
        out_channels=256,
        num_heads=4,
        # based on pre-set dimension
        edge_attr_dim=16
    ).to(device)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)

    print("Training HeteroGAT ...")

    conv_weight_dict_2 = {}
    # define the starting time
    start_time = datetime.now()
    for epoch in range(num_epochs):
        model2.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(next(model2.parameters()).device)  # Move batch to the same device as model
            optimizer2.zero_grad()

            # forward pass
            logits, conv_weight_dict_2 = model2.forward(batch)
            
            # compute loss
            loss = model2.compute_loss(logits, batch)
            # backward pass and optimization
            loss.backward()
            optimizer2.step()

            total_loss += loss.item()
        
        # rank last conv_weight_dict
        model2.rank_edges()

        avg_loss = total_loss/len(train_loader)

        # ----- EVALUATION -----
        model2.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(next(model2.parameters()).device)
                logits, _ = model2(batch)
                all_logits.append(logits)
                all_labels.append(batch['label'])

        # Concatenate
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        metrics = model2.evaluate(all_logits, all_labels)

        model2.plot_metrics(
            all_labels,
            torch.sigmoid(all_logits).cpu().numpy(),
            metrics)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    time_spent = datetime.now() - start_time
    hours, remainder = divmod(time_spent.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time spent for HeteroGAT: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

    torch.save(model2, "heterogat_model.pth")




    print("Training MaskedHeteroGAT ...")
    batch = next(iter(train_loader))

    print("in_channels: ", list(batch.num_node_features.values())[0])

    # Initialize model with required parameters
    model1 = MaskedHeteroGAT(
        # default is 400
        in_channels= list(batch.num_node_features.values())[0],
        hidden_channels=256, 
        out_channels=64, 
        num_heads=4, 
    ).to(device)

    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-4)

    # define the starting time
    start_time = datetime.now()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            
            batch=batch.to(device)

            optimizer.zero_grad()
            probs, loss, attn_weights_pooled = model1(batch)
            labels = batch.y.to(device)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"For MaskedHeteroGAT Time: Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
    
        # --- Evaluation -----
        model1.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(next(model1.parameters()).device)
                logits, _ = model1(batch)
                all_logits.append(logits)
                all_labels.append(batch['label'])
        
        # Concatenate
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        metrics = model1.evaluate(all_logits, all_labels)

        model1.plot_metrics(
            all_labels,
            torch.sigmoid(logits).cpu().numpy(),
            metrics)

    time_spent = datetime.now() - start_time
    hours, remainder = divmod(time_spent.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time spent for MaskedHeteroGAT: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

    # save the model after training
    torch.save(model1, "masked_heterogat_model.pth")



    
