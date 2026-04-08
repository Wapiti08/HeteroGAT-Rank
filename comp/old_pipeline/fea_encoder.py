'''
 # @ Create Time: 2025-01-02 20:09:05
 # @ Modified time: 2025-01-02 20:09:10
 # @ Description: define diverse encoding method for node / edge attrobites

 '''
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

class SeqEncoder:
    ''' cover sequence encoding for path, domain, cmd, user, process, event
    
    '''
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        # Get the embedding size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def __call__(self, value):
        # Handle potential missing, blank, or zero values
        value = "" if pd.isna(value) or value == "" or value == 0 else value
        
        # Process the single value
        x = self.model.encode(value, show_progress_bar=False, convert_to_tensor=True, device=self.device).unsqueeze(0)
        
        # ensure consistent size by padding or truncating to "embedding_dim" --- important for consistence when training
        if x.size(1) < self.embedding_dim:
            # Pad the tensor if its size is less than embedding_dim
            padding_size = self.embedding_dim - x.size(1)
            x = torch.cat([x, torch.zeros((1, padding_size), device=self.device)], dim=1)
        elif x.size(1) > self.embedding_dim:
            # If tensor is too large (shouldn't happen for SentenceTransformer but in case),
            # truncate it to the desired size
            x = x[:, :self.embedding_dim]

        # Fill the missing part with zero (if any)
        zero_vector = torch.zeros((1, self.embedding_dim), device=self.device)
        mask = torch.tensor(value == "", dtype=torch.bool, device=self.device)
        x = torch.where(mask.unsqueeze(-1), zero_vector, x)

        return x.cpu()


class IdenEncoder:
    ''' Cover numeric encoding for ports, or other numeric value/attributes '''
    
    def __init__(self, dtype=None, output_dim=None):
        self.dtype = dtype
        self.output_dim = output_dim
    
    def __call__(self, value):
        x = torch.tensor([[value]]).to(self.dtype)
        x = F.pad(x, (0, self.output_dim - 1), "constant", 0)
        return x


class CateEncoder:
    ''' dynamic and flexiable solution:
    Categorical encoding using integer encoding and embedding, no need for num_categories upfront 
    
    '''
    
    def __init__(self, embedding_dim):
        # Initialize with a default embedding layer (no num_categories initially)
        self.embedding_dim = embedding_dim
        self.mapping = {}  # This will map the categorical values to indices
        self.embedding = nn.Embedding(1, self.embedding_dim)  # Initial embedding layer with 1 category
    
    
    def __call__(self, value):
        ''' Encodes the specified value using integer encoding and embedding '''
        # Convert the category to its corresponding integer index
        category_idx = self.value_to_idx(value)
        # Get the embedding for the integer index
        x = self.embedding(torch.tensor([category_idx]))
        return x

    def value_to_idx(self, value):
        ''' Converts a value to its corresponding integer index, dynamically creates index if needed '''
        if value not in self.mapping:
            # Assign the next index
            self.update(value)
        return self.mapping[value]

    def update(self, value):
        ''' Update the mapping based on new value, resize embedding if necessary '''
        if value not in self.mapping:
            current_size = len(self.mapping)
            # Resize embedding to include the new category
            self.mapping[value] = current_size
            
            # Resize the embedding matrix to accommodate the new category
            new_embedding = nn.Embedding(current_size + 1, self.embedding_dim)
            # Copy the weights from the old embedding to the new one without in-place modification
            with torch.no_grad():
                new_embedding.weight[:current_size] = self.embedding.weight   
            # Replace the embedding with the new one
            self.embedding = new_embedding         

