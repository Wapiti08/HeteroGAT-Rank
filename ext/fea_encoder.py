'''
 # @ Create Time: 2025-01-02 20:09:05
 # @ Modified time: 2025-01-02 20:09:10
 # @ Description: define diverse encoding method for node / edge attrobites

 '''
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    ''' Cover categorical encoding for status, response code '''
    
    def __init__(self):
        self.mapping = {}
    
    def __call__(self, value):
        ''' Encodes the specified value using one-hot encoding '''
        # Update the mapping if the value is not already present
        if value not in self.mapping:
            self.update(value)
        
        # Create a tensor of zeros with a size equal to the number of categories
        x = torch.zeros(1, len(self.mapping))
        
        # Set the appropriate index to 1 based on the value
        x[0, self.mapping[value]] = 1
        
        return x

    def update(self, value):
        ''' Update the mapping based on new value '''
        if value not in self.mapping:
            self.mapping[value] = len(self.mapping)


# class SequenceEncoder:
#     ''' cover sequence encoding for path, domain, cmd, user, process, event
    
#     '''
#     def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
#         self.model = SentenceTransformer(model_name, device=device)
#         self.device = device
#         # get the embedding size
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()

#     @torch.no_grad()
#     def __call__(self, df):
#         values = df.values.flatten().tolist()

#         # handle potential missing, blank, or zero values
#         values = ["" if pd.isna(v) or v == "" or v == 0 else v for v in values]

#         # process when value length is only one
#         if len(values) == 1:
#             x = self.model.encode(values[0],  show_progress_bar=False,
#                               convert_to_tensor=True, device=self.device).unsqueeze(0)
#         else:
#             x = self.model.encode(values, show_progress_bar=True,
#                                 convert_to_tensor=True, device=self.device)

#         # fill the missing part with zero
#         zero_vector = torch.zeros((len(values), self.embedding_dim), device=self.device)
#         mask = torch.tensor([v == "" for v in values], dtype=torch.bool, device=self.device)
#         x = torch.where(mask.unsqueeze(-1), zero_vector, x)

#         return x.cpu()


# class IdentityEncoder:
#     ''' cover numeric encoding for ports, or other numeric value/attributes
    
#     '''
#     def __init__(self, dtype=None, output_dim = None):
#         self.dtype = dtype
#         self.output_dim = output_dim
    
#     def __call__(self, df):
#         x = torch.from_numpy(df.values).view(-1,1).to(self.dtype)
#         x = F.pad(x, (0, self.output_dim - 1), "constant", 0)
#         return x


# class CateEncoder:
#     ''' cover categorical encoding for status, response code
    
#     status: limited choices - one-hot encoding
    
#     '''
#     def __init__(self,):
#         self.mapping = None
    
#     def __call__(self, df):
#         ''' Encodes the specified column in the DataFrame using one-hot encoding
        
#         '''
#         if self.mapping is None:
#             # create a mapping if it doesn't exist
#             # get all the types
#             types = df.values.unique().tolist()
#             # create mapping
#             self.mapping = {type: i for i, type in enumerate(types)}

#         # create a tensor of zeros
#         x = torch.zeros(len(df), len(self.mapping))

#         for i, value in enumerate(df[self.column]):
#             x[i, self.mapping[value]] = 1
        
#         return x

#     def update(self, df):
#         ''' update the mapping based on new data in the DataFrame
        
#         '''
#         if self.mapping is None:
#             self.__call__(df)
        
#         else:
#             new_types = df.values.unique().tolist()
#             for type_ in new_types:
#                 if type_ not in self.mapping:
#                     self.mapping[type_] = len(self.mapping)
        
