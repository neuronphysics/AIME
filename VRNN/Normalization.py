import torch
import torch.nn as nn
import os, sys
import numpy as np 
import einops

from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mask_from_sequence_lengths(
    sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.
    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor

class Normalizer1D(nn.Module):   
    # Data size of (batch_size,  seq_len, input_size)
    def __init__(self, input_dim, inputs , feature_range=(-1,1) ):
        super(Normalizer1D, self).__init__()
        self.input_dim = input_dim
         
        self._mask, self._masked_inputs = self._build_mask(inputs)
        self._norm = self.build_normalizers(self._masked_inputs)
        
        self.to(device) 
    
    def _build_mask(self, x):   
        if torch.is_tensor(x):
           data_ = x.clone()
           new_masked_data = np.zeros_like(x.cpu().detach().numpy())
        else:
           data_ = torch.from_numpy(x)
           new_masked_data = np.zeros_like(x)
        max_len=x.shape[-1]
        length = torch.LongTensor([torch.max((data_[i,0,:]!=0).nonzero()).item()+1 for i in range(data_.shape[0])])
        mask = get_mask_from_sequence_lengths( length, max_len)
        mask = mask.detach().numpy()
        for i in range(x.shape[1]):
            M = np.ma.masked_array(data=x[:,i,:],mask=~mask, fill_value=np.nan)
            new_masked_data[:,i,:]= M.filled()
        return mask, new_masked_data 

      
    def build_normalizers(self, x):
        normalizers = OrderedDict()
        scale_, min_ = [], []
        for i in range(self.input_dim):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(x[:,i,:])
            scale_.append(torch.from_numpy(scaler.scale_))
            min_.append(torch.from_numpy(scaler.min_))
            normalizers[str(i)] = scaler
        self.scale_ = torch.stack(scale_).to(device)
        self.min_   = torch.stack(min_).to(device)
        return normalizers

    def normalize(self, x):
        #(B, D, T)
        d = x.cpu().detach().numpy()
        _, nd= self._build_mask(d)
        n_x=[]
        for i in range(x.shape[1]):
            n_x.append(self._norm[str(i)].transform(nd[:,i,:]))
        x =np.stack(n_x, axis=1)
        return torch.from_numpy(x).to(device)
        
    def unnormalize(self, x):
        #(T, B, D)
        d = x.cpu().detach().numpy()
        m, nd = self._build_mask(d)
        self.unnorm_mask = m
        n_x=[]
        for i in range(x.shape[1]):
            n_x.append(self._norm[str(i)].inverse_transform(nd[:,i,:]))
             
        x =np.stack(n_x, axis=1)
        return torch.from_numpy(x).to(device)


    def unnormalize_mean(self, x_mu):
        d= x_mu.clone()
        normX  = d.sub_(self.min_)
        return normX.div_(self.scale_)
    
    def unnormalize_sigma(self, x_sigma):
        d = x_sigma.clone()
        return d.div_(self.scale_)
        
    
# compute the normalizers
def compute_normalizer(loader_train):
    # definition #batch_size, input_dim, seq_len
    for i, (u, y) in enumerate(loader_train):
        if i ==0:
           #input u output y: shape ==> torch.Size([B, D, T])
           inputs  = u
           outputs = y
        else:
           inputs  = torch.cat([inputs,u], dim=0)
           outputs = torch.cat([outputs,y], dim=0)
    
    inputs  = inputs.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    
    # initialization
    u_normalizer =  Normalizer1D(inputs.shape[1], inputs)
    y_normalizer = Normalizer1D(outputs.shape[1], outputs)

    return u_normalizer, y_normalizer

