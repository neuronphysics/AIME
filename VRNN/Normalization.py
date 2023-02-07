import torch
import torch.nn as nn
import os, sys
import numpy as np 
import einops
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
    def __init__(self, scale, offset):
        super(Normalizer1D, self).__init__()
        self.register_buffer('scale', nn.Parameter(torch.tensor(scale, dtype=torch.float32, device=device) , requires_grad=False))
        self.register_buffer('offset', nn.Parameter(torch.tensor(offset, dtype=torch.float32, device=device), requires_grad=False))
        self.to(device)
        self.extra_repr()
        
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.offset, self.scale)

    def normalize(self, x):
        #(B, D, T)
        length= torch.LongTensor([torch.max((x[i,0,:]!=0).nonzero()).item()+1 for i in range(x.shape[0])])
        max_len=x.size(-1)
        mask = get_mask_from_sequence_lengths( length, max_len)
        #torch.Size([1, D, T]
        m    = einops.repeat(mask, 'm n -> m k n', k = x.size(1)).to(device)
        
        x = x.permute( 0, 2, 1)
        #x torch.Size([1, T, D])
        x = x.sub( self.offset[ None, :]).div( self.scale[ None, :])
        
        x = torch.mul(x, m.permute(0,2,1) )
        
        return x.permute( 0, 2, 1)
        
        
    def unnormalize(self, x):
        #(T, B, D)==>(B, T, D)
        length= torch.LongTensor([torch.max((x[i,0,:]!=0).nonzero()).item()+1 for i in range(x.shape[0])])        
        max_len=x.size(-1)
        mask = get_mask_from_sequence_lengths( length, max_len).to(device)
       
        m    = einops.repeat(mask, 'm n -> m k n', k = x.size(1))
        x = x.permute( 0, 2, 1)

        x = x * self.scale[ None, :] + self.offset[ None, :]
        x = torch.mul(x, m.permute(0,2,1) )
        return x.permute( 0, 2, 1)


    def unnormalize_mean(self, x_mu):
        #(T, B, D)==>(B, T, D)
        length= torch.LongTensor([torch.max((x_mu[i,0,:]!=0).nonzero()).item()+1 for i in range(x_mu.shape[0])])       
        max_len=x_mu.size(-1)
        mask = get_mask_from_sequence_lengths( length, max_len).to(device)
        
        m  = einops.repeat(mask, 'm n -> m k n', k = x_mu.size(1))
        x_mu = x_mu.permute( 0, 2, 1)

        x_mu = x_mu * self.scale[ None, :] + self.offset[ None, :]
        x_mu = torch.mul(x_mu, m.permute(0,2,1) )
        return x_mu.permute( 0, 2, 1)


    def unnormalize_sigma(self, x_sigma):
        #(T, B, D)==>(B, T, D)
        length= torch.LongTensor([torch.max((x_sigma[i,0,:]!=0).nonzero()).item()+1 for i in range(x_sigma.shape[0])])
        max_len=x_sigma.size(-1)
        mask = get_mask_from_sequence_lengths( length, max_len).to(device)

        m = einops.repeat(mask, 'm n -> m k n', k = x_sigma.size(1))
        x_sigma = x_sigma.permute( 0, 2, 1)

        x_sigma = x_sigma * self.scale[ None, :]
        x_sigma = torch.mul(x_sigma, m.permute(0,2,1) )
        return x_sigma.permute( 0, 2, 1)


def compute_mean_variance(data, seq_length):
    #(B, D, T)
    mean_x = torch.zeros([data.shape[0],data.shape[1]])
    std_x  = torch.zeros([data.shape[0],data.shape[1]])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #(B, D, T) => std, mean
            std_x[i,j], mean_x[i,j]=torch.std_mean(data[i,j,:seq_length[i]], keepdim=True, unbiased=False,dim=-1)
                
    #Dimension (B, D)
    return std_x, mean_x

    
# compute the normalizers
def compute_normalizer(loader_train):
    # definition #batch_size, input_dim, seq_len

    # initialization
    for i, (u, y) in enumerate(loader_train):
        #input u torch.Size([B, D, T])
        length = torch.LongTensor([torch.max((u[i,0,:]!=0).nonzero()).item()+1 for i in range(u.shape[0])])
        if i !=0:
            std_inputs, mean_inputs   = compute_mean_variance(u, length)
            std_targets, mean_targets = compute_mean_variance(y, length)
            u_mean = torch.cat([u_mean,mean_inputs], dim=0)
            u_std  = torch.cat([u_std,std_inputs], dim=0)
            y_mean = torch.cat([y_mean,mean_targets], dim=0)
            y_std  = torch.cat([y_std,std_targets], dim=0)
            
        else:
            u_std, u_mean = compute_mean_variance(u, length)
            y_std, y_mean = compute_mean_variance(y, length)
            
    u_std[u_std == 0]=1
    y_std[y_std == 0]=1        
    u_mean = u_mean.mean(dim=0, keepdim=True).cpu().numpy()
    y_mean = y_mean.mean(dim=0, keepdim=True).cpu().numpy()
    u_std = u_std.std(dim=0, keepdim=True)
    y_std = y_std.std(dim=0, keepdim=True)
    u_std[u_std == 0]=1
    y_std[y_std == 0]=1 
    u_std =u_std.cpu().numpy()
    y_std =y_std.cpu().numpy()

    u_normalizer = Normalizer1D(u_std, u_mean )
    y_normalizer = Normalizer1D(y_std, y_mean )

    return u_normalizer, y_normalizer

