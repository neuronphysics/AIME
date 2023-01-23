import torch
import torch.nn as nn
import os, sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Normalizer1D(nn.Module):
    _epsilon = 1e-16

    def __init__(self, scale, offset):
        super(Normalizer1D, self).__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32) + self._epsilon)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))
        self.to(device)

    def normalize(self, x):
        x = x.permute(0, 2, 1)
        x = (x - self.offset) / self.scale
        return x.permute(0, 2, 1)

    def unnormalize(self, x):
        x = x.permute(0, 2, 1)
        x = x * self.scale + self.offset
        return x.permute(0, 2, 1)

    def unnormalize_mean(self, x_mu):
        x_mu = x_mu.permute(0, 2, 1)
        x_mu = x_mu * self.scale + self.offset
        return x_mu.permute(0, 2, 1)

    def unnormalize_sigma(self, x_sigma):
        x_sigma = x_sigma.permute(0, 2, 1)
        x_sigma = x_sigma * self.scale
        return x_sigma.permute(0, 2, 1)


# compute the normalizers
def compute_normalizer(loader_train):
    # definition
    variance_scaler = 1

    # initialization
    total_batches = 0
    u_mean = 0
    y_mean = 0
    u_var = 0
    y_var = 0
    for i, (u, y) in enumerate(loader_train):
        total_batches += u.size()[0]
        u_mean += torch.mean(u, dim=(0, 2))
        y_mean += torch.mean(y, dim=(0, 2))
        u_var += torch.mean(torch.var(u, dim=2, unbiased=False), dim=(0,))
        y_var += torch.mean(torch.var(y, dim=2, unbiased=False), dim=(0,))

    u_mean = u_mean.detach().cpu().numpy()
    y_mean = y_mean.detach().cpu().numpy()
    u_var = u_var.detach().cpu().numpy()
    y_var = y_var.detach().cpu().numpy()

    u_normalizer = Normalizer1D(np.sqrt(u_var / total_batches) * variance_scaler, u_mean / total_batches)
    y_normalizer = Normalizer1D(np.sqrt(y_var / total_batches) * variance_scaler, y_mean / total_batches)

    return u_normalizer, y_normalizer
