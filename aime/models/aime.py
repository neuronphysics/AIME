import torch
from torch import nn

class AIMEModel(nn.Module):
    def __init__(self, vae, rgp):
        super(AIMEModel, self).__init__()
        self.vae = vae
        self.rgp = rgp

class AIMEOptmizier(object):
    def __init__(self, vae_optimizer, rgp_optimizer):
        self.vae_optimizer = vae_optimizer
        self.rgp_optimizer = rgp_optimizer
    
    def zero_grad(self):
        self.vae_optimizer.zero_grad()
        self.rgp_optimizer.zero_grad()
    
    def step(self):
        self.vae_optimizer.step()
        self.rgp_optimizer.step()