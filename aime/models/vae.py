# https://pyro.ai/examples/vae.html
# https://pyro.ai/examples/dirichlet_process_mixture.html

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import Beta, MultivariateNormal, Categorical, Uniform, Dirichlet

import pyro
import pyro.distributions as dist
from torch.distributions import constraints


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_loc = nn.Linear(2 * 2 * 256, latent_dim)
        self.fc_scale = nn.Linear(2 * 2 * 256, latent_dim)

    def forward(self, x):
        *bs, c, h, w = x.shape
        x = x.view(np.prod(bs), c, h, w)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        z_loc = self.fc_loc(x).view(*bs, -1)
        z_scale = self.fc_scale(x).view(*bs, -1) # should be positive
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, latent_dim, image_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        self.fc1 = nn.Linear(latent_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        loc_img = torch.sigmoid(x)
        assert loc_img.shape[-2] == self.image_dim
        assert loc_img.shape[-3] == self.image_dim
        return loc_img

def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

def truncate(alpha, centers, weights):
    threshold = alpha**-1 / 100.
    true_centers = centers[weights > threshold]
    true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
    return true_centers, true_weights

# need to use Dirichlet Process Mixture Models later
# https://pyro.ai/examples/dirichlet_process_mixture.html
class VAE(nn.Module):
    def __init__(self, z_dim=5, image_dim=64, num_sticks=20, alpha=0.1, use_cuda=True):
        super().__init__()
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.num_sticks = num_sticks
        self.alpha = alpha
        # create the encoder and decoder networks
        self.encoder = Encoder(self.z_dim)
        self.decoder = Decoder(self.z_dim, self.image_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
    
    def model(self, data):
        pyro.module("decoder", self.decoder)
        with pyro.plate("beta_plate", self.num_sticks-1):
            beta = pyro.sample("beta", Beta(1, self.alpha))

        with pyro.plate("mu_plate", self.num_sticks):
            mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)))

        with pyro.plate("data", data.shape[0]):
            z = pyro.sample("z", Categorical(mix_weights(beta)))
            pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)

    '''
    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        with pyro.plate("beta_plate", self.num_sticks-1):
            beta = pyro.sample("beta", Beta(1, self.alpha))

        with pyro.plate("mu_plate", self.num_sticks):
            mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)))
        
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            #z = pyro.sample("z", Categorical(mix_weights(beta)))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.image_dim**2))
    '''
    
    def guide(self, data):
        kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([self.num_sticks-1]), constraint=constraints.positive)
        tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2), 3 * torch.eye(2)).sample([self.num_sticks]))
        phi = pyro.param('phi', lambda: Dirichlet(1/self.num_sticks * torch.ones(self.num_sticks)).sample([data.shape[0]]), constraint=constraints.simplex)

        with pyro.plate("beta_plate", self.num_sticks-1):
            q_beta = pyro.sample("beta", Beta(torch.ones(self.num_sticks-1), kappa))

        with pyro.plate("mu_plate", self.num_sticks):
            q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2)))

        with pyro.plate("data", data.shape[0]):
            z = pyro.sample("z", Categorical(phi))

    '''
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
    '''

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
