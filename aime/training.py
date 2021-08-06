import os

import numpy as np
import torch
from torch import nn, optim
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from models.vae import VAE
from models.rgp import RecurrentGP
from models.aime import AIMEModel, AIMEOptmizier

from data.data_loader import load_data

assert pyro.__version__.startswith('1.7.0')
#pyro.distributions.enable_validation(False)
#pyro.set_rng_seed(0)

# all the hyperparameters
horizon_size = 20
lagging_latent_size = 10 # M lagging size
lagging_observation_number = 10 # M_x (coule be different from M)
lagging_action_number = 10 # L_a
num_epochs=1000

vae = VAE()
rgp = RecurrentGP(horizon_size)
vae_optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999))
rgp_optimizer = optim.Adam(rgp.parameters(), lr=0.001, betas=(0.9, 0.999))
aime_model = AIMEModel(vae, rgp)
aime_optimizer = AIMEOptmizier(vae_optimizer, rgp_optimizer)
env_name = "ModifiedPendulumEnv-v0"
train_data = load_data(env_name, mode='train')
val_data = load_data(env_name, mode='validation')

os.makedirs("saved_models", exist_ok=True)

def train_epoch(train_data, val_data, aime_model, aime_optimizer):
    raise NotImplementedError

def validate_loss(val_data, aime_model):
    raise NotImplementedError

def train_loop(train_data, val_data, aime_model, aime_optimizer, num_epochs, print_every=20, validate_every=10):
    bar = tqdm(range(1, num_epochs + 1), leave=False)
    best_validation_loss = None
    for epoch in bar:
        aime_model.train()
        loss = train_epoch(train_data, val_data, aime_model, aime_optimizer)
        if epoch % print_every == 0:
            print(loss)
        aime_model.eval()
        if epoch % validate_every == 0:
            validation_loss = validate_loss(val_data, aime_model)
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(aime_model.state_dict(), f"saved_models/epoch_{epoch}_model.pt")
    return best_validation_loss

best_validation_loss = train_loop(train_data, val_data, aime_model, aime_optimizer, num_epochs)
print(best_validation_loss)