import os

import numpy as np
import torch
from torch import nn, optim
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from .models.vae import VAE
from .models.rgp import RecurrentGP
from .models.aime import AIMEModel, AIMEOptmizier

from .data.loader import (
    data_ingredient,
    load_data,
    encode_batch_of_pairs,
    get_random_collection_env,
    get_env_info,
)

import sacred

assert pyro.__version__.startswith('1.7.0')
#pyro.distributions.enable_validation(False)
#pyro.set_rng_seed(0)

def train_epoch(train_data, val_data, aime_model, aime_optimizer, epoch, num_epochs):
    train_loader = train_data.chunk_loader
    train_bar = tqdm(train_loader, leave=epoch != num_epochs)
    for data in train_bar:
        aime_optimizer.zero_grad()
        print(data)
    return -1

def validate_loss(val_data, aime_model):
    return -1

def train_loop(train_data, val_data, aime_model, aime_optimizer, num_epochs, print_every=20, validate_every=50):
    bar = tqdm(range(1, num_epochs + 1), leave=False)
    best_validation_loss = None
    for epoch in bar:
        aime_model.train()
        loss = train_epoch(train_data, val_data, aime_model, aime_optimizer, epoch, num_epochs)
        if epoch % print_every == 0:
            print(loss)
        aime_model.eval()
        if epoch % validate_every == 0:
            validation_loss = validate_loss(val_data, aime_model)
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(aime_model.state_dict(), f"saved_models/epoch_{epoch}_model.pt")
    return best_validation_loss

experiment_name = "aime"
ex = sacred.Experiment(
    experiment_name,
    ingredients=[data_ingredient],
)

@ex.main
def run_training():
    # model hyperparameters
    horizon_size = 20
    lagging_latent_size = 10 # M lagging size
    lagging_observation_number = 10 # M_x (coule be different from M)
    lagging_action_number = 10 # L_a
    num_epochs=100
    
    # dataset hyperparameters
    n_train_rollouts_total = 500
    n_train_rollouts_subset = 500
    n_val_rollouts_total = 50
    n_val_rollouts_subset = 50

    vae = VAE()
    rgp = RecurrentGP(horizon_size)
    vae_optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999))
    rgp_optimizer = optim.Adam(rgp.parameters(), lr=0.001, betas=(0.9, 0.999))
    aime_model = AIMEModel(vae, rgp)
    aime_model.cuda()
    aime_optimizer = AIMEOptmizier(vae_optimizer, rgp_optimizer)
    
    env_name, env_kwargs = get_random_collection_env()
    train_data = load_data(
                    env_name=env_name,
                    env_kwargs=env_kwargs,
                    split_name="train",
                    n_rollouts_total=n_train_rollouts_total,
                    n_rollouts_subset=n_train_rollouts_subset,
                )
    val_data = load_data(
                    env_name=env_name,
                    env_kwargs=env_kwargs,
                    split_name="val",
                    n_rollouts_total=n_val_rollouts_total,
                    n_rollouts_subset=n_val_rollouts_subset,
                )
    os.makedirs("saved_models", exist_ok=True)
    
    best_validation_loss = train_loop(train_data, val_data, aime_model, aime_optimizer, num_epochs)
    print(best_validation_loss)

if __name__ == "__main__":
    import sys

    ex.run_commandline(argv=sys.argv)