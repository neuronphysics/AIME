import random
import numpy as np
import torch
import torch_optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .kernels import RBFKernel
from .likelihoods import  GaussianLikelihood
from .models_global import GlobalContinualSVGP
from .train_utils import EarlyStopper


def create_gaussian_gp(dataset, M=20, n_f=10, n_var_samples=3,
                       map_est_hypers=False, prev_params=None):

  N = len(dataset)
  out_size = torch.unique(dataset.targets).size(0)

  if prev_params:
    z_prev = prev_params.get('z').cpu()
    z_new = []
    for i in range(out_size):
      z_prev_i = z_prev[i]
      M_prev_i = z_prev_i.shape[0]
      # this silently assumes more inducing points for additional tasks
      M_add_i = M - M_prev_i
      z_add_i = dataset[torch.randperm(N)[:M_add_i]][0]
      z_i = torch.cat([z_prev_i, z_add_i], 0)
      z_new.append(z_i)
    z = torch.stack(z_new)
  else:
    # init inducing points at random data points.
    z = torch.stack([
      dataset[torch.randperm(N)[:M]][0]
      for _ in range(out_size)])

  prior_log_mean, prior_log_logvar = None, None
  if prev_params:
    prior_log_mean = prev_params.get('kernel.log_mean')
    prior_log_logvar = prev_params.get('kernel.log_logvar')

  kernel = RBFKernel(z.size(-1), prior_log_mean=prior_log_mean, prior_log_logvar=prior_log_logvar,
                     map_est=map_est_hypers)
  likelihood = GaussianLikelihood(out_size=n_f)
  gp = GlobalContinualSVGP(z, kernel, likelihood, n_var_samples=n_var_samples,
                     prev_params=prev_params)
  return gp


def train(task_id, train_set, val_set, test_set, map_est_hypers=False,
          epochs=1, M=20, n_f=10, n_var_samples=3, batch_size=512, lr=1e-2, beta=1.0,
          eval_interval=10, patience=20, prev_params=None, logger=None, device=None):
  gp = create_gaussian_gp(train_set, M=M, n_f=n_f, n_var_samples=n_var_samples,
                       map_est_hypers=map_est_hypers,
                       prev_params=prev_params).to(device)

  stopper = EarlyStopper(patience=patience)

  # optim = torch.optim.Adam(gp.parameters(), lr=lr)
  optim = torch_optimizer.Yogi(gp.parameters(), lr=lr)

  N = len(train_set)
  loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

  for e in tqdm(range(epochs)):
    for x, y in tqdm(loader, leave=False):
      optim.zero_grad()

      kl_hypers, kl_u, u_prev_reg, lik = gp.loss(x.to(device), y.to(device))

      loss = beta * kl_hypers + kl_u - u_prev_reg + (N / x.size(0)) * lik
      loss.backward()

      optim.step()

  return gp
