import random
import numpy as np
import torch
import torch_optimizer
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .vargp import VARGP


def train(train_set, val_set, test_set, ep_var_mean=True, map_est_hypers=False, dkl=False,
          epochs=1, M=20, n_f=10, n_var_samples=1, batch_size=512, lr=1e-2, beta=1.0,
          eval_interval=10, patience=20, prev_params=None, logger=None, device=None):
  gp = VARGP.create_clf(train_set, M=M, n_f=n_f, n_var_samples=n_var_samples, prev_params=prev_params,
                        ep_var_mean=ep_var_mean, map_est_hypers=map_est_hypers, dkl=dkl).to(device)

  optim = torch_optimizer.Yogi(gp.parameters(), lr=lr)

  N = len(train_set)
  loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
  losses = []
  for e in tqdm(range(epochs)):
    for x, y in loader:
      optim.zero_grad()

      kl_hypers, kl_u, lik = gp.loss(x.to(device), y.to(device))

      loss = beta * kl_hypers + kl_u + (N / x.size(0)) * lik
      loss.backward()
      losses.append(loss.item())

      optim.step()

  return gp, sum(losses) / len(losses)


def set_seeds(seed=None):
  if seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_accuracy(dataset, gp, batch_size=512, device=None):
  loader = DataLoader(dataset, batch_size=batch_size)

  with torch.no_grad():
    count = 0
    for x, y in tqdm(loader, leave=False):
      preds = gp.predict(x.to(device))

      assert not torch.isnan(preds).any(), 'Found NaNs'

      count += (preds.argmax(dim=-1) == y.to(device)).sum().item()

    acc = count / len(dataset)

  return acc


def compute_acc_ent(dataset, gp, batch_size=512, device=None):
  loader = DataLoader(dataset, batch_size=batch_size)

  with torch.no_grad():
    total_corr = 0
    total_ent = 0.

    for x, y in tqdm(loader, leave=False):
      preds = gp.predict(x.to(device))

      assert not torch.isnan(preds).any(), 'Found NaNs'

      total_corr += (preds.argmax(dim=-1) == y.to(device)).sum().item()
      total_ent += Categorical(probs=preds).entropy().sum().item()

  mean_acc = total_corr / len(dataset)
  mean_ent = total_ent / len(dataset)

  return mean_acc, mean_ent


def compute_bwt(acc_mat):
    assert acc_mat.ndim == 2
    assert acc_mat.shape[0] == acc_mat.shape[1]

    T = acc_mat.shape[0]

    return (acc_mat[-1][:-1] - acc_mat.diagonal()[:-1]).mean()


# Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopper:
  def __init__(self, patience=10, delta=1e-4):
    self.patience = patience
    self.delta = delta

    self._counter = 0

    self._best_info = None
    self._best_score = None

  def is_done(self):
    if self.patience >= 0:
      return self._counter >= self.patience
    return False

  def info(self):
    return self._best_info

  def __call__(self, score, info):
    assert not self.is_done()

    if self._best_score is None:
      self._best_score = score
      self._best_info = info
    elif score < self._best_score + self.delta:
      self._counter += 1
    else:
      self._best_score = score
      self._best_info = info
      self._counter = 0
