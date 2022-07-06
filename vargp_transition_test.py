import torch
from tqdm import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

import urllib.request
import os
from scipy.io import loadmat
from math import floor
import gym

from torch.utils.data import TensorDataset, DataLoader
from var_gp.train_utils import train

# ========================= create data ============================
env = gym.make('Pendulum-v0')
X = []
y = []
state = env.reset()
reward_min = -16.5
reward_max = 0
n = 2000
for i in range(n):
  action = env.action_space.sample()
  new_state, reward, done, _ = env.step(action)
  train_x_sample = torch.cat([torch.tensor(state).float(), torch.tensor(action).float()])
  X.append(train_x_sample)
  y.append(torch.tensor(new_state).float())
  state = new_state
  if done:
    state = env.reset()


X = torch.stack(X, dim=0)
y = torch.stack(y, dim=0)

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

train_dataset = TensorDataset(train_x, train_y)
# change num_tasks to control dataset size
num_tasks = 1
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset) // num_tasks)
test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# =========================== train =============================
transition_prev_params = []
for x, y in tqdm(train_loader):
    td = TensorDataset(x, y)
    transition_model, loss = train(td, None, None,
                                   batch_size=32,
                                   n_f=3,
                                   n_var_samples=10,
                                   epochs=50,
                                   prev_params=transition_prev_params, device='cpu')
    transition_prev_params.append(transition_model.state_dict())
    print(loss)


transition_model.eval()
with torch.no_grad():
    for x, y in tqdm(test_loader):
        y_pred = transition_model.predict(x).mean(dim=0).transpose(0, 1)
        print(torch.mean(torch.pow(y - y_pred, 2)).sqrt())

# ========================== eval ===========================
from matplotlib import pyplot as plt
plt.scatter(y[:, 0], y_pred[:, 0])
plt.savefig('plot0.png')
plt.close()

plt.scatter(y[:, 1], y_pred[:, 1])
plt.savefig('plot1.png')
plt.close()

plt.scatter(y[:, 2], y_pred[:, 2])
plt.savefig('plot2.png')
plt.close()

