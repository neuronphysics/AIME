# https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html

from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F

from torch.distributions import constraints

import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])

def bottle_two_output(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y1, y2 = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y1_size = y1.size()
  y2_size = y2.size()
  return y1.view(x_sizes[0][0], x_sizes[0][1], *y1_size[1:]), y2.view(x_sizes[0][0], x_sizes[0][1], *y2_size[1:])

class SymbolicObservationModel(jit.ScriptModule):
  def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, observation_size)

  @jit.script_method
  def forward(self, belief, state):
    hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    observation = self.fc3(hidden)
    return observation


class VisualObservationModel(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.fc1 = nn.Linear(state_size, embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

  @jit.script_method
  def forward(self, state):
    hidden = self.fc1(state)  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    observation = self.conv4(hidden)
    return observation


def ObservationModel(symbolic, observation_size, state_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicObservationModel(observation_size, -1, state_size, embedding_size, activation_function)
  else:
    return VisualObservationModel(state_size, embedding_size, activation_function)


class SymbolicEncoder(jit.ScriptModule):
  def __init__(self, observation_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.fc1(observation))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden


class VisualEncoder(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, state_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.state_size = state_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
    self.fc_mean = nn.Linear(embedding_size, state_size)
    self.fc_std = nn.Linear(embedding_size, state_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    embedding = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    latent_mean = self.fc_mean(embedding)
    latent_std = F.softplus(self.fc_std(embedding))
    return latent_mean, latent_std


def Encoder(symbolic, observation_size, embedding_size, state_size, activation_function='relu'):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    return VisualEncoder(embedding_size, state_size, activation_function)

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, device, num_inducing=16):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims).to(device=device)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims).to(device=device)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = None
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class TransitionGP(DGPHiddenLayer):
    def __init__(self, latent_size, action_size, lagging_size, device):
      input_size = (latent_size+action_size)*lagging_size
      super(TransitionGP, self).__init__(input_size, latent_size, device)
      self.mean_module = LinearMean(input_size)

class PolicyGP(DGPHiddenLayer):
    def __init__(self, latent_size, action_size, lagging_size, device):
      super(PolicyGP, self).__init__(latent_size*lagging_size, action_size, device)
      self.mean_module = ConstantMean()

class RewardGP(DGPHiddenLayer):
    def __init__(self, latent_size, action_size, lagging_size, device):
      super(RewardGP, self).__init__((latent_size+action_size)*lagging_size, None, device)
      self.mean_module = ZeroMean()

# may be define a wrapper modules that encapsulate several DeepGP for action, transition, and reward ??
class RecurrentGP(DeepGP):
    def __init__(self, horizon_size, latent_size, action_size, lagging_size, device, num_mixture_samples=1):
        super().__init__()
        self.horizon_size = horizon_size
        self.lagging_size = lagging_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.transition_modules = [TransitionGP(latent_size, action_size, lagging_size, device).to(device=device) for _ in range(horizon_size)]
        self.policy_modules = [PolicyGP(latent_size, action_size, lagging_size, device).to(device=device) for _ in range(horizon_size+1)]
        self.reward_gp = RewardGP(latent_size, action_size, lagging_size, device).to(device=device)
        self.num_mixture_samples = num_mixture_samples
        self.likelihood = GaussianLikelihood()
    
    def forward(self, init_states, actions):
        # need to stack actions and latent vectors together (also reshape so that the lagging length dimension is stacked as well)
        init_states = init_states.reshape((init_states.size(0), init_states.size(1), -1))
        actions = actions.reshape((actions.size(0), actions.size(1), -1))
        z_hat = torch.cat([init_states, actions], dim=-1)
        #w_hat = None
        lagging_actions = actions
        lagging_states = init_states
        for i in range(self.horizon_size):
            # policy distribution
            #w_hat = lagging_states # may have to change this to lagging_states[:-1] later
            a = self.policy_modules[i](lagging_states).rsample().mean(dim=0)
            lagging_actions = torch.cat([lagging_actions[..., self.action_size:], a], dim=-1)
            z_hat = torch.cat([lagging_states, lagging_actions], dim=-1)
            # transition distribution
            z = self.transition_modules[i](z_hat).rsample().mean(dim=0)
            # first dimension of z is the number of Gaussian mixtures (z.size(0))
            lagging_states = torch.cat([lagging_states[..., self.latent_size:], z], dim=-1)
        
        # last policy in the horizon
        a = self.policy_modules[self.horizon_size](lagging_states).rsample().mean(dim=0)
        lagging_actions = torch.cat([lagging_actions[..., self.action_size:], a], dim=-1)
        z_hat = torch.cat([lagging_states, lagging_actions], dim=-1)
        # output the final reward
        rewards = self.reward_gp(z_hat)
        return rewards
