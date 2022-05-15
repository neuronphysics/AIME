from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F

from torch.distributions import Normal, Categorical, Independent, constraints

import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


class TransitionModel(jit.ScriptModule):
  __constants__ = ['min_std_dev']

  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.min_std_dev = min_std_dev
    self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
    self.rnn = nn.GRUCell(belief_size, belief_size)
    self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
    self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
    self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
    self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

  # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
  # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
  # t :  0  1  2  3  4  5
  # o :    -X--X--X--X--X-
  # a : -X--X--X--X--X-
  # n : -X--X--X--X--X-
  # pb: -X-
  # ps: -X-
  # b : -x--X--X--X--X--X-
  # s : -x--X--X--X--X--X-
  @jit.script_method
  def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = actions.size(0) + 1
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
      # Compute belief (deterministic hidden state)
      hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])
      # Compute state prior by applying transition dynamics
      hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
      prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
      prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
      if observations is not None:
        # Compute state posterior by applying transition dynamics and using current observation
        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
    # Return new hidden states
    hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    if observations is not None:
      hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
    return hidden


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

#Infinite Gaussian Mixture Variational Autoencoder
def init_mlp(layer_sizes, std=.01, bias_init=0.):
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params.append([
            nn.init.normal_(torch.empty(n_in, n_out)).requires_grad_(True),
            torch.empty(n_out).fill_(bias_init).requires_grad_(True)])
    return params

def mlp(x, params):
    for i, (W, b) in enumerate(params):
        x = x@W + b
        if i < len(params) - 1:
            x = torch.relu(x)
    return x

def compute_nll(x, x_recon_linear):
    size = x.size(1)
    return F.binary_cross_entropy_with_logits(x_recon_linear, x)*size 

def gauss_cross_entropy(mu_post, sigma_post, mu_prior, sigma_prior):
    d = (mu_post - mu_prior)
    d = torch.mul(d,d)
    return torch.sum(-torch.div(d + torch.mul(sigma_post,sigma_post),(2.*sigma_prior*sigma_prior)) - torch.log(sigma_prior*2.506628), dim=1, keepdim=True)


def beta_fn(a,b):
    return torch.exp( torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b) )


def compute_kumar2beta_kld(a, b, alpha, beta):
    # precompute some terms
    ab    = torch.mul(a,b)
    a_inv = torch.pow(a, -1)
    b_inv = torch.pow(b, -1)

    # compute taylor expansion for E[log (1-v)] term
    kl = torch.mul(torch.pow(1+ab,-1), beta_fn(a_inv, b))
    for idx in xrange(10):
        kl += torch.mul(torch.pow(idx+2+ab,-1), beta_fn(torch.mul(idx+2., a_inv), b))
    kl = torch.mul(torch.mul(beta-1,b), kl)

    kl += torch.mul(torch.div(a-alpha,a), -0.57721 - torch.polygamma(b) - b_inv)
    # add normalization constants
    kl += torch.log(ab) + torch.log(beta_fn(alpha, beta))

    # final term
    kl += torch.div(-(b-1),b)

    return kl


def log_normal_pdf(x, mu, sigma):
    d = mu - x
    d2 = torch.mul(-1., torch.mul(d,d))
    s2 = torch.mul(2., torch.mul(sigma,sigma))
    return torch.sum(torch.div(d2,s2) - torch.log(torch.mul(sigma, 2.506628)), dim=1, keepdim=True)


def log_beta_pdf(v, alpha, beta):
    return torch.sum((alpha-1)*torch.log(v) + (beta-1)*torch.log(1-v) - torch.log(beta_fn(alpha,beta)), dim=1, keepdim=True)


def log_kumar_pdf(v, a, b):
    return torch.sum(torch.mul(a-1, torch.log(v)) + torch.mul(b-1, torch.log(1-torch.pow(v,a))) + torch.log(a) + torch.log(b), dim=1, keepdim=True)


def mcMixtureEntropy(pi_samples, z, mu, sigma, K):
    s = torch.mul(pi_samples[0], torch.exp(log_normal_pdf(z[0], mu[0], sigma[0])))
    for k in xrange(K-1):
        s += torch.mul(pi_samples[k+1], torch.exp(log_normal_pdf(z[k+1], mu[k+1], sigma[k+1])))
    return -torch.log(s)


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


class RewardModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)

  @jit.script_method
  def forward(self, belief, state):
    hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    reward = self.fc3(hidden).squeeze(dim=1)
    return reward


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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VisualEncoder(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, z_dim):
    super().__init__()
    #From Eric Nalisnick's code (Variational Autoencoders with Gaussian Mixture Latent Space)
    self.prior = hyperParams['prior']
    self.K = hyperParams['K']

    self.embedding_size = embedding_size

    self.encoder_layers = nn.ModuleList([nn.Sequential(
                 nn.Conv2d(3, 32, kernel_size=4, stride=2),
                 nn.ReLU(),
                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                 nn.ReLU(),
                 nn.Conv2d(64, 128, kernel_size=4, stride=2),
                 nn.ReLU(),
                 nn.Conv2d(128, 256, kernel_size=4, stride=2),
                 nn.ReLU(),
                 Flatten()
                 ), #input to hidden
                [nn.Linear(embedding_size,z_dim/self.K) for k in xrange(self.K)],#hidden to mu
                [nn.Linear(embedding_size,z_dim/self.K) for k in xrange(self.K)],#hidden to logvar???
                nn.Linear(embedding_size,embedding_size),
                mlp(embedding_size,init_mlp([embedding_size, self.K-1], 1e-8)),#kumar_a                
                mlp(embedding_size,init_mlp([embedding_size, self.K-1], 1e-8))#kumar_b
                ])
    self.decoder_layers = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(z_dim, 128, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
                nn.Sigmoid(),
                )


  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    return hidden

class SampleLayer(nn.Module):
  
  def __init__(self, embedding_size, state_size):
    super().__init__()
    self.embedding_size = embedding_size
    self.state_size = state_size
    self.fc_mean = nn.Linear(embedding_size, state_size)
    self.fc_std = nn.Linear(embedding_size, state_size)
  
  def forward(self, embedding):
    latent_mean = self.fc_mean(embedding)
    latent_std = F.softplus(self.fc_std(embedding))
    latent_state = latent_mean + torch.rand_like(latent_mean) * latent_std
    return latent_mean, latent_std, latent_state

def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    return VisualEncoder(embedding_size, activation_function)

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, device, num_inducing=5):
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
      super(RewardGP, self).__init__((latent_size+action_size)*lagging_size, 1, device)
      self.mean_module = ZeroMean()

# may be define a wrapper modules that encapsulate several DeepGP for action, transition, and reward ??
class RecurrentGP(DeepGP):
    def __init__(self, horizon_size, latent_size, action_size, lagging_size, device, num_mixture_samples=1, noise=0.5):
        super().__init__()
        self.horizon_size = horizon_size
        self.lagging_size = lagging_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.transition_modules = [TransitionGP(latent_size, action_size, lagging_size, device).to(device=device) for _ in range(horizon_size)]
        self.policy_modules = [PolicyGP(latent_size, action_size, lagging_size, device).to(device=device) for _ in range(horizon_size+1)]
        self.reward_gp = RewardGP(latent_size, action_size, lagging_size, device).to(device=device)
        self.num_mixture_samples = num_mixture_samples
        self.noise = noise
    
    def forward(self, init_states, actions):
        with gpytorch.settings.num_likelihood_samples(self.num_mixture_samples):
          # need to stack actions and latent vectors together (also reshape so that the lagging length dimension is stacked as well)
          init_states = init_states.reshape((init_states.size(0), init_states.size(1), -1))
          actions = actions.reshape((actions.size(0), actions.size(1), -1))
          z_hat = torch.cat([init_states, actions], dim=-1)
          #w_hat = None
          lagging_actions = actions
          lagging_states = init_states
          posterior_states = torch.empty((self.horizon_size, init_states.size(0), init_states.size(1), self.latent_size))
          posterior_actions = torch.empty((self.horizon_size+1, init_states.size(0), init_states.size(1), self.action_size))
          for i in range(self.horizon_size):
              # policy distribution
              #w_hat = lagging_states # may have to change this to lagging_states[:-1] later
              a = self.policy_modules[i](lagging_states).rsample().squeeze(0)
              a = a + self.noise * torch.rand_like(a)
              posterior_actions[i] = a
              lagging_actions = torch.cat([lagging_actions[..., self.action_size:], a], dim=-1)
              z_hat = torch.cat([lagging_states, lagging_actions], dim=-1)
              # transition distribution
              z = self.transition_modules[i](z_hat).rsample().squeeze(0)
              z = z + self.noise * torch.rand_like(z)
              # first dimension of z is the number of Gaussian mixtures (z.size(0))
              posterior_states[i] = z
              lagging_states = torch.cat([lagging_states[..., self.latent_size:], z], dim=-1)
          
          # last policy in the horizon
          a = self.policy_modules[self.horizon_size](lagging_states).rsample().squeeze(0)
          a = a + self.noise * torch.rand_like(a)
          posterior_actions[self.horizon_size] = a
          lagging_actions = torch.cat([lagging_actions[..., self.action_size:], a], dim=-1)
          z_hat = torch.cat([lagging_states, lagging_actions], dim=-1)
          # output the final reward
          rewards = self.reward_gp(z_hat).rsample().squeeze(0)
          rewards = rewards + self.noise * torch.rand_like(rewards)
          return rewards, posterior_actions, posterior_states