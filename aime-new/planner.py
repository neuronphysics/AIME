from math import inf
import torch
from torch import jit
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

import gpytorch

from models import DGPHiddenLayer
from gpytorch.models.deep_gps import DeepGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

class ValueNetwork(nn.Module):
  def __init__(self, latent_size, num_sample_trajectories, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.latent_size = latent_size
    self.fc1 = nn.Linear(latent_size + num_sample_trajectories, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 1)
  
  def forward(self, embedding):
    hidden = self.act_fn(self.fc1(embedding))
    return self.fc2(hidden)

class QNetwork(nn.Module):
  def __init__(self, latent_size, num_sample_trajectories, action_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.latent_size = latent_size
    self.action_size = action_size
    self.fc1 = nn.Linear(latent_size + num_sample_trajectories + action_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 1)
  
  def forward(self, embedding, action):
    hidden = self.act_fn(self.fc1(torch.cat([embedding, action], dim=-1)))
    return self.fc2(hidden)

class FirstPolicyLayer(DGPHiddenLayer):
  def __init__(self, latent_size, num_sample_trajectories, hidden_size, device):
    super(FirstPolicyLayer, self).__init__(latent_size+num_sample_trajectories, hidden_size, device)
    self.mean_module = LinearMean(latent_size+num_sample_trajectories)

class SecondPolicyLayer(DGPHiddenLayer):
  def __init__(self, hidden_size, action_size, device):
    super(SecondPolicyLayer, self).__init__(hidden_size, action_size, device)
    self.mean_module = ConstantMean()

class PolicyModel(DeepGP):
  def __init__(self, latent_size, action_size, num_sample_trajectories, hidden_size, device):
    super().__init__()
    self.first_policy_layer = FirstPolicyLayer(latent_size, num_sample_trajectories, hidden_size, device)
    self.second_policy_layer = SecondPolicyLayer(hidden_size, action_size, device)
    self.likelihood = GaussianLikelihood()
  
  def forward(self, embedding):
    hidden = self.first_policy_layer(embedding)
    return self.second_policy_layer(hidden)

class FirstTransitionLayer(DGPHiddenLayer):
  def __init__(self, latent_size, action_size, num_sample_trajectories, hidden_size, device):
    super(FirstTransitionLayer, self).__init__(latent_size+num_sample_trajectories+action_size, hidden_size, device)
    self.mean_module = LinearMean(latent_size+num_sample_trajectories+action_size)

class SecondTransitionLayer(DGPHiddenLayer):
  def __init__(self, hidden_size, latent_size, device):
    super(SecondTransitionLayer, self).__init__(hidden_size, latent_size, device)
    self.mean_module = ConstantMean()

class TransitionModel(DeepGP):
  def __init__(self, latent_size, action_size, num_sample_trajectories, hidden_size, device):
    super().__init__()
    self.first_transition_layer = FirstTransitionLayer(latent_size, action_size, num_sample_trajectories, hidden_size, device)
    self.second_transition_layer = SecondTransitionLayer(hidden_size, latent_size, device)
    self.likelihood = GaussianLikelihood()
  
  def forward(self, embedding, action):
    hidden = self.first_transition_layer(torch.cat([embedding, action], dim=-1))
    return self.second_transition_layer(hidden)

class RolloutEncoder(nn.Module):
  def __init__(self, latent_size, action_size, hidden_size, num_sample_trajectories, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.latent_size = latent_size
    self.action_size = action_size
    self.hidden_size = hidden_size
    self.encoder = nn.LSTM(input_size=action_size+latent_size, hidden_size=hidden_size)
    self.fc = nn.Linear(num_sample_trajectories*(hidden_size+action_size+1), hidden_size)
  
  def forward(self, imagined_reward, imagined_actions, imagined_states):
    last_action = imagined_actions[-1]
    rewards = imagined_reward.view(-1, 1)
    state_action_sequence = torch.cat([imagined_actions[:-1], imagined_states], dim=-1)
    _, (state_action_embedding, _) = self.encoder(state_action_sequence)
    embedding = torch.cat([state_action_embedding.squeeze(0), last_action, rewards], dim=-1)
    embedding = embedding.view(1, -1)
    embedding = self.act_fn(self.fc(embedding))
    return embedding

class ActorCriticPlanner(nn.Module):
  def __init__(self, lagging_size, latent_size, action_size, recurrent_gp, min_action, max_action,
               num_sample_trajectories, hidden_size, num_gp_likelihood_samples, device):
    super().__init__()
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.action_scale = (self.max_action - self.min_action) / 2
    self.action_bias = (self.max_action + self.min_action) / 2
    self.latent_size = latent_size
    self.actor = PolicyModel(latent_size, action_size, num_sample_trajectories, hidden_size, device)
    self.policy_mll = DeepApproximateMLL(VariationalELBO(self.actor.likelihood, self.actor, 1))
    self.transition_model = TransitionModel(latent_size, action_size, num_sample_trajectories, hidden_size, device)
    self.transition_mll = DeepApproximateMLL(VariationalELBO(self.transition_model.likelihood, self.transition_model, 1))
    self.critic = ValueNetwork(latent_size, num_sample_trajectories, hidden_size)
    self.q_network = QNetwork(latent_size, num_sample_trajectories, action_size, hidden_size)
    self.recurrent_gp = recurrent_gp
    self.num_sample_trajectories = num_sample_trajectories
    self.lagging_size = lagging_size
    self.num_gp_likelihood_samples = num_gp_likelihood_samples

  def forward(self, lagging_states, lagging_actions, device):
    current_state = lagging_states[-1].view(1, self.latent_size)
    imagined_reward = self.imaginary_rollout(lagging_states, lagging_actions, self.num_sample_trajectories).to(device=device)
    embedding = torch.cat([current_state,imagined_reward], dim=-1)
    policy_dist = self.actor(embedding)
    value = self.critic(embedding)
    return policy_dist, value, embedding
  
  def imaginary_rollout(self, lagging_states, lagging_actions, num_sample_trajectories):
    self.recurrent_gp.eval()
    with torch.no_grad():
      with gpytorch.settings.num_likelihood_samples(self.num_gp_likelihood_samples):
        rewards = self.recurrent_gp(
          torch.flatten(lagging_states).unsqueeze(dim=0).expand(num_sample_trajectories, self.lagging_size * self.latent_size).unsqueeze(dim=0),
          lagging_actions.unsqueeze(dim=0).expand(num_sample_trajectories, self.lagging_size, self.action_size).unsqueeze(dim=0)
        ).sample().mean(dim=0)
    self.recurrent_gp.train()
    return rewards
  
  def act(self, prior_states, prior_actions, device=None):
    # to do: consider lagging actions and states for the first action actor, basically fake lagging actions and states before the episode starts
    policy_dist, value, embedding = self.forward(prior_states, prior_actions, device)
    policy_action = policy_dist.rsample().mean(dim=0)
    policy_log_prob = policy_dist.log_prob(policy_action)
    policy_mll_loss = -self.policy_mll(policy_dist, policy_action)
    transition_dist = self.transition_model(embedding, policy_action)
    normalized_policy_action = torch.tanh(policy_action) * torch.tensor(self.action_scale).to(device=device) + torch.tensor(self.action_bias).to(device=device)
    normalized_policy_action = torch.clamp(normalized_policy_action, min=torch.tensor(self.min_action).to(device=device), max=torch.tensor(self.max_action).to(device=device))
    q_value = self.q_network(embedding, policy_action)
    return normalized_policy_action, policy_log_prob, policy_mll_loss, value, q_value, transition_dist
