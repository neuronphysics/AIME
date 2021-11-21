from math import inf
import torch
from torch import jit
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

import gpytorch

class ValueNetwork(nn.Module):
  def __init__(self, latent_size, num_sample_trajectories, hidden_size):
    super().__init__()
    self.latent_size = latent_size
    self.fc1 = nn.Linear(latent_size + num_sample_trajectories, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 1)
  
  def forward(self, embedding):
    hidden = F.relu(self.fc1(embedding))
    return self.fc2(hidden)

class QNetwork(nn.Module):
  def __init__(self, latent_size, num_sample_trajectories, action_size, hidden_size):
    super().__init__()
    self.latent_size = latent_size
    self.action_size = action_size
    self.fc1 = nn.Linear(latent_size + num_sample_trajectories + action_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, 1)
  
  def forward(self, embedding, action):
    hidden = F.relu(self.fc1(torch.cat([embedding, action], dim=-1)))
    return self.fc2(hidden)

class PolicyNetwork(nn.Module):
  def __init__(self, latent_size, action_size, num_sample_trajectories, hidden_size):
    super().__init__()
    self.latent_size = latent_size
    self.action_size = action_size
    self.fc1 = nn.Linear(latent_size + num_sample_trajectories, hidden_size)
    self.fc_mean = nn.Linear(hidden_size, action_size)
    self.fc_std = nn.Linear(hidden_size, action_size)
  
  def forward(self, embedding):
    hidden = F.relu(self.fc1(embedding))
    policy_mean = self.fc_mean(hidden)
    policy_std = F.softplus(self.fc_std(hidden))
    return policy_mean, policy_std

class TransitionNetwork(nn.Module):
  def __init__(self, latent_size, action_size):
    super().__init__()
    self.latent_size = latent_size
    self.action_size = action_size
    self.fc_mean = nn.Linear(latent_size + action_size, latent_size)
    self.fc_std = nn.Linear(latent_size + action_size, latent_size)
  
  def forward(self, state, action):
    x = torch.cat([state, action], dim=-1)
    next_state_mean = self.fc_mean(x)
    next_state_std = F.softplus(self.fc_std(x))
    return next_state_mean, next_state_std

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
  def __init__(self, lagging_size, latent_size, action_size, recurrent_gp, min_action, max_action, num_sample_trajectories, hidden_size):
    super().__init__()
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.action_scale = (self.max_action - self.min_action) / 2
    self.action_bias = (self.max_action + self.min_action) / 2
    self.latent_size = latent_size
    self.actor = PolicyNetwork(latent_size, action_size, num_sample_trajectories, hidden_size)
    self.critic = ValueNetwork(latent_size, num_sample_trajectories, hidden_size)
    self.q_network = QNetwork(latent_size, num_sample_trajectories, action_size, hidden_size)
    self.recurrent_gp = recurrent_gp
    self.num_sample_trajectories = num_sample_trajectories
    self.lagging_size = lagging_size

  def forward(self, lagging_states, lagging_actions, device):
    current_state = lagging_states[-1].view(1, self.latent_size)
    imagined_reward = self.imaginary_rollout(lagging_states, lagging_actions, self.num_sample_trajectories).to(device=device)
    embedding = torch.cat([current_state,imagined_reward.squeeze(dim=-2)], dim=-1)
    policy_mean, policy_std = self.actor(embedding)
    value = self.critic(embedding)
    return policy_mean, policy_std, value, embedding
  
  def imaginary_rollout(self, lagging_states, lagging_actions, num_sample_trajectories):
    self.recurrent_gp.eval()
    with torch.no_grad():
      with gpytorch.settings.num_likelihood_samples(1):
        rewards = self.recurrent_gp(
          torch.flatten(lagging_states).unsqueeze(dim=0).expand(num_sample_trajectories, self.lagging_size * self.latent_size).unsqueeze(dim=0),
          lagging_actions.unsqueeze(dim=0).expand(num_sample_trajectories, self.lagging_size, self.action_size).unsqueeze(dim=0)
        )
    self.recurrent_gp.train()
    return rewards.rsample()
  
  def act(self, prior_states, prior_actions, device=None):
    # to do: consider lagging actions and states for the first action actor, basically fake lagging actions and states before the episode starts
    policy_mean, policy_std, value, embedding = self.forward(prior_states, prior_actions, device)
    policy_dist = Normal(policy_mean, policy_std)
    policy_action = policy_dist.rsample()
    policy_log_prob = policy_dist.log_prob(policy_action)
    normalized_policy_action = torch.tanh(policy_action) * torch.tensor(self.action_scale).to(device=device) + torch.tensor(self.action_bias).to(device=device)
    normalized_policy_action = torch.min(torch.max(normalized_policy_action, torch.tensor(self.min_action).to(device=device)), torch.tensor(self.max_action).to(device=device))
    q_value = self.q_network(embedding, policy_action)
    return normalized_policy_action, policy_log_prob, value, q_value
