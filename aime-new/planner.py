from math import inf
import torch
from torch import jit
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

import gpytorch

from dlgpd.gp import build_gp

# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(jit.ScriptModule):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  @jit.script_method
  def forward(self, belief, state):
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
    for _ in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
      actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
      # Sample next states
      beliefs, states, _, _ = self.transition_model(state, actions, belief)
      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
      # Update belief with new means and standard deviations
      action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
    # Return first action mean µ_t
    return action_mean[0].squeeze(dim=1)


class ValueNetwork(nn.Module):
  def __init__(self, latent_size, hidden_size):
    super().__init__()
    self.latent_size = latent_size
    self.fc = nn.Linear(latent_size + hidden_size, 1)
  
  def forward(self, x):
    value = self.fc(x)
    return value

class QNetwork(nn.Module):
  def __init__(self, latent_size, action_size):
    super().__init__()
    self.latent_size = latent_size
    self.action_size = action_size
    self.fc = nn.Linear(latent_size + action_size, 1)
  
  def forward(self, state, action):
    q_value = self.fc(torch.cat([state, action], dim=-1))
    return q_value

class PolicyNetwork(nn.Module):
  def __init__(self, latent_size, action_size, hidden_size):
    super().__init__()
    self.latent_size = latent_size
    self.action_size = action_size
    self.fc_mean = nn.Linear(latent_size + hidden_size, action_size)
    self.fc_std = nn.Linear(latent_size + hidden_size, action_size)
  
  def forward(self, x):
    policy_mean = self.fc_mean(x)
    policy_std = F.softplus(self.fc_std(x))
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
  def __init__(self, lagging_size, latent_size, action_size, recurrent_gp, min_action, max_action, action_noise, num_sample_trajectories=10, hidden_size=1, temperature=1):
    super().__init__()
    self.action_size, self.action_noise, self.min_action, self.max_action = action_size, action_noise, min_action, max_action
    self.action_scale = (self.max_action - self.min_action) / 2
    self.action_bias = (self.max_action + self.min_action) / 2
    self.latent_size = latent_size
    self.fc1 = nn.Linear(latent_size + (1+action_size+hidden_size)*num_sample_trajectories, latent_size + 1)
    self.actor = PolicyNetwork(latent_size, action_size, num_sample_trajectories)
    self.critic = ValueNetwork(latent_size, num_sample_trajectories)
    self.q_network = QNetwork(latent_size, action_size)
    self.transition_gp = build_gp(latent_size+action_size, latent_size)
    self.recurrent_gp = recurrent_gp
    self.rollout_encoder = RolloutEncoder(latent_size, action_size, hidden_size, num_sample_trajectories)
    self.num_sample_trajectories = num_sample_trajectories
    self.hidden_size = hidden_size
    self.lagging_size = lagging_size
    self.temperature = temperature

  def forward(self, lagging_states, lagging_actions):
    current_state = lagging_states[-1].view(1, self.latent_size)
    imagined_reward = self.imaginary_rollout(lagging_states, lagging_actions, self.num_sample_trajectories)
    embedding = torch.cat([current_state,imagined_reward.squeeze(dim=-2)], dim=-1)
    policy_mean, policy_std = self.actor(embedding)
    value = self.critic(embedding)
    return policy_mean, policy_std, value, current_state
  
  def imaginary_rollout(self, lagging_states, lagging_actions, num_sample_trajectories):
    self.recurrent_gp.eval()
    with torch.no_grad():
      with gpytorch.settings.num_likelihood_samples(1):
        rewards = self.recurrent_gp(
          torch.flatten(lagging_states).unsqueeze(dim=0).expand(num_sample_trajectories, self.lagging_size * self.latent_size).unsqueeze(dim=0),
          lagging_actions.unsqueeze(dim=0).expand(num_sample_trajectories, self.lagging_size, self.action_size).unsqueeze(dim=0)
        )
    self.recurrent_gp.train()
    return rewards.rsample().cuda()
  
  def act(self, prior_states, prior_actions, explore=False):
    # to do: consider lagging actions and states for the first action actor, basically fake lagging actions and states before the episode starts
    policy_mean, policy_std, value, current_state= self.forward(prior_states, prior_actions)
    policy_dist = Normal(policy_mean, policy_std)
    policy_action = policy_dist.rsample()
    policy_log_prob = policy_dist.log_prob(policy_action)
    normalized_policy_action = torch.tanh(policy_action) * torch.Tensor(self.action_scale) + torch.Tensor(self.action_bias)
    normalized_policy_action = torch.min(torch.max(normalized_policy_action, torch.Tensor(self.min_action)), torch.Tensor(self.max_action))
    q_value = self.q_network(current_state, policy_action)
    return normalized_policy_action, policy_log_prob, value, q_value
