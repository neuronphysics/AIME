from math import inf
import torch
from torch import jit
import torch.nn as nn
from torch.nn import functional as F


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

'''
class AIMEPlanner(nn.Module):
  
  def __init__(self, action_size, planning_horizon, lagging_size, state_size, optimisation_iters, recurrent_gp, min_action=-inf, max_action=inf):
    super().__init__()
    self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
    self.planning_horizon = planning_horizon
    self.recurrent_gp = recurrent_gp
    self.lagging_size = lagging_size
    self.state_size = state_size
  
  def forward(self, prior_states, prior_actions):
    rewards, posterior_actions, posterior_states = self.recurrent_gp(
      torch.flatten(prior_states).unsqueeze(dim=0).expand(50, self.lagging_size * self.state_size).unsqueeze(dim=0),
      prior_actions.unsqueeze(dim=0).expand(50, self.lagging_size, self.action_size).unsqueeze(dim=0)
    ) 
    rewards = rewards.squeeze(dim=0).squeeze(-1)
    posterior_actions = posterior_actions.squeeze(dim=1)
    posterior_states = posterior_states.squeeze(dim=1)
    max_index = torch.argmax(rewards)
    return posterior_actions[0][max_index:max_index+1], posterior_states[0][max_index]
'''

class ValueNetwork(nn.Module):
  def __init__(self, latent_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.latent_size = latent_size
    self.fc2 = nn.Linear(latent_size + 1, 1)
  
  def forward(self, x):
    value = self.fc2(x)
    return value

class PolicyNetwork(nn.Module):
  def __init__(self, latent_size, action_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.latent_size = latent_size
    self.action_size = action_size
    self.fc_mean = nn.Linear(latent_size + 1, action_size)
    self.fc_std = nn.Linear(latent_size + 1, action_size)
  
  def forward(self, x):
    policy_mean = self.fc_mean(x)
    policy_std = F.softplus(self.fc_std(x))
    return policy_mean, policy_std

class RolloutEncoder(nn.Module):
  def __init__(self, latent_size, action_size, hidden_size=8):
    super().__init__()
    self.encoder = nn.LSTM(input_size=action_size+latent_size, hidden_size=hidden_size)
  
  def forward(self, imagined_actions, imagined_states):
    x = torch.cat([imagined_actions, imagined_states], dim=-1)
    x = self.encoder(x)
    return x

class ActorCriticPlanner(nn.Module):
  def __init__(self, latent_size, action_size, recurrent_gp, min_action=-inf, max_action=inf, action_noise=0):
    self.action_size, self.action_noise, self.min_action, self.max_action = action_size, min_action, max_action, action_noise
    self.latent_size = latent_size
    self.fc1 = nn.Linear(latent_size + 1, latent_size + 1)
    self.actor = PolicyNetwork(latent_size, action_size)
    self.critic = ValueNetwork(latent_size)
    self.recurrent_gp = recurrent_gp
    self.rollout_encoder = nn.LSTM(input_size=action_size+latent_size, hidden_size=8)
  
  def forward(self, prior_states, prior_actions):
    current_state = prior_states[-1]
    imagined_reward, imagined_actions, imagined_states = self.imaginary_rollout(prior_states, prior_actions)
    rollout_embedding = self.rollout_encoder(imagined_actions, imagined_states)
    hidden = torch.cat([current_state, imagined_reward, rollout_embedding], dim=-1)
    hidden = self.fc1(hidden)
    policy_mean, policy_std = self.actor(hidden)
    policy_action = policy_mean + torch.rand_like(policy_mean) * policy_std
    value = self.critic(hidden)
    return policy_action, value
  
  def imaginary_rollout(self, prior_states, prior_actions, num_sample_trajectories=50):
    with torch.no_grad():
      rewards, posterior_actions, posterior_states = self.recurrent_gp(
        torch.flatten(prior_states).unsqueeze(dim=0).expand(num_sample_trajectories, self.lagging_size * self.state_size).unsqueeze(dim=0),
        prior_actions.unsqueeze(dim=0).expand(num_sample_trajectories, self.lagging_size, self.action_size).unsqueeze(dim=0)
      )
    return rewards, posterior_actions, posterior_states
  
  def act(self, prior_states, prior_actions, explore=False):
    # to do: consider lagging actions and states for the first action actor, basically fake lagging actions and states before the episode starts
    action = [0.5, 0.5, 0]
    if explore:
      action = action + self.action_noise * torch.randn_like(action)
    action.clamp_(min=self.min_action, max=self.max_action)
    return action