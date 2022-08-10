import utils_planner as utils
from math import inf
import torch
from torch import jit
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import transforms as tT
from torch.distributions.transformed_distribution import TransformedDistribution
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import datetime
import re
from torch.testing._internal.common_utils import TestCase
import collections
import numpy as np
import os
import gin
import gym
import mujoco_py
from absl import app
from absl import flags
from absl import logging

import time
import alf_gym_wrapper
import importlib  

LOG_STD_MIN = -5
LOG_STD_MAX = 0

def get_spec_means_mags(spec):
  means = (spec.maximum + spec.minimum) / 2.0
  mags = (spec.maximum - spec.minimum) / 2.0
  means = Variable(means.type(torch.FloatTensor), requires_grad=False)
  mags  = Variable(mags.type(torch.FloatTensor), requires_grad=False)
  return means, mags

class Split(torch.nn.Module):
    """
    models a split in the network. works with convolutional models (not FC).
    specify out channels for the model to divide by n_parts.
    """
    def __init__(self, module, n_parts: int, dim=1):
        super().__init__()
        self._n_parts = n_parts
        self._dim = dim
        self._module = module

    def forward(self, inputs):
        output = self._module(inputs)
        chunk_size = output.shape[self._dim] // self._n_parts
        return torch.split(output, chunk_size, dim=self._dim)
      
###############################################
##################  Networks  #################
###############################################
class ActorNetwork(nn.Module):
  """Actor network."""

  def __init__(
      self,
      latent_spec,
      action_spec,
      fc_layer_params=(),
      ):
    super(ActorNetwork, self).__init__()
    self._action_spec = action_spec
    self._layers = nn.ModuleList()
    for hidden_size in fc_layer_params:
        if len(self._layers)==0:
           self._layers.append(nn.Linear(latent_spec.size(0), hidden_size))
        else:
           self._layers.append(nn.Linear(hidden_size, hidden_size))
        self._layers.append(nn.ReLU())
    output_layer = nn.Linear(hidden_size,
        self._action_spec.shape[0] * 2
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
      return self._action_spec

  def _get_outputs(self, state):
      h = state
      for l in self._layers:
          h = l(h)
      self._mean_logvar_layers = Split(
         self._layers[-1],
         parts=2,
      )
      mean, log_std = self._mean_logvar_layers(h)
      a_tanh_mode = torch.tanh(mean) * self._action_mags + self._action_means
      log_std = torch.tanh(log_std)
      log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
      std = torch.exp(log_std)
      #base_distribution = torch.normal(0.0, 1.0)
      #transforms = torch.distributions.transforms.ComposeTransform([torch.distributions.transforms.AffineTransform(loc=self._action_means, scale=self._action_mag, event_dim=mean.shape[-1]), torch.nn.Tanh(),torch.distributions.transforms.AffineTransform(loc=mean, scale=std, event_dim=mean.shape[-1])])
      #a_distribution = torch.distributions.transformed_distribution.TransformedDistribution(base_distribution, transforms)
      a_distribution = TransformedDistribution(
                        base_distribution=Normal(loc=torch.full_like(mean, 0), 
                                                 scale=torch.full_like(mean, 1)), 
                        transforms=tT.ComposeTransform([
                                   tT.AffineTransform(loc=self._action_means, scale=self._action_mag, event_dim=mean.shape[-1]), 
                                   tT.TanhTransform(),
                                   tT.AffineTransform(loc=mean, scale=std, event_dim=mean.shape[-1])]))
      #https://www.ccoderun.ca/programming/doxygen/pytorch/classtorch_1_1distributions_1_1transformed__distribution_1_1TransformedDistribution.html
      return a_distribution, a_tanh_mode

  def get_log_density(self, state, action):
    a_dist, _ = self._get_outputs(state)
    log_density = a_dist.log_prob(action)
    return log_density

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weight[0])
    return w_list

  def __call__(self, state):
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample()
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample_n(self, state, n=1):
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample(n)
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample(self, state):
    return self.sample_n(state, n=1)[1][0]

class CriticNetwork(nn.Module):
    """Critic Network."""
    def __init__(
      self,
      latent_spec,
      action_spec,
      fc_layer_params=(),
      ):
      super(CriticNetwork, self).__init__()
      self._action_spec = action_spec
      self._layers = nn.ModuleList()
      for hidden_size in fc_layer_params:
          if len(self._layers)==0:
              self._layers.append(nn.Linear(latent_spec.size(0)+action_spec(0), hidden_size))
          else:
              self._layers.append(nn.Linear(hidden_size, hidden_size))
          self._layers.append(nn.ReLU())
      output_layer = nn.Linear(hidden_size,1)
      self._layers.append(output_layer)
  
    def forward(self, embedding):
        hidden = embedding
        for l in self._layers:
            hidden = l(hidden)
        return hidden
############################
##From utils.py
class Flags(object):

  def __init__(self, **kwargs):
    for key, val in kwargs.items():
      setattr(self, key, val)

def get_modules(model_params, action_spec):
  """Gets pytorch modules for Q-function, policy, and discriminator."""
  model_params, n_q_fns = model_params
  if len(model_params) == 1:
    model_params = tuple([model_params[0]] * 3)
  elif len(model_params) < 3:
    raise ValueError('Bad model parameters %s.' % model_params)
  def q_net_factory():
    return CriticNetwork(
        fc_layer_params=model_params[0])
  def p_net_factory():
    return ActorNetwork(
        action_spec,
        fc_layer_params=model_params[1])
  def c_net_factory():
    return CriticNetwork(
        fc_layer_params=model_params[2])
  modules = Flags(
      q_net_factory=q_net_factory,
      p_net_factory=p_net_factory,
      c_net_factory=c_net_factory,
      n_q_fns=n_q_fns,
      )
  return modules
#######################################
################ AGENT ################
#######################################
ALPHA_MAX = 500.0

class AgentModule(nn.Module):
  """Pytorch module for BRAC dual agent."""
  def __init__(
      self,
      modules=None,
      ):
    super(AgentModule, self).__init__()
    self._modules = modules
    self._build_modules()
    
  def _build_modules(self):
    self._q_nets = []
    n_q_fns = self._modules.n_q_fns
    for _ in range(n_q_fns):
      self._q_nets.append(
          [self._modules.q_net_factory(),  # Learned Q-value.
           self._modules.q_net_factory(),]  # Target Q-value.
          )
    self._p_net = self._modules.p_net_factory()
    self._c_net = self._modules.c_net_factory()
    self._alpha_var = torch.tensor(1.0, requires_grad=True)
    self._alpha_entropy_var = torch.tensor(1.0, requires_grad=True)

  def get_alpha(self, alpha_max=ALPHA_MAX):
    return utils.clip_v2(
        self._alpha_var, 0.0, alpha_max)

  def get_alpha_entropy(self):
    return utils.relu_v2(self._alpha_entropy_var)

  def assign_alpha(self, alpha):
    self._alpha_var=torch.tensor(alpha, requires_grad=True)

  def assign_alpha_entropy(self, alpha):
    self._alpha_entropy_var=torch.tensor(alpha, requires_grad=True)

  @property
  def a_variables(self):
    return [self._alpha_var]

  @property
  def ae_variables(self):
    return [self._alpha_entropy_var]

  @property
  def q_nets(self):
    return self._q_nets

  @property
  def q_source_weights(self):
      q_weights = []
      for q_net, _ in self._q_nets:
         for _, net in q_net._modules.values():
             q_weights += net.weight.data
      return q_weights

  @property
  def q_target_weights(self):
    q_weights = []
    for _, q_net in self._q_nets:
        for k, net in q_net._modules.values():
            q_weights += net.weight.data
    return q_weights

  @property
  def q_source_variables(self):
    vars_ = []
    for q_net, _ in self._q_nets:
        vars = q_net.parameters()
        for v in vars:
            if v.requires_grad:
               vars_ += v
    return tuple(vars_)

  @property
  def q_target_variables(self):
    vars_ = []
    for _, q_net in self._q_nets:
        vars = q_net.parameters()
        for v in vars:
            if v.requires_grad:
               vars_ += v
    return tuple(vars_)

  @property
  def p_net(self):
    return self._p_net

  def p_fn(self, s):
    return self._p_net(s)

  @property
  def p_weights(self):
    return self._p_net.weight.data

  @property
  def p_variables(self):
    vars = self._p_net.parameters()
    vars_=[]
    for v in vars:
        if v.requires_grad:
           vars_.append(v)   
    return vars_

  @property
  def c_net(self):
    return self._c_net

  @property
  def c_weights(self):
    return self._c_net.weight.data

  @property
  def c_variables(self):
    vars = self._c_net.parameters()
    vars_=[]
    for v in vars:
        if v.requires_grad:
           vars_.append(v)   
    return vars_

class Agent(object):
  """Class for learning policy and interacting with environment."""

  def __init__(
      self,
      latent_spec=None,
      action_spec=None,
      time_step_spec=None,
      modules=None,
      optimizers= ((0.001, 0.5, 0.99),),
      batch_size=64,
      weight_decays=(0.0,),
      update_freq=1,
      update_rate=0.005,
      discount=0.99,
      train_data=None,
      resume=False, 
      device=None
      ):
    self._latent_spec = latent_spec
    self._action_spec = action_spec
    self._time_step_spec = time_step_spec
    self._modules = modules
    self._optimizers = optimizers
    self._batch_size = batch_size
    self._weight_decays = weight_decays
    self._train_data = train_data
    self._update_freq = update_freq
    self._update_rate = update_rate
    self._discount = discount
    self._resume = resume 
    self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
    self._build_agent()
    directory = os.getcwd()
    checkpoint_dir=directory+"/run"
    self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

  def _build_agent(self):
    """Builds agent components."""
    self._build_fns()
    self._build_optimizers()
    self._global_step = torch.tensor(0.0, requires_grad=True)
    self._train_info = collections.OrderedDict()
    self._checkpointer = self._build_checkpointer()
    self._test_policies = collections.OrderedDict()
    self._build_test_policies()
    self._online_policy = self._build_online_policy()
    train_batch = self._get_train_batch()
    self._init_vars(train_batch)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)

  def _get_vars(self):
    return []

  def _build_optimizers(self):
     opt = self._optimizers[0]
     self._optimizer = torch.optim.Adam(
            self._agent_module.parameters(),
            lr= opt[0],
            betas=(opt[1], opt[2]),
        )
    

  def _build_loss(self, batch):
    raise NotImplementedError

  def _update_target_vars(self):
        # requires self._vars_learning and self._vars_target as state_dict`s
        for var_name, var_t in self._vars_target.items():
            updated_val = (self._update_rate
                    * self._vars_learning[var_name].data
                    + (1.0 - self._update_rate) * var_t.data)
            var_t.data.copy_(updated_val)

  def _build_test_policies(self):
    raise NotImplementedError

  def _build_online_policy(self):
    return None
  
  def _random_policy_fn(self, state):
      return self._action_spec.sample(), None


  @property
  def test_policies(self):
    return self._test_policies

  @property
  def online_policy(self):
    return self._online_policy

  def _get_train_batch(self):
    """Samples and constructs batch of transitions."""
    batch_indices = np.random.choice(self._train_data.size, self._batch_size)
    batch_ = self._train_data.get_batch(batch_indices)
    transition_batch = batch_
    batch = dict(
        s1=transition_batch.s1,
        s2=transition_batch.s2,
        r=transition_batch.reward,
        dsc=transition_batch.discount,
        a1=transition_batch.a1,
        a2=transition_batch.a2,
        )
    return batch

            
  def train_step(self):
      train_batch = self._get_train_batch()
      loss = self._build_loss(train_batch)
      self._optimizer.zero_grad()
      loss.backward()
      self._optimizer.step()
      self._global_step += 1
      if self._global_step % self._update_freq == 0:
          self._update_target_vars()
    

  def _init_vars(self, batch):
      pass

  def _get_source_target_vars(self):
      return [], []

  def _update_target_fns(self, source_vars, target_vars):
    utils.soft_variables_update(
        source_vars,
        target_vars,
        tau=self._update_rate)

  def print_train_info(self):
      summary_str = utils.get_summary_str(
                step=self._global_step, info=self._train_info)
      logging.info(summary_str)
    

  def write_train_summary(self, summary_writer):
    info = self._train_info
    step = self._global_step.numpy()
    utils.write_summary(summary_writer, step, info)

  def _build_checkpointer(self):
      pass

  def _load_checkpoint(self):
      pass

  @property
  def global_step(self):
    return self._global_step.numpy()
  
################# Policies #################
class GaussianRandomSoftPolicy(nn.Module):
  """Adds Gaussian noise to actor's action."""

  def __init__(self, a_network, std=0.1, clip_eps=1e-3):
    super(GaussianRandomSoftPolicy, self).__init__()
    self._a_network = a_network
    self._std = std
    self._clip_eps = clip_eps

  def __call__(self, observation, state=()):
    action = self._a_network(observation)[1]
    noise = torch.normal(mean=action.shape, std=self._std)
    action = action + noise
    spec = self._a_network.action_spec
    action = torch.clamp(action, spec.minimum + self._clip_eps,
                              spec.maximum - self._clip_eps)
    return action, state
  
class DeterministicSoftPolicy(nn.Module):
  """Returns mode of policy distribution."""

  def __init__(self, a_network):
    super(DeterministicSoftPolicy, self).__init__()
    self._a_network = a_network


  def __call__(self, latent_states):
    action = self._a_network(latent_states)[0]
    return action

class RandomSoftPolicy(nn.Module):
  """Returns sample from policy distribution."""

  def __init__(self, a_network):
    super(RandomSoftPolicy, self).__init__()
    self._a_network = a_network


  def __call__(self, latent_states):
    action = self._a_network(latent_states)[1]
    return action

class MaxQSoftPolicy(nn.Module):
  """Samples a few actions from policy, returns the one with highest Q-value."""

  def __init__(self, a_network, q_network, n=10):
    super(MaxQSoftPolicy, self).__init__()
    self._a_network = a_network
    self._q_network = q_network
    self._n = n

  def __call__(self, latent_state):
    batch_size = latent_state.shape[0]
    actions = self._a_network.sample_n(latent_state, self._n)[1]
    actions_ = torch.reshape(actions, [self._n * batch_size, -1])
    states_ = torch.tile(latent_state[None], (self._n, 1, 1))
    states_ = torch.reshape(states_, [self._n * batch_size, -1])
    qvals = self._q_network(states_, actions_)
    qvals = torch.reshape(qvals, [self._n, batch_size])
    a_indices = torch.argmax(qvals, dim=0)
    gather_indices = torch.stack(
        [a_indices, torch.range(batch_size, dtype=torch.int64)], dim=-1)
    action = utils.gather_nd(actions, gather_indices)
    return action
#############################################
################ D2E Agent ################## 
#############################################
gin.clear_config()
gin.config._REGISTRY.clear()
@gin.configurable
class D2EAgent(Agent):
  """D2E dual agent class."""

  def __init__(
      self,
      alpha=1.0,
      alpha_max=ALPHA_MAX,
      train_alpha=False,
      value_penalty=True,
      target_divergence=0.0,
      alpha_entropy=0.0,
      train_alpha_entropy=False,
      target_entropy=None,
      divergence_name='kl',
      warm_start=2000,
      c_iter=3,
      ensemble_q_lambda=1.0,
      **kwargs):
    self._alpha = alpha
    self._alpha_max = alpha_max
    self._train_alpha = train_alpha
    self._value_penalty = value_penalty
    self._target_divergence = target_divergence
    self._divergence_name = divergence_name
    self._train_alpha_entropy = train_alpha_entropy
    self._alpha_entropy = alpha_entropy
    self._target_entropy = target_entropy
    self._warm_start = warm_start
    self._c_iter = c_iter
    self._ensemble_q_lambda = ensemble_q_lambda
    super(Agent, self).__init__(**kwargs)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)
    self._q_fns = self._agent_module.q_nets
    self._p_fn = self._agent_module.p_fn
    self._c_fn = self._agent_module.c_net
    self._divergence = utils.get_divergence(
        name=self._divergence_name)
    self._agent_module.assign_alpha(self._alpha)
    if self._target_entropy is None:
      self._target_entropy = - self._action_spec.shape[0]
    self._get_alpha_entropy = self._agent_module.get_alpha_entropy
    self._agent_module.assign_alpha_entropy(self._alpha_entropy)

  def _get_alpha(self):
    return self._agent_module.get_alpha(
        alpha_max=self._alpha_max)

  def _get_q_vars(self):
    return self._agent_module.q_source_variables

  def _get_p_vars(self):
    return self._agent_module.p_variables

  def _get_c_vars(self):
    return self._agent_module.c_variables

  def _get_q_weight_norm(self):
    weights = self._agent_module.q_source_weights
    norms = []
    for w in weights:
      norm = torch.sum(torch.square(w))
      norms.append(norm)
    return torch.stack(norms).sum(dim=0)

  def _get_p_weight_norm(self):
    weights = self._agent_module.p_weights
    norms = []
    for w in weights:
      norm = torch.sum(torch.square(w))
      norms.append(norm)
    return torch.stack(norms).sum(dim=0)

  def _get_c_weight_norm(self):
    weights = self._agent_module.c_weights
    norms = []
    for w in weights:
      norm = torch.sum(torch.square(w))
      norms.append(norm)
    return torch.stack(norms).sum(dim=0)

  def ensemble_q(self, qs):
    #compute the ensemble value of different Q networks
    lambda_ = self._ensemble_q_lambda
    return (lambda_ *torch.min(qs, dim=-1)
            + (1 - lambda_) * torch.max(qs, dim=-1))

  def _ensemble_q2_target(self, q2_targets):
    return self.ensemble_q(q2_targets)

  def _ensemble_q1(self, q1s):
    return self.ensemble_q(q1s)

  def _build_q_loss(self, batch):
    s1 = batch['s1']
    s2 = batch['s2']
    a1 = batch['a1']
    a2_b = batch['a2']
    r = batch['r']
    dsc = batch['dsc']
    _, a2_p, log_pi_a2_p = self._p_fn(s2)
    q2_targets = []
    q1_preds = []
    for q_fn, q_fn_target in self._q_fns:
      q2_target_ = q_fn_target(s2, a2_p)
      q1_pred = q_fn(s1, a1)
      q1_preds.append(q1_pred)
      q2_targets.append(q2_target_)
    q2_targets = torch.stack(q2_targets, dim=-1)
    q2_target = self._ensemble_q2_target(q2_targets)
    div_estimate = self._divergence.dual_estimate(
        s2, a2_p, a2_b, self._c_fn)
    v2_target = q2_target - self._get_alpha_entropy() * log_pi_a2_p
    if self._value_penalty: #Equation (5)
       v2_target = v2_target - self._get_alpha() * div_estimate
    with torch.no_grad(): #Q(s,a)=R(s,a)+discount*v(s')
         q1_target = r + dsc * self._discount * v2_target
    q_losses = []
    for q1_pred in q1_preds: #Equation (6)
      q_loss_ = torch.mean(torch.square(q1_pred - q1_target))
      q_losses.append(q_loss_)
    q_loss = torch.stack(q_losses).sum(dim=0)
    q_w_norm = self._get_q_weight_norm()
    norm_loss = self._weight_decays[0] * q_w_norm
    loss = q_loss + norm_loss

    info = collections.OrderedDict()
    info['q_loss'] = q_loss
    info['q_norm'] = q_w_norm
    info['r_mean'] = torch.mean(r)
    info['dsc_mean'] = torch.mean(dsc)
    info['q2_target_mean'] = torch.mean(q2_target)
    info['q1_target_mean'] = torch.mean(q1_target)

    return loss, info

  def _build_p_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    _, a_p, log_pi_a_p = self._p_fn(s)
    q1s = []
    for q_fn, _ in self._q_fns:
      q1_ = q_fn(s, a_p)
      q1s.append(q1_)
    q1s = torch.stack(q1s, dim=-1)
    q1 = self._ensemble_q1(q1s)
    div_estimate = self._divergence.dual_estimate(
        s, a_p, a_b, self._c_fn)
    q_start = torch.gt(self._global_step, self._warm_start).type(torch.float32)
    #Equation 7
    p_loss = torch.mean(
        self._get_alpha_entropy() * log_pi_a_p
        + self._get_alpha() * div_estimate
        - q1 * q_start)
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays[1] * p_w_norm
    loss = p_loss + norm_loss

    info = collections.OrderedDict()
    info['p_loss'] = p_loss
    info['p_norm'] = p_w_norm

    return loss, info

  def _build_c_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    _, a_p, _ = self._p_fn(s)
    c_loss = self._divergence.dual_critic_loss(
        s, a_p, a_b, self._c_fn)
    c_w_norm = self._get_c_weight_norm()
    norm_loss = self._weight_decays[2] * c_w_norm
    loss = c_loss + norm_loss

    info = collections.OrderedDict()
    info['c_loss'] = c_loss
    info['c_norm'] = c_w_norm

    return loss, info

  def _build_a_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    _, a_p, _ = self._p_fn(s)
    alpha = self._get_alpha()
    div_estimate = self._divergence.dual_estimate(
        s, a_p, a_b, self._c_fn)
    a_loss = - torch.mean(alpha * (div_estimate - self._target_divergence))

    info = collections.OrderedDict()
    info['a_loss'] = a_loss
    info['alpha'] = alpha
    info['div_mean'] = torch.mean(div_estimate)
    info['div_std'] = torch.std(div_estimate)

    return a_loss, info

  def _build_ae_loss(self, batch):
    s = batch['s1']
    _, _, log_pi_a = self._p_fn(s)
    alpha = self._get_alpha_entropy()
    ae_loss = torch.mean(alpha * (- log_pi_a - self._target_entropy))

    info = collections.OrderedDict()
    info['ae_loss'] = ae_loss
    info['alpha_entropy'] = alpha

    return ae_loss, info

  def _get_source_target_vars(self):
    return (self._agent_module.q_source_variables,
            self._agent_module.q_target_variables)

  def _build_optimizers(self):
    opts = self._optimizers
    if len(opts) == 1:
      opts = tuple([opts[0]] * 4)
    elif len(opts) < 4:
      raise ValueError('Bad optimizers %s.' % opts)
    if len(self._weight_decays) == 1:
      self._weight_decays = tuple([self._weight_decays[0]] * 3)
      
    self._q_optimizer = utils.OptMirrorAdam(self._get_q_vars(),lr=opts[0][0], betas=(opts[0][1],opts[0][2]), weight_decay=self._weight_decays[0])
    self._p_optimizer = utils.OptMirrorAdam(self._get_p_vars(),lr=opts[1][0], betas=(opts[1][1],opts[1][2]), weight_decay=self._weight_decays[1])
    self._c_optimizer = utils.OptMirrorAdam(self._get_c_vars(),lr=opts[2][0], betas=(opts[2][1],opts[2][2]), weight_decay=self._weight_decays[2])
    self._a_optimizer = torch.optim.Adam(self._a_vars, lr=opts[3][0], betas=(opts[3][1],opts[3][2]))
    self._ae_optimizer = torch.optim.Adam(self._ae_vars, lr=opts[3][0], betas=(opts[3][1],opts[3][2]))

  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    if torch.equal(self._global_step % self._update_freq, 0):
      source_vars, target_vars = self._get_source_target_vars()
      self._update_target_fns(source_vars, target_vars)
    # Update policy network parameter
    #https://bit.ly/3Bno0GC
    # policy network's update should be done before updating q network, or there will make some errors
    self._p_optimizer.zero_grad()
    policy_loss,_=self._build_p_loss(batch)
    policy_loss.backward(retain_graph=True)
    self._p_optimizer.step()
    # Update q networks parameter
    self._q_optimizer.zero_grad()
    q_losses,q_info= self._build_q_loss(batch)
    q_losses.backward()
    self._q_optimizer.step()
    #Update critic network parameter
    self._c_optimizer.zero_grad()
    critic_loss,_= self._build_c_loss( batch)
    critic_loss.backward()
    self._c_optimizer.step()
    
    if self._train_alpha:
       self._a_optimizer.zero_grad()
       a_loss,a_info = self._build_a_loss( batch)
       a_loss.backward()
       self._a_optimizer.step()
    if self._train_alpha_entropy:
       self._ae_optimizer.zero_grad()
       ae_loss,ae_info = self._build_ae_loss( batch)
       ae_loss.backward()
       self._ae_optimizer.step()
    #1)policy loss
    info["policy_loss"]=policy_loss.cpu().item()
    #2)Q loss
    info["Q_loss"]=q_losses.cpu().item()
    info["reward_mean"]= q_info["r_mean"].cpu().item()
    info["dsc_mean"]= q_info["dsc_mean"].cpu().item()
    info["q1_target_mean"]=q_info["q1_target_mean"].cpu().item()
    info["q2_target_mean"]=q_info["q2_target_mean"].cpu().item()
    #7) critic loss
    info["critic_loss"]=critic_loss.cpu().item()
    #8)alpha loss
    if self._train_alpha:
       info["alpha_loss"]=a_loss.cpu().item()
       self._agent_module.assign_alpha(a_info["alpha"])
    #10)alpha entropy loss
    if self._train_alpha_entropy:
       info["alpha_entropy_loss"]=ae_loss.cpu().item()
       self._agent_module.assign_alpha_entropy(ae_info["alpha_entropy"])
    return info
  
  def _extra_c_step(self, batch):
      self._c_optimizer.zero_grad()
      critic_loss,_ = self._build_c_loss( batch)
      critic_loss.backward()
      self._c_optimizer.step()

  def train_step(self):
    train_batch = self._get_train_batch()
    info = self._optimize_step(train_batch)
    for _ in range(self._c_iter - 1):
      train_batch = self._get_train_batch()
      self._extra_c_step(train_batch)
    for key, val in info.items():
      self._train_info[key] = val.numpy()
    self._global_step+=1


  def _build_test_policies(self):
    policy = DeterministicSoftPolicy(
        a_network=self._agent_module.p_net)
    self._test_policies['main'] = policy
    policy = MaxQSoftPolicy(
        a_network=self._agent_module.p_net,
        q_network=self._agent_module.q_nets[0][0],
        )
    self._test_policies['max_q'] = policy

  def _build_online_policy(self):
    return RandomSoftPolicy(
        a_network=self._agent_module.p_net,
        )

  def _init_vars(self, batch):
    self._build_q_loss(batch)
    self._build_p_loss(batch)
    self._build_c_loss(batch)
    self._q_vars = self._get_q_vars()
    self._p_vars = self._get_p_vars()
    self._c_vars = self._get_c_vars()
    self._a_vars = self._agent_module.a_variables
    self._ae_vars = self._agent_module.ae_variables

  def _build_checkpointer(self):
      checkpoint = {
            "policy_net": self._p_fn.state_dict(),
            "critic_net": self._c_fn.state_dict(),
            "q_optimizer": self._q_optimizer.state_dict(),
            "critic_optimizer": self._c_optimizer.state_dict(),
            "policy_optimizer": self._p_optimizer.state_dict(),
            "train_step": self._global_step,
            "episode_num": self.episode_num ###fix??
      }
      for q_fn, q_fn_target in self._q_fns:
          checkpoint["q_net"]        = q_fn.state_dict()
          checkpoint["q_net_target"] = q_fn_target.state_dict()
      if self._train_alpha:
          checkpoint["alpha"] = self._alpha_var
          checkpoint["alpha_optimizer"] = self._a_optimizer.state_dict()
      if self._train_alpha_entropy:
          checkpoint["alpha_entropy"] = self._alpha_entropy_var
          checkpoint["alpha_entropy_optimizer"] = self._ae_optimizer.state_dict()   
      torch.save(checkpoint, self.checkpoint_path)
    
  def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        for q_fn, q_fn_target in self._q_fns:
            q_fn.load_state_dict(checkpoint["q_net"])
            q_fn_target.load_state_dict(checkpoint["q_net_target"])
        self._p_fn.load_state_dict(checkpoint["policy_net"])
        self._c_fn.load_state_dict(checkpoint["critic_net"])
        self._p_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self._q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self._c_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._global_step = checkpoint["train_step"]
        self.episode_num = checkpoint["episode_num"]###???fix
        if self._train_alpha:
            self._alpha_var = checkpoint["alpha"]
            self._a_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if self._train_alpha_entropy:
            self._alpha_entropy_var=checkpoint["alpha_entropy"]
            self._ae_optimizer.load_state_dict(checkpoint["alpha_entropy_optimizer"])
        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self._global_step) + " time step")

PolicyConfig = collections.namedtuple(
    'PolicyConfig', 'ptype, ckpt, wrapper, model_params')


PTYPES = [
    'randinit',
    'load',
]

WRAPPER_TYPES = [
    'none',
    'gaussian',
]

# params: (wrapper_type, *wrapper_params)
# wrapper_type: none, eps, gaussian, gaussianeps


def wrap_policy(a_net, wrapper):
  """Wraps actor network with desired randomization."""
  if wrapper[0] == 'none':
    policy = RandomSoftPolicy(a_net)
  elif wrapper[0] == 'gaussian':
    policy = GaussianRandomSoftPolicy(
        a_net, std=wrapper[1])
  return policy


def load_policy(policy_cfg, action_spec):
  """Loads policy based on config."""
  if policy_cfg.ptype not in PTYPES:
    raise ValueError('Unknown policy type %s.' % policy_cfg.ptype)
  if policy_cfg.ptype in ['randinit', 'load']:
    a_net = ActorNetwork(
        action_spec,
        fc_layer_params=policy_cfg.model_params)
    if policy_cfg.ptype == 'load':
      logging.info('Loading policy from %s...', policy_cfg.ckpt)
      a_net = torch.load(policy_cfg.ckpt)
    policy = wrap_policy(a_net, policy_cfg.wrapper)
  return policy


def parse_policy_cfg(policy_cfg):
  return PolicyConfig(*policy_cfg)

def maybe_makedirs(log_dir):
  import os.path
  if not os.path.exists(log_dir):
     os.mkdir(log_dir)
#####################################
#from train_eval_utils
from typing import Callable
Transition = collections.namedtuple(
    'Transition', 's1, s2, a1, a2, discount, reward')

def eval_policy_episodes(env, policy, n_episodes):
  """Evaluates policy performance."""
  results = []
  for _ in range(n_episodes):
    time_step = env.reset()
    total_rewards = 0.0
    while not time_step.is_last().numpy()[0]:
      action = policy(time_step.observation)[0]
      time_step = env.step(action)
      total_rewards += time_step.reward
    results.append(total_rewards)
  results = torch.tensor(results)
  return torch.mean(results).to(dtype=torch.float32), torch.std(results).to(dtype=torch.float32)


def eval_policies(env, policies, n_episodes):
  results_episode_return = []
  infos = collections.OrderedDict()
  for name, policy in policies.items():
    mean, _ = eval_policy_episodes(env, policy, n_episodes)
    results_episode_return.append(mean)
    infos[name] = collections.OrderedDict()
    infos[name]['episode_mean'] = mean
  results = results_episode_return
  return results, infos

def map_structure(func: Callable, structure):
    
    if not callable(func):
        raise TypeError("func must be callable, got: %s" % func)

    if isinstance(structure, list):
        return [map_structure(func, item) for item in structure]

    if isinstance(structure, dict):
        return {key: map_structure(func, structure[key]) for key in structure}

    return func(structure)


###############

MUJOCO_ENVS = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "InvertedPendulum-v2",
    "InvertedDoublePendulum-v2",
    "Reacher-v2",
    "Swimmer-v2",
    "Walker2d-v2"
]



def get_transition(time_step, next_time_step, action, next_action):
  return Transition(
      s1=time_step.observation,
      s2=next_time_step.observation,
      a1=action,
      a2=next_action,
      reward=next_time_step.reward,
      discount=next_time_step.discount)
  
class DataCollector(object):
  """Class for collecting sequence of environment experience."""

  def __init__(self, env, policy, data):
    self._env = env
    self._policy = policy
    self._data = data
    self._saved_action = None

  def collect_transition(self):
    """Collect single transition from environment."""
    time_step = self._env.current_time_step()
    if self._saved_action is None:
      self._saved_action = self._policy(time_step.observation)[0]
    action = self._saved_action
    next_time_step = self._env.step(action)
    next_action = self._policy(next_time_step.observation)[0]
    self._saved_action = next_action
    if not time_step.is_last()[0].numpy():
      transition = get_transition(time_step, next_time_step,
                                  action, next_action)
      self._data.add_transitions(transition)
      return 1
    else:
      return 0
  
#######################
def gather(params, indices, axis = None):
    if axis is None:
        axis = 0
    if axis < 0:
        axis = len(params.shape) + axis
    if axis == 0:
        return params[indices]
    elif axis == 1:
        return params[:, indices]
    elif axis == 2:
        return params[:, :, indices]
    elif axis == 3:
        return params[:,:,:, indices]

def scatter_update(tensor, indices, updates):
    tensor = torch.tensor(tensor)
    indices = torch.tensor(indices, dtype=torch.long)
    updates = torch.tensor(updates)
    tensor[indices] = updates
    return tensor
  
class DatasetView(object):
  """Interface for reading from dataset."""

  def __init__(self, dataset, indices):
    self._dataset = dataset
    self._indices = indices

  def get_batch(self, indices):
    real_indices = self._indices[indices]
    return self._dataset.get_batch(real_indices)

  @property
  def size(self):
    return self._indices.shape[0]

def save_copy(data, ckpt_name):
  """Creates a copy of the current data and save as a checkpoint."""
  new_data = Dataset(
      observation_spec=data.config['observation_spec'],
      action_spec=data.config['action_spec'],
      size=data.size,
      circular=False)
  full_batch = data.get_batch(np.arange(data.size))
  new_data.add_transitions(full_batch)
  torch.save(new_data, ckpt_name+".pt")
  
class Dataset(nn.Module):
  """Tensorflow module of dataset of transitions."""

  def __init__(
      self,
      observation_spec,
      action_spec,
      size,
      circular=True,
      ):
    super(Dataset, self).__init__()
    self._size = size
    self._circular = circular
    obs_shape = list(observation_spec.shape)
    obs_type = observation_spec.dtype
    action_shape = list(action_spec.shape)
    action_type = action_spec.dtype
    self._s1 = self._zeros([size] + obs_shape, obs_type)
    self._s2 = self._zeros([size] + obs_shape, obs_type)
    self._a1 = self._zeros([size] + action_shape, action_type)
    self._a2 = self._zeros([size] + action_shape, action_type)
    self._discount = self._zeros([size], torch.float32)
    self._reward = self._zeros([size], torch.float32)
    self._data = Transition(
        s1=self._s1, s2=self._s2, a1=self._a1, a2=self._a2,
        discount=self._discount, reward=self._reward)
    self._current_size = torch.autograd.Variable(torch.tensor(0))
    self._current_idx = torch.autograd.Variable(torch.tensor(0))
    self._capacity = torch.autograd.Variable(torch.tensor(self._size))
    self._config = collections.OrderedDict(
        observation_spec=observation_spec,
        action_spec=action_spec,
        size=size,
        circular=circular)

  @property
  def config(self):
    return self._config

  def create_view(self, indices):
    return DatasetView(self, indices)

  def get_batch(self, indices):
    indices = torch.LongTensor(indices)
    def get_batch_(data_):
      return gather(data_, indices)
    transition_batch = map_structure(get_batch_, self._data)
    return transition_batch

  @property
  def data(self):
    return self._data

  @property
  def capacity(self):
    return self._size

  @property
  def size(self):
    return self._current_size.numpy()

  def _zeros(self, shape, dtype):
    """Create a variable initialized with zeros."""
    return torch.autograd.Variable(torch.zeros(shape, dtype))

  def add_transitions(self, transitions):
    assert isinstance(transitions, Transition)
    batch_size = transitions.s1.shape[0]
    effective_batch_size = torch.minimum(
        batch_size, self._size - self._current_idx)
    indices = self._current_idx + torch.arange(effective_batch_size)
    for key in transitions._asdict().keys():
      data = getattr(self._data, key)
      batch = getattr(transitions, key)
      scatter_update(data, indices, batch[:effective_batch_size])
    # Update size and index.
    if torch.less(self._current_size, self._size):
      self._current_size+=effective_batch_size
    self._current_idx+=effective_batch_size
    if self._circular:
      if torch.greater_equal(self._current_idx, self._size):
        self._current_idx=0
#########################
#utils.py
def shuffle_indices_with_steps(n, steps=1, rand=None):
  """Randomly shuffling indices while keeping segments."""
  if steps == 0:
    return np.arange(n)
  if rand is None:
    rand = np.random
  n_segments = int(n // steps)
  n_effective = n_segments * steps
  batch_indices = rand.permutation(n_segments)
  batches = np.arange(n_effective).reshape([n_segments, steps])
  shuffled_batches = batches[batch_indices]
  shuffled_indices = np.arange(n)
  shuffled_indices[:n_effective] = shuffled_batches.reshape([-1])
  return shuffled_indices


#########################
#collect_data.py
#app.flags.DEFINE_string('f', '', 'kernel')
#FLAGS =app.flags.FLAGS

# def del_all_flags(FLAGS):
flags_dict = flags.FLAGS._flags()
keys_list = [keys for keys in flags_dict]
for keys in keys_list:
  flags.FLAGS.__delattr__(keys)
#FLAGS = flags.FLAGS
#app.flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_string('root_offlinerl_dir', '/tmp/offlinerl/data','Root directory for saving data.')
flags.DEFINE_string('sub_offlinerl_dir', '0', 'sub directory for saving data.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'env name.')
flags.DEFINE_string('data_name', 'random', 'data name.')
flags.DEFINE_string('env_loader', 'mujoco', 'env loader, suite/gym.')
flags.DEFINE_string('config_dir',
                    'behavior_regularized_offline_rl.brac.configs',
                    'config file dir.')
flags.DEFINE_string('config_file', 'dcfg_pure', 'config file name.')
flags.DEFINE_string('policy_root_dir', None,
                    'Directory in which to find the behavior policy.')
flags.DEFINE_integer('n_samples', int(1e6), 'number of transitions to collect.')
flags.DEFINE_integer('n_eval_episodes', 20,
                     'number episodes to eval each policy.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_string('hiddenflags', '0', 'Hidden flags parameter.') 

FLAGS = flags.FLAGS

def get_sample_counts(n, distr):
  """Provides size of each sub-dataset based on desired distribution."""
  distr = torch.tensor(distr)
  distr = distr / torch.sum(distr)
  counts = []
  remainder = n
  for i in range(distr.shape[0] - 1):
    count = int(n * distr[i])
    remainder -= count
    counts.append(count)
  counts.append(remainder)
  return counts


def collect_n_transitions(tf_env, policy, data, n, log_freq=10000):
  """Adds desired number of transitions to dataset."""
  collector = DataCollector(tf_env, policy, data)
  time_st = time.time()
  timed_at_step = 0
  steps_collected = 0
  while steps_collected < n:
    count = collector.collect_transition()
    steps_collected += count
    if (steps_collected % log_freq == 0
        or steps_collected == n) and count > 0:
      steps_per_sec = ((steps_collected - timed_at_step)
                       / (time.time() - time_st))
      timed_at_step = steps_collected
      time_st = time.time()
      logging.info('(%d/%d) steps collected at %.4g steps/s.', steps_collected,
                   n, steps_per_sec)


def collect_data(
    log_dir,
    data_config,
    n_samples=int(1e6),
    env_name='HalfCheetah-v2',
    log_freq=int(1e4),
    n_eval_episodes=20,
    ):
  """Creates dataset of transitions based on desired config."""
  seed=0
  torch.manual_seed(seed)
  np.random.seed(seed)
  dm_env = gym.spec(env_name).make()
  env = alf_gym_wrapper.AlfGymWrapper(dm_env)
  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  

  # Initialize dataset.
  sample_sizes = list([cfg[-1] for cfg in data_config])
  sample_sizes = get_sample_counts(n_samples, sample_sizes)
  data = Dataset(
        observation_spec,
        action_spec,
        n_samples,
        circular=False)
  
  # Collect data for each policy in data_config.
  time_st = time.time()
  test_results = collections.OrderedDict()
  for (policy_name, policy_cfg, _), n_transitions in zip(
      data_config, sample_sizes):
    policy_cfg = parse_policy_cfg(policy_cfg)
    policy = load_policy(policy_cfg, action_spec)
    logging.info('Testing policy %s...', policy_name)
    eval_mean, eval_std = eval_policy_episodes(
        env, policy, n_eval_episodes)
    test_results[policy_name] = [eval_mean, eval_std]
    logging.info('Return mean %.4g, std %.4g.', eval_mean, eval_std)
    logging.info('Collecting data from policy %s...', policy_name)
    collect_n_transitions(env, policy, data, n_transitions, log_freq)

  # Save final dataset.
  data_ckpt_name = os.path.join(log_dir, 'data')
  torch.save(data,data_ckpt_name)
  time_cost = time.time() - time_st
  logging.info('Finished: %d transitions collected, '
               'saved at %s, '
               'time cost %.4gs.', n_samples, data_ckpt_name, time_cost)


def main(argv):
  del argv
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  sub_dir = FLAGS.sub_offlinerl_dir
  log_dir = os.path.join(
      FLAGS.root_offlinerl_dir,
      FLAGS.env_name,
      FLAGS.data_name,
      sub_dir,
      )
  maybe_makedirs(log_dir)
  config_module = importlib.import_module(
      '{}.{}'.format(FLAGS.config_dir, FLAGS.config_file))
  collect_data(
      log_dir=log_dir,
      data_config=config_module.get_data_config(FLAGS.env_name,
                                                FLAGS.policy_root_dir),
      n_samples=FLAGS.n_samples,
      env_name=FLAGS.env_name,
      n_eval_episodes=FLAGS.n_eval_episodes)

def test_collect_data():
    flags.FLAGS.sub_offlinerl_dir = '0'
    flags.FLAGS.env_name = 'HalfCheetah-v2'
    flags.FLAGS.data_name = 'example'
    flags.FLAGS.config_file = 'dcfg_example'
    data_dir = 'testdata'
    flags.FLAGS.test_srcdir= '0'
    flags.FLAGS.policy_root_dir = os.path.join(flags.FLAGS.test_srcdir,
                                               data_dir)
    flags.FLAGS.n_samples = 10000  # Short collection.
    flags.FLAGS.n_eval_episodes = 1
    main(None)

########################
def get_datetime():
  now = datetime.datetime.now().isoformat()
  now = re.sub(r'\D', '', now)[:-6]
  return now

@gin.configurable
def train_eval_offline(
    # Basic args.
    log_dir,
    data_file,
    agent_module,
    env_name='HalfCheetah-v2',
    n_train=int(1e6),
    shuffle_steps=0,
    seed=0,
    use_seed_for_data=False,
    # Train and eval args.
    total_train_steps=int(1e6),
    summary_freq=100,
    print_freq=1000,
    save_freq=int(2e4),
    eval_freq=5000,
    n_eval_episodes=20,
    # Agent args.
    model_params=(((200, 200),), 2),
    optimizers=(('adam', 0.001),),
    batch_size=256,
    weight_decays=(0.0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99,
    ):
  """Training a policy with a fixed dataset."""
  # Create tf_env to get specs.
  dm_env = gym.spec(env_name).make()
  env = alf_gym_wrapper.AlfGymWrapper(dm_env)
  
  observation_spec = env.observation_spec()
  action_spec = env.action_spec()

  # Prepare data.
  logging.info('Loading data from %s ...', data_file)
  data  = torch.load(data_file)
  data_size = data.size()      
  full_data = Dataset(observation_spec, action_spec, data_size)
  # Split data.
  n_train = min(n_train, full_data.size)
  logging.info('n_train %s.', n_train)
  if use_seed_for_data:
    rand = np.random.RandomState(seed)
  else:
    rand = np.random.RandomState(0)
  shuffled_indices = shuffle_indices_with_steps(
      n=full_data.size, steps=shuffle_steps, rand=rand)
  train_indices = shuffled_indices[:n_train]
  train_data = full_data.create_view(train_indices)

  # Create agent.
  agent_flags = Flags(
      observation_spec=observation_spec,
      action_spec=action_spec,
      model_params=model_params,
      optimizers=optimizers,
      batch_size=batch_size,
      weight_decays=weight_decays,
      update_freq=update_freq,
      update_rate=update_rate,
      discount=discount,
      train_data=train_data)
  agent_args = agent_module.Config(agent_flags).agent_args
  agent = agent_module.Agent(**vars(agent_args)) #ATTENTION: Debugg ====> should it be D2EAgent here??
  agent_ckpt_name = os.path.join(log_dir, 'agent')

  # Restore agent from checkpoint if there exists one.
  if os.path.exists('{}.index'.format(agent_ckpt_name)):
    logging.info('Checkpoint found at %s.', agent_ckpt_name)
    torch.load(agent, agent_ckpt_name)

  # Train agent.
  train_summary_dir = os.path.join(log_dir, 'train')
  eval_summary_dir = os.path.join(log_dir, 'eval')
  train_summary_writer = SummaryWriter(
      logdir=train_summary_dir)
  eval_summary_writers = collections.OrderedDict()
  for policy_key in agent.test_policies.keys():
    eval_summary_writer = SummaryWriter(
        logdir=os.path.join(eval_summary_dir, policy_key))
    eval_summary_writers[policy_key] = eval_summary_writer
  eval_results = []

  time_st_total = time.time()
  time_st = time.time()
  step = agent.global_step
  timed_at_step = step
  while step < total_train_steps:
    agent.train_step()
    step = agent.global_step
    if step % summary_freq == 0 or step == total_train_steps:
      agent.write_train_summary(train_summary_writer)
    if step % print_freq == 0 or step == total_train_steps:
      agent.print_train_info()
    if step % eval_freq == 0 or step == total_train_steps:
      time_ed = time.time()
      time_cost = time_ed - time_st
      logging.info(
          'Training at %.4g steps/s.', (step - timed_at_step) / time_cost)
      eval_result, eval_infos = eval_policies(
          env, agent.test_policies, n_eval_episodes)
      eval_results.append([step] + eval_result)
      logging.info('Testing at step %d:', step)
      for policy_key, policy_info in eval_infos.items():
        logging.info(utils.get_summary_str(
            step=None, info=policy_info, prefix=policy_key+': '))
        utils.write_summary(eval_summary_writers[policy_key], step, policy_info)
      time_st = time.time()
      timed_at_step = step
    if step % save_freq == 0:
      torch.save(agent, agent_ckpt_name)
      logging.info('Agent saved at %s.', agent_ckpt_name)

  torch.save(agent, agent_ckpt_name)
  time_cost = time.time() - time_st_total
  logging.info('Training finished, time cost %.4gs.', time_cost)
  return torch.tensor(eval_results)

##############################
###train_offline.py
flags.DEFINE_string('data_root_dir',
                    os.path.join(os.getenv('HOME', '/'),
                                 'tmp/offlinerl/data'),
                    'Root directory for data.')
flags.DEFINE_string('data_sub_dir', '0', '')
flags.DEFINE_string('data_name', 'eps1', 'data name.')
flags.DEFINE_string('data_file_name', 'data', 'data checkpoint file name.')

# Flags for offline training.
flags.DEFINE_string('root_dir',
                    os.path.join(os.getenv('HOME', '/'), 'tmp/offlinerl/learn'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('sub_dir', '0', '')
flags.DEFINE_string('agent_name', 'D2E', 'agent name.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'env name.')
flags.DEFINE_integer('seed', 0, 'random seed, mainly for training samples.')
flags.DEFINE_integer('total_train_steps', int(5e5), '')
flags.DEFINE_integer('n_eval_episodes', 20, '')
flags.DEFINE_integer('n_train', int(1e6), '')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS
AGENT_MODULES_DICT = {
    'D2E': D2EAgent, ###Debug ===> is it correct to set it here??? 
}

def main(_):
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  # Setup data file path.
  data_dir = os.path.join(
      FLAGS.data_root_dir,
      FLAGS.env_name,
      FLAGS.data_name,
      FLAGS.data_sub_dir,
      )
  data_file = os.path.join(
      data_dir, FLAGS.data_file_name)

  # Setup log dir.
  if FLAGS.sub_dir == 'auto':
    sub_dir = get_datetime()
  else:
    sub_dir = FLAGS.sub_dir
  log_dir = os.path.join(
      FLAGS.root_dir,
      FLAGS.env_name,
      FLAGS.data_name,
      'n'+str(FLAGS.n_train),
      FLAGS.agent_name,
      sub_dir,
      str(FLAGS.seed),
      )
  maybe_makedirs(log_dir)
  train_eval_offline(
      log_dir=log_dir,
      data_file=data_file,
      agent_module=AGENT_MODULES_DICT[FLAGS.agent_name],
      env_name=FLAGS.env_name,
      n_train=FLAGS.n_train,
      total_train_steps=FLAGS.total_train_steps,
      n_eval_episodes=FLAGS.n_eval_episodes,
      )


class TrainOfflineTest(TestCase):

  def test_train_offline(self):
    data_dir = 'testdata/data'
    flags.FLAGS.data_root_dir = os.path.join(flags.FLAGS.test_srcdir, data_dir)
    flags.FLAGS.sub_dir = '0'
    flags.FLAGS.env_name = 'HalfCheetah-v2'
    flags.FLAGS.data_name = 'example'
    flags.FLAGS.agent_name = 'D2E'
    flags.FLAGS.gin_bindings = [
        'train_eval_offline.model_params=((200, 200),)',
        'train_eval_offline.optimizers=(("adam", 5e-4),)']
    flags.FLAGS.n_train = 100
    flags.FLAGS.n_eval_episodes = 1
    flags.FLAGS.total_train_steps = 100  # Short training.

    main(None)  # Just test that it runs.

#more hyperparameter run_dual.sh


  