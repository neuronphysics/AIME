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
from torch.testing._internal.common_utils import TestCase
import collections
import numpy as np
import os
import gin
import gym
# import mujoco_py
from absl import app
from absl import flags
from absl import logging
import argparse
import math
import tensor_specs

import time
import alf_gym_wrapper
import importlib  

LOG_STD_MIN = -5
LOG_STD_MAX = 0

def weights_init(modules, type='xavier'):
    "Based on shorturl.at/jmqV3"
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Sequential):
        for k, v in m._modules.items():
            if isinstance(v, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(v.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(v.weight)
                else:
                    n = v.kernel_size[0] * v.kernel_size[1] * v.out_channels
                    v.weight.data.normal_(0, np.sqrt(2. / n))

                if v.bias is not None:
                    v.bias.data.zero_()
            elif isinstance(v, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(v.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(v.weight)
                else:
                    n = v.kernel_size[0] * v.kernel_size[1] * v.out_channels
                    v.weight.data.normal_(0, np.sqrt(2. / n))

                if v.bias is not None:
                    v.bias.data.zero_()
            elif isinstance(v, nn.BatchNorm2d):
                v.weight.data.fill_(1.0)
                v.bias.data.zero_()
            elif isinstance(v, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(v.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(v.weight)
                else:
                    v.weight.data.fill_(1.0)

                if v.bias is not None:
                    v.bias.data.zero_()

def get_spec_means_mags(spec):
  print(f'spec in get_spec_means: {spec}')
  means = (spec.high + spec.low) / 2.0
  mags = (spec.high - spec.low) / 2.0
  means = Variable(torch.from_numpy(means), requires_grad=False)
  mags  = Variable(torch.from_numpy(mags), requires_grad=False)
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
        print(f'inputs shape in split.forward is: {inputs.shape}')
        output = self._module(inputs) # .T
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
    self._latent_spec = latent_spec
    self._layers = nn.ModuleList()
    for hidden_size in fc_layer_params:
        if len(self._layers)==0:
           self._layers.append(nn.Linear(self._latent_spec.size, hidden_size))
        else:
           self._layers.append(nn.Linear(hidden_size, hidden_size))
        self._layers.append(nn.ReLU())
    output_layer = nn.Linear(hidden_size,
        self._action_spec.shape[0] * 2
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

    # print(self._layers)
    for layer in self._layers:
      weights_init(layer)
    # print(f'This is an {error}')

  @property
  def action_spec(self):
      return self._action_spec

  def _get_outputs(self, state):
      h = state
      for l in self._layers[:-1]:
          h = l(h)
      self._mean_logvar_layers = Split(
         self._layers[-1],
         n_parts=2,
      )
      print(f'latent_spec size is: {self._latent_spec.size}')
      print(f'h size is: {h.size()}')
      mean, log_std = self._mean_logvar_layers(h)
      print(f'mean is : {mean.shape}')
      utils.check_for_nans_and_nones(mean)

      print(f'log_std is: {log_std.shape}')
      utils.check_for_nans_and_nones(log_std)

      print(f'action_mags shape is: {self._action_mags.shape}')
      print(f'action_means shape is: {self._action_means.shape}')
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
                                   tT.AffineTransform(loc=self._action_means, scale=self._action_mags, event_dim=mean.shape[-1]), 
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
    print(f'state shape in actor __call__: {state.shape}')
    print(f'state type is: {type(state)}')
    # utils.check_for_nans_and_nones(state)

    a_dist, a_tanh_mode = self._get_outputs(state)
    print(f' a_dist : {a_dist}')
    print(f' a_tanh_mode shape: {a_tanh_mode.shape}')
    utils.check_for_nans_and_nones(a_tanh_mode)

    a_sample = a_dist.sample()
    print(f' a_sample shape: {a_sample.shape}')
    utils.check_for_nans_and_nones(a_sample)

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
              self._layers.append(nn.Linear(latent_spec.size+action_spec.shape[0], hidden_size))
          else:
              self._layers.append(nn.Linear(hidden_size, hidden_size))
          self._layers.append(nn.ReLU())
      output_layer = nn.Linear(hidden_size,1)
      self._layers.append(output_layer)
  
    def forward(self, state, action):
        print(state.shape) # 256 x 3
        print(action.shape) # 256 x 1
        embedding = torch.cat((state, action), dim=-1)
        hidden = embedding
        for l in self._layers:
            hidden = l(hidden)
        return hidden.squeeze()
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
  def q_net_factory(latent_spec, action_spec):
    return CriticNetwork(
        latent_spec,
        action_spec,
        fc_layer_params=model_params[0])
  def p_net_factory(latent_spec, action_spec):
    return ActorNetwork(
        latent_spec,
        action_spec,
        fc_layer_params=model_params[1])
  def c_net_factory(latent_spec, action_spec):
    return CriticNetwork(
        latent_spec,
        action_spec,
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

class AgentModule:
  """Pytorch module for BRAC dual agent."""
  def __init__(
      self,
      modules=None,
      latent_spec=None,
      action_spec=None
      ):
    print(f'latent spec in AgentModule: {latent_spec}')
    self._modules = modules # This is a Flags object
    self._build_modules(latent_spec, action_spec)


  def _build_modules(self, latent_spec, action_spec):
    self._q_nets = []
    n_q_fns = self._modules.n_q_fns
    # print(type(n_q_fns))
    for _ in range(n_q_fns):
      self._q_nets.append(
          [self._modules.q_net_factory(latent_spec, action_spec),  # Learned Q-value.
           self._modules.q_net_factory(latent_spec, action_spec),]  # Target Q-value.
          )
    self._p_net = self._modules.p_net_factory(latent_spec, action_spec)
    self._c_net = self._modules.c_net_factory(latent_spec, action_spec)
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
         print(f'q_net._modules.values(): {q_net._modules.values()}')
         for net in list(q_net._modules.values()):
          for layer in list(net):
            # print(layer)
            # print(type(layer))
            if isinstance(layer, nn.Linear):
              q_weights += layer.weight.data
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
            # print(v.is_leaf) # It is indeed a leaf here
            if v.requires_grad:
               vars_.append(v)
    return tuple(vars_)

  @property
  def q_target_variables(self):
    vars_ = []
    for _, q_net in self._q_nets:
        vars = q_net.parameters()
        for v in vars:
            if v.requires_grad:
               vars_.append(v)
    return tuple(vars_)

  @property
  def p_net(self):
    return self._p_net

  def p_fn(self, s):
    return self._p_net(s)

  @property
  def p_weights(self):
    pw = []
    print(list(self.p_net.modules()))
    for layer in list(self.p_net.modules()):
          # print(layer)
          # print(type(layer))
          if isinstance(layer, nn.Linear):
            pw += layer.weight.data
    return pw

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
    cw = []
    print(list(self.c_net.modules()))
    for layer in list(self.c_net.modules()):
          # print(layer)
          # print(type(layer))
          if isinstance(layer, nn.Linear):
            cw += layer.weight.data
    return cw

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
    directory = os.getcwd()
    checkpoint_dir=directory+"/run"
    self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    self._build_agent()

  def _build_agent(self):
    """Builds agent components."""
    self._build_fns()
    self._build_optimizers()
    self._global_step = torch.tensor(0.0, requires_grad=False) # Why requires_grad=True? 
    #   Making it false because "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation." when incrementing it
    self._train_info = collections.OrderedDict()
    self._checkpointer = self._build_checkpointer()
    self._test_policies = collections.OrderedDict()
    self._build_test_policies()
    self._online_policy = self._build_online_policy()
    train_batch = self._get_train_batch()
    self._init_vars(train_batch)

  # This never actually runs. Self._build_fns in _duild_agent calls child class's _build_fns
  def _build_fns(self, latent_spec, action_spec):
    self._agent_module = AgentModule(modules = self._modules, latent_spec=self._latent_spec, action_spec=self._action_spec)

  def _get_vars(self):
    return []

  def _build_optimizers(self):
     opt = self._optimizers[0]
     self._optimizer = torch.optim.Adam(
            self._agent_module.parameters(),
            lr= opt[0], # actually opt[0] might be the name of the optimizer
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
    # batch_indices = np.random.choice(len(self._train_data[-1]), self._batch_size)
    batch_indices = np.random.choice(len(self._train_data._indices), self._batch_size)
    print(f'{len(self._train_data._indices)}')
    # print(f' train_data size: {self._train_data.size}')
    batch_ = self._train_data.get_batch(batch_indices)
  
    transition_batch = batch_
    # transition_batch = []

    batch = dict(
        s1=transition_batch.s1,
        s2=transition_batch.s2,
        r=transition_batch.reward,
        dsc=transition_batch.discount,
        a1=transition_batch.a1,
        a2=transition_batch.a2,
    )

    # train data contains 6 lists, one for each s1, s2, r, gamma, a1, a2
    # for element in self._train_data:
    #   transition_batch.append(element[batch_indices])

    # batch = dict(
    #     s1=transition_batch[3], # state
    #     s2=transition_batch[4], # next state
    #     r=transition_batch[0], # reward
    #     dsc=transition_batch[5], # discount
    #     a1=transition_batch[1], # action1
    #     a2=transition_batch[2], # action2
    #     )

    return batch
  
  def _optimize_step(self, batch):
      pass
            
  def train_step(self):
      train_batch = self._get_train_batch()
      loss = self._optimize_step(train_batch)
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
    return self._global_step.detach().numpy()
  
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

class ContinuousRandomPolicy(nn.Module):
  """Samples actions uniformly at random."""

  def __init__(self, action_spec):
    super(ContinuousRandomPolicy, self).__init__()
    self._action_spec = action_spec

  def __call__(self, observation, state=()):
    action = tensor_specs.sample_bounded_spec(
        self._action_spec, outer_dims=[observation.shape[0]])
    action = self._action_spec.sample()
    return action, state
    return ["random"]


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
    super(D2EAgent, self).__init__(**kwargs)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules, latent_spec=self._latent_spec, action_spec = self._action_spec)
    self._q_fns = self._agent_module.q_nets
    self._p_fn = self._agent_module.p_net
    self._c_fn = self._agent_module.c_net
    self._divergence = utils.get_divergence(
        name=self._divergence_name,
        c = self._c_fn,
        device = self.device)
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
    print(self._agent_module.p_net.modules())
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
    # lambda_ = int(lambda_)
    # print(f'qs shape is :{qs.shape}')
    # print(qs.dtype)
    # print(lambda_)
    # print(f' torch.min(qs, dim=-1): {torch.min(qs, dim=-1)}')
    # print(f'lambda_ shape is :{lambda_.shape}')
    # print(lambda_.dtype) # lambda_ is a float
    to_ret = lambda_ * torch.min(qs, dim=-1).values + (1. - lambda_) * torch.max(qs, dim=-1).values
    print(f'to_ret is length of: {len(to_ret)}')
    return to_ret

  def _ensemble_q2_target(self, q2_targets):
    return self.ensemble_q(q2_targets)

  def _ensemble_q1(self, q1s):
    return self.ensemble_q(q1s)

  def _build_q_loss(self, batch):
    print(batch['a2'].shape)
    s1 = batch['s1']
    s2 = batch['s2'].type(torch.FloatTensor)
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
      print(f'q2_target shape: {q2_target_.shape}')
    q2_targets = torch.stack(q2_targets, dim=-1)
    q2_target = self._ensemble_q2_target(q2_targets)
    div_estimate = self._divergence.dual_estimate(
        s2, a2_p, a2_b, self._c_fn)
    print(f'q2_target length {len(q2_target)}') # tuple
    alpha_entropy_val, alpha_entropy_grad = self._get_alpha_entropy()
    print(f'alpha_entropy_val: {alpha_entropy_val} ')
    print(f'alpha_entropy_grad: {alpha_entropy_grad}')
    print(f'log_pi_a2_p shape {log_pi_a2_p.shape}')
    v2_target = (q2_target - alpha_entropy_val) * log_pi_a2_p

    alpha_val, alpha_grad = self._get_alpha()
    if self._value_penalty: #Equation (5)
       v2_target = v2_target - alpha_val * div_estimate
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
    alpha_val, alpha_grad = self._get_alpha()
    alpha_entropy_val, alpha_entropy_grad = self._get_alpha_entropy()
    p_loss = torch.mean(
        alpha_entropy_val * log_pi_a_p
        + alpha_val * div_estimate
        - q1 * q_start)
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays[1] * p_w_norm
    loss = p_loss + norm_loss

    info = collections.OrderedDict()
    info['p_loss'] = p_loss
    info['p_norm'] = p_w_norm

    return loss, info

  def _build_c_loss(self, batch):
    s = batch['s1'] # 256 x 3
    a_b = batch['a1']
    print(f's shape: {s.shape}')
    # print(f's in _build_c_loss is: {s}')
    print(f'a_b shape: {a_b.shape}')
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
    alpha, alpha_grad = self._get_alpha()
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
    s = batch['a1']
    _, _, log_pi_a = self._p_fn(s)
    alpha, alpha_grad = self._get_alpha_entropy()
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
    print(f'weight_decays are: {self._weight_decays}')
    print(f'opts are : {opts}')
    # print('Seeing the q_vars')
    self._q_optimizer = utils.OptMirrorAdam(self._get_q_vars(),lr=opts[0][1], betas=(opts[0][2],opts[0][3]), weight_decay=self._weight_decays[0])
    self._p_optimizer = utils.OptMirrorAdam(self._get_p_vars(),lr=opts[1][1], betas=(opts[1][2],opts[1][3]), weight_decay=self._weight_decays[1])
    self._c_optimizer = utils.OptMirrorAdam(self._get_c_vars(),lr=opts[2][1], betas=(opts[2][2],opts[2][3]), weight_decay=self._weight_decays[2])
    self._a_optimizer = torch.optim.Adam(self._agent_module.a_variables, lr=opts[3][1], betas=(opts[3][1],opts[3][3]))
    self._ae_optimizer = torch.optim.Adam(self._agent_module.ae_variables, lr=opts[3][1], betas=(opts[3][1],opts[3][3]))

  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    if self._global_step % self._update_freq == 0:
      source_vars, target_vars = self._get_source_target_vars()
      self._update_target_fns(source_vars, target_vars)
    # Update policy network parameter
    #https://bit.ly/3Bno0GC
    # policy network's update should be done before updating q network, or there will make some errors
    def closure_p():
        self._p_optimizer.zero_grad()
        policy_loss,_=self._build_p_loss(batch)
        policy_loss.backward()
        return policy_loss

    self._p_optimizer.zero_grad()
    policy_loss,_=self._build_p_loss(batch)
    policy_loss.backward(retain_graph=True)
    policy_loss = self._p_optimizer.step(closure_p)

    # Update q networks parameter
    def closure_q():
      self._q_optimizer.zero_grad()
      q_losses,q_info= self._build_q_loss(batch)
      q_losses.backward()
      return q_losses
    self._q_optimizer.zero_grad()
    q_losses,q_info= self._build_q_loss(batch)
    q_losses.backward()
    q_losses = self._q_optimizer.step(closure_q)

    #Update critic network parameter
    def closure_c():
      self._c_optimizer.zero_grad()
      critic_loss,_= self._build_c_loss( batch)
      critic_loss.backward()
      return critic_loss
    self._c_optimizer.zero_grad()
    critic_loss,_= self._build_c_loss( batch)
    critic_loss.backward()
    critic_loss = self._c_optimizer.step(closure_c)
    
    if self._train_alpha:
      def closure_alpha():
        self._a_optimizer.zero_grad()
        a_loss,a_info = self._build_a_loss( batch)
        a_loss.backward()
        return a_loss
      self._a_optimizer.zero_grad()
      a_loss,a_info = self._build_a_loss( batch)
      a_loss.backward()
      a_loss = self._a_optimizer.step(closure_alpha)

    if self._train_alpha_entropy:
      def closure_alpha_entropy():
       self._ae_optimizer.zero_grad()
       ae_loss,ae_info = self._build_ae_loss( batch)
       ae_loss.backward()
       return ae_loss
      self._ae_optimizer.zero_grad()
      ae_loss,ae_info = self._build_ae_loss( batch)
      ae_loss.backward()
      ae_loss = self._ae_optimizer.step(closure_alpha_entropy)

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
    def closure_c():
      self._c_optimizer.zero_grad()
      critic_loss,_ = self._build_c_loss( batch)
      critic_loss.backward()
    self._c_optimizer.zero_grad()
    critic_loss,_ = self._build_c_loss( batch)
    critic_loss.backward()
    critic_loss = self._c_optimizer.step(closure_c)
    return critic_loss

  def train_step(self):
    train_batch = self._get_train_batch()
    info = self._optimize_step(train_batch)
    for _ in range(self._c_iter - 1):
      train_batch = self._get_train_batch()
      self._extra_c_step(train_batch)
    for key, val in info.items():
      self._train_info[key] = val #.numpy()
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
            # "episode_num": self.episode_num ###fix??
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
        # self.episode_num = checkpoint["episode_num"]###???fix
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

#####################################
#from train_eval_utils
from typing import Callable
Transition = collections.namedtuple(
    'Transition', 's1, s2, a1, a2, discount, reward')

def eval_policy_episodes(env, policy, n_episodes):
  """Evaluates policy performance."""
  results = []
  for _ in range(n_episodes):
    state = env.reset()
    print(f' initial state shape is: {state.shape}')
    total_rewards = 0.0
    done = False
    while not done:
      print(f'state: {state}')
      action = policy(torch.FloatTensor(np.expand_dims(state, axis=0)))[0]
      next_state, reward, done, _ = env.step(action.detach())
      total_rewards += reward
      state = next_state

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


class AgentConfig(object):
  """Class for handling agent parameters."""

  def __init__(self, agent_flags):
    self._agent_flags = agent_flags
    self._agent_args = self._get_agent_args()

  def _get_agent_args(self):
    """Gets agent parameters associated with config."""
    agent_flags = self._agent_flags
    agent_args = Flags(
        latent_spec=agent_flags.latent_spec,
        action_spec=agent_flags.action_spec,
        optimizers=agent_flags.optimizers,
        batch_size=agent_flags.batch_size,
        weight_decays=agent_flags.weight_decays,
        update_rate=agent_flags.update_rate,
        update_freq=agent_flags.update_freq,
        discount=agent_flags.discount,
        train_data=agent_flags.train_data,
        )
    agent_args.modules = self._get_modules()
    return agent_args

  def _get_modules(self):
    raise NotImplementedError

  @property
  def agent_args(self):
    return self._agent_args



class Config(AgentConfig):

  def _get_modules(self):
    return get_modules(
        self._agent_flags.model_params,
        self._agent_flags.action_spec)
