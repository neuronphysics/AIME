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
from torch.testing._internal.common_utils import run_tests
from collections import Counter
import collections
import numpy as np
import os
import gin
import gym
import mujoco_py
from absl import app
from absl import flags
from absl import logging
import tensor_specs
import time
import alf_gym_wrapper
import importlib  
from torch.testing._internal.common_utils import TestCase
from torch.distributions import Distribution, Independent, MultivariateNormal
from collections import Counter
import torch.distributions as pyd
import math
from ExpectedInformationMaximization import ConditionalMixtureEIM, RecorderKeys, TrainIterationRecMod, ConfigInitialRecMod, DRERecMod, ComponentUpdateRecMod, WeightUpdateRecMod, Recorder, to_tensor
from MUJOCORecorder import MujocoData, MujocoModelRecMod 
import ExpectedInformationMaximization as EIM
from dm_control import suite
import dm_control
local_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LOG_STD_MIN = torch.tensor(-5, dtype=torch.float32, device=local_device,requires_grad=False)

LOG_STD_MAX = torch.tensor(2 , dtype=torch.float32, device=local_device,requires_grad=False)

def get_spec_means_mags(spec,device):
  means = (spec.maximum + spec.minimum) / 2.0
  mags = (spec.maximum - spec.minimum) / 2.0
  means = Variable(torch.tensor(means).type(torch.FloatTensor).to(device), requires_grad=False)
  mags  = Variable(torch.tensor(mags).type(torch.FloatTensor).to(device), requires_grad=False)
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)
        
    def forward(self, inputs):
        output = self._module(inputs)
        if output.ndim==1:
           result=torch.hsplit(output, self._n_parts )
        else:
           chunk_size = output.shape[self._dim] // self._n_parts
           result =torch.split(output, chunk_size, dim=self._dim)

        return result

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(-0.99, 0.99))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


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
           self._layers.append(nn.Linear(latent_spec.shape[0], hidden_size))
        else:
           self._layers.append(nn.Linear(hidden_size, hidden_size))
        self._layers.append(nn.ReLU())
    output_layer = nn.Linear(hidden_size,self._action_spec.shape[0] * 2)
    self._layers.append(output_layer)

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec, self.device)
    
    self.to(device=self.device)

  def to(self, *args, **kwargs):
      super().to(*args, **kwargs)

  @property
  def action_spec(self):
      return self._action_spec

  def _get_outputs(self, state):
      h = state
      
      for l in nn.Sequential(*(list(self._layers.children())[:-1])):
          h = l(h)

      self._mean_logvar_layers = Split(
         self._layers[-1],
         n_parts=2,
      )
      mean, log_std = self._mean_logvar_layers(h)
      
      a_tanh_mode = torch.tanh(mean) * self._action_mags + self._action_means
      log_std = torch.tanh(log_std).to(device=self.device)
      log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
      std = torch.exp(log_std)
      """
      a_distribution = TransformedDistribution(
                        base_distribution=Normal(loc=torch.full_like(mean, 0).to(device=self.device), 
                                                 scale=torch.full_like(mean, 1).to(device=self.device)), 
                        transforms=tT.ComposeTransform([
                                   tT.AffineTransform(loc=self._action_means.to(device=self.device), scale=self._action_mags.to(device=self.device), event_dim=mean.shape[-1]), 
                                   TanhTransform(),
                                   tT.AffineTransform(loc=mean, scale=std, event_dim=mean.shape[-1])]))
      """
      #print(f"size of mean with batch : {mean.shape}")
      if mean.ndim>1:
          mvn = MultivariateNormal(torch.full_like(mean, 0).to(device=self.device), torch.diag_embed(torch.full_like(mean, 1).to(device=self.device)))
          mvn._batch_shape=torch.Size([mean.shape[0]])
          #print(mvn.batch_shape, mvn.event_shape)
          a_distribution= TransformedDistribution(
                                          mvn,
                                          tT.ComposeTransform([
                                          tT.AffineTransform(loc=self._action_means.to(device=self.device), scale=self._action_mags.to(device=self.device),event_dim=2),
                                          TanhTransform(),
                                          tT.AffineTransform(loc=mean, scale=std, event_dim= 2 )]))
      else:
          mvn = MultivariateNormal(torch.full_like(mean, 0).to(device=self.device), torch.diag(torch.full_like(mean, 1).to(device=self.device)))
          mvn._batch_shape=torch.Size([1])
          a_distribution= TransformedDistribution(
                                          mvn,
                                          tT.ComposeTransform([
                                          tT.AffineTransform(loc=self._action_means.to(device=self.device), scale=self._action_mags.to(device=self.device),event_dim=1),
                                          TanhTransform(),
                                          tT.AffineTransform(loc=mean, scale=std, event_dim= 1 )]))
      

      #a_distribution.base_dist._batch_shape=torch.Size([1])
      #https://www.ccoderun.ca/programming/doxygen/pytorch/classtorch_1_1distributions_1_1transformed__distribution_1_1TransformedDistribution.html
      return a_distribution, a_tanh_mode

  def get_log_density(self, state, action):
    a_dist, _ = self._get_outputs(state.to(device=self.device))
    log_density = a_dist.log_prob(action.to(device=self.device))
    return log_density

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weight[0])
    return w_list

  def __call__(self, state):
    a_dist, a_tanh_mode = self._get_outputs(state.to(device=self.device))
    a_sample = a_dist.sample()
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample_n(self, state, n=1):
    a_dist, a_tanh_mode = self._get_outputs(state.to(device=self.device))
    a_sample = a_dist.sample([n])
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
      self._latent_spec = latent_spec
      self._layers = nn.ModuleList()
      for hidden_size in fc_layer_params:
          if len(self._layers)==0:
              
              self._layers.append(nn.Linear(latent_spec.shape[0]+action_spec.shape[0], hidden_size))
          else:
              self._layers.append(nn.Linear(hidden_size, hidden_size))
          self._layers.append(nn.ReLU())
      output_layer = nn.Linear(hidden_size, 1)
      self._layers.append(output_layer)
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.to(device=self.device)

    def forward(self, state,action):
        hidden = torch.cat([state.to(device=self.device), action.to(device=self.device)], dim=-1)
        for l in self._layers:
            hidden = l(hidden)
        return hidden

class ValueNetwork(nn.Module):
    """Value Network."""
    def __init__(
      self,
      latent_spec,
      fc_layer_params=(),
      ):
      super(ValueNetwork, self).__init__()
      self._latent_spec = latent_spec
      self._layers = nn.ModuleList()
      for hidden_size in fc_layer_params:
          if len(self._layers)==0:
              
              self._layers.append(nn.Linear(latent_spec.shape[0], hidden_size))
          else:
              self._layers.append(nn.Linear(hidden_size, hidden_size))
          self._layers.append(nn.ReLU())
      output_layer = nn.Linear(hidden_size,1)
      self._layers.append(output_layer)
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.to(device=self.device)

    def forward(self, state):
        hidden = state.to(device=self.device)
        for l in self._layers:
            hidden = l(hidden)
        return hidden


def get_modules(model_params, observation_spec, action_spec):
  """Gets pytorch modules for Q-function, policy, and discriminator."""
  model_params, n_q_fns = model_params
  
  if len(model_params) == 1:
    model_params = tuple([model_params[0]] * 4)
  elif len(model_params) < 4:
    raise ValueError('Bad model parameters %s.' % model_params)
  def q_net_factory():
    return CriticNetwork(
        observation_spec, 
        action_spec,
        fc_layer_params=model_params[0])
  def p_net_factory():
    return ActorNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=model_params[1])
  def c_net_factory():
    return CriticNetwork(
        observation_spec, 
        action_spec,
        fc_layer_params=model_params[2])
  def v_net_factory():
    return ValueNetwork(
        observation_spec, 
        fc_layer_params=model_params[3])
  modules_list = utils.Flags(
      q_net_factory=q_net_factory,
      p_net_factory=p_net_factory,
      c_net_factory=c_net_factory,
      v_net_factory=v_net_factory,
      n_q_fns=n_q_fns,
      )
  return modules_list
#######################################
################ AGENT ################
#######################################
ALPHA_MAX = 500.0
class GeneralAgent(nn.Module):
  """Tensorflow module for agent."""

  def __init__(
      self,
      modules_list=None,
      ):
    super(GeneralAgent, self).__init__()
    self._modules_list = modules_list
    self._build_modules()
    self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(device=self.device)
    
  def _build_modules(self):
    pass


class AgentModule(GeneralAgent):
  """Pytorch module for BRAC dual agent."""
  def __init__(self,modules_list):
    # invoking the __init__ of the parent class
    super().__init__(modules_list)
  def _build_modules(self):
    self._q_nets = list()  
    n_q_fns = self._modules_list.n_q_fns
    for _ in range(n_q_fns):
      self._q_nets.append(
          [self._modules_list.q_net_factory(),  # Learned Q-value.
           self._modules_list.q_net_factory(),]  # Target Q-value.
          )
    self._p_net = self._modules_list.p_net_factory()
    self._c_net = self._modules_list.c_net_factory()
    self._v_net = self._modules_list.v_net_factory() #value network
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
      for name, param in q_net._layers.named_parameters():
          if 'weight' in name:
             q_weights += list(param)
    return q_weights

  @property
  def q_target_weights(self):
    q_weights = []
    for _, q_net in self._q_nets:
      for name, param in q_net._layers.named_parameters():
          if 'weight' in name:
             q_weights += list(param)
    return q_weights

  @property
  def q_source_variables(self):
    vars_=[]
    for q_net, _ in self._q_nets:
        for param in q_net._layers:
            if not isinstance(param, nn.ReLU):
               vars_+=list(param.parameters())
    return vars_
 
  @property
  def q_target_variables(self):
    vars_ = []
    for _, q_net in self._q_nets:
        for param in q_net._layers:
            if not isinstance(param, nn.ReLU):
               vars_+=list(param.parameters())
    return vars_
  
  @property
  def v_net(self):
    return self._v_net

  @property
  def v_variables(self):
    vars_ = []
    for param in self._v_net._layers:
        if not isinstance(param, nn.ReLU):
               vars_+=list(param.parameters())
    return vars_

  @property
  def v_weights(self):
    v_weights = []
    for name, param in self._v_net._layers.named_parameters():
        if 'weight' in name:
            v_weights += list(param)
    return v_weights

  @property
  def p_net(self):
    return self._p_net

  def p_fn(self, s):
    return self._p_net(s)

  @property
  def p_weights(self):
    p_weights = []
    for name, param in self._p_net._layers.named_parameters():
        if 'weight' in name:
            p_weights += list(param)
    return p_weights


  @property
  def p_variables(self):
    vars_=[]
    for param in self._p_net._layers:
        if not isinstance(param, nn.ReLU):
           vars_+=list(param.parameters())
    return vars_

  @property
  def c_net(self):
    return self._c_net

  @property
  def c_weights(self):
    c_weights = []
    for name, param in self._c_net._layers.named_parameters():
        if 'weight' in name:
            c_weights += list(param)
    return c_weights

  @property
  def c_variables(self):
    vars_=[]
    for param in self._c_net._layers:
        if not isinstance(param, nn.ReLU):
           vars_+=list(param.parameters())
    return vars_

class Agent(object):
  """Class for learning policy and interacting with environment."""

  def __init__(
      self,
      observation_spec=None,
      action_spec=None,
      time_step_spec=None,
      modules=None,
      optimizers= ((0.001, 0.5, 0.99),),
      batch_size=64,
      weight_decays=(0.5,),
      update_freq=1,
      update_rate=0.005,
      discount=0.99,
      env_name='HalfCheetah-v2',      
      train_data=None,
      resume=False, 
      device=None
      ):
    self._latent_spec = observation_spec
    self._action_spec = action_spec
    self._time_step_spec = time_step_spec
    self._modules_list = modules
    self._optimizers = optimizers
    self._batch_size = batch_size
    self._weight_decays = weight_decays
    self._train_data = train_data
    self._update_freq = update_freq
    self._update_rate = update_rate
    self._discount = discount
    self._env_name = env_name    
    self._resume = resume 
    if device is None:
       self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
    else:
       self.device=device
    
    directory = os.getcwd()
    checkpoint_dir=directory+"/run"
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    self._build_agent()

  def _build_agent(self):
    """Builds agent components."""
    if len(self._weight_decays) == 1:
       self._weight_decays = tuple([self._weight_decays[0]] * 4)
    self._build_fns()
    train_batch = self._get_train_batch()
    self._global_step = torch.tensor(0.0, requires_grad=False)
    self._init_vars(train_batch)
    self._build_optimizers()
    self._train_info = collections.OrderedDict()
    self._checkpointer = self._build_checkpointer()
    self._test_policies = collections.OrderedDict()
    self._build_test_policies()
    self._online_policy = self._build_online_policy()
    

  def _build_fns(self):
    self._agent_module = AgentModule(modules_list=self._modules_list)

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
      #?
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
  
  #def _random_policy_fn(self, state):
  #    return self._action_spec.sample(), None


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

            
  def _train_step(self):
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
    utils.write_summary(summary_writer, info, step)

  def _build_checkpointer(self):
      #?save
      pass

  def _load_checkpoint(self):
      #?restore
      pass

  @property
  def global_step(self):
    return self._global_step.numpy()
  
################# Policies #################
class ContinuousRandomPolicy(nn.Module):
  """Samples actions uniformly at random."""

  def __init__(self, action_spec):
    super(ContinuousRandomPolicy, self).__init__()
    self._action_spec = action_spec
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(device=self.device)
    
  def __call__(self, observation, state=()):
    action = tensor_specs.sample_bounded_spec(
        self._action_spec, 
        outer_dims=[observation.shape[0]]
        )
    return action, state

class GaussianRandomSoftPolicy(nn.Module):
  """Adds Gaussian noise to actor's action."""

  def __init__(self, a_network, std=0.1, clip_eps=1e-3):
    super(GaussianRandomSoftPolicy, self).__init__()
    self._a_network = a_network
    self._std = std
    self._clip_eps = clip_eps
    self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
    self.to(device=self.device)

  def __call__(self, observation, state=()):
    action = self._a_network(observation.to(device=self.device))[1]
    action=action.squeeze()
    noise = torch.normal(mean=torch.zeros(action.shape), std=self._std).to(device=self.device)
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
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(device=self.device)

  def __call__(self, latent_states):
    action = self._a_network(latent_states)[0]
    return action

class RandomSoftPolicy(nn.Module):
  """Returns sample from policy distribution."""

  def __init__(self, a_network):
    super(RandomSoftPolicy, self).__init__()
    self._a_network = a_network
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(device=self.device)

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
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def __call__(self, latent_state):

    actions = self._a_network.sample_n(latent_state.to(device=self.device), self._n)[1]
    #logging.info("latent states shape:{}, n: {}, actions: {}......".format(latent_state.shape, self._n, actions.shape))
    batch_size = actions.shape[-1]
    actions_ = torch.reshape(actions, [self._n * batch_size, -1])
    states_  = torch.tile(latent_state[None].to(device=self.device), (self._n, 1, 1))
    states_  = torch.reshape(states_, [self._n * batch_size, -1])
    qvals = self._q_network(states_, actions_)
    qvals = torch.reshape(qvals, [self._n, batch_size]).to(device=self.device)
    a_indices = torch.argmax(qvals, dim=0).to(device=self.device)
    gather_indices = torch.stack(
        [a_indices, torch.arange(batch_size, dtype=torch.int64).to(device=self.device)], dim=-1)
    action = utils.gather_nd(actions, gather_indices)
    return action
#############################################
################ D2E Agent ################## 
#############################################


gin.clear_config()
gin.config._REGISTRY.clear()
@gin.configurable
class D2EAgent(Agent):
  """dual agent class."""

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
      EIM_config = ConditionalMixtureEIM.get_default_config(),
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
    self._EIM_config = EIM_config
    self._warm_start = warm_start
    self._c_iter = c_iter
    self._ensemble_q_lambda = ensemble_q_lambda
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    super(D2EAgent, self).__init__(**kwargs)

  def _build_fns(self):
    self._agent_module = AgentModule(modules_list=self._modules_list)
    self._q_fns = self._agent_module.q_nets
    self._p_fn = self._agent_module.p_fn
    self._c_fn = self._agent_module.c_net
    self._v_fn = self._agent_module.v_net #adding value network
    self._divergence = utils.get_divergence(
        name=self._divergence_name, 
        c=self._c_fn,
        device=self.device)
    #################################################################################################################################
    ############ Setting Up Configuration parameters of the EIM model to compute desity ratio of the transition function ############
    #################################################################################################################################
    self._EIM_config.train_epochs = 25
    self._EIM_config.num_components = self._latent_spec.shape[0] #number of dimension of latent space

    self._EIM_config.components_net_hidden_layers = [30, 30]
    self._EIM_config.components_batch_size = self._latent_spec.shape[0] #is it a correct batch size???
    self._EIM_config.components_num_epochs = 10
    self._EIM_config.components_net_reg_loss_fact = 0.0
    self._EIM_config.components_net_drop_prob = 0.0

    self._EIM_config.gating_net_hidden_layers = [30, 30]
    self._EIM_config.gating_batch_size = self._latent_spec.shape[0] #???
    self._EIM_config.gating_num_epochs = 10
    self._EIM_config.gating_net_reg_loss_fact = 0.0
    self._EIM_config.gating_net_drop_prob = 0.0

    self._EIM_config.dre_reg_loss_fact = 0.0005
    self._EIM_config.dre_early_stopping = True
    self._EIM_config.dre_drop_prob = 0.0
    self._EIM_config.dre_num_iters =  25
    self._EIM_config.dre_batch_size = self._latent_spec.shape[0] ###???
    self._EIM_config.dre_hidden_layers = [30, 30]
    ########################################################################################################################
    ##########################################   END of EIM cofiguration SETUP    ##########################################
    ########################################################################################################################
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
  
  def _get_v_vars(self):
    return self._agent_module.v_variables

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
  
  def _get_v_weight_norm(self):
    weights = self._agent_module.v_weights
    norms = []
    for w in weights:
      norm = torch.sum(torch.square(w))
      norms.append(norm)
    return torch.stack(norms).sum(dim=0)


  def ensemble_q(self, qs):
    lambda_ = self._ensemble_q_lambda
    return (lambda_ *torch.min(qs, dim=-1).values
            + (1 - lambda_) * torch.max(qs, dim=-1).values)

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
    _, a2_p, log_pi_a2_p = self._p_fn(s2.to(device=self.device))
    q2_targets = []
    q1_preds = []
    for q_fn, q_fn_target in self._q_fns:
      q2_target_ = q_fn_target(s2.to(device=self.device), a2_p)
      q1_pred = q_fn(s1.to(device=self.device), a1.to(device=self.device))
      q1_preds.append(q1_pred)
      q2_targets.append(q2_target_)
    q2_targets = torch.stack(q2_targets, dim=-1)
    q2_target = self._ensemble_q2_target(q2_targets)
    div_estimate = self._divergence.dual_estimate(
        s2.to(device=self.device), a2_p, a2_b.to(device=self.device))
    
    v2_target = q2_target - self._v_fn(s2.to(device=self.device))# Equation 21 in Dream to Explore
    if self._value_penalty:
       v2_target = v2_target - self._get_alpha()[0] * div_estimate
    with torch.no_grad():
         q1_target = r.to(device=self.device) + dsc.to(device=self.device) * self._discount * v2_target #Q(s,a)=R(s,a)+discount*v(s')
    q_losses = []
    for q1_pred in q1_preds:
      q_loss_ = torch.mean(torch.square(q1_pred - q1_target))
      q_losses.append(q_loss_)
    q_loss = torch.sum(torch.FloatTensor(q_losses))
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
    info['v2_target_mean'] = torch.mean(v2_target) #added in D2E

    return loss, info

  def _build_p_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    _, a_p, log_pi_a_p = self._p_fn(s.to(device=self.device))
    v1  = self._v_fn(s.to(device=self.device))
    q1s = []
    for q_fn, _ in self._q_fns:
        q1_ = q_fn(s.to(device=self.device), a_p)
        q1s.append(q1_)
    q1s = torch.stack(q1s, dim=-1)
    q1 = self._ensemble_q1(q1s)
    div_estimate = self._divergence.dual_estimate(
        s.to(device=self.device), a_p, a_b.to(device=self.device))
    q_start = torch.gt(self._global_step, self._warm_start).type(torch.float32)
    p_loss = torch.mean(
        self._get_alpha_entropy()[0] * log_pi_a_p
        + self._get_alpha()[0] * div_estimate + v1 #new term based on Equation 10 in dream to explore paper
        - q1 * q_start)
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays[1] * p_w_norm
    loss = p_loss + norm_loss

    info = collections.OrderedDict()
    info['p_loss'] = p_loss
    info['p_norm'] = p_w_norm

    return loss, info
  
  def _recorder_EIM(self,data):
      """configure experiment"""
      plot_realtime = True
      plot_save = False    
      print(f"env: {self._env_name}")
      """Recording"""
      recorder_dict = {
         RecorderKeys.TRAIN_ITER: TrainIterationRecMod(),
         RecorderKeys.INITIAL: ConfigInitialRecMod(),
         #RecorderKeys.MODEL: MujocoModelRecMod(data,
         #                                      self._env_name,
         #                                      train_samples=data.train_samples,
         #                                      test_samples=data.test_samples),
         RecorderKeys.DRE: DRERecMod(data.train_samples),
         RecorderKeys.COMPONENT_UPDATE: ComponentUpdateRecMod(plot=True, summarize=False)}
      if self._EIM_config.num_components > 1:
         recorder_dict[RecorderKeys.WEIGHTS_UPDATE] = WeightUpdateRecMod(plot=True)


      return Recorder(recorder_dict, plot_realtime=plot_realtime, save=plot_save, save_path="rec")

  def _build_v_loss(self, batch):
    s1  = batch['s1']
    a_b = batch['a1']
    s2  = batch['s2']
    _, a_p, _ = self._p_fn(s1.to(device=self.device))
    v_target = self._v_fn(s2.to(device=self.device))
    q1_target = []
    for q_fn, q_fn_target in self._q_fns:
      q1_ = q_fn_target(s1.to(device=self.device), a_b.to(device=self.device))
      q1_target.append(q1_)
    q1_target = torch.stack(q1_target, dim=-1)
    q1_target_ensemble = self.ensemble_q(q1_target)
    div_estimate = self._divergence.dual_estimate(
        s1.to(device=self.device), a_p, a_b.to(device=self.device))
    #########################
    #input_data next_state (s2) and state (s1) 
    data = MujocoData(s1,s2)
    self._EIM_model = ConditionalMixtureEIM(self._EIM_config, train_samples=data.train_samples, seed=torch.initial_seed(), recorder=self._recorder_EIM(data), val_samples=data.val_samples)
    EIM_gate_KL=0
    if self._EIM_model._model.num_components > 1:
       EIM_gate_res =self._EIM_model.update_gating()
       print(f"EIM gate residual {EIM_gate_res}")
       for i in range(self._EIM_model._model.num_components):
           EIM_gate_KL=EIM_gate_res[i][1]
    EIM_res = self._EIM_model.update_components()
    EIM_KL=0 
    for i in range(len(EIM_res)):
        EIM_KL+=EIM_res[i][1]

    #Equation 20 in Dream to Explore paper
    v_loss = torch.mean(q1_target_ensemble - v_target
        - self._get_alpha()[0] * (div_estimate 
        + EIM_KL + EIM_gate_KL))
    v_w_norm = self._get_v_weight_norm()
    norm_loss = self._weight_decays[3] * v_w_norm
    loss = v_loss + norm_loss

    info = collections.OrderedDict()
    info['value_loss'] = v_loss
    info['vale_norm'] = v_w_norm
    return loss, info


  def _build_c_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    _, a_p, _ = self._p_fn(s.to(device=self.device))
    c_loss = self._divergence.dual_critic_loss(
        s.to(device=self.device), a_p, a_b.to(device=self.device))
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
    _, a_p, _ = self._p_fn(s.to(device=self.device))
    alpha = self._get_alpha()
    div_estimate = self._divergence.dual_estimate(
        s.to(device=self.device), a_p, a_b.to(device=self.device))
    a_loss = - torch.mean(alpha * (div_estimate - self._target_divergence))

    info = collections.OrderedDict()
    info['a_loss'] = a_loss
    info['alpha'] = alpha
    info['div_mean'] = torch.mean(div_estimate)
    info['div_std'] = torch.std(div_estimate)

    return a_loss, info

  def _build_ae_loss(self, batch):
    s = batch['s1']
    _, _, log_pi_a = self._p_fn(s.to(device=self.device))
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
      opts = tuple([opts[0]] * 5)
    elif len(opts) < 5:
      raise ValueError('Bad optimizers %s.' % opts)
      
    self._q_optimizer = torch.optim.Adam(self._get_q_vars(),lr=opts[0][0], betas=(opts[0][1],opts[0][2]), weight_decay=self._weight_decays[0])
    self._p_optimizer = torch.optim.Adam(self._get_p_vars(),lr=opts[1][0], betas=(opts[1][1],opts[1][2]), weight_decay=self._weight_decays[1])
    self._c_optimizer = torch.optim.Adam(self._get_c_vars(),lr=opts[2][0], betas=(opts[2][1],opts[2][2]), weight_decay=self._weight_decays[2])
    self._c_optimizer = torch.optim.Adam(self._get_v_vars(),lr=opts[3][0], betas=(opts[3][1],opts[3][2]), weight_decay=self._weight_decays[3])
    self._a_optimizer = torch.optim.Adam(self._a_vars, lr=opts[4][0], betas=(opts[4][1],opts[4][2]))
    self._ae_optimizer = torch.optim.Adam(self._ae_vars, lr=opts[4][0], betas=(opts[4][1],opts[4][2]))

  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    if torch.equal(self._global_step % torch.tensor(self._update_freq), torch.tensor(0, dtype=torch.float32)):
      source_vars, target_vars = self._get_source_target_vars()
      self._update_target_fns(source_vars, target_vars)
    # Update policy network parameter
    #https://bit.ly/3Bno0GC
    # policy network's update should be done before updating q network, or there will make some errors
    self._agent_module.p_net.train()
    self._p_optimizer.zero_grad()
    policy_loss,_=self._build_p_loss(batch)
    policy_loss.backward(retain_graph=True)
    self._p_optimizer.step()
    # Update value network
    self._agent_module.v_net.train()
    self._v_optimizer.zero_grad()
    value_loss,_=self._build_v_loss(batch)
    value_loss.backward(retain_graph=True)
    self._v_optimizer.step()
    # Update q networks parameter
    for q_fn, q_fn_target in self._q_fns:
        q_fn.train()
        q_fn_target.train()
    self._q_optimizer.zero_grad()
    q_losses,q_info= self._build_q_loss(batch)
    q_losses.backward()
    self._q_optimizer.step()
    #Update critic network parameter
    self._agent_module.c_net.train()
    self._c_optimizer.zero_grad()
    critic_loss,_= self._build_c_loss( batch)
    critic_loss.backward()
    self._c_optimizer.step()
    self._extra_c_step( batch)
    #train expected information maximization to compute transition ratio
    self._EIM_model.train_emm()
    
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
    #) value loss
    info["value_loss"]= value_loss.cpu().item()
    #7) critic loss
    info["critic_loss"] = self.critic_loss.cpu().item()
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
      self.critic_loss,_ = self._build_c_loss( batch)
      self.critic_loss.backward()
      self._c_optimizer.step()

  def train_step(self):
    train_batch = self._get_train_batch()
    info = self._optimize_step(train_batch)
    for _ in range(self._c_iter - 1):
      train_batch = self._get_train_batch()
      self._extra_c_step(train_batch)
    for key, val in info.items():
        self._train_info[key] = val
    self._global_step += 1


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
    self._build_v_loss(batch)
    self._q_vars = self._get_q_vars()
    self._p_vars = self._get_p_vars()
    self._c_vars = self._get_c_vars()
    self._v_vars = self._get_v_vars()
    self._a_vars = self._agent_module.a_variables
    self._ae_vars = self._agent_module.ae_variables

  def _build_checkpointer(self):
      checkpoint = {
            "policy_net": self._agent_module.p_net.state_dict(),
            "critic_net": self._agent_module.c_net.state_dict(),
            "value_net" : self._agent_module.v_net.state_dict(),
            "q_optimizer": self._q_optimizer.state_dict(),
            "critic_optimizer": self._c_optimizer.state_dict(),
            "policy_optimizer": self._p_optimizer.state_dict(),
            "value_optimizer": self._v_optimizer.state_dict(),
            "train_step": self._global_step
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
        self._agent_module.p_net.load_state_dict(checkpoint["policy_net"])
        self._agent_module.c_net.load_state_dict(checkpoint["critic_net"])
        self._agent_module.v_net.load_state_dict(checkpoint["value_net"])
        self._p_optimizer.load_state_dict(checkpoint["policy_optimizer1"])
        self._q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self._c_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._v_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self._global_step = checkpoint["train_step"]
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
   'randwalk',
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


def load_policy(policy_cfg, action_spec, observation_spec):
  """Loads policy based on config."""
  if policy_cfg.ptype not in PTYPES:
    raise ValueError('Unknown policy type %s.' % policy_cfg.ptype)
  if policy_cfg.ptype == 'randwalk':
    policy = ContinuousRandomPolicy(action_spec)
  elif policy_cfg.ptype in ['randinit', 'load']:
    a_net = ActorNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=policy_cfg.model_params)
    if policy_cfg.ptype == 'load' and os.path.exists(policy_cfg.ckpt):
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
    while not time_step.is_last():
      action = policy(torch.from_numpy(time_step.observation))[0]
      if action.ndim<1:
          time_step= env.step(action.unsqueeze(0).detach().cpu()) 
      else:  
          time_step= env.step(action.detach().cpu())
      total_rewards += time_step.reward or 0.
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
############################

class AgentConfig(object):
  """Class for handling agent parameters."""

  def __init__(self, agent_flags):
    self._agent_flags = agent_flags
    self._agent_args = self._get_agent_args()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def _get_agent_args(self):
    """Gets agent parameters associated with config."""
    agent_flags = self._agent_flags
    agent_args = utils.Flags(
        observation_spec=agent_flags.observation_spec,
        action_spec=agent_flags.action_spec,
        optimizers=agent_flags.optimizers,
        batch_size=agent_flags.batch_size,
        weight_decays=agent_flags.weight_decays,
        update_rate=agent_flags.update_rate,
        update_freq=agent_flags.update_freq,
        discount=agent_flags.discount,
        env_name=agent_flags.env_name,
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
        self._agent_flags.observation_spec,
        self._agent_flags.action_spec)
#more hyperparameter run_dual.sh


  