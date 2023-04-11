import os
import numpy as np
import itertools
from tqdm import tqdm
import logging
import cv2
import imageio
import re
import functools
import math
from numbers import Number
from collections import namedtuple, deque
from typing import Iterables, Tuple, Dict, List
import gym
import mujoco_py
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim, jit
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from planner_D2E_regularizer import D2EAgent, Config, Agent, parse_policy_cfg, Transition, map_structure, maybe_makedirs, load_policy, eval_policy_episodes
import utils_planner as utils
from Hierarchical_StickBreaking_GMMVAE import InfGaussMMVAE, VAECritic, gradient_penalty
from VRNN.main import ModelState
from VRNN.Normalization import compute_normalizer
from VRNN.Blocks import init_weights
import DataCollectionD2E_WM  as DC
import alf_gym_wrapper
import argparse
from alf_environment import TimeLimit
#####################################
hyperParams = { "batch_size": 40,
                "input_d": 1,
                "prior_alpha": 7., #gamma_alpha
                "prior_beta": 1., #gamma_beta
                "K": 25,
                "image_width": 96,
                "hidden_d": 300,
                "latent_d": 100,
                "latent_w": 200,
                "hidden_transit": 100,
                "LAMBDA_GP": 10, #hyperparameter for WAE with gradient penalty
                "LEARNING_RATE": 2e-4,
                "CRITIC_ITERATIONS" : 5,
                "GAMMA": 0.99,
                "PREDICT_DONE": False,
                "seed": 1234,
                "number_of_mixtures": 8,
                "weight_decay": 1e-5,
                "n_channel": 3,
                "VRNN_Optimizer_Type":"MADGRAD",
                "MAX_GRAD_NORM": 100.,
                "expl_behavior": greedy,
                "expl_noise": 0.0,
                "eval_noise": 0.0,
                "replay_buffer_size":int(1e4),
                }
#####################################
class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

#partially based on https://github.com/sai-prasanna/mpc_and_rl/blob/1fedaec5eec7a2a5844935d17b4fd0ad385d9d6c/dreamerv2_torch/dreamerv2_torch/common/other.py
#inspired by https://github.com/mahkons/Dreamer/blob/003c3cc7a9430e9fa0d8af9cead88d8f4b06e0f4/dreamer/WorldModel.py

@functools.total_ordering
class Counter:
    def __init__(self, initial=0):
        self.value = initial

    def __int__(self):
        return int(self.value)

    def __eq__(self, other):
        return int(self) == other

    def __ne__(self, other):
        return int(self) != other

    def __lt__(self, other):
        return int(self) < other

    def __add__(self, other):
        return int(self) + other

    def increment(self, amount=1):
        self.value += amount

def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step = step.to(torch.float32)
        match = re.match(r"linear\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r"warmup\((.+),(.+)\)", string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clip(step / warmup, 0, 1)
            return scale * value
        match = re.match(r"exp\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r"horizon\((.+),(.+),(.+)\)", string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clip(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)

def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.unbind(outputs, dim=0)
    outputs = torch.stack(outputs, dim=1)
    return outputs

def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns

#####################################################

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(
            batch_shape, validate_args=validate_args
        )
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any(
            (self.a >= self.b)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return (
            super(TruncatedNormal, self).log_prob(self._to_std_rv(value))
            - self._log_scale
        )
class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = (samples,)

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)

class OneHotDist(torch.distributions.OneHotCategorical):
    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample = sample + (probs - probs.detach())
        return sample

class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x

def action_noise(action, amount, act_space):
    if amount == 0:
        return action
    amount = amount.to(action.dtype)
    if hasattr(act_space, "n"):
        probs = amount / action.shape[-1] + (1 - amount) * action
        return OneHotDist(probs=probs).sample()
    else:
        return torch.clip(torch.distributions.Normal(action, amount).sample(), -1, 1)


class StreamNorm(nn.Module):
    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        super().__init__()
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.mag = nn.Parameter(
            torch.ones(shape, dtype=torch.float64), requires_grad=False
        )

    def forward(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics["mean"] = inputs.mean().detach().cpu()
        metrics["std"] = inputs.std().detach().cpu()
        outputs = self.transform(inputs)
        metrics["normed_mean"] = outputs.mean().detach().cpu()
        metrics["normed_std"] = outputs.std().detach().cpu()
        return outputs, metrics

    def reset(self):
        self.mag.data = torch.ones_like(self.mag)

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self._shape)
        mag = torch.abs(batch).mean(0).type(torch.float64)
        self.mag.data = self._momentum * self.mag + (1 - self._momentum) * mag

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self._shape)
        values /= self.mag.data.type(inputs.dtype)[None] + self._eps
        values *= self._scale
        return values.reshape(inputs.shape)


###############################################################################
########### prepare data for transition model by padding episodes #############
def make_episodes(states, actions, next_states, dones):
    pairs, inputs, score, targets =[], [], [], []
    for i in range(len(dones)):
        if dones[i]==False:            
           pairs.append(torch.cat([states[i], actions[i]]))
           score.append(next_states[i])
        else:
           pairs.append(torch.cat([states[i], actions[i]]))
           score.append(next_states[i])
           inputs.append(pairs)
           pairs=[]
           targets.append(score)
           score=[]
    return inputs, targets

def pad(episodes, repeat=True):
    """Pads episodes to all be the same length by repeating the last exp.
    Args:
        episodes (list[list[Experience]]): episodes to pad.
    Returns:
        padded_episodes (list[list[Experience]]): now of shape
            (batch_size, max_len)
        mask (torch.BoolTensor): of shape (batch_size, max_len) with value 0 for
            padded experiences.
    """
    max_len = max(len(episode) for episode in episodes)
    
    mask = torch.zeros((len(episodes), max_len), dtype=torch.bool)
    padded_episodes = []
    for i, episode in enumerate(episodes):
        if repeat:
           padded = episode + [episode[-1]] * (max_len - len(episode))
        else:
           padded = episode + [torch.zeros_like(episode[-1])] * (max_len - len(episode))
        padded_episodes.append(padded)
        mask[i, :len(episode)] = True
    return padded_episodes, mask

def preprocess_transition_data(s1, a1, s2, done):
    inputs, outputs = make_episodes(s1, a1, s2, done)
    padded_trajectories_inputs, mask_input = pad(inputs, repeat=False)
    padded_trajectories_outputs, mask_output = pad(outputs, repeat=False)
    final_inputs  = torch.stack([ torch.stack([padded_trajectories_inputs[i][j] for j in range(mask_input.shape[1])], dim=0) for i in range(mask_input.shape[0])], dim=0)#(batch_size , max_len, state_dim)
    final_outputs = torch.stack([ torch.stack([padded_trajectories_outputs[i][j] for j in range(mask_output.shape[1])], dim=0) for i in range(mask_output.shape[0])], dim=0)#(batch_size , max_len, state_dim)
    #shape of outputs: (B , max_seq_len, D)
    return final_inputs, final_outputs
###############################################################################
class VideoRecorder:
    def __init__(self, root_dir, render_size=64, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)



class logger:
      def __init__ (self):
          self.fmt = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
          self.level = logging.INFO
          self.datefmt = '%Y-%m-%d %H:%M:%S'

class Logger(logging.Logger):
    def __init__(self, name: str, level=None) -> None:
        if level is None:
            level = logger.level
        super().__init__(name, level=level)
        formatter = logging.Formatter(
            fmt=logger.fmt,
            datefmt=logger.datefmt,
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logger.level)
        self.addHandler(handler)

class Optimizer(nn.Module):
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        super().__init__()
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._params = list(parameters)
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(self._params, lr=lr, eps=eps),
            "adamw": lambda: torch.optim.AdamW(self._params, lr= lr, betas = (0.9, 0.999), amsgrad = True),
            "adamax": lambda: torch.optim.Adamax(self._params, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(self._params, lr=lr),
            "momentum": lambda: torch.optim.SGD(self._params, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(self._params, self._clip)
        if self._wd:
            self._apply_weight_decay(self._params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)
#################################

class StandardScaler(object):

    def __init__(self, device):
        self.input_mu = torch.zeros(1).to(device)
        self.input_std = torch.ones(1).to(device)
        self.target_mu = torch.zeros(1).to(device)
        self.target_std = torch.ones(1).to(device)
        self.device = device

    def fit(self, inputs, targets, scale_dim=0):
        """
        Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the input
        targets : torch.Tensor
            A torch Tensor containing the input
        """
        self.input_mu = torch.mean(inputs, dim=scale_dim, keepdims=True).to(self.device)
        self.input_std = torch.std(inputs, dim=scale_dim, keepdims=True).to(self.device)
        self.input_std[self.input_std < 1e-8] = 1.0
        self.target_mu = torch.mean(targets, dim=scale_dim, keepdims=True).to(self.device)
        self.target_std = torch.std(targets, dim=scale_dim, keepdims=True).to(self.device)
        self.target_std[self.target_std < 1e-8] = 1.0

    def transform(self, inputs, targets=None):
        """
        Transforms the input matrix data using the parameters of this scaler.
        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the points to be transformed.
        targets : torch.Tensor
            A torch Tensor containing the points to be transformed.
        Returns
        -------
        norm_inputs : torch.Tensor
            Normalized inputs
        norm_targets : torch.Tensor
            Normalized targets
        """
        norm_inputs = (inputs - self.input_mu) / self.input_std
        norm_targets = None
        if targets is not None:
            norm_targets = (targets - self.target_mu) / self.target_std
        return norm_inputs, norm_targets

    def inverse_transform(self, targets):
        """
        Undoes the transformation performed by this scaler.
        Parameters
        ----------
        targets : torch.Tensor
            A torch Tensor containing the points to be transformed.
        Returns
        -------
        output : torch.Tensor
            The transformed dataset.
        """
        output = self.target_std * targets + self.target_mu
        return output

class Scale(nn.Module):
    """
    Maps inputs from [space.low, space.high] range to [-1, 1] range.
    Parameters
    ----------
    space : gym.Space
        Space to map from.
    Attributes
    ----------
    low : torch.tensor
        Lower bound for unscaled Space.
    high : torch.tensor
        Upper bound for unscaled Space.
    """
    def __init__(self, space):
        super(Scale, self).__init__()
        self.register_buffer("low", torch.from_numpy(space.low))
        self.register_buffer("high", torch.from_numpy(space.high))

    def forward(self, x):
        """
        Maps x from [space.low, space.high] to [-1, 1].
        Parameters
        ----------
        x : torch.tensor
            Input to be scaled
        """
        return 2.0 * ((x - self.low) / (self.high - self.low)) - 1.0


class Unscale(nn.Module):
    """
    Maps inputs from [-1, 1] range to [space.low, space.high] range.
    Parameters
    ----------
    space : gym.Space
        Space to map from.
    Attributes
    ----------
    low : torch.tensor
        Lower bound for unscaled Space.
    high : torch.tensor
        Upper bound for unscaled Space.
    """
    def __init__(self, space):
        super(Unscale, self).__init__()
        self.register_buffer("low", torch.from_numpy(space.low))
        self.register_buffer("high", torch.from_numpy(space.high))

    def forward(self, x):
        """
        Maps x from [-1, 1] to [space.low, space.high].
        Parameters
        ----------
        x : torch.tensor
            Input to be unscaled
        """
        return self.low + (0.5 * (x + 1.0) * (self.high - self.low))


def get_act(name):
    if name == "none":
        return lambda x: x
    if name == "mish":
        return lambda x: x * torch.tanh(nn.Softplus()(x))
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)()
    else:
        raise NotImplementedError(name)

class Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._lazy_modules = nn.ModuleDict({})

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if name not in self._lazy_modules:
            self._lazy_modules[name] = ctor(*args, **kwargs)
        return self._lazy_modules[name]

class DistLayer(Module):
    def __init__(self, shape, dist="mse", min_std=0.1, init_std=0.0):
        super().__init__()
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def forward(self, inputs):
        out = self.get("out", nn.Linear, inputs.shape[-1], int(np.prod(self._shape)))(
            inputs
        )
        out = torch.reshape(out, inputs.shape[:-1] + self._shape)
        if self._dist in ("normal", "tanh_normal", "trunc_normal"):
            std = self.get("std", nn.Linear, inputs.shape[-1], np.prod(self._shape))(
                inputs
            )
            std = torch.reshape(std, inputs.shape[:-1] + self._shape)
        if self._dist == "mse":
            dist = torch.distributions.Normal(out, 1.0)
            return ContDist(torch.distributions.Independent(dist, len(self._shape)))
        if self._dist == "normal":
            dist = torch.distributions.Normal(out, std)
            return ContDist(torch.distributions.Independent(dist, len(self._shape)))
        if self._dist == "binary":
            dist = Bernoulli(torch.distributions.Independent(torch.distributions.Bernoulli(logits=out), len(self._shape)))
            return dist
        if self._dist == "tanh_normal":
            mean = 5 * torch.tanh(out / 5)
            std = nn.Softplus()(std + self._init_std) + self._min_std
            dist = torch.distributions.Normal(mean, std)
            dist = torch.distributions.TransformedDistribution(dist, TanhBijector())
            dist = torch.distributions.Independent(dist, len(self._shape))
            return SampleDist(dist)
        if self._dist == "trunc_normal":
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = TruncatedNormal(torch.tanh(out), std, -1, 1)
            return ContDist(torch.distributions.Independent(dist, 1))
        if self._dist == "onehot":
            return OneHotDist(out)
        raise NotImplementedError(self._dist)


class MLP(Module):
    def __init__(self, shape, layers, units, act="elu", norm="none", **out):
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def forward(self, features):
        x = features
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f"dense{index}", nn.Linear, x.shape[-1], self._units)(x)
            x = self.get(f"norm{index}", NormLayer, x.shape[-1], self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + (x.shape[-1],))
        return self.get("out", DistLayer, self._shape, **self._out)(x)

class D2EDataset(Dataset):
    def __init__(self, REPLAY_SIZE):
        self.buffer = deque(maxlen=REPLAY_SIZE)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, indx):
        item = self.buffer[indx]
        return item.s1, item.s2, item.reward, item.discount, item.a1, item.a2, item.done

    def append(self, Transition):
        self.buffer.append(Transition)


class WorldModel(jit.ScriptModule):
    def __init__(self, 
                 hyperParams, 
                 sequence_length, 
                 dataset,
                 env_name='Hopper-v2', 
                 device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 log_dir = "logs", 
                 restore = False):
        super(WorldModel, self).__init__()
        
        
        self.model_path = os.path.abspath(os.getcwd()) + '/model'
        try:
           os.makedirs(self.model_path, exist_ok=True) 
           print("Directory '%s' created successfully" %self.model_path)
        except OSError as error:
           print("Directory '%s' can not be created")
        self._params = namedtuple('x', hyperParams.keys())(*hyperParams.values())
        self._discount       = self._params.GAMMA
        #set the environment 
        self._env_name = env_name
        env_gym =  gym.spec(env_name).make()
        env = alf_gym_wrapper.AlfGymWrapper(env_gym,discount=self._discount)
        env = TimeLimit(env, DC.MUJOCO_ENVS_LENGTH[env_name])
        self._env            = env
        self.n_discriminator_iter = self._params.CRITIC_ITERATIONS
        self._use_amp        = False
        
        #set the sizes
        self.state_dim       = self._params.latent_d
        self.action_dim      = env.action_spec()
        self. observation_dim= env.observation_spec()
        #question: do I need to define this here???
        self._data =   Dataset(env.observation_spec,
                               env.action_spec,
                               self._params.replay_buffer_size,
                               circular=True,
                               )
        self.dataset = dataset
        self.device          = device
        self._clip_rewards   = "tanh"
        self.heads = nn.ModuleDict()
        self.heads["reward"]    = MLP(shape=1, layers= 4, units= 400, act= "ELU", norm= "none", dist= "mse").to(self.device)
        self.heads["discount"]  = MLP(shape=1, layers= 4, units= 400, act= "ELU", norm= "none", dist= "binary").to(self.device)
        DiscountNetwork.create(self.state_dim + self.action_dim, self._params.PREDICT_DONE, self._discount).to(self.device)
        self.ckpt_path       = self.model_path+'/best_model'
        os.makedirs(self.ckpt_path, exist_ok=True) 
        self.logger = Logger(self.__class__.__name__)
        self.mse_loss = nn.MSELoss()
        modelstate =  ModelState(seed            = self._params.seed,
                                 nu              = self.state_dim + self.action_dim,
                                 ny              = self.state_dim,
                                 sequence_length = sequence_length,
                                 h_dim           = self._params.hidden_transit,
                                 z_dim           = self.state_dim,
                                 n_layers        = 2,
                                 n_mixtures      = self._params.number_of_mixtures,
                                 device          = device,
                                 optimizer_type  = self._params.VRNN_Optimizer_Type,
                                )
        self.standard_scaler = StandardScaler(self.device)
        self.transition_model = modelstate.model
        self.variational_autoencoder = InfGaussMMVAE(hyperParams,
                                                     K          = self._params.K,
                                                     nchannel   = self._params.n_channel,
                                                     z_dim      = self.state_dim,
                                                     w_dim      = self._params.latent_w,
                                                     hidden_dim = self._params.hidden_d,
                                                     device     = self.device,
                                                     img_width  = self._params.image_width,
                                                     batch_size = self._params.batch_size,
                                                     num_layers = 4,
                                                     include_elbo2=True,
                                                     use_mse_loss=True)
        self.encoder = self.variational_autoencoder.GMM_encoder
        self.decoder = self.variational_autoencoder.GMM_decoder
        self.discriminator = VAECritic(self.state_dim)
        self.grad_heads = [ "reward", "discount"]
        self.loss_scales: { reward: 1.0, discount: 1.0}
        for name in self.grad_heads:
            assert name in self.heads, name

        self.parameters = itertools.chain(
            self.transition_model.parameters(),
            self.heads.parameters(),
            self.encoder.parameters(),
            self.decoder.parameters(),
        )
        if restore:
           self.load_checkpoints()

        self.writer = SummaryWriter(log_dir)
    
    def initialize_optimizer(self):
        self.optimizer = Optimizer("world_model",
                                   self.parameters,
                                   lr = self._params.LEARNING_RATE,
                                   wd = self._params.weight_decay,
                                   opt= "adamw",
                                   use_amp=self._use_amp)
        
        self.discriminator_optim = Optimizer("discriminator_model",
                                             self.discriminator.parameters(),
                                             lr = 0.5 * self._params.LEARNING_RATE,
                                             opt= "adam",
                                             use_amp=self._use_amp)
        
    def preprocess(self, 
                   observation, 
                   next_observation, 
                   reward, 
                   done):
        data["reward"]    = {
                            "identity": lambda x: x,
                            "sign": torch.sign,
                            "tanh": torch.tanh,
                            }[self._clip_rewards](reward).unsqueeze(-1)
        data["discount"]  = 1.0 - done.float().unsqueeze(-1)  # .to(dtype)
        data["discount"] *= self._discount
        dtype = torch.float32
        data["observation"], data["next_observation"] = (i.to(dtype) if i.dtype==torch.int32 else i.to(dtype)/255.0-0.5 if i.dtype == torch.uint8 else i.to(i.dtype) for i in [observation, next_observation])
        data["done"] = done
        return data
    
    @torch.jit.script_method
    def _train(self, 
               data: Dict[str, torch.Tensor]
               ):
        
        self.discriminator.train()
        pbar = tqdm(range(self.n_discriminator_iter))
        BS = data["observation"].size(0) #batch size
        for _ in pbar:
            z_real, z_x_mean, z_x_sigma, c_posterior, w_x_mean, w_x_sigma, gmm_dist, z_wc_mean_prior, z_wc_logvar_prior, x_reconstructed = self.variational_autoencoder(data["observation"])
            z_fake = gmm_dist.sample()
            critic_real = self.discriminator(z_real).reshape(-1)
            critic_fake = self.discriminator(z_fake).reshape(-1)
            gp = gradient_penalty(self.discriminator, z_real, z_fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + self._params.LAMBDA_GP * gp
            )
            self.discriminator_optim.zero_grad()
            loss_critic.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self._params.MAX_GRAD_NORM, norm_type=2)
            self.discriminator_optim.step()
            
        gen_fake  = self.discriminator(z_fake).reshape(-1)

        self.transition_model.train()
        self.optimizer.zero_grad()
        w_x, w_x_mean, w_x_sigma, z_next, z_next_mean, z_next_sigma, c_posterior = self.encoder(data["next_observation"])
        ###
        #Prepare & normalize the input/output data for the transition model
        inputs, outputs = preprocess_transition_data(z_real, action, z_next, data["done"])
        train_dataset   = TensorDataset(inputs, outputs)
        data_loader     = DataLoader(train_dataset, batch_size=BS , shuffle=False)
        normalizer_input, normalizer_output = compute_normalizer(data_loader)
        self.transition_model.normalizer_input  = normalizer_input
        self.transition_model.normalizer_output = normalizer_output
        u, y = next(iter(data_loader))
        transition_loss, transition_disc_loss, hidden, real_embed, fake_embed = self.transition_model(u, y)
        
        transition_gradient_penalty = self.transition_model.wgan_gp_reg(real_embed, fake_embed)
        #reward prediction and computing discount factor 
        likes, losses = {}, {}
        for name, head in self.heads.items():
            grad_head = name in self.grad_heads
            inp = inputs if grad_head else inputs.detach()
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = dist.log_prob(data[key])
                likes[key] = like
                losses[key] = -like.mean()
        model_loss = sum(
            self.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        print(f"sum of reward and discount losses: {model_loss}")
       
        metrics, z_posterior, z_posterior_mean, z_posterior_sigma, c_posterior, w_posterior_mean, w_posterior_sigma, dist, z_prior_mean, z_prior_logvar, X_reconst = self.variational_autoencoder.get_ELBO(data["observation"])
        metrics["wasserstein_gp_loss"]   = -torch.mean(gen_fake)
        metrics["total_observe_loss"]    = metrics["loss"] + metrics["wasserstein_gp_loss"]
        ###??? extra loss
        #reward and discount losses
        for k, v in losses.items():
            metrics.update(k + "_loss" : v.detach().cpu())

        metrics["transition_total_loss"] = transition_loss + transition_disc_loss + transition_gradient_penalty

        metrics["total_model_loss"] = metrics["total_observe_loss"]+ metrics["transition_total_loss"]+ model_loss
        metrics["total_model_loss"].backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.parameters, self._params.MAX_GRAD_NORM, norm_type=2)
        self.optimizer.step()
        
        self.writer.add_scalar('model_loss', {'observation loss': metrics["total_observe_loss"].item(),
                                              'transition loss': metrics["transition_total_loss"].item(),
                                              'reward loss': metrics["reward_loss"].item(),
                                              'discount loss': metrics["discount_loss"].item(),
                                              'total loss': metrics["total_model_loss"].item(),
                                              })

        return hidden, real_embed, z_real, z_next, metrics
    
    @torch.jit.script_method
    def imagine(self, 
                agent: Agent, 
                start_state: Dict[str, Tensor], #a combination of previous states & actions 
                horizon: int,
                done: torch.Tensor = None, 
                ) -> Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        tqdm.write("Collect data from imagination:")
        #inspired https://github.com/sai-prasanna/mpc_and_rl/blob/1fedaec5eec7a2a5844935d17b4fd0ad385d9d6c/dreamerv2_torch/dreamerv2_torch/world_model.py#L86
        
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:])) #torch.Size([b,c,h,w])->torch.Size([b*c,h,w])
        start = {k: flatten(v) for k, v in start_state.items()}
        start["action"] = torch.zeros_like(agent._p_fn(start["feat"])[1])
        seq = {k: [v] for k, v in start.items()}
        time_step = self._env.current_time_step()
        seq["observation"] = self.decoder(start["feat"])
        time_step._replace(observation=seq["observation"])
        for i in tqdm(range(horizon)):
            s1=seq["feat"]
            if i==0:
                action = agent._p_fn(seq["feat"])[1]
                seq["action"] = action
            features = torch.cat([seq["feat"], seq["action"]], dim=1)
            state, sample_mu, sample_sigma, hidden = self.transition_model.generate(features, seq_len = features.shape[-1])
            s2= state[:,:,-1]
            seq["feat"] = torch.cat([seq["feature"], state[:,:,-1]], dim=-1)
            #generate observation by decoder
            seq["observation"] = torch.cat([ seq["observation"], self.decoder(state[:,:,-1])], dim=-1)
            next_time_step = self._env.step(action.unsqueeze(0).detach().cpu())
            next_action = self._policy(seq["feat"][:,:,-1])[1]
            #s1, s2, reward, discount, a1, a2, done
            #inspired by https://github.com/Arseni1919/PL_TEMPLATE_PROJECT/blob/dd5d5fa2284c9ea1da35e316a14299fc89272669/alg_datamodule.py
            experience = Transition(s1, s2, torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32), action, next_action, torch.zeros(1, dtype=torch.bool))
            self.dataset.append(experience)
            if not time_step.is_last():
     
                transition = get_transition(time_step, next_time_step,
                                               action, next_action)
                self._data.add_transitions(transition)
            else:
                break
            time_step = next_time_step
            action = next_action
            seq["action"] = torch.cat([seq["action"], action], dim=-1) 

        seq_feat = torch.cat([seq["feat"], seq["action"]], dim=1)
        
        disc = self.heads["discount"](seq_feat).mean()
        seq["reward"] = self.heads["reward"](seq_feat).mode()
        if done is not None:
            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            true_first = 1.0 - flatten(done).to(disc.dtype)
            true_first *= self._discount
            disc = torch.concat([true_first[None], disc[1:]], 0)
        else:
            done = torch.zeros_like(disc, dtype=torch.bool)
            done[disc <= self._discount**DC.MUJOCO_ENVS_LENNGTH[self._env_name]] = True
        seq["discount"] = disc.unsqueeze(-1)   
        seq["weight"]   = torch.cumprod( torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0).unsqueeze(-1)
        seq["done"]     = done
        for i in range(len(self.dataset)):
            self.dataset[i] = self.dataset[i]._replace(discount = seq["discount"][i])
            self.dataset[i] = self.dataset[i]._replace(done = seq["done"][i])
        return seq
    
    def video_pred(self, observation, action):
        
        truth = observation[:6] + 0.5
        inputs = observation[:5]
        embed, embed_mean, embed_sigma, _, _, _, _, _, _, recon = self.variational_autoencoder(inputs)

        next_state, sample_mu, sample_sigma, state = self.transition_model.generate(torch.cat([embed, action], dim=1))
        openl = self.decoder(next_state)
        model = torch.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.permute(1, 2, 0, 3, 4).reshape(T, H, B * W, C).cpu()

    def save(self):
        self.save_path=os.path.join(self.ckpt_path, 'WorldModel.pth')
        torch.save(
            {'transition_model' : self.transition_model.state_dict(),
             'reward_model': self.reward_model.state_dict(),
             'observation_model': self.variational_autoencoder.state_dict(),
             'observation_discriminator_model': self.discriminator.state_dict(),
             'discount_model': self.discount_model.state_dict(),
             'encoder_model': self.encoder.state_dict(),
             'decoder_model': self.decoder.state_dict(),
             'discriminator_optimizer': self.discriminator_optim.state_dict(),
             'world_model_optimizer': self.optimizer.state_dict(),}, self.save_path)

    def load_checkpoints(self):

        if os.path.isfile(self.save_path):
            
            model_dicts = torch.load(self.save_path, map_location=self.device)
            self.transition_model.load_state_dict(model_dicts['transition_model'])
            self.variational_autoencoder.load_state_dict(model_dicts['observation_model'])
            self.discriminator.load_state_dict(model_dicts['observation_discriminator_model'])
            self.reward_model.load_state_dict(model_dicts['reward_model'])
            self.discount_model.load_state_dict(model_dicts['discount_model'])
            self.encoder.load_state_dict(model_dicts['encoder_model'])
            self.decoder.load_state_dict(model_dicts['decoder_model'])
            self.optimizer.load_state_dict(model_dicts['world_model_optimizer'])  
            self.discriminator_optim.load_state_dict(model_dicts['discriminator_optimizer'])
            print("Loading models checkpoints!")
        else:
            print("Checkpoints not found!")

class D2EAlgorithm(nn.Module):
    #https://github.com/sai-prasanna/dreamerv2_torch/tree/main/dreamerv2_torch/agent.py
    def __init__(self, 
                 hyperParams, 
                 sequence_length, 
                 env_name = 'Hopper-v2', 
                 device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 obs_space, 
                 act_space, 
                 step: Counter, *,
                 log_dir,
                 model_params=(((200, 200),), 2, 1),
                 optimizers=(( 0.0001, 0.5, 0.99),),
                 batch_size=50,
                 update_freq=1,
                 update_rate=0.005,
                 discount=0.99,
                 replay_buffer_size=int(1e6),
                 eval_state_mean=False,
                 imag_horizon= 16,
                 precision=32,
                 **kwarg):
        super().__init__()
        self.parameter = namedtuple('x', hyperParams.keys())(*hyperParams.values())
        self._env      = env_name
        self._discount = discount

        self.obs_space = obs_space
        self.act_space = act_space
        
        self.step      = step
        self.device    = device
        self.precision = precision
        self.eval_state_mean  = eval_state_mean
        self._sequence_length = sequence_length
        self.register_buffer("tfstep", torch.ones(()) * int(self.step))
        self.wm = WorldModel(hyperParams, sequence_length= self._sequence_length, env_name= self._env, device=self.device, **kwarg)
        self.RewardNorm = StreamNorm(momentum=0.99, scale=1.0, eps=1e-8)
        self.imag_horizon = imag_horizon
        self._use_amp = True if  self.precision==16 else False

        train_summary_dir = os.path.join(log_dir, 'train')
        self.train_summary_writer = SummaryWriter( train_summary_dir)
        # Construct agent.
        # Initialize dataset.
        
        agent_flags = utils.Flags(
                                  observation_spec= obs_space,
                                  action_spec     = act_space,
                                  model_params    = model_params,
                                  optimizers      = optimizers,
                                  batch_size      = batch_size,
                                  weight_decays   = (self.parameter.weight_decay,),
                                  update_freq     = update_freq,
                                  update_rate     = update_rate,
                                  discount        = discount,
                                  done            = self.parameter.PREDICT_DONE,
                                  env_name        = env_name,
                                  train_data      = self.wm.dataset
                                  )
        agent_args = Config(agent_flags).agent_args
        self._task_behavior = D2EAgent(**vars(agent_args))
        self._task_behavior_params = itertools.chain(
            self._task_behavior._get_q_source_vars(),
            self._task_behavior._get_p_vars(),
            self._task_behavior._get_c_vars(),
            self._task_behavior._get_v_source_vars(),
            self._task_behavior._a_vars,
            self._task_behavior._ae_vars,
            self._task_behavior._transit_discriminator.parameters()
        )

    def policy(self, observation, next_observation, reward, done, state=None, mode="train"):
        obs = self.wm.preprocess(observation, next_observation, reward, done)
        self.tfstep.copy_(torch.tensor([int(self.step)])[0])

        embed      = self.wm.encoder(obs["observation"])
        next_embed = self.wm.encoder(obs["next_observation"])
        sample = (mode == "train") or not self.eval_state_mean
        
        latent, _, _, _ = self.wm.transition_model.generate(
                                                           torch.cat([embed, action], dim=1), 
                                                           seq_len = embed.shape[-1]
                                                           )
        policy_state = latent.copy()
        if mode == "eval":
            a_tanh_mode, action, log_pi_a = self._task_behavior._p_fn(policy_state)
            noise = self.parameter.eval_noise
        elif mode in ["explore", "train"]:
            action = self._expl_behavior._p_fn(policy_state)[1]
            noise  = self.parameter.expl_noise
        action = action_noise(action, noise, self.act_space)
        outputs = {"action": action}
        state = (latent, next_embed)
        return outputs, state

    def _train(self, data, state=None):
        """
        List of operations happen here in this module 
        1)train the world model
        2)imagine state, action, reward, and discount from the world model
        3)normaize reward
        4)update reward and put them as an appropriate format to train D2E agent
        5) train the policy
        6) we should minimize the difference between the reward and discount (maybe observation) here too???
        """
        
        metrics = {}
        _, _, z_real, z_next, mets = self.wm._train(data, state)

        metrics.update(mets)
        start["feat"] = z_real
        with RequiresGrad(self._task_behavior_params):
            with torch.cuda.amp.autocast(self._use_amp):
                seq = self.wm.imagine( 
                                      self._task_behavior, 
                                      start, 
                                      self.imag_horizon,
                                      data["done"], 
                                      )
                
                rewards, mets1 = self.RewardNorm(seq["reward"])
                #update reward in the dataset before training the agent 
                for i in range(len(self.wm.dataset)):
                    self.wm.dataset[i] = self.wm.dataset[i]._replace(reward = rewards[i])
                mets1 = {f"reward_{k}": v for k, v in mets1.items()}
                #train the policy 
                self._task_behavior.train_step()
                step = self._task_behavior.global_step
                #get all the loss values from policy network 
                #here: https://github.com/neuronphysics/AIME/blob/1077b972d2ce85882cf54558370b21d83f687ce5/planner_D2E_regularizer.py#L1329
                mets2 = self._task_behavior._all_train_info
                metrics.update({"expl_" + key: value for key, value in mets2.items()})
                #Number (6) task
                losses = {}
                for name, head in self.wm.heads.items():
                     out = head(seq[name])
                     dists = out if isinstance(out, dict) else {name: out}
                     for key, dist in dists.items():
                         like = dist.log_prob(getattr(self.wm._data, key))
                         likes[key] = like
                         losses[key] = -like.mean()
                metrics.update(**mets1)
                metrics.update(f"{name}_loss": value.detach().cpu() for name, value in losses.items())
        return state, metrics

    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads["decoder"].cnn_keys:
            name = key.replace("/", "_")
            report[f"openl_{name}"] = self.wm.video_pred(data, key)
        return report
