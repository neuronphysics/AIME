# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Divergences for BRAC agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.optim import Optimizer
import datetime
import re
import os

local_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Epsilon for avoiding numerical issues.
EPS = torch.tensor(1e-8, dtype=torch.float64, device=local_device,requires_grad=False)
# Epsilon for clipping actions.
CLIP_EPS = torch.tensor(1e-3 , dtype=torch.float64, device=local_device,requires_grad=False)

def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)


@gin.configurable
class gradient_penalty(object):
  def __init__(self, c, device):
    self.c = c
    self.device = device
    
  def forward(self, s, a_p, a_b, gamma=5.0):
    """Calculates interpolated gradient penalty."""
    batch_size = s.shape[0]
    alpha = torch.rand([batch_size], device=self.device)
    a_intpl = a_p.to(device=self.device) + alpha[:, None] * (a_b.to(device=self.device) - a_p.to(device=self.device))
    c_intpl = self.c(s.to(device=self.device), a_intpl)
    slope = torch.sqrt(EPS + torch.sum(c_intpl ** 2, axis=-1))
    grad_penalty = torch.mean(torch.max(slope - 1.0, torch.zeros_like(slope)) ** 2)
    return grad_penalty * gamma

class Divergence(object):
  """Basic interface for divergence."""
  def __init__(self, c, device):
    self.c = c
    self.gradient_penalty = gradient_penalty(self.c, device)

  def dual_estimate(self, s, a_p, a_b):
    raise NotImplementedError

  def dual_critic_loss(self, s, a_p, a_b):
    return (- torch.mean(self.dual_estimate(s, a_p, a_b))
            + self.gradient_penalty.forward(s, a_p, a_b))

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    raise NotImplementedError

class Flags(object):

  def __init__(self, **kwargs):
    for key, val in kwargs.items():
      setattr(self, key, val)

class FDivergence(Divergence):
  """Interface for f-divergence."""

  def dual_estimate(self, s, a_p, a_b):
    logits_p = self.c(s, a_p)
    logits_b = self.c(s, a_b)
    return self._dual_estimate_with_logits(logits_p, logits_b)

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    raise NotImplementedError

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    _, apn, apn_logp = p_fn.sample_n(s, n_samples)
    _, abn, abn_logb = b_fn.sample_n(s, n_samples)
    # Clip actions here to avoid numerical issues.
    apn_logb = b_fn.get_log_density(
        s, clip_by_eps(apn, action_spec, CLIP_EPS))
    abn_logp = p_fn.get_log_density(
        s, clip_by_eps(abn, action_spec, CLIP_EPS))
    return self._primal_estimate_with_densities(
        apn_logp, apn_logb, abn_logp, abn_logb)

  def _primal_estimate_with_densities(
      self, apn_logp, apn_logb, abn_logp, abn_logb):
    raise NotImplementedError



class KL(FDivergence):
  """KL divergence."""

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    return (- soft_relu(logits_b)
            + torch.log(soft_relu(logits_p) + EPS) + 1.0)

  def _primal_estimate_with_densities(
      self, apn_logp, apn_logb, abn_logp, abn_logb):
    return torch.mean(apn_logp - apn_logb, dim=0)


class W(FDivergence):
  """Wasserstein distance."""

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    return logits_p - logits_b


@gin.configurable
def laplacian_kernel(x1, x2, sigma=20.0):
  d12 = torch.sum(
      torch.abs(x1[None] - x2[:, None]), dim=-1)
  k12 = torch.exp(- d12 / sigma)
  return k12


@gin.configurable
def mmd(x1, x2, kernel, use_sqrt=False):
  k11 = torch.mean(kernel(x1, x1), dim=(0, 1))
  k12 = torch.mean(kernel(x1, x2), dim=(0, 1))
  k22 = torch.mean(kernel(x2, x2), dim=(0, 1))
  if use_sqrt:
    return torch.sqrt(k11 + k22 - 2 * k12 + EPS)
  else:
    return k11 + k22 - 2 * k12


class MMD(Divergence):
  """MMD."""

  def primal_estimate(
      self, s, p_fn, b_fn, n_samples,
      kernel=laplacian_kernel, action_spec=None):
    apn = p_fn.sample_n(s, n_samples)[1]
    abn = b_fn.sample_n(s, n_samples)[1]
    return mmd(apn, abn, kernel)


CLS_DICT = dict(
    kl=KL,
    w=W,
    mmd=MMD,
    )


def get_divergence(name, c, device):
  return CLS_DICT[name](c, device)

def soft_variables_update(source_variables, target_variables, tau=1.0):
    print(f'source_variables len: {len(source_variables)}')
    print(f'target_variables len: {len(target_variables)}')
    for (v_s, v_t) in zip(source_variables, target_variables):
        print(f'v_s: {v_s.shape}')
        print(f'v_t: {v_t.shape}')
        v_t = (1 - tau) * v_t + tau * v_s
    return v_t

def get_summary_str(step=None, info=None, prefix=''):
    summary_str = prefix
    if step is not None:
        summary_str += 'Step {}; '.format(step)
    for key, val in info.items():
        if isinstance(val, (int, np.int32, np.int64)):
            summary_str += '{} {}; '.format(key, val)
        elif isinstance(val, (float, np.float32, np.float64)):
            summary_str += '{} {:.4g}; '.format(key, val)
    return summary_str


def write_summary(writer, info, step):
    """For pytorch. Write summary to tensorboard."""
    for key, val in info.items():
        if isinstance(val,
                (int, float, np.int32, np.int64, np.float32, np.float64)):
            writer.add_scalar(key, val, step)

def clip_by_eps(x, spec, eps=0.0):
  return torch.clamp(
      x, min=spec.minimum + eps, max=spec.maximum - eps)

# TODO: add customized gradient
def clip_v2(x, low, high):
    """Clipping with modified gradient behavior."""
    value = torch.min(torch.max(x, low * torch.ones_like((x))), high * torch.ones_like(x))
    def grad(dy):
       if_y_pos = torch.gt(dy, 0.0).type(torch.float32)
       if_x_g_low = torch.gt(x, low).type(torch.float32)
       if_x_l_high = torch.le(x, high).type(torch.float32)
       return (if_y_pos * if_x_g_low +
             (1.0 - if_y_pos) * if_x_l_high) * dy
    return value, grad

def relu_v2(x):
  """Relu with modified gradient behavior."""
  value = torch.nn.ReLU()(x)
  def grad(dy):
     if_y_pos = torch.gt(dy, 0.0).type(torch.float32)
     if_x_pos = torch.gt(x, 0.0).type(torch.float32)
     return (if_y_pos * if_x_pos + (1.0 - if_y_pos)) * dy
  return value, grad

# class clip_v2(torch.autograd.Function):
#   @staticmethod
#   def forward(ctx, x):
#     ctx.save_for_backward(x)
#     return torch.min(torch.max(x, 0. * torch.ones_like((x))), 500. * torch.ones_like(x))
#   @staticmethod
#   def backward(ctx, grad_output):
#     x, = ctx.saved_tensors
#     grad_cpy = grad_output.clone()
#     if_y_pos = torch.gt(grad_cpy, 0.0).type(torch.float32)
#     if_x_g_low = torch.gt(x, 0.).type(torch.float32)
#     if_x_l_high = torch.le(x, 500.).type(torch.float32)
#     return (if_y_pos * if_x_g_low +
#             (1.0 - if_y_pos) * if_x_l_high) * grad_cpy
def soft_relu(x, device):
  """Compute log(1 + exp(x))."""
  # Note: log(sigmoid(x)) = x - soft_relu(x) = - soft_relu(-x).
  #       log(1 - sigmoid(x)) = - soft_relu(x)
  return torch.log(1.0 + torch.exp(-torch.abs(x))).to(device=device) + torch.max(x, torch.zeros_like(x)).to(device=device)


def local_global_loss(l_enc, g_enc, measure, mode):
        '''
        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        Returns:
            torch.Tensor: Loss.
        '''
        N, local_units, dim_x, dim_y = l_enc.size()
        l_enc = l_enc.view(N, local_units, -1)

        if mode == 'fd':
            loss = fenchel_dual_loss(l_enc, g_enc, measure=measure)
        elif mode == 'nce':
            loss = nce_loss(l_enc, g_enc)
        elif mode == 'dv':
            loss = donsker_varadhan_loss(l_enc, g_enc)
        else:
            raise NotImplementedError(mode)

        return loss

def get_negative_expectation(q_samples, measure, average=True):

    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        #
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
        #Eq = F.softplus(q_samples) #+ q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        q_samples = torch.clamp(q_samples,-1e6,9.5)

        #print("neg q samples ",q_samples.cpu().data.numpy())
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        assert 1==2

    if average:
        return Eq.mean()
    else:
        return Eq

def get_positive_expectation(p_samples, measure, average=True):

    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
        #Ep =  - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples

    elif measure == 'RKL':

        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        assert 1==2

    if average:
        return Ep.mean()
    else:
        return Ep

def fenchel_dual_loss(l, g, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.
    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.
    Args:
        l: Local feature map.
        g: Global features.
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    N, local_units, n_locs = l.size()
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, local_units)

    u = torch.mm(g, l.t())
    u = u.reshape(N, N, -1)
    mask = torch.eye(N).cuda()
    n_mask = 1 - mask

    E_pos = get_positive_expectation(u, measure, average=False).mean(2)
    E_neg = get_negative_expectation(u, measure, average=False).mean(2)
    E_pos = (E_pos * mask).sum() / mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos
    return loss


def multi_fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.
    Used for multiple globals.
    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    mask = torch.eye(N).cuda()
    n_mask = 1 - mask

    E_pos = get_positive_expectation(u, measure, average=False).mean(2).mean(2)
    E_neg = get_negative_expectation(u, measure, average=False).mean(2).mean(2)
    E_pos = (E_pos * mask).sum() / mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos
    return loss


def nce_loss(l, g):
    '''Computes the noise contrastive estimation-based loss.
    Args:
        l: Local feature map.
        g: Global features.
    Returns:
        torch.Tensor: Loss.
    '''
    N, local_units, n_locs = l.size()
    l_p = l.permute(0, 2, 1)
    u_p = torch.matmul(l_p, g.unsqueeze(dim=2))

    l_n = l_p.reshape(-1, local_units)
    u_n = torch.mm(g, l_n.t())
    u_n = u_n.reshape(N, N, n_locs)

    mask = torch.eye(N).unsqueeze(dim=2).cuda()
    n_mask = 1 - mask

    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, -1).unsqueeze(dim=1).expand(-1, n_locs, -1)

    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)
    loss = -pred_log[:, :, 0].mean()
    return loss


def multi_nce_loss(l, m):
    '''
    Used for multiple globals.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
    Returns:
        torch.Tensor: Loss.
    '''
    N, units, n_locals = l.size()
    _, _ , n_multis = m.size()

    l = l.view(N, units, n_locals)
    m = m.view(N, units, n_multis)
    l_p = l.permute(0, 2, 1)
    m_p = m.permute(0, 2, 1)
    u_p = torch.matmul(l_p, m).unsqueeze(2)

    l_n = l_p.reshape(-1, units)
    m_n = m_p.reshape(-1, units)
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    mask = torch.eye(N)[:, :, None, None].cuda()
    n_mask = 1 - mask

    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)
    loss = -pred_log[:, :, 0].mean()

    return loss


def donsker_varadhan_loss(l, g):
    '''
    Args:
        l: Local feature map.
        g: Global features.
    Returns:
        torch.Tensor: Loss.
    '''
    N, local_units, n_locs = l.size()
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, local_units)

    u = torch.mm(g, l.t())
    u = u.reshape(N, N, n_locs)

    mask = torch.eye(N).cuda()
    n_mask = (1 - mask)[:, :, None]

    E_pos = (u.mean(2) * mask).sum() / mask.sum()

    u -= 100 * (1 - n_mask)
    u_max = torch.max(u)
    E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    loss = E_neg - E_pos
    return loss


def multi_donsker_varadhan_loss(l, m):
    '''
    Used for multiple globals.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
    Returns:
        torch.Tensor: Loss.
    '''
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    mask = torch.eye(N).cuda()
    n_mask = 1 - mask

    E_pos = (u.mean(2) * mask).sum() / mask.sum()

    u -= 100 * (1 - n_mask)
    u_max = torch.max(u)
    E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    loss = E_neg - E_pos
    return loss

# Optimistic Mirror Adam - works GREAT


class OptMirrorAdam(Optimizer):
    """
    Implements Optimistic Mirror Descent on Adam algorithm.

        Built on official implementation of Adam by pytorch.
       See "Optimistic Mirror Descent in Saddle-Point Problems: Gointh the Extra (-Gradient) Mile"
       double blind review, paper: https://openreview.net/pdf?id=Bkg8jjC9KQ

    Standard Adam::

        It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(OptMirrorAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OptMirrorAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None

        # Do not allow training with out closure
        if closure is  None:
            raise ValueError("This algorithm requires a closure definition for the evaluation of the intermediate gradient")


        # Create a copy of the initial parameters
        param_groups_copy = self.param_groups.copy()

        # ############### First update of gradients ############################################
        # ######################################################################################
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # @@@@@@@@@@@@@@@ State initialization @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_1'] = torch.zeros_like(p.data)
                    state['exp_avg_2'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq_1'] = torch.zeros_like(p.data)
                    state['exp_avg_sq_2'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq_1'] = torch.zeros_like(p.data)
                        state['max_exp_avg_sq_2'] = torch.zeros_like(p.data)
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




                exp_avg1, exp_avg_sq1 = state['exp_avg_1'], state['exp_avg_sq_1']
                if amsgrad:
                    max_exp_avg_sq1 = state['max_exp_avg_sq_1']
                beta1, beta2 = group['betas']


                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # Step will be updated once
                state['step'] += 1
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg1.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq1.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # *****************************************************
                # Additional steps, to get bias corrected running means
                exp_avg1 = torch.div(exp_avg1, bias_correction1)
                exp_avg_sq1 = torch.div(exp_avg_sq1, bias_correction2)
                # *****************************************************

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq1, exp_avg_sq1, out=max_exp_avg_sq1)
                    # Use the max. for normalizing running avg. of gradient
                    denom1 = max_exp_avg_sq1.sqrt().add_(group['eps'])
                else:
                    denom1 = exp_avg_sq1.sqrt().add_(group['eps'])

                step_size1 = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size1, exp_avg1, denom1)



        # Perform additional backward step to calculate stochastic gradient - WATING STATE
        loss = closure()

        # Re run the optimization with the second averaged moments
        # ############### Second evaluation of gradients ###########################################
        # ######################################################################################
        for (group, group_copy) in zip(self.param_groups,param_groups_copy ):
            for (p, p_copy) in zip(group['params'],group_copy['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]



                exp_avg2, exp_avg_sq2 = state['exp_avg_2'], state['exp_avg_sq_2']
                if amsgrad:
                    max_exp_avg_sq2 = state['max_exp_avg_sq_2']
                beta1, beta2 = group['betas']


                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg2.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq2.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # *****************************************************
                # Additional steps, to get bias corrected running means
                exp_avg2 = torch.div(exp_avg2, bias_correction1)
                exp_avg_sq2 = torch.div(exp_avg_sq2, bias_correction2)
                # *****************************************************

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq2, exp_avg_sq2, out=max_exp_avg_sq2)
                    # Use the max. for normalizing running avg. of gradient
                    denom2 = max_exp_avg_sq2.sqrt().add_(group['eps'])
                else:
                    denom2 = exp_avg_sq2.sqrt().add_(group['eps'])

                step_size2 = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p_copy.data.addcdiv_(-step_size2, exp_avg2, denom2)
                p = p_copy




        return loss

class AdaBoundW(Optimizer):
    """Implements AdaBound algorithm with Decoupled Weight Decay (arxiv.org/abs/1711.05101)
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBoundW, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBoundW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size)

        return loss


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


def get_datetime():
    now = datetime.datetime.now().isoformat()
    now = re.sub(r'\D', '', now)[:-6]
    return now

def maybe_makedirs(log_dir):
  import os.path
  if not os.path.exists(log_dir):
     os.mkdir(log_dir)

def make_base_dir(list_of_dir):
    first = list_of_dir[0]
    if not os.path.exists(first):
      os.mkdir(first)
    for dir in list_of_dir[1:]:
        first = os.path.join(first, dir)
        if not os.path.exists(first):
            os.mkdir(first)

    return first

def check_for_nans_and_nones(arr):
    none_indices = []
    checker_fn = lambda iterable : math.isnan(iterable)
    if isinstance(arr, np.ndarray):
        # checker_fn = lambda iterable : np.isnan(iterable)
        if np.isnan(arr).any():
            print(f'We have {np.isnan(arr).sum()} nones in this np array')
        return
    elif isinstance(arr, torch.Tensor) or isinstance(arr, torch.nn.parameter.Parameter):
        # checker_fn = lambda iterable : torch.isnan(iterable).any()
        if torch.isnan(arr).any():
            print(f'We have {torch.isnan(arr).sum()} nones in this torch array')
        return

    for idx, item in enumerate(arr):
        if item is None or checker_fn(item):
            print("We have a None")
            none_indices.append(idx)

    return none_indices
