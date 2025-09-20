from __future__ import annotations
import random
from copy import deepcopy
from typing import Iterable, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from math import inf
import math
import datetime
from abc import ABC, abstractmethod
import json
import tempfile
import jsbeautifier
import time
import logging
import sys
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Type, Union, Sequence, List, Any
import easydict as edict
import os
import queue
import threading
from torch.optim.lr_scheduler import LRScheduler
from dataclasses import is_dataclass, asdict
CLOSURE = Optional[Callable[[], float]]
LOSS = Optional[float]
BETAS = Union[Tuple[float, float], Tuple[float, float, float], Tuple[None, float]]
DEFAULTS = Dict
GROUP = Dict
PARAMETERS = Optional[Union[Iterable[GROUP], Iterable[torch.Tensor]]]
STATE = Dict
OPTIMIZER = Type[Optimizer]
OPTIMIZER_INSTANCE_OR_CLASS = Union[OPTIMIZER, Optimizer]
SCHEDULER = Type[LRScheduler]

HUTCHINSON_G = Literal['gaussian', 'rademacher']
CLASS_MODE = Literal['binary', 'multiclass', 'multilabel']

DATA_FORMAT = Literal['channels_first', 'channels_last']


class NoSparseGradientError(Exception):
    """Raised when the gradient is sparse gradient.

    :param optimizer_name: str. optimizer name.
    :param note: str. special conditions to note (default '').
    """

    def __init__(self, optimizer_name: str, note: str = ''):
        self.note: str = ' ' if not note else f' w/ {note} '
        self.message: str = f'{optimizer_name}{self.note}does not support sparse gradient.'
        super().__init__(self.message)


class ZeroParameterSizeError(Exception):
    """Raised when the parameter size is 0."""

    def __init__(self):
        self.message: str = 'parameter size is 0'
        super().__init__(self.message)


class NoClosureError(Exception):
    """Raised when there's no closure function."""

    def __init__(self, optimizer_name: str, note: str = ''):
        self.message: str = f'{optimizer_name} requires closure.{note}'
        super().__init__(self.message)


class NegativeLRError(Exception):
    """Raised when learning rate is negative."""

    def __init__(self, lr: float, lr_type: str = ''):
        self.note: str = lr_type if lr_type else 'learning rate'
        self.message: str = f'{self.note} must be positive. ({lr} > 0)'
        super().__init__(self.message)


class NegativeStepError(Exception):
    """Raised when step is negative."""

    def __init__(self, num_steps: int, step_type: str = ''):
        self.note: str = step_type if step_type else 'step'
        self.message: str = f'{self.note} must be positive. ({num_steps} > 0)'
        super().__init__(self.message)


class NoComplexParameterError(Exception):
    """Raised when the dtype of the parameter is complex.

    :param optimizer_name: str. optimizer name.
    :param note: str. special conditions to note (default '').
    """

    def __init__(self, optimizer_name: str, note: str = ''):
        self.note: str = ' ' if not note else f' w/ {note} '
        self.message: str = f'{optimizer_name}{self.note}does not support complex parameter.'
        super().__init__(self.message)

class BaseOptimizer(ABC, Optimizer):
    r"""Base optimizer class. Provides common functionalities for the optimizers."""

    def __init__(self, params: PARAMETERS, defaults: DEFAULTS) -> None:
        super().__init__(params, defaults)

    @staticmethod
    def load_optimizer(optimizer: OPTIMIZER_INSTANCE_OR_CLASS, **kwargs) -> Optimizer:
        r"""Build torch.optim.Optimizer class."""
        if isinstance(optimizer, Optimizer):
            return optimizer

        if 'params' in kwargs:
            params = kwargs.pop('params')
            return optimizer(params, **kwargs)

        raise ValueError('need to pass `params` when you pass the `torch.optim.Optimizer` instance.')

    @staticmethod
    @torch.no_grad()
    def set_hessian(param_groups: PARAMETERS, state: STATE, hessian: List[torch.Tensor]) -> None:
        r"""Set hessian to state from external source. Generally useful when using functorch as a base.

        Example:
        -------
            Here's an example::

                # Hutchinson's Estimator using HVP
                noise = tree_map(lambda v: torch.randn_like(v), params)
                loss_, hvp_est = jvp(grad(run_model_fn), (params,), (noise,))
                hessian_diag_est  = tree_map(lambda a, b: a * b, hvp_est, noise)

                optimizer.set_hessian(hessian_diag_est)
                # OR
                optimizer.step(hessian=hessian_diag_est)

        :param param_groups: PARAMETERS. parameter groups.
        :param state: STATE. optimizer state.
        :param hessian: List[torch.Tensor]. sequence of hessian to set.
        """
        i: int = 0
        for group in param_groups:
            for p in group['params']:
                if p.size() != hessian[i].size():
                    raise ValueError(
                        f'the shape of parameter and hessian does not match. {p.size()} vs {hessian[i].size()}'
                    )

                state[p]['hessian'] = hessian[i]
                i += 1

    @staticmethod
    def zero_hessian(param_groups: PARAMETERS, state: STATE, pre_zero: bool = True) -> None:
        r"""Zero-out hessian.

        :param param_groups: PARAMETERS. parameter groups.
        :param state: STATE. optimizer state.
        :param pre_zero: bool. zero-out hessian before computing the hessian.
        """
        for group in param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None and not p.grad.is_sparse:
                    if 'hessian' not in state[p]:
                        state[p]['hessian'] = torch.zeros_like(p)
                    elif pre_zero:
                        state[p]['hessian'].zero_()

    @staticmethod
    @torch.no_grad()
    def compute_hutchinson_hessian(
        param_groups: PARAMETERS,
        state: STATE,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: HUTCHINSON_G = 'gaussian',
    ) -> None:
        r"""Hutchinson's approximate hessian, added to the state under key `hessian`.

        :param param_groups: PARAMETERS. parameter groups.
        :param state: STATE. optimizer state.
        :param num_samples: int. number of times to sample `z` for the approximation of the hessian trace.
        :param alpha: float. alpha.
        :param distribution: HUTCHINSON_G. type of distribution.
        """
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params: List[torch.Tensor] = [
            p
            for group in param_groups
            for p in group['params']
            if p.requires_grad and p.grad is not None and not p.grad.is_sparse
        ]
        if len(params) == 0:
            return

        grads = [p.grad for p in params]

        for i in range(num_samples):
            if distribution == 'rademacher':
                zs = [torch.randint_like(p, 0, 1) * 2.0 - 1.0 for p in params]
            else:
                zs = [torch.randn_like(p) for p in params]

            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, retain_graph=i < num_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                state[p]['hessian'].add_(h_z * z, alpha=alpha / num_samples)

    @staticmethod
    def apply_weight_decay(
        p: torch.Tensor,
        grad: Optional[torch.Tensor],
        lr: float,
        weight_decay: float,
        weight_decouple: bool,
        fixed_decay: bool,
        ratio: Optional[float] = None,
    ) -> None:
        r"""Apply weight decay.

        :param p: torch.Tensor. parameter.
        :param grad: torch.Tensor. gradient.
        :param lr: float. learning rate.
        :param weight_decay: float. weight decay (L2 penalty).
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
        :param fixed_decay: bool. fix weight decay.
        :param ratio: Optional[float]. scale weight decay.
        """
        if weight_decouple:
            p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr) * (ratio if ratio is not None else 1.0))
        elif weight_decay > 0.0 and grad is not None:
            grad.add_(p, alpha=weight_decay)

    @staticmethod
    def apply_ams_bound(
        ams_bound: bool,
        exp_avg_sq: torch.Tensor,
        max_exp_avg_sq: Optional[torch.Tensor],
        eps: float,
        exp_avg_sq_eps: float = 1e-15,
    ) -> torch.Tensor:
        r"""Apply AMSBound variant.

        :param ams_bound: bool. whether to apply AMSBound.
        :param exp_avg_sq: torch.Tensor. exp_avg_sq.
        :param max_exp_avg_sq: Optional[torch.Tensor]. max_exp_avg_sq.
        :param eps: float. epsilon.
        :param exp_avg_sq_eps: float. eps value for numerical stability for exp_avg_sq.
        """
        if ams_bound:
            if torch.is_complex(max_exp_avg_sq):
                max_exp_avg_sq = torch.view_as_real(max_exp_avg_sq)

            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            de_nom = max_exp_avg_sq.add(exp_avg_sq_eps)
        else:
            de_nom = exp_avg_sq.add(exp_avg_sq_eps)

        return de_nom.sqrt_().add_(eps)

    @staticmethod
    def debias(beta: float, step: int) -> float:
        r"""Adam-style debias correction. Returns `1.0 - beta ** step`.

        :param beta: float. beta.
        :param step: int. number of step.
        """
        return 1.0 - math.pow(beta, step)  # fmt: skip

    @staticmethod
    def debias_beta(beta: float, step: int) -> float:
        r"""Apply the Adam-style debias correction into beta.

        Simplified version of `\^{beta} = beta * (1.0 - beta ** (step - 1)) / (1.0 - beta ** step)`

        :param beta: float. beta.
        :param step: int. number of step.
        """
        beta_n: float = math.pow(beta, step)
        return (beta_n - beta) / (beta_n - 1.0)  # fmt: skip

    @staticmethod
    def apply_adam_debias(adam_debias: bool, step_size: float, bias_correction1: float) -> float:
        r"""Apply AdamD variant.

        :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
        :param step_size: float. step size.
        :param bias_correction1: float. bias_correction.
        """
        return step_size if adam_debias else step_size / bias_correction1

    @staticmethod
    def get_rectify_step_size(
        is_rectify: bool,
        step: int,
        lr: float,
        beta2: float,
        n_sma_threshold: int,
        degenerated_to_sgd: bool,
    ) -> Tuple[float, float]:
        r"""Get step size for rectify optimizer.

        :param is_rectify: bool. whether to apply rectify-variant.
        :param step: int. number of steps.
        :param lr: float. learning rate.
        :param beta2: float. beta2.
        :param n_sma_threshold: float. SMA threshold.
        :param degenerated_to_sgd: bool. degenerated to SGD.
        """
        step_size: float = lr
        n_sma: float = 0.0

        if is_rectify:
            n_sma_max: float = 2.0 / (1.0 - beta2) - 1.0
            beta2_t: float = beta2 ** step  # fmt: skip
            n_sma: float = n_sma_max - 2 * step * beta2_t / (1.0 - beta2_t)

            if n_sma >= n_sma_threshold:
                rt = math.sqrt(
                    (1.0 - beta2_t) * (n_sma - 4) / (n_sma_max - 4) * (n_sma - 2) / n_sma * n_sma_max / (n_sma_max - 2)
                )
            elif degenerated_to_sgd:
                rt = 1.0
            else:
                rt = -1.0

            step_size *= rt

        return step_size, n_sma

    @staticmethod
    def get_adanorm_gradient(
        grad: torch.Tensor, adanorm: bool, exp_grad_norm: Optional[torch.Tensor] = None, r: Optional[float] = 0.95
    ) -> torch.Tensor:
        r"""Get AdaNorm gradient.

        :param grad: torch.Tensor. gradient.
        :param adanorm: bool. whether to use the AdaNorm variant.
        :param exp_grad_norm: Optional[torch.Tensor]. exp_grad_norm.
        :param r: Optional[float]. EMA factor. between 0.9 ~ 0.99 is preferred.
        """
        if not adanorm or exp_grad_norm is None:
            return grad

        if r is None:
            r = 0.95

        grad_norm = torch.linalg.norm(grad)

        exp_grad_norm.mul(r).add_(grad_norm, alpha=1.0 - r)

        return grad.mul(exp_grad_norm).div_(grad_norm) if exp_grad_norm > grad_norm else grad

    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        r"""Get approximation of EMA of squared gradient."""
        r_factor: torch.Tensor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor: torch.Tensor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

    @staticmethod
    def apply_cautious(update: torch.Tensor, grad: torch.Tensor) -> None:
        r"""Apply the Cautious Optimizer feature.

        :param update: torch.Tensor. update. it'll be masked in in-place manner.
        :param grad: torch.Tensor. gradient.
        """
        mask = (update * grad > 0).to(grad.dtype)
        mask.mul_(mask.numel() / (mask.sum() + 1))
        update.mul_(mask)

    @staticmethod
    def get_stable_adamw_rms(grad: torch.Tensor, exp_avg_sq: torch.Tensor, eps: float = 1e-16) -> float:
        r"""Get StableAdamW RMS.

        :param grad: torch.Tensor. gradient.
        :param exp_avg_sq: torch.Tensor. exp_avg_sq.
        :param eps: float. epsilon.
        """
        return grad.pow(2).div_(exp_avg_sq.clip(min=eps)).mean().sqrt_().clip_(min=1.0).item()

    @staticmethod
    def validate_range(x: float, name: str, low: float, high: float, range_type: str = '[)') -> None:
        if range_type == '[)' and not low <= x < high:
            raise ValueError(f'{name} must be in the range [{low}, {high})')
        if range_type == '[]' and not low <= x <= high:
            raise ValueError(f'{name} must be in the range [{low}, {high}]')
        if range_type == '(]' and not low < x <= high:
            raise ValueError(f'{name} must be in the range ({low}, {high}]')
        if range_type == '()' and not low < x < high:
            raise ValueError(f'{name} must be in the range ({low}, {high})')

    @staticmethod
    def validate_non_negative(x: Optional[float], name: str) -> None:
        if x is not None and x < 0.0:
            raise ValueError(f'{name} must be non-negative')

    @staticmethod
    def validate_non_positive(x: Optional[float], name: str) -> None:
        if x is not None and x > 0.0:
            raise ValueError(f'{name} must be non-positive')

    @staticmethod
    def validate_positive(x: Union[float, int], name: str) -> None:
        if x <= 0:
            raise ValueError(f'{name} must be positive')

    @staticmethod
    def validate_boundary(constant: float, boundary: float, bound_type: str = 'upper') -> None:
        if bound_type == 'upper' and constant > boundary:
            raise ValueError(f'constant {constant} must be in a range of (-inf, {boundary}]')
        if bound_type == 'lower' and constant < boundary:
            raise ValueError(f'constant {constant} must be in a range of [{boundary}, inf)')

    @staticmethod
    def validate_step(step: int, step_type: str) -> None:
        if step < 1:
            raise NegativeStepError(step, step_type=step_type)

    @staticmethod
    def validate_options(x: str, name: str, options: List[str]) -> None:
        if x not in options:
            opts: str = ' or '.join([f"'{option}'" for option in options]).strip()
            raise ValueError(f'{name} {x} must be one of ({opts})')

    @staticmethod
    def validate_learning_rate(learning_rate: Optional[float]) -> None:
        if learning_rate is not None and learning_rate < 0.0:
            raise NegativeLRError(learning_rate)

    @staticmethod
    def validate_mod(x: int, y: int) -> None:
        if x % y != 0:
            raise ValueError(f'{x} must be divisible by {y}')

    def validate_betas(self, betas: BETAS, beta_range_type: str = '[)', beta3_range_type: str = '[]') -> None:
        if betas[0] is not None:
            self.validate_range(betas[0], 'beta1', 0.0, 1.0, range_type=beta_range_type)

        self.validate_range(betas[1], 'beta2', 0.0, 1.0, range_type=beta_range_type)

        if len(betas) < 3:
            return

        if betas[2] is not None:
            self.validate_range(betas[2], 'beta3', 0.0, 1.0, range_type=beta3_range_type)

    def validate_nus(self, nus: Union[float, Tuple[float, float]]) -> None:
        if isinstance(nus, float):
            self.validate_range(nus, 'nu', 0.0, 1.0, range_type='[]')
        else:
            self.validate_range(nus[0], 'nu1', 0.0, 1.0, range_type='[]')
            self.validate_range(nus[1], 'nu2', 0.0, 1.0, range_type='[]')

    @abstractmethod
    def init_group(self, group: GROUP, **kwargs) -> None:  # pragma: no cover
        r"""Initialize the group of the optimizer and return is_complex."""
        return

    @staticmethod
    def view_as_real(param, *state_and_grads) -> tuple:
        r"""View imaginary tensors as real tensors."""
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            state_and_grads = tuple(
                torch.view_as_real(s) if (s is not None and torch.is_complex(s)) else s if s is not None else None
                for s in state_and_grads
            )

        return param, *state_and_grads

    @staticmethod
    def maximize_gradient(grad: torch.Tensor, maximize: bool = False) -> None:
        r"""Maximize the objective with respect to the params, instead of minimizing."""
        if maximize:
            grad.neg_()

    def step(self, closure: CLOSURE = None) -> LOSS:  # pragma: no cover
        raise NotImplementedError

def flatten_grad(grads: List[torch.Tensor]) -> torch.Tensor:
    r"""Flatten the gradient."""
    return torch.cat([grad.flatten() for grad in grads])


def un_flatten_grad(grads: torch.Tensor, shapes: List[int]) -> List[torch.Tensor]:
    r"""Unflatten the gradient."""
    idx: int = 0
    un_flatten_grads: List[torch.Tensor] = []
    for shape in shapes:
        length = int(np.prod(shape))
        un_flatten_grads.append(grads[idx:idx + length].view(shape).clone())  # fmt: skip
        idx += length
    return un_flatten_grads


class PCGrad(BaseOptimizer):
    r"""Gradient Surgery for Multi-Task Learning.

    :param optimizer: Optimizer: optimizer instance.
    :param reduction: str. reduction method.
    """

    def __init__(self, optimizer: Optimizer, reduction: str = 'mean'):
        self.validate_options(reduction, 'reduction', ['mean', 'sum'])

        self.optimizer = optimizer
        self.reduction = reduction

    @torch.no_grad()
    def init_group(self):
        self.zero_grad()

    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        return self.optimizer.step()

    def set_grad(self, grads: List[torch.Tensor]) -> None:
        idx: int = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1

    def retrieve_grad(self) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
        r"""Get the gradient of the parameters of the network with specific objective."""
        grad, shape, has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p, device=p.device))
                    has_grad.append(torch.zeros_like(p, device=p.device))
                    continue

                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p, device=p.device))

        return grad, shape, has_grad

    def pack_grad(self, objectives: Iterable) -> Tuple[List[torch.Tensor], List[List[int]], List[torch.Tensor]]:
        r"""Pack the gradient of the parameters of the network for each objective.

        :param objectives: Iterable[nn.Module]. a list of objectives.
        :return: torch.Tensor. packed gradients.
        """
        grads, shapes, has_grads = [], [], []
        for objective in objectives:
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward(retain_graph=True)

            grad, shape, has_grad = self.retrieve_grad()

            grads.append(flatten_grad(grad))
            has_grads.append(flatten_grad(has_grad))
            shapes.append(shape)

        return grads, shapes, has_grads

    def project_conflicting(self, grads: List[torch.Tensor], has_grads: List[torch.Tensor]) -> torch.Tensor:
        r"""Project conflicting.

        :param grads: a list of the gradient of the parameters.
        :param has_grads: a list of mask represent whether the parameter has gradient.
        :return: torch.Tensor. merged gradients.
        """
        shared: torch.Tensor = torch.stack(has_grads).prod(0).bool()

        pc_grad: List[torch.Tensor] = deepcopy(grads)
        for i, g_i in enumerate(pc_grad):
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j: torch.Tensor = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    pc_grad[i] -= g_i_g_j * g_j / (g_j.norm() ** 2)

        merged_grad: torch.Tensor = torch.zeros_like(grads[0])

        shared_pc_gradients: torch.Tensor = torch.stack([g[shared] for g in pc_grad])
        if self.reduction == 'mean':
            merged_grad[shared] = shared_pc_gradients.mean(dim=0)
        else:
            merged_grad[shared] = shared_pc_gradients.sum(dim=0)

        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)

        return merged_grad

    def pc_backward(self, objectives: Iterable[nn.Module]) -> None:
        r"""Calculate the gradient of the parameters.

        :param objectives: Iterable[nn.Module]. a list of objectives.
        """
        grads, shapes, has_grads = self.pack_grad(objectives)

        pc_grad = self.project_conflicting(grads, has_grads)
        pc_grad = un_flatten_grad(pc_grad, shapes[0])

        self.set_grad(pc_grad)


# ----------------------------- Core math: Aligned-MTL ----------------------------- #

def semideepcopy(obj: Any) -> Any:
    """A version of deepcopy that preserves PyTorch parameters.

    Args:
        obj: The object to copy

    Returns:
        A deep copy of the object, but with PyTorch parameters preserved as references
    """
    if isinstance(obj, (torch.nn.Parameter, edict.EasyDict)):
        return obj
    elif isinstance(obj, dict):
        return {key: semideepcopy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [semideepcopy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(semideepcopy(item) for item in obj)
    else:
        return deepcopy(obj)


def build_from_config(config: Any, **kwargs) -> Any:
    """This function recursively builds objects provided by the config.

    Args:
        config (Any): A config dict for building objects or any built object.
        kwargs: keyword arguments only used for building objects from `config`.
    """
    if isinstance(config, edict.EasyDict):
        return config
    if isinstance(config, dict) and config.keys() == {'class', 'args'}:
        # Create a deep copy to avoid modifying input, but preserve parameters
        config_copy = semideepcopy(config)

        # merge args
        assert type(kwargs) == dict, f"{type(kwargs)=}"
        assert set(config_copy.keys()) & set(kwargs.keys()) == set(), f"{config_copy.keys()=}, {kwargs.keys()=}"
        config_copy['args'].update(kwargs)

        # build args
        for key in config_copy['args']:
            config_copy['args'][key] = build_from_config(config_copy['args'][key])

        # build self
        return config_copy['class'](**config_copy['args'])
    elif isinstance(config, dict):
        return {key: build_from_config(val) for key, val in config.items()}
    elif isinstance(config, list):
        return [build_from_config(item) for item in config]
    elif isinstance(config, tuple):
        return tuple(build_from_config(item) for item in config)
    else:
        return config


def apply_tensor_op(
    func: Callable[[torch.Tensor], torch.Tensor],
    inputs: Union[tuple, list, dict, torch.Tensor, float],
) -> Any:
    if type(inputs) == torch.Tensor:
        return func(inputs)
    elif type(inputs) == tuple:
        return tuple(apply_tensor_op(func=func, inputs=tuple_elem) for tuple_elem in inputs)
    elif type(inputs) == list:
        return list(apply_tensor_op(func=func, inputs=list_elem) for list_elem in inputs)
    elif type(inputs) == dict:
        return {key: apply_tensor_op(func=func, inputs=inputs[key]) for key in inputs.keys()}
    else:
        return inputs


def serialize_tensor(obj: Any) -> Any:
    """Serialize torch tensors to lists (backward compatibility)."""
    return apply_tensor_op(func=lambda x: x.detach().tolist(), inputs=obj)


def check_write_file(path: Any) -> str:
    assert type(path) == str, f"{type(path)=}"
    assert os.path.isdir(os.path.dirname(path)), f"{path=}"
    return path

class BaseLogger(ABC):
    """
    Base class for all loggers.
    """

    def __init__(self, filepath: Optional[str] = None) -> None:
        """
        Initialize the base logger.

        Args:
            filepath: Optional filepath to write logs to
        """
        self.filepath = check_write_file(filepath) if filepath is not None else None
        self.buffer = {}
        self._buffer_lock = threading.Lock()
        self._write_queue = queue.Queue()
        self._write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self._write_thread.start()

    def _write_worker(self) -> None:
        """Background thread to handle write operations."""
        while True:
            try:
                data = self._write_queue.get()
                self._process_write(data)
                self._write_queue.task_done()
            except Exception as e:
                print(f"Write worker error: {e}")

    @abstractmethod
    def _process_write(self, data: Tuple[str, Any]) -> None:
        """Process a write operation."""
        pass

    def update_buffer(self, data: Dict[str, Any]) -> None:
        """
        Update the buffer with new data.

        Args:
            data: Dictionary of data to update the buffer with
        """
        self._write_queue.put(("UPDATE_BUFFER", data))

    def flush(self, prefix: Optional[str] = "") -> None:
        """
        Flush the buffer to the output.

        Args:
            prefix: Optional prefix to display before the data
        """
        self._write_queue.put(("FLUSH", prefix))

    def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: The message to log
        """
        self._write_queue.put(("INFO", message))

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: The message to log
        """
        self._write_queue.put(("WARNING", message))

    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: The message to log
        """
        self._write_queue.put(("ERROR", message))

    def page_break(self) -> None:
        """Add a page break to the log."""
        self._write_queue.put(("PAGE_BREAK", None))

class TextLogger(BaseLogger):
    """
    A text-based logger that writes to both a file and the console.
    """

    formatter = logging.Formatter(
        fmt=f"[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # A more complete version of formatter
    # formatter = logging.Formatter(
    #     fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    # ====================================================================================================
    # init methods
    # ====================================================================================================

    def __init__(self, filepath: Optional[str] = None) -> None:
        super(TextLogger, self).__init__(filepath=filepath)
        self._init_core_logger_()
        if not self.core_logger.handlers:
            self._init_file_handler_()
            self._init_stream_handler_()

    def _init_core_logger_(self) -> None:
        self.core_logger = logging.getLogger(name=self.filepath)
        self.core_logger.setLevel(level=logging.INFO)
        self.core_logger.propagate = False

    def _init_file_handler_(self) -> None:
        if self.filepath is None:
            return
        f_handler = logging.FileHandler(filename=self.filepath)
        f_handler.setFormatter(self.formatter)
        f_handler.setLevel(level=logging.INFO)
        self.core_logger.addHandler(f_handler)

    def _init_stream_handler_(self) -> None:
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(self.formatter)
        s_handler.setLevel(level=logging.INFO)
        self.core_logger.addHandler(s_handler)

    # ====================================================================================================
    # logging methods
    # ====================================================================================================

    def _process_write(self, data: Tuple[str, Any]) -> None:
        """Process a write operation."""
        msg_type, content = data
        if msg_type == "INFO":
            self.core_logger.info(content)
        elif msg_type == "WARNING":
            self.core_logger.warning(content)
        elif msg_type == "ERROR":
            self.core_logger.error(content)
        elif msg_type == "PAGE_BREAK":
            self.core_logger.info("")
            self.core_logger.info('=' * 100)
            self.core_logger.info('=' * 100)
            self.core_logger.info("")
        elif msg_type == "UPDATE_BUFFER":
            with self._buffer_lock:
                self.buffer.update(serialize_tensor(content))
        elif msg_type == "FLUSH":
            with self._buffer_lock:
                string = content + " " + ", ".join([f"{key}: {val}" for key, val in self.buffer.items()])
                self.core_logger.info(string)
                self.buffer = {}

    def train(self) -> None:
        """Switch to training mode (no-op for TextLogger)."""
        pass

    def eval(self) -> None:
        """Switch to evaluation mode (no-op for TextLogger)."""
        pass



class BaseOptim(ABC):

    optimizer: torch.optim.Optimizer

    def __init__(self) -> None:
        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.buffer: List[Any] = []

    @abstractmethod
    def backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Abstract method BaseOptimizer.backward not implemented.")

    # ====================================================================================================
    # ====================================================================================================

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimizer.step(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.optimizer.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self.optimizer.load_state_dict(*args, **kwargs)

    # ====================================================================================================
    # ====================================================================================================

    def summarize(self, output_path: Optional[str] = None) -> Any:
        r"""Default summarize method, assuming nothing has been logged to buffer.
        """
        assert len(self.buffer) == 0
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=self.buffer, filepath=output_path)
        return self.buffer
    

class MTLOptimizer(BaseOptim):
    __doc__ = r"""A hook contains custom operations for the optimizer.
    """

    def __init__(
        self,
        optimizer_config: dict,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        logger: Optional[TextLogger] = None,
        **kwargs,
    ):
        r"""
        Args:
            optimizer_config (dict): a config dict for building the core optimizer.
            losses (Dict[str, torch.Tensor]): a dummy loss dictionary for initialization purpose.
            shared_rep (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): a dummy shared representation for initialization purpose.
            logger (utils.logging.logger.TextLogger).
            kwargs (dict): other unused arguments. e.g., wrt_rep and per_layer for gradient balancing methods.
        """
        super(MTLOptimizer, self).__init__()
        self.optimizer = build_from_config(config=optimizer_config)
        self.logger = logger if logger is not None else TextLogger()
        self._init_shared_params_mask_(loss_dict=losses, shared_rep=shared_rep)
        self._init_shared_params_shapes_()
        self.num_tasks: int = len(losses)

    # ====================================================================================================
    # shared parameters related methods
    # ====================================================================================================

    def _init_shared_params_mask_(
        self,
        loss_dict: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        r"""This method initializes `self.shared_params_mask`, a 1D boolean tensor where 1 means the
        current parameters are shared. Order is defined by the double for loop
        "for group in self.optimizer.param_groups for p in group['params']".
        """
        # input checks
        assert type(loss_dict) == dict, f"{type(loss_dict)=}"
        assert all(type(key) == str for key in loss_dict.keys())
        assert all([
            type(value) == torch.Tensor and value.ndim == 0 and value.requires_grad
            for value in loss_dict.values()
        ])
        if type(shared_rep) == torch.Tensor:
            shared_rep = (shared_rep,)
        assert type(shared_rep) == tuple
        assert all(type(elem) == torch.Tensor for elem in shared_rep)
        shared_rep = torch.cat([g.flatten() for g in shared_rep])
        assert shared_rep.ndim == 1, f"{shared_rep.shape=}"
        # compute gradients with method 1
        self.optimizer.zero_grad(set_to_none=True)
        dummy_gradient = torch.zeros_like(shared_rep)
        shared_rep.backward(gradient=dummy_gradient, retain_graph=True)
        shared_params_mask_v1 = torch.tensor([
            p.requires_grad and p.grad is not None
            for group in self.optimizer.param_groups for p in group['params']
        ], dtype=torch.bool, device=shared_rep.device)
        # compute gradients with method 2
        masks: List[torch.Tensor] = []
        for loss in loss_dict.values():
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            masks.append(torch.tensor([
                p.requires_grad and p.grad is not None
                for group in self.optimizer.param_groups for p in group['params']
            ], dtype=torch.bool, device=torch.device('cuda')))
        shared_params_mask_v2 = torch.prod(torch.stack(masks).type(torch.int64), dim=0).type(torch.bool)
        # sanity check
        assert torch.equal(shared_params_mask_v1, shared_params_mask_v2)
        # assign to class attribute
        self.shared_params_mask: torch.Tensor = shared_params_mask_v1

    def _get_shared_params_(self):
        r"""Generator function for the shared parameters.
        """
        idx = 0
        for group in self.optimizer.param_groups:
            #if len(group['params']) == 1:
            #    continue
            for p in group['params']:
                if self.shared_params_mask[idx]:
                    yield p
                idx += 1
        assert idx == len(self.shared_params_mask), f"{idx=}, {len(self.shared_params_mask)=}"

    def _init_shared_params_shapes_(self) -> None:
        r"""This method initializes `self.shared_params_shapes`, a list of `torch.Size` for the shapes
        of the shared parameters.
        """
        shapes: List[torch.Size] = [p.shape for p in self._get_shared_params_()]
        # sanity check on mask and shapes
        assert hasattr(self, 'shared_params_mask') and type(self.shared_params_mask) == torch.Tensor
        shared_count = self.shared_params_mask.type(torch.int64).sum()
        assert shared_count == len(shapes), f"{shared_count=}, {len(shapes)=}"
        # assign to class attribute
        self.shared_params_shapes: List[torch.Size] = shapes

    # ====================================================================================================
    # gradient computation methods
    # ====================================================================================================

    def _get_grad_par_(self, loss: torch.Tensor, per_layer: bool) -> Union[torch.Tensor, List[torch.Tensor]]:
        r"""Get gradient of the given (single) loss w.r.t. shared parameters.

        Args:
            loss (torch.Tensor): the value of the (single) loss function at the current iteration.
            per_layer (bool): if True, gradient w.r.t. each parameter group will not be concatenated.
        Returns:
            grad (torch.Tensor or List[torch.Tensor]): the gradient of loss w.r.t. shared parameters.
        """
        # compute gradients
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grad: List[torch.Tensor] = [
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in self._get_shared_params_()
        ]
        assert len(grad) == len(self.shared_params_shapes)
        grad = [g.flatten() for g in grad]
        if not per_layer:
            grad = torch.cat(grad, dim=0)
            assert grad.ndim == 1, f"{grad.shape=}"
        return grad

    @staticmethod
    def _get_grad_rep_(
        loss: torch.Tensor,
        shared_rep: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        r"""Get gradient of the given (single) loss w.r.t. shared representation.

        Args:
            loss (torch.Tensor): the value of the (single) loss function at the current iteration.
        Returns:
            grad (torch.Tensor): the gradient of loss w.r.t. shared representation.
        """
        # initialization
        if type(shared_rep) == torch.Tensor:
            shared_rep = (shared_rep,)
        # compute gradients
        grad_seq: Tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=[loss], inputs=shared_rep, allow_unused=True, retain_graph=True,
        )
        assert type(grad_seq) == tuple, f"{type(grad_seq)=}"
        assert len(grad_seq) == len(shared_rep), f"{len(grad_seq)=}, {len(shared_rep)=}"
        grad_seq: List[torch.Tensor] = list(grad_seq)
        for idx in range(len(grad_seq)):
            if grad_seq[idx] is None:
                grad_seq[idx] = torch.zeros_like(shared_rep[idx])
            assert type(grad_seq[idx]) == torch.Tensor, f"{idx=}, {type(grad_seq[idx])=}"
            assert grad_seq[idx].shape == shared_rep[idx].shape, f"{grad_seq[idx].shape=}, {shared_rep[idx].shape=}"
        grad: torch.Tensor = torch.cat([g.flatten() for g in grad_seq], dim=0)
        assert grad.ndim == 1, f"{grad.shape=}"
        return grad

    def _get_grads_all_tasks_(
        self,
        loss_dict: Dict[str, torch.Tensor],
        shared_rep: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        wrt_rep: Optional[bool] = False,
        per_layer: Optional[bool] = False,
    ) -> Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        r"""This method computes the gradients for all losses.
        """
        # input checks
        assert type(loss_dict) == dict, f"{type(loss_dict)=}"
        assert all(type(key) == str for key in loss_dict.keys())
        assert all([
            type(value) == torch.Tensor and value.ndim == 0 and value.requires_grad
            for value in loss_dict.values()
        ])
        assert type(wrt_rep) == bool, f"{type(wrt_rep)=}"
        assert type(per_layer) == bool, f"{type(per_layer)=}"
        if wrt_rep:
            assert shared_rep is not None
            if type(shared_rep) != torch.Tensor:
                assert type(shared_rep) == tuple, f"{type(shared_rep)=}"
                assert all(type(elem) == torch.Tensor for elem in shared_rep)
        # initialize time
        grad_time = time.time()
        if wrt_rep:
            grads_dict = {
                name: self._get_grad_rep_(loss=loss_dict[name], shared_rep=shared_rep)
                for name in loss_dict
            }
        else:
            grads_dict = {
                name: self._get_grad_par_(loss=loss_dict[name], per_layer=per_layer)
                for name in loss_dict
            }
        self.logger.update_buffer({'grad_time': time.time() - grad_time})
        return grads_dict

    # ====================================================================================================
    # ====================================================================================================

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        r"""The default backward method.

        Args:
            shared_rep (Union[torch.Tensor, Tuple]): unused argument.
        """
        self.optimizer.zero_grad(set_to_none=True)
        losses_tensor = torch.stack(list(losses.values()))
        avg_loss = losses_tensor.mean()
        avg_loss.backward()

class GradientManipulationBaseOptimizer(MTLOptimizer, ABC):

    def __init__(
        self,
        wrt_rep: Optional[bool] = False,
        per_layer: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super(GradientManipulationBaseOptimizer, self).__init__(**kwargs)
        assert type(wrt_rep) == bool, f"{type(wrt_rep)=}"
        self.wrt_rep = wrt_rep
        assert type(per_layer) == bool, f"{type(per_layer)=}"
        self.per_layer = per_layer

    @abstractmethod
    def _gradient_manipulation_(
        self,
        grads_list: List[torch.Tensor],
        shared_rep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Each gradient manipulation method will implement its own.
        """
        raise NotImplementedError("_gradient_manipulation_ not implemented for abstract base class.")

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # input checks
        assert type(losses) == dict, f"{type(losses)=}"
        assert type(shared_rep) in [torch.Tensor, tuple], f"{type(shared_rep)=}"
        # initialization
        grads_dict: Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]] = self._get_grads_all_tasks_(
            loss_dict=losses, shared_rep=shared_rep, wrt_rep=self.wrt_rep, per_layer=self.per_layer,
        )
        if type(shared_rep) == tuple:
            shared_rep = torch.cat([g.flatten() for g in shared_rep])
        else:
            shared_rep = shared_rep.flatten()
        # compute gradients
        with torch.no_grad():
            if self.per_layer:
                num_layers = len(list(grads_dict.values())[0])
                assert all(len(grads_dict[name]) == num_layers for name in grads_dict)
                manipulated_grad: List[torch.Tensor] = [self._gradient_manipulation_(
                    grads_list=[grads_dict[name][idx] for name in grads_dict], shared_rep=shared_rep,
                ) for idx in range(num_layers)]
                manipulated_grad: torch.Tensor = torch.cat(manipulated_grad, dim=0)
            else:
                manipulated_grad: torch.Tensor = self._gradient_manipulation_(
                    grads_list=list(grads_dict.values()), shared_rep=shared_rep,
                )
        assert manipulated_grad.ndim == 1, f"{manipulated_grad.shape=}"
        # populate gradients for task-specific parameters
        self.optimizer.zero_grad(set_to_none=True)
        for p in self._get_shared_params_():
            assert p.requires_grad
            assert p.grad is None
            p.requires_grad = False
        multi_task_loss = list(losses.values())
        avg_loss = sum(multi_task_loss) / len(multi_task_loss)
        avg_loss.backward(retain_graph=self.wrt_rep)
        for p in self._get_shared_params_():
            assert not p.requires_grad
            assert p.grad is None
            p.requires_grad = True
        # populate gradients for shared parameters
        if self.wrt_rep:
            shared_rep.backward(gradient=manipulated_grad)
        else:
            self._set_grad_(grad=manipulated_grad)

    def _set_grad_(self, grad: torch.Tensor) -> None:
        r"""This function sets the gradients of the shared parameters in the model.

        Returns:
            None
        """
        # input checks
        assert type(grad) == torch.Tensor, f"{type(grad)=}"
        assert grad.ndim == 1, f"{grad.shape=}"
        # populate gradient
        idx = 0
        for p, shape in zip(self._get_shared_params_(), self.shared_params_shapes):
            length = int(torch.prod(torch.tensor(shape)))
            assert p.requires_grad
            assert p.grad is None
            p.grad = grad[idx:idx+length].view(shape)
            idx += length
        assert idx == len(grad), f"{idx=}, {grad.shape=}"

def get_gram_matrix(
    grad_list: List[torch.Tensor],
    other: List[torch.Tensor] = None
) -> torch.Tensor:
    r"""This function computes a matrix whose (i, j)-th entry is torch.dot(grad_list[i], other[j]).
    other is default to grad_list so the output would be the true Gram matrix of grad_list.
    """
    # input checks
    assert type(grad_list) == list
    for g in grad_list:
        assert type(g) == torch.Tensor, f"{type(g)=}"
        assert len(g.shape) == 1
    if other is not None:
        assert type(other) == list and len(other) == len(grad_list)
    # initialization
    num_tasks = len(grad_list)
    # compute result
    result = torch.zeros(size=(num_tasks, num_tasks), dtype=torch.float32, device=torch.device('cuda'))
    for i in range(num_tasks):
        loop = range(i, num_tasks) if other is None else range(num_tasks)
        for j in loop:
            if other is None:
                dot_prod = torch.dot(grad_list[i], grad_list[j])
                result[i, j] = dot_prod
                result[j, i] = dot_prod
            else:
                dot_prod = torch.dot(grad_list[i], other[j])
                result[i, j] = dot_prod
    return result


def serialize_tensor(obj: Any) -> Any:
    """Serialize torch tensors to lists (backward compatibility)."""
    return apply_tensor_op(func=lambda x: x.detach().tolist(), inputs=obj)


def apply_op(
    func: Callable[[Any], Any],
    inputs: Any,
) -> Any:
    """Apply a function recursively to nested data structures.
    
    Recursively applies func to all non-container elements in nested
    tuples, lists, and dictionaries. If the input is not a container
    (tuple, list, dict), applies func directly to it.
    
    Args:
        func: Function to apply to non-container elements
        inputs: Input data structure (can be nested)
        
    Returns:
        Data structure with same nesting as inputs, but with func applied
        to all non-container elements
    """
    if type(inputs) == tuple:
        return tuple(apply_op(func=func, inputs=tuple_elem) for tuple_elem in inputs)
    elif type(inputs) == list:
        return list(apply_op(func=func, inputs=list_elem) for list_elem in inputs)
    elif type(inputs) == dict:
        return {key: apply_op(func=func, inputs=inputs[key]) for key in inputs.keys()}
    else:
        return func(inputs)



def serialize_object(obj: Any) -> Any:
    """Serialize any nested object containing various data types to JSON-compatible format.
    
    Handles:
    - torch.Tensor -> list (via .detach().tolist())
    - numpy.ndarray -> list (via .tolist())
    - datetime -> ISO format string
    - dataclass -> dict (via asdict())
    - All other types -> unchanged (assumes JSON-serializable)
    
    Args:
        obj: Object to serialize (can be nested in tuples, lists, dicts)
        
    Returns:
        JSON-serializable version of the object
    """
    def _serialize_item(item: Any) -> Any:
        # Handle torch tensors
        if isinstance(item, torch.Tensor):
            return item.detach().tolist()
        
        # Handle numpy arrays
        elif isinstance(item, np.ndarray):
            return item.tolist()
        
        # Handle datetime objects
        elif isinstance(item, datetime):
            return item.isoformat()
        
        # Handle dataclass objects
        elif is_dataclass(item):
            # Recursively serialize the dict representation
            return serialize_object(asdict(item))
        
        # All other types pass through unchanged (assumes JSON-serializable)
        else:
            return item
    
    return apply_op(func=_serialize_item, inputs=obj)

def save_json(obj: Any, filepath: str) -> None:
    """Save object to JSON file using atomic writes and automatic serialization.
    
    Uses atomic writes (temp file + rename) to prevent race conditions between processes
    and threads. The rename operation is atomic at the filesystem level, ensuring readers 
    never see partially written files.
    
    Automatically handles dataclasses, torch.Tensor, numpy.ndarray, datetime,
    and all nested data structures without requiring manual conversion.
    
    Args:
        obj: Object to save (dataclasses are automatically converted)
        filepath: Path to save JSON file
        
    Raises:
        RuntimeError: If directory doesn't exist or write operation fails
    """
    try:
        # Auto-create directory if it doesn't exist
        target_dir = os.path.dirname(filepath)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        
        # Atomic write using temp file + rename
        temp_fd = None
        temp_filepath = None
        
        try:
            # Create temp file in same directory as target file
            # (rename is only atomic within the same filesystem)
            temp_fd, temp_filepath = tempfile.mkstemp(
                suffix='.tmp', 
                prefix='json_', 
                dir=target_dir or '.'
            )
            
            # Close the file descriptor - we'll use our own file operations
            os.close(temp_fd)
            temp_fd = None
            
            # Serialize and write to temporary file
            serialized_obj = serialize_object(obj)
            with open(temp_filepath, 'w') as f:
                f.write(jsbeautifier.beautify(
                    json.dumps(serialized_obj), 
                    jsbeautifier.default_options()
                ))
            
            # Atomic rename - this prevents race conditions
            os.rename(temp_filepath, filepath)
            temp_filepath = None  # Success - no cleanup needed
            
        except Exception as e:
            # Cleanup temp file if something went wrong
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except:
                    pass
            if temp_filepath is not None and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            raise
            
    except Exception as e:
        # Re-raise with filepath context for all errors
        raise RuntimeError(f"Error saving JSON to {filepath}: {e}") from e

class MultiPartOptimizer(BaseOptim):

    def __init__(self, optimizer_cfgs: dict) -> None:
        self.optimizers = {
            name: build_from_config(optimizer_cfgs[name])
            for name in optimizer_cfgs
        }
        super(MultiPartOptimizer, self).__init__()

    def reset_buffer(self) -> None:
        for name in self.optimizers:
            self.optimizers[name].reset_buffer()

    def backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("MultiPartOptimizer.backward is unused and should not be called.")

    # ====================================================================================================
    # ====================================================================================================

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: self.optimizers[name].state_dict()
            for name in self.optimizers
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        for name in self.optimizers:
            self.optimizers[name].load_state_dict(state_dict[name])

    # ====================================================================================================
    # ====================================================================================================

    def summarize(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        r"""Summarize each optimizer.
        """
        result = {
            name: self.optimizers[name].summarize(output_path=None)
            for name in self.optimizers
        }
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result



class ProcrustesSolver:
    @staticmethod
    def apply(grads, scale_mode="min"):
        assert (
            len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

        with torch.no_grad():
            cov_grad_matrix_e = grads.permute(0, 2, 1) @ grads
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)

            # singulars, basis = torch.symeig(cov_grad_matrix_e, eigenvectors=True) ***DEPRECATED***
            singulars, basis = torch.linalg.eigh(
                cov_grad_matrix_e
            )  # singular is eigenvalue landa and basis is eigenvector V

            tol = (
                torch.max(singulars)
                * max(cov_grad_matrix_e.shape[-2:])
                * torch.finfo().eps
            )

            rank = sum(singulars > tol)
            if rank == 0:
                rank = 1

            order = torch.argsort(singulars, dim=-1, descending=True)
            singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

            if scale_mode == "min":
                weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
            elif scale_mode == "median":
                weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
            elif scale_mode == "rmse":
                weights = basis * torch.sqrt(singulars.mean())

            weights = weights / torch.sqrt(singulars).view(1, -1)
            weights = weights @ basis.T
            grads = grads @ weights.unsqueeze(0)

            return grads, weights, singulars

class AlignedMTLOptimizer(GradientManipulationBaseOptimizer):
    __doc__ = r"""Paper: Independent Component Alignment for Multi-Task Learning (https://arxiv.org/pdf/2305.19000.pdf)
    """

    def _gradient_manipulation_(
        self,
        grads_list: List[torch.Tensor],
        shared_rep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            grads_list (List[torch.Tensor]): the list of 1D gradient tensors of each objective.
            shared_rep (torch.Tensor): unused argument.
        Returns:
            result (torch.Tensor): the 1D manipulated gradient tensor.
        """
        # input checks
        assert len(grads_list) == self.num_tasks, f"{len(grads_list)=}, {self.num_tasks=}"
        # compute Gram matrix
        gram_matrix = get_gram_matrix(grads_list)
        # compute eigenvalues and eigenvectors
        L, V = torch.linalg.eig(gram_matrix)
        assert L.dtype == V.dtype == torch.complex64, f"{L.dtype=}, {V.dtype=}"
        assert torch.all(L.imag == 0) and torch.all(V.imag == 0)
        L = L.real
        V = V.real
        eps = 1e-10
        L = torch.clamp(L, min=eps)
        # compute balance matrix
        sigma_min = L.min().sqrt()
        balance_matrix = sigma_min * torch.matmul(V, torch.matmul(torch.diag(1/L.sqrt()), V.t()))
        # compute final gradient
        alpha = balance_matrix.mean(dim=1)
        assert alpha.shape == (self.num_tasks,), f"{alpha.shape=}"
        result = sum([grads_list[i] * alpha[i] for i in range(self.num_tasks)])
        return result
    
# ---------------------------- Base Optimizer ----------------------------

class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass

class GradNorm(AbsWeighting):
    r"""Gradient Normalization (GradNorm).
    
    This method is proposed in `GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (ICML 2018) <http://proceedings.mlr.press/v80/chen18a/chen18a.pdf>`_ \
    and implemented by us.

    Args:
        alpha (float, default=1.5): The strength of the restoring force which pulls tasks back to a common training rate.

    """
    def __init__(self):
        super(GradNorm, self).__init__()
        
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([1.0]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        alpha = kwargs['alpha']
        if self.epoch >= 1:
            loss_scale = self.task_num * F.softmax(self.loss_scale, dim=-1)
            grads = self._get_grads(losses, mode='backward')
            if self.rep_grad:
                per_grads, grads = grads[0], grads[1]
                
            G_per_loss = torch.norm(loss_scale.unsqueeze(1)*grads, p=2, dim=-1)
            G = G_per_loss.mean(0)
            L_i = torch.Tensor([losses[tn].item()/self.train_loss_buffer[tn, 0] for tn in range(self.task_num)]).to(self.device)
            r_i = L_i/L_i.mean()
            constant_term = (G*(r_i**alpha)).detach()
            L_grad = (G_per_loss-constant_term).abs().sum(0)
            L_grad.backward()
            loss_weight = loss_scale.detach().clone()
            
            if self.rep_grad:
                self._backward_new_grads(loss_weight, per_grads=per_grads)
            else:
                self._backward_new_grads(loss_weight, grads=grads)
            return loss_weight.cpu().numpy()
        else:
            loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
            loss.backward()
            return np.ones(self.task_num)
        
        
# ---------------------------- weight clipping ----------------------------
class InitBounds:
    '''
    A class to calculate the initial bounds for weight clipping.
    Uniform Kaiming initialization bounds are used.
    Since bias requires knowledge of the previous layer's weights, we keep track of the previous weight tensor in this class.
    Linear: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L106
    Conv2d: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L144
    '''
    def __init__(self):
        self.previous_weight = None
        self.default_bound = 1.0
    def get(self, p):
        if p.dim() == 0:
            return float('inf')  # scalar parameter, no clipping
        if p.dim() == 1:
            if self.previous_weight is None:
                return self.default_bound  
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.previous_weight)
            return 1.0 / math.sqrt(fan_in)
        elif p.dim() >= 2: # Linear or ConvNd weights
            self.previous_weight = p
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(p)
            return  1.0 / math.sqrt(fan_in)
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(p.dim()))


class WeightClipping(torch.optim.Optimizer):
    def __init__(self, params, beta=1.0, optimizer=torch.optim.Adam, clip_last_layer=True, max_grad_norm=None, **kwargs):
        defaults = dict(beta=beta, clip_last_layer=clip_last_layer, max_grad_norm=max_grad_norm)
        super(WeightClipping, self).__init__(params, defaults)
        self.optimizer = optimizer(self.param_groups, **kwargs)
        self.max_grad_norm = max_grad_norm 
        self.param_groups = self.optimizer.param_groups
        self.defaults.update(self.optimizer.defaults)
        self.init_bounds = InitBounds()

    def step(self, closure=None):
        if self.max_grad_norm is not None:
            for group in self.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm)
        loss=self.optimizer.step(closure)
        self.weight_clipping()
        return loss

    def weight_clipping(self):
        clipped_sum, total_sum = 0.0, 0.0
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if i >= len(group["params"])-2 and not group["clip_last_layer"]:
                    # do not clip last layer of weights/biases
                    continue
                bound = self.init_bounds.get(p)
                if bound == float('inf'):
                    continue  # scalar parameter, no clipping
                clipped_sum += (p.data.abs() > group["beta"] * bound).float().sum()
                total_sum += p.data.numel()
                p.data.clamp_(-group["beta"] * bound, group["beta"] * bound)
        return (clipped_sum / total_sum).item()



# ------------------------------- Metric --------------------------------------

class EFDiagonalMetric:
    """
    EF-diagonal metric with a low-rank correction on the current task-gradient
    subspace (LoRA-flavored). Uses Woodbury to apply inverse in O(P·r + r^2).
    U is built on-the-fly from per-task grads J (no long-term memory).
    """
    def __init__(
        self, params, beta=0.95, damping=1e-3, power=1.0, bias_correction=True,
        tau=1.0, rank_cap=None, exclude_fn=None,
        store_device=None, store_dtype=None
    ):
        self.params = [p for p in params if p.requires_grad and (exclude_fn is None or not exclude_fn(p))]
        self.beta, self.damping, self.power = beta, damping, power
        self.bias_correction, self.tau, self.rank_cap = bias_correction, tau, rank_cap
        with torch.no_grad():
            self._sizes = [p.numel() for p in self.params]
            self._device = (self.params[0].device if self.params else torch.device('cpu'))
            self._dtype  = (self.params[0].dtype  if self.params else torch.float32)
            self.v = torch.zeros(sum(self._sizes), device=self._device, dtype=self._dtype)
        self.t = 0
        self._U = {}  # per-parameter {idx: U[n_i, r_i]}
        self._U_device = store_device or self._device
        self._U_dtype  = store_dtype or self._dtype

    def _flatten_grads(self):
        flats = []
        for p in self.params:
            g = p.grad
            flats.append((torch.zeros_like(p) if g is None else g).reshape(-1))
        return torch.cat(flats) if flats else torch.tensor([], device=self._device, dtype=self._dtype)

    @torch.no_grad()
    def update_from_grads(self):
        g = self._flatten_grads()
        if g.numel() == 0:
            return
        self.v.mul_(self.beta).addcmul_(g, g, value=(1.0 - self.beta))

    def _vhat(self):
        if not self.bias_correction or self.t == 0:
            return self.v
        bc = 1.0 - (self.beta ** self.t)
        return self.v / bc

    @torch.no_grad()
    def update_task_subspace(self, J: torch.Tensor):
        """
        J: [T, P] per-task Euclidean grads, already computed by MGDA.
        Build per-parameter U from columns of J (orthonormalized).
        """
        if J.numel() == 0:
            self._U.clear()
            return
        T, P = J.shape
        r_global = T if (self.rank_cap is None) else min(T, self.rank_cap)
        # bias-correction clock tick for consistent denom
        self.t += 1
        vhat = self._vhat()
        denom_full = (vhat + self.damping).clamp_min(1e-12)
        if self.power != 1.0:
            denom_full = denom_full.pow(self.power)

        self._U.clear()
        off = 0
        for idx, p in enumerate(self.params):
            n = p.numel()
            Ji = J[:, off:off+n].T.contiguous()  # [n, T]
            off += n
            if n == 0:
                continue
            r = min(r_global, n, T)
            if r == 0:
                continue
            Ji32 = Ji.to(dtype=torch.float32)
            colnorm = torch.linalg.norm(Ji32, dim=0)
            keep = (colnorm > 1e-12)
            if keep.sum() == 0:
                continue
            Ji32 = Ji32[:, keep]
            Q, _ = torch.linalg.qr(Ji32, mode='reduced')  # [n, <=T]
            if Q.shape[1] == 0:
                continue
            U = Q[:, :r].to(device=self._U_device, dtype=self._U_dtype)  # [n, r]
            self._U[idx] = U

    # Fast flat inverse for MGDA (handles all params in one go)
    def apply_inverse_flat(self, flat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vhat = self._vhat()
            denom_full = (vhat + self.damping).clamp_min(1e-12)
            if self.power != 1.0:
                denom_full = denom_full.pow(self.power)

        if flat.numel() == 0:
            return flat

        out = []
        off_full = 0
        off_d = 0
        for idx, p in enumerate(self.params):
            n = p.numel()
            v = flat[off_full:off_full+n]
            off_full += n
            d = denom_full[off_d:off_d+n]
            off_d += n

            Dinvv = v / d
            U = self._U.get(idx, None)
            if U is None or U.numel() == 0 or self.tau == 0.0:
                out.append(Dinvv)
                continue

            DinvU = U / d.reshape(-1, 1)                 # [n, r]
            UtDinvU = U.T @ DinvU                        # [r, r]
            UtDinvv = (U.T @ (v / d))                    # [r]
            K = (1.0 / self.tau) * torch.eye(UtDinvU.shape[0], device=UtDinvU.device, dtype=UtDinvU.dtype) + UtDinvU
            try:
                sol = torch.linalg.solve(K, UtDinvv.unsqueeze(-1)).squeeze(-1)  # [r]
            except RuntimeError:
                sol = torch.linalg.lstsq(K, UtDinvv.unsqueeze(-1)).solution.squeeze(-1)
            corr = DinvU @ sol
            out.append(Dinvv - corr)

        return torch.cat(out, dim=0)

# ----------------------------- Utilities -------------------------------------

def _gather_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _flatten(vecs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in vecs]) if vecs else torch.tensor([])


def _split_like(flat: torch.Tensor, like: Sequence[nn.Parameter]) -> List[torch.Tensor]:
    out, off = [], 0
    for p in like:
        n = p.numel()
        out.append(flat[off:off+n].view_as(p))
        off += n
    return out


def _egrad2rgrad(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    for p, g in zip(params, grads):
        if hasattr(p, 'manifold') and hasattr(p.manifold, 'egrad2rgrad'):
            out.append(p.manifold.egrad2rgrad(p.data, g))
        else:
            out.append(g)
    return out


def _retract(params: Sequence[nn.Parameter], step: Sequence[torch.Tensor], step_size: float):
    for p, d in zip(params, step):
        if hasattr(p, 'manifold') and hasattr(p.manifold, 'retr'):
            p.data = p.manifold.retr(p.data, -step_size * d)
        else:
            p.data.add_(-step_size * d)

# -------------------------- Implicit FW QP -----------------------------------
class _FWImplicit:
    """Frank–Wolfe for min_{α∈Δ} α^T M α using implicit Mα products.
    Mα is computed by: t = Σ_j α_j g_j ; tn = M^{-1} t ; (Mα)_i = g_i^T tn.
    Each FW iteration: O(T·P) time, O(P+T) memory.
    """
    def __init__(self, task_grads_flat: torch.Tensor, apply_minv: Callable[[torch.Tensor], torch.Tensor]):
        """
        task_grads_flat: Tensor of shape [T, P]
        apply_minv: function that maps flat vector → M^{-1} * vector
        """
        self.Gs = task_grads_flat  # [T, P]
        self.apply_minv = apply_minv
        self.T, self.P = task_grads_flat.shape

    def _M_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        # t = Σ α_j g_j
        t = alpha @ self.Gs  # [P]
        tn = self.apply_minv(t)  # [P]
        # (Mα)_i = g_i^T tn
        return (self.Gs @ tn)  # [T]

    def solve(self, alpha0: Optional[torch.Tensor] = None, max_iter: int = 50, tol: float = 1e-7) -> Tuple[torch.Tensor, Dict]:
        device = self.Gs.device
        if alpha0 is None:
            alpha = torch.full((self.T,), 1.0 / self.T, device=device)
        else:
            alpha = (alpha0 / (alpha0.sum() + 1e-12)).clamp_min(0)
        info = {'converged': False, 'iterations': 0, 'final_gap': None}
        for k in range(max_iter):
            Malpha = self._M_alpha(alpha)   # [T]
            grad = 2.0 * Malpha            # ∇f(α) with f=α^T M α
            kmin = int(torch.argmin(grad).item())
            s = torch.zeros_like(alpha); s[kmin] = 1.0
            d = s - alpha
            # Dual gap ≈ -⟨grad, d⟩
            gap = -(grad @ d)
            if gap.abs().item() < tol:
                info.update({'converged': True, 'iterations': k+1, 'final_gap': gap.item()})
                break
            # Exact line-search for quadratic: γ* = clip( - d^T M α / (d^T M d), [0,1] )
            num = (d @ Malpha)
            Md = self._M_alpha(d)
            den = (d @ Md).clamp_min(1e-12)
            gamma = (-num / den).clamp(0.0, 1.0)
            alpha = (1 - gamma) * alpha + gamma * s
            info['iterations'] = k+1
            info['final_gap'] = gap.item()
        # sanitize
        alpha = alpha.clamp_min(0); alpha = alpha / (alpha.sum() + 1e-12)
        return alpha, info

# ----------------------------- Main Optimizer --------------------------------
# ======================================================================
# Module-level helper (SINGLE definition): Low-rank tangent subspace
# ======================================================================

class _LowRankTangentSubspace:
    """
    Maintain a per-parameter rank-r orthonormal tangent basis {B_p}.
    Enables MGDA in rD without ever materializing J [T×P].
    Memory ~ sum_p |p|·r + O(r^2), with r << min{T,P}.
    """
    def __init__(self, params: List[nn.Parameter], rank: int):
        self.params = [p for p in params if p.requires_grad]
        self.r = int(rank)
        self.B: List[torch.Tensor] = []
        for p in self.params:
            Bp = torch.randn(*p.shape, self.r if p.numel() >= self.r else p.numel(),
                 device=p.device, dtype=p.dtype)

            self.B.append(self._orth(Bp, p))

    @torch.no_grad()
    def _orth(self, Bp: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Build an orthonormal basis with up to min(P, r) columns and pad to r
        P = p.numel()
        r = self.r
        dim = min(P, r)
        # Orthonormalize only the feasible block
        Q, _ = torch.linalg.qr(Bp.reshape(P, dim), mode="reduced")  # [P, dim]
        if dim < r:
            Q_full = torch.zeros(P, r, device=Bp.device, dtype=Bp.dtype)
            Q_full[:, :dim] = Q
            return Q_full.reshape(*p.shape, r)
        return Q.reshape_as(Bp)

    @torch.no_grad()
    def oja_update(self, avg_rgrads_per_param: List[torch.Tensor], eta: float = 0.05):
        """
        Streaming Oja: B_p += eta * ḡ_p * <ḡ_p, B_p>, then re-orthonormalize.
        avg_rgrads_per_param aligned with self.params; each is a *tangent* grad.
        """
        for i, (p, Bp, gp) in enumerate(zip(self.params, self.B, avg_rgrads_per_param)):
            if gp is None:
                continue
            coef = (gp.unsqueeze(-1) * Bp).reshape(-1, self.r).sum(dim=0)  # [r] = <ḡ_p, B_{p,·}>
            Bp.add_(eta * gp.unsqueeze(-1) * coef.view(*([1] * gp.dim()), -1))
            self.B[i] = self._orth(Bp, p)


# ======================================================================
# Clean RiemannianMGDA (memory-efficient)
# ======================================================================

class RiemannianMGDA:
    """
    Memory-efficient Riemannian MGDA with EF-diag metric and optional low-rank tangent subspace.

    • Subspace mode (subspace_rank>0): no T×P buffers; projects tasks to rD on the fly.
    • EF-diagonal M^{-1} (cheap curvature), then Armijo along Retr (fast) or Exp (exact).
    • Full-J path (subspace_rank=0) retained for exactness/compatibility.
    • Armijo snapshot policy controls memory: 'full' (default), 'fp16', 'cpu', or 'none' (delta-tracking).
    """

    def __init__(
        self,
        model: nn.Module,
        base_optim: Optional[torch.optim.Optimizer] = None,
        *,
        metric: Optional[EFDiagonalMetric] = None,
        fw_max_iter: int = 50,
        fw_tol: float = 1e-7,
        armijo_enabled: bool = True,
        armijo_c: float = 1e-4,
        armijo_beta: float = 0.5,
        armijo_max_tries: int = 10,
        normalize_task_grads: bool = False,
        max_nat_grad_norm: Optional[float] = None,
        verbose: bool = False,
        subspace_rank: int = 0,
        subspace_oja_eta: float = 0.05,
        use_expmap: bool = False,
        armijo_snapshot_policy: str = "full",  # 'full' | 'fp16' | 'cpu' | 'none' (delta)
        use_minv_in_subspace: bool = False,    # whiten FW in rD by B^T M^{-1} B (Cholesky)
        rank_adapt_enabled: bool = False,
        rank_variance_threshold: float = 0.95,   # explained variance target
        rank_hysteresis: int = 2,                # min Δr to trigger a resize
        rank_cooldown: int = 20,                 # steps between resize attempts
        rank_min: int = 8,
        rank_max: int = 128,
        ewma_decay: float = 0.9
    ):
        self.model = model
        self.params: List[nn.Parameter] = _gather_params(model)
        self.base_optim = base_optim
        self.metric = metric if metric is not None else EFDiagonalMetric(self.params)

        self.fw_max_iter = fw_max_iter
        self.fw_tol = fw_tol

        self.armijo_enabled = armijo_enabled
        self.armijo_c = armijo_c
        self.armijo_beta = armijo_beta
        self.armijo_max_tries = armijo_max_tries

        self.normalize_task_grads = normalize_task_grads
        self.max_nat_grad_norm = max_nat_grad_norm
        self.verbose = verbose

        self._alpha_prev: Optional[torch.Tensor] = None

        # Subspace controls
        self.subspace_rank = int(subspace_rank)
        self.subspace_oja_eta = float(subspace_oja_eta)
        self.use_expmap = bool(use_expmap)
        self.armijo_snapshot_policy = str(armijo_snapshot_policy)
        self.use_minv_in_subspace = bool(use_minv_in_subspace)

        self.rank_adapt_enabled = bool(rank_adapt_enabled)
        self.rank_variance_threshold = float(rank_variance_threshold)
        self.rank_hysteresis = int(rank_hysteresis)
        self.rank_cooldown = int(rank_cooldown)
        self.rank_min = int(rank_min)
        self.rank_max = int(rank_max)
        self.ewma_decay = float(ewma_decay)

        self._step_idx = 0
        self._rank_cooldown_left = 0
        self._Cr = None
        self._subspace: Optional[_LowRankTangentSubspace] = (
            _LowRankTangentSubspace(self.params, self.subspace_rank) if self.subspace_rank > 0 else None
        )

        # internal state for delta-tracking Armijo
        self._armijo_gamma_acc: float = 0.0  # only used when snapshot_policy == 'none'

        # Telemetry for rank adaptation
        self._last_rank_stats = {}

    @torch.no_grad()
    def _rank_adapt_update_cov(self, tilde_G: torch.Tensor):
        """
        Update EWMA covariance in rD: C_r ← ρ C_r + (1-ρ) * cov(tilde_G).
        tilde_G: [T, r]
        """
        if not self.rank_adapt_enabled or self._subspace is None:
            return
        r = self._subspace.r
        if tilde_G.numel() == 0:
            return
        # Use zero-mean covariance in rD (cheap, stable)
        Z = tilde_G - tilde_G.mean(dim=0, keepdim=True)          # [T, r]
        C = (Z.T @ Z) / max(1, Z.shape[0])                       # [r, r]
        if self._Cr is None or self._Cr.shape != C.shape:
            self._Cr = C.clone()
        else:
            self._Cr.mul_(self.ewma_decay).add_((1.0 - self.ewma_decay) * C)

    @torch.no_grad()
    def _maybe_rank_adapt(self):
        """
        Decide whether to resize the subspace this step, using EWMA covariance _Cr.
        Applies hysteresis + cooldown. Resize happens at the END of step (affects next step).
        """
        if not self.rank_adapt_enabled or self._subspace is None or self._Cr is None:
            return
        if self._rank_cooldown_left > 0:
            self._rank_cooldown_left -= 1
            return

        # eigenvalues in descending order
        evals, evecs = torch.linalg.eigh(self._Cr)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx].clamp_min(0)
        evecs = evecs[:, idx]  # r×r

        total = evals.sum().clamp_min(1e-12)
        csum = torch.cumsum(evals, dim=0) / total
        new_r = int((csum < self.rank_variance_threshold).sum().item()) + 1
        new_r = int(max(self.rank_min, min(self.rank_max, new_r)))
        old_r = self._subspace.r

        if abs(new_r - old_r) < self.rank_hysteresis:
            return  # ignore tiny oscillations

        # Resize basis: rotate by evecs then crop/extend
        self._resize_subspace(new_r, evecs)
        # Reset EWMA to the new size
        self._Cr = None
        self._rank_cooldown_left = self.rank_cooldown
        self._last_rank_stats = {'old_r': old_r, 'new_r': new_r, 'trace': float(total), 'r_eff': int((csum < self.rank_variance_threshold).sum().item()+1), 'cooldown_left': int(self._rank_cooldown_left)}
        self._rank_cooldown_left = self.rank_cooldown

    @torch.no_grad()
    def _resize_subspace(self, new_r: int, evecs: torch.Tensor):
        """
        Rotate current basis B by evecs (align to principal dirs) and resize to new_r.
        evecs: [r_old, r_old] from _Cr eigendecomposition.
        """
        sub = self._subspace
        r_old = sub.r
        assert evecs.shape[0] == r_old
        # rotate existing basis: Bp ← Bp @ evecs
        for i, Bp in enumerate(sub.B):
            sub.B[i] = (Bp @ evecs).contiguous()  # [..., r_old]
        if new_r < r_old:
            # shrink: drop trailing columns, then re-orth for safety
            for i, Bp in enumerate(sub.B):
                sub.B[i] = self._orth_columns(Bp[..., :new_r])
        elif new_r > r_old:
            extra = new_r - r_old
            for i, Bp in enumerate(sub.B):
                sub.B[i] = self._extend_with_orth(Bp, extra)  # add orthonormal complement
        sub.r = new_r

    @torch.no_grad()
    def _orth_columns(self, Bp: torch.Tensor) -> torch.Tensor:
        # Orthonormalize last-dim columns; robust when last-dim > P
        P = Bp.numel() // Bp.shape[-1]
        r = Bp.shape[-1]
        dim = min(P, r)
        Q, _ = torch.linalg.qr(Bp[..., :dim].reshape(P, dim), mode="reduced")  # [P, dim]
        if dim < r:
            Q_full = torch.zeros(P, r, device=Bp.device, dtype=Bp.dtype)
            Q_full[:, :dim] = Q
            return Q_full.reshape_as(Bp)
        return Q.reshape_as(Bp)

    @torch.no_grad()
    def _extend_with_orth(self, Bp: torch.Tensor, k: int) -> torch.Tensor:
        """
        Extend basis with k columns, respecting the ambient limit P.
        When P < desired columns, pad remaining with zeros to keep shape consistent.
        """
        shape = list(Bp.shape); r = shape[-1]
        P = Bp.numel() // r
        # Existing orth block
        r_eff = min(P, r)
        Q, _ = torch.linalg.qr(Bp[..., :r_eff].reshape(P, r_eff), mode="reduced")  # [P, r_eff]
        # How many *new* orth dirs can the space actually hold?
        k_eff = max(min(k, P - r_eff), 0)
        Q_add = torch.zeros(P, 0, device=Bp.device, dtype=Bp.dtype)
        if k_eff > 0:
            R_add = torch.randn(P, k_eff, device=Bp.device, dtype=Bp.dtype)
            R_add = R_add - Q @ (Q.T @ R_add)                      # project out current span
            Q_add, _ = torch.linalg.qr(R_add, mode="reduced")      # [P, k_eff]
        # Assemble full (possibly zero-padded) block to reach r + k columns
        B_full = torch.zeros(P, r + k, device=Bp.device, dtype=Bp.dtype)
        B_full[:, :r_eff] = Q
        if k_eff > 0:
            B_full[:, r_eff:r_eff + k_eff] = Q_add
        return B_full.reshape(*shape[:-1], r + k)


    # -------------------- Metric inverse & grad assignment --------------------

    def _apply_minv_flat(self, flat: torch.Tensor) -> torch.Tensor:
        """
        Apply M^{-1} to a flattened vector (EF-diag or fallback metric).
        Optionally clip nat-grad norm.
        """
        if hasattr(self.metric, "apply_inverse_flat"):
            v = self.metric.apply_inverse_flat(flat)
        else:
            pieces, off = [], 0
            for p in self.params:
                n = p.numel()
                v_p = flat[off:off+n].view_as(p)
                off += n
                pieces.append(self.metric.apply_inverse(v_p))
            v = _flatten(pieces)

        if self.max_nat_grad_norm is not None and self.max_nat_grad_norm > 0:
            nrm = torch.norm(v)
            if float(nrm) > self.max_nat_grad_norm:
                v = v * (self.max_nat_grad_norm / (nrm + 1e-12))
        return v

    def _assign_grad(self, flat_nat: torch.Tensor):
        off = 0
        for p in self.params:
            n = p.numel()
            p.grad = flat_nat[off:off+n].view_as(p).clone()
            off += n

    # -------------------- rD Frank–Wolfe (quadratic) --------------------

    @staticmethod
    def _fw_mgda_rd(tilde_G: torch.Tensor, alpha0: Optional[torch.Tensor], iters: int = 10, tol: float = 1e-7):
        """
        Solve:  min_{alpha in Δ_T} || tilde_G^T alpha ||^2
        tilde_G: [T, r] (task grads projected into rD subspace).
        """
        device, T = tilde_G.device, tilde_G.shape[0]
        G = tilde_G
        gnorm2 = (G**2).sum(dim=1)

        if alpha0 is None or alpha0.numel() != T:
            i0 = torch.argmin(gnorm2)
            alpha = torch.zeros(T, device=device, dtype=G.dtype); alpha[i0] = 1.0
        else:
            alpha = alpha0.clone().to(device)

        v = alpha @ G  # current combination in rD
        for _ in range(iters):
            dot = G @ v          # [T], entries = <g_t, v>
            j = torch.argmin(dot)
            s = torch.zeros_like(alpha); s[j] = 1.0
            d = (s - alpha) @ G  # FW direction in rD
            denom = (d*d).sum()
            if float(denom) <= 1e-16:
                break
            numer = (v * d).sum()
            gamma = torch.clamp(numer / denom, 0.0, 1.0)
            if float(gamma) <= tol:
                break
            alpha = (1 - gamma) * alpha + gamma * s
            v = (1 - gamma) * v + gamma * (G[j])
        return alpha, {'converged': True, 'final_gap': float((G @ v).min())}

    # -------------------- Subspace mass (optional whitening) --------------------

    def _subspace_mass_minv(self) -> torch.Tensor:
        """
        Build B^T M^{-1} B (r×r), applying EF-diag inverse to each basis column.
        """
        assert self._subspace is not None
        r = self._subspace.r
        cols_nat = []
        for j in range(r):
            # collect j-th column per param, flatten and apply M^{-1}
            col_p = [(Bp[..., j]) for Bp in self._subspace.B]
            col_flat = _flatten(col_p)
            cols_nat.append(self._apply_minv_flat(col_flat))
        # Stack nat-space columns and form Gram
        Mnat = torch.stack(cols_nat, dim=1)     # [P, r]
        return Mnat.t().contiguous() @ Mnat     # [r, r]

    # -------------------- Streaming projection (no T×P buffers) --------------------

    def _project_tasks_streaming(self, task_losses: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
          tilde_G: [T, r] task projections onto current basis
          avg_rgrads: list (per-param) average tangent grads for Oja

        Per-task grads are computed once, projected immediately, and released.
        Peak memory: O(P + rT). No J [T×P] ever.
        """
        assert self._subspace is not None
        T, r = len(task_losses), self._subspace.r
        device = self.params[0].device
        dtype = self.params[0].dtype

        tilde_G = torch.zeros(T, r, device=device, dtype=dtype)
        sum_rgrads = [torch.zeros_like(p) for p in self._subspace.params]

        for t, Li in enumerate(task_losses):
            grads = torch.autograd.grad(Li, self.params, retain_graph=(t < T-1), create_graph=False, allow_unused=True)
            rgrads = _egrad2rgrad(self.params, [(g if g is not None else torch.zeros_like(p))
                                                for g, p in zip(grads, self.params)])

            if self.normalize_task_grads:
                denom = (torch.cat([g.reshape(-1) for g in rgrads]).norm() + 1e-12)
                if float(denom) > 0:
                    rgrads = [g / denom for g in rgrads]

            acc = torch.zeros(r, device=device, dtype=dtype)
            for Bp, gp in zip(self._subspace.B, rgrads):
                acc += (gp.unsqueeze(-1) * Bp).reshape(-1, r).sum(dim=0)
            tilde_G[t] = acc

            for i in range(len(sum_rgrads)):
                sum_rgrads[i].add_(rgrads[i])

            del grads, rgrads  # free ASAP

        avg_rgrads = [g / max(T, 1) for g in sum_rgrads]
        return tilde_G, avg_rgrads

    # -------------------- Full-J path (kept for exactness) --------------------

    def _collect_task_grads_flat(self, task_losses: Sequence[torch.Tensor]) -> torch.Tensor:
        flats = []
        T = len(task_losses)
        for i, Li in enumerate(task_losses):
            grads = torch.autograd.grad(Li, self.params, retain_graph=(i < T-1), create_graph=False, allow_unused=True)
            g = [(gi if gi is not None else torch.zeros_like(p)) for gi, p in zip(grads, self.params)]
            flats.append(_flatten(_egrad2rgrad(self.params, g)))
        return torch.stack(flats, dim=0)  # [T, P]

    # -------------------- Armijo snapshot helpers --------------------

    def _make_snapshot(self):
        pol = self.armijo_snapshot_policy.lower()
        if pol == "none":     # delta-tracking (approximate)
            return None
        if pol == "cpu":
            return [p.data.detach().cpu().clone() for p in self.params]
        if pol == "fp16":
            return [p.data.detach().to(dtype=torch.float16).clone() for p in self.params]
        # default full-precision GPU
        return [p.data.detach().clone() for p in self.params]

    def _restore_snapshot(self, saved):
        if saved is None:
            return
        for p, s in zip(self.params, saved):
            if s.device.type != p.data.device.type or s.dtype != p.data.dtype:
                p.data.copy_(s.to(device=p.data.device, dtype=p.data.dtype))
            else:
                p.data.copy_(s)

    # -------------------- Curve stepping (Retr or Exp) --------------------

    def _apply_curve_step(self, tangent_list: List[torch.Tensor], gamma: float):
        """
        Apply θ ← curve(θ, -γ d) for each param:
          - if use_expmap and manifold.expmap exists → Exp
          - else → Retraction
        """
        if self.use_expmap:
            for p, d in zip(self.params, tangent_list):
                if hasattr(p, 'manifold') and hasattr(p.manifold, 'expmap'):
                    p.data = p.manifold.expmap(p.data, -gamma * d)
                else:
                    p.data.add_(-gamma * d)  # Euclidean fallback
        else:
            _retract(self.params, tangent_list, gamma)

    # -------------------- One MGDA step --------------------

    def step(
        self,
        task_losses: Sequence[torch.Tensor],
        closure: Optional[Callable[[], Sequence[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        T = len(task_losses)
        losses_vec = torch.stack([li.detach() for li in task_losses])


        avg_rgrads = None  # for Oja update if subspace is active
# === A) Build MGDA system either in rD (subspace) or full P ===
        if self._subspace is not None:
            tilde_G, avg_rgrads = self._project_tasks_streaming(task_losses)



            # Update EWMA covariance for rank adaptation
            if self.rank_adapt_enabled:
                self._rank_adapt_update_cov(tilde_G)
# Optional: whiten subspace by B^T M^{-1} B for faithful rD geometry
            if self.use_minv_in_subspace:
                S = self._subspace_mass_minv()  # r×r
                R = torch.linalg.cholesky(S + 1e-8 * torch.eye(S.shape[0], device=S.device, dtype=S.dtype))
                RinvT = torch.cholesky_inverse(R).t()  # (R^T)^{-1}
                G_hat = tilde_G @ RinvT            # [T, r]
                alpha0 = self._alpha_prev if (self._alpha_prev is not None and self._alpha_prev.numel()==T) else None
                alpha, fw_info = self._fw_mgda_rd(G_hat, alpha0, iters=self.fw_max_iter, tol=self.fw_tol)
                self._alpha_prev = alpha.detach().clone()
                v_r = alpha @ G_hat
                v_r = torch.linalg.solve(R.t(), v_r)  # back to original subspace
            else:
                alpha0 = self._alpha_prev if (self._alpha_prev is not None and self._alpha_prev.numel()==T) else None
                alpha, fw_info = self._fw_mgda_rd(tilde_G, alpha0, iters=self.fw_max_iter, tol=self.fw_tol)
                self._alpha_prev = alpha.detach().clone()
                v_r = alpha @ tilde_G

            # Lift to full tangent direction and apply M^{-1}
            d_tangent = [(Bp * v_r.view(*([1]*(Bp.dim()-1)), -1)).sum(dim=-1) for Bp in self._subspace.B]
            g_euc = _flatten(d_tangent)
            g_nat = self._apply_minv_flat(g_euc)
            self._assign_grad(g_nat)
        else:
            # Exact, full-J branch (kept for parity)
            J = self._collect_task_grads_flat(task_losses)  # [T, P]
            if hasattr(self.metric, "update_task_subspace"):
                self.metric.update_task_subspace(J)
            fw = _FWImplicit(J, self._apply_minv_flat)
            alpha0 = self._alpha_prev if (self._alpha_prev is not None and self._alpha_prev.numel()==T) else None
            alpha, fw_info = fw.solve(alpha0, max_iter=self.fw_max_iter, tol=self.fw_tol)
            self._alpha_prev = alpha.detach().clone()
            g_euc = alpha @ J
            g_nat = self._apply_minv_flat(g_euc)
            self._assign_grad(g_nat)

        # === B) Armijo line search along Retr/Exp on f(θ)=Σ α_i ℓ_i(θ) ===
        logs: Dict[str, torch.Tensor] = {}
        if self.armijo_enabled and (closure is not None):
            f0 = float((alpha * losses_vec).sum())
            # directional derivative with natural direction; sign for -d step
            dirderiv = float(g_euc @ g_nat)  # <∇f, d>
            if dirderiv <= 0:  # already descent-violating; take tiny step
                gamma = 1e-3
                tangent = _egrad2rgrad(self.params, _split_like(g_nat, self.params))
                self._apply_curve_step(tangent, gamma)
            else:
                gamma = 1.0
                saved = self._make_snapshot()
                ok = False
                self._armijo_gamma_acc = 0.0  # for delta mode

                for _ in range(self.armijo_max_tries):
                    if saved is not None:
                        # restore baseline (exact) then try step
                        self._restore_snapshot(saved)
                        tangent = _egrad2rgrad(self.params, _split_like(g_nat, self.params))
                        self._apply_curve_step(tangent, gamma)
                    else:
                        # delta-tracking (approximate): apply delta from current accumulated gamma
                        delta = gamma - self._armijo_gamma_acc
                        tangent = _egrad2rgrad(self.params, _split_like(g_nat, self.params))
                        self._apply_curve_step(tangent, delta)
                        self._armijo_gamma_acc = gamma

                    new_losses = torch.stack([li.detach() for li in closure()])
                    f_new = float((alpha * new_losses).sum())
                    # Armijo: f(new) <= f0 - c*gamma*dirderiv
                    if f_new <= f0 - self.armijo_c * gamma * dirderiv:
                        losses_vec = new_losses
                        ok = True
                        break

                    # backtrack
                    if saved is None:
                        # delta mode: revert delta approximately
                        tangent = _egrad2rgrad(self.params, _split_like(g_nat, self.params))
                        self._apply_curve_step(tangent, - (gamma - self._armijo_gamma_acc))
                        # (armijo_gamma_acc already equals gamma; set after revert)
                        self._armijo_gamma_acc = 0.0
                    gamma *= self.armijo_beta

                if not ok:
                    # final tiny step from exact baseline
                    if saved is not None:
                        self._restore_snapshot(saved)
                    tangent = _egrad2rgrad(self.params, _split_like(g_nat, self.params))
                    self._apply_curve_step(tangent, 1e-3)

            logs['armijo_gamma'] = torch.tensor(gamma)
        else:
            # No line-search: base optimizer or unit retraction
            if self.base_optim is not None:
                self.base_optim.step()
            else:
                tangent = _egrad2rgrad(self.params, _split_like(g_nat, self.params))
                self._apply_curve_step(tangent, 1.0)

        # === C) Update EF-diag stats from current .grad ===
        if hasattr(self.metric, "update_from_grads"):
            self.metric.update_from_grads()

        # === D) Rank adaptation and Oja update (after parameter update) ===
        if self.rank_adapt_enabled:
            self._maybe_rank_adapt()
        if self._subspace is not None and avg_rgrads is not None:
            # Refresh Oja basis with averaged tangent grads (respects any new r)
            try:
                self._subspace.oja_update(avg_rgrads, eta=self.subspace_oja_eta)
            except Exception:
                pass

        # Logs

        # Attach rank telemetry if available
        if self._subspace is not None:
            logs['rank'] = torch.tensor(int(self._subspace.r))
        if getattr(self, '_last_rank_stats', None):
            for k, v in self._last_rank_stats.items():
                # Use tensors for numeric values for consistency
                if isinstance(v, (int, float)):
                    logs[f'rank_{k}'] = torch.tensor(v)
        logs.update({
            'alpha': alpha.detach().cpu(),
            'losses': losses_vec.detach().cpu(),
            'agg_grad_norm_euc': torch.norm(g_euc).detach().cpu(),
            'agg_grad_norm_nat': torch.norm(g_nat).detach().cpu(),
            'fw_converged': torch.tensor(1 if fw_info.get('converged', False) else 0),
            'fw_gap': torch.tensor(fw_info.get('final_gap', float('inf'))),
        })
        return logs
