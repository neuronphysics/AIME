from __future__ import annotations
import random
from copy import deepcopy
from typing import Iterable, List, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
from collections import defaultdict
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
        if hasattr(self, "grad_index"):
            pass
        else:
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
                grad[beg:end] = param.grad.view(-1).detach()
            count += 1
        return grad
    
    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        for p in self.get_share_params():
            if p.grad is not None:
                p.grad.zero_()
                
    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    if 'MoDo' in [base.__name__ for base in self.__class__.__bases__] or 'SDMGrad' in [base.__name__ for base in self.__class__.__bases__]:
                        losses[tn].backward(retain_graph=True)
                    else:
                        losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    params = list(self.get_share_params())
                    grad_list = torch.autograd.grad(
                        losses[tn], params,
                        retain_graph=True,
                        allow_unused=True,   # <— key: avoid errors when a task doesn’t touch a param
                    )
                    flat = []
                    for g, p in zip(grad_list, params):
                        if g is None:
                            flat.append(torch.zeros_like(p, device=p.device).view(-1))
                        else:
                            flat.append(g.contiguous().view(-1))
                    grads[tn] = torch.cat(flat)
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
                param.grad = new_grads[beg:end].contiguous().view_as(param).detach().clone()
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


class PCGrad(AbsWeighting):
    r"""Project Conflicting Gradients (PCGrad).
    
    This method is proposed in `Gradient Surgery for Multi-Task Learning (NeurIPS 2020) <https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>`_ \
    and implemented by us.

    .. warning::
            PCGrad is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self, device=torch.cuda.current_device()):
        super(PCGrad, self).__init__()
        self.device = device
        
    def backward(self, losses):
        batch_weight = np.ones(len(losses))
        if self.rep_grad:
            raise ValueError('No support method PCGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            random.shuffle(task_index)
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2)+1e-8)
                    batch_weight[tn_j] -= (g_ij/(grads[tn_j].norm().pow(2)+1e-8)).item()
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight
    

class MGDA(AbsWeighting):
    r"""Multiple Gradient Descent Algorithm (MGDA).
    
    This method is proposed in `Multi-Task Learning as Multi-Objective Optimization (NeurIPS 2018) <https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/isl-org/MultiObjectiveOptimization>`_. 

    Args:
        mgda_gn ({'none', 'l2', 'loss', 'loss+'}, default='none'): The type of gradient normalization.

    """
    def __init__(self):
        super(MGDA, self).__init__()
    
    def _find_min_norm_element(self, grads):

        def _min_norm_element_from2(v1v1, v1v2, v2v2):
            if v1v2 >= v1v1:
                gamma = 0.999
                cost = v1v1
                return gamma, cost
            if v1v2 >= v2v2:
                gamma = 0.001
                cost = v2v2
                return gamma, cost
            gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
            cost = v2v2 + gamma*(v1v2 - v2v2)
            return gamma, cost

        def _min_norm_2d(grad_mat):
            dmin = torch.tensor(1e8, dtype=grad_mat.dtype, device=grad_mat.device)  
            sol = None
            for i in range(grad_mat.size()[0]):
                for j in range(i+1, grad_mat.size()[0]):
                    c,d = _min_norm_element_from2(grad_mat[i,i], grad_mat[i,j], grad_mat[j,j])
                    if d < dmin:
                        dmin = d
                        sol = [(i,j),c,d]
            return sol

        def _projection2simplex(y):
            m = len(y)
            sorted_y = torch.sort(y, descending=True)[0]
            tmpsum = y.new_zeros(()).to(y.device)
            tmax_f = (torch.sum(y) - 1.0)/m
            for i in range(m-1):
                tmpsum+= sorted_y[i]
                tmax = (tmpsum - 1)/ (i+1.0)
                if tmax > sorted_y[i+1]:
                    tmax_f = tmax
                    break
            return torch.max(y - tmax_f, torch.zeros(m).to(y.device).to(y.dtype))

        def _next_point(cur_val, grad, n):
            proj_grad = grad - ( torch.sum(grad) / n )
            tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
            tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])

            skippers = torch.sum(tm1<1e-7) + torch.sum(tm2<1e-7)
            t = torch.ones(1).to(grad.device)
            if (tm1>1e-7).sum() > 0:
                t = torch.min(tm1[tm1>1e-7])
            if (tm2>1e-7).sum() > 0:
                t = torch.min(t, torch.min(tm2[tm2>1e-7]))

            next_point = proj_grad*t + cur_val
            next_point = _projection2simplex(next_point)
            return next_point

        MAX_ITER = 50
        STOP_CRIT = 1e-5
        grad_mat = grads.mm(grads.t())  # keep original Gram path
        init_sol = _min_norm_2d(grad_mat)
        
        n = grads.size()[0]
        sol_vec = torch.zeros(n).to(grads.device).to(grad_mat.dtype)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec
    
        iter_count = 0

        while iter_count < MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = _next_point(sol_vec, grad_dir, n)

            v1v1 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*sol_vec.unsqueeze(0).repeat(n, 1)*grad_mat)
            v1v2 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)
            v2v2 = torch.sum(new_point.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)
    
            nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < STOP_CRIT:
                return sol_vec
            sol_vec = new_sol_vec
            iter_count += 1
        return sol_vec
    
    def _gradient_normalizers(self, grads, loss_data, ntype):
        eps= torch.finfo(torch.float32).eps
        if ntype == 'l2':
            gn = grads.pow(2).sum(-1).sqrt()+eps
        elif ntype == 'loss':
            gn = loss_data+eps
        elif ntype == 'loss+':
            gn = (loss_data+eps) * (grads.pow(2).sum(-1).sqrt()+eps)
        elif ntype == 'none':
            gn = torch.ones_like(loss_data).to(self.device)
        else:
            raise ValueError('No support normalization type {} for MGDA'.format(ntype))
        grads = grads / gn.unsqueeze(1).repeat(1, grads.size()[1])
        return grads
    
    def backward(self, losses, **kwargs):
        mgda_gn = kwargs['mgda_gn']
        mode = kwargs.get('mode', 'backward')
        use_bf16 = kwargs.get('use_bf16', False)
        grads = self._get_grads(losses, mode=mode)
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]

        if use_bf16:
            grads = grads.to(torch.bfloat16)
            loss_data = torch.tensor([loss.item() for loss in losses], device=self.device, dtype=torch.float32)
            grads = self._gradient_normalizers(grads, loss_data, ntype=mgda_gn) # l2, loss, loss+, none
            grads = grads.contiguous()
        else:
            loss_data = torch.tensor([loss.item() for loss in losses], device=self.device, dtype=torch.float32)
            grads = self._gradient_normalizers(grads, loss_data, ntype=mgda_gn)
            grads = grads.contiguous().to(torch.float32)
        sol = self._find_min_norm_element(grads)
        sol = sol.to(grads.device).to(dtype=grads.dtype)
        if self.rep_grad:
            self._backward_new_grads(sol, per_grads=per_grads)
        else:
            self._backward_new_grads(sol, grads=grads)
        del grads, loss_data
        return sol.detach().cpu().numpy()
    
class DB_MTL(AbsWeighting):

    def __init__(self):
        super(DB_MTL, self).__init__()

    def init_param(self):
        self.step = 0
        self._compute_grad_dim()
        self.grad_buffer = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        
    def backward(self, losses, **kwargs):
        self.step += 1
        beta = kwargs['DB_beta']
        beta_sigma = kwargs['DB_beta_sigma']

        batch_weight = np.ones(len(losses))
        if self.rep_grad:
            raise ValueError('No support method DB_MTL with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            batch_grads = self._compute_grad(torch.log(losses+1e-8), mode='backward') # [task_num, grad_dim]

        self.grad_buffer = batch_grads + (beta/self.step**beta_sigma) * (self.grad_buffer - batch_grads)

        u_grad = self.grad_buffer.norm(dim=-1)

        alpha = u_grad.max() / (u_grad + 1e-8)
        new_grads = sum([alpha[i] * self.grad_buffer[i] for i in range(self.task_num)])

        self._reset_grad(new_grads)
        return batch_weight


# Neural Information Processing Systems (NeurIPS) 2018
# https://github.com/intel-isl/MultiObjectiveOptimization
class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |c x1 + (1-c) x2|_2^2
        d is the distance (objective) optimized
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        i.e., min_c || sum_i c_i x_i ||_2^2  s.t. sum_i c_i = 1,
        1 >= c_i >= 0 for all i, and exists i,j with c_i + c_j = 1.0
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        # match numpy reference: sum over components
                        dps[(i, j)] += torch.dot(
                            vecs[i][k].reshape(-1), vecs[j][k].reshape(-1)
                        ).item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(
                            vecs[i][k].reshape(-1), vecs[i][k].reshape(-1)
                        ).item()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):  # matches the numpy loop bound
                        dps[(j, j)] += torch.dot(
                            vecs[j][k].flatten(), vecs[j][k].flatten()
                        ).item()
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, solve argmin_z ||y - z||_2  s.t. sum_i z_i = 1 and 1 >= z_i >= 0
        """
        m = len(y)
        # sort descending to mirror np.flip(np.sort(y), axis=0)
        sorted_y, _ = torch.sort(y, descending=True)
        tmpsum = torch.tensor(0.0, device=y.device, dtype=y.dtype)
        tmax_f = (y.sum() - 1.0) / m
        for i in range(m - 1):
            tmpsum = tmpsum + sorted_y[i]
            tmax = (tmpsum - 1.0) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.maximum(y - tmax_f, torch.zeros_like(y))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (grad.sum() / n)
        # masks as in numpy slicing
        mask_neg = proj_grad < 0
        mask_pos = proj_grad > 0
        tm1 = -cur_val[mask_neg] / proj_grad[mask_neg]
        tm2 = (1.0 - cur_val[mask_pos]) / proj_grad[mask_pos]

        # keep skippers to mirror numpy code path (value unused)
        skippers = (tm1 < 1e-7).sum() + (tm2 < 1e-7).sum()

        t = torch.tensor(1.0, device=cur_val.device, dtype=cur_val.dtype)
        if (tm1 > 1e-7).any():
            t = torch.min(tm1[tm1 > 1e-7])
        if (tm2 > 1e-7).any():
            t = torch.min(tm2[tm2 > 1e-7])

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), find the minimum-norm element in the convex hull:
        min ||u||_2  s.t. u = sum_i c_i vecs[i], sum_i c_i = 1.
        """
        # establish device/dtype from inputs
        dev = vecs[0][0].device
        dtype = vecs[0][0].dtype

        # best 2-task solution
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = torch.zeros(n, device=dev, dtype=dtype)
        sol_vec[init_sol[0][0]] = torch.tensor(init_sol[1], device=dev, dtype=dtype)
        sol_vec[init_sol[0][1]] = torch.tensor(1.0 - init_sol[1], device=dev, dtype=dtype)

        if n < 3:
            # optimal for n=2
            return sol_vec, init_sol[2]

        iter_count = 0
        grad_mat = torch.zeros((n, n), device=dev, dtype=dtype)
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = torch.tensor(dps[(i, j)], device=dev, dtype=dtype)

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)

            # recompute inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    dij = dps[(i, j)]
                    v1v1 += float(sol_vec[i].item() * sol_vec[j].item() * dij)
                    v1v2 += float(sol_vec[i].item() * new_point[j].item() * dij)
                    v2v2 += float(new_point[i].item() * new_point[j].item() * dij)

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1.0 - nc) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < MinNormSolver.STOP_CRIT:
                # mirror numpy: return current sol_vec (not new_sol_vec) and nd
                return sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1

        return sol_vec, nd

    @staticmethod
    def find_min_norm_element_FW(vecs):
        """
        Frank–Wolfe variant: same objective as find_min_norm_element.
        """
        dev = vecs[0][0].device
        dtype = vecs[0][0].dtype

        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = torch.zeros(n, device=dev, dtype=dtype)
        sol_vec[init_sol[0][0]] = torch.tensor(init_sol[1], device=dev, dtype=dtype)
        sol_vec[init_sol[0][1]] = torch.tensor(1.0 - init_sol[1], device=dev, dtype=dtype)

        if n < 3:
            return sol_vec, init_sol[2]

        grad_mat = torch.zeros((n, n), device=dev, dtype=dtype)
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = torch.tensor(dps[(i, j)], device=dev, dtype=dtype)

        iter_count = 0
        while iter_count < MinNormSolver.MAX_ITER:
            # t_iter = argmin over columns of grad_mat @ sol_vec
            t_iter = torch.argmin(torch.matmul(grad_mat, sol_vec)).item()

            v1v1 = float(torch.dot(sol_vec, torch.matmul(grad_mat, sol_vec)).item())
            v1v2 = float(torch.dot(sol_vec, grad_mat[:, t_iter]).item())
            v2v2 = float(grad_mat[t_iter, t_iter].item())

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec.clone()
            new_sol_vec[t_iter] += (1.0 - nc)

            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1

        return sol_vec, nd


def gradient_normalizers(grads, losses, normalization_type):
    """
    grads: Dict[int, List[torch.Tensor]]  (each tensor is flattened per-param grad)
    losses: Sequence[torch.Tensor]        (task losses)
    returns: Dict[int, float]             (per-task normalizer)
    """
    eps = torch.finfo(torch.float32).eps

    def l2_norm(g_list):
        # sqrt(sum_i ||g_i||^2) over that task's parameter grads
        return torch.sqrt(sum(g.pow(2).sum() for g in g_list)) + eps

    gn = {}
    for t, g_list in grads.items():
        if normalization_type in ("norm", "l2"):
            val = l2_norm(g_list)
        elif normalization_type == "loss":
            val = losses[t].detach().abs() + eps
        elif normalization_type == "loss+":
            val = (losses[t].detach().abs() + eps) * l2_norm(g_list)
        elif normalization_type == "none":
            val = torch.tensor(1.0, device=(g_list[0].device if len(g_list) else losses[t].device))
        else:
            raise ValueError(f"Invalid Normalization Type: {normalization_type}")
        gn[t] = float(val.item())
    return gn


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm = 1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class LOG_MGDA(WeightMethod):
    """Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    """

    def __init__(
        self, n_tasks, device: torch.device, params="shared", normalization="none",
        max_norm=1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization
        self.max_norm = max_norm

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            dim=0,
        )

    def get_weighted_loss(
        self,
        losses,
        shared_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        last_shared_parameters :
        representation :
        kwargs :

        Returns
        -------

        """
        eps = torch.finfo(torch.float32).eps
        grads = {}
        params = dict(
            rep=representation, shared=shared_parameters, last=last_shared_parameters
        )[self.params]
        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                    allow_unused=True,  # <— key: avoid errors when a task doesn’t touch a param
                )
            )
            # Normalize all gradients, this is optional and not included in the paper.

            #grads[i] = [torch.flatten(grad) for grad in g]
            grads[i] = [(grad if grad is not None else torch.zeros_like(p)).flatten() for p, grad in zip(params, g)]


        gn = gradient_normalizers(grads, losses, self.normalization)
        for t in range(self.n_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        sol, min_norm = self.solver.find_min_norm_element(
            [grads[t] for t in range(len(grads))]
        )
        sol = sol * self.n_tasks  # make sure it sums to self.n_tasks
        
        weighted_loss = sum([losses[i]  * sol[i]  for i in range(len(sol))])
        return weighted_loss, {"weights": sol.detach().to(torch.float32)}

    
class MTLOptim:
    """
    Implements basic operations for customized MTL Optimizers.
    Note that while the optimizer's parameters can be a list of tensors, internally everything is flattened into a
    single vector (see _pack_grad and _flatten_grad).
    """
    def __init__(self, optimizer, scheduler=None):
        self._optim = optimizer
        self._sched = scheduler
        return

    @property
    def optimizer(self):
        return self._optim

    def do_store_norm_sum_grads(self):
        self.store_norm_sum_grads = True

    def compute_norm_sum_grads(self, objectives):
        # NOTE: involves an additional backward pass per task, but it's just for logging (done less frequently)
        # return l2 norm of sum of original grads on the shared parameters (requires additional backward)
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]
        grad, _, shared = self._pack_grad(objectives, retain_graph=True)
        norm_sum_grads = torch.sqrt(sum([g_i[shared] for g_i in grad]).pow(2).sum())
        self.zero_grad()
        return norm_sum_grads, shared

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''
        self._optim.step()
        if self._sched is not None:
            self._sched.step()

    def iterate(self, objectives, **kwargs):
        '''
        compute the new customized update direction, apply it, zero the gradients
        '''
        self.set_auxiliaries(**kwargs)
        self.custom_backwards(objectives)
        self.step()
        self.zero_grad()

    def set_auxiliaries(self, **kwargs):
        # Pass any auxiliary parameter.
        pass

    def custom_backwards(self, objectives):
        '''
        Calculate the gradient of the parameters with a MTL algorithm that computes a custom update direction.
        The direction is stored as gradient within the variables over which to optimize.

        input:
        - objectives: a list of objectives
        '''

        # If the loss hasn't been averaged for the mini-batch, do it now.
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]

        custom_grad, shapes, shared = self._pack_grad(objectives)
        custom_grad = self._get_update_direction(custom_grad, shared, objectives)
        custom_grad = self._unflatten_grad(custom_grad, shapes[0])
        self._set_grad(custom_grad)
        return

    def _get_update_direction(self, grads, shared, objectives, shapes=None):
        # algorithm-dependent method to compute the update direction for MTL
        raise NotImplementedError("children classes of MTLOptimizer need to implement _get_update_direction")

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives, retain_graph=False):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grads, shapes = [], []
        shared = None
        m = len(objectives)
        for idx, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=(idx < (m-1)) or retain_graph)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            # Infer which parameters are shared and which aren't
            if shared is None:
                shared = self._flatten_grad(has_grad, shape)
            else:
                shared = shared * self._flatten_grad(has_grad, shape)
            shapes.append(shape)
        self._optim.zero_grad(set_to_none=True)
        return grads, shapes, shared.bool()

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = int(np.prod(shape))
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


def standard_head_backward(objectives, specialized_params):
    # Perform standard backward pass (of unit scalarization) over the specialized parameters.
    # Overwrites any existing grad on these parameters.
    spec_grad = torch.autograd.grad(sum(objectives), specialized_params, only_inputs=True)
    for idx, sparam in enumerate(specialized_params):
        sparam.grad = spec_grad[idx]


def batch_average(objectives):
    # If the loss hasn't been averaged for the mini-batch, do it now.
    if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
        objectives = [obj.mean(dim=0) for obj in objectives]
    return objectives

class IMTL(MTLOptim):
    # IMTL: https://openreview.net/forum?id=IMPnRXEWpvr
    def __init__(self, optimizer, specialized_parameters=None, scheduler=None, ub=True, learn_scaling=True, numerical_eps=1e-6):
        # UB=True means that the IMTL-G part of the algorithm is applied on the gradient of the shared representation
        # (analogously to MGDA-UB)
        # learn_scaling=False disables IMTL-L
        super().__init__(optimizer, scheduler=scheduler)
        self.ub = ub
        self.shared_repr = None
        self.st = None
        self.learn_scaling = learn_scaling
        self.st_optimizer = None
        self.specialized_params = specialized_parameters
        self.numerical_eps = numerical_eps
        self._alpha_to_log = None

    def _get_update_direction(self, grads, shared, objectives, shapes=None, return_alpha=False):
        if not return_alpha:
            merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
            for idx in range(len(grads)):
                # Use plain gradients for task-specific parameters
                merged_grad[~shared] += grads[idx][~shared].clone()
            shared_grads = [g_i[shared] for g_i in grads]
        else:
            shared_grads = grads

        # Build matrices of gradient differences.
        D = (shared_grads[0] - shared_grads[1]).unsqueeze(-1)
        u_0 = shared_grads[0]/shared_grads[0].norm()
        U = (u_0 - shared_grads[1]/shared_grads[1].norm()).unsqueeze(-1)
        for g_i in shared_grads[2:]:
            D = torch.cat([D, (shared_grads[0] - g_i).unsqueeze(-1)], -1)
            U = torch.cat([U, (u_0 - g_i/g_i.norm()).unsqueeze(-1)], -1)

        # Compute minimal-norm affine combination coefficients.
        A = D.transpose(-2, -1) @ U
        A += self.numerical_eps * torch.eye(A.shape[-1], device=A.device)   # avoid singularity
        alpha = torch.inverse(A) @ torch.mv(U.transpose(-2, -1), shared_grads[0])
        alpha = torch.cat([1 - alpha.sum().unsqueeze(-1), alpha], 0)
        self._alpha_to_log = alpha
        if torch.isnan(alpha).any():
            # Some gradient was 0: the solution of the minimal-norm affine combination is 0.
            alpha = torch.zeros_like(alpha)
        if return_alpha:
            return alpha

        for idx in range(len(grads)):
            # Compute the affine combination for shared parameters
            merged_grad[shared] += shared_grads[idx].mul_(alpha[idx])
        return merged_grad

    def custom_backwards(self, objectives):
        objectives = batch_average(objectives)

        if self.learn_scaling:
            # IMTL-L part.
            # using the initial step size of the optimizer as step size for vanilla SGD: detail not specified in paper
            if self.st == None:
                self.st = torch.zeros((len(objectives),) + tuple(objectives[0].shape), requires_grad=True,
                                      device=objectives[0].device)
                self.st_optimizer = torch.optim.Adam([self.st], lr=self._optim.defaults['lr'])
            scaled_losses = [torch.exp(self.st[idx]) * objectives[idx] - self.st[idx] for idx in range(len(objectives))]
        else:
            scaled_losses = objectives

        if not self.ub:
            # Apply IMTL-G on the gradient of the shared parameters (tasks-specific parameters, including self.st,
            # are updated using the scaled objective sum)
            MTLOptim.custom_backwards(self, scaled_losses)
        else:
            # Apply IMTL-G on the gradient of the shared representation
            # (tasks-specific parameters, including self.st, are updated using the scaled objective sum)
            # Use approximation given by the gradient of the loss w.r.t. shared parameters to find convex combination
            # coefficients, then backward on the scaled sum
            z_grads = []
            for obj in scaled_losses:
                # the z_grads are different for each batch entry: this dimension is linearized as suggested
                # by the MGDA authors (detail not specified in the IMTL paper)
                z_grads.append(
                    torch.autograd.grad(obj, self.shared_repr, only_inputs=True, retain_graph=True)[0].view(-1))
            self.shared_repr = None
            alpha = self._get_update_direction(z_grads, None, scaled_losses, return_alpha=True)
            del z_grads
            objective = sum([obj * alpha[idx] for idx, obj in enumerate(scaled_losses)])
            objective.backward(retain_graph=True)

            # standard_head_backward is called on the specialized parameters as well as on self.st (if learning it)
            standard_sgd_params = self.specialized_params + [self.st] if self.learn_scaling else self.specialized_params
            standard_head_backward(scaled_losses, specialized_params=standard_sgd_params)

        if self.learn_scaling:
            self.st_optimizer.step()  # the grad is accumulated above
            self.st_optimizer.zero_grad()

    def set_auxiliaries(self, **kwargs):
        # Pass any auxiliary parameter.
        if self.ub:
            self.shared_repr = kwargs["shared_repr"]
# --------------------------- weight clipping ----------------------------
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




# ---------------- Fisher–Rao geometry on the simplex ----------------


class DynamicWeightAverage:
    """
    Dynamic Weight Average for multi-task learning
    Based on "End-to-End Multi-Task Learning with Attention"
    """
    
    def __init__(self, num_tasks: int, temperature: float = 2.0):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses = None
        self.weights = torch.ones(num_tasks) / num_tasks
        
    def update_weights(self, current_losses: List[float]) -> List[float]:
        """Update task weights based on loss decrease rates"""
        if self.prev_losses is None:
            self.prev_losses = current_losses
            return self.weights.tolist()
            
        # Calculate relative loss decrease
        loss_ratios = []
        for i in range(self.num_tasks):
            if self.prev_losses[i] > 0:
                ratio = current_losses[i] / self.prev_losses[i]
                loss_ratios.append(ratio)
            else:
                loss_ratios.append(1.0)
                
        loss_ratios = torch.tensor(loss_ratios)
        
        # Apply temperature and softmax
        weights = torch.nn.functional.softmax(loss_ratios / self.temperature, dim=0)
        self.weights = weights * self.num_tasks  # Scale to maintain total weight
        
        self.prev_losses = current_losses
        return self.weights.tolist()




class GradientNormalizer:
    """Normalizes gradients across multiple loss components for balanced training"""
    
    def __init__(self, alpha: float = 0.99, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps
        self.gradient_norms = defaultdict(lambda: 1.0)
        self.gradient_history = defaultdict(list)
        
    def compute_gradient_norms(self, losses: Dict[str, torch.Tensor], 
                              model: nn.Module) -> Dict[str, float]:
        """Compute gradient norms for each loss component"""
        grad_norms = {}
        
        for name, loss in losses.items():
            if loss.requires_grad and loss.grad_fn is not None:
                # Compute gradients for this loss only
                grads = torch.autograd.grad(
                    loss, 
                    model.parameters(), 
                    retain_graph=True,
                    allow_unused=True
                )
                
                # Calculate gradient norm
                total_norm = 0.0
                for grad in grads:
                    if grad is not None:
                        total_norm += grad.norm(2).item() ** 2
                grad_norm = np.sqrt(total_norm)
                
                # Update running average
                self.gradient_norms[name] = (
                    self.alpha * self.gradient_norms[name] + 
                    (1 - self.alpha) * grad_norm
                )
                grad_norms[name] = self.gradient_norms[name]
                
                # Store history for analysis
                self.gradient_history[name].append(grad_norm)
                if len(self.gradient_history[name]) > 1000:
                    self.gradient_history[name].pop(0)
        
        return grad_norms
    
    def get_balanced_weights(self, grad_norms: Dict[str, float], 
                           base_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate balanced weights based on gradient norms"""
        if not grad_norms:
            return base_weights
            
        # Use the first loss as reference (typically the main loss)
        ref_norm = list(grad_norms.values())[0]
        
        balanced_weights = {}
        for name, base_weight in base_weights.items():
            if name in grad_norms and grad_norms[name] > self.eps:
                # Scale weight inversely proportional to gradient norm
                scale = ref_norm / (grad_norms[name] + self.eps)
                balanced_weights[name] = base_weight * scale
            else:
                balanced_weights[name] = base_weight
                
        return balanced_weights

class AdaptiveLossBalancer:
    """Combines multiple loss balancing strategies"""
    
    def __init__(self, 
                 method: str = "grad_norm",
                 alpha: float = 0.99,
                 temperature: float = 2.0,
                 min_weight: float = 0.01,
                 max_weight: float = 10.0):
        self.method = method
        self.alpha = alpha
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize specific balancers
        self.gradient_normalizer = GradientNormalizer(alpha=alpha)
        self.dynamic_average = None
        self.loss_history = defaultdict(list)
        
    def balance_losses(self, 
                      losses: Dict[str, torch.Tensor],
                      base_weights: Dict[str, float],
                      model: Optional[nn.Module] = None) -> Dict[str, float]:
        """Balance losses using the specified method"""
        
        # Update loss history
        for name, loss in losses.items():
            self.loss_history[name].append(loss.item())
            if len(self.loss_history[name]) > 100:
                self.loss_history[name].pop(0)
                
        if self.method == "grad_norm" and model is not None:
            # Gradient norm balancing
            grad_norms = self.gradient_normalizer.compute_gradient_norms(losses, model)
            weights = self.gradient_normalizer.get_balanced_weights(grad_norms, base_weights)
            
        elif self.method == "magnitude":
            # Magnitude-based balancing
            weights = {}
            loss_values = {name: loss.item() for name, loss in losses.items()}
            
            # Use exponential moving average of losses
            avg_losses = {}
            for name in base_weights:
                if name in self.loss_history and self.loss_history[name]:
                    avg_losses[name] = np.mean(self.loss_history[name][-20:])
                else:
                    avg_losses[name] = loss_values.get(name, 1.0)
                    
            # Calculate inverse scaling
            ref_loss = list(avg_losses.values())[0]
            for name, base_weight in base_weights.items():
                if name in avg_losses and avg_losses[name] > 0:
                    scale = ref_loss / avg_losses[name]
                    weights[name] = base_weight * scale
                else:
                    weights[name] = base_weight
                    
        elif self.method == "dynamic":
            # Dynamic weight averaging
            if self.dynamic_average is None:
                self.dynamic_average = DynamicWeightAverage(len(losses), self.temperature)
                
            loss_list = [losses[name].item() for name in sorted(losses.keys())]
            dynamic_weights = self.dynamic_average.update_weights(loss_list)
            
            weights = {}
            for i, name in enumerate(sorted(base_weights.keys())):
                weights[name] = base_weights[name] * dynamic_weights[i]
                
        else:
            # Default: use base weights
            weights = base_weights.copy()
            
        # Clip weights to reasonable range
        for name in weights:
            weights[name] = np.clip(weights[name], self.min_weight, self.max_weight)
            
        return weights
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get balancing statistics for monitoring"""
        stats = {}
        
        # Loss statistics
        for name, history in self.loss_history.items():
            if history:
                stats[f"{name}_loss"] = {
                    "mean": np.mean(history),
                    "std": np.std(history),
                    "min": np.min(history),
                    "max": np.max(history),
                    "recent": history[-1]
                }
                
        # Gradient statistics
        if hasattr(self, 'gradient_normalizer'):
            for name, norm in self.gradient_normalizer.gradient_norms.items():
                stats[f"{name}_grad_norm"] = {"value": norm}
                
        return stats

