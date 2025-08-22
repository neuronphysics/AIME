from __future__ import annotations
import random
from copy import deepcopy
from typing import Iterable, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Type, Union, Sequence

from torch.optim.lr_scheduler import LRScheduler

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
@torch.no_grad()
def _force_symmetric(M: torch.Tensor) -> torch.Tensor:
    """Make matrix explicitly symmetric (lower numerical noise)."""
    return 0.5 * (M + M.transpose(0, 1))

@torch.no_grad()
def _is_finite(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


@torch.no_grad()
def _balance_transform_task_space(
    M: torch.Tensor,
    eps: float = torch.finfo(torch.float32).eps,
    max_tries: int = 3,
    jitter_scale: float = 1e-6,
    use_svd_fallback: bool = True,
) -> torch.Tensor:
    """
    Robust computation of the balance transform B in task space.

    Args:
        M: [T,T] symmetric PSD Gram matrix (J J^T or Z Z^T)
        eps: absolute floor for eigenvalue thresholding
        max_tries: how many retries with increased diagonal jitter if eigh fails
        jitter_scale: diagonal jitter factor relative to trace(M)
        use_svd_fallback: fallback to SVD if eigh does not converge

    Returns:
        B: [T,T] transform; alpha = B @ w
    """
    T = M.shape[0]
    device, dtype = M.device, M.dtype

    # 1) sanitize and upcast to float64 for decomposition
    M = _force_symmetric(M)
    if not _is_finite(M):
        M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    M64 = M.to(torch.float64)

    # 2) scale-aware jitter: trace gives us a magnitude proxy
    tr = torch.trace(M64).abs()
    jitter = (tr / (T + torch.finfo(torch.float64).eps)) * jitter_scale + eps

    # 3) attempt eigh with progressive jitter
    evals = None
    V = None
    for i in range(max_tries):
        try:
            Mreg = M64 + jitter * torch.eye(T, device=device, dtype=torch.float64)
            evals, V = torch.linalg.eigh(Mreg, UPLO='U')   # ascending
            break
        except RuntimeError:
            jitter = jitter * 10.0

    if evals is None and use_svd_fallback:
        # final fallback: SVD
        U, S, _ = torch.linalg.svd(M64 + jitter * torch.eye(T, device=device, dtype=torch.float64), full_matrices=False)
        evals, V = S, U

    if evals is None:
        # give up: identity transform (no alignment, but safe)
        return torch.eye(T, device=device, dtype=dtype)

    # 4) keep positive spectrum with a relative threshold
    rel = evals.max() * T * torch.finfo(torch.float64).eps
    thresh = max(float(eps), float(rel))
    mask = evals > thresh

    if mask.sum() == 0:
        return torch.eye(T, device=device, dtype=dtype)

    # order descending within positive subspace
    pos_vals = evals[mask]
    pos_vecs = V[:, mask]
    order = torch.argsort(pos_vals, descending=True)
    lam = pos_vals[order]               # [R]
    Vpos = pos_vecs[:, order]           # [T,R]

    lam_R = lam[-1]                     # smallest positive eigenvalue
    sigma_inv = torch.diag(1.0 / torch.sqrt(torch.clamp(lam, min=eps)))
    B64 = torch.sqrt(lam_R) * (Vpos @ sigma_inv @ Vpos.transpose(0, 1))

    return B64.to(dtype=dtype)

@torch.no_grad()
def aligned_mtl_weights(
    J: torch.Tensor,
    pref: Optional[torch.Tensor] = None,
    normalize_task_grads: bool = True,
    alpha_blend: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute safe task coefficients and aggregate gradient in parameter space.

    Args:
        J: [T,P] per-task flattened grads wrt shared params
        pref: [T] preference vector (defaults to uniform)
        normalize_task_grads: if True, row-normalize J to equalize task scales
        alpha_blend: convex blend with uniform weights (0=no blend, 1=only aligned)
        eps: small constant for numerical stability

    Returns:
        g_flat: [P] aggregated gradient
        alpha:  [T] task coefficients
    """
    T, P = J.shape
    device, dtype = J.device, J.dtype

    if not _is_finite(J):
        J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

    # optional row-normalization (preconditioning) to avoid one task dominating
    if normalize_task_grads:
        norms = J.norm(dim=1, keepdim=True).clamp_min(eps)
        Jn = J / norms
    else:
        Jn = J

    # preferences
    if pref is None:
        w = torch.full((T,), 1.0 / T, device=device, dtype=dtype)
    else:
        w = pref.to(device=device, dtype=dtype)
        w = w / (w.sum() + eps)

    # Gramian in task space + robust B
    M = Jn @ Jn.t()  # [T,T]
    B = _balance_transform_task_space(M, eps=eps)

    alpha_aligned = B @ w  # [T]

    # blend with uniform to avoid starvation of any task
    if alpha_blend is not None and alpha_blend > 0.0:
        alpha = (1.0 - alpha_blend) * w + alpha_blend * alpha_aligned
    else:
        alpha = alpha_aligned

    # safety: clip extremely large |alpha| to avoid explosions
    alpha = torch.clamp(alpha, min=-10.0, max=10.0)

    # aggregate back to parameter space
    g_flat = J.transpose(0, 1) @ alpha

    # if something went wrong, fall back to simple average
    if not _is_finite(g_flat):
        g_flat = J.mean(dim=0)
        alpha = w

    return g_flat, alpha


# ---------------------- Optional UB variant (representation-space) ---------------------- #

@torch.no_grad()
def aligned_mtl_ub_weights(
    Z: torch.Tensor,
    pref: Optional[torch.Tensor] = None,
    normalize_task_grads: bool = True,
    alpha_blend: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Upper-bound variant in representation space.

    Args:
        Z: [T,D] per-task grads wrt a shared representation H
        pref: [T] preference vector (defaults to uniform)
        normalize_task_grads: row-normalize Z
        alpha_blend: convex blend with uniform weights
        eps: small constant
    """
    T, D = Z.shape
    device, dtype = Z.device, Z.dtype

    if not _is_finite(Z):
        Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize_task_grads:
        norms = Z.norm(dim=1, keepdim=True).clamp_min(eps)
        Zn = Z / norms
    else:
        Zn = Z

    if pref is None:
        w = torch.full((T,), 1.0 / T, device=device, dtype=dtype)
    else:
        w = pref.to(device=device, dtype=dtype)
        w = w / (w.sum() + eps)

    M = Zn @ Zn.t()
    B = _balance_transform_task_space(M, eps=eps)
    alpha_aligned = B @ w
    if alpha_blend is not None and alpha_blend > 0.0:
        alpha = (1.0 - alpha_blend) * w + alpha_blend * alpha_aligned
    else:
        alpha = alpha_aligned
    alpha = torch.clamp(alpha, min=-10.0, max=10.0)

    gH = Z.transpose(0, 1) @ alpha
    if not _is_finite(gH):
        gH = Z.mean(dim=0)
        alpha = w
    return gH, alpha

# ---------------------- grad flatten / scatter ---------------------

def _flatten_grads_per_task(loss: torch.Tensor, params: Sequence[nn.Parameter]) -> torch.Tensor:
    """
    Per-task gradient wrt 'params' (no graph creation).
    Returns a flat vector [P].
    """
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    flat = []
    for p, g in zip(params, grads):
        if g is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(g.reshape(-1))
    return torch.cat(flat, dim=0)

def _assign_flat_grad_to_params(flat: torch.Tensor, params: Sequence[nn.Parameter]) -> None:
    """Scatter a flat gradient vector back into param .grad buffers (in-place)."""
    assert torch.isfinite(flat).all(), "Aggregated gradient contains NaN/Inf."
    offset = 0
    for p in params:
        n = p.numel()
        g_slice = flat[offset:offset + n].view_as(p)
        offset += n
        if p.grad is None:
            p.grad = g_slice.detach().clone()
        else:
            p.grad.copy_(g_slice)

# ---------------------------- optimizer ----------------------------

class AlignedMTLOptimizer:
    """
    Memory-conscious wrapper around a base optimizer.

    - Applies robust Aligned-MTL to 'shared_params'
    - Updates non-shared 'head' params by a single autograd.grad on a weighted sum
    """

    def __init__(
        self,
        model: nn.Module,
        shared_params: Sequence[nn.Parameter],
        base_optim: Optimizer,
        head_update: str = 'pref',
        normalize_task_grads: bool = True,
        alpha_blend: float = 0.5,
    ):
        self.model = model
        self.shared_params = tuple(shared_params)
        self.base_optim = base_optim
        self.head_update = head_update
        self.normalize_task_grads = normalize_task_grads
        self.alpha_blend = alpha_blend

        shared_ids = {id(p) for p in self.shared_params}
        self.head_params = tuple(p for p in self.model.parameters() if id(p) not in shared_ids)

    def zero_grad(self) -> None:
        self.base_optim.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _pref_weights(self, T: int, device, dtype, pref_vector: Optional[torch.Tensor]) -> torch.Tensor:
        if pref_vector is None:
            w = torch.full((T,), 1.0 / T, device=device, dtype=dtype)
        else:
            w = pref_vector.to(device=device, dtype=dtype)
            w = w / (w.sum() + 1e-12)
        return w

    def step(
        self,
        task_losses: Sequence[torch.Tensor],
        pref_vector: Optional[torch.Tensor] = None,
        use_upper_bound: bool = False,
        rep_grads: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self.zero_grad()
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        T = len(task_losses)
        w = self._pref_weights(T, device, dtype, pref_vector)

        if use_upper_bound:
            assert rep_grads is not None, "Pass rep_grads=[T,D] for UB variant."
            gH, alpha = aligned_mtl_ub_weights(
                rep_grads, w, normalize_task_grads=self.normalize_task_grads, alpha_blend=self.alpha_blend
            )
            raise NotImplementedError("Hook your representation H and call H.backward(gH).")
        else:
            flats = []
            for L in task_losses:
                flats.append(_flatten_grads_per_task(L, self.shared_params))
            J = torch.stack(flats, dim=0)

            g_flat, alpha = aligned_mtl_weights(
                J, w, normalize_task_grads=self.normalize_task_grads, alpha_blend=self.alpha_blend
            )

            _assign_flat_grad_to_params(g_flat, self.shared_params)

        if self.head_params and self.head_update != 'none':
            if self.head_update == 'pref':
                head_loss = sum(w[i] * task_losses[i] for i in range(T))
            elif self.head_update == 'sum':
                head_loss = sum(task_losses)
            else:
                raise ValueError(f"Unknown head_update={self.head_update}")

            head_grads = torch.autograd.grad(
                head_loss, self.head_params, retain_graph=False, create_graph=False, allow_unused=True
            )
            for p, g in zip(self.head_params, head_grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.copy_(g)

        self.base_optim.step()
        return {"alpha": alpha.detach(), "g_norm": (g_flat.norm().detach() if not use_upper_bound else torch.tensor(float('nan'), device=device))}

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
