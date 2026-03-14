import torch
import torch.distributed as dist
from torch import nn
from typing import Dict, Tuple


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather(input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return GatherLayer.apply(input)


class SlicedDPGMMCFReg(nn.Module):
    def __init__(
        self,
        knots: int = 17,
        t_max: float = 3.0,
        num_vectors: int = 64,
        vector_chunk_size: int = 16,
        gather_distributed: bool = False,
        use_frequency_window: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if knots <= 1:
            raise ValueError("knots must be > 1")
        if t_max <= 0:
            raise ValueError("t_max must be > 0")
        if num_vectors <= 0:
            raise ValueError("num_vectors must be > 0")
        if vector_chunk_size <= 0:
            raise ValueError("vector_chunk_size must be > 0")

        self.num_vectors = int(num_vectors)
        self.vector_chunk_size = int(vector_chunk_size)
        self.gather_distributed = bool(gather_distributed)
        self.eps = float(eps)

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        trap = torch.full((knots,), dt, dtype=torch.float32)
        trap[1:-1] = 2.0 * dt

        if use_frequency_window:
            window = torch.exp(-0.5 * t.square())
        else:
            window = torch.ones_like(t)

        self.register_buffer("t", t)
        self.register_buffer("weights", trap * window)

    @staticmethod
    def _normalize_pi(pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pi = pi.clamp_min(eps)
        return pi / pi.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _maybe_unsqueeze_proj(proj: torch.Tensor) -> torch.Tensor:
        if proj.dim() == 2:
            proj = proj.unsqueeze(1)  # [B,D] -> [B,1,D]
        if proj.dim() != 3:
            raise ValueError(f"proj must be [B,N,D] or [B,D], got {tuple(proj.shape)}")
        return proj

    def _gather_if_needed(
        self,
        proj: torch.Tensor,
        pi: torch.Tensor,
        means: torch.Tensor,
        log_vars: torch.Tensor,
    ):
        if (
            self.gather_distributed
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            proj = torch.cat(gather(proj), dim=0)
            pi = torch.cat(gather(pi), dim=0)
            means = torch.cat(gather(means), dim=0)
            log_vars = torch.cat(gather(log_vars), dim=0)
        return proj, pi, means, log_vars

    def _generate_unit_vectors(
        self,
        device: torch.device,
        dtype: torch.dtype,
        dim: int,
    ) -> torch.Tensor:
        A = torch.randn(dim, self.num_vectors, device=device, dtype=dtype)
        if (
            self.gather_distributed
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            dist.broadcast(A, src=0)
        return A / A.norm(dim=0, keepdim=True).clamp_min(self.eps)

    def forward(
        self,
        proj: torch.Tensor,
        prior_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        proj = self._maybe_unsqueeze_proj(proj)

        pi = prior_params["pi"]
        means = prior_params["means"]
        log_vars = prior_params["log_vars"]

        proj, pi, means, log_vars = self._gather_if_needed(proj, pi, means, log_vars)

        B, N, D = proj.shape
        _, K = pi.shape

        if means.shape != (B, K, D):
            raise ValueError(f"means must be [{B},{K},{D}], got {tuple(means.shape)}")
        if log_vars.shape != (B, K, D):
            raise ValueError(f"log_vars must be [{B},{K},{D}], got {tuple(log_vars.shape)}")

        pi = self._normalize_pi(pi, self.eps)
        vars_ = log_vars.exp().clamp_min(self.eps)

        t = self.t.to(device=proj.device, dtype=proj.dtype)
        tt = t.view(1, 1, 1, -1)  # [1,1,1,T]
        weights = self.weights.to(device=proj.device, dtype=proj.dtype).view(1, 1, -1)

        A = self._generate_unit_vectors(proj.device, proj.dtype, D)

        total = proj.new_zeros(())
        count = 0

        for A_chunk in A.split(self.vector_chunk_size, dim=1):
            # chunk size m
            x_proj = proj @ A_chunk                                # [B,N,m]
            mu_proj = torch.einsum("bkd,dm->bkm", means, A_chunk)  # [B,K,m]
            var_proj = torch.einsum("bkd,dm->bkm", vars_, A_chunk.square()).clamp_min(self.eps)

            x_t = x_proj.unsqueeze(-1) * tt                        # [B,N,m,T]
            emp_real = x_t.cos().mean(dim=1)                       # [B,m,T]
            emp_imag = x_t.sin().mean(dim=1)                       # [B,m,T]

            atten = torch.exp(-0.5 * tt.square() * var_proj.unsqueeze(-1))  # [B,K,m,T]
            phase = tt * mu_proj.unsqueeze(-1)                               # [B,K,m,T]

            piw = pi.unsqueeze(-1).unsqueeze(-1)                  # [B,K,1,1]
            target_real = (piw * atten * phase.cos()).sum(dim=1) # [B,m,T]
            target_imag = (piw * atten * phase.sin()).sum(dim=1) # [B,m,T]

            err = (emp_real - target_real).square() + (emp_imag - target_imag).square()
            statistic = (err * weights).sum(dim=-1) * float(N)   # [B,m]

            total = total + statistic.sum()
            count += statistic.numel()

        return total / max(count, 1)