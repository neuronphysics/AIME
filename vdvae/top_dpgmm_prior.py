from __future__ import annotations
"""

1) Merge candidates are ranked with an analytic surrogate score computed from
   sufficient statistics, then only the top few are evaluated exactly.

2) Birth initialization uses divisive splitting along the dominant residual
   variance direction instead of random seeding.

3) The design remains close to Hughes & Sudderth 2013 memoized VI:
   - memoized batches
   - cached sufficient statistics
   - nonlocal birth/merge/delete on a global component bank and accept or reject number of clusters

4) The prior remains conditional:
       p(z_t^top | h_t) = sum_k pi_k(h_t) N(z_t^top ; mu_k, Sigma_k)

   with global component bank {mu_k, Sigma_k} and a context-conditioned
   Kumaraswamy gate over weights pi_k(h_t).

Notes:
- Replay buffer stores (h_t, z_t^top, posterior_mean, posterior_logvar, seq_id, t, mask).
- During the online epoch, use a frozen snapshot of this prior for KL.
- At epoch end, fit/update this model on replay-buffer posterior samples first, then freeze it for the next epoch.
"""
import copy
import random, gc, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple,Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vdvae.vae_helpers import Block

LOG2PI = math.log(2.0 * math.pi)
EULER_GAMMA = 0.5772156649

def stable_logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    return m.squeeze(dim) + torch.log(torch.exp(x - m).sum(dim=dim).clamp_min(torch.finfo(x.dtype).eps))


def sample_diag_gaussian(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mean + torch.exp(0.5 * logvar.clamp(min=-7.0, max=4.0)) * torch.randn_like(mean)


# Kumaraswamy / Beta helpers
def beta_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(a.dtype).eps
    return torch.lgamma(a + eps) + torch.lgamma(b + eps) - torch.lgamma(a + b + eps)

def kumar2beta_kl(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    n_approx: int = 10,
    eps: float = 100 * torch.finfo(torch.float32).eps,
) -> torch.Tensor:
    #equation (12): https://arxiv.org/pdf/1605.06197
    a = torch.clamp(a, min=eps, max=20)
    b = torch.clamp(b, min=eps, max=20)
    alpha = torch.clamp(alpha, min=eps)
    beta = torch.clamp(beta, min=eps)
    ab = a * b
    a_inv = torch.reciprocal(a + eps)
    b_inv = torch.reciprocal(b + eps)
    terms = []
    for m in range(1, n_approx + 1):
        terms.append(beta_fn(m * a_inv, b) - torch.log(torch.as_tensor(float(m), device=a.device, dtype=a.dtype) + ab))
    log_taylor = torch.logsumexp(torch.stack(terms, dim=-1), dim=-1)
    kl = (beta - 1.0) * b * torch.exp(log_taylor)
    psi_b = torch.digamma(b + eps)
    term1 = ((a - alpha) / (a + eps)) * (-EULER_GAMMA - psi_b - b_inv)
    term2 = torch.log(ab + eps) + beta_fn(alpha, beta)
    term2 = term2 + (-(b - 1.0) / (b + eps))
    return torch.clamp(kl + term1 + term2, min=0.0)


def kumar_expected_log_v(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 100 * torch.finfo(torch.float32).eps,
) -> torch.Tensor:
    a = torch.clamp(a, min=eps, max=20)
    b = torch.clamp(b, min=eps, max=20)
    a_inv = torch.reciprocal(a + eps)
    b_inv = torch.reciprocal(b + eps)
    return a_inv * (-EULER_GAMMA - torch.digamma(b + eps) - b_inv)


def kumar_expected_log_1mv(
    a: torch.Tensor,
    b: torch.Tensor,
    n_approx: int = 10,
    eps: float = 100 * torch.finfo(torch.float32).eps,
) -> torch.Tensor:
    a = torch.clamp(a, min=eps, max=20)
    b = torch.clamp(b, min=eps, max=20)
    ab = a * b
    a_inv = torch.reciprocal(a + eps)
    terms = []
    for m in range(1, n_approx + 1):
        terms.append(
            beta_fn(m * a_inv, b)
            - torch.log(torch.as_tensor(float(m), device=a.device, dtype=a.dtype) + ab)
        )
    return -b * torch.exp(torch.logsumexp(torch.stack(terms, dim=-1), dim=-1))


def kumar_mean(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 100 * torch.finfo(torch.float32).eps,
) -> torch.Tensor:
    a = torch.clamp(a, min=eps, max=20)
    b = torch.clamp(b, min=eps, max=20)
    one_plus_a_inv = 1.0 + torch.reciprocal(a + eps)
    log_mean = torch.log(b + eps) + beta_fn(one_plus_a_inv, b)
    return torch.exp(log_mean)


def beta_kl(q_a: torch.Tensor, q_b: torch.Tensor, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
    if q_a.numel() == 0:
        return torch.zeros((), device=p_a.device if torch.is_tensor(p_a) else q_b.device, dtype=q_b.dtype)
    log_B_q = torch.lgamma(q_a) + torch.lgamma(q_b) - torch.lgamma(q_a + q_b)
    log_B_p = torch.lgamma(p_a) + torch.lgamma(p_b) - torch.lgamma(p_a + p_b)
    e_log_v = torch.digamma(q_a) - torch.digamma(q_a + q_b)
    e_log_1mv = torch.digamma(q_b) - torch.digamma(q_a + q_b)
    return log_B_p - log_B_q + (q_a - p_a) * e_log_v + (q_b - p_b) * e_log_1mv


def normal_gamma_kl(
    q_m: torch.Tensor,
    q_kappa: torch.Tensor,
    q_alpha: torch.Tensor,
    q_beta: torch.Tensor,
    p_m0: torch.Tensor,
    p_kappa0: float,
    p_alpha0: float,
    p_beta0: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.finfo(q_m.dtype).eps
    p_kappa0_t = torch.as_tensor(p_kappa0, device=q_m.device, dtype=q_m.dtype)
    p_alpha0_t = torch.as_tensor(p_alpha0, device=q_m.device, dtype=q_m.dtype)
    e_tau = q_alpha / q_beta.clamp_min(epsilon)
    e_log_tau = torch.digamma(q_alpha) - torch.log(q_beta.clamp_min(epsilon))
    kl_gamma = (
        q_alpha * torch.log(q_beta.clamp_min(epsilon))
        - torch.lgamma(q_alpha)
        - p_alpha0_t * torch.log(p_beta0.clamp_min(epsilon))
        + torch.lgamma(p_alpha0_t)
        + (q_alpha - p_alpha0_t) * e_log_tau
        + (p_beta0 - q_beta) * e_tau
    )
    kl_normal = 0.5 * ( torch.log(q_kappa.clamp_min(epsilon) / p_kappa0_t) + p_kappa0_t / q_kappa.clamp_min(epsilon) - 1.0 + p_kappa0_t * e_tau * (q_m - p_m0).pow(2))
    return kl_gamma + kl_normal


# Replay buffer

@dataclass
class EpochBatch:
    batch_id: int
    h: torch.Tensor
    z: torch.Tensor
    posterior_mean: Optional[torch.Tensor]
    posterior_logvar: Optional[torch.Tensor]
    valid_mask: Optional[torch.Tensor]
    seq_id: Optional[torch.Tensor]
    t_index: Optional[torch.Tensor]


class ReplayBufferConditional:
    """
    Slot-aware replay buffer.

    Stores:
        h_t              [B, C, H, W]        once per image/time step
        z / mu / logvar   [B, S, D, 1, 1]    slot tensors

    freeze_epoch() returns old-style EpochBatch objects where:
        batch.h          [N, C, H, W]
        batch.z          [N, D, 1, 1]

    So ConditionalTopDPGMM does not need to be rewritten.
    """

    def __init__(self):
        self._next_seq_id = 0
        self.clear()

    def clear(self):
        self._h, self._z = [], []
        self._mu, self._lv = [], []
        self._mask, self._seq, self._t = [], [], []

    def __len__(self) -> int:
        # number of valid slot rows, not number of image contexts
        return int(sum(int(m.sum().item()) for m in self._mask))

    def is_empty(self) -> bool:
        return len(self._z) == 0 or len(self) == 0

    @torch.no_grad()
    def allocate_seq_ids(self, batch_size: int, device=None) -> torch.Tensor:
        start = self._next_seq_id
        stop = start + int(batch_size)
        self._next_seq_id = stop
        out = torch.arange(start, stop, dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

    @staticmethod
    def _as_slot_map(x: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
        """
        Accept [B,S,D] and convert to [B,S,D,1,1].
        Accept [B,S,D,1,1] unchanged.
        Do NOT accept already-flattened [B*S,D,1,1], because then we lose
        the image-to-slot structure needed to store h_t only once.
        """
        if x is None:
            return None

        if x.dim() == 3:
            return x[:, :, :, None, None].contiguous()

        if x.dim() == 5 and x.shape[-2:] == (1, 1):
            return x.contiguous()

        raise ValueError(
            f"{name} must be [B,S,D] or [B,S,D,1,1], got {tuple(x.shape)}"
        )

    @staticmethod
    def _image_level_ids(
        x: Optional[torch.Tensor],
        *,
        B: int,
        S: int,
        device: torch.device,
        fill_value: int,
        name: str,
    ) -> torch.Tensor:
        """
        seq_id and t_index should be image-level [B].
        If caller passes [B,S], verify all slots have the same value and keep [:,0].
        """
        if x is None:
            return torch.full((B,), fill_value, device=device, dtype=torch.long)

        x = x.to(device=device)

        if x.dim() == 1:
            if x.shape[0] != B:
                raise ValueError(f"{name} must have shape [B], got {tuple(x.shape)} with B={B}")
            return x.to(torch.long)

        if x.dim() == 2:
            if x.shape != (B, S):
                raise ValueError(f"{name} must have shape [B,S], got {tuple(x.shape)} expected {(B, S)}")

            # This should be true for your use case because seq_id/t_index are shared by all slots.
            if not torch.equal(x, x[:, :1].expand_as(x)):
                raise ValueError(
                    f"{name} differs across slots. This buffer assumes one {name} per image/time step."
                )
            return x[:, 0].to(torch.long)

        raise ValueError(f"{name} must be [B] or [B,S], got {tuple(x.shape)}")

    @torch.no_grad()
    def add_step_batch(
        self,
        h_t: torch.Tensor,
        z_top_map: torch.Tensor,
        posterior_mean: Optional[torch.Tensor],
        posterior_logvar: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        seq_id: Optional[torch.Tensor] = None,
        t_index: Optional[torch.Tensor] = None,
    ):
        """
        h_t:             [B,C,H,W]
        z_top_map:       [B,S,D] or [B,S,D,1,1]
        posterior_mean:  [B,S,D] or [B,S,D,1,1]
        posterior_logvar:[B,S,D] or [B,S,D,1,1]
        valid_mask:      [B] or [B,S]
        """
        if h_t.dim() != 4:
            raise ValueError(f"h_t must be [B,C,H,W], got {tuple(h_t.shape)}")

        z = self._as_slot_map(z_top_map, "z_top_map")
        mu = self._as_slot_map(posterior_mean, "posterior_mean")
        lv = self._as_slot_map(posterior_logvar, "posterior_logvar")

        B, S = z.shape[:2]

        if h_t.shape[0] != B:
            raise ValueError(
                f"h_t batch size {h_t.shape[0]} does not match slot batch size {B}"
            )

        if mu is not None and mu.shape[:2] != (B, S):
            raise ValueError(f"posterior_mean shape {tuple(mu.shape)} incompatible with z shape {tuple(z.shape)}")

        if lv is not None and lv.shape[:2] != (B, S):
            raise ValueError(f"posterior_logvar shape {tuple(lv.shape)} incompatible with z shape {tuple(z.shape)}")

        if valid_mask is None:
            slot_mask = torch.ones(B, S, device=z.device, dtype=torch.bool)
        else:
            vm = valid_mask.to(device=z.device, dtype=torch.bool)
            if vm.dim() == 1:
                if vm.shape[0] != B:
                    raise ValueError(f"valid_mask [B] has shape {tuple(vm.shape)}, expected [{B}]")
                slot_mask = vm[:, None].expand(B, S).contiguous()
            elif vm.dim() == 2:
                if vm.shape != (B, S):
                    raise ValueError(f"valid_mask [B,S] has shape {tuple(vm.shape)}, expected {(B, S)}")
                slot_mask = vm.contiguous()
            else:
                raise ValueError(f"valid_mask must be [B] or [B,S], got {tuple(vm.shape)}")

        image_keep = slot_mask.any(dim=1)
        if not image_keep.any():
            return

        keep_idx = torch.nonzero(image_keep, as_tuple=False).flatten()

        seq_img = self._image_level_ids(
            seq_id, B=B, S=S, device=z.device, fill_value=-1, name="seq_id"
        )
        t_img = self._image_level_ids(
            t_index, B=B, S=S, device=z.device, fill_value=-1, name="t_index"
        )

        # Store exact h_t values once per valid image/time step.
        self._h.append(h_t.index_select(0, keep_idx).detach().cpu().contiguous())
        self._z.append(z.index_select(0, keep_idx).detach().cpu().contiguous())
        self._mu.append(None if mu is None else mu.index_select(0, keep_idx).detach().cpu().contiguous())
        self._lv.append(None if lv is None else lv.index_select(0, keep_idx).detach().cpu().contiguous())
        self._mask.append(slot_mask.index_select(0, keep_idx).detach().cpu().contiguous())
        self._seq.append(seq_img.index_select(0, keep_idx).detach().cpu().contiguous())
        self._t.append(t_img.index_select(0, keep_idx).detach().cpu().contiguous())

    def freeze_epoch(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> List[EpochBatch]:

        if len(self._z) == 0:
            raise RuntimeError("Replay buffer is empty.")

        # Image-level tensors.
        h = torch.cat(self._h, dim=0).contiguous()          # [N_img,C,H,W]
        z = torch.cat(self._z, dim=0).contiguous()          # [N_img,S,D,1,1]
        mask = torch.cat(self._mask, dim=0).contiguous()    # [N_img,S]
        seq = torch.cat(self._seq, dim=0).contiguous()      # [N_img]
        t = torch.cat(self._t, dim=0).contiguous()          # [N_img]

        mu = None
        if len(self._mu) > 0 and self._mu[0] is not None:
            mu = torch.cat(self._mu, dim=0).contiguous()    # [N_img,S,D,1,1]

        lv = None
        if len(self._lv) > 0 and self._lv[0] is not None:
            lv = torch.cat(self._lv, dim=0).contiguous()    # [N_img,S,D,1,1]

        N_img, S = z.shape[:2]

        flat_img = torch.arange(N_img, dtype=torch.long).view(N_img, 1).expand(N_img, S).reshape(-1)
        flat_slot = torch.arange(S, dtype=torch.long).view(1, S).expand(N_img, S).reshape(-1)
        flat_mask = mask.reshape(-1)

        valid = torch.nonzero(flat_mask.bool(), as_tuple=False).flatten()
        if valid.numel() == 0:
            raise RuntimeError("Replay buffer has no valid slot rows.")

        flat_img = flat_img.index_select(0, valid)
        flat_slot = flat_slot.index_select(0, valid)

        N = flat_img.shape[0]
        perm = torch.randperm(N) if shuffle else torch.arange(N)

        flat_img = flat_img.index_select(0, perm)
        flat_slot = flat_slot.index_select(0, perm)

        out: List[EpochBatch] = []

        for s0 in range(0, N, batch_size):
            e0 = min(s0 + batch_size, N)

            img_idx = flat_img[s0:e0]
            slot_idx = flat_slot[s0:e0]

            # Here is the only place h is repeated, and only for this small fitting batch.
            h_b = h.index_select(0, img_idx).contiguous()       # [batch,C,H,W]
            z_b = z[img_idx, slot_idx].contiguous()             # [batch,D,1,1]

            mu_b = None if mu is None else mu[img_idx, slot_idx].contiguous()
            lv_b = None if lv is None else lv[img_idx, slot_idx].contiguous()

            seq_b = seq.index_select(0, img_idx).contiguous()
            t_b = t.index_select(0, img_idx).contiguous()
            mask_b = torch.ones(e0 - s0, dtype=torch.bool)

            out.append(
                EpochBatch(
                    batch_id=len(out),
                    h=h_b,
                    z=z_b,
                    posterior_mean=mu_b,
                    posterior_logvar=lv_b,
                    valid_mask=mask_b,
                    seq_id=seq_b,
                    t_index=t_b,
                )
            )

        return out


# Fixed-size conditional Kumaraswamy gate
class ConditionalKumaraswamyGate(nn.Module):
    """
    Preallocated gate with active masking.

    We expose only the first active_K - 1 stick rows in the forward pass.

    This preserves optimizer state across birth/delete/merge.
    """
    def __init__(self, h_shape: Tuple[int, ...], max_K: int, hidden_dim: int = 256, use_3x3: bool = True, zero_last: bool = False):
        super().__init__()
        self.h_shape = h_shape
        self.max_K = int(max_K)
        self.hidden_dim = int(hidden_dim)
        self.active_K = int(max_K)

        h_channels, h_height, h_width = h_shape
        self.h_encoder = Block(
            in_width=h_channels,
            middle_width=hidden_dim,
            out_width=hidden_dim,
            down_rate=None,          # keep spatial size here
            residual=False,          # h_channels != hidden_dim in general
            use_3x3=use_3x3,
            zero_last=zero_last,
        )
        self.out_a = nn.Conv2d(hidden_dim, max(1, max_K - 1), kernel_size=(h_height, h_width), stride=1, padding=0, bias=True)
        self.out_b = nn.Conv2d(hidden_dim, max(1, max_K - 1), kernel_size=(h_height, h_width), stride=1, padding=0, bias=True)
        self.register_buffer('active_mask', torch.zeros(max(1, max_K - 1), dtype=torch.bool))
        self.reset_parameters()
        self.set_active_K(self.active_K)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.out_a.weight.mul_(0.01)
            self.out_b.weight.mul_(0.01)

    def _embed(self, h: torch.Tensor) -> torch.Tensor:
        return self.h_encoder(h) #[B, C, H, W] -> [B, hidden_dim, H, W]


    def set_active_K(self, K: int):
        self.active_K = int(max(1, min(K, self.max_K)))
        self.active_mask.zero_()
        n = max(0, self.active_K - 1)
        if n > 0:
            self.active_mask[:n] = True
    @staticmethod
    def inv_softplus(x: float, device, dtype):
        x = torch.as_tensor(x, device=device, dtype=dtype)
        return torch.log(torch.expm1(x).clamp_min(torch.finfo(dtype).eps))

    @torch.no_grad()
    def init_new_rows(self, old_K: int, new_K: int, dp_alpha: float, scale: float = 1e-2):
        """
        Initialize newly exposed stick rows with small values.
        """
        old_rows = max(0, old_K - 1)
        new_rows = max(0, new_K - 1)
        if new_rows <= old_rows:
            return
        sl = slice(old_rows, new_rows)
        self.out_a.weight[sl].normal_(mean=0.0, std=scale)
        self.out_b.weight[sl].normal_(mean=0.0, std=scale)
        self.out_a.bias[sl].fill_(self.inv_softplus(1.0, self.out_a.bias.device, self.out_a.bias.dtype))
        self.out_b.bias[sl].fill_(self.inv_softplus(dp_alpha, self.out_b.bias.device, self.out_b.bias.dtype))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self._embed(h)
        if self.active_K == 1:
            z = feat[:, :0, 0, 0]
            return {'a': z, 'b': z}
        n = self.active_K - 1
        a_full = F.softplus(self.out_a(feat)) + 1e-4
        b_full = F.softplus(self.out_b(feat)) + 1e-4
        return {'a': a_full[:, :n, 0, 0], 'b': b_full[:, :n, 0, 0]}

    def expected_log_pi(self, h: torch.Tensor) -> torch.Tensor:
        if self.active_K == 1:
            return torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
        out = self.forward(h)
        a, b = out['a'], out['b']
        e_log_v = kumar_expected_log_v(a, b)
        e_log_1mv = kumar_expected_log_1mv(a, b, n_approx=10)
        B = h.shape[0]
        elogpi = torch.empty(B, self.active_K, device=h.device, dtype=h.dtype)
        prefix = torch.zeros(B, device=h.device, dtype=h.dtype)
        for k in range(self.active_K):
            if k < self.active_K - 1:
                elogpi[:, k] = e_log_v[:, k] + prefix
                prefix = prefix + e_log_1mv[:, k]
            else:
                elogpi[:, k] = prefix
        return elogpi

    def mean_pi(self, h: torch.Tensor) -> torch.Tensor:
        if self.active_K == 1:
            return torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)
        out = self.forward(h)
        a, b = out['a'], out['b']
        ev = kumar_mean(a, b).clamp_min(torch.finfo(a.dtype).eps)
        e1 = (1.0 - ev).clamp_min(torch.finfo(ev.dtype).eps)
        B = h.shape[0]
        pi = torch.empty(B, self.active_K, device=h.device, dtype=h.dtype)
        prefix = torch.ones(B, device=h.device, dtype=h.dtype)
        for k in range(self.active_K):
            if k < self.active_K - 1:
                pi[:, k] = ev[:, k] * prefix
                prefix = prefix * e1[:, k]
            else:
                pi[:, k] = prefix
        return pi / pi.sum(dim=1, keepdim=True).clamp_min(torch.finfo(pi.dtype).eps)


# Component posterior, cache, snapshots
@dataclass
class TensorDiagComponentPosterior:
    mean: torch.Tensor
    kappa: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


@dataclass
class MemoizedBatchCache:
    resp: Optional[torch.Tensor] = None       # [N,K]
    Nk: Optional[torch.Tensor] = None         # [K]
    S1: Optional[torch.Tensor] = None         # [K,C,H,W]
    S2: Optional[torch.Tensor] = None         # [K,C,H,W]
    entropy: Optional[torch.Tensor] = None    # scalar


@dataclass
class FrozenConditionalTopPrior:
    comp_mean: torch.Tensor
    comp_var: torch.Tensor
    gate_state: Dict[str, torch.Tensor]
    K: int
    max_K: int
    objective: float


@dataclass
class StructuralCheckpoint:
    K: int
    comp: List[TensorDiagComponentPosterior]
    Nk: torch.Tensor
    S1: torch.Tensor
    S2: torch.Tensor
    entropy: torch.Tensor
    cache_resp: Dict[int, torch.Tensor]
    gate_state: Dict[str, torch.Tensor]
    gate_optim_state: Dict[str, Any]


# Main conditional DPGMM
class ConditionalTopDPGMM(nn.Module):
    def __init__(
        self,
        z_shape: Tuple[int, int, int],
        h_shape: Tuple[int, ...],
        max_components: int = 32,
        init_components: int = 4,
        dp_alpha: float = 1.0,
        prior_m0: float = 0.0,
        prior_kappa0: float = 0.5,
        prior_alpha0: float = 2.0,
        prior_beta0: float = 1.0,
        gate_hidden_dim: int = 64,
        gate_lr: float = 1e-4,
        gate_kl_weight: float = 1.0,
        birth_kfresh: int = 4,
        birth_resp_threshold: float = 0.10,
        birth_subset_max: int = 4096,
        merge_top_pairs: int = 8,
        merge_screen_topk: int = 3,
        delete_min_count: float = 2.0,
        delete_min_seqs: int = 2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.C, self.H, self.W = z_shape
        self.z_shape = z_shape
        self.h_shape = h_shape
        self.max_components = int(max_components)
        self.K = int(min(init_components, max_components))
        self.dp_alpha = float(dp_alpha)
        self.prior_m0_scalar = float(prior_m0)
        self.prior_kappa0 = float(prior_kappa0)
        self.prior_alpha0 = float(prior_alpha0)
        self.prior_beta0_scalar = float(prior_beta0)
        self.gate_kl_weight = float(gate_kl_weight)
        self.birth_kfresh = int(birth_kfresh)
        self.birth_resp_threshold = float(birth_resp_threshold)
        self.birth_subset_max = int(birth_subset_max)
        self.merge_top_pairs = int(merge_top_pairs)
        self.merge_screen_topk = int(merge_screen_topk)
        self.delete_min_count = float(delete_min_count)
        self.delete_min_seqs = int(delete_min_seqs)
        self.gate_lr = float(gate_lr)
        self.device_ = device if device is not None else torch.device('cpu')
        self.dtype = dtype

        self.gate = ConditionalKumaraswamyGate(h_shape=h_shape, max_K=self.max_components, hidden_dim=gate_hidden_dim).to(self.device_, self.dtype)
        self.gate.set_active_K(self.K)
        self.gate_optimizer = torch.optim.Adam(self.gate.parameters(), lr=gate_lr)

        self.comp: List[TensorDiagComponentPosterior] = []
        self.batches: List[EpochBatch] = []
        self.cache: Dict[int, MemoizedBatchCache] = {}
        self.Nk = None
        self.S1 = None
        self.S2 = None
        self.entropy = None
        self.to(device=self.device_, dtype=self.dtype)

    #  basic helpers
    @torch.no_grad()
    def clear_fit_cache(self):
        """
        Release epoch-end memoized VI state after frozen_snapshot() has been produced.
        Keeps learned comp/qv/gate parameters for warm-starting the next epoch.
        """
        self.batches = []
        self.cache = {}
        self.Nk = None
        self.S1 = None
        self.S2 = None
        self.entropy = None

    def _prior_m0(self) -> torch.Tensor:
        return torch.full(self.z_shape, self.prior_m0_scalar, device=self.device_, dtype=self.dtype)

    def _prior_beta0(self) -> torch.Tensor:
        return torch.full(self.z_shape, self.prior_beta0_scalar, device=self.device_, dtype=self.dtype)

    def _reset_global_summaries(self):
        self.Nk = torch.zeros(self.K, device=self.device_, dtype=self.dtype)
        self.S1 = torch.zeros(self.K, self.C, self.H, self.W, device=self.device_, dtype=self.dtype)
        self.S2 = torch.zeros_like(self.S1)
        self.entropy = torch.tensor(0.0, device=self.device_, dtype=self.dtype)


    def _cache_from_resp(
        self,
        z: torch.Tensor,
        resp: torch.Tensor,
        posterior_logvar: Optional[torch.Tensor] = None,
    ) -> MemoizedBatchCache:
        resp = resp.detach()
        z = z.detach()

        if posterior_logvar is not None:
            # Use E_q[s^2] = mu^2 + var, not only mu^2.
            var_q = torch.exp(posterior_logvar.clamp(min=-7.0, max=4.0).detach()).clamp_min(1e-8)
            second_moment = z * z + var_q
        else:
            second_moment = z * z

        Nk = resp.sum(dim=0).detach()
        S1 = torch.einsum("nk,nchw->kchw", resp, z).detach()
        S2 = torch.einsum("nk,nchw->kchw", resp, second_moment).detach()

        eps = torch.finfo(resp.dtype).eps
        entropy = (-(resp.clamp_min(eps) * resp.clamp_min(eps).log()).sum()).detach()


        return MemoizedBatchCache(resp=resp, Nk=Nk, S1=S1, S2=S2, entropy=entropy)

    def _add_cache(self, c: MemoizedBatchCache):
        self.Nk = self.Nk + c.Nk
        self.S1 = self.S1 + c.S1
        self.S2 = self.S2 + c.S2
        self.entropy = self.entropy + c.entropy

    def _sub_cache(self, c: MemoizedBatchCache):
        self.Nk = self.Nk - c.Nk
        self.S1 = self.S1 - c.S1
        self.S2 = self.S2 - c.S2
        self.entropy = self.entropy - c.entropy

    def _detach_comp(self, c: TensorDiagComponentPosterior) -> TensorDiagComponentPosterior:
        return TensorDiagComponentPosterior(
            mean=c.mean.detach().clone(),
            kappa=c.kappa.detach().clone() if torch.is_tensor(c.kappa) else torch.as_tensor(
                c.kappa, device=self.device_, dtype=self.dtype
            ),
            alpha=c.alpha.detach().clone() if torch.is_tensor(c.alpha) else torch.as_tensor(
                c.alpha, device=self.device_, dtype=self.dtype
            ),
            beta=c.beta.detach().clone(),
        )

    @torch.no_grad()
    def save_checkpoint(self) -> StructuralCheckpoint:
        return StructuralCheckpoint(
            K=self.K,
            comp=[self._detach_comp(c) for c in self.comp],
            Nk=self.Nk.detach().clone(),
            S1=self.S1.detach().clone(),
            S2=self.S2.detach().clone(),
            entropy=self.entropy.detach().clone(),
            cache_resp={
                bid: self.cache[bid].resp.detach().clone()
                for bid in self.cache
            },
            gate_state={
                k: v.detach().clone()
                for k, v in self.gate.state_dict().items()
            },
            gate_optim_state=copy.deepcopy(self.gate_optimizer.state_dict()),
        )

    @torch.no_grad()
    def restore_checkpoint(self, ckpt: StructuralCheckpoint):
        self.K = ckpt.K
        self.comp = [copy.deepcopy(c) for c in ckpt.comp]
        self.gate.load_state_dict(ckpt.gate_state, strict=True)
        self.gate.set_active_K(self.K)

        self._reset_gate_optimizer()
        self.gate_optimizer.load_state_dict(ckpt.gate_optim_state)
        self._reset_global_summaries()
        for bid, b_cpu in enumerate(self.batches):
            b = self._move_batch(b_cpu)
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            c = self._cache_from_resp(z_fit, ckpt.cache_resp[bid], posterior_logvar=b.posterior_logvar)
            self.cache[bid] = c
            self._add_cache(c)

    # initialization
    @torch.no_grad()
    def _build_component_from_stats(self, mean: torch.Tensor, var: torch.Tensor, count: float = 1.0) -> TensorDiagComponentPosterior:
        count_t = torch.as_tensor(float(max(count, 1.0)), device=self.device_, dtype=self.dtype)
        eps = torch.finfo(mean.dtype).eps
        return TensorDiagComponentPosterior(
            mean=mean.to(device=self.device_, dtype=self.dtype),
            kappa=torch.as_tensor(self.prior_kappa0 + count_t, device=self.device_, dtype=self.dtype),
            alpha=torch.as_tensor(self.prior_alpha0 + 0.5 * count_t, device=self.device_, dtype=self.dtype),
            beta=(self._prior_beta0() + 0.5 * var.to(device=self.device_, dtype=self.dtype) * count_t).clamp_min(eps),
        )

    @torch.no_grad()
    def _kmeanspp_initialize_components(self, z_all_cpu: torch.Tensor, K: int, max_points: int = 4096, n_iter: int = 5) -> List[TensorDiagComponentPosterior]:
        #https://github.com/Hzzone/torch_clustering/blob/master/torch_clustering/
        N = int(z_all_cpu.shape[0])
        if N <= 0:
            raise RuntimeError("Cannot initialize components from empty latent set.")
        K = int(max(1, min(K, N, self.max_components)))

        x = z_all_cpu.reshape(N, -1).float().cpu()
        if N > max_points:
            perm = torch.randperm(N)[:max_points]
            x = x.index_select(0, perm)
        M, D = x.shape

        centers = torch.empty(K, D, dtype=x.dtype)
        first = torch.randint(0, M, (1,)).item()
        centers[0] = x[first]
        min_dist2 = ((x - centers[0:1]) ** 2).sum(dim=1)
        for k in range(1, K):
            probs = min_dist2.clamp_min(torch.finfo(min_dist2.dtype).eps)
            probs = probs / probs.sum().clamp_min(torch.finfo(probs.dtype).eps)
            idx = torch.multinomial(probs, 1).item()
            centers[k] = x[idx]
            dist2_new = ((x - centers[k:k+1]) ** 2).sum(dim=1)
            min_dist2 = torch.minimum(min_dist2, dist2_new)

        for _ in range(max(1, int(n_iter))):
            dist2 = torch.cdist(x, centers, p=2.0) ** 2
            assign = dist2.argmin(dim=1)
            new_centers = centers.clone()
            for k in range(K):
                mask = assign == k
                if mask.any():
                    new_centers[k] = x[mask].mean(dim=0)
            if torch.allclose(new_centers, centers, atol=1e-4, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers

        global_var = x.var(dim=0, unbiased=False).reshape(self.z_shape).clamp_min(1e-2)
        comps: List[TensorDiagComponentPosterior] = []
        dist2 = torch.cdist(x, centers, p=2.0) ** 2
        assign = dist2.argmin(dim=1)
        for k in range(K):
            mask = assign == k
            if mask.any():
                mean_k = x[mask].mean(dim=0).reshape(self.z_shape)
                var_k = x[mask].var(dim=0, unbiased=False).reshape(self.z_shape).clamp_min(1e-3)
                count_k = int(mask.sum().item())
            else:
                mean_k = centers[k].reshape(self.z_shape)
                var_k = global_var
                count_k = 1
            comps.append(self._build_component_from_stats(mean_k, var_k, count=count_k))
        return comps

    @torch.no_grad()
    def initialize_from_data(self, batches: List[EpochBatch], K_init: Optional[int] = None, warm_start: bool = True):
        self.batches = list(batches)
        self.cache = {b.batch_id: MemoizedBatchCache() for b in self.batches}
        z_init = [b.posterior_mean if b.posterior_mean is not None else b.z for b in self.batches]
        z_all_cpu = torch.cat(z_init, dim=0).contiguous()
        if K_init is not None:
            target_K = int(max(1, min(K_init, self.max_components, z_all_cpu.shape[0])))
        else:
            target_K = int(max(1, min(self.K, self.max_components, z_all_cpu.shape[0])))

        can_warm = warm_start and len(self.comp) > 0 and len(self.comp) >= target_K
        self.K = target_K
        self.gate.set_active_K(self.K)

        if can_warm:
            self.comp = [copy.deepcopy(c) for c in self.comp[:self.K]]
        else:
            self.comp = self._kmeanspp_initialize_components(z_all_cpu, self.K)

        self._initialize_caches()

    @torch.no_grad()
    def _initialize_caches(self):
        self._reset_global_summaries()

        for b_cpu in self.batches:
            b = self._move_batch(b_cpu)

            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z

            resp = self.local_step(b.h, z_fit, b.posterior_logvar)
            c = self._cache_from_resp(z_fit, resp, posterior_logvar=b.posterior_logvar)

            self.cache[b.batch_id] = c
            self._add_cache(c)
        self._update_components_from_summaries()    



    def conditional_expected_log_pi(self, h: torch.Tensor) -> torch.Tensor:
        #q(v_k | h_t) = Kumaraswamy(a_k(h_t), b_k(h_t))
        return self.gate.expected_log_pi(h)

    def component_expected_log_likelihood(
        self,
        z_mean: torch.Tensor,
        posterior_logvar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes E_{q(z)q(phi_k)}[log p(z | phi_k)].

        If posterior_logvar is provided, z_mean is treated as the mean of
        q(z|x), and we use E[(z-m_k)^2] = (mu_z-m_k)^2 + var_z.
        """
        N = z_mean.shape[0]
        out = torch.empty(N, self.K, device=z_mean.device, dtype=z_mean.dtype)
        eps = torch.finfo(z_mean.dtype).eps

        if posterior_logvar is None:
            z_var = torch.zeros_like(z_mean)
        else:
            z_var = torch.exp(posterior_logvar).clamp_min(1e-8)

        for k, comp in enumerate(self.comp):
            e_log_tau = torch.digamma(comp.alpha) - torch.log(comp.beta.clamp_min(eps))
            e_tau = comp.alpha / comp.beta.clamp_min(eps)

            expected_quad = (z_mean - comp.mean).pow(2) + z_var

            term = 0.5 * (
                e_log_tau
                - LOG2PI
                - (1.0 / comp.kappa.clamp_min(eps))
                - e_tau * expected_quad
            )

            out[:, k] = term.view(N, -1).sum(dim=1)

        return out

    @torch.no_grad()
    def local_step(
        self,
        h: torch.Tensor,
        z_mean: torch.Tensor,
        posterior_logvar: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        logits = (
            self.conditional_expected_log_pi(h)
            + self.component_expected_log_likelihood(z_mean, posterior_logvar)
        )

        logits = logits / max(float(temperature), torch.finfo(h.dtype).eps)
        return torch.softmax(logits, dim=1)

    # updates
    def _update_components_from_summaries(self):
        m0 = self._prior_m0()
        beta0 = self._prior_beta0()
        new_comp = []
        for k in range(self.K):
            Nk = self.Nk[k].clamp_min(torch.finfo(self.Nk.dtype).eps)
            S1 = self.S1[k]
            S2 = self.S2[k]
            mean_emp = S1 / Nk
            sq_centered = (S2 - (S1 * S1) / Nk).clamp_min(0.0)
            kappa = self.prior_kappa0 + Nk
            mean = (self.prior_kappa0 * m0 + S1) / kappa
            alpha = self.prior_alpha0 + 0.5 * Nk
            beta = (beta0 + 0.5 * sq_centered + 0.5 * (self.prior_kappa0 * Nk / kappa) * (mean_emp - m0).pow(2)).clamp_min(torch.finfo(beta0.dtype).eps)
            new_comp.append(
                TensorDiagComponentPosterior(
                    mean=mean.detach(),
                    kappa=kappa.detach() if torch.is_tensor(kappa) else torch.as_tensor(kappa, device=self.device_, dtype=self.dtype),
                    alpha=alpha.detach() if torch.is_tensor(alpha) else torch.as_tensor(alpha, device=self.device_, dtype=self.dtype),
                    beta=beta.detach(),
                )
            )
        self.comp = new_comp

    def gate_loss_on_batch(self, batch: EpochBatch, resp: torch.Tensor) -> torch.Tensor:
        b = self._move_batch(batch)
        resp = resp.to(device=b.h.device, dtype=b.h.dtype)

        elogpi = self.conditional_expected_log_pi(b.h)

        # Negative ELBO stick-assignment term.
        fit = -(resp.detach() * elogpi).sum(dim=1).mean()

        if self.K == 1:
            return fit

        qh = self.gate.forward(b.h)
        a_h, b_h = qh["a"], qh["b"]

        prior_a = torch.ones_like(a_h)
        prior_b = torch.full_like(b_h, self.dp_alpha)

        kl = kumar2beta_kl(
            a_h,
            b_h,
            prior_a,
            prior_b,
            n_approx=10,
        ).sum(dim=1).mean()

        return fit + self.gate_kl_weight * kl

    def gate_refine(self, n_steps: int = 5, l2_weight: float = 1e-4):
        if n_steps <= 0 or self.K <= 1:
            return
        self.gate.train()
        for _ in range(int(n_steps)):
            order = list(range(len(self.batches)))
            random.shuffle(order)
            for bid in order:
                b = self.batches[bid]
                self.gate_optimizer.zero_grad(set_to_none=True)
                loss = self.gate_loss_on_batch(b, self.cache[bid].resp)
                # L2 penalty on gate weights
                if l2_weight > 0:
                    l2 = torch.zeros((), device=loss.device, dtype=loss.dtype)
                    for name, p in self.gate.named_parameters():
                        if p.requires_grad and p.ndim > 1:   # avoid biases/norm scalars
                            l2 = l2 + p.pow(2).sum()
                    loss = loss + l2_weight * l2

                if not loss.requires_grad:
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.gate.parameters(), max_norm=0.5, norm_type=2.0)
                self.gate_optimizer.step()
        self.gate.eval()

    @torch.no_grad()
    def update_one_batch(self, batch: EpochBatch):
        old = self.cache[batch.batch_id]
        if old.resp is not None:
            self._sub_cache(old)
        b = self._move_batch(batch)
        z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
        new_resp = self.local_step(b.h, z_fit, b.posterior_logvar)
        new_cache = self._cache_from_resp(z_fit, new_resp, posterior_logvar=b.posterior_logvar)
        self.cache[batch.batch_id] = new_cache
        self._add_cache(new_cache)
        self._update_components_from_summaries()

    def memoized_lap(self, gate_refine_steps_after_lap: int = 5):
        order = list(range(len(self.batches)))
        random.shuffle(order)
        for bid in order:
            self.update_one_batch(self.batches[bid])
        self.gate_refine(gate_refine_steps_after_lap)

    def memoized_lap_gate_frozen(self):
        """
        Used during speculative structure proposals.
        """
        order = list(range(len(self.batches)))
        random.shuffle(order)
        for bid in order:
            self.update_one_batch(self.batches[bid])

    #  objective
    def _expected_log_lik_component(self, k: int) -> torch.Tensor:
        return self._expected_log_lik_from_params(
            self.comp[k].mean, self.comp[k].kappa, self.comp[k].alpha, self.comp[k].beta,
            self.Nk[k], self.S1[k], self.S2[k]
        )

    def _expected_log_lik_from_params(
        self,
        mean: torch.Tensor,
        kappa: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        Nk: torch.Tensor,
        S1: torch.Tensor,
        S2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregated E_q[log p(z | phi)] from sufficient statistics under a
        Normal-Gamma posterior over phi.
        """
        e_log_tau = torch.digamma(alpha) - torch.log(beta.clamp_min(torch.finfo(beta.dtype).eps))
        e_tau = alpha / beta.clamp_min(torch.finfo(beta.dtype).eps)
        residual_quad = (S2 - 2.0 * mean * S1 + Nk * mean.pow(2)).clamp_min(0.0)
        return 0.5 * (Nk * (e_log_tau - LOG2PI - (1.0 / kappa.clamp_min(torch.finfo(kappa.dtype).eps))) - e_tau * residual_quad).sum()
        
    @torch.no_grad()
    def global_objective(self) -> torch.Tensor:
        """
        Conditional amortized ELBO over the current replay buffer.

        This is not the exact Hughes global-stick ELBO anymore, because
        q(v_k | h_n) is conditional/amortized. But it is the correct objective
        for validating birth/merge/delete under the current conditional gate.
        """
        data_term = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        kl_gate = torch.tensor(0.0, device=self.device_, dtype=self.dtype)

        for b_cpu in self.batches:
            b = self._move_batch(b_cpu)
            resp = self.cache[b.batch_id].resp.to(device=b.h.device, dtype=b.h.dtype)

            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z

            elogpi = self.conditional_expected_log_pi(b.h)
            eloglik = self.component_expected_log_likelihood(
                z_fit,
                posterior_logvar=b.posterior_logvar,
            )

            data_term = data_term + (resp * (elogpi + eloglik)).sum()

            if self.K > 1:
                qh = self.gate.forward(b.h)
                a_h, b_h = qh["a"], qh["b"]

                prior_a = torch.ones_like(a_h)
                prior_b = torch.full_like(b_h, self.dp_alpha)

                kl_gate = kl_gate + kumar2beta_kl(
                    a_h,
                    b_h,
                    prior_a,
                    prior_b,
                    n_approx=10,
                ).sum()

        m0 = self._prior_m0()
        beta0 = self._prior_beta0()

        kl_phi = torch.tensor(0.0, device=self.device_, dtype=self.dtype)

        for c in self.comp:
            kl_phi = kl_phi + normal_gamma_kl(
                c.mean,
                c.kappa,
                c.alpha,
                c.beta,
                m0,
                self.prior_kappa0,
                self.prior_alpha0,
                beta0,
            ).sum()

        return data_term + self.entropy - kl_phi - self.gate_kl_weight * kl_gate

    #  analytics for structure
    def _component_dispersion(self, k: int) -> torch.Tensor:
        Nk = self.Nk[k].clamp_min(torch.finfo(self.Nk.dtype).eps)
        return (self.S2[k] - (self.S1[k] * self.S1[k]) / Nk).mean()

    def _component_avg_entropy(self, k: int) -> torch.Tensor:
        num = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        den = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        for bid, _ in enumerate(self.batches):
            rk = self.cache[bid].resp[:, k]
            ent = -(self.cache[bid].resp.clamp_min(torch.finfo(self.cache[bid].resp.dtype).eps) * self.cache[bid].resp.clamp_min(torch.finfo(self.cache[bid].resp.dtype).eps).log()).sum(dim=1)
            num = num + (rk * ent).sum()
            den = den + rk.sum()
        return num / den.clamp_min(torch.finfo(den.dtype).eps)

    def choose_birth_target(self) -> int:
        scores = []
        for k in range(self.K):
            if self.Nk[k] < 2.0:
                scores.append(torch.tensor(-torch.finfo(torch.float32).eps, device=self.device_, dtype=self.dtype))
            else:
                scores.append(self._component_dispersion(k) + 0.25 * self._component_avg_entropy(k))
        return int(torch.argmax(torch.stack(scores)).item())

    def _merged_posterior_from_stats(self, Nk: torch.Tensor, S1: torch.Tensor, S2: torch.Tensor):
        m0 = self._prior_m0()
        beta0 = self._prior_beta0()
        Nk_ = Nk.clamp_min(torch.finfo(Nk.dtype).eps)
        mean_emp = S1 / Nk_
        sq_centered = (S2 - (S1 * S1) / Nk_).clamp_min(0.0)
        kappa = self.prior_kappa0 + Nk_
        mean = (self.prior_kappa0 * m0 + S1) / kappa
        alpha = self.prior_alpha0 + 0.5 * Nk_
        beta = (beta0 + 0.5 * sq_centered + 0.5 * (self.prior_kappa0 * Nk_ / kappa) * (mean_emp - m0).pow(2)).clamp_min(torch.finfo(beta0.dtype).eps)
        return mean, kappa, alpha, beta

    def _merge_entropy_delta(self, kA: int, kB: int) -> torch.Tensor:
        """
        Exact entropy delta under fixed responsibilities.
        """
        delta = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        for bid in range(len(self.batches)):
            rA = self.cache[bid].resp[:, kA]
            rB = self.cache[bid].resp[:, kB]
            merged = (rA + rB).clamp_min(torch.finfo(rA.dtype).eps)
            old = -(rA.clamp_min(torch.finfo(rA.dtype).eps) * rA.clamp_min(torch.finfo(rA.dtype).eps).log() + rB.clamp_min(torch.finfo(rB.dtype).eps) * rB.clamp_min(torch.finfo(rB.dtype).eps).log()).sum()
            new = -(merged * merged.log()).sum()
            delta = delta + (new - old)
        return delta

    def merge_score_analytic(self, kA: int, kB: int) -> float:
        Nk_m = self.Nk[kA] + self.Nk[kB]
        S1_m = self.S1[kA] + self.S1[kB]
        S2_m = self.S2[kA] + self.S2[kB]
        mean_m, kappa_m, alpha_m, beta_m = self._merged_posterior_from_stats(Nk_m, S1_m, S2_m)

        ell_m = self._expected_log_lik_from_params(mean_m, kappa_m, alpha_m, beta_m, Nk_m, S1_m, S2_m)
        ell_sep = self._expected_log_lik_component(kA) + self._expected_log_lik_component(kB)

        m0 = self._prior_m0()
        beta0 = self._prior_beta0()
        kl_m = normal_gamma_kl(mean_m, kappa_m, alpha_m, beta_m, m0, self.prior_kappa0, self.prior_alpha0, beta0).sum()
        kl_sep = (
            normal_gamma_kl(self.comp[kA].mean, self.comp[kA].kappa, self.comp[kA].alpha, self.comp[kA].beta, m0, self.prior_kappa0, self.prior_alpha0, beta0).sum()
            + normal_gamma_kl(self.comp[kB].mean, self.comp[kB].kappa, self.comp[kB].alpha, self.comp[kB].beta, m0, self.prior_kappa0, self.prior_alpha0, beta0).sum()
        )

        ent_delta = self._merge_entropy_delta(kA, kB)
        return float((ell_m - ell_sep - kl_m + kl_sep + ent_delta).item())

    # birth init
    def _collect_birth_subset(self, target_k: int):
        zs, lvs = [], []
        have_all_logvars = True
        remaining = int(self.birth_subset_max)

        order = list(range(len(self.batches)))
        random.shuffle(order)

        for bid in order:
            if remaining <= 0:
                break

            # responsibilities already depend on h_t through local_step/cache
            rk = self.cache[bid].resp[:, target_k]
            keep_idx = torch.nonzero(
                rk > self.birth_resp_threshold,
                as_tuple=False,
            ).flatten()

            if keep_idx.numel() == 0:
                continue

            # IMPORTANT: subsample BEFORE moving/collecting tensors
            if keep_idx.numel() > remaining:
                perm = torch.randperm(
                    keep_idx.numel(),
                    device=keep_idx.device,
                )[:remaining]
                keep_idx = keep_idx.index_select(0, perm)

            b = self._move_batch(self.batches[bid])
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z

            zs.append(z_fit.index_select(0, keep_idx).detach().cpu())

            if b.posterior_logvar is not None:
                lvs.append(b.posterior_logvar.index_select(0, keep_idx).detach().cpu())
            else:
                have_all_logvars = False

            remaining -= int(keep_idx.numel())

            del b, z_fit

        if len(zs) == 0:
            z_empty = torch.empty(
                (0, self.C, self.H, self.W),
                dtype=self.dtype,
            )
            return z_empty, None

        z_sub = torch.cat(zs, dim=0).contiguous()

        lv_sub = None
        if have_all_logvars and len(lvs) == len(zs):
            lv_sub = torch.cat(lvs, dim=0).contiguous()
        return z_sub, lv_sub

    #  structural edits
    @torch.no_grad()
    def _rebuild_all_caches_from_current_model(self):
        self._reset_global_summaries()
        for b_cpu in self.batches:
            b = self._move_batch(b_cpu)
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            c = self._cache_from_resp(z_fit, self.local_step(b.h, z_fit, b.posterior_logvar), posterior_logvar=b.posterior_logvar)
            self.cache[b.batch_id] = c
            self._add_cache(c)
        self._update_components_from_summaries()

    def _append_components(self, fresh: List[TensorDiagComponentPosterior]):
        old_K = self.K
        Kadd = min(len(fresh), self.max_components - self.K)
        if Kadd <= 0:
            return
        self.comp.extend(copy.deepcopy(fresh[:Kadd]))
        self.K += Kadd
        self.gate.init_new_rows(old_K, self.K, dp_alpha=self.dp_alpha, scale=1e-2)

        self.gate.set_active_K(self.K)

    def _reset_gate_optimizer(self):
        self.gate_optimizer = torch.optim.Adam(
            self.gate.parameters(),
            lr=self.gate_lr,
        )

    @torch.no_grad()
    def _drop_gate_row(self, k_drop: int):
        if self.K <= 1:
            return

        old_rows = self.K - 1
        if old_rows <= 0:
            return

        # If deleting final component, drop the final active stick row.
        row_drop = min(int(k_drop), old_rows - 1)

        for layer in [self.gate.out_a, self.gate.out_b]:
            if row_drop < old_rows - 1:
                layer.weight[row_drop:old_rows - 1].copy_(
                    layer.weight[row_drop + 1:old_rows].clone()
                )
                layer.bias[row_drop:old_rows - 1].copy_(
                    layer.bias[row_drop + 1:old_rows].clone()
                )

            layer.weight[old_rows - 1].zero_()
            layer.bias[old_rows - 1].zero_()

        self._reset_gate_optimizer()
    
    @torch.no_grad()
    def _merge_structure(self, kA: int, kB: int):
        if kB < kA:
            kA, kB = kB, kA
        for bid, _ in enumerate(self.batches):
            resp = self.cache[bid].resp
            prop = torch.cat([resp[:, :kB], resp[:, kB + 1:]], dim=1)
            prop[:, kA] = resp[:, kA] + resp[:, kB]
            prop = prop / prop.sum(dim=1, keepdim=True).clamp_min(torch.finfo(resp.dtype).eps)
            self.cache[bid].resp = prop
        del self.comp[kB]
        self._drop_gate_row(kB)
        self.K -= 1
        self.gate.set_active_K(self.K)
        self._reset_global_summaries()
        for bid, b_cpu in enumerate(self.batches):
            b = self._move_batch(b_cpu)
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            c = self._cache_from_resp(z_fit, self.cache[bid].resp, posterior_logvar=b.posterior_logvar)
            self.cache[bid] = c
            self._add_cache(c)
        self._update_components_from_summaries()

    @torch.no_grad()
    def _delete_structure(self, kdel: int):
        for bid, _ in enumerate(self.batches):
            resp = self.cache[bid].resp
            prop = torch.cat([resp[:, :kdel], resp[:, kdel + 1:]], dim=1)
            prop = prop / prop.sum(dim=1, keepdim=True).clamp_min(torch.finfo(resp.dtype).eps)
            self.cache[bid].resp = prop
        del self.comp[kdel]
        self._drop_gate_row(kdel)
        self.K -= 1
        self.gate.set_active_K(self.K)
        self._reset_global_summaries()
        for bid, b_cpu in enumerate(self.batches):
            b = self._move_batch(b_cpu)
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            c = self._cache_from_resp(z_fit, self.cache[bid].resp, posterior_logvar=b.posterior_logvar)
            self.cache[bid] = c
            self._add_cache(c)
        self._update_components_from_summaries()

    @torch.no_grad()
    def fit_flat_dpgmm_on_subsample(
        self,
        z_sub: torch.Tensor,
        posterior_logvar_sub: Optional[torch.Tensor] = None,
        K_new: int = 10,
        n_iters: int = 50,
    ) -> List[TensorDiagComponentPosterior]:
        """
        birth creation phase.

        This fits an auxiliary flat DP-GMM on the targeted subsample.
        The auxiliary model uses temporary Beta stick variational parameters,
        as in Hughes 2013, but these are NOT the main model posterior.

        Main model posterior remains:
            q(v_k | h_t) = Kumaraswamy(a_k(h_t), b_k(h_t)).
        """

        N = int(z_sub.shape[0])
        if N < max(4, K_new):
            return []

        K = int(max(1, min(K_new, N, self.max_components - self.K)))
        if K <= 0:
            return []

        x = z_sub.reshape(N, -1).to(device=self.device_, dtype=self.dtype)
        Dflat = x.shape[1]

        if posterior_logvar_sub is not None:
            x_var = torch.exp(
                posterior_logvar_sub.reshape(N, -1).to(device=self.device_, dtype=self.dtype)
            ).clamp_min(1e-8)
        else:
            x_var = torch.zeros_like(x)

        m0 = self._prior_m0().reshape(-1)
        beta0 = self._prior_beta0().reshape(-1)

        # Initialize fresh components with your existing kmeans++ helper.
        comps = self._kmeanspp_initialize_components(z_sub, K)
        q_m = torch.stack([c.mean.reshape(-1) for c in comps], dim=0)
        q_kappa = torch.full(
            (K, 1),
            float(self.prior_kappa0 + 1.0),
            device=self.device_,
            dtype=self.dtype,
        )
        q_alpha = torch.full(
            (K, 1),
            float(self.prior_alpha0 + 0.5),
            device=self.device_,
            dtype=self.dtype,
        )
        q_beta = torch.stack(
            [
                c.beta.reshape(-1).clamp_min(1e-6)
                for c in comps
            ],
            dim=0,
        )

        # Temporary auxiliary Beta posterior for Hughes-style birth only.
        # These are not used as the main model's posterior.
        if K > 1:
            aux_a1 = torch.ones(K - 1, device=self.device_, dtype=self.dtype)
            aux_a0 = torch.full((K - 1,), self.dp_alpha, device=self.device_, dtype=self.dtype)

        for _ in range(int(n_iters)):

            # Expected log stick weights.
            if K == 1:
                elogpi = torch.zeros(1, device=self.device_, dtype=self.dtype)
            else:
                e_log_v = torch.digamma(aux_a1) - torch.digamma(aux_a1 + aux_a0)
                e_log_1mv = torch.digamma(aux_a0) - torch.digamma(aux_a1 + aux_a0)

                elogpi = torch.empty(K, device=self.device_, dtype=self.dtype)
                prefix = torch.tensor(0.0, device=self.device_, dtype=self.dtype)

                for k in range(K):
                    if k < K - 1:
                        elogpi[k] = e_log_v[k] + prefix
                        prefix = prefix + e_log_1mv[k]
                    else:
                        elogpi[k] = prefix

            # Expected Gaussian log-likelihood under Normal-Gamma posterior.
            eps = torch.finfo(self.dtype).eps
            e_log_tau = torch.digamma(q_alpha) - torch.log(q_beta.clamp_min(eps))
            e_tau = q_alpha / q_beta.clamp_min(eps)

            expected_quad = (x[:, None, :] - q_m[None, :, :]).pow(2) + x_var[:, None, :]

            log_comp = 0.5 * (
                e_log_tau[None, :, :]
                - LOG2PI
                - 1.0 / q_kappa[None, :, :].clamp_min(eps)
                - e_tau[None, :, :] * expected_quad
            ).sum(dim=-1)

            resp = torch.softmax(elogpi[None, :] + log_comp, dim=-1)

            # Hughes sufficient statistics.
            Nk_raw = resp.sum(dim=0)
            Nk = Nk_raw.clamp_min(1e-6)

            S1 = resp.t().matmul(x)
            S2 = resp.t().matmul(x * x + x_var)

            mean_emp = S1 / Nk[:, None]
            sq_centered = (S2 - S1.pow(2) / Nk[:, None]).clamp_min(0.0)

            q_kappa = self.prior_kappa0 + Nk[:, None]
            q_m = (self.prior_kappa0 * m0[None, :] + S1) / q_kappa
            q_alpha = self.prior_alpha0 + 0.5 * Nk[:, None]

            q_beta = (
                beta0[None, :]
                + 0.5 * sq_centered
                + 0.5
                * (self.prior_kappa0 * Nk[:, None] / q_kappa)
                * (mean_emp - m0[None, :]).pow(2)
            ).clamp_min(eps)

            # Hughes DP stick update for the auxiliary proposer only.
            if K > 1:
                tail = torch.zeros(K - 1, device=self.device_, dtype=self.dtype)
                running = torch.tensor(0.0, device=self.device_, dtype=self.dtype)

                for k in reversed(range(K - 1)):
                    running = running + Nk_raw[k + 1]
                    tail[k] = running

                aux_a1 = 1.0 + Nk_raw[:-1]
                aux_a0 = self.dp_alpha + tail

        fresh = []
        for k in range(K):
            if Nk_raw[k] < 2.0:
                continue

            fresh.append(
                TensorDiagComponentPosterior(
                    mean=q_m[k].reshape(self.z_shape).detach(),
                    kappa=q_kappa[k, 0].detach(),
                    alpha=q_alpha[k, 0].detach(),
                    beta=q_beta[k].reshape(self.z_shape).detach(),
                )
            )

        return fresh
    # speculative proposals
    def propose_birth(
        self,
        Kfresh: Optional[int] = None,
        adopt_laps: int = 1,
        gate_refine_steps_accept: int = 32,
        gate_refine_steps_speculate: int = 4,
    ) -> bool:
        if self.K >= self.max_components:
            return False

        ckpt = self.save_checkpoint()
        old_obj = float(self.global_objective().item())

        target_k = self.choose_birth_target()
        z_sub, lv_sub = self._collect_birth_subset(target_k)

        Kfresh = int(Kfresh or self.birth_kfresh)
        Kfresh = min(Kfresh, self.max_components - self.K)

        if z_sub.shape[0] < max(4, Kfresh):
            return False

        fresh = self.fit_flat_dpgmm_on_subsample(
            z_sub=z_sub,
            posterior_logvar_sub=lv_sub,
            K_new=Kfresh,
            n_iters=50,
        )

        if len(fresh) == 0:
            return False

        self._append_components(fresh)
        self._rebuild_all_caches_from_current_model()

        for _ in range(int(adopt_laps)):
            self.memoized_lap_gate_frozen()

        if gate_refine_steps_accept > 0 and self.K > 1:
            self.gate_refine(gate_refine_steps_accept)
            self._rebuild_all_caches_from_current_model()

        new_obj = float(self.global_objective().item())

        if new_obj > old_obj:
            return True

        self.restore_checkpoint(ckpt)
        return False

    def candidate_merge_pairs(self, top_pairs: Optional[int] = None) -> List[Tuple[int, int]]:
        top_pairs = int(top_pairs or self.merge_top_pairs)
        if self.K <= 1:
            return []
        scored = []
        for i in range(self.K):
            for j in range(i + 1, self.K):
                scored.append((self.merge_score_analytic(i, j), i, j))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(i, j) for _, i, j in scored[:top_pairs]]

    def propose_merge(
        self,
        kA: int,
        kB: int,
        refine_laps: int = 1,
        gate_refine_steps_accept: int = 8,
    ) -> bool:
        ckpt = self.save_checkpoint()
        old_obj = float(self.global_objective().item())

        # 1. Construct Hughes-style candidate:
        #    pool responsibilities of kA and kB, remove kB.
        self._merge_structure(kA, kB)

        # 2. Global update for component parameters.
        for _ in range(refine_laps):
            self.memoized_lap_gate_frozen()

        # 3. Global update for conditional stick posterior q(v | h).
        #    This is the conditional-Kumaraswamy analogue of Hughes' q(u) update.
        if self.K > 1 and gate_refine_steps_accept > 0:
            self.gate_refine(gate_refine_steps_accept)
            self._rebuild_all_caches_from_current_model()

        # 4. Now evaluate candidate objective.
        new_obj = float(self.global_objective().item())

        if new_obj > old_obj:
            return True

        self.restore_checkpoint(ckpt)
        return False

    def try_best_merge(self, refine_laps: int = 1, gate_refine_steps: int = 8) -> bool:
        cur_obj = float(self.global_objective().item())
        pairs = self.candidate_merge_pairs()

        if len(pairs) == 0:
            return False

        best_gain, best_pair = -float("inf"), None

        for i, j in pairs[: self.merge_screen_topk]:
            ckpt = self.save_checkpoint()

            self._merge_structure(i, j)

            for _ in range(refine_laps):
                self.memoized_lap_gate_frozen()

            if self.K > 1 and gate_refine_steps > 0:
                self.gate_refine(gate_refine_steps)
                self._rebuild_all_caches_from_current_model()

            gain = float(self.global_objective().item()) - cur_obj

            self.restore_checkpoint(ckpt)

            if gain > best_gain:
                best_gain, best_pair = gain, (i, j)

        if best_pair is None or best_gain <= 0.0:
            return False

        return self.propose_merge(
            best_pair[0],
            best_pair[1],
            refine_laps=refine_laps,
            gate_refine_steps_accept=gate_refine_steps,
        )
    def candidate_delete_components(self, resp_thresh =0.01) -> List[int]:
        out = []
        for k in range(self.K):
            if self.Nk[k].item() < self.delete_min_count:
                out.append(k)
                continue

            seq_chunks = []
            has_valid_seq_meta = False
            for bid, b in enumerate(self.batches):
                if b.seq_id is None:
                    continue
                has_valid_seq_meta = True
                rk = self.cache[bid].resp[:, k]
                keep = (rk > resp_thresh).detach().cpu()
                if keep.any():
                    seq_k = b.seq_id[keep]
                    sek_k = seq_k[seq_k >= 0]
                    if sek_k.numel() > 0:
                        seq_chunks.append(sek_k)
            if has_valid_seq_meta:
                if len(seq_chunks) == 0:
                    support = 0
                else:
                    support = int(torch.unique(torch.cat(seq_chunks, dim=0)).numel())

                if support < self.delete_min_seqs:
                    out.append(k)
        return out

    def propose_delete(
        self,
        kdel: int,
        refine_laps: int = 2,
        gate_refine_steps_accept: int = 8,
    ) -> bool:
        if self.K <= 1:
            return False

        ckpt = self.save_checkpoint()
        old_obj = float(self.global_objective().item())

        # 1. Construct delete candidate.
        self._delete_structure(kdel)

        # 2. Global component updates.
        for _ in range(refine_laps):
            self.memoized_lap_gate_frozen()

        # 3. Global conditional-stick update.
        if self.K > 1 and gate_refine_steps_accept > 0:
            self.gate_refine(gate_refine_steps_accept)
            self._rebuild_all_caches_from_current_model()

        # 4. Accept only if objective improves.
        new_obj = float(self.global_objective().item())

        if new_obj > old_obj:
            return True

        self.restore_checkpoint(ckpt)
        return False

    def try_best_delete(self, refine_laps: int = 2, gate_refine_steps: int = 8) -> bool:
        cur_obj = float(self.global_objective().item())

        best_gain, best_k = -float("inf"), None

        for k in self.candidate_delete_components():
            ckpt = self.save_checkpoint()

            self._delete_structure(k)

            for _ in range(refine_laps):
                self.memoized_lap_gate_frozen()

            if self.K > 1 and gate_refine_steps > 0:
                self.gate_refine(gate_refine_steps)
                self._rebuild_all_caches_from_current_model()

            gain = float(self.global_objective().item()) - cur_obj

            self.restore_checkpoint(ckpt)

            if gain > best_gain:
                best_gain, best_k = gain, k

        if best_k is None or best_gain <= 0.0:
            return False

        return self.propose_delete(
            best_k,
            refine_laps=refine_laps,
            gate_refine_steps_accept=gate_refine_steps,
        )

    #  fit epoch / snapshot
    def fit_epoch(
        self,
        buffer: ReplayBufferConditional,
        batch_size: int,
        init_K: Optional[int] = None,
        n_laps: int = 4,
        do_birth: bool = True,
        do_merge: bool = True,
        do_delete: bool = True,
        birth_every: int = 1,
        merge_every: int = 1,
        delete_every: int = 1,
        shuffle_batches: bool = True,
        warm_start: bool = True,
    ) -> FrozenConditionalTopPrior:

        batches = buffer.freeze_epoch(batch_size=batch_size, shuffle=shuffle_batches)
        
        print("[buffer debug] h:", tuple(batches[0].h.shape))
        print("[buffer debug] z:", tuple(batches[0].z.shape))
        print("[buffer debug] mu:", None if batches[0].posterior_mean is None else tuple(batches[0].posterior_mean.shape))
        print("[buffer debug] seq:", tuple(batches[0].seq_id.shape))
        print("[buffer debug] t:", tuple(batches[0].t_index.shape))

        assert batches[0].h.shape[0] == batches[0].z.shape[0]
        assert batches[0].h.dim() == 4
        assert batches[0].z.dim() == 4

        self.initialize_from_data(batches,K_init=init_K or self.K,warm_start=warm_start)

        for lap in range(1, n_laps + 1):
            self.memoized_lap(gate_refine_steps_after_lap=10)

            if do_birth and birth_every > 0 and lap % birth_every == 0 and self.K < self.max_components:
                self.propose_birth()

            if do_merge and merge_every > 0 and lap % merge_every == 0 and self.K > 1:
                improved = True
                while improved:
                    improved = self.try_best_merge(refine_laps=1)

            if do_delete and delete_every > 0 and lap % delete_every == 0 and self.K > 1:
                improved = True
                while improved:
                    improved = self.try_best_delete(refine_laps=2)

        # Important: make snapshot BEFORE clearing fit cache
        snapshot = self.frozen_snapshot()

        # Clean large internal DPGMM fitting state
        self.clear_fit_cache()

        # Clean local replay batch list
        del batches

        # Release Python/CUDA cached memory
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return snapshot

    @torch.no_grad()
    def frozen_snapshot(self) -> FrozenConditionalTopPrior:
        mean = torch.stack([c.mean for c in self.comp], dim=0)

        vars_pred = []
        eps = torch.finfo(self.dtype).eps

        for c in self.comp:
            alpha = c.alpha.clamp_min(1.0 + 1e-4)

            kappa = c.kappa
            if not torch.is_tensor(kappa):
                kappa = torch.as_tensor(kappa, device=self.device_, dtype=self.dtype)
            kappa = kappa.to(device=self.device_, dtype=self.dtype).clamp_min(eps)

            var_k = c.beta / (alpha - 1.0)
            var_k = var_k * (1.0 + 1.0 / kappa)

            vars_pred.append(var_k.clamp_min(1e-6))

        var = torch.stack(vars_pred, dim=0)

        return FrozenConditionalTopPrior(
            comp_mean=mean.detach().clone().cpu().contiguous(),
            comp_var=var.detach().clone().cpu().contiguous(),
            gate_state={k: v.detach().clone().cpu() for k, v in self.gate.state_dict().items()},
            K=self.K,
            max_K=self.max_components,
            objective=float(self.global_objective().detach().cpu().item()),
        )

    @staticmethod
    def load_gate_from_snapshot(snapshot: FrozenConditionalTopPrior, h_shape: Tuple[int, ...], hidden_dim: int, device: torch.device, dtype: torch.dtype):
        gate = ConditionalKumaraswamyGate(h_shape, snapshot.max_K, hidden_dim=hidden_dim).to(device=device, dtype=dtype)
        gate.load_state_dict(snapshot.gate_state, strict=True)
        gate.set_active_K(snapshot.K)
        gate.eval()
        for p in gate.parameters():
            p.requires_grad_(False)
        return gate

    @staticmethod
    def conditional_expected_log_pi_frozen( frozen_gate: ConditionalKumaraswamyGate, h_t: torch.Tensor) -> torch.Tensor:

        return frozen_gate.expected_log_pi(h_t)


    @staticmethod
    def kl_mc_to_frozen_prior(
        snapshot: FrozenConditionalTopPrior,
        frozen_gate: ConditionalKumaraswamyGate,
        h_t: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        n_samples: int = 8,
    ) -> torch.Tensor:
        """
        Differentiable MC estimate of KL[q(z_top|x,h) || p_frozen(z_top|h)].

        Gradients should flow to posterior_mean, posterior_logvar, and h_t.
        Gradients should not update frozen prior parameters, because frozen_gate has
        requires_grad=False and snapshot tensors are detached clones.
        """
        S = int(max(1, n_samples))
        B = posterior_mean.shape[0]
        device = posterior_mean.device
        dtype = posterior_mean.dtype

        means = snapshot.comp_mean.to(device=device, dtype=dtype)                  # [K,C,H,W]
        vars_ = snapshot.comp_var.to(device=device, dtype=dtype).clamp_min(1e-6)   # [K,C,H,W]

        # This is differentiable w.r.t. h_t, but not w.r.t. frozen_gate parameters.
        elogpi = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
            frozen_gate, h_t
        )  # [B,K]

        q_logvar = posterior_logvar.clamp(min=-7.0, max=5.0)
        q_var = torch.exp(q_logvar).clamp_min(1e-6)
        q_std = torch.sqrt(q_var)

        eps = torch.randn(S, *posterior_mean.shape, device=device, dtype=dtype )

        z = posterior_mean.unsqueeze(0) + q_std.unsqueeze(0) * eps  # [S,B,C,H,W]

        log_q = (
            -0.5
            * (
                q_logvar.unsqueeze(0)
                + LOG2PI
                + (z - posterior_mean.unsqueeze(0)).pow(2) / q_var.unsqueeze(0)
            )
        ).flatten(2).sum(dim=2)  # [S,B]

        z_exp = z.unsqueeze(2)                         # [S,B,1,C,H,W]
        means_exp = means.view(1, 1, *means.shape)     # [1,1,K,C,H,W]
        vars_exp = vars_.view(1, 1, *vars_.shape)      # [1,1,K,C,H,W]

        comp_log = (
            -0.5
            * (
                torch.log(vars_exp)
                + LOG2PI
                + (z_exp - means_exp).pow(2) / vars_exp
            )
        ).flatten(3).sum(dim=3)  # [S,B,K]

        log_pi = torch.log_softmax(elogpi, dim=-1)
        log_p = stable_logsumexp(comp_log + log_pi.unsqueeze(0), dim=2)

        return (log_q - log_p).mean(dim=0)  # [B]

    @staticmethod
    @torch.no_grad()
    def sample_from_frozen_prior(snapshot: FrozenConditionalTopPrior, frozen_gate: ConditionalKumaraswamyGate, h_t: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # Use the SAME conditional mixture weights as the KL path
        elogpi = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
            frozen_gate=frozen_gate,
            h_t=h_t,
        )  # [B, K]

        pi = torch.softmax(elogpi/ max(float(temperature), 1e-4), dim=1)  # [B, K]
        idx = torch.multinomial(pi, 1).squeeze(1)  # [B]

        means = snapshot.comp_mean.to(device=h_t.device, dtype=h_t.dtype).index_select(0, idx)
        vars_ = snapshot.comp_var.to(device=h_t.device, dtype=h_t.dtype).index_select(0, idx).clamp_min(1e-6)

        return means +  float(temperature) * torch.sqrt(vars_) * torch.randn_like(means)

    def _to_device(self, t):
        if t is None:
            return None
        if t.is_floating_point():
            return t.to(self.device_, dtype=self.dtype, non_blocking=True)
        return t.to(self.device_, non_blocking=True)

    def _move_batch(self, b):
        return EpochBatch(
            batch_id=b.batch_id,
            h=self._to_device(b.h),
            z=self._to_device(b.z),
            posterior_mean=self._to_device(b.posterior_mean),
            posterior_logvar=self._to_device(b.posterior_logvar),
            valid_mask=self._to_device(b.valid_mask),
            seq_id=self._to_device(b.seq_id),
            t_index=self._to_device(b.t_index),
        )

def compute_top_kl_conditional_frozen(snapshot, frozen_gate, h_t, top_q_mean_map, top_q_logvar_map, n_samples: int = 8):
    return ConditionalTopDPGMM.kl_mc_to_frozen_prior(snapshot, frozen_gate, h_t, top_q_mean_map, top_q_logvar_map, n_samples=n_samples)


def sample_top_conditional_frozen(snapshot, frozen_gate, h_t, temperature: float = 1.0):
    return ConditionalTopDPGMM.sample_from_frozen_prior(snapshot, frozen_gate, h_t, temperature=temperature)

def gaussian_kl_diag_to_components(
    q_mu: torch.Tensor,
    q_logsigma: torch.Tensor,
    comp_mean: torch.Tensor,
    comp_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    q_mu:        [B, S, D]
    q_logsigma:  [B, S, D]
    comp_mean:   [M, D]
    comp_var:    [M, D]

    returns:
        KL[q(s_k) || component_j] with shape [B, S, M]
    """
    q_logvar = 2.0 * q_logsigma.clamp(min=-7.0, max=4.0)
    q_var = torch.exp(q_logvar).clamp_min(eps)

    comp_var = comp_var.clamp_min(eps)
    comp_logvar = torch.log(comp_var)

    qm = q_mu[:, :, None, :]
    qv = q_var[:, :, None, :]
    qlv = q_logvar[:, :, None, :]

    cm = comp_mean[None, None, :, :]
    cv = comp_var[None, None, :, :]
    clv = comp_logvar[None, None, :, :]

    kl = 0.5 * ( clv - qlv - 1.0 + (qv + (qm - cm).pow(2)) / cv)
    return kl.sum(dim=-1)

def compute_slot_kl_conditional_frozen(
    snapshot,
    frozen_gate,
    h_t: torch.Tensor,
    slot_mu: torch.Tensor,
    slot_logsigma: torch.Tensor,
    eps: float = torch.finfo(torch.float32).eps,
):
    """
    Computes augmented KL:

        KL[q(s, c | x, h) || p(s, c | h)]

    slot_mu:       [B, S, D]
    slot_logsigma: [B, S, D]

    returns:
        kl_per_image: [B]
        resp:         [B, S, M]
    """
    B, S, D = slot_mu.shape
    M = snapshot.K

    comp_mean = snapshot.comp_mean.to(
        device=slot_mu.device,
        dtype=slot_mu.dtype,
    ).view(M, -1)

    comp_var = snapshot.comp_var.to(
        device=slot_mu.device,
        dtype=slot_mu.dtype,
    ).view(M, -1).clamp_min(1e-6)

    # elogpi = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
    #     frozen_gate=frozen_gate,
    #     h_t=h_t,
    # )  # [B, M]
    # log_pi = torch.log_softmax(elogpi, dim=-1)
    pi = frozen_gate.mean_pi(h_t).clamp_min(eps)
    log_pi = torch.log(pi)

    kl_q_to_comp = gaussian_kl_diag_to_components(
        q_mu=slot_mu,
        q_logsigma=slot_logsigma,
        comp_mean=comp_mean,
        comp_var=comp_var,
        eps=eps
    )  # [B, S, M]

    log_r = log_pi[:, None, :] - kl_q_to_comp
    resp = torch.softmax(log_r, dim=-1)

    slot_kl = resp * (
        torch.log(resp.clamp_min(eps))
        - log_pi[:, None, :]
        + kl_q_to_comp
    )

    return slot_kl.sum(dim=(1, 2)), resp

def sample_slots_conditional_frozen(
    snapshot,
    frozen_gate,
    h_t: torch.Tensor,
    num_slots: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Samples slots independently from the conditional DPGMM.

    returns:
    mu and sigma
    """
    B = h_t.shape[0]
    M = snapshot.K

    # elogpi = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
    #     frozen_gate=frozen_gate,
    #     h_t=h_t,
    # )

    # temp = max(float(temperature), 1e-4)
    # pi = torch.softmax(elogpi / temp, dim=-1)
    pi = frozen_gate.mean_pi(h_t).clamp_min(torch.finfo(h_t.dtype).eps)

    temp = max(float(temperature), 1e-4)
    if temp != 1.0:
        pi = torch.softmax(torch.log(pi) / temp, dim=-1)
    else:
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(pi.dtype).eps)

    pi_rep = pi[:, None, :].expand(B, num_slots, M).reshape(B * num_slots, M)
    idx = torch.multinomial(pi_rep, 1).squeeze(-1)

    comp_mean = snapshot.comp_mean.to(h_t.device, h_t.dtype).view(M, -1)
    comp_var = snapshot.comp_var.to(h_t.device, h_t.dtype).view(M, -1).clamp_min(1e-6)

    mu = comp_mean.index_select(0, idx)
    var = comp_var.index_select(0, idx)

    return mu.view(B, num_slots, -1).contiguous(), var.view(B, num_slots, -1).contiguous()

