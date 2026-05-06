from __future__ import annotations
"""
1) The gate is PREALLOCATED at max_components and never reconstructs nn.Linear
   during birth/delete/merge. Active components are controlled by active_K.
   This preserves optimizer state and avoids parameter-registration bugs.

2) We only save/restore the mutable structural state needed for speculative moves.

3) Speculative birth/merge/delete proposals keep the gate FROZEN.
   The gate is refined only AFTER a proposal is accepted.

4) Merge candidates are ranked with an analytic surrogate score computed from
   sufficient statistics, then only the top few are evaluated exactly.

5) Birth initialization uses divisive splitting along the dominant residual
   variance direction instead of random seeding.

6) The design remains close to Hughes & Sudderth 2013 memoized VI:
   - memoized batches
   - cached sufficient statistics
   - nonlocal birth/merge/delete on a global component bank and accept or reject number of clusters

7) The prior remains conditional:
       p(z_t^top | h_t) = sum_k pi_k(h_t) N(z_t^top ; mu_k, Sigma_k)

   with global component bank {mu_k, Sigma_k} and a context-conditioned
   Kumaraswamy gate over weights pi_k(h_t).

Notes:
- z_t^top stays tensor-shaped [C,H,W].
- Replay buffer stores (h_t, z_t^top, posterior_mean, posterior_logvar, seq_id, t, mask).
- During the online epoch, use a frozen snapshot of this prior for KL.
- At epoch end, fit/update this model on replay-buffer posterior samples first, then freeze it for the next epoch.
"""
import copy
import random, gc, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


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
    def __init__(self):
        self._next_seq_id = 0
        self.clear()

    def clear(self):
        self._h, self._z = [], []
        self._mu, self._lv = [], []
        self._mask, self._seq, self._t = [], [], []

    def __len__(self) -> int:
        return sum(z.shape[0] for z in self._z)

    def is_empty(self) -> bool:
        return len(self) == 0

    @torch.no_grad()
    def allocate_seq_ids(self, batch_size: int, device=None) -> torch.Tensor:
        start = self._next_seq_id
        stop = start + int(batch_size)
        self._next_seq_id = stop
        out = torch.arange(start, stop, dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

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
        B = z_top_map.shape[0]
        if valid_mask is None:
            valid_mask = torch.ones(B, device=z_top_map.device, dtype=torch.bool)
        keep = torch.nonzero(valid_mask.bool(), as_tuple=False).flatten()
        if keep.numel() == 0:
            return

        def take(x):
            if x is None:
                return None
            return x.index_select(0, keep).detach().cpu()

        self._h.append(take(h_t))
        self._z.append(take(z_top_map))
        self._mu.append(take(posterior_mean))
        self._lv.append(take(posterior_logvar))
        self._mask.append(torch.ones(keep.numel(), dtype=torch.bool))
        self._seq.append(take(seq_id) if seq_id is not None else torch.full((keep.numel(),), -1, dtype=torch.long))
        self._t.append(take(t_index) if t_index is not None else torch.full((keep.numel(),), -1, dtype=torch.long))

    def freeze_epoch(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> List[EpochBatch]:
        if len(self._z) == 0:
            raise RuntimeError("Replay buffer is empty.")
        h = torch.cat(self._h, dim=0).contiguous()
        z = torch.cat(self._z, dim=0).contiguous()
        mu = None if len(self._mu) == 0 or self._mu[0] is None else torch.cat(self._mu, dim=0).contiguous()
        lv = None if len(self._lv) == 0 or self._lv[0] is None else torch.cat(self._lv, dim=0).contiguous()
        mask = torch.cat(self._mask, dim=0).contiguous()
        seq = torch.cat(self._seq, dim=0).contiguous()
        t = torch.cat(self._t, dim=0).contiguous()
        N = z.shape[0]
        perm = torch.randperm(N) if shuffle else torch.arange(N)
        h, z = h.index_select(0, perm), z.index_select(0, perm)
        if mu is not None:
            mu = mu.index_select(0, perm)
        if lv is not None:
            lv = lv.index_select(0, perm)
        mask, seq, t = mask.index_select(0, perm), seq.index_select(0, perm), t.index_select(0, perm)
        out: List[EpochBatch] = []
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            out.append(EpochBatch(len(out), h[s:e], z[s:e], posterior_mean=None if mu is None else mu[s:e], posterior_logvar=None if lv is None else lv[s:e], valid_mask=mask[s:e], seq_id=seq[s:e], t_index=t[s:e]))
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

    @torch.no_grad()
    def init_new_rows(self, old_K: int, new_K: int, scale: float = 1e-2):
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
        self.out_a.bias[sl].fill_(0.0)
        self.out_b.bias[sl].fill_(0.0)

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
    resp: Optional[torch.Tensor] = None
    Nk: Optional[torch.Tensor] = None
    S1: Optional[torch.Tensor] = None
    S2: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None


@dataclass
class FrozenConditionalTopPrior:
    comp_mean: torch.Tensor
    comp_var: torch.Tensor
    qv_a1: torch.Tensor
    qv_a0: torch.Tensor
    gate_state: Dict[str, torch.Tensor]
    K: int
    max_K: int
    objective: float


@dataclass
class StructuralCheckpoint:
    K: int
    comp: List[TensorDiagComponentPosterior]
    qv_a1: torch.Tensor
    qv_a0: torch.Tensor
    Nk: torch.Tensor
    S1: torch.Tensor
    S2: torch.Tensor
    entropy: torch.Tensor
    cache_resp: Dict[int, torch.Tensor]
    gate_state: Dict[str, torch.Tensor]


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
        prior_kappa0: float = 1.0,
        prior_alpha0: float = 2.0,
        prior_beta0: float = 1.0,
        gate_hidden_dim: int = 256,
        gate_lr: float = 1e-3,
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
        self.device_ = device if device is not None else torch.device('cpu')
        self.dtype = dtype

        self.gate = ConditionalKumaraswamyGate(h_shape=h_shape, max_K=self.max_components, hidden_dim=gate_hidden_dim).to(self.device_, self.dtype)
        self.gate.set_active_K(self.K)
        self.gate_optimizer = torch.optim.Adam(self.gate.parameters(), lr=gate_lr)

        self.qv_a1 = None
        self.qv_a0 = None
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

    def _update_global_sticks_from_counts(self, Nk: torch.Tensor):
        if self.K == 1:
            self.qv_a1 = torch.zeros(0, device=self.device_, dtype=self.dtype).detach()
            self.qv_a0 = torch.zeros(0, device=self.device_, dtype=self.dtype).detach()
            return
        tail = torch.zeros(self.K - 1, device=self.device_, dtype=self.dtype)
        running = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        for k in reversed(range(self.K - 1)):
            running = running + Nk[k + 1]
            tail[k] = running
        self.qv_a1 = (1.0 + Nk[:-1]).detach()
        self.qv_a0 = (self.dp_alpha + tail).detach()


    def _cache_from_resp(self, z: torch.Tensor, resp: torch.Tensor) -> MemoizedBatchCache:
        resp = resp.detach()
        z = z.detach()

        Nk = resp.sum(dim=0).detach()
        S1 = torch.einsum('nk,nchw->kchw', resp, z).detach()
        S2 = torch.einsum('nk,nchw->kchw', resp, z * z).detach()
        H = (-(resp.clamp_min(torch.finfo(resp.dtype).eps) * resp.clamp_min(torch.finfo(resp.dtype).eps).log()).sum()).detach()

        return MemoizedBatchCache(resp=resp,Nk=Nk,S1=S1,S2=S2,entropy=H)

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
            qv_a1=self.qv_a1.detach().clone(),
            qv_a0=self.qv_a0.detach().clone(),
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
        )

    @torch.no_grad()
    def restore_checkpoint(self, ckpt: StructuralCheckpoint):
        self.K = ckpt.K
        self.comp = [copy.deepcopy(c) for c in ckpt.comp]
        self.gate.load_state_dict(ckpt.gate_state, strict=True)
        self.gate.set_active_K(self.K)
        self.qv_a1 = ckpt.qv_a1.clone()
        self.qv_a0 = ckpt.qv_a0.clone()
        self._reset_global_summaries()
        for bid, b_cpu in enumerate(self.batches):
            b = self._move_batch(b_cpu)
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            c = self._cache_from_resp(z_fit, ckpt.cache_resp[bid])
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
            if self.qv_a1 is None or self.qv_a0 is None or self.qv_a1.numel() != max(0, self.K - 1):
                self._update_global_sticks_from_counts(torch.zeros(self.K, device=self.device_, dtype=self.dtype))
        else:
            self.comp = self._kmeanspp_initialize_components(z_all_cpu, self.K)
            self._update_global_sticks_from_counts(torch.zeros(self.K, device=self.device_, dtype=self.dtype))

        self._initialize_caches()

    @torch.no_grad()
    def _initialize_caches(self):
        self._reset_global_summaries()

        for b_cpu in self.batches:
            b = self._move_batch(b_cpu)

            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z

            resp = self.local_step(b.h, z_fit)
            c = self._cache_from_resp(z_fit, resp)

            self.cache[b.batch_id] = c
            self._add_cache(c)
        self._update_components_from_summaries()    

    # model expectations
    def expected_log_global_pi(self) -> torch.Tensor:
        if self.K == 1:
            return torch.zeros(1, device=self.device_, dtype=self.dtype)
        e_log_v = torch.digamma(self.qv_a1) - torch.digamma(self.qv_a1 + self.qv_a0)
        e_log_1mv = torch.digamma(self.qv_a0) - torch.digamma(self.qv_a1 + self.qv_a0)
        out = torch.empty(self.K, device=self.device_, dtype=self.dtype)
        prefix = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        for k in range(self.K):
            if k < self.K - 1:
                out[k] = e_log_v[k] + prefix
                prefix = prefix + e_log_1mv[k]
            else:
                out[k] = prefix
        return out

    def conditional_expected_log_pi(self, h: torch.Tensor) -> torch.Tensor:
        return self.expected_log_global_pi().view(1, self.K) + self.gate.expected_log_pi(h)

    def component_expected_log_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        N = z.shape[0]
        out = torch.empty(N, self.K, device=z.device, dtype=z.dtype)
        eps = torch.finfo(z.dtype).eps
        for k, comp in enumerate(self.comp):
            e_log_tau = torch.digamma(comp.alpha) - torch.log(comp.beta.clamp_min(eps))
            e_tau = comp.alpha / comp.beta.clamp_min(eps)
            term = 0.5 * (e_log_tau - LOG2PI - (1.0 / comp.kappa.clamp_min(eps)) - e_tau * (z - comp.mean).pow(2))
            out[:, k] = term.view(N, -1).sum(dim=1)
        return out

    @torch.no_grad()
    def local_step(self, h: torch.Tensor, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = (self.conditional_expected_log_pi(h) + self.component_expected_log_likelihood(z)) / max(float(temperature), torch.finfo(h.dtype).eps)
        return torch.softmax(logits, dim=1)

    # updates
    def _update_components_from_summaries(self):
        self._update_global_sticks_from_counts(self.Nk)
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
        elogpi = self.conditional_expected_log_pi(b.h)
        fit = -(resp.detach() * torch.log_softmax(elogpi, dim=1)).sum(dim=1).mean()
        if self.K == 1:
            return fit
        qh = self.gate.forward(b.h)
        a_h, b_h = qh['a'], qh['b']
        a_g = self.qv_a1.view(1, -1).expand_as(a_h).detach()
        b_g = self.qv_a0.view(1, -1).expand_as(b_h).detach()
        kl = kumar2beta_kl(a_h, b_h, a_g, b_g, n_approx=10).sum(dim=1).mean()
        return fit + self.gate_kl_weight * kl

    def gate_refine(self, n_steps: int = 5):
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
                if not loss.requires_grad:
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.gate.parameters(), 5.0)
                self.gate_optimizer.step()
        self.gate.eval()

    @torch.no_grad()
    def update_one_batch(self, batch: EpochBatch):
        old = self.cache[batch.batch_id]
        if old.resp is not None:
            self._sub_cache(old)
        b = self._move_batch(batch)
        z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
        new_resp = self.local_step(b.h, z_fit)
        new_cache = self._cache_from_resp(z_fit, new_resp)
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

    def global_objective(self) -> torch.Tensor:
        data_term = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        kl_gate = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        for b_cpu in self.batches:
            b = self._move_batch(b_cpu)
            resp = self.cache[b.batch_id].resp
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            data_term = data_term + (resp * (self.conditional_expected_log_pi(b.h) + self.component_expected_log_likelihood(z_fit))).sum()
            if self.K > 1:
                qh = self.gate.forward(b.h)
                a_h, b_h = qh['a'], qh['b']
                a_g = self.qv_a1.view(1, -1).expand_as(a_h).detach()
                b_g = self.qv_a0.view(1, -1).expand_as(b_h).detach()
                kl_gate = kl_gate + kumar2beta_kl(a_h, b_h, a_g, b_g, n_approx=10).sum()
        if self.K > 1:
            kl_sticks = beta_kl(
                self.qv_a1, self.qv_a0,
                torch.ones_like(self.qv_a1),
                torch.full_like(self.qv_a0, self.dp_alpha)
            ).sum()
        else:
            kl_sticks = torch.tensor(0.0, device=self.device_, dtype=self.dtype)

        m0 = self._prior_m0()
        beta0 = self._prior_beta0()
        kl_phi = torch.tensor(0.0, device=self.device_, dtype=self.dtype)
        for c in self.comp:
            kl_phi = kl_phi + normal_gamma_kl(c.mean, c.kappa, c.alpha, c.beta, m0, self.prior_kappa0, self.prior_alpha0, beta0).sum()
        return data_term + self.entropy - kl_sticks - kl_phi - self.gate_kl_weight * kl_gate

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

    #  birth init
    def _collect_birth_subset(self, target_k: int):
        hs, zs = [], []
        for bid, b_cpu in enumerate(self.batches):
            rk = self.cache[bid].resp[:, target_k]
            keep = rk > self.birth_resp_threshold
            if keep.any():
                b = self._move_batch(b_cpu)
                z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
                hs.append(b.h[keep])
                zs.append(z_fit[keep])
        if len(zs) == 0:
            h_empty = torch.empty((0,) + self.h_shape, device=self.device_, dtype=self.dtype)
            z_empty = torch.empty((0, self.C, self.H, self.W), device=self.device_, dtype=self.dtype)
            return h_empty, z_empty
        h_sub = torch.cat(hs, dim=0)
        z_sub = torch.cat(zs, dim=0)
        if z_sub.shape[0] > self.birth_subset_max:
            perm = torch.randperm(z_sub.shape[0], device=z_sub.device)[:self.birth_subset_max]
            h_sub, z_sub = h_sub.index_select(0, perm), z_sub.index_select(0, perm)
        return h_sub, z_sub

    def _split_bucket_principal_axis(self, parent_mean: torch.Tensor, z_bucket: torch.Tensor):
        if z_bucket.shape[0] < 4:
            return [z_bucket]
        residuals = z_bucket - parent_mean.unsqueeze(0)
        flat = residuals.flatten(1)
        if flat.shape[0] < 2:
            return [z_bucket]
        # One-step power iteration is enough here as a cheap direction estimate
        v = torch.randn(flat.shape[1], device=flat.device, dtype=flat.dtype)
        for _ in range(2):
            v = flat.t().matmul(flat.matmul(v))
            v = v / v.norm().clamp_min(torch.finfo(v.dtype).eps)
        proj = flat.matmul(v)
        median = proj.median()
        left = z_bucket[proj <= median]
        right = z_bucket[proj > median]
        buckets = []
        if left.shape[0] >= 2:
            buckets.append(left)
        if right.shape[0] >= 2:
            buckets.append(right)
        return buckets if len(buckets) > 0 else [z_bucket]

    def _init_fresh_components_from_split(self, target_k: int, z_sub: torch.Tensor, Kfresh: int) -> List[TensorDiagComponentPosterior]:
        """
        Divisive split along dominant residual-variance directions.
        """
        parent = self.comp[target_k]
        buckets = [z_sub]
        while len(buckets) < Kfresh:
            # split the bucket with largest residual variance
            scores = []
            for bucket in buckets:
                residual = (bucket - parent.mean.unsqueeze(0)).flatten(1)
                scores.append(float(residual.var(dim=0, unbiased=False).mean().item()))
            idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            bucket = buckets.pop(idx)
            children = self._split_bucket_principal_axis(parent.mean, bucket)
            if len(children) == 1:
                buckets.append(bucket)
                break
            buckets.extend(children)

        comps = []
        for bucket in buckets[:Kfresh]:
            if bucket.shape[0] < 2:
                continue
            N = bucket.shape[0]
            mean_emp = bucket.mean(dim=0)
            var_emp = bucket.var(dim=0, unbiased=False).clamp_min(1e-3)
            Nk = torch.tensor(float(N), device=self.device_, dtype=self.dtype)
            kappa = torch.as_tensor(self.prior_kappa0 + Nk, device=self.device_, dtype=self.dtype)
            alpha = torch.as_tensor(self.prior_alpha0 + 0.5 * Nk, device=self.device_, dtype=self.dtype)
            beta = (self._prior_beta0() + 0.5 * var_emp * Nk).clamp_min(torch.finfo(var_emp.dtype).eps)
            comps.append(TensorDiagComponentPosterior(mean=mean_emp, kappa=kappa, alpha=alpha, beta=beta))
        return comps

    #  structural edits
    @torch.no_grad()
    def _rebuild_all_caches_from_current_model(self):
        self._reset_global_summaries()
        for b_cpu in self.batches:
            b = self._move_batch(b_cpu)
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            c = self._cache_from_resp(z_fit, self.local_step(b.h, z_fit))
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
        self.gate.init_new_rows(old_K, self.K, scale=1e-2)
        with torch.no_grad():
            self._update_global_sticks_from_counts(
                torch.zeros(self.K, device=self.device_, dtype=self.dtype)
            )
        self.gate.set_active_K(self.K)

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
        self.K -= 1
        self.gate.set_active_K(self.K)
        self._reset_global_summaries()
        for bid, b_cpu in enumerate(self.batches):
            b = self._move_batch(b_cpu)
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            c = self._cache_from_resp(z_fit, self.cache[bid].resp)
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
        self.K -= 1
        self.gate.set_active_K(self.K)
        self._reset_global_summaries()
        for bid, b_cpu in enumerate(self.batches):
            b = self._move_batch(b_cpu)
            z_fit = b.posterior_mean if b.posterior_mean is not None else b.z
            c = self._cache_from_resp(z_fit, self.cache[bid].resp)
            self.cache[bid] = c
            self._add_cache(c)
        self._update_components_from_summaries()

    # speculative proposals
    def propose_birth(self, Kfresh: Optional[int] = None, adopt_laps: int = 1, gate_refine_steps_accept: int = 8, gate_refine_steps_speculate: int = 4) -> bool:
        if self.K >= self.max_components:
            return False
        ckpt = self.save_checkpoint()
        old_obj = float(self.global_objective().item())

        target_k = self.choose_birth_target()
        _, z_sub = self._collect_birth_subset(target_k)
        Kfresh = int(Kfresh or self.birth_kfresh)
        if z_sub.shape[0] < max(4, Kfresh):
            return False

        fresh = self._init_fresh_components_from_split(target_k, z_sub, Kfresh)
        if len(fresh) == 0:
            return False

        self._append_components(fresh)
        self._rebuild_all_caches_from_current_model()

        for _ in range(adopt_laps):
            self.memoized_lap_gate_frozen()
        if gate_refine_steps_speculate > 0 and self.K > 1:
            self.gate_refine(gate_refine_steps_speculate)
            self._rebuild_all_caches_from_current_model()
        new_obj = float(self.global_objective().item())
        if new_obj > old_obj:
            self.gate_refine(gate_refine_steps_accept)
            self._rebuild_all_caches_from_current_model()
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

    def propose_merge(self, kA: int, kB: int, refine_laps: int = 1, gate_refine_steps_accept: int = 8) -> bool:
        ckpt = self.save_checkpoint()
        old_obj = float(self.global_objective().item())
        self._merge_structure(kA, kB)
        for _ in range(refine_laps):
            self.memoized_lap_gate_frozen()
        new_obj = float(self.global_objective().item())
        if new_obj > old_obj:
            self.gate_refine(gate_refine_steps_accept)
            self._rebuild_all_caches_from_current_model()
            return True
        self.restore_checkpoint(ckpt)
        return False


    def try_best_merge(self, refine_laps: int = 1) -> bool:
        cur_obj = float(self.global_objective().item())
        pairs = self.candidate_merge_pairs()
        if len(pairs) == 0:
            return False

        best_gain, best_pair = -float('inf'), None
        for i, j in pairs[: self.merge_screen_topk]:
            ckpt = self.save_checkpoint()
            self._merge_structure(i, j)
            for _ in range(refine_laps):
                self.memoized_lap_gate_frozen()
            gain = float(self.global_objective().item()) - cur_obj
            self.restore_checkpoint(ckpt)
            if gain > best_gain:
                best_gain, best_pair = gain, (i, j)

        return best_pair is not None and best_gain > 0.0 and self.propose_merge(best_pair[0], best_pair[1], refine_laps)

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


    def propose_delete(self, kdel: int, refine_laps: int = 2, gate_refine_steps_accept: int = 8) -> bool:
        if self.K <= 1:
            return False
        ckpt = self.save_checkpoint()
        old_obj = float(self.global_objective().item())
        self._delete_structure(kdel)
        for _ in range(refine_laps):
            self.memoized_lap_gate_frozen()
        new_obj = float(self.global_objective().item())
        if new_obj > old_obj:
            self.gate_refine(gate_refine_steps_accept)
            self._rebuild_all_caches_from_current_model()
            return True
        self.restore_checkpoint(ckpt)
        return False


    def try_best_delete(self, refine_laps: int = 2) -> bool:
        cur_obj = float(self.global_objective().item())
        best_gain, best_k = -float('inf'), None
        for k in self.candidate_delete_components():
            ckpt = self.save_checkpoint()
            self._delete_structure(k)
            for _ in range(refine_laps):
                self.memoized_lap_gate_frozen()
            gain = float(self.global_objective().item()) - cur_obj
            self.restore_checkpoint(ckpt)
            if gain > best_gain:
                best_gain, best_k = gain, k
        return best_k is not None and best_gain > 0.0 and self.propose_delete(best_k, refine_laps)

    #  fit epoch / snapshot
    def fit_epoch(
        self,
        buffer: ReplayBufferConditional,
        batch_size: int,
        init_K: Optional[int] = None,
        n_laps: int = 8,
        do_birth: bool = True,
        do_merge: bool = True,
        do_delete: bool = True,
        birth_every: int = 2,
        merge_every: int = 1,
        delete_every: int = 1,
        shuffle_batches: bool = True,
        warm_start: bool = True,
    ) -> FrozenConditionalTopPrior:

        batches = buffer.freeze_epoch(batch_size=batch_size, shuffle=shuffle_batches)

        self.initialize_from_data(batches,K_init=init_K or self.K,warm_start=warm_start)

        for lap in range(1, n_laps + 1):
            self.memoized_lap(gate_refine_steps_after_lap=5)

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
        var = torch.stack([c.beta / (c.alpha - 1.0).clamp_min(1e-3) for c in self.comp], dim=0)
        return FrozenConditionalTopPrior(
            comp_mean=mean.detach().clone(),
            comp_var=var.detach().clone(),
            qv_a1=self.qv_a1.detach().clone(),
            qv_a0=self.qv_a0.detach().clone(),
            gate_state={k: v.detach().clone() for k, v in self.gate.state_dict().items()},
            K=self.K,
            max_K=self.max_components,
            objective=float(self.global_objective().item()),
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
    def conditional_expected_log_pi_frozen(snapshot: FrozenConditionalTopPrior, frozen_gate: ConditionalKumaraswamyGate, h_t: torch.Tensor) -> torch.Tensor:
        if snapshot.K == 1:
            return torch.zeros(h_t.shape[0], 1, device=h_t.device, dtype=h_t.dtype)
        qv_a1 = snapshot.qv_a1.to(device=h_t.device, dtype=h_t.dtype)
        qv_a0 = snapshot.qv_a0.to(device=h_t.device, dtype=h_t.dtype)
        e_log_v = torch.digamma(qv_a1) - torch.digamma(qv_a1 + qv_a0)
        e_log_1mv = torch.digamma(qv_a0) - torch.digamma(qv_a1 + qv_a0)
        elog_global = torch.empty(snapshot.K, device=h_t.device, dtype=h_t.dtype)
        prefix = torch.tensor(0.0, device=h_t.device, dtype=h_t.dtype)
        for k in range(snapshot.K):
            if k < snapshot.K - 1:
                elog_global[k] = e_log_v[k] + prefix
                prefix = prefix + e_log_1mv[k]
            else:
                elog_global[k] = prefix
        return elog_global.view(1, snapshot.K) + frozen_gate.expected_log_pi(h_t)


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
            snapshot, frozen_gate, h_t
        )  # [B,K]

        q_logvar = posterior_logvar.clamp(min=-20.0, max=10.0)
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

        log_p = stable_logsumexp(comp_log + elogpi.unsqueeze(0), dim=2)  # [S,B]

        return (log_q - log_p).mean(dim=0)  # [B]

    @staticmethod
    @torch.no_grad()
    def sample_from_frozen_prior(snapshot: FrozenConditionalTopPrior, frozen_gate: ConditionalKumaraswamyGate, h_t: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # Use the SAME conditional mixture weights as the KL path
        elogpi = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
            snapshot=snapshot,
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
    q_logvar = 2.0 * q_logsigma
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

    elogpi = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
        snapshot=snapshot,
        frozen_gate=frozen_gate,
        h_t=h_t,
    )  # [B, M]

    kl_q_to_comp = gaussian_kl_diag_to_components(
        q_mu=slot_mu,
        q_logsigma=slot_logsigma,
        comp_mean=comp_mean,
        comp_var=comp_var,
        eps=eps
    )  # [B, S, M]

    log_r = elogpi[:, None, :] - kl_q_to_comp
    resp = torch.softmax(log_r, dim=-1)

    slot_kl = resp * (
        torch.log(resp.clamp_min(eps))
        - elogpi[:, None, :]
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
        slot_z: [B, num_slots, D]
    """
    B = h_t.shape[0]
    M = snapshot.K

    elogpi = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
        snapshot=snapshot,
        frozen_gate=frozen_gate,
        h_t=h_t,
    )

    temp = max(float(temperature), 1e-4)
    pi = torch.softmax(elogpi / temp, dim=-1)

    pi_rep = pi[:, None, :].expand(B, num_slots, M).reshape(B * num_slots, M)
    idx = torch.multinomial(pi_rep, 1).squeeze(-1)

    comp_mean = snapshot.comp_mean.to(h_t.device, h_t.dtype).view(M, -1)
    comp_var = snapshot.comp_var.to(h_t.device, h_t.dtype).view(M, -1).clamp_min(1e-6)

    mu = comp_mean.index_select(0, idx)
    var = comp_var.index_select(0, idx)

    slot_z = mu + temp * torch.sqrt(var) * torch.randn_like(mu)

    return slot_z.view(B, num_slots, -1)
