"""Persistence gate conditioned on the previous stochastic latent.

The carry-conditioned gate uses a deterministic feature vector.  A latent gate
instead consumes ``z_{t-1} ~ N(mu, diag(s2))``.  Keeping that feature linear
preserves the closed-form Polya-Gamma update because both required feature
moments are available exactly.

``RecurrentStickiness`` accepts one ``phi`` tensor throughout its public API.
To remain a drop-in replacement, this subclass transports both moments as
``cat([E[phi], diag(Cov[phi])], -1)`` and overrides only the operations that use
the feature's second moment.
"""
from __future__ import annotations

import torch

from .recurrent_stick import RecurrentStickiness


class LatentStickiness(RecurrentStickiness):

    def __init__(self, K: int, feat_dim: int, latent_dim: int, **kwargs):
        # The parent appends its own bias slot.  Callers therefore pass
        # feat_dim=L (+ optional action width), while packed tensors contain
        # two copies of the resulting phi width.
        super().__init__(K=K, feat_dim=feat_dim, **kwargs)
        self.latent_dim = int(latent_dim)
        self.phi_dim = int(feat_dim) + 1

    @staticmethod
    def pack(phibar, variance):
        return torch.cat([phibar, variance], dim=-1)

    def _unpack(self, phi):
        """Accept packed moments or a point feature with zero variance."""
        width = self.phi_dim
        if phi.shape[-1] == 2 * width:
            return phi[..., :width], phi[..., width:].clamp_min(0.0)
        if phi.shape[-1] == width:
            return phi, torch.zeros_like(phi)
        raise ValueError(
            f"LatentStickiness expected phi width {width} or {2 * width}; "
            f"got {phi.shape[-1]}")

    @staticmethod
    def build_phi_moments(z_mean, z_var, action=None):
        """Return moments of ``phi=[z_{t-1}, a_{t-1}, 1]``."""
        parts = [z_mean]
        var_parts = [z_var if z_var is not None else torch.zeros_like(z_mean)]
        if action is not None:
            parts.append(action.to(z_mean.dtype))
            var_parts.append(torch.zeros_like(action, dtype=z_mean.dtype))
        ones = z_mean[..., :1] * 0.0 + 1.0
        parts.append(ones)
        var_parts.append(torch.zeros_like(ones))
        return torch.cat(parts, -1), torch.cat(var_parts, -1)

    def _psi_moments(self, phi):
        """Compute gate-logit moments with random beta and random phi."""
        phibar, phi_var = self._unpack(phi)
        mean_beta = self.m_beta.to(phibar.dtype)
        cov_beta = self.Sigma_beta.to(phibar.dtype)
        second_beta = cov_beta + torch.einsum(
            "kd,ke->kde", mean_beta, mean_beta)
        mean = torch.einsum("...d,kd->...k", phibar, mean_beta)
        quad = torch.einsum(
            "...d,kde,...e->...k", phibar, second_beta, phibar)
        trace = torch.einsum("kdd,...d->...k", second_beta, phi_var)
        return mean, (quad + trace - mean * mean).clamp_min(0.0)

    def pg_stats_from_batch(self, phi, r_mass, row_weight):
        """Closed-form PG natural parameters integrated over q(z)."""
        dtype = self.m_beta.dtype
        phibar, phi_var = self._unpack(phi)
        phibar, phi_var = phibar.to(dtype), phi_var.to(dtype)
        r_mass = r_mass.to(dtype)
        row_weight = row_weight.to(dtype).clamp_min(0.0)
        mean, variance = self._psi_moments(phi.to(dtype))
        c = (mean * mean + variance).clamp_min(1e-12).sqrt()
        expected_omega = (0.5 / c) * torch.tanh(0.5 * c) * row_weight
        precision = torch.einsum(
            "nk,nd,ne->kde", expected_omega, phibar, phibar)
        precision = precision + torch.diag_embed(torch.einsum(
            "nk,nd->kd", expected_omega, phi_var))
        natural_mean = torch.einsum(
            "nk,nd->kd", r_mass - 0.5 * row_weight, phibar)
        return dict(A=precision, h=natural_mean)

    @torch.no_grad()
    def pg_update_statewise(self, phi, r_mass, row_weight, lr=None):
        stats = self.pg_stats_from_batch(phi, r_mass, row_weight)
        if lr is None:
            self.pg_set_totals(stats["A"], stats["h"])
        else:
            rate = float(lr)
            self.pg_set_totals(
                (1 - rate) * self.pg_A + rate * stats["A"],
                (1 - rate) * self.pg_h + rate * stats["h"],
            )

    def hessian_blocks(self, gamma, phi, row_weight):
        """Return the latent gate's positive-semidefinite precision blocks."""
        mean, variance = self._psi_moments(phi)
        c = (mean * mean + variance).clamp_min(1e-12).sqrt()
        expected_omega = (
            (0.5 / c) * torch.tanh(0.5 * c) * row_weight.clamp_min(0.0)
        )
        latent_dim = self.latent_dim
        second_beta = (
            self.Sigma_beta[:, :latent_dim, :latent_dim]
            + torch.einsum(
                "kd,ke->kde",
                self.m_beta[:, :latent_dim],
                self.m_beta[:, :latent_dim],
            )
        )
        blocks = torch.einsum(
            "nk,nk,kde->nde",
            gamma.to(second_beta.dtype),
            expected_omega.to(second_beta.dtype),
            second_beta,
        )
        return 0.5 * (blocks + blocks.transpose(-1, -2))

    def _as_latent(self, other):
        """Preserve this subtype across adaptive-K resize/merge operations."""
        new = LatentStickiness(
            K=int(other.K),
            feat_dim=self.D,
            latent_dim=self.latent_dim,
            prior_persist=self.prior_persist,
            weight_prior_var=self.weight_prior_var,
            bias_prior_var=self.bias_prior_var,
            pg_iters=self.pg_iters,
            uncertainty_correction=self.uncorr,
            device=self.m0.device,
            dtype=self._dtype,
        )
        new.load_state_dict(other.state_dict())
        return new

    def resized_like(self, new_K: int):
        return self._as_latent(super().resized_like(int(new_K)))

    def select_rows(self, keep_idx):
        return self._as_latent(super().select_rows(keep_idx))

    @torch.no_grad()
    def merge_rows(self, i, j):
        """Merge the two rows' additive PG statistics and drop row ``j``."""
        i, j = int(i), int(j)
        if i == j:
            raise ValueError("merge_rows requires i != j")
        merged = self.select_rows(list(range(self.K)))
        merged.pg_A[i] = merged.pg_A[i] + merged.pg_A[j]
        merged.pg_h[i] = merged.pg_h[i] + merged.pg_h[j]
        prior_precision = torch.diag(1.0 / merged.sigma0_diag)
        covariance = torch.linalg.inv(prior_precision + merged.pg_A[i])
        merged.Sigma_beta[i] = covariance
        merged.m_beta[i] = covariance @ (
            prior_precision @ merged.m0 + merged.pg_h[i])
        keep = [k for k in range(merged.K) if k != j]
        return merged.select_rows(keep)
